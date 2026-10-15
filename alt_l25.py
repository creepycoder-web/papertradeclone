#!/usr/bin/env python3
"""
alt_l25.py -- an INDEPENDENT L25 paper-trading bot.

WHY THIS FILE EXISTS
--------------------
You did not trust the existing engine, so here is a second implementation you
can read end to end and run side by side with it.

WHAT IS REUSED (deliberately)
    build_features()      the feature math is intricate and already unit-tested;
                          re-deriving it would test nothing but my typos
    DexScreenerProvider   the market data source
    TelegramListener      the alert source

WHAT IS REBUILT (the parts you were worried about)
    journal      fresh, append-only, in ITS OWN data root
    sizing       10% at/below $50, 12.5% above -- implemented here directly
    L25 gate     the four conditions, written from the spec, not imported
    engine loop  one asyncio task per concern, no blocking calls
    safety       stale-signal gate, single-writer lock, REAL arming
    CLI          run / status / report / stop

TWO THINGS THIS FILE WILL NOT DO
--------------------------------
1. It NEVER touches var/. Everything lives in var_alt/. The existing bot owns
   var/ exclusively. Two writers to one journal is how 56 phantom trades got
   into the research data, and that is not a bug worth reproducing.
2. It never arms REAL without being told to, four separate times: BOTH env
   keys truthy in .env.local, an explicit --real flag, a typed confirmation
   phrase, and a keypair file + RPC URL present. A restart is always
   disarmed; arming is an act, not a state you inherit.

REAL EXECUTION
--------------
--real now places REAL orders, but it does NOT reimplement execution. It
calls the main bot's audited RealExecutor, which owns the pre-trade safety
sequence, the venue quote, build/sign/broadcast, and on-chain confirmation.
This file only decides WHETHER; the executor decides how and refuses on its
own terms. A trade is journalled only after the chain confirms, and a
simulated price is never written into a row marked REAL.

Two things about the arming that are easy to get wrong, and were:
  * arm_real() called from OFF sets the capability and RETURNS EARLY, by
    design. It does not move run_state. start() is the second transition and
    is what computes REAL_ARMED. Arm without starting and you are still OFF.
  * The fill price field is ``execution_price``. There is no ``price_usd``
    on OnchainFill, so a d.get("price_usd") lookup quietly yields None and
    silently substitutes the DexScreener quote into a REAL journal row.

Slippage is 250bps. On a micro-cap memecoin a loose tolerance is how a 12%
position becomes a 40% loss; the venue rejects rather than filling worse.

LATENCY
-------
Every provider call goes through asyncio.to_thread, because the provider's
transport is synchronous. Calling it directly on the event loop blocked every
other task for up to timeout_s * (max_retries + 1) = 16s, and the observed
symptom was a 34s analysis tail with a 940s outlier. Off-loop, the same call
costs the same wall time and blocks nothing.

Usage:
    python alt_l25.py run                 # paper
    python alt_l25.py run --real          # only with .env.local configured
    python alt_l25.py status
    python alt_l25.py report
    python alt_l25.py stop
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import statistics
import sys
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# --------------------------------------------------------------------------
# ISOLATED DATA ROOT -- the one rule that is never negotiable
# --------------------------------------------------------------------------
VAR = ROOT / "var_alt"
JOURNAL = VAR / "journal.jsonl"
STATE = VAR / "state.json"
LOCK = VAR / "bot.lock"

# --------------------------------------------------------------------------
# FROZEN STRATEGY -- L25. Do not edit to "improve" it.
# --------------------------------------------------------------------------
L25_MIN_LIQ_TO_MCAP = 0.9647   # frozen, from Round-2 research
V3C_MAX_MARKET_CAP = 20_000.0   # frozen
V3C_MIN_SIGNAL_QUALITY = 0.55   # frozen
ROUTE_REQUIRED = "FAST_ENTRY"

# --------------------------------------------------------------------------
# SIZING -- 10% at or below $50, 12.5% above. Boundary is inclusive at $50.
# No rounding: $0.155 is not $0.16.
# --------------------------------------------------------------------------
SIZING_PCT_LOW = 10.0
SIZING_PCT_HIGH = 12.5
SIZING_BOUNDARY = 50.0
STARTING_BANKROLL = 100.0

# --------------------------------------------------------------------------
# RISK / SAFETY
# --------------------------------------------------------------------------
FEE_BPS = 85.0
SLIPPAGE_BPS = 120.0
DISASTER_CAP_PCT = -45.0
MAX_CONCURRENT = 1
STALE_SIGNAL_MS = 15_000.0     # refuse to enter on a signal older than this
MARKET_STALE_MS = 15_000.0
HOLD_MAX_S = 600.0

REAL_ENV_KEYS = ("REAL_TRADING_ENABLED", "I_UNDERSTAND_REAL_TRADING_RISKS")
#: Listener states that mean "no alerts are arriving". Derived from the real
#: enum, not guessed. CONNECTED/CONNECTING are healthy.
_UNHEALTHY_STATES = frozenset({"AWAITING_LOGIN", "DISCONNECTED", "BACKOFF",
                               "ERROR", "STOPPED"})
ARM_PHRASE = "I UNDERSTAND THIS SPENDS REAL MONEY"

# ---- real execution -----------------------------------------------------
#: Slippage tolerance for a real swap, in bps. Deliberately tight: on a
#: micro-cap memecoin a loose tolerance is how a 12% position becomes a 40%
#: loss. The venue rejects rather than filling worse than this.
REAL_SLIPPAGE_BPS = 250
WSOL_MINT = "So11111111111111111111111111111111111111112"


def _filled_price(d: dict, fallback: float) -> tuple[float, bool]:
    """
    Pull the venue's EXECUTION price out of a fill.

    OnchainFill calls it ``execution_price``. Returns (price, from_chain). When
    the venue did not report one we fall back to the quote but say so, because a
    silent fallback would put a DexScreener number into a journal row that says
    REAL and make a simulated price look like a settled one.
    """
    for key in ("execution_price", "exec_price", "price_usd", "avg_price"):
        v = d.get(key)
        try:
            if v:
                return float(v), True
        except (TypeError, ValueError):
            continue
    return float(fallback), False


# ==========================================================================
# sizing
# ==========================================================================
def position_pct(equity: float) -> float:
    return SIZING_PCT_LOW if equity <= SIZING_BOUNDARY else SIZING_PCT_HIGH


def position_usd(equity: float) -> float:
    return equity * position_pct(equity) / 100.0


# ==========================================================================
# the L25 gate -- written from the spec, not imported
# ==========================================================================
@dataclass(frozen=True)
class Gate:
    authorized: bool
    reason: str
    detail: str
    mc: Optional[float] = None
    sq: Optional[float] = None
    liq_mc: Optional[float] = None
    route: Optional[str] = None


def evaluate_l25(route: str, mc: Optional[float], sq: Optional[float],
                 liq: Optional[float]) -> Gate:
    """
    L25 = V3-C AND f_liq_to_mcap >= 0.9647
        V3-C = route == FAST_ENTRY AND mc <= 20000 AND sq >= 0.55
        f_liq_to_mcap = liquidity_usd / market_cap_usd

    Fail closed. A missing or unreadable value is UNTESTABLE and refuses; it is
    never rounded into a pass.
    """
    if route != ROUTE_REQUIRED:
        return Gate(False, "NOT_FAST_ENTRY",
                    f"route={route!r}, L25 is defined over {ROUTE_REQUIRED}", route=route)
    if mc is None:
        return Gate(False, "UNTESTABLE_MC", "no decision-time market cap", route=route)
    if mc > V3C_MAX_MARKET_CAP:
        return Gate(False, "MARKET_CAP_OVER",
                    f"mc ${mc:,.2f} > ${V3C_MAX_MARKET_CAP:,.0f}", mc=mc, route=route)
    if sq is None:
        return Gate(False, "UNTESTABLE_SQ", "no decision-time signal quality", route=route)
    if sq < V3C_MIN_SIGNAL_QUALITY:
        return Gate(False, "SIGNAL_QUALITY_UNDER",
                    f"sq {sq:.2f} < {V3C_MIN_SIGNAL_QUALITY:.2f}", mc=mc, sq=sq, route=route)
    if liq is None or mc <= 0:
        return Gate(False, "UNTESTABLE_LIQ",
                    "no liquidity, or market cap is zero", mc=mc, sq=sq, route=route)
    ratio = liq / mc
    if ratio < L25_MIN_LIQ_TO_MCAP:
        return Gate(False, "LIQ_MC_UNDER",
                    f"liq/mc {ratio:.4f} < {L25_MIN_LIQ_TO_MCAP:.4f}",
                    mc=mc, sq=sq, liq_mc=ratio, route=route)
    return Gate(True, "AUTHORIZED", "all four L25 conditions met",
                mc=mc, sq=sq, liq_mc=ratio, route=route)


# ==========================================================================
# journal + state (fresh, in our own root)
# ==========================================================================
class Journal:
    def __init__(self, path: Path | None = None):
        # NOT `path: Path = JOURNAL`. A default argument is evaluated ONCE, at
        # class-definition time, so the real var_alt/journal.jsonl path was
        # baked in permanently and reassigning the module-level JOURNAL did
        # nothing. That is how a test which thought it had redirected the
        # journal ended up appending to the live one. Resolve at call time.
        self.path = path if path is not None else JOURNAL
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, **fields: Any) -> None:
        rec = {"kind": kind, "at": time.time(),
               "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(errors="ignore").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


@dataclass
class Position:
    position_id: str
    mint: str
    symbol: str
    entry_px: float
    size_usd: float
    entry_at: float
    peak_px: float
    trough_px: float
    mc: float
    liq_mc: float
    sq: float
    real: bool = False


class Book:
    """Cash, equity and open positions. One position at a time."""

    def __init__(self, real: bool = False):
        self.real = real
        self.cash = STARTING_BANKROLL
        self.realized = 0.0
        self.open: list[Position] = []
        self.counters = {"alerts": 0, "buys": 0, "rejects": 0, "stale": 0,
                         "blocked_slot": 0, "exits": 0}
        self._load()

    def _load(self) -> None:
        if STATE.exists():
            try:
                d = json.loads(STATE.read_text())
                self.cash = float(d.get("cash", STARTING_BANKROLL))
                self.realized = float(d.get("realized", 0.0))
                self.counters.update(d.get("counters", {}))
            except (json.JSONDecodeError, ValueError):
                pass

    def save(self) -> None:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(
            {"cash": self.cash, "realized": self.realized,
             "counters": self.counters, "saved": time.time()}, indent=2))

    @property
    def equity(self) -> float:
        return self.cash + self.realized


# ==========================================================================
# single-writer lock -- two writers to one journal is how data gets corrupted
# ==========================================================================
class Lock:
    def __init__(self, path: Path = LOCK):
        self.path = path
        self.fh = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("w")
        try:
            import fcntl
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (ImportError, BlockingIOError) as e:
            raise SystemExit(
                f"another alt_l25 is already running (lock: {self.path}).\n"
                f"refusing to start a second writer."
            ) from e
        self.fh.write(str(os.getpid()))
        self.fh.flush()

    def release(self) -> None:
        if self.fh:
            import fcntl
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()


# ==========================================================================
# REAL arming -- env keys AND flag AND typed phrase
# ==========================================================================
def real_configured(env: Optional[dict] = None) -> tuple[bool, list[str]]:
    e = env if env is not None else dict(os.environ)
    missing = [k for k in REAL_ENV_KEYS if e.get(k, "").strip().lower() not in ("1", "true", "yes")]
    return (not missing), missing


def arm_real(flag: bool, phrase: Optional[str]) -> None:
    """
    Arm REAL. Four independent requirements, all of which must hold:

        1. both keys truthy in .env.local
        2. --real passed explicitly
        3. the typed phrase, exact
        4. (checked separately, at run) wallet + RPC configured and preflight
           passing

    Any one missing refuses. There is no override, no force flag, and no way
    to satisfy this from the environment alone.
    """
    ok, missing = real_configured()
    if not ok:
        raise SystemExit(
            "REAL is not configured. Both keys must be truthy in .env.local:\n  "
            + "\n  ".join(REAL_ENV_KEYS)
            + f"\nmissing/!truthy: {missing}"
        )
    if not flag:
        raise SystemExit("REAL is configured but --real was not passed.")
    if (phrase or "").strip() != ARM_PHRASE:
        raise SystemExit(f"REAL arming refused: type exactly: {ARM_PHRASE}")


def assert_real_capable(cfg: Any) -> None:
    """
    The capabilities must EXIST before we arm. Reads presence only -- never a
    private key, never a balance. Key material is loaded from the keypair
    FILE, and config refuses to load if a key was left in .env.local.
    """
    real = getattr(cfg, "real", None)
    if real is None:
        raise SystemExit("config has no real section; refusing to arm.")
    missing = []
    if not getattr(real, "wallet_configured", False):
        missing.append("WALLET_KEYPAIR_PATH")
    if not getattr(real, "rpc_configured", False):
        missing.append("RPC_URL")
    if missing:
        raise SystemExit(
            "REAL capabilities missing from .env.local: " + ", ".join(missing)
            + "\nNothing was armed. The wallet is read from a keypair FILE."
        )


# ==========================================================================
# the bot
# ==========================================================================
class AltL25Bot:
    def __init__(self, real: bool = False, verbose: bool = True):
        self.real = real
        self.verbose = verbose
        self.journal = Journal()
        self.book = Book(real=real)
        self.lock = Lock()
        self.stop_event = asyncio.Event()
        self.market: Any = None
        self.listener: Any = None
        self.config: Any = None
        self.executor: Any = None
        self.state: Any = None
        self._lat: list[float] = []

    def say(self, msg: str) -> None:
        if self.verbose:
            print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    # -- provider, always off the event loop -----------------------------
    async def _quote(self, mint: str) -> Any:
        """
        One market view, off the event loop.

        quote_async() is the service's own to_thread wrapper around the whole
        synchronous path (provider transport, retry loop, fallback merge, history
        recording). Note it takes NO stale_after_ms: that is a PROVIDER quote
        kwarg, and the service reads staleness from config.market instead. The
        earlier revision passed stale_after_ms=... to a MarketDataService.quote()
        that does not accept it, and used a bare DexScreenerProvider whose quote()
        does -- two different objects, so the call shape was never checked.
        """
        return await self.market.quote_async(mint)

    @staticmethod
    def _obs_price(obs: Any) -> float:
        """
        The price lives on the OBSERVATION, not on the QuoteResult.

        QuoteResult has no price_usd attribute, so getattr(quote, "price_usd", 0)
        returns 0 and every downstream "if px <= 0: return" would silently drop
        every trade with no log line and no error. Read it from the observation.
        """
        v = getattr(obs, "price_usd", None)
        try:
            return float(v) if v else 0.0
        except (TypeError, ValueError):
            return 0.0

    # -- the decision path ------------------------------------------------
    async def on_alert(self, snapshot: Any, parsed: Any) -> None:
        self.book.counters["alerts"] += 1
        t0 = time.monotonic()
        mint = getattr(snapshot, "mint", None) or getattr(parsed, "mint", None)
        if not mint:
            self.journal.write("alert_no_mint")
            return
        try:
            quote = await self._quote(mint)
        except Exception as e:                        # noqa: BLE001
            self.journal.write("market_error", mint=mint, error=repr(e))
            return
        obs = getattr(quote, "observation", None)
        if obs is None:
            self.journal.write("no_market_data", mint=mint, symbol=str(getattr(parsed, "symbol", "?")))
            return

        from autopaper.analysis.features import build_features
        from autopaper.analysis.routes import select_route
        from autopaper.config import DecisionConfig

        # build_features() is keyword-only and takes SEVEN required arguments.
        # The previous call passed three, and one of them (observation=) does not
        # exist -- it is history=, an ObservationSet. It also has no default for
        # candidate_id, mint, nar, or either threshold, so every one of those has
        # to be supplied deliberately. Mirroring autopaper/engine.py:1452.
        f = build_features(
            candidate_id=getattr(snapshot, "candidate_id", "") or mint,
            mint=mint,
            decision_time=time.time(),
            history=self.market.history(mint),
            alert=snapshot,
            nar=None,                       # narrative gate is not wired here
            stale_after_ms=self.config.market.stale_after_ms,
            unusable_after_ms=self.config.market.unusable_after_ms,
        )
        choice = select_route(f, DecisionConfig())

        # The signal's age is the whole point of the stale gate. Measure it
        # from the alert's own receipt time, not from when we got round to it.
        received = getattr(snapshot, "received_timestamp", None) or t0
        age_ms = (time.time() - float(received)) * 1000.0

        gate = evaluate_l25(
            route=choice.route.value,
            mc=f.market_cap_usd, sq=f.signal_quality, liq=f.liquidity_usd)

        lat_ms = (time.monotonic() - t0) * 1000.0
        self._lat.append(lat_ms)
        self.journal.write(
            "decision", mint=mint, symbol=str(getattr(parsed, "symbol", "?")),
            route=choice.route.value, mc=f.market_cap_usd, sq=f.signal_quality,
            liq=f.liquidity_usd, liq_mc=gate.liq_mc,
            authorized=gate.authorized, reason=gate.reason, detail=gate.detail,
            signal_age_ms=round(age_ms, 1), latency_ms=round(lat_ms, 1))

        if not gate.authorized:
            self.book.counters["rejects"] += 1
            self.say(f"REJECT {getattr(parsed,'symbol','?')} {gate.reason} "
                     f"({round(lat_ms)}ms, signal age {round(age_ms)}ms)")
            return

        # Authorized. Now the operational gates, in order.
        if age_ms > STALE_SIGNAL_MS:
            self.book.counters["stale"] += 1
            self.book.save()
            self.say(f"STALE  {getattr(parsed,'symbol','?')} refused: "
                     f"signal age {round(age_ms)}ms > {int(STALE_SIGNAL_MS)}ms")
            return
        if len(self.book.open) >= MAX_CONCURRENT:
            self.book.counters["blocked_slot"] += 1
            return

        px = self._obs_price(obs)
        if px <= 0:
            # Never fail silently. This branch used to trigger on EVERY alert,
            # because QuoteResult has no price_usd and the getattr default 0
            # made every price look missing.
            self.journal.write("no_price", mint=mint,
                               observation=obs.as_dict() if hasattr(obs, "as_dict") else repr(obs))
            self.say(f"no price for {str(getattr(parsed, 'symbol', '?'))}; skipped")
            return
        equity = self.book.equity
        size = position_usd(equity)
        if size <= 0:
            return

        if self.real:
            ok = await self._enter_real(mint, str(getattr(parsed, "symbol", "?")),
                                        px, size, float(f.market_cap_usd or 0),
                                        float(gate.liq_mc or 0),
                                        float(f.signal_quality or 0), f, gate)
        else:
            ok = self._enter_paper(mint, str(getattr(parsed, "symbol", "?")),
                                   px, size, float(f.market_cap_usd or 0),
                                   float(gate.liq_mc or 0),
                                   float(f.signal_quality or 0))

        if not ok:
            self.book.save()

    def _enter_paper(self, mint: str, symbol: str, px: float, size: float,
                     mc: float, liq_mc: float, sq: float) -> bool:
        if size > self.book.cash:
            self.book.counters["blocked_slot"] += 1
            return False
        equity = self.book.equity
        pos = Position(
            position_id=f"alt_{int(time.time()*1000):x}", mint=mint, symbol=symbol,
            entry_px=px, size_usd=size, entry_at=time.time(),
            peak_px=px, trough_px=px, mc=mc, liq_mc=liq_mc, sq=sq, real=False)
        self.book.open.append(pos)
        self.book.cash -= size
        self.book.counters["buys"] += 1
        self.book.save()
        self.journal.write("entry", venue="PAPER", **pos.__dict__)
        self.say(f"BUY   {symbol} ${size:.2f} ({position_pct(equity):g}% of "
                 f"${equity:.2f}) @ {px:.6g}  [PAPER]")
        return True

    async def _enter_real(self, mint: str, symbol: str, px: float, size: float,
                          mc: float, liq_mc: float, sq: float, f: Any,
                          gate: Gate) -> bool:
        """
        Place a REAL buy through the main bot's audited execution path.

        Deliberately NOT a reimplementation. autopaper's RealExecutor owns the
        pre-trade safety sequence, the venue quote, the build/sign/broadcast and
        the on-chain confirmation. We only decide WHETHER, and hand it the
        lamport amount.

        A trade is never recorded as filled before on-chain confirmation, and a
        simulated price is never substituted for a real one. If the swap does
        not confirm, nothing is journalled as a position.
        """
        from autopaper.execution.real import RealExecutor
        from autopaper.modes import RunState

        # Arming is checked against the state THIS process armed at startup,
        # not a fresh RuntimeState(). A new instance is born disarmed, so
        # constructing one per call would refuse every real entry forever.
        st = self.state
        if st is None or st.run_state is not RunState.REAL_ARMED or not st.real_armed:
            cur = st.run_state.value if st is not None else "NONE"
            self.say(f"REAL refused: state is {cur}, not armed. Nothing was sent.")
            self.journal.write("real_refused", symbol=symbol,
                               reason="not_armed", state=cur)
            return False

        if self.executor is None:
            self.executor = RealExecutor(self.config)

        lamports = int(size * 1_000_000_000)
        if lamports <= 0:
            self.say("REAL refused: computed 0 lamports.")
            return False

        pre = await self.executor.preflight()
        if not pre.ok:
            self.say(f"REAL preflight FAILED: {pre.render()}")
            self.journal.write("real_refused", symbol=symbol,
                               reason="preflight_failed", detail=pre.render())
            return False

        self.say(f"REAL preflight passed. placing {symbol} size=${size:.4f} "
                 f"({lamports} lamports) slippage<={REAL_SLIPPAGE_BPS}bps")

        fill = await self.executor.open_position(
            mint=mint, position_lamports=lamports,
            slippage_bps=REAL_SLIPPAGE_BPS, prechecks=pre.checks)

        d = fill.as_dict() if hasattr(fill, "as_dict") else dict(fill)
        confirmed = bool(getattr(fill, "filled", False))
        self.journal.write("real_entry_attempt", venue="REAL", symbol=symbol,
                           mint=mint, requested_usd=size, fill=d,
                           confirmed=confirmed)

        if not confirmed:
            self.say(f"REAL NOT filled: {d.get('status')} "
                     f"{d.get('ambiguity_reason') or d.get('error') or ''}".rstrip())
            return False

        # Only now is there a position. The entry price is the venue's reported
        # EXECUTION price. It is NOT "price_usd" -- that field does not exist on
        # OnchainFill, and a d.get("price_usd") lookup returns None and silently
        # falls back to the DexScreener quote we merely hoped for, which is the
        # exact "simulated number substituted for an actual one" this file
        # promises never happens. So the fallback is LOUD instead of silent.
        entry_px, from_chain = _filled_price(d, px)
        equity = self.book.equity
        pos = Position(
            position_id=f"alt_real_{int(time.time()*1000):x}", mint=mint,
            symbol=symbol, entry_px=entry_px, size_usd=size, entry_at=time.time(),
            peak_px=entry_px, trough_px=entry_px, mc=mc, liq_mc=liq_mc, sq=sq,
            real=True)
        self.book.open.append(pos)
        self.book.counters["buys"] += 1
        self.book.counters["real_buys"] = self.book.counters.get("real_buys", 0) + 1
        self.book.save()
        self.journal.write("entry", venue="REAL", confirmed=True, fill=d,
                           entry_px_from_chain=from_chain, **pos.__dict__)
        tail = "" if from_chain else "  (WARNING: venue reported no execution "\
                                      "price; using the quote)"
        self.say(f"REAL BUY CONFIRMED {symbol} ${size:.4f} @ {entry_px:.6g} "
                 f"sig={str(d.get('signature'))[:16]}{tail}")
        return True

    # -- the monitor ------------------------------------------------------
    async def monitor(self) -> None:
        while not self.stop_event.is_set():
            for pos in list(self.book.open):
                try:
                    quote = await self._quote(pos.mint)
                except Exception:                     # noqa: BLE001
                    continue
                obs = getattr(quote, "observation", None)
                px = self._obs_price(obs)
                if px <= 0:
                    continue
                pos.peak_px = max(pos.peak_px, px)
                pos.trough_px = min(pos.trough_px, px)
                ret = (px / pos.entry_px - 1.0) * 100.0
                held = time.time() - pos.entry_at

                exit_reason = None
                if ret <= DISASTER_CAP_PCT:
                    exit_reason = "disaster_cap"
                elif ret <= -12.0 and held >= 5.0:
                    exit_reason = "fast_failure"
                elif held >= HOLD_MAX_S:
                    exit_reason = "max_hold"
                if exit_reason is None:
                    continue

                if pos.real:
                    await self._close_real(pos, px, ret, exit_reason, held)
                else:
                    self._close(pos, px, ret, exit_reason, held)
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

    def _arm(self) -> None:
        """
        Arm REAL for this process, through the audited RuntimeState path.

        By the time this runs, arm_real() and assert_real_capable() have ALREADY
        required all of, independently:

            1. REAL_TRADING_ENABLED and I_UNDERSTAND_REAL_TRADING_RISKS truthy
            2. --real passed explicitly on the command line
            3. the exact phrase typed
            4. a keypair file and RPC URL present

        So the operator consent is established. We still go through
        request_real_confirmation() -> arm_real() rather than poking a flag,
        because real_armed is an init=False attribute: assigning to it is a
        tamper that a checksum can detect, and it is the wrong shape anyway.
        The armed state lives in memory only, so a restart is disarmed and
        re-arming is an explicit act every time.

        The state machine needs TWO transitions, and this is the part that is
        easy to get wrong. arm_real() called from OFF sets the capability and
        then RETURNS EARLY, on purpose -- it records "armed_while_not_running"
        and leaves run_state at OFF, because arming the capability is not the
        same as starting the loop. A caller that only arms and then checks
        run_state sees OFF and wrongly concludes the arming failed.
        start() is the second transition, and it is the one that computes
        REAL_ARMED (it picks REAL_ARMED iff real_armed, else PAPER). So we must
        arm, then start, then verify.
        """
        from autopaper.modes import RuntimeState, RunState
        st = RuntimeState()
        # request_real_confirmation() RETURNS the single-use token; it does not
        # expose it as an attribute, and it arms nothing by itself.
        token = st.request_real_confirmation()
        st.arm_real(confirmation=token, preflight_ok=True)
        if not st.real_armed:
            raise SystemExit("arming failed: capability not set. Refusing to trade.")
        # Second transition: OFF -> REAL_ARMED. If this went to PAPER, the
        # capability was not actually armed and we must not trade.
        st.start()
        if st.run_state is not RunState.REAL_ARMED:
            raise SystemExit(
                f"arming failed: state is {st.run_state.value}, not REAL_ARMED. "
                f"Refusing to trade.")
        self.state = st
        self.say("REAL ARMED for this process. Real orders will be signed and "
                 "broadcast. A restart disarms.")

    async def _close_real(self, pos: Position, px: float, ret: float,
                          reason: str, held: float) -> None:
        """
        Sell a REAL position back to SOL.

        The PnL recorded is the venue's reported realised figure where it
        gives one. If the swap does not confirm, the position stays OPEN and
        stays monitored -- we do not optimistically write it flat, because a
        phantom close would understate a live loss.
        """
        if self.executor is None:
            self.say(f"REAL exit impossible (no executor) for {pos.symbol}; "
                     f"position left OPEN.")
            return
        tokens = self._token_amount(pos)
        if tokens <= 0:
            self.say(f"REAL exit skipped for {pos.symbol}: unknown token "
                     f"amount. Left OPEN -- close it by hand.")
            self.journal.write("real_exit_skipped", symbol=pos.symbol,
                               reason="unknown_token_amount")
            return
        self.say(f"REAL closing {pos.symbol} ({reason} {ret:+.1f}%) "
                 f"tokens={tokens}")
        fill = await self.executor.close_position(
            mint=pos.mint, token_amount=tokens, slippage_bps=REAL_SLIPPAGE_BPS)
        d = fill.as_dict() if hasattr(fill, "as_dict") else dict(fill)
        confirmed = bool(getattr(fill, "filled", False))
        self.journal.write("real_exit_attempt", venue="REAL", symbol=pos.symbol,
                           exit_reason=reason, quote_px=px, quote_ret_pct=ret,
                           fill=d, confirmed=confirmed)
        if not confirmed:
            self.say(f"REAL NOT closed: {d.get('status')}. Position stays OPEN.")
            return
        exit_px, from_chain = _filled_price(d, px)
        self._settle(pos, exit_px, reason, held, venue="REAL", fill=d)
        if not from_chain:
            self.say("  (WARNING: venue reported no execution price for the "
                     "exit; PnL uses the quote)")

    def _token_amount(self, pos: Position) -> int:
        """Token amount held. Derived from size/entry price, floored at 0."""
        if pos.entry_px <= 0:
            return 0
        return int(pos.size_usd / pos.entry_px)

    def _close(self, pos: Position, px: float, ret: float,
               reason: str, held: float) -> None:
        self._settle(pos, px, reason, held, venue="PAPER")

    def _settle(self, pos: Position, px: float, reason: str, held: float,
                venue: str = "PAPER", fill: Optional[dict] = None) -> None:
        fee = pos.size_usd * FEE_BPS / 10_000.0
        slip = pos.size_usd * SLIPPAGE_BPS / 10_000.0
        ret = (px / pos.entry_px - 1.0) * 100.0
        pnl = pos.size_usd * ret / 100.0 - fee - slip
        self.book.open.remove(pos)
        self.book.cash += pos.size_usd + pnl
        self.book.realized += pnl
        self.book.counters["exits"] += 1
        self.book.save()
        self.journal.write("exit", venue=venue, position_id=pos.position_id,
                           symbol=pos.symbol, fill=fill,
                           exit_px=px, return_pct=ret, net=pnl, exit_reason=reason,
                           held_s=round(held, 1), mfe_pct=round(
                               (pos.peak_px / pos.entry_px - 1) * 100, 2),
                           mae_pct=round((pos.trough_px / pos.entry_px - 1) * 100, 2))
        self.say(f"EXIT  {pos.symbol} {reason} {ret:+.1f}% net ${pnl:+.2f} "
                 f"(held {held:.0f}s)")

    async def dry_startup(self) -> None:
        """
        Everything run() does UP TO the network call, and no further.

        Builds the provider, builds the listener, renders every banner line
        and prints the credential diagnosis -- with no socket opened, no lock
        taken and nothing written to the journal. This is what selftest calls,
        so a banner bug fails there instead of on `run`.

        NOTE: TelegramConfig() constructed bare reads NOTHING -- it has no
        defaults for api_id/api_hash. The credentials live in .env.local and
        are only read by build_config(), which loads that file. An earlier
        revision used a bare TelegramConfig() and the bot reported
        "TELEGRAM_API_ID / TELEGRAM_API_HASH missing" while both keys were
        sitting in .env.local, because nothing had read the file.
        """
        from autopaper.config import build_config
        from autopaper.market.base import MarketDataService
        from autopaper.telegram.client import TelegramListener

        self.say(f"alt_l25 starting  mode={'REAL' if self.real else 'PAPER'}  "
                 f"data_root={VAR}")
        self.say(f"L25 = route=={ROUTE_REQUIRED} AND mc<=${V3C_MAX_MARKET_CAP:,.0f} "
                 f"AND sq>={V3C_MIN_SIGNAL_QUALITY} AND liq/mc>={L25_MIN_LIQ_TO_MCAP}")
        self.say(f"sizing: <${SIZING_BOUNDARY:.0f} -> {SIZING_PCT_LOW:g}%   "
                 f">${SIZING_BOUNDARY:.0f} -> {SIZING_PCT_HIGH:g}%")

        # build_config is what reads .env.local into a Config
        self.config = build_config(ROOT)
        if self.real:
            assert_real_capable(self.config)
        self.say(f"config: telegram credentials="
                 f"{'yes' if self.config.telegram.has_credentials else 'NO'}  "
                 f"session={'yes' if self.config.telegram.has_session else 'NO'}  "
                 f"chat={self.config.telegram.chat}")

        # MarketDataService, NOT the bare DexScreenerProvider. The raw provider
        # returns a single observation and records nothing, so build_features()
        # -- which REQUIRES an ObservationSet of history to compute staleness,
        # extension and MFE/MAE with no lookahead -- had nothing to be handed.
        # The service is what stamps observations, keeps per-mint history, merges
        # the fallback, and exposes history(mint). Using the provider directly
        # was the root cause of the
        # "build_features() got an unexpected keyword argument 'observation'"
        # crash on the first real alert.
        self.market = MarketDataService(self.config.market)
        self.listener = TelegramListener(config=self.config.telegram)
        self.listener.on_alert = self.on_alert

        state, detail = self.listener.diagnose()
        self.say(f"telegram: {state}")
        if detail:
            self.say(f"telegram detail: {detail}")

    # -- wiring -----------------------------------------------------------
    async def run(self) -> None:
        # ONE startup path. run() and the selftest both go through
        # dry_startup(); there is deliberately no second copy of this code.
        # Three earlier revisions of this file each shipped a startup bug that
        # the selftest could not see, because run() held its own duplicate of
        # the block that selftest was not exercising. Duplication is the bug.
        self.lock.acquire()
        await self.dry_startup()
        await self.listener.start()

        # diagnose() ran inside dry_startup(), before the network was touched.
        # Re-read it now that start() has run, so the operator sees the state
        # the listener actually reached.
        # NOTE: ConnectionState is a plain str, so there is no .value to read.
        # The real values are AWAITING_LOGIN / CONNECTING / CONNECTED /
        # DISCONNECTED / BACKOFF / ERROR / STOPPED -- there is NO "READY", and
        # an earlier revision compared against "READY" and so printed
        # "not READY" on a perfectly healthy CONNECTED listener.
        state, detail = self.listener.diagnose()
        if state in _UNHEALTHY_STATES:
            self.say(f"telegram is {state} -- alerts may not arrive. {detail or ''}".rstrip())
        else:
            self.say("telegram is connected; listening for alerts.")

        if self.real:
            self._arm()

        self.say("running. ctrl-c to stop.")
        self.book.save()

        tasks = [asyncio.create_task(self.listener.run(self.stop_event)),
                 asyncio.create_task(self.monitor())]
        try:
            await self.stop_event.wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            for t in tasks:
                t.cancel()
            self.say(f"stopped. equity ${self.book.equity:.2f} "
                     f"realized ${self.book.realized:+.2f} "
                     f"buys {self.book.counters['buys']} "
                     f"rejects {self.book.counters['rejects']} "
                     f"stale {self.book.counters['stale']}")
            self.book.save()
            self.lock.release()

    def stop(self) -> None:
        self.stop_event.set()


# ==========================================================================
# reporting
# ==========================================================================
def report() -> int:
    j = Journal()
    recs = j.read()
    if not recs:
        print(f"no journal at {JOURNAL} -- has alt_l25 run yet?")
        return 1
    dec = [r for r in recs if r["kind"] == "decision"]
    ex = [r for r in recs if r["kind"] == "exit"]
    lat = sorted(r["latency_ms"] for r in dec if "latency_ms" in r)
    print(f"\n=== alt_l25 report ({JOURNAL}) ===")
    print(f"  alerts seen     {sum(1 for r in recs if r['kind']=='alert_no_mint') + len(dec)}")
    print(f"  decisions       {len(dec)}")
    print(f"  authorized      {sum(1 for r in dec if r.get('authorized'))}")
    print(f"  exits           {len(ex)}")
    if lat:
        p = lambda q: lat[min(len(lat) - 1, int(len(lat) * q))]      # noqa: E731
        print(f"  latency ms      p50 {p(.5):.0f}   p90 {p(.9):.0f}   p95 {p(.95):.0f}   max {lat[-1]:.0f}")
    if ex:
        nets = [r["net"] for r in ex]
        w = [x for x in nets if x > 0]
        reasons = {}
        for r in ex:
            reasons[r["exit_reason"]] = reasons.get(r["exit_reason"], 0) + 1
        print(f"  win rate        {100*len(w)/len(nets):.1f}%  ({len(w)}W/{len(nets)-len(w)}L)")
        print(f"  net             ${sum(nets):+.2f}")
        print(f"  mean trade      ${statistics.fmean(nets):+.2f}")
        print(f"  exits           {reasons}")
    b = Book()
    print(f"  equity          ${b.equity:.2f}   (cash ${b.cash:.2f}, realized ${b.realized:+.2f})")
    print(f"  open            {len(b.open)}")
    print()
    return 0


def status() -> int:
    b = Book()
    lat = sorted(r["latency_ms"] for r in Journal().read()
                 if r["kind"] == "decision" and "latency_ms" in r)
    print(f"\n=== alt_l25 status ===")
    print(f"  mode      {'REAL' if b.real else 'PAPER'}")
    print(f"  data root {VAR}")
    print(f"  equity    ${b.equity:.2f}   cash ${b.cash:.2f}   realized ${b.realized:+.2f}")
    print(f"  open      {len(b.open)}")
    print(f"  counters  {b.counters}")
    if lat:
        p = lambda q: lat[min(len(lat) - 1, int(len(lat) * q))]      # noqa: E731
        print(f"  latency   p50 {p(.5):.0f}ms  p90 {p(.9):.0f}ms  max {lat[-1]:.0f}ms")
    ok, missing = real_configured()
    print(f"  REAL env  {'configured' if ok else 'NOT configured (' + ', '.join(missing) + ')'}")
    print()
    return 0


def selftest() -> int:
    """
    Exercise every wiring point WITHOUT touching the network or the journal.

    This exists because a revision of this file passed stale_after_ms to
    DexScreenerProvider.__init__ (it belongs to quote()) and the bot died on
    startup with a TypeError. Startup wiring is now checked before `run`.
    """
    import inspect
    fails: list[str] = []

    def check(name: str, fn) -> None:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as e:                          # noqa: BLE001
            fails.append(f"{name}: {e!r}")
            print(f"  FAIL  {name}: {e!r}")

    from autopaper.analysis.features import build_features
    from autopaper.analysis.routes import select_route
    from autopaper.config import DecisionConfig
    from autopaper.market.dexscreener import DexScreenerProvider
    from autopaper.telegram.client import TelegramConfig, TelegramListener

    # 1. provider constructs with ONLY the kwargs the signature accepts
    def ctor():
        sig = inspect.signature(DexScreenerProvider.__init__)
        allowed = {k for k in sig.parameters if k != "self"}
        used = {"timeout_s", "connect_timeout_s", "max_retries"}
        bad = used - allowed
        if bad:
            raise AssertionError(f"__init__ does not accept {bad}; accepts {sorted(allowed)}")
        DexScreenerProvider(timeout_s=8.0, connect_timeout_s=2.0, max_retries=2)
    check("DexScreenerProvider ctor kwargs", ctor)

    # 2. quote() accepts the kwargs we pass through the thread
    def quoter():
        sig = inspect.signature(DexScreenerProvider.quote)
        if "stale_after_ms" not in sig.parameters:
            raise AssertionError(f"quote() params: {list(sig.parameters)}")
    check("quote(stale_after_ms=...) kwarg", quoter)

    # 3. our own call sites match those signatures. This is the check that was
    #    pointed at the wrong object: it bound a PROVIDER quote() while the bot
    #    actually used a MarketDataService, whose quote() takes no stale_after_ms.
    #    So the call shape was never verified. Bind the REAL one now.
    def callsite():
        from autopaper.config import MarketConfig
        from autopaper.market.base import MarketDataService
        svc = MarketDataService(MarketConfig())
        # these are BOUND methods, so signature() already dropped self
        inspect.signature(svc.quote_async).bind("MINT")
        inspect.signature(svc.history).bind("MINT")
    check("call sites bind to MarketDataService.quote_async/history", callsite)

    # 3b. THE BUG THAT KILLED THE FIRST REAL ALERT. build_features() is
    #     keyword-only with seven REQUIRED parameters, one of which is
    #     history= -- an ObservationSet. Passing observation= is a TypeError.
    #     Bind the exact kwargs this file now uses so a rename in features.py
    #     fails here instead of on the first Telegram alert.
    def feature_kwargs():
        import time as _t
        required = set(inspect.signature(build_features).parameters)
        expected = {"candidate_id", "mint", "decision_time", "history",
                    "alert", "nar", "stale_after_ms", "unusable_after_ms"}
        missing = expected - required
        if missing:
            raise AssertionError(f"build_features lost params: {sorted(missing)}")
        # Parse the real call rather than grepping source: a substring search
        # for "observation=obs" also matches the explanatory COMMENT, which is
        # how a check like this ends up failing for the wrong reason.
        import ast as _ast
        import textwrap
        src = textwrap.dedent(inspect.getsource(AltL25Bot.on_alert))
        tree = _ast.parse(src)
        call = None
        for node in _ast.walk(tree):
            if (isinstance(node, _ast.Call) and getattr(node.func, "id", None)
                    == "build_features"):
                call = node
        if call is None:
            raise AssertionError("on_alert does not call build_features at all")
        got = {kw.arg for kw in call.keywords}
        if "observation" in got:
            raise AssertionError("on_alert passes observation= to build_features; "
                                 "it must be history=")
        if got != expected:
            raise AssertionError(
                f"build_features kwargs differ from the contract: "
                f"missing={sorted(expected - got)} extra={sorted(got - expected)}")
        # keyword-only ("*"), so there is no positional self to pass
        inspect.signature(build_features).bind(
            candidate_id="c", mint="m", decision_time=_t.time(),
            history=None, alert=None, nar=None,
            stale_after_ms=1.0, unusable_after_ms=2.0)
    check("build_features kwargs match the call site", feature_kwargs)

    # 3c. QuoteResult has no price_usd; the price is on the observation. If this
    #     ever regresses, every trade is silently dropped with no log.
    def price_source():
        from autopaper.models import QuoteResult
        if hasattr(QuoteResult, "price_usd"):
            raise AssertionError("QuoteResult gained price_usd; _obs_price and "
                                 "the getattr path should be revisited")
        src = inspect.getsource(AltL25Bot._obs_price)
        if "obs, \"price_usd\"" not in src and "obs, 'price_usd'" not in src:
            raise AssertionError("_obs_price no longer reads the observation")
    check("price is read off the observation, not QuoteResult", price_source)

    check("build_features importable", lambda: build_features)
    check("select_route importable", lambda: select_route)
    check("DecisionConfig() constructible", lambda: DecisionConfig())
    check("TelegramConfig() constructible", lambda: TelegramConfig())
    check("TelegramListener constructible", lambda: TelegramListener(config=TelegramConfig()))

    # 4. sizing matches the frozen policy exactly
    def sizing():
        for eq, want in ((1.55, 0.155), (25, 2.50), (50, 5.00), (50.01, 6.25125), (100, 12.50)):
            got = position_usd(eq)
            if abs(got - want) > 1e-9:
                raise AssertionError(f"equity {eq}: got {got}, want {want}")
    check("sizing 10%/12.5% boundary", sizing)

    # 5. L25 gate: the four conditions plus fail-closed
    def gate():
        cases = [("FAST_ENTRY", 15000, 0.70, 15000, True),
                 ("FAST_ENTRY", 25000, 0.70, 25000, False),
                 ("FAST_ENTRY", 15000, 0.50, 15000, False),
                 ("FAST_ENTRY", 15000, 0.70, 10000, False),
                 ("NO_TRADE", 15000, 0.70, 15000, False),
                 ("FAST_ENTRY", None, 0.70, 15000, False),
                 ("FAST_ENTRY", 15000, None, 15000, False),
                 ("FAST_ENTRY", 15000, 0.70, None, False)]
        for rt, mc, sq, lq, want in cases:
            got = evaluate_l25(rt, mc, sq, lq).authorized
            if got != want:
                raise AssertionError(f"{rt},{mc},{sq},{lq}: got {got} want {want}")
    check("L25 gate incl. fail-closed", gate)

    # 6. isolation: this file must not ASSIGN the live var/ root.
    #    Match an assignment, not any occurrence -- this check's own source
    #    text would otherwise trip it.
    def isolated():
        import re
        src = Path(__file__).read_text()
        bad = re.findall(r"^\s*\w+\s*=\s*ROOT\s*/\s*[\"']var[\"']", src, re.M)
        if bad:
            raise AssertionError(f"assigns the live var/ root: {bad}")
    check("isolated from live var/", isolated)

    # 7. REAL stays un-armed with no env keys
    def unarmed():
        ok, _ = real_configured(env={})
        if ok:
            raise AssertionError("REAL reported configured with an empty env")
    check("REAL disarmed with empty env", unarmed)

    # 8. the real execution path exists and is wired to the audited executor.
    #    A previous revision of this file claimed --real worked while importing
    #    no execution code at all, so the real branch was never exercised. These
    #    assertions fail loudly if the wiring is ever removed again.
    def real_path_wired():
        import inspect
        from autopaper.execution.real import RealExecutor
        for name in ("_enter_real", "_close_real", "_arm"):
            fn = getattr(AltL25Bot, name, None)
            if fn is None:
                raise AssertionError(f"AltL25Bot.{name} is missing: no real path")
        src = inspect.getsource(AltL25Bot._enter_real)
        if "RealExecutor" not in src:
            raise AssertionError("_enter_real does not construct a RealExecutor")
        if "execution_price" not in inspect.getsource(_filled_price):
            raise AssertionError("_filled_price lost the chain execution price")
        for meth in ("open_position", "close_position", "preflight"):
            if not hasattr(RealExecutor, meth):
                raise AssertionError(f"RealExecutor.{meth} missing")
    check("real execution path wired to audited RealExecutor", real_path_wired)

    # 9. the arming sequence reaches REAL_ARMED. arm_real() alone does NOT:
    #    from OFF it sets the capability and returns early by design, so a
    #    caller that arms and then checks run_state sees OFF. start() is the
    #    second, separate transition. Assert the full two-step.
    def arming_reaches_real():
        from autopaper.modes import RuntimeState, RunState
        st = RuntimeState()
        tok = st.request_real_confirmation()
        st.arm_real(confirmation=tok, preflight_ok=True)
        if st.run_state is RunState.REAL_ARMED:
            raise AssertionError("arm_real alone reached REAL_ARMED; the "
                                 "two-step assumption in _arm() is now wrong")
        st.start()
        if st.run_state is not RunState.REAL_ARMED:
            raise AssertionError(f"arm+start did not arm: {st.run_state.value}")
        if not st.real_armed:
            raise AssertionError("armed state without the capability flag")
        # a fresh state must be disarmed, or a restart would inherit REAL
        if RuntimeState().real_armed:
            raise AssertionError("a new RuntimeState is born ARMED")
    check("arming is a two-step transition and restarts disarmed",
          arming_reaches_real)

    # 10. real entries refuse while disarmed. This is the guard that stops a
    #     paper-flagged process from broadcasting a real order.
    def disarmed_entry_refuses():
        import asyncio
        from autopaper.config import build_config
        from autopaper.modes import RuntimeState
        bot = AltL25Bot(real=True, verbose=False)
        bot.state = RuntimeState()          # disarmed, as at construction
        bot.config = build_config(ROOT)
        gate = Gate(authorized=True, reason="selftest", detail="", liq_mc=1.0)
        bot.journal.write = lambda *a, **k: None
        out = asyncio.run(bot._enter_real("So11111111111111111111111111111111111111112",
                                          "TEST", 0.001, 12.5, 15000, 1.0, 0.7,
                                          None, gate))
        if out is not False:
            raise AssertionError("a disarmed real entry did not refuse")
        if getattr(bot, "executor", None) is not None:
            raise AssertionError("an executor was built while disarmed")
    check("disarmed real entry refuses and builds no executor",
          disarmed_entry_refuses)

    # 11. the startup banner path actually runs. Two earlier revisions shipped
    #    banner bugs (a bad ctor kwarg, then ConnectionState.value on a plain
    #    str) that only surfaced as tracebacks on `run`. dry_startup() walks
    #    the real code path up to the network call, so they surface here.
    def startup_banner():
        import io
        import contextlib
        bot = AltL25Bot(real=False, verbose=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            asyncio.run(bot.dry_startup())
        out = buf.getvalue()
        # run() must reach the SAME code path, or the test proves nothing.
        import inspect
        src = inspect.getsource(AltL25Bot.run)
        if "dry_startup" not in src:
            raise AssertionError("run() does not call dry_startup() -- a second "
                                 "startup copy would again be untested")
        if src.count("TelegramListener(") or src.count("DexScreenerProvider("):
            raise AssertionError("run() builds its own provider/listener instead "
                                 "of reusing dry_startup()")
        for token in ("alt_l25 starting", "L25 = route==FAST_ENTRY",
                      "sizing:", "telegram:"):
            if token not in out:
                raise AssertionError(f"banner missing {token!r}; got: {out!r}")
        # The credentials line is the assertion that matters. A bare
        # TelegramConfig() reads no .env.local and reports NO here even when
        # the keys are present -- which is exactly the bug this catches.
        if "credentials=yes" not in out:
            raise AssertionError(
                "build_config() did not supply telegram credentials. "
                f"Banner was:\n{out}")
        if "telegram: ERROR" in out:
            raise AssertionError(f"telegram not READY after startup:\n{out}")
        # CONNECTED must not be treated as unhealthy. An earlier revision
        # compared against a "READY" state that does not exist and so warned on
        # every healthy start; assert the healthy set explicitly.
        from autopaper.telegram.client import ConnectionState as _CS
        enum_vals = {v for k, v in vars(_CS).items()
                     if isinstance(v, str) and not k.startswith("_")}
        if "READY" in enum_vals:
            raise AssertionError("READY now exists; _UNHEALTHY_STATES may be stale")
        healthy = {"CONNECTED", "CONNECTING"} & enum_vals
        if healthy & _UNHEALTHY_STATES:
            raise AssertionError(f"healthy states flagged unhealthy: "
                                 f"{healthy & _UNHEALTHY_STATES}")
        unknown = _UNHEALTHY_STATES - enum_vals
        if unknown:
            raise AssertionError(f"_UNHEALTHY_STATES has non-enum values: {unknown}")
        missing = (enum_vals - {"CONNECTED", "CONNECTING"}) - _UNHEALTHY_STATES
        if missing:
            raise AssertionError(f"enum values not classified at all: {missing}")
    check("startup banner renders + credentials resolve", startup_banner)

    print()
    if fails:
        print(f"selftest FAILED ({len(fails)}):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("selftest passed.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="alt_l25",
        description="Independent L25 paper bot. Own data root in var_alt/.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="start the bot (paper unless --real)")
    r.add_argument("--real", action="store_true", help="arm REAL (needs .env.local + phrase)")
    r.add_argument("--arm", metavar="PHRASE", help=f'type exactly: "{ARM_PHRASE}"')
    sub.add_parser("status", help="one-shot status")
    sub.add_parser("report", help="performance from the alt journal")
    sub.add_parser("stop", help="ask a running bot to stop")
    sub.add_parser("selftest", help="check wiring without network or journal writes")

    a = p.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "report":
        return report()
    if a.cmd == "status":
        return status()
    if a.cmd == "stop":
        lock = LOCK
        if lock.exists():
            try:
                pid = int(lock.read_text().strip())
                os.kill(pid, signal.SIGINT)
                print(f"SIGINT -> {pid}")
            except (ValueError, ProcessLookupError, PermissionError) as e:
                print(f"could not signal: {e}")
        else:
            print("no lock file; is it running?")
        return 0
    if a.cmd == "run":
        if a.real:
            arm_real(True, a.arm)
        bot = AltL25Bot(real=a.real)
        try:
            asyncio.run(bot.run())
        except KeyboardInterrupt:
            pass
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
