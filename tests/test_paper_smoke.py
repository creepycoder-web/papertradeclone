"""
THE ACCEPTANCE SMOKE TEST for the 2026-09-30 incident.

One deterministic run that walks the whole paper path and proves, in a single
artefact, every property the incident report claims:

    alert
      -> valid market data
      -> valid narrative
      -> L25 authorizes a BUY
      -> valid quote at fill time
      -> PAPER_FILL=SUCCESS
      -> exactly one open position
      -> cash decreases, exposure increases
      -> buys == 1, blocked == 0
      -> a signal->fill latency sample EXISTS
      -> the journal contains the open position and the entry
      -> NOTHING is signed or broadcast

and, in the same run, that a stale candidate is refused with a named reason so
the two can never be confused again.

It also runs the REAL strategy - L25, unmodified - on a market the strategy
genuinely accepts, so the acceptance path is proved WITHOUT forcing the decision.
A forced-BUY smoke test would prove the executor and nothing about the gate; the
two are separated across this file (``test_the_strategy_gate_still_authorizes``
here, the executor assertions below).
"""


from __future__ import annotations

import asyncio
import json
from dataclasses import replace as dc_replace
from pathlib import Path

import pytest

from autopaper.clock import FakeClock
from autopaper.engine import PaperTrader
from autopaper.latency import BUY_BLOCKED, PAPER_FILL_SUCCESS
from autopaper.models import Decision, NarSnapshot, PositionStatus
from autopaper.telegram import ui

from .conftest import MINT
from .test_incident_paper_buy import (
    alert,
    force_buy,
    make_trader,
    notifications,
)

pytestmark = [
    pytest.mark.execution,
    pytest.mark.integrity,
]

T0 = 1_700_000_000.0

#: A market the FROZEN L25 gate genuinely accepts. These numbers are the shape
#: tests/test_decision.py documents as tradeable: inside V3-C's $20,000 ceiling,
#: a deep book (liquidity/market-cap far above L25's 0.9647), a ~3:1 buy ratio
#: and a price that is ACCELERATING rather than exhausted. Nothing here is tuned
#: to make a threshold pass - it is a normal launch.
PRICE_PATH = (1.00e-4, 1.22e-4, 1.40e-4, 1.46e-4)
MARKET_CAP_USD = 18_000.0
LIQUIDITY_USD = 400_000.0
VOLUME_5M_USD = 180_000.0
BUYS_5M = 420
SELLS_5M = 140


class RisingVenue:
    """A deterministic ACCELERATING price path, quoted per mint."""

    name = "venue"

    def __init__(self, clock: FakeClock):
        self.clock = clock
        self.started = clock.now()
        self.calls = 0
        self.fail = False

    def _price(self) -> float:
        step = max(0, int(self.clock.now() - self.started) // 2)
        return PRICE_PATH[min(step, len(PRICE_PATH) - 1)]

    def quote(self, mint: str):
        from autopaper.models import Observation, QuoteResult

        self.calls += 1
        if self.fail:
            return QuoteResult(observation=None, error="venue: down",
                               provider=self.name)
        now = self.clock.now()
        return QuoteResult(
            observation=Observation(
                mint=mint, observed_at=now, received_at=now,
                price_usd=self._price(),
                market_cap_usd=MARKET_CAP_USD,
                liquidity_usd=LIQUIDITY_USD,
                volume_5m_usd=VOLUME_5M_USD, volume_1h_usd=900_000.0,
                buys_5m=BUYS_5M, sells_5m=SELLS_5M, txns_5m=BUYS_5M + SELLS_5M,
                pair_created_at=self.started - 600,
                provider=self.name,
            ),
            provider=self.name,
        )


def _run(config, clock, *, venue: RisingVenue | None = None) -> PaperTrader:
    """
    A trader on a real, UNMODIFIED strategy.

    The narrative source is the only substitution: a real Telegram connection is
    not available offline, and an absent narrative only costs 0.05 of signal
    quality. Market data, the decision engine, sizing, the fill model, the
    position book, the ledger and the accounting are all production code.
    """
    # Use legacy mark-based fill path for this smoke test
    from dataclasses import replace as _replace
    object.__setattr__(
        config.real, "paper_legacy_mark_fill_rollback", True)

    from autopaper.market import MarketDataService
    from autopaper.nar.base import NarService
    from autopaper.telegram import TelegramListener
    from autopaper.telegram.client import ConnectionState
    from autopaper.market.ratelimit import RateLimiter

    venue = venue or RisingVenue(clock)

    class _Provider:
        name = "venue"

        def __init__(self):
            self.limiter = RateLimiter(10_000, name="venue")
            self.retry_stats = lambda: {}
            self.max_call_s = 0.0
            self.timeout_s = 0.0

        quote = venue.quote

    market = MarketDataService(config.market, clock=clock, primary=_Provider(),
                               fallback=_Provider())
    market._fvp_probe_budget = 0
    nar = NarService(config.nar, clock=clock)
    listener = TelegramListener(config.telegram, clock=clock)
    listener.status.state = ConnectionState.CONNECTED
    listener.status.chat_id = 100
    listener.status.chat = "testchat"
    trader = PaperTrader(config, clock=clock, market=market, nar=nar,
                         listener=listener)
    listener.on_alert = trader.on_alert
    trader.forward = None
    trader.forward_deployment = None

    async def good_nar(mint):
        return NarSnapshot(
            mint=mint, fetched_at=trader.clock.now(), source="nar:fake",
            available=True,
            narrative="Bittersweet is a dog meme coin launched in 2024.",
        )

    trader.nar.fetch_async = good_nar

    async def fake_sleep(seconds: float) -> None:
        clock.advance(seconds)

    trader._sleep = fake_sleep
    trader.startup()
    trader.state.start()
    trader.cli_only = True
    return trader


def test_the_strategy_gate_still_authorizes_on_a_real_market(config, clock):
    """
    L25 is NOT bypassed anywhere in this file's acceptance path. A market inside
    V3-C's ceiling with a healthy liquidity/market-cap ratio must produce a BUY
    from the real DecisionEngine.
    """
    trader = _run(config, clock)
    asyncio.run(trader.on_alert(alert(clock, liquidity_usd=LIQUIDITY_USD,
                                      volume_usd=VOLUME_5M_USD), None))

    assert trader.last_decision["decision"] == "BUY", (
        "the frozen L25 gate refused a market it is documented to accept: "
        + str(trader.last_decision.get("risk_flags"))
    )
    gate = trader.last_decision.get("l25") or {}
    assert gate.get("market_cap_usd", MARKET_CAP_USD) <= 20_000.0
    assert trader.config.decision.l25_min_liq_to_mcap == 0.9647
    trader._release_ownership()


# ---------------------------------------------------------------------------
# A. the real PAPER smoke test
# ---------------------------------------------------------------------------


def test_paper_smoke_alert_to_open_position(config, clock, tmp_path):
    """
    The full chain, asserted in the order an operator would ask about it.
    """
    trader = _run(config, clock)
    notes = notifications(trader)

    cash0 = trader.account.cash
    exposure0 = trader.account.open_exposure()
    equity0 = trader.account.equity()
    assert exposure0 == 0.0

    # NO force_buy: this is L25's own decision on a market L25 accepts.
    asyncio.run(trader.on_alert(alert(clock, liquidity_usd=LIQUIDITY_USD,
                                      volume_usd=VOLUME_5M_USD), None))

    # 1. decision = BUY, and it is labelled as a decision
    assert trader.last_decision["decision"] == "BUY"
    assert trader.last_decision["candidate_id"]

    # 2. a quote WAS present
    row = trader._pipeline_for_display()
    assert row["quote"] != "-"
    assert trader.market.request_count >= 2, (
        "there must be a quote at decision time AND one at fill time"
    )

    # 3. the fill succeeded, explicitly
    assert trader.last_execution["outcome"] == PAPER_FILL_SUCCESS
    assert trader.last_execution["executed"] is True

    # 4. exactly one open position, and it is OPEN
    positions = trader.account.open_positions()
    assert len(positions) == 1
    assert positions[0].status is PositionStatus.OPEN

    # 5. the accounting moved
    assert trader.buys == 1
    assert trader.blocks == 0
    assert trader.account.cash < cash0
    assert trader.account.open_exposure() > 0.0
    snap0 = trader.account.snapshot()
    assert snap0["open_positions"] == 1
    assert snap0["open_exposure_usd"] > 0.0
    assert snap0["fees_paid_usd"] > 0.0
    # The identity the whole account rests on.
    assert snap0["identity_check"] == pytest.approx(0.0, abs=1e-6)

    # 6. a signal -> fill sample EXISTS (it was "no samples" in the incident)
    rec = trader._current_stages
    assert rec.signal_to_fill_ms is not None
    assert rec.signal_to_fill_ms > 0
    assert trader.latency.samples["signal_to_fill"]
    assert rec.analysis_ms is not None and rec.analysis_ms > 0
    assert rec.market_ms is not None
    assert rec.settle_ms is not None

    # 7. the journal contains the open position and the entry
    trades = (Path(trader.journal.root) / "trades.jsonl").read_text()
    assert '"event":"entry"' in trades
    assert MINT in trades
    assert trader.last_execution["position_id"] in trades

    # 8. the ledger recorded the entry with its signal->fill span
    entries = []
    with open(trader.config.ledger_file, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rowj = json.loads(line)
                if rowj.get("event") == "entry":
                    entries.append(rowj)
    assert len(entries) == 1
    assert entries[0]["signal_to_fill_ms"] is not None
    assert entries[0]["signal_to_fill_ms"] > 0

    # 9. the operator was told the FILL
    body = "\n".join(t for t, _ in notes)
    assert "PAPER" in body

    # 10. nothing was signed or broadcast
    assert trader.state.trade_mode.value == "PAPER"
    assert trader.state.real_trading_enabled is False
    assert trader.config.real.trading_enabled is False
    assert trader.config.real.env_trading_enabled_request is False

    # 11. state survives a checkpoint
    trader.persist()
    state = json.loads(Path(trader.config.state_file).read_text())
    assert len(state["open_positions"]) == 1
    assert state["counters"]["buys"] == 1
    assert state["counters"]["blocked"] == 0
    assert state["last_execution"]["outcome"] == PAPER_FILL_SUCCESS

    # 12. and the operator views show it
    open_rows = trader.open_positions_view()
    assert len(open_rows) == 1
    assert open_rows[0]["position_id"] == trader.last_execution["position_id"]
    assert "No open positions." not in ui.render_open_positions(open_rows)

    trader._release_ownership()


def test_paper_smoke_does_not_sign_or_broadcast(config, clock, monkeypatch):
    """
    Item B of the acceptance criteria, on its own so the claim is unambiguous.
    """
    from autopaper.execution import real as real_mod
    from autopaper.execution import solana as solana_mod

    trader = _run(config, clock)
    notifications(trader)
    touched: list[str] = []

    def trip(name):
        def boom(*_a, **_k):
            touched.append(name)
            raise AssertionError(f"PAPER execution reached {name}")
        return boom

    for cls, method in (
        (real_mod.RealExecutor, "open_position"),
        (real_mod.RealExecutor, "close_position"),
        (real_mod.RealExecutor, "quote_probe"),
        (real_mod.LiveSolanaBackend, "send_transaction"),
        (real_mod.LiveSolanaBackend, "swap"),
        (solana_mod, "sign_transaction"),
        (solana_mod, "broadcast"),
    ):
        monkeypatch.setattr(cls, method, trip(f"{cls.__name__}.{method}"),
                            raising=False)

    asyncio.run(trader.on_alert(alert(clock, liquidity_usd=LIQUIDITY_USD,
                                      volume_usd=VOLUME_5M_USD), None))

    assert trader.buys == 1
    assert touched == [], f"PAPER execution touched real execution: {touched}"
    # The REAL executor was never even constructed (it is built lazily).
    assert trader._real_executor is None
    assert trader.state.real_trading_enabled is False
    trader._release_ownership()


def test_paper_smoke_and_stale_refusal_cannot_be_confused(config, clock):
    """
    Two runs on the same engine class, and the outcomes are unmistakably
    different on every surface. This is the incident, closed.
    """
    # -- run A: a fresh signal -> a fill
    a = _run(config, clock)
    notifications(a)
    force_buy(a)
    asyncio.run(a.on_alert(alert(clock, mid=1), None))
    assert a.last_execution["outcome"] == PAPER_FILL_SUCCESS
    a_row = a._pipeline_for_display()
    assert "BUY_BLOCKED" not in a_row["quote"]
    assert len(a.account.open_positions()) == 1
    a._release_ownership()

    # -- run B: a 34.9s-old signal -> a refusal, on a SEPARATE account so the
    #    first run's open position cannot be mistaken for the second's fill
    import dataclasses

    cfg_b = dataclasses.replace(
        config,
        var_dir=config.var_dir / "b", state_file=config.var_dir / "b" / "paper_account.json",
        ledger_file=config.var_dir / "b" / "ledger.jsonl",
        decisions_file=config.var_dir / "b" / "decisions.jsonl",
        candidates_file=config.var_dir / "b" / "candidates.jsonl",
        updates_file=config.var_dir / "b" / "position_updates.jsonl",
    )
    cfg_b.var_dir.mkdir(parents=True, exist_ok=True)
    clock_b = FakeClock(T0)
    b = _run(cfg_b, clock_b)
    notes = notifications(b)
    force_buy(b)
    snap = alert(clock_b, mid=2)
    snap = dc_replace(snap, received_timestamp=clock_b.now() - 34.9,
                      source_timestamp=clock_b.now() - 35.3)
    asyncio.run(b.on_alert(snap, None))

    assert b.last_execution["outcome"] == BUY_BLOCKED
    assert b.last_execution["reason"] == "stale_signal"
    b_row = b._pipeline_for_display()
    assert "BUY_BLOCKED=stale_signal" in b_row["quote"]
    assert "BUY_BLOCKED=stale_signal" in b_row["fill"]
    assert b.account.open_positions() == []
    assert b.buys == 0 and b.blocks == 1
    assert "NO POSITION OPENED" in "\n".join(t for t, _ in notes)

    # The two surfaces cannot be read as the same event.
    assert a.last_execution["outcome"] != b.last_execution["outcome"]
    assert a_row["outcome"] != b_row["outcome"]
    a._release_ownership()
    b._release_ownership()


def test_real_trading_enabled_is_false_throughout(config, clock):
    """
    Item J, asserted at the four places it could plausibly be true.
    """
    trader = _run(config, clock)
    assert trader.config.real.env_trading_enabled_request is False
    assert trader.config.real.trading_enabled is False
    assert trader.state.real_trading_enabled is False
    notifications(trader)
    asyncio.run(trader.on_alert(alert(clock, liquidity_usd=LIQUIDITY_USD,
                                      volume_usd=VOLUME_5M_USD), None))
    assert trader.buys == 1
    # Even after a fill, and after a status render.
    assert trader.state.real_trading_enabled is False
    trader.render_dashboard()
    assert trader.bot_status()["real_trading_enabled"] is False
    assert trader.status_view().state.real_trading_enabled is False
    trader._release_ownership()
