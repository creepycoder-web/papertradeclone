"""
The golden run: a fingerprint that proves later work did not change behaviour.

WHAT THIS IS FOR
----------------
Rounds 2b and 3 add ops, docs, cost and reporting work that must not touch a
single decision, fill or exit. "Must not" is cheap to say and expensive to
verify by reading diffs, so this drives one scripted deterministic session
through the production engine and hashes the outcome. If the hash moves and
nobody approved a behaviour change, the change was accidental.

WHY AN ALLOWLIST AND NOT A STRIPPER
-----------------------------------
Every id in this system is ``new_id`` -> ``uuid4().hex[:12]`` (models.py:142),
so a hash over whole records would be random. Rather than denylisting volatile
keys -- which silently admits any volatile key added later -- the fingerprint is
an EXPLICIT PROJECTION of the fields that encode behaviour: the decision, the
route and regime it was reached through, the money, the exit reason, and the
account identity. A field that matters and is not here is a gap in the
fingerprint, not a false alarm; adding one is a deliberate act with a new hash.

FakeClock times ARE kept. They are part of the behaviour: a fill at +1200ms is
a different fill from one at +0ms.

WHAT IS DELIBERATELY NOT HASHED
-------------------------------
Ids, filesystem paths, process ids and host names. None of them is behaviour,
and all of them vary per run or per machine.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from autopaper.clock import FakeClock
from autopaper.models import Observation, PositionStatus, QuoteResult

from .test_incident_paper_buy import alert, make_trader  # noqa: F401
from .test_paper_smoke import _run

pytestmark = [pytest.mark.integrity]

T0 = 1_700_000_000.0

#: The fingerprint. Update only for a behaviour change Veha explicitly
#: approved, with the old and new hash in the commit message.
GOLDEN_SHA256 = "c7763e338fe1643258ca124119e129cd19c9555a2f03dce5c573a6b3f6fc297f"

#: A market the FROZEN L25 gate accepts: inside V3-C's $20,000 ceiling, a deep
#: book, ~3:1 buys, accelerating. Copied in shape from test_paper_smoke.py, not
#: tuned to pass a threshold.
GOOD_MARKET = dict(
    market_cap_usd=18_000.0,
    liquidity_usd=400_000.0,
    volume_5m_usd=180_000.0,
    buys_5m=420,
    sells_5m=140,
)


class ScriptedVenue:
    """A venue whose price path, book and failure mode are all scripted."""

    name = "venue"

    def __init__(self, clock: FakeClock, prices, *, step_s: float = 2.0,
                 **market):
        self.clock = clock
        self.started = clock.now()
        self.prices = tuple(prices)
        self.step_s = step_s
        self.market = {**GOOD_MARKET, **market}
        self.fail = False
        self.calls = 0

    def _price(self) -> float:
        step = max(0, int((self.clock.now() - self.started) // self.step_s))
        return self.prices[min(step, len(self.prices) - 1)]

    def quote(self, mint: str) -> QuoteResult:
        self.calls += 1
        if self.fail:
            return QuoteResult(observation=None, error="venue: down",
                               provider=self.name)
        now = self.clock.now()
        return QuoteResult(
            observation=Observation(
                mint=mint, observed_at=now, received_at=now,
                price_usd=self._price(),
                volume_1h_usd=900_000.0,
                txns_5m=self.market["buys_5m"] + self.market["sells_5m"],
                pair_created_at=self.started - 600,
                provider=self.name,
                **self.market,
            ),
            provider=self.name,
        )


def _isolated(config, name: str, *, market=None):
    """
    A config with its own account, lock and ledger streams.

    ``market`` overrides the MarketConfig. Used by the vanished-market scenario
    to switch the PROVIDER's retry policy off: with retries on, each failing
    quote costs a real backoff sleep and that one scenario took 93s of the
    fingerprint's 106s. The scenario is about how the ENGINE values a position
    whose data stopped arriving, not about how many times the provider retries
    first, so the retry count is not part of what is being fingerprinted.
    """
    var = config.var_dir / name
    var.mkdir(parents=True, exist_ok=True)
    return dataclasses.replace(
        config,
        market=market or config.market,
        var_dir=var,
        state_file=var / "paper_account.json",
        ledger_file=var / "ledger.jsonl",
        decisions_file=var / "decisions.jsonl",
        candidates_file=var / "candidates.jsonl",
        updates_file=var / "position_updates.jsonl",
        control_file=var / "control.json",
    )


# ---------------------------------------------------------------------------
# the projection
# ---------------------------------------------------------------------------

#: Decision fields that encode behaviour.
_DECISION_KEYS = ("decision", "regime", "route", "data_state",
                  "signal_quality", "confidence")
#: Execution fields that encode behaviour.
_EXEC_KEYS = ("outcome", "executed", "reason")


def _decision_of(trader) -> dict:
    row = trader.last_decision or {}
    return {k: row.get(k) for k in _DECISION_KEYS}


def _execution_of(trader) -> dict:
    row = trader.last_execution or {}
    return {k: row.get(k) for k in _EXEC_KEYS}


def _positions_of(trader) -> list:
    """Open positions, by the numbers that define them."""
    out = []
    for p in sorted(trader.account.positions.values(),
                    key=lambda q: (q.mint, q.entry_timestamp or 0.0)):
        out.append({
            "status": getattr(p.status, "value", str(p.status)),
            "route": getattr(p.route, "value", str(p.route)),
            "regime": getattr(p.regime, "value", str(p.regime)),
            "exit_style": getattr(p.exit_style, "value", str(p.exit_style)),
            "entry_timestamp": p.entry_timestamp,
            "fill_price": p.fill_price,
            "quantity": p.quantity,
            "notional_usd": p.notional_usd,
            "entry_fee_usd": p.entry_fee_usd,
            "entry_slippage_usd": p.entry_slippage_usd,
            "cash_out_usd": p.cash_out_usd,
            "exit_timestamp": p.exit_timestamp,
            "exit_fill_price": p.exit_fill_price,
            "exit_fee_usd": p.exit_fee_usd,
            "exit_reason": p.exit_reason,
            "realized_pnl_usd": p.realized_pnl_usd,
            "return_pct": p.return_pct,
        })
    return out


def _account_of(trader) -> dict:
    snap = trader.account.snapshot()
    return {k: snap.get(k) for k in (
        "starting_bankroll_usd", "cash_usd", "realized_pnl_usd",
        "fees_paid_usd", "slippage_paid_usd", "open_positions",
        "unfilled_attempts", "identity_check",
    )}


def _counters_of(trader) -> dict:
    return {"buys": trader.buys, "blocks": trader.blocks,
            "rejects": trader.rejects, "waits": trader.waits}


def _snapshot(trader) -> dict:
    return {
        "decision": _decision_of(trader),
        "execution": _execution_of(trader),
        "positions": _positions_of(trader),
        "account": _account_of(trader),
        "counters": _counters_of(trader),
    }


def canonical(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str)


def fingerprint(payload) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# the scenarios
# ---------------------------------------------------------------------------


def _ticks(trader, clock, n: int, seconds: float = 15.0) -> None:
    """
    Advance the clock and run the monitor n times.

    Few, long ticks rather than many short ones. The exits this fingerprint
    exercises are time-based -- the 75s no-follow-through window and the 120s
    data-invalid timer -- so the sim time crossed is what matters, and each
    extra tick is real wall-clock cost in a test that runs on every commit.
    """
    for _ in range(n):
        clock.advance(seconds)
        asyncio.run(trader.monitor_once())


def scenario_runner(config) -> dict:
    """
    An alert L25 accepts, held while the price rises.

    IT DOES NOT EXIT, AND THAT IS THE FINDING. Under L25_CURRENT the exit style
    is FAST_FAILURE, whose take-profit comes from a per-position SNAPSHOT that
    is unset by default (paper_live/take_profit.py: for_position -> None), the
    profile declares no max hold (strategy_profile.py: PROFILE_MAX_HOLD_S
    L25_CURRENT -> None), and the live monitor passes momentum and structure as
    None, so momentum_collapse and structure_break cannot fire either. A
    position that is WINNING therefore has no price exit left except the -45%
    disaster cap, the -12% stop and the data-invalid timer.

    So this scenario fingerprints a profitable position sitting OPEN. If a
    later change gives it an exit, this hash must move and the commit must say
    which change did it.
    """
    clock = FakeClock(T0)
    venue = ScriptedVenue(clock, (1.00e-4, 1.22e-4, 1.40e-4, 1.46e-4,
                                  2.10e-4, 3.00e-4, 3.10e-4))
    trader = _run(_isolated(config, "runner"), clock, venue=venue)
    try:
        asyncio.run(trader.on_alert(
            alert(clock, liquidity_usd=GOOD_MARKET["liquidity_usd"],
                  volume_usd=GOOD_MARKET["volume_5m_usd"]), None))
        _ticks(trader, clock, 8)
        return _snapshot(trader)
    finally:
        trader._release_ownership()


def scenario_fast_failure(config) -> dict:
    """The same entry, but the price collapses into its stop."""
    clock = FakeClock(T0)
    venue = ScriptedVenue(clock, (1.00e-4, 1.22e-4, 1.40e-4, 1.46e-4,
                                  1.10e-4, 0.80e-4, 0.60e-4, 0.50e-4))
    trader = _run(_isolated(config, "faststop"), clock, venue=venue)
    try:
        asyncio.run(trader.on_alert(
            alert(clock, liquidity_usd=GOOD_MARKET["liquidity_usd"],
                  volume_usd=GOOD_MARKET["volume_5m_usd"]), None))
        _ticks(trader, clock, 8)
        return _snapshot(trader)
    finally:
        trader._release_ownership()


def scenario_gate_refuses(config) -> dict:
    """
    A market cap far above V3-C's $20,000 ceiling. The FROZEN gate refuses, and
    the refusal is part of the fingerprint: a change that starts taking these
    is a behaviour change.
    """
    clock = FakeClock(T0)
    venue = ScriptedVenue(clock, (1.00e-4, 1.22e-4, 1.40e-4),
                          market_cap_usd=5_000_000.0)
    trader = _run(_isolated(config, "refused"), clock, venue=venue)
    try:
        asyncio.run(trader.on_alert(
            alert(clock, market_cap_usd=5_000_000.0,
                  liquidity_usd=GOOD_MARKET["liquidity_usd"],
                  volume_usd=GOOD_MARKET["volume_5m_usd"]), None))
        return _snapshot(trader)
    finally:
        trader._release_ownership()


def scenario_stale_signal(config) -> dict:
    """An alert older than the stale-entry budget. latency.py must refuse it."""
    clock = FakeClock(T0)
    venue = ScriptedVenue(clock, (1.00e-4, 1.22e-4, 1.40e-4, 1.46e-4))
    trader = _run(_isolated(config, "stale"), clock, venue=venue)
    try:
        snap = alert(clock, liquidity_usd=GOOD_MARKET["liquidity_usd"],
                     volume_usd=GOOD_MARKET["volume_5m_usd"])
        snap = dataclasses.replace(snap,
                                   received_timestamp=clock.now() - 34.9,
                                   source_timestamp=clock.now() - 35.3)
        asyncio.run(trader.on_alert(snap, None))
        return _snapshot(trader)
    finally:
        trader._release_ownership()


def scenario_market_vanishes(config) -> dict:
    """
    A position whose data goes stale and then stops arriving entirely.

    Recorded as it behaves TODAY, not as it ought to behave. The round-3 cost
    audit asks whether a position can sit open indefinitely valued at cost; if
    a fix changes that, this hash must change with it and the commit must say so.
    """
    clock = FakeClock(T0)
    venue = ScriptedVenue(clock, (1.00e-4, 1.22e-4, 1.40e-4, 1.46e-4))
    cfg = _isolated(config, "vanish",
                    market=dataclasses.replace(config.market, max_retries=0,
                                               cache_ttl_s=0.0))
    trader = _run(cfg, clock, venue=venue)
    try:
        asyncio.run(trader.on_alert(
            alert(clock, liquidity_usd=GOOD_MARKET["liquidity_usd"],
                  volume_usd=GOOD_MARKET["volume_5m_usd"]), None))
        venue.fail = True
        _ticks(trader, clock, 8, seconds=30.0)
        return _snapshot(trader)
    finally:
        trader._release_ownership()


def scenario_restart_with_open_position(config) -> dict:
    """
    Open a position, persist, then rebuild the trader on the same state.

    The reloaded position must be the same position. A silent change in what
    survives a restart is exactly the class of regression a fingerprint is for.
    """
    cfg = _isolated(config, "restart")
    clock = FakeClock(T0)
    venue = ScriptedVenue(clock, (1.00e-4, 1.22e-4, 1.40e-4, 1.46e-4))
    first = _run(cfg, clock, venue=venue)
    try:
        asyncio.run(first.on_alert(
            alert(clock, liquidity_usd=GOOD_MARKET["liquidity_usd"],
                  volume_usd=GOOD_MARKET["volume_5m_usd"]), None))
        before = _positions_of(first)
        first.persist()
    finally:
        first._release_ownership()

    clock2 = FakeClock(clock.now())
    venue2 = ScriptedVenue(clock2, (1.46e-4,))
    second = _run(cfg, clock2, venue=venue2)
    try:
        after = _positions_of(second)
        return {
            "before": before,
            "after": after,
            "account": _account_of(second),
            "reloaded_open": len(second.account.open_positions()),
        }
    finally:
        second._release_ownership()


SCENARIOS = (
    ("runner", scenario_runner),
    ("fast_failure", scenario_fast_failure),
    ("gate_refuses", scenario_gate_refuses),
    ("stale_signal", scenario_stale_signal),
    ("market_vanishes", scenario_market_vanishes),
    ("restart_open_position", scenario_restart_with_open_position),
)


def golden_payload(config) -> dict:
    return {name: fn(config) for name, fn in SCENARIOS}


#: The session is deterministic, so one run serves every assertion that only
#: reads it. Each test used to rebuild it, so the module paid four times over.
#: ``config`` is function-scoped (it uses tmp_path), so this is a memo rather
#: than a module-scoped fixture; paths are not in the payload, so the first
#: test's config serves the rest.
_CACHE: dict = {}


def _payload_once(config) -> dict:
    if "payload" not in _CACHE:
        _CACHE["payload"] = golden_payload(config)
    return _CACHE["payload"]


# ---------------------------------------------------------------------------
# the tests
# ---------------------------------------------------------------------------


def test_the_golden_run_matches_its_recorded_fingerprint(config):
    payload = _payload_once(config)
    actual = fingerprint(payload)
    assert actual == GOLDEN_SHA256, (
        "The scripted session's behaviour changed.\n"
        f"  expected {GOLDEN_SHA256}\n"
        f"  actual   {actual}\n"
        "If Veha approved a behaviour change, update GOLDEN_SHA256 and put "
        "the old and new hash in the commit message. If not, this is a "
        "regression: revert the change rather than the hash."
    )


@pytest.mark.slow
def test_the_fingerprint_is_stable_within_one_process(config):
    """
    Two INDEPENDENT runs in one process must agree.

    Marked slow because it is the only test here that cannot reuse the memo --
    proving determinism requires actually running the session twice -- and it
    therefore doubles the module's cost. The default per-commit set gets the
    hash match; this is the deeper check.
    """
    assert fingerprint(golden_payload(config)) == \
        fingerprint(golden_payload(config))


def test_every_scenario_actually_exercised_its_path(config):
    """
    A fingerprint over six scenarios that silently stopped doing anything would
    still be stable. These assertions are what make the hash mean something.
    """
    payload = _payload_once(config)
    assert payload["runner"]["counters"]["buys"] == 1
    assert payload["fast_failure"]["counters"]["buys"] == 1
    assert payload["gate_refuses"]["decision"]["decision"] == "REJECT"
    assert payload["gate_refuses"]["counters"]["buys"] == 0
    assert payload["stale_signal"]["execution"]["reason"] == "stale_signal"
    assert payload["stale_signal"]["counters"]["blocks"] == 1
    assert payload["market_vanishes"]["counters"]["buys"] == 1
    # The tick counts are tuned for speed, so assert the exit paths are still
    # REACHED. Without this, trimming a tick could silently stop exercising an
    # exit while leaving the fingerprint perfectly stable.
    statuses = [p["status"] for p in payload["fast_failure"]["positions"]]
    assert "CLOSED" in statuses, f"fast_failure never closed: {statuses}"
    closed = [p for p in payload["fast_failure"]["positions"]
              if p["status"] == "CLOSED"]
    assert closed[0]["exit_reason"], "fast_failure closed with no reason"
    # The runner is expected to stay OPEN -- see scenario_runner's docstring.
    # Asserted rather than assumed, so a change that gives a winning position
    # an exit shows up here as well as in the hash.
    assert [p["status"] for p in payload["runner"]["positions"]] == ["OPEN"]
    assert payload["runner"]["positions"][0]["realized_pnl_usd"] == 0.0
    assert payload["restart_open_position"]["reloaded_open"] == 1
    # the accounting identity holds in every scenario that moved money
    for name in ("runner", "fast_failure", "market_vanishes"):
        assert payload[name]["account"]["identity_check"] == 0.0, name


def test_the_fingerprint_excludes_ids_and_paths(config):
    """
    Ids are uuid4 (models.py:142) and paths are tmp_path. If either reached the
    payload the hash could never be hard-coded.
    """
    text = canonical(_payload_once(config))
    assert "/tmp" not in text
    assert "cand_" not in text
    assert "pos_" not in text
    assert "trade_" not in text
