"""
R2: counterfactual tracking of REJECT and WAIT decisions.

The point of counterfactual tracking is to find out which gates block winners. That is
only answerable if the counterfactual is computed by the SAME fill and exit
model the live path uses, and if the counterfactual data can never reach a live
decision. Both are asserted here, along with the resource bounds: counterfactuals are
capped, drops are counted, and live work keeps rate-limit priority.
"""

from __future__ import annotations

import json

import pytest

from autopaper.clock import FakeClock
from autopaper.config import ExecutionModel, ExitConfig, CounterfactualConfig
from autopaper.models import (
    DataState,
    Decision,
    Observation,
    ObservationSet,
    Regime,
    Route,
)
from autopaper.counterfactual import COUNTERFACTUAL_FILE, CounterfactualTracker

pytestmark = pytest.mark.integrity

T0 = 1_700_000_000.0
MINT = "So11111111111111111111111111111111111111112"


def obs(at: float, price: float) -> Observation:
    return Observation(
        mint=MINT, observed_at=at, received_at=at, price_usd=price,
        liquidity_usd=60_000.0, market_cap_usd=15_000.0, provider="test",
    )


def tracker(tmp_path, clock=None, **kw) -> CounterfactualTracker:
    return CounterfactualTracker(
        tmp_path / COUNTERFACTUAL_FILE,
        CounterfactualConfig(**kw),
        execution=ExecutionModel(),
        exits=ExitConfig(),
        clock=clock or FakeClock(T0),
    )


def rows(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def track(t, *, decision=Decision.REJECT, now=T0, reasons=("cost_too_high",)):
    return t.track(
        candidate_id="cand-1", mint=MINT, symbol="TEST",
        decision=decision, route=Route.FAST_ENTRY, regime=Regime.EARLY,
        reject_reasons=reasons, price=1.0e-4, liquidity_usd=60_000.0,
        now=now,
    )


def history(prices, start=T0):
    return ObservationSet([obs(start + i * 60.0, p) for i, p in enumerate(prices)])


# ---------------------------------------------------------------------------
# what is tracked
# ---------------------------------------------------------------------------


def test_reject_and_wait_are_tracked(tmp_path):
    t = tracker(tmp_path)
    assert track(t, decision=Decision.REJECT) is True
    assert t.track(
        candidate_id="cand-2", mint=MINT, symbol="TEST",
        decision=Decision.WAIT, route=Route.FAST_ENTRY, regime=Regime.EARLY,
        reject_reasons=("x",), price=1.0e-4, liquidity_usd=60_000.0, now=T0,
    ) is True
    assert len(t.open_tracked()) == 2


def test_the_same_candidate_is_not_tracked_twice(tmp_path):
    t = tracker(tmp_path)
    assert track(t) is True
    assert track(t) is False
    assert len(t.open_tracked()) == 1


def test_a_buy_is_not_tracked(tmp_path):
    """A taken trade is in the ledger. Tracking it would double-count it."""
    t = tracker(tmp_path)
    assert track(t, decision=Decision.BUY) is False
    assert t.open_tracked() == []


def test_disabled_tracks_nothing(tmp_path):
    t = tracker(tmp_path, enabled=False)
    assert track(t) is False
    assert t.open_tracked() == []


# ---------------------------------------------------------------------------
# resource bounds
# ---------------------------------------------------------------------------


def test_the_cap_is_enforced_and_drops_are_counted(tmp_path):
    t = tracker(tmp_path, max_tracked=2)
    for i in range(5):
        t.track(
            candidate_id=f"c{i}", mint=f"M{i}", symbol="T",
            decision=Decision.REJECT, route=Route.FAST_ENTRY,
            regime=Regime.EARLY, reject_reasons=("x",), price=1.0e-4,
            liquidity_usd=60_000.0, now=T0,
        )
    assert len(t.open_tracked()) == 2
    assert t.dropped == 3


def test_live_work_has_rate_limit_priority(tmp_path):
    """
    The tracker never asks for data of its own. It is handed whatever the live
    loop already fetched, so a counterfactual cannot consume a request a decision needs.
    """
    t = tracker(tmp_path)
    assert not hasattr(t, "market")
    assert "market" not in t.needs()


# ---------------------------------------------------------------------------
# the counterfactual
# ---------------------------------------------------------------------------


def test_nothing_is_written_before_the_outcome_resolves(tmp_path):
    t = tracker(tmp_path)
    track(t)
    t.resolve(history([1.0e-4, 1.1e-4]), now=T0 + 120.0)
    assert rows(t.path) == [], "an unresolved counterfactual was written"


def test_a_resolved_counterfactual_is_written_once(tmp_path):
    t = tracker(tmp_path)
    track(t)
    t.resolve(history([1.0e-4] * 40), now=T0 + 1801.0)
    first = rows(t.path)
    assert len(first) == 1
    t.resolve(history([1.0e-4] * 40), now=T0 + 3600.0)
    assert rows(t.path) == first, "the counterfactual resolved twice"
    assert t.open_tracked() == []


def test_the_window_bounds_the_hold(tmp_path):
    t = tracker(tmp_path, window_s=600.0)
    track(t)
    t.resolve(history([1.0e-4] * 40), now=T0 + 601.0)
    assert len(rows(t.path)) == 1


def test_a_winner_and_a_loser_are_recorded_with_their_sign(tmp_path):
    up = tracker(tmp_path / "up")
    track(up)
    up.resolve(history([1.0e-4, 3.0e-4, 5.0e-4] + [5.0e-4] * 40), now=T0 + 1801.0)
    down = tracker(tmp_path / "down")
    track(down)
    down.resolve(history([1.0e-4, 0.5e-4] + [0.4e-4] * 40), now=T0 + 1801.0)
    assert rows(up.path)[0]["net_pnl_usd"] > 0
    assert rows(down.path)[0]["net_pnl_usd"] < 0


def test_the_row_names_the_reasons_that_blocked_the_trade(tmp_path):
    t = tracker(tmp_path)
    track(t, reasons=("cost_too_high", "liquidity_too_low"))
    t.resolve(history([1.0e-4] * 40), now=T0 + 1801.0)
    row = rows(t.path)[0]
    assert row["reject_reasons"] == ["cost_too_high", "liquidity_too_low"]
    assert row["decision"] == "REJECT"
    assert row["counterfactual"] is True


def test_the_exit_reason_comes_from_the_real_exit_policy(tmp_path):
    """
    The counterfactual must be the same exit policy, not a 30-minute
    buy-and-hold. A token that collapses exits on the declared stop.
    """
    t = tracker(tmp_path)
    track(t)
    t.resolve(history([1.0e-4, 0.5e-4] + [0.4e-4] * 40), now=T0 + 1801.0)
    row = rows(t.path)[0]
    assert row["exit_trigger"] in {
        "fast_failure_stop", "disaster_cap", "momentum_collapse",
        "structure_break", "no_follow_through",
    }, row["exit_trigger"]


def test_an_unfillable_resolve_counterfactualss_as_unfilled_not_as_a_trade(tmp_path):
    t = tracker(tmp_path)
    track(t)
    t.resolve(ObservationSet(), now=T0 + 1801.0)
    row = rows(t.path)[0]
    assert row["filled"] is False
    assert row["net_pnl_usd"] is None


# ---------------------------------------------------------------------------
# isolation
# ---------------------------------------------------------------------------


def test_no_lookahead_only_observations_at_or_before_now_are_read(tmp_path):
    """
    A future print must not change a counterfactual outcome. The same history with a
    +400% spike appended AFTER the window must resolve identically.
    """
    base = history([1.0e-4, 1.05e-4] + [1.05e-4] * 40)
    future = ObservationSet(list(base) + [obs(T0 + 5000.0, 50.0e-4)])
    a = tracker(tmp_path / "a")
    track(a)
    a.resolve(base, now=T0 + 1801.0)
    b = tracker(tmp_path / "b")
    track(b)
    b.resolve(future, now=T0 + 1801.0)
    ra, rb = rows(a.path)[0], rows(b.path)[0]
    for key in ("net_pnl_usd", "return_pct", "exit_trigger", "filled"):
        assert ra[key] == rb[key], key


def test_counterfactual_outcomes_never_reach_live_memory(tmp_path):
    """
    ``ResearchMemory`` adapts route sizing from CLOSED trades. A counterfactual is not
    a trade, and the tracker has no way to hand one over.
    """
    from autopaper.memory import ResearchMemory

    t = tracker(tmp_path)
    track(t)
    t.resolve(history([1.0e-4] * 40), now=T0 + 1801.0)
    memory = ResearchMemory()
    assert len(memory) == 0
    assert not hasattr(t, "memory")
    assert "memory" not in t.needs()


def test_the_stream_is_append_only(tmp_path):
    t = tracker(tmp_path)
    track(t)
    t.resolve(history([1.0e-4] * 40), now=T0 + 1801.0)
    t.track(
        candidate_id="cand-2", mint=MINT, symbol="TEST",
        decision=Decision.WAIT, route=Route.FAST_ENTRY, regime=Regime.EARLY,
        reject_reasons=("y",), price=1.0e-4, liquidity_usd=60_000.0,
        now=T0 + 1802.0,
    )
    t.resolve(history([1.0e-4] * 80), now=T0 + 3700.0)
    assert len(rows(t.path)) == 2


def test_status_reports_counts(tmp_path):
    t = tracker(tmp_path, max_tracked=1)
    track(t)
    t.track(candidate_id="c2", mint="M2", symbol="T",
            decision=Decision.REJECT, route=Route.FAST_ENTRY,
            regime=Regime.EARLY, reject_reasons=("x",), price=1.0e-4,
            liquidity_usd=60_000.0, now=T0)
    t.resolve(history([1.0e-4] * 40), now=T0 + 1801.0)
    st = t.status()
    assert st["resolved"] == 1
    assert st["dropped"] == 1
    assert st["open"] == 0
