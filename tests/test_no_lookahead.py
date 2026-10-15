"""
NO LOOKAHEAD.

The rule: at decision time T the agent may only see observations with
``observed_at <= T`` and trades that closed before T. These tests build a
history containing a violent future move and prove the decision does not change.
"""

from __future__ import annotations

import dataclasses

import pytest

from autopaper.analysis import DecisionEngine, build_features
from autopaper.clock import FakeClock
from autopaper.config import DecisionConfig, ExitConfig
from autopaper.memory import ResearchMemory
from autopaper.models import DataState, Decision, NarSnapshot, Observation, ObservationSet, Trade
from autopaper.nar import NullNarSource

from .conftest import MINT, alert, obs

pytestmark = pytest.mark.strategy

T0 = 1_700_000_000.0


def history(*rows: Observation) -> ObservationSet:
    return ObservationSet(rows)


def features_at(obs_set: ObservationSet, t: float, alert=None, nar=None):
    return build_features(
        candidate_id="cand_x",
        mint=MINT,
        decision_time=t,
        history=obs_set,
        alert=alert,
        nar=nar,
        stale_after_ms=15_000.0,
        unusable_after_ms=45_000.0,
    )


def test_observation_set_as_of_hides_the_future():
    past = obs(T0, 1e-4)
    future = obs(T0 + 10, 5e-4)
    oset = history(past, future)
    assert len(oset) == 2
    assert len(oset.as_of(T0)) == 1
    assert oset.as_of(T0).latest().price_usd == 1e-4
    assert oset.latest(T0).price_usd == 1e-4
    assert oset.latest().price_usd == 5e-4


def test_features_ignore_future_observations():
    flat = [obs(T0 - 30, 1e-4), obs(T0 - 20, 1.02e-4), obs(T0 - 10, 1.04e-4), obs(T0, 1.05e-4)]
    with_future = flat + [
        obs(T0 + 5, 9e-4),      # +750%
        obs(T0 + 10, 2e-3),     # +1900%
        obs(T0 + 20, 1e-5),     # crash
    ]
    a = features_at(history(*flat), T0)
    b = features_at(history(*with_future), T0)
    assert a.price == b.price == 1.05e-4
    assert a.change_5m == b.change_5m
    assert a.momentum_30s_pct == b.momentum_30s_pct
    assert a.signal_quality == b.signal_quality
    assert a.history_span_s == b.history_span_s
    assert a.observation_count == b.observation_count == 4


def test_a_future_crash_does_not_change_the_decision():
    engine = DecisionEngine(DecisionConfig(), ExitConfig(), memory=ResearchMemory())
    nar = NullNarSource().fetch(MINT)
    rows = [obs(T0 - 60, 1.0e-4), obs(T0 - 30, 1.6e-4), obs(T0 - 10, 2.1e-4),
            obs(T0, 2.4e-4)]
    good = engine.decide(
        features=features_at(history(*rows), T0, alert=alert(T0), nar=nar),
        alert=alert(T0), nar=nar, analysis_start=T0, analysis_complete=T0 + 0.05,
    )
    # the token dies immediately afterwards; the decision must not know
    with_future = rows + [obs(T0 + 5, 1.2e-4), obs(T0 + 15, 6e-5), obs(T0 + 30, 3e-5)]
    later = engine.decide(
        features=features_at(history(*with_future), T0, alert=alert(T0), nar=nar),
        alert=alert(T0), nar=nar, analysis_start=T0, analysis_complete=T0 + 0.05,
    )
    assert good.decision == later.decision
    assert good.confidence == pytest.approx(later.confidence)
    assert good.regime == later.regime
    assert good.route == later.route


def test_future_rally_does_not_inflate_confidence():
    engine = DecisionEngine(DecisionConfig(), ExitConfig(), memory=ResearchMemory())
    nar = NullNarSource().fetch(MINT)
    rows = [obs(T0 - 45, 1.0e-4), obs(T0 - 20, 1.05e-4), obs(T0, 1.08e-4)]
    base = engine.decide(
        features=features_at(history(*rows), T0, alert=alert(T0), nar=nar),
        alert=alert(T0), nar=nar, analysis_start=T0, analysis_complete=T0,
    )
    moon = rows + [obs(T0 + 5, 1.4e-3), obs(T0 + 10, 3.0e-3)]
    later = engine.decide(
        features=history(*moon) and features_at(history(*moon), T0, alert=alert(T0), nar=nar),
        alert=alert(T0), nar=nar, analysis_start=T0, analysis_complete=T0,
    )
    assert base.confidence == pytest.approx(later.confidence)
    assert base.signal_quality == pytest.approx(later.signal_quality)
    assert base.decision == later.decision


def test_freshness_verdicts():
    o = obs(T0, 1e-4)
    assert o.freshness(T0, 15_000, 45_000) is DataState.LIVE
    assert o.freshness(T0 + 20, 15_000, 45_000) is DataState.STALE
    assert o.freshness(T0 + 60, 15_000, 45_000) is DataState.UNUSABLE
    assert obs(T0, None).freshness(T0, 15_000, 45_000) is DataState.MISSING


def test_snapshot_attributes_are_frozen():
    snap = alert(T0, price_usd=1e-4)
    for field in ("mint", "symbol", "name", "raw_text", "received_timestamp",
                  "source_timestamp", "message_id", "fields", "provenance"):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(snap, field, "tampered")


def test_snapshot_field_map_cannot_be_backfilled():
    """
    The T0 snapshot must not accept later information.

    Attributes are frozen, and the field map is a frozen proxy over an
    immutable copy, so a later market read cannot be written back into the
    record of what was known at T0.
    """
    snap = alert(T0, price_usd=1e-4)
    original = snap.as_dict()
    try:
        snap.fields["liquidity_usd"] = 999  # type: ignore[index]
    except TypeError:
        pass  # immutable mapping refuses the write
    assert snap.as_dict() == original
    assert snap.num("liquidity_usd") is None


def test_memory_only_sees_trades_closed_before_t():
    trades = [
        _trade("a", exit_ts=T0 - 100, net=1.0, ret=10.0),
        _trade("b", exit_ts=T0 - 10, net=-1.0, ret=-10.0),
        _trade("c", exit_ts=T0 + 100, net=50.0, ret=500.0),  # the future
    ]
    mem = ResearchMemory(trades)
    visible = mem.visible_at(T0)
    assert [t.trade_id for t in visible] == ["a", "b"]
    assert len(mem) == 3
    stats = mem.stats_at(T0)
    assert stats["PRECISION_ENTRY"].n == 2
    assert stats["PRECISION_ENTRY"].wins == 1


def test_memory_outcome_does_not_change_a_past_decision():
    """
    Learning only happens after a trade closes. A decision at T must be
    identical whether or not a huge winner happened to close later.
    """
    rows = [obs(T0 - 45, 1.0e-4), obs(T0 - 20, 1.05e-4), obs(T0, 1.08e-4)]
    nar = NullNarSource().fetch(MINT)

    def run(mem: ResearchMemory):
        engine = DecisionEngine(DecisionConfig(), ExitConfig(), memory=mem)
        return engine.decide(
            features=features_at(history(*rows), T0, alert=alert(T0), nar=nar),
            alert=alert(T0), nar=nar, analysis_start=T0, analysis_complete=T0,
        )

    empty = run(ResearchMemory())
    with_future = run(ResearchMemory([
        _trade("late", exit_ts=T0 + 5, net=40.0, ret=400.0, route="PRECISION_ENTRY"),
    ]))
    assert empty.decision == with_future.decision
    assert empty.route_multiplier == pytest.approx(with_future.route_multiplier)
    assert empty.confidence == pytest.approx(with_future.confidence)


def _trade(tid, exit_ts, net, ret, route="PRECISION_ENTRY", regime="EARLY"):
    return Trade(
        trade_id=tid, position_id=f"p_{tid}", candidate_id=f"c_{tid}", mint=MINT,
        symbol="X", source="telegram", strategy=route, route=route,
        regime=regime, entry_signal_timestamp=exit_ts - 30, entry_timestamp=exit_ts - 30,
        entry_price=1e-4, fill_price=1e-4, exit_timestamp=exit_ts, exit_price=1e-4,
        quantity=100.0, entry_fee_usd=0.01, exit_fee_usd=0.01,
        entry_slippage_usd=0.01, exit_slippage_usd=0.01, gross_pnl_usd=net + 0.04,
        net_pnl_usd=net, return_pct=ret, holding_seconds=30.0, mfe_pct=ret,
        mae_pct=-5.0, exit_reason="test", exit_style="MOMENTUM_FAILURE",
        thesis="", invalidation=(), booked_at=exit_ts,
    )
