"""
Dynamic exit policy, runner mode, and loss control.

Exits are route- and regime-dependent on purpose: the whole point is that a
fragile explosive entry and a healthy continuation do not get the same
treatment, and that a big winner is allowed to keep running.
"""

from __future__ import annotations

import pytest

from autopaper.config import ExitConfig
from autopaper.exits import ExitPolicy
from autopaper.models import DataState, ExitStyle, Position, Regime, Route

pytestmark = pytest.mark.strategy

CFG = ExitConfig()


def position(style: ExitStyle, *, fill=1.0, route=Route.FAST_ENTRY,
             regime=Regime.EARLY) -> Position:
    return Position(
        position_id="p1", candidate_id="c", mint="M", symbol="TOK",
        status="OPEN", route=route, regime=regime, exit_style=style,
        strategy=route.value, entry_signal_timestamp=0.0,
        entry_decision_timestamp=0.0, entry_timestamp=0.0, quoted_price=fill,
        entry_reference_price=fill, fill_price=fill, quantity=100.0,
        cash_out_usd=100.0, cost_basis_usd=100.0, peak_price=fill,
        trough_price=fill,
    )


def price(p: Position, price_usd: float) -> None:
    ret = (price_usd / p.fill_price - 1.0) * 100.0
    p.current_price = price_usd
    p.mark_price = price_usd
    p.current_return_pct = ret
    if p.peak_price is None or price_usd > p.peak_price:
        p.peak_price = price_usd
        p.peak_return_pct = ret
    if p.trough_price is None or price_usd < p.trough_price:
        p.trough_price = price_usd
    p.mfe_pct = p.peak_return_pct
    p.mae_pct = ret


def assess(p: Position, price_usd: float | None, now: float = 60.0, **kw):
    if price_usd is not None:
        price(p, price_usd)
    policy = ExitPolicy(CFG)
    kw.setdefault("liquidity_change_pct", None)
    return policy.assess(
        p, now=now, mark_price=p.mark_price, data_state=DataState.LIVE, **kw
    )


# ---------------------------------------------------------------------------
# runner mode
# ---------------------------------------------------------------------------


def test_runner_arms_and_keeps_running():
    p = position(ExitStyle.RUNNER)
    d = assess(p, 1.30, momentum_30s_pct=4.0, structure_30s_pct=2.0)
    assert d.action == "HOLD"
    assert p.runner_mode is True
    assert p.runner_armed_at == 60.0
    # still strong much later: no fixed target, no panic exit
    d2 = assess(p, 2.50, now=300.0, momentum_30s_pct=3.0, structure_30s_pct=1.0)
    assert d2.action == "HOLD"
    # +250% is exactly what a "take profit at +20%" rule would have thrown away
    assert d2.reason is not None


def test_runner_exits_on_giveback():
    p = position(ExitStyle.RUNNER)
    assess(p, 1.30, momentum_30s_pct=4.0, structure_30s_pct=2.0)
    d = assess(p, 1.05, now=200.0, momentum_30s_pct=-1.0, structure_30s_pct=-1.0)
    assert d.action == "EXIT"
    assert d.trigger == "runner_giveback"
    assert "peak" in d.reason


def test_runner_has_a_floor_so_it_can_still_lose():
    p = position(ExitStyle.RUNNER)
    assess(p, 1.30, momentum_30s_pct=4.0, structure_30s_pct=2.0)
    d = assess(p, 0.90, now=120.0, momentum_30s_pct=-1.0, structure_30s_pct=-1.0)
    assert d.action == "EXIT"
    assert d.trigger in ("runner_floor", "runner_giveback", "disaster_cap")


def test_runner_does_not_arm_while_falling():
    p = position(ExitStyle.RUNNER)
    d = assess(p, 1.30, momentum_30s_pct=-9.0, structure_30s_pct=-3.0)
    assert p.runner_mode is False
    assert d.action == "HOLD"


def test_pre_runner_stop_is_quicker_than_a_runner_trail():
    p = position(ExitStyle.RUNNER)
    d = assess(p, 0.86, now=60.0, momentum_30s_pct=-1.0, structure_30s_pct=-1.0)
    assert d.action == "EXIT"
    assert d.trigger == "pre_runner_stop"


# ---------------------------------------------------------------------------
# fast failure
# ---------------------------------------------------------------------------


def test_fast_failure_has_a_tight_stop_and_a_short_fuse():
    p = position(ExitStyle.FAST_FAILURE, route=Route.FAST_ENTRY,
                 regime=Regime.EXPLOSIVE)
    d = assess(p, 0.87, now=30.0, momentum_30s_pct=-2.0, structure_30s_pct=-1.0)
    assert d.action == "EXIT"
    assert d.trigger == "fast_failure_stop"


def test_fast_failure_exits_on_no_follow_through():
    p = position(ExitStyle.FAST_FAILURE)
    d = assess(p, 1.02, now=100.0, momentum_30s_pct=0.5, structure_30s_pct=0.5)
    assert d.action == "EXIT"
    assert d.trigger == "no_follow_through"


def test_fast_failure_respects_a_healthy_move():
    p = position(ExitStyle.FAST_FAILURE)
    d = assess(p, 1.10, now=30.0, momentum_30s_pct=3.0, structure_30s_pct=1.0)
    assert d.action == "HOLD"


def test_momentum_collapse_exits_immediately():
    p = position(ExitStyle.MOMENTUM_FAILURE)
    d = assess(p, 1.02, now=60.0, momentum_30s_pct=-20.0, structure_30s_pct=-3.0)
    assert d.action == "EXIT"
    assert d.trigger == "momentum_collapse"


def test_structure_break_exits():
    p = position(ExitStyle.MOMENTUM_FAILURE)
    d = assess(p, 1.02, now=60.0, momentum_30s_pct=-1.0, structure_30s_pct=-15.0)
    assert d.action == "EXIT"
    assert d.trigger in ("structure_break", "deep_drawdown", "wide_giveback")


# ---------------------------------------------------------------------------
# time decay / fixed target
# ---------------------------------------------------------------------------


def test_time_decay_kills_a_dead_setup():
    p = position(ExitStyle.TIME_DECAY)
    d = assess(p, 1.01, now=CFG.time_decay_no_progress_s + 5,
               momentum_30s_pct=0.0, structure_30s_pct=0.0)
    assert d.action == "EXIT"
    assert d.trigger == "time_decay_no_progress"


def test_time_decay_gives_a_real_setup_room():
    p = position(ExitStyle.TIME_DECAY)
    d = assess(p, 1.15, now=CFG.time_decay_no_progress_s + 5,
               momentum_30s_pct=1.0, structure_30s_pct=0.5)
    assert d.action == "HOLD"


def test_fixed_target_only_where_declared():
    p = position(ExitStyle.FIXED_TARGET)
    d = assess(p, 1.50, now=300.0, momentum_30s_pct=1.0, structure_30s_pct=0.5)
    assert d.action == "EXIT"
    assert d.trigger == "fixed_target"


# ---------------------------------------------------------------------------
# disaster cap + data integrity
# ---------------------------------------------------------------------------


def test_disaster_cap_is_a_hard_boundary():
    for style in (ExitStyle.FAST_FAILURE, ExitStyle.MOMENTUM_FAILURE,
                  ExitStyle.RUNNER, ExitStyle.TIME_DECAY):
        p = position(style)
        d = assess(p, 0.50, now=30.0, momentum_30s_pct=0.0, structure_30s_pct=0.0)
        assert d.action == "EXIT", style
        assert d.style is ExitStyle.DISASTER_CAP, style


def test_unusable_data_holds_then_exits_on_the_timer():
    p = position(ExitStyle.MOMENTUM_FAILURE)
    policy = ExitPolicy(CFG)
    d = policy.assess(
        p, now=10.0, mark_price=None, data_state=DataState.UNUSABLE,
        momentum_30s_pct=None, structure_30s_pct=None, liquidity_change_pct=None,
        data_invalid_since=10.0,
    )
    assert d.action == "HOLD"
    assert "DATA_UNCERTAIN" in d.reason

    d2 = policy.assess(
        p, now=10.0 + CFG.data_invalid_exit_s + 1, mark_price=None,
        data_state=DataState.UNUSABLE, momentum_30s_pct=None,
        structure_30s_pct=None, liquidity_change_pct=None,
        data_invalid_since=10.0,
    )
    assert d2.action == "EXIT"
    assert d2.trigger == "data_invalid"
    assert d2.style is ExitStyle.DATA_INVALID


def test_stale_data_is_not_treated_as_live():
    p = position(ExitStyle.MOMENTUM_FAILURE)
    policy = ExitPolicy(CFG)
    d = policy.assess(
        p, now=60.0, mark_price=0.5, data_state=DataState.STALE,
        momentum_30s_pct=None, structure_30s_pct=None, liquidity_change_pct=None,
    )
    assert d.action == "HOLD", "a stale mark must not trigger a panic exit"
    assert "DATA_UNCERTAIN" in d.reason


def test_liquidity_collapse_blocks_runner_arming():
    p = position(ExitStyle.RUNNER)
    d = assess(p, 1.30, momentum_30s_pct=3.0, structure_30s_pct=1.0,
               liquidity_change_pct=-40.0)
    assert p.runner_mode is False
    assert d.action == "HOLD"


# ---------------------------------------------------------------------------
# invalidation is pre-declared
# ---------------------------------------------------------------------------


def test_every_exit_trigger_has_a_predeclared_rule():
    """
    The trigger string in an exit reason must be reproducible from the exit
    plan that existed at entry. An invented-after-the-fact reason is a failure.
    """
    from autopaper.analysis.routes import build_exit_plan
    from autopaper.analysis.features import FeatureSet

    allowed = {
        "fast_failure_stop", "momentum_collapse", "structure_break",
        "no_follow_through", "runner_giveback", "runner_floor",
        "pre_runner_stop", "pre_runner_giveback", "peak_giveback",
        "deep_drawdown", "wide_giveback", "time_decay_no_progress", "max_hold",
        "fixed_target", "disaster_cap", "data_invalid", "flatten",
    }
    policy = ExitPolicy(CFG)
    for style in ExitStyle:
        p = position(style)
        for price_usd in [0.3, 0.5, 0.8, 0.88, 0.95, 1.0, 1.05, 1.3, 1.6, 2.0, 3.0]:
            for t in (1.0, 30.0, 100.0, 300.0, 700.0, 1000.0, 2000.0):
                for m in (-25.0, -3.0, 0.0, 3.0):
                    q = position(style)
                    d = policy.assess(
                        q, now=t, mark_price=None, data_state=DataState.LIVE,
                        momentum_30s_pct=m, structure_30s_pct=-1.0,
                        liquidity_change_pct=None,
                    )
                    assert d.action in ("HOLD", "EXIT")
                    if d.action == "EXIT":
                        assert d.trigger in allowed, (style, price_usd, t, m, d.trigger)
                        assert d.reason, (style, d)
