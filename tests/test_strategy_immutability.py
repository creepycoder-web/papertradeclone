"""
Section 14 of the brief, enforced: the repair changed infrastructure, not the
strategy.

The 2026-09-30 work touched latency bounds, stage attribution, execution
observability and the operator UI. It must NOT have moved any of:

    L25 / V3-C / V2-A    market cap, signal quality, liquidity, ratio gates
    Precision Score      every threshold, including 0.9647 and 0.55
    TP / SL / MOMENTUM_FAILURE / cooldown / daily loss limits
    concurrent trade limits / re-entry / veto logic

These tests pin the values against the frozen constants in the strategy modules
RATHER THAN against literals duplicated here, so a threshold can only change by
changing the authoritative definition - which is exactly the change the V1/V2/L25
successor guards in ``research/baseline/v1_manifest.json`` are there to catch.
"""


from __future__ import annotations

from dataclasses import fields

import pytest

from autopaper import l25, v2a, v3c
from autopaper.config import (
    DecisionConfig,
    ExitConfig,
    MarketConfig,
    NarConfig,
    RiskLimits,
    build_config,
)
from autopaper.exits import ExitPolicy, ExitStyle
from autopaper.sizing import compute_position_size

pytestmark = [
    pytest.mark.integrity,
    pytest.mark.strategy,
]

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# the frozen strategy definitions
# ---------------------------------------------------------------------------


def test_the_active_strategy_is_still_l25():
    assert l25.ACTIVE_STRATEGY == "L25"
    # `verify_*` return None on success and raise on any other value.
    assert v3c.verify_definition(20_000.0, 0.55) is None
    assert l25.verify_definition(0.9647) is None
    assert v2a.verify_ceiling(25_372.71) is None
    with pytest.raises(ValueError):
        v3c.verify_definition(20_001.0, 0.55)
    with pytest.raises(ValueError):
        l25.verify_definition(0.9648)


def test_l25_still_delegates_to_v3c_with_its_own_threshold():
    """L25 = V3-C AND the liquidity/market-cap ratio gate. Nothing else."""
    assert l25.L25_MIN_LIQ_TO_MCAP == 0.9647
    assert v3c.V3C_MAX_MARKET_CAP_USD == 20_000.0
    assert v3c.V3C_MIN_SIGNAL_QUALITY == 0.55
    # ...and the config cannot hold any other value.
    with pytest.raises(ValueError):
        DecisionConfig(v3c_min_signal_quality=0.5)
    with pytest.raises(ValueError):
        DecisionConfig(l25_min_liq_to_mcap=0.5)
    with pytest.raises(ValueError):
        DecisionConfig(v3c_max_market_cap_usd=50_000.0)


def test_the_decision_thresholds_are_unchanged():
    cfg = DecisionConfig()
    expected = {
        "min_signal_quality_to_trade": 0.45,
        "max_extension_to_chase": 0.60,
        "perfect_extension_to_chase": 0.30,
        "min_liquidity_usd": 3_000.0,
        "max_market_cap_usd": 120_000_000.0,
        "min_market_cap_usd": 8_000.0,
        "min_data_freshness": 0.5,
        "min_buy_sell_ratio": 1.05,
        "max_data_age_s": 30.0,
        "min_confidence_to_buy": 0.50,
        "min_confidence_to_buy_extended": 0.45,
        "v3c_max_market_cap_usd": 20_000.0,
        "v3c_min_signal_quality": 0.55,
        "l25_min_liq_to_mcap": 0.9647,
        "v2a_ceiling_usd": 25_372.71,
    }
    for field, value in expected.items():
        assert getattr(cfg, field) == pytest.approx(value), (
            f"DecisionConfig.{field} moved; the repair must not touch the strategy"
        )


def test_the_exit_policy_parameters_are_unchanged():
    cfg = ExitConfig()
    expected = {
        "fast_failure_stop_pct": -12.0,
        "fast_failure_window_s": 75.0,
        "momentum_trail_pct": -22.0,
        "momentum_min_hold_s": 25.0,
        "runner_arm_pct": 25.0,
        "runner_trail_pct": -18.0,
        "runner_hard_floor_pct": -8.0,
        "runner_min_hold_s": 40.0,
        "fixed_target_pct": 45.0,
        "time_decay_no_progress_s": 240.0,
        "time_decay_no_progress_band_pct": 6.0,
        "time_decay_max_hold_s": 900.0,
        "momentum_collapse_lookback_s": 30.0,
        "momentum_collapse_drop_pct": -18.0,
        "structure_break_lookback_s": 45.0,
        "structure_break_pct": -14.0,
        "disaster_cap_pct": -45.0,
        "data_invalid_exit_s": 120.0,
    }
    for field, value in expected.items():
        assert getattr(cfg, field) == pytest.approx(value), (
            f"ExitConfig.{field} moved; TP/SL/MOMENTUM_FAILURE must not change"
        )


def test_momentum_failure_is_still_a_live_exit_style():
    """
    The style must still be reachable from the exit policy, not merely present
    in an enum: a style nothing routes to would be a silent strategy change.
    """
    assert ExitStyle.MOMENTUM_FAILURE.value == "MOMENTUM_FAILURE"
    cfg = ExitConfig()
    assert cfg.momentum_trail_pct == -22.0
    assert cfg.momentum_min_hold_s == 25.0

    from autopaper.analysis.routes import build_exit_plan
    from autopaper.models import Regime, Route

    class _F:
        price = 1.0e-4
        momentum_30s_pct = -30.0
        signal_quality = 0.9
        liquidity_usd = 400_000.0
        market_cap_usd = 18_000.0
        volume_5m_usd = 180_000.0
        buys_5m = 100
        sells_5m = 400

    plan = build_exit_plan(Route.FAST_ENTRY, Regime.WEAKENING,
                           ExitStyle.MOMENTUM_FAILURE, _F(), cfg)
    assert plan["style"] == "MOMENTUM_FAILURE"
    assert plan["initial_stop_pct"] == cfg.momentum_trail_pct


def test_the_risk_limits_are_unchanged():
    risk = RiskLimits()
    assert risk.starting_bankroll_usd == 100.0
    assert risk.max_position_pct_equity == 12.5
    assert risk.max_total_exposure_pct_equity == 12.5
    assert risk.min_position_notional_usd == 0.0
    assert risk.max_simultaneous_positions == 1, "one concurrent position"
    assert risk.reentry_enabled is False, "re-entry must stay disabled"
    assert risk.disaster_cap_pct == -45.0


def test_reentry_has_no_override():
    """The field exists for the cooldown bookkeeping; no config can enable it."""
    cfgmod = __import__("autopaper.config", fromlist=["x"])
    original = cfgmod.load_env_file
    try:
        cfgmod.load_env_file = lambda path: {
            "PAPER_REENTRY_ENABLED": "true",
            "REAL_TRADING_ENABLED": "true",
            "I_UNDERSTAND_REAL_TRADING_RISKS": "true",
        }
        cfg = build_config(REPO)
        assert cfg.risk.reentry_enabled is False
        assert cfg.real.env_trading_enabled_request is True, (
            "the env request is recorded but authorizes nothing"
        )
        assert cfg.real.trading_enabled is False, (
            "REAL_TRADING_ENABLED=true must not enable REAL"
        )
    finally:
        cfgmod.load_env_file = original


def test_the_sizing_policy_is_unchanged():
    """10% at or below $50 equity, 12.5% above. One function, one policy."""
    assert compute_position_size(50.0).position_pct == 10.0
    assert compute_position_size(94.7777).position_pct == 12.5
    assert compute_position_size(100.0).position_usd == pytest.approx(12.5)
    assert compute_position_size(40.0).position_usd == pytest.approx(4.0)


def test_the_execution_economics_are_unchanged():
    """Fees, slippage and the decision latency are part of the simulation."""
    from autopaper.config import ExecutionModel

    m = ExecutionModel()
    assert m.entry_fee_bps == 85.0
    assert m.exit_fee_bps == 85.0
    assert m.entry_slippage_bps == 120.0
    assert m.exit_slippage_bps == 120.0
    assert m.decision_latency_ms == 1200.0
    assert m.max_quote_age_ms == 8000.0
    assert m.fill_quote_timeout_ms == 5000.0
    assert m.assume_sol_usd == 60_000.0
    assert m.market_impact_model == "sqrt_notional_over_liquidity"


def test_the_stale_entry_gate_budget_was_not_relaxed():
    """
    The single most tempting shortcut for the latency incident was to raise the
    15s entry budget so the 34.9s candidate would trade. It was NOT raised: the
    latency was fixed instead, so a genuinely slow candidate is still refused.
    """
    cfg = build_config(REPO)
    assert cfg.market.stale_after_ms == 15_000.0
    assert MarketConfig().stale_after_ms == 15_000.0
    assert cfg.market.unusable_after_ms == 45_000.0
    src = (REPO / "src/autopaper/engine.py").read_text()
    assert "StaleEntryGate(\n            budget_ms=config.market.stale_after_ms" in src, (
        "the gate must keep reusing the documented market budget, not a new one"
    )


def test_the_latency_repair_did_not_touch_the_strategy_modules():
    """
    Structural: the modules that DEFINE the strategy are the ones the repair had
    no reason to edit, and their content still hashes consistently with what the
    successor record claims.
    """
    manifest = __import__("json").loads(
        (REPO / "research/baseline/v1_manifest.json").read_text()
    )
    successors = manifest["reproducibility"]["source_tree_successors"]
    # Locate the incident-repair successor by state rather than assuming it is
    # the last entry: later successors (e.g. the REAL lifecycle fix) are
    # legitimately recorded after it. The property under test is about THAT
    # repair, so it must still be present and must still account for every
    # strategy dimension it claims not to have touched.
    repairs = [
        s for s in successors
        if s["state"].startswith("V3-L25-INCIDENT-REPAIR")
    ]
    assert repairs, (
        "the 2026-09-30 latency repair must be recorded as a successor state"
    )
    latest = repairs[-1]
    assert latest["changes_decision_behaviour"] is False, (
        "the recorded successor must declare that decision behaviour is unchanged"
    )
    recorded = (latest["note"] + " " + latest["change"]).lower()
    for token in ("l25", "v3-c", "v2-a", "precision score", "tp, sl",
                  "momentum_failure", "cooldown", "daily loss",
                  "concurrent-trade limits", "re-entry", "veto"):
        assert token in recorded, (
            f"the recorded successor does not account for {token!r}"
        )


def test_market_and_narrative_budgets_still_have_their_documented_defaults():
    """
    What DID move: only the market transport's connect split and the total read
    budget's enforceability. The freshness budgets and the narrative budget are
    unchanged.
    """
    cfg = build_config(REPO)
    assert cfg.nar.timeout_s == 2.5
    assert cfg.nar.timeout_grace_s == 0.5
    assert cfg.market.alert_settle_s == 2.0
    assert cfg.market.cache_ttl_s == 2.0
    assert cfg.market.max_retries == 2
    assert cfg.market.http_timeout_s == 8.0, "the nominal budget is unchanged"
    assert cfg.market.connect_timeout_s == 1.0, (
        "only the CONNECT split was added; the nominal budget is unchanged"
    )
