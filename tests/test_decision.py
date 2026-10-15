"""
Decision schema, anti-extension behaviour, and the decision/report contract.

V3-C EDITION
------------
The engine now has exactly one route, so the assertions here are about V3-C:
a candidate is either a V3-C FAST_ENTRY setup, or it is not tradeable. The
market cap in these tests defaults to under $20,000 so that a decision the test
calls a BUY is a V3-C BUY rather than a market-cap rejection, and the
market-cap and signal-quality gates are exercised explicitly below and in
``test_v3c.py``.
"""

from __future__ import annotations

import json

import pytest

from autopaper.analysis import DecisionEngine, build_features, select_route
from autopaper.config import DecisionConfig, ExitConfig
from autopaper.memory import ResearchMemory
from autopaper.models import (
    DataState,
    Decision,
    NarSnapshot,
    Observation,
    ObservationSet,
    Regime,
    RiskFlag,
    Route,
)
from autopaper.nar import NullNarSource

from .conftest import MINT, alert, obs

pytestmark = pytest.mark.strategy

T0 = 1_700_000_000.0
ENGINE = DecisionEngine(DecisionConfig(), ExitConfig())


def _good_nar() -> "NarSnapshot":
    """
    A NAR snapshot with real flow evidence.

    V3-C's signal-quality floor is 0.55 and a bare market series does not reach
    it, so a test that needs V3-C to AUTHORIZE has to supply the additional
    evidence a real candidate would carry. This is the narrative source, used
    only as a feature input - it is never a candidate source.
    """
    return NarSnapshot(
        mint=MINT, fetched_at=T0, source="test", available=True,
        narrative="a token with a real narrative",
        wallet_flow_net=0.42, holder_count=350, top10_holders_pct=22.0,
        dev_holdings_pct=0.4, bundled_pct=8.0, buy_sell_ratio_1h=3.2,
    )


def series(prices, t0=T0, **kw):
    step = 15.0
    return ObservationSet([
        obs(t0 - step * (len(prices) - 1 - i), p, **kw) for i, p in enumerate(prices)
    ])


def decide(prices, *, alert_obj=None, nar=None, t0=T0, decide_at=None,
           market_cap_usd=15_000.0, **obs_kw):
    """
    `t0` is when the last print happened; `decide_at` is when the agent decides.
    Keeping them separate is what lets a test assert the effect of data going
    stale without rebuilding the series.
    """
    obs_kw.setdefault("market_cap_usd", market_cap_usd)
    obs_kw.setdefault("liquidity_usd", 250_000.0)
    hist = series(prices, t0, **obs_kw)
    when = t0 if decide_at is None else decide_at
    nar = nar if nar is not None else NullNarSource().fetch(MINT)
    a = alert_obj if alert_obj is not None else alert(t0, price_usd=prices[-1])
    f = build_features(
        candidate_id="c1", mint=MINT, decision_time=when, history=hist, alert=a,
        nar=nar, stale_after_ms=15_000.0, unusable_after_ms=45_000.0,
    )
    return ENGINE.decide(
        features=f, alert=a, nar=nar, analysis_start=when, analysis_complete=when + 0.05
    )


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_report_is_machine_readable_and_complete():
    r = decide([1.0e-4, 1.05e-4, 1.10e-4])
    d = r.to_json_dict()
    json.dumps(d)  # must be serialisable
    for key in ("decision", "thesis", "risk_flags", "entry_conditions",
                "invalidation_conditions", "exit_plan", "confidence"):
        assert key in d, key
    assert d["decision"] in {"BUY", "WAIT", "REJECT"}
    assert 0.0 <= d["confidence"] <= 1.0
    assert d["confidence_meaning"] == "decision_strength_score_not_probability_of_profit"
    assert isinstance(d["risk_flags"], list)
    for f in d["risk_flags"]:
        assert set(f) == {"code", "detail", "severity"}
        assert f["severity"] in ("info", "warn", "block")


def test_confidence_is_never_described_as_probability():
    r = decide([1.0e-4, 1.05e-4, 1.10e-4])
    text = (r.thesis + " " + json.dumps(r.as_dict())).lower()
    for banned in ("probability of profit", "guaranteed", "certain to",
                   "risk-free", "p(profit)"):
        assert banned not in text


def test_invalidation_always_precedes_entry():
    r = decide([1.0e-4, 1.05e-4, 1.10e-4])
    assert len(r.invalidation_conditions) > 0
    plan = r.exit_plan
    assert plan["style"] in {s.value for s in __import__(
        "autopaper.models", fromlist=["ExitStyle"]).ExitStyle}
    assert plan["disaster_cap_pct"] < 0
    assert "invalidation_conditions" in plan
    # every declared invalidation is a string a human wrote before entry
    assert all(isinstance(c, str) and c for c in r.invalidation_conditions)


def test_reject_and_wait_are_distinct():  # noqa: D401
    # the prints are 20s old at decision time: STALE, not LIVE
    stale = decide([1.0e-4, 1.05e-4, 1.10e-4], decide_at=T0 + 20.0)
    assert stale.data_state is DataState.STALE
    assert stale.decision is Decision.WAIT
    # and if it goes far enough, it is refused outright
    unusable = decide([1.0e-4, 1.05e-4, 1.10e-4], decide_at=T0 + 60.0)
    assert unusable.data_state is DataState.UNUSABLE
    assert unusable.decision is Decision.REJECT

    no_data = ENGINE.decide(
        features=build_features(
            candidate_id="c2", mint=MINT, decision_time=T0,
            history=ObservationSet(), alert=None, nar=None,
            stale_after_ms=15_000.0, unusable_after_ms=45_000.0,
        ),
        alert=None, nar=None, analysis_start=T0, analysis_complete=T0,
    )
    assert no_data.decision is Decision.REJECT
    assert any(f.code == "no_market_data" for f in no_data.risk_flags)


# ---------------------------------------------------------------------------
# the anti-exhaustion rule
# ---------------------------------------------------------------------------


def test_fresh_token_is_buyable():
    r = decide([1.0e-4, 1.02e-4, 1.05e-4], first_valid_price=1.0e-4)
    assert r.regime is Regime.VERY_EARLY
    assert r.route in (Route.FAST_ENTRY)
    assert r.decision in (Decision.BUY, Decision.WAIT)


def test_second_wave_entry_is_rejected():
    """The known failure mode: entering far above the first valid price."""
    r = decide([1.0e-4, 1.6e-4, 1.9e-4], first_valid_price=1.0e-4)
    assert r.features["extension_from_first_valid_price"] > 0.8
    assert r.decision is Decision.REJECT
    assert any(
        f.code in ("extended_entry", "exhausted_entry") for f in r.risk_flags
    ), [f.code for f in r.risk_flags]


def test_extreme_extension_is_always_rejected():
    for prices in ([1.0e-5, 5.0e-5, 9.0e-5], [1.0e-6, 1.0e-4, 1.0e-3]):
        r = decide(prices)
        assert r.decision is Decision.REJECT
        assert r.route is Route.NO_TRADE
        assert any(f.severity == "block" for f in r.risk_flags)


def test_accelerating_token_inside_the_chaseable_band_is_tradeable():
    """Momentum is where the runners live; the band is not a blanket ban."""
    r = decide(
        [1.0e-4, 1.22e-4, 1.40e-4],
        liquidity_usd=400_000.0, market_cap_usd=18_000.0,
        buys_5m=420, sells_5m=140, volume_5m_usd=180_000.0,
    )
    assert r.features["extension_from_first_valid_price"] < 0.6
    assert r.regime in (Regime.ACCELERATING, Regime.EXPLOSIVE)
    assert r.decision is Decision.BUY, r.thesis
    assert r.route in (Route.FAST_ENTRY)
    assert r.preferred_exit_style.value in ("RUNNER", "FAST_FAILURE")


def test_extended_but_structurally_strong_may_still_be_taken():
    """
    The ladder has a rungs above the ban. A +70% extension on a deep, liquid,
    well-distributed book with a 3:1 buy ratio is acceleration, not an
    exhausted second wave - and it is the case that produces the big runners.
    It is allowed, but small and fast-failed.

    The market cap sits INSIDE the V3-C ceiling and a NAR snapshot carrying real
    flow evidence is supplied, so V3-C AUTHORIZES on purpose: this test is about
    the extension ladder, and the V3-C gates are exercised separately in
    test_v3c.py and in
    test_extended_strong_is_still_rejected_above_the_v3c_ceiling.
    """
    r = decide(
        [1.0e-4, 1.35e-4, 1.70e-4],
        liquidity_usd=500_000.0, market_cap_usd=19_500.0,
        buys_5m=480, sells_5m=150, volume_5m_usd=260_000.0,
        txns_5m=630, nar=_good_nar(),
    )
    assert r.v3c.authorized, r.v3c.detail
    assert r.features["extension_from_first_valid_price"] > 0.6
    assert r.decision is not Decision.REJECT, r.thesis
    # allowed only on the strictest rung, so it is small and fast-failed
    assert "extended_very_strong" in r.exit_plan["route_modifiers"]
    assert r.preferred_exit_style.value == "FAST_FAILURE"
    # sizing is NOT a route output. It is the frozen V3-C policy, applied to
    # live equity at execution time, so no plan carries a stale percentage.
    assert r.exit_plan["target_position_pct"] is None
    assert any(f.code == "extended_entry" for f in r.risk_flags)


def test_extended_strong_is_still_rejected_above_the_v3c_ceiling():
    """
    The exact same structurally-strong setup is refused once the decision-time
    market cap is above the V3-C ceiling. Nothing about the structure changes;
    the gate is the only thing that differs, and it blocks.
    """
    r = decide(
        [1.0e-4, 1.35e-4, 1.70e-4],
        liquidity_usd=500_000.0, market_cap_usd=1_100_000.0,
        buys_5m=480, sells_5m=150, volume_5m_usd=260_000.0,
        txns_5m=630,
    )
    # the ladder still ran and still liked it - the gate is what refuses
    assert r.features["extension_from_first_valid_price"] > 0.6
    assert r.decision is Decision.REJECT
    flags = {f.code: f for f in r.risk_flags}
    assert "v3c_market_cap_over_ceiling" in flags
    assert flags["v3c_market_cap_over_ceiling"].severity == "block"
    assert "1,100,000.00" in flags["v3c_market_cap_over_ceiling"].detail
    assert "$20,000" in flags["v3c_market_cap_over_ceiling"].detail
    # V3-C is the tighter ceiling, so it is the one that reports; V2-A's wider
    # $25,372.71 would also refuse this, and is still evaluated.
    assert r.v3c.status == "REJECTED_MARKET_CAP"
    assert r.v3c.max_market_cap_usd == 20_000.0
    from autopaper.v2a import apply_v2a
    assert apply_v2a("FAST_ENTRY", 1_100_000.0).rejects


def test_a_token_between_the_two_ceilings_passes_v2a_and_fails_v3c():
    """
    The band that makes V2-A "represented but dominated" concrete: $24,000
    passes V2-A's $25,372.71 and fails V3-C's $20,000.
    """
    from autopaper.v2a import apply_v2a

    r = decide(
        [1.0e-4, 1.35e-4, 1.70e-4],
        liquidity_usd=500_000.0, market_cap_usd=24_000.0,
        buys_5m=480, sells_5m=150, volume_5m_usd=260_000.0, txns_5m=630,
    )
    assert apply_v2a("FAST_ENTRY", 24_000.0).passes, "V2-A passes here"
    assert r.decision is Decision.REJECT, "but V3-C refuses"
    flags = {f.code: f for f in r.risk_flags}
    assert "v3c_market_cap_over_ceiling" in flags
    assert "$24,000.00" in flags["v3c_market_cap_over_ceiling"].detail


def test_extended_but_fragile_is_still_refused():
    """The same extension on a thin book is the failure mode, and is refused."""
    r = decide(
        [1.0e-4, 1.35e-4, 1.70e-4],
        liquidity_usd=14_000.0, market_cap_usd=1_100_000.0,
        buys_5m=180, sells_5m=170, volume_5m_usd=9_000.0,
    )
    assert r.features["extension_from_first_valid_price"] > 0.6
    assert r.decision is Decision.REJECT
    assert r.route is Route.NO_TRADE
    # refused on quality grounds as well as extension - both must be visible
    assert r.signal_quality < DecisionConfig().min_signal_quality_to_trade
    assert any(f.code == "extended_entry" for f in r.risk_flags)


def test_extension_uses_the_first_valid_price_not_the_alert_price():
    hist = series([1.0e-4, 1.02e-4, 1.04e-4])
    a = alert(T0, price_usd=1.04e-4, first_valid_price=1.0e-4)
    f = build_features(
        candidate_id="c", mint=MINT, decision_time=T0, history=hist, alert=a,
        nar=None, stale_after_ms=15_000.0, unusable_after_ms=45_000.0,
    )
    assert f.first_valid_price == pytest.approx(1.0e-4)
    assert f.extension_from_first_valid_price == pytest.approx(0.04)


def test_extension_anchor_is_the_earliest_price_we_could_have_traded():
    hist = series([1.0e-4, 1.02e-4, 1.04e-4])
    f = build_features(
        candidate_id="c", mint=MINT, decision_time=T0, history=hist, alert=None,
        nar=None, stale_after_ms=15_000.0, unusable_after_ms=45_000.0,
    )
    # the oldest print in our own history is a legitimate anchor
    assert f.extension_anchored is True
    assert f.first_valid_price == pytest.approx(1.0e-4)
    assert f.extension_from_first_valid_price == pytest.approx(0.04)


def test_with_no_anchor_extension_is_not_claimed():
    """
    With one print and no alert price there is no earlier price to measure
    from. The agent must say so, and must not award itself a perfect extension
    score off nothing.
    """
    f = build_features(
        candidate_id="c", mint=MINT, decision_time=T0,
        history=ObservationSet([obs(T0, 1.0e-4)]), alert=None, nar=None,
        stale_after_ms=15_000.0, unusable_after_ms=45_000.0,
    )
    assert f.extension_from_first_valid_price is None
    assert f.extension_anchored is False
    assert f.extension_score <= 0.6, "no anchor must not score as a perfect entry"
    # and the single print still forces UNCERTAIN
    choice = select_route(f, DecisionConfig())
    assert choice.route is Route.NO_TRADE


# ---------------------------------------------------------------------------
# weak evidence resolves to WAIT/REJECT
# ---------------------------------------------------------------------------


def test_thin_liquidity_is_blocking():
    r = decide([1.0e-4, 1.05e-4, 1.1e-4], liquidity_usd=2_000.0)
    assert r.decision is Decision.REJECT
    assert any(f.code == "thin_liquidity" and f.severity == "block"
               for f in r.risk_flags)


def test_reversal_is_blocking():
    r = decide([2.0e-4, 1.4e-4, 1.0e-4], first_valid_price=2.0e-4)
    assert r.decision is Decision.REJECT
    assert r.regime is Regime.WEAKENING


def test_uncertain_regime_prefers_no_trade():
    hist = ObservationSet([obs(T0, 1.0e-4)])
    f = build_features(
        candidate_id="c", mint=MINT, decision_time=T0, history=hist, alert=None,
        nar=None, stale_after_ms=15_000.0, unusable_after_ms=45_000.0,
    )
    assert f.regime_hint is None if hasattr(f, "regime_hint") else True
    choice = select_route(f, DecisionConfig())
    assert choice.route is Route.NO_TRADE
    assert "UNCERTAIN" in choice.rationale or "signal quality" in choice.rationale


def test_missing_nar_downgrades_rather_than_invents():
    unavailable = decide([1.0e-4, 1.05e-4, 1.1e-4])
    assert any(f.code == "nar_unavailable" for f in unavailable.risk_flags)
    assert unavailable.nar_used is False
    assert "NAR evidence absent" in unavailable.thesis

    supplied = NarSnapshot(
        mint=MINT, fetched_at=T0, source="nar:test", available=True,
        narrative="a dog on mars", narrative_score=0.8, social_activity=0.6,
        holder_count=412, top10_holders_pct=18.0, dev_holdings_pct=1.0,
        bundler_count=0, sniper_count=1, insider_count=0, security_flags=(),
        wallet_flow_net=1200.0, dex_paid=True,
    )
    good = decide([1.0e-4, 1.05e-4, 1.1e-4], nar=supplied)
    assert good.nar_used is True
    assert not any(f.code == "nar_unavailable" for f in good.risk_flags)
    assert good.confidence >= unavailable.confidence


def test_nar_security_flags_can_block():
    risky = NarSnapshot(
        mint=MINT, fetched_at=T0, source="nar:test", available=True,
        dev_holdings_pct=9.0, top10_holders_pct=71.0, security_flags=("mintable",),
    )
    r = decide([1.0e-4, 1.05e-4, 1.1e-4], nar=risky)
    assert r.decision is Decision.REJECT
    codes = {f.code for f in r.risk_flags}
    assert "dev_supply" in codes
    assert "concentrated_holders" in codes
    assert "nar_mintable" in codes


# ---------------------------------------------------------------------------
# recovery / reclaim gating
# ---------------------------------------------------------------------------
# the removed strategies
# ---------------------------------------------------------------------------


def test_the_removed_entry_strategies_are_not_importable():
    """
    Recovery / Reclaim, Precision, Confirmed Fast Momentum and Early Fast
    Momentum were removed from the runtime, not merely disabled. Their research
    remains under research/ and is never imported here.
    """
    import autopaper.analysis.routes as routes

    for name in (
        "recovery_reclaim_conditions",
        "limits_min_size",
        "classify_for_route",
    ):
        assert not hasattr(routes, name), name
    assert not hasattr(routes.RouteChoice, "target_position_pct")


def test_no_removed_strategy_can_select_a_route():
    """select_route returns FAST_ENTRY or NO_TRADE and nothing else."""
    import autopaper.analysis.routes as routes
    import inspect

    src = inspect.getsource(routes.select_route)
    for name in ("PRECISION_ENTRY", "CONFIRMATION_ENTRY", "RECOVERY_RECLAIM"):
        assert name not in src, name
    # and the only Route member it can name is FAST_ENTRY
    assert "Route.FAST_ENTRY" in src
    assert src.count("Route.") <= 3


# ---------------------------------------------------------------------------


def test_the_only_route_has_one_exit_style_regardless_of_regime():
    """
    FAST_ENTRY is intrinsically a fast-failure setup, so the regime never
    promotes it to a runner. An earlier design derived the exit style per
    regime, which is how a V3-C position ended up with a runner's trailing stop.
    """
    from autopaper.analysis.routes import exit_style_for

    for regime in Regime:
        assert exit_style_for(Route.FAST_ENTRY, regime).value == "FAST_FAILURE"


def test_a_style_override_still_wins():
    from autopaper.analysis.routes import exit_style_for
    from autopaper.models import ExitStyle

    assert exit_style_for(
        Route.FAST_ENTRY, Regime.ACCELERATING, ExitStyle.RUNNER
    ) is ExitStyle.RUNNER


def test_no_trade_gets_a_time_decay_style():
    from autopaper.analysis.routes import exit_style_for

    assert exit_style_for(Route.NO_TRADE, Regime.EARLY).value == "TIME_DECAY"


def test_every_style_used_is_known_to_the_exit_policy():
    from autopaper.models import ExitStyle

    assert len(list(ExitStyle)) >= 7


# ---------------------------------------------------------------------------
# alert-reported supply metrics
# ---------------------------------------------------------------------------


def _alert_with(**fields):
    from .conftest import alert as mk_alert

    return mk_alert(T0, **fields)


def test_alert_supply_metrics_become_risk_flags():
    """The bot reports holders, top-10, dev supply and bundles. Use them."""
    r = decide(
        [1.0e-4, 1.02e-4, 1.05e-4],
        alert_obj=_alert_with(
            price_usd=1.05e-4,
            holder_count=12,
            top10_pct=71.0,
            dev_holdings_pct=9.5,
            bundle_count=12,
            bundled_pct=64.0,
            dex_paid=True,
        ),
    )
    codes = {f.code for f in r.risk_flags}
    for expected in ("thin_holder_base", "concentrated_holders", "dev_supply",
                     "bundling", "bundled_supply", "dex_paid"):
        assert expected in codes, (expected, sorted(codes))
    assert r.decision is Decision.REJECT


def test_clean_supply_metrics_do_not_block():
    r = decide(
        [1.0e-4, 1.02e-4, 1.05e-4],
        alert_obj=_alert_with(
            price_usd=1.05e-4, holder_count=880, top10_pct=18.0,
            dev_holdings_pct=0.0, bundle_count=1, dex_paid=False,
        ),
    )
    codes = {f.code for f in r.risk_flags}
    assert "concentrated_holders" not in codes
    assert "dev_supply" not in codes
    assert "thin_holder_base" not in codes
    assert r.decision is Decision.BUY, r.thesis


def test_absent_alert_metrics_stay_absent_not_assumed_safe():
    r = decide([1.0e-4, 1.02e-4, 1.05e-4], alert_obj=_alert_with(price_usd=1.05e-4))
    codes = {f.code for f in r.risk_flags}
    for expected in ("thin_holder_base", "concentrated_holders", "dev_supply",
                     "bundling", "dex_paid"):
        assert expected not in codes, (
            f"{expected} was raised for a metric the alert never reported"
        )
