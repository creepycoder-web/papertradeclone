"""
V2_FLOOR: the second selectable entry strategy.

WHAT IS ASSERTED HERE, AND WHY IT IS ASSERTED HERE
--------------------------------------------------

The gate itself (:class:`autopaper.v2floor.V2FloorResult`) is tested as a pure
function, at its boundaries, because a threshold that is off by one on an
inclusive bound produces a result indistinguishable from a working one while
trading a different strategy.

The gate's WIRING is tested end-to-end through ``DecisionEngine``, because the
failure this architecture is built to prevent is not a bad threshold -- it is a
gate that authorizes something the operator did not select, or a strategy that
quietly inherits filters it was measured without.

The three load-bearing wiring facts:

1. ``V2_FLOOR`` authorizes through its OWN gate, and NOT through L25. If it ever
   authorized through L25 it would silently inherit L25's $20,000 market-cap
   ceiling and its 0.9647 liquidity/MC floor -- the two filters the research
   explicitly tested and rejected -- and would trade a different strategy from
   the one that was measured.
2. Selecting ``V2_FLOOR`` changes WHICH gate authorizes. It does not weaken any
   structural, routing or safety check, and it does not arm REAL.
3. Under the L25 family, behaviour is unchanged. Every L25 assertion elsewhere
   in the suite is the regression guard for this.
"""

from __future__ import annotations

import json

import pytest

from autopaper.analysis import DecisionEngine, build_features
from autopaper.config import DecisionConfig, ExitConfig
from autopaper.exits import ExitPolicy
from autopaper.models import DataState, ExitStyle, Decision
from autopaper.nar import NullNarSource
from autopaper.strategy_profile import (
    PROFILE_ENTRY_STRATEGY,
    StrategyProfile,
    V2F_EXIT_PRECEDENCE,
    V2F_STOP_LOSS_PCT,
    V2F_TAKE_PROFIT_PCT,
    V2F_TRIGGER_MAX_HOLD,
    V2F_TRIGGER_STOP_LOSS,
    V2F_TRIGGER_TAKE_PROFIT,
    entry_strategy,
    parse_profile,
    uses_l25_entry,
)
from autopaper.v2floor import (
    AUTHORIZED,
    REJECTED_CHANGE_5M_ABOVE_CEILING,
    REJECTED_CHANGE_5M_BELOW_FLOOR,
    REJECTED_VOL5_PER_MC_BELOW_FLOOR,
    UNTESTABLE,
    V2F_MAX_CHANGE_5M_PCT,
    V2F_MIN_CHANGE_5M_PCT,
    V2F_MIN_VOL5_PER_MC,
    V2FloorResult,
    evaluate,
    f_vol5_per_mc,
    flag_code,
    rule_summary,
    verify_definition,
)

from .conftest import MINT, alert, obs

pytestmark = [pytest.mark.strategy, pytest.mark.paper_live]

T0 = 1_700_000_000.0


class _FakePos:
    """
    What ``ExitPolicy.assess`` and ``_max_hold_for`` read, and nothing else.

    Deliberately a bare object rather than a full ``Position``: these tests are
    about the exit policy's decision given a profile and a hold time, and a real
    Position carries ~90 fields the policy never looks at.
    """

    def __init__(self, entry_timestamp, *, fill_price=1.0,
                 strategy_profile="L25_CURRENT", peak_return_pct=0.0,
                 exit_style=None):
        from autopaper.models import ExitStyle

        self.position_id = "pos_v2f_test"
        self.entry_timestamp = entry_timestamp
        self.fill_price = fill_price
        self.strategy_profile = strategy_profile
        self.peak_return_pct = peak_return_pct
        self.exit_style = exit_style or ExitStyle.FAST_FAILURE
        self.runner_mode = False
        self.exit_hold_count = 0

    def return_at(self, price):
        if not price:
            return None
        return (price - self.fill_price) / self.fill_price


def eng(profile: str = "V2_FLOOR") -> DecisionEngine:
    return DecisionEngine(
        DecisionConfig(strategy_profile=profile), ExitConfig()
    )


# ===========================================================================
# 1. THE GATE -- pure function, boundary-exact
# ===========================================================================


class TestGateBoundaries:
    """
    Every boundary the research specified, asserted on both sides.

    ``vol5_per_mc`` is built from a volume and a market cap so the ratio is
    exact rather than a rounded literal.
    """

    @staticmethod
    def ok(**kw):
        """A passing candidate; override exactly one input per test."""
        base = dict(
            volume_5m_usd=30_000.0,
            market_cap_usd=20_000.0,      # ratio exactly 1.5
            change_5m_pct=50.0,
        )
        base.update(kw)
        return evaluate(**base)

    # -- 1. vol5/mc below the floor -> reject ---------------------------
    def test_vol5_per_mc_below_one_rejects(self):
        r = self.ok(volume_5m_usd=19_999.0, market_cap_usd=20_000.0)
        assert r.authorized is False
        assert r.status == REJECTED_VOL5_PER_MC_BELOW_FLOOR
        assert flag_code(r) == "v2floor_vol5_per_mc_below_floor"
        assert r.vol5_per_mc == pytest.approx(0.99995)

    # -- 2. vol5/mc exactly one -> ACCEPT -------------------------------
    def test_vol5_per_mc_exactly_one_accepts(self):
        r = self.ok(volume_5m_usd=20_000.0, market_cap_usd=20_000.0)
        assert r.authorized is True
        assert r.status == AUTHORIZED
        assert r.vol5_per_mc == pytest.approx(1.0)
        # and the momentum condition still had to pass
        assert r.change_5m_pct == 50.0

    def test_vol5_per_mc_exactly_one_rejects_if_momentum_fails(self):
        """Passing one condition is not enough: all three must hold."""
        r = self.ok(volume_5m_usd=20_000.0, market_cap_usd=20_000.0,
                    change_5m_pct=5.0)
        assert r.authorized is False
        assert r.status == REJECTED_CHANGE_5M_BELOW_FLOOR

    # -- 3. t_c5 below the floor -> reject -------------------------------
    def test_change_5m_below_ten_rejects(self):
        r = self.ok(change_5m_pct=9.99)
        assert r.authorized is False
        assert r.status == REJECTED_CHANGE_5M_BELOW_FLOOR
        assert flag_code(r) == "v2floor_change_5m_below_floor"

    def test_negative_change_5m_rejects(self):
        """A token already FALLING is the regime the floor exists to remove."""
        r = self.ok(change_5m_pct=-40.0)
        assert r.authorized is False
        assert r.status == REJECTED_CHANGE_5M_BELOW_FLOOR

    def test_zero_change_5m_rejects(self):
        assert self.ok(change_5m_pct=0.0).authorized is False

    # -- 4. t_c5 exactly at the floor -> ACCEPT --------------------------
    def test_change_5m_exactly_ten_accepts(self):
        r = self.ok(change_5m_pct=10.0)
        assert r.authorized is True
        assert r.status == AUTHORIZED
        assert r.change_5m_pct == 10.0

    # -- 5. t_c5 above the ceiling -> reject -----------------------------
    def test_change_5m_above_three_hundred_rejects(self):
        r = self.ok(change_5m_pct=300.01)
        assert r.authorized is False
        assert r.status == REJECTED_CHANGE_5M_ABOVE_CEILING
        assert flag_code(r) == "v2floor_change_5m_above_ceiling"

    def test_extreme_change_5m_rejects(self):
        assert self.ok(change_5m_pct=571.0).authorized is False

    # -- 6. t_c5 exactly at the ceiling -> ACCEPT -------------------------
    def test_change_5m_exactly_three_hundred_accepts(self):
        r = self.ok(change_5m_pct=300.0)
        assert r.authorized is True
        assert r.status == AUTHORIZED

    # -- 7. all three pass -> authorized ---------------------------------
    def test_all_three_conditions_passing_authorizes(self):
        r = self.ok(volume_5m_usd=50_000.0, market_cap_usd=20_000.0,
                    change_5m_pct=120.0)
        assert r.authorized is True
        assert r.status == AUTHORIZED
        assert r.vol5_per_mc == pytest.approx(2.5)
        assert "V2_FLOOR" not in r.detail or "authorized" not in r.detail.lower()
        # the detail names both thresholds and the value, so a stored refusal
        # or authorization is readable without the source
        assert "2.5" in r.detail
        assert "+120.0%" in r.detail

    def test_result_is_serialisable_and_complete(self):
        payload = json.dumps(self.ok().as_dict())
        back = json.loads(payload)
        for key in ("strategy", "authorized", "status", "vol5_per_mc",
                    "change_5m_pct", "min_vol5_per_mc", "min_change_5m_pct",
                    "max_change_5m_pct", "detail"):
            assert key in back, key
        assert back["strategy"] == "V2_FLOOR"


class TestGateFailsClosed:
    """
    An unreadable input refuses. It never passes, and never substitutes.

    This is the failure documented in ``paper_live/arms.py::_ratio_of``: an
    earlier rule read a missing field as "not applicable" and refused every
    signal for a reason unrelated to its hypothesis. Here an absent value is
    UNTESTABLE and UNTESTABLE does not authorize.
    """

    def test_an_authorized_result_is_never_given_a_refusal_code(self):
        """
        Regression guard for a real trap in ``flag_code``.

        The function returned ``v2floor_not_authorized`` on the fallthrough, so
        calling it on an AUTHORIZED result produced a code that reads exactly
        like a refusal. Every current caller only flags refusals, so nothing
        was broken in production -- but a caller that logs unconditionally
        would have written "v2floor_not_authorized" into the journal for a
        trade that was taken. The only way to catch that later is to assert it
        now.
        """
        r = evaluate(volume_5m_usd=30_000.0, market_cap_usd=20_000.0,
                     change_5m_pct=50.0)
        assert r.authorized is True
        assert r.rejects is False
        code = flag_code(r)
        assert code == "v2floor_authorized", code
        assert "not_authorized" not in code, (
            "an authorized result must not read like a refusal"
        )

    @pytest.mark.parametrize("volume", [None, 0.0, -1.0])
    def test_missing_or_non_positive_volume_is_untestable(self, volume):
        r = evaluate(volume_5m_usd=volume, market_cap_usd=20_000.0,
                     change_5m_pct=50.0)
        assert r.authorized is False
        assert r.status == UNTESTABLE
        assert flag_code(r) == "v2floor_untestable"

    @pytest.mark.parametrize("mcap", [None, 0.0, -1.0])
    def test_missing_or_non_positive_market_cap_is_untestable(self, mcap):
        r = evaluate(volume_5m_usd=30_000.0, market_cap_usd=mcap,
                     change_5m_pct=50.0)
        assert r.authorized is False
        assert r.status == UNTESTABLE

    @pytest.mark.parametrize("change", [None, float("nan"), float("inf")])
    def test_missing_or_non_finite_change_is_untestable(self, change):
        r = evaluate(volume_5m_usd=30_000.0, market_cap_usd=20_000.0,
                     change_5m_pct=change)
        assert r.authorized is False
        assert r.status == UNTESTABLE

    def test_everything_missing_is_untestable_not_authorized(self):
        r = evaluate(volume_5m_usd=None, market_cap_usd=None,
                     change_5m_pct=None)
        assert r.authorized is False
        assert r.status == UNTESTABLE

    def test_untestable_detail_names_the_missing_input(self):
        """A refusal has to say WHICH input it could not read."""
        r = evaluate(volume_5m_usd=None, market_cap_usd=20_000.0,
                     change_5m_pct=50.0)
        assert "volume_5m_usd" in r.detail
        assert "market_cap_usd" not in r.detail

    def test_ratio_helper_never_substitutes_a_default(self):
        assert f_vol5_per_mc(None, 20_000.0) is None
        assert f_vol5_per_mc(30_000.0, None) is None
        assert f_vol5_per_mc(0.0, 20_000.0) is None
        assert f_vol5_per_mc(30_000.0, 0.0) is None
        assert f_vol5_per_mc(float("nan"), 20_000.0) is None
        assert f_vol5_per_mc(30_000.0, float("inf")) is None
        assert f_vol5_per_mc(30_000.0, 20_000.0) == pytest.approx(1.5)


class TestThresholdsAreFrozen:
    def test_thresholds_match_the_research_rule(self):
        assert V2F_MIN_VOL5_PER_MC == 1.0
        assert V2F_MIN_CHANGE_5M_PCT == 10.0
        assert V2F_MAX_CHANGE_5M_PCT == 300.0

    def test_verify_definition_accepts_the_frozen_values(self):
        verify_definition()          # the defaults must verify

    @pytest.mark.parametrize("bad", [
        dict(min_vol5_per_mc=0.9),
        dict(min_change_5m_pct=0.0),
        dict(max_change_5m_pct=250.0),
    ])
    def test_verify_definition_refuses_any_other_value(self, bad):
        """Raised, not warned: a quiet threshold change is the failure."""
        with pytest.raises(ValueError):
            verify_definition(**bad)

    def test_evaluate_refuses_a_widened_threshold(self):
        """The gate cannot be talked into a looser rule through its kwargs."""
        with pytest.raises(ValueError):
            evaluate(volume_5m_usd=1.0, market_cap_usd=1_000.0,
                     change_5m_pct=0.0, min_vol5_per_mc=0.0001)

    def test_rule_summary_states_the_whole_rule(self):
        s = rule_summary()
        assert ">= 1" in s
        assert "+10%" in s
        assert "+300%" in s


# ===========================================================================
# 2. STRATEGY SELECTION
# ===========================================================================


class TestSelection:
    def test_v2_floor_is_in_the_closed_profile_set(self):
        assert parse_profile("V2_FLOOR") is StrategyProfile.V2_FLOOR
        assert parse_profile("v2_floor") is StrategyProfile.V2_FLOOR   # case
        assert parse_profile("  V2_FLOOR  ") is StrategyProfile.V2_FLOOR
        assert "V2_FLOOR" in {
            p.value for p in StrategyProfile
        }

    def test_it_has_its_own_entry_strategy(self):
        assert entry_strategy("V2_FLOOR") == "V2_FLOOR"
        assert PROFILE_ENTRY_STRATEGY[StrategyProfile.V2_FLOOR] == "V2_FLOOR"
        assert uses_l25_entry("V2_FLOOR") is False

    def test_the_l25_family_still_uses_l25(self):
        for name in ("L25_CURRENT", "L25_60S", "L25_MAX_HOLD"):
            assert entry_strategy(name) == "L25", name
            assert uses_l25_entry(name) is True

    def test_the_default_is_unchanged(self):
        """An operator who never touches the selector is unaffected."""
        assert parse_profile(None) is StrategyProfile.L25_CURRENT
        assert parse_profile("") is StrategyProfile.L25_CURRENT
        assert DecisionConfig().strategy_profile == "L25_CURRENT"

    def test_config_refuses_an_unknown_profile(self):
        with pytest.raises(ValueError):
            DecisionConfig(strategy_profile="V2_FLOOR_TYPO")

    def test_a_typo_does_not_silently_run_l25(self):
        """Fails closed: the caller cannot start on an undefined profile."""
        from autopaper.strategy_profile import StrategyProfileError
        with pytest.raises(StrategyProfileError):
            parse_profile("V2FLOOR")

    def test_the_operator_can_select_it_without_arming_anything(self):
        from autopaper.modes import RuntimeState
        state = RuntimeState()
        before = (state.run_state, state.real_armed, state.emergency_stop,
                  state.entries_blocked)
        accepted, message = state.select_strategy_profile("V2_FLOOR")
        assert accepted is True
        assert state.strategy_profile == "V2_FLOOR"
        assert state.profile is StrategyProfile.V2_FLOOR
        # selecting a strategy is not a trading control
        assert (state.run_state, state.real_armed, state.emergency_stop,
                state.entries_blocked) == before
        assert "arm" in message.lower() and "not" in message.lower()

    def test_real_armed_cannot_be_set_by_a_constructor(self):
        from autopaper.modes import RuntimeState
        with pytest.raises((TypeError, ValueError, AttributeError)):
            RuntimeState(real_armed=True)


# ===========================================================================
# 3. THE GATE IN THE DECISION PATH
# ===========================================================================


def _features(*, volume_5m_usd=50_000.0, market_cap_usd=20_000.0,
              liquidity_usd=250_000.0, change_5m_pct=50.0, decide_at=None,
              t0=T0):
    """A structurally tradeable candidate carrying V2_FLOOR's inputs.

    ``volume_5m_usd`` and ``price_change_5m`` ride on the OBSERVATION, so the
    feature builder reads them through the same ``best()`` chain a real venue
    print uses rather than by injection. ``price_change_5m`` is the venue's own
    reported trailing-5m change -- the runtime equivalent of the research's
    ``t_c5`` -- and ``volume_5m_usd / market_cap_usd`` is the runtime
    equivalent of ``vol5_per_mc``.
    """
    prices = [1.0e-4, 1.02e-4, 1.05e-4]
    step = 15.0
    hist_series = [
        obs(t0 - step * (len(prices) - 1 - i), p,
            market_cap_usd=market_cap_usd,
            liquidity_usd=liquidity_usd,
            volume_5m_usd=volume_5m_usd,
            price_change_5m=change_5m_pct)
        for i, p in enumerate(prices)
    ]
    from autopaper.models import ObservationSet

    hist = ObservationSet(hist_series)
    when = t0 if decide_at is None else decide_at
    a = alert(t0, price_usd=prices[-1])
    nar = NullNarSource().fetch(MINT)
    return build_features(
        candidate_id="c1", mint=MINT, decision_time=when, history=hist,
        alert=a, nar=nar, stale_after_ms=15_000.0, unusable_after_ms=45_000.0,
    ), a, nar, when


def _decide(profile="V2_FLOOR", **kw):
    f, a, nar, when = _features(**kw)
    return eng(profile).decide(
        features=f, alert=a, nar=nar, analysis_start=when,
        analysis_complete=when + 0.05,
    )


def _blockers(report) -> list[str]:
    return [f.code for f in report.risk_flags if f.severity == "block"]


class TestDecisionWiring:
    def test_the_selected_gate_is_the_one_that_authorizes(self):
        r = _decide("V2_FLOOR")
        assert r.strategy_profile == "V2_FLOOR"
        assert r.v2floor is not None
        assert r.selected_strategy_result is r.v2floor
        assert r.strategy_status == r.v2floor.status

    def test_a_v2_floor_rejection_reaches_the_decision(self):
        """A volume-floor refusal must be a REJECT, not a WAIT or a BUY."""
        r = _decide("V2_FLOOR", volume_5m_usd=1_000.0,
                    market_cap_usd=20_000.0)     # ratio 0.05
        assert r.decision is Decision.REJECT
        assert "v2floor_vol5_per_mc_below_floor" in _blockers(r)

    def test_a_momentum_floor_rejection_reaches_the_decision(self):
        """A token already falling at the alert is refused."""
        r = _decide("V2_FLOOR", change_5m_pct=-25.0)
        assert r.decision is Decision.REJECT
        assert "v2floor_change_5m_below_floor" in _blockers(r)

    def test_the_ceiling_rejection_reaches_the_decision(self):
        r = _decide("V2_FLOOR", change_5m_pct=450.0)
        assert r.decision is Decision.REJECT
        assert "v2floor_change_5m_above_ceiling" in _blockers(r)

    def test_an_untestable_input_refuses_rather_than_buying(self):
        """
        A missing volume_5m must not become a pass.

        This is the failure documented in paper_live/arms.py::_ratio_of, where an
        absent field read as "not applicable" produced a dataset that was empty
        for a reason unrelated to the hypothesis.
        """
        r = _decide("V2_FLOOR", volume_5m_usd=None)
        assert r.decision is Decision.REJECT
        assert "v2floor_untestable" in _blockers(r)

    def test_the_l25_refusal_is_journaled_but_never_blocks_v2_floor(self):
        """
        The control A/B needs "would L25 have taken this?".

        Both gates are always evaluated, so a V2_FLOOR record can answer that.
        But the L25 refusal must be at ``info``: if it were blocking, V2_FLOOR
        would silently be L25-plus-a-momentum-floor, which is a different
        strategy from the one that was measured.
        """
        # market cap $20,000 with liquidity $250,000 gives a comfortable
        # liq/mcap; force V3-C's signal-quality floor to refuse instead
        r = _decide("V2_FLOOR", market_cap_usd=20_000.0,
                    liquidity_usd=250_000.0)
        codes = {f.code: f.severity for f in r.risk_flags}
        if "v3c_" in " ".join(codes):
            for code, severity in codes.items():
                if code.startswith(("v3c_", "l25_", "v2a_")):
                    assert severity == "info", (
                        f"{code} is an L25 sub-result and must not block a "
                        f"V2_FLOOR decision; it would reinstate the market-cap "
                        f"and liquidity filters the research rejected"
                    )

    def test_v2a_never_blocks_v2_floor(self):
        """V2-A is an L25 sub-result: its ceiling is an L25 market-cap cap."""
        # $30,000 market cap clears V3-C's $20,000 but breaches V2-A's
        # $25,372.71 -- the exact case that must not block V2_FLOOR.
        r = _decide("V2_FLOOR", market_cap_usd=30_000.0, change_5m_pct=50.0)
        for f in r.risk_flags:
            if f.code.startswith("v2a_"):
                assert f.severity == "info", f.detail
            assert f.code not in _blockers(r)

    def test_under_l25_current_the_v2_gate_never_blocks(self):
        """
        The mirror image.

        V2_FLOOR's refusal must be informational under the L25 family, so the
        default runtime is byte-identical to before this feature existed.
        """
        # a candidate that V2_FLOOR refuses: vol5/mcap far below 1
        r = _decide("L25_CURRENT", volume_5m_usd=100.0,
                    market_cap_usd=20_000.0)
        for f in r.risk_flags:
            if f.code.startswith("v2floor_"):
                assert f.severity == "info", f.detail
                assert "not the selected gate" in f.detail

    def test_the_report_records_the_profile_it_was_made_under(self):
        r = _decide("L25_60S")
        assert r.strategy_profile == "L25_60S"
        d = r.to_json_dict()
        assert json.dumps(d)          # must stay serialisable


class TestStructuralGatesStillApply:
    """
    V2_FLOOR changes WHICH strategy gate authorizes. It must not weaken the
    shared structural refusals in ``analysis/routes.py``.
    """

    def test_data_past_the_unusable_window_is_a_reject(self):
        """
        Beyond ``unusable_after_ms`` the print is not a reading at all, so this
        is a structural refusal rather than a wait.
        """
        r = _decide("V2_FLOOR", decide_at=T0 + 60.0, change_5m_pct=50.0)
        assert r.data_state is DataState.UNUSABLE
        assert r.decision is Decision.REJECT

    def test_a_stale_decision_is_a_wait_not_a_buy(self):
        """
        Inside the STALE band (15s..45s) the answer is WAIT, not REJECT.

        "We could not look properly" is not "we looked and it is not good
        enough": a genuinely good setup judged on a 20s-old print should be
        looked at again, not permanently refused.
        """
        r = _decide("V2_FLOOR", decide_at=T0 + 20.0, change_5m_pct=50.0)
        assert r.data_state is DataState.STALE
        assert r.decision is not Decision.BUY

    def test_the_existing_thin_liquidity_refusal_still_applies(self):
        f, a, nar, when = _features(liquidity_usd=100.0)
        r = eng("V2_FLOOR").decide(
            features=f, alert=a, nar=nar, analysis_start=when,
            analysis_complete=when + 0.05,
        )
        assert r.decision is Decision.REJECT
        assert any("liquidity" in f.detail for f in r.risk_flags)

    def test_the_existing_market_cap_sanity_band_still_applies(self):
        f, a, nar, when = _features(market_cap_usd=200_000_000.0,
                                    change_5m_pct=50.0)
        r = eng("V2_FLOOR").decide(
            features=f, alert=a, nar=nar, analysis_start=when,
            analysis_complete=when + 0.05,
        )
        assert r.decision is Decision.REJECT
        assert any("market cap" in f.detail for f in r.risk_flags)

    def test_the_stale_structural_classification_learns_the_v2_prefix(self):
        """
        A strategy refusal on a STALE print is a WAIT, not a REJECT -- the token
        may be fine on a fresh print. That only works if the refusal code is in
        the known non-structural prefixes, or a stale good-looking setup would
        be permanently rejected.
        """
        f, a, nar, when = _features(decide_at=T0 + 20.0)
        assert f.data_state is DataState.STALE
        r = eng("V2_FLOOR").decide(
            features=f, alert=a, nar=nar, analysis_start=when,
            analysis_complete=when + 0.05,
        )
        if _blockers(r) and all(
            c.startswith(("v3c_", "v2a_", "l25_", "v2floor_"))
            for c in _blockers(r)
        ):
            assert r.decision is Decision.WAIT, (
                "a strategy-only refusal on stale data must be a WAIT"
            )


class TestEntryConditionsAreRecorded:
    def test_the_recorded_conditions_name_the_v2_floor_rule(self):
        r = _decide("V2_FLOOR", volume_5m_usd=5_000_000.0,
                    market_cap_usd=20_000.0, change_5m_pct=50.0)
        text = "\n".join(r.entry_conditions)
        assert "V2_FLOOR" in text
        assert ">= 1" in text
        assert "+10%" in text and "+300%" in text

    def test_the_recorded_conditions_state_the_absent_filters(self):
        """
        A stored record must never imply a filter that was not enforced.

        V2_FLOOR has no market-cap ceiling and no liquidity floor; saying so
        explicitly is what stops a reader assuming L25's were inherited.
        """
        r = _decide("V2_FLOOR", volume_5m_usd=5_000_000.0,
                    market_cap_usd=20_000.0, change_5m_pct=50.0)
        text = "\n".join(r.entry_conditions).lower()
        assert "no market-cap ceiling" in text
        assert "no liquidity/mc floor" in text

    def test_l25_records_are_unchanged(self):
        r = _decide("L25_CURRENT", volume_5m_usd=5_000_000.0,
                    market_cap_usd=15_000.0, change_5m_pct=50.0)
        text = "\n".join(r.entry_conditions)
        assert "L25 authorized" in text
        assert "f_liq_to_mcap >= 0.9647" in text
        assert "market cap <= $20,000" in text


class TestResearchedExitBracket:
    """
    The researched V2_FLOOR exit rule, asserted at every boundary:

        TP = +30%   SL = -30%   MAX HOLD = 90s

        if ret <= -30%:      SELL
        elif ret >= +30%:    SELL
        elif held >= 90s:    SELL
        else:                HOLD

    The thresholds are INCLUSIVE at the boundary: -30.0% sells and -29.9% holds,
    +30.0% sells and +29.9% holds. An off-by-one here is a silent change to a
    researched rule, so both sides of both boundaries are pinned.
    """

    @staticmethod
    def _policy():
        from autopaper.exits import ExitPolicy
        return ExitPolicy(ExitConfig())

    @staticmethod
    def _at(price, held=10.0, *, profile="V2_FLOOR", **kw):
        args = dict(
            data_state=DataState.LIVE, momentum_30s_pct=0.0,
            structure_30s_pct=0.0, liquidity_change_pct=0.0,
        )
        args.update(kw)
        return TestResearchedExitBracket._policy().assess(
            _FakePos(1_000.0, strategy_profile=profile),
            now=1_000.0 + held, mark_price=price, **args,
        )

    # -- the numbers, as stated ----

    def test_the_stop_is_minus_thirty_percent(self):
        assert V2F_STOP_LOSS_PCT == -30.0

    def test_the_take_profit_is_plus_thirty_percent(self):
        assert V2F_TAKE_PROFIT_PCT == 30.0

    def test_the_cap_is_ninety_seconds(self):
        from autopaper.strategy_profile import max_hold_s
        assert max_hold_s("V2_FLOOR") == 90.0

    # -- stop-loss boundary ----

    def test_minus_29_percent_holds(self):
        d = self._at(0.71)
        assert d.action == "HOLD", d.reason
        assert d.trigger is None

    def test_minus_30_percent_sells(self):
        d = self._at(0.70)
        assert d.action == "EXIT"
        assert d.trigger == V2F_TRIGGER_STOP_LOSS

    def test_minus_31_percent_sells(self):
        assert self._at(0.69).trigger == V2F_TRIGGER_STOP_LOSS

    # -- take-profit boundary ----

    def test_plus_29_percent_holds(self):
        d = self._at(1.29)
        assert d.action == "HOLD", d.reason
        assert d.trigger is None

    def test_plus_30_percent_sells(self):
        d = self._at(1.30)
        assert d.action == "EXIT"
        assert d.trigger == V2F_TRIGGER_TAKE_PROFIT

    def test_plus_31_percent_sells(self):
        assert self._at(1.31).trigger == V2F_TRIGGER_TAKE_PROFIT

    # -- cap boundary ----

    def test_89_seconds_holds(self):
        d = self._at(1.05, held=89.0)
        assert d.action == "HOLD", d.reason
        assert d.trigger is None

    def test_90_seconds_sells(self):
        d = self._at(1.05, held=90.0)
        assert d.action == "EXIT"
        assert d.trigger == V2F_TRIGGER_MAX_HOLD

    def test_the_cap_is_evaluated_on_a_tick_not_a_sleep(self):
        """
        No timer, no polling cadence of its own: ``held >= 90`` on an assessment.

        89.999s holds and 90.0s sells, which is only possible if the cap is read
        off the clock at evaluation time.
        """
        assert self._at(1.05, held=89.999).action == "HOLD"
        assert self._at(1.05, held=90.0).action == "EXIT"

    # -- the reason names the researched rule ----

    def test_each_exit_reason_names_its_own_researched_level(self):
        assert "-30%" in self._at(0.70).reason
        assert "+30%" in self._at(1.30).reason
        assert "90s" in self._at(1.05, held=90.0).reason

    def test_the_exit_style_is_the_bracket_not_a_borrowed_l25_style(self):
        """
        The journal must not report a V2 exit as an L25-style exit.

        Every V2 decision carries ``V2_BRACKET``, including the HOLDs, so a
        reader can tell from ``exit_style`` alone that the shared stack was
        bypassed rather than merely not firing.
        """
        assert self._at(1.05).style is ExitStyle.V2_BRACKET
        assert self._at(1.30).style is ExitStyle.V2_BRACKET
        assert self._at(0.70).style is ExitStyle.V2_BRACKET
        assert self._at(1.05, held=90.0).style is ExitStyle.V2_BRACKET

    def test_the_cap_is_not_reachable_through_the_l25_override(self):
        """The 90s is fixed; the L25_MAX_HOLD tuner cannot move it."""
        from autopaper.strategy_profile import set_max_hold_s
        set_max_hold_s(120.0)
        try:
            from autopaper.strategy_profile import max_hold_s
            assert max_hold_s("V2_FLOOR") == 90.0
        finally:
            set_max_hold_s(None)

    def test_the_cap_is_read_off_the_position_not_a_global(self):
        """A mid-flight profile switch must not rewrite an open position."""
        from autopaper.exits import _max_hold_for
        from autopaper.modes import RuntimeState

        opened_as_v2 = _FakePos(1_000.0, strategy_profile="V2_FLOOR")
        assert _max_hold_for(opened_as_v2) == 90.0
        state = RuntimeState()
        state.select_strategy_profile("L25_CURRENT")
        assert state.strategy_profile == "L25_CURRENT"
        assert _max_hold_for(opened_as_v2) == 90.0, (
            "an open position keeps the profile it was opened under"
        )


class TestExitPrecedence:
    """
    When more than one researched exit is true on the same mark, exactly one
    fires and it is the right one.

    The order is SL, then TP, then the cap. SL and TP are mutually exclusive on
    a single mark so their relative order cannot change an outcome -- which is
    itself worth asserting, because it means the cap is the only genuinely
    contested precedence in this bracket.
    """

    @staticmethod
    def _at(price, held):
        return ExitPolicy(ExitConfig()).assess(
            _FakePos(1_000.0, strategy_profile="V2_FLOOR"),
            now=1_000.0 + held, mark_price=price, data_state=DataState.LIVE,
            momentum_30s_pct=0.0, structure_30s_pct=0.0,
            liquidity_change_pct=0.0,
        )

    def test_the_declared_order_is_stop_then_profit_then_cap(self):
        assert V2F_EXIT_PRECEDENCE == (
            V2F_TRIGGER_STOP_LOSS,
            V2F_TRIGGER_TAKE_PROFIT,
            V2F_TRIGGER_MAX_HOLD,
        )

    def test_stop_and_profit_are_mutually_exclusive(self):
        """
        A single mark cannot be both, so their order is not a real choice.

        If this ever failed, the "precedence" would be doing real work and would
        need to be argued about rather than merely asserted.
        """
        assert not (V2F_STOP_LOSS_PCT <= 30.0 and V2F_TAKE_PROFIT_PCT >= -30.0
                    and V2F_STOP_LOSS_PCT > V2F_TAKE_PROFIT_PCT)

    def test_a_stop_past_the_cap_reports_the_stop(self):
        """
        -40% at 200s: the cap has long since passed, but the STOP is the honest
        answer to "why did this close", so the stop is what is reported.
        """
        d = self._at(0.60, 200.0)
        assert d.action == "EXIT"
        assert d.trigger == V2F_TRIGGER_STOP_LOSS
        assert d.trigger != V2F_TRIGGER_MAX_HOLD

    def test_a_profit_past_the_cap_reports_the_profit(self):
        """+35% at 200s reports the take-profit, not the cap."""
        d = self._at(1.35, 200.0)
        assert d.action == "EXIT"
        assert d.trigger == V2F_TRIGGER_TAKE_PROFIT

    def test_the_cap_fires_only_when_neither_level_was_reached(self):
        """+5% at 200s: the only researched exit that is true is the cap."""
        d = self._at(1.05, 200.0)
        assert d.action == "EXIT"
        assert d.trigger == V2F_TRIGGER_MAX_HOLD
        assert "neither" in d.reason

    def test_exactly_one_trigger_ever_fires(self):
        """
        Sweep the whole plane. Every (price, held) pair resolves to exactly one
        outcome, so there is no input for which two rules contend and the
        reader cannot tell which one decided.
        """
        for pct in (-95, -60, -30.0, -29.9, -1, 0, 1, 29.9, 30.0, 45, 120):
            for held in (0, 30, 89.9, 90.0, 300):
                d = self._at(1.0 + pct / 100.0, held)
                assert d.trigger in (
                    None, V2F_TRIGGER_STOP_LOSS, V2F_TRIGGER_TAKE_PROFIT,
                    V2F_TRIGGER_MAX_HOLD,
                ), (pct, held, d.reason)
                if d.action == "HOLD":
                    assert d.trigger is None, (pct, held)
                else:
                    assert d.trigger is not None, (pct, held)


class TestSharedExitIsolation:
    """
    THE POINT OF THE BRACKET: a V2 position is not closed by an L25 rule.

    Each of these is a condition the SHARED ``FAST_FAILURE`` stack would have
    turned into a SELL. Under ``V2_FLOOR`` every one of them must now be inert,
    because none of them appears in the V2 research and two of them fire well
    before the researched -30% stop. They are asserted by their SHARED trigger
    name so that if one ever becomes reachable again, this test names which one
    broke.
    """

    #: ``(shared trigger, label, price, held, momentum)`` -- each a real exit for
    #: the shared stack, and each now inert.
    SHARED_EXITS = (
        ("fast_failure_stop", "-12% stop", 0.88, 10.0, 0.0),
        ("no_follow_through", "flat at 75s", 1.00, 75.0, 0.0),
        ("no_follow_through", "flat at 89s", 1.00, 89.0, 0.0),
        ("momentum_collapse", "momentum -50%", 1.05, 10.0, -50.0),
        ("structure_break", "structure -50%", 1.05, 10.0, 0.0),
    )

    @staticmethod
    def _assess(profile, price, held, momentum, structure=None):
        return ExitPolicy(ExitConfig()).assess(
            _FakePos(1_000.0, strategy_profile=profile),
            now=1_000.0 + held, mark_price=price, data_state=DataState.LIVE,
            momentum_30s_pct=momentum,
            structure_30s_pct=0.0 if structure is None else structure,
            liquidity_change_pct=0.0,
        )

    @pytest.mark.parametrize(
        "trigger,label,price,held,momentum", SHARED_EXITS,
        ids=[f"{t}:{label}" for t, label, *_ in SHARED_EXITS],
    )
    def test_v2_floor_is_not_closed_by_that_shared_trigger(
        self, trigger, label, price, held, momentum
    ):
        structure = -50.0 if trigger == "structure_break" else None
        d = self._assess("V2_FLOOR", price, held, momentum, structure)
        assert d.action == "HOLD", (
            f"{label} closed a V2_FLOOR position via {d.trigger!r}; the "
            f"researched bracket is the whole rule for this profile"
        )
        assert d.trigger is None
        assert d.style is ExitStyle.V2_BRACKET

    @pytest.mark.parametrize(
        "trigger,label,price,held,momentum", SHARED_EXITS,
        ids=[f"{t}:{label}" for t, label, *_ in SHARED_EXITS],
    )
    def test_the_same_condition_still_closes_an_l25_position(
        self, trigger, label, price, held, momentum
    ):
        """
        The mirror image, and the one that matters most.

        The isolation must come from the PROFILE, not from having weakened a
        shared rule. So every condition that is inert for V2_FLOOR must still
        exit an L25_CURRENT position -- otherwise this change would have quietly
        made every strategy looser.
        """
        structure = -50.0 if trigger == "structure_break" else None
        d = self._assess("L25_CURRENT", price, held, momentum, structure)
        assert d.action == "EXIT", (
            f"{label} no longer closes L25_CURRENT; the shared stack was "
            f"weakened rather than the V2 bracket being added"
        )
        assert d.trigger == trigger

    def test_the_disaster_cap_is_reached_only_through_the_v2_stop(self):
        """
        -40% would breach the shared -45% disaster cap's neighbourhood, but the
        researched -30% stop fires first -- so the V2 exit is its own, and the
        simulation boundary is never the thing that closes a V2 position.
        """
        d = self._assess("V2_FLOOR", 0.60, 10.0, 0.0)
        assert d.trigger == V2F_TRIGGER_STOP_LOSS
        assert d.trigger != "disaster_cap"

    def test_a_v2_position_is_not_closed_by_the_shared_60s_cap(self):
        """L25_60S's cap is a profile fact; it does not leak across profiles."""
        d = self._assess("V2_FLOOR", 1.05, 60.0, 1.0)
        assert d.action == "HOLD", d.reason
        assert d.trigger is None

    def test_every_l25_profile_keeps_the_whole_shared_stack(self):
        """
        All three, at the two conditions most likely to have been broken:
        the -12% stop and the 75s no-follow-through.

        ``L25_MAX_HOLD``'s own cap is temporarily raised so the shared stack can
        actually be observed through it -- at its default 60s the cap fires
        before the 75s no-follow-through and the test would pass without proving
        anything. That cap is a deliberate, pre-existing hard-hold behaviour and
        is restored immediately.
        """
        from autopaper.strategy_profile import set_max_hold_s

        for profile in ("L25_CURRENT", "L25_60S"):
            stop = self._assess(profile, 0.88, 10.0, 0.0)
            assert stop.trigger == "fast_failure_stop", profile
            flat = self._assess(profile, 1.00, 75.0, 0.0)
            assert flat.trigger == "no_follow_through", profile

        set_max_hold_s(300.0)
        try:
            hard = self._assess("L25_MAX_HOLD", 0.88, 10.0, 0.0)
            flat = self._assess("L25_MAX_HOLD", 1.00, 75.0, 0.0)
        finally:
            set_max_hold_s(None)
        # a true hard-hold profile has no price or style exits at all; that is
        # its documented behaviour and predates this work
        assert hard.action == "HOLD", hard.reason
        assert flat.action == "HOLD", flat.reason
        # and with its default 60s cap it still exits on time
        assert self._assess(
            "L25_MAX_HOLD", 1.00, 60.0, 1.0).trigger == "max_hold_60s"

    def test_the_l25_profiles_carry_no_bracket(self):
        """
        Declared, so "L25 is unchanged" is a fact in the registry and not only a
        consequence of the bracket lookup happening to miss.
        """
        from autopaper.strategy_profile import PROFILE_EXIT_BRACKET

        for profile in ("L25_CURRENT", "L25_60S", "L25_MAX_HOLD"):
            assert PROFILE_EXIT_BRACKET[StrategyProfile(profile)] is None

    def test_only_v2_floor_produces_the_bracket_style(self):
        """
        The safety net for the whole file: if any other profile can ever emit
        ``V2_BRACKET``, an L25 position is being managed by the wrong rules.
        """
        prices = (0.55, 0.70, 0.88, 1.00, 1.05, 1.30, 1.60)
        for profile in ("L25_CURRENT", "L25_60S", "L25_MAX_HOLD"):
            for price in prices:
                for held in (10.0, 75.0, 90.0):
                    d = self._assess(profile, price, held, 0.0)
                    assert d.style is not ExitStyle.V2_BRACKET, (
                        f"{profile} produced the V2 bracket style"
                    )


class TestDisclosedDivergences:
    """
    What is STILL divergent from the research, after the bracket and the size
    were made exact.

    Two things were divergence and are now closed: the missing take-profit and
    the 12.5% size. Three remain, and they are listed here so a future edit
    cannot close them silently, and cannot quietly OPEN new ones either.
    """

    def test_the_researched_entry_exit_and_size_are_all_now_exact(self):
        """
        The three things that WERE divergent, asserted as closed.

        This test is the counterpart to the note: if someone reimplements the
        bracket on the shared stack again, or drops the sizing override, this
        fails.
        """
        from autopaper.strategy_profile import exit_bracket_for, position_pct_override

        assert exit_bracket_for("V2_FLOOR") == (30.0, -30.0)
        assert position_pct_override("V2_FLOOR") == 10.0
        from autopaper.exits import ExitPolicy
        d = ExitPolicy(ExitConfig()).assess(
            _FakePos(1_000.0, strategy_profile="V2_FLOOR"),
            now=1_030.0, mark_price=1.30, data_state=DataState.LIVE,
            momentum_30s_pct=1.0, structure_30s_pct=0.0,
            liquidity_change_pct=0.0,
        )
        assert d.action == "EXIT"
        assert d.trigger == V2F_TRIGGER_TAKE_PROFIT

    def test_the_remaining_divergence_is_the_narrower_live_population(self):
        """
        Still true: the shared structural route gates apply to V2_FLOOR too.

        Data freshness, the liquidity sanity floor, the market-cap sanity band,
        extension, confidence and the route itself are NOT part of the V2
        research, so the live candidate set is a subset of the researched one.
        The research had no market-cap filter, so a candidate outside the sanity
        band is refused even though V2_FLOOR alone would have authorized it.
        """
        from autopaper.analysis.decision import Decision

        # far above the market-cap sanity band, but a clean V2_FLOOR rule
        report = _decide(market_cap_usd=200_000_000.0,
                         volume_5m_usd=2_000_000_000.0, change_5m_pct=50.0)
        assert report.v2floor.authorized is True, (
            "the V2 gate itself authorized it"
        )
        assert report.decision is Decision.REJECT, (
            "and the shared structural gate still refused it -- this is the "
            "remaining, disclosed divergence"
        )

    def test_the_remaining_divergence_is_unresolved_execution(self):
        """
        Still true, and the reason the operator note says PAPER ONLY.

        The research breaks even at 12.42% execution degradation and the only
        measured buy-side quote gap is 11.5% median on 7 non-stale records, so
        profitability is not established. No code change can close this; only
        prospective PAPER data can.
        """
        from autopaper.strategy_profile import note

        n = note("V2_FLOOR").lower()
        assert "unresolved" in n
        assert "paper only" in n
        assert "break-even" in n or "degradation" in n

    def test_one_position_at_a_time_is_unchanged(self):
        """
        The one-position cap is global, not per-strategy, and is untouched.

        ``max_open_positions`` is a single RealConfig field; V2_FLOOR adds no
        per-strategy position allowance and removes none.
        """
        from autopaper.config import RealConfig

        assert RealConfig().max_open_positions == 1
        assert RealConfig().reentry_enabled is False

    def test_no_rounding_no_minimum_no_all_in(self):
        """
        The researched size is exact, including the micro sizes.

        Rounding a $1.00 stake or nudging it up to a venue minimum would both
        move the position off the researched 10%, and rounding is exactly how a
        $0.155 micro position becomes a $5 order.
        """
        from autopaper.sizing import compute_position_size

        size = compute_position_size(1.55, profile="V2_FLOOR")
        assert size.position_usd == pytest.approx(0.155), (
            "an unrounded 10% of $1.55; no minimum, no round-up"
        )
        assert size.position_usd < 5.0, "and certainly not rounded up to $5"


class TestV2Sizing:
    """
    The researched size: a FLAT 10% of CURRENT equity.

    Flat means at every balance, including above the $50 boundary where the
    shared policy would step to 12.5%. And it is a percentage of the equity
    passed in, which is how compounding works -- the account is re-valued
    before every entry and 10% of that new equity is staked.
    """

    @pytest.mark.parametrize("equity,expected", [
        (100.0, 10.0), (50.0, 5.0), (10.0, 1.0),
    ])
    def test_the_three_stated_examples(self, equity, expected):
        from autopaper.sizing import compute_position_size

        size = compute_position_size(equity, profile="V2_FLOOR")
        assert size.position_usd == pytest.approx(expected), size.rationale
        assert size.position_pct == 10.0

    @pytest.mark.parametrize("equity", [1.55, 10.0, 50.0, 50.01, 100.0,
                                        1_000.0, 10_000.0, 250_000.0])
    def test_it_is_flat_ten_percent_at_every_balance(self, equity):
        """Above $50 too -- which is the whole difference from the shared policy."""
        from autopaper.sizing import compute_position_size

        assert compute_position_size(
            equity, profile="V2_FLOOR").position_pct == 10.0
        assert compute_position_size(
            equity, profile="V2_FLOOR").position_usd == pytest.approx(
                equity * 0.10)

    @pytest.mark.parametrize("equity,expected_pct", [
        (1.55, 10.0), (10.0, 10.0), (50.0, 10.0),
        (50.01, 12.5), (100.0, 12.5),
    ])
    def test_l25_sizing_is_untouched(self, equity, expected_pct):
        """The shared 10%/12.5% ladder, inclusive at $50, for every L25 profile."""
        from autopaper.sizing import compute_position_size

        for profile in ("L25_CURRENT", "L25_60S", "L25_MAX_HOLD", None):
            size = compute_position_size(equity, profile=profile)
            assert size.position_pct == expected_pct, (profile, equity)

    def test_an_unrecognised_profile_gets_the_shared_policy(self):
        """Fail closed to the existing behaviour, never to a guessed size."""
        from autopaper.sizing import compute_position_size

        assert compute_position_size(
            100.0, profile="NOT_A_PROFILE").position_pct == 12.5

    def test_compounding_uses_current_equity_not_a_fixed_stake(self):
        """
        The stake tracks the balance, every time, with no notion of a starting
        amount.

        Three consecutive entries on a re-valued account: $100 -> $10, and the
        account grows 10% (a win at +30% on the notional, rounded for the test),
        so the next stake is 10% of the NEW equity, not another $10.
        """
        from autopaper.sizing import compute_position_size

        equity = 100.0
        stakes = []
        for _ in range(3):
            size = compute_position_size(equity, profile="V2_FLOOR")
            stakes.append(size.position_usd)
            equity += size.position_usd      # stake returned in full
        assert stakes[0] == pytest.approx(10.0)
        assert stakes[1] == pytest.approx(11.0), "10% of $110, not $10 again"
        assert stakes[2] == pytest.approx(12.1), "10% of $121"
        assert stakes == sorted(stakes) and len(set(stakes)) == 3

    def test_a_loss_shrinks_the_next_stake(self):
        """No martingale, in either direction: the stake is always 10% of now."""
        from autopaper.sizing import compute_position_size

        equity = 100.0
        size = compute_position_size(equity, profile="V2_FLOOR")
        equity -= size.position_usd * 3.0     # a -30% stop, i.e. -3 stakes
        nxt = compute_position_size(equity, profile="V2_FLOOR")
        assert nxt.position_usd == pytest.approx(equity * 0.10)
        assert nxt.position_usd < size.position_usd, "and smaller, not bigger"

    def test_the_size_reaches_the_engine_not_just_the_function(self):
        """
        The override has to be WIRED, not merely available.

        The engine passes the SELECTED profile, so selecting V2_FLOOR actually
        changes what gets staked. If the profile were not threaded, every value
        in this class would still pass while production sized at 12.5%.
        """
        import inspect
        import autopaper.engine as engine_mod

        src = inspect.getsource(engine_mod)
        assert "compute_position_size(" in src
        # the entry-sizing call site passes the selected profile
        assert "profile=self.state.strategy_profile" in src, (
            "engine.py does not pass the selected profile to sizing, so "
            "V2_FLOOR would be sized at 12.5% in production"
        )

    def test_the_status_line_shows_the_profile_s_own_size(self):
        """
        An operator reading "12.5%" while the strategy stakes 10% would be being
        told a falsehood about the next order.
        """
        from autopaper.sizing import allocation_label, next_position_label

        assert "10%" in allocation_label(100.0, profile="V2_FLOOR")
        assert "$10.000000" in next_position_label(100.0, profile="V2_FLOOR")
        assert "12.5%" in allocation_label(100.0)
        assert "12.5%" in allocation_label(100.0, profile="L25_CURRENT")

    def test_the_operator_labels_are_threaded_in_telegram(self):
        import inspect
        import autopaper.telegram.ui as ui_mod

        src = inspect.getsource(ui_mod)
        assert "allocation_label(view.equity_usd, profile=" in src
        assert "next_position_label(view.equity_usd, profile=" in src


# ===========================================================================
# 4. PAPER USES THE COMMON EXECUTION PIPELINE
# ===========================================================================


class TestPaperExecutionPath:
    """
    V2_FLOOR is a DECISION, not an execution path.

    These assertions are structural: they check that the strategy contributes a
    gate result and nothing else, and that the fill layer it hands off to is the
    shared one. A simplified backtest-only path would show up here as either a
    V2-specific quote call or a V2-specific fill.
    """

    def test_the_strategy_module_cannot_reach_execution(self):
        """
        The strongest statement available: the gate has no import path to a
        quote, a fill, a signer or a broadcaster, so it cannot bypass them.

        Checked on the AST's IMPORT NODES, not on raw text. A substring scan
        flags the word "signal" inside "signal-quality" and "research" inside a
        provenance note, which is how a naive version of this test reports a
        false breach and then gets deleted.
        """
        import ast
        import pathlib

        src = (pathlib.Path(__file__).resolve().parents[1]
               / "src" / "autopaper" / "v2floor.py").read_text()
        tree = ast.parse(src)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        forbidden = {
            # quote / fill / execution
            "jupiter", "requests", "httpx", "aiohttp", "urllib", "socket",
            # signing / broadcasting / process escape
            "subprocess", "os", "sys", "importlib", "ctypes", "shutil",
            # persistence
            "sqlite3", "json", "pickle",
        }
        leaked = imported & forbidden
        assert not leaked, (
            f"v2floor.py imports {sorted(leaked)}; the entry gate makes a "
            f"decision and execution goes through the common layer"
        )
        # and it is a pure module: no module-level side effects beyond
        # constants and function/class definitions
        for node in tree.body:
            assert isinstance(
                node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                       ast.AsyncFunctionDef, ast.ClassDef, ast.Assign,
                       ast.AnnAssign, ast.Expr, ast.If)
            ), f"unexpected module-level statement: {type(node).__name__}"
            if isinstance(node, ast.Expr) and not isinstance(
                node.value, ast.Constant
            ):
                raise AssertionError(
                    "v2floor.py must have no module-level side effect"
                )

    def test_the_strategy_module_never_sleeps(self):
        """
        No ``sleep``. The 90s cap is a ``held >= cap`` check on each monitor
        tick, never a blocking wait -- a sleep would hold the process, bypass
        the monitor architecture and freeze every other position's exit.
        """
        import ast
        import pathlib

        src = (pathlib.Path(__file__).resolve().parents[1]
               / "src" / "autopaper" / "v2floor.py").read_text()
        tree = ast.parse(src)
        called = {
            node.func.attr for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "sleep" not in called
        assert not any(
            isinstance(n, ast.ImportFrom) and n.module == "time"
            for n in ast.walk(tree)
        )

    def test_the_decision_path_does_not_import_a_profile_into_the_gates(self):
        """
        The profile is read in the decision engine, never inside an entry gate.

        This is the invariant the existing entry-immutability tests assert for
        L25, extended to V2_FLOOR: neither entry module may know a profile
        exists, so neither can change its own behaviour based on one.

        Checked over CODE identifiers, not raw text: a provenance note may
        legitimately mention ``strategy_profile`` while the code does not read
        it, and a text scan would forbid writing the explanation down.
        """
        import ast
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "src" / "autopaper"
        for name in ("l25.py", "v3c.py", "v2a.py", "v2floor.py"):
            tree = ast.parse((root / name).read_text())
            # every name the module actually reads
            read = {
                node.id for node in ast.walk(tree)
                if isinstance(node, ast.Name)
            } | {
                node.attr for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
            }
            # plus everything it imports, which is how the L25 modules would
            # reach the profile registry if they tried
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[-1])
                    imported.update(a.name for a in node.names)
            assert "strategy_profile" not in read, (
                f"{name} reads strategy_profile; an entry gate must not know a "
                f"profile exists"
            )
            assert "strategy_profile" not in imported, (
                f"{name} imports strategy_profile; an entry gate must not know "
                f"a profile exists"
            )

    def test_paper_records_the_profile_it_was_opened_under(self):
        """The execution-aware PAPER path stamps the selected profile."""
        import inspect

        from autopaper import positions as pos_mod

        sig = inspect.signature(pos_mod.PositionBook.open_from_decision)
        assert "strategy_profile" in sig.parameters

    def test_a_position_keeps_the_profile_it_was_opened_under(self):
        """
        Switching profiles mid-flight must not rewrite an open position's exit
        policy. The cap is read off the position, so this is the property that
        makes a profile switch safe.
        """
        from autopaper.exits import _max_hold_for
        from autopaper.modes import RuntimeState

        class P:
            strategy_profile = "L25_CURRENT"

        state = RuntimeState()
        state.select_strategy_profile("V2_FLOOR")
        assert state.strategy_profile == "V2_FLOOR"
        # the open position still carries what it was opened with
        assert _max_hold_for(P()) is None


# ===========================================================================
# 5. REAL IS WIRED ONLY THROUGH THE EXISTING SAFETY GATES
# ===========================================================================


class TestRealGateInteraction:
    """
    Where V2_FLOOR meets REAL, measured rather than assumed.

    REAL pre-trade gate #4 is ``v3c_authorized``, and it reads
    ``report.v3c_authorized`` -- the V3-C SUB-RESULT, not the gate that actually
    made the decision. V3-C is an L25-family gate (its $20,000 market-cap
    ceiling was rejected by the V2_FLOOR research), so a token V2_FLOOR
    authorizes can still be refused by REAL on V3-C grounds.

    This is asserted as-is rather than "fixed", and the reasoning matters:

      * the direction is CONSERVATIVE. It blocks a trade; it never lets one
        through. Making it strategy-aware would RELAX a REAL safety gate, which
        is not a change this work is entitled to make.
      * the alternative is silent and worse: if ``v3c_authorized`` were made to
        read the selected gate, a V2_FLOOR report would journal
        ``v3c_authorized: true`` on a run where V3-C actually refused, and the
        forensic distinction between the two gates would be gone.

    So V2_FLOOR trades that do not also clear V3-C cannot reach REAL. For a
    PAPER-only strategy this costs nothing, and the exact one-line change that
    would lift it (make the gate read ``strategy_authorized``, and RENAME it so
    it stops claiming to be about V3-C) is recorded rather than smuggled in.
    """

    @staticmethod
    def _v2floor_report_above_the_v3c_cap():
        from autopaper.analysis.features import FeatureSet
        from autopaper.config import DecisionConfig, ExitConfig
        from autopaper.analysis.decision import DecisionEngine

        # market cap $150k is above V3-C's $20,000 ceiling but inside V2_FLOOR's
        # rule (vol5/mc = 2.0, 5m change +25%)
        f, a, nar, t = _features(market_cap_usd=150_000.0,
                                 volume_5m_usd=300_000.0, change_5m_pct=25.0)
        return DecisionEngine(DecisionConfig(strategy_profile="V2_FLOOR"),
                              ExitConfig()).decide(
            features=f, alert=a, nar=nar,
            analysis_start=t, analysis_complete=t + 0.05)

    def test_a_v2floor_trade_can_be_refused_by_the_v3c_real_gate(self):
        r = self._v2floor_report_above_the_v3c_cap()
        assert r.strategy_profile == "V2_FLOOR"
        assert r.decision is Decision.BUY, "V2_FLOOR authorized this token"
        assert r.strategy_authorized is True, (
            "the SELECTED gate authorized it -- this is what authorized it"
        )
        assert r.v3c_authorized is False, (
            "V3-C refused it (market cap above the $20,000 L25 ceiling)"
        )
        assert "V3-C" in r.v3c.detail

    def test_the_two_gates_stay_distinguishable_in_the_journal(self):
        """
        If these collapse into one boolean the forensic record is lost.

        ``v3c_authorized`` and ``strategy_authorized`` are separate keys, and
        both must appear, so a reader can always tell WHICH gate authorized a
        trade and WHICH L25 gate would have refused it.
        """
        r = self._v2floor_report_above_the_v3c_cap()
        d = r.as_dict()
        assert d["strategy_authorized"] is True
        assert d["v3c_authorized"] is False
        assert d["strategy_status"] == r.v2floor.status
        assert d["v3c"]["status"] == r.v3c.status
        assert d["strategy_profile"] == "V2_FLOOR"

    def test_every_pretrade_gate_is_still_present_and_unchanged(self):
        """
        The whole gate list, pinned by name and length.

        A gate that disappears under a new strategy is a silent hole; this fails
        loudly instead.
        """
        from autopaper.execution.real import PRE_TRADE_CHECKS

        assert len(PRE_TRADE_CHECKS) == 18, PRE_TRADE_CHECKS
        assert "v3c_authorized" in PRE_TRADE_CHECKS
        # the gates that make REAL dangerous to weaken, all still here
        for gate in ("real_mode_explicit", "real_armed_by_operator",
                     "no_emergency_stop", "wallet_loaded", "wallet_correct",
                     "rpc_healthy", "quote_provider_ready",
                     "sizing_matches_policy", "quote_valid",
                     "quote_freshness", "exposure_within_limit",
                     "open_positions_within_limit", "no_duplicate_position",
                     "no_reentry", "no_concurrent_operation"):
            assert gate in PRE_TRADE_CHECKS, gate

    def test_arm_real_still_refuses_without_a_valid_preflight(self):
        """The arming path is unchanged for the new profile."""
        from autopaper.modes import RuntimeState, ModeError

        state = RuntimeState()
        state.select_strategy_profile("V2_FLOOR")
        token = state.request_real_confirmation()
        with pytest.raises(ModeError, match="preflight"):
            state.arm_real(confirmation=token, preflight_ok=False)
        assert state.real_armed is False
        assert state.real_trading_enabled is False


class TestRealRemainsDisabled:
    def test_selecting_the_strategy_does_not_arm_real(self):
        from autopaper.modes import RuntimeState

        state = RuntimeState()
        state.select_strategy_profile("V2_FLOOR")
        assert state.real_armed is False

    def test_real_armed_is_not_a_constructor_argument(self):
        from autopaper.modes import RuntimeState

        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(RuntimeState)}
        assert "real_armed" in fields
        assert fields["real_armed"].init is False, (
            "real_armed must not be settable at construction"
        )

    def test_real_trading_is_reported_disabled_by_default(self):
        from autopaper.config import RealConfig
        assert RealConfig().trading_enabled is False

    def test_env_request_never_authorizes_real(self):
        from autopaper.safety import assert_real_trading_disabled_at_startup
        # the startup assertion must not raise under any profile selection
        from autopaper.modes import RuntimeState

        state = RuntimeState()
        state.select_strategy_profile("V2_FLOOR")
        assert_real_trading_disabled_at_startup()

    def test_the_safety_audit_passes_with_the_new_profile(self):
        from autopaper.safety import audit
        assert audit().ok is True

    def test_the_audit_names_the_v2_floor_invariants(self):
        from autopaper.safety import audit
        names = {c[0] for c in audit().checks}
        assert "profile_v2_floor_threshold_is_frozen" in names
        assert "profile_v2_floor_cap_is_90" in names
        # the researched exit and the researched size are safety-relevant, so
        # the audit re-verifies both rather than trusting the profile table
        assert "profile_v2_floor_exit_bracket_is_the_research_rule" in names
        assert "profile_v2_floor_sizing_is_ten_percent" in names
        # and the "changed nothing else" guard
        assert "profile_only_v2_floor_overrides_sizing_or_exit" in names
        assert "profile_selecting_v2_floor_does_not_arm_real" in names

    def test_the_pretrade_gates_are_unchanged(self):
        """V2_FLOOR adds no gate and removes none."""
        from autopaper.execution.real import PRE_TRADE_CHECKS

        for required in (
            "real_mode_explicit", "real_armed_by_operator", "no_emergency_stop",
            "v3c_authorized", "wallet_loaded", "wallet_correct", "rpc_healthy",
            "quote_provider_ready", "sizing_matches_policy", "quote_valid",
            "quote_freshness", "no_concurrent_operation",
        ):
            assert required in PRE_TRADE_CHECKS, required

    def test_the_strategy_module_adds_no_real_execution_surface(self):
        """
        No signing, no broadcasting, no authorization, no mode switching.

        Selecting a strategy must not be able to change the trading mode, and
        nothing in the gate may touch REAL. Checked over the AST's names and
        imports rather than raw text, because the module's provenance note
        discusses what it is NOT layered on top of.
        """
        import ast
        import pathlib

        src = (pathlib.Path(__file__).resolve().parents[1]
               / "src" / "autopaper" / "v2floor.py").read_text()
        tree = ast.parse(src)
        names = {
            n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
        } | {
            n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
        }
        forbidden = {
            "real", "arm", "armed", "sign", "signer", "broadcast", "mode",
            "environ", "getenv", "env",
        }
        assert not (names & forbidden), (
            f"v2floor.py references {sorted(names & forbidden)}; the entry "
            f"gate cannot arm, sign, broadcast or switch mode"
        )
        assert not any(
            isinstance(n, ast.Attribute) and n.attr in ("system", "popen")
            for n in ast.walk(tree)
        )


# ===========================================================================
# 6. THE FILL IS THE SHARED ONE
# ===========================================================================


class TestPaperFillIsTheCommonOne:
    """
    The strategy ends at a DECISION. Everything after it is the shared
    execution-aware PAPER pipeline, and V2_FLOOR adds nothing to it.

    These tests are about the SEAM. A V2-specific fill path, a mark-based
    settlement shortcut, or a quote the strategy captured itself would all show
    up here.
    """

    @staticmethod
    def _fill(side="entry", **kw):
        from autopaper.config import ExecutionModel
        from autopaper.execution import FillInputs, simulate_fill
        from .conftest import obs

        base = dict(
            side=side, quantity=1000.0, notional_hint_usd=10.0,
            quoted_price=1e-4, decision_time=T0,
            observation=obs(T0 + 1.2, 1.05e-4, liquidity_usd=50_000.0),
            observed_at=T0 + 1.2,
        )
        base.update(kw)
        return simulate_fill(FillInputs(**base), ExecutionModel())

    def test_the_entry_fill_is_quote_and_slippage_based(self):
        """
        Never a bare mark.

        The fill must be at or worse than the quote, must cost money, and must
        record which reference it used.
        """
        r = self._fill()
        assert r.filled is True
        assert r.fill_price > r.quoted_price, (
            "an entry fill must cost money; a mark-based fill would not"
        )
        assert r.slippage_usd > 0
        assert r.quote_price_source in {
            "EXECUTABLE_QUOTE", "MARK_DUPLICATE", "MARK_ONLY",
            "MARK_ONLY_QUOTE_NOT_CONVERTIBLE",
        }

    def test_the_exit_fill_is_also_adverse(self):
        r = self._fill(side="exit")
        assert r.filled is True
        assert r.fill_price < r.price_at_quote, (
            "a sale must never be modelled as a favourable fill"
        )
        assert r.slippage_usd > 0

    def test_no_market_data_is_a_recorded_refusal_not_a_mark_fill(self):
        """
        The brief's requirement: no mark-based settlement bypass.

        With no observation at all the fill must REFUSE with a reason, rather
        than quietly settling at a price it does not have.
        """
        r = self._fill(observation=None)
        assert r.filled is False
        assert r.reason == "no_quote"
        assert r.fill_price is None
        assert r.fee_usd == 0.0

    def test_an_unusable_price_is_distinguished_from_an_absent_one(self):
        from .conftest import obs
        r = self._fill(observation=obs(T0 + 1.2, None, liquidity_usd=50_000.0))
        assert r.filled is False
        assert r.reason == "market_unavailable"

    def test_a_stale_quote_is_a_recorded_refusal(self):
        from .conftest import obs
        very_stale = obs(T0 - 20.0, 1.05e-4, liquidity_usd=50_000.0)
        r = self._fill(observation=very_stale)
        assert r.filled is False
        assert r.reason == "stale_quote"

    def test_the_fill_records_the_provenance_the_brief_requires(self):
        """
        decision mark, buy quote, paper fill -- all recorded, all distinct.

        The report's whole execution finding is that a mark and an executable
        price are different things, so a fill that cannot say which one it used
        is uninterpretable.
        """
        r = self._fill()
        for key in ("price_at_quote", "fill_price", "quoted_price",
                    "quote_price_source", "slippage_usd", "observed_mark_price",
                    "quote_price_source", "reference_price_provenance",
                    "fill_price_provenance"):
            assert hasattr(r, key), key
        assert r.fill_price != r.quoted_price
        assert r.quote_price_source == "MARK_DUPLICATE"
        assert r.fill_price_provenance == "PAPER_FILL"

    def test_sizing_feeds_the_fill_from_the_shared_policy(self):
        """The strategy never sizes a trade; the shared policy does."""
        from autopaper.sizing import compute_position_size
        size = compute_position_size(10.0)
        assert size.position_usd == pytest.approx(1.0)
        assert size.position_pct == 10.0
        # and the fill's notional comes from the caller, never from the gate
        r = self._fill(notional_hint_usd=size.position_usd, quantity=0.0)
        assert r.filled is True

    def test_the_execution_layer_cannot_even_name_the_strategy(self):
        """
        The modules that MOVE MONEY stay strategy-agnostic.

        ``fill.py``, ``quoting.py`` and ``real_compat.py`` are the execution
        path, and the brief is explicit that V2_FLOOR uses the common one. So
        none of them may name the strategy: if a V2-specific quote or fill ever
        appeared here, it would be a second execution path, and this fails
        immediately rather than at the next quiet divergence.

        ``exits.py`` and ``sizing.py`` are deliberately EXCLUDED, because
        deciding WHEN to sell and HOW MUCH to stake are strategy decisions and
        both are now legitimately profile-aware. What they must not do is touch
        execution -- which the tests below and the shared-fill tests cover.
        """
        import ast
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "src" / "autopaper"
        for rel in ("execution/fill.py", "execution/quoting.py",
                    "paper_live/real_compat.py"):
            path = root / rel
            assert path.exists(), rel
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    assert "V2_FLOOR" not in node.value, (
                        f"{rel} mentions V2_FLOOR; the execution layer must be "
                        f"strategy-agnostic"
                    )

    def test_neither_the_exit_nor_the_sizing_module_can_execute(self):
        """
        The two modules that ARE strategy-aware still cannot move money.

        This is the architectural line, and it is worth pinning: a profile may
        decide when to sell and what notional to request, but it may not quote,
        fill, sign or broadcast. Both modules are scanned for the forbidden
        imports rather than for the string, because the dangerous thing is the
        import, not the name.
        """
        import ast
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "src" / "autopaper"
        forbidden = {
            "jupiter", "txcodec", "solana", "provider", "intents", "planner",
            "simulate_fill", "sign_transaction", "broadcast",
        }
        for rel in ("exits.py", "sizing.py"):
            tree = ast.parse((root / rel).read_text())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                    names += [a.name for a in node.names]
                for imported in names:
                    leaf = str(imported).rsplit(".", 1)[-1]
                    assert leaf not in forbidden, (
                        f"{rel} imports {imported!r}; a strategy may decide to "
                        f"sell, never to execute the sell"
                    )

    def test_the_exit_and_sizing_modules_import_only_the_registry(self):
        """
        Where a profile fact is read from: the one frozen registry.

        Both modules must go through ``strategy_profile`` rather than
        re-declaring the numbers locally, or the registry would stop being the
        single source and the safety audit's frozen-value check would verify a
        constant nothing uses.
        """
        import ast
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "src" / "autopaper"
        for rel in ("exits.py", "sizing.py"):
            tree = ast.parse((root / rel).read_text())
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    imported.update(a.name for a in node.names)
            assert "exit_bracket_for" in imported or rel == "sizing.py", rel
            assert "position_pct_override" in imported or rel == "exits.py", rel


# ===========================================================================
# 7. THE FULL LIFECYCLE, ON THE COMMON PATH
# ===========================================================================


class TestV2LifecycleOnTheCommonPath:
    """
    End to end, and every step is the shared one:

        entry decision -> buy quote -> paper fill -> monitor
        -> V2 exit decision -> sell quote -> paper fill -> settlement

    Written as one test on purpose. The unit tests above each prove a step; this
    proves the SEQUENCE, which is where a mark-based settlement shortcut or a
    strategy-owned fill would actually show up. Every stage records which
    reference it used, and the assertions check that each of those is the
    execution-aware one rather than the raw mark.
    """

    @staticmethod
    def _obs(t, price, **kw):
        from .conftest import obs
        # a token that satisfies V2_FLOOR: vol5/mc = 2.0, 5m change +25%
        base = dict(market_cap_usd=150_000.0, liquidity_usd=50_000.0,
                    volume_5m_usd=300_000.0, price_change_5m=25.0)
        base.update(kw)
        return obs(t, price, **base)

    def test_a_full_win_lifecycle(self):
        """
        +30% is reached and the take-profit fires, on a real sell quote.

        The whole path, in order, with no mark-based shortcut anywhere.
        """
        from autopaper.config import ExecutionModel
        from autopaper.execution import FillInputs, simulate_fill
        from autopaper.sizing import compute_position_size

        t0 = T0
        model = ExecutionModel()

        # -- 1. entry decision ------------------------------------------
        f, a, nar, when = _features(volume_5m_usd=300_000.0,
                                     market_cap_usd=150_000.0,
                                     change_5m_pct=25.0, t0=t0)
        report = eng("V2_FLOOR").decide(
            features=f, alert=a, nar=nar, analysis_start=when,
            analysis_complete=when + 0.05,
        )
        assert report.decision is Decision.BUY
        assert report.strategy_authorized is True
        assert report.strategy_profile == "V2_FLOOR"

        # -- 2. sizing (profile-scoped, 10% of CURRENT equity) ----------
        size = compute_position_size(100.0, profile=report.strategy_profile)
        assert size.position_pct == 10.0
        assert size.position_usd == pytest.approx(10.0)

        # -- 3. buy quote -> 4. paper fill ------------------------------
        buy_obs = self._obs(t0 + 1.2, 1.0e-5)
        buy = simulate_fill(FillInputs(
            side="entry", quantity=size.position_usd / buy_obs.price_usd,
            notional_hint_usd=size.position_usd, quoted_price=buy_obs.price_usd,
            decision_time=t0, observation=buy_obs, observed_at=t0 + 1.2,
        ), model)
        assert buy.filled is True
        assert buy.quote_price_source == "MARK_DUPLICATE"
        assert buy.fill_price_provenance == "PAPER_FILL"
        assert buy.fill_price > buy_obs.price_usd, "the buy must cost money"

        # -- 5. monitor: hold, then the researched +30% -----------------
        pos = _FakePos(t0, fill_price=buy.fill_price,
                      strategy_profile="V2_FLOOR")
        policy = ExitPolicy(ExitConfig())
        live = dict(data_state=DataState.LIVE, momentum_30s_pct=1.0,
                    structure_30s_pct=1.0, liquidity_change_pct=0.0)

        held = policy.assess(pos, now=t0 + 20.0,
                             mark_price=buy.fill_price * 1.10, **live)
        assert held.action == "HOLD", held.reason

        tp_price = buy.fill_price * 1.30
        tp = policy.assess(pos, now=t0 + 45.0, mark_price=tp_price, **live)
        assert tp.action == "EXIT"
        assert tp.trigger == V2F_TRIGGER_TAKE_PROFIT

        # -- 6. sell quote -> 7. paper fill -----------------------------
        sell_obs = self._obs(t0 + 46.2, tp_price * 0.99)
        sell = simulate_fill(FillInputs(
            side="exit", quantity=size.position_usd / buy_obs.price_usd,
            notional_hint_usd=size.position_usd,
            quoted_price=tp_price, decision_time=t0 + 45.0,
            observation=sell_obs, observed_at=t0 + 46.2,
        ), model)
        assert sell.filled is True
        assert sell.slippage_usd > 0
        assert sell.fill_price < sell.price_at_quote, (
            "the sell must not be modelled as a favourable fill"
        )

        # -- 8. settlement ----------------------------------------------
        proceeds = sell.fill_price * (size.position_usd / buy_obs.price_usd)
        realised = (proceeds - size.position_usd) / size.position_usd * 100.0
        assert realised > 0, "a take-profit exit must realise a gain"
        # and it is materially less than the +30% mark, because both the
        # latency and the slippage are real. This is the whole execution
        # finding, and it is why the strategy is PAPER only.
        assert realised < 30.0

    def test_a_full_loss_lifecycle(self):
        """-30% is reached and the stop fires, on a real sell quote."""
        from autopaper.config import ExecutionModel
        from autopaper.execution import FillInputs, simulate_fill
        from autopaper.sizing import compute_position_size

        t0, model = T0, ExecutionModel()
        size = compute_position_size(50.0, profile="V2_FLOOR")
        assert size.position_usd == pytest.approx(5.0)

        buy_obs = self._obs(t0 + 1.2, 1.0e-5)
        buy = simulate_fill(FillInputs(
            side="entry", quantity=size.position_usd / buy_obs.price_usd,
            notional_hint_usd=size.position_usd, quoted_price=buy_obs.price_usd,
            decision_time=t0, observation=buy_obs, observed_at=t0 + 1.2,
        ), model)
        assert buy.filled is True

        pos = _FakePos(t0, fill_price=buy.fill_price,
                      strategy_profile="V2_FLOOR")
        policy = ExitPolicy(ExitConfig())
        live = dict(data_state=DataState.LIVE, momentum_30s_pct=-50.0,
                    structure_30s_pct=-50.0, liquidity_change_pct=-99.0)

        # -12% and -20% are NOT exits for this profile, however bad it looks
        for pct in (-0.12, -0.20):
            d = policy.assess(pos, now=t0 + 20.0,
                              mark_price=buy.fill_price * (1 + pct), **live)
            assert d.action == "HOLD", (
                f"{pct:+.0%} closed a V2_FLOOR position via {d.trigger!r}"
            )

        sl_price = buy.fill_price * 0.70
        sl = policy.assess(pos, now=t0 + 55.0, mark_price=sl_price, **live)
        assert sl.action == "EXIT"
        assert sl.trigger == V2F_TRIGGER_STOP_LOSS

        sell_obs = self._obs(t0 + 56.2, sl_price * 1.01)
        sell = simulate_fill(FillInputs(
            side="exit", quantity=size.position_usd / buy_obs.price_usd,
            notional_hint_usd=size.position_usd, quoted_price=sl_price,
            decision_time=t0 + 55.0, observation=sell_obs,
            observed_at=t0 + 56.2,
        ), model)
        assert sell.filled is True
        proceeds = sell.fill_price * (size.position_usd / buy_obs.price_usd)
        realised = (proceeds - size.position_usd) / size.position_usd * 100.0
        assert realised < 0
        assert realised < -30.0, "worse than the -30% mark, as costs dictate"

    def test_a_time_lifecycle_settles_at_the_cap(self):
        """
        90s with nothing happening, then a real sell quote.

        The cap is a TRIGGER like any other: it still has to be satisfied with an
        executable SELL, so it cannot resolve into a fill by itself.
        """
        from autopaper.config import ExecutionModel
        from autopaper.execution import FillInputs, simulate_fill

        t0, model = T0, ExecutionModel()
        buy_obs = self._obs(t0 + 1.2, 1.0e-5)
        qty = 1_000_000.0
        buy = simulate_fill(FillInputs(
            side="entry", quantity=qty, notional_hint_usd=qty * 1.0e-5,
            quoted_price=1.0e-5, decision_time=t0,
            observation=buy_obs, observed_at=t0 + 1.2,
        ), model)

        pos = _FakePos(t0, fill_price=buy.fill_price,
                      strategy_profile="V2_FLOOR")
        policy = ExitPolicy(ExitConfig())
        live = dict(data_state=DataState.LIVE, momentum_30s_pct=0.0,
                    structure_30s_pct=0.0, liquidity_change_pct=0.0)

        # flat for the whole window -- under the shared stack this would have
        # been closed at 75s by no_follow_through
        for held in (10.0, 40.0, 75.0, 89.0):
            d = policy.assess(pos, now=t0 + held,
                              mark_price=buy.fill_price, **live)
            assert d.action == "HOLD", f"closed at {held}s via {d.trigger!r}"

        cap = policy.assess(pos, now=t0 + 90.0,
                            mark_price=buy.fill_price, **live)
        assert cap.action == "EXIT"
        assert cap.trigger == V2F_TRIGGER_MAX_HOLD

        sell_obs = self._obs(t0 + 91.2, buy.fill_price)
        sell = simulate_fill(FillInputs(
            side="exit", quantity=qty, notional_hint_usd=qty * 1.0e-5,
            quoted_price=buy.fill_price, decision_time=t0 + 90.0,
            observation=sell_obs, observed_at=t0 + 91.2,
        ), model)
        assert sell.filled is True
        # a time exit on an unchanged price is a small loss to costs, not a
        # free flat
        assert sell.fill_price < buy.fill_price

    def test_no_stage_settles_at_the_mark(self):
        """
        No mark-based settlement shortcut, anywhere in the lifecycle.

        Every fill must be worse-or-equal to the reference it was given and must
        carry its provenance, on BOTH legs. A path that settled at the decision
        mark would satisfy the exit assertions above while quietly skipping the
        execution model entirely.
        """
        from autopaper.config import ExecutionModel
        from autopaper.execution import FillInputs, simulate_fill

        model = ExecutionModel()
        for side in ("entry", "exit"):
            for drift in (-0.05, -0.01, 0.0, 0.01, 0.05):
                o = self._obs(T0 + 1.2, 1.0e-5 * (1 + drift))
                r = simulate_fill(FillInputs(
                    side=side, quantity=1_000_000.0, notional_hint_usd=10.0,
                    quoted_price=1.0e-5, decision_time=T0,
                    observation=o, observed_at=T0 + 1.2,
                ), model)
                assert r.filled is True
                assert r.reference_price_provenance
                assert r.fill_price_provenance == "PAPER_FILL"
                assert r.slippage_usd > 0, (side, drift)
                if side == "entry":
                    assert r.fill_price > r.price_at_quote, (side, drift)
                else:
                    assert r.fill_price < r.price_at_quote, (side, drift)

    def test_a_missing_quote_at_the_exit_is_a_refusal_not_a_fill(self):
        """
        The V2 exit still has to survive contact with the quote service.

        A trigger is not a fill: if no quote can be had, the position stays open
        and the reason is recorded. That is the common path's behaviour, reached
        through the V2 bracket without modification.
        """
        from autopaper.config import ExecutionModel
        from autopaper.execution import FillInputs, simulate_fill

        pos = _FakePos(1.0, strategy_profile="V2_FLOOR")
        policy = ExitPolicy(ExitConfig())
        d = policy.assess(pos, now=T0 + 90.0, mark_price=1.30,
                          data_state=DataState.LIVE, momentum_30s_pct=1.0,
                          structure_30s_pct=1.0, liquidity_change_pct=0.0)
        assert d.trigger == V2F_TRIGGER_TAKE_PROFIT

        r = simulate_fill(FillInputs(
            side="exit", quantity=1_000.0, notional_hint_usd=10.0,
            quoted_price=1.30, decision_time=T0 + 90.0,
            observation=None, observed_at=T0 + 90.0,
        ), ExecutionModel())
        assert r.filled is False
        assert r.reason == "no_quote"
        assert r.fill_price is None
        assert r.fee_usd == 0.0
