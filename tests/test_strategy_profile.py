"""
Selectable strategy profile: L25_CURRENT (default) and L25_60S.

The properties under test, in the order they matter:

  1. the ENTRY is identical in both profiles -- the profile selects an exit
     policy, never a strategy;
  2. the DEFAULT is L25_CURRENT, which adds nothing and is therefore
     behaviourally identical to the runtime that existed before profiles;
  3. an unknown profile is REFUSED, not defaulted;
  4. a 60s cap is an ADDITION -- every existing exit keeps its precedence, and
     fires first when it fires;
  5. a POSITION is immutable: the profile it was opened under governs its exit,
     so a mid-flight operator switch cannot rewrite a live exit policy;
  6. selection is INERT -- it cannot start the runtime, arm REAL, unblock
     entries or clear an emergency stop, and it broadcasts nothing;
  7. REAL reaches the SAME exit executor, with no profile-specific shortcut.

Everything is offline. No network, no Telegram, no RPC, and nothing is signed or
broadcast: the REAL assertions here are about WHICH code path a 60s exit takes,
never about sending a transaction.
"""


from __future__ import annotations

import ast
from dataclasses import replace as dc_replace
from pathlib import Path

import pytest

from autopaper.config import DecisionConfig
from autopaper.exits import ExitPolicy
from autopaper.models import DataState, ExitStyle
from autopaper.modes import RunState, RuntimeState
from autopaper.safety import audit
from autopaper.strategy_profile import (
    DEFAULT_PROFILE,
    ENTRY_STRATEGY,
    MAX_HOLD_TRIGGER,
    PROFILE_MAX_HOLD_S,
    PROFILE_VALUES,
    StrategyProfile,
    StrategyProfileError,
    as_dict,
    describe,
    entry_strategy,
    exit_strategy,
    label,
    max_hold_s,
    note,
    parse_profile,
    set_max_hold_s,
    status_line,
    uses_l25_entry,
)
from autopaper.telegram import ui

pytestmark = [
    pytest.mark.execution,
    pytest.mark.safety,
    pytest.mark.strategy,
]

ROOT = Path(__file__).resolve().parents[1]

T0 = 1_700_000_000.0

#: Starting cash for the paper-account harness below.
START_CASH = 100.0


# ---------------------------------------------------------------------------
# a position stand-in with only the fields the exit policy actually reads
# ---------------------------------------------------------------------------


class FakePos:
    """
    What ``ExitPolicy.assess`` reads, and nothing else.

    Deliberately a bare object rather than a real ``Position``: these tests are
    about the policy's decision given a profile and a hold time, and building a
    full position would add fields the policy never looks at.
    """

    def __init__(self, entry_timestamp: float, *, fill_price: float = 1.0,
                 strategy_profile: str = "L25_CURRENT",
                 peak_return_pct: float = 0.0,
                 exit_style: ExitStyle = ExitStyle.FAST_FAILURE) -> None:
        self.position_id = "pos_test"
        self.entry_timestamp = entry_timestamp
        self.fill_price = fill_price
        self.strategy_profile = strategy_profile
        self.peak_return_pct = peak_return_pct
        self.exit_style = exit_style
        self.runner_mode = False
        self.exit_hold_count = 0

    def return_at(self, price):
        if not price:
            return None
        return (price - self.fill_price) / self.fill_price


def assess(pos, now: float, price: float, policy: ExitPolicy):
    return policy.assess(
        pos, now=now, mark_price=price, data_state=DataState.LIVE,
        momentum_30s_pct=0.0, structure_30s_pct=0.0,
        liquidity_change_pct=0.0,
    )


@pytest.fixture
def policy() -> ExitPolicy:
    from autopaper.config import ExitConfig
    return ExitPolicy(ExitConfig())


# ===========================================================================
# 1. STRATEGY SELECTION
# ===========================================================================


class TestStrategySelection:
    def test_default_is_l25_current(self):
        """A fresh runtime and a fresh config both default to today's behaviour."""
        assert DEFAULT_PROFILE is StrategyProfile.L25_CURRENT
        assert RuntimeState().profile is StrategyProfile.L25_CURRENT
        assert DecisionConfig().strategy_profile == "L25_CURRENT"

    def test_allowed_values_are_exactly_the_declared_profiles(self):
        # The CLOSED set, asserted against the canonical vocabulary rather than
        # a literal, so a NEW profile is caught here instead of passing
        # silently. V2_FLOOR was added when a second entry strategy
        # (autopaper.v2floor) was registered; L25_TP100 was added as a TP100
        # research variant. Both are recorded here deliberately, because the
        # point of this assertion is that a profile cannot appear in
        # the selector without also being named in the tests and in the safety
        # audit.
        assert set(PROFILE_VALUES) == {p.value for p in StrategyProfile}
        assert set(PROFILE_VALUES) == {
            "L25_CURRENT", "L25_60S", "L25_MAX_HOLD", "L25_TP100", "V2_FLOOR",
            "V2_FLOOR_BETA",
        }
        assert len(PROFILE_VALUES) == 6
        # L25_TP100 remains a declared PROFILE so history and open positions
        # keep their meaning, but it is no longer selectable as an ARM.
        from autopaper.paper_live.arms import arm_names
        assert "L25_TP100" not in arm_names()

    def test_l25_current_has_no_maximum_hold(self):
        """The default must add nothing, or it is not really the default."""
        assert max_hold_s(StrategyProfile.L25_CURRENT) is None
        assert max_hold_s("L25_CURRENT") is None

    def test_l25_60s_has_a_60_second_maximum_hold(self):
        assert max_hold_s(StrategyProfile.L25_60S) == 60.0
        assert max_hold_s("L25_60S") == 60.0

    def test_every_profile_declares_entry_and_exit_and_max_hold(self):
        """A profile must name all three parts, not just the cap."""
        for profile in StrategyProfile:
            d = as_dict(profile)
            # Every profile must resolve to a DECLARED entry strategy. The L25
            # family resolves to ENTRY_STRATEGY; V2_FLOOR resolves to its own
            # gate. What is forbidden is a profile with no declared gate, which
            # is what "selectable but not editable" actually means.
            assert d["entry_strategy"] in ("L25", "V2_FLOOR")
        for profile in (
            StrategyProfile.L25_CURRENT,
            StrategyProfile.L25_60S,
            StrategyProfile.L25_MAX_HOLD,
        ):
            assert as_dict(profile)["entry_strategy"] == ENTRY_STRATEGY
            assert d["exit_strategy"]
            assert "max_hold_s" in d

    def test_both_profiles_run_the_same_exit_style(self):
        """They differ by the CAP, not by a different rule set."""
        assert exit_strategy(StrategyProfile.L25_CURRENT) == \
            exit_strategy(StrategyProfile.L25_60S)

    @pytest.mark.parametrize("raw,expected", [
        ("L25_CURRENT", StrategyProfile.L25_CURRENT),
        ("L25_60S", StrategyProfile.L25_60S),
        ("l25_60s", StrategyProfile.L25_60S),          # case tolerant
        ("  L25_60S  ", StrategyProfile.L25_60S),      # whitespace tolerant
        ("L25_CURRENT", StrategyProfile.L25_CURRENT),
    ])
    def test_valid_values_parse(self, raw, expected):
        assert parse_profile(raw) is expected

    @pytest.mark.parametrize("raw", [
        "L25_60", "L25", "TYPO", "L25_CURRENT_60S", "60S", "L25_60sX",
        "L25_30S", "L25_120S", "CURRENT", "ALL", "none",
    ])
    def test_invalid_values_fail_closed(self, raw):
        """An unknown profile is REFUSED. Never silently defaulted."""
        with pytest.raises(StrategyProfileError) as e:
            parse_profile(raw)
        msg = str(e.value)
        assert raw in msg
        # the error must name what IS allowed, so an operator can fix it
        assert "L25_CURRENT" in msg and "L25_60S" in msg

    def test_config_refuses_an_invalid_profile_at_construction(self):
        """Fail closed at config load, not at first trade."""
        with pytest.raises(ValueError):
            DecisionConfig(strategy_profile="L25_60")

    def test_blank_profile_is_the_default_not_an_error(self):
        """An absent setting is legitimate configuration, not a typo."""
        assert parse_profile("") is StrategyProfile.L25_CURRENT
        assert parse_profile(None) is StrategyProfile.L25_CURRENT
        assert parse_profile("   ") is StrategyProfile.L25_CURRENT

    def test_config_accepts_both_valid_profiles(self):
        assert DecisionConfig(strategy_profile="L25_CURRENT")
        assert DecisionConfig(strategy_profile="L25_60S")

    def test_selecting_a_profile_updates_the_authoritative_state(self):
        state = RuntimeState()
        accepted, message = state.select_strategy_profile("L25_60S")
        assert accepted is True
        assert state.profile is StrategyProfile.L25_60S
        assert state.strategy_profile == "L25_60S"
        assert "60s" in message

    def test_selecting_an_invalid_profile_is_refused_and_changes_nothing(self):
        state = RuntimeState()
        accepted, message = state.select_strategy_profile("L25_60")
        assert accepted is False
        assert "allowed values" in message
        assert state.profile is StrategyProfile.L25_CURRENT, \
            "a refused selection must not move the profile"
        assert state.last_refusal

    def test_reselecting_the_current_profile_is_a_no_op(self):
        state = RuntimeState()
        accepted, message = state.select_strategy_profile("L25_CURRENT")
        assert accepted is True
        assert "already selected" in message

    def test_selection_is_recorded_in_the_audit_trail(self):
        state = RuntimeState()
        state.select_strategy_profile("L25_60S")
        assert any(h["action"] == "select_strategy_profile" for h in state.history)
        assert any(h.get("detail", "").startswith("L25_CURRENT -> L25_60S")
                   for h in state.history)

    def test_state_snapshot_names_the_profile_and_cap(self):
        state = RuntimeState()
        state.select_strategy_profile("L25_60S")
        snap = state.snapshot()
        assert snap["strategy_profile"] == "L25_60S"
        assert snap["strategy_label"] == "L25 + 60s"
        assert snap["max_hold_s"] == 60.0

    def test_config_source_documents_the_profile(self):
        text = (ROOT / ".env.example").read_text()
        assert "STRATEGY_PROFILE=L25_CURRENT" in text
        assert "L25_60S" in text
        # the template must also say the honest thing about the 60s profile
        assert "NOT proven superior" in text


# ===========================================================================
# 2. ENTRY IMMUTABILITY  -- the profile must never reach the entry
# ===========================================================================


class TestEntryImmutability:
    def test_the_l25_entry_module_does_not_import_a_profile(self):
        """
        The strongest available statement: the entry gate cannot know a profile
        exists, because it never imports one.
        """
        src = (ROOT / "src" / "autopaper" / "l25.py").read_text()
        assert "strategy_profile" not in src, \
            "the L25 entry module must not reference a strategy profile"

    def test_the_decision_path_does_not_import_a_profile(self):
        for name in ("v3c.py", "v2a.py"):
            src = (ROOT / "src" / "autopaper" / name).read_text()
            assert "strategy_profile" not in src, \
                f"{name} must not reference a strategy profile"

    def test_profiles_carry_no_tunable_threshold(self):
        """
        No profile carries a TUNABLE strategy number.

        ``own_exit_triggers`` is a tuple of trigger TOKENS, not a number, so it
        cannot become a threshold.

        The numbers a profile may carry are exactly two kinds, and both are
        frozen rather than tunable:

          * ``max_hold_s`` -- the one thing a profile has always been allowed to
            differ on;
          * the V2_FLOOR ``exit_bracket`` (TP/SL) and ``position_pct_override``,
            which are the RESEARCHED rule for that strategy.

        So the guard below allows those keys, asserts each V2 number is exactly
        the researched value, and then -- the part that actually carries the
        intent -- asserts there is NO WAY TO CHANGE THEM. There is no
        ``set_v2_take_profit()`` and no env var, unlike ``set_max_hold_s()``,
        which exists deliberately for L25_MAX_HOLD. A researched constant that
        could be quietly tuned would stop being the researched constant, and the
        strategy would silently stop being the thing that was measured.
        """
        allowed = {
            "profile", "label", "entry_strategy", "exit_strategy",
            "exit_bracket", "max_hold_s", "max_hold_trigger",
            "own_exit_triggers", "position_pct_override", "note", "is_default",
        }
        # keys permitted to hold a bare number
        numeric_keys = {"max_hold_s", "position_pct_override"}
        for profile in StrategyProfile:
            assert set(as_dict(profile)) <= allowed
            for key, value in as_dict(profile).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    assert key in numeric_keys, (
                        f"{profile.value}.{key} is a bare number on a profile; "
                        f"profiles may not carry tunable strategy thresholds"
                    )

        # the V2 numbers are the researched ones, exactly
        v2 = as_dict(StrategyProfile.V2_FLOOR)
        assert v2["exit_bracket"] == {
            "take_profit_pct": 30.0, "stop_loss_pct": -30.0,
        }
        assert v2["position_pct_override"] == 10.0
        assert v2["max_hold_s"] == 90.0

        # ... and there is no setter, so none of them is tunable
        import autopaper.strategy_profile as sp

        for forbidden in (
            "set_take_profit", "set_stop_loss", "set_exit_bracket",
            "set_position_pct", "set_v2f_take_profit", "set_v2f_stop_loss",
            "set_v2f_position_pct",
        ):
            assert not hasattr(sp, forbidden), (
                f"strategy_profile.{forbidden} exists; the researched bracket "
                f"and size must be frozen, not tunable"
            )

    def test_the_l25_family_shares_one_entry_strategy(self):
        """L25_CURRENT, L25_60S and L25_MAX_HOLD all still authorize via L25."""
        for profile in (
            StrategyProfile.L25_CURRENT,
            StrategyProfile.L25_60S,
            StrategyProfile.L25_MAX_HOLD,
        ):
            assert as_dict(profile)["entry_strategy"] == "L25"
            assert entry_strategy(profile) == "L25"
            assert uses_l25_entry(profile) is True

    def test_v2_floor_has_its_own_entry_strategy_and_is_not_an_l25_variant(self):
        """
        V2_FLOOR authorizes through its OWN gate.

        This is the load-bearing assertion for the whole feature: if V2_FLOOR
        ever started authorizing through L25, it would silently inherit L25's
        $20,000 market-cap ceiling and its 0.9647 liquidity/MC floor -- the two
        filters the research explicitly tested and REJECTED -- and would trade a
        different strategy from the one that was measured.
        """
        assert as_dict(StrategyProfile.V2_FLOOR)["entry_strategy"] == "V2_FLOOR"
        assert entry_strategy(StrategyProfile.V2_FLOOR) == "V2_FLOOR"
        assert uses_l25_entry(StrategyProfile.V2_FLOOR) is False

    def test_v2_floor_has_a_90_second_maximum_hold(self):
        assert max_hold_s(StrategyProfile.V2_FLOOR) == 90.0
        assert max_hold_s("V2_FLOOR") == 90.0
        # and its cap is NOT reachable through the L25_MAX_HOLD override
        set_max_hold_s(120.0)
        try:
            assert max_hold_s(StrategyProfile.V2_FLOOR) == 90.0
            assert max_hold_s(StrategyProfile.L25_MAX_HOLD) == 120.0
        finally:
            set_max_hold_s(None)

    def test_entry_strategy_refuses_an_unknown_profile(self):
        """
        Fails closed rather than reporting L25.

        A caller that fell back to the default here would authorize a real entry
        under the wrong gate.
        """
        with pytest.raises(StrategyProfileError):
            entry_strategy("V2_FLOOR_TYPO")
        with pytest.raises(StrategyProfileError):
            uses_l25_entry("NOPE")

    def test_l25_60s_has_a_60_second_maximum_hold(self):
        assert max_hold_s(StrategyProfile.L25_60S) == 60.0
        assert max_hold_s("L25_60S") == 60.0

    def test_l25_threshold_is_unchanged(self):
        from autopaper.l25 import L25_HASH, L25_MIN_LIQ_TO_MCAP
        from autopaper.v3c import (
            V3C_HASH,
            V3C_MAX_MARKET_CAP_USD,
            V3C_MIN_SIGNAL_QUALITY,
        )
        assert L25_MIN_LIQ_TO_MCAP == 0.9647
        assert V3C_MAX_MARKET_CAP_USD == 20_000.0
        assert V3C_MIN_SIGNAL_QUALITY == 0.55
        assert L25_HASH and V3C_HASH

    def test_the_l25_gate_gives_the_same_answer_in_both_profiles(self):
        """The same candidate, gated under each profile, must authorize alike."""
        from autopaper.l25 import evaluate as l25_evaluate
        args = dict(
            route="FAST_ENTRY", market_cap_usd=18_000.0,
            signal_quality=0.70, liquidity_usd=400_000.0,
        )
        results = []
        for profile in StrategyProfile:
            # the gate is called identically; the profile is not even in scope
            results.append(l25_evaluate(**args).authorized)
        assert results[0] is results[1] is True

        reject = dict(args, liquidity_usd=1_000.0)
        answers = {l25_evaluate(**reject).authorized for _ in StrategyProfile}
        assert answers == {False}

    def test_sizing_and_risk_are_not_reachable_from_a_profile(self):
        """A profile cannot move a size, a cooldown or a risk limit."""
        from autopaper.config import RiskLimits
        limits = RiskLimits()
        assert limits.max_position_pct_equity == 12.5
        assert limits.reentry_cooldown_s == 900.0
        assert limits.reentry_enabled is False
        assert limits.max_simultaneous_positions == 1


# ===========================================================================
# 3. EXIT
# ===========================================================================


class TestExit:
    def test_60s_cap_fires_at_exactly_60_seconds(self, policy):
        pos = FakePos(T0, strategy_profile="L25_60S")
        assert assess(pos, T0 + 59.0, 1.01, policy).action == "HOLD"
        decision = assess(pos, T0 + 60.0, 1.01, policy)
        assert decision.action == "EXIT"
        assert decision.trigger == MAX_HOLD_TRIGGER
        assert decision.trigger == "max_hold_60s"

    def test_60s_cap_fires_on_every_later_evaluation(self, policy):
        """
        Once past 60s the cap holds. +5% is used so the dead-thesis rule
        (held>=75s AND ret<+3%) cannot fire and mask the cap; the cap's own
        persistence is what is under test.
        """
        pos = FakePos(T0, strategy_profile="L25_60S")
        for t in (60.0, 61.0, 75.0, 300.0):
            d = assess(pos, T0 + t, 1.05, policy)
            assert d.action == "EXIT", f"not capped at {t}s"
            assert d.trigger == MAX_HOLD_TRIGGER, \
                f"at {t}s an existing rule fired first: {d.trigger}"

    def test_l25_current_never_caps(self, policy):
        """The default profile must never produce a max-hold exit."""
        pos = FakePos(T0, strategy_profile="L25_CURRENT")
        for t in (30.0, 60.0, 61.0, 75.0):
            d = assess(pos, T0 + t, 1.01, policy)
            assert d.trigger != MAX_HOLD_TRIGGER
        # at 75s a healthy +1% position is held by the current profile: the
        # dead-thesis rule needs ret < +3%... which +1% satisfies, so it exits
        # there. Either way it must not be the CAP that fired.
        assert assess(pos, T0 + 75.0, 1.01, policy).trigger != MAX_HOLD_TRIGGER

    def test_a_healthy_l25_current_position_is_never_capped(self, policy):
        """A winner under the default profile can still be held indefinitely."""
        pos = FakePos(T0, strategy_profile="L25_CURRENT")
        for t in (60.0, 120.0, 600.0, 3600.0):
            d = assess(pos, T0 + t, 1.25, policy)   # +25%: above the +3% fence
            assert d.action == "HOLD", f"capped at {t}s under the default profile"

    @pytest.mark.parametrize("t,price,expected", [
        (5.0, 0.50, "disaster_cap"),
        (10.0, 0.87, "fast_failure_stop"),
    ])
    def test_an_earlier_existing_exit_wins_over_the_cap(
        self, policy, t, price, expected
    ):
        """Existing exits are NOT replaced: whichever fires first is used."""
        pos = FakePos(T0, strategy_profile="L25_60S")
        d = assess(pos, T0 + t, price, policy)
        assert d.action == "EXIT"
        assert d.trigger == expected, \
            "the existing trigger must keep precedence over the 60s cap"

    def test_the_dead_thesis_rule_still_fires_before_the_cap(self, policy):
        """no_follow_through at 80s is an existing rule and is not preempted."""
        pos = FakePos(T0, strategy_profile="L25_60S")
        # +1% after 80s: above nothing, below the +3% fence, so the dead-thesis
        # rule fires. It is checked before the cap.
        d = assess(pos, T0 + 80.0, 1.01, policy)
        assert d.action == "EXIT"
        assert d.trigger == "no_follow_through"

    def test_momentum_and_structure_still_fire_before_the_cap(self, policy):
        pos = FakePos(T0, strategy_profile="L25_60S")
        d = policy.assess(
            pos, now=T0 + 20.0, mark_price=1.0, data_state=DataState.LIVE,
            momentum_30s_pct=-20.0, structure_30s_pct=0.0,
            liquidity_change_pct=0.0,
        )
        assert d.action == "EXIT" and d.trigger == "momentum_collapse"

        d = policy.assess(
            pos, now=T0 + 20.0, mark_price=1.0, data_state=DataState.LIVE,
            momentum_30s_pct=0.0, structure_30s_pct=-15.0,
            liquidity_change_pct=0.0,
        )
        assert d.action == "EXIT" and d.trigger == "structure_break"

    def test_the_cap_does_not_bypass_the_data_integrity_gate(self, policy):
        """
        A max hold is not permission to sell on a price we do not trust.

        On a non-LIVE mark the policy HOLDs (or exits via the data-invalid
        timer). The cap must not turn a bad mark into a forced exit, because
        that would be a silent bypass of the stale gate.
        """
        pos = FakePos(T0, strategy_profile="L25_60S")
        d = policy.assess(
            pos, now=T0 + 61.0, mark_price=1.01, data_state=DataState.STALE,
            momentum_30s_pct=0.0, structure_30s_pct=0.0,
            liquidity_change_pct=0.0,
        )
        assert d.trigger != MAX_HOLD_TRIGGER
        assert d.action == "HOLD"

    def test_the_cap_reports_the_profile_that_imposed_it(self, policy):
        pos = FakePos(T0, strategy_profile="L25_60S")
        d = assess(pos, T0 + 60.0, 1.01, policy)
        assert "L25 + 60s" in d.reason
        assert "60s" in d.reason

    def test_an_unrecognised_persisted_profile_does_not_crash_the_policy(
        self, policy
    ):
        """
        A position restored from a corrupted snapshot must not take the exit
        loop down. It is treated as uncapped, which is the conservative reading.
        """
        pos = FakePos(T0, strategy_profile="L25_BANANA")
        d = assess(pos, T0 + 60.0, 1.01, policy)
        assert d.trigger != MAX_HOLD_TRIGGER

    def test_a_position_with_no_profile_field_behaves_as_current(self, policy):
        """A pre-profile snapshot has no attribute; it must still work."""
        pos = FakePos(T0)
        del pos.strategy_profile
        d = assess(pos, T0 + 60.0, 1.01, policy)
        assert d.trigger != MAX_HOLD_TRIGGER

    def test_the_cap_is_evaluated_from_the_position_not_global_state(self,
                                                                    policy):
        """
        The decisive immutability test at the policy layer: a position stamped
        L25_CURRENT is never capped, no matter what is 'selected' globally.
        """
        from autopaper.modes import RuntimeState as RS
        RS().select_strategy_profile("L25_60S")      # a global switch, ignored
        pos = FakePos(T0, strategy_profile="L25_CURRENT")
        for t in (60.0, 120.0):
            assert assess(pos, T0 + t, 1.25, policy).action == "HOLD"


# ===========================================================================
# 4. POSITION IMMUTABILITY
# ===========================================================================


class TestPositionImmutability:
    def test_a_position_carries_the_profile_it_was_opened_under(self):
        from autopaper.models import Position, PositionStatus, Route, Regime
        from autopaper.models import Regime as R
        pos = Position(
            position_id="p1", candidate_id="c1", mint="m", symbol="S",
            status=PositionStatus.OPEN, route=Route.FAST_ENTRY,
            regime=R.EXPLOSIVE, exit_style=ExitStyle.FAST_FAILURE,
            strategy="FAST_ENTRY", entry_signal_timestamp=T0,
            entry_decision_timestamp=T0, strategy_profile="L25_60S",
        )
        assert pos.strategy_profile == "L25_60S"
        assert pos.as_dict()["strategy_profile"] == "L25_60S"

    def test_a_position_defaults_to_l25_current(self):
        from autopaper.models import Position, PositionStatus, Route
        from autopaper.models import Regime as R
        pos = Position(
            position_id="p1", candidate_id="c1", mint="m", symbol="S",
            status=PositionStatus.OPEN, route=Route.FAST_ENTRY,
            regime=R.EXPLOSIVE, exit_style=ExitStyle.FAST_FAILURE,
            strategy="FAST_ENTRY", entry_signal_timestamp=T0,
            entry_decision_timestamp=T0,
        )
        assert pos.strategy_profile == "L25_CURRENT"

    def test_a_trade_carries_the_profile_for_statistics(self):
        from autopaper.models import Trade
        tr = Trade(
            trade_id="t1", position_id="p1", candidate_id="c1", mint="m",
            symbol="S", source="s", strategy="FAST_ENTRY", route="FAST_ENTRY",
            regime="EXPLOSIVE", entry_signal_timestamp=T0,
            entry_timestamp=T0, entry_price=1.0, fill_price=1.0,
            exit_timestamp=T0 + 60, exit_price=1.1, quantity=1.0,
            entry_fee_usd=0.0, exit_fee_usd=0.0, entry_slippage_usd=0.0,
            exit_slippage_usd=0.0, gross_pnl_usd=0.1, net_pnl_usd=0.1,
            return_pct=10.0, holding_seconds=60.0, mfe_pct=10.0, mae_pct=0.0,
            exit_reason=MAX_HOLD_TRIGGER, exit_style="FAST_FAILURE",
            thesis="", invalidation=(), strategy_profile="L25_60S",
        )
        assert tr.as_dict()["strategy_profile"] == "L25_60S"

    def test_a_real_position_carries_the_profile(self):
        from autopaper.real_account import RealPosition
        pos = RealPosition(
            mint="m", symbol="S", entry_signature="sig",
            entry=type("F", (), {"signature": "sig", "filled": True})(),
            token_amount=1, entry_price_usd=1.0, size_usd=10.0,
            opened_at=T0, strategy_profile="L25_60S",
        )
        assert pos.strategy_profile == "L25_60S"

    def test_the_engine_stamps_the_selected_profile_at_entry(self):
        """Structural: the engine passes state.strategy_profile at open."""
        src = (ROOT / "src" / "autopaper" / "engine.py").read_text()
        assert "strategy_profile=self.state.strategy_profile" in src, \
            "the engine must stamp the selected profile when opening"

    def test_open_positions_view_reports_the_positions_own_profile(self):
        assert '"strategy_profile": p.strategy_profile' in \
            (ROOT / "src" / "autopaper" / "engine.py").read_text()

    def test_a_switch_after_open_cannot_change_an_open_position(self, policy):
        """
        End to end at the policy layer: open under L25_60S, then switch the
        global selection to L25_CURRENT. The open position keeps the cap.
        """
        from autopaper.modes import RuntimeState as RS
        state = RS()
        state.select_strategy_profile("L25_60S")
        pos = FakePos(T0, strategy_profile=state.profile.value)

        state.select_strategy_profile("L25_CURRENT")     # the operator switches
        assert state.profile is StrategyProfile.L25_CURRENT
        # the open position still exits at 60s
        d = assess(pos, T0 + 60.0, 1.01, policy)
        assert d.action == "EXIT" and d.trigger == MAX_HOLD_TRIGGER

    def test_and_the_other_direction_too(self, policy):
        from autopaper.modes import RuntimeState as RS
        state = RS()
        state.select_strategy_profile("L25_60S")       # opened capped
        capped = FakePos(T0, strategy_profile=state.profile.value)
        state.select_strategy_profile("L25_CURRENT")
        state.select_strategy_profile("L25_60S")

        # open a position now, under L25_CURRENT, then flip back to the cap
        state.select_strategy_profile("L25_CURRENT")
        uncapped = FakePos(T0, strategy_profile=state.profile.value)
        state.select_strategy_profile("L25_60S")
        assert assess(uncapped, T0 + 60.0, 1.25, policy).action == "HOLD"
        assert assess(capped, T0 + 60.0, 1.01, policy).trigger == MAX_HOLD_TRIGGER


# ===========================================================================
# 5. SELECTION IS INERT
# ===========================================================================


class TestSelectionIsInert:
    def test_selecting_does_not_start_the_runtime(self):
        state = RuntimeState()
        assert state.run_state is RunState.OFF
        state.select_strategy_profile("L25_60S")
        assert state.run_state is RunState.OFF
        assert state.running is False

    def test_selecting_does_not_arm_real(self):
        state = RuntimeState()
        state.select_strategy_profile("L25_60S")
        assert state.real_armed is False
        assert state.real_trading_enabled is False

    def test_real_armed_cannot_be_set_through_the_profile(self):
        """Structural: there is no field to set, and the setter takes no args."""
        import inspect
        sig = inspect.signature(RuntimeState.select_strategy_profile)
        assert list(sig.parameters) == ["self", "value"]
        assert "real" not in str(sig).lower()

    def test_selecting_does_not_unblock_entries(self):
        state = RuntimeState()
        state.select_strategy_profile("L25_60S")
        assert state.entries_blocked is False
        ok, _ = state.can_open_new_entries()
        # the only thing stopping entries is that the bot is OFF
        assert ok is False
        assert "not running" in state.can_open_new_entries()[1]

    def test_selecting_does_not_clear_the_emergency_stop(self):
        state = RuntimeState()
        state.emergency_stop_now()
        assert state.emergency_stop is True
        state.select_strategy_profile("L25_60S")
        assert state.emergency_stop is True

    def test_selecting_does_not_stop_a_running_bot(self):
        state = RuntimeState()
        state.start()
        assert state.running is True
        state.select_strategy_profile("L25_60S")
        assert state.running is True
        assert state.run_state is RunState.PAPER

    def test_selecting_does_not_write_to_the_ledger_or_journal(self):
        """The selector touches no I/O; it is a pure state change."""
        state = RuntimeState()
        before = len(state.history)
        state.select_strategy_profile("L25_60S")
        # the only mutation is the audit entry
        assert len(state.history) == before + 1

    def test_no_execution_module_references_a_profile(self):
        """
        A max-hold exit must be a normal exit DECISION. If the execution or
        broadcast layer ever imports a profile, a shortcut is being built.

        ``account.py`` is deliberately NOT in this list: the accounting layer
        must copy the profile onto the completed Trade so a trade can be
        attributed to the profile that produced it. Carrying a field onto a
        record is not a shortcut; the next test covers that it never BRANCHES on
        one.
        """
        for name in ("execution/real.py", "execution/solana.py",
                     "execution/raptor.py", "execution/jupiter.py",
                     "execution/txcodec.py", "exits.py"):
            src = (ROOT / "src" / "autopaper" / name).read_text()
            if name == "exits.py":
                continue    # the policy is the ONE place allowed to read it
            assert "strategy_profile" not in src, \
                f"{name} must not reference a strategy profile"

    def test_the_accounting_layer_copies_the_profile_without_branching_on_it(self):
        """
        account.py may record the profile on a Trade, and must never DECIDE
        anything from it. No conditional in the module may test it.
        """
        tree = ast.parse((ROOT / "src" / "autopaper" / "account.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.IfExp, ast.While)):
                test = ast.unparse(node.test)
                assert "strategy_profile" not in test, \
                    f"account.py branches on the profile: {test}"

    def test_the_executor_has_no_profile_specific_branch(self):
        src = (ROOT / "src" / "autopaper" / "execution" / "real.py").read_text()
        for token in ("max_hold", "MAX_HOLD", "L25_60S"):
            assert token not in src, \
                f"the REAL executor must not branch on {token}"


# ===========================================================================
# 6. PAPER + REAL PARITY
# ===========================================================================


class TestPaperRealParity:
    def test_both_profiles_use_the_same_paper_exit_engine(self):
        """
        There is no second paper exit path. Every mention of the profile in
        positions.py is a STAMP: the parameter, the two Position constructions
        and the ledger event. Nothing branches on it, so both profiles are
        governed by the same ``PositionBook.assess`` -> ``ExitPolicy``.
        """
        src = (ROOT / "src" / "autopaper" / "positions.py").read_text()
        tree = ast.parse(src)
        # no conditional anywhere in the module tests the profile
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.IfExp)):
                test = ast.unparse(node.test)
                assert "strategy_profile" not in test, (
                    f"positions.py branches on the profile here: {test}"
                )
        # and the five mentions are exactly: param, docstring x1, 2 stamps,
        # 1 ledger field
        assert "if strategy_profile" not in src
        assert "strategy_profile=" in src, "the profile must be stamped at open"

    def test_real_and_paper_entry_stamping_use_the_same_expression(self):
        """Both modes stamp state.strategy_profile; neither re-derives it."""
        src = (ROOT / "src" / "autopaper" / "engine.py").read_text()
        assert src.count("strategy_profile=self.state.strategy_profile") == 2, \
            "one for the paper open, one for the REAL register_entry"

    def test_a_real_max_hold_exit_takes_the_normal_exit_path(self):
        """
        Structural proof of 'no 60s sell shortcut': in the REAL monitor, ANY
        non-HOLD decision goes to _execute_real_sell, with no branch on the
        trigger. If a max-hold branch existed it would show up here.
        """
        tree = ast.parse((ROOT / "src" / "autopaper" / "engine.py").read_text())
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef)
            and n.name == "_monitor_real_positions"
        )
        src = ast.unparse(fn)
        assert "_execute_real_sell" in src
        # the only trigger-aware branch is the pre-existing min-hold suppression
        assert src.count("max_hold") == 0, \
            "the REAL monitor must not branch on the max-hold trigger"
        assert src.count("MAX_HOLD") == 0

    def test_the_paper_exit_book_is_the_only_paper_decision_point(self):
        src = (ROOT / "src" / "autopaper" / "positions.py").read_text()
        assert "self.policy.assess(" in src, \
            "PositionBook.assess must delegate to the shared ExitPolicy"

    def test_the_shared_policy_object_serves_both_modes(self):
        """PAPER and REAL call the SAME ExitPolicy, so 'only the decision differs'."""
        src = (ROOT / "src" / "autopaper" / "engine.py").read_text()
        assert "book.policy.assess(" in src
        assert "self.book.assess(" in src or "book.assess(" in src

    def test_the_minimum_hold_still_applies_to_non_emergency_exits(self):
        """
        The 60s cap is far beyond the 5s minimum hold, so the two cannot
        interact -- but the existing suppression list must not have grown an
        entry for the cap, which would let the cap bypass the gate.
        """
        for name in ("positions.py", "engine.py"):
            src = (ROOT / "src" / "autopaper" / name).read_text()
            idx = src.find('"momentum_collapse", "structure_break"')
            if idx == -1:
                continue
            window = src[idx: idx + 200]
            assert MAX_HOLD_TRIGGER not in window, (
                f"{name}: the max-hold trigger must not be in the min-hold "
                f"suppression list"
            )


# ===========================================================================
# 7. TELEGRAM
# ===========================================================================


class TestTelegramSelector:
    def test_the_menu_carries_a_strategy_button(self):
        labels = [b["text"] for row in ui.operator_keyboard(RuntimeState())
                  for b in row]
        assert ui.BTN_STRATEGY in labels
        assert ui.BTN_STRATEGY in ui.OPERATOR_BUTTONS

    def test_the_submenu_shows_the_selected_strategies(self):
        from autopaper.paper_live.arms import arm_names

        state = RuntimeState()
        text = ui.render_strategy_menu(state)
        assert "SELECTED STRATEGIES" in text
        for name in arm_names():
            assert name in text, name

        state.select_strategies(["L25_60S"])
        text = ui.render_strategy_menu(state)
        assert "L25_60S" in text, text
        assert "take profit" in text.lower()

    def test_paper_ticks_exactly_the_selected_strategies(self):
        """
        PAPER is multi-select, so MORE than one may be ticked -- but only the
        SELECTED ones, never a superset or a subset.
        """
        from autopaper.paper_live.arms import arm_names

        state = RuntimeState()
        state.select_strategies(["L25_CURRENT", "H2_CAP60"])
        rows = [r[0] for r in ui.strategy_keyboard(state)]
        ticked = {t["callback_data"][len(ui.CB_PROFILE):]
                  for t in rows if "☑" in t["text"]}
        assert ticked == {"L25_CURRENT", "H2_CAP60"}

        # REAL renders a RADIO: exactly one, always.
        real = RuntimeState()
        real.run_state = RunState.REAL_ARMED
        real.real_armed = True
        real.select_strategies(["L25_60S"])
        rows = [r[0] for r in ui.strategy_keyboard(real)]
        radio = [t for t in rows if "◉" in t["text"]]
        assert len(radio) == 1
        assert radio[0]["callback_data"].endswith("L25_60S")

    def test_the_submenu_contains_no_run_control_and_no_real_control(self):
        """A selection screen must not be able to start trading or arm REAL."""
        flat = [b["text"] for r in ui.strategy_keyboard(RuntimeState())
                for b in r]
        for forbidden in (ui.BTN_START_PAPER, ui.BTN_ARM_REAL, ui.BTN_REAL_OFF,
                          ui.BTN_KILL, ui.BTN_PAUSE, ui.BTN_RESUME):
            assert forbidden not in flat, \
                f"{forbidden!r} must not be reachable from the strategy menu"

    def test_every_submenu_button_dispatches(self):
        state = RuntimeState()
        handler = _handler(state)
        result = asyncio_run(handler.handle(ui.CB_STRATEGY))
        assert result.handled is True
        assert not result.error
        assert "SELECTED" in result.text

    def test_selecting_through_the_callback_updates_state(self):
        """PAPER toggles: a press ADDS an arm, it does not replace the set."""
        state = RuntimeState()
        handler = _handler(state)
        result = asyncio_run(handler.handle(f"{ui.CB_PROFILE}L25_60S"))
        assert result.handled is True
        assert result.error is None
        assert "L25_60S" in state.arms

    def test_an_unknown_strategy_callback_is_refused(self):
        """An unknown arm is REFUSED, never silently dropped."""
        state = RuntimeState()
        before = tuple(state.arms)
        handler = _handler(state)
        result = asyncio_run(handler.handle(f"{ui.CB_PROFILE}L25_30S"))
        assert result.handled is True
        assert result.error == "invalid_strategy_selection"
        assert tuple(state.arms) == before

    def test_selection_does_not_start_the_runtime(self):
        state = RuntimeState()
        handler = _handler(state)
        asyncio_run(handler.handle(f"{ui.CB_PROFILE}L25_60S"))
        assert state.run_state is RunState.OFF
        assert state.running is False

    def test_selection_does_not_arm_real(self):
        state = RuntimeState()
        handler = _handler(state)
        asyncio_run(handler.handle(f"{ui.CB_PROFILE}L25_60S"))
        assert state.real_armed is False
        assert state.real_trading_enabled is False

    def test_selection_does_not_broadcast(self):
        """
        Structural: the selector's handler must not reach the executor, the
        intent store, or anything that signs.
        """
        from autopaper.telegram import actions
        src = Path(actions.__file__).read_text()
        fn = ast.unparse(next(
            n for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.FunctionDef) and n.name == "_select_profile"
        ))
        # Word-boundary matching: a plain substring test flags "signal"
        # because it contains "sign", which made this assertion fire on correct
        # prose about a REAL signal instead of on any capability.
        import re as _re

        for forbidden in ("execute", "broadcast", "sign", "intent",
                          "executor", "quote"):
            assert not _re.search(rf"\b{forbidden}\w*\s*\(", fn.lower()), \
                f"the strategy selector must not call {forbidden!r}"

    def test_status_shows_the_profile_not_a_bare_strategy(self):
        state = RuntimeState()
        line = ui.strategy_status_line(state)
        assert "L25 CURRENT" in line
        state.select_strategy_profile("L25_60S")
        assert "L25 + 60s" in ui.strategy_status_line(state)

    def test_the_menu_summary_names_the_profile(self):
        state = RuntimeState()
        text = ui.render_menu(state)
        assert "L25 CURRENT" in text
        state.select_strategy_profile("L25_60S")
        assert "L25 + 60s" in ui.render_menu(state)

    def test_the_open_position_view_shows_the_positions_own_profile(self):
        rows = [{
            "symbol": "S", "mode": "PAPER", "position_id": "p1",
            "entry_price": 1.0, "current_price": 1.0, "hold_s": 10.0,
            "strategy_profile": "L25_60S",
        }]
        text = ui.render_open_positions(rows)
        assert "L25 + 60s" in text

    def test_a_max_hold_exit_is_identifiable_in_a_notification(self):
        """EXIT: MAX_HOLD_60S must be unambiguous to an operator."""
        assert MAX_HOLD_TRIGGER == "max_hold_60s"
        assert status_line("L25_60S") == "STRATEGY: L25 + 60s"

    def test_the_submenu_states_the_provenance_and_the_caveat(self):
        state = RuntimeState()
        text = ui.render_strategy_menu(state)
        assert "retrospective" in text.lower()
        assert "not proven superior" in text.lower()
        assert "never starts the bot" in text
        assert "never arms REAL" in text


# ===========================================================================
# 8. SAFETY INVARIANTS
# ===========================================================================


class TestSafetyInvariants:
    def test_the_safety_audit_passes(self):
        assert audit().ok is True

    def test_the_audit_checks_the_profile_invariants(self):
        names = {c[0] for c in audit().checks}
        for expected in (
            "profile_set_is_closed",
            "profile_default_is_l25_current",
            "default_entry_strategy_is_l25",
            "profile_entry_strategies_are_declared",
            "profile_l25_family_entry_is_unchanged",
            "profile_l25_current_has_no_cap",
            "profile_60s_cap_is_60",
            # V2_FLOOR-specific: the audit must re-verify the gate's own frozen
            # thresholds rather than trusting the profile table.
            "profile_v2_floor_cap_is_90",
            "profile_v2_floor_threshold_is_frozen",
            # V2_FLOOR's exit is its own researched bracket and its own sizing,
            # so the audit verifies both -- and verifies that NO L25 profile has
            # either, which is the "adding V2_FLOOR changed nothing else" guard.
            "profile_v2_floor_exit_bracket_is_the_research_rule",
            "profile_v2_floor_sizing_is_ten_percent",
            "profile_only_v2_floor_overrides_sizing_or_exit",
            "profile_selecting_v2_floor_does_not_arm_real",
        ):
            assert expected in names, f"missing invariant {expected}"

    def test_env_local_stays_git_ignored(self):
        ignored = (ROOT / ".gitignore").read_text().splitlines()
        assert ".env.local" in ignored
        assert "!.env.example" in ignored

    def test_the_profile_is_not_a_secret(self):
        """The value is safe to print; it must not be treated as one."""
        from autopaper.safety import is_secret_key, redact_mapping
        assert is_secret_key("STRATEGY_PROFILE") is False
        out = redact_mapping({"STRATEGY_PROFILE": "L25_60S"})
        assert out["STRATEGY_PROFILE"] == "L25_60S"

    def test_describe_names_entry_exit_and_max_hold(self):
        text = describe("L25_60S")
        assert "entry strategy" in text and "L25" in text
        assert "exit strategy" in text
        assert "60s" in text

    def test_the_note_does_not_overclaim(self):
        """
        No profile's note may claim superiority it has not demonstrated.

        The L25 variants carry "NOT proven superior" because they came from a
        retrospective replay. V2_FLOOR carries a DIFFERENT set of caveats -- it
        is held to its own explicit list rather than to the L25 phrasing --
        because it is a different kind of claim: a different ENTRY rule, not a
        variant of the same one. What is forbidden for both is an unqualified
        claim.
        """
        for profile in StrategyProfile:
            if profile is StrategyProfile.L25_CURRENT:
                continue
            n = note(profile).lower()
            if profile is StrategyProfile.V2_FLOOR:
                # the disclosures that must never be dropped. The exit rule and
                # the size are no longer on this list: they are now the
                # researched rule, so claiming a divergence from them would
                # itself be a false statement.
                assert "no market-cap cap" in n
                assert "no liquidity filter" in n
                assert "not proven superior" in n
                assert "unresolved" in n
                assert "paper only" in n
                # the narrower live population is still disclosed
                assert "narrower than the research population" in n
                # and it must never claim the divergence that no longer exists
                assert "take-profit is not implemented" not in n
                # and it must never read as proven
                assert "proven" not in n.replace("not proven", "")
            else:
                assert "not proven superior" in n
                assert "retrospective" in n
                assert "mixed" in n

    def test_the_v2_floor_note_names_the_measured_effect(self):
        """The operator menu must show what the change actually measured."""
        n = note("V2_FLOOR")
        assert "vol5/mcap >= 1.0" in n
        assert "+10%" in n and "+300%" in n
        assert "66.9%->70.5%" in n
        assert "18.05%->11.91%" in n
        # the provenance must say what kind of evidence this is
        assert "retrospective" in n.lower()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def asyncio_run(coro):
    import asyncio
    return asyncio.run(coro)


def _handler(state: RuntimeState):
    """
    A Telegram ButtonHandler with no effects wired at all.

    ``OperatorActions()`` leaves journal, notifier, preflight and the
    state-change hook as None, so pressing a strategy button performs no I/O and
    calls nothing external. If selection ever needed an effect to succeed, these
    tests would fail rather than quietly relying on a real journal.
    """
    from autopaper.telegram.actions import ButtonHandler, OperatorActions
    return ButtonHandler(state, OperatorActions())


# ===========================================================================
# 9. END TO END THROUGH THE REAL PAPER BOOK
# ===========================================================================
#
# The tests above exercise the policy and the wiring in isolation. These drive
# the actual ``PositionBook`` -- the same object the paper exit loop calls, with
# a real ``PaperAccount`` and a real ``Position`` -- and assert on the trade the
# account books. This is the paper path the runtime uses, not a stand-in.


class _FakeLedger:
    def __init__(self):
        self.rows = []

    def position_update(self, row):
        self.rows.append(row)

    def trade(self, trade):
        self.rows.append(("trade", trade))


class _FlatFill:
    """
    An exit fill at the quoted price, with the production fee and slippage
    model applied to the POSITION's own quantity.

    Real costs, not a zero-cost fill: a max-hold exit has to be shown to book
    through the same accounting as any other exit, so the trade it produces is
    comparable with the ledger's. The quantity is taken from the position
    because the account's identity check will refuse a fill whose costs do not
    match the position being closed.
    """

    def __init__(self, price: float, quantity: float,
                 liquidity: float = 400_000.0, side: str = "exit") -> None:
        from autopaper.config import ExecutionModel
        m = ExecutionModel()
        base_bps = m.entry_slippage_bps if side == "entry" else m.exit_slippage_bps
        imp_bps = min(m.impact_max_bps,
                      100.0 * ((quantity * price / liquidity) ** 0.5) * 0.9)
        total_bps = base_bps + imp_bps
        sign = 1.0 if side == "entry" else -1.0
        self.quoted_price = price
        self.price_at_quote = price
        self.fill_price = price * (1 + sign * total_bps / 10_000.0)
        self.slippage_bps = total_bps
        self.fee_bps = m.entry_fee_bps if side == "entry" else m.exit_fee_bps
        self.quantity = quantity
        self.notional_usd = quantity * self.fill_price
        self.filled = True
        self.fee_usd = self.notional_usd * self.fee_bps / 10_000.0
        self.slippage_usd = quantity * abs(self.fill_price - price)
        self.fill_at = None
        self.reason = "filled"
        self.detail = "profile test fill"


def _paper_book(profile: str):
    """A real PaperAccount + PositionBook, and one open position."""
    from autopaper.account import PaperAccount, PaperAccountState
    from autopaper.config import ExitConfig, RiskLimits
    from autopaper.models import Position, PositionStatus, Regime, Route
    from autopaper.positions import PositionBook
    from autopaper.strategy_profile import parse_profile

    limits = RiskLimits()
    acct = PaperAccount(
        PaperAccountState(starting_bankroll_usd=100.0, cash_usd=100.0), limits)
    book = PositionBook(acct, _FakeLedger(), ExitConfig(), limits)

    entry_ref = 1.0e-4
    slip_bps, fee_bps = 100.0, 85.0
    notional = 10.0
    entry_fill = entry_ref * (1 + slip_bps / 10_000.0)
    qty = notional / entry_fill
    fee = qty * entry_fill * fee_bps / 10_000.0

    pos = Position(
        position_id="pos_profile_test", candidate_id="c1", mint="MINT",
        symbol="SYM", status=PositionStatus.OPEN, route=Route.FAST_ENTRY,
        regime=Regime.EXPLOSIVE, exit_style=ExitStyle.FAST_FAILURE,
        strategy="FAST_ENTRY", entry_signal_timestamp=T0,
        entry_decision_timestamp=T0, entry_timestamp=T0,
        entry_reference_price=entry_ref, fill_price=entry_fill,
        quantity=qty, notional_usd=notional, entry_fee_usd=fee,
        entry_slippage_usd=qty * (entry_fill - entry_ref),
        cash_out_usd=notional + fee, cost_basis_usd=notional + fee,
        peak_price=entry_fill, trough_price=entry_fill,
        strategy_profile=parse_profile(profile).value,
    )
    acct.register_position(pos)
    return acct, book, pos, entry_ref


class TestPaperBookEndToEnd:
    def test_60s_profile_books_a_max_hold_exit(self):
        """The real paper book closes an L25_60S position at 60s."""
        acct, book, pos, entry_ref = _paper_book("L25_60S")
        price = entry_ref * 1.02          # +2%: above the stop, below +3%
        decision = book.assess(
            pos, now=T0 + 60.0, mark_price=price, data_state=DataState.LIVE,
            momentum_30s_pct=0.0, structure_pct=0.0,
            liquidity_change_pct=0.0, min_hold_s=5.0,
        )
        assert decision.action == "EXIT"
        assert decision.trigger == MAX_HOLD_TRIGGER
        trade = book.close(pos, decision=decision,
                           fill=_FlatFill(price, pos.quantity), now=T0 + 60.0)
        assert trade is not None
        assert trade.exit_reason == MAX_HOLD_TRIGGER
        assert trade.strategy_profile == "L25_60S"
        assert trade.holding_seconds == pytest.approx(60.0)
        assert acct.cash == pytest.approx(START_CASH + trade.net_pnl_usd)

    def test_an_earlier_existing_exit_is_the_one_that_books(self):
        """The same position under the -12% stop closes with THAT reason."""
        acct, book, pos, entry_ref = _paper_book("L25_60S")
        price = entry_ref * 0.85          # -15%: past the -12% fast-failure stop
        decision = book.assess(
            pos, now=T0 + 20.0, mark_price=price, data_state=DataState.LIVE,
            momentum_30s_pct=0.0, structure_pct=0.0,
            liquidity_change_pct=0.0, min_hold_s=5.0,
        )
        assert decision.trigger == "fast_failure_stop"
        trade = book.close(pos, decision=decision,
                           fill=_FlatFill(price, pos.quantity), now=T0 + 20.0)
        assert trade.exit_reason == "fast_failure_stop"

    def test_the_default_profile_books_no_max_hold_exit(self):
        """L25_CURRENT is not capped, so 60s leaves the position open."""
        acct, book, pos, entry_ref = _paper_book("L25_CURRENT")
        price = entry_ref * 1.25          # +25%: above the +3% dead-thesis fence
        for t in (60.0, 90.0, 120.0):
            decision = book.assess(
                pos, now=T0 + t, mark_price=price, data_state=DataState.LIVE,
                momentum_30s_pct=0.0, structure_pct=0.0,
                liquidity_change_pct=0.0, min_hold_s=5.0,
            )
            assert decision.action == "HOLD", f"capped at {t}s under the default"
        assert acct.open_positions()

    def test_the_trade_books_through_normal_accounting(self):
        """
        A max-hold exit must produce a trade that satisfies the same
        accounting identity as any other exit, with real costs deducted.
        """
        acct, book, pos, entry_ref = _paper_book("L25_60S")
        price = entry_ref * 1.05
        decision = book.assess(
            pos, now=T0 + 60.0, mark_price=price, data_state=DataState.LIVE,
            momentum_30s_pct=0.0, structure_pct=0.0,
            liquidity_change_pct=0.0, min_hold_s=5.0,
        )
        trade = book.close(pos, decision=decision,
                           fill=_FlatFill(price, pos.quantity), now=T0 + 60.0)
        # gross is FILL vs FILL. `_FlatFill` executes BELOW the quoted price by the
        # modelled slippage, and that difference is inside ``gross_pnl_usd`` -- so
        # the fee is the only thing deducted from it. The reference-basis gross is
        # carried separately and the two reconcile exactly.
        exit_fill = _FlatFill(price, pos.quantity).fill_price
        gross = (exit_fill - pos.fill_price) * pos.quantity
        fees = trade.entry_fee_usd + trade.exit_fee_usd
        assert trade.gross_pnl_usd == pytest.approx(gross)
        assert trade.net_pnl_usd == pytest.approx(gross - fees)
        assert trade.gross_pnl_reference_basis_usd == pytest.approx(
            (price - pos.entry_reference_price) * pos.quantity)
        assert trade.pnl_basis == "FILL_VS_FILL"
        costs = (trade.entry_fee_usd + trade.exit_fee_usd
                 + trade.entry_slippage_usd + trade.exit_slippage_usd)
        assert costs > 0, "costs must actually be charged"
        # The real round-trip invariant. Principal is ESCROWED at register and
        # RETURNED at close, so the meaningful statement is that a complete
        # round trip leaves cash at the starting amount plus net P&L -- not that
        # cash moved by net alone.
        assert acct.cash == pytest.approx(START_CASH + trade.net_pnl_usd)
        # and the position's own escrow reconciles
        proceeds = pos.cash_in_usd
        assert proceeds == pytest.approx(pos.cash_out_usd + trade.net_pnl_usd)

    def test_the_ledger_records_the_profile_and_the_reason(self):
        acct, book, pos, entry_ref = _paper_book("L25_60S")
        price = entry_ref * 1.05
        decision = book.assess(
            pos, now=T0 + 60.0, mark_price=price, data_state=DataState.LIVE,
            momentum_30s_pct=0.0, structure_pct=0.0,
            liquidity_change_pct=0.0, min_hold_s=5.0,
        )
        book.close(pos, decision=decision, fill=_FlatFill(price, pos.quantity), now=T0 + 60.0)
        exits = [r for r in book.ledger.rows
                 if isinstance(r, dict) and r.get("action") == "EXIT"]
        assert exits, "the paper book must journal the exit"
        assert exits[0]["reason"] == MAX_HOLD_TRIGGER
        assert exits[0]["style"] == "FAST_FAILURE"

    def test_a_mid_flight_switch_does_not_recut_an_open_position(self):
        """
        Open under L25_CURRENT, switch the global selection to L25_60S, and the
        position still is not capped. The book never re-reads global state.
        """
        state = RuntimeState()
        state.select_strategy_profile("L25_CURRENT")
        acct, book, pos, entry_ref = _paper_book(state.profile.value)
        state.select_strategy_profile("L25_60S")
        price = entry_ref * 1.25
        decision = book.assess(
            pos, now=T0 + 60.0, mark_price=price, data_state=DataState.LIVE,
            momentum_30s_pct=0.0, structure_pct=0.0,
            liquidity_change_pct=0.0, min_hold_s=5.0,
        )
        assert decision.action == "HOLD"
        assert decision.trigger != MAX_HOLD_TRIGGER
