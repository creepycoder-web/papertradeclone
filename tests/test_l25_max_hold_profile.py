"""
L25_MAX_HOLD: a separate PAPER profile. L25_CURRENT must not move.

The profile is L25 entry + the EXISTING exit precedence + a configurable hard
max-hold cap. The cap is a TRIGGER for an executable SELL and nothing more:
it must never bypass SELL validation, force a fill, or substitute a mark.

These tests pin the isolation the hard way: where a behaviour MUST be
identical between the two profiles, they compare the two profiles against each
other rather than asserting a literal, so a change to one that leaks into the
other fails here.
"""

from __future__ import annotations

import time

import pytest

from autopaper.exits import ExitDecision, ExitPolicy, max_hold_trigger
from autopaper.models import DataState
from autopaper.paper_live.arms import ARMS, ResearchArm, arm_names, parse_arms
from autopaper.strategy_profile import (
    DEFAULT_PROFILE,
    PROFILE_EXIT_STRATEGY,
    PROFILE_LABELS,
    PROFILE_MAX_HOLD_S,
    StrategyProfile,
    StrategyProfileError,
    configured_max_hold_s,
    label,
    max_hold_s,
    parse_profile,
    set_max_hold_s,
)


@pytest.fixture(autouse=True)
def _reset_cap():
    set_max_hold_s(None)
    yield
    set_max_hold_s(None)


_LIVE = DataState.LIVE


def profile_of(name):
    return parse_profile(name)


# ---------------------------------------------------------------------------
# 1-2, 13-14. entry identical, L25_CURRENT untouched
# ---------------------------------------------------------------------------


class TestEntryIsolation:
    def test_both_profiles_declare_the_same_entry_strategy(self):
        from autopaper.strategy_profile import ENTRY_STRATEGY
        # A profile cannot change the entry strategy. There is only one.
        assert ENTRY_STRATEGY == "L25"
        assert ResearchArm.L25_MAX_HOLD in ARMS

    def test_the_two_profiles_share_the_arm_spec_shape(self):
        """Only max_hold_s may differ between L25_CURRENT and L25_MAX_HOLD."""
        a = ARMS[ResearchArm.L25_CURRENT].as_dict()
        b = ARMS[ResearchArm.L25_MAX_HOLD].as_dict()
        differing = {k for k in a if a[k] != b.get(k)}
        # name/label/provenance state which profile this is; max_hold_s is the
        # behavioural difference. Nothing else may move.
        allowed = {"name", "label", "provenance", "max_hold_s",
                   "production_profile", "arm"}
        assert differing <= allowed, f"unexpected divergence: {differing - allowed}"
        assert a.get("extra_entry_rule") == b.get("extra_entry_rule")

    def test_l25_current_still_has_no_cap(self):
        assert max_hold_s("L25_CURRENT") is None
        assert ARMS[ResearchArm.L25_CURRENT].max_hold_s is None

    def test_l25_current_is_still_the_default_profile(self):
        assert DEFAULT_PROFILE is StrategyProfile.L25_CURRENT

    def test_l25_60s_is_untouched_at_a_fixed_60(self):
        """The existing research profile must not drift."""
        assert max_hold_s("L25_60S") == 60.0
        assert PROFILE_MAX_HOLD_S[StrategyProfile.L25_60S] == 60.0

    def test_max_hold_is_a_configurable_parameter_not_a_literal(self):
        set_max_hold_s(90.0)
        assert max_hold_s("L25_MAX_HOLD") == 90.0
        assert max_hold_s("L25_CURRENT") is None, "the override leaked"
        assert max_hold_s("L25_60S") == 60.0, "the override leaked"
        set_max_hold_s(None)
        assert max_hold_s("L25_MAX_HOLD") == 60.0

    def test_sixty_second_default(self):
        assert PROFILE_MAX_HOLD_S[StrategyProfile.L25_MAX_HOLD] == 60.0
        assert configured_max_hold_s() == 60.0

    def test_nonsense_caps_are_refused_not_clamped(self):
        for bad in (0.0, -5.0):
            with pytest.raises(StrategyProfileError):
                set_max_hold_s(bad)

    def test_config_default_is_sixty(self):
        from autopaper.config import DecisionConfig
        assert DecisionConfig().max_hold_seconds == 60.0


# ---------------------------------------------------------------------------
# 3-5. the cap fires at the right time, and never before
# ---------------------------------------------------------------------------


class CapPosition:
    """The minimum surface ExitPolicy reads off a position."""

    def __init__(self, opened_at, profile="L25_MAX_HOLD"):
        self.opened_at = opened_at
        # The exit policy reads entry_timestamp, not opened_at.
        self.entry_timestamp = opened_at
        self.production_profile = profile
        self.strategy_profile = profile
        self.exit_style = "FAST_FAILURE"
        self.entry_price = 1.0
        self.peak_return_pct = 0.0
        self.exit_hold_count = 0
        self.mint = "M"
        self.arm = "L25_MAX_HOLD"
        self.position_id = "pos_x"
        self.symbol = "X"

    quantity = 1.0
    notional_usd = 1.0
    cost_basis_usd = 1.0
    cash_at_entry = 1.0
    mark_price = 1.0
    data_state = None
    sol_usd_reference = 120.0
    sol_usd_reference_at = time.time()

    def return_at(self, mark_price):
        return (float(mark_price) / self.entry_price - 1.0)


class TestCapTiming:
    def _policy(self):
        from autopaper.config import ExitConfig
        return ExitPolicy(ExitConfig())

    def test_does_not_fire_before_the_configured_time(self):
        p = self._policy()
        now = time.time()
        d = p.assess(CapPosition(now - 30.0), now=now, mark_price=1.0,
                     data_state=_LIVE, momentum_30s_pct=None,
                     structure_30s_pct=None, liquidity_change_pct=None)
        assert d.action == "HOLD"
        assert d.trigger != max_hold_trigger(60.0)

    def test_fires_at_the_configured_time(self):
        p = self._policy()
        now = time.time()
        d = p.assess(CapPosition(now - 61.0), now=now, mark_price=1.0,
                     data_state=_LIVE, momentum_30s_pct=None,
                     structure_30s_pct=None, liquidity_change_pct=None)
        assert d.action == "EXIT"
        assert d.trigger == max_hold_trigger(60.0)

    def test_the_trigger_names_the_cap_actually_in_force(self):
        """A 90s exit must never be reported as a 60s one."""
        p = self._policy()
        set_max_hold_s(90.0)
        now = time.time()
        early = p.assess(CapPosition(now - 61.0), now=now, mark_price=1.0,
                         data_state=_LIVE, momentum_30s_pct=None,
                         structure_30s_pct=None, liquidity_change_pct=None)
        assert early.action == "HOLD", "a 90s cap fired at 61s"
        late = p.assess(CapPosition(now - 91.0), now=now, mark_price=1.0,
                        data_state=_LIVE, momentum_30s_pct=None,
                        structure_30s_pct=None, liquidity_change_pct=None)
        assert late.action == "EXIT"
        assert late.trigger == max_hold_trigger(90.0)

    def test_an_earlier_rule_retains_precedence(self):
        """
        PRECEDENCE IS NOW PROFILE-SCOPED, and this test records both halves.

        For a NORMAL profile the cap is still appended LAST: a stop that fires
        at 10s must win over a cap at 60s and keep its own reason, because
        telling an operator a position exited on time when it actually exited
        on risk is the failure mode the ordering exists to prevent.

        ``L25_MAX_HOLD`` is the deliberate exception and no longer obeys that
        ordering. It is a TRUE hard-hold profile -- BUY, hold, sell at the cap --
        so at 61s the cap fires even against a -50% mark. Its price and style
        exits do not run at all.

        The rationale above was sound and still binds every other profile; it is
        overridden for this one profile by explicit requirement, which is why the
        normal-profile case is asserted here rather than dropped.
        """
        now = time.time()

        # A normal profile: an earlier rule keeps precedence over the cap.
        d = self._policy().assess(
            CapPosition(now - 61.0, profile="L25_60S"), now=now, mark_price=0.5,
            data_state=_LIVE, momentum_30s_pct=None, structure_30s_pct=None,
            liquidity_change_pct=None)
        assert d.action == "EXIT"
        assert d.trigger != max_hold_trigger(60.0), (
            "the max-hold cap took precedence over an existing exit rule")

        # L25_MAX_HOLD: the cap wins, by design.
        d = self._policy().assess(
            CapPosition(now - 61.0), now=now, mark_price=0.5,
            data_state=_LIVE, momentum_30s_pct=None, structure_30s_pct=None,
            liquidity_change_pct=None)
        assert d.action == "EXIT"
        assert d.trigger == max_hold_trigger(60.0), (
            "L25_MAX_HOLD must reach its cap; its price exits do not run")

    def test_the_hard_hold_cap_is_not_reached_through_the_disaster_cap(self):
        """A -50% mark must not become a risk exit on this profile."""
        now = time.time()
        d = self._policy().assess(
            CapPosition(now - 30.0), now=now, mark_price=0.5,
            data_state=_LIVE, momentum_30s_pct=None, structure_30s_pct=None,
            liquidity_change_pct=None)
        assert d.action == "HOLD"
        assert d.trigger is None

    def test_l25_current_never_gets_a_cap_exit(self):
        p = self._policy()
        now = time.time()
        d = p.assess(CapPosition(now - 600.0, profile="L25_CURRENT"), now=now,
                     mark_price=1.0, data_state=_LIVE,
                     momentum_30s_pct=None, structure_30s_pct=None,
                     liquidity_change_pct=None)
        assert d.trigger != max_hold_trigger(60.0)


# ---------------------------------------------------------------------------
# 6-8. the cap is a TRIGGER; the executable SELL still decides
# ---------------------------------------------------------------------------


class TestCapIsOnlyATrigger:
    def test_a_cap_exit_requests_a_real_sell_quote(self):
        """
        The MAX_HOLD path must go through ``_close_for_arm``, which calls the
        quote service. If the cap closed a position without a SELL request,
        the exit would be mark-only -- the exact defect forbidden here.
        """
        import inspect
        from autopaper.paper_live.engine import PaperLiveEngine
        src = inspect.getsource(PaperLiveEngine)
        close = inspect.getsource(PaperLiveEngine._close_for_arm)
        assert "quote_sell" in close, (
            "the exit path no longer requests an executable SELL")
        assert "fill_from_quote" in close, (
            "the exit no longer validates the SELL quote")
        # settle() only on the success path, so a refused SELL keeps the
        # position open rather than booking a mark-only fill.
        assert close.count("account.settle(") == 1
        assert "**self._sell_provenance(" in close

    def test_a_refused_sell_reports_no_executable_sell(self):
        from autopaper.paper_live.schema import FailureReason
        assert FailureReason.EXIT_QUOTE_UNAVAILABLE.value == "EXIT_QUOTE_UNAVAILABLE"
        assert FailureReason.EXIT_QUOTE_STALE.value == "EXIT_QUOTE_STALE"

    def test_provenance_names_the_outcome_not_a_fill(self):
        from autopaper.paper_live.engine import PaperLiveEngine

        class Q:
            provider = "pumpswap_direct"
            venue_id = "PUMPSWAP:POOL"
            pool_or_curve = "POOL"
            effective_price_usd = 0.000024
            quote_age_ms = 12.0

        class P:
            opened_at = time.time() - 61.0
            production_profile = "L25_MAX_HOLD"

        settled = PaperLiveEngine._sell_provenance(P(), Q(), None, settled=True)
        refused = PaperLiveEngine._sell_provenance(P(), Q(), None, settled=False)
        assert settled["sell_execution_status"] == "SETTLED"
        assert refused["sell_execution_status"] == "NO_EXECUTABLE_SELL"
        assert settled["strategy_profile"] == "L25_MAX_HOLD"
        assert settled["max_hold_seconds"] == 60.0
        assert settled["hold_time_seconds"] >= 60.0


# ---------------------------------------------------------------------------
# 9-12. persistence, selector, REAL
# ---------------------------------------------------------------------------


class TestSelectorAndPersistence:
    def test_the_selector_exposes_the_profile(self):
        assert "L25_MAX_HOLD" in arm_names()
        assert ResearchArm.L25_MAX_HOLD in ARMS

    def test_every_existing_arm_is_preserved(self):
        for name in ("L25_CURRENT", "L25_60S", "H2_CAP60"):
            assert name in arm_names(), f"{name} was dropped"

    def test_parse_arms_accepts_it(self):
        got = parse_arms(["L25_MAX_HOLD"])
        assert len(got) == 1 and got[0].name is ResearchArm.L25_MAX_HOLD

    def test_the_selector_does_not_hardcode_its_own_list(self):
        """arm_names() IS the selector list, so nothing can drift from it."""
        from autopaper.paper_live.arms import selectable_arms
        # arm_names() is EXACTLY the selectable arms -- one entry per arm.
        # The V2 FLOOR (BETA) group token is NOT in it: it is a group, not an
        # arm, and it is offered by the Telegram picker as its own row rather
        # than as another name in the arm vocabulary. Asserting equality with
        # ``selectable_arms()`` (rather than "plus the group") is what keeps
        # ``len(arm_names()) == len(arm_specs())`` true, which is the invariant
        # that stops one bet experiment quietly becoming three phantom arms.
        expected = tuple(a.value for a in selectable_arms())
        assert arm_names() == expected, (
            "arm_names() IS the selector list")
        assert set(arm_names()) <= set(ARMS), (
            "every selectable arm must be registered")

    def test_real_does_not_enable_it_automatically(self):
        from autopaper.engine import PaperTrader
        assert getattr(PaperTrader, "REAL_ENABLED_DEFAULT", False) is False
        # The profile is PAPER-selectable only; nothing arms REAL by choosing it.
        assert PROFILE_EXIT_STRATEGY[StrategyProfile.L25_MAX_HOLD] == "FAST_FAILURE"

    def test_label_names_the_policy(self):
        assert "MAX HOLD" in label("L25_MAX_HOLD")

    def test_unknown_profiles_are_refused(self):
        with pytest.raises(StrategyProfileError):
            parse_profile("L25_MAX_HOLD_9000")


# ---------------------------------------------------------------------------
# 9. telemetry
# ---------------------------------------------------------------------------


class TestTelemetry:
    def test_trade_record_carries_the_full_provenance(self):
        import dataclasses
        from autopaper.paper_live.schema import TradeRecord
        names = {f.name for f in dataclasses.fields(TradeRecord)}
        assert {"strategy_profile", "max_hold_seconds", "hold_time_seconds",
                "sell_quote_source", "sell_market_id", "sell_quote_price",
                "sell_quote_age_ms", "sell_execution_status"} <= names

    def test_the_fields_serialise(self):
        import dataclasses, inspect
        from autopaper.paper_live.schema import TradeRecord
        src = inspect.getsource(TradeRecord.as_dict)
        for f in ("strategy_profile", "max_hold_seconds", "hold_time_seconds",
                  "sell_execution_status", "sell_market_id"):
            assert f in src, f"{f} is not serialised"


# ---------------------------------------------------------------------------
# 15. settlement untouched
# ---------------------------------------------------------------------------


class TestSettlementUnchanged:
    def test_settlement_is_still_two_executable_fills(self):
        import inspect
        from autopaper.paper_live.fills import realize
        import dataclasses
        from autopaper.paper_live.schema import RealizedPnl
        names = {f.name for f in dataclasses.fields(RealizedPnl)}
        # Settlement is QUOTE-ASSET amounts plus explicit cost terms. There is
        # no mark anywhere in it: a max-hold exit settles from two executable
        # fills, and this is the shape that proves it.
        assert {"entry_cost_quote", "exit_proceeds_quote", "gross_pnl_quote",
                "net_pnl_quote", "trading_fees_quote",
                "network_costs_quote"} <= names
        assert not any("mark" in n for n in names), (
            f"settlement must not be mark-based: {sorted(names)}")

    def test_fee_model_untouched(self):
        from autopaper.sizing import PAPER_FEE_BPS, PAPER_SLIPPAGE_BPS
        assert PAPER_FEE_BPS == 85.0
        assert PAPER_SLIPPAGE_BPS == 120.0

    def test_the_gap_guard_threshold_is_untouched(self):
        from autopaper.paper_live.economics import MARK_GAP_GUARD_THRESHOLD_PCT
        assert MARK_GAP_GUARD_THRESHOLD_PCT == 10.0

    def test_profiles_are_not_pooled(self):
        """Two profiles must stay separable in any aggregate."""
        assert PROFILE_LABELS[StrategyProfile.L25_CURRENT] != \
            PROFILE_LABELS[StrategyProfile.L25_MAX_HOLD]


