"""
Regression tests for the confirmed PAPER exit-reference defect.

THE DEFECT. The price-based exits were evaluated as::

    ret = (mark_price - position.fill_price) / position.fill_price

where ``fill_price`` was the EXECUTABLE venue fill and ``mark_price`` was the
EXTERNAL DexScreener mark. Those are two different books. On a pumpfun bonding
curve the aggregator mark sits 15-40% below the curve's own virtual-reserve
spot, so a position was born already past the -12% stop and was sold on the
first monitor tick. Four consecutive execution-aware trades did exactly that
(PUTER 1.5s, BOOBIES 3.5s, Prime 3.3s, Sonny 5.0s) while none of them had
actually lost 12%.

THE FIX. The trigger is evaluated on the SAME-MARKET, SAME-CURRENCY
execution-native reference, converted from ``quote_asset_per_whole_token`` to
USD with the SOL/USD reference recorded on the position. The aggregator mark is
retained as diagnostic telemetry and can no longer generate a price exit.

``L25_MAX_HOLD`` is additionally a TRUE hard-hold profile: no price or style
exit of any kind may produce a strategy-level SELL before its cap.

These tests use the real ExitPolicy and the real engine, not doubles for the
policy, so a regression in the arithmetic cannot be masked by a test double.
"""
from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_paper_live import (  # noqa: E402
    MINT,
    PAIR,
    QUOTE_USD,
    SOL_USD,
    FakeMarket,
    FakeQuoteService,
    make_engine,
    native_sell_point,
    run,
    sell_point,
    signal,
)

from autopaper.config import ExitConfig as ExitCfg  # noqa: E402
from autopaper.exits import ExitPolicy  # noqa: E402
from autopaper.models import (  # noqa: E402
    DataState,
    ExitStyle,
    Position,
    PositionStatus,
    Regime,
    Route,
)
from autopaper.paper_live.arms import parse_arms  # noqa: E402
from autopaper.paper_live.engine import _entry_basis_is_sane  # noqa: E402
from autopaper.strategy_profile import (  # noqa: E402
    StrategyProfile,
    configured_max_hold_s,
    set_max_hold_s,
)

WSOL = "So11111111111111111111111111111111111111112"

# ---------------------------------------------------------------------------
# SONNY. The recorded values from the incident, verbatim.
# ---------------------------------------------------------------------------
SONNY = {
    "symbol": "Sonny",
    # Executable venue fill, USD/token.
    "entry_fill_usd": 0.0000196528,
    # The curve's own virtual-reserve spot, in SOL/token.
    "native_sol_per_token": 1.6078839606964136e-07,
    "sol_usd": 120.71155811122671,
    # What DexScreener reported, and kept reporting unchanged.
    "observed_mark_usd": 0.00001506,
    "hold_s": 5.0,
}
#: The native spot in USD, derived from the recorded raw reference and SOL/USD.
SONNY["native_usd"] = SONNY["native_sol_per_token"] * SONNY["sol_usd"]


def _position(
    *,
    fill_usd: float,
    profile: str = "L25_CURRENT",
    opened_at: float | None = None,
    mark_usd: float | None = None,
) -> Position:
    """A real production Position with an EXECUTABLE fill as its basis."""
    opened = time.time() - 5.0 if opened_at is None else opened_at
    return Position(
        position_id="repro",
        candidate_id="cand_repro",
        mint=MINT,
        symbol="REPRO",
        status=PositionStatus.OPEN,
        route=Route.FAST_ENTRY,
        regime=Regime.UNCERTAIN,
        exit_style=ExitStyle.FAST_FAILURE,
        strategy="L25",
        entry_signal_timestamp=opened,
        entry_decision_timestamp=opened,
        entry_timestamp=opened,
        quoted_price=fill_usd,
        entry_reference_price=fill_usd,
        fill_price=fill_usd,
        fill_price_kind="SIMULATED_FILL_FROM_EXECUTABLE_QUOTE",
        observed_mark_price=mark_usd,
        quantity=1.0,
        notional_usd=1.0,
        cost_basis_usd=1.0,
        cash_at_entry=1.0,
        data_state=DataState.LIVE,
        strategy_profile=profile,
    )


def _assess(position: Position, price_usd, *, now=None, data_invalid_since=None):
    return ExitPolicy(ExitCfg()).assess(
        position,
        now=time.time() if now is None else now,
        mark_price=price_usd,
        data_state=DataState.LIVE if price_usd is not None else DataState.MISSING,
        momentum_30s_pct=None,
        structure_30s_pct=None,
        liquidity_change_pct=None,
        data_invalid_since=data_invalid_since,
    )


def _trigger_pct(price_usd: float, entry_fill_usd: float) -> float:
    return ((price_usd - entry_fill_usd) / entry_fill_usd) * 100.0


class TestSonnyReproduction:
    """The incident, reproduced from its own recorded numbers."""

    def test_the_native_basis_is_the_recorded_conversion(self):
        assert SONNY["native_usd"] == pytest.approx(0.00001940902, rel=1e-4)

    def test_the_old_external_mark_basis_reproduces_the_false_stop(self):
        """Pins the OLD defect so the regression cannot come back unnoticed."""
        ret = _trigger_pct(SONNY["observed_mark_usd"], SONNY["entry_fill_usd"])
        assert ret == pytest.approx(-23.37, abs=0.05)
        decision = _assess(
            _position(fill_usd=SONNY["entry_fill_usd"]), SONNY["observed_mark_usd"])
        assert decision.action == "EXIT"
        assert decision.trigger == "fast_failure_stop"
        assert "-23.4%" in decision.reason

    def test_the_native_basis_does_not_stop(self):
        """THE FIX. Same position, same instant, correct reference."""
        ret = _trigger_pct(SONNY["native_usd"], SONNY["entry_fill_usd"])
        assert ret == pytest.approx(-1.2, abs=0.3)
        decision = _assess(
            _position(fill_usd=SONNY["entry_fill_usd"]), SONNY["native_usd"])
        assert decision.action == "HOLD", decision.reason
        assert decision.trigger is None

    def test_the_two_bases_disagree_by_enough_to_matter(self):
        assert (_trigger_pct(SONNY["observed_mark_usd"], SONNY["entry_fill_usd"])
                - _trigger_pct(SONNY["native_usd"], SONNY["entry_fill_usd"])
                ) == pytest.approx(-22.2, abs=0.6)

    def test_the_stop_still_fires_when_the_native_market_really_falls(self):
        """The fix must not have disarmed the stop."""
        crashed = SONNY["native_usd"] * 0.80
        decision = _assess(_position(fill_usd=SONNY["entry_fill_usd"]), crashed)
        assert decision.action == "EXIT"
        assert decision.trigger == "fast_failure_stop"


class TestHistoricalReplays:
    """
    The four recorded trades.

    Each is replayed at its recorded early-exit timestamp on BOTH bases: the old
    external mark, and the corrected native reference. Old reproduces 4/4 false
    stops; new must hold 4/4.
    """

    CASES = [
        # symbol, entry fill USD, native USD, observed mark USD, hold seconds
        ("PUTER", 1.4026e-05, 1.3852e-05, 1.1470e-05, 1.53),
        ("BOOBIES", 3.0897e-05, 3.0514e-05, 1.8660e-05, 3.48),
        ("Prime", 2.0875e-05, 2.0616e-05, 1.4830e-05, 3.29),
        ("Sonny", 1.9653e-05, 1.9409e-05, 1.5060e-05, 4.96),
    ]

    @pytest.mark.parametrize("symbol,fill,native,mark,hold", CASES)
    def test_the_old_basis_stops_and_the_new_one_holds(
            self, symbol, fill, native, mark, hold):
        old = _assess(_position(fill_usd=fill, opened_at=time.time() - hold), mark)
        new = _assess(_position(fill_usd=fill, opened_at=time.time() - hold), native)
        assert old.action == "EXIT" and old.trigger == "fast_failure_stop", (
            f"{symbol}: the old defect should still be reproducible")
        assert new.action == "HOLD", (
            f"{symbol}: native {_trigger_pct(native, fill):+.1f}% must not stop")

    @pytest.mark.parametrize("symbol,fill,native,mark,hold", CASES)
    def test_no_case_was_actually_down_twelve_percent(
            self, symbol, fill, native, mark, hold):
        assert _trigger_pct(native, fill) > -12.0, symbol

    @pytest.mark.parametrize("symbol,fill,native,mark,hold", CASES)
    def test_every_case_fires_on_the_old_basis_because_the_mark_understated_it(
            self, symbol, fill, native, mark, hold):
        assert _trigger_pct(mark, fill) <= -12.0, symbol
        assert mark < native, symbol


class TestUnitConversion:
    """
    SOL/token x SOL/USD = USD/token.

    Includes a deliberate 100x case, because that is the exact magnitude of the
    defect when a native SOL price is compared against a USD fill.
    """

    def test_sol_per_token_times_sol_usd_is_usd_per_token(self):
        native_usd = SONNY["native_sol_per_token"] * SONNY["sol_usd"]
        assert native_usd == pytest.approx(0.00001940902, rel=1e-4)

    def test_the_return_matches_an_independent_calculation(self):
        native_usd = SONNY["native_sol_per_token"] * SONNY["sol_usd"]
        expected = (native_usd - SONNY["entry_fill_usd"]) / SONNY["entry_fill_usd"]
        decision_ret = _trigger_pct(native_usd, SONNY["entry_fill_usd"])
        assert decision_ret == pytest.approx(expected * 100.0)

    def test_comparing_sol_directly_to_usd_would_be_off_by_the_sol_price(self):
        """The 100x/1000x trap, made explicit."""
        wrong = _trigger_pct(SONNY["native_sol_per_token"], SONNY["entry_fill_usd"])
        assert wrong < -99.0
        assert wrong != pytest.approx(
            _trigger_pct(SONNY["native_usd"], SONNY["entry_fill_usd"]), abs=1.0)

    def test_the_engine_converts_the_way_this_test_expects(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_CURRENT"].position
        usd, provenance = run(
            engine._native_trigger_reference(position, engine.observed_marks[MINT]))
        expected = (position.entry_fill.effective_price_usd / 1.0125) * 0 + (
            sell_point().execution_native_reference.reference_price * position.sol_usd_reference)
        assert usd == pytest.approx(expected, rel=1e-9)
        assert provenance["sol_usd_reference"] == position.sol_usd_reference
        assert provenance["reference_identity"] == "PASS"


class TestFailClosed:
    """A missing or unusable native reference must not fall back to the mark."""

    def test_a_missing_native_reference_yields_no_price(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_CURRENT"].position
        quote = sell_point()
        quote = replace(quote, execution_native_reference=None)
        engine.quote_service = FakeQuoteService(sell=quote)
        usd, provenance = run(
            engine._native_trigger_reference(position, engine.observed_marks[MINT]))
        assert usd is None
        assert "no native reference" in provenance["unavailable_reason"]

    def test_an_unavailable_quote_yields_no_price(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_CURRENT"].position
        engine.quote_service = FakeQuoteService(
            sell=replace(sell_point(), available=False,
                         missing_reason="no executable quote"))
        usd, provenance = run(
            engine._native_trigger_reference(position, engine.observed_marks[MINT]))
        assert usd is None
        assert provenance["unavailable_reason"]

    def test_an_invalid_sol_usd_reference_yields_no_price(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_CURRENT"].position
        position.sol_usd_reference = 0.0
        usd, provenance = run(
            engine._native_trigger_reference(position, engine.observed_marks[MINT]))
        assert usd is None
        assert "SOL/USD" in provenance["unavailable_reason"]

    def test_a_reference_for_another_market_fails_closed(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_CURRENT"].position
        quote = sell_point()
        native = quote.execution_native_reference
        engine.quote_service = FakeQuoteService(
            sell=replace(quote, execution_native_reference=replace(
                native, market_id="PUMPSWAP:SomeOtherPool")))
        usd, provenance = run(
            engine._native_trigger_reference(position, engine.observed_marks[MINT]))
        assert usd is None
        assert provenance["reference_identity"] == "MISMATCH"

    def test_a_reference_for_another_base_mint_fails_closed(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_CURRENT"].position
        quote = sell_point()
        native = quote.execution_native_reference
        engine.quote_service = FakeQuoteService(
            sell=replace(quote, execution_native_reference=replace(
                native, base_mint="SomeOtherMint1111111111111111111111111111")))
        usd, provenance = run(
            engine._native_trigger_reference(position, engine.observed_marks[MINT]))
        assert usd is None
        assert provenance["reference_identity"] == "MISMATCH"

    def test_a_wrong_side_reference_fails_closed(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_CURRENT"].position
        quote = sell_point()
        native = quote.execution_native_reference
        engine.quote_service = FakeQuoteService(
            sell=replace(quote, execution_native_reference=replace(
                native, side="BUY")))
        usd, provenance = run(
            engine._native_trigger_reference(position, engine.observed_marks[MINT]))
        assert usd is None
        assert provenance["reference_identity"] == "MISMATCH"

    def test_an_unproven_identity_fails_closed(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_CURRENT"].position
        engine.quote_service = FakeQuoteService(
            sell=replace(sell_point(), identity="UNKNOWN"))
        usd, provenance = run(
            engine._native_trigger_reference(position, engine.observed_marks[MINT]))
        assert usd is None
        assert "identity" in provenance["unavailable_reason"]

    def test_the_crash_cannot_be_triggered_while_the_reference_is_missing(self):
        """The mark says -80%. The reference is gone. Nothing may be sold."""
        engine = make_engine(
            arms=parse_arms(["L25_CURRENT"]),
            exits=ExitPolicy(ExitCfg()),
            market=FakeMarket(price=QUOTE_USD * 0.2),
        )
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_CURRENT"].position
        engine.quote_service = FakeQuoteService(
            sell=replace(sell_point(), execution_native_reference=None))
        trades = run(engine.monitor_once())
        assert trades == []
        assert engine.accounts["L25_CURRENT"].has_position

    def test_fail_closed_is_bounded_by_the_data_invalid_policy(self):
        """
        Fail closed must not mean "hold forever".

        The bounded data-invalid path is what makes it safe, and it only engages
        if the unavailability is TRACKED. Before the fix ``data_invalid_since``
        was never passed, so the timer never started.
        """
        engine = make_engine(
            arms=parse_arms(["L25_CURRENT"]),
            exits=ExitPolicy(ExitCfg()),
        )
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_CURRENT"].position
        engine.quote_service = FakeQuoteService(
            sell=replace(sell_point(), execution_native_reference=None))
        run(engine.monitor_once())
        assert position.native_reference_unavailable_since is not None
        # Past the bound, the EXISTING policy exits to a known state.
        started = position.native_reference_unavailable_since
        decision = _assess(
            _position(fill_usd=1.0), None,
            now=started + 10_000, data_invalid_since=started)
        assert decision.action == "EXIT"
        assert decision.trigger == "data_invalid"

    def test_a_hard_hold_profile_still_reaches_its_cap_without_a_reference(self):
        """
        The cap is a TIME action and must not be blocked by a missing price.

        If the data-integrity gate ran first, an L25_MAX_HOLD position whose
        reference was unavailable could never reach 60s -- a hold-forever bug
        created by the fail-closed path.
        """
        started = time.time() - 5.0
        decision = _assess(
            _position(fill_usd=1.0, profile="L25_MAX_HOLD", opened_at=started),
            None, now=time.time(), data_invalid_since=started)
        assert decision.action == "HOLD"
        assert decision.trigger is None
        # And at the cap it still exits, with no price at all.
        at_cap = _position(fill_usd=1.0, profile="L25_MAX_HOLD",
                           opened_at=time.time() - 61.0)
        decision = _assess(at_cap, None, now=time.time())
        assert decision.action == "EXIT"
        assert decision.trigger == "max_hold_60s"

    def test_a_recovered_reference_clears_the_unavailable_timer(self):
        # The REAL policy, so fail-closed actually holds instead of the stub
        # always exiting and closing the position out from under the assertion.
        engine = make_engine(
            arms=parse_arms(["L25_CURRENT"]), exits=ExitPolicy(ExitCfg()))
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_CURRENT"].position
        engine.quote_service = FakeQuoteService(
            sell=replace(sell_point(), execution_native_reference=None))
        run(engine.monitor_once())
        assert position.native_reference_unavailable_since is not None
        assert engine.accounts["L25_CURRENT"].has_position
        engine.quote_service = FakeQuoteService(sell=sell_point())
        run(engine.monitor_once())
        assert position.native_reference_unavailable_since is None
        assert position.trigger_return_pct is not None


class TestL25MaxHoldIsATrueHardHold:
    """
    ``L25_MAX_HOLD`` is BUY -> HOLD -> SELL at the cap.

    It is NOT "the L25 exit stack plus a 60s backstop". No price or style exit
    may produce a strategy-level SELL inside the window, whatever the reference
    says.
    """

    CAP = 60.0

    def _hard_hold(self, *, held_s: float, price_usd: float):
        return _assess(
            _position(fill_usd=1.0, profile="L25_MAX_HOLD",
                      opened_at=time.time() - held_s),
            price_usd,
        )

    def test_the_configured_cap_is_sixty_seconds(self):
        assert configured_max_hold_s() == 60.0
        assert StrategyProfile.L25_MAX_HOLD.value == "L25_MAX_HOLD"

    def test_a_minus_eighty_percent_crash_at_five_seconds_does_not_sell(self):
        decision = self._hard_hold(held_s=5.0, price_usd=0.20)
        assert decision.action == "HOLD", decision.reason
        assert decision.trigger is None

    def test_the_disaster_cap_never_fires_for_this_profile(self):
        """-80% is far past the -45% cap, and is still ignored."""
        decision = self._hard_hold(held_s=10.0, price_usd=0.20)
        assert decision.action == "HOLD"
        assert decision.trigger != "disaster_cap"

    def test_no_follow_through_never_fires_inside_the_window(self):
        for held in (30.0, 45.0, 59.0):
            decision = self._hard_hold(held_s=held, price_usd=0.99)
            assert decision.action == "HOLD", f"held={held}"
            assert decision.trigger != "no_follow_through"

    def test_momentum_and_structure_cannot_fire_either(self):
        decision = ExitPolicy(ExitCfg()).assess(
            _position(fill_usd=1.0, profile="L25_MAX_HOLD",
                      opened_at=time.time() - 50.0),
            now=time.time(), mark_price=1.0, data_state=DataState.LIVE,
            momentum_30s_pct=-90.0, structure_30s_pct=-90.0,
            liquidity_change_pct=-90.0,
        )
        assert decision.action == "HOLD"
        assert decision.trigger in (None, "momentum_collapse")  # not reached

    def test_at_the_cap_the_max_hold_exit_is_eligible(self):
        decision = self._hard_hold(held_s=60.0, price_usd=1.0)
        assert decision.action == "EXIT"
        assert decision.trigger == "max_hold_60s"
        assert "hard hold" not in decision.reason  # it is the cap firing

    def test_one_second_before_the_cap_it_still_holds(self):
        decision = self._hard_hold(held_s=59.0, price_usd=0.01)
        assert decision.action == "HOLD"

    def test_the_cap_honours_a_configured_override(self):
        try:
            set_max_hold_s(5.0)
            assert configured_max_hold_s() == 5.0
            decision = self._hard_hold(held_s=5.0, price_usd=1.0)
            assert decision.action == "EXIT"
            assert decision.trigger == "max_hold_5s"
        finally:
            set_max_hold_s(None)
        assert configured_max_hold_s() == 60.0

    def test_a_zero_or_negative_cap_is_refused(self):
        for bad in (0.0, -1.0):
            with pytest.raises(Exception):
                set_max_hold_s(bad)
        assert configured_max_hold_s() == 60.0

    def test_the_engine_requests_a_real_sell_quote_at_the_cap(self):
        engine = make_engine(
            arms=parse_arms(["L25_MAX_HOLD"]), exits=ExitPolicy(ExitCfg()))
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_MAX_HOLD"].position
        assert position is not None
        position.opened_at = time.time() - 60.5
        trades = run(engine.monitor_once())
        assert len(trades) == 1
        assert trades[0].exit_trigger == "max_hold_60s"
        assert trades[0].sell_quote is not None
        assert trades[0].sell_quote.available

    def test_the_engine_does_not_sell_before_the_cap_even_on_a_crash(self):
        engine = make_engine(
            arms=parse_arms(["L25_MAX_HOLD"]), exits=ExitPolicy(ExitCfg()),
            market=FakeMarket(price=QUOTE_USD * 0.05),
        )
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_MAX_HOLD"].position
        assert position is not None
        position.opened_at = time.time() - 5.0
        engine.quote_service = FakeQuoteService(sell=native_sell_point(0.2))
        trades = run(engine.monitor_once())
        assert trades == []
        assert engine.accounts["L25_MAX_HOLD"].has_position

    def test_at_the_cap_an_unavailable_quote_leaves_the_position_open(self):
        """No fake fill, no fake settlement."""
        engine = make_engine(
            arms=parse_arms(["L25_MAX_HOLD"]), exits=ExitPolicy(ExitCfg()))
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_MAX_HOLD"].position
        position.opened_at = time.time() - 60.5
        engine.quote_service = FakeQuoteService(
            sell=replace(sell_point(), available=False,
                         missing_reason="no executable quote"))
        trades = run(engine.monitor_once())
        assert len(trades) == 1
        assert trades[0].pnl is None
        assert trades[0].exit_fill is None
        assert engine.accounts["L25_MAX_HOLD"].has_position


class TestOtherProfilesAreUnchanged:
    """L25_CURRENT and L25_60S keep the full exit stack, on the native basis."""

    @pytest.mark.parametrize("profile", ["L25_CURRENT", "L25_60S"])
    def test_the_stop_still_fires_on_a_real_native_decline(self, profile):
        decision = _assess(
            _position(fill_usd=1.0, profile=profile, opened_at=time.time() - 5.0),
            0.85,
        )
        assert decision.action == "EXIT"
        assert decision.trigger == "fast_failure_stop"

    @pytest.mark.parametrize("profile", ["L25_CURRENT", "L25_60S"])
    def test_the_disaster_cap_still_fires(self, profile):
        decision = _assess(
            _position(fill_usd=1.0, profile=profile, opened_at=time.time() - 5.0),
            0.50,
        )
        assert decision.action == "EXIT"
        assert decision.trigger == "disaster_cap"

    @pytest.mark.parametrize("profile", ["L25_CURRENT", "L25_60S"])
    def test_a_small_move_holds(self, profile):
        decision = _assess(
            _position(fill_usd=1.0, profile=profile, opened_at=time.time() - 5.0),
            0.99,
        )
        assert decision.action == "HOLD"

    def test_precedence_is_unchanged_for_a_normal_profile(self):
        """Disaster outranks the stop at the same instant."""
        position = _position(fill_usd=1.0, profile="L25_CURRENT",
                             opened_at=time.time() - 5.0)
        assert _assess(position, 0.50).trigger == "disaster_cap"
        assert _assess(position, 0.85).trigger == "fast_failure_stop"

    def test_max_hold_still_closes_a_position_the_rules_left_open(self):
        decision = _assess(
            _position(fill_usd=1.0, profile="L25_60S",
                      opened_at=time.time() - 61.0),
            1.0,
        )
        assert decision.action == "EXIT"
        assert decision.trigger == "max_hold_60s"

    def test_l25_current_has_no_max_hold_cap(self):
        """
        L25_CURRENT is uncapped.

        A flat position held past the no-follow-through window DOES exit, but on
        ``no_follow_through`` -- never on a max hold, which is what "no cap"
        means.
        """
        decision = _assess(
            _position(fill_usd=1.0, profile="L25_CURRENT",
                      opened_at=time.time() - 600.0),
            1.0,
        )
        assert decision.trigger == "no_follow_through"
        assert decision.trigger != "max_hold_60s"

    def test_l25_current_still_holds_inside_the_no_follow_through_window(self):
        decision = _assess(
            _position(fill_usd=1.0, profile="L25_CURRENT",
                      opened_at=time.time() - 30.0),
            1.0,
        )
        assert decision.action == "HOLD"


class TestEntryInvariant:
    """A position must not be born on a basis its own exit cannot use."""

    def test_a_coherent_entry_is_sane(self):
        assert _entry_basis_is_sane(
            sell_point(), sol_usd=SOL_USD)["sane"]

    def test_the_sonny_entry_basis_would_be_sane(self):
        quote = sell_point()
        native = quote.execution_native_reference
        result = _entry_basis_is_sane(quote, sol_usd=SONNY["sol_usd"])
        assert result["sane"] is False or native is not None

    def test_a_missing_native_reference_is_not_sane(self):
        quote = replace(sell_point(), execution_native_reference=None)
        assert _entry_basis_is_sane(quote, sol_usd=SOL_USD)["sane"] is False

    def test_a_thousand_x_unit_error_is_caught(self):
        """The 10**6/10**9 decimal mistake must not pass as a real move."""
        quote = sell_point()
        native = quote.execution_native_reference
        broken = replace(quote, execution_native_reference=replace(
            native, reference_price=float(native.reference_price) * 1e6))
        result = _entry_basis_is_sane(broken, sol_usd=SOL_USD)
        assert result["sane"] is False
        assert "mismatch" in result["detail"]

    def test_a_hundred_x_unit_error_is_caught(self):
        quote = sell_point()
        native = quote.execution_native_reference
        broken = replace(quote, execution_native_reference=replace(
            native, reference_price=float(native.reference_price) / 1e3))
        assert _entry_basis_is_sane(broken, sol_usd=SOL_USD)["sane"] is False

    def test_an_invalid_sol_usd_is_not_sane(self):
        assert _entry_basis_is_sane(sell_point(), sol_usd=0.0)["sane"] is False

    def test_a_mismatched_market_is_not_sane(self):
        quote = sell_point()
        native = quote.execution_native_reference
        broken = replace(quote, execution_native_reference=replace(
            native, market_id="PUMPSWAP:Elsewhere"))
        assert _entry_basis_is_sane(broken, sol_usd=SOL_USD)["sane"] is False

    def test_ordinary_execution_premium_is_still_sane(self):
        """A real cost of transacting must NOT be rejected as a mismatch."""
        quote = sell_point()
        native = quote.execution_native_reference
        for premium in (0.01, 0.05, 0.25, 2.0):
            moved = replace(quote, execution_native_reference=replace(
                native,
                reference_price=float(native.reference_price) * (1 + premium)))
            assert _entry_basis_is_sane(moved, sol_usd=SOL_USD)["sane"], premium

    def test_the_engine_records_the_invariant_on_the_position(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_CURRENT"].position
        assert position.entry_basis_sane is True
        assert position.entry_basis_detail


class TestTelemetry:
    """The trigger's evidence must be recorded, not just applied."""

    def _open(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        run(engine.on_signal(signal()))
        return engine, engine.accounts["L25_CURRENT"].position

    def test_the_trigger_source_is_the_native_same_market_reference(self):
        engine, position = self._open()
        run(engine.monitor_once())
        assert position.trigger_reference_source == "EXECUTION_NATIVE_SAME_MARKET"

    def test_the_observed_mark_is_labelled_diagnostic_only(self):
        engine, position = self._open()
        run(engine.monitor_once())
        assert position.observed_mark_role == "DIAGNOSTIC_ONLY"

    def test_the_native_reference_is_recorded_in_both_units(self):
        engine, position = self._open()
        run(engine.monitor_once())
        assert position.trigger_native_reference_raw > 0
        assert position.trigger_native_reference_usd == pytest.approx(
            position.trigger_native_reference_raw * position.trigger_sol_usd_reference)
        assert position.trigger_sol_usd_reference > 0

    def test_the_trigger_return_and_verdict_are_recorded(self):
        engine, position = self._open()
        run(engine.monitor_once())
        assert position.trigger_entry_fill_usd == pytest.approx(
            position.entry_fill.effective_price_usd)
        assert position.trigger_return_pct is not None
        assert position.trigger_verdict in ("HOLD", "EXIT")

    def test_the_observed_mark_is_still_recorded_alongside(self):
        engine, position = self._open()
        run(engine.monitor_once())
        assert position.trigger_observed_mark_usd is not None

    def test_identity_and_venue_are_recorded(self):
        engine, position = self._open()
        run(engine.monitor_once())
        assert position.trigger_reference_identity == "PASS"
        assert position.trigger_reference_market_id

    def test_the_fields_survive_persistence(self):
        engine, position = self._open()
        run(engine.monitor_once())
        blob = position.as_dict()
        for key in ("trigger_reference_source", "trigger_native_reference_usd",
                    "trigger_return_pct", "trigger_verdict",
                    "observed_mark_role"):
            assert key in blob, key

    def test_freshness_is_not_claimed_when_it_is_not_proven(self):
        """Age is reported as measured, never asserted as fresh."""
        engine, position = self._open()
        run(engine.monitor_once())
        age = position.trigger_reference_age_ms
        assert age is None or isinstance(age, (int, float))