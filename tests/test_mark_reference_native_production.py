"""
The production MARK_GAP guard uses the EXECUTION-NATIVE SAME-MARKET reference.

WHY THIS FILE EXISTS
====================
The guard's reference has been promoted from the external DexScreener observed
mark to the same market's own on-chain spot. That change is invisible to a
verdict-only suite: every offline test injects a fake quote service and most
never reach the real guard wiring, so a silent edit in EITHER direction would
pass. These tests are built to fail on that class of regression in both:

* they call the REAL ``PaperLiveEngine._mark_gap_guard``;
* they assert on the SOURCE of the production guard, not merely its verdict;
* the decisive case makes the two references disagree in OPPOSITE directions,
  so a swapped guard cannot accidentally agree;
* the legacy comparison is still computed, still persisted, and is pinned as
  DIAGNOSTIC-ONLY so it cannot quietly become the decision again.

This file previously pinned the OPPOSITE decision (legacy mark in production,
native as diagnostic). It was converted rather than deleted, because the
reasoning that justified it -- "nothing asserts which evaluator the production
path calls" -- applies symmetrically to the new state, and every invariant in
it about mark age, order size, rejection evidence and REAL staying disarmed
survives the promotion unchanged.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from autopaper.execution.solana import WSOL_MINT
from autopaper.paper_live.economics import MARK_GAP_GUARD_THRESHOLD_PCT
from autopaper.paper_live.engine import (
    PRODUCTION_GUARD_SOURCE,
    PaperLiveEngine,
    _mark_age_status,
    _mark_source_age_ms,
    _mark_source_observed_at,
)
from autopaper.paper_live.schema import (
    ExecutionNativeReference,
    LatencyMarks,
    ObservedMarkPoint,
    QuotePoint,
    SignalRecord,
)

ENGINE_SRC = Path("src/autopaper/paper_live/engine.py")
MARKET_ID = "PUMPSWAP:POOLADDR"


# ---------------------------------------------------------------------------
# doubles built from the REAL structures
# ---------------------------------------------------------------------------


def real_quote(*, effective_price=0.0002, effective_price_usd=None,
               identity="PASS", available=True, native=None,
               input_amount_raw=4_175_000, output_decimals=6):
    """A QuotePoint with the real field set, not a permissive mock."""
    return QuotePoint(
        side="BUY", provider="pumpswap_direct", market_type="POOL",
        token_mint="MINT", quote_mint=WSOL_MINT,
        input_amount_raw=input_amount_raw, input_decimals=9,
        input_amount_decimal=0.004175, output_amount_raw=5000,
        output_decimals=output_decimals, output_amount_decimal=0.005,
        effective_price=effective_price,
        effective_price_usd=(effective_price_usd if effective_price_usd
                             is not None else effective_price * 150.0),
        price_impact_pct=1.26, fee_bps=25, fee_amount=1,
        venue="pumpswap", pool_or_curve="POOLADDR",
        venue_id=MARKET_ID, route=(),
        identity=identity, available=available, missing_reason=None,
        execution_native_reference=native,
    )


def native_reference(*, price, market_id=MARKET_ID, decimals=6,
                     quote_mint=WSOL_MINT, side="BUY", slot=453_107_084):
    """An ExecutionNativeReference built from the real field set."""
    return ExecutionNativeReference(
        market_id=market_id, mint="MINT", base_mint="MINT",
        quote_mint=quote_mint, side=side,
        reference_price=price,
        numerator_raw=30_000_000_000, denominator_raw=1_500_000_000_000,
        calculation_method="PUMPSWAP_EFFECTIVE_RESERVES",
        venue="pumpswap", pool_or_curve="POOLADDR",
        provider="pumpswap_direct", slot=slot,
        observed_at=1000.0, state_age_ms=5.0, token_decimals=decimals,
    )


def real_mark(price, *, available=True):
    return ObservedMarkPoint(
        value_usd=price, pair_address="POOLADDR", dex_id="pumpswap",
        liquidity_usd=18000.0, volume_5m_usd=18000.0,
        market_cap_usd=20000.0, timestamp=1000.0, available=available,
    )


def engine():
    """A real engine with no collaborators, for the real guard methods."""
    return PaperLiveEngine(arms=(), decision_engine=None, exit_policy=None,
                           quote_service=None)


# ---------------------------------------------------------------------------
# 1-3. production guard uses the SAME-MARKET reference; legacy is diagnostic
# ---------------------------------------------------------------------------


class TestProductionGuardSource:
    def test_guard_source_constant_names_the_same_market_reference(self):
        assert PRODUCTION_GUARD_SOURCE == "EXECUTION_NATIVE_SAME_MARKET"

    def test_threshold_is_unchanged_at_ten_percent(self):
        assert MARK_GAP_GUARD_THRESHOLD_PCT == 10.0

    def test_production_guard_calls_the_execution_native_evaluator(self):
        """
        Static check on the REAL source.

        A guard edit is invisible to a verdict-only suite, so this pins WHICH
        evaluator the production path calls -- in both directions.
        """
        src = inspect.getsource(PaperLiveEngine._mark_gap_guard)
        assert "evaluate_execution_native_gap" in src, (
            "the production guard no longer calls the execution-native "
            "evaluator")
        assert "evaluate_mark_gap" not in src, (
            "the PRODUCTION guard is wired back to the legacy aggregator-mark "
            "evaluator -- that is the change this file exists to prevent")
        body = src.split('"""')[2]
        assert "execution_native_reference" in body, (
            "the production guard body must read the native reference")
        assert "mark.value_usd" not in body, (
            "the production guard body must not read the DexScreener mark")

    def test_legacy_comparison_lives_in_a_separate_diagnostic_method(self):
        src = inspect.getsource(PaperLiveEngine._legacy_gap_guard_diagnostic)
        assert "evaluate_mark_gap" in src
        assert "execution_native_reference" not in src, (
            "the diagnostic must not quietly become the native path")


# ---------------------------------------------------------------------------
# 4-9. the two references disagree, in OPPOSITE directions
# ---------------------------------------------------------------------------


class TestReferencesDisagreeOppositeWays:
    def test_a_lagging_mark_rejects_what_the_same_market_accepts(self):
        """
        THE DECISIVE CASE, exercising the REAL guard wiring.

            legacy mark      = 100   (USD/token, a lagging external snapshot)
            native reference = 120   (same market, same slot)
            executable       = 121   (same market, exact size)

            legacy gap = +21.00%  -> the OLD guard would REJECT
            native gap =  +0.83%  -> the NEW production guard PASSES

        This is exactly the historical failure: an authorized L25 entry refused
        because a third party's snapshot trailed the market it described. The
        production verdict must be the native one, and it must be PASS.
        """
        e = engine()
        q = real_quote(effective_price=1.21, effective_price_usd=121.0,
                       native=native_reference(price=1.20))
        m = real_mark(100.0)
        verdict = e._mark_gap_guard(q, m)
        assert verdict.mark_price == pytest.approx(1.20), (
            "the guard must have measured against the native reference")
        assert verdict.gap_pct == pytest.approx((1.21 / 1.20 - 1) * 100)
        assert verdict.result == "PASS"

    def test_the_legacy_gap_on_that_same_record_would_have_rejected(self):
        """The OLD comparison, on the same quote and mark, still says REJECT."""
        e = engine()
        q = real_quote(effective_price=1.21, effective_price_usd=121.0,
                       native=native_reference(price=1.20))
        legacy = e._legacy_gap_guard_diagnostic(q, real_mark(100.0))
        assert legacy.gap_pct == pytest.approx(21.0), (
            "the legacy gap is USD/token vs USD/token; the USD leg is what the "
            "old guard compared")
        assert legacy.result == "REJECT", (
            "the diagnostic must still be able to say REJECT -- that is the "
            "research signal we keep")

    def test_the_opposite_direction_also_flips(self):
        """
        The mirror case, so a swapped guard cannot coincidentally agree.

            legacy mark      = 120
            native reference = 100
            executable       = 105

            legacy gap = -12.5% -> PASS (a quote 12.5% CHEAPER than the mark)
            native gap =  +5.0% -> PASS

        Both pass, but for OPPOSITE reasons. The guard reads the native side.
        """
        e = engine()
        verdict = e._mark_gap_guard(
            real_quote(effective_price=1.05,
                       native=native_reference(price=1.00)),
            real_mark(120.0))
        assert verdict.mark_price == pytest.approx(1.00)
        assert verdict.gap_pct == pytest.approx(5.0)
        assert verdict.result == "PASS"

    def test_a_genuinely_expensive_entry_is_still_rejected(self):
        """
        The guard must not have been weakened. A real +25% cost against its
        own market is refused, which is the whole point of the promotion.
        """
        e = engine()
        verdict = e._mark_gap_guard(
            real_quote(effective_price=1.25,
                       native=native_reference(price=1.00)),
            real_mark(100.0))
        assert verdict.gap_pct == pytest.approx(25.0)
        assert verdict.result == "REJECT"
        assert "25.00%" in verdict.reason

    def test_the_strict_boundary_is_unchanged(self):
        """
        Strict ``>`` is preserved: just under 10.0% passes, just over rejects.

        NOTE ON THE EXACT-EQUALITY CASE. A ratio of exactly 10.0% is not
        representable in binary floating point -- ``1.10/1.00 - 1`` evaluates to
        ``0.10000000000000009`` -- so a candidate landing on mathematically
        exactly 10.0% is decided by float noise rather than by the comparison.
        That is a property of the arithmetic, not of the reference change, and
        it is not reachable in production: a venue fee makes the real gap ~1.26%
        or ~0.90%, never 10.000000%. It is recorded here rather than papered
        over with a tolerance that would weaken the boundary being tested.
        """
        e = engine()
        assert e._mark_gap_guard(
            real_quote(effective_price=1.0999,
                       native=native_reference(price=1.00)),
            real_mark(100.0)).result == "PASS"
        assert e._mark_gap_guard(
            real_quote(effective_price=1.1001,
                       native=native_reference(price=1.00)),
            real_mark(100.0)).result == "REJECT"

    def test_the_dexscreener_mark_cannot_move_the_verdict(self):
        """
        The mark is now diagnostic. Moving it from 100 to 1 must not change a
        single production verdict -- in either direction.
        """
        e = engine()
        q = real_quote(effective_price=1.05,
                       native=native_reference(price=1.00))
        for mark_price in (1.0, 100.0, 1_000_000.0):
            verdict = e._mark_gap_guard(q, real_mark(mark_price))
            assert verdict.gap_pct == pytest.approx(5.0)
            assert verdict.result == "PASS"


# ---------------------------------------------------------------------------
# 10. fail closed -- no fallback of any kind
# ---------------------------------------------------------------------------


class TestFailClosed:
    def test_a_missing_native_reference_refuses(self):
        e = engine()
        verdict = e._mark_gap_guard(real_quote(effective_price=1.05),
                                    real_mark(100.0))
        assert verdict.result == "UNKNOWN"
        assert verdict.gap_pct is None
        assert "observed mark is NOT used as a substitute" in verdict.reason

    def test_a_reference_from_another_pool_is_a_mismatch_not_a_pass(self):
        e = engine()
        verdict = e._mark_gap_guard(
            real_quote(effective_price=1.05,
                       native=native_reference(price=1.00,
                                               market_id="PUMPSWAP:OTHERPOOL")),
            real_mark(100.0))
        assert verdict.result == "UNKNOWN"
        assert "not the executable market" in verdict.reason

    def test_a_decimals_mismatch_refuses(self):
        e = engine()
        verdict = e._mark_gap_guard(
            real_quote(effective_price=1.05,
                       native=native_reference(price=1.00, decimals=9)),
            real_mark(100.0))
        assert verdict.result == "UNKNOWN"
        assert "decimals" in verdict.reason

    def test_a_quote_asset_or_side_mismatch_refuses(self):
        e = engine()
        assert e._mark_gap_guard(
            real_quote(effective_price=1.05,
                       native=native_reference(price=1.00,
                                               quote_mint="EPjFWdd5")),
            real_mark(100.0)).result == "UNKNOWN"
        assert e._mark_gap_guard(
            real_quote(effective_price=1.05,
                       native=native_reference(price=1.00, side="SELL")),
            real_mark(100.0)).result == "UNKNOWN"

    def test_a_stale_quote_refuses(self):
        e = engine()
        # ``stale`` is a derived property, so it is driven through the fields
        # it derives from rather than set directly.
        q = replace(real_quote(effective_price=1.05,
                               native=native_reference(price=1.00)),
                    max_quote_age_ms=1.0, quote_age_ms=9_999.0)
        assert q.stale is True
        assert e._mark_gap_guard(q, real_mark(100.0)).result == "UNKNOWN"


# ---------------------------------------------------------------------------
# 11-12. order size is derived from the quote, not asserted
# ---------------------------------------------------------------------------


class TestSameOrderSize:
    def test_a_quote_carrying_no_size_is_not_treated_as_comparable(self):
        e = engine()
        q = replace(real_quote(effective_price=1.05,
                               native=native_reference(price=1.00)),
                    input_amount_raw=None)
        assert e._mark_gap_guard(q, real_mark(100.0)).result == "UNKNOWN"

    def test_a_sized_quote_is_comparable(self):
        e = engine()
        verdict = e._mark_gap_guard(
            real_quote(effective_price=1.05,
                       native=native_reference(price=1.00)),
            real_mark(100.0))
        assert verdict.result == "PASS"

    def test_guard_never_passes_a_literal_same_order_size(self):
        src = inspect.getsource(PaperLiveEngine._mark_gap_guard)
        assert "same_order_size=True" not in src
        assert "same_order_size=same_order_size" in src

    def test_the_legacy_diagnostic_still_reports_its_own_size_basis(self):
        src = inspect.getsource(PaperLiveEngine._legacy_gap_guard_diagnostic)
        assert "same_order_size=self._same_order_size_basis(quote)" in src
        assert "same_order_size=True" not in src

    def test_the_native_gap_equals_the_measured_price_impact(self):
        """
        §7: verify the calculation from raw quote data rather than trusting that
        the field merely exists.

        For a direct quote, ``price_impact_pct`` IS
        ``(effective_price / spot - 1) * 100`` -- the same arithmetic the guard
        performs. If the two ever disagree, one of them is wrong.
        """
        from decimal import Decimal

        from autopaper.same_market.models import MarketKind  # noqa: F401
        from autopaper.same_market.pumpdirect import (
            BondingCurve, Fees, quote_bonding_curve_buy,
        )
        import time as _time

        curve = BondingCurve(
            address="C" * 44, mint="M" * 44,
            virtual_token_reserves=1_500_000_000_000,
            virtual_quote_reserves=30_000_000_000,
            real_token_reserves=800_000_000_000,
            real_quote_reserves=20_000_000_000,
            token_total_supply=10 ** 15, complete=False,
            creator="C" * 43 + "D", is_mayhem_mode=False,
            is_cashback_coin=False, quote_mint=WSOL_MINT,
            creator_fee_bps=None, can_edit_creator_fee=None,
            account_len=200, slot=453_107_084, timestamp=_time.time(),
        )
        result = quote_bonding_curve_buy(
            curve, sol_amount_lamports=4_186_161, token_decimals=6,
            fees=Fees(lp_fee_bps=0, protocol_fee_bps=125, creator_fee_bps=0),
            mint_supply=10 ** 15,
        )
        recomputed = (result.effective_price / result.spot_price - 1) * 100
        assert float(recomputed) == pytest.approx(
            float(result.price_impact_pct), abs=1e-9)
        assert result.spot_numerator_raw == curve.virtual_quote_reserves
        assert result.spot_denominator_raw == curve.virtual_token_reserves
        assert Decimal(result.spot_price) > 0


# ---------------------------------------------------------------------------
# 13-14. no extra RPC; the reference rides the quote
# ---------------------------------------------------------------------------


class TestNoExtraRead:
    def test_the_reference_is_produced_by_the_quote_itself(self):
        """
        §8/§13: the reference must not be a second read of the same account.
        It is derived inside the quote call from the same decoded state, so it
        carries the quote's slot.
        """
        from autopaper.same_market.pumpdirect import quote_bonding_curve_buy

        curve_spot = quote_bonding_curve_buy.__doc__
        assert curve_spot is not None
        # the spot is computed from the curve object passed IN, not re-read
        src = inspect.getsource(quote_bonding_curve_buy)
        assert "curve.virtual_quote_reserves" in src
        assert "await" not in src, (
            "the bonding-curve quote must contain no await of its own; a second "
            "read is what this design exists to avoid")

    def test_the_reference_carries_the_quote_slot(self):
        q = real_quote(effective_price=1.05,
                       native=native_reference(price=1.00, slot=453_107_084))
        assert q.execution_native_reference.slot == 453_107_084
        assert q.state_slot is None or q.state_slot == 453_107_084


# ---------------------------------------------------------------------------
# 15. diagnostics persisted; evidence survives a refusal
# ---------------------------------------------------------------------------


class TestRejectionEvidence:
    def test_the_quote_is_never_replaced_by_a_placeholder(self):
        src = inspect.getsource(PaperLiveEngine._open_for_arm)
        assert "buy_quote=(quote if quote is not None else QuotePoint(" in src

    def test_both_gaps_and_both_sources_are_persisted(self):
        names = {f.name for f in SignalRecord.__dataclass_fields__.values()}
        assert {"execution_native_gap_pct", "legacy_observed_mark_gap_pct",
                "native_guard_verdict", "production_guard_source"} <= names

    def test_latency_records_the_production_guard_source(self):
        names = {f.name for f in LatencyMarks.__dataclass_fields__.values()}
        assert {"production_guard_source", "execution_native_gap_pct",
                "legacy_observed_mark_gap_pct"} <= names

    def test_the_refusal_reason_names_the_native_path(self):
        src = inspect.getsource(PaperLiveEngine._open_for_arm)
        assert "_guard_failure_reason(guard)" in src

    def test_a_missing_reference_is_reported_as_unavailable(self):
        from autopaper.paper_live.engine import _guard_failure_reason

        class V:
            result = "UNKNOWN"
            reason = ("the executable market's own current price could not be "
                      "read from chain, so there is no same-market reference")
        assert _guard_failure_reason(V()).value == (
            "EXECUTION_NATIVE_REFERENCE_UNAVAILABLE")

    def test_a_pool_mismatch_is_reported_as_a_mismatch(self):
        from autopaper.paper_live.engine import _guard_failure_reason

        class V:
            result = "UNKNOWN"
            reason = ("the reference market PUMPSWAP:X is not the executable "
                      "market, so their price difference is not comparable")
        assert _guard_failure_reason(V()).value == (
            "EXECUTION_NATIVE_REFERENCE_MISMATCH")

    def test_a_genuine_reject_is_still_named_mark_gap(self):
        from autopaper.paper_live.engine import _guard_failure_reason

        class V:
            result = "REJECT"
            reason = "the executable BUY is 25.00% above ..."
        assert _guard_failure_reason(V()).value == "MARK_GAP_GUARD_REJECTED"


# ---------------------------------------------------------------------------
# 16-17. mark age honesty survives; invariants hold
# ---------------------------------------------------------------------------


class TestHonestMarkAge:
    def test_absent_source_timestamp_yields_unknown_not_zero(self):
        class Obs:
            price_usd = 1.0
        assert _mark_source_observed_at(Obs()) is None
        assert _mark_age_status(Obs()) == "UNKNOWN"

    def test_present_source_timestamp_yields_known_and_computed_age(self):
        class Obs:
            price_usd = 1.0
            source_observed_at = 998.0
        assert _mark_age_status(Obs()) == "KNOWN"
        assert _mark_source_age_ms(Obs(), 1000.0) == pytest.approx(2000.0)

    def test_pair_created_at_is_never_treated_as_observation_time(self):
        """
        ``pair_created_at`` dates the PAIR, not the price print. Using it would
        be a category error, so it must not be read as a source timestamp.
        """
        class Obs:
            price_usd = 1.0
            pair_created_at = 1.0
        assert _mark_source_observed_at(Obs()) is None
        assert _mark_age_status(Obs()) == "UNKNOWN"

    def test_future_source_timestamp_is_not_reported_as_negative_age(self):
        class Obs:
            price_usd = 1.0
            source_observed_at = 2000.0
        assert _mark_source_age_ms(Obs(), 1000.0) is None

    def test_default_mark_point_is_unknown_not_fresh(self):
        m = ObservedMarkPoint(value_usd=1.0, pair_address="P", dex_id="d",
                              liquidity_usd=1.0, volume_5m_usd=1.0,
                              market_cap_usd=1.0, timestamp=1.0,
                              available=True)
        assert m.age_status == "UNKNOWN"
        assert m.age_ms is None
        assert m.source_observed_at is None


class TestInvariants:
    def test_real_remains_disabled(self):
        from autopaper.engine import PaperTrader
        assert getattr(PaperTrader, "REAL_ENABLED_DEFAULT", False) is False

    def test_the_threshold_constant_is_untouched(self):
        assert "MARK_GAP_GUARD_THRESHOLD_PCT = 10.0" in (
            Path("src/autopaper/paper_live/economics.py").read_text())

    def test_the_guard_touches_no_strategy_or_sizing_symbol(self):
        """
        The ONLY intended behavioural change is the reference. Nothing in the
        guard may name L25, sizing, fees or exits.
        """
        src = inspect.getsource(PaperLiveEngine._mark_gap_guard)
        for forbidden in ("compute_position_size", "L25_", "size_tradeable",
                          "exit_style", "disaster", "min_hold", "time_decay"):
            assert forbidden not in src, (
                f"the guard now references {forbidden!r}; the reference change "
                "must not reach strategy or sizing")

    def test_engine_module_still_parses_and_imports(self):
        ast.parse(ENGINE_SRC.read_text())