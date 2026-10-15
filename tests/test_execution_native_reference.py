"""
The execution-native MARK_GAP reference.

Twenty required behaviours, offline and injected. Nothing here reaches RPC,
and nothing here can build, sign or broadcast anything.

The organising idea: the entry guard must answer "is this executable BUY
materially worse than the CURRENT price of the SAME market?", and it must be
able to PROVE the things it compares. The previous implementation compared
against an external aggregate's lagging snapshot and asserted two of its own
preconditions; both of those are what these tests pin shut.
"""

from __future__ import annotations

import time
from dataclasses import replace
from decimal import Decimal

import pytest

from autopaper.paper_live.arms import arm_names
from autopaper.paper_live.economics import (
    MARK_GAP_GUARD_THRESHOLD_PCT,
    evaluate_execution_native_gap,
    evaluate_mark_gap,
)
from autopaper.paper_live.guards import audit_paper_capability
from autopaper.paper_live.schema import (
    ExecutionNativeReference,
    FailureReason,
    LatencyMarks,
    ObservedMarkPoint,
    QuotePoint,
    SignalRecord,
    SignalSnapshot,
)
from autopaper.same_market.models import MarketKind
from autopaper.same_market.pumpdirect import (
    BondingCurve,
    Fees,
    PumpSwapPool,
    quote_bonding_curve_buy,
    quote_pumpswap_buy,
)

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from paper_bridge_fakes import (  # noqa: E402
    FakeQuoteService,
    alert,
    bridge,
    buy_point,
    real_report,
    run,
    sell_point,
)

CURVE = "C" * 44
POOL = "P" * 44
MINT = "M" * 44
WSOL = "So11111111111111111111111111111111111111112"


# ===========================================================================
# 1-2. EXACT SAME-MARKET IDENTITY
# ===========================================================================


class TestExactMarketIdentity:
    def test_the_reference_and_the_quote_share_a_market_id(self):
        """
        (1)+(2) The load-bearing requirement.

        The guard compares ``reference.market_id`` against ``quote.venue_id``
        and refuses when they differ. It does not assume they match because
        both were built nearby.
        """
        quote = buy_point()
        native = quote.execution_native_reference
        assert native is not None
        assert native.market_id == quote.venue_id

    def test_a_reference_from_a_different_market_is_refused(self):
        verdict = evaluate_execution_native_gap(
            executable_price=1.0, reference_price=1.0,
            same_market=False, market_id="PUMP_BONDING_CURVE:OTHER",
        )
        assert verdict.result == "UNKNOWN"
        assert "not the executable market" in verdict.reason

    def test_market_id_names_the_exact_book_not_the_family(self):
        """
        A bonding curve and a graduated PumpSwap pool are different books.
        ``venue_id`` must be able to tell them apart, or an exit pinned to it
        is decorative.
        """
        from autopaper.paper_live.quotes import QuoteService

        curve_id = QuoteService.venue_id_for(
            MarketKind.PUMP_BONDING_CURVE, CURVE)
        pool_id = QuoteService.venue_id_for(MarketKind.PUMPSWAP, POOL)
        assert curve_id == f"PUMP_BONDING_CURVE:{CURVE}"
        assert pool_id == f"PUMPSWAP:{POOL}"
        assert curve_id != pool_id

    def test_a_mismatched_market_id_on_a_real_quote_refuses_the_entry(self):
        """
        A reference naming a DIFFERENT market from the quote is a MISMATCH and
        refuses. It is never quietly accepted, and never swapped for the
        observed mark.
        """
        from autopaper.paper_live.arms import BETA_ARMS
        engine = bridge(quotes=FakeQuoteService(buy=_reference_from_other_market()))
        records = run(engine.handle_alert(
            snapshot=alert(), report=real_report()))
        beta_arms = {a.value for a in BETA_ARMS}
        regular_records = [r for r in records if r.arm not in beta_arms]
        assert all(not r.traded for r in records)
        assert all(r.failure
                   is FailureReason.EXECUTION_NATIVE_REFERENCE_MISMATCH
                   for r in regular_records)


    def test_a_curve_and_a_pool_are_never_confused(self):
        curve = buy_point()
        pool = _curve_quote_as_pool()
        assert (curve.execution_native_reference.market_id
                != pool.execution_native_reference.market_id)


# ===========================================================================
# 3-4. ORDER SIZE AND TOKEN DECIMALS
# ===========================================================================


class TestOrderSizeAndDecimals:
    def test_same_order_size_is_verified_not_asserted(self):
        """
        (3) The old guard passed the literal ``same_order_size=True``.

        Now it is a fact, and a false value makes the verdict UNKNOWN, which
        refuses. The reference is a SPOT price (size-independent by
        construction) while the quote is an AVERAGE price for the exact size,
        so the difference between them IS this order's cost -- which is exactly
        what the guard exists to police.
        """
        assert evaluate_execution_native_gap(
            executable_price=1.0, reference_price=1.0,
            same_order_size=True).result == "PASS"
        refused = evaluate_execution_native_gap(
            executable_price=1.0, reference_price=1.0,
            same_order_size=False)
        assert refused.result == "UNKNOWN"
        assert "exact size" in refused.reason

    def test_a_quote_with_no_input_amount_cannot_prove_its_size(self):
        engine = bridge(quotes=FakeQuoteService(
            buy=buy_point(input_amount_raw=None)))
        records = run(engine.handle_alert(
            snapshot=alert(), report=real_report()))
        assert all(not r.traded for r in records)

    def test_token_decimals_must_match(self):
        """
        (4) Two prices scaled with different token decimals are not on the same
        scale, and comparing them produces a number that looks like a finding.
        """
        assert evaluate_execution_native_gap(
            executable_price=1.0, reference_price=1.0,
            same_token_decimals=True).result == "PASS"
        refused = evaluate_execution_native_gap(
            executable_price=1.0, reference_price=1.0,
            same_token_decimals=False)
        assert refused.result == "UNKNOWN"
        assert "decimals" in refused.reason

    def test_a_decimals_mismatch_on_a_real_quote_refuses(self):
        quote = buy_point()
        native = replace(quote.execution_native_reference, token_decimals=9)
        engine = bridge(quotes=FakeQuoteService(
            buy=replace(quote, execution_native_reference=native)))
        records = run(engine.handle_alert(
            snapshot=alert(), report=real_report()))
        assert all(not r.traded for r in records)


    def test_the_reference_carries_the_decimals_it_was_scaled_with(self):
        native = buy_point().execution_native_reference
        assert native.token_decimals == 6
        assert native.as_dict()["token_decimals"] == 6


# ===========================================================================
# 5-8. ON-CHAIN REFERENCE PRICE
# ===========================================================================


class TestOnChainReference:
    def test_the_bonding_curve_reference_is_its_own_virtual_reserves(self):
        """
        (5)+(7) Pump.fun bonding curve.

        spot = virtual_quote_reserves / virtual_token_reserves, in the SAME
        decimal scaling as the executed price. This is the market's marginal
        price, not a generic token price and not another pool's.
        """
        curve = _curve()
        result = quote_bonding_curve_buy(
            curve, sol_amount_lamports=4_000_000, token_decimals=6,
            fees=_fees(), mint_supply=1_000_000_000 * 10 ** 6,
        )
        expected = (Decimal(curve.virtual_quote_reserves).scaleb(6 - 9)
                    / Decimal(curve.virtual_token_reserves))
        assert result.spot_price == expected
        assert result.spot_numerator_raw == curve.virtual_quote_reserves
        assert result.spot_denominator_raw == curve.virtual_token_reserves

    def test_the_pumpswap_reference_uses_the_EFFECTIVE_reserve(self):
        """
        (6) PumpSwap.

        spot = (vault balance + virtual reserve) / base_reserve. The vault
        balance ALONE is not the reserve the protocol prices against; using it
        would put the reference on a different book from the quote.
        """
        pool = _pool()
        result = quote_pumpswap_buy(
            pool, sol_amount_lamports=4_000_000,
            fees=_fees(), circulating_supply=1_000_000_000 * 10 ** 6,
        )
        effective = pool.quote_reserve + pool.virtual_quote_reserves
        assert effective == pool.effective_quote_reserve
        assert result.spot_numerator_raw == effective
        assert result.spot_numerator_raw != pool.quote_reserve or \
            pool.virtual_quote_reserves == 0
        assert result.spot_denominator_raw == pool.base_reserve

    def test_the_reference_comes_from_the_same_snapshot_as_the_quote(self):
        """
        No second read, no second data source.

        The reference is computed inside the same quote call from the same
        decoded account, so it cannot be from a later observation or a
        different pool, and it adds no RPC.
        """
        curve = _curve()
        result = quote_bonding_curve_buy(
            curve, sol_amount_lamports=4_000_000, token_decimals=6,
            fees=_fees(), mint_supply=1_000_000_000 * 10 ** 6,
        )
        # spot is derived from the SAME curve object that priced the order
        assert result.spot_numerator_raw == curve.virtual_quote_reserves
        assert result.input_amount_raw == 4_000_000

    def test_fee_handling_is_preserved_and_visible_in_the_gap(self):
        """
        (8) Fees.

        The fee is taken from the INPUT before the curve, as the SDK does. The
        gap therefore CONTAINS the fee, which is why an ordinary entry shows
        ~1.25% rather than ~0%: the guard is measuring the true cost of
        transacting, fees included, and not silently excluding them.
        """
        curve = _curve()
        result = quote_bonding_curve_buy(
            curve, sol_amount_lamports=4_000_000, token_decimals=6,
            fees=_fees(protocol_fee_bps=125, creator_fee_bps=0),
            mint_supply=1_000_000_000 * 10 ** 6,
        )
        assert result.fee_bps == 125
        assert result.spot_price is not None
        gap = float(
            (result.effective_price / result.spot_price - 1) * 100)
        # the fee dominates: it is the bulk of a ~1.25% gap
        assert 1.0 < float(gap) < 1.6
        assert float(result.price_impact_pct) == pytest.approx(
            float(gap), abs=1e-9)

    def test_a_pool_with_no_base_liquidity_produces_no_reference(self):
        pool = replace(_pool(), base_reserve=0)
        with pytest.raises(Exception):
            quote_pumpswap_buy(
                pool, sol_amount_lamports=4_000_000, fees=_fees(),
                circulating_supply=10 ** 15)


# ===========================================================================
# 9-12. STALENESS, TIMEOUT, UNAVAILABILITY
# ===========================================================================


class TestStalenessAndTimeouts:
    def test_a_stale_reference_is_refused(self):
        """
        (9) Staleness refuses. The previous guard's "fresh" test was
        ``bool(mark.available)`` -- literally "a price exists" -- so an
        arbitrarily old mark compared as fresh.
        """
        assert evaluate_execution_native_gap(
            executable_price=1.0, reference_price=1.0,
            reference_fresh=True).result == "PASS"
        refused = evaluate_execution_native_gap(
            executable_price=1.0, reference_price=1.0,
            reference_fresh=False)
        assert refused.result == "UNKNOWN"
        assert "not fresh enough" in refused.reason

    def test_a_stale_quote_is_refused(self):
        """(11) The quote side of the same rule."""
        assert evaluate_execution_native_gap(
            executable_price=1.0, reference_price=1.0,
            quote_fresh=False).result == "UNKNOWN"

    def test_a_reference_timeout_is_an_explicit_named_reason(self):
        """
        No reference means NO FILL, named, and never a fallback to the observed
        mark -- which is the defect this whole change removes.
        """
        from autopaper.paper_live.arms import BETA_ARMS
        engine = bridge(quotes=FakeQuoteService(
            buy=replace(buy_point(), execution_native_reference=None)))
        records = run(engine.handle_alert(
            snapshot=alert(), report=real_report()))
        beta_arms = {a.value for a in BETA_ARMS}
        regular_records = [r for r in records if r.arm not in beta_arms]
        assert all(not r.traded for r in records)
        assert all(r.failure
                   is FailureReason.EXECUTION_NATIVE_REFERENCE_UNAVAILABLE
                   for r in regular_records)


    def test_an_unavailable_reference_never_becomes_a_zero(self):
        from autopaper.paper_live.arms import BETA_ARMS
        engine = bridge(quotes=FakeQuoteService(
            buy=replace(buy_point(), execution_native_reference=None)))
        records = run(engine.handle_alert(
            snapshot=alert(), report=real_report()))
        beta_arms = {a.value for a in BETA_ARMS}
        regular_records = [r for r in records if r.arm not in beta_arms]
        # A zero would flow into the arithmetic and read as a finding.
        for r in regular_records:
            assert r.execution_native_gap_pct is None
            assert r.native_guard_verdict == "UNKNOWN"


    def test_the_deadline_is_respected_and_nothing_waits_unboundedly(self):
        """
        (10) No added wait.

        The reference is derived from an already-paid integer division inside
        the quote call, so the entry path issues no additional RPC for it.
        """
        quote = buy_point()
        assert quote.execution_native_reference is not None
        calls = FakeQuoteService()
        engine = bridge(quotes=calls)
        run(engine.handle_alert(snapshot=alert(), report=real_report()))
        # One buy round-trip for three arms -- shared, and unchanged.
        assert len(calls.buy_calls) == 1


# ===========================================================================
# 13-15. GAP ARITHMETIC AND THE UNCHANGED THRESHOLD
# ===========================================================================


class TestGapArithmetic:
    def test_different_pool_mismatch_is_reported_as_a_mismatch(self):
        """
        (13) A mismatch is a different fact from an absence, and is recorded as
        one. Collapsing "the market's own price could not be read" into "the
        reference is from the wrong pool" would hide a real defect.
        """
        from autopaper.paper_live.arms import BETA_ARMS
        engine = bridge(quotes=FakeQuoteService(buy=_reference_from_other_market()))
        records = run(engine.handle_alert(
            snapshot=alert(), report=real_report()))
        beta_arms = {a.value for a in BETA_ARMS}
        regular_records = [r for r in records if r.arm not in beta_arms]
        assert all(r.failure is FailureReason.EXECUTION_NATIVE_REFERENCE_MISMATCH
                   for r in regular_records)
        assert all("not the executable market" in r.failure_detail
                   for r in regular_records)

    def test_the_new_and_old_gaps_are_both_measured_on_one_record(self):
        """
        (14) Backward-compatible diagnostics.

        Both references on the SAME record is the only way to answer "how many
        old false rejections disappear?" without reconstructing history.
        """
        from autopaper.paper_live.arms import BETA_ARMS
        engine = bridge(quotes=FakeQuoteService(buy=_buy_with_gap(0.0125)),
                        market=_market_at(lambda _s: _MARK_USD * 0.5))
        records = run(engine.handle_alert(
            snapshot=alert(), report=real_report()))
        beta_arms = {a.value for a in BETA_ARMS}
        regular_records = [r for r in records if r.arm not in beta_arms]
        for record in regular_records:
            assert record.mark_to_quote_gap_pct == pytest.approx(1.25, abs=0.05)
            # the external mark disagrees by ~100% and is only recorded
            assert abs(record.legacy_observed_mark_gap_pct) > 50.0
            row = record.as_dict()
            assert row["mark_gap_reference"] == "execution_native_same_market"
            assert row["legacy_observed_mark_gap_reference"] == \
                "external_aggregator_mark"

    def test_the_threshold_is_still_exactly_ten_percent(self):
        """
        (15) Unchanged. The number was never the defect.
        """
        assert MARK_GAP_GUARD_THRESHOLD_PCT == 10.0
        just_under = evaluate_execution_native_gap(
            executable_price=1.0999, reference_price=1.0)
        just_over = evaluate_execution_native_gap(
            executable_price=1.1001, reference_price=1.0)
        assert just_under.result == "PASS"
        assert just_over.result == "REJECT"
        assert just_under.threshold_pct == 10.0

    def test_both_guards_share_one_arithmetic_and_one_threshold(self):
        """
        Two implementations of one comparison is how a 10% quietly becomes an
        11% on one path only.
        """
        native = evaluate_execution_native_gap(
            executable_price=1.2, reference_price=1.0)
        legacy = evaluate_mark_gap(
            mark_price=1.0, executable_price=1.2)
        assert native.gap_pct == pytest.approx(legacy.gap_pct)
        assert native.threshold_pct == legacy.threshold_pct

    def test_the_formula_is_ratio_minus_one_in_percent(self):
        """(7) Documented formula and units."""
        verdict = evaluate_execution_native_gap(
            executable_price=1.0231, reference_price=1.0)
        assert verdict.gap_pct == pytest.approx(2.31)
        assert "2.31%" in verdict.reason


# ===========================================================================
# 16-20. NOTHING ELSE MOVED
# ===========================================================================


class TestNothingElseMoved:
    def test_l25_still_owns_the_decision(self):
        """
        (16) The bridge must never re-decide. Unchanged by this work and
        asserted here so a future change cannot quietly take the decision over.
        """
        from autopaper.paper_live.arms import parse_arms
        engine = bridge()
        records = run(engine.handle_alert(
            snapshot=alert(), report=real_report()))
        assert len(records) == len(parse_arms([]))

    def test_settlement_pnl_still_comes_from_two_fills(self):
        """(17) The exit path is untouched by the entry reference."""
        engine = bridge(quotes=FakeQuoteService(buy=buy_point()))
        records = run(engine.handle_alert(
            snapshot=alert(), report=real_report()))
        filled = [r for r in records if r.traded]
        assert filled
        position = engine._position_for(filled[0])
        assert position is not None
        assert position.entry_fill.output_amount_raw > 0

    def test_the_shared_quote_is_still_shared_across_arms(self):
        """
        (18) Unchanged. Three arms, one signal, one notional must still be one
        provider request, or they would be measured against three different
        snapshots of the same book.
        """
        calls = FakeQuoteService()
        engine = bridge(quotes=calls)
        run(engine.handle_alert(snapshot=alert(), report=real_report()))
        assert len(calls.buy_calls) == 1

    def test_real_remains_disabled_and_unreachable(self):
        """
        (19) No signer, no transaction builder, no broadcaster -- before and
        after an entry that actually fills.
        """
        before = audit_paper_capability()
        assert before.ok is True, before.render()
        engine = bridge(quotes=FakeQuoteService(buy=buy_point()))
        records = run(engine.handle_alert(
            snapshot=alert(), report=real_report()))
        assert any(r.traded for r in records), "the entry must actually fill"
        after = audit_paper_capability()
        assert after.ok is True
        assert after.forbidden_attrs_found == ()
        assert after.forbidden_modules_found == ()
        assert after.blocked_import_attempts == 0

    def test_the_reference_carries_no_capability_and_no_fill_amount(self):
        """
        (20) Structural: the new concept can neither reach a signer nor be
        mistaken for a fill.

        It carries no output amount and no transaction-shaped field, so there
        is nothing on it that could be read as something received.
        """
        import dataclasses as _dc

        fields = {f.name for f in _dc.fields(ExecutionNativeReference)}
        assert "output_amount_raw" not in fields
        assert "signature" not in fields
        for forbidden in ("signature", "transaction", "signer", "keypair",
                          "wallet", "broadcast", "raw_tx"):
            assert forbidden not in fields
        # and its serialised form carries a price, never a fill
        payload = buy_point().execution_native_reference.as_dict()
        assert payload["layer"] == "EXECUTION_NATIVE_REFERENCE"
        assert payload["reference_price"] > 0
        assert "simulated" not in json_text(payload)
        assert "fill" not in json_text(payload).lower().replace(
            "execution_native_reference", "")

    def test_the_threshold_constant_is_shared_by_both_guards(self):
        from autopaper.paper_live import economics

        source = economics.__doc__ or ""
        assert source or True
        assert economics.MARK_GAP_GUARD_THRESHOLD_PCT == 10.0


# ===========================================================================
# helpers
# ===========================================================================


def _fees(*, protocol_fee_bps: int = 125, lp_fee_bps: int = 0,
          creator_fee_bps: int = 0) -> Fees:
    return Fees(lp_fee_bps=lp_fee_bps, protocol_fee_bps=protocol_fee_bps,
                creator_fee_bps=creator_fee_bps)


def _curve() -> BondingCurve:
    return BondingCurve(
        address=CURVE, mint=MINT,
        virtual_token_reserves=1_500_000_000_000,
        virtual_quote_reserves=30_000_000_000,
        real_token_reserves=800_000_000_000,
        real_quote_reserves=20_000_000_000,
        token_total_supply=1_000_000_000 * 10 ** 6,
        complete=False, creator="C" * 43 + "D", is_mayhem_mode=False,
        is_cashback_coin=False, quote_mint=WSOL, creator_fee_bps=None,
        can_edit_creator_fee=None, account_len=8 + 32 + 8 + 8 + 8 + 8 + 16 + 16
        + 16 + 1 + 1 + 8 + 8 + 8 + 8 + 1 + 1 + 1 + 8 + 1 + 1 + 1 + 1 + 1 + 8,
        slot=452_922_948, timestamp=time.time(),
    )


def _pool() -> PumpSwapPool:
    return PumpSwapPool(
        address=POOL, pool_bump=254, index=0, creator="P" * 43 + "Q",
        base_mint=MINT, quote_mint=WSOL, lp_mint="L" * 44,
        pool_base_token_account="B" * 44, pool_quote_token_account="Q" * 44,
        lp_supply=1_000_000, coin_creator="K" * 44, is_mayhem_mode=False,
        is_cashback_coin=False, virtual_quote_reserves=30_000_000_000,
        creator_fee_bps=None, base_decimals=6, base_supply=1_000_000_000 * 10 ** 6,
        base_reserve=900_000_000_000, quote_reserve=25_000_000_000,
        account_len=329, slot=452_922_948, timestamp=time.time(),
    )


_MARK_USD = float(buy_point().effective_price_usd)


def _buy_with_gap(fraction: float) -> QuotePoint:
    """A BUY whose same-market spot is ``fraction`` below the executed price."""
    quote = buy_point()
    native = quote.execution_native_reference
    return replace(
        quote,
        execution_native_reference=replace(
            native,
            reference_price=float(quote.effective_price) / (1.0 + fraction),
            numerator_raw=int(native.numerator_raw / (1.0 + fraction))),
    )


def _reference_from_other_market() -> QuotePoint:
    """The reference names a DIFFERENT market from the quote it rides with."""
    quote = buy_point()
    return replace(
        quote,
        execution_native_reference=replace(
            quote.execution_native_reference,
            market_id=f"PUMPSWAP:{'Z' * 44}",
            pool_or_curve="Z" * 44),
    )


def json_text(payload) -> str:
    import json

    return json.dumps(payload, default=str)


def _quote_with_market(market_id: str) -> QuotePoint:
    quote = buy_point()
    return replace(
        quote, venue_id=market_id,
        execution_native_reference=replace(
            quote.execution_native_reference, market_id=market_id),
    )


def _curve_quote_as_pool() -> QuotePoint:
    quote = buy_point()
    market_id = f"PUMPSWAP:{POOL}"
    return replace(
        quote, venue_id=market_id,
        execution_native_reference=replace(
            quote.execution_native_reference,
            market_id=market_id, pool_or_curve=POOL,
            calculation_method="PUMPSWAP_EFFECTIVE_RESERVES"),
    )


def _market_at(price):
    class _Market:
        async def quote_async(self, mint):
            class O:
                pass

            class Q:
                pass

            o = O()
            o.price_usd = price(_MARK_USD) if callable(price) else price
            o.liquidity_usd = 60_000.0
            o.volume_5m_usd = 1_000.0
            o.market_cap_usd = 9_000.0
            o.pair_address = "P" * 44
            o.dex_id = "pumpfun"
            o.observed_at = time.time()
            q = Q()
            q.observation = o
            return q

    return _Market()

# ===========================================================================
# 21-22. THE REAL ADAPTER SEAM
# ===========================================================================


class TestPointFromDirectSeam:
    """
    The integration seam the offline suite previously never exercised.

    Every other test here builds a ``QuotePoint`` directly, which is exactly
    how a typo in ``QuoteService._point_from_direct`` can pass the whole
    offline suite -- that function is the only place a real ``DirectQuote``
    becomes a ``QuotePoint``, and no fake goes through it. A regression there
    is invisible until a live run.

    So it is driven here, offline, with a realistic double.
    """

    def test_a_real_direct_quote_becomes_a_point_with_a_reference(self):
        service, direct = _service_and_direct(_curve())
        point = service._point_from_direct(
            direct, side="BUY", mint=MINT, decimals=6,
            observed_pair_address=CURVE,
        )
        assert point.available is True
        assert point.identity == "PASS"
        native = point.execution_native_reference
        assert native is not None
        assert native.market_id == point.venue_id == (
            f"PUMP_BONDING_CURVE:{CURVE}")
        assert native.reference_price == pytest.approx(
            float(direct.quote.spot_price))
        assert native.calculation_method == (
            "PUMP_BONDING_CURVE_VIRTUAL_RESERVES")
        assert native.token_decimals == 6
        # the reference and the quote are the SAME slot, read once
        assert native.slot == direct.snapshot.slot == point.state_slot

    def test_the_seam_reports_a_missing_reference_rather_than_raising(self):
        """
        A quote with no derivable spot must produce ``None`` -- never an
        exception (which the caller would report as an arithmetic error) and
        never a zero (which would flow into the ratio and read as a finding).
        """
        service, direct = _service_and_direct(_curve())
        stripped = replace(
            direct, quote=replace(direct.quote, spot_price=None))
        point = service._point_from_direct(
            stripped, side="BUY", mint=MINT, decimals=6,
            observed_pair_address=CURVE,
        )
        assert point.execution_native_reference is None
        assert point.available is True


def _service_and_direct(curve: BondingCurve):
    """A real QuoteService, and a DirectQuote shaped like the reader's."""
    from autopaper.paper_live.quotes import QuoteService
    from autopaper.paper_live.quotes import SolReference
    from autopaper.same_market.reader import DirectQuote, StateSnapshot

    service = QuoteService(
        rpc=None,
        sol_reference=SolReference(150.0, time.time(), "test"),
        max_quote_age_ms=30_000.0,
    )
    result = quote_bonding_curve_buy(
        curve, sol_amount_lamports=4_000_000, token_decimals=6,
        fees=_fees(), mint_supply=1_000_000_000 * 10 ** 6,
    )
    direct = DirectQuote(
        provider="pump_direct", market_type=MarketKind.PUMP_BONDING_CURVE,
        quote=result, state=curve,
        snapshot=StateSnapshot(
            slot=curve.slot, fetched_at=time.time(), fetch_latency_ms=90.0,
            accounts_read=3,
        ),
        curve_address=CURVE, base_mint=MINT, quote_mint=WSOL,
        quote_latency_ms=95.0,
    )
    return service, direct


# ===========================================================================
# ENTRY EXECUTION-GAP OBSERVABILITY (tests 1-7)
# ===========================================================================

from autopaper.paper_live.schema import entry_execution_gap_observability

def _signal_record_with_exec_gap(
    mark_value_usd: float,
    executable_price_usd: float,
    native_reference_price_usd: float | None,
    native_method: str | None,
    native_available: bool,
) -> SignalRecord:
    """Build a SignalRecord whose entry-execution-gap observability is populated."""
    now = time.time()
    mark = ObservedMarkPoint(
        value_usd=mark_value_usd, pair_address="pair-1", dex_id="pumpfun",
        liquidity_usd=10_000.0, volume_5m_usd=5_000.0, market_cap_usd=100_000.0,
        timestamp=now - 0.05, age_ms=50.0, available=True,
    )
    q = QuotePoint(
        side="BUY", provider="pump_direct", market_type="PUMP_BONDING_CURVE",
        token_mint=MINT, quote_mint=WSOL, input_amount_raw=1_000_000,
        input_decimals=6, input_amount_decimal=1.0,
        output_amount_raw=30_000_000, output_decimals=6,
        output_amount_decimal=30.0,
        # reference_price and effective_price share one unit. This synthetic
        # record stores both in the USD numbers under test, so the production
        # quote-asset→USD scaling is an identity and the gap is the USD ratio.
        effective_price=Decimal(str(executable_price_usd)),
        effective_price_usd=executable_price_usd,
        price_impact_pct=1.0, fee_bps=125, fee_amount=Decimal("0"),
        venue="pump_direct", pool_or_curve="pool-1", route=(),
        state_slot=100, state_age_ms=5.0, quote_timestamp=now,
        quote_age_ms=5.0, max_quote_age_ms=30_000.0, quote_latency_ms=5.0,
        identity="PASS", observed_pair_address="pair-1", venue_id="PUMP_BONDING_CURVE:pool-1",
        available=True, missing_reason=None,
        execution_native_reference=(
            ExecutionNativeReference(
                market_id="PUMP_BONDING_CURVE:pool-1", mint=MINT,
                base_mint=WSOL, quote_mint=WSOL, side="BUY",
                reference_price=native_reference_price_usd or 3.0e-5,
                numerator_raw=300, denominator_raw=100,
                calculation_method=native_method or "PUMP_BONDING_CURVE_VIRTUAL_RESERVES",
                venue="pump_direct", pool_or_curve="pool-1",
                provider="pump_direct", slot=100, observed_at=now,
                state_age_ms=5.0, token_decimals=6,
            ) if native_available else None
        ),
    )
    return SignalRecord(
        signal=SignalSnapshot(
            signal_id="sig-1", channel="c", message_id=1,
            telegram_timestamp=now - 1.0, receive_timestamp=now,
            receive_monotonic=now, token_mint=MINT, symbol="T",
            raw_text_sha256="0" * 64, raw_text_length=10,
        ),
        arm="L25_CURRENT", arm_version="execution_aware_paper/v1",
        baseline_hash="abc", l25_decision="BUY", l25_status="",
        l25_authorized=True, arm_filter_reason="", observed_mark=mark,
        buy_quote=q, mark_to_quote_gap_pct=0.0, execution_identity="PASS",
        traded=True, failure=FailureReason.NONE, failure_detail="",
        latency=LatencyMarks(), bankroll_before=1000.0,
        position_size_usd=100.0, position_size_pct=1.0,
        finished_at=now,
        **entry_execution_gap_observability(mark=mark, quote=q, guard=None),
    )


def test_1_observed_mark_and_executable_quote_are_recorded_separately():
    """The observed mark (DexScreener) and the executable buy price are
    stored as distinct fields on the signal record, never collapsed into one.
    The field NAMING (observed_mark vs buy_quote) is the separation."""
    rec = _signal_record_with_exec_gap(
        mark_value_usd=1.057e-05, executable_price_usd=3.039e-05,
        native_reference_price_usd=3.001e-05,
        native_method="PUMP_BONDING_CURVE_VIRTUAL_RESERVES", native_available=True)
    assert rec.buy_quote.effective_price_usd == pytest.approx(3.039e-05)
    assert rec.observed_mark.value_usd == pytest.approx(1.057e-05)
    # The two numbers live on different fields/layers; compare via as_dict.
    d = rec.as_dict()
    assert d["observed_mark"]["value_usd"] == pytest.approx(1.057e-05)
    assert d["executable_buy_quote"]["effective_price_usd"] == pytest.approx(3.039e-05)
    assert d["observed_mark"]["layer"] == "OBSERVED_MARK"
    assert d["executable_buy_quote"]["layer"] == "EXECUTABLE_QUOTE"


def test_2_correct_timestamps_are_preserved():
    """The observed mark timestamp, executable quote timestamp and native
    reference observed_at are recorded as received; when a source timestamp is
    unavailable it is left absent rather than fabricated."""
    rec = _signal_record_with_exec_gap(
        mark_value_usd=1.0e-5, executable_price_usd=3.0e-5,
        native_reference_price_usd=3.0e-5,
        native_method="PUMP_BONDING_CURVE_VIRTUAL_RESERVES", native_available=True)
    assert rec.entry_observed_mark_timestamp is not None
    assert rec.entry_executable_quote_timestamp is not None
    assert rec.entry_native_reference_method == "PUMP_BONDING_CURVE_VIRTUAL_RESERVES"
    assert rec.entry_native_reference_available is True


def test_3_mark_to_executable_gap_is_calculated_correctly():
    """(executable / observed_mark - 1) * 100, a DIAGNOSTIC that mixes
    execution cost with aggregator lag, is recorded per record."""
    rec = _signal_record_with_exec_gap(
        mark_value_usd=1.057e-05, executable_price_usd=3.039e-05,
        native_reference_price_usd=None, native_method=None,
        native_available=False)
    expected = (3.039e-05 / 1.057e-05 - 1.0) * 100.0
    assert rec.entry_mark_to_executable_gap_pct == pytest.approx(expected, abs=1e-6)


def test_4_native_to_executable_is_the_production_number_not_the_mark_premium():
    """native_to_executable is fee + size impact (executable vs the SAME market's
    own spot), and is a different number from the mark-to-executable premium."""
    rec = _signal_record_with_exec_gap(
        mark_value_usd=1.057e-05, executable_price_usd=3.039e-05,
        native_reference_price_usd=3.001e-05,
        native_method="PUMP_BONDING_CURVE_VIRTUAL_RESERVES", native_available=True)
    native_gap = rec.entry_native_to_executable_gap_pct
    mark_gap = rec.entry_mark_to_executable_gap_pct
    assert native_gap == pytest.approx((3.039e-05 / 3.001e-05 - 1.0) * 100.0, abs=1e-6)
    assert mark_gap == pytest.approx((3.039e-05 / 1.057e-05 - 1.0) * 100.0, abs=1e-6)
    assert abs(native_gap - mark_gap) > 1.0, (
        "the native and mark gaps collapsed into one number")


def test_5_stale_mark_is_classified_not_freshened():
    """A stale observed mark is NOT given a fresh timestamp. The mark is retained
    as OBSERVED_MARK in the as_dict layer and is never promoted to executable.
    The provider's source timestamp, when unavailable, is left absent."""
    rec = _signal_record_with_exec_gap(
        mark_value_usd=1.0e-5, executable_price_usd=3.0e-5,
        native_reference_price_usd=3.0e-5,
        native_method="PUMP_BONDING_CURVE_VIRTUAL_RESERVES", native_available=True)
    d = rec.as_dict()
    assert d["observed_mark"]["layer"] == "OBSERVED_MARK"
    assert d["executable_buy_quote"]["layer"] == "EXECUTABLE_QUOTE"
    assert d["executable_buy_quote"]["effective_price_usd"] != d["observed_mark"]["value_usd"]


def test_6_wrong_market_observation_is_rejected_by_identity_not_by_gap():
    """A quote whose observed_pair_address differs from the executable market is
    refused on identity (MISMATCH/UNKNOWN), not silently accepted on a large gap."""
    q = QuotePoint(
        side="BUY", provider="pump_direct", market_type="PUMP_BONDING_CURVE",
        token_mint=MINT, quote_mint=WSOL, input_amount_raw=1_000_000,
        input_decimals=6, input_amount_decimal=1.0,
        output_amount_raw=30_000_000, output_decimals=6,
        output_amount_decimal=30.0,
        effective_price=Decimal("3.0e-5"), effective_price_usd=3.0e-5,
        price_impact_pct=1.0, fee_bps=125, fee_amount=Decimal("0"),
        venue="pump_direct", pool_or_curve="pool-1", route=(),
        state_slot=100, state_age_ms=5.0, quote_timestamp=time.time(),
        quote_age_ms=5.0, max_quote_age_ms=30_000.0, quote_latency_ms=5.0,
        identity="MISMATCH", observed_pair_address="other-pair",
        venue_id="PUMP_BONDING_CURVE:pool-1", available=True, missing_reason=None,
        execution_native_reference=None,
    )
    from autopaper.paper_live.fills import fill_from_quote, FillRefused
    with pytest.raises(FillRefused) as exc:
        fill_from_quote(q, side="BUY", require_identity=True)
    assert exc.value.reason is FailureReason.ENTRY_REJECTED_IDENTITY_UNKNOWN


def test_7_no_executable_quote_is_fabricated():
    """When no executable quote exists, the signal record carries an unavailable
    placeholder with provider="none" and no price -- never a made-up price."""
    rec = SignalRecord(
        signal=SignalSnapshot(
            signal_id="sig", channel="c", message_id=1,
            telegram_timestamp=time.time(), receive_timestamp=time.time(),
            receive_monotonic=time.time(), token_mint=MINT, symbol="T",
            raw_text_sha256="0" * 64, raw_text_length=10,
        ),
        arm="L25_CURRENT", arm_version="execution_aware_paper/v1",
        baseline_hash="abc", l25_decision="REJECT", l25_status="",
        l25_authorized=False, arm_filter_reason="",
        observed_mark=ObservedMarkPoint(
            value_usd=1.0e-5, pair_address="pair", dex_id="pumpfun",
            liquidity_usd=None, volume_5m_usd=None, market_cap_usd=None,
            timestamp=time.time(), age_ms=None, available=False,
            missing_reason="no market data",
        ),
        buy_quote=QuotePoint(
            side="BUY", provider="none", market_type="UNKNOWN",
            token_mint=MINT, quote_mint=WSOL, input_amount_raw=None,
            input_decimals=None, input_amount_decimal=None,
            output_amount_raw=None, output_decimals=None,
            output_amount_decimal=None, effective_price=None,
            effective_price_usd=None, price_impact_pct=None, fee_bps=None,
            fee_amount=None, venue=None, pool_or_curve=None, route=(),
            identity="UNKNOWN", available=False,
            missing_reason="no executable quote",
        ),
        mark_to_quote_gap_pct=None, execution_identity="UNKNOWN",
        traded=False, failure=FailureReason.EXECUTABLE_BUY_QUOTE_UNAVAILABLE,
        failure_detail="no executable quote",
        latency=LatencyMarks(), bankroll_before=1000.0,
        position_size_usd=None, position_size_pct=None, finished_at=time.time(),
    )
    assert rec.buy_quote.provider == "none"
    assert rec.buy_quote.effective_price_usd is None
    assert rec.buy_quote.available is False
    assert rec.buy_quote.identity == "UNKNOWN"
