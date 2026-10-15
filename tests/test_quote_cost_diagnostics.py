"""
WHAT +1.26% MEANS, PINNED AS TESTS.

The execution-quote audit established that the number this system used to
print as ``Gap: +1.26%`` is the NATIVE execution gap -- executable price over
THIS market's same-slot marginal price -- not the observed-mark premium. On a
Pump.fun bonding-curve BUY it decomposes into the 125 bps trading fee plus a
sub-bps curve-size impact. The observed-mark premium is a different quantity
and an order of magnitude larger (Oushi: +14.63% against a +1.26% native gap).

Every quote here is produced by the VENUE'S OWN arithmetic
(``pumpdirect.quote_bonding_curve_buy`` / ``quote_bonding_curve_sell``): the
real 125 bps schedule, the real integer floor on a BUY input, the real
constant-product curve. The diagnostics are then checked against integers, so
a passing test means the decomposition is right, not that a constant matches
itself. The two exact-reproduction tests carry the historical Oushi and SNORP
integer legs verbatim from the execution-aware PAPER record.
"""

from __future__ import annotations

import sys
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from autopaper.same_market.models import WSOL_MINT
from autopaper.same_market.pumpdirect import (
    DEFAULT_PUBKEY,
    BondingCurve,
    Fees,
    quote_bonding_curve_buy,
    quote_bonding_curve_sell,
)

import autopaper.paper_live  # noqa: F401 - installs the structural barrier
from autopaper.paper_live.fills import FillModel, fill_from_quote
from autopaper.paper_live.quote_diagnostics import (
    QuoteCostBreakdown,
    breakdown_of,
    cost_block,
)
from autopaper.paper_live.schema import ExecutionNativeReference, QuotePoint

pytestmark = [
    pytest.mark.paper_live,
    pytest.mark.same_market,
    pytest.mark.execution,
]

# ---------------------------------------------------------------------------
# the venue, verbatim
# ---------------------------------------------------------------------------

FEE_BPS = 125
FEES = Fees(lp_fee_bps=0, protocol_fee_bps=FEE_BPS, creator_fee_bps=0)
MINT = "CHx22R8snRUhyxFbnNToPsdbMu1LpNWkzxai911Sr9mw"
CURVE_ID = "Fh3tSSXU6eFmE8rVJMTzysXK4WoF1QzBRTXuPaZJ3MKP"
SOL_USD = 122.0
TOKEN_DECIMALS = 6
VIRTUAL_QUOTE = 30_000_000_000      # 30 SOL virtual reserve
VIRTUAL_TOKEN = 1_500_000_000_000

#: (1 + fee) * (1 + impact) == (1 + total) holds MATHEMATICALLY; the stored
#: values are computed in the default Decimal context (28 significant
#: digits), so the recomposition may differ in the last ulps only. Anything
#: above this is a decomposition error, not rounding.
EXACT = Decimal("1e-22")


def _curve() -> BondingCurve:
    return BondingCurve(
        address=CURVE_ID, mint=MINT,
        virtual_token_reserves=VIRTUAL_TOKEN,
        virtual_quote_reserves=VIRTUAL_QUOTE,
        real_token_reserves=800_000_000_000,
        real_quote_reserves=20_000_000_000,
        token_total_supply=1_000_000_000 * 10 ** TOKEN_DECIMALS,
        complete=False, creator=DEFAULT_PUBKEY, is_mayhem_mode=False,
        is_cashback_coin=False, quote_mint=WSOL_MINT, creator_fee_bps=None,
        can_edit_creator_fee=None, account_len=150, slot=1, timestamp=0.0,
    )


def _native(curve: BondingCurve, side: str) -> ExecutionNativeReference:
    """The same-slot marginal price, from the SAME reserve integers."""
    price = Decimal(curve.virtual_quote_reserves).scaleb(
        TOKEN_DECIMALS - 9) / Decimal(curve.virtual_token_reserves)
    return ExecutionNativeReference(
        market_id=f"PUMP_BONDING_CURVE:{curve.address}", mint=MINT,
        base_mint=MINT, quote_mint=WSOL_MINT, side=side,
        reference_price=float(price),
        numerator_raw=curve.virtual_quote_reserves,
        denominator_raw=curve.virtual_token_reserves,
        calculation_method="PUMP_BONDING_CURVE_VIRTUAL_RESERVES",
        venue="pumpfun-bonding-curve", pool_or_curve=curve.address,
        provider="pump_direct", slot=1, observed_at=0.0, state_age_ms=1.0,
        token_decimals=TOKEN_DECIMALS,
    )


def _wrap(result, *, side: str, curve: BondingCurve) -> QuotePoint:
    """The venue's DirectQuoteResult dressed as the production QuotePoint."""
    if side == "BUY":
        in_dec, out_dec = 9, TOKEN_DECIMALS
    else:
        in_dec, out_dec = TOKEN_DECIMALS, 9
    return QuotePoint(
        side=side, provider="pump_direct",
        market_type="PUMP_BONDING_CURVE", token_mint=MINT, quote_mint=WSOL_MINT,
        input_amount_raw=result.input_amount_raw, input_decimals=in_dec,
        input_amount_decimal=Decimal(result.input_amount_raw).scaleb(-in_dec),
        output_amount_raw=result.output_amount_raw, output_decimals=out_dec,
        output_amount_decimal=Decimal(result.output_amount_raw).scaleb(-out_dec),
        effective_price=result.effective_price,
        effective_price_usd=float(result.effective_price) * SOL_USD,
        price_impact_pct=(float(result.price_impact_pct)
                          if result.price_impact_pct is not None else None),
        fee_bps=result.fee_bps, fee_amount=Decimal(0),
        venue="pumpfun-bonding-curve", pool_or_curve=curve.address,
        route=("pump_direct",), quote_timestamp=1_791_275_344.0,
        quote_age_ms=0.0, max_quote_age_ms=30_000.0,
        identity="PASS", identity_evidence="test fixture",
        observed_pair_address=curve.address, available=True,
        execution_native_reference=_native(curve, side),
        amount_after_fee_raw=result.amount_after_fee,
    )


def _buy(lamports: int) -> QuotePoint:
    curve = _curve()
    return _wrap(
        quote_bonding_curve_buy(
            curve, sol_amount_lamports=lamports,
            token_decimals=TOKEN_DECIMALS, fees=FEES,
            mint_supply=curve.token_total_supply,
        ),
        side="BUY", curve=curve,
    )


def _sell(tokens_raw: int) -> QuotePoint:
    curve = _curve()
    return _wrap(
        quote_bonding_curve_sell(
            curve, token_amount_raw=tokens_raw,
            token_decimals=TOKEN_DECIMALS, fees=FEES,
        ),
        side="SELL", curve=curve,
    )


def _fee_is_captured_once(b: QuoteCostBreakdown, *, side: str) -> None:
    """The exact composition ``total = fee (+/-) impact``, by side."""
    one = Decimal(1)
    fee = b.fee_cost_bps / Decimal(10_000)
    impact = b.impact_bps / Decimal(10_000)
    total = b.total_cost_bps / Decimal(10_000)
    factor = (one + fee) if side == "BUY" else (one - fee)
    assert abs(factor * (one + impact) - one - total) < EXACT


# ===========================================================================
# 1-3. curve BUYs, small / medium / large
# ===========================================================================


class TestCurveBuys:
    def test_a_small_curve_buy_splits_into_fee_and_tiny_impact(self):
        q = _buy(1_000_000)          # 0.001 SOL
        b = breakdown_of(q)

        # fee: the integer floor on the input means the extracted rate sits
        # just ABOVE the 125 bps schedule, never below it.
        expected_fee_raw = q.input_amount_raw - q.amount_after_fee_raw
        assert b.fee_amount_quote == Decimal(expected_fee_raw).scaleb(-9)
        assert Decimal(FEE_BPS) <= b.fee_cost_bps < Decimal(FEE_BPS + 1)

        # pure size impact: a 0.001 SOL order on a 30 SOL curve is sub-bps
        assert b.impact_bps is not None and Decimal(0) < b.impact_bps < Decimal(1)

        # total native cost: fee dominates, ~125.3 bps
        assert b.total_cost_bps is not None
        assert Decimal(FEE_BPS) < b.total_cost_bps < Decimal(FEE_BPS + 1)
        _fee_is_captured_once(b, side="BUY")

        # the executable price IS the quote's own effective price
        assert abs(b.effective_price - q.effective_price) <= (
            q.effective_price * Decimal("1e-20"))
        assert b.fee_bps == FEE_BPS
        # quote age rides along when the quote carries a timestamp
        assert b.quote_age_ms == 0.0

    def test_a_medium_curve_buy_keeps_the_same_fee_but_more_impact(self):
        q = _buy(50_000_000)         # 0.05 SOL
        b = breakdown_of(q)
        small = breakdown_of(_buy(1_000_000))

        assert Decimal(FEE_BPS) <= b.fee_cost_bps < Decimal(FEE_BPS + 1)
        assert b.impact_bps > small.impact_bps          # size shows up here
        assert b.total_cost_bps > small.total_cost_bps
        _fee_is_captured_once(b, side="BUY")

    def test_a_large_curve_buy_impact_is_dominated_by_size(self):
        q = _buy(5_000_000_000)      # 5 SOL against a 30 SOL curve
        b = breakdown_of(q)
        medium = breakdown_of(_buy(50_000_000))

        # the fee rate does NOT scale with size; the impact does
        assert Decimal(FEE_BPS) <= b.fee_cost_bps < Decimal(FEE_BPS + 1)
        assert b.impact_bps > medium.impact_bps > Decimal(10)
        assert b.impact_bps > Decimal(1_000)          # >16% of notional
        assert b.total_cost_bps > Decimal(1_000)
        _fee_is_captured_once(b, side="BUY")

# ===========================================================================
# 4. SELL
# ===========================================================================


class TestSell:
    def test_a_sell_charges_the_fee_out_of_the_proceeds(self):
        q = _sell(50_000_000_000)
        b = breakdown_of(q)

        aaf, out = q.amount_after_fee_raw, q.output_amount_raw
        # fee direction: the pre-fee proceeds EXCEED what is received
        assert aaf > out > 0
        # output/fee relationship: output == amount_after_fee - fee, exactly
        assert b.fee_amount_quote == (Decimal(aaf) - Decimal(out)).scaleb(-9)
        assert Decimal(out) == Decimal(aaf) - int(b.fee_amount_quote.scaleb(9))
        assert Decimal(FEE_BPS) <= b.fee_cost_bps < Decimal(FEE_BPS + 1)

    def test_a_sell_executes_at_the_received_amount_not_the_gross(self):
        """The executable price for a SELL is what is RECEIVED per token."""
        q = _sell(50_000_000_000)
        b = breakdown_of(q)

        # QuotePoint's own contract: breakdown reproduces quote.effective_price
        assert abs(b.effective_price - q.effective_price) <= (
            q.effective_price * Decimal("1e-20"))
        # ...and equals output / tokens from the integers alone
        assert abs(b.effective_price - (
            Decimal(q.output_amount_raw).scaleb(-9)
            / Decimal(q.input_amount_raw).scaleb(-TOKEN_DECIMALS)
        )) < Decimal("1e-20")

        # total native gap for a SELL is NEGATIVE (fee + push-down impact),
        # and it matches the venue's own price_impact_pct -- which is the
        # fee+impact gap, whatever its misleading name.
        assert b.total_cost_bps < 0
        assert b.impact_bps < 0 and abs(b.impact_bps) < abs(b.total_cost_bps)
        assert abs(
            b.total_cost_bps / Decimal(100)
            - Decimal(str(q.price_impact_pct))
        ) < Decimal("1e-9")
        _fee_is_captured_once(b, side="SELL")


# ===========================================================================
# 5-6. no double counting
# ===========================================================================


class TestNoDoubleCounting:
    def test_the_fee_is_charged_once_and_not_again_at_fill_time(self):
        q = _buy(50_000_000)
        b = breakdown_of(q)

        # the diagnostics extract the fee from the integers -- one subtraction
        assert b.fee_amount_quote == Decimal(
            q.input_amount_raw - q.amount_after_fee_raw).scaleb(-9)

        # the simulated fill takes the quote's price verbatim: no penalty
        # (FillModel.post_quote_penalty_bps == 0) and fill.fee stays zero
        # because the venue's fee is already inside the quoted output.
        model = FillModel()
        assert model.post_quote_penalty_bps == 0.0
        fill = fill_from_quote(q, side="BUY")
        assert fill.fee == Decimal(0)
        assert fill.effective_price == q.effective_price

        # the fee appears ONCE inside the total: total is ~fee+impact, so a
        # doubled fee would put it above 2x the fee rate.
        assert b.total_cost_bps < b.fee_cost_bps * 2

    def test_the_impact_is_counted_once_never_added_on_top(self):
        q = _buy(50_000_000)
        b = breakdown_of(q)

        # the provider's price_impact_pct (fee+impact) equals the composed
        # total -- it is the SAME quantity, not an extra term to be added.
        assert abs(
            b.total_cost_bps / Decimal(100) - Decimal(str(q.price_impact_pct))
        ) < Decimal("1e-9")
        # a double count would look like fee + provider total instead:
        double = b.fee_cost_bps + Decimal(str(q.price_impact_pct)) * 100
        assert abs(b.total_cost_bps - double) > Decimal(10)

        # and the fill is not moved off the quote by an applied impact
        fill = fill_from_quote(q, side="BUY")
        assert fill.effective_price == q.effective_price
        _fee_is_captured_once(b, side="BUY")


# ===========================================================================
# 7. mark premium != native gap
# ===========================================================================


class TestMarkPremiumIsItsOwnQuantity:
    def test_the_mark_premium_is_not_the_native_execution_gap(self):
        q = _buy(1_000_000)
        # an observed mark 14.6% BELOW the executable price, while the native
        # gap stays at its ~1.25% fee-dominated value.
        mark = float(q.effective_price_usd) / 1.146
        b = breakdown_of(q, observed_mark_usd=mark)

        assert abs(b.mark_premium_pct - Decimal("14.6")) < Decimal("0.05")
        native_pct = b.total_cost_bps / Decimal(100)
        assert abs(native_pct - Decimal("1.25")) < Decimal("0.05")
        assert b.mark_premium_pct - native_pct > Decimal(10)
        # premium is computed against the OBSERVED MARK, gap against the curve
        assert b.mark_premium_pct != native_pct


# ===========================================================================
# 8. rounding
# ===========================================================================


class TestRounding:
    def test_every_derived_number_is_decimal_and_exact_on_the_integer_legs(self):
        q = _buy(1_000_000)
        b = breakdown_of(q)

        for value in (b.fee_cost_bps, b.impact_bps, b.total_cost_bps,
                      b.fee_amount_quote, b.effective_price):
            assert isinstance(value, Decimal)

        # raw integers reconstruct the fee exactly: no float in the path
        fee_raw = q.input_amount_raw - q.amount_after_fee_raw
        assert b.fee_amount_quote == Decimal(fee_raw).scaleb(-9)
        assert b.fee_amount_quote.scaleb(9) == b.fee_amount_quote.scaleb(
            9).to_integral_value()

        # the extracted rate can only round UP off the integer floor
        assert b.fee_cost_bps >= Decimal(FEE_BPS)
        assert b.fee_cost_bps < Decimal(FEE_BPS) + Decimal("1")

        # the total reproduces from the reserve integers at high precision
        native = q.execution_native_reference
        with localcontext() as ctx:
            ctx.prec = 50
            marginal = (Decimal(native.numerator_raw)
                        .scaleb(TOKEN_DECIMALS - 9)
                        / Decimal(native.denominator_raw))
            effective = (Decimal(q.input_amount_raw).scaleb(-9)
                         / Decimal(q.output_amount_raw).scaleb(-TOKEN_DECIMALS))
            expected = (effective / marginal - 1) * Decimal(10_000)
        assert abs(expected - b.total_cost_bps) < Decimal("1e-10")


# ===========================================================================
# 9-10. exact reproductions, from the historical integer legs
# ===========================================================================

# var/paper_state.json, execution-aware PAPER records, verbatim.
OUSHI = dict(
    gross=4_147_411, tokens=27_590_255_347,
    numerator=69_129_069_414, denominator=465_650_708_688_841,
    effective_usd=1.812233782142025e-05, mark_usd=1.581e-05,
    native_gap=Decimal("1.2560340041664161"),
    mark_premium=Decimal("14.6258"),
)
SNORP = dict(
    gross=3_870_321, tokens=28_465_176_796,
    numerator=65_745_620_443, denominator=489_614_365_823_035,
    effective_usd=1.6391774369888714e-05, mark_usd=1.415e-05,
    native_gap=Decimal("1.2559201673269531"),
    mark_premium=Decimal("15.8429"),
)


def _historical_buy(legs: dict) -> tuple[QuotePoint, QuoteCostBreakdown]:
    """Rebuild the recorded Oushi/SNORP BUY from its raw integer legs."""
    aaf = ((legs["gross"] - 1) * 10_000) // (10_000 + FEE_BPS)
    native = ExecutionNativeReference(
        market_id="PUMP_BONDING_CURVE:historical", mint=MINT, base_mint=MINT,
        quote_mint=WSOL_MINT, side="BUY", reference_price=0.0,
        numerator_raw=legs["numerator"], denominator_raw=legs["denominator"],
        calculation_method="PUMP_BONDING_CURVE_VIRTUAL_RESERVES",
        venue="pumpfun-bonding-curve", pool_or_curve="historical",
        provider="pump_direct", slot=453_847_829, observed_at=0.0,
        token_decimals=TOKEN_DECIMALS,
    )
    q = QuotePoint(
        side="BUY", provider="pump_direct",
        market_type="PUMP_BONDING_CURVE", token_mint=MINT, quote_mint=WSOL_MINT,
        input_amount_raw=legs["gross"], input_decimals=9,
        input_amount_decimal=Decimal(legs["gross"]).scaleb(-9),
        output_amount_raw=legs["tokens"], output_decimals=TOKEN_DECIMALS,
        output_amount_decimal=Decimal(legs["tokens"]).scaleb(-TOKEN_DECIMALS),
        effective_price=(Decimal(legs["gross"]).scaleb(-9)
                         / Decimal(legs["tokens"]).scaleb(-TOKEN_DECIMALS)),
        effective_price_usd=legs["effective_usd"],
        price_impact_pct=float(legs["native_gap"]),
        fee_bps=FEE_BPS, fee_amount=Decimal(0),
        venue="pumpfun-bonding-curve", pool_or_curve="historical",
        route=("pump_direct",), quote_age_ms=0.06,
        identity="PASS", available=True,
        execution_native_reference=native, amount_after_fee_raw=aaf,
    )
    return q, breakdown_of(q, observed_mark_usd=legs["mark_usd"])


class TestExactReproduction:
    def test_oushi_native_gap_reproduces_to_the_last_digit(self):
        q, b = _historical_buy(OUSHI)
        assert q.amount_after_fee_raw == 4_096_207
        assert abs(b.total_cost_bps / Decimal(100) - OUSHI["native_gap"]) < (
            Decimal("1e-12"))
        # ...and it is 125 bps of fee plus sub-bps of curve impact
        assert Decimal("125.00") <= b.fee_cost_bps <= Decimal("125.01")
        assert Decimal("0.5") < b.impact_bps < Decimal("0.7")
        _fee_is_captured_once(b, side="BUY")

    def test_snorp_native_gap_reproduces_to_the_last_digit(self):
        q, b = _historical_buy(SNORP)
        assert q.amount_after_fee_raw == 3_822_538
        assert abs(b.total_cost_bps / Decimal(100) - SNORP["native_gap"]) < (
            Decimal("1e-12"))
        assert Decimal("125.00") <= b.fee_cost_bps <= Decimal("125.01")
        assert Decimal("0.5") < b.impact_bps < Decimal("0.7")
        _fee_is_captured_once(b, side="BUY")


# ===========================================================================
# 11-12. regressions
# ===========================================================================


class TestRegressions:
    def test_total_cost_is_not_a_constant_1_26_percent(self):
        """Native total-cost bps VARY with trade size. It is not a constant."""
        sizes = (1_000_000, 50_000_000, 5_000_000_000)
        totals = [breakdown_of(_buy(s)).total_cost_bps for s in sizes]
        assert all(t is not None for t in totals)
        assert totals[0] < totals[1] < totals[2]
        # spread is far beyond rounding: sub-bps impact to >16% impact
        assert totals[2] - totals[0] > Decimal(1_000)
        # and not every quote lands on the recorded Oushi value
        oushi_bps = OUSHI["native_gap"] * 100
        assert any(abs(t - oushi_bps) > Decimal("0.01") for t in totals)

    def test_oushi_mark_premium_is_about_plus_14_63_not_plus_1_26(self):
        _, b = _historical_buy(OUSHI)
        assert abs(b.mark_premium_pct - OUSHI["mark_premium"]) < (
            Decimal("0.001"))
        assert b.mark_premium_pct > Decimal(10)
        # it must never be reported as the ~1.26% native gap
        assert abs(b.mark_premium_pct - OUSHI["native_gap"]) > Decimal(10)
        native_pct = b.total_cost_bps / Decimal(100)
        assert abs(native_pct - OUSHI["native_gap"]) < Decimal("1e-12")


# ===========================================================================
# 13. the display contract
# ===========================================================================

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from tests import paper_bridge_fakes as _fk
except ImportError:  # pragma: no cover - plain-path pytest invocation
    import paper_bridge_fakes as _fk


class TestDisplayLabels:
    def test_the_cost_block_names_each_quantity_separately(self):
        """One block, every quantity its own line -- never one 'Gap:'."""
        q = _buy(50_000_000)
        mark = float(q.effective_price_usd) / 1.146
        lines = cost_block(breakdown_of(q, observed_mark_usd=mark))
        text = "\n".join(lines)

        assert "Gap:" not in text
        for label in ("Curve mid/marginal", "SOL input", "Trading fee",
                      "Fee amount", "Price impact", "Total quote cost",
                      "Mark → executable premium"):
            assert label in text

    def test_the_entry_alert_keeps_simulated_fill_and_observed_mark(self):
        b = _fk.bridge()
        _fk.run(b.handle_alert(snapshot=_fk.alert(),
                               report=_fk.real_report()))
        entries = [m for m in b.drain_messages() if m["kind"] == "entry"]
        assert entries
        text = entries[0]["text"]

        # labels the existing bridge tests require
        assert "Simulated fill" in text
        assert "Observed mark" in text
        # the ambiguous "Gap:" line is gone, replaced by named quantities
        assert "Gap:" not in text
        assert "Total quote cost" in text
        assert "Mark → executable premium" in text
        assert "Trading fee" in text
        assert "Price impact" in text
        assert "Quote age" in text
        assert "Venue" in text

