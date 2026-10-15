"""
Numeric core: decimal normalisation, executable price, mark/executable gap.

The recurring theme: a measurement harness is only as honest as its arithmetic.
Every case here is a way a price comparison can go quietly wrong - a rounding
step before the division, a float losing the low digits of a nine-decimal
amount, a division by zero becoming an infinity that renders as "inf".
"""


from __future__ import annotations

from decimal import Decimal, localcontext

import pytest

from autopaper.same_market.models import SOL_DECIMALS
from autopaper.same_market.numeric import (
    PRECISION,
    DecimalError,
    apply_fee_bps,
    constant_product_out,
    effective_price,
    normalise_amount,
    pct_return,
    price_gap_pct,
    quantise,
    sol_amount_for_usd,
    to_decimal,
    to_float,
)

pytestmark = pytest.mark.same_market

WSOL = "So11111111111111111111111111111111111111112"
TOKEN_DECIMALS = 6


# ---------------------------------------------------------------------------
# 15. decimal normalisation
# ---------------------------------------------------------------------------

def test_raw_amounts_normalise_by_decimals():
    assert normalise_amount(1_928_811, 9) == Decimal("0.001928811")
    assert normalise_amount(7_547_061_883, 6) == Decimal("7547.061883")
    assert normalise_amount(1, 0) == Decimal(1)


def test_normalisation_does_not_go_through_float():
    # 9 decimals on a value large enough that a float would visibly drift.
    raw = 123_456_789_123_456_789
    assert normalise_amount(raw, 9) == to_decimal(raw).scaleb(-9)
    # Exact decimal equality is the assertion; "close enough" is not the bar.
    assert str(normalise_amount(raw, 9)) == "123456789.123456789"


def test_implausible_decimals_are_rejected_not_guessed():
    with pytest.raises(DecimalError):
        normalise_amount(1000, -1)
    with pytest.raises(DecimalError):
        normalise_amount(1000, 99)


def test_non_numeric_input_raises_rather_than_becoming_zero():
    for bad in (None, "", "abc", object(), True):
        with pytest.raises(DecimalError):
            to_decimal(bad)


# ---------------------------------------------------------------------------
# 16. observed mark parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1.783e-05", Decimal("0.00001783")),
        ("0.00001783", Decimal("0.00001783")),
        ("3.059E-5", Decimal("0.00003059")),
        (1.783e-05, Decimal("0.00001783")),
        (0, Decimal(0)),
    ],
)
def test_an_observed_mark_survives_parsing_unchanged(raw, expected):
    assert to_decimal(raw, field="priceUsd") == expected


def test_a_mark_and_its_written_form_are_the_same_number():
    # DexScreener returns priceUsd in scientific notation for micro-caps. A
    # harness that parsed it as a float and printed it as an int would report a
    # price of zero and call it a crash.
    assert to_decimal("1.783e-05") == to_decimal(0.00001783)


# ---------------------------------------------------------------------------
# 17. executable price calculation
# ---------------------------------------------------------------------------

def test_effective_price_normalises_both_sides_exactly_once():
    # 1,928,811 lamports (9dp) -> 7,547,061.883 tokens (6dp)
    price = effective_price(
        1_928_811, 7_547_061_883,
        input_decimals=SOL_DECIMALS, output_decimals=TOKEN_DECIMALS,
    )
    # The expected value is computed at the harness's own precision, so this
    # compares two full-precision results rather than a precise one against an
    # imprecise one.
    with localcontext() as ctx:
        ctx.prec = PRECISION
        expected = Decimal("0.001928811") / Decimal("7547.061883")
    assert price == expected


def test_effective_price_is_not_rounded_before_the_division():
    # Rounding the inputs first would move this in the 6th significant digit.
    price = effective_price(
        1_928_811, 7_547_061_883,
        input_decimals=SOL_DECIMALS, output_decimals=TOKEN_DECIMALS,
    )
    rounded_first = (
        (Decimal("0.001928811").quantize(Decimal("1E-6")))
        / (Decimal("7547.061883").quantize(Decimal("1E-2")))
    )
    assert price != rounded_first


def test_effective_price_of_a_zero_output_is_an_error_not_zero():
    with pytest.raises(DecimalError):
        effective_price(1_000_000, 0, input_decimals=9, output_decimals=6)


def test_effective_price_rejects_a_negative_input():
    with pytest.raises(DecimalError):
        effective_price(-1, 1000, input_decimals=9, output_decimals=6)


# ---------------------------------------------------------------------------
# 18. mark / executable gap calculation
# ---------------------------------------------------------------------------

def test_the_worked_example_from_the_brief():
    # observed 1.783e-05, executable 1.820e-05 -> about +2.08%
    gap = price_gap_pct("1.820e-05", "1.783e-05")
    assert gap == pytest.approx(Decimal("2.08"), abs=Decimal("0.01"))


def test_the_large_gap_example_from_the_brief():
    # observed 1.783e-05, executable 3.108e-05 -> about +74%
    gap = price_gap_pct("3.108e-05", "1.783e-05")
    assert gap == pytest.approx(Decimal("74"), abs=Decimal("0.5"))


def test_gap_is_signed_and_positive_means_the_quote_is_worse():
    worse = price_gap_pct("2.0e-05", "1.0e-05")
    better = price_gap_pct("0.9e-05", "1.0e-05")
    assert worse > 0 > better
    assert worse == Decimal(100)


def test_identical_prices_give_exactly_zero_gap():
    assert price_gap_pct("1.783e-05", "1.783e-05") == 0


def test_a_zero_mark_has_no_percentage_gap():
    # Reporting inf here would be a fabrication.
    with pytest.raises(DecimalError):
        price_gap_pct("1e-5", 0)


def test_a_mark_return_is_a_separate_named_quantity():
    # (3.059e-05 / 1.783e-05 - 1) * 100 = +71.56%
    move = pct_return("1.783e-05", "3.059e-05")
    assert move == pytest.approx(Decimal("71.56"), abs=Decimal("0.01"))


# ---------------------------------------------------------------------------
# constant product + fees
# ---------------------------------------------------------------------------

def test_constant_product_never_exceeds_the_output_reserve():
    out = constant_product_out(30 * 10**9, 800_000_000 * 10**6, 1_150_000)
    assert 0 < out < 800_000_000 * 10**6


def test_constant_product_output_is_monotonic_in_input_size():
    small = constant_product_out(30 * 10**9, 800_000_000 * 10**6, 1_000_000)
    large = constant_product_out(30 * 10**9, 800_000_000 * 10**6, 2_000_000)
    assert large > small


def test_a_bigger_buy_worsens_the_effective_price():
    """Price impact, expressed the only way that is honest: by comparing."""
    def eff(in_lamports: int) -> Decimal:
        out = constant_product_out(30 * 10**9, 800_000_000 * 10**6, in_lamports)
        return effective_price(in_lamports, out, input_decimals=9,
                               output_decimals=6)

    assert eff(2_000_000) > eff(200_000)


def test_a_zero_reserve_is_refused_not_modelled():
    with pytest.raises(DecimalError):
        constant_product_out(0, 1000, 10)
    with pytest.raises(DecimalError):
        constant_product_out(1000, 0, 10)


def test_fee_split_reconciles_exactly():
    for total, bps in ((1_150_000, 100), (999, 25), (1, 0)):
        net, fee = apply_fee_bps(total, bps)
        assert net + fee == to_decimal(total)


def test_a_fee_at_or_above_one_hundred_percent_is_refused():
    with pytest.raises(DecimalError):
        apply_fee_bps(1000, 10_000)
    with pytest.raises(DecimalError):
        apply_fee_bps(1000, -1)


# ---------------------------------------------------------------------------
# order sizing
# ---------------------------------------------------------------------------

def test_usd_notional_converts_via_an_explicit_sol_reference():
    # $0.23 at $200/SOL = 0.00115 SOL = 1,150,000 lamports
    sol = sol_amount_for_usd("0.23", "200")
    assert sol == Decimal("0.00115")
    assert int(sol.scaleb(9)) == 1_150_000


def test_a_non_positive_sol_reference_is_refused():
    with pytest.raises(DecimalError):
        sol_amount_for_usd("0.23", 0)
    with pytest.raises(DecimalError):
        sol_amount_for_usd("0.23", -1)


# ---------------------------------------------------------------------------
# rendering safety
# ---------------------------------------------------------------------------

def test_non_finite_values_render_as_none_not_inf_or_nan():
    assert to_float(Decimal("Infinity")) is None
    assert to_float(Decimal("NaN")) is None
    assert to_float(float("inf")) is None
    assert to_float(None) is None
    assert to_float("abc") is None


def test_quantisation_is_for_display_only_and_documented_as_such():
    value = Decimal("1.783000000000001")
    assert quantise(value, 6) == "1.783000"
    # ...and the underlying value is untouched by having been displayed.
    assert value == Decimal("1.783000000000001")