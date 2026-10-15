"""
GOAT regression (test item 26).

The incident this harness was built to investigate:

    observed entry mark  1.783e-05
    observed exit mark   3.059e-05      ->  +71.56% mark-to-mark
    historical entry     ~7,547,061.883 GOAT for ~1,928,811 lamports SOL

``+71.56%`` reads like a return. It is a move between two OBSERVED MARKS. This
module exists so that reading cannot come back.
"""


from __future__ import annotations

import json
from decimal import Decimal, localcontext

import pytest

from autopaper.same_market.goat import (
    HISTORICAL_ENTRY_SOL_LAMPORTS,
    HISTORICAL_ENTRY_TOKEN_AMOUNT,
    OBSERVED_ENTRY_MARK,
    OBSERVED_EXIT_MARK,
    FillUnavailable,
    classify,
    load_goat_incident,
    render_goat_regression,
    require_actual_fill,
)
from autopaper.same_market.models import ACTUAL_FILL_AVAILABLE
from autopaper.same_market.numeric import PRECISION, pct_return

pytestmark = pytest.mark.same_market


@pytest.fixture(scope="module")
def incident():
    return load_goat_incident()


# ---------------------------------------------------------------------------
# the arithmetic
# ---------------------------------------------------------------------------

def test_the_mark_to_mark_move_is_plus_seventy_one_point_five_six(incident):
    assert incident.observed_mark_return_pct == pytest.approx(
        Decimal("71.56"), abs=Decimal("0.01")
    )
    assert incident.observed_mark_return_pct > 0


def test_the_figure_reproduces_from_the_raw_marks():
    assert pct_return(OBSERVED_ENTRY_MARK, OBSERVED_EXIT_MARK) == \
        incident_return()


def incident_return() -> Decimal:
    return load_goat_incident().observed_mark_return_pct


def test_the_historical_entry_amounts_are_preserved_exactly(incident):
    assert incident.historical_entry_lamports == 1_928_811
    assert incident.historical_entry_token_amount == Decimal("7547061.883")


# ---------------------------------------------------------------------------
# the three values are different things
# ---------------------------------------------------------------------------

def test_the_three_values_are_not_interchangeable(incident):
    assert incident.actual_fill is None
    assert incident.actual_fill_available is False
    assert ACTUAL_FILL_AVAILABLE is False
    # The mark move is not the executable price.
    assert incident.observed_mark_return_pct != incident.executable_price_native


def test_the_mark_return_is_explicitly_not_a_trade_return(incident):
    payload = incident.as_dict()
    assert payload["observed_mark"]["mark_return_is_a_trade_return"] is False
    assert payload["executable_quote"]["is_a_fill"] is False
    assert payload["actual_fill"] is None
    assert payload["actual_fill_available"] is False


def test_the_executable_price_is_denominated_in_sol_not_usd(incident):
    """
    The entry amounts are lamports and tokens. Calling the resulting price USD
    would be a unit error, and a unit error in a measurement harness makes the
    whole comparison meaningless while looking perfectly plausible.
    """
    payload = incident.as_dict()
    assert "SOL per token" in payload["executable_quote"]["unit"]
    assert payload["observed_mark"]["unit"] == "USD per token"
    # 0.001928811 SOL / 7,547,061.883 GOAT.
    #
    # The token figure is a HUMAN count, so the raw integer handed to
    # effective_price is 7,547,061.883 * 10**6 = 7,547,061,883,000. Skipping
    # that scale is a factor-of-a-million bug; applying it twice is the same
    # bug mirrored. This assertion is what catches either one.
    with localcontext() as ctx:
        ctx.prec = PRECISION
        expected = Decimal("0.001928811") / Decimal("7547061.883")
    assert incident.executable_price_native == expected
    assert incident.executable_price_native == Decimal(
        "2.55571112295330307151795750023002693325073401945497205087645E-10"
    )


def test_the_human_token_amount_is_scaled_exactly_once(incident):
    """Guards the specific 10**6 regression above, in both directions."""
    raw = int(HISTORICAL_ENTRY_TOKEN_AMOUNT.scaleb(6))
    assert raw == 7_547_061_883_000
    assert Decimal(raw).scaleb(-6) == HISTORICAL_ENTRY_TOKEN_AMOUNT
    # 2.5e-10 SOL/token, not 2.5e-7 and not 2.5e-4.
    assert Decimal("1e-10") < incident.executable_price_native < Decimal("1e-9")


def test_a_usd_conversion_requires_an_explicit_sol_reference(incident):
    assert incident.executable_price_usd(None) is None
    assert incident.mark_gap_vs_executable(None) is None
    converted = incident.executable_price_usd(200)
    assert converted == incident.executable_price_native * Decimal(200)
    assert incident.mark_gap_vs_executable(200) is not None


def test_a_bad_sol_reference_yields_no_conversion_rather_than_a_guess(incident):
    for bad in (0, -1, "not a price", None):
        assert incident.executable_price_usd(bad) is None


# ---------------------------------------------------------------------------
# the discrepancy the incident is really about
# ---------------------------------------------------------------------------

def test_the_mark_and_the_historical_entry_disagree_by_a_large_factor(incident):
    """
    The entry mark implies the 7,547,061.883 GOAT were worth ~$134.56, while
    the actual entry spent 0.001928811 SOL. Those two facts cannot both
    describe the same market at the same moment - which is the finding, made
    concrete rather than asserted.
    """
    implied = incident.mark_implied_spend()
    assert implied is not None
    assert implied > Decimal(100)

    actual_sol = Decimal(HISTORICAL_ENTRY_SOL_LAMPORTS).scaleb(-9)
    assert actual_sol == Decimal("0.001928811")
    # At any plausible SOL price the historical spend is cents, not $134.
    assert actual_sol * Decimal(500) < implied


# ---------------------------------------------------------------------------
# an actual fill is not obtainable, and that is enforced
# ---------------------------------------------------------------------------

def test_requesting_a_fill_raises_rather_than_returning_a_value():
    with pytest.raises(FillUnavailable) as exc:
        require_actual_fill()
    assert "never obtains an actual fill" in str(exc.value)


def test_classify_labels_the_kinds_apart(incident):
    assert classify(incident.observed_entry_mark, "OBSERVED_MARK") == "OBSERVED_MARK"
    assert classify(incident.executable_price_native,
                    "EXECUTABLE_QUOTE") == "EXECUTABLE_QUOTE"
    assert classify(incident.actual_fill, "ACTUAL_FILL") == "ACTUAL_FILL"


def test_a_mark_cannot_be_passed_off_as_a_fill():
    """
    The confusion the regression guards against, stated as a property: the
    mark is a number and a fill is ``None``, and no coercion turns one into
    the other.
    """
    incident = load_goat_incident()
    assert incident.actual_fill is None
    with pytest.raises(FillUnavailable):
        require_actual_fill()
    # The mark exists; a fill does not. Both facts hold at once.
    assert incident.observed_entry_mark > 0


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def test_the_rendering_labels_everything_and_runs_offline():
    text = render_goat_regression()
    assert "OBSERVED_MARK" in text
    assert "EXECUTABLE_QUOTE" in text
    assert "ACTUAL_FILL" in text
    assert "NOT a trade return" in text
    assert "71.56%" in text
    assert "denominated in SOL, not USD" in text
    assert "UNKNOWN" in text, "no SOL reference was supplied"


def test_the_rendering_with_a_reference_shows_the_cross_check():
    text = render_goat_regression(sol_usd=200)
    assert "at $200/SOL" in text
    assert "UNKNOWN" not in text.split("CROSS-CHECKS")[1].split("INTERPRETATION")[0]


def test_the_fixture_is_deterministic_and_needs_no_network():
    a, b = load_goat_incident(), load_goat_incident()
    assert a.observed_mark_return_pct == b.observed_mark_return_pct
    assert a.executable_price_native == b.executable_price_native
    json.dumps(a.as_dict(), default=str)   # must be serialisable for the report