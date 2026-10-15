"""
The compact closed-position report: readable, labelled, and never lying.

WHAT THESE COVER
================
The old summary was long and its MFE/MAE line printed ``MFE: +377.9% | MAE:
+0.0% (observed mark)`` next to a realised loss. Two problems in one line: the
MAE zero came from a seeding defect on the mark, and the MFE was a mark, which is
a different book from the sold quantity's proceeds.

These tests pin the properties that make the replacement trustworthy:

  * a row's result category is DERIVED from its recorded provenance, so a
    mark-only figure cannot be labelled as a settled result;
  * a missing quote excursion renders as ``N/A`` with a reason, never ``0.0%``;
  * mark-only analytics appear only under an explicit ``Mark-only`` label;
  * the data-quality warnings actually fire and actually don't;
  * the compact row stops repeating CA / trade id / route on every line.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from autopaper.forensics.compact_report import (
    CATEGORY_MARK_ONLY,
    CATEGORY_PAPER_SIM,
    CATEGORY_SETTLED,
    CATEGORY_UNAVAILABLE,
    classify_result,
    compact_summary,
    data_quality_warnings,
    render_compact,
    render_excursion,
    render_mark_only,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE = {
    "symbol": "ALTERNET",
    "realized_pnl_usd": -0.090,
    "return_pct": -26.3,
    "holding_seconds": 284,
    "exit_reason": "market data unusable for 125s",
    "exit_reason_code": "data_invalid",
    "pnl_basis": "PAPER",
    "trade_mode": "PAPER",
    "mfe_basis": "OBSERVED_MARK",
}


def _row(**over) -> dict:
    row = dict(BASE)
    row.update(over)
    return row


# ---------------------------------------------------------------------------
# 1. the compact shape
# ---------------------------------------------------------------------------


def test_the_headline_carries_token_outcome_and_hold():
    s = compact_summary(_row())
    assert s["symbol"] == "ALTERNET"
    assert s["pnl_usd"] == pytest.approx(-0.090)
    assert s["return_pct"] == pytest.approx(-26.3)
    assert s["hold_s"] == pytest.approx(284)
    out = render_compact(_row())
    first = out.splitlines()[0]
    assert "ALTERNET" in first
    assert "-$0.0900" in first
    assert "-26.3%" in first
    assert "held" in first


def test_the_exit_reason_is_its_own_line():
    out = render_compact(_row())
    assert any(line.startswith("Exit:") for line in out.splitlines())


def test_the_compact_row_does_not_repeat_the_per_trade_boilerplate():
    """
    The detail fields belong in the detail view, so a list of forty positions
    is readable instead of forty blocks of CA / route / trade-id.
    """
    out = render_compact(_row(
        mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        trade_id="sig_abc:L25_CURRENT",
        route="pump_direct",
        entry_venue_id="PUMP_BONDING_CURVE:xyz",
    ))
    assert "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v" not in out
    assert "sig_abc:L25_CURRENT" not in out
    assert "PUMP_BONDING_CURVE" not in out


def test_the_detail_view_carries_exactly_what_the_compact_row_drops():
    from autopaper.telegram.ui import render_closed_position_detail
    text = render_closed_position_detail(_row(
        mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        trade_id="sig_abc:L25_CURRENT",
        route="pump_direct",
    ))
    assert "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v" in text
    assert "sig_abc:L25_CURRENT" in text
    assert "pump_direct" in text


# ---------------------------------------------------------------------------
# 2. the four categories are derived and never blended
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row,expected", [
    ({"pnl_basis": "FILL_VS_FILL"}, CATEGORY_PAPER_SIM),
    ({"pnl_basis": "PAPER_SIMULATED_FILL"}, CATEGORY_PAPER_SIM),
    ({"pnl_basis": "MARK_ONLY"}, CATEGORY_MARK_ONLY),
    ({"exit_price_provenance": "MARK_ONLY"}, CATEGORY_MARK_ONLY),
    ({"settlement_status": "SETTLED"}, CATEGORY_SETTLED),
])
def test_the_category_is_derived_from_the_recorded_provenance(row, expected):
    category, _ = classify_result(_row(**row))
    assert category == expected, row


def test_a_row_with_no_pnl_and_no_provenance_is_unavailable():
    category, reason = classify_result(_row(realized_pnl_usd=None,
                                            return_pct=None, pnl_basis=""))
    assert category == CATEGORY_UNAVAILABLE
    assert reason


def test_a_mark_only_row_states_its_category_on_screen():
    out = render_compact(_row(pnl_basis="MARK_ONLY"))
    assert "mark-only analytics" in out
    assert "Result: mark-only analytics" in out


def test_a_settled_row_does_not_need_a_category_label():
    out = render_compact(_row(pnl_basis="SETTLED"))
    assert "Result:" not in out, (
        "a canonical settlement is the default reading and needs no disclaimer")


# ---------------------------------------------------------------------------
# 3. quote MFE/MAE: present, absent, and labelled
# ---------------------------------------------------------------------------


def test_a_measured_quote_excursion_renders_with_ages_and_coverage():
    out = render_compact(_row(
        quote_mfe_mae_basis="QUOTE_BASED",
        quote_mfe_pct=8.4, quote_mae_pct=-14.1,
        quote_mfe_elapsed_ms=21_000, quote_mae_elapsed_ms=46_000,
        quote_coverage_pct=62.0,
    ))
    assert "Quote-MFE: +8.4% @ 21s" in out
    assert "Quote-MAE: -14.1% @ 46s" in out
    assert "coverage 62%" in out


def test_an_unmeasured_quote_excursion_is_na_with_a_reason():
    """
    An absent excursion is still ``N/A`` with a stated reason.

    WHAT CHANGED: the reason is no longer the guessed "no valid same-market SELL
    quote was obtained for this position". That sentence asserts a fact about
    quoting which the row does not necessarily support -- it was false for every
    trade that settled on an accepted quote but was never sampled for one, which
    is most of them. The renderer now reports what the row actually states: that
    it carries no quote-excursion telemetry at all.
    """
    out = render_compact(_row(
        quote_mfe_mae_basis="NOT_MEASURED",
        quote_coverage_pct=12.0,
    ))
    assert "Quote-MFE: N/A" in out
    assert "Quote-MAE: N/A" in out
    # The reason is stated, not left as a bare unexplained N/A.
    assert "no quote-excursion telemetry" in out
    assert "coverage 12%" in out
    # The guessing sentence must not come back through any unmeasured path.
    assert "no valid same-market SELL quote was obtained" not in out


def test_a_settled_but_unsampled_row_says_that_instead_of_claiming_no_quote():
    """
    The specific misreading this fixes: a trade that DID settle on a valid
    executable SELL quote, but recorded no intratrade observation.
    """
    out = render_compact(_row(
        quote_mfe_mae_basis="NOT_MEASURED",
        quote_telemetry_state="NO_INTRATRADE_SAMPLES",
        quote_settlement_status="ACCEPTED",
    ))
    assert "Quote-MFE: N/A" in out
    assert "valid settlement SELL quote was accepted" in out
    assert "no valid same-market SELL quote was obtained" not in out


def test_a_measured_pair_states_how_many_quotes_backed_it():
    out = render_compact(_row(
        quote_mfe_pct=8.4,
        quote_mae_pct=-14.1,
        quote_mfe_mae_basis="QUOTE_BASED",
        quote_mfe_elapsed_ms=21_000.0,
        quote_mae_elapsed_ms=46_000.0,
        quote_observation_count=5,
        quote_coverage_pct=62.0,
    ))
    assert "5 quote samples" in out


def test_a_missing_excursion_is_never_printed_as_zero():
    out = render_compact(_row())
    assert "MAE: 0.0%" not in out
    assert "MAE: +0.0%" not in out
    assert "0.0%" not in out


def test_a_measured_zero_mfe_is_preserved_as_a_real_zero():
    """
    "Never went up" is a measurement, and must render as +0.0% -- not be
    confused with, or collapsed into, a missing measurement.
    """
    out = render_compact(_row(quote_mfe_mae_basis="QUOTE_BASED",
                              quote_mfe_pct=0.0, quote_mae_pct=-4.0,
                              quote_mfe_elapsed_ms=5000,
                              quote_mae_elapsed_ms=9000,
                              quote_coverage_pct=90.0))
    assert "Quote-MFE: +0.0% @ 5s" in out


def test_the_unavailable_reason_the_operator_gave_is_shown_verbatim():
    out = render_compact(_row(
        quote_mfe_mae_basis="NOT_MEASURED",
        quote_unavailable_reason="all-in entry cost is unavailable",
    ))
    assert "all-in entry cost is unavailable" in out


# ---------------------------------------------------------------------------
# 4. mark-only stays labelled, and is never an executable number
# ---------------------------------------------------------------------------


def test_a_mark_excursion_is_only_ever_rendered_under_its_own_label():
    summary = compact_summary(_row(mfe_pct=377.9, mae_pct=0.0))
    line = render_mark_only(summary)
    assert line is not None
    assert "Mark-only MFE" in line
    assert "NOT an executable price" in line
    assert "Quote-MFE" not in line


def test_the_compact_row_shows_the_mark_pair_only_when_it_exists():
    """The mark pair is added by the UI layer, so the whole block is rendered."""
    from autopaper.telegram.ui import _closed_position_block
    with_mark = "\n".join(_closed_position_block(_row(mfe_pct=12.0,
                                                       mae_pct=-3.0)))
    without_mark = "\n".join(_closed_position_block(_row(mfe_pct=None,
                                                          mae_pct=None)))
    assert "Mark-only" in with_mark
    assert "Mark-only" not in without_mark


def test_a_mark_mfe_is_never_shown_as_a_quote_mfe():
    """
    The exact production defect: ``MFE +377.9%`` printed next to a negative
    realised PnL, with no way to tell it was a mark.
    """
    from autopaper.telegram.ui import _closed_position_block
    block = "\n".join(_closed_position_block(_row(
        mfe_pct=377.9, mae_pct=0.0, realized_pnl_usd=-0.09, return_pct=-26.3)))
    # The mark figure appears ONLY after its own label, never on the quote line.
    assert block.index("Mark-only MFE: +377.9%") > block.index("Quote-MFE")
    assert "Quote-MFE: N/A" in block
    # And the mark pair never carries the executable label.
    assert "Quote-MFE: +377.9%" not in block


def test_the_mark_and_quote_pairs_are_on_separate_lines():
    from autopaper.telegram.ui import _closed_position_block
    out = "\n".join(_closed_position_block(_row(
        mfe_pct=377.9, mae_pct=0.0,
        quote_mfe_mae_basis="QUOTE_BASED", quote_mfe_pct=2.0,
        quote_mae_pct=-30.0, quote_mfe_elapsed_ms=1000,
        quote_mae_elapsed_ms=2000, quote_coverage_pct=40.0,
    )))
    lines = out.splitlines()
    quote_lines = [i for i, l in enumerate(lines) if l.startswith("Quote-MFE")]
    mark_lines = [i for i, l in enumerate(lines) if "Mark-only" in l]
    assert len(quote_lines) == 1 and len(mark_lines) == 1
    assert quote_lines[0] != mark_lines[0]


# ---------------------------------------------------------------------------
# 5. data-quality warnings
# ---------------------------------------------------------------------------


def test_a_stale_mark_on_a_mark_exit_is_a_warning():
    summary = compact_summary(_row(mark_age_ms=125_000,
                                   pnl_basis="MARK_ONLY"))
    warnings = data_quality_warnings(summary)
    assert any("125s old" in w for w in warnings), warnings


def test_a_stale_mark_on_a_quote_validated_exit_is_not_a_defect():
    """
    A fresh executable quote validated the exit, so a stale aggregator
    observation is a diagnostic and must not be presented as a defect in the
    result.
    """
    summary = compact_summary(_row(
        mark_age_ms=125_000,
        quote_mfe_mae_basis="QUOTE_BASED", quote_mfe_pct=1.0,
        quote_mae_pct=-1.0, quote_mfe_elapsed_ms=1, quote_mae_elapsed_ms=2,
        quote_coverage_pct=95.0))
    warnings = data_quality_warnings(summary)
    assert not any("old at the exit attempt" in w for w in warnings), warnings


def test_a_low_coverage_position_says_the_excursion_is_partial():
    summary = compact_summary(_row(
        quote_mfe_mae_basis="QUOTE_BASED", quote_mfe_pct=5.0, quote_mae_pct=-5.0,
        quote_mfe_elapsed_ms=1, quote_mae_elapsed_ms=2, quote_coverage_pct=12.0))
    warnings = data_quality_warnings(summary)
    assert any("only 12%" in w for w in warnings), warnings


def test_a_large_sampling_gap_says_intrasecond_moves_are_not_captured():
    summary = compact_summary(_row(
        quote_mfe_mae_basis="QUOTE_BASED", quote_mfe_pct=5.0, quote_mae_pct=-5.0,
        quote_mfe_elapsed_ms=1, quote_mae_elapsed_ms=2, quote_coverage_pct=95.0,
        quote_max_sampling_gap_ms=45_000))
    warnings = data_quality_warnings(summary)
    assert any("longest gap" in w for w in warnings), warnings
    assert any("NOT" in w for w in warnings), warnings


def test_a_healthy_row_gets_no_warning_lines():
    summary = compact_summary(_row(
        quote_mfe_mae_basis="QUOTE_BASED", quote_mfe_pct=5.0, quote_mae_pct=-5.0,
        quote_mfe_elapsed_ms=1, quote_mae_elapsed_ms=2, quote_coverage_pct=95.0,
        quote_max_sampling_gap_ms=3_000, mark_age_ms=2_000))
    assert data_quality_warnings(summary) == []


# ---------------------------------------------------------------------------
# 6. missing data never becomes a number
# ---------------------------------------------------------------------------


def test_a_row_with_no_pnl_renders_a_pnl_unavailable_line_not_a_zero():
    out = render_compact(_row(realized_pnl_usd=None, return_pct=None,
                              pnl_basis=""))
    assert "$0.0000" not in out
    assert "PnL unavailable" in out


@pytest.mark.parametrize("field", [
    "quote_mfe_pct", "quote_mae_pct", "quote_mfe_elapsed_ms",
    "quote_coverage_pct", "quote_max_sampling_gap_ms", "mark_age_ms",
])
def test_a_missing_field_never_becomes_number_zero(field):
    summary = compact_summary(_row())
    assert summary[field] is None, field


def test_a_nan_field_never_survives_into_a_rendered_number():
    out = render_compact(_row(quote_mfe_pct=float("nan"),
                              quote_mfe_mae_basis="QUOTE_BASED",
                              quote_mae_pct=float("nan")))
    assert "nan" not in out.lower()


def test_the_exit_reason_for_a_data_failure_says_how_long_it_was_unknown():
    out = render_compact(_row(mark_age_ms=125_000))
    assert "no fresh executable quote for 125s" in out


def test_a_take_profit_exit_is_not_labelled_as_a_data_failure():
    out = render_compact(_row(
        exit_reason="take-profit +20.0% reached at +21.8% after 34s",
        exit_reason_code="tp20"))
    assert "no fresh executable quote" not in out
    assert "take-profit" in out
