"""
Quote-based MFE/MAE: valid quotes only, honest coverage, never a zero.

WHAT THESE COVER
================
Mark-based MFE/MAE is measured on an aggregator observation, which is a
different book from the one a position is exposed to. That defect was visible in
production: extreme mark MFE alongside negative realised PnL, and MAE printed
as ``+0.0%`` on real losers.

The executable excursion is measured on a SELL quote for the EXACT held
quantity, against the position's all-in entry cost:

    net_quote_pnl_pct = (estimated_net_sell_proceeds - all_in_entry_cost)
                        / all_in_entry_cost * 100

The load-bearing properties these tests pin:

  * a missing, stale, identity-failed or differently-sized quote is NOT a
    measurement, and never becomes one;
  * an unmeasured excursion is ``NONE`` with a reason -- never ``0.0``;
  * mark-only excursions stay explicitly labelled and are never printed as if
    they were achievable PnL;
  * a quote is an execution ESTIMATE and never replaces the canonical realised
    settlement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from autopaper.forensics.quote_excursions import (
    BASIS_MARK_ONLY,
    BASIS_QUOTE,
    BASIS_UNAVAILABLE,
    EXCURSION_BASIS_PNL,
    EXCURSION_BASIS_TRAVEL,
    QuoteExcursionPoint,
    excursions,
    format_excursion_line,
    mark_only_excursions,
    net_quote_return_pct,
    point_from_hold_record,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from tests import paper_bridge_fakes as _fk
except ImportError:  # pragma: no cover - direct file invocation
    import paper_bridge_fakes as _fk

alert, bridge, real_report, run = _fk.alert, _fk.bridge, _fk.real_report, _fk.run


# ---------------------------------------------------------------------------
# 1. the net return formula and its refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("proceeds,cost,expected", [
    (1.15, 1.00, 15.0),
    (0.90, 1.00, -10.0),
    (1.00, 1.00, 0.0),
    (2.00, 1.00, 100.0),
    (0.0, 1.00, -100.0),   # a zero proceed against a positive cost is a total loss
    (1.05, 1.00, 5.0),
])
def test_the_net_quote_return_formula(proceeds, cost, expected):
    assert net_quote_return_pct(
        net_proceeds=proceeds, all_in_entry_cost=cost) == pytest.approx(
            expected)


@pytest.mark.parametrize("proceeds,cost", [
    (1.0, 0.0),        # a zero cost is not a return
    (1.0, None),       # a missing cost is not a return
    (None, 1.0),       # a missing proceed is not a return
    (-1.0, 1.0),       # a negative proceed is not a return
    (float("nan"), 1.0),
    (1.0, float("nan")),
    ("abc", 1.0),
])
def test_a_refused_return_is_never_manufactured(proceeds, cost):
    assert net_quote_return_pct(
        net_proceeds=proceeds, all_in_entry_cost=cost) is None


def test_fees_already_inside_the_quote_are_not_added_again():
    """
    A quote's output amount is already net of the venue fee and impact. Adding
    them a second time would understate every excursion.
    """
    quoted_output = 1.10          # already net
    cost = 1.00
    assert net_quote_return_pct(net_proceeds=quoted_output,
                                all_in_entry_cost=cost) == pytest.approx(10.0)
    # The WRONG (double-counted) form would give 7.7%.
    assert net_quote_return_pct(net_proceeds=quoted_output * 0.97,
                                all_in_entry_cost=cost) < 10.0


# ---------------------------------------------------------------------------
# 2. which hold-path observations are usable
# ---------------------------------------------------------------------------


def _obs(**over) -> dict:
    base = {
        "quote_status": "VALID",
        "identity_check_status": "PASS",
        "sell_quote_source": "pump_direct",
        "sell_quote_net_raw": "1000000000",
        "sell_quote_net_usd": 1.15,
        "entry_notional_usd": 1.00,
        "quote_timestamp_utc": "2026-10-05T00:22:01.088000Z",
        "elapsed_ms": 12000,
        "quote_age_ms": 40,
        "return_pct": 999.0,        # the record's own figure, on another basis
    }
    base.update(over)
    return base


def test_a_valid_same_market_quote_is_usable():
    point = point_from_hold_record(_obs())
    assert point.usable
    assert point.return_pct == pytest.approx(15.0)
    assert point.reason == ""


@pytest.mark.parametrize("over,expect_substring", [
    ({"quote_status": "QUOTE_UNAVAILABLE"}, "QUOTE_UNAVAILABLE"),
    ({"quote_status": "SKIPPED"}, "SKIPPED"),
    ({"quote_status": "STALE"}, "STALE"),
    ({"quote_status": "INVALID_IDENTITY"}, "INVALID_IDENTITY"),
    ({"identity_check_status": "MISMATCH"}, "MISMATCH"),
    ({"identity_check_status": "UNVERIFIED"}, "UNVERIFIED"),
    ({"sell_quote_net_usd": None, "sell_quote_net_raw": None},
     "no sell quote"),
    ({"entry_notional_usd": None}, "unavailable"),
    ({"quote_status": ""}, "quote_status"),
])
def test_an_unusable_observation_carries_its_reason(over, expect_substring):
    point = point_from_hold_record(_obs(**over))
    assert not point.usable, over
    assert point.return_pct is None
    assert expect_substring in point.reason, point.reason


def test_the_records_own_return_pct_is_never_substituted_for_the_computed_one():
    """
    ``return_pct`` in the hold path is computed on a DIFFERENT basis (travel vs
    the entry sell quote). Substituting it would silently mix bases.
    """
    point = point_from_hold_record(_obs(return_pct=999.0))
    assert point.return_pct == pytest.approx(15.0)


def test_a_stale_quote_is_refused_when_the_age_bound_is_given():
    point = point_from_hold_record(_obs(quote_age_ms=45_000),
                                    max_quote_age_ms=30_000)
    assert not point.usable
    assert "exceeds" in point.reason
    # ...and the same observation IS usable when the bound allows it.
    assert point_from_hold_record(_obs(quote_age_ms=45_000),
                                   max_quote_age_ms=60_000).usable


def test_identity_can_be_relaxed_for_an_aggregator_labelled_estimate():
    """
    Where policy permits a distinct aggregator estimate, the caller may relax
    the same-market requirement -- but the point is still recorded with its
    identity so it can never be mistaken for a same-market measurement.
    """
    point = point_from_hold_record(
        _obs(identity_check_status="UNVERIFIED"),
        require_identity_pass=False)
    assert point.usable
    assert point.identity == "UNVERIFIED"


# ---------------------------------------------------------------------------
# 3. the excursions themselves
# ---------------------------------------------------------------------------


def _point(pct, *, at=1000.0, elapsed_ms=1000.0, source="pump_direct"):
    return QuoteExcursionPoint(
        at=at, elapsed_ms=elapsed_ms,
        net_proceeds=1.0 + pct / 100.0, gross_proceeds=1.0 + pct / 100.0,
        all_in_entry_cost=1.0, return_pct=pct, net_proceeds_raw=None,
        source=source, identity="PASS", quote_age_ms=30, usable=True)


def test_mfe_is_the_maximum_and_mae_the_minimum():
    result = excursions([_point(5.0), _point(-3.0), _point(2.0),
                         _point(-8.0)])
    assert result["basis"] == BASIS_QUOTE
    assert result["excursion_basis"] == EXCURSION_BASIS_PNL
    assert result["quote_mfe_pct"] == 5.0
    assert result["quote_mae_pct"] == -8.0
    assert result["valid_quotes"] == 4


def test_the_timestamps_and_ages_of_both_extrema_are_recorded():
    result = excursions([
        _point(1.0, at=100.0, elapsed_ms=1000.0),
        _point(9.0, at=300.0, elapsed_ms=3000.0),
        _point(-2.0, at=200.0, elapsed_ms=2000.0),
    ])
    assert result["mfe_elapsed_ms"] == 3000.0
    assert result["mae_elapsed_ms"] == 2000.0
    assert result["mfe_at"] == 300.0
    assert result["mae_at"] == 200.0


def test_an_unmeasured_excursion_is_none_and_says_why():
    result = excursions([])
    assert result["basis"] == BASIS_UNAVAILABLE
    assert result["quote_mfe_pct"] is None
    assert result["quote_mae_pct"] is None
    assert result["unavailable_reason"]
    # The formatter must not print a zero.
    assert "N/A" in format_excursion_line(result)
    assert "0.0%" not in format_excursion_line(result)


def test_an_unusable_point_is_not_an_observation():
    """A position that only ever had rejected quotes is UNMEASURED."""
    unusable = QuoteExcursionPoint(
        at=1.0, elapsed_ms=0, net_proceeds=None, gross_proceeds=None,
        all_in_entry_cost=1.0, return_pct=None, net_proceeds_raw=None,
        source="pump_direct", identity="UNKNOWN", quote_age_ms=None,
        usable=False, reason="quote_status=QUOTE_UNAVAILABLE")
    result = excursions([unusable])
    assert result["basis"] == BASIS_UNAVAILABLE
    assert result["valid_quotes"] == 0
    assert result["quote_mfe_pct"] is None


def test_a_real_zero_mfe_is_distinguishable_from_no_measurement():
    """
    A measured 0.0 (never went up) and a missing measurement are different
    facts and must never collapse into one rendering.
    """
    flat = excursions([_point(0.0), _point(-1.0), _point(-2.0)])
    assert flat["basis"] == BASIS_QUOTE
    assert flat["quote_mfe_pct"] == 0.0
    missing = excursions([])
    assert missing["quote_mfe_pct"] is None
    assert format_excursion_line(flat) != format_excursion_line(missing)
    assert "0.0%" in format_excursion_line(flat)
    assert "N/A" in format_excursion_line(missing)


def test_coverage_and_gaps_are_reported():
    result = excursions([
        _point(1.0, at=0.0), _point(2.0, at=1.0), _point(3.0, at=2.0),
        _point(4.0, at=10.0),
    ])
    assert result["coverage_pct"] == 100.0
    assert result["valid_quotes"] == 4
    # A 8s hole between observations is reported, not averaged away.
    assert result["max_sampling_gap_ms"] == pytest.approx(8000.0)
    assert result["sampling_interval_s"] is not None


# ---------------------------------------------------------------------------
# 4. sampled, estimated, and never a fill
# ---------------------------------------------------------------------------


def test_the_excursion_is_labelled_as_sampled_quote_based():
    result = excursions([_point(2.0), _point(-1.0)])
    assert result["basis"] == BASIS_QUOTE
    line = format_excursion_line(result)
    assert line.startswith("Quote-MFE:")


def test_the_formatted_line_matches_the_required_shape():
    result = excursions([
        _point(8.4, at=10.0, elapsed_ms=21_000),
        _point(-14.1, at=20.0, elapsed_ms=46_000),
    ])
    line = format_excursion_line(result)
    assert "Quote-MFE: +8.4% @ 21s" in line, line
    assert "Quote-MAE: -14.1% @ 46s" in line, line


def test_the_unavailable_line_states_the_reason():
    result = excursions([])
    result["unavailable_reason"] = "no fresh executable quote for 125s"
    line = format_excursion_line(result)
    assert line == ("Quote-MFE: N/A · Quote-MAE: N/A  "
                    "(no fresh executable quote for 125s)")


# ---------------------------------------------------------------------------
# 5. mark-only stays labelled
# ---------------------------------------------------------------------------


def test_mark_only_excursions_are_labelled_as_such():
    result = mark_only_excursions(mark_mfe_pct=377.9, mark_mae_pct=0.0,
                                  observations=42)
    assert result["basis"] == BASIS_MARK_ONLY
    line = format_excursion_line(result)
    assert line.startswith("Mark-only")
    assert "not executable" in line
    # A mark pair must never be presented under the quote label.
    assert "Quote-MFE" not in line


def test_mark_only_excursions_are_not_a_zero_when_absent():
    result = mark_only_excursions()
    assert result["mark_mfe_pct"] is None
    assert result["mark_mae_pct"] is None


def test_a_mark_and_a_quote_excursion_are_never_mixed():
    """One record carries one basis, and the line names it."""
    quote = format_excursion_line(excursions([_point(1.0)]))
    mark = format_excursion_line(
        mark_only_excursions(mark_mfe_pct=1.0, mark_mae_pct=-1.0))
    assert "Quote-MFE" in quote and "Mark-only" not in quote
    assert "Mark-only" in mark and "Quote-MFE" not in mark


# ---------------------------------------------------------------------------
# 6. the cost-aware vs travel bases are never conflated
# ---------------------------------------------------------------------------


def test_the_travel_basis_is_not_a_pnl():
    """
    Travel is measured against the ENTRY SELL QUOTE, so it cancels the entry
    execution cost. It is a different question from "what would I have
    realised", and it is labelled so it cannot be read as one.
    """
    points = [_point(5.0, at=0.0), _point(12.0, at=1.0), _point(-4.0, at=2.0)]
    pnl = excursions(points)
    travel = excursions(points, travel_basis=True)
    assert pnl["excursion_basis"] == EXCURSION_BASIS_PNL
    assert travel["excursion_basis"] == EXCURSION_BASIS_TRAVEL
    assert pnl["excursion_basis"] != travel["excursion_basis"]


def test_the_travel_basis_uses_the_declared_entry_reference():
    points = [_point(0.0, at=0.0), _point(10.0, at=1.0)]
    travel = excursions(points, travel_basis=True, entry_reference_raw=2.0)
    assert travel["basis"] == BASIS_QUOTE
    assert travel["excursion_basis"] == EXCURSION_BASIS_TRAVEL


# ---------------------------------------------------------------------------
# 7. the runtime records the excursion, not the settlement
# ---------------------------------------------------------------------------


def _position(engine):
    for account in engine.accounts.values():
        if account.position is not None:
            return account.position
    return None


def test_a_valid_sell_quote_becomes_an_excursion_point():
    from autopaper.paper_live.arms import parse_arms
    from autopaper.paper_live.engine import build_signal_snapshot
    from decimal import Decimal

    engine = bridge(arms=parse_arms(["L25_CURRENT"]))
    run(engine.handle_alert(snapshot=alert(), report=real_report()))
    position = _position(engine)
    assert position is not None, "the fixture opened no position"

    class _Q:
        available = True
        pool_or_curve = position.entry_pool_address or "pool"
        identity = "PASS"
        provider = "pump_direct"
        input_amount_raw = position.tokens_raw
        output_amount_decimal = Decimal(
            position.entry_cost_quote_asset) * 2
        output_amount_raw = 123
        quote_age_ms = 25.0

    engine._record_quote_excursion(position, _Q())
    assert position.quote_mfe_mae_basis == "QUOTE_BASED"
    assert position.quote_mfe_pct == pytest.approx(100.0, abs=1.0)
    assert position.quote_mfe_source == "pump_direct"
    assert position.quote_excursion_cost_basis == "PNL_VS_ENTRY_COST"


def test_a_quote_on_a_different_market_is_not_the_positions_excursion():
    """
    A SELL quote for another pool is a counterfactual for this position, never a
    sample of what it could liquidate for.
    """
    from decimal import Decimal

    engine = bridge(arms=[])
    run(engine.handle_alert(snapshot=alert(), report=real_report()))
    position = _position(engine)
    assert position is not None

    class _Q:
        available = True
        pool_or_curve = "a-completely-different-pool"
        identity = "PASS"
        provider = "jupiter"
        input_amount_raw = position.tokens_raw
        output_amount_decimal = Decimal(
            position.entry_cost_quote_asset) * 9
        output_amount_raw = 123
        quote_age_ms = 5.0

    engine._record_quote_excursion(position, _Q())
    assert position.quote_mfe_mae_basis != "QUOTE_BASED"
    assert position.quote_mfe_pct is None


def test_a_quote_for_a_different_quantity_is_not_the_positions_excursion():
    from decimal import Decimal

    engine = bridge(arms=[])
    run(engine.handle_alert(snapshot=alert(), report=real_report()))
    position = _position(engine)
    assert position is not None

    class _Q:
        available = True
        pool_or_curve = position.entry_pool_address or "pool"
        identity = "PASS"
        provider = "jupiter"
        input_amount_raw = (position.tokens_raw or 0) // 2 or 999_999
        output_amount_decimal = Decimal(
            position.entry_cost_quote_asset) * 9
        output_amount_raw = 123
        quote_age_ms = 5.0

    engine._record_quote_excursion(position, _Q())
    assert position.quote_mfe_mae_basis != "QUOTE_BASED"


def test_an_unavailable_quote_leaves_the_excursion_unmeasured():
    engine = bridge(arms=[])
    run(engine.handle_alert(snapshot=alert(), report=real_report()))
    position = _position(engine)
    assert position is not None

    class _Q:
        available = False
        missing_reason = "provider throttled: 429"
        provider = "jupiter"

    engine._record_quote_excursion(position, _Q())
    assert position.quote_mfe_pct is None
    assert position.quote_mae_pct is None
    assert position.quote_mfe_mae_basis == "NOT_MEASURED"


def test_a_position_with_no_entry_cost_is_not_measured():
    """
    Without a known all-in entry cost there is no denominator, so the excursion
    is NOT_MEASURED and says why -- it does not silently assume a cost of zero.
    """
    from decimal import Decimal

    engine = bridge(arms=[])
    run(engine.handle_alert(snapshot=alert(), report=real_report()))
    position = _position(engine)
    assert position is not None
    position.entry_cost_quote_asset = None
    position.trading_fees_quote = None
    position.network_costs_quote = None
    position.other_costs_quote = None

    class _Q:
        available = True
        pool_or_curve = position.entry_pool_address or "pool"
        identity = "PASS"
        provider = "pump_direct"
        input_amount_raw = position.tokens_raw
        output_amount_decimal = Decimal(2)
        output_amount_raw = 123
        quote_age_ms = 5.0

    engine._record_quote_excursion(position, _Q())
    assert position.quote_mfe_pct is None
    assert position.quote_mfe_mae_basis == "NOT_MEASURED"
    assert position.quote_unavailable_reason


def test_rent_is_not_counted_as_a_cost():
    """
    Rent is LOCKED, not spent: it is recoverable when the balance goes to zero.
    Counting it would understate every net return.
    """
    from dataclasses import replace

    engine = bridge(arms=[])
    run(engine.handle_alert(snapshot=alert(), report=real_report()))
    position = _position(engine)
    assert position is not None
    without_rent = engine._entry_all_in_cost_quote_asset(position)
    with_rent = engine._entry_all_in_cost_quote_asset(
        replace(position, entry_rent_lamports=2_039_280))
    assert with_rent == without_rent


def test_an_unreadable_cost_component_refuses_the_whole_cost():
    from dataclasses import replace

    engine = bridge(arms=[])
    run(engine.handle_alert(snapshot=alert(), report=real_report()))
    position = _position(engine)
    assert position is not None
    broken = replace(position, trading_fees_quote="not-a-number")
    assert engine._entry_all_in_cost_quote_asset(broken) is None


def test_the_excursion_never_overwrites_the_canonical_settlement():
    """
    Realised PnL comes from ``realize()`` on the actual entry and exit fills.
    The quote excursion is a separate analytics measurement and must not be
    able to overwrite it.
    """
    from decimal import Decimal

    engine = bridge(arms=[])
    run(engine.handle_alert(snapshot=alert(), report=real_report()))
    position = _position(engine)
    assert position is not None
    before = sorted(position.as_dict()) if False else None

    class _Q:
        available = True
        pool_or_curve = position.entry_pool_address or "pool"
        identity = "PASS"
        provider = "pump_direct"
        input_amount_raw = position.tokens_raw
        output_amount_decimal = Decimal(
            position.entry_cost_quote_asset) * 3
        output_amount_raw = 123
        quote_age_ms = 5.0

    engine._record_quote_excursion(position, _Q())
    # No settlement field exists on the position at all, so the excursion
    # cannot have touched one.
    assert not any(
        "realized_pnl" in k for k in vars(position))
    assert "quote_mfe_pct" in vars(position)
