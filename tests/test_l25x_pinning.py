"""
Offline tests for entry-venue identity persistence and pinned exits.

Covers required tests 5-8: identity persistence, pinned exit, exit-venue
unavailable behaviour, and no mark fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

AI = Path(__file__).resolve().parents[1]
for _p in (AI, AI / "research"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from autopaper.same_market.models import (  # noqa: E402
    CheckState,
    FailureMode,
    MarketKind,
    WSOL_MINT,
)
from l25x.pinning import (  # noqa: E402
    EntryVenueIdentity,
    ExitVenueStatus,
    PositionAction,
    entry_identity_from_selection,
    resolve_exit_venue,
    venue_mismatch,
)
from l25x.venues import (  # noqa: E402
    VenueIdentity,
    VenueQuoteResult,
    VenueSelection,
    select_best_venue,
)

CURVE = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
AMM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"


def venue(addr: str, kind: MarketKind, program: str) -> VenueIdentity:
    return VenueIdentity(
        venue_id=f"{kind.value}:{addr}", market_type=kind, pool_address=addr,
        program_id=program, dex_id=kind.value, quote_asset_mint=WSOL_MINT,
        identity_state=CheckState.PASS,
    )


def q(v: VenueIdentity, price: float, ok: bool = True,
      failure: FailureMode | None = None) -> VenueQuoteResult:
    return VenueQuoteResult(venue=v, ok=ok, output_amount=1_000.0,
                            effective_price_usd=price, quote_source="stub",
                            failure=failure)


CURVE_VENUE = venue("Curve1", MarketKind.PUMP_BONDING_CURVE, CURVE)
AMM_VENUE = venue("Pool1", MarketKind.PUMPSWAP, AMM)


# -- 5. entry venue identity persistence ----------------------------------


def test_entry_identity_distinguishes_bonding_curve_from_amm():
    """
    5. Identity is the program and market type, never a launchpad label. This is
    the GOAT defect: both legs were recorded as "Pump.fun".
    """
    a = entry_identity_from_selection(VenueSelection(
        candidates=(CURVE_VENUE, AMM_VENUE),
        results=(q(CURVE_VENUE, 1e-5),), selected=q(CURVE_VENUE, 1e-5),
        reason="sole", ambiguous=False, proven_best=True, exhaustive=True,
        n_candidate_venues=1, n_quoted_ok=1))
    b = entry_identity_from_selection(VenueSelection(
        candidates=(AMM_VENUE,), results=(q(AMM_VENUE, 1e-5),),
        selected=q(AMM_VENUE, 1e-5), reason="sole", ambiguous=False,
        proven_best=True, exhaustive=True, n_candidate_venues=1, n_quoted_ok=1))
    assert a.venue.market_type is MarketKind.PUMP_BONDING_CURVE
    assert b.venue.market_type is MarketKind.PUMPSWAP
    assert a.venue.venue_id != b.venue.venue_id
    d = a.as_dict()
    assert d["entry_venue_program_id"] == CURVE
    assert d["entry_market_type"] == "PUMP_BONDING_CURVE"


def test_entry_identity_serialises_every_required_field():
    ident = entry_identity_from_selection(VenueSelection(
        candidates=(AMM_VENUE,), results=(q(AMM_VENUE, 1e-5),),
        selected=q(AMM_VENUE, 1e-5), reason="sole", ambiguous=False,
        proven_best=True, exhaustive=True, n_candidate_venues=1, n_quoted_ok=1))
    d = ident.as_dict()
    for k in ("entry_venue_id", "entry_pool_address", "entry_market_type",
              "entry_quote_source", "entry_quote_timestamp",
              "selection_reason", "proven_best", "exhaustive"):
        assert k in d, k


def test_venue_mismatch_detects_a_cross_venue_exit():
    ident = entry_identity_from_selection(VenueSelection(
        candidates=(CURVE_VENUE,), results=(q(CURVE_VENUE, 1e-5),),
        selected=q(CURVE_VENUE, 1e-5), reason="sole", ambiguous=False,
        proven_best=True, exhaustive=True, n_candidate_venues=1, n_quoted_ok=1))
    assert venue_mismatch(ident, ident.venue.venue_id) == "MATCH"
    assert venue_mismatch(ident, AMM_VENUE.venue_id) == "MISMATCH"
    # unknown is unknown, never reported as a match
    assert venue_mismatch(None, AMM_VENUE.venue_id) is None
    assert venue_mismatch(ident, None) is None


# -- 6. pinned exit venue --------------------------------------------------


def test_exit_uses_the_pinned_entry_venue():
    """6. A quotable pinned venue closes the position."""
    ident = entry_identity_from_selection(VenueSelection(
        candidates=(CURVE_VENUE,), results=(q(CURVE_VENUE, 1e-5),),
        selected=q(CURVE_VENUE, 1e-5), reason="sole", ambiguous=False,
        proven_best=True, exhaustive=True, n_candidate_venues=1, n_quoted_ok=1))
    out = resolve_exit_venue(ident, pinned_result=q(CURVE_VENUE, 3.0e-5))
    assert out.status is ExitVenueStatus.PINNED_VENUE_QUOTED
    assert out.action is PositionAction.CLOSE_ON_PINNED_VENUE
    assert out.canonical_quote.venue.pool_address == "Curve1"
    assert out.as_dict()["mark_fallback_used"] is False


def test_counterfactual_better_venue_is_recorded_but_never_settled_on():
    """
    The opportunity cost of pinning stays visible without being settled.
    """
    ident = entry_identity_from_selection(VenueSelection(
        candidates=(CURVE_VENUE,), results=(q(CURVE_VENUE, 1e-5),),
        selected=q(CURVE_VENUE, 1e-5), reason="sole", ambiguous=False,
        proven_best=True, exhaustive=True, n_candidate_venues=1, n_quoted_ok=1))
    other = q(AMM_VENUE, 1.0e-5)          # would have been 3x better
    out = resolve_exit_venue(ident, pinned_result=q(CURVE_VENUE, 3.0e-5),
                             other_results=[other])
    assert out.status is ExitVenueStatus.PINNED_VENUE_QUOTED
    assert out.canonical_quote.venue.pool_address == "Curve1"
    assert out.counterfactual_best_venue_id == AMM_VENUE.venue_id
    assert out.would_have_been_better_elsewhere_pct == pytest_approx(200.0)
    assert any("counterfactual" in n for n in out.notes)


def pytest_approx(x, rel=1e-6):
    import pytest
    return pytest.approx(x, rel=rel)


# -- 7. exit venue unavailable --------------------------------------------


def test_exit_venue_unavailable_leaves_position_open():
    """7. No fallback, no fabricated fill: the position stays OPEN."""
    ident = entry_identity_from_selection(VenueSelection(
        candidates=(CURVE_VENUE,), results=(q(CURVE_VENUE, 1e-5),),
        selected=q(CURVE_VENUE, 1e-5), reason="sole", ambiguous=False,
        proven_best=True, exhaustive=True, n_candidate_venues=1, n_quoted_ok=1))
    failed = VenueQuoteResult(venue=CURVE_VENUE, ok=False,
                              failure=FailureMode.QUOTE_UNAVAILABLE,
                              quote_source="stub", detail="pool gone")
    out = resolve_exit_venue(ident, pinned_result=failed,
                             other_results=[q(AMM_VENUE, 1e-5)])
    assert out.status is ExitVenueStatus.EXIT_VENUE_UNAVAILABLE
    assert out.action is PositionAction.HOLD_OPEN
    assert out.canonical_quote is None
    assert out.retryable is True


def test_exit_with_no_pinned_result_holds_open():
    ident = entry_identity_from_selection(VenueSelection(
        candidates=(AMM_VENUE,), results=(q(AMM_VENUE, 1e-5),),
        selected=q(AMM_VENUE, 1e-5), reason="sole", ambiguous=False,
        proven_best=True, exhaustive=True, n_candidate_venues=1, n_quoted_ok=1))
    out = resolve_exit_venue(ident, pinned_result=None)
    assert out.status is ExitVenueStatus.EXIT_VENUE_UNAVAILABLE
    assert out.action is PositionAction.HOLD_OPEN


def test_missing_entry_identity_refuses_rather_than_inferring():
    out = resolve_exit_venue(None, pinned_result=q(AMM_VENUE, 1e-5))
    assert out.status is ExitVenueStatus.NO_ENTRY_VENUE_RECORDED
    assert out.action is PositionAction.HOLD_OPEN
    assert out.canonical_quote is None


# -- 8. no mark fallback ---------------------------------------------------


def test_no_mark_fallback_anywhere_in_the_exit_path():
    """
    8. Across every failure mode the exit never produces a canonical quote from
    anything other than a successful venue quote.
    """
    ident = entry_identity_from_selection(VenueSelection(
        candidates=(CURVE_VENUE,), results=(q(CURVE_VENUE, 1e-5),),
        selected=q(CURVE_VENUE, 1e-5), reason="sole", ambiguous=False,
        proven_best=True, exhaustive=True, n_candidate_venues=1, n_quoted_ok=1))
    cases = [
        resolve_exit_venue(None, pinned_result=q(AMM_VENUE, 1e-5)),
        resolve_exit_venue(ident, pinned_result=None),
        resolve_exit_venue(ident, pinned_result=VenueQuoteResult(
            venue=CURVE_VENUE, ok=False, failure=FailureMode.QUOTE_UNAVAILABLE)),
    ]
    for out in cases:
        assert out.canonical_quote is None
        assert out.action is PositionAction.HOLD_OPEN
        assert out.as_dict()["mark_fallback_used"] is False


def test_failed_quote_carries_no_price_to_accidentally_use():
    failed = VenueQuoteResult(venue=CURVE_VENUE, ok=False,
                              failure=FailureMode.QUOTE_UNAVAILABLE)
    assert failed.effective_price_usd is None
    assert failed.output_amount is None