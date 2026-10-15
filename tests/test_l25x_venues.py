"""
Offline tests for venue enumeration and best-executable-venue selection.

RESEARCH ONLY. Every quote here is a stub: no network, no RPC, no venue client.
Covers required tests 1-4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

AI = Path(__file__).resolve().parents[1]
for _p in (AI, AI / "research"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from autopaper.same_market.models import (  # noqa: E402
    CheckState,
    DexPair,
    FailureMode,
    MarketKind,
    WSOL_MINT,
)
from l25x.venues import (  # noqa: E402
    VenueIdentity,
    VenueQuoteResult,
    enumerate_venues,
    select_best_venue,
)

MINT = "A5MdMEt99PzbYKsfPaQ5Ee1jFZEsYu3zBvdpDdyX2sXQ"


def pair(dex_id: str, address: str, price: float | None,
         liq: float | None = 1000.0, base: str = MINT) -> DexPair:
    return DexPair(
        chain_id="solana", dex_id=dex_id, pair_address=address,
        base_token_address=base, base_token_symbol="AI",
        quote_token_address=WSOL_MINT, quote_token_symbol="SOL",
        price_usd=price, price_native=price, liquidity_usd=liq,
        volume_24h_usd=None, fdv_usd=None, market_cap_usd=None,
        pair_created_at=None, txns_5m=None, txns_1h=None,
        buys_5m=None, sells_5m=None, buys_1h=None, sells_1h=None,
    )


def venue(addr: str, kind: MarketKind = MarketKind.PUMPSWAP,
          state: CheckState = CheckState.PASS) -> VenueIdentity:
    return VenueIdentity(
        venue_id=f"{kind.value}:{addr}", market_type=kind, pool_address=addr,
        program_id="pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA",
        dex_id="pumpswap", quote_asset_mint=WSOL_MINT, identity_state=state,
    )


def q(v: VenueIdentity, out: float, price: float, ok: bool = True,
      failure: FailureMode | None = None) -> VenueQuoteResult:
    return VenueQuoteResult(
        venue=v, ok=ok, input_amount_raw="1000000", output_amount=out,
        effective_price_usd=price, quote_source="stub",
        failure=failure,
    )


# -- 1. multiple venue discovery ------------------------------------------


def test_multiple_venue_discovery():
    """1. A token trading on two pools yields two venues, typed distinctly."""
    pairs = [
        pair("pumpswap", "PoolAAAA", 4.012e-06, 3723.0),
        pair("pumpfun", "CurveBBBB", 4.990e-05, None),
        pair("raydium", "PoolCCCC", 9.9e-05, 50000.0),
    ]
    venues = enumerate_venues(MINT, pairs)
    assert len(venues) == 3
    kinds = {v.market_type for v in venues}
    assert MarketKind.PUMP_BONDING_CURVE in kinds
    assert MarketKind.PUMPSWAP in kinds
    # distinct pools must never collapse into one venue
    assert len({v.pool_address for v in venues}) == 3
    assert len({v.venue_id for v in venues}) == 3


def test_venue_discovery_drops_pairs_where_token_is_not_base():
    """A pair whose base side is another token is not a venue for this mint."""
    pairs = [
        pair("pumpswap", "PoolAAAA", 4.0e-06),
        pair("pumpswap", "PoolDDDD", 5.0e-06, base="SomeOtherMint1111111111111111111111111111111"),
    ]
    venues = enumerate_venues(MINT, pairs)
    assert [v.pool_address for v in venues] == ["PoolAAAA"]


def test_derived_pool_is_enumerated_even_when_feed_omits_it():
    """A derived pool the feed does not list is still a candidate market."""
    venues = enumerate_venues(MINT, [], extra_pool_addresses=["Derived999"])
    assert len(venues) == 1
    assert venues[0].discovery_source == "derived_pool"
    # unproven until the chain confirms owner/discriminator
    assert venues[0].identity_state is CheckState.UNKNOWN


# -- 2. best executable venue selection ------------------------------------


def test_best_executable_venue_is_chosen_on_output_not_displayed_price():
    """
    2. Selection ranks on executable output for the real size. The venue with
    the WORSE displayed price wins because it returns more tokens.
    """
    good = venue("PoolGOOD")
    bad = venue("PoolBAD")
    results = [
        q(bad, out=1_000.0, price=0.000100),   # nominally "cheaper"
        q(good, out=5_000.0, price=0.000020),  # pays more tokens out
    ]
    sel = select_best_venue([bad, good], results)
    assert sel.selected is not None
    assert sel.selected.venue.pool_address == "PoolGOOD"
    assert sel.proven_best is True
    assert sel.exhaustive is True


def test_venue_spread_reported_between_best_and_worst():
    v1, v2 = venue("A"), venue("B")
    res = [q(v1, 5_000.0, 0.000020), q(v2, 1_000.0, 0.000100)]
    sel = select_best_venue([v1, v2], res)
    assert sel.venue_spread_ratio == pytest.approx(5.0, rel=1e-6)
    assert sel.best_vs_worst_gap_pct == pytest.approx(-80.0, rel=1e-6)


def test_liquidity_only_venue_is_not_preferred_over_executable_result():
    """
    The CURRENT rule in same_market is max(liquidity). Selection must ignore it.
    """
    # liquidity is deliberately NOT part of VenueIdentity: the selection layer
    # has no liquidity input at all, so it cannot prefer on it.
    deep = venue("DeepButExpensive")
    shallow = venue("ShallowButCheap")
    res = [q(deep, 1_000.0, 0.000100), q(shallow, 9_000.0, 0.000011)]
    sel = select_best_venue([deep, shallow], res)
    assert sel.selected.venue.pool_address == "ShallowButCheap"


# -- 3. exact-order-size comparison ----------------------------------------


def test_exact_order_size_is_passed_to_every_venue():
    """
    3. Every venue is quoted for the SAME lamport input. A comparison across
    different sizes would be meaningless.
    """
    from l25x.pipeline import plan_entry

    seen: list[int] = []

    def quoter(v: VenueIdentity, lamports: int) -> VenueQuoteResult:
        seen.append(lamports)
        return q(v, 5_000.0, 0.00002)

    from l25x.rent import TokenAccountState
    plan = plan_entry(
        {"mint": MINT, "symbol": "AI", "f_market_cap_usd": 17000.0,
         "f_signal_quality": 0.6, "f_liquidity_usd": 20000.0, "route": "FAST_ENTRY"},
        planned_notional_usd=10.0, sol_price_usd=200.0,
        pairs=[pair("pumpswap", "A", 1e-5), pair("pumpswap", "B", 2e-5)],
        quote_for=quoter,
        token_account=TokenAccountState(address="ATA1", exists=True,
                                        program_id="Tokenkeg"),
        extra_pool_addresses=[],
    )
    assert plan.status.value == "PLANNED"
    # 10 USD at 200 USD/SOL = 0.05 SOL = 50_000_000 lamports, exactly, twice
    assert seen == [50_000_000, 50_000_000]
    assert len(set(seen)) == 1


# -- 4. ambiguous / unverifiable venue rejection ---------------------------


def test_ambiguous_unquotable_venue_does_not_yield_a_proven_best():
    """
    4. If an eligible venue cannot be quoted, selection is NOT proven best and
    carries a failure so a downstream gate fails closed.
    """
    v1, v2 = venue("Quotable"), venue("Broken")
    res = [
        q(v1, 5_000.0, 0.000020),
        VenueQuoteResult(venue=v2, ok=False, failure=FailureMode.QUOTE_UNAVAILABLE,
                         quote_source="stub", detail="pool reserves unreadable"),
    ]
    sel = select_best_venue([v1, v2], res)
    assert sel.selected is not None, "best of the quotable is still reported"
    assert sel.proven_best is False
    assert sel.exhaustive is False
    assert sel.ambiguous is True
    assert sel.failure is FailureMode.QUOTE_UNAVAILABLE
    assert any("Broken" in n for n in sel.notes)


def test_unproven_identity_venue_is_excluded_and_cannot_claim_best():
    """
    An unproven-identity venue is never selected, and its presence means the
    selection is NOT proven best -- because that venue might have been cheaper.
    Here it returns MORE tokens (9,000 vs 5,000) and is still not chosen.
    """
    good, unknown = venue("Good"), venue("Unproven", state=CheckState.UNKNOWN)
    res = [q(good, 5_000.0, 0.00002), q(unknown, 9_000.0, 0.000011)]
    sel = select_best_venue([good, unknown], res)
    # excluded from the ranking entirely, despite the better executable output
    assert sel.selected is not None
    assert sel.selected.venue.pool_address == "Good"
    # ...and "best" is explicitly NOT claimed
    assert sel.proven_best is False
    assert sel.exhaustive is False
    assert sel.failure is FailureMode.QUOTE_UNAVAILABLE
    assert any("Unproven" in n for n in sel.notes)


def test_no_identity_proven_at_all_yields_no_selection():
    """When nothing can be identified, there is no selection and no fallback."""
    only = venue("Unproven", state=CheckState.UNKNOWN)
    sel = select_best_venue([only], [q(only, 9_000.0, 0.000011)])
    assert sel.selected is None
    assert sel.proven_best is False
    assert sel.failure is FailureMode.MARKET_IDENTITY_UNKNOWN


def test_no_venue_refuses_rather_than_falling_back_to_a_mark():
    """Zero usable quotes -> refusal. A mark is never substituted."""
    v = venue("Only")
    res = [VenueQuoteResult(venue=v, ok=False,
                            failure=FailureMode.QUOTE_UNAVAILABLE,
                            quote_source="stub")]
    sel = select_best_venue([v], res)
    assert sel.selected is None
    assert sel.proven_best is False
    assert sel.failure is FailureMode.QUOTE_UNAVAILABLE
    assert any("mark" in n for n in sel.notes)


def test_sole_candidate_is_recorded_as_such():
    """One eligible venue: the comparison is trivially exhaustive."""
    v = venue("Only")
    sel = select_best_venue([v], [q(v, 5_000.0, 0.00002)])
    assert sel.proven_best is True
    assert "sole" in sel.reason.lower()