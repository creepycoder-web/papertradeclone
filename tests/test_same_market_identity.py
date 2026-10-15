"""
Same-market verification (test items 4, 5, 6, 7 and 22).

These are the tests that matter most. Everything else measures a number; these
decide whether the number is allowed to mean anything. The recurring assertion
is that ``UNKNOWN`` never becomes ``PASS`` - an identity that could not be
proven is the finding, not an obstacle to it.
"""


from __future__ import annotations

import pytest

from autopaper.same_market.dexscreener import parse_pairs, select_pair_from
from autopaper.same_market.identity import (
    compare_mark_to_executable,
    verify_identity,
    verify_pool_identity,
)
from autopaper.same_market.jupiter import JupiterQuoteResult, RouteHop
from autopaper.same_market.models import (
    WSOL_MINT,
    CheckState,
    Decision,
    ExecutableQuote,
    FailureMode,
    IdentityChecks,
    IdentityChecks as IC,
    StalenessPolicy,
)

pytestmark = pytest.mark.same_market

MARK = 1.783e-05
PAIR_A = "pool-AAAA1111"
PAIR_B = "pool-BBBB2222"
MINT = "H9XQrbw5MANzYVgHjXJbScFxWqyGw9CCAec5XDTgpump"


def _pair(address: str = PAIR_A, *, dex: str = "raydium", base: str = MINT,
          quote: str = WSOL_MINT):
    return {
        "chainId": "solana", "dexId": dex, "pairAddress": address,
        "baseToken": {"address": base, "symbol": "GOAT"},
        "quoteToken": {"address": quote, "symbol": "SOL"},
        "priceUsd": MARK, "liquidity": {"usd": 30_000.0},
    }


def _pair_obj(address: str = PAIR_A, *, dex: str = "raydium"):
    return parse_pairs({"pairs": [_pair(address, dex=dex)]}, MINT)[0]


def _quote(**kw) -> ExecutableQuote:
    base = dict(
        input_mint=WSOL_MINT, output_mint=MINT,
        input_amount_raw="1150000", output_amount_raw="64523000",
        input_decimals=9, output_decimals=6,
        input_amount=0.00115, output_amount=64.523,
        effective_price=1.7823e-05, effective_price_native=1.7823e-05,
        price_impact_pct=0.4, fee_bps=25, fee_amount=0.000002875,
        venue="raydium", pool_address=PAIR_A, route=("Raydium",),
        provider="jupiter", state_timestamp=None,
        quote_timestamp=1000.0, quote_latency_ms=40.0,
    )
    base.update(kw)
    return ExecutableQuote(**base)


def _jupiter(amm_key: str | None, hops: int = 1) -> JupiterQuoteResult:
    plan = tuple(
        RouteHop(label="Raydium", amm_key=amm_key, input_mint=WSOL_MINT,
                 output_mint=MINT, in_amount=1150000, out_amount=64523000,
                 fee_amount=0, percent=100.0)
        for _ in range(hops)
    )
    return JupiterQuoteResult(
        input_mint=WSOL_MINT, output_mint=MINT, in_amount=1150000,
        out_amount=64523000, other_amount_threshold=64000000,
        price_impact_pct=0.4, route_plan=plan, timestamp=1000.0,
    )


def _all_pass() -> IdentityChecks:
    return IC(
        token_mint_match=CheckState.PASS, quote_mint_match=CheckState.PASS,
        chain_match=CheckState.PASS, venue_match=CheckState.PASS,
        pool_match=CheckState.PASS, execution_can_identify_market=True,
    )


# ---------------------------------------------------------------------------
# 4. token mint verification
# ---------------------------------------------------------------------------

def test_a_quote_for_the_signal_token_passes():
    checks = verify_identity(expected_mint=MINT, pair=_pair_obj(),
                             quote=_quote())
    assert checks.token_mint_match is CheckState.PASS


def test_a_quote_for_a_different_token_fails():
    checks = verify_identity(
        expected_mint=MINT, pair=_pair_obj(),
        quote=_quote(output_mint="SomeOtherMint1111111111111111111111"),
    )
    assert checks.token_mint_match is CheckState.FAIL
    assert checks.market_match is Decision.REJECT


def test_no_quote_leaves_the_token_mint_unknown_not_passed():
    checks = verify_identity(expected_mint=MINT, pair=_pair_obj(), quote=None)
    assert checks.token_mint_match is CheckState.UNKNOWN
    assert checks.market_match is Decision.REJECT


# ---------------------------------------------------------------------------
# 5. quote mint verification
# ---------------------------------------------------------------------------

def test_wsol_as_the_input_asset_passes():
    checks = verify_identity(expected_mint=MINT, pair=_pair_obj(),
                             quote=_quote(input_mint=WSOL_MINT))
    assert checks.quote_mint_match is CheckState.PASS


def test_a_liquid_stable_quote_asset_passes_and_is_flagged():
    usdc = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    checks = verify_identity(expected_mint=MINT, pair=_pair_obj(),
                             quote=_quote(input_mint=usdc))
    assert checks.quote_mint_match is CheckState.PASS
    assert any("rather than the configured input asset" in n for n in checks.notes)


def test_an_exotic_quote_asset_fails():
    checks = verify_identity(
        expected_mint=MINT, pair=_pair_obj(),
        quote=_quote(input_mint="SomeRandomToken11111111111111111111"),
    )
    assert checks.quote_mint_match is CheckState.FAIL


# ---------------------------------------------------------------------------
# 6. venue verification
# ---------------------------------------------------------------------------

def test_matching_venue_names_across_providers_pass():
    """DexScreener says 'raydium', the route says 'Raydium'. Same venue."""
    checks = verify_identity(expected_mint=MINT, pair=_pair_obj(dex="raydium"),
                             quote=_quote(route=("Raydium",)))
    assert checks.venue_match is CheckState.PASS


def test_different_venues_fail():
    checks = verify_identity(expected_mint=MINT, pair=_pair_obj(dex="raydium"),
                             quote=_quote(route=("Meteora",)))
    assert checks.venue_match is CheckState.FAIL
    assert any("not the same book" in n for n in checks.notes)


def test_pumpfun_and_pumpswap_are_treated_as_equivalent_venues():
    """
    A token that graduated keeps a 'pumpfun' DexScreener label for a while
    while the route uses PumpSwap. Failing that pair as a venue mismatch would
    be a false alarm on exactly the case this harness cares about.
    """
    checks = verify_identity(expected_mint=MINT, pair=_pair_obj(dex="pumpfun"),
                             quote=_quote(route=("PumpSwap",)))
    assert checks.venue_match is CheckState.PASS


def test_a_quote_with_no_route_leaves_the_venue_unknown():
    checks = verify_identity(expected_mint=MINT, pair=_pair_obj(),
                             quote=_quote(route=()))
    assert checks.venue_match is CheckState.UNKNOWN
    assert checks.market_match is Decision.REJECT


# ---------------------------------------------------------------------------
# 7. pool identity verification - the load-bearing check
# ---------------------------------------------------------------------------

def test_a_matching_pool_key_is_the_only_way_to_pass():
    state, notes = verify_pool_identity(pair=_pair_obj(PAIR_A),
                                        quote=_quote(pool_address=PAIR_A),
                                        jupiter_result=_jupiter(PAIR_A))
    assert state is CheckState.PASS
    assert any("same pool" in n for n in notes)


def test_the_route_pool_must_equal_the_marked_pool():
    state, notes = verify_pool_identity(pair=_pair_obj(PAIR_A),
                                        quote=_quote(pool_address=PAIR_B),
                                        jupiter_result=_jupiter(PAIR_B))
    assert state is CheckState.FAIL
    assert any("DIFFERENT" in n for n in notes)


def test_no_disclosed_pool_key_is_unknown_and_never_pass():
    state, notes = verify_pool_identity(pair=_pair_obj(PAIR_A),
                                        quote=_quote(pool_address=None),
                                        jupiter_result=_jupiter(None))
    assert state is CheckState.UNKNOWN
    assert any("UNKNOWN" in n for n in notes)


def test_a_multihop_route_has_no_single_pool_and_is_unknown():
    state, notes = verify_pool_identity(pair=_pair_obj(PAIR_A),
                                        quote=_quote(pool_address=PAIR_A),
                                        jupiter_result=_jupiter(PAIR_A, hops=2))
    assert state is CheckState.UNKNOWN
    assert any("hops" in n for n in notes)


def test_no_observed_pair_means_unknown():
    state, notes = verify_pool_identity(pair=None, quote=_quote(),
                                        jupiter_result=None)
    assert state is CheckState.UNKNOWN


def test_the_provider_can_only_identify_a_market_it_discloses_a_pool_for():
    checks = verify_identity(expected_mint=MINT, pair=_pair_obj(),
                             quote=_quote(route=("Raydium",), pool_address=None),
                             jupiter_result=_jupiter(None))
    assert checks.execution_can_identify_market is False
    assert checks.market_match is Decision.REJECT


def test_a_failed_quote_cannot_identify_a_market():
    checks = verify_identity(
        expected_mint=MINT, pair=_pair_obj(),
        quote=_quote(failure=FailureMode.QUOTE_UNAVAILABLE),
    )
    assert checks.execution_can_identify_market is False


# ---------------------------------------------------------------------------
# 22. ambiguous market rejection, and the verdict
# ---------------------------------------------------------------------------

def test_ambiguous_pairs_are_rejected_even_when_prices_agree_exactly():
    """The trap: identical prices are not evidence of the same market."""
    sel = select_pair_from(parse_pairs({"pairs": [
        _pair("pool-1", dex="pumpfun"), _pair("pool-2", dex="raydium"),
    ]}, MINT), mint=MINT)
    assert sel.ambiguous is True

    verdict = compare_mark_to_executable(
        observed_mark=MARK, executable_price=MARK,   # gap is exactly 0
        threshold_pct=5.0, failures=[sel.failure],
    )
    assert verdict.gap_pct == pytest.approx(0.0)
    assert verdict.decision is Decision.REJECT


def test_unproven_identity_rejects_even_with_a_tiny_gap():
    checks = IC(CheckState.PASS, CheckState.PASS, CheckState.PASS,
                CheckState.PASS, CheckState.UNKNOWN, False)
    verdict = compare_mark_to_executable(
        observed_mark=MARK, executable_price=MARK * 1.0001,
        threshold_pct=5.0, identity=checks,
    )
    assert verdict.decision is Decision.REJECT
    assert "market identity is not proven" in verdict.reason


def test_no_threshold_yields_informative_never_pass():
    verdict = compare_mark_to_executable(
        observed_mark=MARK, executable_price=MARK, threshold_pct=None,
        identity=_all_pass(),
    )
    assert verdict.decision is Decision.INFORMATIVE
    assert verdict.threshold_pct is None
    assert "NO pass/fail verdict is claimed" in verdict.reason


def test_a_missing_measurement_is_reject_even_without_a_threshold():
    for mark, quote in ((None, MARK), (MARK, None), (0.0, MARK), (MARK, 0.0)):
        verdict = compare_mark_to_executable(
            observed_mark=mark, executable_price=quote, threshold_pct=None,
            identity=_all_pass(),
        )
        assert verdict.decision is Decision.REJECT, (mark, quote)


def test_a_proven_small_gap_passes_against_a_configured_threshold():
    verdict = compare_mark_to_executable(
        observed_mark=1.783e-05, executable_price=1.810e-05,
        threshold_pct=5.0, identity=_all_pass(),
    )
    assert verdict.decision is Decision.PASS
    assert verdict.gap_pct == pytest.approx(1.5148, abs=0.001)
    assert verdict.abs_gap_pct == pytest.approx(1.5148, abs=0.001)


def test_a_proven_large_gap_is_rejected():
    verdict = compare_mark_to_executable(
        observed_mark=1.783e-05, executable_price=3.108e-05,
        threshold_pct=5.0, identity=_all_pass(),
    )
    assert verdict.decision is Decision.REJECT
    assert verdict.gap_pct == pytest.approx(74.31, abs=0.1)
    assert "worse than" in verdict.reason


def test_the_threshold_is_compared_against_the_absolute_gap():
    """A quote 50% BETTER than the mark is still 50% away from it."""
    better = compare_mark_to_executable(
        observed_mark=1.783e-05, executable_price=1.783e-05 / 1.5,
        threshold_pct=5.0, identity=_all_pass(),
    )
    assert better.gap_pct < 0
    assert better.abs_gap_pct == pytest.approx(33.33, abs=0.01)
    assert better.decision is Decision.REJECT


def test_any_single_failed_check_rejects_the_whole_identity():
    for field in ("token_mint_match", "quote_mint_match", "chain_match",
                  "venue_match", "pool_match"):
        kwargs = dict(token_mint_match=CheckState.PASS,
                      quote_mint_match=CheckState.PASS,
                      chain_match=CheckState.PASS,
                      venue_match=CheckState.PASS,
                      pool_match=CheckState.PASS,
                      execution_can_identify_market=True)
        kwargs[field] = CheckState.FAIL
        assert IdentityChecks(**kwargs).market_match is Decision.REJECT


def test_any_single_unknown_check_rejects_the_whole_identity():
    for field in ("token_mint_match", "quote_mint_match", "chain_match",
                  "venue_match", "pool_match"):
        kwargs = dict(token_mint_match=CheckState.PASS,
                      quote_mint_match=CheckState.PASS,
                      chain_match=CheckState.PASS,
                      venue_match=CheckState.PASS,
                      pool_match=CheckState.PASS,
                      execution_can_identify_market=True)
        kwargs[field] = CheckState.UNKNOWN
        assert IdentityChecks(**kwargs).market_match is Decision.REJECT


def test_checks_serialise_with_the_exact_labels_the_brief_names():
    payload = verify_identity(expected_mint=MINT, pair=_pair_obj(),
                              quote=_quote(), jupiter_result=_jupiter(PAIR_A)
                              ).as_dict()
    for key in ("TOKEN_MINT_MATCH", "QUOTE_MINT_MATCH", "CHAIN_MATCH",
                "VENUE_MATCH", "POOL_MATCH", "MARKET_MATCH"):
        assert key in payload
    assert payload["POOL_MATCH"] == "PASS"