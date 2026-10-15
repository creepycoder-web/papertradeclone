"""
DexScreener pair selection (test item 3) and observed-mark parsing (item 16).

The property under test is that selection is a RECORDED DECISION with an
explicit rule, never a silent ``max(liquidity)``. Several of these cases are
specifically the ones where blind max-liquidity would produce a confident wrong
answer.
"""


from __future__ import annotations

import pytest

from autopaper.same_market.dexscreener import (
    DexScreenerProbe,
    ProviderRateLimited,
    normalise_venue,
    observed_mark_from,
    parse_pairs,
    select_pair_from,
)
from autopaper.same_market.models import FailureMode

pytestmark = pytest.mark.same_market

WSOL = "So11111111111111111111111111111111111111112"
MINT = "H9XQrbw5MANzYVgHjXJbScFxWqyGw9CCAec5XDTgpump"
OTHER = "6uPEdkU2xPg1iyfLzV1ssUwvve8nndA2sd6nBFrb1n58"


def _pair(
    *,
    address: str = "pool-1",
    chain: str = "solana",
    dex: str = "pumpfun",
    base: str = MINT,
    quote: str = WSOL,
    price: float | None = 1.783e-05,
    liq: float | None = 20_000.0,
    **kw,
):
    return {
        "chainId": chain,
        "dexId": dex,
        "pairAddress": address,
        "baseToken": {"address": base, "symbol": "GOAT"},
        "quoteToken": {"address": quote, "symbol": "SOL"},
        "priceUsd": price,
        "priceNative": 0.0001,
        "liquidity": {"usd": liq},
        "volume": {"h24": 50_000.0},
        "fdv": 100_000.0,
        "pairCreatedAt": 1_700_000_000_000,
        "txns": {"m5": {"count": 90, "buys": 60, "sells": 30},
                 "h1": {"count": 400, "buys": 250, "sells": 150}},
        **kw,
    }


# ---------------------------------------------------------------------------
# 3. pair selection
# ---------------------------------------------------------------------------

def test_a_single_candidate_is_selected_by_elimination():
    sel = select_pair_from(parse_pairs({"pairs": [_pair()]}, MINT), mint=MINT)
    assert sel.rule == "sole_candidate"
    assert sel.pair.pair_address == "pool-1"
    assert sel.ambiguous is False


def test_multiple_pools_are_ambiguous_and_the_deepest_is_only_diagnostic():
    """
    The core anti-guessing case.

    Blind ``max(pairs, key=liquidity)`` returns ``pool-1`` here and reports a
    confident number. The correct behaviour is to refuse to identify the market
    and label the deepest pool as a diagnostic only.
    """
    pairs = parse_pairs({"pairs": [
        _pair(address="pool-1", dex="pumpfun", liq=50_000.0),
        _pair(address="pool-2", dex="raydium", liq=30_000.0),
        _pair(address="pool-3", dex="meteora", liq=10_000.0),
    ]}, MINT)
    sel = select_pair_from(pairs, mint=MINT)

    assert sel.ambiguous is True
    assert sel.rule == "ambiguous_multiple_pools"
    assert sel.failure is FailureMode.PAIR_AMBIGUOUS
    # The deepest is still available to display, but only as a diagnostic.
    assert sel.diagnostic_pair.pair_address == "pool-1"
    assert "DIAGNOSTIC" in sel.reason


def test_a_venue_named_by_the_signal_disambiguates():
    pairs = parse_pairs({"pairs": [
        _pair(address="pool-1", dex="pumpfun", liq=50_000.0),
        _pair(address="pool-2", dex="raydium", liq=30_000.0),
    ]}, MINT)
    sel = select_pair_from(pairs, mint=MINT, venue_hint="Raydium")
    assert sel.rule == "signal_named_venue"
    assert sel.pair.pair_address == "pool-2"
    assert sel.ambiguous is False


def test_a_venue_hint_matching_nothing_is_a_reported_disagreement():
    """
    Not silently ignored. Substituting a different venue for the one the
    signal named would measure a market the signal did not refer to.
    """
    pairs = parse_pairs({"pairs": [_pair(dex="pumpfun")]}, MINT)
    sel = select_pair_from(pairs, mint=MINT, venue_hint="Orca")
    assert sel.rule == "venue_hint_unmatched"
    assert sel.failure is FailureMode.VENUE_MISMATCH
    assert sel.pair is None


def test_a_venue_hint_matching_several_pools_stays_ambiguous():
    pairs = parse_pairs({"pairs": [
        _pair(address="pool-1", dex="pumpswap", liq=50_000.0),
        _pair(address="pool-2", dex="pumpswap", liq=30_000.0),
    ]}, MINT)
    sel = select_pair_from(pairs, mint=MINT, venue_hint="pumpswap")
    assert sel.ambiguous is True
    assert sel.failure is FailureMode.PAIR_AMBIGUOUS
    assert "assumption" in sel.reason


def test_a_token_quoted_as_the_quote_asset_is_rejected():
    """
    Its priceUsd there is the RECIPROCAL of the price we need. Comparing it to
    a USD mark would produce a gap of a few hundred percent that reads like an
    extreme market rather than a wrong pair.
    """
    pairs = parse_pairs({"pairs": [
        _pair(address="reversed", base=WSOL, quote=MINT, price=56120.0),
    ]}, MINT)
    sel = select_pair_from(pairs, mint=MINT)
    assert sel.rule == "token_not_base"
    assert sel.failure is FailureMode.TOKEN_MINT_MISMATCH


def test_a_non_solana_pair_is_not_a_candidate():
    pairs = parse_pairs({"pairs": [_pair(chain="ethereum", dex="uniswap")]}, MINT)
    sel = select_pair_from(pairs, mint=MINT)
    assert sel.rule == "no_solana_pair"
    assert sel.failure is FailureMode.PAIR_NOT_FOUND


def test_pairs_with_no_price_yield_no_mark():
    pairs = parse_pairs({"pairs": [_pair(price=None, liq=5_000.0)]}, MINT)
    sel = select_pair_from(pairs, mint=MINT)
    assert sel.rule == "no_price"
    assert sel.pair is None


def test_no_pairs_at_all_is_pair_not_found():
    sel = select_pair_from([], mint=MINT)
    assert sel.failure is FailureMode.PAIR_NOT_FOUND
    assert sel.pair is None


def test_the_selection_reason_is_always_populated():
    """The reason is the deliverable. An empty string would fail the brief."""
    for pairs in (
        [], [_pair()],
        [_pair(address="a", liq=1.0), _pair(address="b", liq=2.0)],
        [_pair(chain="ethereum")],
    ):
        sel = select_pair_from(
            parse_pairs({"pairs": pairs}, MINT), mint=MINT
        )
        assert sel.reason.strip()
        assert sel.rule.strip()


def test_ambiguity_can_be_forced_only_for_the_failure_path():
    pairs = parse_pairs({"pairs": [
        _pair(address="pool-1", liq=50_000.0),
        _pair(address="pool-2", dex="raydium", liq=30_000.0),
    ]}, MINT)
    forced = select_pair_from(pairs, mint=MINT, allow_ambiguous=True)
    assert forced.pair is not None
    assert forced.ambiguous is True
    assert "failure-path test only" in forced.reason


def test_the_client_reports_a_rate_limit_as_a_named_failure():
    def _limited(url, timeout):
        raise ProviderRateLimited("429")

    probe = DexScreenerProbe(http_get=_limited)
    sel = probe.select_pair(MINT)
    assert sel.failure is FailureMode.PROVIDER_RATE_LIMIT
    assert sel.pair is None


# ---------------------------------------------------------------------------
# 16. observed mark parsing
# ---------------------------------------------------------------------------

def test_every_documented_pair_field_is_parsed():
    pair = parse_pairs({"pairs": [_pair()]}, MINT)[0]
    assert pair.chain_id == "solana"
    assert pair.dex_id == "pumpfun"
    assert pair.base_token_address == MINT
    assert pair.quote_token_address == WSOL
    assert pair.price_usd == pytest.approx(1.783e-05)
    assert pair.price_native == pytest.approx(0.0001)
    assert pair.liquidity_usd == pytest.approx(20_000.0)
    assert pair.volume_24h_usd == pytest.approx(50_000.0)
    assert pair.fdv_usd == pytest.approx(100_000.0)
    assert pair.pair_created_at == 1_700_000_000_000
    assert pair.txns_5m == 90
    assert pair.buys_5m == 60
    assert pair.sells_5m == 30


def test_a_malformed_payload_never_raises():
    for payload in ({}, {"pairs": None}, {"pairs": "nope"}, {"pairs": [None, 1]}):
        assert parse_pairs(payload, MINT) == []


def test_a_pair_without_an_address_is_unusable():
    assert parse_pairs({"pairs": [_pair(address="")]}, MINT) == []


def test_a_failed_selection_produces_no_mark_value():
    sel = select_pair_from(
        parse_pairs({"pairs": [_pair(chain="ethereum")]}, MINT), mint=MINT
    )
    mark = observed_mark_from(sel)
    assert mark.value_usd is None
    assert mark.failure is FailureMode.PAIR_NOT_FOUND


def test_a_successful_selection_produces_a_mark_tied_to_its_pair():
    sel = select_pair_from(parse_pairs({"pairs": [_pair()]}, MINT), mint=MINT)
    mark = observed_mark_from(sel, latency_ms=12.5)
    assert mark.value_usd == pytest.approx(1.783e-05)
    assert mark.pair_address == "pool-1"
    assert mark.request_latency_ms == 12.5


def test_venue_names_are_normalised_across_providers():
    assert normalise_venue("pumpfun") == "pumpfun"
    assert normalise_venue("PumpFun") == "pumpfun"
    assert normalise_venue("PUMP") == "pumpfun"
    assert normalise_venue("pump-amm") == "pumpswap"
    assert normalise_venue("somethingnew") == "somethingnew"
    assert normalise_venue(None) is None


def test_the_client_caches_so_identity_and_mark_share_one_request():
    calls: list[str] = []

    def _ok(url, timeout):
        calls.append(url)
        return {"pairs": [_pair()]}

    probe = DexScreenerProbe(http_get=_ok, cache_ttl_s=30.0)
    first = probe.select_pair(MINT)
    second = probe.select_pair(MINT)

    assert first.pair is not None and second.pair is not None
    assert len(calls) == 1, "the second selection should have been served from cache"
    assert probe.last_from_cache is True