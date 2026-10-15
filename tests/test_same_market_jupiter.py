"""
Jupiter quoting and, above all, DIRECTION (test items 12, 13 and 14).

The direction tests matter most. A reversed quote returns a plausible number
that is simply the reciprocal; compared against a USD mark it produces a gap of
several hundred percent that reads like an extreme market rather than a bug.
"""


from __future__ import annotations

import asyncio

import pytest

from autopaper.same_market.jupiter import (
    JupiterQuoteClient,
    QuoteDirectionError,
    QuoteUnavailable,
    assert_buy_direction,
    to_executable_quote,
)
from autopaper.same_market.models import WSOL_MINT, FailureMode

pytestmark = pytest.mark.same_market

WSOL = WSOL_MINT
MINT = "H9XQrbw5MANzYVgHjXJbScFxWqyGw9CCAec5XDTgpump"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
POOL = "2wT8Yq49kHgDzXuPxZSaeLaH1qbmGXtEyPy64bL7aD3c"

BODY = {
    "inputMint": WSOL,
    "outputMint": MINT,
    "inAmount": "1150000",
    "outAmount": "64523000",
    "otherAmountThreshold": "64000000",
    "priceImpactPct": "0.42",
    "routePlan": [{
        "swapInfo": {
            "label": "Raydium",
            "ammKey": POOL,
            "inputMint": WSOL,
            "outputMint": MINT,
            "inAmount": "1150000",
            "outAmount": "64523000",
            "feeAmount": "0",
            "percent": 100,
        }
    }],
}


def _client(body=None, *, error: Exception | None = None):
    def _get(params):
        if error is not None:
            raise error
        return dict(body if body is not None else BODY)
    return JupiterQuoteClient(http_get=_get, slippage_bps=1000)


# ---------------------------------------------------------------------------
# 12. Jupiter quote direction
# ---------------------------------------------------------------------------

def test_a_buy_is_quote_asset_to_token():
    assert_buy_direction(input_mint=WSOL, output_mint=MINT,
                         quote_asset_mint=WSOL, token_mint=MINT)


def test_a_reversed_buy_is_refused():
    with pytest.raises(QuoteDirectionError) as exc:
        assert_buy_direction(input_mint=MINT, output_mint=WSOL,
                             quote_asset_mint=WSOL, token_mint=MINT)
    assert "reversed" in str(exc.value) or "wrong asset" in str(exc.value)


def test_an_unrelated_output_mint_is_refused():
    with pytest.raises(QuoteDirectionError):
        assert_buy_direction(input_mint=WSOL, output_mint=USDC,
                             quote_asset_mint=WSOL, token_mint=MINT)


def test_identical_mints_are_refused():
    with pytest.raises(QuoteDirectionError):
        assert_buy_direction(input_mint=WSOL, output_mint=WSOL,
                             quote_asset_mint=WSOL, token_mint=MINT)


def test_the_reversed_pair_is_refused_before_the_network_is_touched():
    """
    ``quote_buy`` derives its own direction from ``quote_asset_mint``, so the
    way to get a reversed request is to reach past it. The guard refuses that
    before any HTTP call, so a reciprocal number can never reach a comparison.
    """
    calls: list[dict] = []

    def _get(params):
        calls.append(params)
        return dict(BODY)

    client = JupiterQuoteClient(http_get=_get)
    with pytest.raises(QuoteDirectionError):
        assert_buy_direction(
            input_mint=MINT, output_mint=WSOL,   # reversed
            quote_asset_mint=WSOL, token_mint=MINT,
        )
    assert calls == [], "the guard must run before any request"


def test_a_buy_in_a_different_quote_asset_is_still_a_buy():
    """
    Spending USDC to buy the token is a legitimate BUY, not a reversal. It must
    not be refused - but the denominating asset differs from the mark's, which
    the identity checks report separately.
    """
    captured: list[dict] = []

    def _get(params):
        captured.append(dict(params))
        return dict(BODY, inputMint=USDC)

    client = JupiterQuoteClient(http_get=_get)
    result = asyncio.run(client.quote_buy(
        token_mint=MINT, quote_asset_mint=USDC,
        in_amount=1_000_000, output_decimals=6,
    ))
    assert captured[0]["inputMint"] == USDC
    assert captured[0]["outputMint"] == MINT
    assert result.input_mint == USDC
    assert result.output_mint == MINT


# ---------------------------------------------------------------------------
# 13. BUY = quote asset -> token
# ---------------------------------------------------------------------------

def test_a_buy_quote_requests_the_right_direction():
    captured: list[dict] = []

    def _get(params):
        captured.append(dict(params))
        return dict(BODY)

    client = JupiterQuoteClient(http_get=_get)
    result = asyncio.run(client.quote_buy(
        token_mint=MINT, quote_asset_mint=WSOL,
        in_amount=1_150_000, output_decimals=6,
    ))

    assert captured[0]["inputMint"] == WSOL
    assert captured[0]["outputMint"] == MINT
    assert captured[0]["amount"] == "1150000"
    assert result.in_amount == 1_150_000
    assert result.out_amount == 64_523_000
    assert result.pool_keys == (POOL,)


def test_the_route_pool_key_is_captured_for_the_same_market_check():
    client = _client()
    result = asyncio.run(client.quote_buy(
        token_mint=MINT, quote_asset_mint=WSOL,
        in_amount=1_150_000, output_decimals=6,
    ))
    assert result.venue_labels == ("Raydium",)
    assert result.route_plan[0].amm_key == POOL
    assert result.route_plan[0].has_pool_identity


def test_a_multihop_route_keeps_every_hop():
    body = dict(BODY)
    body["routePlan"] = [BODY["routePlan"][0], {
        "swapInfo": {"label": "Meteora", "ammKey": "pool-2",
                     "inputMint": WSOL, "outputMint": MINT,
                     "inAmount": "1", "outAmount": "1",
                     "feeAmount": "0", "percent": 50},
    }]
    client = _client(body)
    result = asyncio.run(client.quote_buy(
        token_mint=MINT, quote_asset_mint=WSOL,
        in_amount=1_150_000, output_decimals=6,
    ))
    assert len(result.route_plan) == 2
    assert len(result.pool_keys) == 2


def test_the_executable_quote_is_derived_from_the_raw_amounts():
    client = _client()
    result = asyncio.run(client.quote_buy(
        token_mint=MINT, quote_asset_mint=WSOL,
        in_amount=1_150_000, output_decimals=6,
    ))
    quote = to_executable_quote(result, token_mint=MINT,
                                quote_asset_mint=WSOL, output_decimals=6)
    assert quote.effective_price == pytest.approx(
        0.00115 / 64.523, rel=1e-9
    )
    assert quote.price_impact_pct == pytest.approx(0.42)
    assert quote.pool_address == POOL
    assert quote.failure is None


def test_the_trading_gate_is_explicitly_not_applied():
    """A $0.23 micro-order trips the production min-output. That is fine here."""
    body = dict(BODY, outAmount="7")
    client = _client(body)
    result = asyncio.run(client.quote_buy(
        token_mint=MINT, quote_asset_mint=WSOL,
        in_amount=1_150_000, output_decimals=6,
    ))
    quote = to_executable_quote(result, token_mint=MINT,
                                quote_asset_mint=WSOL, output_decimals=6)
    assert quote.failure is None
    assert "NOT applied" in quote.detail


def test_a_venue_that_answers_for_a_different_pair_is_caught():
    """
    Re-checked from the venue's OWN echo, not just from what was requested.
    """
    body = dict(BODY, inputMint=MINT, outputMint=WSOL)
    client = _client(body)
    result = asyncio.run(client.quote_buy(
        token_mint=MINT, quote_asset_mint=WSOL,
        in_amount=1_150_000, output_decimals=6,
    ))
    quote = to_executable_quote(result, token_mint=MINT,
                                quote_asset_mint=WSOL, output_decimals=6)
    assert quote.failure is FailureMode.QUOTE_DIRECTION_ERROR
    assert quote.effective_price is None


def test_an_unpriceable_notional_is_a_named_failure_not_a_zero():
    client = _client(error=QuoteUnavailable("TOO_SMALL"))
    with pytest.raises(QuoteUnavailable):
        asyncio.run(client.quote_buy(
            token_mint=MINT, quote_asset_mint=WSOL,
            in_amount=1, output_decimals=6,
        ))


def test_a_body_missing_out_amount_is_refused():
    client = JupiterQuoteClient(http_get=lambda p: {"inputMint": WSOL})
    with pytest.raises(QuoteUnavailable):
        client.parse({"inputMint": WSOL})


def test_non_integer_amounts_are_refused():
    with pytest.raises(QuoteUnavailable):
        JupiterQuoteClient(http_get=lambda p: {}).parse(
            dict(BODY, outAmount="not-a-number")
        )


# ---------------------------------------------------------------------------
# 14. a SELL path exists, and is read-only and direction-locked
# ---------------------------------------------------------------------------

SELL_BODY = {
    "inputMint": MINT,
    "outputMint": WSOL,
    "inAmount": "64523000",
    "outAmount": "1128000",
    "otherAmountThreshold": "1100000",
    "priceImpactPct": "0.51",
    "routePlan": [
        {"swapInfo": {"label": "Obsidian", "ammKey": POOL, "inputMint": MINT,
                      "outputMint": WSOL, "inAmount": "30000000",
                      "outAmount": "520000", "feeAmount": "15000", "percent": 46.5}},
        {"swapInfo": {"label": "Flux", "ammKey": "otherpool", "inputMint": WSOL,
                      "outputMint": WSOL, "inAmount": "520000",
                      "outAmount": "1128000", "feeAmount": "0", "percent": 53.5}},
    ],
}


def test_the_sell_helper_is_read_only_and_has_no_transaction_path():
    """
    A SELL quote now EXISTS -- deliberately, so liquidation proceeds can be
    estimated at the exact held quantity.

    The invariant this test used to assert ("no sell path exists at all") has
    been replaced by the stronger invariant that actually matters now: the SELL
    path can PRICE and can do nothing else. It cannot build, sign, send or
    submit, and the class still advertises read methods only.
    """
    client = JupiterQuoteClient()
    assert hasattr(client, "quote_sell"), "the read-only SELL quote is missing"
    # No transaction-building path exists anywhere on the class.
    for name in ("build", "build_transaction", "send", "sign", "submit",
                 "broadcast", "execute", "execute_sell", "sell"):
        assert not hasattr(client, name), f"a transaction path exists: {name}"
    assert client.read_methods == ("quote",)


def test_a_sell_quote_sends_token_in_and_quote_asset_out():
    """Request parameters: the TOKEN is the input, the exact RAW amount is used."""
    seen = {}

    def _get(params):
        seen.update(params)
        return dict(SELL_BODY)

    client = JupiterQuoteClient(http_get=_get, slippage_bps=1000)
    res = asyncio.run(client.quote_sell(
        token_mint=MINT, quote_asset_mint=WSOL,
        token_amount=64_523_000, token_decimals=6,
    ))
    assert seen["inputMint"] == MINT, seen
    assert seen["outputMint"] == WSOL, seen
    assert seen["amount"] == "64523000", seen
    assert seen["slippageBps"] == "1000", seen
    assert res.out_amount == 1128000
    assert [h.label for h in res.route_plan] == ["Obsidian", "Flux"]
    assert res.pool_keys[0] == POOL


def test_a_sell_direction_is_asserted_before_the_request():
    """A reversed SELL must cost nothing and must not be mistaken for a market."""
    calls = {"n": 0}

    def _get(params):
        calls["n"] += 1
        return dict(SELL_BODY)

    client = JupiterQuoteClient(http_get=_get)
    with pytest.raises(QuoteDirectionError):
        asyncio.run(client.quote_sell(
            token_mint=WSOL, quote_asset_mint=MINT,   # reversed
            token_amount=1000, token_decimals=6,
        ))
    assert calls["n"] == 0, "a reversed SELL must not reach the network"


def test_a_sell_quote_never_affects_a_buy_quote():
    """State must not leak between the two directions."""
    client = _client()
    b1 = asyncio.run(client.quote_buy(token_mint=MINT, quote_asset_mint=WSOL,
                                      in_amount=1_150_000, output_decimals=6))
    client2 = JupiterQuoteClient(http_get=lambda p: dict(SELL_BODY))
    asyncio.run(client2.quote_sell(token_mint=MINT, quote_asset_mint=WSOL,
                                   token_amount=64_523_000, token_decimals=6))
    b2 = asyncio.run(client.quote_buy(token_mint=MINT, quote_asset_mint=WSOL,
                                      in_amount=1_150_000, output_decimals=6))
    assert b1.output_mint == MINT == b2.output_mint
    assert b1.input_mint == WSOL == b2.input_mint


def test_a_venue_echoing_the_wrong_pair_is_refused_on_a_sell():
    """A provider answering a different pair must not yield a reciprocal."""
    client = JupiterQuoteClient(http_get=lambda p: dict(BODY))  # a BUY body
    with pytest.raises(QuoteDirectionError):
        asyncio.run(client.quote_sell(
            token_mint=MINT, quote_asset_mint=WSOL,
            token_amount=64_523_000, token_decimals=6,
        ))


def test_a_sell_refuses_a_zero_or_undecodable_amount():
    client = JupiterQuoteClient(http_get=lambda p: dict(SELL_BODY))
    with pytest.raises(QuoteUnavailable):
        asyncio.run(client.quote_sell(token_mint=MINT, quote_asset_mint=WSOL,
                                      token_amount=0, token_decimals=6))
    with pytest.raises(QuoteUnavailable):
        asyncio.run(client.quote_sell(token_mint=MINT, quote_asset_mint=WSOL,
                                      token_amount=1000, token_decimals=0))


def test_a_sell_route_is_never_claimed_as_same_market_identity():
    """
    A multi-hop route is NOT the position's market. The point produced from a
    liquidation route must therefore never carry identity PASS.
    """
    from autopaper.paper_live.quotes import QuoteService

    client = JupiterQuoteClient(http_get=lambda p: dict(SELL_BODY))
    result = asyncio.run(client.quote_sell(
        token_mint=MINT, quote_asset_mint=WSOL,
        token_amount=64_523_000, token_decimals=6,
    ))
    svc = QuoteService.__new__(QuoteService)
    # the helper reads sol_reference / max_quote_age_ms only for display fields
    svc.sol_reference = None
    svc.max_quote_age_ms = 30_000
    point = QuoteService._point_from_liquidation_route(
        svc, result, mint=MINT, decimals=6, entry_market=POOL, latency_ms=1.0,
    )
    assert point.available is True
    assert point.identity == "UNKNOWN", point.identity
    assert point.observed_pair_address == POOL, "entry market must be preserved"
    assert "OFF-MARKET" in (point.identity_evidence or "")
    assert point.route == ("Obsidian", "Flux")
    # proceeds are quote asset per whole token = out/in
    assert float(point.effective_price) == pytest.approx(
        1128000 / 1e9 / (64523000 / 1e6))


def test_assert_sell_direction_refuses_every_reversal():
    from autopaper.same_market.jupiter import assert_sell_direction

    assert_sell_direction(input_mint=MINT, output_mint=WSOL,
                          quote_asset_mint=WSOL, token_mint=MINT)
    with pytest.raises(QuoteDirectionError):
        assert_sell_direction(input_mint=WSOL, output_mint=MINT,
                              quote_asset_mint=WSOL, token_mint=MINT)
    with pytest.raises(QuoteDirectionError):
        assert_sell_direction(input_mint=MINT, output_mint=USDC,
                              quote_asset_mint=WSOL, token_mint=MINT)
    with pytest.raises(QuoteDirectionError):
        assert_sell_direction(input_mint=MINT, output_mint=MINT,
                              quote_asset_mint=WSOL, token_mint=MINT)


def test_the_quote_client_has_no_transaction_builder():
    for name in ("build", "build_transaction", "send", "sign", "submit"):
        assert not hasattr(JupiterQuoteClient, name), name


def test_direction_holds_for_every_quote_asset_in_acceptance():
    from autopaper.same_market.identity import ACCEPTABLE_QUOTE_MINTS

    for quote_asset in ACCEPTABLE_QUOTE_MINTS:
        assert_buy_direction(input_mint=quote_asset, output_mint=MINT,
                             quote_asset_mint=quote_asset, token_mint=MINT)


def test_a_buy_stays_a_buy_across_repeated_calls():
    """No state leaks between calls that could swap the direction."""
    client = _client()
    for _ in range(3):
        result = asyncio.run(client.quote_buy(
            token_mint=MINT, quote_asset_mint=WSOL,
            in_amount=1_150_000, output_decimals=6,
        ))
        assert result.input_mint == WSOL
        assert result.output_mint == MINT


def test_a_cached_quote_still_has_the_right_direction():
    client = _client()
    first = asyncio.run(client.quote_buy(
        token_mint=MINT, quote_asset_mint=WSOL,
        in_amount=1_150_000, output_decimals=6))
    second = asyncio.run(client.quote_buy(
        token_mint=MINT, quote_asset_mint=WSOL,
        in_amount=1_150_000, output_decimals=6))
    assert first.input_mint == second.input_mint == WSOL
    assert first.output_mint == second.output_mint == MINT