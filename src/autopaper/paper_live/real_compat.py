"""
REAL compatibility for the executable quote contract.

THE PROBLEM
-----------

PAPER settles through :class:`~autopaper.paper_live.quotes.QuoteService`, which
returns a :class:`~autopaper.paper_live.schema.QuotePoint`. REAL settles through
:class:`~autopaper.execution.quoting.QuoteProvider`, which returns an
:class:`~autopaper.execution.quoting.ExecutableQuote`.

Two objects, one meaning. Left alone that is how a PAPER fill and a REAL fill
drift apart on the same venue in the same second -- the PAPER leg priced on one
model and the REAL leg on another -- and the comparison that the whole dataset
exists to support stops meaning anything.

WHAT THIS IS
------------

A pure TRANSLATION. It maps one quote object onto the other and recomputes
nothing: no price, no fee, no impact, no amount is derived here. Every numeric
field is copied from the source quote or from the chain state that quote already
carries. If a field cannot be filled faithfully from the source, it is left
``None`` and labelled, rather than approximated.

WHAT THIS IS NOT
----------------

It is not a second quote engine, and it does not enable REAL. It has no signer,
no builder and no broadcaster; it constructs quotes and nothing else. REAL
execution remains a deliberate operator action, and this module cannot perform
one.

The adapter also never invents a market identity: if the underlying quote's
identity is unproven, the adapter propagates that rather than presenting the
quote as executable, because a REAL consumer must not be handed a price whose
book nobody could name.
"""

from __future__ import annotations

import time
from typing import Any

from autopaper.execution.quoting import (
    ExecutableQuote,
    QuoteProvider,
    Side,
)
from autopaper.pricing import PriceKind
from autopaper.same_market.models import WSOL_MINT

from .quotes import QuoteService
from .schema import QuotePoint

#: The provider name this adapter reports. It names the ADAPTER, not a venue:
#: the venue that actually answered is on the quote.
ADAPTER_NAME = "same_market_quote_service"


def _side_for(side: str) -> Side:
    return Side.BUY if str(side).upper() == "BUY" else Side.SELL


def to_executable_quote(
    point: QuotePoint,
    *,
    side: str,
    amount_raw: int,
    token_decimals: int,
    quote_decimals: int = 9,
    slippage_bps: int = 0,
    quote_asset_usd: float | None = None,
) -> ExecutableQuote:
    """
    Translate one :class:`QuotePoint` into the canonical REAL quote object.

    EVERY numeric field is copied from ``point``. Nothing is recomputed: a
    translator that adjusted a price "for safety" would be a second pricing
    path, which is the thing this module exists to eliminate.
    """
    resolved = _side_for(side)
    now = time.time()
    in_mint = point.quote_mint if resolved is Side.BUY else point.token_mint
    out_mint = point.token_mint if resolved is Side.BUY else point.quote_mint

    # min_output_amount is what a signer would enforce as a floor. It is derived
    # from the quote's own slippage parameter, so the on-chain floor and the
    # quoted figure cannot disagree. With zero slippage it is the quoted output
    # exactly.
    slippage_bps = int(slippage_bps or 0)
    out_raw = point.output_amount_raw
    min_output = None
    if out_raw is not None:
        min_output = int(
            out_raw * (10_000 - slippage_bps) / 10_000)

    quality = "exact" if point.identity == "PASS" else "unproven"
    if point.stale:
        quality = "stale"

    return ExecutableQuote(
        side=resolved,
        leg={
            "token_mint": point.token_mint,
            "quote_mint": point.quote_mint,
            "in_mint": in_mint,
            "out_mint": out_mint,
            "amount_in_raw": int(amount_raw or 0),
            "amount_out_raw": out_raw,
            "token_decimals": token_decimals,
            "quote_decimals": quote_decimals,
            "effective_price": (
                str(point.effective_price) if point.effective_price is not None
                else None),
            "effective_price_usd": point.effective_price_usd,
        },
        provider=point.provider or ADAPTER_NAME,
        venue=point.venue,
        pool=point.pool_or_curve,
        program_id=None,
        route=tuple(point.route or ()),
        price_impact_pct=float(point.price_impact_pct or 0.0),
        slippage_bps=slippage_bps,
        min_output_amount=min_output,
        quoted_at=float(point.quote_timestamp or now),
        received_at=now,
        context_slot=point.state_slot,
        quote_asset_usd=quote_asset_usd,
        # A direct Pump/PumpSwap read is this system's own model of the book,
        # not the venue's own API answer, and it must be labelled as such. Only
        # a venue's own quote may claim venue_quoted.
        venue_quoted=(point.provider == "jupiter"),
        model=None if point.provider == "jupiter" else (
            f"direct:{point.provider}"),
        stale=bool(point.stale),
        quality=quality,
        provenance={
            "adapter": ADAPTER_NAME,
            "venue_id": point.venue_id,
            "market_type": point.market_type,
            "market_identity": point.identity,
            "identity_evidence": point.identity_evidence,
            "quote_age_ms": point.quote_age_ms,
            "max_quote_age_ms": point.max_quote_age_ms,
            "quote_latency_ms": point.quote_latency_ms,
            "fee_bps": point.fee_bps,
            "fee_amount": (
                str(point.fee_amount) if point.fee_amount is not None else None),
            "observed_pair_address": point.observed_pair_address,
        },
        # VENUE_QUOTE, not ACTUAL_EXECUTION: nothing has executed. A REAL
        # consumer must be able to tell a priced intent from a settled fact, and
        # only the latter may ever be labelled as executed.
        kind=PriceKind.VENUE_QUOTE,
    )


class SameMarketQuoteProvider(QuoteProvider):
    """
    The REAL-side view of the one canonical quote service.

    It implements the existing ``QuoteProvider`` interface by DELEGATING to
    :class:`~autopaper.paper_live.quotes.QuoteService`. There is no second
    venue list and no second price: a REAL consumer and a PAPER consumer asking
    the same question at the same size reach the same market through the same
    verified ``same_market`` implementation.

    Constructing this object does NOT enable REAL. It produces quotes. Whether
    anything is signed, submitted or broadcast is a separate, operator-gated
    decision made elsewhere, and this class has no method that could make it.
    """

    name = ADAPTER_NAME
    #: Direct chain reads are a model, not a venue's own answer.
    venue_quoted = False
    supports_venues = ("pump_bonding_curve", "pumpswap", "jupiter")

    def __init__(self, quote_service: QuoteService) -> None:
        self._quotes = quote_service

    async def quote_swap(
        self,
        *,
        side: Any,
        token_mint: str,
        quote_mint: str,
        amount_raw: int,
        token_decimals: int,
        quote_decimals: int | None = None,
        slippage_bps: int = 0,
        quote_asset_usd: float | None = None,
        observed_pair_address: str | None = None,
        pinned_venue_id: str | None = None,
    ) -> ExecutableQuote:
        """
        One executable quote, side-aware, for the exact declared amount.

        A BUY converts the quote asset INTO the token and a SELL does the
        reverse; getting that backwards is the classic way to sell a position
        you were holding and buy one you were not, so the direction is resolved
        once, explicitly, and carried through the translation.
        """
        resolved = _side_for(getattr(side, "value", side))
        service = self._quotes
        if resolved is Side.BUY:
            point = await service.quote_buy(
                mint=token_mint, sol_lamports=int(amount_raw),
                token_decimals=token_decimals,
                observed_pair_address=observed_pair_address)
        else:
            pinned = getattr(service, "quote_sell_pinned", None)
            if callable(pinned) and pinned_venue_id:
                point, _counterfactual = await pinned(
                    mint=token_mint, token_amount_raw=int(amount_raw),
                    token_decimals=token_decimals,
                    pinned_venue_id=pinned_venue_id,
                    observed_pair_address=observed_pair_address)
            else:
                point = await service.quote_sell(
                    mint=token_mint, token_amount_raw=int(amount_raw),
                    token_decimals=token_decimals,
                    observed_pair_address=observed_pair_address)
        return to_executable_quote(
            point, side=resolved.value, amount_raw=int(amount_raw),
            token_decimals=token_decimals,
            quote_decimals=int(quote_decimals or 9),
            slippage_bps=int(slippage_bps or 0),
            quote_asset_usd=quote_asset_usd,
        )


def assert_quote_contract_compatible(quote: ExecutableQuote) -> None:
    """
    The contract a REAL consumer may rely on.

    Raised rather than logged, because a quote that cannot answer these is not
    executable and a REAL path must not proceed on the assumption that it is.
    """
    if quote.side not in (Side.BUY, Side.SELL):
        raise ValueError(f"quote has no side: {quote.side!r}")
    leg = quote.leg or {}
    for field in ("token_mint", "quote_mint", "in_mint", "out_mint"):
        if not leg.get(field):
            raise ValueError(f"quote leg is missing {field}")
    if leg["in_mint"] == leg["out_mint"]:
        raise ValueError(
            "the quote's input and output mints are the same asset")
    if quote.side is Side.BUY and leg["in_mint"] != WSOL_MINT:
        raise ValueError(
            f"a BUY must spend {WSOL_MINT}, not {leg['in_mint']}")
    if quote.side is Side.SELL and leg["out_mint"] != WSOL_MINT:
        raise ValueError(
            f"a SELL must receive {WSOL_MINT}, not {leg['out_mint']}")
    if leg.get("amount_out_raw") in (None, 0):
        raise ValueError("the quote has no output amount")
    if quote.stale:
        raise ValueError(
            "the quote is stale; a stale quote can never be executed")


__all__ = [
    "ADAPTER_NAME",
    "SameMarketQuoteProvider",
    "assert_quote_contract_compatible",
    "to_executable_quote",
]
