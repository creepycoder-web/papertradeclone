"""Adapters that present the existing swap-build providers as quote providers.

WHY AN ADAPTER
--------------
:mod:`autopaper.execution.raptor` and :mod:`autopaper.execution.jupiter` are
BUILD providers: they quote *and* return an unsigned transaction. The quote
layer in :mod:`autopaper.execution.quoting` only needs the price, and needs it
as an :class:`~autopaper.execution.price_layers.ExecutableQuote` with venue,
pool, route and timestamps attached.

Rather than grow a second code path, each build provider gains a thin adapter
that reuses its existing ``quote()`` and converts the result. The adapter
deliberately does NOT build anything: a pre-trade price and an executable order
are different questions, and a caller that wants a price must not be able to
get a transaction by accident.

DIRECTION
---------
The adapters exist so callers use ``quote_buy`` / ``quote_sell`` from
:class:`~autopaper.execution.quoting.QuoteProvider`, neither of which accepts a
``side``. The direction is therefore known here with certainty -- ``BUY`` spends
the quote asset, ``SELL`` spends the token -- and is passed explicitly to the
inner provider, which validates it against the mints.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from .price_layers import (
    ExecutableQuote,
    RouteHop,
    Side,
    SwapDirectionError,
    leg_for_buy,
    leg_for_sell,
    resolve_side_from_mints,
)
from .quote_errors import QuoteRejected, QuoteUnavailable
from .quoting import QuoteProvider

log = logging.getLogger("execution.aggregators")

__all__ = ["AggregatorQuoteProvider", "QuoteOnlyJupiterProvider"]


class QuoteOnlyJupiterProvider(QuoteProvider):
    """
    A QUOTE-ONLY Jupiter client, for PAPER.

    STRUCTURALLY INCAPABLE OF BUILDING A TRANSACTION
    ------------------------------------------------
    This class has no ``build``, no ``build_transaction``, no signer and no
    broadcaster, and it imports nothing that has one. That is deliberate: a PAPER
    process must be able to ask "what would this cost?" without acquiring any
    capability to send it, and
    ``tests/test_paper_smoke.py::test_paper_smoke_does_not_sign_or_broadcast``
    asserts that PAPER never even constructs the REAL executor.

    Routing PAPER's executable layer through the REAL execution backend would
    break that invariant to satisfy a telemetry requirement. It would also be
    backwards: the REAL backend is the thing PAPER must not touch.

    It is the free public Jupiter tier. No key, no paid provider.
    """

    name = "jupiter"
    venue_quoted = True
    supports_venues = ("aggregated",)

    def __init__(self, *, base_url: str = "https://lite-api.jup.ag/swap/v1",
                 timeout: float = 8.0,
                 client: Optional[Any] = None,
                 max_price_impact_pct: float = 50.0) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = timeout
        self.max_price_impact_pct = float(max_price_impact_pct)
        self._client = client
        self._own_client = client is None

    async def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._own_client:
            await self._client.aclose()
        self._client = None

    async def quote_swap(self, *, side: Any, token_mint: str, quote_mint: str,
                         amount_raw: int, token_decimals: int,
                         quote_decimals: Optional[int] = None,
                         slippage_bps: int = 500,
                         quote_asset_usd: Optional[float] = None,
                         ) -> ExecutableQuote:
        import httpx as _httpx
        from .price_layers import resolve_side_from_mints

        resolved = Side.coerce(side)
        in_mint = quote_mint if resolved is Side.BUY else token_mint
        out_mint = token_mint if resolved is Side.BUY else quote_mint
        resolve_side_from_mints(in_mint, out_mint, declared_side=resolved)
        params = {
            "inputMint": in_mint, "outputMint": out_mint,
            "amount": str(int(amount_raw)), "slippageBps": str(int(slippage_bps)),
            "restrictIntermediateTokens": "true",
        }
        client = await self._ensure_client()
        try:
            resp = await client.get(f"{self.base_url}/quote", params=params,
                                    timeout=self.timeout)
        except _httpx.HTTPError as exc:
            raise QuoteUnavailable(
                f"jupiter quote transport error: {type(exc).__name__}") from exc
        if resp.status_code >= 400:
            raise QuoteRejected(
                f"jupiter quote HTTP {resp.status_code}: {(resp.text or '')[:120]}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise QuoteUnavailable("jupiter quote returned non-JSON") from exc
        out_amount = int(body.get("outAmount") or 0)
        if out_amount <= 0:
            raise QuoteRejected("jupiter quote implies zero output")
        impact = float(body.get("priceImpactPct") or 0.0)
        hops = _hops_from(None, body)
        if resolved is Side.BUY:
            leg = leg_for_buy(token_mint=token_mint, quote_mint=quote_mint,
                              quote_amount=int(amount_raw),
                              quote_decimals=quote_decimals,
                              token_decimals=int(token_decimals),
                              expected_token_amount=out_amount)
        else:
            leg = leg_for_sell(token_mint=token_mint, quote_mint=quote_mint,
                               token_amount=int(amount_raw),
                               quote_decimals=quote_decimals,
                               token_decimals=int(token_decimals),
                               expected_quote_amount=out_amount)
        now = time.time()
        return ExecutableQuote(
            side=resolved, leg=leg, provider=self.name,
            venue=(hops[0].venue if hops else None),
            pool=(hops[0].pool if hops else None),
            program_id=(hops[0].program_id if hops else None),
            route=tuple(hops),
            price_impact_pct=impact * 100.0,
            slippage_bps=int(slippage_bps),
            min_output_amount=(int(body.get("otherAmountThreshold") or 0)
                               or None),
            quoted_at=now, received_at=now,
            quote_asset_usd=quote_asset_usd, venue_quoted=True,
            provenance={"mode": "PAPER", "quote_only": True},
        )


class AggregatorQuoteProvider(QuoteProvider):
    """
    Quote-only view over a build provider.

    ``inner`` is anything with ``provider_name()`` and
    ``await quote(input_mint, output_mint, in_amount, slippage_bps, side=...)``.
    Constructing one is inert.
    """

    name = "aggregator"
    venue_quoted = True

    def __init__(self, inner: Any, *, name: Optional[str] = None,
                 token_decimals: Optional[int] = None) -> None:
        self.inner = inner
        self.name = name or (
            inner.provider_name() if hasattr(inner, "provider_name")
            else "aggregator")
        self.supports_venues = ("aggregated",)
        #: Optional mint-decimals resolver. When supplied, the adapter reads the
        #: TOKEN's decimals rather than trusting the caller, because a per-token
        #: price derived from wrong decimals is wrong by a power of ten.
        self._token_decimals = token_decimals

    async def quote_swap(self, *, side: Any, token_mint: str, quote_mint: str,
                         amount_raw: int, token_decimals: int,
                         quote_decimals: Optional[int] = None,
                         slippage_bps: int = 0,
                         quote_asset_usd: Optional[float] = None,
                         ) -> ExecutableQuote:
        resolved = Side.coerce(side)
        if token_decimals is None:
            token_decimals = self._token_decimals
        if token_decimals is None:
            raise QuoteUnavailable(
                f"{self.name}: the token's decimals are unknown, so no "
                f"per-token price can be stated; refusing rather than guessing "
                f"a scale")

        in_mint = quote_mint if resolved is Side.BUY else token_mint
        out_mint = token_mint if resolved is Side.BUY else quote_mint
        # Establishes the direction from the mints AND checks it against the
        # declared side, so a caller cannot pass a SELL's mints with a BUY label.
        checked = resolve_side_from_mints(in_mint, out_mint,
                                          declared_side=resolved)
        try:
            inner = await self.inner.quote(in_mint, out_mint, int(amount_raw),
                                           int(slippage_bps),
                                           side=checked.value)
        except TypeError:
            # A build provider from before ``side`` existed. Refusing is the
            # only safe answer: the alternative is quoting without a direction.
            raise QuoteUnavailable(
                f"{self.name}: the underlying provider does not accept an "
                f"explicit direction; refusing to quote without one")

        out_amount = int(getattr(inner, "expected_out_amount", 0) or 0)
        if out_amount <= 0:
            raise QuoteRejected(
                f"{self.name}: venue returned a non-positive output amount")
        if resolved is Side.BUY:
            leg = leg_for_buy(token_mint=token_mint, quote_mint=quote_mint,
                              quote_amount=int(amount_raw),
                              quote_decimals=quote_decimals,
                              token_decimals=int(token_decimals),
                              expected_token_amount=out_amount)
        else:
            leg = leg_for_sell(token_mint=token_mint, quote_mint=quote_mint,
                               token_amount=int(amount_raw),
                               quote_decimals=quote_decimals,
                               token_decimals=int(token_decimals),
                               expected_quote_amount=out_amount)

        hops = _hops_from(getattr(inner, "route_steps", None),
                          getattr(inner, "raw", None))
        now = time.time()
        return ExecutableQuote(
            side=resolved, leg=leg, provider=self.name,
            venue=(hops[0].venue if hops and hops[0].venue else None),
            pool=(hops[0].pool if hops and hops[0].pool else None),
            program_id=(hops[0].program_id if hops and hops[0].program_id
                        else None),
            route=tuple(hops),
            price_impact_pct=float(getattr(inner, "price_impact_pct", 0.0) or 0.0),
            slippage_bps=int(slippage_bps),
            min_output_amount=(int(getattr(inner, "min_out_amount", 0) or 0)
                               or None),
            quoted_at=float(getattr(inner, "quote_created_at", 0.0) or 0.0),
            received_at=now,
            context_slot=getattr(inner, "context_slot", None),
            quote_asset_usd=quote_asset_usd,
            venue_quoted=True,
            stale=False, quality="live",
            provenance={
                "route_labels": list(getattr(inner, "route_labels", []) or []),
                "inner": self.name,
                "out_unit": "tokens" if resolved is Side.BUY else "quote_raw",
            },
        )


def _hops_from(route_steps: Any, raw: Any) -> list[RouteHop]:
    hops: list[RouteHop] = []
    for step in (route_steps or []):
        if isinstance(step, RouteHop):
            hops.append(step)
        elif isinstance(step, dict):
            hops.append(RouteHop(
                venue=str(step.get("dex") or step.get("venue") or ""),
                program_id=str(step.get("program_id")
                               or step.get("programId") or ""),
                pool=str(step.get("pool") or ""),
                input_mint=str(step.get("input_mint") or ""),
                output_mint=str(step.get("output_mint") or ""),
                amount_in=str(step.get("amount_in") or ""),
                amount_out=str(step.get("amount_out") or ""),
                percent=float(step.get("percent") or 0.0),
            ))
        else:
            # A provider's own step type. Duck-typed: an exact type match would
            # silently drop the route and leave the venue unverified.
            hops.append(RouteHop(
                venue=str(getattr(step, "dex", "")
                          or getattr(step, "venue", "") or ""),
                program_id=str(getattr(step, "program_id", "")
                               or getattr(step, "programId", "") or ""),
                pool=str(getattr(step, "pool", "") or ""),
                input_mint=str(getattr(step, "input_mint", "") or ""),
                output_mint=str(getattr(step, "output_mint", "") or ""),
                amount_in=str(getattr(step, "amount_in", "") or ""),
                amount_out=str(getattr(step, "amount_out", "") or ""),
                percent=float(getattr(step, "percent", 0.0) or 0.0),
            ))
    if hops:
        return hops
    for leg in ((raw or {}).get("routePlan") or []):
        info = leg.get("swapInfo") or {}
        if not isinstance(info, dict):
            continue
        hops.append(RouteHop(
            venue=str(info.get("label") or ""),
            program_id=str(info.get("programId") or ""),
            pool=str(info.get("ammKey") or ""),
            input_mint=str(info.get("inputMint") or ""),
            output_mint=str(info.get("outputMint") or ""),
            amount_in=str(info.get("inAmount") or ""),
            amount_out=str(info.get("outAmount") or ""),
        ))
    return hops