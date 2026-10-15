"""
Jupiter V2 read-only quote client, for markets that are not Pump.fun.

WHY A SEPARATE CLIENT
=====================

:mod:`autopaper.execution.jupiter` is blocked by the no-broadcast guard,
because it also exposes ``build`` and ``build_transaction`` - the unsigned
transaction builders. Importing it to reuse its quote method would import the
builder along with it. So this module carries its own quote-only client against
the same public endpoint, with no transaction-building surface.

THE INTERESTING PART: ammKey IS THE SAME-MARKET EVIDENCE
==========================================================

Jupiter's route plan reports, per hop, the AMM it used and - the load-bearing
field - the ``ammKey``, which is the POOL ADDRESS. That is what lets this
harness answer the question the whole experiment exists for:

    is the executable quote coming from the same pool DexScreener marked?

Comparing two prices and calling them "the same market" would prove nothing;
two venues can price the same token far apart. Comparing
``route_plan.ammKey`` against ``dexscreener.pairAddress`` is an identity check,
and when it fails the price gap is not even meaningful, because the two numbers
describe different books.

When the route is multi-hop, or the provider omits ``ammKey``, the answer is
``POOL_MATCH = UNKNOWN``. Not PASS. An unknown is not a pass.

DIRECTION IS ENFORCED, NOT ASSUMED
==================================

This is a BUY test. ``quote_buy`` refuses any call whose input is not the
configured quote asset or whose output is not the token under test, so a
reversed direction cannot quietly produce a reciprocal "price" that looks
plausible. SELL helpers exist on the shared shape but are not reachable from
this module's buy path - see :func:`assert_buy_direction` -- and, since
``quote_sell``, its sell path - see :func:`assert_sell_direction`.

The safety envelope of the production provider (``min_output_tokens``,
``max_price_impact_pct``) is deliberately NOT applied here. Those bounds exist
to stop a trade being placed; this harness places nothing, and a $0.23
micro-order will trip a 1-token minimum on a six-decimal memecoin for reasons
that have nothing to do with whether the price is real. Suppressing that gate
is reported in the quote's detail line so it cannot be mistaken for a quote
that passed the production gate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..market.ratelimit import RateLimiter, RetryAfterBackoff
from .models import SOL_DECIMALS, FailureMode, ExecutableQuote
from .numeric import DecimalError, effective_price, to_decimal

JUPITER_QUOTE_BASE = "https://lite-api.jup.ag/swap/v1"
#: Jupiter refuses very small notionals outright. Named so the report can
#: distinguish "the venue would not price this" from "the price was bad".
MIN_NOTIONAL_LAMPORTS = 1


class QuoteDirectionError(RuntimeError):
    """A quote was requested with the direction reversed."""


class QuoteUnavailable(RuntimeError):
    """The venue returned no usable quote for this pair and size."""


class ProviderRateLimited(RuntimeError):
    """The venue refused the request."""


@dataclass(frozen=True)
class RouteHop:
    """One hop of the route, with the pool identity if the venue disclosed it."""

    label: str | None
    amm_key: str | None
    input_mint: str | None
    output_mint: str | None
    in_amount: int | None
    out_amount: int | None
    fee_amount: int | None
    percent: float | None

    @property
    def has_pool_identity(self) -> bool:
        return bool(self.amm_key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "ammKey": self.amm_key,
            "inputMint": self.input_mint,
            "outputMint": self.output_mint,
            "inAmount": self.in_amount,
            "outAmount": self.out_amount,
            "feeAmount": self.fee_amount,
            "percent": self.percent,
        }


@dataclass(frozen=True)
class JupiterQuoteResult:
    """A raw venue quote, before this harness interprets it."""

    input_mint: str
    output_mint: str
    in_amount: int
    out_amount: int
    other_amount_threshold: int
    price_impact_pct: float | None
    route_plan: tuple[RouteHop, ...]
    raw: Mapping[str, Any] = field(default_factory=dict)
    request_latency_ms: float | None = None
    timestamp: float = 0.0

    @property
    def venue_labels(self) -> tuple[str, ...]:
        return tuple(h.label for h in self.route_plan if h.label)

    @property
    def pool_keys(self) -> tuple[str, ...]:
        return tuple(h.amm_key for h in self.route_plan if h.amm_key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_mint": self.input_mint,
            "output_mint": self.output_mint,
            "in_amount": self.in_amount,
            "out_amount": self.out_amount,
            "other_amount_threshold": self.other_amount_threshold,
            "price_impact_pct": self.price_impact_pct,
            "route_plan": [h.as_dict() for h in self.route_plan],
            "request_latency_ms": self.request_latency_ms,
            "timestamp": self.timestamp,
        }


def assert_buy_direction(
    *,
    input_mint: str,
    output_mint: str,
    quote_asset_mint: str,
    token_mint: str,
) -> None:
    """
    Refuse anything that is not ``quote asset -> token``.

    The failure mode this exists to catch: a reversed pair returns a perfectly
    plausible number that is simply the reciprocal, and a reciprocal compared
    against a USD mark produces a gap of a few hundred percent that reads like
    an extreme market condition rather than a bug.
    """
    if input_mint == output_mint:
        raise QuoteDirectionError("input and output mint are identical")
    if input_mint != quote_asset_mint:
        raise QuoteDirectionError(
            f"BUY must spend the quote asset {quote_asset_mint[:8]}… but the "
            f"input is {input_mint[:8]}…; the direction is reversed or the "
            "wrong asset was configured"
        )
    if output_mint != token_mint:
        raise QuoteDirectionError(
            f"BUY must receive the token under test {token_mint[:8]}… but the "
            f"output is {output_mint[:8]}…"
        )


def assert_sell_direction(
    *,
    input_mint: str,
    output_mint: str,
    quote_asset_mint: str,
    token_mint: str,
) -> None:
    """
    Refuse anything that is not ``token -> quote asset``.

    The mirror of :func:`assert_buy_direction`, and it earns its own function
    because the failure it exists to catch is the SAME reciprocal-price trap in
    the other direction: a reversed SELL returns a finite, plausible, wrong
    number.

    It also checks that the quote asset IS a quote asset. Internal consistency
    alone is not enough: a caller that passes the token as ``quote_asset_mint``
    and the quote asset as ``token_mint`` produces a self-consistent pair that
    prices the swap backwards, and nothing downstream would notice.
    """
    from .identity import ACCEPTABLE_QUOTE_MINTS

    if input_mint == output_mint:
        raise QuoteDirectionError("input and output mint are identical")
    if input_mint != token_mint:
        raise QuoteDirectionError(
            f"SELL must give up the token under test {token_mint[:8]}… but the "
            f"input is {input_mint[:8]}…; the direction is reversed or the "
            "wrong asset was configured"
        )
    if output_mint != quote_asset_mint:
        raise QuoteDirectionError(
            f"SELL must receive the quote asset {quote_asset_mint[:8]}… but the "
            f"output is {output_mint[:8]}…"
        )
    if quote_asset_mint not in ACCEPTABLE_QUOTE_MINTS:
        raise QuoteDirectionError(
            f"{quote_asset_mint[:8]}… is not an accepted quote asset "
            f"({', '.join(sorted(m[:4] + '…' for m in ACCEPTABLE_QUOTE_MINTS))}); "
            "refusing to price a liquidation into an unrecognised asset"
        )
    if token_mint in ACCEPTABLE_QUOTE_MINTS:
        raise QuoteDirectionError(
            f"the token under test {token_mint[:8]}… is itself a quote asset; "
            "a SELL leg cannot have quote assets at both ends"
        )


class JupiterQuoteClient:
    """
    Quote-only Jupiter client. No build, no sign, no send.

    Uses Jupiter's public free endpoint. No API key, no paid plan, nothing that
    would put this harness behind a metered provider.
    """

    name = "jupiter"
    read_methods = ("quote",)

    def __init__(
        self,
        *,
        base_url: str = JUPITER_QUOTE_BASE,
        timeout_s: float = 12.0,
        max_rpm: int = 30,
        cache_ttl_s: float = 0.5,
        max_retries: int = 1,
        slippage_bps: int = 1000,
        client: Any = None,
        http_get: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.slippage_bps = int(slippage_bps)
        self.limiter = RateLimiter(max_rpm, name=self.name)
        self.backoff = RetryAfterBackoff(max_attempts=max_retries,
                                         budget_s=self.timeout_s)
        self.cache_ttl_s = float(cache_ttl_s)
        self._client = client
        self._own_client = client is None
        self._http_get = http_get
        self._cache: dict[tuple[str, str, int], tuple[float, JupiterQuoteResult]] = {}
        self.last_from_cache = False

    async def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self.timeout_s)
            self._own_client = True
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._own_client:
            await self._client.aclose()
        self._client = None

    async def _fetch(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._http_get is not None:
            return self._http_get(dict(params))

        client = await self._ensure_client()
        url = f"{self.base_url}/quote"
        resp = await client.get(
            url,
            params=dict(params),
            timeout=self.timeout_s,
            headers={
                "accept": "application/json",
                "user-agent": "AUTONOMOUS_PAPER_AGENT_V1/1.0 "
                              "(same-market read-only test)",
            },
        )
        text = resp.text or ""
        if resp.status_code == 429:
            raise ProviderRateLimited("jupiter rate limited (HTTP 429)")
        if resp.status_code == 400 or "TOO_SMALL" in text.upper():
            raise QuoteUnavailable(
                f"the venue will not price this notional (HTTP "
                f"{resp.status_code}): {text[:180]}"
            )
        if resp.status_code == 404:
            raise QuoteUnavailable("no route or unknown token for this pair")
        if resp.status_code >= 400:
            raise QuoteUnavailable(f"jupiter HTTP {resp.status_code}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise QuoteUnavailable("jupiter returned non-JSON") from exc
        if not isinstance(body, dict):
            raise QuoteUnavailable("jupiter returned a non-object body")
        return body

    def parse(self, body: Mapping[str, Any], *, latency_ms: float | None = None,
              timestamp: float | None = None) -> JupiterQuoteResult:
        """
        Parse a quote body. Pure - no I/O - so it is testable against fixtures.

        Note what is NOT rejected here: a small output, a large price impact, a
        missing minimum-output. The production provider rejects those because
        they make a trade unsafe. This harness places no trade, and the size of
        the impact is precisely one of the numbers being measured.
        """
        if not isinstance(body, Mapping) or "outAmount" not in body:
            raise QuoteUnavailable("quote response is missing outAmount")
        try:
            out_amount = int(body["outAmount"])
            in_amount = int(body["inAmount"])
            threshold = int(body.get("otherAmountThreshold") or 0)
        except (TypeError, ValueError) as exc:
            raise QuoteUnavailable("quote amounts were not integers") from exc

        hops: list[RouteHop] = []
        for leg in (body.get("routePlan") or []):
            if not isinstance(leg, Mapping):
                continue
            swap = leg.get("swapInfo") or {}
            hops.append(RouteHop(
                label=swap.get("label"),
                amm_key=swap.get("ammKey"),
                input_mint=swap.get("inputMint"),
                output_mint=swap.get("outputMint"),
                in_amount=_int_or_none(swap.get("inAmount")),
                out_amount=_int_or_none(swap.get("outAmount")),
                fee_amount=_int_or_none(swap.get("feeAmount")),
                percent=_float_or_none(swap.get("percent")),
            ))

        impact = _float_or_none(body.get("priceImpactPct"))
        return JupiterQuoteResult(
            input_mint=str(body.get("inputMint") or ""),
            output_mint=str(body.get("outputMint") or ""),
            in_amount=in_amount,
            out_amount=out_amount,
            other_amount_threshold=threshold,
            price_impact_pct=impact,
            route_plan=tuple(hops),
            raw=dict(body),
            request_latency_ms=latency_ms,
            timestamp=time.time() if timestamp is None else timestamp,
        )

    async def quote_buy(
        self,
        *,
        token_mint: str,
        quote_asset_mint: str,
        in_amount: int,
        output_decimals: int,
        input_decimals: int = SOL_DECIMALS,
        slippage_bps: int | None = None,
    ) -> JupiterQuoteResult:
        """
        A fresh BUY quote: ``quote asset -> token``.

        The direction is asserted before the request is made, so a reversed
        call costs nothing and cannot be mistaken for a market with an extreme
        price.
        """
        assert_buy_direction(
            input_mint=quote_asset_mint,
            output_mint=token_mint,
            quote_asset_mint=quote_asset_mint,
            token_mint=token_mint,
        )
        if in_amount < MIN_NOTIONAL_LAMPORTS:
            raise QuoteUnavailable(
                f"buy amount {in_amount} is below the smallest notional this "
                "test can price"
            )

        slip = self.slippage_bps if slippage_bps is None else int(slippage_bps)
        params = {
            "inputMint": quote_asset_mint,
            "outputMint": token_mint,
            "amount": str(in_amount),
            "slippageBps": str(slip),
            "restrictIntermediateTokens": "true",
        }

        key = (quote_asset_mint, token_mint, in_amount)
        now = time.time()
        cached = self._cache.get(key)
        if cached is not None and now - cached[0] <= self.cache_ttl_s:
            self.last_from_cache = True
            return cached[1]

        self.limiter.acquire(now)
        started = time.monotonic()
        body = await self.backoff.run(lambda: self._fetch(params))
        latency_ms = max(0.0, (time.monotonic() - started) * 1000.0)

        result = self.parse(body, latency_ms=latency_ms)
        self.last_from_cache = False
        self._cache[key] = (time.time(), result)
        return result

    async def quote_sell(
        self,
        *,
        token_mint: str,
        quote_asset_mint: str,
        token_amount: int,
        token_decimals: int,
        slippage_bps: int | None = None,
    ) -> JupiterQuoteResult:
        """
        A fresh SELL quote: ``token -> quote asset``.

        READ-ONLY. Like every method on this class it performs one HTTPS GET
        against Jupiter's public quote endpoint. There is no ``build``, no sign,
        no send and no transaction of any kind on this path -- the class has no
        method that can produce one, and ``read_methods`` names ``quote`` alone.

        ``token_amount`` is the EXACT raw token quantity a preceding BUY quote
        produced, in the token's base units. It is not a notional, not a
        percentage of a balance, and not re-derived from a price: liquidating an
        estimate only means anything at the size actually held.

        WHAT THE RESULT IS NOT
        ----------------------
        The returned route is whatever venues Jupiter happened to pick. It is a
        *route plan*, not a market identity: a multi-hop route may traverse
        books this repository has never verified. Callers MUST keep this
        separate from the canonical same-market identity of the entry and MUST
        NOT promote it to one -- see ``QuoteService._point_from_liquidation_-
        route``, which records ``identity="UNKNOWN"`` with the route attached as
        provenance rather than claiming ``PASS``.
        """
        assert_sell_direction(
            input_mint=token_mint,
            output_mint=quote_asset_mint,
            quote_asset_mint=quote_asset_mint,
            token_mint=token_mint,
        )
        if token_decimals <= 0:
            raise QuoteUnavailable(
                f"token decimals {token_decimals} are unusable; refusing to "
                "guess the base-unit size of a liquidation"
            )
        if token_amount <= 0:
            raise QuoteUnavailable(
                f"sell amount {token_amount} is not a positive token quantity"
            )

        slip = self.slippage_bps if slippage_bps is None else int(slippage_bps)
        params = {
            "inputMint": token_mint,
            "outputMint": quote_asset_mint,
            "amount": str(int(token_amount)),
            "slippageBps": str(slip),
            "restrictIntermediateTokens": "true",
        }

        key = (token_mint, quote_asset_mint, int(token_amount))
        now = time.time()
        cached = self._cache.get(key)
        if cached is not None and now - cached[0] <= self.cache_ttl_s:
            self.last_from_cache = True
            return cached[1]

        self.limiter.acquire(now)
        started = time.monotonic()
        body = await self.backoff.run(lambda: self._fetch(params))
        latency_ms = max(0.0, (time.monotonic() - started) * 1000.0)

        result = self.parse(body, latency_ms=latency_ms)
        # Re-verify the direction from the VENUE'S OWN echo of the pair. A
        # provider answering a different pair than the one asked about would
        # otherwise be read as a liquidation price for the wrong leg.
        if result.input_mint != token_mint or result.output_mint != quote_asset_mint:
            raise QuoteDirectionError(
                "venue echoed a different pair than the SELL requested "
                f"(asked {token_mint[:8]}…->{quote_asset_mint[:8]}…, got "
                f"{result.input_mint[:8]}…->{result.output_mint[:8]}…); "
                "refusing to read a liquidation value off it"
            )
        self.last_from_cache = False
        self._cache[key] = (time.time(), result)
        return result


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def to_executable_quote(
    result: JupiterQuoteResult,
    *,
    token_mint: str,
    quote_asset_mint: str,
    output_decimals: int,
    input_decimals: int = SOL_DECIMALS,
) -> ExecutableQuote:
    """
    Convert a venue quote into the harness's ``EXECUTABLE_QUOTE`` shape.

    Re-verifies the direction from the VENUE'S OWN echo of the pair, not just
    from what was requested. A provider that answers a different pair than the
    one asked about would otherwise flow straight into the gap calculation.
    """
    if result.output_mint != token_mint or result.input_mint != quote_asset_mint:
        return ExecutableQuote(
            input_mint=result.input_mint,
            output_mint=result.output_mint,
            input_amount_raw=str(result.in_amount),
            output_amount_raw=str(result.out_amount),
            input_decimals=input_decimals,
            output_decimals=output_decimals,
            input_amount=0.0,
            output_amount=0.0,
            effective_price=None,
            effective_price_native=None,
            price_impact_pct=result.price_impact_pct,
            fee_bps=None,
            fee_amount=None,
            venue=None,
            pool_address=result.pool_keys[0] if result.pool_keys else None,
            route=result.venue_labels,
            provider="jupiter",
            state_timestamp=None,
            quote_timestamp=result.timestamp,
            quote_latency_ms=result.request_latency_ms,
            failure=FailureMode.QUOTE_DIRECTION_ERROR,
            detail=(
                f"the venue answered for {result.input_mint[:8]}… -> "
                f"{result.output_mint[:8]}… but the test asked for "
                f"{quote_asset_mint[:8]}… -> {token_mint[:8]}…"
            ),
        )

    try:
        price = effective_price(
            result.in_amount,
            result.out_amount,
            input_decimals=input_decimals,
            output_decimals=output_decimals,
        )
    except DecimalError as exc:
        return ExecutableQuote(
            input_mint=result.input_mint,
            output_mint=result.output_mint,
            input_amount_raw=str(result.in_amount),
            output_amount_raw=str(result.out_amount),
            input_decimals=input_decimals,
            output_decimals=output_decimals,
            input_amount=0.0,
            output_amount=0.0,
            effective_price=None,
            effective_price_native=None,
            price_impact_pct=result.price_impact_pct,
            fee_bps=None,
            fee_amount=None,
            venue=None,
            pool_address=result.pool_keys[0] if result.pool_keys else None,
            route=result.venue_labels,
            provider="jupiter",
            state_timestamp=None,
            quote_timestamp=result.timestamp,
            quote_latency_ms=result.request_latency_ms,
            failure=FailureMode.DECIMAL_ERROR,
            detail=f"cannot compute an effective price: {exc}",
        )

    return ExecutableQuote(
        input_mint=result.input_mint,
        output_mint=result.output_mint,
        input_amount_raw=str(result.in_amount),
        output_amount_raw=str(result.out_amount),
        input_decimals=input_decimals,
        output_decimals=output_decimals,
        input_amount=float(to_decimal(result.in_amount).scaleb(-input_decimals)),
        output_amount=float(to_decimal(result.out_amount).scaleb(-output_decimals)),
        effective_price=float(price),
        effective_price_native=float(price),
        price_impact_pct=result.price_impact_pct,
        fee_bps=None,
        fee_amount=None,
        venue=(result.venue_labels[0] if result.venue_labels else None),
        pool_address=result.pool_keys[0] if result.pool_keys else None,
        route=result.venue_labels,
        provider="jupiter",
        state_timestamp=None,
        quote_timestamp=result.timestamp,
        quote_latency_ms=result.request_latency_ms,
        detail=(
            "read-only quote; the production provider's min-output and "
            "max-price-impact gate is intentionally NOT applied here, so this "
            "is not a quote that passed the trading gate"
        ),
    )


__all__ = [
    "JUPITER_QUOTE_BASE",
    "JupiterQuoteClient",
    "JupiterQuoteResult",
    "MIN_NOTIONAL_LAMPORTS",
    "ProviderRateLimited",
    "QuoteDirectionError",
    "QuoteUnavailable",
    "RouteHop",
    "assert_buy_direction",
    "assert_sell_direction",
    "to_executable_quote",
]