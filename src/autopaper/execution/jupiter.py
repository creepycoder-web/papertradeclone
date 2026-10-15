"""Jupiter quote + swap construction for the REAL executor.

PROVENANCE
----------
Ported from the reference executor in ``~/works/memecoin-bot``
(``bot/execution/jupiter.py``) without behavioural change. The endpoint, the
query parameters, the parsed fields, the failure vocabulary and every safety
bound are the reference implementation's.

Why Jupiter
-----------
The project had no swap/quote provider at all: ``RealExecutor`` took an
optional ``router`` that was always ``None``, so every quote failed closed and
no order could ever be priced. This module is that missing provider, built on
Jupiter's public aggregator API, which is the venue the project's existing
market-data layer (DexScreener for pricing/market cap) is consistent with.

Flow
----
1. ``quote()``   - price a swap, and enforce the safety envelope (price impact,
                   minimum output, output must be non-zero). A quote that fails
                   a safety bound is REFUSED, never rounded or widened.
2. ``build()``   - ask Jupiter for a fully-formed, *unsigned* transaction.
3. The caller signs it (:mod:`.solana`), simulates it, and only
   then broadcasts it.

Safety properties
-----------------
* ``slippage_bps`` is the venue-side protection; it is never raised above the
  configured ceiling to force a fill.
* An order is NEVER resized. If the venue rejects a size, the failure is
  reported (``MICRO_REAL_EXECUTION_UNSUPPORTED``) and the caller stops.
* Quotes are advisory only. Nothing here records a fill.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from ..pricing import execution_price_per_token, normalize_side
from .price_layers import SwapDirectionError, resolve_side_from_mints
from .provider import ExecutionProvider
from .quote_errors import (
    MicroOrderUnsupported as _MicroOrderUnsupported,
    QuoteRejected as _QuoteRejected,
    QuoteUnavailable as _QuoteUnavailable,
)
from .solana import USDC_MINT, WSOL_MINT

log = logging.getLogger("execution.jupiter")

DEFAULT_QUOTE_BASE = "https://lite-api.jup.ag/swap/v1"

#: Refuse a quote whose modelled price impact exceeds this, expressed in percent.
#: A micro-cap memecoin with a thin book will fail this; that is the intended
#: behaviour, not something to work around.
DEFAULT_MAX_PRICE_IMPCT_PCT = 5.0

#: A quote must imply at least this much output to be considered tradable.
MIN_OUTPUT_TOKENS = 1.0


# The failure vocabulary is owned by ``quote_errors`` so a provider that must not
# import Jupiter can still raise the same classes. These names stay as aliases:
# they are the SAME objects, so every existing ``except`` clause and every
# ``isinstance`` check in the executor is unaffected.
QuoteUnavailable = _QuoteUnavailable
QuoteRejected = _QuoteRejected
MicroOrderUnsupported = _MicroOrderUnsupported


@dataclass
class Quote:
    """A priced, safety-checked swap instruction set."""

    input_mint: str
    output_mint: str
    in_amount: int
    expected_out_amount: int
    min_out_amount: int
    price_impact_pct: float
    route_labels: list = field(default_factory=list)
    slippage_bps: int = 0
    raw: dict = field(default_factory=dict)
    #: The declared direction of THIS quote: "BUY" (quote -> token) or "SELL"
    #: (token -> quote). Required for a price; see :meth:`effective_price`.
    side: str = ""

    def effective_price(self, side: Optional[str] = None) -> Optional[float]:
        """
        Price per RAW output-asset unit, oriented for ``side``.

        Always ``output / input`` on a SELL. The previous implementation was an
        unconditional ``in_amount / expected_out_amount``, which is
        tokens-per-lamport for a sell and was journalled as a price.
        """
        resolved = normalize_side(side or self.side)
        if resolved is None:
            return None
        return execution_price_per_token(
            side=resolved,
            input_amount=self.in_amount,
            output_amount=self.expected_out_amount,
        )

    @property
    def price_per_token(self) -> Optional[float]:
        """Resolved from the quote's OWN declared side; ``None`` without one."""
        return self.effective_price(None)

    def as_dict(self) -> dict:
        return {
            "input_mint": self.input_mint,
            "output_mint": self.output_mint,
            "in_amount": self.in_amount,
            "expected_out_amount": self.expected_out_amount,
            "min_out_amount": self.min_out_amount,
            "price_impact_pct": self.price_impact_pct,
            "route_labels": self.route_labels,
            "slippage_bps": self.slippage_bps,
            "side": self.side,
            "price_per_token": self.effective_price(self.side),
        }


@dataclass
class BuiltSwap:
    """An unsigned, venue-built transaction ready for local signing."""

    transaction_b64: str
    last_valid_block_height: Optional[int]
    prioritization_fee_lamports: Optional[int]
    quote: Quote


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class JupiterProvider(ExecutionProvider):
    """Quote and transaction construction over Jupiter's public API.

    Implements the unified :class:`ExecutionProvider` interface so it can be
    used interchangeably with Raptor -- including as the fallback builder when
    Raptor cannot route a token. It never signs and never broadcasts.

    WSOL SEMANTICS ARE JUPITER'S OWN. Its native ``execute_swap`` path prefers a
    pre-provisioned wSOL account (it routes SOL buys through it and refuses when
    one is absent) because the temporary wrap/unwrap route inflates the
    transaction. As a FALLBACK inside Raptor's path it is instead asked for
    temporary wrap/unwrap -- exactly the semantics the provider supports. Either
    way the requirement is Jupiter's, declared here, never imposed by the
    generic execution layer on another provider.
    """

    #: Jupiter's native SOL path prefers/needs an existing wSOL account; only
    #: Jupiter's own executor path acts on this, never the generic layer.
    requires_preprovisioned_wsol = True
    #: Jupiter can consume a pre-provisioned wSOL account OR wrap/unwrap
    #: internally, depending on the ``wrapAndUnwrapSol`` flag the caller sets.
    sol_input_mode = "preprovision_capable"

    def __init__(self, base_url: str = DEFAULT_QUOTE_BASE,
                 timeout: float = 15.0,
                 max_price_impact_pct: float = DEFAULT_MAX_PRICE_IMPCT_PCT,
                 min_output_tokens: float = MIN_OUTPUT_TOKENS,
                 client: Optional[httpx.AsyncClient] = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_price_impact_pct = max_price_impact_pct
        self.min_output_tokens = min_output_tokens
        self._client = client
        self._own_client = client is None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._own_client:
            await self._client.aclose()
        self._client = None

    async def _json(self, method: str, url: str, payload: Optional[dict] = None
                    ) -> dict:
        client = await self._ensure_client()
        try:
            if method == "GET":
                resp = await client.get(url, timeout=self.timeout)
            else:
                resp = await client.post(url, json=payload or {},
                                         timeout=self.timeout)
        except httpx.TimeoutException as exc:
            raise QuoteUnavailable("quote provider timed out") from exc
        except httpx.HTTPError as exc:
            raise QuoteUnavailable("quote provider transport error") from exc

        text = resp.text or ""
        if resp.status_code == 400 or "TOO_SMALL" in text.upper():
            # Jupiter's documented refusal for a below-minimum order.
            raise MicroOrderUnsupported(
                f"venue rejected the notional as too small (HTTP "
                f"{resp.status_code})")
        if resp.status_code == 404:
            raise QuoteRejected("no route or unknown token for this pair")
        if resp.status_code >= 400:
            raise QuoteUnavailable(
                f"quote provider HTTP {resp.status_code}")
        try:
            return resp.json()
        except ValueError as exc:
            raise QuoteUnavailable(
                "quote provider returned non-JSON") from exc

    # ------------------------------------------------------------------
    async def quote(self, input_mint: str, output_mint: str, in_amount: int,
                    slippage_bps: int, *,
                    side: Optional[str] = None) -> Quote:
        """Price a swap and enforce the safety envelope.

        ``side`` may be supplied by a caller that knows its direction. When it
        is not, the direction is ESTABLISHED from the mints by the invariant in
        :func:`resolve_side_from_mints` -- exactly one end of the leg must be a
        quote asset -- and the quote is stamped with it, so
        :meth:`Quote.effective_price` is correct on a SELL. A leg with both ends
        quote assets, or neither, raises
        :class:`~autopaper.execution.price_layers.SwapDirectionError` rather
        than being priced with a guessed orientation.

        Raises :class:`MicroOrderUnsupported` if the venue refuses the size,
        :class:`QuoteUnavailable` for any other failure. Never returns a quote
        that has breached a safety bound.
        """
        try:
            resolved_side = resolve_side_from_mints(
                input_mint, output_mint, declared_side=side)
        except SwapDirectionError as exc:
            raise QuoteUnavailable(str(exc)) from exc
        if in_amount <= 0:
            raise QuoteUnavailable("in_amount must be positive")
        if slippage_bps <= 0:
            # A zero/absent slippage bound is not tradeable: refuse rather than
            # send an unprotected order.
            raise QuoteUnavailable("slippage_bps must be > 0")
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(in_amount),
            "slippageBps": str(slippage_bps),
            "restrictIntermediateTokens": "true",
        }
        body = await self._get_quote(params)
        return self._to_quote(body, input_mint, output_mint, in_amount,
                              slippage_bps, side=resolved_side.value)

    async def _get_quote(self, params: dict) -> dict:
        client = await self._ensure_client()
        url = f"{self.base_url}/quote"
        try:
            resp = await client.get(url, params=params, timeout=self.timeout)
        except httpx.TimeoutException as exc:
            raise QuoteUnavailable("quote provider timed out") from exc
        except httpx.HTTPError as exc:
            raise QuoteUnavailable("quote provider transport error") from exc
        text = resp.text or ""
        if resp.status_code == 400 or "TOO_SMALL" in text.upper():
            raise MicroOrderUnsupported(
                f"venue rejected the notional as too small (HTTP "
                f"{resp.status_code}): {text[:160]}")
        if resp.status_code == 404:
            raise QuoteRejected("no route or unknown token for this pair")
        if resp.status_code >= 400:
            raise QuoteUnavailable(f"quote provider HTTP {resp.status_code}")
        try:
            return resp.json()
        except ValueError as exc:
            raise QuoteUnavailable(
                "quote provider returned non-JSON") from exc

    def _to_quote(self, body: dict, input_mint: str, output_mint: str,
                  in_amount: int, slippage_bps: int,
                  side: str = "") -> Quote:
        if not isinstance(body, dict) or "outAmount" not in body:
            raise QuoteUnavailable("quote response missing outAmount")
        try:
            out_amount = int(body["outAmount"])
            min_out = int(body.get("otherAmountThreshold") or 0)
        except (TypeError, ValueError) as exc:
            raise QuoteUnavailable("quote amounts were not integers") from exc
        impact = _to_float(body.get("priceImpactPct"), 0.0)

        # Output must be real. A zero-output quote is a refused trade, not a
        # rounding artefact to be cleaned up.
        if out_amount <= 0:
            raise QuoteUnavailable("quote implies zero output tokens")
        if float(out_amount) < self.min_output_tokens:
            raise QuoteUnavailable(
                f"quote output {out_amount} is below the minimum tradable "
                f"amount {self.min_output_tokens:g}")
        if impact > self.max_price_impact_pct:
            raise QuoteRejected(
                f"price impact {impact:.3f}% exceeds the safety limit "
                f"{self.max_price_impact_pct:.3f}%")
        if min_out <= 0:
            raise QuoteUnavailable(
                "quote carries no minimum-output protection")
        # Slippage protection must be meaningful relative to the quote.
        if min_out >= out_amount:
            raise QuoteUnavailable(
                "quote minimum output is not below the expected output")

        labels = []
        for leg in (body.get("routePlan") or []):
            try:
                labels.append(leg["swapInfo"]["label"])
            except (KeyError, TypeError):
                continue
        return Quote(
            input_mint=input_mint,
            output_mint=output_mint,
            in_amount=in_amount,
            expected_out_amount=out_amount,
            min_out_amount=min_out,
            price_impact_pct=impact,
            route_labels=labels,
            slippage_bps=slippage_bps,
            raw=body,
            side=side,
        )

    async def build(self, quote: Quote, user_public_key: str,
                    wrap_and_unwrap_sol: bool = True,
                    priority_fee: Any = "auto",
                    only_direct_routes: bool | None = None,
                    dynamic_compute_unit_limit: bool | None = None) -> BuiltSwap:
        """Ask the venue for an unsigned transaction for this quote.

        The returned transaction is signed locally by the caller; nothing is
        broadcast here.

        ``priority_fee`` must be an **int** (lamports) or the literal string
        ``"auto"``. Jupiter's ``PrioritizationFeeLamports`` is an untagged enum
        that rejects a numeric *string* with HTTP 422, so the value is
        normalised here rather than left to each caller.
        """
        if isinstance(priority_fee, str) and priority_fee.strip().isdigit():
            priority_fee = int(priority_fee.strip())
        # These three change the SHAPE of the transaction the venue returns, not
        # just its cost: wrapAndUnwrapSol decides whether a temporary wrapped-SOL
        # account is created and closed inside the swap, and the two route flags
        # change which intermediate tokens appear. They are therefore candidate
        # dimensions for the execution planner, not retries.
        payload = {
            "quoteResponse": quote.raw,
            "userPublicKey": user_public_key,
            "wrapAndUnwrapSol": wrap_and_unwrap_sol,
            "dynamicComputeUnitLimit": (True if dynamic_compute_unit_limit is None
                                        else bool(dynamic_compute_unit_limit)),
            "prioritizationFeeLamports": priority_fee,
        }
        if only_direct_routes is not None:
            payload["onlyDirectRoutes"] = bool(only_direct_routes)
        body = await self._json("POST", f"{self.base_url}/swap", payload)
        tx_b64 = body.get("swapTransaction")
        if not isinstance(tx_b64, str) or not tx_b64:
            raise QuoteUnavailable(
                "venue did not return a transaction to sign")
        lvh = body.get("lastValidBlockHeight")
        try:
            lvh = int(lvh) if lvh is not None else None
        except (TypeError, ValueError):
            lvh = None
        fee = body.get("prioritizationFeeLamports")
        try:
            fee = int(fee) if fee is not None else None
        except (TypeError, ValueError):
            fee = None
        return BuiltSwap(transaction_b64=tx_b64,
                         last_valid_block_height=lvh,
                         prioritization_fee_lamports=fee,
                         quote=quote)

    # ------------------------------------------------------------------
    # unified provider interface
    # ------------------------------------------------------------------
    def provider_name(self) -> str:
        return "jupiter"

    async def build_transaction(
        self,
        quote: Quote,
        user_public_key: str,
        *,
        wrap_and_unwrap_sol: bool = True,
        priority_fee: Any = "auto",
        only_direct_routes: Optional[bool] = None,
        dynamic_compute_unit_limit: Optional[bool] = None,
        tx_version: Optional[str] = None,   # Jupiter has no tx-version control
    ) -> BuiltSwap:
        """Unified alias of :meth:`build`."""
        return await self.build(
            quote, user_public_key,
            wrap_and_unwrap_sol=wrap_and_unwrap_sol,
            priority_fee=priority_fee,
            only_direct_routes=only_direct_routes,
            dynamic_compute_unit_limit=dynamic_compute_unit_limit,
        )

    def route_metadata(self, quote: Quote) -> dict:
        """Sanitized route projection. No transaction payload."""
        raw = getattr(quote, "raw", None)
        steps = (raw.get("routePlan") if isinstance(raw, dict) else None) or []
        return {
            "provider": "jupiter",
            "route_labels": list(getattr(quote, "route_labels", []) or []),
            "route_hops": len(steps),
        }

    # ------------------------------------------------------------------
    async def health(self) -> dict:
        """
        Readiness probe for the preflight report. Never raises.

        Actually REACHES the venue, rather than merely constructing a client. A
        preflight that reports "quote provider ready" because an object was
        allocated has not verified the one thing the operator is being asked to
        rely on.

        The probe is a real WSOL -> USDC quote for a small amount: read-only,
        free, and the same code path a trade would take. It is bounded by the
        client's own timeout, so a hung venue cannot hang the preflight.
        """
        try:
            await self._ensure_client()
        except Exception as exc:                          # pragma: no cover
            return {"provider": "jupiter", "base_url": self.base_url,
                    "reachable": False, "detail": type(exc).__name__}
        try:
            client = await self._ensure_client()
            resp = await client.get(
                f"{self.base_url}/quote",
                params={
                    "inputMint": WSOL_MINT,
                    "outputMint": USDC_MINT,
                    "amount": str(1_000_000),      # 0.001 SOL
                    "slippageBps": "50",
                },
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return {
                    "provider": "jupiter", "base_url": self.base_url,
                    "reachable": False,
                    "detail": f"quote probe HTTP {resp.status_code}",
                }
            body = resp.json()
            out_amount = body.get("outAmount") if isinstance(body, dict) else None
            if not out_amount:
                return {
                    "provider": "jupiter", "base_url": self.base_url,
                    "reachable": False,
                    "detail": "quote probe returned no outAmount",
                }
            return {
                "provider": "jupiter", "base_url": self.base_url,
                "reachable": True,
                "detail": f"quote probe ok (0.001 SOL -> {out_amount} USDC units)",
            }
        except Exception as exc:                          # pragma: no cover
            return {"provider": "jupiter", "base_url": self.base_url,
                    "reachable": False,
                    "detail": f"quote probe failed: {type(exc).__name__}"}
