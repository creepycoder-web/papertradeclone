"""Solana Tracker **Raptor** client: unified swap-routing / transaction build.

WHY THIS EXISTS
---------------
The REAL execution layer used to be a set of per-venue builders
(Pump.fun / PumpSwap / Raydium / Meteora ...). Each venue is a moving target
and each new one multiplied the amount of unverified instruction layout the bot
had to get right. Raptor replaces all of them with a single, hosted router that
returns an *unsigned* transaction; the bot keeps every safety decision.

RAPTOR'S ROLE -- AND ONLY ITS ROLE
----------------------------------
Raptor is execution infrastructure: it quotes a route and builds a serialized
transaction. It does **not** hold the wallet, does **not** sign, does **not**
broadcast, and does **not** decide position size, slippage or freshness. Those
stay in this bot:

    Raptor quote -> Raptor build -> bot deserialize -> bot validate
    -> bot size gate -> bot freshness gate -> bot intent/idempotency
    -> bot simulate -> bot sign -> bot broadcast -> bot confirm -> bot reconcile

SECRET BOUNDARY
---------------
Nothing in this module accepts, stores, transmits or logs private key material,
a seed phrase, or a signed transaction. The only wallet-derived value ever sent
to Raptor is the **public** address (``userPublicKey``), because that is what a
transaction must be built for. The bot never calls Raptor's ``/send-transaction``
endpoint; broadcast always goes through the bot's own :class:`SolanaRpc`.

UNITS (verified against the live hosted API, 2026-10-01)
--------------------------------------------------------
* ``GET /quote``  ``amount`` is in the *smallest unit* (lamports for SOL).
  ``slippageBps`` is a string (or ``"dynamic"``). Quote response:
  ``amountIn/amountOut/minAmountOut/feeAmount`` (decimal strings),
  ``priceImpact`` (a FRACTION: 0.0253 == 2.53%), ``slippageBps``,
  ``routePlan[]`` steps with ``dex``/``programId``/``pool``/``percent``.
* ``POST /swap`` body carries the full ``quoteResponse`` object plus
  ``userPublicKey``, ``wrapUnwrapSol`` (bool), ``txVersion`` (``"V0"``/``"LEGACY"``)
  and optional ``priorityFee``. ``priorityFee`` is a **string**; a numeric string
  is interpreted as a target total priority-fee budget in lamports. The numeric
  int form is rejected with HTTP 422, so the value is stringified here.
  Response: ``swapTransaction`` (base64), ``lastValidBlockHeight``,
  ``contextSlot``, ``prioritizationFeeLamports``.
* SOL is represented as the wrapped-SOL mint (``So111...112``); Raptor wraps
  native SOL into wSOL in ``setupInstructions`` and unwraps on the way out,
  because ``wrapUnwrapSol`` defaults to true. The bot does not pre-provision
  wSOL for Raptor.

NO TRADING POLICY LIVES HERE. Sizing, the slippage ceiling, the freshness gate,
simulation, signing, broadcast and reconciliation are all the caller's.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from ..pricing import (
    execution_price_per_token,
    normalize_side,
)
from .price_layers import SwapDirectionError, resolve_side_from_mints
from .jupiter import MicroOrderUnsupported, QuoteRejected, QuoteUnavailable
from .provider import ExecutionProvider
from .solana import USDC_MINT, WSOL_MINT

log = logging.getLogger("execution.raptor")

#: The hosted Raptor endpoint. No API key / signup is required for the hosted
#: swap API, so no secret is read or sent. Overridable via ``RAPTOR_BASE_URL``.
DEFAULT_RAPTOR_BASE = "https://raptor.solanatracker.io"

#: Refuse a quote whose modelled price impact exceeds this percentage. Same
#: semantic ceiling the Jupiter path uses; a thin memecoin book fails this and
#: that is the intended refusal.
DEFAULT_MAX_PRICE_IMPCT_PCT = 5.0

#: A quote must imply at least this much output to be tradable.
MIN_OUTPUT_TOKENS = 1.0

#: Default route depth. Raptor supports 1-4 hops.
DEFAULT_MAX_HOPS = 4

#: Raptor's transaction format. V0 is the aggregator standard.
RAPTOR_TX_VERSION = "V0"


# ---------------------------------------------------------------------------
# structured errors
# ---------------------------------------------------------------------------
# Every Raptor failure is a NAMED condition carrying a stable ``status`` code
# (the vocabulary section 16 of the migration asks for) plus optional HTTP
# status / provider code / a sanitized provider message. They subclass the
# existing Jupiter exceptions so any code path that already catches
# ``QuoteUnavailable`` / ``QuoteRejected`` / ``MicroOrderUnsupported`` keeps
# working unchanged, while the ``.status`` attribute lets a caller report the
# Raptor-specific reason without string matching.


class RaptorError(Exception):
    """Base: a classified Raptor failure with a stable status code."""

    status: str = "RAPTOR_ERROR"

    def __init__(
        self,
        message: str,
        *,
        http_status: Optional[int] = None,
        provider_code: Any = None,
        provider_message: str = "",
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.provider_code = provider_code
        #: Provider text, truncated. Never contains key material; Raptor never
        #: receives any.
        self.provider_message = (provider_message or "")[:240]


class RaptorNetworkError(RaptorError, QuoteUnavailable):
    status = "RAPTOR_NETWORK_ERROR"


class RaptorInvalidResponse(RaptorError, QuoteUnavailable):
    status = "RAPTOR_INVALID_RESPONSE"


class RaptorQuoteFailed(RaptorError, QuoteRejected):
    status = "RAPTOR_QUOTE_FAILED"


class RaptorUnsupportedRoute(RaptorError, QuoteRejected):
    status = "RAPTOR_UNSUPPORTED_ROUTE"


class RaptorMicroOrderUnsupported(RaptorError, MicroOrderUnsupported):
    #: Preserved so the caller's existing micro-notional handling fires.
    status = "MICRO_REAL_EXECUTION_UNSUPPORTED"


class RaptorBuildFailed(RaptorError, QuoteUnavailable):
    status = "RAPTOR_BUILD_FAILED"


class RaptorTxTooLarge(RaptorError, QuoteUnavailable):
    status = "RAPTOR_TX_TOO_LARGE"


class RaptorStaleQuote(RaptorError, QuoteUnavailable):
    status = "RAPTOR_STALE_QUOTE"


#: Substrings that mark a "this token/route simply cannot be traded" refusal,
#: as opposed to a transport/5xx fault. Kept as data so a wording change
#: degrades to the generic classification rather than misfiring.
_UNSUPPORTED_MARKERS = (
    "no multi-hop route",
    "no route",
    "unable to fetch pools",
    "invalid output mint",
    "invalid input mint",
    "invalid mint",
    "not tradable",
    "no pools",
)
#: Substrings that mark the venue refusing the *notional* as too small. This is
#: the MICRO_REAL_EXECUTION_UNSUPPORTED condition and must never be retried
#: with a larger amount.
_MICRO_MARKERS = ("too small", "minimum", "amount too low", "below minimum")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@dataclass
class RouteStep:
    """One hop of a Raptor route. Sanitized: no transaction payload."""

    dex: str = ""
    program_id: str = ""
    pool: str = ""
    input_mint: str = ""
    output_mint: str = ""
    amount_in: str = ""
    amount_out: str = ""
    percent: float = 0.0
    price_impact_pct: float = 0.0

    def as_dict(self) -> dict:
        return {
            "dex": self.dex,
            "program_id": self.program_id,
            "pool": self.pool,
            "input_mint": self.input_mint,
            "output_mint": self.output_mint,
            "amount_in": self.amount_in,
            "amount_out": self.amount_out,
            "percent": self.percent,
            "price_impact_pct": self.price_impact_pct,
        }


@dataclass
class Quote:
    """A priced, safety-checked Raptor swap instruction set."""

    input_mint: str
    output_mint: str
    in_amount: int
    expected_out_amount: int
    min_out_amount: int
    price_impact_pct: float
    route_labels: list = field(default_factory=list)
    route_steps: list = field(default_factory=list)
    slippage_bps: int = 0
    #: Monotonic-free wall-clock timestamps used by the caller's freshness gate.
    quote_created_at: float = 0.0
    quote_received_at: float = 0.0
    context_slot: Optional[int] = None
    swap_usd_value: Optional[str] = None
    raw: dict = field(default_factory=dict)
    #: The declared direction of THIS quote: ``"BUY"`` (quote -> token) or
    #: ``"SELL"`` (token -> quote). Required. A quote with no direction cannot
    #: produce a price, because the price's orientation depends on it.
    side: str = ""

    def effective_price(self, side: Optional[str] = None) -> Optional[float]:
        """
        Price per RAW output-asset unit, oriented for ``side``.

        THE ORIENTATION IS NOT OPTIONAL. This used to be a ``price_per_token``
        property that always divided ``in_amount`` by ``expected_out_amount``.
        On a SELL that is tokens-per-lamport, and the 2026-10-02 $notify exit
        persisted ``quote_price = 6634.915851383544`` -- a tokens-per-lamport
        figure stored in a field every other consumer read as a price. The
        direction is now a required argument and the quote's own ``side`` is the
        default, so a caller that forgets it gets the quote's own direction
        rather than the wrong one.

        Returns ``None``, never ``0.0``, when the amounts cannot support the
        division or when no direction can be determined.
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
        """
        REMOVED BEHAVIOUR, KEPT AS A NAME.

        This property used to return ``in_amount / expected_out_amount`` for
        every side, which is wrong for a SELL. It now resolves the price from the
        quote's OWN declared ``side`` and returns ``None`` when the quote carries
        no direction, so a caller that ignores ``side`` can no longer be handed
        an inverted number. New code should call :meth:`effective_price`.
        """
        return self.effective_price(None)

    @property
    def selected_dex(self) -> str:
        return self.route_labels[0] if self.route_labels else ""

    def as_dict(self) -> dict:
        return {
            "provider": "raptor",
            "input_mint": self.input_mint,
            "output_mint": self.output_mint,
            "in_amount": self.in_amount,
            "expected_out_amount": self.expected_out_amount,
            "min_out_amount": self.min_out_amount,
            "price_impact_pct": self.price_impact_pct,
            "route_labels": list(self.route_labels),
            "route_steps": [s.as_dict() if isinstance(s, RouteStep) else s
                            for s in self.route_steps],
            "slippage_bps": self.slippage_bps,
            "side": self.side,
            "price_per_token": self.effective_price(self.side),
            "context_slot": self.context_slot,
            "swap_usd_value": self.swap_usd_value,
        }


@dataclass
class BuiltSwap:
    """An unsigned, Raptor-built transaction ready for local signing.

    ``transaction_b64`` is NEVER logged by this module. ``signature_slots`` and
    ``unsigned`` describe the signature envelope so the caller can assert the
    venue did not return a partially- (or fully-) signed transaction.
    """

    transaction_b64: str
    last_valid_block_height: Optional[int]
    prioritization_fee_lamports: Optional[int]
    quote: Quote
    context_slot: Optional[int] = None
    tx_version: str = RAPTOR_TX_VERSION
    signature_slots: int = 0
    unsigned: bool = True
    #: Route / transaction shape, measured by the caller.  Metadata only.
    route_metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _looks_unsigned(tx_b64: str) -> tuple[int, bool]:
    """Return ``(signature_slots, unsigned)`` for a serialized transaction.

    Decoded with solders so the check is a real parse, not a byte guess. A
    Raptor-built transaction is v0/legacy with exactly one all-zero signature
    slot (the payer's). Any non-zero signature means the venue signed something
    and the bot REFUSES it: the whole point is that the bot signs.
    """
    raw = base64.b64decode(tx_b64)
    from solders.transaction import VersionedTransaction

    vtx = VersionedTransaction.from_bytes(raw)
    sigs = list(vtx.signatures)
    unsigned = all(bytes(s) == bytes(64) for s in sigs)
    return len(sigs), unsigned


class RaptorProvider(ExecutionProvider):
    """Quote and unsigned-transaction construction over the hosted Raptor API.

    Constructing it is inert: no client, no network, no key. The HTTP client is
    created on first use and can be injected for tests.

    SOL INPUT IS PROVIDER-MANAGED. Raptor wraps native SOL into wSOL inside the
    swap and unwraps on the way out (``wrapUnwrapSol`` defaults true), so a
    SOL-input BUY needs NO pre-provisioned wSOL account and no ATA setup.
    """

    #: Raptor manages native SOL itself; it never requires the wallet to hold a
    #: funded wrapped-SOL account for a SOL-input swap.
    requires_preprovisioned_wsol = False
    #: Native SOL is wrapped/unwrapped inside the Raptor-built transaction.
    sol_input_mode = "auto_wrap_unwrap"

    def __init__(
        self,
        base_url: str = DEFAULT_RAPTOR_BASE,
        timeout: float = 15.0,
        max_price_impact_pct: float = DEFAULT_MAX_PRICE_IMPCT_PCT,
        min_output_tokens: float = MIN_OUTPUT_TOKENS,
        *,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.base_url = (base_url or DEFAULT_RAPTOR_BASE).rstrip("/")
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

    # -- transport ---------------------------------------------------------
    async def _request(self, method: str, path: str, *, params=None,
                       json=None, kind: str) -> dict:
        client = await self._ensure_client()
        url = f"{self.base_url}{path}"
        try:
            if method == "GET":
                resp = await client.get(url, params=params, timeout=self.timeout)
            else:
                resp = await client.post(url, json=json, timeout=self.timeout)
        except httpx.TimeoutException as exc:
            raise RaptorNetworkError("raptor timed out") from exc
        except httpx.HTTPError as exc:
            raise RaptorNetworkError("raptor transport error") from exc

        text = resp.text or ""
        if resp.status_code >= 400:
            self._raise_http(resp.status_code, text, kind)
        try:
            body = resp.json()
        except ValueError as exc:
            raise RaptorInvalidResponse(
                "raptor returned non-JSON", http_status=resp.status_code,
            ) from exc
        if not isinstance(body, dict):
            raise RaptorInvalidResponse(
                "raptor response was not an object",
                http_status=resp.status_code,
            )
        return body

    def _raise_http(self, status: int, text: str, kind: str) -> None:
        lowered = (text or "").lower()
        provider_msg = ""
        provider_code: Any = None
        try:
            parsed = httpx.Response(status_code=status, text=text).json()
            if isinstance(parsed, dict):
                provider_msg = str(parsed.get("error") or parsed.get("details")
                                   or parsed.get("message") or "")
                provider_code = parsed.get("code")
        except Exception:  # noqa: BLE001 - body may not be JSON
            provider_msg = text[:160]

        if any(m in lowered for m in _MICRO_MARKERS):
            raise RaptorMicroOrderUnsupported(
                f"raptor refused the notional (HTTP {status}): {provider_msg}",
                http_status=status, provider_code=provider_code,
                provider_message=provider_msg)
        if any(m in lowered for m in _UNSUPPORTED_MARKERS):
            raise RaptorUnsupportedRoute(
                f"raptor could not construct a route (HTTP {status}): "
                f"{provider_msg}",
                http_status=status, provider_code=provider_code,
                provider_message=provider_msg)
        if kind == "quote":
            raise RaptorQuoteFailed(
                f"raptor quote failed (HTTP {status}): {provider_msg}",
                http_status=status, provider_code=provider_code,
                provider_message=provider_msg)
        raise RaptorBuildFailed(
            f"raptor swap build failed (HTTP {status}): {provider_msg}",
            http_status=status, provider_code=provider_code,
            provider_message=provider_msg)

    # -- quote -------------------------------------------------------------
    async def quote(
        self,
        input_mint: str,
        output_mint: str,
        in_amount: int,
        slippage_bps: int,
        *,
        side: Optional[str] = None,
        max_hops: int = DEFAULT_MAX_HOPS,
        dexes: Optional[str] = None,
    ) -> Quote:
        """Price a swap and enforce the safety envelope.

        ``side`` may be supplied by a caller that knows its direction. When it
        is not, the direction is ESTABLISHED from the mints by the stated
        invariant in :func:`resolve_side_from_mints` -- exactly one end of the leg
        must be a quote asset -- and the quote is stamped with it. A leg with
        both ends quote assets, or neither, raises
        :class:`~autopaper.execution.price_layers.SwapDirectionError` rather
        than being priced with a guessed orientation.

        The direction is not decoration: it is what makes
        :meth:`Quote.effective_price` correct on a SELL. The previous
        ``price_per_token`` always divided input by output, which on a sell
        published tokens-per-lamport as if it were a price.

        Raises a classified :class:`RaptorError` subclass on failure. Never
        returns a quote that has breached a safety bound, and never resizes the
        order: a refusal is reported, not worked around.
        """
        import time

        try:
            resolved_side = resolve_side_from_mints(
                input_mint, output_mint, declared_side=side)
        except SwapDirectionError as exc:
            raise RaptorQuoteFailed(str(exc)) from exc
        if in_amount <= 0:
            raise RaptorQuoteFailed("in_amount must be positive")
        if slippage_bps <= 0:
            # A zero/absent slippage bound is not tradeable: refuse rather than
            # send an unprotected order.
            raise RaptorQuoteFailed("slippage_bps must be > 0")
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(int(in_amount)),
            # The API expects a STRING here ("50" or "dynamic").
            "slippageBps": str(int(slippage_bps)),
            "maxHops": str(int(max_hops)),
        }
        if dexes:
            params["dexes"] = dexes
        body = await self._request("GET", "/quote", params=params, kind="quote")
        quote = self._to_quote(body, input_mint, output_mint, int(in_amount),
                              int(slippage_bps), side=resolved_side.value)
        quote.quote_received_at = time.time()
        return quote

    def _to_quote(self, body: dict, input_mint: str, output_mint: str,
                  in_amount: int, slippage_bps: int,
                  side: str = "") -> Quote:
        import time

        if "amountOut" not in body:
            raise RaptorInvalidResponse("raptor quote response missing amountOut")
        try:
            out_amount = int(body["amountOut"])
            min_out = int(body.get("minAmountOut") or 0)
        except (TypeError, ValueError) as exc:
            raise RaptorInvalidResponse("raptor quote amounts were not integers") from exc

        # Raptor returns priceImpact as a FRACTION (0.0253 == 2.53%). Convert to
        # a percentage so the same ceiling applies as the Jupiter path.
        impact = _to_float(body.get("priceImpact"), 0.0) * 100.0

        if out_amount <= 0:
            raise RaptorQuoteFailed("raptor quote implies zero output tokens")
        if float(out_amount) < self.min_output_tokens:
            raise RaptorQuoteFailed(
                f"raptor quote output {out_amount} is below the minimum "
                f"tradable amount {self.min_output_tokens:g}")
        if impact > self.max_price_impact_pct:
            raise RaptorQuoteFailed(
                f"price impact {impact:.3f}% exceeds the safety limit "
                f"{self.max_price_impact_pct:.3f}%")
        if min_out <= 0:
            raise RaptorQuoteFailed(
                "raptor quote carries no minimum-output protection")
        if min_out >= out_amount:
            raise RaptorQuoteFailed(
                "raptor quote minimum output is not below the expected output")

        steps: list[RouteStep] = []
        labels: list[str] = []
        for leg in (body.get("routePlan") or []):
            if not isinstance(leg, dict):
                continue
            step = RouteStep(
                dex=str(leg.get("dex") or ""),
                program_id=str(leg.get("programId") or ""),
                pool=str(leg.get("pool") or ""),
                input_mint=str(leg.get("inputMint") or ""),
                output_mint=str(leg.get("outputMint") or ""),
                amount_in=str(leg.get("amountIn") or ""),
                amount_out=str(leg.get("amountOut") or ""),
                percent=_to_float(leg.get("percent"), 0.0),
                price_impact_pct=_to_float(leg.get("priceImpact"), 0.0) * 100.0,
            )
            steps.append(step)
            if step.dex:
                labels.append(step.dex)

        return Quote(
            input_mint=input_mint,
            output_mint=output_mint,
            in_amount=in_amount,
            expected_out_amount=out_amount,
            min_out_amount=min_out,
            price_impact_pct=impact,
            route_labels=labels,
            route_steps=steps,
            slippage_bps=int(body.get("slippageBps") or slippage_bps),
            quote_created_at=time.time(),
            context_slot=_to_int(body.get("contextSlot")),
            swap_usd_value=(str(body["swapUsdValue"])
                            if body.get("swapUsdValue") is not None else None),
            raw=body,
            side=side,
        )

    # -- build -------------------------------------------------------------
    async def build(
        self,
        quote: Quote,
        user_public_key: str,
        wrap_and_unwrap_sol: bool = True,
        priority_fee: Any = "auto",
        only_direct_routes: bool | None = None,
        dynamic_compute_unit_limit: bool | None = None,
        tx_version: str = RAPTOR_TX_VERSION,
    ) -> BuiltSwap:
        """Ask Raptor for an unsigned transaction for this quote.

        The signature accepts the same keyword set the Jupiter path's planner
        uses, so the two providers are interchangeable behind one interface, but
        the flags that only change Jupiter's transaction shape are ignored:
        Raptor owns route selection.

        ``priority_fee`` may be an :class:`int`/numeric-string (a target total
        priority-fee budget in lamports, sent as a string as the API requires)
        or a level name (``"medium"`` etc.). It is never silently dropped and
        never raised.
        """
        if not user_public_key:
            raise RaptorBuildFailed("user public key is required to build a swap")

        payload: dict[str, Any] = {
            "userPublicKey": user_public_key,
            "quoteResponse": quote.raw,
            "wrapUnwrapSol": bool(wrap_and_unwrap_sol),
            "txVersion": tx_version,
        }
        pf = self._priority_fee_payload(priority_fee)
        if pf is not None:
            payload["priorityFee"] = pf
        if only_direct_routes is not None:
            payload["onlyDirectRoutes"] = bool(only_direct_routes)

        body = await self._request("POST", "/swap", json=payload, kind="build")
        tx_b64 = body.get("swapTransaction")
        if not isinstance(tx_b64, str) or not tx_b64:
            raise RaptorInvalidResponse(
                "raptor did not return a swapTransaction to sign")

        # Decode now so a malformed payload is rejected before it can reach the
        # signer, and so an already-signed payload is caught explicitly.
        try:
            slots, unsigned = _looks_unsigned(tx_b64)
        except Exception as exc:  # noqa: BLE001
            raise RaptorInvalidResponse(
                "raptor swapTransaction could not be deserialized"
            ) from exc
        if not unsigned:
            raise RaptorInvalidResponse(
                "raptor returned a signed transaction; refusing to use it "
                "(the bot must be the only signer)")

        return BuiltSwap(
            transaction_b64=tx_b64,
            last_valid_block_height=_to_int(body.get("lastValidBlockHeight")),
            prioritization_fee_lamports=_to_int(
                body.get("prioritizationFeeLamports")),
            quote=quote,
            context_slot=_to_int(body.get("contextSlot")),
            tx_version=tx_version,
            signature_slots=slots,
            unsigned=unsigned,
        )

    @staticmethod
    def _priority_fee_payload(priority_fee: Any) -> Optional[str]:
        """Normalise a priority fee to Raptor's expected **string** form.

        * ``None`` / ``"auto"``          -> ``"auto"``
        * a level name                   -> the name (``"medium"`` ...)
        * an int / numeric string        -> that number as a string, which the
          live API treats as a target *total* priority-fee budget in lamports
          (verified: ``"50000"`` -> ``prioritizationFeeLamports: 50000``). The
          bare integer form is rejected by the API with HTTP 422, so it is
          always stringified here.
        """
        if priority_fee is None:
            return "auto"
        if isinstance(priority_fee, str):
            text = priority_fee.strip()
            if not text or text.lower() == "auto":
                return "auto"
            return text
        # int / float
        return str(int(priority_fee))

    # -- unified provider interface ---------------------------------------
    def provider_name(self) -> str:
        return "raptor"

    async def build_transaction(
        self,
        quote: Quote,
        user_public_key: str,
        *,
        wrap_and_unwrap_sol: bool = True,
        priority_fee: Any = "auto",
        only_direct_routes: Optional[bool] = None,
        dynamic_compute_unit_limit: Optional[bool] = None,
        tx_version: Optional[str] = None,
    ) -> BuiltSwap:
        """Unified alias of :meth:`build`.

        ``dynamic_compute_unit_limit`` is accepted for interface parity and
        ignored: Raptor sizes compute units itself.
        """
        return await self.build(
            quote, user_public_key,
            wrap_and_unwrap_sol=wrap_and_unwrap_sol,
            priority_fee=priority_fee,
            only_direct_routes=only_direct_routes,
            tx_version=tx_version or RAPTOR_TX_VERSION,
        )

    def route_metadata(self, quote: Quote) -> dict:
        """Sanitized route projection. No transaction payload."""
        pools = [s.pool for s in quote.route_steps if s.pool]
        return {
            "provider": "raptor",
            "route_labels": list(quote.route_labels),
            "route_hops": len(quote.route_steps),
            "selected_dex": quote.selected_dex,
            "pools": pools,
            "context_slot": quote.context_slot,
        }

    # -- health ------------------------------------------------------------
    async def health(self) -> dict:
        """Readiness probe. Reaches the venue; never raises.

        A real WSOL -> USDC quote for 0.001 SOL: read-only, free, and the same
        code path a trade takes.
        """
        try:
            body = await self._request(
                "GET", "/quote",
                params={
                    "inputMint": WSOL_MINT,
                    "outputMint": USDC_MINT,
                    "amount": str(1_000_000),
                    "slippageBps": "50",
                },
                kind="quote",
            )
        except RaptorError as exc:
            return {"provider": "raptor", "base_url": self.base_url,
                    "reachable": False,
                    "detail": f"{exc.status}: {type(exc).__name__}"}
        except Exception as exc:  # noqa: BLE001
            return {"provider": "raptor", "base_url": self.base_url,
                    "reachable": False, "detail": type(exc).__name__}
        out_amount = body.get("amountOut")
        if not out_amount:
            return {"provider": "raptor", "base_url": self.base_url,
                    "reachable": False,
                    "detail": "quote probe returned no amountOut"}
        return {
            "provider": "raptor", "base_url": self.base_url,
            "reachable": True,
            "detail": f"quote probe ok (0.001 SOL -> {out_amount} USDC units)",
        }


def transaction_size_report(transaction_b64: str) -> dict:
    """Measure a serialized transaction against the limit for ITS OWN VERSION.

    Deserialize -> identify the real version -> reserialize -> measure -> compare
    against that version's ceiling. No RPC call, no signature, no broadcast.

    This used to apply one unconditional 1232-byte limit, which is correct for
    legacy and v0 and wrong for v1 (4096 bytes). The version is read from the
    deserialized message, never inferred from the fact that Raptor was the
    provider: Raptor builds v0 today, but a provider returning a v1 transaction
    must be judged as one, and a provider returning an oversized v0 must still
    be refused.

    Returns a structured verdict carrying ``transaction_version``, ``raw_bytes``,
    ``max_bytes`` and ``over_by``; the caller decides what to do with it.
    """
    from .txcodec import measure_transaction

    return measure_transaction(transaction_b64)
