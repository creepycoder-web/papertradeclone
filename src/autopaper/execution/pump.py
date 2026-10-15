"""Direct, venue-identified executable quoting for the two Pump venues.

WHAT "DIRECT" IS AND IS NOT HERE
--------------------------------
A trade is only executable against a specific pool. On 2026-10-02 a $GOAT entry
and its exit filled on two different pools under two different programs, the
system's market mark described a third address, and nothing noticed. The fix is
not "add a price source" -- it is "make the price source say which pool it is
about, and refuse when it is about somewhere else".

That is what this module provides. Each provider pins ONE venue, asks for a
quote for that venue, and then PROVES the quote is about the pool it claims:

* the returned route's hops must all belong to the pinned venue, and
* where the provider returns a pool/AMM address, that account is read from the
  free public RPC and its OWNING PROGRAM must be the pinned program.

If either check fails, the provider raises. It does not return a quote "close
enough". A price that cannot be attributed to a pool is not an executable
price, and the $GOAT incident is exactly what happens when one is used as if it
were.

WHY THE CHAIN IS USED AS AN AUTHORITY AND NOT AS A PRICE SOURCE
---------------------------------------------------------------
Deriving a venue's price by re-implementing its reserve math from raw account
bytes would look more "direct", and it would be wrong. Measured on the $GOAT
exit: the pool's wSOL vault held 52.02 SOL against 252,813,974 tokens, a ratio
of 205.8 lamports per token, while the trade actually settled at 271.9 lamports
per token -- 32% BETTER than the vault ratio on a SELL, which a constant-product
pool cannot produce. The vault balance is therefore not the effective reserve,
and a model built on it would have published a confidently wrong price. This
module does not build one. It uses the chain to establish IDENTITY and lets the
venue answer the PRICE.

THE FREE-TIER RULE
------------------
Every provider here runs on the free public Solana RPC plus an aggregator's
public quote endpoint. Helius, Birdeye and every paid RPC/data tier are
explicitly NOT required, and no code path in this module imports or configures
one. A free tier has a free tier's limits, so every read is cached, deduplicated,
rate-limited and retried by :mod:`autopaper.execution.chainstate`, and every
venue-scoped read fails CLOSED.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from .chainstate import PublicRpcClient
from .price_layers import (
    ExecutableQuote,
    RouteHop,
    Side,
    leg_for_buy,
    leg_for_sell,
)
from .quote_errors import (
    ChainStateUnavailable,
    QuoteRejected,
    QuoteUnavailable,
)

log = logging.getLogger("execution.pump")

__all__ = [
    "PUMP_BONDING_CURVE_PROGRAM_ID",
    "PUMP_SWAP_PROGRAM_ID",
    "PumpBondingCurveQuoteProvider",
    "PumpSwapQuoteProvider",
    "VenueIdentity",
    "VenueScopedQuoteProvider",
]

#: The Pump.fun bonding-curve program. Identified in this repository's own
#: 2026-10-02 price-integrity audit as the Pump.fun bonding-curve venue.
PUMP_BONDING_CURVE_PROGRAM_ID = (
    "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")

#: The PumpSwap (pAMM) program -- the venue a token is migrated to.
PUMP_SWAP_PROGRAM_ID = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"

#: The Pump AMM (pumpfun) program id. Distinct from the bonding curve above: a
#: migrated coin is tradable on the AMM, not on the curve.
PUMP_AMM_PROGRAM_ID = PUMP_BONDING_CURVE_PROGRAM_ID


@dataclass(frozen=True)
class VenueIdentity:
    """
    Which venue, by program, and which human labels count as it.

    ``labels`` are what an aggregator's free quote endpoint reports. They are
    NOT authoritative -- a label is a string somebody chose -- which is exactly
    why the owning-program read below exists.
    """

    key: str
    program_ids: tuple
    labels: tuple
    description: str

    def matches_label(self, label: str) -> bool:
        return (label or "").strip().lower() in self.labels


#: The Pump.fun bonding curve.
PUMP_BONDING_CURVE = VenueIdentity(
    key="pump_bonding_curve",
    program_ids=(PUMP_BONDING_CURVE_PROGRAM_ID,),
    labels=("pumpfun", "pump fun", "pump bonding curve", "bonding curve"),
    description="Pump.fun bonding curve (pre-migration venue)",
)

#: The PumpSwap AMM, where a graduated coin is traded.
PUMP_SWAP = VenueIdentity(
    key="pumpswap",
    program_ids=(PUMP_SWAP_PROGRAM_ID,),
    labels=("pumpswap", "pump swap"),
    description="PumpSwap AMM (post-migration venue)",
)


class VenueScopedQuoteProvider:
    """
    Base class: an executable quote pinned to exactly one venue.

    Subclasses supply a :class:`VenueIdentity`. The behaviour lives here so the
    two Pump venues cannot drift apart -- a bug fixed for one is fixed for both.

    Construction is inert: no client, no socket, no key. The aggregator client
    and the chain reader are both injectable, which is how every test in this
    repository exercises the whole path with no network.
    """

    venue_identity: VenueIdentity = PUMP_BONDING_CURVE

    def __init__(
        self,
        inner: Any,
        *,
        chain: Optional[PublicRpcClient] = None,
        require_program_attestation: bool = True,
        name: Optional[str] = None,
    ) -> None:
        self.inner = inner
        self.chain = chain
        self.require_program_attestation = bool(require_program_attestation)
        self.name = name or self.venue_identity.key
        self.supports_venues = (self.venue_identity.key,)
        self.venue_quoted = True

    # -- introspection -----------------------------------------------------

    def as_dict(self) -> dict:
        return {
            "provider": self.name,
            "venue": self.venue_identity.key,
            "program_ids": list(self.venue_identity.program_ids),
            "labels": list(self.venue_identity.labels),
            "venue_quoted": self.venue_quoted,
            "program_attestation_required": self.require_program_attestation,
            "description": self.venue_identity.description,
        }

    # -- the two directions ------------------------------------------------

    async def quote_buy(self, *, token_mint: str, quote_mint: str,
                        quote_amount_raw: int, token_decimals: int,
                        quote_decimals: Optional[int] = None,
                        slippage_bps: int = 500,
                        quote_asset_usd: Optional[float] = None,
                        ) -> ExecutableQuote:
        """``quote_mint -> token_mint``, on the pinned venue, always."""
        side = Side.BUY
        inner = await self.inner.quote(quote_mint, token_mint,
                                       int(quote_amount_raw), int(slippage_bps))
        out_amount = int(getattr(inner, "expected_out_amount", 0) or 0)
        if out_amount <= 0:
            raise QuoteRejected(
                f"{self.name}: venue returned a non-positive output amount")
        leg = leg_for_buy(
            token_mint=token_mint, quote_mint=quote_mint,
            quote_amount=int(quote_amount_raw), quote_decimals=quote_decimals,
            token_decimals=token_decimals, expected_token_amount=out_amount,
        )
        return await self._finalise(
            quote=inner, side=side, leg=leg,
            in_mint=quote_mint, out_mint=token_mint,
            slippage_bps=int(slippage_bps), quote_asset_usd=quote_asset_usd,
            out_unit="tokens",
        )

    async def quote_sell(self, *, token_mint: str, quote_mint: str,
                         token_amount_raw: int, token_decimals: int,
                         quote_decimals: Optional[int] = None,
                         slippage_bps: int = 500,
                         quote_asset_usd: Optional[float] = None,
                         ) -> ExecutableQuote:
        """``token_mint -> quote_mint``, on the pinned venue, always."""
        side = Side.SELL
        inner = await self.inner.quote(token_mint, quote_mint,
                                       int(token_amount_raw), int(slippage_bps))
        out_amount = int(getattr(inner, "expected_out_amount", 0) or 0)
        if out_amount <= 0:
            raise QuoteRejected(
                f"{self.name}: venue returned a non-positive output amount")
        leg = leg_for_sell(
            token_mint=token_mint, quote_mint=quote_mint,
            token_amount=int(token_amount_raw), quote_decimals=quote_decimals,
            token_decimals=token_decimals, expected_quote_amount=out_amount,
        )
        return await self._finalise(
            quote=inner, side=side, leg=leg,
            in_mint=token_mint, out_mint=quote_mint,
            slippage_bps=int(slippage_bps), quote_asset_usd=quote_asset_usd,
            out_unit="lamports",
        )

    # -- venue attestation -------------------------------------------------

    async def _finalise(self, *, quote: Any, side: Side, leg: Any,
                        in_mint: str, out_mint: str, slippage_bps: int,
                        quote_asset_usd: Optional[float], out_unit: str,
                        ) -> ExecutableQuote:
        import time

        hops = _route_hops(quote)
        self._assert_route_is_pinned(quote, hops)

        pool = _first_pool(hops)
        program_id = _first_program(hops)
        attested = None
        if pool and self.chain is not None and self.require_program_attestation:
            attested = await self._attest_pool_program(pool)
            if attested not in self.venue_identity.program_ids:
                raise ChainStateUnavailable(
                    f"{self.name}: pool {pool} is owned by {attested}, not by "
                    f"{'/'.join(self.venue_identity.program_ids)}; the quote is "
                    f"about a different venue and is refused rather than "
                    f"relabelled")
        elif pool and self.require_program_attestation and self.chain is None:
            raise ChainStateUnavailable(
                f"{self.name}: no chain reader is configured, so pool {pool} "
                f"cannot be attested to this venue; refusing rather than "
                f"trusting an unverified pool identity")

        now = time.time()
        return ExecutableQuote(
            side=side, leg=leg, provider=self.name,
            venue=self.venue_identity.key, pool=pool, program_id=program_id,
            route=tuple(hops),
            price_impact_pct=float(getattr(quote, "price_impact_pct", 0.0) or 0.0),
            slippage_bps=slippage_bps,
            min_output_amount=(int(getattr(quote, "min_out_amount", 0) or 0)
                               or None),
            quoted_at=float(getattr(quote, "quote_created_at", 0.0) or 0.0),
            received_at=now,
            context_slot=getattr(quote, "context_slot", None),
            quote_asset_usd=quote_asset_usd,
            venue_quoted=True,
            model=None,
            stale=False, quality="live",
            provenance={
                "inner_provider": getattr(self.inner, "provider_name",
                                          lambda: "unknown")(),
                "venue_identity": self.venue_identity.key,
                "program_attested": attested,
                "out_unit": out_unit,
                "route_labels": list(getattr(quote, "route_labels", []) or []),
            },
        )

    def _assert_route_is_pinned(self, quote: Any, hops: list[RouteHop]) -> None:
        """
        Every hop must belong to the pinned venue.

        An empty route is allowed and means the provider did not itemise one;
        the pool attestation, when a chain reader is configured, is then the only
        identity evidence and it must pass. A route that itemises hops on a
        DIFFERENT venue is refused -- silently accepting it is precisely how the
        $GOAT exit ended up on a pool nobody was marking.
        """
        if not hops:
            return
        for hop in hops:
            if hop.program_id:
                if hop.program_id in self.venue_identity.program_ids:
                    continue
                raise ChainStateUnavailable(
                    f"{self.name}: route hop {hop.venue or hop.program_id} runs "
                    f"on program {hop.program_id}, which is not this venue")
            if hop.venue and self.venue_identity.matches_label(hop.venue):
                continue
            if hop.venue:
                raise ChainStateUnavailable(
                    f"{self.name}: route hop venue {hop.venue!r} is not this "
                    f"venue ({'/'.join(self.venue_identity.labels)})")

    async def _attest_pool_program(self, pool: str) -> Optional[str]:
        """The pool account's owning program, read from the free public RPC."""
        read = await self.chain.get_account(pool)  # type: ignore[union-attr]
        if read.missing or not read.ok:
            raise ChainStateUnavailable(
                f"{self.name}: could not read pool account {pool} from the "
                f"public RPC ({read.error or 'missing'}); refusing to price a "
                f"pool whose venue cannot be established")
        return (read.value or {}).get("owner")


class PumpBondingCurveQuoteProvider(VenueScopedQuoteProvider):
    """Executable BUY/SELL quoting for a coin still on the Pump.fun curve."""

    venue_identity = PUMP_BONDING_CURVE

    def __init__(self, inner: Any, **kwargs) -> None:
        kwargs.setdefault("name", "pump_bonding_curve")
        super().__init__(inner, **kwargs)


class PumpSwapQuoteProvider(VenueScopedQuoteProvider):
    """Executable BUY/SELL quoting for a graduated coin on PumpSwap."""

    venue_identity = PUMP_SWAP

    def __init__(self, inner: Any, **kwargs) -> None:
        kwargs.setdefault("name", "pumpswap")
        super().__init__(inner, **kwargs)


# ---------------------------------------------------------------------------
# route extraction
# ---------------------------------------------------------------------------


def _route_hops(quote: Any) -> list[RouteHop]:
    """
    Normalise whatever route shape a provider returned into :class:`RouteHop`.

    Raptor exposes ``route_steps`` with a program id per hop; Jupiter's free
    quote endpoint exposes ``routePlan[].swapInfo`` with a label and an
    ``ammKey`` and no program id. Both are read. Neither is trusted on its own:
    the program attestation above is what decides.
    """
    hops: list[RouteHop] = []
    for step in (getattr(quote, "route_steps", None) or []):
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
            # A provider's OWN step type (Raptor has its own RouteStep
            # dataclass). Duck-typed rather than isinstance-matched: an exact
            # type match silently DROPS the route, and a dropped route means the
            # venue check below has nothing to check and the quote is relabelled
            # as belonging to this venue. That is the failure this whole helper
            # exists to make impossible.
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
    for leg in ((getattr(quote, "raw", None) or {}).get("routePlan") or []):
        info = leg.get("swapInfo") or {}
        if not isinstance(info, dict):
            continue
        hops.append(RouteHop(
            venue=str(info.get("label") or ""),
            program_id=str(info.get("programId") or info.get("program_id") or ""),
            pool=str(info.get("ammKey") or ""),
            input_mint=str(info.get("inputMint") or ""),
            output_mint=str(info.get("outputMint") or ""),
            amount_in=str(info.get("inAmount") or ""),
            amount_out=str(info.get("outAmount") or ""),
            percent=100.0 if len(leg) == 1 else 0.0,
        ))
    return hops


def _first_pool(hops: list[RouteHop]) -> Optional[str]:
    for hop in hops:
        if hop.pool:
            return hop.pool
    return None


def _first_program(hops: list[RouteHop]) -> Optional[str]:
    for hop in hops:
        if hop.program_id:
            return hop.program_id
    return None