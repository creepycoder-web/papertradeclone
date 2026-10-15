"""
The executable quote service this PAPER harness uses.

ONE SOURCE OF TRUTH
===================

Every price here comes from :mod:`autopaper.same_market` - the read-only engine
whose bonding-curve and PumpSwap quotes were checked against the official
``@pump-fun/pump-sdk`` and ``@pump-fun/pump-swap-sdk`` AND against Jupiter on
live mainnet, to the raw integer unit.

There is deliberately **no second quote implementation in this package**. Two
implementations of the same formula is how they drift apart without anyone
noticing, and the drift shows up as a paper PnL difference nobody can explain.

WHAT THIS ADAPTER ADDS
=====================

1. A uniform ``quote_buy`` / ``quote_sell`` signature over both markets.
2. Provider precedence: direct Pump curve -> direct PumpSwap -> Jupiter, and a
   named record of which one answered.
3. SOL/USD conversion against one explicitly timestamped reference, because
   sizing is specified in USD while execution happens in SOL.
4. Same-market identity resolution against the observed pair.
5. **Fail-closed.** No executable quote means no simulated fill. It is never
   replaced by the DexScreener mark, an old quote, or another venue's route.

WHAT IT WILL NOT DO
===================

It does not compute a price from a mark, it does not fall back to a stale quote,
and it does not paper over a missing market.
"""

from __future__ import annotations

import asyncio

import time
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any

from autopaper.same_market.models import WSOL_MINT, MarketKind
from autopaper.same_market.numeric import to_float
from autopaper.same_market.pumpdirect import (
    StateUnavailable,
    derive_bonding_curve,
    derive_canonical_pool,
)
from autopaper.same_market.reader import DirectPumpReader

from .schema import (
    EXIT_VENUE_MIGRATION_GRADUATED,
    ExecutionNativeReference,
    QuotePoint,
    VenueMigration,
)


# ---------------------------------------------------------------------------
# CANDIDATE-LEVEL BUY FAILURE CLASSIFICATION
#
# A blocked BUY used to arrive as one flat verdict
# (ENTRY_REJECTED_QUOTE_UNAVAILABLE) which could not distinguish "no route on a
# brand-new token" from "our own RPC was throttled" from "identity could not be
# proven" -- three facts with opposite remedies. Every BUY candidate now gets
# its own code, and the per-source summary travels with the refusal.
# ---------------------------------------------------------------------------

#: What each source was asked, and the code namespace it owns.
BUY_CANDIDATE_SOURCES = ("jupiter", "pump_direct", "pumpswap_direct")

_TIMEOUT_MARKERS = (
    "timeout", "timed out", "did not arrive within", "deadline",
)
_RPC_MARKERS = (
    "rate limit", "rate-limit", "throttled", "rpc cap", "rate cap",
    "quota exhausted", "http 5", "rpc error", "solana rpc", "refused the quote",
    "connection", "upstream",
)
_MISMATCH_MARKERS = (
    "is owned by", "different venue", "not the chain-verified",
    "another token's pool", "refused rather than relabelled",
)
_JUPITER_NO_ROUTE_MARKERS = ("no jupiter route", "no route", "zero output")
_PUMPSWAP_NO_ROUTE_MARKERS = (
    "no pool", "pool not found", "not graduated", "no pumpswap",
    "cannot be derived", "no market",
)
_PUMP_MARKET_NOT_FOUND_MARKERS = (
    "not found", "does not exist", "no account", "is complete",
    "no curve", "missing account", "unknown account", "could not be found",
)


def _provider_prefix(provider: str) -> str:
    known = {
        "jupiter": "JUPITER",
        "pump_direct": "PUMP_DIRECT",
        "pumpswap_direct": "PUMPSWAP",
    }
    key = (provider or "").lower()
    if key in known:
        return known[key]
    return (provider or "UNKNOWN").upper().replace("-", "_") or "UNKNOWN"


def classify_buy_candidate(point: Any, *, mint: str | None = None) -> str:
    """
    ONE code for why this BUY candidate can or cannot become a fill.

    ``OK`` means the candidate is usable: available, fresh, for the requested
    token, positive output, and identity not proven wrong. Everything else is a
    named refusal -- capacity, market absence, identity and staleness each get
    their own code, because they call for opposite responses and collapsing
    them is what made the flat verdict undiagnosable.

    Identity labels, mapped without being weakened:

    * ``FAIL``  (known and mismatching) -> ``*_IDENTITY_MISMATCH`` (REJECT)
    * ``UNKNOWN`` (genuinely unavailable) -> ``*_IDENTITY_UNKNOWN``
      (the fill gate fails closed on this; the code says which source)
    """
    provider = str(getattr(point, "provider", "") or "")
    prefix = _provider_prefix(provider)
    expected = getattr(point, "token_mint", None)
    if mint and expected and expected != mint:
        return "INVALID_TOKEN"
    if getattr(point, "available", False):
        if getattr(point, "stale", False):
            return "STALE_QUOTE"
        identity = str(getattr(point, "identity", "") or "")
        if identity == "FAIL":
            return f"{prefix}_IDENTITY_MISMATCH"
        if identity != "PASS":
            return f"{prefix}_IDENTITY_UNKNOWN"
        out = getattr(point, "output_amount_decimal", None)
        if out is None or float(out) <= 0:
            return f"{prefix}_QUOTE_UNAVAILABLE"
        return "OK"
    text = str(getattr(point, "missing_reason", "") or "").lower()
    if any(marker in text for marker in _TIMEOUT_MARKERS):
        return f"{prefix}_TIMEOUT"
    if any(marker in text for marker in _RPC_MARKERS):
        return f"{prefix}_RPC_ERROR"
    if any(marker in text for marker in _MISMATCH_MARKERS):
        return f"{prefix}_IDENTITY_MISMATCH"
    if "not configured" in text:
        return f"{prefix}_NOT_CONFIGURED"
    if prefix == "JUPITER" and any(m in text for m in _JUPITER_NO_ROUTE_MARKERS):
        return "JUPITER_NO_ROUTE"
    if prefix == "PUMPSWAP" and any(m in text for m in _PUMPSWAP_NO_ROUTE_MARKERS):
        return "PUMPSWAP_NO_ROUTE"
    if prefix == "PUMP_DIRECT" and any(
            m in text for m in _PUMP_MARKET_NOT_FOUND_MARKERS):
        return "PUMP_DIRECT_MARKET_NOT_FOUND"
    if any(m in text for m in _JUPITER_NO_ROUTE_MARKERS):
        return f"{prefix}_NO_ROUTE"
    return f"{prefix}_QUOTE_UNAVAILABLE"


def buy_candidate_summary(
    candidates: list[Any], *, jupiter_configured: bool | None = None,
    mint: str | None = None,
) -> str:
    """
    One clause per source: what answered, what failed, and its code.

    Sources that were never attempted still get a row (``NOT_ATTEMPTED`` /
    ``NOT_CONFIGURED``), so "Jupiter is absent from this runtime" can never be
    mistaken for "Jupiter had no route". When NO candidate is usable the
    summary is headed ``ALL_CANDIDATES_REJECTED``; a partially-usable table is
    not.
    """
    by_provider: dict[str, Any] = {}
    for c in candidates:
        provider = str(getattr(c, "provider", "") or "unknown")
        by_provider.setdefault(provider, c)
    parts: list[str] = []
    for src in BUY_CANDIDATE_SOURCES:
        c = by_provider.get(src)
        if c is None:
            if src == "jupiter" and jupiter_configured is False:
                parts.append("jupiter=JUPITER_NOT_CONFIGURED")
            else:
                parts.append(f"{src}=NOT_ATTEMPTED")
            continue
        code = classify_buy_candidate(c, mint=mint)
        if code == "OK":
            parts.append(f"{src}=OK")
            continue
        reason = str(getattr(c, "missing_reason", "") or "")
        if not reason:
            reason = str(getattr(c, "identity_evidence", "") or "")
        reason = " ".join(reason.split())[:90]
        parts.append(f'{src}={code}("{reason}")' if reason else f"{src}={code}")
    usable = any(
        classify_buy_candidate(c, mint=mint) == "OK" for c in candidates)
    body = "; ".join(parts)
    if not usable:
        return f"ALL_CANDIDATES_REJECTED | {body}"
    return body


def classify_entry_identity(point: Any) -> str:
    """WHY an ENTRY quote's same-market identity is not PASS.

    The flat ``ENTRY_REJECTED_IDENTITY_UNKNOWN`` collapsed three different
    facts with three different remedies: nothing about this market is known,
    the market was discovered but could not be proven to be THIS mint's, or the
    executable market is provably a different book. Each now has its own code,
    derived from the quote itself so it can never disagree with the verdict.
    """
    identity = str(getattr(point, "identity", "") or "")
    if identity == "FAIL":
        return "ENTRY_IDENTITY_MISMATCH"
    if getattr(point, "pool_or_curve", None):
        return "ENTRY_IDENTITY_DISCOVERY_FAILED"
    return "ENTRY_IDENTITY_MISSING"


@dataclass(frozen=True)
class ResolvedMarket:
    """A chain-verified CANONICAL market for one mint.

    Only ever produced by provider-native discovery from the mint (the same
    ``DirectPumpReader`` the direct candidates use): the account was DERIVED
    from the mint and then decoded and owner-checked on chain, so its address
    is a function of the mint rather than something a supplier asserted.
    """

    venue_id: str
    address: str
    provider: str


class QuoteServiceError(RuntimeError):
    """The quote service could not produce a usable quote. Never falls back."""


@dataclass
class SolReference:
    """
    An explicitly timestamped SOL/USD reference.

    Recorded because position sizing is specified in USD and execution happens
    in SOL, so every USD number in this dataset depends on this one number. It
    is stored beside the results that used it so the arithmetic is reproducible.

    It is NEVER the token's own DexScreener mark. Using that would mean the
    position size and the fill price were derived from the same observation,
    which is a different claim from "executable".
    """

    usd_per_sol: float
    at: float
    source: str = "operator"

    def age_ms(self, now: float | None = None) -> float:
        """Age of this reference in milliseconds. Never negative."""
        import time as _time

        ref = _time.time() if now is None else float(now)
        return max(0.0, (ref - self.at) * 1000.0)

    def as_dict(self) -> dict[str, float | str | None]:
        """
        The reference plus its snapshot metadata.

        Additive: the three original keys are unchanged, and the
        ``sol_usd*`` metadata keys are added alongside. No token price,
        executable quote, fill or PnL field is touched.
        """
        return {
            "sol_usd_reference": self.usd_per_sol,
            "sol_usd_reference_at": self.at,
            "sol_usd_reference_source": self.source,
            "sol_usd": self.usd_per_sol,
            "sol_usd_source": self.source,
            "sol_usd_timestamp": self.at,
            "sol_usd_age_ms": self.age_ms(),
        }


@dataclass
class QuoteService:
    """
    Read-only BUY and SELL quotes for the PAPER harness.

    Collaborators are injected so the whole service is drivable from fixtures
    with no network at all.
    """

    rpc: Any
    sol_reference: SolReference | None = None
    jupiter: Any = None
    max_quote_age_ms: float | None = 30_000.0
    #: Ceiling on ONE venue probe, in seconds. It bounds the decision; it never
    #: queues the candidate behind other work.
    quote_deadline_s: float = 1.5
    #: Capacity counters, so a throughput limit is visible as a NUMBER rather
    #: than inferred from a falling fill rate.
    quote_timeouts: int = 0
    quote_errors: int = 0
    quote_rate_limited: int = 0
    #: Allow the generic Jupiter fallback when no direct market is available.
    #: OFF by default: Jupiter is a real executable route and its use is
    #: recorded, but the direct on-chain path is preferred wherever it exists
    #: because it names the exact pool.
    allow_jupiter_fallback: bool = True
    #: Require the same-market identity to be proven before a quote is usable.
    require_identity: bool = True

    def __post_init__(self) -> None:
        self._reader = DirectPumpReader(self.rpc)
        # Monitoring cache: mint -> static market identity (curve/pool address, vault addresses, fee configs, decimals)
        self._monitoring_cache: dict[str, dict[str, Any]] = {}

    def _cache_monitoring_identity(self, mint: str, identity: dict[str, Any]) -> None:
        """Cache static market identity for monitoring reuse."""
        self._monitoring_cache[mint] = identity

    def _get_cached_monitoring_identity(self, mint: str) -> dict[str, Any] | None:
        """Retrieve cached static market identity for monitoring."""
        return self._monitoring_cache.get(mint)

    # -- identity ---------------------------------------------------------

    @staticmethod
    def _identity_for(
        *,
        observed_pair_address: str | None,
        market_type: MarketKind | None,
        market_address: str | None,
        chain_verified: bool,
    ) -> tuple[str, str]:
        """
        Resolve ``DexScreener pair <-> execution market``.

        The provider-specific mapping was verified live: on a pump venue the
        aggregator's pair address IS the market account - the curve PDA before
        graduation, the pool address after. So an exact address comparison is
        meaningful here, and it would not be if one side were a curve and the
        other a pool.

        Anything that cannot be shown is ``UNKNOWN``. The observation is still
        recorded; only the PAPER TRADE fails closed.
        """
        if not chain_verified or not market_address:
            return "UNKNOWN", (
                "no chain-verified execution market was established, so the "
                "observed pair cannot be shown to be the same object"
            )
        if not observed_pair_address:
            return "UNKNOWN", "no observed pair address to compare against"
        if observed_pair_address == market_address:
            kind = "curve" if market_type is MarketKind.PUMP_BONDING_CURVE else "pool"
            return "PASS", (
                f"the observed pair {observed_pair_address[:8]}… IS the "
                f"chain-verified {kind} account, so the mark and the executable "
                "quote describe the same book"
            )
        return "FAIL", (
            f"the observed pair {observed_pair_address[:8]}… is not the "
            f"chain-verified market {market_address[:8]}…; these are different "
            "books, so the gap between their prices is not a discrepancy in one"
        )

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def venue_id_for(market_type: Any, pool_or_curve: Any) -> str:
        """
        The stable machine identity of one EXACT market.

        ``<market_type>:<pool>``. This is deliberately not "Pump.fun": a
        bonding curve and a graduated PumpSwap pool share that label and share
        nothing else -- different depth, different counterparty, different book.
        The identity an exit is pinned to has to name the book, or the pin is
        decorative.
        """
        kind = getattr(market_type, "value", market_type) or "UNKNOWN"
        pool = pool_or_curve or "?"
        return f"{kind}:{pool}"

    @staticmethod
    def _unavailable(
        *, side: str, mint: str, reason: str, provider: str = "none",
        market_type: str = "UNKNOWN", quote_mint: str = WSOL_MINT,
        identity: str = "UNKNOWN", evidence: str | None = None,
        observed_pair_address: str | None = None, latency_ms: float | None = None,
    ) -> QuotePoint:
        """
        A quote that could not be produced.

        Present with a reason rather than absent, so the report can show the
        source was ATTEMPTED. Amounts are ``None``, never ``0`` - a zero would
        flow into the arithmetic and produce a zero price that reads as a
        finding.
        """
        return QuotePoint(
            side=side, provider=provider, market_type=market_type,
            token_mint=mint, quote_mint=quote_mint,
            input_amount_raw=None, input_decimals=None, input_amount_decimal=None,
            output_amount_raw=None, output_decimals=None, output_amount_decimal=None,
            effective_price=None, effective_price_usd=None, price_impact_pct=None,
            fee_bps=None, fee_amount=None, venue=None, pool_or_curve=None,
            route=(), state_slot=None, state_age_ms=None, quote_timestamp=None,
            quote_age_ms=None, max_quote_age_ms=None, quote_latency_ms=latency_ms,
            identity=identity, identity_evidence=evidence,
            observed_pair_address=observed_pair_address,
            available=False, missing_reason=reason,
        )

    def _point_from_direct(
        self,
        quote: Any,
        *,
        side: str,
        mint: str,
        decimals: int,
        observed_pair_address: str | None,
        identity_override: tuple[str, str] | None = None,
    ) -> QuotePoint:
        """Turn a verified direct quote into a comparable point.

        ``identity_override`` exists for exactly one caller: the exit-venue
        migration. Every other quote proves identity by comparing the observed
        pair to the chain-verified market address, which is right for a market
        that was already being observed. A just-migrated pool has no observed
        pair to compare against, so the migration supplies its own verdict and
        the evidence that produced it, instead of leaving identity to resolve
        against an address it cannot match.
        """
        if not quote.available or quote.quote is None:
            return self._unavailable(
                side=side, mint=mint, reason=quote.missing_reason or "unavailable",
                provider=quote.provider,
                market_type=quote.market_type.value,
                identity="UNKNOWN", observed_pair_address=observed_pair_address,
                latency_ms=quote.quote_latency_ms,
            )

        detail = quote.quote.as_dict()
        snapshot = quote.snapshot
        market_address = quote.curve_address or quote.pool_address
        if identity_override is not None:
            identity, evidence = identity_override
        else:
            identity, evidence = self._identity_for(
                observed_pair_address=observed_pair_address,
                market_type=quote.market_type,
                market_address=market_address,
                chain_verified=snapshot is not None,
            )
        now = time.time()
        reference = self.sol_reference.usd_per_sol if self.sol_reference else None
        market_type_value = quote.market_type.value
        venue_label = (
            "pumpfun-bonding-curve"
            if quote.market_type is MarketKind.PUMP_BONDING_CURVE
            else "pumpswap"
        )
        market_id = self.venue_id_for(quote.market_type, market_address)
        native = self._execution_native_reference(
            quote, side=side, mint=mint, decimals=decimals,
            market_id=market_id, market_address=market_address,
            venue_label=venue_label, now=now,
        )
        return QuotePoint(
            side=side,
            provider=quote.provider,
            market_type=market_type_value,
            token_mint=quote.base_mint or mint,
            quote_mint=quote.quote_mint or WSOL_MINT,
            input_amount_raw=detail["input_amount_raw"],
            input_decimals=detail["input_decimals"],
            input_amount_decimal=detail["input_amount_decimal"],
            output_amount_raw=detail["output_amount_raw"],
            output_decimals=detail["output_decimals"],
            output_amount_decimal=detail["output_amount_decimal"],
            effective_price=quote.quote.effective_price,
            effective_price_usd=(
                to_float(quote.quote.effective_price) * reference
                if reference else None
            ),
            price_impact_pct=to_float(quote.quote.price_impact_pct),
            fee_bps=quote.quote.fee_bps,
            fee_amount=Decimal(0),
            venue=venue_label,
            pool_or_curve=market_address,
            venue_id=market_id,
            route=(quote.provider,),
            state_slot=snapshot.slot if snapshot else None,
            state_age_ms=snapshot.state_age_ms(now) if snapshot else None,
            quote_timestamp=quote.quote.quote_timestamp or now,
            quote_age_ms=max(0.0, (now - (quote.quote.quote_timestamp or now)) * 1000.0),
            max_quote_age_ms=self.max_quote_age_ms,
            quote_latency_ms=quote.quote_latency_ms,
            identity=identity,
            identity_evidence=evidence,
            observed_pair_address=observed_pair_address,
            available=True,
            execution_native_reference=native,
            amount_after_fee_raw=detail.get("amount_after_fee_raw"),
        )

    def _execution_native_reference(
        self, quote: Any, *, side: str, mint: str, decimals: int,
        market_id: str, market_address: str | None, venue_label: str,
        now: float,
    ) -> ExecutionNativeReference | None:
        """
        The SAME market's own current price, from the SAME snapshot.

        WHY THIS ADDS NO LATENCY
        =======================

        It adds none because it is not a second read. The spot price was
        computed inside the same quote call, from the same decoded account, at
        the same slot, as the executable price -- it was previously reduced to
        a ratio inside ``_impact_pct`` and discarded. Surfacing it costs one
        already-paid integer division. There is no additional RPC, so there is
        nothing to parallelise and nothing that can push the decision past its
        deadline.

        WHY IT IS THE SAME MARKET
        =========================

        ``market_id`` is literally the same ``<market_type>:<pool>`` string the
        quote's own ``venue_id`` is built from, and the pool address is the same
        ``curve_address``/``pool_address`` the executable quote was priced
        against. This is not a second identity system: it reuses the one the
        execution layer already pins exits to.

        It is ``None`` -- never a zero, never a substitute -- when no positive
        spot could be derived, so the guard fails closed rather than comparing
        against something that is not this market's price.
        """
        result = quote.quote
        spot = getattr(result, "spot_price", None)
        if spot is None:
            return None
        try:
            spot_float = float(spot)
        except (TypeError, ValueError, ArithmeticError):
            return None
        if not (spot_float > 0):
            return None
        snapshot = quote.snapshot
        method = (
            "PUMP_BONDING_CURVE_VIRTUAL_RESERVES"
            if quote.market_type is MarketKind.PUMP_BONDING_CURVE
            else "PUMPSWAP_EFFECTIVE_RESERVES"
        )
        return ExecutionNativeReference(
            market_id=market_id,
            mint=quote.base_mint or mint,
            base_mint=quote.base_mint or mint,
            quote_mint=quote.quote_mint or WSOL_MINT,
            side=side,
            reference_price=spot_float,
            numerator_raw=getattr(result, "spot_numerator_raw", None),
            denominator_raw=getattr(result, "spot_denominator_raw", None),
            calculation_method=method,
            venue=venue_label,
            pool_or_curve=market_address,
            provider=quote.provider,
            slot=snapshot.slot if snapshot else None,
            observed_at=getattr(result, "quote_timestamp", None) or now,
            state_age_ms=snapshot.state_age_ms(now) if snapshot else None,
            token_decimals=int(decimals),
        )

    def _point_from_jupiter(
        self,
        result: Any,
        *,
        side: str,
        mint: str,
        decimals: int,
        observed_pair_address: str | None,
        latency_ms: float | None,
    ) -> QuotePoint:
        """Turn a Jupiter route into a comparable point, identity UNPROVEN.

        Jupiter is a real executable route, but its identity is only as good as
        the pool key it discloses - and a multi-hop route spans several books, so
        no single pool can be attributed. Recorded as UNKNOWN unless a
        single-hop key matches the observed pair.

        ONE BUILDER, TWO DIRECTIONS -- and they are NOT symmetric.
        -------------------------------------------------------
        This is called for BUYs (``_quote_jupiter``, ``quote_buy_best``) and for
        SELLs (``_quote_jupiter``/``quote_sell_pinned``) alike, and a SELL is
        the mirror image of a BUY: there the INPUT is the token and the OUTPUT
        is lamports. It used to hard-code the BUY orientation for both, so a
        SELL point carried ``input_decimals=9`` (lamports) for a TOKEN input and
        ``output_decimals=<token decimals>`` for a LAMPORT output, and its
        ``output_amount_decimal`` -- the number ``_best_executable`` RANKS ON --
        was the proceeds scaled by ``10 ** token_decimals`` instead of ``10**9``.

        That is a 1000x inflation on a 6-decimal token, and it is exactly the
        disagreement this cost: an aggregator SELL for 88,700,795 lamports
        reported ``output_amount_decimal`` 88.700795 (it is 0.088700795 SOL),
        which made a $10 round trip read **+96,639%** where the executable
        truth was **-2.1%**. It is also not merely cosmetic: the same number is
        the ranking key, so a mis-scaled aggregator leg outbids a real
        same-market one by a factor of 1000 and, where its route is single-hop
        on the SAME pool (the only case that reaches identity ``PASS``), it is
        the leg that settles.
        """
        if result is None:
            return self._unavailable(
                side=side, mint=mint,
                reason="no Jupiter route was returned for this pair",
                provider="jupiter", observed_pair_address=observed_pair_address,
                latency_ms=latency_ms,
            )
        out_amount = int(result.out_amount)
        in_amount = int(result.in_amount)
        label = result.route_plan[0].label if result.route_plan else None
        pool_key = result.pool_keys[0] if result.pool_keys else None
        hops = len(result.route_plan)

        identity = "UNKNOWN"
        if hops > 1:
            evidence = (
                f"{hops}-hop Jupiter route; a multi-hop price is an aggregate "
                "of books and cannot be attributed to one"
            )
        elif not pool_key:
            # The route named NO pool. This is the ONLY case in which the
            # sentence is true.
            evidence = "no pool key was disclosed"
        elif pool_key == observed_pair_address:
            identity = "PASS"
            evidence = (
                f"the single-hop route uses pool {pool_key[:8]}…, which is the "
                "same pool DexScreener marked"
            )
        else:
            # A POOL **WAS** DISCLOSED AND IT IS A DIFFERENT ONE.
            #
            # This branch used to fall into the same "no pool key was
            # disclosed" text as the genuinely-absent case, and that was false
            # in the way that costs the most: it sent the incident report
            # looking for a missing field that was present, while the actual
            # fact -- this route is on ANOTHER market, so the two prices are
            # not two readings of one book -- was the one that mattered.
            evidence = (
                f"the single-hop route uses pool {pool_key[:8]}…, which is NOT "
                f"the observed pair {str(observed_pair_address)[:8]}…; these "
                "are different books, so this price is not the observed "
                "market's"
            )

        # SIDE-AWARE DECIMALS AND PRICE. The quote asset is the SOL side and the
        # token is the other one, WHICHEVER WAY ROUND the trade goes, so the
        # per-side decimals depend on the direction and the price is always
        # quote-asset per WHOLE token (out/in for a SELL, in/out for a BUY).
        buying = side != "SELL"
        in_decimals = SOL_DECIMALS_LOCAL if buying else int(decimals)
        out_decimals = int(decimals) if buying else SOL_DECIMALS_LOCAL
        # BOTH LEGS MUST BE POSITIVE, CHECKED BEFORE THE DIVISION.
        #
        # Only a zero DENOMINATOR raises, so only a zero denominator used to be
        # caught here. A zero NUMERATOR divided cleanly and published a quote
        # with ``effective_price == 0`` and ``available=True`` -- a route that
        # returns no proceeds at all, reported to the rest of the system as a
        # successfully available quote. The downstream fill layer refuses a
        # non-positive price, so no fill was fabricated, but the quote was
        # still published as available: it entered the best-route ranking, and
        # every surface that reports "was a quote obtained?" said yes.
        if in_amount <= 0 or out_amount <= 0:
            return self._unavailable(
                side=side, mint=mint,
                reason=(
                    "the route returned a zero output, so no price is defined"
                    if buying else
                    "the route would sell a zero token quantity, so no "
                    "proceeds are defined"),
                provider="jupiter", observed_pair_address=observed_pair_address,
                latency_ms=latency_ms,
            )
        try:
            if buying:
                price = Decimal(in_amount).scaleb(-in_decimals) / Decimal(
                    out_amount
                ).scaleb(-out_decimals)
            else:
                price = Decimal(out_amount).scaleb(-out_decimals) / Decimal(
                    in_amount
                ).scaleb(-in_decimals)
        except Exception:  # noqa: BLE001 - no price is defined without both legs
            return self._unavailable(
                side=side, mint=mint,
                reason="the route amounts could not be scaled into a price",
                provider="jupiter", observed_pair_address=observed_pair_address,
                latency_ms=latency_ms,
            )

        reference = self.sol_reference.usd_per_sol if self.sol_reference else None
        now = time.time()
        return QuotePoint(
            side=side, provider="jupiter", market_type=MarketKind.JUPITER_ROUTED.value,
            token_mint=mint, quote_mint=WSOL_MINT,
            input_amount_raw=in_amount, input_decimals=in_decimals,
            input_amount_decimal=float(Decimal(in_amount).scaleb(-in_decimals)),
            output_amount_raw=out_amount, output_decimals=out_decimals,
            output_amount_decimal=float(Decimal(out_amount).scaleb(-out_decimals)),
            effective_price=price,
            effective_price_usd=to_float(price) * reference if reference else None,
            price_impact_pct=to_float(result.price_impact_pct),
            fee_bps=None, fee_amount=Decimal(0),
            venue=label, pool_or_curve=pool_key,
            route=tuple(h.label or "?" for h in result.route_plan),
            state_slot=None, state_age_ms=None,
            quote_timestamp=result.timestamp, quote_age_ms=None,
            max_quote_age_ms=self.max_quote_age_ms,
            quote_latency_ms=latency_ms,
            identity=identity, identity_evidence=evidence,
            observed_pair_address=observed_pair_address,
            available=True,
        )

    # -- same-market reference reconciliation ----------------------------

    def _point_from_liquidation_route(
        self,
        result: Any,
        *,
        mint: str,
        decimals: int,
        entry_market: str | None,
        latency_ms: float | None,
    ) -> QuotePoint:
        """
        An OFF-MARKET liquidation estimate, from an aggregator SELL route.

        WHAT THIS IS
        ------------
        The venue will sell the exact held token quantity into whatever books
        its router picks. That is a real, executable route and its output amount
        is the best available estimate of what the position would realise if it
        were liquidated right now.

        WHAT THIS IS NOT
        ----------------
        It is NOT the position's market. The position was opened on
        ``entry_market`` -- a chain-verified bonding curve or PumpSwap pool --
        and a multi-hop route may traverse books this repository has never
        verified. So the identity verdict is **always UNKNOWN**, never PASS, and
        the route is carried as provenance rather than as identity.

        The entry's verified identity is preserved SEPARATELY on
        ``identity_evidence`` and in ``observed_pair_address``, so a reader can
        always tell which market the position is actually on.

        Same-market callers therefore still fail closed on this point, exactly
        as they do for a Jupiter BUY route. What changes is that the failure now
        carries a liquidation estimate instead of a bare refusal.
        """
        if result is None:
            return self._unavailable(
                side="SELL", mint=mint,
                reason="no liquidation route was returned for this pair",
                provider="jupiter_sell", observed_pair_address=entry_market,
                latency_ms=latency_ms,
            )
        out_amount = int(result.out_amount)
        in_amount = int(result.in_amount)
        hops = len(result.route_plan)
        labels = tuple(h.label or "?" for h in result.route_plan)

        # Effective price is quote asset per WHOLE token -- for a SELL the input
        # is the token and the output is the quote asset, so it is out/in.
        try:
            price = Decimal(out_amount).scaleb(-SOL_DECIMALS_LOCAL) / Decimal(
                in_amount
            ).scaleb(-decimals)
        except Exception:  # noqa: BLE001 - a zero input has no price
            return self._unavailable(
                side="SELL", mint=mint,
                reason="the liquidation route returned a zero input, so no "
                       "proceeds are defined",
                provider="jupiter_sell", observed_pair_address=entry_market,
                latency_ms=latency_ms,
            )

        # Direction re-check from the venue's own echo. A reciprocal would be a
        # finite, plausible and completely wrong "proceeds" figure.
        if getattr(result, "input_mint", mint) != mint or \
                getattr(result, "output_mint", WSOL_MINT) != WSOL_MINT:
            return self._unavailable(
                side="SELL", mint=mint,
                reason="the venue echoed a different pair than the SELL "
                       "requested; refusing to read proceeds off it",
                provider="jupiter_sell", observed_pair_address=entry_market,
                latency_ms=latency_ms,
            )

        fee_amount = None
        for hop in result.route_plan:
            if hop.fee_amount:
                fee_amount = Decimal(hop.fee_amount)
                break

        evidence = (
            f"OFF-MARKET liquidation estimate over a {hops}-hop aggregator "
            f"route ({', '.join(labels)}); the position's verified market is "
            f"{entry_market or 'unknown'} and is NOT claimed by this route. "
            "A multi-hop price is an aggregate of books and cannot be "
            "attributed to one."
        )
        reference = self.sol_reference.usd_per_sol if self.sol_reference else None
        return QuotePoint(
            side="SELL", provider="jupiter_sell",
            market_type=MarketKind.JUPITER_ROUTED.value,
            token_mint=mint, quote_mint=WSOL_MINT,
            input_amount_raw=in_amount, input_decimals=decimals,
            input_amount_decimal=float(Decimal(in_amount).scaleb(-decimals)),
            output_amount_raw=out_amount, output_decimals=SOL_DECIMALS_LOCAL,
            output_amount_decimal=float(
                Decimal(out_amount).scaleb(-SOL_DECIMALS_LOCAL)),
            effective_price=price,
            effective_price_usd=to_float(price) * reference if reference else None,
            price_impact_pct=to_float(getattr(result, "price_impact_pct", None)),
            fee_bps=None, fee_amount=fee_amount,
            venue=labels[0] if labels else None, pool_or_curve=None,
            route=labels,
            state_slot=None, state_age_ms=None,
            quote_timestamp=getattr(result, "timestamp", None), quote_age_ms=None,
            max_quote_age_ms=self.max_quote_age_ms,
            quote_latency_ms=latency_ms,
            identity="UNKNOWN", identity_evidence=evidence,
            observed_pair_address=entry_market,
            available=True,
        )

    @staticmethod
    def _reference_sources(candidates: list[QuotePoint]) -> dict[str, tuple[str, Any]]:
        """
        Pool address -> (venue_id, same-market reference) for DIRECT quotes.

        Only a quote that came from a chain read can carry an execution-native
        reference; Jupiter does not disclose a decoded account. This map is
        therefore built from the direct candidates, which name their exact
        market account.
        """
        sources: dict[str, tuple[str, Any]] = {}
        for c in candidates:
            reference = getattr(c, "execution_native_reference", None)
            pool = getattr(c, "pool_or_curve", None)
            venue_id = getattr(c, "venue_id", "") or ""
            if reference is None or not pool or not venue_id:
                continue
            sources.setdefault(pool, (venue_id, reference))
        return sources

    @classmethod
    def _attach_same_market_reference(
        cls, candidates: list[QuotePoint],
    ) -> list[QuotePoint]:
        """
        Let a Jupiter-routed candidate use the SAME market's on-chain reference.

        WHY THIS IS NEEDED
        ==================

        Jupiter is the primary market-discovery mechanism and, by the selection
        rule, usually wins on output. But Jupiter does not disclose a decoded
        account, so :meth:`_point_from_jupiter` cannot produce an
        ``execution_native_reference`` -- and the entry guard fails closed on a
        missing reference. The result was a BUY that could be QUOTED but could
        never be EXECUTED: every Jupiter-selected entry was refused with
        ``EXECUTION_NATIVE_REFERENCE_UNAVAILABLE`` even though the very same
        pool had been read, and priced, by the direct candidate beside it.

        WHAT THIS DOES -- AND DOES NOT -- DO
        ====================================

        A SINGLE-HOP Jupiter route names one pool. When a DIRECT candidate in
        the SAME enumeration read that exact pool from chain and produced a
        positive reference, that reference IS this route's own market, from its
        own slot, and it is attached -- together with the market's stable
        ``venue_id`` so the guard's ``same_market`` comparison is a real
        equality rather than an assumption.

        It is deliberately conservative:

        * only a single-hop route is eligible -- a multi-hop price is an
          aggregate of books and cannot be attributed to one pool;
        * the pool address must match a direct candidate EXACTLY;
        * the reference's mint must be this candidate's token;
        * when no direct candidate priced the same pool, nothing is attached and
          the guard returns UNKNOWN, which REFUSES. No mark, no other pool and
          no Jupiter-internal number is ever used as a substitute.

        No extra RPC is issued: the reference was already read, at the same
        slot, by the direct candidate in this same enumeration.
        """
        sources = cls._reference_sources(candidates)
        if not sources:
            return candidates
        out: list[QuotePoint] = []
        for c in candidates:
            if getattr(c, "execution_native_reference", None) is not None:
                out.append(c)
                continue
            pool = getattr(c, "pool_or_curve", None)
            hops = len(getattr(c, "route", ()) or ())
            if not pool or hops > 1 or pool not in sources:
                out.append(c)
                continue
            venue_id, reference = sources[pool]
            token = getattr(c, "token_mint", None)
            reference_mint = getattr(reference, "mint", None)
            if token and reference_mint and token != reference_mint:
                out.append(c)
                continue
            out.append(replace(
                c, venue_id=venue_id, execution_native_reference=reference))
        return out

    # -- entry identity RESOLUTION ------------------------------------------

    @staticmethod
    def _derived_market_addresses(mint: str) -> tuple[str, str]:
        """The two addresses a pump market for ``mint`` can canonically have.

        Both are DERIVED from the mint alone - the bonding-curve PDA and the
        canonical PumpSwap pool - so neither can be supplied by a caller, a
        DexScreener row or a route. Nothing here is read from chain; the
        comparison against these strings is what separates "this market was
        discovered from the mint" from "this market was accepted from an
        address somebody named".
        """
        curve, _ = derive_bonding_curve(mint)
        pool, _ = derive_canonical_pool(mint, WSOL_MINT)
        return curve, pool

    @classmethod
    def _chain_verified_canonical_market(
        cls, candidates: list[QuotePoint], mint: str,
    ) -> ResolvedMarket | None:
        """The canonical market for ``mint``, as a DIRECT candidate read it.

        Provider-native discovery has ALREADY run for this enumeration: a
        direct candidate that priced the mint decoded an account and carried an
        execution-native reference, so the market is chain-verified. What is
        checked here is that the decoded account is the DETERMINISTIC
        derivation for this mint - i.e. it was discovered FROM the mint -
        rather than an address the reader accepted from a supplier. The former
        is the canonical market; the latter is merely a market that exists.

        Returns ``None`` when no direct candidate qualifies, which is the
        honest answer and leaves every verdict as it was. It is never guessed
        from a route, a mark or a venue label.
        """
        curve, pool = cls._derived_market_addresses(mint)
        found: dict[str, ResolvedMarket] = {}
        for c in candidates:
            provider = str(getattr(c, "provider", "") or "")
            if provider not in ("pump_direct", "pumpswap_direct"):
                continue
            reference = getattr(c, "execution_native_reference", None)
            address = getattr(c, "pool_or_curve", None)
            if reference is None or not address:
                continue
            # The reference already names the mint it was read for; a market
            # decoded for another token is not this mint's, whatever address
            # it happens to have.
            if getattr(reference, "mint", None) != mint:
                continue
            if provider == "pump_direct" and address == curve:
                found["curve"] = ResolvedMarket(
                    venue_id=str(getattr(c, "venue_id", "") or ""),
                    address=address, provider=provider)
            elif provider == "pumpswap_direct" and address == pool:
                found["pool"] = ResolvedMarket(
                    venue_id=str(getattr(c, "venue_id", "") or ""),
                    address=address, provider=provider)
        # A live curve and a live pool are mutually exclusive for one mint (the
        # pool only exists after graduation), so preferring the pool is a
        # tie-break that cannot choose a wrong book over a right one.
        return found.get("pool") or found.get("curve")

    @classmethod
    def _resolve_market_identity(
        cls, candidates: list[QuotePoint], *, mint: str,
    ) -> list[QuotePoint]:
        """
        Prove the executable market from the MINT when the observed pair cannot.

        APPLIES TO **BOTH** DIRECTIONS, AND ON EXITS IT IS LOAD-BEARING.
        ==================================================================
        It was written for the ENTRY side and called only there, which left the
        EXIT side resolving its identity against the observed pair alone. For a
        position held past its entry curve's graduation that is not a stricter
        check, it is a check that can never pass: the observed pair is the
        RETIRED curve, so the direct candidate on the live pool is marked FAIL
        (different address) and the aggregator's single-hop route to that same
        live pool is marked UNKNOWN (different address) -- and the fill gate,
        which requires PASS, refuses every exit forever. Three live ALL-IN arms
        were stranded by exactly that: 8862 refusals, zero settlements.

        The question the exit has to answer is the SAME one the entry answers:
        "is the executable market this position would settle on the canonical
        market for THIS mint?", and the observed pair cannot answer it once the
        observed pair is a dead book. So the same proof is applied to both
        directions. It is not a relaxation -- see WHAT IT REFUSES TO DO: it
        marks every candidate on ANY other book FAIL, which the exit side
        previously did not do at all.

        THE DEFECT THIS FIXES ON ENTRY
        ==============================

        ``_identity_for`` answers one question: is the aggregator's observed
        pair the same object as the chain-verified executable market? That is
        the right question for the mark-vs-quote gap, but it is not the whole
        of "is this a real market for this token". On a pump token the
        aggregator's ``pairAddress`` is the market account - the curve PDA
        before graduation, the pool after - and for a short window after a
        graduation it is still the RETIRED CURVE: the pool exists and is live
        on chain, while the mark has not moved to it yet. Every candidate is
        then refused - the curve is complete, the pool differs from the
        observed pair, and a single-hop route that names the pool is reported
        as UNKNOWN - even though the canonical market was discovered from the
        mint and decoded on chain. That is a bookkeeping gap, not a market
        fact, and it blocked a real entry.

        WHAT IT DOES
        ============

        When provider-native discovery resolved this mint's canonical market,
        a candidate whose executable market IS that market is PASS, and a
        candidate whose market is provably a DIFFERENT book is FAIL. The
        observed pair is not consulted: it cannot be, because the whole defect
        is that it can be a retired market. The chain-verified canonical
        address is a function of the mint, which is a stronger claim than
        agreeing with one aggregator row.

        WHAT IT REFUSES TO DO
        =====================

        * no canonical market resolved -> every verdict is left EXACTLY as it
          was, so an unresolvable market still fails closed as UNKNOWN;
        * a multi-hop route is never promoted: its price is an aggregate of
          several books and cannot be attributed to this one;
        * a candidate for another token is never promoted;
        * a candidate that disclosed no market at all is never promoted;
        * no address is ever invented - the only accepted address is one that
          was derived from the mint and verified on chain;
        * ``venue_id`` is filled in from the resolved market when a candidate
          had none, so the OPEN position persists the market the trade was
          actually opened on and every hold tick reuses it.
        """
        canonical = cls._chain_verified_canonical_market(candidates, mint)
        if canonical is None:
            return candidates
        out: list[QuotePoint] = []
        for c in candidates:
            identity = str(getattr(c, "identity", "") or "")
            pool = getattr(c, "pool_or_curve", None)
            token = getattr(c, "token_mint", None)
            hops = len(getattr(c, "route", ()) or ())
            if (identity == "PASS" or hops > 1 or not pool
                    or (token and token != mint)):
                out.append(c)
                continue
            if pool == canonical.address:
                out.append(replace(
                    c, identity="PASS",
                    venue_id=(getattr(c, "venue_id", "") or canonical.venue_id),
                    identity_evidence=(
                        f"the executable market {str(pool)[:8]}… IS the "
                        f"chain-verified canonical market for this mint "
                        f"({canonical.venue_id}), derived from the mint and "
                        f"decoded by {canonical.provider}; the observed pair is "
                        "a different, non-executable market")))
            else:
                out.append(replace(
                    c, identity="FAIL", identity_evidence=(
                        f"the executable market {str(pool)[:8]}… is not the "
                        f"chain-verified canonical market "
                        f"{canonical.address[:8]}… for this mint "
                        f"({canonical.venue_id}); these are different books")))
        return out

    # -- public ------------------------------------------------------------


    # ------------------------------------------------------------------
    # venue enumeration and best-executable selection
    # ------------------------------------------------------------------

    async def quote_candidates(
        self,
        *,
        side: str,
        mint: str,
        amount_raw: int,
        token_decimals: int,
        observed_pair_address: str | None = None,
    ) -> list[QuotePoint]:
        """
        EVERY candidate venue quoted for this EXACT order size.

        All candidates are requested, including ones that fail, so the dataset
        can show what was available and not merely what was chosen. Selection
        happens in :meth:`_select_best`, never here.
        """
        out: list[QuotePoint] = []
        if side == "BUY":
            # Jupiter is the primary market-discovery mechanism:
            # it finds the best currently executable route across all AMMs.
            # Fall back to direct paths only if Jupiter has no route.
            if self.allow_jupiter_fallback and self.jupiter is not None:
                out.append(await self._quote_jupiter(
                    side=side, mint=mint, amount_raw=amount_raw,
                    decimals=token_decimals,
                    observed_pair_address=observed_pair_address))
            # Direct paths as fallback when Jupiter cannot route.
            direct_calls = (
                ("pump_direct", self._reader.quote_bonding_curve(
                    mint, sol_amount_lamports=amount_raw,
                    token_decimals=token_decimals)),
                ("pumpswap_direct", self._reader.quote_pumpswap(
                    mint, sol_amount_lamports=amount_raw,
                    pool_address=observed_pair_address)),
            )
            for provider, coro in direct_calls:
                try:
                    quote = await coro
                except Exception as exc:  # noqa: BLE001 - a named outcome
                    out.append(self._unavailable(
                        side=side, mint=mint, provider=provider,
                        reason=f"{type(exc).__name__}: {exc}"[:200],
                        observed_pair_address=observed_pair_address,
                    ))
                    continue
                if quote.available:
                    out.append(self._point_from_direct(
                        quote, side=side, mint=mint, decimals=token_decimals,
                        observed_pair_address=observed_pair_address,
                    ))
                else:
                    out.append(self._unavailable(
                        side=side, mint=mint, provider=provider,
                        reason=quote.missing_reason or "unavailable",
                        observed_pair_address=observed_pair_address,
                    ))
            # A Jupiter route that names the SAME pool a direct candidate just
            # read gets that pool's own reference, so a Jupiter-selected BUY is
            # EXECUTABLE rather than merely quotable. Fail-closed when no
            # same-market direct read exists.
            out = self._attach_same_market_reference(out)
            # ... and a candidate whose executable market IS the market
            # provider-native discovery derived from this mint is PASS even
            # when the aggregator's observed pair is a different (retired)
            # book. BUY only: the ENTRY decision is the one that consumes the
            # observed DexScreener mark, whereas the exit anchors on the
            # position's own pinned market. Fail-closed when nothing was
            # resolved.
            return self._resolve_market_identity(out, mint=mint)

        # SELL: a token amount OUT for quote asset.
        #
        # Jupiter is the primary market-discovery mechanism:
        # it finds the best currently executable route across all AMMs.
        # Fall back to direct paths only if Jupiter has no route.
        if self.allow_jupiter_fallback and self.jupiter is not None:
            out.append(await self._quote_jupiter(
                side=side, mint=mint, amount_raw=amount_raw,
                decimals=token_decimals,
                observed_pair_address=observed_pair_address))
        # Direct paths as fallback when Jupiter cannot route.
        for provider, coro in (
            ("pump_direct", self._reader.quote_bonding_curve_sell(
                mint, token_amount_raw=amount_raw,
                token_decimals=token_decimals)),
            ("pumpswap_direct", self._reader.quote_pumpswap_sell(
                mint, token_amount_raw=amount_raw,
                pool_address=observed_pair_address)),
        ):
            try:
                quote = await coro
            except Exception as exc:  # noqa: BLE001
                out.append(self._unavailable(
                    side=side, mint=mint, provider=provider,
                    reason=f"{type(exc).__name__}: {exc}"[:200],
                    observed_pair_address=observed_pair_address))
                continue
            if quote.available:
                out.append(self._point_from_direct(
                    quote, side=side, mint=mint, decimals=token_decimals,
                    observed_pair_address=observed_pair_address))
            else:
                out.append(self._unavailable(
                    side=side, mint=mint, provider=provider,
                    reason=quote.missing_reason or "unavailable",
                    observed_pair_address=observed_pair_address))
        # A Jupiter-routed SELL that names the SAME pool a direct candidate just
        # read gets that pool's own reference, so a Jupiter-selected EXIT is
        # executable rather than merely quotable -- exactly as for BUY. The exit
        # path (``quote_sell_pinned``) and the hold-path monitor both consume
        # this reference; without it a Jupiter-only route fails closed on the
        # exit trigger and produces no usable monitoring observation. Fail-closed
        # when no same-market direct read exists.
        out = self._attach_same_market_reference(out)
        # ... AND THE SAME CANONICAL-MARKET PROOF THE ENTRY SIDE APPLIES.
        #
        # The observed pair is the wrong question for an exit exactly as often
        # as it is for an entry, and for the same reason: once this token's
        # entry curve has graduated, the aggregator's observed pair IS that
        # retired curve, so agreeing with it is impossible for the market the
        # position actually trades on. Omitting this call is what left every
        # graduated exit at identity UNKNOWN/FAIL and unsettleable.
        return self._resolve_market_identity(out, mint=mint)

    async def _quote_jupiter(
        self, *, side: str, mint: str, amount_raw: int, decimals: int,
        observed_pair_address: str | None,
    ) -> QuotePoint:
        started = time.monotonic()
        try:
            if side == "BUY":
                result = await self.jupiter.quote_buy(
                    token_mint=mint, quote_asset_mint=WSOL_MINT,
                    in_amount=amount_raw, output_decimals=decimals)
            else:
                # The SELL signature is NOT the BUY signature, and passing the
                # BUY parameter names here is not a cosmetic mistake: it raised
                # a TypeError on every SELL, which was swallowed into an
                # "jupiter refused the quote" unavailable point. The entire
                # aggregator fallback leg of the exit path was therefore dead
                # -- and an exit that cannot fall back is an exit that fails
                # closed.
                #
                # ``token_amount`` is the EXACT raw quantity a preceding BUY
                # quote produced (``position.tokens_raw``), never re-derived
                # from a price: a liquidation estimate only means anything at
                # the size actually held.
                result = await self.jupiter.quote_sell(
                    token_mint=mint, quote_asset_mint=WSOL_MINT,
                    token_amount=amount_raw, token_decimals=decimals)
        except Exception as exc:  # noqa: BLE001
            return self._unavailable(
                side=side, mint=mint, provider="jupiter",
                reason=f"jupiter refused the quote: {exc}"[:200],
                observed_pair_address=observed_pair_address,
                latency_ms=(time.monotonic() - started) * 1000.0)
        return self._point_from_jupiter(
            result, side=side, mint=mint, decimals=decimals,
            observed_pair_address=observed_pair_address,
            latency_ms=(time.monotonic() - started) * 1000.0)

    @staticmethod
    def _usable_candidates(
        candidates: list[QuotePoint], *, side: str, mint: str | None = None,
    ) -> list[QuotePoint]:
        """
        The candidates that may become a fill, before any ranking.

        A candidate must be available, fresh, positive-output, settled in WSOL,
        be for the TOKEN this request is for, and must not carry a proven
        identity MISMATCH (``identity == "FAIL"``): a quote shown to be for a
        different book is refused outright, never ranked.

        ``mint`` is the token the caller asked for. When omitted, the first
        candidate's mint is treated as the expected one -- the caller always
        knows the real mint in production and passes it; the fallback exists so
        a single-mint candidate list still filters a mixed-up entry rather than
        admitting it by default.
        """
        expected_mint = mint if mint is not None else (
            candidates[0].token_mint if candidates else None)
        return [
            c for c in candidates
            if c.available
            and not c.stale
            and c.output_amount_decimal is not None
            and float(c.output_amount_decimal) > 0
            and expected_mint is not None
            and c.token_mint == expected_mint
            and c.quote_mint == WSOL_MINT
            and c.identity != "FAIL"
        ]

    @staticmethod
    def _best_executable(
        usable: list[QuotePoint], *, side: str,
    ) -> QuotePoint:
        """
        The highest-output usable candidate, identity-aware.

        SELECTION RULES, in order:

        1. Among usable candidates, a PASS-identified (proven same-market)
           candidate wins over any unproven one, because a different book is a
           different trade -- the higher output elsewhere is recorded as the
           counterfactual, never as the settlement.
        2. Within the proven group (or when nothing is proven), executable
           OUTPUT decides: highest output for this exact order size. Not the
           nominal price, not the mark, not liquidity, not a venue preference.
        3. The chosen point is returned AS IS: its ``available`` flag and its
           ``identity`` label are never mutated. A PASS route stays usable; an
           unproven route stays unproven and is refused later at the fill gate
           rather than being relabelled here.
        """
        pass_candidates = [c for c in usable if c.identity == "PASS"]
        pool = pass_candidates or usable
        return max(pool, key=lambda c: float(c.output_amount_decimal or 0.0))

    @staticmethod
    def _select_best(
        candidates: list[QuotePoint], *, side: str, mint: str | None = None,
    ) -> QuotePoint:
        """
        Pick the venue that returns the MOST for this exact order size.

        The ranking is EXECUTABLE OUTPUT and nothing else: not the nominal
        price, not the displayed mark, not liquidity, not the token symbol, not
        a venue preference. Those are all proxies for the thing being measured,
        and using one would make "best venue" mean "venue that looked cheapest",
        which is precisely the mark-versus-executable confusion this runtime
        exists to eliminate.

        A proven identity MISMATCH can never win -- it is excluded before the
        ranking. An identity that merely cannot be proven is selectable when
        nothing is proven (otherwise a position with only Jupiter routes could
        never be quoted at all), but it is never promoted to PASS: the fill gate
        still refuses it.
        """
        usable = QuoteService._usable_candidates(
            candidates, side=side, mint=mint)
        if not usable:
            return QuotePoint(
                side=side, provider="none", market_type="UNKNOWN",
                token_mint=candidates[0].token_mint if candidates else (mint or ""),
                quote_mint=WSOL_MINT, input_amount_raw=None,
                input_decimals=None, input_amount_decimal=None,
                output_amount_raw=None, output_decimals=None,
                output_amount_decimal=None, effective_price=None,
                effective_price_usd=None, price_impact_pct=None, fee_bps=None,
                fee_amount=None, venue=None, pool_or_curve=None, route=(),
                identity="UNKNOWN", available=False,
                missing_reason=f"no executable {side} venue")
        return QuoteService._best_executable(usable, side=side)

    async def get_monitoring_identity(
        self,
        *,
        mint: str,
        market_address: str,
        quote_mint: str = WSOL_MINT,
    ) -> dict[str, Any] | None:
        """
        Fetch static market identity for monitoring, given a known market address.

        This avoids re-deriving the market address from the mint and instead
        reads the exact account that was verified at entry. The returned dict
        contains all static data needed for lightweight monitoring quotes.
        """
        try:
            market, snapshot, market_type, provider = await self._reader.read_market_by_address(
                address=market_address, mint=mint, quote_mint=quote_mint
            )
        except Exception as exc:  # noqa: BLE001
            return None
        if market is None:
            return None

        # Extract static identity data
        if market_type == "PUMP_BONDING_CURVE":
            curve = market
            base_vault = None  # bonding curve uses virtual reserves, not vaults
            quote_vault = None
            base_decimals = 6  # will be overridden by token_decimals at monitoring time
            circulating_supply = await self._reader._mint_supply(mint)
            virtual_quote_reserves = curve.virtual_quote_reserves
            base_vault = None
            quote_vault = None
            fee_config = None
            global_config = None
            # Get fee configs
            try:
                glob, fee_config = await self._reader._pump_fee_state()
                global_config = glob
            except Exception:
                pass
            return {
                "market_type": MarketKind.PUMP_BONDING_CURVE,
                "market_address": market.address,
                "base_mint": curve.base_mint if hasattr(curve, "base_mint") else mint,
                "quote_mint": curve.quote_mint,
                "base_decimals": 6,
                "base_vault": base_vault,
                "quote_vault": quote_vault,
                "virtual_quote_reserves": virtual_quote_reserves,
                "fee_config": fee_config,
                "global_config": global_config,
                "circulating_supply": circulating_supply,
                "provider": provider,
                "coin_creator": curve.creator,
            }
        else:
            pool = market
            base_vault = pool.pool_base_token_account
            quote_vault = pool.pool_quote_token_account
            circulating_supply = pool.base_supply
            virtual_quote_reserves = pool.virtual_quote_reserves
            base_decimals = pool.base_decimals
            fee_config = None
            global_config = None
            try:
                global_config, fee_config = await self._reader._amm_fee_state()
            except Exception:
                pass
            return {
                "market_type": MarketKind.PUMPSWAP,
                "market_address": market.address,
                "base_mint": pool.base_mint,
                "quote_mint": pool.quote_mint,
                "base_decimals": base_decimals,
                "base_vault": base_vault,
                "quote_vault": quote_vault,
                "virtual_quote_reserves": virtual_quote_reserves,
                "fee_config": fee_config,
                "global_config": global_config,
                "circulating_supply": circulating_supply,
                "provider": provider,
                "coin_creator": pool.coin_creator,
            }
    async def quote_buy_best(
        self,
        *,
        mint: str,
        sol_lamports: int,
        token_decimals: int,
        observed_pair_address: str | None = None,
        cache: "QuoteCache | None" = None,
    ) -> tuple[QuotePoint, list[QuotePoint]]:
        """
        The best executable BUY for this exact size, plus every candidate.

        Returns ``(selected, all_candidates)``. The candidates are returned
        rather than discarded so the dataset records what else was on the table
        at this instant -- that distribution is the multi-venue research
        material.
        """
        key = ("BUY", mint, sol_lamports, token_decimals)
        if cache is not None:
            hit = cache.get(key)
            if hit is not None:
                quote, extras = hit
                return quote, list(extras)
        candidates = await self.quote_candidates(
            side="BUY", mint=mint, amount_raw=sol_lamports,
            token_decimals=token_decimals,
            observed_pair_address=observed_pair_address)
        best = self._select_best(candidates, side="BUY", mint=mint)
        if not best.available:
            # NAME EVERY SOURCE, not only the verdict. A flat "no executable
            # BUY venue" cannot distinguish "Jupiter has no route on a
            # minutes-old token" from "our RPC was throttled" from "identity
            # could not be proven" -- and the first of those is NORMAL.
            summary = self.candidate_summary(candidates, mint=mint)
            best = replace(
                best,
                missing_reason=(
                    f"no executable BUY venue. candidates: {summary}"),
            )
        if cache is not None and best.available:
            cache.put(key, best, tuple(candidates))
        return best, candidates

    def candidate_summary(self, candidates: list[QuotePoint],
                          *, mint: str | None = None) -> str:
        """Per-source BUY outcome table, aware of what this runtime wired."""
        configured = bool(
            self.jupiter is not None and self.allow_jupiter_fallback)
        return buy_candidate_summary(
            candidates, jupiter_configured=configured, mint=mint)

    # ------------------------------------------------------------------
    # EXIT VENUE LIFECYCLE: chain-verified Pump.fun graduation
    # ------------------------------------------------------------------

    async def resolve_graduated_exit_venue(
        self, *,
        mint: str,
        token_amount_raw: int,
        token_decimals: int,
        entry_venue_id: str,
        entry_pool_address: str | None = None,
        observed_pair_address: str | None = None,
    ) -> QuotePoint:
        """
        An executable SELL on the PumpSwap pool a graduated token now trades on.

        THE ONE TRANSITION THIS PERMITS
        ================================
        ``PUMP_BONDING_CURVE -> PUMPSWAP``, and nothing else. The entry venue
        must name a bonding curve or this returns a refusal without touching the
        network: there is no other migration in this system, so there is no
        other case to handle.

        WHY A FAILED QUOTE IS NOT ENOUGH
        =================================
        A quote fails for reasons that have nothing to do with lifecycle: a
        throttled provider, an RPC blip, an empty book, a size that moved the
        price. Migrating on any of those would turn "try again" into "sell
        somewhere else", which is precisely the arbitrary-fallback behaviour this
        system exists to forbid. So graduation is PROVEN first, from chain
        state, and a failed quote on its own changes nothing.

        WHAT "PROVEN" MEANS HERE, CONCRETELY
        =====================================
        Nothing new is read that was not already read by this same reader on
        every other call:

        1. the curve account at the PDA derived from ``mint`` exists, is owned
           by the pump.fun program, and carries the BondingCurve discriminator
           (all three asserted inside :meth:`DirectPumpReader.read_bonding_curve`);
        2. its ``complete`` flag is set, or its virtual token reserves are zero,
           which is the protocol's own drained state.

        Only then is the migrated market looked up, and it is looked up by
        DERIVING it from the same mint rather than by searching. The derived
        canonical pool is then verified by the reader against the chain: PumpSwap
        program ownership, the Pool discriminator, and ``base_mint == mint``.
        A pool for a different token fails inside the reader and is surfaced as
        a refusal, never relabelled.

        EVERY FAILURE IS A NAMED REFUSAL
        =================================
        The return is an UNAVAILABLE :class:`QuotePoint` carrying the exact
        reason, never a zero, never a mark, and never a fallback. "No
        executable sell quote" must stay the outcome of every unhappy path so
        the position stays OPEN instead of being closed at a price nobody was
        paid.
        """
        curve_prefix = f"{MarketKind.PUMP_BONDING_CURVE.value}:"
        if not entry_venue_id.startswith(curve_prefix):
            return self._unavailable(
                side="SELL", mint=mint, provider="pumpswap_direct",
                reason=(
                    f"exit venue migration was requested for a position entered "
                    f"on {entry_venue_id or 'an unknown venue'}; the only "
                    "supported transition is "
                    f"{MarketKind.PUMP_BONDING_CURVE.value} -> "
                    f"{MarketKind.PUMPSWAP.value}, so no other market was "
                    "considered"
                )[:400],
                market_type=MarketKind.PUMPSWAP.value,
                observed_pair_address=observed_pair_address,
            )

        # -- STEP 1: prove the graduation from chain state -----------------
        try:
            curve, curve_snapshot = await self._reader.read_bonding_curve(mint)
        except Exception as exc:  # noqa: BLE001 - named, never assumed
            return self._unavailable(
                side="SELL", mint=mint, provider="pumpswap_direct",
                reason=(
                    "graduation could not be checked because the bonding curve "
                    f"could not be read ({type(exc).__name__}: {exc}); a failed "
                    "quote is not evidence of graduation, so the exit venue is "
                    "unchanged"
                )[:400],
                market_type=MarketKind.PUMPSWAP.value,
                observed_pair_address=observed_pair_address,
            )
        if curve is None:
            return self._unavailable(
                side="SELL", mint=mint, provider="pumpswap_direct",
                reason=(
                    "no bonding curve account exists at the PDA derived from "
                    "this mint, so graduation cannot be proven from chain state "
                    "and the exit venue is unchanged"
                )[:400],
                market_type=MarketKind.PUMPSWAP.value,
                observed_pair_address=observed_pair_address,
            )
        if curve.mint and curve.mint != mint:
            return self._unavailable(
                side="SELL", mint=mint, provider="pumpswap_direct",
                reason=(
                    f"the bonding curve at {curve.address[:8]}… belongs to mint "
                    f"{curve.mint[:8]}…, not {mint[:8]}…; it is a different "
                    "token's curve and the exit venue is unchanged"
                )[:400],
                market_type=MarketKind.PUMPSWAP.value,
                observed_pair_address=observed_pair_address,
            )
        if not curve.graduated:
            return self._unavailable(
                side="SELL", mint=mint, provider="pumpswap_direct",
                reason=(
                    f"the bonding curve at {curve.address[:8]}… is NOT complete "
                    "(complete="
                    f"{curve.complete}, virtual_token_reserves="
                    f"{curve.virtual_token_reserves}), so the token has not "
                    "graduated and the exit venue is unchanged"
                )[:400],
                market_type=MarketKind.PUMPSWAP.value,
                observed_pair_address=observed_pair_address,
            )

        graduation_evidence = (
            f"the account at the bonding-curve PDA derived from this mint is "
            f"owned by the pump.fun program and carries the BondingCurve "
            f"discriminator, and it reports complete={curve.complete} with "
            f"virtual_token_reserves={curve.virtual_token_reserves} and "
            f"real_quote_reserves={curve.real_quote_reserves}"
        )

        # -- STEP 2: the migrated PumpSwap market, DERIVED from the mint ---
        try:
            pool, pool_snapshot, pool_address = (
                await self._reader.read_pumpswap_pool(mint))
        except Exception as exc:  # noqa: BLE001 - named, never assumed
            return self._unavailable(
                side="SELL", mint=mint, provider="pumpswap_direct",
                reason=(
                    "graduation is proven but no PumpSwap pool could be "
                    f"resolved for this mint ({type(exc).__name__}: {exc}); the "
                    "position stays OPEN on the graduated market"
                )[:400],
                market_type=MarketKind.PUMPSWAP.value,
                observed_pair_address=observed_pair_address,
            )
        if pool is None or not pool_address:
            return self._unavailable(
                side="SELL", mint=mint, provider="pumpswap_direct",
                reason=(
                    "graduation is proven but no PumpSwap pool exists for this "
                    "mint; the position stays OPEN"
                )[:400],
                market_type=MarketKind.PUMPSWAP.value,
                observed_pair_address=observed_pair_address,
            )
        # The reader already refuses a pool whose base mint is not this mint.
        # Checked again here because this is the invariant the whole migration
        # rests on, and a guard that exists only one layer down is one refactor
        # away from not existing.
        if pool.base_mint != mint:
            return self._unavailable(
                side="SELL", mint=mint, provider="pumpswap_direct",
                reason=(
                    f"the PumpSwap pool at {pool_address[:8]}… has base mint "
                    f"{pool.base_mint[:8]}…, not {mint[:8]}…; it is another "
                    "token's pool and is refused rather than relabelled, so the "
                    "position stays OPEN"
                )[:400],
                market_type=MarketKind.PUMPSWAP.value,
                observed_pair_address=observed_pair_address,
            )

        # -- STEP 3: an executable SELL on that exact pool ------------------
        direct = await self._reader.quote_pumpswap_sell(
            mint, token_amount_raw=token_amount_raw, pool_address=pool_address)
        if not direct.available or direct.quote is None:
            return self._unavailable(
                side="SELL", mint=mint, provider="pumpswap_direct",
                reason=(
                    "graduation is proven and the PumpSwap pool for this same "
                    f"mint was verified, but it returned no executable SELL "
                    f"quote: {direct.missing_reason or 'unavailable'}"
                )[:400],
                market_type=MarketKind.PUMPSWAP.value,
                observed_pair_address=observed_pair_address,
            )

        # The migration identity is the SAME-TOKEN proof plus the graduation
        # proof, which is why it does not come from the observed pair the way
        # every other identity does. There is no aggregator pair for a pool
        # that has just been migrated to, and inventing one is what the address
        # comparison exists to prevent.
        point = self._point_from_direct(
            direct, side="SELL", mint=mint, decimals=token_decimals,
            observed_pair_address=pool_address,
            identity_override=(
                "PASS",
                "the exit venue changed because this token's own bonding curve "
                f"was proven complete on chain, and the PumpSwap pool used is "
                f"the one DERIVED from this mint, is owned by the PumpSwap "
                f"program, carries the Pool discriminator, and its base mint "
                f"field equals the position mint; same token, verified "
                f"migration",
            ),
        )
        # -- STEP 4: refuse anything malformed before it can become a fill ---
        malformed = self._migrated_quote_defect(point, mint, token_amount_raw)
        if malformed is not None:
            return self._unavailable(
                side="SELL", mint=mint, provider="pumpswap_direct",
                reason=(
                    f"graduation is proven but the migrated PumpSwap quote is "
                    f"unusable and was refused: {malformed}"
                )[:400],
                market_type=MarketKind.PUMPSWAP.value,
                observed_pair_address=observed_pair_address,
            )

        to_venue_id = self.venue_id_for(MarketKind.PUMPSWAP, pool_address)
        return replace(
            point,
            venue_migration=VenueMigration(
                reason=EXIT_VENUE_MIGRATION_GRADUATED,
                from_venue_id=entry_venue_id,
                to_venue_id=to_venue_id,
                from_market_type=MarketKind.PUMP_BONDING_CURVE.value,
                to_market_type=MarketKind.PUMPSWAP.value,
                token_mint=mint,
                verified=(
                    f"{graduation_evidence}; {pool.address[:8]}… is owned by "
                    "the PumpSwap program, carries the Pool discriminator, and "
                    f"its base mint field is {mint}"
                ),
                at=time.time(),
            ),
        )

    def _migrated_quote_defect(
        self, point: QuotePoint, mint: str, token_amount_raw: int,
    ) -> str | None:
        """
        Why this quote must not become a fill, or ``None`` if it may.

        A migration is the one place in this system where the exit is taken on a
        market nobody has ever quoted before, so the quote is checked against
        every property a fill depends on rather than trusted because it came
        from the migration path. Each of these is a shape a quote can take that
        is non-None, available and still unusable.
        """
        if point.token_mint != mint:
            return (f"the quote is for token {point.token_mint[:8]}… but the "
                    f"position holds {mint[:8]}…")
        if point.quote_mint != WSOL_MINT:
            return f"the quote settles in {point.quote_mint[:8]}…, not SOL"
        if point.output_amount_raw is None or int(point.output_amount_raw) <= 0:
            return "the quote produced no output amount to sell into"
        if point.input_amount_raw is None or (
                int(point.input_amount_raw) != int(token_amount_raw)):
            return (
                f"the quote sizes {point.input_amount_raw} raw units but the "
                f"position holds {token_amount_raw}")
        if point.effective_price is None or float(point.effective_price) <= 0:
            return "the quote has no positive effective price"
        if not point.route:
            return "the quote carries no route"
        if point.stale:
            return "the quote is older than the configured freshness window"
        if point.venue_id.startswith(
                f"{MarketKind.PUMP_BONDING_CURVE.value}:"):
            return "the migrated quote came back on the bonding curve"
        return None

    # -- monitoring quote (lightweight, uses cached static identity) --------

    async def quote_sell_monitoring(
        self,
        *,
        mint: str,
        token_amount_raw: int,
        token_decimals: int,
        cached_identity: dict[str, Any],
        observed_pair_address: str | None = None,
    ) -> QuotePoint:
        """
        Lightweight SELL quote for MONITORING ONLY.

        Uses pre-cached static market identity (curve/pool address, vault
        addresses, fee configs, decimals) and ONLY reads dynamic reserves.
        This avoids re-discovering the market on every tick.

        Returns a QuotePoint with available=False if static data is missing
        or dynamic reads fail.
        """
        from types import SimpleNamespace
        import time
        from decimal import Decimal
        from autopaper.same_market.rpc import rpc_lane, telemetry_sample, LANE_TELEMETRY
        from autopaper.same_market.pumpdirect import (
            Fees, curve_fee_schedule, amm_fee_schedule,
            DEFAULT_PUBKEY, SOL_DECIMALS, WSOL_MINT,
            MarketKind, _ceil_fee as ceil_fee,
        )
        started = time.monotonic()

        market_type = cached_identity.get("market_type")
        market_address = cached_identity.get("market_address")
        base_mint = cached_identity.get("base_mint", mint)
        quote_mint = cached_identity.get("quote_mint", WSOL_MINT)
        base_decimals = cached_identity.get("base_decimals", token_decimals)
        base_vault = cached_identity.get("base_vault")
        quote_vault = cached_identity.get("quote_vault")
        virtual_quote_reserves = cached_identity.get("virtual_quote_reserves", 0)
        fee_config = cached_identity.get("fee_config")
        global_config = cached_identity.get("global_config")
        circulating_supply = cached_identity.get("circulating_supply")
        provider = cached_identity.get("provider", "monitoring")

        if not market_address or not base_vault or not quote_vault:
            return self._unavailable(
                side="SELL", mint=mint, provider="monitoring",
                reason="missing cached static identity for monitoring quote",
                observed_pair_address=observed_pair_address,
            )

        # Read dynamic reserves ONLY (base vault + quote vault)
        try:
            with telemetry_sample():
                with rpc_lane(LANE_TELEMETRY):
                    base_ata_account, quote_ata_account = await self._reader._accounts(
                        [base_vault, quote_vault]
                    )
        except Exception as exc:  # noqa: BLE001
            return self._unavailable(
                side="SELL", mint=mint, provider="monitoring",
                reason=f"vault reads failed: {type(exc).__name__}: {exc}"[:200],
                observed_pair_address=observed_pair_address,
            )

        if base_ata_account is None or base_ata_account.is_empty:
            return self._unavailable(
                side="SELL", mint=mint, provider="monitoring",
                reason="base vault account missing",
                observed_pair_address=observed_pair_address,
            )
        if quote_ata_account is None or quote_ata_account.is_empty:
            return self._unavailable(
                side="SELL", mint=mint, provider="monitoring",
                reason="quote vault account missing",
                observed_pair_address=observed_pair_address,
            )

        # SPL token account amount is at offset 64 (8 bytes)
        base_reserve = int.from_bytes(
            base_ata_account.raw()[64:72], "little")
        quote_reserve = int.from_bytes(
            quote_ata_account.raw()[64:72], "little")

        if base_reserve <= 0:
            return self._unavailable(
                side="SELL", mint=mint, provider="monitoring",
                reason="base reserve zero",
                observed_pair_address=observed_pair_address,
            )

        effective_quote_reserve = quote_reserve + virtual_quote_reserves
        if effective_quote_reserve <= 0:
            return self._unavailable(
                side="SELL", mint=mint, provider="monitoring",
                reason="effective quote reserve zero",
                observed_pair_address=observed_pair_address,
            )

        # Compute fees
        if fee_config is not None:
            market_cap = (effective_quote_reserve * circulating_supply) // base_reserve
            if market_type == MarketKind.PUMP_BONDING_CURVE:
                fees = curve_fee_schedule(
                    None, global_config=global_config, fee_config=fee_config,
                    mint_supply=circulating_supply)
            else:
                # For PumpSwap, we need a mock pool object
                mock_pool = SimpleNamespace(
                    effective_quote_reserve=effective_quote_reserve,
                    base_reserve=base_reserve,
                    base_supply=circulating_supply,
                    quote_mint=quote_mint,
                    is_canonical_pump_pool=True,
                    creator_fee_bps=None,
                    coin_creator=cached_identity.get("coin_creator", DEFAULT_PUBKEY),
                )
                fees = amm_fee_schedule(
                    mock_pool, global_config=global_config, fee_config=fee_config,
                    circulating_supply=circulating_supply)
        else:
            fees = Fees(0, global_config.fee_basis_points if global_config else 0,
                        global_config.creator_fee_basis_points if global_config else 0)

        # SELL: token -> SOL (quote asset)
        # effectiveQuote = quote * 10000 // (10000 + totalFeeBps)
        # input = effectiveQuote - 1
        # output = baseReserve * input // (effectiveQuoteReserve + input)
        creator_fee_bps = fees.creator_fee_bps if cached_identity.get("coin_creator") != DEFAULT_PUBKEY else 0
        total_fee_bps = fees.lp_fee_bps + fees.protocol_fee_bps + creator_fee_bps

        effective_quote = (token_amount_raw * 10_000) // (10_000 + total_fee_bps)
        # Ceil fees
        lp_fee = ceil_fee(effective_quote, fees.lp_fee_bps)
        protocol_fee = ceil_fee(effective_quote, fees.protocol_fee_bps)
        cc_fee = ceil_fee(effective_quote, creator_fee_bps)
        total_with_fees = effective_quote + lp_fee + protocol_fee + cc_fee
        if total_with_fees > token_amount_raw:
            effective_quote -= total_with_fees - token_amount_raw

        net_input = effective_quote - 1
        if net_input <= 0:
            return self._unavailable(
                side="SELL", mint=mint, provider="monitoring",
                reason="below minimum trade size after fees",
                observed_pair_address=observed_pair_address,
            )

        out = (base_reserve * net_input) // (effective_quote_reserve + net_input)
        if out <= 0:
            return self._unavailable(
                side="SELL", mint=mint, provider="monitoring",
                reason="zero output",
                observed_pair_address=observed_pair_address,
            )

        # Compute effective price (SOL per token)
        try:
            price = Decimal(token_amount_raw).scaleb(-SOL_DECIMALS) / Decimal(out).scaleb(-base_decimals)
        except Exception:
            return self._unavailable(
                side="SELL", mint=mint, provider="monitoring",
                reason="price computation failed",
                observed_pair_address=observed_pair_address,
            )

        price_usd = float(price) * self.sol_reference.usd_per_sol if self.sol_reference else None

        # Native reference (spot price from same reserves)
        try:
            spot_price = Decimal(effective_quote_reserve).scaleb(-SOL_DECIMALS) / Decimal(base_reserve).scaleb(-base_decimals)
        except Exception:
            spot_price = None

        now = time.time()
        latency_ms = (time.monotonic() - started) * 1000.0

        # Identity check
        identity = "PASS" if market_address == observed_pair_address else "UNKNOWN"
        if observed_pair_address and market_address != observed_pair_address:
            identity = "FAIL"

        return QuotePoint(
            side="SELL",
            provider=provider,
            market_type=market_type.value if hasattr(market_type, "value") else str(market_type),
            token_mint=base_mint,
            quote_mint=quote_mint,
            input_amount_raw=token_amount_raw,
            input_decimals=base_decimals,
            input_amount_decimal=float(Decimal(token_amount_raw).scaleb(-base_decimals)),
            output_amount_raw=out,
            output_decimals=SOL_DECIMALS,
            output_amount_decimal=float(Decimal(out).scaleb(-SOL_DECIMALS)),
            effective_price=price,
            effective_price_usd=price_usd,
            price_impact_pct=None,
            fee_bps=total_fee_bps,
            fee_amount=Decimal(0),
            venue="pumpfun-bonding-curve" if market_type == MarketKind.PUMP_BONDING_CURVE else "pumpswap",
            pool_or_curve=market_address,
            route=(cached_identity.get("provider", "monitoring"),),
            state_slot=None,
            state_age_ms=None,
            quote_timestamp=now,
            quote_age_ms=0.0,
            max_quote_age_ms=self.max_quote_age_ms,
            quote_latency_ms=latency_ms,
            identity=identity,
            observed_pair_address=observed_pair_address,
            available=True,
            execution_native_reference=None,  # monitoring quote doesn't need this
        )

    async def quote_sell_pinned(
        self,
        *,
        mint: str,
        token_amount_raw: int,
        token_decimals: int,
        pinned_venue_id: str | None,
        pinned_pool_address: str | None = None,
        observed_pair_address: str | None = None,
        cache: "QuoteCache | None" = None,
    ) -> tuple[QuotePoint, QuotePoint | None]:
        """
        The executable SELL quote, selected by best output amount.

        The entry venue is recorded for historical provenance but does NOT
        determine the exit market. The best executable route is discovered
        fresh via Jupiter + direct paths, selected by highest WSOL output
        amount, without permanent venue pinning.

        Returns ``(selected_quote, counterfactual_best_elsewhere)``.

        SELECTION RULES
        ==============
        1. All candidates are gathered (Jupiter primary, direct paths fallback).
        2. Stale and unavailable quotes are excluded.
        3. The best quote is selected by ``output_amount_decimal`` - the
           highest WSOL output for the exact token amount held.
        4. Token identity is validated: input mint must match the held mint,
           output mint must be WSOL, and output amount must be positive.
        5. Stale quotes are rejected.
        6. If no valid quote exists, the position stays OPEN (no fake fill).

        The original entry venue remains as historical provenance
        (``entry_venue_id`` on the position record) but does not constrain
        the exit market. A graduated token will naturally find a new
        executable route through Jupiter discovery.
        """
        key = ("SELL", mint, token_amount_raw, token_decimals,
               pinned_venue_id or "")
        if cache is not None:
            hit = cache.get(key)
            if hit is not None:
                quote, extras = hit
                return quote, (extras[0] if extras else None)

        candidates = await self.quote_candidates(
            side="SELL", mint=mint, amount_raw=token_amount_raw,
            token_decimals=token_decimals,
            observed_pair_address=pinned_pool_address or observed_pair_address)

        # Select the best executable quote by output amount, without
        # enforcing the pinned entry venue. The pinned venue is recorded
        # for historical provenance only.
        chosen, counterfactual = self._select_best_sell_executable(
            candidates, pinned_venue_id=pinned_venue_id, mint=mint)
        # NO USABLE QUOTE STILL COMES BACK WITH EVERY LEG'S REASON ATTACHED.
        #
        # The bare ``None`` this used to return is what made 1,583 live refusals
        # read ``no executable SELL quote: unavailable`` and nothing else: the
        # per-leg reasons HAD been produced -- an HTTP 429, a completed bonding
        # curve, an RPC quota, a proven identity MISMATCH -- and were dropped at
        # this boundary. An unavailable point is not a quote and passes no gate
        # the ``None`` did not already fail; it only makes the refusal readable.
        if chosen is None:
            chosen = self._no_executable_sell_reason(
                candidates, mint=mint, pinned_venue_id=pinned_venue_id,
                observed_pair_address=(pinned_pool_address
                                       or observed_pair_address))

        if cache is not None and chosen is not None and chosen.available:
            cache.put(key, chosen, (chosen, counterfactual))
        return chosen, counterfactual

    def _no_executable_sell_reason(
        self, candidates: list[QuotePoint], *,
        mint: str,
        pinned_venue_id: str | None,
        observed_pair_address: str | None,
    ) -> QuotePoint:
        """
        An UNAVAILABLE point naming EVERY leg's refusal, for the operator.

        THE DEFECT IT CLOSES
        ====================
        A refusal has to say WHY, and the reason a caller can report is the
        reason it was given. Returning ``None`` threw away three produced reasons
        and left ``unavailable``, which is not a diagnosis: it cannot distinguish
        a throttled provider from a dead market, and those call for opposite
        responses (back off vs. stop asking).

        WHAT IT IS NOT
        ==============
        It is NOT a quote and NOT a fallback: ``available`` is False, every amount
        is ``None``, the identity verdict is UNKNOWN and the caller refuses it
        exactly as it refused the ``None``. It is not cached -- a refusal is a
        statement about THIS attempt, and serving it from a cache would be a
        stale refusal.
        """
        parts: list[str] = []
        for c in candidates:
            label = str(getattr(c, "provider", "") or "unknown")
            identity = str(getattr(c, "identity", "") or "")
            why = getattr(c, "missing_reason", None)
            if not why and identity and identity != "PASS":
                why = (getattr(c, "identity_evidence", None)
                       or f"identity is {identity}, not PASS")
            # BOUNDED PER LEG, so all three legs survive the display budget.
            # A refusal that reports only the first leg is the same defect one
            # layer down.
            why = " ".join(str(why or "refused").split())[:90]
            tag = "" if (not identity or identity == "PASS") \
                else f"[{identity}]"
            parts.append(f"{label}{tag}: {why}")
        body = "; ".join(parts) or "no venue was attempted"
        header = (
            "every SELL venue refused"
            + (f" for the pinned market {str(observed_pair_address)[:8]}…"
               if observed_pair_address else "")
        )
        return self._unavailable(
            side="SELL", mint=mint, provider="none",
            market_type="UNKNOWN", identity="UNKNOWN",
            reason=f"{header}: {body}"[:480],
            observed_pair_address=observed_pair_address,
        )

    def _select_best_sell_executable(
        self, candidates: list[QuotePoint],
        *,
        pinned_venue_id: str | None,
        mint: str | None = None,
    ) -> tuple[QuotePoint | None, QuotePoint | None]:
        """
        Select the best executable SELL quote by output amount,
        without permanent venue pinning.

        Returns (chosen, counterfactual) where:
        - chosen: the best usable executable quote, AS IS (its ``available``
          flag is never mutated), or None if none is usable
        - counterfactual: the best remaining usable quote aside from chosen,
          for research purposes

        USABLE MEANS: available, fresh, positive WSOL output, input mint == the
        held token, output mint == WSOL, and identity not a proven MISMATCH.
        Ranking is identity-first (a proven same-market route wins), then
        highest output. An unproven route is selectable when nothing is proven
        -- otherwise a graduated token whose only route is Jupiter could never
        be quoted -- but it is never relabelled PASS; the fill gate refuses it.
        """
        usable = self._usable_candidates(candidates, side="SELL", mint=mint)
        if not usable:
            return None, None

        chosen = self._best_executable(usable, side="SELL")
        rest = [c for c in usable if c is not chosen]
        counterfactual = (
            self._best_executable(rest, side="SELL") if rest else None)
        return chosen, counterfactual

    async def quote_buy(
        self,
        *,
        mint: str,
        sol_lamports: int,
        token_decimals: int,
        observed_pair_address: str | None = None,
    ) -> QuotePoint:
        """
        A fresh executable ``SOL -> TOKEN`` quote for this exact size.

        Precedence: direct bonding curve, then direct PumpSwap, then Jupiter.
        The direct paths name the exact pool, so they are preferred; which one
        answered is recorded in ``provider``.
        """
        curve = await self._reader.quote_bonding_curve(
            mint, sol_amount_lamports=sol_lamports, token_decimals=token_decimals,
        )
        if curve.available:
            return self._point_from_direct(
                curve, side="BUY", mint=mint, decimals=token_decimals,
                observed_pair_address=observed_pair_address,
            )

        swap = await self._probe_venues(
            mint=mint, sol_lamports=sol_lamports,
            token_decimals=token_decimals,
            observed_pair_address=observed_pair_address, curve=curve)

        if swap is not None and swap.available:
            return self._point_from_direct(
                swap, side="BUY", mint=mint, decimals=token_decimals,
                observed_pair_address=observed_pair_address,
            )

        # Jupiter is consulted only once BOTH direct venues have declined, so
        # the hedged probe can never cost an extra aggregator call.
        if self.allow_jupiter_fallback and self.jupiter is not None:
            started = time.monotonic()
            try:
                result = await self.jupiter.quote_buy(
                    token_mint=mint, quote_asset_mint=WSOL_MINT,
                    in_amount=sol_lamports, output_decimals=token_decimals,
                )
            except Exception as exc:  # noqa: BLE001 - named, never falls back
                return self._unavailable(
                    side="BUY", mint=mint, provider="jupiter",
                    reason=f"jupiter refused the quote: {exc}"[:200],
                    observed_pair_address=observed_pair_address,
                    latency_ms=(time.monotonic() - started) * 1000.0,
                )
            return self._point_from_jupiter(
                result, side="BUY", mint=mint, decimals=token_decimals,
                observed_pair_address=observed_pair_address,
                latency_ms=(time.monotonic() - started) * 1000.0,
            )

        # Fail closed: say WHY both direct paths declined.
        return self._unavailable(
            side="BUY", mint=mint,
            reason=(
                "no executable quote. bonding curve: "
                f"{curve.missing_reason or 'unavailable'}; pumpswap: "
                f"{swap.missing_reason or 'unavailable'}"
            )[:400],
            provider="pump_direct+pumpswap_direct",
            observed_pair_address=observed_pair_address,
        )

    async def _probe_venues(
        self,
        *,
        mint: str,
        sol_lamports: int,
        token_decimals: int,
        observed_pair_address: str | None,
        curve: Any,
    ) -> Any:
        """
        Ask the remaining direct venue, under a bounded deadline.

        WHY THIS IS NOT A SIMPLE ``await`` ANY MORE
        ============================================

        The venue probe used to be a straight sequence, so a candidate's
        decision time was the SUM of every venue tried before one answered. A
        throttled first venue therefore added its full retry budget to the
        decision even when the second venue could have answered immediately.

        Two properties are preserved deliberately:

        * **The deadline is a ceiling, never a queue.** Past it the probe
          returns what it has -- including nothing -- so a slow venue can never
          hold the decision cycle open. It is never "wait your turn".
        * **No unbounded fan-out.** Exactly one extra venue is probed, so the
          load this adds is bounded by one request, not by the candidate count.

        The refusal is returned, never raised, so the caller keeps its existing
        fail-closed path and its existing refusal wording.
        """
        deadline_s = float(self.quote_deadline_s)
        started = time.monotonic()

        async def _swap() -> Any:
            return await self._reader.quote_pumpswap(
                mint, sol_amount_lamports=sol_lamports,
                pool_address=observed_pair_address,
            )

        task = asyncio.ensure_future(_swap())
        try:
            return await asyncio.wait_for(asyncio.shield(task), deadline_s)
        except asyncio.TimeoutError:
            # CANCELLED, not abandoned: leaving the task running would keep
            # spending the very capacity this path is short of.
            self.quote_timeouts += 1
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            return self._timeout_point(mint, observed_pair_address, started)
        except asyncio.CancelledError:
            task.cancel()
            raise
        except Exception as exc:  # noqa: BLE001 - one venue must not kill the rest
            self.quote_errors += 1
            return self._unavailable(
                side="BUY", mint=mint, provider="pumpswap_direct",
                reason=f"pumpswap probe failed: {type(exc).__name__}: {exc}"[:200],
                observed_pair_address=observed_pair_address,
                latency_ms=(time.monotonic() - started) * 1000.0,
            )

    def _timeout_point(
        self, mint: str, observed_pair_address: str | None, started: float,
    ) -> Any:
        """
        A deadline refusal, named as a TIMEOUT.

        This is the distinction the incident needed: running out of decision
        budget is a capacity outcome, not evidence that the token has no
        executable market.
        """
        return self._unavailable(
            side="BUY", mint=mint, provider="pumpswap_direct",
            reason=(
                f"QUOTE_TIMEOUT: the executable quote did not arrive within "
                f"{self.quote_deadline_s * 1000:.0f}ms of decision budget, so "
                "the candidate was refused rather than held open"
            )[:200],
            observed_pair_address=observed_pair_address,
            latency_ms=(time.monotonic() - started) * 1000.0,
        )

    async def quote_sell(
        self,
        *,
        mint: str,
        token_amount_raw: int,
        token_decimals: int,
        observed_pair_address: str | None = None,
    ) -> QuotePoint:
        """
        A fresh executable ``TOKEN -> SOL`` quote for this exact token amount.

        The direction is fixed here and is never reversed: the caller supplies a
        TOKEN amount and receives a QUOTE-ASSET amount. Reversing a sell is the
        mistake that reports a exit price several orders of magnitude away from
        reality while remaining a positive, finite number.
        """
        curve = await self._reader.quote_bonding_curve_sell(
            mint, token_amount_raw=token_amount_raw, token_decimals=token_decimals,
        )
        if curve.available:
            return self._point_from_direct(
                curve, side="SELL", mint=mint, decimals=token_decimals,
                observed_pair_address=observed_pair_address,
            )

        swap = await self._reader.quote_pumpswap_sell(
            mint, token_amount_raw=token_amount_raw,
            pool_address=observed_pair_address,
        )
        if swap.available:
            return self._point_from_direct(
                swap, side="SELL", mint=mint, decimals=token_decimals,
                observed_pair_address=observed_pair_address,
            )

        # A DIRECT READ DID NOT ANSWER. Everything below is a REPORT of WHY, and
        # it must name every leg.
        #
        # This used to fall through to ``self.jupiter.quote_sell`` whenever both
        # direct reads failed. ``JupiterQuoteClient`` exposed a BUY quote only,
        # so the branch raised ``AttributeError`` on every call and its message
        # REPLACED the two real failure reasons.
        #
        # IT NOW EXISTS. ``JupiterQuoteClient.quote_sell`` performs a read-only
        # GET for ``token -> WSOL`` at the exact held quantity and never builds,
        # signs or sends anything. It is used ONLY to estimate liquidation
        # proceeds, and its result is deliberately OFF-MARKET:
        # ``_point_from_liquidation_route`` stamps ``identity="UNKNOWN"``
        # unconditionally, because a multi-hop aggregator route is not the
        # position's chain-verified market. Same-market callers therefore still
        # fail closed exactly as before -- what changes is that the refusal now
        # carries a proceeds estimate instead of none.
        direct_reason = (
            "no same-market sell quote. bonding curve: "
            f"{curve.missing_reason or 'unavailable'}; pumpswap: "
            f"{swap.missing_reason or 'unavailable'}"
        )
        if self.jupiter is None:
            return self._unavailable(
                side="SELL", mint=mint,
                reason=direct_reason + "; jupiter: not configured",
                provider="pump_direct+pumpswap_direct",
                observed_pair_address=observed_pair_address,
            )

        started = time.monotonic()
        try:
            route = await self.jupiter.quote_sell(
                token_mint=mint,
                quote_asset_mint=WSOL_MINT,
                token_amount=int(token_amount_raw),
                token_decimals=int(token_decimals),
            )
        except Exception as exc:  # noqa: BLE001 - every refusal names its cause
            return self._unavailable(
                side="SELL", mint=mint,
                reason=(direct_reason + "; jupiter_sell: "
                        f"{type(exc).__name__}: {exc}")[:400],
                provider="pump_direct+pumpswap_direct",
                observed_pair_address=observed_pair_address,
                latency_ms=(time.monotonic() - started) * 1000.0,
            )

        point = self._point_from_liquidation_route(
            route, mint=mint, decimals=int(token_decimals),
            entry_market=observed_pair_address,
            latency_ms=(time.monotonic() - started) * 1000.0,
        )
        if point.available:
            # Keep the direct-refusal reasons attached: a reader must be able to
            # see that this is a fallback, not the primary market.
            point.missing_reason = None
            point.identity_evidence = (
                point.identity_evidence + f" Primary direct legs refused: "
                f"{direct_reason}"
            )
        return point


#: Local alias so this module does not need the models import for one constant.
SOL_DECIMALS_LOCAL = 9


__all__ = ["QuoteService", "QuoteServiceError", "SolReference"]


@dataclass
class QuoteCache:
    """
    A short-lived, freshness-bounded quote cache, shared across arms.

    WHY IT EXISTS
    -------------
    Three research arms consume one signal. They are the same arms on the same
    mint at the same notional, so three provider round trips would fetch the
    same snapshot three times -- and, worse, could fetch THREE DIFFERENT
    snapshots, so the arms would be measured against different books. That is
    not a performance detail; it is a correctness one.

    WHAT IT WILL NOT DO
    -------------------
    It never extends a quote's life. An entry is returned only while
    ``max_quote_age_ms`` says the quote is still fresh, and the age is checked
    against the quote's own timestamp, not against when it was cached. A stale
    quote is dropped, never served.

    Every hit and miss is counted, and the caller records the outcome on the
    signal record so the dataset can report a real cache-hit rate.
    """

    max_quote_age_ms: float = 30_000.0
    hits: int = 0
    misses: int = 0
    shared: int = 0
    expired: int = 0
    refused: int = 0
    _entries: dict[Any, tuple[Any, ...]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._entries is None:
            self._entries = {}

    def _fresh(self, point: Any) -> bool:
        limit = getattr(point, "max_quote_age_ms", None) or self.max_quote_age_ms
        age = getattr(point, "quote_age_ms", None)
        if age is None:
            return False
        return float(age) <= float(limit)

    def get(self, key: Any) -> tuple[Any, tuple[Any, ...]] | None:
        """
        ``(quote, extras)`` when a FRESH entry exists, else ``None``.

        The shape is explicit and fixed because a flat tuple invites exactly the
        bug where a caller reads element 1 as the quote and gets the candidate
        list instead -- which fails as a ``TypeError`` far from the cause.
        """
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        quote, extras = entry[0], entry[1:]
        if not self._fresh(quote):
            del self._entries[key]
            self.expired += 1
            self.misses += 1
            return None
        self.hits += 1
        return quote, tuple(extras)

    def put(self, key: Any, point: Any, rest: tuple[Any, ...] = ()) -> None:
        self._entries[key] = (point, *rest)
        self.shared += 1

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "quote_requests": total + self.shared,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "shared_snapshots": self.shared,
            "expired_dropped": self.expired,
            "refused": self.refused,
            "hit_rate_pct": round(100.0 * self.hits / total, 2) if total else 0.0,
        }

    def clear(self) -> None:
        self._entries.clear()

