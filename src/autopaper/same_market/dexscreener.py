"""
DexScreener observation: enumerate every candidate pair, then select ONE
explicitly and record why.

THE POINT OF THIS MODULE
========================

``dexscreener_pair_selection_reason`` is not a diagnostic afterthought; it is
the thing being measured. A memecoin mint frequently has several live pairs -
a Pump.fun curve, a graduated PumpSwap pool, a Raydium or Meteora pool -
sometimes on the same venue. They price the same token differently, sometimes
wildly so.

Picking the highest-liquidity pair and calling the result "the signal's market"
is the error this harness exists to catch, so selection is a reported decision
with these rules, in order:

1. **The signal token must be the base token.** A pair quoted the other way
   round is a different observation, not a cheaper way to get the same one.
2. **A venue named by the signal wins.** Corroborated against the pair's actual
   ``dexId``, never taken on the message's word alone.
3. **One candidate surviving is selected**, with the reason recorded.
4. **Several survivors with no disambiguating signal text is
   ``PAIR_AMBIGUOUS``.** The top-liquidity pair is still reported as a clearly
   labelled diagnostic so the operator can see the shape of the situation, but
   the identity verdict is ``UNKNOWN`` and the run cannot PASS. Refusing is
   the honest outcome: guessing here is what produces a confident, wrong price
   comparison.

Nothing in this module writes, signs, or submits anything.
"""

from __future__ import annotations

import time
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..market.ratelimit import RateLimiter, RetryAfterBackoff
from ..safety import guard_url
from .models import DexPair, FailureMode, MarketKind, ObservedMark

DEXSCREENER_BASE = "https://api.dexscreener.com"
SOLANA = "solana"

#: DexScreener documents 60 requests/minute for the public API. The limiter is
#: set below that because retries share the same budget.
MAX_RPM = 45

#: Venue identifiers DexScreener uses, mapped to the canonical name the rest of
#: the harness uses. Reported so a reader is never comparing a DexScreener
#: label against a Jupiter one and wondering whether they are the same venue.
VENUE_ALIASES: dict[str, str] = {
    "pumpfun": "pumpfun",
    "pump": "pumpfun",
    "pumpfunlaunchpad": "pumpfun",
    "pumpswap": "pumpswap",
    "pump-amm": "pumpswap",
    "raydium": "raydium",
    "raydiumclmm": "raydium",
    "raydiumlaunchlab": "raydium",
    "meteora": "meteora",
    "meteora-dbc": "meteora",
    "meteora-dlmm": "meteora",
    "moonshot": "moonshot",
    "moonshotlaunch": "moonshot",
    "orca": "orca",
    "whales": "whales",
    "believe": "believe",
    "boopfun": "boop",
    "heaven": "heaven",
    "sugar": "sugar",
    "launchlab": "launchlab",
    # Jupiter's label for the PumpSwap AMM (program pAMMBay6...). DexScreener
    # calls the same pool "pumpswap"; both reach here as the same slug.
    "pumpfunamm": "pumpswap",
    "pump.amm": "pumpswap",
    "pumpfun amm": "pumpswap",
    "boop": "boop",
    "letsbonk": "letsbonk",
    "moonit": "moonit",
}


def _slug(dex_id: str | None) -> str:
    """
    Reduce a venue label to comparable characters.

    Providers do not agree on spelling or punctuation: the same PumpSwap pool is
    ``pumpswap`` on DexScreener and ``Pump.fun Amm`` in a Jupiter route. A naive
    string compare calls that a venue MISMATCH for what is demonstrably one
    pool, which is a false alarm on exactly the case this harness exists to
    resolve.
    """
    if not dex_id:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(dex_id).lower())


def normalise_venue(dex_id: str | None) -> str | None:
    """
    Canonical venue name, or a punctuation-free slug if it is unrecognised.

    Lookup tries the exact id, then the alphanumeric-only slug, so real-world
    spellings map onto the canonical names without a hand-maintained entry for
    every provider's stylistic variation.
    """
    if not dex_id:
        return None
    low = str(dex_id).strip().lower()
    if low in VENUE_ALIASES:
        return VENUE_ALIASES[low]
    slug = _slug(dex_id)
    if slug in VENUE_ALIASES:
        return VENUE_ALIASES[slug]
    return slug or low


class ProviderRateLimited(RuntimeError):
    """The provider refused the request. Surfaced, not retried forever."""


class ProviderUnavailable(RuntimeError):
    """The provider did not answer usefully."""


@dataclass(frozen=True)
class PairSelection:
    """
    The selected pair plus the full reasoning trail.

    ``reason`` is a sentence an operator can audit. ``rule`` is the machine
    name of the rule that fired, so a test can assert on the decision procedure
    itself rather than on English.
    """

    pair: DexPair | None
    rule: str
    reason: str
    candidates: tuple[DexPair, ...]
    considered: int
    ambiguous: bool
    venue_hint: str | None = None
    failure: FailureMode | None = None

    @property
    def diagnostic_pair(self) -> DexPair | None:
        """
        Best candidate to display even when the selection failed.

        Deliberately the deepest-liquidity pair, and deliberately only ever a
        DISPLAY value. Reporting it alongside ``ambiguous=True`` keeps the
        report useful without letting it masquerade as the selected market.
        """
        if self.pair is not None:
            return self.pair
        poolable = [p for p in self.candidates if (p.liquidity_usd or 0.0) > 0]
        if poolable:
            return max(poolable, key=lambda p: p.liquidity_usd or 0.0)
        return self.candidates[0] if self.candidates else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "reason": self.reason,
            "ambiguous": self.ambiguous,
            "considered": self.considered,
            "venue_hint": self.venue_hint,
            "selected_pair_address": self.pair.pair_address if self.pair else None,
            "diagnostic_pair_address": (
                self.diagnostic_pair.pair_address if self.diagnostic_pair else None
            ),
            "failure": self.failure.value if self.failure else None,
            "candidates": [p.as_dict() for p in self.candidates],
        }


class DexScreenerProbe:
    """
    Read-only DexScreener client.

    Reads only ``/latest/dex/tokens/{mint}``. It has no method that can place
    an order, and every request URL passes the shared read-only URL guard.
    """

    name = "dexscreener"

    def __init__(
        self,
        *,
        timeout_s: float = 8.0,
        max_rpm: int = MAX_RPM,
        cache_ttl_s: float = 1.0,
        max_retries: int = 2,
        http_get: Any = None,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.cache_ttl_s = float(cache_ttl_s)
        self.limiter = RateLimiter(max_rpm, name=self.name)
        self.backoff = RetryAfterBackoff(max_attempts=max_retries,
                                         budget_s=self.timeout_s)
        self._http_get = http_get
        self._cache: dict[str, tuple[float, Mapping[str, Any]]] = {}
        self.last_latency_ms: float | None = None
        self.last_from_cache = False

    # -- transport ---------------------------------------------------------

    def _fetch(self, url: str) -> Mapping[str, Any]:
        guard_url(url)
        if self._http_get is not None:
            return self._http_get(url, self.timeout_s)
        import httpx

        with httpx.Client(
            timeout=httpx.Timeout(self.timeout_s, connect=min(2.0, self.timeout_s)),
            headers={
                "accept": "application/json",
                "user-agent": "AUTONOMOUS_PAPER_AGENT_V1/1.0 "
                              "(same-market read-only test)",
            },
            follow_redirects=True,
            # Retries live in RetryAfterBackoff where they are visible. A hidden
            # retry inside the transport would make the measured latency a lie.
            transport=httpx.HTTPTransport(retries=0),
        ) as client:
            try:
                resp = client.get(url)
            except httpx.TimeoutException as exc:
                raise ProviderUnavailable("dexscreener timed out") from exc
            except httpx.HTTPError as exc:
                raise ProviderUnavailable("dexscreener transport error") from exc
            if resp.status_code == 429:
                raise ProviderRateLimited("dexscreener rate limited (HTTP 429)")
            if resp.status_code >= 400:
                raise ProviderUnavailable(f"dexscreener HTTP {resp.status_code}")
            try:
                body = resp.json()
            except ValueError as exc:
                raise ProviderUnavailable("dexscreener returned non-JSON") from exc
            if not isinstance(body, dict):
                raise ProviderUnavailable("dexscreener returned a non-object body")
            return body

    def pairs_for(self, mint: str) -> tuple[list[DexPair], float | None, bool]:
        """
        Every candidate pair for a mint, with the request latency.

        Returns ``(pairs, latency_ms, from_cache)``. The cache exists so the
        identity check and the mark can be served from one request; a cached
        answer is labelled as such so it can never be mistaken for a fresh one.
        """
        url = f"{DEXSCREENER_BASE}/latest/dex/tokens/{mint}"
        now = time.time()
        cached = self._cache.get(mint)
        if cached is not None and now - cached[0] <= self.cache_ttl_s:
            self.last_latency_ms = 0.0
            self.last_from_cache = True
            return (parse_pairs(cached[1], mint), 0.0, True)

        self.limiter.acquire(now)
        started = time.monotonic()
        payload = self.backoff.run(lambda: self._fetch(url))
        latency_ms = max(0.0, (time.monotonic() - started) * 1000.0)
        self.last_latency_ms = latency_ms
        self.last_from_cache = False
        self._cache[mint] = (time.time(), payload)
        return parse_pairs(payload, mint), latency_ms, False

    # -- selection ---------------------------------------------------------

    def select_pair(
        self,
        mint: str,
        *,
        venue_hint: str | None = None,
        allow_ambiguous: bool = False,
    ) -> PairSelection:
        """
        Choose the pair the signal refers to, by an explicit recorded rule.

        See the module docstring for the rule order. ``allow_ambiguous`` exists
        only for the failure-path test; it does not relax the identity verdict,
        which is computed separately from the selection.
        """
        try:
            pairs, _latency, _cached = self.pairs_for(mint)
        except ProviderRateLimited as exc:
            return PairSelection(
                pair=None,
                rule="provider_error",
                reason=f"dexscreener refused the request: {exc}",
                candidates=(),
                considered=0,
                ambiguous=False,
                venue_hint=venue_hint,
                failure=FailureMode.PROVIDER_RATE_LIMIT,
            )
        except ProviderUnavailable as exc:
            return PairSelection(
                pair=None,
                rule="provider_error",
                reason=f"dexscreener unavailable: {exc}",
                candidates=(),
                considered=0,
                ambiguous=False,
                venue_hint=venue_hint,
                failure=FailureMode.PAIR_NOT_FOUND,
            )

        return select_pair_from(
            pairs, mint=mint, venue_hint=venue_hint, allow_ambiguous=allow_ambiguous
        )


# ---------------------------------------------------------------------------
# Pure selection logic - no I/O, so the decision procedure is testable alone
# ---------------------------------------------------------------------------


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def parse_pair(raw: Mapping[str, Any]) -> DexPair | None:
    """One wire pair -> :class:`DexPair`. ``None`` when it is unusable."""
    if not isinstance(raw, Mapping):
        return None
    base = raw.get("baseToken") or {}
    quote = raw.get("quoteToken") or {}
    liquidity = raw.get("liquidity") or {}
    volume = raw.get("volume") or {}
    fdv = raw.get("fdv")
    mcap = raw.get("marketCap")
    txns_5m = (raw.get("txns") or {}).get("m5") or {}
    txns_1h = (raw.get("txns") or {}).get("h1") or {}
    pair_address = str(raw.get("pairAddress") or "").strip()
    if not pair_address:
        return None
    return DexPair(
        chain_id=str(raw.get("chainId") or ""),
        dex_id=str(raw.get("dexId") or ""),
        pair_address=pair_address,
        base_token_address=str(base.get("address") or "") or None,
        base_token_symbol=base.get("symbol"),
        quote_token_address=str(quote.get("address") or "") or None,
        quote_token_symbol=quote.get("symbol"),
        price_usd=_f(raw.get("priceUsd")),
        price_native=_f(raw.get("priceNative")),
        liquidity_usd=_f(liquidity.get("usd")),
        volume_24h_usd=_f(volume.get("h24")),
        fdv_usd=_f(fdv),
        market_cap_usd=_f(mcap),
        pair_created_at=_f(raw.get("pairCreatedAt")),
        txns_5m=txns_5m.get("count"),
        txns_1h=txns_1h.get("count"),
        buys_5m=txns_5m.get("buys"),
        sells_5m=txns_5m.get("sells"),
        buys_1h=txns_1h.get("buys"),
        sells_1h=txns_1h.get("sells"),
        url=raw.get("url"),
    )


def parse_pairs(payload: Mapping[str, Any], mint: str) -> list[DexPair]:
    """All parseable pairs for a mint, in the provider's own order preserved."""
    raw_pairs = payload.get("pairs") if isinstance(payload, Mapping) else None
    if not isinstance(raw_pairs, list):
        return []
    out: list[DexPair] = []
    for raw in raw_pairs:
        pair = parse_pair(raw) if isinstance(raw, Mapping) else None
        if pair is not None:
            out.append(pair)
    return out


def select_pair_from(
    pairs: Sequence[DexPair],
    *,
    mint: str,
    venue_hint: str | None = None,
    allow_ambiguous: bool = False,
) -> PairSelection:
    """
    The selection rules, as a pure function of the candidate list.

    Split out from the HTTP client so the decision procedure can be tested
    against hand-built candidate sets, including the awkward ones: several
    pairs at similar liquidity, a venue hint that matches nothing, a pair
    quoted the wrong way round, and a token with one pair on an unrelated chain.
    """
    on_solana = [p for p in pairs if p.chain_id == SOLANA]
    if not on_solana:
        return PairSelection(
            pair=None,
            rule="no_solana_pair",
            reason=(
                f"none of the {len(pairs)} candidate pair(s) is on solana; a "
                "solana signal cannot be priced by a non-solana pool"
            ),
            candidates=tuple(pairs),
            considered=len(pairs),
            ambiguous=False,
            venue_hint=venue_hint,
            failure=FailureMode.PAIR_NOT_FOUND,
        )

    # Rule 1: the signal's token must be the base token. A pair quoting the
    # signal token as its QUOTE asset is a different market and its priceUsd is
    # the reciprocal of the one we want.
    base_side = [p for p in on_solana if p.base_token_address == mint]
    if not base_side:
        quoted_as_quote = [p for p in on_solana if p.quote_token_address == mint]
        return PairSelection(
            pair=None,
            rule="token_not_base",
            reason=(
                f"the token appears as the QUOTE asset in {len(quoted_as_quote)} "
                "pair(s) and as the BASE asset in none; its priceUsd there is "
                "the reciprocal of the quote this harness needs"
            ),
            candidates=tuple(on_solana),
            considered=len(on_solana),
            ambiguous=False,
            venue_hint=venue_hint,
            failure=FailureMode.TOKEN_MINT_MISMATCH,
        )

    priced = [p for p in base_side if p.price_usd]
    if not priced:
        return PairSelection(
            pair=None,
            rule="no_price",
            reason="every candidate pair is missing priceUsd; there is no mark",
            candidates=tuple(base_side),
            considered=len(base_side),
            ambiguous=False,
            venue_hint=venue_hint,
            failure=FailureMode.PAIR_NOT_FOUND,
        )

    # Rule 2: an explicit venue named by the signal, corroborated by dexId.
    if venue_hint:
        wanted = normalise_venue(venue_hint)
        matching = [
            p for p in priced
            if normalise_venue(p.dex_id) == wanted
            or (p.dex_id or "").lower() == str(venue_hint).lower()
        ]
        if len(matching) == 1:
            return PairSelection(
                pair=matching[0],
                rule="signal_named_venue",
                reason=(
                    f"the signal named DEX {venue_hint!r}; exactly one candidate "
                    f"pair is served by {matching[0].dex_id!r}, so the "
                    "signal's market is identified without guessing"
                ),
                candidates=tuple(base_side),
                considered=len(base_side),
                ambiguous=False,
                venue_hint=venue_hint,
            )
        if len(matching) > 1:
            best = _deepest(matching)
            return PairSelection(
                pair=best,
                rule="signal_named_venue_multiple",
                reason=(
                    f"the signal named DEX {venue_hint!r} but {len(matching)} "
                    "candidate pairs share that venue; selected the "
                    "deepest-liquidity of them, which the signal alone does not "
                    "identify - recorded as an assumption, not a proof"
                ),
                candidates=tuple(base_side),
                considered=len(base_side),
                ambiguous=True,
                venue_hint=venue_hint,
                failure=FailureMode.PAIR_AMBIGUOUS,
            )
        # A venue hint that matches nothing is a real disagreement between the
        # signal and the venue, and must be reported rather than ignored.
        return PairSelection(
            pair=None,
            rule="venue_hint_unmatched",
            reason=(
                f"the signal named DEX {venue_hint!r} but no candidate pair is "
                f"served by it (found: "
                f"{sorted({p.dex_id for p in priced})}); refusing to substitute a "
                "different venue for the one the signal named"
            ),
            candidates=tuple(base_side),
            considered=len(base_side),
            ambiguous=False,
            venue_hint=venue_hint,
            failure=FailureMode.VENUE_MISMATCH,
        )

    # Rule 3: a single survivor is identified by elimination.
    if len(priced) == 1:
        only = priced[0]
        return PairSelection(
            pair=only,
            rule="sole_candidate",
            reason=(
                "exactly one solana pair has this token as its base token, so "
                "the signal's market is identified without guessing"
            ),
            candidates=tuple(base_side),
            considered=len(base_side),
            ambiguous=False,
        )

    # Rule 4: several survivors and nothing in the signal to separate them.
    deepest = _deepest(priced)
    others = [p for p in priced if p.pair_address != deepest.pair_address]
    runner_up_liq = max((p.liquidity_usd or 0.0) for p in others)
    deep_liq = deepest.liquidity_usd or 0.0
    gap = (deep_liq - runner_up_liq) / deep_liq if deep_liq > 0 else 0.0

    if allow_ambiguous:
        return PairSelection(
            pair=deepest,
            rule="ambiguous_forced",
            reason=(
                "AMBIGUITY DELIBERATELY ACCEPTED for the failure-path test only; "
                f"{len(priced)} candidate pools and no signal-level disambiguation"
            ),
            candidates=tuple(base_side),
            considered=len(base_side),
            ambiguous=True,
            failure=FailureMode.PAIR_AMBIGUOUS,
        )

    return PairSelection(
        pair=deepest,
        rule="ambiguous_multiple_pools",
        reason=(
            f"{len(priced)} solana pools carry this token as base and the signal "
            "identifies none of them; selecting one would be a guess. The "
            f"deepest ({deep_liq:,.0f} USD, {deepest.dex_id}, gap to the next "
            f"{gap * 100:.1f}%) is shown as a DIAGNOSTIC only. Market identity "
            "is UNKNOWN and this signal cannot PASS"
        ),
        candidates=tuple(base_side),
        considered=len(base_side),
        ambiguous=True,
        failure=FailureMode.PAIR_AMBIGUOUS,
    )


def _deepest(pairs: Sequence[DexPair]) -> DexPair:
    return max(pairs, key=lambda p: (p.liquidity_usd or 0.0, p.price_usd or 0.0))


def observed_mark_from(
    selection: PairSelection,
    *,
    source: str = "dexscreener",
    timestamp: float | None = None,
    latency_ms: float | None = None,
) -> ObservedMark:
    """
    Build the ``OBSERVED_MARK`` from a selection.

    When the selection failed, the mark is still built from the diagnostic pair
    so the report shows what was on screen - but ``value_usd`` is only populated
    for the SELECTED pair. A diagnostic price presented next to an identity
    failure is exactly the number this harness exists to stop anyone acting on.
    """
    now = time.time() if timestamp is None else timestamp
    pair = selection.pair
    if pair is None:
        return ObservedMark(
            value_usd=None,
            pair_address=None,
            dex_id=None,
            source=source,
            timestamp=now,
            latency_ms=latency_ms,
            failure=selection.failure or FailureMode.PAIR_NOT_FOUND,
        )
    return ObservedMark(
        value_usd=pair.price_usd,
        pair_address=pair.pair_address,
        dex_id=pair.dex_id,
        source=source,
        timestamp=now,
        latency_ms=latency_ms,
        request_latency_ms=latency_ms,
    )


def market_kind_for(dex_id: str | None) -> MarketKind:
    """
    What KIND of market a venue implies - a hint, never a determination.

    The chain read in :mod:`autopaper.same_market.pump` decides this for real.
    This only labels what to try, and a token that has graduated carries a
    DexScreener ``pumpfun`` label while trading on PumpSwap.
    """
    venue = normalise_venue(dex_id)
    if venue == "pumpfun":
        return MarketKind.PUMP_BONDING_CURVE
    if venue == "pumpswap":
        return MarketKind.PUMPSWAP
    return MarketKind.UNKNOWN


__all__ = [
    "DEXSCREENER_BASE",
    "DexScreenerProbe",
    "PairSelection",
    "ProviderRateLimited",
    "ProviderUnavailable",
    "SOLANA",
    "VENUE_ALIASES",
    "market_kind_for",
    "normalise_venue",
    "observed_mark_from",
    "parse_pair",
    "parse_pairs",
    "select_pair_from",
]