"""
SOL/USD reference price provider.

WHAT THIS IS
============
A single USD->SOL conversion reference. Position sizing is specified in USD
while execution happens in SOL, so every USD notional in this repository has to
cross exactly one conversion:

    USD notional  ->  [ SOL/USD reference ]  ->  SOL input amount  ->  token quote

This module supplies the middle value and NOTHING else. It is not a token
pricing source, it is not an execution price, and it is never allowed to become
one.

WHAT THIS IS NOT
================
The following are hard prohibitions, enforced by construction and by test:

* It NEVER derives SOL/USD from a token's DexScreener mark. The mark is
  ``OBSERVED_MARK``. Using it here would make the size and the fill price come
  from the same observation.
* It NEVER derives SOL/USD from a token's executable quote. That is a
  token-vs-SOL price; SOL-vs-USD is a different question.
* It NEVER guesses, defaults, interpolates or carries forward a stale price to
  fill a gap. If the reference cannot be obtained, the caller is REFUSED.
* It NEVER introduces a paid provider. Jupiter's public keyless endpoint only.

WHY A SEPARATE PROVIDER
=======================
Because the honest reason the old code demanded ``--sol-usd`` was
reproducibility, and that reason is fully served by *recording* the snapshot --
price, source and timestamp -- not by making the operator type it. An operator
supplied number and a fetched number are the same thing once both carry their
provenance and the arithmetic is logged beside the result.
"""

from __future__ import annotations

import inspect
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from ..market.ratelimit import RateLimiter
# WSOL_MINT comes from the neutral models module, NOT from a quoting module:
# this provider must not depend on the executable-quote layer even for a
# constant, or the two stages would be coupled.
from .models import WSOL_MINT

#: Jupiter's PUBLIC, KEYLESS price endpoint. Same free tier the rest of this
#: repository already uses for quotes (``lite-api.jup.ag``). No paid plan, no
#: metered provider, no new dependency.
JUPITER_PRICE_BASE = "https://lite-api.jup.ag/price/v3"

#: Environment variables consulted for an OPTIONAL Jupiter key. The free
#: keyless endpoint works without one; a key only raises the rate ceiling.
#: The value is read once, held privately, and NEVER rendered, logged or
#: serialised. ``__repr__`` redacts it and ``as_dict`` has no field for it.
JUPITER_KEY_ENV_VARS = ("JUPITER_API_KEY", "JUPITER_PRICE_API_KEY")

#: Keyless requests are documented at 0.5 RPS; with a key the free tier is
#: 1 RPS. This module makes ONE request per process at most (the snapshot is
#: cached), so the limit is about not being the thing that trips a shared cap.
DEFAULT_MAX_RPM = 20

#: Default snapshot lifetime. A SOL/USD reference is used to size notionals; it
#: does not need to be tick-fresh. One minute bounds the worst-case drift in the
#: sizing arithmetic while still being far longer than a single run.
DEFAULT_TTL_S = 60.0

#: Sanity band. Not a trading threshold and not a gate -- a SOL/USD reference
#: outside this band means the response was misparsed or the provider is broken,
#: and using it would silently mis-size every notional in the run. Failing closed
#: is the only safe response.
MIN_PLAUSIBLE_USD = 0.01
MAX_PLAUSIBLE_USD = 100_000.0

SOURCE_OPERATOR = "operator (--sol-usd-override)"
SOURCE_JUPITER = "jupiter price api v3 (lite-api.jup.ag)"
SOURCE_TEST = "test fixture"


class SolPriceUnavailable(RuntimeError):
    """
    The SOL/USD reference could not be obtained.

    Callers MUST treat this as fatal for the sizing step. There is deliberately
    no ``fallback`` attribute and no partial result: a caller that wants a
    default must pass an explicit override, which is auditable.
    """


@dataclass(frozen=True)
class SolUsdSnapshot:
    """
    One fetched (or operator-supplied) SOL/USD reference, with provenance.

    Immutable and self-describing, so it can be stored next to the results that
    depended on it and the arithmetic stays reproducible.
    """

    usd_per_sol: float
    #: When the value was observed. For a fetch this is the fetch instant; for
    #: an override it is the instant the run started, which is the honest
    #: interpretation (the operator asserted it is current NOW).
    timestamp: float
    #: Human-readable provenance. NEVER contains key material.
    source: str
    #: Jupiter's ``blockId`` for the price, when the provider supplied one.
    block_id: Optional[int] = None
    #: True when the value came from the provider rather than an operator.
    fetched: bool = True

    def age_ms(self, now: Optional[float] = None) -> float:
        """Age of this snapshot in milliseconds. Never negative."""
        ref = time.time() if now is None else float(now)
        return max(0.0, (ref - self.timestamp) * 1000.0)

    def as_dict(self, now: Optional[float] = None) -> dict[str, Any]:
        """
        The metadata block recorded beside results.

        Field names are the ones the reporting contract asks for. No token price,
        executable quote, fill or PnL field is touched by this.
        """
        return {
            "sol_usd": self.usd_per_sol,
            "sol_usd_source": self.source,
            "sol_usd_timestamp": self.timestamp,
            "sol_usd_age_ms": self.age_ms(now),
            "sol_usd_block_id": self.block_id,
            "sol_usd_fetched": self.fetched,
        }

    def __repr__(self) -> str:  # pragma: no cover - defensive
        return (f"SolUsdSnapshot(usd_per_sol={self.usd_per_sol!r}, "
                f"source={self.source!r})")


class JupiterSolPriceProvider:
    """
    Read-only SOL/USD price adapter over Jupiter's free price API.

    Design constraints, each of which is load-bearing:

    * **Quote-only.** GET a public price. No build, no sign, no send. It cannot
      place an order even if asked to.
    * **Keyless by default.** Uses ``lite-api.jup.ag``. An optional key raises
      the rate ceiling and is held privately.
    * **Cached.** One request per TTL per process, so multiple arms and signals
      in a run do not duplicate requests against a keyless shared cap.
    * **Fails closed.** Every failure path raises :class:`SolPriceUnavailable`.
      There is no cached-stale fallback and no default.
    """

    name = "jupiter-sol-price"

    def __init__(
        self,
        *,
        base_url: str = JUPITER_PRICE_BASE,
        timeout_s: float = 8.0,
        max_rpm: int = DEFAULT_MAX_RPM,
        ttl_s: float = DEFAULT_TTL_S,
        mint: str = WSOL_MINT,
        api_key: Optional[str] = None,
        clock: Optional[Callable[[], float]] = None,
        http_get: Optional[Callable[..., Any]] = None,
        client: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.ttl_s = float(ttl_s)
        self.mint = mint
        self.limiter = RateLimiter(max_rpm, name=self.name)
        self._clock = clock or time.time
        self._http_get = http_get
        self._client = client
        self._own_client = client is None

        # Private, and deliberately absent from every dict/repr this class
        # produces. Only ever used to build a request header.
        self._api_key = (api_key or self._key_from_env() or "").strip() or None

        self._cached: Optional[SolUsdSnapshot] = None
        #: Observability, never used for control flow.
        self.total_requests = 0
        self.cache_hits = 0
        self.last_error: Optional[str] = None

    # -- key handling ----------------------------------------------------

    @staticmethod
    def _key_from_env() -> Optional[str]:
        """Read an optional key. Never logs, never returns it for display."""
        for name in JUPITER_KEY_ENV_VARS:
            value = os.environ.get(name)
            if value and value.strip():
                return value.strip()
        return None

    @property
    def has_api_key(self) -> bool:
        """Whether a key is configured. Reports the FACT, never the value."""
        return self._api_key is not None

    def _headers(self) -> dict[str, str]:
        headers = {
            "accept": "application/json",
            "user-agent": "AUTONOMOUS_PAPER_AGENT_V1/1.0 (sol-usd reference)",
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key
        return headers

    def _redact(self, text: str) -> str:
        """
        Remove any key material from a provider message before it is stored or
        raised. Defence in depth: httpx can echo a header in an error string.
        """
        out = str(text)
        if self._api_key:
            out = out.replace(self._api_key, "***")
        # Also scrub anything that looks like the header value in a URL/param.
        for name in JUPITER_KEY_ENV_VARS:
            val = os.environ.get(name)
            if val and len(val.strip()) >= 8:
                out = out.replace(val.strip(), "***")
        return out

    def __repr__(self) -> str:  # pragma: no cover - defensive
        return (f"JupiterSolPriceProvider(base_url={self.base_url!r}, "
                f"has_api_key={self.has_api_key})")

    # -- clock -----------------------------------------------------------

    def now(self) -> float:
        return float(self._clock())

    # -- caching ---------------------------------------------------------

    def cached_snapshot(self) -> Optional[SolUsdSnapshot]:
        """The cached snapshot if it is still inside its TTL, else ``None``."""
        snap = self._cached
        if snap is None:
            return None
        if (self.now() - snap.timestamp) > self.ttl_s:
            return None
        return snap

    # -- fetch -----------------------------------------------------------

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

    async def _fetch_json(self) -> Mapping[str, Any]:
        params = {"ids": self.mint}
        if self._http_get is not None:
            # The injected hook may be sync or async; accept both so a test
            # double never has to know which this class happens to use.
            result = self._http_get(dict(params))
            if inspect.isawaitable(result):
                result = await result
            return result

        # Local rate cap first: a keyless shared ceiling is easy to trip by
        # being careless, and tripping it costs the whole run.
        self.limiter.acquire(self.now())
        client = await self._ensure_client()
        self.total_requests += 1
        resp = await client.get(
            self.base_url,
            params=params,
            timeout=self.timeout_s,
            headers=self._headers(),
        )
        status = int(getattr(resp, "status_code", 0) or 0)
        text = getattr(resp, "text", "") or ""
        if status == 429:
            self.limiter.note_retry_after(self.now(), None)
            raise SolPriceUnavailable(
                "SOL/USD provider rate limited (HTTP 429); not guessing a price")
        if status >= 400:
            raise SolPriceUnavailable(
                self._redact(f"SOL/USD provider HTTP {status}"))
        try:
            body = resp.json()
        except ValueError as exc:
            raise SolPriceUnavailable(
                self._redact(f"SOL/USD provider returned non-JSON: {exc}")) from exc
        if not isinstance(body, dict):
            raise SolPriceUnavailable("SOL/USD provider returned a non-object body")
        return body

    def parse(
        self,
        body: Mapping[str, Any],
        *,
        timestamp: Optional[float] = None,
    ) -> SolUsdSnapshot:
        """
        Parse a Price API v3 body. Pure -- no I/O -- so it is testable against
        fixtures without a network.

        Jupiter OMITS a token that has no reliable price rather than returning
        a null, so a missing key is a real outcome and must fail closed.
        """
        entry = body.get(self.mint)
        if not isinstance(entry, Mapping):
            raise SolPriceUnavailable(
                "SOL/USD provider returned no entry for SOL; the price is "
                "unknown, and this provider never substitutes a guess")
        raw = entry.get("usdPrice")
        if raw is None or isinstance(raw, bool):
            raise SolPriceUnavailable("SOL/USD response had no usdPrice")
        try:
            price = float(raw)
        except (TypeError, ValueError) as exc:
            raise SolPriceUnavailable("SOL/USD usdPrice was not a number") from exc
        if price != price or price in (float("inf"), float("-inf")):
            raise SolPriceUnavailable("SOL/USD usdPrice was not finite")
        if not (MIN_PLAUSIBLE_USD <= price <= MAX_PLAUSIBLE_USD):
            raise SolPriceUnavailable(
                f"SOL/USD of ${price:,.4f} is outside the plausible band "
                f"${MIN_PLAUSIBLE_USD:,.2f}-${MAX_PLAUSIBLE_USD:,.2f}; refusing "
                f"to size a notional against it")
        block = entry.get("blockId")
        return SolUsdSnapshot(
            usd_per_sol=price,
            timestamp=self.now() if timestamp is None else float(timestamp),
            source=SOURCE_JUPITER,
            block_id=int(block) if isinstance(block, int) else None,
            fetched=True,
        )

    async def snapshot(self, *, refresh: bool = False) -> SolUsdSnapshot:
        """
        Return the SOL/USD reference, fetching at most once per TTL.

        Raises :class:`SolPriceUnavailable` on every failure path. Callers must
        not substitute a value; that is the whole contract.
        """
        if not refresh:
            hit = self.cached_snapshot()
            if hit is not None:
                self.cache_hits += 1
                return hit
        try:
            body = await self._fetch_json()
            snap = self.parse(body)
        except SolPriceUnavailable as exc:
            self.last_error = self._redact(str(exc))
            raise
        except Exception as exc:  # noqa: BLE001 - any transport fault fails closed
            self.last_error = self._redact(
                f"{type(exc).__name__}: {exc}")
            raise SolPriceUnavailable(
                f"SOL/USD provider transport failure: "
                f"{self._redact(type(exc).__name__)}") from exc
        self._cached = snap
        self.last_error = None
        return snap


def override_snapshot(
    usd_per_sol: Any,
    *,
    now: Optional[float] = None,
    source: str = SOURCE_OPERATOR,
) -> SolUsdSnapshot:
    """
    Wrap an operator-supplied value as a snapshot.

    This is the OFFLINE / TEST-ONLY deterministic path. It validates exactly as
    the fetched path does -- same band, same "no nonsense" rules -- so an
    override cannot smuggle in a value the provider would have rejected.
    """
    if usd_per_sol is None or isinstance(usd_per_sol, bool):
        raise SolPriceUnavailable("SOL/USD override is not a number")
    try:
        price = float(usd_per_sol)
    except (TypeError, ValueError) as exc:
        raise SolPriceUnavailable("SOL/USD override is not a number") from exc
    if price != price or price in (float("inf"), float("-inf")):
        raise SolPriceUnavailable("SOL/USD override was not finite")
    if not (MIN_PLAUSIBLE_USD <= price <= MAX_PLAUSIBLE_USD):
        raise SolPriceUnavailable(
            f"SOL/USD override of ${price:,.4f} is outside the plausible band "
            f"${MIN_PLAUSIBLE_USD:,.2f}-${MAX_PLAUSIBLE_USD:,.2f}")
    return SolUsdSnapshot(
        usd_per_sol=price,
        timestamp=time.time() if now is None else float(now),
        source=source,
        block_id=None,
        fetched=False,
    )


def add_override_arguments(parser: Any, *, env_var: str | None = None) -> None:
    """
    Register the shared CLI surface.

    ``--sol-usd-override`` is the documented offline/test-only deterministic
    input. ``--sol-usd`` is retained ONLY as a deprecated alias for it, so an
    existing script keeps working, and it is never required.

    The old flag is NOT reinterpreted as anything else: it still means "this is
    the SOL/USD reference", which is precisely what the override means.
    """
    help_override = (
        "OFFLINE/TEST-ONLY deterministic SOL/USD reference. Supply this to make "
        "a run reproducible without a network call. Omit it in normal use: the "
        "SOL/USD reference is fetched automatically from the free keyless "
        "Jupiter price API, and the run is refused if it cannot be obtained"
    )
    parser.add_argument(
        "--sol-usd-override",
        dest="sol_usd_override",
        type=float,
        default=None,
        help=help_override,
    )
    parser.add_argument(
        "--sol-usd",
        dest="sol_usd",
        type=float,
        default=None,
        help="DEPRECATED: alias for --sol-usd-override. Kept so existing "
             "commands keep working. It is no longer required in normal use",
    )


def resolve_sol_usd(
    *,
    override: Any = None,
    provider: Optional[JupiterSolPriceProvider] = None,
    deprecated: Any = None,
    now: Optional[float] = None,
) -> SolUsdSnapshot:
    """
    Synchronous resolver: override, else an ALREADY-CACHED snapshot.

    Fetching is async, so this form can only serve the two cases that need no
    I/O: an explicit override, or a snapshot this process fetched earlier. It
    will not block and it will not start a loop. Async callers use
    :func:`resolve_sol_usd_async`, which does the fetch.

    Same two-step contract, no third option, no default.
    """
    chosen = override if override is not None else deprecated
    if chosen is not None:
        src = (SOURCE_OPERATOR if override is not None
               else "operator (--sol-usd, DEPRECATED alias)")
        return override_snapshot(chosen, now=now, source=src)
    cached = provider.cached_snapshot() if provider is not None else None
    if cached is not None:
        if provider is not None:
            provider.cache_hits += 1
        return cached
    raise SolPriceUnavailable(
        "no SOL/USD reference available synchronously: pass "
        "--sol-usd-override for an offline deterministic run, or use the async "
        "resolver to fetch the free keyless Jupiter price API")


async def resolve_sol_usd_async(
    *,
    override: Any = None,
    provider: Optional[JupiterSolPriceProvider] = None,
    deprecated: Any = None,
    now: Optional[float] = None,
) -> SolUsdSnapshot:
    """Async form of :func:`resolve_sol_usd`; identical order and semantics."""
    chosen = override if override is not None else deprecated
    if chosen is not None:
        src = (SOURCE_OPERATOR if override is not None
               else "operator (--sol-usd, DEPRECATED alias)")
        return override_snapshot(chosen, now=now, source=src)
    if provider is None:
        raise SolPriceUnavailable(
            "no SOL/USD reference available: pass --sol-usd-override for an "
            "offline deterministic run, or allow the free keyless Jupiter "
            "price API to be used")
    return await provider.snapshot()


async def resolve_for_cli(
    args: Any,
    *,
    provider: Optional[JupiterSolPriceProvider] = None,
    timeout_s: float = 8.0,
    cache_ttl_s: float = DEFAULT_TTL_S,
    max_rpm: int = DEFAULT_MAX_RPM,
    http_get: Optional[Callable[..., Any]] = None,
) -> SolUsdSnapshot:
    """
    CLI-facing resolver: honour an override, else fetch once and cache.

    Returns a snapshot; raises :class:`SolPriceUnavailable` if neither works.
    Never returns ``None`` and never invents a value.
    """
    override = getattr(args, "sol_usd_override", None)
    deprecated = getattr(args, "sol_usd", None)
    if override is not None or deprecated is not None:
        return await resolve_sol_usd_async(override=override,
                                           deprecated=deprecated)
    prov = provider or JupiterSolPriceProvider(
        timeout_s=timeout_s, ttl_s=cache_ttl_s, max_rpm=max_rpm,
        http_get=http_get,
    )
    return await prov.snapshot()


def describe_snapshot(snap: SolUsdSnapshot, *, now: Optional[float] = None) -> str:
    """One-line provenance for a human. Contains no key material by construction."""
    return (f"${snap.usd_per_sol:,.4f} via {snap.source} "
            f"(fetched {snap.age_ms(now) / 1000.0:.1f}s ago, "
            f"block {snap.block_id})" if snap.block_id is not None else
            f"${snap.usd_per_sol:,.4f} via {snap.source} "
            f"({snap.age_ms(now) / 1000.0:.1f}s old)")