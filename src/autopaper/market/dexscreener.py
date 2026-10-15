"""
DexScreener provider (primary).

Read-only. Uses ``/latest/dex/tokens/{mint}``, picks the deepest liquid Solana
pair where the queried mint is the BASE token, and normalises it into an
``Observation``.

Every numeric value that cannot be obtained is left ``None``. Nothing is
interpolated or guessed.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Mapping

from ..clock import Clock, SystemClock
from ..logging_setup import log
from ..models import DataState, Observation, QuoteResult
from ..safety import guard_url
from .ratelimit import (
    MarketDataError,
    RateLimiter,
    RetryAfterBackoff,
    TTLCache,
    classify_error,
    run_with_deadline,
    split_timeouts,
)

BASE = "https://api.dexscreener.com"
CHAIN = "solana"

# Stable and wrapped assets. Used only to drop STABLE/STABLE and STABLE/native
# pairs, which are not the trade we are modelling.
#
# Note what is deliberately NOT here: a memecoin quoted against USDC or USDT is
# exactly the venue we want - TOKEN/USDC is the deepest pool a new launch has. An
# earlier version excluded stablecoin QUOTE assets outright, which threw away the
# best pool for every realistic candidate (and every pair for a stablecoin mint).
_STABLE_SYMBOLS = {"USDC", "USDT", "USDS", "PYUSD", "DAI", "EURC", "FDUSD", "TUSD",
                   "USDE", "BUSD"}
_WRAPPED_SYMBOLS = {"WBTC", "WETH", "WBTC", "STSOL", "MSOL", "JITOSOL", "WSOL"}


def _f(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _i(value: Any) -> int | None:
    f = _f(value)
    return int(f) if f is not None else None


class DexScreenerProvider:
    name = "dexscreener"

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        max_rpm: int = 110,
        timeout_s: float = 8.0,
        connect_timeout_s: float = 2.0,
        cache_ttl_s: float = 2.0,
        max_retries: int = 2,
        http_get: Callable[[str, float], Mapping[str, Any]] | None = None,
    ) -> None:
        self.clock = clock or SystemClock()
        self.limiter = RateLimiter(max_rpm, name=self.name)
        #: The retry loop's wall-clock budget is the SAME number as the per-call
        #: timeout. Retries then cannot multiply it: a provider call costs at
        #: most ``timeout_s`` in total, however many attempts it makes.
        self.backoff = RetryAfterBackoff(max_attempts=max_retries,
                                         budget_s=timeout_s)
        self.cache = TTLCache()
        self.cache_ttl_s = cache_ttl_s
        self.timeout_s = timeout_s
        self.connect_timeout_s = connect_timeout_s
        # The injected seam keeps its TWO-argument contract - ``(url, timeout_s)``
        # - so a fake transport written against the old signature still works.
        # The per-instance connect budget is bound here, into the default
        # transport only, because an injected transport has no connect phase to
        # configure.
        self._http_get = http_get or (
            lambda url, timeout_s: _default_http_get(
                url, timeout_s, self.connect_timeout_s
            )
        )
        self.last_reject_reason: str | None = None
        # Provider-failure visibility. A timeout that was retried and then
        # succeeded used to leave no trace anywhere, so a provider degrading
        # under load looked identical to a healthy one.
        self.retry_count = 0
        self.last_backoff_s = 0.0
        self.last_error_label: str | None = None
        #: Worst single-call wall clock this provider has actually cost, so the
        #: bound is measured rather than assumed.
        self.max_call_s = 0.0

    def _why_rejected(self, pairs: list, mint: str) -> str:
        """
        Say WHY a mint had no usable pool.

        "no data" and "the only pools are stable/stable, which this agent does
        not model" are very different facts, and only one of them is a bug.
        """
        same_chain = [p for p in pairs if p.get("chainId") == CHAIN]
        if not same_chain:
            return f"no solana pairs at all for {mint} ({len(pairs)} pairs, other chains)"
        as_base = [
            p for p in same_chain
            if ((p.get("baseToken") or {}).get("address") or "") == mint
        ]
        if not as_base:
            return (
                f"mint appears only as a QUOTE token in {len(same_chain)} solana "
                "pair(s); this agent quotes base-token venues only, which is "
                "correct for a memecoin candidate and deliberately not modelled "
                "for a stable asset"
            )
        return (
            f"{len(as_base)} base-token pair(s) but all unusable: stable/stable "
            "or stable/wrapped venues, which this agent does not model"
        )

    # -- public API --------------------------------------------------------

    def quote(self, mint: str, *, stale_after_ms: float = 15_000.0,
              unusable_after_ms: float = 45_000.0) -> QuoteResult:
        started = self.clock.monotonic()
        now = self.clock.now()

        cached = self.cache.get(mint, self.cache_ttl_s, now)
        if cached is not None:
            obs, _age = cached
            return QuoteResult(
                observation=obs,
                provider=self.name,
                latency_ms=0.0,
                from_cache=True,
            )

        try:
            payload = self._fetch(f"{BASE}/latest/dex/tokens/{mint}")
        except MarketDataError as exc:
            # Never fabricate a price. Serve the last good observation as STALE
            # if we have one, otherwise report the failure honestly.
            self.limiter.note_retry_after(self.clock.now(), exc.retry_after_s)
            fallback = self.cache.get_any(mint)
            if fallback is not None:
                # get_any returns (stamped_at, value) - not (value, age)
                _stamped, obs = fallback
                now2 = self.clock.now()
                degraded = _with_received(obs, now2)
                return QuoteResult(
                    observation=degraded,
                    error=f"{self.name}: {exc}",
                    provider=self.name,
                    latency_ms=(self.clock.monotonic() - started) * 1000.0,
                    from_cache=True,
                    rate_limited=exc.rate_limited,
                    retry_after_s=exc.retry_after_s,
                )
            return QuoteResult(
                observation=None,
                error=f"{self.name}: {exc}",
                provider=self.name,
                latency_ms=(self.clock.monotonic() - started) * 1000.0,
                rate_limited=exc.rate_limited,
                retry_after_s=exc.retry_after_s,
            )

        self.last_reject_reason = None
        obs = self.parse(payload, mint, self.clock.now())
        if obs is not None:
            self.cache.put(mint, obs, self.clock.now())
        return QuoteResult(
            observation=obs,
            error=(
                None if obs
                else f"{self.name}: no usable pool for {mint}: "
                     f"{self.last_reject_reason or 'unclassified'}"
            ),
            provider=self.name,
            latency_ms=(self.clock.monotonic() - started) * 1000.0,
        )

    async def quote_async(
        self,
        mint: str,
        *,
        stale_after_ms: float = 15_000.0,
        unusable_after_ms: float = 45_000.0,
    ) -> QuoteResult:
        """The event-loop-safe form of :meth:`quote`.

        The provider transport is SYNCHRONOUS (httpx.Client with a retry loop),
        so calling it from the asyncio loop blocks every other task - including
        the monitor and the Telegram read stream - for up to
        ``timeout_s * (max_retries + 1)`` seconds. With the configured 8s timeout
        and 2 retries that is ~16s of event-loop starvation per call, and the
        observed ~34s analysis tail (with a 940s outlier) is that starvation
        accumulating.

        ``asyncio.to_thread`` runs the EXISTING synchronous implementation on a
        worker thread. Nothing about the fetch changes: same URL, same retry
        semantics, same parsing, same timeout values, same cache. Only the
        thread it runs on changes, so the loop is never blocked.
        """
        return await asyncio.to_thread(
            self.quote,
            mint,
            stale_after_ms=stale_after_ms,
            unusable_after_ms=unusable_after_ms,
        )

    def quote_many(self, mints: list[str]) -> dict[str, QuoteResult]:
        """Batch lookup via the search endpoint, kept deliberately small."""
        out: dict[str, QuoteResult] = {}
        for mint in mints[:20]:
            out[mint] = self.quote(mint)
        return out

    def retry_stats(self) -> dict[str, Any]:
        """Provider-failure counters, for the market health report."""
        return {
            "retry_count": self.retry_count,
            "last_backoff_s": round(self.last_backoff_s, 4),
            "last_error_label": self.last_error_label,
        }

    # -- internals ---------------------------------------------------------

    def _fetch(self, url: str) -> Mapping[str, Any]:
        """
        Fetch and decode, inside a WALL-CLOCK budget.

        The injected ``http_get`` seam has ONE contract: return the decoded
        payload as a Mapping, or raise ``MarketDataError``. Status codes, 429
        handling and decoding live in the default transport and in this method,
        not in every caller - which is what makes the seam trivially fakeable in
        tests without a network.

        Three separate bounds, because they bound different things:

        * ``RetryAfterBackoff(budget_s=...)`` bounds attempts AND backoff
          together, so retries cannot multiply the timeout;
        * ``run_with_deadline`` bounds the whole retry loop, so even a transport
          that ignores its own timeout cannot overrun the caller;
        * ``split_timeouts`` shortens the *connect* phase inside the transport,
          which is the phase charged once per resolved address.
        """
        guard_url(url)
        now = self.clock.now()
        self.limiter.acquire(now)
        budget = max(0.1, float(self.timeout_s))
        started = time.monotonic()
        try:
            data = run_with_deadline(
                lambda: self.backoff.run(lambda: self._http_get(url, budget)),
                budget,
                label=self.name,
            )
        except MarketDataError as exc:
            attempts = self.backoff.last_attempts or 1
            if attempts > 1:
                self.retry_count += 1
                self.last_backoff_s = self.backoff.last_backoff_s
            self.last_error_label = classify_error(exc)
            log(
                "MARKET",
                "%s provider failure after %d attempt(s) [%s], backoff %.2fs, "
                "%.2fs of a %.2fs budget: %s",
                self.name, attempts, self.last_error_label,
                self.backoff.last_backoff_s,
                time.monotonic() - started, budget, exc,
            )
            raise
        self.max_call_s = max(self.max_call_s, time.monotonic() - started)
        if self.backoff.succeeded_after_retry:
            # PARTIAL failure: attempt 1 failed, a later attempt returned.
            self.retry_count += 1
            self.last_backoff_s = self.backoff.last_backoff_s
            self.last_error_label = "retried"
            log(
                "MARKET",
                "%s partial provider failure: %d attempt(s), last error [%s], "
                "recovered after %.2fs backoff",
                self.name, self.backoff.last_attempts,
                self.backoff.last_errors[-1] if self.backoff.last_errors else "unknown",
                self.backoff.last_backoff_s,
            )
        if isinstance(data, (bytes, bytearray, str)):
            try:
                data = json.loads(data)
            except Exception as exc:  # noqa: BLE001
                raise MarketDataError(
                    f"{self.name}: malformed response ({exc})"
                ) from exc
        if not isinstance(data, Mapping):
            raise MarketDataError(
                f"{self.name}: unexpected response shape "
                f"({type(data).__name__})"
            )
        return data

    # -- parsing (pure, unit tested) --------------------------------------

    def parse(self, payload: Mapping[str, Any], mint: str,
              received_at: float) -> Observation | None:
        pairs = payload.get("pairs") or []
        if not isinstance(pairs, list):
            return None
        candidates = [p for p in pairs if _is_usable_pair(p, mint)]
        if not candidates:
            self.last_reject_reason = self._why_rejected(pairs, mint)
            return None
        pair = max(candidates, key=lambda p: _f((p.get("liquidity") or {}).get("usd")) or 0.0)
        return self.parse_pair(pair, mint, received_at)

    def parse_pair(self, pair: Mapping[str, Any], mint: str,
                   received_at: float) -> Observation:
        liq = pair.get("liquidity") or {}
        vol = pair.get("volume") or {}
        chg = pair.get("priceChange") or {}
        txns = pair.get("txns") or {}
        m5 = txns.get("m5") or {}
        h1 = txns.get("h1") or {}
        base = pair.get("baseToken") or {}
        info = pair.get("info") or {}
        created_ms = _f(pair.get("pairCreatedAt"))
        pair_created_at = created_ms / 1000.0 if created_ms else None

        websites = [w.get("url") for w in (info.get("websites") or []) if w.get("url")]
        socials = {
            (s.get("type") or "link"): s.get("url")
            for s in (info.get("socials") or [])
            if s.get("url")
        }
        boosts = pair.get("boosts") or {}
        extra: dict[str, Any] = {
            "dex_id": pair.get("dexId"),
            "labels": pair.get("labels"),
            "boosts_active": len(boosts) if isinstance(boosts, (dict, list)) else None,
            "websites": websites[:3] or None,
            "twitter": socials.get("twitter"),
            "telegram": socials.get("telegram") or socials.get("telegram_channel"),
            "website": websites[0] if websites else None,
            "boost_amount_usd": _f(boosts.get("amount")) if isinstance(boosts, Mapping) else None,
        }
        return Observation(
            mint=mint,
            observed_at=received_at,
            received_at=received_at,
            price_usd=_f(pair.get("priceUsd")),
            price_native=_f(pair.get("priceNative")),
            market_cap_usd=_f(pair.get("marketCap")) or _f(pair.get("fdv")),
            fdv_usd=_f(pair.get("fdv")),
            liquidity_usd=_f(liq.get("usd")),
            volume_5m_usd=_f(vol.get("m5")),
            volume_1h_usd=_f(vol.get("h1")),
            volume_24h_usd=_f(vol.get("h24")),
            price_change_5m=_f(chg.get("m5")),
            price_change_1h=_f(chg.get("h1")),
            price_change_24h=_f(chg.get("h24")),
            buys_5m=_i(m5.get("buys")),
            sells_5m=_i(m5.get("sells")),
            txns_5m=(_i(m5.get("buys")) or 0) + (_i(m5.get("sells")) or 0),
            txns_1h=(_i(h1.get("buys")) or 0) + (_i(h1.get("sells")) or 0),
            pair_created_at=pair_created_at,
            first_valid_price=None,
            provider=self.name,
            venue=pair.get("dexId"),
            pair_address=pair.get("pairAddress"),
            extra={k: v for k, v in extra.items() if v is not None},
        )


def _is_usable_pair(pair: Mapping[str, Any], mint: str) -> bool:
    if pair.get("chainId") != CHAIN:
        return False
    base = pair.get("baseToken") or {}
    if (base.get("address") or "") != mint:
        return False
    base_symbol = (base.get("symbol") or "").upper()
    quote_symbol = ((pair.get("quoteToken") or {}).get("symbol") or "").upper()
    if not _f(pair.get("priceUsd")):
        return False
    # stable/stable and stable/wrapped venues are not a memecoin trade
    if base_symbol in _STABLE_SYMBOLS and (
        quote_symbol in _STABLE_SYMBOLS or quote_symbol in _WRAPPED_SYMBOLS
    ):
        return False
    return True


def _with_received(obs: Observation, received_at: float) -> Observation:
    from dataclasses import replace

    return replace(obs, received_at=received_at)


def _default_http_get(url: str, timeout_s: float,
                      connect_timeout_s: float | None = None) -> Mapping[str, Any]:
    """
    The real transport. One GET, one connection attempt per reachable address.

    ``connect`` is deliberately SHORT while the read/write phases keep the full
    budget. CPython's ``socket.create_connection`` walks ``getaddrinfo``
    results in order and charges the connect timeout to each one, so a host that
    lists unroutable IPv6 addresses ahead of the IPv4 ones costs
    ``n_dead * connect_timeout`` before a single byte moves. Measured here on
    2026-09-30: ``api.dexscreener.com`` and ``api.geckoterminal.com`` each list
    two dead IPv6 addresses ahead of two live IPv4 addresses, so a single
    ``httpx`` timeout of 8.0s became 16.0s per request and the candidate path
    ~35s. Splitting the timeouts removes the multiplication.
    """
    import httpx

    connect_s, phase_s = split_timeouts(timeout_s, connect_s=connect_timeout_s)
    headers = {
        "accept": "application/json",
        "user-agent": "AUTONOMOUS_PAPER_AGENT_V1/1.0 (paper research; read-only)",
    }
    timeout = httpx.Timeout(
        connect=connect_s, read=phase_s, write=phase_s, pool=connect_s
    )
    with httpx.Client(
        timeout=timeout,
        headers=headers,
        follow_redirects=True,
        # httpx's own transport retries are OFF. A retry there would be
        # invisible to RetryAfterBackoff, which is precisely the "hidden retry
        # inside a dependency" this class of bug is about.
        transport=httpx.HTTPTransport(retries=0),
    ) as client:
        resp = client.get(url)
    if resp.status_code == 429:
        retry_after = resp.headers.get("retry-after")
        raise MarketDataError(
            "dexscreener: 429 rate limited",
            rate_limited=True,
            retry_after_s=_f(retry_after),
            status=429,
        )
    if resp.status_code >= 400:
        raise MarketDataError(
            f"dexscreener: HTTP {resp.status_code}", status=resp.status_code
        )
    try:
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        raise MarketDataError(f"dexscreener: malformed JSON ({exc})") from exc
