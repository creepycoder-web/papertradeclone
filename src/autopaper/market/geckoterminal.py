"""
GeckoTerminal provider (fallback + first-valid-price oracle).

Read-only. Two roles:

1. Fallback quote when DexScreener has no pair, is rate limited, or its
   liquidity is implausibly low.
2. Source of the *first valid observed price* for a token via 1-minute OHLCV
   candles, so ``extension_from_first_valid_price`` has a real anchor instead
   of a guess.

Every value is taken verbatim; missing values are ``None``.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from ..clock import Clock, SystemClock
from ..logging_setup import log
from ..models import Observation, QuoteResult
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

BASE = "https://api.geckoterminal.com/api/v2"
NETWORK = "solana"


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


def _ts(value: Any) -> float | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc).timestamp()
    except ValueError:
        return None


class GeckoTerminalProvider:
    name = "geckoterminal"

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        max_rpm: int = 45,
        timeout_s: float = 8.0,
        connect_timeout_s: float = 2.0,
        cache_ttl_s: float = 5.0,
        max_retries: int = 1,
        http_get: Callable[[str, float], Mapping[str, Any]] | None = None,
    ) -> None:
        self.clock = clock or SystemClock()
        # GeckoTerminal's public tier is ~30 calls/min; stay well under.
        self.limiter = RateLimiter(max_rpm, name=self.name)
        self.backoff = RetryAfterBackoff(max_attempts=max_retries,
                                         budget_s=timeout_s)
        self.cache = TTLCache()
        self.cache_ttl_s = cache_ttl_s
        self.timeout_s = timeout_s
        self.connect_timeout_s = connect_timeout_s
        # Two-argument seam, as on the primary provider. See DexScreenerProvider.
        self._http_get = http_get or (
            lambda url, timeout_s: _default_http_get(
                url, timeout_s, self.connect_timeout_s
            )
        )
        self.retry_count = 0
        self.last_backoff_s = 0.0
        self.last_error_label: str | None = None
        self.max_call_s = 0.0

    # -- quote -------------------------------------------------------------

    def quote(self, mint: str) -> QuoteResult:
        started = self.clock.monotonic()
        now = self.clock.now()
        cached = self.cache.get(mint, self.cache_ttl_s, now)
        if cached is not None:
            obs, _ = cached
            return QuoteResult(observation=obs, provider=self.name, from_cache=True)

        try:
            payload = self._fetch(f"{BASE}/networks/{NETWORK}/tokens/{mint}/pools?page=1")
        except MarketDataError as exc:
            self.limiter.note_retry_after(self.clock.now(), exc.retry_after_s)
            return QuoteResult(
                observation=None,
                error=f"{self.name}: {exc}",
                provider=self.name,
                latency_ms=(self.clock.monotonic() - started) * 1000.0,
                rate_limited=exc.rate_limited,
                retry_after_s=exc.retry_after_s,
            )

        obs = self.parse(payload, mint, self.clock.now())
        if obs is not None:
            self.cache.put(mint, obs, self.clock.now())
        return QuoteResult(
            observation=obs,
            error=None if obs else f"{self.name}: no solana pool for {mint}",
            provider=self.name,
            latency_ms=(self.clock.monotonic() - started) * 1000.0,
        )

    async def quote_async(self, mint: str) -> QuoteResult:
        """Non-blocking form of :meth:`quote`. See DexScreenerProvider.

        The synchronous httpx transport and its retry loop are moved onto a
        worker thread so they cannot starve the asyncio event loop.
        """
        return await asyncio.to_thread(self.quote, mint)

    def retry_stats(self) -> dict[str, Any]:
        return {
            "retry_count": self.retry_count,
            "last_backoff_s": round(self.last_backoff_s, 4),
            "last_error_label": self.last_error_label,
        }

    # -- first valid price -------------------------------------------------

    def first_valid_price(self, mint: str, *, lookback: int = 60) -> tuple[float | None, float | None]:
        """
        Return ``(price, candle_open_time)`` for the oldest usable 1m candle.

        This is the anchor for ``extension_from_first_valid_price``. It reads
        historical candles on purpose, and is only ever used as a *context*
        feature at decision time and for post-hoc attribution of rejected
        candidates - never as the entry price.
        """
        try:
            pools = self._fetch(
                f"{BASE}/networks/{NETWORK}/tokens/{mint}/pools?page=1"
            )
            address = _top_pool_address(pools, mint)
            if not address:
                return None, None
            ohlcv = self._fetch(
                f"{BASE}/networks/{NETWORK}/pools/{address}/ohlcv/minute"
                f"?aggregate=1&limit={lookback}&currency=usd"
            )
        except MarketDataError:
            return None, None
        return self.parse_ohlcv(ohlcv)

    # -- parsing (pure) ----------------------------------------------------

    def parse(self, payload: Mapping[str, Any], mint: str,
              received_at: float) -> Observation | None:
        data = payload.get("data") or []
        if not isinstance(data, list) or not data:
            return None
        rows: list[Mapping[str, Any]] = []
        for item in data:
            if not isinstance(item, Mapping):
                continue
            attrs = item.get("attributes") or {}
            rel = item.get("relationships") or {}
            base_rel = ((rel.get("base_token") or {}).get("data") or {})
            if (base_rel.get("id") or "") == mint:
                rows.append(attrs)
        if not rows:
            return None
        attrs = max(rows, key=lambda a: _f((a.get("reserve_in_usd") or 0.0)) or 0.0)
        return self.parse_pool(attrs, mint, received_at)

    def parse_pool(self, attrs: Mapping[str, Any], mint: str,
                   received_at: float) -> Observation:
        vol = attrs.get("volume_usd") or {}
        chg = attrs.get("price_change_percentage") or {}
        txns = attrs.get("transactions") or {}
        m5 = txns.get("m5") or {}
        reserve = _f(attrs.get("reserve_in_usd"))
        vol24 = _f(vol.get("h24"))
        extra = {
            "pool_name": attrs.get("name"),
            "pool_created_at": attrs.get("pool_created_at"),
            "gt_price_native": _f(attrs.get("base_token_price_native_currency")),
        }
        return Observation(
            mint=mint,
            observed_at=received_at,
            received_at=received_at,
            price_usd=_f(attrs.get("base_token_price_usd")),
            price_native=_f(attrs.get("base_token_price_native_currency")),
            market_cap_usd=_f(attrs.get("market_cap_usd")) or _f(attrs.get("fdv_usd")),
            fdv_usd=_f(attrs.get("fdv_usd")),
            liquidity_usd=reserve,
            volume_5m_usd=_f(vol.get("m5")),
            volume_1h_usd=_f(vol.get("h1")),
            volume_24h_usd=vol24,
            price_change_5m=_f(chg.get("m5")),
            price_change_1h=_f(chg.get("h1")),
            price_change_24h=_f(chg.get("h24")),
            buys_5m=_i(m5.get("buys")),
            sells_5m=_i(m5.get("sells")),
            txns_5m=(_i(m5.get("buys")) or 0) + (_i(m5.get("sells")) or 0),
            txns_1h=(_i((txns.get("h1") or {}).get("buys")) or 0)
            + (_i((txns.get("h1") or {}).get("sells")) or 0),
            pair_created_at=_ts(attrs.get("pool_created_at")),
            first_valid_price=None,
            provider=self.name,
            venue="geckoterminal",
            pair_address=attrs.get("address"),
            extra={k: v for k, v in extra.items() if v is not None},
        )

    def parse_ohlcv(self, payload: Mapping[str, Any]) -> tuple[float | None, float | None]:
        rows = ((payload.get("data") or {}).get("attributes") or {}).get("ohlcv_list") or []
        best: tuple[float, float] | None = None
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 5:
                continue
            open_time = _f(row[0])
            close = _f(row[4])
            if open_time is None or close is None or close <= 0:
                continue
            if best is None or open_time < best[0]:
                best = (open_time, close)
        if best is None:
            return None, None
        return best[1], best[0]

    def _fetch(self, url: str) -> Mapping[str, Any]:
        """
        See ``DexScreenerProvider._fetch`` for the ``http_get`` contract and for
        why the wall-clock budget, the retry budget and the connect split are
        three separate bounds.
        """
        guard_url(url)
        self.limiter.acquire(self.clock.now())
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
                raise MarketDataError(f"{self.name}: malformed response ({exc})") from exc
        if not isinstance(data, Mapping):
            raise MarketDataError(
                f"{self.name}: unexpected response shape ({type(data).__name__})"
            )
        return data


def _top_pool_address(payload: Mapping[str, Any], mint: str) -> str | None:
    data = payload.get("data") or []
    best: tuple[float, str] | None = None
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, Mapping):
            continue
        rel = item.get("relationships") or {}
        base = ((rel.get("base_token") or {}).get("data") or {})
        if (base.get("id") or "") != mint:
            continue
        attrs = item.get("attributes") or {}
        reserve = _f(attrs.get("reserve_in_usd")) or 0.0
        addr = attrs.get("address")
        if addr and (best is None or reserve > best[0]):
            best = (reserve, addr)
    return best[1] if best else None


def _default_http_get(url: str, timeout_s: float,
                      connect_timeout_s: float | None = None) -> Mapping[str, Any]:
    """
    See ``DexScreenerProvider._default_http_get``: the short connect timeout is
    what stops an unroutable IPv6 address from costing the FULL budget.

    This provider matters twice over, because :meth:`first_valid_price` performs
    TWO sequential fetches (pools, then candles) from the same method.
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
        transport=httpx.HTTPTransport(retries=0),
    ) as client:
        resp = client.get(url)
    if resp.status_code == 429:
        raise MarketDataError(
            "geckoterminal: 429 rate limited",
            rate_limited=True,
            retry_after_s=_f(resp.headers.get("retry-after")),
            status=429,
        )
    if resp.status_code >= 400:
        raise MarketDataError(
            f"geckoterminal: HTTP {resp.status_code}", status=resp.status_code
        )
    try:
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        raise MarketDataError(f"geckoterminal: malformed JSON ({exc})") from exc
