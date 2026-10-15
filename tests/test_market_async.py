"""
DexScreener / GeckoTerminal must not be able to block the asyncio event loop.

THE DEFECT
----------
The provider transport is synchronous (httpx.Client plus a retry loop):

    data = self.backoff.run(lambda: self._http_get(url, self.timeout_s))

The engine awaited nothing, so that loop ran ON the event loop. With the
configured 8s HTTP timeout and 2 retries one call can occupy the loop for ~16s,
starving the monitor, the Telegram read stream and every other task - which is
where the observed ~34s analysis tail (and the ~940s outlier) came from.

THE FIX
-------
The same synchronous implementation now runs on a worker thread via
``asyncio.to_thread``, exposed as ``quote_async`` on the providers and on
``MarketDataService``. Semantics are unchanged; only the thread changes.

These tests prove the loop keeps running during a slow fetch, and that partial
provider failures are now visible instead of silent.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time

import pytest

from autopaper.clock import FakeClock
from autopaper.market import (
    DexScreenerProvider,
    MarketDataError,
    MarketDataService,
)
from autopaper.market.ratelimit import RetryAfterBackoff, classify_error
from autopaper.config import MarketConfig

from .conftest import MINT

DS_PAIR = {
    "chainId": "solana",
    "dexId": "raydium",
    "pairAddress": "PAIR1",
    "baseToken": {"address": MINT, "symbol": "TST"},
    "quoteToken": {"address": "QUOTE", "symbol": "SOL"},
    "priceNative": "0.00000015",
    "priceUsd": "0.00015",
    "txns": {"m5": {"buys": 120, "sells": 80}, "h1": {"buys": 9, "sells": 5}},
    "volume": {"m5": 25000.0, "h1": 120000.0, "h24": 900000.0},
    "priceChange": {"m5": 12.5, "h1": 40.0, "h24": 300.0},
    "liquidity": {"usd": 55000.0},
    "fdv": 2_000_000.0,
    "marketCap": 1_500_000.0,
    "pairCreatedAt": 1_699_999_400_000,
    "info": {},
}


def _provider(http_get, **kw) -> DexScreenerProvider:
    p = DexScreenerProvider(
        clock=FakeClock(1_700_000_000.0), http_get=http_get, cache_ttl_s=0.0, **kw
    )
    # No real sleeping in retry tests.
    p.backoff.base_s = 0.0
    p.backoff.max_s = 0.0
    return p


# ---------------------------------------------------------------------------
# the loop is not blocked
# ---------------------------------------------------------------------------


def test_a_slow_fetch_does_not_block_the_event_loop():
    """
    The monitor/ticker must keep advancing WHILE a provider fetch is in flight.

    Before the fix this counter would stay at ~0 for the whole fetch.
    """
    worker_threads: list[str] = []

    def slow_get(url: str, timeout_s: float):
        worker_threads.append(threading.current_thread().name)
        time.sleep(0.30)
        return {"pairs": [dict(DS_PAIR)]}

    p = _provider(slow_get)
    ticks = {"n": 0}

    async def main():
        stop = asyncio.Event()

        async def ticker():
            while not stop.is_set():
                ticks["n"] += 1
                await asyncio.sleep(0.01)

        t = asyncio.ensure_future(ticker())
        try:
            return await p.quote_async(MINT)
        finally:
            stop.set()
            await t

    q = asyncio.run(main())
    assert q.observation is not None
    # ~30 ticks in 0.3s if the loop is free; a blocked loop yields ~0.
    assert ticks["n"] > 5, f"the event loop was starved (ticks={ticks['n']})"
    # and the work really did move to another thread
    assert worker_threads and worker_threads[0] != threading.main_thread().name


def test_the_sync_quote_is_still_available_and_unchanged():
    """The synchronous entry point is preserved for tests and offline callers."""
    p = _provider(lambda url, t: {"pairs": [dict(DS_PAIR)]})
    q = p.quote(MINT)
    assert q.observation is not None
    assert q.observation.price_usd == pytest.approx(0.00015)


def test_service_quote_async_returns_the_same_result_as_quote():
    svc = MarketDataService(
        MarketConfig(),
        clock=FakeClock(1_700_000_000.0),
        primary=_provider(lambda url, t: {"pairs": [dict(DS_PAIR)]}),
        fallback=_provider(lambda url, t: {"pairs": [dict(DS_PAIR)]}),
    )
    sync_q = svc.quote(MINT)
    async_q = asyncio.run(svc.quote_async(MINT))
    assert async_q.observation is not None
    assert async_q.observation.price_usd == pytest.approx(
        sync_q.observation.price_usd
    )


# ---------------------------------------------------------------------------
# partial provider failures are now visible
# ---------------------------------------------------------------------------


def test_attempt_1_fails_attempt_2_succeeds_is_recorded():
    """The partial failure that used to leave no trace anywhere."""
    calls = {"n": 0}

    def flaky(url: str, timeout_s: float):
        calls["n"] += 1
        if calls["n"] == 1:
            raise MarketDataError("dexscreener: read timed out")
        return {"pairs": [dict(DS_PAIR)]}

    p = _provider(flaky, max_retries=2)
    q = p.quote(MINT)
    assert q.observation is not None, "the retry should have recovered"
    assert p.retry_count == 1
    assert p.backoff.last_attempts == 2
    assert p.backoff.succeeded_after_retry is True
    assert "timed out" in (p.backoff.last_errors[-1] if p.backoff.last_errors else "")


def test_repeated_timeout_is_recorded_and_classified():
    def always_timeout(url: str, timeout_s: float):
        raise MarketDataError("dexscreener: read timed out")

    p = _provider(always_timeout, max_retries=2)
    q = p.quote(MINT)
    assert q.observation is None
    assert q.error
    assert p.retry_count >= 1
    assert p.last_error_label == "timeout"
    assert p.backoff.last_attempts == 3        # 1 try + max_retries


def test_connection_reset_is_classified_distinctly():
    def reset(url: str, timeout_s: float):
        raise MarketDataError("dexscreener: connection reset by peer")

    p = _provider(reset, max_retries=1)
    p.quote(MINT)
    assert p.last_error_label == "connection_reset"


def test_backoff_duration_is_recorded():
    def flaky(url: str, timeout_s: float):
        if not hasattr(flaky, "done"):
            flaky.done = True  # type: ignore[attr-defined]
            raise MarketDataError("boom")
        return {"pairs": [dict(DS_PAIR)]}

    p = _provider(flaky, max_retries=2)
    p.backoff.base_s = 0.5
    p.backoff.max_s = 0.5
    p.backoff.rng = random.Random(1)      # deterministic, non-zero delay
    p.quote(MINT)
    assert p.last_backoff_s > 0.0
    assert p.retry_stats()["last_backoff_s"] == round(p.last_backoff_s, 4)
    assert p.retry_stats()["retry_count"] == 1


def test_classify_error_labels_the_common_shapes():
    assert classify_error(MarketDataError("timed out")) == "timeout"
    assert classify_error(MarketDataError("connection reset")) == "connection_reset"
    assert classify_error(MarketDataError("connection refused")) == "connection_refused"
    assert classify_error(ConnectionResetError()) == "connection_reset"
    assert classify_error(MarketDataError("weird", status=418)) == "MarketDataError"


def test_retry_backoff_preserves_max_attempts_semantics():
    """The retry count is unchanged by the visibility additions."""
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        raise MarketDataError("boom")

    b = RetryAfterBackoff(max_attempts=2)
    b.base_s = 0.0
    b.max_s = 0.0
    with pytest.raises(MarketDataError):
        b.run(fn, sleep=lambda s: None)
    assert attempts["n"] == 3        # 1 try + 2 retries, exactly as before
    assert b.last_attempts == 3
    assert b.last_ok is False
