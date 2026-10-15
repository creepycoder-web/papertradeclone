"""
Market data: rate limiting, 429 handling, freshness, and no-fabrication.

Every provider is injected with a fake HTTP function, so these tests never touch
the network.
"""

from __future__ import annotations

import json

import pytest

from autopaper.clock import FakeClock
from autopaper.config import MarketConfig
from autopaper.market import (
    DexScreenerProvider,
    GeckoTerminalProvider,
    MarketDataError,
    MarketDataService,
    RateLimiter,
    TTLCache,
)
from autopaper.models import DataState

from .conftest import MINT, obs

T0 = 1_700_000_000.0

DS_PAIR = {
    "chainId": "solana",
    "dexId": "raydium",
    "pairAddress": "PAIR1",
    "baseToken": {"address": MINT, "name": "Test", "symbol": "TST"},
    "quoteToken": {"address": "QUOTE", "name": "SOL", "symbol": "SOL"},
    "priceNative": "0.00000015",
    "priceUsd": "0.00015",
    "txns": {"m5": {"buys": 120, "sells": 80}, "h1": {"buys": 900, "sells": 500}},
    "volume": {"m5": 25000.0, "h1": 120000.0, "h24": 900000.0},
    "priceChange": {"m5": 12.5, "h1": 40.0, "h24": 300.0},
    "liquidity": {"usd": 55000.0, "base": 1, "quote": 2},
    "fdv": 2_000_000.0,
    "marketCap": 1_500_000.0,
    "pairCreatedAt": 1_699_999_400_000,
    "info": {
        "websites": [{"label": "site", "url": "https://x.example"}],
        "socials": [{"type": "twitter", "url": "https://x.com/x"}],
    },
    "boosts": {"active": 1},
}

GT_POOL = {
    "address": "POOL1",
    "attributes": {
        "base_token_price_usd": "0.000151",
        "base_token_price_native_currency": "0.000000151",
        "reserve_in_usd": "56000.0",
        "fdv_usd": "2000000.0",
        "market_cap_usd": "1500000.0",
        "pool_created_at": "2023-11-14T14:36:40Z",
        "name": "TST/SOL",
        "volume_usd": {"m5": "25000.0", "h1": "120000.0", "h24": "900000.0"},
        "price_change_percentage": {"m5": "12.5", "h1": "40.0", "h24": "300.0"},
        "transactions": {"m5": {"buys": 120, "sells": 80},
                         "h1": {"buys": 900, "sells": 500}},
    },
    "relationships": {"base_token": {"data": {"id": MINT}}},
}


def ds_payload(**over):
    p = dict(DS_PAIR)
    p.update(over)
    return {"pairs": [p]}


# ---------------------------------------------------------------------------
# rate limiting
# ---------------------------------------------------------------------------


def test_rate_limiter_blocks_at_the_cap():
    rl = RateLimiter(3, name="test")
    now = T0
    for _ in range(3):
        rl.acquire(now)
        now += 0.1
    with pytest.raises(MarketDataError) as exc:
        rl.acquire(now)
    assert exc.value.rate_limited


def test_rate_limiter_window_slides():
    rl = RateLimiter(2, name="test")
    rl.acquire(T0)
    rl.acquire(T0 + 1)
    with pytest.raises(MarketDataError):
        rl.acquire(T0 + 2)
    rl.acquire(T0 + 61)  # first call has aged out of the 60s window
    rl.acquire(T0 + 61.1)


def test_retry_after_is_obeyed():
    rl = RateLimiter(100, name="test")
    rl.note_retry_after(T0, 30.0)
    allowed, wait = rl.allow(T0 + 1)
    assert not allowed
    assert wait == pytest.approx(29.0)
    assert rl.allow(T0 + 31)[0] is True


def test_retry_after_without_a_header_backs_off_a_full_window():
    rl = RateLimiter(100, name="test")
    rl.note_retry_after(T0, None)
    allowed, wait = rl.allow(T0)
    assert not allowed
    assert wait == pytest.approx(60.0)


def test_ttl_cache_expires():
    c = TTLCache()
    c.put("k", "v", T0)
    assert c.get("k", 2.0, T0 + 1.0)[0] == "v"
    assert c.get("k", 2.0, T0 + 3.0) is None
    assert c.get_any("k") is not None


# ---------------------------------------------------------------------------
# dexscreener parsing
# ---------------------------------------------------------------------------


def test_dexscreener_parses_a_full_pair():
    p = DexScreenerProvider(clock=FakeClock(T0))
    o = p.parse(ds_payload(), MINT, T0)
    assert o is not None
    assert o.price_usd == pytest.approx(0.00015)
    assert o.market_cap_usd == pytest.approx(1_500_000.0)
    assert o.liquidity_usd == pytest.approx(55_000.0)
    assert o.buys_5m == 120
    assert o.sells_5m == 80
    assert o.txns_5m == 200
    assert o.change_5m if hasattr(o, "change_5m") else o.price_change_5m == pytest.approx(12.5)
    assert o.token_age_s == pytest.approx(600.0, abs=1.0)
    assert o.extra["twitter"] == "https://x.com/x"
    assert o.venue == "raydium"


def test_dexscreener_missing_fields_stay_none():
    p = DexScreenerProvider(clock=FakeClock(T0))
    bare = {"chainId": "solana", "dexId": "raydium",
            "baseToken": {"address": MINT}, "quoteToken": {"symbol": "SOL"},
            "priceUsd": "0.001"}
    o = p.parse({"pairs": [bare]}, MINT, T0)
    assert o.price_usd == pytest.approx(0.001)
    for f in ("market_cap_usd", "liquidity_usd", "volume_5m_usd", "buys_5m",
              "pair_created_at", "price_change_5m"):
        assert getattr(o, f) is None, f


def test_dexscreener_ignores_wrong_chain_and_wrong_base():
    p = DexScreenerProvider(clock=FakeClock(T0))
    assert p.parse(ds_payload(chainId="ethereum"), MINT, T0) is None
    wrong = DS_PAIR | {"baseToken": {"address": "OTHER"}}
    assert p.parse({"pairs": [wrong]}, MINT, T0) is None


def test_dexscreener_keeps_a_memecoin_usdc_pool():
    """
    TOKEN/USDC is the deepest pool a new launch has, so a stablecoin QUOTE must
    not be filtered out - only stable/STABLE and stable/wrapped venues are.
    An earlier version excluded stablecoin quote assets outright and threw away
    the best venue for every realistic candidate.
    """
    p = DexScreenerProvider(clock=FakeClock(T0))
    token_usdc = {
        "chainId": "solana", "dexId": "raydium", "pairAddress": "TOKEN_USDC",
        "baseToken": {"address": MINT, "symbol": "TST"},
        "quoteToken": {"address": "U1", "symbol": "USDC"},
        "priceUsd": "0.0002", "liquidity": {"usd": 400_000.0},
    }
    token_usdt = {
        "chainId": "solana", "dexId": "raydium", "pairAddress": "TOKEN_USDT",
        "baseToken": {"address": MINT, "symbol": "TST"},
        "quoteToken": {"address": "U2", "symbol": "USDT"},
        "priceUsd": "0.000199", "liquidity": {"usd": 300_000.0},
    }
    o = p.parse({"pairs": [token_usdt, token_usdc]}, MINT, T0)
    assert o is not None, "a memecoin quoted against USDC must be usable"
    assert o.pair_address == "TOKEN_USDC", "the deepest such pool should win"

    # but a stable/stable venue is not a memecoin trade and is refused
    stable_stable = {
        "chainId": "solana", "dexId": "raydium", "pairAddress": "USDC_USDT",
        "baseToken": {"address": "USDC_ADDR", "symbol": "USDC"},
        "quoteToken": {"address": "U2", "symbol": "USDT"},
        "priceUsd": "1.0", "liquidity": {"usd": 9_000_000.0},
    }
    assert p.parse({"pairs": [stable_stable]}, "USDC_ADDR", T0) is None
    stable_wrapped = dict(stable_stable, quoteToken={"address": "W", "symbol": "WBTC"})
    assert p.parse({"pairs": [stable_wrapped]}, "USDC_ADDR", T0) is None


def test_dexscreener_explains_why_a_mint_has_no_pool():
    p = DexScreenerProvider(clock=FakeClock(T0))
    p.parse({"pairs": []}, MINT, T0)
    assert "no solana pairs" in (p.last_reject_reason or "")
    quote_only = DS_PAIR | {"baseToken": {"address": "OTHER", "symbol": "SOL"}}
    p.parse({"pairs": [quote_only]}, MINT, T0)
    assert "QUOTE token" in (p.last_reject_reason or "")


def test_dexscreener_picks_the_deepest_pair():
    p = DexScreenerProvider(clock=FakeClock(T0))
    shallow = DS_PAIR | {"pairAddress": "SHALLOW", "liquidity": {"usd": 100.0}}
    deep = DS_PAIR | {"pairAddress": "DEEP", "liquidity": {"usd": 900_000.0}}
    o = p.parse({"pairs": [shallow, deep]}, MINT, T0)
    assert o.pair_address == "DEEP"


def test_dexscreener_rejects_zero_price():
    p = DexScreenerProvider(clock=FakeClock(T0))
    assert p.parse(ds_payload(priceUsd="0"), MINT, T0) is None
    assert p.parse({"pairs": []}, MINT, T0) is None


# ---------------------------------------------------------------------------
# 429 / failure handling
# ---------------------------------------------------------------------------


def test_429_uses_retry_after_and_does_not_retry():
    calls = {"n": 0}

    def fake_429(url, timeout):
        calls["n"] += 1
        raise MarketDataError("429", rate_limited=True, retry_after_s=45.0, status=429)

    clock = FakeClock(T0)
    p = DexScreenerProvider(clock=clock, http_get=fake_429)
    r = p.quote(MINT)
    assert r.observation is None
    assert r.rate_limited is True
    assert r.retry_after_s == pytest.approx(45.0)
    assert calls["n"] == 1, "a 429 must not be retried immediately"
    assert p.limiter.blocked_until >= T0 + 45.0


def test_failure_serves_the_last_good_observation_as_degraded():
    good = {"n": 0}

    def flaky(url, timeout):
        good["n"] += 1
        if good["n"] == 1:
            return ds_payload()          # the http_get contract: decoded mapping
        raise MarketDataError("500", status=500)

    clock = FakeClock(T0)
    p = DexScreenerProvider(clock=clock, http_get=flaky, cache_ttl_s=0.0,
                            max_retries=0)
    first = p.quote(MINT)
    assert first.observation is not None
    clock.advance(5)
    second = p.quote(MINT)
    # stale data is better than a fabricated price, but it is reported as an error
    assert second.observation is not None
    assert second.observation.price_usd == pytest.approx(0.00015)
    assert second.error is not None
    assert second.from_cache is True


def test_never_fabricates_a_price():
    p = DexScreenerProvider(
        clock=FakeClock(T0),
        http_get=lambda u, t: (_ for _ in ()).throw(MarketDataError("down")),
        max_retries=0,
    )
    r = p.quote(MINT)
    assert r.observation is None
    assert r.error


# ---------------------------------------------------------------------------
# geckoterminal
# ---------------------------------------------------------------------------


def test_geckoterminal_parses_a_pool():
    g = GeckoTerminalProvider(clock=FakeClock(T0))
    o = g.parse({"data": [GT_POOL]}, MINT, T0)
    assert o.price_usd == pytest.approx(0.000151)
    assert o.liquidity_usd == pytest.approx(56_000.0)
    assert o.buys_5m == 120
    assert o.pair_created_at is not None


def test_geckoterminal_ohlcv_gives_the_oldest_usable_candle():
    g = GeckoTerminalProvider(clock=FakeClock(T0))
    payload = {
        "data": {"attributes": {"ohlcv_list": [
            [T0, 1.0, 1.1, 0.9, 1.05, 1000],
            [T0 - 60, 0.5, 0.6, 0.4, 0.55, 900],
            [T0 - 120, 0.2, 0.3, 0.1, 0.25, 800],
        ]}}
    }
    price, opened = g.parse_ohlcv(payload)
    assert price == pytest.approx(0.25)
    assert opened == pytest.approx(T0 - 120)
    # unusable rows are skipped, not coerced
    assert g.parse_ohlcv({"data": {"attributes": {"ohlcv_list": [[1, 1, 1, 1, 0, 0]]}}}) == (None, None)
    assert g.parse_ohlcv({}) == (None, None)


# ---------------------------------------------------------------------------
# service-level behaviour
# ---------------------------------------------------------------------------


class FakeProvider:
    name = "fake"

    def __init__(self, clock, price=1.0e-4, liquidity=60_000.0, fail=False,
                 rate_limited=False):
        self.clock = clock
        self.price = price
        self.liquidity = liquidity
        self.fail = fail
        self.rate_limited = rate_limited
        self.calls = 0
        from autopaper.market.ratelimit import RateLimiter

        self.limiter = RateLimiter(600, name=self.name)

    def quote(self, mint, **kw):
        from autopaper.models import QuoteResult

        self.calls += 1
        if self.fail:
            return QuoteResult(observation=None, error="fake: down",
                               provider=self.name, rate_limited=self.rate_limited)
        return QuoteResult(
            observation=obs(self.clock.now(), self.price, mint=mint,
                            liquidity_usd=self.liquidity),
            provider=self.name,
        )


def test_service_falls_back_to_the_second_provider():
    clock = FakeClock(T0)
    primary = FakeProvider(clock, fail=True)
    fallback = FakeProvider(clock, price=1.05e-4)
    svc = MarketDataService(MarketConfig(), clock=clock, primary=primary,
                            fallback=fallback)
    r = svc.quote(MINT)
    assert r.observation is not None
    assert r.provider == "fake+fake"
    assert fallback.calls == 1


def test_service_fills_gaps_from_the_fallback():
    """
    The fallback is consulted when the primary has no usable venue - a failed
    or rate-limited fetch, or a pool reporting zero liquidity. It is NOT
    consulted merely because a real pool is thin: thin liquidity is a fact to
    price in, not a reason to go looking for a friendlier venue, and silently
    substituting another venue's price would make the ledger's provenance
    ambiguous.
    """
    clock = FakeClock(T0)
    primary = FakeProvider(clock, liquidity=0.0)   # dead pool
    fallback = FakeProvider(clock, price=1.05e-4, liquidity=90_000.0)
    svc = MarketDataService(MarketConfig(), clock=clock, primary=primary,
                            fallback=fallback)
    o = svc.quote(MINT).observation
    # the primary's pool is not a tradeable venue, so the fallback's price and
    # size are believed rather than gap-filled around a dead pool
    assert o.price_usd == pytest.approx(1.05e-4)
    assert o.liquidity_usd == pytest.approx(90_000.0)


def test_a_real_but_thin_pool_is_not_second_guessed():
    """A $3.5k pool is a cost to model, not a reason to look elsewhere."""
    clock = FakeClock(T0)
    primary = FakeProvider(clock, price=1.0e-4, liquidity=3_500.0)
    fallback = FakeProvider(clock, price=9.9e-4, liquidity=900_000.0)
    svc = MarketDataService(MarketConfig(), clock=clock, primary=primary,
                            fallback=fallback)
    o = svc.quote(MINT).observation
    assert fallback.calls == 0, "the fallback must not be consulted for a thin pool"
    assert o.price_usd == pytest.approx(1.0e-4)
    assert o.liquidity_usd == pytest.approx(3_500.0)


def test_service_reports_missing_without_inventing():
    clock = FakeClock(T0)
    svc = MarketDataService(MarketConfig(), clock=clock,
                            primary=FakeProvider(clock, fail=True),
                            fallback=FakeProvider(clock, fail=True))
    r = svc.quote(MINT)
    assert r.observation is None
    assert svc.state(MINT) is DataState.MISSING
    assert svc.mark_price(MINT) == (None, DataState.MISSING)
    assert svc.error_count >= 1


def test_history_accumulates_and_marks_freshness():
    clock = FakeClock(T0)
    svc = MarketDataService(MarketConfig(), clock=clock,
                            primary=FakeProvider(clock), fallback=FakeProvider(clock))
    for _ in range(3):
        svc.quote(MINT)
        clock.advance(4)
    hist = svc.history(MINT)
    assert len(hist) == 3
    assert svc.state(MINT) is DataState.LIVE
    clock.advance(20)
    assert svc.state(MINT) is DataState.STALE
    price, state = svc.mark_price(MINT)
    assert price is not None and state is DataState.STALE
    clock.advance(60)
    assert svc.state(MINT) is DataState.UNUSABLE


def test_first_valid_price_anchor_is_kept():
    clock = FakeClock(T0)
    p = FakeProvider(clock, price=1.0e-4)
    svc = MarketDataService(MarketConfig(), clock=clock, primary=p, fallback=p)
    svc.quote(MINT)
    assert svc.first_valid_price(MINT) == pytest.approx(1.0e-4)
    clock.advance(10)
    p.price = 3.0e-4
    svc.quote(MINT)
    # the anchor must not move as the price runs
    assert svc.first_valid_price(MINT) == pytest.approx(1.0e-4)


def test_alert_price_can_seed_the_anchor():
    clock = FakeClock(T0)
    svc = MarketDataService(MarketConfig(), clock=clock,
                            primary=FakeProvider(clock), fallback=FakeProvider(clock))
    svc.register_known_price(MINT, 2.5e-5)
    assert svc.first_valid_price(MINT) == pytest.approx(2.5e-5)
    svc.register_known_price(MINT, 9.9e-5)   # a later price must not overwrite
    assert svc.first_valid_price(MINT) == pytest.approx(2.5e-5)


def test_observations_record_observed_at_and_received_at():
    clock = FakeClock(T0)
    svc = MarketDataService(MarketConfig(), clock=clock,
                            primary=FakeProvider(clock), fallback=FakeProvider(clock))
    o = svc.quote(MINT).observation
    assert o.observed_at is not None
    assert o.received_at >= o.observed_at
    assert o.freshness(clock.now(), 15_000, 45_000) is DataState.LIVE
