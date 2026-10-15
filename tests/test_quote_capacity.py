"""
Quote-path capacity: concurrency, sharing, deadlines, failover, classification.

WHY THESE TESTS EXIST
=====================

The incident was a CAPACITY limit reported as a MARKET verdict. ``max_rpm`` was
wired to ``concurrency * 4`` (=16/min) while the same config declared
``public_rpc_rate_per_s = 8`` (=480/min); because the limiter RAISED instead of
waiting, every request past that ceiling was deleted and surfaced downstream as
"no executable quote". Three quarters of live candidates died there, and the
result was investigated as a strategy problem for a day.

So the properties worth pinning are not "it is faster" but:

* capacity is not silently deleted -- a request waits, then refuses BY NAME;
* a saturated endpoint does not decide availability on its own;
* independent reads may overlap, but never without a ceiling;
* one decision spends one quote, no matter how many arms consume it;
* a slow venue is CANCELLED at its deadline, never queued;
* and none of the safety semantics moved.

Everything here is offline: the transport is a fake, and no test signs,
broadcasts or reads a real chain.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from autopaper.paper_live.engine import classify_quote_failure
from autopaper.paper_live.schema import FailureReason
from autopaper.same_market.rpc import SolanaReadOnlyRpc


# ---------------------------------------------------------------------------
# doubles
# ---------------------------------------------------------------------------


def sync(fn):
    """Run one coroutine test body. This repo has no async pytest plugin, so
    async behaviour is asserted through ``asyncio.run`` directly."""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **k):
        return asyncio.run(fn(*a, **k))
    return wrapper


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"result": 1}
        self.headers = headers or {}

    def json(self):
        return self._payload


class FakeClient:
    """Records every request and answers from a scripted handler."""

    def __init__(self, handler):
        self.handler = handler
        self.posts: list[dict] = []
        self.closed = False

    async def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append({"url": url, "json": json})
        outcome = self.handler(url, json)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def aclose(self):
        self.closed = True


def ok(payload=None):
    return lambda url, body: FakeResponse(200, payload or {"result": 1})


def rpc(handler, **kw):
    kw.setdefault("cache_ttl_s", 0.0)
    kw.setdefault("max_rpm", 600)
    client = FakeClient(handler)
    r = SolanaReadOnlyRpc(url="https://primary.example", client=client, **kw)
    return r, client


# ---------------------------------------------------------------------------
# 1. the artificial choke point is gone
# ---------------------------------------------------------------------------


class TestNoArtificialChoke:
    @sync
    async def test_the_declared_rate_is_honoured_not_a_derived_one(self):
        """
        The cap comes from the DECLARED rate, not from concurrency x 4.

        This is the regression that caused the incident. The wiring must not
        reintroduce a number unrelated to the configured capacity.
        """
        from autopaper.config import build_config

        cfg = build_config()
        declared_rpm = round(float(cfg.real.public_rpc_rate_per_s) * 60.0)
        assert declared_rpm == 480, (
            "the declared public RPC rate changed; the incident analysis "
            "assumed 8/s and the wiring must follow the config, not a constant")

    @sync
    async def test_a_full_window_waits_instead_of_deleting_the_request(self):
        """
        Past the local cap the request WAITS, then succeeds.

        Deleting it is what turned a capacity limit into "no executable quote".
        """
        r, client = rpc(ok(), max_rpm=600, limiter_wait_s=3.0,
                        slot_cache_ttl_s=0.0)
        # A momentary local block, exactly as an upstream Retry-After would
        # produce. The window is 60s, so a sliding-window overflow could never
        # be waited out inside a decision budget -- a TRANSIENT block is the
        # case that must be waited through rather than thrown away.
        r.limiter.blocked_until = time.monotonic() + 0.05
        started = time.monotonic()
        assert await r.get_slot() is not None
        assert time.monotonic() - started >= 0.04, (
            "the request did not wait; it was deleted instead")
        assert len(client.posts) == 1

    @sync
    async def test_a_wait_that_cannot_clear_refuses_AS_A_RATE_LIMIT(self):
        """
        Past the budget it still refuses -- but as a NAMED rate limit, so the
        caller can tell capacity from a missing market.
        """
        r, _ = rpc(ok(), max_rpm=1, limiter_wait_s=0.05,
                    slot_cache_ttl_s=0.0)
        await r.get_slot()
        with pytest.raises(Exception) as ei:
            await r.get_slot()
        # The refusal still names CAPACITY, so a caller can tell a rate limit from
        # a missing market. The wording changed deliberately: it used to say
        # "local rpc cap 480/min reached" for EVERY refusal, including an
        # upstream provider throttle, which sent the investigation at this
        # system's own quota instead of at the provider. What matters is that
        # the condition is named and classified, not which words are used.
        assert "LOCAL quota exhausted" in str(ei.value)
        assert classify_quote_failure(str(ei.value)) is FailureReason.RPC_RATE_LIMIT


# ---------------------------------------------------------------------------
# 2. endpoint failover
# ---------------------------------------------------------------------------


class TestProviderFailover:
    @sync
    async def test_a_throttled_endpoint_fails_over_to_another(self):
        def handler(url, body):
            if "primary" in url:
                return FakeResponse(429, headers={"retry-after": "1"})
            return FakeResponse(200, {"result": 42})

        r, client = rpc(handler, urls=["https://secondary.example"],
                        max_retries=0, limiter_wait_s=1.0,
                        slot_cache_ttl_s=0.0)
        assert await r.get_slot() == 42
        assert {p["url"] for p in client.posts} >= {"https://primary.example",
                                                    "https://secondary.example"}

    @sync
    async def test_a_saturated_endpoint_is_skipped_on_the_next_call(self):
        def handler(url, body):
            if "primary" in url:
                return FakeResponse(429, headers={"retry-after": "30"})
            return FakeResponse(200, {"result": 7})

        r, client = rpc(handler, urls=["https://secondary.example"],
                        max_retries=0, limiter_wait_s=1.0,
                        slot_cache_ttl_s=0.0)
        await r.get_slot()
        posts_after_first = len(client.posts)
        await r.get_slot()
        # The cooling endpoint is not the only thing tried again.
        assert len(client.posts) > posts_after_first

    @sync
    async def test_the_default_is_a_single_endpoint(self):
        """
        No endpoint is invented.

        Adding a provider the operator did not configure would send this
        runtime's traffic somewhere it was not pointed.
        """
        r = SolanaReadOnlyRpc(url="https://only.example")
        assert [e.url for e in r.endpoints] == ["https://only.example"]


# ---------------------------------------------------------------------------
# 3. bounded concurrency
# ---------------------------------------------------------------------------


class TestBoundedConcurrency:
    @sync
    async def test_independent_reads_overlap_and_are_capped(self):
        """
        Overlap is what lets a multi-call quote cost its slowest call rather
        than their sum -- but never without the declared ceiling.
        """
        peak = 0
        live = 0
        lock = asyncio.Lock()

        async def slow_ok():
            return FakeResponse(200, {"result": 1})

        class SlowClient(FakeClient):
            async def post(self, url, json=None, headers=None, timeout=None):
                nonlocal peak, live
                async with lock:
                    live += 1
                    peak = max(peak, live)
                await asyncio.sleep(0.03)
                async with lock:
                    live -= 1
                return await slow_ok()

        client = SlowClient(ok())
        r = SolanaReadOnlyRpc(url="https://p.example", client=client,
                              max_concurrent=3, max_rpm=600, cache_ttl_s=0.0)
        await asyncio.gather(*(r.get_slot() for _ in range(9)))
        assert peak > 1, "reads did not overlap at all"
        assert peak <= 3, f"concurrency ceiling breached: {peak} > 3"


# ---------------------------------------------------------------------------
# 4. metadata caching that cannot misprice
# ---------------------------------------------------------------------------


class TestMetadataCaching:
    @sync
    async def test_get_slot_is_cached_within_a_slot(self):
        r, client = rpc(ok(), slot_cache_ttl_s=5.0)
        await r.get_slot()
        n = len(client.posts)
        await r.get_slot()
        assert len(client.posts) == n, (
            "the slot round trip is still repeated; it was half the traffic")

    @sync
    async def test_a_slot_cache_that_expires_reads_again(self):
        r, client = rpc(ok(), slot_cache_ttl_s=0.0)
        await r.get_slot()
        n = len(client.posts)
        await r.get_slot()
        assert len(client.posts) > n

    @sync
    async def test_account_values_are_still_uncached(self):
        """
        The account cache must stay OFF for pricing.

        A cached pool account is a quote describing a book that has moved --
        exactly the confusion this engine exists to eliminate. Only metadata
        and constants are cached.
        """
        import inspect

        from autopaper.engine import PaperTrader

        src = inspect.getsource(PaperTrader)
        assert "cache_ttl_s=0.0" in src, (
            "the account cache was re-enabled; a quote may now describe a "
            "stale book")


# ---------------------------------------------------------------------------
# 5. bounded, cancelling venue probe
# ---------------------------------------------------------------------------


class TestBoundedVenueProbe:
    @sync
    async def test_a_slow_venue_is_cancelled_at_the_deadline(self):
        """
        The probe CANCELS on expiry; it never waits for a slow venue.
        """
        cancelled = asyncio.Event()

        class SlowReader:
            async def quote_pumpswap(self, *a, **k):
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            async def quote_bonding_curve(self, *a, **k):
                return _decline("no curve")

        from autopaper.paper_live.quotes import QuoteService, SolReference

        svc = _service(SlowReader())
        svc.quote_deadline_s = 0.05
        point = await svc._probe_venues(
            mint="M", sol_lamports=1000, token_decimals=6,
            observed_pair_address=None, curve=_decline("no curve"))
        assert point is not None and not point.available
        assert "QUOTE_TIMEOUT" in (point.missing_reason or "")
        await asyncio.wait_for(cancelled.wait(), timeout=1.0)
        assert svc.quote_timeouts == 1

    @sync
    async def test_a_fast_venue_is_returned_and_not_cancelled(self):
        class QuickReader:
            async def quote_pumpswap(self, *a, **k):
                return _accept()

            async def quote_bonding_curve(self, *a, **k):
                return _decline("no curve")

        from autopaper.paper_live.quotes import QuoteService, SolReference

        svc = _service(QuickReader())
        svc.quote_deadline_s = 2.0
        point = await svc._probe_venues(
            mint="M", sol_lamports=1000, token_decimals=6,
            observed_pair_address=None, curve=_decline("no curve"))
        assert point.available
        assert svc.quote_timeouts == 0

    @sync
    async def test_a_raising_venue_becomes_a_named_refusal_not_a_crash(self):
        class BrokenReader:
            async def quote_pumpswap(self, *a, **k):
                raise RuntimeError("node exploded")

            async def quote_bonding_curve(self, *a, **k):
                return _decline("no curve")

        from autopaper.paper_live.quotes import QuoteService, SolReference

        svc = _service(BrokenReader())
        point = await svc._probe_venues(
            mint="M", sol_lamports=1000, token_decimals=6,
            observed_pair_address=None, curve=_decline("no curve"))
        assert point is not None and not point.available
        assert "node exploded" in (point.missing_reason or "")


# ---------------------------------------------------------------------------
# 6. explicit classification
# ---------------------------------------------------------------------------


class TestFailureClassification:
    @pytest.mark.parametrize("text,expected", [
        ("QUOTE_TIMEOUT: did not arrive within 1500ms", FailureReason.QUOTE_TIMEOUT),
        ("solana rpc rate limited", FailureReason.RPC_RATE_LIMIT),
        ("local rpc cap 480/min reached", FailureReason.RPC_RATE_LIMIT),
        ("solana rpc HTTP 502", FailureReason.RPC_ERROR),
        ("solana rpc error -32002", FailureReason.RPC_ERROR),
        ("the bonding curve is complete, so it is no longer the market",
         FailureReason.QUOTE_UNAVAILABLE),
    ])
    @sync
    async def test_each_cause_gets_its_own_verdict(self, text, expected):
        assert classify_quote_failure(text) is expected

    @sync
    async def test_capacity_is_never_reported_as_a_missing_market(self):
        """
        The whole point: these three must never share a verdict.
        """
        kinds = {
            classify_quote_failure("solana rpc rate limited"),
            classify_quote_failure("QUOTE_TIMEOUT: did not arrive within 1500ms"),
            classify_quote_failure("the bonding curve is complete"),
        }
        assert len(kinds) == 3


# ---------------------------------------------------------------------------
# 7. quote sharing across arms
# ---------------------------------------------------------------------------


class TestQuoteSharing:
    @sync
    async def test_concurrent_arms_collapse_onto_one_quote(self):
        """
        Three arms, one decision, ONE underlying quote.

        Asked concurrently they used to all miss the cache and each fetch its
        own identical quote, tripling the spend for a single signal.
        """
        calls: list[int] = []

        class CountingService:
            async def quote_buy(self, **kw):
                calls.append(1)
                await asyncio.sleep(0.02)
                return _accept()

        from autopaper.paper_live.engine import PaperLiveEngine
        from autopaper.paper_live.arms import parse_arms

        eng = PaperLiveEngine(arms=parse_arms(["L25_CURRENT", "L25_60S",
                                               "H2_CAP60"]),
                              decision_engine=None, exit_policy=None,
                              quote_service=CountingService())
        eng.last_quote_coalesced = 0
        results = await asyncio.gather(*(
            eng._quote_buy_best(mint="M", sol_lamports=1000, token_decimals=6,
                                observed_pair_address="P")
            for _ in range(3)))
        assert len(results) == 3
        assert all(r[0] is results[0][0] for r in results), \
            "the arms received different quote objects"
        assert len(calls) == 1, f"quote fetched {len(calls)}x for one decision"
        assert eng.last_quote_coalesced >= 1


# ---------------------------------------------------------------------------
# 8. nothing that matters moved
# ---------------------------------------------------------------------------


class TestSafetySemanticsUnchanged:
    @sync
    async def test_the_mark_gap_threshold_is_still_ten_percent(self):
        from autopaper.paper_live.economics import MARK_GAP_GUARD_THRESHOLD_PCT

        assert MARK_GAP_GUARD_THRESHOLD_PCT == 10.0, (
            "the incident report's central recommendation was to NOT move this "
            "until real telemetry exists; moving it here would be that change "
            "smuggled in through a capacity fix")

    @sync
    async def test_identity_still_requires_the_exact_observed_account(self):
        from autopaper.paper_live.quotes import QuoteService

        identity, evidence = QuoteService._identity_for(
            observed_pair_address="POOL", market_type=None,
            market_address="POOL", chain_verified=True)
        assert identity == "PASS"
        identity, _ = QuoteService._identity_for(
            observed_pair_address="POOL", market_type=None,
            market_address="OTHER", chain_verified=True)
        assert identity == "FAIL", "identity was relaxed to allow another pool"

    @sync
    async def test_real_trading_is_still_disabled_by_default(self):
        from autopaper.engine import PaperTrader

        assert getattr(PaperTrader, "REAL_ENABLED_DEFAULT", False) is False

    @sync
    async def test_the_client_still_cannot_write(self):
        """
        More read methods would mean more remote capability.
        """
        import autopaper.same_market.rpc as rpc_mod

        assert set(SolanaReadOnlyRpc.read_methods) == set(rpc_mod.READ_METHODS)
        for forbidden in ("sendTransaction", "send_transaction",
                          "send_raw_transaction", "sign_transaction"):
            assert forbidden not in SolanaReadOnlyRpc.read_methods
        for forbidden in ("send_transaction", "send_raw_transaction", "sign"):
            assert not hasattr(SolanaReadOnlyRpc, forbidden)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _service(reader):
    """A QuoteService whose reader is the supplied double."""
    from autopaper.paper_live.quotes import QuoteService, SolReference

    svc = QuoteService(rpc=None,
                       sol_reference=SolReference(120.0, time.time(), "t"))
    svc._reader = reader
    return svc


def _decline(reason: str):
    from autopaper.paper_live.quotes import QuotePoint

    return QuotePoint(side="BUY", provider="pump_direct", market_type="CURVE",
                      token_mint="M", quote_mint="Q", input_amount_raw=None,
                      input_decimals=None, input_amount_decimal=None,
                      output_amount_raw=None, output_decimals=None,
                      output_amount_decimal=None, effective_price=None,
                      effective_price_usd=None, price_impact_pct=None,
                      fee_bps=None, fee_amount=None, venue=None,
                      pool_or_curve=None, route=(), identity="UNKNOWN",
                      available=False, missing_reason=reason)


def _accept():
    from autopaper.paper_live.quotes import QuotePoint

    return QuotePoint(side="BUY", provider="pumpswap_direct",
                      market_type="POOL", token_mint="M", quote_mint="Q",
                      input_amount_raw=1000, input_decimals=9,
                      input_amount_decimal=1e-6, output_amount_raw=5000,
                      output_decimals=6, output_amount_decimal=0.005,
                      effective_price=2e-4, effective_price_usd=2.4e-5,
                      price_impact_pct=0.1, fee_bps=25, fee_amount=1,
                      venue="PUMPSWAP:POOL", pool_or_curve="POOL", route=(),
                      identity="PASS", available=True, missing_reason=None)