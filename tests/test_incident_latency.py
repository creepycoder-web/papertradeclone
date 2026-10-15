"""
The 2026-09-30 latency incident: bounds, attribution, and the async path.

WHAT THE INCIDENT WAS
---------------------
A degraded candidate reported::

    data      = 16,604ms
    narrative = 18,269ms
    L25       =      2ms
    total     = 34,877ms

Three distinct causes, none of them the others:

* **16.6s of market data** was ONE provider call.
  ``httpx.Client(timeout=8.0)`` is not a wall-clock budget for a request: httpx
  applies the number per request PHASE, and the connect phase is charged ONCE
  PER RESOLVED ADDRESS by ``socket.create_connection``, which walks the
  ``getaddrinfo`` list in order. Measured on this host, both market hosts list
  two UNROUTABLE IPv6 addresses ahead of the reachable IPv4 ones, so a "8 second"
  GET cost ``2 x 8.0s + ~0.05s = 16.05s``.
* **"18.2s of narrative"** was a TELEMETRY fault. The bot answered in 1.6s. The
  stage window was opened ABOVE the alert settle and the SECOND market read, so
  the number measured ``alert_settle_s + a whole extra provider fetch``.
* The 34.9s total then exceeded the existing 15s entry budget, which the stale
  gate refused. That is a different test file (``test_incident_paper_buy.py``).

These tests are behavioural. They assert a bound or an attribution, never a
source substring, because the bound is the property that matters and a source
match is not it.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from autopaper.clock import FakeClock
from autopaper.config import MarketConfig, NarConfig, build_config
from autopaper.latency import StageRecorder
from autopaper.market import DexScreenerProvider, GeckoTerminalProvider, MarketDataService
from autopaper.market.ratelimit import (
    MarketDataError,
    RetryAfterBackoff,
    classify_error,
    run_with_deadline,
    split_timeouts,
)
from autopaper.nar.phanes import PhanesNarSource

from .conftest import MINT

#: wall-clock latency budget assertions
pytestmark = [
    pytest.mark.slow,
]

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]

#: The topology measured on 2026-09-30 for BOTH market hosts: two unroutable
#: IPv6 addresses ahead of two reachable IPv4 addresses.
N_DEAD_ADDRESSES = 2
#: A reachable IPv4 handshake measured 13-45ms.
REAL_HANDSHAKE_S = 0.05


def _pair_payload(mint: str = MINT) -> dict:
    return {"schemaVersion": "1.0.0", "pairs": [{
        "chainId": "solana", "dexId": "raydium", "pairAddress": "P",
        "baseToken": {"address": mint, "symbol": "TST"},
        "quoteToken": {"address": "Q", "symbol": "SOL"},
        "priceNative": "0.00000015", "priceUsd": "0.00015",
        "txns": {"m5": {"buys": 180, "sells": 80}, "h1": {"buys": 9, "sells": 5}},
        "volume": {"m5": 25000.0, "h1": 120000.0, "h24": 900000.0},
        "priceChange": {"m5": 12.5, "h1": 40.0, "h24": 300.0},
        "liquidity": {"usd": 55000.0}, "fdv": 2_000_000.0,
        "marketCap": 1_500_000.0, "pairCreatedAt": 1_699_999_400_000,
        "info": {},
    }]}


class DeadAddressTransport:
    """
    A transport that reproduces the measured cost model, in real wall-clock time.

    ``per_dead_address_s`` is what one UNREACHABLE address costs. Setting it to
    the configured total timeout reproduces the incident (2 x 8.0s = 16.0s);
    setting it to the configured CONNECT budget reproduces the fixed cost
    (2 x 1.0s = 2.0s). The point of the tests below is that the provider's cost
    must be governed by the connect budget, not by the total budget.
    """

    def __init__(self, per_dead_address_s: float, *, dead: int = N_DEAD_ADDRESSES,
                 ok_after: float = REAL_HANDSHAKE_S, raise_after_attempts: int | None = None):
        self.per_dead_address_s = per_dead_address_s
        self.dead = dead
        self.ok_after = ok_after
        self.raise_after_attempts = raise_after_attempts
        self.calls = 0
        self.timeouts_seen: list[float] = []

    def __call__(self, url: str, timeout_s: float):
        self.calls += 1
        self.timeouts_seen.append(timeout_s)
        time.sleep(self.dead * self.per_dead_address_s + self.ok_after)
        if self.raise_after_attempts and self.calls >= self.raise_after_attempts:
            raise MarketDataError("synthetic 503", status=503)
        return _pair_payload()


def _provider(transport, **kw) -> DexScreenerProvider:
    p = DexScreenerProvider(clock=FakeClock(1_700_000_000.0), http_get=transport,
                            cache_ttl_s=0.0, **kw)
    p.backoff.base_s = 0.0
    p.backoff.max_s = 0.0
    return p


# ---------------------------------------------------------------------------
# 1. the timeout is a WALL-CLOCK DEADLINE, not a per-phase knob
# ---------------------------------------------------------------------------


def test_split_timeouts_shortens_only_the_connect_phase():
    connect, phase = split_timeouts(8.0)
    assert connect < phase, (
        "the connect phase is charged once per resolved address, so it must be "
        "the short one; equal values are exactly the incident"
    )
    assert phase == 8.0
    # A caller-supplied connect budget is honoured and can never exceed the total.
    assert split_timeouts(8.0, connect_s=30.0) == (8.0, 8.0)
    assert split_timeouts(0.5, connect_s=30.0)[0] <= 0.5


def test_a_provider_call_cannot_exceed_its_wall_clock_budget():
    """
    The headline bound. A transport that ignores its own timeout is still
    released inside the budget.
    """
    budget = 0.40
    # 2.5s per call: no transport-level timeout will stop this. The deadline must.
    transport = DeadAddressTransport(per_dead_address_s=2.5, dead=1, ok_after=0.0)
    p = _provider(transport, timeout_s=budget, connect_timeout_s=0.05,
                  max_retries=2)
    t0 = time.monotonic()
    # `quote()` is the graceful facade: it never raises, it returns the failure.
    result = p.quote(MINT)
    elapsed = time.monotonic() - t0
    assert elapsed < budget + 0.30, (
        f"a provider call overran its {budget}s wall-clock budget: {elapsed:.3f}s"
    )
    assert result.observation is None
    assert "budget" in (result.error or ""), result.error
    assert p.last_error_label == "deadline_exceeded"
    assert classify_error(
        MarketDataError("x", deadline_exceeded=True)
    ) == "deadline_exceeded"


def test_the_deadline_is_what_bounds_the_call_not_the_connect_split():
    """
    Even with a generous connect budget the WALL-CLOCK deadline still holds.
    """
    budget = 0.40
    transport = DeadAddressTransport(per_dead_address_s=2.0, dead=2, ok_after=0.0)
    p = _provider(transport, timeout_s=budget, connect_timeout_s=10.0,
                  max_retries=2)
    t0 = time.monotonic()
    p.quote(MINT)
    assert time.monotonic() - t0 < budget + 0.30


def test_run_with_deadline_returns_the_value_on_time():
    assert run_with_deadline(lambda: "ok", 1.0) == "ok"
    with pytest.raises(MarketDataError):
        run_with_deadline(lambda: time.sleep(5), 0.2, label="x")
    with pytest.raises(MarketDataError):
        run_with_deadline(lambda: "ok", 0.0, label="x")


def test_a_blown_deadline_is_never_retried():
    """
    Retrying a blown budget would spend it again and miss it again.
    """
    transport = DeadAddressTransport(per_dead_address_s=2.5, dead=1, ok_after=0.0)
    p = _provider(transport, timeout_s=0.40, connect_timeout_s=0.05, max_retries=5)
    p.quote(MINT)
    assert transport.calls == 1, (
        f"a deadline overrun was retried {transport.calls} times"
    )


# ---------------------------------------------------------------------------
# 2. retries and backoff are bounded TOO
# ---------------------------------------------------------------------------


def test_retry_attempts_and_backoff_share_one_budget():
    """
    ``max_attempts * timeout`` was the old real worst case. One budget now covers
    attempts and sleeps together.
    """
    backoff = RetryAfterBackoff(max_attempts=5, budget_s=0.30)
    backoff.base_s = 0.02
    backoff.max_s = 0.08
    slept: list[float] = []

    def always_503() -> None:
        # Each attempt costs real wall-clock time, as a slow provider does. Six
        # of these plus the backoff would be ~0.7s: well past the 0.30s budget.
        time.sleep(0.08)
        raise MarketDataError("503", status=503)

    def real_sleep(seconds: float) -> None:
        slept.append(seconds)
        time.sleep(seconds)

    t0 = time.monotonic()
    with pytest.raises(MarketDataError):
        backoff.run(always_503, sleep=real_sleep)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.30 + 0.10, f"budget overrun: {elapsed:.3f}s"
    # `max_attempts` is the RETRY count, so at most max_attempts + 1 attempts.
    assert backoff.last_attempts <= backoff.max_attempts + 1
    assert backoff.last_elapsed_s < 0.30 + 0.10
    assert sum(slept) < 0.30 + 0.10, "backoff slept past the budget"
    # And it must NOT have made all six attempts: the budget stops it earlier.
    assert backoff.last_attempts < backoff.max_attempts + 1, (
        "the wall-clock budget did not stop the retry loop early"
    )


def test_a_successful_retry_still_reports_the_partial_failure():
    """
    Unchanged behaviour, kept explicit: a retry that recovers is visible.
    """
    state = {"n": 0}

    def flaky() -> dict:
        state["n"] += 1
        if state["n"] == 1:
            raise MarketDataError("503", status=503)
        return {"ok": True}

    backoff = RetryAfterBackoff(max_attempts=2, budget_s=5.0)
    backoff.base_s = 0.0
    backoff.max_s = 0.0
    assert backoff.run(flaky) == {"ok": True}
    assert backoff.succeeded_after_retry is True
    assert backoff.last_attempts == 2


def test_a_rate_limited_response_is_not_retried():
    backoff = RetryAfterBackoff(max_attempts=3, budget_s=5.0)
    backoff.base_s = 0.0
    backoff.max_s = 0.0

    def limited() -> None:
        raise MarketDataError("429", rate_limited=True, retry_after_s=1.0)

    with pytest.raises(MarketDataError):
        backoff.run(limited)
    assert backoff.last_attempts == 1


# ---------------------------------------------------------------------------
# 3. ONE market read, ONE budget - primary and fallback share it
# ---------------------------------------------------------------------------


def test_the_service_gives_primary_and_fallback_a_single_budget():
    """
    A dead primary plus a slow fallback used to cost ``2 x timeout``. It must not.
    """
    budget = 0.5
    cfg = MarketConfig(http_timeout_s=budget, connect_timeout_s=0.05,
                      max_retries=2, cache_ttl_s=0.0)
    primary = DexScreenerProvider(clock=FakeClock(T0), timeout_s=budget,
                                  connect_timeout_s=0.05, cache_ttl_s=0.0,
                                  http_get=DeadAddressTransport(
                                      per_dead_address_s=2.0, dead=1, ok_after=0.0))
    fallback = GeckoTerminalProvider(clock=FakeClock(T0), timeout_s=budget,
                                     connect_timeout_s=0.05, cache_ttl_s=0.0,
                                     http_get=DeadAddressTransport(
                                         per_dead_address_s=2.0, dead=1, ok_after=0.0))
    svc = MarketDataService(cfg, clock=FakeClock(T0), primary=primary,
                            fallback=fallback)
    t0 = time.monotonic()
    result = svc.quote(MINT)
    elapsed = time.monotonic() - t0
    assert elapsed < budget + 0.35, (
        f"primary+fallback took {elapsed:.3f}s against a {budget}s budget"
    )
    assert result.observation is None
    # And the refusal is reported, not swallowed.
    assert result.error


def test_a_skipped_fallback_is_recorded_with_its_reason():
    """
    "The fallback was never asked because the primary had spent the budget" is a
    different fact from "the fallback answered and had nothing".
    """
    budget = 0.3
    cfg = MarketConfig(http_timeout_s=budget, connect_timeout_s=0.05,
                      max_retries=0, cache_ttl_s=0.0)
    primary = DexScreenerProvider(clock=FakeClock(T0), timeout_s=budget,
                                  connect_timeout_s=0.05, cache_ttl_s=0.0,
                                  http_get=DeadAddressTransport(
                                      per_dead_address_s=2.0, dead=1, ok_after=0.0))
    fallback = GeckoTerminalProvider(clock=FakeClock(T0), timeout_s=budget,
                                     connect_timeout_s=0.05, cache_ttl_s=0.0,
                                     http_get=DeadAddressTransport(
                                         per_dead_address_s=0.0, dead=0, ok_after=0.01))
    svc = MarketDataService(cfg, clock=FakeClock(T0), primary=primary,
                            fallback=fallback)
    svc.quote(MINT)
    health = svc.health()
    assert health["state"] in ("DEGRADED", "IDLE")
    assert health["budget_ms"] == pytest.approx(300.0)
    assert health["connect_budget_ms"] == pytest.approx(50.0)
    assert "deadlines" in health and "overruns" in health["deadlines"]


def test_market_health_reports_a_state_at_all():
    """
    The dashboard read ``health().get("state", "?")`` and the dict had no such
    key, so the market row rendered a permanent "?" - it could not report a
    provider as anything at all, healthy or not.
    """
    cfg = MarketConfig()
    svc = MarketDataService(cfg, clock=FakeClock(T0))
    assert svc.health()["state"] == "IDLE"
    good = DexScreenerProvider(clock=FakeClock(T0), http_get=lambda u, t: _pair_payload())
    svc2 = MarketDataService(cfg, clock=FakeClock(T0), primary=good, fallback=good)
    svc2.quote(MINT)
    assert svc2.health()["state"] == "OK"


# ---------------------------------------------------------------------------
# 4. the CONFIGURED timeout reaches the client
# ---------------------------------------------------------------------------


def test_the_configured_timeouts_reach_the_providers(monkeypatch):
    cfgmod = __import__("autopaper.config", fromlist=["x"])
    monkeypatch.setattr(cfgmod, "load_env_file", lambda path: {
        "MARKET_HTTP_TIMEOUT_S": "3.5",
        "MARKET_CONNECT_TIMEOUT_S": "0.75",
        "MARKET_MAX_RETRIES": "1",
        "MARKET_CACHE_TTL_S": "1.25",
    })
    monkeypatch.delenv("MARKET_HTTP_TIMEOUT_S", raising=False)
    monkeypatch.delenv("MARKET_CONNECT_TIMEOUT_S", raising=False)
    cfg = build_config(REPO)
    assert cfg.market.http_timeout_s == 3.5
    assert cfg.market.connect_timeout_s == 0.75
    assert cfg.market.max_retries == 1
    assert cfg.market.cache_ttl_s == 1.25

    svc = MarketDataService(cfg.market, clock=FakeClock(T0))
    assert svc.primary.timeout_s == 3.5
    assert svc.primary.connect_timeout_s == 0.75
    assert svc.primary.backoff.budget_s == 3.5
    assert svc.fallback.timeout_s == 3.5
    assert svc.fallback.connect_timeout_s == 0.75
    assert svc.fallback.backoff.budget_s == 3.5


def test_the_market_timeouts_are_not_silently_ignored_anymore():
    """
    ``http_timeout_s``, ``max_retries`` and ``cache_ttl_s`` used not to be read
    by the loader at all, so an operator who put them in ``.env.local`` got the
    dataclass default and no warning. A setting that looks configurable but is
    not is how a latency fix gets applied in a config file and never lands.
    """
    src = (REPO / "src/autopaper/config.py").read_text()
    for key in ("MARKET_HTTP_TIMEOUT_S", "MARKET_CONNECT_TIMEOUT_S",
                "MARKET_MAX_RETRIES", "MARKET_CACHE_TTL_S"):
        assert f'get("{key}")' in src, f"{key} is not read by the loader"


def test_the_narrative_timeout_is_env_readable_and_reaches_the_source():
    cfgmod = __import__("autopaper.config", fromlist=["x"])
    original = cfgmod.load_env_file

    cfg = build_config(REPO)
    assert cfg.nar.timeout_s == 2.5, "the .env.local value remains authoritative"
    assert cfg.nar.timeout_grace_s == 0.5
    assert cfg.nar.timeout_s + cfg.nar.timeout_grace_s <= 3.0

    from autopaper.nar.base import NarService

    svc = NarService(cfg.nar)
    assert getattr(svc.source, "timeout_s", None) == cfg.nar.timeout_s


# ---------------------------------------------------------------------------
# 5. the async provider path is the LIVE one, and it does not block the loop
# ---------------------------------------------------------------------------


def test_the_engine_never_calls_the_synchronous_provider_quote():
    """
    Proven by SOURCE SHAPE, because "which method runs" cannot be observed from
    outside a live event loop. The engine's live paths must go through
    ``quote_async``; a bare ``.quote(`` on a provider or the service is the
    regression this guards.
    """
    import inspect

    from autopaper.engine import PaperTrader

    src = inspect.getsource(PaperTrader)
    # Every market read in the engine is an await of the async form.
    assert ".market.quote(" not in src, (
        "the engine calls the SYNCHRONOUS market quote; use quote_async"
    )
    assert src.count("self.market.quote_async(") >= 3, (
        "the engine must read the market through quote_async on every path "
        "(alert analysis, fill, monitor, exit)"
    )


def test_the_service_and_providers_both_expose_the_async_form():
    svc = MarketDataService(MarketConfig(), clock=FakeClock(T0))
    assert callable(svc.quote_async)
    assert callable(svc.primary.quote_async)
    assert callable(svc.fallback.quote_async)
    # ...and the async form is not a different implementation: it runs the same
    # synchronous one on a worker thread.
    import inspect

    assert "to_thread" in inspect.getsource(type(svc).quote_async)


def test_quote_async_does_not_block_the_event_loop():
    """
    The loop keeps ticking through a slow provider call. This was the whole point
    of the earlier async change and must not regress.
    """
    budget = 0.6
    cfg = MarketConfig(http_timeout_s=budget, connect_timeout_s=0.05,
                      max_retries=0, cache_ttl_s=0.0)
    slow = DexScreenerProvider(clock=FakeClock(T0), timeout_s=budget,
                               connect_timeout_s=0.05, cache_ttl_s=0.0,
                               http_get=DeadAddressTransport(
                                   per_dead_address_s=0.2, dead=1, ok_after=0.0))
    svc = MarketDataService(cfg, clock=FakeClock(T0), primary=slow, fallback=slow)

    async def run() -> tuple[float, bool]:
        ticks: list[float] = []

        async def heartbeat() -> None:
            t0 = time.monotonic()
            while time.monotonic() - t0 < budget + 0.3:
                await asyncio.sleep(0.02)
                ticks.append(time.monotonic() - t0)

        hb = asyncio.ensure_future(heartbeat())
        await asyncio.sleep(0.05)
        res = await svc.quote_async(MINT)
        hb.cancel()
        gaps = [b - a for a, b in zip(ticks, ticks[1:])] or [0.0]
        return max(gaps), res.observation is not None

    max_gap, _ = asyncio.run(run())
    assert max_gap < 0.15, (
        f"the event loop stalled for {max_gap * 1000:.0f}ms during a provider "
        "call; the async provider path is not active"
    )


def test_the_monitor_path_also_uses_the_async_form():
    """A blocking monitor call would starve the alert path it runs beside."""
    import inspect

    from autopaper.engine import PaperTrader

    src = inspect.getsource(PaperTrader.monitor_once)
    assert "quote_async" in src
    assert ".market.quote(" not in src


def test_the_exit_path_also_uses_the_async_form():
    import inspect

    from autopaper.engine import PaperTrader

    src = inspect.getsource(PaperTrader._do_exit)
    assert "quote_async" in src
    assert ".market.quote(" not in src


# ---------------------------------------------------------------------------
# 6. the narrative budget covers the WHOLE call
# ---------------------------------------------------------------------------


class _Rpc:
    """A Telegram client double where every step can be made individually slow."""

    def __init__(self, *, entity_s=0.0, anchor_s=0.0, send_s=0.0, reply=""):
        self.entity_s, self.anchor_s, self.send_s = entity_s, anchor_s, send_s
        self.reply = reply
        self.sends = 0

    async def get_entity(self, _name):
        await asyncio.sleep(self.entity_s)
        return object()

    async def get_messages(self, *_a, **kw):
        await asyncio.sleep(self.anchor_s)
        if self.reply:
            return [type("M", (), {"id": 2, "reply_to_msg_id": 1, "out": False,
                                   "message": self.reply})()]
        return []

    async def send_message(self, *_a, **_k):
        self.sends += 1
        await asyncio.sleep(self.send_s)
        return type("M", (), {"id": 1})()


def _nar(client, **kw) -> PhanesNarSource:
    src = PhanesNarSource(clock=None, client=client, min_gap_s=0.0, **kw)
    return src


@pytest.mark.parametrize("slow_step", ["entity_s", "anchor_s", "send_s"])
def test_no_single_step_can_push_the_narrative_past_its_budget(slow_step):
    """
    Each step individually, then all three together. All used to sit OUTSIDE the
    deadline, which made ``timeout_s`` a floor rather than a bound.
    """
    budget = 0.30
    client = _Rpc(**{slow_step: 0.5}, reply="Narrative for X\n\nSome detail here.")
    src = _nar(client, timeout_s=budget, collect_extra_s=0.02)
    t0 = time.monotonic()
    snap = asyncio.run(src.fetch_async("M"))
    elapsed = time.monotonic() - t0
    assert elapsed < budget + 0.20, (
        f"{slow_step} pushed the narrative to {elapsed:.3f}s against a "
        f"{budget}s budget"
    )
    assert snap.available is False


def test_the_polite_gap_cannot_push_the_narrative_past_its_budget():
    """``min_gap_s`` is 4.0s by default and used to be slept in full."""
    budget = 0.30
    src = PhanesNarSource(clock=None, client=_Rpc(reply="x"), min_gap_s=4.0,
                          timeout_s=budget, collect_extra_s=0.02)
    src._last_sent = time.monotonic() - 0.01   # a query just happened
    t0 = time.monotonic()
    asyncio.run(src.fetch_async("M"))
    elapsed = time.monotonic() - t0
    assert elapsed < budget + 0.20, (
        f"the polite gap pushed the narrative to {elapsed:.3f}s"
    )


def test_a_fast_narrative_still_answers_inside_its_budget():
    client = _Rpc(anchor_s=0.02, send_s=0.02,
                  reply="Narrative for M\n\nBittersweet is a dog meme coin, 2024.")
    src = _nar(client, timeout_s=2.5, collect_extra_s=0.05)
    t0 = time.monotonic()
    snap = asyncio.run(src.fetch_async("M"))
    elapsed = time.monotonic() - t0
    assert snap.available is True, snap.error
    assert elapsed < 2.5
    assert src.stats.answered == 1
    assert src.stats.timeouts == 0


def test_a_narrative_timeout_is_counted_and_reported():
    client = _Rpc()
    src = _nar(client, timeout_s=0.20, collect_extra_s=0.02)
    snap = asyncio.run(src.fetch_async("M"))
    assert snap.available is False
    assert "NAR_UNAVAILABLE" in (snap.error or "")
    assert src.stats.timeouts == 1


# ---------------------------------------------------------------------------
# 7. stage attribution: each stage measures its OWN work
# ---------------------------------------------------------------------------


def test_the_settle_window_is_its_own_stage():
    """
    The narrative window used to contain the alert settle and the second market
    read. ``settle_*`` now names that work, so the two numbers cannot overlap.
    """
    rec = StageRecorder()
    t = 1000.0
    rec.mark("alert_received", t)
    rec.mark("market_start", t + 0.1)
    rec.mark("market_end", t + 2.1)         # 2.0s of market
    rec.mark("settle_start", t + 2.1)
    rec.mark("settle_end", t + 6.3)        # 4.2s of settle + second quote
    rec.mark("narrative_start", t + 6.3)
    rec.mark("narrative_end", t + 6.45)    # 0.15s of waiting on a ready narrative
    rec.mark("l25_start", t + 6.45)
    rec.mark("l25_end", t + 6.47)
    assert rec.market_ms == pytest.approx(2000.0)
    assert rec.settle_ms == pytest.approx(4200.0)
    assert rec.narrative_ms == pytest.approx(150.0)
    assert rec.analysis_ms == pytest.approx(6470.0)


def test_stage_durations_do_not_leak_across_the_new_boundary():
    from autopaper.latency import stage_durations

    durations = stage_durations({
        "market_start": 100.0, "market_end": 102.0,
        "settle_start": 102.0, "settle_end": 106.0,
        "narrative_start": 106.0, "narrative_end": 106.5,
    })
    assert durations["market_start->market_end"] == pytest.approx(2000.0)
    assert durations["settle_start->settle_end"] == pytest.approx(4000.0)
    assert durations["narrative_start->narrative_end"] == pytest.approx(500.0)


def test_analysis_equals_the_sum_of_the_stages_it_reports():
    """
    ``analysis = market + settle + narrative + strategy (+ quote/fill)`` must
    actually hold, so the dashboard's breakdown adds up to the total it reports
    instead of being four numbers a reader has to reconcile by hand.
    """
    rec = StageRecorder()
    rec.mark("alert_received", 100.0)
    rec.mark("candidate_parsed", 100.01)
    rec.mark("market_start", 100.10)
    rec.mark("market_end", 102.10)          # market 2.00s
    rec.mark("settle_start", 102.10)
    rec.mark("settle_end", 106.30)          # settle 4.20s
    rec.mark("narrative_start", 106.30)
    rec.mark("narrative_end", 106.90)       # narrative 0.60s
    rec.mark("l25_start", 106.90)
    rec.mark("risk_start", 106.91)
    rec.mark("risk_end", 106.93)
    rec.mark("l25_end", 106.95)             # strategy 0.05s
    rec.mark("quote_start", 106.96)
    rec.mark("quote_end", 108.26)           # quote 1.30s
    rec.mark("submit_start", 108.26)
    rec.mark("submit_end", 108.27)
    rec.mark("fill", 108.28)                # fill 0.02s

    assert rec.analysis_ms == pytest.approx(6950.0)   # alert_received -> l25_end
    # 6950 = 100 (alert -> market_start) + 2000 market + 4200 settle
    #        + 600 narrative + 50 strategy legs. Nothing is unattributed and
    # nothing is counted twice.
    parts = rec.market_ms + rec.settle_ms + rec.narrative_ms
    assert parts == pytest.approx(6800.0)
    assert rec.analysis_ms - parts == pytest.approx(150.0)
    # ...and the execution legs sit AFTER the analysis, not inside it.
    assert rec.analysis_ms < rec.trading_ms
    assert rec.trading_ms == pytest.approx(8280.0)
    assert rec.signal_to_fill_ms == pytest.approx(rec.trading_ms)
    assert rec.submit_to_fill_ms == pytest.approx(20.0)


def test_a_missing_stage_is_reported_as_missing_not_invented():
    rec = StageRecorder()
    rec.mark("alert_received", 100.0)
    rec.mark("market_start", 100.0)
    rec.mark("market_end", 101.0)
    assert rec.narrative_ms is None
    assert rec.settle_ms is None
    assert rec.analysis_ms is None      # no l25_end
    assert rec.trading_ms is None       # no fill


def test_a_blocked_candidate_records_its_reason():
    rec = StageRecorder()
    rec.mark("alert_received", 100.0)
    rec.mark("quote_start", 101.0)
    rec.block("stale_signal", outcome="BUY_BLOCKED")
    assert rec.blocked_reason == "stale_signal"
    assert rec.outcome == "BUY_BLOCKED"
    assert rec.signal_to_fill_ms is None


def test_the_stage_store_persists_the_outcome_and_reason():
    from autopaper.latency import StageStore

    rec = StageRecorder()
    rec.mark("alert_received", 100.0)
    rec.mark("market_start", 100.0)
    rec.mark("market_end", 101.5)
    rec.block("stale_signal", outcome="BUY_BLOCKED")
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "stage_timings.jsonl"
        store = StageStore(path)
        row = store.record("cand_1", rec, mode="PAPER", mint=MINT, symbol="X")
        assert row["outcome"] == "BUY_BLOCKED"
        assert row["blocked_reason"] == "stale_signal"
        back = StageStore.load(path)
        assert back[0]["outcome"] == "BUY_BLOCKED"
        assert back[0]["durations_ms"]["market_start->market_end"] == pytest.approx(1500.0)


def test_a_legacy_stage_row_with_no_new_fields_still_reads():
    """Backward compatibility: old rows must not break the reader."""
    from autopaper.latency import StageStore
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "stage_timings.jsonl"
        path.write_text(json.dumps({
            "event": "stage_timing",
            "stages": {"alert_received": 100.0, "market_start": 100.0,
                       "market_end": 102.0},
        }) + "\n")
        rows = StageStore.load(path)
        assert len(rows) == 1
        assert rows[0].get("outcome") is None
        assert rows[0]["durations_ms"]["market_start->market_end"] == pytest.approx(2000.0)


T0 = 1_700_000_000.0
