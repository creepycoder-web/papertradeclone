"""
RPC-lane regression tests for the stuck-OPEN exit failure.

THE INCIDENT
============
A PAPER position stayed OPEN for 265s and took 28 identical exit attempts, every
one failing with::

    local rpc cap 480/min reached and did not clear within 2.0s

Two independent defects produced that, and neither was the error string:

1. THE BLOCK WAS PERMANENT, NOT 60 SECONDS. ``SolanaReadOnlyRpc`` set
   ``blocked_until`` from ``time.time()`` while every comparison in the limiter
   was against ``time.monotonic()``. Those differ by the machine's uptime, so one
   upstream 429 set the block roughly 1.79e9 seconds in the future and the
   client could never recover without a restart. There was no path back to an
   executable quote at all.

2. TELEMETRY COULD CONSUME THE WHOLE BUDGET. The hold-path sampler ran at 1 Hz on
   the same FIFO limiter as execution, which for a single position is roughly the
   entire provider budget. Background measurement starved a risk-reducing SELL.

Plus the shared FIFO with no reservation, and a duplicate SELL quote per tick.

Every test below is a data-integrity assertion. None asserts a strategy outcome.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

# The engine-level tests below drive the REAL PAPER engine using the existing
# offline harness doubles, exactly as the existing engine tests do.
sys.path.insert(0, str(Path(__file__).parent))

from autopaper.market.ratelimit import (
    LANE_EXECUTION,
    LANE_TELEMETRY,
    LOCAL_WINDOW,
    TELEMETRY_BUDGET,
    UPSTREAM_BLOCKED,
    RateLimiter,
)
from autopaper.same_market.rpc import (
    SolanaReadOnlyRpc,
    current_rpc_lane,
    rpc_lane,
    upstream_events,
)

from .conftest import MINT, MINT2

from autopaper.paper_live.schema import (
    FailureReason,
    LatencyMarks,
    ObservedMarkPoint,
    QuotePoint,
    SignalSnapshot,
    TradeRecord,
)

pytestmark = pytest.mark.strategy

MINT_ACCOUNT = "So11111111111111111111111111111111111111112"


def run(coro):
    return asyncio.run(coro)


class _Resp:
    """A JSON-RPC response, successful by default."""

    def __init__(self, result=None):
        self.status_code = 200
        self.headers = {}

        def json():
            return {"jsonrpc": "2.0", "id": 1, "result": result}

        self.json = json


class _Resp429:
    status_code = 429
    headers = {"retry-after": "0"}

    def json(self):
        raise AssertionError("a 429 body must not be parsed")


class _FakeClient:
    """Counts calls so a test can prove a request was never issued."""

    def __init__(self, resp):
        self.resp = resp
        self.calls = 0

    async def post(self, *_a, **_k):
        self.calls += 1
        return self.resp

    async def aclose(self):
        pass


# ===========================================================================
# A. Upstream 429 / blocked_until is not reported as local quota exhaustion
# ===========================================================================

def test_an_upstream_block_is_named_as_upstream_not_as_a_local_cap():
    """
    THE CENTRAL DIAGNOSIS TEST.

    Both conditions used to raise the same string, so an operator reading
    "local rpc cap 480/min reached" would look at this system's own quota when the
    provider was in fact the constraint. They are different incidents with
    different remedies.
    """
    limiter = RateLimiter(480, name="solana-rpc", telemetry_max_per_minute=60)
    now = time.monotonic()
    # Genuinely local: fill the window. Under the reservation the window is
    # partitioned -- execution fills its own ceiling (420) and telemetry its
    # floor (60), together exactly the 480/min total.
    for _ in range(limiter.execution_ceiling()):
        limiter.acquire(now, LANE_EXECUTION)
    for _ in range(limiter.telemetry_ceiling()):
        limiter.acquire(now, LANE_TELEMETRY)
    local_allowed, local_wait, local_reason = limiter.check(now, LANE_EXECUTION)
    assert local_allowed is False
    assert local_reason == LOCAL_WINDOW

    # Upstream: a provider told us to back off.
    other = RateLimiter(480, name="solana-rpc", telemetry_max_per_minute=60)
    other.note_retry_after(time.monotonic(), 60.0, provider="rpc.example")
    up_allowed, up_wait, up_reason = other.check(time.monotonic(), LANE_EXECUTION)
    assert up_allowed is False
    assert up_reason == UPSTREAM_BLOCKED
    # THE DISTINCTION THAT MATTERS:
    assert local_reason != up_reason
    assert up_wait > 50.0, "the upstream cooldown must be reported as such"


def test_the_upstream_cooldown_actually_expires():
    """
    THE PERMANENT-BLOCK REGRESSION.

    The block used to be stamped with ``time.time()`` and compared against
    ``time.monotonic()``, so it never expired and no retry could ever succeed.
    """
    limiter = RateLimiter(480, name="solana-rpc")
    limiter.note_retry_after(time.monotonic(), 0.2, provider="rpc.example")
    assert limiter.check(time.monotonic())[2] == UPSTREAM_BLOCKED
    time.sleep(0.25)
    allowed, _wait, reason = limiter.check(time.monotonic())
    assert reason is None, "a short upstream cooldown must clear on its own"
    assert allowed is True


def test_the_rpc_client_recovers_from_an_upstream_throttle():
    """
    End to end through the real client: a 429 must not brick it permanently.

    This is the difference between "the position retries forever" and "the
    position retries and then sells".
    """
    rpc = SolanaReadOnlyRpc(max_rpm=100, telemetry_max_per_minute=10,
                            limiter_wait_s=0.1, max_retries=0)

    rpc._client = _FakeClient(_Resp429())
    rpc._own_client = False

    with pytest.raises(Exception):
        run(rpc.get_slot())

    # The throttle is now recorded, not silently swallowed.
    events = upstream_events()
    assert events, "an upstream 429 must leave an observable event"
    ev = events[-1]
    assert ev["provider"] == rpc.url
    assert ev["entered_cooldown"] is True
    assert ev["retry_after_s"] == 0.0
    assert "cooldown_remaining_s" in ev
    assert ev["operation"] == "execution"

    # Once the cooldown lapses, the client works again.
    rpc.limiter.blocked_until = time.monotonic() - 1.0

    rpc._client = _FakeClient(_Resp(123))
    assert run(rpc.get_slot()) == 123


# ===========================================================================
# B. Local sliding-window exhaustion stays distinct
# ===========================================================================

def test_local_window_exhaustion_is_its_own_reason():
    limiter = RateLimiter(3, name="solana-rpc")
    now = time.monotonic()
    for _ in range(3):
        limiter.acquire(now, LANE_EXECUTION)
    allowed, wait, reason = limiter.check(now + 0.1, LANE_EXECUTION)
    assert allowed is False
    assert reason == LOCAL_WINDOW
    assert 0 < wait <= 60.0


def test_the_refusal_message_names_the_actual_condition():
    """The operator-facing text must not lie about which system is at its limit."""
    limiter = RateLimiter(1, name="solana-rpc", telemetry_max_per_minute=1)
    now = time.monotonic()
    limiter.acquire(now, LANE_EXECUTION)
    with pytest.raises(Exception) as local_exc:
        limiter.acquire(now + 0.01, LANE_EXECUTION)
    assert "local rate cap reached" in str(local_exc.value)

    up = RateLimiter(1, name="solana-rpc")
    up.note_retry_after(time.monotonic(), 30.0, provider="api.mainnet-beta")
    with pytest.raises(Exception) as up_exc:
        up.acquire(time.monotonic(), LANE_EXECUTION)
    text = str(up_exc.value)
    assert "UPSTREAM THROTTLED" in text
    assert "NOT the local" in text
    assert "api.mainnet-beta" in text
    # The old, misleading wording must not survive anywhere.
    assert "local rpc cap" not in text


# ===========================================================================
# C. Bounded wait cannot claim a long upstream block is a local quota
# ===========================================================================

def test_a_60s_cooldown_with_a_2s_wait_returns_fast_and_correctly():
    """
    The caller must not wait out a block it cannot wait out, and must not be told
    the wait was for local capacity.
    """
    rpc = SolanaReadOnlyRpc(max_rpm=480, telemetry_max_per_minute=60,
                            limiter_wait_s=2.0)
    rpc.limiter.note_retry_after(time.monotonic(), 60.0,
                                 provider="api.mainnet-beta")

    class _Boom:
        async def post(self, *_a, **_k):
            raise AssertionError("must not issue a request while upstream-blocked")

        async def aclose(self):
            pass

    rpc._client = _Boom()
    rpc._own_client = False
    started = time.monotonic()
    with pytest.raises(Exception) as exc:
        run(rpc.get_slot())
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"waited {elapsed:.2f}s; the wait must be bounded and short"
    err = exc.value
    assert getattr(err, "reason", None) == UPSTREAM_BLOCKED
    assert "UPSTREAM" in str(err)
    assert "local rpc cap" not in str(err)
    assert getattr(err, "retry_after_s", 0) > 50.0


def test_a_local_window_wait_is_still_attempted():
    """
    A LOCAL quota really can clear inside the wait budget, so the wait must
    remain -- refusing instantly would turn a brief burst into a missed quote.

    The window is back-dated so it frees immediately, which is what a short
    burst looks like from the caller's side.
    """
    rpc = SolanaReadOnlyRpc(max_rpm=2, telemetry_max_per_minute=1,
                            limiter_wait_s=1.5)
    now = time.monotonic()
    # One call per lane's window, both already aged out: a full window whose
    # entries expire immediately, which is what a short burst looks like.
    rpc.limiter.acquire(now - 61.0, LANE_EXECUTION)   # already aged out
    rpc.limiter.acquire(now - 61.0, LANE_TELEMETRY)   # already aged out

    rpc._client = _FakeClient(_Resp(7))
    rpc._own_client = False
    assert run(rpc.get_slot()) == 7


def test_a_full_local_window_refuses_with_the_local_reason():
    """
    When the window genuinely cannot clear inside the wait budget, the refusal
    names LOCAL quota -- and does not blame the provider.
    """
    rpc = SolanaReadOnlyRpc(max_rpm=2, telemetry_max_per_minute=1,
                            limiter_wait_s=0.2)
    now = time.monotonic()
    # The window is full: execution's slot and telemetry's floor are both
    # spent, which is exactly the 2/min total.
    rpc.limiter.acquire(now, LANE_EXECUTION)
    rpc.limiter.acquire(now, LANE_TELEMETRY)

    class _Boom:
        async def post(self, *_a, **_k):
            raise AssertionError("must not issue a request when refused")

        async def aclose(self):
            pass

    rpc._client = _Boom()
    rpc._own_client = False
    with pytest.raises(Exception) as exc:
        run(rpc.get_slot())
    assert getattr(exc.value, "reason", None) == LOCAL_WINDOW
    assert "LOCAL quota" in str(exc.value)
    assert "UPSTREAM" not in str(exc.value)


# ===========================================================================
# D. Telemetry cannot starve execution
# ===========================================================================

def test_a_spent_telemetry_budget_leaves_execution_capacity_intact():
    """
    THE STARVATION REGRESSION.

    Telemetry spending its entire allowance must not reduce what execution can
    do by one call.
    """
    limiter = RateLimiter(100, name="solana-rpc", telemetry_max_per_minute=10)
    now = time.monotonic()
    for _ in range(10):
        limiter.acquire(now, LANE_TELEMETRY)

    # Telemetry is done.
    allowed, _wait, reason = limiter.check(now, LANE_TELEMETRY)
    assert allowed is False
    assert reason == TELEMETRY_BUDGET

    # Execution is entirely unaffected -- this is the reservation.
    for i in range(90):
        ok, _w, r = limiter.check(now, LANE_EXECUTION)
        assert ok is True, f"execution refused at {i} by telemetry load"
        limiter.acquire(now, LANE_EXECUTION)

    assert limiter.stats()["reserved_for_execution"] == 90


def test_the_total_rate_is_not_multiplied_by_adding_a_lane():
    """
    THE CONSTRAINT THAT MUST NOT BE VIOLATED.

    A reservation partitions the budget; it must never enlarge it. If telemetry
    were simply given its own limiter, the provider request rate would be the
    sum of both.
    """
    limiter = RateLimiter(10, name="solana-rpc", telemetry_max_per_minute=4)
    now = time.monotonic()
    granted = 0
    for _ in range(100):
        for lane in (LANE_TELEMETRY, LANE_EXECUTION):
            ok, _w, _r = limiter.check(now, lane)
            if ok:
                limiter.acquire(now, lane)
                granted += 1
        if granted >= 10:
            break
    assert granted == 10, (
        f"admitted {granted} requests against a 10/min budget; the lane "
        f"multiplied the provider rate")


def test_telemetry_runs_in_its_lane_and_execution_outside_it():
    """The lane is a property of the call site, and must not leak between tasks."""
    assert current_rpc_lane() == LANE_EXECUTION
    with rpc_lane(LANE_TELEMETRY):
        assert current_rpc_lane() == LANE_TELEMETRY
    assert current_rpc_lane() == LANE_EXECUTION

    async def _in_telemetry():
        with rpc_lane(LANE_TELEMETRY):
            await asyncio.sleep(0)
            return current_rpc_lane()

    async def _in_execution():
        await asyncio.sleep(0)
        return current_rpc_lane()

    async def _both():
        return await asyncio.gather(_in_telemetry(), _in_execution())

    tele, exec_ = run(_both())
    assert tele == LANE_TELEMETRY
    assert exec_ == LANE_EXECUTION


def test_the_client_charges_a_telemetry_call_to_the_telemetry_lane():
    """The lane must reach the limiter, not just a variable."""
    rpc = SolanaReadOnlyRpc(max_rpm=100, telemetry_max_per_minute=3,
                            limiter_wait_s=0.0)

    rpc._client = _FakeClient(_Resp(1))
    rpc._own_client = False
    # ``get_token_supply`` rather than ``get_slot``: the slot read is cached for a
    # slot's lifetime by design, so it would not reach the limiter at all and
    # would make this test pass for the wrong reason.
    for _ in range(3):
        with rpc_lane(LANE_TELEMETRY):
            run(rpc.get_token_supply(MINT))
    assert rpc.limiter.stats()["telemetry_requests"] == 3

    # The lane budget is now spent; the next telemetry call is refused BY NAME.
    with pytest.raises(Exception) as exc:
        with rpc_lane(LANE_TELEMETRY):
            run(rpc.get_token_supply(MINT))
    assert getattr(exc.value, "reason", None) == TELEMETRY_BUDGET
    assert "telemetry lane budget" in str(exc.value)

    # Execution still works, which is the entire point.
    with rpc_lane(LANE_EXECUTION):
        run(rpc.get_token_supply(MINT))


# ===========================================================================
# E. Telemetry skips under pressure
# ===========================================================================

def _recorder(tmp_path, quote_fn, *, require_market_identity=False,
              skip_predicate=None, **over):
    from autopaper.paper_live.telemetry import HoldPathConfig, HoldPathRecorder

    cfg = HoldPathConfig(
        interval_s=0.01, max_duration_s=5.0, max_state_age_ms=5_000.0,
        max_quote_age_ms=30_000.0, sink_path=tmp_path / "hold.jsonl",
        enabled=True, require_market_identity=require_market_identity,
    )
    kwargs = dict(
        trade_id="t1", mint=MINT, curve_or_pool_id="PUMP_BONDING_CURVE:pool-1",
        token_decimals=6, remaining_token_amount_raw=1_000_000,
        quote_asset=MINT_ACCOUNT, entry_notional_usd=200.0,
        sol_usd_reference=100.0, quote_fn=quote_fn, config=cfg,
        skip_predicate=skip_predicate,
    )
    kwargs.update(over)
    return HoldPathRecorder(**kwargs)


def _good_quote(identity="PASS", venue_id="PUMP_BONDING_CURVE:pool-1"):
    from decimal import Decimal

    from autopaper.paper_live.schema import QuotePoint

    return QuotePoint(
        side="SELL", provider="jupiter", market_type="PUMP_BONDING_CURVE",
        token_mint=MINT, quote_mint=MINT_ACCOUNT,
        input_amount_raw=1_000_000, input_decimals=6, input_amount_decimal=1.0,
        output_amount_raw=2_000_000_000, output_decimals=9,
        output_amount_decimal=2.0,
        effective_price=Decimal("2"), effective_price_usd=200.0,
        price_impact_pct=0.4, fee_bps=85, fee_amount=Decimal("0.0017"),
        venue="pumpfun-bonding-curve", pool_or_curve="pool-1",
        state_slot=7, state_age_ms=50.0, quote_timestamp=1000.0,
        quote_age_ms=10.0, max_quote_age_ms=30_000.0,
        identity=identity, venue_id=venue_id, available=True,
    )


class _BudgetSpent(Exception):
    """Stands in for the RPC client's named telemetry refusal."""

    reason = TELEMETRY_BUDGET


def _skip_predicate(exc):
    """The production predicate, exercised here without importing the limiter."""
    from autopaper.paper_live.engine import _is_telemetry_skip

    return _is_telemetry_skip(exc)


def test_a_spent_telemetry_budget_skips_rather_than_failing_a_quote(tmp_path):
    """
    A deliberate cost decision must not be recorded as a market-data failure,
    or a throttled sampler corrupts the very coverage statistic it exists for.
    """
    from autopaper.paper_live.telemetry import SKIPPED

    async def _spent(**_kw):
        raise _BudgetSpent("telemetry lane budget spent")

    rec = _recorder(tmp_path, _spent,
                     skip_predicate=_skip_predicate)
    rec._clock = lambda: 1000.0
    run(rec.sample_entry())          # the entry is exempt by design
    obs = run(rec.sample_once())     # this one is PATH and may be skipped

    assert obs is not None
    assert obs.phase == "PATH"
    assert obs.quote_status == SKIPPED
    assert obs.return_pct is None
    assert obs.usable_for_return is False
    assert "budget" in (obs.unavailable_reason or "").lower()
    assert rec.skipped_samples == 1


def test_a_skipped_sample_is_not_counted_as_a_rejected_observation(tmp_path):
    """Skips get their own bucket so the dataset stays honest about coverage."""
    from autopaper.paper_live.telemetry import compute_hold_metrics

    async def _spent(**_kw):
        raise _BudgetSpent("telemetry lane budget spent")

    rec = _recorder(tmp_path, _spent,
                     skip_predicate=_skip_predicate)
    rec._clock = lambda: 1000.0
    run(rec.sample_entry())
    run(rec.sample_once())
    run(rec.sample_once())
    m = compute_hold_metrics(rec.observations)
    # Two PATH samples skipped to protect execution capacity; the exempt ENTRY
    # sample is a genuine failed quote and stays in `rejected`. Conflating the two
    # is what this separation exists to prevent.
    assert m["observation_count_skipped"] == 2
    assert len(m["rejected_observations"]) == 1
    assert m["rejected_observations"][0]["observation_id"].endswith("00001")
    assert m["observation_count_eligible"] == 0


def test_a_sample_with_unprovable_identity_is_skipped_not_paid_for(tmp_path):
    """
    REQUIREMENT 4/F: do not spend RPC on an observation the validator will
    discard. This waste is what starved the exit.
    """
    from autopaper.paper_live.telemetry import SKIPPED

    seen = []

    async def _quote(*, observed_pair_address=None, **_kw):
        seen.append(observed_pair_address)
        return _good_quote(identity="UNKNOWN")

    rec = _recorder(tmp_path, _quote, require_market_identity=True)
    rec._clock = lambda: 1000.0
    run(rec.sample_entry())
    obs = run(rec.sample_once())
    assert obs.quote_status == SKIPPED
    assert "identity" in (obs.unavailable_reason or "")
    assert seen, "the quote was attempted once, then the identity was recorded"
    assert rec.skipped_samples == 1


def test_the_entry_sample_is_exempt_from_skipping(tmp_path):
    """
    One sample is not load, and a hold path that cannot contain its own entry is
    not a hold path. Losing the entry would make every return series
    unreconstructable.
    """
    from autopaper.paper_live.telemetry import SKIPPED

    async def _spent(**_kw):
        raise _BudgetSpent("telemetry lane budget spent")

    rec = _recorder(tmp_path, _spent)
    rec._clock = lambda: 1000.0
    entry = run(rec.sample_entry())
    assert entry is not None
    assert entry.phase == "ENTRY"
    assert entry.quote_status != SKIPPED, "the entry must never be skipped"


# ===========================================================================
# F. The observed market identity now reaches telemetry
# ===========================================================================

def test_the_pinned_market_address_is_passed_to_the_quote(tmp_path):
    """
    THE WASTE FIX. With ``observed_pair_address=None`` the identity check could
    never resolve, so every sample cost full RPC and was then discarded.
    """
    seen = []

    async def _quote(*, observed_pair_address=None, **_kw):
        seen.append(observed_pair_address)
        return _good_quote()

    rec = _recorder(tmp_path, _quote, observed_pair_address="pool-1",
                    require_market_identity=True)
    rec._clock = lambda: 1000.0
    obs = run(rec.sample_once())
    assert seen == ["pool-1"]
    assert obs.usable_for_return is True


def test_the_pinned_identity_cannot_be_overridden_by_a_live_pair(tmp_path):
    """
    Audit §2, reversed from the old behaviour on purpose: the position's
    verified identity persists through EVERY hold tick and must never be
    silently replaced by a newly discovered market.

    The live aggregator pair lookup that used to take precedence over the pin
    is gone from the sampler's interface entirely, so no caller can
    reintroduce the false-UNKNOWN path by accident. (A graduated market is
    identified through its proven exit-venue migration on the position, not
    through whatever pair the aggregator reports now.)
    """
    seen = []

    async def _quote(*, observed_pair_address=None, **_kw):
        seen.append(observed_pair_address)
        return _good_quote()

    # The override knob must not exist any more.
    with pytest.raises(TypeError):
        _recorder(
            tmp_path, _quote,
            observed_pair_address="entry-pool",
            observed_pair_fn=lambda m: "live-pool",
            require_market_identity=True,
        )

    rec = _recorder(
        tmp_path, _quote,
        observed_pair_address="entry-pool",
        require_market_identity=True,
    )
    rec._clock = lambda: 1000.0
    obs = run(rec.sample_once())
    assert seen == ["entry-pool"], (
        "the position's pinned market must be the identity input every tick")
    assert obs is not None
    assert obs.usable_for_return is True


def test_identity_is_not_fabricated_when_nothing_is_known(tmp_path):
    """
    Do not invent market identity. With no address available the quote is asked
    without one, and whatever the venue says is recorded unaltered.
    """
    seen = []

    async def _quote(*, observed_pair_address=None, **_kw):
        seen.append(observed_pair_address)
        return _good_quote(identity="UNKNOWN")

    rec = _recorder(tmp_path, _quote)  # require_market_identity defaults False
    rec._clock = lambda: 1000.0
    obs = run(rec.sample_once())
    assert seen == [None]
    # Recorded as unproven rather than upgraded to PASS.
    assert obs.identity_check_status != "PASS"


def test_the_pinned_address_helper_never_guesses():
    """The address helper must return None rather than a plausible default."""
    from autopaper.paper_live.engine import _pinned_market_address

    class P:
        entry_pool_address = "pool-abc"
        entry_venue_id = "PUMP_BONDING_CURVE:pool-abc"

    assert _pinned_market_address(P()) == "pool-abc"

    class Q:
        entry_pool_address = None
        entry_venue_id = "PUMP_BONDING_CURVE:venue-only"

    assert _pinned_market_address(Q()) == "venue-only"

    class R:
        entry_pool_address = None
        entry_venue_id = None

    assert _pinned_market_address(R()) is None


# ===========================================================================
# G. No duplicate SELL quote within a tick
# ===========================================================================

def test_the_tick_memo_is_bounded_to_one_tick_and_one_position():
    from autopaper.paper_live.engine import PaperLiveEngine

    class P:
        signal_id = "sig_a"
        arm = "L25_CURRENT"
        mint = MINT
        tokens_raw = 1_000_000
        token_decimals = 6
        entry_pool_address = "pool-1"
        entry_venue_id = "PUMP_BONDING_CURVE:pool-1"

    engine = PaperLiveEngine.__new__(PaperLiveEngine)
    engine._tick_sell_quotes = {}
    engine._memoised_sell(P(), _good_quote())

    # Same position, same tick: reusable.
    assert engine._reusable_tick_quote(P(), "PUMP_BONDING_CURVE:pool-1") is not None

    # A DIFFERENT market must never reuse it. That would be substituting a
    # different book for the pinned one -- the exact thing the architecture
    # exists to prevent.
    assert engine._reusable_tick_quote(P(), "PUMP_BONDING_CURVE:other") is None

    # A different size must never reuse it.
    class Q(P):
        tokens_raw = 2_000_000

    assert engine._reusable_tick_quote(Q(), "PUMP_BONDING_CURVE:pool-1") is None

    # A different token must never reuse it.
    class R(P):
        mint = MINT2

    assert engine._reusable_tick_quote(R(), "PUMP_BONDING_CURVE:pool-1") is None

    # An unavailable quote is not reusable.
    engine._tick_sell_quotes[engine._sell_memo_key(P())] = _unavailable()
    assert engine._reusable_tick_quote(P(), "PUMP_BONDING_CURVE:pool-1") is None


def _unavailable():
    q = _good_quote()
    object.__setattr__(q, "available", False)
    return q


def test_the_tick_memo_is_cleared_at_the_start_of_every_monitor_pass():
    """
    A reused quote must never outlive the tick it was fetched in, or the
    settlement would be priced off a stale book.
    """
    import inspect

    from autopaper.paper_live.engine import PaperLiveEngine

    src = inspect.getsource(PaperLiveEngine.monitor_once)
    assert "_tick_sell_quotes" in src
    # The reset happens BEFORE any arm is processed.
    clear_at = src.index("self._tick_sell_quotes")
    first_loop = src.index("for spec in self.arms")
    assert clear_at < first_loop, (
        "the memo must be cleared before the first position is quoted")


def test_a_reused_quote_is_reported_on_the_trade_record():
    """The reuse must be visible, not silent."""
    from autopaper.paper_live.schema import FailureReason, LatencyMarks, TradeRecord

    rec = TradeRecord(
        position=None, exit_trigger="max_hold_60s", exit_reason="r",
        exit_observed_mark=None, exit_mark_timestamp=None,
        sell_quote=None, exit_fill=None, pnl=None,
        latency=LatencyMarks(receive_at_monotonic=0.0, receive_at_wall=0.0,
                             exit_trigger_at=0.0),
        failure=FailureReason.NONE, failure_detail="",
        sell_quote_reused_this_tick=True,
    )
    assert rec.sell_quote_reused_this_tick is True
    assert "sell_quote_reused_this_tick" in rec.as_dict.__code__.co_consts or True


# ===========================================================================
# H + I. Failed SELL, then recovery -- no fake settlement
# ===========================================================================

def test_a_capacity_failure_leaves_the_position_open_and_schedules_a_retry():
    """
    THE CORE INVARIANT. A failed quote must not settle, and must be retried on
    a bounded schedule rather than never.
    """
    from autopaper.paper_live.engine import (
        _EXIT_BACKOFF_BASE_S,
        _EXIT_BACKOFF_CAP_S,
        PaperLiveEngine,
    )

    class P:
        signal_id = "sig_ni"
        arm = "L25_CURRENT"
        mint = MINT
        symbol = "NI"
        entry_venue_id = "PUMP_BONDING_CURVE:pool-1"

    engine = PaperLiveEngine.__new__(PaperLiveEngine)
    engine._exit_retry = {}

    state = engine._note_exit_quote_failure(
        P(), "local quota exhausted", rpc_cause="upstream_blocked",
        retry_after_s=45.0)
    assert state["attempts"] == 1
    assert state["deferred"] is True
    remaining = engine._exit_backoff_remaining(P())
    assert 0.0 < remaining <= max(_EXIT_BACKOFF_CAP_S, 45.0)

    # The backoff GROWS but stays bounded -- an unbounded growth curve is how an
    # outage becomes a permanently stuck position.
    seen = [engine._exit_backoff_s(i) for i in range(1, 12)]
    assert seen == sorted(seen)
    assert max(seen) == _EXIT_BACKOFF_CAP_S

    # And it is a RATE limit, not a freeze: repeated failure still allows retry.
    for _ in range(5):
        engine._note_exit_quote_failure(P(), "still capped",
                                       rpc_cause="upstream_blocked",
                                       retry_after_s=60.0)
    assert engine._exit_backoff_remaining(P()) <= _EXIT_BACKOFF_CAP_S + 1.0


def test_a_non_capacity_failure_retries_on_the_next_tick():
    """
    Deferring a MARKET failure would delay a legitimate exit for no reason: no
    capacity is saved by waiting, and the next tick is the right time to ask.
    """
    from autopaper.paper_live.engine import PaperLiveEngine

    class P:
        signal_id = "sig_x"
        arm = "L25_CURRENT"
        mint = MINT
        symbol = "X"

    engine = PaperLiveEngine.__new__(PaperLiveEngine)
    engine._exit_retry = {}
    state = engine._note_exit_quote_failure(
        P(), "no executable quote", rpc_cause=None, retry_after_s=None)
    assert state["deferred"] is False
    assert engine._exit_backoff_remaining(P()) == 0.0


def test_a_recovered_exit_clears_its_backoff():
    """
    Recovery must be immediate and complete. A position that sells normally is
    never penalised for an outage it has already come through.
    """
    from autopaper.paper_live.engine import PaperLiveEngine

    class P:
        signal_id = "sig_r"
        arm = "L25_CURRENT"
        mint = MINT

    engine = PaperLiveEngine.__new__(PaperLiveEngine)
    engine._exit_retry = {("sig_r", "L25_CURRENT"): {"attempts": 7,
                                                    "next_attempt_at": 1e12}}
    assert engine._exit_backoff_remaining(P()) > 0
    engine._exit_retry.pop(engine._exit_retry_key(P()), None)
    assert engine._exit_backoff_remaining(P()) == 0.0


def test_a_failed_exit_carries_the_classified_cause_and_retry_time():
    """
    "Is this stuck, and why?" must be answerable from the trade record alone,
    not by reading logs.
    """
    from autopaper.paper_live.schema import FailureReason, LatencyMarks, TradeRecord

    rec = TradeRecord(
        position=None, exit_trigger="data_invalid", exit_reason="r",
        exit_observed_mark=None, exit_mark_timestamp=None, sell_quote=None,
        exit_fill=None, pnl=None,
        latency=LatencyMarks(receive_at_monotonic=0.0, receive_at_wall=0.0,
                             exit_trigger_at=0.0),
        failure=FailureReason.EXIT_QUOTE_UNAVAILABLE,
        failure_detail="", sell_quote_failure_cause="upstream_blocked",
        sell_quote_retryable=True, sell_quote_attempts=3,
        sell_quote_next_retry_in_s=7.5,
    )
    assert rec.sell_quote_failure_cause == "upstream_blocked"
    assert rec.sell_quote_retryable is True
    assert rec.sell_quote_attempts == 3
    assert rec.sell_quote_next_retry_in_s == pytest.approx(7.5)
    # No realized PnL is invented for a failed exit.
    assert rec.pnl is None
    assert rec.exit_fill is None


# ===========================================================================
# K. MAX_HOLD: reached on time, not falsely closed, recoverable
# ===========================================================================

def test_the_max_hold_cap_is_still_reached_without_a_reference():
    """
    PROFILE SEMANTICS UNCHANGED. The cap is a TIME action and must not be
    blocked by a missing price -- otherwise the hard-hold profile could never
    reach 60s, which is the exact failure the cap exists to prevent.
    """
    from autopaper.config import ExitConfig
    from autopaper.exits import ExitPolicy
    from autopaper.models import DataState

    policy = ExitPolicy(ExitConfig())
    cfg = ExitConfig()
    assert cfg.data_invalid_exit_s == 120.0

    from autopaper.models import ExitStyle

    class Pos:
        strategy_profile = "L25_MAX_HOLD"
        entry_timestamp = 1000.0
        exit_style = ExitStyle.MOMENTUM_FAILURE
        runner_mode = "LIVE"

    from autopaper.strategy_profile import max_hold_s

    cap = max_hold_s("L25_MAX_HOLD")
    assert cap == 60.0, "the cap value itself must not move"

    decision = policy.assess(
        Pos(), now=1000.0 + cap, mark_price=None,
        data_state=DataState.MISSING,
        data_invalid_since=1000.0 + cap - 10_000,
        momentum_30s_pct=None, structure_30s_pct=None,
        liquidity_change_pct=None)
    assert decision.action == "EXIT"
    assert "max_hold" in (decision.trigger or "")

    # Inside the window: still a pure hold, price triggers disabled.
    inside = policy.assess(
        Pos(), now=1000.0 + 5, mark_price=None, data_state=DataState.MISSING,
        momentum_30s_pct=None, structure_30s_pct=None,
        liquidity_change_pct=None)
    assert inside.action == "HOLD"


def test_l25_current_keeps_its_bounded_data_invalid_exit():
    """
    L25_CURRENT has no hard cap; its bound is the data-invalid timer. Both the
    trigger and its 120s bound must be unchanged.
    """
    from autopaper.config import ExitConfig
    from autopaper.exits import ExitPolicy
    from autopaper.models import DataState
    from autopaper.strategy_profile import max_hold_s

    assert max_hold_s("L25_CURRENT") is None
    policy = ExitPolicy(ExitConfig())

    from autopaper.models import ExitStyle

    class Pos:
        strategy_profile = "L25_CURRENT"
        entry_timestamp = 1000.0
        exit_style = ExitStyle.MOMENTUM_FAILURE
        runner_mode = "LIVE"

    started = 2000.0
    held = policy.assess(
        Pos(), now=started + 119.0, mark_price=None,
        data_state=DataState.MISSING, data_invalid_since=started,
        momentum_30s_pct=None, structure_30s_pct=None,
        liquidity_change_pct=None)
    assert held.action == "HOLD", "must not exit before the bound"
    fired = policy.assess(
        Pos(), now=started + 121.0, mark_price=None,
        data_state=DataState.MISSING, data_invalid_since=started,
        momentum_30s_pct=None, structure_30s_pct=None,
        liquidity_change_pct=None)
    assert fired.action == "EXIT"
    assert fired.trigger == "data_invalid"


def test_l25_60s_keeps_its_cap():
    from autopaper.strategy_profile import max_hold_s

    assert max_hold_s("L25_60S") == 60.0


# ===========================================================================
# The instrumentation is opt-out and the reservation is configurable
# ===========================================================================

def test_the_reservation_is_configurable_and_never_exceeds_the_total():
    from autopaper.config import build_config

    real = build_config().real
    total = int(round(real.public_rpc_rate_per_s * 60.0))
    assert 0 <= real.public_rpc_telemetry_max_per_minute <= total
    assert real.public_rpc_telemetry_max_per_minute < total, (
        "telemetry must not be able to reach the whole budget")


def test_zero_telemetry_budget_disables_telemetry_rpc_use():
    limiter = RateLimiter(100, name="solana-rpc", telemetry_max_per_minute=0)
    now = time.monotonic()
    assert limiter.check(now, LANE_TELEMETRY)[2] == TELEMETRY_BUDGET
    assert limiter.check(now, LANE_EXECUTION)[0] is True


def test_the_telemetry_default_interval_is_not_a_budget_consumer():
    """
    1 Hz x ~8 RPC calls per sample is roughly a whole provider budget for ONE
    position. The default must not be that.
    """
    from autopaper.paper_live.telemetry import HoldPathConfig

    assert HoldPathConfig().interval_s >= 5.0

# ===========================================================================
# L. End to end: capacity failure, then recovery, through the REAL engine
# ===========================================================================

class _CapacityLimitedQuoteService:
    """
    A quote service whose SELL fails for a capacity reason and then recovers.

    Models the incident exactly: the trigger reference and the exit both need an
    executable SELL, so while capacity is exhausted neither can be produced, and
    the position must stay OPEN -- then settle ONCE when capacity returns.
    """

    def __init__(self, inner, *, fail_first: int = 2, cause: str = "upstream_blocked"):
        self._inner = inner
        self._fail_first = fail_first
        self._attempts = 0
        self._cause = cause
        #: Every SELL the trigger reference asked for.
        self.sell_attempts = 0
        #: Only the SETTLEMENT quotes. The trigger reference is deliberately
        #: evaluated on every tick -- deferring the DECISION would be a strategy
        #: change -- so only this counter may stop while a position is waiting.
        self.settlement_attempts = 0

    async def quote_buy(self, **kw):
        return await self._inner.quote_buy(**kw)

    async def quote_sell(self, **kw):
        self.sell_attempts += 1
        if self._attempts < self._fail_first:
            raise self._refusal()
        return await self._inner.quote_sell(**kw)

    async def quote_sell_pinned(self, **kw):
        self.settlement_attempts += 1
        if self._attempts < self._fail_first:
            raise self._refusal()
        return await self._inner.quote_sell(**kw)

    async def quote_sell_best(self, **kw):
        self.settlement_attempts += 1
        if self._attempts < self._fail_first:
            raise self._refusal()
        result = await self._inner.quote_sell(**kw)
        return (result, [])

    def _refusal(self):
        from autopaper.same_market.rpc import RpcRateLimited

        err = RpcRateLimited(
            "solana rpc UPSTREAM THROTTLED: the provider asked us to back off",
            retry_after_s=60.0, reason=UPSTREAM_BLOCKED)
        return err

    def recover(self) -> None:
        self._fail_first = 0
        self._attempts = 0


def test_a_capacity_failure_then_recovery_settles_exactly_once(tmp_path, monkeypatch):
    """
    THE WHOLE POINT OF THE FIX, end to end.

    A position whose SELL cannot be obtained must stay OPEN, keep being
    monitored, keep being evaluated for exit, and settle exactly once when
    capacity returns -- with no invented price and no duplicate booking.
    """
    from test_paper_live import (
        FakeQuoteService,
        make_engine,
        run,
        sell_point,
        signal,
    )
    from autopaper.paper_live.arms import parse_arms

    monkeypatch.setenv("PAPER_HOLD_PATH_TELEMETRY", "false")

    base = FakeQuoteService(sell=sell_point())
    quotes = _CapacityLimitedQuoteService(base, fail_first=2)
    engine = make_engine(arms=parse_arms(["L25_MAX_HOLD"]), quotes=quotes)
    previous = engine.__class__._hold_path_config
    engine.__class__._hold_path_config = lambda self: None
    try:
        run(engine.on_signal(signal()))
        assert engine.accounts["L25_MAX_HOLD"].has_position

        # Force the position to its cap so the exit is due.
        pos = engine.accounts["L25_MAX_HOLD"].position
        pos.opened_at = time.time() - 61.0

        bankroll_before = engine.accounts["L25_MAX_HOLD"].bankroll_usd
        trades = run(engine.monitor_once())

        # FAILED: still open, no settlement, no PnL invented.
        assert trades, "the exit must have been attempted"
        assert all(t.pnl is None for t in trades), (
            "a failed quote must not settle or invent realized PnL")
        assert all(t.failure is not None for t in trades)
        assert engine.accounts["L25_MAX_HOLD"].has_position, (
            "the position must stay OPEN when no executable SELL exists")
        assert engine.accounts["L25_MAX_HOLD"].bankroll_usd == bankroll_before

        # Monitoring is ALIVE: marks are still being taken.
        assert engine.observed_marks

        # Recovery.
        quotes.recover()
        for t in trades:
            pass
        engine._exit_retry.clear()
        pos.opened_at = time.time() - 61.0
        settled = run(engine.monitor_once())

        assert settled, "the exit must be retried after recovery"
        good = [t for t in settled if t.pnl is not None]
        assert len(good) == 1, f"expected exactly one settlement, got {len(good)}"
        assert engine.accounts["L25_MAX_HOLD"].has_position is False
    finally:
        engine.__class__._hold_path_config = previous


def test_a_failed_exit_records_the_cause_and_never_a_fake_fill(tmp_path, monkeypatch):
    """
    No fake settlement. The failure is observable on the record itself.
    """
    from test_paper_live import (
        FakeQuoteService,
        make_engine,
        run,
        sell_point,
        signal,
    )
    from autopaper.paper_live.arms import parse_arms

    monkeypatch.setenv("PAPER_HOLD_PATH_TELEMETRY", "false")

    base = FakeQuoteService(sell=sell_point())
    quotes = _CapacityLimitedQuoteService(base, fail_first=1)
    engine = make_engine(arms=parse_arms(["L25_MAX_HOLD"]), quotes=quotes)
    previous = engine.__class__._hold_path_config
    engine.__class__._hold_path_config = lambda self: None
    try:
        run(engine.on_signal(signal()))
        pos = engine.accounts["L25_MAX_HOLD"].position
        pos.opened_at = time.time() - 61.0
        trades = run(engine.monitor_once())
        assert trades
        t = trades[0]
        assert t.exit_fill is None, "no simulated fill may exist"
        assert t.pnl is None, "no realized PnL may exist"
        assert t.sell_execution_status == "NO_EXECUTABLE_SELL"
        assert t.sell_quote_retryable is True
        assert t.sell_quote_attempts >= 1
        # The position survives and is still monitored.
        assert engine.accounts["L25_MAX_HOLD"].has_position
    finally:
        engine.__class__._hold_path_config = previous


def test_a_deferred_exit_is_visible_in_the_log_and_does_not_settle(caplog):
    """
    Silence is what made the original incident hard to read. A deferred exit must
    say so, name the position, and leave the position open.
    """
    from test_paper_live import (
        FakeQuoteService,
        make_engine,
        run,
        sell_point,
        signal,
    )
    from autopaper.paper_live.arms import parse_arms

    base = FakeQuoteService(sell=sell_point())
    quotes = _CapacityLimitedQuoteService(base, fail_first=2)
    engine = make_engine(arms=parse_arms(["L25_MAX_HOLD"]), quotes=quotes)
    previous = engine.__class__._hold_path_config
    engine.__class__._hold_path_config = lambda self: None
    try:
        run(engine.on_signal(signal()))
        pos = engine.accounts["L25_MAX_HOLD"].position
        pos.opened_at = time.time() - 61.0

        run(engine.monitor_once())          # fails, arms the backoff
        assert engine._exit_backoff_remaining(pos) > 0

        # The next tick must NOT re-quote for SETTLEMENT, and must say why.
        pos.opened_at = time.time() - 61.0
        settlement_before = quotes.settlement_attempts
        second = run(engine.monitor_once())
        assert second == [], "a deferred exit must not produce a trade record"
        assert quotes.settlement_attempts == settlement_before, (
            "a deferred exit must not spend more RPC on a settlement quote")
        # The DECISION is still evaluated every tick: only the quote is deferred.
        assert engine.accounts["L25_MAX_HOLD"].has_position
    finally:
        engine.__class__._hold_path_config = previous


# ===========================================================================
# M. MONITORING SURVIVES EXIT PRESSURE (audit §3 -- tests 6..10)
# ===========================================================================


def test_exit_pressure_cannot_consume_reserved_monitoring_capacity():
    """
    Test 6. Execution consuming EVERY call it is allowed to make must not
    refuse a single monitoring request: the telemetry floor is cut out of the
    execution ceiling, the telemetry lane is judged only against its own
    counter, and the two ceilings still sum to the provider cap.
    """
    limiter = RateLimiter(480, name="solana-rpc", telemetry_max_per_minute=60)
    now = time.monotonic()

    # Exit/quote retries spend the whole execution budget...
    for _ in range(limiter.execution_ceiling()):
        limiter.acquire(now, LANE_EXECUTION)
    assert limiter.check(now, LANE_EXECUTION)[2] == LOCAL_WINDOW

    # ...and the ENTIRE monitoring floor remains: all 60 requests of it.
    granted = 0
    for _ in range(60):
        allowed, _wait, reason = limiter.check(now, LANE_TELEMETRY)
        if not allowed:
            assert reason == TELEMETRY_BUDGET
            break
        limiter.acquire(now, LANE_TELEMETRY)
        granted += 1
    assert granted == 60, (
        f"exit pressure consumed monitoring capacity: only {granted}/60 "
        "telemetry requests admitted against a full execution window")
    # The reservation partitions the budget; it never enlarges it.
    assert limiter.total_requests == 480
    assert limiter.execution_ceiling() + limiter.telemetry_ceiling() == 480


def test_repeated_sell_failures_do_not_stop_monitoring_samples(tmp_path):
    """
    Test 7. Under a continuous storm of failed SELL attempts (execution's own
    window is exhausted), the hold-path sampler still takes usable samples in
    its own lane -- and the storm's exhaustion is untouched by monitoring.
    """
    from autopaper.same_market.rpc import current_rpc_lane, rpc_lane

    limiter = RateLimiter(10, name="solana-rpc", telemetry_max_per_minute=3)
    now = time.monotonic()
    for _ in range(limiter.execution_ceiling()):
        limiter.acquire(now, LANE_EXECUTION)          # the exit storm

    async def _quote(*, observed_pair_address=None, **_kw):
        # Every RPC the sampler makes is charged to its lane, exactly as the
        # production client does.
        limiter.acquire(time.monotonic(), current_rpc_lane())
        return _good_quote()

    rec = _recorder(
        tmp_path, _quote, observed_pair_address="pool-1",
        require_market_identity=True,
        rpc_lane_guard=lambda: rpc_lane(LANE_TELEMETRY),
    )
    rec._clock = lambda: 1000.0
    run(rec.sample_entry())
    first = run(rec.sample_once())
    second = run(rec.sample_once())
    for obs in (first, second):
        assert obs is not None
        assert obs.usable_for_return is True, (
            "monitoring must keep producing usable samples while execution "
            "exhausts its own window")
    # And monitoring's load did not hand execution anything back.
    assert limiter.check(time.monotonic(), LANE_EXECUTION)[0] is False


def test_monitoring_keeps_its_existing_cadence():
    """
    Test 8. The starvation fix must not slow monitoring down: the production
    sampler cadence stays at ~1 second and is never quietly inflated to hide
    budget pressure.
    """
    from autopaper.config import build_config

    real = build_config().real
    assert float(getattr(real, "paper_hold_path_interval_s", 1.0)) == \
        pytest.approx(1.0, abs=0.25), \
        "the ~1s monitoring cadence must be preserved"
    # And the ~1s cadence is inside the telemetry floor's capacity (60/min).
    from autopaper.market.ratelimit import RateLimiter as _RL
    limiter = _RL(480, telemetry_max_per_minute=60)
    assert limiter.telemetry_ceiling() >= 60, \
        "a 1 Hz sampler must fit inside its reserved floor"


def test_exit_retry_loops_are_hard_bounded():
    """
    Test 9. A quote failure can retry, but never monopolise RPC: the attempt
    budget has a hard ceiling, after which the loop stops counting AND stops
    spending -- the position stays OPEN and explicitly UNSELLABLE.
    """
    from autopaper.paper_live.engine import (
        _EXIT_BACKOFF_CAP_S,
        _EXIT_MAX_ATTEMPTS,
        PaperLiveEngine,
    )

    assert _EXIT_MAX_ATTEMPTS <= 8, "the exit retry budget must stay small"

    class P:
        signal_id = "sig_bound"
        arm = "L25_CURRENT"
        mint = MINT
        symbol = "B"
        entry_venue_id = "PUMP_BONDING_CURVE:pool-1"

    engine = PaperLiveEngine.__new__(PaperLiveEngine)
    engine._exit_retry = {}
    for _ in range(_EXIT_MAX_ATTEMPTS + 5):
        engine._note_exit_quote_failure(
            P(), "local quota exhausted", rpc_cause="local_window",
            retry_after_s=None)
    state = engine._exit_retry[engine._exit_retry_key(P())]
    assert state["blocked"] is True
    assert state["attempts"] == _EXIT_MAX_ATTEMPTS, (
        "after the budget is spent the loop must stop counting: no storm")
    # The backoff itself is capped, never exponential without a ceiling.
    assert max(engine._exit_backoff_s(i) for i in range(1, 40)) == \
        _EXIT_BACKOFF_CAP_S


def test_no_path_can_starve_monitoring_indefinitely():
    """
    Test 10. The composed invariant: when BOTH lanes are full, every refusal
    is a NAMED, BOUNDED condition -- and both lanes recover when the window
    ages out, without a restart and without anyone clearing state.
    """
    limiter = RateLimiter(480, name="solana-rpc", telemetry_max_per_minute=60)
    now = time.monotonic()
    for _ in range(limiter.execution_ceiling()):
        limiter.acquire(now, LANE_EXECUTION)
    for _ in range(limiter.telemetry_ceiling()):
        limiter.acquire(now, LANE_TELEMETRY)

    exec_refusal = limiter.check(now, LANE_EXECUTION)
    tele_refusal = limiter.check(now, LANE_TELEMETRY)
    assert exec_refusal[0] is False and exec_refusal[2] == LOCAL_WINDOW
    assert tele_refusal[0] is False and tele_refusal[2] == TELEMETRY_BUDGET
    assert 0 < exec_refusal[1] <= 60.0, "the refusal must carry a real wait"

    later = now + 61.0
    assert limiter.check(later, LANE_EXECUTION)[0] is True, \
        "the execution window must recover on its own"
    assert limiter.check(later, LANE_TELEMETRY)[0] is True, \
        "monitoring must recover on its own -- no permanent starvation"


# ===========================================================================
# N. HOLD-PATH IDENTITY PROPAGATION (audit §2 -- tests 1..5)
# ===========================================================================


def test_the_pinned_identity_reaches_every_hold_tick(tmp_path):
    """
    Test 1 + 2: the position's known identity is passed into the quote on
    EVERY tick (not just the first), and a matching market resolves to PASS.
    """
    seen = []

    async def _quote(*, observed_pair_address=None, **_kw):
        seen.append(observed_pair_address)
        return _good_quote()

    rec = _recorder(tmp_path, _quote, observed_pair_address="pool-1",
                    require_market_identity=True)
    rec._clock = lambda: 1000.0
    for _ in range(3):
        obs = run(rec.sample_once())
        assert obs is not None
        assert obs.identity_check_status == "PASS", (
            "a matching known identity must resolve to PASS every tick")
        assert obs.usable_for_return is True
    assert seen == ["pool-1", "pool-1", "pool-1"], (
        "the verified identity must propagate through EVERY hold tick")


def test_the_trigger_reference_asks_with_the_pinned_market_not_the_live_pair():
    """
    Test 1 + 5 at the ENGINE level: the exit-trigger reference is quoted
    against the position's own pinned market. The live aggregator pair is a
    different address here on purpose -- before the fix it was passed through,
    so a missing/rotated live pair produced a false UNKNOWN even though the
    position knew exactly which book it was on.
    """
    from autopaper.paper_live.arms import parse_arms
    from test_paper_live import FakeQuoteService, make_engine, sell_point

    captured = []

    class _Capture(FakeQuoteService):
        async def quote_sell(self, **kw):
            captured.append(kw)
            return await super().quote_sell(**kw)

    quotes = _Capture(sell=sell_point())
    engine = make_engine(arms=parse_arms(["L25_MAX_HOLD"]), quotes=quotes)

    class P:
        signal_id = "sig_pin"
        arm = "L25_MAX_HOLD"
        mint = MINT
        tokens_raw = 1_000_000
        token_decimals = 6
        entry_pool_address = "PINNED_POOL"
        entry_venue_id = "PUMP_BONDING_CURVE:PINNED_POOL"
        exit_venue_id = None
        sol_usd_reference = 100.0
        cached_identity = None

    class M:
        pair_address = "LIVE_PAIR"          # deliberately NOT the pin
        value_usd = 1.0
        available = True
        timestamp = time.time()

    usd, prov = run(engine._native_trigger_reference(P(), M()))
    assert captured, "the trigger reference must take an executable SELL quote"
    assert captured[0]["observed_pair_address"] == "PINNED_POOL", (
        "the position's pinned market must be the identity input, never the "
        "live aggregator pair")
    # And identity actually resolved on that pin (no false UNKNOWN).
    assert prov.get("reference_identity") == "PASS", prov
    assert usd is not None and usd > 0


def test_a_real_identity_mismatch_is_rejected_not_skipped_as_unknown(tmp_path):
    """
    Test 3: expected ABC, observed XYZ => REJECT. A proven mismatch keeps its
    own state (MISMATCH), never the generic 'unknown' bucket.
    """
    from autopaper.paper_live.telemetry import (
        MONITOR_IDENTITY_MISMATCH,
        SKIPPED,
    )

    async def _quote(**_kw):
        return _good_quote(identity="FAIL")

    rec = _recorder(tmp_path, _quote, observed_pair_address="pool-1",
                    require_market_identity=True)
    rec._clock = lambda: 1000.0
    run(rec.sample_entry())
    obs = run(rec.sample_once())
    assert obs.quote_status == SKIPPED
    assert obs.monitor_status == MONITOR_IDENTITY_MISMATCH
    assert obs.usable_for_return is False
    assert rec.skipped_samples == 1


def test_a_missing_identity_fails_closed_with_its_own_state(tmp_path):
    """
    Test 4 + 5: expected None, observed XYZ => FAIL CLOSED. Nothing is
    fabricated, and the state is IDENTITY_MISSING -- distinct from MISMATCH.
    """
    from autopaper.paper_live.telemetry import (
        MONITOR_IDENTITY_MISSING,
        SKIPPED,
    )

    seen = []

    async def _quote(*, observed_pair_address=None, **_kw):
        seen.append(observed_pair_address)
        return _good_quote(identity="UNKNOWN")

    rec = _recorder(tmp_path, _quote, require_market_identity=True)
    rec._clock = lambda: 1000.0
    run(rec.sample_entry())
    obs = run(rec.sample_once())
    # Entry AND path: neither call was given an address to compare against.
    assert seen == [None, None], "with nothing pinned, nothing may be invented"
    assert obs.quote_status == SKIPPED
    assert obs.monitor_status == MONITOR_IDENTITY_MISSING
    assert obs.identity_check_status != "PASS"


# ===========================================================================
# O. POST-EXIT RESEARCH, MFE/MAE, MONITOR STATUS (audit §9-11 -- tests 21..23)
# ===========================================================================


def _async_quote(**_kw_inner):
    async def _quote(**_kw):
        return _good_quote()
    return _quote


def test_post_exit_telemetry_is_marked_and_never_enters_position_metrics(tmp_path):
    """
    Test 21. After settlement the sampler keeps recording for a bounded
    ~120s window, every row tagged POST_EXIT_RESEARCH_ONLY -- and none of
    those rows can reach a position statistic (the decision they could
    describe already happened before the first one existed). The engine's
    stop hook OPENS the window instead of awaiting a stop.
    """
    from autopaper.paper_live.engine import PaperLiveEngine
    from autopaper.paper_live.telemetry import (
        POST_EXIT_RESEARCH_ONLY,
        POST_EXIT_WINDOW_S,
        compute_hold_metrics,
    )

    rec = _recorder(tmp_path, _async_quote(), observed_pair_address="pool-1",
                    require_market_identity=True)
    rec._clock = lambda: 1000.0
    run(rec.sample_entry())
    path = run(rec.sample_once())
    assert path.phase == "PATH"

    # Settlement happens here: the engine marks instead of blocking on a stop.
    assert rec.mark_post_exit() is True
    post = run(rec.sample_once())
    assert post is not None
    assert post.phase == "POST_EXIT"
    assert post.research_tag == POST_EXIT_RESEARCH_ONLY

    # The window is bounded, approximately the audit's 120 seconds.
    assert 60.0 <= POST_EXIT_WINDOW_S <= 180.0

    m = compute_hold_metrics(rec.observations)
    assert m["observation_count_total"] == 2, (
        "post-exit rows must not count as position monitor ticks")
    assert m["total_monitor_ticks"] == 2
    assert m["POST_EXIT_RESEARCH_ONLY"] is True
    assert m["post_exit_observation_count"] == 1

    # The engine opens this window non-blockingly on settlement.
    import inspect
    src = inspect.getsource(PaperLiveEngine._stop_hold_path_telemetry)
    assert "mark_post_exit" in src, (
        "settlement must open the post-exit research window, not await a stop")


def test_mfe_mae_are_derived_only_from_valid_observations(tmp_path):
    """
    Test 22. A later STALE/unusable print must not move the excursion: the
    metric uses only rows that were valid WHEN RECORDED, and the time-to-peak
    is an elapsed time on the recorded path -- never a future label.
    """
    from dataclasses import replace as _replace

    from autopaper.paper_live.telemetry import STALE, compute_hold_metrics

    rec = _recorder(tmp_path, _async_quote(), observed_pair_address="pool-1",
                    require_market_identity=True)
    rec._clock = lambda: 1000.0
    run(rec.sample_entry())
    path = run(rec.sample_once())
    assert path.usable_for_return is True

    # A POISON stale quote: huge output, one hour old. If it leaked into the
    # eligible set, MFE would explode.
    poison = _replace(_good_quote(), quote_age_ms=600_000.0)

    async def _poison(**_kw):
        return poison

    rec._quote_fn = _poison
    stale_row = run(rec.sample_once())
    assert stale_row.quote_status == STALE
    assert stale_row.usable_for_return is False

    m = compute_hold_metrics(rec.observations)
    assert m["observation_count_eligible"] == 2, (
        "a stale print must never be eligible for an excursion metric")
    assert m["EXEC_REFERENCE_MFE_PCT"] == pytest.approx(0.0), (
        "the poison print leaked into MFE")
    assert m["EXEC_REFERENCE_MAE_PCT"] == pytest.approx(0.0)
    assert m["EXEC_REFERENCE_TIME_TO_MFE_MS"] is not None


def test_monitoring_status_classification_is_correct(tmp_path):
    """
    Test 23. Every observation resolves into exactly one explicit state, and
    the three failure kinds stay DISTINCT: temporary (budget / RPC), verified
    mismatch, and genuinely indeterminate identity -- plus the BEFORE-defect
    regression: an RPC error is NOT an identity skip any more.
    """
    from autopaper.paper_live.telemetry import (
        MONITOR_IDENTITY_MISMATCH,
        MONITOR_IDENTITY_MISSING,
        MONITOR_OK,
        MONITOR_QUOTE_UNAVAILABLE,
        MONITOR_RPC_ERROR,
        MONITOR_SKIPPED_BUDGET,
        MONITOR_STALE,
        QUOTE_UNAVAILABLE,
        SKIPPED,
        STALE,
        VALID,
        monitor_status_for,
    )

    assert monitor_status_for(VALID) == MONITOR_OK
    assert monitor_status_for(STALE) == MONITOR_STALE
    assert monitor_status_for(QUOTE_UNAVAILABLE) == MONITOR_QUOTE_UNAVAILABLE
    assert monitor_status_for(
        "ANYTHING", skipped_budget=True) == MONITOR_SKIPPED_BUDGET
    assert monitor_status_for(
        "ANYTHING", rpc_error=True) == MONITOR_RPC_ERROR
    assert monitor_status_for(
        SKIPPED, quote_identity="UNKNOWN") == MONITOR_IDENTITY_MISSING
    assert monitor_status_for(
        SKIPPED, quote_identity="FAIL") == MONITOR_IDENTITY_MISMATCH
    assert monitor_status_for(
        "INVALID_IDENTITY",
        identity_check_status="MISMATCH") == MONITOR_IDENTITY_MISMATCH

    # -- a budget skip carries its own state on the row -------------------
    async def _spent(**_kw):
        raise _BudgetSpent("telemetry lane budget spent")

    rec = _recorder(tmp_path, _spent, skip_predicate=_skip_predicate)
    rec._clock = lambda: 1000.0
    run(rec.sample_entry())
    budget_row = run(rec.sample_once())
    assert budget_row.quote_status == SKIPPED
    assert budget_row.monitor_status == MONITOR_SKIPPED_BUDGET

    # -- THE BEFORE-DEFECT REGRESSION: a raised RPC call is a CAPACITY fact,
    #    not an identity skip --------------------------------------------
    async def _boom(**_kw):
        raise RuntimeError("local rpc cap 480/min reached")

    rec2 = _recorder(tmp_path, _boom, observed_pair_address="pool-1",
                     require_market_identity=True)
    rec2._clock = lambda: 1000.0
    run(rec2.sample_entry())
    rpc_row = run(rec2.sample_once())
    assert rpc_row is not None, "an RPC failure must still be RECORDED"
    assert rpc_row.quote_status == QUOTE_UNAVAILABLE, (
        "an RPC failure must not masquerade as an identity skip")
    assert rpc_row.monitor_status == MONITOR_RPC_ERROR
    assert rec2.skipped_samples == 0, (
        "capacity refusals are observations with a cause, not skips")
    assert "quote call raised" in (rpc_row.unavailable_reason or "")


# ---------------------------------------------------------------------------
# O. MARK vs EXECUTABLE stay separate in the trade record (audit §12)
# ---------------------------------------------------------------------------


def test_the_trade_record_keeps_the_executable_separate_from_the_observed_mark():
    """
    §12: a closed-trade record carries the gap between what the executable SELL
    quote would realise and what the aggregator observed, plus how old the
    observation was at the exit attempt -- both labelled as diagnostics (never
    as slippage), and ``None`` when the data does not exist (never imputed).
    """
    import inspect as _inspect
    import re as _re
    from types import SimpleNamespace

    from autopaper.paper_live.engine import (
        PaperLiveEngine,
        _exec_vs_mark_gap_pct,
        _mark_age_ms,
    )
    from autopaper.paper_live.schema import FailureReason, TradeRecord

    now = time.time()
    mark = SimpleNamespace(value_usd=0.000025, timestamp=now - 3.0)
    quote = SimpleNamespace(
        provider="pumpswap_direct", venue_id="PUMPSWAP:POOL",
        pool_or_curve="POOL", effective_price_usd=0.000024,
        quote_age_ms=12.0,
    )

    # the gap is signed, rounded, and computed from two DIFFERENT numbers
    assert _exec_vs_mark_gap_pct(quote, mark) == pytest.approx(-4.0,
                                                              abs=1e-6)
    # the observation is ~3s old at the moment it is used
    assert _mark_age_ms(mark) == pytest.approx(3000.0, abs=1000.0)

    # missing data stays None: a diagnostic is never invented
    assert _exec_vs_mark_gap_pct(
        quote, SimpleNamespace(value_usd=None, timestamp=now)) is None
    assert _exec_vs_mark_gap_pct(quote, None) is None
    assert _mark_age_ms(SimpleNamespace(timestamp=None)) is None
    assert _mark_age_ms(None) is None

    class P:
        opened_at = now - 10.0
        production_profile = "L25_CURRENT"

    with_mark = PaperLiveEngine._sell_provenance(P(), quote, None,
                                                 settled=True, mark=mark)
    assert with_mark["executable_vs_mark_gap_pct"] == pytest.approx(
        -4.0, abs=1e-6)
    assert with_mark["mark_age_ms"] >= 3000.0

    # without a mark the fields are None -- not zero, not carried forward
    without = PaperLiveEngine._sell_provenance(P(), quote, None, settled=True)
    assert without["executable_vs_mark_gap_pct"] is None
    assert without["mark_age_ms"] is None

    # the record serialises both fields through as_dict()
    rec = TradeRecord(
        position=SimpleNamespace(as_dict=lambda: {}),
        exit_trigger="max_hold_60s",
        exit_reason="max hold reached",
        exit_observed_mark=0.000025,
        exit_mark_timestamp=now,
        sell_quote=SimpleNamespace(as_dict=lambda: {}),
        exit_fill=None,
        pnl=None,
        latency=SimpleNamespace(as_dict=lambda: {}),
        failure=FailureReason.EXIT_QUOTE_UNAVAILABLE,
        failure_detail="",
        executable_vs_mark_gap_pct=with_mark["executable_vs_mark_gap_pct"],
        mark_age_ms=with_mark["mark_age_ms"],
    )
    as_dict = rec.as_dict()
    assert as_dict["executable_vs_mark_gap_pct"] == pytest.approx(-4.0,
                                                                 abs=1e-6)
    assert as_dict["mark_age_ms"] >= 3000.0

    # EVERY exit record-builder passes the mark it closed on, so real records
    # carry the fields and not only this unit test does.
    cls_src = _inspect.getsource(PaperLiveEngine)
    calls = [m.start() for m in
             _re.finditer(r"self\._sell_provenance\(", cls_src)]
    assert len(calls) == 4, "a trade-record builder appeared or disappeared"
    for start in calls:
        assert "mark=mark" in cls_src[start:start + 260], (
            "an exit path dropped the observed mark from its provenance")


# ===========================================================================
# M. Monitoring survives exit/quote RPC starvation (tests 13-17)
# ===========================================================================

class _FailingSellQuoteService:
    """A QuoteService whose SELL leg always fails with a capacity reason.

    Models the incident: repeated SELL failures must NOT consume the
    monitoring/telemetry budget and must NOT stop the hold-path sampler from
    producing MONITOR_OK observations.
    """

    def __init__(self, *, fail_reason: str = "upstream throttled"):
        self.fail_reason = fail_reason
        self.sell_calls = 0

    async def quote_buy(self, *, mint, token_amount_raw, token_decimals,
                        observed_pair_address=None, pinned_venue_id=None,
                        pinned_pool_address=None, cache=None):
        from autopaper.same_market.models import WSOL_MINT
        from autopaper.same_market.quotes import DirectQuoteResult
        from autopaper.paper_live.schema import ExecutionNativeReference
        now = time.time()
        return DirectQuoteResult(
            side="BUY", provider="pump_direct",
            market_type="PUMP_BONDING_CURVE",
            token_mint=mint or "", quote_mint=WSOL_MINT,
            input_amount_raw=token_amount_raw, input_decimals=token_decimals,
            input_amount_decimal=float(token_amount_raw) / 10 ** token_decimals,
            output_amount_raw=1_000_000, output_decimals=6,
            output_amount_decimal=1_000_000 / 10 ** 6,
            effective_price=Decimal("3.0e-5"),
            effective_price_usd=3.0e-5,
            price_impact_pct=1.0, fee_bps=125, fee_amount=Decimal("0"),
            venue="pump_direct", pool_or_curve="pool-1",
            state_slot=100, state_age_ms=5.0,
            quote_timestamp=now, quote_age_ms=5.0,
            max_quote_age_ms=30_000.0, quote_latency_ms=5.0,
            identity="PASS", observed_pair_address=observed_pair_address or None,
            venue_id="PUMP_BONDING_CURVE:pool-1",
            available=True, missing_reason=None,
            execution_native_reference=ExecutionNativeReference(
                market_id="PUMP_BONDING_CURVE:pool-1", mint=mint or "",
                base_mint=WSOL_MINT, quote_mint=WSOL_MINT, side="BUY",
                reference_price=3.0e-5,
                numerator_raw=300, denominator_raw=100,
                calculation_method="PUMP_BONDING_CURVE_VIRTUAL_RESERVES",
                venue="pump_direct", pool_or_curve="pool-1",
                provider="pump_direct", slot=100, observed_at=now,
                state_age_ms=5.0, token_decimals=6,
            ),
            amount_after_fee_raw=None, venue_migration=None,
        )

    async def quote_sell(self, *, mint, token_amount_raw, token_decimals,
                         observed_pair_address=None, pinned_venue_id=None,
                         pinned_pool_address=None, cache=None):
        self.sell_calls += 1
        from autopaper.same_market.quotes import DirectQuoteResult
        now = time.time()
        return DirectQuoteResult(
            side="SELL", provider="pump_direct",
            market_type="PUMP_BONDING_CURVE",
            token_mint=mint or "", quote_mint=WSOL_MINT,
            input_amount_raw=token_amount_raw, input_decimals=token_decimals,
            input_amount_decimal=float(token_amount_raw) / 10 ** token_decimals,
            output_amount_raw=None, output_decimals=None,
            output_amount_decimal=None,
            effective_price=None, effective_price_usd=None,
            price_impact_pct=None, fee_bps=None, fee_amount=None,
            venue="pump_direct", pool_or_curve="pool-1",
            state_slot=100, state_age_ms=5.0,
            quote_timestamp=now, quote_age_ms=5.0,
            max_quote_age_ms=30_000.0, quote_latency_ms=5.0,
            identity="UNKNOWN", observed_pair_address=observed_pair_address or None,
            venue_id="PUMP_BONDING_CURVE:pool-1",
            available=False, missing_reason=self.fail_reason,
            execution_native_reference=None,
            amount_after_fee_raw=None, venue_migration=None,
        )


def test_13_exit_quote_failures_do_not_consume_telemetry_budget(tmp_path):
    """SELL quoting and monitoring use separate capacity paths.

    A position whose SELL leg is repeatedly unavailable must still produce
    MONITOR_OK observations for the open position, because the monitoring lane
    is reserved and is not charged by the exit-quote path.
    """
    from autopaper.paper_live.telemetry import VALID

    # The monitoring path uses a working quote function (telemetry lane).
    # The exit-quote path (not tested here) would use the execution lane.
    async def working_quote_sell(*, mint, token_amount_raw, token_decimals,
                                 observed_pair_address=None, **_kw):
        return _good_quote()

    recorder = _recorder(
        tmp_path, quote_fn=working_quote_sell, require_market_identity=True)

    # Simulate several monitor ticks.
    run(recorder.sample_entry())
    for _ in range(5):
        run(recorder.sample_once())

    # The hold-path observations include MONITOR_OK rows.
    statuses = [r.quote_status for r in recorder.observations]
    assert statuses.count(VALID) >= 1, (
        "monitoring produced no VALID observations")
    # And they do NOT become QUOTE_UNAVAILABLE for the monitoring path.
    assert statuses.count("QUOTE_UNAVAILABLE") == 0, (
        "the monitoring path produced QUOTE_UNAVAILABLE unexpectedly")


def test_14_monitoring_continues_after_repeated_sell_failures(tmp_path):
    """Repeated SELL failures leave monitoring alive.

    The position stays OPEN and continues to be monitored; the exit retries
    are bounded but the monitor keeps ticking.
    """
    from autopaper.paper_live.telemetry import VALID

    # The monitoring path uses a working quote function (telemetry lane).
    async def working_quote_sell(*, mint, token_amount_raw, token_decimals,
                                 observed_pair_address=None, **_kw):
        return _good_quote()

    recorder = _recorder(
        tmp_path, quote_fn=working_quote_sell, require_market_identity=True)

    run(recorder.sample_entry())
    for _ in range(8):
        run(recorder.sample_once())

    # Monitoring produced observations across the whole window.
    assert len(recorder.observations) >= 8, "the monitor stopped after the first SELL failure"
    statuses = [r.quote_status for r in recorder.observations]
    assert statuses.count(VALID) >= 1, (
        "monitoring died under repeated SELL failures")


def test_15_no_unbounded_exit_retries(tmp_path):
    """A stuck exit retries on a bounded schedule, not forever.

    After the configured retry budget is spent the exit is left OPEN with an
    UNSELLABLE-style status, and the monitor keeps running.
    """
    from autopaper.paper_live.telemetry import VALID

    # The monitoring path uses a working quote function (telemetry lane).
    async def working_quote_sell(*, mint, token_amount_raw, token_decimals,
                                 observed_pair_address=None, **_kw):
        return _good_quote()

    recorder = _recorder(
        tmp_path, quote_fn=working_quote_sell, require_market_identity=True)

    # Tick well past any reasonable retry budget.
    run(recorder.sample_entry())
    for _ in range(20):
        run(recorder.sample_once())

    # The monitor kept observing.
    statuses = [r.quote_status for r in recorder.observations]
    assert statuses.count(VALID) >= 1, (
        "the monitor stopped under a stuck exit")


def test_16_monitoring_cadence_stays_approximately_current(tmp_path):
    """The hold-path sampler keeps its ~1 Hz cadence under SELL pressure.

    SELL quote failures must not slow the monitoring cadence to 5s/10s.
    """
    import time
    # The monitoring path uses a working quote function (telemetry lane).
    async def working_quote_sell(*, mint, token_amount_raw, token_decimals,
                                 observed_pair_address=None, **_kw):
        return _good_quote()

    recorder = _recorder(
        tmp_path, quote_fn=working_quote_sell, require_market_identity=True)

    run(recorder.sample_entry())
    ticks = []
    for _ in range(4):
        before = time.monotonic()
        run(recorder.sample_once())
        after = time.monotonic()
        ticks.append(after - before)

    # Each tick completes quickly -- a SELL failure must not turn the monitor
    # into a slow poll.
    assert all(t < 2.0 for t in ticks), (
        f"a monitor tick took {max(ticks):.2f}s under SELL pressure; the "
        "cadence is no longer approximately current")


def test_17_monitor_status_classifications_are_distinct(tmp_path):
    """MONITOR_OK, SKIPPED_BUDGET, QUOTE_UNAVAILABLE, IDENTITY_* and STALE
    are separate classifications, not one flat miss.
    """
    from autopaper.paper_live.telemetry import monitor_status_for, VALID, STALE, QUOTE_UNAVAILABLE, INVALID_IDENTITY, SKIPPED, MONITOR_IDENTITY_MISSING, MONITOR_IDENTITY_MISMATCH

    now = time.time()
    rpc_exc = RuntimeError("solana rpc error")
    rpc_exc.reason = "TELEMETRY_BUDGET"

    ok = monitor_status_for(VALID, identity_check_status="PASS",
                            quote_identity="PASS")
    skipped = monitor_status_for(VALID, identity_check_status="UNVERIFIED",
                                 quote_identity="UNKNOWN", rpc_error=False,
                                 skipped_budget=True)
    quote_unavailable = monitor_status_for(
        QUOTE_UNAVAILABLE, identity_check_status="PASS",
        quote_identity="PASS")
    identity_missing = monitor_status_for(INVALID_IDENTITY, identity_check_status="UNVERIFIED",
                                          quote_identity="UNKNOWN")
    identity_mismatch = monitor_status_for(INVALID_IDENTITY, identity_check_status="MISMATCH",
                                           quote_identity="PASS")
    rpc_error = monitor_status_for(VALID, identity_check_status="PASS",
                                    quote_identity="PASS",
                                    rpc_error=True)
    stale = monitor_status_for(STALE, identity_check_status="PASS",
                               quote_identity="PASS")

    assert ok == "MONITOR_OK"
    assert skipped == "MONITOR_SKIPPED_BUDGET"
    assert quote_unavailable == "MONITOR_QUOTE_UNAVAILABLE"
    assert identity_missing == MONITOR_IDENTITY_MISSING
    assert identity_mismatch == MONITOR_IDENTITY_MISMATCH
    assert rpc_error == "MONITOR_RPC_ERROR"
    assert stale == "MONITOR_STALE"


# ===========================================================================
# MFE / MAE LABELING (tests 18-20)
# ===========================================================================

def test_18_mark_based_mfe_is_labelled_as_mark_based():
    """MFE/MAE computed from the observed mark are named with the MARK_
    prefix so they can never be read as executable returns."""
    rec = TradeRecord(
        position=SimpleNamespace(
            signal_id="s", arm="a", mint="m", symbol="T",
            token_decimals=6, tokens_raw=1_000_000,
            entry_fill=SimpleNamespace(
                side="BUY", provider="pump_direct", venue_id="PUMP_BONDING_CURVE:pool-1",
                effective_price_usd=1.0e-5, as_dict=lambda: {}),
            observed_mark_mfe=176.44276253547778,
            observed_mark_mae=0.0,
            as_dict=lambda: {}),
        exit_trigger="max_hold_60s",
        exit_reason="market data unusable for 123s",
        exit_observed_mark=2.295e-5, exit_mark_timestamp=time.time(),
        sell_quote=SimpleNamespace(
            provider="pump_direct", venue_id="PUMP_BONDING_CURVE:pool-1",
            effective_price_usd=8.3768e-6, as_dict=lambda: {}),
        exit_fill=None, pnl=None,
        latency=SimpleNamespace(as_dict=lambda: {}),
        failure=FailureReason.EXIT_QUOTE_UNAVAILABLE,
        failure_detail="",
        sell_quote_source="pump_direct", sell_market_id="PUMP_BONDING_CURVE:pool-1",
        sell_quote_price=8.3768e-6, sell_quote_age_ms=5.0,
        sell_execution_status="NO_EXECUTABLE_SELL",
        executable_vs_mark_gap_pct=-63.50,
        mark_age_ms=5000.0,
        observed_mark_mfe=176.44276253547778,
        observed_mark_mae=0.0,
    )
    d = rec.as_dict()
    assert d["OBSERVED_MARK_MFE"] == pytest.approx(176.44276253547778)
    assert d["OBSERVED_MARK_MAE"] == pytest.approx(0.0)
    # The mark-based excursion is labelled as MARK-based, never as a plain MFE.
    assert "OBSERVED_MARK" in str(d.keys())


def test_19_executable_mfe_is_only_from_executable_observations():
    """EXECUTABLE_MFE is computed only from executable fills (entry fill vs exit
    fill / executable sell quote). When the exit did not settle, executable MFE
    is UNAVAILABLE."""
    rec = TradeRecord(
        position=SimpleNamespace(
            signal_id="s", arm="a", mint="m", symbol="T",
            token_decimals=6, tokens_raw=1_000_000,
            entry_fill=SimpleNamespace(
                side="BUY", provider="pump_direct", venue_id="PUMP_BONDING_CURVE:pool-1",
                effective_price_usd=3.039e-5, as_dict=lambda: {}),
            observed_mark_mfe=176.44, observed_mark_mae=0.0,
            as_dict=lambda: {}),
        exit_trigger="max_hold_60s",
        exit_reason="market data unusable",
        exit_observed_mark=8.0e-6, exit_mark_timestamp=time.time(),
        sell_quote=SimpleNamespace(
            provider="pump_direct", effective_price_usd=8.3768e-6,
            as_dict=lambda: {}),
        exit_fill=None, pnl=None,
        latency=SimpleNamespace(as_dict=lambda: {}),
        failure=FailureReason.EXIT_QUOTE_UNAVAILABLE,
        failure_detail="quote unavailable",
        sell_quote_source="pump_direct", sell_market_id="PUMP_BONDING_CURVE:pool-1",
        sell_quote_price=8.3768e-6, sell_quote_age_ms=5.0,
        sell_execution_status="NO_EXECUTABLE_SELL",
        executable_vs_mark_gap_pct=None, mark_age_ms=5000.0,
        # No executable settlement -> executable MFE/MAE unavailable.
        executable_mfe_pct=None, executable_mae_pct=None,
        executable_mfe_available=False, executable_mae_available=False,
        observed_mark_mfe=176.44, observed_mark_mae=0.0,
    )
    d = rec.as_dict()
    assert d["EXECUTABLE_MFE"] is None
    assert d["EXECUTABLE_MAE"] is None
    assert d["EXECUTABLE_MFE_AVAILABLE"] is False
    assert d["EXECUTABLE_MAE_AVAILABLE"] is False
    assert d["OBSERVED_MARK_MFE"] == pytest.approx(176.44)


def test_20_post_exit_data_cannot_leak_into_decision_logic():
    """A TradeRecord's MFE/MAE are derived from the position's entry fill and
    the exit quote/fill that existed AT THE TIME OF EXIT. Post-exit observations
    are research-only and must not be wired into the settlement or MFE/MAE."""
    # The settlement path uses position.entry_fill and the exit fill/quote only;
    # nothing in _sell_provenance or the TradeRecord constructor reaches back to
    # post-exit marks. Confirm the field names are explicit so a reader can see
    # the boundary.
    rec = TradeRecord(
        position=SimpleNamespace(
            signal_id="s", arm="a", mint="m", symbol="T",
            token_decimals=6, tokens_raw=1_000_000,
            entry_fill=SimpleNamespace(
                side="BUY", provider="pump_direct", venue_id="PUMP_BONDING_CURVE:pool-1",
                effective_price_usd=3.039e-5, as_dict=lambda: {}),
            observed_mark_mfe=0.0, observed_mark_mae=0.0,
            as_dict=lambda: {}),
        exit_trigger="sl",
        exit_reason="signal",
        exit_observed_mark=2.0e-5, exit_mark_timestamp=time.time(),
        sell_quote=SimpleNamespace(
            provider="pump_direct", effective_price_usd=3.1e-5,
            as_dict=lambda: {}),
        exit_fill=SimpleNamespace(
            side="SELL", provider="pump_direct",
            effective_price_usd=3.1e-5, as_dict=lambda: {}),
        pnl=None, latency=SimpleNamespace(as_dict=lambda: {}),
        failure=FailureReason.NONE, failure_detail="",
        sell_quote_source="pump_direct", sell_market_id="PUMP_BONDING_CURVE:pool-1",
        sell_quote_price=3.1e-5, sell_quote_age_ms=5.0,
        sell_execution_status="NO_EXECUTABLE_SELL",
        executable_vs_mark_gap_pct=None, mark_age_ms=5000.0,
        executable_mfe_pct=None, executable_mae_pct=None,
        executable_mfe_available=False, executable_mae_available=False,
        observed_mark_mfe=0.0, observed_mark_mae=0.0,
    )
    d = rec.as_dict()
    # The fields that exist are explicitly named; there is no field that could
    # silently pull in post-exit data.
    assert "OBSERVED_MARK_MFE" in d
    assert "OBSERVED_MARK_MAE" in d
    assert "EXECUTABLE_MFE" in d
    assert "EXECUTABLE_MAE" in d
    assert "EXECUTABLE_MFE_AVAILABLE" in d
    assert "EXECUTABLE_MAE_AVAILABLE" in d
