"""
Shared rate limiting and backoff.

Public market APIs must be treated as a scarce shared resource. This module
implements:

* a token-bucket style per-minute request cap
* mandatory ``Retry-After`` obedience on 429/503
* exponential backoff with jitter for transient failures
* a last-good cache so a rate-limited fetch can still be served as STALE rather
  than failing outright (never silently promoted to LIVE)

WHY A DEADLINE EXISTS HERE (measured, not hypothetical)
--------------------------------------------------------
``httpx.Client(timeout=8.0)`` is NOT a wall-clock budget for a request. httpx
applies that number to each *phase* independently (pool / connect / write /
read), and the connect phase is charged **once per resolved address** by
``socket.create_connection``, which walks ``getaddrinfo`` results in order. On a
host with two unroutable IPv6 addresses ahead of the reachable IPv4 ones, a
single "8 second timeout" GET costs:

    8.0s (dead IPv6 #1) + 8.0s (dead IPv6 #2) + ~0.05s (IPv4, actual response)
    = 16.05s wall clock

Measured on this host for ``api.dexscreener.com`` and
``api.geckoterminal.com`` (2026-09-30): 16.5s per provider call, every call.
That single fact produced the ~35-43s analysis tail, because a candidate reads
the market once before the alert settle and once after it.

So the timeout must be expressed as a DEADLINE - a wall-clock budget that the
whole attempt is held to - not as a per-phase knob. Two mechanisms below:

* :func:`split_timeouts` derives a short *connect* timeout from the overall
  budget, because the connect phase is the only one charged per address.
* :func:`run_with_deadline` holds any synchronous callable to a wall-clock
  budget on a dedicated worker, so even a slow-drip body or an unexpected
  failure mode cannot overrun it.
"""

from __future__ import annotations

import concurrent.futures
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Optional, TypeVar

T = TypeVar("T")


class MarketDataError(RuntimeError):
    def __init__(self, message: str, *, rate_limited: bool = False,
                 retry_after_s: float | None = None, status: int | None = None,
                 deadline_exceeded: bool = False):
        super().__init__(message)
        self.rate_limited = rate_limited
        self.retry_after_s = retry_after_s
        self.status = status
        #: True when the call was abandoned for exceeding its wall-clock budget,
        #: as opposed to the provider having answered with an error. Reported so
        #: "we gave up waiting" is never confused with "the venue said no".
        self.deadline_exceeded = deadline_exceeded


# ---------------------------------------------------------------------------
# wall-clock deadlines
# ---------------------------------------------------------------------------

#: Wall-clock overrun a wall-clock budget is allowed before it counts as blown.
#: A few milliseconds of scheduling noise is not a timeout.
DEADLINE_SLACK_S = 0.05

#: Threads abandoned because they blew their deadline, by reason. Observable so a
#: provider that is permanently over budget is measurable rather than silently
#: slow. A thread that overruns is abandoned, never joined, and it is left to
#: exit on its own once its own socket operations finish.
deadline_stats: dict[str, int] = {
    "calls": 0,
    "timeouts": 0,
    "overruns": 0,
    "last_overrun_s": 0.0,
}
_deadline_stats_lock = threading.Lock()

#: A small dedicated pool. Deliberately NOT ``asyncio.to_thread``'s default
#: executor: a request abandoned at its deadline keeps occupying its worker until
#: the socket operation ends, and that must not eat the pool the event loop uses
#: for its own ``to_thread`` work (the async provider path runs through it).
_deadline_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="market-deadline"
)


def split_timeouts(total_s: float, *, connect_s: float | None = None) -> tuple[float, float]:
    """
    ``(connect, phase)`` timeouts derived from one overall budget.

    ``connect`` is the part that gets charged per resolved address, so it is
    capped hard; the remaining phases get the rest of the budget. Returns plain
    floats so it can be handed straight to ``httpx.Timeout`` without importing
    httpx into this module.
    """
    total = max(0.1, float(total_s))
    if connect_s is None:
        # Two seconds is long enough for a real TLS handshake over a healthy
        # path (measured 13-45ms here) and short enough that an unroutable
        # address costs 2s instead of 8s.
        connect_s = min(2.0, total)
    connect = max(0.25, min(float(connect_s), total))
    return connect, total


def run_with_deadline(fn: Callable[[], T], budget_s: float, *,
                      label: str = "provider") -> T:
    """
    Run ``fn`` and hold it to a WALL-CLOCK budget.

    ``fn`` is synchronous (the provider transport is ``httpx.Client``), so there
    is no cooperative way to interrupt it. Running it on a dedicated worker and
    abandoning it at the budget is the honest implementation of a deadline: the
    CALLER is released on time, which is what keeps the alert->decision path
    inside its budget, and the abandoned thread terminates on its own once its
    socket operations finish.

    Raises :class:`MarketDataError` with ``deadline_exceeded=True`` on overrun.
    """
    budget = float(budget_s)
    if budget <= 0:
        raise MarketDataError(
            f"{label}: non-positive deadline budget ({budget}s)",
            deadline_exceeded=True,
        )
    started = time.monotonic()
    with _deadline_stats_lock:
        deadline_stats["calls"] += 1
    future = _deadline_pool.submit(fn)
    try:
        return future.result(timeout=budget)
    except concurrent.futures.TimeoutError as exc:
        elapsed = time.monotonic() - started
        with _deadline_stats_lock:
            deadline_stats["timeouts"] += 1
            deadline_stats["overruns"] += 1
            deadline_stats["last_overrun_s"] = max(
                float(deadline_stats["last_overrun_s"]), elapsed
            )
        future.cancel()   # a no-op if already running; that is the point
        raise MarketDataError(
            f"{label}: exceeded the {budget:.2f}s wall-clock budget "
            f"(gave up after {elapsed:.2f}s)",
            deadline_exceeded=True,
        ) from exc


def remaining_s(deadline: float) -> float:
    """Seconds left until ``deadline`` (a ``time.monotonic`` stamp)."""
    return max(0.0, deadline - time.monotonic())


def deadline_stats_snapshot() -> dict[str, object]:
    with _deadline_stats_lock:
        return {
            **deadline_stats,
            "overrun_s": round(float(deadline_stats["last_overrun_s"]), 3),
            "workers": _deadline_pool._max_workers,
        }


#: Why a request was refused. THE THREE ARE NOT INTERCHANGEABLE.
#:
#: They used to collapse into one boolean plus a wait, which is why an upstream
#: provider throttle and a local quota were both reported to the operator as
#: "local rate cap reached" -- a diagnosis that points at the wrong system, and
#: therefore invites the wrong fix.
LOCAL_WINDOW = "LOCAL_WINDOW"
UPSTREAM_BLOCKED = "UPSTREAM_BLOCKED"
TELEMETRY_BUDGET = "TELEMETRY_BUDGET"

#: The reservation lane. Execution may use the whole budget; telemetry may use
#: only its own sub-budget, so a background sampler can never consume the
#: capacity an open position needs to sell.
LANE_EXECUTION = "execution"
LANE_TELEMETRY = "telemetry"


class RateLimiter:
    """
    Sliding-window limiter with a RESERVED lane for BOTH directions.

    THE TOTAL RATE NEVER RISES. ``max_per_minute`` still bounds every request
    this limiter admits, so adding a lane cannot multiply the provider request
    rate -- it only decides who gets the existing budget when it is contended.

    TWO PROTECTIONS, both structural:

    1. TELEMETRY CANNOT STARVE EXECUTION (the original reservation): a
       telemetry acquire must satisfy its own smaller sub-budget, so
       ``max - telemetry_max`` requests per minute are unreachable by the
       background sampler and always remain for a risk-reducing SELL.
    2. EXECUTION CANNOT STARVE TELEMETRY (audit §3): the telemetry floor is
       cut OUT of the execution ceiling, and a telemetry acquire is judged
       ONLY against its own counter. Exit/quote retries filling the execution
       window can therefore never refuse a monitoring request -- the core
       invariant that ``SELL failure must NOT disable OPEN POSITION
       monitoring``. The two ceilings sum to ``max_per_minute`` exactly, so
       the reservation partitions the budget instead of enlarging it.

    Execution always retains at least one slot (ceilings are floored at 1), so
    a degenerate reservation can never disable exits entirely.

    ``telemetry_max_per_minute=None`` disables the sub-budget, preserving the
    previous single-lane behaviour exactly for every existing caller.
    """

    def __init__(self, max_per_minute: int, *, name: str = "provider",
                 telemetry_max_per_minute: int | None = None) -> None:
        self.max_per_minute = max(1, int(max_per_minute))
        self.name = name
        self._calls: Deque[float] = deque()
        self._telemetry_calls: Deque[float] = deque()
        #: The window the EXECUTION lane is checked against. Identical to
        #: ``_calls`` in legacy (no-reservation) mode; execution-only acquires
        #: in reserved mode.
        self._exec_calls: Deque[float] = deque()
        #: Ceiling for the telemetry lane alone. ``None`` = unbounded (legacy).
        self.telemetry_max_per_minute = (
            None if telemetry_max_per_minute is None
            else max(0, min(int(telemetry_max_per_minute), self.max_per_minute))
        )
        self.blocked_until: float = 0.0
        self.total_requests = 0
        self.telemetry_requests = 0
        self.rejections = 0
        self.rejections_by_reason: dict[str, int] = {
            LOCAL_WINDOW: 0, UPSTREAM_BLOCKED: 0, TELEMETRY_BUDGET: 0,
        }
        self.rate_limit_events = 0
        #: Set when the block came from an upstream provider telling us to back
        #: off, rather than from the local window. The distinction is the whole
        #: point: one is ours to fix, the other is not.
        self.blocked_by_upstream = False
        self.last_upstream_retry_after_s: float | None = None

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        while self._calls and self._calls[0] <= cutoff:
            self._calls.popleft()
        while self._telemetry_calls and self._telemetry_calls[0] <= cutoff:
            self._telemetry_calls.popleft()
        while self._exec_calls and self._exec_calls[0] <= cutoff:
            self._exec_calls.popleft()

    def telemetry_ceiling(self) -> int:
        """
        Requests/min the TELEMETRY lane may make in one window.

        Clamped to ``max - 1`` so telemetry can never take the execution
        lane's last slot: the two ceilings sum to ``max_per_minute`` exactly.
        """
        if self.telemetry_max_per_minute is None:
            return 0
        return min(self.telemetry_max_per_minute,
                   max(0, self.max_per_minute - 1))

    def execution_ceiling(self) -> int:
        """
        Requests/min the EXECUTION lane may make in one window.

        With a reservation configured, the telemetry floor is cut OUT of this
        ceiling -- execution can never spend the capacity monitoring is
        guaranteed, no matter how hard exit/quote retries press. Floored at 1
        so a degenerate reservation can never disable exits entirely.
        """
        if self.telemetry_max_per_minute is None:
            return self.max_per_minute
        return max(1, self.max_per_minute - self.telemetry_ceiling())

    def check(self, now: float, lane: str = LANE_EXECUTION
              ) -> tuple[bool, float, str | None]:
        """
        Admit or refuse, naming WHICH condition refused.

        Returns ``(allowed, retry_after_s, reason)``. Precedence matters:

        1. An upstream block outranks everything. While a provider has told us to
           stay away, no amount of local budget makes the request succeed, and
           spending local budget on it would be wrong.
        2. The telemetry lane (reserved mode) is judged ONLY against its own
           sub-budget. The shared window being full of exit traffic must never
           refuse a monitoring request -- that is the starvation this design
           exists to prevent, and the clamp in :meth:`telemetry_ceiling` is what
           keeps the total bounded anyway.
        3. The execution window, which is capped BELOW the total by exactly the
           telemetry floor.
        4. Legacy telemetry (no reservation configured) shares the total window
           exactly as the single-lane limiter always did.
        """
        # 1. UPSTREAM. Reported as itself, never as a local cap.
        if now < self.blocked_until:
            return False, self.blocked_until - now, UPSTREAM_BLOCKED
        self._prune(now)

        # 0. AN UNRECOGNISED LANE IS REFUSED, NOT GUESSED.
        #
        # Branches 2 and 3 below test for the two lane constants and everything
        # else fell through to branch 4, the legacy shared window. So a typo --
        # "telemetery" -- was judged against the TOTAL window (where it had
        # headroom while the telemetry sub-budget was spent) and then recorded by
        # ``acquire`` into the EXECUTION counter, because that method treats
        # "anything not LANE_TELEMETRY" as execution. One misspelling therefore
        # both escaped the telemetry budget entirely and overran the execution
        # ceiling, while ``reserved_for_execution()`` still reported the budget
        # as reserved.
        #
        # ``check`` is pure and is called by the hot path, so this raises rather
        # than returning a third reason: a caller cannot act on an invented
        # reason string, and silently reinterpreting the lane is worse than a
        # loud failure on a programming error.
        if lane not in (LANE_TELEMETRY, LANE_EXECUTION):
            raise ValueError(
                f"{self.name}: unknown rate-limit lane {lane!r}; expected "
                f"{LANE_EXECUTION!r} or {LANE_TELEMETRY!r}. Refusing rather "
                "than guessing, because guessing charges the wrong budget."
            )
        return self._decide(now, lane)

    def note_rejection(self, reason: str | None) -> None:
        """
        Record that a caller was refused by the budget.

        ``check`` is deliberately PURE -- it is also reachable from ``allow``,
        which is a plain capacity probe with no intention of spending anything,
        so counting inside ``check`` would inflate the counters from a
        read-only caller.

        It has to be recorded explicitly, because the only production
        lane-using caller (``same_market.rpc``) calls ``check`` and raises its
        own ``RpcRateLimited`` instead of letting ``acquire`` raise. The three
        per-reason counters the operator is shown --
        ``local_window_refusals``, ``upstream_blocked_refusals`` and
        ``telemetry_budget_refusals`` -- therefore stayed at zero for the life
        of the process while the telemetry budget was demonstrably being spent.
        A working limit reported as never having refused anything.
        """
        self.rejections += 1
        if reason:
            self.rejections_by_reason[reason] = (
                self.rejections_by_reason.get(reason, 0) + 1)

    def _decide(self, now: float, lane: str) -> tuple[bool, float, str | None]:
        """The budget decision itself, once the lane has been validated."""
        # 2. RESERVED TELEMETRY LANE -- own counter only.
        if lane == LANE_TELEMETRY and self.telemetry_max_per_minute is not None:
            if len(self._telemetry_calls) >= self.telemetry_ceiling():
                # A zero budget is spent by definition and there is no window to
                # age out, so there is nothing to wait for. Reporting a wait
                # computed from an empty deque would both crash and imply a
                # delay that does not exist.
                if not self._telemetry_calls:
                    return False, 60.0, TELEMETRY_BUDGET
                return (False,
                        max(0.1, 60.0 - (now - self._telemetry_calls[0])),
                        TELEMETRY_BUDGET)
            return True, 0.0, None

        # 3. EXECUTION WINDOW (also used by legacy telemetry below).
        if lane == LANE_EXECUTION:
            if len(self._exec_calls) >= self.execution_ceiling():
                return (False,
                        max(0.1, 60.0 - (now - self._exec_calls[0])),
                        LOCAL_WINDOW)
            return True, 0.0, None

        # 4. LEGACY TELEMETRY: no reservation configured, shares the total.
        if len(self._calls) >= self.max_per_minute:
            return (False,
                    max(0.1, 60.0 - (now - self._calls[0])),
                    LOCAL_WINDOW)
        return True, 0.0, None

    def allow(self, now: float) -> tuple[bool, float]:
        """Backwards-compatible 2-tuple form. See :meth:`check` for the reason."""
        allowed, wait, _reason = self.check(now, LANE_EXECUTION)
        return allowed, wait

    def acquire(self, now: float, lane: str = LANE_EXECUTION) -> float:
        """Record a call. Returns the delay the caller should have honoured."""
        # ``check`` has already counted the refusal if there was one, so this
        # path must NOT increment the counters again.
        allowed, wait, reason = self.check(now, lane)
        if not allowed:
            self.note_rejection(reason)
            raise MarketDataError(
                f"{self.name}: {_describe_refusal(reason, wait, self)}",
                rate_limited=True,
                retry_after_s=wait,
            )
        self._calls.append(now)
        if lane == LANE_TELEMETRY:
            self._telemetry_calls.append(now)
            self.telemetry_requests += 1
            if self.telemetry_max_per_minute is None:
                # Legacy mode: telemetry shares the execution window exactly as
                # the single-lane limiter always did.
                self._exec_calls.append(now)
        else:
            self._exec_calls.append(now)
        self.total_requests += 1
        return 0.0

    def note_retry_after(self, now: float, retry_after_s: float | None,
                         *, provider: str | None = None) -> None:
        self.rate_limit_events += 1
        # RECORD THAT THIS BLOCK CAME FROM UPSTREAM. This is what lets the
        # operator-facing error say "the provider throttled us" instead of
        # "our own quota ran out", which are different incidents with different
        # fixes. Before this, the two were indistinguishable downstream.
        self.blocked_by_upstream = True
        self.last_upstream_retry_after_s = retry_after_s
        self.last_upstream_block_provider = provider
        if retry_after_s is None:
            # 60 requests/min is DexScreener's documented limit; when they say
            # 429 without a header, assume we must back off to that window.
            self.blocked_until = max(self.blocked_until, now + 60.0)
        else:
            self.blocked_until = max(
                self.blocked_until, now + max(0.0, float(retry_after_s))
            )

    def clear_upstream_block(self) -> None:
        """Drop an upstream block once its cooldown has elapsed."""
        if self.blocked_until and self.blocked_until <= time.time():
            self.blocked_until = 0.0
            self.blocked_by_upstream = False
            self.last_upstream_retry_after_s = None
            self.last_upstream_block_provider = None

    def reserved_for_execution(self) -> int:
        """
        Requests/min that telemetry structurally cannot reach.

        This is the reservation made concrete and assertable: a background
        sampler cannot take more than its ceiling from the budget, so at least
        this many remain for execution.
        """
        if self.telemetry_max_per_minute is None:
            return 0
        return max(0, self.max_per_minute - self.telemetry_ceiling())

    def stats(self) -> dict[str, float | int]:
        return {
            "requests": self.total_requests,
            "telemetry_requests": self.telemetry_requests,
            "rejections": self.rejections,
            "rate_limit_events": self.rate_limit_events,
            "max_per_minute": self.max_per_minute,
            "telemetry_max_per_minute": self.telemetry_max_per_minute,
            "reserved_for_execution": self.reserved_for_execution(),
            "rejections_local_window": self.rejections_by_reason.get(LOCAL_WINDOW, 0),
            "rejections_upstream_blocked":
                self.rejections_by_reason.get(UPSTREAM_BLOCKED, 0),
            "rejections_telemetry_budget":
                self.rejections_by_reason.get(TELEMETRY_BUDGET, 0),
            "blocked_by_upstream": self.blocked_by_upstream,
            "last_upstream_retry_after_s": self.last_upstream_retry_after_s,
            "blocked_for_s": max(0.0, self.blocked_until - time.time()),
        }


def _describe_refusal(reason: str | None, wait: float, limiter: "RateLimiter") -> str:
    """
    Name the ACTUAL condition. The old text was "local rate cap reached" for
    every case, which told the operator their own quota was exhausted when the
    provider had in fact throttled them.
    """
    if reason == UPSTREAM_BLOCKED:
        who = getattr(limiter, "last_upstream_block_provider", None) or "the provider"
        return (
            f"{limiter.name}: UPSTREAM THROTTLED US -- {who} asked us to back off; "
            f"cooldown {wait:.1f}s remaining. This is NOT the local "
            f"{limiter.max_per_minute}/min quota"
        )
    if reason == TELEMETRY_BUDGET:
        return (
            f"{limiter.name}: telemetry lane budget of "
            f"{limiter.telemetry_max_per_minute}/min is spent; skipped to keep "
            f"capacity free for execution"
        )
    if reason == LOCAL_WINDOW:
        return (
            f"{limiter.name}: local rate cap reached "
            f"({limiter.max_per_minute}/min); retry in {wait:.1f}s"
        )
    return f"{limiter.name}: rate limited ({reason or 'unknown'}); retry in {wait:.1f}s"


@dataclass
class RetryAfterBackoff:
    """
    Exponential backoff with full jitter; deterministic seed for tests.

    ALSO A VISIBILITY LEDGER. A partial failure (attempt 1 timed out, attempt 2
    succeeded) used to be completely invisible: the caller saw a successful
    return and nothing recorded that the provider had been degraded. ``run``
    now records every attempt, the failures and the backoff actually slept, so a
    provider that is quietly flaking is measurable instead of silent.
    """

    base_s: float = 0.4
    factor: float = 2.0
    max_s: float = 8.0
    max_attempts: int = 2
    rng: random.Random = field(default_factory=lambda: random.Random(0xA70))
    #: Wall-clock budget for the WHOLE ``run()``, retries and backoff included.
    #: ``None`` means unbounded, which is only ever used by tests that inject a
    #: zero-delay backoff. A provider call in the live runtime always carries one,
    #: so ``timeout * (attempts + 1)`` is no longer the real worst case.
    budget_s: Optional[float] = None

    # -- observability (recorded by run(), never used for control flow) --
    #: Attempts made by the MOST RECENT run(), including the successful one.
    last_attempts: int = 0
    #: Human-readable errors from every failed attempt in the most recent run.
    last_errors: list[str] = field(default_factory=list)
    #: Total seconds slept backing off in the most recent run.
    last_backoff_s: float = 0.0
    #: Did the most recent run() return successfully?
    last_ok: bool = False
    #: Wall-clock seconds the most recent run() took, start to finish.
    last_elapsed_s: float = 0.0

    def delay_for(self, attempt: int) -> float:
        raw = min(self.max_s, self.base_s * (self.factor ** max(0, attempt)))
        return self.rng.uniform(0.0, raw)

    def run(self, fn: Callable[[], T], *, sleep: Callable[[float], None] = time.sleep) -> T:
        """
        Call ``fn`` with bounded retries AND a bounded total budget.

        Two things stop the loop: the attempt count, and the wall clock. A
        transient error at attempt 1 no longer costs the full per-attempt
        timeout twice plus two backoff sleeps, because the attempt after the
        deadline is never made. ``last_elapsed_s`` records what the run actually
        cost, so the budget can be checked against reality instead of assumed.
        """
        last: Exception | None = None
        started = time.monotonic()
        deadline = (
            None if self.budget_s is None
            else started + max(0.0, float(self.budget_s))
        )
        self.last_attempts = 0
        self.last_errors = []
        self.last_backoff_s = 0.0
        self.last_ok = False
        for attempt in range(1, self.max_attempts + 2):
            if deadline is not None and remaining_s(deadline) <= DEADLINE_SLACK_S:
                self.last_elapsed_s = time.monotonic() - started
                raise MarketDataError(
                    f"retry budget of {self.budget_s:.2f}s exhausted after "
                    f"{attempt - 1} attempt(s)",
                    deadline_exceeded=True,
                )
            self.last_attempts = attempt
            try:
                result = fn()
                self.last_ok = True
                self.last_elapsed_s = time.monotonic() - started
                return result
            except MarketDataError as exc:
                self.last_errors.append(f"{type(exc).__name__}: {exc}")
                if exc.rate_limited or not _is_transient(exc):
                    self.last_elapsed_s = time.monotonic() - started
                    raise
                last = exc
                if attempt > self.max_attempts:
                    break
                delay = self.delay_for(attempt - 1)
                if deadline is not None:
                    # Never sleep past the deadline: the point of the budget is
                    # that the caller is released on time.
                    delay = min(delay, remaining_s(deadline))
                self.last_backoff_s += delay
                if delay > 0:
                    sleep(delay)
        self.last_elapsed_s = time.monotonic() - started
        assert last is not None
        raise last

    @property
    def succeeded_after_retry(self) -> bool:
        """Attempt 1 failed and a later attempt returned: the partial failure."""
        return self.last_ok and self.last_attempts > 1


def classify_error(exc: BaseException) -> str:
    """A stable, loggable label for a provider failure.

    Distinguishing a timeout from a connection reset is the difference between
    "the venue is slow" and "the venue dropped us", and the fix for one is not
    the fix for the other. Anything unrecognised is still named.
    """
    if isinstance(exc, MarketDataError) and exc.deadline_exceeded:
        # "We stopped waiting" is its own failure mode and must not be reported
        # as if the venue had answered.
        return "deadline_exceeded"
    name = type(exc).__name__
    text = str(exc).lower()
    if "timeout" in text or "timed out" in text or name in ("TimeoutError", "ReadTimeout", "ConnectTimeout"):
        return "timeout"
    if "reset" in text or "connection reset" in text or isinstance(exc, ConnectionResetError):
        return "connection_reset"
    if "refused" in text or isinstance(exc, ConnectionRefusedError):
        return "connection_refused"
    if isinstance(exc, (OSError, ConnectionError)):
        return "network"
    return name


def _is_transient(exc: MarketDataError) -> bool:
    # A blown wall-clock budget is never retried. Retrying would spend the
    # budget again and, by definition, miss the deadline it just missed.
    if exc.deadline_exceeded:
        return False
    if exc.status is None:
        return True
    return exc.status in {408, 425, 429, 500, 502, 503, 504}


class TTLCache:
    """Small cache with an explicit age, used to serve STALE on failure."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[float, T]] = {}

    def put(self, key: str, value: T, now: float) -> None:
        self._items[key] = (now, value)

    def get(self, key: str, ttl_s: float, now: float) -> tuple[T, float] | None:
        entry = self._items.get(key)
        if entry is None:
            return None
        stamped, value = entry
        age = now - stamped
        if age > ttl_s:
            return None
        return value, age

    def get_any(self, key: str) -> tuple[T, float] | None:
        return self._items.get(key)

    def set_stamp(self, key: str, now: float) -> None:
        """Re-stamp an entry as 'just observed' only for the *arrival* time.

        Never use this to fake an observation's venue timestamp.
        """
        entry = self._items.get(key)
        if entry is not None:
            self._items[key] = (now, entry[1])

    def drop(self, key: str) -> None:
        self._items.pop(key, None)

    def keys(self) -> list[str]:
        return list(self._items)
