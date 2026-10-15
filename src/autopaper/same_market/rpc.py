"""
Read-only Solana JSON-RPC client.

WHY THIS EXISTS INSTEAD OF REUSING ``autopaper.execution.solana``
=================================================================

The vendored client in this repository does what an executor needs, which
includes ``send_transaction`` and a wallet that holds a private key. This
harness needs none of that. More importantly, the no-broadcast guarantee is
only credible if the capability is absent rather than unused - so the RPC
surface here is written out longhand with exactly five read methods and no
signing, no keypair and no ``sendTransaction`` in its API.

There is therefore nothing for a caller to call even if it wanted to. That is a
stronger guarantee than a runtime flag, a dry-run boolean, or an import
barrier, and it is why the import barrier in :mod:`autopaper.same_market.guard`
is a second layer rather than the only one.

PUBLIC RPC POLICINESS
=====================

The default endpoint is the free public node, which is shared and aggressively
rate limited. So this client is deliberately slow about it:

* a sliding-window :class:`~autopaper.market.ratelimit.RateLimiter`
* bounded retries with jittered backoff inside a fixed wall-clock budget, so
  retries cannot multiply the request cost
* a TTL cache, so repeated interest in one account does not become repeated
  requests
* in-flight request de-duplication, so concurrent callers asking for the same
  account share one request instead of racing to the node
* ``429`` and ``-32005`` handling that surfaces as a named rate-limit failure
  rather than a generic error
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Sequence

import httpx

from ..market.ratelimit import (
    LANE_EXECUTION,
    LANE_TELEMETRY,
    LOCAL_WINDOW,
    TELEMETRY_BUDGET,
    UPSTREAM_BLOCKED,
    RateLimiter,
    RetryAfterBackoff,
)
from .models import TOKEN_2022_PROGRAM, TOKEN_PROGRAM

#: The free public Solana node. Deliberately the default: this harness must not
#: require a paid provider, and hammering a paid one for a measurement would
#: be worse still.
DEFAULT_RPC_URL = "https://api.mainnet-beta.solana.com"

#: Read methods. This tuple is the client's entire remote capability and is
#: asserted against in the tests, so a method cannot be added without one of
#: them failing.
READ_METHODS: tuple[str, ...] = (
    "getAccountInfo",
    "getMultipleAccounts",
    "getTokenSupply",
    "getSlot",
    "getVersion",
)


class RpcRateLimited(RuntimeError):
    """The node refused the request. Named, never collapsed to a generic error."""

    retry_after_s: float | None

    def __init__(self, message: str, *, retry_after_s: float | None = None,
                 reason: str = LOCAL_WINDOW, blocked_until: float | None = None,
                 provider: str | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s
        #: WHICH condition refused: LOCAL_WINDOW, UPSTREAM_BLOCKED or
        #: TELEMETRY_BUDGET. Never inferred from the message text by a caller.
        self.reason = reason
        self.blocked_until = blocked_until
        self.provider = provider

    @property
    def is_upstream(self) -> bool:
        return self.reason == UPSTREAM_BLOCKED

    @property
    def is_local_quota(self) -> bool:
        return self.reason == LOCAL_WINDOW


#: WHICH LANE THE CURRENT CALL IS IN.
#:
#: A ContextVar rather than a parameter threaded through every reader method,
#: because the lane is a property of the CALL SITE, not of the data being read:
#: the same ``getMultipleAccounts`` is an execution read from one caller and a
#: background read from another. ``asyncio`` copies the context per task, so a
#: telemetry sampler task cannot leak its lane into the monitor task.
_RPC_LANE: ContextVar[str] = ContextVar("autopaper_rpc_lane", default=LANE_EXECUTION)


def current_rpc_lane() -> str:
    return _RPC_LANE.get()


@contextlib.contextmanager
def rpc_lane(lane: str):
    """Run a block in a named RPC lane (``execution`` or ``telemetry``)."""
    token = _RPC_LANE.set(lane)
    try:
        yield
    finally:
        _RPC_LANE.reset(token)


#: TELEMETRY SAMPLE ADMISSION.
#:
#: A background hold-path sample is ONE unit of work that happens to need
#: several chain reads (the curve account, the pump fee config, the mint's
#: supply). Charging the reserved telemetry sub-budget PER RPC CALL made a 1 Hz
#: sampler cost ~250-300 calls/min against a 60/min reservation, so the sampler
#: starved ITSELF: samples were refused mid-read, the refusal was wrapped as
#: "the bonding curve could not be read", and a self-imposed capacity decision
#: was reported as a market failure -- which is how a position with a readable,
#: identified, quotable market produced 73% QUOTE_UNAVAILABLE hold ticks.
#:
#: An admitted sample therefore costs the reservation ONE unit, wherever in the
#: sample the first read happens, and may then make up to
#: ``_TELEMETRY_SAMPLE_CALL_CEILING`` reads uncharged. Past that ceiling the
#: admission closes and per-call charging resumes, so an admitted sample can
#: never become an unbounded escape from the reservation.
_TELEMETRY_SAMPLE_OPEN: ContextVar[bool] = ContextVar(
    "autopaper_telemetry_sample_open", default=False)
_TELEMETRY_SAMPLE_CALLS: ContextVar[int] = ContextVar(
    "autopaper_telemetry_sample_calls", default=0)
_TELEMETRY_SAMPLE_CALL_CEILING = 8


@contextlib.contextmanager
def telemetry_sample():
    """Make ONE telemetry sample cost the reserved budget one unit, not N."""
    open_token = _TELEMETRY_SAMPLE_OPEN.set(True)
    calls_token = _TELEMETRY_SAMPLE_CALLS.set(0)
    try:
        yield
    finally:
        _TELEMETRY_SAMPLE_CALLS.reset(calls_token)
        _TELEMETRY_SAMPLE_OPEN.reset(open_token)


def _telemetry_sample_covers() -> bool:
    """
    Should THIS telemetry-lane call skip its own charge?

    ``False`` for the first read of a sample (it pays for the whole sample) and
    for every read once the ceiling is reached; ``True`` for the reads in
    between. Never affects any other lane.
    """
    if not _TELEMETRY_SAMPLE_OPEN.get():
        return False
    n = _TELEMETRY_SAMPLE_CALLS.get() + 1
    _TELEMETRY_SAMPLE_CALLS.set(n)
    if n >= _TELEMETRY_SAMPLE_CALL_CEILING:
        _TELEMETRY_SAMPLE_OPEN.set(False)
    return n > 1


#: Upstream-throttle events, newest last, capped. THE 429 THAT SETS THE BLOCK WAS
#: PREVIOUSLY INVISIBLE: it set ``blocked_until`` and nothing reported it, so the
#: only symptom an operator ever saw was a downstream "no executable quote".
#: Bounded so a flapping provider cannot grow this without limit.
_UPSTREAM_EVENTS: list[dict[str, Any]] = []
_UPSTREAM_EVENT_CAP = 50


def _record_upstream_event(**fields: Any) -> None:
    _UPSTREAM_EVENTS.append(dict(fields))
    if len(_UPSTREAM_EVENTS) > _UPSTREAM_EVENT_CAP:
        del _UPSTREAM_EVENTS[:-_UPSTREAM_EVENT_CAP]


def upstream_events() -> list[dict[str, Any]]:
    """Structured upstream-throttle events. Read by the engine for logging."""
    return list(_UPSTREAM_EVENTS)


class RpcUnavailable(RuntimeError):
    """The node did not answer, or answered with something unusable."""


@dataclass(frozen=True)
class RpcAccount:
    """
    One account, with its raw bytes retained.

    The raw bytes are kept because interpretation happens in
    :mod:`autopaper.same_market.pump`, and a harness that discarded them could
    not later prove what it decoded.
    """

    address: str
    owner: str
    lamports: int
    data_b64: str
    executable: bool
    rent_epoch: int | None

    @property
    def is_empty(self) -> bool:
        return not self.data_b64

    def raw(self) -> bytes:
        if not self.data_b64:
            return b""
        try:
            return base64.b64decode(self.data_b64)
        except (ValueError, TypeError) as exc:
            raise RpcUnavailable(f"account data for {self.address} is not base64") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "owner": self.owner,
            "lamports": self.lamports,
            "data_len": len(self.data_b64),
            "executable": self.executable,
            "rent_epoch": self.rent_epoch,
        }


#: Slot time on Solana is ~400ms. A slot read is chain METADATA, not market
#: data, so reusing it inside one slot's lifetime cannot misprice anything --
#: it only removes a round trip that was already redundant.
DEFAULT_SLOT_CACHE_TTL_S = 0.4


@dataclass
class _Endpoint:
    """One upstream RPC node plus the health we have learned about it."""

    url: str
    #: Wall-clock until which this endpoint is treated as saturated. An
    #: endpoint in cooldown is skipped in favour of another, which is how one
    #: throttled node stops deciding the availability of the whole runtime.
    cooldown_until: float = 0.0
    rate_limited: int = 0
    errors: int = 0
    calls: int = 0

    def available(self, now: float) -> bool:
        return now >= self.cooldown_until

    def penalise(self, now: float, delay_s: float, *, rate_limited: bool) -> None:
        if rate_limited:
            self.rate_limited += 1
        else:
            self.errors += 1
        self.cooldown_until = max(self.cooldown_until, now + max(0.25, delay_s))


class SolanaReadOnlyRpc:
    """
    Async JSON-RPC client exposing read methods only.

    There is no ``send_transaction``, no ``sign``, no keypair and no
    transaction-building method on this class. Its entire surface is
    :attr:`READ_METHODS` plus the corresponding ``get_*`` wrappers.
    """

    read_methods = READ_METHODS

    def __init__(
        self,
        url: str | None = None,
        *,
        urls: Sequence[str] | None = None,
        timeout_s: float = 8.0,
        max_rpm: int = 20,
        max_concurrent: int = 6,
        slot_cache_ttl_s: float = DEFAULT_SLOT_CACHE_TTL_S,
        limiter_wait_s: float = 2.0,
        cache_ttl_s: float = 1.0,
        max_retries: int = 2,
        telemetry_max_per_minute: int | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.url = (url or DEFAULT_RPC_URL).strip() or DEFAULT_RPC_URL
        # MULTIPLE ENDPOINTS. The public node is shared and aggressively
        # throttled; sending every quote through one URL made an upstream
        # saturation look like "this token has no executable quote". Endpoints
        # are tried in order and a throttled one is skipped while another is
        # tried, so no single node can decide availability on its own.
        raw = [u.strip() for u in (urls or []) if u and u.strip()]
        # The CONFIGURED url stays FIRST: it is the operator's primary and must
        # not be demoted behind a failover endpoint by declaration order.
        self.urls: list[str] = list(dict.fromkeys([self.url] + raw))
        self.endpoints = [_Endpoint(u) for u in self.urls]
        self.timeout_s = float(timeout_s)
        self.cache_ttl_s = float(cache_ttl_s)
        self.slot_cache_ttl_s = float(slot_cache_ttl_s)

        # BOUNDED CONCURRENCY. Independent reads may overlap, which is what
        # keeps p95 decision latency flat while capacity rises. The cap stops
        # that overlap from becoming a self-inflicted thundering herd.
        self.max_concurrent = max(1, int(max_concurrent))
        self._sem = asyncio.Semaphore(self.max_concurrent)
        # A local cap that WAITS inside a bounded budget instead of raising, so
        # a momentary burst delays a request rather than deleting it.
        #
        # ``telemetry_max_per_minute`` RESERVES capacity for execution: the
        # background hold-path sampler can spend at most that much, so the
        # remainder is unreachable by telemetry and always available to an open
        # position trying to sell. The TOTAL is still ``max_rpm``, so this
        # partitions the existing budget rather than enlarging the provider
        # request rate.
        self.limiter = RateLimiter(
            max_rpm, name="solana-rpc",
            telemetry_max_per_minute=telemetry_max_per_minute,
        )
        self.limiter_wait_s = max(0.0, float(limiter_wait_s))
        self.backoff = RetryAfterBackoff(max_attempts=max_retries, budget_s=self.timeout_s)
        self._client = client
        self._own_client = client is None
        self._id = 0
        # De-duplication of concurrent identical requests, so N coroutines
        # wanting the same account produce one node call.
        self._slot_cache: int | None = None
        self._slot_cache_at: float | None = None
        self._inflight: dict[str, asyncio.Task] = {}
        self._cache: dict[str, tuple[float, Any]] = {}
        self.calls = 0
        self.cache_hits = 0
        self.dedupe_hits = 0

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
            self._own_client = True
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._own_client:
            await self._client.aclose()
        self._client = None

    # -- transport ---------------------------------------------------------

    async def _rpc(self, method: str, params: list[Any]) -> Any:
        if method not in self.read_methods:
            # A belt-and-braces assertion on top of the class having no other
            # method. If this ever fires, a write path has been added to a
            # client documented as read-only.
            raise RpcUnavailable(
                f"'{method}' is not a read method on this client; refusing"
            )
        # A BOUNDED WAIT, AND AN HONEST ONE.
        #
        # The wait is bounded by ``limiter_wait_s`` and never extends past it.
        # What it must never do is claim that a LOCAL quota is what is in the way
        # when the provider has actually told us to back off for 60s -- that
        # misdiagnosis is what made an upstream throttle look like a self-
        # inflicted cap, and it invited the wrong fix.
        #
        # So the three conditions are kept apart, and the WAKE-UP SCHEDULE is
        # chosen per condition:
        #   * LOCAL_WINDOW   -- a real, short wait helps; poll until the budget.
        #   * UPSTREAM_BLOCKED -- waiting cannot help inside a 2s budget when the
        #     cooldown is 60s. Refuse IMMEDIATELY and name it, so the caller's own
        #     retry cadence (one monitor tick) becomes the recovery mechanism
        #     instead of a doomed 2s sleep per candidate per tick.
        #   * TELEMETRY_BUDGET -- opportunistic work must never queue behind
        #     anything. Skip at once.
        # THE WAKE-UP SCHEDULE IS CHOSEN PER CONDITION, BY WHETHER WAITING CAN
        # WORK AT ALL:
        #   * A block that fits inside the wait budget -> WAIT. A momentary
        #     upstream Retry-After is cheaper to sit out than to fail, and
        #     failing would report a market we could have quoted.
        #   * A block LONGER than the whole budget -> refuse at once, naming it.
        #     Waiting 2s to learn a 60s cooldown has not cleared is pure latency,
        #     and claiming the local quota merely "needs 2s" is the misdiagnosis
        #     this change exists to remove. The caller's own retry cadence (one
        #     monitor tick) becomes the recovery mechanism.
        #   * A full LOCAL window -> WAIT, up to the budget. That is what the
        #     budget is for.
        #   * An exhausted TELEMETRY budget -> SKIP at once. Opportunistic work
        #     must never queue behind anything.
        lane = _RPC_LANE.get()
        # THE REST OF AN ADMITTED SAMPLE IS NOT CHARGED AGAIN. Exactly one read
        # of a telemetry sample pays the reserved sub-budget; the following
        # reads inside the same sample (bounded by
        # ``_TELEMETRY_SAMPLE_CALL_CEILING``) are the SAME unit of work. Without
        # this a 1 Hz sampler spent the 60/min reservation in a few seconds and
        # then refused its own reads. No other lane is affected.
        _telemetry_covered = (
            lane == LANE_TELEMETRY and _telemetry_sample_covers())
        deadline = time.monotonic() + self.limiter_wait_s
        while not _telemetry_covered:
            now = time.monotonic()
            allowed, wait, reason = self.limiter.check(now, lane)
            if allowed:
                self.limiter.acquire(now, lane)
                break
            # RECORD THE REFUSAL. This loop raises its own ``RpcRateLimited``
            # rather than letting ``acquire`` raise, so without this the
            # limiter's per-reason counters stayed at zero for the life of the
            # process and the operator's "telemetry_budget_refusals" read 0
            # while the telemetry budget was being spent on every tick. Counted
            # exactly once, on the single path that leaves the loop.
            self.limiter.note_rejection(reason)
            if reason == UPSTREAM_BLOCKED and now + wait > deadline:
                # Waiting cannot clear this inside the budget. Say so precisely.
                raise RpcRateLimited(
                    f"solana rpc UPSTREAM THROTTLED: the provider asked us to "
                    f"back off; {wait:.1f}s of cooldown remain, which does not "
                    f"fit the {self.limiter_wait_s:.1f}s wait budget"
                    + (f" (retry-after "
                       f"{self.limiter.last_upstream_retry_after_s:.0f}s)"
                       if self.limiter.last_upstream_retry_after_s else "")
                    + ". This is a provider throttle, NOT the local "
                    f"{self.limiter.max_per_minute}/min quota",
                    retry_after_s=wait, reason=UPSTREAM_BLOCKED,
                    blocked_until=self.limiter.blocked_until,
                    provider=getattr(self.limiter,
                                     "last_upstream_block_provider", None),
                )
            if reason == TELEMETRY_BUDGET:
                # NOT AN ERROR. Telemetry is background work; skipping is the
                # correct outcome and must not surface as a failed quote.
                raise RpcRateLimited(
                    f"solana rpc telemetry lane budget spent "
                    f"({self.limiter.telemetry_max_per_minute}/min); sample "
                    f"skipped so execution capacity stays free",
                    retry_after_s=wait, reason=TELEMETRY_BUDGET,
                )
            # LOCAL_WINDOW, or an upstream block that CAN clear in budget.
            if now + wait > deadline:
                if reason == UPSTREAM_BLOCKED:
                    raise RpcRateLimited(
                        f"solana rpc UPSTREAM THROTTLED: cooldown did not "
                        f"clear within {self.limiter_wait_s:.1f}s "
                        f"({wait:.1f}s remaining). This is a provider throttle, "
                        f"NOT the local {self.limiter.max_per_minute}/min quota",
                        retry_after_s=wait, reason=UPSTREAM_BLOCKED,
                        blocked_until=self.limiter.blocked_until,
                        provider=getattr(self.limiter,
                                         "last_upstream_block_provider", None),
                    )
                raise RpcRateLimited(
                    f"solana rpc LOCAL quota exhausted: "
                    f"{self.limiter.max_per_minute}/min in use, and it did not "
                    f"clear within {self.limiter_wait_s:.1f}s",
                    retry_after_s=wait, reason=LOCAL_WINDOW,
                )
            await asyncio.sleep(min(wait, 0.25))

        self.calls += 1
        client = await self._ensure_client()
        self._id += 1
        payload = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}
        headers = {
            "accept": "application/json",
            "user-agent": "AUTONOMOUS_PAPER_AGENT_V1/1.0 (same-market read-only test)",
        }
        # The live endpoints, healthiest first, skipping any in cooldown.
        now = time.monotonic()
        live = [e for e in self.endpoints if e.available(now)]
        if not live:
            # Everything is cooling down. Try them all anyway rather than give
            # up: a short wait here is strictly better than reporting a token
            # as unquotable because every node was busy a moment ago.
            live = list(self.endpoints)
        live.sort(key=lambda e: (e.cooldown_until > now, e.rate_limited, e.errors))

        # BOUNDED CONCURRENCY. Overlap is what lets a multi-call quote finish
        # in roughly the time of its slowest call instead of their sum, which is
        # how quote availability rises without decision latency rising.
        async def _attempt_on(endpoint: _Endpoint) -> Any:
            endpoint.calls += 1
            resp = await client.post(endpoint.url, json=payload, headers=headers,
                                     timeout=self.timeout_s)
            if resp.status_code == 429:
                delay = _retry_after(resp)
                endpoint.penalise(time.monotonic(), delay or 5.0,
                                  rate_limited=True)
                raise _rate_limited(delay)
            if resp.status_code >= 400:
                endpoint.penalise(time.monotonic(), 2.0, rate_limited=False)
                raise RpcUnavailable(f"solana rpc HTTP {resp.status_code}")
            try:
                body = resp.json()
            except (ValueError, json.JSONDecodeError) as exc:
                raise RpcUnavailable("solana rpc returned non-JSON") from exc
            if not isinstance(body, dict):
                raise RpcUnavailable("solana rpc returned a non-object body")
            error = body.get("error")
            if error:
                code = (error or {}).get("code")
                if code == -32005:
                    # "Node is behind by 51 slots" - the node's own overload
                    # signal, so this endpoint is throttled rather than wrong.
                    endpoint.penalise(time.monotonic(), 5.0, rate_limited=True)
                    raise _rate_limited(None)
                raise RpcUnavailable(
                    f"solana rpc error {code}: {(error or {}).get('message')}"
                )
            return body.get("result")

        async with self._sem:
            # ENDPOINT FAILOVER. If the chosen node is throttled, try the next
            # one rather than surfacing the saturation. A rate limit is a fact
            # about ONE node, not about the token being unquotable.
            for attempt in range(len(live)):
                endpoint = live[attempt]
                endpoint.calls += 1
                try:
                    return await self.backoff.run(
                        lambda e=endpoint: _attempt_on(e))
                except RpcRateLimited as exc:
                    # An upstream 429 is a fact about ONE node, so it cools
                    # DOWN THAT NODE (done in ``_attempt_on``) rather than the
                    # whole client. Recording it into the shared local window
                    # would refuse every later request before any endpoint is
                    # even chosen, which makes the failover above unreachable.
                    #
                    # With a single endpoint there is nowhere else to go, so
                    # the client backs off -- as it must, because continuing to
                    # hammer a throttling provider is what earns a longer ban.
                    if len(live) == 1:
                        # THE CLOCK MUST MATCH THE ONE ``check`` COMPARES AGAINST.
                        #
                        # This passed ``time.time()`` while every comparison in
                        # the limiter is against ``time.monotonic()``. Those two
                        # differ by the machine's uptime, so ``blocked_until`` was
                        # set roughly 1.79e9 seconds in the future and
                        # ``now < blocked_until`` was PERMANENTLY true: one
                        # upstream 429 blocked this client FOREVER, not for the
                        # 60 s the code appeared to promise. An open position
                        # then had no path back to an executable quote at all,
                        # which is precisely the stuck-OPEN failure this whole
                        # change exists to eliminate.
                        cooldown = exc.retry_after_s or 60.0
                        self.limiter.note_retry_after(
                            time.monotonic(), cooldown,
                            provider=endpoint.url)
                        _record_upstream_event(
                            at=time.time(),
                            provider=endpoint.url,
                            method=method,
                            retry_after_s=exc.retry_after_s,
                            cooldown_applied_s=cooldown,
                            blocked_until=self.limiter.blocked_until,
                            cooldown_remaining_s=max(
                                0.0,
                                self.limiter.blocked_until - time.monotonic()),
                            entered_cooldown=True,
                            lane=lane,
                            endpoints_configured=len(self.endpoints),
                            operation="execution" if lane == LANE_EXECUTION
                            else "telemetry",
                        )
                    if attempt + 1 >= len(live):
                        raise
                    # Another endpoint exists: try it instead of giving up.
                    continue
            raise RpcUnavailable("no rpc endpoint could serve the request")


    # -- caching / dedupe --------------------------------------------------

    def _cached(self, key: str, now: float) -> Any | None:
        hit = self._cache.get(key)
        if hit is None:
            return None
        stamped, value = hit
        if now - stamped > self.cache_ttl_s:
            self._cache.pop(key, None)
            return None
        self.cache_hits += 1
        return value

    async def _deduped(self, key: str, fn: Any) -> Any:
        """
        Share one in-flight request among callers asking for the same key.
        """
        existing = self._inflight.get(key)
        if existing is not None:
            self.dedupe_hits += 1
            return await asyncio.shield(existing)
        task = asyncio.ensure_future(fn())
        self._inflight[key] = task
        try:
            value = await task
        finally:
            self._inflight.pop(key, None)
        return value

    # -- reads -------------------------------------------------------------

    async def get_account_info(
        self,
        address: str,
        *,
        encoding: str = "base64",
        commitment: str = "confirmed",
    ) -> RpcAccount | None:
        """
        One account, or ``None`` when the address holds nothing.

        A missing account is a legitimate answer (an ungraduated token has no
        PumpSwap pool), so it is ``None`` rather than an exception.
        """
        key = f"acct:{address}:{encoding}:{commitment}"
        now = time.time()
        cached = self._cached(key, now)
        if cached is not None:
            return cached

        async def _do() -> RpcAccount | None:
            result = await self._rpc("getAccountInfo", [
                address,
                {"encoding": encoding, "commitment": commitment},
            ])
            if not result:
                return None
            value = result.get("value") if isinstance(result, dict) else None
            if not value:
                return None
            account = RpcAccount(
                address=address,
                owner=str(value.get("owner") or ""),
                lamports=int(value.get("lamports") or 0),
                data_b64=str(((value.get("data") or [None])[0]) or ""),
                executable=bool(value.get("executable")),
                rent_epoch=value.get("rentEpoch"),
            )
            self._cache[key] = (time.time(), account)
            return account

        return await self._deduped(key, _do)

    async def get_multiple_accounts(
        self,
        addresses: list[str],
        *,
        encoding: str = "base64",
        commitment: str = "confirmed",
    ) -> list[RpcAccount | None]:
        """Batched read. One request for N accounts, not N requests."""
        if not addresses:
            return []
        key = f"multi:{','.join(addresses)}:{encoding}:{commitment}"
        cached = self._cached(key, time.time())
        if cached is not None:
            return cached

        result = await self._rpc("getMultipleAccounts", [
            addresses,
            {"encoding": encoding, "commitment": commitment},
        ])
        values = (result or {}).get("value") if isinstance(result, dict) else None
        out: list[RpcAccount | None] = []
        for address, value in zip(addresses, values or []):
            if not value:
                out.append(None)
                continue
            out.append(RpcAccount(
                address=address,
                owner=str(value.get("owner") or ""),
                lamports=int(value.get("lamports") or 0),
                data_b64=str(((value.get("data") or [None])[0]) or ""),
                executable=bool(value.get("executable")),
                rent_epoch=value.get("rentEpoch"),
            ))
        return out

    async def get_token_supply(self, mint: str) -> int | None:
        """Circulating supply in raw units, or ``None`` when unavailable."""
        result = await self._rpc("getTokenSupply", [mint])
        if not isinstance(result, dict):
            return None
        value = (result.get("value") or {}).get("amount")
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    async def get_slot(self) -> int | None:
        """
        The current slot, cached for the length of a slot.

        Every account read is preceded by a slot read so the account and the
        slot it was observed at are consistent. That is correct -- and it
        means every read used to cost TWO round trips, one of them repeated
        within the same millisecond.

        A slot is chain METADATA, not market data: reusing it for ~400ms cannot
        change any price, it only removes a redundant request. Account values
        themselves are NOT given this cache -- they are what a quote is made of.
        """
        now = time.monotonic()
        if self._slot_cache_at is not None and (
                now - self._slot_cache_at) < self.slot_cache_ttl_s:
            return self._slot_cache
        result = await self._rpc("getSlot", [])
        try:
            value: int | None = int(result)
        except (TypeError, ValueError):
            value = None
        if value is not None:
            self._slot_cache = value
            self._slot_cache_at = time.monotonic()
        return value

    async def get_version(self) -> str | None:
        result = await self._rpc("getVersion", [])
        if isinstance(result, dict):
            return str((result.get("solana-core") or "")) or None
        return None


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _rate_limited(delay: float | None) -> RpcRateLimited:
    """
    An UPSTREAM provider throttle -- HTTP 429, or JSON-RPC ``-32005``.

    The structured reason is ``UPSTREAM_BLOCKED``, and it must be, because
    :class:`RpcRateLimited` DEFAULTS ``reason`` to ``LOCAL_WINDOW`` and this
    factory did not override it. So a provider saying "I am throttling you"
    arrived downstream carrying "our own local quota was reached" -- the exact
    collapse the limiter's own docstring says the three reasons exist to
    prevent, moved out of the message text and into the field that callers are
    told to trust.

    It matters behaviourally, not just in reporting: ``paper_live.engine``'s
    ``_classify_rpc_cause`` returns ``"local_window"`` for ``LOCAL_WINDOW`` and
    returns immediately, so the correct ``upstream_blocked`` fallback -- which
    consults ``limiter.blocked_until`` -- never ran, and an exit was misfiled as
    a local capacity problem whose remedy is the opposite of the real one.
    """
    err = RpcRateLimited("solana rpc rate limited", reason=UPSTREAM_BLOCKED)
    err.retry_after_s = delay
    return err


def is_token_program(owner: str) -> bool:
    """Is this account owned by an SPL token program?"""
    return owner in (TOKEN_PROGRAM, TOKEN_2022_PROGRAM)


__all__ = [
    "DEFAULT_RPC_URL",
    "READ_METHODS",
    "RpcAccount",
    "RpcRateLimited",
    "RpcUnavailable",
    "SolanaReadOnlyRpc",
    "is_token_program",
]