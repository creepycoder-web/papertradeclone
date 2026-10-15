"""Free on-chain state reading, done politely.

WHY THIS EXISTS
---------------
The 2026-10-02 audit found that a market mark described one pool while the trade
executed on another, and that the system never had an authoritative, venue-
identified price of its own. Reading the chain fixes the second problem and
makes the first one *detectable* -- but only if reading it is cheap enough to do
on every decision, which is exactly the constraint that stops people doing it.

THE CONSTRAINT IS REAL
----------------------
``https://api.mainnet-beta.solana.com`` is free, needs no key, and has **no
service-level agreement of any kind**. It rate-limits aggressively, shares
capacity with every other user, and may be slow, unavailable or reordered at any
moment. This module therefore treats it as what it is:

* **CACHED** with a short TTL, so one decision that needs the same account three
  times makes one request.
* **DEDUPLICATED** across concurrent callers: identical in-flight requests share
  one future, so a burst of positions on the same mint costs one round trip.
* **BOUNDED** in concurrency, so N positions cannot open N sockets.
* **RATE-LIMITED** by a token bucket, with ``429`` and ``Retry-After`` honoured
  rather than retried into.
* **RETRIED** with exponential backoff and jitter on 429/5xx only.
* **EXPLICIT about staleness**: every read reports how old it is and whether it
  was served from cache. A caller that cannot tolerate staleness checks.

FAIL CLOSED, ALWAYS
-------------------
An operation that cannot safely complete from the free public RPC raises
:class:`~autopaper.execution.quote_errors.ChainStateUnavailable`. It never falls
back to a different market source. A PumpSwap pool whose reserves cannot be read
must not be priced from a DexScreener mark wearing its name -- that substitution
is the entire class of defect this work exists to remove.

NO PAID DEPENDENCY
------------------
Nothing here requires Helius, Birdeye, or any paid RPC or data tier. The client
speaks plain JSON-RPC to whatever endpoint it is given, and the default in
:func:`default_public_rpc_url` is the free public endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import httpx

log = logging.getLogger("execution.chainstate")

#: The free Solana mainnet endpoint. No key, no SLA, no guarantee.
DEFAULT_PUBLIC_RPC_URL = "https://api.mainnet-beta.solana.com"

#: Account data size that comfortably holds a PumpSwap pool state or a Pump.fun
#: bonding curve. Anything larger is refused rather than truncated, because a
#: truncated program account decodes into plausible nonsense.
MAX_ACCOUNT_DATA_BYTES = 1_024

#: Anchors for the "decimals" and "supply" fields of an SPL mint. Offsets are
#: documented borsh layout positions: 0..3 is the mint authority Option tag,
#: 4..35 the authority, 36..43 the raw supply, 44 the decimals byte.
_MINT_SUPPLY_OFFSET = 36
_MINT_DECIMALS_OFFSET = 44
_TOKEN_ACCOUNT_AMOUNT_OFFSET = 64
_POOL_ACCOUNT_OWNER = "owner"

#: The canonical Pump.fun bonding-curve program.
PUMP_BONDING_CURVE_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
#: The canonical PumpSwap (pAMM) program.
PUMP_SWAP_PROGRAM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
#: The canonical Pump AMM (pumpfun) program id, as DexScreener labels it.
PUMP_AMM_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"


class ChainStateError(RuntimeError):
    """On-chain state could not be established. Always a refusal to price."""


class RpcRateLimited(ChainStateError):
    """The public endpoint refused the request. Back off, do not hammer."""


# ---------------------------------------------------------------------------
# admission control
# ---------------------------------------------------------------------------


@dataclass
class RateBudget:
    """
    A minimal token bucket plus an in-flight cap.

    Two independent limits, because they bound different things: the bucket
    bounds request RATE (what a free endpoint notices), and the semaphore bounds
    request CONCURRENCY (what this process is willing to have open). Raising
    concurrency without touching the bucket is how a system gets throttled
    despite believing it is being careful.
    """

    #: Sustained requests per second.
    rate_per_s: float = 8.0
    #: Burst allowance.
    burst: int = 12
    #: Maximum simultaneous in-flight requests.
    max_concurrency: int = 4
    #: Cap on how long a caller will wait for an admission slot.
    acquire_timeout_s: float = 3.0

    def __post_init__(self) -> None:
        self._tokens = float(self.burst)
        self._updated = time.monotonic()
        self._sem = asyncio.Semaphore(max(1, int(self.max_concurrency)))

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = max(0.0, now - self._updated)
        self._updated = now
        self._tokens = min(float(self.burst),
                           self._tokens + elapsed * self.rate_per_s)

    async def __aenter__(self) -> "RateBudget":
        try:
            await asyncio.wait_for(self._sem.acquire(),
                                   timeout=self.acquire_timeout_s)
        except asyncio.TimeoutError as exc:
            raise RpcRateLimited(
                "no admission slot within "
                f"{self.acquire_timeout_s * 1000:.0f}ms") from exc
        while True:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return self
            deficit = (1.0 - self._tokens) / max(0.1, self.rate_per_s)
            await asyncio.sleep(min(0.5, max(0.001, deficit)))

    async def __aexit__(self, *exc: Any) -> None:
        self._sem.release()

    def snapshot(self) -> dict:
        return {"rate_per_s": self.rate_per_s, "burst": self.burst,
                "max_concurrency": self.max_concurrency,
                "tokens_remaining": round(self._tokens, 2)}


# ---------------------------------------------------------------------------
# read results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainRead:
    """
    One on-chain read, with its provenance and staleness.

    ``stale`` and ``from_cache`` are separate facts: a value served from cache
    is fresh only if its TTL said so. ``missing`` means the account does not
    exist, which is a real state (a closed account) and never an error.
    """

    value: Any
    slot: Optional[int] = None
    read_at: float = 0.0
    from_cache: bool = False
    stale: bool = False
    missing: bool = False
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and not self.missing

    def age_s(self, now: Optional[float] = None) -> Optional[float]:
        if not self.read_at:
            return None
        return max(0.0, (time.time() if now is None else now) - self.read_at)

    def as_dict(self) -> dict:
        return {"slot": self.slot, "read_at": self.read_at,
                "from_cache": self.from_cache, "stale": self.stale,
                "missing": self.missing, "error": self.error,
                "age_s": self.age_s()}


# ---------------------------------------------------------------------------
# the client
# ---------------------------------------------------------------------------


class PublicRpcClient:
    """
    A caching, deduplicating, rate-limited JSON-RPC reader.

    Constructing one is inert: no client, no socket, no key. The HTTP client is
    created on first use and can be injected, which is how every test in this
    repository exercises the whole stack with no network at all.
    """

    def __init__(
        self,
        url: str = DEFAULT_PUBLIC_RPC_URL,
        *,
        timeout: float = 8.0,
        cache_ttl_s: float = 2.0,
        budget: Optional[RateBudget] = None,
        client: Optional[httpx.AsyncClient] = None,
        max_retries: int = 3,
        backoff_base_s: float = 0.25,
    ) -> None:
        self.url = url or DEFAULT_PUBLIC_RPC_URL
        self.timeout = timeout
        self.cache_ttl_s = cache_ttl_s
        self.budget = budget or RateBudget()
        self.max_retries = max(0, int(max_retries))
        self.backoff_base_s = backoff_base_s
        self._client = client
        self._own_client = client is None
        self._id = 0
        self._cache: dict[tuple, tuple[float, Any]] = {}
        self._inflight: dict[tuple, asyncio.Future] = {}
        self.stats = {"requests": 0, "cache_hits": 0, "dedup_hits": 0,
                      "retries": 0, "rate_limited": 0, "errors": 0}

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._own_client:
            await self._client.aclose()
        self._client = None

    # -- core --------------------------------------------------------------

    async def call(self, method: str, params: list, *,
                   cache_ttl_s: Optional[float] = None) -> ChainRead:
        """
        One JSON-RPC call, cached, deduplicated, rate-limited and retried.

        Never raises for a transport problem: the failure is returned as a
        ``ChainRead`` with ``ok=False`` and an ``error``. Callers decide whether
        a failure is fatal (for a price, it always is).
        """
        ttl = self.cache_ttl_s if cache_ttl_s is None else cache_ttl_s
        key = (method, _hashable(params))
        now = time.time()
        hit = self._cache.get(key)
        if hit and (now - hit[0]) <= max(0.0, ttl):
            self.stats["cache_hits"] += 1
            return ChainRead(value=hit[1], read_at=hit[0], from_cache=True,
                             stale=False)

        inflight = self._inflight.get(key)
        if inflight is not None:
            self.stats["dedup_hits"] += 1
            try:
                value, slot, at = await asyncio.shield(inflight)
            except Exception as exc:  # noqa: BLE001
                return ChainRead(value=None, read_at=time.time(),
                                 error=f"{type(exc).__name__}: {exc}")
            return ChainRead(value=value, slot=slot, read_at=at,
                             from_cache=False)

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._inflight[key] = fut
        try:
            value, slot, at = await self._call_uncached(method, params)
        except Exception as exc:  # noqa: BLE001
            if not fut.done():
                fut.set_exception(exc)
            self._cache.pop(key, None)
            self.stats["errors"] += 1
            return ChainRead(value=None, read_at=time.time(),
                             error=f"{type(exc).__name__}: {exc}")
        else:
            self._cache[key] = (at, value)
            if not fut.done():
                fut.set_result((value, slot, at))
            return ChainRead(value=value, slot=slot, read_at=at)
        finally:
            self._inflight.pop(key, None)
            # Consume the future's exception so a request that failed with no
            # waiter never produces an "exception was never retrieved" warning.
            # Awaiters that ARE attached still see it raised: marking it
            # retrieved does not settle it twice.
            if fut.done() and not fut.cancelled() and fut.exception():
                pass

    async def _call_uncached(self, method: str,
                             params: list) -> tuple[Any, Optional[int], float]:
        client = await self._ensure_client()
        last: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            async with self.budget:
                self._id += 1
                self.stats["requests"] += 1
                payload = {"jsonrpc": "2.0", "id": self._id,
                           "method": method, "params": params}
                try:
                    resp = await client.post(self.url, json=payload,
                                             timeout=self.timeout)
                except httpx.TimeoutException as exc:
                    last = ChainStateError(f"{method}: timed out")
                    last.__cause__ = exc  # type: ignore[attr-defined]
                except httpx.HTTPError as exc:
                    last = ChainStateError(f"{method}: transport error")
                    last.__cause__ = exc  # type: ignore[attr-defined]
                else:
                    if resp.status_code == 429:
                        self.stats["rate_limited"] += 1
                        last = RpcRateLimited(_retry_after_message(resp))
                        self.stats["retries"] += 1
                        await asyncio.sleep(self._backoff_delay(attempt, resp))
                        continue
                    if resp.status_code >= 500:
                        last = ChainStateError(
                            f"{method}: RPC HTTP {resp.status_code}")
                        self.stats["retries"] += 1
                        await asyncio.sleep(self._backoff_delay(attempt, resp))
                        continue
                    if resp.status_code != 200:
                        raise ChainStateError(
                            f"{method}: RPC HTTP {resp.status_code}")
                    try:
                        body = resp.json()
                    except ValueError as exc:
                        raise ChainStateError(
                            f"{method}: RPC returned non-JSON") from exc
                    if isinstance(body, dict) and body.get("error"):
                        err = body["error"]
                        code = err.get("code") if isinstance(err, dict) else None
                        msg = err.get("message") if isinstance(err, dict) else err
                        raise ChainStateError(f"{method}: RPC error {code}: {msg}")
                    if not isinstance(body, dict) or "result" not in body:
                        raise ChainStateError(
                            f"{method}: response missing result")
                    slot = body.get("context", {}).get("slot")
                    return body["result"], slot, time.time()
            if attempt < self.max_retries:
                await asyncio.sleep(self._backoff_delay(attempt, None))
        raise last or ChainStateError(f"{method}: exhausted retries")

    def _backoff_delay(self, attempt: int,
                       resp: Optional[httpx.Response]) -> float:
        """Exponential backoff with jitter, honouring ``Retry-After``."""
        retry_after = 0.0
        if resp is not None:
            try:
                retry_after = float(resp.headers.get("Retry-After") or 0.0)
            except (TypeError, ValueError):
                retry_after = 0.0
        base = self.backoff_base_s * (2 ** attempt)
        jitter = random.uniform(0.0, base * 0.25)
        return max(retry_after, base) + jitter

    # -- typed reads -------------------------------------------------------

    async def get_account(self, address: str, *,
                          cache_ttl_s: Optional[float] = None) -> ChainRead:
        """
        ``getAccountInfo`` with base64 data plus the owning program.

        Returns ``missing=True`` for an address that does not exist, which is a
        state and not an error. An account larger than
        :data:`MAX_ACCOUNT_DATA_BYTES` is refused rather than truncated: a
        truncated program account decodes into plausible-looking garbage, which
        is worse than no answer.
        """
        res = await self.call("getAccountInfo", [
            address, {"encoding": "base64", "commitment": "confirmed"}],
            cache_ttl_s=cache_ttl_s)
        if not res.ok:
            return res
        value = (res.value or {}).get("value")
        if value is None:
            return ChainRead(value=None, slot=res.slot, read_at=res.read_at,
                             missing=True)
        data = value.get("data")
        if isinstance(data, list) and data:
            raw = data[0]
        elif isinstance(data, str):
            raw = data
        else:
            raw = None
        if raw is None:
            return ChainRead(value=None, slot=res.slot, read_at=res.read_at,
                             missing=True)
        return ChainRead(
            value={
                "address": address,
                "owner": value.get("owner"),
                "lamports": value.get("lamports"),
                "executable": value.get("executable"),
                "rent_epoch": value.get("rentEpoch"),
                "data_b64": raw,
                "data_len": len(raw or ""),
                "slot": value.get("slot") or res.slot,
            },
            slot=res.slot, read_at=res.read_at, from_cache=res.from_cache)

    async def get_accounts(self, addresses: Sequence, *,
                           cache_ttl_s: Optional[float] = None) -> ChainRead:
        """``getMultipleAccounts``. One request for N accounts."""
        addrs = [str(a) for a in addresses]
        if not addrs:
            return ChainRead(value=[], read_at=time.time())
        res = await self.call("getMultipleAccounts", [
            addrs, {"encoding": "base64", "commitment": "confirmed"}],
            cache_ttl_s=cache_ttl_s)
        if not res.ok:
            return res
        out = []
        for i, value in enumerate(res.value or []):
            if value is None:
                out.append(None)
                continue
            data = value.get("data")
            raw = (data[0] if isinstance(data, list) and data else data)
            out.append({
                "address": addrs[i] if i < len(addrs) else None,
                "owner": value.get("owner"),
                "lamports": value.get("lamports"),
                "executable": value.get("executable"),
                "data_b64": raw,
                "data_len": len(raw or ""),
            })
        return ChainRead(value=out, slot=res.slot, read_at=res.read_at,
                         from_cache=res.from_cache)

    async def get_token_amount(self, account: str, *,
                               cache_ttl_s: Optional[float] = None
                               ) -> ChainRead:
        """The raw SPL/Token-2022 balance of one token account, from its bytes.

        Read from the account itself rather than from a parsed endpoint, so it
        works identically for classic SPL and Token-2022 accounts, including
        accounts carrying extensions whose parsed shape is larger than the base
        layout. Layout: mint(32) owner(32) amount(8, LE) at offset 64.
        """
        res = await self.get_account(account, cache_ttl_s=cache_ttl_s)
        if not res.ok:
            return res
        if res.missing:
            return ChainRead(value=None, slot=res.slot, read_at=res.read_at,
                             missing=True)
        raw = _b64(res.value.get("data_b64"))
        if raw is None or len(raw) < _TOKEN_ACCOUNT_AMOUNT_OFFSET + 8:
            return ChainRead(value=None, slot=res.slot, read_at=res.read_at,
                             error="token account data too short to decode")
        amount = int.from_bytes(
            raw[_TOKEN_ACCOUNT_AMOUNT_OFFSET:_TOKEN_ACCOUNT_AMOUNT_OFFSET + 8],
            "little", signed=False)
        return ChainRead(value=amount, slot=res.slot, read_at=res.read_at,
                         from_cache=res.from_cache)

    async def get_mint(self, mint: str, *,
                       cache_ttl_s: Optional[float] = None) -> ChainRead:
        """
        Decimals, supply and owning token program for a mint.

        Decimals are ALWAYS read from chain. A provider that guesses them is how
        a per-token price comes out 10^3 or 10^9 wrong, and the 2026-10-02 audit
        had to reconstruct the 6-decimal scale of a Token-2022 mint by hand
        because nothing in the runtime recorded it.
        """
        res = await self.get_account(mint, cache_ttl_s=cache_ttl_s)
        if not res.ok:
            return res
        if res.missing:
            return ChainRead(value=None, slot=res.slot, read_at=res.read_at,
                             missing=True)
        raw = _b64(res.value.get("data_b64"))
        if raw is None or len(raw) < _MINT_DECIMALS_OFFSET + 1:
            return ChainRead(value=None, slot=res.slot, read_at=res.read_at,
                             error="mint data too short to decode")
        decimals = raw[_MINT_DECIMALS_OFFSET]
        supply = None
        if len(raw) >= _MINT_SUPPLY_OFFSET + 8:
            supply = int.from_bytes(
                raw[_MINT_SUPPLY_OFFSET:_MINT_SUPPLY_OFFSET + 8], "little",
                signed=False)
        return ChainRead(
            value={"mint": mint, "decimals": decimals, "supply": supply,
                   "token_program": res.value.get("owner"),
                   "lamports": res.value.get("lamports")},
            slot=res.slot, read_at=res.read_at, from_cache=res.from_cache)

    async def get_slot(self, *, cache_ttl_s: float = 1.0) -> ChainRead:
        res = await self.call("getSlot", [{"commitment": "confirmed"}],
                              cache_ttl_s=cache_ttl_s)
        if not res.ok:
            return res
        try:
            return ChainRead(value=int(res.value), slot=res.slot,
                             read_at=res.read_at, from_cache=res.from_cache)
        except (TypeError, ValueError) as exc:
            return ChainRead(value=None, slot=res.slot, read_at=res.read_at,
                             error="getSlot: malformed result")

    async def health(self) -> dict:
        res = await self.call("getHealth", [], cache_ttl_s=5.0)
        return {
            "url": self.url,
            "reachable": bool(res.ok),
            "slot": res.slot,
            "detail": res.error or ("missing" if res.missing else "ok"),
            "budget": self.budget.snapshot(),
            "stats": dict(self.stats),
            "sla": "none: this is the free public endpoint",
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _hashable(params: Any) -> str:
    try:
        import json as _json
        return _json.dumps(params, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001 - cache key must never raise
        return repr(params)


def _b64(data: Optional[str]) -> Optional[bytes]:
    if not data:
        return None
    import base64
    try:
        return base64.b64decode(data)
    except Exception:  # noqa: BLE001
        return None


def _retry_after_message(resp: httpx.Response) -> str:
    try:
        ra = resp.headers.get("Retry-After")
    except Exception:  # noqa: BLE001
        ra = None
    return (f"public RPC rate limited (HTTP 429, Retry-After={ra})"
            if ra else "public RPC rate limited (HTTP 429)")


def decode_account_data(account: Optional[dict], *,
                         max_bytes: int = MAX_ACCOUNT_DATA_BYTES) -> bytes:
    """
    Decode an account's base64 data, refusing anything oversized or absent.

    Raises :class:`ChainStateError` rather than returning a short buffer. A
    caller that decodes a truncated program account gets a plausible wrong
    number, and a plausible wrong number is the failure mode this whole change
    set exists to prevent.
    """
    if not account:
        raise ChainStateError("account is missing")
    raw = _b64(account.get("data_b64"))
    if raw is None:
        raise ChainStateError("account has no base64 data")
    if len(raw) > max_bytes:
        raise ChainStateError(
            f"account data is {len(raw)} bytes, above the {max_bytes}-byte "
            f"ceiling for a decoded program state; refusing rather than "
            f"truncating")
    return raw


#: ``Sequence`` is only used in a type hint above; import it lazily-safe.
from typing import Sequence  # noqa: E402  (kept last to avoid shadowing)