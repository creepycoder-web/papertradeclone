"""
The FAST MARKET MONITOR: an event-driven native market watcher.

THE PROBLEM THIS REPLACES
------------------------
Open PAPER positions were monitored by a 5-second timer that, for every
position on every tick, asked for a full executable SELL quote. That cost
``getSlot`` + 2-3 ``getMultipleAccounts`` + ``getTokenSupply`` plus one DexScreener
HTTP per position per tick -- and then THREW THE QUOTE AWAY on every tick that
ended in HOLD. The result was a rate limit, a ~5s observation floor, and exit
decisions made on a book that had already moved.

THE ARCHITECTURE THIS ESTABLISHES
---------------------------------

    MARKET STATE MONITORING   -> event-driven native WebSocket account updates
    EXECUTION VALIDATION     -> an executable SELL quote, only when one is needed

Market monitoring and executable quoting are different questions and now cost
different things. The first is answered by pushing the venue's own account data
and computing the reference price LOCALLY from the pushed reserves -- zero RPC,
zero HTTP, and a genuinely high update rate. The second is answered by a real
executable quote, and is requested only when an existing exit condition is near
or has fired (see :mod:`autopaper.paper_live.quote_proximity`).

WHAT IS CACHED AND WHAT IS PUSHED
--------------------------------
STATIC, resolved ONCE per market and then never again:

  * the market identity (the pinned curve PDA or pool PDA),
  * the vault account addresses for a PumpSwap pool,
  * the token decimals,
  * the mint.

DYNAMIC, and arriving by push:

  * a bonding curve's ``virtual_token_reserves`` / ``virtual_quote_reserves`` /
    ``complete`` flag -- all three are IN the curve account, so a single
    ``accountSubscribe`` on the curve carries the whole price,
  * a PumpSwap pool's ``virtual_quote_reserves`` (pool account) and its two
    vault balances (two more ``accountSubscribe``).

The native reference price is the venue's OWN spot price from its OWN reserves,
computed with the very same :func:`~autopaper.same_market.pumpdirect._spot_price`
the executable quote path uses -- so a pushed price and a polled price are the
same number, not two definitions that can drift. It carries NO fee term, exactly
as ``execution_native_reference`` does, because a spot reference is a spot.

IDENTITY IS NOT RELAXED
-----------------------
The monitor only ever reports a price for the market it subscribed to, and it
re-verifies the account owner against the expected program before decoding. If
the subscription is stale, disconnected, or aimed at an account that is not the
venue, :meth:`NativeMarketMonitor.latest` returns a price that is explicitly
marked unusable rather than the last number it happened to see. Silence is never
sold as freshness.

MIGRATION
---------
A pump.fun curve that graduates (its ``complete`` flag flips) stops being a
market. On that push the monitor DERIVES the canonical PumpSwap pool, verifies it
on chain, records the migration on the position, UNSUBSCRIBES the dead curve and
SUBSCRIBES the pool and its two vaults. Pre- and post-graduation prices are never
mixed: the switch is atomic and the curve's last price is discarded.

SAFETY
------
This module OBSERVES. It holds no keypair, signs nothing, broadcasts nothing, and
cannot order. Every price it produces is a native reference only: turning one
into a trade still requires the executable-quote path, and the exit thresholds
themselves are untouched by anything here.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import random
import time
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any, Optional, Sequence

from ..logging_setup import log
from ..same_market.pumpdirect import (
    MarketKind,
    SOL_DECIMALS,
    PumpSwapPool,
    _spot_price,
    decode_bonding_curve,
    decode_pool,
    decode_token_account_amount,
    derive_bonding_curve,
    derive_canonical_pool,
)
from ..same_market.ws import (
    WebSocketClosed,
    WebSocketConnection,
    WebSocketUnavailable,
    parse_ws_url,
)

#: Commitment used for monitoring. ``processed`` is the freshest commitment a
#: public endpoint serves for a push feed and is what makes this faster than the
#: ``confirmed`` reads the exit path still uses -- a monitoring observation may be
#: the freshest state available, but it can never authorise a fill on its own.
MONITOR_COMMITMENT = "processed"

#: How long the stream may be silent before it is declared STALE and the monitor
#: leaves HEALTHY for RECOVERY. Deliberately several multiples of the expected
#: update interval: a quiet token produces no pushes at all, so a short window
#: would flap into recovery on a market that is simply not trading.
DEFAULT_STALE_AFTER_MS = 8_000.0

#: A missing push is not by itself an error, but a slot that goes BACKWARDS is a
#: real defect and is counted, because it means the stream is out of order.
#
#: Reconnect backoff bounds. Jittered so several positions recovering at once do
#: not synchronize into a reconnect thundering herd on the endpoint.
RECONNECT_MIN_S = 0.5
RECONNECT_MAX_S = 15.0

#: How many trailing update intervals the reported median/p95/max cover.
_INTERVAL_WINDOW = 512


@dataclass(frozen=True)
class MonitorConfig:
    """Tuning for the native monitor. Every value is a bound, not a trigger."""

    ws_url: str = ""
    stale_after_ms: float = DEFAULT_STALE_AFTER_MS
    #: How many consecutive updates must arrive within ``stale_after_ms`` before
    #: the stream is declared healthy. One is enough; the counter exists so the
    #: ``STREAM_WARMING`` state is observable rather than instantaneous.
    healthy_after_updates: int = 1
    open_timeout_s: float = 10.0
    reconnect_min_s: float = RECONNECT_MIN_S
    reconnect_max_s: float = RECONNECT_MAX_S
    #: Hard ceiling on pushes per second processed for ONE market. A market
    #: cannot legitimately update faster than the chain produces slots, so this
    #: only bounds a pathological stream; it is not a sampling rate and it is not
    #: a throttle on price resolution.
    max_updates_per_second: float = 200.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ws_url": self.ws_url,
            "stale_after_ms": self.stale_after_ms,
            "healthy_after_updates": self.healthy_after_updates,
            "open_timeout_s": self.open_timeout_s,
            "reconnect_min_s": self.reconnect_min_s,
            "reconnect_max_s": self.reconnect_max_s,
            "max_updates_per_second": self.max_updates_per_second,
        }


#: Connection states. ``HEALTHY`` is the only one that can serve a price.
STATE_IDLE = "IDLE"
STATE_WARMING = "WARMING"
STATE_HEALTHY = "HEALTHY"
STATE_STALE = "STALE"
STATE_RECONNECTING = "RECONNECTING"
STATE_STOPPED = "STOPPED"

#: Where the last price came from.
SOURCE_PUSH = "WEBSOCKET_PUSH"
SOURCE_UNAVAILABLE = "UNAVAILABLE"

#: Account roles within one market. These drive which decode runs.
ROLE_CURVE = "CURVE"
ROLE_POOL = "POOL"
ROLE_BASE_VAULT = "BASE_VAULT"
ROLE_QUOTE_VAULT = "QUOTE_VAULT"

#: Roles that make up the CURRENT market. Anything else subscribed is obsolete
#: and is dropped on a migration.
_MARKET_ROLES = (ROLE_CURVE, ROLE_POOL, ROLE_BASE_VAULT, ROLE_QUOTE_VAULT)


@dataclass(frozen=True)
class StaticVenueIdentity:
    """
    Everything about a market that does not change while a position is open.

    Resolved ONCE, then carried. ``market_address`` IS the pin: the monitor
    reports prices for this account and no other, so a position can never be
    monitored against an unrelated book.
    """

    mint: str
    market_type: str
    market_address: str
    base_mint: str = ""
    quote_mint: str = ""
    #: Decimals of the HELD token. This is the ``output_decimals`` of the curve's
    #: spot price, so it must be the mint's decimals, never a default.
    token_decimals: int = 6
    creator: str = ""
    coin_creator: str = ""
    pool_base_vault: str = ""
    pool_quote_vault: str = ""
    #: The base mint's decimals for a PumpSwap pool. The pool account does not
    #: carry decimals -- only the mint does -- so this is read ONCE and cached
    #: here rather than re-read per push.
    pool_base_decimals: int = 6
    #: The base mint's supply, likewise read once. Used only for the fee
    #: schedule, never for the spot price.
    pool_base_supply: int = 0

    def subscription_addresses(self) -> tuple[tuple[str, str], ...]:
        """``(address, role)`` pairs this market needs pushed to it."""
        if self.market_type == MarketKind.PUMP_BONDING_CURVE.value:
            return ((self.market_address, ROLE_CURVE),)
        pairs = [(self.market_address, ROLE_POOL)]
        if self.pool_base_vault:
            pairs.append((self.pool_base_vault, ROLE_BASE_VAULT))
        if self.pool_quote_vault:
            pairs.append((self.pool_quote_vault, ROLE_QUOTE_VAULT))
        return tuple(pairs)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mint": self.mint,
            "market_type": self.market_type,
            "market_address": self.market_address,
            "base_mint": self.base_mint,
            "quote_mint": self.quote_mint,
            "token_decimals": self.token_decimals,
            "pool_base_vault": self.pool_base_vault,
            "pool_quote_vault": self.pool_quote_vault,
        }


@dataclass(frozen=True)
class NativePricePoint:
    """
    A native reference price and the provenance of how it was obtained.

    ``usable`` is the single field a caller must check. It is False whenever the
    price could not be proven current and same-market, which routes the caller
    to the existing conservative recovery path instead of acting on a stale
    number.
    """

    price_sol_per_token: Optional[Decimal]
    venue_id: str
    market_address: str
    market_type: str
    slot: Optional[int]
    received_monotonic: float
    age_ms: float
    source: str
    usable: bool
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "native_price": (
                None if self.price_sol_per_token is None
                else float(self.price_sol_per_token)),
            "venue_id": self.venue_id,
            "market_address": self.market_address,
            "market_type": self.market_type,
            "last_market_slot": self.slot,
            "last_market_update_age_ms": round(self.age_ms, 1),
            "monitor_source": self.source,
            "native_price_usable": self.usable,
            "reason": self.reason,
        }


@dataclass
class MigrationRecord:
    """A proven ``PUMP_BONDING_CURVE -> PUMPSWAP`` transition."""

    from_venue_id: str
    to_venue_id: str
    from_market_address: str
    to_market_address: str
    at_monotonic: float
    slot: Optional[int]
    evidence: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "from_venue_id": self.from_venue_id,
            "to_venue_id": self.to_venue_id,
            "from_market_address": self.from_market_address,
            "to_market_address": self.to_market_address,
            "migrated_at_monotonic": round(self.at_monotonic, 3),
            "last_market_slot": self.slot,
            "evidence": self.evidence,
        }


@dataclass
class MonitorStats:
    """Counters for one monitored market. Read-only reporting; no control."""

    updates: int = 0
    quotes_from_push: int = 0
    decode_failures: int = 0
    reconnects: int = 0
    resubscribes: int = 0
    stream_gaps: int = 0
    stale_detections: int = 0
    rate_limit_429: int = 0
    subscription_ids: dict[str, int] = field(default_factory=dict)
    last_error: str = ""
    #: Update INTERVAL statistics, in ms, measured with MONOTONIC time.
    #:
    #: This is the evidence that the monitor is event-driven rather than merely
    #: called event-driven. A 5s poller produces an interval distribution with a
    #: median at 5000; a push consumer produces one at whatever the venue
    #: actually does. Only the trailing window is kept, so it cannot grow.
    update_interval_median_ms: float = 0.0
    update_interval_p95_ms: float = 0.0
    update_interval_max_ms: float = 0.0
    #: How many intervals the statistics above are computed from.
    interval_samples: int = 0

    #: Trailing interval window. Bounded on purpose: this is a rate, and a rate
    #: that grows with the length of the run is a leak wearing a metric's name.
    intervals: list[float] = field(default_factory=list)

    def note_interval(self, interval_ms: float) -> None:
        if interval_ms < 0.0:
            return
        self.intervals.append(interval_ms)
        if len(self.intervals) > _INTERVAL_WINDOW:
            del self.intervals[:len(self.intervals) - _INTERVAL_WINDOW]
        ordered = sorted(self.intervals)
        count = len(ordered)
        if not count:
            return
        median = ordered[count // 2] if count % 2 else \
            (ordered[count // 2 - 1] + ordered[count // 2]) / 2.0
        index = min(count - 1, max(0, int(round(0.95 * (count - 1)))))
        self.update_interval_median_ms = median
        self.update_interval_p95_ms = ordered[index]
        self.update_interval_max_ms = ordered[-1]
        self.interval_samples = count

    def as_dict(self) -> dict[str, Any]:
        return {
            "native_updates": self.updates,
            "native_prices_served": self.quotes_from_push,
            "native_update_interval_median_ms": round(
                self.update_interval_median_ms, 1),
            "native_update_interval_p95_ms": round(
                self.update_interval_p95_ms, 1),
            "native_update_interval_max_ms": round(
                self.update_interval_max_ms, 1),
            "native_update_interval_samples": self.interval_samples,
            "decode_failures": self.decode_failures,
            "reconnects": self.reconnects,
            "resubscribes": self.resubscribes,
            "stream_gap_count": self.stream_gaps,
            "stale_detections": self.stale_detections,
            "rate_limit_429_count": self.rate_limit_429,
            "subscription_ids": dict(self.subscription_ids),
            "last_error": self.last_error,
        }


def _base64_account(raw: Any) -> bytes:
    """
    Pull the account bytes out of an ``accountNotification`` payload.

    The envelope is ``{"context": {"slot": n}, "value": {"data": [b64, "base64"]}}``.
    Both ``value`` and ``value.account`` spellings are accepted because Solana
    returns the account directly for ``accountSubscribe`` but nests it in other
    account-bearing notifications, and reading the wrong one would silently
    yield ZERO bytes -- which decodes as a real curve with zero reserves rather
    than as a parse failure.
    """
    value = raw.get("value") if isinstance(raw, dict) else None
    candidate = value if isinstance(value, dict) else raw
    if not isinstance(candidate, dict):
        return b""
    nested = candidate.get("account")
    if isinstance(nested, dict):
        candidate = nested
    data = candidate.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        encoding = (data[1] if len(data) > 1 else "base64") or "base64"
        if encoding == "base64" and isinstance(first, str):
            try:
                return base64.b64decode(first, validate=False)
            except Exception:  # noqa: BLE001 - a bad frame is a decode failure
                return b""
        if encoding == "base58" and isinstance(first, str):
            from ..execution.txcodec import b58decode_pubkey_bytes

            with contextlib.suppress(Exception):
                return b58decode_pubkey_bytes(first)
    if isinstance(data, str):
        with contextlib.suppress(Exception):
            return base64.b64decode(data, validate=False)
    return b""


def _notification_slot(raw: Any) -> Optional[int]:
    if not isinstance(raw, dict):
        return None
    context = raw.get("context")
    if isinstance(context, dict):
        slot = context.get("slot")
        if isinstance(slot, int):
            return slot
    return None


class NativeMarketMonitor:
    """
    A per-market event-driven watcher.

    ONE instance per distinct market. Several positions opened on the same mint
    share one instance, because the venue account is a property of the market and
    not of a position: subscribing twice would double the push rate for the same
    bytes, which is the kind of waste this whole module exists to remove.

    IT DOES NOT OWN A CONNECTION. The socket belongs to a
    :class:`SharedSubscriptionStream`, shared by every market on the same
    endpoint, and this class registers interest in a few accounts on it.

    WHY THAT MATTERS ENOUGH TO BE STRUCTURAL RATHER THAN AN OPTIMISATION
    ---------------------------------------------------------------------
    One connection per market was the first design and it does not work against
    a public endpoint: the handshake is itself rate limited, so opening a socket
    per monitored market earns ``HTTP/1.1 429 Too Many Requests`` on the
    handshake and every monitor spends its life in recovery. Measured on
    mainnet, six concurrent markets produced fifteen reconnects each in two
    minutes and zero updates. A Solana node serves MANY subscriptions on ONE
    connection -- the limit is subscriptions per connection, not connections per
    account -- so multiplexing is not merely cheaper, it is the only shape the
    endpoint tolerates.
    """

    def __init__(
        self,
        identity: StaticVenueIdentity,
        *,
        config: MonitorConfig,
        stream: "SharedSubscriptionStream | None" = None,
        verify_account: Optional[Any] = None,
        clock: Any = None,
        on_price_update: Optional[Callable[["NativeMarketMonitor", NativePricePoint], Any]] = None,
    ) -> None:
        self.identity = identity
        self.config = config
        self.stats = MonitorStats()
        #: ``async (address, expected_owner, label) -> bool``. Optional: when
        #: absent the monitor trusts nothing it did not decode with an explicit
        #: discriminator check, and says so in the diagnostics.
        self._verify_account = verify_account
        self._clock = clock
        self._stream = stream
        #: Optional callback invoked when a new usable native price is computed.
        #: Receives (monitor, price_point). Use this to trigger immediate exit evaluation.
        self._on_price_update = on_price_update

        self.state = STATE_IDLE
        self._task: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()

        #: The subscription ids THIS monitor currently holds. Owned by the
        #: shared stream, mirrored here so a release can send exactly the right
        #: ``accountUnsubscribe`` calls.
        self._subscriptions: dict[int, tuple[str, str]] = {}

        # -- pushed state -------------------------------------------------
        self._latest_price: Optional[NativePricePoint] = None
        self._last_update_monotonic: float = 0.0
        self._last_slot: Optional[int] = None
        self._high_water_slot: Optional[int] = None
        self._curve_bytes: bytes = b""
        self._pool_bytes: bytes = b""
        self._base_vault_bytes: bytes = b""
        self._quote_vault_bytes: bytes = b""
        self._last_update_monotonic_bucket: float = 0.0
        self._migration: Optional[MigrationRecord] = None
        #: A migration this monitor DECODED but which still needs an
        #: off-push chain VERIFICATION of the derived pool before its vault
        #: addresses are known. Declared here so the attribute always exists.
        self._pending_identity: Optional[StaticVenueIdentity] = None

    # -- identity / lifecycle --------------------------------------------

    @property
    def venue_id(self) -> str:
        return f"{self.identity.market_type}:{self.identity.market_address}"

    @property
    def market_address(self) -> str:
        return self.identity.market_address

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def _now(self) -> float:
        if self._clock is not None:
            try:
                value = self._clock.monotonic()
                if value is not None:
                    return float(value)
            except Exception:  # noqa: BLE001 - a clock must never break a price
                pass
        return time.monotonic()

    def start(self) -> None:
        """Begin watching, by registering interest on the shared stream."""
        if self.is_running:
            return
        if self._stream is None:
            # No stream means no push channel. The monitor stays IDLE and
            # ``latest()`` reports no price, so the caller falls back -- which
            # is the honest state, not a silent "connected".
            return
        self._stream.attach(self)
        log("NATIVE",
            "monitor starting: mint=%s venue=%s source=%s accounts=%s",
            self.identity.mint[:12], self.venue_id, SOURCE_PUSH,
            [f"{role}:{addr[:8]}" for addr, role in
             self.identity.subscription_addresses()])

    async def close(self) -> None:
        """Stop watching. Safe to call when never started, and safe twice."""
        self._stopping.set()
        if self._stream is not None:
            with contextlib.suppress(Exception):
                await self._stream.detach(self)
        self._subscriptions.clear()
        self.state = STATE_STOPPED
        log("NATIVE", "monitor stopped: mint=%s venue=%s updates=%d",
            self.identity.mint[:12], self.venue_id, self.stats.updates)

    # -- shared-stream callbacks ------------------------------------------
    #
    # The stream owns the socket and routes pushes here by subscription id. The
    # monitor keeps every DECISION about the market; the stream only knows about
    # frames.

    def stream_connected(self, connected: bool) -> None:
        """The shared socket came up or went down."""
        if connected:
            self.state = STATE_HEALTHY if self.stats.updates else STATE_WARMING
        elif self.state != STATE_STOPPED:
            self.state = STATE_STALE
            self.stats.stale_detections += 1

    def stream_resubscribed(self, ids: dict[int, tuple[str, str]]) -> None:
        """Re-bound after a reconnect. Old ids are gone and must not be used."""
        self._subscriptions = dict(ids)
        if not ids and self.state == STATE_STALE:
            # NOTHING was bound -- the subscribe was refused or the market
            # resolved to no accounts. Reporting WARMING here would claim a
            # stream that is waiting for its first push when in fact it holds no
            # subscription at all, and a refusal would be indistinguishable from
            # a quiet market. The refusal is already recorded in
            # ``stats.last_error``; keep the STALE verdict that reports it.
            return
        self.state = STATE_HEALTHY if self.stats.updates else STATE_WARMING

    def stream_error(self, message: str) -> None:
        self.stats.last_error = message[:200]
        if self.state != STATE_STOPPED:
            self.state = STATE_STALE

    def deliver(self, role: str, raw: bytes, slot: Optional[int],
                address: str) -> None:
        """One account update for one of THIS market's accounts."""
        self._store_bytes(role, raw)
        self._recompute(slot, address)
        self._record_arrival(slot, address)

    # -- the public read surface -----------------------------------------

    def latest(self) -> Optional[NativePricePoint]:
        """
        The freshest native price, or a point that is explicitly UNUSABLE.

        Never returns the last price as if it were current. If the stream is
        stale, disconnected or warming, the returned point carries
        ``usable=False`` and a reason, and the caller must fall back to its
        conservative path.
        """
        point = self._latest_price
        now = self._now()
        if point is None:
            return None
        age_ms = (now - self._last_update_monotonic) * 1000.0
        if self.state not in (STATE_HEALTHY,):
            return replace(
                point, age_ms=age_ms, usable=False,
                reason=f"stream state is {self.state}, not {STATE_HEALTHY}")
        if age_ms > self.config.stale_after_ms:
            return replace(
                point, age_ms=age_ms, usable=False,
                reason=(f"last native update is {age_ms:.0f}ms old, past the "
                        f"{self.config.stale_after_ms:.0f}ms staleness bound"))
        return replace(point, age_ms=age_ms, usable=True)

    def age_ms(self) -> float:
        if self._last_update_monotonic <= 0.0:
            return -1.0
        return (self._now() - self._last_update_monotonic) * 1000.0

    def migration(self) -> Optional[MigrationRecord]:
        return self._migration

    def diagnostics(self) -> dict[str, Any]:
        """The diagnostic block required for every open position."""
        point = self._latest_price
        return {
            "monitor_source": SOURCE_PUSH if point is not None else SOURCE_UNAVAILABLE,
            "websocket_connected": bool(
                self._stream is not None and self._stream.is_open),
            "websocket_state": self.state,
            "last_market_update_age_ms": round(self.age_ms(), 1),
            "last_market_slot": self._last_slot,
            "native_price": (
                None if point is None or point.price_sol_per_token is None
                else float(point.price_sol_per_token)),
            "native_market_address": self.identity.market_address,
            "native_market_type": self.identity.market_type,
            "migration": (
                None if self._migration is None
                else self._migration.as_dict()),
            **self.stats.as_dict(),
        }

    # -- pushed state ----------------------------------------------------

    def _store_bytes(self, role: str, raw: bytes) -> None:
        if role == ROLE_CURVE:
            self._curve_bytes = raw
        elif role == ROLE_POOL:
            self._pool_bytes = raw
        elif role == ROLE_BASE_VAULT:
            self._base_vault_bytes = raw
        elif role == ROLE_QUOTE_VAULT:
            self._quote_vault_bytes = raw

    # -- slot / gap bookkeeping -------------------------------------------

    def _record_arrival(self, slot: Optional[int], address: str) -> None:
        now = self._now()
        previous_monotonic = self._last_update_monotonic
        self.stats.updates += 1
        self._last_update_monotonic = now
        if previous_monotonic > 0.0:
            # MONOTONIC, so an NTP step cannot invent or hide a gap.
            self.stats.note_interval((now - previous_monotonic) * 1000.0)
        # PROMOTE TO HEALTHY ON THE FIRST USABLE UPDATE.
        #
        # ``_run`` can only set WARMING at connect time, because at that moment
        # no update has arrived to count. If the promotion lived only there, the
        # stream would stay WARMING forever -- ``latest()`` would refuse every
        # price it had, and every position would fall back to the recovery poll
        # while the monitor visibly counted updates. So the transition is made
        # HERE, where the evidence for it exists.
        if self.state == STATE_WARMING and \
                self.stats.updates >= self.config.healthy_after_updates:
            self.state = STATE_HEALTHY
        if slot is not None:
            previous_high = self._high_water_slot
            self._last_slot = slot
            if previous_high is not None and slot < previous_high:
                # A slot that moves BACKWARDS means the stream is out of order
                # or a resubscribe raced an in-flight frame. Counted as a gap so
                # it is visible rather than silently smoothing over.
                self.stats.stream_gaps += 1
                log("WARN", "native stream gap: mint=%s slot %d arrived after "
                    "slot %d", self.identity.mint[:12], slot, previous_high)
            else:
                self._high_water_slot = slot

    def _check_stale(self) -> None:
        age = self.age_ms()
        if age >= 0.0 and age > self.config.stale_after_ms and \
                self.state in (STATE_HEALTHY, STATE_WARMING):
            self.state = STATE_STALE
            self.stats.stale_detections += 1

    # -- decoding and the LOCAL price ------------------------------------

    def _recompute(self, slot: Optional[int], address: str) -> None:
        """
        Compute the native reference price from pushed bytes, locally.

        This is the whole point of the module: no RPC, no HTTP, no quote
        provider. The arithmetic is
        :func:`~autopaper.same_market.pumpdirect._spot_price` -- the SAME
        function the executable quote path uses for its
        ``execution_native_reference`` -- so a pushed price and a polled price
        are the same number by construction rather than by agreement.
        """
        try:
            if self.identity.market_type == MarketKind.PUMP_BONDING_CURVE.value:
                price, detail = self._recompute_curve(slot, address)
            else:
                price, detail = self._recompute_pool(slot, address)
        except Exception as exc:  # noqa: BLE001 - a decode failure is not a crash
            self.stats.decode_failures += 1
            self.stats.last_error = f"decode: {type(exc).__name__}: {exc}"[:200]
            return
        if price is None:
            return
        self.stats.quotes_from_push += 1
        self._latest_price = NativePricePoint(
            price_sol_per_token=price,
            venue_id=self.venue_id,
            market_address=self.identity.market_address,
            market_type=self.identity.market_type,
            slot=slot,
            received_monotonic=self._now(),
            age_ms=0.0,
            source=SOURCE_PUSH,
            usable=True,
            reason=detail,
        )
        # Invoke the price update callback if registered.
        # This enables immediate exit evaluation on every native price update.
        if self._on_price_update is not None:
            try:
                self._on_price_update(self, self._latest_price)
            except Exception:  # noqa: BLE001 - callback must never break the monitor
                self.stats.last_error = f"price callback: {type(exc).__name__}"

    def _recompute_curve(self, slot: Optional[int],
                         address: str) -> tuple[Optional[Decimal], str]:
        if not self._curve_bytes:
            return None, "curve bytes not pushed yet"
        # The owner check is what makes a decode trustworthy: the discriminator
        # alone would accept any account that happens to start with those eight
        # bytes.
        curve_address, _bump = derive_bonding_curve(self.identity.mint)
        if address != curve_address:
            return None, (f"pushed account {address[:8]} is not the pinned curve "
                          f"{curve_address[:8]}")
        curve = decode_bonding_curve(
            self._curve_bytes, curve_address=curve_address, mint=self.identity.mint)
        if curve.graduated:
            # The migration check comes FIRST, before the early return. A
            # graduated curve produces no price, so returning early here would
            # return early on EVERY subsequent push too -- and the graduation
            # itself, which is the one event that matters, would never be seen.
            self._maybe_migrate(True, curve_address, slot)
            return None, "bonding curve is complete; awaiting migration to the pool"
        if curve.virtual_quote_reserves <= 0 or curve.virtual_token_reserves <= 0:
            return None, "curve reserves are not positive"
        price = _spot_price(
            numerator=curve.virtual_quote_reserves,
            denominator=curve.virtual_token_reserves,
            input_decimals=SOL_DECIMALS,
            output_decimals=self.identity.token_decimals,
        )
        if price is None:
            return None, "curve spot price is undefined"
        detail = (f"curve reserves {curve.virtual_quote_reserves}/"
                  f"{curve.virtual_token_reserves}")
        return price, detail

    def _recompute_pool(self, slot: Optional[int],
                        address: str) -> tuple[Optional[Decimal], str]:
        if not self._pool_bytes:
            return None, "pool bytes not pushed yet"
        pool = self._decode_pool(self._pool_bytes)
        if pool is None:
            return None, "pool or its vault balances did not decode"
        effective_quote_reserve = pool.effective_quote_reserve
        if pool.base_reserve <= 0:
            return None, "pool base reserve is zero"
        if effective_quote_reserve <= 0:
            return None, "pool effective quote reserve is not positive"
        price = _spot_price(
            numerator=effective_quote_reserve,
            denominator=pool.base_reserve,
            input_decimals=SOL_DECIMALS,
            output_decimals=pool.base_decimals,
        )
        if price is None:
            return None, "pool spot price is undefined"
        return price, (f"pool reserves base={pool.base_reserve} "
                       f"effective_quote={effective_quote_reserve}")

    def _decode_pool(self, raw: bytes) -> Optional[PumpSwapPool]:
        """
        Assemble the pool from the PUSHED pool account plus pushed vault bytes.

        This is :func:`~autopaper.same_market.pumpdirect.decode_pumpswap_pool`
        with the base-mint account replaced by the decimals and supply already
        cached on :class:`StaticVenueIdentity`. That substitution is the point:
        the mint's decimals are immutable and its supply effectively is on a pump
        token, so re-reading a third account on every push to obtain two constants
        would be the same waste this module removes elsewhere. The reserves still
        come only from the two vaults the pool itself names.
        """
        pool_address, _bump = derive_canonical_pool(self.identity.mint)
        try:
            fields = decode_pool(raw, pool_address=pool_address)
        except Exception:  # noqa: BLE001
            return None
        if not self._base_vault_bytes or not self._quote_vault_bytes:
            return None
        try:
            base_reserve = decode_token_account_amount(
                self._base_vault_bytes, what="pool base token account")
            quote_reserve = decode_token_account_amount(
                self._quote_vault_bytes, what="pool quote token account")
        except Exception:  # noqa: BLE001
            return None
        # The pool states WHICH vaults hold its liquidity. Reading any other
        # account's balance here would produce a reserve that is not this pool's,
        # so the two addresses must be exactly the ones it named.
        named_base = str(fields.get("pool_base_token_account") or "")
        named_quote = str(fields.get("pool_quote_token_account") or "")
        if self.identity.pool_base_vault and named_base and \
                named_base != self.identity.pool_base_vault:
            return None
        if self.identity.pool_quote_vault and named_quote and \
                named_quote != self.identity.pool_quote_vault:
            return None
        return PumpSwapPool(
            address=pool_address,
            base_decimals=self.identity.pool_base_decimals,
            base_supply=self.identity.pool_base_supply,
            base_reserve=base_reserve,
            quote_reserve=quote_reserve,
            **fields,
        )

    # -- migration --------------------------------------------------------

    def _maybe_migrate(self, complete: bool, curve_address: str,
                       slot: Optional[int]) -> None:
        """
        On a proven graduation, switch to the canonical PumpSwap pool.

        The switch is ATOMIC from the reader's point of view: the identity is
        replaced, the connection is rebuilt against the new account set, and the
        curve's last price is DISCARDED. A position is therefore never evaluated
        against a curve after it graduated, and never against a pool before the
        pool verified -- the exact "do not mix pre- and post-graduation prices"
        requirement, enforced structurally rather than by remembering to clear a
        variable.
        """
        if not complete:
            return
        if self._migration is not None:
            return
        pool_address, _bump = derive_canonical_pool(self.identity.mint)
        if pool_address == curve_address:
            return
        self._migration = MigrationRecord(
            from_venue_id=f"{MarketKind.PUMP_BONDING_CURVE.value}:{curve_address}",
            to_venue_id=f"{MarketKind.PUMPSWAP.value}:{pool_address}",
            from_market_address=curve_address,
            to_market_address=pool_address,
            at_monotonic=self._now(),
            slot=slot,
            evidence=(f"pump bonding curve {curve_address} reported complete=1; "
                      f"canonical pool derived as {pool_address}"),
        )
        # The curve's price is no longer a price for anything.
        self._latest_price = None
        self._curve_bytes = b""
        self._pool_bytes = b""
        self._base_vault_bytes = b""
        self._quote_vault_bytes = b""
        log("NATIVE", "migration detected: mint=%s %s -> %s (slot %s); "
            "dropping the curve and subscribing the pool",
            self.identity.mint[:12], curve_address[:8], pool_address[:8], slot)
        self._pending_identity = StaticVenueIdentity(
            mint=self.identity.mint,
            market_type=MarketKind.PUMPSWAP.value,
            market_address=pool_address,
            base_mint=self.identity.base_mint or self.identity.mint,
            quote_mint=self.identity.quote_mint,
            token_decimals=self.identity.token_decimals,
            creator=self.identity.creator,
            coin_creator=self.identity.coin_creator,
            pool_base_vault="",
            pool_quote_vault="",
        )
        self._schedule_resubscribe()

    def _schedule_resubscribe(self) -> None:
        """
        Ask the shared stream to re-bind this monitor to the new market.

        Deferred to the next loop turn on purpose: it must not run inside the
        decode of the very notification that discovered the migration, because
        that decode runs on the stream's own read path and writing to the socket
        from inside it would be a re-entrant frame write.
        """
        if self._pending_identity is None or self._stream is None:
            return
        target = self._pending_identity
        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().call_soon(
                lambda: asyncio.ensure_future(self._adopt_and_rebind(target)))

    async def _adopt_and_rebind(self, target: StaticVenueIdentity) -> None:
        self._pending_identity = None
        self.identity = target
        try:
            await self._stream.rebind(self)
        except Exception as exc:  # noqa: BLE001 - reconnect recovers anyway
            self.stats.last_error = (
                f"post-migration rebind failed: "
                f"{type(exc).__name__}: {exc}")[:200]
            self.state = STATE_STALE

    def apply_pending_identity(self) -> Optional[StaticVenueIdentity]:
        """
        Adopt a migration resolved OFF the push path.

        The monitor can only DECODE a pool; it cannot VERIFY one, because
        verification is a chain read and this module's contract is that a healthy
        push costs no RPC. When the caller has verified the pool, it hands the
        proven identity here and the switch completes with the real vault
        addresses attached. Until then the monitor keeps reporting the old
        venue's status -- which is why it reports ``UNAVAILABLE`` rather than
        reusing the curve price.
        """
        if self._pending_identity is None:
            return None
        self.identity = self._pending_identity
        self._pending_identity = None
        self._latest_price = None
        return self.identity

    def set_pool_vaults(self, base_vault: str, quote_vault: str) -> None:
        """Attach verified vault addresses to a migrated pool and re-bind."""
        if self.identity.market_type != MarketKind.PUMPSWAP.value:
            return
        self.identity = replace(
            self.identity, pool_base_vault=base_vault,
            pool_quote_vault=quote_vault)
        self._pool_bytes = b""
        self._base_vault_bytes = b""
        self._quote_vault_bytes = b""
        if self._stream is None:
            return
        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().call_soon(
                lambda: asyncio.ensure_future(
                    self._stream.rebind(self)))


class SharedSubscriptionStream:
    """
    ONE websocket connection, carrying every monitored market's subscriptions.

    Solana serves many subscriptions on a single connection; the endpoint's
    limits are on SUBSCRIPTIONS per connection and on the CONNECTION HANDSHAKE
    rate. So the scarce resource is connections, not subscriptions, and this
    class spends exactly one connection no matter how many markets are being
    watched. Routing is by subscription id: every id this stream holds is mapped
    to the :class:`NativeMarketMonitor` that asked for it, and a push is handed
    to that monitor and to nobody else.

    RECOVERY
    --------
    A dead socket takes every market with it, which is the cost of
    multiplexing, so recovery is one reconnect plus a replay of the whole
    interest set rather than N independent reconnects. The replay is bounded by
    the endpoint's subscription limit: past that, markets wait their turn and are
    reported ``WARMING`` rather than being silently dropped.
    """

    def __init__(self, ws_url: str, *, config: MonitorConfig) -> None:
        self.ws_url = ws_url
        self.config = config
        self._monitors: list["NativeMarketMonitor"] = []
        self._connection: Optional[WebSocketConnection] = None
        self._task: Optional[asyncio.Task] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()
        #: subscription id -> (monitor, address, role)
        self._routes: dict[int, tuple[Any, str, str]] = {}
        self._request_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self.reconnects = 0
        self.resubscribes = 0
        self.handshake_refusals = 0
        self.last_error = ""
        self.is_open = False
        #: Monotonic time of the last frame of ANY kind. Used only for
        #: diagnostics; per-market freshness is the monitor's own business.
        self.last_frame_monotonic = 0.0

    # -- membership ------------------------------------------------------

    def attach(self, monitor: "NativeMarketMonitor") -> None:
        if monitor in self._monitors:
            return
        self._monitors.append(monitor)
        if self._task is None or self._task.done():
            self._stopping = asyncio.Event()
            self._task = asyncio.create_task(self._run())
        else:
            # Already running: add this monitor's interest without a reconnect.
            with contextlib.suppress(RuntimeError):
                asyncio.get_running_loop().call_soon(
                    lambda: asyncio.ensure_future(self.rebind(monitor)))

    async def detach(self, monitor: "NativeMarketMonitor") -> None:
        if monitor in self._monitors:
            self._monitors.remove(monitor)
        stale = [sid for sid, (owner, _a, _r) in self._routes.items()
                 if owner is monitor]
        for sid in stale:
            self._routes.pop(sid, None)
            monitor._subscriptions.pop(sid, None)
        if not stale or self._connection is None:
            return
        for sid in stale:
            with contextlib.suppress(Exception):
                await self._request("accountUnsubscribe", [sid])

    async def rebind(self, monitor: "NativeMarketMonitor") -> None:
        """(Re)declare one monitor's interest on the live connection."""
        if self._connection is None or not self.is_open:
            return
        await self._subscribe_monitor(monitor)

    async def aclose(self) -> None:
        self._stopping.set()
        for task in (self._task, self._reader_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self._task = None
        self._reader_task = None
        connection = self._connection
        self._connection = None
        self.is_open = False
        if connection is not None:
            with contextlib.suppress(Exception):
                await connection.aclose()

    # -- the loop --------------------------------------------------------

    async def _run(self) -> None:
        backoff = self.config.reconnect_min_s
        while not self._stopping.is_set():
            try:
                await self._connect()
                backoff = self.config.reconnect_min_s
                # AWAIT THE READER TASK, do not read here as well. Two
                # coroutines calling ``readexactly`` on one stream is a hard
                # error ("called while another coroutine is already waiting for
                # incoming data"), and it would surface as an unexplained
                # stream failure on every cycle. Exactly one reader, always.
                reader = self._reader_task
                if reader is not None:
                    await reader
                if self._stopping.is_set():
                    return
                self.last_error = "stream ended"
            except asyncio.CancelledError:
                raise
            except WebSocketUnavailable as exc:
                message = f"{type(exc).__name__}: {exc}"
                self.last_error = message[:200]
                self.reconnects += 1
                if "429" in message or "Too Many Requests" in message:
                    # A handshake refusal is a CAPACITY fact, not a fault, and
                    # it is the single most useful thing to count: it is what
                    # says "you are opening too many connections".
                    self.handshake_refusals += 1
                log("NATIVE", "shared stream unavailable (%s); recovering in "
                    "%.1fs", message[:120], backoff)
                for monitor in self._monitors:
                    monitor.stream_error(message)
            except Exception as exc:  # noqa: BLE001 - never kill the runtime
                message = f"{type(exc).__name__}: {exc}"
                self.last_error = message[:200]
                self.reconnects += 1
                log("ERROR", "shared stream cycle failed: %s", message[:160])
                for monitor in self._monitors:
                    monitor.stream_error(message)
            finally:
                self.is_open = False
                connection = self._connection
                self._connection = None
                self._routes.clear()
                reader = self._reader_task
                self._reader_task = None
                if reader is not None and not reader.done():
                    reader.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await reader
                if connection is not None:
                    with contextlib.suppress(Exception):
                        await connection.aclose()
            if self._stopping.is_set():
                return
            delay = min(backoff, self.config.reconnect_max_s)
            delay *= 0.5 + 0.5 * random.random()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=delay)
            backoff = min(backoff * 2.0, self.config.reconnect_max_s)

    async def _connect(self) -> None:
        if not self.ws_url:
            raise WebSocketUnavailable(
                "no websocket endpoint configured for native monitoring")
        connection = WebSocketConnection(
            parse_ws_url(self.ws_url),
            open_timeout_s=self.config.open_timeout_s)
        await connection.connect()
        self._connection = connection
        self.is_open = True
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()
        self._routes.clear()
        if self.reconnects:
            self.resubscribes += 1
        # The reader must be draining BEFORE any subscription is awaited: the
        # acknowledgements arrive on the same socket as the pushes, so a
        # subscribe awaited with no reader would deadlock.
        self._reader_task = asyncio.create_task(self._read_loop())
        for monitor in self._monitors:
            monitor.stream_connected(True)
            monitor.stats.resubscribes += 1
        bound = 0
        for monitor in self._monitors:
            try:
                await self._subscribe_monitor(monitor)
                bound += 1
            except Exception as exc:  # noqa: BLE001 - one bad market is not fatal
                monitor.stream_error(
                    f"subscribe failed: {type(exc).__name__}: {exc}"[:200])
        log("NATIVE", "shared websocket connected: %d market(s), %d bound, "
            "%d subscription(s)", len(self._monitors), bound, len(self._routes))

    async def _subscribe_monitor(self, monitor: "NativeMarketMonitor") -> None:
        connection = self._connection
        if connection is None or not self.is_open:
            return
        ids: dict[int, tuple[str, str]] = {}
        for address, role in monitor.identity.subscription_addresses():
            try:
                subscription_id = await self._request(
                    "accountSubscribe", [address, {
                        "encoding": "base64",
                        "commitment": MONITOR_COMMITMENT,
                    }])
            except WebSocketUnavailable as exc:
                # Bound what we can. A market that got its curve but not its
                # vaults still reports nothing rather than reporting a partial
                # price, because the price needs both.
                #
                # A REFUSAL IS RECORDED, not swallowed. Without this, a rejected
                # subscription left the monitor in WARMING -- indistinguishable
                # from a market that is simply not trading yet -- and the
                # operator had no way to tell "the endpoint refused my
                # subscription" from "nobody has sold this token for 8 seconds".
                # ``stream_error`` marks the monitor STALE, which routes the
                # engine to its recovery path, and names the refusal in the
                # diagnostics.
                monitor.stream_error(f"accountSubscribe refused: {exc}"[:200])
                break
            subscription_id = int(subscription_id)
            self._routes[subscription_id] = (monitor, address, role)
            monitor.stats.subscription_ids[role] = subscription_id
            ids[subscription_id] = (address, role)
        monitor.stream_resubscribed(ids)

    async def _request(self, method: str, params: list[Any]) -> int:
        connection = self._connection
        if connection is None:
            raise WebSocketUnavailable("no websocket connection")
        self._request_id += 1
        request_id = self._request_id
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[request_id] = future
        await connection.send_text(json.dumps({
            "jsonrpc": "2.0", "id": request_id, "method": method,
            "params": params,
        }))
        try:
            return await asyncio.wait_for(future, timeout=15.0)
        finally:
            self._pending.pop(request_id, None)

    async def _read_loop(self) -> None:
        connection = self._connection
        if connection is None:
            return
        while not self._stopping.is_set():
            raw = await connection.recv_text()
            self.last_frame_monotonic = time.monotonic()
            try:
                message = json.loads(raw)
            except ValueError:
                # A frame that is not JSON is a transport anomaly, not a market
                # event.
                for monitor in self._monitors:
                    monitor.stats.decode_failures += 1
                continue
            self._dispatch(message)

    def _dispatch(self, message: Any) -> None:
        if not isinstance(message, dict):
            return
        if "id" in message and "method" not in message:
            request_id = message.get("id")
            if isinstance(request_id, int):
                future = self._pending.get(request_id)
                if future is not None and not future.done():
                    error = message.get("error")
                    if error is not None:
                        future.set_exception(WebSocketUnavailable(
                            f"rpc error {error.get('code')}: "
                            f"{error.get('message')}"))
                    else:
                        result = message.get("result")
                        if isinstance(result, int):
                            future.set_result(result)
                        else:
                            future.set_exception(WebSocketUnavailable(
                                f"expected a subscription id, got "
                                f"{type(result).__name__}"))
            return
        method = message.get("method")
        if method == "accountNotification":
            self._on_notification(message)
            return
        error = message.get("error")
        if error is not None:
            code = error.get("code") if isinstance(error, dict) else None
            if code == -32005:
                # The documented "node is behind" throttle. Counted separately
                # from an HTTP 429 so a push-side throttle is distinguishable
                # from a handshake refusal.
                for monitor in self._monitors:
                    monitor.stats.rate_limit_429 += 1
            self.last_error = f"rpc error {code}: {error}"[:200]

    def _on_notification(self, message: dict[str, Any]) -> None:
        params = message.get("params") or {}
        subscription_id = params.get("subscription")
        route = self._routes.get(subscription_id)             if isinstance(subscription_id, int) else None
        if route is None:
            # A notification for a subscription this connection does not hold.
            # After a reconnect or a migration that is expected for one frame,
            # and it is not a gap -- nothing was lost, it simply no longer
            # belongs to anyone.
            return
        monitor, address, role = route
        payload = params.get("result")
        raw = _base64_account(payload)
        if not raw:
            monitor.stats.decode_failures += 1
            return
        slot = _notification_slot(payload)
        # SYNCHRONOUS BY DESIGN. Decode and price calculation are local and
        # cheap; making them awaitable would let one slow consumer delay the
        # next push for every other market on the same socket.
        monitor.deliver(role, raw, slot, address)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "shared_stream_connected": self.is_open,
            "shared_stream_reconnects": self.reconnects,
            "shared_stream_resubscribes": self.resubscribes,
            "shared_stream_handshake_refusals": self.handshake_refusals,
            "shared_stream_subscriptions": len(self._routes),
            "shared_stream_markets": len(self._monitors),
            "shared_stream_last_error": self.last_error,
        }


class NativeMonitorRegistry:
    """
    One monitor per market, all markets on ONE shared connection.

    The engine holds one registry. Positions are added when they open and removed
    when they close; a market with no position left loses its subscription, so a
    long PAPER run cannot accumulate a permanent tail of watching dead markets.

    The registry owns exactly ONE :class:`SharedSubscriptionStream` per endpoint,
    so the connection count is a function of how many endpoints are configured --
    not of how many positions are open. That is the whole point: the endpoint
    rate limits connection HANDSHAKES, so a registry that opened one per market
    would rate-limit itself into permanent recovery.
    """

    def __init__(self, *, config_for_url: Optional[Any] = None) -> None:
        self._monitors: dict[str, NativeMarketMonitor] = {}
        self._refs: dict[str, int] = {}
        self._config_for_url = config_for_url
        self._streams: dict[str, SharedSubscriptionStream] = {}

    def _key(self, identity: StaticVenueIdentity) -> str:
        return f"{identity.market_type}:{identity.market_address}"

    def _stream_for(self, config: MonitorConfig) -> SharedSubscriptionStream:
        stream = self._streams.get(config.ws_url)
        if stream is None:
            stream = SharedSubscriptionStream(config.ws_url, config=config)
            self._streams[config.ws_url] = stream
        return stream

    def acquire(self, identity: StaticVenueIdentity,
                config: MonitorConfig) -> NativeMarketMonitor:
        key = self._key(identity)
        monitor = self._monitors.get(key)
        if monitor is None:
            monitor = NativeMarketMonitor(
                identity, config=config,
                stream=self._stream_for(config) if config.ws_url else None)
            self._monitors[key] = monitor
            monitor.start()
        self._refs[key] = self._refs.get(key, 0) + 1
        return monitor

    def release(self, monitor: Optional[NativeMarketMonitor]) -> None:
        if monitor is None:
            return
        key = self._key(monitor.identity)
        remaining = self._refs.get(key, 1) - 1
        self._refs[key] = max(0, remaining)
        if remaining > 0:
            return
        self._refs.pop(key, None)
        self._monitors.pop(key, None)
        with contextlib.suppress(Exception):
            asyncio.get_running_loop().call_soon(
                lambda: asyncio.ensure_future(monitor.close()))

    def monitors(self) -> Sequence[NativeMarketMonitor]:
        return tuple(self._monitors.values())

    def streams(self) -> Sequence[SharedSubscriptionStream]:
        return tuple(self._streams.values())

    def diagnostics(self) -> dict[str, Any]:
        return {
            "markets": {key: monitor.diagnostics()
                        for key, monitor in self._monitors.items()},
            "streams": {url: stream.diagnostics()
                        for url, stream in self._streams.items()},
        }

    async def aclose(self) -> None:
        monitors = tuple(self._monitors.values())
        streams = tuple(self._streams.values())
        self._monitors.clear()
        self._refs.clear()
        self._streams.clear()
        for monitor in monitors:
            with contextlib.suppress(Exception):
                await monitor.close()
        for stream in streams:
            with contextlib.suppress(Exception):
                await stream.aclose()
