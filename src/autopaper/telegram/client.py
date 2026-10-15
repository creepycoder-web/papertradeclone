"""
Telegram listener.

Responsibilities:
* authenticate with the credentials in ``.env.local``
* resolve the configured chat (username, invite link, numeric id, or -100 id)
* apply the backlog policy (``TELEGRAM_ALLOW_BACKLOG=false`` -> only messages
  newer than the moment we booted)
* dedupe by ``(chat_id, message_id)`` so a reconnect, a replay or a duplicate
  delivery can never create two paper trades
* expose an accurate connection status, including a precise diagnosis when
  authentication is not possible

The listener NEVER sends a message. It only reads.
"""

from __future__ import annotations

import asyncio
import re
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable

from ..clock import Clock, SystemClock
from ..config import TelegramConfig
from ..forensics.telegram_guard import (
    FailureLog as CallbackFailureLog,
)
from ..forensics.telegram_guard import (
    coerce_positive_float,
    describe_failure,
)
from ..forensics.telegram_guard import (
    safe_handle as _safe_handle_message,
)
from ..logging_setup import log
from ..models import AlertSnapshot, new_id
from .parser import ParsedAlert, looks_like_alert, parse_alert
from .signal_source import FALLBACK, PRIMARY

INVITE_RE = re.compile(r"(?:t\.me/\+|\+|joinchat/)([A-Za-z0-9_-]+)")

#: Channels that are known NOT to be the candidate source. Present in the
#: project's own environment from earlier work, and named here so a
#: misconfiguration fails loudly instead of silently feeding a second feed into
#: the decision path.
FORBIDDEN_SOURCE_NAMES: tuple[str, ...] = (
    "memesdontlies",
    "memes_dont_lies",
    "memecoinbot",
    "test",
    "testchannel",
    "sandbox",
    "dev",
)


def normalise_source_chat_id(value: Any) -> Optional[int]:
    """
    Collapse the two spellings Telegram uses for one supergroup id.

    Telethon reports a supergroup/channel as ``-100XXXXXXXXXX`` in
    ``message.chat_id`` but as the bare ``XXXXXXXXXX`` in ``entity.id``. A raw
    integer comparison therefore rejects EVERY message from the very chat it
    was configured to accept, which is what silently starved the bot of
    candidates.

    Stripping the ``-100`` marker from both sides is safe and does not weaken
    the source gate: Telegram reserves that form for supergroups, so it maps one
    id onto itself and cannot admit a different chat. A basic group has a short
    id and is never affected.

    Returns ``None`` for anything that is not a plain integer, so an
    unparseable id is treated as "not permitted" rather than compared loosely.
    """
    try:
        if isinstance(value, bool) or value is None:
            return None
        n = int(value)
    except (TypeError, ValueError):
        return None
    # The documented convention: a supergroup/channel is the decimal digits of
    # its id with "100" prepended, negated. So "-1002270552242" and
    # "2270552242" are ONE chat. Done as a string operation because the
    # arithmetic form of this rule is easy to get backwards.
    s = str(n)
    if s.startswith("-100"):
        tail = s[4:]
        # a real chat id is never 0, so "-1000" is not a marked id and must not
        # collapse onto 0
        if tail and tail.isdigit() and int(tail) != 0:
            return int(tail)
    return n


class TelegramSourceNotAllowed(RuntimeError):
    """
    The configured chat is not a permitted candidate source.

    Raised at connection time so a wrong channel can never produce a candidate,
    rather than producing one and having it dropped later where the drop would
    be invisible in the journal.
    """



def _flood_wait_error():
    """Resolve FloodWaitError lazily so importing this module never needs telethon."""
    try:
        from telethon.errors import FloodWaitError

        return FloodWaitError
    except Exception:  # noqa: BLE001 - tests run without a client
        return ()


class ConnectionState:
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    AWAITING_LOGIN = "AWAITING_LOGIN"
    CONNECTED = "CONNECTED"
    BACKOFF = "BACKOFF"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


@dataclass
class ListenerStatus:
    state: str = ConnectionState.DISCONNECTED
    detail: str = ""
    chat: str | None = None
    chat_id: int | None = None
    chat_title: str | None = None
    last_message_id: int | None = None
    last_message_at: float | None = None
    last_received_id: int | None = None
    last_processed_id: int | None = None
    last_connected_at: float | None = None
    last_disconnected_at: float | None = None
    gap_detected: bool = False
    messages_seen: int = 0
    messages_accepted: int = 0
    duplicates_dropped: int = 0
    unparsed_dropped: int = 0
    #: Messages dropped because they came from a chat other than the single
    #: permitted candidate source. Must be zero in a healthy run; a non-zero
    #: value means a second feed tried to contribute, and the journal proves it
    #: did not succeed.
    foreign_chat_dropped: int = 0
    reconnects: int = 0
    backlog_mode: str = "new_only"
    since_message_id: int | None = None
    polls: int = 0
    poll_errors: int = 0
    last_poll_at: float | None = None

    @property
    def connected(self) -> bool:
        return self.state == ConnectionState.CONNECTED

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "detail": self.detail,
            "chat": self.chat,
            "chat_id": self.chat_id,
            "chat_title": self.chat_title,
            "last_message_id": self.last_message_id,
            "last_message_at": self.last_message_at,
            "last_received_id": self.last_received_id,
            "last_processed_id": self.last_processed_id,
            "last_connected_at": self.last_connected_at,
            "last_disconnected_at": self.last_disconnected_at,
            "gap_detected": self.gap_detected,
            "messages_seen": self.messages_seen,
            "messages_accepted": self.messages_accepted,
            "duplicates_dropped": self.duplicates_dropped,
            "unparsed_dropped": self.unparsed_dropped,
            "foreign_chat_dropped": self.foreign_chat_dropped,
            "reconnects": self.reconnects,
            "backlog_mode": self.backlog_mode,
            "since_message_id": self.since_message_id,
            "polls": self.polls,
            "poll_errors": self.poll_errors,
        }


class DedupeWindow:
    """Bounded LRU over processed ``(chat_id, message_id)`` keys."""

    def __init__(self, capacity: int = 4096) -> None:
        self.capacity = capacity
        self._items: OrderedDict[tuple[int, int], None] = OrderedDict()

    def __contains__(self, key: tuple[int, int]) -> bool:
        if key in self._items:
            self._items.move_to_end(key)
            return True
        return False

    def add(self, key: tuple[int, int]) -> bool:
        """Return True if the key was new (i.e. not a duplicate)."""
        if key in self:
            return False
        self._items[key] = None
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return True

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class TelegramListener:
    config: TelegramConfig
    on_alert: Callable[[AlertSnapshot, ParsedAlert], Awaitable[None]] | None = None
    clock: Clock = field(default_factory=SystemClock)
    status: ListenerStatus = field(default_factory=ListenerStatus)
    dedupe: DedupeWindow = field(default_factory=DedupeWindow)
    _client: Any = None
    _seen_persisted: set[tuple[int, int]] = field(default_factory=set)
    #: Isolated per-message callback failures, with full context and traceback.
    #: Bounded, so a persistent fault cannot grow without limit.
    _callback_failures: CallbackFailureLog = field(
        default_factory=CallbackFailureLog)
    _callback_failures_total: int = 0
    #: Optional multi-source mode.  Leaving this unset preserves the historical
    #: single-chat test/CLI contract.
    source_chats: dict[str, str] | None = None
    _source_entities: dict[str, Any] = field(default_factory=dict)
    _source_chat_ids: dict[str, int] = field(default_factory=dict)
    _source_cursors: dict[str, int] = field(default_factory=dict)

    # -- lifecycle ---------------------------------------------------------

    def _on_callback_failure(self, ctx) -> None:
        """
        Log one isolated callback failure with everything needed to locate it.

        The historical defect was that only ``str(exc)`` survived, so a crash
        could not be traced to a call site. The traceback is the fix; the message
        identity is what makes it actionable, because the reader is the only
        place that knows which alert was in flight.
        """
        self._callback_failures_total += 1
        log("ERROR", "%s", describe_failure(ctx))
        # Surfaced on the status line too, so the operator sees a degraded
        # reader in /status rather than only in the log.
        self.status.detail = (
            f"stream alive; {self._callback_failures_total} message(s) failed "
            f"in isolation (last: {ctx.exception_type})"
        )

    def callback_failure_stats(self) -> dict[str, Any]:
        """Observable health of the isolation boundary."""
        return {
            "isolated_failures": self._callback_failures_total,
            "stream_alive": self.status.state is ConnectionState.CONNECTED,
            **self._callback_failures.as_dict(),
        }

    # -- lifecycle ---------------------------------------------------------

    def diagnose(self) -> tuple[str, str]:
        """
        Explain, before we touch the network, whether we can even connect.

        Returns ``(state, detail)``. This keeps the operator's failure mode
        legible instead of an opaque ``AuthKeyError``.
        """
        if not self.config.has_credentials:
            return (
                ConnectionState.ERROR,
                "TELEGRAM_API_ID / TELEGRAM_API_HASH missing from .env.local",
            )
        if not self.config.has_session:
            return (
                ConnectionState.AWAITING_LOGIN,
                "TELEGRAM_SESSION is empty - run `python -m autopaper.cli login` "
                "once to create a user session (needs phone + Telegram code). "
                "The engine is running and will pick it up automatically.",
            )
        if not self.config.chat:
            return (ConnectionState.ERROR, "TELEGRAM_CHAT not set in .env.local")
        return (ConnectionState.CONNECTED, "credentials and session present")

    async def start(self) -> None:
        from telethon import TelegramClient
        from telethon.errors import FloodWaitError  # noqa: F401
        from telethon.sessions import StringSession

        state, detail = self.diagnose()
        self.status.detail = detail
        if state in (ConnectionState.ERROR, ConnectionState.AWAITING_LOGIN):
            self.status.state = state
            log("TELEGRAM", "%s: %s", state, detail)
            return

        self.status.state = ConnectionState.CONNECTING
        # Coerced, not trusted. The exponential-backoff doubling below is the one
        # unguarded `X * <int>` in this module's reconnect path; a non-positive or
        # missing value would raise TypeError there and kill the CONNECT loop
        # itself, which is the same failure mode this whole change exists to
        # remove. See autopaper.forensics.telegram_guard.coerce_positive_float.
        backoff = coerce_positive_float(
            self.config.reconnect_backoff_s, 5.0)
        attempt = 0
        while True:
            try:
                self._client = TelegramClient(
                    StringSession(self.config.session or ""),
                    self.config.api_id,
                    self.config.api_hash,
                    connection_retries=3,
                    retry_delay=2,
                    request_retries=2,
                )
                await self._client.connect()
                if not await self._client.is_user_authorized():
                    self.status.state = ConnectionState.AWAITING_LOGIN
                    self.status.detail = (
                        "session rejected: not authorized - re-run "
                        "`python -m autopaper.cli login`"
                    )
                    log("TELEGRAM", "%s", self.status.detail)
                    return
                if self.source_chats:
                    await self._resolve_source_chats()
                else:
                    await self._resolve_chat()
                self.status.state = ConnectionState.CONNECTED
                self.status.last_connected_at = time.time()
                self.status.detail = "connected"
                log("TELEGRAM", "connected chat=%s (id=%s) sources=%s backlog=%s",
                    self.status.chat_title or self.status.chat, self.status.chat_id,
                    ",".join(self._source_chat_ids) or "primary",
                    "allowed" if self.config.allow_backlog else "new_only")
                return
            except (AuthKeyError,) as exc:
                self.status.state = ConnectionState.AWAITING_LOGIN
                self.status.detail = f"auth key invalid ({exc!r}) - re-run login"
                log("TELEGRAM", "%s", self.status.detail)
                return
            except SessionPasswordNeededError:
                self.status.state = ConnectionState.AWAITING_LOGIN
                self.status.detail = "2FA password required - re-run login"
                log("TELEGRAM", "%s", self.status.detail)
                return
            except _flood_wait_error() as exc:
                wait = float(getattr(exc, "seconds", 30) or 30)
                self.status.state = ConnectionState.BACKOFF
                self.status.detail = f"flood wait {wait:.0f}s"
                log("TELEGRAM", "flood wait %.0fs", wait)
                await asyncio.sleep(wait)
                continue
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                self.status.state = ConnectionState.BACKOFF
                self.status.detail = f"{type(exc).__name__}: {exc}"
                self.status.reconnects = attempt
                log("TELEGRAM", "connect failed, retry in %.0fs: %s", backoff, exc)
                await asyncio.sleep(backoff)
                # Both operands coerced: a missing or non-positive setting must
                # not be able to raise `None * int` here, which would kill the
                # CONNECT loop rather than one message.
                backoff = min(
                    coerce_positive_float(self.config.reconnect_backoff_max_s, 120.0),
                    coerce_positive_float(backoff, 5.0) * 2,
                )
                if attempt > 12:
                    self.status.state = ConnectionState.ERROR
                    return

    async def _resolve_chat(self) -> None:
        """
        Resolve the ONE permitted candidate channel.

        The configured chat IS the source of candidates. There is no second
        feed, no fallback channel, no "if the primary is empty, try another".
        Resolving to anything other than the configured chat, or resolving to a
        chat whose title is a known different source, is refused here - at
        connection time - rather than filtered later.
        """
        client = self._client
        target = (self.config.chat or "").strip()
        self.status.chat = target
        if not target:
            self.status.state = ConnectionState.ERROR
            self.status.detail = (
                f"TELEGRAM_CHAT is not configured; the only permitted candidate "
                f"source is {TelegramConfig.CANDIDATE_SOURCE}"
            )
            raise TelegramSourceNotAllowed(self.status.detail)

        entity = None
        if target.lstrip("-").isdigit():
            try:
                entity = await client.get_entity(int(target))
            except (ValueError, TypeError):
                entity = None
        if entity is None:
            username = target.lstrip("@")
            invite = INVITE_RE.search(target)
            if invite:
                try:
                    entity = await client.get_entity(f"https://t.me/+{invite.group(1)}")
                except Exception:  # noqa: BLE001
                    entity = None
            if entity is None:
                try:
                    entity = await client.get_entity(username)
                except Exception as exc:  # noqa: BLE001
                    self.status.state = ConnectionState.ERROR
                    self.status.detail = (
                        f"cannot resolve chat {username!r}: {exc}. Telegram user "
                        "sessions can only see chats the account is a member of."
                    )
                    log("TELEGRAM", "%s", self.status.detail)
                    raise
        self.status.chat_id = getattr(entity, "id", None)
        self.status.chat_title = getattr(entity, "title", None) or getattr(
            entity, "username", None
        )
        self._assert_permitted_source(entity)

    async def _resolve_source_chats(self) -> None:
        """Resolve both feeds once; both cursors are anchored at startup."""
        self._source_entities.clear()
        self._source_chat_ids.clear()
        for source, target in self.source_chats.items():
            target = (target or "").strip()
            if not target:
                continue
            entity = None
            if target.lstrip("-").isdigit():
                entity = await self._client.get_entity(int(target))
            else:
                invite = INVITE_RE.search(target)
                lookup = f"https://t.me/+{invite.group(1)}" if invite else target.lstrip("@")
                entity = await self._client.get_entity(lookup)
            if entity is None:
                raise TelegramSourceNotAllowed(f"cannot resolve {source} source {target!r}")
            self._assert_permitted_source(entity)
            chat_id = getattr(entity, "id", None)
            if chat_id is None:
                raise TelegramSourceNotAllowed(f"{source} source has no Telegram id")
            self._source_entities[source] = entity
            self._source_chat_ids[source] = int(chat_id)
        if PRIMARY not in self._source_chat_ids:
            raise TelegramSourceNotAllowed("primary MooDeng source was not resolved")
        self.status.chat = self.source_chats.get(PRIMARY)
        self.status.chat_id = self._source_chat_ids[PRIMARY]
        self.status.chat_title = getattr(self._source_entities[PRIMARY], "title", None)

    def _assert_permitted_source(self, entity: Any) -> None:
        """
        Refuse a chat that is not the one permitted candidate source.

        Matched case-insensitively on the title, the username and the display
        name, with non-alphanumerics stripped so "MemesDontLies",
        "memes_dont_lies" and "Memes Dont Lies" are all the same name.
        """
        candidates = [
            getattr(entity, "title", None),
            getattr(entity, "username", None),
            getattr(entity, "first_name", None),
        ]
        for raw in candidates:
            if not raw:
                continue
            slug = re.sub(r"[^a-z0-9]", "", str(raw).lower())
            for bad in FORBIDDEN_SOURCE_NAMES:
                if bad in slug:
                    detail = (
                        f"configured chat resolved to {raw!r}, which matches the "
                        f"forbidden candidate source {bad!r}. The only permitted "
                        f"source is {TelegramConfig.CANDIDATE_SOURCE}."
                    )
                    self.status.state = ConnectionState.ERROR
                    self.status.detail = detail
                    log("TELEGRAM", "%s", detail)
                    raise TelegramSourceNotAllowed(detail)

    # -- backlog -----------------------------------------------------------

    async def last_message_id(self) -> int | None:
        """
        Newest message id currently in the chat, used as the "new only" anchor.

        ``get_messages`` has no ``offset`` parameter - passing one raises, and a
        silently failed anchor means no ``min_id``, which means the entire
        backlog gets replayed as if it were live. That is the one failure mode in
        this listener that must never be swallowed, so a failure is logged loudly
        and the caller is told not to proceed without an anchor.
        """
        if self._client is None or self.status.chat_id is None:
            return None
        try:
            # default reverse=True returns the newest message first
            msgs = await self._client.get_messages(self.status.chat_id, limit=1)
        except Exception as exc:  # noqa: BLE001
            log(
                "ERROR",
                "cannot read the backlog anchor: %s. Refusing to stream: without "
                "an anchor the whole history would be replayed as live.",
                exc,
            )
            self.status.state = ConnectionState.ERROR
            self.status.detail = f"backlog anchor unavailable: {exc}"
            return None
        ids = [getattr(m, "id", 0) for m in (msgs or []) if getattr(m, "id", None)]
        if not ids:
            # an empty chat has no history to replay, so streaming is safe
            self.status.since_message_id = 0
            return 0
        self.status.since_message_id = max(ids)
        return self.status.since_message_id

    # -- run loop ----------------------------------------------------------

    async def run(self, stop: asyncio.Event) -> None:
        """
        Read new messages forever.

        With ``TELEGRAM_ALLOW_BACKLOG=false`` (the supplied value) we anchor on
        the newest message id at boot and only ever consider strictly newer
        ones, so restarting the process never replays history as if it were live.
        """
        if self.status.state in (ConnectionState.ERROR, ConnectionState.AWAITING_LOGIN,
                                 ConnectionState.DISCONNECTED):
            return
        if not self.config.allow_backlog:
            self.status.backlog_mode = "new_only"
            if self.source_chats:
                anchors = await self._anchor_sources()
                if anchors is None:
                    log("ERROR", "not starting the read stream: source anchor unavailable")
                    return
                log("TELEGRAM", "backlog disabled: ignoring source messages <= %s", anchors)
            else:
                anchor = await self.last_message_id()
                if anchor is None:
                    log("ERROR", "not starting the read stream: no backlog anchor")
                    return
                log("TELEGRAM", "backlog disabled: ignoring messages <= %s", anchor)
        else:
            self.status.backlog_mode = "allowed"
            log("TELEGRAM", "backlog ENABLED: historical messages will be processed "
                            "as live candidates (this is a simulation artefact)")

        try:
            iterator = self._iter_source_messages(stop) if self.source_chats else self._iter_messages(stop)
            async for item in iterator:
                if stop.is_set():
                    break
                message, source = item if self.source_chats else (item, None)
                # ISOLATION BOUNDARY.
                #
                # Previously `await self._handle(message)` ran bare, so ANY
                # exception in the per-message path unwound this whole `try`,
                # set the listener to ERROR/STOPPED and killed the stream. The
                # supervisor then reconnected -- and because it re-anchors on the
                # last PROCESSED message id, every alert that arrived during the
                # gap was silently lost. One malformed message could therefore
                # cost real signals.
                #
                # `_safe_handle_message` records the full type, message, repr,
                # cause chain and traceback with the message's identity attached,
                # then RETURNS so the loop continues to the next message.
                # `asyncio.CancelledError` is re-raised inside it, so shutdown and
                # Ctrl+C behaviour is unchanged.
                # NOTE: `safe_handle` takes only (handle, message, *, text,
                # failures, stage, on_failure) — the source label belongs to
                # `_handle`, so it is bound here via closure, not passed through.
                _handler = self._handle if source is None else (
                    lambda m, _s=source: self._handle(m, _s)
                )
                await _safe_handle_message(
                    _handler,
                    message,
                    text=getattr(message, "message", None)
                    or getattr(message, "text", None),
                    failures=self._callback_failures,
                    stage="handle_message",
                    on_failure=self._on_callback_failure,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # Reached only for a failure OUTSIDE a single message -- the iterator
            # itself, or the isolation guard itself. Logged with the full
            # traceback for the first time in this reader's history, because the
            # historical `str(exc)`-only line is precisely why 26 occurrences in
            # logs/runner.err could not be traced to a call site.
            self.status.state = ConnectionState.ERROR
            self.status.detail = f"stream error: {exc!r}"
            log("ERROR", "telegram stream error: %s", exc)
            log("ERROR", "telegram stream error traceback:\n%s",
                "".join(traceback.format_exception(
                    type(exc), exc, exc.__traceback__)).rstrip())
        finally:
            self.status.last_disconnected_at = time.time()
            self.status.state = ConnectionState.STOPPED

    async def _anchor_sources(self) -> dict[str, int] | None:
        """
        Anchor each feed once, at first start, and never jump an established
        cursor forward.

        The anchor exists to skip the historical backlog on boot. On a
        reconnect it must NOT run again for a feed that is already streaming:
        moving the cursor to the newest message would silently skip whatever
        arrived while the stream was down, which is the exact alert this reader
        exists to catch. A feed with no cursor yet is anchored as before.
        """
        for source, chat_id in self._source_chat_ids.items():
            if source in self._source_cursors:
                continue
            msgs = await self._client.get_messages(chat_id, limit=1)
            ids = [getattr(m, "id", 0) for m in (msgs or []) if getattr(m, "id", None)]
            self._source_cursors[source] = max(ids) if ids else 0
        return dict(self._source_cursors)

    async def _iter_source_messages(self, stop: asyncio.Event | None = None):
        """Poll both anchored feeds without ever replaying fallback history."""
        while True:
            if stop is not None and stop.is_set():
                return
            for source, chat_id in self._source_chat_ids.items():
                since = self._source_cursors.get(source, 0)
                messages = await self._client.get_messages(chat_id, limit=50, min_id=since)
                for message in sorted(
                    (m for m in (messages or []) if getattr(m, "id", None) and m.id > since),
                    key=lambda m: m.id,
                ):
                    self._source_cursors[source] = message.id
                    yield message, source
            await self._wait_for_next_poll(stop)

    async def _iter_messages(self, stop: asyncio.Event | None = None):
        """
        Yield messages newer than the anchor, then keep yielding as they arrive.

        Telethon's ``iter_messages`` does NOT long-poll when ``min_id`` is set on
        a broadcast channel: it resolves immediately with nothing and the stream
        ends, which would make the agent reconnect in a loop and - worse - could
        re-anchor past alerts it never saw. So "new only" is implemented as an
        explicit poll loop with a durable high-water mark. One request per
        ``poll_interval_s`` on MTProto is well inside the rate limit, and
        ``get_messages`` returns nothing at all when there is nothing new.
        """
        entity = self.status.chat_id
        since = self.status.since_message_id
        if not self.config.allow_backlog and since is None:
            return

        while True:
            # Checked here as well as in ``run``, because ``run`` only re-checks
            # after a message arrives. A quiet channel delivers none, so a check
            # there alone can never fire and Ctrl+C would never take effect.
            if stop is not None and stop.is_set():
                return
            try:
                if self.config.allow_backlog:
                    # one bounded pass over history, then live
                    messages = await self._client.get_messages(entity, limit=200)
                    ascending = sorted(
                        (m for m in (messages or []) if getattr(m, "id", None)),
                        key=lambda m: m.id,
                    )
                    for message in ascending:
                        self.status.since_message_id = max(
                            self.status.since_message_id or 0, message.id
                        )
                        yield message
                    self.config = replace(self.config, allow_backlog=False)
                    self.status.backlog_mode = "new_only"
                    since = self.status.since_message_id
                else:
                    messages = await self._client.get_messages(
                        entity, limit=50, min_id=since
                    )
                    ascending = sorted(
                        (m for m in (messages or [])
                         if getattr(m, "id", None)
                         and (since is None or m.id > since)),
                        key=lambda m: m.id,
                    )
                    for message in ascending:
                        since = max(since or 0, message.id)
                        self.status.since_message_id = since
                        previous = self.status.last_received_id
                        if previous is not None and message.id > previous + 1:
                            self.status.gap_detected = True
                        self.status.last_received_id = message.id
                        yield message
                    self.status.polls += 1
                    self.status.last_poll_at = time.time()
            except asyncio.CancelledError:
                raise
            except _flood_wait_error() as exc:
                wait = float(getattr(exc, "seconds", 30) or 30)
                self.status.state = ConnectionState.BACKOFF
                self.status.detail = f"flood wait {wait:.0f}s"
                log("TELEGRAM", "flood wait %.0fs; pausing the read loop", wait)
                await asyncio.sleep(wait)
                self.status.state = ConnectionState.CONNECTED
                continue
            except Exception as exc:  # noqa: BLE001
                log("ERROR", "telegram read poll failed: %r", exc)
                self.status.poll_errors += 1
                await asyncio.sleep(self.config.poll_error_backoff_s)
                continue
            await self._wait_for_next_poll(stop)

    async def _wait_for_next_poll(self, stop: asyncio.Event | None) -> None:
        """
        Wait out the poll interval, returning early the moment ``stop`` is set.

        A flat ``asyncio.sleep`` would make a shutdown request wait for the
        sleep to run out, and the caller would then poll again before noticing.
        Racing the event against the interval keeps Ctrl+C prompt.
        """
        if stop is None:
            await asyncio.sleep(self.config.poll_interval_s)
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=self.config.poll_interval_s)
        except (asyncio.TimeoutError, TimeoutError):
            return

    async def _consume(self) -> None:
        """Drive the read loop. Split out so tests can run it directly."""
        stop = asyncio.Event()
        await self.run(stop)

    async def _handle(self, message: Any, source_name: str | None = None) -> None:
        mid = getattr(message, "id", None)
        source_name = source_name or PRIMARY
        chat_id = self._source_chat_ids.get(source_name, self.status.chat_id)
        if mid is None or chat_id is None:
            return
        self.status.messages_seen += 1
        self.status.last_processed_id = mid
        if message.sender_id == getattr(self, "own_id", None):
            return

        # Defence in depth: even though the chat is resolved once at connect
        # time, every message's OWN chat is re-checked before it is parsed. A
        # candidate from any other chat is dropped here and counted, so the
        # journal can prove no other source ever contributed.
        #
        # The message's chat is read from the message itself, not from
        # `self.status.chat_id` - comparing the resolved id against itself would
        # be a check that can never fail.
        msg_chat = getattr(message, "chat_id", None)
        if (msg_chat is not None and chat_id is not None
                and normalise_source_chat_id(msg_chat)
                != normalise_source_chat_id(chat_id)):
            self.status.unparsed_dropped += 1
            self.status.foreign_chat_dropped += 1
            log(
                "TELEGRAM",
                "message %s from chat %s is not the permitted source chat %s; "
                "dropped",
                mid, msg_chat, chat_id,
            )
            return

        key = (chat_id, mid)
        if not self.dedupe.add(key) or key in self._seen_persisted:
            self.status.duplicates_dropped += 1
            log("TELEGRAM", "duplicate %s:%s dropped", chat_id, mid)
            return

        text = getattr(message, "message", None) or getattr(message, "text", "") or ""
        source_timestamp = None
        dt = getattr(message, "date", None)
        if dt is not None:
            try:
                source_timestamp = dt.timestamp()
            except Exception:  # noqa: BLE001
                source_timestamp = None
        received_at = self.clock.now()
        self.status.last_message_id = mid
        self.status.last_message_at = received_at

        if not looks_like_alert(text):
            self.status.unparsed_dropped += 1
            return

        parsed = parse_alert(text)
        if not parsed.mint:
            self.status.unparsed_dropped += 1
            log("ALERT", "msg %s has no mint; dropped (warnings=%s)",
                mid, ",".join(parsed.warnings))
            return

        snapshot = self._snapshot(mid, received_at, source_timestamp, text, parsed,
                                  source_name=source_name, chat_id=chat_id)
        self.status.messages_accepted += 1
        latency_ms = (
            (received_at - source_timestamp) * 1000.0 if source_timestamp else None
        )
        log(
            "ALERT",
            "#%s %s mint=%s sym=%s src_lag=%s flds=%d warn=%s",
            mid,
            (snapshot.symbol or "?")[:12],
            (snapshot.mint or "?")[:8],
            snapshot.symbol,
            f"{latency_ms:.0f}ms" if latency_ms is not None else "n/a",
            len(parsed.fields),
            ",".join(parsed.warnings) or "none",
        )
        if self.on_alert is not None:
            await self.on_alert(snapshot, parsed)

    def _snapshot(self, mid: int, received_at: float, source_ts: float | None,
                  text: str, parsed: ParsedAlert, *,
                  source_name: str = PRIMARY, chat_id: int | None = None) -> AlertSnapshot:
        # ``parsed.as_dict()`` deep-copies the mapping contents; the snapshot is
        # frozen so nothing can be backfilled into it later.
        payload = parsed.as_dict()
        fields = dict(parsed.fields)
        provenance = dict(parsed.provenance)
        for key in ("twitter", "telegram", "website"):
            if parsed.fields.get(key):
                fields.setdefault(key, parsed.fields[key])
                provenance.setdefault(key, parsed.provenance.get(key, "url"))
        age_s = parsed.age_seconds()
        if age_s is not None:
            fields["age_seconds"] = age_s
            provenance["age_seconds"] = "alert_label:age"

        return AlertSnapshot(
            candidate_id=new_id("cand"),
            source="telegram",
            chat=str((self.source_chats or {}).get(source_name)
                     or (self.status.chat if source_name == PRIMARY else source_name)
                     or self.config.chat or ""),
            message_id=mid,
            source_timestamp=source_ts,
            received_timestamp=received_at,
            raw_text=text,
            mint=parsed.mint,
            symbol=parsed.symbol,
            name=parsed.name or payload.get("name"),
            fields=fields,
            provenance=provenance,
            parse_warnings=tuple(parsed.warnings),
            signal_source=source_name,
        )

    def restore_seen(self, keys: list[list[int]]) -> None:
        for pair in keys:
            if len(pair) == 2:
                self._seen_persisted.add((int(pair[0]), int(pair[1])))
        for key in self._seen_persisted:
            self.dedupe.add(key)

    def seen_keys(self) -> list[list[int]]:
        return [[a, b] for a, b in sorted(self._seen_persisted | set(self.dedupe._items))]

    async def stop(self) -> None:
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:  # noqa: BLE001
                pass
        self.status.state = ConnectionState.STOPPED


async def interactive_login(api_id: int, api_hash: str, session: str) -> str:
    """
    One-time user login. Run by the operator, never by the engine.

    Returns a ``StringSession`` value to be pasted into ``TELEGRAM_SESSION`` in
    ``.env.local``.
    """
    from telethon import TelegramClient
    from telethon.sessions import StringSession

    client = TelegramClient(StringSession(session or ""), api_id, api_hash)
    await client.connect()
    if not await client.is_user_authorized():
        phone = input("phone (international format): ").strip()
        await client.send_code_request(phone)
        code = input("telegram login code: ").strip()
        try:
            await client.sign_in(phone=phone, code=code)
        except Exception:  # noqa: BLE001 - 2FA
            import getpass

            pw = getpass.getpass("2FA password: ")
            await client.sign_in(password=pw)
    value = client.session.save()
    await client.disconnect()
    return value if isinstance(value, str) else str(value)
