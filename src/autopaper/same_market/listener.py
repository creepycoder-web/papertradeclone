"""
Read-only Telegram listener for one channel.

REUSE, AND WHY IT IS SAFE
=========================

This does not log in again or open a second session. It constructs the
repository's existing :class:`~autopaper.telegram.client.TelegramListener` over
the same ``TELEGRAM_SESSION``, and reuses:

* :meth:`TelegramListener.start` - connect, authorise, resolve the ONE chat
* :meth:`TelegramListener._resolve_chat` - plus its forbidden-source check
* :meth:`TelegramListener._iter_messages` - the backlog anchor and flood-wait
  handling, which is the part that is genuinely subtle

It does NOT reuse :meth:`TelegramListener._handle`, because that method drops a
message whose parse yields no mint before any callback runs, and this harness
must see the ambiguous-mint case to report it. So :meth:`SameMarketListener.
_handle` reimplements the security gates of ``client.py:527`` - own-message
filter, per-message foreign-chat re-check, dedupe - and calls the harness
instead of the trading engine.

The gates are reimplemented rather than wrapped because the base method's
control flow does not admit a hook at the right point; the duplication is
therefore deliberate, and
``test_same_market_listener.py`` asserts the gates still hold.

NOTHING HERE WRITES TO TELEGRAM. No message is sent, no button is pressed, no
update is acknowledged. The Bot API client and the notifier are not imported.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..clock import Clock, SystemClock
from ..config import TelegramConfig
from ..telegram.client import (
    ConnectionState,
    DedupeWindow,
    ListenerStatus,
    TelegramListener,
    normalise_source_chat_id,
)

#: Signature of the harness's per-message hook.
#: ``(text, message_id, telegram_ts, received_at)``
MessageHandler = Callable[[str, int | None, float | None, float], Awaitable[None]]


@dataclass
class Capture:
    """One message handed to the harness, with its timing intact."""

    text: str
    message_id: int | None
    telegram_timestamp: float | None
    received_at: float
    chat_id: int | None = None

    @property
    def transport_latency_ms(self) -> float | None:
        """
        ``local_receive - telegram_message_time``.

        Approximate by construction: it spans two machines' clocks, and
        Telegram stamps messages with whole-second resolution, so up to a
        second of the figure is quantisation rather than measurement.
        """
        if self.telegram_timestamp is None:
            return None
        return max(0.0, (self.received_at - self.telegram_timestamp) * 1000.0)


class SameMarketListener(TelegramListener):
    """
    Read-only listener that hands every new message to a callback.

    Constructed with an explicit :class:`TelegramConfig`, so ``--channel`` on
    the CLI decides which chat is read. The configured chat is the only chat
    read; there is no fallback and no second source.
    """

    def __init__(
        self,
        config: TelegramConfig,
        on_message: MessageHandler,
        *,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(
            config=config,
            # The parent's on_alert is deliberately not wired: this harness
            # never calls into the trading engine.
            on_alert=None,
            clock=clock or SystemClock(),
            status=ListenerStatus(),
            dedupe=DedupeWindow(),
        )
        self._on_message = on_message
        self.messages_forwarded = 0
        self.messages_refused_foreign_chat = 0
        self.messages_refused_duplicate = 0
        self.messages_refused_own = 0

    # -- gates (mirrors TelegramListener._handle, client.py:527) ------------

    async def _handle(self, message: Any) -> None:
        mid = getattr(message, "id", None)
        chat_id = self.status.chat_id
        if mid is None or chat_id is None:
            return

        # Gate 1: never process our own messages.
        if message.sender_id == getattr(self, "own_id", None):
            self.messages_refused_own += 1
            return

        # Gate 2: re-check the chat on THIS message, not against the resolved
        # id we already hold - comparing the resolved id against itself is a
        # check that cannot fail.
        msg_chat = getattr(message, "chat_id", None)
        if (msg_chat is not None and chat_id is not None
                and normalise_source_chat_id(msg_chat)
                != normalise_source_chat_id(chat_id)):
            self.messages_refused_foreign_chat += 1
            return

        # Gate 3: dedupe within the window.
        key = (chat_id, mid)
        if not self.dedupe.add(key):
            self.messages_refused_duplicate += 1
            return

        text = getattr(message, "message", None) or getattr(message, "text", "") or ""
        telegram_ts: float | None = None
        dt = getattr(message, "date", None)
        if dt is not None:
            try:
                telegram_ts = dt.timestamp()
            except Exception:  # noqa: BLE001 - an unusable stamp is UNKNOWN
                telegram_ts = None

        received_at = self.clock.now()
        self.status.last_message_id = mid
        self.status.last_message_at = received_at
        self.status.messages_seen = getattr(self.status, "messages_seen", 0) + 1

        self.messages_forwarded += 1
        await self._on_message(text, mid, telegram_ts, received_at)

    async def start(self) -> None:
        await super().start()

    def diagnostics(self) -> dict[str, Any]:
        """Transport counters. Booleans and counts; never the chat id or token."""
        return {
            "state": self.status.state.value
            if isinstance(self.status.state, ConnectionState) else str(self.status.state),
            "connected": self.status.connected,
            "backlog_mode": self.status.backlog_mode,
            "since_message_id": self.status.since_message_id,
            "polls": self.status.polls,
            "messages_seen": getattr(self.status, "messages_seen", 0),
            "messages_forwarded": self.messages_forwarded,
            "refused_own": self.messages_refused_own,
            "refused_foreign_chat": self.messages_refused_foreign_chat,
            "refused_duplicate": self.messages_refused_duplicate,
        }


@dataclass
class ListenerRun:
    """Outcome of a listener run, so the CLI can report cleanly on Ctrl+C."""

    captures: list[Capture] = field(default_factory=list)
    stopped_reason: str = "unknown"
    diagnostics: dict[str, Any] = field(default_factory=dict)


async def listen(
    config: TelegramConfig,
    on_message: MessageHandler,
    *,
    stop: asyncio.Event | None = None,
    clock: Clock | None = None,
) -> ListenerRun:
    """
    Run until ``stop`` is set.

    ``stop`` defaults to a future Ctrl+C: the process is expected to stay up
    watching the channel, and a clean shutdown is the normal end of a manual
    smoke test rather than an exception.
    """
    listener = SameMarketListener(config, on_message, clock=clock)
    stop = stop or asyncio.Event()

    await listener.start()
    if listener.status.state in (ConnectionState.ERROR,
                                ConnectionState.AWAITING_LOGIN):
        return ListenerRun(
            captures=[],
            stopped_reason=f"not connected: {listener.status.detail}",
            diagnostics=listener.diagnostics(),
        )

    try:
        await listener.run(stop)
    except asyncio.CancelledError:
        return ListenerRun(
            captures=[], stopped_reason="cancelled", diagnostics=listener.diagnostics()
        )
    except Exception as exc:  # noqa: BLE001 - reported, never a traceback dump
        return ListenerRun(
            captures=[],
            stopped_reason=f"listener error: {exc!r}",
            diagnostics=listener.diagnostics(),
        )

    return ListenerRun(
        captures=[],
        stopped_reason="stopped",
        diagnostics=listener.diagnostics(),
    )


__all__ = [
    "Capture",
    "ListenerRun",
    "MessageHandler",
    "SameMarketListener",
    "listen",
]