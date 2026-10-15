"""
The notification side of the Bot API channel.

The engine's notifier hook is SYNCHRONOUS (``_notify(text, keyboard)``) and is
called from deep inside the trading path - decision rejects, fills, exits, mode
changes. So this notifier must return immediately. It therefore does not perform
any I/O in the caller's thread: it appends to a bounded queue and a background
worker performs the delivery.

The consequences of that design are deliberate:

* A slow or dead Telegram cannot slow down or stall a trading decision.
* A queue that fills DROPS notifications and counts them. It never blocks and
  never grows without limit, because an unbounded queue would turn a Telegram
  outage into a memory leak in the trading process.
* A send failure is logged and counted. It can never change a trading state,
  because this class has no reference to the runtime state at all.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from ..logging_setup import log
from .botapi import BotApiError, BotApiTransport, BotStatus, _clean_keyboard

#: Telegram's practical limit is ~30 messages/second globally but ~1/s per chat
#: is comfortable and keeps a chat readable. Enforced locally so a burst of
#: alerts cannot become a burst of API calls.
DEFAULT_MIN_SEND_INTERVAL_S = 0.5


@dataclass
class Notification:
    text: str
    keyboard: Optional[Sequence[Sequence[dict[str, str]]]] = None
    reply_to: Optional[int] = None


@dataclass
class NotifierStatus:
    queued: int = 0
    sent: int = 0
    dropped_queue_full: int = 0
    failed: int = 0
    last_error: str = ""
    attached: bool = False

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["last_error"] = self.last_error[:200]
        return d


class TelegramNotifier:
    """
    Queue-and-send notifier for the existing ``set_notifier`` hook.

    Constructed around an existing :class:`BotApiTransport`; it adds no
    networking of its own and has no access to the runtime state, so it cannot
    influence a decision even if it misbehaves.
    """

    def __init__(
        self,
        transport: BotApiTransport,
        *,
        max_queue: int = 100,
        min_send_interval_s: float = DEFAULT_MIN_SEND_INTERVAL_S,
        start_task: bool = False,
    ) -> None:
        self.transport = transport
        self.max_queue = max(1, int(max_queue))
        self.min_send_interval_s = float(min_send_interval_s)
        self.status = NotifierStatus(attached=True)
        self._queue: asyncio.Queue[Notification] = asyncio.Queue(
            maxsize=self.max_queue
        )
        self._worker: Optional[asyncio.Task] = None
        self._last_send = 0.0
        self._closing = False
        if start_task:
            self.start()

    # -- the synchronous hook ---------------------------------------------

    def send(self, text: str, keyboard: Any = None) -> None:
        """
        Enqueue a notification. Returns immediately; never raises.

        This is what the engine calls. It is called from the trading path, so it
        must not block, must not perform I/O, and must not propagate an error
        into a decision.
        """
        if self._closing:
            return
        item = Notification(
            text=str(text or ""),
            keyboard=_coerce_keyboard(keyboard),
        )
        item.text = _clamp_to_telegram(item.text)
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            # Drop the NEWEST and say so. Dropping the oldest would discard an
            # older state change; the newest is the one the operator is most
            # likely to be looking at, and the counter makes the loss visible.
            self.status.dropped_queue_full += 1
            log("WARNING", "notification dropped: queue full (%d)",
                self.max_queue)
            return
        self.status.queued += 1

    # -- worker ------------------------------------------------------------

    def start(self) -> None:
        """Start the delivery worker. Safe to call once the loop is running."""
        if self._worker is None or self._worker.done():
            self._closing = False
            self._worker = asyncio.ensure_future(self._run())

    async def _run(self) -> None:
        while not self._closing:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return
            try:
                await self._pace()
                await self._deliver(item)
            except asyncio.CancelledError:
                return
            except Exception as exc:  # noqa: BLE001
                self.status.failed += 1
                self.status.last_error = type(exc).__name__
                # The EXCEPTION DETAIL is the diagnostic. Logging only the class
                # name is why "BotApiForbidden" appeared with no cause: the
                # class name hides the HTTP status and Telegram's own
                # description, which are the only things that say whether this
                # is a bad token, a missing permission, or a message too long.
                log("ERROR", "notification delivery failed (%s): %s",
                    type(exc).__name__, str(exc)[:300])
            finally:
                self._queue.task_done()

    async def _pace(self) -> None:
        """Keep at least ``min_send_interval_s`` between API calls."""
        loop = asyncio.get_running_loop()
        now = loop.time()
        wait = self._last_send + self.min_send_interval_s - now
        if wait > 0:
            await asyncio.sleep(wait)
            now = loop.time()
        self._last_send = now

    async def _deliver(self, item: Notification) -> None:
        if item.reply_to is not None:
            await self.transport.edit_message_text(
                item.reply_to, item.text, item.keyboard)
        else:
            await self.transport.send_message(item.text, item.keyboard)
        self.status.sent += 1

    # -- lifecycle ---------------------------------------------------------

    async def flush(self, timeout_s: float = 5.0) -> bool:
        """Wait for the queue to drain. Used by tests and by clean shutdown."""
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout_s)
            return True
        except asyncio.TimeoutError:
            return False

    async def aclose(self) -> None:
        self._closing = True
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._worker = None

    def status_dict(self) -> dict[str, Any]:
        return self.status.as_dict()


def _coerce_keyboard(keyboard: Any) -> Optional[list[list[dict[str, str]]]]:
    """
    Accept only the shape the UI already produces.

    The engine passes ``None`` for plain alerts and a
    ``list[list[{"text","callback_data"}]]`` for button replies. Anything else is
    dropped rather than forwarded, so a malformed keyboard cannot become an API
    rejection or, worse, a keyboard with attacker-chosen callback data.
    """
    if not keyboard:
        return None
    try:
        clean = _clean_keyboard(keyboard)
    except Exception:  # noqa: BLE001
        return None
    return clean or None


#: Telegram's hard ``sendMessage`` limit, in characters. Exceeding it is a
#: 400 Bad Request, NOT a permission problem -- and the old code reported it as
#: ``BotApiForbidden``, which pointed the operator at the bot token instead of
#: at the payload.
TELEGRAM_MAX_MESSAGE_CHARS = 4096

#: Room left for the truncation marker.
_TRUNCATION_NOTICE = "\n\n[truncated for Telegram]" if False else (
    "\n\n[... truncated for Telegram]")


def _clamp_to_telegram(text: str, limit: int = TELEGRAM_MAX_MESSAGE_CHARS
                       ) -> str:
    """
    Clamp a notification to what Telegram will actually accept.

    A refusal detail can be arbitrarily long -- a pre-trade safety message
    embeds its own explanation twice -- and an over-long ``sendMessage`` is
    refused by the API with 400. Truncating HERE, at the queue boundary, keeps
    the operator's alert instead of dropping it entirely.

    Cut on a character budget rather than guessing at a tag boundary: this is
    an operator alert, and half an escaped HTML tag is a worse outcome than a
    visibly truncated line. The visible notice tells the reader it happened.
    """
    body = text or ""
    if len(body) <= limit:
        return body
    keep = max(0, limit - len(_TRUNCATION_NOTICE))
    return body[:keep] + _TRUNCATION_NOTICE
