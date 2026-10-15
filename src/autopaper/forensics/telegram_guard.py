"""
Telegram reader crash forensics: context capture and per-message isolation.

THE DEFECT, AS OBSERVED
=======================
``logs/runner.err`` records 26 occurrences of::

    telegram stream error: unsupported operand type(s) for *: 'NoneType' and 'int'

That line is emitted from exactly one place: the ``except Exception`` handler
around the read loop in ``autopaper.telegram.client.TelegramListener.run``. So
the failure is provably inside the read loop's try block -- either the message
iterator or the per-message handler. Two facts follow, and they are the whole
reason this module exists:

1. **The message stream DIED.** ``run()``'s handler sets
   ``status.state = ERROR`` and the ``finally`` sets ``STOPPED``. Recovery is the
   supervisor's reconnect, which re-anchors on the last processed message id --
   so anything that arrived inside the gap is not replayed. A crash is therefore
   a SIGNAL LOSS event, not just a log line.

2. **The exception's type, message, value, traceback and the message being
   handled were all discarded.** The log kept ``str(exc)``. That is why the
   failing call site could never be identified from the logs, and why 26
   occurrences produced no actionable evidence.

WHAT THIS MODULE DOES
=====================
Two independent, additive protections, both offline-testable:

``CallbackContext``
    Captures everything needed to identify a failure AFTER the fact: the
    exception chain, the full traceback, the message id, the chat, the mint, the
    symbol, the raw text length, and whether the exception was an
    ``asyncio.CancelledError`` (which must still propagate -- shutdown
    behaviour is load-bearing).

``safe_handle``
    Runs one message callback with that context attached, so an exception is
    RECORDED and the loop CONTINUES to the next message instead of unwinding
    the whole stream.

It deliberately does NOT decide anything about trading, parsing, quoting or
strategy. It is a boundary guard.
"""

from __future__ import annotations

import asyncio
import traceback
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

#: Ring-buffer size for retained contexts. Bounded so a persistent failure
#: cannot grow memory without limit; 64 is far more than the 26 observed.
DEFAULT_HISTORY = 64


@dataclass(frozen=True)
class CallbackContext:
    """
    Everything known about one message at the moment its callback failed.

    Frozen, so a context cannot be edited after the fact to look different from
    what was actually captured.
    """

    message_id: Optional[int]
    chat_id: Optional[int]
    symbol: Optional[str]
    mint: Optional[str]
    text_length: int
    stage: str
    exception_type: str
    exception_message: str
    exception_repr: str
    traceback_text: str
    cause_type: Optional[str] = None
    cause_message: Optional[str] = None
    is_cancellation: bool = False

    @classmethod
    def capture(
        cls,
        exc: BaseException,
        *,
        stage: str,
        message: Any = None,
        text: str | None = None,
    ) -> "CallbackContext":
        """
        Snapshot ``exc`` into an immutable, fully-populated context.

        Uses ``traceback.format_exception`` on the WHOLE chain, so a wrapped or
        re-raised exception still names its origin rather than only its outermost
        frame. ``__cause__``/``__context__`` are captured separately because a
        bare ``str(exc)`` loses them entirely -- which is precisely how this bug
        became untraceable.
        """
        tb = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        cause = exc.__cause__ or exc.__context__
        return cls(
            message_id=getattr(message, "id", None),
            chat_id=getattr(message, "chat_id", None),
            symbol=_maybe(getattr(message, "symbol", None)),
            mint=_maybe(getattr(message, "mint", None)),
            text_length=len(text) if isinstance(text, str) else 0,
            stage=stage,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            exception_repr=repr(exc),
            traceback_text=tb.rstrip(),
            cause_type=type(cause).__name__ if cause is not None else None,
            cause_message=str(cause) if cause is not None else None,
            is_cancellation=isinstance(exc, asyncio.CancelledError),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "chat_id": self.chat_id,
            "symbol": self.symbol,
            "mint": self.mint,
            "text_length": self.text_length,
            "stage": self.stage,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "exception_repr": self.exception_repr,
            "cause_type": self.cause_type,
            "cause_message": self.cause_message,
            "is_cancellation": self.is_cancellation,
            "traceback": self.traceback_text,
        }

    def one_line(self) -> str:
        """The single line an operator reads. The traceback goes to the log."""
        return (
            f"stage={self.stage} msg={self.message_id} "
            f"chat={self.chat_id} sym={self.symbol or '-'} "
            f"mint={(self.mint or '-')[:10]} text_len={self.text_length} "
            f"{self.exception_type}: {self.exception_message}"
            + (f" | caused by {self.cause_type}: {self.cause_message}"
               if self.cause_type else "")
        )


def _maybe(v: Any) -> Optional[str]:
    return None if v is None else str(v)


@dataclass
class FailureLog:
    """Bounded, in-memory record of isolated callback failures."""

    history: int = DEFAULT_HISTORY
    contexts: list[CallbackContext] = field(default_factory=list)
    counts_by_type: dict[str, int] = field(default_factory=dict)
    counts_by_stage: dict[str, int] = field(default_factory=dict)

    def record(self, ctx: CallbackContext) -> None:
        self.contexts.append(ctx)
        if len(self.contexts) > self.history:
            del self.contexts[: len(self.contexts) - self.history]
        self.counts_by_type[ctx.exception_type] = (
            self.counts_by_type.get(ctx.exception_type, 0) + 1)
        self.counts_by_stage[ctx.stage] = self.counts_by_stage.get(ctx.stage, 0) + 1

    @property
    def total(self) -> int:
        return sum(self.counts_by_type.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_failures": self.total,
            "retained_contexts": len(self.contexts),
            "history_limit": self.history,
            "by_exception_type": dict(self.counts_by_type),
            "by_stage": dict(self.counts_by_stage),
            "contexts": [c.as_dict() for c in self.contexts],
        }

    def clear(self) -> None:
        self.contexts.clear()
        self.counts_by_type.clear()
        self.counts_by_stage.clear()


async def safe_handle(
    handle: Callable[[Any], Awaitable[None]],
    message: Any,
    *,
    text: str | None = None,
    failures: FailureLog,
    stage: str = "handle",
    on_failure: Optional[Callable[[CallbackContext], None]] = None,
) -> bool:
    """
    Run one message callback so that a failure is recorded, not fatal.

    Returns ``True`` when the callback completed and ``False`` when it raised.
    The loop CONTINUES either way, which is the entire point: the previous
    behaviour let one bad message unwind the reader and cost every message that
    arrived before the supervisor reconnected.

    ``asyncio.CancelledError`` is re-raised. Shutdown must still be able to stop
    the process, so cancellation is never treated as an isolated callback error.
    """
    try:
        await handle(message)
        return True
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - isolation boundary, by design
        ctx = CallbackContext.capture(exc, stage=stage, message=message, text=text)
        failures.record(ctx)
        if on_failure is not None:
            try:
                on_failure(ctx)
            except Exception:  # noqa: BLE001
                # A broken reporter must not become a second failure mode.
                pass
        return False


def describe_failure(ctx: CallbackContext) -> str:
    """
    The log block written for an isolated failure.

    Includes the full traceback, because the whole historical problem was that
    it was not there. Kept as a function so the exact text is asserted by tests
    rather than trusted.
    """
    return (
        f"telegram callback failed, stream CONTINUES: {ctx.one_line()}\n"
        f"--- traceback ---\n{ctx.traceback_text}\n"
        f"--- end traceback ---"
    )


# ---------------------------------------------------------------------------
# the observed defect, recorded as data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArithmeticHazard:
    """One ``X * <int>`` site and whether X can be None there."""

    file: str
    line: int
    expression: str
    operand: str
    reachable_from_reader: bool
    can_be_none_in_current_tree: bool
    note: str


#: Produced by the AST sweep in ``reports/forensics/telegram_reader_crash_audit.md``.
#: Recorded here so the finding is machine-checkable rather than prose.
UNGUARDED_MULTIPLICATION_SITES: tuple[ArithmeticHazard, ...] = (
    ArithmeticHazard(
        file="src/autopaper/telegram/client.py", line=294,
        expression="min(reconnect_backoff_max_s, backoff * 2)",
        operand="backoff", reachable_from_reader=True,
        can_be_none_in_current_tree=False,
        note="backoff is seeded from TelegramConfig.reconnect_backoff_s, whose "
             "dataclass default is 5.0 and which config._as_float can never "
             "return as None. Not the observed crash, but it is the only "
             "unguarded X * <int> in the reader's own reconnect path, so it "
             "is hardened rather than left as a latent trap.",
    ),
    ArithmeticHazard(
        file="src/autopaper/engine.py", line=4921,
        expression="min(reconnect_backoff_max_s, backoff * 2)",
        operand="backoff", reachable_from_reader=True,
        can_be_none_in_current_tree=False,
        note="the supervisor's copy of the same pattern; hardened together "
             "with the listener's.",
    ),
    ArithmeticHazard(
        file="src/autopaper/engine.py", line=2339,
        expression="latency_s * 1000", operand="latency_s",
        reachable_from_reader=True, can_be_none_in_current_tree=False,
        note="seeded from ExecutionModel.decision_latency_ms (default 1200.0).",
    ),
    ArithmeticHazard(
        file="src/autopaper/market/base.py", line=59,
        expression="config.cache_ttl_s * 2", operand="config.cache_ttl_s",
        reachable_from_reader=True, can_be_none_in_current_tree=False,
        note="MarketConfig.cache_ttl_s default 6.0; _as_float cannot yield None.",
    ),
)


def coerce_positive_float(value: Any, default: float) -> float:
    """
    The shared guard for the sites above.

    Accepts a finite, strictly positive number; returns ``default`` for anything
    else, INCLUDING ``None``. Applied at the point of arithmetic so a ``None``
    structurally cannot reach a multiplication, which is what the regression
    tests assert.
    """
    if value is None or isinstance(value, bool):
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float("inf"), float("-inf")):
        return default
    if f <= 0.0:
        return default
    return f
