"""
Latency instrumentation: monotonic for durations, wall clock for correlation.

The rule this module enforces is the one that makes latency numbers meaningful:
**a duration is always a difference of two readings from the same monotonic
clock.** Wall clock is recorded alongside for correlation with logs and
reports, and is used for exactly one duration - Telegram transport latency -
because that genuinely spans two machines and no local monotonic clock can
cover it. That one is reported as approximate, and its imprecision is stated
rather than hidden.

Telegram stamps messages with whole-second resolution, so a "transport latency"
measured against it carries up to 1s of quantisation. Reporting that as a
precise millisecond figure would be inventing resolution the source does not
have, so it is labelled and quantified.

Every stage records both readings even when only one is used, so a stage that
later turns out to need a monotonic duration does not have to be re-run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

#: The stages the harness records, in pipeline order. Used by the renderer to
#: guarantee a stage is never silently missing from the report.
STAGES: tuple[str, ...] = (
    "receive",
    "parse_start",
    "parse_end",
    "dex_request_start",
    "dex_response",
    "rpc_request_start",
    "rpc_response",
    "pool_state",
    "quote_start",
    "quote_end",
    "jupiter_request_start",
    "jupiter_response",
    "decision",
)

#: Telegram's message timestamp is second-resolution, so a transport latency
#: derived from it is only accurate to within this much in the best case.
TELEGRAM_TIMESTAMP_QUANTUM_S = 1.0


class ClockError(RuntimeError):
    """A duration was requested from a stage that was never recorded."""


@dataclass
class StageTrace:
    """
    Dual-clock stage recorder.

    ``wall`` is for correlation: which second of wall time a stage belongs to.
    ``mono`` is for duration: immune to NTP steps and to the operator's clock
    being wrong, which is not hypothetical on a machine that has been running a
    trading process for weeks.
    """

    wall: dict[str, float] = field(default_factory=dict)
    mono: dict[str, float] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    # -- recording ---------------------------------------------------------

    def mark(self, stage: str) -> float:
        """Record a stage. Re-marking overwrites, which is what a retry wants."""
        if stage not in self.order:
            self.order.append(stage)
        now_wall = time.time()
        now_mono = time.monotonic()
        self.wall[stage] = now_wall
        self.mono[stage] = now_mono
        return now_wall

    def mark_at(self, stage: str, wall: float | None = None,
                mono: float | None = None) -> float:
        """
        Record a stage with explicit readings.

        Used when a timestamp came from a remote source (Telegram) or when a
        test drives the trace deterministically.
        """
        if stage not in self.order:
            self.order.append(stage)
        w = time.time() if wall is None else float(wall)
        m = time.monotonic() if mono is None else float(mono)
        self.wall[stage] = w
        self.mono[stage] = m
        return w

    # -- reading -----------------------------------------------------------

    def wall_at(self, stage: str) -> float | None:
        return self.wall.get(stage)

    def duration_ms(self, start: str, end: str) -> float | None:
        """
        Milliseconds between two stages, from the monotonic clock.

        ``None`` when either stage is missing. Never zero as a stand-in for
        "unknown", and never computed across a wall-clock step.
        """
        a = self.mono.get(start)
        b = self.mono.get(end)
        if a is None or b is None:
            return None
        return max(0.0, (b - a) * 1000.0)

    def elapsed_since_ms(self, stage: str, now_mono: float | None = None) -> float | None:
        a = self.mono.get(stage)
        if a is None:
            return None
        b = time.monotonic() if now_mono is None else now_mono
        return max(0.0, (b - a) * 1000.0)

    def ordering_violations(self) -> list[str]:
        """
        Stages recorded out of order on the monotonic clock.

        A non-empty list means the trace is not trustworthy - for example a
        pool state read before the RPC request that supposedly produced it -
        and the caller should treat the associated durations as suspect rather
        than reporting a negative or absurd number.
        """
        seen = [(s, self.mono[s]) for s in self.order if s in self.mono]
        violations: list[str] = []
        for (a_s, a_m), (b_s, b_m) in zip(seen, seen[1:]):
            if b_m < a_m:
                violations.append(f"{b_s} ({b_m:.6f}) precedes {a_s} ({a_m:.6f})")
        return violations

    # -- serialisation -----------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "stages": list(self.order),
            "wall": {k: v for k, v in self.wall.items()},
            "mono": {k: v for k, v in self.mono.items()},
            "ordering_violations": self.ordering_violations(),
        }


@dataclass(frozen=True)
class LatencyBreakdown:
    """
    The six required figures, plus the provenance of each.

    ``telegram_to_receive_ms`` is the only one derived from wall clocks and it
    is marked approximate for the reason above. The rest are monotonic and are
    exact within the clock's resolution.
    """

    telegram_to_receive_ms: float | None
    receive_to_mark_ms: float | None
    mark_to_state_ms: float | None
    state_to_quote_ms: float | None
    quote_to_decision_ms: float | None
    total_signal_to_decision_ms: float | None
    parsing_ms: float | None = None
    dexscreener_ms: float | None = None
    rpc_ms: float | None = None
    pool_state_ms: float | None = None
    quote_ms: float | None = None
    jupiter_ms: float | None = None
    transport_latency_approximate: bool = True
    ordering_violations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "telegram_to_receive_ms": self.telegram_to_receive_ms,
            "telegram_to_receive_approximate": self.transport_latency_approximate,
            "receive_to_mark_ms": self.receive_to_mark_ms,
            "mark_to_state_ms": self.mark_to_state_ms,
            "state_to_quote_ms": self.state_to_quote_ms,
            "quote_to_decision_ms": self.quote_to_decision_ms,
            "total_signal_to_decision_ms": self.total_signal_to_decision_ms,
            "parsing_ms": self.parsing_ms,
            "dexscreener_ms": self.dexscreener_ms,
            "rpc_ms": self.rpc_ms,
            "pool_state_ms": self.pool_state_ms,
            "quote_ms": self.quote_ms,
            "jupiter_ms": self.jupiter_ms,
            "ordering_violations": list(self.ordering_violations),
        }


def build_breakdown(
    trace: StageTrace,
    *,
    telegram_message_ts: float | None = None,
) -> LatencyBreakdown:
    """
    Derive the reported latency figures from a recorded trace.

    A missing stage yields ``None``, which renders as ``UNKNOWN``. It never
    yields ``0``: a zero would claim the step was instantaneous, which is a
    stronger and more wrong statement than admitting it was not measured.
    """
    receive_wall = trace.wall_at("receive")
    transport: float | None = None
    if telegram_message_ts is not None and receive_wall is not None:
        # Wall clocks on two hosts: approximate by construction, and Telegram's
        # own stamp is second-resolution.
        transport = max(0.0, (receive_wall - float(telegram_message_ts)) * 1000.0)

    return LatencyBreakdown(
        telegram_to_receive_ms=transport,
        receive_to_mark_ms=trace.duration_ms("receive", "dex_response"),
        mark_to_state_ms=trace.duration_ms("dex_response", "pool_state"),
        state_to_quote_ms=trace.duration_ms("pool_state", "quote_end"),
        quote_to_decision_ms=trace.duration_ms("quote_end", "decision"),
        total_signal_to_decision_ms=trace.duration_ms("receive", "decision"),
        parsing_ms=trace.duration_ms("parse_start", "parse_end"),
        dexscreener_ms=trace.duration_ms("dex_request_start", "dex_response"),
        rpc_ms=trace.duration_ms("rpc_request_start", "rpc_response"),
        pool_state_ms=trace.duration_ms("rpc_response", "pool_state"),
        quote_ms=trace.duration_ms("quote_start", "quote_end"),
        jupiter_ms=trace.duration_ms("jupiter_request_start", "jupiter_response"),
        transport_latency_approximate=True,
        ordering_violations=tuple(trace.ordering_violations()),
    )


def timed(trace: StageTrace, start_stage: str, end_stage: str,
          fn: Callable[[], Any]) -> Any:
    """
    Run ``fn`` between two marks. For synchronous call sites.
    """
    trace.mark(start_stage)
    try:
        return fn()
    finally:
        trace.mark(end_stage)


async def timed_async(trace: StageTrace, start_stage: str, end_stage: str,
                      coro_fn: Callable[[], Any]) -> Any:
    """
    Run ``await coro_fn()`` between two marks.

    The end mark is written in a ``finally``, so a failed call is still
    bracketed and its latency is measured rather than silently lost.
    """
    trace.mark(start_stage)
    try:
        return await coro_fn()
    finally:
        trace.mark(end_stage)


__all__ = [
    "ClockError",
    "LatencyBreakdown",
    "STAGES",
    "StageTrace",
    "TELEGRAM_TIMESTAMP_QUANTUM_S",
    "build_breakdown",
    "timed",
    "timed_async",
]