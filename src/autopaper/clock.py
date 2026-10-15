"""
Single time authority.

Every timestamp in the agent comes from here, which is what makes the latency
ladder and the no-lookahead guarantee auditable. In live mode this is wall
clock. In tests it is a ``FakeClock`` that can be advanced explicitly, so a test
can prove that a decision at T never saw an observation stamped after T.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

# Telegram sends second-resolution unix timestamps; the rest of the ladder is
# millisecond resolution. All internal times are float unix seconds (UTC).


class Clock(Protocol):
    def now(self) -> float:
        ...

    def monotonic(self) -> float:
        ...


@dataclass
class SystemClock:
    """Wall clock with a monotonic tiebreaker."""

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()


@dataclass
class FakeClock:
    """Deterministic clock for tests. Time only moves when a test moves it."""

    t: float = 1_700_000_000.0
    _mono: float = 0.0
    auto_advance: float = 0.0
    _offsets: list[float] = field(default_factory=list)

    def now(self) -> float:
        if self.auto_advance:
            self.t += self.auto_advance
        return self.t

    def monotonic(self) -> float:
        self._mono += 0.0
        return self._mono

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("FakeClock cannot move backwards")
        self.t += seconds
        return self.t

    def advance_ms(self, ms: float) -> float:
        return self.advance(ms / 1000.0)

    def set(self, t: float) -> None:
        if t < self.t:
            raise ValueError("FakeClock cannot move backwards")
        self.t = t


def iso(ts: float | None) -> str | None:
    """Human-readable UTC ISO-8601 with millisecond precision."""
    if ts is None:
        return None
    whole = int(ts)
    msec = int(round((ts - whole) * 1000))
    if msec >= 1000:  # rounding spill
        whole += 1
        msec -= 1000
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(whole)) + f".{msec:03d}Z"


def hms(ts: float | None) -> str:
    if ts is None:
        return "--:--:--"
    return time.strftime("%H:%M:%S", time.localtime(ts))


def age_ms(now: float, then: float | None) -> float | None:
    if then is None:
        return None
    return max(0.0, (now - then) * 1000.0)
