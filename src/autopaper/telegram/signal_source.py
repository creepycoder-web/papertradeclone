"""Durable primary/fallback signal-source state.

This module deliberately contains no strategy or execution policy.  It only
answers which Telegram feed is currently eligible to enter the existing
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PRIMARY = "MooDeng"
FALLBACK = "PainApe"
AMARIH = "Amarih"
SOURCE_AUTO = "AUTO"
SOURCE_MOODENG = PRIMARY
SOURCE_PAINAPE = FALLBACK
SOURCE_AMARIH = AMARIH
PRIMARY_ACTIVE = "PRIMARY_ACTIVE"
FALLBACK_ACTIVE = "FALLBACK_ACTIVE"
AMARIH_ACTIVE = "AMARIH_ACTIVE"
STARTUP_UNKNOWN = "STARTUP_UNKNOWN"
DEFAULT_THRESHOLD_S = 5 * 60 * 60
OPPORTUNITY_WINDOW_S = 300.0


@dataclass
class SignalSourceState:
    threshold_s: float = DEFAULT_THRESHOLD_S
    primary_last_signal_at: float | None = None
    primary_last_message_id: int | None = None
    primary_last_signal_source: str = PRIMARY
    active_source: str = STARTUP_UNKNOWN
    transitions: int = 0

    def refresh(self, now: float) -> str | None:
        previous = self.active_source
        if self.primary_last_signal_at is None:
            current = STARTUP_UNKNOWN
        else:
            age = max(0.0, now - self.primary_last_signal_at)
            # Exactly five hours is still primary; fallback starts strictly
            # after the configured threshold.
            current = FALLBACK_ACTIVE if age > self.threshold_s else PRIMARY_ACTIVE
        self.active_source = current
        if current != previous:
            self.transitions += 1
            return current
        return None

    def observe_amarih(self, *, at: float, message_id: int | None = None) -> None:
        """Record an Amarih signal receipt from PumpView.

        Amarih is an independent research source; it does not affect the
        primary/fallback transition state but is tracked for status.
        """
        self.primary_last_signal_at = at
        self.primary_last_message_id = message_id

    @property
    def amarih_last_signal_at(self) -> float | None:
        return self.primary_last_signal_at

    @property
    def amarih_last_message_id(self) -> int | None:
        return self.primary_last_message_id

    def observe_primary(self, *, at: float, message_id: int | None = None) -> str | None:
        self.primary_last_signal_at = at
        self.primary_last_message_id = message_id
        return self.refresh(at)

    def accepts(self, source: str, now: float) -> bool:
        self.refresh(now)
        if source == PRIMARY:
            return self.active_source in (PRIMARY_ACTIVE, STARTUP_UNKNOWN)
        if source == FALLBACK:
            return self.active_source == FALLBACK_ACTIVE
        if source == AMARIH:
            return True
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold_s": self.threshold_s,
            "primary_last_signal_at": self.primary_last_signal_at,
            "primary_last_message_id": self.primary_last_message_id,
            "primary_last_signal_source": self.primary_last_signal_source,
            "active_source": self.active_source,
            "transitions": self.transitions,
            "amarih_last_signal_at": self.amarih_last_signal_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "SignalSourceState":
        raw = raw or {}
        return cls(
            threshold_s=float(raw.get("threshold_s", DEFAULT_THRESHOLD_S)),
            primary_last_signal_at=(
                float(raw["primary_last_signal_at"])
                if raw.get("primary_last_signal_at") is not None else None
            ),
            primary_last_message_id=(
                int(raw["primary_last_message_id"])
                if raw.get("primary_last_message_id") is not None else None
            ),
            primary_last_signal_source=str(raw.get("primary_last_signal_source", PRIMARY)),
            active_source=str(raw.get("active_source", STARTUP_UNKNOWN)),
            transitions=int(raw.get("transitions", 0)),
        )
