from __future__ import annotations

from autopaper.telegram.signal_source import (
    FALLBACK_ACTIVE,
    PRIMARY_ACTIVE,
    STARTUP_UNKNOWN,
    SignalSourceState,
)


def test_startup_does_not_activate_fallback_without_primary_evidence():
    state = SignalSourceState()
    assert state.refresh(100.0) is None
    assert state.active_source == STARTUP_UNKNOWN
    assert not state.accepts("PainApe", 100.0)


def test_exact_threshold_is_still_primary_and_later_is_fallback():
    state = SignalSourceState()
    state.observe_primary(at=100.0, message_id=7)
    assert state.active_source == PRIMARY_ACTIVE
    assert state.refresh(100.0 + 5 * 60 * 60) is None
    assert state.active_source == PRIMARY_ACTIVE
    assert state.refresh(100.0 + 5 * 60 * 60 + 0.001) == FALLBACK_ACTIVE


def test_primary_signal_reactivates_without_flapping():
    state = SignalSourceState()
    state.observe_primary(at=0.0)
    assert state.refresh(18_001.0) == FALLBACK_ACTIVE
    assert state.observe_primary(at=18_002.0, message_id=9) == PRIMARY_ACTIVE
    assert state.refresh(18_002.0) is None


def test_state_round_trips_persistence():
    state = SignalSourceState()
    state.observe_primary(at=123.0, message_id=42)
    restored = SignalSourceState.from_dict(state.as_dict())
    assert restored.as_dict() == state.as_dict()
