from __future__ import annotations

import asyncio

from autopaper.modes import RuntimeState
from autopaper.telegram import ui
from autopaper.telegram.actions import ButtonHandler, OperatorActions
from autopaper.telegram.signal_source import FALLBACK_ACTIVE, SignalSourceState


def _handler(saved: list[str]):
    state = RuntimeState()
    return state, ButtonHandler(
        state,
        OperatorActions(
            persist_signal_source=saved.append,
            current_signal_source=lambda: saved[-1] if saved else "AUTO",
        ),
    )


def test_moodeng_selection():
    saved = []
    _, handler = _handler(saved)
    result = asyncio.run(handler.handle(f"{ui.CB_SOURCE_SELECT}MooDeng"))
    assert result.handled and saved == ["MooDeng"]
    assert "SOURCE: <b>MOODENG</b>" in result.text


def test_painape_selection():
    saved = []
    _, handler = _handler(saved)
    result = asyncio.run(handler.handle(f"{ui.CB_SOURCE_SELECT}PainApe"))
    assert result.handled and saved == ["PainApe"]
    assert "SOURCE: <b>PAINAPE</b>" in result.text


def test_smugcalls_is_no_longer_selectable():
    """The retired fallback must not be accepted by the selector anymore."""
    saved = []
    _, handler = _handler(saved)
    result = asyncio.run(handler.handle(f"{ui.CB_SOURCE_SELECT}SmugCalls"))
    assert result.error == "invalid_signal_source"
    assert saved == []


def test_auto_uses_primary_then_fallback_after_strict_threshold():
    state = SignalSourceState()
    state.observe_primary(at=100.0, message_id=1)
    assert state.refresh(100.0 + 5 * 60 * 60) is None
    assert state.refresh(100.0 + 5 * 60 * 60 + 0.01) == FALLBACK_ACTIVE


def test_selection_persistence_callback_is_invoked():
    saved = []
    _, handler = _handler(saved)
    asyncio.run(handler.handle(f"{ui.CB_SOURCE_SELECT}PainApe"))
    assert saved == ["PainApe"]


def test_switching_source_does_not_touch_listener_high_water_marks():
    saved = []
    _, handler = _handler(saved)
    cursors = {"MooDeng": 101, "PainApe": 202}
    asyncio.run(handler.handle(f"{ui.CB_SOURCE_SELECT}PainApe"))
    assert cursors == {"MooDeng": 101, "PainApe": 202}


def test_source_selection_does_not_touch_existing_runtime_positions():
    saved = []
    runtime, handler = _handler(saved)
    existing = {"position_id": "pos-1", "status": "OPEN", "mint": "M"}
    asyncio.run(handler.handle(f"{ui.CB_SOURCE_SELECT}MooDeng"))
    assert existing == {"position_id": "pos-1", "status": "OPEN", "mint": "M"}
    assert runtime.run_state.name == "OFF"
