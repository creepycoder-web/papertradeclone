"""
Command routing and the runtime lifecycle, driven through a REAL engine.

The point of these tests is that the Telegram path is not a parallel
implementation. Each one drives the actual ``PaperTrader`` through the actual
``BotControl`` and asserts on the actual ``RuntimeState``, so a future change
that routes Telegram around the existing control path fails here.

No wallet, no RPC, no network, no signing.
"""

from __future__ import annotations

import asyncio

import pytest

from autopaper.config import build_config
from autopaper.engine import PaperTrader
from autopaper.l25 import ACTIVE_STRATEGY
from autopaper.modes import REAL_CONFIRMATION_TTL_S, RunState, TradeMode
from autopaper.telegram import botapi
from autopaper.telegram.botapi import BotControl

ALLOWED = -1001234567890
FOREIGN = -1009999999999


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class TruncClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt

    def iso(self, t: float | None = None) -> str:
        return "2026-09-30T12:00:00Z"


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """
    A real engine with a bot token and chat configured, OFF and un-armed.

    The token is fake and never leaves the process; no network call is made
    because nothing is started.
    """
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "999:FAKEfakefakefakefakefakefake")
    monkeypatch.setenv("TELEGRAM_BOT_CHAT", str(ALLOWED))
    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    trader = PaperTrader(build_config(tmp_path))
    try:
        yield trader
    finally:
        trader._release_ownership()


@pytest.fixture
def control(engine):
    """A BotControl whose handler is the engine's own handle_button."""
    return BotControl(
        transport=engine.bot_transport,
        handle_button=engine.handle_button,
        allowed_chat=ALLOWED,
    )


def cb(control, data="start", cb_id="cb-1", uid=1, chat=ALLOWED):
    return asyncio.run(control.handle_update({
        "update_id": uid,
        "callback_query": {"id": cb_id, "data": data,
                           "message": {"chat": {"id": chat}}},
    }))


def msg(control, text, uid=1, chat=ALLOWED):
    return asyncio.run(control.handle_update({
        "update_id": uid,
        "message": {"text": text, "chat": {"id": chat}}},
    ))


# ---------------------------------------------------------------------------
# the wiring exists
# ---------------------------------------------------------------------------


def test_the_transport_is_built_and_attached(engine):
    assert engine.bot_transport is not None
    assert engine.bot_notifier is not None
    assert engine.bot_control is not None
    # attached through the EXISTING hook (wrapped in the guard, so compare the
    # underlying target rather than the wrapper)
    assert engine._notifier == engine.bot_notifier.send


def test_no_token_means_no_control_channel(tmp_path, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_CHAT", raising=False)
    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    t = PaperTrader(build_config(tmp_path))
    try:
        assert t.bot_control is None
        assert t.bot_transport is None
    finally:
        t._release_ownership()


def test_a_token_without_a_chat_does_not_open_the_channel(tmp_path, monkeypatch):
    """
    With a token but no allow-listed chat there is nothing to authorise, and
    admitting every chat would be the worst possible default.
    """
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "999:FAKE")
    monkeypatch.delenv("TELEGRAM_BOT_CHAT", raising=False)
    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    t = PaperTrader(build_config(tmp_path))
    try:
        assert t.bot_control is None
    finally:
        t._release_ownership()


# ---------------------------------------------------------------------------
# START / STOP
# ---------------------------------------------------------------------------


def test_start_moves_off_to_paper_running(engine, control):
    assert engine.state.run_state is RunState.OFF
    assert cb(control, "start") is True
    assert engine.state.run_state is RunState.PAPER
    assert engine.state.running is True


def test_start_is_still_routable_as_a_callback(engine, control):
    """
    START is no longer a TYPED command, but the internal callback remains.

    The bot vocabulary was reduced to five operator commands; START still
    exists as a callback because the dashboard keyboard and the audited REAL
    flow use it. This test pins that distinction.
    """
    assert cb(control, "start") is True
    assert engine.state.run_state is RunState.PAPER


def test_start_enables_new_entries(engine, control):
    """The real gate, not a proxy for it."""
    before, _ = engine.state.can_open_new_entries()
    assert before is False
    cb(control, "start")
    after, _ = engine.state.can_open_new_entries()
    assert after is True


def test_stop_returns_to_a_non_trading_state(engine, control):
    cb(control, "start")
    assert engine.state.running is True
    assert cb(control, "stop", cb_id="cb-2", uid=2) is True
    assert engine.state.run_state is RunState.STOPPED
    allowed, _ = engine.state.can_open_new_entries()
    assert allowed is False


def test_the_full_off_start_stop_cycle(engine, control):
    assert engine.state.run_state is RunState.OFF
    assert cb(control, "start") is True
    assert engine.state.run_state is RunState.PAPER
    assert cb(control, "stop", cb_id="s2", uid=2) is True
    assert engine.state.run_state is RunState.STOPPED
    allowed, _ = engine.state.can_open_new_entries()
    assert allowed is False, "STOP must leave no partially-running state"


def test_status_answers_without_changing_state(engine, control):
    before = engine.state.run_state
    assert cb(control, "stats", cb_id="q1", uid=1) is True
    assert engine.state.run_state is before
    assert engine.state.real_trading_enabled is False


def test_an_unknown_command_changes_nothing(engine, control):
    before = engine.state.run_state
    assert cb(control, "definitely_not_a_command") is True  # handled + refused
    assert engine.state.run_state is before
    assert engine.state.real_trading_enabled is False


# ---------------------------------------------------------------------------
# unauthorized
# ---------------------------------------------------------------------------


def test_an_unauthorized_start_leaves_the_engine_off(engine, control):
    assert cb(control, "start", chat=FOREIGN) is False
    assert engine.state.run_state is RunState.OFF
    assert engine.state.running is False
    allowed, _ = engine.state.can_open_new_entries()
    assert allowed is False


def test_an_unauthorized_stop_changes_nothing(engine, control):
    cb(control, "start")
    assert cb(control, "stop", cb_id="x2", uid=2, chat=FOREIGN) is False
    assert engine.state.run_state is RunState.PAPER, "a stranger must not stop it either"


def test_an_unauthorized_real_request_arms_nothing(engine, control):
    assert cb(control, "real", cb_id="r1", uid=1, chat=FOREIGN) is False
    assert engine.state.real_armed is False
    assert engine.state._pending_confirmation is None


# ---------------------------------------------------------------------------
# malformed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("junk", [
    "not a mapping", 42, None, [], {},
    {"update_id": 1},
    {"update_id": 1, "callback_query": {"id": "x"}},
    {"update_id": 1, "message": {"text": "/nonsense",
                                 "chat": {"id": ALLOWED}}},
])
def test_malformed_input_leaves_the_engine_off(engine, control, junk):
    assert asyncio.run(control.handle_update(junk)) is False
    assert engine.state.run_state is RunState.OFF
    assert engine.state.real_trading_enabled is False


def test_a_malformed_start_never_starts_the_engine(engine, control):
    asyncio.run(control.handle_update({
        "update_id": 1,
        "callback_query": {"id": "x", "data": "  ",
                           "message": {"chat": {"id": ALLOWED}}}}))
    assert engine.state.run_state is RunState.OFF


# ---------------------------------------------------------------------------
# duplicates
# ---------------------------------------------------------------------------


def test_a_duplicate_start_leaves_one_runtime(engine, control):
    assert cb(control, "start") is True
    assert cb(control, "start", cb_id="cb-1", uid=1) is False   # same update
    assert cb(control, "start", cb_id="cb-1", uid=2) is False   # same callback
    assert cb(control, "start", cb_id="cb-9", uid=9) is True    # a NEW press
    assert engine.state.run_state is RunState.PAPER
    assert engine.state.running is True


def test_a_duplicate_start_does_not_create_a_second_loop(engine, control):
    """
    A real press, not a replay, also reaches the handler - and starting an
    already-running state is refused by the state machine, not duplicated.
    """
    for uid, cbid in ((1, "a"), (2, "b"), (3, "c")):
        cb(control, "start", cb_id=cbid, uid=uid)
    assert engine.state.run_state is RunState.PAPER
    # no extra task was created by the operator presses
    assert engine._bot_task is None, "operator START must not spawn a poll loop"


# ---------------------------------------------------------------------------
# Telegram failure leaves state consistent
# ---------------------------------------------------------------------------


def test_a_network_failure_while_polling_leaves_state_consistent(engine):
    """A dead Telegram must not corrupt or advance the runtime."""
    from autopaper.telegram.botapi import BotApiTransient

    async def boom(*a, **k):
        raise BotApiTransient("network down")

    engine.bot_transport.get_updates = boom
    c = engine.bot_control
    with pytest.raises(BotApiTransient):
        asyncio.run(c.poll_once(timeout_s=1))
    assert engine.state.run_state is RunState.OFF
    assert engine.state.real_trading_enabled is False
    allowed, _ = engine.state.can_open_new_entries()
    assert allowed is False


def test_a_failing_notifier_does_not_break_a_command(engine, control):
    """A broken notifier must not fail the operator's press."""
    def boom(text, kb=None):
        raise RuntimeError("telegram is down")

    engine._notify = boom
    assert cb(control, "start") is True, "the press still takes effect"
    assert engine.state.run_state is RunState.PAPER


def test_a_failing_notifier_does_not_break_a_trade_decision(engine):
    """
    Every internal notify call site is protected by construction.

    ``set_notifier`` wraps the callable, so a Telegram outage cannot turn into a
    failed decision at ANY of the seven internal call sites - not just the one
    that remembered to guard itself.
    """
    def boom(text, kb=None):
        raise RuntimeError("telegram is down")

    engine.set_notifier(boom)
    engine.state.start()
    for i in range(20):
        engine._notify(f"alert {i}", None)     # must not raise
    assert engine.state.run_state is RunState.PAPER
    assert engine.state.real_trading_enabled is False
    allowed, _ = engine.state.can_open_new_entries()
    assert allowed is True


def test_a_poll_does_not_advance_the_offset_when_it_fails(engine):
    """Otherwise a transient outage would silently skip updates."""
    from autopaper.telegram.botapi import BotApiTransient

    async def boom(*a, **k):
        raise BotApiTransient("down")

    engine.bot_transport.get_updates = boom
    c = engine.bot_control
    assert c._offset is None
    with pytest.raises(BotApiTransient):
        asyncio.run(c.poll_once(timeout_s=1))
    assert c._offset is None


# ---------------------------------------------------------------------------
# restart
# ---------------------------------------------------------------------------


def test_a_restart_does_not_restore_a_started_or_armed_state(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "999:FAKEfake")
    monkeypatch.setenv("TELEGRAM_BOT_CHAT", str(ALLOWED))
    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    cfg = build_config(tmp_path)

    first = PaperTrader(cfg)
    try:
        c = BotControl(transport=first.bot_transport,
                       handle_button=first.handle_button, allowed_chat=ALLOWED)
        asyncio.run(c.handle_update({
            "update_id": 1,
            "callback_query": {"id": "s", "data": "start",
                               "message": {"chat": {"id": ALLOWED}}}}))
        assert first.state.run_state is RunState.PAPER
    finally:
        first._release_ownership()

    second = PaperTrader(cfg)
    try:
        assert second.state.run_state is RunState.OFF
        assert second.state.running is False
        assert second.state.real_trading_enabled is False
        allowed, _ = second.state.can_open_new_entries()
        assert allowed is False
    finally:
        second._release_ownership()
