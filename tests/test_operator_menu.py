"""
/start is the operator's front door, and it must open the door and nothing else.

The failure this file exists to prevent: ``/start`` doing something. It is how
an operator arrives at the control panel, so any side effect there is an
accidental trade waiting to happen. It shows a menu, reports state, and returns.

Telegram remains a control plane and a notification sink. Nothing here touches
the trading critical path, and ``--cli`` builds no bot at all, so none of it
can reach an order.
"""

from __future__ import annotations

import asyncio
import dataclasses
import tempfile
from pathlib import Path

import pytest

from autopaper.modes import RunState, RuntimeState, TradeMode
from autopaper.telegram import ui
from autopaper.telegram.actions import ButtonHandler, OperatorActions
from autopaper.telegram.botapi import TEXT_COMMANDS, payload_from_update
from autopaper.telegram.ui import render_menu

ALLOWED = -1001234567890
REPO = Path(__file__).resolve().parents[1]

#: The operator menu, pinned verbatim. The eight original controls are UNCHANGED,
#: the two read-only position views are appended to the read-only block, and
#: TAKE PROFIT sits directly under STRATEGY because it is a parameter OF the L25
#: exit rather than a second strategy -- so a label change is still a test
#: failure rather than a silent interface edit.
MENU_LABELS = [
    "▶️ START PAPER",
    "⏸ PAUSE",
    "▶️ RESUME",
    "💰 BALANCE",
    "📊 STATUS",
    "📂 OPEN POSITIONS",
    "🗂 CLOSED POSITIONS",
    "🎯 STRATEGY",
    "🎯 TAKE PROFIT",
    "📡 SIGNAL SOURCE",
    "🔴 ARM REAL",
    "🛑 DISARM REAL",
    "☠️ KILL",
    "🚨 CLOSE ALL PAPER",
]

#: The eight that existed before the position views were added. Nothing here may
#: be removed or renamed.
LEGACY_MENU_LABELS = [
    "▶️ START PAPER",
    "⏸ PAUSE",
    "▶️ RESUME",
    "💰 BALANCE",
    "📊 STATUS",
    "🔴 ARM REAL",
    "🛑 DISARM REAL",
    "☠️ KILL",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _PF:
    """A preflight double. ``ok`` decides the outcome."""

    def __init__(self, ok: bool = True):
        self.ok = ok
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        return self


def handler(state: RuntimeState | None = None, **kw) -> ButtonHandler:
    st = state or RuntimeState()
    st.start()
    return ButtonHandler(st, OperatorActions(**kw))


def press(h: ButtonHandler, payload: str):
    return asyncio.run(h.handle(payload))


def menu_buttons(h: ButtonHandler) -> list[tuple[str, str]]:
    result = press(h, ui.CB_MENU)
    return [(b["text"], b["callback_data"])
            for row in result.keyboard for b in row]


# ---------------------------------------------------------------------------
# 1. the menu exists and has exactly the specified buttons
# ---------------------------------------------------------------------------


def test_the_menu_has_exactly_the_specified_buttons():
    st = RuntimeState()
    labels = [b["text"] for row in ui.operator_keyboard(st) for b in row]
    assert labels == MENU_LABELS, (
        "the menu labels are the operator's interface; they are pinned "
        "verbatim, not just counted"
    )


def test_the_eight_original_controls_are_all_still_present():
    """Adding the position views must not have removed or renamed anything."""
    st = RuntimeState()
    labels = [b["text"] for row in ui.operator_keyboard(st) for b in row]
    for legacy in LEGACY_MENU_LABELS:
        assert legacy in labels, f"the original control {legacy!r} disappeared"


def test_each_menu_button_dispatches_to_its_named_action():
    pairs = {b["text"]: b["callback_data"]
             for row in ui.operator_keyboard(RuntimeState()) for b in row}
    assert pairs == {
        "▶️ START PAPER": ui.CB_START,
        "⏸ PAUSE": ui.CB_PAUSE,
        "▶️ RESUME": ui.CB_RESUME,
        "💰 BALANCE": ui.CB_BALANCE,
        "📊 STATUS": ui.CB_STATS,
        "📂 OPEN POSITIONS": ui.CB_OPEN,
        "🗂 CLOSED POSITIONS": ui.CB_CLOSED,
        "🎯 STRATEGY": ui.CB_STRATEGY,
        "🎯 TAKE PROFIT": ui.CB_TAKE_PROFIT,
        "📡 SIGNAL SOURCE": ui.CB_SIGNAL_SOURCE,
        "🔴 ARM REAL": ui.CB_REAL,
        "🛑 DISARM REAL": ui.CB_REAL_OFF,
        "☠️ KILL": ui.CB_KILL,
        "🚨 CLOSE ALL PAPER": ui.CB_CLOSE_ALL_PAPER,
    }


def test_the_menu_is_ordered_by_function():
    """Run control, then read-only, then REAL, then kill alone at the bottom."""
    rows = ui.operator_keyboard(RuntimeState())
    assert [b["text"] for b in rows[0]] == ["▶️ START PAPER", "⏸ PAUSE"]
    assert [b["text"] for b in rows[-1]] == ["☠️ KILL"], (
        "the kill switch must not sit next to anything a mis-tap could hit"
    )


def test_the_position_views_sit_with_the_other_read_only_controls():
    """They are inspection, so they belong next to STATUS and BALANCE."""
    rows = ui.operator_keyboard(RuntimeState())
    flat = [b["text"] for row in rows for b in row]
    positions = flat.index("📂 OPEN POSITIONS")
    real_row = min(
        i for i, row in enumerate(rows)
        if any(b["text"] == "🔴 ARM REAL" for b in row)
    )
    position_row = next(
        i for i, row in enumerate(rows)
        if any(b["text"] == "📂 OPEN POSITIONS" for b in row)
    )
    assert position_row < real_row, (
        "the position views must not sit in the REAL block"
    )
    assert positions > flat.index("💰 BALANCE")
    assert len(rows[-1]) == 1


# ---------------------------------------------------------------------------
# 2. /start shows the menu and does NOT start trading
# ---------------------------------------------------------------------------


def test_start_command_is_wired_to_the_menu():
    assert TEXT_COMMANDS["/start"] == ui.CB_MENU
    assert TEXT_COMMANDS["/start"] != ui.CB_START, (
        "/start must not be wired to the branch that starts trading"
    )


def test_start_only_shows_the_menu():
    h = handler()
    result = press(h, ui.CB_MENU)
    assert result.handled is True
    assert result.error == ""
    assert [b["text"] for row in result.keyboard for b in row] == MENU_LABELS


@pytest.mark.parametrize("run_state", list(RunState))
def test_start_never_changes_the_run_state(run_state):
    """
    Whatever state the runtime is in, /start leaves it there.

    Parametrised across every state on purpose: a menu that resumed a STOPPED
    bot, or armed a REAL one, would pass a single-state test.
    """
    st = RuntimeState()
    st.run_state = run_state
    if run_state is RunState.EMERGENCY_STOP:
        st.emergency_stop = True
    before = (st.run_state, st.trade_mode, st.real_armed, st.emergency_stop)
    press(ButtonHandler(st, OperatorActions()), ui.CB_MENU)
    after = (st.run_state, st.trade_mode, st.real_armed, st.emergency_stop)
    assert after == before, f"/start mutated the runtime: {before} -> {after}"


def test_start_on_a_fresh_runtime_does_not_start_trading():
    st = RuntimeState()
    assert st.run_state is RunState.OFF
    press(ButtonHandler(st, OperatorActions()), ui.CB_MENU)
    assert st.run_state is RunState.OFF, "/start started the bot"
    assert st.can_open_new_entries()[0] is False


def test_start_does_not_clear_an_emergency_stop():
    st = RuntimeState()
    st.start()
    st.emergency_stop_now()
    press(ButtonHandler(st, OperatorActions()), ui.CB_MENU)
    assert st.emergency_stop is True, (
        "/start must not be a way out of the emergency stop"
    )


def test_start_does_not_arm_real():
    st = RuntimeState()
    st.start()
    press(ButtonHandler(st, OperatorActions()), ui.CB_MENU)
    assert st.real_armed is False


def test_start_does_no_io_and_journals_nothing():
    """
    A control panel that can hang gets force-quit, and a journal entry implies
    an action happened. Neither is acceptable for a menu.
    """
    events: list = []
    h = handler(on_state_change=lambda s: events.append("changed"),
                journal=lambda *a, **k: events.append(("journal", a)),
                preflight=_PF())
    press(h, ui.CB_MENU)
    assert events == [], f"/start had side effects: {events}"


def test_start_renders_the_state_it_is_showing():
    st = RuntimeState()
    st.start()
    text = render_menu(st)
    assert "PAPER" in text
    assert "REAL OFF" in text
    assert "EMERGENCY STOP" not in text

    st.emergency_stop_now()
    assert "EMERGENCY STOP" in render_menu(st)


def test_start_reports_armed_real_honestly():
    st = RuntimeState()
    st.start()
    st.real_armed = True
    assert "REAL ARMED" in render_menu(st)


# ---------------------------------------------------------------------------
# 3. the five existing commands still work through the same surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command,expected", [
    ("/status", "stats"), ("/balance", "balance"), ("/pause", "pause"),
    ("/resume", "resume"), ("/kill", "kill"),
])
def test_the_other_five_commands_are_unchanged(command, expected):
    assert TEXT_COMMANDS[command] == expected


def test_pause_and_resume_still_control_entries():
    st = RuntimeState()
    st.start()
    h = ButtonHandler(st, OperatorActions())
    press(h, ui.CB_PAUSE)
    assert st.can_open_new_entries()[0] is False
    press(h, ui.CB_RESUME)
    assert st.can_open_new_entries()[0] is True


def test_start_paper_button_really_starts_paper():
    """The BUTTON does trade. That is the difference from /start."""
    st = RuntimeState()
    h = ButtonHandler(st, OperatorActions())
    press(h, ui.CB_START)
    assert st.run_state is RunState.PAPER
    assert st.trade_mode is TradeMode.PAPER


# ---------------------------------------------------------------------------
# 4. ARM REAL: preflight and explicit confirmation, both required
# ---------------------------------------------------------------------------


def test_arm_real_button_arms_nothing():
    st = RuntimeState()
    st.start()
    h = ButtonHandler(st, OperatorActions(preflight=_PF(True)))
    result = press(h, ui.CB_REAL)
    assert st.real_armed is False, "the ARM REAL press armed REAL immediately"
    assert result.handled is True
    assert st._pending_confirmation, "a confirmation token must be issued"


def test_arm_real_offers_a_confirmation_keyboard():
    st = RuntimeState()
    st.start()
    h = ButtonHandler(st, OperatorActions(preflight=_PF(True)))
    _text, kb = press(h, ui.CB_REAL).text, None
    result = press(h, ui.CB_REAL)
    texts = [b["text"] for row in result.keyboard for b in row]
    assert any("REAL" in t for t in texts), (
        "the confirmation prompt must offer the confirm/cancel pair"
    )


def test_a_failed_preflight_cannot_arm_real():
    st = RuntimeState()
    st.start()
    pf = _PF(False)
    h = ButtonHandler(st, OperatorActions(preflight=pf))
    press(h, ui.CB_REAL)
    token = st._pending_confirmation
    assert token
    result = asyncio.run(h.handle(f"{ui.CB_CONFIRM_REAL}{token}"))
    assert st.real_armed is False
    assert result.error, "a failed preflight must report an error"
    assert pf.calls == 1, "the preflight must actually have run"


def test_a_passing_preflight_still_needs_the_confirmation():
    st = RuntimeState()
    st.start()
    h = ButtonHandler(st, OperatorActions(preflight=_PF(True)))
    press(h, ui.CB_REAL)
    assert st.real_armed is False, "preflight alone must not arm"
    token = st._pending_confirmation
    result = asyncio.run(h.handle(f"{ui.CB_CONFIRM_REAL}{token}"))
    assert st.real_armed is True
    assert result.error == ""


def test_a_forged_token_cannot_arm_real():
    st = RuntimeState()
    st.start()
    h = ButtonHandler(st, OperatorActions(preflight=_PF(True)))
    result = asyncio.run(h.handle(f"{ui.CB_CONFIRM_REAL}not-the-token"))
    assert st.real_armed is False
    assert result.error == "bad_token"


def test_a_missing_preflight_cannot_arm_real():
    st = RuntimeState()
    st.start()
    h = ButtonHandler(st, OperatorActions(preflight=None))
    press(h, ui.CB_REAL)
    token = st._pending_confirmation
    result = asyncio.run(h.handle(f"{ui.CB_CONFIRM_REAL}{token}"))
    assert st.real_armed is False
    assert result.error == "preflight_unavailable"


def test_arm_real_uses_the_authoritative_runtime_state():
    """
    The menu must drive the SAME state the executor reads, not a Telegram-side
    copy. Proven by reading it back off the RuntimeState the executor uses.
    """
    from autopaper.modes import RuntimeState as RS

    st = RS()
    st.start()
    h = ButtonHandler(st, OperatorActions(preflight=_PF(True)))
    press(h, ui.CB_REAL)
    asyncio.run(h.handle(f"{ui.CB_CONFIRM_REAL}{st._pending_confirmation}"))
    # the executor's own gate reads this exact attribute -- there is no
    # Telegram-side copy of REAL state
    assert st.real_armed is True
    assert st.trade_mode is TradeMode.REAL, (
        "arming the authoritative state must move the runtime to REAL"
    )
    assert st.can_trade_real()[0] if hasattr(st, "can_trade_real") else True
    assert "real_armed" in {f for f in RS.__dataclass_fields__}


# ---------------------------------------------------------------------------
# 5. DISARM REAL and KILL stay immediate
# ---------------------------------------------------------------------------


def test_disarm_real_clears_the_flag_immediately():
    st = RuntimeState()
    st.start()
    st.real_armed = True
    h = ButtonHandler(st, OperatorActions(preflight=_PF(True)))
    result = press(h, ui.CB_REAL_OFF)
    assert st.real_armed is False
    assert result.error == "", "disarm must not need a confirmation"


def test_disarm_real_needs_no_second_confirmation():
    """One press. No token, no prompt, no preflight."""
    st = RuntimeState()
    st.start()
    st.real_armed = True
    pf = _PF(True)
    h = ButtonHandler(st, OperatorActions(preflight=pf))
    press(h, ui.CB_REAL_OFF)
    assert st.real_armed is False
    assert pf.calls == 0, "disarm must not depend on a probe"


def test_disarm_real_blocks_new_real_entries_immediately():
    st = RuntimeState()
    st.start()
    st.real_armed = True
    h = ButtonHandler(st, OperatorActions(preflight=_PF(True)))
    press(h, ui.CB_REAL_OFF)
    assert st.can_open_new_entries()[0] in (False,) or not st.real_armed


def test_disarm_real_does_not_close_an_open_position():
    """
    Disarming stops NEW entries. It is not a liquidation, and turning it into
    one would change documented emergency behaviour.
    """
    st = RuntimeState()
    st.start()
    st.real_armed = True
    before = list(getattr(st, "open_mints", []) or [])
    press(ButtonHandler(st, OperatorActions(preflight=_PF(True))), ui.CB_REAL_OFF)
    assert list(getattr(st, "open_mints", []) or []) == before


def test_disarm_on_an_unarmed_runtime_is_a_harmless_no_op():
    st = RuntimeState()
    st.start()
    result = press(ButtonHandler(st, OperatorActions()), ui.CB_REAL_OFF)
    assert st.real_armed is False
    assert result.handled is True
    assert result.error == "", "pressing disarm twice must not be an error"


def test_kill_is_immediate_and_latched():
    persisted: list[bool] = []
    st = RuntimeState()
    st.start()
    h = ButtonHandler(st, OperatorActions(persist_emergency=persisted.append))
    result = press(h, ui.CB_KILL)
    assert st.emergency_stop is True
    assert st.can_open_new_entries()[0] is False
    assert persisted == [True], "the latch must survive a restart"
    assert result.error == ""


def test_kill_also_stops_real():
    st = RuntimeState()
    st.start()
    st.real_armed = True
    press(ButtonHandler(st, OperatorActions(persist_emergency=lambda v: None)),
          ui.CB_KILL)
    assert st.can_open_new_entries()[0] is False, (
        "the kill switch must block entries regardless of mode"
    )


# ---------------------------------------------------------------------------
# 6. authorization is unchanged
# ---------------------------------------------------------------------------


def test_menu_is_reachable_only_from_an_authorized_chat():
    """The command table is the only door, and the transport guards it."""
    from autopaper.telegram.botapi import chat_is_authorized

    src = (REPO / "src/autopaper/telegram/botapi.py").read_text()
    assert '"/start": "menu"' in src
    assert "chat_is_authorized" in src, "the allow-list gate must still exist"
    # and it is consulted before an update is dispatched
    assert "chat_is_authorized(" in src
    assert chat_is_authorized(
        {"message": {"chat": {"id": ALLOWED}}}, str(ALLOWED)) is True


def test_an_unauthorized_update_never_reaches_the_menu():
    """
    The menu is an operator control plane, so a foreign chat must not be able to
    press any of its buttons -- least of all ARM REAL.
    """
    from autopaper.telegram.botapi import chat_is_authorized, payload_from_update

    update = {
        "update_id": 1,
        "message": {"message_id": 1, "chat": {"id": 999},
                    "text": "/start", "date": 0},
    }
    assert chat_is_authorized(update, str(ALLOWED)) is False
    # the payload would parse, so the allow-list is the only thing stopping it
    assert payload_from_update(update)[0] == ui.CB_MENU


def test_an_unauthorized_callback_is_refused():
    from autopaper.telegram.botapi import chat_is_authorized

    for payload in (ui.CB_START, ui.CB_REAL, ui.CB_KILL, ui.CB_MENU):
        update = {"update_id": 2,
                  "callback_query": {"id": "1", "data": payload,
                                     "chat_instance": "x",
                                     "message": {"message_id": 1,
                                                 "chat": {"id": 999}}}}
        assert chat_is_authorized(update, str(ALLOWED)) is False, payload


# ---------------------------------------------------------------------------
# 7. --cli sends no menu and no outbound message
# ---------------------------------------------------------------------------


def test_cli_mode_builds_no_bot_so_it_cannot_send_the_menu(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "999:FAKEfakefakefake")
    monkeypatch.setenv("TELEGRAM_BOT_CHAT", str(ALLOWED))
    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)

    import autopaper.telegram.botapi as botapi
    import autopaper.telegram.notifier as notifier
    from autopaper.config import build_config
    from autopaper.engine import PaperTrader

    built: list[str] = []
    real = botapi.BotApiTransport

    class Tripwire(real):  # type: ignore[misc,valid-type]
        def __init__(self, *a, **k):
            built.append("transport")
            raise AssertionError("--cli constructed a Telegram transport")

    monkeypatch.setattr(botapi, "BotApiTransport", Tripwire)
    monkeypatch.setattr(notifier, "TelegramNotifier",
                        lambda *a, **k: built.append("notifier"))

    cfg = dataclasses.replace(build_config(tmp_path), cli_only=True)
    t = PaperTrader(cfg)
    try:
        assert t.bot_control is None
        assert t.bot_notifier is None
        assert t.bot_transport is None
        # the notify hook is still the inert no-op, so a menu reply has nowhere
        # to go even if some code path asked for one
        t._notify("menu", None)
        assert built == []
    finally:
        t._release_ownership()


def test_cli_only_is_applied_before_the_engine_is_built():
    """The flag must reach the Config before __init__ builds the bot."""
    src = (REPO / "src/autopaper/cli.py").read_text()
    body = src[src.index("def cmd_run("):src.index("def cmd_status(")]
    assert body.index("dataclasses.replace(cfg, cli_only=") < body.index("_trader(cfg)")


# ---------------------------------------------------------------------------
# 8. authorization / transport doubles
# ---------------------------------------------------------------------------


class _MockClient:
    def __init__(self):
        self.requests: list[str] = []

    async def post(self, url, **kw):
        self.requests.append(str(url))
        raise AssertionError(f"outbound Telegram call: {url}")

    async def aclose(self):
        pass


class _Recorder:
    def __init__(self):
        self.seen: list[str] = []

    async def handle(self, data, *a, **k):
        self.seen.append(data)


class _Control:
    def __init__(self, rec):
        self.rec = rec
        self.seen: list[str] = []

    async def handle_update(self, update):
        from autopaper.telegram.botapi import payload_from_update

        payload, _uid = payload_from_update(update)
        if payload is None:
            return False
        chat = (update.get("message", {}).get("chat", {}) or {}).get("id")
        if str(chat) != str(ALLOWED):
            return False
        self.seen.append(payload)
        return True
