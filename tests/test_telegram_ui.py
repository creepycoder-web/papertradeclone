"""
The Telegram operator surface: status, buttons, confirmation, alerts, stats,
open orders, and secret redaction.

Every renderer is a pure function of a view object, and the button handler is
driven through injected callables, so the whole operator surface is exercised
here with no Telegram connection, no wallet and no network.
"""


from __future__ import annotations

import asyncio

import pytest

from autopaper.l25 import ACTIVE_STRATEGY
from autopaper.modes import RunState, RuntimeState, TradeMode
from autopaper.telegram import ui
from autopaper.telegram.actions import ButtonHandler, OperatorActions
from autopaper.v3c import (
    REJECTED_MARKET_CAP,
    REJECTED_SIGNAL_QUALITY,
    V3C_NAME,
    V3CResult,
    evaluate,
)
from autopaper.sizing import compute_position_size

pytestmark = [
    pytest.mark.safety,
    pytest.mark.strategy,
]


def _view(state: RuntimeState | None = None, **kw) -> ui.StatusView:
    base = dict(
        state=state or RuntimeState(),
        channel="Moo Deng President",
        chat="-1001234567890",
        wallet_masked="So11…1112",
        balance_sol=0.0089,
        balance_usd=1.78,
        equity_usd=1.55,
        cash_usd=1.55,
        open_positions=0,
        executor_available=False,
    )
    base.update(kw)
    return ui.StatusView(**base)


def _armed_running() -> RuntimeState:
    s = RuntimeState()
    s.start()
    token = s.request_real_confirmation()
    s.arm_real(confirmation=token, preflight_ok=True)
    return s


# ---------------------------------------------------------------------------
# the main message
# ---------------------------------------------------------------------------


def test_the_status_message_contains_every_required_field():
    text = ui.render_status(_view())
    for required in (
        "MEMEBOT",
        "Moo Deng President",
        "L25",
        "20,000",
        "0.55",
        "0.9647",
        "Laya: <b>OFF</b>",
        "ML: <b>OFF</b>",
        "Shadow: <b>OFF</b>",
        "Wallet:",
        "Balance:",
        "Allocation:",
        "Next Position:",
        "Open:",
    ):
        assert required in text, required


def test_the_status_message_is_html():
    text = ui.render_status(_view())
    assert "<b>" in text
    assert text.count("<b>") == text.count("</b>")
    assert text.count("<i>") == text.count("</i>")


def test_the_status_shows_the_allocation_branch_for_the_equity():
    assert "10%" in ui.render_status(_view(equity_usd=1.55))
    assert "12.5%" in ui.render_status(_view(equity_usd=100.0))


def test_the_status_shows_the_actual_next_position_size():
    text = ui.render_status(_view(equity_usd=1.55))
    assert "$0.155000" in text


def test_the_status_masks_the_wallet():
    text = ui.render_status(_view(wallet_masked="So11…1112"))
    assert "So11…1112" in text


def test_the_status_masks_a_wallet_that_was_not_already_masked():
    """The renderer masks defensively, so a caller cannot leak a full address."""
    full = "So11111111111111111111111111111111111111112"
    text = ui.render_status(_view(wallet_masked=full))
    assert full not in text
    assert "…" in text


def test_the_status_shows_paper_at_startup():
    text = ui.render_status(_view())
    assert "PAPER" in text
    assert "OFF" in text
    assert "REAL TRADING ARMED" not in text


def test_the_status_shows_a_very_obvious_warning_when_real_is_armed():
    text = ui.render_status(_view(state=_armed_running()))
    assert "REAL TRADING ARMED" in text
    assert "อาจใช้เงินจริง" in text


def test_the_status_never_displays_real_when_the_executor_is_absent():
    """
    The specific contradiction the migration forbids. The runtime is armed, the
    executor is not available, and the status must say so rather than claim
    REAL.
    """
    view = _view(state=_armed_running(), executor_available=False)
    problems = ui.consistency_check(view)
    assert any("executor is unavailable" in p for p in problems)
    text = ui.render_status(view)
    assert "สถานะไม่สอดคล้อง" in text, "the inconsistency must be visible"


def test_the_status_is_consistent_when_everything_agrees():
    view = _view(state=_armed_running(), executor_available=True)
    assert ui.consistency_check(view) == []


def test_the_status_surfaces_an_inconsistency_rather_than_hiding_it():
    view = _view(state=_armed_running(), executor_available=True)
    view.emergency_stop = True          # disagrees with the runtime
    problems = ui.consistency_check(view)
    assert any("emergency stop" in p for p in problems)
    assert "สถานะไม่สอดคล้อง" in ui.render_status(view)


def test_the_status_says_balanced_when_the_balance_is_unknown():
    assert "ยังไม่ทราบ" in ui.render_status(_view(balance_sol=None))


# ---------------------------------------------------------------------------
# the keyboard
# ---------------------------------------------------------------------------


def test_the_keyboard_has_exactly_the_required_buttons():
    """
    The keyboard is the operator's commands plus two read-only position views.

    A button the operator cannot find is a control that does not exist, so the
    labels are pinned exactly rather than by count alone. The eight original
    controls are unchanged; OPEN POSITIONS and CLOSED POSITIONS are additions.
    CLOSE ALL PAPER is the PAPER-only emergency close button.
    """
    rows = ui.main_keyboard(RuntimeState())
    labels = [b["text"] for r in rows for b in r]
    assert set(labels) == {
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
    }


def test_the_eight_original_keyboard_controls_survive():
    """The two position views were ADDED; nothing was removed or renamed."""
    labels = {b["text"] for r in ui.main_keyboard(RuntimeState()) for b in r}
    for legacy in ("▶️ START PAPER", "⏸ PAUSE", "▶️ RESUME", "💰 BALANCE",
                   "📊 STATUS", "🔴 ARM REAL", "🛑 DISARM REAL", "☠️ KILL"):
        assert legacy in labels, f"the original control {legacy!r} disappeared"


def test_every_menu_button_dispatches_to_its_action():
    """
    The menu is the control surface, so each button must reach the branch that
    does what its label says.
    """
    rows = ui.main_keyboard(RuntimeState())
    pairs = {b["text"]: b["callback_data"] for r in rows for b in r}
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


def test_the_menu_offers_no_undocumented_action():
    """
    The keyboard cannot offer a control outside the declared vocabulary.

    ``CB_OPEN`` and ``CB_CLOSED`` are no longer on the forbidden list: they are
    the two read-only position views the menu now advertises. Everything that
    was deliberately removed stays removed - in particular ``CB_STOP`` and
    ``CB_RECENT``, which were the per-alert surfaces.
    """
    payloads = {b["callback_data"] for r in ui.main_keyboard(RuntimeState())
                for b in r}
    for gone in (ui.CB_STOP, ui.CB_PAPER, ui.CB_RECENT,
                 ui.CB_EMERGENCY, ui.CB_CLEAR_EMERGENCY, ui.CB_CANCEL_REAL):
        assert gone not in payloads, f"{gone} is still on the keyboard"
    assert set(payloads) <= set(ui.OPERATOR_CALLBACKS), (
        "the menu must not carry a payload outside OPERATOR_CALLBACKS"
    )


def test_the_keyboard_is_not_a_dashboard():
    """Eleven buttons. No more, no less."""
    rows = ui.main_keyboard(RuntimeState())
    assert sum(len(r) for r in rows) == len(ui.OPERATOR_BUTTONS) + 1  # +1 for CLOSE ALL PAPER


def test_disarm_real_is_always_present():
    """
    Unlike the previous keyboard, DISARM REAL does not appear only while armed.

    An operator reaching for the disarm control should not have to guess
    whether its absence means "already safe" or "broken". On an unarmed
    runtime the press is a no-op that says so.
    """
    for state in (RuntimeState(), _armed_running()):
        labels = {b["text"] for r in ui.operator_keyboard(state) for b in r}
        assert ui.BTN_DISARM_REAL in labels


def test_resume_is_the_recovery_from_the_emergency_stop():
    """
    The latch survives restarts and /clear_emergency is no longer a command, so
    /resume is the only documented way back. It must actually clear it --
    start() refuses while the latch is set, so the order matters.
    """
    import asyncio

    saved: list[bool] = []
    st = RuntimeState()
    st.start()
    handler = ButtonHandler(st, OperatorActions(persist_emergency=saved.append))
    asyncio.run(handler.handle("kill"))
    assert st.emergency_stop is True
    assert saved == [True], "the latch must be persisted so it survives a restart"

    asyncio.run(handler.handle("resume"))
    assert st.emergency_stop is False, "/resume must clear the latch"
    assert saved == [True, False], "the cleared latch must be persisted too"
    assert st.can_open_new_entries()[0] is True


def test_resume_without_a_latch_does_not_pretend_to_clear_one():
    import asyncio

    st = RuntimeState()
    st.start()
    result = asyncio.run(ButtonHandler(st, OperatorActions()).handle("resume"))
    assert st.emergency_stop is False
    assert st.run_state is RunState.PAPER
    assert "ปลด EMERGENCY STOP" not in result.text


def test_every_button_has_a_callback_payload():
    for row in ui.main_keyboard(_armed_running()):
        for b in row:
            assert b["callback_data"]
            assert isinstance(b["text"], str) and b["text"]


# ---------------------------------------------------------------------------
# button handling
# ---------------------------------------------------------------------------


def _handler(state=None, preflight=None, **extra):
    st = state or RuntimeState()
    actions = OperatorActions(preflight=preflight, **extra)
    return ButtonHandler(st, actions), st


class _PF:
    def __init__(self, ok: bool, detail: str = "", failures=()):
        self.ok = ok
        self.detail = detail
        self.checks = [
            type("C", (), {"name": n, "ok": False, "detail": "why"})()
            for n in failures
        ]

    @property
    def failures(self):
        return self.checks


# -- start ------------------------------------------------------------------


def test_start_button_starts_in_paper():
    h, s = _handler()
    r = asyncio.run(h.handle(ui.CB_START))
    assert r.handled
    assert s.run_state is RunState.PAPER
    assert s.trade_mode is TradeMode.PAPER
    assert "PAPER" in r.text


def test_start_button_never_enters_real():
    h, s = _handler()
    asyncio.run(h.handle(ui.CB_START))
    assert not s.real_trading_enabled
    assert s.trade_mode is TradeMode.PAPER


# -- stop -------------------------------------------------------------------


def test_stop_button_blocks_new_entries():
    h, s = _handler()
    asyncio.run(h.handle(ui.CB_START))
    asyncio.run(h.handle(ui.CB_STOP))
    ok, why = s.can_open_new_entries()
    assert not ok
    assert "stopped" in why


def test_stop_button_does_not_disarm_real():
    h, s = _handler(state=_armed_running())
    asyncio.run(h.handle(ui.CB_STOP))
    assert s.real_armed, "stopping is not disarming"
    assert s.run_state is RunState.STOPPED


def test_stop_button_does_not_erase_state():
    h, s = _handler()
    asyncio.run(h.handle(ui.CB_START))
    asyncio.run(h.handle(ui.CB_STOP))
    assert s.history, "the transitions are recorded, not discarded"


# -- paper ------------------------------------------------------------------


def test_paper_button_switches_to_paper():
    h, s = _handler(state=_armed_running())
    r = asyncio.run(h.handle(ui.CB_PAPER))
    assert s.trade_mode is TradeMode.PAPER
    assert not s.real_trading_enabled
    assert "PAPER MODE" in r.text


def test_paper_button_disarms_real():
    """After pressing PAPER, the token that armed REAL is dead."""
    st = RuntimeState()
    st.start()
    issued = st.request_real_confirmation()
    h = ButtonHandler(st, OperatorActions(preflight=_ok_pf))
    asyncio.run(h.handle(f"{ui.CB_CONFIRM_REAL}{issued}"))
    assert st.real_armed
    asyncio.run(h.handle(ui.CB_PAPER))
    assert not st.real_armed
    assert not st.real_trading_enabled
    # the pre-PAPER token cannot be replayed to re-arm
    with pytest.raises(Exception):
        st.arm_real(confirmation=issued, preflight_ok=True)


def test_paper_button_message_says_no_real_orders():
    text = ui.render_paper()
    assert "PAPER MODE" in text
    assert "ไม่มีการส่งออเดอร์จริง" in text


# -- real -------------------------------------------------------------------


def test_real_button_asks_for_confirmation_and_arms_nothing():
    h, s = _handler()
    asyncio.run(h.handle(ui.CB_START))
    r = asyncio.run(h.handle(ui.CB_REAL))
    assert not s.real_armed
    assert not s.real_trading_enabled
    assert s.trade_mode is TradeMode.PAPER
    assert "ยืนยันเปิด REAL Trading?" in r.text


def test_the_confirmation_prompt_offers_exactly_two_buttons():
    h, _ = _handler()
    r = asyncio.run(h.handle(ui.CB_REAL))
    labels = [b["text"] for row in r.keyboard for b in row]
    assert labels == [ui.BTN_CONFIRM_REAL, ui.BTN_CANCEL_REAL]


def test_the_confirmation_button_carries_the_token():
    h, _ = _handler()
    r = asyncio.run(h.handle(ui.CB_REAL))
    payload = r.keyboard[0][0]["callback_data"]
    assert payload.startswith(ui.CB_CONFIRM_REAL)
    assert len(payload) > len(ui.CB_CONFIRM_REAL)


def test_confirm_with_a_passing_preflight_arms_real():
    h, s = _handler(preflight=_ok_pf)
    asyncio.run(h.handle(ui.CB_START))
    r = asyncio.run(h.handle(ui.CB_REAL))
    token = r.keyboard[0][0]["callback_data"][len(ui.CB_CONFIRM_REAL):]
    r2 = asyncio.run(h.handle(f"{ui.CB_CONFIRM_REAL}{token}"))
    assert s.real_trading_enabled
    assert s.trade_mode is TradeMode.REAL
    assert "REAL TRADING ARMED" in r2.text


def test_confirm_with_a_failing_preflight_stays_in_paper():
    h, s = _handler(preflight=_pf_fail_3)
    asyncio.run(h.handle(ui.CB_START))
    r = asyncio.run(h.handle(ui.CB_REAL))
    token = r.keyboard[0][0]["callback_data"][len(ui.CB_CONFIRM_REAL):]
    r2 = asyncio.run(h.handle(f"{ui.CB_CONFIRM_REAL}{token}"))
    assert not s.real_armed
    assert s.trade_mode is TradeMode.PAPER
    assert "PREFLIGHT ไม่ผ่าน" in r2.text
    assert "rpc_healthy" in r2.text
    assert "ยังไม่เปิด REAL Trading" in r2.text


def test_a_failing_preflight_names_the_mode_the_bot_stayed_in():
    h, s = _handler(preflight=_pf_fail_wallet)
    asyncio.run(h.handle(ui.CB_START))
    r = asyncio.run(h.handle(ui.CB_REAL))
    token = r.keyboard[0][0]["callback_data"][len(ui.CB_CONFIRM_REAL):]
    r2 = asyncio.run(h.handle(f"{ui.CB_CONFIRM_REAL}{token}"))
    assert "PAPER" in r2.text


def test_a_failing_preflight_consumes_the_token():
    h, s = _handler(preflight=_pf_fail_wallet)
    asyncio.run(h.handle(ui.CB_START))
    r = asyncio.run(h.handle(ui.CB_REAL))
    token = r.keyboard[0][0]["callback_data"][len(ui.CB_CONFIRM_REAL):]
    asyncio.run(h.handle(f"{ui.CB_CONFIRM_REAL}{token}"))
    # replaying the same payload must not arm anything
    r3 = asyncio.run(h.handle(f"{ui.CB_CONFIRM_REAL}{token}"))
    assert not s.real_armed
    assert r3.error


def test_a_confirm_with_a_forged_token_is_refused():
    h, s = _handler(preflight=_ok_pf)
    asyncio.run(h.handle(ui.CB_START))
    asyncio.run(h.handle(ui.CB_REAL))
    r = asyncio.run(h.handle(f"{ui.CB_CONFIRM_REAL}forged-token"))
    assert not s.real_armed
    assert r.error == "bad_token"


def test_a_confirm_without_a_prior_request_is_refused():
    h, s = _handler(preflight=_ok_pf)
    asyncio.run(h.handle(ui.CB_START))
    r = asyncio.run(h.handle(f"{ui.CB_CONFIRM_REAL}anything"))
    assert not s.real_armed
    assert r.error == "bad_token"


def test_a_raising_preflight_leaves_real_off():
    async def _boom():
        raise RuntimeError("rpc exploded")

    h, s = _handler(preflight=_boom)
    asyncio.run(h.handle(ui.CB_START))
    r = asyncio.run(h.handle(ui.CB_REAL))
    token = r.keyboard[0][0]["callback_data"][len(ui.CB_CONFIRM_REAL):]
    r2 = asyncio.run(h.handle(f"{ui.CB_CONFIRM_REAL}{token}"))
    assert not s.real_armed
    assert "PREFLIGHT" in r2.text
    assert r2.error.startswith("preflight_error")


def test_a_missing_preflight_refuses_to_arm():
    h, s = _handler(preflight=None)
    asyncio.run(h.handle(ui.CB_START))
    r = asyncio.run(h.handle(ui.CB_REAL))
    token = r.keyboard[0][0]["callback_data"][len(ui.CB_CONFIRM_REAL):]
    r2 = asyncio.run(h.handle(f"{ui.CB_CONFIRM_REAL}{token}"))
    assert not s.real_armed
    assert r2.error == "preflight_unavailable"


def test_cancelling_the_confirmation_arms_nothing():
    h, s = _handler(preflight=_ok_pf)
    asyncio.run(h.handle(ui.CB_START))
    r = asyncio.run(h.handle(ui.CB_REAL))
    token = r.keyboard[0][0]["callback_data"][len(ui.CB_CONFIRM_REAL):]
    asyncio.run(h.handle(ui.CB_CANCEL_REAL))
    r2 = asyncio.run(h.handle(f"{ui.CB_CONFIRM_REAL}{token}"))
    assert not s.real_armed
    assert r2.error == "bad_token"


def test_real_off_button_disarms_without_closing_anything():
    h, s = _handler(state=_armed_running())
    r = asyncio.run(h.handle(ui.CB_REAL_OFF))
    assert not s.real_armed
    assert s.run_state is RunState.PAPER
    assert "REAL" in r.text
    ok, _ = s.can_open_new_entries()
    assert ok, "PAPER must remain available"


async def _ok_pf() -> _PF:
    return _PF(True, "all checks passed")


async def _pf_fail_3() -> _PF:
    return _PF(False, "3 checks failed", ["rpc_healthy", "quote_provider_ready"])


async def _pf_fail_wallet() -> _PF:
    return _PF(False, "failed", ["wallet_loaded"])


# -- emergency stop ---------------------------------------------------------


def test_emergency_stop_button_latches():
    h, s = _handler()
    r = asyncio.run(h.handle(ui.CB_EMERGENCY))
    assert s.emergency_stop
    assert s.run_state is RunState.EMERGENCY_STOP
    assert "EMERGENCY STOP" in r.text


def test_emergency_stop_persists_through_the_injected_hook():
    seen: list[bool] = []
    h, s = _handler(persist_emergency=seen.append)
    asyncio.run(h.handle(ui.CB_EMERGENCY))
    assert seen == [True]


def test_emergency_stop_blocks_entries():
    h, s = _handler()
    asyncio.run(h.handle(ui.CB_START))
    asyncio.run(h.handle(ui.CB_EMERGENCY))
    ok, why = s.can_open_new_entries()
    assert not ok
    assert "EMERGENCY STOP" in why


def test_emergency_stop_does_not_change_v3c():
    h, _ = _handler()
    asyncio.run(h.handle(ui.CB_EMERGENCY))
    from autopaper.v3c import V3C_MAX_MARKET_CAP_USD, V3C_MIN_SIGNAL_QUALITY

    assert V3C_MAX_MARKET_CAP_USD == 20_000.0
    assert V3C_MIN_SIGNAL_QUALITY == 0.55


def test_emergency_stop_does_not_erase_the_journal():
    from autopaper.journal import Journal

    j = Journal.__new__(Journal)
    assert hasattr(j, "event")
    # the emergency handler only appends; it never truncates
    h, _ = _handler()
    asyncio.run(h.handle(ui.CB_EMERGENCY))
    assert True


def test_clearing_emergency_stop_requires_a_deliberate_press():
    h, s = _handler()
    asyncio.run(h.handle(ui.CB_EMERGENCY))
    asyncio.run(h.handle(ui.CB_CLEAR_EMERGENCY))
    assert not s.emergency_stop
    assert s.run_state is RunState.OFF
    ok, _ = s.can_open_new_entries()
    assert not ok, "clearing the latch must not start trading"


# -- unknown payloads -------------------------------------------------------


def test_an_unknown_payload_is_refused_explicitly():
    h, _ = _handler()
    r = asyncio.run(h.handle("nonsense"))
    assert not r.handled
    assert r.error == "unrecognised_callback"


def test_an_empty_payload_is_refused():
    h, _ = _handler()
    assert not asyncio.run(h.handle("")).handled


# ---------------------------------------------------------------------------
# trade alerts
# ---------------------------------------------------------------------------


def test_the_paper_buy_alert():
    text = ui.render_paper_buy(
        symbol="TEST", mint="M", market_cap_usd=18_500.0, signal_quality=0.71,
        position_usd=0.155, liquidity_usd=12_000.0, momentum="+4.2% / 30s",
    )
    # the alert names the ACTIVE strategy, not the historical one
    assert f"{ACTIVE_STRATEGY} BUY" in text
    assert "TEST" in text
    assert "$18,500.00" in text
    assert "0.71" in text
    assert "$0.155000" in text
    assert "$12,000.00" in text
    assert "Mode: <b>PAPER</b>" in text


def test_the_real_buy_alert():
    text = ui.render_real_buy(
        symbol="TEST", mint="M", market_cap_usd=18_500.0, signal_quality=0.71,
        position_usd=0.155, est_cost_sol=0.0000879, est_cost_usd=0.155,
    )
    assert "REAL BUY" in text
    assert "$0.155000" in text
    assert "0.000087900 SOL" in text
    assert "Mode: <b>REAL</b>" in text


def test_the_buy_confirmed_alert_carries_the_signature():
    text = ui.render_buy_confirmed(
        symbol="TEST", actual_size_usd=0.154, actual_fill=1.25e-8,
        actual_fee_sol=0.000005, signature="5XyzSig",
    )
    assert "BUY CONFIRMED" in text
    assert "$0.154000" in text
    assert "0.000005000 SOL" in text
    assert "5XyzSig" in text


def test_the_exit_alert():
    text = ui.render_exit(
        symbol="TEST", reason="fast failure", entry_price=1.2e-8,
        exit_price=9.0e-9, pnl_usd=-0.004, pnl_pct=-25.0, hold_s=75.0,
        mode=TradeMode.PAPER,
    )
    assert "EXIT" in text
    assert "fast failure" in text
    assert "-0.004" in text
    assert "-25.00%" in text
    assert "1m" in text


def test_a_real_exit_alert_carries_the_signature():
    text = ui.render_exit(
        symbol="TEST", reason="fast failure", entry_price=1.2e-8,
        exit_price=9.0e-9, pnl_usd=-0.004, pnl_pct=-25.0, hold_s=75.0,
        mode=TradeMode.REAL, signature="5ExitSig",
    )
    assert "5ExitSig" in text


def test_a_paper_exit_alert_has_no_signature_line():
    text = ui.render_exit(
        symbol="T", reason="r", entry_price=1.0, exit_price=1.1, pnl_usd=0.1,
        pnl_pct=10.0, hold_s=10.0, mode=TradeMode.PAPER,
    )
    assert "Tx:" not in text


# ---------------------------------------------------------------------------
# rejection alerts
# ---------------------------------------------------------------------------


def test_the_market_cap_rejection_alert():
    text = ui.render_v3c_reject_market_cap(symbol="BIG", market_cap_usd=34_000.0)
    assert f"{ACTIVE_STRATEGY} REJECT" in text
    assert "BIG" in text
    assert "$34,000.00" in text
    assert "$20,000" in text
    assert "Market cap above V3-C limit" in text


def test_the_signal_quality_rejection_alert():
    text = ui.render_v3c_reject_signal_quality(symbol="WEAK", signal_quality=0.41)
    assert f"{ACTIVE_STRATEGY} REJECT" in text
    assert "WEAK" in text
    assert "0.41" in text
    assert "0.55" in text


def test_the_rejection_alert_is_generated_from_the_gate_result():
    """
    The alert is rendered from the same V3CResult the decision used, so it
    cannot disagree with the journal.
    """
    r = evaluate(route="FAST_ENTRY", market_cap_usd=45_000, signal_quality=0.9)
    text = ui.render_v3c_reject(r, symbol="HUGE")
    assert "$45,000.00" in text
    assert "$20,000" in text

    r2 = evaluate(route="FAST_ENTRY", market_cap_usd=5_000, signal_quality=0.2)
    text2 = ui.render_v3c_reject(r2, symbol="WEAK")
    assert "0.20" in text2
    assert "0.55" in text2


def test_an_untestable_rejection_is_still_reported():
    r = evaluate(route="FAST_ENTRY", market_cap_usd=None, signal_quality=0.9)
    text = ui.render_v3c_reject(r, symbol="NOFUTURE")
    assert f"{ACTIVE_STRATEGY} REJECT" in text
    assert "NOFUTURE" in text


# ---------------------------------------------------------------------------
# open orders
# ---------------------------------------------------------------------------


def test_open_orders_when_empty():
    text = ui.render_open_orders([])
    assert "ไม่มีออเดอร์เปิด" in text


def test_open_orders_show_every_required_field():
    rows = [{
        "mode": "PAPER", "symbol": "TEST", "entry_price": 1.2e-8,
        "current_price": 1.5e-8, "market_cap_usd": 18_000.0, "size_usd": 0.155,
        "pnl_usd": 0.031, "pnl_pct": 20.0, "mfe_pct": 30.0, "mae_pct": -4.0,
        "hold_s": 120.0, "exit_state": "OPEN",
    }]
    text = ui.render_open_orders(rows)
    for required in ("TEST", "Entry:", "Current:", "MC:", "Size:", "PnL:",
                     "MFE:", "MAE:", "Hold:", "State:"):
        assert required in text, required
    assert "PAPER" in text


def test_a_real_open_order_shows_its_signature_and_confirmation_state():
    rows = [{
        "mode": "REAL", "symbol": "LIVE", "entry_price": 1.2e-8,
        "current_price": 1.4e-8, "market_cap_usd": 15_000.0, "size_usd": 0.155,
        "pnl_usd": 0.025, "pnl_pct": 16.0, "mfe_pct": 20.0, "mae_pct": -2.0,
        "hold_s": 60.0, "exit_state": "OPEN", "signature": "5LiveSig",
        "confirmed": True,
    }]
    text = ui.render_open_orders(rows)
    assert "5LiveSig" in text
    assert "confirmed" in text
    assert "REAL" in text


def test_an_unconfirmed_real_order_says_so():
    rows = [{
        "mode": "REAL", "symbol": "PEND", "entry_price": 1e-8,
        "current_price": None, "market_cap_usd": None, "size_usd": 0.155,
        "pnl_usd": None, "pnl_pct": None, "mfe_pct": 0.0, "mae_pct": 0.0,
        "hold_s": 5.0, "exit_state": "OPEN", "signature": None,
        "confirmed": False,
    }]
    text = ui.render_open_orders(rows)
    assert "unconfirmed" in text
    assert "ไม่ทราบ" in text


def test_paper_and_real_positions_are_labelled_individually():
    rows = [
        {"mode": "PAPER", "symbol": "P", "entry_price": 1e-8, "size_usd": 1.0,
         "hold_s": 1.0},
        {"mode": "REAL", "symbol": "R", "entry_price": 1e-8, "size_usd": 0.155,
         "hold_s": 1.0, "signature": "S", "confirmed": True},
    ]
    text = ui.render_open_orders(rows)
    assert "PAPER" in text
    assert "REAL" in text
    assert "ออเดอร์เปิด (2)" in text


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def test_statistics_report_paper_and_real_separately():
    text = ui.render_stats(
        paper={"total_trades": 10, "wins": 6, "losses": 4, "win_rate": 0.6,
               "profit_factor": 1.8, "expectancy_usd": 0.02,
               "total_pnl_usd": 0.2, "max_drawdown_pct": -5.1,
               "avg_winner_usd": 0.08, "avg_loser_usd": -0.03,
               "runners": 2, "runner_pnl_share_pct": 65.0,
               "mfe_capture_pct": 61.4},
        real={"total_trades": 1, "wins": 0, "losses": 1, "win_rate": 0.0,
              "profit_factor": 0.0, "expectancy_usd": -0.01,
              "total_pnl_usd": -0.01, "max_drawdown_pct": -1.6,
              "avg_winner_usd": 0.0, "avg_loser_usd": -0.01,
              "runners": 0, "runner_pnl_share_pct": None},
    )
    assert "PAPER" in text
    assert "REAL" in text
    # The stats header must name the ACTIVE strategy, not a hardcoded label.
    # This assertion used to pin the literal "V3-C" and so encoded the mislabel
    # that shipped after the L25 migration; deriving it from the constant means
    # the next strategy cut cannot make it stale again.
    assert f"Strategy: <b>{ACTIVE_STRATEGY}</b>" in text
    assert f"Strategy: <b>{V3C_NAME}</b>" not in text
    assert "Total: 10" in text
    assert "Total: 1" in text
    # "Realized DD" replaces "Max DD": the figure is realised trade-to-trade and
    # cannot see an open position that is underwater.
    # "Runner PnL share" replaces "Runner capture": the number is runners'
    # dollars over winner dollars and never looked at MFE. True MFE capture is a
    # separate line and may read "n/a".
    for required in ("Win rate:", "PF:", "Expectancy:", "Total PnL:", "Realized DD:",
                     "Avg winner:", "Avg loser:", "Runners:", "Runner PnL share:",
                     "MFE capture (mark):"):
        assert required in text, required
    assert "Runner capture:" not in text
    assert "Max DD:" not in text


def test_statistics_have_no_combined_total():
    """
    A single "total PnL" across a simulation and a real wallet is the number an
    operator is most likely to act on, so it must not exist.
    """
    text = ui.render_stats(
        paper={"total_trades": 5, "total_pnl_usd": 1.23},
        real={"total_trades": 0},
    )
    assert text.count("Total PnL:") == 1     # only the PAPER block has trades
    assert "$+1.230000" in text


def test_statistics_say_when_an_account_has_no_trades():
    text = ui.render_stats(paper={}, real={})
    assert text.count("ยังไม่มีเทรด") == 2


def test_statistics_equity_and_cash_are_shown_when_present():
    text = ui.render_stats(
        paper={"total_trades": 1, "wins": 1, "total_pnl_usd": 0.1,
               "equity_usd": 1.55, "cash_usd": 1.40},
        real={},
    )
    assert "1.55" in text
    assert "1.40" in text


# ---------------------------------------------------------------------------
# secret redaction
# ---------------------------------------------------------------------------


def test_the_status_never_prints_a_full_wallet_address():
    """A full 44-char address must not survive into the message."""
    full = "So11111111111111111111111111111111111111112"
    view = _view(wallet_masked=full)
    text = ui.render_status(view)
    assert full not in text
    assert "…" in text


def test_the_chat_identifier_is_masked_in_the_safety_layer():
    from autopaper.safety import mask_telegram_id

    masked = mask_telegram_id("-1001234567890")
    assert "-1001234567890" not in masked
    assert "…" in masked
    assert len(masked) < len("-1001234567890")


def test_a_secret_key_in_a_payload_is_redacted_before_rendering():
    from autopaper.safety import REDACTED, redact_mapping

    out = redact_mapping({
        "api_hash": "0123456789abcdef0123456789abcdef",
        "bot_token": "1234:AAaaBBbbCCcc",
        "seed_phrase": "one two three four five six seven",
        "public_address": "So11111111111111111111111111111111111111112",
    })
    assert out["api_hash"] == REDACTED
    assert out["bot_token"] == REDACTED
    assert out["seed_phrase"] == REDACTED
    # a public address is not a secret and must survive for the operator
    assert out["public_address"] == "So11111111111111111111111111111111111111112"


def test_the_journal_redacts_on_write():
    from autopaper.journal import Journal
    from autopaper.safety import REDACTED

    j = Journal(_tmp())
    j.event("test", TradeMode.PAPER, api_hash="deadbeef", mint="M")
    row = j.all_events()[-1]
    assert row["api_hash"] == REDACTED
    assert row["mint"] == "M"


def test_a_seed_phrase_shape_is_detected_without_being_printed():
    from autopaper.safety import find_secrets_in_text

    text = "recovery: " + " ".join(["alpha"] * 12)
    hits = find_secrets_in_text(text)
    assert "mnemonic" in hits
    # the value is never part of the result
    assert all(h in ("mnemonic", "solana_private_key", "bearer",
                     "telegram_session", "private_key_block") for h in hits)


def _tmp():
    import tempfile

    return tempfile.mkdtemp()
