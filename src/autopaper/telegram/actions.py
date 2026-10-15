"""
Operator button handling.

This is where a button press becomes a state transition, and it is the ONLY
place in the runtime that can arm REAL. Everything else - config, environment,
CLI - can read capabilities but cannot authorize an entry.

The whole handler is written against injected callables and a
:class:`RuntimeState`, so the complete button surface is unit-testable without a
Telegram connection, a wallet or an RPC. That is not a convenience: the
REAL-arming path is the most dangerous code in the project and it needs to be
exercised in tests, not just reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from ..journal import Journal
from ..l25 import ACTIVE_STRATEGY
from ..modes import ModeError, RunState, RuntimeState, TradeMode
from ..paper_live.take_profit import TakeProfitError  # noqa: F401 (used by handlers)
from . import ui

#: Preflight is a coroutine returning an object with ``.ok`` and ``.checks``.
PreflightFn = Callable[[], Awaitable[Any]]


@dataclass
class OperatorActions:
    """
    The callable effects a button press may have.

    Injected rather than imported so the handler stays testable and so the
    engine can wire them to the real preflight and the real journal without
    this module knowing what either does.
    """

    preflight: Optional[PreflightFn] = None
    journal: Optional[Journal] = None
    on_state_change: Optional[Callable[[RuntimeState], None]] = None
    #: Called after a successful REAL arming, to surface the warning.
    on_real_armed: Optional[Callable[[RuntimeState], None]] = None
    on_emergency_stop: Optional[Callable[[], None]] = None
    #: Called to persist the emergency-stop latch so it survives a restart.
    persist_emergency: Optional[Callable[[bool], None]] = None
    #: Called after a successful strategy selection so it survives a restart.
    #: The Telegram handler does not own storage; the engine wires this to the
    #: atomic durable store.
    persist_strategies: Optional[Callable[[list], None]] = None
    #: Called after a successful take-profit selection so it survives a restart.
    #: The Telegram handler does not own storage; the engine wires this to the
    #: atomic durable store.
    persist_take_profit: Optional[Callable[[object], None]] = None
    persist_signal_source: Optional[Callable[[str], None]] = None
    current_signal_source: Optional[Callable[[], str]] = None
    #: Whether the operator has acknowledged the real-trading risks. Required
    #: before the REAL button even offers a confirmation. ``None`` means "not
    #: configured", which is treated as acknowledged so a bare handler in a test
    #: is not silently disabled; production wires this to the real config.
    risk_acknowledged: Optional[Callable[[], bool]] = None
    extras: dict[str, Callable[[], Any]] = field(default_factory=dict)


@dataclass
class ButtonResult:
    """What the handler decided, and what to show the operator."""

    handled: bool
    text: str = ""
    keyboard: Optional[list[list[dict[str, str]]]] = None
    #: The state the runtime is in AFTER the press. The caller reads its own
    #: state; this is returned so a test can assert the transition without
    #: reaching back into the runtime.
    state_after: Optional[RuntimeState] = None
    error: str = ""

    @property
    def armed_real(self) -> bool:
        return bool(self.state_after and self.state_after.real_trading_enabled)


class ButtonHandler:
    """
    Dispatches a callback payload to a state transition and a rendered reply.

    Every branch that could arm REAL goes through the same sequence:

        token valid  AND  preflight ok  AND  not emergency-stopped
                                                       |
                                                       v
                                                   REAL_ARMED

    There is no branch that skips a step, and no parameter that overrides one.
    """

    def __init__(
        self,
        state: RuntimeState,
        actions: Optional[OperatorActions] = None,
    ) -> None:
        self.state = state
        self.actions = actions or OperatorActions()

    # -- helpers -----------------------------------------------------------

    def _changed(self) -> None:
        if self.actions.on_state_change is not None:
            self.actions.on_state_change(self.state)

    def _journal(self, event: str, **payload: Any) -> None:
        if self.actions.journal is not None:
            self.actions.journal.event(
                event, self.state.trade_mode, run_state=self.state.run_state.value,
                **payload,
            )

    def _kb(self) -> list[list[dict[str, str]]]:
        return ui.main_keyboard(self.state)

    # -- dispatch ----------------------------------------------------------

    async def handle(self, data: str) -> ButtonResult:
        """
        Handle one callback payload.

        An unrecognised payload is refused explicitly rather than ignored,
        because a silently-ignored operator press is indistinguishable from a
        broken button.
        """
        data = (data or "").strip()

        if data == ui.CB_MENU:
            return self._menu()
        if data == ui.CB_STRATEGY:
            return self._strategy_menu()
        if data == ui.CB_TAKE_PROFIT:
            return self._take_profit_menu()
        if data.startswith(ui.CB_TAKE_PROFIT_SELECT):
            return self._select_take_profit(data[len(ui.CB_TAKE_PROFIT_SELECT):])
        if data == ui.CB_SIGNAL_SOURCE:
            return self._signal_source_menu()
        if data.startswith(ui.CB_SOURCE_SELECT):
            return self._select_signal_source(data[len(ui.CB_SOURCE_SELECT):])
        if data.startswith(ui.CB_PROFILE):
            return self._select_profile(
                data[len(ui.CB_PROFILE):])
        if data == ui.CB_START:
            return self._start()
        if data == ui.CB_STOP:
            return self._stop()
        if data == ui.CB_PAPER:
            return self._paper()
        if data == ui.CB_REAL:
            return self._real_request()
        if data == ui.CB_REAL_OFF:
            return self._real_off()
        if data == ui.CB_STATS:
            return await self._stats()
        if data == ui.CB_RECENT:
            return await self._recent_orders()
        if data == ui.CB_BALANCE:
            return await self._balance()
        if data == ui.CB_PAUSE:
            return self._pause()
        if data == ui.CB_RESUME:
            return self._resume()
        if data == ui.CB_KILL:
            return self._kill()
        if data == ui.CB_OPEN:
            return await self._open_positions()
        if data == ui.CB_CLOSED:
            return await self._closed_positions()
        if data == ui.CB_EMERGENCY:
            return self._emergency_stop()
        if data == ui.CB_CLEAR_EMERGENCY:
            return self._clear_emergency()
        if data == ui.CB_CANCEL_REAL:
            return self._cancel_real()
        if data.startswith(ui.CB_CONFIRM_REAL):
            token = data[len(ui.CB_CONFIRM_REAL):]
            return await self._confirm_real(token)
        if data == ui.CB_CLOSE_ALL_PAPER:
            return await self._close_all_paper()
        if data == ui.CB_CLOSE_ALL_PAPER_CONFIRM:
            return await self._close_all_paper_confirm()
        if data == ui.CB_CLOSE_ALL_PAPER_CANCEL:
            return self._close_all_paper_cancel()

        return ButtonResult(
            handled=False,
            text="ไม่รู้จักคำสั่งนี้",
            keyboard=self._kb(),
            state_after=self.state,
            error="unrecognised_callback",
        )

    # -- branches ----------------------------------------------------------

    def _strategy_menu(self) -> ButtonResult:
        """
        Open the strategy submenu. Read-only: shows the current profile and
        offers the others. Changes nothing.
        """
        return ButtonResult(
            handled=True,
            text=ui.render_strategy_menu(self.state),
            keyboard=ui.strategy_keyboard(self.state),
            state_after=self.state,
        )

    # -- take profit ------------------------------------------------------

    def _take_profit_menu(self) -> ButtonResult:
        """
        Open the take-profit submenu. Read-only: shows the selected level and
        offers the presets. Changes nothing.
        """
        return ButtonResult(
            handled=True,
            text=ui.render_take_profit_menu(self.state.take_profit_pct),
            keyboard=ui.take_profit_keyboard(self.state.take_profit_pct),
            state_after=self.state,
        )

    def _select_take_profit(self, raw: str) -> ButtonResult:
        """
        Apply an operator's take-profit selection.

        INERT BY CONSTRUCTION, exactly like the strategy selection, and asserted
        below so a future change to this handler cannot silently make it able to
        start the runtime, arm REAL, clear an emergency stop or unblock entries.

        Changing the level applies to NEW positions. It is snapshotted onto a
        position at open, and an open position reads its own copy, so nothing
        already in the book is rewritten.
        """
        from ..paper_live.take_profit import (
            NO_TAKE_PROFIT,
            TakeProfitError,
            normalise,
        )

        before = self.state.run_state
        before_armed = self.state.real_armed
        before_enabled = self.state.real_trading_enabled
        before_blocked = self.state.entries_blocked
        before_stop = self.state.emergency_stop
        before_level = self.state.take_profit_pct

        token = (raw or "").strip()
        if token.upper() == "CUSTOM":
            # Not a level: the button is a PROMPT. The operator types the value
            # as ``/tp <value>`` and it is validated by the same handler, so
            # there is no second validation path and no unbounded callback
            # payload carrying a number a keyboard cannot render.
            from ..paper_live.take_profit import (
                TAKE_PROFIT_MAX_PCT,
                TAKE_PROFIT_MIN_PCT,
            )
            return ButtonResult(
                handled=True,
                text=(
                    "✏️ <b>CUSTOM TAKE PROFIT</b>\n\n"
                    f"Type the level as a command:\n<code>/tp 12.5</code>\n\n"
                    f"Accepted range: "
                    f"<b>+{TAKE_PROFIT_MIN_PCT:g}%</b> to "
                    f"<b>+{TAKE_PROFIT_MAX_PCT:g}%</b>\n"
                    "Type <code>/tp off</code> to disable take profit.\n\n"
                    "A value outside the range is REFUSED and the active "
                    "selection is left unchanged."
                ),
                keyboard=ui.take_profit_keyboard(self.state.take_profit_pct),
                state_after=self.state,
            )
        if token.upper() == "OFF":
            level: float | None = NO_TAKE_PROFIT
        else:
            try:
                level = normalise(token)
            except TakeProfitError as exc:
                # A REFUSED value must leave the previous level in force. A
                # typo that silently disabled a configured take-profit would
                # change what an open position would eventually be allowed to
                # do, so the old value is restored explicitly rather than
                # merely "not overwritten".
                self.state.take_profit_pct = before_level
                return ButtonResult(
                    handled=True,
                    text=(
                        f"❌ Take profit <b>{ui.esc(token)}</b> is not a valid "
                        f"level.\n\n{ui.esc(str(exc))}\n\n"
                        f"Active selection unchanged: "
                        f"<b>{ui.esc(ui.take_profit_label(before_level))}</b>"
                    ),
                    keyboard=ui.take_profit_keyboard(before_level),
                    state_after=self.state,
                    error="invalid_take_profit",
                )

        # A rejected value can never reach here, so these assertions hold for
        # every accepted value too.
        assert self.state.run_state is before, (
            "choosing a take profit must not change the run state")
        assert self.state.real_armed is before_armed, (
            "choosing a take profit must not arm REAL")
        assert self.state.real_trading_enabled is before_enabled, (
            "choosing a take profit must not change REAL authorization")
        assert self.state.entries_blocked is before_blocked, (
            "choosing a take profit must not unblock entries")
        assert self.state.emergency_stop is before_stop, (
            "choosing a take profit must not clear the emergency stop")

        self.state.take_profit_pct = level
        self._journal("operator_select_take_profit",
                      take_profit_pct=level,
                      take_profit_label=ui.take_profit_label(level))
        self._changed()
        if self.actions.persist_take_profit is not None:
            self.actions.persist_take_profit(level)

        return ButtonResult(
            handled=True,
            text=ui.render_take_profit_menu(level),
            keyboard=ui.take_profit_keyboard(level),
            state_after=self.state,
        )

    def _take_profit_menu(self) -> ButtonResult:
        """
        Open the take-profit submenu. Read-only: shows the live level and offers
        the others. Changes nothing.
        """
        return ButtonResult(
            handled=True,
            text=ui.render_take_profit_menu(self.state.take_profit_pct),
            keyboard=ui.take_profit_keyboard(self.state.take_profit_pct),
            state_after=self.state,
        )

    def _source_selection(self) -> str:
        if self.actions.current_signal_source is None:
            return "AUTO"
        return str(self.actions.current_signal_source() or "AUTO")

    def _signal_source_menu(self) -> ButtonResult:
        selection = self._source_selection()
        return ButtonResult(
            handled=True,
            text=ui.render_signal_source_menu(selection),
            keyboard=ui.signal_source_keyboard(selection),
            state_after=self.state,
        )

    def _select_signal_source(self, value: str) -> ButtonResult:
        allowed = {"MooDeng", "PainApe", "AUTO"}
        if value not in allowed:
            return ButtonResult(
                handled=True, text="แหล่งสัญญาณไม่ถูกต้อง", keyboard=self._kb(),
                state_after=self.state, error="invalid_signal_source",
            )
        if self.actions.persist_signal_source is not None:
            self.actions.persist_signal_source(value)
        self._journal("operator_select_signal_source", signal_source=value)
        return ButtonResult(
            handled=True,
            text=ui.render_signal_source_menu(value),
            keyboard=ui.signal_source_keyboard(value),
            state_after=self.state,
        )

    def _select_profile(self, raw: str) -> ButtonResult:
        """
        Apply an operator's strategy selection to the authoritative state.

        PAPER TOGGLES (multi-select, one independent virtual bankroll per arm);
        REAL REPLACES (single-select, because one real signal must never produce
        more than one real entry).

        Selection is INERT by construction: it cannot start the runtime, cannot
        arm REAL (``real_armed`` is ``init=False`` and only ``arm_real()`` sets
        it) and does not touch any open position. The assertions below exist so
        that if any of that ever changed, this test would fail rather than the
        operator discovering it on a live account.
        """
        before = self.state.run_state
        before_armed = self.state.real_armed
        before_enabled = self.state.real_trading_enabled
        before_blocked = self.state.entries_blocked
        before_stop = self.state.emergency_stop

        accepted, message = self.state.toggle_strategy(raw)

        assert self.state.run_state is before, (
            "selecting a strategy must not change the run state"
        )
        assert self.state.real_armed is before_armed, (
            "selecting a strategy must not arm REAL"
        )
        # Before/after, NOT an absolute: the selection is provably inert with
        # respect to REAL authorization even while REAL is armed.
        assert self.state.real_trading_enabled is before_enabled, (
            "selecting a strategy must not change REAL authorization"
        )
        assert self.state.entries_blocked is before_blocked, (
            "selecting a strategy must not unblock entries"
        )
        assert self.state.emergency_stop is before_stop, (
            "selecting a strategy must not clear the emergency stop"
        )

        if accepted:
            self._journal("operator_select_strategy",
                          strategies=list(self.state.arms))
            self._changed()
            if self.actions.persist_strategies is not None:
                self.actions.persist_strategies(list(self.state.arms))
        return ButtonResult(
            handled=True,
            text=f"{message}\n\n{ui.render_strategy_menu(self.state)}",
            keyboard=ui.strategy_keyboard(self.state),
            state_after=self.state,
            error=None if accepted else "invalid_strategy_selection",
        )

    def _menu(self) -> ButtonResult:
        """
        ``/start``. Show the operator menu and change NOTHING.

        This is the entry point an operator arrives at, which makes it the worst
        possible place for a side effect: no state transition, no preflight, no
        network call, no journal entry that implies an action was taken. It is
        also synchronous on purpose -- it must not be able to wait on anything,
        because a control panel that can hang is a control panel that gets
        force-quit.
        """
        return ButtonResult(
            handled=True,
            text=ui.render_menu(self.state, self._source_selection()),
            keyboard=self._kb(),
            state_after=self.state,
        )

    def _start(self) -> ButtonResult:
        try:
            self.state.start()
        except ModeError as exc:
            return ButtonResult(
                handled=True, text=str(exc), keyboard=self._kb(),
                state_after=self.state, error=str(exc),
            )
        self._changed()
        self._journal("operator_start")
        mode = "PAPER" if self.state.trade_mode is TradeMode.PAPER else "REAL"
        # Name the accounting basis when PAPER started. The operator is about to
        # read the fills that follow, and those fills come from executable
        # quotes rather than from marks. Saying so here, once, is the
        # difference between a dataset they can interpret and one they have to
        # reverse-engineer from the columns later.
        note = ""
        if self.state.trade_mode is TradeMode.PAPER:
            note = (
                "\n<i>PAPER · simulated fills from executable quotes</i>\n"
                "<i>ไม่มีการส่งธุรกรรมจริง</i>"
            )
        return ButtonResult(
            handled=True,
            text=(
                f"{ui.EMOJI['ok']} <b>เริ่มบอทแล้ว</b>\n"
                f"โหมด: <b>{mode}</b>\n"
                # Must name the ACTIVE strategy. This is the first thing the
                # operator reads after pressing START, so naming the delegated
                # V3-C leg here tells them the wrong thing at the moment they
                # are relying on the bot.
                f"รอสัญญาณ {ACTIVE_STRATEGY} จาก Moo Deng President"
                f"{note}"
            ),
            keyboard=self._kb(),
            state_after=self.state,
        )

    def _stop(self) -> ButtonResult:
        self.state.stop()
        self._changed()
        self._journal("operator_stop")
        return ButtonResult(
            handled=True,
            text=(
                f"{ui.EMOJI['stop']} <b>หยุดบอทแล้ว</b>\n"
                "ไม่มีการเข้าเทรใหม่\n"
                "ออเดอร์ที่เปิดอยู่ยังถูกติดตาม"
            ),
            keyboard=self._kb(),
            state_after=self.state,
        )

    def _paper(self) -> ButtonResult:
        self.state.select_paper()
        self._changed()
        self._journal("operator_select_paper")
        return ButtonResult(
            handled=True, text=ui.render_paper(), keyboard=self._kb(),
            state_after=self.state,
        )

    def _real_request(self) -> ButtonResult:
        """
        Pressing REAL. Produces a confirmation and arms NOTHING.

        Note the shape: this branch never calls ``arm_real`` and never touches
        ``real_armed``. That is the whole point - the button is a question, and
        the confirmation is the answer.
        """
        # The risk acknowledgement is a GATE, not a decoration. A config that
        # never set I_UNDERSTAND_REAL_TRADING_RISKS cannot even request a
        # confirmation.
        if self.actions.risk_acknowledged is not None:
            try:
                acknowledged = bool(self.actions.risk_acknowledged())
            except Exception:  # noqa: BLE001 - an unreadable gate refuses
                acknowledged = False
            if not acknowledged:
                self._journal("operator_request_real_refused",
                              reason="risk_not_acknowledged")
                return ButtonResult(
                    handled=True,
                    text=(
                        f"{ui.EMOJI['err']} <b>ยังไม่ยืนยันความเสี่ยง</b>\n"
                        "ต้องตั้งค่า I_UNDERSTAND_REAL_TRADING_RISKS=YES "
                        "ใน .env.local ก่อน"
                    ),
                    keyboard=self._kb(),
                    state_after=self.state,
                    error="risk_not_acknowledged",
                )
        token = self.state.request_real_confirmation()
        self._journal("operator_request_real")
        text, kb = ui.render_real_prompt(token)
        return ButtonResult(
            handled=True, text=text, keyboard=kb, state_after=self.state
        )

    def _cancel_real(self) -> ButtonResult:
        self.state.cancel_real_confirmation()
        self._journal("operator_cancel_real")
        return ButtonResult(
            handled=True,
            text=(
                f"{ui.EMOJI['paper']} <b>ยกเลิกแล้ว</b>\n"
                "ยังไม่เปิด REAL Trading"
            ),
            keyboard=self._kb(),
            state_after=self.state,
        )

    async def _confirm_real(self, token: str) -> ButtonResult:
        """
        [✅ ยืนยัน REAL]. The only path to REAL_ARMED.

        Order matters and is not negotiable: the token is validated first (so a
        replayed or fabricated payload is rejected before any I/O), then the
        preflight runs, and only a passing preflight is handed to
        ``arm_real``. A failing preflight consumes the token, so the operator
        has to make a fresh, deliberate request.
        """
        expected = self.state._pending_confirmation
        if not expected or not token or token != expected:
            self._journal("real_confirm_rejected", reason="bad_token")
            return ButtonResult(
                handled=True,
                text=(
                    f"{ui.EMOJI['err']} <b>ยืนยันไม่ถูกต้อง</b>\n"
                    "กด 🔴 REAL ใหม่เพื่อขอยืนยันอีกครั้ง"
                ),
                keyboard=self._kb(),
                state_after=self.state,
                error="bad_token",
            )

        if self.actions.preflight is None:
            self.state.cancel_real_confirmation()
            return ButtonResult(
                handled=True,
                text=(
                    f"{ui.EMOJI['err']} <b>PREFLIGHT ไม่พร้อม</b>\n"
                    "ไม่มีตัวตรวจสอบ เปิด REAL ไม่ได้"
                ),
                keyboard=self._kb(),
                state_after=self.state,
                error="preflight_unavailable",
            )

        try:
            result = await self.actions.preflight()
        except Exception as exc:  # noqa: BLE001 - a failed probe is a refusal
            self.state.cancel_real_confirmation()
            self._journal("real_preflight_error", error=type(exc).__name__)
            return ButtonResult(
                handled=True,
                text=(
                    f"{ui.EMOJI['err']} <b>PREFLIGHT ผิดพลาด</b>\n"
                    f"{ui.esc(type(exc).__name__)}\n\n"
                    "ยังไม่เปิด REAL Trading"
                ),
                keyboard=self._kb(),
                state_after=self.state,
                error=f"preflight_error:{type(exc).__name__}",
            )

        if not getattr(result, "ok", False):
            # Stay exactly where we were, and say where that is.
            self.state.cancel_real_confirmation()
            self._journal(
                "real_preflight_failed",
                failures=[
                    c.name for c in getattr(result, "failures", []) or []
                ],
            )
            return ButtonResult(
                handled=True,
                text=ui.render_preflight_failed(result, self.state),
                keyboard=self._kb(),
                state_after=self.state,
                error="preflight_failed",
            )

        try:
            self.state.arm_real(
                confirmation=token, preflight_ok=True,
                preflight_detail=getattr(result, "detail", ""),
            )
        except ModeError as exc:
            self._journal("real_arm_refused", reason=str(exc))
            return ButtonResult(
                handled=True, text=ui.esc(exc), keyboard=self._kb(),
                state_after=self.state, error=str(exc),
            )

        self._changed()
        self._journal(
            "real_armed",
            preflight_detail=getattr(result, "detail", ""),
        )
        if self.actions.on_real_armed is not None:
            self.actions.on_real_armed(self.state)

        # The MEASURED wallet and equity. This used to pass a literal "—" and
        # omit the equity, so the operator always saw "Equity: $0.0000" whatever
        # the wallet held. An unavailable reading is reported as unavailable
        # with its reason, never as a zero.
        wallet_masked, equity_usd, equity_reason = await self._real_equity()
        return ButtonResult(
            handled=True,
            text=ui.render_real_armed(
                self.state, wallet_masked, equity_usd, equity_reason
            ),
            keyboard=self._kb(),
            state_after=self.state,
        )

    async def _real_equity(self) -> tuple[str, float | None, str]:
        """
        ``(wallet_masked, equity_usd, reason)`` from the runtime.

        Fails safe: if no provider is wired, or it raises, the reply says the
        equity is unavailable rather than showing a plausible zero.
        """
        fn = self.actions.extras.get("real_equity")
        if fn is None:
            return ("(unavailable)", None, "no equity provider configured")
        try:
            result = await fn() if _is_awaitable_call(fn) else fn()
        except Exception as exc:  # noqa: BLE001
            return ("(unavailable)", None,
                    f"equity read failed ({type(exc).__name__})")
        return (
            getattr(result, "wallet_masked", "") or "(unavailable)",
            getattr(result, "equity_usd", None),
            getattr(result, "reason", "") or "",
        )

    async def _balance(self) -> ButtonResult:
        """
        /balance. On demand ONLY.

        Nothing polls the wallet for this view: the query happens when the
        operator asks and not otherwise, so it is never on the trading path.
        A failed read reports "unavailable" and never a $0.00.
        """
        fn = self.actions.extras.get("real_equity")
        if fn is None:
            return ButtonResult(
                handled=True,
                text=ui.render_balance(None, reason="no equity provider configured"),
                keyboard=self._kb(), state_after=self.state,
                error="equity_unavailable",
            )
        try:
            result = await fn() if _is_awaitable_call(fn) else fn()
        except Exception as exc:  # noqa: BLE001
            return ButtonResult(
                handled=True,
                text=ui.render_balance(None,
                                       reason=f"query failed ({type(exc).__name__})"),
                keyboard=self._kb(), state_after=self.state,
                error="equity_unavailable",
            )
        return ButtonResult(
            handled=True, text=ui.render_balance(result),
            keyboard=self._kb(), state_after=self.state,
        )

    def _pause(self) -> ButtonResult:
        """
        /pause. Stops NEW entries, keeps tracking open positions.

        Maps onto the existing audited ``stop()`` transition rather than adding
        a state. No network call and no analysis is triggered.
        """
        self.state.stop()
        self._changed()
        self._journal("operator_pause")
        return ButtonResult(
            handled=True,
            text=("⏸ <b>หยุดชั่วคราว</b>\nไม่มีการเข้าเทรใหม่\n"
                  "ออเดอร์ที่เปิดอยู่ยังถูกติดตาม"),
            keyboard=self._kb(), state_after=self.state,
        )

    def _resume(self) -> ButtonResult:
        """
        /resume. Back to running, still PAPER unless REAL was armed.

        This also CLEARS the emergency-stop latch, and that is deliberate. /kill
        latches across restarts by design, and the command set was reduced to
        five, so /clear_emergency no longer exists. Without this, /kill would be
        a one-way door: the latch survives every restart and the operator would
        have to delete a file by hand to trade again. /resume is the documented
        inverse of both /pause and /kill.

        Clearing the latch is an explicit operator action, never automatic: it
        only happens because someone pressed the button, and it is journalled.
        """
        # Order matters: start() REFUSES while the latch is set, so the latch
        # has to be cleared first or this button would raise on exactly the
        # state it exists to recover from.
        was_latched = self.state.emergency_stop
        if was_latched:
            self.state.clear_emergency_stop()
            persist = self.actions.persist_emergency
            if persist is not None:
                persist(False)
        self.state.start()
        self._changed()
        self._journal("operator_resume")
        cleared = (
            "\nปลด EMERGENCY STOP แล้ว" if was_latched else ""
        )
        return ButtonResult(
            handled=True,
            text=(f"▶️ <b>ทำงานต่อ</b>\nโหมด: <b>"
                  f"{'PAPER' if self.state.trade_mode is TradeMode.PAPER else 'REAL'}</b>"
                  f"{cleared}"),
            keyboard=self._kb(), state_after=self.state,
        )

    def _kill(self) -> ButtonResult:
        """
        /kill. The emergency stop, unchanged.

        Latched and persisted, so it survives a restart. This is the one command
        that must remain able to stop trading without any network call.
        """
        return self._emergency_stop()

    async def _recent_orders(self) -> ButtonResult:
        """The most recently closed trades, for operator inspection."""
        fn = self.actions.extras.get("recent_orders")
        if fn is None:
            return ButtonResult(
                handled=True, text="ยังไม่มีประวัติเทรด", keyboard=self._kb(),
                state_after=self.state,
            )
        rows = await fn() if _is_awaitable_call(fn) else fn()
        return ButtonResult(
            handled=True,
            text=ui.render_recent_orders(rows),
            keyboard=self._kb(),
            state_after=self.state,
        )

    def _real_off(self) -> ButtonResult:
        self.state.turn_off_real()
        self._changed()
        self._journal("operator_real_off")
        return ButtonResult(
            handled=True, text=ui.render_real_off(), keyboard=self._kb(),
            state_after=self.state,
        )

    def _emergency_stop(self) -> ButtonResult:
        self.state.emergency_stop_now()
        if self.actions.persist_emergency is not None:
            self.actions.persist_emergency(True)
        self._changed()
        self._journal("emergency_stop")
        if self.actions.on_emergency_stop is not None:
            self.actions.on_emergency_stop()
        return ButtonResult(
            handled=True, text=ui.render_emergency_stop(), keyboard=self._kb(),
            state_after=self.state,
        )

    def _clear_emergency(self) -> ButtonResult:
        self.state.clear_emergency_stop()
        if self.actions.persist_emergency is not None:
            self.actions.persist_emergency(False)
        self._changed()
        self._journal("emergency_cleared")
        return ButtonResult(
            handled=True, text=ui.render_emergency_cleared(), keyboard=self._kb(),
            state_after=self.state,
        )

    async def _close_all_paper(self) -> ButtonResult:
        """
        First press: show confirmation with count of open PAPER positions.
        """
        # Must be in PAPER mode - this is a PAPER-only kill switch
        if self.state.trade_mode is not TradeMode.PAPER:
            return ButtonResult(
                handled=True,
                text=(
                    f"{ui.EMOJI['err']} <b>REJECTED</b>\n"
                    "CLOSE ALL PAPER is only available in PAPER mode.\n"
                    f"Current mode: <b>{self.state.trade_mode.value}</b>"
                ),
                keyboard=self._kb(),
                state_after=self.state,
                error="not_paper_mode",
            )

        fn = self.actions.extras.get("open_positions")
        if fn is None:
            return ButtonResult(
                handled=True,
                text=(
                    f"{ui.EMOJI['err']} <b>UNAVAILABLE</b>\n"
                    "Cannot list open positions."
                ),
                keyboard=self._kb(), state_after=self.state,
            )
        rows = await fn() if _is_awaitable_call(fn) else fn()

        # Filter to PAPER positions only
        paper_rows = [r for r in (rows or []) if str(r.get("mode", "")).upper() != "REAL"]
        open_count = len(paper_rows)

        if open_count == 0:
            return ButtonResult(
                handled=True,
                text=(
                    f"{ui.EMOJI['paper']} <b>CLOSE ALL PAPER</b>\n\n"
                    "No open PAPER positions to close."
                ),
                keyboard=self._kb(),
                state_after=self.state,
            )

        text, kb = ui.render_close_all_paper_prompt(open_count)
        return ButtonResult(
            handled=True, text=text, keyboard=kb, state_after=self.state
        )

    async def _close_all_paper_confirm(self) -> ButtonResult:
        """
        Confirmed press: execute emergency close for all PAPER positions.
        """
        # Must be in PAPER mode
        if self.state.trade_mode is not TradeMode.PAPER:
            return ButtonResult(
                handled=True,
                text=(
                    f"{ui.EMOJI['err']} <b>REJECTED</b>\n"
                    "CLOSE ALL PAPER is only available in PAPER mode."
                ),
                keyboard=self._kb(),
                state_after=self.state,
                error="not_paper_mode",
            )

        fn = self.actions.extras.get("close_all_paper")
        if fn is None:
            return ButtonResult(
                handled=True,
                text=(
                    f"{ui.EMOJI['err']} <b>UNAVAILABLE</b>\n"
                    "Emergency close function not wired."
                ),
                keyboard=self._kb(), state_after=self.state,
                error="close_all_paper_not_wired",
            )

        try:
            result = await fn() if _is_awaitable_call(fn) else fn()
        except Exception as exc:
            return ButtonResult(
                handled=True,
                text=(
                    f"{ui.EMOJI['err']} <b>ERROR</b>\n"
                    f"{type(exc).__name__}: {exc}"
                ),
                keyboard=self._kb(), state_after=self.state,
                error=f"close_all_paper_failed: {type(exc).__name__}",
            )

        text = ui.render_close_all_paper_result(result)
        return ButtonResult(
            handled=True, text=text, keyboard=self._kb(), state_after=self.state
        )

    def _close_all_paper_cancel(self) -> ButtonResult:
        """Cancel the close all paper operation."""
        return ButtonResult(
            handled=True,
            text=f"{ui.EMOJI['paper']} <b>CANCELLED</b>\nCLOSE ALL PAPER aborted.",
            keyboard=self._kb(),
            state_after=self.state,
        )

    async def _stats(self) -> ButtonResult:
        fn = self.actions.extras.get("stats")
        if fn is None:
            return ButtonResult(
                handled=True, text="สถิติยังไม่พร้อม", keyboard=self._kb(),
                state_after=self.state,
            )
        paper, real = await fn() if _is_awaitable_call(fn) else fn()
        return ButtonResult(
            handled=True,
            text=ui.render_stats(paper=paper, real=real),
            keyboard=self._kb(),
            state_after=self.state,
        )

    async def _open_positions(self) -> ButtonResult:
        """
        OPEN POSITIONS.

        Read-only and on demand. The reader is the engine's
        ``open_positions_view``, which reads the same paper/real account objects
        the accounting layer reads, so this cannot disagree with ``/balance`` or
        ``/status``. It costs no network call: marks come from the engine's own
        cache, refreshed by the monitor loop that is already running.

        Nothing about this branch can change a state. A read-only view that
        could move money would not be a view.
        """
        fn = self.actions.extras.get("open_positions")
        if fn is None:
            return ButtonResult(
                handled=True,
                text=ui.render_open_positions([]),
                keyboard=self._kb(), state_after=self.state,
            )
        rows = await fn() if _is_awaitable_call(fn) else fn()
        return ButtonResult(
            handled=True,
            text=ui.render_open_positions(rows or []),
            keyboard=self._kb(),
            state_after=self.state,
        )

    async def _closed_positions(self, offset: int = 0) -> ButtonResult:
        """
        CLOSED POSITIONS, newest first, one page per press.

        The reader is the engine's ``closed_positions_view``, built from the
        ledger's ``exit`` events - the same records the statistics come from, so
        there is no second "closed trades" list that can drift.

        A fixed recent window per reply, plus the total, rather than one message
        per trade: the ledger holds hundreds of rows and the chat must not.
        """
        fn = self.actions.extras.get("closed_positions")
        if fn is None:
            return ButtonResult(
                handled=True,
                text=ui.render_closed_positions([], total=0, offset=offset),
                keyboard=self._kb(), state_after=self.state,
            )
        result = await fn(offset) if _is_awaitable_call(fn) else fn(offset)
        if isinstance(result, tuple):
            rows, total = result
        else:
            rows, total = result, None
        return ButtonResult(
            handled=True,
            text=ui.render_closed_positions(rows or [], total=total, offset=offset),
            keyboard=self._kb(),
            state_after=self.state,
        )

    async def _open_orders(self) -> ButtonResult:
        fn = self.actions.extras.get("open_orders")
        if fn is None:
            return ButtonResult(
                handled=True, text="ยังไม่มีออเดอร์เปิด", keyboard=self._kb(),
                state_after=self.state,
            )
        rows = await fn() if _is_awaitable_call(fn) else fn()
        return ButtonResult(
            handled=True,
            text=ui.render_open_orders(rows),
            keyboard=self._kb(),
            state_after=self.state,
        )


def _is_awaitable_call(fn: Any) -> bool:
    import inspect

    return inspect.iscoroutinefunction(fn)
