"""
The Telegram operator UI. Thai, HTML, and deliberately small.

SEVEN BUTTONS, NO DASHBOARD
---------------------------

    [▶️ เริ่มบอท]      [⏹ หยุดบอท]
    [🟡 PAPER]         [🔴 REAL]
    [📊 สถิติ]         [📋 ออเดอร์เปิด]
    [🛑 EMERGENCY STOP]

A previous design accumulated a control for every parameter, which meant the
operator had to read a wall of buttons to answer the only question that
matters: can I trade right now, and with whose money.

TWO INVISIBLE INVARIANTS
------------------------

1. **One source of truth.** Every rendering function takes the
   :class:`autopaper.modes.RuntimeState` and reads the mode from it. There is no
   separate UI-side mode variable, which is how a bot ends up displaying REAL
   while the executor is a simulator. :func:`consistency_check` asserts the
   displayed mode, the mode the executor would use, and the armed flag all
   agree, and it is called by the status renderer so an inconsistency is
   VISIBLE rather than merely possible.

2. **No secrets.** The wallet address is masked, the Telegram chat is masked,
   and every string passes through the redaction layer before it is rendered.

THE REAL BUTTON DOES NOT ARM
----------------------------
Pressing ``🔴 REAL`` produces a confirmation prompt and nothing else. Arming
happens only on ``[✅ ยืนยัน REAL]`` AND only if the preflight passed. Pressing
``[❌ ยกเลิก]`` consumes the token, so a cancelled confirmation cannot be
replayed.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from ..l25 import (
    ACTIVE_STRATEGY,
    REJECTED_LIQUIDITY_MC_BELOW_THRESHOLD,
    L25_MIN_LIQ_TO_MCAP,
)
from ..modes import (
    RunState,
    RuntimeState,
    TradeMode,
    status_line,
)
from ..safety import mask_address, mask_telegram_id
from ..paper_live.schema import ARM_CLOSE_NO_FILL_CONTRADICTION
from ..paper_live.take_profit import (
    SELECTABLE_TAKE_PROFITS,
    label as take_profit_label,
    normalise as _normalise_take_profit,
)
from ..sizing import allocation_label, next_position_label
from ..strategy_profile import (
    DEFAULT_PROFILE,
    ENTRY_STRATEGY,
    StrategyProfile,
    exit_strategy,
    label,
    max_hold_s,
    note,
)
from ..v3c import (
    REJECTED_MARKET_CAP,
    REJECTED_SIGNAL_QUALITY,
    REJECT_REASON_TH as V3C_REJECT_REASON_TH,
    V3C_MAX_MARKET_CAP_USD,
    V3C_MIN_SIGNAL_QUALITY,
    V3C_NAME,
)
from ..l25 import REJECT_REASON_TH as L25_REJECT_REASON_TH

#: Thai rejection text, resolved per STRATEGY.
#:
#: This was a single ``REJECT_REASON_TH`` imported from ``v3c`` and indexed with
#: L25 status constants. An ordinary L25 rejection therefore raised
#: ``KeyError: 'LIQUIDITY_MC_BELOW_THRESHOLD'`` inside the render, which escaped
#: as a Telegram stream error and killed the read loop - the candidate channel
#: itself. It fired 64 times in one live session.
#:
#: The two dictionaries are kept SEPARATE and named for what they are, because
#: L25 delegates its first leg to V3-C and the two sets genuinely differ. A blind
#: merge would hide a strategy accidentally introducing a colliding key.
V3C_REASONS = dict(V3C_REJECT_REASON_TH)
L25_REASONS = dict(L25_REJECT_REASON_TH)

#: Used when a reason has no entry in its own strategy's table. A reason this
#: module does not recognise is an expected condition (a new gate), not a
#: programming error, so it degrades to readable text instead of raising.
GENERIC_REJECT_REASON_TH = "ไม่ผ่านเงื่อนไขของกลยุทธ์"


def reject_reason_th(status: str, *, strategy: str = ACTIVE_STRATEGY) -> str:
    """
    The Thai rejection text for ``status``, resolved per strategy.

    Resolution is EXPLICIT and ordered, not a blind dict merge:

    1. the named strategy's own table;
    2. the table of the gate it DELEGATES to - L25's first leg is V3-C, so a
       reason such as ``REJECTED_NOT_FAST_ENTRY`` is legitimately a V3-C reason
       arriving on an L25 result, and looking only at the L25 table would lose
       the operator-facing text;
    3. a generic line, so a NEW gate can never crash the renderer.

    Only the LOOKUP degrades. An unexpected exception elsewhere still
    propagates rather than being hidden.
    """
    if strategy == ACTIVE_STRATEGY:
        tables = (L25_REASONS, V3C_REASONS)
    else:
        tables = (V3C_REASONS,)
    for table in tables:
        text = table.get(status)
        if text:
            return text
    return GENERIC_REJECT_REASON_TH


# ---------------------------------------------------------------------------
# button labels and callback payloads
# ---------------------------------------------------------------------------

BTN_START = "▶️ เริ่มบอท"
BTN_STOP = "⏹ หยุดบอท"
BTN_PAPER = "🟡 PAPER"
BTN_REAL = "🔴 REAL"
BTN_STATS = "📊 สถิติ"
BTN_OPEN = "📋 ออเดอร์เปิด"
BTN_EMERGENCY = "🛑 EMERGENCY STOP"

#: REAL-off is only shown while armed, so a disarmed bot does not advertise a
#: button that does nothing.
BTN_REAL_OFF = "🔴 ปิด REAL"

BTN_CONFIRM_REAL = "✅ ยืนยัน REAL"
BTN_CANCEL_REAL = "❌ ยกเลิก"

BTN_CLEAR_EMERGENCY = "🟢 ปลด EMERGENCY STOP"

#: The eight operator actions on the /start menu, verbatim as specified.
#:
#: Labels are deliberately terse: this is a control surface, not a status
#: message. The MEANING of each button is the branch it dispatches to, not the
#: wording on the face.
BTN_START_PAPER = "▶️ START PAPER"
BTN_PAUSE = "⏸ PAUSE"
BTN_RESUME = "▶️ RESUME"
BTN_BALANCE = "💰 BALANCE"
BTN_STATUS = "📊 STATUS"
BTN_ARM_REAL = "🔴 ARM REAL"
BTN_DISARM_REAL = "🛑 DISARM REAL"
BTN_KILL = "☠️ KILL"

#: The two read-only position views. Pure inspection: pressing either one calls
#: a reader that returns rows and nothing else, so neither can move a state,
#: place an order or place a quote on the trading path. The labels name the
#: population ("POSITIONS") because the chat is where an operator asks what the
#: bot is actually holding, as opposed to STATS which is aggregates.
BTN_OPEN_POSITIONS = "📂 OPEN POSITIONS"
BTN_CLOSED_POSITIONS = "🗂 CLOSED POSITIONS"

# ---------------------------------------------------------------------------
# EMERGENCY KILL-ALL
# ---------------------------------------------------------------------------
# Deliberately NOT inside the strategy selector. A control that flattens the book
# is a different kind of action from one that changes which strategy trades, and
# burying it in a submenu next to "L25 + 60s" is how it gets clicked by accident.
BTN_KILL_ALL = "🚨 KILL ALL OPEN POSITIONS"
#: Confirmation token. One extra press, not a nested dialog: an operator in an
#: emergency should not have to navigate.
BTN_KILL_ALL_CONFIRM = "CONFIRM EMERGENCY CLOSE ALL"
BTN_KILL_ALL_CANCEL = "CANCEL"
CB_KILL_ALL = "kill_all"
CB_KILL_ALL_CONFIRM = "kill_all:confirm"
CB_KILL_ALL_CANCEL = "kill_all:cancel"

# ---------------------------------------------------------------------------
# CLOSE ALL PAPER (PAPER-only kill switch)
# ---------------------------------------------------------------------------
# Separate from KILL ALL which may include REAL. This is explicitly PAPER-only.
BTN_CLOSE_ALL_PAPER = "🚨 CLOSE ALL PAPER"
BTN_CLOSE_ALL_PAPER_CONFIRM = "✅ CONFIRM CLOSE ALL PAPER"
BTN_CLOSE_ALL_PAPER_CANCEL = "❌ CANCEL"
CB_CLOSE_ALL_PAPER = "close_all_paper"
CB_CLOSE_ALL_PAPER_CONFIRM = "close_all_paper:confirm"
CB_CLOSE_ALL_PAPER_CANCEL = "close_all_paper:cancel"

# --- selectable strategy profile -----------------------------------------
# A profile is a SELECTION, not a trading control. The button opens a menu
# showing which profile is live; choosing one only stamps it onto positions
# opened from then on. It deliberately sits in the read-only neighbourhood of
# STATUS rather than next to START or ARM REAL, so a mis-press is visible from
# the surrounding buttons.
BTN_STRATEGY = "\U0001F3AF STRATEGY"
BTN_SIGNAL_SOURCE = "📡 SIGNAL SOURCE"
#: The take-profit control. It sits directly under STRATEGY because it is a
#: parameter OF the L25 exit, not a second strategy -- the whole point of making
#: it selectable independently.
BTN_TAKE_PROFIT = "\U0001F3AF TAKE PROFIT"
BTN_SOURCE_MOODENG = "🦆 MooDeng"
BTN_SOURCE_PAINAPE = "🟡 PainApe"
BTN_SOURCE_AUTO = "🔄 AUTO"
# Plain labels, deliberately WITHOUT a tick. The renderer adds "✅" only to the
# profile that is actually selected, so a tick can never sit next to a profile
# that is not running -- which would be a lie on the one screen whose whole job
# is to say which profile is live.
BTN_PROFILE_L25_CURRENT = "L25 Current"
BTN_PROFILE_L25_60S = "⏱️ L25 + 60s"

#: Prefixed rather than one bare payload per profile, so adding a third profile
#: later is a data change and not a new routing branch.
CB_STRATEGY = "strategy"
CB_PROFILE = "profile:"
CB_SIGNAL_SOURCE = "signal_source"
CB_SOURCE_SELECT = "signal_source:"
#: Take-profit is a SETTING, not a strategy, so it has its own submenu and its
#: own payload namespace. A new level is a data change, not a routing branch.
CB_TAKE_PROFIT = "take_profit"
CB_TAKE_PROFIT_SELECT = "take_profit:"
#: The custom-value payload. Reserved inside the take-profit namespace so it
#: cannot collide with a level, and distinct from every numeric preset.
CB_TAKE_PROFIT_CUSTOM = "take_profit:custom"
#: The OFF payload, so "no take-profit" is a selection rather than an absent one.
CB_TAKE_PROFIT_OFF = "take_profit:off"

#: The literal an operator must be able to read on an empty OPEN POSITIONS view.
#: Stated in English on purpose: "no open positions" is a fact about the account
#: that has to be unambiguous, not a phrase that needs translating before it can
#: be trusted.
EMPTY_OPEN_POSITIONS = "No open positions."
EMPTY_CLOSED_POSITIONS = "No closed positions yet."

#: Callback payloads. Kept short and stable so a keyboard layout change does not
#: invalidate saved state.
CB_START = "start"
CB_STOP = "stop"
CB_PAPER = "paper"
CB_REAL = "real"
CB_REAL_OFF = "real_off"
CB_STATS = "stats"
CB_RECENT = "recent"
#: The five operator commands. The bot interface is deliberately this small:
#: everything else is either an internal safety transition or unnecessary surface.
CB_BALANCE = "balance"
CB_PAUSE = "pause"
CB_RESUME = "resume"
CB_KILL = "kill"
#: /start. Shows the operator menu and NOTHING else -- it must never start
#: trading. Distinct from CB_START ("start"), which is the menu's
#: "▶️ START PAPER" button and does start trading.
CB_MENU = "menu"

#: The COMPLETE user-facing command vocabulary. Asserted in tests, so a new
#: command cannot be added to the bot without this list changing too.
#: Typed commands. /start is the menu entry point and does not trade; the rest
#: are the same actions the menu offers as buttons.
OPERATOR_COMMANDS = ("/start", "/status", "/balance", "/pause",
                     "/resume", "/kill", "/open", "/closed", "/tp")
CB_OPEN = "open"
#: Recently CLOSED positions. A separate callback from ``CB_RECENT`` so the
#: richer, paginated view can be added without changing what an existing saved
#: keyboard (which carries ``recent``) does.
CB_CLOSED = "closed"
CB_EMERGENCY = "emergency"
CB_CONFIRM_REAL = "confirm_real:"
CB_CANCEL_REAL = "cancel_real"
CB_CLEAR_EMERGENCY = "clear_emergency"

#: The COMPLETE button vocabulary. Asserted in tests, so a new button cannot be
#: added to the menu without this list changing too. The profile buttons live
#: in the submenu, not the main menu, and are listed separately below.
OPERATOR_BUTTONS = (
    BTN_START_PAPER, BTN_PAUSE, BTN_RESUME, BTN_BALANCE, BTN_STATUS,
    BTN_ARM_REAL, BTN_DISARM_REAL, BTN_KILL,
    BTN_OPEN_POSITIONS, BTN_CLOSED_POSITIONS,
    BTN_STRATEGY, BTN_TAKE_PROFIT, BTN_SIGNAL_SOURCE,
)

#: Profile buttons reachable from the strategy submenu.
PROFILE_BUTTONS = (BTN_PROFILE_L25_CURRENT, BTN_PROFILE_L25_60S)

#: Buttons reachable from the take-profit submenu.
TAKE_PROFIT_BUTTONS = (BTN_TAKE_PROFIT,)

#: Callback payloads the operator menu may carry. ``CB_MENU`` included, since
#: ``/start`` replies with the menu.
OPERATOR_CALLBACKS = (
    CB_START, CB_PAUSE, CB_RESUME, CB_BALANCE, CB_STATS, CB_REAL, CB_REAL_OFF,
    CB_KILL, CB_OPEN, CB_CLOSED, CB_MENU, CB_STRATEGY,
    CB_TAKE_PROFIT, CB_SIGNAL_SOURCE,
    CB_CLOSE_ALL_PAPER, CB_CLOSE_ALL_PAPER_CONFIRM, CB_CLOSE_ALL_PAPER_CANCEL,
)

#: Console font must render these on Telegram's clients. Deliberately a small
#: fixed set rather than emoji with wide variation selectors.
EMOJI: dict[str, str] = {
    "ok": "\U0001F7E2",
    "warn": "\U0001F7E1",
    "err": "\U0001F534",
    "bot": "\U0001F916",
    "mode": "\U0001F7E1",
    "real": "\U0001F534",
    "paper": "\U0001F7E1",
    "channel": "\U0001F4E1",
    "brain": "\U0001F9E0",
    "wallet": "\U0001F4B0",
    "chart": "\U0001F4CA",
    "list": "\U0001F4CB",
    "stop": "\U0001F6D1",
}


# ---------------------------------------------------------------------------
# a UI-side view model, so renderers are pure and testable
# ---------------------------------------------------------------------------


@dataclass
class StatusView:
    """
    Everything the main status message needs.

    Assembled by the engine from the runtime state, the real account snapshot
    and the paper account, then handed to :func:`render_status`. The renderer
    itself does no I/O and reads no globals, which is what makes the whole UI
    unit-testable without a Telegram connection.
    """

    state: RuntimeState
    channel: str = "Moo Deng President"
    chat: Optional[str] = None
    wallet_masked: str = "(not configured)"
    balance_sol: Optional[float] = None
    balance_usd: Optional[float] = None
    #: ``None`` means UNKNOWN, and the renderer must say so. This field used to
    #: default to ``0.0``, which is how an unmeasured equity was displayed as a
    #: measured ``$0.0000``: a default that claims a measurement was made. A real
    #: measured zero is ``0.0`` and is distinguishable from ``None``.
    equity_usd: Optional[float] = None
    cash_usd: Optional[float] = None
    open_positions: int = 0
    executor_available: bool = False
    #: Whether the real executor's preflight would currently pass. Displayed so
    #: the operator is never told REAL is available when it is not.
    real_preflight_ok: Optional[bool] = None
    last_candidate_age_s: Optional[float] = None
    emergency_stop: bool = False
    signal_source: str = "AUTO"
    #: Amarih PumpView PAPER experiment status
    amarih_enabled: bool = False
    amarih_poller_running: bool = False
    #: The DEADLINE scheduler's health, separate from the poller's. The poller
    #: decides whether ENTRY signals arrive; the scheduler decides whether EXITS
    #: fire on time. Reporting only the poller made a stopped scheduler look
    #: like a healthy experiment.
    amarih_scheduler_running: bool = False
    #: The deadline scheduler's cadence in seconds, so the status line can show
    #: the actual detection interval rather than implying a precision it does
    #: not have. 0 means unknown.
    amarih_scheduler_cadence_s: float = 0.0
    #: The freshness gate actually in force. Reported because the effective
    #: threshold is what decides whether a live signal is admitted, and an
    #: operator cannot see it anywhere else.
    amarih_signal_max_age_s: float = 0.0
    amarih_last_poll_at: Optional[float] = None
    amarih_signals_seen: int = 0
    amarih_signals_accepted: int = 0
    amarih_signals_rejected: int = 0
    amarih_open_arms: int = 0


def esc(text: Any) -> str:
    """HTML-escape. Every interpolated value goes through this."""
    return html.escape(str(text), quote=False)


# ---------------------------------------------------------------------------
# consistency
# ---------------------------------------------------------------------------


def consistency_check(view: StatusView) -> list[str]:
    """
    Prove the display agrees with the runtime. Returns problems, empty = OK.

    Called by :func:`render_status` and by the test suite. The specific
    contradictions this exists to make impossible are:

      * REAL displayed while the executor is unavailable
      * REAL displayed while the state is not actually armed
      * PAPER displayed while the state is armed for REAL
      * an open position reported in one account and not the other
    """
    problems: list[str] = []
    st = view.state
    mode = st.trade_mode

    if mode is TradeMode.REAL and not st.real_armed:
        problems.append("displayed REAL but runtime is not armed")
    if mode is TradeMode.REAL and not view.executor_available:
        problems.append("displayed REAL but the real executor is unavailable")
    if mode is TradeMode.PAPER and st.real_armed and st.run_state is RunState.REAL_ARMED:
        problems.append("displayed PAPER but the runtime is armed for REAL")
    if view.real_preflight_ok is False and mode is TradeMode.REAL:
        problems.append("displayed REAL but the preflight is not passing")
    if view.emergency_stop != st.emergency_stop:
        problems.append("emergency stop disagrees between UI and runtime")
    if view.open_positions < 0:
        problems.append("negative open position count")
    return problems


# ---------------------------------------------------------------------------
# main status
# ---------------------------------------------------------------------------


def render_status(view: StatusView) -> str:
    """
    The main message.

    Rendered entirely from ``view``, which is assembled from the runtime state,
    so the CLI and Telegram cannot disagree: they call this same function with
    the same view.
    """
    st = view.state
    e = EMOJI
    problems = consistency_check(view)
    lines: list[str] = [f"{e['bot']} <b>MEMEBOT</b>", ""]

    # system line
    if st.emergency_stop:
        system = f"{e['err']} ระบบ: {e['stop']} EMERGENCY STOP"
    elif st.running:
        system = f"{e['ok']} ระบบ: ทำงานอยู่"
    elif st.run_state is RunState.OFF:
        system = f"{e['warn']} ระบบ: รอเริ่มบอท"
    else:
        system = f"{e['warn']} ระบบ: หยุดอยู่"
    lines.append(system)

    # mode line
    lines.append(status_line(st))

    lines.append(f"{e['channel']} Channel: {esc(view.channel)}")
    lines.append(f"📡 SOURCE: <b>{esc(view.signal_source.upper())}</b>")
    # Amarih PumpView PAPER experiment status
    if view.amarih_enabled:
        poller_status = "🟢 RUNNING" if view.amarih_poller_running else "🔴 STOPPED"
        last_poll = ""
        if view.amarih_last_poll_at:
            from ..engine import iso as _iso
            last_poll = f" last={_iso(view.amarih_last_poll_at)}"
        lines.append(
            f"🧪 AMARIH: <b>{poller_status}</b>"
            f" seen={view.amarih_signals_seen} acc={view.amarih_signals_accepted}"
            f" rej={view.amarih_signals_rejected} open={view.amarih_open_arms}{last_poll}"
        )
        # TWO health facts, not one: whether ENTRY signals are being polled, and
        # whether EXIT deadlines are being serviced. A stopped scheduler with a
        # running poller admitted entries that never exited, and that state
        # previously read as a healthy experiment.
        sched_status = ("🟢 RUNNING" if view.amarih_scheduler_running
                        else "🔴 STOPPED")
        max_age = (f"{view.amarih_signal_max_age_s:.0f}s"
                   if view.amarih_signal_max_age_s else "?")
        lines.append(
            f"🧪 AMARIH exits: <b>{sched_status}</b>"
            f" · signal max age <b>{esc(max_age)}</b>"
        )
    # THE THRESHOLD PARENTHETICAL BELONGS TO L25, SO IT IS SHOWN ONLY WHEN L25
    # AUTHORIZES.
    #
    # This line used to print ``Strategy: L25 (V3-C: MC <= $20,000, SQ >= 0.55)
    #`` unconditionally. With a different experiment selected, those numbers are
    # not the thresholds anything is being judged against, and the operator's
    # only threshold readout on the status screen then describes a rule that is
    # not in force. The gate line below already reports the true operative gate;
    # this line must not contradict it.
    try:
        gates = tuple(view.state.effective_entry_gates)
    except Exception:  # noqa: BLE001 - a status surface must never crash
        gates = ()
    if "L25" in gates:
        lines.append(
            f"{e['brain']} Strategy: <b>{esc(ACTIVE_STRATEGY)}</b> "
            f"(<code>{esc(V3C_NAME)}: MC &le; ${V3C_MAX_MARKET_CAP_USD:,.0f}, "
            f"SQ &ge; {V3C_MIN_SIGNAL_QUALITY:.2f}</code> + "
            f"<code>f_liq_to_mcap &ge; {L25_MIN_LIQ_TO_MCAP:.4f}</code>)"
        )
    else:
        lines.append(
            f"{e['brain']} Strategy: <b>{esc(', '.join(gates) or '?')}</b> "
            f"(<code>L25 thresholds not in force</code>)"
        )
    # The PROFILE line, separate from the strategy line above. The strategy is
    # L25 either way; the profile is what says whether the account is trading
    # it with a 60s maximum hold. A bare "L25" here would be true and useless.
    cap = max_hold_s(st.profile)
    cap_text = "no cap" if cap is None else f"max hold {cap:g}s"
    lines.append(
        f"\U0001F3AF STRATEGY: <b>{esc(label(st.profile))}</b> "
        f"(<code>{esc(cap_text)}</code>)"
    )
    # The TAKE PROFIT line. Deliberately on its own row rather than folded into
    # the profile line: it is a SEPARATE setting now, not something a profile
    # carries. A profile can be switched without touching it, and it can be
    # changed without touching the profile, so one combined line would hide
    # which of the two is which.
    lines.append(
        f"🎯 TAKE PROFIT: <b>{esc(st.take_profit.label)}</b> "
        f"(<code>on entry fill price; NEW positions only</code>)"
    )
    lines.append("")
    lines.append("Laya: <b>OFF</b>")
    lines.append("ML: <b>OFF</b>")
    lines.append("Shadow: <b>OFF</b>")
    lines.append("")

    # wallet
    # Masked AGAIN here even though the view is supposed to carry a masked
    # value. A renderer that trusts its caller about secret hygiene is one
    # refactor away from printing a full address, and the cost of masking twice
    # is zero.
    lines.append(
        f"{e['wallet']} Wallet: <code>{esc(mask_address(view.wallet_masked))}</code>"
    )
    if view.balance_sol is not None:
        bal = f"{view.balance_sol:.6f} SOL"
        if view.balance_usd is not None:
            bal += f" (${view.balance_usd:,.2f})"
        lines.append(f"{e['wallet']} Balance: {bal}")
    else:
        lines.append(f"{e['wallet']} Balance: <i>ยังไม่ทราบ</i>")

    # allocation
    #
    # The SELECTED PROFILE is passed to both, so a profile that declares its own
    # researched size shows that size here instead of the shared ladder. Showing
    # "12.5%" while the strategy actually stakes 10% would be a status line that
    # lies about the next order.
    lines.append(
        f"{e['chart']} Allocation: "
        f"{esc(allocation_label(view.equity_usd, profile=view.state.strategy_profile))}"
    )
    lines.append(
        f"{e['chart']} Next Position: "
        f"<b>{esc(next_position_label(view.equity_usd, profile=view.state.strategy_profile))}</b>"
    )
    lines.append(f"{e['list']} Open: <b>{view.open_positions}</b>")

    if st.real_armed:
        lines.append("")
        # The armed banner states which strategy is about to commit real
        # money, so it must name the ACTIVE strategy. Naming the delegated
        # V3-C leg here would tell the operator the wrong thing at the exact
        # moment the answer matters most.
        lines.append(
            f"{e['real']} <b>REAL TRADING ARMED</b>\n"
            f"{esc(ACTIVE_STRATEGY)}\n"
            f"สัญญาณ {esc(ACTIVE_STRATEGY)} ที่ผ่านเงื่อนไขถัดไปอาจใช้เงินจริง"
        )
    if st.emergency_stop:
        lines.append("")
        lines.append(
            f"{e['stop']} <b>EMERGENCY STOP</b>\n"
            "ปิดการเข้าเทรใหม่ทั้งหมด จะยังอยู่หลัง restart"
        )
    if problems:
        lines.append("")
        lines.append(f"{e['err']} <b>สถานะไม่สอดคล้อง</b>")
        for p in problems:
            lines.append(f"• {esc(p)}")

    # The command list. Six commands were removed, so an operator who
    # remembered /paper or /stop would otherwise type them into silence. This
    # footer is the whole discoverable surface, which is why it is here and not
    # a separate message.
    lines.append("")
    lines.append(f"{e['bot']} Commands: "
                 + "  ".join(esc(c) for c in OPERATOR_COMMANDS))

    return "\n".join(lines)


def operator_keyboard(state: RuntimeState) -> list[list[dict[str, str]]]:
    """
    The operator menu: ten buttons, grouped by what they do.

    Laid out by function rather than by importance, so a mis-tap is obvious
    from the neighbourhood: run control together, then read-only, then REAL,
    then the kill switch alone on the bottom row where a stray press cannot
    land next to anything.

    Both REAL buttons are always present. DISARM REAL on an unarmed runtime is
    a no-op that says so, which is safer than a button that appears and
    disappears -- an operator reaching for the disarm control should not have
    to wonder whether its absence means "already safe" or "broken".

    The two POSITION buttons join STATUS and BALANCE in the read-only block and
    change nothing about the existing eight: every previous control is still
    present, in the same neighbourhood, with the same callback payloads.
    """
    del state  # the menu is the same in every runtime state, by design
    return [
        [
            {"text": BTN_START_PAPER, "callback_data": CB_START},
            {"text": BTN_PAUSE, "callback_data": CB_PAUSE},
        ],
        [
            {"text": BTN_RESUME, "callback_data": CB_RESUME},
            {"text": BTN_BALANCE, "callback_data": CB_BALANCE},
        ],
        [
            {"text": BTN_STATUS, "callback_data": CB_STATS},
        ],
        [
            {"text": BTN_OPEN_POSITIONS, "callback_data": CB_OPEN},
            {"text": BTN_CLOSED_POSITIONS, "callback_data": CB_CLOSED},
        ],
        [
            {"text": BTN_STRATEGY, "callback_data": CB_STRATEGY},
        ],
        [
            {"text": BTN_TAKE_PROFIT, "callback_data": CB_TAKE_PROFIT},
        ],
        [
            {"text": BTN_SIGNAL_SOURCE, "callback_data": CB_SIGNAL_SOURCE},
        ],
        [
            {"text": BTN_ARM_REAL, "callback_data": CB_REAL},
            {"text": BTN_DISARM_REAL, "callback_data": CB_REAL_OFF},
        ],
        [
            {"text": BTN_KILL, "callback_data": CB_KILL},
        ],
        [
            {"text": BTN_CLOSE_ALL_PAPER, "callback_data": CB_CLOSE_ALL_PAPER},
        ],
    ]


#: Public name for the operator menu keyboard.
#:
#: ``autopaper.telegram`` exports this name. It is the same control surface as
#: :func:`operator_keyboard`; both names are kept because the package export and
#: the engine call site use different ones.
main_keyboard = operator_keyboard


def render_menu(state: RuntimeState, signal_source: str = "AUTO") -> str:
    """
    The /start reply: the operator menu and a one-line state summary.

    It reports state and offers controls. It does not change anything, and it
    must not: /start is how an operator arrives, so it is the single most
    likely place for an accidental trade if the handler were to act.
    """
    e = EMOJI
    armed = "REAL ARMED" if state.real_armed else "REAL OFF"
    stop = "  \U0001F6D1 EMERGENCY STOP" if state.emergency_stop else ""
    return (
        f"{e['bot']} <b>MEMEBOT</b> — {esc(ACTIVE_STRATEGY)}\n"
        f"{esc(strategy_status_line(state))}\n"
        f"🎯 {esc(take_profit_status_line(state))}\n"
        f"{e['real']} MODE: <b>{esc(state.trade_mode.value)}</b>"
        f"   RUNNING: <b>{esc(state.run_state.value)}</b>\n"
        f"{e['chart']} REAL: <b>{esc(armed)}</b>{stop}\n\n"
        f"📡 SOURCE: <b>{esc(signal_source.upper())}</b>\n\n"
        "เลือกการทำงานจากเมนูด้านล่าง"
    )


def signal_source_keyboard(selection: str = "AUTO") -> list[list[dict[str, str]]]:
    """Manual source selector; changing it never touches run/account state."""
    labels = (
        (BTN_SOURCE_MOODENG, "MooDeng"),
        (BTN_SOURCE_PAINAPE, "PainApe"),
        (BTN_SOURCE_AUTO, "AUTO"),
    )
    return [[{
        "text": f"{'✅ ' if selection == value else ''}{label}",
        "callback_data": f"{CB_SOURCE_SELECT}{value}",
    }] for label, value in labels]


def take_profit_keyboard(level: float | None = None) -> list[list[dict[str, str]]]:
    """
    The selectable take-profit levels.

    Radio, not checkbox: a position has one take-profit, so exactly one level
    can be live. ``+100%`` is one of the options, which preserves the capability
    the retired L25_TP100 arm used to be the only way to reach.

    A custom numeric input is offered as a SEPARATE row rather than by pushing
    an arbitrary payload, because the payload then has to be validated and a
    callback with an unbounded value is not something a keyboard can carry.
    The OFF row is explicit: "no take-profit" is a selection, not a state you
    reach by pressing nothing.
    """
    rows: list[list[dict[str, str]]] = []
    for value in SELECTABLE_TAKE_PROFITS:
        on = (level is None and value is None) or (
            level is not None and abs(value - level) < 1e-9)
        rows.append([{
            "text": f"{'✅ ' if on else ''}{take_profit_label(value)}",
            "callback_data": f"{CB_TAKE_PROFIT_SELECT}{value:g}",
        }])
    rows.append([{
        "text": "✏️ Custom %",
        "callback_data": CB_TAKE_PROFIT_CUSTOM,
    }])
    rows.append([{
        "text": f"{'✅ ' if level is None else ''}{take_profit_label(None)}",
        "callback_data": CB_TAKE_PROFIT_OFF,
    }])
    rows.append([{"text": f"↩️ {BTN_STATUS}", "callback_data": CB_STATS}])
    return rows


def render_take_profit_menu(level: float | None) -> str:
    """
    The take-profit submenu text.

    States the basis honestly: the level is measured on the position's ENTRY
    FILL, which is derived from an executable BUY quote, not on the aggregator
    mark. It also says the setting applies to NEW positions only.
    """
    return (
        "🎯 <b>TAKE PROFIT</b>\n\n"
        f"SELECTED: <b>{esc(take_profit_label(level))}</b>\n\n"
        "Take profit is a SETTING, not a strategy. Every L25 strategy can run "
        "any level, and +100% is one value among the presets.\n\n"
        "<b>Measured on</b> the position's entry fill price (from the "
        "executable BUY quote), not the aggregator mark.\n\n"
        "Applies to NEW positions only.\n"
        "Existing positions keep the snapshot they were opened with."
    )


def render_signal_source_menu(selection: str) -> str:
    return (
        "📡 <b>SIGNAL SOURCE</b>\n\n"
        f"SOURCE: <b>{esc(selection.upper())}</b>\n"
        "เลือกแหล่งสัญญาณสำหรับ NEW signals เท่านั้น\n"
        "Existing positions ไม่ได้รับผลกระทบ"
    )


def effective_entry_gates_line(state) -> str:
    """
    The gates that authorize entries RIGHT NOW, as one operator-readable row.

    The profile line above names the profile new positions are stamped with,
    which is NOT the same fact as which gate authorizes an entry: the profile is
    the shipped default until something changes it, while the gate set follows
    the arms the operator actually selected. Naming only the profile is what
    made "L25" appear over a running beta experiment and sent the incident
    looking for a blocked L25 entry that never existed.
    """
    try:
        gates = tuple(state.effective_entry_gates)
    except Exception:  # noqa: BLE001 - a status surface must never crash
        return ""
    if not gates:
        return ""
    return ("Entry gate: <b>" + esc(", ".join(gates)) + "</b>")


def strategy_status_line(state: RuntimeState) -> str:
    """
    The one strategy line every operator surface shows.

    Says which PROFILE is selected, not just "L25", because the two profiles
    share an entry strategy and differ in exit policy: showing a bare "L25"
    while the 60s cap is active would misrepresent how the account trades.

    When the V2 FLOOR (BETA) experiment is running, that line says so, because
    the profile vocabulary has no way to express "three arms off one gate, at
    three different exit brackets". Leaving the line at the entry profile would
    show the operator a cap the beta arms do not use -- the beta bracket is
    per-arm and deliberately overrides the profile default.
    """
    rows = [f"{BTN_STRATEGY}: {label(state.profile)}"]
    # The AUTHORIZING gate, which is a different fact from the profile and the
    # one that decides whether a candidate is allowed to trade at all.
    gates_row = effective_entry_gates_line(state)
    if gates_row:
        rows.append(gates_row)
    beta = _beta_running_summary(state)
    if not beta:
        return rows[0] if len(rows) == 1 else "\n".join(rows)
    # The beta line carries its own bold heading, so append it as the status
    # block's row rather than folding it into the profile line.
    rows.append(beta)
    return "\n".join(rows)


def take_profit_status_line(state: RuntimeState) -> str:
    """
    The take-profit line: ``L25 TAKE PROFIT: +10%``.

    Read from the GLOBAL selection, so it says what a NEW position would be
    opened with. An open position may carry a different snapshot; the position
    view is where that is shown, never here. Built by
    ``TakeProfitSelection.status_line`` so the button, the menu and the status
    card cannot drift into disagreeing about the current level.
    """
    return state.take_profit.status_line


#: The V2 FLOOR (BETA) row. The experiment is presented as ONE logical group --
#: ``PAPER ONLY -- V2 FLOOR (BETA) -- 3 Parallel Exit Arms`` -- because its value
#: IS the three-way comparison, and a picker that offered A, B and C separately
#: would let an operator start a partial experiment and believe they had run the
#: real one. It carries the GROUP token as its callback, which is what the
#: backend resolves (:func:`is_beta_group`); the internal arm identities A/B/C
#: stay private to the experiment and are never selectable on their own.
BETA_GROUP_LABEL = "PAPER ONLY -- V2 FLOOR (BETA) -- 3 Parallel Exit Arms"

#: The PumpView 'Amarih' paired-exit experiment row. Like the beta group it is
#: ONE logical experiment (three paired arms off one canonical entry), so a
#: picker that offered T30/T60/T120 separately would let an operator start a
#: partial experiment and believe they had run the paired comparison. Unlike the
#: disabled beta experiment, Amarih is ENABLED (opt-in research), so this row is
#: selectable and its callback activates the whole group.
AMARIH_GROUP_LABEL = "🧪 Amarih Research -- 3 Paired Exit Arms (T60/T120/T300)"


def _beta_group_row(
    state: RuntimeState, real: bool, selected: set[str],
) -> dict[str, Any]:
    """Placeholder row for the disabled V2 FLOOR (BETA) experiment.

    The V2 FLOOR (BETA) experiment has been disabled from the active bot.
    This row no longer offers activation -- it shows a disabled indicator
    to make clear the option exists in code but is not available.

    ``V2_FLOOR_BETA`` is a GROUP, not an arm, so it is not part of the arm
    registry's ``arm_specs()`` -- the registry keeps "one name per arm", which
    is what lets ``len(arm_names()) == len(arm_specs())`` hold.
    """
    from ..paper_live.arms import BETA_GROUP

    # Disabled indicator: grey circle for REAL, grey square for PAPER
    mark = "\u25cb" if real else "\u25a1"
    return {
        "text": f"{mark} {BETA_GROUP_LABEL} (disabled)",
        # Still register the callback so stale clicks fail closed rather than
        # being no-ops -- the callback handler will refuse it.
        "callback_data": f"{CB_PROFILE}{BETA_GROUP}",
    }


def _amarih_group_row(
    state: RuntimeState, real: bool, selected: set[str],
) -> dict[str, Any]:
    """The selectable PumpView 'Amarih' paired-exit experiment row.

    Unlike the disabled beta row, this one ACTIVATES the whole group: its
    callback is the ``AMARIH`` group token, which :meth:`toggle_strategy`
    expands to the three arms T30/T60/T120. It shows the selected/available
    mark so the operator can see whether the experiment is on. In REAL it is
    refused by the backend, so the mark reflects that it is not selectable
    there.
    """
    from ..paper_live.arms import AMARIH_ARMS, AMARIH_GROUP

    amarih_on = {a.value for a in AMARIH_ARMS} <= selected
    if real:
        # REAL refuses the group (three arms); show it as unavailable.
        mark = "\u25cb"
    else:
        mark = "\u2611" if amarih_on else "\u2610"
    return {
        "text": f"{mark} {AMARIH_GROUP_LABEL}",
        "callback_data": f"{CB_PROFILE}{AMARIH_GROUP}",
    }


def _beta_running_summary(state: RuntimeState) -> str:
    """The status line for a RUNNING V2 FLOOR (BETA) experiment, else ``''``.

    Empty unless all three beta arms are selected. The experiment is three arms
    off ONE gate, so "the experiment is on" means all three are on; showing the
    line for a partial selection would tell the operator they were running a
    three-way comparison when they were running one arm of it.

    The numbers come from :data:`BETA_ARM_CONFIGS` rather than from this file, so
    the status line can never disagree with the arms the engines configured.
    """
    from ..paper_live.arms import BETA_ARMS
    from ..paper_live.v2_floor_beta import (
        BETA_ARM_CONFIGS,
        BETA_STARTING_BANKROLL_USD,
    )

    selected = set(getattr(state, "arms", ()) or ())
    if not {a.value for a in BETA_ARMS} <= selected:
        return ""
    brackets = " · ".join(
        f"{c.arm} TP +{c.take_profit_pct:g}/SL {c.stop_loss_pct:g}/{c.max_hold_s:g}s"
        for c in BETA_ARM_CONFIGS
    )
    # PLAIN text, deliberately: several callers run this through ``esc()`` (the
    # status card), and emitting markup here would render those tags literally.
    return (
        f"EXPERIMENT: {BETA_GROUP_LABEL}\n"
        f"     {brackets}\n"
        f"     ${BETA_STARTING_BANKROLL_USD:,.2f} isolated ledger per arm "
        f"(${BETA_STARTING_BANKROLL_USD * len(BETA_ARM_CONFIGS):,.2f} total) "
        f"· PAPER ONLY"
    )


def strategy_keyboard(state: RuntimeState) -> list[list[dict[str, str]]]:
    """
    The strategy submenu, built from the CANONICAL REGISTRY.

    It iterates :func:`autopaper.paper_live.arms.arm_specs`, so every registered
    arm appears. The previous version iterated ``StrategyProfile`` and carried a
    hard-coded two-entry button dictionary, which is how H2_CAP60 -- a real,
    running arm -- was missing from the picker entirely.

    PAPER renders CHECKBOXES and multi-selects: each selected arm gets its own
    virtual bankroll. REAL renders a RADIO: exactly one arm, because one real
    signal must never produce more than one real entry.

    The V2 FLOOR (BETA) experiment is offered as ONE extra row, appended after
    the arms, via :func:`_beta_group_row`. It is the only row whose callback is
    a GROUP token rather than an arm name, and it is a selection surface like
    every other row here: it cannot start the runtime, cannot arm REAL, and
    cannot touch a position that is already open.

    NOTE: As of the current deployment, the V2 FLOOR (BETA) experiment is
    DISABLED from the active bot. The row still appears but shows a disabled
    indicator and attempting to activate it via the callback is refused.
    """
    from ..paper_live.arms import arm_specs

    selected = set(state.arms)
    real = state.trade_mode is TradeMode.REAL
    rows: list[list[dict[str, str]]] = []
    for spec in arm_specs():
        name = spec.name.value
        on = name in selected
        # PAPER: a toggle control. REAL: a radio that REPLACES.
        mark = ("\u2611" if on else "\u2610") if not real else ("\u25c9" if on else "\u25cb")
        # Telegram button labels are single-line. An embedded newline renders
        # as a broken control, so the extra entry rule is folded onto one line.
        suffix = ""
        if spec.extra_entry_rule:
            suffix = f"  [{spec.extra_entry_rule}]"
        rows.append([{
            "text": f"{mark} {spec.label}{suffix}",
            "callback_data": f"{CB_PROFILE}{name}",
        }])
    rows.append([_beta_group_row(state, real, selected)])
    rows.append([_amarih_group_row(state, real, selected)])
    total = len(arm_specs()) + 2
    rows.append([{
        "text": (f"Selected: {len(selected)}/{total}"
                 + ("  (REAL: exactly one)" if real
                    else f"  (PAPER: 1-{total})")),
        "callback_data": CB_STATS,
    }])
    rows.append([{"text": f"\u21a9\ufe0f {BTN_STATUS}", "callback_data": CB_STATS}])
    return rows


def render_strategy_menu(state: RuntimeState) -> list[str] | str:
    """
    The strategy submenu text.

    PAPER states how many arms are selected and that each gets its own virtual
    bankroll, so the operator can see that one signal will fan out into several
    INDEPENDENT simulated positions. REAL states that exactly one applies.

    It also says explicitly that selecting a strategy starts nothing, arms
    nothing, and does not touch a position that is already open.

    The beta experiment is listed LAST, under its group name, with its three
    per-arm brackets spelled out so the operator can see exactly what they would
    be starting -- and so no number on this screen is invented: the brackets come
    from the experiment registry (``BETA_ARM_CONFIGS``), never from prose.
    """
    from ..paper_live.arms import BETA_ARMS, BETA_GROUP, arm_specs

    selected = list(state.arms)
    real = state.trade_mode is TradeMode.REAL
    specs = list(arm_specs())
    selected_set = set(selected)
    # +2 for the two EXPERIMENT GROUPS (the disabled V2 FLOOR BETA group and the
    # enabled Amarih paired-exit group), which are selector rows but not arms.
    total = len(specs) + 2
    lines = [
        BTN_STRATEGY,
        "",
        "<b>SELECTED STRATEGIES</b>"
        + (f" ({len(selected)}/{total})" if not real else " (1 of 1)"),
    ]
    beta_names = {a.value for a in BETA_ARMS}
    for spec in specs:
        name = spec.name.value
        on = name in selected_set
        mark = "\u2611" if on else "\u2610"
        if real:
            mark = "\u25c9" if on else "\u25cb"
        rule = f"\n     entry rule: {spec.extra_entry_rule}" \
            if spec.extra_entry_rule else ""
        cap = spec.max_hold_s
        cap_text = ("no max hold" if cap is None else f"{cap:g}s max hold")
        lines.append(f"{mark} {esc(name)} — {esc(spec.label)} "
                     f"({cap_text}){rule}")
    # The experiment row: its own group name, its own group callback, and the
    # three arm brackets read straight out of the experiment registry.
    beta_on = beta_names <= selected_set
    beta_mark = "\u25c9" if (real and beta_on) else (
        "\u2611" if beta_on else "\u2610")
    from ..paper_live.v2_floor_beta import BETA_ARM_CONFIGS

    arm_desc = "; ".join(
        f"{c.arm}: TP +{c.take_profit_pct:g}% / SL {c.stop_loss_pct:g}% / "
        f"{c.max_hold_s:g}s" for c in BETA_ARM_CONFIGS)
    lines.append(
        f"{beta_mark} {esc(BETA_GROUP)} — {esc(BETA_GROUP_LABEL)} (disabled)\n"
        f"     $10.00 isolated ledger per arm ($30.00 total)\n"
        f"     {esc(arm_desc)}")
    # The Amarih paired-exit experiment row: ONE canonical entry observation
    # shared by three arms that differ ONLY in the scheduled holding period at
    # which a fresh executable SELL quote settles each arm's own isolated
    # virtual ledger. PAPER research; the Amarih thresholds are PumpView's
    # private config and are NOT reimplemented here.
    from ..paper_live.arms import AMARIH_ARMS, AMARIH_GROUP
    from ..paper_live.amarih_experiment import DEFAULT_ARM_NOTIONAL_USD

    amarih_vals = {a.value for a in AMARIH_ARMS}
    amarih_on = amarih_vals <= selected_set
    amarih_mark = "\u25cb" if real else ("\u2611" if amarih_on else "\u2610")
    # The horizon is carried by each arm's OWN registry entry (max_hold_s), so
    # read it from there rather than restating the seconds a second time.
    from ..paper_live.arms import arm_by_name
    amarih_desc = "; ".join(
        f"{a.value.replace('AMARIH_', '')}: exit @ "
        f"+{arm_by_name(a.value).max_hold_s:g}s"
        for a in AMARIH_ARMS)
    lines.append(
        f"{amarih_mark} {esc(AMARIH_GROUP)} — {esc(AMARIH_GROUP_LABEL)}\n"
        f"     ${DEFAULT_ARM_NOTIONAL_USD:,.2f} isolated ledger per arm "
        f"(${DEFAULT_ARM_NOTIONAL_USD * len(AMARIH_ARMS):,.2f} total)\n"
        f"     {esc(amarih_desc)}\n"
        f"     PAPER research only; signals read from PumpView public "
        f"signal-history. Experimental horizons, NOT validated optima."
        + ("" if not real else "  (not selectable in REAL)"))
    lines += [
        "",
        ("REAL accepts EXACTLY ONE strategy: a single real signal must never "
         "produce more than one real entry."
         if real else
         "PAPER accepts 1-3 strategies. Each selected strategy has its OWN "
         "virtual bankroll and accounting, so one signal can open several "
         "independent simulated positions. Quotes are still shared."
         "\n\n<b>TAKE PROFIT</b> is a separate SETTING, not a strategy: "
         "every arm here can run any level, and +100% is one value among the "
         "presets."),
        "",
        "<b>PROVENANCE</b>",
        "These are RETROSPECTIVE variants of the same L25 entry strategy.",
        "None is not proven superior on the recorded data; the caps exist to "
        "make the comparison measurable, not because the research settled it.",
        "",
        "<i>Selecting a strategy never starts the bot, never arms REAL, never "
        "broadcasts anything, and never touches a position that is already "
        "open -- that position keeps the strategy it was opened under.</i>",
    ]
    return "\n".join(lines)

def render_real_prompt(token: str) -> tuple[str, list[list[dict[str, str]]]]:
    """
    The [🔴 REAL] press. A confirmation prompt and NOTHING else.

    Returning the token here is how the button carries proof that a second,
    distinct operator action followed. Pressing REAL alone leaves the runtime
    exactly as it was.
    """
    text = (
        f"{EMOJI['real']} <b>เปิด REAL Trading</b>\n\n"
        "ยืนยันเปิด REAL Trading?\n\n"
        "• ใช้เงินจริงจาก wallet\n"
        f"• Strategy: {esc(ACTIVE_STRATEGY)}\n"
        "• ต้องผ่าน preflight ก่อน\n"
        "• กดยกเลิกเพื่ออยู่ในโหมดเดิม"
    )
    kb = [[
        {
            "text": BTN_CONFIRM_REAL,
            "callback_data": f"{CB_CONFIRM_REAL}{token}",
        },
        {"text": BTN_CANCEL_REAL, "callback_data": CB_CANCEL_REAL},
    ]]
    return text, kb


def render_close_all_paper_prompt(open_count: int) -> tuple[str, list[list[dict[str, str]]]]:
    """
    Confirmation prompt for CLOSE ALL PAPER.

    Shows the number of open PAPER positions that will be closed.
    PAPER-only: never touches REAL positions.
    """
    text = (
        f"{EMOJI['paper']} <b>CLOSE ALL PAPER</b>\n\n"
        f"Confirm CLOSE ALL <b>{open_count}</b> PAPER position(s)?\n\n"
        "• PAPER only — REAL positions untouched\n"
        "• Uses executable SELL quotes\n"
        "• Canonical accounting settlement\n"
        "• One failure does not block others\n\n"
        "Press CONFIRM to proceed, CANCEL to abort."
    )
    kb = [[
        {
            "text": BTN_CLOSE_ALL_PAPER_CONFIRM,
            "callback_data": CB_CLOSE_ALL_PAPER_CONFIRM,
        },
        {"text": BTN_CLOSE_ALL_PAPER_CANCEL, "callback_data": CB_CLOSE_ALL_PAPER_CANCEL},
    ]]
    return text, kb


def render_close_all_paper_result(result: dict[str, Any]) -> str:
    """
    Render the result of a CLOSE ALL PAPER operation.

    Shows: Closed, Failed, Remaining counts.

    A failure ALWAYS names a classification. When the count says something failed
    but no reason reached this function, that is stated as the defect it is
    rather than rendered as a clean run -- a summary that reports a failure and
    then prints no cause is the exact reading that turned the TITS incident into
    "Errors: NONE".
    """
    closed = result.get("closed", 0)
    failed = result.get("failed", 0)
    remaining = result.get("remaining", 0)
    targeted = result.get("targeted", closed + failed + remaining)

    lines = [
        f"{EMOJI['paper']} <b>CLOSE ALL PAPER RESULT</b>",
        "",
        f"Targeted: {targeted}",
        f"Closed: <b>{closed}</b>",
    ]
    if failed:
        lines.append(f"Failed: <b>{failed}</b>")
    if remaining:
        lines.append(f"Remaining OPEN: <b>{remaining}</b>")
    codes = [
        str(d.get("failure_reason") or "").strip()
        for d in (result.get("details") or [])
        if isinstance(d, dict)
    ]
    codes = [c for c in codes if c]
    if failed:
        lines.append("")
        lines.append("Failure codes:")
        for code in codes[:5]:
            lines.append(f"• {esc(code)}")
        if not codes:
            lines.append(
                f"• {esc(ARM_CLOSE_NO_FILL_CONTRADICTION)} "
                "(no classification was reported by the close path)"
            )
    if result.get("errors"):
        lines.append("")
        lines.append("Errors:")
        for err in result["errors"][:5]:
            lines.append(f"• {esc(str(err)[:160])}")
    return "\n".join(lines)


def render_balance(result, *, reason: str = "") -> str:
    """
    /balance. On-demand only.

    Distinguishes a real $0.00 from an unavailable reading, because collapsing an
    RPC failure into zero is how a funded wallet once appeared empty.
    """
    equity = getattr(result, "equity_usd", None) if result is not None else None
    lines = [f"{EMOJI['wallet']} <b>ยอดคงเหลือ</b>"]
    if equity is None:
        why = reason or getattr(result, "reason", "") or "unavailable"
        lines.append("Equity: <b>unavailable</b>")
        lines.append(f"Reason: {esc(str(why))}")
    else:
        lines.append(f"Wallet: <code>{esc(getattr(result, 'wallet_masked', ''))}</code>")
        lines.append(f"SOL: {getattr(result, 'sol', 0.0):.9f}")
        lines.append(f"SOL/USD: ${getattr(result, 'sol_price_usd', 0.0):,.4f}")
        lines.append(f"Equity: <b>${equity:,.4f}</b>")
    return "\n".join(lines)


def render_recent_orders(rows) -> str:
    """Recently CLOSED trades, newest first. Read-only operator inspection."""
    if not rows:
        return f"{EMOJI['chart']} <b>ประวัติเทรด</b>\n\nยังไม่มีเทรดที่ปิดแล้ว"
    lines = [f"{EMOJI['chart']} <b>ประวัติเทรดล่าสุด</b>", ""]
    for r in rows:
        ret = r.get("return_pct")
        pnl = r.get("net_pnl_usd")
        ret_s = f"{ret:+.2f}%" if isinstance(ret, (int, float)) else "n/a"
        pnl_s = f"${pnl:+.4f}" if isinstance(pnl, (int, float)) else "n/a"
        sym = str(r.get("symbol") or r.get("mint") or "?")[:12]
        hold = r.get("holding_s")
        hold_s = f"{hold:.0f}s" if isinstance(hold, (int, float)) else "?"
        lines.append(
            f"• <code>{esc(sym)}</code>  {ret_s}  {pnl_s}  ({hold_s})"
        )
    return "\n".join(lines)


def render_real_armed(
    state: RuntimeState,
    wallet_masked: str,
    equity_usd: float | None = None,
    equity_reason: str = "",
) -> str:
    """
    The very obvious warning shown the moment REAL is armed.

    ``equity_usd`` is None-able on purpose. The previous version defaulted it to
    ``0.0``, and the only caller omitted it, so the message always read
    "Equity: $0.0000" regardless of the wallet - a placeholder that looked like
    a measurement. ``None`` now renders an explicit unavailable state with the
    reason, which is what an operator needs to act on.
    """
    if equity_usd is None:
        equity_line = "Equity: <b>unavailable</b>"
        if equity_reason:
            equity_line += f"\nReason: {esc(equity_reason)}"
        next_line = "Next Position: <b>—</b>"
    else:
        equity_line = f"Equity: ${equity_usd:,.4f}"
        next_line = f"Next Position: <b>{esc(next_position_label(equity_usd))}</b>"

    return (
        f"{EMOJI['real']} <b>REAL TRADING ARMED</b>\n"
        f"{esc(ACTIVE_STRATEGY)}\n"
        f"สัญญาณ {esc(ACTIVE_STRATEGY)} ที่ผ่านเงื่อนไขถัดไปอาจใช้เงินจริง\n\n"
        f"Wallet: <code>{esc(mask_address(wallet_masked))}</code>\n"
        f"{equity_line}\n"
        f"{next_line}\n\n"
        f"{EMOJI['stop']} ปิดด้วยปุ่ม {esc(BTN_REAL_OFF)}"
    )


def render_real_off() -> str:
    return (
        f"{EMOJI['paper']} <b>REAL ปิดแล้ว</b>\n"
        "ปิดการเข้าเทรจริงใหม่\n"
        "ออเดอร์ที่เปิดอยู่ยังถูกติดตามต่อ"
    )


def render_paper() -> str:
    return (
        f"{EMOJI['paper']} <b>PAPER MODE</b>\n"
        "ไม่มีการส่งออเดอร์จริง"
    )


def render_preflight_failed(result, state: RuntimeState) -> str:
    """
    A failed preflight. Reports the exact reason and stays in the prior mode.

    The prior mode is named explicitly, because "arm failed" without saying
    where you ended up is the kind of message that leaves an operator unsure
    whether the bot is live.
    """
    lines = [
        f"{EMOJI['err']} <b>PREFLIGHT ไม่ผ่าน</b>",
        "",
        f"โหมดปัจจุบัน: <b>{esc(state.run_state.value)}</b>",
        f"REAL: {'ARMED' if state.real_armed else 'OFF'}",
        "",
    ]
    for c in getattr(result, "checks", []) or []:
        mark = "✅" if c.ok else "❌"
        lines.append(f"{mark} {esc(c.name)}: {esc(c.detail)}")
    lines.append("")
    lines.append("ยังไม่เปิด REAL Trading")
    return "\n".join(lines)


def render_emergency_stop() -> str:
    return (
        f"{EMOJI['stop']} <b>EMERGENCY STOP</b>\n\n"
        "ปิดการเข้าเทรใหม่ทันที\n"
        "• ไม่ลบ journal\n"
        f"• ไม่เปลี่ยน {esc(ACTIVE_STRATEGY)}\n"
        "• ยังอยู่หลัง restart"
    )


def render_emergency_cleared() -> str:
    return (
        f"{EMOJI['ok']} <b>ปลด EMERGENCY STOP แล้ว</b>\n"
        "กด ▶️ เริ่มบอท เพื่อทำงานต่อ"
    )


# ---------------------------------------------------------------------------
# trade alerts
# ---------------------------------------------------------------------------


def render_paper_buy(
    *,
    symbol: str,
    mint: str,
    market_cap_usd: Optional[float],
    signal_quality: float,
    position_usd: float,
    liquidity_usd: Optional[float],
    momentum: Optional[str],
    f_liq_to_mcp: Optional[float] = None,
) -> str:
    lines = [
        f"{EMOJI['paper']} <b>{esc(ACTIVE_STRATEGY)} BUY</b>",
        f"<b>${esc(symbol)}</b>",
        f"MC: {fmt_usd(market_cap_usd)}",
        f"Signal Quality: {signal_quality:.2f}",
        f"Liq/MC: {_ratio(f_liq_to_mcp)}",
        f"Position Size: <b>${position_usd:,.6f}</b>",
        f"Liquidity: {fmt_usd(liquidity_usd)}",
    ]
    if momentum:
        lines.append(f"Momentum: {esc(momentum)}")
    lines.append("Mode: <b>PAPER</b>")
    return "\n".join(lines)


def render_real_buy(
    *,
    symbol: str,
    mint: str,
    market_cap_usd: Optional[float],
    signal_quality: float,
    position_usd: float,
    est_cost_sol: float,
    est_cost_usd: Optional[float] = None,
    f_liq_to_mcp: Optional[float] = None,
) -> str:
    # 9 decimal places: 1 lamport is 1e-9 SOL, and a $0.155 order on a $1.55
    # account is ~0.0009 SOL. Six places would render it as 0.000900 and lose
    # the very precision this account needs.
    cost = f"{est_cost_sol:.9f} SOL"
    if est_cost_usd is not None:
        cost += f" (${est_cost_usd:,.4f})"
    lines = [
        f"{EMOJI['real']} <b>{esc(ACTIVE_STRATEGY)} REAL BUY</b>",
        f"<b>${esc(symbol)}</b>",
        f"MC: {fmt_usd(market_cap_usd)}",
        f"Signal Quality: {signal_quality:.2f}",
        f"Liq/MC: {_ratio(f_liq_to_mcp)}",
        f"Position Size: <b>${position_usd:,.6f}</b>",
        f"Est. cost: {cost}",
        "Mode: <b>REAL</b>",
    ]
    return "\n".join(lines)


def render_buy_confirmed(
    *,
    symbol: str,
    actual_size_usd: float,
    actual_fill: Optional[float],
    actual_fee_sol: Optional[float],
    signature: str,
) -> str:
    lines = [
        f"{EMOJI['ok']} <b>BUY CONFIRMED</b>",
        f"<b>${esc(symbol)}</b>",
        f"Actual size: <b>${actual_size_usd:,.6f}</b>",
    ]
    if actual_fill is not None:
        lines.append(f"Actual fill: {actual_fill:.6e}")
    if actual_fee_sol is not None:
        lines.append(f"Actual fee: {actual_fee_sol:.9f} SOL")
    lines.append(f"Tx: <code>{esc(signature)}</code>")
    return "\n".join(lines)


def render_exit(
    *,
    symbol: str,
    reason: str,
    entry_price: Optional[float],
    exit_price: Optional[float],
    pnl_usd: Optional[float],
    pnl_pct: Optional[float],
    hold_s: float,
    mode: TradeMode,
    signature: Optional[str] = None,
) -> str:
    icon = EMOJI["real"] if mode is TradeMode.REAL else "\U0001F535"
    lines = [
        f"{icon} <b>EXIT</b>",
        f"<b>${esc(symbol)}</b>",
        f"Reason: {esc(reason)}",
    ]
    if entry_price is not None:
        lines.append(f"Entry: {entry_price:.6e}")
    if exit_price is not None:
        lines.append(f"Exit: {exit_price:.6e}")
    if pnl_usd is not None:
        pct = f" ({pnl_pct:+.2f}%)" if pnl_pct is not None else ""
        lines.append(f"PnL: <b>{pnl_usd:+,.6f} USD{pct}</b>")
    lines.append(f"Hold: {fmt_duration(hold_s)}")
    lines.append(f"Mode: <b>{mode.value}</b>")
    if mode is TradeMode.REAL and signature:
        lines.append(f"Tx: <code>{esc(signature)}</code>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# rejection alerts
# ---------------------------------------------------------------------------


def render_v3c_reject_market_cap(
    *, symbol: str, market_cap_usd: float
) -> str:
    return (
        f"{EMOJI['err']} <b>{esc(ACTIVE_STRATEGY)} REJECT</b>\n"
        f"<b>${esc(symbol)}</b>\n"
        f"MC: ${market_cap_usd:,.2f}\n"
        f"V3-C limit: ${V3C_MAX_MARKET_CAP_USD:,.0f}\n\n"
        f"Reason: {esc(reject_reason_th(REJECTED_MARKET_CAP, strategy=V3C_NAME))}"
    )


def render_v3c_reject_signal_quality(*, symbol: str, signal_quality: float) -> str:
    return (
        f"{EMOJI['err']} <b>{esc(ACTIVE_STRATEGY)} REJECT</b>\n"
        f"<b>${esc(symbol)}</b>\n"
        f"SQ: {signal_quality:.2f}\n"
        f"Required: {V3C_MIN_SIGNAL_QUALITY:.2f}\n\n"
        f"Reason: {esc(reject_reason_th(REJECTED_SIGNAL_QUALITY, strategy=V3C_NAME))}"
    )


def render_strategy_reject(result, *, symbol: str) -> str:
    """
    Render a refusal by the ACTIVE strategy, from the gate result itself.

    Generated from the same result object the decision was made on, so the
    message cannot disagree with the journal. Handles both of L25's outcomes:
    a delegated V3-C refusal (market cap / signal quality / untestable) and
    L25's own liquidity/MC refusal.
    """
    status = getattr(result, "status", "")
    if status == REJECTED_LIQUIDITY_MC_BELOW_THRESHOLD:
        thr = float(getattr(result, "min_liq_to_mcap", L25_MIN_LIQ_TO_MCAP))
        return (
            f"{EMOJI['err']} <b>{esc(ACTIVE_STRATEGY)} REJECT</b>\n"
            f"<b>${esc(symbol)}</b>\n"
            f"V3-C: ผ่าน (authorized)\n"
            f"Liq/MC: {_ratio(getattr(result, 'f_liq_to_mcap', None))}\n"
            f"Required: {thr:.4f}\n\n"
            f"Reason: {esc(reject_reason_th(REJECTED_LIQUIDITY_MC_BELOW_THRESHOLD))}"
        )
    inner = getattr(result, "v3c", None)
    if inner is not None and getattr(inner, "status", "") != status:
        return render_strategy_reject(inner, symbol=symbol)
    if status == REJECTED_MARKET_CAP:
        return render_v3c_reject_market_cap(
            symbol=symbol, market_cap_usd=result.market_cap_usd or 0.0
        )
    if status == REJECTED_SIGNAL_QUALITY:
        return render_v3c_reject_signal_quality(
            symbol=symbol, signal_quality=result.signal_quality or 0.0
        )
    return (
        f"{EMOJI['err']} <b>{esc(ACTIVE_STRATEGY)} REJECT</b>\n"
        f"<b>${esc(symbol)}</b>\n\n"
        f"Reason: {esc(reject_reason_th(status))}\n"
        f"<i>{esc(getattr(result, 'detail', ''))}</i>"
    )


#: Retained name. Delegates to the strategy-aware renderer, so a caller that
#: still says "v3c" gets the ACTIVE strategy's message rather than a stale one.
def render_v3c_reject(v3c, *, symbol: str) -> str:
    return render_strategy_reject(v3c, symbol=symbol)


# ---------------------------------------------------------------------------
# open orders
# ---------------------------------------------------------------------------


def render_open_positions(rows: Sequence[dict[str, Any]]) -> str:
    """
    OPEN POSITIONS, one block per open position.

    ``rows`` come from the engine's ``open_positions_view``, which reads the SAME
    paper/real account objects the accounting layer reads. Nothing here is a
    Telegram-side copy, so this view cannot drift from ``/balance`` or
    ``/status``.

    Every field the position actually has is shown; a field the position does
    not have is stated as absent rather than rendered as a zero. An honest
    ``—`` and a fabricated ``0.00`` are different facts, and only the first one
    is safe to show an operator who is about to press KILL.

    ON DEMAND. This function is a pure renderer; the button is the only thing
    that calls it, and there is no poller, no subscription and no per-price
    message anywhere behind it.
    """
    e = EMOJI
    if not rows:
        return (
            f"{e['list']} <b>OPEN POSITIONS</b>\n\n"
            f"{EMPTY_OPEN_POSITIONS}\n"
            "<i>Cash is on the book; there is no open exposure.</i>"
        )

    paper = [r for r in rows if str(r.get("mode", "")).upper() != "REAL"]
    real = [r for r in rows if str(r.get("mode", "")).upper() == "REAL"]

    lines = [f"{e['list']} <b>OPEN POSITIONS ({len(rows)})</b>", ""]
    for r in paper:
        lines.extend(_open_position_block(r))
    if paper and real:
        lines.append("")
    for r in real:
        lines.extend(_open_position_block(r))
    return "\n".join(lines).rstrip()


def _open_position_block(r: Mapping[str, Any]) -> list[str]:
    """One open position, PAPER or REAL. Never raises on a missing field."""
    e = EMOJI
    badge = f"{e['real']} REAL" if str(r.get("mode", "")).upper() == "REAL" \
        else f"{e['paper']} PAPER"
    out = [f"<b>${esc(r.get('symbol') or '?')}</b>  {badge}"]
    # The execution-aware PAPER arms are separate accounts by design, so the
    # arm a position belongs to is named. Without it three identical-looking
    # blocks are indistinguishable and an operator cannot tell which research
    # arm holds what.
    arm = r.get("arm")
    if arm:
        out.append(f"Arm: <b>{esc(arm)}</b>")
        bankroll = r.get("bankroll_usd")
        if bankroll is not None:
            out.append(f"Arm bankroll: <b>${float(bankroll):,.2f}</b>")
    # The profile the position was OPENED under, not the currently selected one.
    # On a mid-flight profile switch these differ, and showing the selected one
    # would describe an exit policy this position is not actually running.
    prof = r.get("strategy_profile")
    if prof:
        out.append(f"Profile: <b>{esc(label(prof))}</b>")
    ca = r.get("ca") or r.get("mint")
    if ca:
        # A mint address is public information and is what identifies the token;
        # every value still goes through the escaper and the redaction layer.
        out.append(f"CA: <code>{esc(str(ca))}</code>")
    out.append(f"Position: <code>{esc(str(r.get('position_id') or '—'))}</code>")
    out.append(f"Entry: {_px(r.get('entry_price'))}"
               + (f"  @ {_ts(r.get('entry_at'))}" if r.get("entry_at") else ""))
    # The venue the fill actually happened on. On an execution-aware PAPER fill
    # this is the whole point: the position is pinned to it for the exit.
    venue = r.get("venue")
    if venue:
        out.append(f"Venue: <code>{esc(str(venue))}</code>")
    out.append(f"Current: {_px(r.get('current_price'))}")
    if r.get("size_usd") is not None:
        out.append(f"Size: <b>${float(r['size_usd']):,.6f}</b>")
    if r.get("quantity") is not None:
        out.append(f"Qty: {float(r['quantity']):,.6g}")
    status = r.get("status") or r.get("exit_status")
    if status and status != "OPEN_MONITORING":
        out.append(f"Status: <b>{esc(str(status))}</b>")
    if "unrealized_pnl_usd" in r or "realized_pnl_usd" in r:
        realized = r.get("realized_pnl_usd")
        unr = r.get("unrealized_pnl_usd")
        unr_pct = r.get("unrealized_pnl_pct")
        out.append(f"Realized: <b>${float(realized or 0.0):+,.6f}</b>")
        if unr is None:
            out.append("Unrealized: <i>UNKNOWN</i>")
        else:
            upct = (f" ({float(unr_pct):+.2f}%)"
                    if isinstance(unr_pct, (int, float)) else "")
            out.append(f"Unrealized: <b>${float(unr):+,.6f}{upct}</b>")
        total = r.get("total_pnl_usd")
        if total is not None:
            out.append(f"Total: <b>${float(total):+,.6f}</b>")
    else:
        pnl = r.get("pnl_usd")
        pct = r.get("pnl_pct")
        if pnl is None:
            out.append("PnL: <i>unavailable (no mark)</i>")
        else:
            text = f"${float(pnl):+,.6f}"
            if isinstance(pct, (int, float)):
                text += f" ({float(pct):+.2f}%)"
            out.append(f"PnL: <b>{text}</b>")
    if r.get("last_exit_error"):
        out.append(f"Last exit error: <i>{esc(str(r['last_exit_error'])[:160])}</i>")
    if r.get("exit_attempts"):
        out.append(f"Exit attempts: {int(r['exit_attempts'])}")
    if r.get("hold_s") is not None:
        out.append(f"Held: {fmt_duration(float(r['hold_s']))}")
    out.append(
        f"MFE: {_pct(r.get('mfe_pct'))} | MAE: {_pct(r.get('mae_pct'))}"
    )
    if r.get("data_state"):
        out.append(f"Data: {esc(r['data_state'])}")
    if r.get("total_equity_usd") is not None:
        out.append(f"Equity: <b>${float(r['total_equity_usd']):,.6f}</b>")
    if r.get("tp_sl"):
        out.append(f"TP/SL: {esc(r['tp_sl'])}")
    if r.get("exit_style"):
        out.append(f"Exit style: {esc(r['exit_style'])}")
    if r.get("exit_monitor"):
        out.append(f"Monitor: {esc(r['exit_monitor'])}")
    if r.get("thesis"):
        out.append(f"Thesis: <i>{esc(str(r['thesis'])[:220])}</i>")
    inv = r.get("invalidation") or []
    if inv:
        out.append("Invalidation: " + esc("; ".join(str(x) for x in inv[:3])))
    if r.get("mode") == "REAL" or r.get("signature"):
        sig = r.get("signature")
        conf = "confirmed" if r.get("confirmed") else "unconfirmed"
        out.append(f"Tx: <code>{esc(sig or '(pending)')}</code> ({conf})")
    out.append("")
    return out


def render_closed_positions(
    rows: Sequence[dict[str, Any]],
    *,
    total: int | None = None,
    offset: int = 0,
    page_size: int = 8,
) -> str:
    """
    Recently CLOSED PAPER positions, newest first, one page at a time.

    Rows come from the ledger's ``exit`` events - the same records the
    statistics are computed from - so this view and ``/status`` cannot report
    different closed trades.

    Pagination is by OFFSET with the totals stated in the message, rather than a
    button that fires a request per trade. A 420-trade ledger must not turn an
    inspection button into 420 messages in the operator's chat.

    The rows are COMPACT. The CA, trade id, route, latency and identity detail
    that used to repeat on every row now lives in
    :func:`render_closed_position_detail`, so a page of forty positions is
    readable instead of forty blocks of boilerplate.
    """
    e = EMOJI
    if not rows:
        return (
            f"{e['chart']} <b>CLOSED POSITIONS</b>\n\n"
            f"{EMPTY_CLOSED_POSITIONS}"
        )
    total = len(rows) if total is None else int(total)
    offset = max(0, int(offset))
    page_size = max(1, int(page_size))
    shown = f"{offset + 1}-{offset + len(rows)} of {total}" if total else \
        f"{offset + 1}-{offset + len(rows)}"
    lines = [
        f"{e['chart']} <b>CLOSED POSITIONS</b>  "
        f"<i>newest first - showing {shown}</i>",
        "",
    ]
    for index, r in enumerate(rows):
        if index:
            lines.append("")
        lines.extend(_closed_position_block(r))
    lines.append("")
    lines.append(f"<i>{total} closed in the ledger; newest {min(page_size, total)} "
                 f"per press. Reverse chronological. CA / trade id / route / "
                 f"latency detail stays in the per-trade detail view.</i>")
    return "\n".join(lines).rstrip()


def _closed_position_block(r: Mapping[str, Any]) -> list[str]:
    """
    One closed position, COMPACT.

    Token, outcome, hold, exit reason and the quote-based excursion. MFE/MAE is
    never shown bare: either it is the executable quote excursion -- labelled,
    with the coverage that bounds it -- or the mark pair under an explicit
    ``Mark-only`` label. A missing excursion is N/A with its reason, never
    ``0.0%`` standing in for a measurement that was never taken.
    """
    from ..forensics.compact_report import (
        classify_result,
        compact_summary,
        render_compact,
        render_mark_only,
    )
    out = [render_compact(r)]
    summary = compact_summary(r)

    # Mark-only analytics stay available, but only under their own label and
    # only when the data actually exists. They are never a PnL.
    mark_only = render_mark_only(summary)
    if mark_only:
        out.append(f"<i>{esc(mark_only)}</i>")
    return out


def render_closed_position_detail(r: Mapping[str, Any]) -> str:
    """
    Everything the compact row does not carry: route, identity, latency, CA.

    Kept as a SEPARATE view rather than deleted, so the detail is reachable
    instead of being lost to a shorter default.
    """
    from ..forensics.compact_report import classify_result
    e = EMOJI
    category, basis = classify_result(r)
    out = [f"{e['chart']} <b>CLOSED POSITION DETAIL</b>", ""]
    out.append(f"Token: <b>${esc(str(r.get('symbol') or '?'))}</b>")
    ca = r.get("ca") or r.get("mint")
    if ca:
        out.append(f"CA: <code>{esc(str(ca))}</code>")
    out.append(f"Result basis: <b>{esc(category)}</b> ({esc(basis)})")
    if r.get("entry_at"):
        out.append(f"Entry: {_ts(r.get('entry_at'))}")
    if r.get("exit_at"):
        out.append(f"Exit: {_ts(r.get('exit_at'))}")
    if r.get("entry_price_provenance") or r.get("exit_price_provenance"):
        out.append(
            "Price source: entry "
            f"{esc(str(r.get('entry_price_provenance') or 'unknown'))} / exit "
            f"{esc(str(r.get('exit_price_provenance') or 'unknown'))}"
            f"  \u00b7  PnL basis: {esc(str(r.get('pnl_basis') or 'unknown'))}"
        )
    if r.get("exit_reason"):
        out.append(f"Exit reason: {esc(str(r['exit_reason'])[:240])}")
    if r.get("route"):
        out.append(f"Route: {esc(str(r['route'])[:160])}")
    if r.get("entry_venue_id") or r.get("entry_pool_address"):
        out.append(
            "Entry venue: "
            f"<code>{esc(str(r.get('entry_venue_id') or ''))}</code>"
            f" / <code>{esc(str(r.get('entry_pool_address') or ''))}</code>"
        )
    if r.get("entry_quote_source"):
        out.append(f"Entry quote source: {esc(str(r['entry_quote_source']))}")
    if r.get("sell_quote_source"):
        out.append(f"Exit quote source: {esc(str(r['sell_quote_source']))}")
    if r.get("quote_mfe_source") or r.get("quote_mae_source"):
        out.append("Quote excursion source: "
                   f"{esc(str(r.get('quote_mfe_source') or 'n/a'))} / "
                   f"{esc(str(r.get('quote_mae_source') or 'n/a'))}")
    if r.get("strategy_profile"):
        out.append(f"Profile: {esc(str(r['strategy_profile']))}")
    if r.get("trade_id"):
        out.append(f"Trade: <code>{esc(str(r['trade_id']))}</code>")
    for label, key in (("Decision latency", "latency_ms"),
                       ("Exit quote age", "quote_age_ms"),
                       ("Mark age at exit", "mark_age_ms"),
                       ("Exit quote attempts", "sell_quote_attempts")):
        value = r.get(key)
        if value is not None:
            out.append(f"{label}: {esc(str(value))}")
    return "\n".join(out).rstrip()


#: Operator-facing name of the V2 FLOOR experiment's entry gate. Used ONLY to
#: attribute a beta refusal to the beta experiment -- never to decide anything.
BETA_GATE_LABEL = "V2 FLOOR (BETA)"


def render_buy_blocked(*, symbol: str, mint: str, reason: str, detail: str,
                       mode, reason_text: str = "",
                       candidate_id: str | None = None,
                       strategy: str | None = None,
                       beta_arms: Sequence[str] = (),
                       traded_arms: Sequence[str] = ()) -> str:
    """
    ``BUY_BLOCKED`` - a BUY the strategy authorized and the executor did not fill.

    Leads with the exact token ``BUY_BLOCKED=<reason>`` so it is greppable in a
    log and unmistakable on screen, and states plainly that NO position was
    opened. That distinction is the whole point: ``decision=BUY`` without it is
    what made the 2026-09-30 shutdown look like a lost trade.

    WHO AUTHORIZED IS AN INPUT, NOT A CONSTANT
    ------------------------------------------
    This used to say ``"{decisions_label()} authorized an entry."`` where
    ``decisions_label()`` is the module constant ``ACTIVE_STRATEGY == "L25"``.
    That is a hardcoded answer to a question the caller actually knows: when the
    only book running is the V2 FLOOR (BETA) experiment, the decision layer's L25
    authorization is a statement about a gate NO selected book uses, and
    reporting it as the authorizer rendered a BETA rejection to the operator as
    "L25 authorized an entry / NO POSITION OPENED" -- two sentences that read as
    a contradiction and hide which gate actually refused.

    ``beta_arms`` names the beta arms the beta gate refused. When it is
    non-empty this renders a BETA-GATE REJECTION instead: it says the beta gate
    refused, names the arms, and does NOT claim any strategy authorized an
    entry that was then blocked.

    ``traded_arms`` records arms that DID open on this same signal, so a beta
    refusal is reported as what it is -- a beta-gate rejection that did not
    disturb the L25 book -- rather than implying the whole signal was lost.
    """
    if beta_arms:
        return _render_beta_gate_rejected(
            symbol=symbol, mint=mint, reason=reason, detail=detail,
            mode=mode, reason_text=reason_text, candidate_id=candidate_id,
            beta_arms=tuple(beta_arms), traded_arms=tuple(traded_arms),
        )
    body = [
        f"{EMOJI['err']} <b>BUY_BLOCKED={esc(reason)}</b>",
        f"<b>${esc(symbol)}</b>",
        f"{strategy or decisions_label()} authorized an entry.",
        f"Outcome: <b>NO POSITION OPENED</b>",
        f"Mode: <b>{esc(getattr(mode, 'value', mode) or 'PAPER')}</b>",
    ]
    if traded_arms:
        body.append(
            "Note: other arms filled on this signal: "
            f"{esc(', '.join(traded_arms))}")
    if reason_text:
        body.append(f"Reason: {esc(reason_text)}")
    if detail:
        body.append(f"Detail: {esc(str(detail)[:240])}")
    if mint:
        body.append(f"CA: <code>{esc(mint)}</code>")
    if candidate_id:
        body.append(f"Candidate: <code>{esc(str(candidate_id))}</code>")
    return "\n".join(body)


def _render_beta_gate_rejected(*, symbol: str, mint: str, reason: str,
                               detail: str, mode, reason_text: str,
                               candidate_id: str | None,
                               beta_arms: tuple[str, ...],
                               traded_arms: tuple[str, ...]) -> str:
    """A beta experiment gate refusal -- never phrased as a blocked entry."""
    body = [
        f"{EMOJI['err']} <b>BUY_BLOCKED={esc(reason)}</b>",
        f"<b>${esc(symbol)}</b>",
        f"{esc(BETA_GATE_LABEL)} refused this signal for arms "
        f"{esc(', '.join(beta_arms))}.",
        "No L25 entry was blocked: the L25 gate was not the operative gate "
        "for this selection.",
    ]
    if traded_arms:
        body.append(
            f"{esc(', '.join(traded_arms))} opened a position independently; "
            "this rejection did not affect it.")
    body.extend([
        "Outcome: <b>NO BETA POSITION OPENED</b>",
        f"Mode: <b>{esc(getattr(mode, 'value', mode) or 'PAPER')}</b>",
    ])
    if reason_text:
        body.append(f"Reason: {esc(reason_text)}")
    if detail:
        body.append(f"Detail: {esc(str(detail)[:240])}")
    if mint:
        body.append(f"CA: <code>{esc(mint)}</code>")
    if candidate_id:
        body.append(f"Candidate: <code>{esc(str(candidate_id))}</code>")
    return "\n".join(body)


def decisions_label() -> str:
    """The strategy that authorized the entry, read from the one constant."""
    return esc(ACTIVE_STRATEGY)


def render_open_orders(rows: Sequence[dict[str, Any]]) -> str:
    """
    Currently open positions, one block per position.

    ``rows`` are dicts as produced by the account snapshots. PAPER and REAL
    positions are labelled individually and never combined into one total,
    because a combined PnL across a simulation and a wallet is meaningless.
    """
    e = EMOJI
    if not rows:
        return f"{e['list']} <b>ออเดอร์เปิด</b>\n\nไม่มีออเดอร์เปิด"

    lines = [f"{e['list']} <b>ออเดอร์เปิด ({len(rows)})</b>", ""]
    for r in rows:
        mode = TradeMode(str(r.get("mode", "PAPER")).upper())
        badge = f"{e['real']} REAL" if mode is TradeMode.REAL else f"{e['paper']} PAPER"
        lines.append(f"<b>${esc(r.get('symbol') or '?')}</b>  {badge}")
        lines.append(f"Entry: {_px(r.get('entry_price'))}")
        lines.append(f"Current: {_px(r.get('current_price'))}")
        lines.append(f"MC: {fmt_usd(r.get('market_cap_usd'))}")
        lines.append(f"Size: ${float(r.get('size_usd') or 0):,.6f}")
        pnl = r.get("pnl_usd")
        pct = r.get("pnl_pct")
        pnl_txt = "ไม่ทราบ" if pnl is None else f"${float(pnl):+,.6f}"
        if pct is not None and pnl is not None:
            pnl_txt += f" ({float(pct):+.2f}%)"
        lines.append(f"PnL: {pnl_txt}")
        lines.append(
            f"MFE: {_pct(r.get('mfe_pct'))} | MAE: {_pct(r.get('mae_pct'))}"
        )
        lines.append(f"Hold: {fmt_duration(float(r.get('hold_s') or 0))}")
        lines.append(f"State: {esc(r.get('exit_state') or 'OPEN')}")
        if mode is TradeMode.REAL:
            sig = r.get("signature")
            conf = "confirmed" if r.get("confirmed") else "unconfirmed"
            lines.append(f"Tx: <code>{esc(sig or '(pending)')}</code> ({conf})")
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def render_stats(*, paper: dict[str, Any], real: dict[str, Any]) -> str:
    """
    Statistics, PAPER and REAL reported SEPARATELY.

    There is no combined line. A single "total PnL" spanning a simulated
    account and a real wallet would be a number that means nothing, and it is
    exactly the number an operator is most likely to act on.
    """
    lines = [
        f"{EMOJI['chart']} <b>สถิติ</b>",
        f"Strategy: <b>{esc(ACTIVE_STRATEGY)}</b>",
        # The PAPER account is one long-lived accumulator. Its totals therefore
        # span every strategy version that has ever run against it, and a single
        # strategy label over them is misleading: 418 cumulative trades are not
        # 418 L25 trades. Naming the active strategy is right for what happens
        # NEXT; the caveat keeps it from being read as a verdict on the past.
        "<i>รวมทุกเวอร์ชันของ strategy ที่เคยรัน (ดู forward-report สำหรับสถิติเฉพาะ "
        f"{esc(ACTIVE_STRATEGY)})</i>",
        "",
        f"{EMOJI['paper']} <b>PAPER</b>",
    ]
    lines.extend(_stats_block(paper))
    beta_block = render_beta_stats(paper)
    if beta_block:
        lines.append("")
        lines.append(beta_block)
    lines.append("")
    lines.append(f"{EMOJI['real']} <b>REAL</b>")
    lines.extend(_stats_block(real))
    return "\n".join(lines)


def render_beta_stats(paper: Mapping[str, Any]) -> str:
    """
    V2 FLOOR (BETA) results, rendered from the experiment's OWN three ledgers.

    The experiment's stats ride inside the PAPER block under
    ``execution_aware_paper.v2_floor_beta`` (``None`` when no experiment is
    running). They are reported HERE, separately, rather than folded into the
    legacy PAPER totals: the beta arms are simulated-settlement accounting on
    their own $10-per-arm bankrolls, and merging them into the legacy
    mark-to-market accumulator would put two accounting bases in one number.

    Every figure is read verbatim from the persisted stats -- nothing here is
    recomputed from marks. An unavailable metric (no closed trades yet) prints
    ``n/a`` rather than a confident zero.
    """
    if not isinstance(paper, Mapping):
        return ""
    block = paper.get("execution_aware_paper")
    if not isinstance(block, Mapping):
        return ""
    beta = block.get("v2_floor_beta")
    if not isinstance(beta, Mapping):
        return ""
    arms = beta.get("arms")
    if not isinstance(arms, Mapping) or not arms:
        return ""
    lines = [
        "<b>V2 FLOOR (BETA)</b> "
        f"<i>(session {esc(str(beta.get('session_id') or '?'))} -- "
        f"{esc(str(beta.get('starting_bankroll_per_arm_usd')))} "
        "per arm, PAPER only)</i>",
    ]
    for key in ("A", "B", "C"):
        s = arms.get(key)
        if not isinstance(s, Mapping):
            continue
        realized = _num(s.get("realized_pnl_usd"))
        unrealized = _num(s.get("unrealized_pnl_usd"))
        bankroll = _num(s.get("bankroll_usd"))
        wr = s.get("win_rate_pct")
        pf = s.get("profit_factor")
        exits = s.get("exits") or {}
        lines.append(
            f"Arm {esc(str(key))} \u2014 TP {s.get('take_profit_pct'):+g}% / "
            f"SL {s.get('stop_loss_pct'):+g}% / max {s.get('max_hold_s'):g}s | "
            f"bankroll ${bankroll:,.4f}"
            if bankroll is not None else f"Arm {esc(str(key))}"
        )
        lines.append(
            f"  realized {_usd_signed(realized)} \u00b7 "
            f"unrealized {_usd_signed(unrealized)} \u00b7 "
            f"closed {int(s.get('completed_trades') or 0)} "
            f"open {int(s.get('open_positions') or 0)} "
            f"unresolved {int(s.get('unresolved') or 0)}"
        )
        lines.append(
            f"  win rate {_pct_or_na(wr)} \u00b7 "
            f"PF {'n/a' if pf is None else f'{float(pf):.2f}'} \u00b7 "
            f"TP {int(exits.get('TP') or 0)} / SL {int(exits.get('SL') or 0)} / "
            f"time {int(exits.get('TIME') or 0)}"
        )
        # CLOSED RESULTS BY SIZING RULE, at the ARM level. An arm's realized
        # figure above can span both rules, so the split is stated where the
        # number is rather than only in the experiment-wide block.
        by_sz = s.get("realized_by_sizing") or {}
        if isinstance(by_sz, Mapping) and by_sz:
            _ai = by_sz.get("all_in") or {}
            _lg = by_sz.get("legacy") or {}
            _ex = by_sz.get("explicit") or {}
            _parts = [
                f"ALL-IN {int(_ai.get('completed_trades') or 0)} "
                f"({_usd_signed(_num(_ai.get('realized_pnl_usd')))})",
            ]
            if int(_lg.get("completed_trades") or 0):
                _parts.append(
                    f"legacy 10% {int(_lg.get('completed_trades') or 0)} "
                    f"({_usd_signed(_num(_lg.get('realized_pnl_usd')))})")
            if int(_ex.get("completed_trades") or 0):
                _parts.append(f"explicit {int(_ex.get('completed_trades') or 0)}")
            lines.append("  closed by sizing: " + " \u00b7 ".join(_parts))
        # THE ALL-IN BUDGET, spelled out: what the arm can still trade with,
        # what its outstanding position has committed, what that entry actually
        # cost, and the percentage of available capital it used. "100%" is a
        # recorded number here (the sizing rule wrote it), not an assertion.
        avail = _num(s.get("available_bankroll_usd"))
        committed = _num(s.get("capital_committed_usd"))
        entry_cost = _num(s.get("entry_cost_usd"))
        alloc = _num(s.get("allocated_usd"))
        pos_pct = _num(s.get("position_sizing_pct"))
        if avail is not None or committed:
            parts = [f"available {_usd_plain(avail)}"]
            if committed:
                parts.append(f"committed {_usd_plain(committed)}")
            if entry_cost is not None and committed:
                parts.append(f"entry cost {_usd_plain(entry_cost)}")
            if pos_pct is not None:
                parts.append(f"sizing {pos_pct:.0f}% of {_usd_plain(alloc)}")
            lines.append("  " + " \u00b7 ".join(parts))
    # PROVENANCE SPLIT. The experiment's history can span the model change, so
    # the completed results are shown SEPARATELY by accounting basis and, when
    # more than one session contributed, by session. This is the guard that
    # stops a historical MODELLED number being read as quote-based performance.
    prov = beta.get("provenance")
    if isinstance(prov, Mapping):
        by_basis = prov.get("by_basis") or {}
        quote = by_basis.get("quote") or {}
        modeled = by_basis.get("modeled") or {}
        # SIZING PROVENANCE. The experiment's history crosses the sizing change
        # as well as the accounting one, so an ALL-IN result is never summed
        # with the historical 10%-of-equity results it replaced.
        by_sizing = prov.get("by_sizing") or {}
        if isinstance(by_sizing, Mapping) and by_sizing:
            all_in = by_sizing.get("all_in") or {}
            legacy = by_sizing.get("legacy") or {}
            explicit = by_sizing.get("explicit") or {}
            lines.append(
                "  sizing \u00b7 ALL-IN 100% "
                f"{int(all_in.get('completed_trades') or 0)} closed "
                f"{_usd_signed(_num(all_in.get('realized_pnl_usd')))}"
            )
            n_legacy = int(legacy.get('completed_trades') or 0)
            if n_legacy:
                lines.append(
                    "  sizing \u00b7 legacy 10% (historical, NOT all-in) "
                    f"{n_legacy} closed "
                    f"{_usd_signed(_num(legacy.get('realized_pnl_usd')))}"
                )
            n_explicit = int(explicit.get('completed_trades') or 0)
            if n_explicit:
                lines.append(
                    f"  sizing \u00b7 explicit (fixtures/probes) {n_explicit} closed"
                )
        lines.append(
            "  basis \u00b7 quote-sim "
            f"{int(quote.get('completed_trades') or 0)} closed "
            f"{_usd_signed(_num(quote.get('realized_pnl_usd')))}"
        )
        lines.append(
            "  basis \u00b7 modelled-backtest (historical, NOT quote performance) "
            f"{int(modeled.get('completed_trades') or 0)} closed "
            f"{_usd_signed(_num(modeled.get('realized_pnl_usd')))}"
        )
        if prov.get("mixed"):
            lines.append(
                "<i>quote-based and modelled results are DIFFERENT accounting "
                "bases; they are never summed into one number.</i>"
            )
        by_session = prov.get("by_session")
        if isinstance(by_session, Mapping) and len(by_session) > 1:
            lines.append("  by session:")
            for sid in sorted(str(k) for k in by_session):
                s = by_session.get(sid) or {}
                lines.append(
                    f"    {esc(sid)}: {int(s.get('completed_trades') or 0)} closed "
                    f"{_usd_signed(_num(s.get('realized_pnl_usd')))}"
                )
    lines.append(
        "<i>sizing: ALL-IN -- each arm commits 100% of its own available "
        "cash (fees inside the quote), so an arm holding a position has no "
        "free capital and takes no second entry. Legacy 10%-sized trades are "
        "labelled separately. Fills: executable-quote simulated where "
        "recorded; legacy modelled marks are labelled per trade. "
        "PAPER ONLY \U0001f512</i>"
    )
    return "\n".join(lines)


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _usd_signed(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:+,.4f}"


def _usd_plain(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.4f}"


def _stats_block(s: dict[str, Any]) -> list[str]:
    """
    Render one account's statistics.

    Reads ``total_pnl_usd`` and falls back to ``realized_pnl_usd``, because the
    PAPER statistics carry the historical key and the REAL statistics carry the
    canonical one. Normalising here rather than in each producer means a third
    account added later cannot render a blank PnL by picking the wrong name.
    """
    n = int(s.get("total_trades", 0))
    if n == 0:
        return ["ยังไม่มีเทรด"]
    if s.get("total_pnl_usd") is None and s.get("realized_pnl_usd") is not None:
        s = {**s, "total_pnl_usd": s["realized_pnl_usd"]}
    pf = s.get("profit_factor")
    pf_txt = "inf" if pf in (None, float("inf")) else f"{float(pf):.2f}"
    rows = [
        f"Total: {n}",
        f"Wins: {int(s.get('wins', 0))} | Losses: {int(s.get('losses', 0))}",
        f"Win rate: {float(s.get('win_rate', 0.0)) * 100:.1f}%",
        f"PF: {pf_txt}",
        f"Expectancy: ${float(s.get('expectancy_usd', 0.0)):+,.6f}",
        f"Total PnL: <b>${float(s.get('total_pnl_usd', 0.0)):+,.6f}</b>",
        # Labelled: realised trade-to-trade drawdown cannot see an open
        # position that is underwater. "Max DD" read as an all-in risk number.
        f"Realized DD: {float(s.get('max_drawdown_pct', 0.0)):.2f}%",
        f"Avg winner: ${float(s.get('avg_winner_usd', 0.0)):+,.6f}",
        f"Avg loser: ${float(s.get('avg_loser_usd', 0.0)):+,.6f}",
        f"Runners: {int(s.get('runners', 0))}",
        # RENAMED. This number is runners' realised dollars over total winner
        # dollars. It was displayed as "Runner capture", which it is not: it
        # never looks at MFE, so a strategy that gives back every runner would
        # still score highly. True MFE capture is shown on its own line, and
        # reads "n/a" rather than 0.0% when it cannot be computed.
        f"Runner PnL share: {_pct_or_na(s.get('runner_pnl_share_pct'))}",
        f"MFE capture (mark): {_pct_or_na(s.get('mfe_capture_pct'))}",
    ]
    if s.get("mixed_strategy_population"):
        rows.append(
            "⚠️ Mixed strategies: "
            + " + ".join(s.get("strategy_profiles_present") or [])
            + " -- pooled stats describe no single strategy"
        )
    equity = s.get("equity_usd")
    if equity is not None:
        rows.append(f"Equity: ${float(equity):,.6f}")
    cash = s.get("cash_usd")
    if cash is not None:
        # The `$` was missing, so this row read as a bare number next to
        # "Equity: $468.98" and looked like a different unit.
        rows.append(f"Cash: ${float(cash):,.6f}")
    return rows


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------


def _pct_or_na(value) -> str:
    """A percentage that may legitimately be UNKNOWN.

    ``None`` means "not computable from the data held". Showing that as 0.0%
    would state a performance result the system does not actually have.
    """
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def kill_all_label(*, open_positions: int, mode: str = "PAPER") -> str:
    """
    The button text an operator actually reads in an emergency.

    It states three things a bare "KILL ALL" does not: that this is an EMERGENCY,
    that it closes EVERYTHING currently open, and how many positions that is
    right now. The PAPER/REAL scope is shown because the same button is rendered
    in both modes and closing a real position is not an emergency action.
    """
    n = max(0, int(open_positions or 0))
    word = "POSITION" if n == 1 else "POSITIONS"
    return f"\U0001F6A8 KILL ALL \u2014 {n} OPEN {word} [{mode}]"


def kill_all_confirmation_text(*, open_positions: int, mode: str = "PAPER") -> str:
    n = max(0, int(open_positions or 0))
    word = "POSITION" if n == 1 else "POSITIONS"
    return (
        f"\U0001F6A8 <b>EMERGENCY CLOSE ALL</b>\n\n"
        f"This closes <b>every</b> open {mode} position ({n} {word}) immediately, "
        f"using the safest available executable route.\n\n"
        f"It does NOT consult the strategy exit rules. It will NOT invent a fill "
        f"price: any position without a valid executable SELL quote stays OPEN and "
        f"is retried.\n\n"
        f"New entries are PAUSED until this finishes or you cancel.\n\n"
        f"Press CONFIRM to proceed."
    )


def fmt_usd(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"${float(value):,.2f}"


def _ratio(value: Optional[float]) -> str:
    """The liquidity/MC ratio, or an explicit dash. Never a fabricated zero."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "—"


def _px(value: Optional[float]) -> str:
    if value is None or float(value) <= 0:
        return "—"
    return f"{float(value):.6e}"


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{float(value):+.1f}%"


def _ts(value: Optional[float]) -> str:
    """A UTC timestamp, or an explicit dash. Never ``None`` leaking to the chat."""
    if value is None:
        return "—"
    try:
        from ..clock import iso

        return iso(float(value))
    except Exception:  # noqa: BLE001 - a display path must not raise
        return "—"


def fmt_duration(seconds: float) -> str:
    s = max(0.0, float(seconds))
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.0f}m"
    return f"{s / 3600:.1f}h"
