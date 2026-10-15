"""
The ONE canonical runtime state.

There is exactly one of these per process, and every surface - the Telegram UI,
the CLI, the journal, the executor - reads it. A second source of truth is how
a bot ends up displaying "REAL" while the executor is a simulator, so this
object is the only thing that is allowed to answer "what mode are we in".

STATE MACHINE
-------------

    OFF  --start-->  PAPER  --confirm REAL + preflight-->  REAL_ARMED
     ^                 |                                   |
     |                 |                                   |
     +--stop----------+                                   |
     |                                                     |
     +---------------------turn off REAL--------------------+

    any state --STOP-->              STOPPED
    any state --EMERGENCY STOP-->    EMERGENCY_STOP
    any state --select PAPER-->      PAPER   (REAL becomes disarmed)

REAL ARMING
-----------

``REAL_ARMED`` is reachable ONLY through :meth:`RuntimeState.arm_real`, which:

1. requires an explicit operator confirmation token, and
2. requires a preflight result that passed.

Neither an environment variable nor a config value can produce REAL_ARMED. A
process that reads ``REAL_TRADING_ENABLED=true`` at startup still boots into
PAPER. This is the difference between "REAL is supported" and "REAL can start
itself", and only the first is true here.

EMERGENCY STOP
--------------

Emergency stop is a latch, not a flag. It is persisted by the caller and
re-read on startup, so it survives a restart, and it blocks new entries without
touching V3-C, the journal, or any open position.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .logging_setup import log
from .strategy_profile import (
    DEFAULT_PROFILE,
    StrategyProfile,
    StrategyProfileError,
    label,
    max_hold_s,
    parse_profile,
)


class RunState(str, Enum):
    """Lifecycle of the trading loop."""

    OFF = "OFF"
    """Bot is not processing candidates. The startup state."""

    PAPER = "PAPER"
    """Running, entries authorized against the simulated account."""

    REAL_ARMED = "REAL_ARMED"
    """Running, entries authorized against real funds. Operator-armed only."""

    STOPPED = "STOPPED"
    """Loop was started and then stopped by the operator. No new entries."""

    EMERGENCY_STOP = "EMERGENCY_STOP"
    """Latched halt. No new entries until manually cleared."""


class TradeMode(str, Enum):
    """Which account an entry is authorized against."""

    PAPER = "PAPER"
    REAL = "REAL"


#: States in which the trading loop is actively processing candidates.
RUNNING_STATES = frozenset({RunState.PAPER, RunState.REAL_ARMED})

#: States that imply entries are authorized against real funds. A state in here
#: without :attr:`RuntimeState.real_armed` is impossible and must be downgraded.
REAL_ARMED_STATES = frozenset({RunState.REAL_ARMED})


class ModeError(RuntimeError):
    """An illegal transition was requested. The state is left untouched."""


#: How long a REAL confirmation token stays usable.
#:
#: A token is minted by pressing 🔴 REAL and consumed by pressing ✅ ยืนยัน REAL.
#: Without a deadline, a confirmation dialog left open on a phone could arm REAL
#: hours later with a tap the operator no longer connects to that button. Five
#: minutes is long enough to notice a prompt and respond, and short enough that
#: a stale button cannot authorize anything.
#:
#: Measured on a MONOTONIC clock, so a wall-clock adjustment (NTP, DST, a manual
#: change) can neither extend nor collapse a token's life.
REAL_CONFIRMATION_TTL_S: float = 300.0


def _now() -> float:
    return time.time()


def validate_strategy_selection(
    values: object, *, trade_mode: "TradeMode | str | None" = None,
) -> tuple[bool, str, tuple[str, ...]]:
    """
    The ONE validation rule for a strategy selection, shared by every caller.

    PAPER accepts one, two or three arms. REAL accepts EXACTLY ONE, and this is
    the layer that enforces it: the UI is not a security boundary, so a
    malformed config file, a stale state file, or a direct API call must be
    refused here exactly as a UI press would be.

    Returns ``(accepted, message, normalised_arms)``. An unknown name is
    REFUSED, never dropped -- silently ignoring an unrecognised strategy would
    let a caller believe an arm is running when it is not.
    """
    from .paper_live.arms import (
        AMARIH_ARMS,
        AMARIH_GROUP,
        BETA_ARMS,
        BETA_GROUP,
        arm_names,
        parse_arms,
    )

    if trade_mode is None:
        mode = TradeMode.PAPER
    elif isinstance(trade_mode, TradeMode):
        mode = trade_mode
    else:
        try:
            mode = TradeMode(str(getattr(trade_mode, "value", trade_mode)))
        except ValueError:
            mode = TradeMode.PAPER

    if isinstance(values, (str, bytes)):
        values = [values]
    requested = [str(v) for v in (values or ())]
    if not requested:
        return False, "no strategy selected; PAPER accepts 1-3 arms", ()
    try:
        specs = parse_arms(requested)
    except Exception as exc:  # noqa: BLE001 - the refusal IS the contract
        return False, str(exc), ()
    arms = tuple(s.name.value for s in specs)
    if len(arms) > 1 and mode is TradeMode.REAL:
        return False, (
            f"REAL is single-strategy: {len(arms)} strategies "
            f"({', '.join(arms)}) requested. REAL accepts exactly one, because "
            "a single real signal must never fan out into multiple real "
            "entries."), ()
    # ``known`` is every token that RESOLVES. The arm vocabulary, the beta GROUP
    # token, and the individual beta arm names -- because a beta arm token
    # resolves (to the whole group), so a selection that has already been
    # expanded into arm names by a toggle or restored from disk is not
    # "unknown", it is the group it came from. The Amarih group token and its
    # three arm tokens are known for the SAME reason: an Amarih arm token
    # resolves to the whole paired experiment via parse_arms, so a selection
    # already expanded to AMARIH_T30/T60/T120 by a toggle or restored from disk
    # is the Amarih experiment, not an unknown strategy.
    known = (set(arm_names()) | {BETA_GROUP}
             | {a.value for a in BETA_ARMS}
             | {AMARIH_GROUP}
             | {a.value for a in AMARIH_ARMS})
    unknown = [a for a in requested if a not in known]
    if unknown:
        return False, f"unknown strategy: {', '.join(sorted(set(unknown)))}", ()
    return True, f"{len(arms)} strategy arm(s): {', '.join(arms)}", arms


@dataclass
class RuntimeState:
    """
    The single authoritative runtime state.

    Mutating methods are the only way to change :attr:`run_state`. Direct
    assignment is deliberately possible (dataclass) but the engine never does
    it, and :meth:`snapshot` is what every consumer reads.
    """

    #: Loop lifecycle.
    run_state: RunState = RunState.OFF

    #: Latched emergency stop. Survives restart via the caller.
    emergency_stop: bool = False

    #: Whether the operator pressed "stop"; blocks new entries, keeps state.
    entries_blocked: bool = False

    #: The SELECTED strategy profile. Authoritative: the engine reads this to
    #: stamp newly opened positions, and the Telegram selector writes it. It is
    #: deliberately NOT how an open position decides its exit -- a position
    #: carries the profile it was opened under, so switching this never
    #: rewrites the exit policy of something already open.
    #:
    #: Selection is inert with respect to everything else: it does not start
    #: the runtime, arm REAL, clear an emergency stop or unblock entries. Those
    #: are separate, explicit operator actions.
    strategy_profile: str = DEFAULT_PROFILE.value
    #: The PAPER strategy arms the operator has selected.
    #:
    #: PAPER is MULTI-SELECT: each selected arm gets its own independent virtual
    #: bankroll, so one Telegram signal may open several independent simulated
    #: positions -- one per arm -- while still sharing a single executable quote
    #: snapshot. The default is EVERY arm, which is what the runtime executed
    #: before the selector existed, so this preserves behaviour rather than
    #: quietly narrowing it.
    #:
    #: REAL is SINGLE-SELECT and that is enforced in
    #: :func:`validate_strategy_selection`, not merely in the UI: a real entry
    #: must never be able to fan out into several real orders.
    selected_arms: tuple[str, ...] = ()
    #: The operator's selectable TAKE PROFIT, in percent. ``None`` = no
    #: take-profit.
    #:
    #: This is the GLOBAL setting. It is snapshotted onto each position when
    #: that position is opened, and an open position exits on its own snapshot
    #: -- never on this value. Changing this while a position is open must not
    #: rewrite the exit policy of something already in the book.
    #:
    #: Selection is inert with respect to everything else, exactly like
    #: ``strategy_profile``: it cannot start the runtime, arm REAL, clear an
    #: emergency stop or unblock entries.
    take_profit_pct: float | None = None

    @property
    def take_profit(self):
        """
        The take-profit selection as a validated, labelled object.

        An unreadable stored value resolves to OFF rather than raising, because
        a status surface that crashes is worse than one that says OFF -- and
        ``TakeProfitSelection`` will not let a falsy value be mistaken for a
        configured level.
        """
        from .paper_live.take_profit import TakeProfitSelection
        pct: float | None = None
        try:
            if self.take_profit_pct is not None:
                value = float(self.take_profit_pct)
                pct = value if value > 0 else None
        except (TypeError, ValueError):
            pct = None
        return TakeProfitSelection(pct)

    @property
    def take_profit_status_line(self) -> str:
        """The exact operator line: ``L25 TAKE PROFIT: +10%``."""
        return self.take_profit.status_line

    def set_take_profit(self, pct: float | None) -> float | None:
        """
        Set the take-profit, refusing an invalid value WITHOUT clearing the old.

        A refused value leaves the previous valid selection in force and raises,
        so a typo can never silently disable a configured take-profit.
        """
        from .paper_live.take_profit import normalise
        self.take_profit_pct = normalise(pct)
        return self.take_profit_pct

    @property
    def profile(self) -> StrategyProfile:
        """The selected profile, parsed. The default if somehow unset."""
        return parse_profile(self.strategy_profile or None)

    @property
    def effective_entry_gates(self) -> tuple[str, ...]:
        """
        The entry gates that authorize entries for the CURRENT selection.

        The single source for this fact, shared by the decision layer, the
        operator surfaces and the safety audit so they cannot disagree about
        which gate is in front. Derived from the arms -- see
        ``paper_live.arms.effective_entry_gates`` -- because the profile alone
        does not describe it: the Telegram selector writes the arms and never
        writes the profile in PAPER.
        """
        from .paper_live.arms import effective_entry_gates

        return effective_entry_gates(
            list(self.arms), profile=self.strategy_profile)

    @property
    def max_hold_s(self) -> Optional[float]:
        """Maximum hold for the selected profile, or ``None`` for no cap."""
        return max_hold_s(self.profile)

    #: Set only by arm_real(). Never by config, never by env, never by a
    #: constructor argument -- ``init=False`` makes that structural rather than
    #: a convention someone has to remember.
    #:
    #: ``init=False`` closes a real path found by the safety audit: the field was
    #: previously settable at construction, so ``RuntimeState(real_armed=True)``
    #: produced an armed state without ever passing through arm_real(). The
    #: engine never did this, so it was a latent footgun rather than a live
    #: exploit, but the gate is now unreachable except through the one audited
    #: method.
    real_armed: bool = field(default=False, init=False)

    #: Operator confirmation token. Generated on demand, single use, expiring.
    _pending_confirmation: Optional[str] = field(default=None, repr=False)
    #: Monotonic timestamp at which the pending token was minted. ``None``
    #: whenever there is no pending token.
    _pending_issued_at: Optional[float] = field(default=None, repr=False)
    #: Monotonic clock, injectable so token expiry is deterministically testable
    #: rather than requiring a test to sleep.
    _clock: Callable[[], float] = field(default=lambda: time.monotonic(),
                                        repr=False)

    #: Monotonic-ish audit trail of every accepted transition.
    history: list[dict[str, Any]] = field(default_factory=list)

    #: Reason the last transition was refused, for operator display.
    last_refusal: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Downgrade rather than upgrade. A REAL run_state without the armed
        # flag is an impossible state, and the only safe reading of it is PAPER.
        # Every real-ish run_state is covered: REAL_ARMED is currently the only
        # one, so the check is written against the set to stay correct if the
        # enum ever grows another.
        if self.run_state in REAL_ARMED_STATES and not self.real_armed:
            self.run_state = RunState.PAPER

    # -- views -------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self.run_state in RUNNING_STATES

    @property
    def trade_mode(self) -> TradeMode:
        """The account an entry would be authorized against right now."""
        if self.run_state is RunState.REAL_ARMED and self.real_armed:
            return TradeMode.REAL
        return TradeMode.PAPER

    @property
    def real_trading_enabled(self) -> bool:
        """True only in armed REAL. This is the only definition that counts."""
        return self.run_state is RunState.REAL_ARMED and self.real_armed

    def can_open_new_entries(self) -> tuple[bool, str]:
        """
        The single entry-authorization check. Every entry path calls this and
        nothing else decides for itself.
        """
        if self.emergency_stop:
            return False, "EMERGENCY STOP is latched; no new entries"
        if self.run_state is RunState.EMERGENCY_STOP:
            return False, "EMERGENCY STOP is latched; no new entries"
        if self.run_state in (RunState.OFF, RunState.STOPPED):
            return False, (
                "bot is not running; press start before any entry"
                if self.run_state is RunState.OFF
                else "bot was stopped; press start before any entry"
            )
        if self.entries_blocked:
            return False, "new entries are blocked by the stop command"
        return True, "entry authorization open"

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_state": self.run_state.value,
            "trade_mode": self.trade_mode.value,
            "running": self.running,
            "real_trading_enabled": self.real_trading_enabled,
            "real_armed": self.real_armed,
            "entries_blocked": self.entries_blocked,
            "emergency_stop": self.emergency_stop,
            "strategy_profile": self.profile.value,
            "strategy_label": label(self.profile),
            # The gates that actually AUTHORIZE an entry for the arms that are
            # selected. Carried here so every reader -- /status, the dashboard,
            # the safety audit -- can see that the profile and the operative
            # gate are related but not the same fact. They differ exactly when
            # the beta experiment is selected alongside an L25 arm, or when the
            # profile is its shipped default and the selection says otherwise.
            "effective_entry_gates": list(self.effective_entry_gates),
            "max_hold_s": self.max_hold_s,
            "last_refusal": self.last_refusal,
            "history": list(self.history),
        }

    def snapshot(self) -> dict[str, Any]:
        return self.as_dict()

    # -- internals ---------------------------------------------------------

    def _record(self, action: str, frm: RunState, to: RunState, detail: str = "") -> None:
        self.history.append(
            {
                "at": _now(),
                "action": action,
                "from": frm.value,
                "to": to.value,
                "detail": detail,
            }
        )
        # bounded, so a long-lived process cannot grow this without limit
        if len(self.history) > 500:
            del self.history[:-500]

    def _refuse(self, reason: str) -> None:
        self.last_refusal = reason
        self._record("refused", self.run_state, self.run_state, reason)

    # -- transitions -------------------------------------------------------

    # -- strategy SELECTION (multi for PAPER, single for REAL) ------------

    def select_strategies(self, values: object) -> tuple[bool, str]:
        """
        Set the PAPER strategy arms. Multi-select, validated, never defaulted.

        The list comes from the canonical registry
        (:data:`autopaper.paper_live.arms.ARMS`), so a strategy added there
        appears here without a second list to keep in sync.

        THE VALIDATION IS DELEGATED, NOT REPEATED
        -----------------------------------------
        This used to call ``parse_arms`` directly and then check ``if not
        specs``. That guard was UNREACHABLE, because ``parse_arms([])`` means
        "every registered arm" -- the explicit meaning the multi-arm research
        harness relies on. So ``select_strategies([])`` parsed to all three arms,
        passed the guard, and SELECTED ALL OF THEM, ``L25_TP100`` included, with
        no error anywhere. A caller passing an empty list -- an operator
        deselecting everything, a config layer with nothing to say, a bot
        restoring from an empty file -- therefore got every strategy rather than
        none, and the same guard could not catch it because the value it inspected
        was not the value that was requested.

        Routing through :func:`validate_strategy_selection` fixes it at the
        source: ONE rule decides what a valid selection is, the empty request is
        refused there, and this method no longer has a second opinion to keep in
        step. Multi-arm selection is untouched.
        """
        ok, message, arms = validate_strategy_selection(
            values, trade_mode=self.trade_mode)
        if not ok:
            self._refuse(message)
            return False, message

        previous = self.selected_arms
        self.selected_arms = arms
        # REAL reads its max-hold off ``strategy_profile`` when a position is
        # opened, so a REAL selection must drive that field too -- otherwise the
        # operator picks an arm and the real position silently keeps the old
        # profile's hold.
        if self.trade_mode is TradeMode.REAL and len(arms) == 1:
            from .paper_live.arms import ARMS as _ARMS, ResearchArm as _RA

            self.strategy_profile = _ARMS[_RA(arms[0])].production_profile.value
        self._record("select_strategies", self.run_state, self.run_state,
                     f"{list(previous) or ['<all>']} -> {list(arms)}")
        return True, f"{len(arms)} strategy arm(s) selected: {', '.join(arms)}"

    def toggle_strategy(self, name: str) -> tuple[bool, str]:
        """
        Flip one arm in the PAPER selection.

        In REAL this REPLACES rather than toggles, because REAL is a radio: the
        second selection is not "both", it is "the other one".

        ``V2_FLOOR_BETA`` IS THE GROUP, NOT AN ARM
        ------------------------------------------
        The beta experiment's selector token does not name an arm -- it names
        three arms together, and the experiment only means anything as a set (an
        A or B arm alone is not the comparison the experiment exists to make).
        So the group toggles as a unit: ON when every beta arm is selected, OFF
        when any is missing, and turning it on selects all three. Refusing it
        here as "unknown" would make the beta row in the operator's menu emit a
        callback the backend cannot act on -- a button that looks wired up and
        silently does nothing.
        """
        from .paper_live.arms import BETA_ARMS, is_beta_group

        if is_beta_group(name):
            beta_names = [a.value for a in BETA_ARMS]
            current = list(self.selected_arms)
            if set(beta_names) <= set(current):
                remaining = [a for a in current if a not in set(beta_names)]
                if not remaining:
                    message = (
                        "at least one strategy must stay selected; deselecting "
                        "the last one would leave the bot with nothing to run")
                    self._refuse(message)
                    return False, message
                return self.select_strategies(remaining)
            merged = current + [a for a in beta_names if a not in current]
            return self.select_strategies(merged)

        # AMARIH IS ALSO A GROUP, NOT AN ARM -- and it is ENABLED (opt-in),
        # unlike the disabled beta experiment. It toggles as a unit exactly like
        # beta does: ON selects all three paired-exit arms (T30/T60/T120), OFF
        # deselects all three. Selecting the group token OR any single arm token
        # resolves to the whole experiment via parse_arms, so a partial horizon
        # is never a reachable outcome.
        #
        # REAL REFUSES IT: Amarih is a PAPER research experiment whose three
        # paired arms share one entry and fan out into three simulated
        # positions. REAL accepts exactly ONE strategy because a single real
        # signal must never produce more than one real entry, so the group --
        # three arms by construction -- is refused here in REAL.
        from .paper_live.arms import AMARIH_ARMS, is_amarih_group

        if is_amarih_group(name):
            if self.trade_mode is TradeMode.REAL:
                message = (
                    "Amarih Research is a PAPER-only paired-exit experiment "
                    "(three arms off one signal). REAL accepts exactly one "
                    "strategy, so the Amarih experiment cannot be selected in "
                    "REAL.")
                self._refuse(message)
                return False, message
            amarih_names = [a.value for a in AMARIH_ARMS]
            current = list(self.selected_arms)
            if set(amarih_names) <= set(current):
                remaining = [a for a in current if a not in set(amarih_names)]
                if not remaining:
                    message = (
                        "at least one strategy must stay selected; deselecting "
                        "the last one would leave the bot with nothing to run")
                    self._refuse(message)
                    return False, message
                return self.select_strategies(remaining)
            merged = current + [a for a in amarih_names if a not in current]
            return self.select_strategies(merged)

        from .paper_live.arms import arm_by_name

        try:
            spec = arm_by_name(str(name))
        except Exception as exc:  # noqa: BLE001
            self._refuse(str(exc))
            return False, str(exc)
        current = list(self.selected_arms)
        if self.trade_mode is TradeMode.REAL:
            return self.select_strategies([spec.name.value])
        if spec.name.value in current:
            if len(current) == 1:
                message = (
                    "at least one strategy must stay selected; deselecting the "
                    "last one would leave the bot with nothing to run")
                self._refuse(message)
                return False, message
            current.remove(spec.name.value)
        else:
            current.append(spec.name.value)
        return self.select_strategies(current)

    @property
    def arms(self) -> tuple[str, ...]:
        """
        The selected arms, defaulting to the DEFAULT arm.

        THE DEFAULT USED TO BE EVERY REGISTERED ARM
        -------------------------------------------
        It resolved an empty ``selected_arms`` to ``arm_names()``, so ANY state
        that had never restored a selection -- a fresh install, a missing
        ``var/strategy_selection.json``, or a file that failed validation --
        silently traded ALL THREE arms, ``L25_TP100`` included. An operator who
        selected ``L25_CURRENT``, or who had selected nothing at all, got a bot
        opening trades as ``L25_TP100``, which is the exact reported symptom.

        The rationale that justified it ("an unset selection must not silently
        NARROW what trades") was about not changing behaviour for an operator
        who had deliberately selected several arms. That case still works: an
        explicit multi-selection is preserved verbatim. What it did not cover --
        and could not, because a default is not a selection -- was the operator
        who selected ONE arm, or none. Trading two strategies nobody asked for
        is not "not narrowing"; it is running a different strategy than the one
        selected.

        So the default is now :data:`DEFAULT_ARM`: one arm, named, and logged
        when it is used. Multi-arm PAPER research remains fully available by
        selecting several arms.
        """
        from .paper_live.arms import DEFAULT_ARM, arm_names

        if not self.selected_arms:
            log("WARN",
                "no strategy selection has been restored; defaulting to %s. "
                "Every registered arm (%s) is available but NONE of them is "
                "selected by default, because trading an unselected strategy is "
                "worse than trading the declared default.",
                DEFAULT_ARM.value, ", ".join(arm_names()))
            selected = (DEFAULT_ARM.value,)
        else:
            selected = tuple(self.selected_arms)
        if self.trade_mode is TradeMode.REAL and len(selected) > 1:
            # REAL is single-strategy. A multi-arm state can only exist from a
            # malformed or hand-edited state file, so it is CLAMPED here rather
            # than displayed as a valid multi-strategy REAL configuration.
            return selected[:1]
        return selected

    def select_strategy_profile(self, value: object) -> tuple[bool, str]:
        """
        Select the strategy profile. Operator-initiated, and deliberately INERT.

        What this does: set the profile newly opened positions will carry, and
        record the change in the audit trail.

        What this explicitly does NOT do, because a strategy selector is not a
        trading control:

          * it does not start the runtime (``run_state`` is untouched);
          * it does not arm REAL (``real_armed`` cannot be set from here at
            all -- it is ``init=False`` and only ``arm_real()`` sets it);
          * it does not clear an emergency stop or unblock entries;
          * it does not touch any open position, so a position already open
            keeps the exit policy it was opened under;
          * it broadcasts nothing and places no order.

        An unknown value is REFUSED, never defaulted. Returns
        ``(accepted, message)`` so the caller can show the operator the refusal
        verbatim rather than guessing.
        """
        try:
            profile = parse_profile(value)
        except StrategyProfileError as exc:
            self._refuse(str(exc))
            return False, str(exc)
        previous = self.profile
        self.strategy_profile = profile.value
        self._record("select_strategy_profile", self.run_state, self.run_state,
                     f"{previous.value} -> {profile.value}")
        if previous is profile:
            return True, (
                f"{label(profile)} is already selected; nothing changed. "
                f"Open positions keep the profile they were opened under."
            )
        return True, (
            f"Strategy profile set to {label(profile)}. Applies to newly opened "
            f"positions. Open positions keep the profile they were opened "
            f"under. This did not start the bot, arm REAL, or broadcast "
            f"anything."
        )

    def start(self) -> None:
        """
        [▶️ เริ่มบอท] - start the trading loop.

        Never enters REAL. If the operator had armed REAL, the arming survives
        a stop/start cycle, because disarming is an explicit act and a stop is
        not a disarming. If it was not armed, this enters PAPER.
        """
        if self.emergency_stop:
            self._refuse("EMERGENCY STOP is latched; clear it before starting")
            raise ModeError("EMERGENCY STOP is latched; cannot start")
        if self.running:
            self.last_refusal = None
            return
        frm = self.run_state
        self.run_state = RunState.REAL_ARMED if self.real_armed else RunState.PAPER
        self.entries_blocked = False
        self.last_refusal = None
        self._record("start", frm, self.run_state)

    def stop(self) -> None:
        """
        [⏹ หยุดบอท] - stop NEW entries.

        Does not alter open positions, does not delete state, does not disarm
        REAL. Monitoring of an open REAL position is the operator's to stop
        separately.
        """
        frm = self.run_state
        self.run_state = RunState.STOPPED
        self.entries_blocked = True
        self.last_refusal = None
        self._record("stop", frm, self.run_state)

    def select_paper(self) -> None:
        """
        [🟡 PAPER] - switch entry authorization to PAPER immediately.

        REAL becomes disarmed. This is the reliable way out of a live state and
        it is a single, non-reversible step: after it, no code path in this
        process can authorize a real entry without a fresh operator
        confirmation and a fresh preflight.
        """
        frm = self.run_state
        was_armed = self.real_armed
        self.real_armed = False
        self._clear_pending()
        if self.emergency_stop:
            self.run_state = RunState.EMERGENCY_STOP
        elif self.run_state is RunState.REAL_ARMED:
            self.run_state = RunState.PAPER
        self.last_refusal = None
        self._record(
            "select_paper",
            frm,
            self.run_state,
            "disarmed REAL" if was_armed else "already PAPER",
        )

    def request_real_confirmation(self) -> str:
        """
        [🔴 REAL] pressed - mint a single-use confirmation token.

        Returns the token to embed in the confirm/cancel buttons. Nothing is
        armed by this call; it exists so that arming requires a second, distinct
        operator action.
        """
        import secrets

        self._pending_confirmation = secrets.token_urlsafe(12)
        self._pending_issued_at = self._clock()
        self._record(
            "real_confirmation_requested", self.run_state, self.run_state, ""
        )
        return self._pending_confirmation

    def cancel_real_confirmation(self) -> None:
        self._clear_pending()
        self._record("real_confirmation_cancelled", self.run_state, self.run_state, "")

    def _clear_pending(self) -> None:
        """Consume the pending confirmation token, if any."""
        self._pending_confirmation = None
        self._pending_issued_at = None

    def confirmation_age_s(self) -> Optional[float]:
        """
        Age of the pending token in seconds, or ``None`` when there is none.

        Exposed so an operator-facing surface can show how long a confirmation
        prompt remains valid, and so expiry is observable rather than implicit.
        """
        if self._pending_confirmation is None or self._pending_issued_at is None:
            return None
        return max(0.0, self._clock() - self._pending_issued_at)

    def confirmation_expired(self) -> bool:
        """Is the pending token past its deadline? No token is not 'expired'."""
        age = self.confirmation_age_s()
        if age is None:
            return False
        return age > REAL_CONFIRMATION_TTL_S

    def arm_real(
        self,
        *,
        confirmation: Optional[str],
        preflight_ok: bool,
        preflight_detail: str = "",
    ) -> None:
        """
        [✅ ยืนยัน REAL] - the ONLY transition into REAL_ARMED.

        Requires both a valid single-use confirmation token and a passing
        preflight. There is deliberately no third option: no env override, no
        config flag, no "force" argument. A caller that has neither refuses and
        leaves the state exactly as it was.
        """
        frm = self.run_state
        expected = self._pending_confirmation
        if not expected or not confirmation or confirmation != expected:
            self._refuse(
                "REAL arming refused: no valid Telegram confirmation token"
            )
            raise ModeError(
                "REAL arming refused: no valid Telegram confirmation token"
            )
        if self.confirmation_expired():
            # Fail CLOSED on an expired token and consume it, so a stale button
            # cannot be retried. The operator must press 🔴 REAL again.
            age = self.confirmation_age_s()
            self._clear_pending()
            self._refuse(
                "REAL arming refused: the confirmation token expired after "
                f"{age:.0f}s (limit {REAL_CONFIRMATION_TTL_S:.0f}s)"
            )
            raise ModeError(
                "REAL arming refused: the confirmation token expired"
            )
        if not preflight_ok:
            self._clear_pending()
            self._refuse(
                f"REAL arming refused: preflight failed ({preflight_detail})"
            )
            raise ModeError(
                f"REAL arming refused: preflight failed ({preflight_detail})"
            )
        if self.emergency_stop:
            self._clear_pending()
            self._refuse("REAL arming refused: EMERGENCY STOP is latched")
            raise ModeError("REAL arming refused: EMERGENCY STOP is latched")

        # Consumed. A token authorizes one arming and no more.
        self._clear_pending()
        self.real_armed = True
        if self.run_state in (RunState.OFF, RunState.STOPPED):
            # Arming from a stopped state arms the capability but does not
            # start the loop. The operator still has to press start, and until
            # they do, can_open_new_entries() refuses anyway.
            self._record("real_armed", frm, self.run_state, preflight_detail)
            self._record("armed_while_not_running", frm, self.run_state)
            return
        # Running: PAPER becomes REAL_ARMED. A real entry may now happen.
        self.run_state = RunState.REAL_ARMED
        self.last_refusal = None
        self._record("real_armed", frm, self.run_state, preflight_detail)

    def turn_off_real(self) -> None:
        """
        [🔴 ปิด REAL] - disable new real entries.

        PAPER remains available and any open REAL position remains under
        monitoring. Nothing is forcibly closed; closing is a separate,
        explicitly-requested act.
        """
        frm = self.run_state
        was_armed = self.real_armed
        self.real_armed = False
        self._clear_pending()
        if self.run_state is RunState.REAL_ARMED:
            self.run_state = RunState.PAPER
        self.last_refusal = None
        self._record(
            "real_off",
            frm,
            self.run_state,
            "disarmed REAL" if was_armed else "was not armed",
        )

    def emergency_stop_now(self) -> None:
        """
        [🛑 EMERGENCY STOP] - immediately disable new entries.

        Latched. It does not change V3-C, does not erase journal state and does
        not touch open positions.
        """
        frm = self.run_state
        self.emergency_stop = True
        self.entries_blocked = True
        self.real_armed = False
        self._clear_pending()
        self.run_state = RunState.EMERGENCY_STOP
        self.last_refusal = None
        self._record("emergency_stop", frm, self.run_state)

    def clear_emergency_stop(self) -> None:
        """Operator explicitly clears the latch. Not exposed as a big button."""
        frm = self.run_state
        self.emergency_stop = False
        self.entries_blocked = False
        if self.run_state is RunState.EMERGENCY_STOP:
            self.run_state = RunState.OFF
        self.last_refusal = None
        self._record("emergency_cleared", frm, self.run_state)

    def restore_emergency_stop(self, latched: bool) -> None:
        """
        Re-apply a persisted emergency-stop latch at startup.

        A restart must not quietly un-halt a halted bot. This is called before
        anything else touches the state.
        """
        if not latched:
            return
        frm = self.run_state
        self.emergency_stop = True
        self.entries_blocked = True
        self.run_state = RunState.EMERGENCY_STOP
        self._record("emergency_restored", frm, self.run_state)


# ---------------------------------------------------------------------------
# display helpers
# ---------------------------------------------------------------------------

MODE_ICON: dict[TradeMode, str] = {
    TradeMode.PAPER: "🟡",
    TradeMode.REAL: "🔴",
}

STATE_LINE_TH: dict[RunState, str] = {
    RunState.OFF: "⏸ หยุดอยู่",
    RunState.PAPER: "🟢 ทำงานอยู่",
    RunState.REAL_ARMED: "🟢 ทำงานอยู่",
    RunState.STOPPED: "⏹ หยุดบอท",
    RunState.EMERGENCY_STOP: "🛑 EMERGENCY STOP",
}


def status_line(state: RuntimeState) -> str:
    """
    One-line mode rendering, shared by Telegram and CLI.

    ARMED IS NOT THE SAME AS RUNNING, AND THE LINE MUST SAY WHICH.

    ``real_armed`` and ``run_state`` are independent: REAL can stay authorized
    while the loop is stopped, because disarming is a separate act from
    stopping. Note that :attr:`trade_mode` is DERIVED and returns PAPER whenever
    the loop is not actively running in REAL_ARMED -- so a stopped-but-armed
    account previously rendered as ``PAPER (STOPPED)``, indistinguishable from an
    account with no real authority at all. An operator reading that could
    reasonably conclude no real order can happen, while REAL was still armed and
    would trade the instant the loop restarted. This is the reading the line
    must never permit, so ``real_armed`` is consulted FIRST, before the derived
    mode.

    The states an operator actually needs to tell apart:

    * ``REAL RUNNING`` -- armed AND processing; real orders are live now;
    * ``REAL ARMED, NOT RUNNING`` -- authorized but stopped; trades on restart;
    * ``PAPER RUNNING`` / ``PAPER OFF`` -- no real authority at all.
    """
    icon = MODE_ICON[state.trade_mode]
    if state.real_armed:
        if state.real_trading_enabled:
            return f"{icon} โหมด: <b>REAL RUNNING</b> (real orders are live)"
        return (
            f"{icon} โหมด: <b>REAL ARMED, NOT RUNNING</b>\n"
            f"⚠️ REAL is still AUTHORIZED. The loop is stopped, but any "
            f"restart will trade real funds."
        )
    running = state.run_state in (RunState.PAPER, RunState.REAL_ARMED)
    return f"{icon} โหมด: {'PAPER RUNNING' if running else 'PAPER OFF'}"
