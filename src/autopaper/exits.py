"""
Dynamic exit policy.

One policy, several behaviours. The exit style is chosen from the entry regime
(see ``analysis/routes.py``) and then evaluated against every observation.

The important structural property: every trigger below was declared in
``exit_plan`` at decision time, BEFORE the position existed. Post-entry
management may tighten a stop under these rules, but it cannot invent a reason
after the fact. ``assess`` returns a decision plus the trigger that fired, and
the trigger name is what goes into ``exit_reason``.

Exit styles
-----------

FAST_FAILURE      fragile / explosive entries. Tight stop, short fuse. These
                  give back fast and waiting for structure is how you turn a
                  -12% into a -45%.
MOMENTUM_FAILURE  early entries on a trailing stop from the running peak.
RUNNER            strong continuation. No fixed target. Arm at +25%, then give
                  back a fixed fraction of the peak. A hard floor well above
                  zero so a runner can still become a winner.
TIME_DECAY        weak confirmation. A setup that does not start within a
                  window is dead money; exit rather than hope.
FIXED_TARGET      only where the thesis is explicitly a target.
DISASTER_CAP      simulation boundary. NOT the strategy.
DATA_INVALID      the market stopped telling us anything usable. Exit to a
                  known state instead of guessing - the fill may be poor, and
                  that is recorded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

#: How far into the future an entry timestamp may sit and still be believed.
#: Absorbs ordinary clock jitter without letting a seconds/milliseconds mix
#: hide inside it.
_CLOCK_SKEW_TOLERANCE_S = 5.0


from .config import ExitConfig
from .logging_setup import log
from .memory import ResearchMemory
from .models import DataState, ExitStyle, Position, Route, Trade
from .strategy_profile import (
    MAX_HOLD_TRIGGER,
    max_hold_trigger,
    StrategyProfile,
    StrategyProfileError,
    label,
    max_hold_s,
    parse_profile,
    exit_bracket_for,
    V2F_TRIGGER_MAX_HOLD,
    V2F_TRIGGER_STOP_LOSS,
    V2F_TRIGGER_TAKE_PROFIT,
    V2F_EXIT_PRECEDENCE,
)


@dataclass(frozen=True)
class ExitDecision:
    action: str                      # "HOLD" | "EXIT"
    reason: str
    style: ExitStyle
    trigger: str | None = None
    runner_armed: bool | None = None
    tightened_to: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "style": self.style.value,
            "trigger": self.trigger,
            "runner_armed": self.runner_armed,
            "tightened_to": self.tightened_to,
        }


class ExitPolicy:
    def __init__(
        self,
        config: ExitConfig,
        *,
        memory: ResearchMemory | None = None,
    ) -> None:
        self.cfg = config
        self.memory = memory

    # -- main --------------------------------------------------------------

    def assess(
        self,
        position: Position,
        *,
        now: float,
        mark_price: float | None,
        data_state: DataState,
        momentum_30s_pct: float | None,
        structure_30s_pct: float | None,
        liquidity_change_pct: float | None,
        data_invalid_since: float | None = None,
    ) -> ExitDecision:
        style = position.exit_style

        # ---- hard-hold profiles: no price or style exits, ever -----------
        #
        # ``L25_MAX_HOLD`` is a TRUE hard-hold profile: BUY, hold, sell at the
        # cap. It is NOT "the L25 exit stack plus a 60s backstop". Every price
        # and style trigger below -- disaster cap, fast-failure stop, momentum
        # collapse, structure break, no-follow-through, runner give-back -- is
        # bypassed for this profile, so no strategy-level SELL can be generated
        # while the position is inside its hold window.
        #
        # Placed BEFORE the data-integrity gate deliberately. The cap is a TIME
        # action, not a price action: it does not need a mark, and a hard-hold
        # profile that could never reach its cap because the reference happened
        # to be unavailable would be a position that holds forever -- the exact
        # failure mode the cap exists to prevent. The bounded data-invalid path
        # below still governs every OTHER profile.
        #
        # Scoped to the one profile by identity. ``L25_CURRENT``, ``L25_60S`` and
        # every other profile fall through to the unchanged stack below, in the
        # unchanged order.
        hard_hold_cap = _hard_hold_cap_for(position)
        if hard_hold_cap is not None:
            return _hard_hold_decision(position, style, now, hard_hold_cap)

        # ---- data integrity ------------------------------------------
        #
        # STALE counts here too, not just MISSING/UNUSABLE. Acting on a price we
        # already know is out of date - panicking out of a position because a
        # print is 20s old - is exactly the kind of aggressive decision made on
        # information the agent does not actually have. The data-invalid timer
        # below bounds the wait, so nothing is held forever.
        if mark_price is None or data_state in (
            DataState.MISSING, DataState.UNUSABLE, DataState.STALE
        ):
            if data_invalid_since is not None and \
                    (now - data_invalid_since) >= self.cfg.data_invalid_exit_s:
                return ExitDecision(
                    "EXIT",
                    f"market data unusable for {now - data_invalid_since:.0f}s; "
                    "exiting to a known state rather than holding a blind mark",
                    ExitStyle.DATA_INVALID,
                    trigger="data_invalid",
                )
            return ExitDecision(
                "HOLD",
                f"DATA_UNCERTAIN ({data_state.value}); holding without an "
                "aggressive action until the invalid timer expires",
                style,
            )

        ret = (position.return_at(mark_price) or 0.0) * 100.0
        peak_ret = position.peak_return_pct
        # NB: `is not None`, not truthiness. A 0.0 timestamp is falsy and would
        # silently make every position look like it had just been entered,
        # disabling every time-based exit.
        # ``_held_seconds`` fails CLOSED on an unusable entry timestamp; see
        # its docstring for why an unknown hold must not read as "infinitely
        # old" and satisfy every elapsed-time trigger at once.
        held = _held_seconds(position, now)

        # ---- V2_FLOOR's own researched bracket -------------------------
        #
        # PLACED HERE, and the placement is the whole point:
        #
        #   * AFTER the data-integrity gate, because TP/SL are PRICE actions.
        #     A bracket that fired on a missing or stale mark would be selling
        #     on a number it does not have. (This is the OPPOSITE of
        #     ``L25_MAX_HOLD``'s time-only hard hold above, which correctly sits
        #     before the gate -- a cap needs no price, a bracket does.)
        #
        #   * BEFORE the disaster cap and EVERY style-specific trigger, so the
        #     researched rule is the WHOLE rule for this profile. The shared
        #     stack below cannot express it: its -12% stop and 75s
        #     no-follow-through exit both fire well before the researched -30%
        #     stop, and its momentum/structure exits are L25-specific rules that
        #     were never part of the V2 research. Running the researched entry
        #     with the shared exits would be a different strategy sharing a
        #     name.
        #
        # Identity-scoped on the POSITION's profile, exactly like the hard-hold
        # gate, so an open position keeps the rules it was opened under and an
        # operator switching profile mid-flight cannot rewrite it.
        bracket = _v2floor_bracket_for(position)
        if bracket is not None:
            decision = _v2floor_decision(
                position, now, held, ret, peak_ret, bracket
            )
            if decision is not None:
                return decision

        # ---- runner mode ------------------------------------------------
        if style is ExitStyle.RUNNER and not position.runner_mode:
            if ret >= self.cfg.runner_arm_pct and _healthy(
                momentum_30s_pct, liquidity_change_pct
            ):
                position.runner_mode = True
                position.runner_armed_at = now
                log("HOLD", "%s RUNNER armed at %+.1f%% after %.0fs",
                    position.symbol[:10], ret, held)
                return ExitDecision(
                    "HOLD",
                    f"runner mode armed at {ret:+.1f}%; no fixed target, trailing "
                    f"{self.cfg.runner_trail_pct:.0f}% from peak",
                    style, runner_armed=True,
                )

        # ---- disaster cap (simulation boundary) ------------------------
        if ret <= self.cfg.disaster_cap_pct:
            return ExitDecision(
                "EXIT",
                f"disaster cap {self.cfg.disaster_cap_pct:.0f}% breached at {ret:+.1f}% "
                "(simulation boundary, not the strategy stop)",
                ExitStyle.DISASTER_CAP,
                trigger="disaster_cap",
            )

        # ---- style-specific triggers -----------------------------------
        if style is ExitStyle.FAST_FAILURE:
            # ---- configurable take-profit (position SNAPSHOT) -------------
            #
            # The level comes from the POSITION, taken when it was opened, not
            # from the operator's current selection. An operator who changes the
            # take-profit setting while a position is open must not silently
            # rewrite the exit policy of something already in the book. This is
            # the same invariant the production profile already obeys.
            #
            # ``for_position`` also maps a legacy L25_TP100 profile to +100%, so
            # a position opened before the snapshot field existed keeps the
            # take-profit it was opened under rather than silently losing it.
            #
            # THE BASIS, STATED. ``ret`` is computed from ``position.return_at``
            # against the position's ENTRY FILL, not the aggregator mark: the
            # trigger reference this policy receives is the executable same-market
            # reference (see ``PaperLiveEngine._resolve_trigger_reference``), and
            # ``Position.return_at`` divides by ``fill_price`` which is the
            # simulated executable fill. An executable NET return basis is the
            # better measurement where a fresh SELL quote exists, and the hold
            # path already records it separately; the decision here stays on the
            # price basis so the trigger is unchanged from what ran before.
            #
            # The trigger token is derived from the level, so a +10% exit is
            # never mistaken for a +100% one in the journal. ``tp100`` is kept
            # verbatim at +100% for continuity with historical ledgers.
            _tp_pct = _take_profit_for(position)
            if _tp_pct is not None:
                if ret >= _tp_pct:
                    return ExitDecision(
                        "EXIT",
                        f"take-profit +{_tp_pct:g}% reached at {ret:+.1f}% "
                        f"after {held:.0f}s",
                        style, trigger=_tp_trigger_token(_tp_pct),
                    )

            if ret <= self.cfg.fast_failure_stop_pct:
                return ExitDecision(
                    "EXIT",
                    f"fast-failure stop {self.cfg.fast_failure_stop_pct:.0f}% hit at "
                    f"{ret:+.1f}% after {held:.0f}s",
                    style, trigger="fast_failure_stop",
                )
            if _collapse(momentum_30s_pct, self.cfg):
                return ExitDecision(
                    "EXIT",
                    f"momentum collapsed {momentum_30s_pct:+.1f}% over "
                    f"{self.cfg.momentum_collapse_lookback_s:.0f}s",
                    style, trigger="momentum_collapse",
                )
            if _structure_break(structure_30s_pct, self.cfg):
                return ExitDecision(
                    "EXIT",
                    f"structure broke {structure_30s_pct:+.1f}% over "
                    f"{self.cfg.structure_break_lookback_s:.0f}s",
                    style, trigger="structure_break",
                )
            if held >= self.cfg.fast_failure_window_s and ret < 3:
                return ExitDecision(
                    "EXIT",
                    f"no follow-through after {held:.0f}s (ret {ret:+.1f}%); "
                    "the fast thesis is spent",
                    style, trigger="no_follow_through",
                )

        if style is ExitStyle.RUNNER:
            if position.runner_mode:
                give_back = ret - peak_ret
                if give_back <= self.cfg.runner_trail_pct:
                    return ExitDecision(
                        "EXIT",
                        f"runner give-back {give_back:+.1f}% from the {peak_ret:+.1f}% "
                        f"peak after {held:.0f}s",
                        style, trigger="runner_giveback",
                    )
                if ret <= self.cfg.runner_hard_floor_pct:
                    return ExitDecision(
                        "EXIT",
                        f"runner floor {self.cfg.runner_hard_floor_pct:.0f}% lost",
                        style, trigger="runner_floor",
                    )
            else:
                # not yet a runner: behave like a momentum-failure trade
                if ret <= self.cfg.fast_failure_stop_pct and held < 180:
                    return ExitDecision(
                        "EXIT",
                        f"pre-runner stop {self.cfg.fast_failure_stop_pct:.0f}% hit at "
                        f"{ret:+.1f}%",
                        style, trigger="pre_runner_stop",
                    )
                if peak_ret >= 8 and ret - peak_ret <= self.cfg.momentum_trail_pct:
                    return ExitDecision(
                        "EXIT",
                        f"gave back {ret - peak_ret:+.1f}% from {peak_ret:+.1f}% "
                        "before arming as a runner",
                        style, trigger="pre_runner_giveback",
                    )
            if _collapse(momentum_30s_pct, self.cfg):
                return ExitDecision(
                    "EXIT", f"momentum collapsed {momentum_30s_pct:+.1f}%",
                    style, trigger="momentum_collapse",
                )
            if held >= self.cfg.time_decay_max_hold_s:
                return ExitDecision(
                    "EXIT",
                    f"max hold {self.cfg.time_decay_max_hold_s:.0f}s reached at {ret:+.1f}%",
                    style, trigger="max_hold",
                )

        if style is ExitStyle.MOMENTUM_FAILURE:
            if held >= self.cfg.momentum_min_hold_s and \
                    ret - peak_ret <= self.cfg.momentum_trail_pct:
                return ExitDecision(
                    "EXIT",
                    f"gave back {ret - peak_ret:+.1f}% from the {peak_ret:+.1f}% peak",
                    style, trigger="peak_giveback",
                )
            if _collapse(momentum_30s_pct, self.cfg):
                return ExitDecision(
                    "EXIT", f"momentum collapsed {momentum_30s_pct:+.1f}%",
                    style, trigger="momentum_collapse",
                )
            if _structure_break(structure_30s_pct, self.cfg):
                return ExitDecision(
                    "EXIT", f"structure broke {structure_30s_pct:+.1f}%",
                    style, trigger="structure_break",
                )
            if ret <= self.cfg.fast_failure_stop_pct * 1.5 and held >= self.cfg.momentum_min_hold_s:
                return ExitDecision(
                    "EXIT",
                    f"deep drawdown {ret:+.1f}% with no peak to trail",
                    style, trigger="deep_drawdown",
                )
            if held >= self.cfg.time_decay_max_hold_s:
                return ExitDecision(
                    "EXIT", f"max hold reached at {ret:+.1f}%",
                    style, trigger="max_hold",
                )

        if style is ExitStyle.TIME_DECAY:
            if peak_ret < self.cfg.time_decay_no_progress_band_pct and \
                    held >= self.cfg.time_decay_no_progress_s:
                return ExitDecision(
                    "EXIT",
                    f"no progress (peak {peak_ret:+.1f}%) after {held:.0f}s; "
                    "time decay",
                    style, trigger="time_decay_no_progress",
                )
            if held >= self.cfg.momentum_min_hold_s and \
                    ret - peak_ret <= self.cfg.momentum_trail_pct:
                return ExitDecision(
                    "EXIT", f"gave back {ret - peak_ret:+.1f}% from {peak_ret:+.1f}%",
                    style, trigger="peak_giveback",
                )
            if held >= self.cfg.time_decay_max_hold_s:
                return ExitDecision(
                    "EXIT", f"max hold {self.cfg.time_decay_max_hold_s:.0f}s reached at {ret:+.1f}%",
                    style, trigger="max_hold",
                )

        if style is ExitStyle.FIXED_TARGET and ret >= self.cfg.fixed_target_pct:
            return ExitDecision(
                "EXIT", f"fixed target +{self.cfg.fixed_target_pct:.0f}% reached",
                style, trigger="fixed_target",
            )

        # ---- default trail for any style not covered above ---------------
        if style in (ExitStyle.MOMENTUM_FAILURE, ExitStyle.TIME_DECAY) and \
                peak_ret > 0 and (ret - peak_ret) <= self.cfg.momentum_trail_pct * 1.5:
            return ExitDecision(
                "EXIT",
                f"wide give-back {ret - peak_ret:+.1f}% from {peak_ret:+.1f}% peak",
                style, trigger="wide_giveback",
            )

        # ---- maximum hold (profile-scoped, LAST in precedence) -----------
        #
        # Placed HERE, after every style-specific trigger, on purpose. A
        # maximum hold is an ADDITION to the existing exit policy, never a
        # replacement: if the stop, momentum, structure, dead-thesis or
        # disaster rule is true on the same mark, that rule fires and keeps its
        # own reason, exactly as it does today. The cap only closes a position
        # that would otherwise still be open.
        #
        # The value is read from the POSITION, not from the runtime's current
        # selection, so an operator switching profile mid-flight cannot rewrite
        # the exit policy of something already open. A position restored from
        # a pre-profile snapshot has no attribute, and is treated as the
        # current profile -- i.e. no cap.
        cap = _max_hold_for(position)
        if cap is not None and held >= cap:
            return ExitDecision(
                "EXIT",
                f"max hold {cap:.0f}s reached at {ret:+.1f}% after {held:.0f}s "
                f"({label_of(position)} profile; the existing rules did not "
                f"fire first)",
                style, trigger=max_hold_trigger(cap),
            )

        return ExitDecision(
            "HOLD",
            f"structure intact at {ret:+.1f}% (peak {peak_ret:+.1f}%, held {held:.0f}s, "
            f"{position.exit_hold_count + 1} hold checks)",
            style,
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _v2floor_bracket_for(position: Any) -> Optional[tuple[float, float]]:
    """
    ``(take_profit_pct, stop_loss_pct)`` if this POSITION was opened under a
    bracketed profile, else ``None`` to run the shared stack.

    ``None`` is the answer for every L25 profile and is what keeps them on the
    unchanged code path below. An unrecognised persisted profile also returns
    ``None``: failing to recognise a profile must not silently invent a stop --
    it must fall back to the pre-existing behaviour, which is what every
    position ran before profiles existed.

    For V2_FLOOR_BETA arms, the position carries its own per-arm bracket in
    ``beta_exit_bracket``, which takes precedence over the profile default.
    """
    # Beta experiment arms carry their own bracket
    beta_bracket = getattr(position, "beta_exit_bracket", None)
    if beta_bracket is not None:
        return beta_bracket
    return exit_bracket_for(getattr(position, "strategy_profile", None))


def _v2floor_decision(
    position: Any,
    now: float,
    held: float,
    ret: float,
    peak_ret: float,
    bracket: tuple[float, float],
) -> Optional[ExitDecision]:
    """
    The researched V2_FLOOR bracket, in the researched order:

        if ret <= -30%:       SELL
        elif ret >= +30%:     SELL
        elif held >= 90s:     SELL
        else:                 HOLD

    ``None`` means HOLD, which is returned as a real ``ExitDecision`` rather than
    ``None`` -- the ``None`` return is reserved for "this is not a bracketed
    position", so the caller can fall through to the shared stack.

    PRECEDENCE is the SL, then the TP, then the cap, and it is TOTAL: every
    possible (ret, held) pair resolves to exactly one of the three triggers.
    SL and TP are mutually exclusive on a single mark, so their relative order
    cannot change an outcome; the cap is genuinely last, matching the existing
    convention that a maximum hold is an addition to the policy and never a
    replacement for a real exit reason. A position sitting at +31% and 95s
    reports the take-profit, because that is the better answer to "why did this
    close", and a position at -31% and 200s reports the stop rather than a
    cap that had long since passed.

    The trigger tokens are the ``v2_*`` family so a V2 exit is never mistaken
    for an L25 exit in the journal.
    """
    take_profit, stop_loss = bracket
    name = label_of(position)
    style = ExitStyle.V2_BRACKET
    cap = _max_hold_for(position)
    suppressed = "the shared FAST_FAILURE stack (which was suppressed)"

    # 1. stop loss -- the researched -30%, evaluated FIRST
    if ret <= stop_loss:
        return ExitDecision(
            "EXIT",
            f"V2 FLOOR stop {stop_loss:+.0f}% hit at {ret:+.1f}% after "
            f"{held:.0f}s ({name}; researched bracket, not {suppressed})",
            style, trigger=V2F_TRIGGER_STOP_LOSS,
        )

    # 2. take profit -- the researched +30%
    if ret >= take_profit:
        return ExitDecision(
            "EXIT",
            f"V2 FLOOR take-profit {take_profit:+.0f}% reached at {ret:+.1f}% "
            f"after {held:.0f}s (peak {peak_ret:+.1f}%, {name}; researched "
            f"bracket, not {suppressed})",
            style, trigger=V2F_TRIGGER_TAKE_PROFIT,
        )

    # 3. max hold -- the researched 90s, LAST in precedence
    if cap is not None and held >= cap:
        return ExitDecision(
            "EXIT",
            f"V2 FLOOR max hold {cap:.0f}s reached at {ret:+.1f}% (peak "
            f"{peak_ret:+.1f}%, {name}; neither the {stop_loss:+.0f}% stop nor "
            f"the {take_profit:+.0f}% take-profit fired)",
            style, trigger=V2F_TRIGGER_MAX_HOLD,
        )

    cap_text = f"{cap:.0f}s" if cap is not None else "no cap"
    return ExitDecision(
        "HOLD",
        f"V2 FLOOR bracket holding at {ret:+.1f}% (peak {peak_ret:+.1f}%, held "
        f"{held:.0f}s/{cap_text}; between the {stop_loss:+.0f}% stop and the "
        f"{take_profit:+.0f}% take-profit, and inside the hold window)",
        style, trigger=None,
    )


def _take_profit_for(position: Any) -> Optional[float]:
    """
    The take-profit level this POSITION was opened with.

    Read from the position's snapshot, never from the global setting, so
    changing the operator's selection cannot rewrite the exit policy of an
    already-open position. Resolved lazily to keep this module free of an
    import cycle with ``paper_live``.
    """
    from .paper_live.take_profit import for_position

    return for_position(position)


def _tp_trigger_token(pct: Optional[float]) -> str:
    """
    The journal trigger for a take-profit exit at ``pct``.

    ``tp100`` is retained verbatim at +100% so every historical ledger that
    already recorded it still reads correctly. Every other level gets its own
    token, so a +10% exit can never be mistaken for a +100% one.
    """
    from .paper_live.take_profit import trigger_token

    return trigger_token(pct)


def _healthy(momentum_30s_pct: float | None, liquidity_change_pct: float | None) -> bool:
    if momentum_30s_pct is not None and momentum_30s_pct < -5:
        return False
    if liquidity_change_pct is not None and liquidity_change_pct < -25:
        return False
    return True


def _hard_hold_cap_for(position: Any) -> float | None:
    """
    The hold window of a TRUE hard-hold profile, or ``None`` for every other.

    Identity-scoped on purpose. Only ``L25_MAX_HOLD`` is a hard-hold profile, so
    ``L25_CURRENT``, ``L25_60S`` and any future profile are untouched by this
    gate unless they are explicitly declared hard-hold here.

    An unrecognised persisted profile is NOT treated as hard-hold: failing to
    recognise a profile must not silently disable a stop.
    """
    raw = getattr(position, "strategy_profile", None)
    try:
        # ``parse_profile`` FIRST: a position persists its profile as a plain
        # STRING, so an identity check against the enum would never match and
        # the hard-hold gate would silently never apply in production.
        if parse_profile(raw) is StrategyProfile.L25_MAX_HOLD:
            return max_hold_s(raw)
    except StrategyProfileError:
        log("ERROR",
            "position %s carries unrecognised strategy_profile=%r; not "
            "applying the hard-hold gate",
            getattr(position, "position_id", "?"), raw)
    return None


def _hard_hold_decision(
    position: Any, style: ExitStyle, now: float, cap: float
) -> ExitDecision:
    """
    The ONLY two outcomes a hard-hold profile can produce.

    Inside the window: HOLD, unconditionally -- no price trigger, no style
    trigger, whatever the reference says. At or past the cap: the single
    ``max_hold`` exit, which the caller then has to satisfy with a real
    executable SELL quote. Nothing here settles a position or invents a price.
    """
    held = _held_seconds(position, now)
    if held >= cap:
        return ExitDecision(
            "EXIT",
            f"max hold {cap:.0f}s reached after {held:.0f}s "
            f"({label_of(position)} profile; the existing price and style exits "
            f"are disabled for this profile and never fired)",
            style, trigger=max_hold_trigger(cap),
        )
    return ExitDecision(
        "HOLD",
        f"{label_of(position)} hard hold {held:.0f}s/{cap:.0f}s; price and style "
        f"exits are disabled for this profile",
        style, trigger=None,
    )


def _entry_epoch(position: Any, now: float) -> float:
    """
    The instant the position was opened, or ``now`` when that is unknowable.

    A hold timer is only meaningful against a real entry instant. A persisted
    position whose ``entry_timestamp`` is missing, non-finite, negative or in
    the future is not "infinitely old" -- it is UNKNOWN, and treating an
    unknown as epoch zero made ``now - 0`` about 56 years, which satisfied every
    ``held >= cap`` immediately. A position rebuilt from a record with no
    ``opened_at`` therefore exited on max hold the instant it was restored,
    while looking, in the operator's alert, like a position that had actually
    been held for its full cap.

    Returning ``now`` makes the hold 0s, which fails every elapsed-time
    threshold. That is the safe direction: an unevaluable hold must never
    manufacture an exit. Price triggers are unaffected -- they do not use this.

    NO EPOCH FLOOR IS APPLIED HERE, deliberately. ``exits`` is also driven with
    synthetic timestamps by the legacy path and its tests (``entry_timestamp``
    0.0 against a synthetic ``now``), so a real-world floor would reject a
    legitimate synthetic epoch. Credibility of a PERSISTED ``opened_at`` is
    enforced where that data actually enters the system --
    ``paper_live._position_from_dict`` and ``paper_live._production_position``.
    """
    entered = getattr(position, "entry_timestamp", None)
    try:
        value = float(entered)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        value = float("nan")
    if not math.isfinite(value) or value < 0 or value > now + _CLOCK_SKEW_TOLERANCE_S:
        log("ERROR",
            "position %s has an unusable entry_timestamp=%r; treating its hold "
            "as 0s rather than assuming it is infinitely old",
            getattr(position, "position_id", "?"), entered)
        return now
    return value


def _held_seconds(position: Any, now: float) -> float:
    """Seconds since the position was opened, floored at 0."""
    return max(0.0, now - _entry_epoch(position, now))


def _max_hold_for(position: Any) -> float | None:
    """
    The maximum hold this POSITION was opened under, or ``None`` for no cap.

    Read off the position rather than the runtime, which is what makes an open
    position immune to a later profile switch. ``getattr`` with a default keeps
    a position restored from a snapshot written before profiles existed working
    as the current profile instead of raising.

    For V2_FLOOR_BETA arms, the position carries its own max hold in
    ``beta_max_hold_s``, which takes precedence over the profile default.
    """
    # Beta experiment arms carry their own max hold
    beta_hold = getattr(position, "beta_max_hold_s", None)
    if beta_hold is not None:
        return beta_hold
    raw = getattr(position, "strategy_profile", None)
    try:
        return max_hold_s(raw)
    except StrategyProfileError:
        # An unrecognised persisted value must not silently disable a cap that
        # may have been safety-relevant. Refuse to cap rather than guess.
        log("ERROR",
            "position %s carries unrecognised strategy_profile=%r; "
            "treating as no max hold",
            getattr(position, "position_id", "?"), raw)
        return None


def label_of(position: Any) -> str:
    """Operator-facing name of the profile a position was opened under."""
    try:
        return label(getattr(position, "strategy_profile", None))
    except StrategyProfileError:
        return "L25 CURRENT"


def _collapse(momentum_30s_pct: float | None, cfg: ExitConfig) -> bool:
    return (
        momentum_30s_pct is not None
        and momentum_30s_pct <= cfg.momentum_collapse_drop_pct
    )


def _structure_break(structure_30s_pct: float | None, cfg: ExitConfig) -> bool:
    return (
        structure_30s_pct is not None
        and structure_30s_pct <= cfg.structure_break_pct
    )


def describe_plan(plan: dict[str, Any]) -> str:
    style = plan.get("style", "?")
    stop = plan.get("initial_stop_pct")
    invalidation = plan.get("invalidation_conditions") or []
    lines = [f"style={style} initial_stop={stop}%"]
    for inv in invalidation:
        lines.append(f"  - {inv}")
    return "\n".join(lines)
