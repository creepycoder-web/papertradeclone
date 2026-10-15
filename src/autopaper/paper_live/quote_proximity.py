"""
The QUOTE PROXIMITY BAND: when an executable SELL quote is worth asking for.

THE RULE
--------
Market monitoring and executable quoting are different questions and now cost
different things:

  * "what is the native market doing right now?" is answered by a pushed
    WebSocket update and a local calculation. Free.
  * "what can this position actually be sold for?" is answered by an executable
    SELL quote. Not free, and rate-limited.

Before this module the runtime asked the second question on every market update
and discarded the answer on every HOLD. This module makes the second question
conditional, and the condition is EXPLICIT rather than implied by a timer:

    far from every trigger   -> MONITOR_ONLY    no quote request at all
    near a trigger           -> QUOTE_ALLOWED   quote validation permitted
    a trigger is reached     -> QUOTE_REQUIRED  fresh executable quote now

WHAT THIS IS NOT
----------------
It is not a new exit rule, and it cannot create or suppress one. It gates
QUOTE REQUESTS, never decisions. Every threshold it watches is read from the
existing :class:`~autopaper.config.ExitConfig` and the position's own profile
bracket -- this module declares no number of its own except the proximity width,
and even that is a resource knob, not a strategy knob.

The safety property that makes it safe: when the band says ``MONITOR_ONLY``, the
exit path still runs and can still decide to exit, and the exit path requests its
own fresh quote for itself. So an exit can never be missed because a band was
narrow, and a band can never widen into a different strategy.

TUNING THE BAND
---------------
``band_pct`` is deliberately conservative and small: it must be wide enough that
a fast move crosses into it before it crosses the trigger, and narrow enough that
the far side of it covers the bulk of a position's life. The default is set for
that, not for PnL. Setting ``enabled=False`` restores unconditional quoting, so
the old behaviour remains reachable and the change is reversible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..config import ExitConfig
from ..strategy_profile import (
    StrategyProfile,
    StrategyProfileError,
    exit_bracket_for,
    max_hold_s,
    parse_profile,
)

#: Verdict tokens. Deliberately strings so they land in a log and a JSONL row
#: without a schema change.
MONITOR_ONLY = "MONITOR_ONLY"
QUOTE_ALLOWED = "QUOTE_ALLOWED"
QUOTE_REQUIRED = "QUOTE_REQUIRED"

#: The take-profit level the L25_TP100 profile triggers on. Read as a CONSTANT
#: here rather than duplicated: it is already declared, and read once, by
#: ``exits.ExitPolicy`` (``ret >= 100.0``). This module mirrors that value for
#: the PROXIMITY QUESTION ONLY -- whether to spend a quote near it -- and cannot
#: change whether the exit policy fires it.
TP100_PCT = 100.0

#: The raw enum token, kept next to the level so the two cannot drift apart
#: silently when either is edited.
TP100_NAME = StrategyProfile.L25_TP100.value


@dataclass(frozen=True)
class ProximityPolicy:
    """
    How near a trigger a position must be before a quote is permitted.

    ``band_pct`` is in PERCENTAGE POINTS of return, so ``3.0`` means "within 3
    points of the trigger". It is not a fraction of the trigger, because a
    fraction would silently mean something different for a -45% disaster cap
    than for a +100% take-profit.
    """

    enabled: bool = True
    band_pct: float = 3.0
    #: A band wider than this would start covering most of a position's life and
    #: stop being a filter. Clamped rather than trusted, because an accidental
    #: ``band_pct=100`` would silently restore unconditional quoting while
    #: appearing to be a tuning choice.
    max_band_pct: float = 10.0

    @property
    def effective_band_pct(self) -> float:
        if not self.enabled:
            return float("inf")
        try:
            band = float(self.band_pct)
        except (TypeError, ValueError):
            return 0.0
        if band < 0.0:
            return 0.0
        return min(band, self.max_band_pct)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "band_pct": self.band_pct,
            "max_band_pct": self.max_band_pct,
            "effective_band_pct": (None if self.enabled is False
                                   else self.effective_band_pct),
        }


#: The production default. Monitoring-only when far, quote when near, quote
#: immediately when triggered.
DEFAULT_PROXIMITY = ProximityPolicy()


@dataclass(frozen=True)
class ProximityVerdict:
    """The band decision plus the trigger it was measured against."""

    band: str
    #: The existing threshold closest to the current return, named. Empty when
    #: the position has no price trigger at all.
    nearest_trigger: str = ""
    nearest_threshold_pct: Optional[float] = None
    distance_pct: Optional[float] = None
    #: Why the band came out this way, in one line, for the log.
    reason: str = ""

    @property
    def permits_quote(self) -> bool:
        return self.band in (QUOTE_ALLOWED, QUOTE_REQUIRED)

    @property
    def requires_quote(self) -> bool:
        return self.band == QUOTE_REQUIRED

    def as_dict(self) -> dict[str, Any]:
        return {
            "quote_proximity_band": self.band,
            "quote_proximity_trigger": self.nearest_trigger,
            "quote_proximity_threshold_pct": self.nearest_threshold_pct,
            "quote_proximity_distance_pct": self.distance_pct,
            "quote_proximity_reason": self.reason,
        }


def _take_profit_for(position: Any) -> Optional[float]:
    """
    The take-profit THIS POSITION was opened with, read from its snapshot.

    Never the global selection: an operator who changes the take-profit while a
    position is open must not have that position suddenly watch a different
    trigger. A legacy L25_TP100 position maps to +100%.
    """
    from .take_profit import for_position

    return for_position(position)


def _profile_of(position: Any) -> str:
    """
    The resolved profile on whatever position object was handed in.

    The production ``Position`` (what ``ExitPolicy.assess`` receives) calls it
    ``strategy_profile``; the PAPER ``PaperPosition`` calls it
    ``production_profile``. Both spellings are accepted so the band can be
    evaluated against EITHER, and -- crucially -- so it can be evaluated against
    the very object the exit policy sees. Watching a different position's
    thresholds than the policy will fire is how a band would come to describe a
    strategy nobody is running.
    """
    raw = getattr(position, "strategy_profile", None)
    if not raw:
        raw = getattr(position, "production_profile", None)
    return str(raw or "")


def existing_price_triggers(
    position: Any, cfg: ExitConfig
) -> tuple[tuple[str, float], ...]:
    """
    The price thresholds a position can ACTUALLY reach, read from live config.

    Every value comes from the existing exit configuration or the position's own
    profile. Nothing is invented here, and a threshold the position's style does
    not evaluate is not watched -- so the band never reports "near" for a trigger
    that could not fire.

    The L25 live path runs ``FAST_FAILURE`` (see
    ``paper_live.engine.PaperLiveEngine._production_position``), so its price
    triggers are the fast-failure stop, the disaster cap, and the take-profit
    when the position carries one. Momentum, structure and give-back triggers
    need series the live path does not collect, so they are not watched rather
    than watched against a permanently ``None`` series.

    The take-profit level is read from the POSITION's snapshot, not from the
    operator's current selection: watching the global would report proximity to
    a trigger this position does not actually evaluate.
    """
    profile = _profile_of(position)
    triggers: list[tuple[str, float]] = []
    triggers.append(("fast_failure_stop", float(cfg.fast_failure_stop_pct)))
    triggers.append(("disaster_cap", float(cfg.disaster_cap_pct)))
    # Take-profit comes from the position, taken when it was opened. A legacy
    # L25_TP100 position maps to +100% so it keeps the trigger it was opened
    # under; a position with no take-profit watches none.
    _tp_pct = _take_profit_for(position)
    if _tp_pct is not None:
        triggers.append(("tp100", float(_tp_pct)))
    bracket = exit_bracket_for(profile) if profile else None
    if bracket is not None:
        take_profit, stop_loss = bracket
        triggers.append(("v2_take_profit", float(take_profit)))
        triggers.append(("v2_stop_loss", float(stop_loss)))
    return tuple(triggers)


def evaluate(
    position: Any,
    return_pct: Optional[float],
    *,
    cfg: ExitConfig,
    policy: ProximityPolicy = DEFAULT_PROXIMITY,
    held_s: Optional[float] = None,
) -> ProximityVerdict:
    """
    Where the position sits relative to the exits it can actually reach.

    ``return_pct`` is the CURRENT return on the same basis the exit policy uses
    -- the execution-native reference against the entry fill. Passing ``None``
    (the reference could not be proven) yields ``QUOTE_REQUIRED``: a position
    whose price cannot be trusted is exactly the case where the existing
    fail-closed data-integrity path may need to act, and refusing to quote there
    would be the wrong economy.
    """
    triggers = existing_price_triggers(position, cfg)

    # A time-based exit needs no proximity: once the clock is up, the existing
    # policy fires on that tick and the exit path quotes for itself.
    profile = _profile_of(position)
    cap = max_hold_s(profile) if profile else None
    if cap is not None and held_s is not None and held_s >= cap:
        return ProximityVerdict(
            band=QUOTE_REQUIRED,
            nearest_trigger="max_hold",
            nearest_threshold_pct=None,
            distance_pct=0.0,
            reason=(f"max-hold cap {cap:.0f}s reached at {held_s:.1f}s; the "
                    f"position is on a time exit, which always quotes"),
        )

    if not policy.enabled:
        return ProximityVerdict(
            band=QUOTE_ALLOWED,
            reason="proximity gating disabled; quoting on every evaluation")

    if return_pct is None:
        return ProximityVerdict(
            band=QUOTE_REQUIRED,
            reason=("the native reference is not currently provable, so a "
                    "quote is requested rather than assumed absent"),
        )

    band = policy.effective_band_pct
    best_name = ""
    best_threshold: Optional[float] = None
    best_distance: Optional[float] = None
    reached = False
    #: ANY trigger within the band, as opposed to any trigger reached. These are
    #: different questions and conflating them is what would make a position look
    #: "far" while sitting a point from its stop.
    near_any = False
    near_name = ""

    for name, threshold in triggers:
        distance = return_pct - threshold
        # "Near" is |distance| <= band, for EVERY trigger and in BOTH
        # directions. A trigger is only near when the return is close to it,
        # whichever side of it the return sits on -- and because the band is a
        # flat width in percentage points, an absolute distance cannot
        # accidentally read a +50% position as "near" a -45% cap: they are 95
        # points apart, which is far on any sane band.
        #
        # Direction only matters for REACHED, which is genuinely directional: a
        # take-profit is reached from below and a stop from above.
        near = abs(distance) <= band
        if threshold >= 0.0:
            is_reached = return_pct >= threshold
        else:
            is_reached = return_pct <= threshold
        if is_reached:
            reached = True
            if not near_any:
                near_any, near_name = True, name
        elif near and not near_any:
            near_any, near_name = True, name
        absolute = abs(distance)
        if best_distance is None or absolute < best_distance:
            best_distance = absolute
            best_name = name
            best_threshold = threshold

    if reached:
        return ProximityVerdict(
            band=QUOTE_REQUIRED,
            nearest_trigger=near_name or best_name,
            nearest_threshold_pct=best_threshold,
            distance_pct=best_distance,
            reason=(f"return {return_pct:+.2f}% is at or past an existing exit "
                    f"trigger (nearest {near_name or best_name} "
                    f"{best_threshold:+.0f}%)"),
        )
    if near_any:
        return ProximityVerdict(
            band=QUOTE_ALLOWED,
            nearest_trigger=near_name,
            nearest_threshold_pct=next(
                t for n, t in triggers if n == near_name),
            distance_pct=best_distance,
            reason=(f"return {return_pct:+.2f}% is within {band:.1f} points of "
                    f"the existing exit trigger {near_name}; quote validation "
                    f"is permitted"),
        )
    return ProximityVerdict(
        band=MONITOR_ONLY,
        nearest_trigger=best_name,
        nearest_threshold_pct=best_threshold,
        distance_pct=best_distance,
        reason=(f"return {return_pct:+.2f}% is {best_distance:.2f} points from "
                f"the nearest exit trigger ({best_name} "
                f"{best_threshold:+.0f}%), beyond the {band:.1f} point band"),
    )