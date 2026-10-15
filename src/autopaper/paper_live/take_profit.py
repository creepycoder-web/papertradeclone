"""
Selectable take-profit levels: a SETTING, not a strategy arm.

WHY THIS EXISTS
---------------
Take-profit used to be reachable only by selecting the ``L25_TP100`` ARM -- a
whole strategy arm whose only difference from ``L25_CURRENT`` was a hard-coded
``+100%`` profit-taking exit. That was wrong twice:

* it multiplied the strategy vocabulary to change one number, so the arm list
  the operator sees grew for a parameter, not a strategy; and
* ``+100%`` became unreachable as a *value* -- there was no way to take profit
  at a different level without editing a profile name.

So take-profit becomes a SETTING, not an arm, and ``+100%`` becomes one value
among several. The ``L25_TP100`` arm/profile is RETIRED from the selector: its
entire content was this one number, which is now selectable directly.

THE INVARIANT THIS MODULE EXISTS TO PROTECT
-------------------------------------------
A position's take-profit is a SNAPSHOT taken when the position is opened. It is
never read from the global setting afterwards. Changing the setting must not
silently rewrite the exit policy of something already open -- the same invariant
the production profile already obeys.

WHAT THE VALUE MEANS
====================
``take_profit_pct`` is a percentage gain measured on the POSITION'S ENTRY FILL
PRICE, which is itself derived from an executable BUY quote (see
``PaperPosition.entry_fill.effective_price_usd``). It is deliberately NOT a mark
price: the aggregator observation is a different book from the one the position
cost, and a TP evaluated on a mark can fire (or fail to fire) on a trade the
operator never had. ``EntryExitReport`` records the basis that evaluated.

A quoted EXECUTABLE NET return -- ``(net_sell_proceeds - all_in_entry_cost) /
all_in_entry_cost`` -- is the strictly better basis, and is what the paper
monitoring loop and the hold-path telemetry already produce for reporting. When
that path is used, ``take_profit_trigger_documented`` says so explicitly.

A quote is an EXECUTION ESTIMATE, never a guarantee: a submitted transaction can
fill at a worse output. Nothing here claims otherwise, and no trigger here is a
claim that the target is an assured net return.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class TakeProfitSelection:
    """
    One operator selection: a level, or explicitly "no take-profit".

    A dataclass rather than a bare float so "unset" and "+100%" are different
    states that no falsy check can confuse.
    """

    pct: Optional[float] = None

    @property
    def enabled(self) -> bool:
        return self.pct is not None and float(self.pct) > 0.0

    @property
    def label(self) -> str:
        return label(self.pct)

    @property
    def status_line(self) -> str:
        return status_line(self.pct)

    def as_dict(self) -> dict[str, Any]:
        return {
            "take_profit_pct": self.pct,
            "label": self.label,
            "enabled": self.enabled,
            "presets": list(SELECTABLE_TAKE_PROFITS),
            "bounds_pct": [TAKE_PROFIT_MIN_PCT, TAKE_PROFIT_MAX_PCT],
        }

#: The presets offered in the selector, in the order shown to the operator.
#:
#: ``100.0`` is retained because it was the only reachable value before (via
#: ``L25_TP100``) and removing it would be a silent capability loss. It appears
#: here as a NUMBER, not as a strategy.
SELECTABLE_TAKE_PROFITS: tuple[float, ...] = (
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
    50.0,
    100.0,
)

#: Bounds for a custom value. Wide enough for any sane target, narrow enough
#: that a typo cannot install "take profit at +0.01%" (fires at entry) or a
#: level so far out that it can never be reached.
TAKE_PROFIT_MIN_PCT: float = 0.5
TAKE_PROFIT_MAX_PCT: float = 1000.0

#: The sentinel meaning "no take-profit".
#:
#: Distinct from ``0.0``: ``None`` is "off", and ``0.0`` is NOT selectable
#: because a TP of zero fires on the first tick -- a disable, expressed as an
#: immediate exit. The OFF button writes ``None``.
NO_TAKE_PROFIT: Optional[float] = None

#: Persisted alongside the strategy selection, in its own right.
TAKE_PROFIT_LABEL_OFF = "OFF"

#: The label used when the selected level cannot be read.
TAKE_PROFIT_LABEL_INVALID = "?"


class TakeProfitError(ValueError):
    """An unrecognised or out-of-range take-profit level."""


def is_selectable(pct: Any) -> bool:
    try:
        value = float(pct)
    except (TypeError, ValueError):
        return False
    return any(abs(value - v) < 1e-9 for v in SELECTABLE_TAKE_PROFITS)


def clamp_or_reject(value: Any) -> float:
    """
    Bound-check a candidate level, raising with the exact reason.

    Used for a CUSTOM value. Presets are already inside the bounds by
    construction, so this is the custom-input gate.
    """
    if value is None:
        raise TakeProfitError("take-profit value is missing")
    if isinstance(value, str):
        text = value.strip().rstrip("%").strip()
        if text == "":
            raise TakeProfitError("take-profit value is empty")
        value = text
    if value is True or value is False:
        raise TakeProfitError("take-profit must be a number, not a boolean")
    try:
        pct = float(value)
    except (TypeError, ValueError) as exc:
        raise TakeProfitError(f"unrecognised take-profit {value!r}") from exc
    if pct != pct:
        raise TakeProfitError("take-profit must not be NaN")
    if not pct > 0:
        raise TakeProfitError(
            f"take-profit must be a POSITIVE percentage, got {pct!r}; use OFF "
            f"to disable it")
    if pct < TAKE_PROFIT_MIN_PCT:
        raise TakeProfitError(
            f"take-profit {pct:g}% is below the {TAKE_PROFIT_MIN_PCT:g}% floor")
    if pct > TAKE_PROFIT_MAX_PCT:
        raise TakeProfitError(
            f"take-profit {pct:g}% is above the {TAKE_PROFIT_MAX_PCT:g}% ceiling")
    return pct


def normalise(raw: object) -> Optional[float]:
    """
    Resolve a persisted/selected take-profit into a float, or ``None`` for OFF.

    FAILS CLOSED. An unprescribed level RAISES rather than snapping to a
    neighbour: an operator who selected ``+37%`` must not silently receive
    ``+50%``, and a corrupted settings file must not silently disarm a take
    profit that was protecting real (or simulated) capital.

    A preset is accepted exactly; a CUSTOM value inside the bounds is accepted
    exactly, so an operator who chose something outside the presets keeps it.
    """
    if raw is None:
        return NO_TAKE_PROFIT
    if isinstance(raw, str):
        text = raw.strip().rstrip("%").strip()
        if text == "" or text.upper() in {
            "OFF", "NONE", "DISABLED", "NULL",
        }:
            return NO_TAKE_PROFIT
        raw = text
    if raw is True or raw is False:
        raise TakeProfitError("take-profit must be a number, not a boolean")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise TakeProfitError(f"unrecognised take-profit {raw!r}") from exc
    if value != value:
        raise TakeProfitError("take-profit must not be NaN")
    if not value > 0:
        raise TakeProfitError(
            f"take-profit must be a POSITIVE percentage, got {value!r}; use OFF "
            f"to disable it")
    if is_selectable(value):
        return value
    return clamp_or_reject(value)


def label(pct: Optional[float]) -> str:
    """The operator-facing label. ``OFF`` is explicit, never blank."""
    if pct is None:
        return TAKE_PROFIT_LABEL_OFF
    try:
        return f"+{float(pct):g}%"
    except (TypeError, ValueError):
        return TAKE_PROFIT_LABEL_INVALID


def status_line(pct: Optional[float]) -> str:
    """
    The exact line the operator reads on every status surface.

    Assembled HERE so the button, the menu, the status card and the journal
    cannot drift into disagreeing about what the current level is.
    """
    return f"L25 TAKE PROFIT: {label(pct)}"


def for_position(position: object) -> Optional[float]:
    """
    The take-profit THIS POSITION was opened with.

    Reads the SNAPSHOT, never the global setting. Falls back to the historical
    ``L25_TP100`` profile ONLY for a position that predates the snapshot field,
    so an already-open legacy position keeps the exit it was opened under
    instead of silently losing its take-profit.

    An unreadable stored snapshot fails closed to OFF rather than to a guess: a
    number nobody can verify must not be allowed to trigger a real exit.
    """
    snap = getattr(position, "take_profit_pct", None)
    if snap is not None:
        try:
            return normalise(snap)
        except TakeProfitError:
            return NO_TAKE_PROFIT
    profile = (getattr(position, "production_profile", None)
               or getattr(position, "strategy_profile", None))
    if profile == "L25_TP100":
        return 100.0
    return NO_TAKE_PROFIT


def trigger_token(pct: Optional[float]) -> str:
    """
    The exit trigger recorded when the take-profit fires.

    ``tp100`` is retained for a +100% level because historical ledgers already
    use it. Every other level gets its own token, so a +10% exit can never be
    mistaken for a +100% one.
    """
    if pct is None:
        return "tp100"
    try:
        value = float(pct)
    except (TypeError, ValueError):
        return "tp100"
    if abs(value - 100.0) < 1e-9:
        return "tp100"
    return f"tp{value:g}"


# ---------------------------------------------------------------------------
# documented trigger basis (what the percentage is computed from)
# ---------------------------------------------------------------------------

#: Basis labels, so a recorded trigger never has to be interpreted.
BASIS_ENTRY_FILL_PRICE = "ENTRY_FILL_PRICE"
BASIS_EXECUTABLE_NET_SELL_QUOTE = "EXECUTABLE_NET_SELL_QUOTE"


@dataclass(frozen=True)
class TakeProfitTrigger:
    """
    The outcome of evaluating a take-profit against one observation.

    ``basis`` is the load-bearing field: it says what the percentage was
    computed from, so a trigger recorded from a fill can never later be read
    as an executable outcome and vice versa.
    """

    fired: bool
    basis: str
    return_pct: float
    target_pct: Optional[float]
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "fired": self.fired,
            "basis": self.basis,
            "return_pct": self.return_pct,
            "target_pct": self.target_pct,
            "detail": self.detail,
        }


def net_quote_return_pct(*, net_proceeds: Any,
                         all_in_entry_cost: Any) -> Optional[float]:
    """
    ``(estimated_net_sell_proceeds - all_in_entry_cost) / all_in_entry_cost``.

    Both arguments must be in the SAME currency. Returns ``None`` when either
    is missing, zero or non-numeric, because a return computed against a cost of
    zero is not a return, and inventing one is exactly the class of fabricated
    number this module exists to prevent.

    Fees already embedded in the quote (venue fee, price impact, and the SOL
    routed to a hop) are NOT added again here: the quoted output amount is
    already net of them, and double-counting would understate every trigger.
    """
    try:
        proceeds = float(net_proceeds)
        cost = float(all_in_entry_cost)
    except (TypeError, ValueError):
        return None
    if cost != cost or proceeds != proceeds:
        return None
    if cost <= 0 or proceeds < 0:
        return None
    return (proceeds - cost) / cost * 100.0


def take_profit_trigger_documented(
    *,
    target_pct: Optional[float],
    entry_fill_price: Any = None,
    price: Any = None,
    net_sell_proceeds: Any = None,
    all_in_entry_cost: Any = None,
) -> TakeProfitTrigger:
    """
    Evaluate the take-profit for one observation, on the best honest basis.

    THE EXECUTABLE BASIS IS PREFERRED, AND WHY
    ------------------------------------------
    When an executable SELL quote for the exact held quantity is supplied, the
    trigger is evaluated on the estimated NET LIQUIDATION return. That is the
    number that accounts for the venue's fees, price impact and slippage already
    embedded in the quote, and it is therefore what a take-profit has to mean.
    It is NOT a guarantee: a quote is an execution estimate, not a fill.

    When no such quote exists, the trigger falls back to the ENTRY-FILL price
    basis -- the simulated fill the position was opened at versus the current
    price -- and ``basis`` records that explicitly. The aggregator mark is NEVER
    used here by itself.

    A target of ``None`` or ``<= 0`` reports ``fired=False`` with the target
    unset, rather than firing.
    """
    if target_pct is None:
        return TakeProfitTrigger(
            False, BASIS_ENTRY_FILL_PRICE, 0.0, None,
            "no take-profit configured (OFF)")
    try:
        target = float(target_pct)
    except (TypeError, ValueError):
        return TakeProfitTrigger(
            False, BASIS_ENTRY_FILL_PRICE, 0.0, None,
            f"unreadable take-profit target {target_pct!r}")
    if not target > 0:
        return TakeProfitTrigger(
            False, BASIS_ENTRY_FILL_PRICE, 0.0, None,
            f"take-profit target {target!r} is not a positive percentage")

    if net_sell_proceeds is not None and all_in_entry_cost is not None:
        pct = net_quote_return_pct(
            net_proceeds=net_sell_proceeds,
            all_in_entry_cost=all_in_entry_cost)
        if pct is not None:
            return TakeProfitTrigger(
                pct >= target, BASIS_EXECUTABLE_NET_SELL_QUOTE, pct, target,
                f"estimated net SELL proceeds {float(net_sell_proceeds):.6f} "
                f"against all-in entry cost {float(all_in_entry_cost):.6f} "
                f"= {pct:+.2f}% vs {target:+.0f}% target")
        return TakeProfitTrigger(
            False, BASIS_EXECUTABLE_NET_SELL_QUOTE, 0.0, target,
            "no usable executable SELL net proceeds for this quantity")

    try:
        basis_price = float(entry_fill_price)
        current = float(price)
    except (TypeError, ValueError):
        return TakeProfitTrigger(
            False, BASIS_ENTRY_FILL_PRICE, 0.0, target,
            "entry fill price or current price unavailable")
    if not basis_price > 0 or not current > 0:
        return TakeProfitTrigger(
            False, BASIS_ENTRY_FILL_PRICE, 0.0, target,
            "entry fill price or current price is not positive")
    pct = (current - basis_price) / basis_price * 100.0
    return TakeProfitTrigger(
        pct >= target, BASIS_ENTRY_FILL_PRICE, pct, target,
        f"entry-fill price basis: {current:.10g} vs {basis_price:.10g} = "
        f"{pct:+.2f}% vs {target:+.0f}% target (no executable SELL quote was "
        f"supplied)")


__all__ = [
    "NO_TAKE_PROFIT",
    "SELECTABLE_TAKE_PROFITS",
    "TAKE_PROFIT_LABEL_OFF",
    "TAKE_PROFIT_LABEL_INVALID",
    "TAKE_PROFIT_MIN_PCT",
    "TAKE_PROFIT_MAX_PCT",
    "TakeProfitError",
    "TakeProfitSelection",
    "TakeProfitTrigger",
    "BASIS_ENTRY_FILL_PRICE",
    "BASIS_EXECUTABLE_NET_SELL_QUOTE",
    "clamp_or_reject",
    "for_position",
    "is_selectable",
    "label",
    "net_quote_return_pct",
    "normalise",
    "status_line",
    "take_profit_trigger_documented",
    "trigger_token",
]
