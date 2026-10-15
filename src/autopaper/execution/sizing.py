"""
Paper position sizing.

Hard limits, enforced here, not left to the model:

* a single position may not exceed ``max_position_pct_equity`` (25%)
* total open exposure may not exceed ``max_total_exposure_pct_equity`` (50%)
* the number of simultaneous positions is capped
* one position per mint, and re-entry into a recently exited mint is OFF unless
  explicitly enabled
* cash can never go negative; a rejected size is not a "smaller trade"

If no size clears the floor, the correct answer is NO TRADE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from ..config import RiskLimits
from ..models import Position, PositionStatus


@dataclass(frozen=True)
class SizingDecision:
    approved: bool
    notional_usd: float
    pct_of_equity: float
    reason: str
    binding_constraint: str | None = None
    quantity: float | None = None
    # cash actually leaving the account = notional + entry fee
    cash_required_usd: float = 0.0
    fee_reserved_usd: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "approved": self.approved,
            "notional_usd": round(self.notional_usd, 4),
            "cash_required_usd": round(self.cash_required_usd, 4),
            "fee_reserved_usd": round(self.fee_reserved_usd, 4),
            "pct_of_equity": round(self.pct_of_equity, 2),
            "reason": self.reason,
            "binding_constraint": self.binding_constraint,
            "quantity": self.quantity,
        }


def compute_size(
    *,
    equity_usd: float,
    cash_usd: float,
    open_positions: Iterable[Position],
    target_pct: float,
    price: float,
    limits: RiskLimits,
    mint: str,
    recently_exited: dict[str, float] | None = None,
    now: float | None = None,
    entry_fee_bps: float = 85.0,
) -> SizingDecision:
    """
    Return the notional to risk, or an explicit refusal with the reason.

    ``target_pct`` comes from the route; this function can only ever make it
    SMALLER, never larger, and never exceeds the hard caps.

    IMPORTANT: every cap is enforced against the CASH the position will consume
    (``notional * (1 + fee)``), not against the notional alone. Capping the
    notional while ignoring the fee lets the real exposure creep above the limit
    by the fee amount on every single trade - a small, systematic, and entirely
    avoidable breach.
    """
    open_positions = list(open_positions)
    open_exposure = sum(
        p.cash_out_usd for p in open_positions if p.status is PositionStatus.OPEN
    )
    open_count = sum(1 for p in open_positions if p.status is PositionStatus.OPEN)

    if any(p.mint == mint and p.status is PositionStatus.OPEN for p in open_positions):
        return SizingDecision(False, 0.0, 0.0, "already holding this mint", "duplicate_mint")
    if open_count >= limits.max_simultaneous_positions:
        return SizingDecision(
            False, 0.0, 0.0,
            f"{open_count} positions already open (max {limits.max_simultaneous_positions})",
            "max_positions",
        )
    if recently_exited and mint in recently_exited and not limits.reentry_enabled:
        return SizingDecision(
            False, 0.0, 0.0,
            f"re-entry into this mint is disabled (last exit "
            f"{_fmt_ago(now - recently_exited[mint]) if now else 'recently'})",
            "reentry_disabled",
        )
    if equity_usd <= 0 or cash_usd <= 0:
        return SizingDecision(False, 0.0, 0.0, "account has no cash or equity", "no_cash")
    if not price or price <= 0:
        return SizingDecision(False, 0.0, 0.0, "no valid price to size against", "no_price")

    # 1) route target, clamped into the configured band
    pct = max(limits.default_position_pct_low, min(limits.default_position_pct_high, target_pct))
    binding = "route_target"
    if pct > limits.max_position_pct_equity:
        pct = limits.max_position_pct_equity
        binding = "position_cap"

    # Every cap below is applied to the CASH the position consumes, i.e. the
    # notional PLUS the entry fee. Capping the notional while ignoring the fee
    # would let real exposure exceed the limit by the fee on every single trade.
    fee_mult = 1.0 + max(0.0, entry_fee_bps) / 10_000.0
    cash_needed = equity_usd * pct / 100.0

    # 2) exposure cap
    exposure_room = equity_usd * limits.max_total_exposure_pct_equity / 100.0 - open_exposure
    if exposure_room <= 0:
        return SizingDecision(
            False, 0.0, 0.0,
            f"total exposure ${open_exposure:,.2f} is at the "
            f"{limits.max_total_exposure_pct_equity:.0f}% equity limit",
            "exposure_cap",
        )
    if cash_needed > exposure_room:
        cash_needed = exposure_room
        binding = "exposure_cap"

    # 3) cash cap
    if cash_needed > cash_usd:
        cash_needed = cash_usd
        binding = "cash"

    # 4) per-position cap
    hard = equity_usd * limits.max_position_pct_equity / 100.0
    if cash_needed > hard:
        cash_needed = hard
        binding = "position_cap"

    # 5) floor - below this the trade cannot model realistic costs
    if cash_needed < limits.min_position_notional_usd:
        return SizingDecision(
            False, 0.0, 0.0,
            f"available room ${cash_needed:,.2f} is below the "
            f"${limits.min_position_notional_usd:,.2f} minimum paper position",
            "below_minimum",
        )

    # Round DOWN to cents. Rounding to nearest can push a hard cap over by a
    # fraction of a cent, which is exactly the kind of quiet breach a limit is
    # supposed to make impossible.
    cash_needed = math.floor(cash_needed * 100.0) / 100.0
    notional = math.floor((cash_needed / fee_mult) * 100.0) / 100.0
    quantity = notional / price
    if not math.isfinite(quantity) or quantity <= 0:
        return SizingDecision(False, 0.0, 0.0, "degenerate quantity", "bad_quantity")

    return SizingDecision(
        approved=True,
        notional_usd=notional,
        pct_of_equity=cash_needed / equity_usd * 100.0,
        reason=f"{pct:.0f}% route target, capped by {binding}",
        binding_constraint=binding,
        quantity=quantity,
        cash_required_usd=cash_needed,
        fee_reserved_usd=round(cash_needed - notional, 2),
    )


def scale_target_for_route(
    target_pct: float,
    *,
    liquidity_score: float,
    signal_quality: float,
    route_multiplier: float,
    limits: RiskLimits,
) -> tuple[float, str]:
    """
    Deterministic, pre-declared size adjustment.

    Applied BEFORE the caps. A weak book or weak evidence shrinks the position;
    route memory can shrink it further. It can never exceed the route target.
    """
    pct = target_pct
    notes: list[str] = []
    if liquidity_score < 0.4:
        pct *= 0.7
        notes.append("liquidity_score<0.40 -> x0.7")
    if liquidity_score < 0.3:
        pct *= 0.8
        notes.append("liquidity_score<0.30 -> x0.8")
    if signal_quality < 0.55:
        pct *= 0.85
        notes.append("signal_quality<0.55 -> x0.85")
    if route_multiplier < 1.0:
        pct *= route_multiplier
        notes.append(f"route_memory x{route_multiplier:.2f}")
    pct = min(pct, limits.max_position_pct_equity)
    pct = max(pct, 0.0)
    return pct, ("; ".join(notes) if notes else "no size adjustment")


def _fmt_ago(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    return f"{seconds / 3600:.1f}h ago"
