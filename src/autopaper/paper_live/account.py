"""
One virtual account per research arm.

WHY NOT ``autopaper.account.PaperAccount``
==========================================

That account marks positions to the observed market on purpose: its
``equity()`` is ``cash + market_value(open)`` where ``market_value`` uses the
mark. That is correct for a mark-based paper book and exactly wrong here - a
bankroll marked to observations is the number this whole harness exists to stop
relying on.

So each arm gets a small explicit account whose bankroll moves ONLY on realized
fill-based settlement. ``equity`` is not computed from marks at all; while a
position is open the bankroll is the pre-entry value, and it changes when the
position settles.

INDEPENDENCE
============

Each arm owns its own cash and its own open position. Nothing is shared. Two
arms cannot move each other's bankroll even by accident, because the only
mutating methods are on an instance and no instance holds a reference to
another.

SIZING
======

Delegated to :func:`autopaper.sizing.compute_position_size` - the production
formula, unmodified. This module does not implement a percentage, a tier or a
boundary of its own; it asks production and records the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from autopaper.sizing import PositionSize, compute_position_size

from .schema import FailureReason, PaperPosition, RealizedPnl, TradeRecord


@dataclass
class VirtualAccount:
    """
    One arm's virtual bankroll.

    ``starting_bankroll_usd`` defaults to $5.00, which is the figure the
    comparison is specified at and which keeps the position size small enough
    that a $0.50 order does not consume a meaningful share of a pool's
    liquidity.
    """

    arm: str
    arm_version: str
    baseline_hash: str
    starting_bankroll_usd: float = 5.00
    bankroll_usd: float = field(default=None)  # type: ignore[assignment]
    position: PaperPosition | None = None
    closed_trades: int = 0
    peak_bankroll_usd: float = field(default=None)  # type: ignore[assignment]
    blocked_by_position_active: int = 0
    blocked_by_cooldown: int = 0

    def __post_init__(self) -> None:
        if self.bankroll_usd is None:
            self.bankroll_usd = self.starting_bankroll_usd
        if self.peak_bankroll_usd is None:
            self.peak_bankroll_usd = self.starting_bankroll_usd

    # -- state ------------------------------------------------------------

    @property
    def has_position(self) -> bool:
        return self.position is not None

    @property
    def open_exposure_usd(self) -> float:
        """USD notional committed while a position is open."""
        if self.position is None:
            return 0.0
        return float(getattr(self.position, "usd_notional", 0.0) or 0.0)

    @property
    def cash_usd(self) -> float:
        """
        Cash actually AVAILABLE to this arm.

        ``bankroll_usd`` is the settled total and deliberately does not move
        while a position is open -- the bankroll only ever changes on realized
        settlement between two fills, never on an observation. That is the
        property this whole harness exists to preserve, so it stays.

        Available cash is therefore the settled bankroll MINUS whatever the open
        position committed. Without this, an arm that has just filled a $0.50
        position reports the same cash as one that never traded, and "cash on
        the book; no exposure" gets printed over an open position.
        """
        return self.bankroll_usd - self.open_exposure_usd

    def size_for_bankroll(self) -> PositionSize:
        """
        The PRODUCTION sizing formula applied to this arm's own bankroll.

        $5.00 -> 10% -> $0.50. After $5.40 -> $0.54. The $0.23 diagnostic quote
        size is NOT used here; that was a measurement notional, not a position.
        """
        return compute_position_size(self.bankroll_usd)

    # -- lifecycle --------------------------------------------------------

    def open_position(self, position: PaperPosition) -> None:
        if self.position is not None:
            raise RuntimeError(
                f"arm {self.arm} already holds a position; the one-open-position "
                "constraint is enforced by the caller and must not be bypassed"
            )
        self.position = position

    def settle(self, pnl: RealizedPnl, trade: TradeRecord) -> None:
        """Apply a realized fill-based settlement to the bankroll."""
        self.bankroll_usd = pnl.bankroll_after
        self.peak_bankroll_usd = max(self.peak_bankroll_usd, pnl.bankroll_after)
        self.position = None
        self.closed_trades += 1

    def restore_settled_bankroll(
        self, settled_realized_pnl_usd: float, *, peak_bankroll_usd: float | None = None,
    ) -> None:
        """
        Re-establish this arm's bankroll from its SETTLED history after a restart.

        WHY THIS EXISTS
        ---------------
        ``__post_init__`` seeds the bankroll from ``starting_bankroll_usd`` and
        ``settle()`` is the only thing that moves it. ``__post_init__`` runs on
        every process start, so without this method an arm that lost money came
        back to life at exactly $5.00 after every restart and sized its next BUY
        off a balance it no longer had. That was a PRODUCTION sizing reset, not a
        reporting cosmetic: ``size_for_bankroll()`` reads ``bankroll_usd``, so the
        restored-notional was real money the arm did not have.

        WHAT IS AND IS NOT RESTORED
        ----------------------------
        The caller passes the cumulative realized PnL of this arm's SETTLED
        trades, taken from the canonical keyed trade book with each settlement
        counted EXACTLY ONCE. Nothing is reconstructed here: this method only
        ADDS a total that already exists, and it is additive-once by
        construction, so calling it twice with the same total is idempotent.

        It deliberately does NOT read per-trade ``bankroll_before`` values. Those
        are a chain, and a chain that was interrupted by a reset carries the
        reset forward; summing realized PnL is the only formulation immune to it.
        """
        self.bankroll_usd = self.starting_bankroll_usd + float(
            settled_realized_pnl_usd)
        if peak_bankroll_usd is not None:
            self.peak_bankroll_usd = max(
                self.starting_bankroll_usd, float(peak_bankroll_usd),
                self.bankroll_usd)

    def stats(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "strategy_version": self.arm_version,
            "baseline_hash": self.baseline_hash,
            "starting_bankroll_usd": self.starting_bankroll_usd,
            "bankroll_usd": self.bankroll_usd,
            "cash_usd": self.cash_usd,
            "open_exposure_usd": self.open_exposure_usd,
            "peak_bankroll_usd": self.peak_bankroll_usd,
            "realized_trades": self.closed_trades,
            "has_position": self.has_position,
            "open_position_mint": (
                self.position.mint if self.position is not None else None),
            "entry_blocked_position_active": self.blocked_by_position_active,
            "entry_blocked_cooldown": self.blocked_by_cooldown,
            "basis": (
                "bankroll moves only on realized settlement between two "
                "SIMULATED_FILLs; it is never marked to the observed market"
            ),
        }


__all__ = ["VirtualAccount"]
