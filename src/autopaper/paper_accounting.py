"""
PAPER accounting reconciliation — the single canonical view of "what happened".

WHY THIS EXISTS
---------------
The PAPER screen was able to display two mutually exclusive truths at once:

    Total PnL  = +$368.983     (stats.compute over the ledger's closed trades)
    Equity     = $100.0000     (paper_account.json, never settled)
    Cash       = $100.0000

Both numbers are produced by production code and neither is a bug in isolation.
The defect is that NOTHING CHECKS THEM AGAINST EACH OTHER, so a screen can show
a total and a balance that contradict each other and nothing says so.

This module derives the canonical sequential equity curve from the ledger — the
same records ``stats.compute`` reads — and compares it with the account book.
Where they agree the account is authoritative. Where they do not, the ledger is
reported AND the divergence is stated, because a silent choice between two
contradictory numbers is how a wrong one becomes a headline.

ONE ACCOUNTING PATH
-------------------
``reconcile`` asserts the invariant that matters: total PnL is the sum of each
closed trade exactly once. There is no second accumulator here and no place for
a trade to be counted twice; if a caller needs a per-trade series it slices the
rows this module already produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .stats import Stats, compute as compute_stats

#: Cash must never move away from starting capital AND claim zero realized PnL
#: while the ledger holds settled trades. Anything else is a broken link.
DRIFT_OK = "OK"
DRIFT_UNSETTLED = "UNSETTLED_ACCOUNT"
DRIFT_NONE = "NO_TRADES"
DRIFT_FATAL = "FATAL"


@dataclass
class EquityPoint:
    """One settled trade on the canonical curve."""
    index: int
    trade_id: str
    exit_timestamp: float
    net_pnl_usd: float
    equity_after_usd: float
    peak_equity_usd: float
    drawdown_pct: float
    strategy_profile: str | None


@dataclass
class Reconciliation:
    starting_equity: float
    realized_pnl_usd: float
    final_equity_usd: float
    peak_equity_usd: float
    max_drawdown_pct: float
    max_drawdown_scope: str
    curve: list[EquityPoint] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    account_book: dict[str, Any] = field(default_factory=dict)
    drift_status: str = DRIFT_OK
    drift_usd: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "account": {
                # Both naming conventions are emitted. The dashboard and the
                # Telegram stats block read the ``*_usd`` keys; the JSON contract
                # in the task description uses the bare names. Emitting one and
                # forcing the other to translate is how a display silently shows
                # "n/a" for a number that exists.
                "starting_equity": round(self.starting_equity, 6),
                "starting_equity_usd": round(self.starting_equity, 6),
                "equity": round(self.final_equity_usd, 6),
                "equity_usd": round(self.final_equity_usd, 6),
                # ``cash`` here is the SETTLED cash implied by closed trades
                # alone. It equals equity only because this view contains no open
                # positions; when positions are open the caller must source cash
                # and equity from the account book and mark-to-market instead,
                # and this block says so via ``open_positions_considered``.
                "cash": round(self.final_equity_usd, 6),
                "cash_usd": round(self.final_equity_usd, 6),
                "open_positions_considered": 0,
                "realized_pnl": round(self.realized_pnl_usd, 6),
                "realized_pnl_usd": round(self.realized_pnl_usd, 6),
                "unrealized_pnl": 0.0,
                "unrealized_pnl_usd": 0.0,
                "total_return_pct": round(
                    self.realized_pnl_usd / self.starting_equity * 100.0, 4)
                if self.starting_equity else 0.0,
                "peak_equity": round(self.peak_equity_usd, 6),
                "peak_equity_usd": round(self.peak_equity_usd, 6),
                "max_drawdown_pct": round(self.max_drawdown_pct, 4),
                "max_drawdown_scope": self.max_drawdown_scope,
                "source": "LEDGER_SEQUENTIAL_SETTLED",
            },
            "account_book_reported": self.account_book,
            "drift_status": self.drift_status,
            "drift_usd": round(self.drift_usd, 6),
            "performance": self.stats,
            "notes": list(self.notes),
        }


def reconcile(
    trades: Sequence[Any],
    *,
    starting_equity: float = 100.0,
    account_book: dict[str, Any] | None = None,
) -> Reconciliation:
    """
    Build the canonical sequential equity curve and reconcile it with the
    account book.

    ``trades`` are ``Trade`` objects. They are sorted by exit time, exactly as
    ``stats.compute`` sorts them, so the curve and the statistics describe the
    same ordering.
    """
    rows = sorted(trades, key=lambda t: (t.exit_timestamp or 0.0))
    notes: list[str] = []

    stats: Stats = compute_stats(rows, starting_equity=starting_equity)
    realized = float(stats.realized_pnl_usd)

    equity = float(starting_equity)
    peak = equity
    max_dd = 0.0
    curve: list[EquityPoint] = []
    for i, t in enumerate(rows, start=1):
        equity += float(t.net_pnl_usd)
        peak = max(peak, equity)
        dd = ((equity - peak) / peak * 100.0) if peak > 0 else 0.0
        max_dd = min(max_dd, dd)
        curve.append(EquityPoint(
            index=i,
            trade_id=str(getattr(t, "trade_id", "")),
            exit_timestamp=float(t.exit_timestamp or 0.0),
            net_pnl_usd=float(t.net_pnl_usd),
            equity_after_usd=equity,
            peak_equity_usd=peak,
            drawdown_pct=dd,
            strategy_profile=getattr(t, "strategy_profile", None),
        ))

    # ---- INVARIANT: total PnL is the sum of each trade exactly once -------
    summed = float(sum(float(t.net_pnl_usd) for t in rows))
    if abs(summed - realized) > 1e-6:
        notes.append(
            f"INVARIANT VIOLATION: stats.realized_pnl_usd={realized} but the "
            f"per-trade sum is {summed}"
        )
    seen: set[str] = set()
    dupes = [c.trade_id for c in curve if c.trade_id in seen or seen.add(c.trade_id)]
    if dupes:
        notes.append(f"INVARIANT VIOLATION: duplicate trade ids on the curve: {dupes}")

    book = dict(account_book or {})
    rec = Reconciliation(
        starting_equity=float(starting_equity),
        realized_pnl_usd=realized,
        final_equity_usd=equity,
        peak_equity_usd=peak,
        max_drawdown_pct=max_dd,
        max_drawdown_scope=stats.max_drawdown_scope,
        curve=curve,
        stats=stats.as_dict(),
        account_book=book,
        notes=notes,
    )

    # ---- compare against the account book ---------------------------------
    if not rows:
        rec.drift_status = DRIFT_NONE
        rec.notes.append("no closed trades in scope")
        return rec

    book_realized = book.get("realized_pnl_usd")
    book_cash = book.get("cash_usd")
    if book_realized is None and book_cash is None:
        rec.drift_status = DRIFT_OK
        rec.notes.append("no account book supplied; ledger is the only source")
        return rec

    drift = realized - float(book_realized or 0.0)
    rec.drift_usd = drift
    if abs(drift) > max(1e-6, abs(realized) * 1e-9):
        rec.drift_status = DRIFT_UNSETTLED
        rec.notes.append(
            f"ACCOUNT BOOK NOT SETTLED: the ledger settles ${realized:,.4f} but "
            f"the account book reports ${float(book_realized or 0.0):,.4f} "
            f"(cash ${float(book_cash or 0.0):,.4f}). The ledger figure is shown "
            f"above because it is derived from the closed-trade records that "
            f"produced the headline; the account book was never debited by these "
            f"exits. Do not read the account cash as the balance."
        )
    else:
        rec.drift_status = DRIFT_OK
        rec.notes.append("account book agrees with the ledger")
    return rec


def stats_for_profile(
    trades: Sequence[Any],
    profile: str | None,
    *,
    starting_equity: float = 100.0,
) -> dict[str, Any]:
    """
    Stats for ONE strategy profile, or for all of them when ``profile`` is None.

    This is what a strategy selector must call. Returning a pooled number while
    the operator believes they selected ``L25_CURRENT`` is how two different
    strategies end up reported as one.
    """
    if profile is None or profile == "ALL":
        return compute_stats(trades, starting_equity=starting_equity).as_dict()
    sel = [t for t in trades
           if getattr(t, "strategy_profile", None) == profile]
    return compute_stats(sel, starting_equity=starting_equity,
                         _profile_breakdown=False).as_dict()


def available_profiles(trades: Sequence[Any]) -> list[str]:
    return sorted({str(getattr(t, "strategy_profile", None)) for t in trades
                   if getattr(t, "strategy_profile", None) is not None})