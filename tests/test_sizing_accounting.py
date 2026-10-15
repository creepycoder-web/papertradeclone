"""
Account accounting invariants for the PAPER account.

Cash must never go negative, ``equity = cash + market_value`` must always be
exact, and every closed trade must reconcile.

POSITION SIZING IS NOT TESTED HERE.
It was, until the V3 migration replaced the route-based sizer (a target
percentage, a band clamp, an exposure cap, a cash cap and a $5 floor) with the
single frozen V3-C policy. Those tests now live in ``test_sizing_v3c.py``,
because there is no second sizing path left to test.
"""

from __future__ import annotations

import pytest

from autopaper.account import MarkedPosition, PaperAccount
from autopaper.config import ExitConfig, RiskLimits
from autopaper.ledger import PaperAccountState
from autopaper.models import (
    DataState,
    ExitStyle,
    Position,
    PositionStatus,
    Regime,
    Route,
)

from .conftest import MINT, MINT2

pytestmark = pytest.mark.strategy

LIMITS = RiskLimits()
CFG_EXITS = ExitConfig()


def pos(mint=MINT, notional=10.0, status=PositionStatus.OPEN, *,
        entry_ref=1e-4, entry_slip_bps=100.0, entry_fee_bps=85.0) -> Position:
    """
    Build a position the way the real engine does: a pre-slippage reference, an
    executed fill worse than it, and a cash outlay that is notional + fee.

    Getting this relationship right in the fixture matters - the accounting
    invariant in the engine is only as good as the inputs it is handed.
    """
    entry_fill = entry_ref * (1 + entry_slip_bps / 10_000.0)
    qty = notional / entry_fill
    fee = qty * entry_fill * entry_fee_bps / 10_000.0
    return Position(
        position_id=f"p_{mint}_{notional}_{id(mint)}", candidate_id="c", mint=mint,
        symbol="X", status=status, route=Route.FAST_ENTRY,
        regime=Regime.EARLY, exit_style=ExitStyle.MOMENTUM_FAILURE, strategy="s",
        entry_signal_timestamp=0.0, entry_decision_timestamp=0.0,
        entry_timestamp=0.0, entry_reference_price=entry_ref,
        fill_price=entry_fill, quantity=qty,
        cash_out_usd=qty * entry_fill + fee,
        cost_basis_usd=qty * entry_fill + fee,
        entry_fee_usd=fee,
        entry_slippage_usd=qty * (entry_fill - entry_ref),
    )


# ---------------------------------------------------------------------------
# sizing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# accounting
# ---------------------------------------------------------------------------


def account() -> PaperAccount:
    return PaperAccount(
        PaperAccountState(starting_bankroll_usd=100.0, cash_usd=100.0), LIMITS
    )


def test_entry_moves_cash_not_equity():
    a = account()
    p = pos(notional=20.0)
    a.register_position(p)
    assert a.cash == pytest.approx(100.0 - p.cash_out_usd)
    assert a.equity() == pytest.approx(100.0)      # valued at cost
    assert a.open_exposure() == pytest.approx(p.cash_out_usd)


def test_equity_identity_always_holds():
    a = account()
    p = pos(notional=20.0)
    a.register_position(p)
    a.apply_mark(p, 3e-4, DataState.LIVE, 1.0)
    marks = {p.position_id: MarkedPosition(p, 3e-4, DataState.LIVE)}
    equity = a.equity(marks)
    assert equity == pytest.approx(a.cash + a.market_value(marks))
    assert a.snapshot(marks)["identity_check"] == pytest.approx(0.0, abs=1e-9)
    assert equity == pytest.approx(100.0 - p.cash_out_usd + p.quantity * 3e-4)


def test_exit_realises_and_returns_cash():
    a = account()
    cash_at_start = a.cash
    p = pos(notional=20.0)
    a.register_position(p)
    exit_ref, exit_fee, exit_slip = 3e-4, 0.17, 0.02
    trade = a.close_position(
        p, exit_reference=exit_ref,
        exit_fill_price=exit_ref - exit_slip / p.quantity,
        exit_fee=exit_fee, exit_slippage=exit_slip, now=10.0,
        reason="test", style=ExitStyle.MOMENTUM_FAILURE,
    )
    assert p.status is PositionStatus.CLOSED
    assert a.cash == pytest.approx(cash_at_start + trade.net_pnl_usd)
    assert a.realized == pytest.approx(trade.net_pnl_usd)
    # THE BASIS CHANGED, THE INVARIANTS DID NOT.
    #
    # ``gross_pnl_usd`` is now (exit_fill - entry_fill) * qty -- the two prices that
    # actually moved cash -- so slippage is ALREADY inside it and must not be
    # subtracted again. Fees are still subtracted exactly once.
    assert trade.net_pnl_usd == pytest.approx(
        trade.gross_pnl_usd - p.entry_fee_usd - exit_fee
    )
    # and the basis is stated on the record rather than inferred
    assert trade.pnl_basis == "FILL_VS_FILL"
    # the reference-basis gross is carried alongside, and agrees exactly when the
    # recorded slippage equals the price difference it implies
    assert trade.gross_pnl_reference_basis_usd == pytest.approx(
        (exit_ref - p.entry_reference_price) * p.quantity
    )
    assert trade.pnl_basis_disagreement_usd == pytest.approx(0.0, abs=1e-12)
    assert a.equity() == pytest.approx(a.cash)


def test_pnl_identity_with_fees_and_slippage():
    a = account()
    cash_at_start = a.cash
    p = pos(notional=20.0)
    a.register_position(p)
    exit_ref = 1.5e-4
    exit_slip = 0.03
    exit_fill = exit_ref - exit_slip / p.quantity
    trade = a.close_position(
        p, exit_reference=exit_ref, exit_fill_price=exit_fill,
        exit_fee=0.15, exit_slippage=exit_slip, now=5.0,
        reason="t", style=ExitStyle.MOMENTUM_FAILURE,
    )
    # gross is FILL vs FILL: the executed entry price against the executed exit
    # price. Slippage lives inside both of those numbers, so it is not deducted a
    # second time here; the fee is.
    assert trade.gross_pnl_usd == pytest.approx(
        (exit_fill - p.fill_price) * p.quantity
    )
    assert trade.net_pnl_usd == pytest.approx(
        trade.gross_pnl_usd - p.entry_fee_usd - 0.15
    )
    # THE QUOTED ENTRY IS NOT THE FILLED ENTRY, AND PnL MUST USE THE FILLED ONE.
    # p.quoted_price is deliberately unset on this fixture, so the quoted reference
    # (entry_reference_price) and the filled price differ by exactly the slippage.
    assert p.fill_price > p.entry_reference_price
    reference_basis = (
        (exit_ref - p.entry_reference_price) * p.quantity
        - p.entry_fee_usd - 0.15 - p.entry_slippage_usd - exit_slip
    )
    assert trade.net_pnl_usd == pytest.approx(reference_basis)
    # the invariant that catches a double-counted cost
    assert a.cash - cash_at_start == pytest.approx(trade.net_pnl_usd, abs=1e-9)


def test_cash_never_goes_negative():
    a = account()
    a.state.cash_usd = 1.0
    p = pos(notional=50.0)
    a.register_position(p)
    assert a.cash >= 0.0
    assert p.status is PositionStatus.ERROR


def test_mfe_and_mae_track_the_path():
    a = account()
    p = pos(notional=20.0)
    a.register_position(p)
    for price in (1.0e-4, 1.5e-4, 2.0e-4, 1.2e-4, 0.8e-4):
        a.apply_mark(p, price, DataState.LIVE, 1.0)
    # returns are measured from the EXECUTED fill, which is already worse than
    # the reference, so MFE is not the naive 2e-4/1e-4
    assert p.peak_price == pytest.approx(2.0e-4)
    assert p.trough_price == pytest.approx(0.8e-4)
    assert p.mfe_pct == pytest.approx((2.0e-4 / p.fill_price - 1.0) * 100.0)
    assert p.mae_pct == pytest.approx((0.8e-4 / p.fill_price - 1.0) * 100.0)
    assert p.peak_return_pct == pytest.approx(p.mfe_pct)
    assert p.mfe_pct > 90.0 and p.mae_pct < -20.0


def test_unusable_data_marks_at_cost_and_is_flagged():
    a = account()
    p = pos(notional=20.0)
    a.register_position(p)
    a.apply_mark(p, None, DataState.UNUSABLE, 1.0)
    marks = {p.position_id: MarkedPosition(p, None, DataState.UNUSABLE)}
    assert a.market_value(marks) == pytest.approx(p.cost_basis_usd)
    assert a.unrealized(marks) == 0.0
    assert a.equity(marks) == pytest.approx(100.0)     # identity preserved
    assert p.data_state is DataState.UNUSABLE


def test_cooldown_blocks_reentry_until_enabled():
    a = account()
    a.recently_exited[MINT] = 1000.0
    assert a.cooldown_active(MINT, 1010.0) is True
    assert a.cooldown_active(MINT, 1000.0 + LIMITS.reentry_cooldown_s + 1) is False
    assert LIMITS.reentry_enabled is False


def test_drawdown_tracking():
    a = account()
    a.update_peak(120.0)
    assert a.current_drawdown(120.0) == pytest.approx(0.0)
    assert a.current_drawdown(90.0) == pytest.approx(-25.0)


def test_cash_delta_equals_net_pnl_over_many_random_trades():
    """Fuzz the accounting invariant across fills, fees and directions."""
    import random

    rng = random.Random(1234)
    a = account()
    for i in range(300):
        qty = rng.uniform(1_000, 5_000_000)
        entry_ref = rng.uniform(1e-7, 1e-2)
        entry_fill = entry_ref * (1 + rng.uniform(0.0005, 0.02))
        exit_ref = entry_ref * (1 + rng.uniform(-0.6, 3.0))
        exit_fill = exit_ref * (1 - rng.uniform(0.0005, 0.02))
        e_fee = qty * entry_fill * rng.uniform(0.0005, 0.003)
        x_fee = qty * exit_fill * rng.uniform(0.0005, 0.003)
        e_slip = qty * (entry_fill - entry_ref)
        x_slip = qty * (exit_ref - exit_fill)
        p = pos(mint=f"mint{i}", notional=qty * entry_fill, entry_ref=entry_ref)
        p.quantity = qty
        p.fill_price = entry_fill
        p.entry_reference_price = entry_ref
        p.entry_fee_usd = e_fee
        p.entry_slippage_usd = e_slip
        p.cash_out_usd = qty * entry_fill + e_fee
        if a.cash < p.cash_out_usd:
            continue
        cash_before = a.cash
        a.register_position(p)
        trade = a.close_position(
            p, exit_reference=exit_ref, exit_fill_price=exit_fill, exit_fee=x_fee,
            exit_slippage=x_slip, now=float(i), reason="fuzz",
            style=ExitStyle.MOMENTUM_FAILURE,
        )
        assert a.cash - cash_before == pytest.approx(trade.net_pnl_usd, abs=1e-8)
        # fees deducted exactly once from a gross that already contains slippage
        assert trade.net_pnl_usd == pytest.approx(
            trade.gross_pnl_usd - e_fee - x_fee, abs=1e-12
        )
        # gross is exactly (exit_fill - entry_fill) * qty over 300 random fills
        assert trade.gross_pnl_usd == pytest.approx(
            (exit_fill - entry_fill) * qty, abs=1e-12
        )
        # and the two bases agree whenever slippage is charged honestly
        assert trade.pnl_basis_disagreement_usd == pytest.approx(0.0, abs=1e-9)


def test_snapshot_reports_all_required_fields():
    a = account()
    p = pos(notional=20.0)
    a.register_position(p)
    s = a.snapshot()
    for key in ("cash_usd", "equity_usd", "realized_pnl_usd", "unrealized_pnl_usd",
                "total_pnl_usd", "starting_bankroll_usd", "open_exposure_usd",
                "fees_paid_usd", "slippage_paid_usd", "identity_check"):
        assert key in s, key


def test_unfilled_attempt_is_counted_not_hidden():
    a = account()
    p = pos()
    a.mark_unfilled(p, "no_quote", 5.0)
    assert a.unfilled_attempts == 1
    assert a.cash == pytest.approx(100.0), "an unfilled attempt must not touch cash"


def test_two_overlapping_positions_close_without_false_drift():
    """
    The regression that bit in live trading.

    A whole-account check of `cash_now - cash_at_entry == net_pnl` is only true
    when a single position is open. With two overlapping positions the other
    one's cash flows are in the same delta, and a correct book looks broken -
    the discrepancy is exactly the second position's cost basis.
    """
    a = account()
    first = pos(MINT, 15.0)
    a.register_position(first)
    second = pos(MINT2, 16.0)
    a.register_position(second)
    assert a.cash == pytest.approx(100.0 - first.cash_out_usd - second.cash_out_usd)

    # close the FIRST while the second is still open - the case that fired
    trade = a.close_position(
        first, exit_reference=2e-4, exit_fill_price=2e-4 - 0.02 / first.quantity,
        exit_fee=0.2, exit_slippage=0.02, now=10.0, reason="t",
        style=ExitStyle.RUNNER,
    )
    assert trade.net_pnl_usd > 0
    # and the whole-account delta is NOT this trade's pnl - the second position
    # is still holding its cash out
    assert a.cash != pytest.approx(100.0 + trade.net_pnl_usd)
    # while the position-local identity does hold
    assert trade.net_pnl_usd == pytest.approx(
        (first.quantity * (2e-4 - 0.02 / first.quantity) - 0.2)
        - first.cash_out_usd
    )

    # and the second one closes cleanly afterwards
    trade2 = a.close_position(
        second, exit_reference=1.5e-4,
        exit_fill_price=1.5e-4 - 0.01 / second.quantity,
        exit_fee=0.1, exit_slippage=0.01, now=20.0, reason="t",
        style=ExitStyle.MOMENTUM_FAILURE,
    )
    assert a.realized == pytest.approx(trade.net_pnl_usd + trade2.net_pnl_usd)
    assert a.cash == pytest.approx(100.0 + a.realized, abs=1e-9)


def test_an_unreconcilable_close_is_refused_without_touching_the_books():
    """A refused close must leave cash, status and the position untouched."""
    from autopaper.account import AccountingError
    from autopaper.positions import PositionBook

    class FakeLedger:
        def __init__(self):
            self.rows = []

        def position_update(self, row):
            self.rows.append(row)

        def trade(self, trade):
            self.rows.append({"event": "exit", "trade": trade.trade_id})

    a = account()
    p = pos(MINT, 15.0)
    a.register_position(p)
    cash_before, status_before = a.cash, p.status

    led = FakeLedger()
    book = PositionBook(a, led, CFG_EXITS, LIMITS)

    class LyingFill:
        """
        Claims a reference far from the executed price while charging no
        slippage for the difference - i.e. free money. Must be refused.
        """
        filled = True
        fill_price = 9e-4            # executed here
        price_at_quote = 1.0e-4       # but the books claim the reference was here
        fee_usd = 0.0
        slippage_usd = 0.0
        reason = "filled"

    from autopaper.exits import ExitDecision
    from autopaper.models import ExitStyle

    got = book.close(
        p, decision=ExitDecision("EXIT", "test", ExitStyle.FAST_FAILURE,
                                 trigger="test"),
        fill=LyingFill(), now=30.0,
    )
    assert got is None, "a fill whose books do not reconcile must be refused"
    assert a.cash == pytest.approx(cash_before), "cash must be untouched"
    assert p.status is status_before, "the position must stay open"
    assert any(r.get("action") == "EXIT_REFUSED" for r in led.rows)
    assert not any(r.get("event") == "exit" for r in led.rows)
