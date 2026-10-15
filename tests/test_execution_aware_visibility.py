"""
Operator-visibility tests for the executable-quote PAPER runtime.

A simulated PAPER BUY that notifies the operator MUST be visible in the same
canonical PAPER state the dashboard, OPEN POSITIONS, ``/balance`` and
``/status`` read. These tests drive the real ``PaperTrader`` wiring with an
injected quote service, so a divergence between the arms' state and the
operator's view fails here rather than in a Telegram chat.

OFFLINE: every collaborator is injected; no test reaches RPC.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from autopaper.config import build_config
from autopaper.engine import PaperTrader
from autopaper.modes import TradeMode
from autopaper.paper_live.fills import FillModel
from autopaper.paper_live.quotes import SolReference

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from paper_bridge_fakes import (  # noqa: E402
    FakeQuoteService,
    StubExit,
    alert,
    real_report,
    run,
)

SOL_USD = 150.0


def _trader(tmp_path, quotes=None, exits=None) -> PaperTrader:
    """A runtime with every network collaborator injected at CONSTRUCTION.

    ``quote_service`` is passed to ``PaperTrader`` rather than assigned onto
    the bridge afterwards: conftest's guard refuses a build that would reach
    for a real RPC client, and it is right to.
    """
    trader = PaperTrader(
        build_config(tmp_path), quote_service=quotes or FakeQuoteService())
    bridge = run(trader._paper_bridge_instance())
    bridge.exit_policy = exits or StubExit(hold_until=asyncio.run(
        _far_future()))
    bridge.token_decimals_for = lambda mint: 6
    bridge.sol_reference = SolReference(SOL_USD, 0.0, "test")
    bridge.quote_service.sol_reference = bridge.sol_reference
    return trader


async def _far_future() -> float:
    import time

    return time.time() + 3600.0


def _fill_one(trader: PaperTrader) -> list:
    """Drive one authorized BUY through the REAL runtime entry path."""
    return run(trader._paper_bridge.handle_alert(
        snapshot=alert(), report=real_report()))


# ===========================================================================
# ENTRY
# ===========================================================================


class TestEntryVisibility:
    def test_a_simulated_buy_creates_a_position_in_the_arm_account(
        self, tmp_path):
        trader = _trader(tmp_path)
        records = _fill_one(trader)
        traded = [r for r in records if r.traded]
        assert traded, "the injected quote produced no fill at all"
        account = trader._paper_bridge.accounts[traded[0].arm]
        assert account.position is not None
        assert account.position.mint

    def test_open_positions_view_shows_the_simulated_buy(self, tmp_path):
        """
        THE DEFECT. The arms' positions live in ``VirtualAccount.position``;
        ``open_positions_view`` read only the legacy ``PaperAccount``, so a
        notified fill rendered as an empty OPEN POSITIONS.
        """
        trader = _trader(tmp_path)
        records = _fill_one(trader)
        assert any(r.traded for r in records)

        rows = trader.open_positions_view()
        assert rows, (
            "OPEN POSITIONS is empty while an arm holds a simulated position; "
            "the notified BUY is invisible to the operator")
        mints = {r.get("mint") for r in rows}
        assert records[0].signal.token_mint in mints
        assert all(r.get("mode") == TradeMode.PAPER.value for r in rows)

    def test_each_open_arm_position_is_one_row(self, tmp_path):
        trader = _trader(tmp_path)
        records = _fill_one(trader)
        traded = {r.arm for r in records if r.traded}
        rows = trader.open_positions_view()
        assert len(rows) == len(traded), (
            "one row per open arm position, no more and no fewer")
        assert {r.get("arm") for r in rows} == traded

    def test_the_rows_carry_the_executable_fill_not_the_mark(
        self, tmp_path):
        trader = _trader(tmp_path)
        _fill_one(trader)
        rows = trader.open_positions_view()
        row = next(r for r in rows if r.get("source") == "execution_aware_paper")
        fill = row["entry_price"]
        mark = row.get("observed_mark")
        assert fill is not None and float(fill) > 0
        assert mark is not None
        assert float(fill) != pytest.approx(float(mark), abs=1e-15), (
            "the row's entry price is the mark, not the executable fill")

    def test_the_rows_name_the_arm_and_the_venue(self, tmp_path):
        trader = _trader(tmp_path)
        records = _fill_one(trader)
        traded = next(r for r in records if r.traded)
        row = next(
            r for r in trader.open_positions_view()
            if r.get("arm") == traded.arm)
        assert row["arm"] == traded.arm
        assert row.get("venue")
        assert row.get("venue_id")


class TestCashAndExposure:
    """
    THE ARMS ARE THE CANONICAL PAPER BOOK.

    Not the legacy ``PaperAccount``: that one only runs behind the explicit
    rollback flag, and merging three independent $5 research accounts into the
    $100 legacy balance would blur two accounting bases that the status block
    exists to keep apart. What must hold is that the canonical state the
    operator reads agrees with the arms, on every axis.
    """

    def test_the_arm_accounts_are_the_canonical_paper_book(self, tmp_path):
        trader = _trader(tmp_path)
        records = _fill_one(trader)
        traded = [r for r in records if r.traded]
        assert traded

        bridge = trader._paper_bridge
        assert len(bridge.open_positions()) == len(traded)
        # Every open position is a position the operator can be shown.
        rows = trader.open_positions_view()
        assert len(rows) == len(traded)
        assert {r["position_id"] for r in rows} == {
            f"paperlive_{p.signal_id}_{p.arm}" for p in bridge.open_positions()}

    def test_cash_decreases_and_exposure_increases(self, tmp_path):
        trader = _trader(tmp_path)
        records = _fill_one(trader)
        traded = [r for r in records if r.traded]
        assert traded
        bridge = trader._paper_bridge

        for record in traded:
            account = bridge.accounts[record.arm]
            assert account.position is not None
            assert account.open_exposure_usd > 0, (
                "the arm committed capital but reports zero exposure")
            assert account.cash_usd < account.bankroll_usd, (
                "available cash did not fall when the position was funded")

        assert bridge.open_exposure_usd() == pytest.approx(
            sum(bridge.accounts[r.arm].open_exposure_usd for r in traded))

    def test_the_committed_notional_matches_the_filled_size(self, tmp_path):
        trader = _trader(tmp_path)
        records = _fill_one(trader)
        for record in (r for r in records if r.traded):
            account = trader._paper_bridge.accounts[record.arm]
            row = next(r for r in trader.open_positions_view()
                       if r.get("arm") == record.arm)
            assert account.open_exposure_usd == pytest.approx(
                float(row["size_usd"]), abs=1e-12), (
                "the arm's committed capital and the row's size disagree")

    def test_the_dashboard_never_reports_no_exposure_over_an_open_position(
        self, tmp_path):
        """
        The exact defect: "cash is on the book; there is no open exposure"
        printed while three simulated BUYs were open.
        """
        trader = _trader(tmp_path)
        records = _fill_one(trader)
        traded = [r for r in records if r.traded]
        assert traded

        snapshot = trader._dashboard_snapshot_for_test()
        assert snapshot["open_exposure_usd"] > 0, (
            "the dashboard shows zero exposure while a position is open")
        assert snapshot["open_positions"] == len(traded)
        assert snapshot["execution_aware_paper"]["committed_usd"] > 0

    def test_the_dashboard_says_nothing_about_exposure_when_flat(
        self, tmp_path):
        trader = _trader(tmp_path)
        snapshot = trader._dashboard_snapshot_for_test()
        assert snapshot["open_exposure_usd"] == 0.0
        assert "execution_aware_paper" not in snapshot, (
            "an execution-aware block is rendered for a book with no arms open")


# ===========================================================================
# NOTIFICATION ORDERING
# ===========================================================================


class TestNotificationOrdering:
    def test_the_position_is_committed_before_the_notification_is_sent(
        self, tmp_path):
        """
        The entry alert must not describe a fill the canonical state has not
        accepted. A crash between the two would leave the operator holding a
        receipt for a position that does not exist.
        """
        trader = _trader(tmp_path)
        seen: dict = {}

        def capture() -> None:
            seen.setdefault(
                "open_at_notify",
                len(trader._paper_bridge.open_positions()))

        trader._notify = lambda text, ctx=None: capture()
        run(trader._execute_paper_buy_execution_aware(
            real_report(), alert(), _features()))
        assert seen.get("open_at_notify"), (
            "the entry notification was sent before the position existed in "
            "canonical PAPER state")
        assert trader.last_execution.get("executed") is True
        assert trader.last_execution.get("outcome") == "PAPER_FILL=SUCCESS"

    def test_a_refused_entry_records_a_reason_in_the_canonical_outcome(
        self, tmp_path):
        """
        The counterpart defect: a BUY that filled nothing must not keep reading
        ``executed=False`` with ``reason=None``, which is indistinguishable
        from "never got that far".
        """
        trader = _trader(tmp_path)
        bridge = trader._paper_bridge
        bridge.sol_reference = None
        bridge.quote_service.sol_reference = None
        run(trader._execute_paper_buy_execution_aware(
            real_report(), alert(), _features()))
        assert not bridge.open_positions()
        assert trader.last_execution["executed"] is False
        # The outcome names the ACTUAL refusal rather than a vague bucket. It
        # used to read "BUY_BLOCKED" while ``reason`` said something entirely
        # different, so an operator had to know to read the second field; today
        # the two agree, and neither can be None on an attempted entry.
        assert trader.last_execution["outcome"] == "SOL_REFERENCE_MISSING"
        assert trader.last_execution["reason"] == "SOL_REFERENCE_MISSING"
        assert trader.last_execution["reason"] is not None
        assert not trader.open_positions_view()


def _features():
    """``DecisionReport.features`` is the plain dict the engine actually builds."""
    return {"volume_to_liquidity": 0.75, "signal_quality": 0.72,
            "liquidity_usd": 60_000.0, "market_cap_usd": 20_000.0}


# ===========================================================================
# EXIT
# ===========================================================================


class TestSellPath:
    def test_exit_removes_the_position_and_restores_cash(self, tmp_path):
        trader = _trader(tmp_path)
        run(trader._execute_paper_buy_execution_aware(
            real_report(), alert(), _features()))
        bridge = trader._paper_bridge
        assert bridge.open_positions()
        cash_at_entry = bridge.accounts["L25_CURRENT"].cash_usd

        bridge.exit_policy = _ExitNow()
        trades = run(bridge.tick())

        assert trades, "the exit produced no trade record"
        assert not bridge.open_positions(), (
            "a settled position is still open in canonical PAPER state")
        assert not trader.open_positions_view(), (
            "OPEN POSITIONS still lists a position that has settled")
        restored = bridge.accounts["L25_CURRENT"]
        assert restored.cash_usd != cash_at_entry, (
            "available cash was not restored by the settlement")
        assert restored.open_exposure_usd == 0.0

    def test_the_realized_pnl_lands_on_the_arm_bankroll(self, tmp_path):
        trader = _trader(tmp_path)
        run(trader._execute_paper_buy_execution_aware(
            real_report(), alert(), _features()))
        bridge = trader._paper_bridge
        before = bridge.accounts["L25_CURRENT"].bankroll_usd
        bridge.exit_policy = _ExitNow()
        run(bridge.tick())
        after = bridge.accounts["L25_CURRENT"].bankroll_usd
        assert after != before, (
            "the settlement never moved the arm's bankroll")

    def test_the_settlement_is_written_to_the_ledger(self, tmp_path):
        trader = _trader(tmp_path)
        run(trader._execute_paper_buy_execution_aware(
            real_report(), alert(), _features()))
        trader._paper_bridge.exit_policy = _ExitNow()
        run(trader.monitor_once())
        events = [r for r in trader.ledger.events()
                  if "settlement" in str(r)]
        assert events, (
            "the settlement left no record the operator can audit")


class _ExitNow:
    def assess(self, position, *, now, mark_price, data_state, **kw):
        from autopaper.exits import ExitDecision, ExitStyle

        return ExitDecision(
            "EXIT", "test exit", ExitStyle.FAST_FAILURE, trigger="test",
        )


# ===========================================================================
# PERSISTENCE / RESTART
# ===========================================================================


class TestPersistenceAndRestart:
    def test_state_is_written_to_disk_on_every_entry(self, tmp_path):
        trader = _trader(tmp_path)
        assert not trader.paper_state_path().exists()
        run(trader._execute_paper_buy_execution_aware(
            real_report(), alert(), _features()))
        assert trader.paper_state_path().exists(), (
            "no durable PAPER state was written; a restart would lose the book")

    def test_a_restart_rehydrates_every_open_arm_position(self, tmp_path):
        trader = _trader(tmp_path)
        run(trader._execute_paper_buy_execution_aware(
            real_report(), alert(), _features()))
        open_before = {
            f"{p.arm}::{p.signal_id}"
            for p in trader._paper_bridge.open_positions()
        }
        assert open_before

        # A genuinely fresh runtime over the same state directory.
        restarted = _trader(tmp_path)
        rows = restarted.open_positions_view()
        assert rows, "restart came back with an empty book"
        assert {f"{r['arm']}::{r['signal_id']}" for r in rows} == open_before

    def test_all_three_arms_are_represented_in_persisted_state(self, tmp_path):
        trader = _trader(tmp_path)
        run(trader._execute_paper_buy_execution_aware(
            real_report(), alert(), _features()))
        traded = {
            r["arm"] for r in trader.open_positions_view()
        }
        assert traded, "no arm is open, so there is nothing to persist"

        payload = json.loads(
            trader.paper_state_path().read_text(encoding="utf-8"))
        persisted = {p.get("arm") for p in payload["open_positions"]}
        assert persisted == traded, (
            f"persisted arms {persisted} != open arms {traded}")

    def test_a_closed_position_is_not_revived_by_a_restart(self, tmp_path):
        trader = _trader(tmp_path)
        run(trader._execute_paper_buy_execution_aware(
            real_report(), alert(), _features()))
        trader._paper_bridge.exit_policy = _ExitNow()
        run(trader._paper_bridge.tick())
        trader.persist_paper_dataset()

        restarted = _trader(tmp_path)
        assert not restarted._paper_bridge.open_positions(), (
            "a settled position was resurrected by the restart")
        assert not restarted.open_positions_view()


class TestMonitorIntegration:
    def test_the_monitor_tick_exits_the_canonical_position(self, tmp_path):
        trader = _trader(tmp_path)
        run(trader._execute_paper_buy_execution_aware(
            real_report(), alert(), _features()))
        assert trader.open_positions_view()
        trader._paper_bridge.exit_policy = _ExitNow()
        run(trader.monitor_once())
        assert not trader.open_positions_view(), (
            "monitor_once left the canonical position open after the exit")

    def test_the_monitor_does_not_disturb_the_legacy_book(self, tmp_path):
        """
        The two PAPER books must not bleed into each other. The arms are
        separate accounts by design; a fill must not manufacture a position in
        the legacy marked-to-market account.
        """
        trader = _trader(tmp_path)
        run(trader._execute_paper_buy_execution_aware(
            real_report(), alert(), _features()))
        assert trader.open_positions_view()
        assert not trader.account.open_positions(), (
            "an execution-aware fill leaked into the legacy PAPER book")
