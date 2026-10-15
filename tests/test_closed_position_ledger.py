"""
THE CLOSED-POSITIONS LEDGER MUST CONTAIN WHAT THE CLOSE RECEIPT PROMISED.

THE DEFECT
----------
On 2026-10-06 an arm settled a PAPER position, the operator received

    ✅ PAPER CLOSED   $SNORP   realized +$0.09   hold 124.8s

and CLOSED POSITIONS kept reporting ``showing 1-8 of 42`` with every visible
row dated 2026-10-02.

Nothing was lost. The settlement WAS durable -- twice, as an ``arm_exit`` event
in ``var/ledger.jsonl`` and as a keyed trade in ``var/paper_state.json``. The
READER was the bug: ``closed_positions_view`` built itself from
``ledger.trades()``, and that method filters on ``event == "exit"``. The arms
book through ``Ledger.arm_exit``, a deliberately separate event type, so every
settlement the ACTIVE paper book produced since 2026-10-05 -- NI, UNIPCS,
oushi, SNORP -- was structurally invisible to the view the operator was
reading.

TWO FACTS, NOT ONE
------------------
A trade closing and a closed-trade record being persisted are different facts.
These tests hold both apart:

* a close that did not persist must NOT produce a ``PAPER CLOSED`` receipt
  (requirement 5, and the persist -> commit -> emit ordering);
* a close that DID persist must be visible immediately, newest first, exactly
  once, and must survive a restart (requirements 1-4, 6, 8, 9).

OFFLINE: every collaborator is injected. No RPC, no Telegram, no wallet, and
nothing here can construct, sign or broadcast a transaction.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from autopaper.config import build_config
from autopaper.engine import PaperTrader
from autopaper.exits import ExitDecision, ExitStyle
from autopaper.ledger import Ledger
from autopaper.models import Trade
from autopaper.modes import TradeMode
from autopaper.paper_live.quotes import SolReference

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_bridge_fakes import (  # noqa: E402
    FakeQuoteService,
    alert,
    real_report,
    run,
)

SOL_USD = 150.0

#: The bridge's SUCCESSFUL close receipt, verbatim. Asserted on the exact
#: markup so a failure notice is free to reuse the words "PAPER CLOSED" --
#: which it must, since the operator needs to know a close happened at all.
CLEAN_CLOSE_RECEIPT = "<b>PAPER CLOSED</b>"


class _ExitNow:
    """Unconditional EXIT, the production ``ExitPolicy`` signature."""

    def assess(self, position, *, now, mark_price, data_state, **kw):
        return ExitDecision(
            "EXIT", "test exit", ExitStyle.FAST_FAILURE, trigger="test")


def _features():
    """``DecisionReport.features``, the plain dict the engine builds."""
    return {"volume_to_liquidity": 0.75, "signal_quality": 0.72,
            "liquidity_usd": 60_000.0, "market_cap_usd": 20_000.0}


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


def _var(tmp_path) -> Path:
    """Where the runtime actually writes. Asserted, not assumed."""
    var = Path(tmp_path) / "var"
    assert var.is_dir(), f"build_config did not create {var}"
    return var


def _var_mk(tmp_path) -> Path:
    """``_var`` for tests that seed history before a runtime exists."""
    var = Path(tmp_path) / "var"
    var.mkdir(parents=True, exist_ok=True)
    return var


def _trader(tmp_path, *, notifications=None) -> PaperTrader:
    """A runtime whose only network collaborator is injected at CONSTRUCTION."""
    trader = PaperTrader(
        build_config(tmp_path), quote_service=FakeQuoteService())
    bridge = run(trader._paper_bridge_instance())
    bridge.token_decimals_for = lambda mint: 6
    bridge.sol_reference = SolReference(SOL_USD, 0.0, "test")
    bridge.quote_service.sol_reference = bridge.sol_reference
    trader._test_notifications = notifications if notifications is not None \
        else []
    trader._notifier = lambda text, kb=None: trader._test_notifications.append(
        text)
    return trader


def _open_one(trader: PaperTrader, *, symbol: str = "T", message_id: int = 4242):
    """
    Drive one authorized BUY through the REAL runtime entry path.

    The production entry method, not the bridge's internals, so the durable
    write and the receipt it emits are the ones under test.
    """
    run(trader._execute_paper_buy_execution_aware(
        real_report(symbol=symbol),
        alert(message_id=message_id, symbol=symbol),
        _features(),
    ))
    rows = trader.open_positions_view()
    assert rows, f"no arm filled the BUY for {symbol}, nothing to close"
    return len(rows)


def _close_now(trader: PaperTrader):
    """
    Run one monitor cycle so every arm settles, through ``monitor_once``.

    The runtime's own loop, not a direct call into the booking helper, so the
    persist -> commit -> emit ORDERING under test is the production ordering.
    """
    trader._paper_bridge.exit_policy = _ExitNow()
    run(trader.monitor_once())
    settled = [t for t in trader._paper_bridge.trade_records
               if str(getattr(t.failure, "value", t.failure)) == "NONE"]
    assert settled, "no arm settled; the close path was never exercised"
    return settled


def _cycle(tmp_path, *, symbol: str = "T", message_id: int = 4242,
           trader: PaperTrader | None = None):
    """
    One full open -> settle -> record cycle.

    A NEW runtime per cycle by default, because the arms hold a post-exit
    re-entry cooldown per mint in memory: re-signalling the same token in one
    process is correctly refused, and a test wanting two closed positions
    would then be measuring the cooldown instead of the ledger. Two runtimes
    over ONE var dir is also the more honest shape -- it is what an operator
    actually does, and it means every assertion below is answered by the
    durable record rather than by process memory.
    """
    t = trader if trader is not None else _trader(tmp_path)
    arms = _open_one(t, symbol=symbol, message_id=message_id)
    _close_now(t)
    return t, arms


def _arm_exits(tmp_path) -> list[dict]:
    return Ledger(
        _var(tmp_path) / "ledger.jsonl",
        _var(tmp_path) / "decisions.jsonl",
        _var(tmp_path) / "candidates.jsonl",
        _var(tmp_path) / "position_updates.jsonl",
    ).arm_exits()


def _raw_lines(tmp_path, name: str) -> list[dict]:
    path = _var(tmp_path) / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _all_rows(trader: PaperTrader) -> tuple[list[dict], int]:
    """Every row the view holds, paging through instead of reading one window."""
    from autopaper.engine import CLOSED_PAGE_SIZE

    rows: list[dict] = []
    total, offset = 0, 0
    while True:
        window, seen = trader.closed_positions_view(offset=offset)
        total = seen
        if not window:
            break
        rows.extend(window)
        offset += CLOSED_PAGE_SIZE
    return rows, total


def _seed_legacy_exit(tmp_path, *, trade_id: str, symbol: str, at: float) -> None:
    """
    Write a historical legacy ``exit`` record directly.

    The operator's real ledger holds 37 of these, all from 2026-10-02. They are
    the history the fix must not disturb, and they are only reachable by
    writing the file the runtime itself would have written.

    A REAL ``Trade.as_dict()``, not a hand-rolled dict: the reader reconstructs
    ``Trade`` objects and silently drops any payload it cannot rebuild, so a
    partial fixture would test nothing.
    """
    trade = Trade(
        trade_id=trade_id, position_id=trade_id, candidate_id=f"cand_{trade_id}",
        mint=f"Mint{trade_id}", symbol=symbol, source="telegram",
        strategy="L25", route="FAST_ENTRY", regime="EXPLOSIVE",
        entry_signal_timestamp=at - 30.0, entry_timestamp=at - 30.0,
        entry_price=1.0, fill_price=1.0, exit_timestamp=at, exit_price=1.2,
        quantity=1.0, entry_fee_usd=0.0, exit_fee_usd=0.0,
        entry_slippage_usd=0.0, exit_slippage_usd=0.0, gross_pnl_usd=0.25,
        net_pnl_usd=0.25, return_pct=20.0, holding_seconds=30.0,
        mfe_pct=30.0, mae_pct=-1.0, exit_reason="seeded history",
        exit_style="fast_failure", thesis="", invalidation=(),
        booked_at=at,
    )
    path = _var_mk(tmp_path) / "ledger.jsonl"
    rec = {"event": "exit", "at": at, "trade": trade.as_dict()}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


# ---------------------------------------------------------------------------
# 1. the receipt implies a durable record
# ---------------------------------------------------------------------------


class TestPaperCloseIsPersisted:
    def test_a_closed_arm_position_is_written_to_the_ledger(self, tmp_path):
        trader = _trader(tmp_path)
        arms = _open_one(trader)
        _close_now(trader)

        records = _arm_exits(tmp_path)
        assert len(records) == arms, (
            "the arms settled but the ledger does not hold one record each")
        for rec in records:
            assert rec["symbol"]
            assert rec["pnl_basis"] == "FILL_VS_FILL"
            assert rec["settlement_key"], (
                "the settlement must carry the idempotency key the reader "
                "dedupes on, or a duplicate close event becomes a duplicate "
                "ledger row")
            assert rec["trade"]["failure_reason"] == "NONE"

    def test_the_ledger_record_precedes_the_close_receipt(self, tmp_path):
        """
        persist -> commit -> emit, asserted by ORDER.

        The file is read at the instant each receipt is delivered. If the write
        happened after the notification -- or never -- this sees an empty
        ledger and fails. This is the check that separates "the trade closed"
        from "the closed trade is durable".
        """
        trader = _trader(tmp_path)
        arms = _open_one(trader)

        counts_at_send: list[int] = []

        def spy(text, kb=None):
            if CLEAN_CLOSE_RECEIPT in text:
                counts_at_send.append(len(_arm_exits(tmp_path)))
            trader._test_notifications.append(text)

        trader._notifier = spy
        _close_now(trader)

        receipts = [t for t in trader._test_notifications
                    if CLEAN_CLOSE_RECEIPT in t]
        assert len(receipts) == arms
        assert counts_at_send, "the receipt spy never saw a close receipt"
        assert all(n == arms for n in counts_at_send), (
            "a PAPER CLOSED receipt was delivered before that settlement's "
            f"closed-trade record was durable: {counts_at_send} vs {arms}")

    def test_a_failed_write_withholds_the_close_receipt(self, tmp_path):
        """
        (A) the trade closed and (B) the record persisted are DIFFERENT facts.

        A close whose ledger write fails must not be announced as a clean
        close, or the operator is told the ledger holds something it does not.
        """
        trader = _trader(tmp_path)
        _open_one(trader)

        def boom(_payload):
            raise OSError(28, "No space left on device")

        trader.ledger.arm_exit = boom
        _close_now(trader)

        texts = "\n".join(trader._test_notifications)
        # The bridge's CLEAN receipt marker, exactly. The failure notice below
        # deliberately repeats the words "PAPER CLOSED" -- what must be absent
        # is the unmarked success receipt.
        assert CLEAN_CLOSE_RECEIPT not in texts, (
            "a clean close receipt was delivered for a trade that was never "
            "persisted -- the false receipt this fix exists to prevent")
        assert "LEDGER WRITE FAILED" in texts, (
            "the persistence failure must be reported, not swallowed: "
            f"notifications were {trader._test_notifications!r}")
        assert "No space left on device" in texts, (
            "the failure must name its cause so an operator can act on it")

    def test_a_failed_write_leaves_nothing_for_the_view_to_show(self, tmp_path):
        """
        ...and the view agrees with the failure. No phantom row, and no
        ``PAPER CLOSED``-shaped record anywhere in the durable ledger.
        """
        trader = _trader(tmp_path)
        _open_one(trader)
        trader.ledger.arm_exit = lambda _p: (_ for _ in ()).throw(
            OSError("disk gone"))
        _close_now(trader)

        assert _arm_exits(tmp_path) == []
        _rows, total = trader.closed_positions_view()
        assert total == 0, (
            "CLOSED POSITIONS shows a close that was never persisted")

    def test_the_failure_is_journalled_durably(self, tmp_path):
        trader = _trader(tmp_path)
        _open_one(trader)
        trader.ledger.arm_exit = lambda _p: (_ for _ in ()).throw(
            OSError("disk gone"))
        _close_now(trader)

        blob = json.dumps(_raw_lines(tmp_path, "journal/events.jsonl"),
                          default=str)
        assert "paper_close_persist_failed" in blob, (
            "the persistence failure left no durable trace outside the chat")

    def test_one_failed_arm_does_not_suppress_the_others_receipts(
            self, tmp_path, monkeypatch):
        """
        The alert filter is per settlement. A failure on one arm must not
        swallow the receipts for arms that DID commit.
        """
        trader = _trader(tmp_path)
        arms = _open_one(trader)
        real_write = trader.ledger.arm_exit
        victim = {"done": False}

        def flaky(payload):
            if not victim["done"]:
                victim["done"] = True
                raise OSError("one arm's write failed")
            return real_write(payload)

        trader.ledger.arm_exit = flaky
        _close_now(trader)

        texts = "\n".join(trader._test_notifications)
        assert "LEDGER WRITE FAILED" in texts
        durable = _arm_exits(tmp_path)
        assert len(durable) < arms, "the flaky write failed nothing"
        assert texts.count(CLEAN_CLOSE_RECEIPT) == len(durable), (
            "the durable records and the clean close receipts disagree: "
            f"{texts.count(CLEAN_CLOSE_RECEIPT)} receipts for {len(durable)} "
            "records. One alert opens every arm, so this only holds if the "
            "alert filter is keyed on (arm, signal_id) and not signal_id.")


# ---------------------------------------------------------------------------
# 2. the ledger query returns it -- the bug itself
# ---------------------------------------------------------------------------


class TestTheViewSeesArmSettlements:
    def test_closed_positions_shows_the_newly_settled_arm(self, tmp_path):
        """
        THE REGRESSION. ``closed_positions_view`` used to read only
        ``ledger.trades()`` (event == "exit"), so an arm settlement -- durable,
        notified, and gone from the operator's view -- returned nothing.
        """
        trader = _trader(tmp_path)
        arms = _open_one(trader, symbol="SNORP")
        _close_now(trader)

        rows, total = trader.closed_positions_view()
        assert total == arms, (
            "the arms settled and were recorded, but CLOSED POSITIONS does "
            f"not show them: total={total} expected={arms}")
        assert all(r["symbol"] == "SNORP" for r in rows)
        assert all(r["mode"] == TradeMode.PAPER.value for r in rows)

    def test_the_row_carries_the_numbers_the_receipt_carried(self, tmp_path):
        trader = _trader(tmp_path)
        _open_one(trader, symbol="SNORP")
        _close_now(trader)
        rows, _ = trader.closed_positions_view()

        row = rows[0]
        for field in ("symbol", "ca", "entry_at_iso", "exit_at_iso",
                      "entry_price", "exit_price", "realized_pnl_usd",
                      "return_pct", "hold_s", "exit_reason", "mfe_pct",
                      "mae_pct", "trade_id"):
            assert field in row, f"the closed-position view is missing {field}"
        # The FILLS, not the marks -- the same two prices the receipt quoted.
        assert float(row["entry_price"]) > 0
        assert float(row["exit_price"]) > 0
        assert row["entry_price_kind"] == "SIMULATED_FILL"
        assert row["exit_price_kind"] == "SIMULATED_FILL"
        # ...and the basis is stated, because these are NOT the legacy
        # mark-to-mark trades and must never be summed as if they were.
        assert row["pnl_basis"] == "FILL_VS_FILL"
        assert row["mfe_mae_basis"] == "OBSERVED_MARK"

    def test_the_row_renders_in_the_operator_message(self, tmp_path):
        from autopaper.telegram import ui

        trader = _trader(tmp_path)
        arms = _open_one(trader, symbol="SNORP")
        _close_now(trader)
        rows, total = trader.closed_positions_view()
        text = ui.render_closed_positions(rows, total=total)
        assert "CLOSED POSITIONS" in text
        assert "SNORP" in text
        assert f"of {arms}" in text
        assert "PAPER BUY" not in text, (
            "the close view rendered an entry row")

    def test_a_legacy_arm_exit_without_a_key_still_renders(self, tmp_path):
        """
        THE ON-DISK RECORDS MUST START APPEARING.

        The four settlements already written -- NI, UNIPCS, oushi and the SNORP
        incident itself -- were written by the code BEFORE the fix and
        therefore carry no ``settlement_key``. The reader must derive the key,
        or the historical records stay invisible after the upgrade.
        """
        trader = _trader(tmp_path)
        path = _var(tmp_path) / "ledger.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "event": "arm_exit", "at": 1_700_000_000.0,
                "arm": "L25_CURRENT", "mint": "MintSNORP", "symbol": "SNORP",
                "realized_pnl_usd": 0.09286949,
                "pnl_basis": "FILL_VS_FILL", "trigger": "data_invalid",
                "trade": {
                    "position": {"signal_id": "sig_old", "arm": "L25_CURRENT",
                                 "symbol": "SNORP", "mint": "MintSNORP",
                                 "opened_at": 1_699_999_800.0,
                                 "entry_fill": {"effective_price_usd": 1.6e-5},
                                 "OBSERVED_MARK_MFE": 20.2,
                                 "OBSERVED_MARK_mae": 0.0},
                    "simulated_sell_fill": {"effective_price_usd": 1.9e-5},
                    "realized_pnl": {"realized_pnl_usd": 0.09286949,
                                     "return_pct": 19.9,
                                     "hold_seconds": 124.8},
                    "exit_reason": "market data unusable",
                    "exit_trigger": "data_invalid",
                    "exit_reason_code": "data_invalid",
                    "failure_reason": "NONE",
                },
            }) + "\n")

        rows, total = trader.closed_positions_view()
        assert total == 1
        assert rows[0]["symbol"] == "SNORP"
        assert rows[0]["trade_id"] == "L25_CURRENT::sig_old"
        assert rows[0]["realized_pnl_usd"] == pytest.approx(0.09286949)
        assert rows[0]["return_pct"] == pytest.approx(19.9)


# ---------------------------------------------------------------------------
# 3. ordering, newest first
# ---------------------------------------------------------------------------


class TestNewestFirst:
    def test_the_newest_settlement_is_the_first_row(self, tmp_path):
        _cycle(tmp_path, symbol="FIRST", message_id=1)
        trader, arms = _cycle(tmp_path, symbol="SECOND", message_id=2)

        rows, total = trader.closed_positions_view()
        assert total == 2 * arms
        symbols = [r["symbol"] for r in rows]
        assert symbols == ["SECOND"] * arms + ["FIRST"] * arms, (
            f"CLOSED POSITIONS must be newest first, got {symbols}")
        exits = [r["exit_at"] for r in rows]
        assert exits == sorted(exits, reverse=True)

    def test_a_new_close_tops_the_ledger_over_old_history(self, tmp_path):
        """
        The exact shape of the incident: a ledger full of 2026-10-02 history
        that must not move, with the new settlement at the top.
        """
        for i in range(3):
            _seed_legacy_exit(tmp_path, trade_id=f"old_{i}", symbol=f"OLD{i}",
                              at=1_690_000_000.0 + i)

        trader = _trader(tmp_path)
        arms = _open_one(trader, symbol="SNORP")
        _close_now(trader)

        rows, total = trader.closed_positions_view()
        assert total == 3 + arms, "the history or the new close was dropped"
        assert {r["symbol"] for r in rows[:arms]} == {"SNORP"}, (
            "the newly closed position must be at the TOP, above the history")
        assert {r["symbol"] for r in rows} == {
            "SNORP", "OLD0", "OLD1", "OLD2"}

    def test_paper_and_real_settlements_share_one_reverse_chronology(
            self, tmp_path):
        trader = _trader(tmp_path)
        _seed_legacy_exit(tmp_path, trade_id="old", symbol="OLD",
                          at=1_690_000_000.0)
        arms = _open_one(trader, symbol="SNORP")
        _close_now(trader)
        rows, total = trader.closed_positions_view()
        assert total == arms + 1
        exits = [r["exit_at"] for r in rows]
        assert exits == sorted(exits, reverse=True)

    def test_pagination_still_windows_a_long_ledger(self, tmp_path):
        from autopaper.engine import CLOSED_PAGE_SIZE

        for i in range(CLOSED_PAGE_SIZE):
            _seed_legacy_exit(tmp_path, trade_id=f"p{i}", symbol=f"P{i}",
                              at=1_690_000_000.0 + i)
        trader, arms = _cycle(tmp_path, symbol="SNORP")
        rows, total = trader.closed_positions_view()
        assert total == CLOSED_PAGE_SIZE + arms
        assert len(rows) == CLOSED_PAGE_SIZE
        page2, _ = trader.closed_positions_view(offset=CLOSED_PAGE_SIZE)
        assert page2
        assert not ({r["trade_id"] for r in rows}
                    & {r["trade_id"] for r in page2}), (
            "the second page repeats the first")


# ---------------------------------------------------------------------------
# 4. idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_a_repeated_close_event_does_not_duplicate_the_row(self, tmp_path):
        trader = _trader(tmp_path)
        _open_one(trader)
        settled = _close_now(trader)
        before = len(_arm_exits(tmp_path))

        # THE SAME SETTLEMENTS, DELIVERED AGAIN.
        trader._paper_bridge.pending_messages.clear()
        durable = trader._record_arm_trades(settled)

        assert durable == [], (
            "a repeated close event was treated as a new settlement")
        assert len(_arm_exits(tmp_path)) == before, (
            "the duplicate close event appended second ledger rows")

    def test_a_duplicate_row_already_on_disk_still_renders_once(self, tmp_path):
        """
        The reader half of the same guarantee. The in-memory set dies with the
        process, so deduping on write alone would leave a replayed settlement
        visible twice after a restart.
        """
        trader = _trader(tmp_path)
        _open_one(trader, symbol="SNORP")
        settled = _close_now(trader)
        _rows, before = trader.closed_positions_view()

        # THE POST-RESTART CASE: a fresh process with no memory of the write,
        # replaying the same settlement from the dataset.
        trader._arm_settlement_keys = set()
        for trade in settled:
            trader.ledger.arm_exit({
                "arm": trade.position.arm,
                "mint": trade.position.mint,
                "symbol": trade.position.symbol,
                "realized_pnl_usd": 0.0,
            })
        assert len(_arm_exits(tmp_path)) > before, "the replay wrote nothing"

        rows, total = trader.closed_positions_view()
        assert total == before, (
            "two rows for one settlement: the reader is not idempotent")

    def test_the_receipt_is_sent_once_for_a_duplicate_event(self, tmp_path):
        trader = _trader(tmp_path)
        _open_one(trader)
        settled = _close_now(trader)
        before = len([t for t in trader._test_notifications
                      if CLEAN_CLOSE_RECEIPT in t])
        assert before

        bridge = trader._paper_bridge
        bridge.pending_messages.extend(bridge.render_exits(settled))
        trader._retain_arm_exit_alerts(trader._record_arm_trades(settled))
        trader._flush_paper_bridge_messages()

        after = len([t for t in trader._test_notifications
                     if CLEAN_CLOSE_RECEIPT in t])
        assert after == before, (
            "a duplicate close event re-announced a position already closed")

    def test_distinct_settlements_are_not_deduped_against_each_other(
            self, tmp_path):
        """
        The key must be specific enough. Collapsing two different positions
        into one row would be a worse bug than the one being fixed.
        """
        _cycle(tmp_path, symbol="AAA", message_id=1)
        trader, _ = _cycle(tmp_path, symbol="BBB", message_id=2)

        keys = [r["trade_id"] for r in trader.closed_positions_view()[0]]
        assert len(set(keys)) == len(keys), "two settlements shared a key"


# ---------------------------------------------------------------------------
# 5. restart / reload
# ---------------------------------------------------------------------------


class TestRestart:
    def test_the_closed_position_survives_a_restart(self, tmp_path):
        """
        No process state may be required to see it. The ledger is the record.
        """
        trader = _trader(tmp_path)
        arms = _open_one(trader, symbol="SNORP")
        _close_now(trader)

        # A SECOND runtime over the SAME var dir: what a restart reads.
        restarted = PaperTrader(
            build_config(tmp_path), quote_service=FakeQuoteService())
        rows, total = restarted.closed_positions_view()
        assert total == arms
        assert all(r["symbol"] == "SNORP" for r in rows)
        assert all(r["realized_pnl_usd"] is not None for r in rows)

    def test_the_snapshot_and_the_ledger_agree(self, tmp_path):
        """
        Two durable stores exist for this book. They must not disagree about
        whether the position closed.
        """
        trader = _trader(tmp_path)
        _open_one(trader, symbol="SNORP")
        _close_now(trader)

        _rows, total = trader.closed_positions_view()
        assert total > 0
        assert Path(trader.paper_state_path()).exists(), (
            "the resumable PAPER snapshot was not written")
        assert len(_arm_exits(tmp_path)) == total

    def test_a_restart_does_not_resurrect_the_closed_position(self, tmp_path):
        trader = _trader(tmp_path)
        _open_one(trader)
        _close_now(trader)
        restarted = PaperTrader(
            build_config(tmp_path), quote_service=FakeQuoteService())
        assert not run(restarted._paper_bridge_instance()).open_positions(), (
            "a settled position was resurrected by the restart")


# ---------------------------------------------------------------------------
# 6. no cache: the view is a pure function of the ledger
# ---------------------------------------------------------------------------


class TestNoStaleView:
    def test_the_same_engine_instance_sees_a_new_close_without_a_restart(
            self, tmp_path):
        trader = _trader(tmp_path)
        assert trader.closed_positions_view() == ([], 0)

        arms = _open_one(trader, symbol="SNORP")
        _close_now(trader)

        rows, total = trader.closed_positions_view()
        assert total == arms, (
            "the view is cached: a close on this very process is invisible "
            "until a restart")
        assert all(r["symbol"] == "SNORP" for r in rows)

    def test_the_view_tracks_the_ledger_file_not_a_snapshot(self, tmp_path):
        """
        Proof the reader IS the file: a record appended by anything at all
        shows up, with no engine involvement whatsoever. There is nothing to
        invalidate because there is no cache.
        """
        trader = _trader(tmp_path)
        _seed_legacy_exit(tmp_path, trade_id="t1", symbol="DIRECT",
                          at=1_700_000_500.0)
        rows, total = trader.closed_positions_view()
        assert total == 1
        assert rows[0]["symbol"] == "DIRECT"

        _seed_legacy_exit(tmp_path, trade_id="t2", symbol="SECOND",
                          at=1_700_000_600.0)
        rows, total = trader.closed_positions_view()
        assert total == 2, "the second append was not observed"
        assert rows[0]["symbol"] == "SECOND"

    def test_the_operator_button_path_reads_the_same_rows(self, tmp_path):
        """
        The Telegram button does not read the engine directly; it calls the
        ``extras`` entry. That indirection must not serve a different list.
        """
        from autopaper.telegram import ui

        trader = _trader(tmp_path)
        _open_one(trader, symbol="SNORP")
        _close_now(trader)

        extras = trader._operator_actions().extras
        assert extras["closed_positions"] == trader.closed_positions_view
        rows, total = extras["closed_positions"]()
        assert total > 0
        assert "SNORP" in ui.render_closed_positions(rows, total=total)


# ---------------------------------------------------------------------------
# 7. history is not disturbed
# ---------------------------------------------------------------------------


class TestHistory:
    def test_every_historical_closed_position_is_still_there(self, tmp_path):
        n = 12
        for i in range(n):
            _seed_legacy_exit(tmp_path, trade_id=f"h{i}", symbol=f"H{i}",
                              at=1_690_000_000.0 + i)
        trader = _trader(tmp_path)
        arms = _open_one(trader, symbol="SNORP")
        _close_now(trader)

        rows, total = _all_rows(trader)
        assert total == n + arms
        ids = {r["trade_id"] for r in rows}
        assert {f"h{i}" for i in range(n)} <= ids, (
            "an existing closed position was lost by the fix")

    def test_nothing_is_rewritten_or_deleted(self, tmp_path):
        """
        The ledger is append-only. A fix to a READER must not touch the bytes
        the operator already has.
        """
        _seed_legacy_exit(tmp_path, trade_id="h0", symbol="H0",
                          at=1_690_000_000.0)
        path = _var(tmp_path) / "ledger.jsonl"
        before = path.read_text()

        trader = _trader(tmp_path)
        _open_one(trader)
        _close_now(trader)
        trader.closed_positions_view()

        after = path.read_text()
        assert after.startswith(before), (
            "the ledger was rewritten; existing records must never be "
            "modified or removed")
        assert before in after


# ---------------------------------------------------------------------------
# 8. PAPER and REAL hold the same durable-state contract
# ---------------------------------------------------------------------------


class TestRealDurabilityContract:
    def test_save_real_state_reports_commit(self, tmp_path):
        """
        The REAL book is a SEPARATE store -- it must stay separate -- but it
        obeys the same rule: a caller about to claim a close must be able to
        learn whether the record committed.
        """
        trader = _trader(tmp_path)
        assert trader.save_real_state() is True
        assert Path(trader.real_state_file).exists()

    def test_save_real_state_reports_failure_instead_of_raising(self, tmp_path):
        trader = _trader(tmp_path)
        # A DIRECTORY where the file must go: the write cannot succeed, and
        # must be reported rather than silently swallowed.
        blocked = _var(tmp_path) / "blocked.json"
        blocked.mkdir()
        trader.real_state_file = blocked

        assert trader.save_real_state() is False, (
            "a failed durable write must be reported, never silently "
            "swallowed -- otherwise the caller cannot tell a persisted close "
            "from an in-memory one")

    def test_real_and_paper_still_use_different_stores(self, tmp_path):
        trader = _trader(tmp_path)
        _open_one(trader, symbol="SNORP")
        _close_now(trader)

        _rows, total = trader.closed_positions_view()
        assert total > 0
        assert not trader.real_account.open_positions(), (
            "a PAPER settlement leaked into the REAL account")
        assert all(r["mode"] == TradeMode.PAPER.value
                   for r in trader.closed_positions_view()[0])

    def test_the_real_close_receipt_is_gated_on_persistence(self, tmp_path):
        """
        STATIC CLAIM, asserted structurally. The REAL success receipt must sit
        behind the durable write's result; otherwise a REAL close can report
        success with nothing on disk.
        """
        source = Path(__file__).resolve().parents[1] / "src" / "autopaper" \
            / "engine.py"
        text = source.read_text(encoding="utf-8")
        start = text.index("REAL_SELL=SUCCESS")
        window = text[max(0, start - 4000):start]
        assert "durable = self.save_real_state()" in window, (
            "the REAL close no longer captures the durable write's result")
        assert "if not durable:" in window, (
            "the REAL success receipt is no longer gated on the durable write")


# ---------------------------------------------------------------------------
# 9. the ledger reader itself
# ---------------------------------------------------------------------------


class TestLedgerReader:
    def test_trades_and_arm_exits_stay_separate_accounting_bases(self, tmp_path):
        """
        They are SHOWN together and summed by nobody. ``trades()`` must keep
        returning only legacy EXIT records, or the statistics start mixing a
        mark-to-mark book with a fill-vs-fill book.

        This used to assert ``len(arm_exits()) > 1``, which asserted the arm
        COUNT rather than the separation -- and the arm count was three only
        because an unrestored strategy selection used to fan out to every
        registered arm. The separation is what is under test, so it is asserted
        directly: every arm exit is an arm exit, and none of them appears in the
        legacy book.
        """
        _seed_legacy_exit(tmp_path, trade_id="legacy", symbol="L",
                          at=1_690_000_000.0)
        trader = _trader(tmp_path)
        arms = _open_one(trader)
        _close_now(trader)

        ledger = trader.ledger
        assert len(ledger.trades()) == 1
        arm_exits = ledger.arm_exits()
        assert len(arm_exits) == arms, (
            "every arm that filled the BUY must have exactly one settlement")
        assert all(row["event"] == "arm_exit" for row in arm_exits)
        assert all("settlement_key" in row for row in arm_exits)
        legacy = ledger.trades()
        assert all("settlement_key" not in row for row in legacy), (
            "an arm settlement leaked into the legacy book")
        paper_stats, _real = trader.statistics_by_mode()
        assert paper_stats["total_trades"] == 1, (
            "the arm settlements leaked into the legacy statistics")

    def test_arm_exits_are_read_in_append_order(self, tmp_path):
        for i, sym in enumerate(("AAA", "BBB", "CCC")):
            _cycle(tmp_path, symbol=sym, message_id=10 + i)
        ats = [r["at"] for r in _arm_exits(tmp_path)]
        assert ats == sorted(ats)

    def test_arm_exits_is_empty_for_a_ledger_with_no_arms(self, tmp_path):
        _seed_legacy_exit(tmp_path, trade_id="l", symbol="L",
                          at=1_690_000_000.0)
        trader = _trader(tmp_path)
        assert trader.ledger.arm_exits() == []

    def test_a_failed_arm_settlement_is_not_listed_as_closed(self, tmp_path):
        """
        A settlement that did not happen has no closed position to show and
        must not reach the ledger at all.
        """
        trader = _trader(tmp_path)
        _open_one(trader)
        settled = _close_now(trader)
        broken = dataclasses.replace(
            settled[0], failure=type("F", (), {"value": "NO_EXECUTABLE_QUOTE"})())
        trader._arm_settlement_keys = set()
        trader.ledger.arm_exit = lambda _p: pytest.fail(
            "a failed settlement must not be recorded as closed")

        assert trader._record_arm_trades([broken]) == []

    def test_the_view_ignores_a_record_whose_trade_payload_is_absent(
            self, tmp_path):
        """
        A torn or hand-written record must not crash the operator's view.
        """
        trader = _trader(tmp_path)
        path = _var(tmp_path) / "ledger.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": "arm_exit", "at": 1.0}) + "\n")
        rows, total = trader.closed_positions_view()
        assert total == 0
        assert rows == []
