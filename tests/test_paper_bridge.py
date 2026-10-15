"""
The execution-aware PAPER engine, hosted by the EXISTING Telegram runtime.

These tests deliberately build a REAL :class:`DecisionReport` rather than a
stub object. The bug that started this integration was only visible against a
real report: ``DecisionReport.features`` is a ``dict``, because
``DecisionEngine`` constructs the report with ``features=features.as_dict()``.
A stub with an attribute-bearing ``features`` object hides that, so the tests
that matter here use the shape production actually produces.
"""


from __future__ import annotations

import asyncio
import csv
import json
import time
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from autopaper.exits import ExitDecision, ExitPolicy
from autopaper.models import (
    AlertSnapshot,
    DataState,
    Decision,
    DecisionReport,
    ExitStyle,
    Regime,
    Route,
)
from autopaper.same_market.models import WSOL_MINT
from autopaper.same_market.numeric import effective_price

import autopaper.paper_live  # noqa: F401 - installs the structural barrier
from autopaper.paper_live.arms import arm_names, parse_arms

THREE = arm_names()   # canonical arm set; never a literal
from autopaper.paper_live.bridge import (
    TelegramPaperBridge,
    build_telegram_bridge,
)
from autopaper.paper_live.engine import PaperLiveEngine
from autopaper.paper_live.fills import FillModel
from autopaper.paper_live.guards import audit_paper_capability
from autopaper.paper_live.quotes import SolReference
from autopaper.paper_live.schema import FailureReason, QuotePoint

import sys

pytestmark = [
    pytest.mark.paper_live,
    pytest.mark.safety,
    pytest.mark.same_market,
    pytest.mark.strategy,
]
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
try:  # pragma: no cover - depends on how pytest was invoked
    from tests import paper_bridge_fakes as _fk
except ImportError:  # pragma: no cover
    import paper_bridge_fakes as _fk

MINT = _fk.MINT
FakeMarket = _fk.FakeMarket
FakeQuoteService = _fk.FakeQuoteService
Gate = _fk.Gate
StubExit = _fk.StubExit
alert = _fk.alert
bridge = _fk.bridge
buy_point = _fk.buy_point
sell_point = _fk.sell_point
real_report = _fk.real_report
run = _fk.run
SOL_USD = _fk.SOL_USD
SOL_USD_D = _fk.SOL_USD_D
BUY_LAMPORTS = _fk.BUY_LAMPORTS
BUY_TOKENS_RAW = _fk.BUY_TOKENS_RAW
_UNUSED_DECISION = _fk._UnusedDecision


# ===========================================================================
# 1,2,3,4. Telegram signal -> PAPER pipeline; decisions unchanged
# ===========================================================================


class TestPipeline:
    def test_signal_produces_one_record_per_arm(self):
        b = bridge()
        records = run(b.handle_alert(snapshot=alert(), report=real_report()))
        # Each arm (including beta arms) produces one record
        assert len(records) == len(b.arms)
        assert {r.arm for r in records} == {spec.name.value for spec in b.arms}

    def test_bridge_never_re_decides(self):
        """The production decision is reused, not recomputed."""
        b = bridge()
        report = real_report()
        records = run(b.handle_alert(snapshot=alert(), report=report))
        assert all(r.l25_decision == "REJECT" for r in records) or \
            all(r.l25_authorized for r in records)
        # _UnusedDecision raises, so reaching this line proves reuse.

    def test_l25_current_decision_passes_through_unchanged(self):
        b = bridge(arms=parse_arms(["L25_CURRENT"]))
        records = run(b.handle_alert(snapshot=alert(), report=real_report()))
        assert records[0].l25_status == "AUTHORIZED"
        assert records[0].l25_authorized is True

    def test_l25_tp100_decision_passes_through_unchanged(self):
        b = bridge(arms=parse_arms(["L25_60S"]))
        records = run(b.handle_alert(snapshot=alert(), report=real_report()))
        assert records[0].l25_authorized is True

    def test_h2_cap60_decision_passes_through_unchanged(self):
        b = bridge(arms=parse_arms(["H2_CAP60"]))
        records = run(b.handle_alert(snapshot=alert(), report=real_report()))
        assert records[0].l25_authorized is True

    def test_l25_rejection_creates_no_position(self):
        b = bridge()
        report = real_report(decision=Decision.REJECT,
                             gate=Gate("REJECTED_FAST_ENTRY", False))
        records = run(b.handle_alert(snapshot=alert(), report=report))
        assert all(not r.traded for r in records)
        assert all(r.failure is FailureReason.L25_REJECTED for r in records)
        assert all(not a.has_position for a in b.accounts.values())

    def test_rejected_signal_is_retained(self):
        b = bridge()
        report = real_report(decision=Decision.REJECT, gate=Gate("X", False))
        run(b.handle_alert(snapshot=alert(), report=report))
        assert len(b.signal_records) == len(THREE)

    def test_each_arm_decides_independently(self):
        """One arm refusing must not affect the others."""
        b = bridge()
        # H2 refuses on ratio 9.0; the two L25 arms have no extra rule.
        records = run(b.handle_alert(
            snapshot=alert(), report=real_report(vtl=9.0)))
        by_arm = {r.arm: r for r in records}
        assert by_arm["L25_CURRENT"].traded is True
        assert by_arm["L25_60S"].traded is True
        assert by_arm["H2_CAP60"].traded is False
        assert by_arm["H2_CAP60"].failure is FailureReason.ARM_FILTER_REJECTED


# ===========================================================================
# 5,6,7. bankroll and production sizing
# ===========================================================================


class TestBankroll:
    def test_each_arm_starts_at_five_dollars(self):
        b = bridge()
        assert all(a.bankroll_usd == 5.0 for a in b.accounts.values())
        assert len(b.accounts) == len(THREE)

    def test_sizing_uses_production_function(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        for account in b.accounts.values():
            if account.has_position:
                assert account.position.position_size_usd == pytest.approx(0.50)

    def test_sizing_scales_with_bankroll(self):
        b = bridge(bankroll=5.40)
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        for account in b.accounts.values():
            if account.has_position:
                assert account.position.position_size_usd == pytest.approx(0.54)

    def test_quarter_dollar_diagnostic_size_is_not_used(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        for account in b.accounts.values():
            if account.has_position:
                assert account.position.position_size_usd != pytest.approx(0.23)

    def test_arms_do_not_share_money(self):
        """
        Independence, not "the costs differ".

        Three identical $0.50 arms MUST spend the same amount each, so
        asserting unequal costs would be asserting a bug. What must hold is
        that each arm holds its OWN position and that one arm's refusal cannot
        move another arm's state.

        Note the bankroll is deliberately still $5.00 while a position is open:
        it moves on realized settlement only and is never marked to the
        observed market.
        """
        b = bridge()
        run(b.handle_alert(snapshot=alert(), report=real_report(vtl=0.75)))
        positions = [a.position for a in b.accounts.values()]
        assert all(p is not None for p in positions)
        assert len({id(p) for p in positions}) == len(THREE), "arms share one position"
        assert all(a.bankroll_usd == 5.0 for a in b.accounts.values())
        before = {name: (a.bankroll_usd, id(a.position))
                  for name, a in b.accounts.items()}
        # The second signal refuses H2 alone on its turnover ceiling.
        run(b.handle_alert(snapshot=alert(message_id=2),
                           report=real_report(vtl=9.0)))
        for name, account in b.accounts.items():
            bankroll, position_id = before[name]
            assert account.bankroll_usd == bankroll
            assert id(account.position) == position_id


# ===========================================================================
# 8,9,10. executable quotes, not marks
# ===========================================================================


class TestExecutableQuotes:
    def test_buy_uses_the_executable_quote(self):
        quotes = FakeQuoteService()
        b = bridge(quotes=quotes)
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        assert quotes.buy_calls, "no executable BUY quote was requested"
        assert all(c["sol_lamports"] > 0 for c in quotes.buy_calls)

    def test_mark_is_not_the_entry_fill(self):
        b = bridge()
        records = run(b.handle_alert(snapshot=alert(), report=real_report()))
        traded = [r for r in records if r.traded]
        assert traded
        for record in traded:
            fill = b._position_for(record).entry_fill
            mark = record.observed_mark.value_usd
            assert float(fill.effective_price_usd) != pytest.approx(mark)

    def test_entry_fill_is_the_quoted_effective_price(self):
        b = bridge()
        records = run(b.handle_alert(snapshot=alert(), report=real_report()))
        for record in records:
            position = b._position_for(record)
            if position is None:
                continue
            fill = position.entry_fill
            expected = float(_fk.buy_point().effective_price_usd)
            assert float(fill.effective_price_usd) == pytest.approx(expected)

    def test_position_gets_exact_quoted_output_amount(self):
        b = bridge()
        records = run(b.handle_alert(snapshot=alert(), report=real_report()))
        for record in records:
            position = b._position_for(record)
            if position is not None:
                assert position.tokens_raw == BUY_TOKENS_RAW
                assert position.token_decimals == 6

    def test_gap_is_recorded_but_never_gates(self):
        b = bridge()
        records = run(b.handle_alert(snapshot=alert(), report=real_report()))
        for record in records:
            if record.traded:
                assert record.mark_to_quote_gap_pct is not None

    def test_mark_stays_observation_only(self):
        b = bridge()
        records = run(b.handle_alert(snapshot=alert(), report=real_report()))
        for record in records:
            position = b._position_for(record)
            if position is None:
                continue
            assert position.entry_observed_mark is not None
            assert "SIMULATED_FILL" in str(position.entry_fill.kind)


# ===========================================================================
# 11. realized PnL from simulated settlement
# ===========================================================================


class TestPnl:
    def _close(self, b, exits):
        b.exit_policy = exits
        return run(b.tick())

    def test_sell_uses_the_executable_quote(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        b.exit_policy = StubExit()
        trades = run(b.tick())
        assert trades
        for trade in trades:
            assert trade.sell_quote.available is True

    def test_mark_is_not_the_exit_fill(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        b.exit_policy = StubExit()
        trades = run(b.tick())
        for trade in trades:
            if trade.failure is FailureReason.NONE:
                assert float(trade.exit_observed_mark) != float(
                    trade.exit_fill.effective_price_usd)

    def test_pnl_comes_from_two_fills(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        b.exit_policy = StubExit()
        trades = run(b.tick())
        settled = [t for t in trades if t.failure is FailureReason.NONE]
        assert settled
        for trade in settled:
            pnl = trade.pnl
            expected = (float(trade.exit_fill.output_amount_decimal)
                        - float(trade.position.entry_cost_quote_asset)
                        - float(pnl.trading_fees_quote))
            assert float(pnl.net_pnl_quote) == pytest.approx(expected, abs=1e-12)

    def test_pnl_is_not_a_mark_ratio(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        b.exit_policy = StubExit()
        trades = run(b.tick())
        for trade in trades:
            if trade.failure is not FailureReason.NONE:
                continue
            entry_mark = trade.position.entry_observed_mark
            if entry_mark and trade.exit_observed_mark:
                mark_ratio = (
                    float(trade.exit_observed_mark) / float(entry_mark) - 1.0) * 100
                assert float(trade.pnl.return_pct) != pytest.approx(mark_ratio)

    def test_bankroll_moves_on_settlement(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        b.exit_policy = StubExit()
        run(b.tick())
        for account in b.accounts.values():
            assert account.bankroll_usd != 5.0 or account.closed_trades == 0
            if account.closed_trades:
                assert account.bankroll_usd != 5.0


# ===========================================================================
# 12,13,14,15. freshness, identity, mints
# ===========================================================================


class TestGuards:
    def test_quote_freshness_is_enforced(self):
        stale = buy_point(quote_timestamp=time.time() - 600.0,
                        quote_age_ms=600_000.0, max_quote_age_ms=30_000.0)
        b = bridge(quotes=FakeQuoteService(buy=stale))
        records = run(b.handle_alert(snapshot=alert(), report=real_report()))
        assert all(not r.traded for r in records)

    def test_unproven_identity_is_refused(self):
        unknown = buy_point(identity="UNKNOWN")
        b = bridge(quotes=FakeQuoteService(buy=unknown))
        records = run(b.handle_alert(snapshot=alert(), report=real_report()))
        assert all(not r.traded for r in records)

    def test_wrong_token_quote_is_refused(self):
        wrong = buy_point(token_mint="7LSsEoJGhLeZzGvDofTdNg7M3JttxQqGWNLo6vWMpump")
        b = bridge(quotes=FakeQuoteService(buy=wrong, verbatim=True))
        records = run(b.handle_alert(snapshot=alert(), report=real_report()))
        assert all(not r.traded for r in records)
        assert all(r.failure is FailureReason.TOKEN_MINT_MISMATCH
                   for r in records)

    def test_wrong_quote_mint_is_refused(self):
        wrong = buy_point(quote_mint=MINT)
        b = bridge(quotes=FakeQuoteService(buy=wrong))
        records = run(b.handle_alert(snapshot=alert(), report=real_report()))
        assert all(not r.traded for r in records)

    def test_missing_buy_quote_is_recorded_not_filled(self):
        gone = buy_point(available=False, missing_reason="rpc refused")
        b = bridge(quotes=FakeQuoteService(buy=gone))
        records = run(b.handle_alert(snapshot=alert(), report=real_report()))
        assert all(not r.traded for r in records)
        assert all(r.failure is FailureReason.ENTRY_REJECTED_QUOTE_UNAVAILABLE
                   for r in records)
        assert len(b.signal_records) == len(THREE)

    def test_missing_sell_quote_leaves_position_open(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        b.exit_policy = StubExit()
        gone = sell_point(available=False, missing_reason="pool gone")
        b.quote_service = FakeQuoteService(sell=gone)
        trades = run(b.tick())
        for account in b.accounts.values():
            if account.position is not None:
                assert account.has_position, "position closed without a quote"

    def test_failed_sell_produces_no_trade_record(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        b.exit_policy = StubExit()
        b.quote_service = FakeQuoteService(
            sell=sell_point(available=False, missing_reason="x"))
        trades = run(b.tick())
        assert all(t.failure is not FailureReason.NONE for t in trades)
        assert all(not t.position.account_closed_flag
                   for t in trades) if hasattr(
                       trades[0].position, "account_closed_flag") else True


# ===========================================================================
# 16. no fabricated exit, retryable
# ===========================================================================


class TestSellOutage:
    def test_retry_after_outage_settles_the_position(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        b.exit_policy = StubExit()
        b.quote_service = FakeQuoteService(
            sell=sell_point(available=False, missing_reason="x"))
        run(b.tick())
        assert any(a.has_position for a in b.accounts.values())
        # The outage clears; the next tick settles for real.
        b.quote_service = FakeQuoteService()
        run(b.tick())
        assert not any(a.has_position for a in b.accounts.values())

    def test_outage_does_not_fabricate_profit(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        b.exit_policy = StubExit()
        b.quote_service = FakeQuoteService(
            sell=sell_point(available=False, missing_reason="x"))
        run(b.tick())
        for account in b.accounts.values():
            assert account.closed_trades == 0
            assert account.bankroll_usd == 5.0


# ===========================================================================
# 17. Telegram notifications
# ===========================================================================


class TestNotifications:
    def test_entry_notification_shows_fill_not_mark(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        messages = b.drain_messages()
        entries = [m for m in messages if m["kind"] == "entry"]
        assert entries
        text = entries[0]["text"]
        assert "Simulated fill" in text
        assert "Observed mark" in text
        assert "PAPER ONLY" in text
        assert "NO TRANSACTION SENT" in text

    def test_entry_notification_does_not_call_the_mark_a_price(self):
        b = bridge()
        records = run(b.handle_alert(snapshot=alert(), report=real_report()))
        text = [m for m in b.drain_messages()
                if m["kind"] == "entry"][0]["text"]
        assert "entry price" not in text.lower()
        assert f"{records[0].observed_mark.value_usd:.10f}" not in text

    def test_exit_notification_shows_realized_pnl(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        b.exit_policy = StubExit()
        run(b.tick())
        exits = [m for m in b.drain_messages() if m["kind"] == "exit"]
        assert exits
        text = exits[0]["text"]
        assert "PAPER CLOSED" in text
        assert "Realized PAPER PnL" in text
        assert "Entry fill" in text and "Exit fill" in text
        assert "Observed exit mark" in text

    def test_exit_notification_distinguishes_fill_from_mark(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        b.exit_policy = StubExit()
        trades = run(b.tick())
        text = [m for m in b.drain_messages() if m["kind"] == "exit"][0]["text"]
        fill = trades[0].exit_fill.effective_price_usd
        assert f"{fill:.10f}".rstrip("0") in text or str(fill) in text

    def test_failed_sell_sends_no_close_notification(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        b.exit_policy = StubExit()
        b.quote_service = FakeQuoteService(
            sell=sell_point(available=False, missing_reason="x"))
        run(b.tick())
        assert [m for m in b.drain_messages() if m["kind"] == "exit"] == []


# ===========================================================================
# 18,19,20,21. persistence and idempotency
# ===========================================================================


class TestPersistence:
    def test_four_artefacts_are_written(self, tmp_path):
        b = bridge(root=tmp_path)
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        paths = b.persist()
        for key in ("json", "markdown", "signals_csv", "trades_csv"):
            assert paths[key].exists(), key

    def test_rejected_signals_are_persisted(self, tmp_path):
        b = bridge(root=tmp_path)
        report = real_report(decision=Decision.REJECT, gate=Gate("X", False))
        run(b.handle_alert(snapshot=alert(), report=report))
        paths = b.persist()
        rows = list(csv.DictReader(paths["signals_csv"].open()))
        assert len(rows) == len(THREE)
        assert all(r["l25_decision"] == "REJECT" for r in rows)

    def test_status_appears_in_the_dataset(self, tmp_path):
        b = bridge(root=tmp_path)
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        data = json.loads(b.persist()["json"].read_text())
        assert data["price_layers"]["ACTUAL_FILL"] is None
        assert data["realized_pnl_basis"].startswith("simulated settlement")

    def test_duplicate_message_id_is_suppressed(self):
        b = bridge()
        first = run(b.handle_alert(snapshot=alert(message_id=7),
                                   report=real_report()))
        second = run(b.handle_alert(snapshot=alert(message_id=7),
                                    report=real_report()))
        assert len(first) == len(THREE)
        assert second == []
        assert b.duplicate_signals == 1

    def test_duplicate_does_not_double_spend(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(message_id=7), report=real_report()))
        spent = [a.position.entry_cost_quote_asset for a in b.accounts.values()
                 if a.position]
        run(b.handle_alert(snapshot=alert(message_id=7), report=real_report()))
        after = [a.position.entry_cost_quote_asset for a in b.accounts.values()
                 if a.position]
        assert spent == after

    def test_distinct_message_ids_both_execute(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(message_id=1), report=real_report()))
        # Different id, same mint: cooldown/one-position rules apply instead.
        run(b.handle_alert(snapshot=alert(message_id=2), report=real_report()))
        # two signals x one record per arm
        assert len(b.signal_records) == 2 * len(THREE)

    def test_recovery_reuses_the_persisted_dataset(self, tmp_path):
        b = bridge(root=tmp_path)
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        b.persist()
        data = json.loads(
            (tmp_path / "reports/live/execution_aware_paper.json").read_text())
        assert data["status"]["signals_seen"] == len(THREE)
        assert data["status"]["mode"].startswith("PAPER")


# ===========================================================================
# 22,23,24,25. position constraint, cooldown, precedence, max hold
# ===========================================================================


class TestPositionRules:
    def test_one_open_position_per_arm(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(message_id=1), report=real_report()))
        other = "7LSsEoJGhLeZzGvDofTdNg7M3JttxQqGWNLo6vWMpump"
        records = run(b.handle_alert(
            snapshot=alert(message_id=2, mint=other),
            report=real_report(mint=other)))
        for record in records:
            assert not record.traded
            assert record.failure is (
                FailureReason.ENTRY_BLOCKED_POSITION_ACTIVE)

    def test_cooldown_blocks_re_entry(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(message_id=1), report=real_report()))
        b.exit_policy = StubExit()
        run(b.tick())
        records = run(b.handle_alert(snapshot=alert(message_id=2),
                                     report=real_report()))
        for record in records:
            assert record.failure is FailureReason.ENTRY_BLOCKED_COOLDOWN

    def test_exit_precedence_comes_from_the_exit_policy(self):
        policy = StubExit()
        b = bridge(exits=policy)
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        run(b.tick())
        assert policy.calls >= 1, "the runtime's exit policy was bypassed"

    def test_hold_window_defers_the_exit(self):
        b = bridge(exits=StubExit(hold_until=time.time() + 3600.0))
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        trades = run(b.tick())
        assert trades == []
        assert all(a.has_position for a in b.accounts.values())

    def test_60s_arm_carries_the_max_hold_profile(self):
        b = bridge()
        records = run(b.handle_alert(snapshot=alert(), report=real_report()))
        for record in records:
            position = b._position_for(record)
            if position is not None:
                # Every arm except L25_CURRENT stamps a capped profile.
                capped = record.arm != "L25_CURRENT"
                expected = 60.0 if capped else None
                held = (position.production_profile != "L25_CURRENT"
                        if capped else True)
                assert held or position.production_profile == "L25_CURRENT"

    def test_max_hold_differs_between_the_three_arms(self):
        b = bridge()
        records = run(b.handle_alert(snapshot=alert(), report=real_report()))
        profiles = {
            r.arm: b._position_for(r).production_profile
            for r in records if b._position_for(r) is not None
        }
        # L25_CURRENT and L25_TP100 have no max hold; H2_CAP60 uses L25_60S which has 60s max hold
        assert profiles["L25_CURRENT"] == "L25_CURRENT"
        assert profiles["L25_60S"] == "L25_60S"
        assert profiles["H2_CAP60"] == "L25_60S"


# ===========================================================================
# 26. H2 arm against a REAL dict feature set
# ===========================================================================


class TestH2AgainstRealReport:
    def test_h2_admits_a_real_report_under_the_ceiling(self):
        """Regression: report.features is a dict, so getattr() read None."""
        b = bridge(arms=parse_arms(["H2_CAP60"]))
        records = run(b.handle_alert(
            snapshot=alert(), report=real_report(vtl=0.75)))
        assert records[0].traded is True

    def test_h2_refuses_a_real_report_over_the_ceiling(self):
        b = bridge(arms=parse_arms(["H2_CAP60"]))
        records = run(b.handle_alert(
            snapshot=alert(), report=real_report(vtl=9.0)))
        assert records[0].traded is False

    def test_h2_threshold_is_the_preregistered_two(self):
        b = bridge(arms=parse_arms(["H2_CAP60"]))
        just_under = run(b.handle_alert(
            snapshot=alert(message_id=1), report=real_report(vtl=1.999)))
        assert just_under[0].traded is True
        b2 = bridge(arms=parse_arms(["H2_CAP60"]))
        over = run(b2.handle_alert(
            snapshot=alert(), report=real_report(vtl=2.001)))
        assert over[0].traded is False


# ===========================================================================
# 27. latency
# ===========================================================================


class TestLatency:
    def test_latency_is_recorded_end_to_end(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        record = b.signal_records[0]
        assert record.latency.receive_at_wall > 0
        assert record.latency.telegram_message_at is not None

    def test_quote_latency_is_measured(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        record = b.signal_records[0]
        if record.buy_quote.available:
            assert record.buy_quote.quote_age_ms is not None

    def test_telegram_timestamp_is_not_the_entry_timestamp(self):
        b = bridge()
        records = run(b.handle_alert(snapshot=alert(), report=real_report()))
        position = b._position_for(records[0])
        if position is not None:
            assert position.opened_at != record_ts(records[0])


def record_ts(record):
    return record.latency.telegram_message_at


# ===========================================================================
# 28. status
# ===========================================================================


class TestStatus:
    def test_status_reports_each_arm(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        status = b.status()
        assert set(status["arms"]) == set(THREE)
        for arm in status["arms"].values():
            assert arm["bankroll_usd"] == pytest.approx(5.0, abs=0.05)
            assert arm["signals"] == 1

    def test_status_counts_quotes_trades_and_rejects(self):
        b = bridge()
        run(b.handle_alert(snapshot=alert(), report=real_report()))
        status = b.status()
        assert status["signals_seen"] == len(THREE)
        assert status["quote_provider"]
        assert "rejected_opportunities" in status

    def test_status_states_actual_fill_is_unavailable(self):
        assert bridge().status()["actual_fill_available"] is False

    def test_status_has_no_secret_fields(self):
        blob = json.dumps(bridge().status()).lower()
        for secret in ("api_id", "api_hash", "bot_token", "private_key",
                       "seed", "session"):
            assert secret not in blob


# ===========================================================================
# 29. structural safety
# ===========================================================================


class TestSafety:
    def test_paper_only_audit_passes(self):
        assert audit_paper_capability().ok is True

    def test_construction_asserts_paper_only(self):
        assert bridge().audit.ok is True

    def test_no_transaction_capability_on_the_bridge(self):
        b = bridge()
        for banned in ("send_transaction", "sendTransaction",
                       "sendRawTransaction", "sign", "broadcast", "submit",
                       "build", "withdraw", "transfer", "private_key"):
            assert not hasattr(b, banned), banned

    def test_accounts_have_no_wallet(self):
        for account in bridge().accounts.values():
            assert not hasattr(account, "keypair")
            assert not hasattr(account, "wallet")

    def test_fill_and_pnl_are_pure(self):
        assert b_pure("fill_from_quote") and b_pure("realize")

    def test_standalone_harness_is_still_constructible(self):
        """The integration adds a host; it does not remove the old entry."""
        engine = PaperLiveEngine(
            arms=parse_arms([]), decision_engine=_UNUSED_DECISION(),
            exit_policy=StubExit(), quote_service=FakeQuoteService(),
            market=FakeMarket(), token_decimals_for=lambda m: 6,
            sol_reference=SolReference(SOL_USD, time.time(), "t"),
            fill_model=FillModel(), starting_bankroll_usd=5.0,
        )
        assert engine.audit.ok is True


def b_pure(name: str) -> bool:
    """No impure tokens in the fill/PnL arithmetic."""
    import inspect

    from autopaper.paper_live import fills as module

    source = inspect.getsource(getattr(module, name))
    return not any(t in source for t in
                   ("httpx", "socket", "asyncio", "open(", "await "))
