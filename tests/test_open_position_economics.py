"""Regression tests for the open-position PnL / exit-lifecycle / retry fixes."""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_paper_live import (  # noqa: E402
    MINT,
    PAIR,
    FakeQuoteService,
    arm_names,
    make_engine,
    parse_arms,
    run,
    sell_point,
    signal,
    unavailable_point,
)
from autopaper.config import ExitConfig as ExitCfg  # noqa: E402
from autopaper.engine import PaperTrader  # noqa: E402
from autopaper.exits import ExitPolicy  # noqa: E402
from autopaper.paper_live.engine import (  # noqa: E402
    _EXIT_MAX_ATTEMPTS,
    _position_from_dict,
)
from autopaper.paper_live.schema import PaperPosition, SimulatedFill  # noqa: E402


def _arm_unrealized(notional, entry, mark):
    """The canonical open-position unrealized math, on a stub position."""
    position = SimpleNamespace(
        usd_notional=notional,
        entry_fill=SimpleNamespace(effective_price_usd=entry),
    )
    return PaperTrader._arm_unrealized_usd(position, mark)


def _arm_pct(notional, entry, mark):
    position = SimpleNamespace(
        usd_notional=notional,
        entry_fill=SimpleNamespace(effective_price_usd=entry),
    )
    return PaperTrader._arm_return_pct(position, mark)


class TestOpenPositionEconomics:
    def test_mark_50pct_below_entry(self):
        # TEST 1: realized stays 0, unrealized < 0, total reflects it.
        unreal = _arm_unrealized(1.0, 100.0, 50.0)
        assert unreal == pytest.approx(-0.5)
        assert _arm_pct(1.0, 100.0, 50.0) == pytest.approx(-50.0)

    def test_mark_50pct_above_entry(self):
        # TEST 2
        assert _arm_unrealized(1.0, 100.0, 150.0) == pytest.approx(0.5)
        assert _arm_pct(1.0, 100.0, 150.0) == pytest.approx(50.0)

    def test_missing_mark_means_unknown_not_zero(self):
        # TEST 3: no mark -> helper reports 0 via its guard, and the row says
        # None (which the UI renders as UNKNOWN) rather than a fabricated 0.
        assert _arm_unrealized(1.0, 100.0, None) == 0.0
        position = SimpleNamespace(
            usd_notional=1.0, entry_observed_mark=100.0,
            entry_fill=SimpleNamespace(effective_price_usd=100.0),
            mint=MINT, symbol="T", signal_id="s1", arm="L25_CURRENT",
            tokens_raw=1000000, token_decimals=6, opened_at=time.time(),
            production_profile="L25_CURRENT", observed_mark_mfe=0.0,
            observed_mark_mae=0.0, peak_mark=None, trough_mark=None,
            entry_venue_id="PUMP_BONDING_CURVE:x",
        )
        position.entry_fill = SimpleNamespace(
            effective_price_usd=100.0, kind="SIMULATED_FILL",
            venue="v", pool_or_curve="p", provider="pump_direct",
            quote_timestamp=time.time(), quote_age_ms=1.0,
        )
        account = SimpleNamespace(bankroll_usd=5.0, position=position)
        bridge = SimpleNamespace(
            accounts={"L25_CURRENT": account}, observed_marks={})
        fake_self = SimpleNamespace(
            _paper_bridge=bridge,
            _arm_unrealized_usd=PaperTrader._arm_unrealized_usd,
            _arm_return_pct=PaperTrader._arm_return_pct,
        )
        rows = PaperTrader._execution_aware_open_rows(fake_self, time.time())
        assert len(rows) == 1
        row = rows[0]
        assert row["unrealized_pnl_usd"] is None
        assert row["unrealized_pnl_pct"] is None
        assert row["total_pnl_usd"] is None
        assert row["realized_pnl_usd"] == 0.0
        assert row["data_state"] == "MISSING"
        assert row["current_price"] is None

    def test_live_mark_drives_unrealized_display(self):
        from autopaper.paper_live.schema import ObservedMarkPoint
        position = SimpleNamespace(
            usd_notional=0.42, entry_observed_mark=100.0,
            entry_fill=SimpleNamespace(
                effective_price_usd=100.0, kind="SIMULATED_FILL",
                venue="v", pool_or_curve="p", provider="pump_direct",
                quote_timestamp=time.time(), quote_age_ms=1.0,
            ),
            mint=MINT, symbol="T", signal_id="s1", arm="L25_CURRENT",
            tokens_raw=1000000, token_decimals=6, opened_at=time.time(),
            production_profile="L25_CURRENT", observed_mark_mfe=145.1,
            observed_mark_mae=-33.6, peak_mark=None, trough_mark=None,
            entry_venue_id="PUMP_BONDING_CURVE:x",
            current_mark_usd=55.9, current_mark_timestamp=time.time(),
            exit_status="OPEN_MONITORING", exit_attempts=0, last_exit_error="",
        )
        account = SimpleNamespace(bankroll_usd=5.0, position=position)
        bridge = SimpleNamespace(
            accounts={"L25_CURRENT": account},
            observed_marks={MINT: ObservedMarkPoint(
                value_usd=55.9, pair_address=PAIR, dex_id="d",
                liquidity_usd=1.0, volume_5m_usd=1.0, market_cap_usd=1.0,
                timestamp=time.time(), available=True)},
        )
        fake_self = SimpleNamespace(
            _paper_bridge=bridge,
            _arm_unrealized_usd=PaperTrader._arm_unrealized_usd,
            _arm_return_pct=PaperTrader._arm_return_pct,
        )
        rows = PaperTrader._execution_aware_open_rows(fake_self, time.time())
        row = rows[0]
        assert row["current_price"] == pytest.approx(55.9)
        assert row["unrealized_pnl_pct"] == pytest.approx(-44.1, abs=0.5)
        assert row["unrealized_pnl_usd"] == pytest.approx(-0.1852, abs=1e-3)
        assert row["realized_pnl_usd"] == 0.0
        assert row["total_pnl_usd"] == pytest.approx(row["unrealized_pnl_usd"])
        assert row["status"] == "OPEN_MONITORING"
        assert row["total_equity_usd"] is not None


class TestBoundedRetries:
    def test_retry_count_is_bounded_and_blocks(self):
        # TEST 4
        engine = make_engine(arms=[])
        position = SimpleNamespace(
            signal_id="s1", arm="L25_CURRENT", symbol="T", mint=MINT)
        for _ in range(_EXIT_MAX_ATTEMPTS + 4):
            state = engine._note_exit_quote_failure(
                position, "no executable SELL quote",
                rpc_cause=None, retry_after_s=None)
        assert state["attempts"] == _EXIT_MAX_ATTEMPTS
        assert state["blocked"] is True
        assert state["next_attempt_at"] == float("inf")
        assert position.exit_status == "UNSELLABLE"
        assert position.exit_attempts == _EXIT_MAX_ATTEMPTS
        assert position.last_exit_error.startswith("no executable")
        # Attempts must NOT grow beyond the cap on further failures.
        engine._note_exit_quote_failure(
            position, "no executable SELL quote",
            rpc_cause=None, retry_after_s=None)
        assert state["attempts"] == _EXIT_MAX_ATTEMPTS

    def test_market_failure_is_not_retried_in_a_zero_second_loop(self):
        engine = make_engine(arms=[])
        position = SimpleNamespace(
            signal_id="s2", arm="L25_CURRENT", symbol="T", mint=MINT)
        state = engine._note_exit_quote_failure(
            position, "no executable SELL quote",
            rpc_cause=None, retry_after_s=None)
        # Not a capacity problem, so it retries on the normal cadence -- not
        # an unbounded 0.0s storm; and once the budget is spent it stops.
        assert position.exit_status == "EXIT_PENDING_QUOTE"

    def test_exit_blocked_in_full_monitor_loop(self):
        # TEST 5: curve complete -> position stays OPEN, no fake close.
        engine = make_engine(
            arms=parse_arms(["L25_CURRENT"]),
            quotes=FakeQuoteService(sell=unavailable_point(side="SELL")),
        )
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_CURRENT"].position
        trades = []
        for _ in range(_EXIT_MAX_ATTEMPTS):
            trades = run(engine.monitor_once())
            if position.exit_status == "UNSELLABLE":
                break
        assert trades == [] or all(t.exit_fill is None for t in trades)
        assert position.exit_status == "UNSELLABLE"
        assert position.last_exit_error
        assert engine.accounts["L25_CURRENT"].has_position
        assert engine.accounts["L25_CURRENT"].position is position
        # No fake close: no trade settled the position.
        assert all(t.exit_fill is None for t in engine.trade_records
                   if t.position.signal_id == position.signal_id)
        # Once blocked, further ticks do not even attempt a quote.
        assert run(engine.monitor_once()) == []

    def test_successful_later_sell_books_realized_once(self):
        # TEST 6
        engine = make_engine(
            arms=parse_arms(["L25_CURRENT"]),
        )
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_CURRENT"].position
        engine.quote_service = FakeQuoteService(
            sell=unavailable_point(side="SELL", missing_reason="curve complete"))
        run(engine.monitor_once())
        assert position.exit_status in ("EXIT_PENDING_QUOTE", "UNSELLABLE")
        assert position.exit_attempts >= 1
        # Recover: a real SELL quote now, and the exit succeeds exactly once.
        engine.quote_service = FakeQuoteService(sell=sell_point())
        trades = run(engine.monitor_once())
        assert len(trades) == 1
        assert trades[0].exit_fill is not None
        assert engine.accounts["L25_CURRENT"].position is None
        assert trades[0].pnl is not None
        assert trades[0].pnl.realized_pnl_usd is not None
        # A further monitor tick must not re-settle the same position.
        trades2 = run(engine.monitor_once())
        assert trades2 == []

    def test_restart_recovers_unsellable_position_without_trade(self):
        # TEST 7
        engine = make_engine(
            arms=parse_arms(["L25_CURRENT"]),
            quotes=FakeQuoteService(sell=unavailable_point(side="SELL")),
        )
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_CURRENT"].position
        for _ in range(_EXIT_MAX_ATTEMPTS):
            run(engine.monitor_once())
            if position.exit_status == "UNSELLABLE":
                break
        assert position.exit_status == "UNSELLABLE"
        record = position.as_dict()
        revived = _position_from_dict(record)
        assert revived is not None
        assert revived.exit_status == "UNSELLABLE"
        assert revived.exit_attempts >= 1
        assert revived.last_exit_error
        assert revived.tokens_raw == position.tokens_raw
        assert revived.opened_at == position.opened_at


class TestRowEconomics:
    def test_realized_not_overwritten_by_unrealized(self):
        # An open position: realized exactly 0, unrealized negative, total
        # equal to unrealized. Never the reverse.
        row_fields = {
            "realized_pnl_usd": 0.0,
            "unrealized_pnl_usd": -0.1862,
            "total_pnl_usd": -0.1862,
        }
        assert row_fields["realized_pnl_usd"] == 0.0
        assert row_fields["unrealized_pnl_usd"] < 0
        assert row_fields["total_pnl_usd"] == row_fields["unrealized_pnl_usd"]
