"""
Accounting / dashboard integrity regression tests.

These exist because every defect they cover was invisible on screen: the ledger
summed correctly while the account reported zero, "runner capture" measured
something else entirely, and an unknown pool liquidity was silently replaced by
a number that made every fill look clean.

Each test names the failure it prevents.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autopaper.config import ExecutionModel
from autopaper.dashboard import _num_or_na, _pct_or_na, build_frame
from autopaper.execution.fill import (
    LIQUIDITY_OBSERVED,
    LIQUIDITY_UNKNOWN,
    impact_bps,
    liquidity_source,
)
from autopaper.models import FillResult, Trade
from autopaper.paper_accounting import (
    DRIFT_UNSETTLED,
    available_profiles,
    reconcile,
    stats_for_profile,
)
from autopaper.stats import compute as compute_stats

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "var/ledger.jsonl"
DECISIONS = ROOT / "var/decisions.jsonl"
CANDIDATES = ROOT / "var/candidates.jsonl"
UPDATES = ROOT / "var/position_updates.jsonl"


def _trade(i: int, *, pnl: float, ret: float, mfe: float | None = 100.0,
           mae: float | None = -10.0, hold: float = 200.0,
           profile: str = "L25_CURRENT") -> Trade:
    return Trade(
        trade_id=f"t{i}", position_id=f"p{i}", candidate_id=f"c{i}",
        mint=f"M{i}", symbol=f"S{i}", source="telegram", strategy="FAST_ENTRY",
        route="FAST_ENTRY", regime="EXPLOSIVE",
        entry_signal_timestamp=float(i), entry_timestamp=float(i),
        entry_price=1.0, fill_price=1.0, exit_timestamp=float(i) + hold,
        exit_price=1.0 + ret / 100.0, quantity=100.0,
        entry_fee_usd=0.1, exit_fee_usd=0.1, entry_slippage_usd=0.1,
        exit_slippage_usd=0.1,
        gross_pnl_usd=pnl + 0.4, net_pnl_usd=pnl, return_pct=ret,
        holding_seconds=hold,
        mfe_pct=mfe if mfe is not None else 0.0,
        mae_pct=mae if mae is not None else 0.0,
        exit_reason="x", exit_style="FAST_FAILURE", thesis="",
        invalidation=(),
        strategy_profile=profile,
    )


def _ledger_trades():
    if not LEDGER.exists():
        return []
    from autopaper.ledger import Ledger
    from autopaper.models import Trade as T
    L = Ledger(LEDGER, DECISIONS, CANDIDATES, UPDATES)
    return [T(**r) for r in L.trades()]


# ---------------------------------------------------------------------------
# A. The $60,000 liquidity fiction
# ---------------------------------------------------------------------------

def test_01_unknown_liquidity_is_not_replaced_by_60000():
    """An absent pool liquidity must not become the configured $60,000."""
    m = ExecutionModel()
    unknown = impact_bps(12.68, None, m)
    assert abs(unknown - m.impact_max_bps) < 1e-9, (
        "unknown liquidity must charge the configured worst case, not an "
        "invented pool size")
    assert unknown > 1.0, (
        "the old $60,000 substitution collapsed impact to ~1bp and FLATTERED "
        "the fill; unknown must not be cheaper than a real observation")


def test_02_unknown_liquidity_is_labelled_not_hidden():
    assert liquidity_source(None) == LIQUIDITY_UNKNOWN
    assert liquidity_source(0) == LIQUIDITY_UNKNOWN
    assert liquidity_source(17_600.0) == LIQUIDITY_OBSERVED


def test_03_executable_quote_beats_a_mark_derived_fill():
    """A real quote for this size already contains the order's impact."""
    m = ExecutionModel()
    # Same order, same notional: a real 17.6k pool is ~2.4bp, the invented
    # 60k pool was ~1.3bp. Observed liquidity must cost MORE, not less.
    real = impact_bps(12.68, 17_600.0, m)
    fiction = impact_bps(12.68, 60_000.0, m)
    assert real > fiction


def test_04_missing_numbers_render_as_na_not_zero():
    assert _pct_or_na(None) == "n/a"
    assert _num_or_na(None) == "n/a"
    assert _pct_or_na(0.0) == "0.0%"  # a measured zero is still a zero
    assert _pct_or_na("garbage") == "n/a"


# ---------------------------------------------------------------------------
# B. Accounting
# ---------------------------------------------------------------------------

def test_05_reconcile_total_pnl_is_the_sum_of_each_trade_once():
    trades = [_trade(i, pnl=p, ret=p, mfe=100.0, hold=200.0)
              for i, p in enumerate([10.0, -5.0, 20.0, -2.0, 7.0])]
    rec = reconcile(trades, starting_equity=100.0)
    assert abs(rec.realized_pnl_usd - 30.0) < 1e-9
    assert abs(rec.final_equity_usd - 130.0) < 1e-9
    assert len(rec.curve) == 5, "one curve point per trade, no duplication"
    assert not any("INVARIANT VIOLATION" in n for n in rec.notes)


def test_06_duplicate_trade_ids_are_reported_not_hidden():
    trades = [_trade(1, pnl=5.0, ret=5.0), _trade(1, pnl=5.0, ret=5.0)]
    rec = reconcile(trades, starting_equity=100.0)
    assert any("duplicate trade ids" in n for n in rec.notes)


def test_07_frozen_account_book_is_flagged_not_adopted():
    trades = [_trade(i, pnl=p, ret=p, mfe=100.0, hold=200.0)
              for i, p in enumerate([10.0, -5.0, 20.0])]
    frozen = {"realized_pnl_usd": 0.0, "cash_usd": 100.0, "equity_usd": 100.0}
    rec = reconcile(trades, starting_equity=100.0, account_book=frozen)
    assert rec.drift_status == DRIFT_UNSETTLED
    assert abs(rec.drift_usd - 25.0) < 1e-9
    assert any("ACCOUNT BOOK NOT SETTLED" in n for n in rec.notes)
    # the ledger figure is the one reported
    assert rec.as_dict()["account"]["realized_pnl"] == 25.0
    assert rec.as_dict()["account"]["equity"] == 125.0


def test_08_agreeing_account_book_is_not_flagged():
    trades = [_trade(i, pnl=p, ret=p, mfe=100.0, hold=200.0)
              for i, p in enumerate([10.0, -5.0])]
    book = {"realized_pnl_usd": 5.0, "cash_usd": 105.0}
    rec = reconcile(trades, starting_equity=100.0, account_book=book)
    assert rec.drift_status == "OK"
    assert rec.drift_usd == 0.0


# ---------------------------------------------------------------------------
# C. Dashboard metrics come from the canonical stats
# ---------------------------------------------------------------------------

def test_09_dashboard_perf_matches_canonical_compute():
    trades = [_trade(i, pnl=p, ret=p, mfe=200.0, hold=200.0)
              for i, p in enumerate([30.0, -10.0, 20.0, -5.0])]
    canonical = compute_stats(trades, starting_equity=100.0).as_dict()
    rec = reconcile(trades, starting_equity=100.0).as_dict()
    perf = rec["performance"]
    for key in ("total_trades", "wins", "losses", "win_rate",
                "profit_factor", "expectancy_usd", "realized_pnl_usd"):
        assert perf[key] == canonical[key], key
    frame = build_frame(performance=canonical, account=rec["account"], cols=110)
    pf_row = next(l for l in frame.splitlines() if l.strip().startswith("PF"))
    wr_row = next(l for l in frame.splitlines() if l.strip().startswith("win rate"))
    assert f"{canonical['profit_factor']:.2f}" in pf_row, pf_row
    assert f"{canonical['win_rate'] * 100:.1f}%" in wr_row, wr_row


def test_10_pf_matches_canonical_backend_calculation():
    trades = [_trade(i, pnl=p, ret=p, mfe=100.0, hold=200.0)
              for i, p in enumerate([100.0, -50.0])]
    s = compute_stats(trades, starting_equity=100.0)
    assert abs(s.profit_factor - 2.0) < 1e-9  # gross 100 / gross loss 50
    frame = build_frame(performance=s.as_dict(), cols=100)
    pf_row = next(l for l in frame.splitlines() if l.strip().startswith("PF"))
    assert "2.00" in pf_row, pf_row


def test_11_max_dd_uses_the_sequential_equity_curve():
    trades = [_trade(i, pnl=p, ret=p, mfe=100.0, hold=200.0)
              for i, p in enumerate([50.0, -60.0, 10.0])]
    rec = reconcile(trades, starting_equity=100.0)
    # 100 -> 150 -> 90 -> 100 ; peak 150, trough 90 => -40%
    assert abs(rec.peak_equity_usd - 150.0) < 1e-9
    assert abs(rec.max_drawdown_pct - (-40.0)) < 1e-6
    assert rec.max_drawdown_scope == "realized_trade_to_trade"


def test_12_drawdown_is_labelled_as_realized_in_the_ui():
    trades = [_trade(i, pnl=p, ret=p, mfe=100.0, hold=200.0)
              for i, p in enumerate([50.0, -60.0])]
    s = compute_stats(trades, starting_equity=100.0).as_dict()
    frame = build_frame(performance=s, cols=110)
    assert "realized DD" in frame
    assert "Max DD" not in frame


# ---------------------------------------------------------------------------
# D. Runner metrics
# ---------------------------------------------------------------------------

def test_13_runner_pnl_share_is_distinct_from_mfe_capture():
    """The two numbers are not the same quantity and must not be conflated."""
    trades = [
        _trade(1, pnl=100.0, ret=100.0, mfe=200.0, hold=300.0),
        _trade(2, pnl=10.0, ret=10.0, mfe=20.0, hold=300.0),
    ]
    s = compute_stats(trades, starting_equity=100.0)
    # runner PnL share = 100 / 110
    assert abs(s.runner_pnl_share_pct - 100.0 / 110.0 * 100.0) < 1e-6
    # true MFE capture = (100/200 + 10/20)/2 = 0.5 -> 50%
    assert abs(s.mfe_capture_pct - 50.0) < 1e-6
    assert abs(s.runner_pnl_share_pct - s.mfe_capture_pct) > 1.0


def test_14_runner_capture_label_is_gone_from_both_surfaces():
    from autopaper.telegram import ui as tui
    import autopaper.dashboard as dash
    assert "Runner capture" not in dash.render.__doc__ if dash.render.__doc__ else True
    src_tui = Path(tui.__file__).read_text()
    assert "Runner capture:" not in src_tui
    assert "Runner PnL share" in src_tui
    src_dash = Path(dash.__file__).read_text()
    assert "runner PnL share" in src_dash
    assert "capture {stats" not in src_dash


def test_15_mfe_capture_is_none_not_zero_when_uncomputable():
    trades = [_trade(1, pnl=5.0, ret=5.0, mfe=0.0, hold=10.0)]  # no runner
    s = compute_stats(trades, starting_equity=100.0)
    assert s.runners == 0
    assert s.mfe_capture_pct is None, "no runners => capture is unknown, not 0%"
    frame = build_frame(performance=s.as_dict(), cols=110)
    assert "n/a" in frame


# ---------------------------------------------------------------------------
# E. Strategy integrity
# ---------------------------------------------------------------------------

def test_16_strategy_filter_separates_profiles():
    a = [_trade(i, pnl=10.0, ret=10.0, mfe=100.0, hold=200.0,
                profile="L25_CURRENT") for i in range(4)]
    b = [_trade(10 + i, pnl=-10.0, ret=-10.0, mfe=100.0, hold=200.0,
                profile="L25_60S") for i in range(2)]
    allt = a + b
    pooled = compute_stats(allt, starting_equity=100.0).as_dict()
    cur = stats_for_profile(allt, "L25_CURRENT")
    sixty = stats_for_profile(allt, "L25_60S")
    assert pooled["total_trades"] == 6
    assert cur["total_trades"] == 4 and cur["realized_pnl_usd"] == 40.0
    assert sixty["total_trades"] == 2 and sixty["realized_pnl_usd"] == -20.0
    assert pooled["win_rate"] != cur["win_rate"], (
        "pooling two strategies must not produce one strategy's win rate")
    assert available_profiles(allt) == ["L25_60S", "L25_CURRENT"]


def test_17_mixed_population_is_declared_and_warned_about():
    allt = [_trade(1, pnl=1.0, ret=1.0, profile="L25_CURRENT"),
            _trade(2, pnl=1.0, ret=1.0, profile="L25_TP100")]
    s = compute_stats(allt, starting_equity=100.0).as_dict()
    assert s["mixed_strategy_population"] is True
    assert set(s["strategy_profiles_present"]) == {"L25_CURRENT", "L25_TP100"}
    frame = build_frame(performance=s, cols=110)
    assert "mixed strategies" in frame


def test_18_closed_row_keeps_the_profile_it_was_opened_under():
    tr = _trade(1, pnl=1.0, ret=1.0, profile="L25_60S")
    assert tr.strategy_profile == "L25_60S", (
        "a historical trade keeps its own profile; selection never rewrites it")


def test_19_fill_result_carries_liquidity_provenance_fields():
    r = FillResult(filled=True, reason="filled", quoted_price=1.0, fill_price=1.0,
                   slippage_bps=1.0, slippage_usd=0.0, fee_bps=85.0, fee_usd=0.0,
                   latency_ms=0.0, placed_at=0.0, fill_at=0.0, observed_at=0.0,
                   price_at_quote=1.0, data_state=None)
    assert hasattr(r, "liquidity_source")
    assert hasattr(r, "execution_fidelity")


# ---------------------------------------------------------------------------
# F. End-to-end against the real ledger, when it is present
# ---------------------------------------------------------------------------

def test_20_real_ledger_reconciles_to_the_headline():
    trades = _ledger_trades()
    if not trades:
        pytest.skip("no ledger present")
    rec = reconcile(trades, starting_equity=100.0)
    assert abs(rec.realized_pnl_usd - sum(t.net_pnl_usd for t in trades)) < 1e-6
    s = compute_stats(trades, starting_equity=100.0).as_dict()
    # as_dict() rounds to 4dp for display; compare within that tolerance.
    assert abs(rec.realized_pnl_usd - s["realized_pnl_usd"]) <= 1e-4
    # every trade appears exactly once on the curve
    assert len(rec.curve) == len(trades)
    assert len({p.trade_id for p in rec.curve}) == len(trades)


def test_21_real_ledger_runner_capture_is_not_zero_by_construction():
    trades = _ledger_trades()
    if not trades:
        pytest.skip("no ledger present")
    s = compute_stats(trades, starting_equity=100.0)
    if s.runners == 0:
        pytest.skip("no runners in ledger")
    assert s.mfe_capture_pct is not None
    assert s.mfe_capture_executable_pct is None, (
        "executable MFE is not stored, so executable capture must stay None "
        "rather than borrowing the mark figure")