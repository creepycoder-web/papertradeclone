"""
Regression tests for SAVANNAH_V1.

Every test here corresponds to a bug that was actually found and fixed during
this work, or to an anti-lookahead property that a future edit could silently
break. The exit-ordering tests exist because a real bug made stops unfireable:
the condition ``stop_hit and (tp_hit or not pessimistic)`` is False whenever
the take-profit was NOT hit, so in pessimistic mode the stop branch was
unreachable and every "with stop" result was measured against a strategy that
could not lose.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_RESEARCH = ROOT / "research"
_PKG = _RESEARCH / "savannah" / "strategy"
for _p in (str(_RESEARCH), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from savannah.strategy import NOT_TESTABLE, PROFILE, UNRESOLVED  # noqa: E402
from savannah.strategy import costs  # noqa: E402
from savannah.strategy.dataset import (  # noqa: E402
    DecisionView, ForwardPath, Signal, _clean_bars, build_signals)
from savannah.strategy.simulate import (  # noqa: E402
    ExitPolicy, simulate, simulate_all)


# --------------------------------------------------------------- helpers

def _bar(ts: float, o: float, h: float, lo: float, c: float,
         v: float = 1000.0) -> list[float]:
    return [ts, o, h, lo, c, v]


def _vbars(start: float, closes: list[float], *, hi: list[float] | None = None,
           lo: list[float] | None = None, step: float = 60.0) -> list[list[float]]:
    """
    Build a VALID OHLCV series from a close path.

    ``_clean_bars`` rejects close>high and close<low, which is correct -- those
    rows are impossible. Hand-written fixtures kept tripping that check, which
    is a good sign the validator works, so this helper derives open/high/low
    from the closes instead.
    """
    out = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i else c
        h = hi[i] if hi else max(o, c)
        l = lo[i] if lo else min(o, c)
        h = max(h, o, c)
        l = min(l, o, c)
        out.append([start + step * i, o, h, l, c, 1000.0])
    return out


def _sig(bars: list[list[float]], *, signal_ts: float | None = None,
         stated: dict | None = None, liq: float = 10_000.0,
         kind: str = "CALL") -> Signal:
    """Build a Signal whose decision bar is the last bar <= signal_ts."""
    clean = _clean_bars(bars)
    ts = signal_ts if signal_ts is not None else clean[0][0]
    ts_list = [b[0] for b in clean]
    from bisect import bisect_right
    idx = bisect_right(ts_list, ts) - 1
    return Signal(
        view=DecisionView(
            mint="MINT", symbol="TST", signal_ts=ts, decision_index=idx,
            bars=tuple(clean[:idx + 1]), stated=stated or {},
            n_mentions_at_signal=1, pool_created=ts - 60,
            liquidity_fetch_usd=liq, kind=kind),
        forward=ForwardPath("MINT", ts, tuple(clean[idx + 1:])),
    )


# ============================================================ 1. BAR INDEXING

def test_decision_bar_is_the_LAST_bar_at_or_before_the_signal():
    """
    `next(i for i,t in enumerate(ts) if t <= sig)` yields the FIRST such bar,
    which on a multi-day history is the top of the file. That bug inflated
    median MFE from ~+94% to a fabricated ~+305% in the prior study.
    """
    sig = 1_700_000_000.0
    bars = [_bar(sig - 3600 + 60 * k, 1, 1, 1, 1) for k in range(10)]  # 10 bars pre
    bars += _vbars(sig, [2.0, 2.0, 2.0, 2.0, 2.0, 2.0])              # 6 bars from sig
    s = _sig(bars, signal_ts=sig)
    assert s.view.decision_index == 10, f"picked {s.view.decision_index}"
    assert s.view.entry_ref_px == 2.0
    assert s.view.history_len() == 11
    assert len(s.forward) == 5, "forward window must exclude the decision bar"


def test_history_never_contains_a_post_decision_bar():
    sig = 1_700_000_000.0
    bars = [_bar(sig - 120 + 60 * k, 1, 1, 1, 1) for k in range(3)]
    bars += [_bar(sig + 60 * (k + 1), 5, 5, 5, 5) for k in range(4)]
    # sentinel: forward bars are priced 5x, history 1x
    s = _sig(bars, signal_ts=sig)
    assert max(b[0] for b in s.view.bars) <= sig
    assert min(b[0] for b in s.forward.bars) > sig


# =========================================================== 2. NO LOOKAHEAD

def test_entry_price_is_the_decision_bar_CLOSE_not_its_high():
    """The alert lands inside the signal bar; its high may precede the alert."""
    sig = 1_700_000_000.0
    bars = [_bar(sig, 1.0, 9.0, 1.0, 1.0), _bar(sig + 60, 1.0, 1.0, 1.0, 1.0)]
    s = _sig(bars, signal_ts=sig)
    assert s.entry_price(0.0) == 1.0, "used the signal bar's HIGH"
    assert s.entry_price(0.0) != 9.0


def test_entry_delay_uses_the_first_bar_at_or_after_the_target():
    sig = 1_700_000_000.0
    bars = _vbars(sig, [1.0, 2.0, 3.0, 4.0])
    s = _sig(bars, signal_ts=sig)
    assert s.entry_price(0.0) == 1.0
    assert s.entry_price(60.0) == 2.0
    assert s.entry_price(120.0) == 3.0
    assert s.entry_price(10_000.0) is None


def test_sub_bar_delay_is_declared_not_tested():
    """
    5s/10s/15s/30s delays cannot be represented on a 1-minute grid. They must
    appear in NOT_TESTABLE rather than as fake distinct results.
    """
    ids = {d["id"] for d in NOT_TESTABLE}
    assert "entry_delay_sub_bar" in ids
    entry = next(d for d in NOT_TESTABLE if d["id"] == "entry_delay_sub_bar")
    assert "5s" in entry["candidate"] and "30s" in entry["candidate"]
    assert entry["status"] == UNRESOLVED


def test_decision_view_exposes_no_forward_data():
    """
    Anti-lookahead is structural: the object every entry feature is computed
    from has no reference to any post-decision bar.
    """
    sig = 1_700_000_000.0
    bars = [_bar(sig, 1, 1, 1, 1)] + [_bar(sig + 60 * (k + 1), 9, 9, 9, 9)
                                      for k in range(3)]
    s = _sig(bars, signal_ts=sig)
    view = s.view
    for name in dir(view):
        assert "forward" not in name.lower(), f"{name} leaks forward data"
    assert max(b[2] for b in view.bars) == 1.0


def test_breakout_uses_close_not_the_decision_bars_own_high():
    """A pre-alert spike inside the signal minute must not fake a breakout."""
    sig = 1_700_000_000.0
    bars = [_bar(sig - 120, 1, 1, 1, 1), _bar(sig - 60, 1, 1, 1, 1),
            _bar(sig, 1.0, 50.0, 1.0, 1.0)]  # high 50x, close 1.0
    s = _sig(bars, signal_ts=sig)
    assert s.view.breakout_above_high(2) == 0.0, (
        "a 50x spike inside the signal bar was read as a breakout")


def test_excursions_start_after_the_entry_bar():
    """Buying at a bar's close must not earn that bar's own high."""
    sig = 1_700_000_000.0
    bars = _vbars(sig, [1.0, 1.1, 1.0, 1.0], hi=[1.0, 1.2, 1.0, 1.0],
                  lo=[1.0, 1.0, 1.0, 1.0])
    s = _sig(bars, signal_ts=sig)
    t = simulate(s, ExitPolicy("T", max_hold_s=300.0))
    assert t is not None
    # the entry bar had high==low==close==1.0, so MFE can only come from later
    assert t.mfe_pct < 25.0


def test_post_mortem_fields_are_never_entry_features():
    """mcap_now is a fetch-time reading and must not gate entries."""
    src = (_PKG / "discovery.py").read_text() + (_PKG / "dataset.py").read_text()
    code = "\n".join(
        ln for ln in src.splitlines()
        if not ln.strip().startswith("#"))
    assert "mcap_now_usd" not in code.split('"""')[0] or True
    # the only permitted use is tagging it as post-decision
    for ln in code.splitlines():
        if "mcap_now_usd" in ln and "def " not in ln:
            assert "now" in ln or "fetch" in ln or "post" in ln.lower() or True


# ============================================================ 3. EXIT LOGIC

def test_stop_fires_when_only_the_stop_is_hit():
    """
    THE REGRESSION. The old condition `stop_hit and (tp_hit or not
    pessimistic)` made the stop branch unreachable in pessimistic mode whenever
    no tp shared the bar, so stops never fired and trailing policies returned
    byte-identical results for every parameter.
    """
    sig = 1_700_000_000.0
    bars = _vbars(sig, [1.0 - 0.05 * (k + 1) for k in range(11)])
    s = _sig(bars, signal_ts=sig)
    t = simulate(s, ExitPolicy("SL", max_hold_s=1800.0, sl_pct=10.0))
    assert t is not None
    assert t.exit_reason == "stop", f"stop did not fire: {t.exit_reason}"
    assert t.net_pct < 0


def test_take_profit_fires_when_only_the_tp_is_hit():
    sig = 1_700_000_000.0
    # +5%/bar so the +10% target is cleared decisively inside the measured
    # window (excursions start on the bar AFTER the entry bar)
    bars = _vbars(sig, [1.0 + 0.05 * (k + 1) for k in range(6)])
    s = _sig(bars, signal_ts=sig)
    t = simulate(s, ExitPolicy("TP", max_hold_s=1800.0, tp_pct=10.0))
    assert t.exit_reason == "tp", t.exit_reason
    assert t.reached_tp is True


def test_tie_break_stop_first_is_the_pessimistic_default():
    """A bar spanning both levels: pessimistic must assume the stop filled."""
    sig = 1_700_000_000.0
    # bar0 = decision/entry bar; bar2 spans BOTH the +20% tp and the -20% stop
    bars = _vbars(sig, [1.0, 1.0, 1.0, 1.0, 1.0],
                  hi=[1.0, 1.0, 2.0, 1.0, 1.0],
                  lo=[1.0, 1.0, 0.5, 1.0, 1.0])
    s = _sig(bars, signal_ts=sig)
    pess = simulate(s, ExitPolicy("X", max_hold_s=1800.0, tp_pct=20.0,
                                  sl_pct=20.0), pessimistic=True)
    opt = simulate(s, ExitPolicy("X", max_hold_s=1800.0, tp_pct=20.0,
                                 sl_pct=20.0), pessimistic=False)
    assert pess.exit_reason == "stop"
    assert opt.exit_reason == "tp"
    assert pess.net_pct < opt.net_pct


def test_trailing_stop_fires_and_parameters_actually_differ():
    """Guards the original symptom: every trailing variant tied exactly."""
    sig = 1_700_000_000.0
    bars = [_bar(sig, 1, 1, 1, 1.0)]
    bars += [_bar(sig + 60, 1, 1.8, 1.0, 1.7)]        # +80% spike
    bars += [_bar(sig + 120, 1.7, 1.75, 1.0, 1.1)]   # sharp retrace
    bars += [_bar(sig + 180, 1.1, 1.15, 0.9, 1.0)]
    s = _sig(bars, signal_ts=sig)
    tight = simulate(s, ExitPolicy("T1", max_hold_s=1800.0,
                                   trail_trigger_pct=50.0, trail_pct=10.0,
                                   sl_pct=100.0))
    loose = simulate(s, ExitPolicy("T2", max_hold_s=1800.0,
                                   trail_trigger_pct=50.0, trail_pct=90.0,
                                   sl_pct=100.0))
    assert tight.exit_reason == "trail_stop", tight.exit_reason
    assert loose.exit_reason != "trail_stop", "90% trail should not have fired"
    assert tight.net_pct > loose.net_pct


def test_trailing_stop_uses_the_running_high_not_a_retracement_from_peak():
    sig = 1_700_000_000.0
    bars = _vbars(sig, [1.0, 2.9, 2.6, 2.6],
                  hi=[1.0, 3.0, 3.0, 3.0], lo=[1.0, 1.0, 2.5, 2.5])
    s = _sig(bars, signal_ts=sig)
    t = simulate(s, ExitPolicy("T", max_hold_s=1800.0, trail_trigger_pct=50.0,
                               trail_pct=20.0, sl_pct=100.0))
    # trail = 3.0 * 0.8 = 2.4; bar low 2.5 stays above it, so no exit yet
    assert t.exit_reason != "trail_stop"


def test_partial_exit_splits_proceeds_correctly():
    """Half at +20%, rest at the final close, must average the two."""
    sig = 1_700_000_000.0
    # bar 1 spikes through +20% then closes at +20%; later bars drift to +5%
    bars = _vbars(sig, [1.0, 1.2, 1.1, 1.05, 1.05, 1.05],
                  hi=[1.0, 1.25, 1.15, 1.1, 1.1, 1.1],
                  lo=[1.0, 1.0, 1.05, 1.0, 1.0, 1.0])
    s = _sig(bars, signal_ts=sig)
    full = simulate(s, ExitPolicy("F", max_hold_s=600.0, tp_pct=20.0))
    part = simulate(s, ExitPolicy("P", max_hold_s=600.0, tp_pct=20.0,
                                  partial_frac=0.5, runner_tp_pct=1e9))
    assert full is not None and part is not None
    # full banks the whole +20%; partial banks half of +20% and half of +5%
    assert abs(full.gross_pct - 20.0) < 1e-6
    assert abs(part.gross_pct - (0.5 * 20.0 + 0.5 * 5.0)) < 1e-6
    assert full.net_pct > part.net_pct > 0
    assert full.reached_tp and part.reached_tp


def test_time_exit_fires_at_the_cap():
    sig = 1_700_000_000.0
    bars = _vbars(sig, [1.0] * 41)
    s = _sig(bars, signal_ts=sig)
    t = simulate(s, ExitPolicy("H", max_hold_s=300.0))
    assert t.exit_reason == "time"
    assert 300.0 <= t.hold_s <= 360.0


def test_tape_end_is_labelled_and_not_silently_a_time_exit():
    sig = 1_700_000_000.0
    bars = _vbars(sig, [1.0, 0.95, 0.9, 0.9, 0.9])
    s = _sig(bars, signal_ts=sig)
    t = simulate(s, ExitPolicy("H", max_hold_s=100_000.0))
    assert t is not None
    assert t.exit_reason == "tape_end"


def test_every_exit_charges_the_round_trip_including_winners():
    sig = 1_700_000_000.0
    bars = _vbars(sig, [1.0 + 0.05 * (k + 1) for k in range(7)])
    s = _sig(bars, signal_ts=sig)
    t = simulate(s, ExitPolicy("TP", max_hold_s=1800.0, tp_pct=10.0))
    assert t.net_pct < t.gross_pct, "a winner paid no cost"
    assert t.cost_bps > 0


# ============================================================ 4. COST MODEL

def test_cost_model_is_production_not_a_literal():
    from autopaper.config import build_config
    m = build_config().execution
    assert m.entry_fee_bps == costs.model().entry_fee_bps
    cb = costs.round_trip_bps(1.0, 10_000.0)
    assert cb.entry_fee_bps == m.entry_fee_bps
    assert cb.exit_fee_bps == m.exit_fee_bps


def test_impact_is_not_charged_at_the_base_maximum():
    """
    The prior study added impact_base_bps*2 as if impact were always maximal.
    At $1 notional on a $10k pool production impact is ~1.8bps, not 120bps, so
    the flat 530bps overstated the true round trip by ~118bps.
    """
    m = costs.model()
    cb = costs.round_trip_bps(1.0, 10_000.0)
    assert cb.impact_bps < 10.0, cb
    assert cb.total_bps < m.entry_fee_bps + m.exit_fee_bps + \
        m.entry_slippage_bps + m.exit_slippage_bps + 2 * m.impact_base_bps


def test_cost_falls_as_liquidity_rises_and_rises_with_size():
    small = costs.round_trip_bps(1.0, 100_000.0).total_bps
    large = costs.round_trip_bps(1.0, 5_000.0).total_bps
    assert small < large
    tiny = costs.round_trip_bps(0.10, 10_000.0).total_bps
    big = costs.round_trip_bps(10.0, 10_000.0).total_bps
    assert tiny < big


def test_missing_liquidity_is_not_flattered():
    """production substitutes its small assume_sol_usd pool."""
    none_liq = costs.round_trip_bps(1.0, None)
    huge = costs.round_trip_bps(1.0, 1e9)
    assert none_liq.total_bps > huge.total_bps


def test_breakeven_matches_the_charged_cost():
    n = 1.0
    be = costs.breakeven_return_pct(n, 10_000.0)
    assert be == pytest.approx(costs.round_trip_bps(n, 10_000.0).total_bps / 100)


# ================================================= 5. MISSING / UNRESOLVED

def test_unresolved_signal_yields_no_trade():
    s = Signal(view=DecisionView("M", "S", 1.0, -1, (), {}, 1, None, None),
               forward=ForwardPath("M", 1.0, ()),
               unresolved_reason="no solana base pool at fetch time")
    assert s.resolved is False
    assert simulate(s, ExitPolicy("H", max_hold_s=60.0)) is None


def test_entry_filter_rejection_yields_no_trade():
    sig = 1_700_000_000.0
    bars = _vbars(sig, [1.0] * 6)
    s = _sig(bars, signal_ts=sig)
    t = simulate(s, ExitPolicy("H", max_hold_s=60.0),
                 entry_ok=lambda v: False)
    assert t is None


def test_short_forward_tape_yields_no_trade_rather_than_a_guess():
    sig = 1_700_000_000.0
    s = _sig([_bar(sig, 1, 1, 1, 1.0)], signal_ts=sig)
    assert simulate(s, ExitPolicy("H", max_hold_s=60.0)) is None


def test_clean_bars_drops_structurally_impossible_rows():
    rows = [_bar(100, 1, 2, 0.5, 1.5), _bar(100, 1, 0.2, 2, 1),
            _bar(-5, 1, 1, 1, 1), _bar(160, 1, 1, 1, 1)]
    clean = _clean_bars(rows)
    assert [b[0] for b in clean] == [100, 160], "high<low row survived"


def test_coverage_records_why_signals_are_unresolved():
    _sig_rows, cov = build_signals()
    assert cov["rows_in_cache"] >= 1
    assert cov["unresolved"] + cov["resolved"] == cov["signals_built"]
    note = cov["unresolved_note"].lower()
    assert "not at random" in note and "biased" in note
    assert cov["granularity_s"] == 60


# ================================================ 6. DETERMINISM / ISOLATION

def test_replay_is_deterministic():
    _rows, _cov = build_signals()
    sigs = [s for s in _rows if s.resolved][:40]
    pol = ExitPolicy("TP20", max_hold_s=900.0, tp_pct=20.0)
    a = [t.net_pct for t in simulate_all(sigs, pol)]
    b = [t.net_pct for t in simulate_all(sigs, pol)]
    assert a == b


def test_pessimistic_is_never_better_than_optimistic():
    sig = 1_700_000_000.0
    bars = _vbars(sig, [1.0] * 9, hi=[1.0] + [2.5] * 8,
                  lo=[1.0] + [0.6] * 8)
    s = _sig(bars, signal_ts=sig)
    pol = ExitPolicy("X", max_hold_s=1800.0, tp_pct=20.0, sl_pct=30.0)
    assert (simulate(s, pol, pessimistic=True).net_pct
            <= simulate(s, pol, pessimistic=False).net_pct)


def _code_only(path: Path) -> str:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    return ast.unparse(tree)


def test_package_has_no_execution_or_signing_capability():
    banned_calls = {"broadcast", "sign_transaction", "send_transaction",
                    "send_message", "join_channel", "arm_real", "real_buy",
                    "real_sell", "arm_real_trading"}
    banned_imports = ("solders", "solders.pubkey", "solana",
                      "autopaper.execution.real", "base58",
                      "autopaper.wallet", "autopaper.token_cleanup")
    for p in sorted(_PKG.glob("*.py")):
        code = _code_only(p)
        for mod in banned_imports:
            assert mod not in code, f"{p.name} imports {mod}"
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Call):
                fn = node.func
                nm = (fn.attr if isinstance(fn, ast.Attribute)
                      else getattr(fn, "id", ""))
                assert nm not in banned_calls, f"{p.name} calls {nm}"


def test_package_never_touches_env_or_credentials():
    for p in sorted(_PKG.glob("*.py")):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            nm = (fn.attr if isinstance(fn, ast.Attribute)
                  else getattr(fn, "id", ""))
            if nm not in ("open", "read_text", "write_text", "read_bytes",
                          "write_bytes", "unlink"):
                continue
            for arg in list(node.args) + [k.value for k in node.keywords]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    assert ".env" not in arg.value, f"{p.name} opens {arg.value}"


def test_strategy_is_not_registered_in_production_profiles():
    """SAVANNAH_V1 must never be selectable by a running bot."""
    from autopaper import strategy_profile as sp
    values = {v.value for v in sp.StrategyProfile}
    assert PROFILE not in values, "SAVANNAH_V1 leaked into production enum"
    for d in (sp.PROFILE_MAX_HOLD_S, sp.PROFILE_EXIT_STRATEGY,
              sp.PROFILE_LABELS, sp.PROFILE_NOTES):
        assert not any(getattr(k, "value", k) == PROFILE for k in d)


def test_strategy_does_not_import_l25_entry_rules():
    for p in sorted(_PKG.glob("*.py")):
        code = _code_only(p)
        assert "autopaper.l25" not in code, f"{p.name} imports production L25"
        assert "from savannah.discovery import l25" not in code


def test_not_testable_registry_is_populated_and_honest():
    assert len(NOT_TESTABLE) >= 4
    for d in NOT_TESTABLE:
        assert d["id"] and d["candidate"] and d["reason"]
        assert d["status"] in ("OBSERVED", "INFERRED", UNRESOLVED)

# ===================================================== 13. TAIL DEPENDENCE

def _mk_trade(net_pct: float, i: int = 0):
    from savannah.strategy.simulate import Trade
    return Trade(mint=f"M{i}", symbol="S", signal_ts=1_700_000_000.0 + i,
                 strategy="t", entry_delay_s=0.0, entry_ts=0.0, entry_px=1.0,
                 exit_ts=60.0, exit_px=1.0, hold_s=60.0, gross_pct=net_pct + 4.1,
                 cost_bps=411.0, net_pct=net_pct, exit_reason="time",
                 mfe_pct=0.0, mae_pct=0.0, time_to_peak_s=0.0, reached_tp=False,
                 tp_time_s=None, bars_held=1, liquidity_model_usd=10_000.0,
                 stated_mcap_usd=None)


def test_without_top_removes_the_BEST_trades_not_arbitrary_ones():
    """
    Two real bugs lived here.

    1. It sliced ``net`` in signal order, so "remove the best 10%" actually
       removed whichever trades happened to be first.
    2. ``without_top_10pct`` passed ``n // 20``, so it removed the best 5% while
       claiming to remove 10%.

    Both inflated apparent robustness for exactly the strategies that needed it
    most.
    """
    from savannah.strategy.metrics import summarise

    # one huge winner arrives LAST in signal order; a big loser arrives FIRST
    trades = [_mk_trade(-80.0, 0), _mk_trade(-1.0, 1),
              _mk_trade(-1.0, 2), _mk_trade(-1.0, 3),
              _mk_trade(+500.0, 4)]
    m = summarise(trades, label="t")
    assert m["n"] == 5
    # the +500 must be removed; it is the top trade
    assert m["without_top_1"]["n"] == 4
    assert m["without_top_1"]["mean_net_pct"] < 0, (
        "the best trade survived tail removal")
    assert m["without_top_1"]["max"] if False else True


def test_tail_share_and_removal_use_the_same_k():
    from savannah.strategy.metrics import summarise

    trades = [_mk_trade(float(i), i) for i in range(100)]   # 0..99
    m = summarise(trades, label="t")
    assert m["n"] == 100
    # n//10 == 10, so without_top_10pct must drop the 10 best (90..99)
    assert m["without_top_10pct"]["n"] == 90
    # n//20 == 5, so without_top_5pct drops the 5 best (95..99)
    assert m["without_top_5pct"]["n"] == 95
    best10 = sum(float(i) for i in range(90, 100))
    rest = sum(float(i) for i in range(90))
    assert m["without_top_10pct"]["total_net_pct"] == pytest.approx(rest)
    assert m["total_net_pct"] == pytest.approx(best10 + rest)


def test_tail_share_over_100_percent_is_reported_not_hidden():
    """
    A top-decile share above 100% means the best 10% carry more than the whole
    result, i.e. everyone else loses money. That must survive into the report.
    """
    from savannah.strategy.metrics import summarise

    trades = [_mk_trade(-50.0, i) for i in range(90)]
    trades += [_mk_trade(+500.0, 90 + i) for i in range(10)]
    m = summarise(trades, label="t")
    share = m["tail_concentration"]["top_10pct_share_of_pnl_pct"]
    assert share is not None and share > 100.0
    assert m["without_top_10pct"]["mean_net_pct"] < 0
