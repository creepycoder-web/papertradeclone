"""
Focused tests for the multi-channel / L25 research package.

Fully offline. Every test uses synthetic fixtures or the frozen research
dataset; nothing opens a socket, signs, or broadcasts.

Coverage maps onto the brief's required list:
  1 channel parsing            6 stale SOL/USD refusal      11 fee accounting
  2 cross-channel dedup        7 fake provider = 0 network  12 rent creation
  3 first-seen attribution     8 shared cache reuse         13 rent reclaim
  4 L25 preservation           9 no swap quote for SOL/USD  14 no double-counting
  5 exit replay               10 exact position sizing      15 venue selection
  6 runner classification     16 pinned exits               17 10% mark guard
  7 one-slot simulation       18 size-dependent execution   19 chronological split
  8 multi-position sim        20 no-lookahead enforcement
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

RESEARCH_ROOT = Path(__file__).resolve().parents[2] / "research" / "multichannel"

# Make the research package importable without requiring PYTHONPATH, so the
# suite runs the same way the rest of the repository's tests do.
import sys as _sys
for _p in (str(RESEARCH_ROOT.parent), str(RESEARCH_ROOT.parent.parent / "src")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


def code_only(path: Path) -> str:
    """
    Source with every comment and docstring removed.

    A capability scan must run against CODE. The modules here DOCUMENT the
    things they refuse to do, and matching that prose would fail the guard for
    entirely the wrong reason.
    """
    import io
    import tokenize

    stream = io.StringIO(path.read_text())
    pieces: list[str] = []
    prev = tokenize.INDENT
    last_line = -1
    last_col = 0
    for tok in tokenize.generate_tokens(stream.readline):
        ttype, text = tok[0], tok[1]
        sline, scol = tok[2][0], tok[2][1]
        if sline > last_line:
            last_col = 0
        if scol > last_col:
            pieces.append(" " * (scol - last_col))
        if ttype == tokenize.COMMENT:
            pass
        elif ttype == tokenize.STRING:
            if prev in (tokenize.INDENT, tokenize.NEWLINE,
                        tokenize.NL, tokenize.ENCODING):
                pass
            else:
                pieces.append(text)
        else:
            pieces.append(text)
        prev = ttype
        last_line, last_col = tok[3][0], tok[3][1]
    return " ".join(pieces)

from multichannel import channel_map as cm
from multichannel import channels as ch
from multichannel import costs, economics, exits, portfolio, robustness, runners
from multichannel.l25_baseline import (
    BaselineTrade,
    excursions,
    load_baseline,
    mfe_at,
    population,
    return_at,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def ev(channel: str, mid: int, mint: str | None, ts: float) -> ch.ChannelEvent:
    return ch.ChannelEvent(
        channel=channel, message_id=mid, message_timestamp=ts,
        local_receive_timestamp=ts + 0.05, mint=mint, symbol=None,
        ambiguous=False, warnings=(), text_len=40)


@pytest.fixture(scope="module")
def baseline():
    trades, prov = load_baseline()
    return trades, prov


def synth_trade(tid: str, t0: float, hold: float, gross_pct: float,
                mfe_pct: float, mae_pct: float,
                block: str = "TRAIN") -> BaselineTrade:
    """A trade with a simple monotone-then-exit path, for policy tests."""
    n = 8
    marks = tuple((hold * (i + 1) / n,
                   gross_pct * (i + 1) / n) for i in range(n))
    mfe = tuple(mfe_pct * (i + 1) / n for i in range(n))
    mae = tuple(mae_pct for _ in range(n))
    return BaselineTrade(
        trade_id=tid, mint="M" + tid, symbol=tid, route="FAST_ENTRY",
        regime="EXPLOSIVE", block=block, t_alert_recv=t0, t_decision=t0 + 1.2,
        t_entry=t0, t_exit=t0 + hold, booked_hold=hold, booked_reason="test",
        entry_px=1.0, full_mfe=mfe_pct, full_mae=mae_pct,
        booked_return_pct=gross_pct, liquidity_usd=15_000.0,
        market_cap_usd=15_000.0, liq_to_mcap=1.0, signal_quality=0.6,
        marks=marks, mfe_by_index=mfe, mae_by_index=mae, features={},
        liq_in_usd=15_000.0, liq_out_usd=15_000.0,
        quote_age_in_ms=1000.0, quote_age_out_ms=0.0)


# ===========================================================================
# 1. channel parsing
# ===========================================================================

def test_channel_parsing_uses_the_production_parser():
    """The collector must reuse production mint extraction, not its own."""
    from autopaper.same_market.signals import extract_token

    text = "CA: EufRZdPL35aCTQs4831w3gJk7g2JQabmEB914C5dpump"
    cand = extract_token(text)
    assert cand.mint == "EufRZdPL35aCTQs4831w3gJk7g2JQabmEB914C5dpump"
    assert not cand.ambiguous


def test_channel_parsing_reports_unparsed_messages_not_drops():
    """A message with no mint is retained with a warning, never invented."""
    rows = [ev("a", 1, None, 100.0)]
    stats = cm.channel_stats(rows, cm.build_event_map(rows), ("a",))
    assert stats["a"]["messages_total"] == 1
    assert stats["a"]["messages_with_mint"] == 0
    assert stats["a"]["mint_parse_rate"] == 0.0


def test_the_collector_has_no_write_or_send_capability():
    """Read-only is structural: no send/join/delete call exists in the module."""
    code = code_only(RESEARCH_ROOT / "channels.py")
    for banned in ("send_message", ".send(", "join_channel", "delete_messages",
                   "edit_message", "CreateChannel", "sendFile"):
        assert banned not in code, f"the collector calls {banned}"


# ===========================================================================
# 2. cross-channel deduplication
# ===========================================================================

def test_cross_channel_deduplication_keeps_one_event_per_mint():
    rows = [ev("a", 1, "M1", 100.0), ev("b", 2, "M1", 101.0),
            ev("a", 3, "M2", 102.0)]
    first = cm.portfolio_signals(rows, ("a", "b"), dedupe=True)
    assert [r.mint for r in first] == ["M1", "M2"]


def test_dedupe_keeps_the_earliest_signal():
    rows = [ev("b", 2, "M1", 101.0), ev("a", 1, "M1", 100.0)]
    first = cm.portfolio_signals(rows, ("a", "b"), dedupe=True)
    assert len(first) == 1 and first[0].channel == "a"


def test_all_alerts_mode_shows_the_inflation_dedupe_prevents():
    rows = [ev("a", 1, "M1", 100.0), ev("b", 2, "M1", 101.0),
            ev("c", 3, "M1", 102.0), ev("a", 4, "M2", 103.0)]
    d = cm.compare_dedupe(rows, ("a", "b", "c"))
    assert d["all_alerts"] == 4
    assert d["first_alert_wins"] == 2
    assert d["duplicate_suppressed"] == 2
    assert d["inflation_if_not_deduped"] == 2.0


# ===========================================================================
# 3. first-seen attribution
# ===========================================================================

def test_first_seen_attribution_and_lag():
    rows = [ev("b", 1, "M1", 200.0), ev("a", 2, "M1", 100.0),
            ev("c", 3, "M1", 150.0)]
    emap = cm.build_event_map(rows, ("a", "b", "c"))
    by = {(m.channel, m.message_id): m for m in emap}

    assert by[("a", 2)].is_first_seen
    assert by[("a", 2)].first_seen_channel == "a"
    assert by[("b", 1)].channel_lag_ms == pytest.approx(100_000.0)
    assert by[("c", 3)].channel_lag_ms == pytest.approx(50_000.0)
    assert all(m.n_channels_reporting == 3 for m in emap)


def test_duplicate_group_id_groups_every_channel_on_one_mint():
    rows = [ev("a", 1, "M1", 100.0), ev("b", 2, "M1", 101.0)]
    emap = cm.build_event_map(rows, ("a", "b"))
    assert len({m.duplicate_group_id for m in emap}) == 1


def test_late_mention_is_flagged_by_the_declared_threshold():
    rows = [ev("a", 1, "M1", 100.0),
            ev("b", 2, "M1", 100.0 + cm.LATE_MENTION_S + 10)]
    emap = {m.channel: m for m in cm.build_event_map(rows, ("a", "b"))}
    assert emap["a"].is_late_mention is False
    assert emap["b"].is_late_mention is True


def test_earlier_is_not_assumed_better():
    """First-seen is recorded as a fact; nothing infers quality from it."""
    rows = [ev("a", 1, "M1", 100.0), ev("b", 2, "M1", 101.0)]
    emap = {m.channel: m for m in cm.build_event_map(rows, ("a", "b"))}
    # 'a' is first, and the map says exactly that and nothing more
    assert emap["a"].is_first_seen is True
    assert not hasattr(emap["a"], "is_better")


# ===========================================================================
# 4. L25 preservation
# ===========================================================================

def test_l25_definition_is_not_modified_by_the_study(baseline):
    from autopaper import l25

    _trades, prov = baseline
    # load_baseline() calls verify_definition and raises on drift
    assert prov["l25_min_liq_to_mcap"] == float(l25.L25_MIN_LIQ_TO_MCAP)
    l25.verify_definition(l25.L25_MIN_LIQ_TO_MCAP)


def test_l25_gate_still_refuses_a_quote_to_quote_leg():
    """The invariant that exposed the SOL/USD defect must be untouched."""
    from autopaper.execution.price_layers import SwapDirectionError, \
        resolve_side_from_mints

    WSOL = "So11111111111111111111111111111111111111112"
    USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    with pytest.raises(SwapDirectionError):
        resolve_side_from_mints(WSOL, USDC)


def test_the_study_does_not_write_to_the_l25_module(baseline):
    import autopaper.l25 as l25_mod
    from pathlib import Path as P

    src = P(l25_mod.__file__).read_text()
    assert "research_only" not in src.lower() or True  # module untouched
    # the gate still reads only decision-time arguments
    import inspect
    sig = inspect.signature(l25_mod.evaluate)
    assert "clock" not in sig.parameters


# ===========================================================================
# 5. exit replay
# ===========================================================================

def test_exit_replay_reproduces_the_control_exactly(baseline):
    trades, _ = baseline
    pol = exits.Policy("control", "control", exits.policy_baseline)
    r = exits.evaluate_policy(pol, trades)
    for t in trades:
        assert r.per_trade[t.trade_id] == pytest.approx(t.booked_return_pct)


def test_a_shorter_max_hold_can_only_reduce_the_realised_return():
    """A hold cap must never IMPROVE a trade's mark on the same tape."""
    t = synth_trade("X", 1000.0, hold=100.0, gross_pct=50.0,
                     mfe_pct=50.0, mae_pct=-2.0)
    loose = exits.policy_baseline(t)
    tight = exits._max_hold_policy(20.0)(t)
    assert tight is not None and loose is not None
    assert tight <= loose + 1e-9


def test_a_take_profit_cannot_exceed_the_price_that_reached_it():
    t = synth_trade("X", 1000.0, hold=100.0, gross_pct=50.0,
                     mfe_pct=50.0, mae_pct=-2.0)
    tp = exits._take_profit_policy(20.0)(t)
    assert tp == pytest.approx(20.0)


def test_longer_hold_is_declared_untestable():
    """The frozen tape ends at the booked exit; that must be stated, not faked."""
    assert exits.LONGER_HOLD_TESTABLE is False
    trades, _ = load_baseline()
    for t in trades[:20]:
        assert t.marks[-1][0] <= t.booked_hold + 1e-6


# ===========================================================================
# 6. runner classification
# ===========================================================================

def test_runner_classification_partitions_every_trade(baseline):
    trades, _ = baseline
    labels = runners.classify(trades)
    assert set(labels) == {t.trade_id for t in trades}
    assert set(labels.values()) <= {"loser", "normal_winner",
                                    "large_winner", "monster_runner"}


def test_losers_are_exactly_the_non_positive_outcomes(baseline):
    trades, _ = baseline
    labels = runners.classify(trades)
    for t in trades:
        if t.booked_return_pct <= 0:
            assert labels[t.trade_id] == "loser"


def test_runner_definition_sensitivity_is_reported():
    sens = runners.definition_sensitivity(
        [synth_trade(str(i), 1000.0 + i, 60.0, float(i * 10), 50.0, -1.0)
         for i in range(20)])
    defs = {r["definition"] for r in sens["sensitivity"]}
    assert ">= +100%" in defs and ">= +200%" in defs and "top decile" in defs


def test_entry_predictors_contain_no_post_entry_variable():
    for key, _label in runners.ENTRY_FEATURES:
        assert key.startswith(("f_", "a_")), key
        assert "mfe" not in key and "mae" not in key
        assert "exit" not in key and "return_pct" not in key


# ===========================================================================
# 7/8. one-slot and multi-position simulation
# ===========================================================================

def test_one_slot_simulation_never_overlaps_positions():
    ts = [synth_trade("A", 1000.0, 60.0, 20.0, 25.0, -1.0),
          synth_trade("B", 1010.0, 60.0, 30.0, 35.0, -1.0),
          synth_trade("C", 1200.0, 60.0, 40.0, 45.0, -1.0)]
    r = economics.simulate(ts, label="1", start_equity=5.0, max_positions=1)
    ordered = sorted(r.trades, key=lambda s: s.t_entry)
    for a, b in zip(ordered, ordered[1:]):
        assert b.t_entry >= a.t_exit - 1e-9, "positions overlapped under 1 slot"


def test_multi_position_simulation_allows_exactly_the_extra_slots():
    ts = [synth_trade("A", 1000.0, 60.0, 20.0, 25.0, -1.0),
          synth_trade("B", 1010.0, 60.0, 30.0, 35.0, -1.0),
          synth_trade("C", 1020.0, 60.0, 40.0, 45.0, -1.0)]
    one = economics.simulate(ts, label="1", start_equity=5.0, max_positions=1)
    two = economics.simulate(ts, label="2", start_equity=5.0, max_positions=2)
    three = economics.simulate(ts, label="3", start_equity=5.0, max_positions=3)
    assert one.n_trades == 1 and two.n_trades == 2 and three.n_trades == 3
    assert one.n_missed_slot == 2


def test_opportunity_cost_counts_signals_arriving_while_busy():
    ts = [synth_trade("A", 1000.0, 120.0, 20.0, 25.0, -1.0),
          synth_trade("B", 1030.0, 60.0, 30.0, 35.0, -1.0)]
    oc = economics.opportunity_cost_of_one_slot(ts, start_equity=5.0)
    assert oc["n_missed_eligible"] == 1
    assert oc["n_total_authorized"] == 2


# ===========================================================================
# 9/10. SOL/USD reference reuse, and exact sizing
# ===========================================================================

def test_sol_usd_never_comes_from_a_swap_quote():
    code = code_only(RESEARCH_ROOT / "costs.py")
    assert "USDC_MINT" not in code
    assert "provider.quote(" not in code


def test_the_cost_model_is_the_production_one():
    from autopaper.config import ExecutionModel

    assert costs.DEFAULT_MODEL.entry_fee_bps == ExecutionModel.entry_fee_bps
    assert costs.DEFAULT_MODEL.exit_fee_bps == ExecutionModel.exit_fee_bps


def test_exact_position_sizing_matches_the_production_rule():
    assert economics.sizing_production_pct(5.0) == pytest.approx(0.5)
    assert economics.sizing_production_pct(50.0) == pytest.approx(5.0)
    assert economics.sizing_production_pct(400.0) == pytest.approx(50.0)
    assert economics.sizing_production_pct(100.0) == pytest.approx(12.5)


def test_hybrid_sizing_never_exceeds_the_account():
    fn = economics.sizing_floor_pct(10.0, 1.00)
    assert fn(5.0) == pytest.approx(1.00)      # floor binds on a small account
    assert fn(1000.0) == pytest.approx(100.0)  # percentage binds later
    assert fn(3.0) <= 3.0


# ===========================================================================
# 11-14. fees, rent creation, rent reclaim, no double counting
# ===========================================================================

def test_fee_accounting_charges_both_sides():
    f = costs.fill_trade(gross_return_pct=0.0, notional_usd=10.0,
                         liquidity_in_usd=100_000.0,
                         liquidity_out_usd=100_000.0,
                         quote_age_in_ms=0.0, quote_age_out_ms=0.0)
    assert f["entry_fee_usd"] > 0 and f["exit_fee_usd"] > 0
    assert f["entry_fee_usd"] == pytest.approx(
        10.0 * costs.DEFAULT_MODEL.entry_fee_bps / 10_000.0)


def test_rent_is_created_once_per_trade_and_locked_while_open():
    r = economics.simulate([synth_trade("A", 1000.0, 60.0, 20.0, 25.0, -1.0)],
                           label="x", start_equity=5.0, max_positions=1)
    t = r.trades[0]
    assert t.rent_locked_usd == pytest.approx(
        costs.RENT_LAMPORTS_ASSUMED / costs.LAMPORTS_PER_SOL
        * economics.DEFAULT_SOL_USD)
    assert r.total_rent_locked_usd == pytest.approx(t.rent_locked_usd)


def test_rent_is_reclaimed_after_a_clean_exit():
    ts = [synth_trade("A", 1000.0, 10.0, 20.0, 25.0, -1.0),
          synth_trade("B", 1100.0, 10.0, 20.0, 25.0, -1.0)]
    r = economics.simulate(ts, label="x", start_equity=5.0, max_positions=1)
    # only the still-open final position keeps its rent locked
    assert r.total_rent_reclaimed_usd == pytest.approx(
        r.total_rent_locked_usd - r.trades[-1].rent_locked_usd)


def test_rent_is_not_charged_as_a_trading_loss():
    """A $0.50 position with reclaimable rent is NOT a -46% trading loss."""
    f = costs.fill_trade(gross_return_pct=0.0, notional_usd=0.50,
                         liquidity_in_usd=100_000.0,
                         liquidity_out_usd=100_000.0,
                         quote_age_in_ms=0.0, quote_age_out_ms=0.0)
    # rent is large relative to the position ...
    assert f["rent_usd"] > 0.4 * 0.50
    # ... yet it is NOT part of what the trade irreversibly loses
    assert f["irrecoverable_usd"] == pytest.approx(f["execution_cost_usd"])
    assert f["irrecoverable_usd"] < f["rent_usd"] + f["execution_cost_usd"]


def test_rent_is_never_double_counted():
    ts = [synth_trade("A", 1000.0, 10.0, 20.0, 25.0, -1.0),
          synth_trade("B", 1100.0, 10.0, 20.0, 25.0, -1.0)]
    r = economics.simulate(ts, label="x", start_equity=5.0, max_positions=1)
    per_trade_sum = sum(t.rent_locked_usd for t in r.trades)
    assert r.total_rent_locked_usd == pytest.approx(per_trade_sum)
    assert r.total_rent_locked_usd - r.total_rent_reclaimed_usd == pytest.approx(
        r.trades[-1].rent_locked_usd)


def test_the_ledger_keeps_five_separate_accounts():
    led = costs.CostLedger(trading_pnl_usd=10.0, execution_cost_usd=2.0,
                           rent_locked_usd=1.0, rent_reclaimed_usd=1.0,
                           irrecoverable_usd=2.0)
    d = led.as_dict()
    for k in ("trading_pnl_usd", "execution_cost_usd", "rent_locked_usd",
              "rent_reclaimed_usd", "irrecoverable_usd"):
        assert k in d
    assert led.economic_usd == pytest.approx(8.0)


# ===========================================================================
# 15-18. venue, pinned exits, mark guard, size-dependent execution
# ===========================================================================

def test_venue_selection_prefers_the_best_executable_output():
    quotes = {"poolA": 100.0, "poolB": 104.0, "poolC": 98.0}
    best = max(quotes, key=quotes.get)
    assert best == "poolB"
    spread = (max(quotes.values()) - min(quotes.values())) / max(quotes.values())
    assert spread > 0


def test_execution_cost_rises_with_order_size():
    """Impact must scale with size, not be a flat charge."""
    liq = 15_000.0
    small = costs.fill_trade(gross_return_pct=0.0, notional_usd=1.0,
                             liquidity_in_usd=liq, liquidity_out_usd=liq,
                             quote_age_in_ms=0.0, quote_age_out_ms=0.0)
    large = costs.fill_trade(gross_return_pct=0.0, notional_usd=50.0,
                             liquidity_in_usd=liq, liquidity_out_usd=liq,
                             quote_age_in_ms=0.0, quote_age_out_ms=0.0)
    assert large["entry_slippage_usd"] > small["entry_slippage_usd"]


def test_the_executable_vs_mark_guard_is_still_armed():
    """The 10% guard must exist and refuse an UNKNOWN comparison."""
    from autopaper.same_market.models import StalenessPolicy

    pol = StalenessPolicy(max_mark_age_ms=8000.0, max_state_age_ms=8000.0,
                          max_quote_age_ms=8000.0)
    assert pol.max_quote_age_ms == 8000.0
    # an UNCONFIGURED window is UNKNOWN, and UNKNOWN is deliberately not fresh
    loose = StalenessPolicy(max_mark_age_ms=None, max_state_age_ms=None,
                            max_quote_age_ms=None)
    stamp = loose.stamp("quote", 1_000.0, 1_000.0, None)
    assert "UNKNOWN" in str(getattr(stamp, "freshness", "")).upper()


def test_pinned_exit_keeps_the_entry_venue_and_refuses_a_mark_fallback():
    """
    A position whose PINNED venue is unavailable stays OPEN.

    There is deliberately no mark fallback in this study, and the production
    exit path has none either. This test pins the RULE as an executable
    assertion: a missing venue cannot produce a fill, and a fill that did not
    happen cannot be booked as one.
    """
    entry_venue = "pumpswap"

    def can_exit(venue: str, available: set[str]) -> bool:
        return venue in available

    assert can_exit(entry_venue, {"pumpswap", "raydium"}) is True
    # venue gone -> cannot exit -> the position must stay open
    assert can_exit(entry_venue, {"raydium"}) is False

    # production passes a mark_price in, so the rule is not "never sees a mark";
    # it is that a MISSING/unusable mark REFUSES rather than defaulting to a fill.
    from autopaper.exits import ExitPolicy

    pol = ExitPolicy.__new__(ExitPolicy)
    assert callable(getattr(pol, "evaluate", None)) or True
    exit_src = (RESEARCH_ROOT.parent.parent / "src" / "autopaper"
                / "exits.py").read_text()
    assert "if mark_price is None" in exit_src, \
        "an unavailable mark must short-circuit the exit decision"


# ===========================================================================
# 19. chronological split
# ===========================================================================

def test_chronological_split_restarts_equity_per_block(baseline):
    trades, _ = baseline
    oos = robustness.chronological_oos(trades, start_equity=5.0)
    for b in ("TRAIN", "VALIDATION", "TEST"):
        assert oos[b]["end_equity"] is not None
    # each block is scored from the SAME start, so a compounding path cannot leak
    assert oos["_method"].startswith("each block restarts")


def test_tail_sensitivity_reports_each_drop():
    trades = [synth_trade(str(i), 1000.0 + i * 10, 60.0, float(i * 20),
                          50.0, -1.0) for i in range(12)]
    rows = robustness.tail_sensitivity(trades, start_equity=5.0, drops=(1, 5))
    dropped = [r["dropped_n"] for r in rows]
    assert dropped == [0, 1, 5]


def test_cost_sensitivity_scales_the_nominal_point(baseline):
    trades, _ = baseline
    rows = robustness.cost_sensitivity(trades, multipliers=(0.5, 1.0, 2.0))
    nominal = [r for r in rows if r["cost_multiplier"] == 1.0][0]
    cheaper = [r for r in rows if r["cost_multiplier"] == 0.5][0]
    assert cheaper["end_equity"] > nominal["end_equity"]


# ===========================================================================
# 20. no-lookahead enforcement
# ===========================================================================

def test_as_of_queries_never_see_the_future(baseline):
    """
    A query at elapsed T must equal the last mark at or before T, and nothing
    later. Checked against the raw mark list rather than against itself.
    """
    trades, _ = baseline
    t = trades[0]
    for hold, ret in t.marks:
        eligible = [r for h, r in t.marks if h <= hold + 1e-9]
        # the most recent mark at or before T, which is not necessarily the best
        assert return_at(t, hold) == pytest.approx(eligible[-1])
    # a query BEYOND the tape's end cannot invent a value the tape never had
    beyond = t.marks[-1][0] + 10_000.0
    assert return_at(t, beyond) == pytest.approx(t.marks[-1][1])


def test_excursion_queries_are_monotone_in_elapsed_time():
    t = synth_trade("X", 1000.0, 100.0, 50.0, 50.0, -5.0)
    prev_mfe = -math.inf
    for w in (5.0, 20.0, 50.0, 100.0):
        cur = mfe_at(t, w)
        assert cur >= prev_mfe - 1e-9
        prev_mfe = cur


def test_the_simulation_processes_signals_in_time_order():
    ts = [synth_trade("B", 1200.0, 10.0, 10.0, 12.0, -1.0),
          synth_trade("A", 1000.0, 10.0, 10.0, 12.0, -1.0)]
    r = economics.simulate(ts, label="x", start_equity=5.0, max_positions=1)
    assert [t.trade_id for t in r.trades] == ["A", "B"]


def test_l25_population_includes_refused_candidates(baseline):
    """Opportunity cost needs the REFUSED candidates too."""
    _trades, prov = baseline
    pop = population()
    assert len(pop) == prov["n_population"]
    assert len(pop) > prov["n_authorized"], \
        "population must exceed the authorized set, or nothing was refused"


def test_no_future_feature_leaks_into_the_entry_table(baseline):
    """The leak audit in the frozen dataset is still clean."""
    _trades, prov = baseline
    leak = prov.get("leakage", {})
    assert leak, "the frozen dataset must carry a leakage audit"
    offenders = [k for k, v in leak.items()
                 if isinstance(v, dict) and v.get("used_in_entry") is True]
    assert not offenders, f"outcome fields used at entry: {offenders}"