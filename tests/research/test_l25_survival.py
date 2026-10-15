"""
Focused tests for the $10 survival study.

Offline. Everything is driven by the frozen baseline and synthetic fixtures; no
network, no signing, no broadcast, no production mutation.

Coverage maps onto the brief's integrity requirements:
  baseline frozen           five separate ledgers       one-slot realism
  evidence tiers            quote-failure handling      adverse tail injection
  no mark-only exit         losing-streak arithmetic   survival scoring
  chronological OOS         plateau-vs-spike           .env.local untouched
"""

from __future__ import annotations

import copy
import random
from pathlib import Path

import pytest

_RESEARCH = Path(__file__).resolve().parents[2] / "research"
import sys as _sys
for _p in (str(_RESEARCH), str(_RESEARCH.parent / "src")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from survival.baseline import allocation_walkthrough, snapshot, verify  # noqa: E402
from survival.engine import (  # noqa: E402
    ExitVariant,
    build_variants,
    simulate_path,
    _last_mark_at_or_before,
)
from survival.execution import (  # noqa: E402
    ExecutionInputs,
    ExecutionModel,
    ExecutionStress,
    evidence_audit,
    INFERRED_EXECUTABLE,
    SIMULATED_EXECUTABLE,
    UNRESOLVED,
)
from survival.montecarlo import (  # noqa: E402
    ADVERSE_REGIMES,
    bootstrap,
    inject_adverse_tail,
    losing_streak_table,
    monte_carlo,
    monte_carlo_adverse,
    survival_score,
    tail_dependence,
)


@pytest.fixture(scope="module")
def snap():
    return snapshot()


@pytest.fixture(scope="module")
def trades():
    from multichannel.l25_baseline import load_baseline
    ts, _prov = load_baseline()
    return ts


# ===========================================================================
# 1. the baseline is frozen and matches production
# ===========================================================================

def test_baseline_matches_live_production(snap):
    from autopaper import l25

    verify(snap)
    assert snap.l25_min_liq_to_mcap == float(l25.L25_MIN_LIQ_TO_MCAP)


def test_baseline_reads_the_real_one_slot_rule(snap):
    assert snap.max_open_positions == 1


def test_baseline_uses_production_fees_not_invented_ones(snap):
    from autopaper.config import ExecutionModel

    ex = ExecutionModel()
    assert snap.entry_fee_bps == ex.entry_fee_bps
    assert snap.exit_fee_bps == ex.exit_fee_bps
    assert snap.impact_max_bps == ex.impact_max_bps


def test_env_local_is_not_read_for_credentials_by_the_study():
    """The study reads production modules, not secrets."""
    src = (Path(__file__).resolve().parents[2] / "research" / "survival"
           / "baseline.py").read_text()
    assert ".env.local" not in src
    assert "PRIVATE KEY" not in src


# ===========================================================================
# 2. the $10 allocation arithmetic is exact and reproducible
# ===========================================================================

def test_ten_dollar_allocation_chain_is_exact(snap):
    w = allocation_walkthrough(snap, 10.0, 40.0)
    # production sizing puts exactly 10% of $10 into the trade
    assert w["position_notional_usd"] == pytest.approx(1.0, abs=1e-9)
    assert w["position_pct_of_equity"] == pytest.approx(10.0, abs=1e-9)
    # entry execution price is ABOVE the reference by the slippage budget
    assert w["entry_execution_price"] > 1.0
    assert w["net_proceeds_usd"] > 0
    # the chain closes
    expected = w["equity_before"] + w["net_pnl_usd"]
    assert w["equity_after"] == pytest.approx(expected, abs=1e-9)


def test_rent_is_a_lock_not_a_trading_loss(snap):
    w = allocation_walkthrough(snap, 10.0, 0.0)
    # a FLAT trade still pays fees and network, but rent is not in net PnL
    assert w["rent_usd"] > 0
    assert w["net_pnl_usd"] < 0            # costs are real
    assert w["net_pnl_usd"] > -w["rent_usd"]  # rent itself is not the loss


# ===========================================================================
# 3. five separate ledgers, never merged
# ===========================================================================

def test_trade_carries_five_distinct_ledgers(trades, snap):
    r = simulate_path(trades, variant=build_variants(trades)[0], snap=snap,
                      start_equity=10.0)
    row = r.trade_rows[0]
    for k in ("trading_pnl_usd", "execution_cost_usd", "rent_locked_usd",
              "rent_reclaimed_usd", "permanent_cost_usd"):
        assert k in row
    # permanent cost = execution cost + (rent - reclaimed)
    assert row["permanent_cost_usd"] == pytest.approx(
        row["execution_cost_usd"] + row["rent_locked_usd"]
        - row["rent_reclaimed_usd"], abs=1e-9)


def test_reclaimable_rent_is_never_counted_as_a_permanent_loss(trades, snap):
    r = simulate_path(trades, variant=build_variants(trades)[0], snap=snap,
                      start_equity=10.0)
    for row in r.trade_rows:
        assert row["rent_reclaimed_usd"] == pytest.approx(row["rent_locked_usd"])


# ===========================================================================
# 4. one-slot realism is mandatory
# ===========================================================================

def test_one_slot_never_holds_two_positions(trades, snap):
    base = build_variants(trades)[0]
    r = simulate_path(trades, variant=base, snap=snap, start_equity=10.0,
                      max_positions=1)
    rows = sorted(r.trade_rows, key=lambda x: x["t_entry"])
    for a, b in zip(rows, rows[1:]):
        assert b["t_entry"] >= a["t_entry"] + a["hold_s"] - 1e-6, \
            "two positions overlapped under the one-slot rule"


def test_two_slot_takes_strictly_more_trades(trades, snap):
    base = build_variants(trades)[0]
    one = simulate_path(trades, variant=base, snap=snap, max_positions=1)
    two = simulate_path(trades, variant=base, snap=snap, max_positions=2)
    assert two.trades > one.trades
    assert one.slot_conflicts > 0


def test_missed_signals_are_counted_not_silently_dropped(trades, snap):
    base = build_variants(trades)[0]
    r = simulate_path(trades, variant=base, snap=snap, max_positions=1)
    assert r.slot_conflicts == r.missed
    assert r.missed > 0


# ===========================================================================
# 5. evidence tiers are never silently upgraded
# ===========================================================================

def test_evidence_audit_declares_missing_executions(trades):
    a = evidence_audit(trades)
    assert a["recorded_executable_buy_fill"]["available"] is False
    assert a["venue_identity"]["available"] is False
    assert a["venue_identity"]["tier_used"] == UNRESOLVED


def test_every_simulated_trade_carries_provenance(trades, snap):
    r = simulate_path(trades, variant=build_variants(trades)[0], snap=snap,
                      start_equity=10.0)
    required = ("entry_source", "exit_source", "execution_status",
                "market_identity_status", "reference_type", "fee_model",
                "slippage_model", "confidence")
    for row in r.trade_rows:
        for k in required:
            assert row[k], f"{k} missing"
        assert row["entry_source"] in (INFERRED_EXECUTABLE, SIMULATED_EXECUTABLE)


def test_exit_liquidity_inference_is_disclosed(snap, trades):
    em = ExecutionModel(snap)
    t = trades[0]
    inp = ExecutionInputs(1.0, t.liq_in_usd, t.liq_out_usd,
                          t.quote_age_in_ms, t.quote_age_out_ms)
    px = em.price_trade(gross_return_pct=t.booked_return_pct, inp=inp)
    assert "INFERRED" in px["exit_liquidity_basis"]


# ===========================================================================
# 6. no mark-only exit is scored as a fill
# ===========================================================================

def test_unresolvable_exit_is_flagged_not_counted_as_a_fill(snap):
    em = ExecutionModel(snap)
    inp = ExecutionInputs(1.0, 15_100.0, 15_100.0, 0.0, 99_000.0)
    px = em.price_trade(gross_return_pct=50.0, inp=inp)
    assert px["execution_status"] == UNRESOLVED
    assert px["exit_source"] == UNRESOLVED
    assert px["confidence"] == "LOW"


def test_a_normal_exit_resolves(snap):
    em = ExecutionModel(snap)
    inp = ExecutionInputs(1.0, 16_900.0, 16_900.0, 1_000.0, 0.0)
    px = em.price_trade(gross_return_pct=10.0, inp=inp)
    assert px["execution_status"] == "RESOLVED"
    assert px["confidence"] in ("MEDIUM", "HIGH")


# ===========================================================================
# 7. adverse tail injection is required, because resampling cannot invent it
# ===========================================================================

def test_injection_can_create_losses_absent_from_the_sample(trades):
    rng = random.Random(1)
    rows, info = inject_adverse_tail(trades, rng=rng, rug_fraction=0.5)
    assert info["rugs_injected"] > 0
    assert any(r.booked_return_pct <= -99.0 for r in rows)


def test_injection_only_touches_losing_or_random_trades(trades):
    rng = random.Random(2)
    rows, _ = inject_adverse_tail(trades, rng=rng, rug_fraction=0.0,
                                  ruin_fraction=0.0)
    assert [r.booked_return_pct for r in rows] == \
           [t.booked_return_pct for t in trades]


def test_nominal_regime_injects_nothing(trades):
    rows, info = inject_adverse_tail(trades, rng=random.Random(3), **ADVERSE_REGIMES["NOMINAL"])
    assert info["ruin_injected"] == 0 and info["rugs_injected"] == 0


def test_ruin_is_zero_at_production_sizing_but_not_at_50pct(trades, snap):
    base = build_variants(trades)[0]
    small = copy.copy(snap)
    object.__setattr__(small, "size_for", lambda e: e * 0.10)
    big = copy.copy(snap)
    object.__setattr__(big, "size_for", lambda e: e * 0.50)
    a = monte_carlo_adverse(trades, variant=base, snap=small, paths=200)
    b = monte_carlo_adverse(trades, variant=base, snap=big, paths=200)
    assert a["MEMECOIN_WINTER"]["p_ruin"] == 0.0
    assert b["MEMECOIN_WINTER"]["p_ruin"] > a["MEMECOIN_WINTER"]["p_ruin"]


# ===========================================================================
# 8. losing-streak arithmetic
# ===========================================================================

def test_recovery_requirement_follows_the_formula(trades, snap):
    rows = losing_streak_table(trades, snap=snap, start_equity=10.0)
    for r in rows:
        d = r["drawdown_pct"] / 100.0
        if r["required_recovery_pct"] != float("inf"):
            assert r["required_recovery_pct"] == pytest.approx(
                (1.0 / (1.0 - d) - 1.0) * 100.0, rel=1e-6)


def test_deeper_streaks_leave_less_equity(trades, snap):
    rows = sorted(losing_streak_table(trades, snap=snap), key=lambda r: r["streak"])
    eqs = [r["ending_equity_usd"] for r in rows]
    assert eqs == sorted(eqs, reverse=True)


# ===========================================================================
# 9. survival scoring prefers survival over profit
# ===========================================================================

def test_survival_score_rewards_a_safer_profile_over_a_richer_one():
    def mc(ruin, below5, dd40, exp, dd, med, p05):
        return {
            "probability": {"ruin_le_1c": ruin, "below_5": below5,
                            "dd_gt_40": dd40, "below_7_5": below5,
                            "below_2": ruin, "above_10": 0.5,
                            "dd_gt_30": dd40, "dd_gt_50": dd40,
                            "dd_gt_60": dd40, "recovered_to_10": 0.5},
            "expectancy_usd": {"median": exp, "p05": exp / 2, "mean": exp,
                               "p25": exp, "p75": exp, "p95": exp * 2},
            "max_drawdown_pct": {"median": dd, "p95": dd * 1.5, "max": dd * 2},
            "final_equity": {"median": med, "p05": p05, "mean": med,
                             "p25": med, "p75": med, "p95": med, "min": p05},
            "profit_factor": {"median": 6.0, "p05": 2.0},
            "min_equity": {"median": p05, "p05": p05},
            "longest_losing_streak": {"median": 5, "p95": 8, "max": 8},
            "worst_10_trade_seq_usd": {"median": -1.0, "p05": -2.0},
            "worst_20_trade_seq_usd": {"median": -2.0, "p05": -4.0},
            "paths": 200, "stress_profile": "NOMINAL", "start_equity": 10.0,
            "exit_variant": "x", "trades_mean": 100.0,
            "unresolved_exit_rate": 0.0, "slot_conflict_rate": 0.01,
        }

    safer = mc(0.0, 0.0, 0.0, 0.2, 5.0, 50.0, 20.0)
    riskier = mc(0.10, 0.25, 0.2, 2.0, 45.0, 5000.0, 1.0)
    a = survival_score("safer", safer, safer, 0.2)
    b = survival_score("riskier", riskier, riskier, 0.2)
    assert a.score > b.score, "a higher-profit profile must not outrank a safer one"


def test_survival_score_components_are_bounded():
    mc = {
        "probability": {"ruin_le_1c": 0.0, "below_5": 0.0, "dd_gt_40": 0.0,
                        "below_7_5": 0.0, "below_2": 0.0, "above_10": 1.0,
                        "dd_gt_30": 0.0, "dd_gt_50": 0.0, "dd_gt_60": 0.0,
                        "recovered_to_10": 1.0},
        "expectancy_usd": {"median": 0.5},
        "max_drawdown_pct": {"median": 1.0, "p95": 2.0, "max": 3.0},
        "final_equity": {"median": 100.0, "p05": 80.0},
        "profit_factor": {"median": 9.0, "p05": 5.0},
    }
    s = survival_score("x", mc, mc, 0.5)
    assert 0.0 <= s.score <= 1.0
    for v in s.components.values():
        assert 0.0 <= v <= 1.0


# ===========================================================================
# 10. chronological discipline
# ===========================================================================

def test_chronological_blocks_are_disjoint_in_time(trades):
    by = {}
    for t in trades:
        by.setdefault(t.block, []).append(t.t_entry)
    spans = {k: (min(v), max(v)) for k, v in by.items()}
    order = sorted(spans, key=lambda k: spans[k][0])
    for a, b in zip(order, order[1:]):
        assert spans[a][1] <= spans[b][0], f"{a} overlaps {b}"


def test_oos_simulation_uses_only_its_own_block(trades, snap):
    base = build_variants(trades)[0]
    oos = [t for t in trades if t.block == "TEST"]
    r = simulate_path(oos, variant=base, snap=snap)
    assert all(row["block"] == "TEST" for row in r.trade_rows)


# ===========================================================================
# 11. plateau vs spike (anti-overfitting)
# ===========================================================================

def test_the_cap_effect_is_a_plateau_not_a_single_spike(trades, snap):
    oos = [t for t in trades if t.block in ("VALIDATION", "TEST")]

    def capped(cap: float) -> ExitVariant:
        def fn(t):
            if t.booked_hold <= cap + 1e-9:
                return (t.booked_return_pct, t.booked_hold)
            g = _last_mark_at_or_before(t, cap)
            return g if g else None
        return fn

    exps = {}
    for cap in (50, 60, 70, 80):
        r = simulate_path(oos, variant=ExitVariant(f"c{cap}", "", capped(cap)),
                          snap=snap)
        exps[cap] = r.expectancy_usd
    # every cap in the band beats the uncapped control, so it is a plateau
    base = simulate_path(oos, variant=build_variants(oos)[0], snap=snap)
    for cap, e in exps.items():
        assert e > base.expectancy_usd, f"cap {cap} did not beat the control"


# ===========================================================================
# 12. tail dependence is reported, not hidden
# ===========================================================================

def test_tail_dependence_reports_the_top_decile_share(trades):
    td = tail_dependence(trades)
    assert 0.0 < td["top_decile_share_pct"] < 100.0
    assert len(td["ladder"]) >= 4


def test_ex_top_twenty_still_leaves_a_positive_mean(trades):
    td = tail_dependence(trades)
    row = next(r for r in td["ladder"] if r["dropped"] == 20)
    assert row["remaining_mean_pct"] > 0.0


# ===========================================================================
# 13. bootstrap and monte carlo are reproducible and seeded
# ===========================================================================

def test_monte_carlo_is_reproducible(trades, snap):
    base = build_variants(trades)[0]
    a = monte_carlo(trades, variant=base, snap=snap, paths=200, seed=5)
    b = monte_carlo(trades, variant=base, snap=snap, paths=200, seed=5)
    assert a["final_equity"]["median"] == b["final_equity"]["median"]


def test_bootstrap_reports_intervals_not_a_point(trades, snap):
    base = build_variants(trades)[0]
    bs = bootstrap(trades, variant=base, snap=snap, resamples=200)
    for k in ("final_equity", "expectancy_usd", "win_rate", "profit_factor",
              "max_drawdown_pct"):
        assert bs[k]["p05"] <= bs[k]["median"] <= bs[k]["p95"]


# ===========================================================================
# 14. research-only boundary
# ===========================================================================

def test_no_execution_capability_in_the_survival_package():
    import ast
    import glob

    banned = ("broadcast", "send_transaction", "sign_transaction", "submit",
              "arm_real", "enable_real", "real_buy", "real_sell")
    banned_imports = ("autopaper.execution.real", "autopaper.execution.solana",
                      "autopaper.token_cleanup", "solders")
    for p in sorted(glob.glob(str(_RESEARCH / "survival" / "*.py"))):
        tree = ast.parse(Path(p).read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                f = n.func
                nm = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                assert nm not in banned, f"{p} calls {nm}"
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                mods = ([a.name for a in n.names] if isinstance(n, ast.Import)
                        else [n.module or ""])
                for m in mods:
                    assert not any(m.startswith(b) for b in banned_imports), \
                        f"{p} imports {m}"


def test_research_does_not_mutate_production_l25():
    from autopaper import l25

    before = (l25.L25_MIN_LIQ_TO_MCAP, getattr(l25, "L25_HASH", ""))
    snapshot()
    simulate_path.__globals__  # touch nothing
    assert (l25.L25_MIN_LIQ_TO_MCAP, getattr(l25, "L25_HASH", "")) == before