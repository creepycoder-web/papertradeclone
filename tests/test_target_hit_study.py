"""Regression tests for the Solana memecoin target-hit study.

Each test pins a guarantee that a later edit could silently break. The execution
math and the scenario classifier are the two places where a wrong sign or a
wrong ordering rule would flip the study's headline, so they are tested directly
on synthetic inputs rather than only through the (slow, data-heavy) pipeline.

Offline and fast: the corpus-dependent test skips when the data is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The runtime .venv is deliberately dependency-free; this study needs pandas.
# Skip (never fail) when those research deps are absent.
pytest.importorskip("pandas")
pytest.importorskip("numpy")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

AI = Path(__file__).resolve().parents[1]
CODE = AI / "research" / "target_hit_study" / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import common as C  # noqa: E402
import analyze as A  # noqa: E402

LAMP = C.LAMPORTS_PER_SOL
# a realistic pump.fun launch curve: 30 SOL virtual, 1.073e9 tokens (raw *1e6)
VSOL0 = 30.0 * LAMP
VTOK0 = 1.073e9 * 1e6


# ---------------------------------------------------------------------------
# execution math
# ---------------------------------------------------------------------------
def test_price_sol_convention_matches_corpus_units():
    """price_sol == (v_sol / v_tok) / 1000 (KNOWN_ISSUES §6)."""
    px = (VSOL0 / VTOK0) / 1000.0
    assert px == pytest.approx(2.796e-8, rel=0.02)


def test_buy_then_sell_same_reserves_loses_the_fee_and_impact():
    n = 0.0667 * LAMP  # ~$10 at SOL=$150
    tok = C.buy_tokens_raw(np.array([VSOL0]), np.array([VTOK0]), np.array([n]))[0]
    out = C.sell_sol_lamports(np.array([VSOL0]), np.array([VTOK0]), np.array([tok]))[0]
    assert out < n                       # round trip always loses
    loss = 1 - out / n
    assert 0.019 < loss < 0.05           # ~2% fee + small impact on a deep pool


def test_impact_grows_on_thin_pools():
    n = 0.0667 * LAMP
    deep = C.exec_haircut_bps(30 * LAMP, 1.073e9 * 1e6, n)
    thin = C.exec_haircut_bps(3 * LAMP, 1.073e8 * 1e6, n)
    assert thin > deep > 0


def test_round_trip_return_positive_when_price_rises():
    n = 0.0667 * LAMP
    tok = C.buy_tokens_raw(np.array([VSOL0]), np.array([VTOK0]), np.array([n]))[0]
    # price doubles: same token count, half the v_tok per v_sol -> double v_sol/v_tok
    vsol1, vtok1 = VSOL0 * 2, VTOK0
    r = C.round_trip_return(VSOL0, VTOK0, n, np.array([vsol1]), np.array([vtok1]))
    assert r[0] > 0.8


def test_notional_lamports():
    assert C.notional_lamports(10.0, 150.0) == pytest.approx(10 / 150 * LAMP)


def test_regime_of_date_boundaries():
    assert C.regime_of_date("2026-06-05") == 0
    assert C.regime_of_date("2026-06-09") == 1
    assert C.regime_of_date("2026-07-14") == len(C.REGIME_BREAKS)


# ---------------------------------------------------------------------------
# scenario classifier
# ---------------------------------------------------------------------------
def _fw(rows):
    return pd.DataFrame(rows, columns=["mint", "dt_s", "ex_ret", "mid_ret"])


def _ev(mints_nsell):
    return pd.DataFrame({
        "mint": [m for m, _ in mints_nsell],
        "n_sell": [s for _, s in mints_nsell],
        "fullwin0": [True] * len(mints_nsell),
        "day": ["2026-06-10"] * len(mints_nsell),
    })


def test_classifier_target_stop_timeout_and_unsellable():
    fw = _fw([
        # A: rises to +30 first -> target
        ("A", 1.0, 0.05, 0.05), ("A", 5.0, 0.30, 0.30), ("A", 40.0, 0.10, 0.10),
        # B: drops past -20 first, then recovers to +30 -> stop
        ("B", 3.0, -0.21, -0.21), ("B", 10.0, 0.30, 0.30),
        # C: never reaches either -> timeout, exit at last print
        ("C", 10.0, -0.05, -0.05), ("C", 80.0, 0.02, 0.02),
        # D: no forward rows at all handled by n_sell==0 -> unsellable
        ("D", 5.0, 0.5, 0.5),
    ])
    ev = _ev([("A", 5), ("B", 5), ("C", 5), ("D", 0)])
    t = A.scenario_table(ev, fw, "ex")
    row = t.iloc[0]
    assert row["n_entries"] == 4
    assert row["pct_target"] == pytest.approx(25.0)
    assert row["pct_stop"] == pytest.approx(25.0)
    assert row["pct_timeout"] == pytest.approx(25.0)
    assert row["pct_failed_unsellable"] == pytest.approx(25.0)
    # expectancy: 0.30 + (-0.21) + 0.02 + (-1.0) = -0.89 over 4
    assert row["mean_net_return_expectancy"] == pytest.approx(-0.89 / 4, abs=1e-6)


def test_time_to_target_is_first_crossing():
    fw = _fw([("A", 2.0, 0.31, 0.31), ("A", 9.0, 0.55, 0.55)])
    ev = _ev([("A", 3)])
    t = A.scenario_table(ev, fw, "ex")
    assert t.iloc[0]["median_time_to_target_s"] == pytest.approx(2.0)


def test_stop_uses_actual_through_fill_not_the_level():
    fw = _fw([("A", 1.0, -0.55, -0.55), ("A", 2.0, -0.60, -0.60)])
    ev = _ev([("A", 3)])
    r = A.scenario_returns(ev, fw, "ex", 0.30)
    assert r[0] == pytest.approx(-0.55)   # not -0.20


def test_concentration_flags_single_winner_dominance():
    # one huge winner (+9.0) and nine small losers -> top1 ~100% of positive PnL
    fw = _fw([("W", 1.0, 9.0, 9.0)] + [(f"L{i}", 1.0, -0.2, -0.2) for i in range(9)])
    ev = _ev([("W", 3)] + [(f"L{i}", 3) for i in range(9)])
    c = A.concentration(ev, fw, tp=0.30, basis="ex")
    assert c["top1_share_pct"] > 99
    assert c["n_trades_for_50pct_pnl"] == 1


# ---------------------------------------------------------------------------
# corpus-dependent (skips offline)
# ---------------------------------------------------------------------------
def test_reserves_reconstruct_price_sol_on_real_corpus():
    corpus = AI / "research" / "target_hit_study" / "corpus" / "trades"
    shard = corpus / "trades-00000.parquet"
    if not shard.exists():
        pytest.skip("corpus not downloaded")
    import duckdb
    con = duckdb.connect()
    df = con.sql(f"""
        SELECT v_sol_bonding_curve AS vsol, v_tokens_bonding_curve AS vtok,
               price_sol AS px
        FROM read_parquet('{shard}')
        WHERE price_sol > 0 AND v_sol_bonding_curve > 0 AND v_tokens_bonding_curve > 0
        ORDER BY random() LIMIT 2000
    """).df()
    con.close()
    assert len(df) > 0
    recon = (df["vsol"] / df["vtok"]) / 1000.0
    assert np.allclose(recon, df["px"], rtol=1e-6, atol=1e-15)
