"""Regression tests for the Telegram-alert target-hit study.

Pins the two things a later edit could silently break and thereby flip the
headline: the price-basis transform (percent vs fraction is easy to get wrong —
a real bug caught during this study), and the target/stop ordering bounds. Plus
the audit arithmetic against the existing trade CSVs.

Offline; skips when the research source data is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "research" / "alert_target_hit_study" / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import alert_common as C  # noqa: E402
import analyze as A  # noqa: E402


# ---------------------------------------------------------------------------
# price-basis transforms
# ---------------------------------------------------------------------------
def test_paper_basis_is_exit_multiplier_only():
    """The grid already carries the 1.015 entry markup, so paper = (1+r)*0.9852 - 1."""
    r = np.array([0.0, 0.30, -0.20])
    out = C.to_basis(r, "paper")
    assert np.allclose(out, (1 + r) * 0.9852 - 1)
    # a +30% grid move does NOT clear a net +30%
    assert out[1] < 0.30


def test_paper_break_even_threshold():
    """Grid return needed to net +30% under the paper base is ~31.95%."""
    thr = C.level_for_return(0.30, "paper")
    assert thr == pytest.approx(1.30 / 0.9852 - 1, abs=1e-9)
    assert 0.3194 < thr < 0.3196


def test_cost_basis_applies_round_trip_haircut():
    r = np.array([0.50, 0.0])
    cp = np.array([10.0, 10.0])          # 10% round-trip cost
    out = C.to_basis(r, "cost", cp)
    assert out[0] == pytest.approx(1.50 * 0.90 - 1)
    assert out[1] == pytest.approx(-0.10)


def test_mark_basis_is_identity():
    r = np.array([0.3, -0.2])
    assert np.allclose(C.to_basis(r, "mark"), r)


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------
def test_assign_split_boundaries():
    assert C.assign_split("2026-09-25") == "train"
    assert C.assign_split("2026-09-27") == "valid"
    assert C.assign_split("2026-09-29") == "test"
    assert C.assign_split("2026-09-20") == "outside"


def test_clustered_bootstrap_widens_under_within_token_correlation():
    """The whole point of clustering: repeated alerts on one token are not
    independent, so the i.i.d. bootstrap understates the interval. Build 40
tokens x 5 near-identical alerts each and require the clustered CI to widen."""
    rng = np.random.default_rng(0)
    cluster_vals = rng.normal(0.3, 0.5, 40)
    r = np.repeat(cluster_vals, 5) + rng.normal(0, 1e-6, 200)
    clusters = np.repeat(np.arange(40), 5)
    iid = C.bootstrap_ci(r)
    clu = C.clustered_bootstrap_ci(r, clusters)
    assert (clu[1] - clu[0]) > 1.5 * (iid[1] - iid[0])
    assert (clu[1] - clu[0]) / (iid[1] - iid[0]) == pytest.approx(2.24, abs=0.4)


# ---------------------------------------------------------------------------
# outcome statistics on a synthetic panel
# ---------------------------------------------------------------------------
def _panel(rows, H=60):
    cols = {f"e0_{k}_{H}": [] for k in
            ("mfe", "mae", "ret", "up30", "ttu30", "up50", "ttu50", "dn20", "ttd20")}
    for mfe, mae, ret, up30, ttu30, up50, ttu50, dn20, ttd20 in rows:
        for k, v in zip(("mfe", "mae", "ret", "up30", "ttu30", "up50", "ttu50",
                         "dn20", "ttd20"),
                        (mfe, mae, ret, up30, ttu30, up50, ttu50, dn20, ttd20)):
            cols[f"e0_{k}_{H}"].append(v)
    df = pd.DataFrame(cols)
    df["d_cost_pct"] = 2.0
    return df


def test_outcome_stats_target_before_stop_is_exact_for_plus30_mark():
    # A: reaches +30 at 10s, never -20  -> target first
    # B: hits -20 at 5s, then +30 at 40s -> stop first
    # C: neither                          -> neither
    p = _panel([
        (35.0, -5.0, 10.0, 1, 10.0, 0, np.nan, 0, np.nan),
        (31.0, -25.0, -5.0, 1, 40.0, 0, np.nan, 1, 5.0),
        (5.0, -5.0, 2.0, 0, np.nan, 0, np.nan, 0, np.nan),
    ])
    st = A.outcome_stats(p, 60, 0.30, "mark")
    assert st["n_with_forward_data"] == 3
    assert st["pct_reach_any_time"] == pytest.approx(66.667, abs=0.01)
    assert st["pct_target_before_stop_lower"] == pytest.approx(33.333, abs=0.01)
    assert st["pct_target_before_stop_upper"] == pytest.approx(33.333, abs=0.01)
    assert st["pct_stop_any_time"] == pytest.approx(33.333, abs=0.01)


def test_outcome_stats_paper_lowers_reach():
    # every token reaches exactly +31% (grid): clears mark +30 but NOT net +30
    rows = [(31.0, -5.0, 31.0, 1, 20.0, 0, np.nan, 0, np.nan)] * 4
    p = _panel(rows)
    mark = A.outcome_stats(p, 60, 0.30, "mark")
    paper = A.outcome_stats(p, 60, 0.30, "paper")
    assert mark["pct_reach_any_time"] == pytest.approx(100.0)
    assert paper["pct_reach_any_time"] == pytest.approx(0.0)


def test_outcome_stats_ignores_missing_forward_tape():
    p = _panel([
        (35.0, -5.0, 10.0, 1, 10.0, 0, np.nan, 0, np.nan),
        (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
    ])
    st = A.outcome_stats(p, 60, 0.30, "mark")
    assert st["n_no_forward_tape"] == 1
    assert st["n_with_forward_data"] == 1
    assert st["pct_reach_any_time"] == pytest.approx(100.0)   # over data, not n


# ---------------------------------------------------------------------------
# audit against the shipped trade CSVs
# ---------------------------------------------------------------------------
def test_audit_reproduces_2065_and_the_corrected_number():
    naive = C.S11_DIR / "s11_trades_paperfill_NAIVE_base_MODELED.csv"
    if not naive.exists():
        pytest.skip("v2_floor_validation outputs not present")
    a = A.audit_2065()
    assert a["confirms_2065"] is True
    assert a["naive_base"]["n_trades"] == 189
    assert a["naive_base"]["unique_tokens"] == 188
    # the corrected base is materially lower — the point of the audit
    assert a["corrected_base"]["mean_return"] == pytest.approx(0.0644, abs=0.002)
    assert a["naive_base"]["mean_return"] > 3 * a["corrected_base"]["mean_return"]


def test_panel_builds_from_master():
    if not C.MASTER.exists():
        pytest.skip("master.jsonl not present")
    panel = A.load_panel()
    assert len(panel) == 948
    assert panel.ca.nunique() == 937
    assert panel.usable.sum() == 723
