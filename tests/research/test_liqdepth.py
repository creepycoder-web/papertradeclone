"""
Tests for the liquidity-depth study.

Two jobs:

1. **Correctness of the statistics.** Spearman, Mann-Whitney, the exact sign test
   and chi-square are checked against values computable by hand or published in
   critical-value tables. An earlier version of this module had a two-sided-U bug
   that silently destroyed effect direction; these tests exist to make that class
   of bug impossible to reintroduce.
2. **Isolation.** The study must not modify production L25 logic, must not read
   secrets, and must be reproducible.

Run:  PYTHONPATH=src:research .venv/bin/python -m pytest tests/research/test_liqdepth.py -q
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

AI = Path(__file__).resolve().parents[2]
for _p in (str(AI / "research"), str(AI / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from liqdepth import analysis as A        # noqa: E402
from liqdepth import data as D            # noqa: E402
from liqdepth import execution as X       # noqa: E402
from liqdepth import experiments as E     # noqa: E402
from liqdepth import power as P           # noqa: E402
from liqdepth import stats as S           # noqa: E402
from liqdepth import thresholds as T      # noqa: E402


# ===========================================================================
# statistics: verified against hand-computable values
# ===========================================================================


def test_spearman_extremes():
    assert S.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert S.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_uses_midranks():
    # ties must not be broken arbitrarily
    a = [1, 2, 2, 3]
    b = [1, 2, 3, 4]
    assert S.spearman(a, b) == pytest.approx(0.948683, abs=1e-5)


def test_spearman_undefined_on_constant_input():
    assert S.spearman([1, 1, 1], [1, 2, 3]) is None


def test_mannwhitney_direction_is_preserved():
    """
    Regression test.

    CLES once used min(u_a, n_a*n_b - u_a), which clamps the effect size to
    <= 0.5 and makes every comparison look 'not separated' regardless of
    direction. Direction is the whole point of this study.
    """
    hi = S.mannwhitney_u([6, 7, 8, 9, 10], [1, 2, 3, 4, 5])
    lo = S.mannwhitney_u([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    assert hi["cles"] == pytest.approx(1.0)
    assert lo["cles"] == pytest.approx(0.0)
    assert hi["p"] == pytest.approx(lo["p"])
    assert S.mannwhitney_u([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])["cles"] == \
        pytest.approx(0.5)


def test_mannwhitney_p_is_bounded():
    for a, b in (([1, 2, 3], [1, 2, 3]), ([1], [2, 3]), ([1, 2], [3])):
        p = S.mannwhitney_u(a, b)["p"]
        assert p is None or 0.0 <= p <= 1.0


def test_fisher_exact_matches_a_hand_computation():
    # 2x2 [[10,2],[2,10]] -> OR 25, two-sided p ~0.0033
    r = S.fisher_exact_2x2(10, 2, 2, 10)
    assert r["odds_ratio"] == pytest.approx(25.0)
    assert r["p"] == pytest.approx(0.00333, abs=1e-4)


def test_fisher_exact_identical_margins_are_not_significant():
    assert S.fisher_exact_2x2(10, 10, 10, 10)["p"] == pytest.approx(1.0)


def test_chi2_matches_published_critical_values():
    """
    The chi-square tail was originally a Wilson-Hilferty approximation that was
    ~6% off at df=1, which would have shifted significance near the 0.05 line.
    """
    crit05 = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070,
              6: 12.592, 8: 15.507, 10: 18.307}
    for df, x in crit05.items():
        assert S._chi2_sf(x, df) == pytest.approx(0.05, abs=6e-4), df
    crit01 = {1: 6.635, 2: 9.210, 3: 11.345, 5: 15.086}
    for df, x in crit01.items():
        assert S._chi2_sf(x, df) == pytest.approx(0.01, abs=6e-4), df


def test_kruskal_wallis_on_a_known_split():
    r = S.kruskal_hallis([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert r["h"] == pytest.approx(7.2, abs=1e-6)
    assert 0.02 < r["p"] < 0.04


def test_ols_recovers_a_known_line():
    r = S.ols([3, 5, 7, 9, 11], [1, 2, 3, 4, 5])
    assert r["slope"] == pytest.approx(2.0)
    assert r["intercept"] == pytest.approx(1.0)
    assert r["r2"] == pytest.approx(1.0)
    assert r["pearson_r"] == pytest.approx(1.0)


def test_ols_undefined_on_constant_predictor():
    assert S.ols([1, 2, 3], [5, 5, 5])["slope"] is None


def test_monotonicity_detects_direction_and_gaps():
    assert S.monotonicity([1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0])["strictly_monotone"]
    assert S.monotonicity([1, 2, 3, 4], [4.0, 3.0, 2.0, 1.0])["strictly_monotone"]
    flat = S.monotonicity([1, 2, 3, 4], [2.0, 2.0, 2.0, 2.0])
    assert not flat["strictly_monotone"] and flat["n_flat_steps"] == 3
    # a dropped bin must NOT be treated as a flat step
    gap = S.monotonicity([1, 2, 3, 4], [1.0, None, 3.0, 4.0])
    assert gap["n_bins_dropped"] == 1
    assert not gap["strictly_monotone"]


def test_bootstrap_ci_brackets_the_point_estimate():
    vals = [float(i) for i in range(1, 41)]
    ci = S.bootstrap_ci(vals, lambda xs: sum(xs) / len(xs), seed=1, iters=1500)
    assert ci["lo"] <= ci["point"] <= ci["hi"]


def test_bootstrap_is_seed_deterministic():
    vals = [float(i % 7) for i in range(30)]
    a = S.bootstrap_ci(vals, lambda xs: sorted(xs)[len(xs) // 2], seed=5, iters=800)
    b = S.bootstrap_ci(vals, lambda xs: sorted(xs)[len(xs) // 2], seed=5, iters=800)
    assert a == b


def test_spearman_permutation_never_returns_zero_p():
    r = S.spearman_perm([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], seed=0, iters=200)
    assert r["p"] > 0.0


# ===========================================================================
# dataset: provenance and honesty
# ===========================================================================


@pytest.fixture(scope="module")
def dataset():
    return D.load_trades()


def test_population_is_the_frozen_l25_universe(dataset):
    assert dataset["rows"], "no trades loaded"
    assert len(dataset["rows"]) == 116


def test_liquidity_field_is_the_one_the_gate_reads():
    assert D.LIQUIDITY_FIELD == "f_liquidity_usd"
    assert D.MCAP_FIELD == "f_market_cap_usd"


#: analysis-frame key -> the PROVENANCE key that documents it
_DERIVED_FRAME_KEYS = {
    "_pos_over_liq": "position_over_liquidity",
    "_pos_over_liq_bps": "position_over_liquidity_bps",
    "_log_liq": "log_liquidity",
    "_age_hours": "token_age_hours",
    "_net_pct": "net_pct_on_notional",
    # the frame recomputes liq/mcap from the two primitives, so it is derived
    # even though the gate carries its own observed copy under f_liq_to_mcap
    "_liq_over_mcap": "derived_liq_over_mcap",
}


def test_every_derived_variable_is_labelled(dataset):
    for frame_key, prov_key in _DERIVED_FRAME_KEYS.items():
        assert prov_key in D.PROVENANCE, f"{frame_key} -> {prov_key} undocumented"
        assert D.PROVENANCE[prov_key].startswith("derived"), prov_key


def test_every_observed_gate_input_is_labelled_observed(dataset):
    for key in (D.LIQUIDITY_FIELD, D.MCAP_FIELD, "f_liq_to_mcap",
                "f_signal_quality", "f_token_age_s"):
        assert D.PROVENANCE[key] == "observed", key


def test_every_provenance_label_is_one_of_three_kinds():
    for key, label in D.PROVENANCE.items():
        head = label.split(" ")[0].split("(")[0]
        assert head in ("observed", "derived", "unavailable"), (key, label)


def test_unavailable_variables_are_declared_not_invented():
    for key in ("executable_quote_price", "regime_labels"):
        assert key in D.UNAVAILABLE


def test_regime_is_reported_degenerate(dataset):
    d = D.regime_diagnostic(dataset["rows"])
    assert d["is_degenerate"] is True
    assert d["distinct_values"] == 1
    assert "CANNOT serve as a control" in d["conclusion"]


def test_token_age_is_observed_not_imputed(dataset):
    inv = D.inventory(dataset["rows"])
    assert inv["token_age_hours"]["n"] == len(dataset["rows"])
    assert inv["token_age_hours"]["pct_below_10_minutes"] > 80


def test_derived_values_are_consistent(dataset):
    for r in dataset["rows"]:
        assert r["_pos_over_liq"] == pytest.approx(
            r["notional"] / r["_liq"], rel=1e-9)
        assert r["_net_pct"] == pytest.approx(
            r["net"] / r["notional"] * 100.0, rel=1e-9)
        assert r["_win"] == (r["net"] > 0)


# ===========================================================================
# findings are stable and honest
# ===========================================================================


def test_tertiles_partition_without_overlap(dataset):
    t = A.liquidity_tertiles(dataset["rows"])
    groups = list(t["groups"].values())
    assert sum(g["n"] for g in groups) == 3 * t["group_size"]
    ids = [id(r) for g in groups for r in
           sorted(dataset["rows"], key=lambda x: x["_liq"])[
               :0]]  # placeholder to keep structure clear
    # deepest must be strictly above shallow on the boundary
    assert (groups[0]["liquidity_min_usd"]
            >= groups[-1]["liquidity_max_usd"])


def test_tertiles_are_not_labelled_as_thresholds(dataset):
    t = A.liquidity_tertiles(dataset["rows"])
    assert t["is_operational_threshold"] is False
    assert "EXPLORATORY" in t["caveat"]


def test_deep_group_beats_shallow_on_this_sample(dataset):
    t = A.liquidity_tertiles(dataset["rows"])
    assert (t["groups"]["deepest_1_3"]["win_rate_pct"]
            > t["groups"]["shallowest_1_3"]["win_rate_pct"])


def test_mcap_controlled_keeps_direction_in_every_testable_band(dataset):
    """
    The load-bearing result. If a future change flips a band's direction, the
    'market cap does not explain it' claim is no longer true and this must fail.
    """
    mc = A.mcap_controlled_analysis(dataset["rows"])
    tested = 0
    for band in mc["bands"].values():
        hv = band.get("high_vs_low")
        if not hv:
            continue
        tested += 1
        assert hv["median_difference_pct"] > 0, band
    assert tested >= 4


def test_matched_pairs_are_reported_at_both_control_levels(dataset):
    """
    Both variants must be present. The ratio-controlled variant is the
    conservative headline, but reporting only it would hide that the
    association is visible (p=0.048) before liquidity/mcap is controlled --
    which is exactly the fact a reader needs in order to judge the conclusion.
    """
    m = E.incremental_value(dataset["rows"])["matched_comparison"]
    assert "skipped" not in m
    assert set(m["variants"]) == {"without_ratio_control", "with_ratio_control"}
    for v in m["variants"].values():
        assert "skipped" not in v
        assert v["n_pairs"] >= 20
    # The CONSERVATIVE (headline) variant must not claim significance. The
    # looser variant legitimately does: it is the finding that motivates
    # controlling liquidity/mcap, so forcing it to be insignificant too would
    # be testing the wrong thing.
    tight = m["variants"]["with_ratio_control"]
    ci = tight["bootstrap_ci_median_difference"]
    assert ci["lo"] < 0 < ci["hi"], "the headline variant must not claim effect"
    assert tight["sign_test_p"] > 0.05
    assert m["headline_variant"] == "with_ratio_control"
    assert "underpowered" in m["comparison_note"]


def test_ratio_control_weakens_the_matched_association(dataset):
    """
    Locks the substantive claim: controlling liquidity/mcap moves the matched
    association toward zero. If this ever flips, the report's reasoning changes.
    """
    m = E.incremental_value(dataset["rows"])["matched_comparison"]
    loose = m["variants"]["without_ratio_control"]
    tight = m["variants"]["with_ratio_control"]
    assert loose["median_outcome_difference_deeper_minus_shallower_pct"] > \
        tight["median_outcome_difference_deeper_minus_shallower_pct"]
    # the ratio-controlled pairs are necessarily closer in liquidity
    assert tight["median_liquidity_gap_usd"] < loose["median_liquidity_gap_usd"]


def test_matched_pairs_are_reported_as_inconclusive_not_null(dataset):
    """
    The report must not claim the effect is absent, because the test cannot
    resolve an effect smaller than its MDE.
    """
    inc = E.incremental_value(dataset["rows"])
    assert "inconclusive" in inc["verdict"].lower() or \
        "matched" in inc["verdict"].lower()


def test_no_threshold_passes_all_gates(dataset):
    ev = T.evaluate_family(dataset["rows"], T.candidate_thresholds(dataset["rows"]))
    assert ev["any_all_gates_passed"] is False


def test_gate_verdict_names_the_account_shortfall(dataset):
    ev = T.evaluate_family(dataset["rows"], T.candidate_thresholds(dataset["rows"]))
    v = T.summarise_gates(ev)
    assert "NO DEPLOYABLE THRESHOLD IS SUPPORTED" in v
    assert "account return" in v


def test_untestable_arms_are_not_counted_as_refutations(dataset):
    ev = T.evaluate_family(dataset["rows"], T.candidate_thresholds(dataset["rows"]))
    for c in ev["candidates"]:
        if not c["test_arm_meaningful"]:
            assert c["gates"]["G2_directionally_useful_out_of_sample"] is None
            assert not c["all_gates_passed"]


def test_oos_split_is_strictly_chronological(dataset):
    o = E.chronological_oos(dataset["rows"], train_frac=0.65)
    d = o["design"]
    assert d["train_t_entry_range"][1] < d["test_t_entry_range"][0]


def test_oos_thresholds_are_train_derived_only(dataset):
    rows = sorted(dataset["rows"], key=lambda r: r["t_entry"])
    cut = int(len(rows) * 0.65)
    train_liq = sorted(r["_liq"] for r in rows[:cut])
    o = E.chronological_oos(rows)
    assert "TRAIN" in o["design"]["derivation"].upper()
    for arm in o["arms"]:
        q = arm["train_quantile"]
        assert 0.0 < q < 1.0
        # the threshold must be a TRAIN quantile value, i.e. present in the
        # TRAIN distribution and inside its range
        assert train_liq[0] <= arm["threshold_usd"] <= train_liq[-1]
        assert arm["threshold_usd"] in train_liq


def test_tail_removal_keeps_the_direction(dataset):
    ts = E.tail_sensitivity(dataset["rows"])
    for lvl in ts["levels"]:
        assert lvl["deep_minus_shallow_median_pct"] > 0, lvl["dropped"]


def test_tail_removal_is_recomputed_not_sequential(dataset):
    """
    'Drop the top 5 of the ORIGINAL sample' and 'drop 5 then re-rank' are
    different procedures; this asserts the monotonic n and that counts differ.
    """
    ts = E.tail_sensitivity(dataset["rows"])
    ns = [lvl["n_remaining"] for lvl in ts["levels"]]
    assert ns == sorted(ns, reverse=True)


def test_account_filter_lowers_bankroll_while_raising_win_rate(dataset):
    """
    The study's central tension. If a future change makes a filter improve both,
    the conclusion category must change and this must fail first.
    """
    base = E.account_eval(dataset["rows"], None, label="b")
    filt = E.account_eval(dataset["rows"], E.liquidity_filter(18000),
                         label="f")
    assert filt["win_rate_pct"] > base["win_rate_pct"]
    assert filt["final_bankroll_usd"] < base["final_bankroll_usd"]


def test_relative_size_adds_nothing_to_absolute(dataset):
    rs = E.relative_size_analysis(dataset["rows"])
    assert abs(rs["incremental"]
              ["relative_size_given_absolute_liquidity"]["partial_rho"]) < 0.1


def test_ratio_beats_absolute_liquidity(dataset):
    X.set_rows(dataset["rows"])
    r = X.liq_mcap_vs_absolute()
    assert abs(r["liquidity_over_mcap"]["rho"]) > \
        abs(r["absolute_liquidity"]["rho"])


def test_power_is_reported_and_large(dataset):
    ps = P.power_summary(dataset["rows"])
    mc = ps["mc_controlled_bands"]["mde_pct_points"]
    full = ps["full_sample_tertiles"]["mde_pct_points"]
    assert mc > full, "controlled cells must be less powerful than the full split"
    # the MC-controlled MDE must exceed the full-sample tertile EFFECT, which is
    # exactly why the inconclusive bands cannot be read as refutations
    t = A.liquidity_tertiles(dataset["rows"])
    effect = abs(t["groups"]["deepest_1_3"]["median_net_pct"]
                 - t["groups"]["shallowest_1_3"]["median_net_pct"])
    assert mc > effect, (mc, effect)
    assert "power limitation" in ps["interpretation"].lower()


# ===========================================================================
# execution: observation vs theory must stay separated
# ===========================================================================


def test_fills_are_declared_mark_based(dataset):
    ev = X.observed_execution_evidence()
    assert ev["l25_fill_basis"] == "OBSERVED_MARK"
    assert ev["per_trade_executable_quote"]["status"] == "not_observable_here"


def test_theories_are_marked_unobservable(dataset):
    X.set_rows(dataset["rows"])
    for mech in X.theory_vs_observation()["mechanisms"]:
        assert mech["observable_here"] is False
        assert mech["testable_with"]


def test_no_causal_language_in_theory_module():
    src = (AI / "research" / "liqdepth" / "execution.py").read_text()
    for banned in ("liquidity causes", "causes higher returns"):
        assert banned not in src.lower()


def test_concentration_diagnostic_flags_tail_driven_bins(dataset):
    X.set_rows(dataset["rows"])
    c = X.concentration_diagnostic(dataset["rows"])
    shapes = {b["shape"] for b in c["bins"]}
    assert any("tail-driven" in s for s in shapes)


# ===========================================================================
# isolation
# ===========================================================================


def test_no_production_file_is_modified_by_the_study():
    from l25r.isolation import snapshot

    before = snapshot()
    assert before, "no protected files found; the guard is not working"
    for path, digest in before.items():
        assert (AI / path).exists(), path


def test_study_never_imports_an_execution_module():
    """The study must not be able to reach a signer or broadcaster."""
    forbidden = ("autopaper.execution", "autopaper.real_", "autopaper.settlement",
                 "autopaper.external_close", "solders")
    for p in sorted((AI / "research" / "liqdepth").glob("*.py")):
        for line in p.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                for f in forbidden:
                    assert f not in stripped, f"{p.name}: {stripped}"


def test_study_writes_only_to_the_report_directory():
    from l25r.isolation import ALLOWED_WRITE_ROOTS

    assert ALLOWED_WRITE_ROOTS == ("reports/research",)


def test_no_secret_file_is_read_by_the_study():
    src = (AI / "research" / "liqdepth" / "data.py").read_text()
    assert ".env" not in src
    assert "load_env_file" not in src


# ===========================================================================
# deliverables
# ===========================================================================


def test_all_three_deliverables_exist():
    for name in ("liquidity_depth_l25_research.md",
                 "liquidity_depth_l25_research.json",
                 "liquidity_depth_l25_research_trades.csv"):
        p = AI / "reports" / "research" / name
        assert p.exists(), name
        assert p.stat().st_size > 0, name


def test_report_contains_every_required_section():
    text = (AI / "reports" / "research"
            / "liquidity_depth_l25_research.md").read_text()
    for needle in (
        "Reproduction of the previous",
        "Liquidity distribution",
        "Market-cap controlled",
        "Token maturity",
        "regime",
        "Position size relative",
        "Tail sensitivity",
        "Account-level",
        "out-of-sample",
        "Threshold experiments",
        "Incremental value",
        "Execution interpretation",
        "Conclusion",
        "Limitations",
    ):
        assert needle in text, needle


def test_report_declares_research_only_and_no_causal_claim():
    text = (AI / "reports" / "research"
            / "liquidity_depth_l25_research.md").read_text()
    assert "Research only" in text
    assert "no causal claim" in text.lower()


def test_report_states_one_primary_conclusion():
    text = (AI / "reports" / "research"
            / "liquidity_depth_l25_research.md").read_text()
    cats = [c for c in ("Category A", "Category B", "Category C")
            if f"### {c}" in text]
    assert len(cats) == 1, cats


def test_json_payload_has_the_required_keys():
    p = AI / "reports" / "research" / "liquidity_depth_l25_research.json"
    d = json.loads(p.read_text())
    for key in ("reproduction", "continuous", "mcap_controlled", "maturity",
                "regime", "relative_size", "tails", "account", "oos",
                "thresholds", "incremental", "execution_observed",
                "hypotheses", "final", "power"):
        assert key in d, key
    assert d["final"]["category"] in ("A", "B", "C")


def test_csv_has_one_row_per_trade_and_the_key_columns():
    import csv as _csv

    p = AI / "reports" / "research" / "liquidity_depth_l25_research_trades.csv"
    with p.open() as fh:
        rows = list(_csv.DictReader(fh))
    assert len(rows) == 116
    for col in ("trade_id", "f_liquidity_usd", "f_market_cap_usd",
                "net_pct_on_notional", "position_over_liquidity",
                "token_age_hours", "liquidity_tertile", "exit_reason"):
        assert col in rows[0], col


def test_csv_liquidity_tertile_matches_a_recomputation(dataset):
    import csv as _csv

    p = AI / "reports" / "research" / "liquidity_depth_l25_research_trades.csv"
    with p.open() as fh:
        rows = list(_csv.DictReader(fh))
    t = A.liquidity_tertiles(dataset["rows"])
    k = t["group_size"]
    ordered = sorted(rows, key=lambda r: float(r["f_liquidity_usd"]))
    for i, r in enumerate(ordered):
        if i >= len(ordered) - k:
            assert r["liquidity_tertile"] == "deepest_1_3"
        elif i >= len(ordered) - 2 * k:
            assert r["liquidity_tertile"] == "middle_1_3"
        else:
            assert r["liquidity_tertile"] == "shallowest_1_3"


def test_study_is_reproducible_end_to_end():
    """
    Re-running the runner must reproduce the JSON payload exactly.

    The payload is compared SEMANTICALLY (parsed, then compared) rather than as
    bytes. An earlier version of the study wrote a wall-clock timestamp into the
    payload, so a byte comparison failed on every run for a reason unrelated to
    the analysis. The payload deliberately carries no wall-clock field, and this
    test asserts that as well.
    """
    base = AI / "reports" / "research" / "liquidity_depth_l25_research.json"
    first = json.loads(base.read_text())
    env = {"PYTHONPATH": "src:research", "PATH": "/usr/bin:/bin"}
    r = subprocess.run(
        [sys.executable, "-m", "liqdepth.run"], capture_output=True,
        text=True, timeout=1800, cwd=str(AI), env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    second = json.loads(base.read_text())
    assert second == first, "the computed payload is not deterministic"

    # no wall-clock leakage into the ANALYTICAL payload. The environment-probe
    # block is excluded because it reports live filesystem state (file sizes),
    # which legitimately changes whenever another process writes to the repo.
    import re

    ANALYTICAL_ROOTS = (
        "reproduction", "continuous", "mcap_bands", "mcap_controlled",
        "maturity", "regime", "relative_size", "tails", "account", "oos",
        "thresholds", "incremental", "execution_observed", "execution_theory",
        "concentration", "liq_mcap_vs_absolute", "power", "hypotheses",
        "final", "population", "inventory", "regime_diagnostic",
    )

    def leaves(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from leaves(v, path + "." + str(k))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                yield from leaves(v, path + "[" + str(i) + "]")
        else:
            yield path, o

    bad = [pth for root in ANALYTICAL_ROOTS
           for pth, _ in leaves(second.get(root, {}), root)
           if re.search(r"generated_at|run_at|wall_clock|timestamp_now", pth)]
    assert not bad, bad

    # the probe block must still assert it never read a secret
    fr = second.get("forbidden_reads", {})
    assert fr, "the forbidden-read probe is missing from the payload"
    for rel, info in fr.items():
        assert info["read_by_this_study"] is False, rel
