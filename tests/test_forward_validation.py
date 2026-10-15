"""
Forward-validation infrastructure.

The properties that matter here are negative ones. A validation harness can
always be made to print a good number; the work is making sure it CANNOT print a
good number that was not earned. So the emphasis is on refusals: boundaries that
cannot move, records that cannot be relabelled, verdicts that cannot be reached
with dirty or thin data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autopaper.forward_validation import (
    DEFAULT_POLICY,
    VALIDATING,
    DataQualityError,
    Dataset,
    Deployment,
    ForwardRecorder,
    NON_VALIDATING_STATUSES,
    Status,
    ValidationPolicy,
    build_report,
    check_quality,
    compute_metrics,
    expectancy_ci,
    load_or_create_deployment,
    read_jsonl,
    reconstruct_dataset,
)

T0 = 1_000_000.0
THRESHOLD = 0.9647


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "forward"


@pytest.fixture
def deployment(root: Path) -> Deployment:
    return load_or_create_deployment(
        root, strategy="L25", l25_min_liq_to_mcap=THRESHOLD,
        config_source=".env.local", strategy_hash="deadbeef",
        force_start_at=T0,
    )


@pytest.fixture
def recorder(root: Path, deployment: Deployment) -> ForwardRecorder:
    return ForwardRecorder(root, deployment)


def _feat(liq_to_mcap=2.0):
    return {
        "market_cap_usd": 10_000.0,
        "liquidity_usd": 10_000.0 * liq_to_mcap,
        "liq_to_mcap": liq_to_mcap,
        "signal_quality": 0.9,
    }


def _trade(rec: ForwardRecorder, i: int, ret: float, *, t=None, **extra):
    tid = f"t{i}"
    rec.trade("entry", {"trade_id": tid, "at": (t or T0) + i * 60,
                        "mint": f"M{i}", **extra})
    return rec.trade("exit", {
        "trade_id": tid, "at": (t or T0) + i * 60 + 30, "mint": f"M{i}",
        "return_pct": ret, "net_pnl_usd": ret / 100.0, "holding_s": 30.0,
        "mfe_pct": ret + 1.0, "mae_pct": -1.0,
    })


# ---------------------------------------------------------------------------
# the boundary cannot move
# ---------------------------------------------------------------------------


def test_the_boundary_is_created_once(root: Path):
    d1 = load_or_create_deployment(root, strategy="L25",
                                   l25_min_liq_to_mcap=THRESHOLD,
                                   force_start_at=T0)
    assert d1.started_at == T0


def test_the_boundary_cannot_be_moved_backwards(root: Path):
    """
    Moving a boundary backwards would relabel pre-deployment history as
    prospective. It is refused, not offered.
    """
    load_or_create_deployment(root, strategy="L25",
                              l25_min_liq_to_mcap=THRESHOLD,
                              force_start_at=T0)
    d2 = load_or_create_deployment(root, strategy="L25",
                                   l25_min_liq_to_mcap=THRESHOLD,
                                   force_start_at=T0 - 10_000)
    assert d2.started_at == T0


def test_the_boundary_cannot_be_moved_forwards_either(root: Path):
    load_or_create_deployment(root, strategy="L25",
                              l25_min_liq_to_mcap=THRESHOLD,
                              force_start_at=T0)
    d2 = load_or_create_deployment(root, strategy="L25",
                                   l25_min_liq_to_mcap=THRESHOLD,
                                   force_start_at=T0 + 10_000)
    assert d2.started_at == T0


def test_changing_the_threshold_does_not_reopen_the_boundary(root: Path):
    """A looser threshold must not restart the clock and inherit old data."""
    load_or_create_deployment(root, strategy="L25",
                              l25_min_liq_to_mcap=THRESHOLD,
                              force_start_at=T0)
    d2 = load_or_create_deployment(root, strategy="L25",
                                   l25_min_liq_to_mcap=0.1,
                                   force_start_at=T0 + 5_000)
    assert d2.l25_min_liq_to_mcap == THRESHOLD
    assert d2.deployment_id == _did(root)


def _did(root: Path) -> str:
    return json.loads((root / "deployment.json").read_text())["deployment_id"]


def test_the_deployment_id_encodes_the_frozen_inputs():
    a = load_or_create_deployment(Path("/tmp/_fv_a"), strategy="L25",
                                  l25_min_liq_to_mcap=0.9647,
                                  strategy_hash="x",
                                  force_start_at=T0)
    b = load_or_create_deployment(Path("/tmp/_fv_b"), strategy="L25",
                                  l25_min_liq_to_mcap=0.9647,
                                  strategy_hash="x",
                                  force_start_at=T0)
    c = load_or_create_deployment(Path("/tmp/_fv_c"), strategy="L25",
                                  l25_min_liq_to_mcap=0.9647,
                                  strategy_hash="DIFFERENT",
                                  force_start_at=T0)
    assert a.deployment_id == b.deployment_id
    assert a.deployment_id != c.deployment_id


# ---------------------------------------------------------------------------
# historical data cannot be promoted
# ---------------------------------------------------------------------------


def test_a_record_before_the_boundary_is_refused(recorder: ForwardRecorder):
    with pytest.raises(DataQualityError, match="precedes the deployment boundary"):
        recorder.candidate(candidate_id="old", mint="M", route="FAST_ENTRY",
                           features=_feat(), at=T0 - 1)


def test_a_record_at_the_boundary_is_accepted(recorder: ForwardRecorder):
    rec = recorder.candidate(candidate_id="edge", mint="M",
                             route="FAST_ENTRY", features=_feat(), at=T0)
    assert rec["dataset"] == Dataset.PROSPECTIVE_PAPER.value


def test_reconstruction_never_labels_itself_prospective():
    """The historical re-scoring path is hard-wired to the non-validating label."""
    rows = [{"candidate_id": "h1", "at": 1.0, "route": "FAST_ENTRY",
             "signal_quality": 0.9,
             "features": {"market_cap_usd": 1000, "liquidity_usd": 5000}}]
    out = reconstruct_dataset(rows, l25_min_liq_to_mcap=THRESHOLD)
    assert out[0]["dataset"] == Dataset.POST_DERIVATION_RECONSTRUCTED.value
    assert out[0]["dataset"] != Dataset.PROSPECTIVE_PAPER.value
    assert out[0]["dataset"] != VALIDATING.value


def test_a_reconstructed_row_cannot_be_written_as_prospective(
        root: Path, deployment: Deployment):
    """
    Even if a caller tried, the store is separate and the id must match.

    A reconstructed row has no deployment_id, so quality checking catches it.
    """
    rec = ForwardRecorder(root, deployment)
    rows = [{"candidate_id": "h1", "at": T0 + 10, "route": "FAST_ENTRY",
             "signal_quality": 0.9,
             "features": {"market_cap_usd": 1000, "liquidity_usd": 5000}}]
    out = reconstruct_dataset(rows, l25_min_liq_to_mcap=THRESHOLD)
    out[0]["deployment_id"] = "not-the-real-one"
    rep = build_report(out, deployment=deployment)
    assert rep.status is Status.DATA_QUALITY_PROBLEM


def test_reconstruction_refuses_a_non_frozen_threshold():
    """Scoring history under a different rule is refused, not done quietly."""
    rows = [{"candidate_id": "h", "at": 1.0, "route": "FAST_ENTRY",
             "signal_quality": 0.9,
             "features": {"market_cap_usd": 1, "liquidity_usd": 1}}]
    with pytest.raises(Exception):
        reconstruct_dataset(rows, l25_min_liq_to_mcap=0.10)


def test_a_row_with_no_features_fails_closed_rather_than_erroring():
    """
    A row missing market cap and liquidity is REFUSED by the gate, not skipped.

    Scoring it would be inventing data; erroring would drop the row. Failing
    closed and still recording the row keeps the refusal visible and auditable.
    """
    rows = [{"candidate_id": "bad", "at": 1.0, "route": "FAST_ENTRY",
             "signal_quality": 0.9, "features": {}}]
    out = reconstruct_dataset(rows, l25_min_liq_to_mcap=THRESHOLD)
    assert len(out) == 1, "the row must be recorded, not dropped"
    assert out[0]["l25_passes"] is False
    assert out[0]["l25_status"] == "UNTESTABLE"


def test_a_row_that_raises_is_reported_with_its_error(monkeypatch):
    """If evaluation raises, the row still appears, carrying the error."""
    import autopaper.l25 as l25

    def boom(**kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(l25, "evaluate", boom)
    rows = [{"candidate_id": "x", "at": 1.0, "route": "FAST_ENTRY",
             "signal_quality": 0.9, "features": {"market_cap_usd": 1}}]
    out = reconstruct_dataset(rows, l25_min_liq_to_mcap=THRESHOLD)
    assert len(out) == 1
    assert out[0]["l25_passes"] is None
    assert "simulated failure" in out[0]["error"]


# ---------------------------------------------------------------------------
# the three datasets stay separate
# ---------------------------------------------------------------------------


def test_only_prospective_data_can_be_a_validating_dataset():
    assert VALIDATING is Dataset.PROSPECTIVE_PAPER
    assert VALIDATING is not Dataset.IN_SAMPLE
    assert VALIDATING is not Dataset.POST_DERIVATION_RECONSTRUCTED


def test_the_three_datasets_are_distinct_values():
    assert len({d.value for d in Dataset}) == 3


def test_datasets_are_never_merged(recorder: ForwardRecorder, deployment):
    recorder.candidate(candidate_id="p1", mint="M", route="FAST_ENTRY",
                       features=_feat(), at=T0 + 1)
    recon = reconstruct_dataset(
        [{"candidate_id": "h1", "at": T0 + 2, "route": "FAST_ENTRY",
          "signal_quality": 0.9,
          "features": {"market_cap_usd": 1000, "liquidity_usd": 5000}}],
        l25_min_liq_to_mcap=THRESHOLD)
    stored = recorder.all_records()
    assert all(r["dataset"] == Dataset.PROSPECTIVE_PAPER.value for r in stored)
    assert all(r["dataset"] == Dataset.POST_DERIVATION_RECONSTRUCTED.value
               for r in recon)
    # and mixing them for a verdict is a data-quality problem, not a merge
    rep = build_report(list(stored) + recon, deployment=deployment)
    assert rep.status is Status.DATA_QUALITY_PROBLEM


# ---------------------------------------------------------------------------
# every record is stamped
# ---------------------------------------------------------------------------


def test_every_written_record_carries_its_provenance(recorder: ForwardRecorder,
                                                      deployment: Deployment):
    recorder.candidate(candidate_id="c", mint="M", route="FAST_ENTRY",
                       features=_feat(), at=T0 + 1)
    recorder.trade("entry", {"trade_id": "t", "at": T0 + 2, "mint": "M"})
    for rec in recorder.all_records():
        assert rec["dataset"] == Dataset.PROSPECTIVE_PAPER.value
        assert rec["deployment_id"] == deployment.deployment_id
        assert rec["strategy"] == "L25"
        assert rec["l25_min_liq_to_mcap"] == THRESHOLD
        assert rec["kind"] in ("candidate", "entry", "exit", "unfilled")


def test_an_unknown_trade_kind_is_refused(recorder: ForwardRecorder):
    with pytest.raises(ValueError):
        recorder.trade("teleport", {"trade_id": "t", "at": T0 + 1, "mint": "M"})


def test_missing_required_fields_are_flagged_not_defaulted(recorder):
    rec = recorder.candidate(candidate_id="c", mint=None, route="FAST_ENTRY",
                             features={}, at=T0 + 1)
    assert set(rec["missing_fields"]) >= {"mint", "market_cap_usd",
                                          "liquidity_usd", "liq_to_mcap"}
    # the value stays None -- a hole, not a plausible number
    assert rec["liq_to_mcap"] is None


def test_records_survive_a_round_trip(recorder: ForwardRecorder):
    recorder.candidate(candidate_id="c", mint="M", route="FAST_ENTRY",
                       features=_feat(), at=T0 + 1)
    rows = list(read_jsonl(recorder.candidates_path))
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "c"
    assert rows[0]["liq_to_mcap"] == 2.0


# ---------------------------------------------------------------------------
# the status ladder
# ---------------------------------------------------------------------------


def test_no_deployment_means_no_verdict(recorder: ForwardRecorder):
    rep = build_report(recorder.all_records(), deployment=None)
    assert rep.status is Status.NO_DEPLOYMENT
    assert rep.is_validating is False


def test_too_few_trades_reports_nothing(recorder, deployment):
    for i in range(5):
        _trade(recorder, i, +1.0)
    rep = build_report(recorder.all_records(), deployment=deployment)
    assert rep.status is Status.NOT_ENOUGH_DATA
    assert rep.is_validating is False


def test_a_mid_sample_is_insufficient_not_consistent(recorder, deployment):
    n = DEFAULT_POLICY.min_closed_for_any_read + 5
    for i in range(n):
        _trade(recorder, i, +2.0)
    rep = build_report(recorder.all_records(), deployment=deployment)
    assert rep.status is Status.INSUFFICIENT_SAMPLE
    assert rep.is_validating is False
    assert rep.criteria["sample >= min_for_conclusion"] is False


def test_dirty_data_suppresses_a_good_result(recorder, deployment):
    """
    A strongly positive sample with one missing field must not reach a verdict.

    This is the single most important test in the file.
    """
    n = DEFAULT_POLICY.min_closed_for_conclusion + 20
    for i in range(n):
        _trade(recorder, i, +5.0)
    clean = build_report(recorder.all_records(), deployment=deployment)
    assert clean.status is Status.CONSISTENT, "fixture should be genuinely good"

    # now poison one exit
    bad = recorder.trade("exit", {"trade_id": "ghost", "at": T0 + 9_999,
                                  "mint": "X"})   # no return_pct, no entry
    assert "missing_fields" in bad
    rep = build_report(recorder.all_records(), deployment=deployment)
    assert rep.status is Status.DATA_QUALITY_PROBLEM
    assert rep.is_validating is False
    assert "performance is not reported" in " ".join(rep.notes)


def test_an_unlinked_exit_is_a_quality_problem(recorder, deployment):
    recorder.trade("exit", {"trade_id": "no-entry", "at": T0 + 5, "mint": "M",
                            "return_pct": 1.0, "net_pnl_usd": 0.01,
                            "holding_s": 1.0})
    rep = build_report(recorder.all_records(), deployment=deployment)
    assert rep.status is Status.DATA_QUALITY_PROBLEM


def test_a_foreign_deployment_id_is_a_quality_problem(recorder, deployment):
    _trade(recorder, 1, 1.0)
    rows = list(read_jsonl(recorder.trades_path))
    rows[0]["deployment_id"] = "someone-elses-run"
    with recorder.trades_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    rep = build_report(recorder.all_records(), deployment=deployment)
    assert rep.status is Status.DATA_QUALITY_PROBLEM


def test_a_good_large_sample_is_consistent(recorder, deployment):
    n = DEFAULT_POLICY.min_closed_for_conclusion + 60
    for i in range(n):
        _trade(recorder, i, +3.0 if i % 3 else -1.0)
    rep = build_report(recorder.all_records(), deployment=deployment)
    assert rep.status is Status.CONSISTENT
    assert rep.is_validating is True
    assert rep.dataset is VALIDATING


def test_a_large_losing_sample_is_degraded_not_consistent(recorder, deployment):
    n = DEFAULT_POLICY.min_closed_for_conclusion + 60
    for i in range(n):
        _trade(recorder, i, -2.0 if i % 2 else +0.2)
    rep = build_report(recorder.all_records(), deployment=deployment)
    assert rep.status is Status.DEGRADED
    assert rep.is_validating is False
    assert rep.criteria["expectancy > 0"] is False


def test_consistency_requires_the_interval_to_exclude_zero(recorder, deployment):
    """
    A positive mean with an interval spanning zero is NOT consistent.

    Constructed to be the exact case that separates 'up on average' from
    'demonstrably up'.
    """
    # 96 tiny gains and 4 small losses: the MEAN is positive, but the spread is
    # wide relative to it, so the interval covers zero. Chosen so the CI is the
    # ONLY failing criterion -- the drawdown stays well inside its limit, which
    # is what makes this a test of the CI rather than of the drawdown.
    n = DEFAULT_POLICY.min_closed_for_conclusion
    returns = [0.1] * 96 + [-2.0] * 4
    assert len(returns) == n
    for i, r in enumerate(returns):
        _trade(recorder, i, r)
    rep = build_report(recorder.all_records(), deployment=deployment)
    assert rep.metrics.expectancy_pct > 0, "the naive mean is positive"
    assert rep.criteria["sample >= min_for_conclusion"] is True
    assert rep.criteria["max drawdown within limit"] is True
    assert rep.criteria["95% CI excludes zero"] is False
    assert rep.status is Status.DEGRADED
    assert rep.is_validating is False


def test_excessive_drawdown_fails_even_when_the_mean_is_positive(recorder,
                                                                deployment):
    n = DEFAULT_POLICY.min_closed_for_conclusion + 20
    # many small gains then a crash: positive expectancy, breached drawdown
    for i in range(n):
        _trade(recorder, i, +4.0 if i < n - 3 else -55.0)
    rep = build_report(recorder.all_records(), deployment=deployment)
    assert rep.criteria["max drawdown within limit"] is False
    assert rep.status is Status.DEGRADED


def test_no_status_other_than_consistent_claims_validation():
    for s in Status:
        if s is not Status.CONSISTENT:
            assert s in NON_VALIDATING_STATUSES, s
    assert Status.CONSISTENT not in NON_VALIDATING_STATUSES


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_the_report_is_byte_identical_across_runs(recorder, deployment):
    for i in range(40):
        _trade(recorder, i, +1.5 if i % 2 else -0.5)
    recs = recorder.all_records()
    a = build_report(recs, deployment=deployment).as_dict()
    b = build_report(recs, deployment=deployment).as_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_the_report_does_not_depend_on_record_order(recorder, deployment):
    for i in range(40):
        _trade(recorder, i, +1.5 if i % 2 else -0.5)
    recs = recorder.all_records()
    a = build_report(recs, deployment=deployment).as_dict()
    b = build_report(list(reversed(recs)), deployment=deployment).as_dict()
    # trade ORDER is meaningful for drawdown, so metrics may legitimately differ,
    # but the status must not flip on ordering alone
    assert a["status"] == b["status"]


def test_the_bootstrap_is_seeded_not_random_per_run():
    returns = [1.0, -2.0, 3.0, 0.5, -1.5] * 8
    assert expectancy_ci(returns) == expectancy_ci(returns)
    lo, hi = expectancy_ci(returns)
    assert lo is not None and hi is not None and lo <= hi


def test_a_different_seed_gives_a_different_interval():
    returns = [1.0, -2.0, 3.0, 0.5, -1.5] * 8
    a = expectancy_ci(returns, policy=ValidationPolicy(bootstrap_seed=1))
    b = expectancy_ci(returns, policy=ValidationPolicy(bootstrap_seed=2))
    assert a != b


def test_a_too_small_sample_has_no_interval():
    assert expectancy_ci([]) == (None, None)
    assert expectancy_ci([1.0]) == (None, None)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_no_trades_gives_nulls_not_zeros():
    """A null is honest; a 0.0% expectancy on zero trades is a lie."""
    m = compute_metrics([])
    assert m.n == 0
    assert m.win_rate is None
    assert m.expectancy_pct is None
    assert m.profit_factor is None
    assert m.max_drawdown_pct is None


def test_metrics_on_a_known_sample():
    trades = [
        {"return_pct": 10.0, "holding_s": 10.0, "mfe_pct": 12.0, "mae_pct": -1.0},
        {"return_pct": -5.0, "holding_s": 20.0, "mfe_pct": 2.0, "mae_pct": -6.0},
        {"return_pct": 0.0, "holding_s": 30.0, "mfe_pct": 1.0, "mae_pct": -1.0},
    ]
    m = compute_metrics(trades, policy=ValidationPolicy(bootstrap_draws=50))
    assert m.n == 3
    assert m.wins == 1 and m.losses == 1
    assert m.win_rate == pytest.approx(1 / 3)
    assert m.expectancy_pct == pytest.approx(5 / 3)
    assert m.median_holding_s == pytest.approx(20.0)
    assert m.mean_mfe_pct == pytest.approx(5.0)
    # 10 then -5 then 0: 100 -> 110 -> 104.5; dd = (110 - 104.5) / 110
    assert m.max_drawdown_pct == pytest.approx(5.5 / 110 * 100)


def test_a_trade_with_no_return_is_excluded_not_zeroed():
    trades = [{"return_pct": 10.0}, {"return_pct": None}, {"return_pct": "x"}]
    m = compute_metrics(trades)
    assert m.n == 1
    assert m.expectancy_pct == pytest.approx(10.0)


def test_profit_factor_is_none_when_nothing_was_lost():
    m = compute_metrics([{"return_pct": 5.0}, {"return_pct": 3.0}])
    assert m.profit_factor is None, "infinite profit factor is not a number"


def test_non_finite_returns_are_dropped():
    m = compute_metrics([{"return_pct": float("nan")}, {"return_pct": 2.0},
                         {"return_pct": float("inf")}])
    assert m.n == 1


# ---------------------------------------------------------------------------
# the policy is pre-declared and stated
# ---------------------------------------------------------------------------


def test_the_policy_is_reported_with_every_verdict(recorder, deployment):
    rep = build_report(recorder.all_records(), deployment=deployment)
    d = rep.as_dict()
    assert d["policy"] == DEFAULT_POLICY.as_dict()
    assert d["policy"]["bootstrap_seed"] == BOOT_SEED_EXPECTED


BOOT_SEED_EXPECTED = 20240917


def test_the_thresholds_are_not_tiny():
    """Small gates would let a lucky streak read as validation."""
    assert DEFAULT_POLICY.min_closed_for_any_read >= 30
    assert DEFAULT_POLICY.min_closed_for_conclusion >= 100
    assert DEFAULT_POLICY.ci_level >= 0.95


def test_the_report_says_it_does_not_validate(recorder, deployment):
    rep = build_report(recorder.all_records(), deployment=deployment)
    text = rep.render()
    assert "does NOT validate" in text
    assert "PROSPECTIVE_PAPER" in text


def test_status_renders_human_readable(recorder, deployment):
    for i in range(3):
        _trade(recorder, i, 1.0)
    out = build_report(recorder.all_records(), deployment=deployment).render()
    assert "FORWARD PAPER VALIDATION" in out
    assert "NOT_ENOUGH_DATA" in out
    assert "data quality" in out.lower()


def test_serialisation_is_json_safe(recorder, deployment):
    for i in range(3):
        _trade(recorder, i, 1.0)
    d = build_report(recorder.all_records(), deployment=deployment).as_dict()
    json.dumps(d)  # must not raise
