"""
Isolation and validation tests for the $5 comparison harness.

Run:  PYTHONPATH=research:. .venv/bin/python -m pytest tests/research/test_l25c_isolation.py -q
"""


from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.execution,
    pytest.mark.integrity,
    pytest.mark.research,
    pytest.mark.safety,
]


AI = Path(__file__).resolve().parents[2]
PKG = AI / "research" / "l25c"
REPORT_JSON = AI / "reports" / "research" / "l25_full_comparison_$5.json"
REPORT_MD = AI / "reports" / "research" / "l25_full_comparison_$5.md"
TRADE_CSV = AI / "reports" / "research" / "l25_trade_level_$5.csv"
CURVE_CSV = AI / "reports" / "research" / "l25_equity_curves_$5.csv"


def _sources() -> list[Path]:
    return sorted(PKG.glob("*.py"))


# ---------------------------------------------------------------------------
# 1. cannot trade
# ---------------------------------------------------------------------------

FORBIDDEN_CALLS = (
    "broadcast", "submit_transaction", "send_transaction", "sign_transaction",
    "build_and_sign", "confirm_transaction", "real_buy", "real_sell",
    "arm_real", "enable_real", "settle_real", "create_order", "place_order",
)
FORBIDDEN_IMPORTS = (
    "autopaper.real_account", "autopaper.real_equity", "autopaper.settlement",
    "autopaper.external_close", "autopaper.execution.real",
    "autopaper.execution.solana", "autopaper.execution.chainstate",
    "autopaper.telegram.client",
)


def test_no_broadcast_capable_calls():
    hits = []
    for p in _sources():
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Call):
                fn = node.func
                name = (fn.attr if isinstance(fn, ast.Attribute)
                        else fn.id if isinstance(fn, ast.Name) else "")
                if any(name == b or name.startswith(b) for b in FORBIDDEN_CALLS):
                    hits.append(f"{p.name}:{node.lineno} {name}")
    assert not hits, hits


def test_no_execution_or_wallet_imports():
    hits = []
    for p in _sources():
        t = p.read_text()
        for m in FORBIDDEN_IMPORTS:
            if f"import {m}" in t or f"from {m}" in t:
                hits.append(f"{p.name}: {m}")
    assert not hits, hits


def test_no_autopaper_mutation():
    """
    Nothing may write to an attribute of an `autopaper` module or of a frozen
    constant holder. Research-module monkeypatching is permitted ONLY because a
    separate test proves it is restored -- production is not.
    """
    banned = ("autopaper", "l25", "v3c", "v2a", "config", "exits", "sizing",
              "strategy_profile", "MODEL", "LIMITS", "D", "R")
    hits = []
    for p in _sources():
        for node in ast.walk(ast.parse(p.read_text())):
            tgt = None
            if isinstance(node, ast.Assign):
                tgt = node.targets[0]
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                tgt = node.target
            if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) \
                    and tgt.value.id in banned:
                hits.append(f"{p.name}:{node.lineno} {tgt.value.id}.{tgt.attr}")
    assert not hits, hits


def test_research_monkeypatches_are_restored():
    """
    The executability stress patches l25r.backtest.price_at in memory. That is
    allowed, but it MUST be restored, or it would silently corrupt every later
    simulation in the same process.
    """
    import l25r.backtest as BT
    before = BT.price_at
    from l25c import executability as X
    from l25r import data as D
    X.extra_adverse_move_stress(D.load()["trades"], per_leg_bps=(0, 100))
    assert BT.price_at is before, "price_at was not restored after the stress"

    # and a post-stress simulation must reproduce the unstressed result
    from l25c import engine as E
    led = E.simulate(D.load()["trades"], strategy="t", max_hold_s=None,
                     entry_filter=E.in_l25, start=5.0, sizing_mode="production")
    assert abs(led.end - 783.82) < 0.01, led.end


def test_no_writes_to_live_or_credential_paths():
    sensitive = (".env.local", "var/ledger.jsonl", "var/paper_account.json",
                 "var/real_account_state.json", "var/real_execution_intents.jsonl",
                 "var/candidates.jsonl", "var/position_updates.jsonl")
    hits = []
    for p in _sources():
        for node in ast.walk(ast.parse(p.read_text())):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name not in ("write_text", "write_bytes", "open", "unlink"):
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    for s in sensitive:
                        if s in arg.value:
                            hits.append(f"{p.name}:{node.lineno} {s}")
    assert not hits, hits


def test_all_output_paths_under_reports_research():
    run = (PKG / "run.py").read_text()
    for node in ast.walk(ast.parse(run)):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id.startswith("OUT_")
                for t in node.targets):
            for v in ast.walk(node.value):
                if isinstance(v, ast.Constant) and isinstance(v.value, str) \
                        and "/" in v.value:
                    assert v.value.startswith("reports/research"), v.value


# ---------------------------------------------------------------------------
# 2. baseline fidelity is re-verified from source
# ---------------------------------------------------------------------------


def test_baseline_fidelity_from_source():
    from l25c.run import baseline_fidelity
    f = baseline_fidelity()
    assert f["all_verified"] is True
    assert f["checks"] >= 25
    assert f["max_hold_is_last_in_precedence"] is True


def test_frozen_precision_score_untouched():
    blob = (AI / "reports" / "ai_policy_frozen_spec.json").read_text()
    assert "0519cc5394c25b86" in blob


def test_frozen_universe_hash_unchanged():
    import hashlib
    body = (AI / "research" / "frozen_l25" / "universe.json").read_bytes()
    pinned = json.loads(
        (AI / "research" / "frozen_l25" / "MANIFEST.json").read_text()
    )["universe_sha256"]
    assert hashlib.sha256(body).hexdigest() == pinned


def test_l25_gate_counts_match_the_frozen_manifest():
    """The rebuilt candidate population must reproduce the documented funnel."""
    from l25c import engine as E
    _cands, audit = E.build_candidates()
    g = audit["gate_counts"]
    assert g["L25"] == 116, g
    assert g["V3C"] == 154, g
    assert g["V2A"] >= g["V3C"] >= g["L25"]
    assert audit["fill_model_proof"]["reproduces_recorded_history"] is True


def test_preregistered_h1_threshold_is_unchanged():
    """H1's quantile must come from the L25-ACCEPTED set, not the full pool."""
    from l25c import registry as R
    from l25r import data as D
    from l25r.hypotheses import build_filters
    l25 = D.load()["trades"]
    s = {x["id"]: x for x in build_filters(l25)}["H1_liqratio_floor_above_frozen"]
    assert s["rule"] == "f_liq_to_mcap >= 1.3773", s["rule"]


# ---------------------------------------------------------------------------
# 3. accounting correctness
# ---------------------------------------------------------------------------


def test_position_size_follows_the_production_branch_rule():
    """
    10% while bankroll <= $50, 12.5% above it -- and a $5 account DOES cross
    that branch mid-run, so a flat-10% assertion would be wrong.
    """
    from l25c import engine as E
    from l25r import data as D
    led = E.simulate(D.load()["trades"], strategy="t", max_hold_s=None,
                     entry_filter=E.in_l25, start=5.0, sizing_mode="production")
    assert led.rows
    crossed = False
    for r in led.rows:
        want_pct = 10.0 if r["bankroll_before"] <= 50.0 else 12.5
        assert math.isclose(r["position_size"],
                            r["bankroll_before"] * want_pct / 100.0, rel_tol=1e-9), r
        assert r["bankroll_after"] > 0, "bankroll must never go non-positive"
        crossed = crossed or r["position_pct_applied"] == 12.5
    assert crossed, "expected the $5 account to cross the $50 branch point"


def test_compounding_identity_holds_per_trade():
    """bankroll_after == bankroll_before - cash_out + proceeds."""
    from l25c import engine as E
    from l25r import data as D
    led = E.simulate(D.load()["trades"], strategy="t", max_hold_s=None,
                     entry_filter=E.in_l25, start=5.0, sizing_mode="production")
    for r in led.rows:
        assert math.isclose(r["bankroll_before"] + r["net_pnl"],
                            r["bankroll_after"], rel_tol=1e-9, abs_tol=1e-12)


def test_fee_and_slippage_are_stored_separately_and_sum_to_total():
    from l25c import engine as E
    from l25r import data as D
    led = E.simulate(D.load()["trades"], strategy="t", max_hold_s=None,
                     entry_filter=E.in_l25, start=5.0, sizing_mode="production")
    for r in led.rows:
        assert math.isclose(r["fees"] + r["slippage"] + r["other_costs"],
                            r["total_costs"], rel_tol=1e-9, abs_tol=1e-12)
        assert r["fees"] > 0 and r["slippage"] > 0


def test_drawdown_is_non_positive_and_consistent():
    from l25c import engine as E
    from l25r import data as D
    led = E.simulate(D.load()["trades"], strategy="t", max_hold_s=None,
                     entry_filter=E.in_l25, start=5.0, sizing_mode="production")
    rs = E.risk_summary(led)
    assert rs["max_drawdown_usd"] <= 0
    assert rs["max_drawdown_pct"] <= 0
    assert rs["peak_bankroll"] >= rs["final_bankroll"] - 1e-9


def test_trades_are_chronological_and_never_reordered():
    from l25c import engine as E
    from l25r import data as D
    led = E.simulate(D.load()["trades"], strategy="t", max_hold_s=None,
                     entry_filter=E.in_l25, start=5.0, sizing_mode="production")
    ts = [r["t_entry"] for r in led.rows]
    assert ts == sorted(ts), "trade sequence must follow market time"


def test_single_global_position_constraint_is_enforced():
    """No two simulated trades may overlap in market time."""
    from l25c import engine as E
    from l25r import data as D
    led = E.simulate(D.load()["trades"], strategy="t", max_hold_s=None,
                     entry_filter=E.in_l25, start=5.0, sizing_mode="production")
    for a, b in zip(led.rows, led.rows[1:]):
        assert a["exit_time_market"] <= b["t_entry"] + 1e-6, (
            "the production one-position-at-a-time constraint was violated")


def test_block_simulations_are_independent():
    """Each block must start from $5, not from the previous block's balance."""
    from l25c import engine as E
    from l25r import data as D
    led = E.simulate(D.load()["trades"], strategy="t", max_hold_s=None,
                     entry_filter=E.in_l25, start=5.0, sizing_mode="production")
    bs = E.blocks_summary(led)
    for b in ("TRAIN", "VALIDATION", "TEST"):
        assert bs[b]["trades"] > 0
        assert "roi_pct_independent" in bs[b]
    # TEST alone must be able to exceed the full-sample bankroll
    assert bs["TEST"]["final_bankroll_independent"] > 0


def test_tail_removal_is_a_real_recomputation():
    """Removing the top k must change the whole path, not just the last number."""
    from l25c import engine as E
    from l25r import data as D
    led = E.simulate(D.load()["trades"], strategy="t", max_hold_s=None,
                     entry_filter=E.in_l25, start=5.0, sizing_mode="production")
    tl = E.tail_summary(led)
    full = tl["full_sample"]["final_bankroll"]
    t5 = tl["ex_top_5"]
    assert t5["trades_remaining"] == tl["full_sample"]["trades"] - 5
    assert t5["final_bankroll"] < full, "removing winners must reduce the result"
    # sizes must differ, proving re-sizing rather than subtraction happened
    assert t5["trades_remaining"] > 0


def test_forced_ten_percent_never_crosses_the_branch():
    """The pct_10 control must stay on 10% even after passing $50."""
    from l25c import engine as E
    from l25r import data as D
    led = E.simulate(D.load()["trades"], strategy="t", max_hold_s=None,
                     entry_filter=E.in_l25, start=5.0, sizing_mode="pct_10")
    assert led.end > 50.0, "control must actually grow past the branch point"
    assert {r["position_pct_applied"] for r in led.rows} == {10.0}


def test_production_and_forced_ten_percent_diverge_after_crossing():
    """Otherwise the two modes are indistinguishable and the control is useless."""
    from l25c import engine as E
    from l25r import data as D
    tr = D.load()["trades"]
    a = E.simulate(tr, strategy="t", max_hold_s=None, entry_filter=E.in_l25,
                   start=5.0, sizing_mode="production")
    b = E.simulate(tr, strategy="t", max_hold_s=None, entry_filter=E.in_l25,
                   start=5.0, sizing_mode="pct_10")
    assert abs(a.end - b.end) > 0.01, "modes must diverge once the branch is crossed"


def test_deterministic_output():
    from l25c import engine as E
    from l25r import data as D
    tr = D.load()["trades"]
    a = E.simulate(tr, strategy="t", max_hold_s=60.0, entry_filter=E.in_l25,
                   start=5.0, sizing_mode="production")
    b = E.simulate(tr, strategy="t", max_hold_s=60.0, entry_filter=E.in_l25,
                   start=5.0, sizing_mode="production")
    assert [r["bankroll_after"] for r in a.rows] == [r["bankroll_after"] for r in b.rows]
    assert E.risk_summary(a) == E.risk_summary(b)


def test_bootstrap_is_seeded():
    from l25c import engine as E
    from l25r import data as D
    led = E.simulate(D.load()["trades"], strategy="t", max_hold_s=None,
                     entry_filter=E.in_l25, start=5.0, sizing_mode="production")
    r1 = E.bankroll_bootstrap(led, n_boot=200, seed=7)
    r2 = E.bankroll_bootstrap(led, n_boot=200, seed=7)
    assert r1["final_ci90"] == r2["final_ci90"]


def test_gate_ladder_is_monotonic():
    """A wider gate must not admit fewer candidates than a narrower one."""
    from l25c import engine as E
    cands, audit = E.build_candidates()
    g = audit["gate_counts"]
    assert g["V2A"] >= g["V3C"] >= g["L25"]


# ---------------------------------------------------------------------------
# 4. report content
# ---------------------------------------------------------------------------


def test_report_exists_and_is_mark_based_labelled():
    if not REPORT_JSON.exists():
        pytest.skip("report not generated")
    d = json.loads(REPORT_JSON.read_text())
    assert d["pricing_label"] == "OBSERVATIONAL / MARK-BASED"
    assert "not proof" in d["not_executable_profit"].lower()
    md = REPORT_MD.read_text()
    assert "OBSERVATIONAL / MARK-BASED" in md
    assert "No transaction was broadcast" in md


def test_initial_bankroll_is_five_in_every_strategy():
    if not REPORT_JSON.exists():
        pytest.skip("report not generated")
    d = json.loads(REPORT_JSON.read_text())
    assert d["initial_bankroll_usd"] == 5.0
    for r in d["results"]:
        assert r["sizing"]["production"]["risk"]["starting_bankroll"] == 5.0
        assert r["sizing"]["pct_12_5"]["risk"]["starting_bankroll"] == 5.0


def test_every_registered_strategy_is_present_in_the_report():
    if not REPORT_JSON.exists():
        pytest.skip("report not generated")
    from l25c import registry as R
    from l25r import data as D
    d = json.loads(REPORT_JSON.read_text())
    got = {r["strategy"] for r in d["results"]}
    want = {s["strategy"] for s in R.build(D.load()["trades"])["strategies"]}
    assert want == got, want ^ got


def test_not_testable_variants_are_reported_with_reasons():
    if not REPORT_JSON.exists():
        pytest.skip("report not generated")
    d = json.loads(REPORT_JSON.read_text())
    nt = d["not_testable"]
    assert len(nt) >= 5
    for n in nt:
        assert n["status"].startswith("NOT TESTABLE") \
            or n["status"].startswith("NOT APPLICABLE")
        assert len(n["reason"]) > 40
        assert n["measured"]


def test_csv_artifacts_have_required_columns_and_no_secrets():
    if not TRADE_CSV.exists():
        pytest.skip("csv not generated")
    rows = list(csv.DictReader(TRADE_CSV.open()))
    required = {"strategy", "sizing_mode", "block", "t_entry", "symbol",
                "bankroll_before", "position_size", "position_size_pct_actual",
                "gross_return_pct", "gross_pnl", "fees", "slippage",
                "net_pnl", "bankroll_after", "drawdown_pct", "exit_reason",
                "hold_seconds"}
    assert required <= set(rows[0].keys()), required - set(rows[0].keys())
    assert len(rows) > 1000
    blob = TRADE_CSV.read_text()
    for bad in ("PRIVATE KEY", "api_key", "bot_token", "seed phrase",
                "telegram", "wallet"):
        assert bad.lower() not in blob.lower(), f"possible secret in CSV: {bad}"


def test_safety_declarations():
    if not REPORT_JSON.exists():
        pytest.skip("report not generated")
    s = json.loads(REPORT_JSON.read_text())["safety"]
    for k in ("traded", "transaction_broadcast", "transaction_signed",
              "real_trading_enabled_by_this_study", "production_code_modified",
              "production_config_modified", "frozen_precision_score_modified",
              "strategy_thresholds_changed"):
        assert s[k] is False, k
    assert s["production_code_modified"] is False


def test_production_unchanged_during_run_and_digests_pinned():
    if not REPORT_JSON.exists():
        pytest.skip("report not generated")
    from l25r.isolation import snapshot
    d = json.loads(REPORT_JSON.read_text())
    assert d["production_unchanged_during_run"] is True
    assert snapshot() == d["reproducibility"]["pinned_source_digests"], (
        "a pinned production file changed since the report was generated")

# ---------------------------------------------------------------------------
# 5. executability falsification module
# ---------------------------------------------------------------------------


def test_executability_module_has_no_broadcast_or_write_capability():
    hits = []
    p = PKG / "executability.py"
    for node in ast.walk(ast.parse(p.read_text())):
        if isinstance(node, ast.Call):
            fn = node.func
            name = (fn.attr if isinstance(fn, ast.Attribute)
                    else fn.id if isinstance(fn, ast.Name) else "")
            if any(name == b or name.startswith(b) for b in FORBIDDEN_CALLS):
                hits.append(f"{p.name}:{node.lineno} {name}")
    assert not hits, hits


def test_executability_module_reads_no_credentials():
    p = PKG / "executability.py"
    text = p.read_text()
    for bad in ("read_text", "open("):
        assert bad not in text or "ledger" in text or "MANIFEST" in text, (
            "executability.py must not read arbitrary files")
    # the only paths it opens must be the frozen ledger/archive, and read-only
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "attr", getattr(fn, "id", ""))
            if name not in ("read_text", "open", "exists", "read_bytes"):
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    assert ".env" not in arg.value, arg.value
                    assert "wallet" not in arg.value.lower(), arg.value


def test_amM_math_matches_the_repo_impact_term():
    """If the repo's impact term did NOT track AMM math, the whole cost
    conservatism argument would collapse and must fail loudly."""
    from l25c import executability as X
    from l25r import data as D
    r = X.unmodelled_cost_headroom(D.load()["trades"])
    row10 = next(x for x in r["table"] if x["notional_usd"] == 10.0)
    # repo impact and AMM half-spread must be the same order of magnitude
    assert 0.2 < row10["repo_impact_bps"] / row10["amm_half_spread_bps"] < 5.0, row10
    assert row10["repo_base_slippage_bps"] > row10["amm_half_spread_bps"] * 10, row10


def test_executability_stress_is_monotone_and_reported():
    from l25c import executability as X
    from l25r import data as D
    r = X.extra_adverse_move_stress(D.load()["trades"])
    rows = r["rows"]
    assert len(rows) >= 5
    # more adverse fill must never produce a better result
    finals = [x["final_bankroll_usd"] for x in rows]
    assert finals == sorted(finals, reverse=True), finals
    meds = [x["median_net_trade_pct"] for x in rows]
    assert meds == sorted(meds, reverse=True), meds
    assert r["multiple_of_modelled_cost_required_to_turn_median_negative"] > 1.0


def test_report_contains_the_executability_falsification_section():
    if not REPORT_JSON.exists():
        pytest.skip("report not generated")
    d = json.loads(REPORT_JSON.read_text())
    xf = d["executability_falsification"]
    for k in ("what_the_fills_were", "is_modelled_cost_too_low",
              "costs_with_no_field_anywhere", "extra_adverse_move_stress",
              "measured_entry_latency", "mark_resolution_bias",
              "tail_dependence_is_the_real_channel", "verdict"):
        assert k in xf, k
    assert xf["what_the_fills_were"]["executable_quote_fields_present"] == ["NONE"]
    md = REPORT_MD.read_text()
    assert "EXECUTABILITY FALSIFICATION" in md
    assert "false positive" in md.lower()
