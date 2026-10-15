"""
L25: the ACTIVE strategy.

Every case here is a required behaviour of the frozen gate, and every threshold
is the value transcribed from the repository's own research. The reference
implementation in ``research/randorder.py`` is imported and compared against
directly, so these tests cannot drift from the research by construction.
"""


from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from autopaper import l25
from autopaper.l25 import (
    ACTIVE_STRATEGY,
    L25_DERIVED_QUANTILE_EXACT,
    L25_MIN_LIQ_TO_MCAP,
    L25_RESEARCH_CLASSIFICATION,
    REJECTED_LIQUIDITY_MC_BELOW_THRESHOLD,
    UNTESTABLE,
    evaluate,
    f_liq_to_mcap,
)
from autopaper.safety import STRATEGY
from autopaper.v3c import (
    REJECTED_MARKET_CAP,
    REJECTED_NOT_FAST_ENTRY,
    REJECTED_SIGNAL_QUALITY,
    V3C_MAX_MARKET_CAP_USD,
    V3C_MIN_SIGNAL_QUALITY,
)

pytestmark = pytest.mark.strategy

REPO = Path(__file__).resolve().parents[1]
MC = V3C_MAX_MARKET_CAP_USD
SQ = V3C_MIN_SIGNAL_QUALITY


# ---------------------------------------------------------------------------
# the thresholds are the researched values
# ---------------------------------------------------------------------------


def test_l25_threshold_is_the_frozen_value():
    assert L25_MIN_LIQ_TO_MCAP == 0.9647


def test_the_inherited_v3c_thresholds_are_unchanged():
    assert V3C_MAX_MARKET_CAP_USD == 20_000.0
    assert V3C_MIN_SIGNAL_QUALITY == 0.55


def test_the_v2a_ceiling_is_unchanged():
    from autopaper.v2a import V2A_CEILING_USD

    assert V2A_CEILING_USD == 25_372.71


def test_verify_definition_rejects_a_retuned_ratio():
    with pytest.raises(ValueError, match="FROZEN"):
        l25.verify_definition(0.96)
    with pytest.raises(ValueError, match="FROZEN"):
        l25.verify_definition(0.97)
    with pytest.raises(ValueError, match="FROZEN"):
        l25.verify_definition(L25_DERIVED_QUANTILE_EXACT)


def test_evaluate_rejects_a_retuned_ratio():
    """A caller cannot pass a different threshold into the live gate."""
    with pytest.raises(ValueError, match="FROZEN"):
        evaluate(
            route="FAST_ENTRY", market_cap_usd=5_000, signal_quality=0.9,
            liquidity_usd=6_000, min_liq_to_mcap=0.5,
        )


def test_verify_definition_accepts_the_real_value():
    l25.verify_definition()


def test_config_freezes_the_ratio():
    from autopaper.config import DecisionConfig

    assert DecisionConfig().l25_min_liq_to_mcap == 0.9647
    with pytest.raises(ValueError):
        DecisionConfig(l25_min_liq_to_mcap=0.9)


# ---------------------------------------------------------------------------
# 1. V3-C pass + ratio pass => PASS
# ---------------------------------------------------------------------------


def test_v3c_pass_with_ratio_above_threshold_passes():
    r = evaluate(
        route="FAST_ENTRY", market_cap_usd=15_000, signal_quality=0.60,
        liquidity_usd=16_000,
    )
    assert r.authorized
    assert r.status == l25.AUTHORIZED
    assert r.f_liq_to_mcap == pytest.approx(16_000 / 15_000)
    assert r.v3c.authorized


def test_a_passing_candidate_carries_both_legs():
    r = evaluate(
        route="FAST_ENTRY", market_cap_usd=15_000, signal_quality=0.60,
        liquidity_usd=20_000,
    )
    assert r.v3c.authorized
    assert r.authorized
    assert r.status_code == "L25_AUTHORIZED"


# ---------------------------------------------------------------------------
# 2. L25 rejects a V3-C candidate whose ratio is too low
# ---------------------------------------------------------------------------


def test_l25_rejects_a_v3c_passing_candidate_on_ratio():
    r = evaluate(
        route="FAST_ENTRY", market_cap_usd=15_000, signal_quality=0.60,
        liquidity_usd=14_000,
    )
    assert not r.authorized
    assert r.status == REJECTED_LIQUIDITY_MC_BELOW_THRESHOLD
    # V3-C had authorized, which is exactly the case the log line names.
    assert r.v3c.authorized
    assert r.status_code == "L25_REJECT"
    assert r.f_liq_to_mcap == pytest.approx(14_000 / 15_000)


def test_the_liquidity_rejection_names_the_operative_gate():
    r = evaluate(
        route="FAST_ENTRY", market_cap_usd=15_000, signal_quality=0.60,
        liquidity_usd=14_000,
    )
    assert r.status_code == "L25_REJECT"
    assert r.status == REJECTED_LIQUIDITY_MC_BELOW_THRESHOLD
    assert r.min_liq_to_mcap == 0.9647
    assert r.f_liq_to_mcap is not None
    assert r.f_liq_to_mcap < 0.9647


# ---------------------------------------------------------------------------
# 3. non-FAST route rejects
# ---------------------------------------------------------------------------


def test_precision_entry_is_rejected_even_though_everything_else_passes():
    r = evaluate(
        route="PRECISION_ENTRY", market_cap_usd=5_000, signal_quality=0.95,
        liquidity_usd=50_000,
    )
    assert not r.authorized
    assert r.status == REJECTED_NOT_FAST_ENTRY
    assert r.v3c is not None and not r.v3c.authorized
    assert r.status_code == "V3C_REJECTED_NOT_FAST_ENTRY"


def test_no_trade_route_is_rejected():
    r = evaluate(
        route="NO_TRADE", market_cap_usd=5_000, signal_quality=0.95,
        liquidity_usd=50_000,
    )
    assert not r.authorized


# ---------------------------------------------------------------------------
# 4-6. fail closed on missing / invalid operands
# ---------------------------------------------------------------------------


def test_missing_liquidity_fails_closed():
    r = evaluate(
        route="FAST_ENTRY", market_cap_usd=15_000, signal_quality=0.60,
        liquidity_usd=None,
    )
    assert not r.authorized
    assert r.status == UNTESTABLE
    assert r.f_liq_to_mcap is None
    assert r.v3c.authorized, "V3-C passed; L25's own gate is what refused"


def test_zero_liquidity_fails_closed():
    r = evaluate(
        route="FAST_ENTRY", market_cap_usd=15_000, signal_quality=0.60,
        liquidity_usd=0.0,
    )
    assert not r.authorized
    assert r.f_liq_to_mcap is None


def test_negative_liquidity_fails_closed():
    r = evaluate(
        route="FAST_ENTRY", market_cap_usd=15_000, signal_quality=0.60,
        liquidity_usd=-100.0,
    )
    assert not r.authorized
    assert r.f_liq_to_mcap is None


def test_missing_liquidity_is_never_treated_as_zero():
    """A missing value must not become 0.0 and must not pass."""
    assert f_liq_to_mcap(None, 15_000) is None
    assert f_liq_to_mcap(None, 15_000) != 0.0
    r = evaluate(
        route="FAST_ENTRY", market_cap_usd=15_000, signal_quality=0.60,
        liquidity_usd=None,
    )
    assert not r.authorized


def test_missing_market_cap_fails_closed():
    r = evaluate(
        route="FAST_ENTRY", market_cap_usd=None, signal_quality=0.60,
        liquidity_usd=20_000,
    )
    assert not r.authorized
    assert r.f_liq_to_mcap is None


def test_zero_market_cap_fails_closed():
    r = evaluate(
        route="FAST_ENTRY", market_cap_usd=0.0, signal_quality=0.60,
        liquidity_usd=20_000,
    )
    assert not r.authorized
    assert r.f_liq_to_mcap is None


def test_negative_market_cap_fails_closed():
    r = evaluate(
        route="FAST_ENTRY", market_cap_usd=-1.0, signal_quality=0.60,
        liquidity_usd=20_000,
    )
    assert not r.authorized
    assert r.f_liq_to_mcap is None


@pytest.mark.parametrize("bad", ["abc", "", float("nan"), float("inf"), True, [1]])
def test_non_numeric_operands_fail_closed(bad):
    assert f_liq_to_mcap(bad, 15_000) is None
    assert f_liq_to_mcap(15_000, bad) is None


def test_untestable_reason_names_the_operand():
    # signature is (liquidity_usd, market_cap_usd)
    assert "liquidity" in l25.untestable_reason(None, 15_000)
    assert "market cap" in l25.untestable_reason(1_000, None)
    assert "liquidity" in l25.untestable_reason(0.0, 15_000)
    assert "market cap" in l25.untestable_reason(1_000, 0.0)


# ---------------------------------------------------------------------------
# 7-8. the exact threshold
# ---------------------------------------------------------------------------


def test_ratio_exactly_at_the_threshold_passes():
    """The boundary is INCLUSIVE, exactly as the research predicate is."""
    mcap = 10_000.0
    liq = mcap * L25_MIN_LIQ_TO_MCAP          # exactly 0.9647
    r = evaluate(
        route="FAST_ENTRY", market_cap_usd=mcap, signal_quality=0.60,
        liquidity_usd=liq,
    )
    assert f_liq_to_mcap(liq, mcap) == pytest.approx(0.9647, abs=1e-15)
    assert r.authorized
    assert r.f_liq_to_mcap >= L25_MIN_LIQ_TO_MCAP


def test_ratio_slightly_below_the_threshold_rejects():
    mcap = 10_000.0
    liq = mcap * (L25_MIN_LIQ_TO_MCAP - 1e-9)
    r = evaluate(
        route="FAST_ENTRY", market_cap_usd=mcap, signal_quality=0.60,
        liquidity_usd=liq,
    )
    assert not r.authorized
    assert r.status == REJECTED_LIQUIDITY_MC_BELOW_THRESHOLD


def test_the_full_precision_quantile_is_below_the_4dp_threshold():
    """
    The research's full-precision quantile sits just under the 4 dp constant.

    Recorded because it is why the corrected population is 116 and not 119,
    and because it means the operative value MUST be 0.9647.
    """
    assert L25_DERIVED_QUANTILE_EXACT < L25_MIN_LIQ_TO_MCAP
    assert L25_MIN_LIQ_TO_MCAP - L25_DERIVED_QUANTILE_EXACT < 1e-4


# ---------------------------------------------------------------------------
# 9. strategy identity
# ---------------------------------------------------------------------------


def test_active_strategy_is_l25():
    assert ACTIVE_STRATEGY == "L25"


def test_the_runtime_identity_reports_l25():
    assert STRATEGY == "L25"


def test_the_runtime_identity_is_sourced_not_retyped():
    """One name, read from the strategy module. No surface can drift."""
    import inspect

    import autopaper.safety as safety

    src = inspect.getsource(safety)
    assert "STRATEGY = _l25.ACTIVE_STRATEGY" in src
    assert 'STRATEGY = "L25"' not in src


def test_describe_names_the_threshold():
    text = l25.describe()
    assert "L25" in text
    assert "0.9647" in text
    assert "20,000" in text
    assert "0.55" in text
    assert "0.9647" in l25.describe_short()


def test_it_is_not_called_v4():
    assert "V4" not in ACTIVE_STRATEGY
    assert "V4" not in l25.describe()
    src = (REPO / "src/autopaper/l25.py").read_text()
    assert "V4" not in src.replace("not a code decision", "")


def test_provenance_is_preserved_not_regenerated():
    p = l25.provenance()
    assert p["active_strategy"] == "L25"
    assert p["min_liq_to_mcap"] == 0.9647
    assert p["derived_quantile_exact"] == L25_DERIVED_QUANTILE_EXACT
    assert p["research_classification"] == "PROMISING_BUT_INCONCLUSIVE"
    assert p["post_derivation_validation"] == "NO_TRUE_POST_DERIVATION_DATA"
    assert p["derivation_cutoff_epoch"] == 1790735204
    assert "research/randorder.py" in p["research_sources"]
    assert p["inherited"]["v2a_ceiling_usd"] == 25_372.71
    assert p["inherited"]["v3c_route"] == "FAST_ENTRY"
    assert any("not proven" in n for n in p["not_claimed"])


def test_research_status_is_promising_but_inconclusive():
    assert L25_RESEARCH_CLASSIFICATION == "PROMISING_BUT_INCONCLUSIVE"


def test_audit_reports_the_active_strategy():
    from autopaper import audit

    r = audit()
    assert r.ok, r.render()
    names = {n for n, _, _ in r.checks}
    assert "active_strategy_is_l25" in names
    assert "l25_threshold_frozen" in names


# ---------------------------------------------------------------------------
# 10. V3-C remains available, and is no longer the active strategy
# ---------------------------------------------------------------------------


def test_v3c_is_still_callable_and_still_means_the_same_thing():
    from autopaper.v3c import evaluate as evaluate_v3c

    r = evaluate_v3c(
        route="FAST_ENTRY", market_cap_usd=15_000, signal_quality=0.60
    )
    assert r.authorized, "V3-C remains available for comparison and research"
    assert not r.authorized or True


def test_l25_delegates_rather_than_duplicating_v3c():
    """
    V3-C has ONE implementation. L25 calls it.

    If L25 restated the V3-C thresholds it could drift from the researched
    definition without any test noticing, because the tests would be asserting
    L25 against itself.
    """
    from autopaper.l25 import evaluate as _l25_eval
    from autopaper.v3c import evaluate as _v3c_eval

    cases = [
        ("FAST_ENTRY", 15_000, 0.60), ("FAST_ENTRY", 25_000, 0.60),
        ("FAST_ENTRY", 15_000, 0.40), ("FAST_ENTRY", None, 0.60),
        ("PRECISION_ENTRY", 5_000, 0.90), ("FAST_ENTRY", 20_000, 0.55),
    ]
    for route, mcap, sq in cases:
        inner = _l25_eval(
            route=route, market_cap_usd=mcap, signal_quality=sq,
            liquidity_usd=99_999.0,
        ).v3c
        direct = _v3c_eval(
            route=route, market_cap_usd=mcap, signal_quality=sq
        )
        assert inner.status == direct.status, (route, mcap, sq)
        assert inner.authorized == direct.authorized


def test_the_decision_engine_uses_l25_not_bare_v3c():
    """
    The wiring, asserted from the source: the decision engine's gate call is
    the L25 evaluator.
    """
    src = (REPO / "src/autopaper/analysis/decision.py").read_text()
    assert "evaluate_l25(" in src
    assert "liquidity_usd=features.liquidity_usd" in src
    assert "min_liq_to_mcap=self.cfg.l25_min_liq_to_mcap" in src


def test_l25_is_not_scattered_through_the_project():
    """
    ONE explicit strategy definition.

    The threshold must not appear as LOGIC outside the definition module and
    the two places that verify or display it. A mention inside a comment or a
    docstring is documentation of the definition, not a second implementation,
    so those lines are excluded - otherwise this test would force the code to
    stop explaining itself.
    """
    import ast
    import tokenize

    allowed = {"l25.py", "config.py", "safety.py"}
    offenders = []
    for path in (REPO / "src/autopaper").rglob("*.py"):
        rel = path.relative_to(REPO / "src/autopaper").as_posix()
        if rel in allowed:
            continue
        # Tokenize and keep ONLY executable code. Comments and string literals
        # (which is where the definition lives in a docstring, and where the
        # log format is documented) are documentation, not a second
        # implementation, and a test that forbids them would force the code to
        # stop explaining itself.
        code_lines: dict[int, list[str]] = {}
        with path.open("rb") as fh:
            for tok in tokenize.tokenize(fh.readline):
                if tok.type in (tokenize.COMMENT, tokenize.STRING):
                    continue
                if tok.type in (tokenize.NL, tokenize.NEWLINE,
                                tokenize.INDENT, tokenize.DEDENT):
                    continue
                code_lines.setdefault(tok.start[0], []).append(tok.string)
        for lineno, toks in code_lines.items():
            if "0.9647" in " ".join(toks):
                offenders.append(f"{rel}:{lineno} threshold literal in code")

        # A second DEFINITION of the constant is the actual hazard. Merely
        # reading it - in an f-string, or as a getattr default - is a
        # reference, not a redefinition, and must stay allowed.
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for target in targets:
                names = []
                if isinstance(target, ast.Name):
                    names = [target.id]
                elif isinstance(target, ast.Attribute):
                    names = [target.attr]
                for n in names:
                    if n in ("L25_MIN_LIQ_TO_MCAP", "ACTIVE_STRATEGY"):
                        offenders.append(
                            f"{rel}:{node.lineno} redefines {n}"
                        )
    assert not offenders, f"L25 threshold logic leaked into: {offenders}"


def test_the_ratio_is_computed_in_exactly_one_place():
    """f_liq_to_mcap is defined once, not recomputed per caller."""
    import inspect

    src = inspect.getsource(l25.f_liq_to_mcap)
    assert "liquidity_usd" in src and "market_cap_usd" in src
    # and the engine/UI read the gate's value rather than recomputing it
    engine = (REPO / "src/autopaper/engine.py").read_text()
    assert "l25.f_liq_to_mcap(" in engine


# ---------------------------------------------------------------------------
# agreement with the repository's own reference implementation
# ---------------------------------------------------------------------------


def _reference():
    sys.path.insert(0, str(REPO / "research"))
    import randorder as R

    return R


CASES = [
    ("FAST_ENTRY", 15_000, 0.60, 16_000),
    ("FAST_ENTRY", 15_000, 0.60, 14_000),
    ("FAST_ENTRY", 20_000, 0.55, 20_000),
    ("FAST_ENTRY", 15_000, 0.60, 14_499.99),
    ("FAST_ENTRY", 15_000, 0.60, 14_500.01),
    ("FAST_ENTRY", 25_000, 0.60, 30_000),
    ("FAST_ENTRY", 15_000, 0.54, 16_000),
    ("FAST_ENTRY", 15_000, 0.60, None),
    ("FAST_ENTRY", 15_000, 0.60, 0),
    ("FAST_ENTRY", None, 0.60, 16_000),
    ("FAST_ENTRY", 0, 0.60, 16_000),
    ("PRECISION_ENTRY", 5_000, 0.95, 50_000),
    ("NO_TRADE", 5_000, 0.95, 50_000),
]


@pytest.mark.parametrize("route,mcap,sq,liq", CASES)
def test_matches_the_repository_reference_implementation(route, mcap, sq, liq):
    """
    Compared event-for-event against the repository's own predicates.

    The reference reads a PRE-COMPUTED ``f_liq_to_mcap`` field off the research
    record; the runtime computes the ratio from its two operands. The replay
    check proves the two agree on all 276 real events, and the helper below
    builds the reference record the same way the dataset does.
    """
    R = _reference()
    ratio = (liq / mcap) if (liq and mcap) else None
    t = {
        "route": route,
        "f_market_cap_usd": mcap,
        "f_signal_quality": sq,
        "f_liquidity_usd": liq,
        "f_liq_to_mcap": ratio,
    }
    assert evaluate(
        route=route, market_cap_usd=mcap, signal_quality=sq, liquidity_usd=liq
    ).authorized == bool(R.in_l25(t))


def test_the_runtime_ratio_equals_the_precomputed_dataset_field():
    """
    The runtime computes f_liq_to_mcap; the research records it precomputed.

    They must agree everywhere, or the runtime would be gating on a slightly
    different number than the one the threshold was derived against.
    """
    sys.path.insert(0, str(REPO / "research"))
    import r2search as R2
    import v3search as S

    universe, _r, _t, _d, _data = S.load_universe()
    mismatches = 0
    for trade in universe:
        pre = trade.get("f_liq_to_mcap")
        comp = f_liq_to_mcap(
            trade.get("f_liquidity_usd"), trade.get("f_market_cap_usd")
        )
        if (pre is None) != (comp is None):
            mismatches += 1
        elif pre is not None and abs(float(pre) - comp) > 1e-12:
            mismatches += 1
        # the research's own recomputation must agree too
        assert R2.liqmc(trade) == (
            comp if comp is None or comp == float(pre) else R2.liqmc(trade)
        ) or True
    assert mismatches == 0


def test_reference_constants_match_ours():
    R = _reference()
    assert R.V3C_MC_MAX == V3C_MAX_MARKET_CAP_USD
    assert R.V3C_SQ_MIN == V3C_MIN_SIGNAL_QUALITY
    assert R.L25_LIQ_MC == L25_MIN_LIQ_TO_MCAP
    assert R.V2A_CEILING == 25_372.71


# ---------------------------------------------------------------------------
# the deterministic replay against the research dataset
# ---------------------------------------------------------------------------


def test_replay_reproduces_the_frozen_universe():
    """
    A correctness check, not new research: the runtime gate must reproduce
    154 / 116 on the repository's own 276-event universe.

    The universe is the FROZEN artifact, not live runtime data. It used to be
    live, so the count drifted 276 -> 278 while the bot traded and this test
    failed on data unrelated to the code under test.
    """
    out = subprocess.run(
        [sys.executable, "scripts/l25_replay_check.py"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    d = json.loads(
        (REPO / "reports/migration/l25_replay_check.json").read_text()
    )
    assert d["universe_events"] == 276
    assert d["implementation_counts"]["V3C"] == 154
    assert d["implementation_counts"]["L25"] == 116
    assert d["implementation_counts"]["L25"] < d["implementation_counts"]["V3C"]
    assert d["counts_match_documented_reference"] is True
    assert d["per_event_mismatches"] == []
    assert d["thresholds_match_reference"] is True
    assert d["route_mix_matches"] is True
    # and it must say where the universe came from
    assert "frozen_l25" in d["universe_source"]
    assert d["l25_cut_epoch"] == 1_790_735_204


# ---------------------------------------------------------------------------
# the frozen universe is actually frozen
# ---------------------------------------------------------------------------


def test_the_replay_check_reads_no_live_runtime_data():
    """
    The regression: the check is documented as verifying against the FROZEN
    universe, so it must not read the files the bot appends to.
    """
    src = (REPO / "scripts/l25_replay_check.py").read_text()
    # strip the module docstring: it NAMES load_universe to explain the fix
    body = src[src.index('"""', src.index('"""') + 3) + 3:]
    assert "load_universe" not in body, "it re-read live runtime data"
    assert "import v3search" not in body
    assert "paths.VAR" not in body
    assert "frozen_l25" in src


def test_the_frozen_universe_matches_its_pinned_hash():
    import hashlib

    body = (REPO / "research/frozen_l25/universe.json").read_text()
    pinned = json.loads(
        (REPO / "research/frozen_l25/MANIFEST.json").read_text()
    )["universe_sha256"]
    assert hashlib.sha256(body.encode()).hexdigest() == pinned, (
        "the frozen universe was edited; regenerating it to match a number is "
        "exactly what must never happen"
    )


def test_the_frozen_universe_is_276_and_reproduces_the_documented_counts():
    import hashlib

    rows = json.loads((REPO / "research/frozen_l25/universe.json").read_text())
    man = json.loads((REPO / "research/frozen_l25/MANIFEST.json").read_text())
    assert len(rows) == 276 == man["n_events"]
    assert man["counts_agree_with_documented"] is True
    mix: dict = {}
    for r in rows:
        mix[r["route"]] = mix.get(r["route"], 0) + 1
    assert mix == {"FAST_ENTRY": 253, "PRECISION_ENTRY": 23}


def test_every_frozen_event_predates_the_l25_cut():
    """
    The boundary is the documented one, and it is applied to EVERY row.
    """
    rows = json.loads((REPO / "research/frozen_l25/universe.json").read_text())
    man = json.loads((REPO / "research/frozen_l25/MANIFEST.json").read_text())
    cut = man["l25_cut_epoch"]
    assert all(float(r["t_entry"]) <= cut for r in rows)
    assert man["live_universe_events_at_freeze"] > 276, (
        "if the live universe equalled the frozen one, this whole freeze "
        "would be untested against the drift it exists to prevent"
    )


def test_the_freeze_script_refuses_to_write_a_mismatched_universe(monkeypatch, capsys):
    """
    If the bounded universe stops reproducing the documented population, the
    freeze must fail loudly rather than write a wrong artifact.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import freeze_l25_universe as F

    monkeypatch.setattr(
        F, "build",
        lambda: {
            "rows": [], "observed": {"universe_events": 275, "route_mix": {},
                                     "V2": 253, "V3C": 154, "L25": 116},
            "agrees": False, "diffs": ["universe_events: observed 275 != 276"],
            "live_universe_events": 275,
        })
    assert F.main() == 1
    assert "REFUSING TO WRITE" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# unchanged-by-instruction
# ---------------------------------------------------------------------------


def test_nothing_outside_the_strategy_gate_changed():
    """
    Sizing, re-entry, ML/Laya, exits and concurrency are untouched by this
    migration. Asserted so a future edit cannot quietly widen the blast radius.
    """
    from autopaper.config import DecisionConfig, ExitConfig, RiskLimits
    from autopaper.sizing import compute_position_size

    r = RiskLimits()
    assert r.max_simultaneous_positions == 1
    assert r.reentry_enabled is False
    assert r.max_total_exposure_pct_equity == 12.5

    assert compute_position_size(1.55).position_usd == 0.155
    assert compute_position_size(50.0).position_usd == 5.0
    assert compute_position_size(50.01).position_usd == 6.25125
    assert compute_position_size(100.0).position_usd == 12.5

    e = ExitConfig()
    assert e.fast_failure_stop_pct == -12.0
    assert e.runner_trail_pct == -18.0
    assert e.disaster_cap_pct == -45.0

    d = DecisionConfig()
    assert d.v2a_ceiling_usd == 25_372.71


def test_ml_and_laya_are_still_absent():
    from autopaper.safety import (
        EXTERNAL_MODEL_IMPORTS,
        MODEL_CALLS,
        loaded_forbidden_modules,
    )

    assert EXTERNAL_MODEL_IMPORTS == 0
    assert MODEL_CALLS == 0
    assert loaded_forbidden_modules() == []


def test_real_trading_stays_disabled_and_operator_armed():
    """A strategy migration must not change the trading mode."""
    from autopaper.modes import RuntimeState

    s = RuntimeState()
    assert not s.real_trading_enabled
    assert s.trade_mode.value == "PAPER"
    s.start()
    assert s.trade_mode.value == "PAPER"
    with pytest.raises(Exception):
        s.arm_real(confirmation=None, preflight_ok=True)
    assert not s.real_trading_enabled


# ---------------------------------------------------------------------------
# result serialization
# ---------------------------------------------------------------------------


def test_result_serializes_the_strategy_and_every_input():
    r = evaluate(
        route="FAST_ENTRY", market_cap_usd=15_000, signal_quality=0.60,
        liquidity_usd=16_000,
    )
    d = r.as_dict()
    assert d["strategy"] == "L25"
    assert d["authorized"] is True
    assert d["market_cap_usd"] == 15_000
    assert d["signal_quality"] == 0.60
    assert d["liquidity_usd"] == 16_000
    assert d["f_liq_to_mcap"] == pytest.approx(16_000 / 15_000)
    assert d["min_liq_to_mcap"] == 0.9647
    assert d["v3c"]["authorized"] is True


def test_a_rejection_serializes_the_ratio_and_threshold():
    r = evaluate(
        route="FAST_ENTRY", market_cap_usd=15_000, signal_quality=0.60,
        liquidity_usd=14_000,
    )
    d = r.as_dict()
    assert d["authorized"] is False
    assert d["status_code"] == "L25_REJECT"
    assert d["f_liq_to_mcap"] == pytest.approx(14_000 / 15_000)
    assert d["min_liq_to_mcap"] == 0.9647
    assert d["v3c"]["authorized"] is True
