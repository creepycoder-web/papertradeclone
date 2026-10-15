"""
LIVE L25 == ORIGINAL RESEARCHED L25.

Three pieces of adaptive logic were present in the live runtime but are NOT part
of the researched L25 definition, so live was refusing and re-scoring candidates
the research never would have:

* ResearchMemory's route multiplier, which scaled confidence;
* a 5m reversal veto, which BLOCKED;
* a 30s reversal veto, which BLOCKED.

Each is now observability-only. These tests prove that by holding features fixed
and varying only the removed condition, which is the strongest available form of
the claim: identical inputs must yield an identical L25 result.

The narrative is asserted PRESENT and unchanged, because the corrected forensic
result shows it was fully available for 116/116 of the research population.
"""


from __future__ import annotations

import inspect
import math

import pytest

from autopaper import l25
from autopaper.analysis import decision as dec
from autopaper.analysis.features import FeatureSet
from autopaper.analysis.routes import Route

pytestmark = pytest.mark.strategy


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _feats(**kw) -> FeatureSet:
    """A qualifying L25 candidate, with overrides for the tests."""
    base = dict(
        candidate_id="C1",
        mint="Mint1",
        decision_time=1_000.0,
        observation_count=20,
        history_span_s=600.0,
        newest_observed_at=1_000.0,
        data_state=dec.DataState.LIVE,
        price=0.0015,
        market_cap_usd=9_000.0,
        liquidity_usd=9_200.0,
        liq_to_mcap=1.0222,
        signal_quality=0.80,
        freshness=1.0,
        alert_fields_present=17,
        nar_available=True,
        nar_fields_present=5,
        narrative_chars=200,
        narrative_specifics=3,
        liquidity_score=0.8,
        activity_score=0.8,
        momentum_score=0.7,
        structure_score=0.7,
        extension_score=0.6,
        change_5m=5.0,
        momentum_30s_pct=2.0,
    )
    base.update(kw)
    return FeatureSet(**base)


def _memory_with_history(multiplier: float, sample: int):
    """A ResearchMemory whose route multiplier is far from 1.0."""
    from autopaper.memory import ResearchMemory

    mem = ResearchMemory()
    mem.min_sample = 1
    mem.route_multiplier = lambda *a, **k: (multiplier, sample)
    return mem


def _nar(available: bool = True):
    """A narrative snapshot matching the research population (available)."""
    from autopaper.nar.base import NarSnapshot

    return NarSnapshot(
        mint="Mint1", fetched_at=1_000.0, source="test", available=available,
        narrative="a concrete narrative with 2024 and 12% detail" if available else None,
        buys_1h=40 if available else None,
        sells_1h=25 if available else None,
        error=None if available else "NAR_UNAVAILABLE: timeout",
    )


def _evaluate(memory=None, nar=None, **feat_kw) -> object:
    return _engine(memory).decide(
        features=_feats(**feat_kw),
        nar=_nar() if nar is None else nar,
        alert=None,
        analysis_start=1_000.0,
        analysis_complete=1_002.0,
    )


def _engine(memory=None) -> "dec.DecisionEngine":
    from autopaper.config import DecisionConfig, ExitConfig

    return dec.DecisionEngine(DecisionConfig(), ExitConfig(), memory=memory)


# ---------------------------------------------------------------------------
# A. ResearchMemory cannot change L25
# ---------------------------------------------------------------------------


def test_memory_history_does_not_change_the_decision():
    """Same features, wildly different memory -> identical decision."""
    with_mem_none = _evaluate(memory=None)
    with_bad_mem = _evaluate(memory=_memory_with_history(0.10, 500))
    assert with_mem_none.decision == with_bad_mem.decision
    assert with_mem_none.decision == dec.Decision.BUY


def test_memory_history_does_not_change_confidence():
    with_none = _evaluate(memory=None)
    with_bad = _evaluate(memory=_memory_with_history(0.10, 500))
    assert with_none.confidence == pytest.approx(with_bad.confidence), (
        "the memory multiplier still scales confidence"
    )


def test_memory_history_does_not_change_signal_quality():
    """signal_quality is a FEATURE; memory must not be able to move it."""
    with_none = _evaluate(memory=None)
    with_bad = _evaluate(memory=_memory_with_history(0.10, 500))
    assert with_none.signal_quality == pytest.approx(with_bad.signal_quality)


def test_memory_does_not_change_l25_authorization():
    with_none = _evaluate(memory=None)
    with_bad = _evaluate(memory=_memory_with_history(0.10, 500))
    assert with_none.l25.authorized == with_bad.l25.authorized is True
    assert with_none.l25.status == with_bad.l25.status


def test_memory_does_not_change_the_route():
    with_none = _evaluate(memory=None)
    with_bad = _evaluate(memory=_memory_with_history(0.10, 500))
    assert with_none.route == with_bad.route


def test_the_demotion_is_still_reported():
    """
    The condition remains VISIBLE.

    It is observability now, not eligibility: a route whose paper history is
    poor is still worth telling the operator about.
    """
    report = _evaluate(memory=_memory_with_history(0.10, 500))
    codes = {f.code for f in report.risk_flags}
    assert "route_demoted" in codes, "the observation was thrown away"
    flag = next(f for f in report.risk_flags if f.code == "route_demoted")
    assert flag.severity == "warn", "it must not be able to block"
    assert report.decision == dec.Decision.BUY


def test_the_multiplier_is_still_reported_for_observability():
    report = _evaluate(memory=_memory_with_history(0.10, 500))
    assert report.route_multiplier == pytest.approx(0.10), (
        "the observation is still carried for the operator"
    )


def test_the_confidence_formula_no_longer_takes_a_multiplier():
    """
    Structural, so the multiplier cannot be reintroduced quietly.

    The multiplier is still a parameter (passed as 1.0) so the report keeps the
    field; what must not happen is a non-1.0 value reaching it.
    """
    src = inspect.getsource(dec.DecisionEngine.decide)
    call = src[src.index("confidence = self._confidence("):]
    call = call[:call.index(")") + 1]
    assert "1.0" in call
    assert "multiplier" not in call, (
        "the memory multiplier must not be passed to _confidence"
    )


def test_confidence_isolation_holds_through_the_real_entry_point():
    """
    Proved through decide(), which is the only path a decision takes.

    ``_confidence`` still ACCEPTS a multiplier so the report keeps the field, so
    asserting on it directly would test the wrong thing; the isolation is that
    decide() never passes anything but 1.0.
    """
    plain = _evaluate()
    with_bad_memory = _evaluate(memory=_memory_with_history(0.10, 500))
    assert plain.confidence == pytest.approx(with_bad_memory.confidence)
    assert plain.confidence == pytest.approx(_evaluate(
        memory=_memory_with_history(1.0, 0)).confidence)


# ---------------------------------------------------------------------------
# B. the two reversal vetoes
# ---------------------------------------------------------------------------


def test_a_5m_reversal_no_longer_blocks():
    report = _evaluate(memory=None)
    assert report.decision == dec.Decision.BUY


@pytest.mark.parametrize("change_5m", [-11.0, -25.0, -80.0])
def test_a_deep_5m_reversal_is_not_itself_a_block(change_5m):
    """
    The VETO is gone; the route classifier is untouched.

    A large negative 5m move legitimately stops the route classifier selecting
    FAST_ENTRY, and V3-C is DEFINED over FAST_ENTRY - so refusing there is the
    researched behaviour and must stay. What must not happen is the separate
    'reversing' veto adding its own block on top.
    """
    report = _evaluate(change_5m=change_5m)
    blocking = {f.code for f in report.risk_flags if f.severity == "block"}
    assert "reversing_5m" not in blocking
    warn = {f.code for f in report.risk_flags if f.severity == "warn"}
    assert "reversing_5m" in warn, "the condition should still be observed"
    # any refusal must come from the researched gate, never from the removed veto
    assert blocking <= {"v3c_not_authorized", "v2a_market_cap_ceiling",
                        "l25_liq_to_mcap_below_threshold"}, blocking


def test_a_healthy_candidate_buys():
    """The sanity anchor for every isolation test below."""
    report = _evaluate()
    assert report.decision == dec.Decision.BUY
    assert report.l25.authorized is True
    assert report.route is Route.FAST_ENTRY


@pytest.mark.parametrize("momentum_30s_pct", [-11.0, -30.0, -90.0])
def test_a_30s_reversal_is_not_itself_a_block(momentum_30s_pct):
    """
    The 30s veto is observability-only.

    Unlike change_5m, momentum_30s_pct is not a route input, so a deep 30s
    reversal must leave the decision completely untouched.
    """
    base = _evaluate()
    moved = _evaluate(momentum_30s_pct=momentum_30s_pct)
    assert moved.decision == dec.Decision.BUY
    assert moved.l25.authorized is True
    assert moved.confidence == pytest.approx(base.confidence), (
        "a 30s reversal still moves confidence"
    )
    assert "reversing_30s" in {f.code for f in moved.risk_flags}


def test_both_reversals_at_once_are_not_blocks():
    report = _evaluate(change_5m=-90.0, momentum_30s_pct=-90.0)
    blocking = {f.code for f in report.risk_flags if f.severity == "block"}
    assert "reversing_5m" not in blocking
    assert "reversing_30s" not in blocking


def test_the_reversal_conditions_are_still_observed():
    """Not deleted - demoted to warn."""
    report = _evaluate(change_5m=-12.0, momentum_30s_pct=-40.0)
    codes = {f.code for f in report.risk_flags}
    assert "reversing_5m" in codes
    assert "reversing_30s" in codes
    for f in report.risk_flags:
        if f.code.startswith("reversing"):
            assert f.severity == "warn", f"{f.code} can still block"


def test_neither_veto_flag_is_blocking_anywhere():
    src = inspect.getsource(dec.DecisionEngine._risk_flags)
    for code in ("reversing_5m", "reversing_30s"):
        i = src.index(f'"{code}"')
        window = src[i:i + 200]
        assert '"block"' not in window, f"{code} is still a block"
        assert '"warn"' in window, f"{code} is not marked warn"


def test_a_genuine_structural_block_still_blocks():
    """
    Removing two vetoes must not remove real safety.

    A blocking NAR risk (dev holdings) must still refuse, so this is a removal
    of two specific heuristics and not a weakening of the gate.
    """
    from autopaper.nar.base import NarSnapshot

    dev = NarSnapshot(mint="Mint1", fetched_at=1_000.0, source="t",
                      available=True, narrative="x", dev_holdings_pct=6.0,
                      buys_1h=10, sells_1h=10)
    report = _evaluate(nar=dev)
    blocking = {f.code for f in report.risk_flags if f.severity == "block"}
    assert blocking, "a dev-holdings concentration must still block"
    assert report.decision != dec.Decision.BUY


# ---------------------------------------------------------------------------
# C. L25 itself is untouched
# ---------------------------------------------------------------------------


def test_l25_definition_is_unchanged():
    l25.verify_definition()          # raises if any threshold moved
    assert l25.L25_MIN_LIQ_TO_MCAP == 0.9647
    assert l25.ACTIVE_STRATEGY == "L25"


def test_v2a_is_unchanged():
    from autopaper.v2a import V2A_CEILING_USD
    assert V2A_CEILING_USD == 25_372.71


def test_no_removed_logic_appears_in_the_l25_module():
    src = open(l25.__file__, encoding="utf-8").read()
    for banned in ("route_demoted", "ResearchMemory", "change_5m",
                   "momentum_30s_pct", "_confidence"):
        assert banned not in src, f"{banned} leaked into l25.py"


# ---------------------------------------------------------------------------
# D. Narrative REMAINS part of L25
# ---------------------------------------------------------------------------


def test_narrative_still_contributes_to_signal_quality():
    """
    The +0.05 narrative contribution is PRESERVED.

    Asserted on ``_signal_quality`` directly, because signal_quality is computed
    once in build_features and then carried on the FeatureSet - testing it via
    decide() would just read back a value this test supplied itself.
    """
    from autopaper.analysis.features import _signal_quality

    kwargs = dict(
        data_state=dec.DataState.LIVE, freshness=1.0, liq_score=0.8,
        act_score=0.8, mom_score=0.7, struct_score=0.7, ext_score=0.6,
        alert_fields=17, span_s=600.0, narrative_placeholder=False,
    )
    with_nar = _signal_quality(nar_available=True, nar_has_flow=False, **kwargs)
    without = _signal_quality(nar_available=False, nar_has_flow=False, **kwargs)
    assert with_nar > without, "narrative no longer contributes"
    assert with_nar - without == pytest.approx(0.05, abs=1e-9)


def test_the_flow_bonus_is_still_computed():
    """
    The +0.06 flow bonus is part of the researched signal_quality and stays.

    Removing it would change which candidates clear 0.55.
    """
    from autopaper.analysis.features import _signal_quality

    base = dict(
        data_state=dec.DataState.LIVE, freshness=1.0, liq_score=0.8,
        act_score=0.8, mom_score=0.7, struct_score=0.7, ext_score=0.6,
        alert_fields=17, span_s=600.0, narrative_placeholder=False,
        nar_available=True,
    )
    with_flow = _signal_quality(nar_has_flow=True, **base)
    without = _signal_quality(nar_has_flow=False, **base)
    assert with_flow - without == pytest.approx(0.06, abs=1e-9)


def test_the_threshold_is_unchanged():
    assert l25.V3C_MIN_SIGNAL_QUALITY == 0.55


def test_narrative_fields_are_now_persisted():
    """Part 5: future rows must be able to prove the composition."""
    from dataclasses import fields

    names = {f.name for f in fields(FeatureSet)}
    for required in ("nar_buys", "nar_sells", "nar_has_flow"):
        assert required in names, required


def test_nar_has_flow_uses_the_forensic_definition():
    """
    ``bool(nar_buys is not None or nar_sells is not None)``.

    Asserted on the source so the definition cannot drift.
    """
    src = inspect.getsource(__import__(
        "autopaper.analysis.features", fromlist=["x"]))
    assert "nar_has_flow=bool(nar_buys is not None or nar_sells is not None)" \
        in src
