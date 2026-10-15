"""
THE EFFECTIVE-ENTRY-GATE CONTRACT -- V2 FLOOR (BETA).

THE DEFECT THIS SUITE EXISTS FOR
--------------------------------
``state.strategy_profile`` and ``state.selected_arms`` are two axes written by
two different pieces of code. ``RuntimeState.select_strategies`` writes the
arms; the Telegram profile callback never writes the profile in PAPER. So
selecting ``V2_FLOOR_BETA`` left the profile on its shipped default,
``L25_CURRENT``.

The decision layer took its authorizing gate from the PROFILE alone, so with
the beta experiment selected it applied **L25**:

* ``decision.py`` raised a blocking ``v3c_*`` flag when L25 refused, and
  ``engine.on_alert`` returns on ``Decision.REJECT`` -- so a candidate that
  satisfied V2 FLOOR but failed L25 never reached
  ``BetaExperiment.process_signal`` at all;
* ``_decide`` then re-checked ``l25.authorized`` specifically;
* ``_confidence`` collapsed to 0.05 on any blocking flag, below the buy bar.

The experiment could therefore only ever trade the subset of candidates L25
already liked. That is not the experiment that was selected.

THE CONTRACT
------------
Each selected arm contributes ITS OWN entry gate and the union is operative
(``paper_live.arms.effective_entry_gates``). A gate refusal is that strategy's
verdict and cannot veto a candidate another operative gate authorized; a
STRUCTURAL refusal (absent data, refused route, account limit) refuses for
everyone. Each arm is then authorized downstream by its own gate's result on
the same report.

WHAT IS PINNED HERE
-------------------
1. An explicitly selected beta experiment authorizes through V2 FLOOR.
2. A beta-eligible signal rejected ONLY by L25 still reaches A/B/C.
3. A beta-ineligible signal opens nothing and raises ONE grouped rejection.
4. L25 selected on its own behaves exactly as before.
5. Neither strategy's refusal suppresses the other's eligible entry.
6. No default, empty or legacy selection can activate beta.
7. UI / persisted state / safety audit agree on the effective gates.
8. The three arms share one entry and keep independent exits and ledgers.
9. No beta path reaches live execution.

Every V2 FLOOR threshold asserted here is a REGRESSION LOCK. If one of these
fails because a threshold moved, the fix is to restore the threshold.

OFFLINE: every collaborator is a double. No network, no Telegram, no RPC,
nothing signed or broadcast. PAPER only.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from autopaper import v2floor
from autopaper.analysis import DecisionEngine, build_features
from autopaper.config import DecisionConfig, ExitConfig
from autopaper.modes import RuntimeState, TradeMode, validate_strategy_selection
from autopaper.models import Decision, ObservationSet
from autopaper.nar import NullNarSource
from autopaper.paper_live.arms import (
    ARMS,
    BETA_ARMS,
    BETA_GROUP,
    DEFAULT_ARM,
    GATE_L25,
    GATE_V2_FLOOR,
    ResearchArm,
    entry_gate_for_arm,
    effective_entry_gates,
    parse_arms,
)
from autopaper.paper_live.engine import PaperLiveEngine
from autopaper.paper_live.fills import FillModel
from autopaper.paper_live.quotes import SolReference
from autopaper.paper_live.schema import FailureReason
from autopaper.safety import audit as safety_audit
from autopaper.strategy_profile import DEFAULT_PROFILE, StrategyProfile
from autopaper.telegram import ui

from conftest import MINT as CONFTEST_MINT
from conftest import alert as conf_alert
from conftest import obs as conf_obs
from test_paper_live import (
    FakeMarket,
    FakeQuoteService,
    SOL_USD,
    StubDecision,
    StubExit,
    signal,
)

pytestmark = [
    pytest.mark.paper_live,
    pytest.mark.strategy,
    pytest.mark.safety,
]

T0 = 1_700_000_000.0
MARK_PRICE = 1.0

#: V2 FLOOR ELIGIBLE / L25 INELIGIBLE.
#:   vol5/mcap = 60,000/50,000 = 1.20 >= 1.0   -> V2 FLOOR authorizes
#:   change_5m = +12% inside [+10%, +300%]     -> V2 FLOOR momentum passes
#:   market cap $50,000 > V3-C ceiling $20,000 -> L25/V3-C REFUSES
V2_OK_L25_NO = dict(market_cap_usd=50_000.0, volume_5m_usd=60_000.0,
                    change_5m=12.0)

#: V2 FLOOR INELIGIBLE / L25 ELIGIBLE.
#:   vol5/mcap = 7,269/13,878 = 0.5238 < 1.0   -> V2 FLOOR REFUSES
#:   market cap $13,878 <= $20,000             -> L25 authorizes
V2_NO_L25_OK = dict(market_cap_usd=13_878.0, volume_5m_usd=7_269.0,
                    change_5m=12.0)

#: Both accept.
BOTH_OK = dict(market_cap_usd=15_000.0, volume_5m_usd=40_000.0,
               change_5m=12.0)

BETA_ARM_NAMES = tuple(a.value for a in BETA_ARMS)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def gated_decision(*, selected_arms=None, profile=DEFAULT_PROFILE.value,
                   market_cap_usd=15_000.0, volume_5m_usd=40_000.0,
                   change_5m=12.0, apply_contract=True):
    """A real ``DecisionReport`` for one candidate under one selection.

    Drives the production ``build_features`` over a real three-print series so
    the route classifier, the regime classifier and both strategy gates all see
    a coherent candidate -- then installs the operative gate set derived from
    ``selected_arms`` exactly as ``PaperTrader`` does.
    """
    engine = DecisionEngine(
        DecisionConfig(strategy_profile=profile), ExitConfig())
    if apply_contract and selected_arms is not None:
        engine.set_entry_gates(
            effective_entry_gates(selected_arms, profile=profile))
    step = 15.0
    prices = [1.0e-4, 1.02e-4, 1.05e-4]
    history = ObservationSet([
        conf_obs(T0 - step * (len(prices) - 1 - i), p,
                 market_cap_usd=market_cap_usd, liquidity_usd=250_000.0,
                 volume_5m_usd=volume_5m_usd, price_change_5m=change_5m,
                 first_valid_price=1.0e-4)
        for i, p in enumerate(prices)
    ])
    nar = NullNarSource().fetch(CONFTEST_MINT)
    snap = conf_alert(T0, price_usd=prices[-1])
    features = build_features(
        candidate_id="c_gate", mint=CONFTEST_MINT, decision_time=T0,
        history=history, alert=snap, nar=nar,
        stale_after_ms=15_000.0, unusable_after_ms=45_000.0)
    report = engine.decide(features=features, alert=snap, nar=nar,
                           analysis_start=T0, analysis_complete=T0 + 0.05)
    return report, engine


def beta_engine(tmp_path, arms=None) -> PaperLiveEngine:
    """The PAPER engine on ``arms``, durable state pinned under ``tmp_path``.

    ``state_root`` MUST be explicit: ``PaperStateStore`` falls back to
    ``Path("var")`` relative to the CWD when it is not, which would write the
    operator's live beta state from a test.
    """
    return PaperLiveEngine(
        arms=parse_arms(arms if arms is not None else [BETA_GROUP]),
        decision_engine=StubDecision(),
        exit_policy=StubExit(),
        quote_service=FakeQuoteService(),
        market=FakeMarket(price=MARK_PRICE),
        token_decimals_for=lambda mint: 6,
        sol_reference=SolReference(SOL_USD, 0.0, "test"),
        fill_model=FillModel(),
        starting_bankroll_usd=5.0,
        require_identity=True,
        state_root=tmp_path,
    )


def beta_signal(**gate_inputs):
    """A snapshot carrying the decision-time V2 FLOOR gate inputs."""
    return replace(signal(mint=CONFTEST_MINT, telegram_timestamp=T0,
                          message_id=11), **gate_inputs)


def run(coro):
    import asyncio
    return asyncio.run(coro)


# ===========================================================================
# 0. the contract itself
# ===========================================================================


class TestTheContract:
    def test_each_arm_declares_its_own_gate(self):
        assert entry_gate_for_arm(ARMS[ResearchArm.L25_CURRENT]) == GATE_L25
        for arm in BETA_ARMS:
            assert entry_gate_for_arm(ARMS[arm]) == GATE_V2_FLOOR

    def test_the_gate_set_follows_the_selection(self):
        assert effective_entry_gates(["L25_CURRENT"]) == ("L25",)
        assert effective_entry_gates([BETA_GROUP]) == ("V2_FLOOR",)
        assert effective_entry_gates([a.value for a in BETA_ARMS]) == \
            ("V2_FLOOR",)
        assert effective_entry_gates(
            ["L25_CURRENT", BETA_GROUP]) == ("L25", "V2_FLOOR")

    def test_an_explicit_non_default_profile_still_contributes_its_gate(self):
        """
        ``V2_FLOOR`` is a PROFILE and has no research ARM, so without this an
        operator who explicitly configured it would be silently downgraded to
        L25 by their own arm selection.
        """
        assert effective_entry_gates(
            ["L25_CURRENT"], profile="V2_FLOOR") == ("L25", "V2_FLOOR")
        assert effective_entry_gates(
            [BETA_GROUP], profile="V2_FLOOR_BETA") == ("V2_FLOOR",)

    def test_the_default_profile_is_not_an_instruction(self):
        """
        The shipped default is what an operator who chose nothing is left
        holding. If it contributed, L25 would sit in front of the beta
        experiment again and the veto would return.
        """
        assert effective_entry_gates(
            [BETA_GROUP], profile=DEFAULT_PROFILE.value) == ("V2_FLOOR",)
        assert effective_entry_gates([BETA_GROUP], profile=None) == ("V2_FLOOR",)

    def test_no_gate_set_is_ever_empty(self):
        for selection in ([], ["NOT_AN_ARM"], [BETA_GROUP], ["L25_CURRENT"]):
            assert effective_entry_gates(selection), selection

    def test_setting_gates_fails_closed(self):
        engine = DecisionEngine(DecisionConfig(), ExitConfig())
        for bad in ((), [], None, ("",), ("NOT_A_GATE",), (None,)):
            got = engine.set_entry_gates(bad)
            assert got, f"{bad!r} must not leave the engine with no gate"

    def test_the_order_is_stable(self):
        assert list(effective_entry_gates(
            ["L25_CURRENT", BETA_GROUP])) == \
            list(effective_entry_gates([BETA_GROUP, "L25_CURRENT"]))


# ===========================================================================
# 1. an explicitly selected beta experiment uses the V2 FLOOR path
# ===========================================================================


class TestBetaUsesItsOwnGate:
    def test_selecting_beta_authorizes_through_v2_floor_alone(self):
        report, engine = gated_decision(
            selected_arms=[BETA_GROUP], **V2_OK_L25_NO)
        assert engine.entry_gates == ("V2_FLOOR",)
        assert report.decision is Decision.BUY, (
            "with the beta experiment selected, a V2 FLOOR-eligible candidate "
            "must be BUY even though L25 refuses it")
        assert report.v2floor.authorized is True
        assert report.l25.authorized is False, (
            "the fixture must actually fail L25, or this proves nothing")

    def test_the_l25_refusal_is_recorded_but_not_blocking(self):
        report, _ = gated_decision(selected_arms=[BETA_GROUP],
                                   **V2_OK_L25_NO)
        codes = {f.code: f.severity for f in report.risk_flags}
        assert codes.get("v3c_market_cap_over_ceiling") == "info", (
            "the L25 refusal must stay journaled at info -- it is the "
            "counterfactual a control A/B needs -- without being authoritative")

    def test_the_end_to_end_path_reaches_the_experiment(self, tmp_path):
        """
        THE INVERSE-GATE PROOF, at the layer that owns the experiment.

        ``on_signal`` -> ``BetaExperiment.process_signal`` is where a
        V2 FLOOR-eligible candidate must be able to arrive, and the beta gate
        is the only thing that can refuse it there.
        """
        eng = beta_engine(tmp_path, [BETA_GROUP])
        records = run(eng.on_signal(beta_signal(
            v2_volume_5m_usd=60_000.0, v2_market_cap_usd=50_000.0,
            v2_change_5m_pct=12.0)))
        assert len(records) == 3
        assert all(r.traded for r in records)
        assert eng.beta_experiment.positions, (
            "a V2 FLOOR-eligible signal opened nothing: the candidate never "
            "reached the experiment")
        # exactly one shared parent entry across the three arms
        assert len(eng.beta_experiment.positions) == 1
        assert set(next(iter(eng.beta_experiment.positions.values()))) == \
            {"A", "B", "C"}


# ===========================================================================
# 2. a beta-ineligible signal opens nothing, and says so once
# ===========================================================================


class TestBetaIneligibleSignal:
    def test_no_beta_position_is_opened(self, tmp_path):
        eng = beta_engine(tmp_path, [BETA_GROUP])
        records = run(eng.on_signal(beta_signal(
            v2_volume_5m_usd=7_269.0, v2_market_cap_usd=13_878.0,
            v2_change_5m_pct=12.0)))
        assert all(r.traded is False for r in records)
        assert all(r.failure is FailureReason.V2_FLOOR_REJECTED
                   for r in records)
        assert eng.beta_experiment.positions == {}
        assert eng.open_positions() == []

    def test_the_decision_layer_refuses_and_names_the_gate(self):
        report, _ = gated_decision(selected_arms=[BETA_GROUP], **V2_NO_L25_OK)
        assert report.decision is Decision.REJECT
        assert report.v2floor.authorized is False
        assert report.l25.authorized is True, (
            "the fixture must actually pass L25, or this proves nothing")
        blocking = [f.code for f in report.risk_flags if f.severity == "block"]
        assert "v2floor_vol5_per_mc_below_floor" in blocking

    def test_one_signal_raises_one_grouped_rejection(self):
        """
        The experiment writes one record per arm; the operator sees ONE
        message. Three identical alerts read as a looping malfunction.
        """
        rendered = ui.render_buy_blocked(
            symbol="X", mint=CONFTEST_MINT, reason="V2_FLOOR_REJECTED",
            detail="vol5/market_cap 0.5238 is below the floor", mode=TradeMode.PAPER,
            beta_arms=BETA_ARM_NAMES)
        assert rendered.count("BUY_BLOCKED=") == 1
        assert "V2 FLOOR (BETA)" in rendered
        assert "NO BETA POSITION OPENED" in rendered
        assert "L25 authorized an entry." not in rendered

    def test_the_thresholds_are_unchanged(self):
        assert v2floor.V2F_MIN_VOL5_PER_MC == 1.0
        assert v2floor.V2F_MIN_CHANGE_5M_PCT == 10.0
        assert v2floor.V2F_MAX_CHANGE_5M_PCT == 300.0
        below = v2floor.evaluate(volume_5m_usd=13_877.0, change_5m_pct=12.0,
                                 market_cap_usd=13_878.0)
        above = v2floor.evaluate(volume_5m_usd=13_879.0, change_5m_pct=12.0,
                                 market_cap_usd=13_878.0)
        assert below.authorized is False
        assert above.authorized is True


# ===========================================================================
# 3. L25 on its own is untouched
# ===========================================================================


class TestL25AloneIsUnchanged:
    def test_l25_alone_authorizes_through_l25_alone(self):
        report, engine = gated_decision(selected_arms=["L25_CURRENT"],
                                        **V2_NO_L25_OK)
        assert engine.entry_gates == ("L25",)
        assert report.decision is Decision.BUY
        assert report.l25.authorized is True

    def test_l25_alone_still_refuses_what_l25_refuses(self):
        report, _ = gated_decision(selected_arms=["L25_CURRENT"],
                                   **V2_OK_L25_NO)
        assert report.decision is Decision.REJECT
        assert report.l25.authorized is False

    def test_l25_alone_never_raises_the_v2_gate_as_blocking(self):
        report, _ = gated_decision(selected_arms=["L25_CURRENT"],
                                   **V2_NO_L25_OK)
        v2_flags = [f for f in report.risk_flags if f.code.startswith("v2floor_")]
        assert v2_flags, "the V2 FLOOR result must still be journaled"
        assert all(f.severity == "info" for f in v2_flags)
        assert any("not the selected gate" in f.detail for f in v2_flags)

    def test_the_decision_engine_without_a_contract_is_unchanged(self):
        """
        A caller that knows nothing about arms -- the research harness, the
        unit tests -- must get exactly the pre-existing profile behaviour.
        """
        report, engine = gated_decision(selected_arms=None,
                                        profile="L25_CURRENT",
                                        apply_contract=False,
                                        **V2_NO_L25_OK)
        assert engine.entry_gates == ("L25",)
        assert report.decision is Decision.BUY

    def test_an_l25_arm_still_trades_on_an_l25_eligible_signal(self, tmp_path):
        eng = beta_engine(tmp_path, ["L25_CURRENT"])
        records = run(eng.on_signal(beta_signal(
            v2_volume_5m_usd=7_269.0, v2_market_cap_usd=13_878.0,
            v2_change_5m_pct=12.0)))
        assert len(records) == 1
        assert records[0].arm == "L25_CURRENT"
        assert records[0].traded is True
        assert eng.beta_experiment is None


# ===========================================================================
# 4/5. neither strategy's refusal suppresses the other
# ===========================================================================


class TestPerStrategyAuthorizationIsIndependent:
    def test_both_directions_at_the_decision_layer(self):
        """
        THE INVERSE-GATE MATRIX.

        Rows are candidates, columns are selections. The cell is the decision.
        A strategy's refusal must never remove another strategy's authorization.
        """
        matrix = {
            "V2_ok/L25_no": (
                V2_OK_L25_NO,
                {"beta": Decision.BUY, "l25": Decision.REJECT,
                 "both": Decision.BUY}),
            "V2_no/L25_ok": (
                V2_NO_L25_OK,
                {"beta": Decision.REJECT, "l25": Decision.BUY,
                 "both": Decision.BUY}),
            "both_ok": (
                BOTH_OK,
                {"beta": Decision.BUY, "l25": Decision.BUY,
                 "both": Decision.BUY}),
        }
        selections = {
            "beta": [BETA_GROUP],
            "l25": ["L25_CURRENT"],
            "both": ["L25_CURRENT", BETA_GROUP],
        }
        for label, (features, expected) in matrix.items():
            for key, want in expected.items():
                report, _ = gated_decision(
                    selected_arms=selections[key], **features)
                assert report.decision is want, (
                    f"{label} with the {key} selection: expected {want.value}, "
                    f"got {report.decision.value}")

    def test_a_beta_refusal_does_not_suppress_an_l25_entry(self, tmp_path):
        eng = beta_engine(tmp_path, ["L25_CURRENT", BETA_GROUP])
        records = run(eng.on_signal(beta_signal(
            v2_volume_5m_usd=7_269.0, v2_market_cap_usd=13_878.0,
            v2_change_5m_pct=12.0)))
        by_arm = {r.arm: r for r in records}
        assert set(by_arm) == {"L25_CURRENT", *BETA_ARM_NAMES}
        assert by_arm["L25_CURRENT"].traded is True
        for arm in BETA_ARM_NAMES:
            assert by_arm[arm].traded is False
            assert by_arm[arm].failure is FailureReason.V2_FLOOR_REJECTED
        assert eng.accounts["L25_CURRENT"].position is not None
        assert eng.beta_experiment.positions == {}

    def test_an_l25_refusal_does_not_suppress_a_beta_entry(self, tmp_path):
        eng = beta_engine(tmp_path, ["L25_CURRENT", BETA_GROUP])
        records = run(eng.on_signal(beta_signal(
            v2_volume_5m_usd=60_000.0, v2_market_cap_usd=50_000.0,
            v2_change_5m_pct=12.0)))
        by_arm = {r.arm: r for r in records}
        # the beta arms traded on the V2 FLOOR gate
        assert all(by_arm[a].traded for a in BETA_ARM_NAMES)
        assert len(eng.beta_experiment.positions) == 1

    def test_a_mixed_selection_still_uses_each_arms_own_gate(self):
        """
        A BUY under a mixed selection means "some selected strategy
        authorized". Each arm must then be filtered by ITS OWN gate, or L25
        would open on the beta experiment's approval.
        """
        report, _ = gated_decision(
            selected_arms=["L25_CURRENT", BETA_GROUP], **V2_OK_L25_NO)
        assert report.decision is Decision.BUY
        assert report.v2floor.authorized is True
        assert report.l25.authorized is False

    def test_an_l25_arm_declines_its_own_gate_under_a_mixed_selection(
            self, tmp_path):
        """
        THE OTHER HALF OF PER-STRATEGY AUTHORIZATION.

        Under a mixed selection the decision is BUY because SOME selected
        strategy authorized. If every arm then opened on that, the L25 arm
        would trade a candidate L25 itself refused -- the beta experiment's
        approval leaking into a different strategy's book.
        """
        eng = beta_engine(tmp_path, ["L25_CURRENT", BETA_GROUP])
        report, _ = gated_decision(
            selected_arms=["L25_CURRENT", BETA_GROUP], **V2_OK_L25_NO)
        assert report.decision is Decision.BUY
        assert report.l25.authorized is False

        # give the bridge a decision the arm can be checked against
        eng.decide = lambda **kw: (report, "BUY", True)
        records = run(eng.on_signal(beta_signal(
            v2_volume_5m_usd=60_000.0, v2_market_cap_usd=50_000.0,
            v2_change_5m_pct=12.0)))
        by_arm = {r.arm: r for r in records}
        # the beta arms traded on the V2 FLOOR gate
        assert all(by_arm[a].traded for a in BETA_ARM_NAMES)
        # the L25 arm declined, and says which gate refused it
        assert by_arm["L25_CURRENT"].traded is False
        assert by_arm["L25_CURRENT"].failure is FailureReason.L25_REJECTED
        assert "L25" in by_arm["L25_CURRENT"].failure_detail
        assert "L25_CURRENT" not in [p.arm for p in eng.open_positions()]

    def test_both_gates_refusing_still_refuses(self):
        """
        Two operative gates, both saying no, is still a refusal -- "authorize on
        any" must not degrade into "authorize on none".
        """
        report, _ = gated_decision(
            selected_arms=["L25_CURRENT", BETA_GROUP],
            market_cap_usd=50_000.0, volume_5m_usd=1_000.0, change_5m=12.0)
        assert report.l25.authorized is False
        assert report.v2floor.authorized is False
        assert report.decision is Decision.REJECT

    def test_a_gate_refusal_is_not_classified_as_structural(self):
        """
        The split that makes per-strategy authorization work at all: a gate
        refusal belongs to one strategy, a structural refusal to the candidate.
        """
        from autopaper.analysis.decision import _split_blocking

        report, _ = gated_decision(selected_arms=["L25_CURRENT"],
                                   **V2_OK_L25_NO)
        blocking = [f for f in report.risk_flags if f.severity == "block"]
        gate_flags, structural = _split_blocking(blocking)
        assert gate_flags, "the L25 refusal must be gate-scoped"
        assert structural == [], "a strategy gate is not a structural blocker"


# ===========================================================================
# 6. no default / empty / legacy selection activates beta
# ===========================================================================


class TestBetaIsNeverImplicitlyActivated:
    def test_a_fresh_runtime_never_selects_beta(self):
        state = RuntimeState()
        assert state.arms == (DEFAULT_ARM.value,)
        assert state.effective_entry_gates == ("L25",)

    def test_an_empty_selection_is_refused_not_defaulted_to_beta(self):
        ok, _msg, _arms = validate_strategy_selection(
            [], trade_mode=TradeMode.PAPER)
        assert ok is False
        assert effective_entry_gates([]) == ("L25",)

    def test_an_unknown_arm_name_never_yields_v2_floor(self):
        assert effective_entry_gates(["NOT_AN_ARM"]) == ("L25",)
        assert effective_entry_gates(["L25_CURRENT", "NOT_AN_ARM"]) == ("L25",)

    def test_reaching_the_v2_floor_gate_requires_naming_the_group(self):
        state = RuntimeState()
        assert "V2_FLOOR" not in state.effective_entry_gates
        assert state.toggle_strategy(BETA_GROUP)[0] is True
        assert state.effective_entry_gates == ("V2_FLOOR",)

    def test_a_partial_beta_selection_is_not_the_experiment(self):
        state = RuntimeState()
        state.selected_arms = ("V2_BETA_A",)
        # a lone beta arm is not a valid experiment selection
        ok, _msg, arms = validate_strategy_selection(
            ["V2_BETA_A"], trade_mode=TradeMode.PAPER)
        assert ok is True
        assert set(arms) == set(BETA_ARM_NAMES), (
            "a single beta arm token must round-trip to the whole group")
        assert effective_entry_gates(arms) == ("V2_FLOOR",)


# ===========================================================================
# 7. every surface agrees on the effective gates
# ===========================================================================


class TestSurfacesAgree:
    def test_the_state_snapshot_reports_them(self):
        state = RuntimeState()
        assert state.as_dict()["effective_entry_gates"] == ["L25"]
        state.toggle_strategy(BETA_GROUP)
        assert state.as_dict()["effective_entry_gates"] == ["V2_FLOOR"]

    def test_the_ui_status_line_reports_them(self):
        state = RuntimeState()
        assert "Entry gate: <b>L25</b>" in ui.strategy_status_line(state)
        state.toggle_strategy(BETA_GROUP)
        line = ui.strategy_status_line(state)
        assert "Entry gate: <b>V2_FLOOR</b>" in line
        # the profile line is a DIFFERENT fact and must not be the only one
        assert "L25 CURRENT" in line
        state.toggle_strategy("L25_CURRENT")
        mixed = ui.strategy_status_line(state)
        assert "Entry gate: <b>L25, V2_FLOOR</b>" in mixed

    def test_the_runtime_agrees_with_the_state(self, tmp_path):
        """
        ``PaperTrader`` installs exactly what ``RuntimeState`` derives, so the
        layer that decides and the layer the operator reads cannot disagree.
        """
        from autopaper.config import build_config
        from autopaper.engine import PaperTrader

        trader = PaperTrader(build_config(tmp_path / "var"),
                             quote_service=FakeQuoteService())
        try:
            assert trader.engine.entry_gates == ("L25",)
            trader.state.toggle_strategy(BETA_GROUP)
            trader._sync_effective_entry_gates()
            assert trader.engine.entry_gates == ("V2_FLOOR",)
            assert list(trader.engine.entry_gates) == list(
                trader.state.effective_entry_gates)
        finally:
            trader.state.emergency_stop = True

    def test_the_safety_audit_pins_the_contract(self):
        report = safety_audit()
        rows = {row[0]: row for row in report.checks}
        for name in ("effective_entry_gates_resolve_to_declared_gates",
                     "beta_selection_makes_v2_floor_operative",
                     "default_and_empty_selection_never_activates_beta"):
            assert name in rows, f"the audit lost the gate {name}"
            assert rows[name][1] is True, rows[name][2]
        assert report.ok is True

    def test_the_persisted_selection_still_round_trips(self, tmp_path):
        from autopaper.modes import validate_strategy_selection as _v

        ok, _m, arms = _v([BETA_GROUP], trade_mode=TradeMode.PAPER)
        assert ok
        assert tuple(arms) == BETA_ARM_NAMES
        assert effective_entry_gates(arms) == ("V2_FLOOR",)


# ===========================================================================
# 8. one shared entry, independent exits and ledgers
# ===========================================================================


class TestArmsShareOneEntryAndStayIsolated:
    def test_one_signal_one_parent_across_three_arms(self, tmp_path):
        eng = beta_engine(tmp_path, [BETA_GROUP])
        records = run(eng.on_signal(beta_signal(
            v2_volume_5m_usd=60_000.0, v2_market_cap_usd=50_000.0,
            v2_change_5m_pct=12.0)))
        exp = eng.beta_experiment
        assert len(exp.positions) == 1
        parent = next(iter(exp.positions))
        a, b, c = (exp.positions[parent][k] for k in "ABC")
        assert a.entry_fill_price == b.entry_fill_price == c.entry_fill_price
        assert (a.position_size_usd == b.position_size_usd
                == c.position_size_usd)
        assert {r.parent_signal_id for r in records} == {parent}

    def test_closing_one_arm_leaves_the_others_open(self, tmp_path):
        eng = beta_engine(tmp_path, [BETA_GROUP])
        run(eng.on_signal(beta_signal(
            v2_volume_5m_usd=60_000.0, v2_market_cap_usd=50_000.0,
            v2_change_5m_pct=12.0)))
        exp = eng.beta_experiment
        parent = next(iter(exp.positions))
        # +31% touches arm A's +30% take-profit and nothing else.
        exp.observe(parent, "A", mark_price=1.31, at=T0 + 10)
        exp.settle_arm(parent, "A", now=T0 + 11)
        assert exp.positions[parent]["A"].status == "closed"
        assert exp.positions[parent]["B"].status == "open"
        assert exp.positions[parent]["C"].status == "open"
        assert exp.accounts["B"].bankroll_usd == 10.0
        assert exp.accounts["C"].bankroll_usd == 10.0
        assert [t["arm"] for t in exp.completed] == ["A"]

    def test_the_three_ledgers_never_mix(self, tmp_path):
        eng = beta_engine(tmp_path, [BETA_GROUP])
        run(eng.on_signal(beta_signal(
            v2_volume_5m_usd=60_000.0, v2_market_cap_usd=50_000.0,
            v2_change_5m_pct=12.0)))
        accounts = eng.beta_experiment.accounts
        assert [accounts[k].starting_bankroll_usd for k in "ABC"] == \
            [10.0, 10.0, 10.0]
        assert eng.beta_experiment.total_starting_capital_usd == 30.0

    def test_beta_arms_never_hold_an_engine_position(self, tmp_path):
        eng = beta_engine(tmp_path, ["L25_CURRENT", BETA_GROUP])
        run(eng.on_signal(beta_signal(
            v2_volume_5m_usd=60_000.0, v2_market_cap_usd=50_000.0,
            v2_change_5m_pct=12.0)))
        for arm in BETA_ARM_NAMES:
            assert eng.accounts[arm].position is None


# ===========================================================================
# 9. no beta path reaches live execution
# ===========================================================================


class TestNoLivePath:
    def test_the_paper_only_guard_is_intact(self):
        from autopaper.paper_live.guards import assert_paper_only

        assert_paper_only()

    def test_the_gate_contract_adds_no_execution_symbols(self):
        """
        The contract must be arithmetic over a selection. If any function that
        participates in choosing an entry gate ever grew a call into a live
        executor, this catches it.
        """
        banned = {"send_transaction", "sign_transaction", "broadcast",
                  "execute_real", "sign_and_send", "send_raw_transaction"}
        for source in (
            Path(__file__).resolve().parents[1] / "src/autopaper/paper_live/arms.py",
            Path(__file__).resolve().parents[1] / "src/autopaper/strategy_profile.py",
        ):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            called = {n.func.attr for n in ast.walk(tree)
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
            assert not (called & banned), (
                f"{source.name} reaches a live executor: {called & banned}")

    def test_the_decision_layer_never_mentions_a_live_executor(self):
        source = (Path(__file__).resolve().parents[1]
                  / "src/autopaper/analysis/decision.py")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        called = {n.func.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        assert not any("broadcast" in c or "sign" in c for c in called), called

    def test_the_execution_aware_engine_has_no_live_account_at_all(self, tmp_path):
        """
        The strongest form of "no live path": the PAPER execution-aware engine
        does not merely avoid live execution, it has no object that could
        perform it. A beta position therefore cannot be routed anywhere.
        """
        eng = beta_engine(tmp_path, [BETA_GROUP])
        run(eng.on_signal(beta_signal(
            v2_volume_5m_usd=60_000.0, v2_market_cap_usd=50_000.0,
            v2_change_5m_pct=12.0)))
        assert not hasattr(eng, "real_account")
        assert not any("real" in a for a in vars(eng)), (
            f"the PAPER engine holds a real path: "
            f"{[a for a in vars(eng) if 'real' in a]}")
        assert eng.beta_experiment.positions