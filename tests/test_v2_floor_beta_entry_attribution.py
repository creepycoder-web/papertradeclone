"""
``BUY_BLOCKED=V2_FLOOR_REJECTED`` -- attribution and L25 independence.

THE REPORTED EVENT
------------------
One Telegram signal, one CA::

    6qmSkYwdnPmzJ2j5zDApH6eVWjoEc5zy7vDR8ZFUpump
    5m volume $7,269 / market cap $13,878 = 0.5238
    V2 FLOOR relative-turnover floor 1.0
    -> NO POSITION OPENED, Reason: V2_FLOOR_REJECTED

and the operator notification read, in full::

    BUY_BLOCKED=V2_FLOOR_REJECTED
    L25 authorized an entry.
    Outcome: NO POSITION OPENED
    Reason: V2_FLOOR_REJECTED

Those two lines together are the bug: they name two different gates. "L25
authorized an entry" came from a HARD-CODED module constant, while the refusal
came from the V2 FLOOR experiment's own gate on a book that contained no L25
arm at all. The operator had no way to tell which gate refused, or that no L25
entry was ever in play.

WHAT THESE TESTS LOCK
--------------------
1. The reported event reproduces exactly, and the threshold that rejected it
   is UNCHANGED (a green test here is not permission to move the floor).
2. The rejection is reported as a BETA-GATE REJECTION, and no longer claims an
   L25 entry was blocked.
3. One signal raises ONE notification, not three identical ones.
4. L25 is INDEPENDENT: the V2 FLOOR gate is never applied to the L25 entry
   path, and a beta rejection never suppresses an L25 position that was
   authorized on its own terms.

OFFLINE: every collaborator is a double. No network, no Telegram, no RPC,
nothing signed or broadcast. PAPER only.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from autopaper import v2floor
from autopaper.engine import PaperTrader, _is_beta_arm, ui_render_buy_blocked
from autopaper.modes import (
    RuntimeState,
    TradeMode,
    validate_strategy_selection,
)
from autopaper.paper_live.arms import (
    BETA_GROUP,
    ResearchArm,
    parse_arms,
)
from autopaper.paper_live.engine import PaperLiveEngine
from autopaper.paper_live.fills import FillModel
from autopaper.paper_live.quotes import SolReference
from autopaper.paper_live.schema import FailureReason
from autopaper.strategy_profile import (
    DEFAULT_PROFILE,
    PROFILE_ENTRY_STRATEGY,
    parse_profile,
)
from autopaper.telegram import ui

from paper_bridge_fakes import FakeQuoteService, alert, real_report, run
from test_paper_live import (
    FakeMarket,
    MINT as BASE_MINT,
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

#: The exact candidate from the incident.
MINT = "6qmSkYwdnPmzJ2j5zDApH6eVWjoEc5zy7vDR8ZFUpump"

#: The exact decision-time gate inputs from the incident.
VOL5 = 7_269.0
MCAP = 13_878.0
CHG_5M = 12.0
RATIO = 0.5238

#: The selection the incident was running, verbatim from
#: ``var/strategy_selection.json``.
INCIDENT_SELECTION = ["V2_BETA_A", "V2_BETA_B", "V2_BETA_C"]

BETA_ARM_NAMES = ("V2_BETA_A", "V2_BETA_B", "V2_BETA_C")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def incident_signal(**overrides):
    """A snapshot carrying the incident's exact V2 FLOOR gate inputs."""
    fields = dict(
        v2_volume_5m_usd=VOL5,
        v2_market_cap_usd=MCAP,
        v2_change_5m_pct=CHG_5M,
    )
    fields.update(overrides)
    return replace(signal(mint=MINT, telegram_timestamp=1_700_000_000.0,
                          message_id=11), **fields)


def incident_engine(tmp_path, arms, market=None, decision=None) -> PaperLiveEngine:
    """The PAPER engine on ``arms``, state isolated under ``tmp_path``."""
    return PaperLiveEngine(
        arms=parse_arms(arms),
        decision_engine=decision or StubDecision(),
        exit_policy=StubExit(),
        quote_service=FakeQuoteService(),
        market=market if market is not None else FakeMarket(price=1.0),
        token_decimals_for=lambda mint: 6,
        sol_reference=SolReference(SOL_USD, 0.0, "test"),
        fill_model=FillModel(),
        starting_bankroll_usd=5.0,
        require_identity=True,
        state_root=tmp_path,
    )


def _trader(tmp_path, arms, *, report=None, snap=None) -> PaperTrader:
    """A real ``PaperTrader`` whose bridge runs ``arms``.

    The arm selection goes in through the SAME field the operator selector
    writes (``state.selected_arms``) and BEFORE the bridge is built, because
    the bridge is constructed from ``state.arms`` -- so the experiment, the
    per-arm accounts and the beta gate are all built by production code rather
    than patched into place by the test. Every network collaborator is injected
    at construction; conftest's guard is right to refuse otherwise.
    """
    trader = PaperTrader(_config(tmp_path), quote_service=FakeQuoteService())
    trader.state.selected_arms = tuple(arms)
    bridge = run(trader._paper_bridge_instance())
    bridge.token_decimals_for = lambda mint: 6
    bridge.sol_reference = SolReference(150.0, 0.0, "test")
    bridge.quote_service.sol_reference = bridge.sol_reference
    assert bridge.beta_experiment is not None or "L25_CURRENT" in arms, (
        "the beta experiment must be built by production code for a beta "
        "selection")
    trader._test_report = report or _incident_report()
    trader._test_alert = snap or alert(mint=MINT)
    return trader


def _incident_report():
    """A BUY report whose features carry the incident's exact gate inputs."""
    rep = real_report(mint=MINT)
    features = dict(rep.features)
    features.update({"volume_5m_usd": VOL5, "market_cap_usd": MCAP,
                     "change_5m": CHG_5M})
    return replace(rep, features=features)


def _config(tmp_path):
    from autopaper.config import build_config

    return build_config(tmp_path)


def notifications(trader: PaperTrader) -> list[str]:
    out: list[str] = []
    trader.set_notifier(lambda text, kb=None: out.append(text))
    return out


def drive(trader: PaperTrader):
    """Run one signal through the REAL runtime entry path."""
    run(trader._execute_paper_buy_execution_aware(
        trader._test_report, trader._test_alert,
        trader._test_report.features))


# ===========================================================================
# 1. the reported event reproduces exactly
# ===========================================================================


class TestTheReportedEventReproduces:
    def test_the_selection_was_the_three_beta_arms_and_nothing_else(self):
        """
        THE ACTIVE PROFILE TRACE, pinned.

        ``var/strategy_selection.json`` held exactly the three beta arms. The
        PROFILE axis is separate and was still the shipped default:
        ``toggle_strategy`` writes ``selected_arms`` and, in PAPER, never
        touches ``strategy_profile`` (``RuntimeState.select_strategies``
        updates it only for a REAL single-arm selection). So the decision layer
        evaluated the L25 gate while the only books running were the beta arms.
        """
        ok, _msg, arms = validate_strategy_selection(
            INCIDENT_SELECTION, trade_mode=TradeMode.PAPER)
        assert ok
        assert tuple(arms) == BETA_ARM_NAMES
        assert "L25_CURRENT" not in arms, (
            "the incident selection is beta-only; if an L25 arm were selected "
            "the 'L25 authorized' line would have had a referent")

        state = RuntimeState()
        assert state.strategy_profile == DEFAULT_PROFILE.value == "L25_CURRENT"
        assert PROFILE_ENTRY_STRATEGY[parse_profile(state.strategy_profile)] \
            == "L25", "so the decision layer authorized on the L25 gate"

    def test_beta_was_explicitly_selected_not_a_default_or_a_legacy_fallback(
            self):
        """
        A fresh state runs the DEFAULT arm, and an empty request is REFUSED --
        so the beta arms in the incident file cannot have appeared by default,
        by a missing file, or by a legacy configuration value.
        """
        fresh = RuntimeState()
        assert fresh.arms == ("L25_CURRENT",)
        ok, _msg, _arms = validate_strategy_selection(
            [], trade_mode=TradeMode.PAPER)
        assert ok is False, "an empty selection must refuse, not default"
        assert BETA_GROUP not in fresh.arms
        # ...and reaching the beta book requires asking for it by name.
        assert fresh.toggle_strategy(BETA_GROUP)[0] is True

    def test_the_persisted_incident_file_names_the_three_beta_arms(self):
        """The on-disk evidence, if it is still present, says what we assert."""
        path = Path.home() / "AI-paper-trade" / "var" / "strategy_selection.json"
        if not path.exists():
            pytest.skip("operator state file not present in this checkout")
        data = json.loads(path.read_text())
        assert tuple(data["strategies"]) == BETA_ARM_NAMES
        assert data["trade_mode"] == "PAPER"

    def test_the_gate_maths_is_the_reported_maths_and_is_correct(self):
        result = v2floor.evaluate(volume_5m_usd=VOL5, change_5m_pct=CHG_5M,
                                  market_cap_usd=MCAP)
        assert round(result.vol5_per_mc, 4) == RATIO
        assert result.authorized is False
        assert result.status == "V2F_REJECTED_VOL5_PER_MC_BELOW_FLOOR"
        assert "$7,269" in result.detail and "$13,878" in result.detail

    def test_the_threshold_is_not_loosened(self):
        """
        REGRESSION LOCK. The rejection was correct. If this test fails because
        the floor moved, the fix is to restore the floor -- not to relax it.
        """
        assert v2floor.V2F_MIN_VOL5_PER_MC == 1.0
        assert v2floor.V2F_MIN_CHANGE_5M_PCT == 10.0
        assert v2floor.V2F_MAX_CHANGE_5M_PCT == 300.0
        # One unit either side of the floor decides the outcome.
        just_below = v2floor.evaluate(volume_5m_usd=13_877.0,
                                      change_5m_pct=CHG_5M,
                                      market_cap_usd=MCAP)
        just_above = v2floor.evaluate(volume_5m_usd=13_879.0,
                                      change_5m_pct=CHG_5M,
                                      market_cap_usd=MCAP)
        assert just_below.authorized is False
        assert just_above.authorized is True

    def test_the_event_opens_no_beta_position_in_any_arm(self, tmp_path):
        eng = incident_engine(tmp_path, [BETA_GROUP])
        records = run(eng.on_signal(incident_signal()))
        assert len(records) == 3
        assert [r.arm for r in records] == list(BETA_ARM_NAMES)
        assert all(r.traded is False for r in records)
        assert all(r.failure is FailureReason.V2_FLOOR_REJECTED
                   for r in records)
        assert all(r.l25_status == "V2_FLOOR_REJECTED" for r in records)
        assert eng.beta_experiment.positions == {}
        assert eng.beta_positions == {}
        assert eng.open_positions() == []
        # the engine holds no L25 account at all -- the L25 path never ran
        assert sorted(eng.accounts) == sorted(BETA_ARM_NAMES)
        assert "L25_CURRENT" not in eng.accounts

    def test_every_arm_names_the_gate_that_actually_refused(self, tmp_path):
        eng = incident_engine(tmp_path, [BETA_GROUP])
        records = run(eng.on_signal(incident_signal()))
        for r in records:
            assert "vol5/market_cap 0.5238" in r.failure_detail
            assert "relative-turnover floor 1.0000" in r.failure_detail


# ===========================================================================
# 2. the notification is a beta-gate rejection, not a blocked L25 entry
# ===========================================================================


class TestTheNotificationIsAttributedCorrectly:
    def test_the_runtime_sends_one_beta_rejection_not_three(self, tmp_path):
        """
        THE REPEAT. The experiment writes one record per arm, and the runtime
        notified once per refused record -- so one signal raised three
        byte-identical alerts and read as a looping malfunction.
        """
        trader = _trader(tmp_path, [BETA_GROUP])
        notes = notifications(trader)
        drive(trader)
        blocked = [n for n in notes if "BUY_BLOCKED=" in n]
        assert len(blocked) == 1, (
            f"one signal produced {len(blocked)} BUY_BLOCKED notifications; "
            f"the three beta arms refused ONE shared gate")
        assert "V2_FLOOR_REJECTED" in blocked[0]

    def test_the_notification_no_longer_claims_l25_authorized_an_entry(
            self, tmp_path):
        trader = _trader(tmp_path, [BETA_GROUP])
        notes = notifications(trader)
        drive(trader)
        text = notes[0]
        assert "L25 authorized an entry." not in text, (
            "no L25 arm was running; the decision layer's L25 authorization "
            "is not what refused, and saying so sent the operator looking for "
            "a blocked L25 entry that never existed")
        assert "L25_CURRENT" not in text

    def test_the_notification_names_the_beta_gate_and_the_arms(self, tmp_path):
        trader = _trader(tmp_path, [BETA_GROUP])
        notes = notifications(trader)
        drive(trader)
        text = notes[0]
        assert "V2 FLOOR (BETA)" in text
        assert "refused this signal for arms" in text
        for arm in BETA_ARM_NAMES:
            assert arm in text
        assert "NO BETA POSITION OPENED" in text
        assert "0.5238" in text, "the operator still sees the real numbers"

    def test_the_notification_states_no_l25_entry_was_blocked(self, tmp_path):
        trader = _trader(tmp_path, [BETA_GROUP])
        notes = notifications(trader)
        drive(trader)
        assert "No L25 entry was blocked" in notes[0]

    def test_the_outcome_token_stays_greppable(self, tmp_path):
        """The token operators grep for must survive the fix."""
        trader = _trader(tmp_path, [BETA_GROUP])
        notes = notifications(trader)
        drive(trader)
        assert notes[0].startswith("🔴 <b>BUY_BLOCKED=V2_FLOOR_REJECTED</b>")

    def test_the_legacy_l25_block_message_is_unchanged(self):
        """Every non-beta BUY_BLOCKED keeps its exact previous wording."""
        text = ui_render_buy_blocked(symbol="X", mint=MINT,
                                     reason="stale_signal", detail="",
                                     mode=TradeMode.PAPER)
        assert "L25 authorized an entry." in text
        assert "Outcome: <b>NO POSITION OPENED</b>" in text
        assert "V2 FLOOR (BETA)" not in text

    def test_an_unknown_reason_still_degrades_to_readable_text(self):
        text = ui.render_buy_blocked(symbol="X", mint=MINT,
                                     reason="brand_new_reason", detail="",
                                     mode=TradeMode.PAPER,
                                     reason_text="brand new reason")
        assert "brand_new_reason" in text
        assert "NO POSITION OPENED" in text

    def test_arm_attribution_is_read_from_the_registry(self):
        assert _is_beta_arm("V2_BETA_A") is True
        assert _is_beta_arm("V2_BETA_C") is True
        assert _is_beta_arm("L25_CURRENT") is False
        assert _is_beta_arm("H2_CAP60") is False
        assert _is_beta_arm("not_an_arm") is False


# ===========================================================================
# 3. L25 entry authorization is INDEPENDENT of the beta gate
# ===========================================================================


class TestL25EntryIsIndependent:
    def test_l25_and_beta_can_be_selected_together(self):
        """
        The architecture does NOT prevent L25 from running alongside beta.
        PAPER is multi-select, and ``toggle_strategy`` MERGES the group into
        the current selection. This is asserted rather than assumed, because
        the alternative -- the two being mutually exclusive -- would change
        what the whole design means.
        """
        state = RuntimeState()
        assert state.toggle_strategy("L25_CURRENT")[0] is True
        ok, msg = state.toggle_strategy(BETA_GROUP)
        assert ok, msg
        assert "L25_CURRENT" in state.selected_arms
        assert set(BETA_ARM_NAMES) <= set(state.selected_arms)
        assert len(state.selected_arms) == 4

    def test_a_beta_gate_rejection_does_not_suppress_an_l25_entry(self, tmp_path):
        """
        THE INDEPENDENCE CLAIM, end to end.

        Same signal, same gate inputs that just rejected all three beta arms.
        With L25_CURRENT also selected, the L25 arm FILLS: the V2 FLOOR gate is
        confined to the beta experiment and never reaches the L25 entry path.
        """
        eng = incident_engine(tmp_path, ["L25_CURRENT", BETA_GROUP])
        records = run(eng.on_signal(incident_signal()))
        by_arm = {r.arm: r for r in records}
        assert set(by_arm) == {"L25_CURRENT", *BETA_ARM_NAMES}

        assert by_arm["L25_CURRENT"].traded is True, (
            "the beta gate suppressed an L25 entry that passed on its own terms")
        for arm in BETA_ARM_NAMES:
            assert by_arm[arm].traded is False
            assert by_arm[arm].failure is FailureReason.V2_FLOOR_REJECTED

        # and the two books stayed separate
        assert eng.accounts["L25_CURRENT"].position is not None
        for arm in BETA_ARM_NAMES:
            assert eng.accounts[arm].position is None, (
                "a beta arm must never hold an engine position")
        assert eng.beta_experiment.positions == {}

    def test_the_v2_floor_gate_is_never_applied_to_the_l25_path(self, tmp_path):
        """
        The sharpest form: an L25 arm ALONE, on a candidate that fails the V2
        FLOOR floor by half, still trades. The floor is a beta-experiment rule
        and has no authority over L25.
        """
        eng = incident_engine(tmp_path, ["L25_CURRENT"])
        records = run(eng.on_signal(incident_signal()))
        assert len(records) == 1
        assert records[0].arm == "L25_CURRENT"
        assert records[0].traded is True
        assert records[0].failure is FailureReason.NONE
        assert eng.beta_experiment is None
        # nothing in the engine mentions a V2 FLOOR refusal
        assert all(r.l25_status != "V2_FLOOR_REJECTED" for r in records)

    def test_a_coexisting_l25_fill_is_reported_alongside_the_beta_refusal(
            self, tmp_path):
        trader = _trader(tmp_path, ["L25_CURRENT", BETA_GROUP])
        notes = notifications(trader)
        drive(trader)
        text = notes[0]
        assert "V2 FLOOR (BETA)" in text
        assert "NO BETA POSITION OPENED" in text
        assert "opened a position independently" in text
        assert "L25_CURRENT" in text
        assert "NO POSITION OPENED</b>" not in text, (
            "an L25 position WAS opened; the blanket line would be false")
        assert "L25 authorized an entry." not in text

    def test_a_beta_only_selection_is_governed_by_the_beta_gate_alone(
            self, tmp_path):
        """
        The inverse: the beta book answers to the beta gate and nothing else.
        The decision layer authorizes this candidate (the stub L25 gate does),
        yet no beta position opens, because the beta arms were selected and no
        L25 arm was.
        """
        eng = incident_engine(tmp_path, [BETA_GROUP])
        records = run(eng.on_signal(incident_signal()))
        assert eng.decide is not None
        assert all(r.traded is False for r in records)
        assert eng.beta_experiment.positions == {}
        assert eng.open_positions() == []
        assert eng.beta_experiment.accounts["A"].closed_trades == 0


# ===========================================================================
# 4. the three arms share one eligible signal, isolated on exit
# ===========================================================================


class TestBetaArmsShareOneSignalAndStayIsolated:
    def test_one_eligible_signal_opens_exactly_one_position_per_arm(self,
                                                                    tmp_path):
        eng = incident_engine(tmp_path, [BETA_GROUP])
        records = run(eng.on_signal(incident_signal(
            v2_volume_5m_usd=60_000.0, v2_market_cap_usd=50_000.0)))
        assert len(records) == 3
        assert all(r.traded for r in records)
        positions = eng.beta_experiment.positions
        assert len(positions) == 1, "one signal, one shared parent entry"
        parent = next(iter(positions))
        assert set(positions[parent]) == {"A", "B", "C"}
        assert {r.parent_signal_id for r in records} == {parent}
        a, b, c = (positions[parent][k] for k in "ABC")
        assert a.entry_fill_price == b.entry_fill_price == c.entry_fill_price
        assert (a.position_size_usd == b.position_size_usd
                == c.position_size_usd)

    def test_closing_one_arm_leaves_the_other_two_open(self, tmp_path):
        eng = incident_engine(tmp_path, [BETA_GROUP])
        run(eng.on_signal(incident_signal(
            v2_volume_5m_usd=60_000.0, v2_market_cap_usd=50_000.0)))
        exp = eng.beta_experiment
        pid = next(iter(exp.positions))
        # +31% touches arm A's +30% take-profit and NOTHING else.
        exp.observe(pid, "A", mark_price=1.31, at=1_700_000_010.0)
        exp.settle_arm(pid, "A", now=1_700_000_011.0)
        assert exp.positions[pid]["A"].status == "closed"
        assert exp.positions[pid]["A"].exit_reason == "TP"
        assert exp.positions[pid]["B"].status == "open"
        assert exp.positions[pid]["C"].status == "open"
        assert exp.accounts["B"].bankroll_usd == 10.0
        assert exp.accounts["C"].bankroll_usd == 10.0
        assert [t["arm"] for t in exp.completed] == ["A"]

    def test_the_three_ledgers_never_mix(self, tmp_path):
        eng = incident_engine(tmp_path, [BETA_GROUP])
        run(eng.on_signal(incident_signal(
            v2_volume_5m_usd=60_000.0, v2_market_cap_usd=50_000.0)))
        accounts = eng.beta_experiment.accounts
        assert [accounts[k].starting_bankroll_usd for k in "ABC"] == \
            [10.0, 10.0, 10.0]
        assert eng.beta_experiment.total_starting_capital_usd == 30.0


# ===========================================================================
# 5. the fix touches no threshold and no routing
# ===========================================================================


class TestNothingElseMoved:
    def test_the_paper_only_guard_is_intact(self):
        """No part of this can reach a live path."""
        from autopaper.paper_live.guards import assert_paper_only

        assert_paper_only()

    def test_a_rejection_opens_no_beta_position_and_disturbs_no_other_book(
            self, tmp_path):
        """
        The mixed selection, stated per-book.

        The beta gate refused, so no BETA position exists. The L25 arm was
        authorized on its own terms and is untouched by the beta refusal -- one
        engine position, on the L25 book, and nothing in any beta ledger.
        """
        eng = incident_engine(tmp_path, ["L25_CURRENT", BETA_GROUP])
        records = run(eng.on_signal(incident_signal()))
        by_arm = {r.arm: r for r in records}
        assert by_arm["L25_CURRENT"].traded is True
        assert all(by_arm[a].traded is False for a in BETA_ARM_NAMES)

        assert eng.beta_experiment.positions == {}
        assert all(eng.beta_experiment.accounts[k].closed_trades == 0
                   for k in "ABC")
        assert all(eng.beta_experiment.accounts[k].bankroll_usd == 10.0
                   for k in "ABC")
        # exactly one engine position, and it is not a beta arm's
        assert [p.arm for p in eng.open_positions()] == ["L25_CURRENT"]