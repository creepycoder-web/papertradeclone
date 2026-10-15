"""
V2 FLOOR (BETA) SESSION ROLLOVER -- a clean $10-per-arm session, nothing lost.

WHY THIS IS NOT ``reset``
------------------------
``reset`` settles outstanding arms FIRST, booking every OBSERVED arm closed at
its last observed return. That is right for a kill switch or a profile switch. It
is WRONG for an experiment rollover whose outstanding arms have NO executable
exit: it would stamp an exit price and a P&L onto a position whose market never
paid one, which is the single thing this harness exists to prevent.

``rollover_session`` prices NOTHING. It archives the session verbatim, starts
three ledgers at their starting bankroll, and carries the outstanding arms over
as explicitly NON-SETTLED incident records.

WHAT IS PINNED HERE
-------------------
1. the new session is $10.00 / $10.00 / $10.00 on three INDEPENDENT ledgers -- a
   $30.00 total -- with no positions, no completed trades and no inherited P&L;
2. the previous session file is preserved VERBATIM under a timestamped name;
3. the outstanding arms are carried over with NO exit price, NO proceeds and NO
   P&L, and are never counted as trades;
4. those arms can never settle into the new session's performance;
5. historical results are neither deleted nor double-counted;
6. the forward store is rotated too, with a fresh forward session id;
7. the code-level guarantees survive: independent bankrolls, ALL-IN sizing, the
   affordability check and the unresolved-position block;
8. ``reset`` still flattens -- the rollover did not change it;
9. the carried-over incidents are DURABLE (they survive a reload).

PAPER only. No network, no signer, no transaction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from autopaper.paper_live.v2_floor_beta import (  # noqa: E402
    BETA_ARMS,
    BETA_SIZING_RULE,
    BETA_STARTING_BANKROLL_USD,
    FORWARD_STATE_FILENAME,
    SIZING_ALL_IN,
    BetaExperiment,
    forward_eligible,
    _is_incident_recovery,
    _session_id_of,
)

from test_v2_floor_beta import T0, new_exp, open_p1  # noqa: E402

pytestmark = [pytest.mark.paper_live, pytest.mark.strategy, pytest.mark.safety]

BETA_STATE_FILENAME = "v2_floor_beta_state.json"


# ---------------------------------------------------------------------------
# a session that looks like the live one: history + a settled ALL-IN arm +
# three arms stranded by an exit whose market never paid
# ---------------------------------------------------------------------------


def _session_with_history_and_stranded_arms(tmp_path) -> BetaExperiment:
    """A session carrying BOTH a booked result and an unresolved incident."""
    exp = new_exp(tmp_path)

    # A HISTORICAL MODELLED trade: a legacy 10%-of-equity record. It must stay
    # exactly where it is -- it is not forward eligible and must never be
    # re-based into the new session.
    exp.state.trades["hist::A::1"] = {
        "record_type": "beta_exit", "arm": "A", "parent_signal_id": "hist",
        "exit_at": T0 - 500.0, "net_pnl_usd": 4.25,
        "net_return_pct": 42.5, "status": "closed",
        "session_id": "beta_OLD_SESSION",
        "pnl_basis": "MODELED_LIMIT_OR_TROUGH",
    }
    # A clean ALL-IN quote-based result.
    exp.state.trades["clean::B::2"] = {
        "record_type": "beta_exit", "arm": "B", "parent_signal_id": "clean",
        "exit_at": T0 - 100.0, "net_pnl_usd": -1.75,
        "net_return_pct": -17.5, "status": "closed",
        "session_id": "beta_OLD_SESSION",
        "pnl_basis": "V2_FLOOR_BETA_QUOTE_SIMULATED_FILL",
        "sizing_rule": "ALL_IN_100_PCT_OF_AVAILABLE_CASH",
        "exit_attempts": 0,
    }
    exp.state_store.save(exp.state, note="beta session beta_OLD_SESSION")

    # A live session loaded from that state, plus ONE parent whose three arms
    # have a due exit and no executable quote.
    live = BetaExperiment(starting_bankroll_usd=BETA_STARTING_BANKROLL_USD,
                          state_root=tmp_path)
    open_p1(live, size=2.0, pid="stranded")
    # Past BOTH max-hold brackets (90s / 90s / 120s), so all three arms have a
    # DUE exit, and no quote was supplied -- the exact shape of the incident.
    for arm in BETA_ARMS:
        # The live arms are ALL-IN; the explicit-size primitive is a fixture
        # entry point, so the rule is stamped to match production -- otherwise
        # the arms would not be part of the forward sample at all.
        live.positions["stranded"][arm].sizing_rule = SIZING_ALL_IN
        live.observe("stranded", arm, mark_price=1.31, at=T0 + 5)
        live.settle_arm("stranded", arm, now=T0 + 200, require_quote=True)
    for arm in BETA_ARMS:
        bp = live.positions["stranded"][arm]
        assert bp.status == "unresolved"
        assert bp.exit_attempts >= 1
        assert bp.net_pnl_usd is None
    live._persist()
    return live


# ---------------------------------------------------------------------------
# 1. the new session
# ---------------------------------------------------------------------------


def test_1_the_new_session_is_ten_dollars_per_arm(tmp_path):
    exp = _session_with_history_and_stranded_arms(tmp_path)
    before = {a: exp.accounts[a].bankroll_usd for a in BETA_ARMS}
    assert before["A"] != 10.0, "the fixture must carry inherited P&L"

    out = exp.rollover_session(reason="unit test")

    assert {a: exp.accounts[a].bankroll_usd for a in BETA_ARMS} == {
        "A": 10.0, "B": 10.0, "C": 10.0}
    assert out["bankroll_per_arm_after"] == {"A": 10.0, "B": 10.0, "C": 10.0}
    assert exp.total_starting_capital_usd == 30.0
    assert out["total_starting_capital_usd"] == 30.0
    assert out["flattened"] is False, (
        "the rollover must never flatten; that is what reset() is for")
    assert exp.positions == {}
    assert exp.completed == []


def test_2_the_ledgers_stay_independent_and_the_sizing_rule_is_unchanged(
        tmp_path):
    exp = _session_with_history_and_stranded_arms(tmp_path)
    exp.rollover_session(reason="unit test")
    # Three SEPARATE accounts, not one shared $30.
    assert len({id(exp.accounts[a]) for a in BETA_ARMS}) == 3
    assert {exp.accounts[a].arm for a in BETA_ARMS} == {
        "V2BETA_A", "V2BETA_B", "V2BETA_C"}
    assert BETA_SIZING_RULE.endswith("ALL_IN_100_PCT_OF_AVAILABLE_CASH")
    plan = exp.arm_entry_plan()
    for arm in BETA_ARMS:
        row = plan[arm]
        assert not row["blocked"], (
            "a clean session must not carry a blocked arm")
        # 100% of deployable, and the reserve is what makes it affordability
        # rather than optimism.
        assert row["sizing_pct"] == 100.0
        assert row["notional_usd"] > 0


# ---------------------------------------------------------------------------
# 2. nothing is lost
# ---------------------------------------------------------------------------


def test_3_the_previous_session_is_archived_verbatim(tmp_path):
    exp = _session_with_history_and_stranded_arms(tmp_path)
    out = exp.rollover_session(reason="unit test")
    archived = Path(out["archived_to"])
    assert archived.exists()
    payload = json.loads(archived.read_text(encoding="utf-8"))
    assert payload["counts"]["trades"] == 2, (
        "the archived session must keep every historical trade")
    assert payload["counts"]["open_positions"] == 3, (
        "the archived session must keep every unresolved position")
    keys = {t["parent_signal_id"] for t in payload["trades"]}
    assert keys == {"hist", "clean"}


def test_4_the_unresolved_arms_are_carried_with_no_price_and_no_pnl(tmp_path):
    eng = _session_with_history_and_stranded_arms(tmp_path)
    out = eng.rollover_session(reason="unit test")
    carried = out["outstanding_carried"]
    assert {row["arm"] for row in carried} == set(BETA_ARMS)
    for row in carried:
        assert row["status"] == "unresolved"
        assert row["entry_tokens_raw"] > 0
        assert row["exit_attempts"] >= 1
        assert row["exit_blocked_reason"]

    records = list(eng.state.incident_recovery.values())
    assert len(records) == 3
    for rec in records:
        assert rec["record_type"] == "beta_unresolved_incident"
        assert rec["status"] == "UNRESOLVED_INCIDENT"
        assert rec["settled"] is False
        assert rec["counted_as_trade"] is False
        # THE POINT: nothing was priced.
        assert rec["exit_price"] is None
        assert rec["exit_proceeds_usd"] is None
        assert rec["net_pnl_usd"] is None
        assert rec["session_id"] == "beta_OLD_SESSION"
        assert rec["carried_into_session"] == out["new_session_id"]
        assert rec["exit_blocked_reason"]
        assert rec["note"]


def test_5_the_new_session_counts_none_of_the_history(tmp_path):
    exp = _session_with_history_and_stranded_arms(tmp_path)
    exp.rollover_session(reason="unit test")
    counts = exp.state.as_dict()["counts"]
    assert counts["trades"] == 0, "the new session must start with no trades"
    assert counts["open_positions"] == 0
    assert counts["incident_recovery"] == 3, (
        "the incident must be recorded, not discarded")
    stats = exp.stats()
    assert stats["starting_bankroll_per_arm_usd"] == 10.0
    assert stats["total_starting_capital_usd"] == 30.0
    for arm in BETA_ARMS:
        assert stats["arms"][arm]["completed_trades"] == 0
        assert stats["arms"][arm]["realized_pnl_usd"] == 0.0
        assert stats["arms"][arm]["bankroll_usd"] == 10.0
        assert stats["arms"][arm]["open_positions"] == 0


def test_6_a_carried_arm_cannot_settle_into_the_new_session(tmp_path):
    exp = _session_with_history_and_stranded_arms(tmp_path)
    exp.rollover_session(reason="unit test")
    # The archived parent id is NOT in the new book, so it can never be settled
    # here -- and its future settlement cannot land in this session's figures.
    assert "stranded" not in exp.positions
    exp.observe("stranded", "A", mark_price=99.0, at=T0 + 10)
    assert exp.settle_arm("stranded", "A", now=T0 + 11,
                          require_quote=True) is None
    assert exp.completed == []
    assert exp.accounts["A"].bankroll_usd == 10.0
    assert not forward_eligible({"sizing_rule": SIZING_ALL_IN})


def test_7_the_incidents_survive_a_reload(tmp_path):
    exp = _session_with_history_and_stranded_arms(tmp_path)
    out = exp.rollover_session(reason="unit test")
    again = BetaExperiment(starting_bankroll_usd=BETA_STARTING_BANKROLL_USD,
                           state_root=tmp_path)
    assert again.session_id != out["prior_session_id"]
    assert {a: again.accounts[a].bankroll_usd for a in BETA_ARMS} == {
        "A": 10.0, "B": 10.0, "C": 10.0}
    assert again.positions == {}
    assert again.completed == []
    records = list(again.state.incident_recovery.values())
    assert len(records) == 3
    assert all(r["net_pnl_usd"] is None for r in records)
    assert all(r["carried_into_session"] == again.session_id
               for r in records)


# ---------------------------------------------------------------------------
# 3. the forward store
# ---------------------------------------------------------------------------


def test_8_the_forward_store_is_rotated_with_a_fresh_session_id(tmp_path):
    exp = _session_with_history_and_stranded_arms(tmp_path)
    assert (tmp_path / FORWARD_STATE_FILENAME).exists(), (
        "the fixture must have written a forward store")
    before = exp.forward_session_id
    before_forward = exp.forward_state.as_dict()
    out = exp.rollover_session(reason="unit test")
    assert out["prior_forward_session_id"] == before
    assert out["new_forward_session_id"] != before, (
        "the clean sample must get its own durable id")
    assert out["archived_forward_to"]
    payload = json.loads(
        Path(out["archived_forward_to"]).read_text(encoding="utf-8"))
    assert payload["counts"]["open_positions"] == 3, (
        "the archived forward sample must keep its outstanding arms")
    assert before_forward["counts"]["open_positions"] == 3, (
        "the fixture's arms must be part of the forward sample")
    fresh = json.loads((tmp_path / FORWARD_STATE_FILENAME).read_text(
        encoding="utf-8"))
    assert fresh["counts"]["trades"] == 0
    assert fresh["counts"]["open_positions"] == 0
    assert fresh["counts"]["incident_recovery"] == 0


def test_9_the_forward_progress_restarts_at_zero(tmp_path):
    exp = _session_with_history_and_stranded_arms(tmp_path)
    exp.rollover_session(reason="unit test")
    progress = exp.forward_progress()
    assert progress["unique_parent_signals"] == 0
    assert progress.get("performance_claim") is False


# ---------------------------------------------------------------------------
# 4. reset() is untouched
# ---------------------------------------------------------------------------


def test_10_reset_still_flattens_and_rollover_does_not(tmp_path):
    """The two paths must stay different, and this is the difference."""
    rolled = _session_with_history_and_stranded_arms(tmp_path / "roll")
    out = rolled.rollover_session(reason="unit test")
    assert out["flattened"] is False
    assert rolled.state.as_dict()["counts"]["trades"] == 0

    reset = _session_with_history_and_stranded_arms(tmp_path / "reset")
    summary = reset.reset()
    assert summary["flattened_before_reset"], (
        "reset must still flatten the outstanding arms")
    assert len(summary["flattened_before_reset"]["closed"]) == 3, (
        "reset books the outstanding arms closed at their last observed return")


def test_11_the_rollover_reason_is_recorded_on_the_new_session(tmp_path):
    exp = _session_with_history_and_stranded_arms(tmp_path)
    out = exp.rollover_session(reason="V2 FLOOR BETA all-in $10/arm")
    entry = next(
        rec for rec in exp.state.write_log
        if rec.get("event") == "session_rollover")
    assert entry["reason"] == "V2 FLOOR BETA all-in $10/arm"
    assert entry["new_session_id"] == out["new_session_id"]
    assert entry["flattened"] is False
    assert entry["outstanding_carried_as_incidents"] == 3
    assert entry["starting_bankroll_usd_per_arm"] == 10.0
    assert entry["prior_counts"]["trades"] == 2
    # The archived session id is read back from the FILE, not from memory, so a
    # rollover performed after a restart still names the session it ended.
    assert out["prior_session_id"] == "beta_OLD_SESSION"


def test_13_a_second_rollover_names_the_session_it_actually_ends(tmp_path):
    """A rolled session must be identifiable from its OWN file.

    The carried incidents keep the id of the session that PRODUCED them, which
    after a rollover is the PREVIOUS one. Reading that as "the session this file
    belongs to" made a second rollover name the session before the one it was
    ending -- an audit trail that mislabels its own boundary.
    """
    exp = _session_with_history_and_stranded_arms(tmp_path)
    first = exp.rollover_session(reason="first")

    persisted = json.loads(
        (tmp_path / BETA_STATE_FILENAME).read_text(encoding="utf-8"))
    assert persisted["counts"]["trades"] == 0
    assert _session_id_of(persisted) == first["new_session_id"], (
        "the freshly rolled file must name the session it is, not the one its "
        "carried incidents came from")

    # A restart reads the same answer, because it is read from the FILE.
    again = BetaExperiment(starting_bankroll_usd=BETA_STARTING_BANKROLL_USD,
                           state_root=tmp_path)
    assert _session_id_of(again.state.as_dict()) == first["new_session_id"]

    second = again.rollover_session(reason="second")
    assert second["prior_session_id"] == first["new_session_id"], (
        "the second rollover named the wrong session as the one it ended")
    assert second["new_session_id"] != first["new_session_id"]

    # ... and the carried incidents still point back at the session that
    # produced them, and at the session they were carried INTO -- both facts are
    # provenance and neither may be rewritten by the next boundary.
    for rec in again.state.incident_recovery.values():
        assert rec["session_id"] == "beta_OLD_SESSION"
        assert rec["carried_into_session"] == first["new_session_id"]
        assert rec["net_pnl_usd"] is None
        assert rec["exit_price"] is None and rec["exit_proceeds_usd"] is None
        assert rec["carried_forward_through"] == [second["new_session_id"]], (
            "the incident must record the boundary it survived -- and only "
            "that one, because a boundary is an event, not a running state")
    assert len(again.state.incident_recovery) == 3, (
        "a later rollover must not drop or duplicate an incident")

    # ... and the LIVE file on disk carries them, which is the whole point: an
    # incident that only exists in an archive cannot be seen by the operator.
    live = json.loads(
        (tmp_path / BETA_STATE_FILENAME).read_text(encoding="utf-8"))
    assert live["counts"]["incident_recovery"] == 3
    assert live["counts"]["trades"] == 0
    entry = [e for e in live["write_log"]
             if e.get("event") == "session_rollover"][-1]
    assert entry["incidents_carried_forward"] == 3
    assert entry["outstanding_carried_as_incidents"] == 0

    # A third boundary keeps them too, and is still named correctly.
    third = again.rollover_session(reason="third")
    assert third["prior_session_id"] == second["new_session_id"]
    assert len(again.state.incident_recovery) == 3
    assert all(
        rec["carried_forward_through"]
        == [second["new_session_id"], third["new_session_id"]]
        for rec in again.state.incident_recovery.values()), (
        "each boundary the incident survived must be on the record")


def test_14_a_live_run_still_names_itself_from_its_own_trades(tmp_path):
    """The trade record keeps first authority: it names the run that booked it."""
    exp = _session_with_history_and_stranded_arms(tmp_path)
    exp.state.trades["t::A::9"] = {
        "record_type": "beta_exit", "arm": "A", "parent_signal_id": "t",
        "exit_at": T0, "net_pnl_usd": 1.0, "status": "closed",
        "session_id": "beta_LIVE_RUN", "exit_attempts": 0,
    }
    assert _session_id_of(exp.state.as_dict()) == "beta_LIVE_RUN"
    out = exp.rollover_session(reason="unit test")
    assert out["prior_session_id"] == "beta_LIVE_RUN"


def test_12_no_incident_record_is_ever_forward_eligible(tmp_path):
    exp = _session_with_history_and_stranded_arms(tmp_path)
    exp.rollover_session(reason="unit test")
    for rec in exp.state.incident_recovery.values():
        assert not forward_eligible(rec), (
            "a non-settled incident must never enter the clean cohort")
        assert not rec["counted_as_trade"]
    # And the carried records carry NO forward-eligible marker at all.
    for rec in exp.state.incident_recovery.values():
        assert "pnl_basis" not in rec
        assert "sizing_rule" not in rec
