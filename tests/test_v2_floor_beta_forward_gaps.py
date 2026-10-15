"""Focused tests for the forward-validation GAPS closed in this pass.

WHAT THESE PROVE
================

1. **A pinned SELL quote can actually settle an arm.** The production quote
   service's ``quote_sell_pinned`` returns a PAIR -- ``(chosen, counterfactual)``
   -- and the beta exit read the PAIR itself as the quote, so
   ``getattr(pair, "available", False)`` was False on EVERY call and no pinned
   arm could ever settle. The live experiment is the evidence: three arms,
   1273/1273/1267 refused attempts, zero settlements. The offline doubles never
   exposed it because they define ``quote_sell``/``quote_sell_best`` and no
   ``quote_sell_pinned`` at all.

2. **The forward experiment is an ISOLATED store.** New ALL-IN QUOTE-based
   results are mirrored into ``v2_floor_beta_forward_state.json``; the historical
   MODELLED trades in the main store are excluded BY CONSTRUCTION, are never
   summed into the forward figures and are never rewritten.

3. **Forward progress is a COUNTER, not a claim.** A parent signal counts once
   however many of its three arms settle, A/B/C are grouped under the same
   parent id, an arm that could not be quoted stays UNRESOLVED, and no
   profitability claim is made at any sample size.

4. **The cost model's provenance kinds stay apart** -- DERIVED (from the
   intended transaction configuration), CONFIGURED (a ceiling, not an
   observation), MEASURED, ASSUMED -- and a refused QUOTE REQUEST is
   distinguished from a failed BROADCAST, which is the only one that pays a
   network fee.

Everything is OFFLINE with isolated temporary state roots: no network, no
Telegram, no RPC, no real ``var/`` state, nothing signed or broadcast.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_paper_live import (  # noqa: E402
    FakeQuoteService,
    run,
    unavailable_point,
)
from test_v2_floor_beta import (  # noqa: E402
    T0,
    eligible_signal,
    move_sell_quote,
    new_exp,
    open_p1,
)
from test_v2_floor_beta_integration import (  # noqa: E402
    SizingQuoteService,
    _beta_arm,
    _exit_at_multiple,
)

from autopaper.paper_live.execution_costs import (  # noqa: E402
    COST_ASSUMED,
    COST_CONFIGURED,
    COST_DERIVED,
    PAPER_PRIORITY_FEE_LAMPORTS,
)
from autopaper.paper_live.v2_floor_beta import (  # noqa: E402
    COST_MODEL_LEGACY,
    COST_MODEL_MODELED,
    FORWARD_COHORT,
    FORWARD_STATE_FILENAME,
    FORWARD_TARGET_PARENT_SIGNALS,
    INCIDENT_RECOVERY_COHORT,
    BetaExperiment,
    forward_eligible,
    forward_eligible_open,
)

pytestmark = [pytest.mark.paper_live, pytest.mark.strategy]


# ---------------------------------------------------------------------------
# doubles that reproduce the PRODUCTION provider's SHAPE
# ---------------------------------------------------------------------------

class PinnedSellQuoteService(SizingQuoteService):
    """A provider whose ``quote_sell_pinned`` returns a PAIR, like the real one.

    ``autopaper.paper_live.quotes.QuoteService.quote_sell_pinned`` returns
    ``(chosen, counterfactual)``. The shared doubles do not implement it, which
    is exactly why the defect below survived a green suite.
    """

    def __init__(self, *, sell=None, **kw):
        super().__init__(**kw)
        if sell is not None:
            self.sell = sell
        self.pinned_calls: list[dict] = []

    async def quote_sell_pinned(self, **kw):
        self.pinned_calls.append(kw)
        return self._restamp(self.sell, kw["mint"]), None


# ---------------------------------------------------------------------------
# 1. the pinned SELL quote shape (the live defect)
# ---------------------------------------------------------------------------

def _open_and_trigger(eng):
    """Open one signal and observe every arm past its own TP target.

    The quote -- not the mark -- is what drives the bracket, so move it
    first, then mirror the mark only as a diagnostic.
    """
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    for arm in ("A", "B", "C"):
        pos = _beta_arm(exp, pid, arm)
        move_sell_quote(eng, 1.55)
        exp.observe(pid, arm, mark_price=pos.entry_fill_price * 1.55,
                    at=T0 + 5)
    return pid


def test_a_pinned_sell_quote_PAIR_is_unwrapped_so_the_arm_can_settle(tmp_path):
    from test_v2_floor_beta import beta_engine

    qs = PinnedSellQuoteService()
    eng = beta_engine(tmp_path, quotes=qs)
    run(eng.on_signal(eligible_signal()))
    pid = _open_and_trigger(eng)
    exp = eng.beta_experiment
    assert _beta_arm(exp, pid, "A").status == "open"
    run(eng._monitor_beta_experiment())
    # THE ASSERTION THAT WOULD HAVE FAILED BEFORE THE FIX: the pinned path was
    # taken (`observed_pair` is set and the provider exposes the method)...
    assert qs.pinned_calls, "the pinned SELL quote path was not exercised"
    # ... and the arm SETTLED from the quote instead of being refused forever.
    assert _beta_arm(exp, pid, "A").status == "closed"
    from autopaper.paper_live.v2_floor_beta import EXECUTABLE_QUOTE_BASIS
    assert exp.completed[0]["exit_basis"] == EXECUTABLE_QUOTE_BASIS
    assert exp.completed[0]["net_pnl_usd"] is not None


def test_a_pinned_sell_quote_that_is_UNAVAILABLE_stays_unresolved(tmp_path):
    """The unwrap must not become a way to settle off a missing quote."""
    from test_v2_floor_beta import beta_engine

    qs = PinnedSellQuoteService(sell=unavailable_point(side="SELL"))
    eng = beta_engine(tmp_path, quotes=qs)
    run(eng.on_signal(eligible_signal()))
    pid = _open_and_trigger(eng)
    exp = eng.beta_experiment
    run(eng._monitor_beta_experiment())
    pos = _beta_arm(exp, pid, "A")
    assert pos.status == "unresolved"
    assert not [t for t in exp.completed if t.get("status") == "closed"]
    assert pos.exit_attempts >= 1


def test_a_bare_quote_point_from_the_pinned_path_still_works(tmp_path):
    """A provider that returns the quote directly (older shape) must also work."""
    from test_v2_floor_beta import beta_engine

    class BareQuoteService(SizingQuoteService):
        async def quote_sell_pinned(self, **kw):
            self.sell_calls.append(kw)
            return self._restamp(self.sell, kw["mint"])

    eng = beta_engine(tmp_path, quotes=BareQuoteService())
    run(eng.on_signal(eligible_signal()))
    pid = _open_and_trigger(eng)
    run(eng._monitor_beta_experiment())
    assert _beta_arm(eng.beta_experiment, pid, "A").status == "closed"


# ---------------------------------------------------------------------------
# 2. the forward store: isolated, and only ALL-IN QUOTE-based results
# ---------------------------------------------------------------------------

def _forward_file(tmp_path: Path) -> Path:
    return tmp_path / FORWARD_STATE_FILENAME


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_eligibility_refuses_legacy_and_modelled_records():
    modelled = {"net_pnl_usd": 1.0, "exit_basis": "MODELED_LIMIT_OR_TROUGH"}
    quote_legacy_size = {
        "net_pnl_usd": 1.0,
        "exit_basis": "EXECUTABLE_QUOTE_SIMULATED_FILL",
        "pnl_basis": "V2_FLOOR_BETA_QUOTE_SIMULATED_FILL",
    }
    quote_all_in = dict(quote_legacy_size, sizing_rule="ALL_IN_100_PCT")
    assert forward_eligible(modelled) is False
    # quote-based but 10%-of-equity: refused, that is the historical rule
    assert forward_eligible(quote_legacy_size) is False
    assert forward_eligible(quote_all_in) is True
    assert forward_eligible(None) is False
    # an OUTSTANDING arm needs the same two facts on its ENTRY
    assert forward_eligible_open(
        {"entry_basis": "MODELED_MARK_LEGACY", "sizing_rule": "ALL_IN_100_PCT"}
    ) is False
    assert forward_eligible_open(
        {"entry_basis": "EXECUTABLE_QUOTE_SIMULATED_FILL",
         "sizing_rule": "ALL_IN_100_PCT"}) is True


def test_the_historical_modelled_trades_never_enter_the_forward_store(tmp_path):
    """The main store keeps them; the forward store is empty and stays empty."""
    exp = new_exp(tmp_path)
    open_p1(exp)                       # EXPLICIT_SIZE fixture, not ALL-IN
    pos = _beta_arm(exp, "p1", "A")
    exp.observe("p1", "A", mark_price=pos.entry_fill_price * 1.31, at=T0 + 5)
    # the modelled/backtest booking: no quote, but the path was observed
    exp.settle_arm("p1", "A", now=T0 + 6, exit_fill=None, require_quote=False)
    assert exp.completed, "the modelled trade must still be booked"
    main = _read(tmp_path / "v2_floor_beta_state.json")
    assert len(main["trades"]) == 1
    fwd = _read(_forward_file(tmp_path))
    assert fwd["trades"] == []
    prog = exp.forward_progress()
    assert prog["unique_parent_signals"] == 0
    assert prog["completed_trades"] == 0
    assert prog["performance_claim"] is False
    assert "EXCLUDED" in prog["note"]


def test_a_quote_based_all_in_result_enters_the_forward_store_grouped_by_parent(
        tmp_path):
    from test_v2_floor_beta import beta_engine

    eng = beta_engine(tmp_path, quotes=SizingQuoteService())
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    for arm in ("A", "B", "C"):
        pos = _beta_arm(exp, pid, arm)
        exp.observe(pid, arm, mark_price=pos.entry_fill_price * 1.55, at=T0 + 5)
        exp.settle_arm(pid, arm, now=T0 + 6,
                       exit_fill=_exit_at_multiple(exp, pid, arm, 1.30),
                       require_quote=True)
    prog = exp.forward_progress()
    assert prog["unique_parent_signals"] == 1
    assert prog["completed_trades"] == 3
    assert prog["parents_with_all_three_arms_settled"] == 1
    group = prog["grouped_by_parent"][0]
    assert group["parent_signal_id"] == pid
    assert group["arms_settled"] == ["A", "B", "C"]
    assert group["all_three_arms_settled"] is True
    for arm in ("A", "B", "C"):
        assert group["arms"][arm]["status"] == "closed"
        assert group["arms"][arm]["net_pnl_usd"] is not None
    # the durable forward file carries the same three records ...
    fwd = _read(_forward_file(tmp_path))
    assert len(fwd["trades"]) == 3
    assert {t["arm"] for t in fwd["trades"]} == {"A", "B", "C"}
    # ... and the MAIN store still holds the SAME three trades (one ledger each)
    main = _read(tmp_path / "v2_floor_beta_state.json")
    assert len(main["trades"]) == 3
    assert {t["arm"] for t in main["trades"]} == {"A", "B", "C"}


def test_the_forward_store_is_a_SEPARATE_file_and_survives_a_restart(tmp_path):
    from test_v2_floor_beta import beta_engine

    eng = beta_engine(tmp_path, quotes=SizingQuoteService())
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    pos = _beta_arm(exp, pid, "A")
    fill = _exit_at_multiple(exp, pid, "A", 1.30)
    exp.observe_quote(pid, "A", exit_fill=fill, mark_price=None, at=T0 + 5)
    exp.settle_arm(pid, "A", now=T0 + 6, exit_fill=fill, require_quote=True)
    before = exp.forward_progress()
    assert before["store"] != str(tmp_path / "v2_floor_beta_state.json")
    assert Path(before["store"]).name == FORWARD_STATE_FILENAME
    # a NEW instance over the same root reports the SAME forward sample ...
    revived = BetaExperiment(state_root=tmp_path)
    after = revived.forward_progress()
    assert after["unique_parent_signals"] == before["unique_parent_signals"] == 1
    assert after["completed_trades"] == before["completed_trades"] == 1
    assert after["by_arm"] == before["by_arm"]
    assert after["session_id"] == before["session_id"]
    # ... and reopening moved NEITHER file's record count.
    assert len(_read(_forward_file(tmp_path))["trades"]) == 1


def test_an_unresolved_forward_arm_is_PROGRESS_and_not_a_settled_result(
        tmp_path):
    from test_v2_floor_beta import beta_engine

    eng = beta_engine(tmp_path, quotes=FakeQuoteService(
        sell=unavailable_point(side="SELL")))
    run(eng.on_signal(eligible_signal()))
    pid = _open_and_trigger(eng)
    exp = eng.beta_experiment
    run(eng._monitor_beta_experiment())
    prog = exp.forward_progress()
    assert prog["completed_trades"] == 0
    assert prog["unique_parent_signals"] == 1      # the parent IS progress
    assert prog["unresolved_arms"] == 3
    assert prog["parents_with_all_three_arms_settled"] == 0
    group = prog["grouped_by_parent"][0]
    assert group["all_three_arms_settled"] is False
    assert group["arms"]["A"] is None
    assert group["unresolved"]["A"]["status"] == "unresolved"
    assert group["realized_pnl_usd"] == 0.0
    # the durable file records the outstanding arm AND no fabricated trade
    fwd = _read(_forward_file(tmp_path))
    assert fwd["trades"] == []
    assert len(fwd["open_positions"]) == 3
    assert all(p["status"] == "unresolved" for p in fwd["open_positions"])


def test_progress_reports_the_target_and_never_claims_a_result(tmp_path):
    exp = new_exp(tmp_path)
    prog = exp.forward_progress()
    assert FORWARD_TARGET_PARENT_SIGNALS == 50
    assert prog["target_unique_parent_signals"] == 50
    assert prog["remaining_to_target"] == 50
    assert prog["target_reached"] is False
    assert prog["progress_pct"] == 0.0
    assert prog["performance_claim"] is False
    assert prog["eligibility"]
    assert prog["excludes"]
    # the settings are STATED, not implied: the rule and its label travel with
    # the counter so a reader cannot mistake the sample for a different one.
    assert "ALL_IN_100_PCT" in prog["eligibility"]
    assert "QUOTE_SIMULATED_FILL" in prog["eligibility"]
    assert "10%-of-equity" in prog["excludes"]
    # and it is on the stats surface beside the arms
    stats = exp.stats()
    assert stats["forward_experiment"]["unique_parent_signals"] == 0
    assert stats["forward_experiment"]["performance_claim"] is False
    assert stats["forward_experiment"]["label"].startswith("PAPER ONLY")


# ---------------------------------------------------------------------------
# 2b. the INCIDENT-RECOVERY cohort: delayed settlements are excluded
#
# The live case these pin: three V2 BETA arms were opened on one parent signal
# and their exits were due at a 120s deadline. The pinned-SELL pair-unwrapping
# defect refused the quote on every attempt -- 1374/1374/1368 of them -- so all
# three stayed UNRESOLVED for three hours and were finally settled by the
# corrected code at -79% each. Those are NOT forward results: the exit rule did
# not produce them, an incident delayed them until the price had collapsed.
# ---------------------------------------------------------------------------

def test_a_settlement_spanning_a_restart_is_INCIDENT_RECOVERY(tmp_path):
    """A stranded arm settled after a restart is excluded from the clean cohort."""
    from test_v2_floor_beta import beta_engine

    # (1) the exit is due and the quote is refused: the arm is STRANDED.
    eng = beta_engine(tmp_path, quotes=FakeQuoteService(
        sell=unavailable_point(side="SELL")))
    run(eng.on_signal(eligible_signal()))
    pid = _open_and_trigger(eng)
    run(eng._monitor_beta_experiment())
    exp = eng.beta_experiment
    assert _beta_arm(exp, pid, "A").status == "unresolved"
    assert _beta_arm(exp, pid, "A").exit_attempts >= 1
    assert exp.forward_progress()["completed_trades"] == 0

    # (2) A RESTART: the arm is loaded still unresolved, with its history.
    revived = BetaExperiment(state_root=tmp_path)
    assert _beta_arm(revived, pid, "A").status == "unresolved"

    # (3) the quote arrives LATER, in the new process.
    revived.settle_arm(pid, "A", now=T0 + 600,
                       exit_fill=_exit_at_multiple(revived, pid, "A", 0.70),
                       require_quote=True)
    rec = [t for t in revived.completed if t["arm"] == "A"][0]
    assert rec["status"] == "closed"
    assert rec["cohort"] == INCIDENT_RECOVERY_COHORT
    assert rec["incident_recovery"]["exit_attempts_at_settlement"] >= 1
    assert rec["incident_recovery"]["exit_attempts_at_restore"] >= 1
    assert rec["incident_recovery"]["restored_unresolved_at"] > 0

    # the MAIN store keeps the settlement IN FULL, exactly once
    main = _read(tmp_path / "v2_floor_beta_state.json")
    stored = [t for t in main["trades"] if t["arm"] == "A"]
    assert len(stored) == 1
    assert stored[0]["net_pnl_usd"] == rec["net_pnl_usd"]

    # ... and the CLEAN forward cohort does not contain it.
    fwd = _read(_forward_file(tmp_path))
    assert fwd["trades"] == [], "a delayed settlement entered the clean cohort"
    assert fwd["counts"]["trades"] == 0
    assert fwd["counts"]["incident_recovery"] == 1
    prog = revived.forward_progress()
    assert prog["completed_trades"] == 0
    assert prog["by_arm"]["A"]["completed_trades"] == 0
    # The PARENT is still PROGRESS -- B and C are outstanding -- but NONE of its
    # arms is a settled result, which is the fact the cohort must not lose.
    assert prog["unique_parent_signals"] == 1
    assert prog["parents_with_all_three_arms_settled"] == 0
    assert all(g["arms"][a] is None
               for g in prog["grouped_by_parent"] for a in ("A", "B", "C"))
    assert prog["incident_recovery_trades"] == 1
    assert prog["incident_recovery_parents"] == 1
    assert prog["incident_recovery_records"][0]["arm"] == "A"
    assert prog["incident_recovery_records"][0]["parent_signal_id"] == pid
    assert "DELAYED" in prog["excludes"] or "EXCLUDED" in prog["excludes"]
    assert prog["performance_claim"] is False


def test_a_settlement_on_schedule_stays_in_the_clean_cohort(tmp_path):
    """The control: excluding delayed settlements must not exclude everything."""
    from test_v2_floor_beta import beta_engine

    eng = beta_engine(tmp_path, quotes=SizingQuoteService())
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    pos = _beta_arm(exp, pid, "A")
    fill = _exit_at_multiple(exp, pid, "A", 1.30)
    exp.observe_quote(pid, "A", exit_fill=fill, mark_price=None, at=T0 + 5)
    exp.settle_arm(pid, "A", now=T0 + 6, exit_fill=fill, require_quote=True)
    rec = exp.completed[0]
    assert rec["status"] == "closed"
    assert rec["exit_attempts"] == 0
    assert rec["cohort"] == FORWARD_COHORT
    assert rec["incident_recovery"] is None
    prog = exp.forward_progress()
    assert prog["completed_trades"] == 1
    assert prog["unique_parent_signals"] == 1
    assert prog["incident_recovery_trades"] == 0
    fwd = _read(_forward_file(tmp_path))
    assert len(fwd["trades"]) == 1 and fwd["incident_recovery"] == []


def test_a_delayed_settlement_already_in_the_cohort_is_moved_out(tmp_path):
    """The three live LARP settlements predate this rule; the sweep still holds.

    They were written into the clean cohort by the OLD classification -- no
    ``cohort`` label at all -- so a rule that only labelled NEW settlements
    would leave three incident-recovery trades counted as forward results
    forever. This is that shape, reproduced: the only evidence is the record's
    own ``exit_attempts``.
    """
    from test_v2_floor_beta import beta_engine

    eng = beta_engine(tmp_path, quotes=SizingQuoteService())
    run(eng.on_signal(eligible_signal()))
    pid = next(iter(eng.beta_experiment.positions))
    exp = eng.beta_experiment
    pos = _beta_arm(exp, pid, "A")
    fill = _exit_at_multiple(exp, pid, "A", 1.30)
    exp.observe_quote(pid, "A", exit_fill=fill, mark_price=None, at=T0 + 5)
    exp.settle_arm(pid, "A", now=T0 + 6, exit_fill=fill, require_quote=True)
    path = _forward_file(tmp_path)
    assert len(_read(path)["trades"]) == 1, "the control must enter the cohort"

    # Strip the NEW label from BOTH files -- main store and forward mirror, as
    # an older build would have written them -- and leave the retry count that
    # is the only durable trace of the delay.
    for target in (tmp_path / "v2_floor_beta_state.json", path):
        payload = _read(target)
        for rec in payload["trades"]:
            if rec.get("arm") == "A":
                rec.pop("cohort", None)
                rec.pop("incident_recovery", None)
                rec["exit_attempts"] = 42
        target.write_text(json.dumps(payload, indent=2))

    # A restart, and one more persist: the record must LEAVE the clean cohort
    # and be PRESERVED in the incident section -- not deleted.
    revived = BetaExperiment(state_root=tmp_path)
    revived._persist_forward()
    after = _read(path)
    assert after["trades"] == [], "a delayed record stayed in the clean cohort"
    assert after["counts"]["trades"] == 0
    assert len(after["incident_recovery"]) == 1
    assert after["incident_recovery"][0]["exit_attempts"] == 42
    assert after["incident_recovery"][0]["net_pnl_usd"] is not None
    main = _read(tmp_path / "v2_floor_beta_state.json")
    assert len([t for t in main["trades"] if t["arm"] == "A"]) == 1, (
        "the settlement must be preserved in the main store, not removed")


def test_the_by_cost_model_split_separates_modelled_from_legacy(tmp_path):
    from test_v2_floor_beta import beta_engine

    eng = beta_engine(tmp_path, quotes=SizingQuoteService())
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    pos = _beta_arm(exp, pid, "A")
    fill = _exit_at_multiple(exp, pid, "A", 1.30)
    exp.observe_quote(pid, "A", exit_fill=fill, mark_price=None, at=T0 + 5)
    exp.settle_arm(pid, "A", now=T0 + 6, exit_fill=fill, require_quote=True)
    prog = exp.forward_progress()
    assert prog["by_cost_model"][COST_MODEL_MODELED]["completed_trades"] == 1
    assert prog["by_cost_model"][COST_MODEL_LEGACY]["completed_trades"] == 0


# ---------------------------------------------------------------------------
# 3. cost-model provenance: derived vs configured vs assumed, and failures
# ---------------------------------------------------------------------------

def test_the_base_fee_is_DERIVED_from_the_intended_transaction_config():
    from autopaper.paper_live.execution_costs import (
        SOLANA_BASE_FEE_LAMPORTS,
        intended_transaction_config,
        outside_swap_costs,
    )

    cfg = intended_transaction_config()
    assert cfg["fee_payer_signatures_per_leg"] == 1
    assert cfg["priority_fee_is_a_ceiling"] is True
    assert "RealConfig" in cfg["source"]
    c = outside_swap_costs(sol_usd=100.0)
    assert c.signatures_per_leg == cfg["fee_payer_signatures_per_leg"]
    assert c.base_fee_lamports == (SOLANA_BASE_FEE_LAMPORTS
                                   * c.signatures_per_leg * c.legs)
    assert c.provenance["base_fee_lamports"].startswith(COST_DERIVED)
    assert "signature" in c.provenance["base_fee_lamports"]
    d = c.as_dict()
    assert d["priority_fee_is_ceiling"] is True
    assert d["priority_fee_observed"] is False
    assert "CEILING" in d["observed_note"]
    assert d["derived_from"] == cfg["source"]


def test_the_priority_fee_is_a_CONFIGURED_CEILING_not_an_observed_fee():
    from autopaper.paper_live.execution_costs import (
        PAPER_PRIORITY_FEE_LAMPORTS,
        PRIORITY_FEE_SOURCE_CEILING,
        outside_swap_costs,
    )

    c = outside_swap_costs(sol_usd=100.0)
    assert c.priority_fee_lamports == PAPER_PRIORITY_FEE_LAMPORTS * 2
    assert c.priority_fee_source == PRIORITY_FEE_SOURCE_CEILING
    prov = c.provenance["priority_fee_lamports"]
    assert prov.startswith(COST_CONFIGURED)
    assert "NOT measured" in prov
    assert "CEILING" in prov
    # the rent stays ASSUMED, and a NEW account is the conservative case
    assert c.provenance["rent_lamports"].startswith(COST_ASSUMED)
    assert c.assumes_new_token_account is True


def test_a_refused_QUOTE_REQUEST_costs_no_network_fee_but_a_broadcast_would():
    from autopaper.paper_live.execution_costs import (
        failed_execution_accounting,
        outside_swap_costs,
    )

    f = failed_execution_accounting(quote_failures=7, broadcast_attempts=0)
    assert f["quote_failures"] == 7
    assert f["quote_failure_network_fee_lamports"] == 0
    assert "NO network fee" in f["quote_failure_note"]
    assert f["broadcast_attempts"] == 0
    assert f["broadcast_failure_fee_lamports"] == 0
    # the arithmetic for the OTHER case is stated even though PAPER has none
    f2 = failed_execution_accounting(
        quote_failures=0, broadcast_attempts=2, sol_usd=100.0)
    per = f2["per_failed_broadcast_lamports"]
    c = outside_swap_costs(sol_usd=100.0)
    assert per == c.non_recoverable_lamports // c.legs
    assert f2["broadcast_failure_fee_lamports"] == per * 2
    assert f2["broadcast_failure_fee_usd"] == pytest.approx(per * 2 / 1e9 * 100.0)


def test_the_slippage_boundary_states_the_intended_config_and_that_paper_sets_none():
    from autopaper.paper_live.execution_costs import slippage_boundary

    s = slippage_boundary(measured_price_impact_pct=7.5)
    assert s["intended_slippage_bps"] == 500
    assert s["intended_slippage_pct"] == 5.0
    assert s["applied_by_paper"] is False
    assert "NO transaction" in s["applied_by_paper_note"]
    # a MEASURED impact is checked against the limit that WOULD have applied
    assert s["measured_price_impact_pct"] == 7.5
    assert s["measured_impact_would_breach_limit"] is True
    assert s["measured_impact_provenance"].startswith("MEASURED")
    assert s["not_modelled"]


def test_cost_coverage_on_the_experiment_separates_every_provenance_kind(tmp_path):
    exp = new_exp(tmp_path)
    cov = exp.cost_coverage()
    assert cov["model"] == "OUTSIDE_SWAP_COSTS_MODELED"
    assert cov["provenance"]["base_fee"].startswith("DERIVED")
    assert cov["provenance"]["priority_fee"].startswith("CONFIGURED")
    assert cov["provenance"]["rent"].startswith("ASSUMED")
    assert cov["measured"]
    fe = cov["failed_execution"]
    assert fe["quote_failure_network_fee_lamports"] == 0
    assert fe["broadcast_attempts"] == 0
    assert cov["slippage"]["applied_by_paper"] is False
    # the intended transaction configuration travels WITH the coverage
    assert cov["intended_transaction"]["fee_payer_signatures_per_leg"] == 1
    # ... and it is on the stats surface too
    assert exp.stats()["cost_coverage"]["intended_transaction"]["slippage_bps"] == 500
