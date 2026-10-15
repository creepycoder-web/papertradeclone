"""Focused tests for the OUTSIDE-SWAP cost model and ALL-IN deployable sizing.

WHAT THESE PROVE
================
That the V2 FLOOR (BETA) PAPER experiment costs a round trip BESIDE the swap
quote, not only inside it, and that it does so without ever over-committing:

* an ALL-IN arm commits 100% of its available cash, but the SWAP INPUT it
  deploys is that MINUS the Solana base fee, the priority fee and the
  token-account rent -- costs a wallet must hold or the BUY cannot execute;
* the NON-RECOVERABLE part (both legs' fees) is CHARGED to P&L at settlement,
  and the RECOVERABLE deposit (the rent) is only RESERVED, never charged;
* a missing SOL/USD reference makes the reserve UNKNOWN, and an unknown reserve
  refuses the entry rather than being silently treated as zero;
* every cost carries a PROVENANCE label, and every fill carries the quote
  provenance needed to recompute it -- including an explicit statement that
  PAPER sets no slippage tolerance or minimum-output threshold, because it
  builds and submits no transaction;
* the cost model survives a restart and is reported per arm.

Everything is OFFLINE with isolated temporary state roots: no network, no
Telegram, no RPC, no real ``var/`` state, nothing signed or broadcast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_paper_live import FakeQuoteService, SOL_USD, run  # noqa: E402
from test_v2_floor_beta import (  # noqa: E402
    T0,
    beta_engine,
    eligible_signal,
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
    SOLANA_BASE_FEE_LAMPORTS,
    OutsideSwapCosts,
    deployable_sol_lamports,
    outside_swap_costs,
)
from autopaper.paper_live.v2_floor_beta import (  # noqa: E402
    COST_MODEL_LEGACY,
    COST_MODEL_MODELED,
    BetaExperiment,
    cost_model_bucket,
)
from autopaper.paper_live.quotes import SolReference  # noqa: E402

pytestmark = [pytest.mark.paper_live, pytest.mark.strategy]


# ---------------------------------------------------------------------------
# 1. the cost model itself: components, provenance, split
# ---------------------------------------------------------------------------

def test_the_round_trip_charges_both_legs_of_both_fees_and_the_rent():
    c = outside_swap_costs(sol_usd=100.0, sol_usd_source="unit")
    assert c.known is True
    assert c.legs == 2
    # DERIVED: the protocol's base fee, once per leg.
    assert c.base_fee_lamports == SOLANA_BASE_FEE_LAMPORTS * 2 == 10_000
    # CONFIGURED: the project's own REAL priority fee, once per leg.
    assert c.priority_fee_lamports == PAPER_PRIORITY_FEE_LAMPORTS * 2 == 100_000
    # The rent is a RECOVERABLE deposit and is reported as such.
    assert c.recoverable_lamports > 2_000_000
    assert c.non_recoverable_lamports == 110_000
    assert c.total_reserve_lamports == (c.non_recoverable_lamports
                                       + c.recoverable_lamports)


def test_every_cost_carries_a_provenance_label():
    c = outside_swap_costs(sol_usd=100.0)
    prov = c.provenance
    assert prov["base_fee_lamports"].startswith(COST_DERIVED)
    assert prov["priority_fee_lamports"].startswith(COST_CONFIGURED)
    assert "NOT measured" in prov["priority_fee_lamports"]
    assert prov["rent_lamports"].startswith(COST_ASSUMED)
    assert "conservative" in prov["rent_lamports"]
    # ... and the assumption is stated as a flag too, not only in prose.
    assert c.assumes_new_token_account is True


def test_only_the_non_recoverable_part_is_charged_to_pnl():
    c = outside_swap_costs(sol_usd=100.0)
    assert c.charged_to_pnl_usd == pytest.approx(
        (110_000 / 1e9) * 100.0, abs=1e-12)
    assert c.recoverable_usd == pytest.approx((c.rent_lamports / 1e9) * 100.0,
                                             abs=1e-12)
    # the charge is strictly less than the reserve: the rent is not a loss.
    assert c.charged_to_pnl_usd < c.total_reserve_usd
    assert c.as_dict()["recoverable_deposit_not_a_loss"] is True


def test_a_missing_sol_usd_reference_is_UNKNOWN_not_zero():
    c = outside_swap_costs(sol_usd=None)
    assert c.known is False
    assert c.charged_to_pnl_usd == 0.0
    assert c.total_reserve_usd is None
    assert "UNKNOWN" in c.reason
    # ... and the lamport components are still counted, so the record shows
    # WHAT could not be valued rather than an empty dict.
    assert c.total_reserve_lamports == 110_000 + c.rent_lamports


def test_deployable_lamports_hold_the_reserve_back_and_floor():
    c = outside_swap_costs(sol_usd=100.0)
    lamports, why = deployable_sol_lamports(available_usd=10.0, sol_usd=100.0,
                                            costs=c)
    assert why == ""
    assert lamports == int(10.0 / 100.0 * 1e9) - c.total_reserve_lamports
    # FLOOR, never round up: rounding up would ask for more SOL than is held.
    assert lamports < int(10.0 / 100.0 * 1e9)


def test_deployable_is_refused_when_the_budget_cannot_cover_the_reserve():
    c = outside_swap_costs(sol_usd=100.0)
    # $0.20 buys 2,000,000 lamports, which does not cover a ~2.15M reserve.
    lamports, why = deployable_sol_lamports(available_usd=0.20, sol_usd=100.0,
                                           costs=c)
    assert lamports == 0
    assert "outside-swap costs" in why


def test_deployable_is_refused_without_a_sol_reference():
    c = outside_swap_costs(sol_usd=None)
    lamports, why = deployable_sol_lamports(available_usd=10.0, sol_usd=None,
                                           costs=c)
    assert lamports == 0
    assert "no SOL/USD reference" in why


# ---------------------------------------------------------------------------
# 2. the engine: the entry is sized on DEPLOYABLE cash and refuses to fail open
# ---------------------------------------------------------------------------

def _costs_at_engine_reference():
    return outside_swap_costs(sol_usd=SOL_USD, sol_usd_source="test")


def test_an_entry_deploys_available_cash_less_the_outside_swap_reserve(tmp_path):
    qs = SizingQuoteService()
    eng = beta_engine(tmp_path, quotes=qs)
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    costs = _costs_at_engine_reference()
    pos = _beta_arm(exp, pid, "A")
    assert pos.allocated_usd == pytest.approx(10.0, abs=1e-9)
    assert pos.deployable_usd == pytest.approx(
        10.0 - costs.total_reserve_usd, abs=1e-6)
    assert pos.entry_cost_usd == pytest.approx(pos.deployable_usd, abs=1e-9)
    assert pos.outside_swap_cost_usd == pytest.approx(
        costs.charged_to_pnl_usd, abs=1e-12)
    assert pos.outside_swap_costs["model"] == "OUTSIDE_SWAP_COSTS_MODELED"
    # the requested quote input is the DEPLOYABLE amount, in lamports.
    gross = int(10.0 / SOL_USD * 1_000_000_000)
    assert qs.buy_calls[0]["sol_lamports"] == (
        gross - int(costs.total_reserve_lamports))


def test_an_arm_below_the_reserve_deploys_nothing_and_spends_no_quote(tmp_path):
    """A tiny arm fails CLOSED: no quote is requested for it at all."""
    qs = SizingQuoteService()
    eng = beta_engine(tmp_path, quotes=qs)
    exp = eng.beta_experiment
    exp.accounts["A"].bankroll_usd = 0.20   # below the ~$0.26 reserve
    before = len(qs.buy_calls)
    run(eng.on_signal(eligible_signal()))
    pid = next(iter(exp.positions))
    assert "A" not in exp.positions[pid]
    assert set(exp.positions[pid]) == {"B", "C"}
    # B and C coalesce onto ONE shared quote; A's refusal cost none.
    assert len(qs.buy_calls) - before == 1


def test_no_sol_reference_refuses_the_entry_with_a_reason(tmp_path):
    eng = beta_engine(tmp_path, quotes=SizingQuoteService())
    eng.sol_reference = SolReference(0.0, T0, "none")
    records = run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    assert exp.positions == {}
    assert records and all(r.traded is False for r in records)


# ---------------------------------------------------------------------------
# 3. settlement: the fees are charged, the rent is not
# ---------------------------------------------------------------------------

def test_settlement_charges_the_fees_and_never_the_rent(tmp_path):
    eng = beta_engine(tmp_path, quotes=SizingQuoteService())
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    a = _beta_arm(exp, pid, "A")
    costs = _costs_at_engine_reference()
    _fill = _exit_at_multiple(exp, pid, "A", 1.30)
    exp.observe_quote(pid, "A", exit_fill=_fill, at=T0 + 5)
    # THE MARK IS STILL OBSERVED AND STILL DIAGNOSTIC-ONLY: it cannot
    # fire the bracket any more, so recording it here proves the two
    # bases are actually separate rather than merely relabelled.
    exp.observe(pid, "A", mark_price=a.entry_fill_price * 1.31, at=T0 + 5)
    exp.settle_arm(pid, "A", now=T0 + 6,
                   exit_fill=_fill,
                   require_quote=True)
    assert a.status == "closed"
    # trading P&L is the quote-only number; net P&L is it minus the charge.
    assert a.trading_pnl_usd == pytest.approx(
        a.position_size_usd * a.net_return_pct / 100.0, abs=1e-9)
    assert a.net_pnl_usd == pytest.approx(
        a.trading_pnl_usd - costs.charged_to_pnl_usd, abs=1e-9)
    # the rent was NEVER charged: the difference is exactly the fee charge.
    assert a.trading_pnl_usd - a.net_pnl_usd == pytest.approx(
        costs.charged_to_pnl_usd, abs=1e-9)
    assert (a.trading_pnl_usd - a.net_pnl_usd) < costs.total_reserve_usd
    # ... and the bankroll moved by the FULLY-COSTED amount.
    assert exp.accounts["A"].bankroll_usd == pytest.approx(
        10.0 + a.net_pnl_usd, abs=1e-9)


def test_the_charge_is_recorded_on_the_trade_record(tmp_path):
    eng = beta_engine(tmp_path, quotes=SizingQuoteService())
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    a = _beta_arm(exp, pid, "A")
    _fill = _exit_at_multiple(exp, pid, "A", 1.30)
    exp.observe_quote(pid, "A", exit_fill=_fill, at=T0 + 5)
    # THE MARK IS STILL OBSERVED AND STILL DIAGNOSTIC-ONLY: it cannot
    # fire the bracket any more, so recording it here proves the two
    # bases are actually separate rather than merely relabelled.
    exp.observe(pid, "A", mark_price=a.entry_fill_price * 1.31, at=T0 + 5)
    exp.settle_arm(pid, "A", now=T0 + 6,
                   exit_fill=_fill,
                   require_quote=True)
    rec = exp.completed[0]
    costs = rec["outside_swap_costs"]
    assert costs["model"] == "OUTSIDE_SWAP_COSTS_MODELED"
    assert costs["known"] is True
    assert costs["charged_to_pnl_usd"] == pytest.approx(
        rec["outside_swap_charged_usd"], abs=1e-12)
    assert rec["trading_pnl_usd"] == pytest.approx(
        rec["net_pnl_usd"] + rec["outside_swap_charged_usd"], abs=1e-9)
    assert rec["deployable_usd"] == pytest.approx(
        rec["entry_cost_usd"], abs=1e-9)
    assert rec["reserved_outside_swap_usd"] > rec["outside_swap_charged_usd"]
    assert cost_model_bucket(rec) == COST_MODEL_MODELED


# ---------------------------------------------------------------------------
# 4. provenance: the record can be recomputed from its own receipts
# ---------------------------------------------------------------------------

def test_the_entry_record_carries_full_quote_provenance(tmp_path):
    eng = beta_engine(tmp_path, quotes=SizingQuoteService())
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    prov = _beta_arm(exp, pid, "A").entry_provenance
    # amounts, both legs and their decimals
    assert prov["side"] == "BUY"
    assert prov["requested_input_raw"] > 0
    assert prov["input_amount_raw"] == prov["requested_input_raw"]
    assert prov["output_amount_raw"] > 0
    assert prov["input_decimals"] == 9 and prov["output_decimals"] == 6
    # market identity and route
    assert prov["token_mint"]
    assert prov["quote_mint"]
    assert prov["venue"] or prov["venue_id"]
    assert isinstance(prov["route"], list)
    # the SOL/USD reference the entry was sized with, and its source+timestamp
    assert prov["sol_usd"] == pytest.approx(SOL_USD)
    assert prov["sol_usd_source"] == "test"
    assert prov["sol_usd_at"] is not None
    # MEASURED request latency, not an assumed one
    assert prov["measured_request_ms"] is not None
    assert prov["measured_request_ms"] >= 0.0
    # the reserve and the charge, restated on the provenance itself
    assert prov["reserved_outside_swap_lamports"] > 0
    assert prov["charged_outside_swap_usd"] > 0.0


def test_the_provenance_states_that_paper_sets_no_price_boundary(tmp_path):
    eng = beta_engine(tmp_path, quotes=SizingQuoteService())
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    prov = _beta_arm(exp, pid, "A").entry_provenance
    # NOT RECORDED, and said so -- never silently defaulted to a plausible 0.
    assert prov["slippage_bps"] is None
    assert prov["minimum_output_threshold"] is None
    assert "NO transaction" in prov["price_boundary_note"]
    assert "EXACT size" in prov["price_boundary_note"]
    # the trigger price is labelled diagnostic, not the fill price.
    assert "NOT the price this fill was booked at" in prov["trigger_role"]


def test_the_exit_record_carries_sell_provenance_for_the_exact_quantity(tmp_path):
    """The ENGINE's exit path is what records the SELL provenance."""
    eng = beta_engine(tmp_path, quotes=SizingQuoteService())
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    a = _beta_arm(exp, pid, "A")
    # observe the +30% target so the next monitor pass exits from a real quote.
    exp.observe(pid, "A", mark_price=a.entry_fill_price * 1.31, at=T0 + 5)
    run(eng._monitor_beta_experiment())
    assert a.status == "closed"
    prov = exp.completed[0]["exit_provenance"]
    assert prov["side"] == "SELL"
    # the SELL input is the EXACT acquired TOKEN quantity, not a SOL figure.
    assert prov["requested_input_raw"] == a.entry_tokens_raw
    assert prov["measured_request_ms"] is not None
    assert prov["minimum_output_threshold"] is None


# ---------------------------------------------------------------------------
# 5. persistence and reporting
# ---------------------------------------------------------------------------

def test_the_cost_model_survives_a_restart(tmp_path):
    eng = beta_engine(tmp_path, quotes=SizingQuoteService())
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    costs = _costs_at_engine_reference()
    revived = BetaExperiment(state_root=tmp_path)
    pos = _beta_arm(revived, pid, "A")
    assert pos.status == "open"
    assert pos.outside_swap_costs["model"] == "OUTSIDE_SWAP_COSTS_MODELED"
    assert pos.outside_swap_cost_usd == pytest.approx(
        costs.charged_to_pnl_usd, abs=1e-12)
    assert pos.reserved_outside_swap_usd == pytest.approx(
        costs.total_reserve_usd, abs=1e-6)
    assert pos.entry_provenance["side"] == "BUY"
    # ... and the restored arm is charged the SAME amount when it settles.
    _fill = _exit_at_multiple(revived, pid, "A", 1.30)
    revived.observe_quote(pid, "A", exit_fill=_fill, at=T0 + 5)
    # THE MARK IS STILL OBSERVED AND STILL DIAGNOSTIC-ONLY: it cannot
    # fire the bracket any more, so recording it here proves the two
    # bases are actually separate rather than merely relabelled.
    revived.observe(pid, "A", mark_price=pos.entry_fill_price * 1.31, at=T0 + 5)
    revived.settle_arm(pid, "A", now=T0 + 6,
                   exit_fill=_fill,
                       require_quote=True)
    assert pos.net_pnl_usd == pytest.approx(
        pos.trading_pnl_usd - costs.charged_to_pnl_usd, abs=1e-9)


def test_settlement_does_not_charge_the_fees_twice(tmp_path):
    eng = beta_engine(tmp_path, quotes=SizingQuoteService())
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    a = _beta_arm(exp, pid, "A")
    _fill = _exit_at_multiple(exp, pid, "A", 1.30)
    exp.observe_quote(pid, "A", exit_fill=_fill, at=T0 + 5)
    # THE MARK IS STILL OBSERVED AND STILL DIAGNOSTIC-ONLY: it cannot
    # fire the bracket any more, so recording it here proves the two
    # bases are actually separate rather than merely relabelled.
    exp.observe(pid, "A", mark_price=a.entry_fill_price * 1.31, at=T0 + 5)
    exp.settle_arm(pid, "A", now=T0 + 6,
                   exit_fill=_fill,
                   require_quote=True)
    bankroll = exp.accounts["A"].bankroll_usd
    charged = a.net_pnl_usd
    # re-settling the SAME arm books nothing further.
    exp.settle_arm(pid, "A", now=T0 + 7,
                   exit_fill=_exit_at_multiple(exp, pid, "A", 1.30),
                   require_quote=True)
    exp.monitor_once(now=T0 + 8)
    assert exp.accounts["A"].bankroll_usd == bankroll
    assert a.net_pnl_usd == charged
    assert len([r for r in exp.completed if r["arm"] == "A"]) == 1


def test_the_report_splits_by_cost_model_and_surfaces_the_fee_breakdown(tmp_path):
    eng = beta_engine(tmp_path, quotes=SizingQuoteService())
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    a = _beta_arm(exp, pid, "A")
    _fill = _exit_at_multiple(exp, pid, "A", 1.30)
    exp.observe_quote(pid, "A", exit_fill=_fill, at=T0 + 5)
    # THE MARK IS STILL OBSERVED AND STILL DIAGNOSTIC-ONLY: it cannot
    # fire the bracket any more, so recording it here proves the two
    # bases are actually separate rather than merely relabelled.
    exp.observe(pid, "A", mark_price=a.entry_fill_price * 1.31, at=T0 + 5)
    exp.settle_arm(pid, "A", now=T0 + 6,
                   exit_fill=_fill,
                   require_quote=True)
    stats = exp.stats()
    assert stats["cost_coverage"]["model"] == "OUTSIDE_SWAP_COSTS_MODELED"
    assert "non-recoverable" in stats["cost_coverage"]["charged_to_pnl"]
    assert "recoverable" in stats["cost_coverage"]["reserved_not_charged"]
    # the top-level model splits by what was actually charged.
    split = stats["provenance"]["by_cost_model"]
    assert split[COST_MODEL_MODELED]["completed_trades"] == 1
    assert split[COST_MODEL_MODELED]["outside_swap_charged_usd"] > 0.0
    assert split[COST_MODEL_LEGACY]["completed_trades"] == 0
    # and the per-arm fee breakdown is on the arm.
    arm = stats["arms"]["A"]
    assert arm["outside_swap_cost_model"][COST_MODEL_MODELED] == 1
    assert arm["outside_swap_fees_lamports"] == 110_000
    assert arm["outside_swap_rent_reserved_lamports"] > 2_000_000
    assert arm["outside_swap_rent_recoverable"] is True
    # the arm's charged total IS the fee charge for its one closed trade (the
    # report rounds to 6dp, so its tolerance is that rounding, not more).
    assert arm["outside_swap_costs_usd"] == pytest.approx(
        _costs_at_engine_reference().charged_to_pnl_usd, abs=1e-6)
    assert arm["quote_telemetry"]["priced_quotes"] >= 1
    assert "legacy" in arm["outside_swap_cost_model_note"]


def test_a_legacy_record_is_labelled_legacy_not_costed(tmp_path):
    """A record with no cost model must never look fully costed."""
    assert cost_model_bucket({"net_pnl_usd": 1.0}) == COST_MODEL_LEGACY
    assert cost_model_bucket({}) == COST_MODEL_LEGACY
    assert cost_model_bucket({"outside_swap_costs": {}}) == COST_MODEL_LEGACY
    assert cost_model_bucket(
        {"outside_swap_costs": {"model": "OUTSIDE_SWAP_COSTS_MODELED"}}
    ) == COST_MODEL_MODELED


def test_a_fixture_fill_without_a_cost_model_charges_nothing(tmp_path):
    """The explicit/fixture path has no outside-swap model: it is LEGACY, and
    its net P&L is therefore the quote-only number, unchanged."""
    from test_v2_floor_beta import new_exp, open_p1
    from test_v2_floor_beta_integration import _exit_at_multiple

    exp = new_exp(tmp_path)
    open_p1(exp)
    pos = _beta_arm(exp, "p1", "A")
    assert pos.outside_swap_costs == {}
    assert pos.outside_swap_cost_usd is None
    _fill = _exit_at_multiple(exp, "p1", "A", 1.30)
    exp.observe_quote("p1", "A", exit_fill=_fill, at=T0 + 5)
    # THE MARK IS STILL OBSERVED AND STILL DIAGNOSTIC-ONLY: it cannot
    # fire the bracket any more, so recording it here proves the two
    # bases are actually separate rather than merely relabelled.
    exp.observe("p1", "A", mark_price=pos.entry_fill_price * 1.31, at=T0 + 5)
    exp.settle_arm("p1", "A", now=T0 + 6,
                   exit_fill=_fill,
                   require_quote=True)
    assert pos.net_pnl_usd == pytest.approx(pos.trading_pnl_usd, abs=1e-12)
    assert exp.completed[0]["outside_swap_costs"] == {}
    assert cost_model_bucket(exp.completed[0]) == COST_MODEL_LEGACY


def test_the_aggregate_shared_liquidity_view_is_separate_and_moves_no_ledger(
        tmp_path):
    """A/B/C combined demand is reported as a STRESS VIEW, not a result.

    The arms are independent ledgers, so their COMBINED demand -- the size the
    pool would actually absorb if all three acted on the same mint at once -- is
    reported separately, is labelled a stress view, and moves no ledger.
    """
    # (a) an arm holding a position is blocked, so it contributes no demand.
    eng = beta_engine(tmp_path / "held", quotes=SizingQuoteService())
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    before = {a: exp.accounts[a].bankroll_usd for a in "ABC"}
    plan = exp.arm_entry_plan()
    scen = exp.shared_liquidity_scenario()
    assert scen["arms_free"] == [] and set(scen["arms_blocked"]) == {"A", "B", "C"}
    assert scen["aggregate_available_usd"] == pytest.approx(
        sum(plan[a]["available_cash_usd"] for a in "ABC"), abs=1e-9)
    assert scen["is_a_result"] is False
    assert "STRESS VIEW, NOT A RESULT" in scen["note"]
    assert "UNDERSTATE" in scen["note"]
    # reporting it moved no ledger and opened nothing.
    assert {a: exp.accounts[a].bankroll_usd for a in "ABC"} == before

    # (b) with all three arms free the aggregate is THREE TIMES one arm, and it
    #     is still labelled a stress view -- never folded into a per-arm number.
    eng2 = beta_engine(tmp_path / "free", quotes=SizingQuoteService())
    exp2 = eng2.beta_experiment
    scen2 = exp2.shared_liquidity_scenario()
    assert scen2["arms_free"] == ["A", "B", "C"]
    assert scen2["aggregate_multiple_of_one_arm"] == pytest.approx(3.0, abs=1e-9)
    assert scen2["simultaneous_available_usd"] == pytest.approx(
        scen2["per_arm_available_usd"] * 3, abs=1e-9)
    # ... and it is on the stats surface, clearly separated from the arms.
    stats = exp2.stats()
    assert "shared_liquidity_scenario" in stats
    assert stats["shared_liquidity_scenario"]["is_a_result"] is False
    assert stats["shared_liquidity_scenario"]["arms_free"] == ["A", "B", "C"]


def test_no_transaction_is_ever_built_signed_or_submitted(tmp_path, monkeypatch):
    """PAPER ONLY: the entry path must not reach a send/submit/broadcast API."""
    import autopaper.paper_live.engine as engine_mod
    for name in ("sendTransaction", "sign_transaction", "send_raw_transaction",
                 "submit", "broadcast", "send_transaction"):
        assert not hasattr(engine_mod.PaperLiveEngine, name), name
    eng = beta_engine(tmp_path, quotes=SizingQuoteService())
    run(eng.on_signal(eligible_signal()))
    run(eng._monitor_beta_experiment())
    assert eng.beta_experiment is not None
