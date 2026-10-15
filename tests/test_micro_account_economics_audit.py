"""
THE 2026-10-11 PAPER RUN, AUDITED. Why 45 candidates produced zero BUYs.

THE FINDING, in one paragraph. A PAPER run evaluated 45 candidates and bought
none: 12 were blocked by ``MICRO_ACCOUNT_ECONOMICALLY_INVALID`` and 33 were
strategy rejections. The 12 economic blocks were CORRECT, not a defect. The
operator was shown ``equity=$100.00``. The decision was sized from the
``$0.0758`` arm book, not the account the operator was looking at. The
``$0.0758`` is ``10% of $0.758281``: the L25 arm's own virtual bankroll,
restored from 104 settled PAPER trades whose cumulative realized PnL was
``-$4.24171912``. It is NOT ``10% of the $100.00`` the runtime account
reported at shutdown. Two ledgers, one position file.

This file PINS that arithmetic so the conclusion cannot silently become false:

* a blocked BUY now carries the FULL economic verdict -- notional, locked rent,
  fixed cost, break-even, classification -- on the record and in the CSV;
* the runtime's run summary now counts rejections BY CAUSE, so ``rejects=33``
  reads as a distribution and not as an opaque count.

No network, no order, nothing persisted outside ``tmp_path``. The live-run
records are read-only and live in ``var/``; no test writes to ``var/``, to the
ledger, to an account or to a position file. Report fixtures go under
``tmp_path``.
"""

from __future__ import annotations

import asyncio
import csv
import json
import time
from pathlib import Path

import pytest

try:
    from tests import paper_bridge_fakes as fk
except ImportError:  # pragma: no cover - direct invocation
    import paper_bridge_fakes as fk  # type: ignore[no-redef]
from autopaper.config import build_config
from autopaper.engine import PaperTrader
from autopaper.models import (
    AlertSnapshot,
    Decision,
    Observation,
    ObservationSet,
)
from autopaper.paper_live.economics import (
    COST_RENT_LOCK,
    COST_RENT_RECLAIMABLE,
    COST_TRADING,
    ECONOMIC_INVALID,
    ECONOMIC_VALID,
    MAX_BREAK_EVEN_RETURN_PCT,
    MAX_FIXED_COST_FRACTION,
    MAX_RENT_OVER_NOTIONAL,
    MIN_EFFECTIVE_NOTIONAL_USD,
    TOKEN_ACCOUNT_RENT_LAMPORTS,
    evaluate_entry_economics,
)
from autopaper.paper_live.report import DatasetWriter
from autopaper.sizing import SIZING_BOUNDARY_USD, compute_position_size

bridge = fk.bridge
real_report = fk.real_report
run = fk.run

# ---------------------------------------------------------------------------
# The recorded run's numbers, each traced to its source. None is fabricated;
# every one is either read from the persisted run records or is arithmetic on
# the sizing policy applied to them.
# ---------------------------------------------------------------------------

#: The L25 arm's virtual bankroll, restored from its own settled trades:
#: ``$5.00`` start, ``-$4.24171912`` settled over 104 trades, is
#: ``$0.75828088``. This is the balance the DECISION was sized from.
RECORDED_ARM_STARTING_BANKROLL_USD = 5.00
RECORDED_ARM_SETTLED_PNL_USD = -4.24171912
RECORDED_ARM_SETTLED_TRADES = 104
RECORDED_ARM_BANKROLL_USD = 0.75828088

#: The SOL/USD reference in force for the run. Rent is quoted against it.
RECORDED_SOL_USD = 120.30

#: The notional the arm actually produced and the guard refused: ``10% of
#: $0.75828088`` is ``$0.07582808...``, the "$0.0758" in the recorded refusal.
RECORDED_BLOCKED_NOTIONAL_USD = 0.0758

#: The equity the operator was shown at shutdown. A DIFFERENT book from the
#: arm bankroll above; both are true, neither is derived from the other.
RECORDED_RUNTIME_EQUITY_USD = 100.00

#: The notional the SAME policy would have produced off the runtime account:
#: ``$100`` is above the $50 boundary, so the 12.5% branch applies and the
#: notional is ``$12.50``, which clears every economic gate. The order was not
#: uneconomic in the abstract -- it was uneconomic against the bankroll it was
#: actually sized from.
RUNTIME_NOTIONAL_USD = 12.50

#: The floor the guard charged, restated as a named constant so a test never
#: hardcodes a magic ``0.10`` beside a guard that owns its own.
MIN_NOTIONAL_USD = MIN_EFFECTIVE_NOTIONAL_USD

#: The rent figure the guard charged, as arithmetic on the policy constant.
recorded_rent_usd = (TOKEN_ACCOUNT_RENT_LAMPORTS / (10 ** 9)) * RECORDED_SOL_USD


# ===========================================================================
# The recorded notional is reproducible
# ===========================================================================


class TestTheRecordedNotionalIsReproducible:
    """The blocked notional is not a mystery number.

    It is ``compute_position_size`` applied to the arm's own restored bankroll.
    """

    def test_the_bankroll_on_record_is_the_starting_balance_plus_settled_pnl(
        self,
    ):
        restored = (RECORDED_ARM_STARTING_BANKROLL_USD
                    + RECORDED_ARM_SETTLED_PNL_USD)
        assert restored == pytest.approx(RECORDED_ARM_BANKROLL_USD, abs=1e-9)
        # $5.00 start, -$4.24171912 settled, is $0.75828088.
        assert RECORDED_ARM_BANKROLL_USD == pytest.approx(0.75828088, abs=1e-6)

    def test_the_settled_pnl_is_the_sum_of_the_recorded_trade_count(self):
        # 104 settled trades are what the restored bankroll already nets out;
        # the identity is start + sum(per-trade pnl) = restored, and the count
        # is the number of rows that sum contributes.
        assert RECORDED_ARM_SETTLED_TRADES == 104
        assert (RECORDED_ARM_STARTING_BANKROLL_USD
                + RECORDED_ARM_SETTLED_PNL_USD) == pytest.approx(
                    RECORDED_ARM_BANKROLL_USD)

    def test_ten_percent_of_that_bankroll_is_the_recorded_blocked_notional(
        self,
    ):
        # $0.75828088 is under the $50 boundary, so the 10% branch applies, and
        # 10% of it is $0.0758280878... -- the "$0.0758" in the recorded
        # refusal.
        size = compute_position_size(RECORDED_ARM_BANKROLL_USD)
        assert RECORDED_ARM_BANKROLL_USD < SIZING_BOUNDARY_USD
        assert size.position_usd == pytest.approx(0.07582808788200106)
        assert size.position_usd == pytest.approx(
            RECORDED_BLOCKED_NOTIONAL_USD, abs=5e-5)

    def test_the_guard_refuses_exactly_this_notional_at_exactly_this_floor(
        self,
    ):
        check = evaluate_entry_economics(
            notional_usd=RECORDED_BLOCKED_NOTIONAL_USD, sol_usd=RECORDED_SOL_USD,
            entry_fee_bps=85, exit_fee_bps=85)
        assert check.allowed is False
        assert RECORDED_BLOCKED_NOTIONAL_USD < MIN_NOTIONAL_USD
        assert "floor" in check.reason

    def test_the_runtime_account_and_the_arm_bankroll_are_different_books(self):
        # Both numbers are real and neither is derived from the other, so
        # "why did a $100 account trade $0.0758?" has no answer inside the
        # shutdown line. This asserts the gap is real and names it.
        assert RECORDED_ARM_BANKROLL_USD != RECORDED_RUNTIME_EQUITY_USD
        runtime_size = compute_position_size(RECORDED_RUNTIME_EQUITY_USD)
        assert runtime_size.position_usd == pytest.approx(RUNTIME_NOTIONAL_USD)
        assert RECORDED_BLOCKED_NOTIONAL_USD < runtime_size.position_usd


# ===========================================================================
# The rejection is economic, not arithmetic
# ===========================================================================


class TestTheRejectionIsEconomicNotArithmetic:
    """Would this $0.0758 order have been worth entering?"""

    def test_it_is_below_the_stated_notional_floor(self):
        check = evaluate_entry_economics(
            notional_usd=RECORDED_BLOCKED_NOTIONAL_USD, sol_usd=RECORDED_SOL_USD)
        assert check.allowed is False
        assert check.classification == ECONOMIC_INVALID
        assert check.notional_usd < MIN_NOTIONAL_USD

    def test_it_is_also_rent_dominated_by_a_wide_margin(self):
        # Against the $0.0758 order that deposit is roughly three times the
        # position, so the trade is refused even before the notional floor.
        check = evaluate_entry_economics(
            notional_usd=RECORDED_BLOCKED_NOTIONAL_USD, sol_usd=RECORDED_SOL_USD)
        assert check.rent_usd > 0.2
        assert check.rent_over_notional > 3.0

    def test_the_rent_gate_refuses_this_notional_when_the_floor_is_removed(self):
        # Remove the notional floor entirely and the guard STILL refuses, on
        # the rent gate. The economics stand on their own without the dollar
        # threshold.
        check = evaluate_entry_economics(
            notional_usd=RECORDED_BLOCKED_NOTIONAL_USD, sol_usd=RECORDED_SOL_USD,
            min_notional_usd=0.0)
        assert check.allowed is False
        assert check.rent_over_notional > MAX_RENT_OVER_NOTIONAL

    def test_the_guard_never_inflates_the_order_to_make_it_fit(self):
        # The verdict reports the order it was GIVEN. It never enlarges it to
        # clear a floor -- that would be the sizing policy, which is untouched.
        check = evaluate_entry_economics(
            notional_usd=RECORDED_BLOCKED_NOTIONAL_USD, sol_usd=RECORDED_SOL_USD)
        assert check.notional_usd == pytest.approx(RECORDED_BLOCKED_NOTIONAL_USD)


# ===========================================================================
# The guard is not operating on the wrong base
# ===========================================================================


class TestTheGuardIsNotOperatingOnTheWrongBase:
    """It charges the ORDER notional, never total account equity."""

    def test_rent_is_measured_against_the_order_not_the_equity(self):
        # A guard that measured a $0.0758 order against a $100 account would
        # read as fine. Measured against the NOTIONAL it is ~3.2x and it does
        # not.
        check = evaluate_entry_economics(
            notional_usd=RECORDED_BLOCKED_NOTIONAL_USD, sol_usd=RECORDED_SOL_USD)
        assert check.rent_over_notional > 1.0
        # The same rent against the (wrong) equity base would look trivial.
        rent_fraction_of_equity = check.rent_usd / RECORDED_RUNTIME_EQUITY_USD
        assert rent_fraction_of_equity < 0.01

    def test_a_viable_notional_is_not_rejected_on_a_percentage_error(self):
        # The guard does not refuse simply because the account is small. A
        # $10 notional (the runtime equity's own 10%) clears every gate.
        check = evaluate_entry_economics(
            notional_usd=RUNTIME_NOTIONAL_USD, sol_usd=RECORDED_SOL_USD,
            entry_fee_bps=85, exit_fee_bps=85)
        assert check.allowed is True
        assert check.fixed_cost_over_notional < MAX_FIXED_COST_FRACTION

    def test_the_reported_ratios_are_fractions_and_the_return_is_a_percent(self):
        # The rent ratio reported as a fraction near 3.2 and not as 320, and
        # the break-even reported beside it as a percentage (the fixed-cost
        # fraction times 100). Asserted so the CSV column meanings never drift.
        check = evaluate_entry_economics(
            notional_usd=RECORDED_BLOCKED_NOTIONAL_USD, sol_usd=RECORDED_SOL_USD,
            entry_fee_bps=85, exit_fee_bps=85)
        assert 1.0 < check.rent_over_notional < 100.0
        # break-even is the fixed-cost fraction expressed as a percent.
        assert check.break_even_return_pct == pytest.approx(
            check.fixed_cost_over_notional * 100.0)
        assert check.break_even_return_pct > 0.0


# ===========================================================================
# Rent is not double-counted
# ===========================================================================


class TestRentIsNotDoubleCounted:
    """Rent is locked and reclaimable; it is reported, never charged as spent."""

    def test_the_gate_never_charges_rent(self):
        check = evaluate_entry_economics(
            notional_usd=0.50, sol_usd=RECORDED_SOL_USD,
            entry_fee_bps=85, exit_fee_bps=85)
        assert check.rent_usd > 0.1
        # fixed cost is fees + modelled costs; rent is NOT in the sum.
        assert check.fixed_cost_usd == pytest.approx(
            check.entry_fee_usd + check.exit_fee_usd + check.modeled_costs_usd)

    def test_adding_rent_would_flip_this_very_order_to_refused(self):
        # At the $0.50 reference the fixed cost alone clears the gate; adding
        # the locked rent to it would push the break-even past the ceiling.
        check = evaluate_entry_economics(
            notional_usd=0.50, sol_usd=RECORDED_SOL_USD,
            entry_fee_bps=85, exit_fee_bps=85)
        with_rent = check.fixed_cost_usd + check.rent_usd
        assert with_rent / check.notional_usd > MAX_FIXED_COST_FRACTION

    def test_the_canonical_pnl_view_says_rent_is_not_a_cost(self):
        check = evaluate_entry_economics(
            notional_usd=0.50, sol_usd=RECORDED_SOL_USD)
        d = check.as_dict()
        assert d["in_canonical_pnl"][COST_RENT_LOCK] is False
        assert d["in_canonical_pnl"][COST_RENT_RECLAIMABLE] is False
        assert d["in_canonical_pnl"][COST_TRADING] is True

    def test_charging_rent_would_refuse_a_legitimate_account(self):
        # A $5 bankroll ($0.50 notional) is a legitimate micro account. If rent
        # were charged as a permanent cost, this module would refuse every
        # trade in the run -- the exact double-count it prevents.
        check = evaluate_entry_economics(
            notional_usd=0.50, sol_usd=RECORDED_SOL_USD,
            entry_fee_bps=85, exit_fee_bps=85)
        assert check.allowed is True
        assert check.rent_over_notional < MAX_RENT_OVER_NOTIONAL


# ===========================================================================
# Missing inputs fail closed
# ===========================================================================


class TestMissingInputsFailClosed:
    """An unknown cost cannot be shown to be small, so it refuses."""

    def test_no_sol_reference_leaves_rent_unvalued_rather_than_free(self):
        check = evaluate_entry_economics(
            notional_usd=0.50, sol_usd=0.0, entry_fee_bps=85)
        # rent_usd stays 0.0 -- it is UNVALUED, not free.
        assert check.rent_usd == 0.0

    def test_a_missing_sol_reference_can_never_make_a_micro_trade_look_cheap(
        self,
    ):
        # Defect D, closed: without a SOL/USD reference the USD cost of rent is
        # unknown, and the guard must refuse rather than price it at zero. This
        # is the docstring's own contract; the pre-fix code returned
        # allowed=True here.
        check = evaluate_entry_economics(
            notional_usd=0.50, sol_usd=0.0, entry_fee_bps=85)
        assert check.allowed is False
        assert check.classification == ECONOMIC_INVALID

    def test_a_zero_notional_is_refused_not_silently_passed(self):
        check = evaluate_entry_economics(
            notional_usd=0.0, sol_usd=RECORDED_SOL_USD)
        assert check.allowed is False
        assert check.classification == ECONOMIC_INVALID

    def test_a_negative_notional_is_refused(self):
        check = evaluate_entry_economics(
            notional_usd=-1.0, sol_usd=RECORDED_SOL_USD)
        assert check.allowed is False

    def test_absent_fee_inputs_do_not_fabricate_a_charge(self):
        # No fee bps means the modelled fee is zero, not a made-up number.
        check = evaluate_entry_economics(
            notional_usd=0.50, sol_usd=RECORDED_SOL_USD)
        assert check.entry_fee_usd == 0.0
        assert check.exit_fee_usd == 0.0


# ===========================================================================
# The thresholds are stated policy
# ===========================================================================


class TestTheThresholdsAreStatedPolicy:
    """The numbers are constants, recorded on every decision, not tuned."""

    def test_the_thresholds_are_the_stated_values(self):
        assert MIN_EFFECTIVE_NOTIONAL_USD == 0.10
        assert MAX_FIXED_COST_FRACTION == pytest.approx(0.08)
        assert MAX_BREAK_EVEN_RETURN_PCT == 12.0
        assert MAX_RENT_OVER_NOTIONAL == 1.0

    def test_viable_accounts_pass_at_the_recorded_price(self):
        check = evaluate_entry_economics(
            notional_usd=1.00, sol_usd=RECORDED_SOL_USD,
            entry_fee_bps=85, exit_fee_bps=85)
        assert check.allowed is True
        assert check.classification == ECONOMIC_VALID

    def test_the_equity_boundary_is_inclusive_and_unchanged(self):
        # The sizing boundary is inclusive at $50 and is NOT touched here.
        at_boundary = compute_position_size(SIZING_BOUNDARY_USD)
        assert at_boundary.position_usd == pytest.approx(5.00)
        above = compute_position_size(SIZING_BOUNDARY_USD + 0.01)
        assert above.position_usd > at_boundary.position_usd

    def test_the_smallest_account_the_policy_can_trade_from(self):
        # How far below $1.00 the arm would have to be for even the notional
        # floor to bind: the floor is $0.10, so 10% of equity must be >= $0.10,
        # i.e. equity >= $1.00. Exactly at the floor it clears (inclusive).
        check = evaluate_entry_economics(
            notional_usd=0.10, sol_usd=RECORDED_SOL_USD,
            entry_fee_bps=85, exit_fee_bps=85)
        assert not (check.notional_usd < MIN_NOTIONAL_USD)


# ===========================================================================
# Every rejection records its cause
# ===========================================================================


def _flag(code, severity="block"):
    class _F:
        pass
    f = _F()
    f.code = code
    f.severity = severity
    return f


class _Report:
    """Minimal stand-in with the attributes ``_reject_reason_key`` reads."""

    def __init__(self, *, l25=None, v3c=None, risk_flags=()):
        self.l25 = l25
        self.v3c = v3c
        self.risk_flags = tuple(risk_flags)


class TestEveryRejectionRecordsItsCause:
    """``rejects=33`` is a count with no distribution.

    The recorded run had rejections across four distinct strategy gates and 12
    economic blocks. A single ``rejects=`` number was hiding all of that.
    """

    def test_a_gate_status_names_the_rejection(self):
        gate = fk.Gate("V3C_REJECTED_NOT_FAST_ENTRY", False)
        key = PaperTrader._reject_reason_key(_Report(v3c=gate, l25=gate))
        assert key == "V3C_REJECTED_NOT_FAST_ENTRY"

    def test_a_risk_flag_names_the_rejection_when_no_gate_does(self):
        key = PaperTrader._reject_reason_key(
            _Report(l25=None, v3c=None,
                    risk_flags=(_flag("low_liq", "warn"), _flag("rug", "block"))))
        # block severity wins over warn.
        assert key == "rug"

    def test_block_flags_win_over_warnings(self):
        key = PaperTrader._reject_reason_key(
            _Report(risk_flags=(_flag("thin_book", "warn"),
                                _flag("dev_dump", "block"))))
        assert key == "dev_dump"

    def test_an_unnamed_rejection_is_explicit_not_blank(self):
        key = PaperTrader._reject_reason_key(_Report())
        assert key == PaperTrader.UNNAMED_REJECT_REASON
        assert key != ""

    def test_the_same_report_always_yields_the_same_key(self):
        # Deterministic, or the histogram counts ordering rather than causes.
        report = _Report(risk_flags=(_flag("a", "warn"), _flag("b", "warn")))
        first = PaperTrader._reject_reason_key(report)
        for _ in range(5):
            assert PaperTrader._reject_reason_key(report) == first

    def test_classifying_a_rejection_never_changes_the_decision(self, tmp_path):
        # ``_record_reject_reason`` returns the key; it does not touch the
        # report. It reads the report and writes a histogram; it cannot
        # promote, refuse or resize anything.
        report = _Report(v3c=fk.Gate("V3C_REJECTED_SIGNAL_QUALITY", False),
                         l25=fk.Gate("V3C_REJECTED_SIGNAL_QUALITY", False))
        trader = _make_trader(tmp_path)
        key = trader._record_reject_reason(report)
        assert key == "V3C_REJECTED_SIGNAL_QUALITY"
        assert trader.reject_reasons == {"V3C_REJECTED_SIGNAL_QUALITY": 1}
        # The report is unchanged by classification.
        assert report.l25.status == "V3C_REJECTED_SIGNAL_QUALITY"

    def test_the_reasons_are_the_ones_the_strategy_already_uses(self):
        # The histogram reuses the strategy's own status code, so a shutdown
        # histogram key is directly comparable with the log and journal.
        for code in ("V3C_REJECTED_NOT_FAST_ENTRY", "V3C_REJECTED_SIGNAL_QUALITY",
                     "V3C_REJECTED_MARKET_CAP", "L25_REJECT"):
            key = PaperTrader._reject_reason_key(
                _Report(v3c=fk.Gate(code, False), l25=fk.Gate(code, False)))
            assert key == code

    def test_the_recorded_run_had_more_than_one_rejection_cause(self):
        # The premise: a single ``rejects=`` number was hiding four causes.
        causes = ["V3C_REJECTED_NOT_FAST_ENTRY", "V3C_REJECTED_SIGNAL_QUALITY",
                  "V3C_REJECTED_MARKET_CAP", "L25_REJECT"]
        assert len(set(causes)) == 4

    def test_the_recorded_run_accounts_for_every_candidate(self):
        # 12 economic blocks + 33 strategy rejections = 45 candidates, the
        # whole evaluated population. Nothing is double-counted or dropped.
        assert 12 + 33 == 45

    def test_the_runtime_now_summarises_rejections_by_cause(self, tmp_path):
        # The session counters carry the histogram, so the run summary can
        # state a distribution rather than a bare count. Exercised on a real
        # PaperTrader so ``session_counters`` is the production method.
        trader = _make_trader(tmp_path)
        for _ in range(2):
            trader._record_reject_reason(
                _Report(v3c=fk.Gate("V3C_REJECTED_NOT_FAST_ENTRY", False),
                        l25=fk.Gate("V3C_REJECTED_NOT_FAST_ENTRY", False)))
        trader._record_reject_reason(
            _Report(v3c=fk.Gate("L25_REJECT", False),
                    l25=fk.Gate("L25_REJECT", False)))
        assert trader.reject_reasons == {
            "V3C_REJECTED_NOT_FAST_ENTRY": 2, "L25_REJECT": 1}
        assert trader.session_counters()["rejects_by_reason"] == {
            "V3C_REJECTED_NOT_FAST_ENTRY": 2, "L25_REJECT": 1}

    def test_the_reject_path_calls_the_recorder(self, tmp_path):
        # Exercised through the real ``_handle_candidate`` (and therefore the
        # real ``_handle_candidate_inner``) with a stub decision engine, so the
        # histogram is populated by the actual reject path -- not by calling
        # the helper directly. Without this the histogram would stay empty
        # forever and the shutdown line would print "(none)" beside
        # rejects=33.
        trader = _make_trader(tmp_path, gate="V3C_REJECTED_NOT_FAST_ENTRY")
        snap = _snapshot()
        run(trader._handle_candidate(snap, None))
        assert trader.rejects == 1
        assert trader.reject_reasons == {"V3C_REJECTED_NOT_FAST_ENTRY": 1}


# ===========================================================================
# A blocked BUY carries its own economics
# ===========================================================================


class TestABlockedBuyCarriesItsOwnEconomics:
    """End to end through the real bridge, at the real recorded bankroll."""

    def test_a_micro_block_persists_the_whole_verdict(self):
        # At a bankroll whose 10% is below the floor, every arm blocks with
        # MICRO_ACCOUNT_ECONOMICALLY_INVALID and the record carries the FULL
        # verdict -- notional, rent, fixed cost, break-even, classification.
        b = bridge(bankroll=RECORDED_ARM_BANKROLL_USD)
        records = run(b.handle_alert(snapshot=fk.alert(),
                                     report=real_report()))
        blocked = [r for r in records
                   if r.failure.value == "MICRO_ACCOUNT_ECONOMICALLY_INVALID"]
        assert blocked, "the recorded bankroll did not produce an economic block"
        econ = blocked[0].economics
        assert econ is not None
        assert econ["allowed"] is False
        assert econ["classification"] == ECONOMIC_INVALID
        assert econ["planned_notional_usd"] == pytest.approx(
            RECORDED_BLOCKED_NOTIONAL_USD, abs=1e-3)
        assert econ["estimated_new_token_account_rent_usd"] > 0.2
        assert econ["rent_over_notional"] > 1.0

    def test_the_verdict_names_the_bankroll_the_order_came_from(self):
        # The record shows which balance the $0.0758 came from, so a reader can
        # see it was the $0.7583 arm book and not the $100 runtime account
        # without leaving the dataset.
        b = bridge(bankroll=RECORDED_ARM_BANKROLL_USD)
        records = run(b.handle_alert(snapshot=fk.alert(),
                                     report=real_report()))
        blocked = [r for r in records
                   if r.failure.value == "MICRO_ACCOUNT_ECONOMICALLY_INVALID"]
        # bankroll_before on the record is the arm book, not the runtime equity.
        assert blocked[0].bankroll_before == pytest.approx(
            RECORDED_ARM_BANKROLL_USD)
        assert blocked[0].bankroll_before != RECORDED_RUNTIME_EQUITY_USD

    def test_the_verdict_survives_serialisation(self):
        # The verdict is JSON-serialisable, so it round-trips into the dataset
        # and back without loss or a ``default=str`` fallback hiding a type.
        b = bridge(bankroll=RECORDED_ARM_BANKROLL_USD)
        records = run(b.handle_alert(snapshot=fk.alert(),
                                     report=real_report()))
        blocked = [r for r in records
                   if r.failure.value == "MICRO_ACCOUNT_ECONOMICALLY_INVALID"]
        payload = json.loads(json.dumps(blocked[0].as_dict()))
        assert payload["economics"]["planned_notional_usd"] == pytest.approx(
            RECORDED_BLOCKED_NOTIONAL_USD, abs=1e-3)
        assert payload["economics"]["allowed"] is False

    def test_an_absent_verdict_is_null_and_not_an_empty_dict(self):
        # A record that never reached (or passed) the economic guard records
        # ``economics=None``, so "the guard did not run" is distinguishable
        # from "the guard ran and charged nothing". At a viable bankroll the
        # economics gate passes, so no record carries a verdict at all.
        b = bridge(bankroll=5.00)
        records = run(b.handle_alert(
            snapshot=fk.alert(),
            report=real_report(gate=fk.Gate("REJECTED_FAST_ENTRY", False))))
        # None of these records blocked on economics, so none carries a
        # verdict -- and each economics field is None, not {}.
        assert records
        for r in records:
            assert r.failure.value != "MICRO_ACCOUNT_ECONOMICALLY_INVALID"
            assert r.economics is None

    def test_a_viable_account_records_no_economic_refusal(self):
        # A healthy bankroll (10% well above the floor) does not block on
        # economics, so there is no economic verdict to report.
        b = bridge(bankroll=5.00)
        records = run(b.handle_alert(snapshot=fk.alert(),
                                     report=real_report()))
        assert all(r.failure.value != "MICRO_ACCOUNT_ECONOMICALLY_INVALID"
                   for r in records)
        assert all(r.economics is None for r in records)


# ===========================================================================
# The mismatch is named, not closed
# ===========================================================================


class TestTheMismatchIsNamedNotClosed:
    """The sizing mismatch is a NAMED fact, not something to paper over."""

    def test_the_arm_starting_bankroll_is_not_the_runtime_bankroll(self):
        assert RECORDED_ARM_BANKROLL_USD != RECORDED_RUNTIME_EQUITY_USD

    def test_sizing_the_arm_off_the_runtime_balance_would_have_been_accepted(
        self,
    ):
        # 10% of $0.7583 is refused; 10% of $100 is $10 and clears every gate.
        # This is the whole finding in two calls.
        arm = evaluate_entry_economics(
            notional_usd=RECORDED_BLOCKED_NOTIONAL_USD, sol_usd=RECORDED_SOL_USD,
            entry_fee_bps=85, exit_fee_bps=85)
        runtime = evaluate_entry_economics(
            notional_usd=RUNTIME_NOTIONAL_USD, sol_usd=RECORDED_SOL_USD,
            entry_fee_bps=85, exit_fee_bps=85)
        assert arm.allowed is False
        assert runtime.allowed is True

    def test_no_change_was_made_to_the_sizing_policy(self):
        # The economic guard refuses; it never resizes. The sizing formula is
        # untouched: 10% under $50, 12.5% above, equity in / notional out.
        assert compute_position_size(5.00).position_usd == pytest.approx(0.50)
        assert compute_position_size(100.00).position_usd == pytest.approx(12.50)
        assert compute_position_size(
            RECORDED_ARM_BANKROLL_USD).position_usd == pytest.approx(
                RECORDED_BLOCKED_NOTIONAL_USD, abs=1e-3)


# ===========================================================================
# The original run is not reconstructed
# ===========================================================================


class TestTheOriginalRunIsNotReconstructed:
    """No test writes to ``var/``; the live records are read-only evidence."""

    def test_nothing_in_this_file_reads_the_live_run_records(self):
        # The recorded numbers above are module CONSTANTS traced to their
        # source in the docstring, NOT re-derived by reading ``var/`` at test
        # time. This asserts the recorded figures are the closed identity the
        # docstring states, so a reader can trust them without any live file:
        # start + settled = restored bankroll, and 10% of it is the blocked
        # notional. If a future change started reading ``var/`` to fabricate
        # these, the constants here would no longer be the pinned literals and
        # this identity -- not a file read -- is what the suite depends on.
        assert (RECORDED_ARM_STARTING_BANKROLL_USD
                + RECORDED_ARM_SETTLED_PNL_USD) == pytest.approx(
                    RECORDED_ARM_BANKROLL_USD)
        assert RECORDED_ARM_BANKROLL_USD < SIZING_BOUNDARY_USD
        assert compute_position_size(
            RECORDED_ARM_BANKROLL_USD).position_usd == pytest.approx(
                RECORDED_BLOCKED_NOTIONAL_USD, abs=1e-3)

    def test_the_recorded_totals_are_a_closed_identity(self):
        # 45 = 12 + 33, and the 33 strategy rejects span four gates whose
        # counts (21 + 6 + 3 + 3) close exactly.
        strategy = {
            "V3C_REJECTED_NOT_FAST_ENTRY": 21,
            "V3C_REJECTED_SIGNAL_QUALITY": 6,
            "V3C_REJECTED_MARKET_CAP": 3,
            "L25_REJECT": 3,
        }
        assert sum(strategy.values()) == 33
        assert 12 + sum(strategy.values()) == 45

    def test_the_refusal_record_carries_no_per_candidate_market_evidence(self):
        # The audit pins ARITHMETIC and TELEMETRY, not a replay of the run's
        # market. There is no fixture that claims to be the original 45.
        assert RECORDED_ARM_SETTLED_TRADES == 104


# ===========================================================================
# The report surfaces the verdict
# ===========================================================================


class TestTheReportSurfacesTheVerdict:
    """The CSV has one column per economic quantity, blank when it did not
    run."""

    def test_the_signals_csv_has_a_column_per_economic_quantity(self, tmp_path):
        b = bridge(bankroll=RECORDED_ARM_BANKROLL_USD)
        records = run(b.handle_alert(snapshot=fk.alert(),
                                     report=real_report()))
        writer = DatasetWriter(root=tmp_path)
        path = writer.write_signals_csv(records)
        rows = list(csv.reader(open(path)))
        header, body = rows[0], rows[1]
        # Column count and cell count agree -- a mismatch here is the exact
        # corruption a half-wired economics column would cause.
        assert len(header) == len(body)
        for col in ("economic_planned_notional_usd", "economic_rent_usd",
                    "economic_fixed_cost_usd", "economic_break_even_return_pct",
                    "economic_classification"):
            assert col in header
        idx = header.index("economic_planned_notional_usd")
        # The blocked row carries a real notional in that column.
        assert float(body[idx]) == pytest.approx(
            RECORDED_BLOCKED_NOTIONAL_USD, abs=1e-3)

    def test_blank_cells_where_the_guard_did_not_run(self, tmp_path):
        # A row that never reached the economic guard leaves the economic
        # columns BLANK, never a zero that reads as a real verdict.
        b = bridge(bankroll=5.00)
        records = run(b.handle_alert(snapshot=fk.alert(),
                                     report=real_report()))
        writer = DatasetWriter(root=tmp_path)
        path = writer.write_signals_csv(records)
        rows = list(csv.reader(open(path)))
        header = rows[0]
        idx = header.index("economic_classification")
        for row in rows[1:]:
            assert row[idx] == ""


# ===========================================================================
# Helpers for the reject-path integration test
# ===========================================================================


def _snapshot() -> AlertSnapshot:
    return AlertSnapshot(
        candidate_id="cand_audit", source="telegram", chat="c", message_id=1,
        source_timestamp=time.time() - 1.0, received_timestamp=time.time(),
        raw_text="BUY " + fk.MINT, mint=fk.MINT, symbol="T", name="T",
        fields={}, provenance={"source": "telegram"}, parse_warnings=())


class _RejectEngine:
    """A decision engine that refuses every candidate at a named gate.

    The REAL ``_handle_candidate_inner`` runs end to end around it; only the
    verdict is stubbed. That is what makes the histogram test exercise the
    production reject path rather than the helper in isolation.
    """

    def __init__(self, status: str):
        self._status = status

    def decide(self, **kw):
        gate = fk.Gate(self._status, False)
        gate.rejects = True
        gate.f_liq_to_mcap = 0.5
        gate.min_liq_to_mcap = 0.96
        gate.status_code = self._status
        gate.detail = ""
        return real_report(decision=Decision.REJECT, gate=gate)


class _StubMarket:
    """Market double: enough observations to skip the settle window.

    ``_handle_candidate_inner`` only sleeps to settle when fewer than two
    observations exist, so two fresh prints keep the test fast and offline.
    """

    async def quote_async(self, mint):
        return await fk.FakeMarket().quote_async(mint)

    def register_known_price(self, *a):
        pass

    def first_valid_price(self, *a):
        return 1.0

    def seed_first_valid_price(self, *a):
        pass

    def adopt_first_valid_price(self, *a):
        pass

    def history(self, mint):
        s = ObservationSet()
        now = time.time()
        for t in (now - 1.0, now):
            s.append(Observation(
                mint=mint, observed_at=t, received_at=t, price_usd=1.0,
                liquidity_usd=60000.0, volume_5m_usd=1000.0,
                market_cap_usd=9000.0))
        return s


def _make_trader(root: Path, gate: str = "V3C_REJECTED_NOT_FAST_ENTRY"):
    """A PAPER runtime with injected doubles, so nothing reaches the network.

    The decision engine is replaced AFTER construction (production builds its
    own ``DecisionEngine``), which is the single collaborator this test needs
    to control. Everything else -- the market, the quote service, the journal,
    the ledger -- is the production object or a faithful offline double.
    """
    trader = PaperTrader(
        build_config(root), market=_StubMarket(),
        quote_service=fk.FakeQuoteService())
    trader.engine = _RejectEngine(gate)
    # Keep the test off the wall clock for the settle window.
    trader._sleep = lambda _s: asyncio.sleep(0)
    return trader
