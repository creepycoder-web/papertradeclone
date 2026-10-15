"""
Offline tests for micro-account economics and rent-aware sizing.

Covers required tests 9-13: existing ATA, missing ATA, unknown account size,
rent fraction, and economic status.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

AI = Path(__file__).resolve().parents[1]
for _p in (AI, AI / "research"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from l25x.economics import (  # noqa: E402
    EconomicPolicy,
    EconomicStatus,
    NOTIONAL_GRID,
    assess_economics,
    notional_curve,
)
from l25x.rent import (  # noqa: E402
    RentBasis,
    RentSchedule,
    TokenAccountState,
    incremental_rent_lamports,
)

TOKEN = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022 = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
SOL = 121.604733

#: The $GOAT trade's real token-account rent, from the chain settlement.
GOAT_RENT_LAMPORTS = 1_513_840
GOAT_NOTIONAL_USD = 0.23455257


# -- 9. existing ATA -> zero incremental rent -------------------------------


def test_existing_ata_costs_zero_incremental_rent():
    """9. The account already exists, so this trade locks nothing."""
    acct = TokenAccountState(address="ATA1", exists=True, program_id=TOKEN,
                             lamports=2_039_280)
    lamports, basis, _ = incremental_rent_lamports(acct, None)
    assert lamports == 0
    assert basis is RentBasis.MEASURED_PREEEXISTING


def test_existing_ata_zero_rent_in_the_economic_assessment():
    a = assess_economics(planned_notional_usd=0.25, sol_price_usd=SOL,
                         token_account=TokenAccountState(address="ATA1",
                                                        exists=True,
                                                        program_id=TOKEN))
    assert a.estimated_new_token_account_rent_usd == 0.0
    assert a.rent_basis is RentBasis.MEASURED_PREEEXISTING
    # only fees remain as fixed cost
    assert a.total_fixed_cost_usd == pytest.approx(0.25 * 170 / 10_000.0)


# -- 10. missing ATA -> estimated rent -------------------------------------


def test_missing_ata_with_a_configured_schedule_is_estimated():
    """10. Verified absent + a schedule for its program -> an ESTIMATED figure."""
    sched = RentSchedule(by_program={TOKEN: 2_039_280},
                         provenance="test schedule")
    acct = TokenAccountState(address=None, exists=False, program_id=TOKEN)
    lamports, basis, why = incremental_rent_lamports(acct, sched)
    assert lamports == 2_039_280
    # labelled ESTIMATED, never passed off as measured
    assert basis is RentBasis.ESTIMATED_PROGRAM_SCHEDULE
    assert "not measured" in why


def test_rent_is_not_hard_coded_to_one_universal_number():
    """
    Different programs carry different rent. Token-2022 accounts with extensions
    are LARGER, which is why the repository reads rent per account.
    """
    sched = RentSchedule(by_program={TOKEN: 2_039_280,
                                     TOKEN_2022: GOAT_RENT_LAMPORTS})
    classic = incremental_rent_lamports(
        TokenAccountState(address=None, exists=False, program_id=TOKEN), sched)
    t22 = incremental_rent_lamports(
        TokenAccountState(address=None, exists=False, program_id=TOKEN_2022), sched)
    assert classic[0] != t22[0]
    assert t22[0] == GOAT_RENT_LAMPORTS


# -- 11. unknown account size -> fail closed -------------------------------


def test_unverified_account_existence_is_unknown_not_zero():
    """
    11. exists=None must not collapse to "free". It yields UNKNOWN so the
    economic layer can refuse rather than assume.
    """
    acct = TokenAccountState(address=None, exists=None, program_id=TOKEN)
    lamports, basis, why = incremental_rent_lamports(acct, None)
    assert lamports is None
    assert basis is RentBasis.UNKNOWN
    assert "never verified" in why


def test_absent_account_with_no_schedule_is_unknown():
    acct = TokenAccountState(address=None, exists=False, program_id="UnknownProg")
    lamports, basis, _ = incremental_rent_lamports(acct, RentSchedule())
    assert lamports is None
    assert basis is RentBasis.UNKNOWN


def test_unknown_rent_propagates_to_an_unknown_status_not_a_cheap_one():
    a = assess_economics(planned_notional_usd=10.0, sol_price_usd=SOL,
                         token_account=TokenAccountState(address=None,
                                                        exists=None,
                                                        program_id=TOKEN))
    assert a.status is EconomicStatus.ECONOMICALLY_UNKNOWN
    assert a.rent_fraction_of_notional is None
    assert a.break_even_return_pct is None


def test_unknown_rent_can_be_configured_to_fail_closed():
    a = assess_economics(
        planned_notional_usd=10.0, sol_price_usd=SOL,
        token_account=TokenAccountState(address=None, exists=None,
                                        program_id=TOKEN),
        policy=EconomicPolicy(fail_closed_on_unknown_rent=True))
    assert a.status is EconomicStatus.ECONOMICALLY_INVALID


# -- 12. rent fraction calculation -----------------------------------------


def test_rent_fraction_matches_the_goat_trade():
    """
    12. Reproduces the audited GOAT economics: $0.184090 of rent on a $0.234553
    position is 78.5% of it.
    """
    a = assess_economics(
        planned_notional_usd=GOAT_NOTIONAL_USD, sol_price_usd=SOL,
        token_account=TokenAccountState(address=None, exists=False,
                                        program_id=TOKEN_2022),
        rent_lamports=GOAT_RENT_LAMPORTS)
    assert a.estimated_new_token_account_rent_usd == pytest.approx(
        GOAT_RENT_LAMPORTS / 1e9 * SOL, rel=1e-9)
    assert a.rent_fraction_of_notional == pytest.approx(0.785, abs=0.005)
    assert any("78" in n or "fixed account-creation cost" in n
               for n in a.notes)


def test_break_even_is_the_return_needed_to_cover_fixed_costs():
    """
    Break-even EXCLUDES principal, because principal comes back on exit. The
    audited report framed the same number two ways: 81.3% using the all-in debit
    against the swap input, and -42.91% using the full debit as the denominator.
    Both are right under their own definition; this module's is the round-trip
    one, and it cross-checks to within ~1pp of the audit.
    """
    a = assess_economics(
        planned_notional_usd=GOAT_NOTIONAL_USD, sol_price_usd=SOL,
        token_account=TokenAccountState(address=None, exists=False,
                                        program_id=TOKEN_2022),
        rent_lamports=GOAT_RENT_LAMPORTS)
    # rent 0.184090 + model fees 0.003987
    assert a.total_fixed_cost_usd == pytest.approx(0.188078, abs=1e-5)
    assert a.fixed_cost_fraction_of_notional == pytest.approx(0.8019, abs=1e-3)
    # a round trip must gain this much merely to return principal plus costs
    assert a.break_even_return_pct == pytest.approx(80.19, abs=0.1)
    # and it agrees with the audited 81.3% within the fee-model difference
    assert abs(a.break_even_return_pct - 81.3) < 2.0


def test_rent_fraction_falls_as_notional_rises():
    """Rent is a function of the ACCOUNT, not the position."""
    curve = {r["planned_notional_usd"]: r for r in notional_curve(
        rent_lamports=GOAT_RENT_LAMPORTS, sol_price_usd=SOL)}
    assert curve[0.25]["rent_fraction_of_notional"] > 0.7
    assert curve[10.0]["rent_fraction_of_notional"] < 0.03
    assert curve[100.0]["rent_fraction_of_notional"] < 0.005
    # monotonically decreasing
    fr = [curve[n]["rent_fraction_of_notional"] for n in NOTIONAL_GRID]
    assert all(fr[i] > fr[i + 1] for i in range(len(fr) - 1))


# -- 13. micro-account economic status -------------------------------------


def test_no_threshold_is_hard_coded_by_default():
    """
    13. The default policy sets nothing, so a 78.5%-rent position is OBSERVED,
    not silently rejected. Choosing a threshold is a separate, future decision.
    """
    a = assess_economics(
        planned_notional_usd=GOAT_NOTIONAL_USD, sol_price_usd=SOL,
        token_account=TokenAccountState(address=None, exists=False,
                                        program_id=TOKEN_2022),
        rent_lamports=GOAT_RENT_LAMPORTS)
    assert a.status is EconomicStatus.ECONOMICALLY_VALID


def test_configured_rent_threshold_rejects_and_explains():
    a = assess_economics(
        planned_notional_usd=GOAT_NOTIONAL_USD, sol_price_usd=SOL,
        token_account=TokenAccountState(address=None, exists=False,
                                        program_id=TOKEN_2022),
        rent_lamports=GOAT_RENT_LAMPORTS,
        policy=EconomicPolicy(max_rent_fraction_of_notional=0.10))
    assert a.status is EconomicStatus.ECONOMICALLY_INVALID
    assert any("rent is" in r for r in a.reasons)


def test_marginal_band_distinguishes_from_rejection():
    a = assess_economics(
        planned_notional_usd=5.0, sol_price_usd=SOL,
        token_account=TokenAccountState(address=None, exists=False,
                                        program_id=TOKEN_2022),
        rent_lamports=GOAT_RENT_LAMPORTS,
        # at $5 the rent fraction is 3.68%, so a 3% marginal band bites
        policy=EconomicPolicy(marginal_rent_fraction=0.03))
    assert a.rent_fraction_of_notional == pytest.approx(0.0368, abs=1e-3)
    assert a.status is EconomicStatus.ECONOMICALLY_MARGINAL


def test_break_even_threshold_can_reject_independently():
    a = assess_economics(
        planned_notional_usd=GOAT_NOTIONAL_USD, sol_price_usd=SOL,
        token_account=TokenAccountState(address=None, exists=False,
                                        program_id=TOKEN_2022),
        rent_lamports=GOAT_RENT_LAMPORTS,
        policy=EconomicPolicy(max_break_even_return_pct=25.0))
    assert a.status is EconomicStatus.ECONOMICALLY_INVALID
    assert any("break-even" in r for r in a.reasons)


def test_all_required_quantities_are_reported():
    a = assess_economics(
        planned_notional_usd=10.0, sol_price_usd=SOL,
        token_account=TokenAccountState(address=None, exists=False,
                                        program_id=TOKEN),
        rent_schedule=RentSchedule(by_program={TOKEN: 2_039_280}))
    d = a.as_dict()
    for k in ("planned_notional_usd", "planned_notional_sol",
              "estimated_entry_fees_usd", "estimated_exit_fees_usd",
              "estimated_new_token_account_rent_usd", "rent_basis",
              "rent_fraction_of_notional", "total_fixed_cost_usd",
              "fixed_cost_fraction_of_notional", "break_even_return_pct",
              "status"):
        assert k in d, k
    assert d["planned_notional_sol"] == pytest.approx(10.0 / SOL, rel=1e-9)