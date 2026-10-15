"""
Offline tests for the rent lifecycle, reclaimability, and no-double-counting.

Covers required tests 14-17.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

AI = Path(__file__).resolve().parents[1]
for _p in (AI, AI / "research"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from l25x.rent import (  # noqa: E402
    CostBucket,
    RentBasis,
    RentLedger,
    RentLifecycleState,
    ReclaimStatus,
    TokenAccountState,
    close_eligibility,
    economic_adjusted_view,
)

TOKEN_2022 = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
GOAT_RENT = 1_513_840
WALLET = "wUEUbVmz2EfcLfchHmucTKhRpZgDpsBvo3FttZ4Wbt6"
SOL = 121.604733


def acct(*, exists=True, amount=0, lamports=GOAT_RENT, owner=WALLET,
         delegate=False, frozen=False, authority=None,
         program=TOKEN_2022) -> TokenAccountState:
    return TokenAccountState(
        address="ATA1", exists=exists, program_id=program,
        account_len=170 if program == TOKEN_2022 else 165,
        lamports=lamports, token_amount_raw=amount, owner=owner,
        close_authority=authority, delegate_present=delegate, frozen=frozen,
    )


def fresh_ledger(rent=GOAT_RENT) -> RentLedger:
    led = RentLedger(trade_id="t1", mint="Mint1", token_account="ATA1")
    led.on_entry(acct(), rent, RentBasis.MEASURED_ON_CHAIN)
    return led


# -- 14. no double counting ------------------------------------------------


def test_rent_buckets_partition_the_locked_lamports():
    """14. Locked + reclaimed + irrecoverable == entry rent, always."""
    led = fresh_ledger()
    b = led.buckets()
    assert (b[CostBucket.TEMPORARY_RENT_LOCK.value]
            + b[CostBucket.RECLAIMED_RENT.value]
            + b[CostBucket.IRRECOVERABLE_COST.value]) == led.entry_rent_lamports
    led.assert_consistent()


def test_reclaimed_rent_leaves_the_lock_bucket():
    led = fresh_ledger()
    led.on_reclaimed(GOAT_RENT)
    b = led.buckets()
    assert b[CostBucket.TEMPORARY_RENT_LOCK.value] == 0
    assert b[CostBucket.RECLAIMED_RENT.value] == GOAT_RENT
    led.assert_consistent()


def test_partial_reclaim_splits_cleanly():
    led = fresh_ledger()
    led.on_reclaimed(500_000)
    b = led.buckets()
    assert b[CostBucket.RECLAIMED_RENT.value] == 500_000
    assert b[CostBucket.TEMPORARY_RENT_LOCK.value] == GOAT_RENT - 500_000
    led.assert_consistent()


def test_cannot_reclaim_more_than_was_locked():
    led = fresh_ledger()
    with pytest.raises(ValueError, match="cannot reclaim"):
        led.on_reclaimed(GOAT_RENT + 1)


def test_trading_pnl_bucket_is_never_written_by_this_module():
    """Canonical settlement PnL is carried, never authored here."""
    led = fresh_ledger()
    led.trading_pnl_usd = 0.0015961594
    led.on_reclaimed(GOAT_RENT)
    d = led.as_dict()
    assert d["buckets_lamports"][CostBucket.TRADING_PNL.value] == 0
    assert d["canonical_pnl_contains_rent"] is False
    # the economic view is SEPARATE and does not overwrite it
    view = economic_adjusted_view(led, sol_price_usd=SOL)
    assert view["canonical_trading_pnl_usd"] == 0.0015961594
    assert view["rent_temporarily_locked_usd"] == 0.0
    assert view["rent_reclaimed_usd"] == pytest.approx(
        GOAT_RENT / 1e9 * SOL, rel=1e-9)


def test_economic_view_subtracts_an_unreclaimed_lock():
    led = fresh_ledger()
    led.trading_pnl_usd = 0.0015961594
    view = economic_adjusted_view(led, sol_price_usd=SOL)
    assert view["rent_temporarily_locked_usd"] == pytest.approx(
        GOAT_RENT / 1e9 * SOL, rel=1e-9)
    # canonical untouched, economic view shows the lock's cost
    assert view["canonical_trading_pnl_usd"] == 0.0015961594
    assert view["economic_adjusted_pnl_usd"] < view["canonical_trading_pnl_usd"]


def test_inconsistent_ledger_is_detected():
    """
    The invariant is enforced, not merely documented. ``irrecoverable`` is part
    of the partition so setting it alone stays consistent; the double-count this
    guards against is a lamport appearing as BOTH reclaimed and locked.
    """
    led = fresh_ledger()
    # claiming MORE irrecoverable than was ever locked is a genuine
    # inconsistency: the derived locked bucket clamps at 0, so the buckets can
    # no longer sum to the entry figure.
    led.irrecoverable_lamports = GOAT_RENT + 5
    with pytest.raises(ValueError, match="counted"):
        led.assert_consistent()


def test_partition_is_derived_so_double_counting_is_structurally_impossible():
    """
    Stronger than detecting a tamper: the locked bucket is DERIVED from
    entry/reclaimed/irrecoverable, so no assignment of public fields can put one
    lamport in two buckets.
    """
    led = fresh_ledger()
    for reclaimed in (0, 1, 500_000, GOAT_RENT):
        led2 = fresh_ledger()
        led2.reclaimed_lamports = reclaimed
        b = led2.buckets()
        assert (b[CostBucket.TEMPORARY_RENT_LOCK.value]
                + b[CostBucket.RECLAIMED_RENT.value]
                + b[CostBucket.IRRECOVERABLE_COST.value]) == GOAT_RENT
        led2.assert_consistent()
    # TRADING_PNL is hard-zeroed and never authored by this module
    assert led.buckets()[CostBucket.TRADING_PNL.value] == 0


# -- 15. zero-token-balance close eligibility ------------------------------


def test_zero_balance_empty_account_is_close_eligible():
    ok, why = close_eligibility(acct(amount=0), expected_owner=WALLET)
    assert ok is True
    assert "empty" in why


def test_close_eligibility_refuses_without_an_expected_owner():
    """Ownership that cannot be CHECKED is a refusal, never a pass."""
    ok, why = close_eligibility(acct(amount=0))
    assert ok is False
    assert "expected owner" in why


def test_close_eligibility_requires_verified_existence():
    ok, why = close_eligibility(acct(exists=None), expected_owner=WALLET)
    assert ok is False
    assert "existence" in why


def test_close_eligibility_refuses_foreign_owner_and_delegate():
    # a token account owned by a DIFFERENT wallet is never close-eligible
    assert close_eligibility(acct(owner="SomeoneElse"),
                             expected_owner=WALLET)[0] is False
    assert close_eligibility(acct(delegate=True), expected_owner=WALLET)[0] is False
    assert close_eligibility(acct(frozen=True), expected_owner=WALLET)[0] is False
    assert close_eligibility(acct(authority="AnotherAuthority"),
                             expected_owner=WALLET)[0] is False


# -- 16. dust prevents reclaimability --------------------------------------


def test_dust_prevents_close_eligibility():
    """
    16. A non-zero raw balance is NOT close-eligible. There is no proven safe
    dust policy in this repository, so the module refuses rather than inventing
    one.
    """
    ok, why = close_eligibility(acct(amount=1), expected_owner=WALLET)
    assert ok is False
    assert "raw units" in why and "dust" in why.lower()


def test_dust_strands_the_account_in_the_ledger():
    led = RentLedger(trade_id="t2", mint="M", token_account="ATA1")
    led.on_entry(acct(amount=0), GOAT_RENT, RentBasis.MEASURED_ON_CHAIN)
    led.on_exit(acct(amount=7), expected_owner=WALLET)          # dust left behind
    assert led.lifecycle is RentLifecycleState.BLOCKED
    assert led.reclaim_status is ReclaimStatus.BLOCKED
    assert led.reclaim_blocker is not None
    d = led.as_dict()
    assert d["reclaimable_rent_lamports"] == 0
    # the rent is still locked, and still accounted as locked, not lost
    assert d["rent_locked_lamports"] == GOAT_RENT


def test_unreadable_balance_prevents_close_eligibility():
    assert close_eligibility(acct(amount=None), expected_owner=WALLET)[0] is False


# -- 17. reclaim accounting state machine ----------------------------------


def test_full_lifecycle_created_to_reclaimable_to_reclaimed():
    led = fresh_ledger()
    assert led.lifecycle is RentLifecycleState.POSITION_OPEN
    led.on_exit(acct(amount=0), expected_owner=WALLET)
    assert led.lifecycle is RentLifecycleState.RENT_RECLAIMABLE
    assert led.reclaim_status is ReclaimStatus.RECLAIMABLE_PLANNED
    seen = [h[0] for h in led.history]
    assert seen == ["TOKEN_ACCOUNT_CREATED_BY_TRADE", "POSITION_OPEN",
                    "TOKEN_BALANCE_ZERO", "CLOSE_ELIGIBLE", "RENT_RECLAIMABLE"]
    led.on_reclaimed(GOAT_RENT)
    assert led.lifecycle is RentLifecycleState.RENT_RECLAIMED
    assert led.reclaim_status is ReclaimStatus.RECLAIMED
    d = led.as_dict()
    assert d["reclaimed_rent_lamports"] == GOAT_RENT
    assert d["rent_locked_lamports"] == 0
    assert d["reclaimable_rent_lamports"] == 0


def test_preexisting_account_skips_creation_states():
    led = RentLedger(trade_id="t3", mint="M", token_account="ATA1")
    led.on_entry(acct(exists=True), 0, RentBasis.MEASURED_PREEEXISTING)
    assert led.lifecycle is RentLifecycleState.TOKEN_ACCOUNT_PREEXISTING
    assert led.entry_rent_lamports == 0
    d = led.as_dict()
    assert d["entry_rent_lamports"] == 0
    assert d["rent_locked_lamports"] == 0


def test_unknown_rent_blocks_the_trade_at_entry():
    led = RentLedger(trade_id="t4", mint="M", token_account=None)
    led.on_entry(acct(exists=None), None, RentBasis.UNKNOWN)
    assert led.lifecycle is RentLifecycleState.BLOCKED
    assert led.reclaim_status is ReclaimStatus.BLOCKED
    assert led.entry_rent_lamports == 0


def test_on_chain_reclamation_is_declared_unimplemented():
    """Accounting and planning only. There is no closure code path."""
    led = fresh_ledger()
    led.on_exit(acct(amount=0), expected_owner=WALLET)
    d = led.as_dict()
    assert d["on_chain_reclamation_implemented"] is False
    assert d["reclaim_status"] == "RECLAIMABLE_PLANNED"
    for forbidden in ("execute", "broadcast", "send", "close_account"):
        assert not hasattr(RentLedger, forbidden), forbidden