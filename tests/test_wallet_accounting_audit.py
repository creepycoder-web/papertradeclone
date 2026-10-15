"""
Deterministic tests for the wallet-accounting / reconciliation / cleanup work.

No network, no signing, no broadcast. Every RPC double REFUSES to send.

Coverage:
  wallet      SPL detection, Token-2022 detection, mixed assets, empty accounts,
              unavailable-price handling
  settlement  SOL / USDC / other-token settlement classification
  reconciliation  external sell, manual Phantom sell, missing position,
              duplicated state
  cleanup     empty-ATA detection and every safe-close rule
  execution   no broadcast; execute_cleanup refuses without confirmation
  freshness   price ages and stale verdict
"""


from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.execution

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from autopaper.execution.solana import (                    # noqa: E402
    SolanaRpc,
    TOKEN_2022_PROGRAM_ID,
    TOKEN_PROGRAM_ID,
    TOKEN_PROGRAMS,
    WSOL_MINT,
)
from autopaper.price_freshness import PriceTrace              # noqa: E402
from autopaper.reconciliation import (                        # noqa: E402
    BALANCE_UNREADABLE,
    CONSISTENT_CLOSED,
    CONSISTENT_OPEN,
    EXTERNALLY_CLOSED,
    MANUAL_SETTLEMENT,
    SETTLED_NON_SOL,
    UNKNOWN_EXTERNAL_ACTIVITY,
    USDC_MINT,
    classify_position_against_chain,
    classify_settlement_asset,
    reconcile_journal_against_chain,
)
from autopaper.token_cleanup import (                         # noqa: E402
    CONFIRM_TOKEN,
    SKIPPED_AMBIGUOUS,
    audit_close_instructions,
    build_close_instructions,
    execute_cleanup,
    plan_token_account_cleanup,
)
from autopaper.wallet_assets import (                         # noqa: E402
    VALUATION_UNAVAILABLE,
    scan_wallet,
    summarize_scan,
)

OWNER = "wUEUbVmz2EfcLfchHmucTKhRpZgDpsBvo3FttZ4Wbt6"
SPL_MINT = "So11111111111111111111111111111111111111112"      # WSOL (classic)
T2022_MINT = "552y8SENvNqmFH6MR4LNiWd89DX8av9sYBLDsnkDpump"   # $SI (Token-2022)
OTHER_MINT = "HeEC24Fabc1111111111111111111111111111111111"


def token_acct(program, mint, amount, decimals, addr, owner=OWNER,
               lamports=2_039_280):
    return {
        "pubkey": addr,
        "account": {
            "lamports": lamports,
            "owner": program,
            "data": {"parsed": {"info": {
                "mint": mint, "owner": owner,
                "tokenAmount": {"amount": str(amount), "decimals": decimals},
            }}},
        },
    }


class _Rpc:
    """Duck-typed RPC. Records every method call; broadcasts nothing."""

    def __init__(self, *, lamports=6_408_463, by_program=None, rows=None,
                 raise_balance=False):
        self.lamports = lamports
        self.by_program = by_program or {}
        self.rows = rows
        self.raise_balance = raise_balance
        self.calls: list[str] = []

    async def get_balance_lamports(self, address):
        self.calls.append("get_balance_lamports")
        if self.raise_balance:
            raise RuntimeError("rpc down")
        return self.lamports

    async def token_balances(self, owner):
        self.calls.append("token_balances")
        if self.rows is not None:
            return list(self.rows)
        out = []
        for program, accts in self.by_program.items():
            for a in accts:
                out.append({
                    "mint": a["mint"], "amount": a["amount"],
                    "decimals": a["decimals"], "program": program,
                    "account": a["account"], "lamports": a["lamports"],
                })
        return out


# ===========================================================================
# 1. WALLET SCANNER
# ===========================================================================
class TestRealRpcTokenScan:
    """The RPC itself must query BOTH token programs."""

    def _rpc_with(self, payload):
        rpc = SolanaRpc("https://rpc.invalid")

        async def _fake(method, params):
            assert method == "getTokenAccountsByOwner"
            program = params[1]["programId"]
            return {"value": {"accounts": payload.get(program, [])}}

        rpc._rpc = _fake                                   # type: ignore[assignment]
        return rpc

    def test_both_token_programs_are_queried(self):
        seen: list[str] = []

        async def _fake(method, params):
            seen.append(params[1]["programId"])
            return {"value": {"accounts": []}}

        rpc = SolanaRpc("https://rpc.invalid")
        rpc._rpc = _fake                                   # type: ignore[assignment]
        asyncio.run(rpc.token_balances(OWNER))
        assert set(seen) == set(TOKEN_PROGRAMS)
        assert TOKEN_2022_PROGRAM_ID in seen

    def test_a_token_2022_account_is_detected(self):
        payload = {
            TOKEN_2022_PROGRAM_ID: [
                token_acct(TOKEN_2022_PROGRAM_ID, T2022_MINT, 4_061_052_725, 6,
                           "ATA-2022")],
        }
        rpc = self._rpc_with(payload)
        rows = asyncio.run(rpc.token_balances(OWNER))
        assert len(rows) == 1
        row = rows[0]
        assert row["mint"] == T2022_MINT
        assert row["amount"] == 4_061_052_725
        assert row["decimals"] == 6
        assert row["program"] == TOKEN_2022_PROGRAM_ID
        assert row["account"] == "ATA-2022"
        assert row["lamports"] == 2_039_280

    def test_a_classic_account_is_detected(self):
        payload = {TOKEN_PROGRAM_ID: [
            token_acct(TOKEN_PROGRAM_ID, SPL_MINT, 1_000, 9, "ATA-SPL")]}
        rows = asyncio.run(self._rpc_with(payload).token_balances(OWNER))
        assert [r["mint"] for r in rows] == [SPL_MINT]
        assert rows[0]["program"] == TOKEN_PROGRAM_ID

    def test_mixed_wallet_returns_both(self):
        payload = {
            TOKEN_PROGRAM_ID: [token_acct(TOKEN_PROGRAM_ID, SPL_MINT, 5, 9, "A")],
            TOKEN_2022_PROGRAM_ID: [
                token_acct(TOKEN_2022_PROGRAM_ID, T2022_MINT, 0, 6, "B"),
                token_acct(TOKEN_2022_PROGRAM_ID, OTHER_MINT, 7, 6, "C"),
            ],
        }
        rows = asyncio.run(self._rpc_with(payload).token_balances(OWNER))
        progs = {r["program"] for r in rows}
        assert progs == {TOKEN_PROGRAM_ID, TOKEN_2022_PROGRAM_ID}
        assert len(rows) == 3


class TestWalletScan:
    def test_mixed_assets_are_all_reported(self):
        rpc = _Rpc(by_program={
            TOKEN_PROGRAM_ID: [
                {"mint": SPL_MINT, "amount": 5_000_000, "decimals": 9,
                 "account": "A", "lamports": 2_039_280}],
            TOKEN_2022_PROGRAM_ID: [
                {"mint": T2022_MINT, "amount": 4_061_052_725, "decimals": 6,
                 "account": "B", "lamports": 2_039_280}],
        })
        scan = asyncio.run(scan_wallet(
            rpc, OWNER, sol_price_usd=118.0,
            price_of=lambda mint, symbol: 1.0 if mint == T2022_MINT else None,
            now=1000.0, wallet_masked="wUEU…bt6"))
        assert scan.native_lamports == 6_408_463
        assert len(scan.assets) == 2
        # WSOL is valued at the SOL price automatically
        wsol = next(a for a in scan.assets if a.mint == SPL_MINT)
        assert wsol.value_usd == pytest.approx(5_000_000 / 1e9 * 118.0)
        # the Token-2022 asset is detected and priced
        t22 = next(a for a in scan.assets if a.mint == T2022_MINT)
        assert t22.is_token_2022
        assert t22.asset_detected
        assert t22.valuation == "priced"
        assert scan.equity_usd is not None

    def test_unknown_asset_is_detected_but_not_zeroed(self):
        """The core rule: an unpriced asset is DETECTED and UNVALUED, not zero."""
        rpc = _Rpc(by_program={TOKEN_2022_PROGRAM_ID: [
            {"mint": T2022_MINT, "amount": 123, "decimals": 6,
             "account": "B", "lamports": 2_039_280}]})
        scan = asyncio.run(scan_wallet(
            rpc, OWNER, sol_price_usd=118.0, price_of=lambda m, s: None,
            wallet_masked="wUEU…bt6"))
        asset = scan.assets[0]
        assert asset.asset_detected is True
        assert asset.valuation == VALUATION_UNAVAILABLE
        assert asset.price_usd is None
        assert asset.value_usd is None
        assert scan.unpriced_mints == (T2022_MINT,)
        # equity is UNAVAILABLE, never a partial sum
        assert scan.equity_usd is None
        assert scan.available is False

    def test_empty_accounts_report_rent(self):
        rpc = _Rpc(by_program={TOKEN_2022_PROGRAM_ID: [
            {"mint": T2022_MINT, "amount": 0, "decimals": 6,
             "account": "B", "lamports": 2_039_280},
            {"mint": OTHER_MINT, "amount": 0, "decimals": 6,
             "account": "C", "lamports": 2_039_280}]})
        scan = asyncio.run(scan_wallet(
            rpc, OWNER, sol_price_usd=118.0, wallet_masked="wUEU…bt6"))
        assert len(scan.empty_accounts) == 2
        assert scan.rent_recoverable_lamports == 4_078_560
        # empty accounts do not make equity unavailable
        assert scan.equity_usd == pytest.approx(6_408_463 / 1e9 * 118.0)

    def test_unreadable_balance_is_unavailable_not_zero(self):
        rpc = _Rpc(raise_balance=True)
        scan = asyncio.run(scan_wallet(rpc, OWNER, sol_price_usd=118.0))
        assert scan.native_lamports is None
        assert scan.balance_readable is False
        assert scan.equity_usd is None

    def test_token_2022_count_in_summary(self):
        rpc = _Rpc(by_program={TOKEN_2022_PROGRAM_ID: [
            {"mint": T2022_MINT, "amount": 1, "decimals": 6, "account": "B",
             "lamports": 2_039_280}]})
        scan = asyncio.run(scan_wallet(rpc, OWNER, sol_price_usd=118.0,
                                       price_of=lambda m, s: 1.0))
        assert summarize_scan(scan)["token_2022_count"] == 1


# ===========================================================================
# 2. SETTLEMENT CLASSIFICATION
# ===========================================================================
class TestSettlementClassification:
    def test_sol_settlement(self):
        state, is_sol = classify_settlement_asset("SOL")
        assert state == EXTERNALLY_CLOSED and is_sol is True

    def test_wsol_settlement_is_sol(self):
        state, is_sol = classify_settlement_asset(WSOL_MINT)
        assert state == EXTERNALLY_CLOSED and is_sol is True

    def test_usdc_settlement(self):
        state, is_sol = classify_settlement_asset(USDC_MINT)
        assert state == SETTLED_NON_SOL and is_sol is False

    def test_other_token_settlement(self):
        state, is_sol = classify_settlement_asset(OTHER_MINT)
        assert state == SETTLED_NON_SOL and is_sol is False

    def test_unobserved_asset_is_external_not_zero(self):
        state, is_sol = classify_settlement_asset("")
        assert state == EXTERNALLY_CLOSED and is_sol is False


# ===========================================================================
# 3. RECONCILIATION
# ===========================================================================
class TestReconciliation:
    def test_external_sell_settling_in_usdc(self):
        """The $SI shape: journal OPEN, chain 0, settled in USDC."""
        f = classify_position_against_chain(
            mint=T2022_MINT, journal_status="OPEN", chain_balance=0,
            settlement_asset=USDC_MINT)
        assert f.state == SETTLED_NON_SOL
        assert f.is_sol_settlement is False
        assert f.requires_operator is True

    def test_external_close_without_observed_asset(self):
        f = classify_position_against_chain(
            mint=T2022_MINT, journal_status="OPEN", chain_balance=0)
        assert f.state == EXTERNALLY_CLOSED

    def test_manual_phantom_sell(self):
        f = classify_position_against_chain(
            mint=T2022_MINT, journal_status="OPEN", chain_balance=0, manual=True)
        assert f.state == MANUAL_SETTLEMENT

    def test_missing_position_open_and_holding(self):
        f = classify_position_against_chain(
            mint=T2022_MINT, journal_status="OPEN", chain_balance=100)
        assert f.state == CONSISTENT_OPEN and f.consistent

    def test_duplicated_state_journal_closed_chain_holds(self):
        f = classify_position_against_chain(
            mint=T2022_MINT, journal_status="CLOSED", chain_balance=100)
        assert f.state == UNKNOWN_EXTERNAL_ACTIVITY
        assert f.fails_closed

    def test_unreadable_balance_fails_closed(self):
        f = classify_position_against_chain(
            mint=T2022_MINT, journal_status="OPEN", chain_balance=None)
        assert f.state == BALANCE_UNREADABLE

    def test_journal_and_chain_consistent_closed(self):
        f = classify_position_against_chain(
            mint=T2022_MINT, journal_status="CLOSED", chain_balance=0)
        assert f.state == CONSISTENT_CLOSED

    def test_reconcile_report_never_mutates_and_names_actionable(self):
        class _Pos:
            def __init__(self, mint, status):
                self.mint = mint
                self.symbol = "SI"
                self.status = status

        report = reconcile_journal_against_chain(
            [_Pos(T2022_MINT, "OPEN")],
            {T2022_MINT: 0},
            settlements={T2022_MINT: USDC_MINT},
        )
        assert report["checked"] == 1
        assert report["consistent"] is False
        assert report["by_state"][SETTLED_NON_SOL] == 1
        assert report["actionable"][0]["state"] == SETTLED_NON_SOL


# ===========================================================================
# 4. CLEANUP
# ===========================================================================
def _acct_row(mint, amount=0, program=TOKEN_2022_PROGRAM_ID, owner=OWNER,
              account="ATA", lamports=2_039_280, close_authority=OWNER,
              delegate=None, state="initialized", chain=True, owner_program=None,
              frozen=False):
    """
    A listing row PLUS the chain's own reading of that account.

    ``chain=True`` attaches a ``chain_state`` because the planner refuses to
    close an account it has not read from the chain. That default is the whole
    point of the change; ``chain=False`` is how a test proves the refusal.
    """
    row = {"mint": mint, "amount": amount, "decimals": 6, "program": program,
           "owner": owner, "account": account, "lamports": lamports}
    if chain:
        row["chain_state"] = {
            "address": account, "exists": True, "is_token_account": True,
            "mint": mint, "token_owner": owner, "state": state,
            "amount": amount, "decimals": 6, "delegate": delegate,
            "close_authority": close_authority,
            "close_authority_option": 1 if close_authority else 0,
            "frozen": frozen,
            "owner_program": owner_program or program,
            "lamports": lamports, "executable": False,
        }
    return row


class TestCleanupRules:
    def test_empty_ata_is_safe_to_close(self):
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT)], owner=OWNER)
        assert plan.accounts_found == 1
        assert len(plan.safe_to_close) == 1
        assert plan.estimated_rent_return_lamports == 2_039_280
        assert plan.safe_to_close[0].is_token_2022

    def test_empty_classic_spl_account_is_safe_to_close(self):
        plan = plan_token_account_cleanup(
            [_acct_row("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                       program=TOKEN_PROGRAM_ID, account="ClassicAta",
                       lamports=1_488_440)],
            owner=OWNER)
        assert len(plan.safe_to_close) == 1
        assert not plan.safe_to_close[0].is_token_2022
        assert plan.safe_to_close[0].program == TOKEN_PROGRAM_ID
        assert plan.estimated_rent_return_lamports == 1_488_440

    def test_empty_token_2022_account_is_safe_to_close(self):
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT, program=TOKEN_2022_PROGRAM_ID,
                       account="A2022", lamports=1_513_840)],
            owner=OWNER)
        assert len(plan.safe_to_close) == 1
        assert plan.safe_to_close[0].is_token_2022
        assert plan.estimated_rent_return_lamports == 1_513_840

    def test_no_close_authority_is_held_until_the_chain_proves_it(self):
        """A None close authority is HELD, never closed on inference."""
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT, close_authority=None)], owner=OWNER)
        assert not plan.safe_to_close
        assert len(plan.simulation_required) == 1
        assert plan.estimated_rent_return_lamports == 0

    def test_chain_proof_promotes_a_held_account(self):
        row = _acct_row(T2022_MINT, close_authority=None)
        plan = plan_token_account_cleanup(
            [row], owner=OWNER,
            closability={row["account"]: {"closable": True, "unproven": False,
                                         "simulated": True,
                                         "detail": "simulation ok"}})
        assert len(plan.safe_to_close) == 1
        assert not plan.simulation_required
        assert "the chain confirms" in plan.safe_to_close[0].reason

    def test_chain_refusal_blocks_the_account(self):
        row = _acct_row(T2022_MINT, close_authority=None)
        plan = plan_token_account_cleanup(
            [row], owner=OWNER,
            closability={row["account"]: {"closable": False, "unproven": False,
                                         "simulated": True,
                                         "detail": "SIMULATION_FAILED_TX_REJECTED"}})
        assert not plan.safe_to_close
        assert len(plan.blocked) == 1
        assert "the chain does not accept the close" in plan.blocked[0].reason

    def test_transport_failure_is_unproven_not_a_refusal(self):
        """A 429 must not masquerade as the chain rejecting the close."""
        row = _acct_row(T2022_MINT, close_authority=None)
        plan = plan_token_account_cleanup(
            [row], owner=OWNER,
            closability={row["account"]: {"closable": False, "unproven": True,
                                         "simulated": False,
                                         "detail": "RPC HTTP 429"}})
        assert not plan.safe_to_close
        assert len(plan.simulation_required) == 1
        assert not plan.blocked

    def test_foreign_close_authority_blocks_close(self):
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT,
                       close_authority="SomeoneElse1111111111111111111111")],
            owner=OWNER)
        assert not plan.safe_to_close
        assert "close authority is" in plan.blocked[0].reason

    def test_live_delegate_blocks_close(self):
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT, delegate="Delegate1111111111111111111111")],
            owner=OWNER)
        assert not plan.safe_to_close
        assert "delegate" in plan.blocked[0].reason

    def test_frozen_account_blocks_close(self):
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT, frozen=True)], owner=OWNER)
        assert not plan.safe_to_close
        assert "frozen" in plan.blocked[0].reason

    def test_uninitialized_state_blocks_close(self):
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT, state="frozen")], owner=OWNER)
        assert not plan.safe_to_close
        assert "not 'initialized'" in plan.blocked[0].reason

    def test_row_without_chain_state_is_refused(self):
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT, chain=False)], owner=OWNER)
        assert not plan.safe_to_close
        assert len(plan.simulation_required) == 0
        assert "no on-chain state" in plan.blocked[0].reason

    def test_on_chain_owner_disagreement_blocks_even_when_the_row_agrees(self):
        row = _acct_row(T2022_MINT)
        row["chain_state"]["token_owner"] = "SomeoneElse1111111111111111111111"
        plan = plan_token_account_cleanup([row], owner=OWNER)
        assert not plan.safe_to_close
        assert "on-chain owner" in plan.blocked[0].reason

    def test_program_mismatch_blocks_close(self):
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT, program=TOKEN_PROGRAM_ID,
                       owner_program=TOKEN_2022_PROGRAM_ID)],
            owner=OWNER)
        assert not plan.safe_to_close
        assert "does not match the row's program" in plan.blocked[0].reason

    def test_non_token_account_is_refused(self):
        row = _acct_row(T2022_MINT)
        row["chain_state"]["is_token_account"] = False
        row["chain_state"]["owner_program"] = "11111111111111111111111111111111"
        plan = plan_token_account_cleanup([row], owner=OWNER)
        assert not plan.safe_to_close
        assert "not a token account" in plan.blocked[0].reason

    def test_disagreement_between_row_and_chain_amount_blocks(self):
        row = _acct_row(T2022_MINT, amount=0)
        row["chain_state"]["amount"] = 7
        plan = plan_token_account_cleanup([row], owner=OWNER)
        assert not plan.safe_to_close
        assert "not zero" in plan.blocked[0].reason or "disagrees" in (
            plan.blocked[0].reason)

    def test_wrapped_sol_account_is_never_closed(self):
        plan = plan_token_account_cleanup(
            [_acct_row(WSOL_MINT, program=TOKEN_PROGRAM_ID, account="WsolAta")],
            owner=OWNER)
        assert not plan.safe_to_close
        assert "wrapped-SOL" in plan.blocked[0].reason

    def test_rent_destination_must_be_the_configured_wallet(self):
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT)], owner=OWNER,
            rent_destination="DeveloperWallet11111111111111111111")
        assert not plan.safe_to_close
        assert plan.rent_destination == "DeveloperWallet11111111111111111111"
        assert "rent destination" in plan.blocked[0].reason

    def test_rent_destination_defaults_to_the_configured_wallet(self):
        plan = plan_token_account_cleanup([_acct_row(T2022_MINT)], owner=OWNER)
        assert plan.rent_destination == OWNER
        assert plan.as_dict()["rent_destination_is_configured_wallet"] is True

    def test_ambiguous_runtime_reference_is_skipped_ambiguous(self):
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT)], owner=OWNER,
            reserved_mints=[T2022_MINT])
        assert not plan.safe_to_close
        assert len(plan.skipped_ambiguous) == 1
        assert SKIPPED_AMBIGUOUS in plan.skipped_ambiguous[0].reason

    def test_open_paper_position_is_ambiguity_not_a_plain_block(self):
        """An open PAPER position is a runtime reservation, not a chain fact."""
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT)], owner=OWNER, reserved_mints=[T2022_MINT],
            open_mints=[T2022_MINT])
        assert len(plan.skipped_ambiguous) == 0     # open_mints is checked first
        assert "OPEN position" in plan.blocked[0].reason

    def test_non_empty_is_blocked(self):
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT, amount=5)], owner=OWNER)
        assert not plan.safe_to_close
        assert "not zero" in plan.blocked[0].reason

    def test_open_position_blocks_close(self):
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT)], owner=OWNER, open_mints=[T2022_MINT])
        assert not plan.safe_to_close
        assert "OPEN position" in plan.blocked[0].reason

    def test_pending_transaction_blocks_close(self):
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT)], owner=OWNER, pending_mints=[T2022_MINT])
        assert not plan.safe_to_close
        assert "PENDING" in plan.blocked[0].reason

    def test_foreign_owner_blocks_close(self):
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT, owner="SomeoneElse1111111111111111111")],
            owner=OWNER)
        assert not plan.safe_to_close
        assert "not the configured wallet" in plan.blocked[0].reason

    def test_unknown_program_blocks_close(self):
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT, program="BogusProgram1111111111111111111111")],
            owner=OWNER)
        assert not plan.safe_to_close
        assert "unknown token program" in plan.blocked[0].reason

    def test_unnamed_account_blocks_close(self):
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT, account="")], owner=OWNER)
        assert not plan.safe_to_close
        assert "no token account address" in plan.blocked[0].reason

    def test_mixed_wallet_counts_only_safe_rent(self):
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT, account="A"),
             _acct_row(OTHER_MINT, amount=9, account="B")],
            owner=OWNER)
        assert len(plan.safe_to_close) == 1
        assert plan.estimated_rent_return_lamports == 2_039_280

    def test_close_instruction_uses_the_account_own_program(self):
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT,
                       account="11111111111111111111111111111111")],
            owner=OWNER)
        ixs = build_close_instructions(plan)
        assert len(ixs) == 1
        assert str(ixs[0].program_id) == TOKEN_2022_PROGRAM_ID
        assert bytes(ixs[0].data) == bytes([9])          # CloseAccount


class TestCleanupNoBroadcast:
    def test_execute_refuses_without_confirmation(self):
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT)], owner=OWNER)
        rpc = _Rpc()
        result = asyncio.run(execute_cleanup(rpc, wallet=None, plan=plan,
                                             confirm=""))
        assert result["executed"] is False
        assert "confirmation token" in result["reason"]
        assert rpc.calls == []          # nothing was even read, let alone sent

    def test_dry_run_is_the_default_and_broadcasts_nothing(self):
        plan = plan_token_account_cleanup(
            [_acct_row(T2022_MINT)], owner=OWNER)
        # A plan is pure data; it holds no transaction and no signature.
        assert plan.has_safe_accounts
        assert not hasattr(plan, "signature")
        assert "DRY RUN" in plan.render()


# ===========================================================================
# 5. PRICE FRESHNESS
# ===========================================================================
class TestPriceFreshness:
    def test_ages_and_change(self):
        t = PriceTrace(
            signal_at=0.0, decision_at=2.0, quote_at=2.9, execution_at=3.1,
            signal_price=1.8620e-05, decision_price=1.8620e-05,
            quote_price=2.7229e-05, fill_price=2.5e-05)
        assert t.price_age_at_decision_ms == pytest.approx(2000.0)
        assert t.price_age_at_quote_ms == pytest.approx(2900.0)
        assert t.decision_to_quote_ms == pytest.approx(900.0)
        change = t.price_change_before_fill_pct
        assert change == pytest.approx((2.5e-05 / 1.8620e-05 - 1) * 100)

    def test_missing_timestamp_is_none_not_zero(self):
        t = PriceTrace(signal_at=None, decision_at=5.0)
        assert t.price_age_at_decision_ms is None
        assert t.is_stale(15_000) is None       # unknown is not "fresh"

    def test_stale_verdict(self):
        t = PriceTrace(signal_at=0.0, decision_at=20.0)
        assert t.is_stale(15_000) is True
        t2 = PriceTrace(signal_at=0.0, decision_at=5.0)
        assert t2.is_stale(15_000) is False

    def test_from_stamps_uses_alert_as_signal(self):
        t = PriceTrace.from_stamps({
            "alert_received": 100.0, "l25_end": 102.0,
            "quote_start": 103.0, "fill": 104.0})
        assert t.price_age_at_decision_ms == pytest.approx(2000.0)
        assert t.price_age_at_quote_ms == pytest.approx(3000.0)


# ===========================================================================
# 6. REAL SAFETY UNCHANGED
# ===========================================================================
class TestRealSafetyUnchanged:
    def test_real_trading_is_not_armed_by_this_work(self):
        from autopaper.config import build_config
        cfg = build_config()
        # REAL is armed only by a confirmed Telegram operator action plus a
        # passing preflight; no config value can arm it.
        assert cfg.real.trading_enabled is False
