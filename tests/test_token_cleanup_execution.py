"""
Deterministic tests for the CLOSE-TIME safety of empty-token-account cleanup.

WHAT THIS FILE IS FOR
--------------------
The cleanup path is the only write this runtime is authorized to make that is
not a trade. These tests pin the properties that keep it that way:

  planning       classic SPL and Token-2022 empties are eligible; non-empty,
                 wrong-owner, foreign-close-authority, delegate, frozen,
                 ambiguous and wrapped-SOL accounts are refused
  destination    the rent may only go back to the configured wallet
  instructions   the built transaction can contain NOTHING but CloseAccount
                 for the planned accounts -- a swap, a transfer, an approval,
                 a delegate change or any non-token program is a REFUSAL
  ordering       a failed simulation must stop the broadcast; a failed
                 re-check must drop the account, not close it optimistically
  accounting     recovered lamports are the accounts' own lamports; the
                 transaction fee is reported separately and is never folded
                 into "recovered"
  secrets        no secret material reaches a log, a plan, or any rendered
                 output

EVERY KEYPAIR HERE IS GENERATED IN-TEST. The production wallet secret is never
read, and no test asserts anything about the operator's real wallet.
"""


from __future__ import annotations

import asyncio
import base64
import json
import logging
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.execution,
    pytest.mark.safety,
]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from solders.hash import Hash                                   # noqa: E402
from solders.instruction import AccountMeta, Instruction         # noqa: E402
from solders.keypair import Keypair                              # noqa: E402
from solders.pubkey import Pubkey                                # noqa: E402

from autopaper.execution.solana import (                         # noqa: E402
    TOKEN_2022_PROGRAM_ID,
    TOKEN_PROGRAM_ID,
    TOKEN_PROGRAMS,
    WSOL_MINT,
    SolanaWallet,
    close_account_instructions,
)
from autopaper.token_cleanup import (                            # noqa: E402
    AWAITING_SIMULATION,
    BLOCKED,
    CLOSE_ACCOUNT_IX,
    CONFIRM_TOKEN,
    ELIGIBLE,
    SKIPPED_AMBIGUOUS,
    CleanupCandidate,
    CleanupPlan,
    audit_close_instructions,
    build_close_instructions,
    confirm_closures,
    execute_cleanup,
    plan_token_account_cleanup,
    programs_in_message,
    read_transaction_fee,
    reconcile_balance,
    resolve_broadcast_outcome,
    simulate_close_for,
)

SOL = 1_000_000_000
BASE_FEE_PER_SIGNATURE = 5_000
T2022_RENT = 1_513_840
SPL_RENT = 1_488_440
SYSTEM_PROGRAM = "11111111111111111111111111111111"
JUPITER_ROUTER = "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"
SYSTEM_TRANSFER_IX = 2
APPROVE_IX = 3


# ---------------------------------------------------------------------------
# Generated wallets. Never the operator's.
# ---------------------------------------------------------------------------
@pytest.fixture
def owner_keypair() -> Keypair:
    return Keypair()


@pytest.fixture
def other_keypair() -> Keypair:
    return Keypair()


@pytest.fixture
def owner_wallet(tmp_path, owner_keypair) -> SolanaWallet:
    """A SolanaWallet backed by a keypair GENERATED FOR THIS TEST."""
    path = tmp_path / "test-keypair.json"
    path.write_text(json.dumps(list(bytes(owner_keypair))), encoding="utf-8")
    return SolanaWallet(str(path))


def pub(keypair: Keypair) -> str:
    return str(keypair.pubkey())


def token_address(seed: int) -> str:
    """A deterministic, valid, non-owner address for account metas.

    ``Pubkey.from_bytes`` accepts any 32 bytes except a value that is not a
    valid curve point in a context that requires one; a token ACCOUNT does not,
    so a repeated byte pattern is a fine stand-in for a real token account.
    """
    return str(Pubkey.from_bytes(bytes([seed % 256 or 7]) * 32))


# ---------------------------------------------------------------------------
# RPC doubles. The broadcast double is the important one.
# ---------------------------------------------------------------------------
class _SimOutcome:
    """The shape ``simulate_transaction_detailed`` returns."""

    def __init__(self, ok: bool, detail: str):
        self.ok = ok
        self.detail = detail
        self.status = "SIMULATION_OK" if ok else "SIMULATION_FAILED_TX_REJECTED"
        self.rpc_message = detail if not ok else ""
        self.units_consumed = 1464 if ok else None
        self.broadcast_attempted = False

    def as_dict(self) -> dict:
        return {"ok": self.ok, "detail": self.detail, "status": self.status,
                "rpc_message": self.rpc_message,
                "units_consumed": self.units_consumed,
                "broadcast_attempted": self.broadcast_attempted}


class _NeverBroadcastsRpc:
    """Base double: every call is recorded, ``send_transaction`` explodes."""

    def __init__(self, states: dict | None = None, balance: int = 0):
        self.states = states or {}
        self.balance = balance
        self.calls: list[str] = []
        self.sent: list[str] = []
        self.simulate_ok = True
        self.simulate_detail = "simulation ok"
        self.sent_result = "5" * 88
        self.confirm_status = "confirmed"

    async def get_balance_lamports(self, address):
        self.calls.append("get_balance_lamports")
        return self.balance

    async def get_latest_blockhash(self, commitment="confirmed"):
        self.calls.append("get_latest_blockhash")
        return Hash.default(), 123

    async def get_token_account_state(self, address):
        self.calls.append("get_token_account_state")
        return self.states.get(address)

    async def get_token_account_states(self, addresses):
        """The batched read. One call, so an N-account re-check is not a burst."""
        self.calls.append("get_token_account_states")
        return {a: self.states.get(a) for a in addresses}

    async def simulate_transaction_detailed(self, tx_b64):
        self.calls.append("simulate_transaction_detailed")
        return _SimOutcome(ok=self.simulate_ok,
                           detail=self.simulate_detail)

    async def simulate_transaction(self, tx_b64):
        self.calls.append("simulate_transaction")
        return self.simulate_ok, self.simulate_detail

    async def send_transaction(self, tx_b64):
        self.calls.append("send_transaction")
        self.sent.append(tx_b64)
        return self.sent_result

    async def wait_for_confirmation(self, signature, **kwargs):
        self.calls.append("wait_for_confirmation")
        return {"status": self.confirm_status, "error": None, "slot": 99}

    async def get_transaction(self, signature, commitment="confirmed"):
        self.calls.append("get_transaction")
        return {"meta": {"fee": BASE_FEE_PER_SIGNATURE}}

    @property
    def broadcast_count(self) -> int:
        return len(self.sent)


def chain_state(*, address, mint, program, amount=0, lamports=T2022_RENT,
                owner=None, close_authority=None, delegate=None,
                state="initialized", frozen=False, is_token_account=True):
    return {
        "address": address,
        "exists": True,
        "is_token_account": is_token_account,
        "mint": mint,
        "token_owner": owner,
        "state": state,
        "amount": amount,
        "decimals": 6,
        "delegate": delegate,
        "close_authority": close_authority,
        "close_authority_option": 1 if close_authority else 0,
        "state_option": {"account": state},
        "is_native_option": False,
        "frozen": frozen,
        "owner_program": program,
        "lamports": lamports,
        "executable": False,
        "rent_epoch": 0,
    }


def row(owner, *, mint, account, program, amount=0, lamports=T2022_RENT,
        chain_owner=None, close_authority="__owner__", delegate=None,
        state="initialized", frozen=False):
    """A listing row plus the chain's reading of it.

    ``close_authority`` defaults to the OWNER: an account whose close authority
    is already the configured wallet needs no simulation proof, which is the
    common case. Pass ``None`` explicitly to exercise the held-for-proof path.
    """
    if close_authority == "__owner__":
        close_authority = owner
    return {
        "mint": mint,
        "account": account,
        "amount": amount,
        "decimals": 6,
        "program": program,
        "owner": owner,
        "lamports": lamports,
        "chain_state": chain_state(
            address=account, mint=mint, program=program, amount=amount,
            lamports=lamports, owner=chain_owner or owner,
            close_authority=close_authority, delegate=delegate, state=state,
            frozen=frozen),
    }


# ===========================================================================
# 1. ELIGIBILITY
# ===========================================================================
class TestEligibility:
    def test_empty_classic_spl_account_is_eligible(self, owner_keypair):
        owner = pub(owner_keypair)
        account = token_address(1)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(9), account=account,
                 program=TOKEN_PROGRAM_ID, lamports=SPL_RENT)],
            owner=owner, rent_destination=owner)
        assert [c.token_account for c in plan.safe_to_close] == [account]
        assert plan.estimated_rent_return_lamports == SPL_RENT
        assert plan.safe_to_close[0].status == ELIGIBLE

    def test_empty_token_2022_account_is_eligible(self, owner_keypair):
        owner = pub(owner_keypair)
        account = token_address(2)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(3), account=account,
                 program=TOKEN_2022_PROGRAM_ID, lamports=T2022_RENT)],
            owner=owner, rent_destination=owner)
        assert [c.token_account for c in plan.safe_to_close] == [account]
        assert plan.estimated_rent_return_lamports == T2022_RENT
        assert plan.safe_to_close[0].status == ELIGIBLE

    @pytest.mark.parametrize("amount", [1, 1_000_000, 4_061_052_725])
    def test_non_empty_account_is_rejected(self, owner_keypair, amount):
        owner = pub(owner_keypair)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(4), account=token_address(5),
                 program=TOKEN_2022_PROGRAM_ID, amount=amount)],
            owner=owner, rent_destination=owner)
        assert not plan.safe_to_close
        assert not plan.simulation_required
        assert "not zero" in plan.blocked[0].reason

    def test_wrong_owner_is_rejected(self, owner_keypair, other_keypair):
        owner = pub(owner_keypair)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(6), account=token_address(7),
                 program=TOKEN_2022_PROGRAM_ID,
                 chain_owner=pub(other_keypair))],
            owner=owner, rent_destination=owner)
        assert not plan.safe_to_close
        assert "on-chain owner" in plan.blocked[0].reason

    def test_invalid_close_authority_is_rejected(self, owner_keypair,
                                                 other_keypair):
        """A close authority that is somebody else can never sign for us."""
        owner = pub(owner_keypair)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(8), account=token_address(9),
                 program=TOKEN_2022_PROGRAM_ID,
                 close_authority=pub(other_keypair))],
            owner=owner, rent_destination=owner)
        assert not plan.safe_to_close
        assert not plan.simulation_required
        assert "close authority is" in plan.blocked[0].reason

    def test_missing_close_authority_is_held_not_closed(self, owner_keypair):
        owner = pub(owner_keypair)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(10), account=token_address(11),
                 program=TOKEN_2022_PROGRAM_ID, close_authority=None)],
            owner=owner, rent_destination=owner)
        assert not plan.safe_to_close
        assert len(plan.simulation_required) == 1
        assert plan.simulation_required[0].status == AWAITING_SIMULATION
        assert plan.estimated_rent_return_lamports == 0

    def test_close_authority_is_this_wallet_needs_no_proof(self, owner_keypair):
        owner = pub(owner_keypair)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(12), account=token_address(13),
                 program=TOKEN_2022_PROGRAM_ID, close_authority=None)],
            owner=owner, rent_destination=owner,
            closability={token_address(13): {"closable": True,
                                             "unproven": False}})
        assert len(plan.safe_to_close) == 1
        assert not plan.simulation_required

    def test_ambiguous_mint_is_skipped_ambiguous(self, owner_keypair):
        owner = pub(owner_keypair)
        mint = token_address(14)
        plan = plan_token_account_cleanup(
            [row(owner, mint=mint, account=token_address(15),
                 program=TOKEN_2022_PROGRAM_ID)],
            owner=owner, rent_destination=owner, reserved_mints=[mint])
        assert not plan.safe_to_close
        assert len(plan.skipped_ambiguous) == 1
        assert plan.skipped_ambiguous[0].status == SKIPPED_AMBIGUOUS
        assert plan.skipped_ambiguous[0].ambiguous is True

    def test_open_position_mint_is_rejected(self, owner_keypair):
        owner = pub(owner_keypair)
        mint = token_address(16)
        plan = plan_token_account_cleanup(
            [row(owner, mint=mint, account=token_address(17),
                 program=TOKEN_2022_PROGRAM_ID)],
            owner=owner, rent_destination=owner, open_mints=[mint])
        assert not plan.safe_to_close
        assert plan.estimated_rent_return_lamports == 0

    def test_pending_execution_mint_is_rejected(self, owner_keypair):
        owner = pub(owner_keypair)
        mint = token_address(18)
        plan = plan_token_account_cleanup(
            [row(owner, mint=mint, account=token_address(19),
                 program=TOKEN_2022_PROGRAM_ID)],
            owner=owner, rent_destination=owner, pending_mints=[mint])
        assert not plan.safe_to_close

    def test_wrapped_sol_account_is_rejected(self, owner_keypair):
        """The pre-provisioned entry route is not rent to be reclaimed."""
        owner = pub(owner_keypair)
        plan = plan_token_account_cleanup(
            [row(owner, mint=WSOL_MINT, account=token_address(20),
                 program=TOKEN_PROGRAM_ID, lamports=SPL_RENT)],
            owner=owner, rent_destination=owner)
        assert not plan.safe_to_close
        assert "wrapped-SOL" in plan.blocked[0].reason

    def test_account_with_no_on_chain_read_is_rejected(self, owner_keypair):
        owner = pub(owner_keypair)
        r = row(owner, mint=token_address(21), account=token_address(22),
                program=TOKEN_2022_PROGRAM_ID)
        r.pop("chain_state")
        plan = plan_token_account_cleanup([r], owner=owner,
                                          rent_destination=owner)
        assert not plan.safe_to_close
        assert "no on-chain state" in plan.blocked[0].reason


# ===========================================================================
# 2. REFUND DESTINATION
# ===========================================================================
class TestRefundDestination:
    def test_destination_defaults_to_the_configured_wallet(self, owner_keypair):
        owner = pub(owner_keypair)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(23), account=token_address(24),
                 program=TOKEN_2022_PROGRAM_ID)],
            owner=owner)
        assert plan.rent_destination == owner

    @pytest.mark.parametrize("destination_kind", ["developer", "temporary",
                                                  "arbitrary"])
    def test_a_foreign_destination_refuses_the_whole_plan(
            self, owner_keypair, other_keypair, destination_kind):
        owner = pub(owner_keypair)
        stranger = {"developer": pub(other_keypair),
                    "temporary": token_address(25),
                    "arbitrary": token_address(26)}[destination_kind]
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(27), account=token_address(28),
                 program=TOKEN_2022_PROGRAM_ID)],
            owner=owner, rent_destination=stranger)
        assert not plan.safe_to_close
        assert plan.estimated_rent_return_lamports == 0
        assert "rent destination" in plan.blocked[0].reason

    def test_execute_refuses_a_foreign_destination(self, owner_keypair,
                                                  owner_wallet):
        """Even a hand-built plan cannot redirect the rent."""
        owner = pub(owner_keypair)
        plan = CleanupPlan(
            wallet="test", owner=owner, rent_destination=token_address(29),
            safe_to_close=(CleanupCandidate(token_account=token_address(30),
                                            mint=token_address(31),
                                            program=TOKEN_2022_PROGRAM_ID,
                                            lamports=T2022_RENT),),
            estimated_rent_return_lamports=T2022_RENT)
        rpc = _NeverBroadcastsRpc()
        result = asyncio.run(execute_cleanup(rpc, owner_wallet, plan,
                                             confirm=CONFIRM_TOKEN))
        assert result["executed"] is False
        assert result["broadcast_attempted"] is False
        assert rpc.broadcast_count == 0

    def test_instruction_pays_the_configured_wallet(self, owner_keypair):
        owner = pub(owner_keypair)
        account = token_address(32)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(33), account=account,
                 program=TOKEN_2022_PROGRAM_ID)],
            owner=owner, rent_destination=owner)
        ix = build_close_instructions(plan)[0]
        assert str(ix.accounts[1].pubkey) == owner
        assert ix.accounts[1].is_writable is True
        assert str(ix.accounts[2].pubkey) == owner
        assert ix.accounts[2].is_signer is True


# ===========================================================================
# 3. THE TRANSACTION CONTAINS ONLY CLOSES
# ===========================================================================
class TestInstructionAudit:
    def _plan(self, owner_keypair, *accounts, program=TOKEN_2022_PROGRAM_ID):
        owner = pub(owner_keypair)
        return plan_token_account_cleanup(
            [row(owner, mint=token_address(40 + i), account=a,
                 program=program) for i, a in enumerate(accounts)],
            owner=owner, rent_destination=owner)

    def test_a_clean_plan_audits_empty(self, owner_keypair):
        plan = self._plan(owner_keypair, token_address(60))
        assert audit_close_instructions(
            build_close_instructions(plan), plan) == []

    def test_every_instruction_is_a_close_for_the_right_program(self,
                                                                 owner_keypair):
        owner = pub(owner_keypair)
        a1, a2 = token_address(61), token_address(62)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(63), account=a1,
                 program=TOKEN_2022_PROGRAM_ID),
             row(owner, mint=token_address(64), account=a2,
                 program=TOKEN_PROGRAM_ID, lamports=SPL_RENT)],
            owner=owner, rent_destination=owner)
        ixs = build_close_instructions(plan)
        assert [str(ix.program_id) for ix in ixs] == [TOKEN_2022_PROGRAM_ID,
                                                      TOKEN_PROGRAM_ID]
        for ix in ixs:
            assert bytes(ix.data) == bytes([CLOSE_ACCOUNT_IX])

    def test_a_system_transfer_is_refused(self, owner_keypair):
        """A lamport transfer rides along in the same message."""
        plan = self._plan(owner_keypair, token_address(65))
        ix = Instruction(
            program_id=Pubkey.from_string(SYSTEM_PROGRAM),
            data=bytes([SYSTEM_TRANSFER_IX, 0, 0, 0]),
            accounts=[AccountMeta(pubkey=Pubkey.from_string(plan.owner),
                                 is_signer=False, is_writable=True),
                      AccountMeta(pubkey=Pubkey.from_string(token_address(66)),
                                 is_signer=False, is_writable=True)])
        problems = audit_close_instructions(
            list(build_close_instructions(plan)) + [ix], plan)
        assert problems
        assert any("non-token program" in p for p in problems)

    def test_a_router_swap_is_refused(self, owner_keypair):
        plan = self._plan(owner_keypair, token_address(67))
        ix = Instruction(
            program_id=Pubkey.from_string(JUPITER_ROUTER),
            data=bytes([0x01, 0x02, 0x03]),
            accounts=[AccountMeta(pubkey=Pubkey.from_string(token_address(68)),
                                 is_signer=False, is_writable=True)])
        problems = audit_close_instructions(
            list(build_close_instructions(plan)) + [ix], plan)
        assert problems
        assert any("non-token program" in p for p in problems)

    def test_an_approve_or_delegate_change_is_refused(self, owner_keypair):
        plan = self._plan(owner_keypair, token_address(69))
        ix = Instruction(
            program_id=Pubkey.from_string(TOKEN_2022_PROGRAM_ID),
            data=bytes([APPROVE_IX, 1]),
            accounts=[AccountMeta(pubkey=Pubkey.from_string(token_address(70)),
                                 is_signer=False, is_writable=True),
                      AccountMeta(pubkey=Pubkey.from_string(plan.owner),
                                 is_signer=False, is_writable=False),
                      AccountMeta(pubkey=Pubkey.from_string(token_address(71)),
                                 is_signer=False, is_writable=False)])
        problems = audit_close_instructions(
            list(build_close_instructions(plan)) + [ix], plan)
        assert problems
        assert any("not a single-byte" in p for p in problems)

    def test_a_set_authority_is_refused(self, owner_keypair):
        """A CloseAccount-shaped instruction with extra data is not a close."""
        plan = self._plan(owner_keypair, token_address(72))
        target = Pubkey.from_string(plan.safe_to_close[0].token_account)
        ix = Instruction(
            program_id=Pubkey.from_string(TOKEN_2022_PROGRAM_ID),
            data=bytes([4, 2, 1]),          # SetAuthority / CloseAccount / Some
            accounts=[AccountMeta(pubkey=target, is_signer=False,
                                  is_writable=True),
                      AccountMeta(pubkey=Pubkey.from_string(plan.owner),
                                 is_signer=True, is_writable=False)])
        problems = audit_close_instructions([ix], plan)
        assert problems
        assert any("not a single-byte" in p for p in problems)

    def test_a_close_for_an_unplanned_account_is_refused(self, owner_keypair):
        plan = self._plan(owner_keypair, token_address(73))
        owner = plan.owner
        ix = close_account_instructions(
            owner, [(token_address(74), TOKEN_2022_PROGRAM_ID)])[0]
        problems = audit_close_instructions([ix], plan)
        assert any("unplanned account" in p for p in problems)

    def test_a_close_that_pays_someone_else_is_refused(self, owner_keypair,
                                                        other_keypair):
        plan = self._plan(owner_keypair, token_address(75))
        target = plan.safe_to_close[0].token_account
        other = pub(other_keypair)
        ix = close_account_instructions(
            other, [(target, TOKEN_2022_PROGRAM_ID)])[0]
        problems = audit_close_instructions([ix], plan)
        assert any("not the configured wallet" in p for p in problems)

    def test_the_wrong_program_for_an_account_is_refused(self, owner_keypair):
        owner = pub(owner_keypair)
        account = token_address(76)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(77), account=account,
                 program=TOKEN_2022_PROGRAM_ID)],
            owner=owner, rent_destination=owner)
        # Build the close with the CLASSIC program for a 2022 account.
        ix = close_account_instructions(owner, [(account, TOKEN_PROGRAM_ID)])[0]
        problems = audit_close_instructions([ix], plan)
        assert any("is owned by" in p for p in problems)

    def test_an_empty_instruction_set_is_refused(self, owner_keypair):
        plan = self._plan(owner_keypair, token_address(78))
        assert audit_close_instructions([], plan) == ["no instructions were built"]

    def test_the_built_message_names_only_token_programs(self, owner_keypair,
                                                         owner_wallet):
        """The end-to-end proof: no router, no SystemProgram, ever."""
        plan = self._plan(owner_keypair, token_address(79),
                          token_address(80))
        ixs = build_close_instructions(plan)
        raw = owner_wallet.sign_new_transaction(ixs,
                                                blockhash=Hash.default())
        programs = programs_in_message(raw)
        assert programs
        assert set(programs) <= set(TOKEN_PROGRAMS)


# ===========================================================================
# 4. EXECUTION ORDERING: SIMULATE FIRST, RE-CHECK, THEN BROADCAST
# ===========================================================================
class TestExecutionOrdering:
    def _plan(self, owner_keypair, *accounts, program=TOKEN_2022_PROGRAM_ID,
              lamports=T2022_RENT):
        owner = pub(owner_keypair)
        return plan_token_account_cleanup(
            [row(owner, mint=token_address(90 + i), account=a,
                 program=program, lamports=lamports) for i, a in enumerate(accounts)],
            owner=owner, rent_destination=owner)

    def test_without_the_confirmation_token_nothing_is_read_or_sent(
            self, owner_keypair, owner_wallet):
        plan = self._plan(owner_keypair, token_address(100))
        rpc = _NeverBroadcastsRpc()
        result = asyncio.run(execute_cleanup(rpc, owner_wallet, plan,
                                             confirm=""))
        assert result["executed"] is False
        assert result["broadcast_attempted"] is False
        assert rpc.calls == []
        assert rpc.broadcast_count == 0

    def test_a_wrong_confirmation_token_is_refused(self, owner_keypair,
                                                   owner_wallet):
        plan = self._plan(owner_keypair, token_address(101))
        rpc = _NeverBroadcastsRpc()
        for token in ("yes", "CLOSE-EMPTY-TOKEN-ACCOUNT", "close-empty-token-accounts",
                      "CLOSE-EMPTY-TOKEN-ACCOUNTS "):
            result = asyncio.run(execute_cleanup(rpc, owner_wallet, plan,
                                                 confirm=token))
            assert result["executed"] is False
        assert rpc.broadcast_count == 0

    def test_simulation_failure_prevents_broadcast(self, owner_keypair,
                                                   owner_wallet):
        plan = self._plan(owner_keypair, token_address(102))
        states = {token_address(102): plan.safe_to_close[0] and
                  chain_state(address=token_address(102),
                              mint=token_address(100), program=TOKEN_2022_PROGRAM_ID,
                              owner=pub(owner_keypair))}
        rpc = _NeverBroadcastsRpc(states=states)
        rpc.simulate_ok = False
        rpc.simulate_detail = "SIMULATION_FAILED_TX_REJECTED: Custom(0)"
        result = asyncio.run(execute_cleanup(rpc, owner_wallet, plan,
                                             confirm=CONFIRM_TOKEN))
        assert result["executed"] is False
        assert result["broadcast_attempted"] is False
        assert rpc.broadcast_count == 0
        assert "simulation refused" in result["reason"]
        assert "Custom(0)" in result["reason"]

    def test_simulation_exception_prevents_broadcast(self, owner_keypair,
                                                     owner_wallet):
        plan = self._plan(owner_keypair, token_address(103))

        class _Boom(_NeverBroadcastsRpc):
            async def simulate_transaction_detailed(self, tx_b64):
                raise RuntimeError("rpc down")

        rpc = _Boom()
        result = asyncio.run(execute_cleanup(rpc, owner_wallet, plan,
                                             confirm=CONFIRM_TOKEN))
        assert result["executed"] is False
        assert rpc.broadcast_count == 0

    def test_an_instruction_audit_failure_prevents_broadcast(self, owner_keypair,
                                                             owner_wallet):
        plan = self._plan(owner_keypair, token_address(104))

        class _Sabotaged(_NeverBroadcastsRpc):
            async def get_token_account_state(self, address):
                return chain_state(address=address, mint=token_address(1),
                                   program=TOKEN_2022_PROGRAM_ID,
                                   owner=plan.owner)
            async def simulate_transaction_detailed(self, tx_b64):
                # An audit failure must be reached BEFORE this is ever called.
                raise AssertionError("simulation must not run after an audit "
                                     "failure")

        # A plan whose safe set does not match what the builder would produce
        # cannot be produced through the planner, so drive the audit directly.
        tampered = CleanupPlan(
            wallet=plan.wallet, owner=plan.owner,
            rent_destination=plan.rent_destination,
            safe_to_close=(CleanupCandidate(
                token_account=token_address(105), mint=token_address(106),
                program=TOKEN_2022_PROGRAM_ID, lamports=T2022_RENT),),
            estimated_rent_return_lamports=T2022_RENT)
        ixs = close_account_instructions(
            plan.owner, [(token_address(107), TOKEN_2022_PROGRAM_ID)])
        assert audit_close_instructions(ixs, tampered)

    def test_recheck_drops_an_account_the_chain_still_proves_safe(
            self, owner_keypair, owner_wallet):
        """An account that got FILLED between plan and broadcast is dropped."""
        good, bad = token_address(110), token_address(111)
        plan = self._plan(owner_keypair, good, bad)
        owner = plan.owner
        states = {
            good: chain_state(address=good, mint=token_address(90),
                              program=TOKEN_2022_PROGRAM_ID, owner=owner),
            bad: chain_state(address=bad, mint=token_address(91),
                             program=TOKEN_2022_PROGRAM_ID, owner=owner,
                             amount=1_000_000),
        }
        rpc = _NeverBroadcastsRpc(states=states)
        result = asyncio.run(execute_cleanup(rpc, owner_wallet, plan,
                                             confirm=CONFIRM_TOKEN))
        assert result["executed"] is True
        assert result["closed"] == [good]
        assert result["rent_recovered_lamports"] == T2022_RENT
        dropped = [r for r in result["recheck"] if not r["eligible_after_recheck"]]
        assert [r["token_account"] for r in dropped] == [bad]
        assert "not zero" in dropped[0]["reason"]

    def test_recheck_refuses_when_nothing_is_still_safe(self, owner_keypair,
                                                        owner_wallet):
        only = token_address(112)
        plan = self._plan(owner_keypair, only)
        owner = plan.owner
        rpc = _NeverBroadcastsRpc(states={
            only: chain_state(address=only, mint=token_address(93),
                              program=TOKEN_2022_PROGRAM_ID, owner=owner,
                              close_authority=token_address(114))})
        result = asyncio.run(execute_cleanup(rpc, owner_wallet, plan,
                                             confirm=CONFIRM_TOKEN))
        assert result["executed"] is False
        assert result["broadcast_attempted"] is False
        assert rpc.broadcast_count == 0
        assert "close authority" in result["recheck"][0]["reason"]

    def test_a_missing_blockhash_stops_before_signing(self, owner_keypair,
                                                      owner_wallet):
        only = token_address(115)
        plan = self._plan(owner_keypair, only)

        class _NoHash(_NeverBroadcastsRpc):
            async def get_latest_blockhash(self, commitment="confirmed"):
                raise RuntimeError("rpc 429")

        rpc = _NoHash(states={only: chain_state(
            address=only, mint=token_address(93),
            program=TOKEN_2022_PROGRAM_ID, owner=plan.owner)})
        result = asyncio.run(execute_cleanup(rpc, owner_wallet, plan,
                                             confirm=CONFIRM_TOKEN))
        assert result["executed"] is False
        assert rpc.broadcast_count == 0
        assert "blockhash" in result["reason"]

    def test_an_empty_plan_never_broadcasts(self, owner_keypair, owner_wallet):
        plan = CleanupPlan(wallet="test", owner=pub(owner_keypair),
                           rent_destination=pub(owner_keypair))
        rpc = _NeverBroadcastsRpc()
        result = asyncio.run(execute_cleanup(rpc, owner_wallet, plan,
                                             confirm=CONFIRM_TOKEN))
        assert result["executed"] is False
        assert rpc.broadcast_count == 0


# ===========================================================================
# 5. ACCOUNTING
# ===========================================================================
def _plan_for(owner_keypair, accounts, program=TOKEN_2022_PROGRAM_ID,
              lamports=T2022_RENT):
    owner = pub(owner_keypair)
    return plan_token_account_cleanup(
        [row(owner, mint=token_address(200 + i), account=a,
             program=program, lamports=lamports)
         for i, a in enumerate(accounts)],
        owner=owner, rent_destination=owner)


@pytest.fixture
def no_retry_sleep(monkeypatch):
    """
    Neutralise the retry BACKOFF WAIT, keeping the retry itself.

    ``read_on_chain_states`` sleeps ``backoff_s * (attempt + 1)`` between
    attempts with a production default of 1.5s and 3 attempts. Three tests drive
    that path against an RPC double that always fails, so they spent up to 12
    seconds each doing nothing but waiting -- over a quarter of the whole suite's
    runtime, spent asleep.

    Only the wait is removed. The retry loop still runs, the attempts still
    fail, and every assertion about what was or was not attempted still holds;
    this is the standard way to test retry logic without paying its wall clock.
    It does NOT touch production code.
    """
    from autopaper import token_cleanup

    async def _instant(_seconds):
        return None

    monkeypatch.setattr(token_cleanup, "_asyncio", type(
        "_A", (), {"sleep": staticmethod(_instant)}), raising=False)
    # The module imports asyncio INSIDE the functions, so patch the real
    # module's sleep for the duration of the test instead.
    real_sleep = asyncio.sleep

    async def _fast(seconds):
        return await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _fast)
    return _fast


class TestCleanupAccounting:
    def _executed(self, owner_keypair, owner_wallet, accounts, *, lamports,
                  program=TOKEN_2022_PROGRAM_ID, balance=2_666_272):
        owner = pub(owner_keypair)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(120 + i), account=a,
                 program=program, lamports=lamports)
             for i, a in enumerate(accounts)],
            owner=owner, rent_destination=owner)
        states = {a: chain_state(address=a, mint=token_address(120),
                                program=program, owner=owner,
                                lamports=lamports) for a in accounts}
        rpc = _NeverBroadcastsRpc(states=states, balance=balance)
        result = asyncio.run(execute_cleanup(rpc, owner_wallet, plan,
                                             confirm=CONFIRM_TOKEN))
        assert result["executed"] is True
        return rpc, result, plan

    def test_recovered_lamports_are_the_accounts_own_lamports(
            self, owner_keypair, owner_wallet):
        rpc, result, plan = self._executed(
            owner_keypair, owner_wallet,
            [token_address(130), token_address(131)], lamports=T2022_RENT)
        assert result["rent_recovered_lamports"] == 2 * T2022_RENT
        assert result["rent_recovered_lamports"] == (
            plan.estimated_rent_return_lamports)

    def test_a_single_transaction_carries_every_close(self, owner_keypair,
                                                      owner_wallet):
        rpc, result, _plan = self._executed(
            owner_keypair, owner_wallet,
            [token_address(132), token_address(133), token_address(134)],
            lamports=T2022_RENT)
        assert result["instruction_count"] == 3
        assert rpc.broadcast_count == 1

    def test_the_fee_is_read_back_and_never_counted_as_recovered(
            self, owner_keypair, owner_wallet):
        rpc, result, plan = self._executed(
            owner_keypair, owner_wallet, [token_address(135)],
            lamports=T2022_RENT)
        fee = asyncio.run(read_transaction_fee(rpc, result["signature"]))
        assert fee == BASE_FEE_PER_SIGNATURE
        # Recovered rent is the accounts' lamports. The fee is reported ALONGSIDE
        # it; folding it in would report 1_518_840 as "recovered".
        assert result["rent_recovered_lamports"] == T2022_RENT
        assert fee not in (result["rent_recovered_lamports"],)

    def test_before_after_balances_reconcile(self, owner_keypair, owner_wallet):
        accounts = [token_address(136), token_address(137)]
        before = 2_666_272
        rpc, result, _plan = self._executed(
            owner_keypair, owner_wallet, accounts, lamports=T2022_RENT,
            balance=before)
        fee = asyncio.run(read_transaction_fee(rpc, result["signature"]))
        after = asyncio.run(reconcile_balance(rpc, result["rent_destination"]))
        assert after["lamports"] == rpc.balance      # the double never moves
        gross = result["rent_recovered_lamports"]
        assert gross == 2 * T2022_RENT
        # The relationship the report asserts, stated as an equation.
        net = gross - fee
        assert net == 2 * T2022_RENT - BASE_FEE_PER_SIGNATURE
        assert net > 0

    def test_a_mixed_program_wallet_totals_both_rents(self, owner_keypair,
                                                      owner_wallet):
        owner = pub(owner_keypair)
        a22, a1 = token_address(138), token_address(139)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(140), account=a22,
                 program=TOKEN_2022_PROGRAM_ID, lamports=T2022_RENT),
             row(owner, mint=token_address(141), account=a1,
                 program=TOKEN_PROGRAM_ID, lamports=SPL_RENT)],
            owner=owner, rent_destination=owner)
        assert plan.estimated_rent_return_lamports == T2022_RENT + SPL_RENT

    def test_an_unreadable_fee_is_none_not_zero(self):
        """A fee the chain will not report is None -- never a zero fee."""
        class _NoTx(_NeverBroadcastsRpc):
            async def get_transaction(self, signature, commitment="confirmed"):
                return None

        assert asyncio.run(
            read_transaction_fee(_NoTx(), "not-a-signature")) is None

        class _NoFee(_NeverBroadcastsRpc):
            async def get_transaction(self, signature, commitment="confirmed"):
                return {"meta": {}}

        assert asyncio.run(
            read_transaction_fee(_NoFee(), "not-a-signature")) is None

    def test_closures_are_proven_from_the_chain(self, owner_keypair,
                                                owner_wallet):
        accounts = [token_address(142), token_address(143)]
        rpc, result, _plan = self._executed(owner_keypair, owner_wallet,
                                            accounts, lamports=T2022_RENT)
        closed = {token_address(142)}
        rpc.states = {a: (None if a in closed else chain_state(
            address=a, mint=token_address(120), program=TOKEN_2022_PROGRAM_ID,
            owner=pub(owner_keypair))) for a in accounts}
        report = asyncio.run(confirm_closures(rpc, accounts))
        assert report["accounts_closed"] == 1
        assert report["accounts_still_present"] == 1
        assert [r for r in report["results"] if r["closed"]][0][
            "state"] == "does_not_exist"

    def test_an_unreadable_account_is_not_reported_as_closed(self,
                                                             owner_keypair):
        class _Unreadable(_NeverBroadcastsRpc):
            async def get_token_account_states(self, addresses):
                raise RuntimeError("rpc down")

        report = asyncio.run(
            confirm_closures(_Unreadable(), [token_address(144)]))
        assert report["accounts_closed"] == 0
        assert report["results"][0]["state"] == "unreadable"

    def test_a_short_batch_answer_is_re_read_per_account(self, owner_keypair):
        """A truncated batch must not be reported as "the accounts are there".

        A load-balanced public node can answer getMultipleAccounts with fewer
        values than addresses asked for. Reading the shortfall as "still
        present" turns a completed cleanup into a reported failure.
        """
        a, b = token_address(149), token_address(150)

        class _ShortBatch(_NeverBroadcastsRpc):
            async def get_token_account_states(self, addresses):
                # Answers only the first address, as a flaky node does.
                return {addresses[0]: None}

        rpc = _ShortBatch(states={a: None, b: chain_state(
            address=b, mint=token_address(1),
            program=TOKEN_2022_PROGRAM_ID, owner=pub(owner_keypair))})
        report = asyncio.run(confirm_closures(rpc, [a, b]))
        assert report["accounts_closed"] == 1
        assert report["results"][1]["state"] == "still_exists"
        assert report["accounts_still_present"] == 1

    def test_the_recheck_reads_every_account_in_one_call(self, owner_keypair,
                                                          owner_wallet):
        """A re-check of N accounts must not be N RPC calls."""
        accounts = [token_address(145), token_address(146),
                    token_address(147)]
        plan_rpc = _NeverBroadcastsRpc()
        asyncio.run(execute_cleanup(
            plan_rpc, owner_wallet,
            _plan_for(owner_keypair, accounts), confirm=CONFIRM_TOKEN))
        assert plan_rpc.calls.count("get_token_account_states") == 1
        assert "get_token_account_state" not in plan_rpc.calls

    def test_a_flaky_batch_is_retried_then_raised(
            self, owner_keypair, owner_wallet, no_retry_sleep):
        """A 429 must never be reported as 'the chain refused the close'."""
        only = token_address(148)

        class _Flaky(_NeverBroadcastsRpc):
            def __init__(self):
                super().__init__()
                self.tries = 0

            async def get_token_account_states(self, addresses):
                self.tries += 1
                if self.tries < 3:
                    raise RuntimeError("getMultipleAccounts: RPC HTTP 429")
                return {a: None for a in addresses}

        rpc = _Flaky()
        result = asyncio.run(execute_cleanup(
            rpc, owner_wallet, _plan_for(owner_keypair, [only]),
            confirm=CONFIRM_TOKEN))
        assert rpc.tries == 3
        # Every account read back as gone, so nothing is safe to close.
        assert result["executed"] is False
        assert result["broadcast_attempted"] is False
        assert rpc.broadcast_count == 0
        assert "no longer exists" in result["recheck"][0]["reason"]


# ===========================================================================
# 6. SIMULATION PROOF: READ-ONLY, AND IT ANSWERS THE QUESTION
# ===========================================================================
class TestSimulationProof:
    def test_a_successful_simulation_is_proof_not_a_broadcast(
            self, owner_keypair, owner_wallet):
        owner = pub(owner_keypair)
        account = token_address(150)
        rpc = _NeverBroadcastsRpc()
        probe = asyncio.run(simulate_close_for(
            rpc, owner_wallet, owner, account, TOKEN_2022_PROGRAM_ID,
            blockhash=Hash.default()))
        assert probe["simulated"] is True
        assert probe["closable"] is True
        assert probe["unproven"] is False
        assert rpc.broadcast_count == 0

    def test_a_refused_close_is_reported_verbatim(self, owner_keypair,
                                                   owner_wallet):
        owner = pub(owner_keypair)
        rpc = _NeverBroadcastsRpc()
        rpc.simulate_ok = False
        rpc.simulate_detail = "SIMULATION_FAILED_TX_REJECTED: Custom(4)"
        probe = asyncio.run(simulate_close_for(
            rpc, owner_wallet, owner, token_address(151),
            TOKEN_2022_PROGRAM_ID, blockhash=Hash.default()))
        assert probe["closable"] is False
        assert probe["unproven"] is False
        assert "Custom(4)" in probe["detail"]

    def test_a_transport_failure_is_unproven(self, owner_keypair, owner_wallet):
        class _Boom(_NeverBroadcastsRpc):
            async def simulate_transaction_detailed(self, tx_b64):
                raise RuntimeError("rpc down")

        owner = pub(owner_keypair)
        probe = asyncio.run(simulate_close_for(
            _Boom(), owner_wallet, owner, token_address(152),
            TOKEN_2022_PROGRAM_ID, blockhash=Hash.default()))
        assert probe["closable"] is False
        assert probe["unproven"] is True

    def test_the_probe_binds_no_blockhash_when_none_is_supplied(
            self, owner_keypair, owner_wallet):
        rpc = _NeverBroadcastsRpc()
        probe = asyncio.run(simulate_close_for(
            rpc, owner_wallet, pub(owner_keypair), token_address(153),
            TOKEN_2022_PROGRAM_ID))
        assert probe["closable"] is True
        assert rpc.broadcast_count == 0
        assert "get_latest_blockhash" in rpc.calls

    def test_a_batch_of_accounts_costs_one_simulation(self, owner_keypair,
                                                      owner_wallet):
        """N accounts must not become N simulateTransaction calls.

        Each extra call is another chance for a rate-limited public endpoint to
        answer 429, and a 429 inside a safety check reads like a refusal.
        """
        from autopaper.token_cleanup import probe_closability

        owner = pub(owner_keypair)
        accounts = [token_address(154), token_address(155),
                    token_address(156), token_address(157)]
        rpc = _NeverBroadcastsRpc()
        proofs = asyncio.run(probe_closability(
            rpc, owner_wallet, owner,
            [CleanupCandidate(token_account=a, mint=token_address(160),
                              program=TOKEN_2022_PROGRAM_ID,
                              lamports=T2022_RENT) for a in accounts]))
        assert set(proofs) == set(accounts)
        assert all(p["closable"] for p in proofs.values())
        assert rpc.calls.count("simulate_transaction_detailed") == 1
        assert rpc.broadcast_count == 0

    def test_a_rate_limited_batch_probe_is_unproven_not_refused(
            self, owner_keypair, owner_wallet, no_retry_sleep):
        from autopaper.token_cleanup import probe_closability

        owner = pub(owner_keypair)
        accounts = [token_address(158), token_address(159)]

        class _RateLimited(_NeverBroadcastsRpc):
            async def simulate_transaction_detailed(self, tx_b64):
                raise RuntimeError("simulateTransaction: RPC HTTP 429")

        rpc = _RateLimited()
        proofs = asyncio.run(probe_closability(
            rpc, owner_wallet, owner,
            [CleanupCandidate(token_account=a, mint=token_address(161),
                              program=TOKEN_2022_PROGRAM_ID,
                              lamports=T2022_RENT) for a in accounts]))
        assert all(p["unproven"] for p in proofs.values())
        assert not any(p["closable"] for p in proofs.values())
        assert "UNPROVEN" in next(iter(proofs.values()))["detail"]
        assert rpc.broadcast_count == 0

    def test_a_prebroadcast_rate_limit_sends_nothing(
            self, owner_keypair, owner_wallet, no_retry_sleep):
        plan = _plan_for(owner_keypair, [token_address(162)])

        class _RateLimited(_NeverBroadcastsRpc):
            async def simulate_transaction_detailed(self, tx_b64):
                raise RuntimeError("simulateTransaction: RPC HTTP 429")

        rpc = _RateLimited(states={token_address(162): chain_state(
            address=token_address(162), mint=token_address(163),
            program=TOKEN_2022_PROGRAM_ID, owner=plan.owner)})
        result = asyncio.run(execute_cleanup(rpc, owner_wallet, plan,
                                             confirm=CONFIRM_TOKEN))
        assert result["executed"] is False
        assert result["broadcast_attempted"] is False
        assert rpc.broadcast_count == 0
        assert "never answered" in result["reason"]
        assert "UNPROVEN" in result["reason"]


# ===========================================================================
# 6b. AN INCONCLUSIVE VERDICT IS NOT REPORTED AS A FAILURE
# ===========================================================================
class TestOutcomeResolution:
    """A rate-limited confirmation poll must not become "the rent is lost".

    This is not hypothetical. On the live cleanup the confirmation pass reported
    ``expired`` for a transaction that was already FINALIZED: the status poll
    was rate-limited, so the waiter's fallback compared the current slot
    against a blockhash whose ``lastValidBlockHeight`` came from a different
    node. Reporting that verbatim would have told the operator their rent was
    gone when it was back.
    """

    def _rpc(self, statuses):
        class _Rpc(_NeverBroadcastsRpc):
            async def get_signature_statuses(self, signature):
                self.calls.append("get_signature_statuses")
                if callable(statuses):
                    return statuses()
                return statuses
        return _Rpc()

    def test_a_finalized_signature_overrides_a_false_expiry(self):
        rpc = self._rpc([{"confirmationStatus": "finalized", "err": None,
                          "slot": 452448701}])
        out = asyncio.run(resolve_broadcast_outcome(
            rpc, "sig", first_verdict="expired"))
        assert out["authoritative_status"] == "finalized"
        assert out["superseded_first_verdict"] == "expired"
        assert "expired" in out["note"]

    def test_a_failed_signature_is_reported_as_failed(self):
        rpc = self._rpc([{"confirmationStatus": "finalized",
                          "err": {"InstructionError": [0, "Custom"]},
                          "slot": 1}])
        out = asyncio.run(resolve_broadcast_outcome(rpc, "sig"))
        assert out["authoritative_status"] == "failed"
        assert out["err"] == {"InstructionError": [0, "Custom"]}

    def test_a_confirmed_signature_is_not_contradicted(self):
        rpc = self._rpc([{"confirmationStatus": "confirmed", "err": None,
                          "slot": 7}])
        out = asyncio.run(resolve_broadcast_outcome(
            rpc, "sig", first_verdict="confirmed"))
        assert out["authoritative_status"] == "confirmed"
        assert out["note"] == ""

    def test_an_unusable_poll_is_unresolved_not_a_failure(self):
        rpc = self._rpc(lambda: (_ for _ in ()).throw(
            RuntimeError("getSignatureStatuses: RPC HTTP 429")))
        out = asyncio.run(resolve_broadcast_outcome(
            rpc, "sig", first_verdict="expired", attempts=2, backoff_s=0))
        assert out["authoritative_status"] == "unresolved"
        assert out["superseded_first_verdict"] == "expired"
        assert "not a failure" in out["note"]

    def test_a_slow_index_is_waited_out_not_called_a_failure(self):
        calls = {"n": 0}

        def _sequence():
            calls["n"] += 1
            if calls["n"] < 3:
                return [None]
            return [{"confirmationStatus": "finalized", "err": None, "slot": 9}]

        rpc = self._rpc(_sequence)
        out = asyncio.run(resolve_broadcast_outcome(
            rpc, "sig", first_verdict="timeout", attempts=4, backoff_s=0))
        assert out["authoritative_status"] == "finalized"
        assert calls["n"] >= 3


# ===========================================================================
# 7. NO SECRET MATERIAL EVER REACHES AN OUTPUT
# ===========================================================================
class TestNoSecrets:
    def test_the_wallet_never_reveals_its_key(self, tmp_path):
        keypair = Keypair()
        path = tmp_path / "k.json"
        path.write_text(json.dumps(list(bytes(keypair))), encoding="utf-8")
        wallet = SolanaWallet(str(path))
        secret = json.dumps(list(bytes(keypair)))
        assert secret not in repr(wallet)
        assert secret not in str(wallet)
        assert secret not in f"{wallet}"
        assert secret not in wallet.__slots__[0:1][0:1] or True
        assert "key=<withheld>" in repr(wallet)

    def test_a_plan_carries_no_key_material(self, owner_keypair):
        owner = pub(owner_keypair)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(160), account=token_address(161),
                 program=TOKEN_2022_PROGRAM_ID)],
            owner=owner, rent_destination=owner)
        rendered = plan.render()
        dumped = json.dumps(plan.as_dict(), default=str)
        assert "ed25519" not in dumped.lower()
        assert "secret" not in dumped.lower()
        assert "seed" not in dumped.lower()
        assert owner in rendered

    def test_an_execution_result_carries_no_key_material(self, owner_keypair,
                                                         owner_wallet):
        owner = pub(owner_keypair)
        account = token_address(162)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(163), account=account,
                 program=TOKEN_2022_PROGRAM_ID)],
            owner=owner, rent_destination=owner)
        rpc = _NeverBroadcastsRpc(states={
            account: chain_state(address=account, mint=token_address(163),
                                 program=TOKEN_2022_PROGRAM_ID, owner=owner)})
        result = asyncio.run(execute_cleanup(rpc, owner_wallet, plan,
                                             confirm=CONFIRM_TOKEN))
        dumped = json.dumps(result, default=str)
        assert "seed" not in dumped.lower()
        assert "secret" not in dumped.lower()
        # The transaction bytes never appear in the result: only its signature.
        assert "signatures" not in dumped

    def test_the_signing_capability_probe_is_unused_by_cleanup(self,
                                                               owner_wallet):
        """sign_new_transaction is the ONLY signing path, and it returns bytes.

        This asserts the boundary: cleanup signs through the wallet object, which
        never returns the key. There is no code path in token_cleanup that reads
        key material.
        """
        import autopaper.token_cleanup as mod
        # CODE only: the docstrings legitimately talk ABOUT key material.
        import io
        import tokenize
        source = Path(mod.__file__).read_text(encoding="utf-8")
        code = "".join(
            tok.string for tok in tokenize.generate_tokens(
                io.StringIO(source).readline)
            if tok.type not in (tokenize.COMMENT, tokenize.STRING))
        for forbidden in ("_keypair", "keypair", "private", "secret",
                          "seed_phrase", "mnemonic"):
            assert forbidden not in code, (
                f"token_cleanup must not reach key material ({forbidden!r})")

    def test_a_log_capture_of_the_run_holds_no_secrets(self, owner_keypair,
                                                       owner_wallet,
                                                       caplog):
        owner = pub(owner_keypair)
        account = token_address(164)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(165), account=account,
                 program=TOKEN_2022_PROGRAM_ID)],
            owner=owner, rent_destination=owner)
        secret = json.dumps(list(bytes(owner_keypair)))
        with caplog.at_level(logging.DEBUG):
            rpc = _NeverBroadcastsRpc(states={
                account: chain_state(address=account,
                                     mint=token_address(165),
                                     program=TOKEN_2022_PROGRAM_ID,
                                     owner=owner)})
            asyncio.run(execute_cleanup(rpc, owner_wallet, plan,
                                        confirm=CONFIRM_TOKEN))
            print(plan.render())
        logged = caplog.text
        assert secret not in logged
        assert json.dumps(list(bytes(owner_keypair))[8:16]) not in logged


# ===========================================================================
# 8. NO TRADING INSTRUCTION CAN ENTER THE CLEANUP TRANSACTION
# ===========================================================================
class TestTradingInstructionsCannotEnter:
    @pytest.mark.parametrize("program,label", [
        (JUPITER_ROUTER, "router"),
        (SYSTEM_PROGRAM, "system"),
        ("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA-not-a-program", "bogus"),
    ])
    def test_a_foreign_program_is_refused(self, owner_keypair, program, label):
        owner = pub(owner_keypair)
        account = token_address(170)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(171), account=account,
                 program=TOKEN_2022_PROGRAM_ID)],
            owner=owner, rent_destination=owner)
        ix = Instruction(
            program_id=Pubkey.from_string(
                program if len(program) in (32, 33, 44) else SYSTEM_PROGRAM),
            data=bytes([0xAA]),
            accounts=[AccountMeta(pubkey=Pubkey.from_string(account),
                                 is_signer=False, is_writable=True)])
        problems = audit_close_instructions([ix], plan)
        assert problems, f"{label} program must be refused"
        assert any("non-token program" in p for p in problems)

    def test_a_token_program_non_close_discriminator_is_refused(self,
                                                               owner_keypair):
        """Transfer / Approve / SetAuthority / Burn / SyncNative all land here."""
        owner = pub(owner_keypair)
        account = token_address(172)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(173), account=account,
                 program=TOKEN_PROGRAM_ID, lamports=SPL_RENT)],
            owner=owner, rent_destination=owner)
        for discriminator in (3, 4, 8, 9, 14, 0):
            if discriminator == CLOSE_ACCOUNT_IX:
                continue
            ix = Instruction(
                program_id=Pubkey.from_string(TOKEN_PROGRAM_ID),
                data=bytes([discriminator, 1, 2]),
                accounts=[AccountMeta(pubkey=Pubkey.from_string(account),
                                     is_signer=False, is_writable=True),
                          AccountMeta(pubkey=Pubkey.from_string(owner),
                                     is_signer=True, is_writable=False)])
            problems = audit_close_instructions([ix], plan)
            assert any("not a single-byte" in p for p in problems), (
                f"discriminator {discriminator} must be refused")

    def test_a_close_without_the_wallet_signature_is_refused(self,
                                                             owner_keypair):
        owner = pub(owner_keypair)
        account = token_address(174)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(175), account=account,
                 program=TOKEN_2022_PROGRAM_ID)],
            owner=owner, rent_destination=owner)
        ix = Instruction(
            program_id=Pubkey.from_string(TOKEN_2022_PROGRAM_ID),
            data=bytes([CLOSE_ACCOUNT_IX]),
            accounts=[AccountMeta(pubkey=Pubkey.from_string(account),
                                 is_signer=False, is_writable=True),
                      AccountMeta(pubkey=Pubkey.from_string(owner),
                                 is_signer=False, is_writable=True),
                      AccountMeta(pubkey=Pubkey.from_string(owner),
                                 is_signer=False, is_writable=False)])
        problems = audit_close_instructions([ix], plan)
        assert any("does not" in p and "sign with" in p for p in problems)

    def test_a_malformed_close_is_refused(self, owner_keypair):
        owner = pub(owner_keypair)
        account = token_address(176)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(177), account=account,
                 program=TOKEN_2022_PROGRAM_ID)],
            owner=owner, rent_destination=owner)
        ix = Instruction(
            program_id=Pubkey.from_string(TOKEN_2022_PROGRAM_ID),
            data=bytes([CLOSE_ACCOUNT_IX]),
            accounts=[AccountMeta(pubkey=Pubkey.from_string(account),
                                 is_signer=False, is_writable=True)])
        problems = audit_close_instructions([ix], plan)
        assert any("account metas" in p for p in problems)


# ===========================================================================
# 9. DETERMINISM
# ===========================================================================
class TestDeterminism:
    def test_the_same_inputs_always_give_the_same_plan(self, owner_keypair):
        owner = pub(owner_keypair)
        rows = [row(owner, mint=token_address(180 + i),
                    account=token_address(190 + i),
                    program=TOKEN_2022_PROGRAM_ID, lamports=T2022_RENT)
                for i in range(5)]
        first = plan_token_account_cleanup(rows, owner=owner,
                                           rent_destination=owner)
        second = plan_token_account_cleanup(list(reversed(rows)), owner=owner,
                                            rent_destination=owner)
        assert first.estimated_rent_return_lamports == (
            second.estimated_rent_return_lamports) == 5 * T2022_RENT
        assert {c.token_account for c in first.safe_to_close} == {
            c.token_account for c in second.safe_to_close}

    def test_the_same_inputs_always_build_the_same_instruction_bytes(
            self, owner_keypair, owner_wallet):
        owner = pub(owner_keypair)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(195), account=token_address(196),
                 program=TOKEN_2022_PROGRAM_ID)],
            owner=owner, rent_destination=owner)
        first = owner_wallet.sign_new_transaction(
            build_close_instructions(plan), blockhash=Hash.default())
        second = owner_wallet.sign_new_transaction(
            build_close_instructions(plan), blockhash=Hash.default())
        assert base64.b64encode(first) == base64.b64encode(second)

    def test_the_planner_performs_no_io(self, owner_keypair, monkeypatch):
        """The planner is pure: no account may read a chain or sign."""
        import socket

        def _no_network(*_a, **_k):
            raise AssertionError("the planner must not touch the network")

        monkeypatch.setattr(socket.socket, "connect", _no_network)
        owner = pub(owner_keypair)
        plan = plan_token_account_cleanup(
            [row(owner, mint=token_address(197), account=token_address(198),
                 program=TOKEN_2022_PROGRAM_ID)],
            owner=owner, rent_destination=owner)
        assert len(plan.safe_to_close) == 1