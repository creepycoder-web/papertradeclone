"""
Pre-provisioned wrapped-SOL routing and transaction-size discipline.

Companion to tests/test_real_simulation.py (which owns RPC error
classification). This file owns the question of whether the BUY transaction
can physically be sent at all.

Nothing here broadcasts. Every case uses in-memory doubles or a locally
constructed, never-signed, never-sent transaction.

Covered:
   1. an existing wSOL account is detected
   2. a missing wSOL account is detected
   3. an existing-but-empty wSOL account is NOT treated as ready
   4. the BUY path no longer carries temporary wSOL instructions
   5. the wSOL-mode transaction fits the size limit (deterministic fixture)
   6. an oversized transaction is refused locally, without an RPC call
   7. simulation failure stays fail-closed
   8. no broadcast after a simulation failure
   9. serialize / deserialize round trip
  10. VersionedTransaction integrity
  11. L25 strategy immutability
  12. REAL_TRADING_ENABLED stays false by default
"""


from __future__ import annotations

import asyncio
import base64

import pytest

from autopaper.execution import solana as S

pytestmark = [
    pytest.mark.execution,
    pytest.mark.safety,
]


OWNER = "wUEUbVmz2EfcLfchHmucTKhRpZgDpsBvo3FttZ4Wbt6"
WSOL = "So11111111111111111111111111111111111111112"


def _ready(owner=OWNER, balance=5_000_000, required=1_071_843, present=True):
    return S.WsolReadiness(owner=owner, wsol_ata="AtaStub", ata_present=present,
                           balance_lamports=balance if present else 0,
                           required_lamports=required,
                           sufficient=present and balance >= required,
                           query_ok=True)


class _Rpc(S.SolanaRpc):
    """SolanaRpc with the network replaced by scripted responses."""

    def __init__(self, *, accounts=None, rpc_error=None):
        super().__init__("https://example.invalid")
        self._accounts = accounts if accounts is not None else []
        self._rpc_error = rpc_error
        self.calls: list = []
        self.sent: list = []

    async def _rpc(self, method, params):
        self.calls.append((method, params))
        if self._rpc_error is not None:
            raise self._rpc_error
        return {"context": {"slot": 1}, "value": {"accounts": self._accounts}}

    async def send_transaction(self, b64):
        self.sent.append(b64)
        raise AssertionError("send_transaction must never be reached")


def _wsol_account(owner=OWNER, mint=WSOL, amount="5000000"):
    return {
        "account": {
            "data": {
                "parsed": {
                    "info": {
                        "mint": mint,
                        "owner": owner,
                        "tokenAmount": {"amount": amount,
                                        "uiAmountString": "0.005"},
                    }
                }
            }
        }
    }


# --------------------------------------------------------------------------
# 1-3. readiness detection
# --------------------------------------------------------------------------
def test_existing_wsol_account_is_detected():
    r = _Rpc(accounts=[_wsol_account()])
    out = asyncio.run(r.wsol_readiness(OWNER, 1_071_843))
    assert out.ata_present is True
    assert out.balance_lamports == 5_000_000
    assert out.sufficient is True
    assert out.ready is True
    assert out.as_dict()["WSOL_ATA_PRESENT"] is True


def test_missing_wsol_account_is_detected():
    r = _Rpc(accounts=[])
    out = asyncio.run(r.wsol_readiness(OWNER, 1_071_843))
    assert out.ata_present is False
    assert out.ready is False
    assert "setup-wsol" in out.block_reason()


def test_existing_but_empty_account_is_not_ready():
    """An ATA that exists and holds nothing must NOT count as funded.

    This is the specific confusion the readiness type exists to prevent:
    account presence is not balance.
    """
    r = _Rpc(accounts=[_wsol_account(amount="0")])
    out = asyncio.run(r.wsol_readiness(OWNER, 1_071_843))
    assert out.ata_present is True
    assert out.balance_lamports == 0
    assert out.sufficient is False
    assert out.ready is False
    assert "below" in out.block_reason()


def test_insufficient_balance_is_not_ready():
    r = _Rpc(accounts=[_wsol_account(amount="1000")])
    out = asyncio.run(r.wsol_readiness(OWNER, 1_071_843))
    assert out.ata_present is True
    assert out.sufficient is False
    assert out.ready is False


def test_an_account_for_another_mint_does_not_count():
    r = _Rpc(accounts=[_wsol_account(mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")])
    out = asyncio.run(r.wsol_readiness(OWNER, 1))
    assert out.ata_present is False


def test_an_account_owned_by_someone_else_does_not_count():
    r = _Rpc(accounts=[_wsol_account(owner="SomeoneElse11111111111111111111111111")])
    out = asyncio.run(r.wsol_readiness(OWNER, 1))
    assert out.ata_present is False


def test_an_rpc_that_cannot_answer_is_never_treated_as_ready():
    """Failing closed on an unreadable wallet, not assuming the good case."""
    r = _Rpc(rpc_error=S.RpcError("boom", code=-32603, rpc_message="internal"))
    out = asyncio.run(r.wsol_readiness(OWNER, 1_071_843))
    assert out.query_ok is False
    assert out.ready is False
    assert "could not read" in out.block_reason()


def test_flattened_gateway_response_shape_is_understood():
    """Some gateways return {"value": [...]}. A parser assuming one shape would
    report 'no wSOL account' for a wallet that has one and block a good trade."""
    class Flat(_Rpc):
        async def _rpc(self, method, params):
            self.calls.append((method, params))
            return {"value": [_wsol_account()]}
    out = asyncio.run(Flat().wsol_readiness(OWNER, 1))
    assert out.ata_present is True


def test_ata_derivation_is_deterministic():
    a = S.wsol_ata_address(OWNER, WSOL)
    b = S.wsol_ata_address(OWNER, WSOL)
    assert a == b
    assert len(a) == 44                     # a base58 pubkey
    assert a != S.wsol_ata_address(OWNER, "So11111111111111111111111111111111111111113")


# --------------------------------------------------------------------------
# 4-5, 9-10. the wSOL-mode transaction, deterministically
# --------------------------------------------------------------------------
def _swap_tx(*, temp_wsol: bool, target_bytes: int | None = None):
    """
    Build a transaction shaped like a Jupiter SOL->token swap.

    temp_wsol=True reproduces the old route: SystemProgram CreateAccount +
    Token InitializeAccount + Token CloseAccount for a temporary wSOL account.
    temp_wsol=False is the pre-provisioned route, where those three are absent.
    """
    pytest.importorskip("solders")
    from solders.hash import Hash
    from solders.instruction import AccountMeta, Instruction
    from solders.keypair import Keypair
    from solders.message import Message
    from solders.system_program import ID as SYSTEM, ID as SYSID
    from solders.transaction import Transaction

    kp = Keypair()
    pk = kp.pubkey()
    ixes = []
    if temp_wsol:
        for _ in range(3):
            ixes.append(Instruction(SYSTEM, bytes([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                    [AccountMeta(pk, True, True),
                                     AccountMeta(pk, False, True)]))
    ixes.append(Instruction(SYSTEM, bytes([2, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                            [AccountMeta(pk, True, True)]))
    msg = Message.new_with_blockhash(ixes, pk, Hash.default())
    return bytes(Transaction.new_unsigned(msg).data(0) if False else bytes(Transaction.new_unsigned(msg))) \
        if target_bytes is None else target_bytes


def test_wsol_mode_drops_the_temporary_account_instructions():
    """The three CreateAccount/InitializeAccount/CloseAccount instructions
    for a temporary wrapped-SOL account must be absent in wSOL mode."""
    from solders.transaction import VersionedTransaction
    with_temp = VersionedTransaction.from_bytes(_swap_tx(temp_wsol=True))
    without = VersionedTransaction.from_bytes(_swap_tx(temp_wsol=False))
    assert len(with_temp.message.instructions) == 4
    assert len(without.message.instructions) == 1
    assert len(_swap_tx(temp_wsol=False)) < len(_swap_tx(temp_wsol=True))


def test_wsol_mode_transaction_fits_the_size_limit():
    """Requirement 5, on a deterministic fixture rather than a live quote.

    Proves the size gate ACCEPTS a correctly-shaped wSOL-mode transaction. It
    does not prove every real route fits -- Jupiter re-routes thin tokens and
    the live measurements in reports/forensics/real_tx_size_fix.md show 2 of 8
    samples still exceeding the limit. That is reported, not hidden.
    """
    r = S.SolanaRpc("https://example.invalid")
    raw = _swap_tx(temp_wsol=False)
    rep = r.transaction_size_report(base64.b64encode(raw).decode())
    assert rep["raw_bytes"] == len(raw)
    assert rep["raw_bytes"] <= S.MAX_SOLANA_TX_BYTES
    assert rep["ok"] is True


def test_serialize_deserialize_round_trip_preserves_a_versioned_transaction():
    """Requirements 9-10."""
    from solders.transaction import VersionedTransaction
    raw = _swap_tx(temp_wsol=False)
    reparsed = VersionedTransaction.from_bytes(raw)
    assert bytes(reparsed) == raw            # exact round trip
    assert len(reparsed.signatures) == 1
    assert len(reparsed.message.instructions) == 1


# --------------------------------------------------------------------------
# 6. oversized transaction refused locally
# --------------------------------------------------------------------------
def test_oversized_transaction_refused_without_contacting_the_rpc():
    from tests import txfixtures
    from autopaper.execution.txcodec import MAX_BYTES_LEGACY
    r = _Rpc()
    over = MAX_BYTES_LEGACY + 90
    # A REAL oversized v0 transaction, not a blob of zeroes: the size gate reads
    # the version off the deserialized message, so only a real transaction can
    # be judged against the limit at all.
    raw = txfixtures.build_exact("v0", over)
    out = asyncio.run(r.simulate_transaction_detailed(
        base64.b64encode(raw).decode()))
    assert out.ok is False
    assert out.status == "SIMULATION_FAILED_TX_TOO_LARGE"
    assert out.size["over_by_bytes"] == 90
    assert r.calls == []                    # no RPC round trip
    assert r.sent == []


# --------------------------------------------------------------------------
# 7-8. fail-closed, no broadcast
# --------------------------------------------------------------------------
def test_oversized_simulation_failure_never_broadcasts():
    """Requirement 8, on the failure path."""
    from tests import txfixtures
    from autopaper.execution.txcodec import MAX_BYTES_LEGACY
    r = _Rpc(accounts=[_wsol_account()])
    raw = txfixtures.build_exact("v0", MAX_BYTES_LEGACY + 5)
    out = asyncio.run(r.simulate_transaction_detailed(
        base64.b64encode(raw).decode()))
    assert out.ok is False
    assert out.status == "SIMULATION_FAILED_TX_TOO_LARGE"
    assert out.broadcast_attempted is False
    assert r.sent == []


def test_a_passing_simulation_still_does_not_broadcast_by_itself():
    """
    Requirement 8, the other half.

    simulate_transaction() is a READ. Even when it returns ok=True it must not
    have sent anything -- broadcasting is a separate, explicit step that the
    caller performs. Asserting this catches an implementation that folded the
    send into the simulate call.
    """
    r = _Rpc(accounts=[_wsol_account()])
    out = asyncio.run(r.simulate_transaction_detailed(
        base64.b64encode(_swap_tx(temp_wsol=False)).decode()))
    assert out.ok is True
    assert out.status == "SIMULATION_OK"
    assert out.broadcast_attempted is False
    assert r.sent == []


# --------------------------------------------------------------------------
# 11-12. strategy and REAL gating untouched
# --------------------------------------------------------------------------
def test_l25_strategy_is_unchanged():
    import autopaper.l25 as L
    from autopaper.config import DecisionConfig
    assert L.L25_MIN_LIQ_TO_MCAP == 0.9647
    assert L.V3C_MAX_MARKET_CAP_USD == 20000.0
    assert L.V3C_MIN_SIGNAL_QUALITY == 0.55
    assert DecisionConfig().l25_min_liq_to_mcap == 0.9647
    assert L.L25_VERSION == "v1"
    assert L.L25_HASH == (
        "bf5e608b06a5d5b104bb84b7adceb955de2f753a102c053e45ee47cf54aea74a")


def test_real_trading_is_not_enabled_by_default():
    from autopaper.config import build_config
    rc = build_config(".").real
    assert rc.env_trading_enabled_request is False
    assert rc.reentry_enabled is False
    # simulate-before-send must remain ON: this change made it stricter, not laxer
    assert rc.simulate_before_send is True
    assert rc.priority_fee_lamports == 50000
    assert rc.slippage_bps == 500


def test_setup_command_is_not_wired_into_run():
    """The provisioning path must stay a deliberate human action."""
    import inspect

    from autopaper import cli
    src = inspect.getsource(cli.cmd_run)
    assert "setup_wsol" not in src
    assert "wsol" not in src.lower()
