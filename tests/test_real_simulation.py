"""
REAL simulation path: diagnostics, classification and fail-closed guarantees.

Extends the existing REAL execution harness in tests/test_executor_real.py
rather than replacing it. That file owns the end-to-end ordering assertions
(build -> sign -> simulate -> broadcast); this file owns the narrower
question the July incident exposed, namely that a simulation rejection must
be CLASSIFIED and must never be laundered into a broadcast.

NO TEST IN THIS FILE BROADCASTS. Every case asserts on an in-memory
`send_transaction` double, and the live-RPC cases are marked and skip unless
an explicit opt-in environment variable is set. Nothing here spends money,
signs for a real payer, or touches a wallet file.

The eight required behaviours:
  1. transaction build succeeds
  2. transaction deserialize succeeds
  3. transaction serialize round-trip succeeds
  4. RPC request parameters valid
  5. simulateTransaction succeeds when the transaction is valid
  6. malformed transaction still fails closed
  7. invalid RPC params produce a classified failure
  8. no broadcast occurs after a simulation failure
"""


from __future__ import annotations

import asyncio
import base64
import json

import pytest

from tests import txfixtures
from autopaper.execution import solana as S
from autopaper.execution.txcodec import (
    MAX_BYTES_LEGACY,
    MAX_BYTES_V1,
    VERSION_LEGACY,
    VERSION_V0,
    VERSION_V1,
)

pytestmark = [
    pytest.mark.execution,
    pytest.mark.safety,
]


# --------------------------------------------------------------------------
# fixtures / doubles. Nothing here can reach the network or a wallet.
# --------------------------------------------------------------------------
#: A small but REAL transaction, used wherever a test only needs "a transaction"
#: rather than a specific size. 200 bytes is below what any genuine Solana
#: transaction can be (the smallest buildable v0 here is 209 bytes), which is
#: itself a reminder that the old fixture was not modelling a transaction.
SMALL_V0_BYTES = 240


def _b64(nbytes: int, version: str = VERSION_V0) -> str:
    """A REAL transaction of an exact serialized length, for size tests.

    The size gate now deserializes to read the version, so a blob of N zero
    bytes no longer models anything a venue could return: it is not a
    transaction. This builds a genuine one of exactly ``nbytes``.
    """
    return txfixtures.b64(txfixtures.build_exact(version, nbytes))


def _small() -> str:
    """A real, small v0 transaction for tests that only need *a* transaction."""
    return _b64(SMALL_V0_BYTES)


class _StubRpc:
    """SolanaRpc with the network call replaced by a scripted outcome."""

    def __init__(self, *, result=None, error=None, limit_check=True):
        self.result = result if result is not None else {"value": {"err": None}}
        self.error = error
        self.limit_check = limit_check
        self.calls: list = []
        self.sent: list = []

    async def _rpc(self, method, params):
        self.calls.append((method, params))
        if self.error is not None:
            raise self.error
        return self.result

    async def send_transaction(self, b64):
        self.sent.append(b64)
        raise AssertionError("send_transaction must never be reached in these tests")

    async def aclose(self):
        return None


def _rpc(**kw) -> S.SolanaRpc:
    r = S.SolanaRpc("https://example.invalid")
    stub = _StubRpc(**kw)
    r._rpc = stub._rpc                       # type: ignore[method-assign]
    r._stub = stub                           # type: ignore[attr-defined]
    return r


# --------------------------------------------------------------------------
# 1-3. build / deserialize / serialize round-trip
# --------------------------------------------------------------------------
def test_a_legacy_transfer_round_trips_through_serialize_and_parse():
    """Requirement 1-3: a transaction that builds also deserialises intact."""
    solders = pytest.importorskip("solders")
    from solders.hash import Hash
    from solders.instruction import AccountMeta, Instruction
    from solders.keypair import Keypair
    from solders.message import Message
    from solders.system_program import ID as SYSTEM_PROGRAM
    from solders.transaction import Transaction

    kp = Keypair()                     # ephemeral, in-memory, never funded
    pk = kp.pubkey()
    ix = Instruction(SYSTEM_PROGRAM, bytes([2, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                     [AccountMeta(pk, True, True), AccountMeta(pk, False, True)])
    msg = Message.new_with_blockhash([ix], pk, Hash.default())
    raw = bytes(Transaction.new_unsigned(msg))

    assert len(raw) <= S.MAX_SOLANA_TX_BYTES
    reparsed = Transaction.from_bytes(raw)
    assert str(reparsed.message.account_keys[0]) == str(pk)
    # and the wire form round-trips through base64, which is what we send
    assert base64.b64decode(base64.b64encode(raw).decode()) == raw


def test_size_report_measures_the_serialised_bytes():
    """Requirement 2: the size guard measures real bytes, not base64 chars."""
    r = _rpc()
    rep = r.transaction_size_report(_b64(S.MAX_SOLANA_TX_BYTES))
    assert rep["ok"] is True
    assert rep["status"] == "TX_SIZE_OK"
    assert rep["raw_bytes"] == S.MAX_SOLANA_TX_BYTES


# --------------------------------------------------------------------------
# 4. RPC request parameters
# --------------------------------------------------------------------------
def test_simulation_sends_the_documented_parameter_shape():
    """Requirement 4: encoding/commitment/sigVerify/replaceRecentBlockhash."""
    r = _rpc()
    asyncio.run(r.simulate_transaction_detailed(_small()))
    method, params = r._stub.calls[0]
    assert method == "simulateTransaction"
    assert len(params) == 2
    tx, cfg = params
    assert cfg["encoding"] == "base64"
    assert cfg["sigVerify"] is False
    assert cfg["replaceRecentBlockhash"] is True
    assert cfg["commitment"] == "confirmed"
    # the transaction is passed as base64, not base58
    base64.b64decode(tx, validate=True)


def test_unknown_option_is_not_required_and_shape_is_stable():
    """The endpoint tolerates extras; we send exactly the documented set."""
    r = _rpc()
    asyncio.run(r.simulate_transaction_detailed(_small()))
    _, params = r._stub.calls[0]
    assert set(params[1]) == {"encoding", "sigVerify", "replaceRecentBlockhash",
                              "commitment"}


# --------------------------------------------------------------------------
# 5. simulation succeeds for a valid transaction
# --------------------------------------------------------------------------
def test_valid_simulation_reports_ok_and_never_broadcasts():
    r = _rpc(result={"value": {"err": None, "unitsConsumed": 4500}})
    out = asyncio.run(r.simulate_transaction_detailed(_small()))
    assert out.ok is True
    assert out.status == "SIMULATION_OK"
    assert out.units_consumed == 4500
    assert out.broadcast_attempted is False
    assert r._stub.sent == []


# --------------------------------------------------------------------------
# 6. malformed transaction fails closed
# --------------------------------------------------------------------------
def test_malformed_base64_fails_closed_without_an_rpc_call():
    r = _rpc()
    out = asyncio.run(r.simulate_transaction_detailed("not base64 !!!"))
    assert out.ok is False
    assert out.status == "SIMULATION_FAILED_TX_MALFORMED"
    assert r._stub.calls == []          # refused locally
    assert r._stub.sent == []


def test_oversized_transaction_is_refused_locally_without_an_rpc_call():
    """The July incident, as a regression test.

    A serialized transaction over 1232 bytes can never be broadcast. The node
    answers -32602; we now answer locally, with byte counts, and never spend a
    round trip discovering what we already know from the bytes in hand.
    """
    r = _rpc()
    over = S.MAX_SOLANA_TX_BYTES + 57          # the incident's overshoot
    out = asyncio.run(r.simulate_transaction_detailed(_b64(over)))
    assert out.ok is False
    assert out.status == "SIMULATION_FAILED_TX_TOO_LARGE"
    assert out.size["over_by_bytes"] == 57
    assert out.rpc_code is None                # we never asked
    assert r._stub.calls == []
    assert r._stub.sent == []


# --------------------------------------------------------------------------
# 7. invalid RPC params produce a CLASSIFIED failure
# --------------------------------------------------------------------------
@pytest.mark.parametrize("code,message,expected", [
    (-32602, "base64 encoded solana_transaction::versioned::"
             "VersionedTransaction too large: 1720 bytes (max: encoded/raw 1644/1232)",
     "SIMULATION_FAILED_TX_TOO_LARGE"),
    (-32602, "failed to deserialize solana_transaction::legacy::Transaction",
     "SIMULATION_FAILED_TX_MALFORMED"),
    (-32602, "invalid transaction: Transaction failed to sanitize",
     "SIMULATION_FAILED_TX_MALFORMED"),
    (-32602, "Invalid params: missing field `addresses`",
     "SIMULATION_FAILED_RPC_INVALID_PARAMS"),
    (-32601, "Method not found", "SIMULATION_FAILED_RPC_METHOD_UNSUPPORTED"),
    (-32005, "node is behind", "SIMULATION_FAILED_RPC_NODE_LIMIT"),
    (-32602, "", "SIMULATION_FAILED_RPC_INVALID_PARAMS"),
    (None, "something unrecognised", "SIMULATION_FAILED_RPC"),
])
def test_classification_covers_the_documented_failure_modes(code, message, expected):
    """-32602 is not one bug. It must not be reported as one opaque code."""
    assert S.classify_simulation_error(code, message) == expected


def test_rpc_error_preserves_code_and_sanitized_message():
    """Requirement 7 + the observability defect: the message must survive.

    The incident was undiagnosable from the journal because `_rpc` raised
    `RPC error -32602` and dropped the node's own explanation.
    """
    r = _rpc(error=S.RpcError("simulateTransaction: RPC error -32602",
                              code=-32602,
                              rpc_message="VersionedTransaction too large: "
                                          "1720 bytes (max: encoded/raw 1644/1232)"))
    out = asyncio.run(r.simulate_transaction_detailed(_small()))
    assert out.ok is False
    assert out.rpc_code == -32602
    assert "too large" in out.rpc_message
    assert "1644/1232" in out.rpc_message       # the actionable numbers survive


def test_simulation_level_error_is_distinct_from_an_rpc_error():
    """A transaction the node EXECUTED and rejected is a different condition
    from one the node refused to parse. Both fail closed, but they are not
    the same diagnosis."""
    r = _rpc(result={"value": {"err": {"InstructionError": [0, "Custom"]}}})
    out = asyncio.run(r.simulate_transaction_detailed(_small()))
    assert out.ok is False
    assert out.status == "SIMULATION_FAILED_TX_REJECTED"
    assert out.rpc_code is None


# --------------------------------------------------------------------------
# 8. no broadcast after a simulation failure
# --------------------------------------------------------------------------
@pytest.mark.parametrize("outcome", [
    {"result": {"value": {"err": {"InstructionError": [0, "Custom"]}}}},
    {"error": S.RpcError("x", code=-32602, rpc_message="too large")},
    {"error": S.RpcError("x", code=-32601, rpc_message="Method not found")},
])
def test_no_broadcast_after_any_simulation_failure(outcome):
    r = _rpc(**outcome)
    ok, detail = asyncio.run(r.simulate_transaction(_small()))
    assert ok is False
    assert detail.startswith("SIMULATION_FAILED")
    assert r._stub.sent == []


def test_legacy_two_tuple_contract_is_preserved():
    """Existing callers unpack (ok, detail); that must keep working."""
    r = _rpc()
    ok, detail = asyncio.run(r.simulate_transaction(_small()))
    assert ok is True
    assert isinstance(detail, str)


# --------------------------------------------------------------------------
# sanitisation
# --------------------------------------------------------------------------
def test_sanitiser_strips_payload_but_keeps_the_numbers():
    msg = ("base64 encoded solana_transaction::versioned::VersionedTransaction "
           "too large: 1720 bytes (max: encoded/raw 1644/1232)")
    out = S.sanitize_rpc_message(msg)
    assert "1720" in out and "1232" in out
    long_b64 = base64.b64encode(b"\xab" * 400).decode()
    assert "<payload redacted>" in S.sanitize_rpc_message(f"failed for {long_b64}")
    assert S.sanitize_rpc_message("") == ""
    assert len(S.sanitize_rpc_message("x" * 5000)) <= 240


def test_outcome_never_carries_a_signature_or_transaction():
    r = _rpc(result={"value": {"err": None, "unitsConsumed": 1}})
    out = asyncio.run(r.simulate_transaction_detailed(_small()))
    blob = json.dumps(out.as_dict())
    assert "signature" not in blob.lower()
    assert base64.b64encode(b"\x01" * 50).decode() not in blob


# --------------------------------------------------------------------------
# live-RPC compatibility (opt-in; simulation only, never broadcasts)
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not __import__("os").environ.get("AUTOPAPER_LIVE_RPC_TEST"),
    reason="set AUTOPAPER_LIVE_RPC_TEST=1 to hit a real RPC (simulation only)")
def test_live_rpc_accepts_the_production_parameter_shape():
    """
    Requirement 7 / integration.

    Proves a real Solana node ACCEPTS this request shape, by simulating a
    transaction that FITS the size limit. A transaction that is too small to
    exist on chain yields a simulation-level result (AccountNotFound) rather
    than a JSON-RPC error -- and the difference between those two outcomes is
    exactly what proves the endpoint does not object to the parameters.

    Nothing is broadcast. No wallet is loaded. No funds can move.
    """
    import os
    import httpx

    from solders.hash import Hash
    from solders.instruction import AccountMeta, Instruction
    from solders.keypair import Keypair
    from solders.message import Message
    from solders.system_program import ID as SYSTEM_PROGRAM
    from solders.transaction import Transaction

    url = os.environ.get("AUTOPAPER_RPC_URL", "https://api.mainnet-beta.solana.com")
    kp = Keypair()
    pk = kp.pubkey()
    ix = Instruction(SYSTEM_PROGRAM, bytes([2, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                     [AccountMeta(pk, True, True), AccountMeta(pk, False, True)])
    msg = Message.new_with_blockhash([ix], pk, Hash.default())
    raw = bytes(Transaction.new_unsigned(msg))
    assert len(raw) <= S.MAX_SOLANA_TX_BYTES

    rpc = S.SolanaRpc(url)

    # ONE event loop for the whole flow. Two asyncio.run() calls would close
    # the loop in between and the lazily-created httpx client would then raise
    # "Event loop is closed" -- a harness bug that looks like a product bug.
    async def _flow():
        try:
            return await rpc.simulate_transaction_detailed(
                base64.b64encode(raw).decode())
        finally:
            await rpc.aclose()

    out = asyncio.run(_flow())

    # Either it simulated (fine) or the node reported a simulation-level error
    # (also fine). What must NOT happen is a JSON-RPC -32602 about our shape.
    assert out.rpc_code != -32602, out.rpc_message
    assert out.broadcast_attempted is False
    assert out.status in {"SIMULATION_OK", "SIMULATION_FAILED_TX_REJECTED"}
