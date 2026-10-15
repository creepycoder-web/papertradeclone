"""Route-failure audit: Raptor -> Jupiter fallback and failure classification.

Deterministic and offline. No test reaches a real endpoint, RPC or wallet, and
no test signs for broadcast or broadcasts anything.

The properties pinned here are the ones the $bluechip and $DOTCOM events turned
on:

  A. A Raptor HTTP 500 "No multi-hop route found" is a ROUTE failure and the
     Jupiter fallback IS attempted -- both at the executor and at the pre-trade
     ``quote_valid`` gate that previously asked only the primary provider.
  B. Raptor no-route + Jupiter no-route => a safe ``route_unavailable`` refusal,
     no position, no broadcast.
  C. A Raptor SIMULATION failure does NOT trigger the fallback.
  D. A ``Custom`` simulation error is classified as a DEX PROGRAM ERROR and the
     emitting program and instruction index are named from the transaction.
  E. A custom error is never guessed into a success.
  F. A structured simulation rejection is a TERMINAL failure, never UNKNOWN.
  G. Nothing is broadcast on any failure path.
"""


from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import httpx

from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.transaction import VersionedTransaction

from autopaper.execution.failures import (
    DEX_PROGRAM_ERROR,
    ROUTE_UNAVAILABLE,
    SIMULATION_FAILED,
    classify_execution_failure,
    parse_instruction_error,
    program_label,
)
from autopaper.execution.intents import ExecutionIntentStore, FAILED
from autopaper.execution.jupiter import JupiterProvider
from autopaper.execution.raptor import RaptorProvider
from autopaper.execution.real import LiveSolanaBackend, RealExecutor
from autopaper.engine import PaperTrader
from autopaper.execution.solana import SimulationOutcome, WSOL_MINT
from autopaper.latency import BLOCK_REASONS

MINT = "H2LWQMQdEGamrEQh8v4jiwBhW3hh1iqcDwU815xJ881t"
#: The wallet pubkey the doubles report; must match the reconciled owner.
OWNER = "SoWaLanaPubKey1111111111111111111111111111111"
#: A real, well-known venue program, used as instruction 6's program.
VENUE_PROGRAM = "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK"  # Raydium CLMM
VENUE_LABEL = "Raydium CLMM"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _tx_b64(instructions=None) -> str:
    ixs = instructions if instructions is not None else [
        Instruction(Pubkey.default(), b"\x00\x01", [
            AccountMeta(Keypair().pubkey(), False, True)])]
    msg = MessageV0.try_compile(Keypair().pubkey(), ixs, [], Hash.default())
    tx = VersionedTransaction.populate(msg, [Signature.default()])
    return base64.b64encode(bytes(tx)).decode()


def _tx_with_venue_at(index: int = 6, program: str = VENUE_PROGRAM) -> str:
    """A transaction whose instruction ``index`` is owned by ``program``."""
    ixs = [
        Instruction(Pubkey.default(), bytes([i]), [
            AccountMeta(Keypair().pubkey(), False, True)])
        for i in range(index)
    ]
    ixs.append(Instruction(Pubkey.from_string(program), b"\x01\x02", [
        AccountMeta(Keypair().pubkey(), False, True)]))
    return _tx_b64(ixs)


def _raptor_quote(**over) -> dict:
    q = {
        "inputMint": WSOL_MINT, "outputMint": MINT,
        "amountIn": "1071843", "amountOut": "3280257",
        "minAmountOut": "3116000", "feeAmount": "2403",
        "priceImpact": 0.0123, "slippageBps": 500,
        "routePlan": [{"programId": "P1", "dex": "Raydium CLMM",
                       "pool": "pool1", "percent": 100}],
        "contextSlot": 1,
    }
    q.update(over)
    return q


def _jup_quote(**over) -> dict:
    q = {
        "inputMint": WSOL_MINT, "inAmount": "1071843", "outputMint": MINT,
        "outAmount": "3280000", "otherAmountThreshold": "3116000",
        "priceImpactPct": "0.0012", "slippageBps": 500,
        "routePlan": [{"swapInfo": {"label": "Raydium"}}],
    }
    q.update(over)
    return q


class _RapSrv:
    def __init__(self, *, quote_status=200, quote_text=None, swap_status=200,
                 swap=None):
        self.quote_status = quote_status
        self.quote_text = quote_text
        self.swap_status = swap_status
        self.swap = swap or {"swapTransaction": _tx_b64(),
                             "lastValidBlockHeight": 100,
                             "prioritizationFeeLamports": 50000}
        self.calls: list = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode()) if request.content else None
        self.calls.append((request.url.path, body))
        if request.url.path.endswith("/quote"):
            if self.quote_text is not None:
                return httpx.Response(self.quote_status, text=self.quote_text)
            return httpx.Response(self.quote_status, json=_raptor_quote())
        if request.url.path.endswith("/swap"):
            return httpx.Response(self.swap_status, json=self.swap)
        return httpx.Response(404, json={})

    def provider(self) -> RaptorProvider:
        return RaptorProvider(
            client=httpx.AsyncClient(transport=httpx.MockTransport(self.handler)))


class _JupSrv:
    def __init__(self, *, quote_status=200, quote_text=None, swap_status=200,
                 swap=None):
        self.quote_status = quote_status
        self.quote_text = quote_text
        self.swap_status = swap_status
        self.swap = swap or {"swapTransaction": _tx_b64(),
                             "lastValidBlockHeight": 100}
        self.calls: list = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode()) if request.content else None
        self.calls.append((request.url.path, body))
        if request.url.path.endswith("/quote"):
            if self.quote_text is not None:
                return httpx.Response(self.quote_status, text=self.quote_text)
            return httpx.Response(self.quote_status, json=_jup_quote())
        if request.url.path.endswith("/swap"):
            return httpx.Response(self.swap_status, json=self.swap)
        return httpx.Response(404, json={})

    def provider(self) -> JupiterProvider:
        return JupiterProvider(
            base_url="https://jup.invalid/swap/v1",
            client=httpx.AsyncClient(transport=httpx.MockTransport(self.handler)))


def _raptor_no_route() -> _RapSrv:
    return _RapSrv(quote_status=500,
                   quote_text=json.dumps({"error": "No multi-hop route found"}))


def _jup_no_route() -> _JupSrv:
    return _JupSrv(quote_status=404, quote_text=json.dumps({"error": "no route"}))


class _Rpc:
    def __init__(self, *, sim_ok=True, raw_err=None, logs=None, swap_tx=None):
        self.sim_ok = sim_ok
        self.raw_err = raw_err
        self.logs = logs or []
        self.sent: list = []
        self.sim_calls = 0

    async def simulate_transaction_detailed(self, tx_b64):
        self.sim_calls += 1
        if self.sim_ok:
            return SimulationOutcome(
                ok=True, status="SIMULATION_OK", detail="simulation ok",
                stage="SIMULATION", rpc_code=None, rpc_message="",
                broadcast_attempted=False)
        return SimulationOutcome(
            ok=False, status="SIMULATION_FAILED_TX_REJECTED",
            detail=f"simulation error: {self.raw_err}", stage="SIMULATION",
            rpc_code=None, rpc_message=str(self.raw_err),
            broadcast_attempted=False, raw_err=self.raw_err,
            logs=list(self.logs))

    async def send_transaction(self, tx_b64):
        self.sent.append(tx_b64)
        return "5Sig"

    async def wait_for_confirmation(self, sig, **kw):
        return {"status": "confirmed", "error": None, "signature": sig, "slot": 7}

    async def get_address_lookup_table(self, address):
        return {"table": address, "query_ok": True, "usable": False,
                "addresses": [], "source": "double", "detail": "none"}

    async def get_transaction(self, sig, commitment="confirmed"):
        return {
            "slot": 7,
            "transaction": {"message": {"accountKeys": ["x"]}, "meta": {
                "err": None, "fee": 5000, "logMessages": [],
                "preBalances": [1_071_843], "postBalances": [0],
                "preTokenBalances": [],
                "postTokenBalances": [{
                    "mint": MINT, "owner": OWNER,
                    "uiTokenAmount": {"amount": "3280000", "decimals": 0}}]}},
        }


class _Wallet:
    def __init__(self):
        self.pubkey = OWNER
        self.masked = "(masked)"
        self.signed: list = []

    def sign_wire(self, raw: bytes) -> bytes:
        self.signed.append(raw)
        return raw


class _Cfg:
    def __init__(self, **kw):
        self.wallet_keypair_path = None
        self.rpc_url = "https://rpc.invalid"
        self.slippage_bps = 500
        self.simulate_before_send = True
        self.priority_fee_lamports = 50_000
        self.confirmation_timeout_s = 5.0
        self.quote_max_age_ms = 8000.0
        self.max_price_impact_pct = 5.0
        self.raptor_enabled = True
        self.raptor_base_url = "https://raptor.solanatracker.io"
        self.raptor_fallback_enabled = kw.get("raptor_fallback_enabled", True)


def _backend(primary, fallback, *, rpc=None, wallet=None, cfg=None):
    wallet = wallet or _Wallet()
    rpc = rpc or _Rpc()
    b = LiveSolanaBackend(cfg or _Cfg(), wallet=wallet, rpc=rpc,
                          provider=primary, fallback_provider=fallback)
    return b, wallet, rpc


def _run(coro):
    return asyncio.run(coro)


def _buy(backend, **kw):
    return _run(backend.execute_swap("ENTRY", WSOL_MINT, MINT, 1_071_843, 500,
                                     **kw))


def _probe(backend):
    ex = RealExecutor(_Cfg(), backend=backend)
    check, fill = _run(ex.quote_probe(
        mint=MINT, position_lamports=1_071_843, slippage_bps=500, side="ENTRY"))
    return ex, check, fill


# ---------------------------------------------------------------------------
# A. Raptor HTTP 500 + no route -> Jupiter fallback attempted
# ---------------------------------------------------------------------------
def test_A_raptor_http_500_no_route_attempts_jupiter_fallback():
    rap, jup = _raptor_no_route(), _JupSrv()
    backend, wallet, rpc = _backend(rap.provider(), jup.provider())
    fill = _buy(backend)
    assert fill.status == "confirmed"
    assert fill.route_metadata["provider_name"] == "jupiter"
    assert any(p.endswith("/quote") for p, _ in jup.calls), \
        "the fallback provider must have been consulted"
    assert len(rpc.sent) == 1, "exactly one broadcast"
    assert len(wallet.signed) == 1


def test_A2_pretrade_quote_probe_reaches_the_fallback():
    """The $bluechip root cause: the pre-trade gate asked Raptor only."""
    rap, jup = _raptor_no_route(), _JupSrv()
    backend, wallet, rpc = _backend(rap.provider(), jup.provider())
    ex, check, fill = _probe(backend)
    assert check.ok, check.detail
    assert ex.last_quote_probe["fallback_attempted"] is True
    assert ex.last_quote_probe["provider"] == "jupiter"
    assert ex.last_quote_probe["category"] == ""
    assert fill is None
    assert jup.calls, "the fallback provider must have been consulted"
    assert rpc.sent == [], "probing never broadcasts"


def test_A3_no_route_from_both_is_route_unavailable_at_the_probe():
    backend, _, _ = _backend(_raptor_no_route().provider(),
                             _jup_no_route().provider())
    ex, check, fill = _probe(backend)
    assert not check.ok
    assert ex.last_quote_probe["category"] == ROUTE_UNAVAILABLE
    assert ex.last_quote_probe["fallback_attempted"] is True
    assert fill is None


# ---------------------------------------------------------------------------
# B. Raptor no route + Jupiter no route -> safe rejection
# ---------------------------------------------------------------------------
def test_B_both_no_route_is_a_safe_route_unavailable_rejection():
    backend, wallet, rpc = _backend(_raptor_no_route().provider(),
                                    _jup_no_route().provider())
    fill = _buy(backend)
    assert fill.status == "REAL_BUY_BLOCKED_NO_VALID_EXECUTION_ROUTE"
    assert not fill.filled
    assert fill.route_metadata["failure"]["category"] == ROUTE_UNAVAILABLE
    assert fill.route_metadata["failure"]["fallback_attempted"] is True
    assert rpc.sent == [] and wallet.signed == []


# ---------------------------------------------------------------------------
# C. Raptor simulation failure -> no fallback
# ---------------------------------------------------------------------------
def test_C_raptor_simulation_failure_does_not_fall_back():
    jup = _JupSrv()
    backend, wallet, rpc = _backend(
        _RapSrv().provider(), jup.provider(),
        rpc=_Rpc(sim_ok=False, raw_err={"InstructionError": [
            3, {"Custom": 6001}]}))
    fill = _buy(backend)
    assert fill.status == "RAPTOR_SIMULATION_FAILED"
    assert jup.calls == [], "a simulation failure is not a fallback trigger"
    assert fill.route_metadata["failure"]["fallback_attempted"] is False
    assert rpc.sent == [] and wallet.signed == []


# ---------------------------------------------------------------------------
# D. Custom 6001 -> exact program/error classification
# ---------------------------------------------------------------------------
def test_D_custom_6001_names_the_program_and_instruction_index():
    tx = _tx_with_venue_at(6)
    rap = _RapSrv(swap={"swapTransaction": tx, "lastValidBlockHeight": 100,
                        "prioritizationFeeLamports": 50000})
    jup = _JupSrv()
    rpc = _Rpc(sim_ok=False, raw_err={"InstructionError": [6, {"Custom": 6001}]},
               logs=["Program CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK "
                     "invoke [1]",
                     "Program ... failed: custom program error: 0x1771"])
    backend, wallet, r = _backend(rap.provider(), jup.provider(), rpc=rpc)
    fill = _buy(backend)
    f = fill.route_metadata["failure"]
    assert f["category"] == DEX_PROGRAM_ERROR
    assert f["instruction_index"] == 6
    assert f["custom_code"] == 6001
    assert f["program_id"] == VENUE_PROGRAM
    assert f["program_label"] == VENUE_LABEL
    assert f["error_kind"] == "Custom"
    assert any("invoke" in line for line in f.get("simulation_logs", []))
    assert fill.status == "RAPTOR_SIMULATION_FAILED"
    assert not fill.filled
    assert r.sent == [] and wallet.signed == []


def test_D2_6001_has_no_universal_meaning_it_is_program_scoped():
    """The same byte code under a different program is not the same error."""
    tx = _tx_with_venue_at(6, program="TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
    rap = _RapSrv(swap={"swapTransaction": tx, "lastValidBlockHeight": 100})
    rpc = _Rpc(sim_ok=False, raw_err={"InstructionError": [6, {"Custom": 6001}]})
    backend, _, _ = _backend(rap.provider(), _JupSrv().provider(), rpc=rpc)
    fill = _buy(backend)
    f = fill.route_metadata["failure"]
    assert f["category"] == DEX_PROGRAM_ERROR
    assert f["program_id"] == "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
    assert f["program_label"] == "SPL Token"


# ---------------------------------------------------------------------------
# E. Unknown custom error -> never guessed as success
# ---------------------------------------------------------------------------
def test_E_unknown_custom_error_is_never_a_success():
    tx = _tx_with_venue_at(6, program="MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")
    rap = _RapSrv(swap={"swapTransaction": tx, "lastValidBlockHeight": 100})
    rpc = _Rpc(sim_ok=False, raw_err={"InstructionError": [6, {"Custom": 4242}]})
    backend, _, r = _backend(rap.provider(), _JupSrv().provider(), rpc=rpc)
    fill = _buy(backend)
    f = fill.route_metadata["failure"]
    # The program is named even though its error dictionary is unknown, and the
    # code is reported verbatim -- never interpreted as a success.
    assert f["custom_code"] == 4242
    assert f["program_id"] == "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"
    assert f["category"] == DEX_PROGRAM_ERROR
    assert fill.filled is False
    assert fill.status == "RAPTOR_SIMULATION_FAILED"
    assert r.sent == []


def test_E2_a_plain_simulation_failure_stays_simulation_failed():
    assert classify_execution_failure(
        "RAPTOR_SIMULATION_FAILED", detail="simulation error") \
        == SIMULATION_FAILED
    # A resolved custom code is a DEX-program refusal, not a generic failure.
    assert classify_execution_failure(
        "RAPTOR_SIMULATION_FAILED",
        detail="simulation error: {'InstructionError': [6, {'Custom': 6001}]}",
        custom_code=6001, program_id=VENUE_PROGRAM) == DEX_PROGRAM_ERROR


# ---------------------------------------------------------------------------
# F. Structured simulation rejection -> terminal, never UNKNOWN
# ---------------------------------------------------------------------------
def test_F_structured_simulation_rejection_is_terminal_not_unknown(tmp_path):
    tx = _tx_with_venue_at(6)
    rap = _RapSrv(swap={"swapTransaction": tx, "lastValidBlockHeight": 100})
    rpc = _Rpc(sim_ok=False, raw_err={"InstructionError": [6, {"Custom": 6001}]})
    store = ExecutionIntentStore(tmp_path / "intents.jsonl")
    backend, wallet, r = _backend(rap.provider(), _JupSrv().provider(), rpc=rpc)
    fill = _buy(backend, intent_store=store, logical_key=f"ENTRY:{MINT}")
    assert fill.status == "RAPTOR_SIMULATION_FAILED"
    assert fill.status != "AMBIGUOUS"
    assert fill.signature == ""
    assert fill.requires_recovery is False
    intents = [i for i in store.all_intents() if i.logical_key == f"ENTRY:{MINT}"]
    assert len(intents) == 1
    assert intents[0].state == FAILED, \
        "a structured rejection with no signature is terminal FAILED"
    assert intents[0].state != "UNKNOWN"
    assert r.sent == [] and wallet.signed == []


# ---------------------------------------------------------------------------
# G. No transaction broadcast on any failure path
# ---------------------------------------------------------------------------
def test_G_no_transaction_is_broadcast_on_any_failure():
    cases = [
        (_raptor_no_route().provider(), _jup_no_route().provider(), _Rpc()),
        (_RapSrv().provider(), _JupSrv().provider(),
         _Rpc(sim_ok=False, raw_err={"InstructionError": [6, {"Custom": 6001}]})),
        (_raptor_no_route().provider(), _JupSrv().provider(),
         _Rpc(sim_ok=False, raw_err=None)),
    ]
    for primary, fallback, rpc in cases:
        backend, wallet, r = _backend(primary, fallback, rpc=rpc)
        fill = _buy(backend)
        assert not fill.filled
        assert r.sent == [], "no failure path may broadcast"
        assert wallet.signed == [], "no failure path may sign for broadcast"
    assert parse_instruction_error(
        {"InstructionError": [6, {"Custom": 6001}]}) == (6, "Custom", 6001)
    assert program_label(VENUE_PROGRAM) == VENUE_LABEL


# ---------------------------------------------------------------------------
# H. The operator log states the CATEGORY, not "execution_exception"
# ---------------------------------------------------------------------------
def test_H_pretrade_block_reason_names_the_category():
    quote_failure = SimpleNamespace(name="quote_valid")
    fake = SimpleNamespace(real_executor=SimpleNamespace(
        last_quote_probe={"category": ROUTE_UNAVAILABLE}))
    assert PaperTrader._pretrade_block_reason(fake, [quote_failure]) \
        == ROUTE_UNAVAILABLE

    # A probe that ran but produced no category is a quote-invalid refusal.
    bare = SimpleNamespace(real_executor=SimpleNamespace(last_quote_probe={}))
    assert PaperTrader._pretrade_block_reason(bare, [quote_failure]) \
        == "quote_invalid"

    # A non-execution gate is NOT called an execution exception.
    exposure = SimpleNamespace(name="exposure_within_limit")
    assert PaperTrader._pretrade_block_reason(fake, [exposure]) \
        == "pretrade_refused"


def test_H2_every_category_is_a_described_block_reason():
    for reason in (ROUTE_UNAVAILABLE, "quote_invalid", "simulation_failed",
                   "slippage_exceeded", "dex_program_error", "rpc_rejected",
                   "transaction_unknown", "execution_failed",
                   "pretrade_refused"):
        assert BLOCK_REASONS.get(reason), reason
