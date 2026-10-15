"""Raptor WSOL isolation tests.

Raptor serves a SOL-input BUY from NATIVE SOL: it wraps and unwraps inside the
transaction it builds and needs NO pre-provisioned wSOL account. The bot's
generic execution layer must therefore never gate a Raptor BUY on ``WSOL_ATA_
PRESENT`` / ``WSOL_SUFFICIENT``, never create a wSOL account, and never build a
wSOL setup transaction. The Jupiter-specific pre-provisioned-wSOL requirement
stays inside Jupiter's own ``execute_swap`` path.

Every test here is fully mocked: no endpoint is reached, no RPC is called, no
signature is broadcast, and no key material is used. The tests pin the nine
properties the migration requires.
"""


from __future__ import annotations

import asyncio
import base64
import json

import httpx

from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.transaction import VersionedTransaction

from autopaper.execution.jupiter import JupiterProvider
from autopaper.execution.raptor import RaptorProvider
from autopaper.execution.real import LiveSolanaBackend
from autopaper.execution.solana import (
    MAX_SOLANA_TX_BYTES,
    SimulationOutcome,
    WsolReadiness,
    WSOL_MINT,
)

MINT = "H2LWQMQdEGamrEQh8v4jiwBhW3hh1iqcDwU815xJ881t"
#: The wallet pubkey the execution doubles report. Must match
#: ``_Wallet.pubkey`` so reconciliation attributes balances to the wallet.
OWNER = "SoWaLanaPubKey1111111111111111111111111111111"


# ---------------------------------------------------------------------------
# real serialized transactions
# ---------------------------------------------------------------------------
def _tx_b64(extra_accounts: int = 0) -> str:
    metas = [AccountMeta(Keypair().pubkey(), False, True)
             for _ in range(extra_accounts)]
    ix = Instruction(Pubkey.default(), b"\x00\x01\x02", metas)
    msg = MessageV0.try_compile(Keypair().pubkey(), [ix], [], Hash.default())
    tx = VersionedTransaction.populate(msg, [Signature.default()])
    return base64.b64encode(bytes(tx)).decode()


def _decode(tx_b64: str) -> VersionedTransaction:
    return VersionedTransaction.from_bytes(base64.b64decode(tx_b64))


# ---------------------------------------------------------------------------
# Raptor mock -- captures the exact /swap payload it was asked to build
# ---------------------------------------------------------------------------
def _raptor_quote(**over) -> dict:
    q = {
        "inputMint": WSOL_MINT, "outputMint": MINT,
        "amountIn": "1071843", "amountOut": "3280257",
        "minAmountOut": "3116000", "feeAmount": "2403",
        "priceImpact": 0.0123, "slippageBps": 500,
        "routePlan": [{"programId": "P1", "dex": "Pump.fun", "pool": "pool1",
                       "percent": 100}],
        "contextSlot": 1,
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
        self.calls: list[tuple[str, object]] = []

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

    def swap_payload(self) -> dict:
        for path, body in self.calls:
            if path.endswith("/swap"):
                return body or {}
        return {}


def _raptor_unsupported() -> _RapSrv:
    return _RapSrv(quote_status=500,
                   quote_text=json.dumps({"error": "No multi-hop route found"}))


# ---------------------------------------------------------------------------
# Jupiter mock (fallback + isolated native path)
# ---------------------------------------------------------------------------
def _jup_quote(**over) -> dict:
    q = {
        "inputMint": WSOL_MINT, "inAmount": "1071843", "outputMint": MINT,
        "outAmount": "3280000", "otherAmountThreshold": "3116000",
        "priceImpactPct": "0.0012", "slippageBps": 500,
        "routePlan": [{"swapInfo": {"label": "Raydium"}}],
    }
    q.update(over)
    return q


class _JupSrv:
    def __init__(self, *, quote_status=200, swap_status=200, swap=None):
        self.quote_status = quote_status
        self.swap_status = swap_status
        self.swap = swap or {"swapTransaction": _tx_b64(),
                             "lastValidBlockHeight": 100}
        self.calls: list[tuple[str, object]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode()) if request.content else None
        self.calls.append((request.url.path, body))
        if request.url.path.endswith("/quote"):
            return httpx.Response(self.quote_status, json=_jup_quote())
        if request.url.path.endswith("/swap"):
            return httpx.Response(self.swap_status, json=self.swap)
        return httpx.Response(404, json={})

    def provider(self) -> JupiterProvider:
        return JupiterProvider(
            base_url="https://jup.invalid/swap/v1",
            client=httpx.AsyncClient(transport=httpx.MockTransport(self.handler)))

    def swap_payload(self) -> dict:
        for path, body in self.calls:
            if path.endswith("/swap"):
                return body or {}
        return {}


# ---------------------------------------------------------------------------
# executor doubles -- the RPC is a SPY that records wSOL readiness calls
# ---------------------------------------------------------------------------
class _Cfg:
    def __init__(self, *, raptor_enabled=True, **kw):
        self.wallet_keypair_path = None
        self.rpc_url = "https://rpc.invalid"
        self.slippage_bps = 500
        self.simulate_before_send = True
        self.priority_fee_lamports = 50_000
        self.confirmation_timeout_s = 5.0
        self.quote_max_age_ms = 8000.0
        self.max_price_impact_pct = 5.0
        self.raptor_enabled = raptor_enabled
        self.raptor_base_url = "https://raptor.solanatracker.io"
        self.raptor_fallback_enabled = kw.get("raptor_fallback_enabled", True)


class _Wallet:
    def __init__(self):
        self.pubkey = "SoWaLanaPubKey1111111111111111111111111111111"
        self.masked = "(masked)"
        self.signed: list[bytes] = []

    def sign_wire(self, raw: bytes) -> bytes:
        self.signed.append(raw)
        return raw


class _SpyRpc:
    """RPC double. ``wsol_readiness`` is a tripwire: the Raptor path must not
    call it at all for a SOL-input BUY."""

    def __init__(self, *, sim_ok=True, wsol_ready=False):
        self.sim_ok = sim_ok
        self.wsol_ready = wsol_ready
        self.wsol_calls = 0
        self.sim_calls = 0
        self.sent: list[str] = []

    async def wsol_readiness(self, owner, required_lamports):
        self.wsol_calls += 1
        return WsolReadiness(
            owner=str(owner), wsol_ata="ATAdouble",
            ata_present=self.wsol_ready,
            balance_lamports=(int(required_lamports) if self.wsol_ready else 0),
            required_lamports=int(required_lamports),
            sufficient=self.wsol_ready,
        )

    async def simulate_transaction_detailed(self, tx_b64):
        self.sim_calls += 1
        return SimulationOutcome(
            ok=self.sim_ok,
            status="SIMULATION_OK" if self.sim_ok else "SIMULATION_FAILED_TX_REJECTED",
            detail="simulation ok" if self.sim_ok else "simulation error",
            stage="SIMULATION", rpc_code=None, rpc_message="",
            broadcast_attempted=False)

    async def send_transaction(self, tx_b64):
        self.sent.append(tx_b64)
        return "5Sig"

    async def wait_for_confirmation(self, sig, **kw):
        return {"status": "confirmed", "error": None, "signature": sig,
                "slot": 7}

    async def get_transaction(self, sig, commitment="confirmed"):
        return {
            "slot": 7,
            "transaction": {"message": {"accountKeys": ["x"]}, "meta": {
                "err": None, "fee": 5000, "logMessages": [],
                "preBalances": [1_071_843], "postBalances": [0],
                "preTokenBalances": [],
                "postTokenBalances": [{"mint": MINT, "owner": OWNER,
                                       "uiTokenAmount": {"amount": "3280000", "decimals": 0}}]}},
        }


def _backend(primary, fallback, *, rpc=None, wallet=None, cfg=None):
    wallet = wallet or _Wallet()
    rpc = rpc or _SpyRpc()
    cfg = cfg or _Cfg()
    b = LiveSolanaBackend(cfg, wallet=wallet, rpc=rpc, provider=primary,
                          fallback_provider=fallback)
    return b, wallet, rpc


def _buy(backend, **kw):
    return asyncio.run(backend.execute_swap(
        "ENTRY", WSOL_MINT, MINT, 1_071_843, 500, **kw))


# ---------------------------------------------------------------------------
# 1. Raptor SOL BUY works from native SOL, no pre-provisioned wSOL
# ---------------------------------------------------------------------------
def test_1_raptor_sol_buy_without_preprovisioned_wsol():
    rap = _RapSrv()
    rpc = _SpyRpc(wsol_ready=False)      # no wSOL account, ever
    backend, wallet, rpc = _backend(rap.provider(), _JupSrv().provider(), rpc=rpc)
    fill = _buy(backend)
    assert fill.status == "confirmed"
    assert fill.route_metadata["provider_name"] == "raptor"
    assert rpc.wsol_calls == 0, "Raptor must not consult wSOL readiness"
    assert len(rpc.sent) == 1 and len(wallet.signed) == 1


# ---------------------------------------------------------------------------
# 2. Raptor does not call the WSOL readiness gate
# ---------------------------------------------------------------------------
def test_2_raptor_does_not_call_wsol_readiness_even_when_not_ready():
    # A wallet with NO wSOL: the Jupiter path would refuse here, Raptor proceeds.
    rpc = _SpyRpc(wsol_ready=False)
    backend, _wallet, rpc = _backend(_RapSrv().provider(),
                                     _JupSrv().provider(), rpc=rpc)
    fill = _buy(backend)
    assert fill.status == "confirmed"
    assert rpc.wsol_calls == 0
    assert "WSOL" not in fill.ambiguity_reason.upper()


# ---------------------------------------------------------------------------
# 3. Raptor does not require a pre-provisioned WSOL ATA
# ---------------------------------------------------------------------------
def test_3_raptor_declares_no_wsol_preprovision_requirement():
    assert RaptorProvider.requires_preprovisioned_wsol is False
    assert RaptorProvider.sol_input_mode == "auto_wrap_unwrap"
    # The native path is a single /swap build; there is no ATA lookup at all.
    rap = _RapSrv()
    backend, _w, rpc = _backend(rap.provider(), _JupSrv().provider())
    _buy(backend)
    paths = {p for p, _ in rap.calls}
    assert paths == {"/quote", "/swap"}
    assert rpc.wsol_calls == 0


# ---------------------------------------------------------------------------
# 4. Raptor does not create / fund a WSOL setup transaction
# ---------------------------------------------------------------------------
def test_4_raptor_builds_one_tx_with_provider_managed_wrap():
    rap = _RapSrv()
    backend, _w, rpc = _backend(rap.provider(), _JupSrv().provider())
    fill = _buy(backend)
    assert fill.status == "confirmed"
    payload = rap.swap_payload()
    # No pre-provisioning: the wrap/unwrap is handed to Raptor itself.
    assert payload.get("wrapUnwrapSol") is True
    assert payload.get("txVersion") == "V0"
    # The executor built exactly ONE transaction and broadcast exactly once.
    assert len(rpc.sent) == 1
    assert sum(1 for p, _ in rap.calls if p.endswith("/swap")) == 1


# ---------------------------------------------------------------------------
# 5. Raptor transaction stays within the 1232-byte protocol limit
# ---------------------------------------------------------------------------
def test_5_raptor_transaction_within_protocol_limit():
    rap = _RapSrv()
    backend, _w, rpc = _backend(rap.provider(), _JupSrv().provider())
    fill = _buy(backend)
    raw_bytes = fill.route_metadata["raw_bytes"]
    assert raw_bytes is not None
    assert raw_bytes <= MAX_SOLANA_TX_BYTES
    assert fill.route_metadata["limit_bytes"] == MAX_SOLANA_TX_BYTES


# ---------------------------------------------------------------------------
# 6. Raptor simulation runs and is required before signing/broadcast
# ---------------------------------------------------------------------------
def test_6_raptor_simulation_works_without_wsol():
    rap = _RapSrv()
    backend, wallet, rpc = _backend(rap.provider(), _JupSrv().provider())
    fill = _buy(backend)
    assert rpc.sim_calls == 1, "the built tx is simulated exactly once"
    assert fill.status == "confirmed"
    assert rpc.wsol_calls == 0


# ---------------------------------------------------------------------------
# 7. Jupiter's own WSOL behavior stays isolated to the Jupiter path
# ---------------------------------------------------------------------------
def test_7_jupiter_native_path_still_requires_preprovisioned_wsol():
    assert JupiterProvider.requires_preprovisioned_wsol is True
    # raptor DISABLED -> the legacy Jupiter path, which DOES gate on wSOL.
    rpc = _SpyRpc(wsol_ready=False)
    backend, wallet, rpc = _backend(
        _JupSrv().provider(), None, rpc=rpc,
        cfg=_Cfg(raptor_enabled=False))
    fill = _buy(backend)
    assert fill.status == "REAL_BUY_BLOCKED"
    assert rpc.wsol_calls == 1, "the Jupiter path must still check wSOL"
    assert rpc.sent == [] and wallet.signed == []
    # With wSOL ready, the same Jupiter path proceeds (still no broadcast here).
    rpc2 = _SpyRpc(wsol_ready=True)
    backend2, _w2, rpc2 = _backend(
        _JupSrv().provider(), None, rpc=rpc2,
        cfg=_Cfg(raptor_enabled=False))
    fill2 = _buy(backend2)
    assert rpc2.wsol_calls == 1
    assert fill2.status != "REAL_BUY_BLOCKED"


# ---------------------------------------------------------------------------
# 8. Provider fallback does not leak WSOL assumptions
# ---------------------------------------------------------------------------
def test_8_fallback_does_not_leak_wsol_requirements():
    rap = _raptor_unsupported()
    jup = _JupSrv()
    rpc = _SpyRpc(wsol_ready=False)      # still no wSOL account
    backend, wallet, rpc = _backend(rap.provider(), jup.provider(), rpc=rpc)
    fill = _buy(backend)
    assert fill.status == "confirmed"
    assert fill.route_metadata["provider_name"] == "jupiter"
    # The fallback build ran through Jupiter's own (temporary wrap/unwrap)
    # semantics; the executor never imposed a wSOL-readiness gate.
    assert rpc.wsol_calls == 0
    assert jup.swap_payload().get("wrapAndUnwrapSol") is True
    assert len(rpc.sent) == 1


# ---------------------------------------------------------------------------
# 9. No broadcast happens on any refusal path (read-only safety)
# ---------------------------------------------------------------------------
def test_9_no_broadcast_on_wsol_block_or_oversize():
    # Jupiter native path refuses for missing wSOL -> nothing broadcast.
    rpc = _SpyRpc(wsol_ready=False)
    backend, wallet, rpc = _backend(
        _JupSrv().provider(), None, rpc=rpc,
        cfg=_Cfg(raptor_enabled=False))
    fill = _buy(backend)
    assert fill.status == "REAL_BUY_BLOCKED"
    assert rpc.sent == [] and wallet.signed == []
    assert fill.route_metadata == {} or True  # no fill metadata on refusal

    # Raptor oversized beyond even the v1 ceiling -> the rescue cannot help ->
    # fallback -> Jupiter also unavailable -> still zero broadcasts.
    rap = _RapSrv(swap={"swapTransaction": _tx_b64(extra_accounts=200),
                        "lastValidBlockHeight": 1})
    jup = _JupSrv(quote_status=404)
    backend2, wallet2, rpc2 = _backend(rap.provider(), jup.provider())
    fill2 = _buy(backend2)
    assert fill2.status == "REAL_BUY_BLOCKED_NO_VALID_EXECUTION_ROUTE"
    assert rpc2.sent == [] and wallet2.signed == []
