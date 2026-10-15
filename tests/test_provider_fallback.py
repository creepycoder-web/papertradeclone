"""Provider-fallback tests: Raptor primary, Jupiter fallback.

All network is mocked. No test reaches a real endpoint, RPC or wallet, and none
requires operator secrets. The tests pin the critical properties:

  * Jupiter is ONLY invoked when Raptor is unsupported or oversized.
  * Both providers run the SAME size gate, freshness, simulation, signing,
    broadcast and reconciliation pipeline.
  * Exactly one intent is reserved and at most one broadcast happens per trade.
  * Both failing => REAL_BUY_BLOCKED=NO_VALID_EXECUTION_ROUTE, no broadcast.
"""


from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import httpx
import pytest

from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.transaction import VersionedTransaction

from autopaper.execution.intents import ExecutionIntentStore
from autopaper.execution.jupiter import JupiterProvider
from autopaper.execution.raptor import RaptorProvider
from autopaper.execution.real import LiveSolanaBackend
from autopaper.execution.solana import (
    MAX_SOLANA_TX_BYTES,
    SimulationOutcome,
    WSOL_MINT,
)

pytestmark = [
    pytest.mark.execution,
    pytest.mark.safety,
]

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


def _oversized_tx_b64() -> str:
    """Oversized as v0 (over 1232) but rescuable: a v1 re-encoding fits 4096."""
    return _tx_b64(extra_accounts=60)


def _unrescuable_tx_b64() -> str:
    """Oversized even after a faithful v1 re-encoding.

    v1 materialises every address lookup table entry, so a message with enough
    distinct accounts exceeds 4096 no matter how it is re-encoded. 200 account
    keys is past both ceilings (v0 stops at 1232, v1 at 4096), which is the
    honest "this route cannot be sent" case rather than the "this route is a
    bit too big" one.
    """
    return _tx_b64(extra_accounts=200)


# ---------------------------------------------------------------------------
# Raptor mock
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


def _raptor_unsupported() -> _RapSrv:
    return _RapSrv(quote_status=500,
                   quote_text=json.dumps({"error": "No multi-hop route found"}))


def _raptor_network_error() -> RaptorProvider:
    def boom(request):
        raise httpx.ConnectError("down", request=request)
    return RaptorProvider(
        client=httpx.AsyncClient(transport=httpx.MockTransport(boom)))


# ---------------------------------------------------------------------------
# Jupiter mock
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
    def __init__(self, *, quote_status=200, quote_text=None, swap_status=200,
                 swap=None):
        self.quote_status = quote_status
        self.quote_text = quote_text
        self.swap_status = swap_status
        self.swap = swap or {"swapTransaction": _tx_b64(),
                             "lastValidBlockHeight": 100}
        self.calls: list[tuple[str, object]] = []

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


def _jup_unsupported() -> _JupSrv:
    return _JupSrv(quote_status=404,
                   quote_text=json.dumps({"error": "no route"}))


def _jup_oversized() -> _JupSrv:
    return _JupSrv(swap={"swapTransaction": _oversized_tx_b64(),
                         "lastValidBlockHeight": 100})


def _jup_not_tradable() -> _JupSrv:
    return _JupSrv(quote_status=400,
                   quote_text=json.dumps({"error": "The token X is not "
                                                   "tradable",
                                          "errorCode": "TOKEN_NOT_TRADABLE"}))


# ---------------------------------------------------------------------------
# executor doubles
# ---------------------------------------------------------------------------
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


class _Wallet:
    def __init__(self):
        self.pubkey = "SoWaLanaPubKey1111111111111111111111111111111"
        self.masked = "(masked)"
        self.signed: list[bytes] = []

    def sign_wire(self, raw: bytes) -> bytes:
        self.signed.append(raw)
        return raw


class _Rpc:
    def __init__(self, *, sim_ok=True):
        self.sim_ok = sim_ok
        self.sim_calls = 0
        self.sent: list[str] = []

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

    async def get_address_lookup_table(self, address):
        """The rescue reads lookup tables. These fixtures use none."""
        return {"table": address, "query_ok": True, "usable": False,
                "addresses": [], "source": "double",
                "detail": "fixture has no lookup table"}

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


def _backend(primary, fallback, *, rpc=None, wallet=None, cfg=None,
             intent_store=None):
    wallet = wallet or _Wallet()
    rpc = rpc or _Rpc()
    cfg = cfg or _Cfg()
    b = LiveSolanaBackend(cfg, wallet=wallet, rpc=rpc, provider=primary,
                          fallback_provider=fallback)
    return b, wallet, rpc


def _run(coro):
    return asyncio.run(coro)


def _buy(backend, **kw):
    return _run(backend.execute_swap("ENTRY", WSOL_MINT, MINT, 1_071_843, 500,
                                     **kw))


# ---------------------------------------------------------------------------
# 1. Raptor success
# ---------------------------------------------------------------------------
def test_1_raptor_success_no_jupiter_call():
    rap, jup = _RapSrv(), _JupSrv()
    backend, wallet, rpc = _backend(rap.provider(), jup.provider())
    fill = _buy(backend)
    assert fill.status == "confirmed"
    assert fill.route_metadata["provider_name"] == "raptor"
    assert jup.calls == [], "Jupiter must not be called when Raptor succeeds"
    assert len(rpc.sent) == 1 and len(wallet.signed) == 1


# ---------------------------------------------------------------------------
# 2. Raptor unsupported -> Jupiter success
# ---------------------------------------------------------------------------
def test_2_raptor_unsupported_falls_back_to_jupiter():
    rap, jup = _raptor_unsupported(), _JupSrv()
    backend, wallet, rpc = _backend(rap.provider(), jup.provider())
    fill = _buy(backend)
    assert fill.status == "confirmed"
    assert fill.route_metadata["provider_name"] == "jupiter"
    assert any(p.endswith("/quote") for p, _ in jup.calls)
    assert len(rpc.sent) == 1, "exactly one broadcast"
    attempts = fill.route_metadata["attempts"]
    assert [a["status"] for a in attempts
            if a["provider"] == "raptor"] == ["RAPTOR_UNSUPPORTED_ROUTE"]


# ---------------------------------------------------------------------------
# 3. Raptor unsupported -> Jupiter oversized -> blocked
# ---------------------------------------------------------------------------
def test_3_raptor_unsupported_jupiter_oversized_is_rescued_as_v1():
    """An oversized v0 is re-encoded as v1, not blocked.

    This is the change that turns the incident from "unsendable" into
    "slightly too big for v0". The rebuilt transaction is only used because the
    executor proved it equivalent; the broadcast is still exactly one, and the
    provider still never signs.
    """
    rap, jup = _raptor_unsupported(), _jup_oversized()
    backend, wallet, rpc = _backend(rap.provider(), jup.provider())
    fill = _buy(backend)
    assert fill.status == "confirmed"
    assert fill.route_metadata["provider_name"] == "jupiter"
    assert fill.route_metadata["transaction_version"] == "v1"
    assert fill.route_metadata["raw_bytes"] <= 4096
    assert len(rpc.sent) == 1


def test_3b_oversized_and_unrescuable_is_still_blocked():
    """The rescue is not a way to send something that genuinely cannot be sent.

    A transaction that is over 4096 even after a faithful v1 re-encoding is
    refused, with nothing signed and nothing broadcast.
    """
    rap = _raptor_unsupported()
    jup = _JupSrv(swap={"swapTransaction": _unrescuable_tx_b64(),
                        "lastValidBlockHeight": 1})
    backend, wallet, rpc = _backend(rap.provider(), jup.provider())
    fill = _buy(backend)
    assert fill.status == "REAL_BUY_BLOCKED_NO_VALID_EXECUTION_ROUTE"
    assert not fill.filled
    assert wallet.signed == [] and rpc.sent == []


# ---------------------------------------------------------------------------
# 4. Raptor network error -> NO fallback
# ---------------------------------------------------------------------------
def test_4_raptor_network_error_does_not_fall_back():
    jup = _JupSrv()
    backend, wallet, rpc = _backend(_raptor_network_error(), jup.provider())
    fill = _buy(backend)
    assert fill.status == "RAPTOR_NETWORK_ERROR"
    assert jup.calls == [], "a provider outage must not trigger the fallback"
    assert rpc.sent == []


# ---------------------------------------------------------------------------
# 5. Jupiter build failure (after Raptor unsupported) -> both failed
# ---------------------------------------------------------------------------
def test_5_jupiter_build_failure_after_fallback():
    rap = _raptor_unsupported()
    jup = _JupSrv(swap_status=500,
                  swap=json.dumps({"error": "build failed"}))
    backend, wallet, rpc = _backend(rap.provider(), jup.provider())
    fill = _buy(backend)
    assert fill.status == "REAL_BUY_BLOCKED_NO_VALID_EXECUTION_ROUTE"
    assert rpc.sent == [] and wallet.signed == []


# ---------------------------------------------------------------------------
# 6. both unsupported
# ---------------------------------------------------------------------------
def test_6_both_unsupported():
    backend, wallet, rpc = _backend(_raptor_unsupported().provider(),
                                    _jup_unsupported().provider())
    fill = _buy(backend)
    assert fill.status == "REAL_BUY_BLOCKED_NO_VALID_EXECUTION_ROUTE"
    assert rpc.sent == []


# ---------------------------------------------------------------------------
# 7. both oversized
# ---------------------------------------------------------------------------
def test_7_both_oversized_and_unrescuable_is_blocked():
    rap = _RapSrv(swap={"swapTransaction": _unrescuable_tx_b64(),
                        "lastValidBlockHeight": 1})
    jup = _JupSrv(swap={"swapTransaction": _unrescuable_tx_b64(),
                        "lastValidBlockHeight": 1})
    backend, wallet, rpc = _backend(rap.provider(), jup.provider())
    fill = _buy(backend)
    assert fill.status == "REAL_BUY_BLOCKED_NO_VALID_EXECUTION_ROUTE"
    assert rpc.sent == [] and wallet.signed == []


def test_7b_oversized_primary_is_rescued_without_reaching_the_fallback():
    """Raptor stays primary: an oversized v0 is rescued in place, not abandoned.

    The rescue runs before the fallback, so a Raptor transaction that only
    needed a version bump never causes a second provider call. This matters for
    latency and for the provider's own rate limits, and it is what "Raptor
    remains primary" has to mean once V1 exists.
    """
    rap = _RapSrv(swap={"swapTransaction": _oversized_tx_b64(),
                        "lastValidBlockHeight": 1})
    jup = _JupSrv()
    backend, wallet, rpc = _backend(rap.provider(), jup.provider())
    fill = _buy(backend)
    assert fill.status == "confirmed"
    assert fill.route_metadata["provider_name"] == "raptor"
    assert fill.route_metadata["transaction_version"] == "v1"
    assert jup.calls == [], "a rescued primary must not trigger the fallback"
    assert len(rpc.sent) == 1


# ---------------------------------------------------------------------------
# 8. simulation failure on Raptor
# ---------------------------------------------------------------------------
def test_8_raptor_simulation_failure_no_fallback_no_broadcast():
    jup = _JupSrv()
    backend, wallet, rpc = _backend(_RapSrv().provider(), jup.provider(),
                                    rpc=_Rpc(sim_ok=False))
    fill = _buy(backend)
    assert fill.status == "RAPTOR_SIMULATION_FAILED"
    assert fill.route_metadata["provider_name"] == "raptor"
    assert jup.calls == [], "simulation failure is not a fallback trigger"
    assert rpc.sent == [] and wallet.signed == []


# ---------------------------------------------------------------------------
# 9. simulation failure on Jupiter (after Raptor unsupported)
# ---------------------------------------------------------------------------
def test_9_jupiter_simulation_failure_no_broadcast():
    backend, wallet, rpc = _backend(_raptor_unsupported().provider(),
                                    _JupSrv().provider(), rpc=_Rpc(sim_ok=False))
    fill = _buy(backend)
    assert fill.status == "JUPITER_SIMULATION_FAILED"
    assert rpc.sent == [] and wallet.signed == []


# ---------------------------------------------------------------------------
# 10. fallback does not duplicate the intent
# ---------------------------------------------------------------------------
def test_10_fallback_reserves_intent_exactly_once(tmp_path: Path):
    store = ExecutionIntentStore(tmp_path / "intents.jsonl")
    backend, wallet, rpc = _backend(_raptor_unsupported().provider(),
                                    _JupSrv().provider())
    fill = _buy(backend, intent_store=store, logical_key=f"ENTRY:{MINT}")
    assert fill.status == "confirmed"
    intents = [i for i in store.all_intents() if i.logical_key == f"ENTRY:{MINT}"]
    assert len(intents) == 1, "one trade intent, one reservation"
    assert intents[0].state == "CONFIRMED"
    assert intents[0].signature == "5Sig"
    assert len(rpc.sent) == 1


# ---------------------------------------------------------------------------
# 11. no broadcast after the failed first provider (Raptor oversized -> Jupiter)
# ---------------------------------------------------------------------------
def test_11_unrescuable_raptor_oversized_falls_back_and_broadcasts_once():
    """When the primary CANNOT be rescued, the fallback still gets its turn.

    The ONE broadcast is Jupiter's transaction. Raptor's unrescuable bytes are
    never signed and never sent.
    """
    rap = _RapSrv(swap={"swapTransaction": _unrescuable_tx_b64(),
                        "lastValidBlockHeight": 1})
    jup = _JupSrv()
    jup_tx = jup.swap["swapTransaction"]
    backend, wallet, rpc = _backend(rap.provider(), jup.provider())
    fill = _buy(backend)
    assert fill.status == "confirmed"
    assert fill.route_metadata["provider_name"] == "jupiter"
    assert len(rpc.sent) == 1
    assert rpc.sent[0] == jup_tx
    assert len(wallet.signed) == 1


# ---------------------------------------------------------------------------
# 12. per-provider transaction sizes recorded
# ---------------------------------------------------------------------------
def test_12_provider_sizes_recorded():
    """Every size_gate attempt records the version and the limit that applied.

    The recorded limit is the one for the transaction's ACTUAL version, so a
    reader can tell "over the v0 limit but under the v1 limit" from "over every
    limit" without re-deriving anything.
    """
    rap = _RapSrv(swap={"swapTransaction": _unrescuable_tx_b64(),
                        "lastValidBlockHeight": 1})
    jup = _JupSrv()
    backend, wallet, rpc = _backend(rap.provider(), jup.provider())
    fill = _buy(backend)
    md = fill.route_metadata
    assert md["provider_chain"] == ["raptor", "jupiter"]
    sizes = {(a["provider"], a["stage"]): a
             for a in md["attempts"] if "raw_bytes" in a}
    rap_size = sizes[("raptor", "size_gate")]
    jup_size = sizes[("jupiter", "size_gate")]
    assert rap_size["transaction_version"] == "v0"
    assert rap_size["raw_bytes"] > rap_size["max_bytes"]
    assert rap_size["over_by"] == rap_size["raw_bytes"] - rap_size["max_bytes"]
    assert rap_size["v1_rescue"]["attempted"] is True
    assert rap_size["v1_rescue"]["ok"] is False
    assert jup_size["transaction_version"] == "v0"
    assert jup_size["raw_bytes"] <= jup_size["max_bytes"]
    assert md["provider"] == "JUPITER"
    assert md["raw_bytes"] == jup_size["raw_bytes"]


# ---------------------------------------------------------------------------
# 13. a Jupiter "not tradable" 400 is an UNSUPPORTED ROUTE, not a micro refusal
# ---------------------------------------------------------------------------
def test_13_jupiter_not_tradable_is_unsupported_route():
    backend, wallet, rpc = _backend(_raptor_unsupported().provider(),
                                    _jup_not_tradable().provider())
    fill = _buy(backend)
    assert fill.status == "REAL_BUY_BLOCKED_NO_VALID_EXECUTION_ROUTE"
    statuses = [a["status"] for a in fill.route_metadata["attempts"]]
    assert "JUPITER_UNSUPPORTED_ROUTE" in statuses
    assert rpc.sent == [] and wallet.signed == []
