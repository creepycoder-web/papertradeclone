"""Raptor execution-migration tests.

Every network call is MOCKED. No test reaches the real Raptor endpoint, a real
RPC, or real funds, and none requires operator secrets. The tests assert the
security contract explicitly: the bot never sends Raptor a private key or a
signed transaction, and the bot keeps signing, simulation, broadcast and
idempotency control.
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

from autopaper.execution.intents import (
    ExecutionIntentStore,
    UNKNOWN as INTENT_UNKNOWN,
)
from autopaper.execution.raptor import (
    RaptorInvalidResponse,
    RaptorMicroOrderUnsupported,
    RaptorNetworkError,
    RaptorProvider,
    RaptorQuoteFailed,
    RaptorUnsupportedRoute,
    transaction_size_report,
)
from autopaper.execution.jupiter import QuoteRejected, QuoteUnavailable
from autopaper.execution.real import LiveSolanaBackend
from autopaper.execution.solana import (
    MAX_SOLANA_TX_BYTES,
    SimulationOutcome,
    WSOL_MINT,
)

pytestmark = pytest.mark.execution

MINT = "H2LWQMQdEGamrEQh8v4jiwBhW3hh1iqcDwU815xJ881t"


# ---------------------------------------------------------------------------
# transaction fixtures (real serialized bytes)
# ---------------------------------------------------------------------------
def _payer() -> Pubkey:
    return Keypair().pubkey()


def _unsigned_tx_b64(extra_accounts: int = 0) -> str:
    metas = []
    for _ in range(extra_accounts):
        metas.append(AccountMeta(Keypair().pubkey(), False, True))
    ix = Instruction(Pubkey.default(), b"\x00\x01\x02", metas)
    msg = MessageV0.try_compile(_payer(), [ix], [], Hash.default())
    tx = VersionedTransaction.populate(msg, [Signature.default()])
    return base64.b64encode(bytes(tx)).decode()


def _oversized_tx_b64(extra_accounts: int = 60) -> str:
    """Over the 1232-byte v0 ceiling.

    60 extra account metas add ~33 raw bytes each, so 60 lands comfortably past
    the limit while a v1 re-encoding of the same transaction still fits 4096.
    200 is the opposite case: over the v1 ceiling too, so no re-encoding helps.
    """
    return _unsigned_tx_b64(extra_accounts=extra_accounts)


def _signed_tx_b64() -> str:
    msg = MessageV0.try_compile(_payer(), [], [], Hash.default())
    tx = VersionedTransaction.populate(
        msg, [Signature.from_bytes(bytes([1] * 64))])
    return base64.b64encode(bytes(tx)).decode()


# ---------------------------------------------------------------------------
# mock transport
# ---------------------------------------------------------------------------
class _Srv:
    """A scripted Raptor HTTP server that records every request body."""

    def __init__(self, *, quote=None, swap=None, quote_status=200,
                 swap_status=200, quote_text=None, swap_text=None):
        self.quote = quote if quote is not None else _quote_json()
        self.swap = swap if swap is not None else _swap_json()
        self.quote_status = quote_status
        self.swap_status = swap_status
        self.quote_text = quote_text
        self.swap_text = swap_text
        self.requests: list[tuple[str, object]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = None
        if request.content:
            try:
                body = json.loads(request.content.decode())
            except Exception:  # noqa: BLE001
                body = request.content.decode(errors="replace")
        self.requests.append((path, body))
        if path.endswith("/quote"):
            if self.quote_text is not None:
                return httpx.Response(self.quote_status, text=self.quote_text)
            return httpx.Response(self.quote_status, json=self.quote)
        if path.endswith("/swap"):
            if self.swap_text is not None:
                return httpx.Response(self.swap_status, text=self.swap_text)
            return httpx.Response(self.swap_status, json=self.swap)
        return httpx.Response(404, json={"error": "not found"})

    def provider(self, **kw) -> RaptorProvider:
        client = httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
        return RaptorProvider(client=client, **kw)


def _quote_json(**over) -> dict:
    base = {
        "inputMint": WSOL_MINT,
        "outputMint": MINT,
        "amountIn": "1071843",
        "amountOut": "3280257",
        "minAmountOut": "3116000",
        "feeAmount": "2403",
        "priceImpact": 0.0123,
        "slippageBps": 500,
        "routePlan": [
            {"programId": "P1", "dex": "Pump.fun", "pool": "pool1",
             "inputMint": WSOL_MINT, "outputMint": MINT,
             "amountIn": "1071843", "amountOut": "3280257",
             "feeAmount": "2403", "priceImpact": 0.0123, "percent": 100},
        ],
        "contextSlot": 452176241,
        "timeTaken": 0.0012,
        "swapUsdValue": "0.186",
    }
    base.update(over)
    return base


def _swap_json(**over) -> dict:
    base = {
        "swapTransaction": _unsigned_tx_b64(),
        "lastValidBlockHeight": 123456,
        "contextSlot": 452176241,
        "prioritizationFeeLamports": 50000,
    }
    base.update(over)
    return base


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# client: quote parsing
# ---------------------------------------------------------------------------
def test_quote_parses_and_converts_price_impact():
    srv = _Srv()
    q = _run(srv.provider().quote(WSOL_MINT, MINT, 1_071_843, 500))
    assert q.expected_out_amount == 3_280_257
    assert q.min_out_amount == 3_116_000
    # 0.0123 fraction -> 1.23%
    assert q.price_impact_pct == pytest.approx(1.23)
    assert q.slippage_bps == 500
    assert q.route_labels == ["Pump.fun"]
    assert q.route_steps[0].dex == "Pump.fun"
    assert q.selected_dex == "Pump.fun"
    assert q.context_slot == 452176241
    # the full quote object is preserved for the build call
    assert q.raw["amountOut"] == "3280257"


def test_quote_rejects_missing_amount_out():
    srv = _Srv(quote=_quote_json(), quote_text=json.dumps({"amountIn": "1"}))
    with pytest.raises(RaptorInvalidResponse):
        _run(srv.provider().quote(WSOL_MINT, MINT, 1_071_843, 500))


def test_quote_unsupported_route_is_named():
    srv = _Srv(quote_status=500,
               quote_text=json.dumps({"error": "Failed to get quote: "
                                               "No multi-hop route found"}))
    with pytest.raises(RaptorUnsupportedRoute) as ei:
        _run(srv.provider().quote(WSOL_MINT, MINT, 1_071_843, 500))
    assert ei.value.status == "RAPTOR_UNSUPPORTED_ROUTE"
    assert ei.value.http_status == 500


def test_quote_invalid_mint_is_unsupported_route():
    srv = _Srv(quote_status=400,
               quote_text=json.dumps({"error": "Invalid output mint",
                                      "code": 400}))
    with pytest.raises(RaptorUnsupportedRoute):
        _run(srv.provider().quote(WSOL_MINT, MINT, 1_071_843, 500))


def test_quote_micro_notional_is_the_micro_condition():
    srv = _Srv(quote_status=400,
               quote_text=json.dumps({"error": "amount too small"}))
    with pytest.raises(RaptorMicroOrderUnsupported) as ei:
        _run(srv.provider().quote(WSOL_MINT, MINT, 155_000, 500))
    assert ei.value.status == "MICRO_REAL_EXECUTION_UNSUPPORTED"


def test_quote_generic_server_error_is_quote_failed():
    srv = _Srv(quote_status=500, quote_text=json.dumps({"error": "boom"}))
    with pytest.raises(RaptorQuoteFailed) as ei:
        _run(srv.provider().quote(WSOL_MINT, MINT, 1_071_843, 500))
    assert ei.value.status == "RAPTOR_QUOTE_FAILED"


def test_quote_price_impact_over_limit_refused():
    srv = _Srv(quote=_quote_json(priceImpact=0.20))  # 20%
    with pytest.raises(RaptorQuoteFailed, match="price impact"):
        _run(srv.provider(max_price_impact_pct=5.0).quote(
            WSOL_MINT, MINT, 1_071_843, 500))


def test_quote_min_out_not_below_expected_refused():
    srv = _Srv(quote=_quote_json(minAmountOut="3280257"))
    with pytest.raises(RaptorQuoteFailed, match="minimum output"):
        _run(srv.provider().quote(WSOL_MINT, MINT, 1_071_843, 500))


def test_network_error_is_classified():
    def boom(request):
        raise httpx.ConnectError("no route to host", request=request)
    client = httpx.AsyncClient(transport=httpx.MockTransport(boom))
    prov = RaptorProvider(client=client)
    with pytest.raises(RaptorNetworkError) as ei:
        _run(prov.quote(WSOL_MINT, MINT, 1_071_843, 500))
    assert ei.value.status == "RAPTOR_NETWORK_ERROR"


# ---------------------------------------------------------------------------
# client: build parsing + security contract
# ---------------------------------------------------------------------------
def test_build_parses_unsigned_transaction_and_route_metadata():
    srv = _Srv()
    q = _run(srv.provider().quote(WSOL_MINT, MINT, 1_071_843, 500))
    built = _run(srv.provider().build(
        q, "SoWaLanaPubKey1111111111111111111111111111111",
        priority_fee=50_000))
    assert built.unsigned is True
    assert built.signature_slots == 1
    assert built.last_valid_block_height == 123456
    assert built.prioritization_fee_lamports == 50000
    assert built.transaction_b64 == srv.swap["swapTransaction"]


def test_build_missing_transaction_is_invalid_response():
    srv = _Srv(swap={"lastValidBlockHeight": 1})
    q = _run(srv.provider().quote(WSOL_MINT, MINT, 1_071_843, 500))
    with pytest.raises(RaptorInvalidResponse):
        _run(srv.provider().build(q, "payer", priority_fee=50_000))


def test_build_undecodable_transaction_is_invalid_response():
    srv = _Srv(swap=_swap_json(swapTransaction="!!!not-base64!!!"))
    q = _run(srv.provider().quote(WSOL_MINT, MINT, 1_071_843, 500))
    with pytest.raises(RaptorInvalidResponse):
        _run(srv.provider().build(q, "payer", priority_fee=50_000))


def test_build_refuses_a_signed_transaction():
    """The venue must never return anything already signed."""
    srv = _Srv(swap=_swap_json(swapTransaction=_signed_tx_b64()))
    q = _run(srv.provider().quote(WSOL_MINT, MINT, 1_071_843, 500))
    with pytest.raises(RaptorInvalidResponse, match="signed transaction"):
        _run(srv.provider().build(q, "payer", priority_fee=50_000))


def test_build_sends_only_the_public_key_and_never_a_secret():
    srv = _Srv()
    pub = "SoWaLanaPubKey1111111111111111111111111111111"
    q = _run(srv.provider().quote(WSOL_MINT, MINT, 1_071_843, 500))
    _run(srv.provider().build(q, pub, priority_fee=50_000))
    _, body = [r for r in srv.requests if r[0].endswith("/swap")][0]
    assert body["userPublicKey"] == pub
    assert set(body) == {"userPublicKey", "quoteResponse", "wrapUnwrapSol",
                         "txVersion", "priorityFee"}
    blob = json.dumps(body).lower()
    for forbidden in ("private", "secret", "seed", "mnemonic",
                      "signedtransaction", "signature"):
        assert forbidden not in blob, f"request leaked {forbidden!r}"


def test_build_normalises_priority_fee_to_a_string():
    srv = _Srv()
    q = _run(srv.provider().quote(WSOL_MINT, MINT, 1_071_843, 500))
    _run(srv.provider().build(q, "payer", priority_fee=50_000))
    _, body = [r for r in srv.requests if r[0].endswith("/swap")][0]
    assert body["priorityFee"] == "50000"
    # a level name is passed through unchanged
    srv2 = _Srv()
    q2 = _run(srv2.provider().quote(WSOL_MINT, MINT, 1_071_843, 500))
    _run(srv2.provider().build(q2, "payer", priority_fee="medium"))
    _, body2 = [r for r in srv2.requests if r[0].endswith("/swap")][0]
    assert body2["priorityFee"] == "medium"


def test_build_never_calls_send_transaction():
    srv = _Srv()
    q = _run(srv.provider().quote(WSOL_MINT, MINT, 1_071_843, 500))
    _run(srv.provider().build(q, "payer", priority_fee=50_000))
    assert not any(p.endswith("/send-transaction") for p, _ in srv.requests)


# ---------------------------------------------------------------------------
# transaction size gate (client helper)
# ---------------------------------------------------------------------------
def test_size_report_fits_and_oversized():
    ok = transaction_size_report(_unsigned_tx_b64())
    assert ok["ok"] is True and ok["raw_bytes"] <= MAX_SOLANA_TX_BYTES
    big = transaction_size_report(_oversized_tx_b64())
    assert big["ok"] is False
    assert big["status"] == "TX_TOO_LARGE"
    assert big["raw_bytes"] > MAX_SOLANA_TX_BYTES
    assert big["over_by"] == big["raw_bytes"] - MAX_SOLANA_TX_BYTES


def test_size_report_malformed():
    """A payload that is not a transaction is refused, whatever it decodes to.

    Two distinct malformations, both refused, and the reason kept separate so an
    operator can tell "the provider sent us garbage that is not even base64"
    from "the provider sent us base64 that is not a transaction".
    """
    not_base64 = transaction_size_report("!!!")
    assert not_base64["ok"] is False
    assert not_base64["status"] == "TX_MALFORMED_ENCODING"

    import base64 as _b64
    # 0xff as a signature count is not a representable shortvec, so no
    # deserializer can accept this and the version is genuinely unknowable.
    not_a_transaction = transaction_size_report(
        _b64.b64encode(b"\xff" * 200).decode())
    assert not_a_transaction["ok"] is False
    assert not_a_transaction["status"] == "TX_MALFORMED"
    assert not_a_transaction["transaction_version"] is None
    assert not_a_transaction["max_bytes"] is None


def test_size_report_reads_the_version_from_the_bytes_not_the_provider():
    """The limit follows the transaction, not the caller's opinion of it.

    Two transactions differing only in version are judged against their own
    ceilings: a 1250-byte v0 is unsendable and a 1250-byte v1 is not. This is
    the whole point of the upgrade, and it is asserted with real transactions
    rather than by monkey-patching the version.
    """
    from tests import txfixtures
    from autopaper.execution.txcodec import (MAX_BYTES_LEGACY, MAX_BYTES_V1,
                                             measure_transaction)
    size = 1250
    v0 = measure_transaction(
        txfixtures.b64(txfixtures.build_exact("v0", size)))
    v1 = measure_transaction(
        txfixtures.b64(txfixtures.build_exact("v1", size)))
    assert v0["raw_bytes"] == v1["raw_bytes"] == size
    assert v0["transaction_version"] == "v0"
    assert v1["transaction_version"] == "v1"
    assert v0["max_bytes"] == MAX_BYTES_LEGACY == 1232
    assert v1["max_bytes"] == MAX_BYTES_V1 == 4096
    assert v0["ok"] is False and v0["over_by"] == size - 1232
    assert v1["ok"] is True and v1["over_by"] == 0


# ---------------------------------------------------------------------------
# executor integration
# ---------------------------------------------------------------------------
class _Cfg:
    def __init__(self, **kw):
        self.wallet_keypair_path = kw.get("wallet_keypair_path")
        self.rpc_url = kw.get("rpc_url", "https://rpc.invalid")
        self.slippage_bps = 500
        self.simulate_before_send = True
        self.priority_fee_lamports = 50_000
        self.confirmation_timeout_s = 5.0
        self.quote_max_age_ms = kw.get("quote_max_age_ms", 8000.0)
        self.max_price_impact_pct = 5.0
        self.raptor_enabled = kw.get("raptor_enabled", True)
        self.raptor_base_url = kw.get("raptor_base_url",
                                      "https://raptor.solanatracker.io")
        self.raptor_fallback_enabled = kw.get("raptor_fallback_enabled", True)
        self.v1_rescue_enabled = kw.get("v1_rescue_enabled", True)


class _Wallet:
    def __init__(self):
        self.pubkey = "SoWaLanaPubKey1111111111111111111111111111111"
        self.masked = "(masked)"
        self.signed: list[bytes] = []

    def sign_wire(self, raw: bytes) -> bytes:
        self.signed.append(raw)
        return raw


class _Rpc:
    def __init__(self, *, sim_ok=True, sim_detail="simulation ok",
                 send_exc=None, confirm=None, get_tx=None):
        self.sim_ok = sim_ok
        self.sim_detail = sim_detail
        self.send_exc = send_exc
        self.sim_calls = 0
        self.sent: list[str] = []
        self.confirm = confirm or {"status": "confirmed", "error": None,
                                   "signature": "sig", "slot": 7}
        self.get_tx = get_tx

    async def simulate_transaction_detailed(self, tx_b64):
        self.sim_calls += 1
        return SimulationOutcome(
            ok=self.sim_ok,
            status="SIMULATION_OK" if self.sim_ok else "SIMULATION_FAILED_TX_REJECTED",
            detail=self.sim_detail, stage="SIMULATION", rpc_code=None,
            rpc_message="", broadcast_attempted=False)

    async def send_transaction(self, tx_b64):
        if self.send_exc:
            raise self.send_exc
        self.sent.append(tx_b64)
        return "5XyzSignature"

    async def wait_for_confirmation(self, sig, **kw):
        return dict(self.confirm)

    async def get_transaction(self, sig, commitment="confirmed"):
        return self.get_tx


def _chain_tx(owner: str, in_lamports: int, out_tokens: int, fee: int = 5000):
    return {
        "slot": 7,
        "transaction": {
            "message": {"accountKeys": [owner]},
            "meta": {
                "err": None, "fee": fee,
                "logMessages": ["Program log: swap"],
                "preBalances": [in_lamports],
                "postBalances": [0],
                "preTokenBalances": [],
                "postTokenBalances": [{"mint": MINT, "owner": owner,
                                       "uiTokenAmount": {"amount": str(out_tokens), "decimals": 0}}],
            },
        },
    }


class _FallbackUnsupported:
    """A Jupiter-shaped fallback double that cannot route. No network."""

    def provider_name(self) -> str:
        return "jupiter"

    async def quote(self, *a, **k):
        raise QuoteRejected("no route")

    async def build_transaction(self, *a, **k):
        raise QuoteUnavailable("not reachable")

    def route_metadata(self, quote):
        return {"provider": "jupiter"}


def _backend(srv: _Srv, *, rpc=None, wallet=None, cfg=None, fallback=None):
    wallet = wallet or _Wallet()
    rpc = rpc or _Rpc(get_tx=_chain_tx(wallet.pubkey, 1_071_843, 3_280_257))
    cfg = cfg or _Cfg()
    if fallback is None and getattr(cfg, "raptor_fallback_enabled", True):
        fallback = _FallbackUnsupported()
    return LiveSolanaBackend(cfg, wallet=wallet, rpc=rpc,
                             provider=srv.provider(),
                             fallback_provider=fallback), wallet, rpc


def test_executor_raptor_path_end_to_end():
    srv = _Srv()
    backend, wallet, rpc = _backend(srv)
    fill = _run(backend.execute_swap("ENTRY", WSOL_MINT, MINT, 1_071_843, 500))
    assert fill.status == "confirmed"
    assert fill.filled
    assert fill.signature == "5XyzSignature"
    assert fill.route_metadata["provider"] == "RAPTOR"
    assert fill.route_metadata["raw_bytes"] <= MAX_SOLANA_TX_BYTES
    assert fill.route_metadata["requested_priority_fee_lamports"] == 50_000
    assert fill.route_metadata["effective_priority_fee_lamports"] == 50_000
    assert fill.route_metadata["requested_slippage_bps"] == 500
    assert fill.route_metadata["effective_slippage_bps"] == 500
    assert wallet.signed, "the bot signed locally"
    assert len(rpc.sent) == 1, "the bot broadcast exactly once"
    assert rpc.sim_calls == 1
    paths = [p for p, _ in srv.requests]
    assert any(p.endswith("/quote") for p in paths)
    assert any(p.endswith("/swap") for p in paths)
    assert not any(p.endswith("/send-transaction") for p in paths)


def test_executor_rescues_oversized_v0_as_v1_before_simulation():
    """An oversized v0 is re-encoded as v1 and then simulated normally.

    The rescue is a RE-ENCODING: same instructions, same accounts, same payer,
    same blockhash, and the equivalence is proved before the transaction is used.
    What follows is the ordinary pipeline -- freshness, simulation, signing,
    broadcast -- with nothing relaxed.
    """
    srv = _Srv(swap=_swap_json(swapTransaction=_oversized_tx_b64()))
    backend, wallet, rpc = _backend(
        srv, cfg=_Cfg(raptor_fallback_enabled=False))
    fill = _run(backend.execute_swap("ENTRY", WSOL_MINT, MINT, 1_071_843, 500))
    assert fill.status == "confirmed"
    assert fill.route_metadata["transaction_version"] == "v1"
    assert fill.route_metadata["raw_bytes"] <= 4096
    assert fill.route_metadata["max_bytes"] == 4096
    assert rpc.sim_calls >= 1, "the rebuilt transaction is simulated, not waved through"
    assert len(wallet.signed) == 1 and len(rpc.sent) == 1


def test_executor_blocks_unrescuable_oversized_before_simulation():
    """Over its OWN v1 ceiling, it is still refused before simulation.

    The rescue is not a loophole: a transaction that is too large even as v1 is
    rejected locally, with no simulation, no signature and no broadcast.
    """
    srv = _Srv(swap=_swap_json(swapTransaction=_oversized_tx_b64(200)))
    backend, wallet, rpc = _backend(
        srv, cfg=_Cfg(raptor_fallback_enabled=False))
    fill = _run(backend.execute_swap("ENTRY", WSOL_MINT, MINT, 1_071_843, 500))
    assert fill.status == "RAPTOR_TX_TOO_LARGE"
    assert not fill.filled
    assert "REAL_BUY_BLOCKED=TX_TOO_LARGE" in fill.ambiguity_reason
    assert "provider=RAPTOR" in fill.ambiguity_reason
    assert "transaction_version=v0" in fill.ambiguity_reason
    assert rpc.sim_calls == 0, "oversized is refused before simulation"
    assert wallet.signed == [], "nothing signed"
    assert rpc.sent == [], "nothing broadcast"


def test_v1_rescue_can_be_switched_off():
    """The rescue is a feature flag, and OFF means the old refusal.

    So an operator who wants the previous, stricter behaviour gets exactly it,
    and can compare the two paths on the same alerts.
    """
    srv = _Srv(swap=_swap_json(swapTransaction=_oversized_tx_b64()))
    backend, wallet, rpc = _backend(
        srv, cfg=_Cfg(raptor_fallback_enabled=False, v1_rescue_enabled=False))
    fill = _run(backend.execute_swap("ENTRY", WSOL_MINT, MINT, 1_071_843, 500))
    assert fill.status == "RAPTOR_TX_TOO_LARGE"
    assert rpc.sim_calls == 0
    assert wallet.signed == [] and rpc.sent == []


def test_executor_stale_quote_refused_before_signing():
    import time
    srv = _Srv()
    backend, wallet, rpc = _backend(srv, cfg=_Cfg(quote_max_age_ms=1.0))
    prov = backend.provider
    origin = prov.quote

    async def stale(*a, **k):
        q = await origin(*a, **k)
        q.quote_received_at = time.time() - 60.0
        return q

    prov.quote = stale  # type: ignore[assignment]
    fill = _run(backend.execute_swap("ENTRY", WSOL_MINT, MINT, 1_071_843, 500))
    assert fill.status == "RAPTOR_STALE_QUOTE"
    assert wallet.signed == []
    assert rpc.sent == []


def test_executor_simulation_failure_prevents_broadcast():
    rpc = _Rpc(sim_ok=False, sim_detail="simulation error: bad route")
    backend, wallet, rpc = _backend(_Srv(), rpc=rpc)
    fill = _run(backend.execute_swap("ENTRY", WSOL_MINT, MINT, 1_071_843, 500))
    assert not fill.filled
    assert fill.status == "RAPTOR_SIMULATION_FAILED"
    assert rpc.sent == []
    assert wallet.signed == [], "signing happens only after a clean simulation"


def test_executor_unroutable_token_fails_closed():
    srv = _Srv(quote_status=500,
               quote_text=json.dumps({"error": "No multi-hop route found"}))
    # Fallback disabled: this asserts the Raptor-only unsupported outcome.
    backend, wallet, rpc = _backend(
        srv, cfg=_Cfg(raptor_fallback_enabled=False))
    fill = _run(backend.execute_swap("ENTRY", WSOL_MINT, MINT, 1_071_843, 500))
    assert fill.status == "RAPTOR_UNSUPPORTED_ROUTE"
    assert not fill.filled
    assert not any(p.endswith("/swap") for p, _ in srv.requests)
    assert wallet.signed == [] and rpc.sent == []


def test_executor_build_failure_is_reported():
    srv = _Srv(swap_status=500, swap_text=json.dumps({"error": "internal"}))
    backend, wallet, rpc = _backend(srv)
    fill = _run(backend.execute_swap("ENTRY", WSOL_MINT, MINT, 1_071_843, 500))
    assert fill.status == "RAPTOR_BUILD_FAILED"
    assert wallet.signed == [] and rpc.sent == []


def test_executor_broadcast_failure_is_not_a_fill():
    from autopaper.execution.solana import RpcError
    rpc = _Rpc(send_exc=RpcError("node said no", code=-32002))
    backend, wallet, rpc = _backend(_Srv(), rpc=rpc)
    fill = _run(backend.execute_swap("ENTRY", WSOL_MINT, MINT, 1_071_843, 500))
    assert fill.status == "BROADCAST_REJECTED"
    assert not fill.filled


def test_executor_confirmation_timeout_is_ambiguous():
    rpc = _Rpc(confirm={"status": "timeout", "error": "expired",
                        "signature": "sig", "slot": None})
    backend, wallet, rpc = _backend(_Srv(), rpc=rpc)
    fill = _run(backend.execute_swap("ENTRY", WSOL_MINT, MINT, 1_071_843, 500))
    assert fill.status == "AMBIGUOUS"
    assert not fill.filled


def test_executor_duplicate_intent_is_refused(tmp_path: Path):
    store = ExecutionIntentStore(tmp_path / "intents.jsonl")
    store.reserve(logical_key=f"ENTRY:{MINT}", side="ENTRY", mint=MINT,
                  requested_amount=1_071_843)
    srv = _Srv()
    backend, wallet, rpc = _backend(srv)
    fill = _run(backend.execute_swap(
        "ENTRY", WSOL_MINT, MINT, 1_071_843, 500,
        intent_store=store, logical_key=f"ENTRY:{MINT}"))
    assert fill.status == "DUPLICATE_EXECUTION"
    assert rpc.sent == []
    assert not any(p.endswith("/quote") for p, _ in srv.requests)


def test_executor_restart_recovery_blocks_retry(tmp_path: Path):
    """A crashed earlier attempt leaves an UNKNOWN intent; a retry must refuse."""
    path = tmp_path / "intents.jsonl"
    store = ExecutionIntentStore(path)
    intent = store.reserve(logical_key=f"ENTRY:{MINT}", side="ENTRY", mint=MINT,
                           requested_amount=1_071_843)
    store.transition(intent.execution_id, "READY")
    store.transition(intent.execution_id, "SIGNING")
    store.transition(intent.execution_id, "BROADCASTED", signature="sig")
    store.transition(intent.execution_id, INTENT_UNKNOWN)

    # A fresh store over the same file (a restarted process).
    restarted = ExecutionIntentStore(path)
    assert restarted.has_unresolved(mint=MINT) is True

    srv = _Srv()
    backend, wallet, rpc = _backend(srv)
    fill = _run(backend.execute_swap(
        "ENTRY", WSOL_MINT, MINT, 1_071_843, 500,
        intent_store=restarted, logical_key=f"ENTRY:{MINT}"))
    assert fill.status == "DUPLICATE_EXECUTION"
    assert rpc.sent == []


def test_executor_intent_lifecycle_persisted(tmp_path: Path):
    store = ExecutionIntentStore(tmp_path / "intents.jsonl")
    backend, wallet, rpc = _backend(_Srv())
    fill = _run(backend.execute_swap(
        "ENTRY", WSOL_MINT, MINT, 1_071_843, 500,
        intent_store=store, logical_key=f"ENTRY:{MINT}"))
    assert fill.status == "confirmed"
    intent = store.find(f"ENTRY:{MINT}")
    assert intent is not None and intent.state == "CONFIRMED"
    assert intent.signature == "5XyzSignature"


def test_raptor_disabled_uses_the_existing_provider():
    backend = LiveSolanaBackend(_Cfg(raptor_enabled=False))
    from autopaper.execution.jupiter import JupiterProvider
    assert isinstance(backend.provider, JupiterProvider)


def test_raptor_enabled_selects_raptor_provider():
    backend = LiveSolanaBackend(_Cfg(raptor_enabled=True))
    assert isinstance(backend.provider, RaptorProvider)
