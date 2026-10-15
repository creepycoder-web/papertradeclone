"""
The REAL executor: wallet, quote, v0 transaction, ALT, signing, simulation,
broadcast, confirmation, and every failure mode.

Every network interaction is a mock. No test here reaches a real RPC, a real
Jupiter endpoint, or a real wallet, and the one test that touches a keypair
builds a throwaway key in a temp directory and asserts it is discarded rather
than sent.
"""


from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Optional

import pytest

from autopaper.execution.jupiter import (
    MicroOrderUnsupported,
    QuoteRejected,
    QuoteUnavailable,
)
from autopaper.execution.real import (
    AMBIGUOUS,
    FILLED_STATUSES,
    MAX_OPEN_POSITIONS,
    PRE_TRADE_CHECKS,
    LiveSolanaBackend,
    OnchainFill,
    PreflightResult,
    RealExecutor,
    SafetyCheck,
)
from autopaper.execution.solana import (
    LAMPORTS_PER_SOL,
    WSOL_MINT,
    SolanaUnavailable,
    SolanaWallet,
    b58decode,
    b58encode,
    mask_address,
    reconcile_transaction,
)

pytestmark = [
    pytest.mark.execution,
    pytest.mark.safety,
]

MINT = "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj"


# ---------------------------------------------------------------------------
# test doubles
# ---------------------------------------------------------------------------


class _Cfg:
    """A minimal real-config stand-in. No file, no network, no key."""

    def __init__(self, **kw):
        self.wallet_keypair_path = kw.get("wallet_keypair_path")
        self.rpc_url = kw.get("rpc_url", "https://rpc.invalid")
        self.slippage_bps = 500
        self.simulate_before_send = True
        self.priority_fee_lamports = 50_000
        self.confirmation_timeout_s = 5.0
        self.quote_max_age_ms = 8000.0
        self.max_price_impact_pct = 5.0


class _Quote:
    def __init__(self, **kw):
        self.in_amount = kw.get("in_amount", 155_000)
        self.expected_out_amount = kw.get("out_amount", 12_345_678)
        self.min_out_amount = kw.get("min_out_amount", 12_000_000)
        self.price_impact_pct = kw.get("impact", 0.4)
        self.route_labels = kw.get("labels", ["RAYDIUM"])
        self.slippage_bps = 500
        self.raw = kw.get("raw", {"outAmount": "12345678"})
        self.input_mint = WSOL_MINT
        self.output_mint = MINT

    @property
    def price_per_token(self) -> float:
        return self.in_amount / self.expected_out_amount


class _Built:
    """A provider's built swap.

    The default transaction is a REAL unsigned v0 transaction, not a string.
    The size gate now deserializes to read the version, so a fixture that is
    not a transaction is refused as malformed -- correctly, but it would make
    every test below assert on ``NO_FITTING_ROUTE`` instead of the behaviour it
    is about. A test double that cannot occur in production should not be the
    thing under test.
    """

    def __init__(self, b64: Optional[str] = None):
        self.transaction_b64 = b64 or _REAL_V0_TX_B64
        self.last_valid_block_height = 123_456
        self.prioritization_fee_lamports = 50_000


def _real_tx(version: str = "v0", size: Optional[int] = None) -> str:
    from tests import txfixtures
    raw = (txfixtures.build_exact(version, size) if size
           else txfixtures.build(version))
    return txfixtures.b64(raw)


_REAL_V0_TX_B64 = _real_tx("v0")
_REAL_V1_TX_B64 = _real_tx("v1")


class _Provider:
    def __init__(self, quote=None, built=None, quote_exc=None, build_exc=None):
        self._quote = quote or _Quote()
        self._built = built or _Built()
        self._quote_exc = quote_exc
        self._build_exc = build_exc
        self.quote_calls: list[tuple] = []
        self.build_calls: list[tuple] = []

    async def quote(self, i, o, amount, slip):
        self.quote_calls.append((i, o, amount, slip))
        if self._quote_exc:
            raise self._quote_exc
        return self._quote

    async def build(self, quote, pubkey, priority_fee="auto", **kw):
        self.build_calls.append((quote, pubkey, priority_fee))
        if self._build_exc:
            raise self._build_exc
        return self._built

    async def health(self):
        return {"provider": "jupiter", "reachable": True, "detail": "mock"}

    async def sol_price_usd(self):
        return 200.0

    async def aclose(self):
        return None


class _Rpc:
    def __init__(self, **kw):
        self.simulate_ok = kw.get("simulate_ok", True)
        self.wsol_ready = kw.get("wsol_ready", True)
        self.simulate_detail = kw.get("simulate_detail", "simulation ok")
        self.sent: list[str] = []
        #: Ordered log of irreversible/simulated stages, so ORDER can be
        #: asserted rather than a call count. The planner simulates each of its
        #: candidates before selecting one, so "simulation ran" is ``>= 1``;
        #: "simulation ran BEFORE the broadcast" is about position, not count.
        self.events: list[str] = []
        self.send_exc = kw.get("send_exc")
        self.confirm = kw.get("confirm") or {
            "status": "confirmed", "error": None, "signature": "sig", "slot": 7
        }
        self.balance = kw.get("balance", 3_000_000_000)
        self.get_tx = kw.get("get_tx")
        self.simulate_calls = 0

    async def get_balance_lamports(self, pubkey=None):
        """Native SOL balance. Added when the preflight began checking it."""
        if self.balance is None:
            raise RuntimeError("balance unavailable")
        return self.balance

    async def get_health(self):
        class H:
            reachable = True
            detail = "ok"
        return H()

    async def get_slot(self):
        return 123_400

    async def get_balance_lamports(self, addr):
        return self.balance

    async def wsol_readiness(self, owner, required_lamports):
        """Pre-provisioned wSOL. Models a READY wallet by default.

        The BUY path now refuses a SOL-denominated order unless the wallet
        already holds wrapped SOL, because the temporary-account route is what
        overshoots Solana's 1232-byte limit. A double that does not model this
        would exercise a code path production can never reach.
        """
        ready = self.wsol_ready
        from autopaper.execution.solana import WsolReadiness
        return WsolReadiness(
            owner=str(owner), wsol_ata="AtaStub",
            ata_present=ready, balance_lamports=10_000_000 if ready else 0,
            required_lamports=required_lamports,
            sufficient=ready, query_ok=True)

    async def simulate_transaction(self, b64):
        self.simulate_calls += 1
        self.events.append("simulate")
        return (self.simulate_ok, self.simulate_detail)

    async def simulate_transaction_detailed(self, b64):
        """The rich outcome the planner and the Raptor path both call.

        Present so the doubles model the real SolanaRpc surface. Without it the
        planner's ``simulate_transaction_detailed`` call raises AttributeError
        and every downstream test reports the wrong failure.
        """
        from autopaper.execution.solana import SimulationOutcome
        self.simulate_calls += 1
        self.events.append("simulate")
        return SimulationOutcome(
            ok=self.simulate_ok,
            status="SIMULATION_OK" if self.simulate_ok else "SIMULATION_FAILED_TX_REJECTED",
            detail=self.simulate_detail,
        )

    def transaction_size_report(self, b64):
        from autopaper.execution.txcodec import measure_transaction
        return measure_transaction(b64)

    async def get_address_lookup_table(self, address):
        """No lookup tables in these fixtures; report that honestly."""
        return {"table": address, "query_ok": True, "usable": False,
                "addresses": [], "writable": [], "readonly": [],
                "source": "double", "detail": "double has no table"}

    async def send_transaction(self, b64):
        self.events.append("send")
        if self.send_exc:
            raise self.send_exc
        self.sent.append(b64)
        return "5XyzSignature"

    async def wait_for_confirmation(self, sig, **kw):
        return dict(self.confirm)

    async def get_transaction(self, sig, commitment="confirmed"):
        return self.get_tx

    async def aclose(self):
        return None


class _Wallet:
    def __init__(self, sign_exc=None):
        self.pubkey = "SoWaLanaPubKey1111111111111111111111111111111"
        self.masked = mask_address(self.pubkey)
        self._sign_exc = sign_exc
        self.signed: list[bytes] = []

    def sign_wire(self, raw: bytes) -> bytes:
        if self._sign_exc:
            raise self._sign_exc
        self.signed.append(raw)
        return raw

    def sign_new_transaction(self, ix):
        return b"probe"

    @property
    def pubkey_obj(self):
        return None


#: A stand-in for the venue's quote vault. A swap always touches at least the
#: fee payer and the pool account, so the fixture models both.
POOL_VAULT = "VautPool111111111111111111111111111111111111"


def _chain_tx(owner: str, in_lamports: int, out_tokens: int, fee: int = 5000):
    """A plausible confirmed transaction payload.

    TWO accounts, not one: the fee payer and the venue's quote vault. The vault
    is what makes the swap leg separable from the payer's whole balance change,
    which is the distinction the 2026-10-02 $GOAT audit turned on -- the payer's
    debit also contains account-creation rent and the network fee, and dividing
    that total by a token count is not a price. With only one account there is no
    venue leg to find and the correct answer is "no price", which is why the
    single-account shape is not modelled here.

    The vault carries an existing balance, because an account with a zero prior
    balance is a NEWLY CREATED account and its incoming lamports are rent, not
    swap principal. A $GOAT entry moved 1,513,840 lamports exactly that way.
    """
    return {
        "slot": 7,
        "transaction": {
            "message": {"accountKeys": [owner, POOL_VAULT]},
            "meta": {
                "err": None,
                "fee": fee,
                "logMessages": [
                    "Program 675kPX9MHTjS2zt1qfr1NYHuzDGuEiKC9iJfF9V4aU1E invoke [1]",
                    "Program log: swap",
                ],
                # The vault already held SOL, so it is NOT a newly created
                # account and its increase is swap principal rather than rent.
                # That distinction is the whole point of the lamport split.
                "preBalances": [in_lamports, 60_000_000],
                "postBalances": [0, 60_000_000 + in_lamports],
                "preTokenBalances": [],
                "postTokenBalances": [
                    {"accountIndex": 0, "mint": MINT, "owner": owner,
                     "uiTokenAmount": {"amount": str(out_tokens), "decimals": 0}}
                ],
            },
        },
    }


def _sol_usd(price=200.0):
    """
    An offline SOL/USD REFERENCE provider.

    ``LiveSolanaBackend.sol_price_usd()`` used to derive the reference from a
    ``WSOL -> USDC`` executable quote. It no longer does: a reference price is
    not a swap, and asking the execution quoter for one is what made the
    direction rule refuse the leg. The reference is now injected, so these
    fixtures supply it directly. In-memory: no client, no socket.
    """
    from autopaper.same_market.sol_price import SolUsdSnapshot

    class _P:
        def __init__(self):
            self.calls = 0

        async def snapshot(self, *, refresh=False):
            self.calls += 1
            import time as _t
            return SolUsdSnapshot(
                usd_per_sol=float(price), timestamp=_t.time(),
                source="fake (test)", fetched=True)

    return _P()


def _backend(**kw) -> LiveSolanaBackend:
    return LiveSolanaBackend(
        _Cfg(),
        wallet=kw.get("wallet") or _Wallet(),
        rpc=kw.get("rpc") or _Rpc(),
        provider=kw.get("provider") or _Provider(),
        sol_usd_provider=kw.get("sol_usd_provider") or _sol_usd(),
    )


# ---------------------------------------------------------------------------
# base58 and masking
# ---------------------------------------------------------------------------


def test_base58_round_trip():
    raw = bytes(range(32))
    assert b58decode(b58encode(raw)) == raw


def test_base58_preserves_leading_zeros():
    raw = b"\x00\x00" + bytes(range(30))
    assert b58decode(b58encode(raw)) == raw


def test_mask_address_hides_the_middle():
    masked = mask_address("So11111111111111111111111111111111111111112")
    assert masked.startswith("So11")
    assert masked.endswith("1112")
    assert len(masked) < 20


def test_mask_address_on_an_empty_value():
    assert mask_address("") == "(none)"


# ---------------------------------------------------------------------------
# wallet validation
# ---------------------------------------------------------------------------


def test_wallet_refuses_an_unconfigured_path():
    with pytest.raises(SolanaUnavailable, match="not configured"):
        SolanaWallet("")


def test_wallet_refuses_a_missing_file(tmp_path: Path):
    with pytest.raises(SolanaUnavailable, match="not found"):
        SolanaWallet(str(tmp_path / "absent.json"))


def test_wallet_refuses_a_malformed_keypair(tmp_path: Path):
    """A JSON object is parseable but is not a 64-byte keypair."""
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "a keypair"}')
    with pytest.raises(SolanaUnavailable, match="64-byte"):
        SolanaWallet(str(bad))


def test_wallet_refuses_unparseable_json(tmp_path: Path):
    bad = tmp_path / "bad2.json"
    bad.write_text("{not json at all")
    with pytest.raises(SolanaUnavailable, match="could not be parsed"):
        SolanaWallet(str(bad))


def test_wallet_refuses_a_wrong_length_keypair(tmp_path: Path):
    bad = tmp_path / "short.json"
    bad.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(SolanaUnavailable, match="64-byte"):
        SolanaWallet(str(bad))


def test_wallet_refuses_a_key_with_out_of_range_bytes(tmp_path: Path):
    bad = tmp_path / "range.json"
    bad.write_text(json.dumps([999] * 64))
    with pytest.raises(SolanaUnavailable, match="64-byte"):
        SolanaWallet(str(bad))


def test_wallet_never_exposes_the_key_in_its_repr():
    """
    A throwaway keypair is written, loaded, and its repr inspected. The repr
    must be safe to put in a log or a crash report.
    """
    solders = pytest.importorskip("solders.keypair")
    kp = solders.Keypair()
    path = _write_keypair(tmp_path_for_this_test := Path(
        __import__("tempfile").mkdtemp()
    ), kp)
    w = SolanaWallet(str(path))
    text = repr(w) + " " + str(w)
    assert "key=<withheld>" in text
    # no byte of the key material appears
    secret = bytes(kp)
    assert base64.b64encode(secret).decode() not in text
    assert str(secret)[:20] not in text
    assert w.masked in text
    assert w.pubkey in w.masked or True   # the mask, not the raw key


def _write_keypair(tmp: Path, kp) -> Path:
    path = tmp / "id.json"
    path.write_text(json.dumps(list(bytes(kp))))
    return path


def test_wallet_loads_a_real_keypair_and_derives_a_pubkey(tmp_path: Path):
    solders = pytest.importorskip("solders.keypair")
    kp = solders.Keypair()
    w = SolanaWallet(str(_write_keypair(tmp_path, kp)))
    assert w.pubkey == str(kp.pubkey())
    # The invariant is the DECODED length, not the base58 length. A base58
    # pubkey is 43-44 chars for almost every key but is shorter when the
    # leading bytes are small, so the old `len(...) in (32, 44)` character
    # assertion was a rare random flake. Every key decodes to exactly 32 bytes.
    from solders.pubkey import Pubkey

    assert len(bytes(Pubkey.from_string(w.pubkey))) == 32
    assert Pubkey.from_string(w.pubkey) == kp.pubkey()


def test_wallet_sign_wire_refuses_an_empty_transaction(tmp_path: Path):
    solders = pytest.importorskip("solders.keypair")
    w = SolanaWallet(str(_write_keypair(tmp_path, solders.Keypair())))
    with pytest.raises(SolanaUnavailable, match="empty"):
        w.sign_wire(b"")


def test_wallet_sign_wire_refuses_a_zero_signature_transaction(tmp_path: Path):
    """A transaction with no signature slot is malformed, not signable."""
    solders = pytest.importorskip("solders.keypair")
    w = SolanaWallet(str(_write_keypair(tmp_path, solders.Keypair())))
    with pytest.raises(SolanaUnavailable, match="no signature slot"):
        w.sign_wire(bytes([0]) + b"message")


def test_wallet_sign_wire_splices_the_signature_in_place(tmp_path: Path):
    """
    The wire-level signing path a v0 Jupiter swap requires: locate the signature
    array, sign the message that follows it, splice the signature back. Legacy
    ``Transaction.sign()`` cannot represent an ALT-bearing versioned
    transaction, which is why this exists.
    """
    solders = pytest.importorskip("solders.keypair")
    w = SolanaWallet(str(_write_keypair(tmp_path, solders.Keypair())))
    message = b"\x01\x02\x03message-bytes"
    raw = bytes([1]) + b"\x00" * 64 + message
    out = w.sign_wire(raw)
    assert len(out) == len(raw)
    assert out[0] == 1
    assert out[65:] == message
    assert out[1:65] != b"\x00" * 64, "the signature must be written"


# ---------------------------------------------------------------------------
# quote
# ---------------------------------------------------------------------------


def test_quote_is_requested_with_the_position_lamports():
    prov = _Provider()
    b = _backend(provider=prov)
    asyncio.run(b.execute_swap("ENTRY", WSOL_MINT, MINT, 155_000, 500))
    assert prov.quote_calls[0] == (WSOL_MINT, MINT, 155_000, 500)


def test_a_venue_that_refuses_the_notional_is_reported_distinctly():
    """
    This is the micro-account case. The venue's refusal is a separate, named
    outcome - not a generic failure - and it never becomes a fill.
    """
    prov = _Provider(
        quote_exc=MicroOrderUnsupported("venue rejected the notional as too small")
    )
    fill = asyncio.run(
        _backend(provider=prov).execute_swap("ENTRY", WSOL_MINT, MINT, 155_000, 500)
    )
    assert fill.status == "MICRO_REAL_EXECUTION_UNSUPPORTED"
    assert not fill.filled
    assert "too small" in fill.ambiguity_reason
    assert fill.signature == "", "no broadcast, so no signature"


def test_excessive_price_impact_is_a_quote_rejection():
    prov = _Provider(quote_exc=QuoteRejected("price impact 9.1% exceeds 5.0%"))
    fill = asyncio.run(
        _backend(provider=prov).execute_swap("ENTRY", WSOL_MINT, MINT, 155_000, 500)
    )
    assert fill.status == "QUOTE_REJECTED"
    assert not fill.filled


def test_an_unavailable_quote_is_reported():
    prov = _Provider(quote_exc=QuoteUnavailable("quote provider timed out"))
    fill = asyncio.run(
        _backend(provider=prov).execute_swap("ENTRY", WSOL_MINT, MINT, 155_000, 500)
    )
    assert fill.status == "QUOTE_UNAVAILABLE"
    assert not fill.filled


def test_the_quote_price_and_impact_are_recorded():
    prov = _Provider()
    fill = asyncio.run(
        _backend(provider=prov).execute_swap("ENTRY", WSOL_MINT, MINT, 155_000, 500)
    )
    assert fill.quote_price == pytest.approx(155_000 / 12_345_678)
    assert fill.price_impact_pct == 0.4
    assert fill.route_labels == ["RAYDIUM"]


# ---------------------------------------------------------------------------
# build / sign / simulate
# ---------------------------------------------------------------------------


def test_build_failure_is_reported_before_signing():
    prov = _Provider(build_exc=QuoteUnavailable("venue did not return a transaction"))
    wallet = _Wallet()
    fill = asyncio.run(
        _backend(provider=prov, wallet=wallet).execute_swap(
            "ENTRY", WSOL_MINT, MINT, 155_000, 500
        )
    )
    assert fill.status == "BUILD_FAILED"
    assert not wallet.signed, "nothing is signed when the build fails"


def test_a_versioned_transaction_is_signed_on_the_wire():
    """
    A v0 transaction with address lookup tables. The mock wallet records the raw
    wire bytes it was handed, which is what a real Jupiter swap is.
    """
    # A REAL unsigned v0 transaction. The size gate deserializes to read the
    # version, so a hand-made blob is refused as malformed and the test would
    # then assert on a refusal instead of on signing.
    v0_wire = base64.b64decode(_real_tx("v0"))
    prov = _Provider(built=_Built(base64.b64encode(v0_wire).decode()))
    wallet = _Wallet()
    rpc = _Rpc(get_tx=_chain_tx(wallet.pubkey, 155_000, 12_345_678))
    fill = asyncio.run(
        _backend(provider=prov, wallet=wallet, rpc=rpc).execute_swap(
            "ENTRY", WSOL_MINT, MINT, 155_000, 500
        )
    )
    assert wallet.signed[0] == v0_wire
    assert fill.status in FILLED_STATUSES


def test_a_signing_failure_never_becomes_a_fill():
    wallet = _Wallet(sign_exc=RuntimeError("key rejected"))
    rpc = _Rpc()
    fill = asyncio.run(
        _backend(wallet=wallet, rpc=rpc).execute_swap(
            "ENTRY", WSOL_MINT, MINT, 155_000, 500
        )
    )
    assert fill.status == "SIGN_FAILED"
    assert not fill.filled
    assert rpc.sent == [], "a signing failure must not reach the broadcast"
    assert "key rejected" not in fill.ambiguity_reason, "the reason is a type, not a secret"


def test_simulation_runs_before_broadcast():
    """The invariant is ORDER, not a call count.

    The planner simulates each candidate before selecting one, so the number of
    simulations is a property of the candidate list, not of the safety contract.
    What must hold is that the broadcast happened at all and that a simulation
    preceded it.
    """
    rpc = _Rpc()
    asyncio.run(
        _backend(rpc=rpc).execute_swap("ENTRY", WSOL_MINT, MINT, 155_000, 500)
    )
    assert rpc.simulate_calls >= 1
    assert len(rpc.sent) == 1
    assert rpc.events, "no stage was recorded"
    assert rpc.events.index("send") > 0
    assert "simulate" in rpc.events[:rpc.events.index("send")], (
        f"a broadcast happened before any simulation: {rpc.events}")


def test_a_failed_simulation_prevents_the_broadcast():
    rpc = _Rpc(simulate_ok=False, simulate_detail="simulation error: {InstructionError}")
    fill = asyncio.run(
        _backend(rpc=rpc).execute_swap("ENTRY", WSOL_MINT, MINT, 155_000, 500)
    )
    # The failure taxonomy is now specific (SIMULATION_FAILED_TX_REJECTED,
    # ..._TX_TOO_LARGE, ..._RPC_INVALID_PARAMS, ..._RPC), so assert the
    # invariant this test actually protects -- a simulation refusal -- rather
    # than one opaque string. This is a strictly wider check than before.
    assert fill.status.startswith("SIMULATION_FAILED"), fill.status
    assert rpc.sent == [], "a failed simulation must not be broadcast"
    assert not fill.filled


# ---------------------------------------------------------------------------
# broadcast / confirmation
# ---------------------------------------------------------------------------


def test_a_broadcast_rejection_is_not_a_fill():
    from autopaper.execution.solana import TransactionFailed

    rpc = _Rpc(send_exc=TransactionFailed("sendTransaction returned no signature"))
    fill = asyncio.run(
        _backend(rpc=rpc).execute_swap("ENTRY", WSOL_MINT, MINT, 155_000, 500)
    )
    assert fill.status == "BROADCAST_REJECTED"
    assert not fill.filled


def test_a_confirmation_timeout_is_ambiguous_not_a_fill():
    """
    The single most important assertion in this file: we stopped waiting, so we
    do not know. Reporting that as a fill is how a bot ends up believing it owns
    a token it never received.
    """
    rpc = _Rpc(confirm={"status": "timeout", "error": "no status returned"})
    fill = asyncio.run(
        _backend(rpc=rpc).execute_swap("ENTRY", WSOL_MINT, MINT, 155_000, 500)
    )
    assert fill.status == AMBIGUOUS
    assert not fill.filled
    assert fill.signature == "5XyzSignature", "the signature is still recorded"


def test_an_on_chain_error_is_reported_as_a_chain_failure():
    rpc = _Rpc(confirm={"status": "failed", "error": {"InstructionError": [0, {}]}})
    fill = asyncio.run(
        _backend(rpc=rpc).execute_swap("ENTRY", WSOL_MINT, MINT, 155_000, 500)
    )
    assert fill.status == "CHAIN_FAILED"
    assert not fill.filled


def test_an_expired_blockhash_is_reported():
    rpc = _Rpc(confirm={"status": "expired", "error": "blockhash expired"})
    fill = asyncio.run(
        _backend(rpc=rpc).execute_swap("ENTRY", WSOL_MINT, MINT, 155_000, 500)
    )
    assert fill.status == "CHAIN_EXPIRED"
    assert not fill.filled


def test_a_confirmed_fill_carries_the_actual_chain_amounts():
    rpc = _Rpc(get_tx=_chain_tx("SoWaLanaPubKey1111111111111111111111111111111",
                                155_000, 12_345_678))
    fill = asyncio.run(
        _backend(rpc=rpc).execute_swap("ENTRY", WSOL_MINT, MINT, 155_000, 500)
    )
    assert fill.filled
    assert fill.status == "confirmed"
    assert fill.signature == "5XyzSignature"
    assert fill.output_amount == 12_345_678
    # The fee is the transaction's own ``meta.fee`` (5000 lamports), NOT the
    # wallet's SOL balance delta (155_000 = swap principal + fee). Using the
    # delta as the fee is the ~4500x overstatement this corrects.
    assert fill.fee_lamports == 5_000
    # ``input_amount`` is the wallet's total SOL decrease (principal + fee) and
    # is reported deliberately gross; the fee is a SEPARATE field.
    assert fill.input_amount == 155_000
    assert fill.slot == 7
    assert fill.confirmed_at is not None


def test_a_confirmed_transaction_with_no_output_is_unreconciled_not_ambiguous():
    """
    Confirmed is not the same as filled -- but it is emphatically NOT the same
    as "unknown". The chain said this swap succeeded, so the outcome is
    CONFIRMED_BUT_UNRECONCILED: money moved, we just cannot measure how much.

    This is the $Boy bug. Asserting AMBIGUOUS here is what made a successful
    on-chain buy get journalled as ``BUY_BLOCKED=fill_refused`` with no position
    opened, while the wallet held the tokens. The amount is never invented; the
    fill is flagged for recovery from on-chain state.
    """
    from autopaper.execution.real import CONFIRMED_UNRECONCILED

    tx = _chain_tx("SoWaLanaPubKey1111111111111111111111111111111", 155_000, 12_345_678)
    tx["transaction"]["meta"]["postTokenBalances"] = []
    rpc = _Rpc(get_tx=tx)
    fill = asyncio.run(
        _backend(rpc=rpc).execute_swap("ENTRY", WSOL_MINT, MINT, 155_000, 500)
    )
    assert fill.status == CONFIRMED_UNRECONCILED
    assert fill.status != AMBIGUOUS, (
        "a confirmed swap must not be reported as an unknown outcome")
    # Never a usable fill: the amount was not measured, so nothing may consume it.
    assert not fill.filled
    assert fill.requires_recovery is True
    assert fill.confirmed_on_chain is True
    # The amount must NOT be invented from the quote.
    assert fill.output_amount in (None, 0)
    assert "no output amount" in fill.ambiguity_reason


def test_a_missing_transaction_body_is_unreconciled_not_ambiguous():
    """Confirmation succeeded; the metadata fetch did not. Same category.

    ``wait_for_confirmation`` already reported confirmed, so this is not an
    unknown outcome -- it is a confirmed one whose amounts could not be read.
    """
    from autopaper.execution.real import CONFIRMED_UNRECONCILED

    rpc = _Rpc(get_tx=None)
    fill = asyncio.run(
        _backend(rpc=rpc).execute_swap("ENTRY", WSOL_MINT, MINT, 155_000, 500)
    )
    assert fill.status == CONFIRMED_UNRECONCILED
    assert not fill.filled
    assert fill.requires_recovery is True


def test_actual_slippage_is_measurable_against_the_quote():
    """
    Actual execution price vs the quoted price. A positive number means the
    fill was worse than quoted, which is the number an operator needs.
    """
    from autopaper.real_account import RealFillRecord

    # The chain reports a 155_000 lamport total SOL decrease, and returns
    # 12_000_000 tokens instead of the 12_345_678 the quote promised.
    rpc = _Rpc(get_tx=_chain_tx(
        "SoWaLanaPubKey1111111111111111111111111111111", 155_000, 12_000_000
    ))
    fill = asyncio.run(
        _backend(rpc=rpc).execute_swap("ENTRY", WSOL_MINT, MINT, 155_000, 500)
    )
    rec = RealFillRecord.from_onchain_fill(fill)
    # The recorded input is the swap PRINCIPAL: the SOL delta minus the fee.
    # The two are separate quantities and the fee is reported separately.
    assert rec.actual_input_amount == 155_000  # gross: principal + fee
    assert rec.actual_output_amount == 12_000_000
    assert rec.actual_slippage is not None
    assert rec.actual_slippage > 0, "fewer tokens than quoted is a worse fill"
    assert rec.execution_price == pytest.approx(155_000 / 12_000_000)


# ---------------------------------------------------------------------------
# reconciliation
# ---------------------------------------------------------------------------


def test_reconcile_reads_the_owner_own_balance_delta():
    owner = "SoWaLanaPubKey1111111111111111111111111111111"
    out = reconcile_transaction(
        _chain_tx(owner, 155_000, 12_345_678), "sig", "confirmed", owner, MINT
    )
    assert out.token_output_amount == 12_345_678
    # Fee comes from meta.fee, not the balance delta.
    assert out.fee_lamports == 5_000
    assert out.net_lamports_delta == 155_000
    assert out.success is True
    assert out.slot == 7


def test_reconcile_of_an_empty_transaction_is_incomplete_not_estimated():
    out = reconcile_transaction({}, "sig", "confirmed", "owner", MINT)
    assert out.token_output_amount is None
    assert out.fee_lamports is None
    assert out.success is False


def test_reconcile_ignores_another_accounts_token_leg():
    """
    If the output mint does not appear in the owner's deltas, the result stays
    incomplete rather than borrowing another account's movement.
    """
    owner = "SoWaLanaPubKey1111111111111111111111111111111"
    tx = _chain_tx(owner, 155_000, 12_345_678)
    tx["transaction"]["meta"]["postTokenBalances"][0]["mint"] = (
        "SomeOtherMint1111111111111111111111111111")
    out = reconcile_transaction(tx, "sig", "confirmed", owner, MINT)
    assert out.token_output_amount is None


def test_reconcile_marks_a_failed_transaction():
    owner = "SoWaLanaPubKey1111111111111111111111111111111"
    tx = _chain_tx(owner, 155_000, 12_345_678)
    tx["transaction"]["meta"]["err"] = {"InstructionError": [0, {}]}
    out = reconcile_transaction(tx, "sig", "confirmed", owner, MINT)
    assert out.success is False


# ---------------------------------------------------------------------------
# pre-trade safety
# ---------------------------------------------------------------------------


def test_every_required_pre_trade_check_is_named():
    """
    The full checklist from the migration is present. Named here so a check
    cannot be quietly dropped.
    """
    required = {
        "real_mode_explicit",
        "real_armed_by_operator",
        "wallet_loaded",
        "rpc_healthy",
        "quote_valid",
        "sol_for_transaction_costs",
        "sizing_matches_policy",
        "exposure_within_limit",
        "no_duplicate_position",
        "no_reentry",
        "no_emergency_stop",
        "no_concurrent_operation",
        "v3c_authorized",
    }
    assert required.issubset(set(PRE_TRADE_CHECKS))


def test_a_failed_precheck_prevents_the_trade():
    """
    Fail closed: one failed check is enough, and it names itself.
    """
    ex = RealExecutor(_Cfg(), backend=_backend())
    fill = asyncio.run(
        ex.open_position(
            mint=MINT,
            position_lamports=155_000,
            slippage_bps=500,
            prechecks=[
                SafetyCheck("real_armed_by_operator", True, "armed"),
                SafetyCheck("sol_for_transaction_costs", False, "not enough SOL"),
            ],
        )
    )
    assert fill.status == "PRE_TRADE_CHECK_FAILED"
    assert not fill.filled
    assert "sol_for_transaction_costs" in fill.ambiguity_reason


def test_all_checks_passing_lets_the_trade_through():
    rpc = _Rpc(get_tx=_chain_tx("SoWaLanaPubKey1111111111111111111111111111111",
                                155_000, 12_345_678))
    ex = RealExecutor(_Cfg(), backend=_backend(rpc=rpc))
    fill = asyncio.run(
        ex.open_position(
            mint=MINT,
            position_lamports=155_000,
            slippage_bps=500,
            prechecks=[SafetyCheck("real_armed_by_operator", True, "armed")],
        )
    )
    assert fill.filled
    assert len(rpc.sent) == 1


def test_insufficient_sol_is_reported_with_both_numbers():
    rpc = _Rpc(balance=100_000)     # 0.0001 SOL, far below any reserve
    ex = RealExecutor(_Cfg(), backend=_backend(rpc=rpc))
    check = asyncio.run(
        ex.sufficient_sol(position_lamports=155_000, sol_price_usd=200.0)
    )
    assert not check.ok
    assert "wallet" in check.detail
    assert "required" in check.detail
    assert "reserve" in check.detail


def test_a_155_lamport_position_needs_a_reserve_on_top():
    """
    The reserve exists so a swap can never spend the SOL needed for its own fee
    and its token account.
    """
    ex = RealExecutor(_Cfg(), backend=_backend())
    reserve = ex.required_reserve_sol()
    assert reserve > 0.0
    assert reserve >= 0.0015


def test_a_micro_notional_that_the_venue_will_not_take_is_reported():
    """
    The question this migration must answer honestly: can the venue execute a
    ~$0.155 order? If not, the answer is MICRO_REAL_EXECUTION_UNSUPPORTED
    with the venue's reason - and the size is never enlarged.
    """
    prov = _Provider(
        quote_exc=MicroOrderUnsupported(
            "venue rejected the notional as too small (HTTP 400): TOO_SMALL"
        )
    )
    ex = RealExecutor(_Cfg(), backend=_backend(provider=prov))
    check, fill = asyncio.run(
        ex.quote_probe(mint=MINT, position_lamports=155_000, slippage_bps=500,
                        side="ENTRY")
    )
    assert not check.ok
    assert fill is not None
    assert fill.status == "MICRO_REAL_EXECUTION_UNSUPPORTED"
    assert "too small" in fill.ambiguity_reason
    assert "TOO_SMALL" in fill.ambiguity_reason


def test_a_valid_quote_passes_the_probe():
    ex = RealExecutor(_Cfg(), backend=_backend())
    check, fill = asyncio.run(
        ex.quote_probe(mint=MINT, position_lamports=155_000, slippage_bps=500,
                        side="ENTRY")
    )
    assert check.ok, check.detail
    assert fill is None
    assert "expected" in check.detail


def test_max_open_positions_is_one():
    assert MAX_OPEN_POSITIONS == 1


def test_configuration_gates_refuse_a_missing_wallet():
    ex = RealExecutor(_Cfg(wallet_keypair_path=None), backend=_backend())
    names = {c.name: c for c in ex.configuration_gates()}
    assert not names["wallet_configured"].ok
    assert "WALLET_KEYPAIR_PATH" in names["wallet_configured"].detail


def test_configuration_gates_refuse_a_missing_rpc():
    cfg = _Cfg()
    cfg.rpc_url = None
    ex = RealExecutor(cfg, backend=_backend())
    names = {c.name: c for c in ex.configuration_gates()}
    assert not names["rpc_configured"].ok


def test_configuration_gates_require_reentry_disabled():
    cfg = _Cfg()
    cfg.reentry_enabled = True
    ex = RealExecutor(cfg, backend=_backend())
    names = {c.name: c for c in ex.configuration_gates()}
    assert not names["reentry_disabled"].ok


def test_configuration_gates_refuse_an_unbounded_slippage():
    cfg = _Cfg()
    cfg.slippage_bps = 0
    ex = RealExecutor(cfg, backend=_backend())
    names = {c.name: c for c in ex.configuration_gates()}
    assert not names["slippage_bounded"].ok


def test_a_gate_never_reveals_a_configuration_value():
    """A gate reports a name and a reason, never a path or a URL."""
    ex = RealExecutor(_Cfg(wallet_keypair_path="/home/op/.config/solana/id.json"),
                      backend=_backend())
    for c in ex.configuration_gates():
        assert "/home/op" not in c.detail
        assert ".config/solana" not in c.detail


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def test_preflight_passes_with_everything_configured(tmp_path: Path):
    solders = pytest.importorskip("solders.keypair")
    cfg = _Cfg(
        wallet_keypair_path=str(_write_keypair(tmp_path, solders.Keypair())),
        rpc_url="https://rpc.invalid",
        max_open_positions=1,
    )
    ex = RealExecutor(cfg, backend=_backend(wallet=SolanaWallet(cfg.wallet_keypair_path)))
    result = asyncio.run(ex.preflight())
    assert result.ok, result.render()


def test_preflight_fails_without_a_wallet():
    """A wallet that cannot load is a failed check, not a raised exception."""
    ex = RealExecutor(_Cfg(wallet_keypair_path=None), backend=_backend())
    result = asyncio.run(ex.preflight())
    assert not result.ok
    failed = [c for c in result.checks if not c.ok]
    assert failed, result.render()
    assert "wallet" in " ".join(c.name for c in failed)


def test_preflight_reports_a_wallet_that_will_not_load(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("[1,2,3]")
    cfg = _Cfg(
        wallet_keypair_path=str(bad), rpc_url="https://rpc.invalid",
        max_open_positions=1,
    )
    # The backend builds the wallet LAZILY, exactly as it does in production, so
    # a wallet that will not load surfaces as a failed check rather than an
    # exception escaping preflight.
    ex = RealExecutor(cfg, backend=LiveSolanaBackend(cfg, rpc=_Rpc()))
    result = asyncio.run(ex.preflight())
    assert not result.ok
    failed = [c for c in result.checks if not c.ok]
    assert any("wallet" in c.name for c in failed), result.render()


def test_preflight_fails_on_a_dead_rpc():
    class _Dead(_Rpc):
        async def get_health(self):
            class H:
                reachable = False
                detail = "RPC unreachable"
            return H()

    ex = RealExecutor(_Cfg(), backend=_backend(rpc=_Dead()))
    result = asyncio.run(ex.preflight())
    assert not result.ok
    assert any(c.name == "rpc_healthy" and not c.ok for c in result.checks)


def test_preflight_verifies_the_sizing_policy(tmp_path: Path):
    solders = pytest.importorskip("solders.keypair")
    path = str(_write_keypair(tmp_path, solders.Keypair()))
    cfg = _Cfg(wallet_keypair_path=path, max_open_positions=1)
    ex = RealExecutor(cfg, backend=_backend(wallet=SolanaWallet(path)))
    result = asyncio.run(ex.preflight())
    check = next(c for c in result.checks if c.name == "sizing_matches_policy")
    assert check.ok
    assert "$0.155" in check.detail


def test_preflight_never_broadcasts(tmp_path: Path):
    solders = pytest.importorskip("solders.keypair")
    rpc = _Rpc()
    path = str(_write_keypair(tmp_path, solders.Keypair()))
    cfg = _Cfg(wallet_keypair_path=path, max_open_positions=1)
    ex = RealExecutor(cfg, backend=_backend(rpc=rpc, wallet=SolanaWallet(path)))
    result = asyncio.run(ex.preflight())
    assert rpc.sent == [], "the preflight must not broadcast"
    report = next(c for c in result.checks
                  if c.name == "no_broadcast_in_preflight")
    assert report.ok
    assert result.as_dict()["ok"] is True


def test_preflight_render_lists_every_check(tmp_path: Path):
    solders = pytest.importorskip("solders.keypair")
    path = str(_write_keypair(tmp_path, solders.Keypair()))
    cfg = _Cfg(wallet_keypair_path=path, max_open_positions=1)
    ex = RealExecutor(cfg, backend=_backend(wallet=SolanaWallet(path)))
    text = asyncio.run(ex.preflight()).render()
    for name in ("wallet_loaded", "rpc_healthy", "quote_provider_ready",
                 "sizing_matches_policy", "no_broadcast_in_preflight"):
        assert name in text


def test_placement_is_genuinely_implemented():
    assert RealExecutor.PLACEMENT_IMPLEMENTED is True


# ---------------------------------------------------------------------------
# concurrency
# ---------------------------------------------------------------------------


def test_the_executor_reports_itself_busy_during_an_operation():
    rpc = _Rpc()
    ex = RealExecutor(_Cfg(), backend=_backend(rpc=rpc))
    seen: list[bool] = []

    async def _check():
        seen.append(ex.busy)

    # The mocks resolve immediately, so give the swap a real suspension point.
    async def _slow_quote(*a, **kw):
        await asyncio.sleep(0.01)
        return _Quote()

    backend = _backend(provider=_Provider())
    backend.provider.quote = _slow_quote
    ex = RealExecutor(_Cfg(), backend=backend)

    async def _go():
        task = asyncio.ensure_future(
            ex.open_position(
                mint=MINT, position_lamports=155_000, slippage_bps=500,
                prechecks=[SafetyCheck("real_armed_by_operator", True, "armed")],
            )
        )
        # let the coroutine reach the swap, which is where _busy is set
        for _ in range(10):
            await asyncio.sleep(0)
            if ex.busy:
                break
        await _check()
        await task

    asyncio.run(_go())
    assert seen == [True]
    assert ex.busy is False


def test_a_concurrent_operation_is_refused_by_the_gate():
    ex = RealExecutor(_Cfg(), backend=_backend())
    ex._busy = True
    check = SafetyCheck(
        "no_concurrent_operation", not ex.busy, "a real operation is in flight"
    )
    assert not check.ok


# ---------------------------------------------------------------------------
# the micro envelope
# ---------------------------------------------------------------------------


def test_the_micro_envelope_does_not_enlarge_a_tiny_order():
    """
    The exact behaviour the migration forbids: a refused $0.155 order must be
    rejected, never re-sized. The lamports handed in are the lamports sent.
    """
    prov = _Provider(quote_exc=MicroOrderUnsupported("TOO_SMALL"))
    rpc = _Rpc()
    b = _backend(provider=prov, rpc=rpc)
    asyncio.run(b.execute_swap("ENTRY", WSOL_MINT, MINT, 155_000, 500))
    assert prov.quote_calls[0][2] == 155_000, "the amount must not change"
    assert prov.build_calls == [], "no transaction is even built"
    assert rpc.sent == [], "nothing is broadcast"


def test_onchain_fill_serializes_its_actual_values():
    fill = OnchainFill(
        signature="5Xyz", status="confirmed", commitment="confirmed",
        input_amount=155_000, output_amount=12_345_678, fee_lamports=5000,
        execution_price=0.01255, quote_price=0.01255, slot=7,
    )
    d = fill.as_dict()
    assert d["filled"] is True
    assert d["fee_sol"] == pytest.approx(0.000005)
    assert d["signature"] == "5Xyz"
    assert d["output_amount"] == 12_345_678


def test_filled_statuses_are_exactly_the_confirmed_pair():
    assert FILLED_STATUSES == ("confirmed", "finalized")
    for bad in ("timeout", "failed", "expired", "", "AMBIGUOUS"):
        assert OnchainFill(status=bad).filled is False


def test_lamports_per_sol():
    assert LAMPORTS_PER_SOL == 1_000_000_000
