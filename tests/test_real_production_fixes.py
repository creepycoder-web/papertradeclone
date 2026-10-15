"""
Production REAL-path tests.

These are the tests that were MISSING: the previous suite passed with the REAL
exit path absent, the REAL buy path unreachable, the exit PnL in mixed units and
no durable idempotency record. Nothing here signs or broadcasts a real
transaction: wallets, RPC and the venue are in-process doubles, and the fake
signer returns bytes without touching key material.
"""


from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autopaper.clock import FakeClock
from autopaper.config import build_config
from autopaper.execution.intents import (
    BROADCASTED,
    CONFIRMED,
    CREATED,
    FAILED,
    READY,
    SIGNING,
    UNKNOWN,
    DuplicateExecutionError,
    IntentStateError,
    ExecutionIntentStore,
)
from autopaper.execution.real import LiveSolanaBackend, RealExecutor
from autopaper.execution.solana import (
    FEE_SANITY_MAX_LAMPORTS,
    LAMPORTS_PER_SOL,
    WSOL_MINT,
    reconcile_transaction,
)
from autopaper.real_account import (
    POSITION_CLOSED,
    RealAccount,
    RealFillRecord,
)

pytestmark = [
    pytest.mark.execution,
    pytest.mark.strategy,
]

BASE_T = 1_800_000_000.0
MINT = "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj"
OWNER = "SoWaLanaPubKey1111111111111111111111111111111"


# ---------------------------------------------------------------------------
# in-process doubles (no network, no key material, no broadcast)
# ---------------------------------------------------------------------------


class _Cfg:
    def __init__(self, **kw):
        self.wallet_keypair_path = kw.get("wallet_keypair_path", "/tmp/fake-id.json")
        self.rpc_url = kw.get("rpc_url", "https://rpc.invalid")
        self.slippage_bps = 500
        self.simulate_before_send = True
        self.priority_fee_lamports = 50_000
        self.confirmation_timeout_s = 5.0
        self.quote_max_age_ms = 8_000.0
        self.max_price_impact_pct = 5.0
        self.max_open_positions = 1
        self.reentry_enabled = False
        self.trading_enabled = False
        self.risk_acknowledged = True


class _Quote:
    def __init__(self, in_amount, out_amount, impact=0.4):
        self.in_amount = in_amount
        self.expected_out_amount = out_amount
        self.min_out_amount = int(out_amount * 0.99)
        self.price_impact_pct = impact
        self.route_labels = ["RAYDIUM"]

    @property
    def price_per_token(self) -> float:
        return self.in_amount / max(1, self.expected_out_amount)


class _Built:
    transaction_b64 = "AQAB"
    last_valid_block_height = 123_456
    prioritization_fee_lamports = 50_000


class _Provider:
    """Venue double. Returns a SOL/USD quote for the price probe and a swap quote
    for the order."""

    def __init__(self, sol_price_usd=200.0, out_tokens=12_345_678, quote_exc=None):
        self.sol_price_usd = sol_price_usd
        self.out_tokens = out_tokens
        self.quote_exc = quote_exc
        self.quote_calls = []
        self.build_calls = []

    async def quote(self, i, o, amount, slip):
        self.quote_calls.append((i, o, amount, slip))
        if self.quote_exc:
            raise self.quote_exc
        if o == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v":
            # SOL -> USDC: 1 SOL == sol_price_usd USDC (USDC has 6 decimals).
            return _Quote(amount, int(self.sol_price_usd * 1e6))
        return _Quote(amount, self.out_tokens)

    async def build(self, quote, pubkey, priority_fee="auto", **kw):
        self.build_calls.append((quote, pubkey, priority_fee))
        return _Built()

    async def health(self):
        return {"provider": "jupiter", "reachable": True, "detail": "mock"}

    async def aclose(self):
        return None


class _Rpc:
    def __init__(self, **kw):
        self.balance = kw.get("balance", 3_000_000_000)
        self.sent = []
        self.simulate_ok = kw.get("simulate_ok", True)
        self.wsol_ready = kw.get("wsol_ready", True)
        self.confirm = kw.get("confirm") or {
            "status": "confirmed", "error": None, "signature": "Sig111", "slot": 9,
        }
        self.get_tx = kw.get("get_tx")
        self.signature_statuses = kw.get("signature_statuses")
        self.token_balances = kw.get("token_balances", {})
        self.send_exc = kw.get("send_exc")

    async def get_balance_lamports(self, addr=None):
        if self.balance is None:
            raise RuntimeError("balance unavailable")
        return self.balance

    async def get_health(self):
        return SimpleNamespace(reachable=True, detail="ok")

    async def get_slot(self):
        return 120_000

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
        return (self.simulate_ok, "simulation ok" if self.simulate_ok else "bad ix")

    async def send_transaction(self, b64):
        if self.send_exc:
            raise self.send_exc
        self.sent.append(b64)
        return "Sig111"

    async def wait_for_confirmation(self, sig, **kw):
        return dict(self.confirm)

    async def get_transaction(self, sig, commitment="confirmed"):
        return self.get_tx

    async def get_signature_statuses(self, sig):
        if self.signature_statuses is None:
            return []
        return self.signature_statuses

    async def get_token_account_balance(self, owner, mint):
        return self.token_balances.get(mint)

    async def aclose(self):
        return None


class _Wallet:
    def __init__(self):
        self.pubkey = OWNER
        self.masked = "SoWa…1111"
        self.signed = []

    def sign_wire(self, raw: bytes) -> bytes:
        self.signed.append(raw)
        return raw

    def sign_new_transaction(self, ix):
        return b"probe"

    @property
    def pubkey_obj(self):
        return None


def _entry_tx(owner=OWNER, sol_delta=375_000_000, out_tokens=12_345_678, fee=5_000):
    """A confirmed BUY: wallet SOL decreases by ``sol_delta``, tokens arrive."""
    return {
        "slot": 9,
        "transaction": {
            "message": {"accountKeys": [owner]},
            "meta": {
                "err": None,
                "fee": fee,
                "logMessages": ["Program log: swap"],
                "preBalances": [3_000_000_000],
                "postBalances": [3_000_000_000 - sol_delta],
                "preTokenBalances": [],
                "postTokenBalances": [
                    {"mint": MINT, "owner": owner,
                     "uiTokenAmount": {"amount": str(out_tokens), "decimals": 0}}
                ],
            },
        },
    }


def _exit_tx(owner=OWNER, sol_gain=10_000_000, in_tokens=12_345_678, fee=5_000):
    """A confirmed SELL: wallet SOL increases, tokens leave."""
    return {
        "slot": 11,
        "transaction": {
            "message": {"accountKeys": [owner]},
            "meta": {
                "err": None,
                "fee": fee,
                "logMessages": ["Program log: swap"],
                "preBalances": [3_000_000_000],
                "postBalances": [3_000_000_000 + sol_gain],
                "preTokenBalances": [
                    {"mint": MINT, "owner": owner,
                     "uiTokenAmount": {"amount": str(in_tokens), "decimals": 0}}
                ],
                "postTokenBalances": [],
            },
        },
    }


def _backend(**kw) -> LiveSolanaBackend:
    return LiveSolanaBackend(
        _Cfg(),
        wallet=kw.get("wallet") or _Wallet(),
        rpc=kw.get("rpc") or _Rpc(),
        provider=kw.get("provider") or _Provider(),
    )


# ---------------------------------------------------------------------------
# idempotency store
# ---------------------------------------------------------------------------


def test_intent_store_survives_a_restart(tmp_path: Path):
    path = tmp_path / "intents.jsonl"
    store = ExecutionIntentStore(path)
    intent = store.reserve(logical_key="ENTRY:X", side="ENTRY", mint="X",
                           requested_amount=123)
    store.transition(intent.execution_id, READY)
    store.transition(intent.execution_id, SIGNING)
    store.transition(intent.execution_id, BROADCASTED, signature="SIGABC")

    # A brand-new store (a restarted process) sees the same unresolved intent.
    fresh = ExecutionIntentStore(path)
    found = fresh.find("ENTRY:X")
    assert found is not None
    assert found.state == BROADCASTED
    assert found.signature == "SIGABC"
    assert fresh.has_unresolved()
    with pytest.raises(DuplicateExecutionError):
        fresh.reserve(logical_key="ENTRY:X", side="ENTRY", mint="X")


def test_intent_store_confirmed_is_terminal_and_reusable(tmp_path: Path):
    store = ExecutionIntentStore(tmp_path / "i.jsonl")
    a = store.reserve(logical_key="EXIT:M", side="EXIT", mint="M")
    store.transition(a.execution_id, READY)
    store.transition(a.execution_id, SIGNING)
    store.transition(a.execution_id, BROADCASTED, signature="S1")
    store.transition(a.execution_id, CONFIRMED, signature="S1")
    assert not store.has_unresolved()
    # A terminal intent does not block a new reservation of the same key.
    b = store.reserve(logical_key="EXIT:M", side="EXIT", mint="M")
    assert b.execution_id != a.execution_id


def test_intent_store_rejects_an_illegal_transition(tmp_path: Path):
    store = ExecutionIntentStore(tmp_path / "i.jsonl")
    a = store.reserve(logical_key="K", side="ENTRY", mint="M")
    with pytest.raises(IntentStateError):
        store.transition(a.execution_id, CONFIRMED)  # CREATED -> CONFIRMED illegal


def test_intent_store_unknown_is_not_terminal(tmp_path: Path):
    store = ExecutionIntentStore(tmp_path / "i.jsonl")
    a = store.reserve(logical_key="K", side="ENTRY", mint="M")
    store.transition(a.execution_id, READY)
    store.transition(a.execution_id, SIGNING)
    store.transition(a.execution_id, BROADCASTED)
    store.transition(a.execution_id, UNKNOWN, detail="confirmation timeout")
    assert store.has_unresolved()
    assert not store.find("K").terminal


def test_intent_store_tolerates_a_truncated_final_line(tmp_path: Path):
    path = tmp_path / "i.jsonl"
    store = ExecutionIntentStore(path)
    a = store.reserve(logical_key="K", side="ENTRY", mint="M")
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"execution_id": "partial", "state": "BR')  # hard kill
    fresh = ExecutionIntentStore(path)
    assert fresh.find("K") is not None
    assert fresh.get(a.execution_id) is not None


# ---------------------------------------------------------------------------
# fee accounting (B9)
# ---------------------------------------------------------------------------


def test_fee_is_read_from_meta_not_the_balance_delta():
    tx = _entry_tx(sol_delta=375_000_000, fee=7_500)
    out = reconcile_transaction(tx, "sig", "confirmed", OWNER, MINT)
    assert out.fee_lamports == 7_500
    assert out.net_lamports_delta == 375_000_000
    assert out.fee_lamports < out.net_lamports_delta
    # The old bug: fee == delta (~4500x overstated).
    assert out.fee_lamports != out.net_lamports_delta


def test_absurd_fee_is_refused_not_clamped():
    tx = _entry_tx(fee=FEE_SANITY_MAX_LAMPORTS + 1)
    out = reconcile_transaction(tx, "sig", "confirmed", OWNER, MINT)
    assert out.fee_lamports is None, "an implausible fee must not be reported"
    assert out.sanity_warnings


# ---------------------------------------------------------------------------
# PnL units (B5)
# ---------------------------------------------------------------------------


def _fill(signature="SIG", side="ENTRY", in_amt=None, out_amt=None, fee=5_000):
    return RealFillRecord(
        signature=signature, side=side, commitment="confirmed",
        actual_input_amount=in_amt, actual_output_amount=out_amt,
        fee_lamports=fee, execution_price=None, quote_price=None,
    )


def test_close_position_pnl_is_usd_minus_usd_not_sol_minus_usd():
    """
    The exact regression from the audit: a $50 position that exits at roughly
    break-even booked about -49.875 because it subtracted a USD notional from a
    SOL quantity. With correct units the same exit is near break-even.
    """
    acct = RealAccount()
    acct.observe_wallet(lamports=int(2 * LAMPORTS_PER_SOL), address=OWNER,
                        sol_price_usd=200.0, now=BASE_T)
    acct.register_entry(
        mint=MINT, symbol="T", fill=_fill(in_amt=250_000_000, out_amt=1_000_000),
        token_amount=1_000_000, entry_price_usd=0.00025, size_usd=50.0,
        now=BASE_T, sol_price_usd=200.0,
    )
    # Exit with 0.25 SOL out (0.25 * $200 = $50): break-even.
    closed = acct.close_position(
        mint=MINT,
        fill=_fill(signature="SELL", side="EXIT", in_amt=1_000_000,
                   out_amt=250_000_000),
        exit_price_usd=0.00025, reason="take_profit", now=BASE_T + 60,
        sol_price_usd=200.0,
    )
    assert closed.status == POSITION_CLOSED
    # $50 proceeds - $50 notional - $0.001 entry fee ~= -0.001, NOT -49.875.
    assert closed.realized_pnl_usd == pytest.approx(-0.001, abs=0.005)
    assert abs(closed.realized_pnl_usd) < 1.0, "SOL must never be subtracted from USD"
    assert closed.exit_price_usd == 0.00025
    assert acct.realized_pnl_usd == pytest.approx(-0.001, abs=0.005)


def test_close_position_refuses_without_a_sol_price():
    acct = RealAccount()
    acct.register_entry(mint=MINT, symbol="T",
                        fill=_fill(in_amt=1, out_amt=1), token_amount=1,
                        entry_price_usd=1.0, size_usd=1.0, now=BASE_T)
    with pytest.raises(ValueError, match="SOL/USD"):
        acct.close_position(
            mint=MINT, fill=_fill(signature="S", side="EXIT", out_amt=1),
            exit_price_usd=1.0, reason="x", now=BASE_T + 1, sol_price_usd=None,
        )


def test_real_account_round_trips_through_state():
    acct = RealAccount()
    acct.register_entry(mint=MINT, symbol="T",
                        fill=_fill(in_amt=1, out_amt=2), token_amount=2,
                        entry_price_usd=0.5, size_usd=1.0, now=BASE_T,
                        sol_price_usd=200.0, exit_style=None)
    restored = RealAccount.from_state(acct.to_state())
    assert MINT in restored.positions
    assert restored.positions[MINT].token_amount == 2
    assert restored.positions[MINT].entry_signature == "SIG"


# ---------------------------------------------------------------------------
# engine-level REAL lifecycle
# ---------------------------------------------------------------------------


class _Obs:
    def __init__(self, price, observed_at):
        self.price_usd = price
        self.observed_at = observed_at
        self.market_cap_usd = 1_000_000.0
        self.liquidity_usd = 50_000.0

    def freshness(self, now, stale_ms, unusable_ms):
        from autopaper.models import DataState
        return DataState.LIVE


class _History:
    def __len__(self):
        return 0

    def latest(self):
        return None

    def last_n(self, n):
        return []

    def prices(self, now):
        return []


class _Market:
    """
    Market-data double.

    Implements ``mark_price`` and ``latest`` as well as ``quote_async``, because
    the entry quote guard (added 2026-10-02) reads the mark through the same
    service rather than off the decision features. A double missing one of those
    is not a stricter double -- it just makes the guard fall back.
    """

    def __init__(self, price=0.00025):
        self.price = price

    async def quote_async(self, mint):
        return SimpleNamespace(observation=_Obs(self.price, BASE_T))

    def latest(self, mint):
        return _Obs(self.price, BASE_T)

    def mark_price(self, mint, now=None):
        return self.price, SimpleNamespace(value="LIVE")

    def history(self, mint):
        return _History()

    def register_known_price(self, mint, price):
        return None


def _gate():
    return SimpleNamespace(
        market_cap_usd=1_000_000.0, signal_quality=0.9,
        f_liq_to_mcap=1.0, status_code="L25", as_dict=lambda: {},
    )


def _report(decided_at=BASE_T, authorized=True):
    return SimpleNamespace(
        mint=MINT, symbol="T", candidate_id="cand1",
        v3c_authorized=authorized, decided_at=decided_at,
        v3c=_gate(), l25=None, exit_style=None,
    )


def _snapshot(received=BASE_T):
    return SimpleNamespace(
        mint=MINT, symbol="T", candidate_id="cand1",
        received_timestamp=received,
    )


def _sol_usd(price):
    """
    An offline SOL/USD REFERENCE provider.

    The canonical provider is a Jupiter PRICE read, not a swap quote, so these
    tests supply the price directly instead of expecting an executable route to
    be mistaken for a reference. Fully in-memory: no client, no socket.
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


def _engine(tmp_path, monkeypatch, *, market=None, backend=None):
    monkeypatch.setenv("WALLET_KEYPAIR_PATH", "/tmp/fake-id.json")
    monkeypatch.setenv("RPC_URL", "https://rpc.invalid")
    monkeypatch.setenv("I_UNDERSTAND_REAL_TRADING_RISKS", "YES")
    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    from autopaper.engine import PaperTrader
    from autopaper.modes import RunState, TradeMode

    clock = FakeClock(BASE_T)
    cfg = build_config(tmp_path)
    # The canonical SOL/USD REFERENCE is injected, never derived from a swap
    # quote. These fixtures used to get $200 out of the backend's WSOL->USDC
    # quote, which was the very defect this seam removes: a reference price
    # asked of an execution quoter. The price is now supplied where it belongs,
    # so the tests assert on behaviour and not on a coincidence.
    trader = PaperTrader(cfg, clock=clock, sol_usd_provider=_sol_usd(200.0))
    trader.market = market or _Market()
    trader._real_executor = RealExecutor(
        cfg, backend=backend or _backend(rpc=_Rpc(get_tx=_entry_tx()))
    )
    # A REAL entry is refused unless a fresh EXECUTABLE quote can be obtained
    # (the mark/quote guard added 2026-10-02). Supply one offline, priced at the
    # mark these fixtures use, so the entry is allowed for the reason under test
    # rather than for the absence of a guard.
    from tests.execguard_fakes import install_fake_executable_quote
    install_fake_executable_quote(
        trader, price_fn=lambda: float(trader.market.price), sol_usd=200.0)
    trader._build_book()
    token = trader.state.request_real_confirmation()
    trader.state.arm_real(confirmation=token, preflight_ok=True)
    trader.state.start()
    assert trader.state.trade_mode is TradeMode.REAL
    return trader


def test_real_buy_path_is_reachable_and_uses_wallet_equity(tmp_path, monkeypatch):
    """
    The exact blocker from the audit: without wallet observation, sizing refuses
    with "account equity is zero". With the observation wired in, the BUY path
    reaches a confirmed fill and books a REAL position.
    """
    rpc = _Rpc(get_tx=_entry_tx())
    trader = _engine(tmp_path, monkeypatch, backend=_backend(rpc=rpc))
    assert trader.real_account.equity_usd() is None  # nothing observed yet

    report = _report()
    features = SimpleNamespace(price=0.00025)
    asyncio.run(trader._execute_real_buy(report, _snapshot(), features))

    # The wallet was observed and used for sizing.
    assert trader.real_account.equity_usd() == pytest.approx(600.0, rel=1e-6)
    open_positions = trader.real_account.open_positions()
    assert len(open_positions) == 1, "a confirmed fill must book a REAL position"
    pos = open_positions[0]
    assert pos.size_usd == pytest.approx(75.0)  # 12.5% of $600
    assert pos.token_amount == 12_345_678
    assert rpc.sent, "the buy reached broadcast exactly once"
    assert trader.last_execution["outcome"] == "REAL_FILL=SUCCESS"


def test_real_buy_fails_closed_when_wallet_observation_fails(tmp_path, monkeypatch):
    rpc = _Rpc(balance=None, get_tx=_entry_tx())  # balance read raises
    trader = _engine(tmp_path, monkeypatch, backend=_backend(rpc=rpc))
    asyncio.run(
        trader._execute_real_buy(_report(), _snapshot(), SimpleNamespace(price=1e-4))
    )
    assert not rpc.sent, "no observation -> no broadcast"
    assert not trader.real_account.open_positions()
    assert trader.last_execution["outcome"] == "BUY_BLOCKED"
    assert trader.last_execution["reason"] == "wallet_observation_failed"


def test_real_buy_refuses_when_an_execution_may_be_in_flight(tmp_path, monkeypatch):
    rpc = _Rpc(get_tx=_entry_tx())
    trader = _engine(tmp_path, monkeypatch, backend=_backend(rpc=rpc))
    intent = trader.execution_intents.reserve(
        logical_key="ENTRY:earlier", side="ENTRY", mint="earlier"
    )
    trader.execution_intents.transition(intent.execution_id, READY)
    trader.execution_intents.transition(intent.execution_id, SIGNING)
    trader.execution_intents.transition(intent.execution_id, BROADCASTED,
                                        signature="PRIOR")
    asyncio.run(
        trader._execute_real_buy(_report(), _snapshot(), SimpleNamespace(price=1e-4))
    )
    assert not rpc.sent, "an unresolved execution must block a new order"
    assert not trader.real_account.open_positions()


def test_real_buy_fill_time_stale_gate_refuses_an_old_signal(tmp_path, monkeypatch):
    rpc = _Rpc(get_tx=_entry_tx())
    trader = _engine(tmp_path, monkeypatch, backend=_backend(rpc=rpc))
    # 60s-old signal, far beyond the 15s budget.
    old = SimpleNamespace(mint=MINT, symbol="T", candidate_id="cand1",
                          received_timestamp=BASE_T - 60)
    asyncio.run(
        trader._execute_real_buy(_report(), old, SimpleNamespace(price=1e-4))
    )
    assert not rpc.sent, "a stale signal must never reach broadcast"
    assert trader.last_execution["reason"] in (
        "stale_signal", "stale_quote", "entry_latency_exceeded"
    )


def _real_open_position(trader, *, size_usd=2.0, price=0.00025, tokens=12_345_678):
    # The entry swap leg must agree with ``size_usd``: $2.00 at $200/SOL is
    # 0.01 SOL = 10,000,000 lamports. It used to claim a 0.375 SOL leg against a
    # $2.00 size, which was only survivable while the cost basis was taken from
    # the PLANNED notional. Realized PnL is measured against what actually left
    # the wallet, so the fixture has to be coherent with itself.
    swap_lamports = int(round(size_usd / 200.0 * 1_000_000_000))
    fill = RealFillRecord(
        signature="ENTRY-SIG", side="ENTRY", commitment="confirmed",
        actual_input_amount=swap_lamports, actual_output_amount=tokens,
        fee_lamports=5_000,
    )
    trader.real_account.register_entry(
        mint=MINT, symbol="T", fill=fill, token_amount=tokens,
        entry_price_usd=price, size_usd=size_usd, now=BASE_T,
        sol_price_usd=200.0,
    )
    trader.save_real_state()


def test_monitor_once_closes_a_real_position_through_the_real_path(
    tmp_path, monkeypatch
):
    """B1: a REAL position must be sellable. monitor_once() reaches
    RealExecutor.close_position and realizes the PnL in USD."""
    rpc = _Rpc(get_tx=_exit_tx(sol_gain=10_000_000), token_balances={MINT: 12_345_678})
    # A crashed mark triggers the exit policy (disaster cap).
    trader = _engine(tmp_path, monkeypatch, market=_Market(price=1e-6),
                     backend=_backend(rpc=rpc))
    _real_open_position(trader)
    assert trader.real_account.open_positions()

    asyncio.run(trader.monitor_once())

    assert rpc.sent, "monitor_once must construct and broadcast a real SELL"
    assert not trader.real_account.open_positions(), "the position must be closed"
    closed = trader.real_account.positions[MINT]
    assert closed.status == POSITION_CLOSED
    assert closed.exit is not None, "a closed REAL position carries its exit fill"
    assert closed.realized_pnl_usd is not None
    # USD units: 0.01 SOL * $200 = $2 proceeds, not a SOL-USD mix.
    assert closed.realized_pnl_usd == pytest.approx(2.0 - 2.001, abs=0.01)
    real_exits = [r for r in trader.journal.trades_for("REAL")
                  if r.get("event") == "exit"]
    assert len(real_exits) == 1
    assert real_exits[0]["tx_signature"]


def test_restart_reconciliation_confirms_a_broadcast_and_restores_positions(
    tmp_path, monkeypatch
):
    rpc = _Rpc(get_tx=_exit_tx(), signature_statuses=[
        {"err": None, "confirmationStatus": "confirmed", "slot": 9}
    ], token_balances={MINT: 12_345_678})
    trader = _engine(tmp_path, monkeypatch, backend=_backend(rpc=rpc))
    _real_open_position(trader)
    intent = trader.execution_intents.reserve(
        logical_key="EXIT:" + MINT, side="EXIT", mint=MINT
    )
    trader.execution_intents.transition(intent.execution_id, READY)
    trader.execution_intents.transition(intent.execution_id, SIGNING)
    trader.execution_intents.transition(intent.execution_id, BROADCASTED,
                                        signature="PENDING-SIG")

    # A NEW process pointed at the same root.
    from autopaper.engine import PaperTrader
    cfg = build_config(tmp_path)
    restarted = PaperTrader(cfg, clock=FakeClock(BASE_T))
    restarted._real_executor = RealExecutor(cfg, backend=_backend(rpc=rpc))
    report = asyncio.run(restarted.reconcile_real_state())

    assert report["restored"] is True
    assert MINT in restarted.real_account.positions
    # The broadcast was resolved against the chain, not re-sent.
    assert report["intents_confirmed"] == 1
    assert not restarted.execution_intents.has_unresolved()
    assert not rpc.sent, "reconciliation must not broadcast"


def test_reconciliation_fails_closed_on_a_chain_mismatch(tmp_path, monkeypatch):
    # Wallet holds NO tokens, but local state has an OPEN position.
    rpc = _Rpc(token_balances={MINT: 0})
    trader = _engine(tmp_path, monkeypatch, backend=_backend(rpc=rpc))
    _real_open_position(trader)
    from autopaper.engine import PaperTrader
    cfg = build_config(tmp_path)
    restarted = PaperTrader(cfg, clock=FakeClock(BASE_T))
    restarted._real_executor = RealExecutor(cfg, backend=_backend(rpc=rpc))
    report = asyncio.run(restarted.reconcile_real_state())
    assert report["mismatches"], "chain truth disagreeing must be surfaced"
    assert report["consistent"] is False
    assert restarted._real_state_consistent is False


def test_real_sell_refuses_a_stale_exit_quote(tmp_path, monkeypatch):
    rpc = _Rpc(get_tx=_exit_tx())
    trader = _engine(tmp_path, monkeypatch, market=_Market(),
                     backend=_backend(rpc=rpc))
    _real_open_position(trader)
    from autopaper.exits import ExitDecision
    from autopaper.models import ExitStyle

    # The market observation is 60s older than `now`.
    old_obs = _Obs(1e-6, BASE_T - 60)
    trader.market = SimpleNamespace(
        quote_async=lambda m: _as_coro(SimpleNamespace(observation=old_obs)),
        history=lambda m: _History(),
    )
    decision = ExitDecision("EXIT", "test", ExitStyle.DISASTER_CAP,
                            trigger="disaster_cap")

    async def _run():
        trader._current_stages = None
        await trader._execute_real_sell(
            trader.real_account.positions[MINT], decision, BASE_T
        )

    asyncio.run(_run())
    assert not rpc.sent, "a stale exit quote must not be broadcast"
    assert trader.last_execution["outcome"] == "SELL_BLOCKED"
    assert trader.last_execution["reason"] == "STALE_QUOTE"


def _as_coro(value):
    async def _inner():
        return value
    return _inner()


def test_executor_records_unknown_on_a_confirmation_timeout(tmp_path):
    """An ambiguous broadcast must leave a durable UNKNOWN, never a silence,
    and the signature must be recorded the moment it is known."""
    store = ExecutionIntentStore(tmp_path / "i.jsonl")
    rpc = _Rpc(confirm={"status": "timeout", "error": "timeout",
                        "signature": "Sig111"})
    fill = asyncio.run(
        _backend(rpc=rpc).execute_swap(
            "ENTRY", WSOL_MINT, MINT, 375_000_000, 500,
            intent_store=store, logical_key="ENTRY:" + MINT,
        )
    )
    assert not fill.filled
    assert fill.signature == "Sig111"
    intent = store.find("ENTRY:" + MINT)
    assert intent.state == UNKNOWN
    assert intent.signature == "Sig111"
    assert store.has_unresolved()
    with pytest.raises(DuplicateExecutionError):
        store.reserve(logical_key="ENTRY:" + MINT, side="ENTRY", mint=MINT)


def test_executor_records_failed_on_a_simulation_failure(tmp_path):
    store = ExecutionIntentStore(tmp_path / "i.jsonl")
    rpc = _Rpc(simulate_ok=False, get_tx=_entry_tx())
    fill = asyncio.run(
        _backend(rpc=rpc).execute_swap(
            "ENTRY", WSOL_MINT, MINT, 375_000_000, 500,
            intent_store=store, logical_key="ENTRY:" + MINT,
        )
    )
    # See test_executor_real.py: the taxonomy is specific now, so assert the
    # invariant (a simulation refusal) instead of one opaque status string.
    assert fill.status.startswith("SIMULATION_FAILED"), fill.status
    assert not rpc.sent, "simulation must precede broadcast"
    assert store.find("ENTRY:" + MINT).state == FAILED
    assert not store.has_unresolved()


def test_kill_does_not_orphan_an_open_real_position(tmp_path, monkeypatch):
    """KILL stops NEW entries; an existing REAL position remains manageable and
    is still closed through the REAL path."""
    rpc = _Rpc(get_tx=_exit_tx(), token_balances={MINT: 12_345_678})
    trader = _engine(tmp_path, monkeypatch, market=_Market(price=1e-6),
                     backend=_backend(rpc=rpc))
    trader.state.emergency_stop_now()
    assert trader.state.real_trading_enabled is False  # fired first
    _real_open_position(trader)
    asyncio.run(trader.monitor_once())
    assert rpc.sent, "KILL must not prevent an existing position being sold"
    assert not trader.real_account.open_positions()


def test_real_buy_records_stage_telemetry(tmp_path, monkeypatch):
    from autopaper.latency import StageRecorder

    rpc = _Rpc(get_tx=_entry_tx())
    trader = _engine(tmp_path, monkeypatch, backend=_backend(rpc=rpc))
    stages = StageRecorder()
    trader._current_stages = stages
    asyncio.run(
        trader._execute_real_buy(_report(), _snapshot(), SimpleNamespace(price=1e-4))
    )
    stamps = stages.stamps
    # REAL-specific stages are individually timed, not one opaque span.
    for expected in (
        "real_wallet_observation_start", "real_wallet_observation_end",
        "real_pretrade_start", "real_stale_gate_start", "exec_intent_reserve_start",
        "tx_build_start", "exec_signing_start", "exec_simulate_start",
        "exec_broadcast_start", "exec_confirm_start", "exec_reconcile_start",
    ):
        assert expected in stamps, f"missing REAL stage {expected}"
    assert stages.ordering_violations() == []
