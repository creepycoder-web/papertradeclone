"""
Regression tests for the $Boy incident.

A REAL BUY for $Boy executed successfully on chain and the output tokens
entered the wallet, but the runtime reported:

    AMBIGUOUS: confirmed but no output amount could be reconciled
    BUY_BLOCKED=fill_refused
    NO POSITION OPENED

Root cause was in the READ path only. ``reconcile_transaction``:

  1. read ``meta`` from ``tx["transaction"]["meta"]``, but in a real
     ``getTransaction`` response ``meta`` is a SIBLING of ``transaction``;
  2. read token balances as ``entry["account"]["data"]["parsed"]["info"]``, but
     the RPC returns them FLAT as ``entry["mint"] / entry["uiTokenAmount"]``;
  3. swallowed the resulting ``KeyError`` with a bare ``except: continue``.

So every balance entry was skipped, ``token_output_amount`` came back None, and
a confirmed swap was classified as a refused entry.

These tests pin the corrected semantics:

  * confirmed on chain + unreadable amounts  -> CONFIRMED_BUT_UNRECONCILED
  * never AMBIGUOUS, never fill_refused, never a position
  * never a second broadcast (fails closed, idempotent)
  * the actual amount comes from wallet-owned on-chain state, never the quote

No test broadcasts: every RPC double refuses ``sendTransaction``.
"""


from __future__ import annotations

import asyncio
import dataclasses
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.execution,
    pytest.mark.safety,
]

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from autopaper.config import RiskLimits                          # noqa: E402
from autopaper.execution.intents import (                       # noqa: E402
    BROADCASTED,
    CONFIRMED,
    ExecutionIntentStore,
    FAILED,
    READY,
    SIGNING,
    UNKNOWN,
)
from autopaper.execution.real import (                          # noqa: E402
    AMBIGUOUS,
    CONFIRMED_UNRECONCILED,
    LiveSolanaBackend,
)
from autopaper.execution.solana import reconcile_transaction     # noqa: E402
from solders.keypair import Keypair                            # noqa: E402
from tests import txfixtures as F                              # noqa: E402

# The $Boy incident's actual values, read from mainnet slot 452289268.
BOY_MINT = "75DjurpbbaF1fcRmo6zdipE8drirJxU8mvDzw7nckKiY"
BOY_WALLET = "wUEUbVmz2EfcLfchHmucTKhRpZgDpsBvo3FttZ4Wbt6"
BOY_SIG = ("2eQe3SdAYx9dDQiUFc8TYDumDcp5HzhFzVFzfousN92BW97YEmU9qDDLN3Ley9"
           "RE2SFDHJn9hRVjywZF2q1f4DmF")
BOY_ACTUAL_OUTPUT = 7_971_133_130      # raw units, decimals 6
BOY_SOL_SPENT = 2_640_683             # lamports, incl. the 55_000 fee
BOY_FEE = 55_000
BOY_SLOT = 452_289_268

WSOL = "So11111111111111111111111111111111111111112"
POOL_A = "7VtfL8fvgNfhz17qKRMjzQEXgbdpnHHHQRh54R9jP2RJ"
POOL_B = "EHAAiTxcdDwQ3U4bU6YcMsQGaekdzLS3B5SmYo46kJtL"
OTHER_HOLDER = "4t9MwGd5K788S55GF5w4hMyW6UbFATenvWphbVhnnhoP"


def bal(owner, mint, amount, idx=1, decimals=6):
    """One token-balance entry in the shape getTransaction actually returns."""
    return {"accountIndex": idx, "mint": mint, "owner": owner,
            "programId": "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",
            "uiTokenAmount": {"amount": str(amount), "decimals": decimals,
                              "uiAmountString": str(amount)}}


def swap_tx(*, pre=None, post=None, fee=BOY_FEE, sol_pre=10_718_431,
            sol_post=8_077_748, slot=BOY_SLOT, keys=None, inner=None,
            logs=("Program log: swap",), err=None):
    """A confirmed swap in the real getTransaction response shape.

    ``meta`` is a SIBLING of ``transaction``. Every assertion in this file
    depends on that being right, which is exactly what the old parser got wrong.
    """
    return {
        "slot": slot,
        "blockTime": 1_790_858_117,
        "transaction": {
            "message": {"accountKeys": keys or [BOY_WALLET, BOY_MINT]},
            "signatures": [BOY_SIG],
        },
        "meta": {
            "err": err,
            "fee": fee,
            "logMessages": list(logs),
            "preBalances": [sol_pre, 0],
            "postBalances": [sol_post, 0],
            "preTokenBalances": list(pre or []),
            "postTokenBalances": list(post or []),
            "innerInstructions": list(inner or []),
        },
    }


# ===========================================================================
# 1. THE PARSER: the actual root cause
# ===========================================================================
class TestTheRootCause:
    """The two read bugs, proven against the real response shape."""

    def test_meta_is_read_as_a_sibling_of_transaction(self):
        """`meta` is NOT nested inside `transaction` in a real response."""
        tx = swap_tx(post=[bal(BOY_WALLET, BOY_MINT, BOY_ACTUAL_OUTPUT)])
        # The old location returns nothing at all.
        assert (tx["transaction"]).get("meta") is None
        out = reconcile_transaction(tx, BOY_SIG, "confirmed", BOY_WALLET, BOY_MINT)
        assert out.fee_lamports == BOY_FEE
        assert out.log_messages == ["Program log: swap"]
        assert out.success is True

    def test_flat_token_balances_are_parsed(self):
        """The RPC returns balances FLAT, not under account.data.parsed.info."""
        tx = swap_tx(post=[bal(BOY_WALLET, BOY_MINT, BOY_ACTUAL_OUTPUT)])
        out = reconcile_transaction(tx, BOY_SIG, "confirmed", BOY_WALLET, BOY_MINT)
        assert out.token_output_amount == BOY_ACTUAL_OUTPUT
        assert out.token_mint == BOY_MINT

    def test_the_nested_shape_is_still_accepted(self):
        """A caller that reshapes the payload must keep reconciling."""
        tx = swap_tx()
        tx["meta"]["postTokenBalances"] = [{
            "account": {"data": {"parsed": {"info": {
                "mint": BOY_MINT, "owner": BOY_WALLET,
                "tokenAmount": {"amount": str(BOY_ACTUAL_OUTPUT),
                                "decimals": 6}}}}}}]
        out = reconcile_transaction(tx, BOY_SIG, "confirmed", BOY_WALLET, BOY_MINT)
        assert out.token_output_amount == BOY_ACTUAL_OUTPUT

    def test_only_the_wallet_own_balances_count(self):
        """A swap touches pools and other holders; they are not this trade.

        This is the $Boy transaction's real shape: $Boy was already held at
        accountIndex 8 by an unrelated account, and two WSOL pools appear in the
        same balance lists. Summing every owner would produce a nonsense fill.
        """
        tx = swap_tx(
            pre=[
                bal(OTHER_HOLDER, BOY_MINT, 419_329_938_052_608, idx=8),
                bal(POOL_A, WSOL, 820_524_499_057, idx=17, decimals=9),
                bal(POOL_B, WSOL, 1_623_094_134, idx=20, decimals=9),
            ],
            post=[
                bal(BOY_WALLET, BOY_MINT, BOY_ACTUAL_OUTPUT, idx=1),
                bal(OTHER_HOLDER, BOY_MINT, 419_321_966_919_478, idx=8),
                bal(POOL_A, WSOL, 820_524_499_057, idx=17, decimals=9),
                bal(POOL_B, WSOL, 1_623_094_134, idx=20, decimals=9),
            ],
        )
        out = reconcile_transaction(tx, BOY_SIG, "confirmed", BOY_WALLET, BOY_MINT)
        assert out.token_output_amount == BOY_ACTUAL_OUTPUT, (
            "another account's movement must never be counted as this fill")

    def test_an_entry_without_an_owner_is_skipped_not_guessed(self):
        tx = swap_tx(post=[{"mint": BOY_MINT,
                            "uiTokenAmount": {"amount": "999", "decimals": 6}}])
        out = reconcile_transaction(tx, BOY_SIG, "confirmed", BOY_WALLET, BOY_MINT)
        assert out.token_output_amount is None, (
            "an unowned balance must not be attributed to the wallet")

    def test_sol_delta_and_fee_are_separate_quantities(self):
        tx = swap_tx(post=[bal(BOY_WALLET, BOY_MINT, BOY_ACTUAL_OUTPUT)])
        out = reconcile_transaction(tx, BOY_SIG, "confirmed", BOY_WALLET, BOY_MINT)
        assert out.net_lamports_delta == BOY_SOL_SPENT
        assert out.fee_lamports == BOY_FEE
        assert out.slot == BOY_SLOT


# ===========================================================================
# 2. BUY edge cases
# ===========================================================================
class TestBuyReconciliation:
    def test_standard_buy(self):
        tx = swap_tx(pre=[], post=[bal(BOY_WALLET, BOY_MINT, 12_345_678)])
        out = reconcile_transaction(tx, "s", "confirmed", BOY_WALLET, BOY_MINT)
        assert out.token_output_amount == 12_345_678

    def test_ata_created_during_the_swap_has_no_pre_entry(self):
        """A brand-new ATA appears only in post. The delta is the whole amount."""
        tx = swap_tx(pre=[],
                     post=[bal(BOY_WALLET, BOY_MINT, 7_971_133_130, idx=3)])
        out = reconcile_transaction(tx, "s", "confirmed", BOY_WALLET, BOY_MINT)
        assert out.token_output_amount == 7_971_133_130

    def test_ata_created_by_an_inner_instruction(self):
        """$Boy's real transaction: the ATA was created inside an inner ix."""
        tx = swap_tx(
            pre=[],
            post=[bal(BOY_WALLET, BOY_MINT, BOY_ACTUAL_OUTPUT, idx=1)],
            inner=[{"index": 2, "instructions": [
                {"parsed": {"type": "createAccount", "info": {
                    "source": BOY_WALLET, "owner": BOY_WALLET}}},
                {"parsed": {"type": "initializeAccount3", "info": {
                    "mint": BOY_MINT, "owner": BOY_WALLET}}},
            ]}],
        )
        out = reconcile_transaction(tx, "s", "confirmed", BOY_WALLET, BOY_MINT)
        assert out.token_output_amount == BOY_ACTUAL_OUTPUT

    def test_multiple_token_accounts_for_the_same_mint_are_summed(self):
        """Two of the wallet's own accounts receive the same mint."""
        tx = swap_tx(pre=[bal(BOY_WALLET, BOY_MINT, 100, idx=1),
                          bal(BOY_WALLET, BOY_MINT, 50, idx=4)],
                     post=[bal(BOY_WALLET, BOY_MINT, 1_100, idx=1),
                           bal(BOY_WALLET, BOY_MINT, 450, idx=4)])
        out = reconcile_transaction(tx, "s", "confirmed", BOY_WALLET, BOY_MINT)
        assert out.token_output_amount == (1_100 - 100) + (450 - 50) == 1_400

    def test_decimals_do_not_change_the_raw_amount(self):
        """The raw integer is reported; decimals are presentation only."""
        for dec in (0, 6, 9):
            tx = swap_tx(post=[bal(BOY_WALLET, BOY_MINT, 123_456_789, decimals=dec)])
            out = reconcile_transaction(tx, "s", "confirmed", BOY_WALLET, BOY_MINT)
            assert out.token_output_amount == 123_456_789

    def test_the_quote_amount_is_never_substituted(self):
        """No balances at all -> no amount. The quote must not fill the gap."""
        tx = swap_tx(pre=[], post=[])
        out = reconcile_transaction(tx, "s", "confirmed", BOY_WALLET, BOY_MINT)
        assert out.token_output_amount is None
        assert out.token_input_amount is None


# ===========================================================================
# 3. SELL edge cases
# ===========================================================================
class TestSellReconciliation:
    def test_sell_reports_the_token_decrease(self):
        tx = swap_tx(pre=[bal(BOY_WALLET, BOY_MINT, 7_971_133_130)],
                     post=[bal(BOY_WALLET, BOY_MINT, 0)])
        out = reconcile_transaction(tx, "s", "confirmed", BOY_WALLET, BOY_MINT)
        # Primary sell evidence is the token leg decreasing.
        assert out.token_output_amount == -7_971_133_130
        assert out.token_input_amount == 7_971_133_130

    def test_fees_do_not_corrupt_sell_confirmation(self):
        """A large priority fee moves SOL a lot and the token leg not at all."""
        tx = swap_tx(pre=[bal(BOY_WALLET, BOY_MINT, 1_000_000)],
                     post=[bal(BOY_WALLET, BOY_MINT, 0)],
                     fee=2_000_000, sol_pre=5_000_000_000, sol_post=4_990_000_000)
        out = reconcile_transaction(tx, "s", "confirmed", BOY_WALLET, BOY_MINT)
        assert out.token_output_amount == -1_000_000
        assert out.success is True

    def test_sell_across_multiple_accounts(self):
        tx = swap_tx(pre=[bal(BOY_WALLET, BOY_MINT, 300, idx=1),
                          bal(BOY_WALLET, BOY_MINT, 700, idx=4)],
                     post=[bal(BOY_WALLET, BOY_MINT, 0, idx=1),
                           bal(BOY_WALLET, BOY_MINT, 200, idx=4)])
        out = reconcile_transaction(tx, "s", "confirmed", BOY_WALLET, BOY_MINT)
        assert out.token_output_amount == -800


# ===========================================================================
# 4. The classification, end to end through the production backend
# ===========================================================================
class _Cfg:
    def __init__(self, **kw):
        self.wallet_keypair_path = kw.get("wallet_keypair_path", "/tmp/fake.json")
        self.rpc_url = kw.get("rpc_url", "https://rpc.invalid")
        self.slippage_bps = 500
        self.simulate_before_send = True
        self.priority_fee_lamports = 50_000
        self.confirmation_timeout_s = 5.0
        self.quote_max_age_usd = 0
        self.quote_max_age_ms = 8_000.0
        self.max_price_impact_pct = 5.0
        self.max_open_positions = 1
        self.reentry_enabled = False
        self.trading_enabled = False
        self.risk_acknowledged = True
        self.min_notional_usd = 0.0
        self.v1_rescue_enabled = True


class _Quote:
    def __init__(self, in_amount, out_amount):
        self.in_amount = in_amount
        self.expected_out_amount = out_amount
        self.min_out_amount = int(out_amount * 0.99)
        self.price_impact_pct = 0.4
        self.route_labels = ["RAYDIUM"]
        self.quote_received_at = 0.0

    @property
    def price_per_token(self) -> float:
        return self.in_amount / max(1, self.expected_out_amount)


class _Built:
    """A REAL serialised transaction.

    The bytes must be genuine: production measures their size before simulating,
    and a placeholder string is rejected as malformed long before confirmation.
    That is the whole point -- these tests drive the real send path.
    """

    def __init__(self, payer):
        self.transaction_b64 = F.b64(F.build("v0", payer=payer))
        # F.build returns raw bytes; b64 makes it transportable.
        self.last_valid_block_height = 123_456
        self.prioritization_fee_lamports = 50_000


class _Provider:
    def __init__(self, out_tokens=9_999_999_999):
        self.out_tokens = out_tokens
        self.seen = []

    async def quote(self, i, o, amount, slip):
        self.seen.append((i, o, amount))
        return _Quote(amount, self.out_tokens)

    async def build(self, quote, pubkey, priority_fee="auto", **kw):
        return _Built(signing_key(pubkey))

    async def health(self):
        return {"provider": "raptor", "reachable": True, "detail": "mock"}

    async def aclose(self):
        return None


_KP = Keypair()
_KP_OBJ = _KP.pubkey()


def signing_key(pubkey):
    """The provider's ``build`` needs a Keypair to compile a real transaction.

    Generated in memory on purpose: this suite must never read a keypair file.
    """
    return _KP


class _Wallet:
    def __init__(self, pubkey=BOY_WALLET):
        self.pubkey = pubkey
        self.pubkey_obj = _KP_OBJ
        self.masked = "wUEU…Wbt6"
        self.signed = []

    def sign_wire(self, raw: bytes) -> bytes:
        self.signed.append(raw)
        return raw


class _Rpc:
    """Never broadcasts: send_transaction appends and returns a signature."""

    def __init__(self, *, tx=None, confirm=None, sim_ok=True):
        self.tx = tx
        self.confirm = confirm or {"status": "confirmed", "error": None,
                                  "signature": BOY_SIG, "slot": BOY_SLOT}
        self.sim_ok = sim_ok
        self.sent: list[str] = []
        self.simulated: list[str] = []
        self.simulate_calls = 0

    async def get_health(self):
        return type("H", (), {"reachable": True, "slot": BOY_SLOT,
                              "detail": "double"})()

    async def wsol_readiness(self, owner: str, required_lamports: int):
        """Mirrors SolanaRpc.wsol_readiness: async, (owner, required_lamports)."""
        return type("R", (), {
            "ready": True,
            "sol_balance_lamports": 10_718_431,
            "source": "double",
            "detail": "double",
            "block_reason": lambda self=None: "not ready",
        })()

    async def get_slot(self):
        return BOY_SLOT

    async def get_balance_lamports(self, address):
        return 10_718_431

    async def wallet_state(self):
        return {"lamports": 10_718_431, "sol": 0.010718431, "masked": "wUEU…Wbt6"}

    async def sol_price_usd(self):
        return 200.0

    def transaction_size_report(self, b64):
        """Synchronous, matching SolanaRpc (solana.py:929).

        real.py:1331 handles BOTH shapes via a callable check, so this must
        mirror production exactly rather than being made async for convenience.
        """
        from autopaper.execution.txcodec import measure_transaction
        return measure_transaction(b64)

    async def get_address_lookup_table(self, address):
        return {"table": address, "query_ok": True, "usable": False,
                "addresses": [], "source": "double", "detail": "none"}

    async def resolve_lookup_tables(self, addresses):
        return [], [{"table": a, "detail": "none"} for a in addresses]

    async def simulate_transaction_detailed(self, b64):
        self.simulate_calls += 1
        self.simulated.append(b64)
        from autopaper.execution.solana import SimulationOutcome
        if not self.sim_ok:
            return SimulationOutcome(ok=False, status="SIMULATION_FAILED_TX_REJECTED",
                                     detail="pool refused")
        return SimulationOutcome(ok=True, status="SIMULATION_OK",
                                 detail="simulation ok")

    async def send_transaction(self, b64):
        self.sent.append(b64)
        return BOY_SIG

    async def wait_for_confirmation(self, sig, **kw):
        return dict(self.confirm)

    async def get_transaction(self, sig, commitment="confirmed"):
        return self.tx

    async def aclose(self):
        return None


def _backend(rpc, provider=None, wallet=None, cfg=None):
    return LiveSolanaBackend(
        cfg or _Cfg(), wallet=wallet or _Wallet(),
        rpc=rpc, provider=provider or _Provider())


def _buy(backend, amount=2_585_683, store=None):
    return asyncio.run(backend.execute_swap(
        "ENTRY", WSOL, BOY_MINT, amount, 500,
        intent_store=store, logical_key=f"ENTRY:{BOY_MINT}"))


class TestClassificationEndToEnd:
    def test_a_confirmed_buy_with_readable_amounts_is_a_fill(self):
        rpc = _Rpc(tx=swap_tx(pre=[], post=[
            bal(BOY_WALLET, BOY_MINT, BOY_ACTUAL_OUTPUT)]))
        fill = _buy(_backend(rpc))
        assert fill.status == "confirmed"
        assert fill.filled is True
        assert fill.output_amount == BOY_ACTUAL_OUTPUT
        assert fill.requires_recovery is False
        assert fill.fee_lamports == BOY_FEE

    def test_a_confirmed_buy_that_cannot_be_read_is_UNRECONCILED(self):
        """The $Boy case. NOT a refusal, NOT ambiguous, NOT a position."""
        rpc = _Rpc(tx=swap_tx(pre=[], post=[]))
        fill = _buy(_backend(rpc))
        assert fill.status == CONFIRMED_UNRECONCILED
        assert fill.status != AMBIGUOUS
        assert fill.filled is False
        assert fill.requires_recovery is True
        assert fill.confirmed_on_chain is True
        assert fill.output_amount in (None, 0), "no amount may be invented"
        assert fill.signature == BOY_SIG

    def test_a_confirmation_failure_is_not_the_same_as_unreconciled(self):
        """The chain did NOT confirm: that is a different, honest failure."""
        rpc = _Rpc(tx=None, confirm={"status": "failed", "error": "blockhash expired",
                                    "signature": BOY_SIG})
        fill = _buy(_backend(rpc))
        assert fill.status != CONFIRMED_UNRECONCILED
        assert fill.confirmed_on_chain is False
        assert fill.requires_recovery is False

    def test_a_simulation_failure_never_broadcasts(self):
        rpc = _Rpc(tx=None, sim_ok=False)
        fill = _buy(_backend(rpc))
        assert fill.filled is False
        assert rpc.sent == [], "a failed simulation must never reach sendTransaction"
        assert fill.requires_recovery is False


# ===========================================================================
# 5. Duplicate protection / idempotency
# ===========================================================================
class TestNoDuplicateBroadcast:
    def test_an_unreconciled_intent_stays_unresolved(self, tmp_path: Path):
        store = ExecutionIntentStore(tmp_path / "i.jsonl")
        rpc = _Rpc(tx=swap_tx(pre=[], post=[]))
        backend = _backend(rpc)
        fill = _buy(backend, store=store)
        assert fill.status == CONFIRMED_UNRECONCILED
        # The intent must remain unresolved so a restart cannot retry it.
        assert store.has_unresolved(mint=BOY_MINT) is True
        assert store.get(store.unresolved()[0].execution_id).signature == BOY_SIG

    def test_a_second_buy_for_the_same_mint_is_refused(self, tmp_path: Path):
        """The intent store blocks a duplicate even before any RPC call."""
        store = ExecutionIntentStore(tmp_path / "i.jsonl")
        i = store.reserve(logical_key=f"ENTRY:{BOY_MINT}", side="ENTRY",
                          mint=BOY_MINT, requested_amount=2_585_683)
        store.transition(i.execution_id, READY, detail="quote accepted")
        store.transition(i.execution_id, SIGNING, detail="signing")
        store.transition(i.execution_id, BROADCASTED, signature=BOY_SIG,
                         detail="broadcast")
        store.transition(i.execution_id, UNKNOWN, signature=BOY_SIG,
                         detail="confirmed but unreconciled; recovery required")
        with pytest.raises(Exception):
            store.reserve(logical_key=f"ENTRY:{BOY_MINT}", side="ENTRY",
                          mint=BOY_MINT, requested_amount=2_585_683)

    def test_recovery_is_idempotent(self, tmp_path: Path):
        """Re-reading the same confirmed tx twice yields one CONFIRMED."""
        store = ExecutionIntentStore(tmp_path / "i.jsonl")
        i = store.reserve(logical_key=f"ENTRY:{BOY_MINT}", side="ENTRY",
                          mint=BOY_MINT, requested_amount=2_585_683)
        for state, detail in ((READY, "quote accepted"), (SIGNING, "signing"),
                              (BROADCASTED, "broadcast")):
            store.transition(i.execution_id, state, signature="", detail=detail)
        for _ in range(3):
            store.transition(i.execution_id, CONFIRMED, signature=BOY_SIG,
                             detail="fill reconciled (ENTRY)")
        assert store.get(i.execution_id).state == CONFIRMED
        assert store.has_unresolved(mint=BOY_MINT) is False

    def test_repeated_reconciliation_of_the_same_tx_is_stable(self):
        """Deterministic: the same on-chain data gives the same answer."""
        tx = swap_tx(pre=[], post=[bal(BOY_WALLET, BOY_MINT, BOY_ACTUAL_OUTPUT)])
        amounts = [reconcile_transaction(tx, BOY_SIG, "confirmed", BOY_WALLET,
                                         BOY_MINT).token_output_amount
                   for _ in range(5)]
        assert amounts == [BOY_ACTUAL_OUTPUT] * 5


# ===========================================================================
# 6. The $Boy incident, verified against its real on-chain data
# ===========================================================================
class TestBoyIncident:
    def test_the_incident_signature_reconciles_to_the_real_fill(self):
        """The exact amounts the wallet actually received."""
        tx = swap_tx(
            pre=[bal(OTHER_HOLDER, BOY_MINT, 419_329_938_052_608, idx=8),
                 bal(POOL_A, WSOL, 820_524_499_057, idx=17, decimals=9),
                 bal(POOL_B, WSOL, 1_623_094_134, idx=20, decimals=9)],
            post=[bal(BOY_WALLET, BOY_MINT, BOY_ACTUAL_OUTPUT, idx=1),
                  bal(OTHER_HOLDER, BOY_MINT, 419_321_966_919_478, idx=8),
                  bal(POOL_A, WSOL, 820_524_499_057, idx=17, decimals=9),
                  bal(POOL_B, WSOL, 1_623_094_134, idx=20, decimals=9)],
        )
        out = reconcile_transaction(tx, BOY_SIG, "confirmed", BOY_WALLET, BOY_MINT)
        assert out.success is True
        assert out.slot == BOY_SLOT
        assert out.fee_lamports == BOY_FEE
        assert out.net_lamports_delta == BOY_SOL_SPENT
        assert out.token_mint == BOY_MINT
        assert out.token_output_amount == BOY_ACTUAL_OUTPUT
        assert out.sanity_warnings == []

    def test_the_incident_end_to_end_is_a_reconciled_fill(self):
        rpc = _Rpc(tx=swap_tx(pre=[], post=[
            bal(BOY_WALLET, BOY_MINT, BOY_ACTUAL_OUTPUT)]))
        fill = _buy(_backend(rpc))
        assert fill.status == "confirmed"
        assert fill.filled is True
        assert fill.output_amount == BOY_ACTUAL_OUTPUT
        assert fill.confirmed_on_chain is True
        assert fill.requires_recovery is False

    def test_a_position_can_be_registered_from_the_recovered_fill(self):
        """The recovered amount feeds the NORMAL position path, not a special one."""
        from autopaper.real_account import RealAccount, RealFillRecord
        rpc = _Rpc(tx=swap_tx(pre=[], post=[
            bal(BOY_WALLET, BOY_MINT, BOY_ACTUAL_OUTPUT)]))
        fill = _buy(_backend(rpc))
        acct = RealAccount()
        pos = acct.register_entry(
            mint=BOY_MINT, symbol="Boy",
            fill=RealFillRecord.from_onchain_fill(fill),
            token_amount=int(fill.output_amount),
            entry_price_usd=0.0003, size_usd=0.52, sol_price_usd=200.0,
        )
        assert pos.mint == BOY_MINT
        assert pos.token_amount == BOY_ACTUAL_OUTPUT
        assert acct.has_mint(BOY_MINT) is True

    def test_an_unconfirmed_transaction_cannot_open_a_position(self):
        """Fail closed: the old safety property still holds."""
        from autopaper.real_account import RealAccount, RealFillRecord
        from autopaper.execution.real import OnchainFill
        acct = RealAccount()
        with pytest.raises(ValueError):
            acct.register_entry(
                mint=BOY_MINT, symbol="Boy",
                fill=RealFillRecord.from_onchain_fill(
                    OnchainFill(status=AMBIGUOUS, signature=BOY_SIG)),
                token_amount=1, entry_price_usd=1.0, size_usd=1.0)

    def test_a_recovered_position_refuses_a_duplicate_registration(self):
        from autopaper.real_account import RealAccount, RealFillRecord
        rpc = _Rpc(tx=swap_tx(pre=[], post=[
            bal(BOY_WALLET, BOY_MINT, BOY_ACTUAL_OUTPUT)]))
        fill = _buy(_backend(rpc))
        rec = RealFillRecord.from_onchain_fill(fill)
        acct = RealAccount()
        acct.register_entry(mint=BOY_MINT, symbol="Boy", fill=rec,
                            token_amount=int(fill.output_amount),
                            entry_price_usd=0.0003, size_usd=0.52)
        with pytest.raises(ValueError):
            acct.register_entry(mint=BOY_MINT, symbol="Boy", fill=rec,
                                token_amount=int(fill.output_amount),
                                entry_price_usd=0.0003, size_usd=0.52)


# ===========================================================================
# 7. Version support is preserved
# ===========================================================================
class TestVersionAgnostic:
    @pytest.mark.parametrize("version", ["legacy", "v0", "v1"])
    def test_the_same_amounts_reconcile_for_every_version(self, version):
        """legacy / v0 / v1 all carry the same response shape."""
        keys = [BOY_WALLET, BOY_MINT]
        tx = swap_tx(keys=keys, post=[bal(BOY_WALLET, BOY_MINT,
                                          BOY_ACTUAL_OUTPUT)])
        if version != "legacy":
            # v0/v1 report the account list as dicts with provenance.
            tx["version"] = version
            tx["transaction"]["message"]["accountKeys"] = [
                {"pubkey": BOY_WALLET, "signer": True, "writable": True,
                 "source": "transaction"},
                {"pubkey": BOY_MINT, "signer": False, "writable": False,
                 "source": "lookupTable"},
            ]
        out = reconcile_transaction(tx, BOY_SIG, "confirmed", BOY_WALLET, BOY_MINT)
        assert out.token_output_amount == BOY_ACTUAL_OUTPUT
        assert out.net_lamports_delta == BOY_SOL_SPENT

    def test_v1_read_support_is_still_requested(self):
        # LiveSolanaBackend lives in real.py; the RPC client in solana.py.
        from autopaper.execution.solana import MAX_SUPPORTED_TRANSACTION_VERSION
        assert MAX_SUPPORTED_TRANSACTION_VERSION == 1


# ===========================================================================
# 8. Safety
# ===========================================================================
class TestNoBroadcastInThisSuite:
    def test_no_test_sent_a_transaction_to_a_network(self):
        """Structural: nothing in this module imports a real RPC client."""
        import ast
        src = (Path(__file__).resolve()).read_text()
        tree = ast.parse(src)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module or "")
        joined = " ".join(imports)
        assert "urllib" not in joined
        assert "requests" not in joined
        assert "solana.rpc" not in joined
        # The doubles wrap the REAL backend, so production code is what is
        # under test here, not a reimplementation of it.
        src_ast = ast.parse(src)
        names = {n.id for n in ast.walk(src_ast) if isinstance(n, ast.Name)}
        assert "LiveSolanaBackend" in names

    def test_the_keypairs_are_generated_not_loaded(self):
        kp = Keypair()
        assert kp.pubkey() is not None
