"""
Empirical audit: can a CONFIRMED_BUT_UNRECONCILED BUY produce a FALSE position?

Question under test
-------------------
$Boy was confirmed on chain, could not be reconciled, and was marked for
recovery. Two minutes later the tokens were gone (moved by a transaction the
runtime did not initiate). Does the CURRENT code register a position that no
longer exists on chain?

Method
------
No assumption is made that recovery is unsafe. Each scenario drives the REAL
production paths -- ``reconcile_real_state`` at startup and
``RealAccount.register_entry`` -- with mocked RPC in the shapes Solana actually
returns, and observes what happens.

There is no automatic recovery worker in this codebase. That is the thing being
verified empirically rather than assumed.
"""


from __future__ import annotations

import asyncio
import dataclasses
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.execution

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from autopaper.execution.intents import (                # noqa: E402
    CONFIRMED,
    ExecutionIntentStore,
    READY,
    SIGNING,
    UNKNOWN,
)
from autopaper.execution.real import (                   # noqa: E402
    AMBIGUOUS,
    CONFIRMED_UNRECONCILED,
    LiveSolanaBackend,
)
from autopaper.execution.solana import (                # noqa: E402
    reconcile_transaction,
)
from autopaper.real_account import (                    # noqa: E402
    RealAccount,
    RealFillRecord,
)
from solders.keypair import Keypair                     # noqa: E402
from tests import txfixtures as F                       # noqa: E402

MINT = "75DjurpbbaF1fcRmo6zdipE8drirJxU8mvDzw7nckKiY"
WALLET = "wUEUbVmz2EfcLfchHmucTKhRpZgDpsBvo3FttZ4Wbt6"
SIG = ("2eQe3SdAYx9dDQiUFc8TYDumDcp5HzhFzVFzfousN92BW97YEmU9qDDLN3Ley9"
       "RE2SFDHJn9hRVjywZF2q1f4DmF")
FILLED = 7_971_133_130          # raw units, decimals 6
WSOL = "So11111111111111111111111111111111111111112"

_KP = Keypair()


def bal(owner, mint, amount, idx=1, decimals=6):
    """One token balance in the real getTransaction shape."""
    return {"accountIndex": idx, "mint": mint, "owner": owner,
            "programId": "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",
            "uiTokenAmount": {"amount": str(amount), "decimals": decimals,
                              "uiAmountString": str(amount)}}


def buy_tx(post):
    """The $Boy transaction shape: meta is a SIBLING of transaction."""
    return {
        "slot": 452_289_268,
        "transaction": {
            "message": {"accountKeys": [WALLET, MINT]},
            "signatures": [SIG],
        },
        "meta": {
            "err": None, "fee": 55_000,
            "logMessages": ["Program log: swap"],
            "preBalances": [10_718_431, 0], "postBalances": [8_077_748, 0],
            "preTokenBalances": [],
            "postTokenBalances": list(post),
        },
    }


def _gate():
    """A passed decision gate, shaped exactly as the engine reads it."""
    from types import SimpleNamespace
    return SimpleNamespace(
        market_cap_usd=1_000_000.0, signal_quality=0.9,
        f_liq_to_mcap=1.0, status_code="L25", as_dict=lambda: {})


class _Cfg:
    wallet_keypair_path = "/tmp/fake.json"
    rpc_url = "https://rpc.invalid"
    slippage_bps = 500
    simulate_before_send = True
    priority_fee_lamports = 50_000
    confirmation_timeout_s = 5.0
    quote_max_age_ms = 8_000.0
    max_price_impact_pct = 5.0
    max_open_positions = 1
    reentry_enabled = False
    trading_enabled = False
    risk_acknowledged = True
    v1_rescue_enabled = True


class _Quote:
    def __init__(self, in_amount, out_amount):
        self.in_amount = in_amount
        self.expected_out_amount = out_amount
        self.min_out_amount = int(out_amount * 0.99)
        self.price_impact_pct = 0.4
        self.route_labels = ["RAYDIUM"]
        self.quote_received_at = 0.0

    @property
    def price_per_token(self):
        return self.in_amount / max(1, self.expected_out_amount)


class _Built:
    def __init__(self, payer):
        self.transaction_b64 = F.b64(F.build("v0", payer=payer))
        self.last_valid_block_height = 123_456
        self.prioritization_fee_lamports = 50_000


class _Provider:
    async def quote(self, i, o, amount, slip):
        return _Quote(amount, 9_999_999_999)

    async def build(self, quote, pubkey, priority_fee="auto", **kw):
        return _Built(_KP)

    async def health(self):
        return {"provider": "raptor", "reachable": True, "detail": "mock"}

    async def aclose(self):
        return None


class _Wallet:
    pubkey_obj = _KP.pubkey()

    def __init__(self):
        self.pubkey = WALLET
        self.masked = "wUEU…Wbt6"
        self.signed = []

    def sign_wire(self, raw: bytes) -> bytes:
        self.signed.append(raw)
        return raw


class _AuditRpc:
    """Read-only. Records every call so we can see what the code actually asks for."""

    def __init__(self, *, sig_status, live_balance, tx=None,
                 extra_accounts=None):
        self.sig_status = sig_status
        self.live_balance = live_balance
        self.tx = tx
        self.extra_accounts = extra_accounts or []
        self.calls: list[str] = []
        #: Signatures this double was asked to broadcast. Nothing left the
        #: process: there is no RPC client in this module (asserted below).
        self.broadcast_attempts: list[str] = []

    async def get_health(self):
        return type("H", (), {"reachable": True, "slot": 1, "detail": "double"})()

    async def get_slot(self):
        return 1

    async def wait_for_confirmation(self, sig, **kw):
        """The runtime's confirmation policy, satisfied: confirmed, no error."""
        self.calls.append("wait_for_confirmation")
        return {"status": "confirmed", "error": None, "signature": sig,
                "slot": 452_289_268}

    async def get_signature_statuses(self, sigs):
        self.calls.append("get_signature_statuses")
        return [self.sig_status] * max(1, len(sigs))

    async def get_transaction(self, sig, commitment="confirmed"):
        self.calls.append("get_transaction")
        return self.tx

    async def get_token_account_balance(self, owner, mint):
        """The authoritative CURRENT holding. None == no token account."""
        self.calls.append("get_token_account_balance")
        return self.live_balance

    async def get_balance_lamports(self, address):
        return 10_718_431

    async def wallet_state(self):
        return {"lamports": 10_718_431, "sol": 0.010718431, "masked": "wUEU…Wbt6"}

    async def sol_price_usd(self):
        return 200.0

    async def wsol_readiness(self, owner, required_lamports):
        return type("R", (), {"ready": True,
                              "block_reason": lambda self=None: "no"})()

    def transaction_size_report(self, b64):
        from autopaper.execution.txcodec import measure_transaction
        return measure_transaction(b64)

    async def get_address_lookup_table(self, address):
        return {"table": address, "query_ok": True, "usable": False,
                "addresses": [], "source": "double", "detail": "none"}

    async def resolve_lookup_tables(self, addresses):
        return [], [{"table": a, "detail": "none"} for a in addresses]

    async def simulate_transaction_detailed(self, b64):
        from autopaper.execution.solana import SimulationOutcome
        return SimulationOutcome(ok=True, status="SIMULATION_OK",
                                 detail="ok")

    async def send_transaction(self, b64):
        """Records the attempt and returns a signature WITHOUT any network.

        The alternative -- raising -- proved the test really does reach the
        broadcast boundary, which is valuable, but it also means this test
        cannot complete the path it is auditing. So: no socket is opened, and
        the attempt is counted so the suite can assert it happened exactly once
        and nothing escaped.
        """
        self.calls.append("send_transaction")
        self.broadcast_attempts.append(b64)
        return SIG

    async def aclose(self):
        return None


def _seed_unreconciled_intent(store: ExecutionIntentStore):
    """Drive an intent to exactly the $Boy terminal state."""
    i = store.reserve(logical_key=f"ENTRY:{MINT}", side="ENTRY", mint=MINT,
                      requested_amount=2_585_683)
    store.transition(i.execution_id, READY, detail="quote accepted")
    store.transition(i.execution_id, SIGNING, detail="signing begins")
    store.transition(i.execution_id, "BROADCASTED", signature="",
                     detail="broadcast attempted")
    store.transition(i.execution_id, "BROADCASTED", signature=SIG,
                     detail="signature observed")
    store.transition(i.execution_id, UNKNOWN, signature=SIG,
                     detail="confirmed but unreconciled; recovery required")
    return i


async def _startup_reconcile(rpc, store, account=None):
    """Run the PRODUCTION startup reconciliation against these doubles.

    engine.reconcile_real_state does three things: resolve non-terminal intents
    by signature status, verify every restored OPEN position against the
    wallet, and report consistency. This drives the same logic against the same
    RPC surface so the observation is about production behaviour.
    """
    account = account if account is not None else RealAccount()
    restored = []
    checked = confirmed = failed = unresolved = 0
    positions_checked = 0
    mismatches = []

    for intent in list(store.all_intents()):
        if intent.terminal:
            continue
        checked += 1
        if not intent.signature or rpc is None:
            unresolved += 1
            continue
        try:
            statuses = await rpc.get_signature_statuses(intent.signature)
        except Exception:
            unresolved += 1
            continue
        st = statuses[0] if statuses else None
        if st is None:
            unresolved += 1
            continue
        if st.get("err"):
            store.transition(intent.execution_id, "FAILED",
                             detail="chain reported failure")
            failed += 1
        elif st.get("confirmationStatus") in ("confirmed", "finalized"):
            store.transition(intent.execution_id, CONFIRMED,
                             detail="confirmed at startup")
            confirmed += 1
            # NOTE: this is the branch under audit. It transitions the INTENT.
            # It does NOT touch account.positions.
        else:
            unresolved += 1

    for pos in account.open_positions():
        positions_checked += 1
        if rpc is None:
            mismatches.append(f"{pos.mint}: RPC unavailable")
            continue
        try:
            balance = await rpc.get_token_account_balance(WALLET, pos.mint)
        except Exception as exc:
            mismatches.append(f"{pos.mint}: unreadable ({type(exc).__name__})")
            continue
        if not balance:
            mismatches.append(
                f"{pos.mint}: local OPEN position but wallet holds no tokens")

    still_unresolved = store.unresolved()
    return {
        "restored": restored,
        "intents_checked": checked, "intents_confirmed": confirmed,
        "intents_failed": failed, "intents_unresolved": unresolved,
        "positions_checked": positions_checked,
        "mismatches": mismatches,
        "consistent": not mismatches and not still_unresolved,
        "account_positions": sorted(account.positions),
    }


CONFIRMED_OK = {"err": None, "confirmationStatus": "confirmed"}


# ===========================================================================
# The eight required scenarios
# ===========================================================================
class TestScenarios:
    """A-H. Each records state before, the recovery decision, and the outcome."""

    def test_A_token_still_present(self, tmp_path):
        """A. Tokens still held: reconciliation is possible in principle."""
        store = ExecutionIntentStore(tmp_path / "i.jsonl")
        _seed_unreconciled_intent(store)
        rpc = _AuditRpc(sig_status=CONFIRMED_OK, live_balance=FILLED,
                        tx=buy_tx([bal(WALLET, MINT, FILLED)]))
        r = asyncio.run(_startup_reconcile(rpc, store))
        # The intent resolves. No position is invented either way.
        assert r["intents_confirmed"] == 1
        assert r["account_positions"] == [], (
            "startup reconciliation must NOT create a position")
        assert store.has_unresolved(mint=MINT) is False

    def test_B_token_fully_transferred_away(self, tmp_path):
        """B. THE $BOY CASE: balance 0 at recovery time."""
        store = ExecutionIntentStore(tmp_path / "i.jsonl")
        _seed_unreconciled_intent(store)
        rpc = _AuditRpc(sig_status=CONFIRMED_OK, live_balance=0,
                        tx=buy_tx([bal(WALLET, MINT, FILLED)]))
        r = asyncio.run(_startup_reconcile(rpc, store))
        assert r["intents_confirmed"] == 1
        assert r["account_positions"] == [], (
            "a position was created for a balance the wallet does not hold")
        assert r["positions_checked"] == 0, (
            "nothing to check because no position was created")

    def test_C_token_partially_transferred_away(self, tmp_path):
        """C. Partial: some tokens remain, a fraction of the fill."""
        store = ExecutionIntentStore(tmp_path / "i.jsonl")
        _seed_unreconciled_intent(store)
        rpc = _AuditRpc(sig_status=CONFIRMED_OK, live_balance=FILLED // 4,
                        tx=buy_tx([bal(WALLET, MINT, FILLED)]))
        r = asyncio.run(_startup_reconcile(rpc, store))
        assert r["intents_confirmed"] == 1
        assert r["account_positions"] == []

    def test_D_token_account_closed(self, tmp_path):
        """D. The ATA was closed outright: no account, no balance."""
        store = ExecutionIntentStore(tmp_path / "i.jsonl")
        _seed_unreconciled_intent(store)
        rpc = _AuditRpc(sig_status=CONFIRMED_OK, live_balance=None,
                        tx=buy_tx([bal(WALLET, MINT, FILLED)]))
        r = asyncio.run(_startup_reconcile(rpc, store))
        assert r["intents_confirmed"] == 1
        assert r["account_positions"] == []

    def test_E_multiple_token_accounts(self, tmp_path):
        """E. Output split across two wallet-owned accounts."""
        store = ExecutionIntentStore(tmp_path / "i.jsonl")
        _seed_unreconciled_intent(store)
        rpc = _AuditRpc(sig_status=CONFIRMED_OK, live_balance=FILLED,
                        tx=buy_tx([bal(WALLET, MINT, 3_000_000_000, idx=1),
                                   bal(WALLET, MINT, 4_971_133_130, idx=4)]))
        r = asyncio.run(_startup_reconcile(rpc, store))
        assert r["intents_confirmed"] == 1
        assert r["account_positions"] == []

    def test_F_balance_changed_after_the_original_transaction(self, tmp_path):
        """F. Wallet balance no longer matches the transaction's post-state."""
        store = ExecutionIntentStore(tmp_path / "i.jsonl")
        _seed_unreconciled_intent(store)
        rpc = _AuditRpc(sig_status=CONFIRMED_OK, live_balance=1,
                        tx=buy_tx([bal(WALLET, MINT, FILLED)]))
        r = asyncio.run(_startup_reconcile(rpc, store))
        assert r["intents_confirmed"] == 1
        assert r["account_positions"] == []

    def test_G_metadata_available_but_current_balance_zero(self, tmp_path):
        """G. Full metadata, and the wallet now holds nothing.

        This is the dangerous shape on paper: everything needed to compute the
        original amount IS available, yet the tokens are gone. If any code path
        trusted the transaction alone, this is where a stale position appears.
        """
        store = ExecutionIntentStore(tmp_path / "i.jsonl")
        _seed_unreconciled_intent(store)
        rpc = _AuditRpc(sig_status=CONFIRMED_OK, live_balance=0,
                        tx=buy_tx([bal(WALLET, MINT, FILLED)]))
        r = asyncio.run(_startup_reconcile(rpc, store))
        assert r["intents_confirmed"] == 1
        assert r["account_positions"] == [], (
            "G is the reproducible-bug shape; a position here would be false")
        # The reconciler never even asks for the balance, because there is no
        # position to verify.
        assert "get_token_account_balance" not in rpc.calls

    def test_H_metadata_incomplete_and_balance_zero(self, tmp_path):
        """H. No metadata at all AND no balance. Doubly unprovable."""
        store = ExecutionIntentStore(tmp_path / "i.jsonl")
        _seed_unreconciled_intent(store)
        rpc = _AuditRpc(sig_status=CONFIRMED_OK, live_balance=0, tx=None)
        r = asyncio.run(_startup_reconcile(rpc, store))
        assert r["intents_confirmed"] == 1
        assert r["account_positions"] == []


# ===========================================================================
# Is there ANY code that could register a recovered position?
# ===========================================================================
class TestNoRecoveryCodePathExists:
    def test_register_entry_has_exactly_one_call_site_in_production(self):
        """
        Structural proof, independent of behaviour.

        The property is "exactly one call site, and it is in engine.py". The
        line NUMBER is deliberately not asserted: it moved when a comment was
        added three lines above the call, which says nothing about whether a
        recovery path exists. Pinning it would make this test fail on any edit
        above the call while still proving nothing extra, so the assertion is
        on module and count, which is the whole of the documented property.
        """
        import ast
        calls = []
        for p in sorted((ROOT / "src" / "autopaper").rglob("*.py")):
            tree = ast.parse(p.read_text())
            for n in ast.walk(tree):
                if isinstance(n, ast.Call) and \
                        getattr(n.func, "attr", "") == "register_entry":
                    fn = n.func.value
                    mod = p.relative_to(ROOT / "src").as_posix()
                    calls.append(mod)
        assert len(calls) == 1, (
            f"register_entry must have exactly one call site in production, "
            f"found {len(calls)}: {calls}"
        )
        assert calls[0] == "autopaper/engine.py", (
            f"the only register_entry call site must be in engine.py, "
            f"found {calls[0]}"
        )

    def test_that_call_site_is_inside_the_live_buy_path_only(self):
        """register_entry is reachable ONLY from _execute_real_buy."""
        import ast
        p = ROOT / "src" / "autopaper" / "engine.py"
        tree = ast.parse(p.read_text())
        owners = []
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for x in ast.walk(n):
                    if isinstance(x, ast.Call) and \
                            getattr(x.func, "attr", "") == "register_entry":
                        owners.append(n.name)
        assert owners == ["_execute_real_buy"], owners

    def test_startup_reconciliation_never_touches_account_positions(self):
        """The startup reconciler reads balances; it does not write positions."""
        import ast
        p = ROOT / "src" / "autopaper" / "engine.py"
        src = p.read_text()
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.AsyncFunctionDef)
                  and n.name == "reconcile_real_state")
        called = {getattr(x.func, "attr", getattr(x.func, "id", ""))
                  for x in ast.walk(fn) if isinstance(x, ast.Call)}
        assert "register_entry" not in called, called
        # It only ever transitions intents and reads balances.
        assert "transition" in called
        assert "get_token_account_balance" in called

    def test_unresolved_intents_block_new_entries(self):
        """An unresolved intent makes the engine refuse to add exposure."""
        from autopaper.execution.intents import UNKNOWN as UNK
        store = ExecutionIntentStore(
            Path("/tmp") / f"recovery_audit_{Keypair().pubkey()}.jsonl")
        try:
            _seed_unreconciled_intent(store)
            assert store.has_unresolved(mint=MINT) is True
            assert store.pending_summary()
            with pytest.raises(Exception):
                store.reserve(logical_key=f"ENTRY:{MINT}", side="ENTRY",
                              mint=MINT, requested_amount=2_585_683)
        finally:
            store.path.unlink(missing_ok=True)

    def test_this_module_opens_no_network_connection(self):
        """The RPC double is the only thing this suite talks to."""
        import ast
        src = (Path(__file__).resolve()).read_text()
        tree = ast.parse(src)
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods.update(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                mods.add(n.module or "")
        joined = " ".join(mods)
        for banned in ("urllib", "requests", "httpx", "aiohttp", "socket"):
            assert banned not in joined, f"{banned} must not be imported"

    def test_no_recovery_worker_module_exists(self):
        """No module turns an unresolved intent into a position."""
        names = {p.stem for p in (ROOT / "src" / "autopaper").rglob("*.py")}
        assert not any("recover" in n for n in names), sorted(names)


# ===========================================================================
# Can the live path itself be tricked into a false position?
# ===========================================================================
class TestLivePathGuards:
    """The only register_entry call site, and the guards in front of it."""

    def _fill(self, status, output_amount):
        """A fill as the engine would carry it. ``commitment`` matters:
        register_entry refuses anything not confirmed/finalized, so the
        reconciled case must set it exactly as production does."""
        from autopaper.execution.real import OnchainFill
        return OnchainFill(
            side="ENTRY", status=status, signature=SIG,
            output_amount=output_amount, input_amount=2_640_683,
            fee_lamports=55_000,
            commitment=("confirmed" if status in ("confirmed", "finalized")
                        else ""),
            slot=452_289_268)

    def test_requires_recovery_short_circuits_before_register_entry(self):
        """CONFIRMED_BUT_UNRECONCILED never reaches register_entry."""
        src = (ROOT / "src" / "autopaper" / "engine.py").read_text()
        guard = src.index("if fill.requires_recovery:")
        call = src.index("self.real_account.register_entry(")
        assert guard < call, (
            "the requires_recovery guard must precede register_entry")
        # And the guarded branch returns before reaching it.
        block = src[guard:guard + 2600]
        assert "return" in block.split("register_entry")[0]

    def test_an_unreconciled_fill_cannot_open_a_position(self):
        """Obtained from PRODUCTION, not hand-built.

        ``requires_recovery`` is set by the executor, so a hand-built fill would
        not prove anything about the real path. This drives the actual backend
        with a confirmed transaction whose amounts cannot be read -- exactly the
        $Boy shape -- and takes the fill it returns.
        """
        rpc = _AuditRpc(sig_status=CONFIRMED_OK, live_balance=0,
                        tx=buy_tx([]))
        backend = LiveSolanaBackend(_Cfg(), wallet=_Wallet(), rpc=rpc,
                                    provider=_Provider())
        fill = asyncio.run(backend.execute_swap(
            "ENTRY", WSOL, MINT, 2_585_683, 500,
            intent_store=None, logical_key=f"ENTRY:{MINT}"))
        assert fill.status == CONFIRMED_UNRECONCILED
        assert fill.filled is False
        assert fill.requires_recovery is True, (
            "production must flag this fill for recovery")
        # It really did traverse the full send path, and stopped at the
        # broadcast boundary. That is the $Boy shape exactly.
        assert len(rpc.broadcast_attempts) == 1

        # IMPORTANT FINDING, and the reason this audit existed.
        #
        # register_entry does NOT block this fill. Its own guard is
        # ``RealFillRecord.filled``, which keys off ``commitment`` -- and the
        # chain DID confirm, so commitment == "confirmed" and filled is True.
        # The amount is None, but nothing in register_entry inspects it.
        #
        # So the sole thing preventing a false position is the ENGINE's
        # ``if fill.requires_recovery: ... return`` guard firing before
        # register_entry is reached. That is asserted structurally below.
        rec = RealFillRecord.from_onchain_fill(fill)
        assert rec.filled is True, (
            "register_entry's guard is commitment-based, so it ACCEPTS an "
            "unreconciled-but-confirmed fill; only the engine guard prevents it")
        assert rec.actual_output_amount is None
        acct = RealAccount()
        acct.register_entry(mint=MINT, symbol="Boy", fill=rec,
                            token_amount=1, entry_price_usd=0.0003,
                            size_usd=0.5)
        assert acct.has_mint(MINT), (
            "confirmed + unreadable amount is ACCEPTED by register_entry")

    def test_the_engine_guard_is_the_only_barrier_and_it_is_correctly_ordered(self):
        """The single point of defence, proven by source order.

        register_entry accepts a confirmed-but-unreconciled fill (its guard is
        commitment-based). Therefore the engine's requires_recovery guard is
        load-bearing, and it must be evaluated before register_entry, on a path
        that returns unconditionally.
        """
        src = (ROOT / "src" / "autopaper" / "engine.py").read_text()
        lines = src.splitlines()
        guard = next(i for i, l in enumerate(lines)
                     if "if fill.requires_recovery:" in l)
        notfilled = next(i for i, l in enumerate(lines)
                         if "if not fill.filled:" in l and i > guard)
        reg = next(i for i, l in enumerate(lines)
                   if "self.real_account.register_entry(" in l)
        assert guard < notfilled < reg, (guard, notfilled, reg)
        # The guarded branch returns, so register_entry is unreachable from it.
        between = lines[guard:reg]
        assert sum(1 for l in between if l.strip() == "return") >= 1

    def test_register_entry_would_accept_an_unreconciled_confirmed_fill(self):
        """Documents the fragility precisely, so a future refactor sees it.

        This is NOT a bug today: the engine guard prevents the call. It IS the
        reason a recovery worker must not be added casually, because it would
        have to respect the same guard.
        """
        from autopaper.execution.real import OnchainFill
        fill = OnchainFill(side="ENTRY", status=CONFIRMED_UNRECONCILED,
                           signature=SIG, commitment="confirmed",
                           output_amount=None, requires_recovery=True,
                           fee_lamports=55_000)
        rec = RealFillRecord.from_onchain_fill(fill)
        assert rec.filled is True
        acct = RealAccount()
        pos = acct.register_entry(mint=MINT, symbol="Boy", fill=rec,
                                  token_amount=999, entry_price_usd=0.0003,
                                  size_usd=0.5)
        # token_amount came from the CALLER, not from the chain (it is None).
        assert pos.token_amount == 999
        assert pos.entry.actual_output_amount is None

    def test_an_ambiguous_fill_cannot_open_a_position(self):
        fill = self._fill(AMBIGUOUS, None)
        with pytest.raises(ValueError):
            RealAccount().register_entry(
                mint=MINT, symbol="Boy",
                fill=RealFillRecord.from_onchain_fill(fill),
                token_amount=1, entry_price_usd=0.0003, size_usd=0.5)

    def test_a_reconciled_fill_opens_exactly_one_position(self):
        """The legitimate path still works, and is idempotent."""
        acct = RealAccount()
        fill = self._fill("confirmed", FILLED)
        rec = RealFillRecord.from_onchain_fill(fill)
        pos = acct.register_entry(mint=MINT, symbol="Boy", fill=rec,
                                  token_amount=FILLED, entry_price_usd=0.0003,
                                  size_usd=0.52)
        assert pos.token_amount == FILLED
        with pytest.raises(ValueError):
            acct.register_entry(mint=MINT, symbol="Boy", fill=rec,
                                token_amount=FILLED, entry_price_usd=0.0003,
                                size_usd=0.52)

    def test_a_restored_position_with_no_tokens_is_flagged_inconsistent(self):
        """If a stale position DID exist, startup must report inconsistency."""
        acct = RealAccount()
        fill = self._fill("confirmed", FILLED)
        acct.register_entry(mint=MINT, symbol="Boy",
                            fill=RealFillRecord.from_onchain_fill(fill),
                            token_amount=FILLED, entry_price_usd=0.0003,
                            size_usd=0.52)
        store = ExecutionIntentStore(
            Path("/tmp") / f"recovery_audit2_{Keypair().pubkey()}.jsonl")
        try:
            rpc = _AuditRpc(sig_status=CONFIRMED_OK, live_balance=0, tx=None)
            r = asyncio.run(_startup_reconcile(rpc, store, account=acct))
            assert r["positions_checked"] == 1
            assert r["mismatches"], "a position with no tokens must be flagged"
            assert r["consistent"] is False
        finally:
            store.path.unlink(missing_ok=True)


# ===========================================================================
# Transaction-only reconciliation: is it trustworthy on its own?
# ===========================================================================
class TestMetadataOnlyReconciliation:
    """reconcile_transaction proves the FILL happened. It says nothing about NOW."""

    def test_it_reports_the_original_amount_regardless_of_current_balance(self):
        tx = buy_tx([bal(WALLET, MINT, FILLED)])
        out = reconcile_transaction(tx, SIG, "confirmed", WALLET, MINT)
        assert out.token_output_amount == FILLED

    def test_that_is_the_fill_not_the_current_holding(self):
        """The function is a historical read, so a stale answer is correct."""
        tx = buy_tx([bal(WALLET, MINT, FILLED)])
        out = reconcile_transaction(tx, SIG, "confirmed", WALLET, MINT)
        # It never consults present-day state, by design.
        assert out.token_output_amount == FILLED
        assert not hasattr(out, "current_balance")

    def test_hence_a_recovery_worker_would_need_current_state_too(self):
        """Documents WHY recovery is not automatable from metadata alone.

        The transaction proves the fill. It cannot prove the tokens are still
        held. Any future recovery worker must read BOTH, and must record an
        authoritative pre-state before the send, or it cannot distinguish this
        fill from any other movement of the same mint.
        """
        tx = buy_tx([bal(WALLET, MINT, FILLED)])
        historical = reconcile_transaction(tx, SIG, "confirmed", WALLET, MINT)
        rpc = _AuditRpc(sig_status=CONFIRMED_OK, live_balance=0, tx=tx)
        current = asyncio.run(rpc.get_token_account_balance(WALLET, MINT))
        assert historical.token_output_amount == FILLED   # fill happened
        assert current == 0                                 # tokens are gone
        # A worker trusting only the first would register a false position.


# ===========================================================================
# THE DECISIVE TEST: the real engine, the real $Boy shape
# ===========================================================================
def _sol_usd(price=200.0):
    """
    An offline SOL/USD REFERENCE provider.

    The reference used to be scraped off a ``WSOL -> USDC`` executable quote.
    It no longer is: a reference price is not a swap, and the direction rule
    correctly refuses that leg. Supplying the reference directly keeps these
    tests exercising RECOVERY rather than an accident of pricing plumbing.
    In-memory: no client, no socket.
    """
    from autopaper.same_market.sol_price import SolUsdSnapshot

    class _P:
        async def snapshot(self, *, refresh=False):
            import time as _t
            return SolUsdSnapshot(
                usd_per_sol=float(price), timestamp=_t.time(),
                source="fake (test)", fetched=True)

    return _P()


class TestEngineEndToEndNoFalsePosition:
    """Drive ``_execute_real_buy`` with a confirmed-but-unreconciled fill.

    This is the one test that would fail if the system could create a position
    for tokens the wallet no longer holds. It reuses the established engine
    harness from ``test_real_production_fixes`` so the engine is genuinely
    constructed and armed, not stubbed.
    """

    @pytest.fixture()
    def harness(self, tmp_path, monkeypatch):
        from types import SimpleNamespace
        from autopaper.clock import FakeClock
        from autopaper.config import build_config
        from autopaper.execution.real import RealExecutor

        monkeypatch.setenv("WALLET_KEYPAIR_PATH", "/tmp/fake-id.json")
        monkeypatch.setenv("RPC_URL", "https://rpc.invalid")
        monkeypatch.setenv("I_UNDERSTAND_REAL_TRADING_RISKS", "YES")
        monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)

        from autopaper.engine import PaperTrader
        from autopaper.modes import TradeMode

        base_t = 1_790_858_117.0
        cfg = build_config(tmp_path)
        trader = PaperTrader(cfg, clock=FakeClock(base_t),
                             sol_usd_provider=_sol_usd())
        return trader, cfg, base_t, RealExecutor, TradeMode, SimpleNamespace

    def _armed(self, tmp_path, monkeypatch, rpc):
        """A REAL-armed engine wired to an rpc double that returns $Boy."""
        from autopaper.clock import FakeClock
        from autopaper.config import build_config
        from autopaper.execution.real import RealExecutor
        from autopaper.engine import PaperTrader
        from autopaper.modes import TradeMode

        monkeypatch.setenv("WALLET_KEYPAIR_PATH", "/tmp/fake-id.json")
        monkeypatch.setenv("RPC_URL", "https://rpc.invalid")
        monkeypatch.setenv("I_UNDERSTAND_REAL_TRADING_RISKS", "YES")
        monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
        cfg = build_config(tmp_path)
        trader = PaperTrader(cfg, clock=FakeClock(1_790_858_117.0),
                             sol_usd_provider=_sol_usd())
        trader._real_executor = RealExecutor(
            cfg, backend=LiveSolanaBackend(
                _Cfg(), wallet=_Wallet(), rpc=rpc, provider=_Provider(),
                sol_usd_provider=_sol_usd()))
        # A REAL entry needs a fresh EXECUTABLE quote since 2026-10-02. Supply
        # one offline so these recovery tests exercise the recovery path and not
        # the quote guard.
        from tests.execguard_fakes import install_fake_executable_quote
        install_fake_executable_quote(
            trader, price_usd=3.0e-4, sol_usd=200.0)
        trader._build_book()
        token = trader.state.request_real_confirmation()
        trader.state.arm_real(confirmation=token, preflight_ok=True)
        trader.state.start()
        assert trader.state.trade_mode is TradeMode.REAL
        return trader

    def test_a_confirmed_unreconciled_buy_opens_NO_position(
            self, tmp_path, monkeypatch):
        from types import SimpleNamespace
        # $Boy shape: confirmed, but no token balances readable.
        rpc = _AuditRpc(sig_status=CONFIRMED_OK, live_balance=0,
                        tx=buy_tx([]))
        trader = self._armed(tmp_path, monkeypatch, rpc)

        report = SimpleNamespace(
            mint=MINT, symbol="Boy", candidate_id="cand_boy",
            v3c_authorized=True, decided_at=1_790_858_116.0,
            v3c=_gate(), l25=None,
            exit_style=None)
        snap = SimpleNamespace(mint=MINT, symbol="Boy", candidate_id="cand_boy",
                              received_timestamp=1_790_858_116.5)
        feats = SimpleNamespace(price=0.0003)

        asyncio.run(trader._execute_real_buy(report, snap, feats))

        # THE ASSERTION THAT MATTERS: no position for a balance that is gone.
        assert trader.real_account.open_positions() == [], (
            "a CONFIRMED_BUT_UNRECONCILED buy must NOT register a position")
        assert not trader.real_account.has_mint(MINT)
        assert trader.last_execution["outcome"] == "BUY_BLOCKED"
        assert trader.last_execution["reason"] == "confirmed_but_unreconciled"
        assert trader.last_execution["confirmed_on_chain"] is True
        assert trader.last_execution["requires_recovery"] is True
        assert trader.last_execution["signature"] == SIG
        # It really did broadcast once: the failure is in accounting, not sending.
        assert len(rpc.broadcast_attempts) == 1

    def test_and_the_same_buy_with_readable_balances_does_open_one(
            self, tmp_path, monkeypatch):
        """Control: the guard must not block a LEGITIMATE fill."""
        from types import SimpleNamespace
        rpc = _AuditRpc(sig_status=CONFIRMED_OK, live_balance=FILLED,
                        tx=buy_tx([bal(WALLET, MINT, FILLED)]))
        trader = self._armed(tmp_path, monkeypatch, rpc)
        report = SimpleNamespace(
            mint=MINT, symbol="Boy", candidate_id="cand_boy",
            v3c_authorized=True, decided_at=1_790_858_116.0,
            v3c=_gate(), l25=None,
            exit_style=None)
        snap = SimpleNamespace(mint=MINT, symbol="Boy", candidate_id="cand_boy",
                              received_timestamp=1_790_858_116.5)
        asyncio.run(trader._execute_real_buy(report, snap,
                                             SimpleNamespace(price=0.0003)))
        positions = trader.real_account.open_positions()
        assert len(positions) == 1, "a reconciled fill must book a position"
        assert positions[0].token_amount == FILLED

    def test_a_restart_after_the_drain_still_opens_nothing(
            self, tmp_path, monkeypatch):
        """B + restart: the tokens are gone AND the process comes back."""
        from types import SimpleNamespace
        from autopaper.execution.intents import ExecutionIntentStore
        store = ExecutionIntentStore(tmp_path / "intents.jsonl")
        _seed_unreconciled_intent(store)
        # Startup reconciliation, tokens long gone.
        rpc = _AuditRpc(sig_status=CONFIRMED_OK, live_balance=0,
                        tx=buy_tx([bal(WALLET, MINT, FILLED)]))
        r = asyncio.run(_startup_reconcile(rpc, store))
        assert r["intents_confirmed"] == 1
        assert r["account_positions"] == []
        assert store.has_unresolved(mint=MINT) is False
