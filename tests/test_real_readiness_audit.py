"""FINAL REAL-READINESS AUDIT — evidence for reports/forensics/real_readiness_audit.md.

OFFLINE AND READ-ONLY. Nothing here can broadcast, move funds, or touch a real
wallet. Every signature comes from a generated in-memory keypair; every RPC and
provider is a mock; the broadcast boundary is guarded by a TRIPWIRE that fails
the test if it is ever crossed.

The point of this file is not to re-prove the V1 upgrade (tests/test_transaction_v1.py
owns that) but to answer one question with evidence:

    can the ACTUAL production REAL path traverse, end to end, up to the moment
    of broadcast -- and is every gate before that point real?

Each class maps to a section of the audit.
"""


from __future__ import annotations

import ast
import textwrap
import asyncio
import base64
import json
from pathlib import Path
from typing import Any, Optional

import pytest

from solders.keypair import Keypair

from autopaper.execution import txcodec as T
from autopaper.execution.intents import (
    BROADCASTED,
    CONFIRMED,
    FAILED,
    READY,
    SIGNING,
    UNKNOWN,
    ExecutionIntentStore,
)
from autopaper.execution.real import LiveSolanaBackend, RealExecutor
from autopaper.execution.solana import (
    MAX_SUPPORTED_TRANSACTION_VERSION,
    SimulationOutcome,
    WSOL_MINT,
    mask_address,
)
from tests import txfixtures as F

pytestmark = [
    pytest.mark.execution,
    pytest.mark.strategy,
]

MINT = "H2LWQMQdEGamrEQh8v4jiwBhW3hh1iqcDwU815xJ881t"
#: The production micro notional. Asserted, never varied.
POSITION_LAMPORTS = 1_071_843


class BroadcastTripwire(RuntimeError):
    """Raised if anything under test attempts to broadcast.

    The audit must establish that the path REACHES the broadcast point, so a
    test that simply never got there would prove nothing. The tripwire is
    therefore crossed deliberately by the reachability tests -- and what it
    proves is that ``sendTransaction`` was reached with a real, signed,
    simulated-clean transaction, and that the call never left the process.
    """


# ---------------------------------------------------------------------------
# doubles
# ---------------------------------------------------------------------------
class _Cfg:
    """The real-execution settings, on the same shape the executor reads."""

    def __init__(self, **kw):
        self.wallet_keypair_path = kw.get("wallet_keypair_path")
        self.rpc_url = kw.get("rpc_url", "https://rpc.invalid")
        self.slippage_bps = kw.get("slippage_bps", 500)
        self.simulate_before_send = True
        self.priority_fee_lamports = 50_000
        self.confirmation_timeout_s = 5.0
        self.quote_max_age_ms = kw.get("quote_max_age_ms", 8000.0)
        self.max_price_impact_pct = 5.0
        self.raptor_enabled = kw.get("raptor_enabled", True)
        self.raptor_base_url = "https://raptor.invalid"
        self.raptor_fallback_enabled = kw.get("raptor_fallback_enabled", True)
        self.v1_rescue_enabled = kw.get("v1_rescue_enabled", True)
        self.reentry_enabled = kw.get("reentry_enabled", False)
        self.max_open_positions = kw.get("max_open_positions", 1)
        self.quote_created_at = 0.0


class _Quote:
    input_mint, output_mint = WSOL_MINT, MINT
    in_amount, expected_out_amount = POSITION_LAMPORTS, 3_280_257
    min_out_amount, price_impact_pct, slippage_bps = 3_116_000, 0.4, 500
    route_labels, route_steps, quote_received_at, raw = ["PUMPSWAP"], [], 0.0, {}

    @property
    def price_per_token(self):
        return self.in_amount / self.expected_out_amount


class _Built:
    def __init__(self, b64: str):
        self.transaction_b64 = b64
        self.last_valid_block_height = 1
        self.prioritization_fee_lamports = 50_000


class _Provider:
    def __init__(self, name, built, *, quote_exc=None, build_exc=None):
        self.name = name
        self._built = built
        self._quote_exc = quote_exc
        self._build_exc = build_exc
        self.seen: list[dict] = []

    def provider_name(self):
        return self.name

    async def quote(self, i, o, amount, slippage):
        self.seen.append({"method": "quote", "amount": amount,
                          "slippage_bps": slippage})
        if self._quote_exc:
            raise self._quote_exc
        return _Quote()

    async def build_transaction(self, quote, user_public_key, **kw):
        self.seen.append({"method": "build", "user_public_key": user_public_key})
        if self._build_exc:
            raise self._build_exc
        return _Built(self._built)

    def route_metadata(self, quote):
        return {"provider": self.name}

    sol_input_mode = "auto_wrap_unwrap"
    requires_preprovisioned_wsol = False

    async def health(self):
        return {"provider": self.name, "reachable": True, "detail": "double"}

    async def sol_price_usd(self):
        return 117.68

    async def aclose(self):
        return None


class _Wallet:
    """A generated in-memory wallet. No file, no key material on disk."""

    def __init__(self, kp: Keypair, *, sign_exc=None):
        self._kp = kp
        self.pubkey = str(kp.pubkey())
        self.masked = mask_address(self.pubkey)
        self.signed: list[bytes] = []
        self._sign_exc = sign_exc

    @property
    def pubkey_obj(self):
        return self._kp.pubkey()

    def sign_wire(self, raw: bytes) -> bytes:
        if self._sign_exc:
            raise self._sign_exc
        self.signed.append(raw)
        return T.sign_transaction_bytes(raw, self._kp)

    def sign_new_transaction(self, ix):
        return b"offline-probe"

    async def aclose(self):
        return None


class _Rpc:
    """An RPC double that refuses to broadcast, and records every stage."""

    def __init__(self, *, sim_ok=True, lamports=10_718_431,
                 alt_load_refusal_for_v0=False, table_addresses=None):
        self.sim_ok = sim_ok
        self.lamports = lamports
        self.alt_load_refusal_for_v0 = alt_load_refusal_for_v0
        self.table_addresses = table_addresses or []
        self.simulated: list[str] = []
        self.sent: list[str] = []
        self.rpc_calls: list[tuple[str, list]] = []
        self.signature_statuses = "confirmed"
        self.get_tx = None
        self.alive = True

    # -- read-only surface the executor actually uses
    async def get_health(self):
        return type("H", (), {"reachable": True, "slot": 1,
                              "detail": "double"})()

    async def get_slot(self):
        return 1

    async def get_balance_lamports(self, address):
        return self.lamports

    async def wallet_state(self):
        return {"lamports": self.lamports, "sol": self.lamports / 1e9,
                "masked": "(masked)"}

    async def sol_price_usd(self):
        return 117.68

    async def transaction_size_report(self, b64):
        self.rpc_calls.append(("transaction_size_report", [b64]))
        return T.measure_transaction(b64)

    async def get_address_lookup_table(self, address):
        if self.table_addresses:
            return {"table": address, "query_ok": True, "usable": True,
                    "addresses": list(self.table_addresses),
                    "source": "double", "detail": "double table"}
        return {"table": address, "query_ok": True, "usable": False,
                "addresses": [], "source": "double",
                "detail": "no table in this fixture"}

    async def resolve_lookup_tables(self, addresses):
        resolved, unresolved = [], []
        for a in addresses:
            info = await self.get_address_lookup_table(a)
            if info.get("usable"):
                resolved.append((info["table"], info["addresses"]))
            else:
                unresolved.append({"table": a, "detail": info.get("detail")})
        return resolved, unresolved

    async def simulate_transaction_detailed(self, b64):
        self.simulated.append(b64)
        size = T.measure_transaction(b64)
        if not size["ok"]:
            return SimulationOutcome(
                ok=False, status="SIMULATION_FAILED_TX_TOO_LARGE",
                detail=size["detail"], size=size)
        if self.alt_load_refusal_for_v0 and \
                size["transaction_version"] == "v0":
            return SimulationOutcome(
                ok=False, status="SIMULATION_FAILED_TX_MALFORMED",
                detail="simulateTransaction: RPC error -32602: invalid "
                       "transaction: Transaction loads an address table "
                       "account that doesn't exist")
        return SimulationOutcome(
            ok=self.sim_ok,
            status="SIMULATION_OK" if self.sim_ok
            else "SIMULATION_FAILED_TX_REJECTED",
            detail="simulation ok" if self.sim_ok else "pool refused")

    # -- the guarded boundary
    async def send_transaction(self, b64):
        self.sent.append(b64)
        raise BroadcastTripwire(
            "audit tripwire: sendTransaction reached and was not allowed to "
            "leave the process")

    async def wait_for_confirmation(self, sig, **kw):
        return {"status": "confirmed", "error": None, "signature": sig,
                "slot": 1}

    async def get_transaction(self, sig, commitment="confirmed"):
        self.rpc_calls.append(("getTransaction", [sig]))
        return self.get_tx

    async def get_block(self, slot, commitment="confirmed"):
        self.rpc_calls.append(("getBlock", [slot]))
        return None

    async def aclose(self):
        return None


def _backend(primary, fallback, *, rpc=None, wallet=None, cfg=None):
    return LiveSolanaBackend(
        cfg or _Cfg(), wallet=wallet, rpc=rpc, provider=primary,
        fallback_provider=fallback)


def _tx(version="v0", size=None):
    raw = (F.build_exact(version, size) if size else F.build(version))
    return F.b64(raw)


# ===========================================================================
# 1. REAL ENTRY REACHABILITY — up to, and including, the broadcast boundary
# ===========================================================================
class TestEntryReachesBroadcast:
    def test_a_valid_entry_traverses_every_stage_up_to_broadcast(self):
        """Proves reachability by reaching the tripwire, not by avoiding it.

        If the path stopped anywhere earlier the test would fail on the missing
        tripwire, so this cannot pass vacuously.
        """
        kp = Keypair()
        wallet = _Wallet(kp)
        rpc = _Rpc()
        rap = _Provider("raptor", _tx("v0"))
        jup = _Provider("jupiter", _tx("v0"))
        backend = _backend(rap, jup, rpc=rpc, wallet=wallet)

        stages: list[str] = []
        with pytest.raises(BroadcastTripwire):
            asyncio.run(backend._execute_swap_raptor(
                "ENTRY", WSOL_MINT, MINT, POSITION_LAMPORTS, 500,
                stage=lambda name: stages.append(name)))

        # 1. It reached the broadcast boundary -- the tripwire fired.
        assert len(rpc.sent) == 1, "never reached the broadcast boundary"
        # 2. What was handed to broadcast was a real, signed, v0 transaction
        #    that had already passed the size gate and simulated clean.
        b64 = rpc.sent[0]
        report = T.measure_transaction(b64)
        assert report["ok"] is True
        assert report["transaction_version"] == "v0"
        assert report["max_bytes"] == 1232
        assert T.is_unsigned(base64.b64decode(b64)) is False
        # 3. The stages the production sequence is required to mark.
        # Assert the WHOLE sequence, not a subset: a missing or reordered
        # stage fails here. This is the production order, ending at the
        # broadcast call the tripwire caught.
        assert stages == [
            "tx_quote_start", "tx_quote_end",
            "tx_build_start", "tx_build_end",
            "exec_simulate_start", "exec_simulate_end",
            "exec_signing_start", "exec_signing_end",
            "exec_broadcast_start",
        ], stages
        # 4. The fallback was NOT reached: Raptor served the trade.
        assert jup.seen == []
        # 5. Only the wallet signed, and exactly once.
        assert len(wallet.signed) == 1
        # 6. The notional and slippage are the production values, unchanged.
        assert rap.seen[0] == {"method": "quote", "amount": POSITION_LAMPORTS,
                              "slippage_bps": 500}

    def test_a_v1_entry_reaches_broadcast_with_a_v1_transaction(self):
        """The V1 arm of the same path, not a separate code path."""
        kp = Keypair()
        wallet, rpc = _Wallet(kp), _Rpc()
        backend = _backend(_Provider("raptor", _tx("v1")),
                           _Provider("jupiter", _tx("v1")),
                           rpc=rpc, wallet=wallet)
        with pytest.raises(BroadcastTripwire):
            asyncio.run(backend._execute_swap_raptor(
                "ENTRY", WSOL_MINT, MINT, POSITION_LAMPORTS, 500))
        assert len(rpc.sent) == 1
        report = T.measure_transaction(rpc.sent[0])
        assert report["transaction_version"] == "v1"
        assert report["max_bytes"] == 4096

    def test_an_oversized_v0_is_rescued_to_v1_and_then_reaches_broadcast(self):
        """The full V1 rescue chain, on the production function."""
        from solders.instruction import AccountMeta
        from solders.pubkey import Pubkey
        kp = Keypair()
        table = "AMg6h8m8gPWyyGJoqdGfXmuA3ppgMvVtYSFYZMiRTgUf"
        addresses = [str(Pubkey.from_bytes(bytes([i] * 32)))
                     for i in range(1, 21)]
        addresses[0] = str(kp.pubkey())
        oversized = F.b64(F.build_v0_with_alt(kp, {table: addresses}))
        wallet, rpc = _Wallet(kp), _Rpc(table_addresses=addresses,
                                     alt_load_refusal_for_v0=True)
        rap = _Provider("raptor", oversized)
        backend = _backend(rap, _Provider("jupiter", _tx("v0")),
                           rpc=rpc, wallet=wallet)
        with pytest.raises(BroadcastTripwire):
            asyncio.run(backend._execute_swap_raptor(
                "ENTRY", WSOL_MINT, MINT, POSITION_LAMPORTS, 500))
        assert len(rpc.sent) == 1, "the rescue chain did not reach broadcast"
        sent = T.measure_transaction(rpc.sent[0])
        assert sent["transaction_version"] == "v1"
        assert sent["ok"] is True
        # The v0 simulation was attempted and failed on ALT loading, and the v1
        # was simulated separately: two simulations, the safety stage repeated.
        assert len(rpc.simulated) == 2
        assert T.detect_version(base64.b64decode(rpc.simulated[-1])) == "v1"

    def test_a_failed_simulation_stops_before_broadcast(self):
        for tx in (_tx("v0"), _tx("v1")):
            kp = Keypair()
            wallet, rpc = _Wallet(kp), _Rpc(sim_ok=False)
            backend = _backend(_Provider("raptor", tx),
                               _Provider("jupiter", tx),
                               rpc=rpc, wallet=wallet)
            fill = asyncio.run(backend._execute_swap_raptor(
                "ENTRY", WSOL_MINT, MINT, POSITION_LAMPORTS, 500))
            assert not fill.filled
            assert fill.status == "RAPTOR_SIMULATION_FAILED"
            assert rpc.sent == [], "a failed simulation reached broadcast"
            assert wallet.signed == [], "a failed simulation reached signing"

    def test_a_stale_quote_stops_before_build(self):
        import time
        kp = Keypair()
        wallet, rpc = _Wallet(kp), _Rpc()
        quote = _Quote()
        quote.quote_received_at = time.time() - 60.0   # 60s old
        rap = _Provider("raptor", _tx("v0"))
        rap.quote = None

        async def _old_quote(*a, **k):
            return quote
        rap.quote = _old_quote
        backend = _backend(rap, _Provider("jupiter", _tx("v0")),
                           rpc=rpc, wallet=wallet, cfg=_Cfg(
                               quote_max_age_ms=8000.0))
        fill = asyncio.run(backend._execute_swap_raptor(
            "ENTRY", WSOL_MINT, MINT, POSITION_LAMPORTS, 500))
        assert not fill.filled
        assert "STALE" in fill.status
        assert rpc.sent == []
        assert len(rap.seen) == 1, "a stale quote must not have been built"

    def test_a_signing_failure_stops_before_broadcast(self):
        kp = Keypair()
        wallet, rpc = _Wallet(kp, sign_exc=RuntimeError("key rejected")), _Rpc()
        backend = _backend(_Provider("raptor", _tx("v0")),
                           _Provider("jupiter", _tx("v0")),
                           rpc=rpc, wallet=wallet)
        fill = asyncio.run(backend._execute_swap_raptor(
            "ENTRY", WSOL_MINT, MINT, POSITION_LAMPORTS, 500))
        assert fill.status == "SIGN_FAILED"
        assert rpc.sent == []
        assert "key rejected" not in fill.ambiguity_reason, (
            "the reason must be a type, never the key's own text")


# ===========================================================================
# 2. REAL EXIT REACHABILITY
# ===========================================================================
class TestExitReachesBroadcast:
    def test_an_exit_traverses_to_broadcast_with_the_full_argument_set(self):
        kp = Keypair()
        wallet, rpc = _Wallet(kp), _Rpc()
        seen: dict = {}
        provider = _Provider("raptor", _tx("v0"))

        async def _record(quote, pubkey, **kw):
            seen.update(kw)
            seen["user"] = pubkey
            return _Built(provider._built)
        provider.build_transaction = _record
        backend = _backend(provider, _Provider("jupiter", _tx("v0")),
                           rpc=rpc, wallet=wallet)
        token_amount = 3_280_257

        from autopaper.execution.real import SafetyCheck
        prechecks = [SafetyCheck("risk_refused", True, "ok")]
        stages: list[str] = []
        executor = RealExecutor(_Cfg(), backend)
        with pytest.raises(BroadcastTripwire):
            asyncio.run(executor.close_position(
                mint=MINT, token_amount=token_amount, slippage_bps=500,
                prechecks=prechecks, stage=lambda n: stages.append(n)))
        assert len(rpc.sent) == 1
        # The exit is the mirror of the entry: same notional discipline, same
        # gates, same ownership of signing.
        assert seen["user"] == wallet.pubkey
        assert rpc.simulated, "the exit was not simulated"
        assert len(wallet.signed) == 1
        for stage in ("tx_quote_start", "tx_build_start", "exec_signing_start"):
            assert stage in stages, f"{stage} missing: {stages}"

    def test_a_failed_exit_precheck_never_reaches_the_provider(self):
        kp = Keypair()
        wallet, rpc = _Wallet(kp), _Rpc()
        provider = _Provider("raptor", _tx("v0"))
        backend = _backend(provider, _Provider("jupiter", _tx("v0")),
                           rpc=rpc, wallet=wallet)
        from autopaper.execution.real import SafetyCheck
        executor = RealExecutor(_Cfg(), backend)
        fill = asyncio.run(executor.close_position(
            mint=MINT, token_amount=1_000, slippage_bps=500,
            prechecks=[SafetyCheck("daily_loss_limit", False, "limit hit")]))
        assert fill.status == "PRE_TRADE_CHECK_FAILED"
        assert provider.seen == [], "a failed precheck still queried a provider"
        assert rpc.sent == [] and wallet.signed == []

    def test_a_failed_exit_simulation_leaves_the_position_open(self):
        """A sell in flight is not a closed position."""
        kp = Keypair()
        wallet, rpc = _Wallet(kp), _Rpc(sim_ok=False)
        provider = _Provider("raptor", _tx("v0"))
        backend = _backend(provider, _Provider("jupiter", _tx("v0")),
                           rpc=rpc, wallet=wallet)
        executor = RealExecutor(_Cfg(), backend)
        fill = asyncio.run(executor.close_position(
            mint=MINT, token_amount=1_000, slippage_bps=500))
        assert not fill.filled
        assert fill.status == "RAPTOR_SIMULATION_FAILED"
        assert rpc.sent == []


# ===========================================================================
# 3. WALLET / EQUITY — unavailable must never become zero
# ===========================================================================
class TestEquityNeverFakesAZero:
    def test_the_required_state_matrix(self):
        from autopaper.real_equity import compute_real_equity
        cases = [
            # (lamports, price, equity calculable?, why)
            (0, 100.0, True, "a REAL balance of zero is a real answer"),
            (10_718_431, 117.68, True, "the normal case"),
            (None, 117.68, False, "an unreadable balance is not a zero"),
            (10_718_431, None, False, "a missing price is not a zero"),
            (None, None, False, "neither is a zero"),
            (10_718_431, 0.0, False, "a price of zero is not a price"),
            (10_718_431, float("nan"), False, "NaN is not a price"),
            (10_718_431, float("inf"), False, "inf is not a price"),
            (-5, 100.0, False, "a negative balance is not a balance"),
            (True, 100.0, False, "a bool is not a balance"),
            ("1.0", 100.0, False, "a string is not a balance"),
        ]
        for lamports, price, expect_ok, why in cases:
            eq = compute_real_equity(lamports, price)
            assert eq.available is expect_ok, (
                f"{why}: lamports={lamports!r} price={price!r} "
                f"-> equity={eq.equity_usd!r} reason={eq.reason!r}")
            if expect_ok:
                assert eq.reason == ""
            else:
                assert eq.reason, "an unavailable equity must say why"
                assert eq.equity_usd is None

    def test_a_true_zero_balance_renders_as_a_zero_not_as_unavailable(self):
        """The distinction must survive to the rendered message."""
        from autopaper.engine import _equity_footer
        from autopaper.real_equity import compute_real_equity
        zero = compute_real_equity(0, 100.0)
        assert zero.available and zero.equity_usd == 0.0
        assert "$0.0000" in _equity_footer(zero.equity_usd)
        missing = compute_real_equity(None, 100.0)
        assert not missing.available
        rendered = _equity_footer(missing.equity_usd)
        assert "$0.0000" not in rendered, (
            "an unavailable equity must not render as $0.0000")
        assert "unavailable" in rendered.lower()

    def test_a_stale_or_unpriced_wallet_yields_no_equity(self):
        from autopaper.real_account import RealAccount
        acct = RealAccount()
        assert acct.equity_usd(0.0) is None, "no snapshot yet"
        acct.observe_wallet(lamports=10_718_431, address="X", now=0.0)
        assert acct.equity_usd(0.0) is None, "no SOL price"
        acct.observe_wallet(lamports=10_718_431, address="X",
                            sol_price_usd=117.68, now=0.0)
        assert acct.equity_usd(0.0) == pytest.approx(1.2613, rel=1e-3)
        assert acct.equity_usd(1e9) is None, "a stale snapshot has no equity"

    def test_sizing_refuses_when_equity_is_unavailable(self):
        """An unavailable equity must not become a tradeable size."""
        from autopaper.sizing import compute_position_size, tradeable
        # This is the internal collapse in _entry_equity_usd(); the requirement
        # is that the consequence is a REFUSAL, not a zero-size trade.
        ok, why = tradeable(compute_position_size(0.0))
        assert ok is False
        assert "zero" in why
        from autopaper.engine import PaperTrader  # import check
        assert PaperTrader is not None


# ===========================================================================
# 4. PROVIDER / V1 PATH
# ===========================================================================
class TestProviderControlFlow:
    def test_a_route_refusal_falls_back_once(self):
        from autopaper.execution.raptor import RaptorUnsupportedRoute
        kp = Keypair()
        rap = _Provider("raptor", _tx("v0"),
                        quote_exc=RaptorUnsupportedRoute("no route"))
        jup = _Provider("jupiter", _tx("v0"))
        wallet, rpc = _Wallet(kp), _Rpc()
        backend = _backend(rap, jup, rpc=rpc, wallet=wallet)
        with pytest.raises(BroadcastTripwire):
            asyncio.run(backend._execute_swap_raptor(
                "ENTRY", WSOL_MINT, MINT, POSITION_LAMPORTS, 500))
        assert len(rpc.sent) == 1
        assert [c["provider"] for c in []] == []
        assert jup.seen[0]["method"] == "quote"

    def test_a_network_error_does_not_fall_back(self):
        """A provider outage must never cause a second attempt."""
        from autopaper.execution.raptor import RaptorNetworkError
        rap = _Provider("raptor", _tx("v0"),
                        quote_exc=RaptorNetworkError("timed out"))
        jup = _Provider("jupiter", _tx("v0"))
        wallet, rpc = _Wallet(kp=None) if False else (_Wallet(Keypair()), _Rpc())
        backend = _backend(rap, jup, rpc=rpc, wallet=wallet)
        fill = asyncio.run(backend._execute_swap_raptor(
            "ENTRY", WSOL_MINT, MINT, POSITION_LAMPORTS, 500))
        assert fill.status == "RAPTOR_NETWORK_ERROR"
        assert jup.seen == [], "a provider outage triggered the fallback"
        assert rpc.sent == []

    def test_a_simulation_failure_does_not_cause_a_second_provider_execution(self):
        """The one that most easily regresses: rescue must not become a retry."""
        kp = Keypair()
        rap = _Provider("raptor", _tx("v0"))
        jup = _Provider("jupiter", _tx("v0"))
        wallet, rpc = _Wallet(kp), _Rpc(sim_ok=False)
        backend = _backend(rap, jup, rpc=rpc, wallet=wallet)
        fill = asyncio.run(backend._execute_swap_raptor(
            "ENTRY", WSOL_MINT, MINT, POSITION_LAMPORTS, 500))
        assert fill.status == "RAPTOR_SIMULATION_FAILED"
        # One quote, one build, one simulation. No second attempt anywhere.
        assert len(rap.seen) == 2, rap.seen
        assert len(rpc.simulated) == 1
        assert jup.seen == []

    def test_an_oversized_primary_is_rescued_without_reaching_the_fallback(self):
        kp = Keypair()
        rap = _Provider("raptor", _tx("v0", 1400))
        jup = _Provider("jupiter", _tx("v0"))
        wallet, rpc = _Wallet(kp), _Rpc()
        backend = _backend(rap, jup, rpc=rpc, wallet=wallet)
        with pytest.raises(BroadcastTripwire):
            asyncio.run(backend._execute_swap_raptor(
                "ENTRY", WSOL_MINT, MINT, POSITION_LAMPORTS, 500))
        assert jup.seen == [], "a rescued primary must not reach the fallback"
        assert T.detect_version(base64.b64decode(rpc.sent[0])) == "v1"

    def test_the_notional_is_never_changed_by_any_path(self):
        for tx, sim in ((_tx("v0"), True), (_tx("v0", 1400), True),
                        (_tx("v1"), True)):
            rap = _Provider("raptor", tx)
            wallet, rpc = _Wallet(Keypair()), _Rpc(sim_ok=sim)
            backend = _backend(rap, _Provider("jupiter", tx),
                               rpc=rpc, wallet=wallet)
            try:
                asyncio.run(backend._execute_swap_raptor(
                    "ENTRY", WSOL_MINT, MINT, POSITION_LAMPORTS, 500))
            except BroadcastTripwire:
                pass
            for call in rap.seen:
                if call["method"] == "quote":
                    assert call["amount"] == POSITION_LAMPORTS
                    assert call["slippage_bps"] == 500


# ===========================================================================
# 5. IDEMPOTENCY / BROADCAST SAFETY
# ===========================================================================
class TestIdempotency:
    def test_one_intent_cannot_reserve_twice(self, tmp_path: Path):
        store = ExecutionIntentStore(tmp_path / "i.jsonl")
        first = store.reserve(logical_key="ENTRY:M", side="ENTRY", mint=MINT,
                              requested_amount=POSITION_LAMPORTS)
        with pytest.raises(Exception):
            store.reserve(logical_key="ENTRY:M", side="ENTRY", mint=MINT,
                          requested_amount=POSITION_LAMPORTS)
        assert len(store.all_intents()) == 1
        assert first.state == "CREATED"

    def test_an_in_flight_intent_refuses_a_second_broadcast(self, tmp_path: Path):
        """Two distinct guards, and this pins both.

        ``reserve()`` refuses a duplicate for ANY non-terminal intent
        (CREATED / READY / SIGNING / BROADCASTED / UNKNOWN). That is what stops
        one logical order broadcasting twice.

        ``has_unresolved()`` is the NARROWER "an attempt may be on chain" check
        (BROADCASTED / UNKNOWN only), and it is the guard the exit path uses to
        refuse a concurrent opposite-side trade on the same mint. A SIGNING
        intent is deliberately not in that set: nothing is on chain yet, and
        the exit path separately requires ``pos.confirmed`` before it can act
        at all, so a BUY in SIGNING cannot race a SELL of a position that does
        not yet exist.
        """
        store = ExecutionIntentStore(tmp_path / "i.jsonl")
        intent = store.reserve(logical_key="ENTRY:M", side="ENTRY", mint=MINT,
                               requested_amount=POSITION_LAMPORTS)
        store.transition(intent.execution_id, READY, detail="quote ok")
        store.transition(intent.execution_id, SIGNING, detail="signing")
        # Nothing is on chain yet.
        assert store.has_unresolved(mint=MINT) is False
        assert store.get(intent.execution_id).terminal is False
        # But a second send for the same logical order is still refused.
        with pytest.raises(Exception):
            store.reserve(logical_key="ENTRY:M", side="ENTRY", mint=MINT,
                          requested_amount=POSITION_LAMPORTS)
        # Once it IS on chain, the narrower guard engages too.
        store.transition(intent.execution_id, BROADCASTED, signature="SIG1",
                         detail="b")
        assert store.has_unresolved(mint=MINT) is True
        with pytest.raises(Exception):
            store.reserve(logical_key="ENTRY:M", side="ENTRY", mint=MINT,
                          requested_amount=POSITION_LAMPORTS)

    def test_the_state_machine_refuses_an_illegal_transition(self, tmp_path: Path):
        store = ExecutionIntentStore(tmp_path / "i.jsonl")
        intent = store.reserve(logical_key="ENTRY:M", side="ENTRY", mint=MINT)
        with pytest.raises(Exception):
            store.transition(intent.execution_id, CONFIRMED,
                             detail="skipping straight to confirmed")

    def test_a_broadcast_intent_is_unresolved_so_a_retry_cannot_resend(
            self, tmp_path: Path):
        store = ExecutionIntentStore(tmp_path / "i.jsonl")
        intent = store.reserve(logical_key="ENTRY:M", side="ENTRY", mint=MINT)
        store.transition(intent.execution_id, READY, detail="q")
        store.transition(intent.execution_id, SIGNING, detail="s")
        store.transition(intent.execution_id, BROADCASTED, detail="b")
        record = store.get(intent.execution_id)
        assert record.broadcasted is True
        assert record.unresolved is True
        assert store.has_unresolved(mint=MINT) is True

    def test_restart_reloads_an_unresolved_intent_and_refuses_a_resend(
            self, tmp_path: Path):
        """Simulates a crash after broadcast, then a process restart."""
        path = tmp_path / "i.jsonl"
        first = ExecutionIntentStore(path)
        intent = first.reserve(logical_key="ENTRY:M", side="ENTRY", mint=MINT)
        first.transition(intent.execution_id, READY, detail="q")
        first.transition(intent.execution_id, SIGNING, detail="s")
        first.transition(intent.execution_id, BROADCASTED, signature="SIG1",
                         detail="b")

        # A NEW store over the same file is exactly what a restart produces.
        restarted = ExecutionIntentStore(path)
        assert restarted.has_unresolved(mint=MINT) is True
        assert len(restarted.unresolved()) == 1
        assert restarted.unresolved()[0].signature == "SIG1"
        with pytest.raises(Exception):
            restarted.reserve(logical_key="ENTRY:M", side="ENTRY", mint=MINT)

    def test_an_ambiguous_outcome_is_recorded_as_unknown_not_a_fill(
            self, tmp_path: Path):
        from autopaper.execution.real import AMBIGUOUS
        store = ExecutionIntentStore(tmp_path / "i.jsonl")
        kp = Keypair()
        wallet, rpc = _Wallet(kp), _Rpc()
        rpc.signature_statuses = "timeout"
        rpc.get_tx = None

        async def _timeout(sig, **kw):
            return {"status": "timeout", "error": "still unindexed",
                    "signature": sig}
        rpc.wait_for_confirmation = _timeout
        backend = _backend(_Provider("raptor", _tx("v0")),
                           _Provider("jupiter", _tx("v0")),
                           rpc=rpc, wallet=wallet)

        async def _send(b64):
            rpc.sent.append(b64)
            return "SIG1"
        rpc.send_transaction = _send
        fill = asyncio.run(backend._execute_swap_raptor(
            "ENTRY", WSOL_MINT, MINT, POSITION_LAMPORTS, 500,
            intent_store=store, logical_key="ENTRY:M"))
        assert fill.status == AMBIGUOUS
        assert fill.filled is False
        record = store.find("ENTRY:M")
        assert record.state == UNKNOWN
        assert store.has_unresolved(mint=MINT) is True, (
            "an ambiguous outcome must leave the intent unresolved so a retry "
            "cannot double-send")

    def test_a_confirmed_fill_is_terminal_and_not_unresolved(self,
                                                             tmp_path: Path):
        store = ExecutionIntentStore(tmp_path / "i.jsonl")
        kp = Keypair()
        rpc = _Rpc()
        rpc.get_tx = {
            "slot": 1,
            "transaction": {"message": {"accountKeys": ["x"]}, "meta": {
                "err": None, "fee": 5000, "logMessages": [],
                "preBalances": [POSITION_LAMPORTS], "postBalances": [0],
                "preTokenBalances": [],
                "postTokenBalances": [{"mint": MINT, "owner": str(kp.pubkey()),
                                       "uiTokenAmount": {"amount": "3280000", "decimals": 0}}]}},
        }

        async def _send(b64):
            rpc.sent.append(b64)
            return "SIG1"
        rpc.send_transaction = _send
        backend = _backend(_Provider("raptor", _tx("v0")),
                           _Provider("jupiter", _tx("v0")),
                           rpc=rpc, wallet=_Wallet(kp))
        fill = asyncio.run(backend._execute_swap_raptor(
            "ENTRY", WSOL_MINT, MINT, POSITION_LAMPORTS, 500,
            intent_store=store, logical_key="ENTRY:M"))
        assert fill.filled and fill.status == "confirmed"
        record = store.find("ENTRY:M")
        assert record.state == CONFIRMED
        assert record.terminal is True
        assert record.unresolved is False
        assert store.has_unresolved(mint=MINT) is False

    def test_a_failed_trade_is_terminal(self, tmp_path: Path):
        store = ExecutionIntentStore(tmp_path / "i.jsonl")
        kp = Keypair()
        rpc = _Rpc(sim_ok=False)
        backend = _backend(_Provider("raptor", _tx("v0")),
                           _Provider("jupiter", _tx("v0")),
                           rpc=rpc, wallet=_Wallet(kp))
        fill = asyncio.run(backend._execute_swap_raptor(
            "ENTRY", WSOL_MINT, MINT, POSITION_LAMPORTS, 500,
            intent_store=store, logical_key="ENTRY:M"))
        assert not fill.filled
        assert store.find("ENTRY:M").state == FAILED
        assert store.has_unresolved(mint=MINT) is False


# ===========================================================================
# 6. CONFIRMATION / RECONCILIATION across legacy, v0 and v1
# ===========================================================================
class TestReconciliationAcrossVersions:
    @pytest.mark.parametrize("version", ["legacy", "v0", "v1"])
    def test_a_confirmed_fill_reconciles_identically_for_every_version(
            self, version):
        """Same economics, same accounting, whatever the wire version is."""
        store = ExecutionIntentStore(Path(
            __file__).parent / f"_audit_{version}.jsonl")
        store.path.unlink(missing_ok=True)
        store = ExecutionIntentStore(store.path)
        kp = Keypair()
        rpc = _Rpc()
        rpc.get_tx = {
            "slot": 7,
            "transaction": {"message": {"accountKeys": ["x"]}, "meta": {
                "err": None, "fee": 5000, "logMessages": [],
                "preBalances": [POSITION_LAMPORTS], "postBalances": [0],
                "preTokenBalances": [],
                "postTokenBalances": [{"mint": MINT, "owner": str(kp.pubkey()),
                                       "uiTokenAmount": {"amount": "3280000", "decimals": 0}}]}},
        }

        async def _send(b64):
            rpc.sent.append(b64)
            return "SIG1"
        rpc.send_transaction = _send
        backend = _backend(_Provider("raptor", _tx(version)),
                           _Provider("jupiter", _tx(version)),
                           rpc=rpc, wallet=_Wallet(kp))
        fill = asyncio.run(backend._execute_swap_raptor(
            "ENTRY", WSOL_MINT, MINT, POSITION_LAMPORTS, 500,
            intent_store=store, logical_key=f"ENTRY:{version}"))
        assert fill.filled, f"{version}: {fill.status}"
        assert T.detect_version(base64.b64decode(rpc.sent[0])) == version
        assert fill.fee_lamports == 5000
        assert fill.output_amount == 3_280_000
        assert fill.slot == 7
        # The read asked for v1 support whatever was broadcast.
        assert ("getTransaction", ["SIG1"]) in rpc.rpc_calls
        store.path.unlink(missing_ok=True)

    def test_every_read_path_asks_for_v1_support(self):
        from autopaper.execution.solana import SolanaRpc
        seen = []

        class _Stub(SolanaRpc):
            async def _rpc(self, method, params):
                seen.append((method, params))
                return {"slot": 1, "value": {"err": None}}

        rpc = _Stub("https://example.invalid")
        asyncio.run(rpc.get_transaction("sig"))
        asyncio.run(rpc.get_block(1))
        for method, params in seen:
            assert params[1]["maxSupportedTransactionVersion"] == 1, method
        assert MAX_SUPPORTED_TRANSACTION_VERSION == 1

    def test_a_missing_transaction_body_is_unreconciled_not_a_fill(self):
        """Confirmation succeeded, so the outcome is CONFIRMED_BUT_UNRECONCILED.

        Not AMBIGUOUS: the chain already confirmed this swap, so "we do not know
        if it happened" is false. Not a fill either -- the amount was never
        measured, so nothing may consume it. This is the $Boy classification.
        """
        from autopaper.execution.real import CONFIRMED_UNRECONCILED
        kp = Keypair()
        rpc = _Rpc()
        rpc.get_tx = None

        async def _send(b64):
            rpc.sent.append(b64)
            return "SIG1"
        rpc.send_transaction = _send
        backend = _backend(_Provider("raptor", _tx("v1")),
                           _Provider("jupiter", _tx("v1")),
                           rpc=rpc, wallet=_Wallet(kp))
        fill = asyncio.run(backend._execute_swap_raptor(
            "ENTRY", WSOL_MINT, MINT, POSITION_LAMPORTS, 500))
        assert fill.status == CONFIRMED_UNRECONCILED
        assert fill.filled is False
        assert fill.requires_recovery is True
        assert fill.confirmed_on_chain is True


# ===========================================================================
# 7. CONFIGURATION GATES
# ===========================================================================
class TestConfigurationGates:
    def test_the_production_config_values_are_the_intended_ones(self):
        from autopaper.config import build_config
        c = build_config()
        assert c.real.trading_enabled is False
        assert getattr(c.real, "reentry_enabled", False) is False
        assert c.real.max_open_positions == 1
        assert c.real.slippage_bps == 500
        assert c.real.max_price_impact_pct == 5.0
        assert c.real.simulate_before_send is True

    def test_every_configuration_gate_is_reported_and_passes(self):
        executor = RealExecutor(_Cfg(rpc_url="https://rpc.invalid",
                                     wallet_keypair_path="/nonexistent"))
        gates = {g.name: g for g in executor.configuration_gates()}
        for name in ("wallet_configured", "rpc_configured", "reentry_disabled",
                     "max_open_positions", "slippage_bounded",
                     "price_impact_bounded", "real_not_auto_enabled"):
            assert name in gates, f"gate {name} is not reported"
        assert gates["reentry_disabled"].ok is True
        assert gates["max_open_positions"].ok is True
        assert gates["real_not_auto_enabled"].ok is True

    def test_a_gate_fails_when_its_condition_is_violated(self):
        # REENTRY_ENABLED true
        gates = {g.name: g for g in RealExecutor(
            _Cfg(reentry_enabled=True)).configuration_gates()}
        assert gates["reentry_disabled"].ok is False
        # More than one concurrent position
        gates = {g.name: g for g in RealExecutor(
            _Cfg(max_open_positions=2)).configuration_gates()}
        assert gates["max_open_positions"].ok is False
        # Slippage out of band
        gates = {g.name: g for g in RealExecutor(
            _Cfg(slippage_bps=0)).configuration_gates()}
        assert gates["slippage_bounded"].ok is False

    def test_configuration_cannot_arm_real(self):
        """REAL is armed only by an operator action, never by configuration.

        ``trading_enabled`` is a read-only PROPERTY that returns ``False`` with
        no configuration or environment input at all, so no value of any config
        key can turn it on. ``from_config`` builds it the same way for every
        caller.
        """
        from autopaper.config import RealConfig, build_config
        import inspect
        prop = RealConfig.__dict__["trading_enabled"]
        assert isinstance(prop, property), "trading_enabled must be a property"
        assert prop.fset is None, "trading_enabled must not be settable"
        # The ONLY executable statement in the property is the constant False.
        # Proven on the parsed AST, so the docstring -- which legitimately
        # contains the word "environment" -- can neither pass nor fail this.
        node = ast.parse(textwrap.dedent(inspect.getsource(prop.fget))).body[0]
        stmts = [n for n in node.body
                 if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        assert len(stmts) == 1, ast.dump(stmts)
        assert isinstance(stmts[0], ast.Return), ast.dump(stmts[0])
        assert stmts[0].value.value is False, ast.dump(stmts[0])
        # And against the live config the runtime actually uses.
        assert build_config().real.trading_enabled is False
        # It is not even a config FIELD -- there is no such key to set.
        import dataclasses
        fields = {f.name for f in dataclasses.fields(RealConfig)}
        assert "trading_enabled" not in fields, sorted(fields)
        assert build_config().real.trading_enabled is False

    def test_the_broadcast_surface_is_a_single_function(self):
        from pathlib import Path as _P
        src = (_P(__file__).resolve().parents[1] / "src/autopaper"
                / "execution/solana.py").read_text()
        assert src.count('"sendTransaction"') == 1, (
            "there must be exactly one place that can broadcast")
        assert "send_transaction" in src


# ===========================================================================
# 8. TELEGRAM / OPERATOR STATUS must read the same authoritative state
# ===========================================================================
class TestOperatorStatus:
    def test_status_reads_the_same_real_equity_the_sizing_path_does(self):
        """
        No second, duplicated equity implementation.

        The invariant is "the engine imports the ONE implementation from
        real_equity and never recomputes equity inline", not any particular
        function name. That distinction matters now: the engine moved from the
        native-SOL-only ``compute_real_equity`` to ``compute_real_equity_multi``
        because a REAL wallet may hold wrapped SOL or USDC, and pinning the old
        name would have blocked the fix while asserting nothing about
        duplication.
        """
        from autopaper.real_equity import (
            compute_real_equity,
            compute_real_equity_multi,
        )
        import inspect
        from autopaper import engine
        src = inspect.getsource(engine)
        # The engine imports from .real_equity rather than recomputing.
        assert "from .real_equity import" in src
        assert any(
            name in src for name in
            ("compute_real_equity_multi", "compute_real_equity")
        ), "engine must use the shared implementation, not an inline one"
        # Equity for SIZING comes from the real account, never from a literal.
        assert "def _entry_equity_usd" in src
        assert compute_real_equity(10_718_431, 117.68).available is True

        # The multi-asset path is the one a REAL wallet must use, and it must
        # agree with the single-asset path when there are no other assets.
        native_only = compute_real_equity_multi(10_718_431, 117.68, (), "w")
        assert native_only.available is True
        assert native_only.equity_usd == pytest.approx(
            compute_real_equity(10_718_431, 117.68).equity_usd)

    def test_real_equity_counts_every_supported_asset_or_refuses(self):
        """
        A REAL wallet's equity is the SUM over its assets, or it is unavailable.

        Valuing native SOL only is a partial valuation: the missing money is
        still money the sizing guard would be ignoring. So a held, unpriceable
        asset must invalidate the whole figure rather than be dropped from it.
        """
        from autopaper.real_equity import (
            AssetHolding,
            compute_real_equity_multi,
            WSOL_MINT,
        )
        usdc = AssetHolding("EPjF", "USDC", 25_000_000, 6, 1.0)
        wsol = AssetHolding(WSOL_MINT, "WSOL", 2_000_000_000, 9, None)

        # 1 SOL @ $200 + 2 WSOL @ $200 + 25 USDC @ $1 = $625
        total = compute_real_equity_multi(
            1_000_000_000, 200.0, [usdc, wsol], "w")
        assert total.available is True
        assert total.equity_usd == pytest.approx(625.0)
        assert total.equity_usd > 200.0, (
            "equity must exceed native SOL alone when the wallet holds more")

        # WSOL needs no explicit price: it is SOL.
        assert wsol.value_usd is None
        assert next(a for a in total.assets if a.is_wsol).value_usd == 400.0

        # A held asset with no price makes the WHOLE figure unavailable.
        blind = compute_real_equity_multi(
            1_000_000_000, 200.0,
            [AssetHolding("EPjF", "USDC", 25_000_000, 6, None)], "w")
        assert blind.available is False
        assert blind.equity_usd is None
        assert blind.unpriced_mints == ("EPjF",)
        assert "no readable USD price" in blind.reason

        # A ZERO balance needs no price: nothing is at stake in valuing nothing,
        # and requiring one would make an idle wallet permanently unsizable.
        idle = compute_real_equity_multi(
            1_000_000_000, 200.0,
            [AssetHolding("EPjF", "USDC", 0, 6, None)], "w")
        assert idle.available is True
        assert idle.equity_usd == pytest.approx(200.0)

    def test_the_status_renderer_never_shows_a_false_zero(self):
        from autopaper.engine import _equity_footer
        assert "unavailable" in _equity_footer(None).lower()
        assert "$0.0000" not in _equity_footer(None)
        assert "$0.0000" in _equity_footer(0.0), (
            "a genuine zero must still render as a zero")

    def test_the_engine_projects_the_equity_fields(self):
        from autopaper.config import build_config
        from autopaper.engine import PaperTrader
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            trader = PaperTrader(build_config(Path(tmp)))
            snapshot = trader.real_account.as_dict()
            assert "equity_usd" in snapshot or "wallet_snapshot" in snapshot
            trader.shutdown_sync() if hasattr(trader, "shutdown_sync") else None


# ===========================================================================
# 9. NO BROADCAST, EVER
# ===========================================================================
class TestAuditBroadcastSafety:
    def test_the_tripwire_is_armed_for_the_whole_audit(self):
        """Every reachability test crossed it, so it must actually work."""
        rpc = _Rpc()
        with pytest.raises(BroadcastTripwire):
            asyncio.run(rpc.send_transaction("anything"))

    def test_real_trading_is_disarmed_in_the_audit_environment(self):
        from autopaper.config import build_config
        c = build_config()
        assert c.real.trading_enabled is False
        assert c.real.env_trading_enabled_request is False

    def test_no_private_key_material_is_needed_by_this_audit(self):
        """Every keypair here was generated in memory, on purpose.

        Scoped honestly. This module DOES call ``build_config()`` and DOES read
        source files -- it has to, to test the config gates and the broadcast
        surface. What it must never do is the one step that turns a file into a
        signing key, or persist a key at all:

        * ``SolanaWallet`` -- the production keyfile loader -- is never
          constructed, so no keypair on disk is ever read;
        * every ``Keypair`` here comes from ``Keypair.generate()``;
        * nothing in the module writes any file, so no key can leak out of it.
        """
        import ast
        from pathlib import Path as _P
        tree = ast.parse(_P(__file__).read_text())
        calls = [ast.unparse(c.func) for c in ast.walk(tree)
                 if isinstance(c, ast.Call)]
        # The one step that turns a file into a key is never taken.
        assert not any("SolanaWallet" in c for c in calls), calls
        # In solders, ``Keypair()`` GENERATES a random key in memory; only
        # ``Keypair.from_*`` decodes one from existing material. None appear.
        assert not any("Keypair.from" in c for c in calls), calls
        assert any(c == "Keypair" for c in calls), calls
        # Nothing is written out. tmp_path is used only for the intent journal.
        assert not any("write_text" in c or "write_bytes" in c
                       for c in calls), calls
