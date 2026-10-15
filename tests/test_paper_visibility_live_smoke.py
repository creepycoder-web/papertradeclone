"""
LIVE smoke: a simulated PAPER BUY must be visible in the operator's state.

Read-only against the chain. It builds the REAL ``PaperTrader`` -- the same
object the Telegram bot runs -- with the real quote service and the real market
data service, drives one authorized BUY through
``_execute_paper_buy_execution_aware`` (the production entry path, not the
bridge directly), and then asserts what the OPERATOR reads:

  * OPEN POSITIONS shows the position
  * the dashboard reports non-zero exposure
  * durable state exists on disk
  * a fresh runtime over the same state directory rehydrates the position
  * the SELL path removes it and restores cash

No signer, no transaction builder and no broadcaster is reachable from here:
``audit_paper_capability`` asserts that before and after.

Marked ``live`` because it reads public chain state.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from autopaper.config import build_config
from autopaper.engine import PaperTrader
from autopaper.modes import TradeMode

pytestmark = pytest.mark.live

# A real PumpSwap/Pump market, read through the public RPC.
MINT = "CGKz2hEAoGQuVgguavMgvqvUBdm8qYwm8Lc9gEb2m6DU"


def _features() -> dict:
    return {"volume_to_liquidity": 0.5, "signal_quality": 0.8,
            "liquidity_usd": 60_000.0, "market_cap_usd": 20_000.0}


class _Snapshot:
    """A real ``AlertSnapshot`` for the live mint."""

    def __new__(cls, mint: str):
        from autopaper.models import AlertSnapshot

        return AlertSnapshot(
            candidate_id="live-visibility-1", source="telegram",
            chat="MooDengPresidentCallers", message_id=int(time.time()),
            source_timestamp=time.time() - 1.0,
            received_timestamp=time.time(),
            raw_text=f"BUY ${mint}", mint=mint, symbol="LIVE", name="LIVE",
            fields={}, provenance={"source": "live smoke"},
            parse_warnings=(),
        )


def _report(snapshot):
    from autopaper.models import (
        DataState, Decision, DecisionReport, ExitStyle, Regime, Route,
    )

    class _Gate:
        status = "AUTHORIZED"
        authorized = True

    return DecisionReport(
        candidate_id=snapshot.candidate_id, mint=snapshot.mint,
        symbol=snapshot.symbol, decision=Decision.BUY, regime=Regime.UNCERTAIN,
        route=Route.FAST_ENTRY, signal_quality=0.8, confidence=0.6,
        thesis="live visibility smoke", risk_flags=(), entry_conditions=(),
        invalidation_conditions=(), exit_plan={},
        preferred_exit_style=ExitStyle.FAST_FAILURE, decided_at=time.time(),
        features=_features(), data_state=DataState.LIVE,
        v3c=_Gate(), l25=_Gate(),
    )


def _trader(tmp_path: Path, *, sol_reference=None) -> PaperTrader:
    """
    The production runtime, wired exactly as the Telegram bot wires it.

    Only the SOL/USD reference is supplied from outside, and it is read-only
    (``JupiterSolPriceProvider``). The runtime's own resolver is
    REAL-path-only, so a PAPER process reaches no reference on its own -- that
    is the separate defect reported alongside this fix, and it is why the
    standalone harness and the live smoke both inject one.
    """
    for name in ("SOLANA_PRIVATE_KEY", "SOLANA_KEYPAIR", "WALLET_KEYPAIR_PATH"):
        import os

        os.environ.pop(name, None)
    trader = PaperTrader(build_config(tmp_path))
    bridge = asyncio.run(trader._paper_bridge_instance())
    if sol_reference is not None:
        bridge.sol_reference = sol_reference
        bridge.quote_service.sol_reference = sol_reference
    return trader


def _live_sol_reference():
    from autopaper.paper_live.quotes import SolReference
    from autopaper.same_market.sol_price import JupiterSolPriceProvider

    snap = asyncio.run(JupiterSolPriceProvider().snapshot())
    assert float(snap.usd_per_sol) > 0, "the live SOL/USD read failed"
    return SolReference(float(snap.usd_per_sol), snap.timestamp, snap.source)


def test_a_live_simulated_buy_is_visible_and_resumable(tmp_path):
    from autopaper.paper_live.guards import audit_paper_capability

    before = audit_paper_capability()
    assert before.ok is True, before.render()

    trader = _trader(tmp_path, sol_reference=_live_sol_reference())
    assert trader.state.trade_mode is TradeMode.PAPER
    assert trader.state.real_trading_enabled is False

    snapshot = _Snapshot(MINT)
    report = _report(snapshot)
    sent: list[str] = []
    trader._notify = lambda text, ctx=None: sent.append(str(text))

    asyncio.run(trader._execute_paper_buy_execution_aware(
        report, snapshot, _features()))

    # -- the fill happened, and the operator was told -----------------------
    assert sent, "no entry alert was emitted at all"
    assert any("PAPER BUY" in text for text in sent), (
        f"no PAPER BUY alert; got {sent}")

    bridge = trader._paper_bridge
    open_positions = bridge.open_positions()
    if not open_positions:
        # A legitimate live refusal (identity unknown, no venue, stale quote)
        # is not a visibility defect. Recorded, never silently swallowed.
        pytest.skip(
            "no live fill this run; every arm refused: "
            + "; ".join(
                str(getattr(r, "failure_detail", "")) for r in
                bridge.signal_records[-3:]))

    # -- THE FIX: OPEN POSITIONS shows it ----------------------------------
    rows = trader.open_positions_view()
    assert rows, (
        "OPEN POSITIONS is empty while the arms hold live simulated positions")
    assert {r["position_id"] for r in rows} == {
        f"paperlive_{p.signal_id}_{p.arm}" for p in open_positions}
    for row in rows:
        assert row["mode"] == TradeMode.PAPER.value
        assert row["source"] == "execution_aware_paper"
        assert row["arm"], "an arm position was rendered without its arm"
        assert row["venue"], "the executable venue was not surfaced"
        assert float(row["entry_price"]) > 0, (
            "the row's entry price is missing; it must be the FILL")

    # -- the dashboard cannot say "no exposure" ----------------------------
    snapshot_dash = trader._dashboard_snapshot_for_test()
    assert snapshot_dash["open_exposure_usd"] > 0, (
        "the dashboard reports zero exposure over live open positions")
    assert snapshot_dash["open_positions"] == len(open_positions)
    assert snapshot_dash["execution_aware_paper"]["committed_usd"] > 0

    # -- the canonical outcome says a fill happened ------------------------
    assert trader.last_execution["executed"] is True
    assert trader.last_execution["outcome"] == "PAPER_FILL=SUCCESS"

    # -- durable state ------------------------------------------------------
    state = trader.paper_state_path()
    assert state.exists(), f"no durable PAPER state at {state}"
    payload = json.loads(state.read_text(encoding="utf-8"))
    assert {p["arm"] for p in payload["open_positions"]} == {
        p.arm for p in open_positions}

    # -- RESTART rehydrates -------------------------------------------------
    restarted = _trader(tmp_path, sol_reference=_live_sol_reference())
    resumed = restarted.open_positions_view()
    assert resumed, "a restart came back with an empty book"
    assert {r["position_id"] for r in resumed} == {
        r["position_id"] for r in rows}, (
        "the restarted book differs from the book that was persisted")

    # -- SELL: removed, cash restored --------------------------------------
    cash_before = {
        p.arm: restarted._paper_bridge.accounts[p.arm].cash_usd
        for p in open_positions
    }
    restarted._paper_bridge.exit_policy = _ExitNow()
    asyncio.run(restarted.monitor_once())
    assert not restarted._paper_bridge.open_positions(), (
        "the SELL path left a settled position open")
    assert not restarted.open_positions_view(), (
        "OPEN POSITIONS still lists a position that has settled")
    for arm, cash in cash_before.items():
        assert restarted._paper_bridge.accounts[arm].cash_usd != cash, (
            f"arm {arm} cash was not restored by the settlement")
        assert restarted._paper_bridge.accounts[arm].open_exposure_usd == 0.0

    restarted.persist_paper_dataset()
    third = _trader(tmp_path, sol_reference=_live_sol_reference())
    assert not third.open_positions_view(), (
        "a settled position was resurrected by the next restart")

    # -- and nothing could broadcast ---------------------------------------
    after = audit_paper_capability()
    assert after.ok is True, after.render()
    assert third.state.real_trading_enabled is False
    assert third.state.trade_mode is TradeMode.PAPER


class _ExitNow:
    def assess(self, position, *, now, mark_price, data_state, **kw):
        from autopaper.exits import ExitDecision
        from autopaper.models import ExitStyle

        return ExitDecision("EXIT", "live smoke exit",
                            ExitStyle.FAST_FAILURE, trigger="live_smoke")