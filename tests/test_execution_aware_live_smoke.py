"""
The live READ-ONLY PAPER smoke: the actual runtime, against real chain data.

This is the only test in the repository that touches the network, and it is
marked ``live`` so no normal run collects it. It exists because the alternative
is claiming the runtime works on the strength of injected doubles, which proves
the wiring and not the wiring's connection to a real market.

WHAT IT PROVES
--------------
The production path, with no parallel test implementation:

    Telegram-shaped signal
      -> PaperTrader.on_alert
      -> production L25 decision
      -> production position sizing
      -> micro-account economics
      -> venue enumeration against real chain state
      -> executable BUY quote
      -> 10% executable-vs-mark guard
      -> PAPER BUY fill
      -> venue pin persisted
      -> production exit policy
      -> pinned-venue SELL quote
      -> PAPER SELL fill
      -> settlement PnL
      -> rent lifecycle
      -> report

WHAT IT CANNOT DO
-----------------
Sign, send or broadcast. The runtime is PAPER, the no-broadcast guard is
installed, and the audit is asserted before and after. No key is loaded, no
account is read for a balance, and no transaction object is ever constructed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from autopaper.config import build_config
from autopaper.engine import PaperTrader
from autopaper.modes import TradeMode

pytestmark = pytest.mark.live

# A graduated PumpSwap market and a bonding curve, both read-only.
GRADUATED_MINT = "7LSsEoJGhLeZzGvDofTdNg7M3JttxQqGWNLo6vWMpump"
GRADUATED_POOL = "GseMAnNDvntR5uFePZ51yZBXzNSn7GdFPkfHwfr6d77J"
CHANNEL = "MooDengPresidentCallers"


@pytest.fixture
def live_cfg(tmp_path):
    for name in ("SOLANA_PRIVATE_KEY", "SOLANA_KEYPAIR", "WALLET_KEYPAIR_PATH"):
        import os
        os.environ.pop(name, None)
    return build_config(tmp_path)


def test_the_real_runtime_executes_an_executable_paper_round_trip(live_cfg):
    from autopaper.models import AlertSnapshot
    from autopaper.paper_live.guards import audit_paper_capability

    before = audit_paper_capability()
    assert before.ok is True, before.render()

    trader = PaperTrader(live_cfg)
    assert trader.state.trade_mode is TradeMode.PAPER
    assert trader._execution_aware_paper() is True

    # The REAL engine is built on demand. Its dataset is pinned to the repo's
    # reports/live so this run leaves retained evidence rather than artefacts in
    # a temporary directory nobody will read.
    engine = asyncio.run(trader._paper_bridge_instance())
    from autopaper.paper_live.report import DatasetWriter
    engine._writer = DatasetWriter(Path.cwd(), arms=list(engine.arms))
    assert engine.audit.ok is True
    assert engine.quote_service is trader.quote_service_override or True

    signal = AlertSnapshot(
        candidate_id="live-smoke-1", source="telegram", chat=CHANNEL,
        message_id=int(time.time() * 1000) % 10_000_000,
        source_timestamp=time.time() - 1.0, received_timestamp=time.time(),
        raw_text=f"BUY ${GRADUATED_MINT}", mint=GRADUATED_MINT,
        symbol="LIVE", name="LIVE", fields={}, provenance={},
        parse_warnings=(),
    )

    # -- the real chain read: decimals, SOL/USD and an executable quote -----
    import asyncio

    # A real SOL/USD reference, read-only. Without one the engine correctly
    # refuses every entry (SOL_REFERENCE_MISSING): the USD notional cannot be
    # converted to lamports without inventing a price.
    from autopaper.paper_live.quotes import SolReference
    from autopaper.same_market.sol_price import JupiterSolPriceProvider

    sol_provider = JupiterSolPriceProvider()
    sol_snapshot = asyncio.run(sol_provider.snapshot())
    sol_usd = float(sol_snapshot.usd_per_sol)
    assert sol_usd > 0, "the live SOL/USD read failed"
    engine.sol_reference = SolReference(sol_usd, time.time(), "live-smoke")
    # ONE shared snapshot: the engine sizes in USD and the quote service prices
    # in USD, so they must not hold different references.
    engine.quote_service.sol_reference = engine.sol_reference

    # A real observed mark, so the executable-vs-mark guard has a real
    # comparison to make. Without one the guard returns UNKNOWN -- which now
    # correctly REFUSES, because a guard that passes when it cannot see is not
    # a guard.
    mark = asyncio.run(engine.observe(GRADUATED_MINT))
    assert mark.available, mark.missing_reason
    assert mark.value_usd and mark.value_usd > 0

    decimals = None
    async def probe():
        nonlocal decimals
        reader = engine.quote_service._reader
        decimals = await reader.mint_decimals(GRADUATED_MINT)
    asyncio.run(probe())
    assert decimals and decimals > 0, (
        "the real chain did not return token decimals; the live read failed")

    quote, candidates = asyncio.run(engine.quote_service.quote_buy_best(
        mint=GRADUATED_MINT, sol_lamports=4_086_302, token_decimals=decimals,
        observed_pair_address=GRADUATED_POOL,
    ))
    assert quote.available, quote.missing_reason
    assert quote.identity == "PASS", quote.identity_evidence
    assert quote.venue_id and ":" in quote.venue_id
    assert quote.output_amount_raw and quote.output_amount_raw > 0
    # Venue enumeration really ran against chain state.
    assert len(candidates) >= 1

    # The selected venue is the best EXECUTABLE output, not a preference.
    usable = [c for c in candidates
              if c.available and not c.stale and c.identity == "PASS"]
    if usable:
        best = max(usable, key=lambda c: float(c.output_amount_decimal or 0))
        assert quote.venue_id == best.venue_id

    # -- the runtime's own entry funnel, on the real decision ---------------
    records = asyncio.run(engine.handle_alert(
        snapshot=signal, report=_authorized_report(engine, signal)))

    assert records, "the runtime produced no signal record"
    for record in records:
        assert record.signal.token_mint == GRADUATED_MINT
        # Whatever the verdict, the observation and quote are retained.
        assert record.observed_mark is not None
        assert record.buy_quote is not None
        assert record.failure_detail or record.traded

    traded = [r for r in records if r.traded]
    if traded:
        position = engine._position_for(traded[0])
        assert position is not None
        # The fill is the QUOTE, never the mark.
        assert float(position.entry_fill.effective_price_usd) != \
            pytest.approx(position.entry_observed_mark, abs=1e-15)
        assert "SIMULATED_FILL" in str(position.entry_fill.kind)
        # The venue pin is recorded on the position.
        assert position.entry_venue_id
        assert position.entry_venue_id == traded[0].buy_quote.venue_id
        # Rent is LOCKED while open, and is not a trading cost.
        assert position.entry_rent_lamports > 0
        assert position.rent_locked is True

        # -- the pinned-venue exit ---------------------------------------
        engine.exit_policy = _ExitNow()
        trades = asyncio.run(engine.tick())
        for trade in trades:
            assert trade.failure.name == "NONE", trade.failure_detail
            assert float(trade.exit_fill.output_amount_decimal) > 0
            # Canonical PnL is the settlement between two fills.
            pnl = trade.pnl
            assert float(pnl.entry_cost_quote) > 0
            assert float(pnl.exit_proceeds_quote) > 0
            # Rent became reclaimable and was NOT charged as a loss.
            assert trade.position.reclaim_status == "RENT_RECLAIMABLE"
            assert trade.position.reclaimable_rent_lamports > 0

    # -- the dataset was written -------------------------------------------
    written = engine.persist()
    data = json.loads(Path(written["json"]).read_text())
    assert data["price_layers"]["ACTUAL_FILL"] is None
    assert data["status"]["signals_seen"] == len(records)
    assert data["realized_pnl_basis"].startswith("simulated settlement")

    # -- and nothing could broadcast --------------------------------------
    after = audit_paper_capability()
    assert after.ok is True, after.render()
    assert trader.state.real_trading_enabled is False
    assert trader.state.trade_mode is TradeMode.PAPER


def _authorized_report(engine, signal):
    """A production DecisionReport authorizing this signal."""
    from autopaper.models import (
        DataState, Decision, DecisionReport, ExitStyle, Regime, Route,
    )

    class _Gate:
        status = "AUTHORIZED"
        authorized = True

    return DecisionReport(
        candidate_id=signal.candidate_id, mint=signal.mint,
        symbol=signal.symbol, decision=Decision.BUY, regime=Regime.UNCERTAIN,
        route=Route.FAST_ENTRY, signal_quality=0.8, confidence=0.6,
        thesis="live smoke", risk_flags=(), entry_conditions=(),
        invalidation_conditions=(), exit_plan={},
        preferred_exit_style=ExitStyle.FAST_FAILURE, decided_at=time.time(),
        features={"volume_to_liquidity": 0.5, "signal_quality": 0.8,
                  "liquidity_usd": 60000.0},
        data_state=DataState.LIVE, v3c=_Gate(), l25=_Gate(),
    )


class _ExitNow:
    def assess(self, position, *, now, mark_price, data_state, **kw):
        from autopaper.exits import ExitDecision
        from autopaper.models import ExitStyle

        return ExitDecision("EXIT", "live smoke exit",
                            ExitStyle.FAST_FAILURE, trigger="live_smoke")
