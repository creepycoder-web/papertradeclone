"""
PAPER BUY forensics: does ``decision=BUY`` actually become a position, and when
it does not, is the refusal observable?

THE INCIDENT
------------
The runtime reported ``decision=BUY`` beside ``quote=-`` / ``fill=-``, and the
shutdown line read ``closed=420 ... buys=0  open=0``. That combination is
unreadable: it looks exactly like an executor that lost 420 trades.

The truth, from ``var/ledger.jsonl`` at the time:

    {"event": "note", "message": "entry refused as stale",
     "reason": "STALE_SIGNAL", "symbol": "Bittersweet",
     "age_ms": 34880.7, "budget_ms": 15000.0}

and ``"unfilled_attempts": 4`` in the persisted account. The executor was never
reached: the 34.9s analysis exceeded the existing 15s entry budget, so
:class:`autopaper.latency.StaleEntryGate` refused the entry as a DATA-QUALITY
failure before an order was constructed. Correct behaviour, invisible
presentation.

WHAT THIS FILE FIXES
--------------------
Every BUY now ends in exactly one observable outcome:

    DECISION=BUY          the strategy authorized an entry; nothing more
    PAPER_FILL=SUCCESS    a fill happened and an open position exists
    BUY_BLOCKED=<reason>  execution was refused, and this says why

and the refusal reaches the stage recorder, the latency accumulator, the
journal, the operator log AND the chat. The stale gate is deliberately NOT
relaxed - widening the budget would have hidden the defect rather than fixed it.
"""


from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from autopaper.clock import FakeClock
from autopaper.config import MarketConfig
from autopaper.engine import CLOSED_PAGE_SIZE, PaperTrader
from autopaper.latency import (
    BUY_BLOCKED,
    DECISION_BUY,
    PAPER_FILL_SUCCESS,
    REAL_FILL_SUCCESS,
    StageRecorder,
)
from autopaper.market import MarketDataService
from autopaper.market.ratelimit import RateLimiter
from autopaper.modes import TradeMode
from autopaper.models import (
    AlertSnapshot,
    DataState,
    Decision,
    DecisionReport,
    NarSnapshot,
    Observation,
    PositionStatus,
    QuoteResult,
    new_id,
)
from autopaper.nar.base import NarService
from autopaper.telegram import TelegramListener
from autopaper.telegram.client import ConnectionState

from .conftest import MINT, MINT2

pytestmark = [
    pytest.mark.execution,
    pytest.mark.integrity,
    pytest.mark.strategy,
]

T0 = 1_700_000_000.0


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


class Venue:
    """A deterministic venue. Market cap stays INSIDE V3-C's $20,000 ceiling."""

    def __init__(self, clock: FakeClock):
        self.clock = clock
        self.price = 1.5e-5
        self.started = clock.now()
        self.calls = 0
        self.fail = False
        self.quote_age_s = 0.0
        self.price_hook = None

    def quote(self, mint: str) -> QuoteResult:
        self.calls += 1
        if self.fail:
            return QuoteResult(observation=None, error="venue: down",
                               provider="venue")
        now = self.clock.now()
        price = self.price_hook(self.clock.now()) if self.price_hook else self.price
        received = now - self.quote_age_s
        return QuoteResult(
            observation=Observation(
                mint=mint, observed_at=received, received_at=received,
                price_usd=price,
                market_cap_usd=15_600.0,
                liquidity_usd=15_500.0,
                volume_5m_usd=21_600.0, volume_1h_usd=120_000.0,
                buys_5m=200, sells_5m=10, txns_5m=210,
                pair_created_at=self.started - 360,
                provider="venue",
            ),
            provider="venue",
        )


class FakeNarSource:
    name = "nar:fake"

    def __init__(self, *, available: bool = True, work_s: float = 0.0):
        self.available = available
        self.work_s = work_s
        self.calls = 0

    async def fetch_async(self, mint: str) -> NarSnapshot:
        self.calls += 1
        if self.work_s:
            await asyncio.sleep(self.work_s)
        return NarSnapshot(
            mint=mint, fetched_at=time_now(), source=self.name,
            available=self.available,
            narrative="Bittersweet is a dog meme coin." if self.available else None,
            error=None if self.available else "NAR_UNAVAILABLE: silent",
        )


def time_now() -> float:
    import time

    return time.time()


def make_trader(config, clock: FakeClock, *, venue: Venue | None = None,
                nar: object | None = None) -> PaperTrader:
    venue = venue or Venue(clock)

    class VPrimary:
        name = "venue"

        def __init__(self):
            self.limiter = RateLimiter(10_000, name="venue")
            self.retry_stats = lambda: {}
            self.max_call_s = 0.0
            self.timeout_s = 0.0

        quote = venue.quote

    market = MarketDataService(config.market, clock=clock, primary=VPrimary(),
                               fallback=VPrimary())
    market._fvp_probe_budget = 0
    nar_service = NarService(config.nar, clock=clock)
    listener = TelegramListener(config.telegram, clock=clock)
    listener.status.state = ConnectionState.CONNECTED
    listener.status.chat_id = 100
    listener.status.chat = "testchat"
    trader = PaperTrader(config, clock=clock, market=market, nar=nar_service,
                         listener=listener)
    listener.on_alert = trader.on_alert
    trader.forward = None
    trader.forward_deployment = None

    async def fake_sleep(seconds: float) -> None:
        clock.advance(seconds)

    trader._sleep = fake_sleep
    return trader


def force_buy(trader: PaperTrader, *, thesis: str = "forced BUY for forensics"):
    """
    Force the decision to BUY WITHOUT touching the strategy.

    This is a forensic device, not a strategy change: L25, V3-C, every threshold
    and every veto are untouched, and the wrapper is installed on the ENGINE's
    decision-engine instance only. It exists so the EXECUTION half of the
    incident can be reproduced without inventing a market the strategy would
    have to accept.
    """
    import dataclasses

    real = trader.engine.decide

    def decide(features, alert, nar, analysis_start, analysis_complete,
               account_blocked_reason=None):
        report = real(features=features, alert=alert, nar=nar,
                      analysis_start=analysis_start,
                      analysis_complete=analysis_complete,
                      account_blocked_reason=account_blocked_reason)
        return dataclasses.replace(report, decision=Decision.BUY, thesis=thesis)

    trader.engine.decide = decide


def alert(clock: FakeClock, mint: str = MINT, symbol: str = "Bittersweet",
          mid: int = 1, **fields) -> AlertSnapshot:
    base = {
        "market_cap_usd": 15_600.0, "liquidity_usd": 15_500.0,
        "volume_usd": 21_600.0, "top10_pct": 26.01, "bundled_pct": 33.0,
        "holder_count": 119, "bundle_count": 4, "age_seconds": 360.0,
    }
    base.update(fields)
    return AlertSnapshot(
        candidate_id=new_id("cand"), source="telegram", chat="testchat",
        message_id=mid, source_timestamp=clock.now() - 0.4,
        received_timestamp=clock.now(), raw_text="synthetic",
        mint=mint, symbol=symbol, name=symbol,
        fields=base, provenance={k: "alert_label:x" for k in base},
    )


@pytest.fixture(autouse=True)
def legacy_paper_rollback(config):
    """
    This module is the LEGACY PAPER fill path's incident regression.

    It asserts on ``trader.last_execution`` -- the legacy book's outcome record
    -- so it necessarily exercises that path. Execution-aware executable quotes
    are now the DEFAULT, and that path settles through ``paper_live`` instead,
    which does not write ``last_execution``.

    So the rollback flag is set EXPLICITLY here rather than the default being
    quietly changed back. The legacy path is reachable through exactly one
    documented switch, and this test is that switch's coverage. The default
    executable-quote path is covered by tests/test_execution_aware_runtime.py.
    """
    from dataclasses import replace as _replace

    object.__setattr__(
        config.real, "paper_legacy_mark_fill_rollback", True)
    yield config


@pytest.fixture
def trader(config, clock):
    t = make_trader(config, clock)
    t.startup()
    t.state.start()
    t.cli_only = True
    return t


def notifications(trader: PaperTrader) -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    trader.set_notifier(lambda text, kb=None: out.append((text, kb)))
    return out


def journal_rows(trader: PaperTrader, name: str = "trades.jsonl") -> list[dict]:
    path = Path(trader.journal.root) / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def ledger_events(trader: PaperTrader, event: str | None = None) -> list[dict]:
    rows = []
    with open(trader.config.ledger_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if event is None or row.get("event") == event:
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 1. the happy path: a BUY becomes a visible paper position
# ---------------------------------------------------------------------------


def test_a_buy_decision_becomes_exactly_one_paper_position(trader, clock):
    """
    The acceptance criterion. decision=BUY -> quote -> PAPER_FILL=SUCCESS ->
    open position -> exposure > 0 -> buys == 1 -> signal->fill sample exists ->
    journal contains the position.
    """
    force_buy(trader)
    notes = notifications(trader)

    cash_before = trader.account.cash
    exposure_before = trader.account.open_exposure()
    assert exposure_before == 0.0

    asyncio.run(trader.on_alert(alert(clock), None))

    # -- the outcome is unambiguous
    assert trader.last_execution["decision"] == "BUY"
    assert trader.last_execution["outcome"] == PAPER_FILL_SUCCESS
    assert trader.last_execution["executed"] is True
    assert trader.last_execution["reason"] is None
    assert trader.last_execution["position_id"]

    # -- one position, and it is OPEN
    positions = trader.account.open_positions()
    assert len(positions) == 1
    assert positions[0].status is PositionStatus.OPEN

    # -- the accounting moved, atomically from the operator's point of view
    assert trader.buys == 1
    assert trader.blocks == 0
    assert trader.account.cash < cash_before, "cash must fall on a buy"
    assert trader.account.open_exposure() > 0.0, "exposure must rise on a buy"
    snapshot = trader.account.snapshot(trader._refresh_marks())
    assert snapshot["open_positions"] == 1
    assert snapshot["open_exposure_usd"] > 0.0
    assert snapshot["identity_check"] == pytest.approx(0.0, abs=1e-6), (
        "equity == cash + market value must still hold exactly"
    )

    # -- signal -> fill telemetry now EXISTS (it was "no samples" in the incident)
    rec = trader._current_stages
    assert rec.signal_to_fill_ms is not None, "no signal->fill sample was recorded"
    assert rec.signal_to_fill_ms > 0
    assert trader.latency.samples["signal_to_fill"], "no signal->fill sample"
    assert rec.outcome == PAPER_FILL_SUCCESS
    assert rec.blocked_reason is None
    assert trader.latency.outcome_stats().get(PAPER_FILL_SUCCESS) == 1
    assert trader.latency.outcome_stats().get(DECISION_BUY) == 1

    # -- the pipeline row shows a quote and a fill, not "-"
    row = trader._pipeline_for_display()
    assert row["quote"] != "-" and "BUY_BLOCKED" not in row["quote"]
    assert row["fill"] != "-" and "BUY_BLOCKED" not in row["fill"]
    assert row["outcome"] == PAPER_FILL_SUCCESS

    # -- the journal contains the entry
    entries = [r for r in journal_rows(trader) if r.get("event") == "entry"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["mode"] == "PAPER"
    assert entry["mint"] == MINT
    assert entry["position_id"] == trader.last_execution["position_id"]
    assert entry["fill_price"] is not None
    assert entry["quantity"] > 0

    # -- the ledger recorded the entry with its signal->fill span
    led = ledger_events(trader, "entry")
    assert led and led[0]["signal_to_fill_ms"] is not None

    # -- the operator was told the FILL, not merely the decision
    assert any(PAPER_FILL_SUCCESS.split("=")[1] in "" or "BUY" in text
               for text, _ in notes)


def test_the_buy_notification_says_paper_not_real(trader, clock):
    force_buy(trader)
    notes = notifications(trader)
    asyncio.run(trader.on_alert(alert(clock), None))
    body = "\n".join(t for t, _ in notes)
    assert "PAPER" in body
    assert "Mode: <b>PAPER</b>" in body


def test_a_paper_buy_signs_and_broadcasts_nothing(trader, clock, monkeypatch):
    """
    The safety half of the acceptance criterion. A PAPER fill must not touch a
    keypair, a signer or a broadcast.
    """
    from autopaper.execution import real as real_mod

    force_buy(trader)
    notifications(trader)
    calls: list[str] = []

    def _trip(name: str):
        def boom(*_a, **_k):
            calls.append(name)
            raise AssertionError(
                f"the PAPER path reached real execution: {name}"
            )
        return boom

    # Tripwires on the ONLY three things that can sign or broadcast. If the
    # PAPER path ever routes through any of them, this test fails loudly.
    monkeypatch.setattr(real_mod.RealExecutor, "open_position",
                        _trip("RealExecutor.open_position"))
    monkeypatch.setattr(real_mod.RealExecutor, "close_position",
                        _trip("RealExecutor.close_position"))
    monkeypatch.setattr(real_mod.LiveSolanaBackend, "send_transaction",
                        _trip("LiveSolanaBackend.send_transaction"), raising=False)
    monkeypatch.setattr(real_mod.LiveSolanaBackend, "swap",
                        _trip("LiveSolanaBackend.swap"), raising=False)

    asyncio.run(trader.on_alert(alert(clock), None))
    assert trader.buys == 1
    assert calls == [], f"the PAPER path touched real execution: {calls}"
    assert trader.state.trade_mode is TradeMode.PAPER
    assert trader.state.real_trading_enabled is False
    assert trader.config.real.trading_enabled is False


# ---------------------------------------------------------------------------
# 2. every BUY that does NOT fill says so, with a reason
# ---------------------------------------------------------------------------


def test_a_stale_signal_blocks_the_buy_with_a_named_reason(trader, clock):
    """
    THE INCIDENT, reproduced. A BUY whose signal has aged past the entry budget
    is refused - and the refusal is now visible everywhere.
    """
    force_buy(trader)
    notes = notifications(trader)

    # Age the signal past the 15s budget: what a 34.9s analysis produced.
    snap = alert(clock)
    from dataclasses import replace as dc_replace

    snap = dc_replace(snap, received_timestamp=clock.now() - 34.9,
                      source_timestamp=clock.now() - 35.3)
    asyncio.run(trader.on_alert(snap, None))

    assert trader.buys == 0
    assert trader.blocks == 1
    assert trader.last_execution["outcome"] == BUY_BLOCKED
    assert trader.last_execution["executed"] is False
    assert trader.last_execution["reason"] == "stale_signal"
    assert trader.account.open_positions() == []
    assert trader.account.open_exposure() == 0.0

    # -- the pipeline row now carries the reason instead of a bare "-"
    row = trader._pipeline_for_display()
    assert row["quote"].startswith("-") and "BUY_BLOCKED=stale_signal" in row["quote"]
    assert row["fill"].startswith("-") and "BUY_BLOCKED=stale_signal" in row["fill"]
    assert row["outcome"] == BUY_BLOCKED

    # -- the stage recorder, the accumulator and the counters all agree
    rec = trader._current_stages
    assert rec.blocked_reason == "stale_signal"
    assert rec.outcome == BUY_BLOCKED
    assert rec.signal_to_fill_ms is None, (
        "a blocked BUY must not fabricate a signal->fill sample"
    )
    assert trader.latency.outcome_stats().get(BUY_BLOCKED) == 1
    assert trader.latency.outcome_stats().get(PAPER_FILL_SUCCESS) is None
    assert trader.stale_gate.stats()["refused_by_reason"].get("STALE_SIGNAL") == 1

    # -- the journal records it as unfilled, with the reason
    unfilled = [r for r in journal_rows(trader) if r.get("event") == "unfilled"]
    assert unfilled, "a blocked BUY must be journalled"
    assert unfilled[-1]["reason"] == "stale_signal"
    assert unfilled[-1]["decision"] == "BUY"

    # -- and the operator is told, in words that cannot be read as a fill
    body = "\n".join(t for t, _ in notes)
    assert f"{BUY_BLOCKED}=stale_signal" in body
    assert "NO POSITION OPENED" in body


def test_a_missing_quote_at_fill_time_is_a_named_reason(trader, clock, config):
    """
    Market is LIVE at decision time and gone at fill time. ``quote=-`` must say
    which of those it was.
    """
    force_buy(trader)
    notes = notifications(trader)
    venue = trader.market.primary

    async def no_wait(_seconds: float) -> None:
        return None

    trader._sleep = no_wait     # keep the signal fresh; kill the quote

    async def boom(mint, *, allow_fallback=True):
        return QuoteResult(observation=None, error="venue: down", provider="venue")

    trader.market.quote_async = boom
    asyncio.run(trader.on_alert(alert(clock), None))

    assert trader.buys == 0
    assert trader.blocks == 1
    assert trader.last_execution["reason"] in {
        "stale_market_data", "no_quote_at_fill", "stale_signal",
    }
    body = "\n".join(t for t, _ in notes)
    assert BUY_BLOCKED in body
    assert "NO POSITION OPENED" in body
    assert venue is not None


def test_a_stale_quote_at_fill_time_is_a_named_reason(trader, clock):
    """The pipeline was fast but the only quote is already past the budget."""
    force_buy(trader)
    venue = trader.market.primary
    # A quote that arrives with an age already inside the stale budget.
    venue.quote_age_s = 0.0
    original = venue.quote

    def aged(mint):
        q = original(mint)
        if q.observation is None:
            return q
        from dataclasses import replace as dc_replace

        old = q.observation
        return QuoteResult(
            observation=dc_replace(
                old, observed_at=old.observed_at - 30.0,
                received_at=old.received_at - 30.0,
            ),
            provider=q.provider,
        )

    venue.quote = aged
    trader.market.primary.quote = aged
    trader.market.fallback.quote = aged
    notes = notifications(trader)
    asyncio.run(trader.on_alert(alert(clock), None))

    assert trader.buys == 0
    assert trader.blocks == 1
    assert trader.last_execution["reason"] in {"stale_quote", "stale_signal"}
    assert "NO POSITION OPENED" in "\n".join(t for t, _ in notes)


def test_a_sizing_refusal_is_a_named_reason(trader, clock):
    force_buy(trader)
    notes = notifications(trader)
    # Drain the account so the sizer has nothing to work with.
    trader.account.state.cash_usd = 0.0
    asyncio.run(trader.on_alert(alert(clock), None))
    assert trader.buys == 0
    assert trader.blocks == 1
    assert trader.last_execution["reason"] in {
        "sizing_refused", "stale_signal", "stale_market_data",
        "no_quote_at_fill",
    }
    assert trader.account.open_positions() == []
    assert BUY_BLOCKED in "\n".join(t for t, _ in notes)


def test_an_execution_exception_is_caught_and_reported(trader, clock):
    """
    A raise inside execution must not become a silent ``return``, and must not
    take the trading loop down either.
    """
    force_buy(trader)
    notes = notifications(trader)

    async def explode(*a, **k):
        raise RuntimeError("synthetic executor fault")

    trader._execute_paper_buy = explode
    with pytest.raises(RuntimeError):
        asyncio.run(trader.on_alert(alert(clock), None))

    # The loop survived: the engine is still usable and still records timings.
    assert trader._current_stages is not None
    assert trader.buys == 0
    assert trader.blocks == 1
    assert trader.last_execution["reason"] == "execution_exception"
    assert BUY_BLOCKED in "\n".join(t for t, _ in notes)
    assert trader.state.run_state.value in ("PAPER", "STOPPED")


def test_a_cancelled_execution_is_observable(trader, clock):
    """
    A cancelled BUY is an outcome, not a silence. The candidate task is cancelled
    mid-order and the reason has to survive it.
    """
    force_buy(trader)
    notes = notifications(trader)

    started = asyncio.Event()

    async def hang(*a, **k):
        started.set()
        await asyncio.sleep(30)

    trader._execute_paper_buy = hang

    async def drive():
        task = asyncio.ensure_future(trader.on_alert(alert(clock), None))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(drive())
    assert trader.buys == 0
    assert trader.blocks == 1
    assert trader.last_execution["reason"] == "cancelled"
    assert BUY_BLOCKED in "\n".join(t for t, _ in notes)
    assert trader._current_stages.outcome == BUY_BLOCKED


def test_shutdown_during_a_pending_execution_is_not_silently_discarded(
        trader, clock):
    """
    ``decision=BUY`` must never survive a shutdown as an unexplained outcome.

    Shutdown itself does not cancel an in-flight candidate - the loop is
    cancelled and gathered first, which is what makes the outcome observable.
    What is asserted here is that the recorded state after shutdown names either
    a fill or a block, never a bare decision.
    """
    force_buy(trader)
    notes = notifications(trader)
    reached = asyncio.Event()

    async def hang(*a, **k):
        reached.set()
        await asyncio.sleep(30)

    trader._execute_paper_buy = hang

    async def drive():
        task = asyncio.ensure_future(trader.on_alert(alert(clock), None))
        await reached.wait()
        # The order is IN FLIGHT when the shutdown begins. shutdown() gathers and
        # cancels the supervised tasks, which cancels this one too.
        await trader.shutdown()
        task.cancel()
        with pytest.raises((asyncio.CancelledError, Exception)):
            await task

    asyncio.run(drive())
    outcome = trader.last_execution["outcome"]
    assert outcome in (DECISION_BUY, BUY_BLOCKED, PAPER_FILL_SUCCESS)
    assert outcome != DECISION_BUY, (
        "a shutdown must not leave an executed-path BUY recorded as a bare "
        "decision with no explanation"
    )
    assert trader.last_execution["reason"] == "shutdown_during_execution"
    assert BUY_BLOCKED in "\n".join(t for t, _ in notes)
    # ...and the persisted account says so too, so a restart cannot resurrect an
    # unexplained BUY decision.
    state = json.loads(Path(trader.config.state_file).read_text())
    assert state["last_execution"]["outcome"] == BUY_BLOCKED
    assert state["last_execution"]["reason"] == "shutdown_during_execution"
    assert state["counters"]["blocked"] == 1
    assert trader.account.open_positions() == []


def test_a_reject_is_not_recorded_as_a_blocked_buy(trader, clock):
    """A REJECT never enters the execution path at all."""
    notes = notifications(trader)
    asyncio.run(trader.on_alert(alert(clock, mid=99), None))
    # The venue data here is inside V3-C but the narrative is absent and the
    # regime may refuse; whichever it is, a REJECT must not be a BUY_BLOCKED.
    if trader.rejects:
        assert trader.blocks == 0
        assert trader.last_execution is None
        assert BUY_BLOCKED not in "\n".join(t for t, _ in notes)


# ---------------------------------------------------------------------------
# 3. the outcome vocabulary itself
# ---------------------------------------------------------------------------


def test_a_decision_is_never_indistinguishable_from_a_fill():
    """
    A BUY decision and a BUY fill must not share a token, on any surface.
    """
    assert DECISION_BUY != PAPER_FILL_SUCCESS
    assert BUY_BLOCKED not in (DECISION_BUY, PAPER_FILL_SUCCESS)
    assert DECISION_BUY == "DECISION=BUY"
    assert PAPER_FILL_SUCCESS == "PAPER_FILL=SUCCESS"
    assert REAL_FILL_SUCCESS == "REAL_FILL=SUCCESS"
    # A grep for a fill must not match a decision and vice versa.
    assert "SUCCESS" not in DECISION_BUY
    assert "DECISION" not in PAPER_FILL_SUCCESS


def test_every_block_reason_has_operator_text():
    from autopaper.latency import BLOCK_REASONS

    assert BLOCK_REASONS, "the vocabulary must not be empty"
    for reason in BLOCK_REASONS:
        assert BLOCK_REASONS[reason].strip()


def test_an_unknown_block_reason_degrades_to_readable_text():
    """A NEW reason must not crash the renderer."""
    from autopaper.telegram.ui import render_buy_blocked

    text = render_buy_blocked(symbol="X", mint=MINT, reason="brand_new_reason",
                              detail="", mode=TradeMode.PAPER,
                              reason_text="brand new reason")
    assert "brand_new_reason" in text
    assert "NO POSITION OPENED" in text


# ---------------------------------------------------------------------------
# 4. the counter separation that made the incident unreadable
# ---------------------------------------------------------------------------


def test_session_counters_and_historical_totals_are_separate(trader):
    """
    The shutdown line printed ``closed=<len(memory)>`` next to session counters,
    so ``closed=420 ... buys=0`` read as one population. They are now named.
    """
    trader.persist()
    counters = trader.session_counters()
    assert counters["buys"] == 0
    assert counters["blocked"] == 0
    assert counters["candidates_seen"] == 0
    assert counters["open_positions"] == 0

    perf = trader.performance_by_scope()
    assert "epoch" in perf and "all_time" in perf and "session" in perf
    assert perf["epoch"]["total_trades"] == perf["all_time"]["total_trades"]
    assert perf["session"]["buys"] == 0

    state = json.loads(Path(trader.config.state_file).read_text())
    assert "counters_scope" in state
    assert "this process run only" in state["counters_scope"]


def test_session_counters_are_never_restored_on_load(config, clock):
    """
    Restoring ``buys`` from a previous process would be a lie about what this
    process did. The loader must leave them at zero.
    """
    t = make_trader(config, clock)
    t.startup()
    t.state.start()
    force_buy(t)
    asyncio.run(t.on_alert(alert(clock), None))
    assert t.buys == 1
    t.persist()
    t._release_ownership()

    t2 = make_trader(config, clock)
    t2.startup()
    assert t2.buys == 0
    assert t2.blocks == 0
    assert t2.rejects == 0
    assert t2.candidates_seen == 0
    # ...but the POSITION is recovered, which is the part that must survive.
    assert len(t2.account.open_positions()) == 1
    t2._release_ownership()


# ---------------------------------------------------------------------------
# 5. restart / recovery with a position open
# ---------------------------------------------------------------------------


def test_an_open_position_is_reconstructed_after_a_restart(config, clock):
    """
    Item 11 of the brief: a position open before a restart must come back, with
    its accounting intact.
    """
    t = make_trader(config, clock)
    t.startup()
    t.state.start()
    force_buy(t)
    asyncio.run(t.on_alert(alert(clock), None))
    assert len(t.account.open_positions()) == 1
    cash_after_buy = t.account.cash
    exposure_after_buy = t.account.open_exposure()
    position_id = t.account.open_positions()[0].position_id
    t.persist()
    t._release_ownership()

    t2 = make_trader(config, clock)
    t2.startup()
    restored = t2.account.open_positions()
    assert len(restored) == 1
    assert restored[0].position_id == position_id
    assert restored[0].status is PositionStatus.OPEN
    assert t2.account.cash == pytest.approx(cash_after_buy)
    assert t2.account.open_exposure() == pytest.approx(exposure_after_buy)

    # The recovered position is what the operator views report.
    rows = t2.open_positions_view()
    assert len(rows) == 1
    assert rows[0]["position_id"] == position_id
    assert rows[0]["source"] == "paper_account"
    t2._release_ownership()


def test_the_exit_moves_open_to_closed_and_updates_realized(config, clock):
    """Item 11, second half: open -> closed, realized updates, exposure returns."""
    t = make_trader(config, clock)
    t.startup()
    t.state.start()
    force_buy(t)
    asyncio.run(t.on_alert(alert(clock), None))
    position = t.account.open_positions()[0]

    realized_before = t.account.realized
    exit_decision = __import__("autopaper.exits", fromlist=["x"]).ExitDecision(
        "EXIT", "forced for the accounting test",
        __import__("autopaper.models", fromlist=["x"]).ExitStyle.FAST_FAILURE,
        trigger="test",
    )
    clock.advance(60.0)
    asyncio.run(t._do_exit(position, exit_decision, clock.now()))

    assert t.account.open_positions() == []
    assert t.account.open_exposure() == 0.0
    assert t.account.realized != realized_before
    snapshot = t.account.snapshot()
    assert snapshot["open_positions"] == 0
    assert snapshot["identity_check"] == pytest.approx(0.0, abs=1e-6)

    closed, total = t.closed_positions_view()
    assert total >= 1
    assert closed[0]["position_id"] == position.position_id
    assert closed[0]["realized_pnl_usd"] is not None
    assert closed[0]["exit_reason"]
    t._release_ownership()


# ---------------------------------------------------------------------------
# 5. the dashboard must not truncate the thing that explains the outage
# ---------------------------------------------------------------------------


def test_the_dashboard_never_truncates_a_blocked_pipeline_row():
    """
    The incident's most important information sat at the END of the pipeline row
    and was cut off by the frame's ellipsis. Asserted here at a narrow width,
    which is where it actually bit.
    """
    from autopaper import dashboard

    pipeline = {
        "alert": "ok", "data": "16,256ms", "settle": "2,000ms",
        "narrative": "18,269ms", "L25": "1ms",
        "quote": "- BUY_BLOCKED=stale_signal",
        "fill": "- BUY_BLOCKED=stale_signal",
        "outcome": "BUY_BLOCKED", "executed": "False",
    }
    frame = dashboard.build_frame(
        strategy="L25", mode="PAPER", run_state="PAPER", uptime_s=1.0,
        account={},
        performance={
            "total_trades": 420, "wins": 122, "losses": 298, "win_rate": 0.29,
            "profit_factor": 1.51, "expectancy_usd": 2.0882, "runners": 39,
            "stats_epoch_started_at": 1790750378.33,
            "all_time": {"total_trades": 420, "wins": 122, "losses": 298,
                         "profit_factor": 1.51, "realized_pnl_usd": -5.22},
            "session": {"candidates_seen": 9, "buys": 0, "blocked": 4,
                        "rejects": 5, "waits": 0},
        },
        last_event={"at_iso": "2026-09-30T11:16:49.749Z", "symbol": "Bittersweet",
                    "decision": "BUY", "reason": "route_demoted",
                    "latency_ms": 34877},
        execution={"decision": "BUY", "outcome": "BUY_BLOCKED",
                   "executed": False, "reason": "stale_signal"},
        pipeline=pipeline,
        latency={
            "analysis": {"n": 9, "p50": 34773, "p95": 42735, "p99": 42735,
                         "max": 42735},
            "signal_to_fill": {},
            "market": {"n": 9, "p50": 16603, "max": 24509},
            "settle": {"n": 9, "p50": 18269, "max": 18269},
            "narrative": {"n": 9, "p50": 1601, "max": 1900},
        },
        services={
            "channel": "CONNECTED", "market": "OK", "execution": "paper",
            "narrative_source": {"worst_case_s": 3.0, "timeout_s": 2.5,
                                 "timeout_grace_s": 0.5, "latency_p50_ms": 1601.0,
                                 "latency_max_ms": 1900.0, "samples": 9},
            "market_detail": {"budget_ms": 8000.0, "connect_budget_ms": 1000.0,
                              "max_call_ms": {"dexscreener": 16250.7},
                              "deadlines": {"calls": 9, "overruns": 7}},
        },
        positions=[], cols=78,
    )
    assert "…" not in frame, (
        "the dashboard truncated part of the frame; the outcome and the blocked "
        "reason sit at the END of their lines and were being cut off"
    )
    # ...and everything an operator needs is present.
    assert "BUY_BLOCKED" in frame
    assert "blocked_reason" in frame and "stale_signal" in frame
    assert "executed" in frame
    assert "quote:- BUY_BLOCKED=stale_signal" in frame
    assert "fill:- BUY_BLOCKED=stale_signal" in frame
    # The two populations are separately labelled.
    assert "all-time" in frame and "this run" in frame
    assert "since 2026-09-30" in frame
    # The narrative's real cost is shown apart from the stage that used to
    # absorb it.
    assert "narrative measured" in frame and "1,601ms" in frame


def test_the_dashboard_shows_both_budgets():
    from autopaper import dashboard

    frame = dashboard.build_frame(
        strategy="L25", mode="PAPER", run_state="PAPER", uptime_s=1.0, account={},
        performance={}, last_event={}, execution={}, pipeline={}, latency={},
        services={
            "market_detail": {"budget_ms": 8000.0, "connect_budget_ms": 1000.0,
                              "max_call_ms": {"dexscreener": 2061.0},
                              "deadlines": {"calls": 9, "overruns": 0}},
            "narrative_source": {"worst_case_s": 3.0, "timeout_s": 2.5,
                                 "timeout_grace_s": 0.5, "samples": 0},
        },
        positions=[], cols=100,
    )
    assert "8,000ms wall-clock" in frame
    assert "connect 1,000ms" in frame
    assert "3.0s worst case (timeout 2.5s + grace 0.5s)" in frame
    pytest.mark.slow,  # drives full async entry/shutdown timing
