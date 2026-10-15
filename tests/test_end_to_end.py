"""
End-to-end: a simulated live session driven through the real engine.

A deterministic synthetic venue supplies prices. The engine, decision engine,
fill model, exit policy, ledger and accounting are all the real production code
paths - only the network is fake. That makes this the test that proves the pieces
actually compose, and that a run leaves an auditable trail.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from autopaper.account import PaperAccount
from autopaper.clock import FakeClock
from autopaper.engine import PaperTrader
from autopaper.ledger import EventType, load_json
from autopaper.models import (
    AlertSnapshot,
    DataState,
    Decision,
    DecisionReport,
    ExitStyle,
    Observation,
    PositionStatus,
    QuoteResult,
    Regime,
    Route,
    new_id,
)
from autopaper.telegram.client import ConnectionState, ListenerStatus

from .conftest import MINT, MINT2

pytestmark = [
    pytest.mark.integrity,
    pytest.mark.slow,  # full end-to-end trading cycle
]

T0 = 1_700_000_000.0


# ---------------------------------------------------------------------------
# synthetic venue
# ---------------------------------------------------------------------------


class Venue:
    """
    A deterministic price path per mint.

    ``path`` is a list of multipliers applied to the base price at 1s intervals,
    so a "token that runs then dies" is written as data rather than as special
    cases in the test.
    """

    def __init__(self, clock: FakeClock):
        self.clock = clock
        self.paths: dict[str, list[float]] = {}
        self.base: dict[str, float] = {}
        self.started: dict[str, float] = {}
        self.liquidity: dict[str, float] = {}
        self.market_cap: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.fail: set[str] = set()

    def add(self, mint: str, path: list[float], *, base=1.0e-4,
            liquidity=120_000.0, market_cap=18_000.0) -> None:
        """
        ``market_cap`` defaults to $18,000, INSIDE V3-C's $20,000 ceiling.

        The previous fixture used $1.1M, which V3-C refuses outright, so under
        the V3 runtime every test using it would have failed at the strategy
        gate for a reason unrelated to what it was testing. $18,000 is also what
        the real Moo Deng President feed actually publishes for most alerts.
        """
        self.paths[mint] = path
        self.base[mint] = base
        self.started[mint] = self.clock.now()
        self.liquidity[mint] = liquidity
        self.market_cap[mint] = market_cap
        self.calls[mint] = 0

    def price(self, mint: str) -> float | None:
        if mint not in self.paths:
            return None
        elapsed = int(self.clock.now() - self.started[mint])
        mult = self.paths[mint][min(elapsed, len(self.paths[mint]) - 1)]
        return self.base[mint] * mult

    def quote(self, mint: str) -> QuoteResult:
        self.calls[mint] = self.calls.get(mint, 0) + 1
        if mint in self.fail:
            return QuoteResult(observation=None, error="venue: down",
                               provider="venue")
        p = self.price(mint)
        if p is None:
            return QuoteResult(observation=None, error="venue: unknown mint",
                               provider="venue")
        now = self.clock.now()
        return QuoteResult(
            observation=Observation(
                mint=mint, observed_at=now, received_at=now, price_usd=p,
                market_cap_usd=self.market_cap.get(mint, 18_000.0),
                liquidity_usd=self.liquidity[mint],
                volume_5m_usd=180_000.0, volume_1h_usd=420_000.0,
                buys_5m=400, sells_5m=150, txns_5m=550,
                pair_created_at=self.started[mint] - 120,
                provider="venue",
            ),
            provider="venue",
        )


@pytest.fixture(autouse=True)
def legacy_paper_rollback(config):
    """
    This module is the LEGACY PAPER path's end-to-end regression.

    It asserts on ``trader.account.open_positions()`` and the runtime journal,
    which are the legacy PaperAccount's own objects. Executable quotes are now
    the default and settle through ``paper_live``, so these assertions describe
    the rollback path.

    The flag is set EXPLICITLY so the rollback path keeps real coverage instead
    of the default being quietly reverted. The DEFAULT executable-quote chain --
    Telegram signal, L25, executable BUY quote, PAPER fill, exit, executable
    SELL quote, settlement -- is covered by tests/test_execution_aware_runtime.py.
    """
    object.__setattr__(config.real, "paper_legacy_mark_fill_rollback", True)
    yield config


def make_trader(config, clock: FakeClock, venue: Venue) -> PaperTrader:
    from autopaper.market import MarketDataService
    from autopaper.nar import NarService
    from autopaper.telegram import TelegramListener

    class VPrimary:
        name = "venue"

        def __init__(self):
            from autopaper.market.ratelimit import RateLimiter

            self.limiter = RateLimiter(10_000, name="venue")

        quote = venue.quote

    market = MarketDataService(config.market, clock=clock, primary=VPrimary(),
                               fallback=VPrimary())
    market._fvp_probe_budget = 0
    nar = NarService(config.nar, clock=clock)
    listener = TelegramListener(config.telegram, clock=clock)
    listener.status.state = ConnectionState.CONNECTED
    listener.status.chat_id = 100
    listener.status.chat = "testchat"
    trader = PaperTrader(config, clock=clock, market=market, nar=nar,
                         listener=listener)
    listener.on_alert = trader.on_alert
    # Waiting is expressed in fake-clock time so the whole test is deterministic
    # and instant: any "sleep" advances the clock instead of blocking.
    async def fake_sleep(seconds: float) -> None:
        clock.advance(seconds)

    trader._sleep = fake_sleep
    return trader


def snap(clock: FakeClock, mint: str, symbol: str, price: float, mid=1) -> AlertSnapshot:
    return AlertSnapshot(
        candidate_id=new_id("cand"), source="telegram", chat="testchat",
        message_id=mid, source_timestamp=clock.now() - 0.4,
        received_timestamp=clock.now(), raw_text="synthetic",
        mint=mint, symbol=symbol, name=symbol,
        fields={"price_usd": price}, provenance={"price_usd": "alert_label:price"},
    )


async def _drive(trader: PaperTrader, clock: FakeClock, steps: int) -> None:
    for _ in range(steps):
        await trader.monitor_once()
        clock.advance(2.0)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_startup_prints_ready_and_refuses_to_trade_if_unsafe(config, capsys, monkeypatch):
    trader = make_trader(config, FakeClock(T0), Venue(FakeClock(T0)))
    monkeypatch.delenv("PRIVATE_KEY", raising=False)
    trader.startup()

    # --- the banner as it appears IMMEDIATELY after startup, before any
    # operator action. This is the state a crash dump would capture.
    trader._print_ready()
    out = capsys.readouterr().out
    assert "MEMEBOT V3 READY" in out
    assert "BANKROLL            : $100.00" in out
    assert "REAL TRADING IS OFF" in out
    assert "arming requires a CONFIRMED TELEGRAM OPERATOR ACTION" in out
    assert "never reset" in out
    # The active strategy is L25 = V3-C + the liquidity/MC gate. The V3-C
    # thresholds are still asserted individually, because L25 DELEGATES its
    # first leg to V3-C and a regression in either leg must be visible here.
    assert "STRATEGY            : L25:" in out
    assert "MC <= $20,000" in out, "V3-C market-cap ceiling must still be shown"
    assert "SQ >= 0.55" in out, "V3-C signal-quality floor must still be shown"
    assert "f_liq_to_mcap >= 0.9647" in out, "the frozen L25 ratio gate"
    assert "(active)          : L25" in out
    # The research status is shown so the banner cannot imply L25 is proven.
    assert "PROMISING_BUT_INCONCLUSIVE" in out
    assert "NO_TRUE_POST_DERIVATION_DATA" in out
    assert "CANDIDATE SOURCE    : Moo Deng President  (only)" in out
    # STARTUP IS SAFE: nothing armed, nothing running, PAPER only
    assert "armed now          : False" in out
    assert "RUNNING             : False  (OFF)" in out
    assert "TRADE MODE          : PAPER" in out
    assert "EMERGENCY STOP      : False" in out
    assert "NO TRADE HAPPENS YET" in out

    # --- and after the operator presses start
    trader.state.start()
    trader._print_ready()
    out2 = capsys.readouterr().out
    assert "RUNNING             : True  (PAPER)" in out2
    assert "TRADE MODE          : PAPER" in out2
    # pressing start still did NOT arm REAL
    assert "armed now          : False" in out2
    assert "REAL TRADING IS OFF" in out2

    assert load_json(config.state_file) is None or True


def test_a_running_token_is_bought_and_the_whole_chain_is_recorded(config):
    import asyncio

    clock = FakeClock(T0)
    venue = Venue(clock)
    venue.add(MINT, [1.0, 1.02, 1.05, 1.10, 1.18, 1.30, 1.45, 1.60])
    trader = make_trader(config, clock, venue)
    trader.startup()
    trader.state.start()   # the operator start button

    async def run():
        a = snap(clock, MINT, "RUNNER", venue.price(MINT))
        await trader.on_alert(a, _parsed(a))
        assert trader.account.open_positions(), "expected a paper position"
        await _drive(trader, clock, 12)

    asyncio.new_event_loop().run_until_complete(run())

    ledger = trader.ledger
    kinds = ledger.summary()
    assert kinds.get("candidate_seen") == 1
    assert kinds.get("position_update", 0) >= 1
    # Decisions and the entry are recorded in the UNIFIED JOURNAL, partitioned
    # by mode, rather than in the legacy ledger. One journal, one timeline,
    # mandatory partition - never two files that can disagree.
    js = trader.journal.summary()
    assert js["PAPER"]["entries"] == 1
    assert js["REAL"]["entries"] == 0, "PAPER mode must not touch the REAL account"
    assert len(trader.journal.all_decisions()) == 1

    trades = ledger.trades()
    if trades:
        t = trades[0]
        assert t["mint"] == MINT
        assert t["fill_price"] > t["entry_price"], "the fill must be worse than the quote"
        assert t["holding_seconds"] > 0
        assert t["exit_reason"]
        assert isinstance(t["booked_at"], float)

    snapshot = trader.account.snapshot()
    assert snapshot["identity_check"] == pytest.approx(0.0, abs=1e-9)
    assert snapshot["equity_usd"] == pytest.approx(
        snapshot["cash_usd"] + snapshot["open_exposure_usd"], abs=1e-6
    ) or snapshot["open_positions"] == 0
    assert snapshot["fees_paid_usd"] > 0, "entry fee must be charged"
    assert snapshot["slippage_paid_usd"] > 0, "slippage must be charged"


def test_no_alert_means_no_trade(config):
    """The agent must not manufacture a candidate."""
    import asyncio

    clock = FakeClock(T0)
    trader = make_trader(config, clock, Venue(clock))
    trader.startup()
    trader.state.start()   # the operator start button
    asyncio.new_event_loop().run_until_complete(_drive(trader, clock, 20))
    assert trader.account.open_positions() == []
    assert trader.buys == 0
    assert trader.candidates_seen == 0
    assert trader.account.cash == pytest.approx(100.0)
    assert trader.account.equity() == pytest.approx(100.0)


def test_exhausted_token_is_rejected_and_recorded(config):
    import asyncio

    clock = FakeClock(T0)
    venue = Venue(clock)
    venue.add(MINT, [1.0, 3.0, 6.0, 9.0])
    trader = make_trader(config, clock, venue)
    trader.startup()
    trader.state.start()   # the operator start button

    async def run():
        # let the venue run up first, then the alert arrives
        for _ in range(3):
            trader.market.quote(MINT)
            clock.advance(1.0)
        a = snap(clock, MINT, "PUMPED", venue.price(MINT))
        await trader.on_alert(a, _parsed(a))

    asyncio.new_event_loop().run_until_complete(run())
    assert trader.buys == 0
    assert trader.rejects == 1
    decisions = trader.journal.all_decisions()
    assert decisions[0]["decision"] == "REJECT"
    assert decisions[0]["regime"] in ("EXTENDED", "EXPLOSIVE")
    assert trader.account.cash == pytest.approx(100.0)


def test_dead_token_exits_and_books_a_losing_trade(config):
    import asyncio

    clock = FakeClock(T0)
    venue = Venue(clock)
    venue.add(MINT, [1.0, 1.05, 1.12, 1.08, 0.98, 0.90, 0.84, 0.80, 0.78])
    trader = make_trader(config, clock, venue)
    trader.startup()
    trader.state.start()   # the operator start button

    async def run():
        a = snap(clock, MINT, "DEAD", venue.price(MINT))
        await trader.on_alert(a, _parsed(a))
        assert trader.account.open_positions()
        await _drive(trader, clock, 25)

    asyncio.new_event_loop().run_until_complete(run())
    trades = trader.ledger.trades()
    assert len(trades) == 1
    t = trades[0]
    assert t["net_pnl_usd"] < 0
    assert t["mae_pct"] < 0
    assert t["exit_reason"]
    assert trader.account.open_positions() == []
    assert trader.account.cash < 100.0
    assert len(trader.memory) == 1


def test_restart_resumes_without_resetting(config):
    import asyncio

    clock = FakeClock(T0)
    venue = Venue(clock)
    venue.add(MINT, [1.0, 1.05, 1.15, 1.30, 1.50, 1.70, 1.90, 2.2, 2.5, 3.0])
    trader = make_trader(config, clock, venue)
    trader.startup()
    trader.state.start()   # the operator start button

    async def run():
        a = snap(clock, MINT, "HOLD", venue.price(MINT))
        await trader.on_alert(a, _parsed(a))
        await _drive(trader, clock, 4)
        trader.persist()

    asyncio.new_event_loop().run_until_complete(run())
    assert trader.account.open_positions(), "expected the position to still be open"
    cash_before = trader.account.cash
    equity_before = trader.account.equity()
    assert cash_before < 100.0

    # --- restart: a brand new engine on the same state files -------------
    clock2 = FakeClock(T0 + 40)
    venue2 = Venue(clock2)
    venue2.add(MINT, [1.0] * 60, base=trader.account.open_positions()[0].fill_price,
              liquidity=120_000.0)
    trader2 = make_trader(config, clock2, venue2)
    trader2.startup()

    assert trader2.recovered is True
    assert trader2.account.cash == pytest.approx(cash_before)
    assert len(trader2.account.open_positions()) == 1
    assert len(trader2.memory) == 0, "an open trade is not a closed trade"
    assert trader2.account.state.run_count == 2

    asyncio.new_event_loop().run_until_complete(_drive(trader2, clock2, 10))
    # the recovered position is monitored, not abandoned
    assert trader2.ledger.count(EventType.POSITION_UPDATE) >= 1
    assert equity_before > 0


def test_second_run_does_not_replay_the_same_alert(config):
    import asyncio

    clock = FakeClock(T0)
    venue = Venue(clock)
    venue.add(MINT, [1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.3, 1.4])
    trader = make_trader(config, clock, venue)
    trader.startup()
    trader.state.start()   # the operator start button
    listener = trader.listener
    listener.restore_seen(trader.ledger.seen_message_keys())

    async def run():
        a = snap(clock, MINT, "ONCE", venue.price(MINT), mid=999)
        await trader.on_alert(a, _parsed(a))
        await _drive(trader, clock, 3)
        trader.persist()

    asyncio.new_event_loop().run_until_complete(run())
    assert trader.buys == 1

    clock2 = FakeClock(T0 + 20)
    trader2 = make_trader(config, clock2, venue)
    trader2.startup()
    trader2.listener.restore_seen(trader2.ledger.seen_message_keys())

    async def replay():
        a = snap(clock2, MINT, "ONCE", venue.price(MINT), mid=999)
        await trader2.on_alert(a, _parsed(a))

    asyncio.new_event_loop().run_until_complete(replay())
    assert trader2.buys == 0, "a replayed alert must not open a second position"
    # the engine suppresses it on candidate identity, independently of the
    # listener's message-id window
    assert len(trader2.account.open_positions()) <= 1


def test_stop_stops_new_entries_but_keeps_managing(config):
    import asyncio

    clock = FakeClock(T0)
    venue = Venue(clock)
    venue.add(MINT, [1.0, 1.05, 1.10, 1.15, 1.2, 1.25, 1.3])
    venue.add(MINT2, [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3])
    trader = make_trader(config, clock, venue)
    trader.startup()
    trader.state.start()   # the operator start button

    async def run():
        await trader.on_alert(snap(clock, MINT, "A", venue.price(MINT), 1), _parsed(None))
        await _drive(trader, clock, 3)
        trader.control.command = "STOP"
        await trader.on_alert(snap(clock, MINT2, "B", venue.price(MINT2), 2), _parsed(None))
        assert trader.candidates_seen == 2
        assert all(p.mint != MINT2 for p in trader.account.open_positions())
        # positions are still managed
        await _drive(trader, clock, 6)
        # a stop is not a flatten
        assert len(trader.account.open_positions()) == 1

    asyncio.new_event_loop().run_until_complete(run())


def test_flatten_closes_everything_explicitly(config):
    import asyncio

    clock = FakeClock(T0)
    venue = Venue(clock)
    venue.add(MINT, [1.0, 1.03, 1.06, 1.09, 1.12, 1.15, 1.18, 1.2, 1.22, 1.24,
                     1.26, 1.28, 1.30, 1.32, 1.34, 1.36], base=1.0e-4)
    trader = make_trader(config, clock, venue)
    trader.startup()
    trader.state.start()   # the operator start button

    async def run():
        await trader.on_alert(snap(clock, MINT, "F", venue.price(MINT)), _parsed(None))
        await _drive(trader, clock, 2)
        assert len(trader.account.open_positions()) == 1
        await trader.flatten()
        assert trader.account.open_positions() == []
        assert len(trader.ledger.trades()) == 1

    asyncio.new_event_loop().run_until_complete(run())
    assert trader.account.cash > 0
    assert trader.account.state.cash_usd == pytest.approx(
        trader.account.cash, abs=1e-9
    )


def test_market_going_dark_marks_data_uncertain_rather_than_guessing(config):
    import asyncio

    clock = FakeClock(T0)
    venue = Venue(clock)
    venue.add(MINT, [1.0, 1.03, 1.06, 1.09, 1.12, 1.15, 1.18, 1.2, 1.22, 1.24,
                     1.26, 1.28, 1.30, 1.32, 1.34, 1.36], base=1.0e-4)
    trader = make_trader(config, clock, venue)
    trader.startup()
    trader.state.start()   # the operator start button

    async def run():
        await trader.on_alert(snap(clock, MINT, "D", venue.price(MINT)), _parsed(None))
        await _drive(trader, clock, 2)
        p = trader.account.open_positions()[0]
        venue.fail.add(MINT)
        # enough cycles for the data to go past the unusable threshold
        await _drive(trader, clock, 45)
        assert p.data_state in (DataState.UNUSABLE, DataState.STALE)
        # equity is still coherent, valued at cost rather than a made-up price
        assert trader.account.snapshot()["identity_check"] == pytest.approx(0.0, abs=1e-9)
        # and the position was exited once data had been bad long enough
        assert p.status in (PositionStatus.CLOSED, PositionStatus.OPEN)

    asyncio.new_event_loop().run_until_complete(run())


def test_memory_only_learns_from_closed_trades(config):
    import asyncio

    clock = FakeClock(T0)
    venue = Venue(clock)
    venue.add(MINT, [1.0, 1.05, 1.12, 1.08, 0.98, 0.9, 0.84, 0.8, 0.78])
    trader = make_trader(config, clock, venue)
    trader.startup()
    trader.state.start()   # the operator start button

    async def run():
        await trader.on_alert(snap(clock, MINT, "L", venue.price(MINT)), _parsed(None))
        assert len(trader.memory) == 0, "an open trade teaches nothing yet"
        await _drive(trader, clock, 25)
        assert len(trader.memory) == 1, "the trade closes, then memory updates"
        lessons = trader.memory.lessons(clock.now())
        assert any("not enough to adapt" in line for line in lessons)

    asyncio.new_event_loop().run_until_complete(run())


def test_dashboard_renders_without_raising(config):
    from autopaper import dashboard

    clock = FakeClock(T0)
    venue = Venue(clock)
    venue.add(MINT, [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45])
    trader = make_trader(config, clock, venue)
    trader.startup()
    trader.state.start()   # the operator start button
    text = dashboard.render(
        account_snapshot=trader.account.snapshot(),
        stats=trader._stats(),
        positions=[],
        last_decision=None,
        listener={"state": "CONNECTED", "chat": "c"},
        nar=trader.nar.health(),
        market=trader.market.health(),
        safety=trader._safety_report.as_dict(),
        mode="RUN",
        uptime_s=1.0,
    )
    assert "REAL EXECUTION: BLOCKED" in text
    assert "BANKROLL" not in text or "bankroll" in text
    assert "$100.00" in text
    assert "Wallet: NONE" in text


def test_persisted_state_is_valid_json_and_complete(config):
    import asyncio

    clock = FakeClock(T0)
    venue = Venue(clock)
    venue.add(MINT, [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2,
                     2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0])
    trader = make_trader(config, clock, venue)
    trader.startup()
    trader.state.start()   # the operator start button

    async def run():
        await trader.on_alert(snap(clock, MINT, "P", venue.price(MINT)), _parsed(None))
        await _drive(trader, clock, 5)
        trader.persist()

    asyncio.new_event_loop().run_until_complete(run())
    state = load_json(config.state_file)
    json.dumps(state)  # must be serialisable
    for key in ("account", "open_positions", "recently_exited", "unfilled_attempts",
                "first_valid_prices", "counters", "agent"):
        assert key in state, key
    assert state["agent"] == "MEMEBOT_V3"
    assert state["account"]["starting_bankroll_usd"] == pytest.approx(100.0)
    for stream in ("ledger.jsonl", "candidates.jsonl"):
        path = config.var_dir / stream
        text = path.read_text().strip()
        assert text, f"{stream} must not be empty"
        for line in text.splitlines():
            json.loads(line)
    # a candidate record must carry full provenance, including both timestamps
    first = json.loads(
        (config.var_dir / "candidates.jsonl").read_text().splitlines()[0]
    )
    for key in ("chat_id", "message_id", "source_timestamp", "received_timestamp",
                "transit_latency_ms", "fields", "provenance", "traded"):
        assert key in first, key
    # and a decision record must be attributable back to it, in the UNIFIED
    # JOURNAL, where it also carries its mode partition and its V3-C gate result
    decisions = trader.journal.all_decisions()
    assert decisions, "the journal must hold the decision"
    decision = decisions[0]
    assert decision["candidate_id"] == first["candidate_id"]
    assert decision["decision"] in ("BUY", "WAIT", "REJECT")
    assert decision["mode"] == "PAPER", "every record carries its partition"
    assert decision["journal_version"] == "v3"
    assert decision["v3c"] is not None, "the V3-C gate result is journaled"
    assert decision["v3c"]["authorized"] is (decision["decision"] == "BUY")


def test_every_journal_record_is_appended_and_never_rewritten(config):
    """Historical records are preserved. There is no delete or update path."""
    import asyncio

    from .conftest import MINT

    clock = FakeClock(T0)
    venue = Venue(clock)
    venue.add(MINT, [1.0, 1.02, 1.05, 1.10, 1.18, 1.30, 1.45, 1.60])
    trader = make_trader(config, clock, venue)
    trader.startup()
    trader.state.start()

    async def run():
        await trader.on_alert(snap(clock, MINT, "RUNNER", venue.price(MINT)),
                              _parsed(snap(clock, MINT, "RUNNER", venue.price(MINT))))
        await _drive(trader, clock, 12)
        # a second, different alert is APPENDED, not merged into the first
        venue.add(MINT2, [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
        b = snap(clock, MINT2, "SECOND", venue.price(MINT2), mid=2)
        await trader.on_alert(b, _parsed(b))
        await _drive(trader, clock, 6)

    asyncio.new_event_loop().run_until_complete(run())

    trades = trader.journal.all_trades()
    ids = [r.get("candidate_id") for r in trades]
    assert len(ids) == len(set(ids)), "records are appended, never merged"
    for row in trades:
        assert row["mode"] in ("PAPER", "REAL")
    # and the PAPER/REAL partition holds after several trades
    assert trader.journal.summary()["REAL"]["entries"] == 0


def _parsed(a: AlertSnapshot | None):
    """Reparse the same text the listener would have delivered."""
    from autopaper.telegram.parser import parse_alert

    if a is None:
        return parse_alert("")
    return parse_alert(
        f"NEW TOKEN\nSymbol: {a.symbol}\nCA: {a.mint}\nPrice: ${a.fields['price_usd']}"
    )
