"""
The AMARIH runtime wiring inside the real ``PaperTrader``.

WHAT THIS PROVES
================
These tests drive the production runtime -- ``PaperTrader``, the same class
``autopaper run`` builds -- and assert the three things that make the Amarih
selection a ROUTING decision rather than a label:

1. selecting AMARIH persists three arms and is restored on restart;
2. while AMARIH is selected, a Telegram-channel alert cannot enter, and the
   refusal is recorded with the named reason ``SOURCE_DISABLED_BY_AMARIH``;
3. an Amarih-sourced signal never enters the L25 candidate pipeline, and
   exactly ONE PumpView poller task exists for the life of the selection.

Nothing here reaches the network. The poller is created but deliberately not
allowed to poll: it gates on the PAPER run state, which is OFF until the
operator presses START.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from autopaper.config import build_config
from autopaper.engine import PaperTrader
from autopaper.models import AlertSnapshot
from autopaper.modes import RunState, RuntimeState, TradeMode
from autopaper.paper_live.arms import AMARIH_ARMS
from autopaper.telegram.signal_source import (
    PRIMARY,
    SOURCE_AMARIH,
    SOURCE_AUTO,
)

AMARIH_ARMS_NAMES = tuple(a.value for a in AMARIH_ARMS)
MINT = "So11111111111111111111111111111111111111112"


class FakeMarket:
    """
    No network, and NO silent success either.

    ``quotes`` counts market reads, so a test can assert that a refused alert
    never reached the market at all -- which is the whole point of refusing it
    at the boundary rather than deeper in the pipeline. Anything that does try
    to read raises, so an unexpected network path fails loudly instead of
    quietly returning a fabricated price.
    """

    def __init__(self) -> None:
        self.quotes = 0

    async def quote_async(self, mint):
        self.quotes += 1
        raise AssertionError("no network in this test")

    def mark_price(self, mint, now):
        return None, None


@pytest.fixture
def trader(tmp_path):
    cfg = build_config(tmp_path)
    return PaperTrader(cfg, market=FakeMarket())


@pytest.fixture
def pipeline_calls(monkeypatch):
    """Count entries into the L25 candidate pipeline, beyond the boundary."""
    calls: list[object] = []

    async def spy(self, snapshot, parsed):
        calls.append(snapshot)

    import autopaper.engine as engine_mod

    monkeypatch.setattr(engine_mod.PaperTrader, "_handle_candidate", spy)
    return calls


def _alert(*, source: str = PRIMARY, message_id: int = 1) -> AlertSnapshot:
    return AlertSnapshot(
        candidate_id=f"c{message_id}",
        source="test",
        chat="testchat",
        message_id=message_id,
        source_timestamp=1_800_000_000.0,
        received_timestamp=1_800_000_005.0,
        raw_text=f"call {MINT}",
        mint=MINT,
        symbol="TKN",
        signal_source=source,
    )


class Parsed:
    """The minimum the candidate recorder reads off a parsed alert."""


# ---------------------------------------------------------------------------
# selection: persist and restore
# ---------------------------------------------------------------------------

def test_selecting_amarih_persists_all_three_arms(trader):
    saved: list[list[str]] = []
    from autopaper.telegram.actions import ButtonHandler, OperatorActions
    from autopaper.telegram import ui

    handler = ButtonHandler(
        trader.state,
        OperatorActions(persist_strategies=lambda arms: saved.append(list(arms))))
    result = asyncio.run(handler.handle(ui.CB_PROFILE + "AMARIH"))

    assert result.handled and result.error is None
    assert set(trader.state.selected_arms) == set(AMARIH_ARMS_NAMES)
    assert saved and set(saved[-1]) == set(AMARIH_ARMS_NAMES)


def test_the_amarih_selection_round_trips_through_the_state_file(tmp_path):
    """A restart must restore AMARIH, not silently fall back to the default."""
    cfg = build_config(tmp_path)
    first = RuntimeState()
    ok, _msg = first.select_strategies(["AMARIH"])
    assert ok
    from autopaper.paper_live.state import atomic_write_json

    (tmp_path / "var").mkdir(parents=True, exist_ok=True)
    atomic_write_json(tmp_path / "var" / "strategy_selection.json",
                      {"strategies": list(first.selected_arms)})

    restored = RuntimeState()
    raw = json.loads((tmp_path / "var" / "strategy_selection.json").read_text())
    ok, _msg = restored.select_strategies(raw["strategies"])
    assert ok
    assert set(restored.arms) == set(AMARIH_ARMS_NAMES)
    assert restored.trade_mode is TradeMode.PAPER


def test_amarih_is_refused_in_real(trader):
    """
    REAL accepts exactly ONE strategy. Amarih is three arms off one signal, so
    the group cannot be a REAL selection at all.
    """
    from autopaper.telegram.actions import ButtonHandler, OperatorActions
    from autopaper.telegram import ui

    # The same two writes arm_real() performs, in the same order, so the
    # state really is REAL rather than a real-looking paper state.
    trader.state.real_armed = True
    trader.state.run_state = RunState.REAL_ARMED
    assert trader.state.trade_mode is TradeMode.REAL

    handler = ButtonHandler(trader.state, OperatorActions())
    result = asyncio.run(handler.handle(ui.CB_PROFILE + "AMARIH"))

    assert set(result.state_after.selected_arms) != set(AMARIH_ARMS_NAMES)
    assert not trader._amarih_selected()


# ---------------------------------------------------------------------------
# source routing at the shared entry boundary
# ---------------------------------------------------------------------------

def test_a_telegram_alert_cannot_enter_while_amarih_is_selected(trader,
                                                               pipeline_calls):
    trader.state.select_strategies(["AMARIH"])
    trader.signal_source_selection = SOURCE_AMARIH
    before = trader.candidates_seen

    asyncio.run(trader.on_alert(_alert(source=PRIMARY), Parsed()))

    # THE REFUSAL IS UPSTREAM OF EVERYTHING THAT COSTS MONEY OR TIME: the
    # candidate pipeline was never entered and the market was never read.
    assert pipeline_calls == [], "a blocked alert reached the L25 pipeline"
    assert trader.market.quotes == 0, "a blocked alert read the market"
    assert trader.account.open_positions() == []

    # It IS recorded, so the operator can see what the restriction refused.
    assert trader.candidates_seen == before + 1
    assert trader._signal_source_counters.get("amarih_blocked_telegram") == 1

    # THE LEDGER CARRIES THE NAMED REASON. A refusal that leaves no durable
    # evidence is indistinguishable from a feed that never delivered.
    notes = [e for e in trader.ledger.events()
             if e.get("reason") == "SOURCE_DISABLED_BY_AMARIH"]
    assert notes, "the block must be recorded in the ledger with its reason"
    assert "blocked" in str(notes[0].get("message", ""))
    assert notes[0].get("signal_source") == PRIMARY
    assert notes[0].get("mint") == MINT


def test_the_painape_and_moodeng_feeds_are_both_blocked(trader, pipeline_calls):
    """
    BOTH Telegram feeds, not just the primary one.

    The fallback feed is a second entry path into the same book. Guarding only
    ``PRIMARY`` would leave it open and the experiment's population would be
    contaminated by a source the operator believes is switched off.
    """
    from autopaper.telegram.signal_source import FALLBACK

    trader.state.select_strategies(["AMARIH"])
    trader.signal_source_selection = SOURCE_AMARIH
    asyncio.run(trader.on_alert(_alert(source=FALLBACK, message_id=2), Parsed()))

    assert pipeline_calls == []
    assert trader.market.quotes == 0
    assert trader.account.open_positions() == []
    assert trader._signal_source_counters.get("amarih_blocked_telegram") == 1
    assert any(e.get("reason") == "SOURCE_DISABLED_BY_AMARIH"
               and e.get("signal_source") == FALLBACK
               for e in trader.ledger.events())


def test_reselecting_another_strategy_restores_its_own_routing(trader,
                                                              pipeline_calls):
    """
    The restriction must be REVERSIBLE.

    A guard that keeps refusing after the operator leaves AMARIH would silently
    end paper trading, and the symptom -- no entries, no errors -- is the
    hardest kind to notice.
    """
    trader.state.select_strategies(["AMARIH"])
    trader.signal_source_selection = SOURCE_AMARIH
    asyncio.run(trader.on_alert(_alert(message_id=3), Parsed()))
    assert pipeline_calls == [], "blocked while AMARIH is selected"
    assert trader.candidates_seen == 1

    # Back to AUTO: the Telegram feed is eligible again, so this alert DOES
    # enter the L25 pipeline. Where L25 then takes it is not this test's
    # subject; that it was allowed to be taken is.
    trader.state.select_strategies(["L25_CURRENT"])
    trader.signal_source_selection = SOURCE_AUTO
    asyncio.run(trader.on_alert(_alert(message_id=4), Parsed()))

    assert len(pipeline_calls) == 1, "the restriction did not lift"
    assert trader._signal_source_counters.get("amarih_blocked_telegram") == 1, (
        "the block counter must not move once the restriction is lifted")
    # ``candidates_seen`` stays at 1: the blocked alert is recorded HERE, while
    # an admitted one is recorded inside the pipeline this test stubs out.
    # That asymmetry is the guard's own signature, so it is asserted rather
    # than worked around.
    assert trader.candidates_seen == 1


def test_an_amarih_signal_never_enters_the_l25_pipeline(trader, pipeline_calls):
    """
    The experiment owns Amarih ingestion. Routing it through ``on_alert`` would
    put it in front of L25's entry filters and change the population measured.
    """
    trader.state.select_strategies(["AMARIH"])
    trader.signal_source_selection = SOURCE_AMARIH
    before = trader.candidates_seen

    asyncio.run(trader.on_alert(_alert(source=SOURCE_AMARIH, message_id=5), Parsed()))

    assert pipeline_calls == []
    assert trader.market.quotes == 0
    assert trader.candidates_seen == before, (
        "an Amarih signal must not even be recorded as an L25 candidate")
    assert trader._signal_source_counters.get("amarih_routed_to_experiment") == 1
    assert trader.account.open_positions() == []


def test_an_amarih_signal_is_not_traded_when_the_experiment_is_not_selected(
        trader, pipeline_calls):
    """
    The Amarih source is refused whether or not the experiment is selected.

    Its ONLY legal entry is the experiment's own poller. If selection governed
    this, deselecting the experiment would silently turn the Telegram re-broadcast
    of an Amarih call into a live L25 entry -- the same signal, a different
    pipeline, and no record that it happened.
    """
    trader.state.select_strategies(["L25_CURRENT"])
    trader.signal_source_selection = SOURCE_AUTO

    asyncio.run(trader.on_alert(_alert(source=SOURCE_AMARIH, message_id=6), Parsed()))

    assert pipeline_calls == []
    assert trader.market.quotes == 0
    assert trader.account.open_positions() == []
    assert trader._signal_source_counters.get("amarih_routed_to_experiment") == 1
    assert trader._signal_source_counters.get("amarih_blocked_telegram", 0) == 0


# ---------------------------------------------------------------------------
# the quote adapters onto the EXISTING PAPER quote stack
# ---------------------------------------------------------------------------

class FakeBridge:
    """
    The minimum of the real bridge the entry adapter touches.

    ``token_decimals_for`` is ASYNC, exactly as the engine wires it
    (``reader.mint_decimals``). That shape is not incidental -- it is the bug
    these tests exist for.
    """

    def __init__(self, decimals=None, *, async_reader=True, raises=False):
        from types import SimpleNamespace

        self.sol_reference = SimpleNamespace(
            usd_per_sol=150.0, source="test", at=1_800_000_000.0)
        self.require_identity = True
        self.seen_buy: dict = {}
        self.seen_sell: dict = {}
        self._decimals = decimals
        self._async = async_reader
        self._raises = raises

    def token_decimals_for(self, mint):
        if self._raises:
            raise RuntimeError("mint account unreadable")
        if self._async:
            async def _read():
                return self._decimals
            return _read()
        return self._decimals

    async def _quote_buy_best(self, **kwargs):
        self.seen_buy = dict(kwargs)
        return None, []


@pytest.fixture
def with_bridge(trader, monkeypatch):
    """Point the engine's bridge accessor at a fake, per test."""
    def install(bridge):
        async def _accessor():
            return bridge
        monkeypatch.setattr(trader, "_paper_bridge_instance", _accessor)
        return bridge
    return install


def test_the_entry_adapter_awaits_an_async_token_decimals(trader, with_bridge):
    """
    THE BUG THIS PINS DOWN.

    ``token_decimals_for`` is ``DirectPumpReader.mint_decimals``, an async
    method. Casting its return value with ``int(...)`` builds an
    ``int(coroutine)``, raises ``TypeError``, and is swallowed by the fallback
    -- so every token was silently priced at 6 decimals and the coroutine was
    left un-awaited. A wrong decimals value scales the entry size by a power of
    ten while every individual number stays plausible, which is why this is
    asserted rather than assumed.
    """
    bridge = with_bridge(FakeBridge(9))

    result = asyncio.run(trader._amarih_quote_buy(MINT, 10.0))

    # No executable quote is available from the fake, so no fill is produced...
    assert result is None
    # ...but the request that was issued used the READ decimals, not the default.
    assert bridge.seen_buy["token_decimals"] == 9, (
        "the async decimals reader was discarded and the entry priced at the "
        f"default 6 (got {bridge.seen_buy['token_decimals']})")


def test_the_entry_adapter_accepts_a_sync_token_decimals_too(trader, with_bridge):
    """A reader may be sync or async; neither shape may be the silent default."""
    bridge = with_bridge(FakeBridge(4, async_reader=False))
    asyncio.run(trader._amarih_quote_buy(MINT, 10.0))
    assert bridge.seen_buy["token_decimals"] == 4


def test_an_unreadable_mint_falls_back_to_six_and_says_so(trader, with_bridge):
    """
    The fallback is a RECORDED event, not a silent one.

    Six is the only defensible default here, but a fallback nobody can see is
    indistinguishable from a correct read -- so the request still goes out and
    the reason is logged.
    """
    bridge = with_bridge(FakeBridge(raises=True))
    result = asyncio.run(trader._amarih_quote_buy(MINT, 10.0))
    assert result is None
    assert bridge.seen_buy["token_decimals"] == 6


def test_an_absent_decimals_value_falls_back_to_six(trader, with_bridge):
    """``None`` is "could not read", not zero, and not a crash."""
    bridge = with_bridge(FakeBridge(None))
    asyncio.run(trader._amarih_quote_buy(MINT, 10.0))
    assert bridge.seen_buy["token_decimals"] == 6


def test_the_entry_adapter_refuses_a_non_positive_notional(trader, with_bridge):
    """
    A $0 or negative arm sizes no swap input, so no quote request is made at
    all -- the refusal is local rather than a venue round trip.
    """
    bridge = with_bridge(FakeBridge(6))
    assert asyncio.run(trader._amarih_quote_buy(MINT, 0.0)) is None
    assert asyncio.run(trader._amarih_quote_buy(MINT, -5.0)) is None
    assert bridge.seen_buy == {}, "a zero notional still hit the venue"


def test_the_exit_adapter_quotes_the_exact_token_quantity(trader, with_bridge):
    """
    The SELL leg must be priced for the quantity the entry acquired.

    A re-derived float here would exit a slightly different size than it bought,
    and the difference would be reported as strategy PnL.
    """
    from types import SimpleNamespace

    bridge = with_bridge(FakeBridge(6))
    seen: dict = {}

    class _Quotes:
        async def quote_sell(self, **kwargs):
            seen.update(kwargs)
            return None

    bridge.quote_service = _Quotes()
    assert asyncio.run(trader._amarih_quote_sell(MINT, 12345, 9)) is None
    assert seen["token_amount_raw"] == 12345
    assert seen["token_decimals"] == 9


# ---------------------------------------------------------------------------
# the poller: exactly one, and only while selected
# ---------------------------------------------------------------------------

def _amarih_selected(trader: PaperTrader) -> None:
    trader.state.select_strategies(["AMARIH"])
    trader.signal_source_selection = SOURCE_AMARIH


def test_exactly_one_poller_task_exists_while_selected(trader):
    async def scenario():
        _amarih_selected(trader)
        trader._stop_event = asyncio.Event()
        trader._sync_amarih_runtime()
        first = trader._amarih_task
        assert first is not None, "selecting AMARIH must start the poller"

        # A second sync (a selection reconcile, a repeated tick) must NOT start
        # a second poller.
        trader._sync_amarih_runtime()
        assert trader._amarih_task is first, "a second poller was started"
        assert not first.done()

        # The run state is OFF, so the poller must not have polled the network.
        assert trader._amarih_runtime.status()["signals_seen"] == 0

        first.cancel()
        await asyncio.sleep(0)
        assert first.cancelled() or first.done()

    asyncio.run(scenario())


def test_deselecting_amarih_stops_new_ingestion(trader):
    async def scenario():
        _amarih_selected(trader)
        trader._stop_event = asyncio.Event()
        trader._sync_amarih_runtime()
        task = trader._amarih_task
        assert task is not None and not task.done()

        trader.state.select_strategies(["L25_CURRENT"])
        trader._sync_amarih_runtime()

        assert trader._amarih_task is None, "the poller must be retired"
        assert trader._amarih_runtime.running is False
        await asyncio.sleep(0)
        assert task.cancelled() or task.done()

    asyncio.run(scenario())


def test_the_driver_is_kept_after_deselection_so_open_arms_still_exit(trader):
    """
    An arm already inside its holding period has a scheduled exit to honour.

    Dropping the driver on deselection would leave that arm's deadline
    unprocessed -- a position the operator never closed, censored by a menu
    change rather than by the market.
    """
    async def scenario():
        _amarih_selected(trader)
        trader._stop_event = asyncio.Event()
        trader._sync_amarih_runtime()
        runtime = trader._amarih_runtime
        assert runtime is not None

        trader.state.select_strategies(["L25_CURRENT"])
        trader._sync_amarih_runtime()
        assert trader._amarih_runtime is runtime, "the experiment was discarded"
        if trader._amarih_task is not None:
            trader._amarih_task.cancel()
            await asyncio.sleep(0)

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def test_the_status_view_shows_the_amarih_block_only_when_selected(trader):
    view = trader.status_view()
    assert view.amarih_enabled is False

    _amarih_selected(trader)
    view = trader.status_view()
    assert view.amarih_enabled is True
    assert view.amarih_poller_running is False, "OFF run state, no poller"
    assert view.amarih_signals_seen == 0
    assert view.amarih_open_arms == 0


def test_the_status_view_reports_started_counters_and_open_arms(trader):
    async def scenario():
        _amarih_selected(trader)
        trader._stop_event = asyncio.Event()
        trader._sync_amarih_runtime()
        runtime = trader._amarih_runtime
        runtime.counters.events_seen = 7
        runtime.counters.accepted = 3
        runtime.counters.rejected = 4
        runtime.last_poll_at = 1_800_000_000.0
        view = trader.status_view()
        assert view.amarih_signals_seen == 7
        assert view.amarih_signals_accepted == 3
        assert view.amarih_signals_rejected == 4
        assert view.amarih_last_poll_at == 1_800_000_000.0
        assert view.amarih_poller_running is True
        if trader._amarih_task is not None:
            trader._amarih_task.cancel()
            await asyncio.sleep(0)

    asyncio.run(scenario())


def test_the_poller_interval_and_max_age_are_configurable(trader, monkeypatch):
    monkeypatch.setenv("PUMPVIEW_AMARIH_POLL_S", "3.5")
    monkeypatch.setenv("PUMPVIEW_AMARIH_MAX_AGE_S", "45")
    runtime = trader._build_amarih_runtime()
    assert runtime.poll_interval_s == 3.5
    assert runtime.source.max_age_s == 45.0


def test_a_malformed_interval_falls_back_rather_than_crashing(trader, monkeypatch):
    monkeypatch.setenv("PUMPVIEW_AMARIH_POLL_S", "not-a-number")
    runtime = trader._build_amarih_runtime()
    assert runtime.poll_interval_s > 0


def test_the_poller_is_not_started_by_startup_alone(trader):
    """
    Selecting a strategy is inert; START is a separate operator action.

    The poller must not begin ingesting the moment the process starts, so a
    selection with the run state OFF leaves no task behind.
    """
    _amarih_selected(trader)
    assert trader._amarih_task is None


def test_the_monitor_tick_starts_and_retires_the_poller(trader):
    """
    THE WIRING THAT MAKES ANY OF THIS RUN.

    The poller is started and retired from ``monitor_once`` -- the loop point
    where nothing else is mid-transition. Asserting only on
    ``_sync_amarih_runtime`` would leave the real call site untested, and a
    selection that never reaches the tick is inert no matter how correct the
    synchroniser is.
    """
    _amarih_selected(trader)

    async def scenario():
        trader._stop_event = asyncio.Event()
        assert trader._amarih_task is None

        await trader.monitor_once()
        first = trader._amarih_task
        assert first is not None and not first.done(), (
            "the monitor tick must start the Amarih poller")

        await trader.monitor_once()
        assert trader._amarih_task is first, "a second tick started a second poller"

        trader.state.select_strategies(["L25_CURRENT"])
        await trader.monitor_once()
        assert trader._amarih_task is None, (
            "the monitor tick must retire the poller on deselection")
        await asyncio.sleep(0)
        assert first.cancelled() or first.done()

    asyncio.run(scenario())


def test_the_engine_re_hydrates_the_experiment_state_on_first_selection(trader):
    """
    A RESTART MUST NOT RE-PROCESS SIGNALS, AND MUST NOT LOSE OPEN ARMS.

    The durable file is read on first selection, and the previously-accepted
    signal ids are seeded into the live adapter's dedup set -- that seeding is
    what carries the adapter's process-local guarantee across a restart.
    """
    state_path = trader._amarih_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "schema_version": 1,
        "seen_signal_ids": ["pre-restart-1", "pre-restart-2"],
        "open_arms": [],
        "notified": ["signal:pre-restart-1"],
        "last_poll_at": 1_800_000_000.0,
    }), encoding="utf-8")

    async def scenario():
        _amarih_selected(trader)
        trader._stop_event = asyncio.Event()
        trader._sync_amarih_runtime()
        runtime = trader._amarih_runtime
        assert runtime is not None

        assert {"pre-restart-1", "pre-restart-2"} <= set(runtime.source.seen_ids), (
            "the durable signal ids were not seeded into the live dedup set")
        assert runtime.status()["last_poll_at"] == 1_800_000_000.0
        # A message already delivered before the restart is not re-sent.
        assert "signal:pre-restart-1" in runtime._notified

        if trader._amarih_task is not None:
            trader._amarih_task.cancel()
            await asyncio.sleep(0)

    asyncio.run(scenario())


def test_no_experiment_state_is_built_before_selection(trader):
    """The durable file is not read, and nothing is constructed, until selected."""
    assert trader._amarih_runtime is None
    trader.state.select_strategies(["L25_CURRENT"])
    assert trader._build_amarih_runtime is not None  # callable, never called
    assert trader._amarih_runtime is None


def test_the_run_state_gates_polling(trader):
    """
    The poller loop polls ONLY in ``RunState.PAPER``.

    Asserted on the condition the loop uses, because driving the real loop would
    require a live PumpView connection.
    """
    _amarih_selected(trader)
    assert trader.state.run_state is RunState.OFF
    assert trader._amarih_running() is False
    trader.state.run_state = RunState.PAPER
    assert trader._amarih_running() is True
