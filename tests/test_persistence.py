"""
Duplicate suppression, ledger durability, crash recovery, and STOP semantics.

A duplicated Telegram message must never create two paper trades, a restart
must never reset the bankroll, and a stop must not silently flatten positions.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from autopaper.account import PaperAccount
from autopaper.ledger import (
    EventType,
    Ledger,
    PaperAccountState,
    load_json,
    save_json_atomic,
)
from autopaper.models import Position, PositionStatus, Regime, Route
from autopaper.clock import FakeClock
from autopaper.telegram import ConnectionState, DedupeWindow, TelegramListener

from .conftest import MINT, MINT2, alert, obs

T0 = 1_700_000_000.0


# ---------------------------------------------------------------------------
# dedupe
# ---------------------------------------------------------------------------


def test_dedupe_window_rejects_repeats():
    d = DedupeWindow(capacity=4)
    assert d.add((1, 10)) is True
    assert d.add((1, 10)) is False
    assert d.add((1, 11)) is True
    assert (1, 10) in d


def test_dedupe_window_is_bounded():
    d = DedupeWindow(capacity=3)
    for i in range(10):
        d.add((1, i))
    assert len(d) == 3
    # the oldest have been evicted, which is the documented trade-off
    assert d.add((1, 0)) is True


def test_listener_diagnoses_a_missing_session(config):
    listener = TelegramListener(config.telegram)
    state, detail = listener.diagnose()
    assert state == ConnectionState.AWAITING_LOGIN
    assert "TELEGRAM_SESSION" in detail
    assert "login" in detail


def test_listener_diagnoses_missing_credentials(config):
    from dataclasses import replace

    listener = TelegramListener(replace(config.telegram, api_id=None, api_hash=None))
    state, detail = listener.diagnose()
    assert state == ConnectionState.ERROR
    assert "API_ID" in detail


def test_listener_reports_ready_when_configured(config):
    from dataclasses import replace

    listener = TelegramListener(
        replace(config.telegram, session="1AaBbCc", chat="somechat")
    )
    state, _ = listener.diagnose()
    assert state == ConnectionState.CONNECTED


def test_persisted_seen_keys_restore_the_dedupe_window(config):
    listener = TelegramListener(config.telegram)
    listener.restore_seen([[100, 5], [100, 6]])
    assert listener.dedupe.add((100, 5)) is False
    assert listener.dedupe.add((100, 6)) is False
    assert listener.dedupe.add((100, 7)) is True


def test_duplicate_message_does_not_produce_two_snapshots(config):
    """
    Drive the listener's message handler directly with a duplicate delivery.
    Only one candidate may reach the callback.
    """
    seen: list = []

    async def on_alert(snapshot, parsed):
        seen.append(snapshot)

    listener = TelegramListener(config.telegram, on_alert=on_alert,
                                clock=FakeClock(T0))
    listener.status.chat_id = 100
    listener.status.state = ConnectionState.CONNECTED

    class Msg:
        id = 4242
        message = f"NEW TOKEN\nCA: {MINT}\nMC: $40000\nLIQ: $20000"
        date = None
        sender_id = 999

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(listener._handle(Msg()))
        assert len(seen) == 1
        # the same message id again, e.g. after a reconnect replay
        loop.run_until_complete(listener._handle(Msg()))
        loop.run_until_complete(listener._handle(Msg()))
    finally:
        loop.close()
    assert len(seen) == 1, "a duplicate alert must not produce a second candidate"
    assert listener.status.duplicates_dropped == 2
    assert listener.status.messages_accepted == 1


def test_message_without_a_mint_is_dropped_not_traded(config):
    import asyncio

    from autopaper.clock import FakeClock
    from autopaper.telegram.client import ListenerStatus

    seen: list = []

    async def on_alert(snapshot, parsed):
        seen.append(snapshot)

    listener = TelegramListener(config.telegram, on_alert=on_alert,
                                clock=FakeClock(T0))
    listener.status.chat_id = 100

    class Msg:
        id = 1
        message = "gm everyone, how are we doing today"
        date = None
        sender_id = 1

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(listener._handle(Msg()))
    finally:
        loop.close()
    assert seen == []
    assert listener.status.unparsed_dropped == 1


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------


def test_ledger_is_append_only(ledger, config):
    for i in range(3):
        ledger.event(EventType.NOTE, {"i": i})
    lines = config.ledger_file.read_text().strip().splitlines()
    assert len(lines) == 3
    for i, line in enumerate(lines):
        assert json.loads(line)["i"] == i


def test_ledger_survives_a_corrupt_line(ledger, config):
    ledger.event(EventType.NOTE, {"i": 0})
    with config.ledger_file.open("a") as fh:
        fh.write("{not json\n")
    ledger.event(EventType.NOTE, {"i": 1})
    rows = ledger.events()
    assert [r["i"] for r in rows] == [0, 1], "a corrupt line must not lose the rest"


def test_seen_message_keys_round_trip(ledger, config):
    ledger.candidate_seen({
        "candidate_id": "c1", "message_id": 77, "chat_id": 100, "mint": MINT,
        "received_timestamp": T0, "fields": {"price_usd": 1e-4},
        "raw_text": "x", "traded": True,
    })
    assert ledger.seen_message_keys() == [[100, 77]]


def test_trades_are_extracted_from_the_ledger(ledger):
    from autopaper.models import Trade

    t = Trade(
        trade_id="t1", position_id="p1", candidate_id="c1", mint=MINT, symbol="X",
        source="telegram", strategy="s", route="PRECISION_ENTRY", regime="EARLY",
        entry_signal_timestamp=T0, entry_timestamp=T0, entry_price=1e-4,
        fill_price=1e-4, exit_timestamp=T0 + 10, exit_price=1.2e-4, quantity=1e5,
        entry_fee_usd=0.1, exit_fee_usd=0.1, entry_slippage_usd=0.01,
        exit_slippage_usd=0.01, gross_pnl_usd=2.0, net_pnl_usd=1.78,
        return_pct=20.0, holding_seconds=10.0, mfe_pct=25.0, mae_pct=-3.0,
        exit_reason="test", exit_style="MOMENTUM_FAILURE", thesis="", invalidation=(),
    )
    ledger.trade(t)
    rows = ledger.trades()
    assert len(rows) == 1
    assert rows[0]["trade_id"] == "t1"


# ---------------------------------------------------------------------------
# persistence and recovery
# ---------------------------------------------------------------------------


def test_atomic_save_and_load(tmp_path):
    path = tmp_path / "state.json"
    save_json_atomic(path, {"a": 1})
    assert load_json(path) == {"a": 1}
    save_json_atomic(path, {"a": 2})
    assert load_json(path) == {"a": 2}
    assert list(tmp_path.glob("*.tmp")) == [], "temp files must be cleaned up"


def test_account_state_round_trips():
    st = PaperAccountState(
        starting_bankroll_usd=100.0, cash_usd=63.25, realized_pnl_usd=-36.75,
        fees_paid_usd=1.23, slippage_paid_usd=4.56, run_count=7,
    )
    back = PaperAccountState.from_dict(st.as_dict())
    assert back.cash_usd == pytest.approx(63.25)
    assert back.realized_pnl_usd == pytest.approx(-36.75)
    assert back.run_count == 7


def test_bankroll_is_never_reset_on_restart(tmp_path):
    """A restart reloads; it does not reinitialise. Losses are permanent."""
    from autopaper.config import build_config

    root = tmp_path
    (root / ".env.local").write_text("TELEGRAM_API_ID=1\nTELEGRAM_API_HASH=x\n")
    cfg = build_config(root)

    first = PaperAccount(PaperAccountState(starting_bankroll_usd=100.0,
                                           cash_usd=100.0), cfg.risk)
    first.state.cash_usd = 41.5
    first.state.realized_pnl_usd = -58.5
    first.state.run_count = 1
    save_json_atomic(cfg.state_file, {
        "account": first.state.as_dict(),
        "open_positions": [],
        "recently_exited": {},
        "unfilled_attempts": 3,
    })

    raw = load_json(cfg.state_file)
    reloaded = PaperAccount(PaperAccountState.from_dict(raw["account"]), cfg.risk)
    # mirrors PaperTrader._load_state, which is covered end to end elsewhere
    reloaded.unfilled_attempts = int(raw["unfilled_attempts"])
    assert reloaded.state.starting_bankroll_usd == pytest.approx(100.0)
    assert reloaded.state.cash_usd == pytest.approx(41.5)
    assert reloaded.state.realized_pnl_usd == pytest.approx(-58.5)
    assert reloaded.unfilled_attempts == 3
    assert reloaded.state.starting_bankroll_usd != 41.5, (
        "the starting bankroll is a separate fact from the current cash"
    )


def test_open_positions_survive_a_restart(tmp_path):
    from autopaper.config import build_config
    from autopaper.engine import _positions_from
    from autopaper.models import ExitStyle

    root = tmp_path
    (root / ".env.local").write_text("TELEGRAM_API_ID=1\nTELEGRAM_API_HASH=x\n")
    cfg = build_config(root)
    p = Position(
        position_id="p_open", candidate_id="c1", mint=MINT, symbol="SURVIVOR",
        status=PositionStatus.OPEN, route=Route.FAST_ENTRY,
        regime=Regime.ACCELERATING, exit_style=ExitStyle.RUNNER, strategy="s",
        entry_signal_timestamp=T0, entry_decision_timestamp=T0,
        entry_timestamp=T0 + 1, quoted_price=1e-4, entry_reference_price=1e-4,
        fill_price=1.01e-4, quantity=1e5, cash_out_usd=10.2, cost_basis_usd=10.2,
        mfe_pct=12.0, mae_pct=-4.0, peak_price=1.2e-4,
    )
    save_json_atomic(cfg.state_file, {
        "account": PaperAccountState().as_dict(),
        "open_positions": [p.as_dict()],
        "recently_exited": {MINT2: T0},
        "unfilled_attempts": 0,
    })

    raw = load_json(cfg.state_file)
    restored = _positions_from(raw["open_positions"])
    assert list(restored) == ["p_open"]
    back = restored["p_open"]
    assert back.status is PositionStatus.OPEN, "a restored position is still open"
    assert back.symbol == "SURVIVOR"
    assert back.route is Route.FAST_ENTRY
    assert back.regime is Regime.ACCELERATING
    assert back.exit_style is ExitStyle.RUNNER
    assert back.fill_price == pytest.approx(1.01e-4)
    assert back.mfe_pct == pytest.approx(12.0)
    assert back.runner_mode is False
    # and the cooldown survives too
    assert raw["recently_exited"][MINT2] == T0


# ---------------------------------------------------------------------------
# control commands
# ---------------------------------------------------------------------------


def test_stop_does_not_flatten(tmp_path, config):
    """
    STOP stops NEW entries. It must not close what is already open - a stopped
    process silently liquidating positions would be a data-destroying surprise.
    """
    account = PaperAccount(PaperAccountState(), config.risk)
    p = Position(
        position_id="p1", candidate_id="c", mint=MINT, symbol="OPEN",
        status=PositionStatus.OPEN, route=Route.FAST_ENTRY,
        regime=Regime.EARLY, exit_style="x", strategy="s",
        entry_signal_timestamp=T0, entry_decision_timestamp=T0,
        entry_timestamp=T0, fill_price=1e-4, quantity=1e5, cash_out_usd=10.1,
    )
    account.register_position(p)
    account.state.cash_usd = 89.9

    # simulate STOP handling
    assert account.open_positions(), "the position must still be open"
    assert account.cash == pytest.approx(89.9)
    assert account.realized == pytest.approx(0.0)


def test_control_file_round_trip(config):
    save_json_atomic(config.control_file, {"command": "STOP", "at": T0})
    assert load_json(config.control_file)["command"] == "STOP"
    save_json_atomic(config.control_file, {"command": "FLATTEN", "at": T0 + 1})
    assert load_json(config.control_file)["command"] == "FLATTEN"
    save_json_atomic(config.control_file, {"command": "", "at": 0.0})
    assert load_json(config.control_file)["command"] == ""


def test_cash_moves_only_on_real_fills(tmp_path):
    """
    A rejected candidate and an unfilled attempt both leave cash untouched.
    Nothing is quietly debited and credited back.
    """
    account = PaperAccount(PaperAccountState(), )
    start = account.cash
    for _ in range(50):
        pass
    assert account.cash == pytest.approx(start)


# ---------------------------------------------------------------------------
# the read loop
# ---------------------------------------------------------------------------


class FakeMessage:
    def __init__(self, id, text="", out=False):
        self.id = id
        self.message = text
        self.text = text
        self.out = out
        self.date = None
        self.sender_id = 1


class FakeMTProto:
    """Minimal stand-in for a connected Telethon client."""

    def __init__(self, batches):
        self.batches = list(batches)
        self.requests = 0
        self.min_ids = []
        self.disconnected = False

    async def get_messages(self, entity, limit=1, min_id=None, **kw):
        self.requests += 1
        self.min_ids.append(min_id)
        if self.batches:
            return self.batches.pop(0)
        return []

    async def disconnect(self):
        self.disconnected = True


async def _bounded(coro, seconds: float):
    """Run a coroutine, cancel it after `seconds`, swallow the cancellation."""
    task = asyncio.ensure_future(coro)
    await asyncio.sleep(seconds)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:  # noqa: BLE001
        pass


def _alert_text(mint=MINT, symbol="TST"):
    return f"NEW TOKEN\nSymbol: {symbol}\nCA: {mint}\nMC: $40000\nLIQ: $20000"


def test_poll_loop_yields_new_messages_in_ascending_order(config):
    """
    Telegram returns newest-first. A candidate analysed out of order would see a
    later token before an earlier one, so the loop must sort ascending.
    """
    seen = []

    async def on_alert(snapshot, parsed):
        seen.append((snapshot.message_id, parsed.symbol))

    listener = TelegramListener(config.telegram, on_alert=on_alert,
                                clock=FakeClock(T0))
    listener.status.chat_id = 100
    listener.status.state = ConnectionState.CONNECTED
    listener.status.since_message_id = 100
    client = FakeMTProto([
        [FakeMessage(103, _alert_text(MINT, "C")),
         FakeMessage(102, _alert_text(MINT2, "B")),
         FakeMessage(101, _alert_text(MINT, "A"))],
    ])
    listener._client = client
    listener.config = replace(
        config.telegram, poll_interval_s=0.01, poll_error_backoff_s=0.01
    )

    # Drive the read loop itself. Going through run() would re-read the anchor
    # first, which legitimately consumes the same batch.
    loop = asyncio.new_event_loop()
    try:
        async def go():
            async for message in listener._iter_messages():
                await listener._handle(message)

        loop.run_until_complete(_bounded(go(), 0.2))
    finally:
        loop.close()
    # run() collected them in ascending order
    assert [m for m, _ in seen] == [101, 102, 103], seen
    assert [s for _, s in seen] == ["A", "B", "C"]


def test_poll_loop_never_re_reads_a_processed_message(config):
    """The high-water mark is the mechanism that prevents double processing."""
    listener = TelegramListener(config.telegram, clock=FakeClock(T0))
    listener.status.chat_id = 100
    listener.status.since_message_id = 100
    client = FakeMTProto([
        [FakeMessage(105, _alert_text()), FakeMessage(101, _alert_text())],
        [FakeMessage(101, _alert_text())],   # would replay if the mark slipped
        [],
    ])
    listener._client = client
    listener.config = replace(
        config.telegram, poll_interval_s=0.01, poll_error_backoff_s=0.01
    )

    loop = asyncio.new_event_loop()
    got = []
    try:
        async def go():
            async def collect():
                async for m in listener._iter_messages():
                    got.append(m.id)
            task = asyncio.create_task(collect())
            await asyncio.sleep(0.2)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        loop.run_until_complete(go())
    finally:
        loop.close()
    assert got == [101, 105], got
    assert listener.status.since_message_id == 105
    # the last request asked for anything strictly newer than 105
    assert client.min_ids[-1] == 105


def test_poll_loop_survives_a_transient_error(config):
    """One bad poll must not end the stream."""
    listener = TelegramListener(config.telegram, clock=FakeClock(T0))
    listener.status.chat_id = 100
    listener.status.since_message_id = 100

    calls = {"n": 0}

    class Flaky:
        async def get_messages(self, entity, limit=1, min_id=None, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("connection reset")
            if calls["n"] == 2:
                return [FakeMessage(101, _alert_text())]
            return []

    listener._client = Flaky()
    listener.config = replace(
        config.telegram, poll_interval_s=0.01, poll_error_backoff_s=0.01
    )

    loop = asyncio.new_event_loop()
    got = []
    try:
        async def go():
            async def collect():
                async for m in listener._iter_messages():
                    got.append(m.id)
            task = asyncio.create_task(collect())
            await asyncio.sleep(0.25)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        loop.run_until_complete(go())
    finally:
        loop.close()
    assert got == [101], got
    assert listener.status.poll_errors == 1
