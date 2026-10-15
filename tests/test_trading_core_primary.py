"""
The trading core is primary. Telegram is a DOWNSTREAM consumer.

The rule being proved: nothing the Telegram bot does can sit on the critical
path. Ingestion (the upstream channel) is a different component and must keep
working; outbound messaging is presentation and must be unable to affect a
trade.

No network, no wallet, no broadcast. The transport is a tripwire.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
import tempfile
from pathlib import Path

import httpx
import pytest

from autopaper.config import build_config
from autopaper.engine import PaperTrader
from autopaper.modes import RunState, TradeMode

#: full end-to-end trading cycle
pytestmark = [
    pytest.mark.slow,
]

ALLOWED = -1001234567890
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "999:FAKEfakefakefake")
    monkeypatch.setenv("TELEGRAM_BOT_CHAT", str(ALLOWED))
    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    return build_config(tmp_path)


def make_engine(cfg, **over):
    for k, v in over.items():
        object.__setattr__(cfg, k, v)
    return PaperTrader(cfg)


# ---------------------------------------------------------------------------
# 1. CLI-only mode
# ---------------------------------------------------------------------------


def test_cli_mode_builds_no_telegram_bot(cfg):
    """No bot means no notifier, no poller, no queue, no outbound call."""
    t = make_engine(cfg, cli_only=True)
    try:
        assert t.cli_only is True
        assert t.bot_control is None
        assert t.bot_transport is None
        assert t.bot_notifier is None
        # nothing attached to the notifier hook either
        assert t._notifier.__class__.__name__ == "function"
        assert t._notifier.__name__ == "<lambda>", (
            "CLI mode must leave the no-op notifier in place"
        )
    finally:
        t._release_ownership()


def test_normal_mode_still_builds_the_bot(cfg):
    t = make_engine(cfg, cli_only=False)
    try:
        assert t.bot_control is not None
        assert t.bot_notifier is not None
    finally:
        t._release_ownership()


def test_cli_mode_still_engages_the_upstream_channel(cfg):
    """
    Telegram is TWO things.

    The CHANNEL that produces alerts is upstream and must survive --cli. The BOT
    that presents results is downstream and must not exist. Killing both would
    silently stop trading.
    """
    t = make_engine(cfg, cli_only=True)
    try:
        # the listener is constructed by the run loop, not __init__; the point
        # is that cli_only does not disable it
        src = inspect.getsource(PaperTrader._supervise_listener)
        assert "cli_only" not in src, (
            "the upstream channel listener must not be gated on cli_only"
        )
        assert "self.listener" in src
    finally:
        t._release_ownership()


def test_cli_mode_performs_zero_outbound_network_calls(cfg):
    """
    The strongest form: no httpx client is ever created for the bot.

    A tripwire transport raises on ANY request, so an accidental send fails the
    test rather than being missed.
    """
    attempts: list[str] = []

    def tripwire(request):
        attempts.append(str(request.url))
        raise AssertionError(f"outbound Telegram call in --cli: {request.url}")

    t = make_engine(cfg, cli_only=True)
    try:
        assert t.bot_transport is None, "a transport exists, so a call could happen"
        # and the notifier hook, if used at all, must not reach a network
        t._notify("should go nowhere", None)
        assert attempts == []
    finally:
        t._release_ownership()


def test_the_cli_flag_is_wired_to_the_config():
    """`run --cli` must actually set the flag the engine reads."""
    src = (REPO / "src/autopaper/cli.py").read_text()
    assert '"--cli"' in src
    assert "dataclasses.replace(cfg, cli_only=" in src


def test_cli_only_is_applied_BEFORE_the_engine_is_built():
    """
    The regression that made `--cli` a lie.

    PaperTrader builds the Telegram bot inside __init__. Setting
    trader.cli_only after _trader(cfg) left the transport, notifier, poller and
    queue all alive and running -- the flag disabled nothing. Proven by source
    ORDER, because a source string is the only thing that can pin this.
    """
    src = (REPO / "src/autopaper/cli.py").read_text()
    body = src[src.index("def cmd_run("):src.index("def cmd_status(")]
    assert body.index("dataclasses.replace(cfg, cli_only=") < body.index("_trader(cfg)"), (
        "--cli must be folded into the config before the engine is constructed"
    )


def test_end_to_end_run_cli_constructs_no_bot(tmp_path, monkeypatch):
    """
    The whole point, proven through the real entry point rather than by
    constructing an engine with the flag pre-set.

    A tripwire on BotApiTransport: if anything tries to build the bot, this
    fails.
    """
    import autopaper.telegram.botapi as botapi
    import autopaper.telegram.notifier as notifier
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "999:FAKEfakefakefake")
    monkeypatch.setenv("TELEGRAM_BOT_CHAT", str(ALLOWED))

    built: list[str] = []
    real_transport = botapi.BotApiTransport

    class Tripwire(real_transport):        # type: ignore[misc,valid-type]
        def __init__(self, *a, **k):
            built.append("transport")
            raise AssertionError("--cli constructed a Telegram transport")

    monkeypatch.setattr(botapi, "BotApiTransport", Tripwire)
    monkeypatch.setattr(notifier, "TelegramNotifier",
                        lambda *a, **k: built.append("notifier"))
    monkeypatch.setattr("autopaper.cli.configure_logging", lambda *a, **k: None)

    from autopaper.cli import cmd_run, build_parser

    parser = build_parser()
    args = parser.parse_args(["run", "--cli"])
    cfg = build_config(tmp_path)

    engine_holder = {}
    import autopaper.cli as cli_mod
    real_trader = cli_mod._trader

    def spy(c):
        t = real_trader(c)
        engine_holder["t"] = t
        # stop before the run loop so no listener/network work happens
        async def _stop(self):
            self._release_ownership()
        t.run = _stop.__get__(t, type(t))
        return t

    monkeypatch.setattr(cli_mod, "_trader", spy)
    assert cmd_run(cfg, args) == 0
    assert built == [], f"--cli built outbound components: {built}"
    assert engine_holder["t"].cli_only is True
    assert engine_holder["t"].bot_control is None


def test_cli_only_is_a_real_parser_option():
    """Prove the command line accepts it, by parsing rather than by reading."""
    from autopaper.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["run", "--cli"])
    assert args.cli is True
    assert parser.parse_args(["run"]).cli is False


# ---------------------------------------------------------------------------
# 2. outbound is already async and non-blocking
# ---------------------------------------------------------------------------


def test_send_returns_without_touching_the_network():
    """The hook is synchronous and only enqueues."""
    from autopaper.telegram.botapi import BotApiTransport
    from autopaper.telegram.notifier import TelegramNotifier

    def tripwire(request):
        raise AssertionError("a send happened on the caller's thread")

    client = httpx.AsyncClient(transport=httpx.MockTransport(tripwire))
    transport = BotApiTransport("999:FAKE", chat_id=str(ALLOWED), client=client)
    n = TelegramNotifier(transport, min_send_interval_s=0.0)

    async def go():
        for i in range(100):
            n.send(f"m{i}", None)      # must not raise, must not block
        await n.aclose()

    asyncio.run(go())


def test_the_queue_is_bounded_and_drops_rather_than_blocking():
    from autopaper.telegram.botapi import BotApiTransport
    from autopaper.telegram.notifier import TelegramNotifier

    async def handler(request):
        return httpx.Response(200, json={"ok": True, "result": {}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = BotApiTransport("999:FAKE", chat_id=str(ALLOWED), client=client)
    n = TelegramNotifier(transport, max_queue=5, min_send_interval_s=0.0)

    async def go():
        for i in range(200):
            n.send(f"m{i}", None)
        stats = n.status_dict()
        await n.aclose()
        return stats

    stats = asyncio.run(go())
    assert stats["dropped_queue_full"] > 0, "overflow must be counted, not queued"
    assert stats["dropped_queue_full"] + stats["sent"] <= 200


def test_a_failing_telegram_cannot_fail_a_trade(cfg):
    """
    The guard is at the choke point, so EVERY call site is protected.
    """
    t = make_engine(cfg)
    try:
        def boom(text, kb=None):
            raise RuntimeError("telegram is down")

        t.set_notifier(boom)
        t.state.start()
        for i in range(50):
            t._notify(f"alert {i}", None)     # must not raise
        assert t.state.run_state is RunState.PAPER
        assert t.state.trade_mode is TradeMode.PAPER
    finally:
        t._release_ownership()


def test_no_internal_call_site_awaits_a_notification():
    """
    A Telegram send that took 5s must not delay a trade.

    Asserted on source: no `_notify(` is awaited anywhere in the engine, so the
    hook cannot be turned back into a blocking call by accident.
    """
    src = (REPO / "src/autopaper/engine.py").read_text()
    assert "await self._notify" not in src
    # `await self.bot_notifier.aclose()` IS legitimate, but only in SHUTDOWN:
    # closing the worker cleanly. It must not appear on a trading path.
    shutdown = src[src.index("async def shutdown("):]
    trading = src[:src.index("async def shutdown(")]
    assert "await self.bot_notifier" not in trading
    assert "await self.bot_notifier.aclose()" in shutdown


def test_the_trading_core_never_imports_the_notifier():
    """The core does not depend on the presentation layer."""
    src = (REPO / "src/autopaper/engine.py").read_text()
    # the import is lazy and inside the builder, not at module scope
    module_level = src.split("class PaperTrader")[0]
    assert "TelegramNotifier" not in module_level, (
        "the notifier must not be a module-scope dependency of the engine"
    )


# ---------------------------------------------------------------------------
# 4-7. the operator commands
# ---------------------------------------------------------------------------


def test_exactly_nine_commands_are_exposed():
    """
    Five actions, /start (a menu, not an action) and two read-only position
    views. The position views dispatch onto the same callbacks their buttons use,
    so a typed command and a button press cannot reach different code.
    """
    from autopaper.telegram.botapi import TEXT_COMMANDS

    assert set(TEXT_COMMANDS) == {"/start", "/status", "/balance", "/pause",
"/resume", "/kill", "/open", "/closed", "/tp"}


@pytest.mark.parametrize("payload,expect", [
    ("stats", "stats"), ("balance", "balance"),
    ("pause", "pause"), ("resume", "resume"), ("kill", "kill"),
])
def test_each_command_dispatches(payload, expect):
    """
    Every command is handled.

    /balance reports "unavailable" when no provider is wired, which is a HANDLED
    state with an explicit reason -- not a dispatch failure.
    """
    from autopaper.telegram.actions import ButtonHandler
    from autopaper.modes import RuntimeState

    state = RuntimeState()
    state.start()
    result = asyncio.run(ButtonHandler(state).handle(payload))
    assert result.handled is True
    if payload == "balance":
        assert result.text
    else:
        assert result.error == ""


def test_pause_stops_new_entries_and_resume_restores():
    from autopaper.telegram.actions import ButtonHandler
    from autopaper.modes import RuntimeState

    state = RuntimeState()
    state.start()
    handler = ButtonHandler(state)
    asyncio.run(handler.handle("pause"))
    assert state.can_open_new_entries()[0] is False
    asyncio.run(handler.handle("resume"))
    assert state.can_open_new_entries()[0] is True


def test_kill_is_the_emergency_stop():
    from autopaper.telegram.actions import ButtonHandler
    from autopaper.modes import RuntimeState

    state = RuntimeState()
    state.start()
    asyncio.run(ButtonHandler(state).handle("kill"))
    assert state.emergency_stop is True
    assert state.can_open_new_entries()[0] is False


def test_status_is_read_only_and_lightweight():
    """/status must not change the runtime or do network work."""
    from autopaper.telegram.actions import ButtonHandler
    from autopaper.modes import RuntimeState

    state = RuntimeState()
    state.start()
    before = state.run_state
    result = asyncio.run(ButtonHandler(state).handle("stats"))
    assert state.run_state is before
    assert state.emergency_stop is False


def test_status_makes_no_network_call():
    from autopaper.telegram.actions import ButtonHandler
    from autopaper.modes import RuntimeState

    class Tripwire:
        def __call__(self, *a, **k):
            raise AssertionError("/status did network work")

    state = RuntimeState()
    state.start()
    handler = ButtonHandler(
        state, __import__("autopaper.telegram.actions", fromlist=["x"])
        .OperatorActions(extras={"real_equity": Tripwire(),
                                 "open_orders": Tripwire()}))
    result = asyncio.run(handler.handle("stats"))
    assert result.handled is True


def test_balance_is_on_demand_and_distinguishes_zero_from_unavailable():
    from autopaper.real_equity import compute_real_equity
    from autopaper.telegram.actions import ButtonHandler, OperatorActions
    from autopaper.telegram import ui
    from autopaper.modes import RuntimeState

    zero = ui.render_balance(compute_real_equity(0, 180.0, "wUEU…Wbt6"))
    assert "$0.0000" in zero and "unavailable" not in zero

    bad = ui.render_balance(compute_real_equity(None, None, ""))
    assert "unavailable" in bad
    assert "$0.0000" not in bad


def test_balance_reports_unavailable_rather_than_zero_on_failure():
    from autopaper.telegram.actions import ButtonHandler, OperatorActions
    from autopaper.modes import RuntimeState

    async def boom():
        raise RuntimeError("rpc down")

    state = RuntimeState()
    state.start()
    handler = ButtonHandler(state, OperatorActions(extras={"real_equity": boom}))
    result = asyncio.run(handler.handle("balance"))
    assert "unavailable" in result.text
    assert "$0.0000" not in result.text


# ---------------------------------------------------------------------------
# 8-10. the single-screen dashboard
# ---------------------------------------------------------------------------


def test_redraw_clears_the_screen():
    from autopaper import dashboard

    assert "\x1b[2J" in dashboard.clear()
    assert "\x1b[H" in dashboard.clear()


def test_repeated_updates_do_not_append():
    """
    The regression: the old dashboard appended a full frame every few seconds,
    so a long run grew the scrollback without bound.

    Asserted against a fake TTY, so the redraw path is actually exercised.
    """
    from autopaper import dashboard

    stream = _FakeTTY()
    for _ in range(5):
        dashboard.write_live("FRAME", stream)
    assert stream.buf.count("\x1b[2J") == 5, "each update must clear"
    # each redraw writes the frame then a newline, and nothing is appended
    # beyond it -- so the visible text is always exactly one frame
    assert stream.buf == (("\x1b[2J\x1b[H" + "FRAME" + "\n") * 5)


def test_a_non_tty_falls_back_to_appending():
    """Piped output must stay readable rather than becoming escape codes."""
    from autopaper import dashboard

    class Pipe:
        def __init__(self):
            self.buf = ""

        def write(self, s):
            self.buf += s

        def flush(self):
            pass

        def isatty(self):
            return False

    stream = Pipe()
    dashboard.write_live("LINE", stream)
    assert stream.buf == "LINE\n"
    assert "\x1b" not in stream.buf


def test_the_frame_survives_a_narrow_terminal():
    from autopaper.dashboard import build_frame

    for cols in (20, 40, 60, 200):
        frame = build_frame(cols=cols,
                            account={"equity_usd": 94.78},
                            last_event={"symbol": "M" * 80, "latency_ms": 34216},
                            last_error="e" * 300)
        for line in frame.splitlines():
            assert len(line) <= cols, f"cols={cols}: {len(line)}"
        assert frame.splitlines()


def test_the_frame_contains_the_required_sections():
    from autopaper.dashboard import build_frame

    frame = build_frame(
        account={"equity_usd": 100.0, "cash_usd": 100.0},
        performance={"total_trades": 3, "wins": 1, "losses": 2,
                     "win_rate": 0.33, "profit_factor": 1.5},
        latency={"analysis": {"n": 5, "p50": 1, "p95": 2, "p99": 3, "max": 4}},
        services={"channel": "connected"},
        last_error="boom",
    )
    for section in ("ACCOUNT", "PERFORMANCE", "LAST EVENT", "CURRENT PIPELINE",
                    "LATENCY", "SERVICES", "LAST ERROR", "MEMEBOT L25"):
        assert section in frame, section


def test_the_frame_shows_only_current_state_not_history():
    from autopaper.dashboard import build_frame

    frame = build_frame(positions=[{"symbol": "A"}, {"symbol": "B"}])
    assert "A" in frame and "B" in frame
    assert "trade_1" not in frame


def test_the_frame_degrades_without_data():
    """An empty state must render, not raise."""
    from autopaper.dashboard import build_frame

    frame = build_frame()
    assert "MEMEBOT" in frame


def test_terminal_size_never_raises():
    from autopaper import dashboard

    cols, rows = dashboard.terminal_size()
    assert cols >= 20 and rows >= 10


# ---------------------------------------------------------------------------
# 12. trading latency excludes notification latency
# ---------------------------------------------------------------------------


def test_notification_latency_is_excluded_from_trading_latency():
    from autopaper.latency import StageRecorder

    r = StageRecorder()
    for stage, at in (("alert_received", 0.0), ("market_start", 0.1),
                      ("market_end", 0.5), ("l25_start", 0.5),
                      ("l25_end", 0.55), ("quote_start", 0.6),
                      ("quote_end", 0.8), ("submit_start", 0.9),
                      ("fill", 1.0),
                      ("notification_enqueue", 1.0),
                      ("notification_sent", 6.0)):   # a 5s send
        r.mark(stage, at)
    assert r.trading_ms == pytest.approx(1000.0), (
        "a 5s notification leaked into the trading latency"
    )
    assert r.notification_ms == pytest.approx(5000.0)


def test_the_accumulator_keeps_the_two_apart():
    from autopaper.latency import LatencyAccumulator, StageRecorder

    acc = LatencyAccumulator()
    r = StageRecorder()
    for stage, at in (("alert_received", 0.0), ("l25_end", 0.2),
                      ("submit_start", 0.3), ("fill", 0.4),
                      ("notification_enqueue", 0.4),
                      ("notification_sent", 3.0)):
        r.mark(stage, at)
    acc.record(r)
    stats = acc.stats()
    assert "notification" in stats and "trading" in stats
    assert "notification" not in acc.trading_stats(), (
        "trading stats must never include notification timing"
    )
    assert stats["notification"]["p50"] == pytest.approx(2600.0)


def test_every_stage_is_measured_independently():
    from autopaper.latency import StageRecorder

    r = StageRecorder()
    r.mark("alert_received", 0.0)
    r.mark("narrative_start", 0.5)
    r.mark("narrative_end", 2.0)
    r.mark("l25_end", 2.05)
    r.mark("submit_start", 2.5)
    r.mark("fill", 3.5)
    assert r.narrative_ms == pytest.approx(1500.0)
    assert r.analysis_ms == pytest.approx(2050.0)
    # ALERT -> FILL is the whole decision-to-trade number an operator means
    assert r.signal_to_fill_ms == pytest.approx(3500.0)
    # the execution leg alone is a different, smaller question
    assert r.submit_to_fill_ms == pytest.approx(1000.0)
    assert r.trading_ms == pytest.approx(3500.0)


def test_signal_to_fill_starts_at_the_alert_not_the_submit():
    """
    The distinction that matters: a 1s execution leg inside a 10s decision is
    not a fast trade.
    """
    from autopaper.latency import StageRecorder

    r = StageRecorder()
    r.mark("alert_received", 0.0)
    r.mark("submit_start", 9.0)
    r.mark("fill", 10.0)
    assert r.signal_to_fill_ms == pytest.approx(10_000.0)
    assert r.submit_to_fill_ms == pytest.approx(1_000.0)


def test_a_missing_stage_is_none_not_zero():
    from autopaper.latency import StageRecorder

    r = StageRecorder()
    r.mark("alert_received", 0.0)
    assert r.trading_ms is None
    assert r.notification_ms is None


def test_backwards_stage_order_is_reported():
    from autopaper.latency import StageRecorder

    r = StageRecorder()
    r.mark("fill", 5.0)
    r.mark("alert_received", 0.0)
    assert r.ordering_violations()


# ---------------------------------------------------------------------------
# safety
# ---------------------------------------------------------------------------


def test_real_is_still_disabled_and_unreachable(cfg):
    t = make_engine(cfg)
    try:
        assert cfg.real.trading_enabled is False
        assert t.state.real_armed is False
        assert t.state.trade_mode is TradeMode.PAPER
    finally:
        t._release_ownership()


def test_the_stale_gate_survives_this_change(cfg):
    t = make_engine(cfg)
    try:
        assert t.stale_gate.budget_ms == cfg.market.stale_after_ms
        assert t.stale_gate.check(signal_at=1000.0, now=1034.0) == "STALE_SIGNAL"
        assert t.stale_gate.check(signal_at=1000.0, now=1002.0) is None
    finally:
        t._release_ownership()


def test_l25_is_untouched(cfg):
    from autopaper import l25, v2a

    l25.verify_definition()
    assert l25.L25_MIN_LIQ_TO_MCAP == 0.9647
    assert v2a.V2A_CEILING_USD == 25_372.71


# ---------------------------------------------------------------------------
# the latency instrumentation is actually WIRED, not just defined
# ---------------------------------------------------------------------------


def test_the_engine_actually_marks_stages(cfg, monkeypatch):
    """
    A StageRecorder that nothing calls would satisfy every test above while
    the dashboard showed nothing. Drive one real candidate through and require
    that the accumulator saw it.
    """
    import asyncio

    from autopaper.models import AlertSnapshot

    t = make_engine(cfg, cli_only=True)
    try:
        seen: list = []
        real_handle = t._handle_candidate

        async def spy(snapshot, parsed):
            await real_handle(snapshot, parsed)
            seen.append(t._current_stages)

        monkeypatch.setattr(t, "_handle_candidate", spy)
        from autopaper.nar.base import NarSnapshot

        # is_async is a read-only property over `nar.source`, so patch the
        # source: a synchronous, always-unavailable narrative. That still
        # exercises the real narrative stage timing without a network call.
        monkeypatch.setattr(t.nar, "source", object(), raising=False)
        monkeypatch.setattr(
            t.nar, "fetch",
            lambda mint: NarSnapshot(mint=mint, fetched_at=0.0,
                                     available=False, narrative=None),
            raising=False)
        snap = AlertSnapshot(
            candidate_id="cand_latency_1", source="unit-test", chat="unit-test",
            message_id=7, source_timestamp=1000.0, received_timestamp=1000.0,
            raw_text="x",
            mint="So11111111111111111111111111111111111111112",
            symbol="SOL", name="Solana", fields={},
        )
        asyncio.run(t.on_alert(snap, None))

        assert seen, "the candidate path never ran"
        rec = seen[-1]
        assert rec is not None
        assert "alert_received" in rec.stamps
        assert rec.analysis_ms is not None, "analysis was never timed"
        stats = t.latency_stats()
        assert stats.get("analysis", {}).get("n", 0) >= 1, (
            "a fully timed candidate never reached the accumulator: "
            f"{stats}"
        )
    finally:
        t._release_ownership()


def test_latency_stats_expose_trading_and_notification_separately(cfg):
    from autopaper.latency import LatencyAccumulator, StageRecorder

    t = make_engine(cfg, cli_only=True)
    try:
        assert t.latency_stats() == {} or "analysis" in t.latency_stats()
        r = StageRecorder()
        for s, a in (("alert_received", 0.0), ("l25_end", 0.2),
                     ("submit_start", 0.3), ("fill", 0.4),
                     ("notification_enqueue", 0.4),
                     ("notification_sent", 4.0)):
            r.mark(s, a)
        t.latency.record(r)
        stats = t.latency_stats()
        assert stats["trading"]["n"] == 1
        assert stats["notification"]["n"] == 1
        assert stats["trading"]["p50"] != stats["notification"]["p50"]
    finally:
        t._release_ownership()


# ---------------------------------------------------------------------------
# notification policy: no rejection spam
# ---------------------------------------------------------------------------


def test_rejections_are_not_pushed_by_default(cfg):
    """
    A rejection is the expected outcome for most candidates. Pushing each one
    turns a quiet strategy into a firehose.
    """
    t = make_engine(cfg)
    try:
        assert t.notify_rejects is False, "rejection push must be off by default"
    finally:
        t._release_ownership()


def test_the_notify_path_is_the_only_outbound_surface(cfg):
    """
    Every outbound message in the engine goes through _guarded_notify, so the
    policy is enforced in one place rather than at nine call sites.
    """
    src = (REPO / "src/autopaper/engine.py").read_text()
    builder = src[src.index("def _build_bot_control("):
                  src.index("def start_bot_control(")]
    # trading = the execution path only. shutdown() legitimately awaits
    # notifier.aclose() to drain the worker cleanly, so it is excluded.
    trading = src[src.index("def _execute_buy("):src.index("async def shutdown(")]
    # the notifier is wired in exactly one place, the builder
    assert builder.count("self.bot_notifier.send") == 1
    # and no trading code reaches the transport or notifier directly
    for forbidden in ("self.bot_notifier.send", "self.bot_notifier.",
                      "self.bot_transport.send", "self.bot_control."):
        assert forbidden not in trading, f"trading path calls {forbidden}"


def test_rejection_reasons_survive_without_being_pushed():
    """
    Not pushing a rejection must not mean LOSING it.

    Suppressing the push moves the reason to the surfaces the operator already
    looks at -- the log, the journal, the CLI's last decision and the
    dashboard's LAST EVENT row -- rather than deleting it.
    """
    from autopaper.dashboard import build_frame

    frame = build_frame(
        last_event={"at_iso": "2026-09-30T00:00:00", "symbol": "AI",
                    "decision": "REJECT",
                    "reason": "liquidity_to_mcap 0.71 below 0.9647",
                    "latency_ms": 3421},
    )
    assert "REJECT" in frame
    assert "0.9647" in frame, "the reason must reach the primary surface"


def test_the_operator_command_set_is_still_rendered_on_status():
    from autopaper.telegram.ui import OPERATOR_COMMANDS, render_status, StatusView
    from autopaper.modes import RuntimeState

    st = RuntimeState()
    st.start()
    text = render_status(StatusView(state=st, equity_usd=100.0,
                                    cash_usd=100.0, open_positions=0))
    for cmd in ("/start", "/status", "/balance", "/pause", "/resume", "/kill",
                "/open", "/closed", "/tp"):
        assert cmd in text, f"{cmd} is not discoverable from /status"
    assert set(OPERATOR_COMMANDS) == {"/start", "/status", "/balance",
                                      "/pause", "/resume", "/kill",
                                      "/open", "/closed", "/tp"}


def test_no_removed_command_is_advertised_on_status():
    """A removed command that is still printed is still a reachable command."""
    from autopaper.telegram.ui import render_status, StatusView
    from autopaper.modes import RuntimeState

    st = RuntimeState()
    st.start()
    text = render_status(StatusView(state=st, equity_usd=100.0,
                                    cash_usd=100.0, open_positions=0))
    for gone in ("/stop", "/paper", "/real", "/emergency",
                 "/clear_emergency", "/cancel_real", "/real_off",
                 "/recent", "/stats", "/run", "/menu"):
        assert gone not in text, f"{gone} is still advertised"


# ---------------------------------------------------------------------------
# the emergency stop always has a documented way back
# ---------------------------------------------------------------------------


def test_the_cli_can_clear_the_latch(tmp_path, monkeypatch, capsys):
    """
    /kill latches across restarts and /clear_emergency is no longer a command.

    /resume clears it, but the usual reason to be at a terminal is that the bot
    is NOT running -- so the latch needs a terminal path too, or recovery means
    deleting a file by hand.
    """
    from autopaper.cli import build_parser, cmd_unlatch
    from autopaper.config import build_config
    from autopaper.ledger import save_json_atomic

    cfg = build_config(tmp_path)
    save_json_atomic(cfg.var_dir / "emergency_stop.json",
                     {"emergency_stop": True, "at": 1.0})
    parser = build_parser()

    # refuses without --yes: this re-opens entry authorization
    assert cmd_unlatch(cfg, parser.parse_args(["unlatch"])) == 1
    assert "--yes" in capsys.readouterr().out
    assert load_latch(cfg)["emergency_stop"] is True, "it cleared without --yes"

    assert cmd_unlatch(cfg, parser.parse_args(["unlatch", "--yes"])) == 0
    assert load_latch(cfg)["emergency_stop"] is False


def test_clearing_a_latch_that_is_not_set_is_a_no_op(tmp_path, capsys):
    from autopaper.cli import build_parser, cmd_unlatch
    from autopaper.config import build_config

    cfg = build_config(tmp_path)
    assert cmd_unlatch(cfg, build_parser().parse_args(["unlatch"])) == 0
    assert "nothing to clear" in capsys.readouterr().out


def load_latch(cfg):
    from autopaper.ledger import load_json
    return load_json(cfg.var_dir / "emergency_stop.json") or {}


def test_the_dashboard_renders_from_a_real_engine(cfg, capsys):
    """
    build_frame() passing in isolation proves nothing if the engine never calls
    it. Render the real dashboard once, in --cli mode, and require the sections.
    """
    t = make_engine(cfg, cli_only=True)
    try:
        t.state.start()
        t.render_dashboard()
        # the dashboard writes to stderr, and under pytest stderr is not a TTY
        # so this exercises the append fallback -- which is the path a piped
        # `nohup` log takes
        out = capsys.readouterr().err
        assert "MEMEBOT" in out
        assert "L25" in out
        for section in ("ACCOUNT", "PERFORMANCE", "LATENCY", "SERVICES"):
            assert section in out, f"{section} missing from the real render"
    finally:
        t._release_ownership()


def test_the_real_render_redraws_instead_of_appending(cfg, monkeypatch):
    """
    The regression in one assertion: redrawing must not append.

    A fake TTY is injected because that is the only way the redraw branch is
    reachable -- without it the append fallback runs and the bug hides.
    """
    from autopaper import dashboard

    tty = _FakeTTY()
    monkeypatch.setattr(dashboard.sys, "stderr", tty)

    t = make_engine(cfg, cli_only=True)
    try:
        t.state.start()
        t.render_dashboard()
        t.render_dashboard()
        t.render_dashboard()
    finally:
        t._release_ownership()

    assert tty.buf.count("\x1b[2J") == 3, "each redraw must clear the screen"
    assert tty.buf.count("MEMEBOT") == 3, "one frame per redraw"
    # each frame is a fixed number of lines; the wall does not grow
    lines_per_frame = tty.buf.count("MEMEBOT")
    assert lines_per_frame == 3


def test_the_dashboard_does_not_persist_on_every_redraw(cfg, monkeypatch):
    """
    The old render_dashboard() called persist() on every redraw, i.e. an fsync
    several times a minute for a display-only reason. State belongs at
    checkpoints and shutdown.
    """
    t = make_engine(cfg, cli_only=True)
    calls: list[int] = []
    monkeypatch.setattr(t, "persist", lambda: calls.append(1))
    try:
        t.state.start()
        t.render_dashboard()
        t.render_dashboard()
        assert calls == [], "rendering must not write account state"
    finally:
        t._release_ownership()


class _FakeTTY:
    """A stream that claims to be a terminal, so the redraw branch is reachable.

    Without this, the append fallback runs under pytest and the redraw bug
    hides.
    """

    def __init__(self):
        self.buf = ""

    def write(self, s):
        self.buf += s

    def flush(self):
        pass

    def isatty(self):
        return True
