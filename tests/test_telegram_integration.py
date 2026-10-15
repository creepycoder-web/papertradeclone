"""
Notifier behaviour and forward-validation integration.

Two claims are pinned here:

* a Telegram failure cannot alter trading state, and cannot block the loop;
* forward validation still has exactly ONE recording path (the Ledger), and
  enabling Telegram does not turn historical reconstruction into prospective
  data.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from autopaper.config import build_config
from autopaper.engine import PaperTrader
from autopaper.forward_validation import (
    VALIDATING,
    Dataset,
    build_report,
    reconstruct_dataset,
)
from autopaper.telegram import botapi
from autopaper.telegram.botapi import BotApiTransport, BotControl, BotStatus
from autopaper.telegram.notifier import TelegramNotifier

ALLOWED = -1001234567890


# ---------------------------------------------------------------------------
# notifier: queueing and delivery
# ---------------------------------------------------------------------------


def make_notifier(handler, *, transport_kw=None, **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    t = BotApiTransport("999:FAKE", chat_id=str(ALLOWED), client=client,
                        **(transport_kw or {}))
    return TelegramNotifier(t, min_send_interval_s=0.0, **kw), t


def test_a_successful_send_is_delivered():
    sent = []

    async def handler(request):
        sent.append(request)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    async def go():
        n, t = make_notifier(handler)
        n.start()
        n.send("hello", None)
        assert await n.flush()
        await n.aclose()
        return n

    n = asyncio.run(go())
    assert len(sent) == 1
    assert n.status.sent == 1


def test_send_returns_immediately_without_io():
    """The engine calls this from the trading path, so it must not block."""

    async def handler(request):
        await asyncio.sleep(5)          # a slow Telegram
        return httpx.Response(200, json={"ok": True, "result": {}})

    async def go():
        n, _ = make_notifier(handler)
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        for _ in range(50):
            n.send("x", None)
        elapsed = loop.time() - t0
        await n.aclose()
        return elapsed

    assert asyncio.run(go()) < 0.5, "enqueueing must not await the network"


def test_a_full_queue_drops_rather_than_blocking():
    async def handler(request):
        return httpx.Response(200, json={"ok": True, "result": {}})

    async def go():
        n, _ = make_notifier(handler, max_queue=5)
        for i in range(50):
            n.send(f"m{i}", None)
        stats = n.status_dict()
        await n.aclose()
        return stats

    stats = asyncio.run(go())
    assert stats["dropped_queue_full"] > 0, "overflow must be counted"
    assert stats["attached"] is True


def test_a_send_failure_is_counted_and_does_not_raise():
    async def handler(request):
        return httpx.Response(401, json={"ok": False, "description": "no"})

    async def go():
        n, t = make_notifier(handler, transport_kw={"max_retries": 0})
        n.start()
        n.send("hello", None)
        await n.flush()
        await n.aclose()
        return n

    n = asyncio.run(go())
    assert n.status.sent == 0
    assert n.status.failed == 1, "a failure must be visible in diagnostics"


def test_a_send_timeout_does_not_raise():
    async def handler(request):
        raise httpx.ReadTimeout("slow")

    async def go():
        n, t = make_notifier(handler, transport_kw={"max_retries": 0})
        n.start()
        n.send("hello", None)
        await n.flush()
        await n.aclose()
        return n

    n = asyncio.run(go())
    assert n.status.sent == 0
    assert n.status.failed == 1


def test_sends_are_paced():
    """A burst must not become a burst of API calls."""
    times = []

    async def handler(request):
        times.append(asyncio.get_running_loop().time())
        return httpx.Response(200, json={"ok": True, "result": {}})

    async def go():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        t = BotApiTransport("999:FAKE", chat_id=str(ALLOWED), client=client)
        n = TelegramNotifier(t, min_send_interval_s=0.05)
        n.start()
        for i in range(4):
            n.send(f"m{i}", None)
        await n.flush()
        await n.aclose()

    asyncio.run(go())
    assert len(times) == 4
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert all(g >= 0.04 for g in gaps), gaps


def test_a_malformed_keyboard_is_dropped_not_forwarded():
    from autopaper.telegram.notifier import _coerce_keyboard

    good = [[{"text": "a", "callback_data": "start"}]]
    assert _coerce_keyboard(good) == good
    for junk in ([{"nonsense": 1}], "not a keyboard", None, 42, [], [[]]):
        assert _coerce_keyboard(junk) is None, junk


def test_the_notifier_holds_no_runtime_state():
    """
    It cannot alter trading state because it has no reference to any.

    Structural: no state import, no reference to RuntimeState.
    """
    from autopaper.telegram import notifier as mod

    src = open(mod.__file__, encoding="utf-8").read()
    assert "RuntimeState" not in src
    assert "arm_real" not in src
    assert "real_armed" not in src


# ---------------------------------------------------------------------------
# integration: engine + transport + forward validation
# ---------------------------------------------------------------------------


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "999:FAKEfakefake")
    monkeypatch.setenv("TELEGRAM_BOT_CHAT", str(ALLOWED))
    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    t = PaperTrader(build_config(tmp_path))
    try:
        yield t
    finally:
        t._release_ownership()


def test_start_reaches_the_paper_runtime_and_its_gate(engine):
    c = BotControl(transport=engine.bot_transport,
                   handle_button=engine.handle_button, allowed_chat=ALLOWED)
    allowed, _ = engine.state.can_open_new_entries()
    assert allowed is False
    asyncio.run(c.handle_update({
        "update_id": 1,
        "callback_query": {"id": "s", "data": "start",
                           "message": {"chat": {"id": ALLOWED}}}}))
    allowed, why = engine.state.can_open_new_entries()
    assert allowed is True, why


def test_the_operator_start_is_journalled(engine):
    """The existing journal records it, so the lifecycle is auditable."""
    c = BotControl(transport=engine.bot_transport,
                   handle_button=engine.handle_button, allowed_chat=ALLOWED)
    asyncio.run(c.handle_update({
        "update_id": 1,
        "callback_query": {"id": "s", "data": "start",
                           "message": {"chat": {"id": ALLOWED}}}}))
    events = [json_line for json_line in engine.journal.all_events()]
    assert any(e.get("event") == "operator_start" for e in events), events


def test_forward_validation_is_still_the_single_recording_path(engine):
    """
    The Ledger mirrors into forward validation. Nothing else does.

    So there is exactly one attachment point, and a Telegram update cannot open
    a second route into the prospective store.
    """
    from autopaper.ledger import Ledger

    assert engine.ledger.forward is engine.forward
    # the engine holds no other recorder
    recorders = [v for v in vars(engine).values()
                 if hasattr(v, "all_records") and hasattr(v, "candidates_path")]
    assert len(recorders) == 1, "more than one prospective store"
    # and the Ledger is what owns it
    assert isinstance(engine.ledger, Ledger)
    assert engine.ledger.forward is engine.forward


def test_the_deployment_boundary_is_unaffected_by_telegram(engine, tmp_path):
    """Enabling Telegram must not move or reset the boundary."""
    first = engine.forward_deployment.started_at
    c = BotControl(transport=engine.bot_transport,
                   handle_button=engine.handle_button, allowed_chat=ALLOWED)
    asyncio.run(c.handle_update({
        "update_id": 1,
        "callback_query": {"id": "s", "data": "start",
                           "message": {"chat": {"id": ALLOWED}}}}))
    engine._release_ownership()

    second = PaperTrader(build_config(tmp_path))
    try:
        assert second.forward_deployment.started_at == first
        assert (second.forward_deployment.deployment_id
                == engine.forward_deployment.deployment_id)
    finally:
        second._release_ownership()


def test_historical_reconstruction_is_still_not_prospective():
    """
    Enabling Telegram must not relabel anything.

    A pre-L25 row re-scored offline stays POST_DERIVATION_RECONSTRUCTED and can
    never be a validating dataset.
    """
    rows = [{"candidate_id": "h1", "at": 1.0, "route": "FAST_ENTRY",
             "signal_quality": 0.9,
             "features": {"market_cap_usd": 1000.0, "liquidity_usd": 5000.0}}]
    out = reconstruct_dataset(rows, l25_min_liq_to_mcap=0.9647)
    assert out[0]["dataset"] == Dataset.POST_DERIVATION_RECONSTRUCTED.value
    assert out[0]["dataset"] is not VALIDATING

    # and it cannot be smuggled into the prospective store
    out[0]["deployment_id"] = "forged"
    report = build_report(out, deployment=None)
    assert report.dataset is VALIDATING or report.status.value in (
        "NO_DEPLOYMENT", "DATA_QUALITY_PROBLEM")


def test_a_telegram_start_does_not_create_prospective_records_by_itself(engine):
    """
    Starting the runtime is not a trade. The prospective store stays empty until
    a real decision and fill happen through the Ledger.
    """
    c = BotControl(transport=engine.bot_transport,
                   handle_button=engine.handle_button, allowed_chat=ALLOWED)
    asyncio.run(c.handle_update({
        "update_id": 1,
        "callback_query": {"id": "s", "data": "start",
                           "message": {"chat": {"id": ALLOWED}}}}))
    assert engine.forward.all_records() == []
    report = build_report(engine.forward.all_records(),
                          deployment=engine.forward_deployment)
    assert report.n_closed == 0
    assert report.is_validating is False
    assert report.status.value in ("NOT_ENOUGH_DATA", "NO_DEPLOYMENT")


# ---------------------------------------------------------------------------
# observability
# ---------------------------------------------------------------------------


def test_status_is_sanitised(engine):
    st = engine.bot_status()
    blob = repr(st)
    assert "FAKE" not in blob, "the token must never appear in status"
    assert "999:" not in blob
    assert str(ALLOWED) not in blob, "the allow-listed chat must not be published"
    assert st["token_configured"] is True
    assert st["chat_configured"] is True
    assert st["control_ready"] is True
    assert st["notifier_attached"] is True
    assert st["transport_initialized"] is True
    assert st["forward_validation_active"] is True
    assert st["real_trading_enabled"] is False
    assert st["trade_mode"] == "PAPER"
    assert st["run_state"] == "OFF"


def test_status_answers_the_diagnostic_questions(engine):
    st = engine.bot_status()
    for key in ("transport_initialized", "chat_configured", "notifier_attached",
                "run_state", "trade_mode", "real_trading_enabled",
                "forward_validation_active"):
        assert key in st, key
    # transport-specific diagnostics are namespaced, not flattened, so a
    # transport field can never clobber an engine-level one
    assert "last_error" in st["control"]
    for key in ("polling", "updates_seen", "updates_refused_chat",
                "updates_refused_replay", "chat_authorized", "replay_guard"):
        assert key in st["control"], key
    for key in ("queued", "sent", "failed", "dropped_queue_full", "attached"):
        assert key in st["notifier"], key


def test_namespacing_prevents_a_field_collision():
    """
    A transport field and an engine field may share a name without one
    overwriting the other. This is the regression that a flat update caused.
    """
    st = BotStatus()
    assert not hasattr(st, "notifier_attached"), (
        "BotStatus must not own a notifier field: the transport does not know "
        "about the notifier, and the collision silently reported False"
    )


def test_status_records_a_failure_without_its_detail(engine):
    from autopaper.telegram.botapi import BotApiTransient

    async def boom(*a, **k):
        raise BotApiTransient("network gone")

    engine.bot_transport.get_updates = boom
    with pytest.raises(BotApiTransient):
        asyncio.run(engine.bot_control.poll_once(timeout_s=1))
    st = engine.bot_status()
    assert st["control"]["last_error"], "a failure must be visible"
    assert "999:" not in repr(st)
    assert "FAKE" not in repr(st)


def test_status_when_nothing_is_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_CHAT", raising=False)
    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    t = PaperTrader(build_config(tmp_path))
    try:
        st = t.bot_status()
        assert st["control_ready"] is False
        assert st["token_configured"] is False
        assert st["notifier_attached"] is False
        assert st["real_trading_enabled"] is False
    finally:
        t._release_ownership()
