"""
Read-only Telegram listener gates, and an end-to-end pipeline run on fixtures.

The listener tests matter for a reason beyond hygiene: this harness reads a
live channel using the operator's real session, so the gates that keep other
chats out are security-relevant, not cosmetic.
"""


from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from autopaper.config import TelegramConfig
from autopaper.same_market.listener import SameMarketListener

pytestmark = pytest.mark.same_market

CHAT = 2270552242


class _Entity:
    def __init__(self, eid=CHAT, chat_id=CHAT):
        self.id = eid
        self.chat_id = chat_id
        self.title = "Moo Deng President Callers"


class _Msg:
    #: sender 500 is "someone else"; the listener's own_id is 99, so these are
    #: genuinely inbound messages rather than our own echo.
    def __init__(self, mid, text, *, chat_id=CHAT, sender_id=500,
                 when=None):
        self.id = mid
        self.message = text
        self.text = text
        self.chat_id = chat_id
        self.sender_id = sender_id
        self.date = when or datetime(2026, 10, 2, 10, 15, 0, tzinfo=timezone.utc)


def _listener() -> tuple[SameMarketListener, list]:
    seen: list = []

    async def _cb(text, mid, ts, received):
        seen.append((text, mid, ts, received))

    listener = SameMarketListener(
        TelegramConfig(api_id=1, api_hash="h", session="s", chat=str(CHAT)),
        _cb,
    )
    listener.status.chat_id = CHAT
    listener.own_id = 99
    return listener, seen


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def test_a_new_message_is_forwarded_with_its_timing():
    listener, seen = _listener()
    msg = _Msg(1, "CA: H9XQrbw5MANzYVgHjXJbScFxWqyGw9CCAec5XDTgpump")
    asyncio.run(listener._handle(msg))
    assert len(seen) == 1
    text, mid, ts, received = seen[0]
    assert "CA:" in text
    assert mid == 1
    assert ts == msg.date.timestamp()
    assert received > 0


def test_a_message_from_another_chat_is_refused():
    """
    The chat is re-checked on THIS message, not against the already-resolved
    id - comparing the resolved id against itself is a check that cannot fail.
    """
    listener, seen = _listener()
    asyncio.run(listener._handle(_Msg(2, "x", chat_id=999)))
    assert seen == []
    assert listener.messages_refused_foreign_chat == 1


def test_our_own_message_is_refused():
    listener, seen = _listener()
    asyncio.run(listener._handle(_Msg(3, "x", sender_id=99)))
    assert seen == []
    assert listener.messages_refused_own == 1


def test_a_duplicate_message_id_is_forwarded_only_once():
    listener, seen = _listener()
    msg = _Msg(4, "x")
    asyncio.run(listener._handle(msg))
    asyncio.run(listener._handle(msg))
    assert len(seen) == 1
    assert listener.messages_refused_duplicate == 1


def test_a_message_without_an_id_is_ignored():
    listener, seen = _listener()
    asyncio.run(listener._handle(_Msg(None, "x")))
    assert seen == []


def test_a_message_with_an_unusable_date_yields_unknown_latency_not_zero():
    """A Telegram timestamp the harness cannot read must become UNKNOWN."""
    listener, seen = _listener()
    msg = _Msg(5, "x")
    msg.date = object()          # set on the instance: __init__ would overwrite
    asyncio.run(listener._handle(msg))
    assert len(seen) == 1
    assert seen[0][2] is None, "an unusable timestamp must be UNKNOWN, not 0"


def test_transport_latency_is_none_without_a_telegram_timestamp():
    from autopaper.same_market.listener import Capture

    assert Capture("x", 1, None, 100.0).transport_latency_ms is None
    assert Capture("x", 1, 99.58, 100.0).transport_latency_ms == pytest.approx(420.0)


def test_the_listener_diagnostics_expose_no_chat_id_or_token():
    listener, _ = _listener()
    payload = listener.diagnostics()
    assert "chat_id" not in payload
    assert "session" not in payload
    assert "api_hash" not in payload
    for value in payload.values():
        assert not (isinstance(value, str) and "2270552242" in value)


def test_the_listener_cannot_send_a_telegram_message():
    for forbidden in ("send_message", "send_file", "sign_in", "send_code_request"):
        assert not hasattr(SameMarketListener, forbidden), forbidden


# ---------------------------------------------------------------------------
# transport latency, as reported
# ---------------------------------------------------------------------------

def test_transport_latency_uses_both_timestamps():
    from autopaper.same_market.listener import Capture

    capture = Capture("x", 1, 1_700_000_000.0, 1_700_000_000.420)
    assert capture.transport_latency_ms == pytest.approx(420.0)


def test_a_backwards_timestamp_is_clamped():
    from autopaper.same_market.listener import Capture

    assert Capture("x", 1, 2_000.0, 1_000.0).transport_latency_ms == 0.0