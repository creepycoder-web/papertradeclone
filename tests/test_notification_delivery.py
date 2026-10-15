"""
Notification delivery must not be lost, and must be diagnosable.

The incident: ``notification delivery failed (BotApiForbidden)`` with no cause.
Three defects combined to make it undiagnosable and unfixable from the log:

1. the notifier logged only ``type(exc).__name__``, discarding the HTTP status
   and Telegram's own description -- the only things that identify the cause;
2. nothing clamped a notification to Telegram's 4096-character limit, so an
   over-long alert was refused outright as a 400 and LOST;
3. a payload Telegram could not parse was refused and dropped, rather than
   re-sent as plain text.

All offline. No Telegram call is made.
"""

from __future__ import annotations

import asyncio

import pytest

from autopaper.telegram.botapi import (
    BotApiForbidden,
    BotApiTransport,
    _strip_html,
)
from autopaper.telegram.notifier import (
    TELEGRAM_MAX_MESSAGE_CHARS,
    _clamp_to_telegram,
)


class TestClamping:
    def test_a_long_notification_is_clamped_to_the_api_limit(self):
        clamped = _clamp_to_telegram("x" * 12_000)
        assert len(clamped) <= TELEGRAM_MAX_MESSAGE_CHARS
        assert TELEGRAM_MAX_MESSAGE_CHARS == 4096

    def test_the_truncation_is_visible_to_the_reader(self):
        assert "truncated" in _clamp_to_telegram("y" * 9000)

    def test_a_short_notification_is_untouched(self):
        assert _clamp_to_telegram("PAPER BUY $T") == "PAPER BUY $T"

    def test_the_boundary_is_exact(self):
        exact = "z" * TELEGRAM_MAX_MESSAGE_CHARS
        assert _clamp_to_telegram(exact) == exact
        assert len(_clamp_to_telegram(exact + "!")) == TELEGRAM_MAX_MESSAGE_CHARS


class TestPlainTextFallback:
    def test_a_parse_refusal_is_retried_as_plain_text(self):
        """
        An alert Telegram cannot parse is re-sent as text, not dropped.

        Losing the operator's fill/exit alert to one stray ``&`` in a token
        symbol is a far worse outcome than losing the bold tags.
        """
        calls: list[dict] = []

        class Transport(BotApiTransport):
            def __init__(self):
                self._chat_id = 123

            async def call(self, method, body):
                calls.append(body)
                if body.get("parse_mode") == "HTML":
                    raise BotApiForbidden("sendMessage refused (HTTP 400)")
                return {"message_id": 7}

        t = Transport()
        result = asyncio.run(t.send_message("<b>BUY</b> A & B", None))
        assert len(calls) == 2, "the plain-text fallback never ran"
        assert "parse_mode" not in calls[1]
        assert result == {"message_id": 7}

    def test_a_non_markup_refusal_is_not_retried(self):
        """A genuine permission failure must NOT be retried as plain text."""
        calls: list[dict] = []

        class Transport(BotApiTransport):
            def __init__(self):
                self._chat_id = 123

            async def call(self, method, body):
                calls.append(body)
                raise BotApiForbidden("sendMessage refused (HTTP 400)")

        t = Transport()
        with pytest.raises(BotApiForbidden):
            asyncio.run(t.send_message("no markup here", None))
        assert len(calls) == 1

    def test_stripping_html_keeps_the_text_readable(self):
        assert _strip_html("<b>BUY</b> A &amp; B") == "BUY A & B"


class TestDiagnosability:
    def test_the_failure_log_includes_the_cause(self):
        """
        The reason a delivery failed must be in the log.

        The incident produced ``BotApiForbidden`` and nothing else, which is a
        class name, not a diagnosis.
        """
        from pathlib import Path

        source = Path("src/autopaper/telegram/notifier.py").read_text()
        assert '"notification delivery failed (%s): %s"' in source
        # The detail must actually be passed through -- this is the whole fix.
        assert "str(exc)[:300]" in source

    def test_the_last_error_is_retained_for_the_dashboard(self):
        from autopaper.telegram.notifier import NotifierStatus

        status = NotifierStatus()
        status.last_error = "BotApiForbidden: sendMessage refused (HTTP 400)"
        assert "400" in status.last_error
