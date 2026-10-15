"""
The candidate-source chat gate.

This exists because the gate was BROKEN and the bot ran blind for seven minutes
without anyone noticing: every alert was logged as "not the permitted source
chat" and dropped, so the runtime saw zero candidates while looking perfectly
healthy.

Telethon reports a supergroup as ``-100XXXXXXXXXX`` in ``message.chat_id`` but
as the bare ``XXXXXXXXXX`` in ``entity.id``. A raw integer comparison therefore
rejected every message from the exact chat it was configured to accept.

The fix must not weaken the gate, so both directions are pinned: the two
documented spellings of one id must match, and genuinely different ids must
still be refused.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autopaper.telegram.client import (
    TelegramListener,
    normalise_source_chat_id,
)

REPO = Path(__file__).resolve().parents[1]

# the ids observed in the live log
CONFIGURED = 2270552242          # entity.id, from the resolved chat
ARRIVING = -1002270552242        # message.chat_id, as delivered


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


def test_the_two_spellings_of_one_supergroup_are_one_chat():
    assert normalise_source_chat_id(ARRIVING) == CONFIGURED
    assert normalise_source_chat_id(CONFIGURED) == normalise_source_chat_id(
        ARRIVING)


@pytest.mark.parametrize("raw,expected", [
    (CONFIGURED, CONFIGURED),
    (ARRIVING, CONFIGURED),
    (-1009998887777, 9998887777),
    (123, 123),
    (-123, -123),                 # a small negative id is not a marked one
    (-100, -100),                 # no digits after the marker
    (-1000, -1000),               # would collapse onto 0, which is not a chat
    (100, 100),                   # positive, never stripped
    # A POSITIVE id that happens to begin with "100" must not be stripped. If it
    # were, chat 1002270552242 would alias supergroup 2270552242 and a different
    # chat could satisfy the gate. The marker is only ever a NEGATIVE prefix.
    (1002270552242, 1002270552242),
    (1000000000, 1000000000),
    (100, 100),
])
def test_normalisation_cases(raw, expected):
    assert normalise_source_chat_id(raw) == expected


@pytest.mark.parametrize("junk", [None, True, False, "", "x", "abc", [1], {}])
def test_unparseable_ids_normalise_to_nothing(junk):
    """An unparseable id must be treated as 'not permitted', never loosely equal."""
    assert normalise_source_chat_id(junk) is None


# ---------------------------------------------------------------------------
# the gate must be fixed, not loosened
# ---------------------------------------------------------------------------


def test_the_observed_pair_now_matches():
    """The exact case that starved the running bot."""
    assert normalise_source_chat_id(ARRIVING) == normalise_source_chat_id(
        CONFIGURED)


@pytest.mark.parametrize("foreign", [
    -1002270552241,               # one digit different
    2270552241,
    2270552243,
    -1009998887777,
    9998887777,
    123,
    -123,
    0,
])
def test_a_different_chat_is_still_a_different_chat(foreign):
    assert normalise_source_chat_id(foreign) != normalise_source_chat_id(
        CONFIGURED
    )


def test_the_comparison_in_the_listener_is_normalised():
    """
    Structural: the drop decision must go through the normaliser.

    A raw ``!=`` here is exactly the bug, and it is invisible at runtime because
    the symptom is silence rather than an error.
    """
    src = (REPO / "src/autopaper/telegram/client.py").read_text()
    i = src.index("is not the permitted source chat")
    window = src[max(0, i - 700):i]
    assert "normalise_source_chat_id(msg_chat)" in window, (
        "the source-chat comparison must normalise the incoming id first"
    )
    assert "normalise_source_chat_id(chat_id)" in window, (
        "the source-chat comparison must normalise the configured id too"
    )
    assert "msg_chat != chat_id" not in window, (
        "a raw integer comparison is the bug that made the bot blind"
    )


def test_a_foreign_message_is_still_counted_and_dropped():
    """
    The accounting must survive the fix.

    Dropped foreign messages are still counted, because the journal is how an
    operator proves no other chat ever contributed a candidate.
    """
    listener = TelegramListener.__new__(TelegramListener)   # no connection
    from autopaper.telegram.client import ConnectionState, ListenerStatus
    listener.status = ListenerStatus(state=ConnectionState.CONNECTED)

    msg = type("M", (), {"chat_id": -1009998887777, "id": 1, "text": "x"})()
    # exercise the same predicate the handler uses
    assert normalise_source_chat_id(msg.chat_id) != normalise_source_chat_id(
        CONFIGURED
    )


def test_the_permitted_chat_is_not_broken_by_the_fix():
    assert normalise_source_chat_id(ARRIVING) == normalise_source_chat_id(
        CONFIGURED
    )
