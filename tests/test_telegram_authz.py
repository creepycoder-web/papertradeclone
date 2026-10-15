"""
The chat allow-list is the authorization boundary.

An update is only ever dispatched when its numeric chat id equals the configured
one. These tests pin the refusal cases, because the failure mode of a control
channel is not "the operator is annoyed" -- it is "a stranger starts the bot".
"""

from __future__ import annotations

import asyncio

import pytest

from autopaper.telegram import botapi
from autopaper.telegram.botapi import (
    BotControl,
    ReplayGuard,
    chat_is_authorized,
    extract_chat_id,
    normalise_chat_id,
    payload_from_update,
)

ALLOWED = -1001234567890


def msg_update(chat, uid=1, text="/status"):
    return {"update_id": uid, "message": {"text": text, "chat": {"id": chat}}}


def cb_update(chat, uid=1, data="start", cb_id="cb1"):
    return {
        "update_id": uid,
        "callback_query": {
            "id": cb_id, "data": data,
            "message": {"chat": {"id": chat}},
        },
    }


class Recorder:
    """Stands in for the engine's handle_button."""

    def __init__(self):
        self.seen: list[str] = []

    async def __call__(self, data):
        self.seen.append(data)


def control(allowed=ALLOWED, rec=None):
    t = botapi.BotApiTransport("TOK", chat_id=str(allowed))
    rec = rec or Recorder()
    c = BotControl(transport=t, handle_button=rec, allowed_chat=allowed)
    return c, rec


# ---------------------------------------------------------------------------
# id parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    (ALLOWED, ALLOWED),
    (str(ALLOWED), ALLOWED),
    (f"  {ALLOWED}  ", ALLOWED),
    (12345, 12345),
    ("12345", 12345),
    (-1001234567890, -1001234567890),
])
def test_canonical_ids_parse(raw, expected):
    assert normalise_chat_id(raw) == expected


@pytest.mark.parametrize("raw", [
    "@someusername", "@", "", "   ", None, "abc", "12a", "1.5",
    " 1 2", "+12345", "0012345", True, False, [], {}, 3.5,
    "١٢٣",                     # non-ASCII digits: a homograph, not an id
])
def test_non_canonical_ids_are_refused(raw):
    """Usernames, padded zeros, floats and bools are all refusals."""
    assert normalise_chat_id(raw) is None


def test_a_bool_is_not_an_id():
    """bool subclasses int; True must not become chat 1."""
    assert normalise_chat_id(True) is None
    assert normalise_chat_id(False) is None


def test_padded_zero_does_not_alias_a_real_id():
    assert normalise_chat_id("00123") is None
    assert normalise_chat_id("123") == 123


# ---------------------------------------------------------------------------
# chat extraction
# ---------------------------------------------------------------------------


def test_chat_id_is_extracted_from_each_container():
    assert extract_chat_id(msg_update(ALLOWED)) == ALLOWED
    assert extract_chat_id(cb_update(ALLOWED)) == ALLOWED
    assert extract_chat_id(
        {"update_id": 1, "edited_message": {"chat": {"id": ALLOWED}}}) == ALLOWED
    assert extract_chat_id(
        {"update_id": 1, "channel_post": {"chat": {"id": ALLOWED}}}) == ALLOWED


def test_a_callback_with_no_resolvable_chat_extracts_nothing():
    """
    A callback_query whose message is absent must NOT fall back to some other
    container, or an update could be authorized by an unrelated field.
    """
    assert extract_chat_id(
        {"update_id": 1, "callback_query": {"id": "x", "data": "start"}}) is None
    assert extract_chat_id({"update_id": 1}) is None
    assert extract_chat_id("not a mapping") is None
    assert extract_chat_id({"update_id": 1, "message": {}}) is None


# ---------------------------------------------------------------------------
# authorization
# ---------------------------------------------------------------------------


def test_the_configured_chat_is_authorized():
    assert chat_is_authorized(msg_update(ALLOWED), ALLOWED) is True
    assert chat_is_authorized(msg_update(str(ALLOWED)), ALLOWED) is True


@pytest.mark.parametrize("foreign", [
    0, 1, -1, 999999, -1009999999999, "@admin", "not-an-id", None, "", True,
    -1001234567891,      # one digit different
])
def test_every_other_chat_is_refused(foreign):
    assert chat_is_authorized(msg_update(foreign), ALLOWED) is False


def test_a_malformed_chat_in_the_update_is_refused():
    assert chat_is_authorized(msg_update("@admin"), ALLOWED) is False
    assert chat_is_authorized({"update_id": 1, "message": {"chat": {}}},
                              ALLOWED) is False
    assert chat_is_authorized({"update_id": 1, "message": {"text": "/start"}},
                              ALLOWED) is False


def test_nothing_is_authorized_without_a_valid_allow_list():
    """With no usable configured chat, the channel is closed, not open."""
    for bad in (None, "", "@admin", "abc", True, []):
        assert chat_is_authorized(msg_update(ALLOWED), bad) is False


def test_a_callback_from_an_unauthorized_chat_is_refused():
    assert chat_is_authorized(cb_update(-1), ALLOWED) is False
    assert chat_is_authorized(cb_update(None), ALLOWED) is False


def test_the_update_chat_is_not_read_from_the_payload():
    """
    An attacker-controlled field must not influence authorization.

    The text says the allow-listed id, and the real chat is foreign: still
    refused, because authorization reads the chat container only.
    """
    hostile = msg_update(-1, text=f"/start {ALLOWED}")
    assert chat_is_authorized(hostile, ALLOWED) is False


# ---------------------------------------------------------------------------
# dispatch requires authorization
# ---------------------------------------------------------------------------


def test_an_authorized_command_is_dispatched():
    c, rec = control()
    assert asyncio.run(c.handle_update(msg_update(ALLOWED))) is True
    assert rec.seen == ["stats"]


@pytest.mark.parametrize("foreign", [-1, 0, 42, None, "@admin"])
def test_an_unauthorized_start_is_never_dispatched(foreign):
    c, rec = control()
    assert asyncio.run(c.handle_update(msg_update(foreign))) is False
    assert rec.seen == []
    assert c.status_dict()["updates_refused_chat"] == 1


def test_an_unauthorized_callback_is_never_dispatched():
    c, rec = control()
    assert asyncio.run(c.handle_update(cb_update(-1, data="confirm_real:abc"))) \
        is False
    assert rec.seen == []


def test_a_callback_with_no_chat_is_never_dispatched():
    c, rec = control()
    update = {"update_id": 1, "callback_query": {"id": "x", "data": "start"}}
    assert asyncio.run(c.handle_update(update)) is False
    assert rec.seen == []


def test_a_refusal_is_reported_to_the_hook_without_the_payload():
    seen = []
    t = botapi.BotApiTransport("TOK", chat_id=str(ALLOWED))
    c = BotControl(transport=t, handle_button=Recorder(), allowed_chat=ALLOWED,
                   on_refused=lambda why, u: seen.append((why, u)))
    asyncio.run(c.handle_update(msg_update(-1, text="attacker text")))
    assert seen and seen[0][0] == "unauthorized_chat"


# ---------------------------------------------------------------------------
# payload parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("/start", "menu"), ("/START", "menu"), ("/start now", "menu"),
    ("/status", "stats"), ("/STATUS", "stats"), ("/status now", "stats"),
    ("  /balance  ", "balance"), ("/pause", "pause"), ("/resume", "resume"),
    ("/kill", "kill"),
])
def test_the_five_operator_commands_map_onto_the_vocabulary(text, expected):
    assert payload_from_update(msg_update(ALLOWED, text=text))[0] == expected


@pytest.mark.parametrize("removed", [
    "/run", "/stop", "/stats", "/paper", "/real", "/emergency",
    "/clear_emergency", "/cancel_real", "/real_off", "/recent",
    "/menu",
])
def test_every_removed_command_is_gone(removed):
    """
    The bot interface was reduced to five commands.

    Each removed one used to be a way to steer the runtime, or a source of
    per-alert spam on the critical path.
    """
    assert payload_from_update(msg_update(ALLOWED, text=removed))[0] is None, removed


def test_the_command_set_is_exactly_nine():
    """
    Eight commands: the menu, five actions, and two read-only position views.

    /start is a MENU, not a trading command -- it is pinned separately below,
    because "it starts the bot" is the specific wrong behaviour. /open and
    /closed are pure reads that dispatch onto the same callbacks the position
    buttons use.
    """
    from autopaper.telegram.botapi import TEXT_COMMANDS
    from autopaper.telegram.ui import OPERATOR_COMMANDS

    assert set(TEXT_COMMANDS) == set(OPERATOR_COMMANDS)
    assert set(TEXT_COMMANDS) == {"/start", "/status", "/balance", "/pause",
"/resume", "/kill", "/open", "/closed", "/tp"}


def test_start_is_a_menu_and_not_a_trading_command():
    """
    The regression this prevents: /start must NOT start the bot.

    It is the way an operator arrives at the control panel, so it is the single
    most likely place for an accidental trade. The menu's "START PAPER" BUTTON
    is what starts trading, and that is a different payload.
    """
    from autopaper.telegram.botapi import TEXT_COMMANDS
    from autopaper.telegram import ui

    assert TEXT_COMMANDS["/start"] == ui.CB_MENU
    assert TEXT_COMMANDS["/start"] != ui.CB_START, (
        "/start must not be wired to the action that starts trading"
    )


def test_an_unknown_text_command_yields_no_payload():
    assert payload_from_update(msg_update(ALLOWED, text="/drop_table"))[0] is None
    assert payload_from_update(msg_update(ALLOWED, text="hello"))[0] is None
    assert payload_from_update(msg_update(ALLOWED, text=""))[0] is None


def test_a_callback_with_no_data_yields_no_payload():
    assert payload_from_update(
        {"update_id": 1, "callback_query": {"id": "x"}}
    )[0] is None
    assert payload_from_update(
        {"update_id": 1, "callback_query": {"id": "x", "data": ""}}
    )[0] is None
    assert payload_from_update(
        {"update_id": 1, "callback_query": {"id": "x", "data": 42}}
    )[0] is None


def test_a_confirm_callback_payload_is_preserved_verbatim():
    """The token in a confirm payload must reach the handler unmangled."""
    data, _ = payload_from_update(
        cb_update(ALLOWED, data="confirm_real:AbC-123_x"))
    assert data == "confirm_real:AbC-123_x"


def test_a_malformed_update_is_refused_without_dispatch():
    c, rec = control()
    for junk in ("string", 42, None, [], {}):
        assert asyncio.run(c.handle_update(junk)) is False
    assert rec.seen == []
