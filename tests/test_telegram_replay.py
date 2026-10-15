"""
Replay protection.

Telegram promises at-most-once delivery, and the runtime must not rely on it: a
retried poll, a duplicated relay, or a hostile client can present the same
payload repeatedly. Two distinct duplications matter, and both are covered:

* the same ``update_id`` delivered twice;
* two DIFFERENT update ids carrying the same ``callback_query.id`` - one
  operator intent, redelivered.

The second is the subtler one, because an update-id-only guard would let it
through and re-run the transition.
"""

from __future__ import annotations

import asyncio

import pytest

from autopaper.telegram import botapi
from autopaper.telegram.botapi import BotControl, ReplayGuard

ALLOWED = -1001234567890


def msg_update(chat, uid, text="/status"):
    return {"update_id": uid, "message": {"text": text, "chat": {"id": chat}}}


def cb_update(chat, uid, data="start", cb_id="cb-1"):
    return {"update_id": uid,
            "callback_query": {"id": cb_id, "data": data,
                               "message": {"chat": {"id": chat}}}}


class Recorder:
    def __init__(self):
        self.seen: list[str] = []

    async def __call__(self, data):
        self.seen.append(data)


def control(rec=None, **kw):
    t = botapi.BotApiTransport("TOK", chat_id=str(ALLOWED))
    rec = rec or Recorder()
    return BotControl(transport=t, handle_button=rec,
                      allowed_chat=ALLOWED, **kw), rec


# ---------------------------------------------------------------------------
# the guard itself
# ---------------------------------------------------------------------------


def test_a_first_sighting_is_claimed():
    g = ReplayGuard()
    assert g.claim("update", 1) is True


def test_a_second_sighting_is_refused():
    g = ReplayGuard()
    assert g.claim("update", 1) is True
    assert g.claim("update", 1) is False
    assert g.claim("update", 1) is False


def test_different_kinds_do_not_collide():
    g = ReplayGuard()
    assert g.claim("update", 7) is True
    assert g.claim("callback", 7) is True, "same id, different namespace"


def test_an_identueless_claim_is_refused():
    """
    An update with no usable id cannot be made idempotent, so it is refused
    rather than allowed through every time.
    """
    g = ReplayGuard()
    assert g.claim("update", None) is False
    assert g.claim("update", True) is False


def test_a_string_id_still_works_as_an_opaque_key():
    g = ReplayGuard()
    assert g.claim("callback", "abc") is True
    assert g.claim("callback", "abc") is False


def test_the_guard_is_bounded():
    g = ReplayGuard(capacity=10)
    for i in range(100):
        g.claim("update", i)
    assert len(g) <= 10, f"grew to {len(g)}"
    assert g.stats()["capacity"] == 10


def test_bounding_evicts_the_oldest():
    g = ReplayGuard(capacity=4)
    for i in range(4):
        g.claim("update", i)
    g.claim("update", 99)
    assert g.claim("update", 0) is True, "0 was evicted, so it is new again"
    assert g.claim("update", 99) is False, "99 is still remembered"


# ---------------------------------------------------------------------------
# duplicate update ids
# ---------------------------------------------------------------------------


def test_a_duplicate_update_id_is_dispatched_once():
    c, rec = control()
    u = msg_update(ALLOWED, 1)
    assert asyncio.run(c.handle_update(u)) is True
    assert asyncio.run(c.handle_update(dict(u))) is False
    assert rec.seen == ["stats"], rec.seen
    assert c.status_dict()["updates_refused_replay"] == 1


def test_ten_duplicates_execute_once():
    c, rec = control()
    for _ in range(10):
        asyncio.run(c.handle_update(msg_update(ALLOWED, 1, "/pause")))
    assert rec.seen == ["pause"]


def test_distinct_ids_all_execute():
    c, rec = control()
    for uid in range(1, 6):
        asyncio.run(c.handle_update(msg_update(ALLOWED, uid, "/status")))
    assert rec.seen == ["stats"] * 5


# ---------------------------------------------------------------------------
# duplicate callback ids under different update ids
# ---------------------------------------------------------------------------


def test_the_same_callback_id_under_a_new_update_id_is_still_refused():
    """
    The subtle case: an update-id-only guard would dispatch this twice.

    Telegram can redeliver a callback under a fresh update_id. One operator
    press is one intent, so the callback id is claimed independently.
    """
    c, rec = control()
    assert asyncio.run(c.handle_update(cb_update(ALLOWED, 1, "stop", "cb-7"))) is True
    assert asyncio.run(c.handle_update(cb_update(ALLOWED, 2, "stop", "cb-7"))) is False
    assert rec.seen == ["stop"]


def test_a_replayed_confirm_callback_cannot_re_arm():
    """
    The dangerous replay: a real confirm-real press, delivered twice.

    The first is refused by the handler (no live token), and the replay must not
    reach the handler at all.
    """
    c, rec = control()
    payload = "confirm_real:SOMETOKEN"
    asyncio.run(c.handle_update(cb_update(ALLOWED, 1, payload, "cb-r")))
    asyncio.run(c.handle_update(cb_update(ALLOWED, 2, payload, "cb-r")))
    assert rec.seen == [payload], "exactly one delivery"


def test_a_replayed_start_callback_does_not_restart():
    c, rec = control()
    for uid in (1, 2, 3):
        asyncio.run(c.handle_update(cb_update(ALLOWED, uid, "start", "cb-s")))
    assert rec.seen == ["start"]


def test_a_callback_with_no_id_falls_back_to_the_update_id():
    """
    Without a callback id, the update-id claim is the only idempotency
    available. It is still claimed, so a redelivery is refused.
    """
    c, rec = control()
    u = {"update_id": 1, "callback_query": {
        "data": "start", "message": {"chat": {"id": ALLOWED}}}}
    assert asyncio.run(c.handle_update(dict(u))) is True
    assert asyncio.run(c.handle_update(dict(u))) is False
    assert rec.seen == ["start"]


# ---------------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------------


def test_a_replay_is_refused_before_the_payload_is_parsed():
    """
    Authorization and parsing happen AFTER the replay claim, so a replayed
    payload is never interpreted at all.
    """
    seen_payloads = []

    t = botapi.BotApiTransport("TOK", chat_id=str(ALLOWED))
    c = BotControl(transport=t, handle_button=Recorder(), allowed_chat=ALLOWED)
    real = botapi.payload_from_update

    def spy(update):
        seen_payloads.append(1)
        return real(update)

    botapi.payload_from_update = spy
    try:
        asyncio.run(c.handle_update(msg_update(ALLOWED, 1)))
        asyncio.run(c.handle_update(msg_update(ALLOWED, 1)))
    finally:
        botapi.payload_from_update = real
    assert len(seen_payloads) == 1, "the replay must not be parsed"


def test_the_claim_happens_before_dispatch_so_reentrancy_is_caught():
    """
    A handler that re-delivers its own update must not cause a second run.
    """
    class Reentrant:
        def __init__(self):
            self.seen = []
            self.c = None

        async def __call__(self, data):
            self.seen.append(data)
            await self.c.handle_update(msg_update(ALLOWED, 1))

    rec = Reentrant()
    c, _ = control(rec)
    rec.c = c
    assert asyncio.run(c.handle_update(msg_update(ALLOWED, 1))) is True
    assert rec.seen == ["stats"], rec.seen


# ---------------------------------------------------------------------------
# malformed
# ---------------------------------------------------------------------------


def test_a_malformed_update_is_refused_and_not_dispatched():
    c, rec = control()
    for junk in ("str", 42, None, [], {}, {"no": "update_id"},
                 {"update_id": None, "message": {"chat": {"id": ALLOWED}}}):
        assert asyncio.run(c.handle_update(junk)) is False
    assert rec.seen == []


def test_an_update_with_no_dispatchable_payload_is_refused():
    c, rec = control()
    assert asyncio.run(c.handle_update(msg_update(ALLOWED, 1, "/nonsense"))) is False
    assert rec.seen == []
    assert c.status_dict()["updates_malformed"] >= 1
