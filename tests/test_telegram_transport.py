"""
Bot API transport: failure handling, bounded retries, and token hygiene.

The token is embedded in the request URL, and URLs leak into exception reprs,
request objects and log lines. So the assertions here are as much about what the
transport does NOT reveal as about what it sends.
"""


from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from autopaper.telegram.botapi import (
    API_HOST,
    BotApiError,
    BotApiForbidden,
    BotApiMalformed,
    BotApiNotConfigured,
    BotApiRetryAfter,
    BotApiTimeout,
    BotApiTransient,
    BotApiTransport,
    BotApiUnauthorized,
    _clean_keyboard,
)

pytestmark = pytest.mark.safety

TOKEN = "123456789:AAFAKEfakefakefakefakefakefakefakefake"
CHAT = "-1001234567890"


def make_client(handler, **kw) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kw)


def transport(handler, **kw) -> BotApiTransport:
    slept: list[float] = []
    t = BotApiTransport(
        TOKEN, chat_id=CHAT, client=make_client(handler),
        sleeper=lambda d: _record(slept, d), **kw
    )
    t.slept = slept          # type: ignore[attr-defined]
    return t


async def _record(store: list[float], delay: float) -> None:
    store.append(delay)


def ok(payload: dict) -> httpx.Response:
    return httpx.Response(200, json={"ok": True, "result": payload})


# ---------------------------------------------------------------------------
# success
# ---------------------------------------------------------------------------


def test_a_successful_call_returns_the_result():
    async def handler(request):
        assert request.url.host == API_HOST
        assert TOKEN in str(request.url)
        return ok({"message_id": 7})

    t = transport(handler)
    data = asyncio.run(t.call("sendMessage", {"chat_id": CHAT, "text": "hi"}))
    assert data["result"]["message_id"] == 7


def test_send_message_builds_the_expected_body():
    seen = {}

    async def handler(request):
        seen.update(json.loads(request.content))
        return ok({"message_id": 1})

    t = transport(handler)
    asyncio.run(t.send_message("hello", [[{"text": "a", "callback_data": "start"}]]))
    assert seen["chat_id"] == int(CHAT)
    assert seen["text"] == "hello"
    assert seen["reply_markup"] == {
        "inline_keyboard": [[{"text": "a", "callback_data": "start"}]]
    }


def test_send_message_omits_reply_markup_when_there_is_no_keyboard():
    seen = {}

    async def handler(request):
        seen.update(json.loads(request.content))
        return ok({"message_id": 1})

    t = transport(handler)
    asyncio.run(t.send_message("plain"))
    assert "reply_markup" not in seen


def test_get_updates_returns_a_list_and_tolerates_an_empty_result():
    async def handler(request):
        return ok([])

    t = transport(handler)
    assert asyncio.run(t.get_updates(None, 5)) == []


def test_get_updates_drops_non_object_entries():
    async def handler(request):
        return ok([{"update_id": 1}, "junk", None, {"update_id": 2}])

    t = transport(handler)
    got = asyncio.run(t.get_updates(None, 1))
    assert [u["update_id"] for u in got] == [1, 2]


def test_the_long_poll_timeout_is_bounded_at_50s():
    """Asking Telegram for more than 50s is rejected server-side."""
    seen = {}

    async def handler(request):
        seen.update(json.loads(request.content))
        return ok([])

    t = transport(handler)
    asyncio.run(t.get_updates(None, 9999))
    assert seen["timeout"] == 50


# ---------------------------------------------------------------------------
# timeouts
# ---------------------------------------------------------------------------


def test_a_timeout_is_translated_and_retried():
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        raise httpx.ReadTimeout("timed out")

    t = transport(handler, max_retries=2)
    with pytest.raises(BotApiTimeout):
        asyncio.run(t.call("sendMessage", {}))
    assert calls["n"] == 3, "initial attempt plus two retries"


def test_a_transient_timeout_recovers_within_the_retry_budget():
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("no route")
        return ok({"message_id": 2})

    t = transport(handler, max_retries=3)
    data = asyncio.run(t.call("sendMessage", {}))
    assert data["result"]["message_id"] == 2
    assert calls["n"] == 3


# ---------------------------------------------------------------------------
# HTTP errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code,exc", [
    (401, BotApiUnauthorized),
    (403, BotApiUnauthorized),
    (400, BotApiForbidden),
    (404, BotApiForbidden),
])
def test_permanent_http_errors_map_to_the_right_exception(code, exc):
    async def handler(request):
        return httpx.Response(code, json={"ok": False, "description": "nope"})

    t = transport(handler, max_retries=3)
    with pytest.raises(exc):
        asyncio.run(t.call("sendMessage", {}))


def test_a_4xx_is_not_retried():
    """Retrying a rejected credential just burns the rate limit."""
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        return httpx.Response(401, json={"ok": False, "description": "bad"})

    t = transport(handler, max_retries=5)
    with pytest.raises(BotApiUnauthorized):
        asyncio.run(t.call("sendMessage", {}))
    assert calls["n"] == 1, "a permanent error must be raised immediately"


def test_a_5xx_is_transient_and_retried():
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        return httpx.Response(503, json={"ok": False})

    t = transport(handler, max_retries=2)
    with pytest.raises(BotApiError):
        asyncio.run(t.call("sendMessage", {}))
    assert calls["n"] == 3


def test_ok_false_with_429_raises_retry_after():
    async def handler(request):
        return httpx.Response(200, json={
            "ok": False, "error_code": 429,
            "parameters": {"retry_after": 7},
            "description": "Too Many Requests",
        })

    t = transport(handler, max_retries=0)
    with pytest.raises(BotApiRetryAfter) as ei:
        asyncio.run(t.call("sendMessage", {}))
    assert ei.value.retry_after == 7.0


def test_rate_limit_backoff_honours_retry_after():
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={
                "ok": False, "error_code": 429,
                "parameters": {"retry_after": 3}, "description": "slow down",
            })
        return ok({"message_id": 3})

    t = transport(handler, max_retries=2, retry_backoff_s=0.1)
    data = asyncio.run(t.call("sendMessage", {}))
    assert data["result"]["message_id"] == 3
    assert t.slept and t.slept[0] >= 3.0, "must wait what Telegram asked for"


# ---------------------------------------------------------------------------
# malformed responses
# ---------------------------------------------------------------------------


def test_a_non_json_body_is_malformed():
    async def handler(request):
        return httpx.Response(200, content=b"<html>gateway error</html>")

    t = transport(handler, max_retries=2)
    with pytest.raises(BotApiMalformed):
        asyncio.run(t.call("sendMessage", {}))
    assert t.slept == [], "a malformed body is permanent, not retried"


def test_a_json_array_body_is_malformed():
    async def handler(request):
        return httpx.Response(200, json=[1, 2, 3])

    t = transport(handler)
    with pytest.raises(BotApiMalformed):
        asyncio.run(t.call("sendMessage", {}))


def test_a_get_updates_result_that_is_not_a_list_is_malformed():
    async def handler(request):
        return ok({"unexpected": "object"})

    t = transport(handler)
    with pytest.raises(BotApiMalformed):
        asyncio.run(t.get_updates(None, 1))


# ---------------------------------------------------------------------------
# bounded retry budget
# ---------------------------------------------------------------------------


def test_retries_are_bounded():
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        raise httpx.ConnectError("down")

    t = transport(handler, max_retries=2)
    with pytest.raises(BotApiTransient):
        asyncio.run(t.call("sendMessage", {}))
    assert calls["n"] == 3, "1 initial + 2 retries, then give up"


def test_backoff_grows_and_is_capped():
    async def handler(request):
        raise httpx.ConnectError("down")

    t = transport(handler, max_retries=5, retry_backoff_s=1.0,
                  retry_backoff_max_s=4.0)
    with pytest.raises(BotApiTransient):
        asyncio.run(t.call("sendMessage", {}))
    assert t.slept == [1.0, 2.0, 4.0, 4.0, 4.0], t.slept


def test_zero_retries_means_one_attempt():
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        raise httpx.ConnectError("down")

    t = transport(handler, max_retries=0)
    with pytest.raises(BotApiTransient):
        asyncio.run(t.call("sendMessage", {}))
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# token hygiene
# ---------------------------------------------------------------------------


def test_no_raised_exception_contains_the_token():
    """The token is in the URL, so it must not survive into any error."""
    cases = [
        lambda r: httpx.Response(401, json={"ok": False, "description": "no"}),
        lambda r: httpx.Response(400, json={"ok": False, "description": "no"}),
        lambda r: httpx.Response(500, json={"ok": False}),
        lambda r: httpx.Response(200, content=b"not json"),
        lambda r: (_ for _ in ()).throw(httpx.ConnectError("boom")),
        lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("slow")),
    ]
    for make_response in cases:
        async def handler(request, _f=make_response):
            return _f(request)

        t = transport(handler, max_retries=1)
        with pytest.raises(BotApiError) as ei:
            asyncio.run(t.call("sendMessage", {}))
        blob = f"{ei.value!r} {ei.value}"
        assert TOKEN not in blob, f"token leaked via {type(ei.value).__name__}: {blob}"
        assert t.status.last_error
        assert TOKEN not in t.status.last_error


def test_a_4xx_description_is_never_included_at_all():
    """
    A 4xx is answered from the status code only.

    The body is not even read, so a token echoed by Telegram cannot reach an
    exception. Strictly safer than redacting.
    """
    async def handler(request):
        return httpx.Response(400, json={
            "ok": False, "description": f"Bad Request: token {TOKEN} rejected",
        })

    t = transport(handler)
    with pytest.raises(BotApiForbidden) as ei:
        asyncio.run(t.call("sendMessage", {}))
    assert TOKEN not in str(ei.value)
    assert "rejected" not in str(ei.value)


def test_a_description_echoing_the_token_is_redacted():
    """Where a description IS read (200 + ok:false), it is redacted."""
    async def handler(request):
        return httpx.Response(200, json={
            "ok": False, "error_code": 400,
            "description": f"Bad Request: token {TOKEN} rejected",
        })

    t = transport(handler)
    with pytest.raises(BotApiForbidden) as ei:
        asyncio.run(t.call("sendMessage", {}))
    assert TOKEN not in str(ei.value)
    assert "<redacted>" in str(ei.value)


def test_the_api_host_is_a_constant_not_a_parameter():
    """
    A configurable base URL could send the token to a third party.

    So the host is not accepted from config, and the URL is built from a module
    constant. Asserted by checking the transport has no host/base_url knob.
    """
    import inspect

    params = set(inspect.signature(BotApiTransport.__init__).parameters)
    assert "base_url" not in params
    assert "host" not in params
    assert "api_base" not in params
    t = transport(lambda r: ok({}))
    assert t._url("sendMessage").startswith(f"https://{API_HOST}/bot{TOKEN}/")


def test_redirects_are_not_followed():
    """A redirect could carry the token to another host."""
    async def handler(request):
        return ok({})

    t = BotApiTransport(TOKEN, client=make_client(handler))
    # the default client this module builds must not follow redirects
    import inspect
    src = inspect.getsource(BotApiTransport._ensure_client)
    assert "follow_redirects=False" in src


# ---------------------------------------------------------------------------
# not configured
# ---------------------------------------------------------------------------


def test_without_a_token_every_call_refuses():
    t = BotApiTransport(None)
    with pytest.raises(BotApiNotConfigured):
        asyncio.run(t.call("sendMessage", {}))
    with pytest.raises(BotApiNotConfigured):
        asyncio.run(t.get_updates(None, 1))


def test_without_a_chat_sending_refuses():
    t = BotApiTransport(TOKEN, chat_id=None,
                        client=make_client(lambda r: ok({})))
    with pytest.raises(BotApiNotConfigured):
        asyncio.run(t.send_message("hi"))


def test_a_malformed_chat_id_refuses_to_send():
    t = BotApiTransport(TOKEN, chat_id="@notanumber",
                        client=make_client(lambda r: ok({})))
    with pytest.raises(BotApiNotConfigured):
        asyncio.run(t.send_message("hi"))


# ---------------------------------------------------------------------------
# keyboard sanitisation
# ---------------------------------------------------------------------------


def test_keyboard_cleaning_keeps_only_text_and_callback_data():
    kb = [[{"text": "a", "callback_data": "start", "url": "https://evil",
            "style": "danger"}]]
    assert _clean_keyboard(kb) == [[{"text": "a", "callback_data": "start"}]]


def test_keyboard_cleaning_drops_rows_with_no_callback():
    kb = [[{"text": "no callback"}], [{"text": "ok", "callback_data": "stop"}]]
    assert _clean_keyboard(kb) == [[{"text": "ok", "callback_data": "stop"}]]


def test_keyboard_cleaning_tolerates_junk():
    assert _clean_keyboard(None) == []
    assert _clean_keyboard([]) == []
    assert _clean_keyboard([[]]) == []
