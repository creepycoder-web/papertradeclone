"""
Telegram Bot API transport: the operator control and notification channel.

SCOPE
-----
This module is a TRANSPORT. It moves messages. It does not decide anything.

Every inbound command is handed to the EXISTING
:meth:`autopaper.engine.PaperTrader.handle_button`, which delegates to the
EXISTING :class:`autopaper.telegram.actions.ButtonHandler`. There is no second
command path, no shortcut into the runtime, and no state transition authored
here. In particular this module cannot arm REAL: the only path to REAL is
``RuntimeState.arm_real()``, reached only from the button handler's
confirm-real branch, which requires a valid unexpired token AND a passing
preflight.

WHY THE TOKEN HANDLING IS DELICATE
----------------------------------
The Bot API address is ``https://api.telegram.org/bot<TOKEN>/<METHOD>``. The
token is therefore a bearer credential embedded in a URL, and URLs leak: they
appear in ``httpx`` exception reprs, in request objects, and in anything that
formats a failure. So:

* the token never appears in an exception raised by this module;
* the token never appears in a log line;
* the token never appears in a status/diagnostic payload;
* httpx errors are translated into sanitised errors whose message is built from
  the METHOD NAME and status code only.

The authorization boundary is the chat allow-list, not the token and not a
username: an update is only ever dispatched when its numeric chat id equals the
configured one.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence

import httpx

# The project's tagged logger, so these lines carry a tag like every other line
# in the package and are filtered by the same configuration.
from ..logging_setup import log

#: The only host this module will ever talk to. An absolute URL to anywhere else
#: is refused, so a mis-set base URL cannot exfiltrate the token.
API_HOST = "api.telegram.org"

#: Telegram caps a long-poll at 50s. Anything above that is rejected server-side,
#: and asking for more only invites timeouts.
MAX_POLL_TIMEOUT_S = 50.0


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


def _strip_html(text: str) -> str:
    """
    Flatten our own formatting to readable text.

    Removes the tags this UI emits and unescapes the entities, so a payload
    that Telegram rejected as HTML still arrives as a legible message rather
    than raw markup.
    """
    import html as _html
    import re as _re

    out = _re.sub(r"<br\s*/?>", "\n", text, flags=_re.I)
    out = _re.sub(r"</?(?:b|i|u|s|code|pre|a)\b[^>]*>", "", out, flags=_re.I)
    return _html.unescape(out)


class BotApiError(RuntimeError):
    """Base for every transport failure. Never carries the token."""


class BotApiNotConfigured(BotApiError):
    """No token, or no allow-listed chat. Control stays unavailable."""


class BotApiUnauthorized(BotApiError):
    """Telegram rejected the credential. Retrying will not help."""


class BotApiForbidden(BotApiError):
    """Telegram refused the request as malformed or not permitted."""


class BotApiRetryAfter(BotApiError):
    """Telegram asked us to wait. Carries the delay, never the token."""

    def __init__(self, retry_after: float, *, method: str = "") -> None:
        super().__init__(
            f"Telegram rate-limited {method or 'a request'}; "
            f"retry after {retry_after:.1f}s"
        )
        self.retry_after = float(retry_after)


class BotApiTimeout(BotApiError):
    """The request exceeded its bounded timeout."""


class BotApiTransient(BotApiError):
    """A network-level failure that is worth a bounded retry."""


class BotApiMalformed(BotApiError):
    """Telegram replied with something that is not a usable JSON object."""


class UnauthorizedChat(BotApiError):
    """The update came from a chat that is not allow-listed. Never dispatched."""


class ReplayedUpdate(BotApiError):
    """A duplicate update or callback id. Refused before any state transition."""


# ---------------------------------------------------------------------------
# authorization
# ---------------------------------------------------------------------------


def normalise_chat_id(value: Any) -> Optional[int]:
    """
    Parse a Telegram chat id to its canonical integer form.

    Telegram chat ids are integers and may be negative (supergroups/channels).
    They are compared NUMERICALLY, never as text, so ``"@name"``, ``" 123 "`` and
    ``"+123"`` cannot be used to smuggle a different identity past the
    allow-list. ``@username`` text is rejected outright: usernames are
    reassignable and are not an identity.

    Returns ``None`` for anything that is not a plain integer, which the caller
    treats as a refusal.
    """
    if isinstance(value, bool):          # bool is an int subclass; not a chat
        return None
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    # no leading +, no leading zeros, no spaces inside, no @, no username
    if s.startswith("@") or s.startswith("+") or " " in s:
        return None
    digits = s[1:] if s.startswith("-") else s
    # ``str.isdigit`` is NOT enough: it is true for Arabic-Indic and other
    # non-ASCII digits, and ``int()`` parses them, so U+0661 U+0662 U+0663 would
    # alias chat 123. Only ASCII 0-9 may appear.
    if not digits or not all("0" <= c <= "9" for c in digits):
        return None
    if len(s.lstrip("-")) > 1 and s.lstrip("-").startswith("0"):
        return None                      # "0123" is not a canonical id
    try:
        return int(s)
    except ValueError:
        return None


def chat_is_authorized(update: Mapping[str, Any], allowed_chat: Any) -> bool:
    """
    Is this update from the one allow-listed chat?

    The chat id is read from the message/callback container, converted with
    :func:`normalise_chat_id`, and compared to the configured id numerically.
    A missing, malformed or foreign id is refused. There is no fallback that
    admits an update whose chat cannot be determined.
    """
    want = normalise_chat_id(allowed_chat)
    if want is None:
        return False                     # no valid allow-list -> nothing allowed
    got = normalise_chat_id(extract_chat_id(update))
    if got is None:
        return False
    return got == want


def extract_chat_id(update: Mapping[str, Any]) -> Any:
    """
    Pull the chat id out of whichever container the update uses.

    ``callback_query`` and messages nest it differently, and getting this wrong
    is how an update ends up authorized by accident, so the lookup is explicit
    and returns ``None`` rather than guessing.
    """
    if not isinstance(update, Mapping):
        return None
    cq = update.get("callback_query")
    if isinstance(cq, Mapping):
        msg = cq.get("message")
        if isinstance(msg, Mapping):
            chat = msg.get("chat")
            if isinstance(chat, Mapping) and "id" in chat:
                return chat.get("id")
        # a callback with no resolvable chat is NOT authorized
        return None
    for key in ("message", "edited_message", "channel_post"):
        msg = update.get(key)
        if isinstance(msg, Mapping):
            chat = msg.get("chat")
            if isinstance(chat, Mapping) and "id" in chat:
                return chat.get("id")
    return None


# ---------------------------------------------------------------------------
# replay protection
# ---------------------------------------------------------------------------


class ReplayGuard:
    """
    Idempotency for inbound updates and callbacks.

    Telegram guarantees at-most-once delivery of an update, and the runtime must
    not depend on that: a retried poll, a duplicated webhook relay, or a
    malicious client can present the same payload repeatedly. A second START
    would otherwise re-run the transition, and a second confirm-real would
    re-present an already-consumed token.

    Bounded by insertion order, so a long-lived process cannot grow it without
    limit, and it records the id on FIRST sight - before dispatch - so a
    re-entrant delivery during handling is also caught.
    """

    def __init__(self, capacity: int = 4096) -> None:
        self.capacity = max(1, int(capacity))
        self._seen: dict[str, float] = {}
        self._seq = 0

    def _remember(self, key: str) -> None:
        self._seen[key] = 0.0
        if len(self._seen) > self.capacity:
            # drop the oldest insertion; dicts preserve insertion order
            for old in list(self._seen)[: len(self._seen) - self.capacity]:
                self._seen.pop(old, None)

    def claim(self, kind: str, ident: Any) -> bool:
        """
        Claim ``ident`` for ``kind``. True if this is the first time seen.

        A missing or unhashable id is refused (returns False) rather than
        allowed: an unidentifiable update cannot be made idempotent, so it is
        not dispatched.
        """
        if ident is None or isinstance(ident, bool):
            return False
        try:
            key = f"{kind}:{int(ident)}"
        except (TypeError, ValueError):
            # a non-integer id is still usable as an opaque string key
            key = f"{kind}:{ident!r}"
        if key in self._seen:
            return False
        self._remember(key)
        return True

    def __len__(self) -> int:
        return len(self._seen)

    def stats(self) -> dict[str, int]:
        return {"remembered": len(self._seen), "capacity": self.capacity}


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------


def _redact(text: str, token: str) -> str:
    """Remove the token from any string before it can be logged or raised."""
    if not token:
        return text
    return text.replace(token, "<redacted>")


@dataclass
class BotStatus:
    """Observability. Booleans and labels only - never the token."""

    transport_initialized: bool = False
    token_configured: bool = False
    chat_configured: bool = False
    polling: bool = False
    updates_seen: int = 0
    updates_authorized: int = 0
    updates_refused_chat: int = 0
    updates_refused_replay: int = 0
    updates_malformed: int = 0
    last_error: str = ""
    last_error_at: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["last_error"] = self.last_error[:200]
        return d


class BotApiTransport:
    """
    A narrow async Bot API client.

    Deliberately small: only the three methods the runtime needs
    (``getUpdates``, ``sendMessage``, ``editMessageText``), bounded timeouts,
    bounded retries with exponential backoff, and no generic framework around
    them. Every failure is translated into a sanitised :class:`BotApiError`.
    """

    def __init__(
        self,
        token: Optional[str],
        *,
        chat_id: Optional[str] = None,
        request_timeout_s: float = 30.0,
        max_retries: int = 3,
        retry_backoff_s: float = 0.5,
        retry_backoff_max_s: float = 8.0,
        client: Optional[httpx.AsyncClient] = None,
        sleeper: Optional[Callable[[float], Awaitable[None]]] = None,
    ) -> None:
        self._token = token or None
        self._chat_id = chat_id
        self.request_timeout_s = float(request_timeout_s)
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_s = float(retry_backoff_s)
        self.retry_backoff_max_s = float(retry_backoff_max_s)
        # injected so tests never touch the network and never really sleep
        self._client = client
        self._owns_client = client is None
        self._sleep = sleeper or asyncio.sleep
        self.status = BotStatus(
            transport_initialized=True,
            token_configured=bool(token),
            chat_configured=bool(chat_id),
        )

    # -- lifecycle ---------------------------------------------------------

    @property
    def configured(self) -> bool:
        return bool(self._token)

    def _url(self, method: str) -> str:
        """
        Build the API URL. The token is here and ONLY here.

        The host is a constant, not a parameter, so the token cannot be
        redirected to a third party by configuration.
        """
        if not self._token:
            raise BotApiNotConfigured("no bot token configured")
        return f"https://{API_HOST}/bot{self._token}/{method}"

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.request_timeout_s),
                follow_redirects=False,   # a redirect could carry the token off-host
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None

    # -- error translation -------------------------------------------------

    def _sanitise(self, exc: BaseException, method: str) -> BotApiError:
        """
        Turn any library exception into a token-free BotApiError.

        httpx exceptions embed the request URL, which embeds the token, so the
        original message is never propagated. Only the method name and a
        classification survive.
        """
        if isinstance(exc, BotApiError):
            return exc
        if isinstance(exc, httpx.TimeoutException):
            return BotApiTimeout(f"{method} timed out after "
                                 f"{self.request_timeout_s:.0f}s")
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            if code in (401, 403):
                return BotApiUnauthorized(
                    f"{method} rejected by Telegram (HTTP {code}); check the "
                    f"bot token"
                )
            if code == 429:
                ra = _retry_after_from(exc.response)
                return BotApiRetryAfter(ra, method=method)
            if 400 <= code < 500:
                return BotApiForbidden(f"{method} refused (HTTP {code})")
            return BotApiTransient(f"{method} failed (HTTP {code})")
        if isinstance(exc, (httpx.TransportError, httpx.HTTPError)):
            return BotApiTransient(f"{method} network failure "
                                   f"({type(exc).__name__})")
        return BotApiTransient(f"{method} failed ({type(exc).__name__})")

    def _note_error(self, exc: BaseException) -> None:
        self.status.last_error = _redact(str(exc), self._token or "")
        self.status.last_error_at = 0.0

    # -- the call ----------------------------------------------------------

    async def call(
        self,
        method: str,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        One Bot API call, with bounded retries for transient failures.

        Permanent failures (401/403/400) are raised immediately - retrying a
        bad credential just burns the rate limit. Transient failures and 429 are
        retried up to ``max_retries`` with exponential backoff, honouring
        ``retry_after`` when Telegram supplies it.
        """
        if not self.configured:
            raise BotApiNotConfigured("no bot token configured")
        client = await self._ensure_client()
        url = self._url(method)
        body = dict(payload or {})
        attempt = 0
        delay = self.retry_backoff_s

        while True:
            attempt += 1
            try:
                response = await client.post(
                    url, json=body,
                    timeout=httpx.Timeout(timeout or self.request_timeout_s),
                )
                if response.status_code >= 400:
                    # raise_for_status would build an exception holding the URL
                    err = self._sanitise(
                        httpx.HTTPStatusError(
                            f"HTTP {response.status_code}", request=response.request,
                            response=response,
                        ),
                        method,
                    )
                else:
                    try:
                        data = response.json()
                    except Exception as exc:  # noqa: BLE001
                        # Note: the cause is named in the MESSAGE, not assigned
                        # to __cause__. Assigning a string there raises TypeError
                        # ("cause must be None or derive from BaseException"),
                        # which the outer handler would then misclassify as a
                        # transient failure - turning a permanently broken
                        # endpoint into a retry storm.
                        err = BotApiMalformed(
                            f"{method} returned a non-JSON body "
                            f"({type(exc).__name__})"
                        )
                    else:
                        if not isinstance(data, dict):
                            raise BotApiMalformed(
                                f"{method} returned {type(data).__name__}, "
                                f"not an object"
                            )
                        if data.get("ok") is False:
                            desc = str(data.get("description") or "no description")
                            code = data.get("error_code")
                            if code == 429:
                                err = BotApiRetryAfter(
                                    float(data.get("parameters", {}).get(
                                        "retry_after", 1) or 1), method=method)
                            elif code in (401, 403):
                                err = BotApiUnauthorized(
                                    f"{method} rejected: {_redact(desc, self._token or '')}"
                                )
                            else:
                                err = BotApiForbidden(
                                    f"{method} refused: {_redact(desc, self._token or '')}"
                                )
                        else:
                            return data
            except BotApiError as exc:
                err = exc
            except Exception as exc:  # noqa: BLE001 - anything else is transient
                err = self._sanitise(exc, method)

            self._note_error(err)
            permanent = isinstance(
                err, (BotApiUnauthorized, BotApiForbidden, BotApiNotConfigured,
                      BotApiMalformed)
            )
            if permanent or attempt > self.max_retries:
                raise err
            wait = delay
            if isinstance(err, BotApiRetryAfter):
                wait = max(wait, err.retry_after)
            await self._sleep(min(wait, self.retry_backoff_max_s))
            delay = min(delay * 2.0, self.retry_backoff_max_s)

    # -- the three methods we need ----------------------------------------

    async def get_updates(
        self, offset: Optional[int] = None, timeout_s: float = 20.0
    ) -> list[dict[str, Any]]:
        """One long-poll. Returns [] on a clean empty poll."""
        poll = max(0.0, min(float(timeout_s), MAX_POLL_TIMEOUT_S))
        body: dict[str, Any] = {
            "timeout": int(poll),
            # asking for no more than we can hold keeps memory bounded
            "allowed_updates": ["message", "callback_query"],
        }
        if offset is not None:
            body["offset"] = int(offset)
        # the HTTP timeout must outlive the long-poll or a healthy quiet chat
        # would be aborted as a client timeout
        data = await self.call(
            "getUpdates", body, timeout=max(poll + 10.0, self.request_timeout_s)
        )
        result = data.get("result")
        if result is None:
            return []
        if not isinstance(result, list):
            raise BotApiMalformed("getUpdates result was not a list")
        return [u for u in result if isinstance(u, Mapping)]

    async def send_message(
        self,
        text: str,
        keyboard: Optional[Sequence[Sequence[Mapping[str, str]]]] = None,
        chat_id: Any = None,
    ) -> dict[str, Any]:
        """
        Send a message, optionally with the existing inline keyboard.

        ``keyboard`` is the SAME shape ``ui.main_keyboard`` already produces, so
        the operator UI needed no change to be deliverable.
        """
        target = chat_id if chat_id is not None else self._chat_id
        cid = normalise_chat_id(target)
        if cid is None:
            raise BotApiNotConfigured("no valid chat id to send to")
        body: dict[str, Any] = {
            "chat_id": cid,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        if keyboard:
            body["reply_markup"] = {"inline_keyboard": _clean_keyboard(keyboard)}
        try:
            return await self.call("sendMessage", body)
        except BotApiForbidden:
            # A payload Telegram cannot PARSE is refused as a 400, which the
            # generic 4xx branch labels ``BotApiForbidden`` -- a name that points
            # at the bot token rather than at the markup. One token symbol
            # containing ``&`` or ``<`` is enough to lose the whole alert.
            #
            # So fall back to PLAIN TEXT once. The operator loses formatting,
            # not the message; a dropped alert is far worse than ugly text.
            if "<" not in text:
                raise
            plain = {k: v for k, v in body.items() if k != "parse_mode"}
            plain["text"] = _strip_html(text)
            return await self.call("sendMessage", plain)

    async def edit_message_text(
        self,
        message_id: int,
        text: str,
        keyboard: Optional[Sequence[Sequence[Mapping[str, str]]]] = None,
        chat_id: Any = None,
    ) -> dict[str, Any]:
        """Edit a previously sent message, so replies do not spam the chat."""
        target = chat_id if chat_id is not None else self._chat_id
        cid = normalise_chat_id(target)
        if cid is None:
            raise BotApiNotConfigured("no valid chat id to send to")
        body: dict[str, Any] = {
            "chat_id": cid,
            "message_id": int(message_id),
            "text": text,
            "parse_mode": "HTML",
        }
        if keyboard:
            body["reply_markup"] = {"inline_keyboard": _clean_keyboard(keyboard)}
        return await self.call("editMessageText", body)


def _retry_after_from(response: httpx.Response) -> float:
    """Read Retry-After / parameters.retry_after, defaulting to 1s."""
    try:
        data = response.json()
        if isinstance(data, Mapping):
            params = data.get("parameters")
            if isinstance(params, Mapping) and params.get("retry_after") is not None:
                return float(params["retry_after"])
    except Exception:  # noqa: BLE001
        pass
    try:
        return float(response.headers.get("Retry-After", 1) or 1)
    except (TypeError, ValueError):
        return 1.0


def _clean_keyboard(
    keyboard: Iterable[Iterable[Mapping[str, str]]],
) -> list[list[dict[str, str]]]:
    """
    Keep only the two keys Telegram accepts.

    Anything else the UI attached is dropped rather than forwarded, so a stray
    key cannot turn into an API rejection at send time.
    """
    out: list[list[dict[str, str]]] = []
    for row in keyboard or []:
        clean = [
            {"text": str(b.get("text", "")), "callback_data": str(b.get("callback_data", ""))}
            for b in row
            if isinstance(b, Mapping) and b.get("callback_data")
        ]
        if clean:
            out.append(clean)
    return out


# ---------------------------------------------------------------------------
# command routing
# ---------------------------------------------------------------------------

#: Text commands, mapped onto the EXISTING callback vocabulary. This is a
#: translation layer, not a second command set: every value on the right is a
#: constant the existing ButtonHandler already dispatches on.
#: The ONLY commands the bot accepts from an operator.
#:
#: Reduced from eleven. The removed ones were per-alert, per-analysis and
#: per-rejection surface that produced "watching"/"reject" spam and, worse, put
#: bot messaging in the middle of the trading path. Internal callback_data values
#: (CB_START, CB_STOP, the REAL confirmation pair) remain routable because the
#: dashboard keyboard and the audited REAL flow still use them, but they are not
#: reachable as typed commands.
TEXT_COMMANDS: dict[str, str] = {
    # /start shows the operator menu and NOTHING else. It is deliberately NOT
    # mapped to "start": the menu's "START PAPER" BUTTON is what starts
    # trading, and a command that started the bot on sight would be the most
    # dangerous possible way to arrive at the control panel.
    "/start": "menu",
    "/status": "stats",
    "/balance": "balance",
    "/pause": "pause",
    "/resume": "resume",
    "/kill": "kill",
    # Typed aliases for the two read-only position views. They dispatch onto the
    # SAME callbacks the buttons use - a translation layer, not a second command
    # set - and neither can place an order.
    "/open": "open",
    "/closed": "closed",
    # Typed alias for the take-profit selector's submenu. Dispatches onto the
    # SAME callback the TAKE PROFIT button carries, so a typed command and a
    # button press are the same action rather than two implementations.
    "/tp": "take_profit",
}

#: The typed ``/tp <value>`` argument prefix, and why it is a constant rather
#: than free text. The take-profit level is a NUMBER with a bounded vocabulary,
#: so the transport accepts AT MOST ONE extra token after the command and hands
#: the result to the SAME validating handler the keyboard buttons use. Anything
#: that is not inside the take-profit payload namespace is dropped, which is why
#: arbitrary operator text can never reach a callback that is not one of these.
TEXT_ARG_COMMAND = "/tp"
TAKE_PROFIT_TEXT_PREFIX = "take_profit:"


def payload_from_update(update: Mapping[str, Any]) -> tuple[Optional[str], Any]:
    """
    Extract ``(callback_data, update_id)`` from an update.

    Returns ``(None, update_id)`` for anything with no dispatchable payload -
    including a malformed callback whose data is not a string - so an
    unparseable payload fails closed instead of reaching the handler.
    """
    if not isinstance(update, Mapping):
        return (None, None)
    update_id = update.get("update_id")
    cq = update.get("callback_query")
    if isinstance(cq, Mapping):
        data = cq.get("data")
        if isinstance(data, str) and data.strip():
            return (data.strip(), update_id)
        # a callback_query with no usable data is refused
        return (None, update_id)
    for key in ("message", "edited_message"):
        msg = update.get(key)
        if isinstance(msg, Mapping):
            text = msg.get("text")
            if isinstance(text, str):
                token = text.strip().split()[0].lower() if text.strip() else ""
                if token == TEXT_ARG_COMMAND:
                    # ``/tp <value>`` becomes ``take_profit:<value>`` and is
                    # dispatched to the SAME validating handler the keyboard
                    # buttons use, so an invalid value is refused by the same
                    # code either way. Only the first token after the command
                    # is read, and only if it names the take-profit namespace;
                    # anything else is dropped rather than reaching a handler.
                    rest = text.strip().split()[1:]
                    if rest:
                        candidate = rest[0].lstrip("+").rstrip("%").strip()
                        if candidate.replace(".", "", 1).isdigit():
                            return (f"{TAKE_PROFIT_TEXT_PREFIX}{candidate}",
                                    update_id)
                        if candidate.upper() in {"OFF", "NONE", "DISABLED"}:
                            return (f"{TAKE_PROFIT_TEXT_PREFIX}off",
                                    update_id)
                    return ("take_profit", update_id)
                return (TEXT_COMMANDS.get(token), update_id)
    return (None, update_id)


#: What the engine-side handler is called. Injected so the transport can be
#: tested without an engine, and so the wiring is visible in one place.
HandleButton = Callable[[str], Awaitable[Any]]


@dataclass
class BotControl:
    """
    The inbound half: poll, authorize, de-duplicate, dispatch.

    Dispatch is delegated to the injected ``handle_button`` - which in production
    is ``PaperTrader.handle_button``, the existing entry point into
    ``ButtonHandler``. This class authors no state transition of its own.
    """

    transport: BotApiTransport
    handle_button: HandleButton
    allowed_chat: Any = None
    guard: ReplayGuard = field(default_factory=ReplayGuard)
    #: Optional hook so the runtime can record a refused update.
    on_refused: Optional[Callable[[str, Mapping[str, Any]], None]] = None
    _offset: Optional[int] = None
    _status: BotStatus = field(default_factory=BotStatus)

    def __post_init__(self) -> None:
        self._status = self.transport.status

    # -- one update --------------------------------------------------------

    async def handle_update(self, update: Mapping[str, Any]) -> bool:
        """
        Process one update. True if a command was dispatched.

        Order is deliberate and is the whole security argument:
        claim the id, then authorize the chat, then parse. A replayed or foreign
        update is refused before any payload is interpreted, so neither can
        reach a state transition.
        """
        if not isinstance(update, Mapping):
            self._status.updates_malformed += 1
            return False

        update_id = update.get("update_id")
        # 1. replay. Claimed before dispatch so a re-entrant copy is also caught.
        if not self.guard.claim("update", update_id):
            self._status.updates_refused_replay += 1
            log("TELEGRAM", "refused replayed update id")
            return False
        self._status.updates_seen += 1

        # 2. authorization, before the payload is even looked at
        if not chat_is_authorized(update, self.allowed_chat):
            self._status.updates_refused_chat += 1
            log("WARNING", "refused update from a non-allow-listed chat")
            self._refuse("unauthorized_chat", update)
            return False
        self._status.updates_authorized += 1

        # 3. parse
        data, _uid = payload_from_update(update)
        if not data:
            self._status.updates_malformed += 1
            self._refuse("no_dispatchable_payload", update)
            return False

        # 4. a second, payload-scoped idempotency claim. A different update id
        #    carrying the same callback is still a repeat of one operator intent.
        cq = update.get("callback_query")
        if isinstance(cq, Mapping):
            cid = cq.get("id")
            if cid is not None and not self.guard.claim("callback", cid):
                self._status.updates_refused_replay += 1
                log("TELEGRAM", "refused replayed callback id")
                return False

        # 5. dispatch to the EXISTING handler
        try:
            await self.handle_button(data)
        except Exception as exc:  # noqa: BLE001
            # a handler failure must not kill the poll loop
            log("ERROR", "operator command failed (%s): %s",
                type(exc).__name__, exc)
            self._status.last_error = f"handler:{type(exc).__name__}"
        return True

    def _refuse(self, why: str, update: Mapping[str, Any]) -> None:
        if self.on_refused is not None:
            try:
                self.on_refused(why, update)
            except Exception:  # noqa: BLE001
                pass

    # -- the poll loop -----------------------------------------------------

    async def poll_once(self, timeout_s: float = 20.0) -> int:
        """
        One long-poll. Returns the number of updates dispatched.

        A failed poll is reported to the caller and does not advance the offset,
        so a transient outage cannot cause updates to be skipped.
        """
        try:
            updates = await self.transport.get_updates(self._offset, timeout_s)
        except BotApiError as exc:
            # Record it here too, not only inside call(), so a failure injected
            # above the transport is still visible to the operator.
            self._status.last_error = _redact(
                str(exc), self.transport._token or "")
            raise
        dispatched = 0
        highest = self._offset
        for update in updates:
            uid = update.get("update_id")
            if isinstance(uid, int):
                # advance past this update regardless of whether it was
                # dispatched, so a refused update is not re-fetched forever
                highest = uid if highest is None else max(highest, uid + 1)
            if await self.handle_update(update):
                dispatched += 1
        self._offset = highest
        return dispatched

    async def run(self, stop: asyncio.Event, poll_timeout_s: float = 20.0,
                  idle_s: float = 0.0) -> None:
        """
        Poll until ``stop`` is set.

        Bounded exponential backoff on failure, reset on the first clean poll,
        so an outage costs a handful of requests rather than a hot loop.
        """
        self._status.polling = True
        delay = 0.5
        try:
            while not stop.is_set():
                try:
                    await self.poll_once(poll_timeout_s)
                    delay = 0.5
                except BotApiNotConfigured:
                    raise
                except BotApiError as exc:
                    self._status.last_error = _redact(
                        str(exc), self.transport._token or "")
                    log("WARNING", "telegram poll failed: %s", exc)
                    if isinstance(exc, BotApiUnauthorized):
                        # a bad credential is not going to fix itself
                        raise
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=delay)
                    except asyncio.TimeoutError:
                        pass
                    delay = min(delay * 2.0, 30.0)
                    continue
                if idle_s:
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=idle_s)
                    except asyncio.TimeoutError:
                        pass
        finally:
            self._status.polling = False

    def status_dict(self) -> dict[str, Any]:
        d = self._status.as_dict()
        d["chat_authorized"] = normalise_chat_id(self.allowed_chat) is not None
        d["replay_guard"] = self.guard.stats()
        return d
