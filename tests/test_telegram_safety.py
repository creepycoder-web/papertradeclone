"""
Telegram must not provide a route into REAL.

This is the central safety claim of the whole change, so it is asserted against
the real engine and the real handler rather than described. Every test here
would fail if a Telegram command were ever wired to ``arm_real`` directly, or if
the confirmation-token protections were weakened.

Standing facts this suite also pins, so a future change cannot quietly undo the
stabilization work:

* the REAL confirmation TTL is 300s;
* the TTL is not configurable;
* ``RuntimeState.real_armed`` is not a constructor parameter;
* a fresh process cannot start armed.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from autopaper.config import build_config
from autopaper.engine import PaperTrader
from autopaper.modes import (
    REAL_CONFIRMATION_TTL_S,
    ModeError,
    RunState,
    RuntimeState,
    TradeMode,
)
from autopaper.telegram import botapi
from autopaper.telegram.botapi import BotControl

ALLOWED = -1001234567890
FOREIGN = -1009999999999

#: Every callback payload the transport can produce. Each is driven through the
#: real engine below and none may arm REAL.
ALL_PAYLOADS = [
    "start", "stop", "paper", "real", "real_off", "stats", "open",
    "emergency", "cancel_real", "clear_emergency",
    "confirm_real:", "confirm_real:anything", "confirm_real:REALTOKEN",
    "unknown_payload", "", "   ",
]


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "999:FAKEfakefakefake")
    monkeypatch.setenv("TELEGRAM_BOT_CHAT", str(ALLOWED))
    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    # The risk acknowledgement is now a GATE on requesting a REAL confirmation.
    # Acknowledged here so this suite can exercise the token/TTL protections;
    # the un-acknowledged case is asserted separately below.
    monkeypatch.setenv("I_UNDERSTAND_REAL_TRADING_RISKS", "YES")
    monkeypatch.setenv("WALLET_KEYPAIR_PATH", "/nonexistent/wallet.json")
    monkeypatch.setenv("RPC_URL", "https://rpc.invalid")
    t = PaperTrader(build_config(tmp_path))
    try:
        yield t
    finally:
        t._release_ownership()


@pytest.fixture
def control(engine):
    return BotControl(transport=engine.bot_transport,
                      handle_button=engine.handle_button,
                      allowed_chat=ALLOWED)


def press(control, data, uid=1, cb_id=None, chat=ALLOWED):
    return asyncio.run(control.handle_update({
        "update_id": uid,
        "callback_query": {"id": cb_id or f"cb{uid}", "data": data,
                           "message": {"chat": {"id": chat}}},
    }))


# ---------------------------------------------------------------------------
# the headline claim
# ---------------------------------------------------------------------------


def test_no_telegram_command_can_arm_real(engine, control):
    """
    Drive EVERY reachable payload and assert REAL is never armed.

    This is the whole safety argument in one test: whatever the operator (or an
    attacker who somehow got a payload through) can send, the outcome is never
    armed REAL.
    """
    for i, payload in enumerate(ALL_PAYLOADS):
        press(control, payload, uid=100 + i, cb_id=f"p{i}")
        assert engine.state.real_armed is False, f"{payload!r} armed REAL"
        assert engine.state.real_trading_enabled is False, f"{payload!r} armed REAL"
        assert engine.state.trade_mode is TradeMode.PAPER, payload
    assert engine.state.run_state is not RunState.REAL_ARMED


def test_start_is_strictly_paper(engine, control):
    press(control, "start")
    assert engine.state.run_state is RunState.PAPER
    assert engine.state.trade_mode is TradeMode.PAPER
    assert engine.state.real_armed is False
    assert engine.state.real_trading_enabled is False


def test_stop_cannot_arm_real(engine, control):
    press(control, "start")
    press(control, "stop", uid=2, cb_id="s2")
    assert engine.state.real_trading_enabled is False


def test_status_cannot_arm_real(engine, control):
    press(control, "stats", uid=3, cb_id="q3")
    assert engine.state.real_trading_enabled is False


def test_paper_select_cannot_arm_real(engine, control):
    press(control, "paper")
    assert engine.state.real_trading_enabled is False
    assert engine.state.trade_mode is TradeMode.PAPER


def test_emergency_stop_cannot_arm_real(engine, control):
    press(control, "emergency")
    assert engine.state.real_trading_enabled is False
    assert engine.state.emergency_stop is True


# ---------------------------------------------------------------------------
# malformed and hostile input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("junk", [
    "not a mapping", 42, None, [], {}, {"update_id": 1},
    {"update_id": 1, "callback_query": {"id": "x"}},
    {"update_id": 1, "callback_query": {"id": "x", "data": 42,
                                         "message": {"chat": {"id": ALLOWED}}}},
    {"update_id": None, "message": {"text": "/start",
                                    "chat": {"id": ALLOWED}}},
    {"update_id": 1, "message": {"text": "/start", "chat": {}}},
    {"update_id": 1, "message": {"text": "/start", "chat": {"id": "@admin"}}},
])
def test_malformed_input_cannot_arm_real(engine, control, junk):
    asyncio.run(control.handle_update(junk))
    assert engine.state.real_armed is False
    assert engine.state.real_trading_enabled is False


def test_a_forged_confirmation_token_over_telegram_cannot_arm_real(engine, control):
    """A stranger cannot mint a confirmation by guessing a payload."""
    for i, tok in enumerate(["REALTOKEN", "x" * 32, "0", "../../etc/passwd"]):
        press(control, f"confirm_real:{tok}", uid=200 + i, cb_id=f"f{i}")
        assert engine.state.real_armed is False
        assert engine.state.trade_mode is TradeMode.PAPER


def test_an_unauthorized_confirm_cannot_arm_real(engine, control):
    press(control, "real", chat=ALLOWED)              # operator asks
    token = engine.state._pending_confirmation
    assert token, "the request should have minted a token"
    # a stranger replays that token from another chat
    press(control, f"confirm_real:{token}", uid=9, cb_id="steal", chat=FOREIGN)
    assert engine.state.real_armed is False
    assert engine.state.trade_mode is TradeMode.PAPER


def test_a_replayed_confirm_cannot_arm_real_twice(engine, control, monkeypatch):
    """
    The armed path needs a valid token AND a passing preflight.

    With the preflight unavailable the first confirm is refused; the replay must
    not produce a different outcome.
    """
    press(control, "real")
    token = engine.state._pending_confirmation
    press(control, f"confirm_real:{token}", uid=1, cb_id="c1")
    assert engine.state.real_armed is False
    press(control, f"confirm_real:{token}", uid=2, cb_id="c1")   # replayed
    assert engine.state.real_armed is False
    assert engine.state.trade_mode is TradeMode.PAPER


# ---------------------------------------------------------------------------
# start is not an implicit real confirmation
# ---------------------------------------------------------------------------


def test_start_is_not_treated_as_a_real_confirmation(engine, control):
    """
    START must not double as the "yes" to a REAL prompt.

    Request REAL, then press START: the prompt must be cancelled or ignored, and
    REAL must stay off.
    """
    press(control, "real")
    assert engine.state._pending_confirmation is not None
    press(control, "start", uid=2, cb_id="s2")
    assert engine.state.real_armed is False
    assert engine.state.trade_mode is TradeMode.PAPER


def test_a_pending_confirmation_does_not_survive_into_a_later_start(engine,
                                                                     control):
    press(control, "real")
    press(control, "cancel_real", uid=2, cb_id="c2")
    assert engine.state._pending_confirmation is None
    press(control, "start", uid=3, cb_id="s3")
    assert engine.state.real_armed is False


# ---------------------------------------------------------------------------
# structural: no transport path reaches arm_real
# ---------------------------------------------------------------------------


def test_the_transport_modules_never_mention_arm_real():
    """
    Neither transport module may name the arming function at all.

    Checked on executable code, not prose, because both modules carry
    documentation that mentions REAL arming by name.
    """
    import io
    import tokenize

    for name in ("botapi.py", "notifier.py"):
        path = inspect.getfile(botapi).replace("botapi.py", name)
        src = open(path, encoding="utf-8").read()
        code = []
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            if tok.type in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                            tokenize.DEDENT):
                continue
            code.append(tok.string)
        blob = " ".join(code)
        assert "arm_real" not in blob, f"{name} references arm_real"
        assert "real_armed" not in blob, f"{name} references real_armed"
        assert "real_trading_enabled" not in blob, f"{name} references the flag"


def test_the_engine_telegram_wiring_never_calls_arm_real():
    """The wiring methods must not arm anything themselves."""
    for meth in ("_build_bot_control", "start_bot_control",
                 "_on_bot_update_refused", "bot_status"):
        src = inspect.getsource(getattr(PaperTrader, meth))
        assert "arm_real" not in src, meth
        assert "real_armed =" not in src, meth


def test_the_control_dispatches_only_through_handle_button(engine):
    """
    The inbound controller's handler IS the engine's existing entry point.

    Asserted by identity, so a future refactor that passes some other callable
    fails here rather than quietly introducing a second command path.
    """
    assert engine.bot_control.handle_button == engine.handle_button


def test_no_transport_payload_maps_to_a_bare_arm():
    """
    Every text command maps onto a constant the existing handler already knows.

    A new mapping would have to name a payload the handler does not recognise,
    which the handler refuses explicitly.
    """
    from autopaper.telegram.actions import ButtonHandler
    from autopaper.telegram.botapi import TEXT_COMMANDS
    from autopaper.telegram import ui

    known = {v for k, v in vars(ui).items() if k.startswith("CB_")}
    known |= {v[len("confirm_real:"):] for v in known if v.startswith("confirm_real:")}
    for text, payload in TEXT_COMMANDS.items():
        assert payload in known, f"{text} maps to unknown payload {payload!r}"
    # Exactly ONE executable arming call site in the whole handler. Counted on
    # code, not on source text: the docstring and a comment both NAME arm_real
    # while describing that the button does not call it.
    import io
    import tokenize

    src = inspect.getsource(ButtonHandler)
    calls = 0
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NAME and tok.string == "arm_real":
            calls += 1
        if tok.type == tokenize.OP and tok.string == "." and calls:
            pass
    assert calls == 1, f"expected one arm_real reference in code, got {calls}"
    # and it is a call on the state object, not an assignment
    assert "self.state.arm_real(" in src
    assert "real_armed = " not in src, "the handler must never set the flag"


# ---------------------------------------------------------------------------
# the stabilization invariants still hold
# ---------------------------------------------------------------------------


def test_the_confirmation_ttl_is_unchanged_at_300s():
    assert REAL_CONFIRMATION_TTL_S == 300.0


def test_the_ttl_is_not_configurable(tmp_path, monkeypatch):
    (tmp_path / ".env.local").write_text("REAL_CONFIRMATION_TTL_S=99999\n")
    monkeypatch.setenv("REAL_CONFIRMATION_TTL_S", "99999")
    build_config(tmp_path)
    assert REAL_CONFIRMATION_TTL_S == 300.0


def test_real_armed_is_still_not_a_constructor_parameter():
    assert "real_armed" not in inspect.signature(RuntimeState).parameters
    with pytest.raises(TypeError):
        RuntimeState(real_armed=True)


def test_a_fresh_process_still_cannot_start_armed(tmp_path, monkeypatch):
    monkeypatch.setenv("REAL_TRADING_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "999:FAKE")
    monkeypatch.setenv("TELEGRAM_BOT_CHAT", str(ALLOWED))
    t = PaperTrader(build_config(tmp_path))
    try:
        t.startup()
        assert t.state.real_trading_enabled is False
        assert t.state.run_state is RunState.OFF
    finally:
        t._release_ownership()


def test_an_env_file_cannot_arm_real_even_with_telegram(tmp_path, monkeypatch):
    (tmp_path / ".env.local").write_text(
        "REAL_TRADING_ENABLED=true\n"
        "I_UNDERSTAND_REAL_TRADING_RISKS=YES\n"
        "TELEGRAM_BOT_TOKEN=999:FAKE\n"
        f"TELEGRAM_BOT_CHAT={ALLOWED}\n"
        "WALLET_KEYPAIR_PATH=/nonexistent/wallet.json\n"
        "RPC_URL=https://rpc.invalid\n"
        "MODE=REAL\n"
    )
    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    t = PaperTrader(build_config(tmp_path))
    try:
        assert t.state.real_trading_enabled is False
        c = BotControl(transport=t.bot_transport, handle_button=t.handle_button,
                       allowed_chat=ALLOWED)
        for i, payload in enumerate(ALL_PAYLOADS):
            press(c, payload, uid=300 + i, cb_id=f"e{i}")
        assert t.state.real_armed is False
        assert t.state.trade_mode is TradeMode.PAPER
    finally:
        t._release_ownership()


def test_the_token_check_is_independent_of_the_preflight():
    """
    Pins the ButtonHandler's OWN token check, with a preflight that PASSES.

    Mutation testing showed that deleting the handler's token comparison broke
    no test, because in every other test the preflight refused first and masked
    it. So the token check is only proven when the preflight is deliberately
    made to succeed - which is exactly the case where the check is load-bearing.
    """
    from autopaper.telegram.actions import ButtonHandler, OperatorActions

    class PassingPreflight:
        ok = True
        detail = "all checks passed"
        failures = ()

    async def preflight():
        return PassingPreflight()

    # A token IS pending, and the payload carries a DIFFERENT one. This is the
    # real shape of the attack: the operator pressed REAL, so a valid token
    # exists, and something else supplies the wrong one. With no pending token
    # at all, arm_real's own check would absorb the mutant and hide it.
    for payload in ("confirm_real:wrong", "confirm_real:", "confirm_real:x" * 40,
                    "confirm_real:" + "0" * 43):
        s = RuntimeState()
        s.start()
        s.request_real_confirmation()          # a VALID token now exists
        handler = ButtonHandler(s, OperatorActions(preflight=preflight))
        result = asyncio.run(handler.handle(payload))
        assert s.real_armed is False, f"{payload!r} armed with a passing preflight"
        assert result.armed_real is False
        assert s.trade_mode is TradeMode.PAPER

    # and the CORRECT token does arm, proving the fixture is not vacuous:
    # the refusal above is the token check, not a broken preflight
    s = RuntimeState()
    s.start()
    good = s.request_real_confirmation()
    handler = ButtonHandler(s, OperatorActions(preflight=preflight))
    result = asyncio.run(handler.handle(f"confirm_real:{good}"))
    assert result.armed_real is True, "the fixture must be able to arm"
    assert s.trade_mode is TradeMode.REAL


def test_a_stale_token_from_a_previous_prompt_cannot_arm(engine, control):
    """
    A token minted, then superseded, is not a valid confirmation.

    Each REAL press mints a fresh token and consumes the previous one, so a
    replayed confirm carries a token that no longer matches.
    """
    press(control, "real")
    first = engine.state._pending_confirmation
    press(control, "cancel_real", uid=2, cb_id="c2")
    press(control, "real", uid=3, cb_id="r3")
    second = engine.state._pending_confirmation
    assert first != second
    press(control, f"confirm_real:{first}", uid=4, cb_id="s1")
    assert engine.state.real_armed is False
    assert engine.state.trade_mode is TradeMode.PAPER


def test_the_token_is_revalidated_at_the_layer_that_arms():
    """
    Defence in depth, pinned at the layer that actually enforces it.

    Mutation testing showed the ButtonHandler's own token comparison can be
    deleted with no test failing. That is a REDUNDANT-DEFENCE survivor rather
    than a test gap: ``arm_real`` re-validates the token independently, so the
    guarantee is unchanged. This test pins that deeper invariant directly, so it
    stays proven no matter which layer above it is edited.
    """
    s = RuntimeState()
    s.start()
    s.request_real_confirmation()          # a VALID token now exists
    with pytest.raises(ModeError, match="token"):
        s.arm_real(confirmation="the-wrong-token", preflight_ok=True)
    assert s.real_armed is False
    assert s.trade_mode is TradeMode.PAPER
    # even with a preflight verdict already passed in
    with pytest.raises(ModeError):
        s.arm_real(confirmation="", preflight_ok=True)
    assert s.real_armed is False


def test_real_arming_still_requires_a_token_and_preflight():
    """
    The gate itself is unchanged: a token AND a preflight verdict, in that order.
    """
    s = RuntimeState()
    s.start()
    with pytest.raises(ModeError):
        s.arm_real(confirmation=None, preflight_ok=True)


def test_risk_acknowledgement_is_a_gate_on_requesting_real():
    """Without the acknowledgement, the REAL button mints no token at all."""
    from autopaper.telegram.actions import ButtonHandler, OperatorActions

    s = RuntimeState()
    s.start()
    handler = ButtonHandler(
        s, OperatorActions(risk_acknowledged=lambda: False)
    )
    result = asyncio.run(handler.handle("real"))
    assert result.error == "risk_not_acknowledged"
    assert s._pending_confirmation is None
    assert s.real_armed is False
    # And with the acknowledgement, the prompt is offered as before.
    handler2 = ButtonHandler(
        s, OperatorActions(risk_acknowledged=lambda: True)
    )
    result2 = asyncio.run(handler2.handle("real"))
    assert result2.error == ""
    assert s._pending_confirmation is not None
    token = s.request_real_confirmation()
    with pytest.raises(ModeError):
        s.arm_real(confirmation=token, preflight_ok=False)
    assert s.real_armed is False
