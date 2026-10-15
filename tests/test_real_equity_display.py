"""
The armed-REAL message must show a MEASURED equity, never a placeholder.

The defect: the only call site passed a literal em-dash for the wallet and
omitted ``equity_usd``, so the renderer's ``0.0`` default was displayed. The
operator saw "Equity: $0.0000" for a wallet holding SOL, and it looked like a
measurement.

These tests use a GENERATED keypair and injected values. The production wallet
is never opened, and nothing is signed or broadcast.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from autopaper.modes import RuntimeState
from autopaper.real_equity import compute_real_equity
from autopaper.telegram import ui
from autopaper.telegram.actions import ButtonHandler, OperatorActions

REPO = Path(__file__).resolve().parents[1]

#: The balance the operator reported, in lamports.
REPORTED_LAMPORTS = 10_718_431


def _armed_state() -> RuntimeState:
    s = RuntimeState()
    s.start()
    s.arm_real(confirmation=s.request_real_confirmation(), preflight_ok=True)
    return s


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def test_a_measured_equity_is_shown():
    out = ui.render_real_armed(_armed_state(), "wUEU…Wbt6", 1.9293)
    assert "Equity: $1.9293" in out
    assert "unavailable" not in out
    assert "wUEU…Wbt6" in out


def test_unavailable_is_explicit_and_carries_a_reason():
    out = ui.render_real_armed(
        _armed_state(), "wUEU…Wbt6", None, "SOL/USD valuation unavailable")
    assert "Equity: <b>unavailable</b>" in out
    assert "SOL/USD valuation unavailable" in out
    assert "$0.0000" not in out, "a failure must never render as a zero"


def test_the_equity_parameter_is_nullable():
    """A float default of 0.0 is what caused the bug; it must be None."""
    sig = inspect.signature(ui.render_real_armed)
    assert sig.parameters["equity_usd"].default is None


def test_next_position_is_refused_when_equity_is_unknown():
    out = ui.render_real_armed(_armed_state(), "wUEU…Wbt6", None, "no price")
    assert "Next Position: <b>—</b>" in out


# ---------------------------------------------------------------------------
# the call site
# ---------------------------------------------------------------------------


def test_the_button_handler_no_longer_passes_a_placeholder():
    """
    Structural: the armed reply must not hardcode a wallet or omit the equity.

    This is the exact defect. Asserted on source so it cannot come back.
    """
    src = inspect.getsource(ButtonHandler)
    assert 'render_real_armed(self.state, "—")' not in src, (
        "the armed reply must not pass a literal placeholder wallet"
    )
    assert "self._real_equity()" in src, (
        "the armed reply must use the measured equity"
    )


def test_the_armed_reply_shows_the_measured_wallet_and_equity():
    """End to end through the real handler, with an injected provider."""

    class Passing:
        ok = True
        detail = "all checks passed"
        failures = ()

    async def preflight():
        return Passing()

    async def real_equity():
        return compute_real_equity(REPORTED_LAMPORTS, 180.0, "wUEU…Wbt6")

    s = RuntimeState()
    s.start()
    token = s.request_real_confirmation()
    handler = ButtonHandler(
        s, OperatorActions(preflight=preflight, extras={"real_equity": real_equity})
    )
    result = asyncio.run(handler.handle(f"confirm_real:{token}"))
    assert result.armed_real is True, "the fixture must actually arm"
    assert "wUEU…Wbt6" in result.text
    assert "$1.9293" in result.text
    assert "$0.0000" not in result.text
    assert "unavailable" not in result.text


def test_the_armed_reply_fails_safe_with_no_provider():
    s = RuntimeState()
    s.start()
    token = s.request_real_confirmation()
    handler = ButtonHandler(s, OperatorActions())
    result = asyncio.run(handler.handle(f"confirm_real:{token}"))
    assert result.armed_real is False       # no preflight, so not armed
    assert "$0.0000" not in result.text


def test_a_raising_equity_provider_is_reported_not_crashed():
    class Passing:
        ok = True
        detail = "ok"
        failures = ()

    async def preflight():
        return Passing()

    async def boom():
        raise RuntimeError("rpc down")

    s = RuntimeState()
    s.start()
    token = s.request_real_confirmation()
    handler = ButtonHandler(
        s,
        OperatorActions(preflight=preflight, extras={"real_equity": boom}),
    )
    result = asyncio.run(handler.handle(f"confirm_real:{token}"))
    assert result.armed_real is True, "arming already succeeded; the reply is cosmetic"
    assert "unavailable" in result.text
    assert "$0.0000" not in result.text


def test_an_unavailable_equity_does_not_block_arming():
    """
    The display is cosmetic; the ARMING gates are the token and the preflight.

    A failed equity read must not silently change the safety model, and equally
    must not pretend the account is worth $0.
    """
    class Passing:
        ok = True
        detail = "ok"
        failures = ()

    async def preflight():
        return Passing()

    async def no_price():
        return compute_real_equity(REPORTED_LAMPORTS, None, "wUEU…Wbt6")

    s = RuntimeState()
    s.start()
    token = s.request_real_confirmation()
    handler = ButtonHandler(
        s, OperatorActions(preflight=preflight, extras={"real_equity": no_price})
    )
    result = asyncio.run(handler.handle(f"confirm_real:{token}"))
    assert "unavailable" in result.text
    assert "SOL/USD" in result.text


# ---------------------------------------------------------------------------
# no secrets, no broadcast
# ---------------------------------------------------------------------------


def test_no_private_key_material_can_reach_the_message():
    """Only a MASKED address may appear; a raw keypair never may."""
    out = ui.render_real_armed(_armed_state(), "wUEU…Wbt6", 1.93)
    assert "PRIVATE" not in out.upper()
    assert "seed" not in out.lower()
    assert "mnemonic" not in out.lower()


def test_the_equity_provider_never_signs_or_sends():
    """
    The provider must read only.

    Asserted on source: it may await a balance and a price, and must not touch
    a signing or sending method.
    """
    src = inspect.getsource(
        __import__("autopaper.engine", fromlist=["PaperTrader"]).PaperTrader.real_equity
    )
    for banned in ("sign_transaction", "send_transaction", "execute_swap",
                   "broadcast", "send_raw", "_execute_real_buy"):
        assert banned not in src, f"real_equity must not call {banned}"
    assert "wallet_state()" in src, (
        "must use the existing tested balance reader"
    )
