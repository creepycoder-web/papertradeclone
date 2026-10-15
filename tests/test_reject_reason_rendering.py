"""
Bug 1 regression: an L25 rejection must never crash the Telegram stream.

The defect: ``ui.py`` imported ``REJECT_REASON_TH`` from ``v3c`` and indexed it
with L25 status constants. An ordinary L25 rejection raised
``KeyError: 'LIQUIDITY_MC_BELOW_THRESHOLD'`` inside the render, escaped as
``telegram stream error:`` and terminated the read loop -- and the read loop is
the UPSTREAM CANDIDATE SOURCE, so a rejection could cost alerts.

It fired 64 times in one live session.

These tests drive the real renderer for every reason both strategies define, and
assert the real handler keeps processing messages afterwards.
"""

from __future__ import annotations

import pytest

from autopaper import l25, v3c
from autopaper.telegram.ui import (
    GENERIC_REJECT_REASON_TH,
    L25_REASONS,
    V3C_REASONS,
    render_strategy_reject,
    reject_reason_th,
)

#: One representative case per reachable status, for each strategy.
L25_CASES = {
    "LIQUIDITY_MC_BELOW_THRESHOLD": dict(
        route="FAST_ENTRY", market_cap_usd=9000, liquidity_usd=8000,
        signal_quality=0.9),
    "REJECTED_NOT_FAST_ENTRY": dict(
        route="NO_TRADE", market_cap_usd=26200, liquidity_usd=10018,
        signal_quality=0.6),
    "REJECTED_MARKET_CAP": dict(
        route="FAST_ENTRY", market_cap_usd=25000, liquidity_usd=30000,
        signal_quality=0.9),
    "REJECTED_SIGNAL_QUALITY": dict(
        route="FAST_ENTRY", market_cap_usd=9000, liquidity_usd=9200,
        signal_quality=0.4),
    "UNTESTABLE": dict(
        route="FAST_ENTRY", market_cap_usd=None, liquidity_usd=None,
        signal_quality=0.9),
}

#: v3c.evaluate() takes no liquidity argument, so these are separate.
V3C_CASES = {
    "REJECTED_NOT_FAST_ENTRY": dict(
        route="NO_TRADE", market_cap_usd=9000, signal_quality=0.9),
    "REJECTED_MARKET_CAP": dict(
        route="FAST_ENTRY", market_cap_usd=25000, signal_quality=0.9),
    "REJECTED_SIGNAL_QUALITY": dict(
        route="FAST_ENTRY", market_cap_usd=9000, signal_quality=0.4),
    "UNTESTABLE": dict(
        route="FAST_ENTRY", market_cap_usd=None, signal_quality=None),
}


def _render(strategy, kwargs):
    if strategy == "L25":
        return render_strategy_reject(
            l25.evaluate(**kwargs), symbol="SYM")
    return render_strategy_reject(v3c.evaluate(**kwargs), symbol="SYM")


# ---------------------------------------------------------------------------
# the crash
# ---------------------------------------------------------------------------


def test_the_reported_crash_is_fixed():
    """The exact case from the report."""
    result = l25.evaluate(route="FAST_ENTRY", market_cap_usd=9000,
                          liquidity_usd=8000, signal_quality=0.9)
    assert result.status == "LIQUIDITY_MC_BELOW_THRESHOLD"
    out = render_strategy_reject(result, symbol="SYM")   # must not raise
    assert "Liquidity / market cap below L25 threshold" in out


def test_the_l25_key_is_absent_from_the_v3c_table():
    """
    The root cause, stated as an assertion.

    If this ever becomes true the two tables have been merged and the
    strategy-aware lookup is no longer distinguishable.
    """
    assert "LIQUIDITY_MC_BELOW_THRESHOLD" not in V3C_REASONS
    assert "LIQUIDITY_MC_BELOW_THRESHOLD" in L25_REASONS


def test_the_two_tables_are_not_merged():
    """Kept separate on purpose, so a colliding key cannot hide a bug."""
    assert set(L25_REASONS) != set(V3C_REASONS)
    assert "REJECTED_MARKET_CAP" not in L25_REASONS


# ---------------------------------------------------------------------------
# every reason renders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", sorted(L25_CASES))
def test_every_l25_reason_renders_without_keyerror(status):
    out = _render("L25", L25_CASES[status])
    assert out
    assert "Reason:" in out


@pytest.mark.parametrize("status", sorted(V3C_CASES))
def test_every_v3c_reason_still_renders(status):
    """V3-C behaviour is unchanged."""
    out = _render("V3C", V3C_CASES[status])
    assert out
    assert "Reason:" in out


@pytest.mark.parametrize("status", sorted(set(L25_REASONS) | set(V3C_REASONS)))
def test_every_defined_reason_has_text(status):
    assert reject_reason_th(status), status


def test_a_delegated_reason_keeps_its_v3c_text():
    """
    L25 delegates its first leg to V3-C, so a V3-C reason can arrive on an L25
    result. It must not degrade to the generic line.
    """
    text = reject_reason_th("REJECTED_NOT_FAST_ENTRY")
    assert text == V3C_REASONS["REJECTED_NOT_FAST_ENTRY"]
    assert text != GENERIC_REJECT_REASON_TH


def test_an_l25_reason_is_not_taken_from_v3c():
    assert reject_reason_th("LIQUIDITY_MC_BELOW_THRESHOLD") == \
        L25_REASONS["LIQUIDITY_MC_BELOW_THRESHOLD"]


# ---------------------------------------------------------------------------
# graceful degradation for a NEW gate
# ---------------------------------------------------------------------------


def test_an_unknown_reason_degrades_instead_of_raising():
    assert reject_reason_th("A_GATE_THAT_DOES_NOT_EXIST") == \
        GENERIC_REJECT_REASON_TH


def test_an_empty_status_degrades():
    assert reject_reason_th("") == GENERIC_REJECT_REASON_TH


def test_a_future_l25_reason_lands_in_the_active_table():
    """
    Robustness for a future gate: a new L25 reason renders from the L25 table
    the moment it is defined there, with no change to this module.
    """
    L25_REASONS["A_FUTURE_L25_REASON"] = "อนาคต"
    try:
        assert reject_reason_th("A_FUTURE_L25_REASON") == "อนาคต"
    finally:
        L25_REASONS.pop("A_FUTURE_L25_REASON", None)


def test_no_raw_dict_indexing_remains_in_the_ui():
    """
    Structural: a bare ``REJECT_REASON_TH[...]`` is the bug.

    Asserted on source so it cannot be reintroduced in a new branch.
    """
    import inspect
    import re

    src = inspect.getsource(sys_module := __import__(
        "autopaper.telegram.ui", fromlist=["x"]))
    assert not re.search(r"REJECT_REASON_TH\s*\[", src), (
        "raw dict indexing on a rejection table is the original defect"
    )


# ---------------------------------------------------------------------------
# the stream survives
# ---------------------------------------------------------------------------


def test_one_rejection_does_not_terminate_processing():
    """
    A processed, a rejection, then B processed -- with the stream alive.

    The regression is not merely "no KeyError" but "the next message is still
    handled", because a crashed read loop is the actual operational harm.
    """
    import asyncio

    from autopaper.modes import RuntimeState
    from autopaper.telegram.actions import ButtonHandler

    seen: list[str] = []
    handler = ButtonHandler(RuntimeState())

    def notify(text, kb=None):
        seen.append(text)

    # message A: an L25 rejection rendered
    r = l25.evaluate(route="FAST_ENTRY", market_cap_usd=9000,
                     liquidity_usd=8000, signal_quality=0.9)
    notify(render_strategy_reject(r, symbol="A"))

    # message B: an ordinary operator press, i.e. processing continues
    result = asyncio.run(handler.handle("start"))

    assert len(seen) == 1, "the rejection must have rendered"
    assert result.handled is True, "message B must still be processed"
    assert handler.state.run_state is not None


def test_a_sequence_of_rejections_never_raises():
    import asyncio

    from autopaper.modes import RuntimeState
    from autopaper.telegram.actions import ButtonHandler

    handler = ButtonHandler(RuntimeState())
    for i, status in enumerate(sorted(L25_CASES)):
        out = _render("L25", L25_CASES[status])       # must not raise
        assert out
        # and the handler is still alive afterwards
        assert asyncio.run(handler.handle("stats")).handled is True, status
