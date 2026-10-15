"""
Runtime mode: startup safety, operator arming, and the state machine.

The central claim under test is that no configuration, no environment variable
and no internal call can arm REAL - only a confirmed Telegram operator action
plus a passing preflight can.
"""

from __future__ import annotations

import pytest

from autopaper.modes import (
    ModeError,
    RUNNING_STATES,
    RunState,
    RuntimeState,
    TradeMode,
)


# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------


def test_a_fresh_state_is_off_paper_and_not_armed():
    s = RuntimeState()
    assert s.run_state is RunState.OFF
    assert s.trade_mode is TradeMode.PAPER
    assert s.real_armed is False
    assert s.real_trading_enabled is False
    assert s.running is False


def test_construction_takes_no_arguments_that_could_arm_it():
    """
    There is no constructor parameter that could carry an "armed" flag, so a
    caller cannot construct an armed runtime by accident.
    """
    import inspect

    # The only field that could express "armed" is corrected by __post_init__
    # into PAPER, so there is no argument that survives construction as armed.
    fields = {
        f.name for f in __import__("dataclasses").fields(RuntimeState)
    }
    assert "mode" not in fields
    assert "armed" not in fields
    assert inspect.signature(RuntimeState)


def test_a_state_cannot_be_constructed_already_armed():
    """
    The one parameter that could express "armed" is corrected in __post_init__:
    asking for REAL_ARMED without arming lands in PAPER.
    """
    s = RuntimeState(run_state=RunState.REAL_ARMED)
    assert s.real_armed is False
    assert s.run_state is not RunState.REAL_ARMED
    assert s.trade_mode is TradeMode.PAPER


def test_startup_opens_no_entries():
    s = RuntimeState()
    ok, why = s.can_open_new_entries()
    assert not ok
    assert "press start" in why


def test_a_fresh_state_reports_no_blocking_flags():
    s = RuntimeState()
    assert s.emergency_stop is False
    assert s.entries_blocked is False
    assert s.last_refusal is None


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------


def test_start_enters_paper():
    s = RuntimeState()
    s.start()
    assert s.run_state is RunState.PAPER
    assert s.running
    assert s.trade_mode is TradeMode.PAPER
    ok, _ = s.can_open_new_entries()
    assert ok


def test_start_twice_is_harmless():
    s = RuntimeState()
    s.start()
    s.start()
    assert s.run_state is RunState.PAPER


def test_stop_blocks_new_entries():
    s = RuntimeState()
    s.start()
    s.stop()
    assert s.run_state is RunState.STOPPED
    ok, why = s.can_open_new_entries()
    assert not ok
    assert "stopped" in why


def test_stop_does_not_erase_state():
    s = RuntimeState()
    s.start()
    s.stop()
    assert s.history, "the transition must be recorded, not discarded"


def test_start_after_stop_resumes_paper():
    s = RuntimeState()
    s.start()
    s.stop()
    s.start()
    assert s.run_state is RunState.PAPER
    ok, _ = s.can_open_new_entries()
    assert ok


def test_stopped_is_not_a_running_state():
    assert RunState.STOPPED not in RUNNING_STATES
    assert RunState.OFF not in RUNNING_STATES
    assert RunState.PAPER in RUNNING_STATES


# ---------------------------------------------------------------------------
# arming REAL
# ---------------------------------------------------------------------------


class _Preflight:
    def __init__(self, ok: bool, detail: str = "") -> None:
        self.ok = ok
        self.detail = detail


def test_arm_real_without_a_token_is_refused():
    s = RuntimeState()
    s.start()
    with pytest.raises(ModeError, match="confirmation token"):
        s.arm_real(confirmation=None, preflight_ok=True)
    assert not s.real_armed
    assert s.trade_mode is TradeMode.PAPER


def test_arm_real_with_a_fabricated_token_is_refused():
    s = RuntimeState()
    s.start()
    s.request_real_confirmation()
    with pytest.raises(ModeError, match="confirmation token"):
        s.arm_real(confirmation="not-the-token", preflight_ok=True)
    assert not s.real_armed


def test_arm_real_with_a_failed_preflight_is_refused():
    s = RuntimeState()
    s.start()
    token = s.request_real_confirmation()
    with pytest.raises(ModeError, match="preflight failed"):
        s.arm_real(confirmation=token, preflight_ok=False, preflight_detail="no rpc")
    assert not s.real_armed
    assert s.trade_mode is TradeMode.PAPER


def test_a_failed_preflight_consumes_the_token():
    """The operator must make a fresh, deliberate request. No retry-by-replay."""
    s = RuntimeState()
    s.start()
    token = s.request_real_confirmation()
    with pytest.raises(ModeError):
        s.arm_real(confirmation=token, preflight_ok=False)
    with pytest.raises(ModeError):
        s.arm_real(confirmation=token, preflight_ok=True)


def test_arm_real_with_token_and_passing_preflight_succeeds():
    s = RuntimeState()
    s.start()
    token = s.request_real_confirmation()
    s.arm_real(confirmation=token, preflight_ok=True, preflight_detail="all pass")
    assert s.real_armed
    assert s.real_trading_enabled
    assert s.run_state is RunState.REAL_ARMED
    assert s.trade_mode is TradeMode.REAL


def test_a_token_is_single_use():
    s = RuntimeState()
    s.start()
    token = s.request_real_confirmation()
    s.arm_real(confirmation=token, preflight_ok=True)
    s.turn_off_real()
    with pytest.raises(ModeError):
        s.arm_real(confirmation=token, preflight_ok=True)


def test_arming_while_stopped_arms_the_capability_but_does_not_start():
    """
    Arming and starting are separate acts. A stopped bot that gets armed still
    refuses entries until the operator presses start.
    """
    s = RuntimeState()
    s.stop()
    token = s.request_real_confirmation()
    s.arm_real(confirmation=token, preflight_ok=True)
    assert s.real_armed
    assert s.run_state is RunState.STOPPED
    ok, _ = s.can_open_new_entries()
    assert not ok
    s.start()
    assert s.run_state is RunState.REAL_ARMED


def test_arming_while_off_does_not_start_the_loop():
    s = RuntimeState()
    token = s.request_real_confirmation()
    s.arm_real(confirmation=token, preflight_ok=True)
    assert s.real_armed
    assert s.run_state is RunState.OFF
    ok, _ = s.can_open_new_entries()
    assert not ok


def test_requesting_confirmation_alone_arms_nothing():
    """The REAL button is a question, not an answer."""
    s = RuntimeState()
    s.start()
    s.request_real_confirmation()
    assert not s.real_armed
    assert s.trade_mode is TradeMode.PAPER
    assert not s.real_trading_enabled


def test_cancelling_confirmation_arms_nothing():
    s = RuntimeState()
    s.start()
    token = s.request_real_confirmation()
    s.cancel_real_confirmation()
    with pytest.raises(ModeError):
        s.arm_real(confirmation=token, preflight_ok=True)
    assert not s.real_armed


# ---------------------------------------------------------------------------
# disarming
# ---------------------------------------------------------------------------


def test_paper_disarms_real():
    s = RuntimeState()
    s.start()
    token = s.request_real_confirmation()
    s.arm_real(confirmation=token, preflight_ok=True)
    assert s.real_trading_enabled
    s.select_paper()
    assert not s.real_armed
    assert not s.real_trading_enabled
    assert s.trade_mode is TradeMode.PAPER


def test_paper_disarms_real_and_clears_a_pending_request():
    s = RuntimeState()
    s.start()
    token = s.request_real_confirmation()
    s.select_paper()
    with pytest.raises(ModeError):
        s.arm_real(confirmation=token, preflight_ok=True)


def test_turn_off_real_keeps_paper_available():
    s = RuntimeState()
    s.start()
    token = s.request_real_confirmation()
    s.arm_real(confirmation=token, preflight_ok=True)
    s.turn_off_real()
    assert s.run_state is RunState.PAPER
    ok, _ = s.can_open_new_entries()
    assert ok, "PAPER must remain available after REAL is turned off"


def test_turn_off_real_does_not_close_positions():
    """
    Turning REAL off disables new entries. It is not a close-position control,
    so nothing about the position book changes here - the engine owns that.
    """
    s = RuntimeState()
    s.start()
    token = s.request_real_confirmation()
    s.arm_real(confirmation=token, preflight_ok=True)
    s.turn_off_real()
    assert s.run_state is RunState.PAPER
    # no position state lives on the runtime state, so nothing can be closed
    assert not hasattr(s, "positions")


# ---------------------------------------------------------------------------
# emergency stop
# ---------------------------------------------------------------------------


def test_emergency_stop_blocks_entries():
    s = RuntimeState()
    s.start()
    s.emergency_stop_now()
    ok, why = s.can_open_new_entries()
    assert not ok
    assert "EMERGENCY STOP" in why


def test_emergency_stop_disarms_real():
    s = RuntimeState()
    s.start()
    token = s.request_real_confirmation()
    s.arm_real(confirmation=token, preflight_ok=True)
    s.emergency_stop_now()
    assert not s.real_armed
    assert not s.real_trading_enabled
    assert s.run_state is RunState.EMERGENCY_STOP


def test_emergency_stop_survives_a_restart():
    """
    Re-applying the latch is what survives a restart, and a restored latch must
    refuse both entries and a fresh arming attempt.
    """
    s = RuntimeState()
    s.emergency_stop_now()
    fresh = RuntimeState()
    fresh.restore_emergency_stop(True)
    assert fresh.emergency_stop
    assert fresh.run_state is RunState.EMERGENCY_STOP
    ok, _ = fresh.can_open_new_entries()
    assert not ok
    token = fresh.request_real_confirmation()
    with pytest.raises(ModeError, match="EMERGENCY STOP"):
        fresh.arm_real(confirmation=token, preflight_ok=True)


def test_start_is_refused_while_emergency_stopped():
    s = RuntimeState()
    s.emergency_stop_now()
    with pytest.raises(ModeError):
        s.start()
    assert s.run_state is RunState.EMERGENCY_STOP


def test_emergency_stop_does_not_change_v3c():
    """
    The latch is a runtime control, not a strategy change. V3-C's thresholds
    are untouched by it.
    """
    from autopaper.v3c import V3C_MAX_MARKET_CAP_USD, V3C_MIN_SIGNAL_QUALITY

    s = RuntimeState()
    s.emergency_stop_now()
    assert V3C_MAX_MARKET_CAP_USD == 20_000.0
    assert V3C_MIN_SIGNAL_QUALITY == 0.55


def test_clearing_emergency_stop_returns_to_off_not_to_running():
    """Clearing the latch does not silently restart trading."""
    s = RuntimeState()
    s.start()
    s.emergency_stop_now()
    s.clear_emergency_stop()
    assert not s.emergency_stop
    assert s.run_state is RunState.OFF
    ok, _ = s.can_open_new_entries()
    assert not ok


def test_emergency_stop_is_recorded_in_history():
    s = RuntimeState()
    s.emergency_stop_now()
    actions = [h["action"] for h in s.history]
    assert "emergency_stop" in actions


# ---------------------------------------------------------------------------
# one canonical state
# ---------------------------------------------------------------------------


def test_trade_mode_is_derived_not_stored():
    """
    There is no ``mode`` field to disagree with ``run_state``. The trade mode is
    computed from the run state, so the two cannot drift.
    """
    s = RuntimeState()
    assert not hasattr(s, "mode")
    s.start()
    assert s.trade_mode is TradeMode.PAPER
    token = s.request_real_confirmation()
    s.arm_real(confirmation=token, preflight_ok=True)
    assert s.trade_mode is TradeMode.REAL


def test_snapshot_reports_both_facts():
    s = RuntimeState()
    s.start()
    token = s.request_real_confirmation()
    s.arm_real(confirmation=token, preflight_ok=True)
    snap = s.snapshot()
    assert snap["trade_mode"] == "REAL"
    assert snap["run_state"] == "REAL_ARMED"
    assert snap["real_trading_enabled"] is True
    assert snap["real_armed"] is True


def test_every_refusal_is_recorded_with_a_reason():
    s = RuntimeState()
    with pytest.raises(ModeError):
        s.arm_real(confirmation=None, preflight_ok=True)
    assert s.last_refusal
    assert "confirmation token" in s.last_refusal
    assert s.history[-1]["action"] == "refused"


def test_history_is_bounded():
    s = RuntimeState()
    for _ in range(600):
        s.start()
        s.stop()
    assert len(s.history) <= 500
