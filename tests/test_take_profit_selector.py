"""
The selectable L25 take-profit: UI reachability, validation, persistence and
per-position snapshotting.

WHAT THESE COVER
================
Take-profit used to be reachable only by selecting the whole ``L25_TP100``
STRATEGY ARM. It is now a SETTING with its own submenu, its own validated
vocabulary and its own durable file, and the arm that existed to carry it is
retired from the selector.

The load-bearing safety property is the last one: a position records the level
it was OPENED with, and every exit reads the position's copy. Changing the
operator's selection afterwards must not rewrite the exit policy of something
already in the book. These tests pin that.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

from autopaper.modes import RuntimeState
from autopaper.paper_live.arms import NON_SELECTABLE_ARMS, ResearchArm
from autopaper.paper_live.take_profit import (
    NO_TAKE_PROFIT,
    SELECTABLE_TAKE_PROFITS,
    TAKE_PROFIT_MAX_PCT,
    TAKE_PROFIT_MIN_PCT,
    TakeProfitError,
    for_position,
    label,
    normalise,
    status_line,
)
from autopaper.telegram import ui
from autopaper.telegram.actions import ButtonHandler, OperatorActions
from autopaper.telegram.botapi import TEXT_COMMANDS, payload_from_update

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from tests import paper_bridge_fakes as _fk
except ImportError:  # pragma: no cover - direct file invocation
    import paper_bridge_fakes as _fk

alert, bridge, real_report, run = _fk.alert, _fk.bridge, _fk.real_report, _fk.run

#: The presets the specification asks for, verbatim.
SPEC_PRESETS = (5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0)


def _press(handler: ButtonHandler, payload: str):
    return asyncio.run(handler.handle(payload))


# ---------------------------------------------------------------------------
# 1. the selector is reachable from the REAL UI
# ---------------------------------------------------------------------------


def test_the_take_profit_control_is_on_the_real_main_menu():
    """A control the operator cannot find is not a control."""
    rows = ui.operator_keyboard(RuntimeState())
    found = [(b["text"], b["callback_data"])
             for row in rows for b in row
             if b["callback_data"] == ui.CB_TAKE_PROFIT]
    assert found, "TAKE PROFIT is not on the operator menu at all"
    assert found[0][0] == ui.BTN_TAKE_PROFIT


def test_the_take_profit_button_opens_a_menu_that_changes_nothing():
    state = RuntimeState()
    handler = ButtonHandler(state, OperatorActions())
    before = (state.run_state, state.real_armed, state.emergency_stop,
              state.entries_blocked)
    result = _press(handler, ui.CB_TAKE_PROFIT)
    assert result.handled
    # Read-only, exactly like the strategy submenu.
    assert (state.run_state, state.real_armed, state.emergency_stop,
            state.entries_blocked) == before


def test_the_take_profit_control_is_in_the_declared_button_vocabulary():
    assert ui.BTN_TAKE_PROFIT in ui.OPERATOR_BUTTONS
    assert ui.CB_TAKE_PROFIT in ui.OPERATOR_CALLBACKS


def test_the_active_selection_is_displayed_clearly():
    """``L25 TAKE PROFIT: +10%`` must be readable, not a bare 10."""
    for pct, expected in ((10.0, "L25 TAKE PROFIT: +10%"),
                          (100.0, "L25 TAKE PROFIT: +100%"),
                          (None, "L25 TAKE PROFIT: OFF")):
        state = RuntimeState()
        state.take_profit_pct = pct
        assert state.take_profit.status_line == expected, pct
        assert status_line(pct) == expected, pct
        # And it reaches the surfaces the operator reads.
        assert state.take_profit_status_line == expected
        assert expected in ui.render_menu(state)


def test_the_status_card_shows_the_take_profit_level():
    from autopaper.telegram.ui import StatusView, render_status
    state = RuntimeState()
    state.start()
    state.take_profit_pct = 30.0
    view = StatusView(state=state, equity_usd=100.0, cash_usd=100.0,
                      open_positions=0)
    text = render_status(view)
    assert "L25 TAKE PROFIT" in text or "TAKE PROFIT" in text
    assert "+30%" in text


# ---------------------------------------------------------------------------
# 2. every preset works
# ---------------------------------------------------------------------------


def test_the_selector_offers_exactly_the_requested_presets():
    assert tuple(SELECTABLE_TAKE_PROFITS) == SPEC_PRESETS


@pytest.mark.parametrize("pct", SPEC_PRESETS)
def test_every_supported_tp_preset_works(pct):
    """Each preset is reachable through the REAL dispatch path and takes."""
    state = RuntimeState()
    handler = ButtonHandler(state, OperatorActions())
    result = _press(handler, f"{ui.CB_TAKE_PROFIT_SELECT}{pct:g}")
    assert result.handled, result.error
    assert state.take_profit_pct == pct
    assert state.take_profit.label == f"+{pct:g}%"
    assert not result.error


def test_every_preset_renders_on_the_submenu_with_one_tick():
    state = RuntimeState()
    state.take_profit_pct = 50.0
    rows = ui.take_profit_keyboard(state.take_profit_pct)
    ticked = [r[0]["text"] for r in rows if r[0]["text"].startswith("✅")]
    # exactly one preset plus no other row is ticked
    assert ticked == ["✅ +50%"], ticked


def test_tp100_is_a_selectable_VALUE_not_a_strategy():
    """
    +100% must stay reachable -- but as a TP value, not as a strategy profile.
    """
    state = RuntimeState()
    handler = ButtonHandler(state, OperatorActions())
    result = _press(handler, f"{ui.CB_TAKE_PROFIT_SELECT}100")
    assert result.handled and not result.error
    assert state.take_profit_pct == 100.0
    # ...and the retired profile is NOT on the strategy picker.
    assert "L25_TP100" not in "\n".join(
        r[0]["text"] for r in ui.strategy_keyboard(state))


def test_no_take_profit_is_selectable_and_distinct_from_a_level():
    state = RuntimeState()
    state.take_profit_pct = 20.0
    handler = ButtonHandler(state, OperatorActions())
    result = _press(handler, ui.CB_TAKE_PROFIT_OFF)
    assert result.handled and not result.error
    assert state.take_profit_pct is None
    assert state.take_profit.label == "OFF"


# ---------------------------------------------------------------------------
# 3. custom-input validation
# ---------------------------------------------------------------------------


def test_a_custom_positive_percentage_is_accepted():
    state = RuntimeState()
    handler = ButtonHandler(state, OperatorActions())
    result = _press(handler, f"{ui.CB_TAKE_PROFIT_SELECT}12.5")
    assert result.handled and not result.error, result.error
    assert state.take_profit_pct == 12.5


def test_the_custom_button_offers_a_validated_route_not_a_raw_text_box():
    """The UI has no free-text state machine, so the button is a PROMPT."""
    state = RuntimeState()
    handler = ButtonHandler(state, OperatorActions())
    result = _press(handler, ui.CB_TAKE_PROFIT_CUSTOM)
    assert result.handled
    # It must not silently change the level, and it must tell the operator how.
    assert state.take_profit_pct is None
    assert "/tp" in result.text


@pytest.mark.parametrize("bad", [
    "0", "-5", "abc", "1e", "nan", "inf",
    "999999", "0.0001", "5%5", "+",
])
def test_an_invalid_custom_value_is_refused_and_the_old_one_survives(bad):
    """
    A typo must never silently disable a configured take-profit: the level in
    force BEFORE the bad value is the level AFTER it, exactly.
    """
    state = RuntimeState()
    state.take_profit_pct = 20.0
    handler = ButtonHandler(state, OperatorActions())
    result = _press(handler, f"{ui.CB_TAKE_PROFIT_SELECT}{bad}")
    assert state.take_profit_pct == 20.0, (bad, state.take_profit_pct)
    assert result.error == "invalid_take_profit", (bad, result.error)


def test_the_custom_bounds_are_stated():
    assert TAKE_PROFIT_MIN_PCT > 0
    assert TAKE_PROFIT_MAX_PCT > TAKE_PROFIT_MIN_PCT


def test_a_valid_persisted_file_is_restored_across_a_restart(tmp_path):
    from autopaper.config import build_config
    from autopaper.engine import PaperTrader

    class _NoNetworkMarket:
        async def quote_async(self, mint):
            raise AssertionError("no network in this test")

    cfg = build_config(tmp_path)
    first = PaperTrader(cfg, market=_NoNetworkMarket())
    first._persist_take_profit(15.0)

    stored = json.loads(
        (tmp_path / "var" / "take_profit_setting.json").read_text())
    assert stored["take_profit_pct"] == 15.0

    # A fresh process restores the selection the operator actually made.
    second = PaperTrader(cfg, market=_NoNetworkMarket())
    assert second.state.take_profit_pct == 15.0
    assert second.state.take_profit.status_line == "L25 TAKE PROFIT: +15%"


def test_an_absent_file_means_no_take_profit_not_a_default_level(tmp_path):
    """
    The default must preserve L25_CURRENT's behaviour: it had NO take-profit.
    Modelling "no trigger" as +100% would invent an exit the operator never
    had, so the default is explicitly OFF.
    """
    from autopaper.config import build_config
    from autopaper.engine import PaperTrader

    class _NoNetworkMarket:
        async def quote_async(self, mint):
            raise AssertionError("no network in this test")

    cfg = build_config(tmp_path)
    trader = PaperTrader(cfg, market=_NoNetworkMarket())
    assert trader.state.take_profit_pct is None
    assert trader.state.take_profit.label == "OFF"


def test_a_corrupt_persisted_file_falls_back_to_off_and_not_a_guess(tmp_path):
    from autopaper.config import build_config
    from autopaper.engine import PaperTrader

    class _NoNetworkMarket:
        async def quote_async(self, mint):
            raise AssertionError("no network in this test")

    cfg = build_config(tmp_path)
    (tmp_path / "var").mkdir(parents=True, exist_ok=True)
    (tmp_path / "var" / "take_profit_setting.json").write_text(
        json.dumps({"take_profit_pct": "not-a-number"}))
    trader = PaperTrader(cfg, market=_NoNetworkMarket())
    assert trader.state.take_profit_pct is None


def test_tp_selection_is_not_written_to_env_local(tmp_path):
    """The whole feature must survive without touching ``.env.local``."""
    from autopaper.config import build_config
    from autopaper.engine import PaperTrader

    class _NoNetworkMarket:
        async def quote_async(self, mint):
            raise AssertionError("no network in this test")

    cfg = build_config(tmp_path)
    trader = PaperTrader(cfg, market=_NoNetworkMarket())
    trader.state.take_profit_pct = 25.0
    trader._persist_take_profit(25.0)
    # the durable copy lives in var/, not in the environment
    assert (tmp_path / "var" / "take_profit_setting.json").exists()
    assert not (tmp_path / ".env.local").exists()


# ---------------------------------------------------------------------------
# 4. the duplicate TP100 strategy/profile is GONE from the selector
# ---------------------------------------------------------------------------


def test_the_tp100_arm_is_not_selectable():
    assert ResearchArm.L25_TP100 in NON_SELECTABLE_ARMS
    assert "L25_TP100" not in ui.strategy_keyboard(RuntimeState()) or True
    text = "\n".join(
        r[0]["text"] for r in ui.strategy_keyboard(RuntimeState()))
    assert "L25_TP100" not in text


def test_the_tp100_arm_is_not_in_the_strategy_submenu_text():
    assert "L25_TP100" not in ui.render_strategy_menu(RuntimeState())


def test_a_retired_arm_cannot_be_selected_through_the_real_ui():
    state = RuntimeState()
    handler = ButtonHandler(state, OperatorActions())
    result = _press(handler, f"{ui.CB_PROFILE}L25_TP100")
    assert not _ok(result)
    assert "L25_TP100" not in state.arms


def test_a_persisted_tp100_selection_is_not_restored(tmp_path):
    """The old file must not resurrect the retired arm on the next start."""
    from autopaper.config import build_config
    from autopaper.engine import PaperTrader

    class _NoNetworkMarket:
        async def quote_async(self, mint):
            raise AssertionError("no network in this test")

    cfg = build_config(tmp_path)
    (tmp_path / "var").mkdir(parents=True, exist_ok=True)
    (tmp_path / "var" / "strategy_selection.json").write_text(json.dumps(
        {"strategies": ["L25_TP100"], "trade_mode": "PAPER", "at": time.time()}))
    trader = PaperTrader(cfg, market=_NoNetworkMarket())
    assert "L25_TP100" not in trader.state.arms


def _ok(result) -> bool:
    """A ButtonResult is only a success when it was handled AND not refused."""
    return bool(result.handled) and not result.error


# ---------------------------------------------------------------------------
# 5. the position carries its own snapshot
# ---------------------------------------------------------------------------


def _signal(mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"):
    """``on_signal`` takes a SignalSnapshot; the bridge asserts PAPER-only."""
    from autopaper.paper_live.engine import build_signal_snapshot
    return build_signal_snapshot(
        channel="MooDengPresidentCallers", text=f"BUY ${mint}",
        message_id=4242, telegram_timestamp=time.time() - 1.2,
        mint=mint, symbol="T")


def _opened_engine(take_profit_pct, arms=("L25_CURRENT", "L25_MAX_HOLD")):
    """
    A bridge with a genuinely OPEN position per arm.

    ``handle_alert`` is the execution-aware entry point: the decision is already
    taken and the bridge is what opens the book, which is the path that stamps
    the take-profit snapshot.
    """
    from autopaper.paper_live.arms import parse_arms
    engine = bridge(arms=parse_arms(list(arms)),
                    take_profit_pct=take_profit_pct)
    run(engine.handle_alert(snapshot=alert(), report=real_report()))
    return engine


def test_each_new_position_records_the_tp_snapshot():
    engine = _opened_engine(10.0)
    opened = [a.position for a in engine.accounts.values()
              if a.position is not None]
    assert opened, "the fixture opened no position, so the test is vacuous"
    for position in opened:
        assert position.take_profit_pct == 10.0, position.arm


def test_changing_the_global_tp_does_not_mutate_open_positions():
    """
    THE invariant. The position's take-profit is read from the POSITION, so a
    mid-flight selection change cannot rewrite the exit policy of a book that
    is already open.
    """
    engine = _opened_engine(10.0)
    before = {a: getattr(p, "take_profit_pct", None)
              for a, acc in engine.accounts.items()
              if (p := acc.position) is not None}
    assert set(before.values()) == {10.0}, before
    engine.take_profit_pct = 100.0
    after = {a: getattr(p, "take_profit_pct", None)
             for a, acc in engine.accounts.items()
             if (p := acc.position) is not None}
    assert after == before, "changing the global take-profit mutated an OPEN position"


def test_the_position_view_shows_the_snapshot_it_was_opened_with():
    from autopaper.paper_live.take_profit import for_position as _fp
    engine = _opened_engine(20.0)
    position = next(a.position for a in engine.accounts.values()
                    if a.position is not None)
    assert _fp(position) == 20.0


def test_a_position_with_no_snapshot_has_no_take_profit():
    engine = _opened_engine(None)
    position = next(a.position for a in engine.accounts.values()
                    if a.position is not None)
    assert position.take_profit_pct is None
    assert for_position(position) is None


def test_a_legacy_tp100_position_keeps_its_hundred_percent_target():
    """
    A position recorded under the retired profile must not silently LOSE its
    take-profit just because the profile is no longer selectable.
    """

    class _Legacy:
        take_profit_pct = None
        production_profile = "L25_TP100"
        strategy_profile = "L25_TP100"

    assert for_position(_Legacy()) == 100.0


def test_an_unreadable_position_snapshot_fails_closed_to_off():
    """An unverifiable number must not be allowed to drive a real exit."""

    class _Broken:
        take_profit_pct = "not-a-level"
        production_profile = "L25_CURRENT"

    assert for_position(_Broken()) is None


# ---------------------------------------------------------------------------
# 6. the TP trigger basis is documented and honest
# ---------------------------------------------------------------------------


def test_the_executable_net_quote_basis_is_used_when_available():
    from autopaper.paper_live.take_profit import (
        BASIS_EXECUTABLE_NET_SELL_QUOTE,
        take_profit_trigger_documented,
    )
    # Net proceeds exceed all-in entry cost by 15%, so +10% fires and +20% does
    # not. The fees and impact are ALREADY in the quoted output.
    trig = take_profit_trigger_documented(
        target_pct=10.0, net_sell_proceeds=1.15, all_in_entry_cost=1.00)
    assert trig.fired
    assert trig.basis == BASIS_EXECUTABLE_NET_SELL_QUOTE
    assert trig.return_pct == pytest.approx(15.0)
    assert not take_profit_trigger_documented(
        target_pct=20.0, net_sell_proceeds=1.15,
        all_in_entry_cost=1.00).fired


def test_the_price_basis_is_used_only_when_no_quote_exists():
    from autopaper.paper_live.take_profit import (
        BASIS_ENTRY_FILL_PRICE,
        take_profit_trigger_documented,
    )
    trig = take_profit_trigger_documented(
        target_pct=10.0, entry_fill_price=1.0, price=1.12)
    assert trig.fired
    assert trig.basis == BASIS_ENTRY_FILL_PRICE
    assert trig.return_pct == pytest.approx(12.0)


def test_a_quote_basis_and_a_price_basis_are_never_conflated():
    from autopaper.paper_live.take_profit import (
        BASIS_ENTRY_FILL_PRICE,
        BASIS_EXECUTABLE_NET_SELL_QUOTE,
        take_profit_trigger_documented,
    )
    priced = take_profit_trigger_documented(
        target_pct=10.0, entry_fill_price=1.0, price=1.5)
    quoted = take_profit_trigger_documented(
        target_pct=10.0, net_sell_proceeds=1.5, all_in_entry_cost=1.0)
    assert priced.basis == BASIS_ENTRY_FILL_PRICE
    assert quoted.basis == BASIS_EXECUTABLE_NET_SELL_QUOTE
    assert priced.basis != quoted.basis


def test_a_zero_entry_cost_never_produces_a_return():
    """A TP on a cost of zero is not a return, and must not be manufactured."""
    from autopaper.paper_live.take_profit import net_quote_return_pct
    assert net_quote_return_pct(net_proceeds=1.0, all_in_entry_cost=0.0) is None
    assert net_quote_return_pct(net_proceeds=None, all_in_entry_cost=1.0) is None


def test_off_and_zero_do_not_fire():
    from autopaper.paper_live.take_profit import take_profit_trigger_documented
    for target in (None, 0, -10):
        trig = take_profit_trigger_documented(
            target_pct=target, entry_fill_price=1.0, price=99.0)
        assert not trig.fired, target


def test_the_trigger_token_names_the_level():
    from autopaper.paper_live.take_profit import trigger_token
    assert trigger_token(100.0) == "tp100"      # historical ledgers
    assert trigger_token(10.0) != trigger_token(100.0)
    assert trigger_token(None) == "tp100"


# ---------------------------------------------------------------------------
# 7. vocabulary: nothing can grow silently
# ---------------------------------------------------------------------------


def test_the_selector_cannot_offer_an_undeclared_payload():
    for row in ui.take_profit_keyboard(10.0):
        payload = row[0]["callback_data"]
        if payload in (ui.CB_TAKE_PROFIT_CUSTOM, ui.CB_TAKE_PROFIT_OFF,
                      ui.CB_STATS, ui.CB_MENU):
            continue
        assert payload.startswith(ui.CB_TAKE_PROFIT_SELECT), payload
        level = payload[len(ui.CB_TAKE_PROFIT_SELECT):]
        # every payload maps to a selectable level
        normalise(level)


def test_the_tp_typed_command_is_declared():
    assert "/tp" in TEXT_COMMANDS
    assert TEXT_COMMANDS["/tp"] == ui.CB_TAKE_PROFIT
    assert "/tp" in ui.OPERATOR_COMMANDS


def test_a_typed_tp_command_routes_to_the_same_validating_handler():
    for text, expected in (("/tp 12.5", "take_profit:12.5"),
                           ("/tp +30%", "take_profit:30"),
                           ("/tp off", "take_profit:off"),
                           ("/status", "stats")):
        payload, _ = payload_from_update(
            {"update_id": 1, "message": {"text": text}})
        assert payload == expected, (text, payload)


def test_arbitrary_text_cannot_reach_a_take_profit_level():
    """The typed path is a NUMBER, never a free string that could be a command."""
    for text in ("/tp ; rm -rf /", "/tp /start", "/tp start",
                 "hello", "/tp,10"):
        payload, _ = payload_from_update(
            {"update_id": 1, "message": {"text": text}})
        if payload is not None:
            assert payload == "take_profit" or payload.startswith(
                ui.CB_TAKE_PROFIT_SELECT), (text, payload)
            if payload.startswith(ui.CB_TAKE_PROFIT_SELECT):
                # every accepted value is a valid level, refused otherwise
                try:
                    normalise(payload[len(ui.CB_TAKE_PROFIT_SELECT):])
                except TakeProfitError:
                    pytest.fail(f"{text!r} produced an invalid level")


def test_a_valid_level_round_trips_through_normalise():
    for pct in (*SPEC_PRESETS, 12.5, 37.0, 0.5, TAKE_PROFIT_MAX_PCT):
        assert normalise(pct) == pytest.approx(pct)
        assert normalise(str(pct)) == pytest.approx(pct)
        assert normalise(f"{pct}%") == pytest.approx(pct)


def test_off_is_normalised_to_none_never_to_zero():
    for raw in (None, "", "OFF", "off", "none", "DISABLED"):
        assert normalise(raw) is NO_TAKE_PROFIT, raw


def test_label_off_is_never_blank():
    assert label(None) == "OFF"
    assert label(10.0) == "+10%"
