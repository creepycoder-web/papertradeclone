"""
Strategy selector: PAPER multi-select, REAL single-select.

The defect these cover: the canonical registry has SELECTABLE arms
(L25_CURRENT, L25_TP100, H2_CAP60) but the selector iterated ``StrategyProfile``,
which has only TWO members, and carried a hard-coded two-entry button map. So
H2_CAP60 -- a real, running arm with its own virtual bankroll -- could not be
selected, and a future arm would have been silently omitted the same way.

PAPER may select 1-3 arms, each with an independent virtual bankroll. REAL may
select exactly ONE, and that is enforced in the shared validator, at the config
boundary, AND at the real-entry path -- the UI is not a security boundary.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from autopaper.modes import RunState, RuntimeState, TradeMode, validate_strategy_selection
from autopaper.paper_live.arms import ARMS, BETA_GROUP, arm_names, arm_specs, parse_arms
from autopaper.strategy_profile import StrategyProfile
from autopaper.telegram.ui import render_strategy_menu, strategy_keyboard
from autopaper.telegram.ui import BETA_GROUP_LABEL

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from tests import paper_bridge_fakes as _fk
except ImportError:
    import paper_bridge_fakes as _fk

alert, bridge, real_report, run = _fk.alert, _fk.bridge, _fk.real_report, _fk.run
FakeQuoteService = _fk.FakeQuoteService

# The experiment-GROUP row(s) in arm_names(): selector options that are NOT
# arms. AMARIH is the enabled Amarih paired-exit group (expands to three arms
# on selection). The disabled V2 FLOOR BETA group never reaches arm_names().
GROUP_TOKENS = {"AMARIH"}

# The SELECTABLE PER-ARM set, read from the CANONICAL REGISTRY so a new arm
# cannot silently fall out of these assertions.
#
# L25_TP100 is deliberately ABSENT: a take-profit is a parameter, not a strategy,
# so the arm that used to be the only way to reach +100% is retired and +100% is
# now one selectable take-profit VALUE (see ``paper_live.take_profit``). The token
# still resolves for historical records; it is simply not offered for new
# selection. ``test_take_profit_objectives.py`` covers that trade explicitly.
SELECTABLE = tuple(n for n in arm_names() if n not in GROUP_TOKENS)
assert "L25_TP100" not in SELECTABLE, (
    "the duplicate TP-100% strategy option is still offered for selection; "
    "take profit is a SETTING, not an arm")
assert "L25_CURRENT" in SELECTABLE
assert "L25_MAX_HOLD" in SELECTABLE, "L25_MAX_HOLD is selectable and must appear"
assert "L25_60S" in SELECTABLE, "L25_60S is selectable and must appear"
# The enabled Amarih experiment GROUP must be offered by the selector, but as
# a group row -- never as an arm spec (it expands to three arms on selection,
# exactly like the disabled beta group's pattern before its removal). It is
# NOT part of arm_names() (that would break the documented
# ``len(arm_names()) == len(arm_specs())`` invariant); it is recognised by
# ``is_selectable_arm`` / ``is_amarih_group`` instead.
from autopaper.paper_live.arms import is_selectable_arm

assert "AMARIH" not in arm_names(), (
    "AMARIH must NOT join arm_names(): it is a group token and arm_names() is "
    "the per-arm vocabulary (len(arm_names()) == len(arm_specs()))")
assert is_selectable_arm("AMARIH"), (
    "the enabled Amarih paired-exit experiment group must be selectable")
assert "AMARIH" not in ARMS, (
    "AMARIH is a GROUP token, not an arm; it must expand to three arms")
assert "AMARIH" not in SELECTABLE, (
    "AMARIH is a group, not a per-arm selection")
assert "V2_FLOOR_BETA" not in arm_names(), (
    "the disabled beta experiment must not appear as selectable")
# NOTE: is_selectable_arm(BETA_GROUP) is intentionally True -- it is the
# recognised selector token whose disabled row renders and fails closed in
# validate_strategy_selection/parse_arms.


def _real_state(arms=("L25_CURRENT",)) -> RuntimeState:
    state = RuntimeState()
    state.run_state = RunState.REAL_ARMED
    state.real_armed = True
    state.selected_arms = tuple(arms)
    return state


def _keyboard_text(state: RuntimeState) -> list[str]:
    return [row[0]["text"] for row in strategy_keyboard(state)]


# 1 -------------------------------------------------------------------------
def test_every_registered_strategy_appears():
    """The registry, not a second hand-written list, drives the selector.

    GROUP tokens (BETA_GROUP is refused outright; AMARIH is the enabled
    experiment group) are selector rows, not ARMS entries -- only per-arm
    names must be present in the arm registry.
    """
    # arm_names() == every selectable row: the per-arm names plus the group.
    assert tuple(n for n in arm_names() if n not in GROUP_TOKENS) == SELECTABLE
    for name in SELECTABLE:
        assert name in ARMS, name
    state = RuntimeState()
    text = "\n".join(_keyboard_text(state))
    for name in SELECTABLE:
        assert name.split("_")[0] in text, name
    # The Amarih experiment group is rendered by its operator label.
    assert "Amarih" in text
    # The extra entry rule is shown, so H2 is recognisable as more than a rename.
    assert "f_volume_to_liquidity" in text


def test_a_new_registry_arm_appears_without_touching_the_ui():
    """
    The selector cannot silently omit a strategy.

    Asserted structurally: the keyboard is built from the registry accessor
    ``arm_specs()`` and from nothing else, so an arm added to the registry
    necessarily appears. That is the property the old hard-coded two-entry
    button dictionary destroyed.
    """
    import inspect

    from autopaper.telegram import ui

    source = inspect.getsource(ui.strategy_keyboard)
    assert "arm_specs()" in source, "the keyboard no longer reads the registry"
    # It must NOT iterate the profile enum, which is what hid H2_CAP60.
    assert "for profile in StrategyProfile" not in source
    # arm_specs() is every PER-ARM spec and arm_names() is every per-arm name;
    # they must agree exactly (the documented invariant). The selector's
    # experiment-GROUP rows (the enabled Amarih row; beta is refused outright)
    # have no ArmSpec -- they expand to arms on selection -- and are offered
    # by explicit rows in strategy_keyboard, recognised by is_selectable_arm.
    from autopaper.paper_live.arms import is_selectable_arm

    assert len(arm_specs()) == len(arm_names()) == len(SELECTABLE)
    assert set(arm_names()) == {s.name.value for s in arm_specs()}
    assert is_selectable_arm("AMARIH")
    # The beta GROUP token is a RECOGNISED selector token (its disabled row
    # renders); the refusal lives in validate_strategy_selection/parse_arms.
    assert is_selectable_arm(BETA_GROUP)


# 2,3,4 ---------------------------------------------------------------------
def test_paper_single_select_works():
    state = RuntimeState()
    ok, message = state.select_strategies(["L25_MAX_HOLD"])
    assert ok is True, message
    assert state.arms == ("L25_MAX_HOLD",)


def test_paper_multi_select_works():
    state = RuntimeState()
    ok, message = state.select_strategies(["L25_CURRENT", "H2_CAP60"])
    assert ok is True, message
    assert set(state.arms) == {"L25_CURRENT", "H2_CAP60"}


def test_paper_all_registered_work_simultaneously():
    state = RuntimeState()
    ok, _ = state.select_strategies(list(SELECTABLE))
    assert ok is True
    assert state.arms == SELECTABLE
    # The picker's denominator counts the ROWS it offers: one per SELECTABLE
    # arm, PLUS the two experiment-group rows (the disabled V2 FLOOR (BETA)
    # row and the enabled Amarih row). With every ARM selected the counter
    # reads "4/(4+2)". Asserting the number keeps a future group row or arm
    # from being able to slip into the menu without this failing.
    text = "\n".join(_keyboard_text(state))
    assert f"{len(SELECTABLE)}/{len(SELECTABLE) + 2}" in text, text
    assert BETA_GROUP_LABEL in text, (
        "the V2 FLOOR (BETA) group option must be offered by the picker")
    # Take profit is a SETTING now, so L25_TP100 is no longer an ARM.
    assert set(SELECTABLE) == {"L25_CURRENT", "L25_MAX_HOLD", "L25_60S", "H2_CAP60"}


def test_paper_toggle_adds_and_removes():
    state = RuntimeState()
    state.select_strategies(["L25_CURRENT"])
    ok, _ = state.toggle_strategy("H2_CAP60")
    assert ok and set(state.arms) == {"L25_CURRENT", "H2_CAP60"}
    ok, _ = state.toggle_strategy("H2_CAP60")
    assert ok and state.arms == ("L25_CURRENT",)


def test_paper_cannot_deselect_everything():
    state = RuntimeState()
    state.select_strategies(["L25_CURRENT"])
    ok, message = state.toggle_strategy("L25_CURRENT")
    assert ok is False and state.arms == ("L25_CURRENT",)
    assert "at least one" in message.lower()


# 5 -------------------------------------------------------------------------
def test_independent_paper_bankrolls_per_arm():
    """Three arms, three virtual accounts, three separate balances."""
    engine = bridge(arms=parse_arms(list(SELECTABLE)))
    assert set(engine.accounts) == set(SELECTABLE)
    for account in engine.accounts.values():
        assert account.bankroll_usd == 5.0
    records = run(engine.handle_alert(
        snapshot=alert(), report=real_report()))
    assert {r.arm for r in records} == set(SELECTABLE)
    positions = [engine._position_for(r) for r in records if r.traded]
    assert len(positions) == len(SELECTABLE)
    assert len({id(p) for p in positions}) == len(SELECTABLE), "arms shared one position"
    assert len({p.entry_cost_quote_asset for p in positions}) == 1
    for account in engine.accounts.values():
        assert account.has_position

    # An independent exit on ONE arm must not touch the others.
    from autopaper.exits import ExitDecision
    from autopaper.models import ExitStyle

    class _OnlyL25Current:
        def assess(self, position, *, now, mark_price, data_state, **kw):
            if "L25_CURRENT" in getattr(position, "position_id", ""):
                return ExitDecision("EXIT", "probe", ExitStyle.FAST_FAILURE,
                                    trigger="probe")
            return ExitDecision("HOLD", "held", ExitStyle.FAST_FAILURE,
                                trigger="probe")

    engine.exit_policy = _OnlyL25Current()
    engine.recently_exited = {a: {} for a in engine.accounts}
    run(engine.tick())
    assert not engine.accounts["L25_CURRENT"].has_position
    assert engine.accounts["L25_MAX_HOLD"].has_position
    assert engine.accounts["H2_CAP60"].has_position


# 6 -------------------------------------------------------------------------
def test_shared_quote_across_paper_arms():
    """Multi-select must NOT mean three identical provider requests."""
    quotes = FakeQuoteService()
    engine = bridge(arms=parse_arms(list(SELECTABLE)), quotes=quotes)
    records = run(engine.handle_alert(
        snapshot=alert(), report=real_report()))
    assert len(quotes.buy_calls) == 1, (
        f"{len(quotes.buy_calls)} provider requests for one signal across "
        f"{len(SELECTABLE)} arms; they must share a quote snapshot")
    assert engine.quote_cache.stats()["cache_hits"] == len(SELECTABLE) - 1
    timestamps = {r.buy_quote.quote_timestamp for r in records}
    assert len(timestamps) == 1


# 7,8 -----------------------------------------------------------------------
def test_real_requires_exactly_one_strategy():
    ok, message, arms = validate_strategy_selection(["L25_MAX_HOLD"],
                                                   trade_mode=TradeMode.REAL)
    assert ok is True and arms == ("L25_MAX_HOLD",)


def test_real_rejects_multiple_strategies():
    for selection in (["L25_CURRENT", "L25_MAX_HOLD"],
                      ["L25_CURRENT", "H2_CAP60"],
                      list(SELECTABLE)):
        ok, message, arms = validate_strategy_selection(
            selection, trade_mode=TradeMode.REAL)
        assert ok is False, selection
        assert arms == ()
        assert "single-strategy" in message.lower()


def test_real_select_strategies_refuses_a_multi_request():
    state = _real_state()
    ok, message = state.select_strategies(["L25_CURRENT", "L25_MAX_HOLD"])
    assert ok is False
    assert "single-strategy" in message.lower()
    assert len(state.arms) == 1, "a refused selection still changed the state"


def test_real_toggle_replaces_rather_than_accumulates():
    state = _real_state(["L25_CURRENT"])
    ok, _ = state.toggle_strategy("H2_CAP60")
    assert ok is True
    assert state.arms == ("H2_CAP60",), "REAL accumulated a second arm"


# 9 -------------------------------------------------------------------------
def test_malformed_runtime_and_config_cannot_bypass_the_real_rule():
    """
    The UI is not a security boundary.

    A hand-edited state file, a direct validator call and a tampered state
    object must all still be refused for REAL.
    """
    # direct validator
    ok, _, _ = validate_strategy_selection(list(SELECTABLE), trade_mode="REAL")
    assert ok is False

    # a state object carrying an illegal multi-arm REAL selection
    tampered = _real_state(SELECTABLE)
    assert tampered.trade_mode is TradeMode.REAL
    # ``arms`` clamps, and the entry-path gate refuses outright.
    assert len(tampered.arms) == 1
    ok, message, _ = validate_strategy_selection(
        tampered.selected_arms, trade_mode=tampered.trade_mode)
    assert ok is False and "single-strategy" in message.lower()

    # an unknown strategy is refused, never silently dropped
    ok, message, _ = validate_strategy_selection(["L25_CURRENT", "MADE_UP"],
                                                 trade_mode=TradeMode.PAPER)
    assert ok is False and "MADE_UP" in message


def test_unknown_strategy_is_refused_not_ignored():
    state = RuntimeState()
    ok, message = state.select_strategies(["L25_CURRENT", "NOT_AN_ARM"])
    assert ok is False
    assert state.arms != ("L25_CURRENT",) or "NOT_AN_ARM" in message


# 10 ------------------------------------------------------------------------
def test_restart_preserves_paper_strategy_selection(tmp_path):
    from autopaper.config import build_config
    from autopaper.engine import PaperTrader
    from autopaper.modes import RuntimeState

    class _Market:
        async def quote_async(self, mint):
            raise AssertionError("no network in this test")

    cfg = build_config(tmp_path)
    first = PaperTrader(cfg, market=_Market())
    first.state.select_strategies(["L25_CURRENT", "H2_CAP60"])
    first._persist_strategies(list(first.state.arms))

    stored = json.loads((tmp_path / "var" / "strategy_selection.json").read_text())
    assert set(stored["strategies"]) == {"L25_CURRENT", "H2_CAP60"}

    # A fresh process restores the selection.
    second = PaperTrader(cfg, market=_Market())
    assert set(second.state.arms) == {"L25_CURRENT", "H2_CAP60"}


def test_persisted_file_is_validated_on_restore(tmp_path):
    from autopaper.config import build_config
    from autopaper.engine import PaperTrader

    class _Market:
        async def quote_async(self, mint):
            raise AssertionError("no network in this test")

    cfg = build_config(tmp_path)
    (tmp_path / "var").mkdir(parents=True, exist_ok=True)
    (tmp_path / "var" / "strategy_selection.json").write_text(
        json.dumps({"strategies": ["L25_CURRENT", "NOT_AN_ARM"]}))
    trader = PaperTrader(cfg, market=_Market())
    # Refused -> the default (all arms), not the tampered list.
    assert "NOT_AN_ARM" not in trader.state.arms
    # L25_CURRENT should still be valid
    assert "L25_CURRENT" in trader.state.arms


# 11 ------------------------------------------------------------------------
def test_existing_l25_logic_is_unchanged():
    """
    The selector is a selection surface, not a strategy.

    The arms and their production profiles come from the untouched registry, so
    a selection change cannot alter what an arm MEANS.
    """
    # Each arm stamps ITS OWN profile -- no arm borrows another's exit policy,
    # and H2_CAP60 keeps its existing L25_60S profile.
    assert {a.name.value: a.production_profile.value for a in arm_specs()} == {
        "L25_CURRENT": "L25_CURRENT",
        "L25_60S": "L25_60S",
        "L25_MAX_HOLD": "L25_MAX_HOLD",
        "H2_CAP60": "L25_60S",
    }
    # Only H2_CAP60 carries an extra ENTRY rule.
    entry_rules = {a.name.value: a.extra_entry_rule for a in arm_specs()}
    assert entry_rules["H2_CAP60"] == "f_volume_to_liquidity <= 2.0"
    for plain in ("L25_CURRENT", "L25_60S", "L25_MAX_HOLD"):
        assert entry_rules[plain] is None
    # Declaration order is part of the closed vocabulary: the operator menu and
    # the Telegram keyboard render it, so a reordering would silently reshuffle
    # the picker. The PROFILE vocabulary still names the retired TP100 token so
    # historical ledgers and already-open positions keep their meaning, but it is
    # no longer offered as an ARM -- take profit is a selectable setting now.
    # V2_FLOOR_BETA is a first-class PROFILE (it needs one: the beta arms carry
    # their own exit bracket and a flat 10% size, and ``safety`` must recognise
    # it), while V2_FLOOR is the entry strategy the whole experiment reuses.
    assert [p.value for p in StrategyProfile] == [
        "L25_CURRENT", "V2_FLOOR", "L25_60S", "L25_MAX_HOLD", "L25_TP100",
        "V2_FLOOR_BETA"]
    assert "L25_TP100" not in SELECTABLE
    # Selecting must not arm REAL, start the runtime, or clear a stop.
    state = RuntimeState()
    before = (state.run_state, state.real_armed, state.emergency_stop,
              state.entries_blocked)
    state.select_strategies(["L25_MAX_HOLD"])
    assert (state.run_state, state.real_armed, state.emergency_stop,
            state.entries_blocked) == before


def test_menu_states_the_paper_and_real_difference():
    paper = render_strategy_menu(RuntimeState())
    real = render_strategy_menu(_real_state())
    assert "1-3" in paper and "own virtual bankroll" in paper.lower()
    assert "EXACTLY ONE" in real
    assert "never" in real.lower() or "more than one real" in real.lower()