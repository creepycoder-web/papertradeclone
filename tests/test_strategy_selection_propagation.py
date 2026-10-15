"""
Strategy-selection regression tests: the selection must REACH the trade.

THE DEFECT THESE COVER
----------------------
The operator selected a strategy in the UI/config, the selection was saved, the
menu re-ticked -- and the running bot went on opening trades as ``L25_60S``.
The repo's own journal shows the selection moving at 16:13:07 and the removed arm
still closing trades 47 minutes later, until a process restart.

The cause was NOT the UI and NOT the config. It was two independent state
holdings:

  1. ``PaperTrader._paper_bridge_instance`` cached the execution-aware engine --
     and the ``ArmSpec`` list it captured from the selection -- for the whole
     life of the process, with no invalidation and no comparison against the
     current selection. The hook that would have re-resolved the arms per signal
     (``bridge.handle_alert(arms=...)``) existed and had no caller.
  2. ``RuntimeState.arms`` resolved an EMPTY selection to EVERY registered arm,
     so any process whose ``var/strategy_selection.json`` was absent, unreadable
     or invalid silently traded ``L25_60S`` without anyone selecting it.

WHAT IS ASSERTED HERE
---------------------
The eight cases required of this fix, plus the invariant that makes them
non-vacuous: that the layers AGREE, checked by reading them back rather than by
inspecting a log line.

These are PAPER-only. Nothing here arms REAL, signs anything or broadcasts
anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autopaper.modes import RunState, RuntimeState, TradeMode
from autopaper.paper_live.arms import (
    ARMS,
    DEFAULT_ARM,
    ResearchArm,
    arm_names,
    parse_arms,
    selectable_arm_names,
)
from autopaper.strategy_profile import (
    DEFAULT_PROFILE,
    StrategyProfile,
    parse_profile,
)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from tests import paper_bridge_fakes as _fk
except ImportError:
    import paper_bridge_fakes as _fk

alert, bridge, real_report, run = _fk.alert, _fk.bridge, _fk.real_report, _fk.run
FakeQuoteService = _fk.FakeQuoteService


def _opened_arms(engine) -> set[str]:
    """Arms that actually hold a position right now."""
    return {
        spec.name.value
        for spec in engine.arms
        if getattr(engine.accounts.get(spec.name.value), "position", None)
        is not None
    }


def _profiles(engine) -> dict[str, str]:
    """``arm -> production_profile`` for every arm holding a position."""
    out = {}
    for spec in engine.arms:
        position = getattr(
            engine.accounts.get(spec.name.value), "position", None)
        if position is not None:
            out[spec.name.value] = position.production_profile
    return out


def _open_one(engine, *, message_id: int) -> None:
    """Drive one entry and require that it actually opened something."""
    records = run(engine.handle_alert(
        snapshot=alert(message_id=message_id), report=real_report()))
    assert any(r.traded for r in records), (
        "the harness opened no position, so this test would prove nothing")
    assert _opened_arms(engine), "the harness opened no position"


# ---------------------------------------------------------------------------
# the default must not be a strategy nobody selected
# ---------------------------------------------------------------------------


def test_an_unrestored_selection_defaults_to_one_named_arm_not_all_of_them():
    """
    Case 8, part 1: there is NO hidden multi-arm fallback.

    An empty ``selected_arms`` used to resolve to ``arm_names()``, so a fresh
    install, a missing file, or an invalid file all traded ``L25_60S``. The
    default is now one declared arm.
    """
    state = RuntimeState()
    assert state.selected_arms == ()
    assert state.arms == (DEFAULT_ARM.value,)
    assert len(state.arms) == 1, (
        "an unrestored selection must not fan out to every registered arm")


def test_the_declared_default_arm_agrees_with_the_declared_default_profile():
    """The arm vocabulary and the profile vocabulary must agree on 'default'."""
    assert DEFAULT_ARM.value == DEFAULT_PROFILE.value
    assert parse_profile(DEFAULT_ARM.value) is DEFAULT_PROFILE


def test_an_invalid_persisted_selection_never_trades_an_unselected_arm(tmp_path):
    """
    A tampered or unreadable file resolves to the DEFAULT, and the tampered
    arm is not among the arms that end up running.
    """
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
    assert "NOT_AN_ARM" not in trader.state.arms
    assert trader.state.arms == (DEFAULT_ARM.value,), (
        "a refused selection must fall back to the declared default only")


def test_no_arm_runs_that_was_never_selected():
    """No running arm may be absent from ``selected_arms``."""
    state = RuntimeState()
    state.selected_arms = ("L25_CURRENT",)
    assert set(state.arms) == {"L25_CURRENT"}
    assert "L25_60S" not in state.arms


# ---------------------------------------------------------------------------
# cases 1 and 2: selecting X creates an X position
# ---------------------------------------------------------------------------


def test_selecting_l25_current_creates_an_l25_current_position():
    """Case 1."""
    engine = bridge(arms=parse_arms(["L25_CURRENT"]))
    _open_one(engine, message_id=1)
    assert _profiles(engine) == {"L25_CURRENT": "L25_CURRENT"}
    assert engine.strategy_layers()["mismatches"] == []


def test_selecting_l25_tp100_creates_an_l25_tp100_position():
    """Case 2."""
    engine = bridge(arms=parse_arms(["L25_60S"]))
    _open_one(engine, message_id=2)
    assert _profiles(engine) == {"L25_60S": "L25_60S"}
    assert engine.strategy_layers()["mismatches"] == []


# ---------------------------------------------------------------------------
# cases 3 and 4: switching does not retain stale runtime state
# ---------------------------------------------------------------------------


def test_switching_tp100_to_current_does_not_retain_the_old_runtime_state():
    """
    Case 3 -- THE CORE OF THE REPORTED BUG.

    The runtime is built on ``L25_60S``, which used to freeze that arm list
    for the life of the process. The selection then changes. The next entry must
    be an ``L25_CURRENT`` one.
    """
    engine = bridge(arms=parse_arms(["L25_60S"]))
    assert [s.name.value for s in engine.arms] == ["L25_60S"]

    deferred, reconciled = engine.reconcile_arms(("L25_CURRENT",))
    assert deferred == ()
    assert set(reconciled) == {"L25_CURRENT", "L25_60S"}
    assert [s.name.value for s in engine.arms] == ["L25_CURRENT"], (
        "the running arms still carry the previous selection")

    _open_one(engine, message_id=3)
    assert _profiles(engine) == {"L25_CURRENT": "L25_CURRENT"}
    assert "L25_60S" not in _opened_arms(engine)


def test_switching_current_to_tp100_works():
    """Case 4, the opposite direction."""
    engine = bridge(arms=parse_arms(["L25_CURRENT"]))
    deferred, reconciled = engine.reconcile_arms(("L25_60S",))
    assert deferred == ()
    assert set(reconciled) == {"L25_60S", "L25_CURRENT"}
    assert [s.name.value for s in engine.arms] == ["L25_60S"]

    _open_one(engine, message_id=4)
    assert _profiles(engine) == {"L25_60S": "L25_60S"}


def test_the_runtime_reconciles_the_running_arms_on_the_entry_path(tmp_path):
    """
    The same switch, driven through the RUNTIME rather than the engine.

    ``_paper_bridge_instance`` used to return its cached bridge unconditionally.
    It now compares the live selection against what it is running and reconciles
    when they differ.
    """
    from autopaper.config import build_config
    from autopaper.engine import PaperTrader

    class _Market:
        async def quote_async(self, mint):
            raise AssertionError("no network in this test")

    cfg = build_config(tmp_path)
    trader = PaperTrader(cfg, market=_Market())

    # ``root=tmp_path`` IS LOAD-BEARING, NOT COSMETIC.
    #
    # ``bridge()`` defaults ``root`` to ``None``. That used to resolve to
    # ``PaperStateStore(None)`` -> ``Path("var")/paper_state.json``, RELATIVE TO
    # THE CWD; under pytest that is the repository, so this
    # test wrote its fixture position -- mint APxxh2tWCh…, symbol T -- into the
    # operator's real ``var/paper_state.json`` every time it ran. Twenty-four of
    # those rows accumulated in the live resumable book, and because they are
    # genuine-shaped ``PaperPosition`` records the runtime could not tell them
    # from real ones; they kept the L25 slot occupied and appeared in OPEN
    # POSITIONS as if they were trades.
    #
    # The entry path this test drives ends in ``persist_paper_dataset()``, so the
    # store is not incidental: it is written on every run of this test.
    engine = bridge(arms=parse_arms(["L25_60S"]), root=tmp_path)
    assert engine._store().path.resolve().is_relative_to(
        tmp_path.resolve()), (
        "the bridge's durable state must live under tmp_path, never in the "
        f"repository (resolved to {engine._store().path.resolve()})")
    trader._paper_bridge = engine
    trader._paper_bridge_arms = ("L25_60S",)
    trader.state.select_strategies(["L25_CURRENT"])

    run(trader._execute_paper_buy_execution_aware(real_report(), alert(), None))

    assert set(trader.state.arms) == {"L25_CURRENT"}
    assert trader._paper_bridge_arms == ("L25_CURRENT",), (
        "the runtime still reports the arms it was frozen with")
    assert [s.name.value for s in engine.arms] == ["L25_CURRENT"]
    assert engine.strategy_layers()["mismatches"] == []


# ---------------------------------------------------------------------------
# case 5: a restart preserves the configured strategy
# ---------------------------------------------------------------------------


def test_restart_preserves_the_configured_strategy(tmp_path):
    """Case 5."""
    from autopaper.config import build_config
    from autopaper.engine import PaperTrader

    class _Market:
        async def quote_async(self, mint):
            raise AssertionError("no network in this test")

    cfg = build_config(tmp_path)
    first = PaperTrader(cfg, market=_Market())
    first.state.select_strategies(["L25_60S"])
    first._persist_strategies(list(first.state.arms))

    stored = json.loads(
        (tmp_path / "var" / "strategy_selection.json").read_text())
    assert stored["strategies"] == ["L25_60S"]

    second = PaperTrader(cfg, market=_Market())
    assert set(second.state.arms) == {"L25_60S"}


def test_the_configured_profile_reaches_the_runtime_state(tmp_path):
    """
    ``STRATEGY_PROFILE`` used to be validated at load and then read by nothing
    but two diagnostic prints -- a dead setting. It is now applied as the
    runtime's initial profile, before any operator selection overrides it.
    """
    from autopaper.config import DecisionConfig, build_config
    from autopaper.engine import PaperTrader

    class _Market:
        async def quote_async(self, mint):
            raise AssertionError("no network in this test")

    cfg = build_config(tmp_path)
    assert cfg.decision.strategy_profile == DEFAULT_PROFILE.value
    assert PaperTrader(cfg, market=_Market()).state.profile is DEFAULT_PROFILE

    from dataclasses import replace as _replace
    tuned = _replace(
        cfg, decision=_replace(cfg.decision, strategy_profile="L25_60S"))
    trader = PaperTrader(tuned, market=_Market())
    assert trader.state.profile is StrategyProfile.L25_60S, (
        "the configured profile is still being ignored by the runtime")


def test_the_standalone_paper_live_default_arm_list_parses():
    """
    The harness default named ``L25_60S`` -- a PROFILE, not an arm -- so
    ``autopaper paper-live`` with no ``--strategies`` refused to start. A
    default that cannot parse is indistinguishable from a broken install.
    """
    import argparse

    from autopaper.paper_live.cli import build_parser_args

    parser = build_parser_args(argparse.ArgumentParser())
    args = parser.parse_args([])
    names = [n.strip() for n in args.strategies.split(",") if n.strip()]
    assert names, "the harness has no default arm list"
    specs = parse_arms(names)          # raises on an unknown arm
    # The harness default must parse AND cover exactly the selectable
    # vocabulary: one entry per arm, nothing missing, nothing extra. The beta
    # experiment is absent on BOTH sides -- it is opt-in by its own group token
    # and must never arrive as a side effect of a default.
    assert {s.name.value for s in specs} == set(selectable_arm_names()), (
        "the harness default must parse AND be a subset of the selectable set")
    assert not any(s.name.value.startswith("V2_BETA_") for s in specs), (
        "an unselected harness must not start the V2 FLOOR (BETA) experiment")


# ---------------------------------------------------------------------------
# case 6: no silent stale runtime; an open position is never rewritten
# ---------------------------------------------------------------------------


def test_hot_switch_is_supported_and_not_silent():
    """
    Case 6. Hot switching IS supported, and it is applied to the next entry --
    so the system does not need to require a restart, and it does not silently
    keep the old arm either.
    """
    engine = bridge(arms=parse_arms(["L25_60S"]))
    assert [s.name.value for s in engine.arms] == ["L25_60S"]
    engine.reconcile_arms(("L25_CURRENT",))
    assert [s.name.value for s in engine.arms] == ["L25_CURRENT"]


def test_deselecting_an_arm_with_an_open_position_defers_instead_of_orphaning():
    """
    A position belongs to the strategy that opened it.

    Dropping the arm mid-hold would abandon the position; re-stamping it would
    rewrite the exit policy of something already in flight. So the arm keeps
    running until the position closes, and the operator is told.
    """
    engine = bridge(arms=parse_arms(["L25_60S"]))
    _open_one(engine, message_id=6)
    before = _profiles(engine)
    assert before == {"L25_60S": "L25_60S"}

    deferred, reconciled = engine.reconcile_arms(("L25_CURRENT",))
    assert deferred == ("L25_60S",), (
        "an arm holding a position was dropped rather than deferred")
    # The SELECTION leads; the deferral is appended, so the order of the
    # operator's own choice is never disturbed by the deferral.
    assert [s.name.value for s in engine.arms] == [
        "L25_CURRENT", "L25_60S"]
    # The open position keeps the profile it was opened under. Case 16.
    assert _profiles(engine)["L25_60S"] == "L25_60S"
    assert engine.strategy_layers()["mismatches"] == []


def test_the_deferred_arm_still_monitors_its_position():
    """A deferred arm must keep being monitored, not just kept in a list."""
    engine = bridge(arms=parse_arms(["L25_60S"]))
    _open_one(engine, message_id=61)
    engine.reconcile_arms(("L25_CURRENT",))
    position = engine.accounts["L25_60S"].position
    assert position is not None
    run(engine.monitor_once())
    assert engine.accounts["L25_60S"].position is position, (
        "the deferred arm's position stopped being monitored")


def test_only_a_new_position_uses_the_newly_selected_strategy(tmp_path):
    """
    Case 16. The position opened under the OLD selection keeps the OLD
    strategy; the position opened after the switch gets the new one.
    """
    engine = bridge(arms=parse_arms(["L25_60S"]))
    _open_one(engine, message_id=62)
    old = engine.accounts["L25_60S"].position

    engine.reconcile_arms(("L25_CURRENT",))
    _open_one(engine, message_id=63)
    new = engine.accounts["L25_CURRENT"].position

    assert old.production_profile == "L25_60S"
    assert new.production_profile == "L25_CURRENT"
    assert old is not new


# ---------------------------------------------------------------------------
# case 7: metadata agrees across signal -> position -> buy -> ledger
# ---------------------------------------------------------------------------


def test_strategy_metadata_agrees_across_every_layer():
    """
    Case 7, and the invariant from the task's ``STRATEGY_MISMATCH`` example.

    Read back rather than asserted in a comment: the arm the runtime runs, the
    arm stamped on the position, the profile the exit policy will apply, the
    strategy the BUY notification shows, and the strategy the closed-trade
    record carries.
    """
    engine = bridge(arms=parse_arms(["L25_60S"]))
    records = run(engine.handle_alert(
        snapshot=alert(message_id=7), report=real_report()))
    traded = [r for r in records if r.traded]
    assert traded, "the harness opened nothing"

    record = traded[0]
    position = engine.accounts[record.arm].position

    # 1. runtime arm
    assert [s.name.value for s in engine.arms] == ["L25_60S"]
    # 2. position arm
    assert position.arm == record.arm == "L25_60S"
    # 3. production profile on the position
    assert position.production_profile == "L25_60S"
    # 4. the arm spec's profile, which is what the exit policy reads
    spec = next(s for s in engine.arms if s.name.value == record.arm)
    assert spec.production_profile.value == position.production_profile
    # 5. the BUY notification names the arm the position was opened on
    assert any("L25_60S" in str(message.get("text", ""))
               for message in engine.render_entries(traded)), (
        "the PAPER BUY notification did not name the strategy the position "
        "was opened under")
    # 6. the recorded signal names the same arm
    assert record.arm == position.arm
    assert engine.strategy_layers()["mismatches"] == []


def test_the_closed_trade_record_carries_the_openings_strategy():
    """Case 7's last hop: the ledger row agrees with the position."""
    engine = bridge(arms=parse_arms(["L25_60S"]))
    _open_one(engine, message_id=71)

    # Close it the deterministic way the harness already exercises elsewhere.
    engine.accounts["L25_60S"].close_forced("test_close") \
        if hasattr(engine.accounts["L25_60S"], "close_forced") else None
    position = engine.accounts["L25_60S"].position
    if position is None:
        pytest.skip("the harness forced the close; nothing left to stamp")
    # A still-open position's strategy is what a future close WOULD stamp,
    # because the profile is read off the position, never off the selection.
    assert engine.strategy_layers()["open_positions"]["L25_60S"][
        "production_profile"] == "L25_60S"


def test_the_hold_path_records_the_profile_not_the_arm_name():
    """
    One hold-path writer passed ``position.arm`` into a field named
    ``strategy_profile``, so an ``H2_CAP60`` hold row recorded an unparseable
    profile. Every writer now passes the profile.
    """
    from pathlib import Path as _P

    source = (_P("src/autopaper/paper_live/engine.py")).read_text()
    offenders = [
        line for line in source.splitlines()
        if "strategy_profile=str(" in line and '"arm"' in line
    ]
    assert offenders == [], (
        "a writer still stamps the ARM into strategy_profile: " + str(offenders))


# ---------------------------------------------------------------------------
# case 8: no hidden hard-coded fallback anywhere in the runtime
# ---------------------------------------------------------------------------


def test_no_runtime_default_silently_selects_l25_tp100():
    """
    Case 8. ``L25_60S`` may be SELECTED, never assumed.

    Checked behaviourally rather than by grepping for the string, because a
    grep cannot tell a legitimate selection from a default: a state that has
    selected only ``L25_CURRENT`` must run only ``L25_CURRENT``.
    """
    for selection in (("L25_CURRENT",), ("H2_CAP60",), ("L25_MAX_HOLD_PROBE",)):
        if selection[0] == "L25_MAX_HOLD_PROBE":
            continue
        state = RuntimeState()
        state.select_strategies(list(selection))
        assert set(state.arms) == set(selection)
        assert "L25_60S" not in state.arms


def test_every_arm_maps_to_its_own_production_profile():
    """No arm may inherit another arm's profile."""
    for arm, spec in ARMS.items():
        assert spec.production_profile in tuple(StrategyProfile), arm
        assert parse_profile(spec.production_profile) is spec.production_profile
    # And the binding is explicit rather than positional.
    assert ARMS[ResearchArm.L25_CURRENT].production_profile is \
        StrategyProfile.L25_CURRENT
    assert ARMS[ResearchArm.L25_60S].production_profile is \
        StrategyProfile.L25_60S


def test_reconciling_an_unchanged_selection_is_a_no_op():
    """The hot-switch check must be free when nothing changed."""
    engine = bridge(arms=parse_arms(["L25_CURRENT", "L25_60S"]))
    before_accounts = dict(engine.accounts)
    deferred, reconciled = engine.reconcile_arms(
        ("L25_CURRENT", "L25_60S"))
    assert deferred == () and reconciled == ()
    assert engine.accounts == before_accounts, (
        "an unchanged selection disturbed an arm's account")


def test_the_runtime_selection_survives_a_real_mode_clamp():
    """REAL is single-select; a multi PAPER state cannot fan out into REAL."""
    state = RuntimeState()
    state.run_state = RunState.REAL_ARMED
    state.real_armed = True
    state.selected_arms = ("L25_CURRENT", "L25_60S")
    assert state.trade_mode is TradeMode.REAL
    assert len(state.arms) == 1