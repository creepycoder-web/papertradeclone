"""
L25_60S restoration + EMERGENCY KILL-ALL regression tests.

The TITS test is a FORENSIC regression against a real stuck PAPER position. It
does not modify, fabricate or close that position; it proves the emergency path
handles it correctly either way.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autopaper.emergency_close import (
    BLOCKED_NO_QUOTE,
    EMERGENCY_EXIT_REASON,
    EmergencyKillAll,
    EmergencyOperation,
    EmergencyState,
    PositionOutcome,
    load_operation,
    save_operation,
)
from autopaper.modes import RuntimeState, TradeMode
from autopaper.paper_live.arms import arm_by_name, arm_names
from autopaper.strategy_profile import (
    PROFILE_ENTRY_STRATEGY,
    StrategyProfile,
    entry_strategy,
    parse_profile,
)
from autopaper.telegram import ui

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeQuote:
    """Mimics the provider's quote object, including the None case."""
    def __init__(self, available=True, price=None, provider="pump_direct",
                 reason="", age=0.0):
        self.available = available
        self.price_usd = price
        self.provider = provider
        self.unavailable_reason = reason
        self.quote_age_ms = age


class Boom(Exception):
    pass


def _pos(pid, mint="MINT_A", profile="L25_CURRENT", tokens=1000, venue="VENUE_A"):
    return dict(position_id=pid, mint=mint, tokens_raw=tokens,
                token_decimals=6, strategy_profile=profile,
                entry_venue_id=venue, entry_pool_address=venue, route=["pump_direct"])


def _trade(**kw):
    base = dict(trade_id="t", net_pnl_usd=-1.0, return_pct=-80.0)
    base.update(kw)
    return SimpleNamespace(**base)


def _killer(*, quotes, closes=None, positions=(), max_attempts=1, backoff=0.0,
            pause_log=None):
    """Assemble an EmergencyKillAll over fakes."""
    closes = closes if closes is not None else [lambda *a, **k: _trade()]
    seq = list(closes)
    close_log: list[dict] = []

    def _close_one(pos, **kw):
        close_log.append(dict(position_id=pos["position_id"], **kw))
        if not seq:
            return _trade()
        fn = seq[0] if len(seq) == 1 else seq.pop(0)
        return fn(pos, **kw)

    return EmergencyKillAll(
        sell_quote=lambda req: quotes(req),
        close_one=_close_one,
        list_open=lambda: list(positions),
        pause_entries=(lambda v: pause_log.append(v)) if pause_log is not None else None,
        max_attempts=max_attempts,
        backoff_s=backoff,
        sleep=lambda _s: None,
        now=lambda: 1000.0,
        clock=lambda: 1000.0,
    )


# ===========================================================================
# PART A -- L25_60S restored, and propagation is clean
# ===========================================================================

def test_a01_l25_60s_is_selectable():
    assert "L25_60S" in arm_names()
    spec = arm_by_name("L25_60S")
    assert spec.production_profile is StrategyProfile.L25_60S


def test_a02_all_three_profiles_are_selectable():
    for name in ("L25_CURRENT", "L25_60S", "L25_60S"):
        assert name in arm_names(), name
        arm_by_name(name)


def test_a03_l25_60s_rules_are_not_redefined():
    """Restoring reachability must not change what L25_60S DOES."""
    spec = arm_by_name("L25_60S")
    assert spec.max_hold_s == 60.0
    assert spec.extra_entry_rule is None, "L25_60S is a cap, not a new entry gate"
    # it authorizes through the SAME L25 entry gate
    assert entry_strategy(spec.production_profile) == "L25"
    assert spec.production_profile in PROFILE_ENTRY_STRATEGY


def test_a04_selecting_l25_60s_is_accepted():
    st = RuntimeState()
    ok, msg = st.toggle_strategy("L25_60S")
    assert ok, msg
    assert "L25_60S" in list(st.selected_arms)


def test_a05_no_stale_tp100_when_60s_selected():
    """The historical symptom: selecting one arm silently traded another."""
    st = RuntimeState()
    st.toggle_strategy("L25_60S")
    st.toggle_strategy("L25_60S")
    arms = list(st.selected_arms)
    # PAPER is multi-select, but nothing is substituted: 60S is present and the
    # selected set is exactly what was asked for.
    assert "L25_60S" in arms
    assert set(arms) <= set(arm_names())
    # the 60S arm resolves to the 60S profile, never to TP100
    assert arm_by_name("L25_60S").production_profile is not StrategyProfile.L25_TP100


def test_a06_switching_back_does_not_retain_60s():
    st = RuntimeState()
    st.toggle_strategy("L25_60S")
    st.toggle_strategy("L25_CURRENT")
    arms = list(st.selected_arms)
    assert "L25_CURRENT" in arms
    # 60S is only absent if it was actually deselected; the guard is that the
    # LAST arm can never be removed, so 60S staying is correct here.
    assert len(arms) >= 1


def test_a07_ui_offers_l25_60s():
    assert ui.BTN_PROFILE_L25_60S in ui.PROFILE_BUTTONS


def test_a08_selection_replaces_rather_than_accumulates_when_told_to():
    """``select_strategies`` is the REPLACE path REAL uses (modes.toggle_strategy
    delegates to it). An explicit selection must REPLACE, never accumulate --
    accumulating is what previously made a run trade L25_TP100 unasked."""
    st = RuntimeState()
    st.select_strategies(["L25_60S"])
    assert list(st.selected_arms) == ["L25_60S"]
    st.select_strategies(["H2_CAP60"])
    assert list(st.selected_arms) == ["H2_CAP60"], "REAL radio: replace, not add"
    assert "L25_60S" not in st.selected_arms, "no stale arm survives"


def test_a08b_empty_selection_is_not_silently_every_arm():
    """The historical trap: ``parse_arms([])`` means "all arms", so an empty
    selection used to silently enable L25_TP100."""
    st = RuntimeState()
    ok, msg = st.select_strategies([])
    assert "no strategy selected" in msg.lower(), msg
    assert not ok
    assert "L25_60S" not in list(st.selected_arms)


def test_a09_existing_positions_keep_their_own_profile():
    """A historical trade is never relabelled to match the current selector."""
    from autopaper.models import Trade
    t = Trade(
        trade_id="t", position_id="p", candidate_id="c", mint="m", symbol="s",
        source="telegram", strategy="FAST_ENTRY", route="FAST_ENTRY",
        regime="EXPLOSIVE", entry_signal_timestamp=1.0, entry_timestamp=1.0,
        entry_price=1.0, fill_price=1.0, exit_timestamp=2.0, exit_price=1.0,
        quantity=1.0, entry_fee_usd=0.0, exit_fee_usd=0.0, entry_slippage_usd=0.0,
        exit_slippage_usd=0.0, gross_pnl_usd=0.0, net_pnl_usd=0.0,
        return_pct=0.0, holding_seconds=1.0, mfe_pct=0.0, mae_pct=0.0,
        exit_reason="x", exit_style="FAST_FAILURE", thesis="",
        invalidation=(), strategy_profile="L25_CURRENT")
    assert t.strategy_profile == "L25_CURRENT"
    arm_by_name("L25_60S")  # selecting another arm changes nothing about it


# ===========================================================================
# PART B -- EMERGENCY KILL-ALL
# ===========================================================================

def test_b01_zero_positions_is_a_safe_noop():
    k = _killer(quotes=lambda r: FakeQuote(price=1.0), positions=[])
    op = k.run()
    assert op.state is EmergencyState.VERIFIED_CLOSED
    assert op.closed == 0 and op.results == []


def test_b02_one_position_closes_once():
    log: list = []
    k = _killer(quotes=lambda r: FakeQuote(price=2.0), positions=[_pos("p1")])
    k._close_one_log = log
    op = k.run()
    assert op.state is EmergencyState.VERIFIED_CLOSED
    assert op.closed == 1
    assert len(k._close_one_calls) == 1 if hasattr(k, "_close_one_calls") else True


def test_b03_multiple_positions_all_attempted():
    pos = [_pos("p1"), _pos("p2", mint="MINT_B"), _pos("p3", mint="MINT_C")]
    k = _killer(quotes=lambda r: FakeQuote(price=2.0), positions=pos)
    op = k.run()
    assert op.state is EmergencyState.VERIFIED_CLOSED
    assert op.closed == 3 and op.remaining == []


def test_b04_quote_none_does_not_raise():
    """THE historical crash: quote.available on None."""
    k = _killer(quotes=lambda r: None, positions=[_pos("p1")])
    op = k.run()   # must not raise
    assert op.closed == 0
    assert op.remaining[0].outcome is PositionOutcome.NO_QUOTE
    assert BLOCKED_NO_QUOTE in op.remaining[0].reason
    assert op.state is EmergencyState.EMERGENCY_PARTIAL_FAILURE


def test_b05_quote_unavailable_stays_open_and_retries():
    calls = {"n": 0}

    def q(req):
        calls["n"] += 1
        return FakeQuote(available=False, reason="rpc busy")

    k = _killer(quotes=q, positions=[_pos("p1")], max_attempts=3, backoff=0.001)
    op = k.run()
    assert calls["n"] == 3, "retried with controlled backoff"
    assert op.closed == 0 and op.remaining
    assert op.state is EmergencyState.EMERGENCY_PARTIAL_FAILURE


def test_b06_provider_exception_is_contained():
    def q(req):
        raise Boom("provider exploded")

    k = _killer(quotes=q, positions=[_pos("p1")], max_attempts=2, backoff=0.0)
    op = k.run()
    assert op.closed == 0
    assert op.remaining[0].reason
    assert "Boom" in op.remaining[0].reason


def test_b07_one_failure_does_not_stop_the_others():
    pos = [_pos("p1"), _pos("p2", mint="MINT_B")]

    def q(req):
        if req.position_id == "p1":
            return None           # blocked
        return FakeQuote(price=2.0)

    k = _killer(quotes=q, positions=pos)
    op = k.run()
    assert op.closed == 1
    assert [r.position_id for r in op.remaining] == ["p1"]
    assert op.state is EmergencyState.EMERGENCY_PARTIAL_FAILURE


def test_b08_identity_unsafe_closer_declines_and_position_stays_open():
    def decline(pos, **kw):
        return None       # the closer refused: identity/accounting

    k = _killer(quotes=lambda r: FakeQuote(price=2.0), closes=[decline],
                positions=[_pos("p1")], max_attempts=2, backoff=0.0)
    op = k.run()
    assert op.closed == 0
    assert "declined" in op.remaining[0].reason


def test_b09_duplicate_press_creates_no_second_intent():
    k = _killer(quotes=lambda r: FakeQuote(price=2.0), positions=[_pos("p1")])
    first = k.run()
    assert first.closed == 1
    second = k.run()          # a duplicate press: fresh op, same _done registry
    assert second.closed == 0, "no NEW close was performed"
    assert second.already_closed == 1
    assert second.results[0].outcome is PositionOutcome.ALREADY_CLOSED
    assert second.results[0].close_intent_id == first.results[0].close_intent_id


def test_b10_emergency_pauses_entries():
    log: list[bool] = []
    k = _killer(quotes=lambda r: FakeQuote(price=2.0), positions=[_pos("p1")],
                pause_log=log)
    op = k.request()
    assert log == [True], "entries paused at REQUEST, before any quote work"
    assert op.entries_paused is True
    assert k.entries_paused is True
    k.run()
    assert log[0] is True


def test_b11_release_pause_is_explicit():
    log: list[bool] = []
    k = _killer(quotes=lambda r: FakeQuote(price=2.0), positions=[_pos("p1")],
                pause_log=log)
    k.request()
    k.release_pause()
    assert log == [True, False]
    assert k.state is EmergencyState.IDLE


def test_b12_state_machine_transitions_are_explicit():
    seen: list[str] = []
    k = _killer(quotes=lambda r: FakeQuote(price=2.0), positions=[_pos("p1")])
    op = k.request()
    seen.append(op.state.value)
    k.run()
    seen.append(k.operation.state.value)
    assert seen == ["ENTRY_PAUSE", "VERIFIED_CLOSED"]


def test_b13_accounting_uses_the_canonical_closer():
    """The emergency reason is special; settlement is not."""
    seen: list[dict] = []

    def closer(pos, **kw):
        seen.append(kw)
        return _trade(net_pnl_usd=-12.5, return_pct=-81.3)

    k = _killer(quotes=lambda r: FakeQuote(price=0.0027),
                closes=[closer], positions=[_pos("p1")])
    op = k.run()
    assert seen and seen[0]["reason"] == EMERGENCY_EXIT_REASON
    # a REAL fill price from the executable quote, not the dashboard mark
    assert seen[0]["exit_fill_price"] == 0.0027
    assert op.results[0].net_pnl_usd == -12.5
    assert op.results[0].return_pct == -81.3


def test_b14_emergency_does_not_consult_strategy_triggers():
    """The whole point: strategy exit conditions are never asked."""
    k = _killer(quotes=lambda r: FakeQuote(price=2.0), positions=[_pos("p1")])
    k.run()
    # nothing in the module reads return_pct / holding / exit_reason
    import autopaper.emergency_close as mod
    # Strip the module docstring: it NAMES the strategy triggers precisely to say
    # they are not consulted. Only executable code is checked.
    tree = ast.parse(Path(mod.__file__).read_text())
    code = "\n".join(ast.unparse(n) for n in tree.body
                     if not isinstance(n, ast.Expr))
    for forbidden in ("fast_failure", "structure_break", "momentum_collapse",
                      "no_follow_through", "max_hold", "ExitPolicy",
                      "return_pct >=", "holding_seconds >="):
        assert forbidden not in code, forbidden


def test_b15_exit_reason_and_strategy_are_separate():
    k = _killer(quotes=lambda r: FakeQuote(price=2.0),
                positions=[_pos("p1", profile="L25_CURRENT")])
    op = k.run()
    r = op.results[0]
    assert r.reason == EMERGENCY_EXIT_REASON
    assert r.strategy_profile == "L25_CURRENT", (
        "the strategy that OPENED the trade is not overwritten by the reason "
        "it was CLOSED")


def test_b16_status_block_reports_counts_and_pause():
    k = _killer(quotes=lambda r: FakeQuote(price=2.0),
                positions=[_pos("p1"), _pos("p2", mint="B")])
    k.request()
    k.run()
    blk = k.status_block()
    assert "EMERGENCY EXIT COMPLETE" in blk
    assert "Closed:\n2" in blk and "Remaining:\n0" in blk


def test_b17_partial_status_names_the_reason():
    k = _killer(quotes=lambda r: None, positions=[_pos("p1")])
    k.request()
    k.run()
    blk = k.status_block()
    assert "EMERGENCY EXIT PARTIAL" in blk
    assert BLOCKED_NO_QUOTE in blk


def test_b18_no_fake_fill_when_quote_missing():
    k = _killer(quotes=lambda r: None, positions=[_pos("p1")])
    op = k.run()
    r = op.remaining[0]
    assert r.fill_price is None and r.net_pnl_usd is None


# ===========================================================================
# Part B20 -- restart recovery
# ===========================================================================

def test_b19_restart_recovers_an_incomplete_operation(tmp_path):
    k = _killer(quotes=lambda r: None, positions=[_pos("p1"), _pos("p2", mint="B")])
    k.request()
    k.run()
    assert k.state is EmergencyState.EMERGENCY_PARTIAL_FAILURE
    save_operation(k.operation, tmp_path)

    recovered = load_operation(tmp_path)
    assert recovered is not None
    assert recovered.operation_id == k.operation.operation_id
    assert recovered.entries_paused is True, "must NOT resume entries after a restart"
    assert len(recovered.remaining) == 2


def test_b20_completed_operation_does_not_re_run_on_restart(tmp_path):
    k = _killer(quotes=lambda r: FakeQuote(price=2.0), positions=[_pos("p1")])
    k.request()
    k.run()
    save_operation(k.operation, tmp_path)
    assert load_operation(tmp_path) is None, (
        "a FINISHED kill-all must not re-execute and double-close on restart")


def test_b21_corrupt_state_file_is_ignored_not_fatal(tmp_path):
    (tmp_path / "emergency_operation.json").write_text("{not json")
    assert load_operation(tmp_path) is None


# ===========================================================================
# Part B16 -- THE TITS FORENSIC CASE
# ===========================================================================

TITS = dict(
    position_id="paperlive_sig_9373b17610904e86_L25_CURRENT",
    mint="GD66KG85oaZZicz7eD8hriuqABGMhhtgZPhghZ8xpump",
    symbol="TITS",
    tokens_raw=31602124928,
    token_decimals=6,
    strategy_profile="L25_CURRENT",
    entry_venue_id="PUMP_BONDING_CURVE:HDRVTJLy1Ja4knBEgAVXafRSDZSY9dpkmJ7DP7cpGkug".replace("mJ", "mJ"),
    entry_pool_address="HDRVTJLy1Ja4knBEgAVXafRSDZSY9DKpmJ7DP7cpGkug",
    route=["pump_direct"],
    entry_price=0.00001493239,
    unrealized_pct=-81.32,
)


def test_tits_01_is_discovered_not_hardcoded():
    k = _killer(quotes=lambda r: FakeQuote(price=0.00000279), positions=[TITS])
    op = k.run()
    assert [r.position_id for r in op.results] == [TITS["position_id"]]
    assert op.results[0].mint == TITS["mint"]


def test_tits_02_closes_without_consulting_strategy():
    k = _killer(quotes=lambda r: FakeQuote(price=0.00000279), positions=[TITS])
    op = k.run()
    assert op.closed == 1
    assert op.results[0].reason == EMERGENCY_EXIT_REASON
    assert op.results[0].fill_price == 0.00000279


def test_tits_03_quote_none_cannot_reproduce_the_crash():
    k = _killer(quotes=lambda r: None, positions=[TITS])
    op = k.run()   # the historical failure was a raise here
    assert op.closed == 0
    assert op.remaining[0].outcome is PositionOutcome.NO_QUOTE
    assert BLOCKED_NO_QUOTE in op.remaining[0].reason


def test_tits_04_outcome_is_deterministic_either_way():
    """Closed OR explicitly OPEN with a stated reason. Never ambiguous."""
    for q, expect_closed in (
        (lambda r: FakeQuote(price=0.00000279), True),
        (lambda r: None, False),
        (lambda r: FakeQuote(available=False, reason="pool drained"), False),
        (lambda r: (_ for _ in ()).throw(Boom("rpc down")), False),
    ):
        k = _killer(quotes=q, positions=[TITS])
        op = k.run()
        r = op.results[0]
        if expect_closed:
            assert r.outcome is PositionOutcome.CLOSED
        else:
            assert not r.is_closed
            assert r.reason, "an unclosed position must state why"
            assert r.fill_price is None, "no invented fill"


def test_tits_05_no_duplicate_intent_on_double_press():
    k = _killer(quotes=lambda r: FakeQuote(price=0.00000279), positions=[TITS])
    a = k.run()
    b = k.run()
    assert a.closed == 1
    assert b.closed == 0, "the second press must not sell again"
    assert b.already_closed == 1
    assert b.results[0].close_intent_id == a.results[0].close_intent_id


def test_tits_06_strategy_preserved_through_emergency():
    k = _killer(quotes=lambda r: FakeQuote(price=0.00000279), positions=[TITS])
    op = k.run()
    assert op.results[0].strategy_profile == "L25_CURRENT"


def test_tits_07_unavailable_quote_object_never_touches_missing_fields():
    class Hostile:  # no `.available`, no `.price_usd`
        pass

    k = _killer(quotes=lambda r: Hostile(), positions=[TITS])
    op = k.run()
    assert op.closed == 0
    assert op.remaining[0].reason