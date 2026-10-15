"""
The PumpView 'Amarih' paired-exit PAPER experiment -- focused regression suite.

Covers §13 scenarios 1-5 and 19-22 (selector, default, gate, L25/beta, state,
no-signing). Scenarios 6-18 are covered in the second half of this file.

Everything is offline: no network, no Telegram, no RPC, nothing signed or
broadcast. PAPER only.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from autopaper.modes import RuntimeState, validate_strategy_selection
from autopaper.paper_live.amarih_experiment import (
    AMARIH_ARMS,
    ENTRY_VALID,
    AmarihArm,
    AmarihExperiment,
    AmarihSignalRecord,
    EntryObservation,
)
from autopaper.paper_live.arms import (
    AMARIH_GROUP,
    BETA_GROUP,
    ResearchArm,
    parse_arms,
)
from autopaper.strategy_profile import (
    GATE_AMARIH,
    KNOWN_ENTRY_GATES,
    PROFILE_ENTRY_STRATEGY,
    StrategyProfile,
)
from autopaper.telegram import ui

NOW = 1_800_000_000.0
MINT = "M" * 40

def _sig(sid: str = "S1", ts: float = NOW, status: str = "ACCEPTED",
         mint: str = MINT):
    from autopaper.paper_live.amarih_experiment import AmarihSignalRecord
    return AmarihSignalRecord(
        signal_id=sid, mint=mint, symbol="X", strategy="Amarih",
        signal_ts=ts, receipt_ts=ts + 0.5,
        signal_to_receipt_latency_ms=500.0,
        market_cap_signal_usd=20000.0, buy_score=7.0,
        owner_wallet="tg:1", status=status)


def _entry(sid: str = "S1", ts: float = NOW + 1.0, qty: float = 1000.0,
           cost: float = 10.0, valid: str = "VALID"):
    from autopaper.paper_live.amarih_experiment import EntryObservation
    return EntryObservation(
        signal_id=sid, mint=MINT, symbol="X", decision_ts=ts - 0.4,
        entry_ts=ts, signal_to_entry_latency_ms=1000.0,
        market_id="pool1", provider="jupiter", venue="PUMP",
        quote_ts=ts - 0.1, quote_age_ms=100.0, input_amount_usd=cost,
        token_quantity=qty, token_decimals=6, fees_usd=0.1,
        quote_asset="SOL", entry_basis="EXECUTABLE_QUOTE_SIMULATED_FILL",
        market_cap_usd=20000.0, liquidity_usd=5000.0, volume_to_mcap=1.2,
        validation=valid,
        rejection_reason=None if valid == "VALID" else "no executable quote")


def _quote(proceeds: float, qts: float, **kw) -> dict:
    q = {"proceeds_usd": proceeds, "quote_ts": qts,
         "valid_market": True, "provider": "jupiter",
         "venue": "PUMP", "fees_usd": 0.05}
    q.update(kw)
    return q


def test_selector_renders_amarih_row():
    st = RuntimeState()
    st.select_strategies(["L25_CURRENT"])
    kb = ui.strategy_keyboard(st)
    rows = [(b["text"], b["callback_data"]) for r in kb for b in r]
    amarih = [r for r in rows if r[1] == f"profile:{AMARIH_GROUP}"]
    assert amarih, "the strategy keyboard must contain an AMARIH row"
    assert "Amarih Research" in amarih[0][0]
    assert "Amarih Research" in ui.render_strategy_menu(st)


def test_amarih_is_disabled_by_default():
    st = RuntimeState()
    assert st.arms == ("L25_CURRENT",)
    assert st.effective_entry_gates == ("L25",)
    names = {s.name.value for s in parse_arms([])}
    assert "AMARIH" not in names
    assert not any(v.startswith("AMARIH_T") for v in names)


def test_selecting_amarih_activates_only_its_gate():
    st = RuntimeState()
    ok, _ = st.select_strategies(["AMARIH"])
    assert ok
    assert set(st.arms) == {"AMARIH_T60", "AMARIH_T120", "AMARIH_T300"}
    assert st.effective_entry_gates == (GATE_AMARIH,)
    ok, _ = st.select_strategies(["L25_CURRENT", "AMARIH"])
    assert ok
    assert st.effective_entry_gates == ("L25", "AMARIH")


def test_l25_profiles_and_gates_unchanged():
    assert set(PROFILE_ENTRY_STRATEGY.values()) == {"L25", "V2_FLOOR"}
    for p in (StrategyProfile.L25_CURRENT, StrategyProfile.L25_60S,
              StrategyProfile.L25_MAX_HOLD, StrategyProfile.L25_TP100):
        assert PROFILE_ENTRY_STRATEGY[p] == "L25"
    assert GATE_AMARIH in KNOWN_ENTRY_GATES


def test_v2_floor_beta_is_not_reintroduced():
    with pytest.raises(ValueError, match="disabled"):
        parse_arms([BETA_GROUP])
    for tok in ("V2_BETA_A", "V2_BETA_B", "V2_BETA_C"):
        with pytest.raises(ValueError, match="disabled"):
            parse_arms([tok])
    ok, msg, _ = validate_strategy_selection([BETA_GROUP])
    assert ok is False and "disabled" in msg
    specs = parse_arms(["AMARIH"])
    assert {s.name for s in specs} == set(AMARIH_ARMS)
    assert not any("V2_BETA" in s.name.value for s in specs)


# ---------------------------------------------------------------------------
# 6-10: paired entry and deadlines share one observation; SELL the entry qty
# ---------------------------------------------------------------------------

def test_one_signal_creates_three_paired_arms():
    exp = AmarihExperiment(arm_notional_usd=10.0, max_signal_age_s=1e9)
    assert exp.note_signal(_sig("S1")) is True
    entry = _entry("S1", ts=NOW + 1.0)
    opened = exp.open_from_entry(entry)
    assert {a for a in opened} == set(AMARIH_ARMS)
    assert len(exp.open_positions()) == 3


def test_all_arms_share_the_same_immutable_entry_observation():
    exp = AmarihExperiment(arm_notional_usd=10.0, max_signal_age_s=1e9)
    exp.note_signal(_sig("S1"))
    entry = _entry("S1", ts=NOW + 1.0, qty=1234.5)
    opened = exp.open_from_entry(entry)
    qtys = {opened[a].token_quantity for a in opened}
    assert qtys == {1234.5}
    basis = {opened[a].entry_basis for a in opened}
    assert basis == {"EXECUTABLE_QUOTE_SIMULATED_FILL"}
    ts = {opened[a].entry_ts for a in opened}
    assert ts == {NOW + 1.0}
    costs = {opened[a].entry_cost_usd for a in opened}
    assert costs == {10.0}


def test_arm_quantities_and_settlements_are_independent():
    exp = AmarihExperiment(arm_notional_usd=10.0, max_signal_age_s=1e9)
    exp.note_signal(_sig("S1"))
    opened = exp.open_from_entry(_entry("S1", ts=NOW + 1.0))
    # Mutating one arm's settled PnL must not touch the others.
    dl = opened[AmarihArm.T60].scheduled_deadline
    exp.settle_arm(signal_id="S1", arm=AmarihArm.T60, observed_ts=dl,
                   sell_quote=_quote(12.0, dl - 0.5))
    assert round(exp.accounts[AmarihArm.T60].bankroll_usd, 6) == 12.0
    assert round(exp.accounts[AmarihArm.T120].bankroll_usd, 6) == 10.0
    assert round(exp.accounts[AmarihArm.T300].bankroll_usd, 6) == 10.0
    assert len(exp.open_positions()) == 2  # T120 + T300 still open


def test_scheduled_deadlines_measured_from_entry_timestamp():
    exp = AmarihExperiment(arm_notional_usd=10.0, max_signal_age_s=1e9)
    exp.note_signal(_sig("S1"))
    opened = exp.open_from_entry(_entry("S1", ts=NOW + 1.0))
    assert opened[AmarihArm.T60].scheduled_deadline == NOW + 1.0 + 60.0
    assert opened[AmarihArm.T120].scheduled_deadline == NOW + 1.0 + 120.0
    assert opened[AmarihArm.T300].scheduled_deadline == NOW + 1.0 + 300.0


def test_sell_uses_exact_entry_quantity():
    exp = AmarihExperiment(arm_notional_usd=10.0, max_signal_age_s=1e9)
    exp.note_signal(_sig("S1"))
    opened = exp.open_from_entry(_entry("S1", ts=NOW + 1.0, qty=777.0))
    dl = opened[AmarihArm.T60].scheduled_deadline
    r = exp.settle_arm(signal_id="S1", arm=AmarihArm.T60, observed_ts=dl,
                       sell_quote=_quote(11.0, dl - 0.5))
    assert r.token_quantity == 777.0
    assert r.settlement_status == "SETTLED"



# ---------------------------------------------------------------------------
# 4, 5: restart-safe dedup and staleness
# ---------------------------------------------------------------------------

def test_signal_dedup_survives_restart():
    exp = AmarihExperiment(arm_notional_usd=10.0, max_signal_age_s=1e9)
    assert exp.note_signal(_sig("S1")) is True
    assert exp.note_signal(_sig("S1")) is False  # duplicate within process
    # Simulate a restart: a FRESH experiment with the persisted id set.
    exp2 = AmarihExperiment(arm_notional_usd=10.0, max_signal_age_s=1e9)
    exp2.seen_signal_ids = set(exp.seen_signal_ids)
    assert exp2.note_signal(_sig("S1")) is False  # replayed id refused
    assert exp2.note_signal(_sig("S2")) is True   # new id accepted


def test_stale_signals_are_rejected():
    exp = AmarihExperiment(arm_notional_usd=10.0, max_signal_age_s=10.0)
    stale = _sig("OLD")
    stale = stale.__class__(**{**stale.__dict__,
                               "receipt_ts": stale.signal_ts + 3600.0})
    # Stale at intake: refused, never becomes an entry. The wiring contract is
    # that a refused note_signal result means open_from_entry is never called
    # with that signal -- so assert the refusal and that nothing opened.
    assert exp.note_signal(stale) is False
    assert exp.open_positions() == {}
    # The refusal is recorded for audit.
    assert any(r.signal_id == "OLD" for r in exp.signal_records())
    assert "OLD" not in exp.seen_signal_ids


# ---------------------------------------------------------------------------
# 11-13: identity mismatch, missing/stale quote, independent attempts
# ---------------------------------------------------------------------------

def test_same_market_mismatch_blocks_exit():
    exp = AmarihExperiment(arm_notional_usd=10.0, max_signal_age_s=1e9)
    exp.note_signal(_sig("S1"))
    opened = exp.open_from_entry(_entry("S1", ts=NOW + 1.0))
    dl = opened[AmarihArm.T60].scheduled_deadline
    r = exp.settle_arm(signal_id="S1", arm=AmarihArm.T60, observed_ts=dl,
                       sell_quote=_quote(11.0, dl - 0.5, valid_market=False))
    assert r.settlement_status == "IDENTITY_MISMATCH"
    assert r.net_pnl_usd is None
    assert round(exp.accounts[AmarihArm.T60].bankroll_usd, 6) == 10.0


def test_missing_and_stale_quotes_never_become_fills():
    from autopaper.paper_live.amarih_experiment import (
        EXIT_STALE_QUOTE, RESULT_CENSORED)
    exp = AmarihExperiment(arm_notional_usd=10.0, max_signal_age_s=1e9)
    exp.note_signal(_sig("S1"))
    opened = exp.open_from_entry(_entry("S1", ts=NOW + 1.0))
    dl60 = opened[AmarihArm.T60].scheduled_deadline
    dl120 = opened[AmarihArm.T120].scheduled_deadline
    # Missing quote.
    r1 = exp.settle_arm(signal_id="S1", arm=AmarihArm.T60, observed_ts=dl60,
                        sell_quote=None)
    assert r1.settlement_status == "QUOTE_UNAVAILABLE"
    assert r1.net_pnl_usd is None
    # Stale quote (older than the 30s freshness bound).
    r2 = exp.settle_arm(signal_id="S1", arm=AmarihArm.T120, observed_ts=dl120,
                        sell_quote=_quote(11.0, dl120 - 3600.0))
    assert r2.settlement_status == EXIT_STALE_QUOTE
    assert r2.classification == RESULT_CENSORED
    assert r2.net_pnl_usd is None
    # Neither bankroll moved.
    assert round(exp.accounts[AmarihArm.T60].bankroll_usd, 6) == 10.0
    assert round(exp.accounts[AmarihArm.T120].bankroll_usd, 6) == 10.0


def test_failed_exit_does_not_block_other_arms():
    exp = AmarihExperiment(arm_notional_usd=10.0, max_signal_age_s=1e9)
    exp.note_signal(_sig("S1"))
    opened = exp.open_from_entry(_entry("S1", ts=NOW + 1.0))
    dl60 = opened[AmarihArm.T60].scheduled_deadline
    dl120 = opened[AmarihArm.T120].scheduled_deadline
    dl300 = opened[AmarihArm.T300].scheduled_deadline
    # T60 has no quote -> censored. T120 and T300 still settle normally.
    exp.settle_arm(signal_id="S1", arm=AmarihArm.T60, observed_ts=dl60,
                   sell_quote=None)
    r120 = exp.settle_arm(signal_id="S1", arm=AmarihArm.T120, observed_ts=dl120,
                          sell_quote=_quote(11.0, dl120 - 0.5))
    r300 = exp.settle_arm(signal_id="S1", arm=AmarihArm.T300, observed_ts=dl300,
                          sell_quote=_quote(9.0, dl300 - 0.5))
    assert r120.settlement_status == "SETTLED"
    assert r300.settlement_status == "SETTLED"
    assert round(r120.net_return_pct, 6) == 10.0
    assert round(r300.net_return_pct, 6) == -10.0



# ---------------------------------------------------------------------------
# 14-18: emergency classification, failed entries, idempotent settle, report
# ---------------------------------------------------------------------------

def test_emergency_exits_classified_separately():
    from autopaper.paper_live.amarih_experiment import (
        EXIT_EMERGENCY, RESULT_EMERGENCY)
    exp = AmarihExperiment(arm_notional_usd=10.0, max_signal_age_s=1e9)
    exp.note_signal(_sig("S1"))
    opened = exp.open_from_entry(_entry("S1", ts=NOW + 1.0))
    # Emergency exit 10s in (before the 60s deadline).
    r = exp.settle_arm(signal_id="S1", arm=AmarihArm.T60,
                       observed_ts=NOW + 11.0, emergency=True,
                       sell_quote=_quote(9.5, NOW + 10.5))
    assert r.settlement_status == EXIT_EMERGENCY
    assert r.classification == RESULT_EMERGENCY
    assert r.exit_reason == "EMERGENCY_EXIT"
    rep = exp.build_report()
    assert rep["per_arm"]["AMARIH_T60"]["emergency"] == 1
    assert rep["per_arm"]["AMARIH_T60"]["n"] == 1


def test_failed_entries_stay_in_denominator_and_are_counted():
    from autopaper.paper_live.amarih_experiment import ENTRY_QUOTE_UNAVAILABLE
    exp = AmarihExperiment(arm_notional_usd=10.0, max_signal_age_s=1e9)
    exp.note_signal(_sig("S1"))
    exp.note_signal(_sig("S2"))
    # S2's entry validation fails: NO arms open for it.
    opened = exp.open_from_entry(_entry("S2", valid=ENTRY_QUOTE_UNAVAILABLE))
    assert opened == {}
    assert exp.open_positions() == {}
    rep = exp.build_report()
    assert rep["signals_received"] == 2
    assert rep["signals_accepted"] == 2
    assert rep["entry_failures_by_reason"] == {ENTRY_QUOTE_UNAVAILABLE: 1}
    assert rep["valid_entry_observations"] == 0


def test_duplicate_settlement_is_impossible():
    exp = AmarihExperiment(arm_notional_usd=10.0, max_signal_age_s=1e9)
    exp.note_signal(_sig("S1"))
    opened = exp.open_from_entry(_entry("S1", ts=NOW + 1.0))
    dl = opened[AmarihArm.T60].scheduled_deadline
    r1 = exp.settle_arm(signal_id="S1", arm=AmarihArm.T60, observed_ts=dl,
                        sell_quote=_quote(11.0, dl - 0.5))
    # A second, different quote must NOT change the first outcome or re-book.
    r2 = exp.settle_arm(signal_id="S1", arm=AmarihArm.T60, observed_ts=dl + 5,
                        sell_quote=_quote(999.0, dl + 4.5))
    assert r2 is r1
    assert round(exp.accounts[AmarihArm.T60].bankroll_usd, 6) == 11.0


def test_reporting_joins_all_arms_by_signal_id():
    exp = AmarihExperiment(arm_notional_usd=10.0, max_signal_age_s=1e9)
    for sid in ("S1", "S2"):
        exp.note_signal(_sig(sid))
        opened = exp.open_from_entry(_entry(sid, ts=NOW + 1.0))
        for a in opened:
            dl = opened[a].scheduled_deadline
            exp.settle_arm(signal_id=sid, arm=a, observed_ts=dl,
                           sell_quote=_quote(11.0, dl - 0.5))
    rep = exp.build_report()
    assert rep["paired_evaluable_signals"] == 2
    assert rep["per_arm"]["AMARIH_T60"]["n"] == 2
    assert rep["per_arm"]["AMARIH_T120"]["n"] == 2
    assert rep["per_arm"]["AMARIH_T300"]["n"] == 2
    # The three arms are PAIRED, not 3x independent: the paired count is the
    # number of SIGNALS, and the analyzer says so.
    assert rep["paired_evaluable_signals"] == 2 < 6
    assert "PAIRED" in rep["note"]


def test_confidence_blocks_respect_paired_observations():
    # The paired mean is over SIGNAL means, not over arm outcomes pooled: with
    # two signals at (+10,+10,+10) and (-10,-10,-10), the paired mean return is
    # exactly 0.0 across 2 signals -- not over 6 arms.
    exp = AmarihExperiment(arm_notional_usd=10.0, max_signal_age_s=1e9)
    exp.note_signal(_sig("S1"))
    opened = exp.open_from_entry(_entry("S1", ts=NOW + 1.0))
    for a in opened:
        dl = opened[a].scheduled_deadline
        exp.settle_arm(signal_id="S1", arm=a, observed_ts=dl,
                       sell_quote=_quote(11.0, dl - 0.5))
    exp.note_signal(_sig("S2"))
    opened2 = exp.open_from_entry(_entry("S2", ts=NOW + 2000.0))
    for a in opened2:
        dl = opened2[a].scheduled_deadline
        exp.settle_arm(signal_id="S2", arm=a, observed_ts=dl,
                       sell_quote=_quote(9.0, dl - 0.5))
    rep = exp.build_report()
    assert rep["paired_evaluable_signals"] == 2
    assert rep["paired_mean_return"]["n"] == 2
    assert round(rep["paired_mean_return"]["mean"], 6) == 0.0



# ---------------------------------------------------------------------------
# 21-22: no live state mutation by tests, no signing/broadcasting code
# ---------------------------------------------------------------------------

def test_tests_do_not_mutate_live_state(tmp_path):
    import hashlib
    import time as _time

    live_markers = [
        Path("/Users/lnwper/AI-paper-trade/var/paper_state.json"),
        Path("/Users/lnwper/AI-paper-trade/var/paper_account.json"),
        Path("/Users/lnwper/AI-paper-trade/.env.local"),
        Path("/Users/lnwper/AI-paper-trade/var/v2_floor_beta_state.json"),
        Path("/Users/lnwper/AI-paper-trade/var/v2_floor_beta_forward_state.json"),
    ]
    before = {}
    for p in live_markers:
        if p.exists():
            before[p] = (p.stat().st_mtime_ns,
                         hashlib.sha256(p.read_bytes()).hexdigest())
    # Run a full experiment lifecycle; the experiment touches only its own
    # isolated ledgers and (when research_dir is set) its own telemetry dir.
    exp = AmarihExperiment(arm_notional_usd=10.0, max_signal_age_s=1e9,
                           research_dir=tmp_path / "research" / "amarih_experiment")
    exp.note_signal(_sig("S1"))
    opened = exp.open_from_entry(_entry("S1", ts=NOW + 1.0))
    for a in opened:
        dl = opened[a].scheduled_deadline
        exp.settle_arm(signal_id="S1", arm=a, observed_ts=dl,
                       sell_quote=_quote(11.0, dl - 0.5))
    # Telemetry went to the TMP dir only.
    assert (tmp_path / "research" / "amarih_experiment" / "events.jsonl").exists()
    _time.sleep(0.01)
    for p, (mtime, digest) in before.items():
        assert p.stat().st_mtime_ns == mtime, f"{p} was modified"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, \
            f"{p} content changed"


def test_no_signing_or_broadcasting_code():
    banned = {"send_transaction", "sign_transaction", "broadcast",
              "execute_real", "sign_and_send", "send_raw_transaction",
              "solders", "txcodec"}
    for rel in ("src/autopaper/paper_live/amarih_experiment.py",
                "src/autopaper/paper_live/pumpview.py"):
        source = (Path(__file__).resolve().parents[1] / rel)
        tree = ast.parse(source.read_text(encoding="utf-8"))
        called = {n.func.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        imported = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imported |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                imported.add(n.module.split(".")[0])
        assert not (called & banned), f"{rel}: {called & banned}"
        assert not (imported & banned), f"{rel} imports: {imported & banned}"

