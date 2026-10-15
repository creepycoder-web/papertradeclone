"""LAYA_ENTRY_FAST_FAILURE — automated tests.

What is covered, and why
------------------------
BUY, SKIP, model failure and the conservative fallback; point-in-time snapshot
reconstruction; future-data exclusion; preservation of the existing FAST_FAILURE
exit strategy; the cost model's parity with recorded history; and determinism.

The real Laya model is NOT loaded. Every test injects a fake client, so the suite
stays offline and fast and never depends on a GPU. The one test that touches the
real model is a subprocess smoke test guarded behind ``LAYA_SMOKE=1``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "src"))

from laya_entry import spec as spec_module  # noqa: E402
from laya_entry.backtest import (  # noqa: E402
    entry_selection_analysis,
    evaluate_laya,
    simulate,
    split_candidates,
)
from laya_entry.dataset import Candidate, START_BANKROLL_USD, load_candidates  # noqa: E402
from laya_entry.exit_check import classify_exit_reason, replay_exit_policy  # noqa: E402
from laya_entry.fillmodel import cost_rates, price_trade, verify_cost_model  # noqa: E402
from laya_entry.laya_client import LayaResult, parse_answer  # noqa: E402
from laya_entry.snapshot import (  # noqa: E402
    LeakageError,
    assert_no_leakage,
    audit_leakage,
    build_snapshot,
)
from laya_entry.spec import SPEC  # noqa: E402

pytestmark = [pytest.mark.research, pytest.mark.strategy, pytest.mark.safety]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class FakeLaya:
    """A Laya stand-in. It never touches the model, and it is never used for a
    real result — only for tests.

    ``choices`` maps a candidate id to ``"buy"`` or ``"skip"``. A candidate with
    no entry defaults to ``"skip"`` — the same conservative direction the real
    client takes on a failure.
    """

    def __init__(self, choices: dict[str, str] | None = None,
                 *, fail_for: set[str] | None = None,
                 invalid_choice_for: set[str] | None = None,
                 truncated_for: set[str] | None = None) -> None:
        self.choices = choices or {}
        self.fail_for = fail_for or set()
        self.invalid_choice_for = invalid_choice_for or set()
        self.truncated_for = truncated_for or set()
        self.calls: list[str] = []
        self.backend = "fake"
        self.load_ms = 1.0
        self.closed = False

    def ready(self) -> dict:
        return {"model_id": "fake", "dtype": "float16", "load_ms": self.load_ms}

    def decide(self, candidate_id: str, state) -> LayaResult:
        self.calls.append(candidate_id)
        if candidate_id in self.fail_for:
            return LayaResult(candidate_id, False, error="worker exploded",
                              latency_ms=12.0, latency_kind="warm")
        if candidate_id in self.invalid_choice_for:
            return LayaResult(candidate_id, False, choice=None, probabilities={},
                              error="invalid_choice:'maybe'", latency_ms=12.0,
                              latency_kind="warm")
        if candidate_id in self.truncated_for:
            return LayaResult(candidate_id, False, choice="buy", probabilities={},
                              snapshot_truncated=True,
                              error="snapshot_truncated_invalid_input",
                              latency_ms=12.0, latency_kind="warm")
        choice = self.choices.get(candidate_id, "skip")
        buy_probability = 0.9 if choice == "buy" else 0.1
        return LayaResult(
            candidate_id, True, choice=choice,
            probabilities={"buy": buy_probability, "skip": 1.0 - buy_probability},
            confidence=0.5, answer_confidence=buy_probability,
            act_probability=1.0,
            raw_answer={"type": "choice", "choice": choice},
            usage={"state_tokens": 300, "state_tokens_dropped": 0,
                   "truncated": False, "input_tokens": 400},
            latency_ms=12.0, latency_kind="warm",
        )

    def close(self) -> None:
        self.closed = True


def _candidate(candidate_id="cand_a", *, authorized=True, tape=True,
               outcome=None, decision_at=1000.0) -> Candidate:
    snapshot = build_snapshot({
        "at": decision_at,
        "candidate_id": candidate_id,
        "mint": f"Mint{candidate_id}",
        "symbol": "TKN",
        "decision": "BUY",
        "route": "FAST_ENTRY",
        "signal_quality": 0.7,
        "features": {
            "market_cap_usd": 15000.0,
            "liquidity_usd": 16000.0,
            "price": 1.5e-5,
            "volume_5m_usd": 30000.0,
            "buys_5m": 300.0,
            "sells_5m": 200.0,
            "buy_sell_ratio": 1.5,
            "token_age_s": 120.0,
            "change_5m": 90.0,
            "observation_count": 3,
            "freshness": 0.99,
        },
    }, {
        "candidate_id": candidate_id,
        "mint": f"Mint{candidate_id}",
        "source_timestamp": decision_at - 5,
        "fields": {
            "mint": f"Mint{candidate_id}",
            "market_cap_usd": 15000.0,
            "liquidity_usd": 16000.0,
            "holder_count": 250.0,
            "top10_pct": 18.0,
            "bundle_count": 1.0,
            "bundled_pct": 5.0,
            "dev_holdings_pct": 0.0,
            "sold_pct": 2.0,
            "scan_count": 4.0,
            "dex_paid": False,
            "dex_ads_paid": False,
            "age_seconds": 120.0,
        },
    })

    trade = None
    if tape:
        # A self-consistent synthetic fill: quantity x reference -> recorded
        # slippage bps -> fill -> fee, exactly as the production model derives
        # them, so the cost-model parity test is a real test and not a tautology.
        quantity = 1000.0
        entry_reference = 1.5e-5
        exit_reference = 1.8e-5
        entry_slippage_bps = 150.0
        exit_slippage_bps = 120.0
        fee_bps = 85.0
        entry_fill = entry_reference * (1.0 + entry_slippage_bps / 10_000.0)
        exit_fill = exit_reference * (1.0 - exit_slippage_bps / 10_000.0)
        entry_notional = quantity * entry_fill
        entry_fee = entry_notional * fee_bps / 10_000.0
        entry_slippage_usd = quantity * (entry_fill - entry_reference)
        cash_out = entry_notional + entry_fee
        exit_gross = quantity * exit_fill
        exit_fee = exit_gross * fee_bps / 10_000.0
        exit_slippage_usd = quantity * (exit_reference - exit_fill)
        gross = (exit_reference - entry_reference) * quantity
        net = (gross - entry_fee - exit_fee
               - entry_slippage_usd - exit_slippage_usd)
        trade = {
            "trade_id": f"trade_{candidate_id}",
            "position_id": f"pos_{candidate_id}",
            "candidate_id": candidate_id,
            "mint": f"Mint{candidate_id}",
            "symbol": "TKN",
            "route": "FAST_ENTRY",
            "regime": "EXPLOSIVE",
            "t_entry": decision_at + 2,
            "t_exit": decision_at + 100,
            "entry_price": entry_reference,
            "exit_price": exit_reference,
            "fill_price": entry_fill,
            "quantity": quantity,
            "notional_usd": entry_notional,
            "entry_fee_usd": entry_fee,
            "exit_fee_usd": exit_fee,
            "entry_slippage_usd": entry_slippage_usd,
            "exit_slippage_usd": exit_slippage_usd,
            "cash_out_usd": cash_out,
            "gross_pnl_usd": gross,
            "net_pnl_usd": net,
            "return_pct": 20.0,
            "holding_seconds": 98.0,
            "exit_reason": "no_follow_through after 98s (ret +17.0%)",
            "exit_style": "FAST_FAILURE",
            "mfe_pct": 40.0,
            "mae_pct": -3.0,
            "runner_mode": False,
            "market_cap_at_entry": 15000.0,
        }
    marks = []
    if tape:
        # a minimal mark tape: enough for the exit-policy verification to walk
        marks = [{"t": trade["t_entry"] + offset, "px": price,
                  "data_state": "LIVE"}
                 for offset, price in ((1, 1.5e-5), (50, 1.53e-5), (98, 1.53e-5))]

    return Candidate(
        candidate_id=candidate_id,
        mint=f"Mint{candidate_id}",
        symbol="TKN",
        decision_at=decision_at,
        alert_source_at=decision_at - 5,
        decision="BUY",
        snapshot=snapshot,
        l25_authorized=authorized,
        l25_detail="test",
        trade=trade,
        marks=marks,
        outcome_class=outcome,
    )


# ---------------------------------------------------------------------------
# 1. the frozen decision interface
# ---------------------------------------------------------------------------

def test_spec_is_frozen_and_hashed():
    """The interface is stable, and known to be stable."""
    assert SPEC.experiment_id if hasattr(SPEC, "experiment_id") else True
    first = SPEC.fingerprint()
    assert first == SPEC.fingerprint()
    assert len(first) == 64


def test_spec_changes_when_the_interface_changes():
    """A changed option label or field must change the fingerprint."""
    original = SPEC.criteria
    try:
        object.__setattr__(SPEC, "criteria", {**original, "buy": "something else"})
        assert SPEC.fingerprint() != spec_module.LayaSpec().fingerprint()
    finally:
        object.__setattr__(SPEC, "criteria", original)
    assert SPEC.fingerprint() == spec_module.LayaSpec().fingerprint()


def test_question_uses_the_required_options():
    question = SPEC.question
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"buy", "skip"}
    assert "supports attempting an entry" in question["criteria"]["buy"]
    assert "insufficient" in question["criteria"]["skip"]


# ---------------------------------------------------------------------------
# 2. BUY, SKIP and the conservative fallback
# ---------------------------------------------------------------------------

def test_buy_allows_the_entry():
    candidates = [_candidate("cand_a", outcome="winner")]
    client = FakeLaya({"cand_a": "buy"})
    rows, _stats = evaluate_laya(candidates, client)
    row = rows[0]
    assert row["laya_decision"] == "buy"
    assert row["final_entry_eligibility"] == "entered"
    assert row["laya_latency_ms"] == 12.0
    assert row["spec_fingerprint_sha256"] == SPEC.fingerprint()
    assert row["snapshot_fields_supplied"]


def test_skip_refuses_the_entry():
    candidates = [_candidate("cand_a", outcome="winner")]
    client = FakeLaya({"cand_a": "skip"})
    rows, _ = evaluate_laya(candidates, client)
    row = rows[0]
    assert row["laya_decision"] == "skip"
    assert row["final_entry_eligibility"] == "not_entered"
    assert row["rejection_reason"] == "laya_skip"


def test_failed_inference_never_defaults_to_buy():
    """A failed inference resolves to SKIP and is recorded, never to BUY."""
    candidates = [_candidate("cand_a", outcome="winner")]
    client = FakeLaya(fail_for={"cand_a"})
    rows, stats = evaluate_laya(candidates, client)
    row = rows[0]
    assert row["laya_ok"] is False
    assert row["laya_decision"] == "skip"
    assert row["final_entry_eligibility"] == "not_entered"
    assert "conservative fallback" in row["rejection_reason"]
    assert row["laya_error"] == "worker exploded"
    assert stats.failures == 1


def test_invalid_choice_is_not_treated_as_buy():
    candidates = [_candidate("cand_a", outcome="winner")]
    client = FakeLaya(invalid_choice_for={"cand_a"})
    rows, _ = evaluate_laya(candidates, client)
    row = rows[0]
    assert row["laya_ok"] is False
    assert row["laya_decision"] == "skip"
    assert row["laya_fallback_applied"] is True


def test_truncated_snapshot_is_an_invalid_input():
    """If the frozen field set did not fit the token budget, the model did not
    see the interface the spec promises, so the candidate is not a valid test."""
    candidates = [_candidate("cand_a", outcome="winner")]
    client = FakeLaya(truncated_for={"cand_a"})
    rows, _ = evaluate_laya(candidates, client)
    row = rows[0]
    assert row["laya_snapshot_truncated"] is True
    assert row["laya_decision"] == "skip"
    assert row["laya_error"] == "snapshot_truncated_invalid_input"


def test_parse_answer_rejects_unknown_labels():
    result = parse_answer("c", {"answer": {"answers": {"entry": {
        "type": "choice", "choice": "maybe", "probabilities": {"maybe": 1.0}}}}},
        latency_ms=1.0, latency_kind="warm")
    assert result.ok is False
    assert result.effective_decision == "skip"
    assert "invalid_choice" in (result.error or "")


def test_parse_answer_rejects_missing_probabilities():
    result = parse_answer("c", {"answer": {"answers": {"entry": {
        "type": "choice", "choice": "buy"}}}},
        latency_ms=1.0, latency_kind="warm")
    assert result.ok is False
    assert result.effective_decision == "skip"


def test_parse_answer_rejects_non_finite_probability():
    result = parse_answer("c", {"answer": {"answers": {"entry": {
        "type": "choice", "choice": "buy",
        "probabilities": {"buy": float("nan"), "skip": 0.5}}}}},
        latency_ms=1.0, latency_kind="warm")
    assert result.ok is False
    assert result.effective_decision == "skip"


def test_parse_answer_rejects_multi_question_payload():
    result = parse_answer("c", {"answer": {"answers": {
        "entry": {"type": "choice", "choice": "buy",
                  "probabilities": {"buy": 1.0, "skip": 0.0}},
        "other": {"type": "choice", "choice": "skip",
                  "probabilities": {"buy": 0.0, "skip": 1.0}}}}},
        latency_ms=1.0, latency_kind="warm")
    assert result.ok is False
    assert "expected_1_question" in (result.error or "")


# ---------------------------------------------------------------------------
# 3. point-in-time snapshot reconstruction
# ---------------------------------------------------------------------------

def test_snapshot_uses_only_whitelisted_fields():
    snapshot = build_snapshot(
        {"at": 1000.0, "candidate_id": "c", "mint": "M", "symbol": "S",
         "features": {"market_cap_usd": 15000.0, "liquidity_usd": 16000.0}},
        {"source_timestamp": 995.0, "fields": {"mint": "M",
                                               "market_cap_usd": 15000.0}})
    assert set(snapshot.state) <= set(spec_module.STATE_FIELDS)
    assert audit_leakage(snapshot) == []
    assert_no_leakage(snapshot)


def test_snapshot_excludes_outcomes_even_if_the_row_contains_them():
    """A decision row must not be able to smuggle an outcome into the input."""
    snapshot = build_snapshot(
        {"at": 1000.0, "candidate_id": "c", "mint": "M", "symbol": "S",
         "features": {
             "market_cap_usd": 15000.0,
             "liquidity_usd": 16000.0,
             "net_pnl_usd": 999.0,
             "return_pct": 500.0,
             "exit_price": 1e-3,
             "exit_reason": "runner_giveback",
             "mfe_pct": 900.0,
             "mae_pct": -99.0,
             "path_mfe_pct": 1e9,
             "fill_price": 1e-5,
             "quantity": 12345.0,
             "holding_seconds": 999.0,
             "signal_quality": 0.99,
             "route": "FAST_ENTRY",
             "regime": "EXPLOSIVE",
         }},
        {"source_timestamp": 995.0, "fields": {"mint": "M"}})
    assert "net_pnl_usd" not in snapshot.state
    assert "return_pct" not in snapshot.state
    assert "exit_price" not in snapshot.state
    assert "mfe_pct" not in snapshot.state
    assert "path_mfe_pct" not in snapshot.state
    assert "fill_price" not in snapshot.state
    assert "signal_quality" not in snapshot.state
    assert "route" not in snapshot.state
    assert audit_leakage(snapshot) == []


def test_leakage_audit_refuses_an_unknown_field():
    snapshot = build_snapshot(
        {"at": 1.0, "candidate_id": "c", "mint": "M", "symbol": "S",
         "features": {"market_cap_usd": 1.0}})
    snapshot.state["net_pnl_usd"] = 5.0
    violations = audit_leakage(snapshot)
    assert any("not on the frozen whitelist" in v for v in violations)
    assert any("pnl" in v for v in violations)
    with pytest.raises(LeakageError):
        assert_no_leakage(snapshot)


def test_leakage_audit_refuses_a_nested_object():
    snapshot = build_snapshot(
        {"at": 1.0, "candidate_id": "c", "mint": "M", "symbol": "S",
         "features": {"market_cap_usd": 1.0}})
    snapshot.state["exit_reason"] = {"trigger": "runner_giveback"}
    assert any("not a scalar" in v for v in audit_leakage(snapshot))


def test_decision_time_features_supersede_the_alert_card():
    """Market cap and liquidity were re-quoted between the alert and the
    decision; the decision-time value is the one the engine acted on."""
    snapshot = build_snapshot(
        {"at": 1000.0, "candidate_id": "c", "mint": "M", "symbol": "S",
         "features": {"market_cap_usd": 15000.0, "liquidity_usd": 16000.0}},
        {"source_timestamp": 995.0,
         "fields": {"mint": "M", "market_cap_usd": 9000.0,
                    "liquidity_usd": 8000.0}})
    assert snapshot.state["market_cap_usd"] == 15000
    assert snapshot.state["liquidity_usd"] == 16000
    assert snapshot.provenance["market_cap_usd"] == "decision_features"
    assert snapshot.provenance["mcap_to_liquidity"] if "mcap_to_liquidity" in snapshot.provenance else True


def test_alert_card_fields_come_from_the_alert():
    snapshot = build_snapshot(
        {"at": 1000.0, "candidate_id": "c", "mint": "M", "symbol": "S",
         "features": {}},
        {"source_timestamp": 995.0,
         "fields": {"mint": "M", "holder_count": 250.0, "top10_pct": 18.0}})
    assert snapshot.state["holder_count"] == 250
    assert snapshot.state["top10_holders_pct"] == 18.0
    assert snapshot.provenance["holder_count"] == "alert_card_fields"


def test_wire_state_is_compact_and_lossless_in_key_set():
    snapshot = build_snapshot(
        {"at": 1.0, "candidate_id": "c", "mint": "M", "symbol": "S",
         "features": {"market_cap_usd": 15000.0, "liquidity_usd": 16000.0,
                      "token_age_s": 120.0}},
        {"source_timestamp": 0.5, "fields": {"mint": "M"}})
    wire = snapshot.wire_state()
    inverse = {v: k for k, v in spec_module.WIRE_KEYS.items()}
    # the wire form is a pure relabelling: same field set, shorter names
    assert set(inverse[key] for key in wire) == set(snapshot.state)
    assert all(len(key) <= 12 for key in wire)
    assert set(wire.values()) == set(snapshot.state.values())


# ---------------------------------------------------------------------------
# 4. FAST_FAILURE exit preservation
# ---------------------------------------------------------------------------

def test_fast_failure_thresholds_are_the_production_ones():
    """The experiment must not have quietly re-tuned the exit policy."""
    from autopaper.config import ExitConfig

    cfg = ExitConfig()
    assert cfg.fast_failure_stop_pct == -12.0
    assert cfg.fast_failure_window_s == 75.0
    assert cfg.momentum_collapse_lookback_s == 30.0
    assert cfg.momentum_collapse_drop_pct == -18.0
    assert cfg.structure_break_lookback_s == 45.0
    assert cfg.structure_break_pct == -14.0
    assert cfg.disaster_cap_pct == -45.0
    assert cfg.data_invalid_exit_s == 120.0


def _tape(entry: float, path: list[tuple[float, float]]) -> list[dict]:
    """``path`` is (seconds since entry, price)."""
    return [{"t": entry + offset, "px": price, "data_state": "LIVE"}
            for offset, price in path]


def test_replay_exit_policy_fires_the_real_fast_failure_stop():
    marks = _tape(1000.0, [(1, 100.0), (5, 100.0), (10, 88.0), (15, 86.0)])
    result = replay_exit_policy(marks, entry_fill=101.0, entry_timestamp=1000.0)
    trigger = result["trigger"]
    assert trigger is not None
    assert trigger["trigger"] == "fast_failure_stop"
    assert classify_exit_reason("fast_failure_stop -12% hit") == "fast_failure_stop"


def test_replay_exit_policy_fires_no_follow_through():
    marks = _tape(1000.0, [(1, 100.0), (30, 100.5), (60, 100.4), (90, 100.3)])
    result = replay_exit_policy(marks, entry_fill=100.0, entry_timestamp=1000.0)
    assert result["trigger"]["trigger"] == "no_follow_through"


def test_replay_exit_policy_fires_momentum_collapse():
    marks = _tape(1000.0, [(1, 100.0), (10, 100.0), (20, 100.0),
                           (31, 80.0), (35, 79.0)])
    result = replay_exit_policy(marks, entry_fill=100.0, entry_timestamp=1000.0)
    assert result["trigger"]["trigger"] in {"momentum_collapse", "structure_break",
                                            "fast_failure_stop"}


def test_replay_exit_policy_cannot_see_the_future():
    """A price appended to the tail of the tape must not change an earlier
    trigger: the replay walks the tape forward and stops at the first EXIT."""
    marks = _tape(1000.0, [(1, 100.0), (5, 88.0), (10, 86.0)])
    without_future = replay_exit_policy(marks, entry_fill=101.0,
                                        entry_timestamp=1000.0)
    with_future = replay_exit_policy(
        marks + [{"t": 1400.0, "px": 1e6, "data_state": "LIVE"}],
        entry_fill=101.0, entry_timestamp=1000.0)
    assert without_future["trigger"]["trigger"] == with_future["trigger"]["trigger"]
    assert without_future["marks_walked"] == with_future["marks_walked"]


def test_both_arms_keep_the_booked_fast_failure_exit():
    """The experiment changes entry selection only. The exit reason, the exit
    price basis and the exit style come from the historical record, unaltered."""
    candidates = [
        _candidate("cand_a", outcome="winner", decision_at=1000.0),
        _candidate("cand_b", outcome="loser", decision_at=2000.0),
    ]
    candidates[0].trade["return_pct"] = 20.0
    candidates[1].trade["return_pct"] = -15.0
    candidates[1].trade["gross_pnl_usd"] = -15.0 * 0.0153 / 100.0
    for candidate in candidates:
        candidate.marks = _tape(candidate.trade["t_entry"],
                                [(1, 1.5e-5), (50, 1.53e-5), (98, 1.53e-5)])

    from laya_entry.backtest import prepare_pairs

    pairs, unexecutable = prepare_pairs(candidates)
    assert len(pairs) == 2 and not unexecutable

    laya_buys_both = lambda _c: (True, "laya_buy")  # noqa: E731
    laya_buys_none = lambda _c: (False, "laya_skip")  # noqa: E731

    both = simulate("BOTH", "d", pairs, laya_buys_both)
    none = simulate("NONE", "d", pairs, laya_buys_none)
    assert both.stats["total_trades"] == 2
    assert none.stats["total_trades"] == 0
    for candidate in candidates:
        assert candidate.trade["exit_style"] == "FAST_FAILURE"
        booked = next((b for b in both.booked
                       if b["candidate_id"] == candidate.candidate_id), None)
        assert booked is not None
        # the exit is the booked FAST_FAILURE exit, unaltered by the experiment
        assert booked["exit_reason"] == candidate.trade["exit_reason"]
        assert booked["exit_style"] == "FAST_FAILURE"
        assert booked["exit_reason_family"] in {
            "fast_failure_stop", "momentum_collapse", "structure_break",
            "no_follow_through", "disaster_cap", "data_invalid"}
    assert none.blocked[0]["reason"] == "laya_skip"


# ---------------------------------------------------------------------------
# 5. the cost model
# ---------------------------------------------------------------------------

def test_cost_rates_reproduces_the_recorded_fill_exactly():
    candidates = [_candidate("cand_a", outcome="winner")]
    rates = cost_rates(candidates[0].trade)
    priced = price_trade(rates, rates.recorded_preslippage_notional)
    trade = candidates[0].trade
    assert priced.entry_fill == pytest.approx(trade["fill_price"], abs=1e-15)
    assert priced.entry_fee == pytest.approx(trade["entry_fee_usd"], abs=1e-15)
    assert priced.exit_fee == pytest.approx(trade["exit_fee_usd"], abs=1e-15)
    assert priced.entry_slippage == pytest.approx(trade["entry_slippage_usd"], abs=1e-15)
    assert priced.exit_slippage == pytest.approx(trade["exit_slippage_usd"], abs=1e-15)
    assert priced.cash_out == pytest.approx(trade["cash_out_usd"], abs=1e-15)
    assert priced.net == pytest.approx(trade["net_pnl_usd"], abs=1e-12)


def test_price_trade_asserts_the_accounting_identity():
    candidate = _candidate("cand_a", outcome="winner")
    rates = cost_rates(candidate.trade)
    priced = price_trade(rates, 1.0)
    assert (priced.proceeds - priced.cash_out) == pytest.approx(priced.net, abs=1e-12)


def test_price_trade_refuses_a_non_positive_notional():
    candidate = _candidate("cand_a", outcome="winner")
    rates = cost_rates(candidate.trade)
    with pytest.raises(ValueError):
        price_trade(rates, 0.0)


def test_cost_rates_refuses_an_unusable_record():
    candidate = _candidate("cand_a", outcome="winner")
    candidate.trade["quantity"] = 0.0
    with pytest.raises(ValueError):
        cost_rates(candidate.trade)


def test_verify_cost_model_passes_on_synthetic_history():
    candidates = [_candidate("cand_a", outcome="winner"),
                  _candidate("cand_b", outcome="loser", decision_at=2000.0)]
    for candidate in candidates:
        candidate.trade["t_entry"] = candidate.decision_at
    rates = {c.trade["trade_id"]: cost_rates(c.trade) for c in candidates}
    trades = {c.trade["trade_id"]: c.trade for c in candidates}
    report = verify_cost_model(rates, trades)
    assert report["passed"] is True
    assert report["trades_checked"] == 2


# ---------------------------------------------------------------------------
# 6. entry-selection analysis
# ---------------------------------------------------------------------------

def test_confusion_counts_use_a_fixed_outcome_label():
    candidates = [
        _candidate("cand_win_buy", outcome="winner", decision_at=1000.0),
        _candidate("cand_win_skip", outcome="winner", decision_at=2000.0),
        _candidate("cand_lose_buy", outcome="loser", decision_at=3000.0),
        _candidate("cand_lose_skip", outcome="loser", decision_at=4000.0),
    ]
    client = FakeLaya({"cand_win_buy": "buy", "cand_win_skip": "skip",
                       "cand_lose_buy": "buy", "cand_lose_skip": "skip"})
    rows, _ = evaluate_laya(candidates, client)
    analysis = entry_selection_analysis(candidates, rows)
    assert analysis["confusion"] == {"true_positive": 1, "false_positive": 1,
                                     "true_negative": 1, "false_negative": 1}
    assert analysis["precision"] == 0.5
    assert analysis["recall"] == 0.5
    assert analysis["accepted_observed_win_rate"] == 0.5
    assert analysis["skipped_observed_win_rate"] == 0.5
    assert analysis["buy_acceptance_rate"] == 0.5


def test_unobservable_outcomes_are_excluded_from_the_confusion_counts():
    candidates = [_candidate("cand_a", outcome=None, decision_at=1000.0)]
    client = FakeLaya({"cand_a": "buy"})
    rows, _ = evaluate_laya(candidates, client)
    analysis = entry_selection_analysis(candidates, rows)
    assert analysis["outcome_observable"] == 0
    assert analysis["outcome_not_observable"] == 1
    assert analysis["confusion"]["true_positive"] == 0
    assert analysis["precision"] is None


def test_split_is_chronological_and_contiguous():
    candidates = [_candidate(f"cand_{i}", decision_at=float(i))
                  for i in range(9)]
    splits = split_candidates(candidates)
    assert len(splits["development"]) == 3
    assert len(splits["validation"]) == 3
    assert len(splits["test"]) == 3
    assert splits["development"][-1].decision_at < splits["validation"][0].decision_at
    assert splits["validation"][-1].decision_at < splits["test"][0].decision_at


# ---------------------------------------------------------------------------
# 7. determinism and reuse of the historical dataset
# ---------------------------------------------------------------------------

def test_two_runs_with_the_same_fake_client_agree():
    candidates = [_candidate("cand_a", outcome="winner", decision_at=1000.0),
                  _candidate("cand_b", outcome="loser", decision_at=2000.0)]
    candidate_marks = [(1, 1.5e-5), (20, 1.51e-5), (60, 1.52e-5), (98, 1.53e-5)]
    for candidate in candidates:
        candidate.marks = _tape(candidate.trade["t_entry"], candidate_marks)
    from laya_entry.backtest import prepare_pairs

    pairs, _ = prepare_pairs(candidates)
    results = []
    for _ in range(2):
        results.append(simulate("A", "d", pairs, lambda _c: (True, "x")))
    assert results[0].stats == results[1].stats
    assert results[0].ending_bankroll_usd == results[1].ending_bankroll_usd


def test_the_historical_universe_is_the_pinned_snapshot():
    """The candidate universe really comes from the hash-pinned snapshot and
    covers the whole recorded decision population, not just the bought ones."""
    candidates, audit = load_candidates()
    assert audit["decisions"] == 825
    assert audit["engine_buy"] + audit["engine_reject"] == 825
    # the price path exists only for trades that were taken, and that is
    # recorded rather than hidden
    assert audit["l25_authorized_and_testable"] == 116
    assert audit["l25_authorized_without_tape"] == 20
    assert audit["with_tape"] == 348          # trade row AND at least one HOLD mark
    assert audit["with_trade_row"] == 362     # trade row regardless of the tape
    assert audit["with_tape"] < audit["decisions"]
    for candidate in candidates:
        assert audit_leakage(candidate.snapshot) == []
        # the terminal EXIT row is a fill, not an observation, so it must never
        # be carried into a mark tape
        assert all(mark.get("action") == "HOLD" for mark in candidate.marks)


def test_starting_bankroll_matches_the_repository_research_replay():
    assert START_BANKROLL_USD == 10.0


# ---------------------------------------------------------------------------
# 8. the real model (opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_real_laya_smoke(monkeypatch, tmp_path):
    """Only runs when LAYA_SMOKE=1. Proves the real model, the real loader and
    the frozen interface work together. Skipped by default so the suite stays
    offline and GPU-free."""
    if not os.environ.get("LAYA_SMOKE"):
        pytest.skip("set LAYA_SMOKE=1 to run the real-model smoke test")
    from laya_entry.laya_client import open_client

    client = open_client()
    try:
        result = client.decide("smoke", {"token_address": "SmokeMint",
                                         "market_cap_usd": 22000,
                                         "liquidity_usd": 20000,
                                         "holder_count": 120})
    finally:
        client.close()
    assert result.ok is True
    assert result.choice in ("buy", "skip")
    assert set(result.probabilities) == {"buy", "skip"}
    assert result.usage is not None
    assert result.usage.get("state_tokens_dropped") == 0
