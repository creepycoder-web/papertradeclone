"""
The live runtime must actually produce prospective records.

The forward-validation module can be perfect and still useless if nothing calls
it. These tests drive the real Ledger (the single live data path) and assert
that a decision and a closed trade arrive in the forward store with the fields
validation scores on -- so the harness is wired, not merely present.
"""


from __future__ import annotations

import json
from pathlib import Path

import pytest

from autopaper.forward_validation import (
    VALIDATING,
    Dataset,
    build_report,
    load_or_create_deployment,
    read_jsonl,
)
from autopaper.ledger import Ledger

pytestmark = [
    pytest.mark.integrity,
    pytest.mark.safety,
]

T0 = 1_700_000_000.0


class FakeClock:
    def now(self) -> float:
        return T0

    def iso(self, t: float | None = None) -> str:
        return "2023-11-14T22:13:20Z"


@pytest.fixture
def wired(tmp_path: Path):
    dep = load_or_create_deployment(
        tmp_path / "fv", strategy="L25", l25_min_liq_to_mcap=0.9647,
        strategy_hash="sha", force_start_at=T0 - 10,
    )
    from autopaper.forward_validation import ForwardRecorder
    rec = ForwardRecorder(tmp_path / "fv", dep)
    led = Ledger(tmp_path / "ledger.jsonl", tmp_path / "decisions.jsonl",
                 tmp_path / "cand.jsonl", tmp_path / "upd.jsonl",
                 clock=FakeClock(), forward=rec)
    return led, rec, dep


def _trade_dict(**over):
    """A Trade.as_dict() with the REAL field names, taken from the ledger."""
    base = {
        "trade_id": "T1", "position_id": "P1", "candidate_id": "C1",
        "mint": "Mint1", "symbol": "AAA", "strategy": "L25",
        "route": "FAST_ENTRY", "regime": "TRENDING", "signal_quality": 0.9,
        "entry_timestamp": T0, "exit_timestamp": T0 + 45.0,
        "entry_price": 0.001, "exit_price": 0.0012, "fill_price": 0.001,
        "quantity": 155000.0, "return_pct": 20.0, "gross_pnl_usd": 0.031,
        "net_pnl_usd": 0.0285, "holding_seconds": 45.0,
        "mfe_pct": 25.0, "mae_pct": -3.0,
        "entry_fee_usd": 0.00031, "exit_fee_usd": 0.00037,
        "entry_slippage_usd": 0.00015, "exit_slippage_usd": 0.00018,
        "exit_reason": "TARGET", "exit_style": "TARGET",
        "runner_mode": "OFF", "booked_at": T0 + 45.0,
        "market_cap_at_entry": 9000.0, "liquidity_at_entry": 9100.0,
        "source": "telegram", "thesis": "t", "confidence": 0.7,
        "invalidation": "", "liq_bucket": "b", "mc_bucket": "b",
        "age_bucket": "b", "token_age_s_at_entry": 100.0,
        "entry_signal_timestamp": T0, "liquidity_score": 0.5,
    }
    base.update(over)
    return base


def test_a_closed_trade_reaches_the_forward_store(wired):
    led, rec, _ = wired
    led.trade(_FakeTrade(_trade_dict()))
    rows = list(read_jsonl(rec.trades_path))
    assert len(rows) == 1
    r = rows[0]
    assert r["kind"] == "exit"
    assert r["dataset"] == Dataset.PROSPECTIVE_PAPER.value
    assert r["trade_id"] == "T1"
    assert r["mint"] == "Mint1"


def test_the_fields_validation_scores_on_are_populated(wired):
    """Every field the metrics and the report read must survive the mirror."""
    led, rec, _ = wired
    led.trade(_FakeTrade(_trade_dict()))
    r = list(read_jsonl(rec.trades_path))[0]
    for field in ("return_pct", "net_pnl_usd", "holding_s", "mfe_pct",
                  "mae_pct", "entry_price", "exit_price", "entry_timestamp",
                  "exit_timestamp", "quantity"):
        assert r.get(field) is not None, f"{field} was lost in the mirror"
    assert "missing_fields" not in r, r.get("missing_fields")


def test_fees_and_slippage_are_summed_across_both_legs(wired):
    """
    Both legs are summed. Reporting only the entry fee would make every trade
    look cheaper than it was, which is exactly the bias a validation report must
    not carry.
    """
    led, rec, _ = wired
    led.trade(_FakeTrade(_trade_dict()))
    r = list(read_jsonl(rec.trades_path))[0]
    assert r["fees_paid_usd"] == pytest.approx(0.00031 + 0.00037)
    assert r["slippage_paid_usd"] == pytest.approx(0.00015 + 0.00018)


def test_strategy_provenance_survives(wired):
    led, rec, _ = wired
    led.trade(_FakeTrade(_trade_dict()))
    r = list(read_jsonl(rec.trades_path))[0]
    assert r["strategy"] == "L25"
    assert r["route"] == "FAST_ENTRY"
    assert r["regime"] == "TRENDING"
    assert r["signal_quality"] == 0.9
    assert r["l25_min_liq_to_mcap"] == 0.9647


def test_a_decision_reaches_the_forward_store(wired):
    led, rec, _ = wired
    led.decision(_FakeReport())
    rows = list(read_jsonl(rec.decisions_path))
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "C1"
    assert rows[0]["dataset"] == Dataset.PROSPECTIVE_PAPER.value


def test_a_candidate_reaches_the_forward_store(wired):
    led, rec, _ = wired
    led.candidate_seen({"candidate_id": "C1", "mint": "Mint1",
                        "route": "FAST_ENTRY",
                        "features": {"market_cap_usd": 9000.0,
                                     "liquidity_usd": 9100.0,
                                     "liq_to_mcap": 1.011,
                                     "signal_quality": 0.9}})
    rows = list(read_jsonl(rec.candidates_path))
    assert len(rows) == 1
    assert rows[0]["liq_to_mcap"] == 1.011
    assert "missing_fields" not in rows[0]


def test_the_end_to_end_path_produces_a_scoreable_report(wired):
    """A real capture, scored by the real report -- no fixtures in between."""
    led, rec, dep = wired
    led.candidate_seen({"candidate_id": "C1", "mint": "Mint1",
                        "route": "FAST_ENTRY",
                        "features": {"market_cap_usd": 9000.0,
                                     "liquidity_usd": 9100.0,
                                     "liq_to_mcap": 1.011,
                                     "signal_quality": 0.9}})
    led.decision(_FakeReport())
    for i in range(40):
        led.trade(_FakeTrade(_trade_dict(
            trade_id=f"T{i}", return_pct=3.0 if i % 3 else -1.0,
            net_pnl_usd=0.004, exit_timestamp=T0 + 60 + i,
            booked_at=T0 + 60 + i)))
    rep = build_report(rec.all_records(), deployment=dep)
    assert rep.n_candidates == 1
    assert rep.n_decisions == 1
    assert rep.n_closed == 40
    assert rep.status is not None
    assert rep.dataset is VALIDATING
    assert rep.metrics.n == 40
    assert rep.metrics.expectancy_pct is not None
    # a real capture must never be reported as "no deployment"
    assert rep.status.value != "NO_DEPLOYMENT"


def test_mirroring_never_raises_into_the_trading_path(wired):
    """
    A broken forward store must not stop a paper trade.

    Validation is evidence ABOUT the strategy; it is not a precondition for
    taking a trade. A recorder that raises is logged and the ledger write that
    preceded it has already succeeded.
    """
    led, rec, _ = wired

    def boom(*a, **k):
        raise RuntimeError("forward store is broken")

    rec.trade = boom
    led.trade(_FakeTrade(_trade_dict()))     # must not raise
    # and the primary ledger write still happened
    rows = list(read_jsonl(led.ledger_path))
    assert any(r.get("event") == "exit" for r in rows)


def test_no_forward_recorder_means_no_forward_writes(tmp_path: Path):
    """The default is off, so nothing is written unless capture is wired on."""
    led = Ledger(tmp_path / "l.jsonl", tmp_path / "d.jsonl",
                 tmp_path / "c.jsonl", tmp_path / "u.jsonl",
                 clock=FakeClock())
    assert led.forward is None
    led.trade(_FakeTrade(_trade_dict()))
    assert not (tmp_path / "fv").exists()


# ---------------------------------------------------------------------------
# engine wiring
# ---------------------------------------------------------------------------


def test_the_engine_wires_the_recorder(tmp_path: Path, monkeypatch):
    from autopaper.config import build_config
    from autopaper.engine import PaperTrader

    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    trader = PaperTrader(build_config(tmp_path))
    try:
        assert trader.forward is not None
        assert trader.forward_deployment is not None
        assert trader.ledger.forward is trader.forward
        assert trader.forward_deployment.l25_min_liq_to_mcap == 0.9647
        assert trader.forward_deployment.strategy == "L25"
    finally:
        trader._release_ownership()


def test_the_engine_does_not_revert_an_existing_boundary(tmp_path: Path,
                                                         monkeypatch):
    """A restart must keep the original boundary, or old data looks prospective."""
    from autopaper.config import build_config
    from autopaper.engine import PaperTrader

    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    first = PaperTrader(build_config(tmp_path))
    original = first.forward_deployment.started_at
    first._release_ownership()

    second = PaperTrader(build_config(tmp_path))
    try:
        assert second.forward_deployment.started_at == original
        assert second.forward_deployment.deployment_id == \
            first.forward_deployment.deployment_id
    finally:
        second._release_ownership()


def test_the_deployment_is_written_next_to_the_other_var_state(tmp_path: Path,
                                                                monkeypatch):
    from autopaper.config import build_config
    from autopaper.engine import PaperTrader

    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    trader = PaperTrader(build_config(tmp_path))
    try:
        assert (tmp_path / "var/forward_validation/deployment.json").exists()
    finally:
        trader._release_ownership()


def test_deployment_stores_no_secret(tmp_path: Path, monkeypatch):
    """The committed-adjacent store must not carry credentials."""
    from autopaper.config import build_config
    from autopaper.engine import PaperTrader

    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    cfg = build_config(tmp_path)
    trader = PaperTrader(cfg)
    try:
        blob = (tmp_path / "var/forward_validation/deployment.json").read_text()
    finally:
        trader._release_ownership()
    for bad in ("PRIVATE", "SECRET", "token", "TOKEN", "key", "KEY",
                "mnemonic", "seed"):
        assert bad not in blob, f"{bad} leaked into deployment.json"


class _FakeTrade:
    def __init__(self, d):
        self._d = d

    def as_dict(self):
        return dict(self._d)

    @property
    def booked_at(self):
        return self._d.get("booked_at")

    @property
    def exit_timestamp(self):
        return self._d.get("exit_timestamp")


class _FakeReport:
    candidate_id = "C1"
    mint = "Mint1"
    symbol = "AAA"
    decided_at = T0

    def as_dict(self):
        return self.to_json_dict()

    def to_json_dict(self):
        # mirrors the real DecisionReport.as_dict() keys that matter here
        return {
            "candidate_id": self.candidate_id, "mint": self.mint,
            "symbol": self.symbol, "decided_at": self.decided_at,
            "decision": "ENTER", "route": "FAST_ENTRY", "signal_quality": 0.9,
            "l25": {"authorized": True, "min_liq_to_mcap": 0.9647},
            "v3c": {"authorized": True},
            "features": {"market_cap_usd": 9000.0, "liquidity_usd": 9100.0,
                         "liq_to_mcap": 1.011},
        }
