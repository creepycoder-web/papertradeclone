"""
The dashboard is a display surface. It must never be the reason a trade fails.

The live bug this file pins: the dashboard renderer read the last decision with
attribute access (``d.decided_at``) while the engine stores a DICT
(``report.as_dict()``), so every redraw raised
``AttributeError("'dict' object has no attribute 'decided_at'")``. Trading
continued, because the dashboard loop catches -- but the operator was left
staring at a dead display, which is its own kind of failure.

Two rules follow, and both are tested here:

  1. the renderer accepts EITHER representation and produces the same row
  2. a missing or malformed field degrades the row, it does not raise
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path

import pytest

from autopaper.dashboard import build_frame
from autopaper.engine import _last_event_row
from autopaper.models import Decision, DecisionReport, Regime, RiskFlag, Route

REPO = Path(__file__).resolve().parents[1]
ALLOWED = -1001234567890

#: The exact shape the engine stores: report.as_dict().
DICT_DECISION = {
    "candidate_id": "cand_9999613cecd8",
    "mint": "4UWwX3MknFzvdsy4YxjzE5y3WZVQpn7aHU6qxUqaSTNK",
    "symbol": "AI",
    "decision": "REJECT",
    "regime": "EARLY",
    "route": "FAST_ENTRY",
    "signal_quality": 0.75,
    "confidence": 0.05,
    "thesis": "V3-C refused: market cap above limit",
    "risk_flags": [
        {"code": "V3C_REJECTED_MARKET_CAP", "detail": "25257 > 20000"},
        {"code": "SECOND_FLAG", "detail": "x"},
        {"code": "THIRD_FLAG", "detail": "y"},
    ],
    "entry_conditions": [],
    "invalidation_conditions": [],
    "exit_plan": {},
    "preferred_exit_style": "MOMENTUM_FAILURE",
    "decided_at": 1_790_735_187.0,
    "data_state": "LIVE",
    "latency_ms": 34319.07,
    "nar_used": True,
    "route_multiplier": 0.55,
    "route_sample_size": 25,
}


def _object_decision() -> DecisionReport:
    return DecisionReport(
        candidate_id="cand_9999613cecd8",
        mint=DICT_DECISION["mint"],
        symbol="AI",
        decision=Decision.REJECT,
        regime=Regime.EARLY,
        route=Route.FAST_ENTRY,
        signal_quality=0.75,
        confidence=0.05,
        thesis=DICT_DECISION["thesis"],
        risk_flags=(
            RiskFlag(code="V3C_REJECTED_MARKET_CAP", detail="25257 > 20000"),
            RiskFlag(code="SECOND_FLAG", detail="x"),
            RiskFlag(code="THIRD_FLAG", detail="y"),
        ),
        entry_conditions=(),
        invalidation_conditions=(),
        exit_plan={},
        preferred_exit_style=None,
        decided_at=1_790_735_187.0,
        latency_ms=34319.07,
    )


# ---------------------------------------------------------------------------
# 1. a dict-backed decision renders correctly  (the live crash)
# ---------------------------------------------------------------------------


def test_a_dict_backed_decision_renders():
    """The exact input that crashed. It must produce a row, not an exception."""
    row = _last_event_row(DICT_DECISION)
    assert row["symbol"] == "AI"
    assert row["decision"] == "REJECT"
    assert row["latency_ms"] == pytest.approx(34319.07)
    assert row["confidence"] == 0.05
    assert row["route"] == "FAST_ENTRY"


def test_the_dict_renders_in_the_frame():
    frame = build_frame(last_event=_last_event_row(DICT_DECISION))
    assert "REJECT" in frame
    assert "V3C_REJECTED_MARKET_CAP" in frame
    assert "34,319" in frame or "34319" in frame


def test_the_symbol_falls_back_to_the_mint():
    row = _last_event_row({**DICT_DECISION, "symbol": None})
    assert row["symbol"] == DICT_DECISION["mint"]


# ---------------------------------------------------------------------------
# 2. an object-backed decision renders identically
# ---------------------------------------------------------------------------


def test_an_object_backed_decision_renders():
    row = _last_event_row(_object_decision())
    assert row["symbol"] == "AI"
    assert row["decision"] == "REJECT"
    assert row["latency_ms"] == pytest.approx(34319.07)


def test_both_representations_produce_the_same_row():
    """
    The point of a normaliser: one mapping, so the operator cannot be shown two
    different stories depending on which caller built the view.
    """
    from_dict = _last_event_row(DICT_DECISION)
    from_obj = _last_event_row(_object_decision())
    assert from_dict == from_obj


def test_risk_flags_are_capped_at_two():
    """Three flags, two slots: the row is a summary, not a dump."""
    row = _last_event_row(DICT_DECISION)
    assert row["reason"] == "V3C_REJECTED_MARKET_CAP; SECOND_FLAG"
    assert "THIRD_FLAG" not in row["reason"]


# ---------------------------------------------------------------------------
# 3. a missing decided_at does not crash
# ---------------------------------------------------------------------------


def test_a_missing_decided_at_does_not_crash():
    row = _last_event_row({"symbol": "AI", "decision": "WAIT"})
    assert row["at_iso"] is None
    assert row["symbol"] == "AI"


def test_an_empty_decision_dict_does_not_crash():
    assert _last_event_row({})["symbol"] == ""


def test_a_none_decision_does_not_crash():
    assert _last_event_row(None) == {}


def test_an_already_formatted_timestamp_passes_through():
    row = _last_event_row({"decided_at": "2026-09-30T00:00:00Z"})
    assert row["at_iso"] == "2026-09-30T00:00:00Z", (
        "a string timestamp must not be mangled by the numeric formatter"
    )


def test_a_numeric_timestamp_is_formatted_for_humans():
    row = _last_event_row({"decided_at": 1_790_735_187.0})
    assert row["at_iso"].startswith("2026-")
    assert "T" in row["at_iso"]


@pytest.mark.parametrize("junk", [
    {"decision": object()},
    {"risk_flags": "not-a-list"},
    {"risk_flags": [object(), object()]},
    {"decided_at": float("nan")},
    {"latency_ms": "n/a"},
    {"confidence": None},
])
def test_junk_fields_degrade_rather_than_raise(cfg, junk):
    """A display surface must not throw on a field it does not understand."""
    row = _last_event_row({**DICT_DECISION, **junk})
    assert isinstance(row, dict)
    build_frame(last_event=row)          # must not raise


def test_unknown_extra_fields_are_ignored():
    row = _last_event_row({**DICT_DECISION, "some_future_field": object()})
    assert row["symbol"] == "AI"


# ---------------------------------------------------------------------------
# 4. repeated redraws stay clean
# ---------------------------------------------------------------------------


def test_repeated_redraws_produce_no_errors(cfg, monkeypatch, caplog):
    """
    The live symptom was an error on EVERY redraw, so one clean render proves
    nothing. Fifty must be silent.
    """
    from autopaper import dashboard

    class TTY:
        def __init__(self):
            self.buf = ""

        def write(self, s):
            self.buf += s

        def flush(self):
            pass

        def isatty(self):
            return True

    tty = TTY()
    monkeypatch.setattr(dashboard.sys, "stderr", tty)

    trader = _engine(cfg)
    try:
        trader.last_decision = dict(DICT_DECISION)
        for _ in range(50):
            trader.render_dashboard()
    finally:
        trader._release_ownership()

    assert tty.buf.count("MEMEBOT") == 50
    assert tty.buf.count("\x1b[2J") == 50


def test_redraws_survive_a_decision_that_changes_shape(cfg):
    """The runtime swaps in fresh dicts constantly; shape must not matter."""
    from autopaper import dashboard

    trader = _engine(cfg)
    try:
        for decision in (None, {}, DICT_DECISION, _object_decision(),
                         {"symbol": "X"}, {"decided_at": 1.0}):
            trader.last_decision = decision
            trader.render_dashboard()      # must not raise
    finally:
        trader._release_ownership()


# ---------------------------------------------------------------------------
# 5. a dashboard failure cannot stop trading
# ---------------------------------------------------------------------------


def test_a_dashboard_exception_is_caught_and_logged(cfg, monkeypatch, caplog):
    """
    The loop catches, so one bad frame costs a display, not a trade. This is
    what let the live bot keep processing candidates through the crash.
    """
    trader = _engine(cfg)
    try:
        def boom():
            raise AttributeError("'dict' object has no attribute 'decided_at'")

        monkeypatch.setattr(trader, "render_dashboard", boom)
        stop = asyncio.Event()
        with caplog.at_level("ERROR"):
            asyncio.run(asyncio.wait_for(
                trader._dashboard_loop(stop), timeout=0.5))
    except (asyncio.TimeoutError, TimeoutError):
        pass                      # expected: the loop runs until stopped
    finally:
        stop.set()
        trader._release_ownership()
    assert "dashboard failed" in caplog.text.lower()


def test_a_dashboard_failure_leaves_the_trading_state_untouched(cfg, monkeypatch):
    trader = _engine(cfg)
    try:
        trader.state.start()
        before = (trader.state.run_state, trader.state.trade_mode,
                  trader.candidates_seen, trader.rejects)
        monkeypatch.setattr(
            trader, "render_dashboard",
            lambda: (_ for _ in ()).throw(AttributeError("boom")))
        stop = asyncio.Event()
        try:
            asyncio.run(asyncio.wait_for(trader._dashboard_loop(stop),
                                         timeout=0.4))
        except (asyncio.TimeoutError, TimeoutError):
            pass
        stop.set()
        after = (trader.state.run_state, trader.state.trade_mode,
                 trader.candidates_seen, trader.rejects)
        assert after == before, "the dashboard changed trading state"
    finally:
        trader._release_ownership()


def test_the_dashboard_is_not_imported_into_the_decision_path():
    """
    The renderer must not be reachable from analysis, so a display change can
    never alter a decision.
    """
    for name in ("decision", "features", "engine_core", "routes"):
        path = REPO / "src/autopaper/analysis" / f"{name}.py"
        if path.exists():
            assert "dashboard" not in path.read_text(), name


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------


def _engine(cfg):
    from autopaper.engine import PaperTrader

    return PaperTrader(cfg)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "999:FAKEfakefakefake")
    monkeypatch.setenv("TELEGRAM_BOT_CHAT", str(ALLOWED))
    monkeypatch.delenv("REAL_TRADING_ENABLED", raising=False)
    from autopaper.config import build_config

    return build_config(tmp_path)
