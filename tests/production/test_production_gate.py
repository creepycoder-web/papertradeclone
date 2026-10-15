"""Small, deterministic production-readiness gate suite.

This suite is intentionally separate from the broad research/regression suite.
It checks only invariants that must hold before an operator can even consider a
REAL enablement review; it does not enable REAL or contact a provider.
"""

from __future__ import annotations

import json
from pathlib import Path

from autopaper.analysis.features import build_features
from autopaper.models import Observation, ObservationSet
from autopaper.telegram.client import ListenerStatus


ROOT = Path(__file__).resolve().parents[2]


def test_machine_readable_gate_is_fail_closed():
    report = json.loads((ROOT / "reports" / "production_readiness.json").read_text())
    assert report["real_ready"] is False
    assert report["verdict"] == "NOT REAL-READY"
    assert report["real"] is False


def test_short_history_is_explicitly_unavailable():
    t0 = 1_700_000_000.0
    history = ObservationSet([
        Observation(mint="M", observed_at=t0 - 2, received_at=t0 - 2,
                    provider="test", venue="test", price_usd=1.0),
        Observation(mint="M", observed_at=t0, received_at=t0,
                    provider="test", venue="test", price_usd=1.1),
    ])
    features = build_features(
        candidate_id="c", mint="M", decision_time=t0, history=history,
        alert=None, nar=None, stale_after_ms=15_000, unusable_after_ms=45_000,
    )
    assert features.history_span_s == 2
    assert features.momentum_30s_pct is None
    assert features.momentum_60s_pct is None
    assert features.momentum_30s_available is False
    assert features.momentum_60s_available is False
    assert features.momentum_30s_source == "unavailable"


def test_telegram_status_exposes_reconnect_gap_evidence():
    status = ListenerStatus()
    data = status.as_dict()
    for key in (
        "last_received_id", "last_processed_id", "last_connected_at",
        "last_disconnected_at", "gap_detected", "foreign_chat_dropped",
    ):
        assert key in data


def test_production_paths_contain_no_unconditional_mark_force_close():
    source = (ROOT / "src" / "autopaper" / "engine.py").read_text()
    force_close = source[source.index("async def close_all_paper"):source.index("async def flatten")]
    assert "self.market.quote(" not in force_close
    assert "quote.observation" not in force_close
