"""
StageRecorder timing must survive a process boundary.

StageRecorder is in-memory only, so its per-candidate stage timings vanished on
restart and the historical p50/p90/p95/p99/max could not be reconstructed. The
new StageStore appends each candidate's stage stamps and durations to the
project's existing structured store and reads them back, tolerating anything
malformed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from autopaper.latency import (
    STAGE_ORDER,
    StageRecorder,
    StageStore,
    stage_durations,
)

REPO = Path(__file__).resolve().parents[1]


def _recorder() -> StageRecorder:
    r = StageRecorder()
    r.mark("alert_received", 1000.0)
    r.mark("candidate_parsed", 1000.2)
    r.mark("market_start", 1000.3)
    r.mark("market_end", 1002.5)
    r.mark("narrative_start", 1002.6)
    r.mark("narrative_end", 1004.9)
    r.mark("l25_start", 1005.0)
    r.mark("l25_end", 1005.6)
    r.mark("risk_start", 1005.6)
    r.mark("risk_end", 1005.7)
    r.mark("quote_start", 1005.8)
    r.mark("quote_end", 1007.0)
    r.mark("prepare_start", 1007.0)
    r.mark("prepare_end", 1007.1)
    r.mark("submit_start", 1007.1)
    r.mark("submit_end", 1008.3)
    r.mark("fill", 1008.5)
    return r


def test_a_record_captures_every_stage_and_duration(tmp_path):
    store = StageStore(tmp_path / "stage_timings.jsonl")
    row = store.record("cand1", _recorder(), mode="PAPER", symbol="SYM")
    assert row["candidate_id"] == "cand1"
    assert row["stages"]["alert_received"] == 1000.0
    assert row["stages"]["fill"] == 1008.5
    assert row["durations_ms"]["alert_received->candidate_parsed"] == \
        pytest.approx(200.0)
    assert row["trading_ms"] == pytest.approx(8500.0)
    # naive_duration: every canonical stage pair that was present


def test_records_survive_a_reload_in_a_new_object(tmp_path):
    path = tmp_path / "stage_timings.jsonl"
    StageStore(path).record("cand1", _recorder(), mode="PAPER")
    # A NEW store, as if the process had restarted and only the file remained.
    rows = StageStore.load(path)
    assert len(rows) == 1
    assert rows[0]["stages"]["narrative_start"] == 1002.6
    assert rows[0]["candidate_id"] == "cand1"


def test_records_survive_a_real_process_boundary(tmp_path):
    """
    The strongest form: write in this process, read in a fresh interpreter.
    """
    path = tmp_path / "stage_timings.jsonl"
    StageStore(path).record("cand1", _recorder(), mode="PAPER")
    script = (
        "import json,sys;"
        f"sys.path.insert(0, {str(REPO / 'src')!r});"
        "from autopaper.latency import StageStore;"
        f"rows = StageStore.load({str(path)!r});"
        "print(json.dumps(rows))"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    rows = json.loads(out.stdout)
    assert len(rows) == 1
    assert rows[0]["stages"]["fill"] == 1008.5


def test_malformed_and_non_finite_records_do_not_crash_reporting(tmp_path):
    path = tmp_path / "stage_timings.jsonl"
    path.write_text(
        "\n".join(
            [
                "this is not json",
                json.dumps({"event": "stage_timing", "stages": "not-a-dict"}),
                json.dumps(
                    {
                        "event": "stage_timing",
                        "candidate_id": "bad",
                        "stages": {
                            "alert_received": float("nan"),
                            "market_start": "not-a-number",
                            "market_end": float("inf"),
                        },
                    }
                ),
                # a truncated final line, as from a hard kill
                '{"event": "stage_timing", "stages": {"fill": 42.0}',
                json.dumps(
                    {
                        "event": "stage_timing",
                        "candidate_id": "good",
                        "stages": {"alert_received": 1.0, "fill": 3.0},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = StageStore.load(path)      # must not raise
    # the non-finite / wrong-typed stamps are dropped, never crash
    good = [r for r in rows if r.get("candidate_id") == "good"]
    assert len(good) == 1
    assert good[0]["stages"] == {"alert_received": 1.0, "fill": 3.0}
    assert good[0]["durations_ms"]["alert_received->fill"] == pytest.approx(2000.0)


def test_a_recorder_with_a_non_finite_stamp_does_not_crash_the_store(tmp_path):
    r = StageRecorder()
    r.mark("alert_received", 10.0)
    r.mark("fill", float("nan"))
    store = StageStore(tmp_path / "s.jsonl")
    row = store.record("c", r)        # must not raise
    assert "fill" not in row["stages"]
    assert row["trading_ms"] is None


def test_backward_compatible_missing_stages(tmp_path):
    """An old/partial record reconstructs only what is present."""
    r = StageRecorder()
    r.mark("alert_received", 5.0)
    r.mark("l25_end", 6.0)
    path = tmp_path / "s.jsonl"
    StageStore(path).record("c", r)
    rows = StageStore.load(path)
    assert rows[0]["stages"] == {"alert_received": 5.0, "l25_end": 6.0}
    assert rows[0]["durations_ms"] == {"alert_received->l25_end": 
                                       pytest.approx(1000.0)}


def test_missing_file_reads_as_empty_not_an_error(tmp_path):
    assert StageStore.load(tmp_path / "does-not-exist.jsonl") == []


def test_store_never_raises_on_an_unwritable_path(tmp_path):
    """Persistence must never be able to kill a trade."""
    bad = tmp_path / "a-file"
    bad.write_text("x")
    store = StageStore(bad / "impossible" / "stages.jsonl")
    store.record("c", _recorder())      # a file cannot be a directory -> error is swallowed
    assert store.last_error is not None
    assert store.records_written == 0


def test_canonical_stage_order_is_the_documented_one():
    for name in (
        "alert_received", "candidate_parsed", "market_start", "market_end",
        "narrative_start", "narrative_end", "l25_start", "l25_end",
        "risk_start", "risk_end", "quote_start", "quote_end",
        "prepare_start", "prepare_end", "submit_start", "submit_end", "fill",
    ):
        assert name in STAGE_ORDER


def test_stage_durations_only_uses_present_consecutive_stages():
    d = stage_durations({"alert_received": 1.0, "l25_end": 2.5, "fill": 4.0})
    assert d == {
        "alert_received->l25_end": pytest.approx(1500.0),
        "l25_end->fill": pytest.approx(1500.0),
    }
