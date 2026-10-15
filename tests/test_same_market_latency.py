"""
Latency instrumentation (test items 24 and 25).

Two properties matter:

* every reported duration is a difference of MONOTONIC readings, so a clock
  step cannot invent or destroy a latency;
* an unmeasured stage yields ``None`` (rendered UNKNOWN), never ``0``. A zero
  claims the step was instantaneous, which is a stronger and more wrong
  statement than admitting it was not measured.
"""


from __future__ import annotations

import time

import pytest

from autopaper.same_market.trace import (
    STAGES,
    TELEGRAM_TIMESTAMP_QUANTUM_S,
    StageTrace,
    build_breakdown,
    timed,
)

pytestmark = pytest.mark.same_market


def _full_trace() -> StageTrace:
    """A trace with every stage marked 100ms apart on the monotonic clock."""
    trace = StageTrace()
    t_mono, t_wall = 5000.0, 1_700_000_000.0
    for stage in ("receive", "parse_start", "parse_end", "dex_request_start",
                  "dex_response", "rpc_request_start", "rpc_response",
                  "pool_state", "quote_start", "quote_end", "decision"):
        trace.mark_at(stage, wall=t_wall, mono=t_mono)
        # Monotonic is in SECONDS, so 100ms is 0.1.
        t_mono += 0.1
        t_wall += 0.1
    return trace


# ---------------------------------------------------------------------------
# 24. latency measurement
# ---------------------------------------------------------------------------

def test_the_six_required_figures_are_produced():
    trace = _full_trace()
    breakdown = build_breakdown(trace, telegram_message_ts=trace.wall["receive"] - 0.42)

    for field in ("telegram_to_receive_ms", "receive_to_mark_ms",
                  "mark_to_state_ms", "state_to_quote_ms",
                  "quote_to_decision_ms", "total_signal_to_decision_ms"):
        value = getattr(breakdown, field)
        assert value is not None, field

    assert breakdown.telegram_to_receive_ms == pytest.approx(420.0)
    assert breakdown.total_signal_to_decision_ms == pytest.approx(1000.0)


def test_every_declared_stage_is_representable():
    trace = StageTrace()
    for stage in STAGES:
        trace.mark(stage)
    assert set(STAGES).issubset(trace.order)


def test_a_missing_timestamp_is_unknown_not_zero():
    trace = _full_trace()
    breakdown = build_breakdown(trace, telegram_message_ts=None)
    assert breakdown.telegram_to_receive_ms is None
    assert "UNKNOWN" in _render(breakdown.telegram_to_receive_ms)


def test_a_missing_stage_yields_unknown_not_zero():
    trace = StageTrace()
    trace.mark("receive")
    trace.mark("decision")
    breakdown = build_breakdown(trace, telegram_message_ts=None)
    assert breakdown.receive_to_mark_ms is None
    assert breakdown.parsing_ms is None
    assert breakdown.total_signal_to_decision_ms is not None


def test_transport_latency_is_labelled_approximate():
    """
    It spans two machines' clocks AND Telegram stamps messages to whole
    seconds. Presenting it as a precise millisecond figure would invent
    resolution the source does not have.
    """
    trace = _full_trace()
    breakdown = build_breakdown(trace, telegram_message_ts=trace.wall["receive"] - 0.42)
    payload = breakdown.as_dict()
    assert payload["telegram_to_receive_approximate"] is True
    assert TELEGRAM_TIMESTAMP_QUANTUM_S == 1.0


def test_a_backwards_transport_timestamp_is_clamped_not_negative():
    """Clock skew between hosts must not produce a negative latency."""
    trace = _full_trace()
    breakdown = build_breakdown(
        trace, telegram_message_ts=trace.wall["receive"] + 5.0)
    assert breakdown.telegram_to_receive_ms == 0.0


def test_the_breakdown_serialises_every_reported_figure():
    payload = build_breakdown(_full_trace(), telegram_message_ts=None).as_dict()
    for key in ("telegram_to_receive_ms", "receive_to_mark_ms",
                "mark_to_state_ms", "state_to_quote_ms",
                "quote_to_decision_ms", "total_signal_to_decision_ms",
                "parsing_ms", "dexscreener_ms", "rpc_ms", "pool_state_ms",
                "quote_ms", "jupiter_ms", "ordering_violations"):
        assert key in payload


def test_the_sub_stage_durations_are_individually_available():
    b = build_breakdown(_full_trace(), telegram_message_ts=None)
    assert b.parsing_ms == pytest.approx(100.0)
    assert b.dexscreener_ms == pytest.approx(100.0)
    assert b.rpc_ms == pytest.approx(100.0)
    assert b.quote_ms == pytest.approx(100.0)


def test_jupiter_latency_is_unknown_when_jupiter_was_not_used():
    assert build_breakdown(_full_trace(), telegram_message_ts=None).jupiter_ms is None


# ---------------------------------------------------------------------------
# 25. monotonic duration measurement
# ---------------------------------------------------------------------------

def test_durations_come_from_the_monotonic_clock_not_the_wall_clock():
    """
    Wall clock is deliberately wrong here: the two stages are 5 seconds apart
    on the wall clock but 0.25s apart monotonically. A harness that derived
    durations from wall time would report 5,000ms for a 250ms operation.
    """
    trace = StageTrace()
    trace.mark_at("receive", wall=1_000.0, mono=100.0)
    trace.mark_at("decision", wall=1_005.0, mono=100.25)
    assert trace.duration_ms("receive", "decision") == pytest.approx(250.0)


def test_a_negative_duration_is_clamped_rather_than_reported():
    trace = StageTrace()
    trace.mark_at("a", wall=1.0, mono=100.0)
    trace.mark_at("b", wall=2.0, mono=99.0)
    assert trace.duration_ms("a", "b") == 0.0


def test_out_of_order_stages_are_detected_rather_than_silently_accepted():
    """
    A trace where the pool state precedes the RPC request that supposedly
    produced it is not trustworthy, and the caller needs to be told rather
    than handed a confident negative duration.
    """
    trace = StageTrace()
    trace.mark_at("receive", wall=1.0, mono=100.0)
    trace.mark_at("pool_state", wall=2.0, mono=90.0)
    violations = trace.ordering_violations()
    assert violations
    assert "pool_state" in violations[0]


def test_a_well_formed_trace_has_no_ordering_violations():
    assert _full_trace().ordering_violations() == []


def test_re_marking_a_stage_records_the_later_reading():
    trace = StageTrace()
    trace.mark_at("receive", wall=1.0, mono=100.0)
    trace.mark_at("receive", wall=2.0, mono=150.0)
    assert trace.mono["receive"] == 150.0
    assert trace.order.count("receive") == 1, "a retry must not duplicate a stage"


def test_timed_brackets_a_call_even_when_it_raises():
    trace = StageTrace()

    def _boom():
        raise RuntimeError("provider down")

    with pytest.raises(RuntimeError):
        timed(trace, "quote_start", "quote_end", _boom)
    assert trace.duration_ms("quote_start", "quote_end") is not None, (
        "a failed call's latency should still be measured, not lost"
    )


def test_the_markers_are_timezone_aware_wall_clock():
    """Correlation uses wall clock, which the repo renders as UTC ISO-8601."""
    from autopaper.clock import iso

    trace = _full_trace()
    stamp = iso(trace.wall["receive"])
    assert stamp is not None
    assert stamp.endswith("Z")
    assert "T" in stamp


def test_a_real_monotonic_span_is_measurable():
    trace = StageTrace()
    trace.mark("receive")
    time.sleep(0.01)
    trace.mark("decision")
    elapsed = trace.duration_ms("receive", "decision")
    assert elapsed is not None and 8.0 <= elapsed <= 500.0


def _render(value: float | None) -> str:
    from autopaper.same_market.report import _ms

    return _ms(value)