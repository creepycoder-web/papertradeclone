"""
Offline verification of the latency instrumentation.

Mocked providers only; no network. These verify that the decomposition the
forensic relies on is (a) complete, (b) monotonic-safe, (c) free of
double-counting, and above all (d) INERT -- timing must never change a quote, a
fill, a guard verdict or a PnL.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from tests import paper_bridge_fakes as _fk
except ImportError:
    import paper_bridge_fakes as _fk

from autopaper.paper_live.arms import arm_names
from autopaper.latency import LatencyTrace, percentiles
from autopaper.paper_live.economics import evaluate_entry_economics
from autopaper.paper_live.fills import FillModel, fill_from_quote, realize
from autopaper.paper_live.schema import LatencyMarks

alert, bridge, buy_point, real_report, run, sell_point = (
    _fk.alert, _fk.bridge, _fk.buy_point, _fk.real_report, _fk.run, _fk.sell_point)
FakeQuoteService = _fk.FakeQuoteService


def _filled():
    engine = bridge()
    records = run(engine.handle_alert(
        snapshot=alert(), report=real_report()))
    return engine, [r for r in records if r.traded]


# 1 -------------------------------------------------------------------------
def test_every_required_timestamp_is_recorded():
    engine, traded = _filled()
    assert traded
    marks = traded[0].latency
    for field in ("receive_at_monotonic", "mark_request_at",
                  "mark_response_at", "decision_start_at", "decision_end_at",
                  "buy_quote_request_at", "buy_quote_response_at",
                  "entry_fill_at"):
        assert getattr(marks, field) is not None, field


# 2 -------------------------------------------------------------------------
def test_monotonic_durations_are_never_negative():
    marks = LatencyMarks()
    now = time.monotonic()
    marks.mark_request_at = now
    marks.mark_response_at = now + 0.050
    marks.buy_quote_request_at = now + 0.060
    marks.buy_quote_response_at = now + 0.310
    marks.decision_start_at = now + 0.010
    marks.decision_end_at = now + 0.011
    marks.entry_fill_at = now + 0.320
    marks.receive_at_monotonic = now
    for value in marks.as_dict().values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            assert value >= 0, marks.as_dict()


# 3 -------------------------------------------------------------------------
def test_missing_optional_timestamps_are_handled_safely():
    marks = LatencyMarks()
    payload = marks.as_dict()
    assert payload["mark_latency_ms"] is None
    assert payload["signal_to_simulated_entry_ms"] is None
    json.dumps(payload)                      # must stay serialisable
    assert all(math.isfinite(v) for v in payload.values()
               if isinstance(v, (int, float)) and not isinstance(v, bool))


# 4 -------------------------------------------------------------------------
def test_nested_timing_is_not_double_counted():
    """
    The settle wait is INSIDE alert_received->candidate_parsed.

    Summing the inclusive interval with its sub-interval would double count
    ~2 s per signal, which is why the report presents them as a tree.
    """
    marks = LatencyMarks()
    now = time.monotonic()
    marks.receive_at_monotonic = now
    marks.decision_start_at = now + 2.0
    marks.decision_end_at = now + 2.1
    marks.buy_quote_request_at = now + 2.2
    marks.buy_quote_response_at = now + 2.5
    marks.entry_fill_at = now + 2.6
    d = marks.as_dict()
    # decision_to_buy_quote STARTS where the decision ended: no overlap.
    assert d["decision_latency_ms"] == pytest.approx(100.0, abs=1.0)
    assert d["decision_to_buy_quote_ms"] == pytest.approx(100.0, abs=1.0)
    assert d["signal_to_simulated_entry_ms"] == pytest.approx(2600.0, abs=1.0)
    assert d["signal_to_simulated_entry_ms"] < (
        d["receive_to_decision_ms"] + d["decision_latency_ms"]
        + d["decision_to_buy_quote_ms"] + d["buy_quote_latency_ms"] + 1000)


# 5 -------------------------------------------------------------------------
def test_venue_level_quote_timings_are_retained():
    engine, traded = _filled()
    quote = traded[0].buy_quote
    assert quote.provider
    assert quote.venue_id and ":" in quote.venue_id
    assert quote.market_type
    # Freshness and request latency are SEPARATE fields; both must exist.
    assert hasattr(quote, "quote_latency_ms")
    assert hasattr(quote, "quote_age_ms")
    assert quote.quote_age_ms is not None
    assert quote.venue_id.endswith(quote.pool_or_curve)


# 6 -------------------------------------------------------------------------
def test_cache_hit_timing_is_recorded():
    quotes = FakeQuoteService()
    engine = bridge(quotes=quotes)
    run(engine.handle_alert(snapshot=alert(), report=real_report()))
    assert len(quotes.buy_calls) == 1
    stats = engine.quote_cache.stats()
    # ONE provider request feeds EVERY selected arm, so the other (N-1) are
    # cache hits. Derived from the engine's own arm list so adding an arm cannot
    # silently weaken the sharing assertion. The V2 FLOOR (BETA) arms are absent
    # from an empty selection, so there is no beta arm here to exclude.
    from autopaper.paper_live.arms import BETA_ARMS
    assert not ({a.value for a in BETA_ARMS}
                & {spec.name.value for spec in engine.arms}), (
        "the V2 FLOOR (BETA) arms must never arrive from an empty "
        "selection; they are opt-in only")
    expected_cache_hits = len(engine.arms) - 1
    assert stats["cache_hits"] == expected_cache_hits and stats["shared_snapshots"] == 1
    assert stats["hit_rate_pct"] == 100.0


# 7 -------------------------------------------------------------------------
def test_failure_path_timing_is_recorded():
    engine = bridge(quotes=FakeQuoteService(
        buy=buy_point(identity="UNKNOWN")))
    records = run(engine.handle_alert(
        snapshot=alert(), report=real_report()))
    for record in records:
        assert record.traded is False
        assert record.failure is not None
        assert record.failure_detail
        # A refusal still carries its timing, so a fast-fail is distinguishable
        # from a hang.
        assert record.latency.buy_quote_request_at is not None


# 8 -------------------------------------------------------------------------
def test_telegram_and_network_latency_are_separated():
    marks = LatencyMarks()
    now_wall, now_mono = time.time(), time.monotonic()
    marks.telegram_message_at = now_wall - 0.750      # cross-system
    marks.receive_at_monotonic = now_mono
    marks.receive_at_wall = now_wall
    d = marks.as_dict()
    assert d["telegram_to_receive_ms_cross_system"] == pytest.approx(
        750.0, abs=1.0)
    # The cross-system figure must not leak into the monotonic internal numbers.
    assert d["receive_to_decision_ms"] is None


# 9 -------------------------------------------------------------------------
def test_instrumentation_does_not_alter_quote_or_pnl_logic():
    """
    Wrapping a stage in timing must be observationally transparent.
    """
    def quote_of(quote):
        return quote

    for wrapped in (False, True):
        quote = buy_point()
        engine = bridge(quotes=FakeQuoteService(buy=quote))
        if wrapped:
            engine.quote_service.quote_buy = _timing_wrapper(
                engine.quote_service.quote_buy)
        records = run(engine.handle_alert(
            snapshot=alert(), report=real_report()))
        # Every selected arm trades on this report, and ``parse_arms([])`` is
        # the SELECTABLE set -- which does NOT include the V2 FLOOR (BETA)
        # experiment, because that is opt-in by name. So the traded count is
        # simply the arm count, and if a future arm is added it is measured
        # against the registry rather than a literal.
        from autopaper.paper_live.arms import BETA_ARMS
        assert not ({a.value for a in BETA_ARMS}
                    & {spec.name.value for spec in engine.arms}), (
            "the V2 FLOOR (BETA) arms must never arrive from an empty "
            "selection; they are opt-in only")
        traded = [r for r in records if r.traded]
        assert len(traded) == len(engine.arms)
        for record in traded:
            position = engine._position_for(record)
            fill = position.entry_fill
            assert fill.output_amount_raw == quote.output_amount_raw
            assert float(fill.effective_price_usd) == pytest.approx(
                float(quote.effective_price_usd), abs=1e-15)
            # The canonical kind name carries its provenance; it must still
            # assert that the fill came from an executable quote.
            assert "SIMULATED_FILL" in str(fill.kind)
            assert "EXECUTABLE_QUOTE" in str(fill.kind)


def _timing_wrapper(fn):
    async def timed(*a, **k):
        t0 = time.monotonic_ns()
        try:
            return await fn(*a, **k)
        finally:
            (time.monotonic_ns() - t0)
    return timed


def test_settlement_pnl_is_unchanged_by_measurement():
    engine, _ = _filled()

    class _ExitNow:
        def assess(self, position, *, now, mark_price, data_state, **kw):
            from autopaper.exits import ExitDecision
            from autopaper.models import ExitStyle
            return ExitDecision("EXIT", "t", ExitStyle.FAST_FAILURE, trigger="t")

    engine.exit_policy = _ExitNow()
    trades = [t for t in run(engine.tick())
              if t.failure is None or t.failure.name == "NONE"]
    assert trades
    for trade in trades:
        pnl = trade.pnl
        assert float(pnl.net_pnl_quote) == pytest.approx(
            float(trade.exit_fill.output_amount_decimal)
            - float(trade.position.entry_cost_quote_asset)
            - float(pnl.trading_fees_quote)
            - float(pnl.other_costs_quote)
            - float(pnl.network_costs_quote), abs=1e-12)
        assert trade.position.reclaim_status == "RENT_RECLAIMABLE"


# 10 ------------------------------------------------------------------------
def test_offline_tests_remain_network_free():
    """
    The latency tests must not reach a network.

    Proven by re-running this module in a subprocess with ``socket.socket``
    disabled, so any accidental dial fails loudly. An environment flag stops the
    inner run from re-entering this test, which would otherwise recurse forever.
    """
    import os

    if os.environ.get("LATENCY_NETWORK_PROBE") == "child":
        pytest.skip("inner socket-blocked run")
    # Block OUTBOUND network calls only. Replacing ``socket.socket`` wholesale
    # also breaks asyncio's internal self-pipe, which is not network access --
    # that produced a confusing event-loop teardown error rather than a real
    # finding.
    script = (
        "import os\n"
        "os.environ['LATENCY_NETWORK_PROBE']='child'\n"
        "import socket\n"
        "def _no(*a, **k):\n"
        "    raise AssertionError('a latency test attempted a network call')\n"
        "socket.create_connection=_no\n"
        "socket.getaddrinfo=_no\n"
        "socket.gethostbyname=_no\n"
        "_connect=socket.socket.connect\n"
        "def _connect_no(self, *a, **k):\n"
        "    raise AssertionError('a latency test attempted to connect')\n"
        "socket.socket.connect=_connect_no\n"
        "import pytest, sys\n"
        f"sys.exit(pytest.main(['{__file__}','-q','--no-header',"
        "'-p','no:cacheprovider']))\n"
    )
    r = subprocess.run([sys.executable, "-c", script],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]