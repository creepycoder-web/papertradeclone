"""
Entry-latency accounting, percentiles, and the fail-closed stale gate.

Two things are pinned:

* the distribution is reported with p95/p99/MAX and a sample count, because a
  median-only report hid a 16-minute outlier and a 34s p95;
* an entry whose signal is older than the project's EXISTING freshness budget is
  REFUSED rather than executed late, so a latency problem cannot become a
  strategy loss.
"""


from __future__ import annotations

import pytest

from autopaper.analysis import decision as dec
from autopaper.latency import (
    ENTRY_LATENCY_EXCEEDED,
    REJECT_REASON_TH,
    STALE_MARKET_DATA,
    STALE_QUOTE,
    STALE_SIGNAL,
    LatencyTrace,
    StaleEntryGate,
    percentiles,
    render_percentiles,
)

pytestmark = pytest.mark.strategy

BUDGET = 15_000.0
T0 = 1_000_000.0


def _ms(seconds: float) -> float:
    return seconds * 1000.0


# ---------------------------------------------------------------------------
# percentiles
# ---------------------------------------------------------------------------


def test_percentiles_of_a_known_sample():
    """
    Nearest-rank, which never UNDER-reports the tail.

    For n=10 that puts p50 at index 5 (value 60), not the interpolated median of
    55. That is deliberate: an optimistic percentile would understate exactly
    the tail this gate exists to catch.
    """
    s = percentiles([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    assert s["n"] == 10
    assert s["p50"] == 60
    assert s["p90"] == 100
    assert s["max"] == 100
    assert s["min"] == 10


def test_the_outlier_is_visible_in_max():
    """The whole point: a 16-minute outlier must not vanish into a median."""
    samples = [2100] * 99 + [958_498]
    s = percentiles(samples)
    assert s["p50"] == 2100
    assert s["max"] == 958_498
    assert s["p99"] >= 2100


def test_the_count_is_always_reported():
    assert percentiles([1, 2, 3])["n"] == 3
    assert percentiles([])["n"] == 0


def test_no_samples_reports_nulls_not_zeros():
    """A 0.0 p95 with no data would read as a measurement."""
    s = percentiles([])
    assert s["p50"] is None and s["max"] is None
    assert "no samples" in render_percentiles("x", s)


def test_junk_is_excluded_from_the_distribution():
    s = percentiles([10, None, "x", 20, float("nan")])
    assert s["n"] == 2
    assert s["max"] == 20


def test_the_render_line_carries_every_percentile():
    out = render_percentiles("analysis", percentiles([1, 2, 3, 4, 5]))
    for token in ("n=", "p50=", "p90=", "p95=", "p99=", "max="):
        assert token in out, token


# ---------------------------------------------------------------------------
# stage timestamps
# ---------------------------------------------------------------------------


def test_each_stage_is_measured():
    t = LatencyTrace(signal_at=T0, decision_at=T0 + 2.0, quote_at=T0 + 2.5,
                     submit_at=T0 + 3.0, fill_at=T0 + 3.4)
    assert t.analysis_ms == pytest.approx(2000)
    assert t.decision_to_quote_ms == pytest.approx(500)
    assert t.quote_age_at_submit_ms == pytest.approx(500)
    assert t.signal_to_fill_ms == pytest.approx(3400)
    assert t.ordering_violations() == []


def test_a_missing_stage_yields_none_not_zero():
    t = LatencyTrace(signal_at=T0)
    assert t.analysis_ms is None
    assert t.signal_to_fill_ms is None


def test_backwards_timestamps_are_reported():
    t = LatencyTrace(signal_at=T0 + 10, decision_at=T0)
    assert t.ordering_violations(), "a backwards clock must be visible"


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def test_a_fresh_candidate_is_allowed():
    g = StaleEntryGate(budget_ms=BUDGET)
    assert g.check(signal_at=T0, now=T0 + 2.0) is None


def test_a_2_second_old_signal_passes():
    g = StaleEntryGate(budget_ms=BUDGET)
    assert g.check(signal_at=T0, now=T0 + 2.0) is None


def test_a_10_second_old_signal_passes_if_otherwise_valid():
    """Inside the 15s budget: still tradeable."""
    g = StaleEntryGate(budget_ms=BUDGET)
    assert g.check(signal_at=T0, now=T0 + 10.0,
                   decision_at=T0 + 9.0) is None


def test_a_34_second_old_signal_is_refused():
    """
    The measured live case. 34s is beyond the 15s budget, so it must be
    refused rather than traded late.
    """
    g = StaleEntryGate(budget_ms=BUDGET)
    assert g.check(signal_at=T0, now=T0 + 34.0) == STALE_SIGNAL


def test_the_boundary_is_exclusive_at_the_budget():
    g = StaleEntryGate(budget_ms=BUDGET)
    assert g.check(signal_at=T0, now=T0 + BUDGET / 1000.0) is None
    assert g.check(signal_at=T0, now=T0 + BUDGET / 1000.0 + 0.01) == STALE_SIGNAL


def test_a_slow_analysis_alone_is_refused():
    g = StaleEntryGate(budget_ms=BUDGET)
    assert g.check(signal_at=T0, now=T0 + 1.0, decision_at=T0 + 34.0) \
        == STALE_SIGNAL


def test_a_stale_quote_is_named_as_such():
    g = StaleEntryGate(budget_ms=BUDGET)
    reason = g.check(signal_at=T0, now=T0 + 1.0, decision_at=T0 + 1.0,
                     quote_at=T0 + 2.0, submit_at=T0 + 40.0)
    assert reason == STALE_QUOTE


def test_a_stale_quote_cannot_become_a_fill():
    g = StaleEntryGate(budget_ms=BUDGET)
    reason = g.check(signal_at=T0, now=T0 + 1.0, decision_at=T0 + 1.0,
                     quote_at=T0 + 2.0, fill_at=T0 + 60.0)
    assert reason == STALE_QUOTE


def test_a_slow_pipeline_to_fill_is_named():
    """
    The quote stays FRESH but the whole pipeline is slow, so the reason is the
    pipeline rather than the quote.
    """
    g = StaleEntryGate(budget_ms=BUDGET)
    reason = g.check(signal_at=T0, now=T0 + 1.0, decision_at=T0 + 1.0,
                     quote_at=T0 + 9.0, submit_at=T0 + 9.2,
                     fill_at=T0 + 20.0)
    assert reason == ENTRY_LATENCY_EXCEEDED


def test_the_most_specific_reason_wins():
    """
    A stale quote is named in preference to a generic pipeline slowness, so the
    operator is told the thing that actually failed.
    """
    g = StaleEntryGate(budget_ms=BUDGET)
    reason = g.check(signal_at=T0, now=T0 + 1.0, decision_at=T0 + 1.0,
                     quote_at=T0 + 2.0, fill_at=T0 + 60.0)
    assert reason == STALE_QUOTE


def test_a_missing_signal_timestamp_fails_closed():
    """
    An unmeasurable age is not a fresh age.

    Without this, a candidate with no timestamp would pass every comparison by
    default and be traded on an unknown age.
    """
    g = StaleEntryGate(budget_ms=BUDGET)
    assert g.check(signal_at=None, now=T0) == STALE_SIGNAL


def test_the_gate_only_ever_refuses():
    """It cannot authorise an entry, whatever it is fed."""
    g = StaleEntryGate(budget_ms=BUDGET)
    for signal in (T0, T0 - 10_000, None):
        result = g.check(signal_at=signal, now=T0)
        assert result in (None, STALE_SIGNAL)
    assert g.stats()["refused_total"] >= 1


def test_refusals_are_counted_by_reason():
    g = StaleEntryGate(budget_ms=BUDGET)
    g.check(signal_at=T0, now=T0 + 34.0)
    g.check(signal_at=T0, now=T0 + 40.0)
    g.check(signal_at=None, now=T0)
    stats = g.stats()
    assert stats["refused_total"] == 3
    assert stats["refused_by_reason"][STALE_SIGNAL] == 3
    assert stats["budget_ms"] == BUDGET


def test_every_reason_has_thai_text():
    for reason in (STALE_SIGNAL, STALE_MARKET_DATA, STALE_QUOTE,
                   ENTRY_LATENCY_EXCEEDED):
        assert REJECT_REASON_TH.get(reason), reason


# ---------------------------------------------------------------------------
# the budget is the project's, not a new one
# ---------------------------------------------------------------------------


def test_the_budget_reuses_market_stale_after_ms():
    """
    No second, conflicting definition of "fresh".

    Asserted against the live config so a divergence is impossible.
    """
    import os
    import pathlib

    from autopaper.config import build_config
    from autopaper.engine import PaperTrader

    os.environ.pop("REAL_TRADING_ENABLED", None)
    cfg = build_config(pathlib.Path(__file__).resolve().parents[1])
    trader = PaperTrader(cfg)
    try:
        assert trader.stale_gate.budget_ms == cfg.market.stale_after_ms
        assert trader.stale_gate.budget_ms == 15_000.0
    finally:
        trader._release_ownership()


def test_the_stale_gate_runs_before_any_order_is_built():
    """
    Structural, on the real entry path: the gate is checked BEFORE the paper or
    real order branch, so nothing is prepared, sized, signed or sent for a
    refused candidate.
    """
    import inspect

    from autopaper.engine import PaperTrader

    src = inspect.getsource(PaperTrader._execute_buy)
    i_gate = src.index("self.stale_gate.check")
    assert i_gate < src.index("_execute_paper_buy")
    assert i_gate < src.index("_execute_real_buy")


def test_a_stale_quote_is_rechecked_at_fill_time():
    """
    The pre-order gate runs before the decision-latency sleep; the paper path
    re-runs it against the real fill time so a stale/unusable quote cannot
    become an order.
    """
    import inspect

    from autopaper.engine import PaperTrader

    src = inspect.getsource(PaperTrader._execute_paper_buy)
    assert "gate_reason = self.stale_gate.check(" in src
    assert "STALE_QUOTE" in src


def test_the_gate_is_not_an_l25_parameter():
    """It governs data quality, so it must not appear in the strategy."""
    from autopaper import l25

    src = open(l25.__file__, encoding="utf-8").read()
    assert "StaleEntryGate" not in src
    assert "stale_after" not in src
    assert l25.L25_MIN_LIQ_TO_MCAP == 0.9647, "L25 must be untouched"


# ---------------------------------------------------------------------------
# narrative budget
# ---------------------------------------------------------------------------


def test_the_narrative_budget_is_2_5_seconds():
    """
    The researched L25 ran on a ~2s median decision path, so a 12s narrative
    wait was the largest single contributor to the ~34s tail.
    """
    import os
    import pathlib

    from autopaper.config import build_config

    os.environ.pop("REAL_TRADING_ENABLED", None)
    cfg = build_config(pathlib.Path(__file__).resolve().parents[1])
    assert cfg.nar.timeout_s == 2.5


def test_the_narrative_worst_case_is_bounded():
    """
    The budget plus its grace, not the old hardcoded +2.0.

    The grace used to be a literal 2.0, so a 2.5s budget actually cost 4.5s.
    """
    import os
    import pathlib

    from autopaper.config import build_config

    os.environ.pop("REAL_TRADING_ENABLED", None)
    cfg = build_config(pathlib.Path(__file__).resolve().parents[1])
    total = cfg.nar.timeout_s + cfg.nar.timeout_grace_s
    assert total <= 3.0, f"narrative worst case is {total}s, expected <= 3.0s"


def test_the_grace_is_explicit_not_hardcoded():
    """
    The engine must add the CONFIGURED grace, not a literal.

    This is what stopped 4.5s being spent on a 2.5s budget.
    """
    import inspect

    from autopaper.engine import PaperTrader

    src = inspect.getsource(PaperTrader)
    # The budget is now computed as timeout_s + timeout_grace_s
    assert "self.config.nar.timeout_s + self.config.nar.timeout_grace_s" in src, \
        "the grace must be configured via timeout_grace_s"
    # No hardcoded + 2.0 literal
    assert "+ 2.0" not in src, "the hardcoded +2.0 is back"


def test_the_poll_sleep_cannot_overshoot_the_budget():
    """
    A 1.5s poll interval inside a 2.5s budget previously ran to ~3.0s.

    The sleep is capped to the time remaining. This is now checked BEHAVIOURALLY
    - by measuring - rather than by reading the source: the deadline moved from
    a local `time.monotonic()` inside `_query` into a single deadline shared by
    every wait in the call, so a source substring is no longer the thing that
    matters. What matters is that the call returns inside its budget.
    """
    import asyncio
    import time as _time

    from autopaper.nar.phanes import PhanesNarSource

    class _SilentClient:
        async def get_entity(self, _name):
            return object()

        async def get_messages(self, *_a, **_k):
            return []

        async def send_message(self, *_a, **_k):
            return type("M", (), {"id": 1})()

    src = PhanesNarSource(clock=None, client=_SilentClient(),
                          collect_extra_s=1.5, min_gap_s=0.0, timeout_s=0.30)
    t0 = _time.monotonic()
    snap = asyncio.run(src.fetch_async("M"))
    elapsed = _time.monotonic() - t0
    assert snap.available is False
    # 0.30s budget; the uncapped 1.5s poll sleep would have run to ~1.5s.
    assert elapsed < 0.30 + 0.25, (
        f"a silent narrative overran its budget: {elapsed:.3f}s for a 0.30s budget"
    )
    assert src.stats.timeouts == 1


def test_the_whole_narrative_query_is_inside_the_budget():
    """
    The budget must cover the WHOLE call, not only the polling loop.

    This is the fault found on 2026-09-30: `get_entity`, the anchor read, the
    polite `min_gap_s` sleep and `send_message` all sat outside the deadline, so
    the "2.5s budget" was a floor rather than a bound. Each of those steps is
    made slow here and the call must still return inside its budget.
    """
    import asyncio
    import time as _time

    from autopaper.nar.phanes import PhanesNarSource

    class _SlowStep:
        """One slow RPC, then silence - the shape that used to overrun."""

        def __init__(self):
            self.n = 0

        async def get_entity(self, _name):
            await asyncio.sleep(0.25)
            return object()

        async def get_messages(self, *_a, **_k):
            self.n += 1
            if self.n == 1:
                # the ANCHOR read: this is what used to sit outside the deadline
                await asyncio.sleep(0.25)
            return []

        async def send_message(self, *_a, **_k):
            await asyncio.sleep(0.25)
            return type("M", (), {"id": 1})()

    src = PhanesNarSource(clock=None, client=_SlowStep(),
                          collect_extra_s=0.05, min_gap_s=4.0, timeout_s=0.40)
    t0 = _time.monotonic()
    snap = asyncio.run(src.fetch_async("M"))
    elapsed = _time.monotonic() - t0
    assert snap.available is False
    assert elapsed < 0.40 + 0.25, (
        "entity/anchor/send/gap must all be inside the narrative budget; "
        f"took {elapsed:.3f}s for a 0.40s budget"
    )


def test_a_narrative_timeout_is_observable():
    """
    A timeout must be a metric, not a silent quality reduction.

    The failure this exists for: a narrative outage quietly cost 0.05 of
    signal_quality and so silently changed the candidate POPULATION.
    """
    from autopaper.nar.base import NarMetrics

    m = NarMetrics()
    m.record(available=True, latency_ms=1700.0)
    m.record(available=False, latency_ms=2500.0)
    d = m.as_dict()
    assert d["requested"] == 2
    assert d["available"] == 1
    assert d["unavailable"] == 1
    assert d["unavailable_rate"] == pytest.approx(0.5)
    for key in ("latency_p50_ms", "latency_p90_ms", "latency_p95_ms",
                "latency_p99_ms", "latency_max_ms", "samples"):
        assert key in d, key
    assert d["samples"] == 2


def test_an_unavailable_narrative_is_not_converted_into_a_score():
    """
    Fail-safe, not fail-open.

    No invented narrative score; the existing reduced-quality behaviour stands
    and is now measurable.
    """
    from autopaper.analysis.features import _signal_quality

    base = dict(
        data_state=dec.DataState.LIVE, freshness=1.0, liq_score=0.8,
        act_score=0.8, mom_score=0.7, struct_score=0.7, ext_score=0.6,
        alert_fields=17, span_s=600.0, narrative_placeholder=False,
        nar_has_flow=False,
    )
    with_nar = _signal_quality(nar_available=True, **base)
    without = _signal_quality(nar_available=False, **base)
    assert without < with_nar
    assert without >= 0.0, "quality must never be negative"


def test_narrative_metrics_are_exposed_on_the_engine():
    import inspect

    from autopaper.engine import PaperTrader

    src = inspect.getsource(PaperTrader.bot_status)
    assert '"narrative"' in src
    assert "self.nar.metrics.as_dict()" in src
    assert "narrative_timeout_s" in src
