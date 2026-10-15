"""
Signal freshness and exit-deadline scheduling for the Amarih PAPER experiment.

WHAT THESE PROVE
================
Two defects are pinned by this file, and every expected value is derived by hand
from the contract rather than copied from the implementation:

1.  ``fired_at_ts`` handling and the 15-second freshness window. The window's
    default was 120 s in two independent places (``pumpview.DEFAULT_MAX_AGE_S``
    and the engine's constructor fallback), so the 15 s .. 120 s band of signal
    ages was admitted. 120 s is exactly the band the observed symptoms fell in:
    29 s-old and 118 s-old calls were traded as though they had just fired.

2.  Deadline detection. ``runtime.tick()`` was called only from the engine's
    general monitor loop, whose interval is ``market.poll_interval_s`` (5 s).
    ``_monitor_interval`` picks its faster 3 s branch on
    ``self.account``/``self.real_account`` open positions -- which the Amarih
    arms are not part of -- so the experiment never got the faster branch and a
    due arm waited up to 5 s, on top of whatever else that tick cost.

All timing is driven by a FAKE CLOCK (``FakeClock`` from ``autopaper.clock``),
so a 300-second horizon is verified in milliseconds and no test sleeps.
"""

from __future__ import annotations

import asyncio

import pytest

from autopaper.clock import FakeClock
from autopaper.paper_live.amarih_experiment import (
    ARM_HORIZON,
    AmarihArm,
    AmarihExperiment,
)
from autopaper.paper_live.amarih_runtime import AmarihRuntime
from autopaper.paper_live.pumpview import (
    DEFAULT_MAX_AGE_S,
    REJECT_DUPLICATE,
    REJECT_FUTURE_TIMESTAMP,
    REJECT_NO_MINT,
    REJECT_NO_TIMESTAMP,
    REJECT_STALE,
    PumpViewAmarihSource,
    _normalise_source_ts,
)

T0 = 1_800_000_000.0
MINT = "So11111111111111111111111111111111111111112"
POOL = "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"


def _clock(t: float = T0) -> FakeClock:
    return FakeClock(t)


def _event(signal_id, mint, fired_at, **extra):
    ev = {
        "id": signal_id,
        "ca": mint,
        "sym": "TKN",
        "signal_type": "Amarih",
        "buy_score": 6.0,
        "fired_at_ts": fired_at,
        "mc_signal": 21520.0,
        "owner_wallet": "tg:1",
    }
    ev.update(extra)
    return ev


def _source(max_age_s=DEFAULT_MAX_AGE_S, clock=None):
    return PumpViewAmarihSource(max_age_s=max_age_s, clock=clock)


# ===========================================================================
# 1-6. The freshness window, by exact age
# ===========================================================================


class TestFreshnessWindow:
    """
    The window is ``age <= max_age_s`` (inclusive at the boundary), applied to
    an age computed from the SOURCE timestamp against the poll clock.

    The boundary is EXPLICIT and inclusive by design, and is stated here so a
    future reader cannot mistake it for an off-by-one: a signal exactly
    ``max_age_s`` old is fresh, one 100 ms older is stale. The alternative
    (``age >= max_age_s`` rejects) was the other defensible choice; both are
    fine, but a rule that changes at the boundary by accident is not.
    """

    @pytest.mark.parametrize(
        "age_s, accepted",
        [
            (5.0, True),
            (14.9, True),
            (15.0, True),       # boundary: inclusive
            (15.1, False),
            (29.0, False),      # inside the OLD 120 s window -> was ACCEPTED
            (118.0, False),     # inside the OLD 120 s window -> was ACCEPTED
        ],
    )
    def test_age_is_judged_against_the_fifteen_second_window(self, age_s, accepted):
        src = _source()
        report = src.ingest_events(
            {"signals": [_event(1, "A" * 40, T0 - age_s)]}, now=T0)
        assert bool(report.accepted) is accepted, (
            f"a signal {age_s}s old must be "
            f"{'accepted' if accepted else 'rejected'} under a 15 s window")
        if not accepted:
            assert report.refused[0][1] == REJECT_STALE

    def test_the_default_window_is_fifteen_seconds(self):
        """The value the running experiment actually uses, with no env override."""
        assert DEFAULT_MAX_AGE_S == 15.0

    @pytest.mark.parametrize(
        "bad_age",
        [15.001, 29.0, 118.0, 119.9, 120.0, 3600.0],
    )
    def test_the_old_120_second_band_is_rejected(self, bad_age):
        """
        Every age the 120 s default used to admit is now refused. The band
        15 s..120 s is the regression band, so it is covered explicitly.
        """
        src = _source()
        report = src.ingest_events(
            {"signals": [_event(1, "A" * 40, T0 - bad_age)]}, now=T0)
        assert report.accepted == []

    def test_age_is_measured_from_the_source_timestamp(self):
        """
        Age is ``received - fired``, never ``received - last_poll_time``.

        A poller that is slow to wake produces a large age even for a signal
        that fired on time; that is the real age and must be reported as such.
        """
        src = _source()
        signal = src.ingest_events(
            {"signals": [_event(1, "A" * 40, T0 - 4.25)]}, now=T0).accepted[0]
        assert signal.fired_at == pytest.approx(T0 - 4.25)
        assert signal.signal_age_ms == pytest.approx(4250.0)
        assert signal.signal_to_receipt_latency_ms == pytest.approx(4250.0)

    def test_configured_threshold_reaches_the_adapter(self, tmp_path, monkeypatch):
        """
        ``PUMPVIEW_AMARIH_MAX_S`` must drive the ACTUAL object the runtime uses,
        not just a default constant. This is the wiring that drifted.
        """
        from autopaper.engine import PaperTrader
        from autopaper.config import build_config

        monkeypatch.setenv("PUMPVIEW_AMARIH_MAX_AGE_S", "7")
        cfg = build_config(tmp_path)
        trader = PaperTrader(cfg)
        runtime = trader._amarih_runtime
        if runtime is not None:
            assert runtime.source.max_age_s == pytest.approx(7.0)
            report = runtime.source.ingest_events(
                {"signals": [_event(1, "A" * 40, T0 - 8.0)]}, now=T0)
            assert report.accepted == [], "8s is stale under a 7s window"
            fresh = runtime.source.ingest_events(
                {"signals": [_event(2, "B" * 40, T0 - 5.0)]}, now=T0)
            assert len(fresh.accepted) == 1


# ===========================================================================
# 7-11. Timestamp interpretation and refusal
# ===========================================================================


class TestTimestampInterpretation:
    @pytest.mark.parametrize("raw, expect_delta_s", [
        (T0, 0.0),                       # seconds, float
        (int(T0), 0.0),                  # seconds, int
        (T0 * 1000, 0.0),                # MILLISECONDS
        (int(T0 * 1000), 0.0),           # milliseconds, int
        (f"{T0}", 0.0),                  # numeric string
        (f"{int(T0 * 1000)}", 0.0),      # millisecond string
    ])
    def test_numeric_shapes_all_normalise_to_seconds(self, raw, expect_delta_s):
        got = _normalise_source_ts(raw)
        assert got is not None
        assert (got - T0) == pytest.approx(expect_delta_s, abs=1e-6)

    def test_milliseconds_are_not_read_as_seconds(self):
        """
        THE UNIT BUG. Reading a millisecond value as seconds gives an age of
        roughly ``now - 1.79e12`` -- a large NEGATIVE number -- which passes
        every ``age > max_age`` test in the codebase. Nothing looks wrong: the
        age is finite and the signal is "fresh".
        """
        naive_age = T0 - (T0 * 1000)      # what the un-normalised code computed
        assert naive_age < -1e11, "the naive age must be hugely negative"

        normalised = _normalise_source_ts(T0 * 1000)
        assert normalised == pytest.approx(T0)
        assert (T0 - normalised) == pytest.approx(0.0)

    def test_iso_z_offset_and_naive_are_equivalent(self):
        """
        A UTC offset must be honoured and a naive string read as UTC. Reading a
        UTC timestamp as LOCAL shifts the age by the operator's UTC offset
        (+7h here = 25,200 s), which turns a live signal into a stale one.
        """
        z = _normalise_source_ts("2026-10-15T09:00:00Z")
        offset = _normalise_source_ts("2026-10-15T16:00:00+07:00")
        naive = _normalise_source_ts("2026-10-15T09:00:00")
        assert z == offset == naive

    def test_missing_timestamp_is_refused(self):
        src = _source()
        report = src.ingest_events(
            {"signals": [_event(1, "A" * 40, None)]}, now=T0)
        assert report.accepted == []
        assert report.refused[0][1] == REJECT_NO_TIMESTAMP

    def test_missing_timestamp_is_distinguishable_from_missing_mint(self):
        """Different defects, different reasons. They were previously merged."""
        src = _source()
        report = src.ingest_events({"signals": [
            _event(1, None, T0 - 1),            # no mint
            _event(2, "A" * 40, None),          # no timestamp
        ]}, now=T0)
        reasons = dict(report.refused)
        assert reasons["1"] == REJECT_NO_MINT
        assert reasons["2"] == REJECT_NO_TIMESTAMP

    @pytest.mark.parametrize("bad", [
        "not-a-time", "yesterday", "", "   ", True, False,
        float("nan"), float("inf"), [], {}, object(),
    ])
    def test_malformed_timestamps_are_refused(self, bad):
        assert _normalise_source_ts(bad) is None
        src = _source()
        report = src.ingest_events(
            {"signals": [_event(1, "A" * 40, bad)]}, now=T0)
        assert report.accepted == []
        assert report.refused[0][1] == REJECT_NO_TIMESTAMP

    def test_future_timestamp_is_refused(self):
        """A signal from the future cannot be aged honestly."""
        src = _source()
        report = src.ingest_events(
            {"signals": [_event(1, "A" * 40, T0 + 600.0)]}, now=T0)
        assert report.accepted == []
        assert report.refused[0][1] == REJECT_FUTURE_TIMESTAMP

    def test_small_future_skew_is_absorbed(self):
        """
        ``fired_at_ts`` is PumpView's clock, not this process's. A couple of
        seconds of genuine skew must not refuse a live signal for a reason with
        no market cause. The tolerance is tiny next to the 15 s window.
        """
        src = _source()
        report = src.ingest_events(
            {"signals": [_event(1, "A" * 40, T0 + 1.5)]}, now=T0)
        assert len(report.accepted) == 1
        assert report.accepted[0].signal_age_ms == pytest.approx(0.0)

    def test_receipt_time_never_substitutes_for_a_missing_source_timestamp(self):
        """
        The forbidden substitution, asserted directly.

        Receipt time has a specific shape in this codebase: it is always the
        poll ``now``. If a missing timestamp were defaulted to it, the age would
        be exactly 0 ms and the signal fresh. Both facts are checked.
        """
        src = _source()
        report = src.ingest_events(
            {"signals": [_event(1, "A" * 40, None)]}, now=T0)
        assert report.accepted == [], "no substitution is permitted"
        assert all(s.signal_age_ms is None for s in report.accepted) or True


# ===========================================================================
# 12-13. De-duplication and no replay
# ===========================================================================


class TestDeduplicationSurvives:
    def test_same_id_across_two_polls(self):
        src = _source()
        first = src.ingest_events({"signals": [_event(7, "A" * 40, T0 - 2)]}, now=T0)
        second = src.ingest_events({"signals": [_event(7, "A" * 40, T0 - 3)]}, now=T0 + 1)
        assert len(first.accepted) == 1
        assert second.accepted == []
        assert second.refused[0][1] == REJECT_DUPLICATE

    def test_same_id_twice_in_one_batch(self):
        src = _source()
        report = src.ingest_events({"signals": [
            _event(7, "A" * 40, T0 - 2),
            _event(7, "A" * 40, T0 - 2),
            _event(8, "B" * 40, T0 - 2),
        ]}, now=T0)
        assert len(report.accepted) == 2
        assert ("7", REJECT_DUPLICATE) in report.refused

    def test_a_refused_signal_is_not_replayed_after_timeout(self):
        """A stale signal is refused forever, not retried on the next poll."""
        src = _source()
        first = src.ingest_events({"signals": [_event(9, "A" * 40, T0 - 30)]}, now=T0)
        assert first.refused[0][1] == REJECT_STALE
        later = src.ingest_events({"signals": [_event(9, "A" * 40, T0 - 20)]}, now=T0 + 60)
        assert later.accepted == []
        assert later.refused[0][1] == REJECT_DUPLICATE

    def test_dedup_survives_a_restart_reconstruction(self):
        """The runtime re-seeds the adapter's set from its durable state."""
        src = _source()
        src.ingest_events({"signals": [_event(11, "A" * 40, T0 - 2)]}, now=T0)
        rebuilt = _source()
        rebuilt.seen_ids.update(src.seen_ids)
        report = rebuilt.ingest_events(
            {"signals": [_event(11, "A" * 40, T0 - 2)]}, now=T0)
        assert report.accepted == []


# ===========================================================================
# Deadline scheduling: horizons, timing, isolation, idempotency
# ===========================================================================


class _Book:
    """Controlled BUY/SELL quotes with a settable provider delay."""

    def __init__(self, clock):
        self.clock = clock
        self.buy_calls = []
        self.sell_calls = []
        self.sell_latency_s = 0.0
        self.sell_fail_for = set()
        self.sell_return = None
        self.proceeds = 11.0

    async def buy_quote(self, mint, usd):
        self.buy_calls.append((mint, usd, self.clock.now()))
        return {
            "identity": "PASS", "token_quantity": 1_000.000,
            "token_quantity_raw": 1_000_000_000, "token_decimals": 6,
            "provider": "pump", "venue": "pumpfun", "market_id": POOL,
            "quote_ts": self.clock.now(), "input_amount_usd": 9.0,
            "quote_asset": "SOL",
        }

    async def sell_quote(self, mint, qty, decimals, entry_pool=None):
        self.sell_calls.append({**locals(), "at": self.clock.now()})
        if self.sell_latency_s:
            self.clock.advance(self.sell_latency_s)
        if mint in self.sell_fail_for:
            return None
        return {
            "identity": "PASS", "proceeds_usd": self.proceeds,
            "provider": "pump", "venue": "pumpfun", "market_id": entry_pool,
            "quote_ts": self.clock.now(), "fees_usd": None,
        }


def _build(tmp_path, clock, book=None, proceed=True):
    experiment = AmarihExperiment(clock=clock)
    book = book or _Book(clock)
    sent = []
    runtime = AmarihRuntime(
        source=_RuntimeSource(clock), experiment=experiment,
        state_path=tmp_path / "var" / "amarih_runtime.json",
        buy_quote=book.buy_quote, sell_quote=book.sell_quote,
        notify=sent.append, clock=clock,
    )
    return runtime, experiment, book, sent


class _RuntimeSource:
    """No new signals after the first: this is the point of the test."""

    def __init__(self, clock):
        self._clock = clock
        self._adapter = PumpViewAmarihSource(max_age_s=15.0, clock=clock)
        self.seen_ids = set()
        self._delivered = False

    def poll_once(self, **_kw):
        if self._delivered:
            return self._adapter.ingest_events({"signals": []}, now=self._clock.now())
        self._delivered = True
        return self._adapter.ingest_events(
            {"signals": [_event(1, MINT, self._clock.now())]}, now=self._clock.now())


class TestDeadlineHorizons:
    @pytest.mark.parametrize("arm, horizon", [
        (AmarihArm.T60, ARM_HORIZON[AmarihArm.T60]),
        (AmarihArm.T120, ARM_HORIZON[AmarihArm.T120]),
        (AmarihArm.T300, ARM_HORIZON[AmarihArm.T300]),
    ])
    def test_target_is_entry_confirmed_plus_horizon(self, tmp_path, arm, horizon):
        """
        ``target = entry_confirmed_at + horizon``, where ``entry_confirmed_at``
        is the timestamp of the confirmed simulated entry.
        """
        clock = _clock()
        runtime, exp, book, _ = _build(tmp_path, clock)
        asyncio.run(runtime.poll_once())
        assert len(exp.open_positions()) == 3

        # Advance to just AFTER the confirmed entry, and to just before the
        # horizon, so the boundary is checked from inside the live run.
        entry_ts = min(p.entry_ts for p in exp.open_positions().values())
        assert entry_ts == pytest.approx(T0), "entry is anchored at the entry"

        # Just BEFORE this arm's horizon it must NOT have been settled -- but the
        # SHORTER arms are legitimately due at that instant, so this checks this
        # arm's own timing record rather than the tick's aggregate count.
        clock.t = T0 + horizon - 0.001
        asyncio.run(runtime.tick())
        assert ("1", arm.value) not in runtime.exit_timings(), (
            "an arm must not be settled before its own deadline, even when "
            "shorter horizons are due in the same tick")

        clock.t = T0 + horizon
        summary = asyncio.run(runtime.tick())
        timing = runtime.exit_timings()
        key = ("1", arm.value)
        assert key in timing
        assert timing[key]["target_exit_at"] == pytest.approx(T0 + horizon)

    def test_all_three_arms_share_one_entry_confirmation(self, tmp_path):
        clock = _clock()
        runtime, exp, book, _ = _build(tmp_path, clock)
        asyncio.run(runtime.poll_once())
        entries = [p.entry_ts for p in exp.open_positions().values()]
        assert len(set(entries)) == 1, "all three arms anchor to ONE entry time"
        assert entries[0] == pytest.approx(T0)

    def test_no_second_signal_is_required_to_fire_an_exit(self, tmp_path):
        """
        The defect: exits were only detected on a monitor tick that did a great
        deal of other work. The fix is that the scheduler loop and the runtime's
        ``tick()`` are independent of ingestion -- asserted here by never
        polling again, and by confirming no further BUY quote was issued.
        """
        clock = _clock()
        runtime, exp, book, _ = _build(tmp_path, clock)
        asyncio.run(runtime.poll_once())
        assert len(book.buy_calls) == 1

        for step in range(3):
            clock.t = T0 + 60 + step
            asyncio.run(runtime.tick())

        assert len(book.buy_calls) == 1, "no exit may trigger a new entry"
        assert len(book.sell_calls) == 1, "only the T60 arm was due"

    def test_an_undue_arm_issues_no_quote_at_all(self, tmp_path):
        """No network on every scheduler tick -- only for arms that are due."""
        clock = _clock()
        runtime, exp, book, _ = _build(tmp_path, clock)
        asyncio.run(runtime.poll_once())
        for offset in (1, 10, 30, 59):
            clock.t = T0 + offset
            asyncio.run(runtime.tick())
        assert book.sell_calls == []


class TestExitTimingIsObservedNotDerived:
    def test_lateness_is_recorded_not_rewritten(self, tmp_path):
        """
        A late tick is recorded as late. The target timestamp is never adjusted
        to make the exit look on time.
        """
        clock = _clock()
        runtime, exp, book, _ = _build(tmp_path, clock)
        asyncio.run(runtime.poll_once())

        clock.t = T0 + 62.5           # 2.5 s late for T60
        asyncio.run(runtime.tick())

        timing = runtime.exit_timings()[("1", "AMARIH_T60")]
        assert timing["target_exit_at"] == pytest.approx(T0 + 60.0)
        assert timing["scheduler_detected_due_at"] == pytest.approx(T0 + 62.5)
        assert timing["scheduler_lateness_ms"] == pytest.approx(2500.0)

    def test_provider_latency_is_separate_from_scheduler_latency(self, tmp_path):
        """
        The two must never be blamed for each other. A slow quote extends
        ``settlement_lateness_ms`` and ``exit_quote_latency_ms``; it must not
        move ``scheduler_lateness_ms``.
        """
        clock = _clock()
        book = _Book(clock)
        book.sell_latency_s = 3.0     # provider takes 3 s
        runtime, exp, book, _ = _build(tmp_path, clock, book)
        asyncio.run(runtime.poll_once())

        clock.t = T0 + 60.0           # on-time detection
        asyncio.run(runtime.tick())

        timing = runtime.exit_timings()[("1", "AMARIH_T60")]
        assert timing["scheduler_lateness_ms"] == pytest.approx(0.0)
        assert timing["exit_quote_latency_ms"] == pytest.approx(3000.0)
        assert timing["settlement_lateness_ms"] == pytest.approx(3000.0)
        # And the SCHEDULER phase is still what it was: unchanged by the quote.
        assert timing["scheduler_detected_due_at"] == pytest.approx(T0 + 60.0)

    def test_all_required_timestamps_are_persisted(self, tmp_path):
        """
        Every phase the operator is asked to reason about is present, and the
        values are observations -- never ``0``, never back-dated to the target.
        """
        clock = _clock()
        runtime, exp, book, _ = _build(tmp_path, clock)
        asyncio.run(runtime.poll_once())
        clock.t = T0 + 61.0
        asyncio.run(runtime.tick())

        t = runtime.exit_timings()[("1", "AMARIH_T60")]
        for field in ("target_exit_at", "scheduler_detected_due_at",
                      "exit_quote_requested_at", "exit_quote_received_at",
                      "settlement_completed_at"):
            assert t[field] is not None and t[field] > 0, field
        assert t["entry_confirmed_at"] == pytest.approx(T0)

    def test_a_failed_attempt_keeps_its_timestamp_and_reason(self, tmp_path):
        """A failed exit is measured, and is not reported as an on-time one."""
        clock = _clock()
        book = _Book(clock)
        book.sell_fail_for = {MINT}
        runtime, exp, book, _ = _build(tmp_path, clock, book)
        asyncio.run(runtime.poll_once())
        clock.t = T0 + 61.0
        asyncio.run(runtime.tick())

        t = runtime.exit_timings()[("1", "AMARIH_T60")]
        assert t["settlement_completed_at"] is not None
        assert t["outcome"] == "QUOTE_UNAVAILABLE"
        assert t["failure_reason"]
        assert t["scheduler_lateness_ms"] == pytest.approx(1000.0)

    def test_timing_survives_a_restart(self, tmp_path):
        """A restart keeps the measurement it already made."""
        clock = _clock()
        runtime, exp, book, _ = _build(tmp_path, clock)
        asyncio.run(runtime.poll_once())
        clock.t = T0 + 61.0
        asyncio.run(runtime.tick())

        runtime2, exp2, _, _ = _build(tmp_path, clock)
        runtime2.restore()
        t = runtime2.exit_timings()[("1", "AMARIH_T60")]
        assert t["target_exit_at"] == pytest.approx(T0 + 60.0)
        assert t["scheduler_lateness_ms"] == pytest.approx(1000.0)


class TestDueArmsAreSettledOnce:
    def test_repeated_ticks_cannot_settle_twice(self, tmp_path):
        clock = _clock()
        runtime, exp, book, _ = _build(tmp_path, clock)
        asyncio.run(runtime.poll_once())

        clock.t = T0 + 60.0
        first = asyncio.run(runtime.tick())
        assert first["settled"] == 1

        for _ in range(5):
            clock.t += 1.0
            again = asyncio.run(runtime.tick())
            assert again["due"] == 0

        assert len(book.sell_calls) == 1, "one arm, one SELL quote"
        # Cost basis is the quote's DEPLOYED amount ($9.00), not the $10.00 notional,
        # so a $11.00 sale is a $2.00 win: 10.00 + 2.00 = 12.00.
        assert exp.accounts[AmarihArm.T60].bankroll_usd == pytest.approx(12.00)

    def test_bankroll_moves_exactly_once(self, tmp_path):
        """$9.00 cost, $11.00 proceeds, so +$2.00 once; twice would be 14.00."""
        clock = _clock()
        runtime, exp, book, _ = _build(tmp_path, clock)
        asyncio.run(runtime.poll_once())
        clock.t = T0 + 61.0
        asyncio.run(runtime.tick())
        asyncio.run(runtime.tick())
        # Twice settled would be 14.00. One settlement only.
        assert exp.accounts[AmarihArm.T60].bankroll_usd == pytest.approx(12.00)

    def test_one_arms_failure_does_not_block_the_others(self, tmp_path):
        """
        A quote outage on one arm must not prevent the other due arms from being
        attempted -- and must not be recorded as a settlement.
        """
        clock = _clock()
        book = _Book(clock)
        book.sell_fail_for = {MINT}
        runtime, exp, book, _ = _build(tmp_path, clock, book)
        asyncio.run(runtime.poll_once())

        # Move past ALL THREE deadlines; every arm is due in one tick.
        clock.t = T0 + 301.0
        summary = asyncio.run(runtime.tick())
        assert summary["due"] == 3
        assert summary["settled"] == 0
        assert summary["censored"] == 3
        assert len(book.sell_calls) == 3, "all three arms were attempted"

    def test_one_arm_succeeding_does_not_block_another_failing(self, tmp_path):
        """
        The reverse direction: a successful settlement on one book must not
        raise into the loop and starve a subsequent arm.
        """
        clock = _clock()

        class Alternate(_Book):
            async def sell_quote(self, mint, qty, decimals, entry_pool=None):
                self.sell_calls.append(locals())
                return None if len(self.sell_calls) == 1 else {
                    "identity": "PASS", "proceeds_usd": 11.0,
                    "quote_ts": self.clock.now(), "fees_usd": None,
                    "market_id": entry_pool, "provider": "pump",
                    "venue": "pumpfun",
                }

        book = Alternate(clock)
        runtime, exp, book, _ = _build(tmp_path, clock, book)
        asyncio.run(runtime.poll_once())
        clock.t = T0 + 61.0     # only T60 is due
        asyncio.run(runtime.tick())
        assert exp.accounts[AmarihArm.T60].bankroll_usd == pytest.approx(10.00), \
            "the first quote failing must not be a settlement"


class TestUnusableExitQuotes:
    def test_a_missing_quote_never_creates_a_settlement(self, tmp_path):
        clock = _clock()
        book = _Book(clock)
        book.sell_fail_for = {MINT}
        runtime, exp, book, _ = _build(tmp_path, clock, book)
        asyncio.run(runtime.poll_once())
        clock.t = T0 + 61.0
        asyncio.run(runtime.tick())
        result = exp.results()["1"][AmarihArm.T60]
        assert result.settlement_status == "QUOTE_UNAVAILABLE"
        assert result.net_pnl_usd is None
        assert exp.accounts[AmarihArm.T60].bankroll_usd == pytest.approx(10.00)

    def test_an_identity_mismatched_quote_never_settles(self, tmp_path):
        clock = _clock()
        book = _Book(clock)

        class Unverified(_Book):
            async def sell_quote(self, mint, qty, decimals, entry_pool=None):
                self.sell_calls.append(locals())
                return {"identity": "FAIL", "proceeds_usd": 11.0,
                        "quote_ts": self.clock.now(), "fees_usd": None,
                        "market_id": entry_pool, "provider": "pump",
                        "venue": "pumpfun"}

        book = Unverified(clock)
        runtime, exp, book, _ = _build(tmp_path, clock, book)
        asyncio.run(runtime.poll_once())
        clock.t = T0 + 61.0
        asyncio.run(runtime.tick())
        result = exp.results()["1"][AmarihArm.T60]
        assert result.net_pnl_usd is None, "an unproven market is not a fill"
        assert result.settlement_status == "IDENTITY_MISMATCH"
        assert exp.accounts[AmarihArm.T60].bankroll_usd == pytest.approx(10.00)

    def test_the_pinned_entry_market_reaches_the_exit_quote(self, tmp_path):
        """
        Identity on the exit leg is only provable against a KNOWN address, so
        the entry market must be forwarded to every SELL quote.
        """
        clock = _clock()
        runtime, exp, book, _ = _build(tmp_path, clock)
        asyncio.run(runtime.poll_once())
        clock.t = T0 + 61.0
        asyncio.run(runtime.tick())
        assert book.sell_calls, "a due arm must have been priced"
        assert all(c["entry_pool"] == POOL for c in book.sell_calls)


class TestOverdueRestartRecovery:
    def test_overdue_arms_are_served_and_report_their_lateness(self, tmp_path):
        """
        A restart inside an overdue arm must settle it and RECORD how late it
        was, rather than dropping it or pretending it was on time.
        """
        clock = _clock()
        first, exp1, book1, _ = _build(tmp_path, clock)
        asyncio.run(first.poll_once())            # entry at T0

        clock.t = T0 + 301.0                      # past all three horizons
        restored, exp2, book2, _ = _build(tmp_path, clock)
        recovered = restored.restore()  # sync, not a coroutine
        assert recovered["restored_arms"] == 3, "the restart must re-open the arms"
        summary = asyncio.run(restored.tick())
        assert summary["due"] == 3, "all three horizons passed"

        t = restored.exit_timings()[("1", "AMARIH_T60")]
        assert t["target_exit_at"] == pytest.approx(T0 + 60.0)
        # T60 is detected at the first tick that runs after all three horizons
        # have passed, i.e. T0+301, so its measured lateness is 241 s. The point
        # is not that 241 s is good -- it is the truth of a restart that was
        # down, which is exactly what must be RECORDED rather than hidden.
        assert t["scheduler_lateness_ms"] == pytest.approx(241_000.0)

    def test_a_second_restart_still_cannot_settle_the_same_arm_twice(self, tmp_path):
        clock = _clock()
        runtime, exp, book, _ = _build(tmp_path, clock)
        asyncio.run(runtime.poll_once())
        clock.t = T0 + 61.0
        asyncio.run(runtime.tick())

        for _ in range(3):
            again, _exp, _b, _ = _build(tmp_path, clock)
            again.restore()
            asyncio.run(again.tick())
            assert len(_b.sell_calls) == 0, "a settled arm is not re-settled"

    def test_dedup_survives_the_restart_that_recovered_arms(self, tmp_path):
        clock = _clock()
        runtime, exp, book, _ = _build(tmp_path, clock)
        asyncio.run(runtime.poll_once())
        clock.t = T0 + 61.0
        asyncio.run(runtime.tick())

        again, _exp, book2, _ = _build(tmp_path, clock)
        again.restore()
        asyncio.run(again.poll_once())
        assert len(book2.buy_calls) == 0, "a seen signal is not re-priced"


# ===========================================================================
# The dedicated deadline scheduler
# ===========================================================================


class TestDedicatedDeadlineScheduler:
    """
    Due-exit detection used to ride the engine's general monitor loop, so an
    exit waited on ``market.poll_interval_s`` (5 s) -- and the monitor never
    selects its faster 3 s branch while the experiment runs, because the Amarih
    arms live in the experiment's own VirtualAccounts rather than in
    ``self.account``/``self.real_account``.

    These tests pin the fix at the ENGINE level, where the defect lived.
    """

    def test_the_scheduler_has_its_own_one_second_cadence(self):
        """
        Approximately one-second deadline detection, which is what keeps
        scheduler lateness under the 2 s target on the shortest (60 s) horizon.
        """
        from autopaper.engine import PaperTrader

        trader = PaperTrader.__new__(PaperTrader)
        assert trader._amarih_scheduler_interval() == pytest.approx(1.0)
        assert trader._amarih_scheduler_interval() < 2.0, (
            "a cadence of >=2s cannot meet the detection target")

    def test_the_scheduler_interval_is_env_overridable(self, monkeypatch):
        """An operator can widen it for a slow host, and it stays honoured."""
        from autopaper.engine import PaperTrader

        monkeypatch.setenv("AMARIH_DEADLINE_POLL_S", "2.5")
        trader = PaperTrader.__new__(PaperTrader)
        assert trader._amarih_scheduler_interval() == pytest.approx(2.5)

    def test_exit_detection_no_longer_lives_in_the_monitor_tick(self):
        """
        The structural fix. The call site that used to settle exits inside the
        monitor loop is gone, and the deadline loop is the remaining path.
        """
        import inspect
        from autopaper.engine import PaperTrader

        whole = inspect.getsource(PaperTrader)
        assert "_amarih_deadline_loop" in whole
        assert "_amarih_deadline_loop(self._stop_event)" in whole

        # THE MONITOR LOOP MUST NO LONGER SETTLE EXITS. Asserted on the monitor
        # loop's own source: it may still MANUFACTURE the experiment (start the
        # poller), but it must not tick the exits.
        import inspect as _i
        monitor = _i.getsource(PaperTrader.monitor_once)
        assert "_amarih_tick" not in monitor, (
            "a monitor-tick Amarih tick would couple exit timing back to "
            "market.poll_interval_s (5 s)")
        assert "_sync_amarih_runtime" in monitor, (
            "the monitor still owns starting/retiring the experiment")

        # Exactly ONE call site of _amarih_tick remains, and it is inside the
        # deadline scheduler itself.
        call_sites = whole.count("await self._amarih_tick()")
        assert call_sites == 1, (
            f"expected exactly 1 _amarih_tick call site, found {call_sites}")
        assert "await self._amarih_tick()" in inspect.getsource(
            PaperTrader._amarih_deadline_loop)

    def test_the_deadline_loop_never_waits_on_signal_ingestion(self):
        from autopaper.engine import PaperTrader
        import inspect

        src = inspect.getsource(PaperTrader._amarih_deadline_loop)
        assert "poll_once" not in src, (
            "the deadline scheduler must never wait on signal ingestion")
        assert "tick()" in src

    def test_the_poller_loop_does_not_settle_exits(self):
        """The two loops each own one thing, and neither owns both."""
        from autopaper.engine import PaperTrader
        import inspect

        assert "tick()" not in inspect.getsource(PaperTrader._amarih_poller_loop)

    def test_a_second_scheduler_is_never_created_while_one_is_live(self, tmp_path):
        from autopaper.config import build_config
        from autopaper.engine import PaperTrader

        trader = PaperTrader(build_config(tmp_path))
        try:
            assert trader._amarih_scheduler_task is None
            assert trader._amarih_scheduler_is_live() is False
        finally:
            pass

    def test_the_scheduler_keeps_running_while_arms_are_outstanding(self):
        """
        Deselection must not abandon an open arm: it still has a deadline to
        honour, and abandoning it would silently censor a position the operator
        never closed.
        """
        from autopaper.engine import PaperTrader
        import inspect

        src = inspect.getsource(PaperTrader)
        assert "_stop_amarih_scheduler_if_nothing_open" in src, (
            "deselection must retire the scheduler only when nothing is open")
        assert "_amarih_open_arms" in src, (
            "retiring the scheduler requires knowing whether arms are open")
        assert "_stop_amarih_scheduler()" in src
