"""
The Amarih PAPER runtime driver, end to end, offline.

WHAT THIS PROVES
================
Each test below drives :class:`AmarihRuntime` through the REAL
:class:`PumpViewAmarihSource` normaliser and the REAL
:class:`AmarihExperiment`, and injects only the two things that would otherwise
reach the network: the executable BUY quote and the executable SELL quote. That
is the same seam the production engine uses, so passing here means the lifecycle
-- poll, price once, fan out three paired arms, fire each arm's scheduled exit,
settle through the canonical path, survive a restart -- is wired, not merely
imported.

Everything unreachable is FAKED EXPLICITLY AND BY NAME. No test claims a live
end-to-end run; that requires the running PAPER runtime against live PumpView and
is called out separately in the report.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from autopaper.paper_live.amarih_experiment import (
    AMARIH_ARMS,
    AmarihArm,
    AmarihExperiment,
)
from autopaper.paper_live.amarih_runtime import (
    SOURCE_DISABLED_BY_AMARIH,
    AmarihRuntime,
)
from autopaper.paper_live.pumpview import IngestReport, PumpViewAmarihSource

T0 = 1_800_000_000.0
MINT_A = "4RiJPcL6GXecaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
MINT_B = "GTGxYY7P4snJbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


class Clock:
    def __init__(self, t: float = T0) -> None:
        self.t = t

    def now(self) -> float:
        return self.t


def _event(signal_id: str, mint: str, fired_at: float, *, score: float = 6.0) -> dict:
    return {
        "id": signal_id,
        "ca": mint,
        "sym": "TKN",
        "signal_type": "Amarih",
        "buy_score": score,
        "fired_at_ts": fired_at,
        "mc_signal": 21_520.0,
        "price_signal": 0.0000204,
        "owner_wallet": "tg:1458395479",
    }


class FakeSource:
    """The real adapter, fed a canned payload instead of HTTP."""

    def __init__(self, clock: Clock, payload_factory) -> None:
        self._adapter = PumpViewAmarihSource(max_age_s=120.0, clock=clock)
        self._clock = clock
        self._payload_factory = payload_factory
        self.poll_calls = 0

    @property
    def seen_ids(self) -> set:
        return self._adapter.seen_ids

    def poll_once(self, *, limit: int = 200, offset: int = 0, timeout_s: float = 30.0):
        self.poll_calls += 1
        return self._adapter.ingest_events(
            self._payload_factory(), now=self._clock.now())


class FailingSource(FakeSource):
    def poll_once(self, **_kwargs):
        raise RuntimeError("pumpview unreachable")


class FlakySource(FakeSource):
    """Fails while ``fail`` is set, answers otherwise."""

    def __init__(self, clock: Clock, payload_factory) -> None:
        super().__init__(clock, payload_factory)
        self.fail = True

    def poll_once(self, **kwargs):
        if self.fail:
            raise RuntimeError("pumpview unreachable")
        return super().poll_once(**kwargs)


def _fresh(*events: dict):
    return lambda: {"signals": list(events)}


class QuoteBook:
    """Executable BUY/SELL quotes, with every refusal switchable by name."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.buy: dict = {}
        self.sell: dict = {}
        self.buy_calls: list[tuple[str, float]] = []
        self.sell_calls: list[tuple] = []

    async def buy_quote(self, mint: str, usd: float):
        self.buy_calls.append((mint, usd))
        override = self.buy.get(mint, "default")
        if override is None:
            return None
        if override == "raise":
            raise RuntimeError("quote provider down")
        if isinstance(override, dict):
            return override
        return {
            "identity": "verified",
            "token_quantity": 1000.0,
            "token_quantity_raw": 1_000_000_000,
            "token_decimals": 6,
            "provider": "pumpportal",
            "venue": "pumpfun",
            "market_id": f"pair:{mint}",
            "quote_ts": self.clock.now(),
            "fees_usd": 0.01,
            "quote_asset": "SOL",
        }

    async def sell_quote(self, mint: str, qty: float, decimals: int):
        self.sell_calls.append((mint, qty, decimals))
        override = self.sell.get(mint, "default")
        if override is None:
            return None
        if override == "raise":
            raise RuntimeError("quote provider down")
        if isinstance(override, dict):
            return override
        return {
            "identity": "verified",
            "proceeds_usd": 11.0,
            "provider": "pumpportal",
            "venue": "pumpfun",
            "quote_ts": self.clock.now(),
            "fees_usd": 0.01,
        }


def build(tmp_path: Path, clock: Clock, source, *, notify=None):
    book = QuoteBook(clock)
    experiment = AmarihExperiment(clock=clock)
    sent: list[str] = []
    runtime = AmarihRuntime(
        source=source,
        experiment=experiment,
        state_path=tmp_path / "var" / "amarih_runtime.json",
        buy_quote=book.buy_quote,
        sell_quote=book.sell_quote,
        notify=notify or sent.append,
        clock=clock,
    )
    return runtime, experiment, book, sent


# ---------------------------------------------------------------------------
# entry: one observation, three paired arms
# ---------------------------------------------------------------------------

def test_a_fresh_signal_opens_exactly_three_paired_arms(tmp_path):
    clock = Clock()
    source = FakeSource(clock, _fresh(_event("1", MINT_A, T0 - 5)))
    runtime, experiment, book, sent = build(tmp_path, clock, source)

    summary = __import__("asyncio").run(runtime.poll_once())

    assert summary["accepted"] == 1
    assert summary["arms_opened"] == 3
    # ONE quote for the notional, not one per arm.
    assert book.buy_calls == [(MINT_A, 10.0)]
    assert set(experiment.open_positions()) == {
        ("1", AmarihArm.T60), ("1", AmarihArm.T120), ("1", AmarihArm.T300)}
    assert len(experiment.entry_records()) == 1
    entry = experiment.entry_records()[0]
    assert entry.validation == "VALID"
    assert entry.token_quantity == 1000.0

    horizons = {a: p.scheduled_deadline - p.entry_ts
                for (sid, a), p in experiment.open_positions().items()}
    assert horizons[AmarihArm.T60] == 60.0
    assert horizons[AmarihArm.T120] == 120.0
    assert horizons[AmarihArm.T300] == 300.0

    assert any("1 entry observation" in m for m in sent)
    assert any("PAIRED outcomes of ONE signal" in m for m in sent), (
        "the operator must be told the arms are paired, not three signals")


def test_every_arm_has_its_own_isolated_ledger(tmp_path):
    clock = Clock()
    source = FakeSource(clock, _fresh(_event("1", MINT_A, T0 - 5)))
    runtime, experiment, _book, _sent = build(tmp_path, clock, source)
    __import__("asyncio").run(runtime.poll_once())
    assert len({id(experiment.accounts[a]) for a in AMARIH_ARMS}) == 3


# ---------------------------------------------------------------------------
# refusals fail closed
# ---------------------------------------------------------------------------

def test_a_stale_signal_opens_nothing(tmp_path):
    clock = Clock()
    source = FakeSource(clock, _fresh(_event("1", MINT_A, T0 - 600)))
    runtime, experiment, book, sent = build(tmp_path, clock, source)

    summary = __import__("asyncio").run(runtime.poll_once())

    assert summary["accepted"] == 0
    assert summary["arms_opened"] == 0
    assert book.buy_calls == [], "a stale signal must not be priced at all"
    assert experiment.open_positions() == {}
    assert runtime.counters.stale == 1


def test_an_unavailable_buy_quote_opens_nothing_and_says_why(tmp_path):
    clock = Clock()
    source = FakeSource(clock, _fresh(_event("1", MINT_A, T0 - 5)))
    runtime, experiment, book, sent = build(tmp_path, clock, source)
    book.buy[MINT_A] = None

    __import__("asyncio").run(runtime.poll_once())

    assert experiment.open_positions() == {}
    assert len(experiment.entry_records()) == 1
    assert experiment.entry_records()[0].validation == "QUOTE_UNAVAILABLE"
    assert any("NOT traded" in m and "QUOTE_UNAVAILABLE" in m for m in sent)


def test_an_unknown_market_identity_fails_closed(tmp_path):
    clock = Clock()
    source = FakeSource(clock, _fresh(_event("1", MINT_A, T0 - 5)))
    runtime, experiment, book, _sent = build(tmp_path, clock, source)
    book.buy[MINT_A] = {"identity": "UNKNOWN", "token_quantity": 1000.0,
                        "token_decimals": 6}

    __import__("asyncio").run(runtime.poll_once())

    assert experiment.open_positions() == {}
    assert experiment.entry_records()[0].validation == "IDENTITY_UNKNOWN"


def test_a_mismatched_market_identity_is_rejected(tmp_path):
    clock = Clock()
    source = FakeSource(clock, _fresh(_event("1", MINT_A, T0 - 5)))
    runtime, experiment, book, _sent = build(tmp_path, clock, source)
    book.buy[MINT_A] = {"identity": "mismatch", "token_quantity": 1000.0,
                        "token_decimals": 6}

    __import__("asyncio").run(runtime.poll_once())

    assert experiment.open_positions() == {}
    record = experiment.entry_records()[0]
    assert record.validation == "IDENTITY_UNKNOWN"
    assert "IDENTITY_MISMATCH" in (record.rejection_reason or "")


def test_a_buy_quote_that_raises_is_a_refusal_not_a_crash(tmp_path):
    clock = Clock()
    source = FakeSource(clock, _fresh(_event("1", MINT_A, T0 - 5)))
    runtime, experiment, book, _sent = build(tmp_path, clock, source)
    book.buy[MINT_A] = "raise"

    __import__("asyncio").run(runtime.poll_once())

    assert experiment.open_positions() == {}
    assert experiment.entry_records()[0].validation == "QUOTE_UNAVAILABLE"


# ---------------------------------------------------------------------------
# dedup
# ---------------------------------------------------------------------------

def test_the_same_signal_is_never_priced_twice(tmp_path):
    clock = Clock()
    source = FakeSource(clock, _fresh(_event("1", MINT_A, T0 - 5)))
    runtime, experiment, book, _sent = build(tmp_path, clock, source)

    __import__("asyncio").run(runtime.poll_once())
    clock.t = T0 + 20
    __import__("asyncio").run(runtime.poll_once())

    assert len(book.buy_calls) == 1, "the second poll must not re-price"
    assert len(experiment.entry_records()) == 1


# ---------------------------------------------------------------------------
# scheduled exits and canonical settlement
# ---------------------------------------------------------------------------

def test_each_arm_settles_at_its_own_scheduled_deadline(tmp_path):
    clock = Clock()
    source = FakeSource(clock, _fresh(_event("1", MINT_A, T0 - 5)))
    runtime, experiment, book, sent = build(tmp_path, clock, source)
    asyncio = __import__("asyncio")
    asyncio.run(runtime.poll_once())

    # Nothing is due before the shortest deadline.
    clock.t = T0 + 59
    assert asyncio.run(runtime.tick())["due"] == 0

    clock.t = T0 + 61
    first = asyncio.run(runtime.tick())
    # The count fields. Asserted by key rather than by whole-dict equality:
    # ``tick()`` now also returns ``max_scheduler_lateness_ms``, a diagnostic
    # and not an outcome, and a test that pinned the dict verbatim would have
    # been asserting the absence of diagnostics rather than the presence of a
    # correct result.
    assert first["due"] == 1
    assert first["settled"] == 1
    assert first["censored"] == 0
    assert set(experiment.open_positions()) == {
        ("1", AmarihArm.T120), ("1", AmarihArm.T300)}

    clock.t = T0 + 121
    assert asyncio.run(runtime.tick())["settled"] == 1
    clock.t = T0 + 301
    assert asyncio.run(runtime.tick())["settled"] == 1
    assert experiment.open_positions() == {}

    # The exit leg must quote the EXACT integer quantity the entry acquired.
    assert book.sell_calls and all(
        call[1] == 1_000_000_000 for call in book.sell_calls), book.sell_calls

    results = experiment.results()["1"]
    assert set(results) == set(AMARIH_ARMS)
    for arm, result in results.items():
        assert result.settlement_status == "SETTLED"
        assert result.classification == "NORMAL"
        assert result.net_sell_proceeds_usd == 11.0
        assert result.net_pnl_usd == pytest.approx(1.0)
        assert result.net_return_pct == pytest.approx(10.0)

    for arm in AMARIH_ARMS:
        account = experiment.accounts[arm]
        assert account.bankroll_usd == pytest.approx(11.0), (
            "each arm must settle its OWN ledger")

    assert sum("SETTLED" in m for m in sent) == 3


def test_a_missing_exit_quote_never_fabricates_a_fill(tmp_path):
    """
    A refused exit quote is CENSORED, and the deadline outcome is FINAL.

    Accepting a later, recovered quote would convert a censored arm into a
    settled one and bias every horizon the same way, so the arm is retired with
    its recorded reason instead.
    """
    clock = Clock()
    source = FakeSource(clock, _fresh(_event("1", MINT_A, T0 - 5)))
    runtime, experiment, book, sent = build(tmp_path, clock, source)
    asyncio = __import__("asyncio")
    asyncio.run(runtime.poll_once())
    book.sell[MINT_A] = None

    clock.t = T0 + 61
    summary = asyncio.run(runtime.tick())

    assert summary["due"] == 1 and summary["settled"] == 0
    assert summary["censored"] == 1
    result = experiment.results()["1"][AmarihArm.T60]
    assert result.settlement_status == "QUOTE_UNAVAILABLE"
    assert result.net_pnl_usd is None, "no quote must never mean a zero PnL"
    assert experiment.accounts[AmarihArm.T60].bankroll_usd == pytest.approx(10.0)
    assert any("NOT settled" in m for m in sent)
    assert runtime.counters.arms_censored == 1

    # Retired, not retried: the deadline observation stands.
    assert ("1", AmarihArm.T60.value) not in runtime._open_arms
    book.sell[MINT_A] = "default"
    clock.t = T0 + 500
    asyncio.run(runtime.tick())
    assert experiment.results()["1"][AmarihArm.T60].settlement_status \
        == "QUOTE_UNAVAILABLE", "a later quote must not overwrite the deadline"


def test_a_stale_exit_quote_is_refused(tmp_path):
    clock = Clock()
    source = FakeSource(clock, _fresh(_event("1", MINT_A, T0 - 5)))
    runtime, experiment, book, _sent = build(tmp_path, clock, source)
    asyncio = __import__("asyncio")
    asyncio.run(runtime.poll_once())
    book.sell[MINT_A] = {"identity": "verified", "proceeds_usd": 11.0,
                         "quote_ts": clock.now() - 600.0}

    clock.t = T0 + 61
    asyncio.run(runtime.tick())
    assert experiment.results()["1"][AmarihArm.T60].settlement_status == "STALE_QUOTE"


def test_an_identity_mismatched_exit_quote_is_refused(tmp_path):
    clock = Clock()
    source = FakeSource(clock, _fresh(_event("1", MINT_A, T0 - 5)))
    runtime, experiment, book, _sent = build(tmp_path, clock, source)
    asyncio = __import__("asyncio")
    asyncio.run(runtime.poll_once())
    book.sell[MINT_A] = {"identity": "mismatch", "proceeds_usd": 11.0,
                         "quote_ts": clock.now()}

    clock.t = T0 + 61
    asyncio.run(runtime.tick())
    result = experiment.results()["1"][AmarihArm.T60]
    assert result.settlement_status == "IDENTITY_MISMATCH"


# ---------------------------------------------------------------------------
# restart
# ---------------------------------------------------------------------------

def test_a_restart_restores_outstanding_arms_and_settles_them(tmp_path):
    clock = Clock()
    source = FakeSource(clock, _fresh(_event("1", MINT_A, T0 - 5)))
    runtime, experiment, _book, _sent = build(tmp_path, clock, source)
    asyncio = __import__("asyncio")
    asyncio.run(runtime.poll_once())
    assert runtime.open_arm_count() == 3
    state_file = tmp_path / "var" / "amarih_runtime.json"
    assert state_file.exists()
    on_disk = json.loads(state_file.read_text())
    assert len(on_disk["open_arms"]) == 3
    assert "1" in on_disk["seen_signal_ids"]

    # "Restart": a brand-new experiment and driver over the same var/ directory.
    clock.t = T0 + 100
    source2 = FakeSource(clock, _fresh(_event("1", MINT_A, T0 - 5)))
    runtime2, experiment2, book2, sent2 = build(tmp_path, clock, source2)
    report = runtime2.restore()

    assert report["restored_arms"] == 3
    assert runtime2.open_arm_count() == 3

    # The already-accepted id is NOT re-entered after the restart.
    asyncio.run(runtime2.poll_once())
    assert book2.buy_calls == [], "a restart must not re-open a seen signal"

    clock.t = T0 + 121
    summary = asyncio.run(runtime2.tick())
    assert summary["settled"] == 2, "T60 and T120 are due at T0+121"
    assert experiment2.results()["1"][AmarihArm.T120].settlement_status == "SETTLED"
    assert any("SETTLED" in m for m in sent2)


def test_a_restart_does_not_revive_an_already_settled_arm(tmp_path):
    """
    The durable ``open_arms`` list is the authority across a restart.

    A settled arm is REMOVED from it, so a replay of the same on-disk state
    cannot re-open a position whose outcome is already recorded -- the
    protection has to hold across processes, not just within one.
    """
    clock = Clock()
    source = FakeSource(clock, _fresh(_event("1", MINT_A, T0 - 5)))
    runtime, experiment, _book, _sent = build(tmp_path, clock, source)
    asyncio = __import__("asyncio")
    asyncio.run(runtime.poll_once())
    clock.t = T0 + 61
    asyncio.run(runtime.tick())
    assert ("1", AmarihArm.T60.value) not in runtime._open_arms

    runtime2, experiment2, _book2, _sent2 = build(tmp_path, clock, source)
    runtime2.restore()
    assert runtime2.open_arm_count() == 2
    assert ("1", AmarihArm.T60.value) not in runtime2._open_arms
    assert set(experiment2.open_positions()) == {
        ("1", AmarihArm.T120), ("1", AmarihArm.T300)}

    # And the settled arm is not re-settled by a later tick.
    clock.t = T0 + 400
    asyncio.run(runtime2.tick())
    assert set(experiment2.results().get("1", {})) == {
        AmarihArm.T120, AmarihArm.T300}, (
        "only the two arms still outstanding may settle, and each exactly once")


def test_delivered_notifications_are_not_repeated_after_a_restart(tmp_path):
    clock = Clock()
    source = FakeSource(clock, _fresh(_event("1", MINT_A, T0 - 5)))
    runtime, _experiment, _book, sent = build(tmp_path, clock, source)
    asyncio = __import__("asyncio")
    asyncio.run(runtime.poll_once())
    assert len(sent) == 2, "the signal and the arms-opened message"
    assert runtime.flush_notifications() == 0, "nothing left queued"

    runtime2, _experiment2, _book2, sent2 = build(tmp_path, clock, source)
    runtime2.restore()
    runtime2._queue("signal:1", "duplicate")
    assert runtime2.flush_notifications() == 0
    assert sent2 == []


# ---------------------------------------------------------------------------
# poller failure and status
# ---------------------------------------------------------------------------

def test_a_slow_poll_does_not_stall_the_event_loop(tmp_path):
    """
    PumpView is remote, so its latency is unbounded from our side.

    The adapter's transport is a synchronous ``httpx`` call. Awaiting it on the
    loop would freeze the monitor tick that services every open PAPER position
    for as long as the remote takes to answer -- the same defect the market
    stack already fixed by moving its provider transport to a worker thread.
    """
    import time

    clock = Clock()

    class SlowSource(FakeSource):
        """A 0.4s blocking send, like httpx against a slow origin."""

        def poll_once(self, **_kwargs):
            time.sleep(0.4)
            return IngestReport()

    runtime, _experiment, _book, _sent = build(
        tmp_path, clock, SlowSource(clock, _fresh()))

    async def scenario() -> int:
        ticks = 0

        async def heartbeat() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        hb = asyncio.create_task(heartbeat())
        await runtime.poll_once()
        hb.cancel()
        try:
            await hb
        except asyncio.CancelledError:
            pass
        return ticks

    ticks = asyncio.run(scenario())
    assert ticks >= 10, (
        f"the loop advanced only {ticks} times across a 0.4s poll; the blocking "
        "fetch is running on the event loop instead of a worker thread")


def test_a_poll_failure_is_reported_and_does_not_raise(tmp_path):
    clock = Clock()
    source = FailingSource(clock, _fresh())
    runtime, _experiment, _book, sent = build(tmp_path, clock, source)

    summary = __import__("asyncio").run(runtime.poll_once())

    assert summary["ok"] is False
    assert "poll failed" in runtime.last_error
    assert any("poll FAILED" in m for m in sent)


def test_a_continuing_poll_failure_notifies_only_once(tmp_path):
    """
    AN OUTAGE IS ONE EVENT, NOT ONE PER POLL CYCLE.

    The key used to include the cycle's own timestamp, so at the 15s cadence an
    hour-long PumpView outage sent ~240 identical pages. An operator paged every
    15 seconds stops reading the pages -- and the first, informative one is
    buried. The clock is advanced here precisely because that is what the defect
    needed in order to fire at all.
    """
    clock = Clock()
    source = FailingSource(clock, _fresh())
    runtime, _experiment, _book, sent = build(tmp_path, clock, source)

    for _ in range(5):
        clock.t += 15.0                      # five poll cycles
        asyncio.run(runtime.poll_once())

    failures = [m for m in sent if "poll FAILED" in m]
    assert len(failures) == 1, (
        f"a five-cycle outage sent {len(failures)} pages; it must send one")
    assert runtime.counters.polls == 5, "every cycle must still be attempted"
    assert "RuntimeError" in failures[0], "the page must name the failure"


def test_a_new_outage_after_a_recovery_notifies_again(tmp_path):
    """
    Deduplication must not become silence.

    A failure after a successful poll is a NEW event, so an operator who fixed
    the first outage is told about the second one.
    """
    clock = Clock()
    source = FlakySource(clock, _fresh())
    runtime, _experiment, _book, sent = build(tmp_path, clock, source)

    for _ in range(3):
        clock.t += 15.0
        asyncio.run(runtime.poll_once())      # outage
    source.fail = False
    clock.t += 15.0
    asyncio.run(runtime.poll_once())          # recovered
    source.fail = True
    for _ in range(3):
        clock.t += 15.0
        asyncio.run(runtime.poll_once())      # NEW outage

    failures = [m for m in sent if "poll FAILED" in m]
    assert len(failures) == 2, (
        f"two separate outages sent {len(failures)} pages; it must send two")
    assert runtime._poller_failure_reported is True


def test_a_restart_mid_outage_pages_again(tmp_path):
    """
    The outage flag is process state, the message keys are durable.

    A process that starts while PumpView is still down must not stay silent
    because a previous process already reported an outage.
    """
    clock = Clock()
    source = FailingSource(clock, _fresh())
    runtime, _experiment, _book, sent = build(tmp_path, clock, source)
    asyncio.run(runtime.poll_once())
    assert len([m for m in sent if "poll FAILED" in m]) == 1

    # A NEW process over the same durable state, still down, later in time.
    clock.t += 3600.0
    rebuilt, _e2, _b2, sent2 = build(tmp_path, clock, source)
    rebuilt.restore()
    asyncio.run(rebuilt.poll_once())

    assert len([m for m in sent2 if "poll FAILED" in m]) == 1, (
        "a restart mid-outage must report the continuing outage")


def test_status_reports_poller_counters_and_open_arms(tmp_path):
    clock = Clock()
    source = FakeSource(clock, _fresh(_event("1", MINT_A, T0 - 5)))
    runtime, _experiment, _book, _sent = build(tmp_path, clock, source)
    asyncio = __import__("asyncio")
    asyncio.run(runtime.poll_once())

    status = runtime.status()
    assert status["running"] is False
    assert status["last_poll_at"] == T0
    assert status["signals_accepted"] == 1
    assert status["open_arms"] == 3
    assert status["last_error"] is None


def test_the_disabled_source_reason_is_a_named_constant():
    assert SOURCE_DISABLED_BY_AMARIH == "SOURCE_DISABLED_BY_AMARIH"


def test_valid_entry_with_insufficient_bankroll_reports_economic_reason(tmp_path):
    """A valid quote that cannot open arms due to economics must report why."""
    clock = Clock()
    source = FakeSource(clock, _fresh(_event("162759", MINT_A, T0 - 5)))
    runtime, experiment, book, sent = build(tmp_path, clock, source)
    book.buy[MINT_A] = {
        "identity": "verified",
        "token_quantity": 1000.0,
        "token_quantity_raw": 1_000_000_000,
        "token_decimals": 6,
        "input_amount_usd": 100.0,  # exceeds bankroll
        "provider": "pumpportal",
        "venue": "pumpfun",
        "market_id": f"pair:{MINT_A}",
        "quote_ts": clock.now(),
        "fees_usd": 0.01,
        "quote_asset": "SOL",
    }
    __import__("asyncio").run(runtime.poll_once())
    assert experiment.open_positions() == {}
    assert any("NOT traded" in m for m in sent)
    assert any("VALID" in m for m in sent)
    # Must include a meaningful reason, not just VALID
    refused = [m for m in sent if "NOT traded" in m]
    assert refused, sent
    assert "ECONOMICALLY_INVALID" in refused[0] or "cost_usd" in refused[0]


def test_buy_quote_raising_fillrefused_reports_correctly(tmp_path):
    """When buy quote raises FillRefused, report QUOTE_UNAVAILABLE with reason."""
    from autopaper.paper_live.fills import FillRefused
    from autopaper.paper_live.schema import FailureReason
    
    clock = Clock()
    source = FakeSource(clock, _fresh(_event("162750", MINT_A, T0 - 5)))
    runtime, experiment, book, sent = build(tmp_path, clock, source)
    
    async def raise_fillrefused(m, u):
        raise FillRefused(FailureReason.ENTRY_REJECTED_QUOTE_UNAVAILABLE, "no route")
    
    runtime.buy_quote = raise_fillrefused
    __import__("asyncio").run(runtime.poll_once())
    assert experiment.open_positions() == {}
    refused = [m for m in sent if "NOT traded" in m]
    assert refused
    assert "QUOTE_UNAVAILABLE" in refused[0]
    assert "FillRefused" in refused[0]
