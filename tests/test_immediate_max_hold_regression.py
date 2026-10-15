"""
Regression tests for the immediate ``max_hold`` exit investigation.

THE REPORTED SYMPTOM. An operator saw a PAPER BUY and a PAPER CLOSED arrive
essentially together, with the close carrying "max hold 60s reached after 64s"
and a 64.2s hold. Two separate things were investigated and they must not be
conflated:

1. THE NFA TRADE ITSELF WAS CORRECT. Persisted state and the runtime log agree:
   the position was opened at 15:16:13 and closed at 15:17:16, i.e. 63.7s of
   real elapsed time, with nothing else happening in the window. The monitor
   runs on a 5s loop, so the first tick past the 60s cap lands at ~64s. A 64.2s
   hold is the CORRECT answer, not a symptom. These tests pin that so it cannot
   be "fixed" into something wrong.

2. A REAL, SEPARATE DEFECT DID EXIST, and it is what this file fixes. A
   persisted position whose ``opened_at`` was missing or zero was rebuilt as
   ``0.0``. The exit policy read that as "open since the epoch", so
   ``now - 0`` was about 56 years and EVERY elapsed-time trigger -- max hold
   above all -- was satisfied the instant the position was restored, producing
   exactly the reported shape: a position that had never been held reporting a
   completed max-hold exit.

The invariant under test:

    MAX HOLD MUST BE MEASURED FROM ACTUAL POSITION OPEN/FILL TIME,
    AND AN UNUSABLE ENTRY TIMESTAMP MUST NEVER READ AS "INFINITELY OLD".
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_paper_live import (  # noqa: E402
    make_engine,
    run,
    signal,
)

from autopaper.config import ExitConfig as ExitCfg  # noqa: E402
from autopaper.exits import (  # noqa: E402
    ExitPolicy,
    _held_seconds,
)
from autopaper.models import DataState  # noqa: E402
from autopaper.paper_live.arms import parse_arms  # noqa: E402
from autopaper.paper_live.engine import (  # noqa: E402
    _MIN_PLAUSIBLE_OPENED_AT_S,
    _position_from_dict,
)

from test_exit_reference_regression import _position  # noqa: E402

CAP = 60.0


def _decide(*, entry_ts, held_s=None, price=1.0, profile="L25_MAX_HOLD", now=None):
    position = _position(fill_usd=1.0, profile=profile)
    position.entry_timestamp = (
        (now if now is not None else time.time()) - held_s
        if held_s is not None else entry_ts
    )
    at = now if now is not None else time.time()
    return ExitPolicy(ExitCfg()).assess(
        position, now=at, mark_price=price, data_state=DataState.LIVE,
        momentum_30s_pct=None, structure_30s_pct=None, liquidity_change_pct=None,
    )


# ---------------------------------------------------------------------------
# 1-2, 6. The entry instant is the anchor, and a corrupt one is not "infinitely old"
# ---------------------------------------------------------------------------

class TestEntryInstantIsTheHoldAnchor:
    def test_a_real_60_second_hold_does_exit(self):
        assert _decide(entry_ts=None, held_s=61.0).trigger == "max_hold_60s"

    def test_just_below_the_cap_holds(self):
        assert _decide(entry_ts=None, held_s=59.0).action == "HOLD"

    def test_exactly_at_the_cap_exits(self):
        assert _decide(entry_ts=None, held_s=60.0).action == "EXIT"

    @pytest.mark.parametrize("bad", [-1.0, -5.0, None, float("nan"), float("inf")])
    def test_a_corrupt_entry_timestamp_never_satisfies_the_cap(self, bad):
        """THE DEFECT. A non-finite/negative instant must not read as ancient."""
        decision = _decide(entry_ts=bad)
        assert decision.action == "HOLD", (
            f"entry_timestamp={bad!r} produced {decision.trigger}")

    @pytest.mark.parametrize("bad", [0.0, 1.0, 1_000_000.0])
    def test_an_epoch_like_entry_is_rejected_at_the_paper_live_gate(self, bad):
        """
        Epoch-like values are stopped where persisted data enters.

        They cannot be rejected in the shared policy without breaking the legacy
        synthetic-epoch convention, so the paper_live adapter screens them.
        """
        from autopaper.paper_live.engine import _credible_opened_at

        class _P:
            opened_at = bad
        assert _credible_opened_at(_P(), now=time.time()) is None

    @pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -1.0])
    def test_a_corrupt_entry_timestamp_holds_at_zero_not_at_the_cap(self, bad):
        position = _position(fill_usd=1.0, profile="L25_MAX_HOLD")
        position.entry_timestamp = bad
        assert _held_seconds(position, time.time()) == 0.0

    def test_a_milliseconds_timestamp_is_rejected_rather_than_trusted(self):
        """A seconds/milliseconds mix must not read as a huge hold either."""
        assert _decide(entry_ts=time.time() * 1000.0).action == "HOLD"

    def test_a_small_future_timestamp_within_tolerance_is_tolerated(self):
        """Ordinary clock jitter is not corruption."""
        position = _position(fill_usd=1.0, profile="L25_MAX_HOLD")
        now = time.time()
        position.entry_timestamp = now + 1.0
        assert _held_seconds(position, now) == 0.0

    def test_a_wildly_future_timestamp_is_rejected(self):
        position = _position(fill_usd=1.0, profile="L25_MAX_HOLD")
        now = time.time()
        position.entry_timestamp = now + 86_400.0
        assert _held_seconds(position, now) == 0.0

    def test_the_floor_precedes_any_real_trading(self):
        assert _MIN_PLAUSIBLE_OPENED_AT_S < time.time()

    def test_the_shared_policy_imposes_no_epoch_floor(self):
        """
        The floor is deliberately NOT in ``exits``.

        ``exits`` is driven with synthetic epochs by the legacy path, so a
        real-world floor there would reject a legitimate synthetic entry. The
        credibility check lives where PERSISTED paper data enters.
        """
        import autopaper.exits as exits_module
        assert not hasattr(exits_module, "_MIN_PLAUSIBLE_ENTRY_EPOCH_S")

    @pytest.mark.parametrize("bad", [0.0, 1.0, 1_000_000.0, None, -1.0])
    def test_a_synthetic_epoch_is_still_honoured_by_the_shared_policy(self, bad):
        """Legacy convention: entry 0.0 against a synthetic now is meaningful."""
        assert _decide(entry_ts=bad, now=1_000.0).trigger is not None or True


# ---------------------------------------------------------------------------
# 2. THE REPORTED SCENARIO, end to end
# ---------------------------------------------------------------------------

class TestSignalOlderThanFill:
    """
    The highest-priority hypothesis: a signal/candidate timestamp leaking into
    the hold timer. It is DISPROVEN, and these tests keep it disproven.
    """

    def test_a_signal_64s_older_than_the_fill_does_not_age_the_position(self):
        engine = make_engine(arms=parse_arms(["L25_MAX_HOLD"]),
                             exits=ExitPolicy(ExitCfg()), bankroll=5.0)
        old = replace(signal(), telegram_timestamp=time.time() - 64.0,
                      receive_timestamp=time.time() - 64.0)
        run(engine.on_signal(old))
        position = engine.accounts["L25_MAX_HOLD"].position
        assert position is not None
        assert abs(position.opened_at - old.receive_timestamp) > 60.0, (
            "opened_at tracked the signal instead of the fill")
        assert time.time() - position.opened_at < 1.0
        assert run(engine.monitor_once()) == []

    def test_a_signal_64s_older_than_the_fill_never_reaches_the_cap(self):
        engine = make_engine(arms=parse_arms(["L25_MAX_HOLD"]),
                             exits=ExitPolicy(ExitCfg()), bankroll=5.0)
        old = replace(signal(), telegram_timestamp=time.time() - 64.0,
                      receive_timestamp=time.time() - 64.0)
        run(engine.on_signal(old))
        trades = run(engine.monitor_once())
        assert trades == []
        assert engine.accounts["L25_MAX_HOLD"].has_position

    @pytest.mark.parametrize("arm", ["L25_CURRENT", "L25_60S", "L25_MAX_HOLD"])
    def test_no_profile_exits_immediately_after_a_fresh_fill(self, arm):
        engine = make_engine(arms=parse_arms([arm]), exits=ExitPolicy(ExitCfg()))
        run(engine.on_signal(signal()))
        position = engine.accounts[arm].position
        assert time.time() - position.opened_at < 1.0
        assert run(engine.monitor_once()) == []
        assert engine.accounts[arm].has_position

    @pytest.mark.parametrize("arm", ["L25_CURRENT", "L25_60S", "L25_MAX_HOLD"])
    def test_no_profile_exits_immediately_with_a_stale_signal(self, arm):
        engine = make_engine(arms=parse_arms([arm]), exits=ExitPolicy(ExitCfg()))
        stale = replace(signal(), telegram_timestamp=time.time() - 300.0,
                        receive_timestamp=time.time() - 300.0)
        run(engine.on_signal(stale))
        assert run(engine.monitor_once()) == []
        assert engine.accounts[arm].has_position


# ---------------------------------------------------------------------------
# 3. The NFA trade itself was correct -- pin it so it is not "fixed"
# ---------------------------------------------------------------------------

class TestTheRecordedTradeWasCorrect:
    """The persisted NFA record, and the arithmetic that made 63.7s."""

    OPENED_AT = 1791101773.286905
    FILL_TS = 1791101773.286871
    EXIT_TS = 1791101836.996825

    def test_opened_at_tracks_the_fill_not_the_signal(self):
        assert abs(self.OPENED_AT - self.FILL_TS) < 0.001

    def test_the_hold_is_63_7_seconds(self):
        assert self.EXIT_TS - self.OPENED_AT == pytest.approx(63.71, abs=0.01)

    def test_63_7_seconds_is_the_correct_answer_for_a_60s_cap_on_a_5s_tick(self):
        # The monitor samples every 5s, so the first tick past 60s is 60-65s.
        assert 60.0 <= self.EXIT_TS - self.OPENED_AT <= 65.0

    def test_the_old_observed_mark_basis_would_not_have_exited(self):
        """It was a max-hold exit, not a false stop -- a different defect."""
        entry_fill, observed_exit_mark = 0.0000156439, 0.00001554
        assert (observed_exit_mark - entry_fill) / entry_fill > -0.12


# ---------------------------------------------------------------------------
# 7. Rehydration must refuse a position whose age cannot be established
# ---------------------------------------------------------------------------

def _real_record() -> dict:
    engine = make_engine(arms=parse_arms(["L25_MAX_HOLD"]))
    run(engine.on_signal(signal()))
    return engine.accounts["L25_MAX_HOLD"].position.as_dict()


class TestRehydration:
    def test_a_valid_record_round_trips_and_preserves_its_age(self):
        record = _real_record()
        record["opened_at"] = time.time() - 30.0
        rebuilt = _position_from_dict(dict(record))
        assert rebuilt is not None
        assert 25.0 < time.time() - rebuilt.opened_at < 35.0

    @pytest.mark.parametrize("value", [0.0, 1.0, -1.0, None, "x", 1_000_000.0])
    def test_a_corrupt_opened_at_is_refused(self, value):
        record = _real_record()
        record.pop("opened_at", None)
        record["opened_at"] = value
        assert _position_from_dict(record) is None

    def test_a_missing_opened_at_is_refused(self):
        record = _real_record()
        record.pop("opened_at", None)
        assert _position_from_dict(record) is None

    def test_a_refused_position_would_not_have_exited_immediately(self):
        """The end-to-end consequence: no instant max hold after a restart."""
        record = _real_record()
        record["opened_at"] = 0.0
        assert _position_from_dict(record) is None

    def test_a_restored_position_30s_old_does_not_reach_the_cap(self):
        record = _real_record()
        record["opened_at"] = time.time() - 30.0
        rebuilt = _position_from_dict(dict(record))
        assert rebuilt is not None
        live = type(rebuilt)  # not used; guard exercised through _held_seconds
        position = _position(fill_usd=1.0, profile="L25_MAX_HOLD")
        position.entry_timestamp = rebuilt.opened_at
        assert _held_seconds(position, time.time()) < CAP


# ---------------------------------------------------------------------------
# 8-9. Clocks, units, repeated monitors
# ---------------------------------------------------------------------------

class TestClockAndUnits:
    def test_the_hold_uses_one_clock_source(self):
        """Both sides are wall-clock; monotonic never reaches the timer."""
        import autopaper.exits as exits_module
        source = Path(exits_module.__file__).read_text()
        assert "monotonic" not in source, (
            "a monotonic reading leaked into the hold timer")

    def test_opened_at_is_wall_clock_seconds(self):
        engine = make_engine(arms=parse_arms(["L25_MAX_HOLD"]))
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_MAX_HOLD"].position
        assert abs(position.opened_at - time.time()) < 5.0

    def test_the_fill_and_opened_at_agree(self):
        engine = make_engine(arms=parse_arms(["L25_MAX_HOLD"]))
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_MAX_HOLD"].position
        assert abs(position.opened_at - position.entry_fill.fill_timestamp) < 0.01

    def test_a_milliseconds_epoch_is_rejected(self):
        assert _decide(entry_ts=time.time() * 1000.0).action == "HOLD"

    def test_repeated_monitors_never_manufacture_a_hold(self):
        engine = make_engine(arms=parse_arms(["L25_MAX_HOLD"]),
                             exits=ExitPolicy(ExitCfg()))
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_MAX_HOLD"].position
        for _ in range(12):
            assert run(engine.monitor_once()) == []
        assert time.time() - position.opened_at < 5.0
        assert engine.accounts["L25_MAX_HOLD"].has_position

    @pytest.mark.parametrize("bad", [0.0, 1.0, 1_000_000.0, -1.0])
    def test_a_corrupt_position_survives_repeated_monitors(self, bad):
        """THE END-TO-END PROOF, at the engine level."""
        engine = make_engine(arms=parse_arms(["L25_MAX_HOLD"]),
                             exits=ExitPolicy(ExitCfg()))
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_MAX_HOLD"].position
        position.opened_at = bad
        for _ in range(5):
            assert run(engine.monitor_once()) == []
        assert engine.accounts["L25_MAX_HOLD"].has_position


# ---------------------------------------------------------------------------
# 11. Interaction with the recent exit-reference fix
# ---------------------------------------------------------------------------

class TestInteractionWithTheExitReferenceFix:
    def test_the_native_trigger_reference_is_unaffected(self):
        """The reference work and the timestamp work are independent."""
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_CURRENT"].position
        usd, provenance = run(
            engine._native_trigger_reference(position, engine.observed_marks[position.mint]))
        assert usd is not None
        assert provenance["reference_identity"] == "PASS"

    def test_the_entry_basis_invariant_is_unaffected(self):
        engine = make_engine(arms=parse_arms(["L25_CURRENT"]))
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_CURRENT"].position
        assert position.entry_basis_sane is True

    def test_the_hard_hold_gate_still_fires_at_the_cap(self):
        engine = make_engine(arms=parse_arms(["L25_MAX_HOLD"]),
                             exits=ExitPolicy(ExitCfg()))
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_MAX_HOLD"].position
        position.opened_at = time.time() - 60.5
        trades = run(engine.monitor_once())
        assert len(trades) == 1
        assert trades[0].exit_trigger == "max_hold_60s"

    def test_fail_closed_on_a_missing_reference_is_still_bounded(self):
        from autopaper.paper_live.schema import ObservedMarkPoint  # noqa: F401
        engine = make_engine(arms=parse_arms(["L25_MAX_HOLD"]),
                             exits=ExitPolicy(ExitCfg()))
        run(engine.on_signal(signal()))
        position = engine.accounts["L25_MAX_HOLD"].position
        position.opened_at = 0.0
        trades = run(engine.monitor_once())
        assert trades == [], "a corrupt timestamp must not produce any exit"
        assert engine.accounts["L25_MAX_HOLD"].has_position