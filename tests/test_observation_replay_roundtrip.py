"""
D1: the observation stream is replay-ready.

A future replay can only be trusted if the stream it reads reproduces exactly
what the live agent saw. "Exactly" means two things, and both are asserted
here:

* LOSSLESS. Every field of the `Observation` dataclass survives the round trip,
  so a replayed decision reads the same inputs as the live one did.
* IDENTICAL UNDER `as_of`. The no-lookahead gate is the only way history is
  read, so the test compares `as_of(t)` output rather than file contents --
  including at timestamps that straddle a rotation boundary, where a naive
  reader would silently drop one side.

A truncated final line is expected, not exceptional: the process can be killed
mid-write at any moment, and a reader that raises on that would make every
crash-interrupted session unreadable.
"""

from __future__ import annotations

import gzip
import json

import pytest

from autopaper.clock import FakeClock
from autopaper.config import ObservationsConfig
from autopaper.models import Observation
from autopaper.observations import (
    OBS_SCHEMA_VERSION,
    ObservationStore,
    load_observations,
)

pytestmark = pytest.mark.integrity

T0 = 1_700_000_000.0
MINT = "So11111111111111111111111111111111111111112"
CAND = "cand_roundtrip_1"


def obs(at: float, price: float, *, mint: str = MINT) -> Observation:
    """An observation with every numeric field populated, to prove losslessness."""
    return Observation(
        mint=mint, observed_at=at, received_at=at - 0.25, price_usd=price,
        price_native=price / 150.0, market_cap_usd=15_000.0, fdv_usd=16_000.0,
        liquidity_usd=40_000.0, volume_5m_usd=20_000.0, volume_1h_usd=90_000.0,
        volume_24h_usd=250_000.0, price_change_5m=12.5, price_change_1h=-3.25,
        price_change_24h=400.0, buys_5m=80, sells_5m=40, txns_5m=120,
        txns_1h=900, pair_created_at=at - 180.0, first_valid_price=price * 0.9,
        provider="dexscreener", venue="raydium", pair_address="pair-1",
        extra={"gt_note": "kept"},
    )


def store(tmp_path, **kw) -> ObservationStore:
    return ObservationStore(
        tmp_path / "observations", ObservationsConfig(**kw),
        clock=FakeClock(T0),
    )


# ---------------------------------------------------------------------------
# the record carries what replay needs
# ---------------------------------------------------------------------------


def test_every_record_carries_schema_version_candidate_and_received_at(tmp_path):
    s = store(tmp_path)
    s.record(MINT, obs(T0, 1.0e-4), candidate_id=CAND)
    s.close()
    row = json.loads(s.active_path.read_text().splitlines()[0])
    assert row["schema_version"] == OBS_SCHEMA_VERSION
    assert row["candidate_id"] == CAND
    assert row["received_at"] == pytest.approx(T0 - 0.25)
    assert row["provider"] == "dexscreener"


def test_a_record_without_a_candidate_is_recorded_as_unattributed(tmp_path):
    """
    Position monitoring has no candidate. Writing null rather than omitting the
    key keeps every row the same shape, and keeps "no candidate" distinct from
    "the writer forgot".
    """
    s = store(tmp_path)
    s.record(MINT, obs(T0, 1.0e-4))
    s.close()
    row = json.loads(s.active_path.read_text().splitlines()[0])
    assert row["candidate_id"] is None


# ---------------------------------------------------------------------------
# the round trip
# ---------------------------------------------------------------------------


def _record_session(s, prices, *, candidate=CAND, step=5.0):
    for i, price in enumerate(prices):
        s.record(MINT, obs(T0 + i * step, price), candidate_id=candidate)
    s.close()


def test_as_of_is_identical_after_a_round_trip(tmp_path):
    s = store(tmp_path)
    prices = [1.0e-4 + i * 1.0e-6 for i in range(40)]
    live = [obs(T0 + i * 5.0, p) for i, p in enumerate(prices)]
    _record_session(s, prices)

    loaded = load_observations(s.all_paths())
    assert set(loaded) == {CAND}
    replayed = loaded[CAND]

    from autopaper.models import ObservationSet
    live_set = ObservationSet(live)
    for i in range(-1, 42):
        t = T0 + i * 5.0 + 0.5
        a = [(o.observed_at, o.price_usd) for o in live_set.as_of(t)]
        b = [(o.observed_at, o.price_usd) for o in replayed.as_of(t)]
        assert a == b, f"as_of({t}) diverged"


def test_the_round_trip_is_lossless_field_by_field(tmp_path):
    s = store(tmp_path)
    original = obs(T0, 1.0e-4)
    s.record(MINT, original, candidate_id=CAND)
    s.close()
    back = list(load_observations(s.all_paths())[CAND].as_of(T0 + 1))[0]
    for field, value in original.as_dict().items():
        assert getattr(back, field, None) == value, field


def test_as_of_is_identical_across_a_rotation_boundary(tmp_path):
    """
    The case a naive reader gets wrong: half the session is in a gzipped
    rotated file and half in the live one.
    """
    s = store(tmp_path, rotate_mb=0.002)
    prices = [1.0e-4 + i * 1.0e-6 for i in range(120)]
    live = [obs(T0 + i * 5.0, p) for i, p in enumerate(prices)]
    _record_session(s, prices)
    assert s.rotated_paths(), "the session never rotated, so nothing was proven"

    from autopaper.models import ObservationSet
    live_set = ObservationSet(live)
    replayed = load_observations(s.all_paths())[CAND]
    assert len(replayed) == len(live)
    for i in range(0, 122, 7):
        t = T0 + i * 5.0 + 0.5
        a = [o.observed_at for o in live_set.as_of(t)]
        b = [o.observed_at for o in replayed.as_of(t)]
        assert a == b, f"as_of({t}) diverged across the rotation boundary"


def test_records_are_split_by_candidate(tmp_path):
    s = store(tmp_path)
    s.record(MINT, obs(T0, 1.0e-4), candidate_id="cand_a")
    s.record(MINT, obs(T0 + 1, 2.0e-4), candidate_id="cand_b")
    s.close()
    loaded = load_observations(s.all_paths())
    assert sorted(loaded) == ["cand_a", "cand_b"]
    assert len(loaded["cand_a"]) == 1


# ---------------------------------------------------------------------------
# gzip, flushing, truncation
# ---------------------------------------------------------------------------


def test_rotated_files_are_gzipped(tmp_path):
    s = store(tmp_path, rotate_mb=0.002)
    _record_session(s, [1.0e-4] * 120)
    rotated = s.rotated_paths()
    assert rotated, "nothing rotated"
    assert all(p.name.endswith(".jsonl.gz") for p in rotated), rotated
    with gzip.open(rotated[0], "rt", encoding="utf-8") as fh:
        assert json.loads(fh.readline())["schema_version"] == OBS_SCHEMA_VERSION


def test_the_cap_counts_compressed_size(tmp_path):
    """
    Gzip is the reason the cap is honest: counting uncompressed size would
    under-use the disk by ~10x and prune data that fitted.
    """
    s = store(tmp_path, rotate_mb=0.002, max_mb=0.004)
    _record_session(s, [1.0e-4] * 400)
    assert s.total_bytes() <= 0.004 * 1024 * 1024
    assert s.pruned_files > 0


def test_writes_are_buffered_and_flushed_on_the_interval(tmp_path):
    clock = FakeClock(T0)
    s = ObservationStore(
        tmp_path / "observations", ObservationsConfig(flush_s=5.0), clock=clock,
    )
    s.record(MINT, obs(T0, 1.0e-4), candidate_id=CAND)
    assert s.buffered == 1
    clock.advance(1.0)
    s.record(MINT, obs(T0 + 1, 1.0e-4), candidate_id=CAND)
    assert s.buffered == 2, "flushed before the interval elapsed"
    clock.advance(10.0)
    s.record(MINT, obs(T0 + 11, 1.0e-4), candidate_id=CAND)
    assert s.buffered == 0, "the interval elapsed and nothing was flushed"
    assert len(s.active_path.read_text().splitlines()) == 3


def test_close_flushes_what_is_buffered(tmp_path):
    s = store(tmp_path, flush_s=3600.0)
    s.record(MINT, obs(T0, 1.0e-4), candidate_id=CAND)
    assert not s.active_path.exists() or s.active_path.read_text() == ""
    s.close()
    assert len(s.active_path.read_text().splitlines()) == 1


def test_a_truncated_last_line_is_skipped_not_raised(tmp_path):
    """
    The process can be killed mid-write. A reader that raises would make every
    crash-interrupted session unreadable, which is the opposite of durable.
    """
    s = store(tmp_path)
    _record_session(s, [1.0e-4, 2.0e-4, 3.0e-4])
    with s.active_path.open("a", encoding="utf-8") as fh:
        fh.write('{"mint": "So111", "observed_at": 170000')   # no newline, no close
    loaded = load_observations(s.all_paths())
    assert len(loaded[CAND]) == 3, "the three intact rows were not all recovered"


def test_a_truncated_line_in_the_middle_is_also_skipped(tmp_path):
    s = store(tmp_path)
    _record_session(s, [1.0e-4, 2.0e-4])
    text = s.active_path.read_text().splitlines()
    s.active_path.write_text(text[0] + "\n" + '{"broken": ' + "\n" + text[1] + "\n")
    loaded = load_observations(s.all_paths())
    assert len(loaded[CAND]) == 2
    assert load_observations(s.all_paths(), report=True)[1]["skipped_lines"] == 1


def test_load_tolerates_a_missing_path(tmp_path):
    s = store(tmp_path)
    _record_session(s, [1.0e-4])
    loaded = load_observations([*s.all_paths(), tmp_path / "gone.jsonl"])
    assert len(loaded[CAND]) == 1


def test_capture_covers_every_declared_observation_field(tmp_path):
    """
    Guards the bug this module was written around: `Observation.as_dict()` is a
    reporting view that omits `volume_24h_usd`, so capturing through it dropped
    that field from every row. Asserting against the dataclass fields means a
    field added later cannot silently stop being captured.
    """
    import dataclasses

    s = store(tmp_path)
    s.record(MINT, obs(T0, 1.0e-4), candidate_id=CAND)
    s.close()
    row = json.loads(s.active_path.read_text().splitlines()[0])
    declared = {f.name for f in dataclasses.fields(Observation)}
    assert declared <= set(row), f"not captured: {sorted(declared - set(row))}"
