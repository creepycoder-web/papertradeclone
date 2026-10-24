"""
R1: every observation is captured to an append-only stream for replay.

The stream is the input to ``replay``, so its properties matter more than its
convenience: it is append-only, it is bounded on disk, it prunes only its OWN
files, and a failure inside it can never reach the trading loop.
"""

from __future__ import annotations

import json

import pytest

from autopaper.clock import FakeClock
from autopaper.config import MarketConfig, ObservationsConfig
from autopaper.market import MarketDataService
from autopaper.market.ratelimit import RateLimiter
from autopaper.models import Observation
from autopaper.observations import ACTIVE_NAME, ObservationStore

pytestmark = pytest.mark.integrity

T0 = 1_700_000_000.0
MINT = "So11111111111111111111111111111111111111112"


def obs(at: float, price: float = 1.0e-4, mint: str = MINT) -> Observation:
    return Observation(
        mint=mint, observed_at=at, received_at=at, price_usd=price,
        liquidity_usd=40_000.0, provider="test",
    )


def store(tmp_path, **kw) -> ObservationStore:
    cfg = ObservationsConfig(**kw)
    return ObservationStore(tmp_path / "observations", cfg, clock=FakeClock(T0))


def lines(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# the stream itself
# ---------------------------------------------------------------------------


def test_one_json_line_per_observation(tmp_path):
    s = store(tmp_path)
    s.record(MINT, obs(T0))
    s.record(MINT, obs(T0 + 5, price=2.0e-4))
    rows = lines(s.active_path)
    assert len(rows) == 2
    assert rows[0]["mint"] == MINT
    assert rows[0]["observed_at"] == T0
    assert rows[1]["price_usd"] == 2.0e-4


def test_the_stream_is_append_only_across_reopens(tmp_path):
    s = store(tmp_path)
    s.record(MINT, obs(T0))
    s.close()
    again = store(tmp_path)
    again.record(MINT, obs(T0 + 1))
    assert len(lines(again.active_path)) == 2


def test_disabled_writes_nothing(tmp_path):
    s = store(tmp_path, enabled=False)
    s.record(MINT, obs(T0))
    assert not s.active_path.exists()


# ---------------------------------------------------------------------------
# rotation and the disk cap
# ---------------------------------------------------------------------------


def test_rotation_happens_at_the_configured_size(tmp_path):
    s = store(tmp_path, rotate_mb=0.001)          # 1 KB
    for i in range(40):
        s.record(MINT, obs(T0 + i))
    assert s.rotated_paths(), "no file was rotated"
    assert s.active_path.exists()


def test_the_cap_deletes_the_oldest_rotated_file_first(tmp_path):
    s = store(tmp_path, rotate_mb=0.001, max_mb=0.004)
    for i in range(200):
        s.record(MINT, obs(T0 + i))
    rotated = s.rotated_paths()
    assert rotated, "nothing rotated, so the cap was never exercised"
    assert s.total_bytes() <= 0.004 * 1024 * 1024
    # the survivors are the NEWEST rotated files: the oldest went first
    names = [p.name for p in rotated]
    assert names == sorted(names), "rotated files are not ordered by age"
    assert s.pruned_files > 0


def test_the_active_file_is_never_pruned(tmp_path):
    s = store(tmp_path, rotate_mb=0.001, max_mb=0.0005)   # cap below one file
    for i in range(80):
        s.record(MINT, obs(T0 + i))
    assert s.active_path.exists()
    assert s.active_path.name == ACTIVE_NAME


def test_pruning_touches_only_its_own_files(tmp_path):
    """Ledgers and state live in var/ too. The store may never reach them."""
    var = tmp_path
    ledger = var / "ledger.jsonl"
    state = var / "paper_account.json"
    ledger.write_text("x" * 5000)
    state.write_text("y" * 5000)
    s = ObservationStore(
        var / "observations",
        ObservationsConfig(rotate_mb=0.001, max_mb=0.002),
        clock=FakeClock(T0),
    )
    for i in range(200):
        s.record(MINT, obs(T0 + i))
    assert s.pruned_files > 0
    assert ledger.read_text() == "x" * 5000
    assert state.read_text() == "y" * 5000


def test_the_cap_reached_flag_is_reported(tmp_path):
    s = store(tmp_path, rotate_mb=0.001, max_mb=0.002)
    assert s.cap_reached is False
    for i in range(200):
        s.record(MINT, obs(T0 + i))
    assert s.cap_reached is True


# ---------------------------------------------------------------------------
# the engine wiring
# ---------------------------------------------------------------------------


class _Provider:
    name = "test"

    def __init__(self, observation):
        self.limiter = RateLimiter(10_000, name="test")
        self.retry_stats = lambda: {}
        self.max_call_s = 0.0
        self.timeout_s = 0.0
        self._obs = observation

    def quote(self, mint):
        from autopaper.models import QuoteResult

        return QuoteResult(observation=self._obs, provider="test")


def _service(tmp_path, observation, store_obj):
    clock = FakeClock(T0)
    svc = MarketDataService(
        MarketConfig(), clock=clock,
        primary=_Provider(observation), fallback=_Provider(observation),
        observations=store_obj,
    )
    svc._fvp_probe_budget = 0
    return svc


def test_every_recorded_observation_reaches_the_store(tmp_path):
    s = store(tmp_path)
    svc = _service(tmp_path, obs(T0), s)
    svc.quote(MINT)
    assert len(lines(s.active_path)) == 1
    assert len(svc.history(MINT)) == 1


def test_a_store_failure_never_reaches_the_trading_loop(tmp_path):
    """
    Capture is telemetry. It must not be able to refuse an observation the
    agent needs in order to decide.
    """
    s = store(tmp_path)

    def boom(*_a, **_k):
        raise OSError("disk on fire")

    s.record = boom                      # type: ignore[assignment]
    svc = _service(tmp_path, obs(T0), s)
    svc.quote(MINT)
    assert len(svc.history(MINT)) == 1, "the observation was lost"


def test_no_store_configured_is_simply_no_capture(tmp_path):
    svc = _service(tmp_path, obs(T0), None)
    svc.quote(MINT)
    assert len(svc.history(MINT)) == 1
