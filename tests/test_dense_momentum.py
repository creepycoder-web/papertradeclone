"""Direct tests for the dense-momentum research builder.

The module under test lives in the production repo at
``~/works/memecoin-bot/research/dense_momentum.py``. It is research-only: the
trader never imports it, so these tests exercise it in isolation.

Two invariants are pinned here, both of which were real defects found by the
snapshot audit:

1. ``DENSE_CUTOFF`` pins the snapshot. The tape is append-only and the
   accumulator keeps running, so without a cutoff the analysis, the metrics and
   the report can each describe a different amount of data.

2. Observations timestamped before T0 cannot reach a decision-time feature. A
   T0-anchored return computed from a pre-decision price is lookahead, which is
   exactly what ``ret_from_t0_pct`` must never be populated from.

Everything here is offline and deterministic: no network, no sleeps, no sqlite
(candidates carry no mint, so the production journal is never opened), and
fixtures are a handful of jsonl lines in ``tmp_path``.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.integrity

# The offsets that produced the real leak: alert-phase prints that landed
# before the agent's own decision, for `trade`, `TWINX` and `two`.
AUDIT_PRE_T0_OFFSETS = (-1.90, -0.86, -1.85, -1.62)

CID = "cand_test0001"
SYMBOL = "TESTCOIN"
ALERT_TS = 97.0
T0 = 100.0

# Poisoned price used to make any influence on a feature unmissable.
POISON = 1.0e9

# Features that must be identical with and without pre-T0 observations.
# Bookkeeping columns (n_dense_obs, n_pre_t0, tape_span_s) are deliberately
# excluded: they are *meant* to count the whole tape.
BOOKKEEPING = {"n_dense_obs", "n_pre_t0", "tape_span_s", "symbol", "candidate_id"}


def _load_module():
    path = Path.home() / "works" / "memecoin-bot" / "research" / "dense_momentum.py"
    if not path.exists():
        # Module-level skip: the research module lives outside this repo, so
        # without allow_module_level the whole file errors at COLLECTION instead
        # of skipping cleanly.
        pytest.skip(
            f"module under test not present at {path}",
            allow_module_level=True,
        )
    spec = importlib.util.spec_from_file_location("dense_momentum_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dm = _load_module()


def _write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _obs(ts: float, price: float):
    return {
        "candidate_id": CID,
        "symbol": SYMBOL,
        "ts": ts,
        "price": price,
        "dt_from_alert_s": round(ts - ALERT_TS, 3),
    }


def _setup(tmp_path: Path, monkeypatch, observations, agent_price=10.0):
    """Point the module at a throwaway tree and return it."""
    ai = tmp_path / "ai"
    _write_jsonl(ai / "var" / "candidates.jsonl", [{
        "event": "candidate_seen", "candidate_id": CID, "symbol": SYMBOL,
        "received_timestamp": ALERT_TS,
        # no mint -> `mints` is empty -> the production journal is never opened
        "mint": None,
    }])
    features = {"token_age_s": 42.0, "market_cap_usd": 1_000_000.0,
                "liquidity_usd": 50_000.0, "change_5m": 55.0}
    if agent_price is not None:
        features["price"] = agent_price
    _write_jsonl(ai / "var" / "decisions.jsonl", [{
        "event": "decision", "candidate_id": CID, "decided_at": T0,
        "decision": "BUY", "route": "FAST_ENTRY", "regime": "EXPLOSIVE",
        "features": features,
    }])
    _write_jsonl(ai / "var" / "ledger.jsonl", [])
    _write_jsonl(ai / "research" / "tape_live" / "observations.jsonl", observations)

    monkeypatch.setattr(dm, "AI", str(ai))
    monkeypatch.setattr(dm, "SIDECAR", str(ai / "research" / "tape_live" / "observations.jsonl"))
    monkeypatch.setattr(dm, "SNAP", str(tmp_path / "snapshot"))
    monkeypatch.delenv("DENSE_CUTOFF", raising=False)
    return ai


# A clean post-T0 tape: 1s apart, spanning past the +300s horizon.
CLEAN_TAPE = [_obs(T0 + i, 10.0 + 0.1 * i) for i in range(0, 31)]

# Same tape with the audit's pre-T0 observations, at a poison price.
def _tape_with_pre_t0():
    return [_obs(T0 + off, POISON) for off in AUDIT_PRE_T0_OFFSETS] + list(CLEAN_TAPE)


def _one_row(rows):
    assert len(rows) == 1, f"expected exactly one candidate, got {len(rows)}"
    return rows[0]


def _decision_features(row):
    """Everything derived from the price path at or after T0."""
    return {k: v for k, v in row.items() if k not in BOOKKEEPING}


# --------------------------------------------------------------------------
# 1. DENSE_CUTOFF snapshot pinning
# --------------------------------------------------------------------------

def test_cutoff_pins_the_snapshot(tmp_path, monkeypatch):
    """Observations after the cutoff are dropped, and the cutoff is recorded."""
    tape = list(CLEAN_TAPE) + [_obs(T0 + 40, 99.0), _obs(T0 + 50, 123.0)]
    _setup(tmp_path, monkeypatch, tape)

    assert dm.build()[2] is None, "no cutoff set -> CUTOFF must be None"

    monkeypatch.setenv("DENSE_CUTOFF", str(T0 + 30))
    rows, n_raw, cutoff = dm.build()

    assert cutoff == T0 + 30
    assert n_raw == len(CLEAN_TAPE), "post-cutoff observations must not be counted"
    assert _one_row(rows)["n_dense_obs"] == len(CLEAN_TAPE)

    snapshot = json.loads((tmp_path / "snapshot" / "dense_rows.json").read_text())
    assert snapshot["cutoff"] == T0 + 30
    assert snapshot["observations_in_snapshot"] == len(CLEAN_TAPE)


def test_observation_after_cutoff_cannot_change_the_result(tmp_path, monkeypatch):
    """REGRESSION: a later observation must not move any computed value.

    The post-cutoff print is a 1e9 spike inside the +300s outcome window, so
    without the pin it would dominate ``mfe_300s``.
    """
    spike = _obs(T0 + 40, POISON)

    # Control: with no cutoff the spike *does* change the outcome.
    _setup(tmp_path, monkeypatch, list(CLEAN_TAPE) + [spike])
    uncontrolled = _one_row(dm.build()[0])
    assert uncontrolled["n_dense_obs"] == len(CLEAN_TAPE) + 1
    assert uncontrolled["mfe_300s"] > 1.0e6, "control must actually move"

    # Pinned: the same tape read at a cutoff before the spike.
    _setup(tmp_path, monkeypatch, list(CLEAN_TAPE) + [spike])
    monkeypatch.setenv("DENSE_CUTOFF", str(T0 + 30))
    with_spike_present = _one_row(dm.build()[0])

    # Same pinned cutoff, spike never written at all.
    _setup(tmp_path, monkeypatch, list(CLEAN_TAPE))
    monkeypatch.setenv("DENSE_CUTOFF", str(T0 + 30))
    without_spike = _one_row(dm.build()[0])

    assert with_spike_present == without_spike, \
        "an observation after DENSE_CUTOFF changed the result"
    assert with_spike_present["mfe_300s"] is not None
    assert with_spike_present["mfe_300s"] < 1.0e6


# --------------------------------------------------------------------------
# 2. Pre-T0 return guard
# --------------------------------------------------------------------------

def test_pre_t0_observations_cannot_affect_any_post_t0_feature(tmp_path, monkeypatch):
    """A pre-decision print must not move a single decision-time feature.

    Uses the exact offsets from the audit, at a poison price so that any
    influence is impossible to miss.
    """
    _setup(tmp_path, monkeypatch, _tape_with_pre_t0())
    with_pre = _one_row(dm.build()[0])

    _setup(tmp_path, monkeypatch, list(CLEAN_TAPE))
    without_pre = _one_row(dm.build()[0])

    assert with_pre["n_pre_t0"] == len(AUDIT_PRE_T0_OFFSETS)
    assert with_pre["n_post_t0"] == without_pre["n_post_t0"]

    differing = {k for k in _decision_features(without_pre)
                 if _decision_features(with_pre).get(k) != without_pre.get(k)}
    assert not differing, f"pre-T0 observations changed: {sorted(differing)}"

    # ... and they are genuinely present, so this is not a vacuous pass.
    assert with_pre["n_dense_obs"] > without_pre["n_dense_obs"]
    assert with_pre["integrity_ok"] is True


def test_pre_t0_price_is_never_used_as_the_return_base_when_agent_price_exists(tmp_path, monkeypatch):
    """The T0 anchor is the agent's own decision-time price, not a pre-T0 print."""
    _setup(tmp_path, monkeypatch, _tape_with_pre_t0(), agent_price=10.0)
    row = _one_row(dm.build()[0])

    assert row["base_source"] == "agent_decision_time_price"
    assert row["base_price"] == 10.0
    assert row["base_price"] != POISON
    assert row["anchor"] == "decision_ts"
    assert row["t0"] == T0
    # cum_ret_at_t0 is the T0-anchored return: exactly zero, never a poison spike.
    assert row["cum_ret_at_t0"] == 0.0

    # `ret_1s` must be anchored at T0 itself: the print at T0+0 (10.0) to the
    # print at T0+1 (10.1) is +1.0%. Anchoring at the first print *after* the
    # horizon instead would silently report 0.0%.
    assert row["ret_1s"] == pytest.approx(1.0)
    assert row["ret_1s"] != pytest.approx(0.0)


def test_pre_t0_print_is_used_only_as_the_documented_base_fallback(tmp_path, monkeypatch):
    """The one place a pre-T0 price is allowed: an explicit, labelled fallback.

    With no decision-time price available, the last print at or before T0
    becomes the base, and ``base_source`` must say so.
    """
    _setup(tmp_path, monkeypatch, _tape_with_pre_t0(), agent_price=None)
    row = _one_row(dm.build()[0])

    assert row["base_source"] == "last_at_or_before_t0"
    assert row["base_price"] == POISON, "fallback should take the last pre-T0 print"
    # Still anchored at T0: the anchored return at T0 is zero by construction.
    assert row["cum_ret_at_t0"] == 0.0


def test_ret_from_t0_pct_is_not_populated_from_a_pre_t0_observation(tmp_path, monkeypatch):
    """A T0-anchored return must not exist for any observation before T0.

    ``ret_from_t0_pct`` is produced by the observations export, not by the
    builder, so the invariant is asserted where the column actually lives: the
    shipped artifact. Skipped if the research run has not produced it.
    """
    artifact = Path.home() / "works" / "memecoin-bot" / "reports" / "dense_momentum_observations.csv"
    if not artifact.exists():
        pytest.skip(f"observations artifact not present at {artifact}")

    offenders = []
    for row in csv.DictReader(artifact.open()):
        if float(row["dt_from_t0_s"]) < 0 and row.get("ret_from_t0_pct") not in ("", None):
            offenders.append((row["candidate_id"], row["dt_from_t0_s"]))
    assert not offenders, f"ret_from_t0_pct populated from pre-T0 rows: {offenders}"


def test_inconsistent_dt_from_alert_is_flagged(tmp_path, monkeypatch):
    """The `dt_from_alert` reconstruction is enforced, not merely computed.

    The module requires ``ts <= dt_from_alert_s + received_timestamp + 0.01``.
    An observation whose alert-relative offset cannot account for its own
    timestamp must be reported, not silently accepted.
    """
    # inconsistent: dt_from_alert_s + received_timestamp = 98.0 << ts = 110.0
    bad = _obs(T0 + 10, 11.0)
    bad["dt_from_alert_s"] = 1.0
    _setup(tmp_path, monkeypatch, [bad])
    row = _one_row(dm.build()[0])
    assert "dt_from_alert_inconsistent" in row["integrity_violations"]
    assert row["integrity_ok"] is False

    # a well-formed tape at the same instant is not flagged
    _setup(tmp_path, monkeypatch, [_obs(T0 + 10, 11.0)])
    assert _one_row(dm.build()[0])["integrity_ok"] is True


def test_dt_from_alert_tolerance_admits_the_collectors_rounding(tmp_path, monkeypatch):
    """The 0.01s tolerance exists because the collector rounds to 3dp.

    A value that is correct only to within that rounding must pass; the
    tolerance must not become a loophole for a materially wrong offset.
    """
    good = _obs(T0 + 10, 11.0)          # dt_from_alert_s rounded to 3dp
    _setup(tmp_path, monkeypatch, [good])
    assert _one_row(dm.build()[0])["integrity_ok"] is True

    # 0.05s of drift is beyond the tolerance and must be caught
    drift = _obs(T0 + 10, 11.0)
    drift["dt_from_alert_s"] = round(T0 + 10 - ALERT_TS, 3) - 0.05
    _setup(tmp_path, monkeypatch, [drift])
    assert "dt_from_alert_inconsistent" in _one_row(dm.build()[0])["integrity_violations"]


def test_post_t0_classification_is_audited_before_filtering(tmp_path, monkeypatch):
    """A misclassified pre-T0 print is DETECTED, not absorbed by the filter.

    The old code built ``post`` from ``ts >= t0`` and then asserted its members
    had ``ts >= t0``, so ``post_before_t0`` could never fire. Here an
    observation is classified post-T0 by the alert-relative clock while its
    stored ``ts`` is still before T0 - a real inconsistency that must surface.
    """
    offender = {"candidate_id": CID, "symbol": SYMBOL, "ts": T0 - 1.0,
                "price": POISON, "dt_from_alert_s": 3.0}   # reconstructs to T0
    _setup(tmp_path, monkeypatch, [offender] + list(CLEAN_TAPE))
    row = _one_row(dm.build()[0])

    assert "post_before_t0" in row["integrity_violations"], \
        "a pre-T0 observation classified as post-T0 was not detected"
    assert row["integrity_ok"] is False

    # the converse: a future print misfiled as pre-T0 must also surface
    future = {"candidate_id": CID, "symbol": SYMBOL, "ts": T0 + 2.0,
              "price": POISON, "dt_from_alert_s": -3.0}   # reconstructs to T0-3
    _setup(tmp_path, monkeypatch, [future] + list(CLEAN_TAPE))
    assert "future_in_pre" in _one_row(dm.build()[0])["integrity_violations"]

    # a fully consistent tape is still clean
    _setup(tmp_path, monkeypatch, _tape_with_pre_t0())
    assert _one_row(dm.build()[0])["integrity_ok"] is True
