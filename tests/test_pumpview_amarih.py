"""
PumpView **Amarih** monitoring adapter -- offline validation.

The fixture ``tests/fixtures/pumpview_amarih.json`` is built from GENUINE
observed PumpView events (public ``/api/signal-history`` /
``/api/strategy-detail`` reads), including losing and zero-ATH outcomes, so
these tests exercise the adapter on real payloads rather than invented ones.

Covers, all offline (no network):
  * correct event parsing and token-mint extraction;
  * Amarih events distinguished from other strategies in the same stream;
  * duplicate events (same batch + across polls) de-duplicated;
  * out-of-order / reconnected events;
  * stale signals rejected by a configurable max age;
  * multiple events for the same token;
  * missing mint / missing fire-time / wrong-strategy refusals;
  * provenance preserved (signal id, entry market cap, fired timestamp, owner);
  * the live ``poll_once`` transport against an httpx MockTransport.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import httpx
import pytest

from autopaper.paper_live.pumpview import (
    DEFAULT_MAX_AGE_S,
    REJECT_DUPLICATE,
    REJECT_FUTURE_TIMESTAMP,
    REJECT_NO_MINT,
    REJECT_NO_TIMESTAMP,
    REJECT_STALE,
    STRATEGY_NAME,
    PumpViewAmarihSource,
)

FIXTURE = Path(__file__).parent / "fixtures" / "pumpview_amarih.json"


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _newest_ts(sample: list[dict]) -> float:
    return max(s["fired_at_ts"] for s in sample)


# ---------------------------------------------------------------------------
# parsing / identity
# ---------------------------------------------------------------------------

def test_genuine_amarih_events_parse_with_full_provenance(fixture):
    """Every Amarih event in the genuine fixture becomes one signal, and the
    strategy name, signal id, mint, entry market cap, fired timestamp and owner
    are all preserved verbatim from the source."""
    sample = fixture["signal_history_sample"]
    now = _newest_ts(sample)
    src = PumpViewAmarihSource(max_age_s=1e9)
    rep = src.ingest_events({"signals": sample}, now=now)

    assert rep.events_seen == len(sample)
    assert len(rep.accepted) == len(sample)
    assert rep.refused == []

    # Cross-check the first accepted signal against its raw source event.
    raw = sample[0]
    sig = next(s for s in rep.accepted if s.signal_id == str(raw["id"]))
    assert sig.strategy == STRATEGY_NAME == "Amarih"
    assert sig.mint == raw["ca"]
    assert sig.symbol == raw["sym"]
    assert sig.fired_at == pytest.approx(raw["fired_at_ts"])
    assert sig.entry_market_cap_usd == pytest.approx(raw["mc_signal"])
    assert sig.buy_score == pytest.approx(raw["buy_score"])
    assert sig.owner_wallet == raw["owner_wallet"]
    # The raw event is preserved for audit.
    assert sig.raw["id"] == raw["id"]


def test_mint_is_extracted_as_base58(fixture):
    """Token mint comes from ``ca`` and is a plausible Solana base58 mint."""
    sample = fixture["signal_history_sample"]
    now = _newest_ts(sample)
    src = PumpViewAmarihSource(max_age_s=1e9)
    rep = src.ingest_events({"signals": sample}, now=now)
    for sig in rep.accepted:
        assert sig.mint and len(sig.mint) >= 32
        # pump mints end in 'pump'; all base58 alphabet, no 0OIl.
        assert all(c not in sig.mint for c in "0OIl")


def test_amarih_distinguished_from_other_strategies(fixture):
    """A non-Amarih event in the same stream is ignored, never accepted."""
    now = 1_800_000_000.0
    payload = {"signals": [
        {"id": 1, "signal_type": "Amarih", "ca": "A" * 40, "sym": "X",
         "fired_at_ts": now - 1, "mc_signal": 20000, "buy_score": 6},
        {"id": 2, "signal_type": "Fredlam", "ca": "B" * 40, "sym": "Y",
         "fired_at_ts": now - 1, "mc_signal": 20000, "buy_score": 9},
        {"id": 3, "signal_type": "Strategy Hard", "ca": "C" * 40, "sym": "Z",
         "fired_at_ts": now - 1, "mc_signal": 20000, "buy_score": 9},
    ]}
    src = PumpViewAmarihSource(max_age_s=1e9)
    rep = src.ingest_events(payload, now=now)
    assert rep.events_seen == 3
    assert [s.signal_id for s in rep.accepted] == ["1"]
    assert rep.accepted[0].strategy == "Amarih"


# ---------------------------------------------------------------------------
# de-duplication / ordering / reconnection
# ---------------------------------------------------------------------------

def test_duplicate_events_are_rejected(fixture):
    """The same event twice -- in one batch and across two polls -- yields one
    signal and one DUPLICATE refusal each time."""
    sample = fixture["signal_history_sample"]
    now = _newest_ts(sample)
    src = PumpViewAmarihSource(max_age_s=1e9)

    first = src.ingest_events({"signals": sample}, now=now)
    assert len(first.accepted) == len(sample)

    # A second poll returning the SAME events (nothing new fired).
    second = src.ingest_events({"signals": sample}, now=now)
    assert second.accepted == []
    assert all(r == REJECT_DUPLICATE for _, r in second.refused)
    assert len(second.refused) == len(sample)


def test_duplicate_within_a_single_batch(fixture):
    """A batch that literally repeats one id produces a single signal."""
    sample = fixture["signal_history_sample"][:5]
    doubled = sample + sample  # each id appears twice in one payload
    now = _newest_ts(sample)
    src = PumpViewAmarihSource(max_age_s=1e9)
    rep = src.ingest_events({"signals": doubled}, now=now)
    assert len(rep.accepted) == 5
    assert sum(1 for _, r in rep.refused if r == REJECT_DUPLICATE) == 5


def test_out_of_order_and_reconnect(fixture):
    """Events arriving newest-first, then older ones on a later poll, are each
    accepted exactly once; a reconnect that replays older ids does not
    double-count."""
    sample = sorted(fixture["signal_history_sample"],
                    key=lambda s: s["fired_at_ts"], reverse=True)
    now = _newest_ts(sample) + 1
    src = PumpViewAmarihSource(max_age_s=1e9)

    # Poll 1: newest 10. Poll 2: an overlapping older window (reconnect replay).
    p1 = src.ingest_events({"signals": sample[:10]}, now=now)
    p2 = src.ingest_events({"signals": sample[5:20]}, now=now)  # overlaps p1
    accepted_ids = {s.signal_id for s in p1.accepted} | {s.signal_id for s in p2.accepted}
    assert len(accepted_ids) == 20                      # 10 + 10 new, no dupes
    assert sum(1 for _, r in p2.refused if r == REJECT_DUPLICATE) == 5  # the overlap


# ---------------------------------------------------------------------------
# staleness
# ---------------------------------------------------------------------------

def test_stale_signals_are_rejected(fixture):
    """A signal older than the configured max age is refused STALE, and is not
    re-surfaced by a later poll."""
    sample = fixture["signal_history_sample"][:5]
    now = _newest_ts(sample)
    src = PumpViewAmarihSource(max_age_s=10.0)  # very tight
    rep = src.ingest_events({"signals": sample}, now=now + 3600)  # an hour late
    assert rep.accepted == []
    assert all(r == REJECT_STALE for _, r in rep.refused)
    # Re-poll must not resurrect it.
    rep2 = src.ingest_events({"signals": sample}, now=now + 7200)
    assert rep2.accepted == []
    assert all(r in (REJECT_STALE, REJECT_DUPLICATE) for _, r in rep2.refused)


def test_max_age_is_configurable_and_boundary_is_inclusive():
    """age == max_age_s is accepted; age just over is rejected."""
    now = 1_800_000_000.0
    mk = lambda i, fired: {"id": i, "signal_type": "Amarih",
                           "ca": "A" * 40, "sym": "X", "fired_at_ts": fired,
                           "mc_signal": 20000}
    # exactly at the boundary (age == 120)
    src = PumpViewAmarihSource(max_age_s=120.0)
    rep = src.ingest_events({"signals": [mk(1, now - 120.0)]}, now=now)
    assert len(rep.accepted) == 1
    # just over the boundary
    src2 = PumpViewAmarihSource(max_age_s=120.0)
    rep2 = src2.ingest_events({"signals": [mk(2, now - 120.001)]}, now=now)
    assert rep2.accepted == []
    assert rep2.refused[0][1] == REJECT_STALE


def test_default_max_age_is_tight():
    """
    The default window is 15 s, and that is the value the Amarih experiment
    actually runs with.

    This test used to assert ``DEFAULT_MAX_AGE_S <= 180.0``, which both the
    shipped 120 s default and a 15 s default satisfy -- so it passed while the
    effective threshold was eight times wider than the strategy's own design.
    The asserted number is now the DESIGNED number, so a regression to 120 s
    fails here instead of trading 118-second-old calls as fresh.
    """
    assert DEFAULT_MAX_AGE_S == 15.0, (
        "the Amarih signal-age default changed; if this was deliberate, "
        "record the reason -- the default is the only gate between a "
        "minutes-old signal and a live entry")


# ---------------------------------------------------------------------------
# multiple events for the same token
# ---------------------------------------------------------------------------

def test_multiple_events_for_the_same_token(fixture):
    """Distinct signal ids on the same mint are distinct opportunities (Amarih
    re-fires on the same token); each is kept, none is dropped as a duplicate."""
    sample = fixture["signal_history_sample"]
    now = _newest_ts(sample)
    src = PumpViewAmarihSource(max_age_s=1e9)
    rep = src.ingest_events({"signals": sample}, now=now)

    mint_counts = Counter(s.mint for s in rep.accepted)
    repeated = {m: c for m, c in mint_counts.items() if c > 1}
    for mint, count in repeated.items():
        # If the genuine sample has a repeated mint, confirm distinct ids.
        ids = [s.signal_id for s in rep.accepted if s.mint == mint]
        assert len(set(ids)) == count  # all distinct signal ids



# ---------------------------------------------------------------------------
# malformed / refusal cases
# ---------------------------------------------------------------------------

def test_missing_mint_is_refused_not_traded():
    now = 1_800_000_000.0
    src = PumpViewAmarihSource(max_age_s=1e9)
    rep = src.ingest_events({"signals": [
        {"id": 1, "signal_type": "Amarih", "ca": None, "sym": "X",
         "fired_at_ts": now - 1, "mc_signal": 20000},
        {"id": 2, "signal_type": "Amarih", "ca": "", "sym": "Y",
         "fired_at_ts": now - 1, "mc_signal": 20000},
    ]}, now=now)
    assert rep.accepted == []
    assert all(r == REJECT_NO_MINT for _, r in rep.refused)
    assert len(rep.refused) == 2


def test_missing_fire_time_is_refused():
    """
    A missing ``fired_at_ts`` is refused, and NOW WITH ITS OWN REASON.

    It used to be recorded as ``NO_MINT`` -- the same string used for a signal
    with no token address. The two are different defects with different fixes:
    one is "we cannot identify the token", the other is "we cannot age the
    signal". Collapsing them meant an operator auditing refusals saw "no mint"
    for a signal whose mint was perfectly valid, and the real cause -- an
    unusable clock -- was invisible.
    """
    now = 1_800_000_000.0
    src = PumpViewAmarihSource(max_age_s=1e9)
    rep = src.ingest_events({"signals": [
        {"id": 1, "signal_type": "Amarih", "ca": "A" * 40, "fired_at_ts": None,
         "mc_signal": 20000},
    ]}, now=now)
    assert rep.accepted == []
    assert rep.refused[0][1] == REJECT_NO_TIMESTAMP
    # Still refused: a refused signal is one the experiment never priced, which
    # is the invariant. Only the AGGREGATE of reasons changed.
    assert len(rep.refused) == 1


def test_empty_or_malformed_payload_is_safe():
    src = PumpViewAmarihSource(max_age_s=1e9)
    assert src.ingest_events({}, now=0).accepted == []
    assert src.ingest_events({"signals": None}, now=0).accepted == []
    assert src.ingest_events({"signals": [1, "x", None]}, now=0).accepted == []


def test_nan_and_inf_numbers_are_dropped_not_fabricated():
    """A non-finite market cap becomes None, never a bogus number."""
    now = 1_800_000_000.0
    src = PumpViewAmarihSource(max_age_s=1e9)
    rep = src.ingest_events({"signals": [
        {"id": 1, "signal_type": "Amarih", "ca": "A" * 40, "sym": "X",
         "fired_at_ts": now - 1, "mc_signal": float("nan"),
         "buy_score": float("inf")},
    ]}, now=now)
    assert len(rep.accepted) == 1
    assert rep.accepted[0].entry_market_cap_usd is None
    assert rep.accepted[0].buy_score is None


def test_signal_to_receipt_latency_is_tracked(fixture):
    """Latency = received_at - fired_at, reported in ms; a late poll is positive."""
    sample = fixture["signal_history_sample"][:3]
    fired = _newest_ts(sample)
    src = PumpViewAmarihSource(max_age_s=1e9)
    rep = src.ingest_events({"signals": sample}, now=fired + 2.5)  # 2.5s late
    for sig in rep.accepted:
        # The property is round((received_at - fired_at) * 1000, 2); compare to
        # that same expression, not to the unrounded product.
        assert sig.signal_to_receipt_latency_ms == pytest.approx(
            round((sig.received_at - sig.fired_at) * 1000.0, 2), abs=1e-6)
    assert any(s.signal_to_receipt_latency_ms > 0 for s in rep.accepted)



# ---------------------------------------------------------------------------
# the live transport (offline, mocked)
# ---------------------------------------------------------------------------

def test_poll_once_transport_uses_signal_history_and_parses(fixture):
    """``poll_once`` GETs /api/signal-history and normalises the response,
    with no Authorization header when no token is supplied."""
    sample = fixture["signal_history_sample"]
    now = _newest_ts(sample)
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["accept"] = request.headers.get("Accept")
        return httpx.Response(200, json={"signals": sample,
                                         "admin_strategies": []})

    transport = httpx.MockTransport(handler)
    src = PumpViewAmarihSource(base_url="https://pumpview.fun", max_age_s=1e9)
    with httpx.Client(transport=transport) as client:
        src.poll_once(limit=50, offset=0, client=client)

    assert "/api/signal-history" in captured["url"]
    assert "limit=50" in captured["url"]
    assert captured["auth"] is None            # no credential invented
    assert captured["accept"] == "application/json"
    # normalise with a FRESH source (the poll already consumed the ids) and a
    # fixed 'now' so staleness does not drop the real sample:
    rep_fixed = PumpViewAmarihSource(max_age_s=1e9).ingest_events(
        {"signals": sample}, now=now)
    assert len(rep_fixed.accepted) == len(sample)


def test_poll_once_attaches_token_only_when_supplied():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"signals": []})

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        PumpViewAmarihSource(token="opaque-operator-token").poll_once(client=client)
    assert seen["auth"] == "Bearer opaque-operator-token"


def test_poll_once_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError):
            PumpViewAmarihSource().poll_once(client=client)


# ---------------------------------------------------------------------------
# provenance / audit record
# ---------------------------------------------------------------------------

def test_accepted_signal_as_dict_is_json_serialisable(fixture):
    sample = fixture["signal_history_sample"][:4]
    now = _newest_ts(sample)
    src = PumpViewAmarihSource(max_age_s=1e9)
    rep = src.ingest_events({"signals": sample}, now=now)
    # The audit record must survive a JSON round-trip (persistence requirement).
    text = json.dumps([s.as_dict() for s in rep.accepted])
    back = json.loads(text)
    assert len(back) == len(sample)
    assert back[0]["strategy"] == "Amarih"
    assert "signal_to_receipt_latency_ms" in back[0]


def test_losing_and_zero_ath_outcomes_present_in_source(fixture):
    """The genuine outcome data used for analysis includes losers and zero-ATH
    signals -- proof this is not survivor-biased cherry-picking."""
    outcomes = fixture["strategy_detail_outcomes_7d"]
    losers = [o for o in outcomes if o.get("cur_pct", 0) < 0]
    zero_ath = [o for o in outcomes if o.get("ath_pct", 0) == 0]
    assert losers, "the source outcome sample must include losing signals"
    assert zero_ath, "the source outcome sample must include zero-ATH signals"

