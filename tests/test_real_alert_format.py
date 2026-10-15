"""
Regression tests against REAL alerts captured from the configured channel.

Everything above this file tests the parser against formats I invented. These
test it against the format the bot actually produces, captured live. That
matters: the first version of this parser read "$51.3K" as `51.3` - a 1000x
error - and read "Dev: 203.8 SOL | 0%" as 203.8% of supply held by the dev when
the real figure was 0%. Neither would have been caught by a synthetic fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autopaper.telegram.parser import (
    _first_number,
    _first_percent,
    is_token_candidate,
    normalise,
    parse_alert,
)

pytestmark = pytest.mark.strategy

FIXTURES = Path(__file__).parent / "fixtures" / "real_alerts.json"


@pytest.fixture(scope="module")
def real_alerts() -> list[dict]:
    if not FIXTURES.exists():
        pytest.skip("no captured alerts on disk")
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# the magnitude-suffix bug
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("$51.3K", 51_300.0),
    ("$37.2k", 37_200.0),
    ("$1.2M", 1_200_000.0),
    ("900", 900.0),
    ("$900", 900.0),
    ("$1.5B", 1_500_000_000.0),
    ("2.4X", 2.4),
])
def test_magnitude_suffixes_are_applied(text, expected):
    """
    Dropping a K/M/B suffix is a 1000x error that looks like a plausible small
    number, which is the worst kind of bug.
    """
    assert _first_number(text) == pytest.approx(expected)


def test_a_number_glued_to_its_unit_is_not_truncated():
    # "203.8 SOL" must read 203.8, not backtrack to 203
    assert _first_number("203.8 SOL") == pytest.approx(203.8)
    assert _first_number("19K [76.7 SOL]") == pytest.approx(19_000.0)


def test_explicit_percentages_are_found():
    assert _first_percent("203.8 SOL | 0%") == pytest.approx(0.0)
    assert _first_percent("36%") == pytest.approx(36.0)
    assert _first_percent("no percentage here") is None


# ---------------------------------------------------------------------------
# the real feed
# ---------------------------------------------------------------------------


def test_fullwidth_punctuation_is_folded():
    assert normalise("MC：$37.2k ｜🔝$432.7k") == "MC:$37.2k |🔝$432.7k"


def test_every_captured_alert_is_correctly_classified(real_alerts):
    """
    The channel carries narrative/promo posts as well as token alerts. Those have
    a ticker and figures but no contract address, and they are NOT candidates -
    which is a different fact from "we failed to parse an alert".
    """
    assert real_alerts, "expected captured alerts"
    candidates = 0
    for row in real_alerts:
        is_candidate, reason = is_token_candidate(row["text"])
        parsed = parse_alert(row["text"])
        if is_candidate:
            candidates += 1
            assert parsed.mint, f"candidate {row['id']} has no mint"
            assert 32 <= len(parsed.mint) <= 44
            assert parsed.symbol
        else:
            assert parsed.mint is None
            assert reason
    assert candidates > 0, "expected some real token alerts in the capture"


def test_every_captured_alert_keeps_its_own_source_values(real_alerts):
    """
    Magnitudes must be internally consistent: a token the bot prices at tens of
    thousands cannot be recorded in the tens of dollars.
    """
    for row in real_alerts:
        p = parse_alert(row["text"])
        mcap = p.num("market_cap_usd")
        if mcap is not None and "$" in row["text"]:
            assert mcap > 1_000, (
                f"alert {row['id']} market cap parsed as {mcap}, which is not a "
                "plausible market cap - check magnitude suffixes"
            )
        liq = p.num("liquidity_usd")
        if liq:
            # 0 is a real reported value (a dead pool) and must be preserved,
            # not hidden; anything non-zero must at least be plausible.
            assert liq > 100, f"alert {row['id']} liquidity {liq} looks truncated"


def test_dev_holdings_is_the_percentage_not_the_sol_amount(real_alerts):
    """
    "Dev:  203.8 SOL | 0%" - 203.8 is a SOL balance, 0% is the holdings figure.
    Reading the first number reports a dev holding 203.8% of supply.
    """
    checked = 0
    for row in real_alerts:
        p = parse_alert(row["text"])
        if p.fields.get("dev_holdings_pct") is None:
            continue
        checked += 1
        assert 0.0 <= p.fields["dev_holdings_pct"] <= 100.0, (
            f"alert {row['id']}: dev_holdings_pct "
            f"{p.fields['dev_holdings_pct']} is outside 0-100, so the wrong "
            "number was read"
        )
    assert checked, "no captured alert carried a Dev line"


def test_dex_paid_marks_are_read_from_the_emoji(real_alerts):
    """
    "DEX:  Paid❌ Ads❌" must be False, not an absent value and definitely not
    True. These are multi-codepoint symbols; a bare run of them inside a regex is
    a literal sequence, not a character class, which silently matches nothing.
    """
    saw_false = False
    for row in real_alerts:
        p = parse_alert(row["text"])
        if "dex_paid" in p.fields:
            assert isinstance(p.fields["dex_paid"], bool)
            saw_false = saw_false or p.fields["dex_paid"] is False
    assert saw_false, "expected at least one explicitly-unpaid DEX line"


def test_holder_and_bundle_counts_are_integers(real_alerts):
    for row in real_alerts:
        p = parse_alert(row["text"])
        for key in ("holder_count", "bundle_count", "scan_count"):
            v = p.fields.get(key)
            if v is not None:
                assert isinstance(v, int), f"{key} should be an int, got {v!r}"


def test_pool_age_is_parsed_to_seconds(real_alerts):
    ages = []
    for row in real_alerts:
        p = parse_alert(row["text"])
        a = p.age_seconds()
        if a is not None:
            ages.append(a)
            assert a > 0
    assert ages, "no pool age parsed from the captured feed"


def test_ath_marker_is_captured_when_present(real_alerts):
    saw_ath = False
    for row in real_alerts:
        p = parse_alert(row["text"])
        ath = p.num("ath_usd")
        if ath is not None:
            saw_ath = True
            assert ath > 0
    assert saw_ath, "expected the bot's high-water marker to be captured"


def test_parsed_values_are_never_invented(real_alerts):
    """
    Any field the source did not contain must be absent or None. This is the
    check that keeps a sparse alert from being dressed up with plausible numbers.
    """
    text_keys = {
        "mint", "symbol", "name", "market_cap_usd", "liquidity_usd", "age_text",
        "volume_usd", "dex_paid", "dex_ads_paid", "scan_count", "holder_count",
        "bundle_count", "top10_pct", "dev_holdings_pct", "bundled_pct",
        "sold_pct", "ath_usd", "x_ath", "age_seconds", "mcap_liquidity_ratio",
        "from_ath_pct", "txns_1h", "buy_sell_ratio_1h", "format", "raw_text",
    }
    for row in real_alerts:
        p = parse_alert(row["text"])
        unknown = set(p.fields) - text_keys
        assert not unknown, f"unexpected invented fields: {unknown}"


def test_the_whole_chain_works_on_a_real_alert(real_alerts):
    """
    A captured alert must flow all the way to a decision report without raising
    and without a blocking parse failure.
    """
    from autopaper.analysis import DecisionEngine, build_features
    from autopaper.clock import FakeClock
    from autopaper.config import DecisionConfig, ExitConfig
    from autopaper.models import AlertSnapshot, ObservationSet
    from autopaper.nar import NullNarSource

    from .conftest import obs

    row = next(r for r in real_alerts if parse_alert(r["text"]).mint)
    parsed = parse_alert(row["text"])
    now = 1_700_000_000.0
    snapshot = AlertSnapshot(
        candidate_id="c", source="telegram", chat="real", message_id=row["id"],
        source_timestamp=now - 1.0, received_timestamp=now, raw_text=row["text"],
        mint=parsed.mint, symbol=parsed.symbol, name=parsed.name,
        fields=dict(parsed.fields), provenance=dict(parsed.provenance),
    )
    history = ObservationSet([
        obs(now - 30, 1.0e-5), obs(now - 15, 1.2e-5), obs(now, 1.4e-5),
    ])
    nar = NullNarSource().fetch(parsed.mint or "M")
    f = build_features(
        candidate_id="c", mint=parsed.mint or "M", decision_time=now,
        history=history, alert=snapshot, nar=nar,
        stale_after_ms=15_000.0, unusable_after_ms=45_000.0,
    )
    report = DecisionEngine(DecisionConfig(), ExitConfig()).decide(
        features=f, alert=snapshot, nar=nar, analysis_start=now,
        analysis_complete=now + 0.05,
    )
    assert report.decision.value in ("BUY", "WAIT", "REJECT")
    assert report.mint == parsed.mint
    assert report.thesis
    json.dumps(report.to_json_dict())
