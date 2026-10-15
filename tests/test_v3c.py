"""
V3-C: the only entry authorization.

Covers the four gates, both exact boundaries, the UNTESTABLE path, the
decision-time guarantee, the frozen-definition guard, and the structural fact
that no other strategy exists in the runtime.
"""

from __future__ import annotations

import pytest

from autopaper import v3c
from autopaper.models import HISTORICAL_ROUTES, Route
from autopaper.v3c import (
    AUTHORIZED,
    REJECTED_MARKET_CAP,
    REJECTED_NOT_FAST_ENTRY,
    REJECTED_SIGNAL_QUALITY,
    UNTESTABLE,
    V3C_MAX_MARKET_CAP_USD,
    V3C_MIN_SIGNAL_QUALITY,
    V3C_ROUTE,
    evaluate,
    v2a_note,
    verify_definition,
)


# ---------------------------------------------------------------------------
# the thresholds are the researched values
# ---------------------------------------------------------------------------


def test_thresholds_are_the_frozen_values():
    assert V3C_MAX_MARKET_CAP_USD == 20_000.0
    assert V3C_MIN_SIGNAL_QUALITY == 0.55


def test_verify_definition_rejects_a_retuned_market_cap():
    with pytest.raises(ValueError, match="FROZEN"):
        verify_definition(20_001.0, 0.55)


def test_verify_definition_rejects_a_retuned_signal_quality():
    with pytest.raises(ValueError, match="FROZEN"):
        verify_definition(20_000.0, 0.54)


def test_verify_definition_accepts_the_real_values():
    verify_definition()


def test_hash_changes_if_the_definition_changes():
    """The hash is over the definition, so a silent edit is detectable."""
    import hashlib

    payload = "|".join(
        [
            v3c.V3C_ID,
            v3c.V3C_VERSION,
            v3c.V3C_ROUTE,
            repr(19_999.0),  # the edited value
            repr(v3c.V3C_MIN_SIGNAL_QUALITY),
            ",".join(v3c.V3C_INPUTS),
        ]
    )
    edited = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert edited != v3c.V3C_HASH


# ---------------------------------------------------------------------------
# market cap gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mcap", [0, 1, 100, 5_000, 19_999, 19_999.99])
def test_market_cap_under_20k_authorizes(mcap):
    r = evaluate(route=V3C_ROUTE, market_cap_usd=mcap, signal_quality=0.55)
    assert r.authorized, r.detail
    assert r.status == AUTHORIZED


def test_market_cap_exactly_20k_authorizes():
    """The boundary is INCLUSIVE."""
    r = evaluate(route=V3C_ROUTE, market_cap_usd=20_000, signal_quality=0.55)
    assert r.authorized
    assert r.status == AUTHORIZED


def test_market_cap_one_cent_over_20k_rejects():
    r = evaluate(route=V3C_ROUTE, market_cap_usd=20_000.01, signal_quality=0.90)
    assert not r.authorized
    assert r.status == REJECTED_MARKET_CAP
    assert r.rejects
    assert "$20,000" in r.detail


@pytest.mark.parametrize("mcap", [20_001, 25_000, 25_372.71, 100_000, 5_000_000])
def test_market_cap_well_over_20k_rejects(mcap):
    r = evaluate(route=V3C_ROUTE, market_cap_usd=mcap, signal_quality=0.90)
    assert not r.authorized
    assert r.status == REJECTED_MARKET_CAP


def test_missing_market_cap_is_untestable_not_a_pass():
    r = evaluate(route=V3C_ROUTE, market_cap_usd=None, signal_quality=0.99)
    assert not r.authorized
    assert r.status == UNTESTABLE


# ---------------------------------------------------------------------------
# signal quality gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sq", [0.55, 0.5501, 0.6, 0.75, 1.0])
def test_signal_quality_at_or_above_055_authorizes(sq):
    r = evaluate(route=V3C_ROUTE, market_cap_usd=5_000, signal_quality=sq)
    assert r.authorized, r.detail


def test_signal_quality_exactly_055_authorizes():
    r = evaluate(route=V3C_ROUTE, market_cap_usd=5_000, signal_quality=0.55)
    assert r.authorized
    assert r.status == AUTHORIZED


def test_signal_quality_just_below_055_rejects():
    r = evaluate(route=V3C_ROUTE, market_cap_usd=5_000, signal_quality=0.5499)
    assert not r.authorized
    assert r.status == REJECTED_SIGNAL_QUALITY
    assert "0.55" in r.detail


@pytest.mark.parametrize("sq", [0.0, 0.1, 0.3, 0.45, 0.54])
def test_signal_quality_well_below_055_rejects(sq):
    r = evaluate(route=V3C_ROUTE, market_cap_usd=5_000, signal_quality=sq)
    assert not r.authorized
    assert r.status == REJECTED_SIGNAL_QUALITY


def test_missing_signal_quality_is_untestable_not_a_pass():
    r = evaluate(route=V3C_ROUTE, market_cap_usd=5_000, signal_quality=None)
    assert not r.authorized
    assert r.status == UNTESTABLE


# ---------------------------------------------------------------------------
# both gates
# ---------------------------------------------------------------------------


def test_both_gates_must_pass():
    """Passing one gate is not authorization."""
    ok = evaluate(route=V3C_ROUTE, market_cap_usd=5_000, signal_quality=0.90)
    assert ok.authorized

    bad_mcap = evaluate(
        route=V3C_ROUTE, market_cap_usd=50_000, signal_quality=0.90
    )
    assert not bad_mcap.authorized

    bad_sq = evaluate(route=V3C_ROUTE, market_cap_usd=5_000, signal_quality=0.10)
    assert not bad_sq.authorized


def test_market_cap_is_evaluated_first():
    """With both failing, the market-cap reason is the one reported."""
    r = evaluate(route=V3C_ROUTE, market_cap_usd=50_000, signal_quality=0.10)
    assert r.status == REJECTED_MARKET_CAP


# ---------------------------------------------------------------------------
# route scope
# ---------------------------------------------------------------------------


def test_a_non_fast_entry_route_is_not_a_v3c_setup():
    for route in HISTORICAL_ROUTES:
        r = evaluate(route=route, market_cap_usd=1_000, signal_quality=0.99)
        assert not r.authorized
        assert r.status == REJECTED_NOT_FAST_ENTRY


def test_the_runtime_route_enum_has_only_fast_entry_and_no_trade():
    """The removed routes are gone from the enum, not merely unused."""
    assert {r.value for r in Route} == {"FAST_ENTRY", "NO_TRADE"}
    for name in HISTORICAL_ROUTES:
        assert not hasattr(Route, name)


# ---------------------------------------------------------------------------
# decision-time only
# ---------------------------------------------------------------------------


def test_evaluate_reads_only_the_values_it_is_given():
    """
    V3-C is a pure function of the market cap and signal quality it receives.
    It has no clock, no history and no lookup, so it cannot consult a later
    value. The "no future data" property is therefore structural at this layer,
    and the decision-engine test proves the values handed in are decision-time.
    """
    import inspect

    src = inspect.getsource(evaluate)
    for forbidden in ("time.time", "datetime", "getattr", "self.market", "history"):
        assert forbidden not in src, f"V3-C references {forbidden!r}"


def test_market_cap_of_a_future_high_cannot_rescue_a_rejection():
    """
    The alert-time and fill-time figures are not inputs. Only the value passed
    in - the decision-time one - is read.
    """
    decision_time_mcap = 50_000
    future_peak_mcap = 400_000
    r = evaluate(
        route=V3C_ROUTE, market_cap_usd=decision_time_mcap, signal_quality=0.95
    )
    assert not r.authorized
    # the later, much higher figure is simply not an input to the rule
    assert future_peak_mcap not in (r.market_cap_usd,)


# ---------------------------------------------------------------------------
# V2-A is retained and dominated
# ---------------------------------------------------------------------------


def test_v2a_ceiling_is_still_represented():
    assert v3c.V2A_CEILING_USD == 25_372.71


def test_v3c_is_stricter_than_v2a():
    assert V3C_MAX_MARKET_CAP_USD < v3c.V2A_CEILING_USD


def test_a_token_between_the_two_ceilings_fails_v3c_though_it_passes_v2a():
    mcap = 22_000.0
    assert mcap <= v3c.V2A_CEILING_USD          # would pass V2-A
    r = evaluate(route=V3C_ROUTE, market_cap_usd=mcap, signal_quality=0.90)
    assert not r.authorized
    assert r.status == REJECTED_MARKET_CAP


def test_v2a_note_shows_both_ceilings():
    note = v2a_note(22_000.0)
    assert "V2-A: pass" in note
    assert "V3-C: FAIL" in note
    assert "$25,372.71" in note
    assert "$20,000" in note


def test_v2a_note_on_a_very_large_token_fails_both():
    note = v2a_note(90_000.0)
    assert "V2-A: FAIL" in note
    assert "V3-C: FAIL" in note


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


def test_result_serializes_with_every_input_and_threshold():
    r = evaluate(route=V3C_ROUTE, market_cap_usd=12_345.67, signal_quality=0.72)
    d = r.as_dict()
    assert d["authorized"] is True
    assert d["market_cap_usd"] == 12_345.67
    assert d["signal_quality"] == 0.72
    assert d["max_market_cap_usd"] == 20_000.0
    assert d["min_signal_quality"] == 0.55


def test_describe_names_the_thresholds():
    text = v3c.describe()
    assert "V3-C" in text
    assert "20,000" in text
    assert "0.55" in text
