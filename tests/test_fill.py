"""
Paper fill model.

The point of these tests is that the fill can never be flattering: it must be at
or worse than both the quote and the print, it must cost money, and every
failure mode must be a recorded outcome rather than a silent retry.
"""

from __future__ import annotations

import pytest

from autopaper.config import ExecutionModel
from autopaper.execution import FillInputs, impact_bps, simulate_fill
from autopaper.models import DataState

from .conftest import MINT, obs

pytestmark = pytest.mark.execution

T0 = 1_700_000_000.0
MODEL = ExecutionModel()


def inputs(**kw) -> FillInputs:
    base = dict(
        side="entry",
        quantity=1000.0,
        notional_hint_usd=10.0,
        quoted_price=1e-4,
        decision_time=T0,
        observation=obs(T0 + 1.2, 1.05e-4, liquidity_usd=50_000.0),
        observed_at=T0 + 1.2,
    )
    base.update(kw)
    return FillInputs(**base)


def test_fill_uses_the_price_after_latency_not_the_quote():
    r = simulate_fill(inputs(), MODEL)
    assert r.filled
    # quote was 1e-4, the print at fill time was 1.05e-4; entry takes the worse
    assert r.price_at_quote == pytest.approx(1.05e-4)
    assert r.fill_price > r.price_at_quote          # slippage on top
    assert r.fill_price > r.quoted_price
    assert r.latency_ms == pytest.approx(1200.0, abs=1.0)


def test_fill_is_never_better_than_the_quote():
    """Even if the market rallies during the latency, the fill must not improve."""
    r = simulate_fill(
        inputs(observation=obs(T0 + 1.2, 5e-5, liquidity_usd=50_000.0)), MODEL
    )
    assert r.price_at_quote == pytest.approx(1e-4), "must use max(quote, print)"
    assert r.fill_price > r.quoted_price


def test_exit_takes_the_worse_price_too():
    r = simulate_fill(
        inputs(side="exit", observation=obs(T0 + 0.5, 1.2e-4, liquidity_usd=50_000.0)),
        MODEL,
    )
    # exit is a sale: we receive the worse of mark and print, and slippage is
    # subtracted. Selling must never be modelled as a favourable fill.
    assert r.price_at_quote == pytest.approx(1e-4)
    assert r.fill_price < r.price_at_quote
    assert r.slippage_usd > 0


def test_exit_slippage_direction_is_never_favourable():
    for print_price in (5e-5, 1e-4, 5e-4):
        r = simulate_fill(
            inputs(side="exit", observation=obs(T0 + 0.2, print_price,
                                                liquidity_usd=50_000.0)),
            MODEL,
        )
        assert r.fill_price <= min(r.price_at_quote, print_price)


def test_fees_are_charged():
    r = simulate_fill(inputs(), MODEL)
    assert r.fee_bps == MODEL.entry_fee_bps
    assert r.fee_usd == pytest.approx(r.notional_usd * MODEL.entry_fee_bps / 10_000)


def test_slippage_scales_with_size_against_liquidity():
    small = simulate_fill(inputs(quantity=1_000.0), MODEL)
    big = simulate_fill(inputs(quantity=2_000_000.0), MODEL)
    assert big.slippage_bps > small.slippage_bps
    assert big.slippage_usd > small.slippage_usd


def test_impact_is_deterministic_and_capped():
    a = impact_bps(5_000, 25_000, MODEL)
    b = impact_bps(5_000, 25_000, MODEL)
    assert a == b
    assert 0 < a <= MODEL.impact_max_bps
    assert impact_bps(10_000_000, 25_000, MODEL) == MODEL.impact_max_bps


def test_unknown_liquidity_does_not_make_the_fill_cheaper():
    # a realistic ~$15 notional, into a deep known pool vs a pool we know nothing about
    deep = simulate_fill(
        inputs(quantity=150_000.0,
               observation=obs(T0 + 1.2, 1.05e-4, liquidity_usd=5_000_000.0)),
        MODEL,
    )
    unknown = simulate_fill(
        inputs(quantity=150_000.0,
               observation=obs(T0 + 1.2, 1.05e-4, liquidity_usd=None)),
        MODEL,
    )
    # missing liquidity must never be treated as a DEEP pool. It used to be
    # substituted with a fabricated $60,000, which made an unknown liquidity
    # cheaper (1bp) than a real $17.6k pool (2.4bp) -- flattering every fill
    # whose venue state could not be read.
    assert unknown.slippage_bps > deep.slippage_bps
    assert unknown.liquidity_usd is None
    # the absence is named, in whatever casing the detail uses
    assert "unknown" in unknown.detail.lower()
    # ...and it is carried structurally, not only in prose
    assert unknown.liquidity_source == "UNKNOWN"
    assert unknown.execution_fidelity == "PAPER/LOW_FIDELITY_NO_LIQUIDITY"
    # an observed liquidity is labelled as such and priced off the real number
    assert deep.liquidity_source == "OBSERVED"


def test_no_quote_is_unfilled_not_faked():
    r = simulate_fill(inputs(observation=None), MODEL)
    assert not r.filled
    assert r.reason == "no_quote"
    assert r.fill_price is None
    assert r.fee_usd == 0.0


def test_market_unavailable_is_distinct_from_no_quote():
    r = simulate_fill(inputs(observation=obs(T0 + 1.2, None)), MODEL)
    assert not r.filled
    assert r.reason == "market_unavailable"


def test_stale_quote_refuses_to_fill():
    stale = obs(T0 - 0.2, 1.05e-4)          # 1400ms old at fill time
    ok = obs(T0 + 1.19, 1.05e-4)           # 10ms old
    assert simulate_fill(inputs(observation=stale), MODEL).filled
    very_stale = obs(T0 - 20.0, 1.05e-4)   # 21s old
    r = simulate_fill(inputs(observation=very_stale), MODEL)
    assert not r.filled
    assert r.reason == "stale_quote"
    assert simulate_fill(inputs(observation=ok), MODEL).filled


def test_staleness_adds_slippage():
    fresh = simulate_fill(inputs(observation=obs(T0 + 1.1, 1.05e-4)), MODEL)
    old = simulate_fill(inputs(observation=obs(T0 - 2.0, 1.05e-4)), MODEL)
    assert old.slippage_bps > fresh.slippage_bps


def test_fill_records_full_provenance():
    r = simulate_fill(inputs(), MODEL)
    d = r.as_dict()
    for key in ("quoted_price", "fill_price", "slippage_bps", "slippage_usd",
                "fee_bps", "fee_usd", "latency_ms", "placed_at", "fill_at",
                "observed_at", "data_state", "notional_usd", "quantity",
                "liquidity_usd", "detail"):
        assert key in d, key
    assert d["observed_at"] == pytest.approx(T0 + 1.2)


def test_data_state_is_propagated():
    r = simulate_fill(inputs(), MODEL, data_state=DataState.STALE)
    assert r.data_state is DataState.STALE


def test_execution_model_is_documented_and_configurable():
    m = ExecutionModel(
        entry_fee_bps=50.0, exit_fee_bps=50.0, entry_slippage_bps=10.0,
        exit_slippage_bps=10.0, decision_latency_ms=0.0,
    )
    r = simulate_fill(
        FillInputs(
            side="entry", quantity=1000.0, notional_hint_usd=10.0, quoted_price=1e-4,
            decision_time=T0, observed_at=T0,
            observation=obs(T0, 1.0e-4, liquidity_usd=1_000_000.0),
        ),
        m,
    )
    assert r.fee_bps == 50.0
    assert r.latency_ms == pytest.approx(0.0)
    assert r.slippage_bps < MODEL.entry_slippage_bps + 20
    d = m.as_dict()
    assert d["market_impact_model"] == "sqrt_notional_over_liquidity"
    assert d["decision_latency_ms"] == 0.0
