"""Explicit assertion that a non-stale available quote is NOT rejected.

This is the acceptance-criteria test for the quote guard ordering fix.
"""
from __future__ import annotations

import time

from autopaper.paper_live.fills import _non_stale_available_quote_is_not_rejected
from autopaper.paper_live.schema import QuotePoint


def test_non_stale_available_quote_is_not_rejected():
    now = time.time()
    quote = QuotePoint(
        side="SELL",
        provider="pump_direct",
        market_type="PUMP_BONDING_CURVE",
        token_mint="MINT",
        quote_mint="WSOL",
        input_amount_raw=1000,
        input_decimals=6,
        input_amount_decimal=0.001,
        output_amount_raw=500,
        output_decimals=9,
        output_amount_decimal=5e-7,
        effective_price=0.5,
        effective_price_usd=60.0,
        price_impact_pct=0.0,
        fee_bps=0,
        fee_amount=0,
        venue="pumpfun-bonding-curve",
        pool_or_curve="PAIR",
        route=("pump_direct",),
        state_slot=1,
        state_age_ms=10.0,
        quote_timestamp=now,
        quote_age_ms=5.0,
        max_quote_age_ms=30_000.0,
        quote_latency_ms=9.0,
        identity="PASS",
        identity_evidence="the marked pair IS the pool",
        observed_pair_address="PAIR",
        available=True,
        venue_id="PUMP_BONDING_CURVE:PAIR",
        execution_native_reference=None,
    )
    _non_stale_available_quote_is_not_rejected(quote, side="SELL")
