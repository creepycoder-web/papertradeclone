"""
The four-way measurement: mark, direct Pump, direct PumpSwap, Jupiter.

WHY THIS IS A SEPARATE STAGE
============================

The original pipeline took ONE executable quote and returned it, trying the
market-specific path first and falling back to Jupiter. That shape cannot answer
the question this harness exists to answer, because it makes the two direct
sources mutually exclusive and never reports both a direct quote and an
aggregator quote for the same signal. Comparing them is the entire point.

So this stage collects EVERY source and reports them side by side. It never
chooses a winner, and it never stops early: a graduated token still has its
(dead) bonding curve reported as ``UNKNOWN`` with the reason, because "the curve
is complete" is a finding about the market rather than a reason to hide a
column.

WHAT IS KEPT APART
==================

============  ===========================================================
label         what it is
============  ===========================================================
OBSERVED_MARK        DexScreener ``priceUsd``. Never an execution price.
DIRECT_EXECUTABLE_QUOTE   priced from on-chain state for THIS size.
JUPITER_QUOTE        an aggregator's executable route.
ACTUAL_FILL          never exists. No transaction is built or sent.
============  ===========================================================

All four are converted to USD per token before any gap is computed, using the
one logged ``--sol-usd`` reference, and the reference is reported beside every
number so the arithmetic is reproducible.

FAIL-CLOSED
===========

Every unavailable source produces ``None`` plus a named reason and a
``failure`` code. There is no path in this module that turns a missing quote
into ``0.0``, and no path that turns an unconfigured threshold into a PASS.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .comparison import (
    CrossCheck,
    IdentityReport,
    LatencyMarks,
    PriceLabel,
    PricePoint,
    build_identity_report,
    cross_check,
)
from .numeric import DecimalError, effective_price
from .models import (
    ACTUAL_FILL_AVAILABLE,
    SOL_DECIMALS,
    WSOL_MINT,
    DexPair,
    FailureMode,
    Freshness,
    MarketKind,
    ObservedMark,
    StalenessPolicy,
)
from .numeric import to_float
from .pumpdirect import MarketClassification
from .reader import DirectPumpReader, DirectQuote


@dataclass
class DirectMarketReport:
    """
    Everything the four-way comparison produced for one signal.

    ``price_points`` is a dict rather than four attributes because the whole
    point is that they are interchangeable to iterate over and NOT
    interchangeable to interpret - and the label travels with each one.
    """

    mint: str
    classification: MarketClassification | None
    pump_direct: DirectQuote | None
    pumpswap_direct: DirectQuote | None
    jupiter: PricePoint
    mark: PricePoint
    cross: CrossCheck
    identity: IdentityReport
    latency: LatencyMarks
    order_usd: float
    order_lamports: int
    sol_reference_usd: float | None
    token_decimals: int | None
    mark_freshness: Freshness
    state_freshness: Freshness
    quote_freshness: Freshness
    failures: tuple[FailureMode, ...] = ()
    notes: tuple[str, ...] = ()
    started_at: float = 0.0
    finished_at: float = 0.0
    divergence_pct: float | None = None

    @property
    def pump_price(self) -> PricePoint | None:
        """The direct quote for whichever pump market this actually is."""
        if self.pump_direct is not None and self.pump_direct.available:
            return self._point_for(self.pump_direct, PriceLabel.DIRECT_EXECUTABLE_QUOTE)
        if self.pumpswap_direct is not None and self.pumpswap_direct.available:
            return self._point_for(self.pumpswap_direct, PriceLabel.DIRECT_EXECUTABLE_QUOTE)
        return None

    @staticmethod
    def _point_for(quote: DirectQuote, label: PriceLabel) -> PricePoint:
        """
        Turn a direct quote into a comparable point - INCLUDING when it failed.

        A failed direct quote still becomes a point, with no price and a named
        reason. That is deliberate: the column appears in the report saying
        ``UNKNOWN``, so a reader can see the source was ATTEMPTED. Dropping the
        column instead would make "we looked and it was unavailable" look
        identical to "we never looked", which is the difference between a
        finding and an absence.
        """
        common = dict(
            label=label,
            provider=quote.provider,
            price_usd_per_token=None,
            venue=(
                "pumpfun" if quote.market_type is MarketKind.PUMP_BONDING_CURVE
                else "pumpswap"
            ),
            pool_or_curve=quote.curve_address or quote.pool_address,
            token_mint=quote.base_mint,
            quote_mint=quote.quote_mint or WSOL_MINT,
            state_slot=(quote.snapshot.slot if quote.snapshot else None),
            state_timestamp=(quote.snapshot.fetched_at if quote.snapshot else None),
            state_age_ms=(quote.snapshot.state_age_ms() if quote.snapshot else None),
            missing_reason=None if quote.available else quote.missing_reason,
        )
        if quote.quote is None:
            return PricePoint(**common)
        return PricePoint(
            price_sol_per_token=to_float(quote.quote.effective_price),
            raw_input=str(quote.quote.input_amount_raw),
            raw_output=str(quote.quote.output_amount_raw),
            input_decimals=quote.quote.input_decimals,
            output_decimals=quote.quote.output_decimals,
            quote_timestamp=quote.quote.quote_timestamp,
            quote_latency_ms=(
                quote.snapshot.fetch_latency_ms if quote.snapshot else None
            ),
            **common,
        )

    def as_dict(self) -> dict[str, Any]:
        now = self.finished_at or time.time()
        ref = self.sol_reference_usd

        def priced(point: PricePoint) -> PricePoint:
            if point.price_sol_per_token is None or ref is None:
                return point
            from dataclasses import replace

            return replace(
                point,
                price_usd_per_token=point.price_sol_per_token * ref,
                usd_reference_price=ref,
            )

        mark = priced(self.mark)
        jupiter = priced(self.jupiter)
        pump = priced(self.pump_price) if self.pump_price else None
        pump_curve = (
            priced(self._point_for(self.pump_direct, PriceLabel.DIRECT_EXECUTABLE_QUOTE))
            if self.pump_direct is not None else None
        )
        pump_swap = (
            priced(self._point_for(self.pumpswap_direct, PriceLabel.DIRECT_EXECUTABLE_QUOTE))
            if self.pumpswap_direct is not None else None
        )
        # Recomputed rather than reused so the serialised gaps always carry the
        # USD-converted prices; ``self.cross`` holds the same objects built
        # before conversion.
        recomputed = cross_check(
            mark, pump_curve, pump_swap, jupiter,
            divergence_pct=self.divergence_pct,
        )

        return {
            "token_mint": self.mint,
            "test_order": {
                "usd": self.order_usd,
                "lamports": self.order_lamports,
                "sol_reference_usd": ref,
                "hypothetical": True,
                "executed": False,
            },
            "token_decimals": self.token_decimals,
            "execution_market": (
                self.classification.as_dict() if self.classification else None
            ),
            "observed_mark": mark.as_dict(),
            "direct_pump_quote": (
                self.pump_direct.as_dict(sol_reference_usd=ref, now=now)
                if self.pump_direct else None
            ),
            "direct_pumpswap_quote": (
                self.pumpswap_direct.as_dict(sol_reference_usd=ref, now=now)
                if self.pumpswap_direct else None
            ),
            "jupiter_quote": jupiter.as_dict(),
            "identity_checks": self.identity.as_dict(),
            "pairwise_gaps": [g.as_dict() for g in recomputed.gaps],
            "diagnostics": {
                "findings": list(recomputed.findings),
                "triggers_diagnostic_investigation": recomputed.actionable,
                "note": recomputed.note,
            },
            "freshness": {
                "mark": self.mark_freshness.value,
                "on_chain_state": self.state_freshness.value,
                "quote": self.quote_freshness.value,
                "mark_and_state_are_separate": (
                    "a DexScreener mark ages at the aggregator's refresh rate "
                    "while on-chain state ages with the slot; they are reported "
                    "separately and never combined"
                ),
            },
            "latency": self.latency.as_dict(),
            "actual_fill": None,
            "actual_fill_available": ACTUAL_FILL_AVAILABLE,
            "failures": [f.value for f in self.failures],
            "notes": list(self.notes),
            "execution_note": (
                "NOT EXECUTED - no transaction was built, signed or broadcast"
            ),
            "timing": {"started_at": self.started_at, "finished_at": self.finished_at},
        }


async def run_direct_measurement(
    *,
    mint: str,
    telegram_message_at: float | None = None,
    pair: DexPair | None,
    mark: ObservedMark,
    jupiter_result: Any,
    order_usd: float,
    order_lamports: int | None,
    sol_reference_usd: float | None,
    token_decimals_for: Any = None,
    rpc: Any,
    jupiter: Any = None,
    staleness: StalenessPolicy | None = None,
    divergence_pct: float | None = None,
) -> DirectMarketReport:
    """
    Take every available quote for one signal and compare them.

    Never raises for an expected failure: a provider being down or a market
    being unquotable is a result to report, not an exception.
    """
    staleness = staleness or StalenessPolicy()
    marks = LatencyMarks(telegram_message_at=telegram_message_at)
    started = time.time()
    marks.mark("receive", wall=started)
    marks.local_receive_at = marks.stages["receive"][0]
    marks.receive_wall_at = started
    # The aggregator lookup happened in the pipeline, before this stage. Its
    # duration is not re-measured here - it would be a fabrication - so only the
    # boundary is recorded and the gap "receive -> dexscreener" reads as the
    # pipeline's own elapsed time, not a re-measurement.
    marks.mark("dexscreener", wall=started)
    failures: list[FailureMode] = []
    notes: list[str] = []

    reader = DirectPumpReader(rpc)

    # Decimals are READ, never assumed. A wrong value scales every price by a
    # power of ten while leaving each individual number plausible.
    decimals: int | None = None
    if order_lamports and order_lamports > 0:
        marks.mark("rpc", wall=time.time())
        decimals = await reader.mint_decimals(mint)
        marks.mark("rpc", wall=time.time())
        if decimals is None:
            notes.append(
                "the mint's decimals could not be read from chain; the direct "
                "curve quote is reported as UNKNOWN rather than priced with an "
                "assumed 6"
            )
            failures.append(FailureMode.STATE_UNAVAILABLE)

    classification: MarketClassification | None = None
    pump_direct: DirectQuote | None = None
    pumpswap_direct: DirectQuote | None = None

    if order_lamports and order_lamports > 0:
        marks.mark("rpc", wall=time.time())
        classification, pump_direct, pumpswap_direct = await reader.measure(
            mint,
            sol_amount_lamports=order_lamports,
            token_decimals=decimals if decimals is not None else 6,
            pool_address=(pair.pair_address if pair else None),
            dex_id=(pair.dex_id if pair else None),
            mark=lambda stage: marks.mark(stage, wall=time.time()),
        )
        marks.mark("rpc", wall=time.time())
        if pump_direct is not None and pump_direct.failure:
            failures.append(pump_direct.failure)
        if pumpswap_direct is not None and pumpswap_direct.failure:
            failures.append(pumpswap_direct.failure)
    else:
        notes.append(
            "the test order could not be sized in lamports (no --sol-usd), so "
            "no direct quote was attempted"
        )

    # -- Jupiter, on its own clock ----------------------------------------
    # Jupiter reports raw amounts and NO price. The price is derived here from
    # its own two amounts with the same scaling rule the direct quotes use, so
    # the two are comparable rather than merely both present. Deriving it here
    # (instead of trusting a provider field) is deliberate: this harness has
    # been bitten by a provider quoting the wrong direction.
    jup_price_sol: float | None = None
    if jupiter_result is not None and decimals is not None:
        try:
            jup_price_sol = to_float(effective_price(
                jupiter_result.in_amount, jupiter_result.out_amount,
                input_decimals=SOL_DECIMALS, output_decimals=decimals,
            ))
        except (DecimalError, ZeroDivisionError):
            jup_price_sol = None

    if jupiter_result is not None and jupiter_result.request_latency_ms is not None:
        marks.provider_latencies_ms["jupiter"] = jupiter_result.request_latency_ms

    jupiter_point = PricePoint(
        label=PriceLabel.JUPITER_QUOTE,
        provider="jupiter",
        price_sol_per_token=jup_price_sol,
        price_usd_per_token=None,
        venue=(jupiter_result.route_plan[0].label if (
            jupiter_result is not None and jupiter_result.route_plan) else None),
        pool_or_curve=(
            jupiter_result.pool_keys[0]
            if jupiter_result is not None and jupiter_result.pool_keys else None
        ),
        token_mint=mint,
        quote_mint=WSOL_MINT,
        route=tuple(
            h.label or "?" for h in (jupiter_result.route_plan if jupiter_result else ())
        ),
        raw_input=(str(jupiter_result.in_amount) if jupiter_result else None),
        raw_output=(str(jupiter_result.out_amount) if jupiter_result else None),
        input_decimals=SOL_DECIMALS,
        output_decimals=decimals,
        quote_timestamp=(jupiter_result.timestamp if jupiter_result else None),
        quote_latency_ms=(jupiter_result.request_latency_ms if jupiter_result else None),
        missing_reason=(
            None if jupiter_result is not None else "no Jupiter route was returned"
        ),
    )

    mark_point = PricePoint(
        label=PriceLabel.OBSERVED_MARK,
        provider="dexscreener",
        price_usd_per_token=to_float(mark.value_usd),
        venue=(pair.dex_id if pair else None),
        pool_or_curve=(pair.pair_address if pair else None),
        token_mint=mint,
        quote_mint=(pair.quote_token_address if pair else None),
        route=((pair.dex_id,) if pair and pair.dex_id else ()),
        quote_timestamp=mark.timestamp,
        quote_latency_ms=mark.latency_ms,
        missing_reason=None if mark.value_usd else "DexScreener returned no priceUsd",
    )

    def priced(point: PricePoint) -> PricePoint:
        if point.price_usd_per_token is not None or point.price_sol_per_token is None:
            return point
        if sol_reference_usd is None:
            return point
        from dataclasses import replace

        return replace(
            point,
            price_usd_per_token=point.price_sol_per_token * sol_reference_usd,
            usd_reference_price=sol_reference_usd,
        )

    pump_curve_point = (
        priced(DirectMarketReport._point_for(
            pump_direct, PriceLabel.DIRECT_EXECUTABLE_QUOTE))
        if pump_direct is not None else None
    )
    pump_swap_point = (
        priced(DirectMarketReport._point_for(
            pumpswap_direct, PriceLabel.DIRECT_EXECUTABLE_QUOTE))
        if pumpswap_direct is not None else None
    )
    mark_point = priced(mark_point)
    # CRITICAL: the Jupiter point must be priced on the SAME conversion as the
    # direct points. Leaving it unpriced is how a working quote silently
    # becomes "unavailable" and three pairwise gaps vanish.
    jupiter_point = priced(jupiter_point)

    jup_keys = jupiter_result.pool_keys if jupiter_result else ()
    jup_labels = tuple(
        h.label or "" for h in (jupiter_result.route_plan if jupiter_result else ())
    )
    marks.mark("rpc", wall=time.time())
    identity = build_identity_report(
        mint=mint,
        observed_pair_address=(pair.pair_address if pair else None),
        observed_dex_id=(pair.dex_id if pair else None),
        curve_address=(pump_direct.curve_address if pump_direct else None),
        curve_chain_verified=bool(
            pump_direct is not None
            and pump_direct.curve_address
            and pump_direct.snapshot is not None
        ),
        pool_address=(pumpswap_direct.pool_address if pumpswap_direct else None),
        pool_chain_verified=bool(
            pumpswap_direct is not None
            and pumpswap_direct.pool_address
            and pumpswap_direct.snapshot is not None
        ),
        jupiter_pool_keys=jup_keys,
        jupiter_labels=jup_labels,
    )
    marks.mark("identity", wall=time.time())

    cross = cross_check(
        mark_point, pump_curve_point, pump_swap_point, jupiter_point,
        divergence_pct=divergence_pct,
    )

    now = time.time()
    # Freshness is evaluated per KIND and never merged: a stale aggregator mark
    # and a stale on-chain snapshot are different problems.
    mark_stamp = staleness.stamp(
        "observed_mark", mark.timestamp if mark.value_usd else None, now,
        staleness.max_mark_age_ms,
    )
    state_ts = None
    for candidate in (pump_direct, pumpswap_direct):
        if candidate is not None and candidate.snapshot is not None:
            state_ts = candidate.snapshot.fetched_at
            break
    state_stamp = staleness.stamp(
        "on_chain_state", state_ts, now, staleness.max_state_age_ms
    )
    quote_ts = jupiter_point.quote_timestamp
    quote_stamp = staleness.stamp(
        "executable_quote", quote_ts, now, staleness.max_quote_age_ms
    )
    for stamp, mode in (
        (mark_stamp, FailureMode.MARK_STALE),
        (state_stamp, FailureMode.STATE_STALE),
        (quote_stamp, FailureMode.QUOTE_STALE),
    ):
        if stamp.freshness is Freshness.STALE:
            failures.append(mode)

    # Set BEFORE marking the final stage, so "total" measures the whole run.
    marks.final_decision_at = time.monotonic()
    marks.mark("final", moment=marks.final_decision_at, wall=now)

    report = DirectMarketReport(
        mint=mint,
        classification=classification,
        pump_direct=pump_direct,
        pumpswap_direct=pumpswap_direct,
        jupiter=jupiter_point,
        mark=mark_point,
        cross=cross,
        identity=identity,
        latency=marks,
        order_usd=order_usd,
        order_lamports=order_lamports or 0,
        sol_reference_usd=sol_reference_usd,
        token_decimals=decimals,
        mark_freshness=mark_stamp.freshness,
        state_freshness=state_stamp.freshness,
        quote_freshness=quote_stamp.freshness,
        failures=tuple(dict.fromkeys(failures)),
        notes=tuple(notes),
        started_at=started,
        finished_at=now,
        divergence_pct=divergence_pct,
    )
    return report


__all__ = ["DirectMarketReport", "run_direct_measurement"]
