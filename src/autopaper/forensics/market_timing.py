"""
READ-ONLY market state / quote timing diagnostics.

PURPOSE
=======
Explain WHY an observed DexScreener mark can diverge from an executable price
even when pool IDENTITY matches. Identity is necessary but not sufficient: two
prices from the same pool at different instants are not comparable, and the
project has no executable-quote telemetry at all.

This module is MEASUREMENT ONLY. It:
* records what was observed, when, from where, and at what slot
* derives one status label per observation
* defines NO trading threshold and rejects NOTHING

It extends, and does not modify, the existing identity rules in
``autopaper.same_market.identity``. Provider-specific identity handling is left
exactly where it is; this module consumes its output rather than re-deciding it.

THE FIVE STATUSES
=================
``SAME_WINDOW``
    The mark and the quote were obtained inside the same measurement window and
    the window is inside the observation's stated freshness. Nothing to explain.

``STALE_MARK``
    The mark's own timestamp is older than the window the quote was taken in.
    The gap is on the OBSERVED side.

``STALE_STATE``
    The on-chain state backing the quote is older than the quote's own claim.
    The gap is on the STATE side.

``STALE_QUOTE``
    The quote is older than the instant it is being compared against, i.e. the
    comparison itself is late.

``TIMING_UNKNOWN``
    A required timestamp is missing. **Never silently treated as SAME_WINDOW**,
    because a missing timestamp is the absence of evidence.

NO THRESHOLD IS DEFINED HERE. The freshness budget belongs to
``MarketConfig.stale_after_ms`` and is passed IN, so this module cannot invent
one and cannot drift from production's.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence


class MarkStateTimingStatus(str, Enum):
    SAME_WINDOW = "SAME_WINDOW"
    STALE_MARK = "STALE_MARK"
    STALE_STATE = "STALE_STATE"
    STALE_QUOTE = "STALE_QUOTE"
    TIMING_UNKNOWN = "TIMING_UNKNOWN"


#: Which side of the comparison a given status blames. Useful for grouping
#: without re-deriving the rule.
STATUS_SIDE = {
    MarkStateTimingStatus.SAME_WINDOW: "none",
    MarkStateTimingStatus.STALE_MARK: "observed_mark",
    MarkStateTimingStatus.STALE_STATE: "on_chain_state",
    MarkStateTimingStatus.STALE_QUOTE: "quote",
    MarkStateTimingStatus.TIMING_UNKNOWN: "unmeasurable",
}


def _num(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


@dataclass(frozen=True)
class MarkQuoteObservation:
    """
    One mark/quote pair, with every timing input retained.

    Nothing is derived at construction: this is the raw record, so a later
    analysis can re-derive deltas under a different freshness budget without
    going back to the source.
    """

    #: -- identity (provider-specific; supplied by the caller, never inferred) --
    mint: str
    dexscreener_pair_address: Optional[str]
    execution_pool: Optional[str]
    provider: Optional[str]
    #: A pool key disclosed by the quote provider, when it discloses one.
    quote_pool_key: Optional[str] = None
    #: The identity verdict, produced by same_market.identity. Carried, not
    #: recomputed: those rules are load-bearing and provider-specific.
    identity_state: Optional[str] = None

    #: -- the observed mark --
    observed_mark: Optional[float] = None
    observed_mark_timestamp: Optional[float] = None

    #: -- the executable quote --
    quote_price: Optional[float] = None
    quote_timestamp: Optional[float] = None
    quote_slot: Optional[int] = None

    #: -- the on-chain state behind the quote --
    state_slot: Optional[int] = None
    state_fetch_timestamp: Optional[float] = None

    #: -- where / when this record was assembled --
    provider_latency_ms: Optional[float] = None
    captured_at: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mint": self.mint,
            "dexscreener_pair_address": self.dexscreener_pair_address,
            "execution_pool": self.execution_pool,
            "provider": self.provider,
            "quote_pool_key": self.quote_pool_key,
            "identity_state": self.identity_state,
            "observed_mark": self.observed_mark,
            "observed_mark_timestamp": self.observed_mark_timestamp,
            "quote_price": self.quote_price,
            "quote_timestamp": self.quote_timestamp,
            "quote_slot": self.quote_slot,
            "state_slot": self.state_slot,
            "state_fetch_timestamp": self.state_fetch_timestamp,
            "provider_latency_ms": self.provider_latency_ms,
            "captured_at": self.captured_at,
        }


@dataclass(frozen=True)
class TimingDiagnostics:
    """Derived view of one observation. Measurement only; no verdict on quality."""

    mark_state_timing_status: MarkStateTimingStatus
    timed_side: str
    mark_age_ms: Optional[float]
    quote_latency_ms: Optional[float]
    wall_clock_delta_ms: Optional[float]
    slot_delta: Optional[int]
    state_age_ms: Optional[float]
    price_gap_pct: Optional[float]
    freshness_budget_ms: Optional[float]
    reason: str
    missing_fields: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "MARK_STATE_TIMING_STATUS": self.mark_state_timing_status.value,
            "timed_side": self.timed_side,
            "mark_age_ms": self.mark_age_ms,
            "quote_latency_ms": self.quote_latency_ms,
            "wall_clock_delta_ms": self.wall_clock_delta_ms,
            "slot_delta": self.slot_delta,
            "state_age_ms": self.state_age_ms,
            "price_gap_pct": self.price_gap_pct,
            "freshness_budget_ms": self.freshness_budget_ms,
            "reason": self.reason,
            "missing_fields": list(self.missing_fields),
            "is_a_trading_decision": False,
        }


def classify_timing(
    obs: MarkQuoteObservation,
    *,
    freshness_budget_ms: Optional[float] = None,
    reference_time: Optional[float] = None,
) -> TimingDiagnostics:
    """
    Derive the timing status of one mark/quote pair.

    ``freshness_budget_ms`` is REQUIRED to say anything other than
    SAME_WINDOW or TIMING_UNKNOWN, and is passed in from the project's own
    ``MarketConfig.stale_after_ms`` rather than defined here. That is deliberate:
    this module must not be able to invent or drift from a trading threshold.
    """
    budget = _num(freshness_budget_ms)
    ref = _num(reference_time)
    if ref is None:
        ref = _num(obs.captured_at)
    if ref is None:
        ref = time.time()

    mark_ts = _num(obs.observed_mark_timestamp)
    quote_ts = _num(obs.quote_timestamp)
    state_fetch = _num(obs.state_fetch_timestamp)

    missing: list[str] = []
    if mark_ts is None:
        missing.append("observed_mark_timestamp")
    if quote_ts is None:
        missing.append("quote_timestamp")

    mark_age = (ref - mark_ts) * 1000.0 if mark_ts is not None else None
    quote_latency = (ref - quote_ts) * 1000.0 if quote_ts is not None else None
    wall = ((quote_ts - mark_ts) * 1000.0
            if (quote_ts is not None and mark_ts is not None) else None)
    slot_delta = (
        obs.quote_slot - obs.state_slot
        if (obs.quote_slot is not None and obs.state_slot is not None) else None)
    state_age = ((quote_ts - state_fetch) * 1000.0
                 if (quote_ts is not None and state_fetch is not None) else None)

    gap = None
    mp, qp = _num(obs.observed_mark), _num(obs.quote_price)
    if mp is not None and qp is not None and mp != 0:
        gap = (qp / mp - 1.0) * 100.0

    def done(status: MarkStateTimingStatus, reason: str) -> TimingDiagnostics:
        return TimingDiagnostics(
            mark_state_timing_status=status,
            timed_side=STATUS_SIDE[status],
            mark_age_ms=mark_age, quote_latency_ms=quote_latency,
            wall_clock_delta_ms=wall, slot_delta=slot_delta,
            state_age_ms=state_age, price_gap_pct=gap,
            freshness_budget_ms=budget, reason=reason,
            missing_fields=tuple(missing),
        )

    # A missing timestamp can never be SAME_WINDOW. Reporting it as such would
    # claim a freshness that was never established.
    if mark_ts is None or quote_ts is None:
        return done(
            MarkStateTimingStatus.TIMING_UNKNOWN,
            "cannot compare: " + ", ".join(missing)
            + " unavailable; this is the absence of evidence, not evidence of "
              "freshness")

    if budget is None or budget <= 0:
        return done(
            MarkStateTimingStatus.TIMING_UNKNOWN,
            "no freshness budget supplied, so no timing verdict can be made; "
            "pass MarketConfig.stale_after_ms")

    # The ORDER matters and is deliberate: the state is checked before the
    # quote, because a state that is already older than the quote cannot be
    # rescued by a fresh quote. Then the mark. The quote's own age is last,
    # because it describes the comparison rather than the data.
    if state_age is not None and state_age > budget:
        return done(
            MarkStateTimingStatus.STALE_STATE,
            f"on-chain state is {state_age:.0f}ms older than the quote, past "
            f"the {budget:.0f}ms freshness budget")
    if mark_age is not None and mark_age > budget:
        return done(
            MarkStateTimingStatus.STALE_MARK,
            f"observed mark is {mark_age:.0f}ms old at comparison time, past "
            f"the {budget:.0f}ms freshness budget")
    if quote_latency is not None and quote_latency > budget:
        return done(
            MarkStateTimingStatus.STALE_QUOTE,
            f"quote is {quote_latency:.0f}ms old at comparison time, past the "
            f"{budget:.0f}ms freshness budget")
    return done(
        MarkStateTimingStatus.SAME_WINDOW,
        f"mark, state and quote are all inside the {budget:.0f}ms budget "
        f"(mark age {mark_age:.0f}ms, quote age {quote_latency:.0f}ms)")


def build_observation(
    *,
    mint: str,
    dexscreener_pair_address: Optional[str] = None,
    execution_pool: Optional[str] = None,
    provider: Optional[str] = None,
    quote_pool_key: Optional[str] = None,
    identity_state: Optional[str] = None,
    observed_mark: Optional[float] = None,
    observed_mark_timestamp: Optional[float] = None,
    quote_price: Optional[float] = None,
    quote_timestamp: Optional[float] = None,
    quote_slot: Optional[int] = None,
    state_slot: Optional[int] = None,
    state_fetch_timestamp: Optional[float] = None,
    provider_latency_ms: Optional[float] = None,
    captured_at: Optional[float] = None,
) -> MarkQuoteObservation:
    """Named-argument constructor, so a caller cannot silently mis-order fields."""
    return MarkQuoteObservation(
        mint=mint,
        dexscreener_pair_address=dexscreener_pair_address,
        execution_pool=execution_pool,
        provider=provider,
        quote_pool_key=quote_pool_key,
        identity_state=identity_state,
        observed_mark=observed_mark,
        observed_mark_timestamp=observed_mark_timestamp,
        quote_price=quote_price,
        quote_timestamp=quote_timestamp,
        quote_slot=quote_slot,
        state_slot=state_slot,
        state_fetch_timestamp=state_fetch_timestamp,
        provider_latency_ms=provider_latency_ms,
        captured_at=captured_at,
    )


# ---------------------------------------------------------------------------
# aggregate view
# ---------------------------------------------------------------------------


def summarise(
    pairs: Sequence[tuple[MarkQuoteObservation, TimingDiagnostics]]
) -> dict[str, Any]:
    """
    Distribution over a batch. Counts only -- no quality verdict, no threshold.

    The point of the aggregate is to answer "WHERE is the divergence coming
    from", which is a different question from "was the quote good".
    """
    by_status: dict[str, int] = {}
    by_side: dict[str, int] = {}
    gaps: list[float] = []
    marks: list[float] = []
    quotes: list[float] = []
    slots: list[int] = []
    for _obs, diag in pairs:
        s = diag.mark_state_timing_status.value
        by_status[s] = by_status.get(s, 0) + 1
        by_side[diag.timed_side] = by_side.get(diag.timed_side, 0) + 1
        if diag.price_gap_pct is not None:
            gaps.append(diag.price_gap_pct)
        if diag.mark_age_ms is not None:
            marks.append(diag.mark_age_ms)
        if diag.quote_latency_ms is not None:
            quotes.append(diag.quote_latency_ms)
        if diag.slot_delta is not None:
            slots.append(diag.slot_delta)

    def stat(xs: list[float]) -> Optional[dict[str, float]]:
        if not xs:
            return None
        s = sorted(xs)
        n = len(s)
        return {
            "n": n,
            "min": s[0],
            "median": s[n // 2],
            "p90": s[min(n - 1, int(0.9 * n))],
            "max": s[-1],
        }

    return {
        "pairs": len(pairs),
        "by_status": by_status,
        "by_timed_side": by_side,
        "price_gap_pct": stat(gaps),
        "mark_age_ms": stat(marks),
        "quote_latency_ms": stat(quotes),
        "slot_delta": stat([float(x) for x in slots]) if slots else None,
        "defines_a_threshold": False,
        "rejects_trades": False,
        "note": (
            "a mark/quote gap on a SAME_WINDOW pair is NOT evidence of "
            "mispricing: it is the cost of crossing the book plus the "
            "provider's own aggregation. This module measures timing so a gap "
            "can be attributed, and deliberately draws no conclusion about "
            "whether any particular gap was avoidable."),
    }


def identity_caveats() -> dict[str, Any]:
    """
    The provider-identity rules this module must NOT flatten.

    Carried explicitly so a future reader does not "simplify" them and start
    comparing a DexScreener ``pairAddress`` to a Jupiter ``ammKey`` as though
    they were the same object type.
    """
    return {
        "dexscreener_pair_address": (
            "the observed pool. An indexer-reported pool identity for one "
            "provider's view of a market."),
        "jupiter_quote_pool_key": (
            "a per-hop pool key from a route. A multi-hop route has NO single "
            "pool, so comparing it to a pair address is an honest UNKNOWN, not "
            "a mismatch and not a pass."),
        "direct_pool_address": (
            "a pool address known from the chain for a direct pump path; "
            "directly comparable to the observed pair address."),
        "rule": (
            "identity is decided in autopaper.same_market.identity with "
            "provider-specific rules. This module carries the verdict as an "
            "input and never re-derives it."),
        "venue_names": (
            "compared on canonical venue name, because providers disagree on "
            "casing ('pumpfun' vs 'Pumpfun')."),
        "unknown_is_not_pass": (
            "UNKNOWN identity stays UNKNOWN. There is no path that upgrades an "
            "unverifiable identity to a pass."),
    }
