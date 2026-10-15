"""
EXECUTION-NATIVE HOLD-PATH TELEMETRY -- measurement only.

WHY THIS EXISTS
===============
The aggregator mark (``o.price_usd``) and the position-sized executable sell
quote on a pumpfun bonding curve were measured to differ by a median of ~+25%
and by up to +92%, in BOTH directions, with a 43pp standard deviation. On short
holds the mark never updates at all. Every mark-derived number therefore lies:
``OBSERVED_MARK_MFE`` read +21% on a trade that settled at -44.7%.

That makes ``L25_MAX_HOLD`` -- a pure 60-second hold -- unmeasurable, because the
only record of what happened DURING the hold was the unreliable mark.

WHAT THIS RECORDS
=================
For each open position, a timestamped path of the position's own
position-sized SELL quote, taken with the SAME canonical quote implementation the
executor uses. That is the quantity the strategy is actually judged against.

    reference_return_pct = (net_sell_quote_now / net_sell_quote_at_entry - 1) * 100

UNITS -- THE POINT THAT MATTERS
===============================
The quote is denominated in the QUOTE ASSET (SOL), not USD. The MFE/MAE
*ratios* above are therefore UNIT-INDEPENDENT: they divide quote-asset by
quote-asset, so no SOL/USD rate can distort them. This is deliberate. The
position's ``sol_usd_reference`` is captured once at entry and goes stale over a
60-second hold, and a stale FX rate applied to an excursion would corrupt exactly
the number this module exists to measure.

Absolute USD figures are ALSO recorded, but they are a labelled convenience
derived from the entry-time reference rate, NOT the metric of record.

GROSS vs NET
============
For a SELL the canonical implementation returns ``output_amount_raw = net``::

    gross = amount * vQR // (vTR + amount)
    net   = gross - protocolFee(gross) - creatorFee(gross)

``net`` is what a seller receives, so ``net`` is the honest basis for "what
could I have got". GROSS is not exposed by the quote API and is recorded as
unavailable rather than reconstructed.

ISOLATION -- THE SAFETY ARGUMENT
================================
* This module imports NOTHING from the exit policy, the economics module, the
  guards, sizing or the decision engine. It cannot reach a decision.
* It is driven by its OWN asyncio task. It never runs inside ``monitor_once``,
  ``on_signal`` or ``_close_for_arm``.
* Every public method swallows its own exceptions. A telemetry failure cannot
  prevent an exit, extend a hold, alter an outcome, change entry eligibility,
  change sizing, or move a state transition.
* It writes to its OWN append-only file. It never mutates a position, an
  account, a quote or a trade record.
* If the whole feature is disabled, or the file cannot be opened, every entry
  point becomes a no-op and the runtime is byte-for-byte unchanged.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Sequence

# ---------------------------------------------------------------------------
# Observation status vocabulary
# ---------------------------------------------------------------------------
# Chosen so an invalid sample is RETAINED with its reason rather than dropped:
# an audit needs to see that a sample failed, not to see it missing.

VALID = "VALID"
INVALID_IDENTITY = "INVALID_IDENTITY"
STALE = "STALE"
QUOTE_UNAVAILABLE = "QUOTE_UNAVAILABLE"
PARTIAL_DATA = "PARTIAL_DATA"
UNSUPPORTED_SOURCE = "UNSUPPORTED_SOURCE"
#: NOT AN OBSERVATION AND NOT A FAILURE. The sampler declined to spend RPC:
#: either it could not establish the executable market's identity, or its own
#: lane budget was spent keeping capacity free for an exit. Recording these
#: distinctly matters -- folding them into ``QUOTE_UNAVAILABLE`` would report a
#: deliberate cost decision as a market-data failure and corrupt the very
#: coverage statistic this instrument exists to produce.
SKIPPED = "SKIPPED"
UNAVAILABLE_HISTORICAL_DATA = "UNAVAILABLE_HISTORICAL_DATA"

# ---------------------------------------------------------------------------
# MONITOR STATUS -- one small explicit state per monitoring observation
# ---------------------------------------------------------------------------
# Audit §9: every monitoring observation resolves into exactly one of these,
# so "is monitoring healthy?" is a counter rather than an inference from a pile
# of free-text reasons. They are derived FROM the status vocabulary above (or
# carried verbatim when the recorder set them), never replacing it.

MONITOR_OK = "MONITOR_OK"
MONITOR_SKIPPED_BUDGET = "MONITOR_SKIPPED_BUDGET"
MONITOR_QUOTE_UNAVAILABLE = "MONITOR_QUOTE_UNAVAILABLE"
MONITOR_IDENTITY_MISSING = "MONITOR_IDENTITY_MISSING"
MONITOR_IDENTITY_MISMATCH = "MONITOR_IDENTITY_MISMATCH"
MONITOR_RPC_ERROR = "MONITOR_RPC_ERROR"
MONITOR_STALE = "MONITOR_STALE"
#: Fallback for a skipped sample whose reason is neither budget nor identity
#: (a custom skip predicate). Deliberately generic rather than mislabelled.
MONITOR_SKIPPED = "MONITOR_SKIPPED"

#: The bounded post-exit research window (audit §10). After settlement the
#: sampler keeps recording for this many seconds, every row tagged
#: POST_EXIT_RESEARCH_ONLY. Research output ONLY: it can never feed an entry
#: feature or an exit decision, because the decision it describes already
#: happened before the first such row exists.
POST_EXIT_PHASE = "POST_EXIT"
POST_EXIT_RESEARCH_ONLY = "POST_EXIT_RESEARCH_ONLY"
POST_EXIT_WINDOW_S = 120.0


def monitor_status_for(
    status: str,
    *,
    identity_check_status: str | None = None,
    quote_identity: str | None = None,
    rpc_error: bool = False,
    skipped_budget: bool = False,
) -> str:
    """
    Map a raw observation status onto the explicit monitor state.

    The three states the audit insists must not be collapsed into one stay
    distinct: a TEMPORARY monitoring failure (RPC error / budget skip), a
    VERIFIED market mismatch, and a GENUINELY indeterminate market (missing
    identity).
    """
    if skipped_budget:
        return MONITOR_SKIPPED_BUDGET
    if rpc_error:
        return MONITOR_RPC_ERROR
    if status == VALID:
        return MONITOR_OK
    if status == STALE:
        return MONITOR_STALE
    if status in (INVALID_IDENTITY, SKIPPED):
        mismatch = (
            identity_check_status == "MISMATCH" or quote_identity == "FAIL")
        return MONITOR_IDENTITY_MISMATCH if mismatch else MONITOR_IDENTITY_MISSING
    if status == QUOTE_UNAVAILABLE:
        return MONITOR_QUOTE_UNAVAILABLE
    return f"MONITOR_{status}"


def monitor_status_for_observation(obs: "HoldPathObservation") -> str:
    """
    The monitor state for ONE recorded observation.

    Prefers the recorder's own explicit ``monitor_status``; falls back to a
    derivation from the status + reason so rows written BEFORE this field
    existed (including the audit's BEFORE dataset) classify identically.
    """
    recorded = getattr(obs, "monitor_status", "") or ""
    if recorded:
        return recorded
    status = obs.quote_status
    reason = (obs.unavailable_reason or "").lower()
    if status == SKIPPED:
        if "budget" in reason or "telemetry lane" in reason:
            return MONITOR_SKIPPED_BUDGET
        if "identity" in reason:
            if "'fail'" in reason or '"fail"' in reason:
                return MONITOR_IDENTITY_MISMATCH
            return MONITOR_IDENTITY_MISSING
        return MONITOR_SKIPPED
    if status == QUOTE_UNAVAILABLE and (
            "quote call raised" in reason or "local rpc cap" in reason
            or "rate limited" in reason):
        # A refused/raised RPC call is a CAPACITY fact, not a market fact. The
        # pre-hardening recorder folded many of these into identity skips; rows
        # that do carry the transport reason recover correctly here.
        return MONITOR_RPC_ERROR
    return monitor_status_for(
        status, identity_check_status=getattr(obs, "identity_check_status", None))

#: Statuses whose observation may enter a valid-path metric.
ELIGIBLE_STATUSES = frozenset({VALID})

#: Which market sources this module knows how to interpret. Anything else is
#: recorded as UNSUPPORTED_SOURCE instead of being silently priced.
SUPPORTED_MARKET_TYPES = frozenset({"PUMP_BONDING_CURVE", "PUMPSWAP"})

#: The quote asset this module knows how to value. A sell settling in anything
#: else is recorded as UNSUPPORTED_SOURCE rather than priced on a guess.
QUOTE_ASSET_SOL_LIKE = "So11111111111111111111111111111111111111112"


@dataclass(frozen=True)
class HoldPathConfig:
    """
    Telemetry tuning. Every threshold is explicit and documented.

    Defaults are deliberately CONSERVATIVE on cost and PERMISSIVE on validity:
    the instrument must not be able to interfere with quoting, so it samples
    slowly enough to be a rounding error against the 5s monitor tick, and it
    refuses to judge freshness strictly rather than loosely, because a stale
    sample recorded as fresh would corrupt the metric this exists to protect.
    """

    #: Target seconds between samples while a position is open.
    #:
    #: THIS IS A BACKGROUND INSTRUMENT AND IS PRICED LIKE ONE. It used to
    #: default to 1.0s, which for a single position is ~8 RPC calls per second
    #: -- the entire configured provider budget -- because each sample quotes
    #: both venues. It now samples on the monitor's own cadence by default, and
    #: in any case runs in the reserved telemetry lane, so it cannot consume
    #: capacity an open position needs to sell. Set it finer deliberately; do not
    #: let it become the default again.
    interval_s: float = 5.0
    #: Refuse to sample unless the position's executable market identity can be
    #: established. A sample whose identity cannot be proven is discarded by the
    #: validator anyway, so paying RPC for it is pure waste -- and that waste is
    #: what starved an open position's exit.
    #:
    #: DEFAULT OFF, so a direct caller that has no identity input keeps the
    #: previous behaviour. The PRODUCTION wiring turns it on, because that is the
    #: path where the pinned market is known and the waste actually happened.
    require_market_identity: bool = False
    #: Hard cap on how long the sampler runs for one position. A runaway task
    #: that outlives its position is a leak, not a measurement.
    max_duration_s: float = 180.0
    #: Reject a quote whose on-chain STATE is older than this. Bonding curves
    #: move on every trade, so this bounds how far in the past the curve state
    #: behind a sample may be. ``None`` disables the check and records the
    #: limitation rather than pretending freshness was established.
    max_state_age_ms: float | None = 5_000.0
    #: Reject a quote whose RESPONSE is older than this (cache/transport lag).
    max_quote_age_ms: float | None = 30_000.0
    #: Where observations are appended. One JSON object per line.
    sink_path: Path | None = None
    #: Master switch. False makes every entry point a no-op.
    enabled: bool = True

    @staticmethod
    def disabled() -> "HoldPathConfig":
        return HoldPathConfig(enabled=False, sink_path=None)


@dataclass
class HoldPathObservation:
    """
    ONE sample of the position's own executable reference.

    ``sell_quote_net_raw`` is the metric of record: raw quote-asset units
    (lamports for SOL), an INTEGER, so no float rounding can touch a token
    quantity. USD fields are labelled convenience values.
    """

    trade_id: str
    mint: str
    curve_or_pool_id: str | None
    observation_id: str
    #: Wall clock, for audit and cross-referencing with logs.
    observed_at_utc: float
    #: Monotonic-derived milliseconds since the sampler started. Durations use
    #: this, never a wall-clock difference, so an NTP step cannot invent or
    #: erase a holding period.
    elapsed_ms: float
    remaining_hold_ms: float | None
    remaining_token_amount_raw: int
    token_decimals: int
    quote_asset: str

    #: The SOL/USD rate used to render USD conveniences for THIS observation.
    #: Recorded per observation so a reader can tell which rate produced a USD
    #: figure. It is deliberately NOT used in any return metric: the reference
    #: returns divide quote-asset by quote-asset and are therefore immune to a
    #: stale FX rate, whereas a USD-denominated return over a 60s hold would not
    #: be.
    sol_usd_reference: float | None = None

    # -- the three distinct price concepts ------------------------------
    aggregator_mark_usd: float | None = None
    #: Marginal/spot price implied by the curve's own reserves.
    curve_spot_price_quote_asset: Decimal | None = None
    #: Position-sized SELL quote, NET of fees, raw quote-asset units.
    sell_quote_net_raw: int | None = None
    #: Position-sized SELL quote, GROSS. Not exposed by the quote API.
    sell_quote_gross_raw: int | None = None
    sell_quote_net_usd: float | None = None
    #: Average execution price per WHOLE token, net. NOT a marginal spot price
    #: and NOT the aggregator mark: it is what selling the whole holding costs.
    effective_liquidation_price_quote_asset: Decimal | None = None

    sell_quote_source: str | None = None
    quote_timestamp_utc: float | None = None
    quote_age_ms: float | None = None
    state_slot: int | None = None
    state_age_ms: float | None = None

    #: Exact reserve ratio behind the spot, so a quote can be re-derived.
    curve_reserve_numerator_raw: int | None = None
    curve_reserve_denominator_raw: int | None = None

    price_impact_pct: float | None = None
    fee_bps: int | None = None
    fees_quote_asset: float | None = None

    quote_status: str = VALID
    unavailable_reason: str | None = None
    identity_check_status: str = "UNVERIFIED"
    identity_evidence: str | None = None
    market_type: str | None = None

    # -- derived, per observation -----------------------------------------
    #: Return of the position-sized executable SELL reference against the ENTRY
    #: FILL, in percent. ``None`` when either leg is missing, never 0.0 -- a missing
    #: observation and a flat one are different facts.
    return_pct: float | None = None
    #: Which kind of number the reference behind ``return_pct`` is, from the
    #: five-value PriceProvenance vocabulary. A PATH sample is EXECUTABLE_QUOTE when
    #: its quote is valid, and UNAVAILABLE otherwise.
    reference_price_provenance: str = "UNAVAILABLE"
    #: Whether this observation may be used in a return statistic. An observation
    #: with no valid quote is recorded but is NOT an implicit zero.
    usable_for_return: bool = False
    #: Why this observation is or is not usable. Always present when it is not.
    data_quality_reason: str | None = None
    #: The explicit monitoring state for this tick (audit §9): one of
    #: MONITOR_OK / MONITOR_SKIPPED_BUDGET / MONITOR_QUOTE_UNAVAILABLE /
    #: MONITOR_IDENTITY_MISSING / MONITOR_IDENTITY_MISMATCH / MONITOR_RPC_ERROR /
    #: MONITOR_STALE. Empty on rows written before the field existed; the
    #: metrics layer derives the same value for those.
    monitor_status: str = ""
    #: POST_EXIT_RESEARCH_ONLY on rows sampled inside the bounded post-exit
    #: window; empty otherwise. Research output, never a decision input.
    research_tag: str = ""

    #: Compact input snapshot: enough to reproduce the quote offline.
    raw_snapshot_reference: dict[str, Any] = field(default_factory=dict)

    # -- identity of the policy, and of the return basis ------------------
    #: Which strategy profile the position was opened under, so a hold path can be
    #: partitioned by exit policy without joining back to the trade.
    strategy_profile: str | None = None
    #: ``ENTRY`` for the single observation taken at the fill, ``PATH`` for every
    #: observation after it. A hold path that does not include its own entry cannot
    #: produce a return series, which is why the entry is sampled explicitly rather
    #: than reconstructed from the first PATH sample.
    phase: str = "PATH"
    #: The entry FILL PRICE for ONE WHOLE TOKEN, in USD. A per-token price.
    entry_fill_usd: float | None = None
    #: WHAT THE RETURN IS ACTUALLY MEASURED AGAINST: the entry fill applied to
    #: the whole position, in USD.
    #:
    #: Both are recorded because they are different quantities. Comparing a
    #: per-token entry price against a total SELL proceeds figure reports a
    #: ~500,000 % return on a position that actually moved about -3 %, which is
    #: what the first recorded hold paths reported. A return is only defined
    #: between like quantities, so the basis is stated on the row.
    entry_notional_usd: float | None = None
    #: WHOLE TOKENS HELD, the bridge between the two figures above.
    position_tokens: float | None = None
    entry_price_provenance: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if isinstance(value, Decimal):
                # Decimal is not JSON-serialisable; keep it EXACT as a string
                # rather than degrading it through float.
                out[key] = str(value)
            elif isinstance(value, Path):
                out[key] = str(value)
            else:
                out[key] = value
        return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_observation(
    *,
    quote: Any,
    mint: str,
    expected_venue_id: str | None,
    expected_token_amount_raw: int,
    config: HoldPathConfig,
    observed_at: float | None = None,
) -> tuple[str, str, str | None]:
    """
    Classify a quote BEFORE it is allowed to influence any metric.

    Returns ``(status, identity_status, reason)``. Never raises: a validation
    failure is a DATA QUALITY fact to record, not an error to propagate into the
    runtime.

    Ordering matters. Identity is checked before freshness, because a quote for
    the wrong market is wrong regardless of how new it is, and reporting it as
    merely STALE would misdescribe the failure.
    """
    if quote is None:
        return QUOTE_UNAVAILABLE, "UNVERIFIED", "no quote object was returned"

    if not getattr(quote, "available", False):
        return (
            QUOTE_UNAVAILABLE, "UNVERIFIED",
            f"quote unavailable: {getattr(quote, 'missing_reason', None)}",
        )

    market_type = getattr(quote, "market_type", None)
    if market_type is not None and market_type not in SUPPORTED_MARKET_TYPES:
        # Refuse to price a market shape this module cannot interpret, rather
        # than treating an unfamiliar venue as if it were a curve.
        return (
            UNSUPPORTED_SOURCE, "UNVERIFIED",
            f"market_type {market_type!r} is not an interpretable source here",
        )

    # -- identity ------------------------------------------------------
    quote_mint = getattr(quote, "token_mint", None)
    if quote_mint and quote_mint != mint:
        return (
            INVALID_IDENTITY, "MISMATCH",
            f"quote is for token {quote_mint}, position holds {mint}",
        )

    quote_venue = getattr(quote, "venue_id", None)
    if expected_venue_id and quote_venue and quote_venue != expected_venue_id:
        # NOTE: deliberately NOT compared against the ENTRY venue. A curve that
        # graduates is legitimately exited through the pool it became, so a
        # hard entry-venue equality here would reject valid samples. The
        # internal consistency below is the check that must hold.
        return (
            INVALID_IDENTITY, "MISMATCH",
            f"quote venue {quote_venue} differs from the sampled market "
            f"{expected_venue_id}",
        )

    # -- the quote must be for the position we are measuring -----------
    input_raw = getattr(quote, "input_amount_raw", None)
    if input_raw is not None and int(input_raw) != int(expected_token_amount_raw):
        return (
            INVALID_IDENTITY, "MISMATCH",
            f"quote is sized for {input_raw} raw tokens, position holds "
            f"{expected_token_amount_raw}",
        )

    # The quote must be PAID IN the expected quote asset. A sell denominated in
    # something else is a different trade whose proceeds are not comparable.
    expected_quote_asset = getattr(quote, "quote_mint", None)
    if expected_quote_asset and expected_quote_asset != QUOTE_ASSET_SOL_LIKE:
        # Not an identity failure by itself -- a venue may legitimately settle
        # in USDC -- but it is NOT an interpretable reference for this module,
        # so it is refused rather than priced.
        return (
            UNSUPPORTED_SOURCE, "PASS",
            f"quote settles in {expected_quote_asset}, not a recognised "
            f"quote asset for an execution reference",
        )

    if getattr(quote, "side", "SELL") != "SELL":
        # A buy quote is the inverse trade. Accepting one would invert the very
        # quantity being measured.
        return (
            INVALID_IDENTITY, "MISMATCH",
            f"quote side is {getattr(quote, 'side')!r}, not SELL",
        )

    identity = getattr(quote, "identity", "UNKNOWN")
    if identity != "PASS":
        return (
            INVALID_IDENTITY, "UNPROVEN",
            f"same-market identity not proven (identity={identity!r})",
        )

    # -- numbers must be real -----------------------------------------
    output_raw = getattr(quote, "output_amount_raw", None)
    if output_raw is None or int(output_raw) <= 0:
        return (
            PARTIAL_DATA, "PASS",
            f"sell quote produced no positive proceeds ({output_raw!r})",
        )
    price = getattr(quote, "effective_price", None)
    # Order matters: ``Decimal("nan") > 0`` raises InvalidOperation rather than
    # returning False, so finiteness is checked BEFORE any comparison. The
    # numeric coercion is also defensive because a provider-shaped double could
    # carry a non-numeric price.
    try:
        price_f = float(price) if price is not None else float("nan")
    except (TypeError, ValueError):
        price_f = float("nan")
    if not math.isfinite(price_f) or price_f <= 0:
        return PARTIAL_DATA, "PASS", f"sell quote price unusable ({price!r})"

    # -- causality -----------------------------------------------------
    #
    # A QUOTE CANNOT BE STAMPED IN THE FUTURE. A provider clock that runs ahead,
    # or a bad field, would otherwise hand the sampler a price it had no way of
    # having seen yet, and every downstream statistic would inherit a lookahead
    # that is invisible in the output. One second of slack absorbs ordinary clock
    # skew between the venue and this process without admitting a real lookahead.
    if observed_at is not None:
        for field_name in ("quote_timestamp", "received_at", "quoted_at"):
            stamp = getattr(quote, field_name, None)
            if stamp is None:
                continue
            try:
                stamp_f = float(stamp)
            except (TypeError, ValueError):
                return (
                    INVALID_IDENTITY, "PASS",
                    f"{field_name} is not a number ({stamp!r})",
                )
            if stamp_f > float(observed_at) + 1.0:
                return (
                    STALE, "PASS",
                    f"{field_name} {stamp_f:.3f} is after the observation time "
                    f"{float(observed_at):.3f}; a quote cannot be from the future",
                )

    # -- freshness -----------------------------------------------------
    if config.max_state_age_ms is not None:
        state_age = getattr(quote, "state_age_ms", None)
        if state_age is None:
            # Freshness cannot be ESTABLISHED. That is a limitation to record,
            # not a freshness to assume.
            return (
                STALE, "PASS",
                "quote carried no state_age_ms, so on-chain state freshness "
                "cannot be established",
            )
        if float(state_age) > config.max_state_age_ms:
            return (
                STALE, "PASS",
                f"curve state is {float(state_age):.0f}ms old, limit is "
                f"{config.max_state_age_ms:.0f}ms",
            )

    if config.max_quote_age_ms is not None:
        quote_age = getattr(quote, "quote_age_ms", None)
        if quote_age is not None and float(quote_age) > config.max_quote_age_ms:
            return (
                STALE, "PASS",
                f"quote response is {float(quote_age):.0f}ms old, limit is "
                f"{config.max_quote_age_ms:.0f}ms",
            )

    return VALID, "PASS", None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _pct(now: Decimal | int, base: Decimal | int) -> float | None:
    if base == 0:
        return None
    return (float(Decimal(now) / Decimal(base) - 1) * 100.0)


def _post_exit_research_block(
    post_rows: Sequence[HoldPathObservation],
    position_rows: Sequence[HoldPathObservation],
) -> dict[str, Any]:
    """
    Research metrics over the bounded post-exit window ONLY (audit §10).

    Answers "did the token pump after we sold" and "what was the rebound",
    against two explicitly named bases: the last VALID hold-path reference (the
    price the position could actually have exited at) and the entry reference.
    Purely descriptive; nothing here feeds a decision.
    """
    block: dict[str, Any] = {
        "POST_EXIT_RESEARCH_ONLY": bool(post_rows),
        "post_exit_observation_count": len(post_rows),
    }
    if not post_rows:
        return block
    nets = [
        (o.elapsed_ms, int(o.sell_quote_net_raw))
        for o in post_rows if o.sell_quote_net_raw
    ]
    if not nets:
        return block
    best_elapsed, best = max(nets, key=lambda pair: pair[1])
    block["POST_EXIT_MAX_NET_RAW"] = best
    block["POST_EXIT_TIME_TO_MAX_MS"] = best_elapsed
    path_valid = [
        int(o.sell_quote_net_raw) for o in position_rows
        if o.quote_status in ELIGIBLE_STATUSES and o.sell_quote_net_raw
    ]
    if path_valid:
        block["POST_EXIT_MAX_VS_LAST_HOLD_REFERENCE_PCT"] = (
            (best / path_valid[-1] - 1.0) * 100.0)
        block["POST_EXIT_MAX_VS_ENTRY_REFERENCE_PCT"] = (
            (best / path_valid[0] - 1.0) * 100.0)
    block["post_exit_basis"] = (
        "position-sized NET sell quote (raw quote-asset units); research only, "
        "never a decision input and never part of a position statistic")
    return block


def compute_hold_metrics(
    observations: Sequence[HoldPathObservation],
    *,
    expected_window_ms: float | None = None,
    include_entry_observation: bool = True,
) -> dict[str, Any]:
    """
    Execution-reference MFE / MAE and data-quality indicators.

    POLICY, stated explicitly so the number is interpretable:

    * The ENTRY observation IS included in MFE and MAE. Its reference return is
      0 by construction, so inclusion guarantees ``MFE >= 0 >= MAE`` and keeps
      every trade comparable under one rule. ``include_entry_observation=False``
      switches it off and the fact is recorded in the output.
    * The entry reference is the position-sized NET SELL quote at entry -- an
      ESTIMATED LIQUIDATION VALUE, not the buy cost. Returns measured against it
      answer "did this holding become more or less valuable", which is the
      question MFE/MAE is for. It is NOT the PnL the trade eventually booked.
    * A metric over fewer than two eligible observations is reported
      UNAVAILABLE. MFE/MAE is NEVER reported as zero for want of data.
    """
    ordered = sorted(observations, key=lambda o: o.elapsed_ms)
    # POST-EXIT rows are a SEPARATE RESEARCH WINDOW (audit §10): they never
    # enter a position statistic -- MFE/MAE, exit return, coverage, gaps --
    # because the decision they could describe already happened before the
    # first such row existed.
    post_exit_rows = [
        o for o in ordered if o.phase == POST_EXIT_PHASE]
    ordered = [
        o for o in ordered if o.phase != POST_EXIT_PHASE]
    eligible = [
        o for o in ordered
        if o.quote_status in ELIGIBLE_STATUSES and o.sell_quote_net_raw
    ]
    # A SKIPPED sample is neither eligible nor a failed observation. It is
    # reported in its own bucket so a throttled sampler is distinguishable from
    # a market that could not be read -- otherwise "we chose not to ask" would
    # be recorded as "we asked and got nothing", which is precisely the
    # misdiagnosis this module exists to prevent.
    skipped = [
        {"observation_id": o.observation_id, "status": o.quote_status,
         "reason": o.unavailable_reason}
        for o in ordered if o.quote_status == SKIPPED
    ]
    rejected = [
        {"observation_id": o.observation_id, "status": o.quote_status,
         "reason": o.unavailable_reason}
        for o in ordered
        if o not in eligible and o.quote_status != SKIPPED
    ]

    # THE MONITORING HEALTH COUNTERS (audit §9): every tick resolves into
    # exactly one explicit state, so usable/total is a number, not an inference.
    monitor_counts: dict[str, int] = {}
    for o in ordered:
        key = monitor_status_for_observation(o)
        monitor_counts[key] = monitor_counts.get(key, 0) + 1
    usable = monitor_counts.get(MONITOR_OK, 0)

    result: dict[str, Any] = {
        "observation_count_total": len(ordered),
        "observation_count_eligible": len(eligible),
        "observation_count_skipped": len(skipped),
        "entry_observation_included": include_entry_observation,
        "skipped_observations": skipped,
        "rejected_observations": rejected,
        # usable / total over monitoring ticks (post-exit rows excluded).
        "total_monitor_ticks": len(ordered),
        "usable_observations": usable,
        "usable_observations_pct": (
            round(100.0 * usable / len(ordered), 2) if ordered else None),
        "monitor_status_counts": monitor_counts,
        "metrics_status": VALID,
    }
    result.update(_post_exit_research_block(post_exit_rows, ordered))

    if len(eligible) < 2:
        result["metrics_status"] = PARTIAL_DATA
        result["unavailable_reason"] = (
            f"only {len(eligible)} eligible observation(s); MFE/MAE need at "
            f"least 2 and are reported UNAVAILABLE rather than as zero"
        )
        for key in (
            "EXEC_REFERENCE_MFE_PCT", "EXEC_REFERENCE_MAE_PCT",
            "EXEC_REFERENCE_EXIT_RETURN_PCT", "EXEC_REFERENCE_TIME_TO_MFE_MS",
            "EXEC_REFERENCE_TIME_TO_MAE_MS", "EXEC_REFERENCE_COVERAGE_PCT",
            "EXEC_REFERENCE_MAX_SAMPLING_GAP_MS",
        ):
            result[key] = None
        return result

    entry = eligible[0]
    base = Decimal(entry.sell_quote_net_raw)  # type: ignore[arg-type]
    result["entry_reference_sell_quote_net_raw"] = int(base)
    result["entry_reference_basis"] = (
        "position-sized NET sell quote at entry (estimated liquidation value, "
        "NOT the buy cost)"
    )

    series: list[tuple[HoldPathObservation, float]] = []
    for obs in eligible:
        ret = _pct(Decimal(obs.sell_quote_net_raw), base)  # type: ignore[arg-type]
        if ret is not None:
            series.append((obs, ret))
    if len(series) < 2:
        result["metrics_status"] = PARTIAL_DATA
        result["unavailable_reason"] = "fewer than 2 computable reference returns"
        return result

    if not include_entry_observation:
        series = series[1:] or series

    peak = max(series, key=lambda pair: pair[1])
    trough = min(series, key=lambda pair: pair[1])
    last = series[-1]

    result["EXEC_REFERENCE_MFE_PCT"] = peak[1]
    result["EXEC_REFERENCE_MAE_PCT"] = trough[1]
    result["EXEC_REFERENCE_EXIT_RETURN_PCT"] = last[1]
    result["EXEC_REFERENCE_TIME_TO_MFE_MS"] = peak[0].elapsed_ms
    result["EXEC_REFERENCE_TIME_TO_MAE_MS"] = trough[0].elapsed_ms
    result["EXEC_REFERENCE_EXIT_ELAPSED_MS"] = last[0].elapsed_ms

    # -- coverage and sampling quality --------------------------------
    if expected_window_ms and expected_window_ms > 0:
        result["expected_window_ms"] = expected_window_ms
        result["EXEC_REFERENCE_COVERAGE_PCT"] = min(
            100.0, (last[0].elapsed_ms / expected_window_ms) * 100.0
        )
    else:
        result["expected_window_ms"] = None
        result["EXEC_REFERENCE_COVERAGE_PCT"] = None

    # ``series`` holds (observation, return) pairs, so the timing has to come
    # off element [0].
    gaps = [
        later[0].elapsed_ms - earlier[0].elapsed_ms
        for earlier, later in zip(series, series[1:])
    ]
    result["EXEC_REFERENCE_MAX_SAMPLING_GAP_MS"] = max(gaps) if gaps else None

    # -- mark versus reference, per observation -----------------------
    mark_gaps: list[float] = []
    for obs in eligible:
        if obs.curve_spot_price_quote_asset is None or not obs.curve_spot_price_quote_asset:
            continue
        spot = Decimal(obs.curve_spot_price_quote_asset)
        if spot <= 0 or entry.sol_usd_reference in (None, 0):
            continue
        # Both sides converted with the SAME rate, so the ratio is the pure
        # price discrepancy and no FX noise enters it.
        mark_native = Decimal(str(obs.aggregator_mark_usd)) / Decimal(
            str(entry.sol_usd_reference)
        )
        gap = (float(mark_native / spot - 1) * 100.0)
        mark_gaps.append(gap)
    if mark_gaps:
        result["MARK_VS_REFERENCE_GAP_PCT_MEDIAN"] = sorted(mark_gaps)[len(mark_gaps) // 2]
        result["MARK_VS_REFERENCE_GAP_PCT_MIN"] = min(mark_gaps)
        result["MARK_VS_REFERENCE_GAP_PCT_MAX"] = max(mark_gaps)
        result["mark_gap_formula"] = (
            "(aggregator_mark_usd / sol_usd_reference) / curve_spot_quote_asset - 1"
        )
    else:
        result["MARK_VS_REFERENCE_GAP_PCT_MEDIAN"] = None
        result["mark_gap_formula"] = (
            "(aggregator_mark_usd / sol_usd_reference) / curve_spot_quote_asset - 1"
        )

    # -- frozen aggregator marks --------------------------------------
    marks = [o.aggregator_mark_usd for o in eligible if o.aggregator_mark_usd]
    frozen = 0
    for a, b in zip(marks, marks[1:]):
        if a == b:
            frozen += 1
    result["frozen_mark_intervals"] = frozen
    result["mark_observation_count"] = len(marks)

    return result


# ---------------------------------------------------------------------------
# The recorder
# ---------------------------------------------------------------------------

class _NullLaneGuard:
    """
    The no-lane fallback, written out rather than imported.

    This module's import allowlist is deliberately narrow and does not include
    ``contextlib``. Rather than widen a safety boundary to save three lines, the
    no-op guard is spelled out. If no guard is injected, telemetry simply runs
    outside a named lane -- which is the correct degradation for a module that
    must not depend on the rate limiter to work at all.
    """

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: Any) -> bool:
        return False


def _never_a_skip(_exc: BaseException) -> bool:
    """
    Default skip predicate: nothing is a budget decision.

    Without this, ANY exception is treated as a failed quote -- which is the safe
    default, because a missed observation is recoverable and a mislabelled skip
    is not.
    """
    return False


class HoldPathRecorder:
    """
    Samples one position's executable reference on its own schedule.

    DESIGN CONTRACT
    ---------------
    * Own asyncio task. Never awaited by the trading path.
    * ``sample_once`` is the only method that touches a quote, and it swallows
      everything.
    * Every RPC call is charged to the TELEMETRY LANE, which is capped below the
      total provider budget. When that lane is spent the sample is SKIPPED, so
      this instrument cannot starve the exit path of the position it measures.
    * The ENTRY sample is exempt from skipping: one sample is not load, and a
      hold path without its own entry is not a hold path.
    * The sink is opened per append and closed immediately, so a telemetry
      failure cannot leave a handle that interferes with anything.
    * No reference to the engine's decision objects is retained beyond the
      plain values captured at construction.
    """

    def __init__(
        self,
        *,
        trade_id: str,
        mint: str,
        curve_or_pool_id: str | None,
        token_decimals: int,
        remaining_token_amount_raw: int,
        quote_asset: str,
        entry_fill_usd: float | None = None,
        entry_notional_usd: float | None = None,
        entry_price_provenance: str | None = None,
        strategy_profile: str | None = None,
        #: The pinned executable market for this position. Passed into the quote
        #: so same-market identity can be PROVEN rather than assumed. It is the
        #: address the position was actually opened on, never a guess, and it is
        #: the ONLY identity input: a live aggregator pair lookup can no longer
        #: override it (audit §2 -- never silently replace the authoritative
        #: market with a newly discovered one).
        observed_pair_address: str | None = None,
        #: Context manager charging this sampler's RPC calls to the reserved
        #: telemetry lane. INJECTED rather than imported so this module keeps its
        #: documented isolation: it must not be able to reach the rate limiter or
        #: the execution path even by accident.
        rpc_lane_guard: Callable[[], Any] | None = None,
        #: Predicate answering "was this refusal OUR budget decision?". Injected
        #: for the same reason as the lane guard: this module must not know the
        #: rate limiter's vocabulary, and string-matching a reason name would make
        #: a rename silently reclassify a deliberate skip as a market failure.
        skip_predicate: Callable[[BaseException], bool] | None = None,
        quote_fn: Callable[..., Any],
        config: HoldPathConfig,
        mark_fn: Callable[[str], Any] | None = None,
        expected_venue_id: str | None = None,
        expected_window_ms: float | None = None,
        sol_usd_reference: float | None = None,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self.trade_id = trade_id
        self.mint = mint
        self.curve_or_pool_id = curve_or_pool_id
        self.token_decimals = int(token_decimals)
        self.remaining_token_amount_raw = int(remaining_token_amount_raw)
        self.quote_asset = quote_asset
        self.entry_fill_usd = entry_fill_usd
        self.entry_price_provenance = entry_price_provenance
        self.observed_pair_address = observed_pair_address
        self.rpc_lane_guard = rpc_lane_guard
        self.skip_predicate = skip_predicate or _never_a_skip
        self.strategy_profile = strategy_profile
        self._phase = "ENTRY"
        #: Whole tokens held. Recorded explicitly so the entry notional is not a
        #: reconstruction the reader has to trust.
        self.position_tokens = (
            float(self.remaining_token_amount_raw) / (10 ** self.token_decimals))
        # THE RETURN BASIS IS THE WHOLE-POSITION OUTLAY.
        #
        # ``entry_fill_usd`` is a PER-TOKEN price. The sampler observes TOTAL SELL
        # proceeds, and a ratio between a total and a unit price is not a return --
        # it reported +500,259 % on positions that moved about -3 %. If only a
        # per-token fill was supplied it is scaled by the tokens held, rather than
        # silently used as though it were the notional.
        if entry_notional_usd is not None and entry_notional_usd > 0:
            self.entry_notional_usd = float(entry_notional_usd)
        elif entry_fill_usd and entry_fill_usd > 0:
            self.entry_notional_usd = entry_fill_usd * self.position_tokens
        else:
            self.entry_notional_usd = None
        self.expected_venue_id = expected_venue_id
        self.expected_window_ms = expected_window_ms
        self.sol_usd_reference = sol_usd_reference
        self.config = config

        self._quote_fn = quote_fn
        self._mark_fn = mark_fn
        self._clock = clock
        self._monotonic = monotonic
        self._sleep = sleep

        self.observations: list[HoldPathObservation] = []
        self._done = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._start_monotonic: float | None = None
        self._sequence = 0
        #: Monotonic instant the bounded post-exit research window closes, or
        #: ``None`` while the position is (still) open.
        self._post_exit_deadline: float | None = None
        #: Set if the sampler itself failed. Reported, never raised.
        self.telemetry_error: str | None = None
        #: Samples deliberately NOT taken, to keep execution capacity free or
        #: because identity could not be established. Reported so a thin dataset
        #: is explainable rather than merely small.
        self.skipped_samples: int = 0

    # -- lifecycle -----------------------------------------------------

    def mark_post_exit(self) -> bool:
        """
        Open the bounded post-exit research window (audit §10). Never raises.

        Called by the engine AFTER settlement, non-blocking: the trading path
        never waits on this. The sampler continues for ``POST_EXIT_WINDOW_S``
        seconds in the ``POST_EXIT`` phase, every row tagged
        ``POST_EXIT_RESEARCH_ONLY`` -- did the token pump after the exit, how far
        did it draw down, what was the rebound, what is the opportunity cost.

        RETURNS False when telemetry is disabled or the sampler has already
        finished; the caller may then fall back to :meth:`aclose`.

        Structurally cannot affect a decision: the position is already settled
        before this runs, post-exit rows are excluded from every position
        statistic in :func:`compute_hold_metrics`, and nothing here can reach
        the trading path.
        """
        try:
            if not self.config.enabled or self.config.sink_path is None:
                return False
            if self._done.is_set():
                return False
            if self._task is not None and self._task.done():
                return False
            # ``_task is None`` (a one-shot caller) is safe: marking only sets
            # the phase and the deadline that the sampler's own loop -- if it
            # ever runs -- is bounded by.
            if self._post_exit_deadline is None:
                self._phase = POST_EXIT_PHASE
                self._post_exit_deadline = self._monotonic() + POST_EXIT_WINDOW_S
            return True
        except Exception:  # noqa: BLE001 - telemetry never breaks a caller
            return False

    async def sample_entry(self) -> HoldPathObservation | None:
        """
        Take the single ENTRY observation.

        Separate from ``start`` so a caller can guarantee the hold path CONTAINS its
        entry. A path that begins at the first post-fill sample cannot produce a
        return series, and reconstructing the entry afterwards would mean quoting a
        price nobody observed.
        """
        self._phase = "ENTRY"
        return await self.sample_once()

    def start(self, *, with_entry_observation: bool = False) \
            -> asyncio.Task | None:
        """Launch the sampler. Returns None when telemetry is disabled."""
        if not self.config.enabled or self.config.sink_path is None:
            return None
        try:
            if with_entry_observation:
                # The entry sample runs FIRST and INSIDE the sampler task, so the
                # hold path is guaranteed to contain the fill it is measured
                # against, and so the first PATH sample cannot overtake it.
                self._task = asyncio.ensure_future(self._run(with_entry=True))
            else:
                self._task = asyncio.ensure_future(self._run())
        except Exception as exc:  # noqa: BLE001 - telemetry never breaks a caller
            self.telemetry_error = f"could not start sampler: {type(exc).__name__}"
            return None
        return self._task

    async def aclose(self, timeout_s: float = 2.0) -> None:
        """
        Stop the sampler and flush. Never raises.

        Called from the engine on position close. A failure here must not delay
        or prevent the exit, so everything is contained.
        """
        self._done.set()
        task = self._task
        if task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
            # A sampler that will not stop is cancelled outright rather than
            # left running against a position that no longer exists.
            try:
                task.cancel()
            except Exception:  # noqa: BLE001
                pass

    # -- sampling ------------------------------------------------------

    async def _run(self, *, with_entry: bool = False) -> None:
        self._start_monotonic = self._monotonic()
        deadline = self.config.max_duration_s
        try:
            if with_entry:
                await self.sample_entry()
            await self.sample_once()
            while not self._done.is_set():
                try:
                    await self._sleep(self.config.interval_s)
                except asyncio.CancelledError:
                    return
                if self._done.is_set():
                    break
                now_mono = self._monotonic()
                elapsed = now_mono - (self._start_monotonic or 0.0)
                if self._phase == POST_EXIT_PHASE:
                    # BOUNDED: the post-exit window closes at the deadline set
                    # when it opened (mark time + POST_EXIT_WINDOW_S), regardless
                    # of the hold-path duration budget.
                    if now_mono >= (self._post_exit_deadline or 0.0):
                        break
                elif elapsed > deadline:
                    break
                await self.sample_once()
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001 - telemetry never breaks a caller
            self.telemetry_error = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            try:
                self._finalise()
            except Exception as exc:  # noqa: BLE001
                self.telemetry_error = f"finalise failed: {type(exc).__name__}"

    def _finalise(self) -> None:
        """Append the summary row. Never raises."""
        try:
            metrics = self.summary()
            self._append({"record_type": "SUMMARY", "trade_id": self.trade_id,
                          **metrics})
        except Exception:  # noqa: BLE001
            pass

    async def sample_once(self) -> HoldPathObservation | None:
        """
        Take ONE observation. Returns it, or None if it could not be built.

        Every failure mode is recorded on the observation rather than raised, so
        a broken quote shows up as ``QUOTE_UNAVAILABLE`` in the audit trail.

        THE SAMPLE RUNS IN THE TELEMETRY LANE, AND MAY BE SKIPPED.

        A background sampler that shares a rate-limit budget with the exit path
        can starve the very position it is measuring. So every RPC call made here
        is charged to the telemetry lane, which is capped below the total budget,
        and when that lane is spent the sample is SKIPPED -- recorded as
        ``SKIPPED``, not as a failed quote. The exit path's reserved capacity is
        therefore structurally unreachable from here.
        """
        try:
            now_wall = self._clock()
            # A one-shot caller (for example a single entry observation) never
            # goes through ``_run``, so the monotonic baseline is established
            # lazily here. Without this every direct sample reports elapsed 0
            # and the whole hold path collapses onto one instant.
            if self._start_monotonic is None:
                self._start_monotonic = self._monotonic()
            elapsed_ms = (self._monotonic() - self._start_monotonic) * 1000.0
            self._sequence += 1
            obs_id = f"{self.trade_id}#{self._sequence:05d}"

            mark_usd: float | None = None
            if self._mark_fn is not None:
                try:
                    raw_mark = self._mark_fn(self.mint)
                    mark_usd = float(raw_mark) if raw_mark is not None else None
                    if mark_usd is not None and not math.isfinite(mark_usd):
                        mark_usd = None
                except Exception:  # noqa: BLE001
                    mark_usd = None

            quote = None
            quote_error: str | None = None
            # THE OBSERVED MARKET IDENTITY IS PART OF THE INPUT, NOT AN
            # AFTERTHOUGHT. It used to be hard-coded ``None``, which made
            # ``_identity_for`` return UNKNOWN for EVERY sample: each one paid
            # full RPC cost and was then discarded by this module's own identity
            # gate. The pinned market IS known on the position, so pass the
            # market the position actually holds and let identity be proven or
            # refused on its merits.
            # THE POSITION'S OWN PINNED MARKET IS THE ONLY IDENTITY INPUT.
            #
            # A live aggregator pair lookup used to take precedence here, and
            # with it the verified entry identity could be lost on a tick where
            # the aggregator returned no pair or a different pool -- the false
            # UNKNOWN this hardening pass exists to remove. The pin persisted on
            # the position is authoritative; when it is absent, identity fails
            # closed to UNKNOWN rather than being resolved against a newly
            # discovered market (audit §2).
            observed_pair = self.observed_pair_address or None
            try:
                # THE LANE GUARD IS INJECTED. This module must not import the
                # rate limiter, so the engine supplies a context manager that
                # charges these RPC calls to the reserved telemetry lane.
                guard = self.rpc_lane_guard
                with (guard() if guard is not None else _NullLaneGuard()):
                    quote = await self._quote_fn(
                        mint=self.mint,
                        token_amount_raw=self.remaining_token_amount_raw,
                        token_decimals=self.token_decimals,
                        observed_pair_address=observed_pair,
                    )
            except Exception as exc:  # noqa: BLE001
                quote_error = f"{type(exc).__name__}: {exc}"[:200]
                # A SPENT TELEMETRY BUDGET IS A SKIP, NOT A FAILED OBSERVATION.
                # The ENTRY sample is exempt: one sample is not load, and a path
                # that cannot contain its own entry is not a hold path.
                if self.skip_predicate(exc) and self._phase != "ENTRY":
                    return self._record_skip(now_wall, quote_error)

            if (self.config.require_market_identity
                    and self._phase != "ENTRY"
                    and getattr(quote, "available", False)
                    and getattr(quote, "identity", "UNKNOWN") != "PASS"):
                # Only an ACTUAL quote can fail identity. A raised RPC call or
                # an unavailable quote used to reach this gate (its identity is
                # trivially UNKNOWN) and was recorded as an identity skip --
                # misclassifying a capacity failure as a market-identity fact.
                # Those now fall through to validate_observation and land as
                # QUOTE_UNAVAILABLE / RPC errors. Spending the budget on a
                # genuinely unproven sample is still declined; record it so the
                # gap is visible.
                _qident = getattr(quote, "identity", "UNKNOWN")
                return self._record_skip(
                    now_wall,
                    "executable-market identity could not be established, so "
                    f"the sample would be discarded as unusable "
                    f"(identity={_qident!r}); "
                    f"no exit capacity spent on it",
                    quote=quote,
                    monitor_status=(
                        MONITOR_IDENTITY_MISMATCH if _qident == "FAIL"
                        else MONITOR_IDENTITY_MISSING),
                )

            status, identity, reason = validate_observation(
                quote=quote, mint=self.mint,
                expected_venue_id=self.curve_or_pool_id,
                expected_token_amount_raw=self.remaining_token_amount_raw,
                config=self.config,
                observed_at=now_wall,
            )
            if quote_error and status == QUOTE_UNAVAILABLE:
                reason = f"quote call raised: {quote_error}"

            monitor_status = monitor_status_for(
                status,
                identity_check_status=identity,
                quote_identity=(
                    getattr(quote, "identity", None) if quote else None),
                rpc_error=bool(quote_error),
            )

            net_raw = getattr(quote, "output_amount_raw", None) if quote else None
            spot = None
            reserve_num = reserve_den = None
            native = getattr(quote, "execution_native_reference", None) if quote else None
            if native is not None:
                ref = getattr(native, "reference_price", None)
                if ref is not None and float(ref) > 0:
                    spot = Decimal(str(ref))
                reserve_num = getattr(native, "numerator_raw", None)
                reserve_den = getattr(native, "denominator_raw", None)

            effective = getattr(quote, "effective_price", None) if quote else None
            net_usd = None
            if net_raw is not None and self.sol_usd_reference:
                net_usd = float(Decimal(int(net_raw)) / Decimal(10 ** 9)
                                * Decimal(str(self.sol_usd_reference)))

            remaining_ms = None
            if self.expected_window_ms is not None:
                remaining_ms = max(0.0, self.expected_window_ms - elapsed_ms)

            # THE RETURN IS AGAINST THE ENTRY FILL, and it exists only when both
            # legs are real. A missing observation is ``None`` with a reason, never
            # 0.0, because "we could not see it" and "it did not move" are different
            # facts and a hold-path study must not be able to confuse them.
            return_pct = None
            provenance = "UNAVAILABLE"
            usable = False
            dq_reason = None
            if status != VALID:
                dq_reason = reason or f"observation status {status}"
            if status == VALID:
                # ONLY A VALID QUOTE MAY PRODUCE A NUMBER.
                #
                # This used to compute the return whenever both legs were merely
                # PRESENT, so a refused quote still produced a figure: a zero
                # output became -100 %, and a one-hour-old quote was accepted at
                # face value. A study reading those numbers would see positions
                # going to zero and prices that never moved, which is the exact
                # failure this instrument exists to prevent. An unusable
                # observation carries no return at all.
                # BOTH LEGS MUST BE THE SAME KIND OF QUANTITY.
                #
                # net_usd is TOTAL proceeds for the whole position; a per-token
                # entry fill price is not comparable to it. The basis is the entry
                # fill applied to the tokens held. Returning a ratio between a
                # total and a unit price is how a -3 % position reported
                # +500,259 %.
                if net_usd is None:
                    dq_reason = (
                        "quote is valid but carries no USD conversion, so no "
                        "return can be computed")
                elif not self.entry_notional_usd or self.entry_notional_usd <= 0:
                    dq_reason = ("no entry fill notional recorded, so no return "
                                 "can be computed")
                else:
                    return_pct = (
                        net_usd / self.entry_notional_usd - 1.0) * 100.0
                    provenance = "EXECUTABLE_QUOTE"
                    usable = True
            elif (not self.entry_notional_usd
                  and "entry" not in (dq_reason or "").lower()):
                # Keep the missing-basis fact visible even when the quote failed.
                dq_reason = f"{dq_reason}; also no entry fill basis recorded"

            obs = HoldPathObservation(
                trade_id=self.trade_id,
                mint=self.mint,
                curve_or_pool_id=self.curve_or_pool_id,
                observation_id=obs_id,
                strategy_profile=self.strategy_profile,
                phase=self._phase,
                entry_fill_usd=self.entry_fill_usd,
                entry_notional_usd=self.entry_notional_usd,
                position_tokens=self.position_tokens,
                entry_price_provenance=self.entry_price_provenance,
                observed_at_utc=now_wall,
                elapsed_ms=elapsed_ms,
                remaining_hold_ms=remaining_ms,
            remaining_token_amount_raw=self.remaining_token_amount_raw,
                token_decimals=self.token_decimals,
                quote_asset=self.quote_asset,
                sol_usd_reference=self.sol_usd_reference,
                aggregator_mark_usd=mark_usd,
                curve_spot_price_quote_asset=spot,
                sell_quote_net_raw=int(net_raw) if net_raw is not None else None,
                sell_quote_gross_raw=None,  # not exposed by the quote API
                sell_quote_net_usd=net_usd,
                effective_liquidation_price_quote_asset=(
                    Decimal(str(effective)) if effective is not None else None),
                sell_quote_source=getattr(quote, "provider", None) if quote else None,
                quote_timestamp_utc=getattr(quote, "quote_timestamp", None) if quote else None,
                quote_age_ms=getattr(quote, "quote_age_ms", None) if quote else None,
                state_slot=getattr(quote, "state_slot", None) if quote else None,
                state_age_ms=getattr(quote, "state_age_ms", None) if quote else None,
                curve_reserve_numerator_raw=reserve_num,
                curve_reserve_denominator_raw=reserve_den,
                price_impact_pct=(
                    float(getattr(quote, "price_impact_pct"))
                    if quote is not None
                    and getattr(quote, "price_impact_pct", None) is not None else None),
                fee_bps=getattr(quote, "fee_bps", None) if quote else None,
                quote_status=status,
                unavailable_reason=reason,
                identity_check_status=identity,
                identity_evidence=getattr(quote, "identity_evidence", None) if quote else None,
                market_type=getattr(quote, "market_type", None) if quote else None,
                return_pct=return_pct,
                reference_price_provenance=provenance,
                usable_for_return=usable,
                data_quality_reason=dq_reason,
                monitor_status=monitor_status,
                research_tag=(
                    POST_EXIT_RESEARCH_ONLY
                    if self._phase == POST_EXIT_PHASE else ""),
                raw_snapshot_reference={
                    "spot_numerator_raw": reserve_num,
                    "spot_denominator_raw": reserve_den,
                    "input_amount_raw": getattr(quote, "input_amount_raw", None) if quote else None,
                    "state_slot": getattr(quote, "state_slot", None) if quote else None,
                },
            )
            self.observations.append(obs)
            self._append({"record_type": "OBSERVATION", **obs.as_dict()})
            # Only the first sample is the entry. Everything after it is the
            # path -- and a POST_EXIT phase is never clobbered back to PATH.
            if self._phase == "ENTRY":
                self._phase = "PATH"
            return obs
        except Exception as exc:  # noqa: BLE001 - telemetry never breaks a caller
            self.telemetry_error = f"sample failed: {type(exc).__name__}: {exc}"[:300]
            return None

    def _record_skip(self, now_wall: float, reason: str, *,
                     quote: Any = None,
                     monitor_status: str = MONITOR_SKIPPED_BUDGET
                     ) -> HoldPathObservation:
        """
        Record a sample that was DELIBERATELY NOT TAKEN.

        Persisted rather than dropped, so the dataset can distinguish "the
        sampler was throttled to protect an exit" from "the market moved and we
        were not looking". Only the entry is forced through: a hold path that
        cannot contain its own entry is not a hold path, and one entry sample is
        not a meaningful load.
        """
        self._sequence += 1
        obs = HoldPathObservation(
            trade_id=self.trade_id,
            mint=self.mint,
            curve_or_pool_id=self.curve_or_pool_id,
            observation_id=f"{self.trade_id}#{self._sequence:05d}",
            strategy_profile=self.strategy_profile,
            phase=self._phase,
            entry_fill_usd=self.entry_fill_usd,
            entry_notional_usd=self.entry_notional_usd,
            position_tokens=self.position_tokens,
            entry_price_provenance=self.entry_price_provenance,
            observed_at_utc=now_wall,
            elapsed_ms=(self._monotonic() - (self._start_monotonic
                                             or self._monotonic())) * 1000.0,
            remaining_token_amount_raw=self.remaining_token_amount_raw,
            token_decimals=self.token_decimals,
            quote_asset=self.quote_asset,
            sol_usd_reference=self.sol_usd_reference,
            aggregator_mark_usd=None,
            sell_quote_source=getattr(quote, "provider", None),
            quote_status=SKIPPED,
            unavailable_reason=reason[:300],
            data_quality_reason=reason[:300],
            monitor_status=monitor_status,
            research_tag=(
                POST_EXIT_RESEARCH_ONLY
                if self._phase == POST_EXIT_PHASE else ""),
            reference_price_provenance="UNAVAILABLE",
            return_pct=None,
            usable_for_return=False,
            remaining_hold_ms=(
                max(0.0, self.expected_window_ms
                    - (self._monotonic() - (self._start_monotonic
                                            or self._monotonic())) * 1000.0)
                if self.expected_window_ms is not None else None),
        )
        self.observations.append(obs)
        self._append({"record_type": "OBSERVATION", "skipped": True, **obs.as_dict()})
        self.skipped_samples += 1
        # Only the first sample is the entry; a forced entry sample still counts.
        # A POST_EXIT phase is never clobbered back to PATH.
        if self._phase == "ENTRY":
            self._phase = "PATH"
        return obs

    # -- output --------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Metrics for this position. Never raises."""
        try:
            return compute_hold_metrics(
                self.observations, expected_window_ms=self.expected_window_ms)
        except Exception as exc:  # noqa: BLE001
            return {"metrics_status": PARTIAL_DATA,
                    "unavailable_reason": f"summary failed: {type(exc).__name__}"}

    def _append(self, row: dict[str, Any]) -> None:
        """
        Append one JSON line. Opens and closes per write.

        Append-only and never truncating: a telemetry append can add a row but
        cannot damage an existing one.
        """
        path = self.config.sink_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
        except Exception as exc:  # noqa: BLE001
            self.telemetry_error = f"append failed: {type(exc).__name__}"[:200]


def load_observations(path: Path) -> list[HoldPathObservation]:
    """Read back observations from an append-only sink. Used by research."""
    out: list[HoldPathObservation] = []
    if not path.exists():
        return out
    allowed = set(HoldPathObservation.__dataclass_fields__)  # type: ignore[attr-defined]
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("record_type") != "OBSERVATION":
            continue
        kwargs = {k: v for k, v in row.items() if k in allowed}
        for key in ("curve_spot_price_quote_asset",
                    "effective_liquidation_price_quote_asset"):
            if isinstance(kwargs.get(key), str):
                try:
                    kwargs[key] = Decimal(kwargs[key])
                except Exception:  # noqa: BLE001
                    kwargs[key] = None
        try:
            out.append(HoldPathObservation(**kwargs))
        except Exception:  # noqa: BLE001
            continue
    return out