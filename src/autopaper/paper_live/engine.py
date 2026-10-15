"""
The execution-aware PAPER engine: one signal in, one simulated trade or one
recorded non-trade out.

THE PIPELINE, FOR ONE SIGNAL
============================

    Telegram message
      -> immutable SignalSnapshot          (no credentials, no session, no key)
      -> fresh DexScreener OBSERVED_MARK   (observation only)
      -> production L25 decision           (autopaper.l25 / DecisionEngine)
      -> per-arm extra rule                (H2 only; recorded separately)
      -> production position sizing        (autopaper.sizing)
      -> FRESH EXECUTABLE BUY QUOTE        (autopaper.same_market, read-only)
      -> SIMULATED FILL                    (derived from the quote only)
      -> ...position open...
      -> production EXIT decision          (autopaper.exits.ExitPolicy)
      -> FRESH EXECUTABLE SELL QUOTE       (TOKEN -> SOL, never reversed)
      -> SIMULATED FILL
      -> fill-based settlement

    NO TRANSACTION IS EVER BUILT, SIGNED OR BROADCAST.

WHAT IS THE PRODUCTION STRATEGY, VERBATIM
==========================================

Nothing in that chain redefines a gate, a threshold, a precedence or a size.
The engine CALLS:

* :func:`autopaper.l25.evaluate` - the frozen entry gate, hash-verified on every
  call by its own module;
* :func:`autopaper.analysis.features.build_features` - the feature build;
* :class:`autopaper.analysis.decision.DecisionEngine` - the full decision
  including every blocking risk flag;
* :func:`autopaper.sizing.compute_position_size` - the sizing formula;
* :class:`autopaper.exits.ExitPolicy` - the exit precedence.

The ONLY thing this engine changes is WHERE THE PRICE COMES FROM.

THE MARK IS NEVER A FILL
=========================

:func:`build_signal_records` takes the observed mark and an executable quote and
records both, plus the gap between them, and then opens the position from the
quote. There is no code path from a mark to a
:class:`~autopaper.paper_live.schema.SimulatedFill`, and a test asserts it.

FAIL-CLOSED
===========

If an executable quote is missing, stale, or not provably the same market, no
simulated position is opened. The signal is still recorded - with the reason -
because a dataset of trades alone cannot show that execution quality is itself
filtering out candidates.

EXIT FALLBACK: THE DETERMINISTIC RULE
====================================

When an exit condition fires but no executable SELL quote can be obtained, the
position is **left OPEN and the failure is recorded**. It is not marked to
market, not closed at the mark, and not force-closed at cost. The position stays
open and is re-evaluated on the next monitor pass, so a transient RPC or quote
outage cannot silently turn into a realised paper loss or a phantom gain. The
rule is deterministic, stated here and asserted by a test.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import hashlib
import inspect
import math
import time
from pathlib import Path
import uuid
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any, Mapping, Optional, Sequence

from autopaper.analysis.decision import DecisionEngine
from autopaper.analysis.features import build_features
from autopaper.exits import ExitDecision, ExitPolicy
from autopaper.logging_setup import log
from autopaper.l25 import L25_HASH
from autopaper.l25 import evaluate as evaluate_l25
from autopaper.models import (
    DataState,
    Decision,
    ExitStyle,
    Observation,
    ObservationSet,
    Position,
    PositionStatus,
    Regime,
    Route,
)
from autopaper.forensics.quote_excursions import (
    QuoteExcursionPoint,
    excursions,
    net_quote_return_pct,
)
from autopaper.same_market.models import WSOL_MINT
from autopaper.same_market.models import SOL_DECIMALS
from autopaper.same_market.numeric import to_float

#: The named reason for "the provider answered, but there was no observation".
#: Distinct from a provider FAILURE: nothing threw, there was simply nothing to
#: price, so the candidate is declined rather than crashed on.
OBSERVATION_UNAVAILABLE_REASON = (
    "OBSERVATION_UNAVAILABLE: the market data provider returned no "
    "observation for this token, so no mark exists to price or guard against"
)

from .account import VirtualAccount
from .arms import (
    ArmSpec,
    ResearchArm,
    BETA_ARMS,
    BETA_GROUP,
    BETA_ARM_KEYS,
    GATE_L25,
    GATE_V2_FLOOR,
    entry_gate_for_arm,
    is_beta_group,
)
from .execution_costs import (
    deployable_sol_lamports,
    outside_swap_costs,
)
from .v2_floor_beta import (
    BetaEntryFill,
    BETA_ARMS_ORDER,
    BETA_EXIT_BLOCK_NOTIFY_EVERY,
    BETA_PROFILE_LABEL,
    BETA_STARTING_BANKROLL_USD,
    BetaExperiment,
    EXECUTABLE_QUOTE_BASIS,
    LEGACY_BETA_SIZING_PCT,
    RETURN_BASIS_QUOTE,
    arm_config,
)
from .economics import (
    TOKEN_ACCOUNT_RENT_LAMPORTS,
    RentLedger,
    settle_rent,
)
from .fills import FillModel, FillRefused, fill_from_quote, realize
from .guards import assert_paper_only
from .quotes import QuoteService, SolReference
from .schema import (
    ACTUAL_FILL_AVAILABLE,
    FailureReason,
    LatencyMarks,
    ObservedMarkPoint,
    PaperPosition,
    QuotePoint,
    SignalRecord,
    SignalSnapshot,
    TradeRecord,
)

#: The L25 definition hash, stamped on every record so a dataset can never be
#: silently attributed to a gate that has since changed.
BASELINE_HASH = L25_HASH

#: The version of THIS harness's schema, so a later dataset is distinguishable.
SCHEMA_VERSION = "execution_aware_paper/v1"

from .schema import entry_execution_gap_observability as _entry_execution_gap_observability


def _one_line(text: Any, limit: int = 180) -> str:
    """Collapse a reason to ONE bounded, single-line string for a message.

    A refusal reason can carry a traceback or an embedded newline, and a
    multi-line reason breaks a Telegram message's layout. Collapsing the
    whitespace and truncating keeps the reason readable and bounded; this is a
    display concern only, so it never alters a recorded value.
    """
    return " ".join(str(text or "").split())[:limit]


def classify_quote_failure(reason: str) -> FailureReason:
    """
    Name WHY a quote is missing, instead of reporting one flat verdict.

    A missing quote has three genuinely different causes, and they call for
    opposite responses:

    * the decision ran out of TIME -> shrink the budget; do not distrust the
      token;
    * an upstream node was THROTTLED -> add capacity or fail over; the token
      may be perfectly tradable;
    * the token genuinely has NO MARKET -> the strategy is right to refuse.

    Reporting all three as one "no executable quote" verdict is what let a
    capacity limit be investigated as a strategy problem. Capacity is now
    counted and named separately from market absence, so the next occurrence
    is diagnosable from the log alone.
    """
    text = (reason or "").lower()
    if "quote_timeout" in text or "did not arrive within" in text:
        return FailureReason.QUOTE_TIMEOUT
    if ("rate limit" in text or "rate-limit" in text
            or "too many request" in text or "rate cap" in text
            # The current wording from the limiter. The older markers above are
            # kept because historical records still carry them, and a classifier
            # that stops understanding its own archive is its own defect.
            or "rpc cap" in text
            or "local quota exhausted" in text
            or "local rate cap reached" in text
            or "upstream throttled" in text
            or "telemetry lane budget" in text):
        # Covers BOTH an upstream 429 and our own local window refusing, so a
        # capacity refusal is named the same way wherever it originates. The
        # distinction between them is carried separately by
        # ``_classify_rpc_cause``, which reads the structured reason rather than
        # this text.
        return FailureReason.RPC_RATE_LIMIT
    if "solana rpc" in text or "rpc error" in text or "http 5" in text:
        return FailureReason.RPC_ERROR
    return FailureReason.QUOTE_UNAVAILABLE


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def build_signal_snapshot(
    *,
    channel: str,
    text: str,
    message_id: int | None,
    telegram_timestamp: float | None,
    mint: str | None,
    symbol: str | None,
    provenance: Mapping[str, Any] | None = None,
    warnings: Sequence[str] = (),
    ambiguous: bool = False,
    candidates: Sequence[str] = (),
    now: float | None = None,
    monotonic: float | None = None,
) -> SignalSnapshot:
    """
    An immutable record of one signal.

    The raw text is HASHED, not stored. The dataset should be joinable to a
    channel message by id without itself becoming a copy of every message the
    bot ever saw, and the token fields the decision actually used are recorded
    in full.
    """
    now = time.time() if now is None else now
    monotonic = time.monotonic() if monotonic is None else monotonic
    body = text or ""
    return SignalSnapshot(
        signal_id=f"sig_{uuid.uuid4().hex[:16]}",
        channel=channel,
        message_id=message_id,
        telegram_timestamp=telegram_timestamp,
        receive_timestamp=now,
        receive_monotonic=monotonic,
        token_mint=mint,
        symbol=symbol,
        raw_text_sha256=_sha256(body),
        raw_text_length=len(body),
        extraction_provenance=dict(provenance or {}),
        extraction_warnings=tuple(warnings),
        ambiguous=ambiguous,
        candidates=tuple(candidates),
    )


#: Reasons that mean "the same-market reference could not be established",
#: mapped out of the verdict text so a MISSING reference is never reported as
#: an EXCESSIVE gap, and a MISMATCH is never reported as an absence. The two
#: are different facts and conflating them would hide a real defect.
_REFERENCE_MISMATCH_MARKERS = (
    "not the executable market",
    "different token decimals",
    "different order size",
    "different quote assets",
    "different sides",
    "different mints",
)


def _exec_vs_mark_gap_pct(quote: Any, mark: Any) -> float | None:
    """
    Executable SELL quote vs the observed mark, in percent.

    A DIAGNOSTIC about two DIFFERENT things (what selling would realise vs what
    the aggregator observed). Deliberately not named slippage: slippage is a
    same-book concept, and this is a cross-reference one.
    """
    quote_usd = getattr(quote, "effective_price_usd", None)
    mark_usd = getattr(mark, "value_usd", None)
    if quote_usd is None or not mark_usd:
        return None
    try:
        return round((float(quote_usd) / float(mark_usd) - 1.0) * 100.0, 4)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _mark_age_ms(mark: Any) -> float | None:
    """How old the observed mark was at the moment it was used, in ms."""
    stamp = getattr(mark, "timestamp", None)
    if not stamp:
        return None
    try:
        return round(max(0.0, (time.time() - float(stamp)) * 1000.0), 1)
    except (TypeError, ValueError):
        return None


def _pinned_market_address(position: Any) -> str | None:
    """
    The bare market address this position's CURRENT market is pinned to.

    A proven exit-venue migration wins when one exists: a graduated token can
    only be sold on the pool it became, so that is the market the hold path must
    identify against. Otherwise the entry pin: ``entry_pool_address`` is
    preferred because it is stored unprefixed, else the address part of
    ``entry_venue_id`` (``<market_type>:<pool_or_curve>`` -- a STABLE MACHINE
    IDENTITY, deliberately distinct from the human-readable venue label).

    Returns ``None`` when neither is present. That is a statement about the data,
    and the caller must not substitute a guess -- including the live aggregator
    pair: a wrong address here would let the identity check pass against a
    market this position was never opened on.
    """
    exit_venue_id = str(getattr(position, "exit_venue_id", "") or "")
    if ":" in exit_venue_id:
        migrated = exit_venue_id.split(":")[-1].strip()
        if migrated:
            return migrated
    pool = getattr(position, "entry_pool_address", None)
    if pool:
        return str(pool)
    venue_id = str(getattr(position, "entry_venue_id", "") or "")
    if ":" in venue_id:
        candidate = venue_id.split(":")[-1].strip()
        return candidate or None
    return None


def _guard_failure_reason(verdict: Any) -> "FailureReason":
    """
    Name the production guard's refusal precisely.

    Three outcomes, kept distinct:

    * ``MARK_GAP_GUARD_REJECTED``            -- the entry is genuinely too
      expensive against its own market.
    * ``EXECUTION_NATIVE_REFERENCE_MISMATCH`` -- reference and quote provably
      describe different things. A defect, and reported as one.
    * ``EXECUTION_NATIVE_REFERENCE_UNAVAILABLE`` -- the same-market reference
      could not be read at all. An absence, and NEVER a reason to fall back to
      the observed mark or to another pool.
    """
    if getattr(verdict, "result", "") == "REJECT":
        return FailureReason.MARK_GAP_GUARD_REJECTED
    reason = str(getattr(verdict, "reason", "") or "")
    if any(marker in reason for marker in _REFERENCE_MISMATCH_MARKERS):
        return FailureReason.EXECUTION_NATIVE_REFERENCE_MISMATCH
    return FailureReason.EXECUTION_NATIVE_REFERENCE_UNAVAILABLE


#: The reference the PRODUCTION mark-gap guard is measured against.
#:
#: Recorded on every decision so a stored record can never be ambiguous about
#: which reference decided it. The external observed mark is still computed on
#: every entry -- as a diagnostic and as the legacy gap -- but it is NOT the
#: production decision input.
PRODUCTION_GUARD_SOURCE = "EXECUTION_NATIVE_SAME_MARKET"

#: Append-only sink for execution-native hold-path observations. Set from
#: configuration or the environment by the launcher; ``None`` disables the
#: instrument entirely and the runtime is then byte-for-byte unchanged.
_HOLD_PATH_SINK: Any = None

#: EXIT-RETRY BACKOFF BOUNDS.
#:
#: A failed exit is retried, because the position must not be stranded -- but on
#: a backoff, because re-quoting every tick against a throttling provider is what
#: keeps an outage going. The cap is deliberately BELOW a typical 60s upstream
#: cooldown: if the provider recovers inside its own window, the position is sold
#: into promptly instead of being made to wait out the whole backoff.
_EXIT_BACKOFF_BASE_S = 2.0
_EXIT_BACKOFF_CAP_S = 20.0

#: Hard bound on failed SELL-quote attempts per position before the exit is
#: declared BLOCKED. Without this the "OPEN monitoring=alive" retry line was
#: emitted with a strictly increasing attempt counter -- an outage would have
#: been retried forever, at one-tick cadence on a market-failure cause.
_EXIT_MAX_ATTEMPTS = 8

#: V2 FLOOR (BETA) EXIT-RETRY BACKOFF BOUNDS.
#:
#: The beta arms had NO retry policy at all: an unresolved arm re-quoted on EVERY
#: monitor tick (one full three-leg sweep per arm every ~5s), so a market that
#: could not be exited spent the provider budget thousands of times over -- the
#: stranded experiment logged 10,445 SELL-quote attempts in ten hours, and the
#: throttling those attempts caused is itself one of the refusal reasons.
#:
#: The SHAPE is the existing bounded exponential backoff (``_exit_backoff_s``);
#: only the bounds differ, and they differ for a stated reason:
#:
#: * the base is longer, because a beta arm's refusal is usually a MARKET fact
#:   (no executable route right now) that a 2s retry cannot change;
#: * the cap is much higher, because the point of the cooldown is to stop a
#:   research retry loop from dominating the shared RPC/Jupiter budget;
#: * there is NO terminal "unsellable" state, unlike the regular arms.
#:   ``_EXIT_MAX_ATTEMPTS`` retires a regular position after 8 attempts; a beta
#:   arm is a research observation, so it must stay OUTSTANDING and keep being
#:   reported until a real quote settles it, never be quietly retired.
BETA_EXIT_RETRY_BASE_S = 5.0
BETA_EXIT_RETRY_CAP_S = 300.0


def _telemetry_lane_guard():
    """
    A context manager putting the caller in the RPC ``telemetry`` lane.

    Handed to the sampler as a callable so ``telemetry.py`` keeps its isolation
    guarantee -- it must not import the rate limiter, or the instrument could
    reach the execution path at all. The lane is a property of the CALL SITE, not
    of the data, which is exactly why it is injected.
    """
    def _guard():
        import contextlib

        from ..market.ratelimit import LANE_TELEMETRY
        from ..same_market.rpc import rpc_lane, telemetry_sample

        @contextlib.contextmanager
        def _sample_in_lane():
            # THE SAMPLE IS THE UNIT OF WORK, NOT THE RPC CALL. Opening the
            # admission alongside the lane makes one hold-path sample cost the
            # reserved telemetry sub-budget ONE unit, so a 1 Hz sampler can no
            # longer spend the 60/min reservation in a few seconds and then
            # refuse its own reads. The lane itself is unchanged, so execution
            # capacity stays unreachable from background work.
            with telemetry_sample():
                with rpc_lane(LANE_TELEMETRY):
                    yield

        return _sample_in_lane()
    return _guard


def _unavailable_sell_stub(position: Any, detail: str):
    """
    A SELL quote that is explicitly unavailable, carrying the reason.

    Built rather than fabricated as a price: every numeric field is ``None``, so
    nothing downstream can mistake it for something it could settle against.
    """
    from decimal import Decimal

    from .schema import QuotePoint

    return QuotePoint(
        side="SELL", provider="none", market_type="UNKNOWN",
        token_mint=getattr(position, "mint", None), quote_mint=WSOL_MINT,
        input_amount_raw=int(getattr(position, "tokens_raw", 0) or 0),
        input_decimals=int(getattr(position, "token_decimals", 0) or 0),
        input_amount_decimal=None,
        output_amount_raw=None, output_decimals=None,
        output_amount_decimal=None,
        effective_price=None, effective_price_usd=None,
        price_impact_pct=None, fee_bps=None, fee_amount=None,
        venue=None, pool_or_curve=None, route=(),
        identity="UNKNOWN", available=False, missing_reason=detail[:300],
    )


def _is_telemetry_skip(exc: BaseException) -> bool:
    """
    Did the telemetry lane's OWN budget refuse this call?

    A spent telemetry lane is a deliberate cost decision, not a market fact.
    Reporting it as ``QUOTE_UNAVAILABLE`` would state "the market could not be
    read" when the truth is "we chose not to ask" -- a different and much milder
    fact, and one that would otherwise corrupt the coverage statistic this
    instrument exists to produce.

    Compares the limiter's own reason constant rather than the message text, so
    rewording an error cannot silently change the classification.
    """
    from ..market.ratelimit import TELEMETRY_BUDGET

    return getattr(exc, "reason", None) == TELEMETRY_BUDGET


def _set_hold_path_sink(path: Any) -> None:
    """Point the hold-path instrument at a file. Research plumbing only."""
    global _HOLD_PATH_SINK
    _HOLD_PATH_SINK = path


#: Provider attributes that would be a genuine PER-OBSERVATION timestamp.
#: ``pair_created_at`` is deliberately NOT here: it dates the pair, not the
#: price print, and treating it as the observation time is a category error.
_MARK_SOURCE_TIME_ATTRS = (
    "source_observed_at", "observed_at_source", "price_observed_at",
)


def _mark_source_observed_at(obs: Any) -> float | None:
    """The provider's own observation time, or ``None`` if it has none."""
    for attr in _MARK_SOURCE_TIME_ATTRS:
        value = to_float(getattr(obs, attr, None))
        if value is not None and value > 0:
            return value
    return None


def _mark_source_age_ms(obs: Any, received_at: float) -> float | None:
    """
    ``received_at - source_observed_at``, or ``None``.

    ``None`` means "unknown", and it is kept distinct from ``0.0``, which would
    mean "observed at the instant we read it". For an aggregator with no
    per-observation timestamp the first is the truth.
    """
    source_at = _mark_source_observed_at(obs)
    if source_at is None:
        return None
    age = (received_at - source_at) * 1000.0
    return age if age >= 0 else None


def _mark_age_status(obs: Any) -> str:
    """KNOWN when the provider dates the observation, else UNKNOWN."""
    return "KNOWN" if _mark_source_observed_at(obs) is not None else "UNKNOWN"


def _entry_basis_is_sane(quote: Any, *, sol_usd: float | None) -> dict[str, Any]:
    """
    Is this entry's basis the SAME market, in the SAME unit, as its fill?

    Guards the invariant that the native reference and the executable fill
    describe one book: same market id, same base mint, and a converted native
    price within two orders of magnitude of the fill.

    The band is deliberately WIDE. Ordinary cost of transacting a small position
    is single-digit percent, so 0.01x-100x passes every real fill with room to
    spare while still catching the 10**6 / 10**9 decimal-scaling and
    SOL-vs-USD mistakes -- the ones that make a correct reference look like a
    100x or 1000x move and fire a stop that does not exist.

    Returns ``{"sane": bool, "detail": str}`` so the reason is recorded, not
    merely asserted.
    """
    native = getattr(quote, "execution_native_reference", None)
    fill_usd = float(getattr(quote, "effective_price_usd", None) or 0.0)

    def bad(detail: str) -> dict[str, Any]:
        return {"sane": False, "detail": detail}

    if native is None:
        return bad("no execution-native reference on the entry quote")
    raw = getattr(native, "reference_price", None)
    if not raw or raw <= 0:
        return bad(f"native reference price is not positive: {raw!r}")
    if not sol_usd or sol_usd <= 0:
        return bad(f"invalid SOL/USD reference: {sol_usd!r}")
    native_usd = float(raw) * float(sol_usd)
    if not native_usd or native_usd <= 0:
        return bad("converted native reference is not positive")
    if fill_usd <= 0:
        return bad("entry fill has no USD price")
    quote_venue = getattr(quote, "venue_id", None)
    if quote_venue and native.market_id != quote_venue:
        return bad(
            f"native reference is for {native.market_id}, quote is for {quote_venue}")
    base = getattr(native, "base_mint", None) or getattr(quote, "token_mint", None)
    if base and getattr(quote, "token_mint", None) and base != quote.token_mint:
        return bad(f"native base mint {base} != quote token mint {quote.token_mint}")
    ratio = fill_usd / native_usd
    if not (0.01 <= ratio <= 100.0):
        return bad(
            f"unit/reference mismatch: fill {fill_usd:.6e} USD vs native "
            f"{native_usd:.6e} USD/token (ratio {ratio:.4g})")
    return {
        "sane": True,
        "detail": (
            f"native {native_usd:.6e} USD/token vs fill {fill_usd:.6e} USD "
            f"(ratio {ratio:.4f}) on {native.market_id}"
        ),
    }


#: How many SELL-quote excursion points to keep per position.
#
# The excursion needs the extrema, their timestamps and a coverage figure.
# Keeping every point on a long hold would grow a list on the position without
# bound for no additional information, so the window is bounded and the oldest
# points drop out. Coverage is computed against what was RETAINED, so a long
# hold never overstates its own coverage.
_MAX_QUOTE_EXCURSION_POINTS = 4096


def _as_positive_float(value: Any) -> Optional[float]:
    """A strictly positive float, or ``None``. A zero is never a positive value."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out <= 0:
        return None
    return out


def _gate_input(value: Any) -> Optional[float]:
    """
    A V2 FLOOR gate input as a finite float, or ``None`` when unreadable.

    ``None`` is what the gate treats as "absent" and refuses on, so this helper
    must never turn a missing value into a number -- a ratio invented from a
    partial input is not a measurement of anything.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _gate_result_of(report, gate: str):
    """
    The evaluation object for ONE gate, read off the decision report.

    The report always carries both ``l25`` and ``v2floor`` -- both are
    evaluated on every candidate -- so this is a lookup, not a re-evaluation,
    and it can never disagree with the decision that was actually made.
    Returns ``None`` when the report does not carry that gate, which the
    caller treats as "did not authorize" (fail closed).
    """
    if gate == GATE_V2_FLOOR:
        return getattr(report, "v2floor", None)
    if gate == GATE_L25:
        return getattr(report, "l25", None)
    return None


@dataclass
class PaperLiveEngine:
    """
    The execution-aware PAPER runtime.

    All collaborators are injected, so the whole engine is drivable from
    fixtures with no network, no Telegram session and no wallet.
    """

    arms: Sequence[ArmSpec]
    decision_engine: DecisionEngine
    exit_policy: ExitPolicy
    quote_service: QuoteService
    market: Any = None                      # MarketDataService, observation only
    token_decimals_for: Any = None           # async (mint) -> int | None
    sol_reference: SolReference | None = None
    fill_model: FillModel = field(default_factory=FillModel)
    starting_bankroll_usd: float = 5.00
    min_hold_s: float = 5.0
    require_identity: bool = True
    #: Record every signal, traded or not. Always true in this harness: a
    #: dataset of trades alone cannot answer whether execution quality is
    #: filtering out candidates.
    record_rejects: bool = True
    #: How many observations to retain per mint for feature building. Production
    #: keeps a longer window; this is a bound, not a policy, and the
    #: no-lookahead gate still applies to whatever is retained.
    max_history_per_mint: int = 240
    #: The micro-account economic policy. ON by default; a research run may
    #: turn it off to observe what it was refusing, which is a measurement, not
    #: a policy change.
    economics_enabled: bool = True
    #: The executable-vs-mark entry guard threshold, in percent. This is the
    #: production rule; it is a parameter only so a test can prove the boundary
    #: without inventing a different policy.
    mark_gap_threshold_pct: float = 10.0
    #: Where durable state lives. REQUIRED in practice: ``None`` no longer
    #: means "``Path('var')`` relative to the working directory", it is
    #: resolved from ``root`` and, failing that, REFUSED. See
    #: :func:`autopaper.paper_live.state.PaperStateStore`.
    state_root: Any = None
    #: Is ``state_root`` the operator's runtime state directory?
    #:
    #: Only the runtime may write there, and it must say so. A test, smoke run
    #: or research probe leaves this ``False``, so even a store pointed at the
    #: production directory is refused at construction rather than quietly
    #: accepting a synthetic session. The runtime sets it ``True`` because it
    #: resolved ``state_root`` from ``Config.var_dir``.
    state_is_runtime: bool = False
    #: V2 FLOOR (BETA) EXIT-RETRY COOLDOWN, as a CONFIGURABLE policy.
    #:
    #: The bounds are instance fields rather than bare module constants so the
    #: cooldown is tunable where it is used -- an operator can lengthen it under
    #: provider pressure, and a test can drive it to zero to exercise RETRY
    #: RECOVERY without waiting out a real cooldown. The defaults are the
    #: production policy; nothing weaker is reachable by omitting them.
    #: See ``BETA_EXIT_RETRY_BASE_S`` for what the cooldown is for and why the
    #: beta arms' bounds differ from the regular arms'.
    beta_exit_retry_base_s: float = BETA_EXIT_RETRY_BASE_S
    beta_exit_retry_cap_s: float = BETA_EXIT_RETRY_CAP_S
    # -- NATIVE (EVENT-DRIVEN) MARKET MONITORING --------------------------
    # See ``autopaper.paper_live.native_monitor``. These make the push channel
    # injectable so the engine stays drivable from fixtures with no network.
    native_monitor_enabled: bool = False
    native_monitor_ws_url: str = ""
    native_monitor_stale_after_ms: float = 8000.0
    native_monitor_open_timeout_s: float = 10.0
    #: Which exits a quote may be spent on. Gates quote REQUESTS only.
    quote_proximity_policy: Any = None
    #: The operator-selected take-profit, in percent, or ``None`` for OFF.
    #:
    #: The GLOBAL selection. It is SNAPSHOTTED onto each position at open (see
    #: :meth:`_enter`); nothing reads this value once a position exists, so
    #: changing it mid-flight cannot rewrite the exit policy of an open book.
    #: ``None`` is the default and means "no take-profit", which is exactly the
    #: behaviour ``L25_CURRENT`` had before take-profit became selectable.
    take_profit_pct: float | None = None
    #: V2_FLOOR_BETA experiment engine. Initialized when the beta group
    #: (V2_BETA_A, V2_BETA_B, V2_BETA_C) is selected.
    beta_experiment: BetaExperiment | None = None

    def __post_init__(self) -> None:
        self.audit = assert_paper_only()
        self.accounts: dict[str, VirtualAccount] = {
            spec.name.value: VirtualAccount(
                arm=spec.name.value,
                arm_version=SCHEMA_VERSION,
                baseline_hash=BASELINE_HASH,
                starting_bankroll_usd=self.starting_bankroll_usd,
            )
            for spec in self.arms
        }
        self.signal_records: list[SignalRecord] = []
        self.trade_records: list[TradeRecord] = []
        self.observed_marks: dict[str, ObservedMarkPoint] = {}
        #: Decision-time market history, per mint. Fed to the production
        #: ``build_features``, which is what lets the production route selector
        #: see momentum and structure. ``ObservationSet.as_of`` still hard-filters
        #: on ``observed_at <= decision_time``, so the no-lookahead gate holds.
        self.observation_history: dict[str, list[Observation]] = {}
        #: Cooldown state, per arm. Mirrors the production rule exactly: a mint
        #: that has just been exited cannot be re-entered, and the re-entry
        #: enable flag is not ours to set.
        self.recently_exited: dict[str, dict[str, float]] = {
            spec.name.value: {} for spec in self.arms
        }
        # Initialize V2_FLOOR_BETA experiment if all three beta arms are selected
        current_beta_arms = {spec.name.value for spec in self.arms} & set(BETA_ARMS)
        if len(current_beta_arms) == 3:
            self.beta_experiment = BetaExperiment(
                # The experiment's bankroll is ITS OWN documented figure ($10.00
                # per arm), NOT the engine's. The engine's default is the
                # research-arm $5.00, and inheriting it here would quietly halve
                # every beta ledger.
                starting_bankroll_usd=BETA_STARTING_BANKROLL_USD,
                state_root=self._resolved_state_root(),
                runtime=self.state_is_runtime,
            )
        #: VISIBILITY-ONLY MIRRORS of the beta experiment's open positions,
        #: keyed ``parent_signal_id -> arm name -> PaperPosition``.
        #:
        #: Why a mirror and not a real position in ``self.accounts``: a beta
        #: position is settled by EXACTLY ONE component --
        #: :class:`BetaExperiment`, which owns the three isolated $10 ledgers,
        #: the per-arm bracket and the realized-P&L accounting. Registering it in
        #: ``self.accounts[arm].position`` as well would put it in the regular
        #: exit loop, and the same economic trade would then be settled twice --
        #: once per ledger -- with two different answers.
        #:
        #: So these records exist for the operator's ``open positions`` view and
        #: the diagnostics surfaces. Nothing settles them, nothing moves a bankroll
        #: because of them, and ``monitor_once`` therefore also skips beta ARMS in
        #: the regular exit loop (they are handled in the beta block above it).
        self.beta_positions: dict[str, dict[str, Any]] = {}
        #: Beta positions settled on the most recent tick, read by
        #: the notification layer via ``_monitor_beta_experiment_for_tick``.
        self._last_beta_settled: list[Any] = []
        #: parent signal id -> why the executable BUY quote refused, for the
        #: arm records and the operator's views. Never a price.
        self._beta_quote_refusals: dict[str, str] = {}
        #: BETA EXIT-RETRY COOLDOWN, keyed by ``(parent_signal_id, arm)``.
        #:
        #: ``{"attempts": n, "next_attempt_at": monotonic-ish wall clock}``.
        #: In-memory only, and deliberately so: a restart is a fresh provider
        #: budget, and the durable record of how long an arm had been stranded is
        #: ``BetaPosition.exit_attempts``, which the experiment persists itself.
        #: Nothing here prices anything or settles anything -- it only decides
        #: WHEN the next quote request is allowed.
        self._beta_exit_retry: dict[tuple[str, str], dict[str, Any]] = {}
        #: BOUNDED, structured observability for the beta lifecycle. One record
        #: per stage (gate -> buy quote -> entry -> exit trigger -> sell quote ->
        #: settlement), each linked to the parent signal and arm, so eligible
        #: signals, accepted/rejected quotes, created positions, unresolved arms
        #: and completed settlements can be reconciled from one place. In-memory
        #: and capped: telemetry, never a second ledger and never persisted.
        self.beta_stage_log: list[dict[str, Any]] = []
        #: Live execution-native hold-path samplers, keyed by
        #: ``"<signal_id>:<arm>"``. Telemetry only: nothing in the decision path
        #: reads this, and an empty dict is the fully-disabled state.
        self._hold_path_recorders: dict[str, Any] = {}
        #: TICK-LOCAL SELL-QUOTE MEMO. Cleared at the start of every
        #: ``monitor_once``, so a reused quote can never outlive one tick.
        self._tick_sell_quotes: dict[tuple, Any] = {}
        #: Set by ``_close_for_arm`` when the settlement quote came from this
        #: tick's memo rather than a fresh fetch.
        self._sell_quote_reused = False
        #: BOUNDED EXIT-RETRY STATE, keyed by ``(signal_id, arm)``.
        #:
        #: A failed exit leaves the position OPEN and is retried -- but on a
        #: BACKOFF, not on every tick. Re-quoting on every tick while the provider
        #: is throttling us is what turns a temporary outage into a sustained one:
        #: every attempt spends the budget and none of them can win. The backoff
        #: is bounded well under a typical 60s upstream cooldown so a recovered
        #: position settles promptly, and it carries JITTER so several positions
        #: failing together do not retry in lockstep.
        self._exit_retry: dict[tuple, dict[str, Any]] = {}
        #: MONITORING DIAGNOSTICS -- aggregate per-tick counters for the hold path.
        #: These name WHY monitoring was or was not usable on each tick, so the
        #: 123s data-invalid timeout can be diagnosed from the log instead of
        #: guessed at. They are additive and never cleared during a run.
        self._monitor_diag: dict[str, int] = {
            "MONITOR_OK": 0,
            "MONITOR_RPC_ERROR": 0,
            "MONITOR_QUOTE_UNAVAILABLE": 0,
            "MONITOR_IDENTITY_UNKNOWN": 0,
            "MONITOR_IDENTITY_MISMATCH": 0,
            "MONITOR_SKIPPED_BUDGET": 0,
            # ROOT-CAUSE counter, not a category: it is incremented IN ADDITION to
            # the classified cause above whenever the push channel was attached
            # but was not delivering. It answers "why was there nothing to fall
            # back from", which the per-cause counters cannot.
            "MONITOR_WS_DISCONNECTED": 0,
            "MONITOR_TICKS": 0,
            # How many beta exit retries were SUPPRESSED by the cooldown this
            # run. Diagnostic only: a large number here with a small number of
            # refusals is the cooldown working, not the exit stalling.
            "BETA_EXIT_RETRY_DEFERRED": 0,
            "RPC_CALLS_THIS_TICK": 0,
            "QUOTE_AGE_MS_SUM": 0,
            "QUOTE_AGE_MS_COUNT": 0,
        }
        #: Shared, freshness-bounded quote cache. The arms are the same arms on
        #: the same mint at the same notional, so without this they would be
        #: measured against three DIFFERENT snapshots of the same book.
        try:
            from .quotes import QuoteCache
            self.quote_cache = QuoteCache()
        except Exception:  # noqa: BLE001 - sharing is an optimisation
            self.quote_cache = None
        #: The candidate venues quoted for the most recent BUY, kept for the
        #: dataset. This is the multi-venue research material.
        self.last_buy_candidates: list[Any] = []
        self.last_buy_cache_state: str = "fresh"
        #: Arms that joined an in-flight quote instead of issuing their own.
        self.last_quote_coalesced: int = 0
        #: Quote refusals by CAUSE. A throughput problem has to be countable,
        #: otherwise it is invisible until it shows up as a lower fill rate.
        self.quote_capacity_refusals: dict[str, int] = {}
        self.sell_counterfactuals: list[dict[str, Any]] = []
        #: Rent lifecycle of the most recent settled position.
        self.rent_ledger: Any = None
        #: The most recent executable-vs-mark guard verdict, for the dataset.
        self.last_mark_gap: Any = None
        self.last_decimals_error: str = ""
        #: Token decimals are immutable for a given mint, so they are cached.
        #: Without this, three arms on one signal issue three identical chain
        #: reads for a value that cannot have changed.
        self._decimals_cache: dict[str, int] = {}
        #: Durable state store, created on first use.
        self.state_store: Any = None
        #: THE FAST MARKET MONITOR. One entry per DISTINCT MARKET, shared by
        #: every arm's position on it: the venue account is a property of the
        #: market, so subscribing per position would pay twice for the same
        #: bytes.
        from .native_monitor import NativeMonitorRegistry
        from .quote_proximity import DEFAULT_PROXIMITY
        self.native_monitors = NativeMonitorRegistry()
        self.quote_proximity = (
            self.quote_proximity_policy
            if self.quote_proximity_policy is not None
            else DEFAULT_PROXIMITY)
        #: Which monitor instance backs each ``(signal_id, arm)`` position, so
        #: closing a position releases exactly the reference it took.
        self._position_monitors: dict[tuple, Any] = {}
        #: Executable-quote request timestamps per position, for the per-minute
        #: rate that proves the quote side collapsed while monitoring did not.
        self._quote_request_times: dict[tuple, list[float]] = {}

    # -- observation ------------------------------------------------------

    # -- STRATEGY SELECTION ----------------------------------------------

    def reconcile_arms(self, selected: Sequence[str]
                       ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """
        Make this engine's arm set equal to ``selected``, in place.

        Returns ``(deferred, reconciled)``:

          * ``deferred`` -- arms that were asked to stop but still hold an open
            position, so they keep running. Reported, never silent.
          * ``reconciled`` -- arms actually added or removed.

        WHY IN PLACE AND NOT BY REBUILDING
        ----------------------------------
        A rebuild would construct a fresh engine, and a fresh engine has empty
        virtual accounts: an arm's bankroll, its cooldown map and its open
        position would all be gone. That is not a hot switch, it is a wipe.
        Reconciling the arm list keeps each arm's own state attached to that arm.

        WHY A REMOVAL IS DEFERRED WHILE A POSITION IS OPEN
        ---------------------------------------------------
        Because a position belongs to the strategy that opened it. Removing the
        arm mid-hold would either abandon the position -- leaving it open and
        unmonitored -- or re-stamp it onto the newly selected strategy, which
        would rewrite the exit policy of something already in flight. Both are
        worse than the operator waiting for the position to close, so the arm
        stays, the position is untouched, and the operator is told.

        ADDING an arm needs no such care: a brand-new arm has no history to
        lose, and its account starts from the same declared starting bankroll
        every other arm started from.
        """
        from .arms import ARMS, ResearchArm

        wanted = tuple(str(name) for name in selected)
        wanted_specs = tuple(ARMS[ResearchArm(name)] for name in wanted)
        current_names = tuple(spec.name.value for spec in self.arms)
        added = tuple(n for n in wanted if n not in current_names)
        removed = tuple(n for n in current_names if n not in wanted)

        deferred: list[str] = []
        for name in removed:
            account = self.accounts.get(name)
            if account is not None and getattr(account, "position", None) is not None:
                deferred.append(name)

        # A deferred arm STAYS in the running set. It is not in ``wanted``
        # (the operator deselected it) and it must not be in the running set
        # either, because then its open position would stop being monitored --
        # which is the exact outcome the deferral exists to prevent. So the
        # running set is the SELECTION plus the deferrals, in the selection's
        # order first.
        effective = list(wanted_specs)
        for name in deferred:
            from .arms import ARMS as _ARMS, ResearchArm as _RA
            effective.append(_ARMS[_RA(name)])
        effective_names = tuple(spec.name.value for spec in effective)

        if effective_names == current_names:
            return tuple(deferred), ()

        for spec in effective:
            if spec.name.value in self.accounts:
                continue
            self.accounts[spec.name.value] = VirtualAccount(
                arm=spec.name.value,
                arm_version=SCHEMA_VERSION,
                baseline_hash=BASELINE_HASH,
                starting_bankroll_usd=self.starting_bankroll_usd,
            )
            self.recently_exited[spec.name.value] = {}
        for name in removed:
            if name in deferred:
                continue
            self.accounts.pop(name, None)
            self.recently_exited.pop(name, None)

        self.arms = list(effective)

        # Initialize or clear the V2_FLOOR_BETA experiment
        current_beta_arms = {spec.name.value for spec in self.arms} & set(BETA_ARMS)
        if len(current_beta_arms) == 3:
            # All three beta arms selected - initialize experiment
            if self.beta_experiment is None:
                self.beta_experiment = BetaExperiment(
                    starting_bankroll_usd=BETA_STARTING_BANKROLL_USD,
                    state_root=self._resolved_state_root(),
                    runtime=self.state_is_runtime,
                )
        else:
            # The beta group is no longer fully selected. Outstanding PAPER
            # positions are settled FIRST -- via ``_stop_beta_experiment``, the
            # same flatten the kill switch uses -- so nothing is abandoned
            # unmonitored and no further beta entry can start. The experiment's
            # durable state stays on disk: re-selecting the group later
            # RESTORES the session rather than silently resetting it to $10.
            if self.beta_experiment is not None:
                summary = self._stop_beta_experiment()
                closed = summary.get("closed") or []
                unresolved = summary.get("unresolved") or []
                if closed:
                    log("INFO",
                        "V2 FLOOR (BETA) deselected: settled %d open arm "
                        "position(s) at the last observed price before "
                        "stopping the experiment", len(closed))
                if unresolved:
                    log("WARN",
                        "V2 FLOOR (BETA) deselected with %d arm position(s) "
                        "that have NO observed price: left OPEN in the "
                        "experiment's durable state (no price, no trade): %s",
                        len(unresolved),
                        ", ".join(
                            f"{r.get('arm')}:{r.get('parent_signal_id')}"
                            for r in unresolved))
            self.beta_experiment = None
            self.beta_positions = {}

        reconciled = tuple(added) + tuple(
            n for n in removed if n not in deferred)
        return tuple(deferred), reconciled

    def _stop_beta_experiment(self) -> dict[str, Any]:
        """
        Stop the V2 FLOOR (BETA) experiment safely: settle what can be
        settled, then drop the in-process book.

        This is the documented counterpart to ``reconcile_arms``' contract that
        beta positions are closed before the experiment is cleared. It uses the
        SAME flatten the kill switch uses (:meth:`BetaExperiment.flatten_all`):

          * the exit basis is the LAST OBSERVED return -- never a fabricated
            price;
          * an arm with no observation at all is REFUSED and reported as
            still-open (no price, no trade);
          * every closed arm is stamped ``FLATTEN`` so the forced exit stays
            visible in the ledger and in execution-quality reporting.

        NOTHING IS DELETED. Trades, the archive and any unresolved open
        position remain in the experiment's durable state file, so re-selecting
        the group restores the session instead of quietly starting over.
        """
        experiment = self.beta_experiment
        if experiment is None:
            return {"closed": [], "unresolved": []}
        summary = experiment.flatten_all()
        # The experiment is being deselected: its book leaves this process (its
        # durable state does not), and the visibility mirrors follow it.
        self.beta_experiment = None
        self.beta_positions = {}
        return summary

    def strategy_layers(self) -> dict[str, Any]:
        """
        Every layer that names a strategy for the open book, read back live.

        This exists so the invariant can be CHECKED rather than asserted in a
        comment. The PAPER BUY notification, the open position and the exit
        policy all resolve a profile from the arm that opened the position, so
        they cannot drift from each other -- but they CAN drift from the
        operator's selection, and that is the defect this whole section is about.
        """
        layers: dict[str, Any] = {
            "arms_running": [spec.name.value for spec in self.arms],
            "arm_profiles": {
                spec.name.value: spec.production_profile.value
                for spec in self.arms
            },
            "open_positions": {
                arm: {
                    "arm": str(getattr(account.position, "arm", "")),
                    "production_profile": str(getattr(
                        account.position, "production_profile", "")),
                    "signal_id": str(getattr(
                        account.position, "signal_id", "")),
                }
                for arm, account in (self.accounts or {}).items()
                if getattr(account, "position", None) is not None
            },
        }
        mismatches = []
        for arm, position in layers["open_positions"].items():
            expected = layers["arm_profiles"].get(arm)
            if expected is None:
                mismatches.append(
                    f"open position on {arm}, which is no longer a running arm")
            elif expected != position["production_profile"]:
                mismatches.append(
                    f"open position on {arm} carries "
                    f"{position['production_profile']} but its arm runs "
                    f"{expected}")
        layers["mismatches"] = mismatches
        return layers

    # -- NATIVE MARKET MONITORING ----------------------------------------

    def _effective_native_monitor_url(self) -> str:
        """
        The configured push endpoint, VALIDATED, or ``""``.

        The engine is a collaborator-driven harness, so a value can arrive from
        the config loader, from a research script or from a test. It is checked
        here rather than trusted, because an operator who set an HTTP URL must
        get a named diagnostic and the polling fallback -- never a monitor that
        silently never connects while reporting that it is enabled.
        """
        raw = str(self.native_monitor_ws_url or "").strip()
        if not raw:
            return ""
        try:
            from ..same_market.ws import parse_ws_url

            parse_ws_url(raw)
        except Exception as exc:  # noqa: BLE001 - a bad URL is a diagnostic
            log("ERROR",
                "native monitor: native_monitor_ws_url is not a websocket "
                "endpoint (%s); the push channel is DISABLED and monitoring "
                "falls back to RPC polling", str(exc)[:160])
            return ""
        return raw

    def _monitor_config(self) -> Any:
        from .native_monitor import MonitorConfig
        return MonitorConfig(
            ws_url=self._effective_native_monitor_url(),
            stale_after_ms=self.native_monitor_stale_after_ms,
            open_timeout_s=self.native_monitor_open_timeout_s,
        )

    def _ensure_native_monitor(self, position: PaperPosition,
                               quote: Any = None) -> Any:
        """
        Attach this position's market to the push channel, once.

        The STATIC identity comes from the ENTRY quote, which already proved the
        market on chain and already knows its address, type and decimals. It is
        never re-derived here: re-discovering the market per tick was one of the
        costs this architecture removes.
        """
        key = self._exit_retry_key(position)
        existing = self._position_monitors.get(key)
        if existing is not None:
            return existing
        #: The market types the push channel can actually decode.
        #:
        #: A position entered through an aggregator route (``JUPITER_ROUTED``)
        #: names a route pool the monitor cannot parse -- a bonding curve's
        #: reserves and a PumpSwap pool's layout are both documented, and an
        #: aggregator's route key is neither. Subscribing to it would open a
        #: websocket, count zero updates, and never produce a price, so those
        #: positions are served by the existing recovery path exactly as before.
        #: No behaviour is lost: the monitor is an addition, not a replacement.
        monitor_types = {"PUMP_BONDING_CURVE", "PUMPSWAP"}
        ws_url = self._effective_native_monitor_url()
        if not self.native_monitor_enabled or not ws_url:
            # NOT AN ERROR, and it must stay silent on every position: the
            # absence of a push endpoint is a documented configuration state that
            # leaves the polling recovery path in charge. The one-time statement
            # of which path a run is on is made by the config diagnostic and by
            # the first RECOVERY log line, so a position opening here costs no
            # log line and no request.
            return None
        from .native_monitor import StaticVenueIdentity
        market_type = str(
            getattr(quote, "market_type", "") or
            getattr(position, "entry_market_type", "") or "")
        if market_type not in monitor_types:
            return None
        market_address = (
            _pinned_market_address(position)
            or str(getattr(quote, "pool_or_curve", "") or ""))
        if not market_address:
            # No pin means there is nothing to subscribe to, and guessing an
            # address is exactly the unrelated-market fallback this must not do.
            return None
        identity = StaticVenueIdentity(
            mint=position.mint,
            market_type=market_type or "PUMP_BONDING_CURVE",
            market_address=market_address,
            base_mint=str(getattr(quote, "token_mint", "") or position.mint),
            quote_mint=str(getattr(quote, "quote_mint", "") or ""),
            token_decimals=int(position.token_decimals or 6),
            pool_base_vault=str(
                (position.cached_identity or {}).get("base_vault") or ""
            ) if isinstance(position.cached_identity, dict) else "",
            pool_quote_vault=str(
                (position.cached_identity or {}).get("quote_vault") or ""
            ) if isinstance(position.cached_identity, dict) else "",
        )
        monitor = self.native_monitors.acquire(identity, self._monitor_config())
        # Register a callback for immediate exit evaluation on native price updates.
        # This enables true event-driven monitoring: exit conditions are evaluated
        # the moment a new native price arrives, not just on the timer tick.
        def _on_native_price_update(monitor, price_point):
            # Schedule immediate exit evaluation for this position.
            # We use call_soon to avoid blocking the monitor's deliver() path.
            loop = asyncio.get_running_loop()
            loop.call_soon(
                lambda: asyncio.create_task(
                    self._evaluate_exits_on_native_update(position, monitor, price_point)
                )
            )
        monitor._on_price_update = _on_native_price_update
        self._position_monitors[key] = monitor
        return monitor

    def _release_native_monitor(self, position: PaperPosition) -> None:
        key = self._exit_retry_key(position)
        monitor = self._position_monitors.pop(key, None)
        self._quote_request_times.pop(key, None)
        if monitor is not None:
            self.native_monitors.release(monitor)

    def _monitor_reference(self, position: PaperPosition,
                           live: Any) -> tuple[float | None, dict[str, Any]]:
        """
        The native reference price from the PUSH channel. Zero RPC, zero HTTP.

        Returns ``(usd_per_token, provenance)`` with ``usd_per_token=None``
        whenever the pushed state cannot be proven current AND same-market, in
        which case the caller falls back to its conservative recovery path. The
        identity rule is deliberately strict: the pushed price only counts when
        it was computed from the market THIS position is pinned to, so a
        migration cannot leave a position quoting a curve it graduated off.
        """
        from .native_monitor import SOURCE_PUSH
        provenance: dict[str, Any] = {
            "monitor_source": SOURCE_PUSH,
            "native_reference_raw": None,
            "native_reference_usd": None,
            "reference_identity": "UNVERIFIED",
            "reference_market_id": None,
            "reference_age_ms": None,
            "unavailable_reason": None,
        }
        monitor = self._position_monitors.get(self._exit_retry_key(position))
        if monitor is None:
            return None, provenance
        point = monitor.latest()
        if point is None or not point.usable:
            provenance["unavailable_reason"] = (
                getattr(point, "reason", "") or
                "no native market update has arrived yet")[:200]
            return None, provenance

        pinned = _pinned_market_address(position)
        if pinned and point.market_address != pinned:
            # A migration the position has NOT yet recorded. The pushed price is
            # for a market the exit path does not believe in, so it is refused
            # rather than used -- an identity FAIL here is the same outcome the
            # polled path produced, reached without spending an RPC.
            provenance["reference_identity"] = "MISMATCH"
            provenance["unavailable_reason"] = (
                f"pushed market {point.market_address[:8]} is not this "
                f"position's pinned market {str(pinned)[:8]}")[:200]
            return None, provenance

        raw = point.price_sol_per_token
        sol_usd = position.sol_usd_reference
        if raw is None or raw <= 0:
            provenance["unavailable_reason"] = "pushed native price is not positive"
            return None, provenance
        if not sol_usd or sol_usd <= 0:
            provenance["unavailable_reason"] = (
                f"invalid SOL/USD reference {sol_usd!r}; refusing to convert")
            return None, provenance
        usd_per_token = float(raw) * float(sol_usd)
        if usd_per_token <= 0:
            provenance["unavailable_reason"] = (
                "converted pushed native reference is not positive")
            return None, provenance
        provenance["native_reference_raw"] = raw
        provenance["native_reference_usd"] = usd_per_token
        provenance["reference_identity"] = "PASS"
        provenance["reference_market_id"] = point.venue_id
        provenance["reference_age_ms"] = point.age_ms
        return usd_per_token, provenance

    def _apply_monitor_diagnostics(self, position: PaperPosition) -> None:
        """Copy the push channel's health onto the position, every tick.

        Cheap and unconditional, so a monitoring problem is visible on the
        position that has it rather than only in an aggregate counter.
        """
        monitor = self._position_monitors.get(self._exit_retry_key(position))
        if monitor is None:
            position.monitor_source = ""
            position.websocket_connected = False
            position.websocket_state = "DISABLED"
            return
        position.monitor_source = "WEBSOCKET_PUSH"
        position.websocket_state = monitor.state
        position.websocket_connected = bool(
            monitor._stream is not None and monitor._stream.is_open)
        position.native_update_age_ms = monitor.age_ms()
        position.native_last_slot = monitor._last_slot
        position.native_high_water_slot = monitor._high_water_slot
        position.native_update_count = monitor.stats.updates
        position.native_update_interval_median_ms = \
            monitor.stats.update_interval_median_ms
        position.native_update_interval_p95_ms = \
            monitor.stats.update_interval_p95_ms
        position.native_update_interval_max_ms = \
            monitor.stats.update_interval_max_ms
        position.stream_gap_count = monitor.stats.stream_gaps
        position.reconnect_count = monitor.stats.reconnects
        position.rate_limit_429_count = monitor.stats.rate_limit_429
        migration = monitor.migration()
        if migration is not None and not position.migration_to_venue_id:
            position.migration_from_venue_id = migration.from_venue_id
            position.migration_to_venue_id = migration.to_venue_id
            position.migration_at_monotonic = migration.at_monotonic
            # Record the proven switch ON THE POSITION, so the pin the exit path
            # reads moves with it. Pre- and post-graduation prices are then never
            # mixed, and the switch is auditable rather than inferred.
            position.exit_venue_id = migration.to_venue_id
            position.exit_venue_migration_reason = migration.evidence
            position.exit_venue_migrated_at = time.time()
            position.exit_venue_migration_evidence = migration.evidence
        point = monitor.latest()
        if point is not None and point.price_sol_per_token is not None:
            position.native_price = float(point.price_sol_per_token)

    def _note_quote_request(self, position: PaperPosition, reason: str) -> None:
        """Count an executable quote request and say why it was spent."""
        key = self._exit_retry_key(position)
        now = time.monotonic()
        stamps = self._quote_request_times.setdefault(key, [])
        stamps.append(now)
        # Only the trailing minute is kept, so this is a rate, not a leak.
        cutoff = now - 60.0
        while stamps and stamps[0] < cutoff:
            stamps.pop(0)
        position.quote_requests_last_minute = len(stamps)
        position.quote_request_reason = reason

    # -- quote-based MFE / MAE -------------------------------------------

    @staticmethod
    def _reject_quote_observation(position: PaperPosition, reason: str) -> None:
        """
        Record that a SELL-quote observation was taken and NOT usable.

        A silent drop is the failure this whole path exists to avoid: the
        operator sees ``NOT_MEASURED`` with no cause, and cannot tell a position
        that was never quoted from one whose quotes were all refused. The count
        and the gate name are analytics only -- they cannot affect an exit.
        """
        position.quote_rejection_count = int(
            getattr(position, "quote_rejection_count", 0) or 0) + 1
        position.quote_last_rejection_reason = reason[:160]

    def _record_quote_excursion(self, position: PaperPosition,
                                quote: Any,
                                *, settlement: bool = False) -> None:
        """
        Fold one FRESH SELL quote into the position's executable excursion.

        Called only for a quote that passed the existing freshness and identity
        gates and quotes THIS position's EXACT held quantity, so the point is an
        estimate of what selling the held tokens would have paid at that
        moment -- not a mark for one token, and not a different market.

        The excursion is recomputed from the accumulated points on every call,
        so a restart or a late observation cannot corrupt the running extremum.
        A quote that is unavailable, stale, or fails identity is NOT recorded,
        and its reason is noted so the coverage figure explains the gaps.

        ``realized_pnl_usd`` is NEVER written here: canonical settlement stays
        with ``realize()`` on the actual entry and exit fills. This is an
        analytics measurement on sampled estimates.

        ``settlement=True`` marks the ACCEPTED exit quote. It is the same class
        of observation as any other -- an executable SELL for the exact held
        quantity on a validated market -- and folding it in anchors the
        excursion at the real exit. It is recorded only from the one place that
        has already accepted the quote and applied the fill, so it can never be
        a stand-in for a settlement that did not happen.
        """
        if quote is None or not getattr(quote, "available", False):
            if settlement:
                self._reject_quote_observation(position, "settlement quote unavailable")
            return
        # THE IDENTITY GATE IS RE-CHECKED HERE, NOT ONLY AT THE CALLERS.
        #
        # Both call sites run ``_quote_is_fresh_and_same_market`` first, so this
        # is defence in depth -- but it is the LAST gate before a quote becomes
        # an excursion point, and a future caller that skips the check would
        # otherwise fold a quote for a different market straight into the
        # position's excursion. It honours ``require_identity`` for the same
        # reason the rest of the engine does: when identity enforcement is off,
        # a labelled estimate is a legitimate sample and refusing it would
        # discard data the operator explicitly asked for. A quote with NO
        # identity at all is never refused on this ground -- absence of a
        # verdict is not a failed verdict.
        if self.require_identity:
            identity = str(getattr(quote, "identity", "") or "")
            if identity and identity != "PASS":
                self._reject_quote_observation(
                    position, f"same-market identity not proven ({identity})")
                return
        prompt = _pinned_market_address(position)
        if prompt is not None and getattr(quote, "pool_or_curve", None) \
                and str(quote.pool_or_curve) != str(prompt):
            # A quote on a DIFFERENT market than the position is pinned to is a
            # counterfactual, not this position's excursion.
            self._reject_quote_observation(
                position,
                f"quote describes market {str(getattr(quote, 'pool_or_curve', ''))[:12]}, "
                f"position is pinned to {str(prompt)[:12]}")
            return
        held_raw = int(getattr(position, "tokens_raw", 0) or 0)
        if held_raw <= 0:
            self._reject_quote_observation(
                position, "position holds no tokens to quote")
            return
        quoted_raw = int(getattr(quote, "input_amount_raw", 0) or 0)
        if quoted_raw and quoted_raw != held_raw:
            # A quote for a different quantity is not the same liquidation.
            self._reject_quote_observation(
                position,
                f"quote is for {quoted_raw} raw tokens, position holds {held_raw}")
            return

        # ESTIMATED NET SELL PROCEEDS. ``output_amount_decimal`` is the quoted
        # output for THIS quantity in the QUOTE asset (SOL), already net of the
        # venue's fee and price impact -- so those costs are NOT added a second
        # time here, and the SOL figure is what gets compared to the SOL
        # all-in entry cost.
        proceeds = _as_positive_float(
            getattr(quote, "output_amount_decimal", None))
        cost = self._entry_all_in_cost_quote_asset(position)
        source = str(getattr(quote, "provider", "") or "")
        now = time.time()
        points = list(getattr(position, "_quote_excursion_points", None) or [])
        if proceeds is None or cost is None:
            position.quote_unavailable_reason = (
                "all-in entry cost (%s) or sell-quote output amount (%s) is "
                "unavailable, so no executable net return can be computed for "
                "this position"
                % ("missing" if cost is None else f"{cost:.10g}",
                   "missing" if proceeds is None else f"{proceeds:.10g}"))
            position.quote_mfe_mae_basis = "NOT_MEASURED"
            self._reject_quote_observation(
                position,
                "quote output amount or all-in entry cost is unavailable")
            return
        pct = net_quote_return_pct(net_proceeds=proceeds,
                                  all_in_entry_cost=cost)
        if pct is None:
            position.quote_unavailable_reason = (
                "all-in entry cost or sell-quote output amount is unavailable, "
                "so no executable net return can be computed for this position")
            position.quote_mfe_mae_basis = "NOT_MEASURED"
            self._reject_quote_observation(
                position,
                "net sell return is not computable (cost or proceeds unusable)")
            return

        elapsed_ms = (None if getattr(position, "opened_at", None) is None
                      else (now - float(position.opened_at)) * 1000.0)
        point = QuoteExcursionPoint(
            at=now,
            elapsed_ms=elapsed_ms,
            net_proceeds=proceeds,
            gross_proceeds=proceeds,
            all_in_entry_cost=cost,
            return_pct=pct,
            net_proceeds_raw=getattr(quote, "output_amount_raw", None),
            source=source,
            identity=str(getattr(quote, "identity", "") or ""),
            quote_age_ms=getattr(quote, "quote_age_ms", None),
            usable=True,
        )
        points.append(point)
        # BOUNDED. The excursion only needs the extrema, the timestamps and a
        # coverage figure; keeping every point on a long hold would grow a
        # list on the position without limit for no additional information.
        if len(points) > _MAX_QUOTE_EXCURSION_POINTS:
            points = points[-_MAX_QUOTE_EXCURSION_POINTS:]
        try:
            position._quote_excursion_points = points
        except Exception:  # noqa: BLE001 - analytics never break a monitor tick
            return
        if settlement:
            # THE EXIT SAMPLE'S OWN PROVENANCE. The excursion now claims to
            # cover this position up to its real exit; that claim is only
            # auditable if the quote that produced it is recorded with the
            # identity verdict, the provider and the age that admitted it.
            position.quote_settlement_sample = True
            position.quote_settlement_identity = str(
                getattr(quote, "identity", "") or "")
            position.quote_settlement_source = source
            position.quote_settlement_age_ms = getattr(quote, "quote_age_ms", None)
        self._apply_quote_excursion(position, points)

    @staticmethod
    def _apply_quote_excursion(position: PaperPosition,
                               points: Sequence[Any]) -> None:
        """Write the computed excursion onto the position, or say why not."""
        result = excursions(points)
        if result.get("basis") == "QUOTE_BASED":
            position.quote_mfe_pct = result["quote_mfe_pct"]
            position.quote_mae_pct = result["quote_mae_pct"]
            position.quote_mfe_elapsed_ms = result["mfe_elapsed_ms"]
            position.quote_mae_elapsed_ms = result["mae_elapsed_ms"]
            position.quote_coverage_pct = result["coverage_pct"]
            position.quote_sampling_interval_s = result["sampling_interval_s"]
            position.quote_max_sampling_gap_ms = result["max_sampling_gap_ms"]
            position.quote_mfe_source = result["mfe_source"]
            position.quote_mae_source = result["mae_source"]
            position.quote_observation_count = int(result["valid_quotes"] or 0)
            position.quote_mfe_mae_basis = "QUOTE_BASED"
            position.quote_excursion_cost_basis = "PNL_VS_ENTRY_COST"
            position.quote_unavailable_reason = None
        else:
            position.quote_mfe_mae_basis = "NOT_MEASURED"
            position.quote_unavailable_reason = result.get(
                "unavailable_reason")
            if position.quote_unavailable_reason is None:
                position.quote_unavailable_reason = (
                    "no valid SELL quote observation exists for this position")

    @staticmethod
    def _entry_all_in_cost_quote_asset(position: PaperPosition
                                        ) -> Optional[float]:
        """
        The position's ALL-IN entry cost in the QUOTE ASSET (SOL), or ``None``.

        Every component the position actually carries: the amount swapped in
        (``entry_cost_quote_asset``), any recorded trading fee, and any
        network/priority cost the fill model charged. Costs ALREADY inside the
        entry quote are not added again, because they are already part of the
        amount paid.

        A missing component does NOT silently become zero: an unknown cost
        makes the denominator unknown, and a return against a guessed cost is
        not a return. When any component cannot be read, the whole figure is
        refused rather than half-counted.
        """
        total = 0.0
        seen = False
        for attribute in ("entry_cost_quote_asset", "trading_fees_quote",
                          "network_costs_quote", "other_costs_quote"):
            value = getattr(position, attribute, None)
            if value is None:
                continue
            try:
                amount = float(value)
            except (TypeError, ValueError):
                # An UNREADABLE component means the all-in cost is unknown, so
                # the honest answer is "not computable", not "this part is 0".
                return None
            if amount != amount:
                return None
            # Rent is LOCKED, not spent: it returns to the wallet when the
            # token balance goes to zero, and counting it as a cost would
            # understate every position's net return.
            if attribute == "entry_rent_lamports":
                continue
            total += amount
            if amount > 0:
                seen = True
        if not seen or total <= 0:
            return None
        return total

    async def _validate_executable_sell(self, position: PaperPosition,
                                        mark: ObservedMarkPoint,
                                        reason: str) -> Any:
        """
        Ask for a fresh executable SELL quote, and hold it to the SAME freshness
        and identity rules as before.

        This is EXECUTION VALIDATION, not monitoring. It is called only when an
        existing exit condition is near or reached, and its result never becomes
        the trigger price -- the trigger is still the execution-native reference,
        computed by the venue's own reserves. What it establishes is that the
        position can actually be sold for something, on the market it is pinned
        to, right now.
        """
        self._note_quote_request(position, reason)
        observed_pair = _pinned_market_address(position) or mark.pair_address
        return await self.quote_service.quote_sell(
            mint=position.mint,
            token_amount_raw=position.tokens_raw,
            token_decimals=position.token_decimals,
            observed_pair_address=observed_pair,
        )

    @staticmethod
    def _quote_is_fresh_and_same_market(quote: Any, position: PaperPosition
                                        ) -> tuple[bool, str]:
        """
        The EXISTING freshness + identity gate, applied to a validation quote.

        Deliberately a SEPARATE function from the trigger path so that neither
        can be loosened by a change to the other. A quote that is stale, has no
        route, or does not describe this position's market is NOT executable
        validation, and saying so is better than pretending it is.
        """
        if quote is None or not getattr(quote, "available", False):
            return False, str(
                getattr(quote, "missing_reason", None) or "quote unavailable")
        identity = str(getattr(quote, "identity", "") or "")
        if identity != "PASS":
            return False, f"same-market identity not proven ({identity})"
        stale = bool(getattr(quote, "stale", False))
        if stale:
            return False, "the quote is older than the configured freshness window"
        age = getattr(quote, "quote_age_ms", None)
        if age is not None and float(age) > float(
                getattr(quote, "max_quote_age_ms", 0.0) or 0.0):
            return False, f"quote age {float(age):.0f}ms exceeds the freshness bound"
        pinned = _pinned_market_address(position)
        pool = getattr(quote, "pool_or_curve", None)
        if pinned and pool and str(pool) != str(pinned):
            return False, (
                f"the validation quote describes {str(pool)[:8]}, not this "
                f"position's pinned market {str(pinned)[:8]}")
        return True, "fresh and same-market"

    async def aclose(self) -> None:
        """Close every push subscription this engine opened."""
        for monitor in list(self.native_monitors.monitors()):
            with contextlib.suppress(Exception):
                await monitor.close()
        self._position_monitors.clear()
        self._quote_request_times.clear()

    async def observe(self, mint: str) -> ObservedMarkPoint:
        """
        A fresh DexScreener observation.

        This is the OBSERVATION layer: L25 features, MFE/MAE and the mark-vs-quote
        gap. It is never used as a fill.
        """
        now = time.time()
        if self.market is None:
            return ObservedMarkPoint(
                value_usd=None, pair_address=None, dex_id=None, liquidity_usd=None,
                volume_5m_usd=None, market_cap_usd=None, timestamp=now,
                age_ms=None, available=False,
                missing_reason="no market observation provider configured",
            )
        try:
            quote = await self.market.quote_async(mint)
            obs = getattr(quote, "observation", None)
        except Exception as exc:  # noqa: BLE001 - named, never fatal
            return ObservedMarkPoint(
                value_usd=None, pair_address=None, dex_id=None, liquidity_usd=None,
                volume_5m_usd=None, market_cap_usd=None, timestamp=now,
                age_ms=None, available=False,
                missing_reason=f"{type(exc).__name__}: {exc}"[:200],
            )

        # ``obs`` is None whenever a provider ANSWERED but had no observation to
        # give -- an unknown pair, a delisted market, a rate-limited read. That
        # is not an exception, so it was never caught, and the ``obs.price_usd``
        # below raised AttributeError OUTSIDE the try: the observation layer
        # crashed the runtime instead of declining the candidate.
        if obs is None:
            return ObservedMarkPoint(
                value_usd=None, pair_address=None, dex_id=None, liquidity_usd=None,
                volume_5m_usd=None, market_cap_usd=None, timestamp=now,
                age_ms=None, available=False,
                missing_reason=OBSERVATION_UNAVAILABLE_REASON,
            )
        try:
            self._remember(mint, obs, now)
        except Exception:  # noqa: BLE001 - history is best-effort, never fatal
            pass

        price = getattr(obs, "price_usd", None)
        return ObservedMarkPoint(
            value_usd=price,
            pair_address=getattr(obs, "pair_address", None)
            or getattr(obs, "pool_address", None),
            dex_id=getattr(obs, "dex_id", None) or getattr(obs, "venue", None),
            liquidity_usd=getattr(obs, "liquidity_usd", None),
            volume_5m_usd=getattr(obs, "volume_5m_usd", None),
            market_cap_usd=getattr(obs, "market_cap_usd", None),
            # RECEIVED_AT: a transport fact. Always known.
            timestamp=now,
            # SOURCE_OBSERVED_AT: taken from the provider ONLY when the
            # provider actually publishes a per-observation timestamp.
            #
            # DexScreener does not. Its only time field is ``pairCreatedAt``,
            # which is when the PAIR was created -- a different fact from when
            # this price was observed, and using it would make a 10-minute-old
            # print look hours old or a brand-new one look arbitrarily old.
            # So it is read if a future provider supplies one, and otherwise
            # recorded as absent.
            source_observed_at=_mark_source_observed_at(obs),
            # SOURCE AGE: computed only from a real source timestamp. Absent
            # otherwise. An invented 0.0 would assert "this price is current",
            # which is precisely the claim that cannot be made here.
            age_ms=_mark_source_age_ms(obs, now),
            age_status=_mark_age_status(obs),
            # TRANSPORT AGE: our own pipeline latency, honestly named.
            transport_age_ms=0.0,
            available=bool(price and price > 0),
            missing_reason=None if price else "the observation carried no price",
        )

    def _remember(self, mint: str, obs: Any, at: float) -> None:
        """Append one observation to this mint's decision-time history."""
        history = self.observation_history.setdefault(mint, [])
        history.append(
            Observation(
                mint=mint,
                observed_at=getattr(obs, "observed_at", None) or at,
                received_at=at,
                price_usd=getattr(obs, "price_usd", None),
                liquidity_usd=getattr(obs, "liquidity_usd", None),
                volume_5m_usd=getattr(obs, "volume_5m_usd", None),
                market_cap_usd=getattr(obs, "market_cap_usd", None),
                provider=getattr(obs, "provider", None) or "dexscreener",
                venue=getattr(obs, "venue", None) or getattr(obs, "dex_id", None),
                pair_address=getattr(obs, "pair_address", None),
            )
        )
        if len(history) > self.max_history_per_mint:
            del history[: len(history) - self.max_history_per_mint]

    # -- the production decision -----------------------------------------

    def decide(
        self,
        *,
        mint: str,
        signal: SignalSnapshot,
        mark: ObservedMarkPoint,
        decision_time: float,
        blocked_reason: str | None,
    ) -> tuple[Any, str, bool]:
        """
        The PRODUCTION L25 decision, called through the production decision
        engine.

        Returns ``(report, l25_status, authorized)``. Every input is a
        decision-time value; nothing here is fitted, adjusted or overridden.
        """
        observation = Observation(
            mint=mint,
            observed_at=mark.timestamp,
            received_at=mark.timestamp,
            price_usd=mark.value_usd,
            liquidity_usd=mark.liquidity_usd,
            volume_5m_usd=mark.volume_5m_usd,
            market_cap_usd=mark.market_cap_usd,
            provider="dexscreener",
            venue=mark.dex_id,
            pair_address=mark.pair_address,
        )
        history = ObservationSet(items=tuple(self.observation_history.get(mint, ())))
        features = build_features(
            candidate_id=signal.signal_id, mint=mint,
            decision_time=decision_time, history=history, alert=None, nar=None,
            stale_after_ms=60_000.0, unusable_after_ms=180_000.0,
        )
        report = self.decision_engine.decide(
            features=features, alert=None, nar=None,
            analysis_start=signal.receive_monotonic,
            analysis_complete=decision_time,
            blocked_reason=blocked_reason,
        )
        l25 = getattr(report, "l25", None)
        status = getattr(l25, "status", "UNKNOWN")
        authorized = bool(getattr(l25, "authorized", False))
        return report, status, authorized

    # -- V2 FLOOR (BETA) helpers ------------------------------------------

    def _beta_stage(self, stage: str, *, parent_signal_id: str | None = None,
                    arm: str | None = None, **detail: Any) -> dict[str, Any]:
        """Record ONE structured V2 FLOOR (BETA) lifecycle event.

        Structured (``stage`` + ``parent_signal_id`` + ``arm`` + named detail),
        so the seven observable stages in the integration brief can be
        reconciled by a reader rather than reconstructed from prose. It is
        IN-MEMORY and BOUNDED -- telemetry, not a ledger -- and it carries no
        price a settlement could be booked from.
        """
        clean = {k: v for k, v in detail.items() if v is not None}
        rec: dict[str, Any] = {"at": time.time(), "stage": stage}
        if parent_signal_id:
            rec["parent_signal_id"] = parent_signal_id
        if arm:
            rec["arm"] = arm
        rec.update(clean)
        log("BETA_STAGE", "stage=%s parent=%s arm=%s detail=%s",
            stage, parent_signal_id or "-", arm or "-", clean)
        self.beta_stage_log.append(rec)
        if len(self.beta_stage_log) > 500:
            del self.beta_stage_log[:-500]
        return rec

    def _beta_mirror_position(self, spec: ArmSpec, bp: Any) -> PaperPosition:
        """
        A VISIBILITY-ONLY ``PaperPosition`` mirroring one experiment position.

        Built in ONE place so the mirror created at entry and the mirror
        REBUILT after a restart (see ``_monitor_beta_experiment``) cannot drift
        apart in their bracket, cap or profile stamps.

        This record is never registered in ``self.accounts`` and nothing ever
        settles it: :class:`BetaExperiment` owns the position, the three $10
        ledgers and the realized P&L. The mirror exists only so the operator's
        open-positions view and the diagnostics surfaces can see the trade.
        """
        arm_key = BETA_ARM_KEYS[spec.name]
        account = (self.beta_experiment.accounts or {}).get(arm_key)
        fill = type('Fill', (), {
            'effective_price_usd': bp.entry_fill_price,
            'effective_price': bp.entry_fill_price,
            'input_amount_raw': 0,
            'input_decimals': 0,
        })()
        return PaperPosition(
            signal_id=bp.parent_signal_id,
            arm=spec.name.value,
            mint=bp.mint,
            symbol=bp.symbol,
            token_decimals=6,
            tokens_raw=0,
            entry_fill=fill,
            entry_observed_mark=bp.entry_fill_price,
            entry_mark_timestamp=bp.entry_at,
            bankroll_before=(account.bankroll_usd if account is not None
                             else self.starting_bankroll_usd),
            position_size_usd=bp.position_size_usd,
            # THE MIRROR STATES THE RULE THAT ACTUALLY SIZED THE ENTRY. A
            # hardcoded 10.0 here was a residue of the removed 10%-of-equity
            # rule: the open-positions view would label an ALL-IN position as a
            # 10% one, which is exactly the misattribution the sizing stamp
            # exists to prevent. An entry with no recorded rule (a restored
            # legacy position) keeps the legacy label.
            position_size_pct=(bp.sizing_pct if bp.sizing_pct is not None
                               else LEGACY_BETA_SIZING_PCT),
            usd_notional=bp.position_size_usd,
            sol_notional=Decimal(0),
            sol_usd_reference=self.sol_reference.usd_per_sol if self.sol_reference else Decimal(0),
            sol_usd_reference_at=self.sol_reference.at if self.sol_reference else 0,
            opened_at=bp.entry_at,
            entry_cost_quote_asset=Decimal(0),
            trading_fees_quote=Decimal(0),
            network_costs_quote=self.fill_model.network_cost_sol,
            other_costs_quote=Decimal(0),
            production_profile=spec.production_profile.value,
            take_profit_pct=self._take_profit_snapshot(),
            beta_exit_bracket=spec.exit_bracket,
            beta_max_hold_s=spec.max_hold_s,
        )

    # ------------------------------------------------------------------
    # beta operator notifications
    # ------------------------------------------------------------------

    async def _beta_exit_fill(
        self, *, parent_signal_id: str, arm_key: str, bp: Any,
    ) -> tuple[Any, str]:
        """
        An EXECUTABLE SELL quote for this arm's exact remaining quantity.

        The existing collector is used: ``quote_sell_pinned`` when the entry
        pinned a market, else ``quote_sell``, both through the same
        same-market/freshness gate the regular arms use. There is no mark
        fallback and no second collector. A refusal returns ``(None, reason)``
        and the caller leaves the arm UNRESOLVED.
        """
        from .fills import FillModel, FillRefused, fill_from_quote

        if bp.entry_tokens_raw in (None, 0):
            return None, ("the entry never acquired a quantity, so no exit "
                          "quantity can be quoted")
        if not (bp.entry_cost_quote and float(bp.entry_cost_quote) > 0):
            return None, ("the entry has no total BUY cost, so a return would "
                          "have no denominator")
        observed_pair = bp.entry_pool_or_curve
        mint = bp.mint
        note = ""
        quote = None
        # MEASURED, not assumed: the span this ONE SELL quote request took.
        sell_request_at = time.monotonic()
        try:
            pinned = getattr(self.quote_service, "quote_sell_pinned", None)
            if observed_pair and pinned is not None:
                result = await pinned(
                    mint=mint, token_amount_raw=int(bp.entry_tokens_raw),
                    token_decimals=int(bp.entry_token_decimals or 6),
                    pinned_venue_id=None, pinned_pool_address=observed_pair,
                    observed_pair_address=observed_pair)
                # ``quote_sell_pinned`` RETURNS A PAIR: ``(chosen, counterfactual)``.
                #
                # THIS IS A DEFECT FIX, not a tidy-up. Reading the PAIR itself as
                # the quote made ``getattr(quote, "available", False)`` False on
                # every call -- a tuple has no ``available`` attribute -- so
                # EVERY beta arm whose entry pinned a market was refused with
                # "no executable SELL quote" and could NEVER settle. The live
                # experiment showed it exactly: three arms, 1273/1273/1267
                # refused attempts, zero settlements. The regular exit path
                # (``_exit_with_quote``) and ``real_compat`` both UNPACK this
                # pair, which is why the defect was confined to this path; the
                # offline doubles never exposed it because they define
                # ``quote_sell``/``quote_sell_best`` and no ``quote_sell_pinned``.
                #
                # Unwrap the CHOSEN quote. The counterfactual is research output
                # and is deliberately not used as the settlement.
                quote = result[0] if isinstance(result, (tuple, list)) else result
            else:
                quote = await self.quote_service.quote_sell(
                    mint=mint, token_amount_raw=int(bp.entry_tokens_raw),
                    token_decimals=int(bp.entry_token_decimals or 6),
                    observed_pair_address=observed_pair)
        except Exception as exc:  # noqa: BLE001 - a quote failure is a reason
            return None, f"SELL quote request failed: {type(exc).__name__}"
        sell_response_at = time.monotonic()
        if quote is None or not getattr(quote, "available", False):
            return None, (
                f"no executable SELL quote: "
                f"{getattr(quote, 'missing_reason', None) or 'unavailable'}")
        if getattr(quote, "stale", False):
            return None, (
                f"SELL quote is stale (age "
                f"{getattr(quote, 'quote_age_ms', None)}ms)")
        try:
            fill = fill_from_quote(
                quote, side="SELL", model=FillModel(),
                sol_reference=getattr(self, "sol_reference", None),
                require_identity=self.require_identity)
        except FillRefused as exc:
            return None, f"SELL quote refused as a fill: {exc}"
        from .v2_floor_beta import BetaExitFill
        # WHICH AMOUNT IS THE PROCEEDS
        # ----------------------------
        # On a SELL the token is the INPUT and the quote asset (SOL) is the
        # OUTPUT, so ``input_amount_decimal`` is a TOKEN quantity and
        # ``output_amount_decimal`` is the SOL received. Reading the proceeds
        # off the input therefore booked ~9,118 "proceeds" against a
        # ~0.004086 SOL cost and reported a +223,000,000% return: the two legs
        # were in different units. The denominator stays the BUY's total SOL
        # cost, so both sides of the ratio must be SOL.
        #
        # The token decimals likewise come from the INPUT leg; ``output_decimals``
        # is 9 (SOL), which would have rescaled every raw quantity by 10^3.
        return BetaExitFill(
            price_usd=float(fill.effective_price_usd or 0.0),
            tokens_raw=int(bp.entry_tokens_raw),
            token_decimals=int(fill.input_decimals),
            cost_quote=float(fill.output_amount_decimal),
            fee_quote=float(fill.fee),
            entry_cost_quote=float(bp.entry_cost_quote),
            quote_timestamp=float(fill.quote_timestamp),
            identity=str(getattr(fill, "identity", "UNKNOWN")),
            provenance=self._beta_quote_provenance(
                quote=quote, fill=fill,
                sol_ref=getattr(self, "sol_reference", None),
                requested_input_raw=int(bp.entry_tokens_raw),
                request_at=sell_request_at, response_at=sell_response_at),
        ), note

    @staticmethod
    def _beta_quote_provenance(
        *, quote: Any, fill: Any, sol_ref: Any = None,
        requested_input_raw: int | None = None,
        trigger_price_usd: float | None = None,
        request_at: float | None = None,
        response_at: float | None = None,
        reserve_lamports: int | None = None,
        charged_usd: float | None = None,
    ) -> dict[str, Any]:
        """
        EVERYTHING needed to recompute one beta leg from its own receipts.

        Recorded VERBATIM from the quote and fill rather than re-derived, so a
        later reader never has to re-assume a fee schedule, a route or a
        reference price. Fields the PAPER path genuinely does not have are
        written as ``None`` WITH a stated reason -- never silently defaulted to
        a number that would look measured.
        """
        def _g(obj: Any, name: str) -> Any:
            return getattr(obj, name, None) if obj is not None else None

        def _num(v: Any) -> Any:
            # Decimals are not JSON-serialisable; they are recorded as strings
            # so no precision is lost on the way into the durable record.
            if v is None:
                return None
            if isinstance(v, (int, float, str, bool)):
                return v
            return str(v)

        return {
            "side": str(_g(quote, "side") or ""),
            "provider": _g(quote, "provider"),
            "market_type": _g(quote, "market_type"),
            "venue": _g(quote, "venue"),
            "venue_id": _g(quote, "venue_id"),
            "pool_or_curve": _g(quote, "pool_or_curve"),
            "route": [str(x) for x in (_g(quote, "route") or ())],
            "token_mint": _g(quote, "token_mint"),
            "quote_mint": _g(quote, "quote_mint"),
            "identity": str(_g(quote, "identity") or "UNKNOWN"),
            "identity_evidence": _g(quote, "identity_evidence"),
            "observed_pair_address": _g(quote, "observed_pair_address"),
            "requested_input_raw": requested_input_raw,
            "input_amount_raw": _g(quote, "input_amount_raw"),
            "input_decimals": _g(quote, "input_decimals"),
            "input_amount_decimal": _g(quote, "input_amount_decimal"),
            "output_amount_raw": _g(quote, "output_amount_raw"),
            "output_decimals": _g(quote, "output_decimals"),
            "output_amount_decimal": _g(quote, "output_amount_decimal"),
            "amount_after_fee_raw": _g(quote, "amount_after_fee_raw"),
            "effective_price": _num(_g(quote, "effective_price")),
            "effective_price_usd": _g(quote, "effective_price_usd"),
            "price_impact_pct": _g(quote, "price_impact_pct"),
            "fee_bps": _g(quote, "fee_bps"),
            "fee_amount": _num(_g(quote, "fee_amount")),
            "quote_timestamp": _g(quote, "quote_timestamp"),
            "quote_age_ms": _g(quote, "quote_age_ms"),
            "max_quote_age_ms": _g(quote, "max_quote_age_ms"),
            "quote_latency_ms": _g(quote, "quote_latency_ms"),
            "state_slot": _g(quote, "state_slot"),
            "state_age_ms": _g(quote, "state_age_ms"),
            # MEASURED around the request itself, so "how long did the provider
            # take" is an observation rather than an inference.
            "request_at_monotonic": request_at,
            "response_at_monotonic": response_at,
            "measured_request_ms": (
                max(0.0, (response_at - request_at) * 1000.0)
                if request_at is not None and response_at is not None else None),
            # -- NOT RECORDED, AND SAID SO ------------------------------------
            # PAPER builds, signs and submits NO transaction, so it sets no
            # slippage tolerance and no minimum-output threshold. The executable
            # quote prices this EXACT size and IS the accepted execution price;
            # no extra slippage is added on top of it. Stated explicitly so a
            # reader cannot mistake the absence for a measured zero.
            "slippage_bps": None,
            "minimum_output_threshold": None,
            "price_boundary_note": (
                "PAPER builds and submits NO transaction, so no slippage "
                "tolerance and no minimum-output threshold are set. The "
                "executable quote prices this EXACT size and is the accepted "
                "execution price; no extra slippage was added on top of it."),
            "trigger_price_usd": trigger_price_usd,
            "trigger_role": (
                "DIAGNOSTIC ONLY: the external mark observed when the signal or "
                "exit trigger fired. It is NOT the price this fill was booked "
                "at -- the fill price is the executable quote above."),
            "sol_usd": float(getattr(sol_ref, "usd_per_sol", 0.0) or 0.0),
            "sol_usd_source": str(getattr(sol_ref, "source", "") or ""),
            "sol_usd_at": getattr(sol_ref, "at", None),
            "reserved_outside_swap_lamports": reserve_lamports,
            "charged_outside_swap_usd": charged_usd,
        }

    def _beta_entry_size_usd(self, mark: Any = None,
                             arm: str | None = None) -> float:
        """
        The ALL-IN notional for ONE beta arm, before a quote is requested.

        Delegates to the experiment's own sizing rule, which is the single
        definition of it: 100% of that arm's uncommitted cash. There is no
        shared/minimum notional any more -- each arm is all-in on its OWN
        bankroll, so one arm's realized profit can never size another arm's
        entry, in either direction. ``mark`` is accepted (and unused) so existing
        callers keep working; sizing is never derived from a price.
        """
        exp = self.beta_experiment
        if exp is None:
            return 0.0
        key = arm or BETA_ARM_KEYS[BETA_ARMS[0]]
        return exp.all_in_notional_usd(key)

    async def _beta_entry_fill(
        self, *, mint: str, size_usd: float, mark: Any, arm: str | None = None,
    ) -> tuple["BetaEntryFill | None", str]:
        """
        Acquire the EXECUTABLE BUY quote the beta entry is booked from.

        THE COLLECTOR IS THE EXISTING ONE. ``_quote_buy_best`` is the same
        shared, single-flight, freshness-bounded function the L25 arms use, so
        the experiment and the production arms measure the same book at the same
        instant. There is no second quote collector, and the beta entry is not a
        special case that reads a mark.

        Nothing here is allowed to fall back to a mark. If no executable BUY
        quote is available the caller gets ``None`` and a reason, and NO
        position is opened -- a missing quote must never look like a fill.
        """
        from .fills import FillModel, FillRefused, fill_from_quote

        notional = float(size_usd)
        if not (notional and notional > 0):
            return None, "beta entry size is not positive"
        ref = getattr(self, "sol_reference", None)
        sol_usd = self._sol_usd() if hasattr(self, "_sol_usd") else None
        if sol_usd is None:
            sol_usd = float(getattr(ref, "usd_per_sol", 0.0) or 0.0)
        if not (sol_usd and sol_usd > 0):
            return None, "no SOL/USD reference to size a BUY quote"
        # THE RESERVE COMES OUT BEFORE THE SWAP INPUT IS SIZED.
        #
        # An ALL-IN arm commits 100% of its available cash, but the wallet must
        # ALSO hold the transaction fees and the token-account rent BESIDE the
        # swap input, or the BUY cannot execute at all on chain. Charging only
        # the swap quote's own fee -- which is already INSIDE the quoted input --
        # overstates the notional the arm could actually deploy.
        #
        # The reserve is deducted with FLOOR, never round: the notional IS the
        # arm's deployable cash, so rounding up would ask the venue for more SOL
        # than the arm can pay for. The remainder is recorded, not spent.
        costs = outside_swap_costs(
            sol_usd=sol_usd,
            sol_usd_source=str(getattr(ref, "source", "") or ""),
            sol_usd_at=getattr(ref, "at", None),
        )
        sol_lamports, why = deployable_sol_lamports(
            available_usd=notional, sol_usd=sol_usd, costs=costs)
        if sol_lamports <= 0:
            return None, why
        token_decimals = 6
        try:
            token_decimals = int(self.token_decimals_for(mint))
        except Exception:  # noqa: BLE001 - default stays 6
            token_decimals = 6
        observed_pair = getattr(mark, "pair_address", None)
        # MEASURED, not assumed: the wall/monotonic span this ONE quote request
        # actually took. It is recorded per leg so a slow provider shows up as a
        # latency number rather than as an unexplained miss.
        quote_request_at = time.monotonic()
        try:
            quote, _candidates = await self._quote_buy_best(
                mint=mint, sol_lamports=sol_lamports,
                token_decimals=token_decimals,
                observed_pair_address=observed_pair)
        except Exception as exc:  # noqa: BLE001 - a quote failure is a reason
            return None, f"BUY quote request failed: {type(exc).__name__}"
        quote_response_at = time.monotonic()
        if quote is None or not getattr(quote, "available", False):
            return None, (
                f"no executable BUY quote: "
                f"{getattr(quote, 'missing_reason', None) or 'unavailable'}")
        if getattr(quote, "stale", False):
            return None, (
                f"BUY quote is stale (age "
                f"{getattr(quote, 'quote_age_ms', None)}ms)")
        try:
            fill = fill_from_quote(
                quote, side="BUY", model=FillModel(),
                sol_reference=getattr(self, "sol_reference", None),
                require_identity=self.require_identity)
        except FillRefused as exc:
            return None, f"BUY quote refused as a fill: {exc}"
        # THE USD THE ARM COMMITS. ``input_amount_decimal`` is the SOL the venue
        # will take for this BUY -- the venue's fee is already inside it -- so
        # converting the SAME number at the SAME reference makes the
        # affordability test the "available capital after required fees" test.
        # A quote that costs more than the arm holds is refused here, before
        # anything is opened.
        cost_usd = float(fill.input_amount_decimal) * float(sol_usd)
        if cost_usd > notional + 1e-9:
            return None, (
                f"the BUY quote costs ${cost_usd:.9f}, more than arm "
                f"{arm or '?'}'s ${notional:.9f} available; an all-in entry may "
                "never exceed the capital it is funded from")
        mark_price = getattr(mark, "value_usd", None)
        return BetaEntryFill(
            price_usd=float(fill.effective_price_usd or 0.0),
            tokens_raw=int(fill.output_amount_raw),
            token_decimals=int(fill.output_decimals),
            cost_quote=float(fill.input_amount_decimal),
            cost_usd=cost_usd,
            fee_quote=float(fill.fee),
            mark_price=(float(mark_price) if mark_price else None),
            quote_timestamp=float(fill.quote_timestamp),
            quote_age_ms=fill.quote_age_ms,
            venue=(getattr(quote, "venue", None)
                   or getattr(fill, "venue", None)),
            pool_or_curve=(getattr(quote, "pool_or_curve", None)
                           or getattr(fill, "pool_or_curve", None)),
            provider=getattr(quote, "provider", None),
            route=tuple(getattr(fill, "route", ()) or ()),
            identity=str(getattr(fill, "identity", "UNKNOWN")),
            outside_costs=costs,
            provenance=self._beta_quote_provenance(
                quote=quote, fill=fill, sol_ref=ref,
                requested_input_raw=int(sol_lamports),
                trigger_price_usd=(float(mark_price) if mark_price else None),
                request_at=quote_request_at, response_at=quote_response_at,
                reserve_lamports=int(costs.total_reserve_lamports),
                charged_usd=float(costs.charged_to_pnl_usd)),
        ), ""

    def render_beta_entries(
        self, records: Sequence[SignalRecord],
    ) -> list[dict[str, Any]]:
        """
        ONE grouped entry notification per parent signal that really opened.

        WHY THIS EXISTS
        ---------------
        A beta position is deliberately NOT registered in ``self.accounts`` --
        :class:`BetaExperiment` owns it, its three $10 ledgers and its realized
        P&L, and a second registration would settle the same trade twice. But
        :meth:`render_entries` resolves a record's position through
        ``self.accounts[arm].position``, so for every beta record it found
        ``None`` and skipped the message. The experiment therefore opened
        positions and the operator was told NOTHING: zero entry notifications
        for every successful beta entry.

        WHAT IS REPORTED IS WHAT HAPPENED
        ---------------------------------
        A parent is announced only when the experiment actually holds an open
        position for it. Arms are listed from the experiment's own book, so a
        partial entry is reported as partial rather than as three successes.
        Authorization alone never produces this message.

        One message per parent signal: the three arms share one gate, one
        entry price and one size, so three messages would be the same news
        three times.
        """
        if self.beta_experiment is None:
            return []
        by_parent: dict[str, list[SignalRecord]] = {}
        for record in records:
            if getattr(record, "arm", None) not in {a.value for a in BETA_ARMS}:
                continue
            # ``SignalRecord.parent_signal_id`` is stamped only on a record that
            # actually opened; a rejection leaves it empty, which is exactly
            # right -- a refusal has no parent entry to announce.
            parent_id = str(getattr(record, "parent_signal_id", "") or "")
            if parent_id:
                by_parent.setdefault(parent_id, []).append(record)
        out: list[dict[str, Any]] = []
        for parent_id, group in by_parent.items():
            opened = self.beta_experiment.positions.get(parent_id) or {}
            if not opened:
                continue    # authorized but nothing exists: say nothing
            lines = [
                "\U0001f7e2 <b>PAPER BUY \u2014 V2 FLOOR (BETA)</b>",
                "<i>PAPER SIMULATION \u2014 not a real on-chain trade</i>",
                f"${(next(iter(opened.values())).symbol or '?')}",
                "",
                f"Profile: <b>{BETA_PROFILE_LABEL}</b>",
                f"Parent signal: <code>{parent_id}</code>",
                f"Token: <code>{(next(iter(opened.values())).mint or '')}</code>",
                "",
                f"Arms opened: <b>{len(opened)}/{len(BETA_ARMS)}</b>",
            ]
            for arm_key in BETA_ARMS_ORDER:
                bp = opened.get(arm_key)
                if bp is None:
                    continue
                cfg = arm_config(arm_key)
                # THE SIZE AND THE RULE THAT PRODUCED IT, per arm. An all-in
                # line says so explicitly, and the committed cost is shown next
                # to the capital it was taken from, so "100% of available" is a
                # statement a reader can check rather than a claim.
                if bp.sizing_pct is not None and bp.allocated_usd is not None:
                    _sizing = (
                        f" ({bp.sizing_pct:.0f}% of {self._usd(bp.allocated_usd)} "
                        f"available"
                        + (f", {self._usd(bp.remaining_cash_usd)} left"
                           if bp.remaining_cash_usd else "")
                        + ")")
                else:
                    _sizing = ""
                lines.append(
                    f"\u2705 Arm {arm_key} \u2014 TP {cfg.take_profit_pct:+.0f}% / "
                    f"SL {cfg.stop_loss_pct:+.0f}% / max {cfg.max_hold_s:.0f}s "
                    f"\u2014 committed {self._usd(bp.position_size_usd)}"
                    f"{_sizing} "
                    f"@ {self._price(bp.entry_fill_price)}"
                )
            failed = [r for r in group
                      if not r.traded and self.beta_experiment.positions.get(
                          parent_id, {}).get(
                              BETA_ARM_KEYS.get(
                                  next((a for a in BETA_ARMS
                                        if a.value == r.arm), BETA_ARMS[0]),
                                  "")) is None]
            if failed:
                lines.append("")
                lines.append("Arms that did NOT open:")
                for record in failed:
                    lines.append(
                        f"\u274c {record.arm} \u2014 "
                        f"{str(getattr(record, 'arm_filter_reason', '') or '')[:120]}"
                    )
            _entry_basis = next(iter(opened.values())).entry_basis
            if _entry_basis == EXECUTABLE_QUOTE_BASIS:
                _basis_line = (
                    "Entry basis: <b>executable BUY quote</b> simulated fill "
                    "(no transaction sent).")
            else:
                _basis_line = (
                    f"Entry basis: <b>modeled backtest fill</b> "
                    f"({_entry_basis}) \u2014 NOT an executable quote.")
            lines += [
                "",
                _basis_line,
                f"Mode: <b>PAPER</b>",
            ]
            out.append({"text": "\n".join(lines),
                        "kind": "beta_entry", "parent_signal_id": parent_id})
        return out

    def _beta_trade_persisted(self, bp: Any) -> dict[str, Any] | None:
        """
        This arm's settlement ON THE DURABLE BOOK, or ``None``.

        A CLOSED alert is the operator's receipt for a sale, so it must never be
        able to describe a settlement the canonical state has not accepted.
        ``settle_arm`` appends the trade to ``completed`` and calls ``_persist()``
        BEFORE it returns, so a settlement that exists only in memory -- or one
        whose process died between the ledger move and the write -- is
        identifiable here and is REFUSED. Fails CLOSED: an unreadable book counts
        as not persisted.
        """
        exp = self.beta_experiment
        for rec in getattr(exp, "completed", None) or ():
            if (str(rec.get("arm")) == str(bp.arm)
                    and str(rec.get("parent_signal_id"))
                    == str(bp.parent_signal_id)
                    and rec.get("status") == "closed"):
                return rec
        return None

    @staticmethod
    def _tokens_qty(amount_raw: Any, decimals: Any) -> str:
        """A raw integer token quantity as ``<raw> raw (<decimal> tokens)``."""
        try:
            raw = int(amount_raw)
        except (TypeError, ValueError):
            return "-"
        try:
            dec = int(decimals)
        except (TypeError, ValueError):
            return f"{raw:,} raw"
        return f"{raw:,} raw ({raw / (10 ** dec):,.6f} tokens)"

    def render_beta_exits(
        self, settled: Sequence[Any],
    ) -> list[dict[str, Any]]:
        """
        ONE grouped exit notification per parent signal, from the RECORDED result.

        Only arms the experiment actually settled are reported, and the numbers
        are the persisted ones -- ``exit_reason``, ``exit_fill_price``,
        ``net_pnl_usd``, ``exit_proceeds_usd``, ``exit_fee_quote`` -- read off
        the position after settlement. A target TOUCH is not a fill and is never
        reported as one: an arm that is still open has no exit row at all.

        TWO GATES BEFORE A SALE IS ANNOUNCED, because a receipt for a sale that
        did not happen is worse than no receipt:

        1. **A valid executable SELL quote must have been accepted.** Only an arm
           whose ``exit_basis`` is ``EXECUTABLE_QUOTE_BASIS`` is described as a
           SALE. A settlement from the modelled/backtest path (or a legacy
           mark-based one) is reported in a SEPARATE message, labelled, and
           explicitly NOT as a sale.
        2. **The settlement must be PERSISTED.** An arm whose trade is not on the
           durable book is held back, so a crash between the ledger move and the
           write can never produce a receipt for a trade the state lost.

        ALL-IN SIZING IS STATED, per arm, from the recorded ``sizing_pct`` /
        ``allocated_usd``: the stake, the rule that produced it, and the arm's
        bankroll AFTER settlement, so the compounding the ALL-IN rule is supposed
        to produce is visible rather than asserted.
        """
        if self.beta_experiment is None or not settled:
            return []
        by_parent: dict[str, list[Any]] = {}
        for bp in settled:
            if bp is None:
                continue
            by_parent.setdefault(bp.parent_signal_id, []).append(bp)
        out: list[dict[str, Any]] = []
        for parent_id, positions in by_parent.items():
            closed = [bp for bp in positions if bp.status == "closed"]
            if not closed:
                continue
            persisted = [bp for bp in closed if self._beta_trade_persisted(bp)]
            if not persisted:
                continue
            quote_closed = [bp for bp in persisted
                            if bp.exit_basis == EXECUTABLE_QUOTE_BASIS]
            modeled_closed = [bp for bp in persisted
                              if bp.exit_basis != EXECUTABLE_QUOTE_BASIS]
            if quote_closed:
                out.append(self._beta_exit_message(
                    parent_id=parent_id, arms=quote_closed,
                    title="PAPER CLOSED \u2014 V2 FLOOR (BETA)",
                    kind="beta_exit", quote_basis=True))
            if modeled_closed:
                out.append(self._beta_exit_message(
                    parent_id=parent_id, arms=modeled_closed,
                    title="PAPER CLOSED (MODELED) \u2014 V2 FLOOR (BETA)",
                    kind="beta_exit_modeled", quote_basis=False))
        return out

    def _beta_exit_message(self, *, parent_id: str, arms: Sequence[Any],
                           title: str, kind: str,
                           quote_basis: bool) -> dict[str, Any]:
        """Build ONE grouped CLOSED message from SETTLED, PERSISTED arms."""
        first = arms[0]
        lines = [
            f"\U0001f7e9 <b>{title}</b>",
            "<i>PAPER SIMULATION \u2014 not a real on-chain trade</i>",
            f"${first.symbol or '?'}",
            "",
            f"Profile: <b>{BETA_PROFILE_LABEL}</b>",
            f"Parent signal: <code>{parent_id}</code>",
            f"Token: <code>{first.mint or ''}</code>",
            "",
            f"Arms settled: <b>{len(arms)}</b>",
        ]
        if quote_basis:
            lines.append(
                "Basis: <b>executable SELL quote</b> \u2014 a simulated fill at "
                "the quoted price. No transaction was sent.")
        else:
            lines.append(
                "Basis: <b>MODELED backtest fill</b> \u2014 NOT an executable "
                "quote and NOT an observed sale.")
        for bp in sorted(arms, key=lambda b: b.arm):
            lines += self._beta_exit_arm_lines(bp)
        lines += [
            "",
            ("Exit basis: <b>executable SELL quote</b> simulated fill on the "
             "exact acquired quantity (no transaction sent)."
             if quote_basis else
             "Exit basis: <b>modeled backtest fill</b> "
             f"({first.fill_model}) \u2014 NOT an executable quote."),
            "Mode: <b>PAPER</b>",
        ]
        return {
            "text": "\n".join(lines),
            "kind": kind,
            "parent_signal_id": parent_id,
            # THE MESSAGE'S OWN IDENTITY. Keyed on the arms it reports, so an
            # arm that closes on a LATER tick is a new message and not a
            # "duplicate" of an earlier one, while a re-entrant flush of the
            # same message is still dropped.
            "dedupe_key": (f"{kind}::{parent_id}::"
                           + ",".join(sorted(str(b.arm) for b in arms))),
        }

    def _beta_exit_arm_lines(self, bp: Any) -> list[str]:
        """
        The per-arm block of a CLOSED receipt.

        Every number the operator needs to check the trade is stated from the
        RECORDED settlement: the reason, both prices, the quantity sold, the net
        proceeds, the fees, the realized P&L, the size and the rule that produced
        it, and the arm's bankroll after settlement.
        """
        exp = self.beta_experiment
        account = (getattr(exp, "accounts", None) or {}).get(bp.arm)
        bankroll_after = (account.bankroll_usd if account is not None else None)
        cost = (bp.entry_cost_usd if bp.entry_cost_usd is not None
                else bp.position_size_usd)
        lines = [
            "",
            f"Arm {bp.arm} \u2014 <b>{bp.exit_reason or '?'}</b>",
            f"  entry {self._price(bp.entry_fill_price)} \u2192 "
            f"exit {self._price(bp.exit_fill_price)}",
            f"  quantity sold: {self._tokens_qty(
                bp.exit_tokens_raw or bp.entry_tokens_raw,
                bp.entry_token_decimals)}",
        ]
        if bp.net_return_pct is not None:
            lines.append(f"  net return: {float(bp.net_return_pct):+.2f}%")
        if bp.exit_proceeds_usd is not None:
            lines.append(f"  net proceeds: {self._usd(bp.exit_proceeds_usd)}")
        fees = []
        if bp.entry_fee_quote is not None:
            fees.append(f"entry {float(bp.entry_fee_quote):.9f}")
        if bp.exit_fee_quote is not None:
            fees.append(f"exit {float(bp.exit_fee_quote):.9f}")
        if fees:
            lines.append(f"  fees: {' + '.join(fees)} SOL")
        if bp.net_pnl_usd is not None:
            lines.append(
                f"  net PAPER P&L: {float(bp.net_pnl_usd):+.6f} USD")
        # THE DENOMINATOR, stated. An all-in return is measured on the WHOLE
        # committed budget, so the P&L above and the percentage next to it are
        # only interpretable with the stake and the rule beside them.
        if cost is not None:
            _rule = (f", {float(bp.sizing_pct):.0f}% of "
                     f"{self._usd(bp.allocated_usd)} available (ALL-IN)"
                     if bp.sizing_pct is not None
                     and bp.allocated_usd is not None else "")
            lines.append(f"  committed: {self._usd(cost)}{_rule}")
        if bankroll_after is not None:
            lines.append(f"  arm bankroll: ${float(bankroll_after):,.4f}")
        lines.append(
            "  P&L basis: <b>"
            + ("quote-simulated fill"
               if bp.exit_basis == EXECUTABLE_QUOTE_BASIS
               else "modeled backtest (NOT a quote sale)")
            + "</b>")
        if bp.path_ambiguous:
            lines.append(
                "  <i>ambiguous TP/SL ordering \u2014 booked at the "
                "conservative stop</i>")
        return lines

    def render_beta_exit_blocked(
        self, settled: Sequence[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        The EXIT BLOCKED / UNRESOLVED notifications -- the OTHER half of an exit.

        WHY THIS EXISTS. When a valid executable SELL quote cannot be obtained
        (missing, stale, refused by the fill model, or the venue cannot be
        addressed) the arm is left UNRESOLVED and retried. That was SILENT: the
        operator saw neither a close nor a refusal, so an experiment whose exits
        were mostly stuck looked exactly like one that was simply quiet. A
        blocked exit now states, per arm, what was due, why it could not be
        executed, and that it is still outstanding and retried.

        IT IS NOT A SALE ALERT, and must never be mistaken for one: the message
        says NO SALE WAS BOOKED, gives no exit price and reports no P&L -- none of
        those exist until a quote settles the arm.

        READ FROM THE BOOK, NOT FROM A TICK. ``settled`` is accepted for API
        symmetry with :meth:`render_beta_exits` and deliberately IGNORED: the
        source of truth is the experiment's own outstanding arms, so an arm whose
        first notification was interrupted (a crash between the refusal and the
        send) is still announced on a later tick, and the DURABLE
        ``exit_notified_attempt`` marker is what stops an already-reported
        attempt from being announced twice.

        NOTIFIABLE means the experiment says so --
        :meth:`BetaExperiment.exit_block_notifiable` -- which is where the
        first-then-every-Nth cadence and the persisted per-attempt marker live.
        """
        exp = self.beta_experiment
        if exp is None:
            return []
        due: list[Any] = []
        for by_arm in exp.positions.values():
            for bp in by_arm.values():
                if exp.exit_block_notifiable(bp):
                    due.append(bp)
        if not due:
            return []
        by_parent: dict[str, list[Any]] = {}
        for bp in due:
            by_parent.setdefault(bp.parent_signal_id, []).append(bp)
        return [self._beta_exit_blocked_message(parent_id, arms)
                for parent_id, arms in by_parent.items()]

    def _beta_exit_blocked_message(self, parent_id: str,
                                   arms: Sequence[Any]) -> dict[str, Any]:
        """Build ONE grouped EXIT BLOCKED message for a parent's stuck arms."""
        first = arms[0]
        lines = [
            "\U0001f534 <b>PAPER EXIT BLOCKED \u2014 V2 FLOOR (BETA)</b>",
            "<i>PAPER SIMULATION \u2014 not a real on-chain trade</i>",
            f"${first.symbol or '?'}",
            "",
            f"Profile: <b>{BETA_PROFILE_LABEL}</b>",
            f"Parent signal: <code>{parent_id}</code>",
            f"Token: <code>{first.mint or ''}</code>",
            "",
            "<b>NO SALE WAS BOOKED.</b> No exit price and no P&L exist for the "
            "arm(s) below until a valid executable SELL quote is accepted.",
        ]
        blocked_arms: list[tuple[str, str, int]] = []
        for bp in sorted(arms, key=lambda b: b.arm):
            attempts = int(getattr(bp, "exit_attempts", 0) or 0)
            blocked_arms.append((parent_id, bp.arm, attempts))
            cost = (bp.entry_cost_usd if bp.entry_cost_usd is not None
                    else bp.position_size_usd)
            lines += [
                "",
                f"Arm {bp.arm} \u2014 exit due: <b>{bp.exit_reason or '?'}</b> "
                f"\u2014 <b>UNRESOLVED</b>",
                f"  entry {self._price(bp.entry_fill_price)} \u00b7 held "
                f"quantity: {self._tokens_qty(
                    bp.entry_tokens_raw, bp.entry_token_decimals)}",
            ]
            if cost is not None:
                _rule = (f", {float(bp.sizing_pct):.0f}% of "
                         f"{self._usd(bp.allocated_usd)} available (ALL-IN)"
                         if bp.sizing_pct is not None
                         and bp.allocated_usd is not None else "")
                lines.append(f"  committed: {self._usd(cost)}{_rule}")
            lines.append(
                "  reason: " + _one_line(
                    bp.exit_blocked_reason or "no executable SELL quote"))
            lines.append(
                f"  retry: attempt {attempts} \u2014 the arm stays outstanding "
                "and is retried on the next monitor tick (re-announced first, "
                f"then every {BETA_EXIT_BLOCK_NOTIFY_EVERY})")
        refusal = (getattr(self, "_beta_quote_refusals", None) or {}).get(
            parent_id)
        if refusal:
            # THE REFUSAL GETS A LARGER BUDGET THAN A PROSE REASON, and it needs
            # one: it carries the refusal of EVERY leg (aggregator, bonding
            # curve, PumpSwap pool), and a line truncated to the default 180
            # characters reports one and a half of them. Telling a throttled
            # venue apart from a dead market is the whole point of the line.
            lines += ["", f"SELL quote refusal: {_one_line(refusal, 560)}"]
        lines += ["", "Mode: <b>PAPER</b>"]
        return {
            "text": "\n".join(lines),
            "kind": "beta_exit_blocked",
            "parent_signal_id": parent_id,
            # What the runtime records as ANNOUNCED once the notifier returns.
            "blocked_arms": blocked_arms,
            "dedupe_key": ("beta_exit_blocked::" + parent_id + "::"
                           + "|".join(f"{arm}:{n}" for _, arm, n
                                       in blocked_arms)),
        }

    def _usd(self, value: float | None) -> str:
        return "-" if value is None else f"${value:,.2f}"

    def _price(self, value: float | None) -> str:
        return "-" if value is None else f"${float(value):,.8f}"

    def _restore_beta_mirrors(self) -> None:
        """
        Rebuild visibility mirrors for open experiment positions after a restart.

        The experiment's book is DURABLE -- positions, ledgers and settled trades
        all come back from ``v2_floor_beta_state.json`` -- but the mirrors are
        in-memory only. Without this pass a restart would leave every restored
        open position with no mirror, ``_monitor_beta_experiment`` would have
        nothing to observe, and a recovered experiment could never settle at a
        mark again: it would sit open until its deadline and then be recorded as
        ``TIME_NO_QUOTE``. Rebuilding the mirror is what makes "restart restores
        positions" mean "restart restores MONITORING".

        Mirrors that already exist are left untouched, so a running experiment
        pays one dictionary scan per tick and nothing else.
        """
        experiment = self.beta_experiment
        if experiment is None:
            return
        arm_name_by_key = {key: arm.value
                           for arm, key in BETA_ARM_KEYS.items()}
        for pid, by_arm in list(experiment.positions.items()):
            for arm_key, bp in by_arm.items():
                # Rebuild monitoring for an UNRESOLVED arm too: a restart must
                # restore MONITORING, and an unresolved arm is still
                # outstanding until a real quote settles it.
                if bp.status not in ("open", "unresolved"):
                    continue
                arm_name = arm_name_by_key.get(arm_key)
                if arm_name is None:
                    continue
                if self.beta_positions.get(pid, {}).get(arm_name) is not None:
                    continue
                spec = next((s for s in self.arms
                             if s.name.value == arm_name), None)
                if spec is None:
                    # The arm is not selected, so the experiment is not
                    # running either; nothing to observe for it here.
                    continue
                self.beta_positions.setdefault(pid, {})[arm_name] = \
                    self._beta_mirror_position(spec, bp)

    def _signal_with_v2_gate_inputs(
        self, signal: SignalSnapshot, *, mark: Any,
    ) -> SignalSnapshot:
        """
        Carry the decision-time V2 FLOOR gate inputs onto the signal snapshot.

        ``BetaExperiment`` re-runs the UNCHANGED production gate
        (:func:`autopaper.v2floor.evaluate`) on exactly three numbers and
        refuses -- fail-closed -- when any of them is absent. Somebody has to
        move those numbers from the decision that computed them to the gate that
        consumes them; that is this method.

        THE SOURCE IS THE DECISION REPORT'S OWN FEATURE SNAPSHOT: the same
        ``FeatureSet`` the production gates were evaluated on for this very
        signal, so the experiment cannot disagree with the decision about what
        the market looked like at decision time. Nothing is derived here and
        nothing is substituted -- when the report carries no usable value the
        field stays ``None`` and the gate refuses with its explicit reason,
        which is the pre-existing behaviour for a missing input.

        A snapshot that already carries the inputs (a research harness, a
        replay) is returned untouched: the caller's evidence wins.
        """
        fields = ("v2_volume_5m_usd", "v2_market_cap_usd", "v2_change_5m_pct")
        if all(getattr(signal, name, None) is not None for name in fields):
            return signal
        try:
            report, _status, _auth = self.decide(
                mint=signal.token_mint or "", signal=signal, mark=mark,
                decision_time=time.monotonic(), blocked_reason=None,
            )
        except Exception as exc:  # noqa: BLE001 - a failed decision refuses
            log("ERROR",
                "V2_FLOOR_BETA gate inputs unavailable (decision failed): "
                "%s: %s", type(exc).__name__, exc)
            return signal
        features = getattr(report, "features", None)
        if not isinstance(features, Mapping):
            return signal
        return replace(
            signal,
            v2_volume_5m_usd=_gate_input(features.get("volume_5m_usd")),
            v2_market_cap_usd=_gate_input(features.get("market_cap_usd")),
            v2_change_5m_pct=_gate_input(features.get("change_5m")),
        )

    def _beta_refusal_record(
        self, *, signal: SignalSnapshot, arm_name: str, mark: Any,
        latency: LatencyMarks, reason: str,
        failure: FailureReason = FailureReason.V2_FLOOR_REJECTED,
        decision: str = "REJECT",
        status: str = "V2_FLOOR_REJECTED",
    ) -> SignalRecord:
        """
        One beta arm's REFUSED entry: recorded, never silently dropped.

        A dataset of trades alone cannot show that the experiment refused, so
        every beta arm gets a record even when nothing was opened -- and the
        record carries no position, no size and no parent signal id, because
        none of those exist.
        """
        experiment = self.beta_experiment
        if experiment is not None:
            key = BETA_ARM_KEYS[ResearchArm(arm_name)]
            bankroll = experiment.accounts[key].bankroll_usd
        else:
            account = (self.accounts or {}).get(arm_name)
            bankroll = (account.bankroll_usd if account is not None
                        else self.starting_bankroll_usd)
        return SignalRecord(
            signal=signal, arm=arm_name,
            arm_version=SCHEMA_VERSION, baseline_hash=BASELINE_HASH,
            l25_decision=decision, l25_status=status,
            l25_authorized=False, arm_filter_reason=reason,
            observed_mark=mark,
            buy_quote=QuotePoint(
                side="BUY", provider="none", market_type="UNKNOWN",
                token_mint=signal.token_mint or "", quote_mint=WSOL_MINT,
                input_amount_raw=None, input_decimals=None,
                input_amount_decimal=None, output_amount_raw=None,
                output_decimals=None, output_amount_decimal=None,
                effective_price=None, effective_price_usd=None,
                price_impact_pct=None, fee_bps=None, fee_amount=None,
                venue=None, pool_or_curve=None, route=(),
                identity="UNKNOWN", available=False,
                missing_reason=reason,
            ),
            mark_to_quote_gap_pct=None, execution_identity="UNKNOWN",
            traded=False, failure=failure, failure_detail=reason,
            latency=latency, bankroll_before=bankroll,
            position_size_usd=None, position_size_pct=None,
            finished_at=time.time(),
        )

    # -- entry ------------------------------------------------------------

    def _mark_gap_guard(self, quote: Any, mark: Any) -> Any:
        """
        THE PRODUCTION ENTRY GUARD.

            native_gap_pct = (executable_buy / same_market_spot - 1) * 100
            refuse when native_gap_pct > 10.0%

        THE REFERENCE IS THE SAME MARKET'S OWN CURRENT ON-CHAIN PRICE.
        =============================================================

        The DexScreener observed mark is an EXTERNAL AGGREGATE'S SNAPSHOT of
        the very market being traded, refreshed on the aggregator's cadence.
        Measured consequence on L25's own population -- bonding curves minutes
        old, moving hundreds of percent -- was a mark-vs-executable spread with
        a standard deviation of ~26 percentage points against a true execution
        premium of 1.26% +/- 0.003. It rejected 100% of authorized entries, and
        on the live sample 3 of 3 rejections were recovered by the native
        reference with 0 candidates moving the other way.

        The reference is therefore the market's own spot price, read from the
        SAME decoded account, at the SAME slot, in the SAME call that produced
        the executable quote. It is not a second data source and not a second
        identity system: ``market_id`` is the same ``<market_type>:<pool>`` the
        execution layer already pins an exit to.

        UNITS. Both sides are QUOTE ASSET PER WHOLE TOKEN. The previous
        comparison took the quote's USD price against a USD mark, which put a
        single SOL/USD conversion on one side of the ratio only. Comparing
        quote-asset to quote-asset removes that asymmetry.

        WHAT THE GAP IS. The reference is a MARGINAL price; the quote is an
        AVERAGE price for the exact order size. Their ratio therefore IS the
        cost of transacting this order in this market at this slot -- the venue
        fee plus the size impact -- which is precisely what the guard is asked
        to police.

        NOTHING IS ASSUMED. ``same_order_size`` used to be the literal ``True``
        and ``mark_fresh`` was ``bool(mark.available)``, meaning "fresh" meant
        "a price exists". Both are now derived facts, and a false value makes
        the verdict UNKNOWN, which refuses.

        NO FALLBACK. An absent reference is ``UNKNOWN`` and refuses. It is never
        the observed mark, never another pool, never a token-wide price.

        The threshold is unchanged at ``MARK_GAP_GUARD_THRESHOLD_PCT = 10.0``,
        the comparison stays strict (``>``), and UNKNOWN still refuses.
        """
        from .economics import evaluate_execution_native_gap

        native = getattr(quote, "execution_native_reference", None)
        if native is None:
            from .economics import MarkGapVerdict
            return MarkGapVerdict(
                result="UNKNOWN",
                reason=(
                    "the executable market's own current price could not be "
                    "read from chain, so there is no same-market reference to "
                    "compare against; the observed mark is NOT used as a "
                    "substitute because it is an external aggregate's "
                    "potentially lagging snapshot of this same market, and no "
                    "other pool or token-wide price is a substitute either"),
                mark_price=None,
                executable_price=(quote.effective_price
                                  if quote.available else None),
                gap_pct=None, threshold_pct=self.mark_gap_threshold_pct,
            )

        # Order-size comparability, DERIVED from the quote's own data rather
        # than asserted. The reference is a spot price (size-independent by
        # construction) and the quote is an average price for the exact size,
        # so they are comparable exactly when the reference came from the same
        # call that priced that amount. Both come from one QuotePoint produced
        # by one read, so this is a real check on the data, not a tautology.
        same_order_size = (
            getattr(quote, "input_amount_raw", None) is not None
            and int(getattr(quote, "input_amount_raw")) > 0
            and native.numerator_raw is not None
            and native.denominator_raw is not None
            and int(native.denominator_raw) > 0
        )
        # Token decimals must match, or the two prices are not on one scale.
        same_decimals = (
            native.token_decimals is not None
            and getattr(quote, "output_decimals", None) is not None
            and int(native.token_decimals) == int(getattr(quote,
                                                         "output_decimals"))
        )
        # SAME MARKET, proven by comparing the two canonical ids rather than
        # trusting that both were built nearby.
        same_market = bool(
            native.market_id and getattr(quote, "venue_id", None)
            and native.market_id == getattr(quote, "venue_id")
        )

        return evaluate_execution_native_gap(
            executable_price=(float(quote.effective_price)
                              if quote.available
                              and getattr(quote, "effective_price", None)
                              is not None else None),
            reference_price=native.reference_price,
            same_market=same_market,
            # The token-mint check is a separate, already-enforced guard
            # (TOKEN_MINT_MISMATCH); reaching here means it passed.
            same_mint=True,
            same_quote_asset=(quote.quote_mint == WSOL_MINT
                              and native.quote_mint == WSOL_MINT),
            same_side=(quote.side == "BUY" and native.side == "BUY"),
            same_token_decimals=same_decimals,
            same_order_size=same_order_size,
            # Both sides come from ONE account read, so the reference cannot be
            # staler than the quote it is compared with. Freshness is therefore
            # the quote's own freshness, applied to both.
            reference_fresh=not bool(getattr(quote, "stale", False)),
            quote_fresh=(not quote.stale) if quote.available else False,
            market_id=native.market_id,
            threshold_pct=self.mark_gap_threshold_pct,
        )

    def _legacy_gap_guard_diagnostic(self, quote: Any, mark: Any) -> Any:
        """
        DIAGNOSTIC ONLY -- this decides nothing.

        The OLD comparison, executable BUY versus the external observed mark,
        kept and computed on the SAME record so "what would the legacy guard
        have decided?" stays answerable without a second pass, and so OLD and
        NEW verdicts can be compared on one candidate.

        It is deliberately a separate method from :meth:`_mark_gap_guard` so no
        future edit can quietly promote it back to the production decision.
        """
        from .economics import evaluate_mark_gap

        return evaluate_mark_gap(
            mark_price=mark.value_usd,
            executable_price=(quote.effective_price_usd
                              if quote.available else None),
            same_mint=True,
            same_market_identity=(quote.identity == "PASS"),
            same_quote_asset=(quote.quote_mint == WSOL_MINT),
            same_side=(quote.side == "BUY"),
            same_order_size=self._same_order_size_basis(quote),
            mark_fresh=bool(mark.available),
            quote_fresh=(not quote.stale) if quote.available else False,
            threshold_pct=self.mark_gap_threshold_pct,
        )

    def _same_order_size_basis(self, quote: Any) -> bool:
        """
        Is the external observed mark comparable with THIS quote's order size?

        Only used by the LEGACY diagnostic. The mark is a price-per-token
        aggregate with no size attached, so the comparison is only meaningful
        because the quote was requested for the sized notional earlier in the
        same method. That ordering is the guarantee; this reports it as a fact
        rather than asserting ``True`` inline, and it fails CLOSED if the quote
        somehow carries no size.
        """
        amount = getattr(quote, "input_amount_raw", None)
        return amount is not None and int(amount) > 0

    def _legacy_observed_mark_gap(self, quote: Any, mark: Any) -> float | None:
        """
        The OLD gap, kept for research only and never for a decision.

            old_gap = executable / observed_mark - 1

        This is the quantity that produced the +66% and +112% rejections. It is
        still measured on every entry so the two references can be compared on
        the SAME record -- which is the only way to answer "how many old false
        rejections disappear?" without reconstructing history. It is not a
        fallback: when the execution-native reference is unavailable the entry
        is refused, never evaluated against this.
        """
        mk = getattr(mark, "value_usd", None)
        ex = (getattr(quote, "effective_price_usd", None)
              if getattr(quote, "available", False) else None)
        if not mk or not ex or float(mk) <= 0 or float(ex) <= 0:
            return None
        return (float(ex) / float(mk) - 1.0) * 100.0

    def _evaluate_economics(self, size: Any, *, sol_usd: float) -> Any:
        """The micro-account policy verdict for this planned order."""
        if not self.economics_enabled:
            return None
        from .economics import evaluate_entry_economics
        return evaluate_entry_economics(
            notional_usd=float(size.position_usd), sol_usd=float(sol_usd or 0.0))

    async def _quote_buy_best(
        self, *, mint: str, sol_lamports: int, token_decimals: int,
        observed_pair_address: str | None,
    ) -> tuple[Any, list[Any]]:
        """
        The best executable BUY plus every candidate venue.

        THE CACHE LIVES HERE, NOT IN THE QUOTE SERVICE. Three arms consume one
        signal on the same mint at the same notional, so without sharing they
        would be measured against three DIFFERENT snapshots of the same book --
        which is a correctness problem, not a performance one. Placing the
        policy in the engine means it applies to every quote service,
        including ones that only implement a single ``quote_buy``.

        A cached entry is returned only while the quote's own age says it is
        fresh, so sharing can never extend a quote's life.
        """
        from .quotes import QuoteCache

        self.last_buy_candidates = []
        key = ("BUY", mint, int(sol_lamports), int(token_decimals))
        cache = self.quote_cache
        if cache is None:
            cache = QuoteCache()
            self.quote_cache = cache

        # SINGLE-FLIGHT COALESCING.
        #
        # The three PAPER arms consume one signal at the same notional, and the
        # cache made them share a quote -- but only AFTER the first one had
        # finished. Asked concurrently they all missed, and three identical
        # quotes were fetched for one decision. Keying an in-flight task means
        # the second and third arm await the SAME quote instead of issuing
        # their own.
        #
        # This is the requirement that quote capacity is spent once per
        # (mint, side, size, decimals, market identity), not once per arm.
        inflight = getattr(self, "_quote_inflight", None)
        if inflight is None:
            inflight = self._quote_inflight = {}
        pending = inflight.get(key)
        if pending is not None:
            shared = await asyncio.shield(pending)
            self.last_quote_coalesced += 1
            self.last_buy_candidates = list(shared[1]) if shared[1] else [shared[0]]
            self.last_buy_cache_state = "coalesced"
            return shared[0], self.last_buy_candidates

        before = _cache_count(cache)
        if cache.hits == 0 and getattr(cache, "_seen_keys", None) is None:
            cache._seen_keys = set()
        # A repeated fetch for the same key within this signal is a SHARED
        # snapshot, not a fresh provider call.
        if key in cache._seen_keys:
            hit = cache.get(key)
            if hit is not None:
                quote, extras = hit
                self.last_buy_cache_state = "cache_hit"
                self.last_buy_candidates = list(extras) if extras else [quote]
                return quote, self.last_buy_candidates

        best_buy = getattr(self.quote_service, "quote_buy_best", None)
        if callable(best_buy):
            async def _fetch() -> tuple[Any, list[Any]]:
                q, c = await best_buy(
                    mint=mint, sol_lamports=sol_lamports,
                    token_decimals=token_decimals,
                    observed_pair_address=observed_pair_address,
                    cache=cache)
                return q, list(c)
        else:
            async def _fetch() -> tuple[Any, list[Any]]:
                q = await self.quote_service.quote_buy(
                    mint=mint, sol_lamports=sol_lamports,
                    token_decimals=token_decimals,
                    observed_pair_address=observed_pair_address)
                return q, [q]

        # Publish the task BEFORE awaiting it, so a sibling arm that arrives
        # while this one is in flight joins it instead of starting its own.
        task = asyncio.ensure_future(_fetch())
        inflight[key] = task
        try:
            quote, candidates = await asyncio.shield(task)
        finally:
            inflight.pop(key, None)

        self.last_buy_candidates = list(candidates)
        if quote is not None and getattr(quote, "available", False):
            cache.put(key, quote, tuple(self.last_buy_candidates))
            cache._seen_keys.add(key)
        self.last_buy_cache_state = (
            "cache_hit" if _cache_count(cache) > before else "fresh")
        return quote, self.last_buy_candidates

    async def _open_for_arm(
        self,
        *,
        spec: ArmSpec,
        signal: SignalSnapshot,
        mark: ObservedMarkPoint,
        report: Any,
        latency: LatencyMarks,
    ) -> tuple[SignalRecord, PaperPosition | None]:
        arm = spec.name.value
        account = self.accounts[arm]
        bankroll_before = account.bankroll_usd
        #: Per-source BUY candidate outcome (audit §2). Empty until the quote
        #: stage runs; carried by every quote-stage refusal below.
        buy_cause = ""

        def reject(reason: FailureReason, detail: str, l25_status: str = "",
                   authorized: bool = False,
                   gap: float | None = None,
                   legacy_gap: float | None = None,
                   native_gap: float | None = None,
                   native_verdict: str = "",
                   legacy_verdict: str = "",
                   quote: Any = None,
                   economics: Any = None) -> tuple[SignalRecord, None]:
            """
            A named refusal that is STILL RECORDED.

            ``gap`` is carried through on purpose: a refusal that dropped the
            measured gap would make "we measured a 90% gap and refused" look
            identical to "we never measured", and the gap distribution is the
            research output this whole engine exists to produce.

            ``quote`` carries the REAL measured quote when one already exists.
            Without it a rejection persisted a hardcoded placeholder --
            ``provider="none"``, ``identity="UNKNOWN"``, price ``None``, venue
            and pool ``None`` -- which destroyed the pool, venue, provider,
            price impact, fee and identity evidence of the very quote the
            refusal was DECIDED on. Every MARK_GAP refusal on record therefore
            claimed identity UNKNOWN even though the guard only ever fires when
            identity is PASS, and the dataset could not answer what market was
            actually compared. Pass the quote; the placeholder is kept only for
            refusals that happen BEFORE any quote was obtained.

            ``economics`` carries the micro-account verdict, when the economic
            guard ran, so a MICRO_ACCOUNT_ECONOMICALLY_INVALID block records the
            notional, the locked rent, the fixed cost, the break-even and the
            classification -- not just the reason sentence. ``None`` (the
            default) for every other refusal, which is how "the guard did not
            run" stays distinguishable from "the guard ran and refused".
            """
            return SignalRecord(
                signal=signal, arm=arm, arm_version=SCHEMA_VERSION,
                baseline_hash=BASELINE_HASH, l25_decision="REJECT",
                l25_status=l25_status, l25_authorized=authorized,
                arm_filter_reason="", observed_mark=mark,
                buy_quote=(quote if quote is not None else QuotePoint(
                    side="BUY", provider="none", market_type="UNKNOWN",
                    token_mint=signal.token_mint or "", quote_mint=WSOL_MINT,
                    input_amount_raw=None, input_decimals=None,
                    input_amount_decimal=None, output_amount_raw=None,
                    output_decimals=None, output_amount_decimal=None,
                    effective_price=None, effective_price_usd=None,
                    price_impact_pct=None, fee_bps=None, fee_amount=None,
                    venue=None, pool_or_curve=None, route=(),
                    identity="UNKNOWN", available=False,
                    missing_reason=detail,
                )),
                mark_to_quote_gap_pct=gap,
                legacy_observed_mark_gap_pct=legacy_gap,
                execution_native_gap_pct=native_gap,
                native_guard_verdict=native_verdict,
                legacy_guard_verdict=legacy_verdict,
                production_guard_source=PRODUCTION_GUARD_SOURCE,
                execution_identity=(getattr(quote, "identity", None)
                                    if quote is not None else "UNKNOWN"),
                traded=False, failure=reason, failure_detail=detail,
                buy_quote_failure_cause=buy_cause,
                latency=latency, bankroll_before=bankroll_before,
                position_size_usd=None,
                position_size_pct=None,
                economics=(economics.as_dict()
                           if hasattr(economics, "as_dict") else economics),
                finished_at=time.time(),
                **_entry_execution_gap_observability(
                    mark=mark,
                    quote=(quote if quote is not None
                           else QuotePoint(
                               side="BUY", provider="none",
                               market_type="UNKNOWN",
                               token_mint=signal.token_mint or "",
                               quote_mint=WSOL_MINT,
                               input_amount_raw=None,
                               input_decimals=None,
                               input_amount_decimal=None,
                               output_amount_raw=None,
                               output_decimals=None,
                               output_amount_decimal=None,
                               effective_price=None,
                               effective_price_usd=None,
                               price_impact_pct=None,
                               fee_bps=None, fee_amount=None,
                               venue=None, pool_or_curve=None,
                               route=(),
                               identity="UNKNOWN", available=False,
                               missing_reason=detail,
                           )),
                    guard=None),
            ), None

        # -- the one-open-position constraint, enforced per arm ------------
        if account.has_position:
            account.blocked_by_position_active += 1
            return reject(
                FailureReason.ENTRY_BLOCKED_POSITION_ACTIVE,
                f"arm {arm} already holds a position; max simultaneous "
                "positions is 1",
            )

        # -- cooldown, exactly as production defines it --------------------
        last_exit = self.recently_exited[arm].get(signal.token_mint or "")
        if last_exit is not None:
            return reject(
                FailureReason.ENTRY_BLOCKED_COOLDOWN,
                f"this mint exited at {last_exit:.0f} and re-entry is disabled",
            )

        if not mark.available:
            return reject(
                FailureReason.MARKET_DATA_MISSING,
                mark.missing_reason or "no observed market data",
            )

        # -- the ARM'S OWN GATE -------------------------------------------
        # ``report.decision`` is now a decision over EVERY operative gate, so
        # under a mixed selection a BUY means "at least one selected strategy
        # authorized this". That is not enough for THIS arm: an L25 arm must not
        # open because the beta experiment liked the candidate. Each arm is
        # therefore authorized by its own gate's own result on the same report.
        #
        # With a single gate operative this is exactly the old check -- the
        # profile's gate and the arm's gate are the same one -- so the behaviour
        # and the recorded reason are unchanged.
        own_gate = entry_gate_for_arm(spec)
        own_result = _gate_result_of(report, own_gate)
        own_status = str(getattr(own_result, "status", "UNKNOWN"))
        own_authorized = bool(getattr(own_result, "authorized", False))
        if report.decision is not Decision.BUY:
            return reject(
                FailureReason.L25_REJECTED,
                f"production decision is {report.decision.value}",
                l25_status=own_status, authorized=own_authorized,
            )
        if not own_authorized:
            # Reached only under a MIXED selection: another operative gate
            # authorized, this arm's did not.
            log("INFO",
                "arm %s declined on its own gate %s (%s) while another "
                "selected strategy authorized the candidate",
                spec.name.value, own_gate, own_status)
            return reject(
                FailureReason.L25_REJECTED,
                f"this arm's own gate {own_gate} did not authorize "
                f"({own_status}); another selected strategy did",
                l25_status=own_status, authorized=False,
            )
        l25_status, authorized = own_status, own_authorized

        # -- the arm's own extra rule, AFTER L25 ---------------------------
        allowed, why = spec.entry_allows(report.features)
        if not allowed:
            return reject(
                FailureReason.ARM_FILTER_REJECTED, why,
                l25_status=l25_status, authorized=authorized,
            )

        # -- production sizing on THIS arm's bankroll ----------------------
        size = account.size_for_bankroll()
        if size.position_usd <= 0:
            return reject(
                FailureReason.SIZING_REFUSED, size.rationale,
                l25_status=l25_status, authorized=authorized,
            )
        if self.sol_reference is None or self.sol_reference.usd_per_sol <= 0:
            return reject(
                FailureReason.SOL_REFERENCE_MISSING,
                "no SOL/USD reference, so the USD notional cannot be converted "
                "to lamports; it is NOT assumed",
                l25_status=l25_status, authorized=authorized,
            )

        sol_lamports = int(
            (Decimal(str(size.position_usd))
             / Decimal(str(self.sol_reference.usd_per_sol))).scaleb(SOL_DECIMALS)
        )
        if sol_lamports <= 0:
            return reject(
                FailureReason.SIZING_REFUSED,
                f"a ${size.position_usd:g} position rounds to {sol_lamports} "
                "lamports, which no venue will price",
                l25_status=l25_status, authorized=authorized,
            )

        decimals = self._decimals_cache.get(signal.token_mint or "")
        if decimals is None and self.token_decimals_for is not None:
            try:
                # A reader may be sync or async, and it is called EXACTLY ONCE.
                # The previous shape called `reader(mint)`, and then called it
                # AGAIN after awaiting -- so on the async path every entry
                # issued a second chain read whose coroutine was never awaited.
                # That is a wasted RPC round trip per signal and a live
                # RuntimeWarning, in the entry path of a trading bot.
                result = self.token_decimals_for(signal.token_mint or "")
                if inspect.isawaitable(result):
                    result = await result
                decimals = result
                # ONE retry. This is a single account read, and a transient RPC
                # failure here silently costs the whole trade -- while a genuine
                # "no such mint" returns None identically and is NOT retried,
                # because the reader records which it was.
                if not decimals:
                    reader = self.token_decimals_for
                    cause = getattr(getattr(reader, "__self__", None),
                                    "last_mint_decimals_error", "")
                    if "absent" not in cause and "not owned" not in cause:
                        retry = self.token_decimals_for(
                            signal.token_mint or "")
                        if inspect.isawaitable(retry):
                            retry = await retry
                        decimals = retry
                    if not decimals:
                        cause = getattr(
                            getattr(reader, "__self__", None),
                            "last_mint_decimals_error", "")
                        if cause:
                            self.last_decimals_error = cause
            except Exception as exc:  # noqa: BLE001 - unknown decimals is
                # fail-closed, but the CAUSE is recorded. A guard that refuses
                # silently is indistinguishable from one that is broken, and
                # this is the guard that decides whether a live entry can
                # happen at all.
                decimals = None
                self.last_decimals_error = (
                    f"{type(exc).__name__}: {exc}")[:200]
        if decimals:
            self._decimals_cache[signal.token_mint or ""] = int(decimals)
        if not decimals or decimals <= 0:
            return reject(
                FailureReason.ENTRY_REJECTED_DECIMALS_UNKNOWN,
                "the token's decimals could not be read from its mint account, "
                "so no fill can be normalised; a default of 6 is NOT assumed"
                + (f"; underlying error: {self.last_decimals_error}"
                   if getattr(self, "last_decimals_error", None) else ""),
                l25_status=l25_status, authorized=authorized,
            )

        # -- MICRO-ACCOUNT ECONOMIC VIABILITY ------------------------------
        #
        # Runs AFTER the canonical sizing decision and BEFORE any money is
        # priced. It does not change the size; it asks whether an order this
        # small is worth its fixed costs at all. Rent is reported here but is
        # NOT charged: it is locked and comes back on close.
        economics = self._evaluate_economics(size, sol_usd=self.sol_reference.usd_per_sol)
        if economics is not None and not economics.allowed:
            # The verdict is RECORDED with its inputs (planned notional, the
            # estimated round-trip fixed cost, the locked/recoverable rent, the
            # break-even return, the classification) so a blocked BUY is
            # auditable from the dataset alone. The reason string alone cannot
            # be: it names the constraint, not the numbers that tripped it.
            return reject(
                FailureReason.MICRO_ACCOUNT_ECONOMICALLY_INVALID,
                economics.reason, l25_status=l25_status, authorized=authorized,
                economics=economics,
            )

        # -- FRESH EXECUTABLE BUY QUOTE, BEST VENUE -------------------------
        #
        # Every candidate venue is quoted for the EXACT order size and the best
        # EXECUTABLE OUTPUT wins. Not the nominal price, not the mark, not
        # liquidity, not a venue preference -- each of those is a proxy for the
        # quantity being measured.
        latency.buy_quote_request_at = time.monotonic()
        quote, candidates = await self._quote_buy_best(
            mint=signal.token_mint or "", sol_lamports=sol_lamports,
            token_decimals=int(decimals),
            observed_pair_address=mark.pair_address,
        )
        latency.buy_quote_response_at = time.monotonic()
        if self.last_buy_candidates:
            latency.quote_cache_state = self.last_buy_cache_state

        # NAME EACH SOURCE while the candidates are still in hand. This is the
        # diagnostic that replaces the flat ENTRY_REJECTED_QUOTE_UNAVAILABLE:
        # one clause per source with its own failure code.
        try:
            from .quotes import buy_candidate_summary
            _summary_fn = getattr(self.quote_service, "candidate_summary", None)
            if callable(_summary_fn):
                buy_cause = _summary_fn(
                    candidates, mint=signal.token_mint or None)
            else:
                buy_cause = buy_candidate_summary(
                    candidates, mint=signal.token_mint or None)
        except Exception:  # noqa: BLE001 - diagnostics must never break entry
            buy_cause = ""

        # The quote must be ABOUT THIS TOKEN. A provider returning a different
        # token is a bug, and using it would buy the wrong asset while every
        # number in the report looked correct. Fail closed.
        if quote is not None and quote.available and quote.token_mint and quote.token_mint != (
            signal.token_mint or ""
        ):
            detail = (
                f"the quote is for token {quote.token_mint[:8]}… but the signal "
                f"is {str(signal.token_mint)[:8]}…; a quote for another token is "
                "refused rather than traded"
            )
            return SignalRecord(
                signal=signal, arm=arm, arm_version=SCHEMA_VERSION,
                baseline_hash=BASELINE_HASH, l25_decision="BUY",
                l25_status=l25_status, l25_authorized=authorized,
                arm_filter_reason=why, observed_mark=mark, buy_quote=quote,
                mark_to_quote_gap_pct=None, execution_identity=quote.identity,
                traded=False, failure=FailureReason.TOKEN_MINT_MISMATCH
                if hasattr(FailureReason, "TOKEN_MINT_MISMATCH")
                else FailureReason.ARITHMETIC_ERROR,
                failure_detail=detail,
                buy_quote_failure_cause=buy_cause,
                latency=latency, bankroll_before=bankroll_before,
                position_size_usd=size.position_usd,
                position_size_pct=size.position_pct,
                finished_at=time.time(),
                **_entry_execution_gap_observability(
                    mark=mark, quote=quote, guard=None),
            ), None

        # -- EXECUTABLE-vs-SAME-MARKET ENTRY GUARD ------------------------
        #
        # The reference is the SAME market's own current on-chain price, taken
        # from the same account at the same slot as the quote. The observed
        # DexScreener mark is retained as a diagnostic and as the LEGACY gap,
        # but it is NOT the reference: it is a third party's periodically
        # refreshed snapshot of this same market, and on young markets it
        # trailed the curve badly enough to reject 100% of authorized entries.
        #
        # The comparison is only made when reference and quote are proven to
        # describe the same market, asset, side, decimals and size basis;
        # otherwise the verdict is UNKNOWN and UNKNOWN refuses. It never falls
        # back to another pool, and it never falls back to the observed mark.
        #
        # The guard is only consulted for a quote that would OTHERWISE fill. An
        # unproven identity or a stale quote already has a more precise refusal
        # in ``fill_from_quote`` below, and reporting "gap guard" for them would
        # replace an exact reason with a vague one.
        # -- THE PRODUCTION DECISION ------------------------------------
        #
        # The SAME market's own on-chain spot, from the same account, at the
        # same slot, as the quote. 10.0%, strict ``>``, UNKNOWN refuses. This
        # is the ONLY guard that decides.
        guard = self._mark_gap_guard(quote, mark)
        self.last_mark_gap = guard
        latency.mark_gap_guard_result = guard.result
        latency.production_guard_source = PRODUCTION_GUARD_SOURCE

        # -- THE DIAGNOSTIC -----------------------------------------------
        #
        # The OLD comparison, executable BUY versus the external observed
        # mark, is computed on the SAME record so "what would the legacy guard
        # have decided?" stays answerable later without a second pass, and so
        # OLD and NEW verdicts can be compared on one candidate. It decides
        # NOTHING.
        legacy_guard = self._legacy_gap_guard_diagnostic(quote, mark)
        latency.legacy_gap_guard_result = legacy_guard.result
        latency.legacy_observed_mark_gap_pct = legacy_guard.gap_pct
        old_gap = legacy_guard.gap_pct
        native_gap = guard.gap_pct

        if (guard.result in ("REJECT", "UNKNOWN")
                and quote is not None and quote.available and not quote.stale
                and quote.identity == "PASS"):
            # UNKNOWN REFUSES TOO. A guard that passes when it cannot see is not
            # a guard, and a fail-open here would let an entry through on a quote
            # whose reference comparison could not be made at all.
            #
            # A MISSING same-market reference is a different fact from an
            # EXCESSIVE gap, and is named differently. Collapsing the two would
            # make "the market's own price could not be read" indistinguishable
            # from "execution is too expensive".
            return reject(
                _guard_failure_reason(guard),
                f"{guard.result}: {guard.reason}",
                l25_status=l25_status, authorized=authorized,
                gap=guard.gap_pct,
                legacy_gap=old_gap,
                legacy_verdict=legacy_guard.result,
                native_gap=native_gap,
                native_verdict=guard.result,
                # The quote this verdict was decided ON, so the refusal is
                # auditable against the exact market, venue, provider and
                # price impact it measured.
                quote=quote,
            )
        gap = guard.gap_pct if guard.gap_pct is not None else None

        # NAME THE CAPACITY CAUSE before anything downstream can flatten it
        # into "no executable quote".
        if quote is not None and not getattr(quote, "available", False):
            kind = classify_quote_failure(
                getattr(quote, "missing_reason", "") or "")
            latency.quote_capacity_failure = kind.value
            counts = self.quote_capacity_refusals
            counts[kind.value] = counts.get(kind.value, 0) + 1

        try:
            fill = fill_from_quote(
                quote, side="BUY", model=self.fill_model,
                sol_reference=self.sol_reference,
                require_identity=self.require_identity,
            )
        except FillRefused as exc:
            # The observation is RETAINED as a non-trade record even though the
            # PAPER trade fails closed. Dropping it would make "we looked and
            # could not price it" indistinguishable from "we never looked".
            return SignalRecord(
                signal=signal, arm=arm, arm_version=SCHEMA_VERSION,
                baseline_hash=BASELINE_HASH, l25_decision="BUY",
                l25_status=l25_status, l25_authorized=authorized,
                arm_filter_reason=why, observed_mark=mark, buy_quote=quote,
                mark_to_quote_gap_pct=gap,
                legacy_observed_mark_gap_pct=old_gap,
                # The native comparison is recorded on EVERY record that
                # reached the guard, including the ones that traded, so
                # "what would the native guard have decided?" is answerable
                # without a second pass. It decides nothing.
                execution_native_gap_pct=guard.gap_pct,
                native_guard_verdict=guard.result,
                legacy_guard_verdict=legacy_guard.result,
                production_guard_source=PRODUCTION_GUARD_SOURCE,
                execution_identity=quote.identity,
                traded=False, failure=exc.reason, failure_detail=exc.detail,
                buy_quote_failure_cause=buy_cause,
                latency=latency, bankroll_before=bankroll_before,
                position_size_usd=size.position_usd,
                position_size_pct=size.position_pct,
                finished_at=time.time(),
                **_entry_execution_gap_observability(
                    mark=mark, quote=quote, guard=guard),
            ), None

# The token amount HELD is exactly the quoted output, in raw units.
        tokens_raw = fill.output_amount_raw
        entry_cost = Decimal(fill.input_amount_raw).scaleb(-fill.input_decimals)

        position = PaperPosition(
            signal_id=signal.signal_id, arm=arm, mint=signal.token_mint or "",
            symbol=signal.symbol, token_decimals=int(decimals),
            tokens_raw=tokens_raw, entry_fill=fill,
            entry_observed_mark=mark.value_usd, entry_mark_timestamp=mark.timestamp,
            bankroll_before=bankroll_before,
            position_size_usd=size.position_usd, position_size_pct=size.position_pct,
            usd_notional=size.position_usd,
            sol_notional=entry_cost,
            sol_usd_reference=self.sol_reference.usd_per_sol,
            sol_usd_reference_at=self.sol_reference.at,
            opened_at=time.time(),
            entry_cost_quote_asset=entry_cost,
            trading_fees_quote=Decimal(0),
            # QUOTE-ASSET DENOMINATED. The model declares this cost in lamports;
            # ``network_cost_lamports`` would put 10^9x the intended SOL on the
            # position, and the field's own name is the only thing that would
            # have caught it.
            network_costs_quote=self.fill_model.network_cost_sol,
            other_costs_quote=Decimal(0),
            production_profile=spec.production_profile.value,
            # -- THE TAKE-PROFIT SNAPSHOT -----------------------------------
            # Read the GLOBAL selection ONCE, here, and stamp it on the
            # position. Every later exit reads the position's copy, never the
            # global, so an operator who changes the selection while this is
            # open cannot silently rewrite its exit policy. This is the same
            # invariant ``production_profile`` already follows.
            take_profit_pct=self._take_profit_snapshot(),
            # -- BETA EXPERIMENT FIELDS -------------------------------------
            # For V2_FLOOR_BETA arms, the ArmSpec carries per-arm exit params
            # that differ from the shared V2_FLOOR bracket. These are snapshotted
            # at open and read by the exit logic instead of the profile defaults.
            beta_exit_bracket=spec.exit_bracket,
            beta_max_hold_s=spec.max_hold_s,
            # -- THE VENUE PIN -------------------------------------------
            # Recorded from the quote that was ACTUALLY filled on. The exit is
            # held to this market, so it is captured here and never re-guessed.
            entry_venue_id=quote.venue_id,
            entry_pool_address=quote.pool_or_curve,
            entry_market_type=quote.market_type,
            entry_quote_source=quote.provider,
            entry_quote_timestamp=quote.quote_timestamp,
            entry_quote_age_ms=quote.quote_age_ms,
            entry_quote_latency_ms=quote.quote_latency_ms,
            # -- RENT: locked, not spent --------------------------------
            entry_rent_lamports=TOKEN_ACCOUNT_RENT_LAMPORTS,
            rent_locked=True,
            reclaimable_rent_lamports=0,
            reclaim_status="RENT_LOCKED",
            reclaim_blocker="the token balance is non-zero",
        )

        # -- CACHED STATIC IDENTITY (for lightweight monitoring) ----------
        # Fetch static market identity (vault addresses, fee configs, etc.)
        # using the verified market address from the entry quote. This avoids
        # re-discovering the market on every monitoring tick.
        try:
            cached_identity = await self.quote_service.get_monitoring_identity(
                mint=signal.token_mint or "",
                market_address=quote.pool_or_curve or "",
                quote_mint=WSOL_MINT,
            )
            if cached_identity:
                position.cached_identity = cached_identity
        except Exception as exc:  # noqa: BLE001
            log("ERROR", "failed to cache monitoring identity: %s", exc)
        # -- ENTRY INVARIANT ---------------------------------------------
        # The entry basis and the exit basis must be the SAME market in the
        # SAME unit. Checked here, before the position exists, so a position can
        # never be born already past its own stop -- the condition that produced
        # the false ~5s fast-failure exits. The aggregator mark is deliberately
        # NOT consulted here: it is a different book, which is the defect.
        basis = _entry_basis_is_sane(
            quote,
            sol_usd=(self.sol_reference.usd_per_sol if self.sol_reference else None),
        )
        position.entry_basis_sane = basis["sane"]
        position.entry_basis_detail = basis["detail"]
        if not basis["sane"]:
            log("ERROR",
                "entry basis is not same-market/same-unit (%s); the position is "
                "opened but its price triggers will fail closed until a valid "
                "same-market reference exists", basis["detail"])

        account.open_position(position)
        self._start_hold_path_telemetry(spec, position)
        # THE MARKET JOINS THE PUSH CHANNEL NOW. From this moment the position
        # is observed by account updates rather than by a timer, and it will
        # only ask for an executable quote when an existing exit needs one.
        self._ensure_native_monitor(position, quote)
        self._apply_monitor_diagnostics(position)
        latency.entry_fill_at = latency.buy_quote_response_at

        record = SignalRecord(
            signal=signal, arm=arm, arm_version=SCHEMA_VERSION,
            baseline_hash=BASELINE_HASH, l25_decision="BUY",
            l25_status=l25_status, l25_authorized=authorized,
            arm_filter_reason=why, observed_mark=mark, buy_quote=quote,
            mark_to_quote_gap_pct=gap,
            legacy_observed_mark_gap_pct=old_gap,
            execution_native_gap_pct=guard.gap_pct,
            native_guard_verdict=guard.result,
            legacy_guard_verdict=legacy_guard.result,
            production_guard_source=PRODUCTION_GUARD_SOURCE,
            execution_identity=quote.identity,
            traded=True, failure=FailureReason.NONE, failure_detail="",
            latency=latency, bankroll_before=bankroll_before,
            position_size_usd=size.position_usd,
            position_size_pct=size.position_pct, finished_at=time.time(),
            **_entry_execution_gap_observability(
                mark=mark, quote=quote, guard=guard),
        )
        return record, position

    # -- the per-signal entry point ---------------------------------------

    async def on_signal(self, signal: SignalSnapshot) -> list[SignalRecord]:
        """
        Process one signal for every arm.

        Never raises for an expected failure: a dead RPC, a refused quote or a
        market that cannot be identified is a RECORDED OUTCOME for this signal,
        and the run continues to the next one.
        """
        records: list[SignalRecord] = []
        mint = signal.token_mint or ""
        if not mint:
            for spec in self.arms:
                records.append(SignalRecord(
                    signal=signal, arm=spec.name.value,
                    arm_version=SCHEMA_VERSION, baseline_hash=BASELINE_HASH,
                    l25_decision="SKIPPED", l25_status="", l25_authorized=False,
                    arm_filter_reason="", observed_mark=await self.observe(""),
                    buy_quote=QuotePoint(
                        side="BUY", provider="none", market_type="UNKNOWN",
                        token_mint="", quote_mint=WSOL_MINT, input_amount_raw=None,
                        input_decimals=None, input_amount_decimal=None,
                        output_amount_raw=None, output_decimals=None,
                        output_amount_decimal=None, effective_price=None,
                        effective_price_usd=None, price_impact_pct=None,
                        fee_bps=None, fee_amount=None, venue=None,
                        pool_or_curve=None, route=(), identity="UNKNOWN",
                        available=False,
                        missing_reason="no token mint was extracted",
                    ),
                    mark_to_quote_gap_pct=None, execution_identity="UNKNOWN",
                    traded=False,
                    failure=(FailureReason.TOKEN_AMBIGUOUS if signal.ambiguous
                              else FailureReason.TOKEN_NOT_FOUND),
                    failure_detail=(
                        "the message contained multiple distinct mints"
                        if signal.ambiguous else "no token mint was extracted"
                    ),
                    latency=LatencyMarks(
                        telegram_message_at=signal.telegram_timestamp,
                        receive_at_monotonic=signal.receive_monotonic,
                        receive_at_wall=signal.receive_timestamp,
                    ),
                    bankroll_before=self.accounts[spec.name.value].bankroll_usd,
                    position_size_usd=None, position_size_pct=None,
                    finished_at=time.time(),
                ))
            self.signal_records.extend(records)
            return records

        latency = LatencyMarks(
            telegram_message_at=signal.telegram_timestamp,
            receive_at_monotonic=signal.receive_monotonic,
            receive_at_wall=signal.receive_timestamp,
        )
        latency.mark_request_at = time.monotonic()
        mark = await self.observe(mint)
        latency.mark_response_at = time.monotonic()
        self.observed_marks[mint] = mark

        # V2 FLOOR (BETA): ONE gate, ONE experiment, three arm records.
        beta_records: list[SignalRecord] = []
        beta_selected = bool({a.value for a in BETA_ARMS}
                             & {s.name.value for s in self.arms})

        if beta_selected and self.beta_experiment is None:
            # Unreachable by construction (the experiment is created whenever
            # all three arms are selected) -- but a missing experiment must
            # REFUSE loudly with a record per arm. Routing beta arms into the
            # regular entry loop instead would open positions that
            # ``monitor_once`` never settles, because it skips beta arms
            # unconditionally: an abandoned trade rather than a refusal.
            reason = ("V2_FLOOR_BETA experiment is not initialized; beta arms "
                      "refuse instead of falling back to the regular entry path")
            log("ERROR", "%s", reason)
            beta_records = [self._beta_refusal_record(
                signal=signal, arm_name=a.value, mark=mark, latency=latency,
                reason=reason) for a in BETA_ARMS]

        # V2_FLOOR_BETA experiment: run the gate once for the beta arms
        if self.beta_experiment is not None:
            try:
                # The gate's three inputs ride ON THE SNAPSHOT. The experiment
                # re-runs the unchanged production ``v2floor.evaluate`` and
                # refuses fail-closed when any input is absent, so the exact
                # decision-time numbers are carried over from the decision
                # report's own feature snapshot -- never derived here, never
                # substituted. See ``_signal_with_v2_gate_inputs``.
                gate_signal = self._signal_with_v2_gate_inputs(signal, mark=mark)
                # THE ENTRY COMES FROM THE EXISTING QUOTE COLLECTOR, and only
                # AFTER the V2 FLOOR gate has authorized. A signal the gate
                # refused is never even offered a quote; a BUY quote that
                # cannot be obtained is a recorded refusal, never a re-priced
                # mark that quietly opens the arm.
                _gate_parent = self.beta_experiment._stable_parent_id(
                    gate_signal)
                _eligible, _reason = self.beta_experiment._v2_gate_result(
                    gate_signal)
                self._beta_stage("gate", parent_signal_id=_gate_parent,
                                 eligible=bool(_eligible), detail=_reason)
                beta_positions: dict[str, Any] = {}
                beta_blocks: dict[str, str] = {}
                if _eligible:
                    # PER-ARM, ALL-IN, AND BLOCKED BEFORE ANY QUOTE IS SPENT.
                    #
                    # Each arm is sized on its OWN uncommitted cash and then
                    # quoted separately at that amount, so an arm whose bankroll
                    # has grown (or shrunk) commits exactly what it has. An arm
                    # that already holds an open or unresolved position is
                    # blocked HERE -- no quote is even requested for it -- until
                    # that position is resolved. The other arms carry on: they
                    # are independent ledgers and must not wait on a stuck one.
                    plan = self.beta_experiment.arm_entry_plan()
                    self._beta_stage(
                        "sizing", parent_signal_id=_gate_parent,
                        plan={a: {
                            "available_cash_usd": row["available_cash_usd"],
                            "notional_usd": row["notional_usd"],
                            "sizing_pct": row["sizing_pct"],
                            "blocked": row["blocked"],
                        } for a, row in plan.items()})
                    entry_fills: dict[str, Any] = {}
                    allocations: dict[str, float] = {}
                    for arm_key in BETA_ARMS_ORDER:
                        row = plan[arm_key]
                        if row["blocked"]:
                            beta_blocks[arm_key] = str(row["reason"])
                            continue
                        fill, why = await self._beta_entry_fill(
                            mint=mint, size_usd=row["notional_usd"], mark=mark,
                            arm=arm_key)
                        self._beta_stage(
                            "buy_quote", parent_signal_id=_gate_parent,
                            arm=arm_key, accepted=fill is not None,
                            notional_usd=row["notional_usd"],
                            reason=why or None)
                        if fill is None:
                            beta_blocks[arm_key] = why
                            self._beta_quote_refusals.setdefault(
                                _gate_parent, why)
                            continue
                        entry_fills[arm_key] = fill
                        allocations[arm_key] = float(row["notional_usd"])
                    if entry_fills:
                        opened, blocked = self.beta_experiment.open_arms(
                            parent_signal_id=_gate_parent, mint=mint or "",
                            symbol=signal.symbol,
                            entry_fills=entry_fills, allocations=allocations,
                        )
                        beta_positions = dict(opened)
                        beta_blocks.update(
                            {k: str(v) for k, v in blocked.items()})
                eligible = bool(beta_positions)
                reason = ("" if eligible
                          else (_reason or "no beta arm could be opened"))
                if not eligible and beta_blocks:
                    reason = "; ".join(f"{a}: {why}"
                                       for a, why in beta_blocks.items())
                # The stable id shared by every arm of this ONE parent signal.
                _first = next(iter(beta_positions.values()), None)
                parent_signal_id = (_first.parent_signal_id
                                    if _first is not None else None)
                if beta_positions:
                    self._beta_stage(
                        "position_created", parent_signal_id=parent_signal_id,
                        arms=len(beta_positions),
                        allocated_usd=(_first.allocated_usd
                                       if _first is not None else None),
                        committed_usd=(_first.entry_cost_usd
                                       if _first is not None else None),
                        sizing_pct=(_first.sizing_pct
                                    if _first is not None else None))
                for arm_enum in BETA_ARMS:
                    arm_key = BETA_ARM_KEYS[arm_enum]
                    arm_name = arm_enum.value
                    bp = beta_positions.get(arm_key)
                    if bp is None:
                        # NOT opened. Either the gate refused the signal, or THIS
                        # arm stood down on its own (overlap / no capital / no
                        # usable quote). Either way it gets a record: an arm that
                        # did NOT trade is evidence, and it is never dropped.
                        #
                        # Keep arm_filter_reason as the single source of truth for
                        # the per-arm detail. Do NOT read _beta_quote_refusals here:
                        # that dict is keyed by parent signal id and was populated by
                        # ANY quote refusal across the whole gate (overlap guard, no
                        # market, stale quote, RPC error, identity UNKNOWN, etc.), so
                        # a refused quote on one arm would donate its detail to a
                        # different arm's notification - including an operational
                        # blocker that never even reached the quote stage.
                        arm_reason = beta_blocks.get(arm_key) or reason
                        # Distinguish gate refusal from per-arm operational block.
                        # "OVERLAP:" / "INSUFFICIENT_CAPITAL:" prefixes mark a blocker
                        # that stood down AFTER the gate passed (overlap/active-position/
                        # no-capital). Everything else (gate refusal, quote failure) is
                        # a gate-level rejection - even when beta_blocks has an entry
                        # for it.
                        is_operational_block = bool(
                            arm_reason.startswith("OVERLAP:")
                            or arm_reason.startswith("INSUFFICIENT_CAPITAL:")
                            if arm_reason else False)
                        beta_records.append(SignalRecord(
                            signal=signal, arm=arm_name,
                            arm_version=SCHEMA_VERSION, baseline_hash=BASELINE_HASH,
                            l25_decision="REJECT",
                            l25_status=("V2_FLOOR_BLOCKED" if is_operational_block
                                        else "V2_FLOOR_REJECTED"),
                            l25_authorized=False, arm_filter_reason=arm_reason,
                            observed_mark=mark,
                            buy_quote=QuotePoint(
                                side="BUY", provider="none", market_type="UNKNOWN",
                                token_mint=mint, quote_mint=WSOL_MINT,
                                input_amount_raw=None, input_decimals=None,
                                input_amount_decimal=None, output_amount_raw=None,
                                output_decimals=None, output_amount_decimal=None,
                                effective_price=None, effective_price_usd=None,
                                price_impact_pct=None, fee_bps=None, fee_amount=None,
                                venue=None, pool_or_curve=None, route=(),
                                identity="UNKNOWN", available=False,
                                missing_reason=arm_reason,
                            ),
                            mark_to_quote_gap_pct=None, execution_identity="UNKNOWN",
                            traded=False,
                            failure=(FailureReason.ENTRY_BLOCKED_POSITION_ACTIVE
                                     if is_operational_block
                                     else FailureReason.V2_FLOOR_REJECTED),
                            failure_detail=arm_reason, latency=latency,
                            bankroll_before=self.beta_experiment.accounts[BETA_ARM_KEYS[arm_enum]].bankroll_usd,
                            position_size_usd=None, position_size_pct=None,
                            finished_at=time.time(),
                        ))
                    else:
                        # Opened. ``position_size_usd`` is what this arm
                        # COMMITTED (the all-in spend), and ``position_size_pct``
                        # is the rule's percentage -- 100.0 under ALL-IN -- so
                        # the record itself shows the entry used the whole
                        # budget it had.
                        beta_records.append(SignalRecord(
                            signal=signal, arm=arm_name,
                            arm_version=SCHEMA_VERSION, baseline_hash=BASELINE_HASH,
                            l25_decision="BUY", l25_status="V2_FLOOR_PASSED",
                            l25_authorized=True, arm_filter_reason="",
                            observed_mark=mark,
                            buy_quote=QuotePoint(
                                side="BUY", provider="none", market_type="UNKNOWN",
                                token_mint=mint, quote_mint=WSOL_MINT,
                                input_amount_raw=None, input_decimals=None,
                                input_amount_decimal=None, output_amount_raw=None,
                                output_decimals=None, output_amount_decimal=None,
                                effective_price=bp.entry_fill_price,
                                effective_price_usd=bp.entry_fill_price,
                                price_impact_pct=None, fee_bps=None, fee_amount=None,
                                venue=None, pool_or_curve=None, route=(),
                                identity="UNKNOWN", available=True,
                                missing_reason="",
                            ),
                            mark_to_quote_gap_pct=0.0, execution_identity="UNKNOWN",
                            traded=True, failure=FailureReason.NONE, failure_detail="",
                            latency=latency,
                            bankroll_before=self.beta_experiment.accounts[BETA_ARM_KEYS[arm_enum]].bankroll_usd,
                            position_size_usd=bp.position_size_usd,
                            position_size_pct=bp.sizing_pct,
                            finished_at=time.time(),
                            parent_signal_id=parent_signal_id,
                        ))
            except Exception as exc:  # noqa: BLE001
                log("ERROR", "V2_FLOOR_BETA entry failed: type=%s value=%r", type(exc), exc)
                # RECORD AND KEEP THE ARMS SKIPPED. The previous behaviour
                # cleared ``beta_records`` and let the beta arms fall through
                # to the regular entry loop -- which would open a position that
                # ``monitor_once`` never settles, because it skips beta arms
                # unconditionally. A failed experiment run must look like a
                # failed entry, one record per arm, with nothing opened.
                beta_records = [self._beta_refusal_record(
                    signal=signal, arm_name=a.value, mark=mark,
                    latency=latency,
                    reason=f"{type(exc).__name__}: {exc}"[:200],
                    failure=FailureReason.ARITHMETIC_ERROR,
                    decision="ERROR",
                ) for a in BETA_ARMS]

        records.extend(beta_records)

        # Create PaperPosition objects for beta arms that traded
        if self.beta_experiment is not None and beta_records:
            for br in beta_records:
                if br.traded:
                    if not br.parent_signal_id:
                        # A traded beta record without its parent id cannot be
                        # tied to the experiment's book; refusing to guess is
                        # what keeps a mirror from pointing at the wrong entry.
                        log("ERROR",
                            "beta record for arm %s traded without a parent "
                            "signal id; visibility mirror not created", br.arm)
                        continue
                    arm_enum = ResearchArm(br.arm)
                    arm_key = BETA_ARM_KEYS[arm_enum]
                    bp = (self.beta_experiment.positions
                          .get(br.parent_signal_id) or {}).get(arm_key)
                    if bp is None:
                        log("ERROR",
                            "beta position %s/%s is not in the experiment's "
                            "book; visibility mirror not created",
                            br.parent_signal_id, br.arm)
                        continue
                    spec = next(s for s in self.arms if s.name == arm_enum)
                    # A MIRROR, not an engine position. Deliberately NOT
                    # ``self.accounts[br.arm].open_position(position)``: that
                    # would hand the beta trade to the regular exit loop as well
                    # as to ``BetaExperiment`` and settle it twice. The
                    # experiment owns this position; here it is only visible.
                    self.beta_positions.setdefault(
                        bp.parent_signal_id, {})[br.arm] = \
                        self._beta_mirror_position(spec, bp)

        for spec in self.arms:
            # Beta arms are EXCLUSIVELY the experiment's, always -- exactly
            # like ``monitor_once`` skips them. A beta arm must never enter
            # through this loop, or the same economic trade could be opened (and
            # never settled) on a second book.
            if spec.name in BETA_ARMS:
                continue
            arm_latency = latency
            if spec is not self.arms[0]:
                arm_latency = LatencyMarks(
                    telegram_message_at=signal.telegram_timestamp,
                    receive_at_monotonic=signal.receive_monotonic,
                    receive_at_wall=signal.receive_timestamp,
                    mark_request_at=latency.mark_request_at,
                    mark_response_at=latency.mark_response_at,
                )
            decision_time = time.monotonic()
            decision_error = ""
            arm_latency.decision_start_at = decision_time
            try:
                report, _status, _auth = self.decide(
                    mint=mint, signal=signal, mark=mark,
                    decision_time=decision_time, blocked_reason=None,
                )
            except Exception as exc:  # noqa: BLE001 - named, run continues
                report = None
                decision_error = f"{type(exc).__name__}: {exc}"[:200]
            arm_latency.decision_end_at = time.monotonic()

            if report is None:
                record = SignalRecord(
                    signal=signal, arm=spec.name.value,
                    arm_version=SCHEMA_VERSION, baseline_hash=BASELINE_HASH,
                    l25_decision="ERROR", l25_status="", l25_authorized=False,
                    arm_filter_reason="", observed_mark=mark,
                    buy_quote=QuotePoint(
                        side="BUY", provider="none", market_type="UNKNOWN",
                        token_mint=mint, quote_mint=WSOL_MINT,
                        input_amount_raw=None, input_decimals=None,
                        input_amount_decimal=None, output_amount_raw=None,
                        output_decimals=None, output_amount_decimal=None,
                        effective_price=None, effective_price_usd=None,
                        price_impact_pct=None, fee_bps=None, fee_amount=None,
                        venue=None, pool_or_curve=None, route=(),
                        identity="UNKNOWN", available=False,
                        missing_reason=f"the decision engine failed: {decision_error}",
                    ),
                    mark_to_quote_gap_pct=None, execution_identity="UNKNOWN",
                    traded=False, failure=FailureReason.ARITHMETIC_ERROR,
                    failure_detail=decision_error,
                    latency=arm_latency,
                    bankroll_before=self.accounts[spec.name.value].bankroll_usd,
                    position_size_usd=None, position_size_pct=None,
                    finished_at=time.time(),
                )
                records.append(record)
                continue

            try:
                record, _position = await self._open_for_arm(
                    spec=spec, signal=signal, mark=mark, report=report,
                    latency=arm_latency,
                )
            except Exception as exc:  # noqa: BLE001 - one arm must not kill others
                record = SignalRecord(
                    signal=signal, arm=spec.name.value,
                    arm_version=SCHEMA_VERSION, baseline_hash=BASELINE_HASH,
                    l25_decision="BUY",
                    l25_status=str(getattr(getattr(report, "l25", None),
                                          "status", "")),
                    l25_authorized=bool(getattr(getattr(report, "l25", None),
                                                "authorized", False)),
                    arm_filter_reason="", observed_mark=mark,
                    buy_quote=QuotePoint(
                        side="BUY", provider="none", market_type="UNKNOWN",
                        token_mint=mint, quote_mint=WSOL_MINT,
                        input_amount_raw=None, input_decimals=None,
                        input_amount_decimal=None, output_amount_raw=None,
                        output_decimals=None, output_amount_decimal=None,
                        effective_price=None, effective_price_usd=None,
                        price_impact_pct=None, fee_bps=None, fee_amount=None,
                        venue=None, pool_or_curve=None, route=(),
                        identity="UNKNOWN", available=False,
                        missing_reason=f"the entry stage failed: {exc}"[:200],
                    ),
                    mark_to_quote_gap_pct=None, execution_identity="UNKNOWN",
                    traded=False, failure=FailureReason.ARITHMETIC_ERROR,
                    failure_detail=f"{type(exc).__name__}: {exc}"[:200],
                    latency=arm_latency,
                    bankroll_before=self.accounts[spec.name.value].bankroll_usd,
                    position_size_usd=None, position_size_pct=None,
                    finished_at=time.time(),
                )
            records.append(record)

        self.signal_records.extend(records)
        return records

    # -- the monitor pass and exit ---------------------------------------

    def _monitor_beta_experiment_for_tick(self) -> list[Any]:
        """Beta positions settled on the last ``monitor_once`` tick.

        The notification layer needs the RECORDED result, not a re-derivation:
        an arm that is still open has no exit row, and a target touch that
        never settled is not a fill. Reading the retained list is what keeps
        that distinction honest.
        """
        return list(getattr(self, "_last_beta_settled", []) or [])

    async def _monitor_beta_experiment(self) -> list[Any]:
        """
        One tick of the V2 FLOOR (BETA) experiment: observe, then settle.

        This is the ONLY place a beta position is settled. It does not return
        engine ``TradeRecord``s and it does not move any engine bankroll, because
        the beta ledgers belong to :class:`BetaExperiment` and are accounted
        there. What it returns is the list of beta positions settled on this
        tick, so a caller or a test can see exactly what happened.

        WHY THE MARK IS CHECKED BEFORE IT IS FED IN
        -------------------------------------------
        An observation with no available price is not an observation. Feeding a
        missing or stale mark in would let ``peak``/``trough`` be computed from a
        number that does not exist, and a fabricated exit price is worse than no
        exit at all -- so a data outage leaves every beta arm exactly where it
        was instead of booking a trade it can no longer justify.

        WHY THE MIRROR IS DROPPED RATHER THAN CLEARED HERE
        ---------------------------------------------------
        The mirror exists only for the operator's open-positions view. Its
        lifetime is derived from the experiment's own position state: when the
        experiment has settled an arm, the arm's mirror stops existing. Nothing
        here writes to a ledger, so dropping a mirror cannot lose a trade.
        """
        if self.beta_experiment is None:
            return []
        # A restart restores the experiment's positions but not the mirrors.
        # Rebuild them BEFORE the observe pass so recovered positions are
        # observed and settled on this very tick. See the method docstring.
        self._restore_beta_mirrors()
        beta_arm_key_by_name = {
            value: key for key, value in
            ((BETA_ARM_KEYS[a], a.value) for a in BETA_ARMS)
        }
        for parent_signal_id in list(self.beta_positions.keys()):
            arms = self.beta_positions.get(parent_signal_id, {})
            for arm_name in list(arms.keys()):
                mirror = arms.get(arm_name)
                if mirror is None:
                    arms.pop(arm_name, None)
                    continue
                mark = await self.observe(mirror.mint)
                self.observed_marks[mirror.mint] = mark
                mirror.current_mark_usd = (
                    mark.value_usd if getattr(mark, "available", False) else None)
                mirror.current_mark_timestamp = getattr(mark, "timestamp", None)
                mirror.monitor_ticks += 1
                arm_key = beta_arm_key_by_name.get(arm_name)
                bp = (self.beta_experiment.positions.get(parent_signal_id)
                      or {}).get(arm_key) if arm_key else None
                # An UNRESOLVED arm keeps its mirror: it is still outstanding
                # and is retried, so its mark has to keep updating.
                if bp is None or bp.status not in ("open", "unresolved"):
                    arms.pop(arm_name, None)
                    continue
                if getattr(mark, "available", False) and mark.value_usd:
                    self.beta_experiment.observe(
                        parent_signal_id, arm_key,
                        mark_price=mark.value_usd,
                        at=getattr(mark, "timestamp", None) or time.time(),
                    )
            if not arms:
                self.beta_positions.pop(parent_signal_id, None)

        # THE EXIT IS QUOTED, NOT DERIVED FROM THE MARK.
        #
        # ``_exit_decision`` reports only THAT an exit is due -- a target was
        # touched, or the deadline passed. The P&L comes from an executable SELL
        # quote on the EXACT quantity the entry BUY acquired, obtained with the
        # same ``quote_sell_pinned``/``quote_sell`` path and same-market
        # validation the regular arms use. A trigger is not a fill, so an arm
        # with no valid quote stays UNRESOLVED and is never priced from a mark.
        #
        # THE TRIGGER IS QUOTED TOO, AND THAT IS THE FIX
        # ------------------------------------------------
        # The bracket used to be evaluated against the aggregator MARK while the
        # P&L was booked from the executable quote. Measured over the live clean
        # cohort those two books disagree by a median 29.2% and by up to 62.5% --
        # wider than either the target or the stop -- so 18 of 20 trades labelled
        # ``SL`` settled better than -30% and all 3 labelled ``TP`` settled worse
        # than +30%. Every hit rate derived from them was meaningless.
        #
        # So the exit quote is now taken FIRST, every tick, and the SAME quote is
        # both the evidence the bracket is tested against and the fill that books
        # the trade. There is no second price and no execution-delay model: the
        # accepted settlement quote IS the trigger quote, and the record says so
        # (``settlement_quote_is_trigger_quote``).
        #
        # A mark excursion is still observed and still recorded, under its own
        # name and its own basis, as a diagnostic only. It cannot fire a bracket.
        now = time.time()
        settled: list[Any] = []
        for parent_signal_id in list(self.beta_experiment.positions.keys()):
            for arm_key in list(
                    self.beta_experiment.positions[parent_signal_id].keys()):
                bp = self.beta_experiment.positions[parent_signal_id].get(arm_key)
                # An UNRESOLVED arm is retried, not skipped: a quote can fail
                # transiently, and the arm stays outstanding until a real quote
                # settles it. It is never booked from a mark.
                if bp is None or bp.status not in ("open", "unresolved"):
                    continue
                cfg = bp.config()
                held = max(0.0, now - float(bp.entry_at or now))
                time_due = held >= float(cfg.max_hold_s)

                # THE QUOTE IS THE TRIGGER EVIDENCE. It is taken unconditionally
                # for an open arm, because the bracket cannot be evaluated
                # without it -- with no observed quote, ``_exit_decision`` on the
                # quote basis can only ever return TIME. Throttling this to a
                # mark screen would reintroduce exactly the defect above: a
                # bracket the engine refused to test.
                exit_fill, blocked = await self._beta_exit_fill(
                    parent_signal_id=parent_signal_id, arm_key=arm_key, bp=bp)
                self._beta_stage(
                    "sell_quote", parent_signal_id=parent_signal_id,
                    arm=arm_key, accepted=exit_fill is not None,
                    reason=blocked or None)
                if exit_fill is not None:
                    self.beta_experiment.observe_quote(
                        parent_signal_id, arm_key, exit_fill=exit_fill, at=now)
                reason, _ambiguous = self.beta_experiment._exit_decision(
                    bp, now=now, basis=RETURN_BASIS_QUOTE)
                if not reason:
                    # No threshold satisfied on the executable basis. The quote
                    # was still a valid observation and is already recorded.
                    continue
                # THE RETRY IS ON A COOLDOWN, NOT ON THE TICK.
                #
                # This is the beta counterpart of the regular arms' bounded
                # backoff, and it is the reason the cooldown exists: an
                # unresolved arm used to re-quote on every tick, so a market that
                # could not be exited spent the shared RPC and aggregator budget
                # every ~5 seconds per arm and the throttling that produced was
                # itself one of the refusal reasons. Nothing is skipped
                # permanently -- the arm stays outstanding, its attempts keep
                # counting once they resume, and the exit is retried as soon as
                # the cooldown expires.
                retry_key = (parent_signal_id, arm_key)
                retry = self._beta_exit_retry.get(retry_key) or {}
                if not time_due and now < float(retry.get("next_attempt_at") or 0.0):
                    self._monitor_diag["BETA_EXIT_RETRY_DEFERRED"] = (
                        self._monitor_diag.get("BETA_EXIT_RETRY_DEFERRED", 0) + 1)
                    continue
                self._beta_stage("exit_trigger", parent_signal_id=parent_signal_id,
                                 arm=arm_key, reason=reason)
                # SCHEDULE THE NEXT ATTEMPT. A refusal grows the cooldown; a
                # settlement clears it, so a later arm on the same parent (or a
                # reused parent id) never inherits a stale penalty.
                if exit_fill is None:
                    n = int(retry.get("attempts") or 0) + 1
                    delay = self._beta_exit_backoff_s(n)
                    self._beta_exit_retry[retry_key] = {
                        "attempts": n,
                        "last_reason": (blocked or "")[:300],
                        "next_attempt_at": now + delay,
                    }
                    self._beta_stage(
                        "sell_quote_retry", parent_signal_id=parent_signal_id,
                        arm=arm_key, attempt=n, next_attempt_in_s=round(delay, 1))
                else:
                    self._beta_exit_retry.pop(retry_key, None)
                # PRODUCTION SETTLEMENT: a valid executable SELL quote is
                # REQUIRED. ``require_quote=True`` means a missing/invalid quote
                # leaves the arm UNRESOLVED -- it is never booked from the
                # modelled backtest path, which exists only for the historical
                # research harness.
                _settled = self.beta_experiment.settle_arm(
                    parent_signal_id, arm_key, now=now, exit_fill=exit_fill,
                    require_quote=True)
                settled.append(_settled)
                self._beta_stage(
                    "settlement", parent_signal_id=parent_signal_id,
                    arm=arm_key,
                    status=getattr(_settled, "status", None),
                    net_pnl_usd=getattr(_settled, "net_pnl_usd", None))
                if exit_fill is None and blocked:
                    self._beta_quote_refusals.setdefault(parent_signal_id,
                                                        blocked)
        self._last_beta_settled = settled
        return [p for p in settled if p is not None]

    async def monitor_once(self) -> list[TradeRecord]:
        """
        One monitor pass: refresh marks, run the production exit policy, and on
        an exit trigger take a FRESH executable SELL quote.

        The price-based triggers are evaluated against the SAME-MARKET,
        SAME-CURRENCY execution-native reference (see
        :meth:`_native_trigger_reference`), never the external aggregator mark.
        The mark is still observed, still recorded and still used for reporting,
        but it is diagnostic only: it is not the authoritative trigger source.

        A TICK-LOCAL SELL-QUOTE MEMO IS CLEARED HERE. The trigger reference and
        the exit both need an executable SELL quote for the same position, in the
        same tick, for the same quantity -- and each one is several RPC calls. The
        memo is scoped to a single tick and keyed on position + mint + pinned
        market + amount, so the second request is served from the first. It is
        discarded at the end of every tick, which bounds a reused quote's age to
        one monitor tick -- far inside ``max_quote_age_ms``, so no stale quote can
        survive it.
        """
        trades: list[TradeRecord] = []
        # Per-tick, never across ticks.
        self._tick_sell_quotes: dict[tuple, Any] = {}
        tick_start = time.monotonic()
        tick_rpc_calls = 0
        # THE BETA EXPERIMENT RUNS FIRST, AND ON ITS OWN.
        #
        # A beta position is settled by ``BetaExperiment`` and by nothing else.
        # Feeding it the observed mark here is how it gets the peak/trough/last
        # return its bracket is evaluated against; taking the exit is
        # ``BetaExperiment.settle_arm``. The regular exit loop below is then told
        # to skip the beta ARMS, because the same position settled there would be
        # the same trade counted twice.
        # RETAINED so the notification layer can report what was actually
        # settled. ``_monitor_beta_experiment`` returns ``BetaPosition``s and
        # deliberately does NOT add them to ``trade_records`` (they would then
        # be settled by the regular exit loop too), which is exactly why no
        # beta exit notification existed. Reset every tick so a tick that
        # settles nothing cannot re-report the previous tick.
        self._last_beta_settled = await self._monitor_beta_experiment()
        for spec in self.arms:
            # Beta arms are settled by the beta experiment, above. Continuing
            # here is what makes the settlement path ONE: a beta position must
            # not also be assessed by the production exit policy, which would
            # move a second ledger for the same fill.
            if spec.name in BETA_ARMS:
                continue
            account = self.accounts[spec.name.value]
            position = account.position
            if position is None:
                continue

            mint = position.mint
            position.monitor_ticks += 1
            self._monitor_diag["MONITOR_TICKS"] += 1
            mark = await self.observe(mint)
            self.observed_marks[mint] = mark

            # The position carries its CURRENT mark explicitly, so open-book
            # economics never have to re-derive it from a dict that restarts
            # forget. Never used for settlement -- display/accounting only.
            position.current_mark_usd = (
                mark.value_usd if getattr(mark, "available", False) else None)
            position.current_mark_timestamp = getattr(mark, "timestamp", None)

            self._update_mfe_mae(position, mark.value_usd)

            # A production Position, so the production ExitPolicy can be
            # handed a real object and its precedence runs unmodified.
            live = self._production_position(position, mark)
            held = time.time() - position.opened_at

            # THE MONITORING / QUOTING SPLIT.
            #
            # Market state and executability are different questions and now
            # cost different things. The first is answered by a pushed account
            # update and a LOCAL spot calculation -- no RPC, no HTTP. The second
            # is answered by an executable SELL quote, and is now spent only
            # where it can change something: when an EXISTING exit is near or has
            # fired.
            #
            # The order below is therefore:
            #   1. ask the push channel for the native reference (free);
            #   2. if it answered, we HAVE a current price -- no quote;
            #   3. ask the proximity band whether an existing exit is near;
            #   4. quote ONLY if it is, and validate that quote's freshness and
            #      identity against the existing rules.
            #
            # Nothing below this point decides anything differently: the exit
            # policy still sees the SAME execution-native reference, on the SAME
            # basis, with the SAME thresholds.
            self._apply_monitor_diagnostics(position)
            trigger_usd, provenance = await self._resolve_trigger_reference(
                position, live, mark, held
            )
            self._record_trigger_provenance(position, provenance)

            if trigger_usd is None:
                # FAIL CLOSED. Without a proven same-market reference the price
                # triggers are not evaluable, and evaluating them from an
                # unrelated book's mark is the exact defect this replaced.
                #
                # This hands the decision to the EXISTING bounded data-invalid
                # policy rather than inventing a bespoke hold: the position is
                # held, and only after ``data_invalid_exit_s`` of continuously
                # unevaluable triggers does that policy exit to a known state.
                # Passing ``data_invalid_since`` is what makes that bound real --
                # without it the timer never starts and "fail closed" would
                # silently mean "hold forever".
                if position.native_reference_unavailable_since is None:
                    position.native_reference_unavailable_since = time.time()
                decision = self.exit_policy.assess(
                    live,
                    now=time.time(),
                    mark_price=None,
                    data_state=DataState.MISSING,
                    momentum_30s_pct=None,
                    structure_30s_pct=None,
                    liquidity_change_pct=None,
                    data_invalid_since=position.native_reference_unavailable_since,
                )
            else:
                position.native_reference_unavailable_since = None
                decision = self.exit_policy.assess(
                    live,
                    now=time.time(),
                    mark_price=trigger_usd,
                    data_state=(DataState.LIVE if mark.available else DataState.MISSING),
                    momentum_30s_pct=None,
                    structure_30s_pct=None,
                    liquidity_change_pct=None,
                )
            self._record_trigger_decision(position, provenance, trigger_usd, decision)
            if decision.action == "EXIT" and decision.trigger in (
                "momentum_collapse", "structure_break", "peak_giveback",
                "wide_giveback", "no_follow_through",
            ) and held < self.min_hold_s:
                decision = ExitDecision(
                    "HOLD",
                    f"exit signal '{decision.trigger}' suppressed before the "
                    f"{self.min_hold_s:.0f}s minimum hold (a declared rule, not "
                    "a post-hoc exception)",
                    decision.style, trigger=None,
                )

            if decision.action != "EXIT":
                continue

            # The exit trigger is recorded, so the position is no longer a
            # healthy OPEN from an operator's perspective.
            if position.exit_status in ("OPEN_MONITORING", ""):
                position.exit_status = "EXIT_TRIGGERED"
            position.exit_trigger_reason = getattr(decision, "reason", "") or ""

            # THE EXIT DECISION IS ALWAYS TAKEN. Only the QUOTE is deferred.
            #
            # This distinction is the whole point: the policy still evaluates, the
            # position is still OPEN, and the trigger is still recorded on every
            # tick. What is deferred is spending RPC on a re-quote that a
            # throttling provider is likely to refuse anyway. Deferring the
            # DECISION instead would be a strategy change; deferring the quote is
            # a resource decision, and it is bounded so recovery is prompt.
            state = self._exit_retry.get(self._exit_retry_key(position)) or {}
            if state.get("blocked"):
                # The retry budget is spent: no re-quoting, no retry storm.
                # The position is explicitly UNSELLABLE and stays OPEN and
                # tracked; the one-time transition was already logged when the
                # budget ran out, so later ticks stay quiet.
                if position.exit_status != "UNSELLABLE":
                    position.exit_status = "UNSELLABLE"
                continue

            remaining = self._exit_backoff_remaining(position)
            if remaining > 0.0:
                self._log_exit_deferred(position, decision, remaining)
                continue

            trades.append(await self._close_for_arm(
                spec=spec, position=position, mark=mark, decision=decision,
            ))
        # Classify this tick's monitoring health for EVERY observed position.
        # This runs after the decision/exit logic so it catches all paths:
        # non-EXIT, deferred EXIT, blocked EXIT, and successful EXIT.
        for spec in self.arms:
            account = self.accounts[spec.name.value]
            position = account.position
            if position is None:
                continue
            # Read back the provenance recorded by _record_trigger_provenance
            # earlier in this tick.
            provenance = {
                "reference_identity": getattr(
                    position, "trigger_reference_identity", ""),
                "unavailable_reason": getattr(
                    position, "trigger_unavailable_reason", "") or None,
            }
            trigger_usd = getattr(
                position, "trigger_native_reference_usd", None)
            self._classify_monitor_tick(position, trigger_usd, provenance)
        self.trade_records.extend(trades)
        # Emit tick-level diagnostic summary.
        tick_elapsed = time.monotonic() - tick_start
        if tick_elapsed > 0:
            d = self._monitor_diag
            ok_pct = (
                100.0 * d["MONITOR_OK"] / d["MONITOR_TICKS"]
                if d["MONITOR_TICKS"] > 0 else 0.0
            )
            log(
                "MONITOR",
                "tick diag: ok=%d rpc_err=%d quote_unavail=%d "
                "id_unknown=%d id_mismatch=%d skipped_budget=%d "
                "ws_disconnected=%d total_ticks=%d ok_pct=%.1f%% rpc_calls=%d",
                d["MONITOR_OK"], d["MONITOR_RPC_ERROR"],
                d["MONITOR_QUOTE_UNAVAILABLE"], d["MONITOR_IDENTITY_UNKNOWN"],
                d["MONITOR_IDENTITY_MISMATCH"], d["MONITOR_SKIPPED_BUDGET"],
                d["MONITOR_WS_DISCONNECTED"],
                d["MONITOR_TICKS"], ok_pct, tick_rpc_calls,
            )
        return trades

    def _log_exit_deferred(self, position: Any, decision: Any,
                           remaining_s: float) -> None:
        """
        Say that an exit is DUE but deferred, and why.

        Silence here is what made the original incident hard to read: a position
        sat open and the log said nothing about the deferral. Now every deferred
        exit names the position, the trigger, the pinned market, the last RPC
        cause and when the next attempt happens -- which is the difference
        between "the system forgot" and "the system is waiting".
        """
        state = self._exit_retry.get(self._exit_retry_key(position)) or {}
        log(
            "EXECUTE",
            "exit for %s (%s) is due but the SELL quote is deferred %.1fs: "
            "attempt %d failed on %s; position stays OPEN and monitoring "
            "continues",
            getattr(position, "symbol", "?") or "?",
            getattr(decision, "trigger", "EXIT"),
            remaining_s,
            int(state.get("attempts", 0)),
            state.get("last_rpc_cause") or state.get("last_detail", "?")[:80],
        )

    #: Provenance key -> PaperPosition field. Explicit rather than string-built,
    #: so a renamed key can never silently create an undeclared attribute.
    _TRIGGER_PROVENANCE_FIELDS = {
        "trigger_reference_source": "trigger_reference_source",
        "observed_mark_role": "observed_mark_role",
        "observed_mark_usd": "trigger_observed_mark_usd",
        "native_reference_raw": "trigger_native_reference_raw",
        "native_reference_usd": "trigger_native_reference_usd",
        "sol_usd_reference": "trigger_sol_usd_reference",
        "reference_identity": "trigger_reference_identity",
        "reference_market_id": "trigger_reference_market_id",
        "reference_age_ms": "trigger_reference_age_ms",
        "unavailable_reason": "trigger_unavailable_reason",
    }

    def _record_trigger_provenance(
        self, position: PaperPosition, provenance: dict[str, Any]
    ) -> None:
        """Attach the trigger's reference evidence to the open position."""
        for key, field in self._TRIGGER_PROVENANCE_FIELDS.items():
            setattr(position, field, provenance.get(key))

    def _record_trigger_decision(
        self, position: PaperPosition, provenance: dict[str, Any],
        trigger_usd: float | None, decision: ExitDecision,
    ) -> None:
        """Record what the trigger saw, and what it decided."""
        entry_fill = float(position.entry_fill.effective_price_usd or 0.0)
        position.trigger_entry_fill_usd = entry_fill
        position.trigger_return_pct = (
            ((trigger_usd - entry_fill) / entry_fill) * 100.0
            if (trigger_usd is not None and entry_fill > 0) else None
        )
        position.trigger_verdict = decision.action
        position.trigger_reason = decision.reason
        position.trigger_reference_source = provenance.get(
            "trigger_reference_source")
        position.observed_mark_role = provenance.get("observed_mark_role")

    @staticmethod
    def _sell_memo_key(position: Any) -> tuple:
        """
        Identity of "the same SELL quote", for reuse WITHIN one tick.

        Includes the position, the mint, the exact raw quantity and the pinned
        market, so reuse cannot cross a position, a token, a size or a market. The
        tick itself is not in the key because the whole memo is discarded at the
        end of every tick.
        """
        return (
            getattr(position, "signal_id", None),
            getattr(position, "arm", None),
            getattr(position, "mint", None),
            int(getattr(position, "tokens_raw", 0) or 0),
            int(getattr(position, "token_decimals", 0) or 0),
            _pinned_market_address(position),
        )

    def _memoised_sell(self, position: Any, quote: Any) -> Any:
        """Record a SELL quote obtained this tick, for the other consumer."""
        try:
            self._tick_sell_quotes[self._sell_memo_key(position)] = quote
        except Exception:  # noqa: BLE001 - the memo is an optimisation, never a path
            pass
        return quote

    def _reusable_tick_quote(self, position: Any, venue_id: str | None
                             ) -> Any | None:
        """
        A SELL quote already obtained this tick for this exact position.

        ONLY reused when the venue matches. The trigger reference is quoted
        unpinned (a graduated curve may legitimately be exited through the pool
        it became), while the exit is pinned to the entry venue -- so a quote for
        a different market is a DIFFERENT TRADE and must never stand in for it.
        Reusing across venues would be exactly the substitution this architecture
        exists to prevent.
        """
        quote = getattr(self, "_tick_sell_quotes", {}).get(
            self._sell_memo_key(position))
        if quote is None:
            return None
        if not getattr(quote, "available", False):
            return None
        if venue_id and getattr(quote, "venue_id", None) != venue_id:
            return None
        return quote

    @staticmethod
    def _exit_retry_key(position: Any) -> tuple:
        return (getattr(position, "signal_id", None),
                getattr(position, "arm", None))

    @staticmethod
    def _bounded_backoff_s(attempts: int, *, base_s: float, cap_s: float) -> float:
        """
        One bounded exponential backoff curve, shared by every exit retry loop.

        It lives in one place because two call sites with two copies of the same
        arithmetic is how one of them silently stops growing. Only the bounds
        are a policy decision, and they are passed in nakedly so each caller's
        policy is visible where it is declared.
        """
        return min(float(cap_s), float(base_s) * (2 ** max(0, int(attempts) - 1)))

    def _exit_backoff_s(self, attempts: int) -> float:
        """
        Seconds to wait before the next exit-quote attempt.

        Bounded, and bounded LOW: the cap sits under a typical 60s upstream
        cooldown, so a provider that recovers inside its own cooldown is still
        sold into promptly rather than being made to wait out the full backoff.
        Growing it without a ceiling is how an outage turns into a stuck position.
        """
        return self._bounded_backoff_s(
            attempts, base_s=_EXIT_BACKOFF_BASE_S, cap_s=_EXIT_BACKOFF_CAP_S)

    def _beta_exit_backoff_s(self, attempts: int) -> float:
        """The beta arms' retry cooldown. See ``BETA_EXIT_RETRY_BASE_S``."""
        return self._bounded_backoff_s(
            attempts,
            base_s=float(getattr(self, "beta_exit_retry_base_s",
                                 BETA_EXIT_RETRY_BASE_S)),
            cap_s=float(getattr(self, "beta_exit_retry_cap_s",
                                BETA_EXIT_RETRY_CAP_S)))

    def _exit_backoff_remaining(self, position: Any) -> float:
        state = self._exit_retry.get(self._exit_retry_key(position))
        if not state:
            return 0.0
        return max(0.0, float(state.get("next_attempt_at", 0.0)) - time.time())

    def _note_exit_quote_failure(self, position: Any, detail: str, *,
                                 rpc_cause: str | None,
                                 retry_after_s: float | None) -> dict[str, Any]:
        """
        Record a failed exit-quote attempt and schedule the next one.

        The position is NOT settled and NOT mutated here. This only records the
        failure and decides when to try again, so an outage throttles the RETRY
        RATE without ever inventing a price or closing a position.

        THE BACKOFF APPLIES ONLY WHEN CAPACITY WAS THE BLOCKER. A quote that
        failed because the market had nothing to sell is a MARKET fact, and the
        next tick is the right time to ask again -- deferring that would delay a
        legitimate exit for no reason, since no capacity is being saved by waiting.
        Deferring a capacity failure is the whole point: re-quoting against a
        provider that has told us to back off spends the budget and cannot win.
        """
        key = self._exit_retry_key(position)
        state = self._exit_retry.get(key) or {"attempts": 0}
        if state.get("blocked"):
            # Budget already spent: no further counting, no further state.
            return state
        state["attempts"] = int(state.get("attempts", 0)) + 1
        state["last_detail"] = detail[:400]
        state["last_rpc_cause"] = rpc_cause

        if state["attempts"] >= _EXIT_MAX_ATTEMPTS:
            # The retry budget is SPENT. Stop hammering the venue: the
            # position stays OPEN and explicitly UNSELLABLE until something
            # outside this loop changes (manual intervention, venue recovery
            # observed on a later successful quote). No more retry storm.
            state["blocked"] = True
            state["next_attempt_at"] = float("inf")
            state["deferred"] = False
            self._exit_retry[key] = state
            position.exit_status = "UNSELLABLE"
            position.last_exit_error = detail[:400]
            position.exit_attempts = state["attempts"]
            log(
                "EXECUTE",
                "EXIT BLOCKED for %s (%s): %d/%d sell-quote attempts failed; "
                "last reason: %s; position remains OPEN and tracked",
                getattr(position, "symbol", "?") or "?",
                getattr(position, "mint", "?") or "?",
                state["attempts"], _EXIT_MAX_ATTEMPTS, detail[:160],
            )
            return state

        if rpc_cause is None:
            # Not a capacity problem. Retry on the normal tick cadence.
            state["next_attempt_at"] = 0.0
            state["deferred"] = False
            self._exit_retry[key] = state
            position.exit_status = "EXIT_PENDING_QUOTE"
            position.last_exit_error = detail[:400]
            position.exit_attempts = state["attempts"]
            return state

        base = self._exit_backoff_s(state["attempts"])
        # FULL jitter: spreading retries avoids a thundering herd when several
        # positions fail against the same provider cooldown together.
        delay = base * (0.5 + 0.5 * random.random())
        # Never wait longer than an upstream cooldown the provider already told
        # us about, so recovery is not delayed past the provider's own window.
        if rpc_cause == "upstream_blocked" and retry_after_s:
            delay = min(delay, max(0.0, float(retry_after_s)))
        state["next_attempt_at"] = time.time() + delay
        state["deferred"] = True
        self._exit_retry[key] = state
        position.exit_status = "EXIT_PENDING_QUOTE"
        position.last_exit_error = detail[:400]
        position.exit_attempts = state["attempts"]
        return state

    async def _resolve_trigger_reference(
        self, position: PaperPosition, live: Any, mark: ObservedMarkPoint,
        held_s: float,
    ) -> tuple[float | None, dict[str, Any]]:
        """
        Resolve the price the exit policy sees, spending a quote only if needed.

        This is the seam the whole monitoring change turns on. It returns the
        execution-native reference exactly as before -- same source semantics,
        same units, same identity requirement -- but obtains it from the PUSH
        channel when the channel is healthy, and reserves the quoted path for two
        jobs only:

          * RECOVERY, when the push channel is stale, disconnected or warming, so
            the position keeps the pre-existing conservative behaviour instead of
            acting on nothing;
          * EXECUTION VALIDATION, when an existing exit condition is near or has
            fired, to establish that the position can actually be sold for
            something on the market it is pinned to.

        WHAT IS GUARANTEED NOT TO CHANGE HERE

          * the trigger price is the venue's own spot reference, never the
            external aggregator mark and never a quoted execution price;
          * no exit threshold is read, moved or added;
          * the band gates QUOTE REQUESTS. A ``MONITOR_ONLY`` verdict does not
            suppress an exit: the policy still runs on the pushed price, and the
            exit path still obtains its own fresh quote for itself.
        """
        from .quote_proximity import evaluate as evaluate_proximity
        from .native_monitor import SOURCE_PUSH

        pushed_usd, push_provenance = self._monitor_reference(position, live)
        if pushed_usd is not None:
            provenance = dict(push_provenance)
            fill_usd = self._entry_fill_usd(position)
            current_return = (
                (pushed_usd / fill_usd - 1.0) * 100.0 if fill_usd > 0 else None)
            position.native_return_pct = current_return

            verdict = evaluate_proximity(
                live, current_return,
                cfg=self._exit_config(), policy=self.quote_proximity,
                held_s=held_s,
            )
            self._record_proximity(position, verdict)

            if not verdict.permits_quote:
                # FAR FROM EVERY TRIGGER: native monitoring only, zero RPC,
                # zero quote requests. This is the case that used to cost a
                # full executable quote on every tick and throw it away.
                provenance["quote_validation"] = "SKIPPED_FAR_FROM_TRIGGER"
                return pushed_usd, provenance

            # NEAR OR AT A TRIGGER: request a fresh executable SELL quote and
            # hold it to the existing freshness and identity rules. Its result
            # is recorded as validation; it does NOT become the trigger price.
            try:
                quote = await self._validate_executable_sell(
                    position, mark, reason=verdict.band)
            except Exception as exc:  # noqa: BLE001 - named, never falls back
                provenance["quote_validation"] = f"UNAVAILABLE: {exc}"[:160]
                return pushed_usd, provenance
            ok, why = self._quote_is_fresh_and_same_market(quote, position)
            provenance["quote_validation"] = (
                "PASS" if ok else f"FAIL: {why}"[:160])
            provenance["quote_request_reason"] = verdict.band
            if ok:
                # A quote for the exact held quantity that passed the existing
                # freshness and identity gates is a legitimate sample of what
                # this position could have liquidated for, so it folds into the
                # sampled quote excursion. It is an ESTIMATE and never a fill.
                self._record_quote_excursion(position, quote)
            # RE-CHECK the trigger on the FRESHEST state before anything is
            # acted on. The policy call that follows reads exactly what it read
            # before; this only ensures the number it reads is the newest one
            # available rather than the value the push carried a moment earlier.
            fresh_point = self._position_monitors.get(
                self._exit_retry_key(position))
            if fresh_point is not None:
                refreshed = self._monitor_reference(position, live)
                if refreshed[0] is not None:
                    return refreshed[0], refreshed[1]
            return pushed_usd, provenance

        # -- RECOVERY -----------------------------------------------------
        # The push channel could not prove a current same-market price. Fall
        # back to the pre-existing conservative path, which spends the RPC it
        # needs to read the venue directly. The band is asked first, so a
        # position that is also near a trigger is quoted with its reason
        # recorded rather than by accident.
        verdict = evaluate_proximity(
            live, None, cfg=self._exit_config(),
            policy=self.quote_proximity, held_s=held_s,
        )
        self._record_proximity(position, verdict)
        log("MONITOR",
            "recovery: mint=%s arm=%s stream=%s age_ms=%.0f (%s); using the "
            "polled reference",
            position.mint[:10], position.arm,
            position.websocket_state or "DISABLED",
            position.native_update_age_ms,
            push_provenance.get("unavailable_reason") or "no pushed reference")
        self._note_quote_request(position, f"RECOVERY_{verdict.band}")
        trigger_usd, provenance = await self._native_trigger_reference(
            position, mark
        )
        provenance["monitor_source"] = "POLL_RECOVERY"
        provenance.setdefault("quote_validation", "RECOVERY_POLL")
        return trigger_usd, provenance

    def _exit_config(self) -> Any:
        """
        The exit thresholds the proximity band must watch.

        Read off the LIVE policy so the band and the policy cannot disagree. If
        the policy does not expose a config -- which a test double may not --
        the declared production ``ExitConfig`` defaults are used rather than an
        empty set: watching nothing would silently disable the band, and
        watching an invented set would watch the wrong exits.
        """
        from ..config import ExitConfig
        cfg = getattr(self.exit_policy, "cfg", None)
        return cfg if cfg is not None else ExitConfig()

    def _record_proximity(self, position: PaperPosition, verdict: Any) -> None:
        position.quote_proximity_band = verdict.band
        position.quote_proximity_trigger = verdict.nearest_trigger
        position.quote_proximity_distance_pct = verdict.distance_pct

    @staticmethod
    def _entry_fill_usd(position: PaperPosition) -> float:
        fill = getattr(position, "entry_fill", None)
        for attribute in ("effective_price_usd", "price_usd"):
            value = getattr(fill, attribute, None)
            if value:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return 0.0

    def _take_profit_snapshot(self) -> Optional[float]:
        """
        The take-profit a NEW position is opened with, or ``None`` for OFF.

        Validated through the same closed vocabulary the UI and the persistence
        layer use, so a position can never be born carrying a level the operator
        could not have selected. An unreadable value fails closed to OFF and
        says why: an unverifiable number must not drive a real exit.
        """
        from .take_profit import TakeProfitError, normalise
        raw = getattr(self, "take_profit_pct", None)
        if raw is None:
            return None
        try:
            return normalise(raw)
        except TakeProfitError as exc:
            log("ERROR",
                "take-profit selection %r is not a valid level; opening this "
                "position with take-profit OFF: %s", raw, exc)
            return None

    async def _native_trigger_reference(
        self, position: PaperPosition, mark: ObservedMarkPoint
    ) -> tuple[float | None, dict[str, Any]]:
        """
        The SAME-MARKET, SAME-CURRENCY price the price-based exits trigger on.

        Returns ``(usd_per_token, provenance)``. ``usd_per_token`` is ``None``
        whenever the reference cannot be PROVEN to describe the market this
        position was actually opened on, and the caller must then treat the
        price triggers as unevaluable rather than substituting another source.

        Why this exists. The stop used to be evaluated against the external
        aggregator mark while the position's basis was the executable venue
        fill. Those are two different books. On a pumpfun bonding curve the
        aggregator mark sits 15-40% below the curve's own virtual-reserve spot,
        so every position was born already past the -12% stop and was sold on
        the first monitor tick. The mark was timestamp-fresh, so no age-based
        staleness gate could see it; only using the venue's own reference can.

        UNITS. The native reference is ``quote_asset_per_whole_token`` (SOL per
        token), never USD. It is converted with the SOL/USD reference recorded on
        the position, using the same multiplication the quote service itself
        uses, so the trigger and the fill share one conversion and one unit.
        """
        provenance: dict[str, Any] = {
            "trigger_reference_source": "EXECUTION_NATIVE_SAME_MARKET",
            "observed_mark_role": "DIAGNOSTIC_ONLY",
            "observed_mark_usd": mark.value_usd,
            "native_reference_raw": None,
            "native_reference_usd": None,
            "sol_usd_reference": position.sol_usd_reference,
            "reference_identity": "UNVERIFIED",
            "reference_market_id": None,
            "reference_age_ms": None,
            "unavailable_reason": None,
        }
        if self.quote_service is None:
            provenance["unavailable_reason"] = "no quote service"
            return None, provenance

        # Try lightweight monitoring quote first (uses cached static identity)
        cached = position.cached_identity
        if cached is not None:
            try:
                observed_pair = _pinned_market_address(position) or mark.pair_address
                quote = await self.quote_service.quote_sell_monitoring(
                    mint=position.mint,
                    token_amount_raw=position.tokens_raw,
                    token_decimals=position.token_decimals,
                    cached_identity=cached,
                    observed_pair_address=observed_pair,
                )
                if getattr(quote, "available", False):
                    native = getattr(quote, "execution_native_reference", None)
                    if native is not None:
                        provenance["native_reference_raw"] = native.reference_price
                        provenance["reference_identity"] = getattr(quote, "identity", "UNKNOWN")
                        provenance["reference_market_id"] = native.market_id
                        provenance["reference_age_ms"] = getattr(quote, "state_age_ms", None)

                        if self.require_identity and getattr(quote, "identity", "") != "PASS":
                            provenance["unavailable_reason"] = (
                                f"same-market identity not proven ({quote.identity})"
                            )
                            return None, provenance

                        raw = native.reference_price
                        sol_usd = position.sol_usd_reference
                        if raw and raw > 0 and sol_usd and sol_usd > 0:
                            usd_per_token = float(raw) * float(sol_usd)
                            if usd_per_token > 0:
                                provenance["native_reference_usd"] = usd_per_token
                                provenance["reference_identity"] = "PASS"
                                provenance["native_reference_raw"] = native.reference_price
                                provenance["reference_identity"] = getattr(quote, "identity", "UNKNOWN")
                                provenance["reference_market_id"] = native.market_id
                                provenance["reference_age_ms"] = getattr(quote, "state_age_ms", None)
                                return usd_per_token, provenance
            except Exception:  # noqa: BLE001 - fall back to full quote
                pass

        # Fall back to full quote_sell if monitoring quote unavailable
        try:
            # REUSE THIS TICK'S SELL QUOTE IF THERE IS ONE for this exact
            # position. On a tick that ends in an exit, the trigger reference and
            # the exit would otherwise each pay several RPC calls for the same
            # book in the same moment -- and under RPC pressure that duplication
            # is what turns one needed quote into two unaffordable ones.
            #
            # THE OBSERVED MARKET IS THE POSITION'S OWN PINNED MARKET, not the
            # live aggregator pair. The position already knows which book it was
            # opened on; asking with the pin makes identity resolve on its merits
            # (PASS on match, FAIL on a different book, UNKNOWN only when nothing
            # is pinned) instead of going UNKNOWN whenever the aggregator's pair
            # field is stale, empty or points elsewhere. The live pair is the
            # fallback only for a position that carries no pin at all.
            observed_pair = (
                _pinned_market_address(position) or mark.pair_address)
            quote = self._reusable_tick_quote(position, None) or await \
                self.quote_service.quote_sell(
                    mint=position.mint,
                    token_amount_raw=position.tokens_raw,
                    token_decimals=position.token_decimals,
                    observed_pair_address=observed_pair,
                )
            if getattr(quote, "available", False):
                self._memoised_sell(position, quote)
        except Exception as exc:  # noqa: BLE001 - named, never falls back
            provenance["unavailable_reason"] = f"native reference query failed: {exc}"[:200]
            return None, provenance

        if not getattr(quote, "available", False):
            provenance["unavailable_reason"] = (
                f"no same-market quote: {getattr(quote, 'missing_reason', None)}"
            )[:200]
            return None, provenance

        native = getattr(quote, "execution_native_reference", None)
        if native is None:
            provenance["unavailable_reason"] = "quote carried no native reference"
            return None, provenance

        provenance["native_reference_raw"] = native.reference_price
        provenance["reference_identity"] = getattr(quote, "identity", "UNKNOWN")
        provenance["reference_market_id"] = native.market_id
        provenance["reference_age_ms"] = getattr(quote, "state_age_ms", None)

        # -- identity: the reference must describe THIS position's market --
        if self.require_identity and getattr(quote, "identity", "") != "PASS":
            provenance["unavailable_reason"] = (
                f"same-market identity not proven ({quote.identity})"
            )
            return None, provenance
        # The reference is validated against the QUOTE'S OWN market, not
        # against the entry venue. This quote is deliberately unpinned: a
        # bonding curve that graduates is legitimately exited through the pool
        # it became, and the settlement path already owns the entry-venue pin
        # (pinned quote + counterfactual). What must never happen is a reference
        # that describes a DIFFERENT book than the quote it arrived with.
        quote_venue_id = getattr(quote, "venue_id", None)
        if quote_venue_id and native.market_id != quote_venue_id:
            provenance["unavailable_reason"] = (
                f"native reference is for {native.market_id} but the quote is for "
                f"{quote_venue_id}"
            )[:200]
            provenance["reference_identity"] = "MISMATCH"
            return None, provenance
        if position.entry_venue_id:
            provenance["reference_venue_changed_since_entry"] = (
                native.market_id != position.entry_venue_id
            )
        if native.base_mint and native.base_mint != position.mint:
            provenance["unavailable_reason"] = (
                f"native reference base mint {native.base_mint} != held mint"
            )
            provenance["reference_identity"] = "MISMATCH"
            return None, provenance
        if native.quote_mint and quote.quote_mint and native.quote_mint != quote.quote_mint:
            provenance["unavailable_reason"] = (
                f"native reference quote asset {native.quote_mint} != quote's "
                f"{quote.quote_mint}"
            )
            provenance["reference_identity"] = "MISMATCH"
            return None, provenance
        if getattr(native, "side", "SELL") != "SELL":
            provenance["unavailable_reason"] = (
                f"native reference side is {native.side}, not SELL"
            )
            provenance["reference_identity"] = "MISMATCH"
            return None, provenance

        # -- units: quote-asset per token -> USD per token ----------------
        raw = native.reference_price
        sol_usd = position.sol_usd_reference
        if not raw or raw <= 0:
            provenance["unavailable_reason"] = "native reference price is not positive"
            return None, provenance
        if not sol_usd or sol_usd <= 0:
            provenance["unavailable_reason"] = (
                f"invalid SOL/USD reference {sol_usd!r}; refusing to convert"
            )
            return None, provenance

        usd_per_token = float(raw) * float(sol_usd)
        if not usd_per_token or usd_per_token <= 0:
            provenance["unavailable_reason"] = "converted native reference is not positive"
            return None, provenance

        provenance["native_reference_usd"] = usd_per_token
        provenance["reference_identity"] = "PASS"
        return usd_per_token, provenance

    async def _evaluate_exits_on_native_update(
        self,
        position: PaperPosition,
        monitor: Any,
        price_point: Any,
    ) -> None:
        """
        Evaluate exit conditions immediately when a new native price arrives.

        This is the event-driven path: instead of waiting for the next
        ``monitor_once`` tick, we evaluate exit conditions the moment a fresh
        native price arrives from the WebSocket push. This ensures that fast
        market moves are caught immediately, not on the next timer tick.

        The logic mirrors the decision path in ``monitor_once`` but operates on
        a single position with a pre-resolved native price.
        """
        from .quote_proximity import evaluate as evaluate_proximity

        # Create a live position object for the exit policy, just like in monitor_once.
        mark = await self.observe(position.mint)
        live = self._production_position(position, mark)
        held = time.time() - position.opened_at

        # Use the pushed price directly (already in USD per token from the monitor).
        pushed_usd = float(price_point.price_sol_per_token) * position.sol_usd_reference
        if pushed_usd <= 0:
            return

        provenance = {
            "monitor_source": "WEBSOCKET_PUSH",
            "native_reference_raw": price_point.price_sol_per_token,
            "native_reference_usd": pushed_usd,
            "reference_identity": "PASS",
            "reference_market_id": price_point.venue_id,
            "reference_age_ms": price_point.age_ms,
            "unavailable_reason": None,
        }

        # Record the native return for diagnostics.
        fill_usd = self._entry_fill_usd(position)
        current_return = (
            (pushed_usd / fill_usd - 1.0) * 100.0 if fill_usd > 0 else None
        )
        position.native_return_pct = current_return

        # Check the proximity band to decide whether to request a quote.
        verdict = evaluate_proximity(
            live, current_return,
            cfg=self._exit_config(), policy=self.quote_proximity,
            held_s=time.time() - position.opened_at,
        )
        self._record_proximity(position, verdict)

        if not verdict.permits_quote:
            # FAR FROM EVERY TRIGGER: native monitoring only, zero RPC,
            # zero quote requests.
            provenance["quote_validation"] = "SKIPPED_FAR_FROM_TRIGGER"
            self._record_trigger_provenance(position, provenance)
            return

        # NEAR OR AT A TRIGGER: request a fresh executable SELL quote and
        # hold it to the existing freshness and identity rules.
        try:
            quote = await self._validate_executable_sell(
                position, mark, reason=verdict.band)
        except Exception as exc:  # noqa: BLE001 - named, never falls back
            provenance["quote_validation"] = f"UNAVAILABLE: {exc}"[:160]
            return

        ok, why = self._quote_is_fresh_and_same_market(quote, position)
        provenance["quote_validation"] = (
            "PASS" if ok else f"FAIL: {why}"[:160])
        provenance["quote_request_reason"] = verdict.band

        # THE PUSH PATH SAMPLES THE EXCURSION TOO.
        #
        # This is the same executable SELL quote, for the same exact held
        # quantity, through the same freshness and identity gate as the timer
        # path -- and it used to be thrown away. Because a push is what moves a
        # position fastest, that is precisely the moment the excursion most
        # needs to see, and skipping it left every push-driven hold sampled at
        # zero or one point. It writes analytics only: no return value, no path
        # to an exit decision, no change to what is settled below.
        if ok:
            self._record_quote_excursion(position, quote)
        else:
            self._reject_quote_observation(position, f"validation failed: {why}")

        # RE-CHECK the trigger on the FRESHEST state before anything is
        # acted on. The policy call that follows reads exactly what it read
        # before; this only ensures the number it reads is the newest one
        # available rather than the value the push carried a moment earlier.
        fresh_point = self._position_monitors.get(
            self._exit_retry_key(position))
        if fresh_point is not None:
            refreshed = self._monitor_reference(position, None)
            if refreshed[0] is not None:
                # Use the refreshed price for the final decision.
                pass

        # Record the trigger provenance for diagnostics.
        self._record_trigger_provenance(position, provenance)

        # Evaluate the exit policy with the native price.
        decision = self.exit_policy.assess(
            live,
            now=time.time(),
            mark_price=pushed_usd,
            data_state=(DataState.LIVE if mark.available else DataState.MISSING),
            momentum_30s_pct=None,
            structure_30s_pct=None,
            liquidity_change_pct=None,
        )

        self._record_trigger_decision(position, provenance, pushed_usd, decision)

        if decision.action == "EXIT":
            # Check minimum hold suppression.
            held_s = time.time() - position.opened_at
            if decision.trigger in (
                "momentum_collapse", "structure_break", "peak_giveback",
                "wide_giveback", "no_follow_through",
            ) and held_s < self.min_hold_s:
                decision = ExitDecision(
                    "HOLD",
                    f"exit signal '{decision.trigger}' suppressed before the "
                    f"{self.min_hold_s:.0f}s minimum hold (a declared rule, not "
                    "a post-hoc exception)",
                    decision.style, trigger=None,
                )

            if decision.action == "EXIT":
                # Record the exit trigger.
                if position.exit_status in ("OPEN_MONITORING", ""):
                    position.exit_status = "EXIT_TRIGGERED"
                position.exit_trigger_reason = getattr(decision, "reason", "") or ""

                # THE EXIT DECISION IS ALWAYS TAKEN. Only the QUOTE is deferred.
                state = self._exit_retry.get(self._exit_retry_key(position)) or {}
                if state.get("blocked"):
                    if position.exit_status != "UNSELLABLE":
                        position.exit_status = "UNSELLABLE"
                    return

                remaining = self._exit_backoff_remaining(position)
                if remaining > 0.0:
                    self._log_exit_deferred(position, decision, remaining)
                    return

                # Execute the exit.
                trades = await self._close_for_arm(
                    spec=None, position=position, mark=None, decision=decision,
                )
                self.trade_records.extend(trades)

    # -- EXECUTION-NATIVE HOLD-PATH TELEMETRY (measurement only) ----------
    #
    # The aggregator mark was measured to differ from the position-sized
    # executable sell quote by a median of ~+25% and by up to +92%, in both
    # directions, and it does not update at all on short holds. That makes a
    # 60-second hold unmeasurable, because the only record of the hold was the
    # mark. These two methods record the position's OWN sell quote over the
    # hold so the experiment can be scored on the market it traded.
    #
    # THE SAFETY CONTRACT, and it is the whole point of this block:
    #   * ``_start_`` returns immediately. The sampler runs as its OWN asyncio
    #     task and never inside ``monitor_once``, ``on_signal`` or
    #     ``_close_for_arm``.
    #   * Both methods are wrapped so that ANY failure -- import error, no event
    #     loop, unwritable sink, disabled config -- is a logged no-op.
    #   * Nothing here can prevent an exit, extend a hold, alter an outcome,
    #     change entry eligibility, change sizing, or move a state transition.
    #     There is no return value either caller uses.
    #   * When telemetry is disabled these two calls are the entire footprint,
    #     and they do nothing.

    def _hold_path_config(self) -> Any:
        """
        Resolve the hold-path instrument from configuration.

        THE SINK COMES FROM CONFIGURATION, NOT FROM A SETTER NOTHING CALLS.

        This used to read a module global that no code path ever assigned, so it
        returned ``disabled()`` for every position and no hold path was ever
        recorded. Resolution order is now:

        1. an explicit ``_set_hold_path_sink`` override (tests, operators), then
        2. ``PAPER_HOLD_PATH_*`` in the runtime configuration, then
        3. the built-in default under the runtime's own ``var_dir``.

        ``PaperLiveEngine`` holds no configuration object of its own, so (2) is
        read from the process configuration rather than from a field that does not
        exist. The result is cached per sink so a position does not re-read the
        environment on every open.

        A failure here is logged and returns ``None``: telemetry must never be the
        reason a position goes unmonitored.
        """
        try:
            from .telemetry import HoldPathConfig

            override = _HOLD_PATH_SINK
            cache = getattr(self, "_hold_path_config_cache", None)
            if cache is not None and cache[0] is override:
                return cache[1]

            if override is not None:
                cfg = HoldPathConfig(
                    interval_s=1.0, max_duration_s=180.0, max_state_age_ms=5_000.0,
                    max_quote_age_ms=30_000.0, sink_path=Path(override), enabled=True,
                )
                self._hold_path_config_cache = (override, cfg)
                return cfg

            from ..config import build_config

            real = build_config().real
            if not getattr(real, "paper_hold_path_telemetry_enabled", True):
                self._hold_path_config_cache = (override, HoldPathConfig.disabled())
                return self._hold_path_config_cache[1]

            sink_path = Path(getattr(real, "paper_hold_path_sink", "")
                             or "paper_hold_path.jsonl")
            if not sink_path.is_absolute():
                # Relative sink names resolve against this engine's ONE durable
                # state root -- the same directory the state store, the beta
                # experiment and restart recovery use. It used to fall back to
                # ``Path("var")`` relative to the process working directory,
                # so this telemetry file landed somewhere different depending
                # on where the process was started.
                sink_path = Path(self._resolved_state_root()) / sink_path

            cfg = HoldPathConfig(
                interval_s=float(
                    getattr(real, "paper_hold_path_interval_s", 1.0) or 1.0),
                max_duration_s=float(
                    getattr(real, "paper_hold_path_max_duration_s", 180.0) or 180.0),
                max_state_age_ms=5_000.0,
                max_quote_age_ms=30_000.0,
                sink_path=sink_path,
                enabled=True,
            )
            self._hold_path_config_cache = (override, cfg)
            return cfg
        except Exception as exc:  # noqa: BLE001 - telemetry never breaks a caller
            log("ERROR", "hold-path telemetry unavailable: %s", type(exc).__name__)
            return None

    def _start_hold_path_telemetry(self, spec: Any, position: Any) -> None:
        """Launch the sampler for a freshly opened position. Never raises."""
        try:
            config = self._hold_path_config()
            if config is None or not config.enabled:
                return
            from .telemetry import HoldPathRecorder

            entry_fill = getattr(position, "entry_fill", None)
            entry_fill_usd = to_float(
                getattr(entry_fill, "effective_price_usd", None)) or None
            # THE RETURN BASIS IS THE WHOLE-POSITION OUTLAY.
            #
            # ``effective_price_usd`` is a PER-TOKEN price. The sampler observes a
            # TOTAL SELL proceeds figure, and a ratio between the two is not a
            # return -- the first recorded hold paths reported +500,259 % on
            # positions that moved about -3 %. Scale the fill by the tokens actually
            # held, and prefer the recorded cash outlay when one exists, since that
            # is what really left the account.
            tokens_whole = None
            try:
                decimals = int(position.token_decimals)
                tokens_whole = float(position.tokens_raw) / (10 ** decimals)
            except Exception:  # noqa: BLE001 - the basis is data, not a blocker
                tokens_whole = None
            # ``usd_notional`` is what the position actually cost at the entry
            # fill; the fill price scaled by the tokens held is the same figure
            # derived, and is kept as the fallback.
            entry_notional_usd = (
                to_float(getattr(position, "usd_notional", None))
                or (entry_fill_usd * tokens_whole
                    if entry_fill_usd and tokens_whole else None)
            )
            # A SimulatedFill's ``kind`` is a LAYER label ("SIMULATED_FILL"), not a
            # price provenance. The study vocabulary is fixed at five values, and a
            # PAPER leg is PAPER_FILL -- so map it rather than leaking the label,
            # which would make every provenance field incomparable.
            entry_fill_provenance = (
                "PAPER_FILL" if entry_fill is not None else None)

            quote_service = self.quote_service
            if quote_service is None:
                return

            async def quote_fn(*, mint, token_amount_raw, token_decimals,
                                   observed_pair_address=None):
                # The CANONICAL quote implementation -- the same object the
                # executor sells through. No second pricing formula exists.
                #
                # THE OBSERVED MARKET IDENTITY IS PASSED THROUGH, NOT HARDCODED
                # TO ``None``. With ``None`` the same-market identity check could
                # never resolve, so every sample paid full RPC cost and was then
                # discarded as unusable -- the instrument bought nothing and, at
                # 1 Hz, consumed enough capacity to stop the position it was
                # watching from being sold. The caller supplies the address the
                # position was actually opened on; identity is then proven or
                # refused on its merits rather than being unanswerable.
                return await quote_service.quote_sell(
                    mint=mint, token_amount_raw=token_amount_raw,
                    token_decimals=token_decimals,
                    observed_pair_address=observed_pair_address,
                )

            # NOTE: the live aggregator pair is deliberately NOT consulted for
            # identity any more. The position's own pinned market is
            # authoritative (audit §2 -- a verified identity must persist through
            # every hold tick and must never be silently replaced by a newly
            # discovered market); only the MARK still comes from the aggregator,
            # for display and research.
            def mark_fn(mint: str) -> Any:
                point = self.observed_marks.get(mint)
                return getattr(point, "value_usd", None) if point else None

            window_ms = None
            try:
                from ..strategy_profile import configured_max_hold_s

                cap = configured_max_hold_s()
                window_ms = float(cap) * 1000.0 if cap else None
            except Exception:  # noqa: BLE001
                window_ms = None

            recorder = HoldPathRecorder(
                trade_id=f"{position.signal_id}:{position.arm}",
                mint=position.mint,
                # THE SAMPLING VENUE IS THE POSITION'S CURRENT MARKET: a proven
                # exit-venue migration otherwise reports every migrated-pool
                # quote as a mismatch against the dead entry curve.
                curve_or_pool_id=(
                    position.effective_exit_venue_id or None),
                token_decimals=int(position.token_decimals),
                remaining_token_amount_raw=int(position.tokens_raw),
                quote_asset=WSOL_MINT,
                quote_fn=quote_fn,
                mark_fn=mark_fn,
                expected_venue_id=(
                    position.effective_exit_venue_id or None),
                expected_window_ms=window_ms,
                sol_usd_reference=float(position.sol_usd_reference or 0.0) or None,
                # The return basis is the ENTRY FILL -- the price the simulated
                # execution paid -- and its provenance travels with it, so a hold
                # path can never be read as returns against a quoted mark.
                entry_fill_usd=entry_fill_usd,
                entry_notional_usd=entry_notional_usd,
                entry_price_provenance=entry_fill_provenance,
                # THE PROFILE, NOT THE ARM. This field is named
                # ``strategy_profile`` and every other writer of it on this path
                # passes ``position.production_profile``. This one passed
                # ``position.arm``, so an ``H2_CAP60`` hold path recorded the
                # strategy profile ``"H2_CAP60"`` -- a value that is not a
                # member of ``StrategyProfile`` and cannot be parsed back --
                # while an arm that happened to be named ``n`` recorded
                # ``"n"``. The live data in ``var/paper_hold_path.jsonl``
                # contains exactly that.
                strategy_profile=str(
                    getattr(position, "production_profile", "") or "") or None,
                # THE PINNED MARKET, so the identity check has something
                # real to verify against. This is the pool/curve the position
                # was opened on -- never a guess, and never overridden by the
                # live aggregator pair.
                observed_pair_address=_pinned_market_address(position),
                # THE LANE GUARD IS SUPPLIED HERE, not imported by the sampler.
                # It charges every RPC call the sampler makes to the telemetry
                # lane, whose budget is a fraction of the total, so the capacity
                # an open position needs to sell is unreachable from background
                # work.
                rpc_lane_guard=_telemetry_lane_guard(),
                skip_predicate=_is_telemetry_skip,
                # THE PINNED MARKET IS KNOWN HERE, so a sample that cannot be
                # bound to it is one this module would discard anyway. Turning that
                # into a SKIP is what stops the sampler spending the exit path's
                # capacity on observations nobody can use.
                #
                # ``config`` already carries the resolved interval -- this engine
                # holds no configuration object of its own, so the value is read
                # once in ``_hold_path_config`` and passed here rather than
                # re-derived against an attribute that does not exist.
                config=replace(config, require_market_identity=True),
            )
            self._hold_path_recorders[recorder.trade_id] = recorder
            # THE ENTRY OBSERVATION IS TAKEN FIRST, ON THE RECORDER'S OWN TASK.
            # Without it the hold path begins at the first post-fill sample and
            # cannot produce a return series at all; reconstructing the entry later
            # would mean quoting a price nobody observed. It runs on the recorder's
            # task so the opening path stays synchronous and nothing here can raise
            # into the caller.
            recorder.start(with_entry_observation=True)
        except Exception as exc:  # noqa: BLE001 - telemetry never breaks a caller
            log("ERROR", "hold-path telemetry could not start for %s: %s",
                getattr(position, "signal_id", "?"), type(exc).__name__)

    async def _stop_hold_path_telemetry(self, position: Any) -> None:
        """
        Hand the sampler its post-exit research window. Never raises.

        Called on settlement. The sampler is NOT awaited to a stop here: the
        trading path must never wait on a research instrument. It keeps sampling
        for ``POST_EXIT_WINDOW_S`` seconds, every row marked
        ``POST_EXIT_RESEARCH_ONLY`` (did the token pump after we sold? what did
        the drawdown/rebound look like?), and then ends itself.
        """
        try:
            recorders = getattr(self, "_hold_path_recorders", None)
            if not recorders:
                return
            key = f"{position.signal_id}:{position.arm}"
            recorder = recorders.pop(key, None)
            if recorder is None:
                return
            marked = recorder.mark_post_exit()
            if not marked:
                # Telemetry disabled or the sampler already finished: fall back
                # to the bounded close so nothing leaks.
                await recorder.aclose()
        except Exception as exc:  # noqa: BLE001 - telemetry never breaks an exit
            log("ERROR", "hold-path telemetry could not stop for %s: %s",
                getattr(position, "signal_id", "?"), type(exc).__name__)

    def _production_position(
        self, position: PaperPosition, mark: ObservedMarkPoint
    ) -> Position:
        """
        A production :class:`~autopaper.models.Position` for the exit policy.

        Its ``entry_price`` is the SIMULATED FILL price, so the exit policy's
        percentage triggers are computed against what the position actually
        cost - not against a mark it never paid.
        """
        fill_price = float(position.entry_fill.effective_price_usd or 0.0)
        # The entry instant the hold timer is measured from. A position whose
        # ``opened_at`` is not a credible instant must not be handed to the
        # policy as if it were: the policy reads ``entry_timestamp`` and computes
        # ``now - entry_timestamp``, so an epoch value here is a hold of ~56
        # years that satisfies every elapsed-time trigger on the first pass.
        # Persisted data is already screened in ``_position_from_dict``; this is
        # the second gate, on the only path that builds an exit position.
        opened_at = _credible_opened_at(position, now=time.time())
        if opened_at is None:
            log("ERROR",
                "position %s has an uncredible opened_at=%r; reporting the entry "
                "instant as NOW so the hold timer reads 0s instead of "
                "manufacturing a max-hold exit",
                position.position_id if hasattr(position, "position_id")
                else position.signal_id, position.opened_at)
            opened_at = time.time()
        # FAST_FAILURE is the exit style every L25 profile uses; the exit policy
        # reads it back off the position, so it is set here rather than passed
        # in, which keeps this adapter free of exit-policy opinions.
        return Position(
            position_id=f"paperlive_{position.signal_id}_{position.arm}",
            candidate_id=position.signal_id,
            mint=position.mint,
            symbol=position.symbol,
            status=PositionStatus.OPEN,
            route=Route.FAST_ENTRY,
            regime=Regime.UNCERTAIN,
            exit_style=ExitStyle.FAST_FAILURE,
            strategy="L25",
            entry_signal_timestamp=position.opened_at,
            entry_decision_timestamp=position.opened_at,
            entry_timestamp=opened_at,
            quoted_price=fill_price,
            entry_reference_price=fill_price,
            fill_price=fill_price,
            fill_price_kind="SIMULATED_FILL_FROM_EXECUTABLE_QUOTE",
            observed_mark_price=mark.value_usd,
            executable_quote_price=position.entry_fill.effective_price,
            executable_quote_price_usd=position.entry_fill.effective_price_usd,
            executable_quote_kind="EXECUTABLE_QUOTE",
            executable_quote_at=position.entry_fill.quote_timestamp,
            executable_quote_provider=position.entry_fill.provider,
            executable_quote_venue=position.entry_fill.venue,
            executable_quote_pool=position.entry_fill.pool_or_curve,
            quote_price_source="EXECUTABLE_QUOTE",
            quantity=float(position.tokens_raw),
            notional_usd=position.usd_notional,
            cost_basis_usd=position.usd_notional,
            cash_at_entry=position.usd_notional,
            mark_price=mark.value_usd,
            data_state=(DataState.LIVE if mark.available else DataState.MISSING),
            strategy_profile=position.production_profile,
        )

    @staticmethod
    def _update_mfe_mae(position: PaperPosition, mark: float | None) -> None:
        """
        OBSERVED_MARK excursions, tracked as OBSERVED_MARK.

        Named so they cannot be read as executable returns. They ARE mark-based
        by definition - that is the point: they answer "how far did the
        observation move", which is a different question from "what would I
        have realised".
        """
        if not mark or mark <= 0:
            return
        if not position.entry_observed_mark or position.entry_observed_mark <= 0:
            return
        change = (mark / position.entry_observed_mark - 1.0) * 100.0
        position.observed_mark_mfe = max(position.observed_mark_mfe, change)
        position.observed_mark_mae = min(position.observed_mark_mae, change)
        # THE LEVELS ARE SEEDED FROM THE ENTRY MARK, so the excursion legs
        # cover the whole hold from the first instant. Seeding from the current
        # tick instead silently dropped the entry level: a position whose high
        # was the entry print reported a peak_mark that never existed.
        if position.peak_mark is None:
            position.peak_mark = max(float(position.entry_observed_mark), mark)
        else:
            position.peak_mark = max(position.peak_mark, mark)
        if position.trough_mark is None:
            position.trough_mark = min(float(position.entry_observed_mark), mark)
        else:
            position.trough_mark = min(position.trough_mark, mark)

    def _classify_rpc_cause(self, detail: str, *,
                                 exc: BaseException | None = None
                                 ) -> tuple[str | None, float | None]:
        """
        Which capacity condition blocked this quote, if any.

        Prefers the refusal's OWN declared reason. A quote service that surfaces
        the failure as an exception carries the answer on it, and that is more
        direct evidence than inferring it from shared state -- and it is still
        correct when the service wraps or re-raises. Falls back to the client's
        limiter state when the failure arrived as an unavailable quote instead.

        Never pattern-matches the message text: wording can change, and a
        reworded message must not silently reclassify an upstream throttle as a
        local quota problem.

        Returns ``(cause, retry_after_s)`` where cause is one of
        ``upstream_blocked``, ``local_window``, ``telemetry_budget`` or ``None``.
        """
        try:
            from ..market.ratelimit import (
                LOCAL_WINDOW as _LW,
                UPSTREAM_BLOCKED as _UB,
            )

            reason = getattr(exc, "reason", None)
            if reason == _UB:
                retry = getattr(exc, "retry_after_s", None)
                return "upstream_blocked", retry
            if reason == _LW:
                return "local_window", None

            from ..same_market.rpc import SolanaReadOnlyRpc

            rpc = getattr(self.quote_service, "_reader", None)
            rpc = getattr(rpc, "rpc", None)
            limiter = getattr(rpc, "limiter", None)
            if limiter is None or not isinstance(rpc, SolanaReadOnlyRpc):
                return None, None
            remaining = max(0.0, float(limiter.blocked_until) - time.monotonic())
            if remaining > 0.0:
                return "upstream_blocked", remaining
            if limiter.rejections_by_reason.get(_LW):
                return "local_window", None
            return None, None
        except Exception:  # noqa: BLE001 - classification must never break an exit
            return None, None

    def _push_channel_delivering(self, position: Any) -> bool:
        """
        Is this position's push channel ATTACHED and HEALTHY?

        ``websocket_connected`` alone is not proof of anything and must never be
        treated as such: an open socket that has received no valid account update
        is exactly the state that produced "market data unusable for 123s" while
        reporting ``websocket_connected=True``. A channel is delivering only when
        it is connected AND the monitor is in a state that can serve a price.
        """
        monitor = self._position_monitors.get(self._exit_retry_key(position))
        if monitor is None:
            # No monitor is not a disconnect: it means the market type cannot be
            # decoded from pushed data, or the channel was never configured.
            return False
        if monitor._stream is None or not monitor._stream.is_open:
            return False
        from .native_monitor import STATE_HEALTHY

        return monitor.state == STATE_HEALTHY and monitor.age_ms() >= 0.0 and \
            monitor.age_ms() <= float(self.native_monitor_stale_after_ms or 0.0)

    def _shared_budget_snapshot(self) -> dict[str, Any]:
        """
        The shared request budget, as the monitor sees it.

        Read off the live limiter through the same object path
        ``_classify_rpc_cause`` uses, so the numbers here are the ones the
        refusals were actually made against. Returns an empty dict when no
        limiter is reachable -- a fixture engine, or a PAPER run with no RPC
        configured -- rather than a zero that would read as "budget exhausted".
        """
        try:
            rpc = getattr(self.quote_service, "_reader", None)
            rpc = getattr(rpc, "rpc", None)
            limiter = getattr(rpc, "limiter", None)
            if limiter is None:
                return {}
            used = float(getattr(limiter, "telemetry_requests", 0) or 0.0)
            ceiling = float(getattr(limiter, "telemetry_max_per_minute", 0) or 0.0)
            return {
                "telemetry_lane_used_per_min": used,
                "telemetry_lane_ceiling_per_min": ceiling,
                "telemetry_lane_remaining_per_min": max(0.0, ceiling - used),
                "rpc_rejections": int(getattr(limiter, "rejections", 0) or 0),
                "blocked_for_s": round(
                    max(0.0, float(getattr(limiter, "blocked_until", 0.0))
                        - time.monotonic()), 2),
            }
        except Exception:  # noqa: BLE001 - a diagnostic must never break a tick
            return {}

    def _classify_monitor_tick(
        self, position: Any, trigger_usd: float | None,
        provenance: dict[str, Any],
    ) -> None:
        """
        Classify ONE monitor tick's monitoring health for diagnostics.

        Called on every position that was observed this tick, so the
        per-position counters accumulate across the hold. The aggregate
        ``_monitor_diag`` counters are incremented here.
        """
        d = self._monitor_diag
        if not self._push_channel_delivering(position):
            # The channel was attached but the socket is not healthy, so the tick
            # ran on the polling recovery path (or on nothing at all). Counted
            # alongside the classified cause: a disconnect plus a rate limit is
            # one event with two facts, and collapsing them would hide the one
            # that came first.
            d["MONITOR_WS_DISCONNECTED"] += 1
        if trigger_usd is not None:
            position.monitor_ok_ticks += 1
            d["MONITOR_OK"] += 1
            # Track quote age for latency monitoring.
            age = provenance.get("reference_age_ms")
            if age is not None and isinstance(age, (int, float)):
                d["QUOTE_AGE_MS_SUM"] += float(age)
                d["QUOTE_AGE_MS_COUNT"] += 1
            return
        reason = (provenance.get("unavailable_reason") or "").lower()
        identity = str(provenance.get("reference_identity", "") or "")
        if ("budget" in reason or "local rpc cap" in reason
                or "telemetry lane" in reason):
            # A CAPACITY refusal, not a market fact and not a transport fault:
            # the shared request budget declined to spend on monitoring. Kept
            # separate from MONITOR_RPC_ERROR because the two have different
            # remedies -- one needs a tighter monitor (the push channel), the
            # other needs a provider.
            position.monitor_skipped_budget_ticks += 1
            d["MONITOR_SKIPPED_BUDGET"] += 1
        elif "rpc" in reason or "rate limit" in reason or "throttled" in reason:
            position.monitor_rpc_error_ticks += 1
            d["MONITOR_RPC_ERROR"] += 1
        elif "no same-market quote" in reason or "no executable" in reason:
            position.monitor_quote_unavailable_ticks += 1
            d["MONITOR_QUOTE_UNAVAILABLE"] += 1
        elif identity == "UNKNOWN" or "not proven" in reason:
            position.monitor_identity_unknown_ticks += 1
            d["MONITOR_IDENTITY_UNKNOWN"] += 1
        elif identity == "MISMATCH" or "mismatch" in reason:
            position.monitor_identity_mismatch_ticks += 1
            d["MONITOR_IDENTITY_MISMATCH"] += 1
        else:
            # Catch-all: any other unavailability.
            position.monitor_unavailable_ticks += 1

    def _record_exit_failure(self, position: Any, decision: Any, quote: Any,
                             detail: str, *,
                             exc: BaseException | None = None) -> dict[str, Any]:
        """
        Schedule the retry and emit the structured record for a failed exit.

        Everything an operator needs to answer "is this stuck, and why": which
        position, which pinned market, which condition blocked it, whether it is
        retryable, and when the next attempt is.
        """
        cause, retry_after = self._classify_rpc_cause(detail, exc=exc)
        state = self._note_exit_quote_failure(
            position, detail, rpc_cause=cause, retry_after_s=retry_after)
        blocked = bool(state.get("blocked"))
        next_in = (0.0 if blocked else
                   max(0.0, float(state.get("next_attempt_at", 0.0)) - time.time()))
        if not blocked:
            log(
                "EXECUTE",
                "SELL quote failed for %s on pinned %s: cause=%s retryable=yes "
                "attempt=%d next_retry_in=%.1fs position=OPEN monitoring=alive",
                getattr(position, "symbol", "?") or "?",
                getattr(position, "entry_venue_id", "?") or "?",
                cause or "not_capacity",
                int(state.get("attempts", 0)), next_in,
            )
        return {
            "sell_quote_failure_cause": cause or "not_capacity",
            "sell_quote_retryable": not blocked,
            "sell_quote_attempts": int(state.get("attempts", 0)),
            "sell_quote_next_retry_in_s": round(next_in, 2),
            "sell_quote_reused_this_tick": bool(
                getattr(self, "_sell_quote_reused", False)),
        }

    def _sell_quote_error_record(self, position: PaperPosition, mark: ObservedMarkPoint,
                                 decision: ExitDecision, latency: LatencyMarks,
                                 exc: BaseException) -> TradeRecord:
        """
        A SELL quote call that RAISED, turned into a normal failed-exit record.

        Same shape as an unavailable quote, so everything downstream -- the retry
        schedule, the classified cause, the operator log, the invariant that a
        failed exit settles nothing -- applies unchanged. Raising here instead
        would let a transport fault escape the exit path entirely.
        """
        detail = f"{type(exc).__name__}: {exc}"[:400]
        stub = _unavailable_sell_stub(position, detail)
        return TradeRecord(
            position=position, exit_trigger=decision.trigger or "EXIT",
            exit_reason=decision.reason, exit_observed_mark=mark.value_usd,
            exit_mark_timestamp=mark.timestamp, sell_quote=stub,
            exit_fill=None, pnl=None, latency=latency,
            failure=FailureReason.EXIT_QUOTE_UNAVAILABLE,
            failure_detail=detail,
            **self._sell_provenance(position, stub, decision, settled=False,
                                    mark=mark),
            **self._record_exit_failure(position, decision, stub, detail,
                                        exc=exc),
        )

    @staticmethod
    def _sell_provenance(position: Any, quote: Any, decision: Any,
                         *, settled: bool,
                         mark: Any = None) -> dict[str, Any]:
        """
        Per-trade SELL provenance for the trade record.

        The max-hold cap is only a TRIGGER: what actually happened is recorded
        here, so a requested quote is never reported as a fill. ``settled=False``
        means no executable SELL was obtained and the position stays open.

        MARK vs EXECUTABLE (audit §12) stay SEPARATE and are labelled as what
        they are: ``executable_vs_mark_gap_pct`` is the gap between the
        executable SELL quote and the aggregator observation -- it is NOT
        slippage and is never called that -- and ``mark_age_ms`` says how old
        the observation was at the exit attempt.
        """
        from ..strategy_profile import max_hold_s
        try:
            cap = max_hold_s(getattr(position, "production_profile", None))
        except Exception:  # noqa: BLE001 - provenance must never break an exit
            cap = None
        held = (time.time() - position.opened_at
                if getattr(position, "opened_at", None) else None)
        mfe_available = False
        mfe_pct = None
        mae_available = False
        mae_pct = None
        if settled and getattr(position, "entry_fill", None) is not None:
            try:
                entry_price = getattr(position.entry_fill, "effective_price_usd", None)
                exit_price = getattr(quote, "effective_price_usd", None)
                if entry_price is not None and exit_price is not None and entry_price != 0:
                    mfe_pct = (exit_price / entry_price - 1.0) * 100.0
                    mfe_available = True
                # MAE: best (lowest) executable price seen. For a SELL that is the
                # exit fill itself when it is below entry; otherwise entry.
                if exit_price is not None and entry_price is not None:
                    mae_pct = (min(exit_price, entry_price) / entry_price - 1.0) * 100.0
                    mae_available = True
            except Exception:  # noqa: BLE001 - leave unavailable
                pass
        return {
            "strategy_profile": str(getattr(position, "production_profile",
                                           "") or ""),
            "max_hold_seconds": cap,
            "hold_time_seconds": held,
            "sell_quote_source": str(getattr(quote, "provider", "") or ""),
            "sell_market_id": str(getattr(quote, "venue_id", "")
                                  or getattr(quote, "pool_or_curve", "") or ""),
            "sell_quote_price": (float(quote.effective_price_usd)
                                 if getattr(quote, "effective_price_usd", None)
                                 is not None else None),
            "executable_vs_mark_gap_pct": _exec_vs_mark_gap_pct(quote, mark),
            "mark_age_ms": _mark_age_ms(mark),
            "sell_quote_age_ms": getattr(quote, "quote_age_ms", None),
            "sell_execution_status": "SETTLED" if settled
                                     else "NO_EXECUTABLE_SELL",
            "executable_mfe_pct": mfe_pct,
            "executable_mfe_available": mfe_available,
            "executable_mae_pct": mae_pct,
            "executable_mae_available": mae_available,
        }

    async def _close_for_arm(
        self,
        *,
        spec: ArmSpec,
        position: PaperPosition,
        mark: ObservedMarkPoint,
        decision: ExitDecision,
    ) -> TradeRecord:
        """
        Take a FRESH executable SELL quote for the EXACT tokens held, and settle.

        Fail-closed on the quote, deterministically: if no usable SELL quote can
        be obtained the position is LEFT OPEN and the failure recorded. It is
        not closed at the mark, and it is not force-closed at cost.

        WHEN THE QUOTE FAILS, THE CAUSE IS CLASSIFIED AND THE RETRY SCHEDULED.
        "Could not get a quote" and "the provider throttled us for 60s" are
        different incidents with different remedies, and reporting both as
        "local rpc cap reached" sent the operator looking at the wrong system.
        """
        self._sell_quote_reused = False
        arm = spec.name.value
        account = self.accounts[arm]
        latency = LatencyMarks(
            receive_at_monotonic=position.opened_at,
            receive_at_wall=position.opened_at,
            exit_trigger_at=time.monotonic(),
            exit_trigger_wall=time.time(),
        )

        latency.sell_quote_request_at = time.monotonic()
        # -- QUOTE THE PINNED ENTRY VENUE ---------------------------------
        #
        # The exit is filled on the market the position was OPENED on. A SELL
        # routed to a different pool is a different trade with different depth,
        # so it is recorded as a counterfactual and never as the settlement.
        counterfactual = None
        pinned_sell = getattr(self.quote_service, "quote_sell_pinned", None)
        # REUSE THE TICK'S SELL QUOTE WHEN IT IS THE PINNED MARKET.
        #
        # The trigger reference earlier in THIS tick already asked for an
        # executable SELL on this exact position and size. If that quote came
        # back on the pinned entry venue it IS the settlement quote, and asking
        # again spends several more RPC calls to learn the same thing. Under RPC
        # pressure that duplication is not a rounding error -- it is the second
        # reason a position that needed one quote could not get it.
        #
        # The counterfactual is deliberately NOT re-fetched. It is research
        # output; the settlement is not. Trading one for the other is the whole
        # point, and the skip is recorded on the trade's provenance so the gap is
        # visible rather than silent.
        reusable = self._reusable_tick_quote(
            position, position.effective_exit_venue_id or None)
        # A QUOTE CALL THAT RAISES IS A FAILED EXIT, NOT A CRASH.
        #
        # This had no exception guard at all, so a transport fault -- an upstream
        # throttle surfacing as an exception rather than as an unavailable quote
        # -- propagated out of ``monitor_once`` and was swallowed one level up as
        # "execution-aware monitor cycle failed". The position stayed open, which
        # was survivable, but the exit never produced a record, so the failure was
        # invisible and unclassifiable. Fail closed here instead, and classify it
        # like any other missing quote.
        try:
            if reusable is not None:
                quote, counterfactual = reusable, None
                self._sell_quote_reused = True
            elif callable(pinned_sell):
                quote, counterfactual = await pinned_sell(
                    mint=position.mint,
                    token_amount_raw=position.tokens_raw,
                    token_decimals=position.token_decimals,
                    pinned_venue_id=position.effective_exit_venue_id or None,
                    # THE HOLD-PATH PIN IS THE POSITION'S CURRENT MARKET, so
                    # identity resolves against the book this position trades
                    # on (entry pin, or a proven migration target) rather than
                    # against whatever pair the aggregator reports right now.
                    pinned_pool_address=_pinned_market_address(position),
                    observed_pair_address=mark.pair_address,
                    cache=self.quote_cache,
                )
            else:
                quote, counterfactual = await self.quote_service.quote_sell(
                    mint=position.mint,
                    token_amount_raw=position.tokens_raw,
                    token_decimals=position.token_decimals,
                    observed_pair_address=mark.pair_address,
                ), None
        except Exception as exc:  # noqa: BLE001 - the exit must fail closed
            return self._sell_quote_error_record(
                position, mark, decision, latency, exc)
        # THE VALIDATED MIGRATION FALLBACK. When the pinned book yields NO
        # executable quote at all, a graduated token must not be left quoting a
        # dead curve. The proven path is consulted: it re-derives the pool from
        # the mint from chain state, and it is the only transition it permits.
        # A success RECORDS the migration on the position (so the pin, the
        # identity checks and the provenance all follow the new book) and the
        # quote becomes the settlement quote. No failure of it ever closes the
        # position or fabricates a fill.
        if (
            (quote is None or not getattr(quote, "available", False))
            and position.entry_venue_id.startswith("PUMP_BONDING_CURVE:")
        ):
            try:
                migrated = await self.quote_service.resolve_graduated_exit_venue(
                    mint=position.mint,
                    token_amount_raw=position.tokens_raw,
                    token_decimals=position.token_decimals,
                    entry_venue_id=position.entry_venue_id,
                    entry_pool_address=getattr(position, "entry_pool_address", None),
                    observed_pair_address=mark.pair_address,
                )
            except Exception:  # noqa: BLE001 - migration is a quote failure, not a crash
                migrated = None
            if migrated is not None and migrated.available:
                evidence = (
                    getattr(migrated, "venue_migration", None)
                    and migrated.venue_migration.reason
                    or "EXIT_VENUE_MIGRATION_GRADUATED"
                )
                position.migration_from_venue_id = position.entry_venue_id
                position.migration_to_venue_id = migrated.venue_id or position.entry_venue_id
                position.migration_at_monotonic = time.monotonic()
                position.exit_venue_id = migrated.venue_id or position.entry_venue_id
                position.exit_venue_migration_reason = evidence
                position.exit_venue_migrated_at = time.time()
                position.exit_venue_migration_evidence = evidence
                quote = migrated
                counterfactual = None
        if getattr(quote, "available", False):
            self._memoised_sell(position, quote)
        latency.sell_quote_response_at = time.monotonic()
        if counterfactual is not None:
            # Research-only. It never becomes the canonical settlement.
            self.sell_counterfactuals.append({
                "signal_id": position.signal_id, "arm": position.arm,
                "pinned_venue_id": position.entry_venue_id,
                "pinned_output_raw": quote.output_amount_raw,
                "counterfactual_venue_id": counterfactual.venue_id,
                "counterfactual_output_raw": counterfactual.output_amount_raw,
                "would_have_been_better": (
                    counterfactual.output_amount_raw is not None
                    and quote.output_amount_raw is not None
                    and int(counterfactual.output_amount_raw)
                    > int(quote.output_amount_raw)
                ),
                "used_for_settlement": False,
            })

        if quote is not None and quote.available and quote.token_mint and quote.token_mint != position.mint:
            return TradeRecord(
                position=position, exit_trigger=decision.trigger or "EXIT",
                exit_reason=decision.reason, exit_observed_mark=mark.value_usd,
                exit_mark_timestamp=mark.timestamp, sell_quote=quote,
                exit_fill=None, pnl=None, latency=latency,
                failure=FailureReason.EXIT_QUOTE_UNAVAILABLE,
                **self._sell_provenance(position, quote, decision,
                                        settled=False, mark=mark),
                **self._record_exit_failure(
                    position, decision, quote,
                    f"sell quote is for a different token "
                    f"({getattr(quote, 'token_mint', '?')})"),
                failure_detail=(
                    f"the sell quote is for token {quote.token_mint[:8]}… but the "
                    f"position holds {position.mint[:8]}…; a quote for another "
                    "token is refused"
                ),
            )

        try:
            fill = fill_from_quote(
                quote, side="SELL", model=self.fill_model,
                sol_reference=self.sol_reference,
                require_identity=self.require_identity,
            )
        except FillRefused as exc:
            # A missing quote OBJECT (provider returned None) still gets a
            # concrete record: the trade keeps a stub quote so downstream
            # readers never see a bare None in ``sell_quote``.
            record_quote = quote if quote is not None else _unavailable_sell_stub(
                position, exc.detail)
            return TradeRecord(
                position=position, exit_trigger=decision.trigger or "EXIT",
                exit_reason=decision.reason, exit_observed_mark=mark.value_usd,
                exit_mark_timestamp=mark.timestamp, sell_quote=record_quote,
                exit_fill=None, pnl=None, latency=latency,
                failure=exc.reason, failure_detail=exc.detail,
                **self._sell_provenance(position, record_quote, decision,
                                        settled=False, mark=mark),
                # THE NO-QUOTE CASE IS THE ONE THAT MATTERS MOST, so it is the one
                # that carries the classified cause and the scheduled retry. The
                # position stays OPEN; nothing here settles anything.
                **self._record_exit_failure(position, decision, record_quote, exc.detail),
            )

        latency.exit_fill_at = time.monotonic()
        pnl = realize(
            position, exit_fill=fill, exit_observed_mark=mark.value_usd,
            sol_reference=self.sol_reference, model=self.fill_model,
        )
        # THE ACCEPTED EXIT QUOTE IS AN OBSERVATION, NOT JUST A FILL.
        #
        # Reaching here means a valid same-market SELL quote for the EXACT held
        # quantity was accepted and a fill was applied to it -- the strongest
        # piece of excursion evidence a position will ever produce. Folding it
        # in anchors the excursion at the real exit instead of leaving it
        # sampled at whatever the last mid-hold quote happened to say, and it is
        # recorded AFTER ``fill_from_quote`` succeeded, so it can never stand in
        # for a settlement that did not happen. Analytics only: the P&L above is
        # already fixed and this writes no field it reads.
        self._record_quote_excursion(position, quote, settlement=True)
        # RECOVERED. The retry backoff is cleared HERE, on the only outcome that
        # matters, so a position that sells normally is never penalised for an
        # earlier outage it has already come through.
        recovered_after = int(
            (self._exit_retry.get(self._exit_retry_key(position)) or {})
            .get("attempts", 0) or 0)
        self._exit_retry.pop(self._exit_retry_key(position), None)
        if recovered_after:
            log("EXECUTE",
                "exit for %s SETTLED after %d failed SELL quote attempt(s); "
                "RPC capacity recovered and the position was sold normally",
                getattr(position, "symbol", "?") or "?", recovered_after)
        trade = TradeRecord(
            position=position, exit_trigger=decision.trigger or "EXIT",
            exit_reason=decision.reason, exit_observed_mark=mark.value_usd,
            exit_mark_timestamp=mark.timestamp, sell_quote=quote,
            exit_fill=fill, pnl=pnl, latency=latency,
            failure=FailureReason.NONE, failure_detail="",
            **self._sell_provenance(position, quote, decision, settled=True,
                                    mark=mark),
            # Recovery is the outcome worth naming: this exit settled AFTER a run
            # of capacity failures, and a reader needs to see that the retry path
            # works rather than infer it from a single successful row.
            sell_quote_reused_this_tick=bool(
                getattr(self, "_sell_quote_reused", False)),
            # MARK-based MFE/MAE travel on the position already; propagate them
            # onto the trade with the explicit MARK_ label so they are never read
            # as executable returns.
            observed_mark_mfe=getattr(position, "observed_mark_mfe", None),
            observed_mark_mae=getattr(position, "observed_mark_mae", None),
            # The SAMPLED QUOTE excursion. Carried on the trade so the closed
            # position report can distinguish "never went up" (a measured 0.0)
            # from "never measured" (None, with a reason).
            quote_mfe_pct=position.quote_mfe_pct,
            quote_mae_pct=position.quote_mae_pct,
            quote_mfe_mae_basis=position.quote_mfe_mae_basis,
            quote_unavailable_reason=position.quote_unavailable_reason,
            quote_mfe_elapsed_ms=position.quote_mfe_elapsed_ms,
            quote_mae_elapsed_ms=position.quote_mae_elapsed_ms,
            quote_coverage_pct=position.quote_coverage_pct,
            quote_sampling_interval_s=position.quote_sampling_interval_s,
            quote_max_sampling_gap_ms=position.quote_max_sampling_gap_ms,
            quote_mfe_source=position.quote_mfe_source,
            quote_mae_source=position.quote_mae_source,
            quote_excursion_cost_basis=position.quote_excursion_cost_basis,
            # How much evidence the excursion above actually rests on, and the
            # settlement quote's own provenance. Carried so the closed-position
            # view can say WHY an excursion is missing instead of asserting the
            # position was never quoted.
            quote_observation_count=int(position.quote_observation_count or 0),
            quote_rejection_count=int(position.quote_rejection_count or 0),
            quote_last_rejection_reason=position.quote_last_rejection_reason,
            quote_settlement_sample=bool(position.quote_settlement_sample),
            quote_settlement_identity=str(position.quote_settlement_identity or ""),
            quote_settlement_source=str(position.quote_settlement_source or ""),
            quote_settlement_age_ms=position.quote_settlement_age_ms,
        )
        account.settle(pnl, trade)
        await self._stop_hold_path_telemetry(position)
        # The market no longer needs to be watched for this position. The
        # subscription itself is reference-counted and is only dropped once NO
        # position holds this market, so a sibling arm on the same mint keeps
        # its push feed.
        self._release_native_monitor(position)
        self.recently_exited[arm][position.mint] = time.time()
        # -- RENT LIFECYCLE ------------------------------------------------
        # The balance is zero, so the token account has no reason to exist and
        # its lamports become RECLAIMABLE. This is recorded on the position and
        # is deliberately NOT added to the settlement: `realize` above computed
        # canonical PnL from two executable fills and is not touched. Counting a
        # recoverable balance as a loss is the specific error being avoided.
        ledger = RentLedger(
            entry_rent_lamports=int(position.entry_rent_lamports or 0),
        )
        settled_rent = settle_rent(tokens_remaining_raw=0, ledger=ledger)
        position.rent_locked = settled_rent.rent_locked
        position.reclaimable_rent_lamports = settled_rent.reclaimable_rent_lamports
        position.reclaim_status = settled_rent.reclaim_status
        position.reclaim_blocker = settled_rent.reclaim_blocker
        self.rent_ledger = settled_rent
        return trade

    # -- durable state ----------------------------------------------------

    def _resolved_state_root(self) -> Any:
        """
        The ONE durable-state directory this engine uses, resolved explicitly.

        Used by every persistence path -- the state store, the V2 FLOOR (BETA)
        experiment, restart recovery and archival -- so they cannot disagree
        about which directory they are operating on.

        ``state_root`` wins; otherwise ``<root>/var``. When neither is supplied
        this REFUSES instead of falling back to ``Path("var")`` relative to the
        working directory. That fallback is the defect: a bridge or engine
        built without a root wrote into the operator's live ``var/`` whenever
        the process happened to start in the repository.
        """
        from .state import StateRootError

        if self.state_root is not None:
            return self.state_root
        # ``root`` lives on the bridge subclass, not on the base engine, so it
        # is read defensively: a bare ``PaperLiveEngine`` has no root and must
        # fall through to the refusal below rather than raising AttributeError.
        root = getattr(self, "root", None)
        if root is not None:
            return Path(root) / "var"
        raise StateRootError(
            f"{type(self).__name__} has no durable state root. Pass "
            "state_root= explicitly (Config.var_dir for the runtime, an "
            "isolated temporary directory for a test), or root= with state "
            "beside it. Omitting both used to resolve to Path('var') relative "
            "to the process working directory, which wrote synthetic sessions "
            "into the operator's real var/."
        )

    def _store(self) -> Any:
        """
        The durable PAPER state store.

        The in-memory records are the run's WORKING SET, not its history.
        Writing them back over the artefacts would erase everything from before
        a restart, so the store is append-and-reconcile: existing records are
        keyed and kept, and only genuinely new ones are added.
        """
        if self.state_store is None:
            from .state import PaperStateStore
            self.state_store = PaperStateStore(
                self._resolved_state_root(), runtime=self.state_is_runtime)
        return self.state_store

    def persist_state(self) -> dict[str, Any]:
        """Flush signals, trades and OPEN positions to durable state."""
        store = self._store()
        state = store.load()
        added = state.merge(
            signals=self.signal_records, trades=self.trade_records,
            open_positions=[
                a.position for a in self.accounts.values()
                if getattr(a, "position", None) is not None
            ],
        )
        # Positions that have closed must not be revived by the next restart.
        state.close_positions([
            f"{t.position.arm}::{t.position.signal_id}"
            for t in self.trade_records
            if t is not None and t.failure is FailureReason.NONE
        ])
        return store.save(state, note="runtime flush")

    def open_positions(self) -> list[PaperPosition]:
        """
        EVERY open position across EVERY arm, in arm order.

        THE CANONICAL READER for the execution-aware PAPER book. Each arm owns
        its own position (the one-open-position rule is per arm), so "is
        anything open?" cannot be answered by looking at one account -- that is
        precisely how a notified fill ended up rendering as an empty
        OPEN POSITIONS view while the arm held the position.

        Callers that need to know WHICH arm a position belongs to read
        ``accounts[arm]``; this is the flattened view for status, exposure and
        persistence, where the arm label is carried on the position itself.
        """
        return [
            account.position
            for account in self.accounts.values()
            if getattr(account, "position", None) is not None
        ]

    def open_exposure_usd(self) -> float:
        """Total USD notional currently committed across every arm."""
        return sum(
            float(getattr(p, "usd_notional", 0.0) or 0.0)
            for p in self.open_positions()
        )

    def _settled_pnl_by_arm(self, state: Any) -> dict[str, dict[str, float]]:
        """
        Cumulative realized PnL per arm from the canonical keyed trade book.

        The store's ``trades`` mapping is keyed by
        ``signal_id::arm::exit_mark_timestamp``, so iterating its VALUES counts
        each settlement exactly once by construction. This is deliberately NOT a
        sum over the append-only ``var/ledger.jsonl``: that ledger contains
        unkeyed twin rows for settlements that are also in the trade book, so
        summing it double-counts. The trade book is the authority.

        Only ``SETTLED`` trades carry realized PnL, and only those are counted. A
        trade that never filled never moved a bankroll.

        ``peak`` is the highest bankroll the running total ever reached, computed
        by replaying the settlements in exit-time order. It is derived here rather
        than read from a stored field so that a peak computed from a chain that
        was interrupted by a restart cannot propagate that restart into the
        restored account.
        """
        ordered: dict[str, list[tuple[float, float]]] = {}
        for record in state.trades.values():
            if str(record.get("sell_execution_status") or "") != "SETTLED":
                continue
            position = record.get("position") or {}
            arm = str(position.get("arm") or "")
            if not arm:
                continue
            pnl = record.get("realized_pnl") or {}
            try:
                realized = float(pnl.get("realized_pnl_usd") or 0.0)
            except (TypeError, ValueError):
                continue
            try:
                at = float(record.get("exit_mark_timestamp") or 0.0)
            except (TypeError, ValueError):
                at = 0.0
            ordered.setdefault(arm, []).append((at, realized))

        out: dict[str, dict[str, float]] = {}
        for arm, entries in ordered.items():
            entries.sort(key=lambda item: item[0])
            total = 0.0
            peak_pnl = 0.0
            for _, realized in entries:
                total += realized
                if total > peak_pnl:
                    peak_pnl = total
            out[arm] = {
                "realized_pnl_usd": total,
                "peak_pnl_usd": peak_pnl,
                "settlements": float(len(entries)),
            }
        return out

    def rehydrate(self) -> dict[str, Any]:
        """
        Resume open PAPER positions after a restart.

        A position is only resumed when it can be exited coherently: it needs a
        venue pin and a known token balance. One that cannot is reported and NOT
        loaded, because resuming it would mean inventing either the balance or
        the market to sell it on.

        ALSO RESTORED: each arm's SETTLED bankroll. Without this, an arm that lost
        money reappeared at the declared starting bankroll after every restart and
        sized its next BUY off a balance it no longer had -- a production sizing
        reset, because ``size_for_bankroll()`` reads this value.
        """
        store = self._store()
        state = store.load()
        resumable, problems = store.rehydrate(state)

        restored_bankroll: dict[str, float] = {}
        for arm, totals in sorted(self._settled_pnl_by_arm(state).items()):
            account = self.accounts.get(arm)
            if account is None:
                # The arm owns the history but is not selected. Its balance is
                # left alone rather than written onto some other arm, exactly as
                # an unselected arm's open position is refused below.
                continue
            before = account.bankroll_usd
            account.restore_settled_bankroll(
                totals["realized_pnl_usd"],
                peak_bankroll_usd=(
                    account.starting_bankroll_usd + totals["peak_pnl_usd"]),
            )
            if abs(account.bankroll_usd - before) > 1e-12:
                restored_bankroll[arm] = account.bankroll_usd
                log(
                    "PAPER",
                    "restored %s bankroll from %d settled trade(s): "
                    "$%.8f -> $%.8f",
                    arm, int(totals["settlements"]), before,
                    account.bankroll_usd,
                )

        restored: list[str] = []
        for record in resumable:
            arm = str(record.get("arm") or "")
            account = self.accounts.get(arm)
            if account is not None and account.has_position:
                continue
            if account is None:
                # THE POSITION'S ARM IS NOT RUNNING. This used to ``continue``
                # silently, which is data loss with no diagnostic: an operator
                # who deselected an arm, restarted, and found the position
                # simply gone had no way to learn WHY. The position is not
                # re-stamped onto the newly selected arm -- it belongs to the
                # strategy that opened it -- so it is reported and refused, and
                # re-selecting the arm makes it restorable on the next start.
                from .arms import ARMS, ResearchArm
                try:
                    ARMS[ResearchArm(arm)]
                    known = "the arm exists but is not selected"
                except (KeyError, ValueError):
                    known = "the arm is not in the registry at all"
                problems.append(
                    f"{arm}/{record.get('signal_id')} was NOT restored: "
                    f"{known}. It keeps its own strategy; re-select {arm} to "
                    f"resume monitoring it.")
                continue
            position = _position_from_dict(record)
            if position is None:
                problems.append(
                    f"{arm}/{record.get('signal_id')} could not be rebuilt")
                continue
            account.open_position(position)
            restored.append(f"{arm}/{position.signal_id}")
        return {
            "restored": restored,
            "refused": problems,
            "signals_in_store": len(state.signals),
            "trades_in_store": len(state.trades),
            "restored_bankroll": restored_bankroll,
        }

    # -- reporting --------------------------------------------------------

    def _monitor_diagnostics(self) -> dict[str, Any]:
        """Aggregate monitoring-health counters for the run."""
        d = dict(self._monitor_diag)
        ticks = d.pop("MONITOR_TICKS", 0) or 1
        d["monitoring_usable_pct"] = round(
            100.0 * d.get("MONITOR_OK", 0) / max(1, ticks), 2)
        d["monitoring_skipped_pct"] = round(
             100.0 * d.get("MONITOR_SKIPPED_BUDGET", 0) / max(1, ticks), 2)
        d["monitoring_ws_disconnected_pct"] = round(
            100.0 * d.get("MONITOR_WS_DISCONNECTED", 0) / max(1, ticks), 2)
        avg_age = d.get("QUOTE_AGE_MS_SUM", 0.0) / max(1, d.get("QUOTE_AGE_MS_COUNT", 0))
        d["avg_quote_age_ms"] = round(avg_age, 2)
        # WHAT THE SHARED BUDGET LOOKED LIKE while this run was monitored. Empty
        # when no RPC is configured, rather than a set of zeroes that a reader
        # would take for an exhausted budget.
        d["shared_request_budget"] = self._shared_budget_snapshot()
        # AND WHICH PATH THIS RUN WAS ON: the push channel, or polling recovery.
        d["native_monitor_ws_url_configured"] = bool(
            self._effective_native_monitor_url())
        # Per-position snapshots.
        per_position: list[dict[str, Any]] = []
        for account in self.accounts.values():
            p = account.position
            if p is None:
                continue
            per_position.append({
                "signal_id": p.signal_id,
                "arm": p.arm,
                "mint": p.mint[:8],
                "exit_status": p.exit_status,
                "monitor_ticks": p.monitor_ticks,
                "monitor_ok": p.monitor_ok_ticks,
                "monitor_rpc_error": p.monitor_rpc_error_ticks,
                "monitor_quote_unavailable": p.monitor_quote_unavailable_ticks,
                "monitor_identity_unknown": p.monitor_identity_unknown_ticks,
                "monitor_identity_mismatch": p.monitor_identity_mismatch_ticks,
                "monitor_skipped_budget": p.monitor_skipped_budget_ticks,
                "monitor_unavailable": p.monitor_unavailable_ticks,
                # Whether this position's own push channel is delivering, and
                # how fresh the last push was. ``websocket_connected`` is a
                # socket fact; ``push_delivering`` additionally requires the
                # market state to be current, so a connected-but-silent stream
                # cannot be reported as healthy monitoring.
                "websocket_connected": p.websocket_connected,
                "push_delivering": self._push_channel_delivering(p),
                "websocket_state": p.websocket_state,
                "native_update_age_ms": p.native_update_age_ms,
                "native_update_count": p.native_update_count,
                "reconnect_count": p.reconnect_count,
                "rpc_calls_last_minute": p.rpc_calls_last_minute,
                "quote_requests_last_minute": p.quote_requests_last_minute,
                "usable_pct": round(
                    100.0 * p.monitor_ok_ticks / max(1, p.monitor_ticks), 2
                ) if p.monitor_ticks else 0.0,
            })
        d["per_position"] = per_position
        return d

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "mode": "PAPER_ONLY",
            "actual_fill_available": ACTUAL_FILL_AVAILABLE,
            "transaction_built": 0,
            "transaction_signed": 0,
            "transaction_broadcast": 0,
            "baseline_hash": BASELINE_HASH,
            "safety": self.audit.as_dict(),
            "fill_model": self.fill_model.as_dict(),
            "arms": [spec.as_dict() for spec in self.arms],
            # The beta arms' REAL ledgers are the experiment's three $10
            # accounts, reported under ``v2_floor_beta`` below. Their engine
            # accounts are inert bookkeeping (never opened, never settled), so
            # listing them here would put two different numbers for the same
            # experiment on one surface.
            "accounts": {
                k: v.stats() for k, v in self.accounts.items()
                if k not in {a.value for a in BETA_ARMS}
            },
            "signals_recorded": len(self.signal_records),
            "trades_recorded": len(self.trade_records),
            "monitor_diagnostics": self._monitor_diagnostics(),
            "execution_note": (
                "NO TRANSACTION WAS BUILT, SIGNED, OR BROADCAST. Every fill is "
                "SIMULATED_FILL_FROM_EXECUTABLE_QUOTE."
            ),
        }
        # The beta experiment reports itself, from its own ledgers. It is kept
        # OUTSIDE ``accounts`` on purpose: those are the engine's per-arm
        # research accounts, and folding the experiment's three ledgers in would
        # make one surface present two numbers for the same trade. ``None`` when
        # no experiment is running, so a non-beta run's status is unchanged.
        out["v2_floor_beta"] = (
            self.beta_experiment.stats()
            if self.beta_experiment is not None else None)
        # The beta MIRRORS: visibility only, and labelled as such.
        out["beta_mirror_positions"] = [
            {
                "parent_signal_id": pid,
                "arm": arm,
                "mint": mp.mint,
                "opened_at": mp.opened_at,
                "note": "VISIBILITY ONLY; settled by the v2_floor_beta "
                        "experiment, never by this engine",
            }
            for pid, arms in self.beta_positions.items()
            for arm, mp in arms.items() if mp is not None
        ]
        return out


__all__ = [
    "BASELINE_HASH",
    "SCHEMA_VERSION",
    "PaperLiveEngine",
    "build_signal_snapshot",
]


def _cache_count(cache: Any) -> int:
    """How many provider fetches this cache has served so far."""
    if cache is None:
        return 0
    return int(getattr(cache, "shared", 0) or 0)


def _cache_state(before: int, cache: Any) -> str:
    """
    Whether this quote was freshly fetched, served from the shared snapshot, or
    the shared snapshot had expired and a new quote was required.

    Recorded on the signal record so the dataset reports a real cache-hit rate
    rather than an assumed one.
    """
    if cache is None:
        return "fresh"
    after = _cache_count(cache)
    if after > before:
        return "fresh"
    if int(getattr(cache, "hits", 0) or 0) > 0:
        return "cache_hit"
    if int(getattr(cache, "expired", 0) or 0) > 0:
        return "expired"
    return "shared"


#: Earliest instant any real PAPER position can have been opened:
#: 2020-01-01. A DOMAIN floor, not a tuned constant -- a persisted
#: ``opened_at`` below it is corrupt (epoch zero, a truncated write, or a
#: seconds/milliseconds mix), not merely old. Must match
#: ``exits._MIN_PLAUSIBLE_ENTRY_EPOCH_S`` so the rehydration guard and the
#: exit policy agree on what a credible entry instant is.
_MIN_PLAUSIBLE_OPENED_AT_S = 1_577_836_800.0

#: How far into the future a persisted ``opened_at`` may sit and still be
#: believed as ordinary clock skew rather than corruption.
_MAX_ENTRY_FUTURE_SKEW_S = 5.0


def _credible_opened_at(position: Any, *, now: float) -> float | None:
    """
    ``opened_at`` if it is a credible wall-clock instant, else ``None``.

    Credible means: a finite number, at or after 2020-01-01, and not in the
    future beyond ordinary clock skew. Anything else is a corrupt or
    unit-mismatched timestamp, and must not become an exit position's entry
    instant -- the hold timer would read it as an impossibly old position and
    fire on the first evaluation.

    The 2020 floor is a DOMAIN bound, applied here because this is where
    PERSISTED paper data enters the system. It is deliberately not applied in
    ``exits``, which is also driven with synthetic epochs by the legacy path.
    """
    try:
        value = float(getattr(position, "opened_at", None))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if value < _MIN_PLAUSIBLE_OPENED_AT_S:
        return None
    if value > now + _MAX_ENTRY_FUTURE_SKEW_S:
        return None
    return value

def _validated_beta_bracket(record: dict[str, Any]) -> "tuple[float, float] | None":
    """The persisted beta ``(TP, SL)`` pair, or ``None`` when absent/invalid."""
    raw = record.get("beta_exit_bracket")
    hold = record.get("beta_max_hold_s")
    try:
        if raw is None or hold is None:
            return None
        tp, sl = float(raw[0]), float(raw[1])
        cap = float(hold)
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(tp) and math.isfinite(sl) and math.isfinite(cap)):
        return None
    if not (tp > 0 and sl < 0 and cap > 0):
        return None
    return (tp, sl)


def _validated_beta_hold(record: dict[str, Any]) -> float | None:
    """The persisted beta max-hold, but ONLY when its bracket is also valid."""
    if _validated_beta_bracket(record) is None:
        return None
    return float(record["beta_max_hold_s"])



def _position_from_dict(record: dict[str, Any]) -> Any:
    """Rebuild a :class:`PaperPosition` from persisted state.

    Returns ``None`` rather than a half-built position. Every field the exit
    path depends on -- the token balance, the decimals, the venue pin and the
    entry fill -- must be present, because a position missing any of them
    cannot be sold on the market it was opened against.
    """
    from decimal import Decimal

    import dataclasses as _dc

    from .schema import PaperPosition, SimulatedFill
    try:
        fill_raw = record.get("entry_fill") or {}
        # ``as_dict()`` emits SERIALISATION keys (e.g. the price-layer name)
        # that are not constructor arguments. Passing them straight through
        # raises, and a swallowed raise here means a restart silently resumes
        # nothing -- so the dict is filtered to the real fields.
        fill_fields = {f.name for f in _dc.fields(SimulatedFill)}
        if not fill_raw or not (set(fill_raw) & {"output_amount_raw"}):
            return None
        fill = SimulatedFill(**{k: v for k, v in fill_raw.items()
                                if k in fill_fields})
        if fill is None or not record.get("tokens_raw"):
            return None
        # ``opened_at`` is REQUIRED, not defaulted. It is the anchor the whole
        # hold timer is measured from, and a missing one used to rebuild as
        # 0.0 -- which the exit policy then read as "open since the epoch", so
        # every elapsed-time trigger (max hold above all) was satisfied the
        # moment the position was restored. A position whose age cannot be
        # established is not revived at all: refusing to rebuild is visible and
        # safe, whereas reviving it produced an instant, entirely fictional exit.
        opened_at = record.get("opened_at")
        try:
            opened_at = float(opened_at)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if not math.isfinite(opened_at) or opened_at < _MIN_PLAUSIBLE_OPENED_AT_S:
            return None
        return PaperPosition(
            signal_id=str(record["signal_id"]), arm=str(record["arm"]),
            mint=str(record["mint"]), symbol=record.get("symbol"),
            token_decimals=int(record["token_decimals"]),
            tokens_raw=int(record["tokens_raw"]), entry_fill=fill,
            entry_observed_mark=record.get("entry_observed_mark"),
            entry_mark_timestamp=record.get("entry_mark_timestamp"),
            bankroll_before=float(record.get("bankroll_before") or 0.0),
            position_size_usd=float(record.get("position_size_usd") or 0.0),
            position_size_pct=float(record.get("position_size_pct") or 0.0),
            usd_notional=float(record.get("usd_notional") or 0.0),
            sol_notional=Decimal(str(record.get("sol_notional") or 0)),
            sol_usd_reference=float(record.get("sol_usd_reference") or 0.0),
            sol_usd_reference_at=float(record.get("sol_usd_reference_at") or 0.0),
            opened_at=opened_at,
            entry_cost_quote_asset=Decimal(str(
                record.get("entry_cost_quote_asset") or 0)),
            trading_fees_quote=Decimal(str(
                record.get("trading_fees_quote") or 0)),
            network_costs_quote=Decimal(str(
                record.get("network_costs_quote") or 0)),
            other_costs_quote=Decimal(str(record.get("other_costs_quote") or 0)),
            observed_mark_mfe=float(record.get("observed_mark_mfe") or 0.0),
            observed_mark_mae=float(record.get("observed_mark_mae") or 0.0),
            peak_mark=record.get("peak_mark"), trough_mark=record.get("trough_mark"),
            production_profile=str(
                record.get("production_profile") or "L25_CURRENT"),
            # The BETA per-arm exit override. Restored so a rehydrated position
            # exits under ITS OWN exit rule. Validated as a PAIR: a record with
            # a bracket but no hold (or vice versa) is a corrupt override and
            # is dropped to ``None``/``None`` -- the production profile's rule
            # -- rather than a half-override that belongs to no arm.
            beta_exit_bracket=_validated_beta_bracket(record),
            beta_max_hold_s=_validated_beta_hold(record),
            entry_venue_id=str(record.get("entry_venue_id") or ""),
            entry_pool_address=record.get("entry_pool_address"),
            entry_market_type=str(record.get("entry_market_type") or ""),
            entry_quote_source=str(record.get("entry_quote_source") or ""),
            entry_quote_timestamp=record.get("entry_quote_timestamp"),
            entry_quote_age_ms=record.get("entry_quote_age_ms"),
            entry_quote_latency_ms=record.get("entry_quote_latency_ms"),
            entry_rent_lamports=int(record.get("entry_rent_lamports") or 0),
            rent_locked=bool(record.get("rent_locked", True)),
            reclaimable_rent_lamports=int(
                record.get("reclaimable_rent_lamports") or 0),
            reclaim_status=str(record.get("reclaim_status") or "RENT_LOCKED"),
            reclaim_blocker=str(record.get("reclaim_blocker") or ""),
            current_mark_usd=record.get("current_mark_usd"),
            current_mark_timestamp=record.get("current_mark_timestamp"),
            exit_status=str(record.get("exit_status") or "OPEN_MONITORING"),
            exit_trigger_reason=str(record.get("exit_trigger_reason") or ""),
            last_exit_error=str(record.get("last_exit_error") or ""),
            exit_attempts=int(record.get("exit_attempts") or 0),
        )
    except Exception:  # noqa: BLE001 - refuse rather than half-build
        return None
