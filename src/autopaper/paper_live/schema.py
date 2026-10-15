"""
The forward-validation schema.

WHY SO MANY TYPES INSTEAD OF ONE PRICE FIELD
============================================

The failure this schema exists to prevent is a single ``price`` column being
written by a mark on one side and a quote on the other. That is not a tidy
problem: it produced a ``+71.56%`` paper gain on a trade that lost 42.91% of its
cash. So each of the four price layers is its own object, and each carries the
provenance needed to tell it apart from the others after the fact:

* :class:`ObservedMarkPoint` - what DexScreener said. Never a fill.
* :class:`QuotePoint` - a read-only executable quote for this exact size.
* :class:`SimulatedFill` - what this harness assumed happened, derived only
  from a quote, labelled ``SIMULATED_FILL_FROM_EXECUTABLE_QUOTE``.
* :class:`RealizedPnl` - the settlement arithmetic, with every component
  itemised so the number can be taken apart.

``ACTUAL_FILL`` exists as a label and is always ``None``:
:data:`ACTUAL_FILL_AVAILABLE` is a hard ``False``.

WHAT IS NOT STORED
==================

No credential, no session, no key. :class:`SignalSnapshot` keeps a HASH of the
raw signal text plus the token fields actually used, rather than the message
body, so the dataset can be joined back to a message without the dataset itself
becoming a copy of the channel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping

from autopaper.same_market.numeric import to_float

#: There is no actual fill in PAPER. A hard constant so no code path can imply
#: otherwise.
ACTUAL_FILL_AVAILABLE: bool = False

#: The label attached to every simulated fill. Explicit at the point of
#: creation, because "a fill" is exactly the word that gets dropped.
SIMULATED_FILL_KIND = "SIMULATED_FILL_FROM_EXECUTABLE_QUOTE"


class PriceLayer(str, Enum):
    """The price kinds. Never interchangeable."""

    OBSERVED_MARK = "OBSERVED_MARK"
    EXECUTABLE_QUOTE = "EXECUTABLE_QUOTE"
    SIMULATED_FILL = "SIMULATED_FILL"
    ACTUAL_FILL = "ACTUAL_FILL"
    #: The SAME market's own current price, from its own on-chain state, used
    #: as the entry guard's reference. Distinct from OBSERVED_MARK on purpose:
    #: that one is an external aggregator's view of the market, this one is the
    #: market. Collapsing them is the defect this layer was added to fix.
    EXECUTION_NATIVE_REFERENCE = "EXECUTION_NATIVE_REFERENCE"


class FailureReason(str, Enum):
    """
    Every way a simulated trade can fail, named.

    The fail-closed set matters most: an unavailable or stale executable quote
    means NO simulated fill. It is never replaced by the observed mark, an old
    quote, or a different venue's route.
    """

    NONE = "NONE"
    TOKEN_NOT_FOUND = "TOKEN_NOT_FOUND"
    TOKEN_AMBIGUOUS = "TOKEN_AMBIGUOUS"
    ENTRY_BLOCKED_POSITION_ACTIVE = "ENTRY_BLOCKED_POSITION_ACTIVE"
    ENTRY_BLOCKED_COOLDOWN = "ENTRY_BLOCKED_COOLDOWN"
    L25_REJECTED = "L25_REJECTED"
    ARM_FILTER_REJECTED = "ARM_FILTER_REJECTED"
    SIZING_REFUSED = "SIZING_REFUSED"
    ENTRY_REJECTED_QUOTE_UNAVAILABLE = "ENTRY_REJECTED_QUOTE_UNAVAILABLE"
    ENTRY_REJECTED_QUOTE_STALE = "ENTRY_REJECTED_QUOTE_STALE"
    ENTRY_REJECTED_IDENTITY_UNKNOWN = "ENTRY_REJECTED_IDENTITY_UNKNOWN"
    TOKEN_MINT_MISMATCH = "TOKEN_MINT_MISMATCH"
    QUOTE_MINT_MISMATCH = "QUOTE_MINT_MISMATCH"
    ENTRY_REJECTED_DECIMALS_UNKNOWN = "ENTRY_REJECTED_DECIMALS_UNKNOWN"
    EXECUTABLE_BUY_QUOTE_UNAVAILABLE = "EXECUTABLE_BUY_QUOTE_UNAVAILABLE"
    MARK_GAP_GUARD_REJECTED = "MARK_GAP_GUARD_REJECTED"
    MARK_GAP_GUARD_UNKNOWN = "MARK_GAP_GUARD_UNKNOWN"
    #: The V2 FLOOR (BETA) entry gate refused the signal.
    V2_FLOOR_REJECTED = "V2_FLOOR_REJECTED"
    #: The SAME market could not produce its own current price, so the entry
    #: guard has nothing valid to compare the executable quote against.
    #:
    #: It is deliberately NOT a fall back to the observed mark. The mark is an
    #: external aggregate's snapshot; when the market's own state cannot be
    #: read, comparing against it would reintroduce exactly the lagging-
    #: reference defect this reference exists to remove. Fail closed.
    EXECUTION_NATIVE_REFERENCE_UNAVAILABLE = (
        "EXECUTION_NATIVE_REFERENCE_UNAVAILABLE")
    #: The reference and the executable quote could be shown to describe
    #: DIFFERENT markets, or to differ in side, decimals, quote asset or order
    #: size. Separate from UNAVAILABLE on purpose: this one is a mismatch, not
    #: an absence, and conflating them would hide a real defect.
    EXECUTION_NATIVE_REFERENCE_MISMATCH = (
        "EXECUTION_NATIVE_REFERENCE_MISMATCH")
    EXIT_VENUE_UNAVAILABLE = "EXIT_VENUE_UNAVAILABLE"
    MICRO_ACCOUNT_ECONOMICALLY_INVALID = "MICRO_ACCOUNT_ECONOMICALLY_INVALID"
    EXIT_QUOTE_UNAVAILABLE = "EXIT_QUOTE_UNAVAILABLE"
    EXIT_QUOTE_STALE = "EXIT_QUOTE_STALE"
    EXIT_QUOTE_IDENTITY_UNKNOWN = "EXIT_QUOTE_IDENTITY_UNKNOWN"
    EXIT_QUOTE_MINT_MISMATCH = "EXIT_QUOTE_MINT_MISMATCH"
    MARKET_DATA_MISSING = "MARKET_DATA_MISSING"
    SOL_REFERENCE_MISSING = "SOL_REFERENCE_MISSING"
    # -- QUOTE CAPACITY, kept DISTINCT on purpose -----------------------
    #
    # These four used to collapse into one "no executable quote" verdict. That
    # made an upstream node being throttled indistinguishable from a token
    # genuinely having no market, which is how a CAPACITY limit got reported
    # as a QUALITY verdict and then investigated as a strategy problem.
    QUOTE_TIMEOUT = "QUOTE_TIMEOUT"
    RPC_RATE_LIMIT = "RPC_RATE_LIMIT"
    RPC_ERROR = "RPC_ERROR"
    QUOTE_UNAVAILABLE = "QUOTE_UNAVAILABLE"
    ARITHMETIC_ERROR = "ARITHMETIC_ERROR"


#: A close whose trade record says ``NONE`` and yet carries no exit fill.
#:
#: It is not a quote problem, so it is not reported as one: a record that claims
#: success and holds no fill is an internal contradiction, and the only honest
#: description of it is a fill that could not be constructed. It replaces the
#: bare string ``"NONE"``, which is what the kill switch used to print for a
#: position that had actually closed -- an operator cannot act on "NONE", and an
#: unreadable failure is indistinguishable from no failure at all.
ARM_CLOSE_NO_FILL_CONTRADICTION = "EXIT_FILL_CONSTRUCTION_FAILED"

#: A close that settled in memory but whose closed record could not be written.
#:
#: Distinct from every quote failure on purpose. The sale is not the problem; the
#: durability of its record is. Reporting it as a quote failure would send an
#: operator to investigate the provider for a disk or ledger problem, and
#: reporting it as a close would hide a position that a restart can revive.
ARM_CLOSE_SETTLEMENT_FAILED = "EXIT_SETTLEMENT_FAILED"


@dataclass
class LatencyMarks:
    """
    Timing for one signal, on monotonic clocks.

    Mutable, because it is FILLED IN as the signal progresses. Frozen would be
    tidier but wrong: these stamps are recorded at the moment each stage
    actually completes, which is the only way the numbers mean anything.

    Durations use ``time.monotonic``, which cannot jump backwards when the wall
    clock is adjusted mid-run. Wall-clock stamps are kept separately and used
    only for CORRELATION - pairing a Telegram message with a report - never for
    a duration.

    The distinction that matters most here: ``telegram_message_at`` is NOT the
    entry time. The hypothetical entry happens when this engine receives the
    signal, runs L25 and obtains a fresh executable quote, and
    ``signal_to_simulated_entry_ms`` measures that, not the age of the message.
    """

    telegram_message_at: float | None = None
    receive_at_monotonic: float | None = None
    receive_at_wall: float | None = None
    mark_request_at: float | None = None
    mark_response_at: float | None = None
    decision_start_at: float | None = None
    decision_end_at: float | None = None
    buy_quote_request_at: float | None = None
    #: When the executable quote actually STARTED, as distinct from the moment
    #: the decision began. Without this, "quote latency" and "quote wait time"
    #: are the same number and capacity stalls cannot be told apart from slow
    #: pricing.
    quote_started_at: float | None = None
    #: The decision budget the quote was given, so a refusal can say whether it
    #: ran out of time rather than ran out of market.
    quote_deadline_ms: float | None = None
    #: Set when the quote could not be produced, using the named capacity
    #: classification (RPC_RATE_LIMIT / RPC_TIMEOUT / RPC_ERROR /
    #: QUOTE_UNAVAILABLE) rather than a generic miss.
    quote_capacity_failure: str = ""
    #: Concurrent in-flight quote requests that were collapsed onto one shared
    #: quote. Non-zero means quote sharing did its job.
    quote_coalesced: int = 0
    #: The executable-vs-mark entry guard verdict for this signal.
    mark_gap_guard_result: str = ""
    #: The research-only legacy gap (executable vs external observed mark),
    #: measured on every entry so both references are comparable on one record.
    legacy_observed_mark_gap_pct: float | None = None
    #: THE PRODUCTION GAP: executable vs the SAME MARKET's own on-chain spot.
    #: This is the number the 10% guard decided on.
    execution_native_gap_pct: float | None = None
    #: The production guard's verdict.
    native_guard_verdict: str = ""
    #: DIAGNOSTIC ONLY. What the LEGACY aggregator-mark guard WOULD have decided,
    #: kept so OLD and NEW verdicts stay comparable on one candidate.
    legacy_guard_verdict: str = ""
    #: "EXECUTION_NATIVE_SAME_MARKET" for every production decision. Persisted
    #: so a reader can never assume which reference decided the trade.
    production_guard_source: str = ""
    #: The execution-native comparison. DIAGNOSTIC ONLY -- it does not decide.
    execution_native_gap_pct: float | None = None
    legacy_gap_guard_result: str = ""
    #: Which reference the production guard actually used, recorded so a record
    #: can never be ambiguous about what decided it.
    production_guard_source: str = ""
    buy_quote_response_at: float | None = None
    entry_fill_at: float | None = None
    exit_trigger_at: float | None = None
    #: WALL-CLOCK companion to :attr:`exit_trigger_at`. ALL durations keep
    #: using the monotonic stamp; this exists only so a report row can carry a
    #: trigger timestamp that is comparable to the other (wall-clock)
    #: timestamps in the same row. Without it, the CSV emitted a monotonic
    #: value (~3e5) in a timestamp column next to UNIX stamps (~1.79e9).
    exit_trigger_wall: float | None = None
    sell_quote_request_at: float | None = None
    sell_quote_response_at: float | None = None
    exit_fill_at: float | None = None

    @staticmethod
    def _ms(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return max(0.0, (b - a) * 1000.0)

    @property
    def mark_latency_ms(self) -> float | None:
        return self._ms(self.mark_request_at, self.mark_response_at)

    @property
    def decision_latency_ms(self) -> float | None:
        return self._ms(self.decision_start_at, self.decision_end_at)

    @property
    def buy_quote_latency_ms(self) -> float | None:
        return self._ms(self.buy_quote_request_at, self.buy_quote_response_at)

    @property
    def sell_quote_latency_ms(self) -> float | None:
        return self._ms(self.sell_quote_request_at, self.sell_quote_response_at)

    @property
    def receive_to_decision_ms(self) -> float | None:
        return self._ms(self.receive_at_monotonic, self.decision_start_at)

    @property
    def decision_to_buy_quote_ms(self) -> float | None:
        return self._ms(self.decision_end_at, self.buy_quote_request_at)

    @property
    def signal_to_simulated_entry_ms(self) -> float | None:
        """
        Receive -> the simulated entry fill existing.

        This is the honest "how fast could we have acted" number. It is NOT
        ``telegram_timestamp - entry``, which would credit the engine with the
        Telegram round trip it did not control.
        """
        return self._ms(self.receive_at_monotonic, self.entry_fill_at)

    @property
    def exit_trigger_to_sell_quote_ms(self) -> float | None:
        return self._ms(self.exit_trigger_at, self.sell_quote_request_at)

    @property
    def exit_trigger_to_fill_ms(self) -> float | None:
        return self._ms(self.exit_trigger_at, self.exit_fill_at)

    def as_dict(self) -> dict[str, Any]:
        return {
            "clock": (
                "monotonic for every duration; wall clock only for "
                "cross-system correlation, where clock skew between Telegram "
                "and this host is included"
            ),
            "telegram_to_receive_ms_cross_system": (
                None
                if self.telegram_message_at is None or self.receive_at_wall is None
                else max(0.0, (self.receive_at_wall - self.telegram_message_at) * 1000.0)
            ),
            "mark_latency_ms": to_float(self.mark_latency_ms),
            "receive_to_decision_ms": to_float(self.receive_to_decision_ms),
            "decision_latency_ms": to_float(self.decision_latency_ms),
            "decision_to_buy_quote_ms": to_float(self.decision_to_buy_quote_ms),
            "buy_quote_latency_ms": to_float(self.buy_quote_latency_ms),
            "signal_to_simulated_entry_ms": to_float(self.signal_to_simulated_entry_ms),
            "sell_quote_latency_ms": to_float(self.sell_quote_latency_ms),
            "exit_trigger_to_sell_quote_ms": to_float(self.exit_trigger_to_sell_quote_ms),
            "exit_trigger_to_fill_ms": to_float(self.exit_trigger_to_fill_ms),
        }


@dataclass(frozen=True)
class SignalSnapshot:
    """
    One immutable observation of a signal.

    ``raw_text_sha256`` rather than the text: the dataset should be able to be
    joined back to a channel message by id, and should not itself become a copy
    of every message the bot ever saw. The token fields actually used by the
    decision ARE recorded in full.
    """

    signal_id: str
    channel: str
    message_id: int | None
    telegram_timestamp: float | None
    receive_timestamp: float
    receive_monotonic: float
    token_mint: str | None
    symbol: str | None
    raw_text_sha256: str
    raw_text_length: int
    extraction_provenance: Mapping[str, Any] = field(default_factory=dict)
    extraction_warnings: tuple[str, ...] = ()
    ambiguous: bool = False
    candidates: tuple[str, ...] = ()
    #: V2 FLOOR gate inputs, when the signal was evaluated under the V2 FLOOR
    #: (BETA) experiment. ``None`` when the signal never reached the gate ---
    #: kept distinct from 0.0, which would read as a measured zero. Stored so
    #: the entry decision is reproducible from the record alone.
    v2_volume_5m_usd: float | None = None
    v2_market_cap_usd: float | None = None
    v2_change_5m_pct: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "channel": self.channel,
            "message_id": self.message_id,
            "telegram_timestamp": self.telegram_timestamp,
            "receive_timestamp": self.receive_timestamp,
            "token_mint": self.token_mint,
            "symbol": self.symbol,
            "raw_text_sha256": self.raw_text_sha256,
            "raw_text_length": self.raw_text_length,
            "extraction_provenance": dict(self.extraction_provenance),
            "extraction_warnings": list(self.extraction_warnings),
            "ambiguous": self.ambiguous,
            "candidates": list(self.candidates),
            "v2_volume_5m_usd": to_float(self.v2_volume_5m_usd),
            "v2_market_cap_usd": to_float(self.v2_market_cap_usd),
            "v2_change_5m_pct": to_float(self.v2_change_5m_pct),
            "layer": PriceLayer.OBSERVED_MARK.value,
        }


@dataclass(frozen=True)
class ExecutionNativeReference:
    """
    ``EXECUTION_NATIVE_REFERENCE`` - the current price of the SAME market the
    order will actually trade in, read from that market's own on-chain state.

    WHY THIS EXISTS
    ===============

    The entry guard used to compare the executable BUY against DexScreener's
    ``priceUsd``. That is an EXTERNAL AGGREGATE'S SNAPSHOT of the same pool,
    refreshed on the aggregator's own cadence. Measured consequence: on L25's
    target population -- bonding curves minutes old, moving hundreds of percent
    -- the aggregate's snapshot trailed the curve by a median spread with a
    standard deviation of ~26 percentage points, while the true execution
    premium was 1.26% with a standard deviation of 0.003. The guard was
    therefore measuring aggregator lag, not execution quality, and rejected
    100% of authorized entries.

    So the reference is now the market's OWN current price, taken from the same
    decoded account, at the same slot, in the same call that produced the
    executable quote. It is not a second data source and not a second identity
    system: ``market_id`` is the same ``<market_type>:<pool>`` the execution
    layer already pins an exit to.

    WHAT THE PRICE MEANS
    ===================

    ``reference_price`` is the market's SPOT (marginal) price in QUOTE ASSET
    PER WHOLE TOKEN -- what this market charges before this particular order is
    layered on. The executable quote is an AVERAGE price for the exact order
    size. Those are deliberately different quantities, and the difference
    between them is exactly the thing the guard is asked to police:

        gap = (average_price_for_this_size / marginal_price_now) - 1

    is the fee plus the size impact of transacting this order right now, in
    this pool, at this slot. Comparing an average fill against a spot price is
    not a category error here; the gap between them IS the measurement.

    NOT A PRICE ANYONE RECEIVED. This is a reference, never a fill, and no
    field on it can be mistaken for one: it carries no output amount.
    """

    market_id: str
    mint: str
    base_mint: str
    quote_mint: str
    side: str
    #: Quote asset per WHOLE token. NOT USD: the quote is taken against the
    #: quote asset, and calling this "usd" is the exact SOL-vs-USD mislabelling
    #: that produced three-figure "gaps" elsewhere in this repository. The USD
    #: figure is derived by the caller against a logged SOL reference.
    reference_price: float
    #: The exact integer reserve ratio the price came from, so the derivation
    #: is reproducible and auditable rather than asserted.
    numerator_raw: int | None
    denominator_raw: int | None
    #: ``PUMP_BONDING_CURVE_VIRTUAL_RESERVES`` or
    #: ``PUMPSWAP_EFFECTIVE_RESERVES``. Named so a reader can tell a
    #: constant-product virtual-reserve curve from a real-vault pool.
    calculation_method: str
    venue: str | None
    pool_or_curve: str | None
    provider: str
    #: The on-chain slot the state was read at. Two references from different
    #: slots are not the same moment, which is why this is recorded rather than
    #: inferred.
    slot: int | None = None
    #: When the state was FETCHED, and its age at the moment of comparison.
    observed_at: float = 0.0
    state_age_ms: float | None = None
    #: The token decimals the reference was scaled with, so a mismatch against
    #: the quote's decimals is detectable rather than assumed away.
    token_decimals: int | None = None

    def age_ms(self, now: float | None = None) -> float | None:
        if not self.observed_at:
            return None
        import time as _time

        return max(0.0, ((now or _time.time()) - self.observed_at) * 1000.0)

    def as_dict(self, now: float | None = None) -> dict[str, Any]:
        return {
            "layer": PriceLayer.EXECUTION_NATIVE_REFERENCE.value,
            "market_id": self.market_id,
            "mint": self.mint,
            "base_mint": self.base_mint,
            "quote_mint": self.quote_mint,
            "side": self.side,
            "reference_price": self.reference_price,
            "reference_price_unit": "quote_asset_per_whole_token",
            "numerator_raw": self.numerator_raw,
            "denominator_raw": self.denominator_raw,
            "calculation_method": self.calculation_method,
            "venue": self.venue,
            "pool_or_curve": self.pool_or_curve,
            "provider": self.provider,
            "slot": self.slot,
            "observed_at": self.observed_at,
            "age_ms": self.age_ms(now),
            "state_age_ms": self.state_age_ms,
            "token_decimals": self.token_decimals,
        }


@dataclass(frozen=True)
class ObservedMarkPoint:
    """
    ``OBSERVED_MARK`` - DexScreener's view. An observation about a pool.

    Used for L25 features, for MFE/MAE, and -- for RESEARCH ONLY -- as the
    legacy ``old_gap_pct`` comparison.

    It is NOT the entry guard's reference any more. That role belongs to
    :class:`ExecutionNativeReference`, because this is an external aggregate's
    snapshot of the market rather than the market. It is retained in full so
    "did the token move?" can be answered separately from "was execution
    expensive?", and so the old and new gaps can be measured against each other
    on the same record.

    NEVER used as a fill, and this type carries no field that could be mistaken
    for one: there is no ``fill`` here to reach for.
    """

    value_usd: float | None
    pair_address: str | None
    dex_id: str | None
    liquidity_usd: float | None
    volume_5m_usd: float | None
    market_cap_usd: float | None
    #: WHEN WE RECEIVED IT. A transport fact, and always known.
    timestamp: float
    available: bool
    missing_reason: str | None = None
    #: SOURCE OBSERVATION TIME, or ``None`` when the provider does not expose
    #: one. DexScreener publishes no per-observation price timestamp -- only
    #: ``pairCreatedAt``, which is when the PAIR was created, not when this
    #: price was observed. Those are different facts, and conflating them is
    #: how a stale print came to look fresh.
    source_observed_at: float | None = None
    #: SOURCE AGE: ``received_at - source_observed_at``. ``None`` whenever the
    #: source timestamp is unavailable, because an invented 0.0 asserts "this
    #: price is current" -- precisely the claim that cannot be made here.
    age_ms: float | None = None
    #: KNOWN | UNKNOWN | INVALID. ``UNKNOWN`` is the honest state for an
    #: aggregator with no per-observation timestamp, and it is NOT fresh.
    age_status: str = "UNKNOWN"
    #: ``decision_time - received_at``. Named TRANSPORT age on purpose: it
    #: measures our own pipeline, never the provider's staleness.
    transport_age_ms: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer": PriceLayer.OBSERVED_MARK.value,
            "value_usd": to_float(self.value_usd),
            "pair_address": self.pair_address,
            "dex_id": self.dex_id,
            "liquidity_usd": to_float(self.liquidity_usd),
            "volume_5m_usd": to_float(self.volume_5m_usd),
            "market_cap_usd": to_float(self.market_cap_usd),
            "timestamp": self.timestamp,
            "age_ms": to_float(self.age_ms),
            "available": self.available,
            "missing_reason": self.missing_reason,
        }


@dataclass(frozen=True)
class VenueMigration:
    """
    A PROVEN change of the market a token trades on.

    WHY THIS EXISTS AS ITS OWN RECORD
    =================================
    A position is pinned to the market it was FILLED on, because a different
    pool is a different book with different depth and a better price there is a
    different trade rather than a better exit. That rule is correct, and it has
    exactly one exception in this system: a Pump.fun bonding curve is not a
    permanent book. It completes, it drains, and the same token continues on a
    PumpSwap pool. Refusing to follow that transition is not conservatism, it is
    a position that can never be sold.

    So the transition is recorded as a PROOF rather than inferred from a failed
    quote. Every field here is evidence a reader can re-check:

    * ``reason`` is the lifecycle event, and there is exactly one supported
      value -- ``BONDING_CURVE_COMPLETED``. A transition that cannot be named
      this way does not get this record.
    * ``from_venue_id`` / ``to_venue_id`` name the exact books on either side,
      so a venue mismatch on a migrated position is always visible, never
      hidden behind a rounded or shared venue label.
    * ``verified`` is the chain evidence, as text. A reader can see that the
      curve account was owned by the pump.fun program, carried the
      BondingCurve discriminator and reported ``complete``/drained reserves
      BEFORE any PumpSwap pool was considered.
    * ``minted_by`` is the token whose identity was checked on the far side. The
      pool's own base mint must equal it; a pool for another token is refused
      rather than relabelled.
    * ``at`` is when it was proven, so the transition can be placed in the
      position's timeline.

    NOT IN HERE, DELIBERATELY
    =========================
    No price, no proceeds, no PnL. A venue change moves where a fill happens;
    it settles nothing. Anything that carried a number here would be an
    invitation to read a migration as a trade.
    """

    reason: str
    from_venue_id: str
    to_venue_id: str
    from_market_type: str
    to_market_type: str
    #: The token whose identity was verified on BOTH sides of the transition.
    token_mint: str
    #: Chain evidence for the graduation, in words a reader can re-check.
    verified: str
    at: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "from_venue_id": self.from_venue_id,
            "to_venue_id": self.to_venue_id,
            "from_market_type": self.from_market_type,
            "to_market_type": self.to_market_type,
            "token_mint": self.token_mint,
            "verified": self.verified,
            "at": self.at,
        }


#: The ONE lifecycle transition that may move an exit venue. A transition with
#: any other cause is not supported and must not be constructed.
EXIT_VENUE_MIGRATION_GRADUATED = "BONDING_CURVE_COMPLETED"


@dataclass(frozen=True)
class QuotePoint:
    """
    ``EXECUTABLE_QUOTE`` - a read-only price for the EXACT size, from state.

    A quote is not a promise: the market can move before anything is submitted.
    It is not a fill either; it is the input to one.

    ``identity`` is the same-market verdict. ``"UNKNOWN"`` means exact identity
    could not be proven, and the PAPER trade then FAILS CLOSED - the mark is
    retained as a non-trade observation.
    """

    side: str                      # "BUY" | "SELL"
    provider: str
    market_type: str
    token_mint: str
    quote_mint: str
    input_amount_raw: int | None
    input_decimals: int | None
    input_amount_decimal: float | None
    output_amount_raw: int | None
    output_decimals: int | None
    output_amount_decimal: float | None
    effective_price: Decimal | None      # quote asset per whole token
    effective_price_usd: float | None
    #: NAMING HAZARD, recorded so the next reader does not have to rediscover
    #: it. For a direct Pump quote this is
    #: ``(effective_price / execution_native_reference - 1)``, i.e. the FEE
    #: plus the size impact -- which is what a Jupiter quote calls
    #: ``priceImpactPct`` only by coincidence of label, not by equality of
    #: quantity. Nothing in this system ARITHMETICS on it; it is stored and
    #: displayed. The pure size impact on its own is
    #: :class:`~autopaper.paper_live.quote_diagnostics.QuoteCostBreakdown.impact_bps`.
    price_impact_pct: float | None
    fee_bps: int | None
    fee_amount: Decimal | None
    venue: str | None
    pool_or_curve: str | None
    route: tuple[str, ...] = ()
    state_slot: int | None = None
    state_age_ms: float | None = None
    quote_timestamp: float | None = None
    quote_age_ms: float | None = None
    max_quote_age_ms: float | None = None
    quote_latency_ms: float | None = None
    identity: str = "UNKNOWN"
    identity_evidence: str | None = None
    observed_pair_address: str | None = None
    #: A STABLE MACHINE identity for the exact market, deliberately distinct
    #: from ``venue``, which is a human-readable label. "Pump.fun" names a
    #: FAMILY: a bonding curve and a graduated PumpSwap pool are different books
    #: with different depth and different counterparties, and an exit pinned to
    #: the wrong one is not the position that was opened. So ``venue_id`` is
    #: ``<market_type>:<pool_or_curve>`` and can never be merely "Pump.fun".
    venue_id: str = ""
    available: bool = False
    missing_reason: str | None = None
    #: The SAME market's current on-chain price, from the same snapshot that
    #: produced this quote. This is what the entry guard compares against.
    execution_native_reference: "ExecutionNativeReference | None" = None
    #: THE QUOTE-SIDE AMOUNT AFTER THE VENUE'S FEE, in raw integer units:
    #: the quote-asset leg BEFORE the fee on a BUY, and the quote-asset leg
    #: BEFORE the fee on a SELL. This is the one number that lets a reader
    #: split ``total cost = fee + size impact`` from integers alone, without
    #: re-deriving or re-assuming any fee schedule.
    #:
    #: ``None`` for providers that do not disclose it (Jupiter) and for
    #: unavailable quotes. It is deliberately NOT written into ``fee_amount``,
    #: which stays as it was: the simulated fill's fee accounting must not
    #: change because a display line got more informative.
    amount_after_fee_raw: int | None = None
    #: SET ONLY when this quote was taken on a DIFFERENT market from the one the
    #: position was opened on, because a chain-verified lifecycle transition
    #: moved the token's market. The single supported transition is
    #: ``PUMP_BONDING_CURVE -> PUMPSWAP`` (Pump.fun graduation), and it may only
    #: be set once graduation has been PROVEN on chain and the new market's
    #: identity has been verified as the same token.
    #:
    #: ``None`` for every ordinary quote, which is the overwhelming majority. It
    #: is deliberately not a boolean: a reader has to be able to tell WHICH
    #: transition occurred and what evidence supported it, and a flag cannot say
    #: that.
    venue_migration: "VenueMigration | None" = None

    @property
    def market_id(self) -> str:
        """
        The canonical identity of the EXACT market this quote trades in.

        Identical to the reference's ``market_id`` by construction -- both name
        ``<market_type>:<pool_or_curve>`` -- and the guard compares them
        explicitly rather than assuming the match.
        """
        return self.venue_id

    @property
    def stale(self) -> bool:
        """Too old to act on. A configured window that is exceeded."""
        return (
            self.max_quote_age_ms is not None
            and self.quote_age_ms is not None
            and self.quote_age_ms > self.max_quote_age_ms
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer": PriceLayer.EXECUTABLE_QUOTE.value,
            "side": self.side,
            "provider": self.provider,
            "market_type": self.market_type,
            "token_mint": self.token_mint,
            "quote_mint": self.quote_mint,
            "input_amount_raw": self.input_amount_raw,
            "input_decimals": self.input_decimals,
            "input_amount_decimal": to_float(self.input_amount_decimal),
            "output_amount_raw": self.output_amount_raw,
            "output_decimals": self.output_decimals,
            "output_amount_decimal": to_float(self.output_amount_decimal),
            "effective_price": to_float(self.effective_price),
            "effective_price_usd": to_float(self.effective_price_usd),
            "price_impact_pct": to_float(self.price_impact_pct),
            "fee_bps": self.fee_bps,
            "fee_amount": to_float(self.fee_amount),
            "venue": self.venue,
            "pool_or_curve": self.pool_or_curve,
            "route": list(self.route),
            "state_slot": self.state_slot,
            "state_age_ms": to_float(self.state_age_ms),
            "quote_timestamp": self.quote_timestamp,
            "quote_age_ms": to_float(self.quote_age_ms),
            "max_quote_age_ms": to_float(self.max_quote_age_ms),
            "quote_latency_ms": to_float(self.quote_latency_ms),
            "identity": self.identity,
            "identity_evidence": self.identity_evidence,
            "observed_pair_address": self.observed_pair_address,
            "available": self.available,
            "stale": self.stale,
            "missing_reason": self.missing_reason,
            "amount_after_fee_raw": self.amount_after_fee_raw,
            "venue_migration": (
                self.venue_migration.as_dict()
                if self.venue_migration is not None else None),
        }


@dataclass(frozen=True)
class SimulatedFill:
    """
    ``SIMULATED_FILL`` - derived ONLY from an executable quote.

    Two rules are structural rather than documented:

    1. The fill price IS the quote's effective price. Nothing is added on top.
       An executable quote already prices this exact size, so adding a slippage
       percentage again would double-count it. Price impact reported by the
       venue is retained SEPARATELY, not compounded into the price.

    2. There is no field that a mark could be written into. ``source_price_kind``
       is fixed at :data:`EXECUTABLE_QUOTE` and is asserted by the test that a
       mark cannot become a fill.
    """

    side: str                       # "BUY" | "SELL"
    input_amount_raw: int
    input_mint: str
    input_decimals: int
    input_amount_decimal: float
    output_amount_raw: int
    output_mint: str
    output_decimals: int
    output_amount_decimal: float
    effective_price: Decimal         # quote asset per whole token
    effective_price_usd: float | None
    fee: Decimal
    fee_bps: int | None
    price_impact_pct: float | None
    quote_timestamp: float
    quote_age_ms: float | None
    max_quote_age_ms: float | None
    provider: str
    venue: str | None
    pool_or_curve: str | None
    route: tuple[str, ...] = ()
    identity: str = "UNKNOWN"
    fill_timestamp: float = 0.0
    #: Fixed label. Declared last so it is a default, and asserted by the test
    #: that a mark cannot be written into a fill: the kind is always this, and
    #: the price always came from a quote.
    kind: str = SIMULATED_FILL_KIND
    note: str = (
        "simulated from an executable quote; no transaction was built, signed "
        "or broadcast; no extra slippage was added on top of the quote"
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer": PriceLayer.SIMULATED_FILL.value,
            "kind": self.kind,
            "side": self.side,
            "input_amount_raw": self.input_amount_raw,
            "input_mint": self.input_mint,
            "input_decimals": self.input_decimals,
            "input_amount_decimal": self.input_amount_decimal,
            "output_amount_raw": self.output_amount_raw,
            "output_mint": self.output_mint,
            "output_decimals": self.output_decimals,
            "output_amount_decimal": self.output_amount_decimal,
            "effective_price": to_float(self.effective_price),
            "effective_price_usd": to_float(self.effective_price_usd),
            "fee": to_float(self.fee),
            "fee_bps": self.fee_bps,
            "price_impact_pct": to_float(self.price_impact_pct),
            "quote_timestamp": self.quote_timestamp,
            "quote_age_ms": to_float(self.quote_age_ms),
            "max_quote_age_ms": to_float(self.max_quote_age_ms),
            "fill_timestamp": self.fill_timestamp,
            "provider": self.provider,
            "venue": self.venue,
            "pool_or_curve": self.pool_or_curve,
            "route": list(self.route),
            "identity": self.identity,
            "note": self.note,
        }


@dataclass
class PaperPosition:
    """One open simulated position, for one arm."""

    signal_id: str
    arm: str
    mint: str
    symbol: str | None
    token_decimals: int
    #: Exactly the quoted output amount, in raw token units. NEVER recomputed
    #: from a mark and never rounded.
    tokens_raw: int
    entry_fill: SimulatedFill
    entry_observed_mark: float | None
    entry_mark_timestamp: float | None
    bankroll_before: float
    position_size_usd: float
    position_size_pct: float
    usd_notional: float
    sol_notional: Decimal
    sol_usd_reference: float
    sol_usd_reference_at: float
    opened_at: float
    entry_cost_quote_asset: Decimal
    trading_fees_quote: Decimal
    network_costs_quote: Decimal
    other_costs_quote: Decimal
    #: OBSERVED_MARK excursions. Named so they cannot be read as executable
    #: returns.
    observed_mark_mfe: float = 0.0
    observed_mark_mae: float = 0.0
    peak_mark: float | None = None
    trough_mark: float | None = None
    production_profile: str = "L25_CURRENT"
    #: THE TAKE-PROFIT THIS POSITION WAS OPENED WITH -- a snapshot, in percent.
    #:
    #: ``None`` means "no take-profit". It is deliberately NOT read from the
    #: global setting at exit time: an operator who changes the setting while a
    #: position is open must not silently rewrite the exit policy of something
    #: already in the book. The snapshot is taken once, at construction, and
    #: travels with the position through persistence and rehydration.
    #:
    #: A position opened before this field existed reads ``None`` here; see
    #: ``take_profit.for_position``, which falls back to the historical
    #: ``L25_TP100`` profile token so such a position does not silently lose the
    #: take-profit it was opened under.
    take_profit_pct: float | None = None
    # -- QUOTE-BASED MFE / MAE (execution-aware excursions) ----------------
    #
    # ``observed_mark_mfe/mae`` above are MARK excursions and say so. These are
    # the executable ones: computed from SELL quotes for this position's EXACT
    # held quantity, each with its own provenance.
    #
    # ``None`` means "not measured", and is NEVER rendered as 0.0. A zero MFE
    # is a real measurement -- the position never went up at all -- and a
    # missing MFE is the absence of evidence. Collapsing the two would report
    # unmeasured positions as ones that never moved.
    quote_mfe_pct: float | None = None
    quote_mae_pct: float | None = None
    #: How many SELL-quote observations contributed to the excursions above.
    quote_observation_count: int = 0
    #: "MARK_ONLY" when no usable SELL quote was ever obtained, so a reader can
    #: tell "never went up" from "was never measured".
    quote_mfe_mae_basis: str = "NOT_MEASURED"
    #: WHY no usable quote exists, when none does. ``None`` when the excursion IS
    #: measured. Stated rather than inferred, so an unmeasured position says why
    #: it is unmeasured instead of leaving a reader to guess.
    quote_unavailable_reason: str | None = None
    #: Holding ages (ms from ``opened_at``) at which the two extrema occurred.
    quote_mfe_elapsed_ms: float | None = None
    quote_mae_elapsed_ms: float | None = None
    #: Fraction of the SAMPLE at which a valid SELL quote existed, 0-100. A low
    #: figure is a data-quality statement, not a performance one, and it is
    #: shown so that an excursion covering 5% of the hold is not read as
    #: covering the whole of it.
    quote_coverage_pct: float | None = None
    #: Median seconds between usable quote observations, and the longest gap in
    #: ms. Together they bound what the SAMPLED excursion can have seen: an
    #: intrasecond spike between two observations 5s apart is not in either
    #: number, and saying so is the honest way to report them.
    quote_sampling_interval_s: float | None = None
    quote_max_sampling_gap_ms: float | None = None
    #: Which quote source produced each extreme, so the route is auditable.
    quote_mfe_source: str | None = None
    quote_mae_source: str | None = None
    #: The COST-AWARE vs TRAVEL basis the percentages were computed on. See
    #: :data:`autopaper.forensics.quote_excursions.EXCURSION_BASIS_PNL`.
    quote_excursion_cost_basis: str = ""
    # -- WHY THE EXCURSION IS ABSENT, AND WHAT THE SETTLEMENT PROVED ---------
    #
    #: The excursion is sampled, so "not measured" has several distinct causes
    #: and a reader must be able to tell them apart. These fields record WHICH,
    #: so the reporting layer explains the absence instead of guessing at it.
    #: Without them every absence collapses into one sentence, and a trade that
    #: was quoted and settled correctly looks like it was never quoted at all.
    #
    #: Counters, not booleans: one rejection on a long hold is noise, forty are
    #: a sampling outage, and the difference is the whole diagnosis.
    quote_rejection_count: int = 0
    #: The most recent rejection's gate, so the last failure names itself
    #: (identity / stale / wrong quantity / different market / no cost basis)
    #: instead of leaving a bare count.
    quote_last_rejection_reason: str | None = None
    #: True when the ACCEPTED settlement SELL quote -- the one the position
    #: actually sold on, for its exact held quantity, on a validated market --
    #: was folded into the excursion as a final observation. It is a genuine
    #: executable quote, so counting it is not an estimate; what it means is
    #: that the excursion is anchored at the real exit, not merely sampled.
    quote_settlement_sample: bool = False
    #: The settlement quote's own provenance: same-market identity verdict, the
    #: provider that served it, and its age at settlement. Enough to re-derive
    #: the identity decision that admitted the observation.
    quote_settlement_identity: str = ""
    quote_settlement_source: str = ""
    quote_settlement_age_ms: float | None = None

    # -- PRICE-TRIGGER REFERENCE (what the stop actually saw) --------------
    #
    # The stop is evaluated on the SAME-MARKET, SAME-CURRENCY execution-native
    # reference, never on the external aggregator mark. These fields make the
    # reference auditable after the fact: which reference decided, what it was
    # in its native unit, what it became in USD, and what the aggregator said
    # alongside it. ``observed_mark_role`` is recorded explicitly so the mark is
    # never mistaken for the authoritative trigger.
    trigger_reference_source: str = ""
    trigger_native_reference_raw: float | None = None
    trigger_native_reference_usd: float | None = None
    trigger_sol_usd_reference: float | None = None
    trigger_observed_mark_usd: float | None = None
    trigger_reference_identity: str = ""
    trigger_reference_market_id: str | None = None
    trigger_reference_age_ms: float | None = None
    trigger_unavailable_reason: str | None = None
    trigger_entry_fill_usd: float | None = None
    trigger_return_pct: float | None = None
    trigger_verdict: str = ""
    trigger_reason: str = ""
    observed_mark_role: str = "DIAGNOSTIC_ONLY"
    #: When the same-market trigger reference last became unevaluable. Feeds the
    #: bounded data-invalid policy so a fail-closed trigger cannot hold forever.
    native_reference_unavailable_since: float | None = None
    #: ENTRY INVARIANT result: was the entry fill same-market/same-unit as the
    #: native reference? Recorded so an incoherent basis is visible on the
    #: position instead of only in a log line.
    entry_basis_sane: bool = True
    entry_basis_detail: str = ""

    # -- ENTRY VENUE IDENTITY (pinned for the lifetime of the position) -----
    #
    # The exit MUST quote the market the entry was filled on. A SELL routed to
    # a different pool is a different trade with different depth, so it is
    # recorded as a counterfactual and never as the canonical settlement.
    # These fields are the pin, and they are persisted so the pin survives a
    # restart rather than being re-guessed from whatever is cheapest now.
    entry_venue_id: str = ""
    entry_pool_address: str | None = None
    entry_market_type: str = ""
    entry_quote_source: str = ""
    entry_quote_timestamp: float | None = None
    entry_quote_age_ms: float | None = None
    entry_quote_latency_ms: float | None = None

    # -- CACHED STATIC IDENTITY (for lightweight monitoring quotes) ----------
    #
    # Populated at entry from the executable quote. Contains static market
    # identity (curve/pool address, vault addresses, fee configs, decimals)
    # so monitoring ticks only need to read dynamic reserves.
    cached_identity: dict[str, Any] = None

    # -- CURRENT EXIT VENUE (migratable; the entry pin is NOT overwritten) ----
    #
    # ``entry_venue_id`` above records where the position was FILLED and is
    # immutable: it is the basis the exit PnL is measured against, and changing
    # it would change the meaning of every number on this position.
    #
    # ``exit_venue_id`` is where the exit will be QUOTED, and it may differ from
    # the entry venue for exactly one reason: a chain-verified Pump.fun
    # graduation moved this token's market from a bonding curve to a PumpSwap
    # pool. It is ``None`` while the position still trades on its entry venue,
    # so "no migration" is the default state and cannot be confused with one.
    #
    # Both are persisted, so a restart resumes the exit on the venue it was
    # already migrated to instead of reverting to a bonding curve that no longer
    # prices anything.
    exit_venue_id: str | None = None
    exit_venue_migration_reason: str | None = None
    exit_venue_migrated_at: float | None = None
    exit_venue_migration_evidence: str | None = None

    @property
    def effective_exit_venue_id(self) -> str:
        """
        The venue an exit quote should be taken on, right now.

        The entry venue unless a graduation has been proven and persisted. Every
        exit-quote call reads THIS, so a migrated position keeps quoting on the
        pool it migrated to instead of falling back to the dead curve.
        """
        return self.exit_venue_id or self.entry_venue_id

    @property
    def has_migrated_exit_venue(self) -> bool:
        return bool(
            self.exit_venue_id and self.exit_venue_id != self.entry_venue_id)

    # -- RENT LIFECYCLE ----------------------------------------------------
    #
    # A PAPER position buys a token account, and that account is rent-exempt
    # SOL that is LOCKED, not spent. Counting it as a permanent trading loss is
    # the error this tracks: it is recorded separately and is RECLAIMABLE once
    # the balance reaches zero. Canonical PnL never includes it.
    entry_rent_lamports: int = 0
    rent_locked: bool = True
    reclaimable_rent_lamports: int = 0
    reclaim_status: str = "RENT_LOCKED"
    reclaim_blocker: str = ""

    # -- V2 FLOOR (BETA) PER-ARM EXIT OVERRIDE --------------------------------
    #
    # The beta experiment runs three arms off ONE V2_FLOOR entry gate. The arms
    # differ ONLY in their exit parameters: ``beta_exit_bracket`` =
    # (take_profit_pct, stop_loss_pct) and ``beta_max_hold_s``. When BOTH are
    # set (never one without the other -- a lone bracket with the wrong hold
    # would be a different strategy wearing this arm's name), the V2 exit rule
    # uses them for THIS position and ignores the production profile's bracket.
    # ``None``/``None`` (every non-beta position) means "use the production
    # profile's", which is the pre-existing behaviour. Snapshotted at open and
    # persisted, so a restart or a later selection change cannot rewrite what
    # an open position was opened under -- the same invariant
    # ``production_profile`` and ``take_profit_pct`` already follow.
    beta_exit_bracket: tuple[float, float] | None = None
    beta_max_hold_s: float | None = None

    # -- OPEN-POSITION ECONOMICS / EXIT LIFECYCLE ---------------------------
    #
    # The current OBSERVED MARK travels on the position so an open book is
    # visible as cash + marked value rather than as "$0 PnL". It is updated
    # every monitor tick and persisted, so a restart restores the last mark
    # rather than inventing one. It is NEVER used to settle.
    current_mark_usd: float | None = None
    current_mark_timestamp: float | None = None
    #: Explicit exit lifecycle state. An exit that cannot be quoted is NOT a
    #: healthy open position; it is EXIT_TRIGGERED -> EXIT_PENDING_QUOTE ->
    #: UNSELLABLE (after the bounded retry budget is spent). CLOSED only on a
    #: real modelled fill.
    exit_status: str = "OPEN_MONITORING"
    exit_trigger_reason: str = ""
    last_exit_error: str = ""
    exit_attempts: int = 0

    # -- MONITORING DIAGNOSTICS (per-position, per-tick counters) -----------
    #: How many monitor ticks this position has been observed through.
    monitor_ticks: int = 0
    #: Ticks where the native reference was obtained and valid.
    monitor_ok_ticks: int = 0
    #: Ticks where the native reference was RPC-error blocked.
    monitor_rpc_error_ticks: int = 0
    #: Ticks where the native reference was quote-unavailable (market fact).
    monitor_quote_unavailable_ticks: int = 0
    #: Ticks where the native reference identity was unknown.
    monitor_identity_unknown_ticks: int = 0
    #: Ticks where the native reference identity was a mismatch.
    monitor_identity_mismatch_ticks: int = 0
    #: Ticks skipped due to telemetry budget exhaustion.
    monitor_skipped_budget_ticks: int = 0
    #: Ticks where the native reference was unevaluable (any cause).
    monitor_unavailable_ticks: int = 0

    # -- NATIVE (EVENT-DRIVEN) MARKET MONITORING --------------------------
    #
    # Where the freshest native price came from, and how old it is. These are
    # the fields that make the difference between "event-driven" and "claims to
    # be event-driven" checkable: a HEALTHY push with a growing
    # ``native_update_count`` and a small ``native_update_age_ms`` is the proof,
    # and a stale one is visible rather than silently substituted for.
    #: ``WEBSOCKET_PUSH`` when the price came from a pushed account update, or
    #: the recovery source when it did not. Never blank on an open position.
    monitor_source: str = ""
    websocket_connected: bool = False
    websocket_state: str = ""
    #: Age of the newest native update at the time of writing, in ms. -1 means
    #: no update has ever arrived, which is very different from "arrived long
    #: ago" and must not be reported as the same number.
    native_update_age_ms: float = -1.0
    #: The newest slot the stream has seen. Monotonic bookkeeping; a slot that
    #: moves backwards is counted as a stream gap rather than smoothed over.
    native_last_slot: int | None = None
    native_high_water_slot: int | None = None
    #: Cumulative pushes seen for this position's market, and the interval
    #: statistics of those pushes. These are what answer "how often is the
    #: market actually being observed".
    native_update_count: int = 0
    native_update_interval_median_ms: float = 0.0
    native_update_interval_p95_ms: float = 0.0
    native_update_interval_max_ms: float = 0.0
    stream_gap_count: int = 0
    reconnect_count: int = 0
    rate_limit_429_count: int = 0
    #: The native reference price, in quote asset per WHOLE token (SOL/token),
    #: exactly as the executable quote reports it. ``native_return_pct`` is that
    #: price against the entry fill, i.e. the number the exit policy sees.
    native_price: float | None = None
    native_return_pct: float | None = None
    #: RPC / executable-quote budget accounting. The whole point of the split is
    #: that a healthy push spends neither, so both counters are expected to sit
    #: at zero for most of a position's life.
    rpc_calls_last_minute: int = 0
    quote_requests_last_minute: int = 0
    #: Why the last executable quote was requested. Empty while the position is
    #: being monitored natively and no quote has been needed.
    quote_request_reason: str = ""
    quote_proximity_band: str = ""
    quote_proximity_trigger: str = ""
    quote_proximity_distance_pct: float | None = None
    #: A proven ``PUMP_BONDING_CURVE -> PUMPSWAP`` switch for THIS position.
    #: Recorded so the post-migration market is auditable rather than inferred.
    migration_from_venue_id: str = ""
    migration_to_venue_id: str = ""
    migration_at_monotonic: float = 0.0

    def monitor_diagnostics(self) -> dict[str, Any]:
        """Per-position monitoring health counters."""
        ticks = max(1, self.monitor_ticks)
        return {
            "monitor_ticks": self.monitor_ticks,
            "monitor_ok_ticks": self.monitor_ok_ticks,
            "monitor_rpc_error_ticks": self.monitor_rpc_error_ticks,
            "monitor_quote_unavailable_ticks": self.monitor_quote_unavailable_ticks,
            "monitor_identity_unknown_ticks": self.monitor_identity_unknown_ticks,
            "monitor_identity_mismatch_ticks": self.monitor_identity_mismatch_ticks,
            "monitor_skipped_budget_ticks": self.monitor_skipped_budget_ticks,
            "monitor_unavailable_ticks": self.monitor_unavailable_ticks,
            "usable_pct": round(100.0 * self.monitor_ok_ticks / ticks, 2),
            "skipped_pct": round(
                100.0 * self.monitor_skipped_budget_ticks / ticks, 2),
            # -- event-driven monitoring -------------------------------
            "monitor_source": self.monitor_source,
            "websocket_connected": self.websocket_connected,
            "websocket_state": self.websocket_state,
            "last_market_update_age_ms": round(self.native_update_age_ms, 1),
            "last_market_slot": self.native_last_slot,
            "native_price": to_float(self.native_price),
            "native_return_pct": (
                None if self.native_return_pct is None
                else round(self.native_return_pct, 4)),
            "native_update_count": self.native_update_count,
            "native_update_interval_median_ms": round(
                self.native_update_interval_median_ms, 1),
            "native_update_interval_p95_ms": round(
                self.native_update_interval_p95_ms, 1),
            "native_update_interval_max_ms": round(
                self.native_update_interval_max_ms, 1),
            "stream_gap_count": self.stream_gap_count,
            "reconnect_count": self.reconnect_count,
            "rate_limit_429_count": self.rate_limit_429_count,
            "rpc_calls_last_minute": self.rpc_calls_last_minute,
            "quote_requests_last_minute": self.quote_requests_last_minute,
            "quote_request_reason": self.quote_request_reason,
            "quote_proximity_band": self.quote_proximity_band,
            "quote_proximity_trigger": self.quote_proximity_trigger,
            "quote_proximity_distance_pct": (
                None if self.quote_proximity_distance_pct is None
                else round(self.quote_proximity_distance_pct, 4)),
            "migration_from_venue_id": self.migration_from_venue_id,
            "migration_to_venue_id": self.migration_to_venue_id,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "arm": self.arm,
            "mint": self.mint,
            "symbol": self.symbol,
            "token_decimals": self.token_decimals,
            "tokens_raw": self.tokens_raw,
            "tokens_decimal": self.tokens_raw / (10 ** self.token_decimals),
            "entry_fill": self.entry_fill.as_dict(),
            "entry_observed_mark": to_float(self.entry_observed_mark),
            "entry_mark_timestamp": self.entry_mark_timestamp,
            "bankroll_before": to_float(self.bankroll_before),
            "position_size_usd": to_float(self.position_size_usd),
            "position_size_pct": to_float(self.position_size_pct),
            "usd_notional": to_float(self.usd_notional),
            "sol_notional": to_float(self.sol_notional),
            "sol_usd_reference": to_float(self.sol_usd_reference),
            "sol_usd_reference_at": self.sol_usd_reference_at,
            "opened_at": self.opened_at,
            "entry_cost_quote_asset": to_float(self.entry_cost_quote_asset),
            "trading_fees_quote": to_float(self.trading_fees_quote),
            "network_costs_quote": to_float(self.network_costs_quote),
            "other_costs_quote": to_float(self.other_costs_quote),
"OBSERVED_MARK_MFE": to_float(self.observed_mark_mfe),
            "OBSERVED_MARK_mae": to_float(self.observed_mark_mae),
            # The excursion legs themselves. Persisting only the percentages would
            # leave the levels unrecoverable, and the levels are what an auditor
            # needs in order to confirm the percentages were measured against this
            # position's own observed marks rather than someone else's.
            "peak_mark": to_float(self.peak_mark),
            "trough_mark": to_float(self.trough_mark),
            "production_profile": self.production_profile,
            # The take-profit SNAPSHOT. Persisted so a rehydrated position
            # exits under the rule it was opened with, not under whatever the
            # global setting says at the moment of the restart.
            "take_profit_pct": to_float(self.take_profit_pct),
            # The BETA per-arm exit override. Persisted for the same reason:
            # a restart resumes the position under ITS OWN exit rule.
            "beta_exit_bracket": (
                None if self.beta_exit_bracket is None
                else [float(self.beta_exit_bracket[0]),
                      float(self.beta_exit_bracket[1])]
            ),
            "beta_max_hold_s": (
                None if self.beta_max_hold_s is None
                else float(self.beta_max_hold_s)
            ),
            # Execution-aware excursions. None stays None: absent evidence is
            # not a zero excursion.
            "quote_mfe_pct": to_float(self.quote_mfe_pct),
            "quote_mae_pct": to_float(self.quote_mae_pct),
            "quote_observation_count": int(self.quote_observation_count or 0),
            "quote_mfe_mae_basis": self.quote_mfe_mae_basis,
            "quote_unavailable_reason": self.quote_unavailable_reason,
            "quote_mfe_elapsed_ms": to_float(self.quote_mfe_elapsed_ms),
            "quote_mae_elapsed_ms": to_float(self.quote_mae_elapsed_ms),
            "quote_coverage_pct": to_float(self.quote_coverage_pct),
            "quote_sampling_interval_s": to_float(self.quote_sampling_interval_s),
            "quote_max_sampling_gap_ms": to_float(self.quote_max_sampling_gap_ms),
            "quote_mfe_source": self.quote_mfe_source,
            "quote_mae_source": self.quote_mae_source,
            "quote_excursion_cost_basis": self.quote_excursion_cost_basis,
            "quote_rejection_count": int(self.quote_rejection_count or 0),
            "quote_last_rejection_reason": self.quote_last_rejection_reason,
            "quote_settlement_sample": bool(self.quote_settlement_sample),
            "quote_settlement_identity": self.quote_settlement_identity,
            "quote_settlement_source": self.quote_settlement_source,
            "quote_settlement_age_ms": to_float(self.quote_settlement_age_ms),
            # -- PRICE-TRIGGER REFERENCE ----------------------------------
            # Persisted so a future reader can prove WHICH price decided the
            # exit and on what basis, instead of inferring it from a mark that
            # is only diagnostic.
            "trigger_reference_source": self.trigger_reference_source,
            "trigger_native_reference_raw": to_float(self.trigger_native_reference_raw),
            "trigger_native_reference_usd": to_float(self.trigger_native_reference_usd),
            "trigger_sol_usd_reference": to_float(self.trigger_sol_usd_reference),
            "trigger_observed_mark_usd": to_float(self.trigger_observed_mark_usd),
            "trigger_reference_identity": self.trigger_reference_identity,
            "trigger_reference_market_id": self.trigger_reference_market_id,
            "trigger_reference_age_ms": to_float(self.trigger_reference_age_ms),
            "trigger_unavailable_reason": self.trigger_unavailable_reason,
            "trigger_entry_fill_usd": to_float(self.trigger_entry_fill_usd),
            "trigger_return_pct": to_float(self.trigger_return_pct),
            "trigger_verdict": self.trigger_verdict,
            "trigger_reason": self.trigger_reason,
            "observed_mark_role": self.observed_mark_role,
            # -- EVENT-DRIVEN NATIVE MONITORING --------------------------
            # Persisted for the same reason as the trigger reference above: a
            # monitoring claim is only auditable if the numbers behind it
            # survive the process.
            **self.monitor_diagnostics(),
            # -- THE VENUE PIN ---------------------------------------------
            # Persisted because the EXIT needs it. An entry that recorded its
            # market in memory but not on disk is an entry whose exit has to
            # guess, and a guessed venue is a different trade.
            "entry_venue_id": self.entry_venue_id,
            "entry_pool_address": self.entry_pool_address,
            "entry_market_type": self.entry_market_type,
            "entry_quote_source": self.entry_quote_source,
            "entry_quote_timestamp": self.entry_quote_timestamp,
            "entry_quote_age_ms": self.entry_quote_age_ms,
            "entry_quote_latency_ms": self.entry_quote_latency_ms,
            # -- CURRENT EXIT VENUE ----------------------------------------
            # Persisted beside the entry pin, and deliberately separate from it:
            # the entry venue is where the position was filled and never
            # changes, while this is where the exit is now quoted.
            "exit_venue_id": self.exit_venue_id,
            "exit_venue_migration_reason": self.exit_venue_migration_reason,
            "exit_venue_migrated_at": self.exit_venue_migrated_at,
            "exit_venue_migration_evidence": self.exit_venue_migration_evidence,
            # -- RENT LIFECYCLE ---------------------------------------------
            "entry_rent_lamports": self.entry_rent_lamports,
            "rent_locked": self.rent_locked,
            "reclaimable_rent_lamports": self.reclaimable_rent_lamports,
            "reclaim_status": self.reclaim_status,
            "reclaim_blocker": self.reclaim_blocker,
            # -- OPEN-POSITION ECONOMICS / EXIT LIFECYCLE -----------------
            "current_mark_usd": to_float(self.current_mark_usd),
            "current_mark_timestamp": self.current_mark_timestamp,
            "exit_status": self.exit_status,
            "exit_trigger_reason": self.exit_trigger_reason,
            "last_exit_error": self.last_exit_error,
            "exit_attempts": self.exit_attempts,
        }


@dataclass(frozen=True)
class RealizedPnl:
    """
    ``REALIZED_PNL`` - from simulated settlement, never from two marks.

    Every component is itemised so the number can be taken apart rather than
    trusted:

        net = exit_proceeds - entry_cost - fees - modeled_costs

    Quote-asset PnL is the PRIMARY accounting number, because the entry and exit
    both happened in the quote asset. The USD figure is a conversion at an
    explicitly timestamped reference and is labelled as such; it is never
    derived from the token's own mark.
    """

    entry_cost_quote: Decimal
    exit_proceeds_quote: Decimal
    trading_fees_quote: Decimal
    network_costs_quote: Decimal
    other_costs_quote: Decimal
    gross_pnl_quote: Decimal
    net_pnl_quote: Decimal
    realized_pnl_usd: float | None
    realized_pnl_sol: float | None
    return_pct: float | None
    bankroll_before: float
    bankroll_after: float
    drawdown_pct: float | None
    hold_seconds: float
    sol_usd_reference: float | None
    sol_usd_reference_at: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer": "REALIZED_PNL",
            "basis": "simulated settlement (exit fill - entry fill - fees - costs)",
            "never": "exit_mark / entry_mark - 1",
            "simulated_entry_cost_quote_asset": to_float(self.entry_cost_quote),
            "simulated_exit_proceeds_quote_asset": to_float(self.exit_proceeds_quote),
            "simulated_trading_fees": to_float(self.trading_fees_quote),
            "simulated_network_costs": to_float(self.network_costs_quote),
            "simulated_other_costs": to_float(self.other_costs_quote),
            "gross_pnl_quote": to_float(self.gross_pnl_quote),
            "fees_quote": to_float(self.trading_fees_quote),
            "costs_quote": to_float(
                self.network_costs_quote + self.other_costs_quote
            ),
            "net_pnl_quote": to_float(self.net_pnl_quote),
            "realized_pnl_sol": to_float(self.realized_pnl_sol),
            "realized_pnl_usd": to_float(self.realized_pnl_usd),
            "return_pct": to_float(self.return_pct),
            "bankroll_before": to_float(self.bankroll_before),
            "bankroll_after": to_float(self.bankroll_after),
            "drawdown_pct": to_float(self.drawdown_pct),
            "hold_seconds": to_float(self.hold_seconds),
            "sol_usd_reference": to_float(self.sol_usd_reference),
            "sol_usd_reference_at": self.sol_usd_reference_at,
            "ACTUAL_FILL": None,
            "actual_fill_available": ACTUAL_FILL_AVAILABLE,
        }


@dataclass(frozen=True)
class SignalRecord:
    """
    One signal's outcome for one arm.

    Written for EVERY signal, including - especially - the rejected ones.
    Recording only trades would make it impossible to study whether execution
    quality is itself filtering out candidates, which is one of the main things
    this dataset exists to answer.
    """

    signal: SignalSnapshot
    arm: str
    arm_version: str
    baseline_hash: str
    l25_decision: str
    l25_status: str
    l25_authorized: bool
    arm_filter_reason: str
    observed_mark: ObservedMarkPoint
    buy_quote: QuotePoint
    #: THE GAP THE GUARD DECIDED ON: executable vs the same market's own
    #: on-chain price. ``None`` when the entry was refused before the
    #: comparison could be made.
    mark_to_quote_gap_pct: float | None
    execution_identity: str
    traded: bool
    failure: FailureReason
    failure_detail: str
    latency: LatencyMarks
    bankroll_before: float
    position_size_usd: float | None
    position_size_pct: float | None
    finished_at: float = 0.0
    #: THE LEGACY GAP: executable vs the external observed mark. Research only,
    #: never a decision input, and carried on the SAME record as the gap above
    #: so the two references can be compared without reconstructing history.
    legacy_observed_mark_gap_pct: float | None = None
    #: DIAGNOSTIC ONLY. What the execution-native guard WOULD have decided.
    execution_native_gap_pct: float | None = None
    native_guard_verdict: str = ""
    #: "LEGACY_DEXSCREENER_MARK" for every production decision. Persisted so a
    #: reader can never assume the native reference decided the trade.
    production_guard_source: str = ""
    #: DIAGNOSTIC verdict of the legacy DexScreener-mark guard, recorded so
    #: "what would the old guard have decided?" stays answerable per record.
    legacy_guard_verdict: str = ""
    #: WHY EACH BUY CANDIDATE SOURCE FAILED, in one clause per source
    #: (``jupiter=JUPITER_NO_ROUTE(...); pump_direct=OK; ...``). Populated for
    #: every quote-stage refusal, so a blocked BUY is never diagnosable only by
    #: its flat failure_reason. Empty when no quote stage ran.
    buy_quote_failure_cause: str = ""
    #: THE ECONOMIC VERDICT, verbatim, when the micro-account viability guard
    #: ran. ``EconomicCheck.as_dict()`` -- the planned notional, the estimated
    #: round-trip cost, the locked (recoverable) rent, the break-even return and
    #: the thresholds, each named.
    #:
    #: It is recorded because the reason STRING alone cannot be audited: it says
    #: the order was below the floor, not what the floor was, what the order
    #: actually was, or what the rent would have been against the same notional.
    #: That absence is what let one run report a single opaque
    #: ``MICRO_ACCOUNT_ECONOMICALLY_INVALID`` bucket for twelve different
    #: decisions without showing that the sizing bankroll had drifted to
    #: $0.76 against a $100 account. ``None`` when the guard did not run, which
    #: is NOT the same as ``{}``.
    economics: dict | None = None
    # -- ENTRY EXECUTION-GAP OBSERVABILITY (additive; never overwrites history) -
    #
    # These fields make the entry price quality answerable per record WITHOUT
    # touching historical rows. Each field is recorded only when the underlying
    # observation exists; when it does not, it is left absent rather than
    # fabricated. Missing values are therefore explicit, not hidden.
    #
    # The observed mark is DexScreener's view. The executable buy quote is the
    # actual quoted price for this size. The native reference is the SAME market's
    # own on-chain spot/marginal price. Those three are deliberately kept distinct,
    # and the two gaps below are computed only from values that actually exist.
    entry_executable_buy_price_usd: float | None = None
    entry_executable_quote_timestamp: float | None = None
    entry_executable_quote_age_ms: float | None = None
    entry_mark_to_executable_gap_pct: float | None = None
    entry_native_to_executable_gap_pct: float | None = None
    entry_native_reference_price_usd: float | None = None
    entry_native_reference_method: str | None = None
    entry_venue_id: str = ""
    entry_pool_address: str | None = None
    entry_observed_mark_timestamp: float | None = None
    entry_mark_age_ms: float | None = None
    entry_quote_age_ms: float | None = None
    entry_native_reference_available: bool = False
    #: THE V2 FLOOR (BETA) PARENT SIGNAL ID. ``None`` for every ordinary arm
    #: record. The three beta arms' records for one entry all carry the SAME
    #: id -- the experiment's stable ``channel:mint:telegram_timestamp`` hash --
    #: which is what lets a reader tie the three records to ONE economic entry
    #: and lets the engine find its visibility mirror for them. It is a
    #: reference, not a position: settlement still belongs to
    #: ``BetaExperiment`` alone.
    parent_signal_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        native = self.buy_quote.execution_native_reference
        return {
            "signal": self.signal.as_dict(),
            "strategy": self.arm,
            "strategy_version": self.arm_version,
            "baseline_hash": self.baseline_hash,
            "l25_decision": self.l25_decision,
            "l25_status": self.l25_status,
            "l25_authorized": self.l25_authorized,
            "arm_filter_reason": self.arm_filter_reason,
            # The external mark is RETAINED, in full, as a diagnostic -- so
            # "did the token move?" stays answerable separately from "was
            # execution expensive?".
            "observed_mark": self.observed_mark.as_dict(),
            "execution_native_reference": (
                native.as_dict() if native is not None else None),
            "executable_buy_quote": self.buy_quote.as_dict(),
            "mark_gap_pct": to_float(self.mark_to_quote_gap_pct),
            "mark_gap_reference": "execution_native_same_market",
            "legacy_observed_mark_gap_pct": to_float(
                self.legacy_observed_mark_gap_pct),
            "execution_native_gap_pct": to_float(
                self.execution_native_gap_pct),
            "native_guard_verdict": self.native_guard_verdict,
            "legacy_guard_verdict": self.legacy_guard_verdict,
            "production_guard_source": self.production_guard_source,
            "legacy_observed_mark_gap_reference": "external_aggregator_mark",
            "execution_identity": self.execution_identity,
            "traded": self.traded,
            "failure_reason": self.failure.value,
            "failure_detail": self.failure_detail,
            "buy_quote_failure_cause": self.buy_quote_failure_cause,
            # ``None`` when the economic guard never ran. Emitted as ``None`` and
            # not omitted, so "the guard did not run" is distinguishable from
            # "the guard ran and charged nothing".
            "economics": self.economics,
            "latency": self.latency.as_dict(),
            "bankroll_before": to_float(self.bankroll_before),
            "position_size_usd": to_float(self.position_size_usd),
            "position_size_pct": to_float(self.position_size_pct),
            "finished_at": self.finished_at,
            # -- ENTRY EXECUTION-GAP OBSERVABILITY (additive) ----------------
            # Recorded so entry price quality is answerable per record without
            # touching historical rows. Each field is emitted only when the
            # underlying observation exists; missing values are left absent rather
            # than fabricated, so "unavailable" is explicit, not inferred.
            "entry_executable_buy_price_usd": to_float(
                self.entry_executable_buy_price_usd),
            "entry_executable_quote_timestamp": self.entry_executable_quote_timestamp,
            "entry_executable_quote_age_ms": to_float(
                self.entry_executable_quote_age_ms),
            "entry_mark_to_executable_gap_pct": to_float(
                self.entry_mark_to_executable_gap_pct),
            "entry_native_to_executable_gap_pct": to_float(
                self.entry_native_to_executable_gap_pct),
            "entry_native_reference_price_usd": to_float(
                self.entry_native_reference_price_usd),
            "entry_native_reference_method": self.entry_native_reference_method,
            "entry_venue_id": self.entry_venue_id,
            "entry_pool_address": self.entry_pool_address,
            "entry_observed_mark_timestamp": self.entry_observed_mark_timestamp,
            "entry_mark_age_ms": to_float(self.entry_mark_age_ms),
            "entry_quote_age_ms": to_float(self.entry_quote_age_ms),
            "entry_native_reference_available": self.entry_native_reference_available,
            # The V2 FLOOR (BETA) parent signal, when this record belongs to
            # the experiment; ``None`` for every ordinary arm record.
            "parent_signal_id": self.parent_signal_id,
        }


# =============================================================================
# ENTRY EXECUTION-GAP OBSERVABILITY HELPER
# =============================================================================

def entry_execution_gap_observability(
    *,
    mark: Any,
    quote: Any,
    guard: Any,
) -> dict[str, Any]:
    """
    Additive entry execution-gap observability for one SIGNAL RECORD.

    These fields make the entry price-quality question answerable per record
    WITHOUT touching historical rows. Each is populated only from values that
    actually exist at construction time; when a value is unavailable it is left
    absent rather than fabricated, so "unavailable" is explicit.

    The observed mark is DexScreener's view. The executable buy quote is the
    actual quoted price for this size. The native reference is the SAME market's
    own on-chain spot/marginal price. Those three are kept deliberately distinct.

    The mark-to-executable gap mixes execution cost with aggregator lag and is
    a DIAGNOSTIC ONLY. The native-to-executable gap (the guard's number) is the
    production measure: fee plus size impact in this pool at this slot.
    """
    import time
    # Observed mark timestamp / age.
    mark_ts = getattr(mark, "timestamp", None)
    mark_age = getattr(mark, "age_ms", None)
    if mark_age is None and mark_ts is not None:
        mark_age = max(0.0, (time.time() - mark_ts) * 1000.0)

    # Executable buy price / quote timestamp / age.
    q_price = getattr(quote, "effective_price_usd", None)
    q_ts = getattr(quote, "quote_timestamp", None)
    q_age = getattr(quote, "quote_age_ms", None)

    # Execution-native reference.
    native = getattr(quote, "execution_native_reference", None)
    native_available = native is not None
    native_price_usd = None
    native_method = None
    if native is not None:
        native_method = getattr(native, "calculation_method", None)
        # Derive a USD figure for the reference from the quote's own USD path
        # where available, so the reference can be compared on the same USD
        # footing as the executable buy without inventing a conversion.
        ref_price_raw = getattr(native, "reference_price", None)
        if ref_price_raw is not None and q_price is not None:
            q_eff = getattr(quote, "effective_price", None)
            if q_eff is not None and q_eff != 0:
                try:
                    native_price_usd = float(ref_price_raw) / float(q_eff) * q_price
                except Exception:  # noqa: BLE001 - leave unavailable
                    native_price_usd = None

    # Gaps.
    mark_to_executable = None
    native_to_executable = None
    if q_price is not None and mark_ts is not None and getattr(mark, "value_usd", None) is not None:
        try:
            mark_to_executable = (q_price / float(mark.value_usd) - 1.0) * 100.0
        except Exception:  # noqa: BLE001
            mark_to_executable = None
    if native_price_usd is not None and q_price is not None and q_price != 0:
        try:
            native_to_executable = (q_price / native_price_usd - 1.0) * 100.0
        except Exception:  # noqa: BLE001
            native_to_executable = None

    return {
        "entry_observed_mark_timestamp": mark_ts,
        "entry_mark_age_ms": mark_age,
        "entry_native_reference_price_usd": native_price_usd,
        "entry_native_reference_method": native_method,
        "entry_native_reference_available": native_available,
        "entry_executable_buy_price_usd": q_price,
        "entry_executable_quote_timestamp": q_ts,
        "entry_executable_quote_age_ms": q_age,
        "entry_mark_to_executable_gap_pct": mark_to_executable,
        "entry_native_to_executable_gap_pct": native_to_executable,
        "entry_venue_id": getattr(quote, "venue_id", "") or "",
        "entry_pool_address": getattr(quote, "pool_or_curve", None),
    }


@dataclass(frozen=True)
class TradeRecord:
    """One completed simulated trade, for one arm."""

    position: PaperPosition
    exit_trigger: str
    exit_reason: str
    exit_observed_mark: float | None
    exit_mark_timestamp: float | None
    sell_quote: QuotePoint
    exit_fill: SimulatedFill | None
    pnl: RealizedPnl | None
    latency: LatencyMarks
    failure: FailureReason
    failure_detail: str
    #: PROVENANCE OF THIS EXIT, so a max-hold exit is never reported as a
    #: strategy exit and two profiles are never pooled into one aggregate.
    strategy_profile: str = ""
    #: The cap this position was OPENED under, in seconds. ``None`` = no cap.
    max_hold_seconds: float | None = None
    #: How long it actually held. Measured, never declared.
    hold_time_seconds: float | None = None
    #: SELL provenance. ``sell_execution_status`` names the OUTCOME
    #: ("SETTLED" / "NO_EXECUTABLE_SELL"), so a REQUESTED quote can never be
    #: mistaken for a fill.
    sell_quote_source: str = ""
    sell_market_id: str = ""
    sell_quote_price: float | None = None
    sell_quote_age_ms: float | None = None
    sell_execution_status: str = ""
    # -- WHY A SELL QUOTE FAILED, AND WHAT HAPPENS NEXT -------------------
    #
    # A stuck position is a resource problem, so the record has to carry the
    # resource facts. ``sell_quote_failure_cause`` names WHICH capacity condition
    # refused -- an upstream provider throttle and our own exhausted quota are
    # different incidents with different remedies -- and the retry fields make
    # "is this stuck, and when does it try again" answerable from the record
    # alone rather than by reading logs.
    sell_quote_failure_cause: str = ""
    sell_quote_retryable: bool = False
    sell_quote_attempts: int = 0
    sell_quote_next_retry_in_s: float | None = None
    #: True when the settlement quote came from this tick's memo rather than a
    #: fresh fetch. Recorded because it is a deliberate trade of a second RPC read
    #: for a single decision.
    sell_quote_reused_this_tick: bool = False
    #: MARK vs EXECUTABLE, kept separate and labelled (audit §12): the gap
    #: between the executable SELL quote's USD price and the observed mark at
    #: the exit attempt, in percent. NOT slippage -- a cross-reference
    #: diagnostic between two different kinds of number, never a fee.
    executable_vs_mark_gap_pct: float | None = None
    #: How old the observed mark was at the exit attempt, in milliseconds.
    mark_age_ms: float | None = None
    #: MARK-BASED excursion labels. These are computed from the observed mark
    #: (DexScreener), which is NOT an executable price. They are named with the
    #: MARK_ prefix so they can never be read as executable returns.
    observed_mark_mfe: float | None = None
    observed_mark_mae: float | None = None
    #: EXECUTABLE excursion labels. These are computed ONLY from executable
    #: observations (the entry fill vs the exit fill / executable sell quote).
    #: When no executable sell evidence exists, these are UNAVAILABLE and must be
    #: reported as such rather than invented from the mark.
    executable_mfe_pct: float | None = None
    executable_mae_pct: float | None = None
    executable_mfe_available: bool = False
    executable_mae_available: bool = False
    #: SAMPLED QUOTE EXCURSION -- the MFE/MAE of the sampled SELL-quote path
    #: across the hold, in percent of the position's all-in entry cost.
    #:
    #: ``None`` means NOT MEASURED and is never rendered as 0.0. ``mae_basis``
    #: distinguishes a measured ``QUOTE_BASED`` pair from a position that never
    #: obtained a usable same-market SELL quote, and ``unavailable_reason`` says
    #: why. This is a SAMPLED estimate from executable quotes, not a fill and
    #: not a guarantee, and the canonical realised PnL on this same record
    #: comes from ``realize()`` on the actual entry and exit fills.
    quote_mfe_pct: float | None = None
    quote_mae_pct: float | None = None
    quote_mfe_mae_basis: str = "NOT_MEASURED"
    quote_unavailable_reason: str | None = None
    quote_mfe_elapsed_ms: float | None = None
    quote_mae_elapsed_ms: float | None = None
    quote_coverage_pct: float | None = None
    quote_sampling_interval_s: float | None = None
    quote_max_sampling_gap_ms: float | None = None
    quote_mfe_source: str | None = None
    quote_mae_source: str | None = None
    quote_excursion_cost_basis: str = ""
    #: How many SELL-quote observations the excursion above is built from, and
    #: how many were rejected on their way in. ``None``/absent on a record
    #: written before this field existed, which is how a historical record is
    #: told apart from a live one that measured nothing.
    quote_observation_count: int = 0
    quote_rejection_count: int = 0
    quote_last_rejection_reason: str | None = None
    #: The ACCEPTED settlement quote, and whether it was itself folded into the
    #: excursion. A quote-based settlement ALWAYS has the former; the settlement
    #: invariant is what produces it, and nothing here relaxes it.
    quote_settlement_sample: bool = False
    quote_settlement_identity: str = ""
    quote_settlement_source: str = ""
    quote_settlement_age_ms: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "position": self.position.as_dict(),
            "exit_trigger": self.exit_trigger,
            "exit_reason": self.exit_reason,
            "observed_exit_mark": to_float(self.exit_observed_mark),
            "exit_mark_timestamp": self.exit_mark_timestamp,
            "executable_sell_quote": self.sell_quote.as_dict(),
            "simulated_sell_fill": (
                self.exit_fill.as_dict() if self.exit_fill else None
            ),
            "realized_pnl": self.pnl.as_dict() if self.pnl else None,
            "latency": self.latency.as_dict(),
            "failure_reason": self.failure.value,
            "failure_detail": self.failure_detail,
            "exit_reason_code": self.exit_trigger,
            # Per-profile attribution. Two profiles are NEVER pooled: a report
            # that cannot separate L25_CURRENT from L25_MAX_HOLD cannot answer
            # which one the max-hold cap actually helped.
            "strategy_profile": self.strategy_profile,
            "max_hold_seconds": to_float(self.max_hold_seconds),
            "hold_time_seconds": to_float(self.hold_time_seconds),
            "sell_quote_source": self.sell_quote_source,
            "sell_market_id": self.sell_market_id,
            "sell_quote_price": to_float(self.sell_quote_price),
            "sell_quote_age_ms": to_float(self.sell_quote_age_ms),
            "sell_execution_status": self.sell_execution_status,
            # Resource facts for a stuck exit: which capacity condition refused,
            # whether it is retryable, and when the next attempt is.
            "sell_quote_failure_cause": self.sell_quote_failure_cause,
            "sell_quote_retryable": self.sell_quote_retryable,
            "sell_quote_attempts": self.sell_quote_attempts,
            "sell_quote_next_retry_in_s": to_float(
                self.sell_quote_next_retry_in_s),
            "sell_quote_reused_this_tick": self.sell_quote_reused_this_tick,
            "executable_vs_mark_gap_pct": to_float(
                self.executable_vs_mark_gap_pct),
            "mark_age_ms": to_float(self.mark_age_ms),
            # -- MFE / MAE LABELING -----------------------------------------
            # MARK-based excursions are labelled MARK_ so they are never mistaken
            # for executable returns. EXECUTABLE excursions are emitted ONLY when
            # executable evidence exists; otherwise they are reported as
            # UNAVAILABLE rather than invented from the mark.
            "OBSERVED_MARK_MFE": to_float(self.observed_mark_mfe),
            "OBSERVED_MARK_MAE": to_float(self.observed_mark_mae),
            "EXECUTABLE_MFE": (
                to_float(self.executable_mfe_pct)
                if self.executable_mfe_available else None),
            "EXECUTABLE_MAE": (
                to_float(self.executable_mae_pct)
                if self.executable_mae_available else None),
            "EXECUTABLE_MFE_AVAILABLE": self.executable_mfe_available,
            "EXECUTABLE_MAE_AVAILABLE": self.executable_mae_available,
            # The SAMPLED QUOTE excursion. Emitted as None when unmeasured, so
            # "never went up" and "never measured" stay distinguishable in the
            # machine-readable artefact exactly as they are on screen.
            "QUOTE_MFE": to_float(self.quote_mfe_pct),
            "QUOTE_MAE": to_float(self.quote_mae_pct),
            "QUOTE_MFE_MAE_BASIS": self.quote_mfe_mae_basis,
            "QUOTE_MFE_UNAVAILABLE_REASON": self.quote_unavailable_reason,
            "QUOTE_MFE_ELAPSED_MS": to_float(self.quote_mfe_elapsed_ms),
            "QUOTE_MAE_ELAPSED_MS": to_float(self.quote_mae_elapsed_ms),
            "QUOTE_COVERAGE_PCT": to_float(self.quote_coverage_pct),
            "QUOTE_SAMPLING_INTERVAL_S": to_float(
                self.quote_sampling_interval_s),
            "QUOTE_MAX_SAMPLING_GAP_MS": to_float(
                self.quote_max_sampling_gap_ms),
            "QUOTE_MFE_SOURCE": self.quote_mfe_source,
            "QUOTE_MAE_SOURCE": self.quote_mae_source,
            "QUOTE_EXCURSION_COST_BASIS": self.quote_excursion_cost_basis,
            "QUOTE_OBSERVATION_COUNT": int(self.quote_observation_count or 0),
            "QUOTE_REJECTION_COUNT": int(self.quote_rejection_count or 0),
            "QUOTE_LAST_REJECTION_REASON": self.quote_last_rejection_reason,
            "QUOTE_SETTLEMENT_SAMPLED": bool(self.quote_settlement_sample),
            "QUOTE_SETTLEMENT_IDENTITY": self.quote_settlement_identity,
            "QUOTE_SETTLEMENT_SOURCE": self.quote_settlement_source,
            "QUOTE_SETTLEMENT_QUOTE_AGE_MS": to_float(self.quote_settlement_age_ms),
        }


__all__ = [
    "ACTUAL_FILL_AVAILABLE",
    "ARM_CLOSE_NO_FILL_CONTRADICTION",
    "ARM_CLOSE_SETTLEMENT_FAILED",
    "EXIT_VENUE_MIGRATION_GRADUATED",
    "SIMULATED_FILL_KIND",
    "VenueMigration",
    "FailureReason",
    "LatencyMarks",
    "ObservedMarkPoint",
    "PaperPosition",
    "PriceLayer",
    "QuotePoint",
    "RealizedPnl",
    "SignalRecord",
    "SignalSnapshot",
    "SimulatedFill",
    "TradeRecord",
]
