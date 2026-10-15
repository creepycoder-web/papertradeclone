"""
Typed vocabulary for the same-market quote harness.

THE THREE VALUES THIS EXPERIMENT KEEPS SEPARATE
================================================

The whole point of the harness is that these are NOT interchangeable, and that
the report can never let a reader confuse them:

``OBSERVED_MARK``
    What DexScreener says the price is. An observation about a venue. Never an
    execution price, never a PnL source.

``EXECUTABLE_QUOTE``
    A read-only price calculated from the exact market/pool/curve that an
    execution would use, for the exact test order size. What you *could* get.
    Still not a fill.

``ACTUAL_FILL``
    What actually happened on chain. In this harness there is none, ever: no
    transaction is built, signed or broadcast. :data:`ACTUAL_FILL_AVAILABLE` is
    a hard ``False`` and the status ``NOT_AVAILABLE`` is not a failure state -
    it is the correct state for a read-only experiment.

Everything that can go wrong is a named member of :class:`FailureMode`. There is
no generic "error" bucket and, critically, no failure mode that degrades to
zero: an absent observation is ``None`` plus a named reason, never ``0.0``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

#: There is no fill in this harness. Stated as a constant so the report cannot
#: drift into implying one exists.
ACTUAL_FILL_AVAILABLE: bool = False

#: Solana's native wrapped-SOL mint, used as the default BUY input asset.
WSOL_MINT = "So11111111111111111111111111111111111111112"
#: SPL token program, used to tell a real mint from a PDA.
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
SOL_DECIMALS = 9


class FailureMode(str, Enum):
    """
    Every way this measurement can fail, named.

    These are carried through to the report verbatim. None of them is ever
    converted into a zero price, a zero latency, or a PASS.
    """

    TOKEN_NOT_FOUND = "TOKEN_NOT_FOUND"
    MULTIPLE_TOKEN_MATCHES = "MULTIPLE_TOKEN_MATCHES"
    TOKEN_EXTRACTION_AMBIGUOUS = "TOKEN_EXTRACTION_AMBIGUOUS"
    PAIR_NOT_FOUND = "PAIR_NOT_FOUND"
    PAIR_AMBIGUOUS = "PAIR_AMBIGUOUS"
    TOKEN_MINT_MISMATCH = "TOKEN_MINT_MISMATCH"
    QUOTE_MINT_MISMATCH = "QUOTE_MINT_MISMATCH"
    VENUE_MISMATCH = "VENUE_MISMATCH"
    POOL_MISMATCH = "POOL_MISMATCH"
    STATE_UNAVAILABLE = "STATE_UNAVAILABLE"
    CURVE_UNAVAILABLE = "CURVE_UNAVAILABLE"
    QUOTE_UNAVAILABLE = "QUOTE_UNAVAILABLE"
    QUOTE_STALE = "QUOTE_STALE"
    MARK_STALE = "MARK_STALE"
    STATE_STALE = "STATE_STALE"
    RPC_RATE_LIMIT = "RPC_RATE_LIMIT"
    PROVIDER_RATE_LIMIT = "PROVIDER_RATE_LIMIT"
    DECIMAL_ERROR = "DECIMAL_ERROR"
    QUOTE_DIRECTION_ERROR = "QUOTE_DIRECTION_ERROR"
    MARKET_IDENTITY_UNKNOWN = "MARKET_IDENTITY_UNKNOWN"


class Decision(str, Enum):
    """
    Verdict on a single signal.

    ``INFORMATIVE`` is a first-class outcome, not a soft PASS. It is used when
    the gap was measured successfully but no threshold was configured, because
    inventing a threshold to manufacture a verdict is the exact behaviour this
    harness exists to catch.
    """

    PASS = "PASS"
    REJECT = "REJECT"
    INFORMATIVE = "INFORMATIVE"


class MarketKind(str, Enum):
    """Which execution market the quote was taken against."""

    PUMP_BONDING_CURVE = "PUMP_BONDING_CURVE"
    PUMPSWAP = "PUMPSWAP"
    JUPITER_ROUTED = "JUPITER_ROUTED"
    UNKNOWN = "UNKNOWN"


class Freshness(str, Enum):
    """Freshness verdict for one external observation."""

    FRESH = "FRESH"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"
    MISSING = "MISSING"


class CheckState(str, Enum):
    """Tri-state for an identity check. ``UNKNOWN`` is never treated as PASS."""

    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


def _clean(value: Any) -> Any:
    """
    Make a value safe for JSON.

    Non-finite floats are the classic trap: ``json.dumps`` emits bare ``NaN``,
    which is not valid JSON and which a strict consumer will reject. A gap
    computed from a zero denominator is ``inf``; it is reported as ``None`` with
    a named failure mode instead, and this is the last line of defence.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


@dataclass(frozen=True)
class FreshnessStamp:
    """
    One external observation with an explicit age.

    ``age_ms`` is measured against a caller-supplied ``now`` (wall clock), and
    the verdict comes from a caller-configured window. An unconfigured window
    yields :attr:`Freshness.UNKNOWN`, never a silent ``FRESH``.
    """

    label: str
    timestamp: float | None
    age_ms: float | None
    freshness: Freshness
    max_age_ms: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "timestamp": _clean(self.timestamp),
            "age_ms": _clean(self.age_ms),
            "freshness": self.freshness.value,
            "max_age_ms": _clean(self.max_age_ms),
        }


@dataclass(frozen=True)
class StalenessPolicy:
    """
    Configurable freshness windows, in milliseconds.

    ``None`` means "the operator did not configure this". That is reported as
    ``UNKNOWN`` freshness, which is deliberately NOT treated as fresh: the
    harness would rather say it does not know than say "fine".
    """

    max_mark_age_ms: float | None = None
    max_state_age_ms: float | None = None
    max_quote_age_ms: float | None = None

    def stamp(
        self,
        label: str,
        timestamp: float | None,
        now: float,
        window_ms: float | None,
    ) -> FreshnessStamp:
        if timestamp is None:
            return FreshnessStamp(label, None, None, Freshness.MISSING, window_ms)
        age = max(0.0, (now - timestamp) * 1000.0)
        if window_ms is None:
            return FreshnessStamp(label, timestamp, age, Freshness.UNKNOWN, None)
        fresh = age <= window_ms
        return FreshnessStamp(
            label,
            timestamp,
            age,
            Freshness.FRESH if fresh else Freshness.STALE,
            window_ms,
        )


@dataclass(frozen=True)
class TokenCandidate:
    """
    Result of token extraction from one Telegram message.

    ``ambiguous`` is a first-class outcome. The production parser resolves a
    multi-mint message by picking the most frequent base58 string; a
    measurement harness must not, because a wrong token means the whole price
    comparison describes the wrong asset. So multiple distinct mints produce
    ``ambiguous=True`` and the pipeline stops.
    """

    mint: str | None
    symbol: str | None
    ambiguous: bool
    candidates: tuple[str, ...]
    venue_hint: str | None
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "mint": self.mint,
            "symbol": self.symbol,
            "ambiguous": self.ambiguous,
            "candidates": list(self.candidates),
            "venue_hint": self.venue_hint,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class DexPair:
    """
    One DexScreener pair, reduced to the fields the identity checks need.

    Kept deliberately close to the wire shape. A harness that reshapes its
    input into a summary cannot later be used to check which field it trusted.
    """

    chain_id: str
    dex_id: str | None
    pair_address: str
    base_token_address: str | None
    base_token_symbol: str | None
    quote_token_address: str | None
    quote_token_symbol: str | None
    price_usd: float | None
    price_native: float | None
    liquidity_usd: float | None
    volume_24h_usd: float | None
    fdv_usd: float | None
    market_cap_usd: float | None
    pair_created_at: float | None
    txns_5m: int | None
    txns_1h: int | None
    buys_5m: int | None
    sells_5m: int | None
    buys_1h: int | None
    sells_1h: int | None
    url: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: _clean(v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class ObservedMark:
    """
    ``OBSERVED_MARK`` - DexScreener's view. An observation, never a fill.

    Carries the pair it came from, because the mark is only meaningful
    relative to the pool it describes.
    """

    value_usd: float | None
    pair_address: str | None
    dex_id: str | None
    source: str
    timestamp: float
    latency_ms: float | None
    from_cache: bool = False
    request_latency_ms: float | None = None
    failure: FailureMode | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "value_usd": _clean(self.value_usd),
            "pair_address": self.pair_address,
            "dex_id": self.dex_id,
            "source": self.source,
            "timestamp": _clean(self.timestamp),
            "latency_ms": _clean(self.latency_ms),
            "request_latency_ms": _clean(self.request_latency_ms),
            "from_cache": self.from_cache,
            "failure": self.failure.value if self.failure else None,
        }


@dataclass(frozen=True)
class ExecutableQuote:
    """
    ``EXECUTABLE_QUOTE`` - a read-only price for the exact test order size.

    Built from the market an execution would actually use. Not a fill, and not
    a promise: the venue can move before anything is submitted.
    """

    input_mint: str
    output_mint: str
    input_amount_raw: str
    output_amount_raw: str
    input_decimals: int
    output_decimals: int
    input_amount: float
    output_amount: float
    effective_price: float | None
    effective_price_native: float | None
    price_impact_pct: float | None = None
    fee_bps: int | None = None
    fee_amount: float | None = None
    venue: str | None = None
    pool_address: str | None = None
    route: tuple[str, ...] = ()
    provider: str = "unknown"
    state_timestamp: float | None = None
    quote_timestamp: float = 0.0
    quote_latency_ms: float | None = None
    failure: FailureMode | None = None
    detail: str | None = None
    #: The same price expressed in USD, using an explicitly logged SOL
    #: reference.
    #:
    #: THIS FIELD IS WHAT GETS COMPARED TO THE MARK. DexScreener's ``priceUsd``
    #: is USD per token; an executable quote taken with a SOL-denominated input
    #: is SOL per token. Comparing them directly produces a gap of a few
    #: hundred percent that reads like an extreme market rather than a unit
    #: error - and it is silent, because both numbers are non-negative and
    #: finite. ``effective_price`` is kept alongside so the SOL figure remains
    #: visible; neither is ever silently substituted for the other.
    effective_price_usd: float | None = None
    #: The SOL price used for that conversion. Logged next to the result so the
    #: comparison is reproducible.
    usd_reference_price: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_mint": self.input_mint,
            "output_mint": self.output_mint,
            "input_amount_raw": self.input_amount_raw,
            "output_amount_raw": self.output_amount_raw,
            "input_decimals": self.input_decimals,
            "output_decimals": self.output_decimals,
            "input_amount": _clean(self.input_amount),
            "output_amount": _clean(self.output_amount),
            "effective_price": _clean(self.effective_price),
            "effective_price_native": _clean(self.effective_price_native),
            "effective_price_usd": _clean(self.effective_price_usd),
            "usd_reference_price": _clean(self.usd_reference_price),
            "price_impact_pct": _clean(self.price_impact_pct),
            "fee_bps": self.fee_bps,
            "fee_amount": _clean(self.fee_amount),
            "venue": self.venue,
            "pool_address": self.pool_address,
            "route": list(self.route),
            "provider": self.provider,
            "state_timestamp": _clean(self.state_timestamp),
            "quote_timestamp": _clean(self.quote_timestamp),
            "quote_latency_ms": _clean(self.quote_latency_ms),
            "failure": self.failure.value if self.failure else None,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class IdentityChecks:
    """
    Pair-identity verification, reported check by check.

    ``POOL_MATCH`` is the load-bearing one. A DexScreener pair existing does
    not make it executable, and two venues can price the same token differently
    enough that the comparison is meaningless. :attr:`pool_match` stays
    ``UNKNOWN`` whenever the executable path could not be shown to be the same
    pool, and ``UNKNOWN`` can never be upgraded to ``PASS``.
    """

    token_mint_match: CheckState
    quote_mint_match: CheckState
    chain_match: CheckState
    venue_match: CheckState
    pool_match: CheckState
    execution_can_identify_market: bool | None
    notes: tuple[str, ...] = ()

    @property
    def market_match(self) -> Decision:
        """
        The aggregate. Any FAIL is REJECT; any UNKNOWN is UNKNOWN, which is
        reported but is not a valid PASS.
        """
        states = (
            self.token_mint_match,
            self.quote_mint_match,
            self.chain_match,
            self.venue_match,
            self.pool_match,
        )
        if any(s is CheckState.FAIL for s in states):
            return Decision.REJECT
        if any(s is CheckState.UNKNOWN for s in states):
            return Decision.REJECT
        return Decision.PASS

    @property
    def unknown(self) -> bool:
        return self.market_match is Decision.REJECT and any(
            s is CheckState.UNKNOWN for s in (
                self.token_mint_match,
                self.quote_mint_match,
                self.chain_match,
                self.venue_match,
                self.pool_match,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "TOKEN_MINT_MATCH": self.token_mint_match.value,
            "QUOTE_MINT_MATCH": self.quote_mint_match.value,
            "CHAIN_MATCH": self.chain_match.value,
            "VENUE_MATCH": self.venue_match.value,
            "POOL_MATCH": self.pool_match.value,
            "execution_can_identify_market": self.execution_can_identify_market,
            "MARKET_MATCH": self.market_match.value,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class PoolState:
    """
    Latest on-chain state of the market the quote is taken against.

    ``identity_verified`` exists because reading an account is not the same as
    knowing whose account it is. Bytes at an address are meaningless until they
    have been shown to belong to the token under test.
    """

    kind: MarketKind
    address: str | None
    program_id: str | None
    identity_verified: bool
    timestamp: float
    fetch_latency_ms: float | None
    detail: Mapping[str, Any] = field(default_factory=dict)
    failure: FailureMode | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "address": self.address,
            "program_id": self.program_id,
            "identity_verified": self.identity_verified,
            "timestamp": _clean(self.timestamp),
            "fetch_latency_ms": _clean(self.fetch_latency_ms),
            "detail": {k: _clean(v) for k, v in dict(self.detail).items()},
            "failure": self.failure.value if self.failure else None,
        }


@dataclass(frozen=True)
class SameMarketVerdict:
    """
    The comparison this harness exists to produce.

    ``gap_pct`` is signed and exact. ``abs_gap_pct`` is what the threshold is
    compared against, because the question "how far is the executable price
    from the mark" has no direction - a mark that is 74% ABOVE the executable
    price is just as untradeable as one that is 74% below it.
    """

    observed_mark: float | None
    executable_price: float | None
    gap_pct: float | None
    abs_gap_pct: float | None
    threshold_pct: float | None
    decision: Decision
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "observed_mark": _clean(self.observed_mark),
            "executable_price": _clean(self.executable_price),
            "gap_pct": _clean(self.gap_pct),
            "abs_gap_pct": _clean(self.abs_gap_pct),
            "threshold_pct": _clean(self.threshold_pct),
            "decision": self.decision.value,
            "reason": self.reason,
        }


__all__ = [
    "ACTUAL_FILL_AVAILABLE",
    "CheckState",
    "Decision",
    "DexPair",
    "ExecutableQuote",
    "FailureMode",
    "Freshness",
    "FreshnessStamp",
    "IdentityChecks",
    "MarketKind",
    "ObservedMark",
    "PoolState",
    "SOL_DECIMALS",
    "SameMarketVerdict",
    "StalenessPolicy",
    "TOKEN_2022_PROGRAM",
    "TOKEN_PROGRAM",
    "TokenCandidate",
    "WSOL_MINT",
]