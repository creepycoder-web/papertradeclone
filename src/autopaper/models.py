"""
Domain model.

Two rules are enforced structurally here:

* ``AlertSnapshot`` is frozen. It is written once at T0 (alert received) and can
  never be mutated, so "never backfill future information into the alert
  snapshot" is a type-level guarantee, not a convention.
* Every ``Observation`` carries its own provenance and freshness. A decision
  function is given a list of observations and a decision time; it must filter
  itself to ``observed_at <= decision_time`` (enforced by
  ``ObservationSet.as_of``).
"""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence


# ---------------------------------------------------------------------------
# enums
# ---------------------------------------------------------------------------


class Decision(str, Enum):
    BUY = "BUY"
    WAIT = "WAIT"
    REJECT = "REJECT"


class Regime(str, Enum):
    VERY_EARLY = "VERY_EARLY"
    EARLY = "EARLY"
    ACCELERATING = "ACCELERATING"
    EXPLOSIVE = "EXPLOSIVE"
    EXTENDED = "EXTENDED"
    STABILIZING = "STABILIZING"
    WEAKENING = "WEAKENING"
    UNCERTAIN = "UNCERTAIN"


class Route(str, Enum):
    """
    The runtime's route vocabulary.

    V3-C is the only strategy, so there are only two possible outcomes of route
    classification: the candidate is a V3-C FAST_ENTRY setup, or it is not
    tradeable. The removed members (PRECISION_ENTRY, CONFIRMATION_ENTRY,
    RECOVERY_RECLAIM) are gone from the enum itself, not merely unused, so no
    code in this process can name one.

    Historical journal rows and research artifacts still carry the old route
    strings. They are read as opaque text, never parsed into this enum, so
    removing the members does not rewrite or invalidate a single historical
    record.
    """

    FAST_ENTRY = "FAST_ENTRY"
    NO_TRADE = "NO_TRADE"


#: Route names that existed before V3-C and may still appear in historical
#: journal rows. Listed so a reader can see they are preserved, not lost.
HISTORICAL_ROUTES: tuple[str, ...] = (
    "PRECISION_ENTRY",
    "CONFIRMATION_ENTRY",
    "RECOVERY_RECLAIM",
)


class ExitStyle(str, Enum):
    FAST_FAILURE = "FAST_FAILURE"
    MOMENTUM_FAILURE = "MOMENTUM_FAILURE"
    RUNNER = "RUNNER"
    TIME_DECAY = "TIME_DECAY"
    FIXED_TARGET = "FIXED_TARGET"
    DISASTER_CAP = "DISASTER_CAP"
    DATA_INVALID = "DATA_INVALID"
    STOP_COMMAND = "STOP_COMMAND"
    FLATTEN = "FLATTEN"
    #: The researched V2_FLOOR bracket: TP +30%, SL -30%, max hold 90s, and
    #: nothing else. Selected by the POSITION's strategy_profile rather than by
    #: the route, because it is a property of the strategy that was researched,
    #: not of the regime the router happened to assign. The L25 family never
    #: produces this style.
    V2_BRACKET = "V2_BRACKET"


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"
    UNFILLED = "UNFILLED"


class DataState(str, Enum):
    LIVE = "LIVE"
    STALE = "STALE"
    UNUSABLE = "UNUSABLE"
    MISSING = "MISSING"


class ExcursionState(str, Enum):
    """
    How much of the hold path the MFE/MAE pair actually covers.

    Added because ``mae_pct`` was the RETURN AT THE CURRENT MARK rather than the
    trough, which made a losing trade able to book a POSITIVE "maximum adverse
    excursion" (52 of 77 historical fills did). A number that cannot be wrong is
    not trustworthy, so the coverage of the number is now part of the record.

    ``COMPLETE``
        Every mark the runtime obtained while the position was open was folded into
        the extrema, and at least one observation beyond the entry exists.
    ``PARTIAL_DATA``
        The entry is a valid basis and at least one further observation exists, but
        some marks were unavailable, stale or missing. The extrema are real
        observations, and this says so rather than implying full coverage.
    ``UNAVAILABLE_HISTORICAL_DATA``
        No valid observation beyond the entry was ever recorded for this position
        -- typically a trade restored from a record written before the path was
        instrumented. The extrema must then be read as "at least this", never as
        the true maximum or minimum.
    """

    COMPLETE = "COMPLETE"
    PARTIAL_DATA = "PARTIAL_DATA"
    UNAVAILABLE_HISTORICAL_DATA = "UNAVAILABLE_HISTORICAL_DATA"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


# ---------------------------------------------------------------------------
# T0 alert snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlertSnapshot:
    """
    Immutable record of exactly what was known at T0.

    ``fields`` preserves the source's raw values together with provenance, e.g.
    ``{"liquidity_usd": {"value": 18400.0, "source": "dexscreener", "at": 1.7e9}}``.
    Nothing may be added after construction. If a value did not exist at T0 it
    is ``None`` here forever.
    """

    candidate_id: str
    source: str
    chat: str
    message_id: int
    source_timestamp: float | None
    received_timestamp: float
    raw_text: str
    mint: str | None = None
    symbol: str | None = None
    name: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, str] = field(default_factory=dict)
    parse_warnings: tuple[str, ...] = ()
    #: Canonical signal feed attribution; ``source`` remains the transport.
    signal_source: str = "MooDeng"

    def __post_init__(self) -> None:
        # The field/provenance maps are wrapped in read-only proxies so that
        # "never backfill future information into the alert snapshot" is
        # enforced by the type, not by discipline.
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def value(self, key: str) -> Any:
        entry = self.fields.get(key)
        if entry is None:
            return None
        if isinstance(entry, Mapping):
            return entry.get("value")
        return entry

    def num(self, key: str) -> float | None:
        return _num(self.value(key))

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "chat": self.chat,
            "message_id": self.message_id,
            "source_timestamp": self.source_timestamp,
            "received_timestamp": self.received_timestamp,
            "mint": self.mint,
            "symbol": self.symbol,
            "name": self.name,
            "raw_text": self.raw_text,
            "fields": _clean(dict(self.fields)),
            "provenance": dict(self.provenance),
            "parse_warnings": list(self.parse_warnings),
            "signal_source": self.signal_source,
        }


# ---------------------------------------------------------------------------
# market observations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """One market observation with full provenance and a freshness verdict."""

    mint: str
    observed_at: float           # when the venue says the data is from
    received_at: float           # when we got it
    price_usd: float | None = None
    price_native: float | None = None
    market_cap_usd: float | None = None
    fdv_usd: float | None = None
    liquidity_usd: float | None = None
    volume_5m_usd: float | None = None
    volume_1h_usd: float | None = None
    volume_24h_usd: float | None = None
    price_change_5m: float | None = None
    price_change_1h: float | None = None
    price_change_24h: float | None = None
    buys_5m: int | None = None
    sells_5m: int | None = None
    txns_5m: int | None = None
    txns_1h: int | None = None
    pair_created_at: float | None = None
    first_valid_price: float | None = None
    provider: str = "unknown"
    venue: str | None = None
    pair_address: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    # -- freshness ---------------------------------------------------------

    def age_ms(self, now: float) -> float:
        return max(0.0, (now - self.observed_at) * 1000.0)

    def freshness(self, now: float, stale_ms: float, unusable_ms: float) -> DataState:
        if self.price_usd is None or self.price_usd <= 0:
            return DataState.MISSING
        age = self.age_ms(now)
        if age > unusable_ms:
            return DataState.UNUSABLE
        if age > stale_ms:
            return DataState.STALE
        return DataState.LIVE

    def age_seconds(self, now: float) -> float:
        return max(0.0, now - self.observed_at)

    @property
    def token_age_s(self) -> float | None:
        if self.pair_created_at is None:
            return None
        return max(0.0, self.observed_at - self.pair_created_at)

    @property
    def buy_sell_ratio(self) -> float | None:
        if self.buys_5m is None or self.sells_5m is None or self.sells_5m == 0:
            return None
        return self.buys_5m / self.sells_5m

    def as_dict(self) -> dict[str, Any]:
        return {
            "mint": self.mint,
            "observed_at": self.observed_at,
            "received_at": self.received_at,
            "provider": self.provider,
            "venue": self.venue,
            "pair_address": self.pair_address,
            "price_usd": self.price_usd,
            "price_native": self.price_native,
            "market_cap_usd": self.market_cap_usd,
            "fdv_usd": self.fdv_usd,
            "liquidity_usd": self.liquidity_usd,
            "volume_5m_usd": self.volume_5m_usd,
            "volume_1h_usd": self.volume_1h_usd,
            "price_change_5m": self.price_change_5m,
            "price_change_1h": self.price_change_1h,
            "price_change_24h": self.price_change_24h,
            "buys_5m": self.buys_5m,
            "sells_5m": self.sells_5m,
            "txns_5m": self.txns_5m,
            "txns_1h": self.txns_1h,
            "pair_created_at": self.pair_created_at,
            "first_valid_price": self.first_valid_price,
            "extra": _clean(dict(self.extra)),
        }


class ObservationSet:
    """
    Point-in-time view over observations.

    ``as_of(t)`` is the ONLY way the agent reads market history, and it hard
    filters on ``observed_at <= t``. This is the mechanical no-lookahead gate.
    """

    __slots__ = ("_items",)

    def __init__(self, items: Iterable[Observation] = ()) -> None:
        self._items: list[Observation] = sorted(items, key=lambda o: o.observed_at)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def as_of(self, t: float) -> "ObservationSet":
        """Observations observed at or before ``t``. Nothing from the future."""
        return ObservationSet([o for o in self._items if o.observed_at <= t])

    def append(self, obs: Observation) -> None:
        self._items.append(obs)
        self._items.sort(key=lambda o: o.observed_at)

    def latest(self, t: float | None = None) -> Observation | None:
        pool = self._items if t is None else self.as_of(t)._items
        return pool[-1] if pool else None

    def last_n(self, n: int, t: float | None = None) -> list[Observation]:
        pool = self._items if t is None else self.as_of(t)._items
        return pool[-n:]

    def span_s(self) -> float | None:
        if len(self._items) < 2:
            return None
        return self._items[-1].observed_at - self._items[0].observed_at

    def prices(self, t: float | None = None) -> list[tuple[float, float]]:
        pool = self._items if t is None else self.as_of(t)._items
        return [
            (o.observed_at, o.price_usd)
            for o in pool
            if o.price_usd is not None and o.price_usd > 0
        ]

    def to_list(self) -> list[dict[str, Any]]:
        return [o.as_dict() for o in self._items]


# ---------------------------------------------------------------------------
# market data quality / failure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuoteResult:
    """Result of asking a provider for the current best market view."""

    observation: Observation | None
    error: str | None = None
    provider: str = "unknown"
    latency_ms: float = 0.0
    from_cache: bool = False
    rate_limited: bool = False
    retry_after_s: float | None = None

    @property
    def ok(self) -> bool:
        return self.observation is not None


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskFlag:
    code: str
    detail: str
    severity: str = "warn"  # info | warn | block

    def __str__(self) -> str:
        return f"{self.code}:{self.severity}"

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "detail": self.detail, "severity": self.severity}


@dataclass(frozen=True)
class DecisionReport:
    """
    The structured, machine-readable rationale for one candidate.

    ``confidence`` is a decision-strength score in [0, 1]. It is NOT a
    probability of profit and is never reported as one.
    """

    candidate_id: str
    mint: str | None
    symbol: str | None
    decision: Decision
    regime: Regime
    route: Route
    signal_quality: float
    confidence: float
    thesis: str
    risk_flags: tuple[RiskFlag, ...]
    entry_conditions: tuple[str, ...]
    invalidation_conditions: tuple[str, ...]
    exit_plan: Mapping[str, Any]
    preferred_exit_style: ExitStyle
    decided_at: float
    snapshot: AlertSnapshot | None = None
    features: Mapping[str, Any] = field(default_factory=dict)
    data_state: DataState = DataState.LIVE
    latency_ms: float = 0.0
    nar_used: bool = False
    route_multiplier: float = 1.0
    route_sample_size: int = 0
    #: The V3-C evaluation that authorized or refused this decision, carried
    #: verbatim so the journal, the Telegram alerts and any post-hoc reader all
    #: see the exact same gate result the decision was made on. V3-C is L25's
    #: delegated first leg, so this is populated on every L25 evaluation too.
    v3c: Any = None
    #: The L25 family's full evaluation. L25 = V3-C AND the liquidity/MC ratio
    #: gate, so under the L25 family this is the result the decision was made
    #: on; the ``v3c`` field above is its first leg, kept separately so a
    #: refusal can always name which gate was operative.
    #:
    #: Under a ``V2_FLOOR`` profile this is STILL EVALUATED and still populated,
    #: so a record can answer "would L25 have taken this?" -- which is what a
    #: control A/B needs. It is simply not the authorizing gate there.
    l25: Any = None
    #: The V2_FLOOR gate's evaluation: vol5/mcap >= 1.0 AND 5m change in
    #: [+10%, +300%]. Populated on EVERY decision for the same reason ``l25``
    #: is: always evaluated, authorizing only under the V2_FLOOR profile.
    v2floor: Any = None
    #: The profile the decision was made under, recorded verbatim so a stored
    #: record never has to infer which gate authorized it.
    strategy_profile: str = "L25_CURRENT"

    @property
    def blocking_flags(self) -> tuple[RiskFlag, ...]:
        return tuple(f for f in self.risk_flags if f.severity == "block")

    @property
    def v3c_authorized(self) -> bool:
        return bool(self.v3c is not None and getattr(self.v3c, "authorized", False))

    @property
    def v3c_status(self) -> str:
        return str(getattr(self.v3c, "status", "NOT_EVALUATED"))

    @property
    def selected_strategy_result(self) -> Any:
        """
        The evaluation of the gate this decision was ACTUALLY made on.

        Resolved through the recorded ``strategy_profile``, so it cannot report
        authorization from a gate that was not selected.

        The previous implementation returned True if EITHER ``l25`` or ``v3c``
        authorized. That disagreed with ``DecisionEngine._decide``, which
        requires ``l25.authorized`` specifically -- so a report could say
        "authorized" for a decision that was in fact refused. With one gate that
        was latent; with two selectable gates it would have become reachable.
        """
        from .strategy_profile import (
            PROFILE_ENTRY_STRATEGY,
            StrategyProfileError,
            parse_profile,
        )

        try:
            profile = parse_profile(self.strategy_profile)
        except StrategyProfileError:
            # An unrecognised persisted profile must not authorize anything.
            return None
        if PROFILE_ENTRY_STRATEGY[profile] == "V2_FLOOR":
            return self.v2floor
        return self.l25

    @property
    def strategy_authorized(self) -> bool:
        """Authorization by the SELECTED strategy's gate, and only that gate."""
        result = self.selected_strategy_result
        if result is not None and hasattr(result, "authorized"):
            return bool(result.authorized)
        return False

    @property
    def strategy_status(self) -> str:
        result = self.selected_strategy_result
        if result is not None and hasattr(result, "status"):
            return str(result.status)
        return "NOT_EVALUATED"

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "candidate_id": self.candidate_id,
            "mint": self.mint,
            "symbol": self.symbol,
            "regime": self.regime.value,
            "route": self.route.value,
            "signal_quality": round(self.signal_quality, 4),
            "confidence": round(self.confidence, 4),
            "confidence_meaning": "decision_strength_score_not_probability_of_profit",
            "thesis": self.thesis,
            "risk_flags": [f.as_dict() for f in self.risk_flags],
            "entry_conditions": list(self.entry_conditions),
            "invalidation_conditions": list(self.invalidation_conditions),
            "exit_plan": _clean(dict(self.exit_plan)),
            "preferred_exit_style": self.preferred_exit_style.value,
            "decided_at": self.decided_at,
            "data_state": self.data_state.value,
            "latency_ms": round(self.latency_ms, 2),
            "nar_used": self.nar_used,
            "route_memory_multiplier": round(self.route_multiplier, 4),
            "route_memory_sample_size": self.route_sample_size,
            "v3c": (
                dict(self.v3c.as_dict()) if hasattr(self.v3c, "as_dict") else None
            ),
            "v3c_authorized": self.v3c_authorized,
            "l25": (
                dict(self.l25.as_dict()) if hasattr(self.l25, "as_dict") else None
            ),
            "strategy_profile": self.strategy_profile,
            "strategy_authorized": self.strategy_authorized,
            "strategy_status": self.strategy_status,
            "features": _clean(dict(self.features)),
        }

    def to_json_dict(self) -> dict[str, Any]:
        d = self.as_dict()
        d["exit_plan"] = _clean(dict(self.exit_plan))
        return d


# ---------------------------------------------------------------------------
# NAR (narrative) augmentation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NarSnapshot:
    """
    NAR-sourced context. Every field defaults to ``None``.

    The agent never fabricates buy counts, sell counts, holder counts, wallet
    flow, liquidity, security flags, narrative or social activity. If the
    configured NAR source cannot supply a value it stays ``None`` and the
    decision engine treats it as missing evidence.
    """

    mint: str
    fetched_at: float
    source: str | None = None
    available: bool = False
    narrative: str | None = None
    narrative_score: float | None = None
    social_activity: float | None = None
    holder_count: int | None = None
    top10_holders_pct: float | None = None
    dev_holdings_pct: float | None = None
    bundler_count: int | None = None
    sniper_count: int | None = None
    insider_count: int | None = None
    security_flags: tuple[str, ...] | None = None
    wallet_flow_net: float | None = None
    dex_paid: bool | None = None
    website: str | None = None
    twitter: str | None = None
    telegram: str | None = None
    error: str | None = None
    # --- fields a stats-card source genuinely reports ---------------------
    price_usd: float | None = None
    price_change_pct: float | None = None
    market_cap_usd: float | None = None
    volume_usd: float | None = None
    liquidity_usd: float | None = None
    supply: float | None = None
    change_1h_pct: float | None = None
    buys_1h: int | None = None
    sells_1h: int | None = None
    buy_sell_ratio_1h: float | None = None
    from_ath_pct: float | None = None
    ath_age_s: float | None = None
    age_s: float | None = None
    fresh_1d_pct: float | None = None
    fresh_7d_pct: float | None = None
    dev_token_count: int | None = None
    dev_large_wallets: int | None = None
    bundled_pct: float | None = None
    sniper_pct: float | None = None
    card_consistency: float | None = None
    token_name: str | None = None
    token_symbol: str | None = None

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict, fields

        out = {f.name: getattr(self, f.name) for f in fields(self)}
        if self.security_flags is not None:
            out["security_flags"] = list(self.security_flags)
        return {k: _clean(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# positions and trades
# ---------------------------------------------------------------------------


@dataclass
class Position:
    position_id: str
    candidate_id: str
    mint: str
    symbol: str
    status: PositionStatus
    route: Route
    regime: Regime
    exit_style: ExitStyle
    strategy: str
    entry_signal_timestamp: float
    entry_decision_timestamp: float
    entry_timestamp: float | None = None
    alert_received_timestamp: float | None = None
    quoted_price: float | None = None
    # `entry_reference_price` is the pre-slippage market reference at the moment
    # of the fill; `fill_price` is the price actually executed, i.e. the
    # reference moved against us by `entry_slippage_usd`. P&L is computed from
    # the references and slippage is then deducted, so the required identity
    #   net = gross - entry_fee - exit_fee - entry_slippage - exit_slippage
    # is exact and slippage is never counted twice.
    entry_reference_price: float | None = None
    fill_price: float | None = None
# -- THE THREE PRICE LAYERS AT ENTRY ------------------------------
    # PAPER settles nothing, so the layer that exists here is a SIMULATED_FILL.
    # It is built on an OBSERVED_MARK plus a formulaic slippage haircut, and it is
    # recorded next to (a) the mark it was built on and (b) the venue's own
    # EXECUTABLE_QUOTE, so a reader can never mistake one for another. A
    # missing layer is ``None`` WITH A REASON, never ``0.0``.
    fill_price_kind: str = "SIMULATED_FILL"
    #: WHICH KIND OF NUMBER each side of this position is priced from, as one of
    #: the five :class:`autopaper.pricing.PriceProvenance` labels. The entry label
    #: is stamped from the fill result; the exit label is stamped when the exit
    #: reference is chosen. ``MARK_ONLY`` on the exit means the position will be
    #: closed against an indexer print, and the closed trade will say so.
    entry_price_provenance: str = "MARK_ONLY"
    exit_price_provenance: str = "MARK_ONLY"
    observed_mark_price: float | None = None
    observed_mark_timestamp: float | None = None
    executable_quote_price: float | None = None
    executable_quote_price_usd: float | None = None
    executable_quote_kind: str | None = None
    executable_quote_at: float | None = None
    executable_quote_provider: str | None = None
    executable_quote_venue: str | None = None
    executable_quote_pool: str | None = None
    executable_quote_missing_reason: str = ""
    #: The execution-native reference price (native marginal/spot from the same
    #: market snapshot as the executable quote). This is the curve/pool's own
    #: current price before this order's impact is applied.
    native_reference_price: float | None = None
    native_reference_price_usd: float | None = None
    native_reference_timestamp: float | None = None
    #: Which leg supplied the reference price: EXECUTABLE_QUOTE, MARK_DUPLICATE
    #: (the historical inert case) or MARK_ONLY.
    quote_price_source: str = "MARK_ONLY"
    quote_guard_effective: bool = True
    #: Entry execution-gap observability (computed at fill time)
    mark_to_executable_gap_pct: float | None = None
    native_to_executable_gap_pct: float | None = None
    mark_age_ms: float | None = None
    quote_age_ms: float | None = None
    quantity: float = 0.0
    notional_usd: float = 0.0
    entry_fee_usd: float = 0.0
    entry_slippage_usd: float = 0.0
    cash_out_usd: float = 0.0
    cost_basis_usd: float = 0.0
    cash_at_entry: float | None = None   # pre-entry cash, for the invariant
    current_price: float | None = None
    mark_price: float | None = None
    market_value_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0
    current_return_pct: float = 0.0
    # -- EXCURSIONS ---------------------------------------------------------
    # ``mfe_pct`` is the return at ``peak_price``; ``mae_pct`` is the return at
    # ``trough_price``. Both are relative to ``excursion_basis_price``, which is
    # the ENTRY FILL -- the price actually paid -- not the quoted mark.
    #
    # They are seeded from the entry fill, so a position with a valid entry and at
    # least one subsequent observation satisfies ``mfe_pct >= 0 >= mae_pct`` by
    # construction. Neither is ever ``0`` merely because data was missing: when no
    # observation exists at all the state says so and the value means "at least".
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    peak_price: float | None = None
    trough_price: float | None = None
    peak_return_pct: float = 0.0
    trough_return_pct: float = 0.0
    #: The price every excursion percentage is measured against.
    excursion_basis_price: float | None = None
    #: ENTRY_FILL today. Recorded so a reader can tell what "return" means without
    #: inferring it from the field name.
    excursion_basis: str = "ENTRY_FILL"
    #: How much of the hold path the extrema cover. See :class:`ExcursionState`.
    excursion_state: str = ExcursionState.UNAVAILABLE_HISTORICAL_DATA.value
    #: Marks folded into the extrema, excluding the entry itself.
    excursion_observations: int = 0
    #: Marks that were offered and rejected (unusable, non-positive, or not LIVE).
    excursion_observations_rejected: int = 0
    #: MFE/MAE calculated from OBSERVED MARKS only (not executable prices).
    #: These are the historical "mark-based" excursions.
    mark_mfe_pct: float = 0.0
    mark_mae_pct: float = 0.0
    mark_peak_price: float | None = None
    mark_trough_price: float | None = None
    #: MFE/MAE calculated from EXECUTABLE PRICES (native reference or quote).
    #: Only populated when executable observations are available.
    executable_mfe_pct: float | None = None
    executable_mae_pct: float | None = None
    executable_peak_price: float | None = None
    executable_trough_price: float | None = None
    exit_timestamp: float | None = None
    exit_price: float | None = None           # pre-slippage reference
    exit_fill_price: float | None = None      # executed
    exit_fee_usd: float = 0.0
    exit_slippage_usd: float = 0.0
    cash_in_usd: float = 0.0
    realized_pnl_usd: float = 0.0
    return_pct: float = 0.0
    holding_seconds: float = 0.0
    exit_reason: str | None = None
    exit_style_used: ExitStyle | None = None
    thesis: str = ""
    invalidation: tuple[str, ...] = ()
    entry_conditions: tuple[str, ...] = ()
    exit_plan: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    signal_quality: float = 0.0
    runner_mode: bool = False
    runner_armed_at: float | None = None
    data_state: DataState = DataState.LIVE
    last_observed_at: float | None = None
    exit_hold_count: int = 0
    error: str | None = None
    notes: tuple[str, ...] = ()
    # context captured at entry, used for the by-bucket statistics
    liquidity_at_entry: float | None = None
    market_cap_at_entry: float | None = None
    token_age_s_at_entry: float | None = None
    alert_lag_ms: float | None = None
    analysis_latency_ms: float | None = None
    sizing_note: str = ""
    #: The SELECTABLE strategy profile this position was OPENED under. Copied
    #: from the runtime's selection at open time and never re-read afterwards,
    #: which is what makes an open position immutable: switching the global
    #: profile later cannot change how this position exits.
    #:
    #: Declared in the defaulted block because it has a default, and defaults
    #: must follow the required fields. A position restored from a snapshot
    #: written before profiles existed takes the default, which is the
    #: current profile -- i.e. no cap, i.e. exactly the old behaviour.
    strategy_profile: str = "L25_CURRENT"
    #: Token decimals, read from chain at entry time. Used for native reference
    #: queries during hold-path monitoring. Never assumed.
    token_decimals: int | None = None
    signal_source: str = "MooDeng"

    def unrealized_at(self, price: float | None) -> float:
        if price is None or price <= 0 or self.fill_price is None:
            return 0.0
        return (price - self.fill_price) * self.quantity

    def return_at(self, price: float | None) -> float | None:
        if price is None or price <= 0 or not self.fill_price:
            return None
        return (price - self.fill_price) / self.fill_price

    def as_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class Trade:
    """A completed paper trade. Appended to the ledger exactly once."""

    trade_id: str
    position_id: str
    candidate_id: str
    mint: str
    symbol: str
    source: str
    strategy: str
    route: str
    regime: str
    entry_signal_timestamp: float
    entry_timestamp: float
    entry_price: float
    fill_price: float
    exit_timestamp: float
    exit_price: float
    quantity: float
    entry_fee_usd: float
    exit_fee_usd: float
    entry_slippage_usd: float
    exit_slippage_usd: float
    gross_pnl_usd: float
    net_pnl_usd: float
    return_pct: float
    holding_seconds: float
    mfe_pct: float
    mae_pct: float
    exit_reason: str
    exit_style: str
    thesis: str
    invalidation: tuple[str, ...]
    liquidity_at_entry: float | None = None
    market_cap_at_entry: float | None = None
    age_bucket: str | None = None
    mc_bucket: str | None = None
    liq_bucket: str | None = None
    token_age_s_at_entry: float | None = None
    runner_mode: bool = False
    confidence: float = 0.0
    signal_quality: float = 0.0
    booked_at: float = 0.0
    #: The profile the position was opened under, carried onto the completed
    #: trade so per-trade statistics and reports can identify it without
    #: re-deriving it from the position record.
    strategy_profile: str = "L25_CURRENT"
    signal_source: str = "MooDeng"

    # -- PRICE PROVENANCE, ON THE COMPLETED TRADE --------------------------
    # A closed trade must never require a reader to guess whether its prices were
    # marks, quotes or executions. These are the five-value
    # :class:`autopaper.pricing.PriceProvenance` labels, carried per side.
    entry_price_provenance: str = "MARK_ONLY"
    exit_price_provenance: str = "MARK_ONLY"
    #: ``FILL_VS_FILL`` when ``net_pnl_usd`` is derived from the entry and exit
    #: execution prices. Anything else names what was used instead, and the
    #: disagreement against the fill basis is recorded in
    #: ``pnl_basis_disagreement_usd``.
    pnl_basis: str = "FILL_VS_FILL"
    #: gross PnL implied by the pre-slippage reference prices. Carried so the
    #: fill-based figure can be checked against it rather than recomputed.
    gross_pnl_reference_basis_usd: float | None = None
    #: |fill-basis gross - reference-basis gross|. Zero for every trade whose
    #: recorded slippage equals the price difference it implies.
    pnl_basis_disagreement_usd: float | None = None
    #: The excursion coverage of ``mfe_pct`` / ``mae_pct`` on this trade.
    excursion_state: str = ExcursionState.UNAVAILABLE_HISTORICAL_DATA.value
    excursion_observations: int = 0
    excursion_basis: str = "ENTRY_FILL"
    excursion_basis_price: float | None = None
    peak_price: float | None = None
    trough_price: float | None = None
    #: The lowest mark the runtime observed during the hold, and the highest. These
    #: are what a forensic reconciliation compares a booked exit against, so they
    #: have to be on the record rather than reconstructable only from a separate
    #: telemetry file.
    last_observed_mark: float | None = None
    booked_exit_within_observed_band: bool | None = None
    #: Entry execution-gap observability (copied from position at close)
    entry_observed_mark_price: float | None = None
    entry_observed_mark_timestamp: float | None = None
    entry_executable_quote_price: float | None = None
    entry_executable_quote_price_usd: float | None = None
    entry_executable_quote_kind: str | None = None
    entry_executable_quote_at: float | None = None
    entry_executable_quote_provider: str | None = None
    entry_executable_quote_venue: str | None = None
    entry_executable_quote_pool: str | None = None
    entry_executable_quote_missing_reason: str = ""
    entry_native_reference_price: float | None = None
    entry_native_reference_price_usd: float | None = None
    entry_native_reference_timestamp: float | None = None
    entry_mark_to_executable_gap_pct: float | None = None
    entry_native_to_executable_gap_pct: float | None = None
    entry_mark_age_ms: float | None = None
    entry_quote_age_ms: float | None = None
    #: MFE/MAE labeling distinction
    mark_mfe_pct: float | None = None
    mark_mae_pct: float | None = None
    executable_mfe_pct: float | None = None
    executable_mae_pct: float | None = None

    @property
    def is_win(self) -> bool:
        return self.net_pnl_usd > 0

    def as_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


# ---------------------------------------------------------------------------
# fill result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FillResult:
    filled: bool
    reason: str
    quoted_price: float | None
    fill_price: float | None
    slippage_bps: float
    slippage_usd: float
    fee_bps: float
    fee_usd: float
    latency_ms: float
    placed_at: float
    fill_at: float | None
    observed_at: float | None
    price_at_quote: float | None
    data_state: DataState
    notional_usd: float = 0.0
    quantity: float = 0.0
    liquidity_usd: float | None = None
    #: HOW ``liquidity_usd`` was obtained: ``OBSERVED`` when the venue's pool
    #: was actually read, ``UNKNOWN`` when it was not. ``UNKNOWN`` is never
    #: replaced by an assumed pool size; it changes how the fill is priced and
    #: it is surfaced so a reader can discount the number accordingly.
    liquidity_source: str = "UNKNOWN"
    #: One of ``PAPER/EXECUTION_AWARE`` (a venue quote for this size priced the
    #: fill), ``PAPER/MARK_BASED`` (indexer mark plus a modelled haircut, with
    #: liquidity observed), or ``PAPER/LOW_FIDELITY_NO_LIQUIDITY`` (mark, and the
    #: venue's liquidity could not be read at all).
    execution_fidelity: str = "UNKNOWN"
    detail: str = ""
    #: WHAT KIND OF PRICE produced ``fill_price``. A PAPER fill is always
    #: ``OBSERVED_MARK``: it is a third-party indexer price plus a formulaic
    #: slippage haircut, NOT a quote from any venue for this order's size. The
    #: label is carried on the result so a downstream report cannot present a
    #: simulated mark fill as if it were an executable price. See
    #: :mod:`autopaper.pricing`.
    price_kind: str = "OBSERVED_MARK"
    #: True when an INDEPENDENT EXECUTABLE_QUOTE took part in the "worse of"
    #: selection. Recorded rather than hidden: on 2026-10-01 this guard was
    #: silently inert for every PAPER exit, because the "quote" leg was the same
    #: DexScreener mark the observation came from.
    quote_guard_effective: bool = True
    # -- THE THREE PRICE LAYERS, SEPARATE FIELDS -------------------------
    #: 1. OBSERVED_MARK: the indexer price the fill model saw.
    observed_mark_price: float | None = None
    #: 2. EXECUTABLE_QUOTE: the venue's price for this size. ``None`` means
    #: "not established" -- never zero, and never the mark standing in for it.
    executable_quote_price: float | None = None
    executable_quote_price_usd: float | None = None
    executable_quote_kind: str | None = None
    executable_quote_at: float | None = None
    executable_quote_provider: str | None = None
    executable_quote_venue: str | None = None
    executable_quote_pool: str | None = None
    #: WHY the executable quote is absent. Required whenever it is: a missing
    #: layer has to be legible, not invisible.
    executable_quote_missing_reason: str = ""
    #: Which leg supplied the reference price: ``EXECUTABLE_QUOTE``,
    #: ``MARK_DUPLICATE`` (the historical inert case), ``MARK_ONLY``, or
    #: ``MARK_ONLY_QUOTE_NOT_CONVERTIBLE`` (a quote existed but carried no
    #: USD conversion, so comparing it against a USD mark would mix units).
    quote_price_source: str = "MARK_ONLY"
    #: WHICH KIND OF NUMBER produced ``price_at_quote``. One of
    #: :class:`autopaper.pricing.PriceProvenance`. ``MARK_ONLY`` here means the
    #: reference is an indexer print and is NOT an execution.
    reference_price_provenance: str = "MARK_ONLY"
    #: WHICH KIND OF NUMBER produced ``fill_price``. A PAPER fill is always
    #: ``PAPER_FILL``: it is derived, never signed, and must never be recorded
    #: anywhere as ``ONCHAIN_FILL``.
    fill_price_provenance: str = "PAPER_FILL"
    #: Token decimals for the position's mint. Used for native reference queries
    #: during hold-path monitoring. Read from the executable quote or chain.
    token_decimals: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


def with_enum_values(obj: Any) -> Any:
    return _clean(obj)
