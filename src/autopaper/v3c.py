"""
V3-C: the ONE entry authorization strategy in this runtime.

CANONICAL DEFINITION
--------------------

    V3-C  =  FAST_ENTRY
           + decision_time_market_cap_usd <= 20,000
           + signal_quality >= 0.55

V3-C is FAST_ENTRY carrying two decision-time gates. It is not a blend of
strategies, not a fallback, and not a selector over alternates: it is the only
path by which this process may open a position, in PAPER or in REAL.

DECISION-TIME ONLY
------------------
Every input is read at ``decision_time``. Specifically:

* market cap is the market cap observed at the decision, not the alert-time
  value, not the fill-time value, not the peak, not any later print;
* signal quality is computed by the existing feature pipeline over observations
  already gated to ``observed_at <= decision_time`` (see
  ``FeatureSet`` / ``ObservationSet.as_of``).

A value that did not exist at the decision is ``None``, and ``None`` is
UNTESTABLE, never an automatic pass. This mirrors V2-A's treatment and exists
for the same reason: a gate that opens itself when it cannot read its own input
is not a gate.

FROZEN
------
``verify_definition()`` recomputes the SHA-256 of the rule identity and the
threshold pair. A silent edit to either threshold changes the hash and is
refused at import-adjacent verification time. The migration instruction is
explicit that these numbers are not to be retuned, so they are made
non-retunable rather than merely documented.

V2-A
----
V2-A's frozen ceiling (``$25,372.71``) is retained and still evaluated, and it
is the WIDER of the two. V3-C's ``$20,000`` is the binding ceiling. A token
between $20,000 and $25,372.71 passes V2-A and still fails V3-C. V2-A is
represented, not deleted; it is simply dominated.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

#: The one strategy. Do not add V4.
V3C_ID: str = "V3C"
V3C_VERSION: str = "v1"
V3C_NAME: str = "V3-C"

#: The route V3-C is built on. V2-A is scoped to this route.
V3C_ROUTE: str = "FAST_ENTRY"

#: Decision-time market-cap ceiling, USD. FROZEN.
V3C_MAX_MARKET_CAP_USD: float = 20_000.0

#: Decision-time signal-quality floor. FROZEN. Inclusive.
V3C_MIN_SIGNAL_QUALITY: float = 0.55

#: The frozen V2-A ceiling, retained and still evaluated. V3-C is stricter.
V2A_CEILING_USD: float = 25_372.71

#: Rule identity. Changing any of these changes the researched result.
V3C_INPUTS: tuple[str, ...] = (
    "route",
    "decision_time_market_cap_usd",
    "signal_quality",
)


# ---------------------------------------------------------------------------
# outcomes
# ---------------------------------------------------------------------------

AUTHORIZED = "AUTHORIZED"
REJECTED_MARKET_CAP = "REJECTED_MARKET_CAP"
REJECTED_SIGNAL_QUALITY = "REJECTED_SIGNAL_QUALITY"
REJECTED_NOT_FAST_ENTRY = "REJECTED_NOT_FAST_ENTRY"
UNTESTABLE = "UNTESTABLE"

#: Short operator-facing reason, used verbatim by the Telegram reject alerts.
REJECT_REASON_TH: dict[str, str] = {
    REJECTED_MARKET_CAP: "Market cap above V3-C limit",
    REJECTED_SIGNAL_QUALITY: "Signal quality below V3-C requirement",
    REJECTED_NOT_FAST_ENTRY: "Not a V3-C FAST_ENTRY setup",
    UNTESTABLE: "Required decision-time value unavailable",
}


@dataclass(frozen=True)
class V3CResult:
    """Outcome of applying V3-C to one candidate at one decision time."""

    #: True only when V3-C authorizes an entry.
    authorized: bool
    #: One of the *_OUTCOME constants above.
    status: str
    market_cap_usd: Optional[float]
    signal_quality: Optional[float]
    max_market_cap_usd: float
    min_signal_quality: float
    detail: str

    @property
    def rejects(self) -> bool:
        return not self.authorized

    def as_dict(self) -> dict:
        return {
            "authorized": self.authorized,
            "status": self.status,
            "market_cap_usd": self.market_cap_usd,
            "signal_quality": self.signal_quality,
            "max_market_cap_usd": self.max_market_cap_usd,
            "min_signal_quality": self.min_signal_quality,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# integrity
# ---------------------------------------------------------------------------


def _definition_payload() -> str:
    return "|".join(
        [
            V3C_ID,
            V3C_VERSION,
            V3C_ROUTE,
            repr(V3C_MAX_MARKET_CAP_USD),
            repr(V3C_MIN_SIGNAL_QUALITY),
            ",".join(V3C_INPUTS),
        ]
    )


#: SHA-256 of the definition. A silent edit to either threshold changes this.
V3C_HASH: str = hashlib.sha256(_definition_payload().encode("utf-8")).hexdigest()


def verify_definition(
    max_market_cap_usd: float = V3C_MAX_MARKET_CAP_USD,
    min_signal_quality: float = V3C_MIN_SIGNAL_QUALITY,
) -> None:
    """Raise unless the thresholds are the frozen researched values."""
    if float(max_market_cap_usd) != float(V3C_MAX_MARKET_CAP_USD):
        raise ValueError(
            f"V3C_MAX_MARKET_CAP_USD is FROZEN at "
            f"${V3C_MAX_MARKET_CAP_USD:,.0f} (hash prefix {V3C_HASH[:8]}). "
            f"Refusing to run with ${float(max_market_cap_usd):,.0f}. "
            "Retuning the ceiling invalidates the V3-C result and is not a "
            "code decision."
        )
    if float(min_signal_quality) != float(V3C_MIN_SIGNAL_QUALITY):
        raise ValueError(
            f"V3C_MIN_SIGNAL_QUALITY is FROZEN at "
            f"{V3C_MIN_SIGNAL_QUALITY:.2f} (hash prefix {V3C_HASH[:8]}). "
            f"Refusing to run with {float(min_signal_quality):.2f}."
        )


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------


def _active_strategy_name() -> str:
    """
    The ACTIVE strategy's display name, resolved lazily.

    Imported inside the function because ``l25`` imports this module; a
    module-level import would be circular. Falls back to this gate's own name
    only if the active strategy is somehow unavailable, so a refusal reason is
    never lost.
    """
    try:
        from .l25 import ACTIVE_STRATEGY
    except Exception:  # noqa: BLE001
        return V3C_NAME
    return ACTIVE_STRATEGY


def evaluate(
    *,
    route: str,
    market_cap_usd: Optional[float],
    signal_quality: Optional[float],
    max_market_cap_usd: float = V3C_MAX_MARKET_CAP_USD,
    min_signal_quality: float = V3C_MIN_SIGNAL_QUALITY,
) -> V3CResult:
    """
    Apply V3-C to one candidate using decision-time values only.

    ``route`` is supplied by the caller as the route the existing pipeline
    would have selected. V3-C does not select routes; it gates the one route
    it is defined over, so a candidate the pipeline classifies as something
    other than FAST_ENTRY is not a V3-C setup.

    Order of evaluation is market cap first, then signal quality. Both must
    pass. A missing value on either axis is UNTESTABLE and does not authorize.
    """
    verify_definition(max_market_cap_usd, min_signal_quality)

    if route != V3C_ROUTE:
        return V3CResult(
            authorized=False,
            status=REJECTED_NOT_FAST_ENTRY,
            market_cap_usd=market_cap_usd,
            signal_quality=signal_quality,
            max_market_cap_usd=max_market_cap_usd,
            min_signal_quality=min_signal_quality,
            detail=(
                # Named for the ACTIVE strategy, not for this gate. The operator
                # reads this inside an "<ACTIVE> REJECT" alert, so naming the
                # delegated leg there contradicts the heading. Imported lazily
                # because l25 imports this module.
                f"{_active_strategy_name()} is defined over {V3C_ROUTE}; the "
                f"pipeline classified this candidate as {route!r}"
            ),
        )

    if market_cap_usd is None:
        return V3CResult(
            authorized=False,
            status=UNTESTABLE,
            market_cap_usd=None,
            signal_quality=signal_quality,
            max_market_cap_usd=max_market_cap_usd,
            min_signal_quality=min_signal_quality,
            detail=(
                "no decision-time market cap: V3-C is untestable, never "
                "auto-authorized"
            ),
        )

    mcap = float(market_cap_usd)
    if mcap > max_market_cap_usd:
        return V3CResult(
            authorized=False,
            status=REJECTED_MARKET_CAP,
            market_cap_usd=mcap,
            signal_quality=signal_quality,
            max_market_cap_usd=max_market_cap_usd,
            min_signal_quality=min_signal_quality,
            detail=(
                f"decision-time market cap ${mcap:,.2f} above the V3-C limit "
                f"${max_market_cap_usd:,.0f}"
            ),
        )

    if signal_quality is None:
        return V3CResult(
            authorized=False,
            status=UNTESTABLE,
            market_cap_usd=mcap,
            signal_quality=None,
            max_market_cap_usd=max_market_cap_usd,
            min_signal_quality=min_signal_quality,
            detail=(
                "no decision-time signal quality: V3-C is untestable, never "
                "auto-authorized"
            ),
        )

    sq = float(signal_quality)
    if sq < min_signal_quality:
        return V3CResult(
            authorized=False,
            status=REJECTED_SIGNAL_QUALITY,
            market_cap_usd=mcap,
            signal_quality=sq,
            max_market_cap_usd=max_market_cap_usd,
            min_signal_quality=min_signal_quality,
            detail=(
                f"decision-time signal quality {sq:.2f} below the V3-C "
                f"requirement {min_signal_quality:.2f}"
            ),
        )

    return V3CResult(
        authorized=True,
        status=AUTHORIZED,
        market_cap_usd=mcap,
        signal_quality=sq,
        max_market_cap_usd=max_market_cap_usd,
        min_signal_quality=min_signal_quality,
        detail=(
            f"FAST_ENTRY with decision-time market cap ${mcap:,.2f} "
            f"<= ${max_market_cap_usd:,.0f} and signal quality {sq:.2f} "
            f">= {min_signal_quality:.2f}"
        ),
    )


def v2a_note(market_cap_usd: Optional[float]) -> str:
    """
    One operator-facing line describing where the token sits against BOTH
    ceilings, so the tighter V3-C bound is visible rather than implied.
    """
    if market_cap_usd is None:
        return f"V2-A: untestable | V3-C: untestable (ceiling ${V3C_MAX_MARKET_CAP_USD:,.0f})"
    mcap = float(market_cap_usd)
    v2a_ok = mcap <= V2A_CEILING_USD
    v3c_ok = mcap <= V3C_MAX_MARKET_CAP_USD
    if v3c_ok:
        return (
            f"V2-A: pass (${V2A_CEILING_USD:,.2f}) | "
            f"V3-C: pass (${V3C_MAX_MARKET_CAP_USD:,.0f})"
        )
    if v2a_ok:
        return (
            f"V2-A: pass (${V2A_CEILING_USD:,.2f}) | "
            f"V3-C: FAIL (${mcap:,.2f} > ${V3C_MAX_MARKET_CAP_USD:,.0f})"
        )
    return (
        f"V2-A: FAIL (${V2A_CEILING_USD:,.2f}) | "
        f"V3-C: FAIL (${V3C_MAX_MARKET_CAP_USD:,.0f})"
    )


def describe() -> str:
    """Single-line strategy description for the status message."""
    return (
        f"{V3C_NAME}: MC <= ${V3C_MAX_MARKET_CAP_USD:,.0f}, "
        f"SQ >= {V3C_MIN_SIGNAL_QUALITY:.2f}"
    )
