"""
V2-A: the frozen decision-time market-cap ceiling.

V2 = the existing FAST_ENTRY strategy + this rule, and nothing else.

    IF decision_time_market_cap_usd > 25,372.71:  REJECT
    ELSE:                                         preserve the V1 decision

PROVENANCE - this is a research result, not a parameter
---------------------------------------------------
The value is the 2/3 tercile of ``f_market_cap_usd`` measured on the V2-A
discovery window only, and it was re-derived and asserted to 1e-6 against the
frozen inputs::

    assert abs(fit["threshold_usd"] - CEILING) < 1e-6

The durable record of the result is the written validation output under
``reports/``, not this comment: the V2-A focused validation classified the
result PARTIALLY_REPLICATED, and the paired comparison put V2 ahead of V1 on
17 of 17 metrics. This module deliberately does NOT import or reference the
research tree; live code must never reach back into it, and the isolation test
enforces that.

SCOPE - deliberately FAST_ENTRY only
-----------------------------------
The research predicate is scoped to FAST_ENTRY
(the frozen V2-A rejection predicate)::

    if trade.get("route") != "FAST_ENTRY":  return False

so any other route is NOT_APPLICABLE. That matters in practice: the historical
book is 178 FAST_ENTRY and 10 PRECISION_ENTRY trades, and V2-A was validated
without touching the latter. This module reproduces that scope exactly. It does
not disable, reweight or otherwise alter any other route.

A MISSING market cap is UNTESTABLE, never an automatic pass. That mirrors the
research predicate returning ``None`` rather than ``False``, and it is why this
module does not raise on ``market_cap_usd=None``: the caller already records
``mcap_unknown`` and V2-A simply declines to act on a value it cannot read.

BOUNDARY
---------
Inclusive: a market cap of exactly ``25,372.71`` PASSES. ``25,372.72`` is
REJECTED. There is no epsilon and no rounding.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

#: The frozen V2-A decision-time market-cap ceiling, in USD. DO NOT RETUNE.
V2A_CEILING_USD: float = 25_372.71

#: Rule identity. Changing any of these changes the researched result.
V2A_ID: str = "V2A_MARKET_CAP_CEILING"
V2A_VERSION: str = "v1"
V2A_ROUTE: str = "FAST_ENTRY"
V2A_INPUTS: tuple[str, ...] = ("route", "decision_time_market_cap_usd")

#: A market cap that cannot be read is UNTESTABLE, and untestable is not a pass.
V2A_UNTESTABLE = "UNTESTABLE"


def _definition_payload() -> str:
    return "|".join([
        V2A_ID, V2A_VERSION, V2A_ROUTE,
        repr(V2A_CEILING_USD), ",".join(V2A_INPUTS),
    ])


#: SHA-256 of the definition. A silent edit to the ceiling changes this.
V2A_HASH: str = hashlib.sha256(_definition_payload().encode("utf-8")).hexdigest()


def verify_ceiling(ceiling: float = V2A_CEILING_USD) -> None:
    """Raise unless ``ceiling`` is the frozen researched value."""
    if float(ceiling) != V2A_CEILING_USD:
        raise ValueError(
            f"V2A_CEILING_USD is FROZEN at {V2A_CEILING_USD:,.2f} USD "
            f"(frozen V2-A discovery-window fit, hash prefix "
            f"{V2A_HASH[:8]}). Refusing to run with "
            f"{float(ceiling):,.2f}. "
            "Retuning the ceiling invalidates the V2-A validation and is not a "
            "code decision."
        )


@dataclass(frozen=True)
class V2AResult:
    """Outcome of applying V2-A to one route decision."""

    #: True only when the rule did not reject AND did not decline to judge.
    passes: bool
    #: WITHIN_CEILING | REJECTED_OVER_CEILING | UNTESTABLE | NOT_APPLICABLE
    status: str
    market_cap_usd: Optional[float]
    ceiling_usd: float
    detail: str

    @property
    def rejects(self) -> bool:
        return self.status == "REJECTED_OVER_CEILING"

    def as_dict(self) -> dict:
        return {
            "passes": self.passes,
            "status": self.status,
            "rejects": self.rejects,
            "market_cap_usd": self.market_cap_usd,
            "ceiling_usd": self.ceiling_usd,
            "detail": self.detail,
        }


def apply_v2a(
    route: str,
    market_cap_usd: Optional[float],
    ceiling: float = V2A_CEILING_USD,
) -> V2AResult:
    """Apply the frozen V2-A market-cap ceiling to one route decision.

    A faithful port of the frozen V2-A rejection predicate.

    * route is not FAST_ENTRY      -> NOT_APPLICABLE (rule is scoped)
    * market cap missing           -> UNTESTABLE, and NOT a pass
    * market cap >  ceiling        -> REJECTED_OVER_CEILING
    * market cap <= ceiling        -> passes
    """
    verify_ceiling(ceiling)

    if route != V2A_ROUTE:
        return V2AResult(
            True, "NOT_APPLICABLE", market_cap_usd, ceiling,
            f"V2-A is scoped to {V2A_ROUTE}; route was {route!r}",
        )

    if market_cap_usd is None:
        return V2AResult(
            False, V2A_UNTESTABLE, None, ceiling,
            "no decision-time market cap: V2-A is untestable, never auto-passed",
        )

    mcap = float(market_cap_usd)
    if mcap > ceiling:
        return V2AResult(
            False, "REJECTED_OVER_CEILING", mcap, ceiling,
            f"decision-time market cap ${mcap:,.2f} above the frozen V2-A ceiling "
            f"${ceiling:,.2f}",
        )
    return V2AResult(
        True, "WITHIN_CEILING", mcap, ceiling,
        f"decision-time market cap ${mcap:,.2f} at or below the frozen V2-A "
        f"ceiling ${ceiling:,.2f}",
    )
