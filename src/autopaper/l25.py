"""
L25: the ACTIVE entry strategy.

    L25 = V3-C
          AND f_liq_to_mcap >= 0.9647

    V3-C     = route == FAST_ENTRY
               AND decision_time_market_cap_usd <= 20000
               AND signal_quality >= 0.55
    f_liq_to_mcap = liquidity_usd / market_cap_usd

THIS IS NOT A NEW STRATEGY AND NOT A NEW FINDING
------------------------------------------------
L25 is the frozen Round-2 research candidate. It was derived, reported and
classified before this module existed. Nothing here was invented, tuned or
re-derived; the definition is transcribed from the repository's own research:

    research/randorder.py        L25_LIQ_MC = 0.9647, V3C_MC_MAX = 20000.0,
                                 V3C_SQ_MIN = 0.55, in_v3c(), in_l25()
    research/r2report.py         "V3-C AND f_liq_to_mcap >= 0.9647",
                                 derivation = "TRAIN-only quantile of
                                 f_liq_to_mcap over V3-C survivors"
    research/r2search.py         liqmc(): the ratio and its fail-closed rule
    reports/research/v3c_round2_search.{md,json}   the candidate ledger
    reports/research/v3c_random_order.md            the corrected spec universe

RESEARCH STATUS: PROMISING_BUT_INCONCLUSIVE
-------------------------------------------
`NO_TRUE_POST_DERIVATION_DATA` is untouched: every event in the research
universe predates the L25 derivation cutoff (epoch 1790735204). L25 has NOT been
validated on unseen data. Activating it in the runtime does not promote it, does
not make it "proven", and does not make it "universally better". The only valid
next step for the research question remains forward paper data past the cutoff.

The correction this implementation deliberately carries
-------------------------------------------------------
`research/v3c_random_order.md` §2 records that the Round-2 search implementation
omitted the `route == FAST_ENTRY` test and therefore admitted 5 PRECISION_ENTRY
events (V3-C 159 vs 154, L25 119 vs 116). This module INCLUDES the route test,
so it implements the SPEC, which is the smaller and correct population. The
counts 154 / 116 on the 276-event universe are a verification reference; they
are deliberately not hardcoded anywhere in this runtime.

FAIL-CLOSED, NEVER SUBSTITUTED
------------------------------
``f_liq_to_mcap`` is ``None`` - and the candidate is refused - when:

    market cap is missing
    market cap is zero or negative
    liquidity is missing
    liquidity is zero or negative

A missing liquidity value is NOT treated as zero, and a missing liquidity value
does NOT pass. It refuses. The same holds for a missing market cap. This mirrors
``research/r2search.py::liqmc``, which returns ``None`` unless both operands are
present and truthy.

V3-C IS DELEGATED TO, NOT DUPLICATED
------------------------------------
``L25Result.v3c`` holds the :class:`autopaper.v3c.V3CResult` produced by
``v3c.evaluate``. The V3-C gate has one implementation in this process. V3-C
remains available and is reported per-candidate, because when L25 refuses a
candidate the operator needs to know whether V3-C had also refused it (in which
case the market-cap or signal-quality gate is the real reason) or whether L25's
liquidity gate is the only thing standing in the way.

PROVENANCE
----------
V3-C thresholds and V2-A ceiling are owned by :mod:`autopaper.v3c` and
:mod:`autopaper.v2a` and are re-verified here at import time, so a drift in
either is caught rather than silently inherited.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional

from .v2a import V2A_CEILING_USD, verify_ceiling
from .v3c import (
    AUTHORIZED as V3C_AUTHORIZED,
    V3C_MAX_MARKET_CAP_USD,
    V3C_MIN_SIGNAL_QUALITY,
    V3C_NAME,
    V3CResult,
    evaluate as evaluate_v3c,
    verify_definition as verify_v3c_definition,
)

#: The active strategy's identity. The runtime, the logs, the Telegram UI and
#: the tests all read this, so there is exactly one name to change and no way
#: for a surface to advertise a different strategy than the one running.
ACTIVE_STRATEGY = "L25"

#: The frozen L25 name as the research spells it.
L25_ID = "L25"
L25_VERSION = "v1"

#: THE frozen threshold. From ``research/randorder.py::L25_LIQ_MC``.
#:
#: The Round-2 ledger records the underlying TRAIN-only quantile at full
#: precision as 0.9646640935338305. The operative value is the 4 dp form,
#: because that is the constant the reference implementation uses and the form
#: that reproduces the reference population. It is reproduced here verbatim and
#: is not re-derived, rounded, nudged or re-tuned.
L25_MIN_LIQ_TO_MCAP: float = 0.9647

#: The full-precision quantile, recorded for provenance only. NEVER used in a
#: comparison: using it would silently change the frozen population.
L25_DERIVED_QUANTILE_EXACT: float = 0.9646640935338305

#: Research classification, carried so no surface can imply more than is known.
L25_RESEARCH_CLASSIFICATION = "PROMISING_BUT_INCONCLUSIVE"
L25_POST_DERIVATION_VALIDATION = "NO_TRUE_POST_DERIVATION_DATA"
L25_DERIVATION_CUTOFF_EPOCH = 1790735204
L25_DERIVATION_METHOD = (
    "TRAIN-only quantile of f_liq_to_mcap over V3-C survivors"
)

#: Research sources this definition was transcribed from.
L25_RESEARCH_SOURCES: tuple[str, ...] = (
    "research/randorder.py",
    "research/r2report.py",
    "research/r2search.py",
    "reports/research/v3c_round2_search.md",
    "reports/research/v3c_round2_search.json",
    "reports/research/v3c_random_order.md",
)

#: Rule identity. Changing any of these changes the researched result.
L25_INPUTS: tuple[str, ...] = (
    "route",
    "decision_time_market_cap_usd",
    "signal_quality",
    "f_liq_to_mcap",
)


# ---------------------------------------------------------------------------
# outcomes
# ---------------------------------------------------------------------------

AUTHORIZED = "AUTHORIZED"
REJECTED_LIQUIDITY_MC_BELOW_THRESHOLD = "LIQUIDITY_MC_BELOW_THRESHOLD"
UNTESTABLE = "UNTESTABLE"

#: Short operator-facing reason per outcome, used verbatim by the Telegram
#: rejection alert. Keys are the OUTCOME codes, not the v3c status strings.
REJECT_REASON_TH: dict[str, str] = {
    REJECTED_LIQUIDITY_MC_BELOW_THRESHOLD: (
        "Liquidity / market cap below L25 threshold"
    ),
    UNTESTABLE: "L25 required value unavailable",
}


@dataclass(frozen=True)
class L25Result:
    """
    Outcome of applying L25 to one candidate at one decision time.

    ``v3c`` is the delegated V3-C sub-result, so a caller can always say WHICH
    gate refused. It is ``None`` only if V3-C could not be evaluated at all,
    which cannot happen in practice but is typed explicitly rather than assumed.
    """

    authorized: bool
    status: str
    market_cap_usd: Optional[float]
    signal_quality: Optional[float]
    liquidity_usd: Optional[float]
    f_liq_to_mcap: Optional[float]
    min_liq_to_mcap: float
    detail: str
    v3c: Optional[V3CResult] = None

    @property
    def rejects(self) -> bool:
        return not self.authorized

    @property
    def status_code(self) -> str:
        """
        The outcome code, namespaced to L25.

        The strategy's own log line is ``L25_REJECT``; the reason code below it
        says which gate. Both are needed: "L25 rejected it" is not actionable,
        "liquidity/MC was 0.71 against a 0.9647 floor" is.
        """
        if self.authorized:
            return "L25_AUTHORIZED"
        if self.v3c is not None and not self.v3c.authorized:
            # V3-C refused first. The V3-C status is the operative reason, and
            # it is preserved verbatim rather than flattened to "L25 rejected".
            return f"V3C_{self.v3c.status}"
        if self.status == UNTESTABLE:
            return "L25_UNTESTABLE"
        return "L25_REJECT"

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": ACTIVE_STRATEGY,
            "authorized": self.authorized,
            "status": self.status,
            "status_code": self.status_code,
            "market_cap_usd": self.market_cap_usd,
            "signal_quality": self.signal_quality,
            "liquidity_usd": self.liquidity_usd,
            "f_liq_to_mcap": self.f_liq_to_mcap,
            "min_liq_to_mcap": self.min_liq_to_mcap,
            "detail": self.detail,
            "v3c": self.v3c.as_dict() if self.v3c is not None else None,
        }


# ---------------------------------------------------------------------------
# integrity
# ---------------------------------------------------------------------------


def _definition_payload() -> str:
    return "|".join(
        [
            L25_ID,
            L25_VERSION,
            V3C_NAME,
            repr(V3C_MAX_MARKET_CAP_USD),
            repr(V3C_MIN_SIGNAL_QUALITY),
            repr(L25_MIN_LIQ_TO_MCAP),
            ",".join(L25_INPUTS),
        ]
    )


#: SHA-256 of the definition. A silent edit to any threshold changes this.
L25_HASH: str = hashlib.sha256(_definition_payload().encode("utf-8")).hexdigest()


def verify_definition(
    min_liq_to_mcap: float = L25_MIN_LIQ_TO_MCAP,
) -> None:
    """
    Raise unless the thresholds are the researched values.

    Verifies the L25 ratio threshold AND re-verifies the V3-C thresholds and the
    V2-A ceiling it inherits, so a drift anywhere in the chain is caught here
    rather than silently inherited by the runtime.
    """
    if float(min_liq_to_mcap) != float(L25_MIN_LIQ_TO_MCAP):
        raise ValueError(
            f"L25_MIN_LIQ_TO_MCAP is FROZEN at {L25_MIN_LIQ_TO_MCAP:.4f} "
            f"(hash prefix {L25_HASH[:8]}). Refusing to run with "
            f"{float(min_liq_to_mcap):.6f}. The threshold was derived as the "
            f"{L25_DERIVATION_METHOD} and re-tuning it invalidates the L25 "
            f"result; it is not a code decision."
        )
    verify_v3c_definition(V3C_MAX_MARKET_CAP_USD, V3C_MIN_SIGNAL_QUALITY)
    verify_ceiling(V2A_CEILING_USD)


# ---------------------------------------------------------------------------
# the ratio
# ---------------------------------------------------------------------------


def f_liq_to_mcap(
    liquidity_usd: Optional[float],
    market_cap_usd: Optional[float],
) -> Optional[float]:
    """
    ``liquidity_usd / market_cap_usd``, or ``None`` when it cannot be computed.

    ``None`` - and therefore a refusal upstream - when either operand is
    missing, zero, negative, non-finite or not a number. Transcribed from
    ``research/r2search.py::liqmc``, which returns ``None`` unless both operands
    are present and truthy.

    A missing liquidity value is NOT coerced to zero and does NOT pass. There is
    no default and no fallback: an invented ratio would be a fabricated input to
    a frozen gate.
    """
    liq = _as_positive_number(liquidity_usd)
    mcap = _as_positive_number(market_cap_usd)
    if liq is None or mcap is None:
        return None
    try:
        ratio = liq / mcap
    except ZeroDivisionError:  # pragma: no cover - mcap > 0 already
        return None
    if ratio != ratio or ratio in (float("inf"), float("-inf")):
        return None
    return ratio


def _as_positive_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    if f <= 0.0:
        return None
    return f


def untestable_reason(
    liquidity_usd: Optional[float],
    market_cap_usd: Optional[float],
) -> str:
    """Which operand made the ratio uncomputable. For the operator's log line."""
    if _as_positive_number(market_cap_usd) is None:
        return "market cap missing or non-positive"
    if _as_positive_number(liquidity_usd) is None:
        return "liquidity missing or non-positive"
    return "ratio not computable"


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------


def evaluate(
    *,
    route: str,
    market_cap_usd: Optional[float],
    signal_quality: Optional[float],
    liquidity_usd: Optional[float] = None,
    min_liq_to_mcap: float = L25_MIN_LIQ_TO_MCAP,
) -> L25Result:
    """
    Apply L25 to one candidate using decision-time values only.

    Order of evaluation is V3-C first, then the liquidity ratio. That order is
    the frozen definition's own: L25 is V3-C AND the ratio, so a candidate V3-C
    already refused must not be re-examined for a liquidity reason.

    Every input is a decision-time value. There is no clock, no history and no
    lookup in this function, so it cannot consult a later print.
    """
    verify_definition(min_liq_to_mcap)

    v3c = evaluate_v3c(
        route=route,
        market_cap_usd=market_cap_usd,
        signal_quality=signal_quality,
    )
    if not v3c.authorized:
        # V3-C's own reason is the operative one, and its UNTESTABLE semantics
        # (a missing market cap or signal quality never auto-passes) carry
        # through unchanged.
        return L25Result(
            authorized=False,
            status=v3c.status,
            market_cap_usd=v3c.market_cap_usd,
            signal_quality=v3c.signal_quality,
            liquidity_usd=liquidity_usd,
            f_liq_to_mcap=None,
            min_liq_to_mcap=float(min_liq_to_mcap),
            detail=v3c.detail,
            v3c=v3c,
        )

    ratio = f_liq_to_mcap(liquidity_usd, market_cap_usd)
    if ratio is None:
        return L25Result(
            authorized=False,
            status=UNTESTABLE,
            market_cap_usd=v3c.market_cap_usd,
            signal_quality=v3c.signal_quality,
            liquidity_usd=liquidity_usd,
            f_liq_to_mcap=None,
            min_liq_to_mcap=float(min_liq_to_mcap),
            detail=(
                "V3-C authorized but f_liq_to_mcap is untestable "
                f"({untestable_reason(liquidity_usd, market_cap_usd)}); "
                "L25 refuses rather than substitute a value"
            ),
            v3c=v3c,
        )

    if ratio < float(min_liq_to_mcap):
        return L25Result(
            authorized=False,
            status=REJECTED_LIQUIDITY_MC_BELOW_THRESHOLD,
            market_cap_usd=v3c.market_cap_usd,
            signal_quality=v3c.signal_quality,
            liquidity_usd=liquidity_usd,
            f_liq_to_mcap=ratio,
            min_liq_to_mcap=float(min_liq_to_mcap),
            detail=(
                f"V3-C authorized but f_liq_to_mcap {ratio:.4f} is below the "
                f"frozen L25 threshold {float(min_liq_to_mcap):.4f}"
            ),
            v3c=v3c,
        )

    return L25Result(
        authorized=True,
        status=AUTHORIZED,
        market_cap_usd=v3c.market_cap_usd,
        signal_quality=v3c.signal_quality,
        liquidity_usd=liquidity_usd,
        f_liq_to_mcap=ratio,
        min_liq_to_mcap=float(min_liq_to_mcap),
        detail=(
            f"V3-C authorized (FAST_ENTRY, market cap "
            f"${v3c.market_cap_usd:,.2f} <= ${V3C_MAX_MARKET_CAP_USD:,.0f}, "
            f"signal quality {v3c.signal_quality:.2f} >= "
            f"{V3C_MIN_SIGNAL_QUALITY:.2f}) and f_liq_to_mcap {ratio:.4f} >= "
            f"{float(min_liq_to_mcap):.4f}"
        ),
        v3c=v3c,
    )


# ---------------------------------------------------------------------------
# display
# ---------------------------------------------------------------------------


def describe() -> str:
    """One-line active-strategy description for the status message."""
    return (
        f"{ACTIVE_STRATEGY}: {V3C_NAME} (MC <= ${V3C_MAX_MARKET_CAP_USD:,.0f}, "
        f"SQ >= {V3C_MIN_SIGNAL_QUALITY:.2f}) + f_liq_to_mcap >= "
        f"{L25_MIN_LIQ_TO_MCAP:.4f}"
    )


def describe_short() -> str:
    return f"{ACTIVE_STRATEGY} (f_liq_to_mcap >= {L25_MIN_LIQ_TO_MCAP:.4f})"


def provenance() -> dict[str, Any]:
    """
    The research provenance record.

    Preserved rather than regenerated: this identifies WHICH research produced
    the thresholds, so the strategy keeps its research identity instead of
    acquiring a fresh "optimized" one.
    """
    return {
        "active_strategy": ACTIVE_STRATEGY,
        "definition": (
            "L25 = V3-C AND f_liq_to_mcap >= 0.9647, where "
            "V3-C = route == FAST_ENTRY AND decision_time_market_cap_usd <= "
            "20000 AND signal_quality >= 0.55, and "
            "f_liq_to_mcap = liquidity_usd / market_cap_usd"
        ),
        "min_liq_to_mcap": L25_MIN_LIQ_TO_MCAP,
        "derived_quantile_exact": L25_DERIVED_QUANTILE_EXACT,
        "derivation_method": L25_DERIVATION_METHOD,
        "derivation_cutoff_epoch": L25_DERIVATION_CUTOFF_EPOCH,
        "research_classification": L25_RESEARCH_CLASSIFICATION,
        "post_derivation_validation": L25_POST_DERIVATION_VALIDATION,
        "research_sources": list(L25_RESEARCH_SOURCES),
        "definition_hash": L25_HASH,
        "inherited": {
            "v3c_market_cap_ceiling_usd": V3C_MAX_MARKET_CAP_USD,
            "v3c_signal_quality_floor": V3C_MIN_SIGNAL_QUALITY,
            "v3c_route": "FAST_ENTRY",
            "v2a_ceiling_usd": V2A_CEILING_USD,
        },
        "not_claimed": [
            "L25 is not proven",
            "L25 is not universally better than V3-C",
            "L25 has no post-derivation validation data",
        ],
    }
