"""
V2_FLOOR: the volume-floor momentum-band entry gate.

WHAT THIS IS
============

A SELECTABLE entry strategy, defined by exactly three decision-time conditions:

    volume_5m_usd / market_cap_usd  >=  1.0     (relative turnover floor)
    change_5m_pct                   >=  +10.0  (momentum floor)
    change_5m_pct                   <=  +300.0 (anti-exhaustion ceiling)

Provenance: ``research/strategy_discovery_v2``. Measured over 193 trades on a
frozen tape -- win rate 66.9% -> 70.5%, expectancy +23.84% -> +28.37%, profit
factor 3.81 -> 4.98, max drawdown 18.05% -> 11.91%, longest losing streak 5 ->
4, against the uncapped V1/RULE control on the identical 254-alert universe.
The floor is the ONLY difference from that control.

WHAT THIS IS NOT
================

It is not the L25 entry gate, and it is not a superset of it. L25 is
``V3-C AND f_liq_to_mcap >= 0.9647``; V3-C is a decision-time market-cap
ceiling of $20,000, a signal-quality floor of 0.55, and a ``route ==
FAST_ENTRY`` requirement. V2_FLOOR contains NONE of those. In particular it
carries no market-cap cap and no liquidity floor, because the research
explicitly tested and REJECTED both (``V2 MC`` failed the sensitivity check on
fewer out-of-sample trades than it saved; ``V2 LIQ`` failed it outright at
96%). Layering V2_FLOOR on top of L25 would silently reinstate exactly the two
filters the study removed, and would trade a different strategy from the one
that was measured. They are therefore separate, independently selectable gates.

WHY A SEPARATE MODULE AND NOT A PARAMETER ON L25
================================================

Three reasons, each of which is a property of the existing architecture rather
than a preference:

1. ``l25.py`` and ``v3c.py`` are hash-locked. ``L25_HASH`` is re-verified on
   every ``evaluate()`` call, and a test asserts that neither module may so much
   as mention ``strategy_profile``. Adding a flag would either break the hash or
   weaken the immutability guarantee that makes the L25 result trustworthy.

2. A refusal has to name the gate that refused. ``_strategy_flag_code`` maps a
   refusal to a distinct risk-flag code so the journal always records WHICH
   threshold failed. Two gates with one code would make the journal unable to
   distinguish them, which is the failure mode that made an earlier research arm
   in this repository uninterpretable.

3. The frozen pipeline route selection (``analysis/routes.py``) still applies to
   V2_FLOOR and is deliberately untouched: thin liquidity, the market-cap sanity
   band, the extension ladder and the data-state gate are the existing common
   structural architecture, not strategy-level filters. What changes is only
   WHICH strategy gate authorizes the entry.

FAIL-CLOSED DISCIPLINE
======================

Every condition refuses rather than passes when the value it needs is absent.
This follows ``l25.f_liq_to_mcap`` exactly, and for the same reason: a filter
that admits what it cannot see is not a filter. The failure that motivates it is
documented in ``paper_live/arms.py::_ratio_of`` -- an earlier rule read a
missing field as ``None``, treated it as "not applicable", and refused every
signal for a reason that had nothing to do with the hypothesis. Here an
untestable ratio is ``UNTESTABLE``, and untestable does not authorize.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

#: Relative-turnover floor. ``volume_5m_usd / market_cap_usd >= 1.0`` reads as
#: "the pool turned over its whole market cap in five minutes", which on a small
#: float is a demand signal rather than a price signal.
#:
#: Provenance: research/strategy_discovery_v2 REPORT section 2.1. Monotone across
#: 0.5 -> 3.0 with no sign change; 1.0 is the value whose out-of-sample CI first
#: excludes zero while retaining n=25 on the control's own universe.
V2F_MIN_VOL5_PER_MC: float = 1.0

#: Momentum FLOOR. The single strategy-level change from the V1/RULE control,
#: which had only a ceiling.
#:
#: Provenance: research/strategy_discovery_v2 REPORT section 2.3. Adding it
#: raised win rate, expectancy, profit factor, drawdown and losing streak
#: SIMULTANEOUSLY, and it is the only change in the whole study that did.
#: Mechanistically it is not "momentum predicts continuation": after the
#: vol5/MC >= 1.0 filter, vol5/MC itself has no remaining power to separate
#: winners from losers (AUC 0.52, Mann-Whitney p=0.65). The floor works by
#: removing a distinct negative-momentum REGIME -- the losers' lower-quartile
#: t_c5 is -2.7% against +26.8% for winners.
V2F_MIN_CHANGE_5M_PCT: float = 10.0

#: Anti-exhaustion CEILING, retained unchanged from the V1/RULE control.
#:
#: Provenance: research/strategy_discovery_v2 REPORT section 2.2. The excluded
#: population (already up more than 4x in the trailing five minutes) returned
#: +0.8% median against +16.3% for the included one, in all three splits.
V2F_MAX_CHANGE_5M_PCT: float = 300.0

#: Operator-facing name, and the token the runtime records.
V2F_NAME = "V2_FLOOR"
V2F_STRATEGY = "V2_FLOOR"

# Outcome tokens. Distinct per gate, so a refusal in the journal always names
# which condition was operative.
AUTHORIZED = "V2F_AUTHORIZED"
REJECTED_VOL5_PER_MC_BELOW_FLOOR = "V2F_REJECTED_VOL5_PER_MC_BELOW_FLOOR"
REJECTED_CHANGE_5M_BELOW_FLOOR = "V2F_REJECTED_CHANGE_5M_BELOW_FLOOR"
REJECTED_CHANGE_5M_ABOVE_CEILING = "V2F_REJECTED_CHANGE_5M_ABOVE_CEILING"
UNTESTABLE = "V2F_UNTESTABLE"

_REJECTED_STATUSES = frozenset({
    REJECTED_VOL5_PER_MC_BELOW_FLOOR,
    REJECTED_CHANGE_5M_BELOW_FLOOR,
    REJECTED_CHANGE_5M_ABOVE_CEILING,
})


def f_vol5_per_mc(
    volume_5m_usd: float | None, market_cap_usd: float | None
) -> float | None:
    """
    Trailing-5m volume over market cap, or ``None`` if it cannot be computed.

    ``None`` on ANY missing, non-positive or non-finite operand, and never a
    substituted default. Mirrors ``l25.f_liq_to_mcap``: the caller must treat
    ``None`` as untestable and refuse, because a ratio invented from a partial
    input is not a measurement of anything.
    """
    if volume_5m_usd is None or market_cap_usd is None:
        return None
    try:
        v = float(volume_5m_usd)
        m = float(market_cap_usd)
    except (TypeError, ValueError):
        return None
    if not (v > 0.0) or not (m > 0.0):
        return None
    if m != m or v != v:            # NaN
        return None
    if m in (float("inf"), float("-inf")) or v in (
        float("inf"), float("-inf")
    ):
        return None
    return v / m


def _finite(value: float | None) -> bool:
    if value is None:
        return False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return f == f and f not in (float("inf"), float("-inf"))


def untestable_reason(
    volume_5m_usd: float | None,
    market_cap_usd: float | None,
    change_5m_pct: float | None,
) -> str:
    """Exactly which input was unreadable. Never a generic 'bad data'."""
    missing = []
    if not _finite(volume_5m_usd):
        missing.append(f"volume_5m_usd={volume_5m_usd!r}")
    if not _finite(market_cap_usd):
        missing.append(f"market_cap_usd={market_cap_usd!r}")
    if not _finite(change_5m_pct):
        missing.append(f"change_5m_pct={change_5m_pct!r}")
    return ", ".join(missing) if missing else "no reason identified"


@dataclass(frozen=True)
class V2FloorResult:
    """Outcome of applying the V2_FLOOR gate to one candidate."""

    authorized: bool
    status: str
    vol5_per_mc: Optional[float]
    change_5m_pct: Optional[float]
    min_vol5_per_mc: float
    min_change_5m_pct: float
    max_change_5m_pct: float
    detail: str

    @property
    def rejects(self) -> bool:
        return not self.authorized

    def as_dict(self) -> dict:
        return {
            "strategy": V2F_STRATEGY,
            "authorized": self.authorized,
            "status": self.status,
            "vol5_per_mc": self.vol5_per_mc,
            "change_5m_pct": self.change_5m_pct,
            "min_vol5_per_mc": self.min_vol5_per_mc,
            "min_change_5m_pct": self.min_change_5m_pct,
            "max_change_5m_pct": self.max_change_5m_pct,
            "detail": self.detail,
        }


#: Hash of the frozen threshold block, re-verified on every ``evaluate`` call.
#: Same discipline as ``l25.L25_HASH``: a threshold cannot be edited at runtime,
#: through config, or by a monkeypatch without this failing loudly.
_DEFINITION_SOURCE = (
    f"V2F_MIN_VOL5_PER_MC={V2F_MIN_VOL5_PER_MC!r};"
    f"V2F_MIN_CHANGE_5M_PCT={V2F_MIN_CHANGE_5M_PCT!r};"
    f"V2F_MAX_CHANGE_5M_PCT={V2F_MAX_CHANGE_5M_PCT!r}"
)
V2F_HASH: str = hashlib.sha256(_DEFINITION_SOURCE.encode("utf-8")).hexdigest()


def verify_definition(
    min_vol5_per_mc: float = V2F_MIN_VOL5_PER_MC,
    min_change_5m_pct: float = V2F_MIN_CHANGE_5M_PCT,
    max_change_5m_pct: float = V2F_MAX_CHANGE_5M_PCT,
) -> None:
    """
    Refuse any threshold other than the frozen one.

    Raised, not warned. A quietly different threshold produces a result that is
    indistinguishable from a working one while trading a different strategy.
    """
    if (
        float(min_vol5_per_mc) != V2F_MIN_VOL5_PER_MC
        or float(min_change_5m_pct) != V2F_MIN_CHANGE_5M_PCT
        or float(max_change_5m_pct) != V2F_MAX_CHANGE_5M_PCT
    ):
        raise ValueError(
            "V2_FLOOR thresholds are frozen at "
            f"vol5/mc >= {V2F_MIN_VOL5_PER_MC}, "
            f"change_5m in [{V2F_MIN_CHANGE_5M_PCT}, "
            f"{V2F_MAX_CHANGE_5M_PCT}]; got "
            f"({min_vol5_per_mc}, {min_change_5m_pct}, "
            f"{max_change_5m_pct})"
        )


def evaluate(
    *,
    volume_5m_usd: float | None,
    market_cap_usd: float | None,
    change_5m_pct: float | None,
    min_vol5_per_mc: float = V2F_MIN_VOL5_PER_MC,
    min_change_5m_pct: float = V2F_MIN_CHANGE_5M_PCT,
    max_change_5m_pct: float = V2F_MAX_CHANGE_5M_PCT,
) -> V2FloorResult:
    """
    The V2_FLOOR entry gate, evaluated at one decision time.

    Conditions in a fixed order, each refusing rather than passing when its
    input is unreadable:

      1. ``vol5_per_mc`` computable, else UNTESTABLE
      2. ``vol5_per_mc >= min_vol5_per_mc``
      3. ``change_5m_pct`` finite, else UNTESTABLE
      4. ``change_5m_pct >= min_change_5m_pct``
      5. ``change_5m_pct <= max_change_5m_pct``

    Boundary values are INCLUSIVE on both ends: ``vol5_per_mc == 1.0``,
    ``change_5m == +10.0`` and ``change_5m == +300.0`` all authorize. That is
    the reading the research used, and an off-by-one on an inclusive bound is
    exactly the kind of quiet difference that makes a reproduction fail.
    """
    verify_definition(
        min_vol5_per_mc, min_change_5m_pct, max_change_5m_pct
    )
    lo_v = float(min_vol5_per_mc)
    lo_c = float(min_change_5m_pct)
    hi_c = float(max_change_5m_pct)

    ratio = f_vol5_per_mc(volume_5m_usd, market_cap_usd)
    change = (
        float(change_5m_pct) if _finite(change_5m_pct) else None
    )

    if ratio is None or change is None:
        return V2FloorResult(
            authorized=False,
            status=UNTESTABLE,
            vol5_per_mc=ratio,
            change_5m_pct=change,
            min_vol5_per_mc=lo_v,
            min_change_5m_pct=lo_c,
            max_change_5m_pct=hi_c,
            detail=(
                "V2_FLOOR is untestable at decision time ("
                f"{untestable_reason(volume_5m_usd, market_cap_usd, change_5m_pct)}"
                "); it refuses rather than pass an absent value or substitute "
                "a default"
            ),
        )

    if ratio < lo_v:
        return V2FloorResult(
            authorized=False,
            status=REJECTED_VOL5_PER_MC_BELOW_FLOOR,
            vol5_per_mc=ratio,
            change_5m_pct=change,
            min_vol5_per_mc=lo_v,
            min_change_5m_pct=lo_c,
            max_change_5m_pct=hi_c,
            detail=(
                f"vol5/market_cap {ratio:.4f} is below the V2_FLOOR relative-"
                f"turnover floor {lo_v:.4f} (5m volume "
                f"${float(volume_5m_usd):,.0f} over market cap "
                f"${float(market_cap_usd):,.0f})"
            ),
        )

    if change < lo_c:
        return V2FloorResult(
            authorized=False,
            status=REJECTED_CHANGE_5M_BELOW_FLOOR,
            vol5_per_mc=ratio,
            change_5m_pct=change,
            min_vol5_per_mc=lo_v,
            min_change_5m_pct=lo_c,
            max_change_5m_pct=hi_c,
            detail=(
                f"5m change {change:+.1f}% is below the V2_FLOOR momentum floor "
                f"{lo_c:+.1f}%; the token was already falling when the alert "
                "fired, and the research measured that regime as a distinct "
                "losing population"
            ),
        )

    if change > hi_c:
        return V2FloorResult(
            authorized=False,
            status=REJECTED_CHANGE_5M_ABOVE_CEILING,
            vol5_per_mc=ratio,
            change_5m_pct=change,
            min_vol5_per_mc=lo_v,
            min_change_5m_pct=lo_c,
            max_change_5m_pct=hi_c,
            detail=(
                f"5m change {change:+.1f}% is above the V2_FLOOR anti-exhaustion "
                f"ceiling {hi_c:+.1f}%; a token already up more than 4x in the "
                "trailing 5m returned +0.8% median in the research"
            ),
        )

    return V2FloorResult(
        authorized=True,
        status=AUTHORIZED,
        vol5_per_mc=ratio,
        change_5m_pct=change,
        min_vol5_per_mc=lo_v,
        min_change_5m_pct=lo_c,
        max_change_5m_pct=hi_c,
        detail=(
            f"vol5/market_cap {ratio:.4f} >= {lo_v:.4f} and 5m change "
            f"{change:+.1f}% within [{lo_c:+.1f}%, {hi_c:+.1f}%]"
        ),
    )


def flag_code(result: V2FloorResult) -> str:
    """
    The risk-flag code for a V2_FLOOR refusal.

    ``v2floor_``-prefixed and distinct per gate, matching the existing
    ``v3c_`` / ``l25_`` convention so the journal always names which threshold
    was operative and a V2_FLOOR refusal can never be mistaken for an L25 one.
    """
    if result.status == REJECTED_VOL5_PER_MC_BELOW_FLOOR:
        return "v2floor_vol5_per_mc_below_floor"
    if result.status == REJECTED_CHANGE_5M_BELOW_FLOOR:
        return "v2floor_change_5m_below_floor"
    if result.status == REJECTED_CHANGE_5M_ABOVE_CEILING:
        return "v2floor_change_5m_above_ceiling"
    if result.status == UNTESTABLE:
        return "v2floor_untestable"
    # An AUTHORIZED result has no refusal flag, and must never be handed one.
    # Falling through to a "not_authorized" code here would let any caller that
    # logs unconditionally write a refusal into the journal for a trade that
    # was taken -- the one failure mode this whole module exists to prevent.
    # Callers that only flag refusals still get the right answer; this makes
    # the misuse visible instead of silent.
    return "v2floor_authorized"


def is_rejected_status(status: str) -> bool:
    """Whether a status token is a definite refusal (vs UNTESTABLE)."""
    return status in _REJECTED_STATUSES


def rule_summary() -> str:
    """The exact rule, one line, for operator surfaces and logs."""
    return (
        f"vol5/mcap >= {V2F_MIN_VOL5_PER_MC:g} AND "
        f"{V2F_MIN_CHANGE_5M_PCT:+g}% <= 5m change <= "
        f"{V2F_MAX_CHANGE_5M_PCT:+g}%"
    )
