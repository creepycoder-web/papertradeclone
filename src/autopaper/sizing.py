"""
V3-C position sizing. THE single function that produces a position size.

POLICY (exact, and the only policy)
-----------------------------------

    IF equity <= $50.00:   position_pct = 10%
    IF equity >  $50.00:   position_pct = 12.5%

    position_usd = equity * position_pct

Worked examples, all of which are asserted in the test suite:

    $1.55   -> $0.155
    $5      -> $0.50
    $10     -> $1.00
    $25     -> $2.50
    $50     -> $5.00
    $50.01  -> $6.25125
    $100    -> $12.50

The boundary is inclusive at exactly $50.00, so $50.00 -> $5.00 (10%) and
$50.01 -> $6.25125 (12.5%).

PER-PROFILE OVERRIDE (the only second branch, and it is one lookup)
------------------------------------------------------------------

A profile may declare its own percentage via
``strategy_profile.PROFILE_POSITION_PCT``. ``V2_FLOOR`` declares a FLAT 10%,
because that is the size the V2 research was measured at at every balance.
``None`` -- the value for all three L25 profiles -- means "use the shared
policy above", so this cannot change any existing strategy's size.

The override is a percentage of CURRENT equity, exactly like the shared policy,
so compounding is unchanged: the account is re-valued before every entry and
10% of that new equity is staked. There is no notion of a fixed starting stake.

The researched number lives in the profile registry, not here. This module reads
it. That is what keeps the global risk logic free of per-strategy ladders.

WHAT IS DELIBERATELY NOT HERE
----------------------------

* no all-in
* no martingale, no size increase after losses
* no Telegram-supplied size
* no automatic round-up to a venue minimum
* no per-strategy EQUITY ladder -- only a flat percentage override

MICRO ACCOUNT
-------------

With ~$1.55 of real SOL, equity sits in the 10% branch and the first order is
$0.155. This module NEVER enlarges that. If the venue cannot execute it, the
venue executor returns ``MICRO_REAL_EXECUTION_UNSUPPORTED`` and the trade is
rejected with the exact reason. Silently converting $0.155 into $5 is the
single worst thing this code could do, so there is no branch that could do it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .strategy_profile import position_pct_override

#: Percentage-of-equity applied when equity is at or below the $50 boundary.
SIZING_PCT_UNDER_50: float = 10.0

#: Percentage-of-equity applied when equity is strictly above $50.
SIZING_PCT_OVER_50: float = 12.5

#: The equity boundary. Inclusive: equity == 50.00 uses the 10% branch.
SIZING_BOUNDARY_USD: float = 50.0

#: The paper simulator's modelled costs. They are reported alongside the size;
#: they are NOT subtracted from ``position_usd``, because the policy states the
#: position size as a percentage of equity and shrinking the notional to make a
#: fee fit would silently move the position off the policy.
#: (The previous implementation sized against post-fee cash, which is how a
#: $1.55 account quietly ended up trading a different percentage.)
PAPER_FEE_BPS: float = 85.0
PAPER_SLIPPAGE_BPS: float = 120.0


@dataclass(frozen=True)
class PositionSize:
    """A resolved, immutable position size."""

    equity_usd: float
    position_pct: float
    position_usd: float
    branch: str
    estimated_fee_usd: float
    estimated_slippage_usd: float
    rationale: str

    def as_dict(self) -> dict:
        return {
            "equity_usd": self.equity_usd,
            "position_pct": self.position_pct,
            "position_usd": self.position_usd,
            "branch": self.branch,
            "estimated_fee_usd": self.estimated_fee_usd,
            "estimated_slippage_usd": self.estimated_slippage_usd,
            "rationale": self.rationale,
        }


def position_pct_for_equity(
    equity_usd: float, *, profile: Optional[object] = None
) -> float:
    """
    The percentage-of-equity for an account, and nothing else.

    This is the whole policy. Every other size in the system flows from it.

    ``profile`` is OPTIONAL and defaults to ``None`` = the shared policy. That
    default is load-bearing: every call site that does not know about profiles
    -- and every test written before them -- gets the shared ladder unchanged.
    """
    override = position_pct_override(profile)
    if override is not None:
        return float(override)
    if float(equity_usd) <= SIZING_BOUNDARY_USD:
        return SIZING_PCT_UNDER_50
    return SIZING_PCT_OVER_50


def size_for_equity(
    equity_usd: float, *, profile: Optional[object] = None
) -> float:
    """
    ``equity * position_pct/100``, unrounded.

    Rounding is deliberately absent. $0.155 is not $0.16, and $6.25125 is not
    $6.25 - rounding a micro position up by a third of itself is the same class
    of quiet breach as rounding it up to a venue minimum. Downstream, the venue
    (not this function) decides the smallest executable amount and reports a
    rejection when there isn't one.
    """
    pct = position_pct_for_equity(equity_usd, profile=profile)
    return float(equity_usd) * pct / 100.0


def compute_position_size(
    equity_usd: float, *, profile: Optional[object] = None
) -> PositionSize:
    """
    Resolve the position size for an account, optionally under a strategy
    profile.

    The ONE function that turns equity into a notional. There is no other.

    ``equity_usd`` must be the CURRENT equity at the moment of the decision, not
    a starting balance: compounding is simply this function being called again on
    a re-valued account. Both the shared policy and the profile override are
    percentages of whatever is passed in.
    """
    equity = float(equity_usd)
    override = position_pct_override(profile)
    pct = position_pct_for_equity(equity, profile=profile)
    usd = equity * pct / 100.0
    if override is not None:
        branch = f"profile:{_profile_value(profile)}"
    else:
        branch = (
            "equity<=50" if equity <= SIZING_BOUNDARY_USD else "equity>50"
        )
    fee = usd * PAPER_FEE_BPS / 10_000.0
    slip = usd * PAPER_SLIPPAGE_BPS / 10_000.0
    if override is not None:
        rationale = (
            f"equity ${equity:,.4f} -> {pct:g}% "
            f"(own researched sizing for {_profile_token(profile)}; the shared "
            f"$50 ladder is overridden) -> ${usd:,.6f}"
        )
    else:
        rationale = (
            f"equity ${equity:,.4f} {branch} -> {pct:g}% -> "
            f"${usd:,.6f}"
        )
    return PositionSize(
        equity_usd=equity,
        position_pct=pct,
        position_usd=usd,
        branch=branch,
        estimated_fee_usd=fee,
        estimated_slippage_usd=slip,
        rationale=rationale,
    )


def _profile_token(profile: object) -> str:
    """The profile's own name for a human-readable branch/rationale."""
    from .strategy_profile import label as _label
    from .strategy_profile import StrategyProfileError
    try:
        return _label(profile)
    except StrategyProfileError:
        return str(profile)


def _profile_value(profile: object) -> str:
    """The profile's stable token, for machine-readable fields like ``branch``."""
    from .strategy_profile import StrategyProfileError, parse_profile
    try:
        return parse_profile(profile).value
    except StrategyProfileError:
        return str(profile)


def next_position_label(
    equity_usd: float | None, *, profile: Optional[object] = None
) -> str:
    """
    Operator-facing 'Next Position' line for the status message.

    ``None`` is UNKNOWN equity, and it is rendered as such. Previously this
    called ``compute_position_size(None)``, which does ``float(None)`` and
    raises -- so a REAL wallet whose equity could not be valued would have
    crashed the status message instead of reporting that it does not know. An
    unknown equity must also never print a notional, because a notional derived
    from no equity is a fabricated order size.
    """
    if equity_usd is None:
        return "— (equity unavailable)"
    size = compute_position_size(equity_usd, profile=profile)
    if size.equity_usd <= 0:
        return "—"
    return f"${size.position_usd:,.6f} ({size.position_pct:g}%)"


def allocation_label(
    equity_usd: float | None, *, profile: Optional[object] = None
) -> str:
    """
    Operator-facing 'Allocation' line for the status message.

    None-safe for the same reason as :func:`next_position_label`: an unknown
    equity gets no percentage, because picking a tier would assert which side of
    the $50 boundary a number we do not have falls on.
    """
    if equity_usd is None:
        return "— (equity unavailable)"
    size = compute_position_size(equity_usd, profile=profile)
    if position_pct_override(profile) is not None:
        return (f"{size.position_pct:g}% "
                f"({_profile_token(profile)} own researched sizing)")
    if size.equity_usd <= SIZING_BOUNDARY_USD:
        return f"10% (equity <= $50)"
    return f"12.5% (equity > $50)"


def describe_policy(profile: Optional[object] = None) -> str:
    """The policy in words, for the operator and the safety probe."""
    override = position_pct_override(profile)
    if override is not None:
        return (
            f"{override:g}% flat, own researched sizing for "
            f"{_profile_token(profile)} (shared $50 ladder overridden)"
        )
    return (
        f"equity <= $50 -> {SIZING_PCT_UNDER_50:g}%, "
        f"equity > $50 -> {SIZING_PCT_OVER_50:g}%"
    )


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

MIN_EQUITY_USD: float = 0.0


def tradeable(size: PositionSize, *, available_cash_usd: Optional[float] = None) -> tuple[bool, str]:
    """
    Is this size actually placeable right now?

    Deliberately does NOT apply a venue minimum. If the venue has one, the venue
    executor is what reports it, with the venue's own reason. This function only
    refuses a size that cannot be attempted at all.
    """
    if size.equity_usd <= MIN_EQUITY_USD:
        return False, "account equity is zero; nothing to size"
    if size.position_usd <= 0:
        return False, "computed position size is zero; nothing to size"
    if (
        available_cash_usd is not None
        and available_cash_usd < size.position_usd
    ):
        return (
            False,
            f"available ${available_cash_usd:,.6f} is below the required "
            f"${size.position_usd:,.6f}",
        )
    return True, "size is placeable"
