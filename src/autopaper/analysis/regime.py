"""
Regime classification.

A token's condition at decision time, derived only from currently available
information. When the data cannot support a confident label, the answer is
``UNCERTAIN`` - the requirement is explicit that we should not pretend detailed
regimes exist if the data is insufficient.
"""

from __future__ import annotations

from ..models import DataState, Regime
from .features import FeatureSet

# Regimes that are inherently fragile: explosive moves on thin books give back
# fast, so entries there get tighter failure protection.
FRAGILE_REGIMES = {Regime.EXPLOSIVE, Regime.EXTENDED}
# Regimes that tend to be worth holding.
CONTINUATION_REGIMES = {Regime.VERY_EARLY, Regime.EARLY, Regime.ACCELERATING}


def classify(f: FeatureSet) -> tuple[Regime, str]:
    """Return ``(regime, reason)``. ``reason`` is human-readable justification."""
    if f.data_state is DataState.MISSING:
        return Regime.UNCERTAIN, "no market observation at decision time"
    if f.data_state is DataState.UNUSABLE:
        return Regime.UNCERTAIN, "all market data beyond the unusable window"
    if f.price is None:
        return Regime.UNCERTAIN, "no price available"

    # A single print cannot characterise a move, whatever the derived momentum
    # field happens to say. With one observation the momentum helpers echo the
    # current price back as 0.0%, which would otherwise read as "flat and calm"
    # and hand a brand-new token a confident VERY_EARLY label off no evidence.
    if f.observation_count <= 1:
        return Regime.UNCERTAIN, "single observation, no move measured yet"

    c5 = f.change_5m
    ext = f.extension_from_first_valid_price
    dd = f.drawdown_from_high_pct
    age = f.token_age_s

    # Momentum: use the longest window we actually have, not the longest window
    # we wish we had. A token observed for 6 seconds has no 5m change, so the
    # short-span measure is the honest one - and labelling it as such in the
    # reason string keeps the reader from mistaking it for a 5-minute read.
    move: float | None = None
    window = ""
    for value, name, w in (
        (f.change_5m, "5m", "5m"),
        (f.change_15m, "15m", "15m"),
        (f.momentum_60s_pct, "60s", "60s"),
        (f.momentum_30s_pct, "30s", "30s"),
        (f.change_over_span_pct, "span", "observed span"),
    ):
        if value is not None and (move is None or _weight(w) > _weight(window)):
            move, window = value, w
    if c5 is None and move is not None:
        c5 = move

    # --- clearly weakening / broken -------------------------------------
    if dd is not None and dd <= -30 and (c5 is None or c5 < 0):
        return Regime.WEAKENING, f"off high by {dd:.0f}% with no positive 5m momentum"
    if c5 is not None and c5 <= -12 and f.buy_sell_ratio is not None and f.buy_sell_ratio < 1.0:
        return Regime.WEAKENING, f"5m change {c5:.0f}% with sells dominating"

    # --- extended --------------------------------------------------------
    if ext is not None and ext >= 1.0:
        return Regime.EXTENDED, f"{ext * 100:.0f}% above the first valid price"
    if c5 is not None and c5 >= 90:
        return Regime.EXPLOSIVE, f"5m change {c5:.0f}%"

    # --- explosive -------------------------------------------------------
    if c5 is not None and c5 >= 40:
        fragile = (f.liquidity_score < 0.5) or (f.activity_score < 0.45)
        return (
            Regime.EXPLOSIVE,
            f"5m change {c5:.0f}%"
            + (" on a fragile book" if fragile else " with real depth"),
        )
    if f.momentum_30s_pct is not None and f.momentum_30s_pct >= 25:
        return Regime.ACCELERATING, f"+{f.momentum_30s_pct:.0f}% in the last 30s"

    # --- early -----------------------------------------------------------
    #
    # When no earlier price exists at all (a token we saw for the first time),
    # extension is genuinely unknown - not zero. The signal-quality score
    # already discounts that case, and the route logic caps the entry size, so
    # refusing to classify here would just label every live launch UNCERTAIN.
    if ext is None:
        if (age is None or age <= 900) and move is not None and move > 0:
            return (
                Regime.VERY_EARLY,
                f"first sighting, +{move:.0f}% over the {window} with no earlier "
                "price to measure extension from",
            )
        if f.signal_quality < 0.45:
            return Regime.UNCERTAIN, "first sighting with weak structure"
    if ext is not None and ext <= 0.15 and (age is None or age <= 900):
        return Regime.VERY_EARLY, f"only {ext * 100:.0f}% above first valid price"
    if ext is not None and ext <= 0.35:
        return Regime.EARLY, f"{ext * 100:.0f}% above first valid price"

    # --- stabilising -----------------------------------------------------
    if (c5 is not None and -4 <= c5 <= 6) and f.activity_score < 0.4:
        return Regime.STABILIZING, "5m move has flattened and activity is fading"
    if c5 is not None and 6 < c5 < 20 and f.activity_score >= 0.5:
        return Regime.ACCELERATING, f"steady 5m bid: {c5:.0f}%"

    # --- weak signal -----------------------------------------------------
    if move is None:
        return Regime.UNCERTAIN, "no usable momentum measure at any window"

    if f.signal_quality < 0.35:
        return Regime.UNCERTAIN, f"signal quality only {f.signal_quality:.2f}"

    return Regime.UNCERTAIN, "conditions do not match a confident regime"


_WINDOW_WEIGHT = {"5m": 5, "15m": 4, "60s": 3, "30s": 2, "span": 1, "observed span": 1, "": 0}


def _weight(window: str) -> int:
    return _WINDOW_WEIGHT.get(window, 0)


def is_fragile(regime: Regime) -> bool:
    return regime in FRAGILE_REGIMES


def is_continuation(regime: Regime) -> bool:
    return regime in CONTINUATION_REGIMES
