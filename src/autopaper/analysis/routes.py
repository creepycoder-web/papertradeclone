"""
V3-C route resolution.

V3-C is the ONE route. The other entry routes that used to live here -
PRECISION_ENTRY, CONFIRMATION_ENTRY, RECOVERY_RECLAIM - have been REMOVED from
the runtime. They are not disabled, not gated behind a flag, not reachable by
configuration: they no longer exist as code paths. Historical research into
their historical study is archived outside this package and is
deliberately not imported here.

What remains is the *shape* V3-C needs:

* a FAST_ENTRY classification, so the token's current condition is still read
  (liquidity, activity, extension, regime) and a setup the pipeline would never
  have traded is still refused;
* an exit plan, written before the trade exists.

Sizing is NOT decided here. It comes from :mod:`autopaper.sizing`, the one
function that turns equity into a notional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..config import DecisionConfig, ExitConfig
from ..models import DataState, ExitStyle, Regime, Route
from .features import FeatureSet
from .regime import classify, is_fragile


@dataclass(frozen=True)
class RouteChoice:
    """
    The route V3-C resolved to, with the reason.

    ``route`` is FAST_ENTRY or NO_TRADE, and nothing else. V3-C itself is
    applied one level up, in :mod:`autopaper.v3c` and the decision engine, on
    decision-time market cap and signal quality.
    """

    route: Route
    exit_style: ExitStyle
    rationale: str
    modifiers: tuple[str, ...] = ()
    #: Set when the entry's own shape requires a specific exit style regardless
    #: of the regime label. A position bought far above the first valid price is
    #: a fast-fail even if the regime reads ACCELERATING, because the thing that
    #: makes it dangerous is the entry, not the current momentum.
    style_override: ExitStyle | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "route": self.route.value,
            "exit_style": self.exit_style.value,
            "rationale": self.rationale,
            "modifiers": list(self.modifiers),
        }


#: The one route, mapped to its exit style. V3-C is FAST_ENTRY; there is no
#: second row.
ROUTE_EXIT_STYLE: Mapping[Route, ExitStyle] = {
    Route.FAST_ENTRY: ExitStyle.FAST_FAILURE,
    Route.NO_TRADE: ExitStyle.TIME_DECAY,
}


def exit_style_for(
    route: Route, regime: Regime, override: ExitStyle | None = None
) -> ExitStyle:
    """
    The exit style a V3-C position gets.

    FAST_ENTRY is intrinsically a fast-failure setup, so the regime never
    promotes it to a runner. ``override`` wins when the entry was only
    permitted by breaching the extension ladder.
    """
    if override is not None:
        return override
    if route in ROUTE_EXIT_STYLE:
        return ROUTE_EXIT_STYLE[route]
    return ExitStyle.MOMENTUM_FAILURE


def _no_trade(reason: str, mods: tuple[str, ...] = ()) -> RouteChoice:
    return RouteChoice(
        route=Route.NO_TRADE, exit_style=ExitStyle.TIME_DECAY, rationale=reason,
        modifiers=mods,
    )


def select_route(f: FeatureSet, cfg: DecisionConfig) -> RouteChoice:
    """
    Classify a candidate as a V3-C FAST_ENTRY setup, or refuse it.

    This is the *data quality and structure* gate. The V3-C thresholds
    (market cap and signal quality) are applied separately by
    :func:`autopaper.v3c.evaluate`, because they are decision-time gates and
    this function is about whether the setup is tradeable at all.

    Refuses, with a distinct reason in each case:
      * absent or expired market data
      * pipeline signal quality below the (looser) pipeline floor
      * thin liquidity
      * market cap outside the pipeline's own sanity band
      * an exhausted extension (the known failure mode)
      * a reversing token
    """
    if f.data_state in (DataState.MISSING, DataState.UNUSABLE):
        return _no_trade(
            f"data state {f.data_state.value}: no trade on absent or expired data"
        )
    if (
        f.signal_quality < cfg.min_signal_quality_to_trade
        and f.data_state is not DataState.STALE
    ):
        # "We looked and it is not good enough" is a refusal. "We could not look
        # properly" is not: a STALE print lowers signal quality through the
        # staleness discount, not because the token is weak, so it falls
        # through to V3-C and the decision layer answers WAIT - look again in a
        # few seconds - rather than REJECT, which would be a permanent verdict
        # on a token that was never really examined.
        reason = (
            f"signal quality {f.signal_quality:.2f} below pipeline floor "
            f"{cfg.min_signal_quality_to_trade:.2f}"
        )
        return _no_trade(reason)

    regime, _reason = classify(f)
    liq = f.liquidity_usd
    mcap = f.market_cap_usd
    ext = f.extension_from_first_valid_price
    mods: list[str] = []
    style_override: ExitStyle | None = None

    # --- structural gates ------------------------------------------------
    if liq is not None and liq < cfg.min_liquidity_usd:
        return _no_trade(
            f"liquidity ${liq:,.0f} below ${cfg.min_liquidity_usd:,.0f}"
        )
    if mcap is not None and mcap > cfg.max_market_cap_usd:
        return _no_trade(
            f"market cap ${mcap:,.0f} above ${cfg.max_market_cap_usd:,.0f}"
        )
    if mcap is not None and mcap < cfg.min_market_cap_usd:
        return _no_trade(
            f"market cap ${mcap:,.0f} below ${cfg.min_market_cap_usd:,.0f}"
        )

    # --- the anti-exhaustion ladder --------------------------------------
    #
    # The known failure mode: entries far above the first valid observed price
    # perform poorly. High momentum is also where the biggest runners live, so
    # this is NOT a blanket "reject anything hot" rule. It is a three-rung
    # permission ladder and the rungs tighten as extension grows.
    if ext is None:
        # Unknown extension is neither a pass nor a fail: it removes the
        # chaseable band, so the setup must stand on its own liquidity and
        # activity.
        if f.liquidity_score < 0.5 or f.activity_score < 0.5:
            return _no_trade(
                "no earlier price to measure extension from, and the book does "
                "not stand on its own",
                ("extension_unanchored",),
            )
        mods.append("extension_unanchored")
    else:
        perfect = cfg.perfect_extension_to_chase
        max_chase = max(cfg.max_extension_to_chase, perfect)
        strong = (
            regime in (Regime.ACCELERATING, Regime.EXPLOSIVE)
            and f.liquidity_score >= 0.55
            and f.activity_score >= 0.55
            and f.buy_sell_ratio is not None
            and f.buy_sell_ratio >= 1.3
        )
        very_strong = (
            strong
            and f.liquidity_score >= 0.60
            and f.activity_score >= 0.60
            and f.buy_sell_ratio is not None
            and f.buy_sell_ratio >= 2.0
        )
        if ext > 1.6:
            return _no_trade(
                f"{ext * 100:.0f}% above first valid price - exhausted",
                ("extension_exhausted",),
            )
        if ext > max_chase:
            if not very_strong:
                return _no_trade(
                    f"{ext * 100:.0f}% above first valid price with only "
                    f"liq={f.liquidity_score:.2f} act={f.activity_score:.2f} "
                    "- this is the exhausted second wave, not the acceleration",
                    ("extension_exhausted",),
                )
            mods.append("extended_very_strong")
            style_override = ExitStyle.FAST_FAILURE
        elif ext > perfect and not strong:
            return _no_trade(
                f"{ext * 100:.0f}% above first valid price is past the "
                f"{perfect * 100:.0f}% chaseable band and the book does not "
                f"support it (liq={f.liquidity_score:.2f} "
                f"act={f.activity_score:.2f})",
                ("extension_stretched",),
            )
        elif ext > perfect:
            mods.append("extended_but_strong")
            style_override = ExitStyle.FAST_FAILURE

    # --- the one refusal that is about the token's own direction ----------
    if regime is Regime.WEAKENING:
        return _no_trade("token is already reversing")

    if regime is Regime.UNCERTAIN:
        return _no_trade("UNCERTAIN regime: waiting for a clearer read")

    if regime is Regime.STABILIZING:
        return _no_trade(
            "STABILIZING is not a FAST_ENTRY setup; V3-C requires momentum"
        )

    # Everything left is a momentum setup: VERY_EARLY, EARLY, ACCELERATING,
    # EXPLOSIVE or EXTENDED.
    fragile = is_fragile(regime)
    if fragile:
        mods.append("fragile_regime")
    return RouteChoice(
        route=Route.FAST_ENTRY,
        exit_style=ExitStyle.FAST_FAILURE,
        rationale=(
            f"{regime.value}: momentum setup with a decidable read, taken "
            f"under V3-C's decision-time gates"
        ),
        modifiers=tuple(mods),
        style_override=style_override,
    )


def build_exit_plan(
    route: Route,
    regime: Regime,
    style: ExitStyle,
    f: FeatureSet,
    cfg: ExitConfig,
    target_position_pct: float | None = None,
) -> dict[str, object]:
    """
    The exit plan is written BEFORE the trade exists.

    ``invalidation_conditions`` here are the ones the agent would have written
    at T0. Post-entry management may tighten a stop under a predefined
    deterministic policy, but it may not invent a new reason after the fact.
    """
    price = f.price
    invalidation: list[str] = []

    if style in (ExitStyle.FAST_FAILURE, ExitStyle.DISASTER_CAP):
        stop_pct = cfg.fast_failure_stop_pct
        invalidation += [
            f"price trades {stop_pct:.0f}% below fill within "
            f"{cfg.fast_failure_window_s:.0f}s",
            "5m momentum turns negative with sells dominating",
            "liquidity falls below the entry book by more than 25%",
        ]
    elif style is ExitStyle.RUNNER:
        stop_pct = cfg.runner_hard_floor_pct
        invalidation += [
            f"price loses {cfg.runner_hard_floor_pct:.0f}% from fill "
            "(runner floor)",
            f"give-back of {cfg.runner_trail_pct:.0f}% from the running peak",
            "momentum collapse: 30s drop beyond the pre-declared threshold",
            "structure break on the 45s lookback",
        ]
    elif style is ExitStyle.MOMENTUM_FAILURE:
        stop_pct = cfg.momentum_trail_pct
        invalidation += [
            f"give-back of {cfg.momentum_trail_pct:.0f}% from the running peak",
            "momentum failure: 30s move beyond the pre-declared threshold",
            "structure break on the 45s lookback",
        ]
    elif style is ExitStyle.TIME_DECAY:
        stop_pct = cfg.momentum_trail_pct
        invalidation += [
            f"no progress of {cfg.time_decay_no_progress_band_pct:.0f}% within "
            f"{cfg.time_decay_no_progress_s:.0f}s",
            f"give-back of {cfg.momentum_trail_pct:.0f}% from the running peak",
            f"hard timeout at {cfg.time_decay_max_hold_s:.0f}s",
        ]
    else:  # FIXED_TARGET
        stop_pct = cfg.momentum_trail_pct
        invalidation += [
            f"target of +{cfg.fixed_target_pct:.0f}% reached",
            f"give-back of {cfg.momentum_trail_pct:.0f}% from the running peak",
        ]

    invalidation.append(
        f"simulation disaster cap at {cfg.disaster_cap_pct:.0f}% "
        "(boundary, not the strategy)"
    )
    invalidation.append(
        "market data becomes unusable for 120s - exit to a known state rather "
        "than guess"
    )

    return {
        "style": style.value,
        "route": route.value,
        "regime": regime.value,
        "target_position_pct": target_position_pct,
        "initial_stop_pct": stop_pct,
        "invalidation_conditions": invalidation,
        "runner_arm_pct": cfg.runner_arm_pct if style is ExitStyle.RUNNER else None,
        "runner_trail_pct": (
            cfg.runner_trail_pct if style is ExitStyle.RUNNER else None
        ),
        "trail_from_peak_pct": cfg.momentum_trail_pct,
        "momentum_collapse_lookback_s": cfg.momentum_collapse_lookback_s,
        "momentum_collapse_drop_pct": cfg.momentum_collapse_drop_pct,
        "structure_break_lookback_s": cfg.structure_break_lookback_s,
        "structure_break_pct": cfg.structure_break_pct,
        "time_decay_no_progress_s": cfg.time_decay_no_progress_s,
        "time_decay_no_progress_band_pct": cfg.time_decay_no_progress_band_pct,
        "time_decay_max_hold_s": cfg.time_decay_max_hold_s,
        "fixed_target_pct": (
            cfg.fixed_target_pct if style is ExitStyle.FIXED_TARGET else None
        ),
        "disaster_cap_pct": cfg.disaster_cap_pct,
        "data_invalid_exit_s": cfg.data_invalid_exit_s,
        "declared_at_price": price,
    }
