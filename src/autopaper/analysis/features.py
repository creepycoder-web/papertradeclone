"""
Feature extraction.

Every feature is derived from data observed at or before ``decision_time``. The
``as_of`` gate is applied here, once, so no downstream module can accidentally
read a future price. ``FeatureSet`` records the horizon it was computed over so
the decision report can prove the bound.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

# Concrete detail in narrative prose: years, percentages, capitalised terms.
# Used only to tell "a real narrative" from "a sentence of nothing".
_SPECIFIC_RE = re.compile(
    r"(\b(19|20)\d{2}\b)|(\b\d+(?:\.\d+)?\s*%)|(\b[A-Z][a-zA-Z]{3,}\b)"
)

from ..models import AlertSnapshot, DataState, NarSnapshot, Observation, ObservationSet
from ..v2floor import V2F_MIN_VOL5_PER_MC, f_vol5_per_mc


def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    try:
        return a / b
    except ZeroDivisionError:
        return None


def _pct(a: float | None, b: float | None) -> float | None:
    r = _safe_div(a, b)
    return None if r is None else (r - 1.0) * 100.0


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class FeatureSet:
    # provenance
    candidate_id: str
    mint: str
    decision_time: float
    data_state: DataState
    observation_count: int
    history_span_s: float | None
    newest_observed_at: float | None

    # price
    price: float | None = None
    price_at_t0: float | None = None
    first_valid_price: float | None = None
    extension_from_first_valid_price: float | None = None
    change_5m: float | None = None
    change_15m: float | None = None
    change_1h: float | None = None
    change_since_t0_pct: float | None = None
    momentum_30s_pct: float | None = None
    momentum_60s_pct: float | None = None
    # A momentum value is only meaningful when the corresponding observation
    # window is actually covered.  These fields make that distinction explicit
    # in persisted decision evidence instead of treating a short two-second
    # tape as a zero/available 30s or 60s signal.
    momentum_30s_available: bool = False
    momentum_60s_available: bool = False
    momentum_30s_source: str = "unavailable"
    momentum_60s_source: str = "unavailable"
    # Change across whatever span we actually have. On a token that launched two
    # minutes ago there is no 5m or 15m window at all, and a classifier that only
    # reads fixed long windows would call every live memecoin UNCERTAIN.
    change_over_span_pct: float | None = None
    change_2s_pct: float | None = None
    realized_vol_60s: float | None = None
    drawdown_from_high_pct: float | None = None
    high_to_current_pct: float | None = None

    # size / structure
    market_cap_usd: float | None = None
    liquidity_usd: float | None = None
    volume_5m_usd: float | None = None
    volume_1h_usd: float | None = None
    volume_24h_usd: float | None = None
    volume_to_liquidity: float | None = None
    liq_to_mcap: float | None = None
    #: Trailing-5m volume over market cap. The V2_FLOOR relative-turnover
    #: floor. ``None`` when either operand is missing or non-positive --
    #: computed by :func:`autopaper.v2floor.f_vol5_per_mc`, the same
    #: fail-closed helper the gate itself uses, so the feature and the gate can
    #: never disagree about whether the ratio is computable.
    vol5_per_mc: float | None = None
    token_age_s: float | None = None

    # activity
    buys_5m: int | None = None
    sells_5m: int | None = None
    txns_5m: int | None = None
    txns_1h: int | None = None
    buy_sell_ratio: float | None = None
    sells_per_min_5m: float | None = None

    # data quality
    data_age_ms: float | None = None
    freshness: float = 0.0
    alert_has_price: bool = False
    alert_fields_present: int = 0
    nar_available: bool = False
    nar_fields_present: int = 0
    extension_anchored: bool = False
    # The Phanes bot supplies prose, not metrics. These record what is
    # legitimately observable about that prose - that a narrative exists, and
    # how much concrete detail it contains. Neither is a hype score.
    narrative_chars: int = 0
    narrative_specifics: int = 0
    # NO TRAILING COMMA. ``narrative_placeholder: bool = False,`` makes the
    # default the TUPLE ``(False,)``, which is TRUTHY -- so every FeatureSet
    # built without this field explicitly read as "the narrative bot replied
    # with a placeholder" and carried the -0.06 signal-quality discount.
    narrative_placeholder: bool = False
    # Flow evidence, PERSISTED so a future audit can prove how signal_quality
    # was composed. These were computed for the +0.06 flow bonus but never
    # stored, so it was impossible to verify the bonus on any given record.
    # Historical rows are NOT rewritten or back-filled: a fabricated value would
    # be worse than a known gap.
    nar_buys: int | None = None
    nar_sells: int | None = None
    nar_has_flow: bool = False

    # derived quality
    signal_quality: float = 0.0
    liquidity_score: float = 0.0
    activity_score: float = 0.0
    momentum_score: float = 0.0
    structure_score: float = 0.0
    extension_score: float = 0.0

    nar: Mapping[str, Any] = field(default_factory=dict)
    alert: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {
            k: v
            for k, v in asdict(self).items()
            if k not in ("nar", "alert")
        }


# ---------------------------------------------------------------------------
# history helpers (all pre-filtered by as_of)
# ---------------------------------------------------------------------------


def _price_at_window_start(
    history: ObservationSet, t: float, window_s: float, *,
    require_full_window: bool = False,
) -> float | None:
    """
    The price as of the START of the lookback window.

    This must be the OLDEST usable print in ``[t - window_s, t]``, not the
    newest. Returning the newest (which is the current price) would make every
    windowed change exactly zero and silently disable the entire momentum axis -
    a bug that looks like "the market is never moving" rather than like a bug.
    If the window holds no print, fall back to the newest print at or before
    ``t``, which yields no change rather than a fabricated one.
    """
    pool = [
        o
        for o in history.as_of(t)
        if o.price_usd and o.price_usd > 0 and o.observed_at >= t - window_s
    ]
    if require_full_window and (history.span_s() or 0.0) < window_s:
        return None
    if pool:
        return pool[0].price_usd
    before = [o for o in history.as_of(t) if o.price_usd and o.price_usd > 0
              and o.observed_at < t - window_s]
    if before:
        return before[-1].price_usd
    latest = history.latest(t)
    return latest.price_usd if latest and latest.price_usd else None


def _series(history: ObservationSet, t: float, window_s: float) -> list[tuple[float, float]]:
    return [
        (ts, px)
        for ts, px in history.prices(t)
        if ts >= t - window_s
    ]


def _stdev(values: list[float]) -> float | None:
    if len(values) < 3:
        return None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


# ---------------------------------------------------------------------------
# main builder
# ---------------------------------------------------------------------------


def build_features(
    *,
    candidate_id: str,
    mint: str,
    decision_time: float,
    history: ObservationSet,
    alert: AlertSnapshot | None,
    nar: NarSnapshot | None,
    stale_after_ms: float,
    unusable_after_ms: float,
) -> FeatureSet:
    # ---- the no-lookahead gate -----------------------------------------
    past = history.as_of(decision_time)
    latest = past.latest()

    data_state = (
        latest.freshness(decision_time, stale_after_ms, unusable_after_ms)
        if latest is not None
        else DataState.MISSING
    )
    data_age_ms = latest.age_ms(decision_time) if latest is not None else None

    price = latest.price_usd if latest is not None else None
    alert_price = alert.num("price_usd") if alert is not None else None
    # The alert's price is itself an observation at T0 (it is what the alert
    # reported), so it is legitimately inside the decision-time information set.
    if price is None and alert_price:
        price = alert_price

    # The anchor for "extension from first valid price" is the earliest price
    # this agent could actually have acted on, resolved by TIME:
    #   1. a registered anchor (the first print the agent ever saw for this mint)
    #   2. an explicit first price supplied by the alert
    #   3. whichever came first in wall-clock terms - the alert's own price, or
    #      the oldest observation already in our history
    #   4. failing all of that, the current price, which means extension is 0 by
    #      definition and the score is capped (see _extension_score)
    #
    # ANCHORED IS TRACKED EXPLICITLY, not inferred from ``price != fvp``.
    # Inferring it conflated two different states: "there is no earlier price"
    # (case 4) and "there IS an earlier price and the token has not moved since"
    # -- which is extension 0%, the LEAST extended entry the ladder can see. A
    # token sitting flat on its first observed price was therefore reported
    # unanchored, scored 0.35 instead of 1.0, raised ``extension_unanchored``,
    # and lost the chaseable band in ``select_route``.
    fvp = None
    fvp_anchored = False
    if latest is not None and latest.first_valid_price:
        fvp = latest.first_valid_price
        fvp_anchored = True
    if fvp is None and alert is not None:
        explicit = alert.num("first_valid_price")
        if explicit:
            fvp = explicit
            fvp_anchored = True
    if fvp is None:
        candidates: list[tuple[float, float]] = []
        if alert is not None and alert_price:
            candidates.append((alert.received_timestamp, alert_price))
        oldest_first = list(past)[0] if len(past) else None
        if oldest_first is not None and oldest_first.price_usd:
            candidates.append((oldest_first.observed_at, oldest_first.price_usd))
        if candidates:
            anchor_at, anchor_price = min(candidates, key=lambda c: c[0])
            # An anchor has to be EARLIER than the price it is measured against.
            # The newest print cannot anchor itself: with one observation and no
            # alert price there is genuinely no earlier price to measure from.
            if latest is not None and anchor_at < latest.observed_at:
                fvp = anchor_price
                fvp_anchored = True
    if fvp is None and price:
        fvp = price

    def best(field_name: str) -> float | None:
        """
        Freshest market observation first, then the alert, then the NAR card.

        Order matters: a live venue print beats a number the alert published at
        T0, and both beat a figure relayed by a third-party narrative bot.
        """
        v = getattr(latest, field_name, None) if latest is not None else None
        if v is not None:
            return v
        v = alert.num(field_name) if alert is not None else None
        if v is not None:
            return v
        return getattr(nar, field_name, None) if nar is not None else None

    liq = best("liquidity_usd")
    mcap = best("market_cap_usd") or best("fdv_usd")
    if liq is None and nar is not None:
        liq = nar.liquidity_usd
    vol5 = best("volume_5m_usd") or (nar.volume_usd if nar is not None else None)
    vol1h = best("volume_1h_usd")
    vol24 = best("volume_24h_usd")

    token_age = latest.token_age_s if latest is not None else None
    if token_age is None and alert is not None:
        token_age = alert.num("age_seconds")
    if token_age is None and nar is not None:
        token_age = nar.age_s

    change_5m = best("price_change_5m")
    change_1h = best("price_change_1h") or (nar.change_1h_pct if nar else None)
    change_24h = best("price_change_24h")
    buys = best("buys_5m")
    sells = best("sells_5m")
    txns5 = best("txns_5m")
    txns1h = best("txns_1h")
    # the NAR card carries genuine 1h buy/sell counts, which no price endpoint
    # reports. They are a far better read on real demand than a price change.
    nar_buys = getattr(nar, "buys_1h", None) if nar is not None else None
    nar_sells = getattr(nar, "sells_1h", None) if nar is not None else None
    if buys is None and nar_buys is not None:
        buys = nar_buys
    if sells is None and nar_sells is not None:
        sells = nar_sells

    # 15m change derived from our own series when the venue does not supply it.
    #
    # ``require_full_window``, exactly like the 30s/60s momentum below. Without
    # it a token observed for ten seconds has no 15m window, the window-start
    # helper falls back to the newest print, and the result is a FABRICATED
    # ``change_15m == 0.0``. ``regime.classify`` ranks 15m ABOVE the 60s/30s
    # windows, so that zero outranked the momentum actually measured and a
    # live launch read as flat -- which is how a moving token was classified
    # STABILIZING and refused.
    change_15m = None
    p15 = _price_at_window_start(
        past, decision_time, 900.0, require_full_window=True
    )
    if price and p15 and p15 > 0:
        change_15m = (price - p15) / p15 * 100.0

    all_prices = [(o.observed_at, o.price_usd) for o in past
                  if o.price_usd and o.price_usd > 0]
    change_span = None
    if len(all_prices) >= 2 and all_prices[0][1] > 0:
        change_span = (all_prices[-1][1] - all_prices[0][1]) / all_prices[0][1] * 100.0
    # The 2s window start, NOT the oldest print in the whole history. Measuring
    # from ``all_prices[0]`` made this field a duplicate of
    # ``change_over_span_pct`` under a name that says two seconds.
    change_2s = None
    p2 = _price_at_window_start(past, decision_time, 2.0)
    if price and p2 and p2 > 0:
        change_2s = (price - p2) / p2 * 100.0

    mom30 = None
    p30 = _price_at_window_start(
        past, decision_time, 30.0, require_full_window=True
    )
    if price and p30 and p30 > 0:
        mom30 = (price - p30) / p30 * 100.0
    mom60 = None
    p60 = _price_at_window_start(
        past, decision_time, 60.0, require_full_window=True
    )
    if price and p60 and p60 > 0:
        mom60 = (price - p60) / p60 * 100.0

    series60 = _series(past, decision_time, 60.0)
    rets = [
        (series60[i][1] - series60[i - 1][1]) / series60[i - 1][1]
        for i in range(1, len(series60))
        if series60[i - 1][1] > 0
    ]
    vol_60 = (_stdev(rets) or 0.0) * 100.0 if len(rets) >= 3 else None

    dd_from_high = None
    if price:
        prices_all = [px for _ts, px in past.prices(decision_time) if px > 0]
        if alert_price and alert_price > 0:
            prices_all.append(alert_price)
        if prices_all:
            hi = max(prices_all)
            dd_from_high = (price - hi) / hi * 100.0

    bsr = _safe_div(buys, sells) if (buys is not None and sells is not None) else None
    if buys == 0 and sells == 0:
        bsr = None
    spm = _safe_div(sells, 5.0) if sells is not None else None

    alert_fields = len(alert.fields) if alert is not None else 0
    nar_fields = (
        sum(
            1
            for k, v in (nar.as_dict().items() if nar else [])
            if v not in (None, False, [], {})
        )
        if nar is not None
        else 0
    )

    freshness = _freshness_score(data_age_ms, stale_after_ms, unusable_after_ms,
                                 data_state, past)

    liq_score = _liquidity_score(liq, mcap, vol5)
    act_score = _activity_score(txns5, bsr, spm, vol5, liq)
    mom_score = _momentum_score(change_5m, change_15m, mom30, mom60, vol_60)
    struct_score = _structure_score(liq, mcap, token_age, bsr, vol5, txns1h)
    ext_score, extension, anchored = _extension_score(
        price, fvp, dd_from_high, change_5m, anchored=fvp_anchored
    )

    quality = _signal_quality(
        data_state=data_state,
        freshness=freshness,
        liq_score=liq_score,
        act_score=act_score,
        mom_score=mom_score,
        struct_score=struct_score,
        ext_score=ext_score,
        alert_fields=alert_fields,
        nar_available=bool(nar and nar.available),
        span_s=past.span_s(),
        narrative_placeholder=(
            bool(nar is not None and nar.error and "PLACEHOLDER" in nar.error)
        ),
        nar_has_flow=bool(nar_buys is not None or nar_sells is not None),
    )

    return FeatureSet(
        candidate_id=candidate_id,
        mint=mint,
        decision_time=decision_time,
        data_state=data_state,
        observation_count=len(past),
        history_span_s=past.span_s(),
        newest_observed_at=latest.observed_at if latest is not None else None,
        price=price,
        price_at_t0=alert_price,
        first_valid_price=fvp,
        extension_from_first_valid_price=extension,
        change_5m=change_5m,
        change_15m=change_15m,
        change_1h=change_1h,
        change_since_t0_pct=(
            (price - alert_price) / alert_price * 100.0
            if price and alert_price and alert_price > 0
            else None
        ),
        momentum_30s_pct=mom30,
        momentum_60s_pct=mom60,
        momentum_30s_available=mom30 is not None,
        momentum_60s_available=mom60 is not None,
        momentum_30s_source=("observations" if mom30 is not None else "unavailable"),
        momentum_60s_source=("observations" if mom60 is not None else "unavailable"),
        change_over_span_pct=change_span,
        change_2s_pct=change_2s,
        realized_vol_60s=vol_60,
        drawdown_from_high_pct=dd_from_high,
        high_to_current_pct=dd_from_high,
        market_cap_usd=mcap,
        liquidity_usd=liq,
        volume_5m_usd=vol5,
        volume_1h_usd=vol1h,
        volume_24h_usd=vol24,
        volume_to_liquidity=_safe_div(vol5, liq),
        liq_to_mcap=_safe_div(liq, mcap),
        vol5_per_mc=f_vol5_per_mc(vol5, mcap),
        token_age_s=token_age,
        buys_5m=int(buys) if buys is not None else None,
        sells_5m=int(sells) if sells is not None else None,
        txns_5m=int(txns5) if txns5 is not None else None,
        txns_1h=int(txns1h) if txns1h is not None else None,
        buy_sell_ratio=bsr,
        sells_per_min_5m=spm,
        data_age_ms=data_age_ms,
        freshness=freshness,
        alert_has_price=alert_price is not None,
        alert_fields_present=alert_fields,
        nar_available=bool(nar and nar.available),
        nar_fields_present=nar_fields,
        extension_anchored=anchored,
        narrative_chars=len(nar.narrative or "") if nar is not None else 0,
        narrative_specifics=(
            len(set(m.group(0).lower() for m in _SPECIFIC_RE.finditer(nar.narrative or "")))
            if nar is not None and nar.narrative else 0
        ),
        narrative_placeholder=bool(
            nar is not None and nar.error and "PLACEHOLDER" in nar.error
        ),
        # PERSISTED so a future audit can prove exactly how signal_quality was
        # composed. These were computed for the +0.06 flow bonus but never
        # stored, so it was impossible to verify the contribution on any given
        # record. The definition matches the one used for the bonus itself.
        nar_buys=(int(nar_buys) if nar_buys is not None else None),
        nar_sells=(int(nar_sells) if nar_sells is not None else None),
        nar_has_flow=bool(nar_buys is not None or nar_sells is not None),
        signal_quality=quality,
        liquidity_score=liq_score,
        activity_score=act_score,
        momentum_score=mom_score,
        structure_score=struct_score,
        extension_score=ext_score,
        nar=dict(nar.as_dict()) if nar is not None else {},
        alert=dict(alert.as_dict()) if alert is not None else {},
    )


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def _freshness_score(age_ms, stale_ms, unusable_ms, state, past) -> float:
    if state is DataState.MISSING:
        return 0.0
    if age_ms is None:
        return 0.3
    if state is DataState.LIVE:
        return _clip(1.0 - (age_ms / max(stale_ms, 1.0)) * 0.5)
    if state is DataState.STALE:
        return _clip(0.35 * (1.0 - (age_ms - stale_ms) / max(unusable_ms - stale_ms, 1.0)))
    return 0.0


def _liquidity_score(liq, mcap, vol5) -> float:
    if liq is None or liq <= 0:
        return 0.0
    # 8k -> 0.25, 25k -> 0.5, 100k -> 0.85, 500k+ -> 1.0 (log-ish ramp)
    base = _clip(math.log10(max(liq, 1.0)) / math.log10(1_000_000.0))
    liq_mcap = _safe_div(liq, mcap)
    penalty = 0.0
    if liq_mcap is not None:
        # a pool holding <3% of mcap in liquidity is fragile
        if liq_mcap < 0.02:
            penalty = 0.25
        elif liq_mcap < 0.05:
            penalty = 0.10
    thin_vol = False
    if vol5 is not None and liq > 0:
        if vol5 / liq < 0.10:
            thin_vol = True
    if thin_vol:
        penalty += 0.10
    return _clip(base - penalty)


def _activity_score(txns5, bsr, spm, vol5, liq) -> float:
    if txns5 is None and vol5 is None:
        return 0.25
    score = 0.0
    if txns5 is not None:
        score += _clip(math.log10(max(txns5, 1)) / math.log10(500.0)) * 0.45
    if bsr is not None:
        # 1.0 balanced, 3.0 strong, >6 suspicious one-sided
        if bsr < 1.0:
            score += 0.10
        elif bsr <= 2.0:
            score += 0.30
        elif bsr <= 4.0:
            score += 0.25
        elif bsr <= 6.0:
            score += 0.12
        else:
            score += 0.0
    # Heavy sell pressure, but only when sells actually dominate. Applying this
    # penalty to the raw sell count double-counts the same evidence the buy/sell
    # ratio already scored, and would punish a healthy 3:1 book purely for
    # being busy - which is what a working market looks like.
    if spm is not None and spm > 12 and (bsr is None or bsr < 1.0):
        score -= 0.15
    if vol5 is not None and liq:
        ratio = vol5 / liq
        if ratio > 3.0:
            score -= 0.10  # churn far beyond the pool size is not real demand
    return _clip(score)


def _momentum_score(c5, c15, m30, m60, vol60) -> float:
    score = 0.0
    if c5 is not None:
        if c5 > 60:
            score += 0.30
        elif c5 > 25:
            score += 0.42
        elif c5 > 8:
            score += 0.35
        elif c5 > 0:
            score += 0.20
        elif c5 > -5:
            score += 0.10
        else:
            score += 0.0
    if m30 is not None:
        if m30 > 4:
            score += 0.20
        elif m30 > 0:
            score += 0.10
        elif m30 < -8:
            score -= 0.15
    if m60 is not None and m60 > 0:
        score += 0.10
    if vol60 is not None and vol60 > 25:
        score -= 0.05  # chaotic, harder to hold
    return _clip(score)


def _structure_score(liq, mcap, age, bsr, vol5, txns1h) -> float:
    score = 0.4
    if liq and mcap:
        ratio = liq / mcap
        if 0.03 <= ratio <= 0.35:
            score += 0.25
        elif ratio < 0.02:
            score -= 0.15
    if age is not None:
        if age < 120:
            score += 0.05   # genuinely early
        elif age > 86_400:
            score -= 0.15   # not a fresh launch, momentum may be stale
    if bsr is not None and bsr < 0.9:
        score -= 0.20
    if txns1h is not None and txns1h == 0 and vol5 is not None and vol5 > 0:
        score -= 0.10
    return _clip(score)


def _extension_score(price, fvp, dd_from_high, c5, *, anchored=None):
    """
    The anti-exhaustion axis.

    Returns ``(score, extension, anchored)``. ``score`` is HIGH when we are NOT
    extended. ``extension`` is the fraction above the first valid price we could
    have traded.

    ``anchored`` is False when the "first valid price" is just the current price
    - i.e. we have no history at all. In that case the score is deliberately
    capped below its maximum: claiming a perfect 1.0 extension score off a single
    print would be the most flattering possible reading of no information.

    THE CALLER DECIDES WHETHER AN ANCHOR EXISTS. ``build_features`` knows which
    of its four resolution steps produced ``fvp`` and passes the answer in. The
    old local inference - ``abs(price - fvp) > 0`` - could not tell "no earlier
    price" from "an earlier price that happens to equal this one", and the
    second of those is extension 0%, the least extended entry there is. The
    inference is retained only for a caller that supplies no flag, so a direct
    call written before this parameter existed behaves as it did.
    """
    usable = bool(fvp and price and fvp > 0)
    if anchored is None:
        anchored = usable and abs(price - fvp) > 0
    else:
        anchored = bool(anchored) and usable
    extension = (price - fvp) / fvp if anchored else None
    if extension is None:
        return 0.35, None, False
    unanchored_cap = 0.6
    if extension < 0.10:
        score = 1.0
    elif extension < 0.25:
        score = 0.85
    elif extension < 0.40:
        score = 0.65
    elif extension < 0.60:
        score = 0.40
    elif extension < 1.00:
        score = 0.20
    elif extension < 1.60:
        score = 0.08
    else:
        score = 0.0
    # already off its high
    if dd_from_high is not None:
        if dd_from_high < -35:
            score *= 0.45
        elif dd_from_high < -20:
            score *= 0.75
        elif dd_from_high > -5 and (c5 or 0) < 3:
            score *= 0.80  # flat at the high = momentum gone
    if not anchored:
        score = min(score, unanchored_cap)
    return _clip(score), extension, anchored


def _signal_quality(
    *,
    data_state: DataState,
    freshness: float,
    liq_score: float,
    act_score: float,
    mom_score: float,
    struct_score: float,
    ext_score: float,
    alert_fields: int,
    nar_available: bool,
    span_s: float | None,
    narrative_placeholder: bool = False,
    nar_has_flow: bool = False,
) -> float:
    """
    How much evidence do we actually have?

    Not a view on whether the token is good - a view on whether the evidence is
    sufficient to act on at all. Weighted so that stale/missing data dominates
    the penalty: with weak data the answer should be WAIT or REJECT.
    """
    if data_state is DataState.MISSING:
        return 0.0
    if data_state is DataState.UNUSABLE:
        return 0.05

    base = (
        0.28 * liq_score
        + 0.20 * act_score
        + 0.18 * mom_score
        + 0.14 * struct_score
        + 0.20 * ext_score
    )
    quality = base * (0.55 + 0.45 * freshness)

    # a history with real depth is worth something
    if span_s is not None:
        if span_s >= 120:
            quality += 0.05
        elif span_s < 15:
            quality -= 0.05
    else:
        quality -= 0.08  # a single print is not a market read

    if alert_fields >= 6:
        quality += 0.04
    if nar_available:
        quality += 0.05
    # a card carrying real 1h flow counts is materially better evidence than
    # prose: it is the only source of buy/sell pressure we have.
    if nar_has_flow:
        quality += 0.06
    # A placeholder narrative is the bot saying it has nothing. That is weaker
    # than no reply at all, because it looks like evidence. Discount it rather
    # than letting a "narrative found" tick stand in for missing metrics.
    if narrative_placeholder:
        quality -= 0.06
    if data_state is DataState.STALE:
        quality *= 0.55
    # Rounded to 4 decimals BEFORE the value is compared against a threshold.
    #
    # This is a sum of products, so a mathematically-exact 0.55 arrives as
    # 0.5499999999999999 and V3-C's floor would refuse it. Four decimals is
    # thirteen orders of magnitude finer than any real difference between two
    # tokens' evidence, so rounding here costs no information and makes the
    # exact boundary deterministic instead of float-noise dependent.
    return round(_clip(quality), 4)
