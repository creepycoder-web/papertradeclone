"""
Research memory and route adaptation.

Two rules are absolute here:

1. **Time correctness.** ``snapshot_at(t)`` only ever sees trades whose
   ``exit_timestamp <= t``. A trade that closed at T+60s cannot influence a
   decision made at T. This is enforced by construction (the filter is inside
   the accessor, and it is the only accessor) and unit tested.

2. **No black-box learning.** Route adaptation is a small, readable, deterministic
   estimator: Beta-Bernoulli shrinkage for hit rate, an EWMA for per-trade
   return, and a shrinkage factor that pulls any small sample back to the prior.
   No model is trained, no feature is invented, no trade can rewrite the strategy
   on its own - the prior dominates until the minimum sample size is met.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .models import Route, Trade

# Prior: over many small-sample memecoin paper trades a route wins a bit less
# than half the time. alpha/beta encode a deliberately flat prior so a handful
# of trades cannot tilt the agent.
PRIOR_ALPHA = 2.0
PRIOR_BETA = 2.0
PRIOR_RATE = PRIOR_ALPHA / (PRIOR_ALPHA + PRIOR_BETA)  # 0.5

MIN_SAMPLE = 8          # below this, adaptation is heavily shrunk
SHRINK_K = 12.0         # pseudo-count strength of the pull-to-prior
EWMA_ALPHA = 0.25       # weight on the newest completed trade

MULT_MIN, MULT_MAX = 0.55, 1.35


@dataclass(frozen=True)
class RouteStats:
    """Everything the decision engine is allowed to know about a route so far."""

    key: str
    n: int = 0
    wins: int = 0
    losses: int = 0
    gross_win_usd: float = 0.0
    gross_loss_usd: float = 0.0
    mean_return_pct: float = 0.0
    ewma_return_pct: float = 0.0
    posterior_win_rate: float = PRIOR_RATE
    raw_win_rate: float | None = None
    mean_holding_s: float = 0.0
    runner_capture_pct: float = 0.0
    best_return_pct: float = 0.0
    worst_return_pct: float = 0.0
    total_fees_usd: float = 0.0
    shrinkage: float = 0.0
    multiplier: float = 1.0
    usable: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "n": self.n,
            "wins": self.wins,
            "losses": self.losses,
            "raw_win_rate": self.raw_win_rate,
            "posterior_win_rate": round(self.posterior_win_rate, 4),
            "mean_return_pct": round(self.mean_return_pct, 3),
            "ewma_return_pct": round(self.ewma_return_pct, 3),
            "expectancy_pct": round(self.ewma_return_pct, 3),
            "mean_holding_s": round(self.mean_holding_s, 1),
            "runner_capture_pct": round(self.runner_capture_pct, 2),
            "shrinkage": round(self.shrinkage, 3),
            "multiplier": round(self.multiplier, 3),
            "usable": self.usable,
        }


def _key(route: str, regime: str | None = None) -> str:
    return route if not regime else f"{route}|{regime}"


class ResearchMemory:
    """
    Append-only memory of COMPLETED paper trades.

    ``remember`` is called once per closed trade. Nothing else writes to it.
    """

    def __init__(self, trades: Sequence[Trade] = (), *, ewma_alpha: float = EWMA_ALPHA,
                 min_sample: int = MIN_SAMPLE, shrink_k: float = SHRINK_K) -> None:
        self.ewma_alpha = ewma_alpha
        self.min_sample = min_sample
        self.shrink_k = shrink_k
        self._trades: list[Trade] = sorted(trades, key=lambda t: t.exit_timestamp)
        self._cache: dict[tuple[float, int], dict[str, RouteStats]] = {}

    # -- writing -----------------------------------------------------------

    def remember(self, trade: Trade) -> None:
        self._trades.append(trade)
        self._trades.sort(key=lambda t: t.exit_timestamp)
        self._cache.clear()

    @property
    def trades(self) -> list[Trade]:
        return list(self._trades)

    def __len__(self) -> int:
        return len(self._trades)

    # -- reading (time-correct) -------------------------------------------

    def visible_at(self, t: float) -> list[Trade]:
        """
        The ONLY way to read memory.

        A trade is visible at ``t`` if and only if it had already closed by
        ``t``. Nothing in progress, nothing that closed later.
        """
        return [tr for tr in self._trades if tr.exit_timestamp <= t]

    def stats_at(self, t: float) -> dict[str, RouteStats]:
        cached = self._cache.get((t, len(self._trades)))
        if cached is not None:
            return cached
        out = self._compute(self.visible_at(t))
        self._cache[(t, len(self._trades))] = out
        return out

    def route_at(self, route: Route, t: float, regime: str | None = None) -> RouteStats:
        stats = self.stats_at(t)
        overall = stats.get(route.value, RouteStats(key=route.value))
        if regime:
            scoped = stats.get(_key(route.value, regime))
            if scoped and scoped.n >= self.min_sample:
                return scoped
        return overall

    def route_multiplier(self, route: Route, t: float, regime: str | None = None) -> tuple[float, int]:
        s = self.route_at(route, t, regime)
        return s.multiplier, s.n

    # -- estimation --------------------------------------------------------

    def _compute(self, trades: Iterable[Trade]) -> dict[str, RouteStats]:
        buckets: dict[str, list[Trade]] = defaultdict(list)
        for tr in trades:
            buckets[tr.route].append(tr)
            if tr.regime:
                buckets[_key(tr.route, tr.regime)].append(tr)
        return {k: self._stats(k, v) for k, v in buckets.items()}

    def _stats(self, key: str, trades: list[Trade]) -> RouteStats:
        n = len(trades)
        if n == 0:
            return RouteStats(key=key)

        wins = [t for t in trades if t.net_pnl_usd > 0]
        losses = [t for t in trades if t.net_pnl_usd <= 0]
        win_ret = [t.return_pct for t in wins]
        all_ret = [t.return_pct for t in trades]

        ewma = 0.0
        for t in sorted(trades, key=lambda x: x.exit_timestamp):
            r = t.return_pct
            ewma = self.ewma_alpha * r + (1 - self.ewma_alpha) * ewma

        gross_win = sum(t.net_pnl_usd for t in wins)
        gross_loss = abs(sum(t.net_pnl_usd for t in losses))

        # runners: winners that were given room to run (>+50% and held >120s)
        runners = [t for t in wins if t.return_pct >= 50 and t.holding_seconds >= 120]
        capture = (
            (sum(t.net_pnl_usd for t in runners) / gross_win * 100.0)
            if gross_win > 0
            else 0.0
        )

        posterior = (len(wins) + PRIOR_ALPHA) / (n + PRIOR_ALPHA + PRIOR_BETA)
        raw_wr = len(wins) / n
        # shrinkage: 0 until MIN_SAMPLE, ramping to 1 over the next SHRINK_K
        if n <= self.min_sample:
            shrink = 0.0
        else:
            shrink = min(1.0, (n - self.min_sample) / self.shrink_k)

        blended = (1 - shrink) * PRIOR_RATE + shrink * posterior
        if blended <= 0:
            mult = MULT_MIN
        else:
            mult = blended / PRIOR_RATE

        # blend the multiplier with 1.0 by the same shrinkage so an unused route
        # never gets a strong opinion
        multiplier = 1.0 + shrink * (mult - 1.0)
        multiplier = max(MULT_MIN, min(MULT_MAX, multiplier))

        return RouteStats(
            key=key,
            n=n,
            wins=len(wins),
            losses=len(losses),
            gross_win_usd=gross_win,
            gross_loss_usd=gross_loss,
            mean_return_pct=(sum(all_ret) / n),
            ewma_return_pct=ewma,
            posterior_win_rate=posterior,
            raw_win_rate=raw_wr,
            mean_holding_s=sum(t.holding_seconds for t in trades) / n,
            runner_capture_pct=capture,
            best_return_pct=max(all_ret),
            worst_return_pct=min(all_ret),
            total_fees_usd=sum(t.entry_fee_usd + t.exit_fee_usd for t in trades),
            shrinkage=shrink,
            multiplier=multiplier,
            usable=n >= self.min_sample,
        )

    # -- reporting ---------------------------------------------------------

    def summary_at(self, t: float) -> dict[str, Any]:
        stats = self.stats_at(t)
        return {
            "as_of": t,
            "visible_trades": len(self.visible_at(t)),
            "total_trades": len(self._trades),
            "routes": {
                k: v.as_dict() for k, v in sorted(stats.items()) if "|" not in k
            },
        }

    def lessons(self, t: float) -> list[str]:
        """
        Human-readable notes the agent carries forward.

        Only drawn from trades already closed at ``t``.
        """
        stats = self.stats_at(t)
        out: list[str] = []
        for route, s in sorted(stats.items()):
            if "|" in route:
                continue
            if not s.usable:
                out.append(f"{route}: {s.n} closed trade(s) - not enough to adapt (need {self.min_sample})")
                continue
            direction = "outperforming" if s.multiplier > 1.03 else (
                "underperforming" if s.multiplier < 0.97 else "neutral"
            )
            out.append(
                f"{route}: n={s.n} posterior_win={s.posterior_win_rate:.2f} "
                f"ewma_ret={s.ewma_return_pct:+.1f}% -> {direction} (x{s.multiplier:.2f})"
            )
        for key, s in sorted(stats.items()):
            if "|" not in key or not s.usable:
                continue
            if s.ewma_return_pct < -5:
                out.append(f"{key}: n={s.n} ewma_ret={s.ewma_return_pct:+.1f}% - avoid this combination")
            elif s.ewma_return_pct > 20:
                out.append(f"{key}: n={s.n} ewma_ret={s.ewma_return_pct:+.1f}% - preserve runners here")
        return out


def bucket_age(age_s: float | None) -> str | None:
    if age_s is None:
        return None
    if age_s < 300:
        return "<5m"
    if age_s < 3600:
        return "5m-1h"
    if age_s < 21600:
        return "1h-6h"
    if age_s < 86400:
        return "6h-24h"
    return ">24h"


def bucket_mcap(mc: float | None) -> str | None:
    if mc is None:
        return None
    if mc < 50_000:
        return "<50k"
    if mc < 150_000:
        return "50k-150k"
    if mc < 500_000:
        return "150k-500k"
    if mc < 2_000_000:
        return "500k-2m"
    if mc < 10_000_000:
        return "2m-10m"
    return ">10m"


def bucket_liq(liq: float | None) -> str | None:
    if liq is None:
        return None
    if liq < 10_000:
        return "<10k"
    if liq < 30_000:
        return "10k-30k"
    if liq < 100_000:
        return "30k-100k"
    if liq < 500_000:
        return "100k-500k"
    return ">500k"
