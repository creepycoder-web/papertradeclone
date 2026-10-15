"""
Trade statistics.

Everything here is derived from the append-only ledger. Nothing is excluded:
losing trades, unfilled attempts, stale-data periods and rejected candidates are
all counted. The only thing this module will not do is agree with a number that
would look better.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .models import Trade


@dataclass
class Stats:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    scratches: int = 0
    win_rate: float = 0.0
    realized_pnl_usd: float = 0.0
    total_return_pct: float = 0.0
    expectancy_usd: float = 0.0
    expectancy_pct: float = 0.0
    profit_factor: float = 0.0
    avg_winner_usd: float = 0.0
    avg_loser_usd: float = 0.0
    largest_winner_usd: float = 0.0
    largest_loser_usd: float = 0.0
    avg_holding_s: float = 0.0
    fees_paid_usd: float = 0.0
    slippage_paid_usd: float = 0.0
    runner_capture_pct: float = 0.0
    #: runners' realised dollars / total winner dollars. See the long comment at
    #: the assignment: this is NOT MFE capture and must not be labelled as such.
    runner_pnl_share_pct: float | None = None
    #: True MFE capture: mean(realized_return / MFE) over runners, in percent.
    #: ``None`` when there is no runner with a usable MFE -- never 0.0.
    mfe_capture_pct: float | None = None
    mfe_capture_population: int = 0
    mfe_capture_basis: str = ""
    mfe_capture_executable_pct: float | None = None
    #: "realized_trade_to_trade" today. Named so a reader is never left thinking
    #: the number covers open-position risk.
    max_drawdown_scope: str = "realized_trade_to_trade"
    runners: int = 0
    max_drawdown_pct: float = 0.0
    max_consecutive_losses: int = 0
    current_losing_streak: int = 0
    avg_mfe_pct: float = 0.0
    avg_mae_pct: float = 0.0
    giveback_pct: float = 0.0
    by_route: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_regime: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_exit_style: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_age_bucket: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_mc_bucket: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_liq_bucket: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Per-strategy-profile stats. L25_CURRENT / L25_60S / L25_TP100 are distinct
    #: strategies and must never be pooled into one headline.
    by_strategy_profile: dict[str, dict[str, Any]] = field(default_factory=dict)
    strategy_profiles_present: list[str] = field(default_factory=list)
    mixed_strategy_population: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "realized_pnl_usd": round(self.realized_pnl_usd, 4),
            "total_return_pct": round(self.total_return_pct, 3),
            "expectancy_usd": round(self.expectancy_usd, 4),
            "expectancy_pct": round(self.expectancy_pct, 3),
            "profit_factor": round(self.profit_factor, 3),
            "avg_winner_usd": round(self.avg_winner_usd, 4),
            "avg_loser_usd": round(self.avg_loser_usd, 4),
            "largest_winner_usd": round(self.largest_winner_usd, 4),
            "largest_loser_usd": round(self.largest_loser_usd, 4),
            "avg_holding_s": round(self.avg_holding_s, 1),
            "fees_paid_usd": round(self.fees_paid_usd, 4),
            "slippage_paid_usd": round(self.slippage_paid_usd, 4),
            "runners": self.runners,
            "runner_pnl_share_pct": (
                None if self.runner_pnl_share_pct is None
                else round(self.runner_pnl_share_pct, 2)),
            "runner_capture_pct": (  # DEPRECATED alias of the above
                None if self.runner_pnl_share_pct is None
                else round(self.runner_pnl_share_pct, 2)),
            "mfe_capture_pct": self.mfe_capture_pct,
            "mfe_capture_population": self.mfe_capture_population,
            "mfe_capture_basis": self.mfe_capture_basis,
            "mfe_capture_executable_pct": self.mfe_capture_executable_pct,
            "max_drawdown_scope": self.max_drawdown_scope,
            "strategy_profiles_present": list(self.strategy_profiles_present),
            "mixed_strategy_population": self.mixed_strategy_population,
            "by_strategy_profile": self.by_strategy_profile,
            "max_drawdown_pct": round(self.max_drawdown_pct, 3),
            "max_consecutive_losses": self.max_consecutive_losses,
            "current_losing_streak": self.current_losing_streak,
            "avg_mfe_pct": round(self.avg_mfe_pct, 2),
            "avg_mae_pct": round(self.avg_mae_pct, 2),
            "avg_giveback_pct": round(self.giveback_pct, 2),
        }


def _group(trades: Sequence[Trade], key) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        k = key(t)
        if k is not None:
            buckets[str(k)].append(t)
    out: dict[str, dict[str, Any]] = {}
    for k, group in sorted(buckets.items()):
        wins = [t for t in group if t.net_pnl_usd > 0]
        out[k] = {
            "n": len(group),
            "wins": len(wins),
            "win_rate": round(len(wins) / len(group), 3),
            "net_pnl_usd": round(sum(t.net_pnl_usd for t in group), 4),
            "avg_return_pct": round(sum(t.return_pct for t in group) / len(group), 2),
        }
    return out


def compute(
    trades: Sequence[Trade],
    *,
    starting_equity: float = 100.0,
    current_equity: float | None = None,
    _profile_breakdown: bool = True,
) -> Stats:
    """
    Canonical PAPER statistics for a set of closed trades.

    ``_profile_breakdown`` is internal: the per-profile recursion calls this with
    it False so a nested block does not try to break itself down again. It is
    not part of the public contract.
    """
    trades = sorted(trades, key=lambda t: t.exit_timestamp)
    s = Stats(total_trades=len(trades))
    if not trades:
        return s

    wins = [t for t in trades if t.net_pnl_usd > 0]
    losses = [t for t in trades if t.net_pnl_usd < 0]
    scratches = [t for t in trades if t.net_pnl_usd == 0]
    s.wins, s.losses, s.scratches = len(wins), len(losses), len(scratches)
    s.win_rate = len(wins) / len(trades)
    s.realized_pnl_usd = sum(t.net_pnl_usd for t in trades)
    s.total_return_pct = (
        s.realized_pnl_usd / starting_equity * 100.0 if starting_equity else 0.0
    )
    s.expectancy_usd = s.realized_pnl_usd / len(trades)
    s.expectancy_pct = sum(t.return_pct for t in trades) / len(trades)
    gross_win = sum(t.net_pnl_usd for t in wins)
    gross_loss = abs(sum(t.net_pnl_usd for t in losses))
    s.profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_win > 0 else 0.0
    )
    if wins:
        s.avg_winner_usd = gross_win / len(wins)
        s.largest_winner_usd = max(t.net_pnl_usd for t in wins)
    if losses:
        s.avg_loser_usd = -gross_loss / len(losses)
        s.largest_loser_usd = min(t.net_pnl_usd for t in losses)
    s.avg_holding_s = sum(t.holding_seconds for t in trades) / len(trades)
    s.fees_paid_usd = sum(t.entry_fee_usd + t.exit_fee_usd for t in trades)
    s.slippage_paid_usd = sum(t.entry_slippage_usd + t.exit_slippage_usd for t in trades)

    runners = [t for t in trades if t.return_pct >= 50 and t.holding_seconds >= 120]
    s.runners = len(runners)
    # ---------------------------------------------------------------------------
    # RUNNER PNL SHARE -- the old "runner_capture_pct", RENAMED.
    # ---------------------------------------------------------------------------
    # The old field was called "runner capture" and was displayed to operators as
    # if it measured how much of each runner's peak profit was kept. It does not:
    # it is runners' realised DOLLARS over total winner dollars, i.e. "what share
    # of my gross profit came from trades that ran". A strategy with catastrophic
    # giveback scores just as highly here as one that holds its runners, because
    # the metric never looks at MFE at all.
    #
    # It is kept under its accurate name. The old key is retained as a
    # deprecated alias so an existing consumer cannot silently read a missing
    # field -- but nothing new should display the old label.
    s.runner_pnl_share_pct = (
        sum(t.net_pnl_usd for t in runners) / gross_win * 100.0 if gross_win > 0 else None
    )
    s.runner_capture_pct = s.runner_pnl_share_pct  # DEPRECATED alias, see above

    # TRUE MFE CAPTURE = realised return / MFE, over the runner population.
    # Returns None -- never 0.0 -- when it cannot be computed, because "0% of the
    # peak was kept" and "we have no MFE to compare against" are different facts
    # and only one of them is a performance result.
    #
    # FIDELITY WARNING: ``mfe_pct`` is measured on the OBSERVED MARK feed, not on
    # executable quotes. This is therefore a MARK capture, and is named as such.
    # It must never be presented as execution-aware.
    caps = [
        (t.return_pct / t.mfe_pct)
        for t in runners
        if t.mfe_pct is not None and t.mfe_pct > 0
    ]
    s.mfe_capture_pct = (
        round(sum(caps) / len(caps) * 100.0, 4) if caps else None
    )
    s.mfe_capture_population = len(caps)
    s.mfe_capture_basis = "REALIZED_RETURN_OVER_MARK_MFE; runners only; mark-based MFE"
    s.mfe_capture_executable_pct = None  # not observable: no executable MFE is stored

    # DRAWDOWN SCOPE. The curve below is realised, trade-to-trade. It cannot see
    # an open position that is underwater, so it is not a risk measure for a
    # strategy that holds positions for minutes at a time. Stated, not implied.
    s.max_drawdown_scope = "realized_trade_to_trade"

    s.avg_mfe_pct = sum(t.mfe_pct for t in trades) / len(trades)
    s.avg_mae_pct = sum(t.mae_pct for t in trades) / len(trades)
    s.giveback_pct = (s.avg_mfe_pct - s.expectancy_pct) if trades else 0.0

    # equity curve drawdown on realized PnL alone (unrealized excluded so the
    # number does not wobble with intratrade marks)
    equity = starting_equity
    peak = equity
    max_dd = 0.0
    for t in trades:
        equity += t.net_pnl_usd
        peak = max(peak, equity)
        if peak > 0:
            max_dd = min(max_dd, (equity - peak) / peak * 100.0)
    s.max_drawdown_pct = max_dd

    streak = best = 0
    for t in trades:
        # ``< 0``, NOT ``<= 0``, and it must match the population definition.
        #
        # ``losses`` is built from ``net_pnl_usd < 0`` and ``scratches`` from
        # ``== 0`` a few lines above, so an exactly-flat trade is explicitly NOT
        # a losing trade. This loop used ``<= 0`` and so counted it as one,
        # which made the same dataset report ``losses=2`` and
        # ``max_consecutive_losses=3`` simultaneously -- two different answers
        # to "how many losing trades were there", on the same trades. A scratch
        # also wrongly extended a live streak, so an operator watching the
        # losing streak saw it grow on a trade that made no money.
        if t.net_pnl_usd < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    s.max_consecutive_losses = best
    s.current_losing_streak = streak

    s.by_route = _group(trades, lambda t: t.route)
    s.by_regime = _group(trades, lambda t: t.regime)
    s.by_exit_style = _group(trades, lambda t: t.exit_style)
    s.by_age_bucket = _group(trades, lambda t: t.age_bucket)
    s.by_mc_bucket = _group(trades, lambda t: t.mc_bucket)
    s.by_liq_bucket = _group(trades, lambda t: t.liq_bucket)
    # ---------------------------------------------------------------------------
    # PER-PROFILE BREAKDOWN. L25_CURRENT, L25_60S and L25_TP100 are DIFFERENT
    # strategies: different max-hold behaviour, therefore different exit
    # distributions, therefore not comparable numbers. Pooling them into one win
    # rate and one profit factor produces a statistic that describes no strategy
    # that actually ran. Every profile gets its own block so a viewer can select
    # one and be shown only that one.
    # ---------------------------------------------------------------------------
    s.by_strategy_profile = {}
    if _profile_breakdown:
        for profile, group in _grouped(
            trades, lambda t: getattr(t, "strategy_profile", None)
        ).items():
            s.by_strategy_profile[profile] = compute(
                group, starting_equity=starting_equity, _profile_breakdown=False
            ).as_dict()
        s.strategy_profiles_present = sorted(s.by_strategy_profile)
        s.mixed_strategy_population = len(s.by_strategy_profile) > 1
    return s


def _grouped(trades: Sequence[Trade], key) -> dict[str, list[Trade]]:
    buckets: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        k = key(t)
        if k is not None:
            buckets[str(k)].append(t)
    return dict(buckets)


def format_summary(s: Stats) -> list[str]:
    pf = "inf" if s.profit_factor == float("inf") else f"{s.profit_factor:.2f}"
    return [
        f"trades={s.total_trades} W/L={s.wins}/{s.losses} "
        f"win%={s.win_rate * 100:.1f}",
        f"realized={s.realized_pnl_usd:+.2f} ({s.total_return_pct:+.1f}%) "
        f"expectancy={s.expectancy_usd:+.3f} PF={pf}",
        f"avgW={s.avg_winner_usd:+.2f} avgL={s.avg_loser_usd:+.2f} "
        f"maxDD={s.max_drawdown_pct:.1f}% streak={s.current_losing_streak}",
        f"fees=${s.fees_paid_usd:.2f} slip=${s.slippage_paid_usd:.2f} "
        f"hold={s.avg_holding_s:.0f}s runners={s.runners} "
        f"capture={s.runner_capture_pct:.0f}%",
    ]


def attribution(decisions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """
    Alert -> analysis -> decision -> outcome funnel.

    Used to count missed runners on rejected candidates. A rejected candidate is
    only ever labelled a ``missed_runner`` if a later, genuinely observable
    price exists - it is never retroactively re-decided.
    """
    funnel: dict[str, int] = defaultdict(int)
    for d in decisions:
        funnel["alerts"] += 1
        funnel[str(d.get("decision", "?"))] += 1
    return dict(funnel)
