"""
READ-ONLY cost-model forensics.

Traces every component of the historical research cost model back to its source,
then independently computes constant-product AMM price impact for representative
liquidity/notional pairs.

THE POINT
=========
Separate three things that are routinely conflated:

1. **MEASURED** -- a number that came from a real venue. Almost nothing here is.
2. **MODELED** -- a formula the project applies. Most of the cost model is.
3. **ASSUMED** -- a value substituted because the real one was unavailable. The
   single largest of these is ``assume_sol_usd``, and pretending otherwise is how
   an assumed liquidity becomes an "observed" one in a report.

Nothing here changes the cost model. The point is to be able to state exactly
what each basis point is made of.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

# ---------------------------------------------------------------------------
# the traced formula
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostComponent:
    name: str
    expression: str
    source: str
    kind: str            # "modeled" | "assumed" | "measured"
    per_side: bool
    value: Optional[float]
    unit: str
    note: str


#: Every component, traced. Values are the shipped defaults from
#: ``autopaper.config.ExecutionModel``; this module IMPORTS the live model
#: rather than restating it, so the trace cannot drift from production.
def traced_components() -> list[CostComponent]:
    from autopaper.config import ExecutionModel

    m = ExecutionModel()
    return [
        CostComponent(
            name="trading_fee",
            expression=f"{m.entry_fee_bps:g} bps of the post-fill notional, per side",
            source="autopaper.config.ExecutionModel.entry_fee_bps / exit_fee_bps",
            kind="modeled", per_side=True, value=m.entry_fee_bps, unit="bps",
            note=("documented in config.py as 'typical DEX aggregator taker fee "
                  "+ priority'. It is a SINGLE blended allowance: there is no "
                  "separate network-fee component anywhere in the PAPER model, "
                  "so it cannot be decomposed further from the data."),
        ),
        CostComponent(
            name="slippage_base",
            expression=f"{m.entry_slippage_bps:g} bps, per side, adverse",
            source="autopaper.config.ExecutionModel.entry_slippage_bps",
            kind="modeled", per_side=True, value=m.entry_slippage_bps, unit="bps",
            note=("a FLAT blended allowance applied adversely on both sides. It "
                  "is not a measured spread and not derived from any order "
                  "book; the repository contains no bid, ask, quote or depth "
                  "field at all."),
        ),
        CostComponent(
            name="size_impact",
            expression=(f"min({m.impact_max_bps:g}, "
                        f"{m.impact_base_bps:g} * sqrt(N / L)) bps"),
            source="autopaper.execution.fill.impact_bps",
            kind="modeled", per_side=True, value=None, unit="bps",
            note=("N is the notional and L the pool liquidity. This is the ONLY "
                  "component that depends on size, which makes small-bankroll "
                  "replays systematically flatter than large ones. Capped, so "
                  "beyond a certain size it stops growing -- an optimistic "
                  "property for large notionals."),
        ),
        CostComponent(
            name="staleness",
            expression=(f"min({m.staleness_penalty_max_bps:g}, "
                        f"age_s * {m.staleness_penalty_bps_per_sec:g}) bps"),
            source="autopaper.execution.fill (staleness term)",
            kind="modeled", per_side=True, value=None, unit="bps",
            note=("derived from the measured age of the quote used for the "
                  "fill. Recovered ages on the L25 population are small "
                  "(median ~0 on the exit leg), so this term is near zero in "
                  "practice rather than absent in principle."),
        ),
        CostComponent(
            name="assumed_liquidity",
            expression=f"{m.assume_sol_usd:,.0f} USD when no liquidity is known",
            source="autopaper.config.ExecutionModel.assume_sol_usd",
            kind="assumed", per_side=False, value=m.assume_sol_usd, unit="usd",
            note=("THE LARGEST ASSUMPTION IN THE COST MODEL. When the fill had "
                  "no liquidity to read, the model substitutes a $60,000 pool. "
                  "That is an ASSUMPTION, not an observation, and it must never "
                  "be reported as observed liquidity. On the 116 L25 trades "
                  "only 2 fell back to it, but the substitution is silent when "
                  "it happens."),
        ),
        CostComponent(
            name="network_fee",
            expression="absent from the PAPER fill model",
            source="(none) RealConfig.priority_fee_lamports exists but is not "
                   "applied in PAPER",
            kind="assumed", per_side=False, value=None, unit="usd",
            note=("A priority/network fee is configured for REAL mode "
                  "(priority_fee_lamports = 50000) and is NOT part of the PAPER "
                  "model. A PAPER result therefore carries NO network cost at "
                  "all, which is a real omission and not a rounding detail."),
        ),
        CostComponent(
            name="token_transfer_tax",
            expression="absent entirely",
            source="(none) no such field exists on any recorded trade",
            kind="assumed", per_side=False, value=None, unit="bps",
            note=("A 1-6% per-side charge is routine on this asset class. There "
                  "is no field for it anywhere, so it is neither modelled nor "
                  "measurable from history. See "
                  "autopaper.forensics.transfer_fee for the read-only detector "
                  "that can establish it going forward."),
        ),
        CostComponent(
            name="decision_latency",
            expression=f"{m.decision_latency_ms:g} ms waited before filling",
            source="autopaper.config.ExecutionModel.decision_latency_ms",
            kind="modeled", per_side=False, value=m.decision_latency_ms,
            unit="ms",
            note=("a WAIT is applied, but no adverse price movement is charged "
                  "for the time it covers, because the fill is referenced to "
                  "the decision-time mark. That is a systematic optimism."),
        ),
    ]


def cost_formula_text() -> str:
    """The formula, written out, for the report."""
    from autopaper.config import ExecutionModel

    m = ExecutionModel()
    return f"""
slippage_bps(side) = {m.entry_slippage_bps:g}                      # flat base
                    + min({m.impact_max_bps:g},
                          {m.impact_base_bps:g} * sqrt(N / L))       # size impact
                    + min({m.staleness_penalty_max_bps:g},
                          age_s * {m.staleness_penalty_bps_per_sec:g})  # staleness

entry_fill = entry_ref * (1 + slippage_bps('entry')  / 10000)
exit_fill  = exit_ref  * (1 - slippage_bps('exit')   / 10000)

entry_fee = qty * entry_fill * {m.entry_fee_bps:g} / 10000
exit_fee  = qty * exit_fill  * {m.exit_fee_bps:g}  / 10000

gross      = (exit_ref - entry_ref) * qty
net        = gross - entry_fee - exit_fee
                       - qty * (entry_fill - entry_ref)
                       - qty * (exit_ref  - exit_fill)

where  N = notional (target size, PRE-slippage)
       L = recovered pool liquidity, or {m.assume_sol_usd:,.0f} when unknown (ASSUMED)
       qty = N / entry_ref

A symmetric, all-in round trip therefore costs approximately:
    2 * ({m.entry_fee_bps:g} + {m.entry_slippage_bps:g} + impact + staleness) bps
""".strip()


# ---------------------------------------------------------------------------
# independent AMM impact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AmmComparison:
    notional_usd: float
    liquidity_usd: float
    liquidity_is_assumed: bool
    amm_half_spread_bps: float
    amm_full_cross_bps: float
    repo_impact_bps: float
    repo_flat_base_bps: float
    repo_all_in_round_trip_bps: float
    amm_all_in_round_trip_bps: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "notional_usd": self.notional_usd,
            "liquidity_usd": self.liquidity_usd,
            "liquidity_is_assumed": self.liquidity_is_assumed,
            "amm_half_spread_bps": round(self.amm_half_spread_bps, 3),
            "amm_full_cross_bps": round(self.amm_full_cross_bps, 3),
            "repo_impact_bps": round(self.repo_impact_bps, 3),
            "repo_flat_base_bps": self.repo_flat_base_bps,
            "repo_all_in_round_trip_bps": round(
                self.repo_all_in_round_trip_bps, 2),
            "amm_all_in_round_trip_bps": round(
                self.amm_all_in_round_trip_bps, 2),
            "repo_minus_amm_bps": round(
                self.repo_all_in_round_trip_bps - self.amm_all_in_round_trip_bps, 2),
        }


def amm_half_spread_bps(notional_usd: float, liquidity_usd: float) -> float:
    """
    Constant-product AMM half-spread, in basis points.

    For a pool whose USD-side depth is L, moving N USD shifts the marginal price
    by N/L; the AVERAGE execution price therefore sits about N/(2L) away from the
    mid. Crossing a round trip pays it twice.

    This is pool-curve geometry only. It excludes every fixed cost, so it is a
    LOWER bound on what a round trip costs, never an estimate of it.
    """
    if notional_usd <= 0 or liquidity_usd <= 0:
        return 0.0
    return 1e4 * notional_usd / (2.0 * liquidity_usd)


def compare_to_repo(
    notional_usd: float,
    liquidity_usd: float,
    *,
    liquidity_is_assumed: bool = False,
) -> AmmComparison:
    """
    Put the repo's cost model next to closed-form AMM geometry.

    If the repo charges MORE than the pool geometry, the cost model is
    conservative on that pair; if LESS, it is optimistic. Stating which, per
    pair, is the whole value of this function.
    """
    from autopaper.config import ExecutionModel
    from autopaper.execution.fill import impact_bps

    m = ExecutionModel()
    half = amm_half_spread_bps(notional_usd, liquidity_usd)
    repo_imp = impact_bps(notional_usd, liquidity_usd or None, m)
    repo_rt = 2 * (m.entry_fee_bps + m.entry_slippage_bps + repo_imp)
    amm_rt = 2 * half
    return AmmComparison(
        notional_usd=notional_usd,
        liquidity_usd=liquidity_usd,
        liquidity_is_assumed=liquidity_is_assumed,
        amm_half_spread_bps=half,
        amm_full_cross_bps=2 * half,
        repo_impact_bps=repo_imp,
        repo_flat_base_bps=m.entry_slippage_bps,
        repo_all_in_round_trip_bps=repo_rt,
        amm_all_in_round_trip_bps=amm_rt,
    )


#: The representative grid requested by the audit.
REPRESENTATIVE_NOTIONALS: tuple[float, ...] = (0.50, 0.625, 1.0, 5.0, 10.0, 100.0)
REPRESENTATIVE_LIQUIDITIES: tuple[tuple[float, bool], ...] = (
    (10_000.0, False),
    (16_900.0, False),   # the median recovered L25 entry-side liquidity
    (60_000.0, True),    # exactly ExecutionModel.assume_sol_usd
)


def sensitivity_grid() -> list[dict[str, Any]]:
    """Every requested notional x liquidity pair, with the assumed flag honest."""
    out: list[dict[str, Any]] = []
    for liq, assumed in REPRESENTATIVE_LIQUIDITIES:
        for n in REPRESENTATIVE_NOTIONALS:
            out.append(compare_to_repo(n, liq, liquidity_is_assumed=assumed).as_dict())
    return out


def small_bankroll_bias(
    *,
    median_observed_liquidity_usd: float = 16_900.0,
    large_bankroll_notional_usd: float = 888.0,
    small_bankroll_notional_usd: float = 0.50,
) -> dict[str, Any]:
    """
    Quantify the small-bankroll bias created by the size-dependent impact term.

    At $0.50 the impact term collapses; at a $100-bankroll run's mean notional it
    does not. So two replays of the SAME trade carry different modelled costs,
    and the smaller account is systematically flattered per unit of percentage
    risk. Stating the ratio is more useful than asserting the direction.
    """
    a = compare_to_repo(small_bankroll_notional_usd, median_observed_liquidity_usd)
    b = compare_to_repo(large_bankroll_notional_usd, median_observed_liquidity_usd)
    return {
        "median_observed_liquidity_usd": median_observed_liquidity_usd,
        "small_notional_usd": small_bankroll_notional_usd,
        "small_repo_impact_bps": round(a.repo_impact_bps, 3),
        "large_notional_usd": large_bankroll_notional_usd,
        "large_repo_impact_bps": round(b.repo_impact_bps, 3),
        "impact_ratio_small_over_large": (
            round(a.repo_impact_bps / b.repo_impact_bps, 5)
            if b.repo_impact_bps else None),
        "verdict": (
            "the size-dependent impact term is what makes a small-bankroll "
            "replay cheaper per trade than a large-bankroll replay of the same "
            "trade. Fees and the flat slippage allowance are notional-"
            "independent and cancel out of this comparison. A $5-bankroll "
            "result is therefore FLATTERED relative to a $100-bankroll result "
            "on identical marks, and the two are not directly comparable."),
        "what_is_not_modelled_here": [
            "the size impact CAP means beyond some size the model stops "
            "growing the penalty, which flatters LARGE notionals",
            "token transfer tax, which would penalise small transfers far "
            "harder in percentage terms because it has no size dependence",
            "any fixed per-transaction cost, which penalises small transfers "
            "harder still",
        ],
    }


def assumptions_register() -> list[dict[str, str]]:
    """
    Every assumption, stated as an assumption.

    Kept as data so a report cannot accidentally present one as a measurement.
    """
    from autopaper.config import ExecutionModel

    m = ExecutionModel()
    return [
        {"id": "A1", "assumption": (
            f"the flat {m.entry_slippage_bps:g}bp per side stands in for "
            "crossing the spread"),
         "reality": ("no bid, ask, depth or quote field exists anywhere in the "
                     "dataset"),
         "direction": "conservative if the real spread is smaller; optimistic "
                       "if larger"},
        {"id": "A2", "assumption": (
            f"liquidity {m.assume_sol_usd:,.0f} when none was readable at fill"),
         "reality": ("the recorded fill_model.detail states 'liquidity unknown "
                     "(assumed $60,000)'"),
         "direction": "understates impact whenever the true pool is smaller"},
        {"id": "A3", "assumption": (
            f"the {m.entry_fee_bps:g}bp fee covers taker fee AND priority"),
         "reality": ("no network-fee component exists in the PAPER model, so a "
                     "PAPER result carries no network cost"),
         "direction": "optimistic by the amount of a real network fee"},
        {"id": "A4",
         "assumption": "no token transfer tax is charged",
         "reality": "routine 1-6% per side on this asset class, unmodelled",
         "direction": ("strongly optimistic, and size-independent, so it hurts "
                       "small transfers as much as large ones")},
        {"id": "A5", "assumption": (
            "filling at the decision-time mark, with no adverse price movement "
            f"charged for the {m.decision_latency_ms:g}ms latency that is waited"),
         "reality": "the fill is referenced to a mark the bot had already seen",
         "direction": "optimistic"},
        {"id": "A6", "assumption": (
            f"impact grows as sqrt(N/L) and is capped at {m.impact_max_bps:g}bp"),
         "reality": "an approximation of pool geometry, not a measured curve",
         "direction": "flatters large notionals, where the cap binds"},
    ]


def full_audit() -> dict[str, Any]:
    """Everything, for the JSON companion."""
    return {
        "pricing_label": "OBSERVATIONAL / MARK-BASED",
        "formula": cost_formula_text(),
        "components": [c.__dict__ for c in traced_components()],
        "sensitivity_grid": sensitivity_grid(),
        "small_bankroll_bias": small_bankroll_bias(),
        "assumptions": assumptions_register(),
        "changes_the_model": False,
        "note": (
            "This module traces and audits. It does not alter the cost model, "
            "the sizing, L25 or any production behaviour."),
    }
