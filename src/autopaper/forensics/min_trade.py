"""
READ-ONLY minimum-trade, rounding and decimal audit.

QUESTION
========
Can a $5-bankroll simulation realistically represent a $0.50 position, a $0.625
position, and the larger positions that follow from compounding?

The honest answer requires separating what the repository PROVES from what it
ASSUMES, and reporting the residue as ``UNKNOWN`` rather than inventing a
minimum.

THE ACCOUNTING CHAIN, IN ORDER
===============================
1. position USD (percentage of bankroll)
2. SOL lamports           -- integer; truncation is a real, silent loss
3. token base units       -- integer; truncation again
4. venue minimum input    -- a venue-specific rejection, not a rounding
5. venue minimum output   -- ditto
6. account rent           -- a balance the wallet must keep free

Steps 4-6 are PROPERTIES OF A VENUE AND AN ACCOUNT, and the frozen historical
dataset contains none of them. So they are reported as UNKNOWN, with the reason,
rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

#: SOL has 9 decimals; USDC has 6.
SOL_DECIMALS = 9
USDC_DECIMALS = 6
#: A Solana account must retain roughly 0.00303944 SOL to stay rent-exempt.
#: This is a PUBLISHED protocol constant, not an observed value.
ACCOUNT_RENT_EXEMPT_LAMPORTS = 3_039_440

#: Documented venue floors. Each is recorded as CLAIMED, with its source and
#: whether this repository can verify it. None is applied as a trading rule here.
KNOWN_VENUE_MINIMUMS: tuple[dict[str, Any], ...] = (
    {"venue": "pump.fun", "field": "min_sol_lamports", "value": 1_000_000,
     "value_sol": 0.001,
     "verifiable_here": False,
     "note": "widely published, venue-side setting, and it can change without "
             "notice. The frozen dataset records no venue minimum field."},
    {"venue": "jupiter-aggregator", "field": "min_in", "value": None,
     "value_sol": None,
     "verifiable_here": False,
     "note": "route-dependent and not fixed; must be read from a live quote, "
             "which the historical dataset does not contain."},
)


class Verdict(str, Enum):
    """Whether a quantity can be represented, and how confidently."""
    OK = "OK"
    TRUNCATED = "TRUNCATED"
    BELOW_MINIMUM = "BELOW_MINIMUM"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class DecimalAudit:
    """One step of the accounting chain, with the loss it would cause."""

    step: str
    input_value: Optional[float]
    unit: str
    decimals: Optional[int]
    integer_value: Optional[int]
    truncation_loss: Optional[float]
    verdict: Verdict
    note: str
    #: Whether the value clears a floor that could NOT be verified. Kept out of
    #: ``verdict`` on purpose: an unverified floor cannot produce a pass.
    clears_unverified_floor: Optional[bool] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "input": self.input_value,
            "unit": self.unit,
            "decimals": self.decimals,
            "integer": self.integer_value,
            "truncation_loss": self.truncation_loss,
            "verdict": self.verdict.value,
            "note": self.note,
        }


def usd_to_base_units(amount_usd: float, price_usd: float,
                      decimals: int) -> DecimalAudit:
    """
    Convert a USD amount to integer token base units.

    Integer truncation is REAL and SILENT: $0.50 of a token quoted at
    $0.00002234 buys 22,381 base units, and a fraction short of that is simply
    discarded. At very small sizes with many decimals this is a non-trivial
    fraction of the intended position, and it always rounds AGAINST the trader.
    """
    if price_usd is None or price_usd <= 0:
        return DecimalAudit("usd->base_units", amount_usd, "token", decimals,
                            None, None, Verdict.UNKNOWN,
                            "price unavailable, so the conversion cannot be "
                            "computed")
    raw = amount_usd / price_usd
    units = int(raw * (10 ** decimals))      # truncation, not rounding
    achieved = units / (10 ** decimals) * price_usd
    loss = amount_usd - achieved
    return DecimalAudit(
        step="usd->base_units",
        input_value=amount_usd, unit="token", decimals=decimals,
        integer_value=units, truncation_loss=loss,
        verdict=Verdict.OK if units > 0 else Verdict.BELOW_MINIMUM,
        note=(f"{amount_usd:.6f} USD at {price_usd:.3e} -> {units} base units; "
              f"{loss:.2e} USD discarded by truncation"
              + (" (position is UNBUYABLE: rounds to zero units)"
                 if units == 0 else "")),
    )


def usd_to_lamports(amount_usd: float, sol_usd: float) -> DecimalAudit:
    """
    Convert a USD amount to integer SOL lamports.

    The mirror of the token conversion: always rounds against the trader, and at
    a $0.50 size the SOL amount is large enough that this loss is negligible,
    which is worth stating rather than assuming.
    """
    if sol_usd is None or sol_usd <= 0:
        return DecimalAudit("usd->lamports", amount_usd, "SOL", SOL_DECIMALS,
                            None, None, Verdict.UNKNOWN,
                            "SOL price unavailable")
    raw = amount_usd / sol_usd
    lamports = int(raw * (10 ** SOL_DECIMALS))
    achieved = lamports / (10 ** SOL_DECIMALS) * sol_usd
    loss = amount_usd - achieved
    rel = (loss / amount_usd * 100.0) if amount_usd else 0.0
    return DecimalAudit(
        step="usd->lamports",
        input_value=amount_usd, unit="SOL", decimals=SOL_DECIMALS,
        integer_value=lamports, truncation_loss=loss,
        verdict=Verdict.OK if lamports > 0 else Verdict.BELOW_MINIMUM,
        note=(f"{amount_usd:.6f} USD at {sol_usd:.2f} -> {lamports} lamports; "
              f"{loss:.3e} USD lost ({rel:.6f}% of the position)"
              + (" (position is below one lamport)" if lamports == 0 else "")),
    )


def token_decimal_audit(
    amount_usd: float,
    price_usd: float,
    token_decimals: int,
    sol_usd: float,
) -> list[DecimalAudit]:
    return [usd_to_lamports(amount_usd, sol_usd),
            usd_to_base_units(amount_usd, price_usd, token_decimals)]


def venue_minimum_status(position_usd: float) -> DecimalAudit:
    """
    Whether a position clears the KNOWN venue floors.

    The verdict is ``UNKNOWN`` in BOTH directions on purpose. An unverifiable
    floor cannot produce a confident ``OK``: doing so would report "this trade
    would have been accepted" on the strength of a constant nobody has checked
    against a live venue. The comparison is still reported, as a separate
    boolean, so the arithmetic is visible without becoming a verdict.
    """
    sol_usd = 200.0  # only used to express the floor in USD; see note
    lam = usd_to_lamports(position_usd, sol_usd)
    floor = KNOWN_VENUE_MINIMUMS[0]
    clears = (lam.integer_value is not None
              and lam.integer_value >= floor["value"])
    return DecimalAudit(
        step="venue_minimum_input",
        input_value=position_usd, unit="USD", decimals=None,
        integer_value=lam.integer_value, truncation_loss=lam.truncation_loss,
        verdict=Verdict.UNKNOWN,
        clears_unverified_floor=clears,
        note=(f"claimed floor {floor['value_sol']} SOL (1e6 lamports) is "
              f"{'cleared' if clears else 'NOT cleared'} at ${position_usd:.4f}"
              ", but the floor is NOT verifiable from the frozen dataset and "
              "the USD comparison assumes a placeholder SOL price, so the "
              "verdict is UNKNOWN rather than a pass"),
    )


def account_rent_status(available_lamports: Optional[int]) -> DecimalAudit:
    """
    Whether the wallet can act without dropping below rent-exemption.

    This is a real, computable protocol constraint and the only one here that is
    not UNKNOWN -- but it needs the wallet's free balance, which the research
    dataset does not carry.
    """
    if available_lamports is None:
        return DecimalAudit("account_rent", None, "lamports", None, None, None,
                            Verdict.UNKNOWN,
                            "no free-balance figure in the dataset, so rent "
                            "headroom cannot be assessed")
    head = available_lamports - ACCOUNT_RENT_EXEMPT_LAMPORTS
    return DecimalAudit(
        step="account_rent", input_value=float(available_lamports),
        unit="lamports", decimals=None, integer_value=head,
        truncation_loss=None,
        verdict=Verdict.OK if head > 0 else Verdict.BELOW_MINIMUM,
        note=(f"{available_lamports} lamports free, "
              f"{ACCOUNT_RENT_EXEMPT_LAMPORTS} reserved for rent, "
              f"{head} lamports of headroom"),
    )


@dataclass
class SmallBankrollAudit:
    """Verdict per sizing mode for the requested small-bankroll positions."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    unknown_items: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "unknown_items": self.unknown_items,
            "changes_production_sizing": False,
        }


def audit_small_bankroll(
    positions_usd: tuple[float, ...] = (0.50, 0.625, 1.00, 5.00, 10.00),
    *,
    token_decimals: int = 6,
    token_price_usd: float = 2.234e-05,
    sol_usd: float = 200.0,
) -> SmallBankrollAudit:
    """
    Audit every requested position size through the whole accounting chain.

    ``token_decimals=6`` is the DEFAULT for pump.fun-style tokens and is stated
    as an assumption, not a measurement: the frozen dataset does not record the
    mint's decimals.
    """
    out = SmallBankrollAudit()
    out.unknown_items = [
        "token mint decimals (assumed 6; the dataset records no decimals field)",
        "venue minimum input amount (claimed 1e6 lamports; unverifiable here)",
        "venue minimum output amount (route-dependent; not in the dataset)",
        "account rent headroom (needs a wallet balance the dataset lacks)",
        "aggregator route minimums (vary per route; not in the dataset)",
        "whether a real venue would have accepted any of these sizes",
    ]
    for p in positions_usd:
        steps = token_decimal_audit(p, token_price_usd, token_decimals, sol_usd)
        steps.append(venue_minimum_status(p))
        steps.append(account_rent_status(None))
        out.rows.append({
            "position_usd": p,
            "steps": [s.as_dict() for s in steps],
            "max_buyable_token_price_usd": max_price_for_buyable(
                p, token_decimals),
            "fully_representable": all(
                s.verdict in (Verdict.OK,) for s in steps[:2]),
            "any_unknown": any(s.verdict is Verdict.UNKNOWN for s in steps),
        })
    return out


def max_price_for_buyable(position_usd: float, token_decimals: int) -> float:
    """
    The highest token price at which ``position_usd`` still buys >= 1 base unit.

    Above this price the position truncates to ZERO units and is simply not
    tradeable -- not expensive, unbuyable. That is a hard cliff, not a
    proportional cost, and it is the one integer-conversion effect that can
    actually invalidate a small-bankroll simulation.
    """
    return position_usd * (10 ** token_decimals)


def unbuyable_boundary(
    positions_usd: tuple[float, ...] = (0.50, 0.625, 1.00),
    token_decimals: tuple[int, ...] = (0, 2, 6, 9),
) -> dict[str, Any]:
    """
    Where integer truncation stops being a rounding detail and becomes an
    untradeable position.

    Reported as a table rather than a single number because the answer depends
    entirely on ``token_decimals`` -- which the frozen dataset does not record.
    That dependency IS the finding.
    """
    rows = []
    for d in token_decimals:
        for p in positions_usd:
            rows.append({
                "token_decimals": d,
                "position_usd": p,
                "max_buyable_price_usd": max_price_for_buyable(p, d),
                "median_l25_price_usd": 2.234e-05,
                "median_l25_is_buyable": (
                    max_price_for_buyable(p, d) >= 2.234e-05),
            })
    return {
        "rows": rows,
        "reading": (
            "at the median L25 token price (~2.2e-05) a $0.50 position buys "
            "tens of millions of base units at any plausible decimal count, so "
            "integer truncation is a rounding detail here, not a barrier. The "
            "cliff only matters for a token quoted ABOVE the row's "
            "max_buyable_price_usd, which is a different price regime entirely."),
        "why_decimals_matters": (
            "the frozen dataset records no mint decimals, so this table cannot "
            "be collapsed to one number. Token decimals on this asset class "
            "commonly range from 6 to 9."),
    }


def lamport_truncation_is_material(position_usd: float, sol_usd: float,
                                  threshold_pct: float = 0.01) -> dict[str, Any]:
    """
    Is SOL-lamport truncation material at this size?

    Reported rather than assumed: at $0.50 and a $200 SOL price the loss is
    around 1e-9 USD, which is immaterial. The TOKEN conversion is the one that
    can matter, because token prices are ~1e-5 and a base unit is a large step.
    """
    a = usd_to_lamports(position_usd, sol_usd)
    rel = ((a.truncation_loss or 0.0) / position_usd * 100.0
           if position_usd else 0.0)
    return {
        "position_usd": position_usd,
        "sol_usd_assumed": sol_usd,
        "lamports": a.integer_value,
        "truncation_loss_usd": a.truncation_loss,
        "loss_pct_of_position": rel,
        "material": rel >= threshold_pct,
        "threshold_pct": threshold_pct,
        "note": ("SOL-lamport truncation is immaterial at these sizes. The "
                 "TOKEN base-unit conversion is the one that can matter, "
                 "because a token quoted near 1e-5 makes a single base unit a "
                 "large step of the position."),
    }


def full_audit(
    positions_usd: tuple[float, ...] = (0.50, 0.625, 1.00, 5.00, 10.00),
    *,
    token_decimals: int = 6,
    token_price_usd: float = 2.234e-05,
    sol_usd: float = 200.0,
) -> dict[str, Any]:
    return {
        "pricing_label": "OBSERVATIONAL / MARK-BASED",
        "small_bankroll": audit_small_bankroll(
            positions_usd, token_decimals=token_decimals,
            token_price_usd=token_price_usd, sol_usd=sol_usd).as_dict(),
        "unbuyable_boundary": unbuyable_boundary(),
        "lamport_truncation": [
            lamport_truncation_is_material(p, sol_usd) for p in positions_usd],
        "known_venue_minimums": [dict(v) for v in KNOWN_VENUE_MINIMUMS],
        "changes_production_sizing": False,
        "headline": (
            "integer representation is NOT the barrier at these sizes: at the "
            "median L25 token price a $0.50 position is tens of millions of "
            "base units at any plausible decimal count. What cannot be "
            "established from the frozen dataset is whether a real venue would "
            "have ACCEPTED any of these sizes, because the dataset records no "
            "venue minimum and no order was ever placed. Those are reported as "
            "UNKNOWN, not guessed."),
    }
