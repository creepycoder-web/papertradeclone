"""
Costs a PAPER trade would pay OUTSIDE the swap input, and the deployable budget.

THE DEFECT THIS CLOSES
======================

PAPER charged the SWAP fee and nothing else. The swap fee is real and is already
inside the executable quote's input amount, so it was never missing. What was
missing is everything a Solana round trip costs BESIDE the swap:

* the **base transaction fee** (5,000 lamports per signature), twice -- once for
  the BUY and once for the SELL;
* the **priority fee**, twice -- the project configures 50,000 lamports for
  REAL transactions (``RealConfig.priority_fee_lamports``), and
  ``forensics/cost_forensics.py`` records that this value is NOT part of the
  PAPER model at all;
* the **token-account rent**, 2,039,280 lamports, which a first buy of a mint
  must lock before any token exists.

An ALL-IN arm commits 100% of its cash to the SWAP INPUT. On chain that cannot
execute: the wallet has nothing left to pay the transaction fees or to fund the
associated token account, so the BUY would fail outright. A PAPER experiment
that ignores this overstates both the size it could deploy and the P&L it books.

WHAT IS CHARGED AND WHAT IS ONLY RESERVED
=========================================

Two different things, and conflating them is the error this module exists to
avoid:

* **NON-RECOVERABLE**, charged to P&L: the base fee and the priority fee for
  both legs. That SOL is spent.
* **RECOVERABLE DEPOSIT**, reserved but NOT charged: the token-account rent. It
  must be AVAILABLE at entry time, and it comes back when the account is closed.
  ``economics.py`` already models this as locked-not-spent; charging it as a
  round-trip loss would double-count a recoverable balance.

So the reserve reduces the DEPLOYABLE notional (the wallet must hold it), and
only the non-recoverable part reduces the arm's bankroll.

WHAT IS ASSUMED, AND IS SAID SO
===============================

Every number here carries a PROVENANCE label, because the difference between a
measured cost and a configured one is the difference between evidence and a
setting:

* ``DERIVED``   -- a protocol constant or an arithmetic identity.
* ``CONFIGURED``-- a value this project already sets for REAL transactions.
* ``MEASURED``  -- an observation from a live provider/RPC read.
* ``ASSUMED``   -- a worst case chosen to FAIL CLOSED, not a measurement.

The priority fee is CONFIGURED (the project's own REAL default), the base fee
and the rent are DERIVED (protocol constants), and "a new token account is
required" is ASSUMED -- conservative, because a token account that already
exists only makes the trade cheaper, so assuming the expensive case cannot
overstate the reserve.

DERIVED FROM THE INTENDED TRANSACTION, NOT RESTATED
===================================================

The fee numbers are derived from the transaction this project would ACTUALLY
intend to build (see :func:`intended_transaction_config`), rather than being
restated as bare constants:

* the base fee is ``signatures x 5000``; the intended swap has ONE fee payer and
  therefore ONE signature per leg, and the multiplier is applied from the
  declared configuration;
* the priority fee is the **CEILING** the intended transaction configuration
  requests, which is what a ``dynamicComputeUnitLimit`` transaction pays AT
  MOST. The fee a real broadcast would OBSERVE is
  ``ceil(compute_units_used x price_micro_lamports / 1e6)`` and can only be
  lower -- deriving it exactly would require simulating/signing the intended
  transaction, which this PAPER path never does. So it is reported as a
  CONFIGURED CEILING with ``observed=False``, never as the observed fee.

FAILED REQUESTS ARE NOT FAILED TRANSACTIONS
===========================================

A failed QUOTE REQUEST is a refusal from an HTTP provider: nothing was built,
nothing was signed, nothing reached the network, and **no lamport of network fee
was incurred**. A transaction that is BROADCAST and then fails DOES pay its base
and priority fees. PAPER never broadcasts, so its failed executions are
quote-side refusals at zero network cost; the arithmetic for the other case is
stated by :func:`failed_execution_accounting` so the distinction is explicit
rather than implied.

A MISSING SOL/USD REFERENCE IS NOT A ZERO
=========================================

Without a SOL/USD price the reserve cannot be expressed in the arm's USD ledger.
Unknown is reported as ``known=False`` with the reason -- never silently applied
as zero. The caller decides what to do with an unknown; this module only refuses
to invent one.

PAPER ONLY
==========

Nothing here builds, signs, submits or broadcasts anything. It is arithmetic
over constants and a price reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .economics import SOL_DECIMALS, TOKEN_ACCOUNT_RENT_LAMPORTS

#: Solana's base fee, in lamports, per SIGNATURE. A swap transaction has one
#: signer (the payer), so one fee per leg.
#: Source: the protocol's documented ``5000`` lamports base fee.
SOLANA_BASE_FEE_LAMPORTS = 5_000

#: The priority fee assumed for a PAPER leg, in lamports.
#:
#: CONFIGURED, not measured: this mirrors ``RealConfig.priority_fee_lamports``
#: (50,000), the value this project already uses for a REAL transaction. It is
#: declared here rather than read from REAL config so a PAPER research result
#: cannot change because an operator edited a REAL setting -- and it is recorded
#: on every trade as an ASSUMPTION with this value, so a reader can see exactly
#: what was charged.
PAPER_PRIORITY_FEE_LAMPORTS = 50_000

#: How many transactions one PAPER round trip costs: the BUY and the SELL.
ROUND_TRIP_LEGS = 2

#: Cost provenance labels. Every cost carries one.
COST_DERIVED = "DERIVED"
COST_CONFIGURED = "CONFIGURED"
COST_MEASURED = "MEASURED"
COST_ASSUMED = "ASSUMED"

#: SIGNATURES PER LEG, from the INTENDED transaction configuration: a swap has
#: exactly one fee payer that must sign, so the base fee is ``1 x 5000``. Kept as
#: a declared input rather than folded into a literal, so a transaction shape
#: with a different signer count cannot silently inherit the one-signature fee.
INTENDED_SIGNATURES_PER_LEG = 1

#: WHERE THE PRIORITY FEE COMES FROM. It is the CEILING the intended transaction
#: requests -- a ``dynamicComputeUnitLimit`` transaction pays
#: ``ceil(cu_used x price_micro_lamports / 1e6)``, which is at most the requested
#: total. The observed fee is therefore NOT derivable without simulating and
#: signing the intended transaction, and is never claimed to be measured.
PRIORITY_FEE_SOURCE_CEILING = "CONFIGURED_CEILING_NOT_OBSERVED"

#: The INTENDED transaction configuration -- what the REAL swap path would build.
#:
#: WHY DECLARED AND NOT IMPORTED
#: ------------------------------
#: These mirror ``autopaper.config.RealConfig``'s defaults. They are DECLARED
#: here rather than read from that dataclass so a PAPER research result can never
#: change because an operator edited a REAL setting -- and so this module stays
#: importable without constructing a REAL config. Every consumer that reports a
#: derived cost names this table as its derivation source.
INTENDED_TX_CONFIG: dict[str, Any] = {
    "fee_payer_signatures_per_leg": INTENDED_SIGNATURES_PER_LEG,
    "priority_fee_lamports_per_tx": PAPER_PRIORITY_FEE_LAMPORTS,
    "priority_fee_is_a_ceiling": True,
    "dynamic_compute_unit_limit": True,
    "slippage_bps": 500,
    "max_price_impact_pct": 5.0,
    "simulate_before_send": True,
    "quote_max_age_ms": 8000.0,
    "source": (
        "mirrors autopaper.config.RealConfig defaults: slippage_bps=500, "
        "priority_fee_lamports=50000, max_price_impact_pct=5.0, "
        "simulate_before_send=True, quote_max_age_ms=8000.0, "
        "dynamicComputeUnitLimit=True"),
}

#: The two ways an attempt can fail, and they do NOT cost the same:
#: a refused QUOTE request is an HTTP-side refusal, a failed BROADCAST is a
#: transaction that reached the chain and paid its fees.
ATTEMPT_QUOTE_REQUEST = "QUOTE_REQUEST"
ATTEMPT_BROADCAST = "BROADCAST"


def intended_transaction_config() -> dict[str, Any]:
    """The intended transaction configuration, WITH a provenance label per field.

    A reader can therefore tell which of the fee numbers is derived from this
    configuration and which is only declared. Returning a copy keeps a caller
    from mutating the module-level table.
    """
    cfg = dict(INTENDED_TX_CONFIG)
    cfg["provenance"] = {
        "fee_payer_signatures_per_leg": (
            f"{COST_DERIVED}: a swap has one fee payer and therefore one "
            "signature per leg"),
        "priority_fee_lamports_per_tx": (
            f"{COST_CONFIGURED}: {PRIORITY_FEE_SOURCE_CEILING} -- the total the "
            "intended transaction requests; the OBSERVED fee is "
            "ceil(cu_used x price_microlamports / 1e6) and is at most this"),
        "slippage_bps": (
            f"{COST_CONFIGURED}: mirrors RealConfig.slippage_bps; the intended "
            "transaction would set a minimum-output floor from it. PAPER builds "
            "no transaction, so it applies NO price boundary"),
        "max_price_impact_pct": (
            f"{COST_CONFIGURED}: mirrors RealConfig.max_price_impact_pct; "
            "PAPER refuses nothing on impact because it submits nothing"),
        "quote_max_age_ms": (
            f"{COST_CONFIGURED}: mirrors RealConfig.quote_max_age_ms; PAPER "
            "DOES apply this, on both legs"),
    }
    return cfg


@dataclass(frozen=True)
class OutsideSwapCosts:
    """
    One PAPER round trip's costs that live OUTSIDE the swap input.

    All lamport fields are for the WHOLE round trip (both legs), not per leg.
    """

    base_fee_lamports: int = 0
    priority_fee_lamports: int = 0
    rent_lamports: int = 0
    legs: int = ROUND_TRIP_LEGS
    #: Signatures per leg, from the intended transaction configuration -- the
    #: multiplier behind ``base_fee_lamports``.
    signatures_per_leg: int = INTENDED_SIGNATURES_PER_LEG
    #: True when ``priority_fee_lamports`` is the configured CEILING rather than
    #: an observed charge. It is True on every branch this module produces.
    priority_fee_is_ceiling: bool = True
    #: Which provenance the priority fee carries, machine-readable so a consumer
    #: does not have to parse the prose.
    priority_fee_source: str = PRIORITY_FEE_SOURCE_CEILING
    lamports_per_sol: int = 10 ** SOL_DECIMALS
    sol_usd: float = 0.0
    sol_usd_source: str = ""
    sol_usd_at: float | None = None
    #: Is the USD value of these costs known? ``False`` means the SOL/USD
    #: reference was missing, so the reserve was NOT applied in USD terms.
    known: bool = False
    reason: str = ""
    provenance: Mapping[str, str] = field(default_factory=dict)
    #: True when a NEW token account is assumed (the conservative case). The
    #: rent is reserved only when this is true.
    assumes_new_token_account: bool = True

    @property
    def non_recoverable_lamports(self) -> int:
        """Base fee + priority fee for both legs. SPENT, not returned."""
        return int(self.base_fee_lamports) + int(self.priority_fee_lamports)

    @property
    def recoverable_lamports(self) -> int:
        """Token-account rent. LOCKED at entry, returned when the account closes."""
        return int(self.rent_lamports)

    @property
    def total_reserve_lamports(self) -> int:
        """Everything the wallet must hold BESIDE the swap input."""
        return self.non_recoverable_lamports + self.recoverable_lamports

    def _usd(self, lamports: int) -> float | None:
        if not self.known:
            return None
        price = float(self.sol_usd or 0.0)
        if price <= 0:
            return None
        return (int(lamports) / float(self.lamports_per_sol)) * price

    @property
    def non_recoverable_usd(self) -> float | None:
        return self._usd(self.non_recoverable_lamports)

    @property
    def recoverable_usd(self) -> float | None:
        return self._usd(self.recoverable_lamports)

    @property
    def total_reserve_usd(self) -> float | None:
        return self._usd(self.total_reserve_lamports)

    @property
    def charged_to_pnl_usd(self) -> float:
        """
        The part of the reserve that is really SPENT: the two legs' fees.

        Never the rent -- that is a deposit. Returns 0.0 when the SOL/USD
        reference is missing, because an unvaluable cost cannot be charged; the
        trade is then flagged ``known=False`` instead of being charged a guess.
        """
        return float(self.non_recoverable_usd or 0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": "OUTSIDE_SWAP_COSTS_MODELED",
            "known": self.known,
            "reason": self.reason,
            "legs": int(self.legs),
            "base_fee_lamports": int(self.base_fee_lamports),
            "priority_fee_lamports": int(self.priority_fee_lamports),
            "rent_lamports": int(self.rent_lamports),
            "non_recoverable_lamports": self.non_recoverable_lamports,
            "recoverable_lamports": self.recoverable_lamports,
            "total_reserve_lamports": self.total_reserve_lamports,
            "non_recoverable_usd": self.non_recoverable_usd,
            "recoverable_usd": self.recoverable_usd,
            "total_reserve_usd": self.total_reserve_usd,
            "charged_to_pnl_usd": self.charged_to_pnl_usd,
            "recoverable_deposit_not_a_loss": True,
            "assumes_new_token_account": bool(self.assumes_new_token_account),
            "sol_usd": float(self.sol_usd or 0.0),
            "sol_usd_source": self.sol_usd_source,
            "sol_usd_at": self.sol_usd_at,
            "provenance": dict(self.provenance),
            # -- DERIVATION AND WHAT IS **NOT** OBSERVED -------------------
            # The fee arithmetic is derived from the INTENDED transaction
            # configuration, and the priority fee is its CEILING. Said in
            # machine-readable form as well as in prose, so a report cannot
            # present a configured cap as an observed charge by accident.
            "derived_from": intended_transaction_config()["source"],
            "signatures_per_leg": int(self.signatures_per_leg),
            "priority_fee_is_ceiling": bool(self.priority_fee_is_ceiling),
            "priority_fee_source": self.priority_fee_source,
            "priority_fee_observed": False,
            "observed_note": (
                "the priority fee is the requested CEILING, not an observed "
                "charge: a dynamic-compute-unit-limit transaction pays "
                "ceil(compute_units_used x price_microlamports / 1e6), which is "
                "at most this. Deriving it exactly requires simulating and "
                "signing the intended transaction; PAPER does neither"),
        }


def failed_execution_accounting(
    *,
    quote_failures: int = 0,
    broadcast_attempts: int = 0,
    base_fee_lamports: int = SOLANA_BASE_FEE_LAMPORTS,
    priority_fee_lamports: int = PAPER_PRIORITY_FEE_LAMPORTS,
    signatures_per_leg: int = INTENDED_SIGNATURES_PER_LEG,
    sol_usd: float | None = None,
) -> dict[str, Any]:
    """Separate a FAILED QUOTE REQUEST from a FAILED TRANSACTION.

    They are not the same event and they do not cost the same:

    * ``quote_failures`` -- a provider refused or could not answer a quote
      request. Nothing was built, signed, simulated or sent, so the network cost
      is EXACTLY ZERO lamports. Reporting these as "failed attempts" that paid a
      fee would overstate the cost of the experiment.
    * ``broadcast_attempts`` -- transactions that reached the chain and failed.
      Each one DOES pay its base and priority fee (a failed transaction is still
      a charged transaction). PAPER has NONE: it builds and sends nothing.

    Returns the counts, the fee arithmetic for the broadcast case, and an
    explicit statement of what is modelled and what is not.
    """
    qf = max(0, int(quote_failures))
    bcast = max(0, int(broadcast_attempts))
    per_failed_tx = (int(base_fee_lamports) * int(signatures_per_leg)
                     + int(priority_fee_lamports))
    price = float(sol_usd or 0.0)
    failed_tx_fee = per_failed_tx * bcast
    return {
        "quote_failures": qf,
        "quote_failure_network_fee_lamports": 0,
        "quote_failure_note": (
            "a refused/failed QUOTE REQUEST builds, signs, simulates and sends "
            "nothing, so it incurs NO network fee; it is a provider-side "
            "refusal, not a failed transaction"),
        "broadcast_attempts": bcast,
        "per_failed_broadcast_lamports": per_failed_tx,
        "broadcast_failure_fee_lamports": failed_tx_fee,
        "broadcast_failure_fee_usd": (
            round(failed_tx_fee / 10 ** SOL_DECIMALS * price, 9)
            if price > 0 else None),
        "broadcast_failure_note": (
            "a BROADCAST transaction that fails still pays its base + priority "
            "fee. MODELLED HERE BUT NOT APPLIED: PAPER submits nothing, so this "
            "count is 0 by construction"),
        "modelled": [
            "quote-side refusals at zero network cost (both cases counted)"],
        "not_modelled": [
            "a real broadcast failure's fee is stated arithmetically but is "
            "never charged, because PAPER never broadcasts"],
    }


def slippage_boundary(
    *,
    measured_price_impact_pct: float | None = None,
    slippage_bps: int = INTENDED_TX_CONFIG["slippage_bps"],
    max_price_impact_pct: float = INTENDED_TX_CONFIG["max_price_impact_pct"],
) -> dict[str, Any]:
    """What the intended transaction WOULD set for slippage, and what PAPER sets.

    The intended transaction carries a minimum-output floor derived from
    ``slippage_bps`` and would be refused if the quote's price impact breached
    ``max_price_impact_pct``. PAPER builds no transaction, so it applies NO
    boundary of its own; the numbers are reported so the gap is stated rather
    than implied, and a MEASURED impact can be checked against the limit that
    would have applied.
    """
    floor_note = (
        f"{COST_CONFIGURED}: the intended transaction would set a minimum-output "
        f"floor from slippage_bps={int(slippage_bps)} "
        f"({int(slippage_bps) / 100.0:.2f}%)")
    impact = (None if measured_price_impact_pct is None
              else float(measured_price_impact_pct))
    return {
        "intended_slippage_bps": int(slippage_bps),
        "intended_slippage_pct": round(int(slippage_bps) / 100.0, 4),
        "intended_max_price_impact_pct": float(max_price_impact_pct),
        "applied_by_paper": False,
        "applied_by_paper_note": (
            "PAPER builds, signs and submits NO transaction, so it sets no "
            "minimum-output threshold and refuses nothing on price impact: the "
            "fill is simulated at the QUOTED price for the EXACT size"),
        "slippage_provenance": floor_note,
        "measured_price_impact_pct": impact,
        "measured_impact_provenance": (
            None if impact is None else
            f"{COST_MEASURED}: read from the live quote used by the probe"),
        "measured_impact_would_breach_limit": (
            None if impact is None else impact > float(max_price_impact_pct)),
        "not_modelled": [
            "the loss a real minimum-output boundary would cause on a fast "
            "adverse move: PAPER has no boundary to breach and no partial fill"],
    }


def outside_swap_costs(
    *,
    sol_usd: float | None,
    sol_usd_source: str = "",
    sol_usd_at: float | None = None,
    priority_fee_lamports: int = PAPER_PRIORITY_FEE_LAMPORTS,
    base_fee_lamports: int = SOLANA_BASE_FEE_LAMPORTS,
    signatures_per_leg: int = INTENDED_SIGNATURES_PER_LEG,
    legs: int = ROUND_TRIP_LEGS,
    assumes_new_token_account: bool = True,
    rent_lamports: int = TOKEN_ACCOUNT_RENT_LAMPORTS,
) -> OutsideSwapCosts:
    """
    The outside-swap costs for one round trip, or an explicitly UNKNOWN one.

    The base fee is DERIVED from the intended transaction configuration as
    ``signatures_per_leg x base_fee_lamports`` per leg, and the priority fee is
    the CONFIGURED CEILING the intended transaction requests -- never an observed
    charge. See :func:`intended_transaction_config`.

    A missing or non-positive SOL/USD reference produces ``known=False`` with the
    reason stated. It does NOT produce zero costs: an unvaluable cost is unknown,
    and the caller must not present it as free.
    """
    legs = int(legs)
    sigs = int(signatures_per_leg)
    base = int(base_fee_lamports) * sigs * legs
    priority = int(priority_fee_lamports) * legs
    rent = int(rent_lamports) if assumes_new_token_account else 0
    price = float(sol_usd or 0.0)
    provenance = {
        "base_fee_lamports": (
            f"{COST_DERIVED}: intended transaction has {sigs} signature(s) per "
            f"leg x Solana base fee {int(base_fee_lamports)} lamports/signature "
            f"x {legs} legs = {base}"),
        "priority_fee_lamports": (
            f"{COST_CONFIGURED}: PAPER priority fee "
            f"{int(priority_fee_lamports)} lamports/tx x {legs} legs = "
            f"{priority} ({PRIORITY_FEE_SOURCE_CEILING}: this is the CEILING the "
            "intended transaction requests -- mirrors "
            "RealConfig.priority_fee_lamports; NOT measured)"),
        "rent_lamports": (
            f"{COST_ASSUMED}: token-account rent {int(rent_lamports)} lamports "
            "reserved as if a NEW account is required (conservative: an existing "
            f"account only makes it cheaper) x "
            f"{1 if assumes_new_token_account else 0}"),
        "sol_usd": (COST_CONFIGURED if price > 0 else COST_ASSUMED),
    }
    if price <= 0:
        return OutsideSwapCosts(
            base_fee_lamports=base, priority_fee_lamports=priority,
            rent_lamports=rent, legs=legs, signatures_per_leg=sigs,
            sol_usd=0.0,
            sol_usd_source=sol_usd_source, sol_usd_at=sol_usd_at,
            known=False, assumes_new_token_account=assumes_new_token_account,
            reason=(
                "no SOL/USD reference, so the USD value of the outside-swap "
                "costs is UNKNOWN; the reserve is NOT applied in USD terms "
                "rather than being silently treated as zero"),
            provenance=provenance)
    return OutsideSwapCosts(
        base_fee_lamports=base, priority_fee_lamports=priority,
        rent_lamports=rent, legs=legs, signatures_per_leg=sigs,
        sol_usd=price,
        sol_usd_source=sol_usd_source, sol_usd_at=sol_usd_at,
        known=True, assumes_new_token_account=assumes_new_token_account,
        reason="", provenance=provenance)


def deployable_sol_lamports(
    *, available_usd: float, sol_usd: float | None,
    costs: OutsideSwapCosts,
) -> tuple[int, str]:
    """
    The swap-input lamports an ALL-IN arm can actually deploy, and why if none.

    ``deployable = floor(available_usd / sol_usd * 1e9) - reserve``

    FLOOR, never round: the notional IS the arm's whole deployable cash, so
    rounding up would ask the venue for more SOL than the arm can pay for once
    the fees are taken. The remainder is sub-lamport dust and is recorded rather
    than spent.

    When the SOL/USD reference is missing the reserve cannot be sized, so the
    answer is UNKNOWN and the caller must not open an arm on it.
    """
    price = float(sol_usd or 0.0)
    if price <= 0:
        return 0, (
            "no SOL/USD reference, so the outside-swap reserve cannot be "
            "sized and an ALL-IN amount cannot be shown to leave enough for it")
    available = float(available_usd or 0.0)
    if available <= 0:
        return 0, "the arm has no uncommitted cash to deploy"
    gross = int(available / price * 10 ** SOL_DECIMALS)
    reserve = int(costs.total_reserve_lamports)
    net = gross - reserve
    if net <= 0:
        return 0, (
            f"the arm's ${available:.9f} buys {gross} lamports, which does not "
            f"cover the {reserve} lamports of outside-swap costs (fees + token-"
            "account rent), so there is nothing deployable")
    return net, ""


__all__ = [
    "ATTEMPT_BROADCAST",
    "ATTEMPT_QUOTE_REQUEST",
    "COST_ASSUMED",
    "COST_CONFIGURED",
    "COST_DERIVED",
    "COST_MEASURED",
    "INTENDED_SIGNATURES_PER_LEG",
    "INTENDED_TX_CONFIG",
    "OutsideSwapCosts",
    "PAPER_PRIORITY_FEE_LAMPORTS",
    "PRIORITY_FEE_SOURCE_CEILING",
    "ROUND_TRIP_LEGS",
    "SOLANA_BASE_FEE_LAMPORTS",
    "deployable_sol_lamports",
    "failed_execution_accounting",
    "intended_transaction_config",
    "outside_swap_costs",
    "slippage_boundary",
]
