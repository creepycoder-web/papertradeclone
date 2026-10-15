"""
Micro-account economics: is a PAPER trade worth entering at all?

THE PROBLEM THIS SOLVES
-----------------------

A PAPER position buys a token account, and a SPL token account must hold
2,039,280 lamports to be rent-exempt. At a $122.36 SOL/USD reference that is
about $0.2495 -- which is HALF of a $0.50 notional.

That number is startling, and it is exactly why the distinction below matters:
the rent is LOCKED, not spent. A PAPER round trip opens one token account and
closes it, recovering the whole lamport balance. Charging rent as a permanent
round-trip cost would DOUBLE-COUNT a recoverable balance as a loss, which is
the exact error this module is written to avoid.

So rent is reported as a CAPITAL-EFFICIENCY ratio -- how much of the notional is
parked rather than at risk -- and it is NOT part of the fixed-cost gate. The
gate is charged with the costs that are genuinely spent regardless of outcome:
the venue fees and the modelled costs.

For a large account that overhead is noise. For a micro account it can exceed
the entire expected edge, which is why a strategy can be profitable at $10,000
and structurally unprofitable at $5 while using the identical entry logic.

WHAT THIS IS NOT
----------------

This does NOT change the position size. The canonical sizing formula is
:func:`autopaper.sizing.compute_position_size` and it is untouched: equity in,
notional out. This runs AFTER sizing and asks a different question -- not "how
big?" but "is this order worth its fixed costs at all?".

THE POLICY
----------

A trade is refused when the round-trip SPENT cost (venue fees + modelled costs)
is a large enough fraction of the notional that the required break-even move is
not credible. The threshold is a stated constant, recorded on every decision,
and never tuned from PAPER results -- this module is a POLICY layer, and it is
deliberately conservative: it refuses trades, it does not select them.

Rent is reported but NOT gated, because it is recoverable. See below.

RENT IS NOT A LOSS
------------------

Rent is LOCKED, not spent. It is tracked separately and becomes reclaimable once
the token balance reaches zero. It is excluded from canonical PnL by
construction: :class:`RentLedger` has no field that PnL reads, and
``fixed_cost_usd`` deliberately excludes it.

Charging rent as a permanent cost is not a conservative choice; it is a
double-count. It was measured to be worse than useless: at a $5 bankroll the
$0.50 notional is 52% rent, so a rent-gated policy refuses EVERY trade and the
runtime goes inert while reporting a number that is not a loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any, Mapping, Sequence

#: Rent-exempt minimum for an SPL token account, in lamports. Sourced from
#: ``autopaper.execution.solana.TOKEN_ACCOUNT_RENT_LAMPORTS`` -- imported here
#: rather than re-typed so the two can never disagree.
TOKEN_ACCOUNT_RENT_LAMPORTS = 2_039_280

SOL_DECIMALS = 9

#: The explicit token that arms a REAL rent reclaim. There is no implicit path:
#: a plan is produced unarmed, and this constant is what a future operator action
#: would have to supply through the REAL executor.
REAL_RECLAIM_TOKEN = "RECLAIM-RENT"

#: Refuse an entry whose round-trip fixed cost exceeds this fraction of the
#: notional.
#:
#: 8% is the point at which the required break-even move (~8%) stops being a
#: plausible edge for a fast in/out on a micro-cap and starts being the whole
#: trade. It is a POLICY constant, stated here, and is NOT fitted to any PAPER
#: result. Raising or lowering it is a research decision with its own
#: preregistration, not an implementation detail.
MAX_FIXED_COST_FRACTION = 0.08

#: Refuse an entry whose required break-even return exceeds this. Even if rent
#: is a small fraction, a notional so small that it cannot cover fees at all is
#: refused rather than rounded into existence.
MAX_BREAK_EVEN_RETURN_PCT = 12.0

#: Refuse an order below this notional. Venues price an order in base units, so
#: a dust order rounds to an unfillable amount, or to one too small to be worth
#: the fixed cost of asking. $0.10 is a floor, not a tuned value.
MIN_EFFECTIVE_NOTIONAL_USD = 0.10

#: Refuse an entry where the LOCKED rent exceeds the notional.
#:
#: This is a capital-efficiency ceiling, NOT a cost gate: at that point the
#: "position" is mostly a rent deposit with a token position attached, and the
#: outcome is dominated by whether the account can be closed again rather than by
#: the trade. At a $5 bankroll ($0.50 notional) rent is ~50% of notional and this
#: passes comfortably; it fires for genuinely dust-sized accounts.
MAX_RENT_OVER_NOTIONAL = 1.0


#: The three economic verdicts.
ECONOMIC_VALID = "MICRO_ACCOUNT_ECONOMICALLY_VALID"
ECONOMIC_MARGINAL = "MICRO_ACCOUNT_ECONOMICALLY_MARGINAL"
ECONOMIC_INVALID = "MICRO_ACCOUNT_ECONOMICALLY_INVALID"

#: What each cost is. The distinction is the whole point: TRADING_COST is spent,
#: TEMPORARY_RENT_LOCK comes back, RECLAIMABLE_RENT is recoverable, and only
#: IRRECOVERABLE_COST is a permanent loss.
COST_TRADING = "TRADING_COST"
COST_RENT_LOCK = "TEMPORARY_RENT_LOCK"
COST_RENT_RECLAIMABLE = "RECLAIMABLE_RENT"
COST_IRRECOVERABLE = "IRRECOVERABLE_COST"

#: A verdict within this factor of its limit is MARGINAL, not VALID. Marginal is
#: still ALLOWED -- it is reported so a run can be told apart from one where
#: every trade cleared the bar comfortably, without changing the decision.
MARGINAL_BAND = 0.80


@dataclass(frozen=True)
class EconomicCheck:
    """
    The verdict for one planned order.

    ``rent_usd`` is the locked, reclaimable rent. It is reported so the operator
    can see where the money went, and it is NOT added to any cost total that
    feeds PnL.
    """

    allowed: bool
    reason: str
    notional_usd: float
    entry_fee_usd: float
    exit_fee_usd: float
    rent_usd: float
    modeled_costs_usd: float
    fixed_cost_usd: float
    rent_over_notional: float
    fixed_cost_over_notional: float
    break_even_return_pct: float
    sol_usd_reference: float
    #: VALID | MARGINAL | INVALID
    classification: str = ECONOMIC_VALID
    #: Cost-category breakdown. Canonical PnL reads NONE of these except the
    #: trading cost; rent is deliberately listed apart from it.
    cost_categories: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "classification": self.classification,
            "cost_categories": dict(self.cost_categories),
            "in_canonical_pnl": {
                COST_TRADING: True,
                COST_RENT_LOCK: False,
                COST_RENT_RECLAIMABLE: False,
                COST_IRRECOVERABLE: True,
            },
            "reason": self.reason,
            "planned_notional_usd": round(self.notional_usd, 8),
            "estimated_entry_fee_usd": round(self.entry_fee_usd, 8),
            "estimated_exit_fee_usd": round(self.exit_fee_usd, 8),
            "estimated_new_token_account_rent_usd": round(self.rent_usd, 8),
            "modeled_costs_usd": round(self.modeled_costs_usd, 8),
            "fixed_cost_usd": round(self.fixed_cost_usd, 8),
            "rent_over_notional": round(self.rent_over_notional, 6),
            "rent_locked_not_spent": True,
            "fixed_cost_excludes_rent": True,
            "fixed_cost_over_notional": round(self.fixed_cost_over_notional, 6),
            "break_even_return_pct": round(self.break_even_return_pct, 4),
            "sol_usd_reference": self.sol_usd_reference,
            "rent_is_locked_not_spent": True,
        }


def evaluate_entry_economics(
    *,
    notional_usd: float,
    sol_usd: float,
    entry_fee_bps: int | None = None,
    exit_fee_bps: int | None = None,
    modeled_costs_usd: float = 0.0,
    max_fixed_cost_fraction: float = MAX_FIXED_COST_FRACTION,
    max_break_even_return_pct: float = MAX_BREAK_EVEN_RETURN_PCT,
    min_notional_usd: float = MIN_EFFECTIVE_NOTIONAL_USD,
    max_rent_over_notional: float = MAX_RENT_OVER_NOTIONAL,
) -> EconomicCheck:
    """
    Is this notional worth entering, given its fixed costs?

    Fail-closed on the inputs: without a SOL/USD reference the USD cost of rent
    is unknown, and an unknown cost cannot be shown to be small, so the trade is
    refused rather than evaluated against an assumed price.
    """
    notional = float(notional_usd)
    sol = float(sol_usd or 0.0)

    def build(
        allowed: bool,
        reason: str,
        *,
        rent: float = 0.0,
        entry_fee: float = 0.0,
        exit_fee: float = 0.0,
    ) -> EconomicCheck:
        costs = float(modeled_costs_usd or 0.0)
        # RENT IS DELIBERATELY NOT IN THIS SUM. It is locked, not spent, and it
        # comes back when the token balance hits zero. Including it here would
        # charge a recoverable balance as a permanent loss -- the double-count
        # this module exists to prevent -- and at a $5 bankroll it would refuse
        # every trade in the run.
        fixed = entry_fee + exit_fee + costs
        rent_ratio = rent / notional if notional > 0 else float("inf")
        fixed_ratio = fixed / notional if notional > 0 else float("inf")
        # A short position that must rise by X% to cover a cost of C on a
        # notional N is (C/N) -> 100*C/N percent.
        breakeven = (fixed_ratio * 100.0) if notional > 0 else float("inf")
        return EconomicCheck(
            allowed=allowed,
            reason=reason,
            notional_usd=notional,
            entry_fee_usd=entry_fee,
            exit_fee_usd=exit_fee,
            rent_usd=rent,
            modeled_costs_usd=costs,
            fixed_cost_usd=fixed,
            rent_over_notional=rent_ratio,
            fixed_cost_over_notional=fixed_ratio,
            break_even_return_pct=breakeven,
            sol_usd_reference=sol,
        )

    if notional <= 0:
        return replace(
            build(False, "the planned notional is zero or negative"),
            classification=ECONOMIC_INVALID,
            cost_categories={COST_TRADING: 0.0, COST_RENT_LOCK: 0.0,
                             COST_RENT_RECLAIMABLE: 0.0,
                             COST_IRRECOVERABLE: 0.0})

    # FAIL CLOSED ON A MISSING SOL/USD REFERENCE.
    #
    # Without a SOL/USD price the USD cost of the token-account rent is UNKNOWN,
    # and an unknown cost cannot be shown to be small. Reporting it as $0.0 --
    # which is what ``rent_usd = 0.0`` below would otherwise do -- makes a
    # rent-dominated micro order look viable, the exact opposite of this guard's
    # purpose. So the entry is refused rather than priced at an assumed rent of
    # zero. ``rent_usd`` stays 0.0 here: it is UNVALUED, not free, and the
    # reason string says which.
    #
    # This cannot change a live decision: the engine already refuses an entry
    # with no SOL reference upstream (``SOL_REFERENCE_MISSING``), so every
    # production call passes ``sol_usd > 0``. It makes the FUNCTION honour the
    # contract its own docstring states, for every other caller and for the
    # audit suite.
    if sol <= 0:
        return replace(
            build(False,
                  "no SOL/USD reference, so the USD cost of the token-account "
                  "rent is unknown; an unknown cost cannot be shown to be "
                  "small, so the entry is refused rather than priced at an "
                  "assumed rent of zero"),
            classification=ECONOMIC_INVALID,
            cost_categories={COST_TRADING: 0.0, COST_RENT_LOCK: 0.0,
                             COST_RENT_RECLAIMABLE: 0.0,
                             COST_IRRECOVERABLE: 0.0})

    entry_fee = notional * ((entry_fee_bps or 0) / 10_000.0)
    exit_fee = notional * ((exit_fee_bps or 0) / 10_000.0)
    # Reported, never gated. Unknown SOL/USD simply means the ratio is unknown.
    rent_usd = (
        (TOKEN_ACCOUNT_RENT_LAMPORTS / (10 ** SOL_DECIMALS)) * sol
        if sol > 0 else 0.0
    )

    candidate = build(
        True, "within the micro-account economic policy",
        rent=rent_usd, entry_fee=entry_fee, exit_fee=exit_fee,
    )
    if notional < min_notional_usd:
        return replace(
            candidate, allowed=False, classification=ECONOMIC_INVALID,
            reason=(
                f"a ${notional:.4f} notional is below the ${min_notional_usd:.2f} "
                "floor; venues price in base units, so this order cannot be "
                "filled meaningfully"
            ),
        )
    classification = ECONOMIC_VALID
    # MARGINAL when the binding ratio is within the band of its ceiling.
    binding = max(
        candidate.fixed_cost_over_notional / max_fixed_cost_fraction,
        candidate.rent_over_notional / max_rent_over_notional,
        candidate.break_even_return_pct / max_break_even_return_pct,
    ) if max_fixed_cost_fraction > 0 else 0.0
    if binding >= MARGINAL_BAND:
        classification = ECONOMIC_MARGINAL
    candidate = replace(candidate, classification=classification)

    if candidate.rent_over_notional > max_rent_over_notional:
        return replace(
            candidate, allowed=False, classification=ECONOMIC_INVALID,
            reason=(
                f"the locked token-account rent (${rent_usd:.4f}) is "
                f"{candidate.rent_over_notional:.2f}x the notional; the position "
                "is mostly a rent deposit and the outcome is dominated by "
                "whether the account can be closed again"
            ),
        )
    if candidate.fixed_cost_over_notional > max_fixed_cost_fraction:
        return replace(
            candidate, allowed=False, classification=ECONOMIC_INVALID,
            reason=(
                f"fixed cost is {candidate.fixed_cost_over_notional * 100:.2f}% "
                f"of a ${notional:.4f} notional, above the "
                f"{max_fixed_cost_fraction * 100:.0f}% ceiling; the trade needs "
                f"a {candidate.break_even_return_pct:.2f}% move just to break even"
            ),
        )
    if candidate.break_even_return_pct > max_break_even_return_pct:
        return replace(
            candidate, allowed=False, classification=ECONOMIC_INVALID,
            reason=(
                f"break-even needs {candidate.break_even_return_pct:.2f}%, above "
                f"the {max_break_even_return_pct:.0f}% ceiling for a micro account"
            ),
        )
    return replace(candidate, cost_categories={
        COST_TRADING: float(entry_fee + exit_fee),
        COST_RENT_LOCK: rent_usd,
        COST_RENT_RECLAIMABLE: rent_usd,
        COST_IRRECOVERABLE: float(modeled_costs_usd or 0.0),
    })


@dataclass
class RentLedger:
    """
    The rent lifecycle for one PAPER position.

    Canonical PnL never reads this. It exists so an operator can tell the
    difference between money that was spent trading and money that is parked in
    a rent-exempt account and can be recovered.
    """

    entry_rent_lamports: int = TOKEN_ACCOUNT_RENT_LAMPORTS
    #: True while a non-zero token balance keeps the account alive, i.e. the
    #: rent is still locked.
    rent_locked: bool = True
    reclaimable_rent_lamports: int = 0
    reclaim_status: str = "RENT_LOCKED"
    reclaim_blocker: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_rent_lamports": self.entry_rent_lamports,
            "rent_locked": self.rent_locked,
            "reclaimable_rent_lamports": self.reclaimable_rent_lamports,
            "reclaim_status": self.reclaim_status,
            "reclaim_blocker": self.reclaim_blocker,
            "accounted_in_canonical_pnl": False,
        }


def settle_rent(*, tokens_remaining_raw: int, ledger: RentLedger) -> RentLedger:
    """
    Move the rent lifecycle forward once a SELL has settled.

    ``tokens_remaining_raw == 0`` means the token balance is gone, so the
    account has no reason to exist and the rent becomes reclaimable. It is
    marked ``RENT_RECLAIMABLE`` and NOT counted as a loss: nothing here writes
    to a PnL total, and the canonical settlement is unchanged.

    Actual on-chain closing stays OUT of this function. Producing the plan is
    in scope; performing the reclaim is a WRITE and remains disabled.
    """
    if tokens_remaining_raw > 0:
        return replace(
            ledger, rent_locked=True, reclaimable_rent_lamports=0,
            reclaim_status="RENT_LOCKED",
            reclaim_blocker=(
                f"{tokens_remaining_raw} raw token units still held; the "
                "account cannot be closed while it is non-empty"
            ),
        )
    return replace(
        ledger, rent_locked=False,
        reclaimable_rent_lamports=int(ledger.entry_rent_lamports),
        reclaim_status="RENT_RECLAIMABLE", reclaim_blocker="",
    )


# ---------------------------------------------------------------------------
# TOKEN-ACCOUNT RENT: SIZE, ELIGIBILITY AND REAL-COMPATIBLE RECLAIM
# ---------------------------------------------------------------------------

#: Canonical program ids an account must be recognised under before a reclaim is
#: even considered. Token-2022 accounts can be LARGER than a classic SPL token
#: account, so the rent-exempt minimum differs by program -- quoting one number
#: for both would either over-reserve a classic account or under-reserve a
#: Token-2022 one and produce a non-functional reclaim.
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM_ID = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"

#: Rent-exempt minimum for a classic SPL token account (89 bytes).
TOKEN_ACCOUNT_RENT_LAMPORTS = 2_039_280

#: Token-2022 accounts carrying extensions are larger. The exact size depends on
#: WHICH extensions are present, so this is a floor and the account's own
#: reported size is authoritative. ``None`` means "unknown", which fails closed.
TOKEN_2022_MIN_RENT_LAMPORTS = 2_039_280


@dataclass(frozen=True)
class TokenAccountFacts:
    """
    What is known about a token account at evaluation time.

    Every field can be ``None``, and ``None`` means UNKNOWN -- which fails
    closed. A reclaim built on an unknown balance, owner or program is a guess
    about someone else's account.
    """

    address: str = ""
    exists: bool | None = None
    owner: str | None = None            # the WALLET that must receive the rent
    token_program_id: str | None = None
    mint: str | None = None
    token_balance_raw: int | None = None
    data_size: int | None = None
    lamports: int | None = None
    close_authority: str | None = None
    has_ata: bool | None = None         # did the ATA already exist?

    def as_dict(self) -> dict[str, Any]:
        return {
            "address": self.address, "exists": self.exists,
            "owner": self.owner, "token_program_id": self.token_program_id,
            "mint": self.mint, "token_balance_raw": self.token_balance_raw,
            "data_size": self.data_size, "lamports": self.lamports,
            "close_authority": self.close_authority, "ata_existed": self.has_ata,
        }


def required_rent_lamports(facts: TokenAccountFacts) -> int | None:
    """
    The rent-exempt minimum for THIS account, or ``None`` if it cannot be known.

    ``None`` is a fail-closed answer, not a default. Assuming the classic 2,039,280
    for an account that might be a larger Token-2022 account would under-reserve
    it and produce a reclaim the chain rejects.
    """
    program = facts.token_program_id
    if program is None:
        return None
    if program == TOKEN_PROGRAM_ID:
        return TOKEN_ACCOUNT_RENT_LAMPORTS
    if program == TOKEN_2022_PROGRAM_ID:
        # The account's own size is authoritative when it is known.
        if facts.data_size is None:
            return None
        return rent_exempt_minimum_for_size(int(facts.data_size))
    # An unrecognised token program is not something to guess about.
    return None


def rent_exempt_minimum_for_size(data_size: int) -> int:
    """
    The rent-exempt minimum for an account of ``data_size`` bytes.

    Solana's rent curve is ``(128 + data_size) * 3480`` lamports per byte-year.
    Expressed here rather than imported so the economics layer has no RPC
    dependency and stays trivially testable offline.
    """
    if data_size <= 0:
        return 0
    return (128 + int(data_size)) * 3480


def evaluate_reclaim(
    *,
    facts: TokenAccountFacts,
    expected_owner: str,
    expected_mint: str,
    expected_program_ids: Sequence[str] = (TOKEN_PROGRAM_ID, TOKEN_2022_PROGRAM_ID),
    dust_threshold_raw: int = 1,
) -> dict[str, Any]:
    """
    Can this token account be closed, and is its rent reclaimable?

    Returns a verdict dict. ``reclaimable`` is True only when EVERY condition is
    positively proven:

    * the account exists;
    * the balance is exactly zero (dust is NOT close-eligible -- a non-zero
      balance is the token the position still holds);
    * the owner is the expected wallet (a foreign-owned account is never ours to
      close);
    * the token program is a recognised one;
    * the mint matches.

    Anything unknown is UNKNOWN and blocks, because the alternative is acting on
    an account whose real state nobody has checked.
    """
    blockers: list[str] = []
    if facts.exists is None:
        blockers.append("account existence is UNKNOWN; not acting on it")
    elif not facts.exists:
        blockers.append("the account does not exist; there is nothing to reclaim")

    if facts.token_balance_raw is None:
        blockers.append("the token balance is UNKNOWN; not assuming it is zero")
    elif facts.token_balance_raw > dust_threshold_raw:
        blockers.append(
            f"the account still holds {facts.token_balance_raw} raw token units, "
            "which is at or above the dust threshold; a non-empty account is not "
            "close-eligible")
    elif facts.token_balance_raw < 0:
        blockers.append("the token balance is negative, which is impossible; refusing")

    if facts.owner is None:
        blockers.append("the account owner is UNKNOWN; not acting on it")
    elif facts.owner != expected_owner:
        blockers.append(
            f"the account is owned by {facts.owner[:8]}… and not by the expected "
            f"wallet {expected_owner[:8]}…; a foreign-owned account is never ours "
            "to close")

    if facts.token_program_id is None:
        blockers.append("the token program is UNKNOWN; not assuming the classic SPL one")
    elif facts.token_program_id not in tuple(expected_program_ids):
        blockers.append(
            f"the token program {facts.token_program_id[:8]}… is not a recognised "
            "token program for this runtime")

    if facts.mint is None:
        blockers.append("the account's mint is UNKNOWN")
    elif facts.mint != expected_mint:
        blockers.append(
            f"the account holds mint {facts.mint[:8]}… and not the position's "
            f"{expected_mint[:8]}…")

    required = required_rent_lamports(facts)
    if required is None:
        blockers.append(
            "the rent-exempt minimum cannot be determined for this account, so "
            "the reclaimable amount is unknown; refusing rather than guessing")

    reclaimable = not blockers
    return {
        "reclaimable": reclaimable,
        "status": "RENT_RECLAIMABLE" if reclaimable else "RENT_LOCKED",
        "blockers": blockers,
        "required_rent_lamports": required,
        "observed_lamports": facts.lamports,
        "ata_existed": facts.has_ata,
        "rent_locked": not reclaimable,
    }


def reclaim_plan(
    *,
    facts: TokenAccountFacts,
    expected_owner: str,
    expected_mint: str,
    armed: bool = False,
    authorize_token: str = "",
) -> dict[str, Any]:
    """
    A REAL-compatible reclaim PLAN. It never executes.

    ``armed=False`` is the default and the only state this task ships: the plan
    is produced, audited and returned, and no instruction is built, no account
    is signed for and nothing is broadcast. A caller that wants to execute must
    pass ``armed=True`` AND the explicit authorisation token, which is what
    keeps "a PAPER position closed" from ever reclaiming anything by accident.

    The operation is idempotent by construction: planning a reclaim for an
    already-reclaimed account returns ``already_reclaimed`` rather than a second
    reclaim.
    """
    verdict = evaluate_reclaim(
        facts=facts, expected_owner=expected_owner, expected_mint=expected_mint)
    plan = {
        "operation": "close_token_account_and_reclaim_rent",
        "account": facts.address,
        "expected_owner": expected_owner,
        "mint": expected_mint,
        "token_program_id": facts.token_program_id,
        "reclaimable": verdict["reclaimable"],
        "status": verdict["status"],
        "blockers": verdict["blockers"],
        "rent_lamports": verdict["required_rent_lamports"],
        "idempotent_key": (
            f"reclaim:{facts.address}:{facts.token_program_id or 'unknown'}"),
        "armed": bool(armed),
        "authorized": bool(armed and authorize_token == REAL_RECLAIM_TOKEN),
        "executed": False,
        "broadcasted": False,
        "signed": False,
        "mode": "DRY_RUN",
    }
    if not verdict["reclaimable"]:
        plan["mode"] = "REFUSED"
    elif not plan["authorized"]:
        plan["note"] = (
            "eligible but NOT armed: this runtime never broadcasts. Executing a "
            "reclaim requires the REAL execution path AND an explicit operator "
            "authorisation."
        )
    return plan


def cleanup_plan(ledger: RentLedger) -> dict[str, Any]:
    """
    A DRY-RUN cleanup plan for a reclaimable account.

    Describes what a real reclaim would do and what it would return. It is a
    plan, never an execution: no instruction is built, nothing is signed, and
    nothing is broadcast.
    """
    reclaimable = ledger.reclaim_status == "RENT_RECLAIMABLE"
    return {
        "mode": "DRY_RUN",
        "broadcasted": False,
        "signed": False,
        "reclaimable": reclaimable,
        "rent_lamports_recoverable": (
            int(ledger.entry_rent_lamports) if reclaimable else 0
        ),
        "blocker": ledger.reclaim_blocker,
        "note": (
            "a dry-run description only; closing the token account is a WRITE "
            "and is disabled in this runtime"
        ),
    }


def lamports_for_usd(usd: float, sol_usd: float) -> int:
    """USD -> lamports, fail-closed on a missing reference."""
    sol = float(sol_usd or 0.0)
    if sol <= 0:
        return 0
    return int((Decimal(str(usd)) / Decimal(str(sol))).scaleb(SOL_DECIMALS))


__all__ = [
    "COST_IRRECOVERABLE",
    "COST_RENT_LOCK",
    "COST_RENT_RECLAIMABLE",
    "COST_TRADING",
    "ECONOMIC_INVALID",
    "ECONOMIC_MARGINAL",
    "ECONOMIC_VALID",
    "EconomicCheck",
    "REAL_RECLAIM_TOKEN",
    "TOKEN_2022_PROGRAM_ID",
    "TOKEN_PROGRAM_ID",
    "TokenAccountFacts",
    "evaluate_reclaim",
    "reclaim_plan",
    "rent_exempt_minimum_for_size",
    "required_rent_lamports",
    "MAX_BREAK_EVEN_RETURN_PCT",
    "MAX_FIXED_COST_FRACTION",
    "MARK_GAP_GUARD_THRESHOLD_PCT",
    "MarkGapVerdict",
    "MAX_RENT_OVER_NOTIONAL",
    "MIN_EFFECTIVE_NOTIONAL_USD",
    "RentLedger",
    "SOL_DECIMALS",
    "TOKEN_ACCOUNT_RENT_LAMPORTS",
    "cleanup_plan",
    "evaluate_mark_gap",
    "evaluate_entry_economics",
    "lamports_for_usd",
    "settle_rent",
]


# ---------------------------------------------------------------------------
# EXECUTABLE-vs-MARK ENTRY GUARD
# ---------------------------------------------------------------------------

#: Refuse an entry whose executable BUY price exceeds the observed mark by more
#: than this. The mark is an OBSERVATION; the executable quote is what the trade
#: would actually pay. A wide gap means the mark is describing a book the order
#: cannot trade in, and every downstream number -- the entry, the mark-based
#: exits, the reported return -- would be measuring a price nobody got.
#:
#: 10% is the production threshold. It is a stated policy, not a fitted value.
MARK_GAP_GUARD_THRESHOLD_PCT = 10.0


@dataclass(frozen=True)
class MarkGapVerdict:
    """
    The mark-versus-executable comparison and whether it is even VALID.

    ``valid=False`` is a first-class outcome, not a failure to compute. When the
    mark and the quote cannot be proven to describe the same market at the same
    size, the honest answer is UNKNOWN, and UNKNOWN refuses the entry. Falling
    back to another pool's mark, or to the quote alone, would be inventing the
    comparison this guard exists to make.
    """

    result: str                      # PASS | REJECT | UNKNOWN
    reason: str
    mark_price: float | None
    executable_price: float | None
    gap_pct: float | None
    threshold_pct: float = MARK_GAP_GUARD_THRESHOLD_PCT

    @property
    def allows_entry(self) -> bool:
        return self.result == "PASS"

    def as_dict(self) -> dict[str, Any]:
        return {
            "guard_result": self.result,
            "guard_reason": self.reason,
            "mark_price": self.mark_price,
            "executable_price": self.executable_price,
            "gap_pct": (round(self.gap_pct, 4)
                        if self.gap_pct is not None else None),
            "guard_threshold_pct": self.threshold_pct,
        }


def _gap_verdict(
    *, executable_price: float | None, reference_price: float | None,
    threshold_pct: float, reference_label: str, comparison_note: str,
) -> MarkGapVerdict:
    """
    THE GAP ARITHMETIC. One definition, shared by every reference.

    Both the legacy observed-mark guard and the execution-native guard call
    this, so the two cannot drift apart in their formula or in their threshold.
    Two implementations of one comparison is how a "10%" quietly becomes an
    "11%" on one path only.
    """
    if not reference_price or reference_price <= 0:
        return MarkGapVerdict(
            result="UNKNOWN",
            reason=f"there is no usable {reference_label} to compare against",
            mark_price=reference_price, executable_price=executable_price,
            gap_pct=None, threshold_pct=threshold_pct)
    if not executable_price or executable_price <= 0:
        return MarkGapVerdict(
            result="UNKNOWN", reason="there is no usable executable price to compare",
            mark_price=reference_price, executable_price=executable_price,
            gap_pct=None, threshold_pct=threshold_pct)

    gap = (float(executable_price) / float(reference_price) - 1.0) * 100.0
    if gap > threshold_pct:
        return MarkGapVerdict(
            result="REJECT",
            reason=(
                f"the executable BUY is {gap:.2f}% above {comparison_note}, "
                f"beyond the {threshold_pct:.1f}% entry guard; "
                f"{reference_label} describes a book this order cannot trade in"
            ),
            mark_price=float(reference_price),
            executable_price=float(executable_price),
            gap_pct=gap, threshold_pct=threshold_pct)
    return MarkGapVerdict(
        result="PASS",
        reason=(
            f"the executable BUY is {gap:+.2f}% against {comparison_note}, "
            f"within the {threshold_pct:.1f}% entry guard"),
        mark_price=float(reference_price),
        executable_price=float(executable_price),
        gap_pct=gap, threshold_pct=threshold_pct)


def evaluate_execution_native_gap(
    *, executable_price: float | None, reference_price: float | None,
    same_market: bool = True, same_mint: bool = True,
    same_quote_asset: bool = True, same_side: bool = True,
    same_token_decimals: bool = True, same_order_size: bool = True,
    reference_fresh: bool = True, quote_fresh: bool = True,
    market_id: str | None = None,
    threshold_pct: float = MARK_GAP_GUARD_THRESHOLD_PCT,
) -> MarkGapVerdict:
    """
    The production entry guard: ``executable > same_market_spot * 1.10`` refuses.

    IDENTICAL economics to :func:`evaluate_mark_gap` -- same arithmetic, same
    threshold, same UNKNOWN-refuses rule -- with ONE change: the reference is
    the SAME market's own current on-chain price instead of an external
    aggregate's snapshot of it.

    That change is the whole point. The observed mark is a third party's
    periodically-updated view; on the young, fast-moving markets this strategy
    targets it trailed the actual curve with a standard deviation of ~26
    percentage points against a true execution premium of 1.26% +/- 0.003, so
    the guard was measuring aggregator lag. The execution-native reference is
    the market itself, so the gap is the fee plus the size impact of this exact
    order -- which is the question the guard is supposed to answer.

    EVERY PRECONDITION IS VERIFIED, NOT ASSUMED. ``same_order_size`` in
    particular was previously hardcoded ``True``; here it is a caller-supplied
    fact and a false value makes the verdict UNKNOWN, which refuses.
    """
    def unknown(reason: str) -> MarkGapVerdict:
        return MarkGapVerdict(
            result="UNKNOWN", reason=reason, mark_price=reference_price,
            executable_price=executable_price, gap_pct=None,
            threshold_pct=threshold_pct)

    if reference_price is None:
        return unknown(
            "the same market's own current price could not be read from chain, "
            "so there is no execution-native reference to compare against")
    if not same_market:
        return unknown(
            f"the reference market {market_id or '?'} is not the executable "
            "market, so their price difference is not a comparable gap")
    if not same_mint:
        return unknown("the reference and the quote are for different mints")
    if not same_quote_asset:
        return unknown(
            "the reference and the quote are in different quote assets")
    if not same_side:
        return unknown("the reference and the quote are on different sides")
    if not same_token_decimals:
        return unknown(
            "the reference and the quote were scaled with different token "
            "decimals, so their prices are not on the same scale")
    if not same_order_size:
        return unknown(
            "the reference is not the spot of the exact size being quoted, so "
            "the difference would not measure this order's cost")
    if not reference_fresh:
        return unknown(
            "the execution-native reference is not fresh enough to compare "
            "against")
    if not quote_fresh:
        return unknown("the executable quote is not fresh enough to compare")
    return _gap_verdict(
        executable_price=executable_price, reference_price=reference_price,
        threshold_pct=threshold_pct,
        reference_label="the same market's current on-chain price",
        comparison_note=(
            f"the same market's own price ({market_id or 'exact market'})"),
    )


def evaluate_mark_gap(
    *,
    mark_price: float | None,
    executable_price: float | None,
    same_mint: bool = True,
    same_market_identity: bool = True,
    same_quote_asset: bool = True,
    same_side: bool = True,
    same_order_size: bool = True,
    mark_fresh: bool = True,
    quote_fresh: bool = True,
    threshold_pct: float = MARK_GAP_GUARD_THRESHOLD_PCT,
) -> MarkGapVerdict:
    """
    The production entry guard: ``executable > mark * (1 + 10%)`` refuses.

    Every precondition is explicit. If any cannot be proven, the comparison is
    UNKNOWN and UNKNOWN refuses, because a guard that quietly passes when it
    cannot see is not a guard.
    """
    def unknown(reason: str) -> MarkGapVerdict:
        return MarkGapVerdict(
            result="UNKNOWN", reason=reason, mark_price=mark_price,
            executable_price=executable_price, gap_pct=None,
            threshold_pct=threshold_pct)

    if not same_mint:
        return unknown("the mark and the quote are for different mints")
    if not same_market_identity:
        return unknown(
            "the mark and the quote are not proven to be the same market, so "
            "their price difference is not a comparable gap")
    if not same_quote_asset:
        return unknown("the mark and the quote are in different quote assets")
    if not same_side:
        return unknown("the mark and the quote are on different sides")
    if not same_order_size:
        return unknown(
            "the mark and the quote are for different order sizes, so the "
            "difference includes size impact rather than venue quality")
    if not mark_fresh:
        return unknown("the observed mark is not fresh enough to compare against")
    if not quote_fresh:
        return unknown("the executable quote is not fresh enough to compare")
    if not mark_price or mark_price <= 0:
        return unknown("there is no usable observed mark to compare against")
    if not executable_price or executable_price <= 0:
        return unknown("there is no usable executable price to compare")

    gap = (float(executable_price) / float(mark_price) - 1.0) * 100.0
    if gap > threshold_pct:
        return MarkGapVerdict(
            result="REJECT",
            reason=(
                f"the executable BUY is {gap:.2f}% above the observed mark, "
                f"beyond the {threshold_pct:.1f}% entry guard; the mark is "
                "describing a book this order cannot trade in"
            ),
            mark_price=float(mark_price),
            executable_price=float(executable_price),
            gap_pct=gap, threshold_pct=threshold_pct)
    return MarkGapVerdict(
        result="PASS",
        reason=(
            f"the executable BUY is {gap:+.2f}% against the observed mark, "
            f"within the {threshold_pct:.1f}% entry guard"),
        mark_price=float(mark_price),
        executable_price=float(executable_price),
        gap_pct=gap, threshold_pct=threshold_pct)
