"""
Journal <-> chain reconciliation states.

THE CHAIN IS AUTHORITATIVE
--------------------------
The bot's journal is a record of what the bot DID. The chain is a record of what
HAPPENED. They can disagree, and when they do the chain wins. This module names
the kinds of disagreement so the runtime can report one instead of silently
overwriting state, deleting a record, or refusing every future order with an
undifferentiated "mismatch".

THE $SI LESSON
--------------
A REAL position stayed ``OPEN`` in the journal while its on-chain balance was
zero, because the operator moved the tokens out in a transaction the bot never
saw. The reconciliation correctly detected the mismatch and then had NO state to
put it in -- so REAL stayed frozen process-wide. Detection without a named,
resolvable state is not reconciliation; it is a dead end.

THE STATES
----------
``EXTERNALLY_CLOSED``
    The journal held the position OPEN; the chain holds zero; value left without
    the bot selling. Resolvable by the operator with an on-chain signature.
``MANUAL_SETTLEMENT``
    The close was performed by the operator (Phantom/Jupiter/Pumpfun etc.) and
    the proceeds arrived in ANY asset -- SOL, WSOL, USDC or another token.
``SETTLED_NON_SOL``
    A close settled in an asset that is not spendable as SOL (USDC or another
    SPL/Token-2022 asset). It is a real sale with real proceeds; it is not a
    failed sale and must not be booked as SOL received = 0.
``UNKNOWN_EXTERNAL_ACTIVITY``
    The journal and the chain disagree in a way that cannot be attributed:
    e.g. the journal says CLOSED while the chain still holds tokens, or an
    unreadable balance. Fail closed and escalate.

NON-NEGOTIABLE RULES
--------------------
* No record is deleted.
* No historical record is overwritten. A reconciliation produces a NEW finding
  that references the old record; it never mutates it.
* An unreadable balance is a mismatch, never a pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# explicit reconciliation states
# ---------------------------------------------------------------------------

#: Bot sold it, or the operator moved it out; the chain says zero.
EXTERNALLY_CLOSED = "EXTERNALLY_CLOSED"
#: The operator closed it by hand and proceeds arrived in some asset.
MANUAL_SETTLEMENT = "MANUAL_SETTLEMENT"
#: A close settled in a non-SOL asset (USDC / other SPL / Token-2022).
SETTLED_NON_SOL = "SETTLED_NON_SOL"
#: Journal and chain disagree in a way that cannot be attributed. Fail closed.
UNKNOWN_EXTERNAL_ACTIVITY = "UNKNOWN_EXTERNAL_ACTIVITY"
#: Journal and chain agree: the position is open and the tokens are present.
CONSISTENT_OPEN = "CONSISTENT_OPEN"
#: Journal and chain agree: the position is closed and the tokens are absent.
CONSISTENT_CLOSED = "CONSISTENT_CLOSED"
#: The chain could not be read. Never treated as a match.
BALANCE_UNREADABLE = "BALANCE_UNREADABLE"

#: Every state this module can return.
RECONCILIATION_STATES = (
    EXTERNALLY_CLOSED,
    MANUAL_SETTLEMENT,
    SETTLED_NON_SOL,
    UNKNOWN_EXTERNAL_ACTIVITY,
    CONSISTENT_OPEN,
    CONSISTENT_CLOSED,
    BALANCE_UNREADABLE,
)

#: States that describe a real, resolvable external event.
EXTERNAL_EVENT_STATES = (EXTERNALLY_CLOSED, MANUAL_SETTLEMENT, SETTLED_NON_SOL)

#: States that must fail closed.
FAIL_CLOSED_STATES = (UNKNOWN_EXTERNAL_ACTIVITY, BALANCE_UNREADABLE)

#: Mints that count as "spendable as SOL" for settlement attribution.
SOL_LIKE_SETTLEMENT_MINTS = frozenset({
    "SOL",
    "So11111111111111111111111111111111111111112",   # WSOL
})

#: USDC. The asset the $SI position actually settled in.
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


@dataclass(frozen=True)
class ReconciliationFinding:
    """One journal position checked against the chain. Never mutates state."""

    mint: str
    symbol: str = ""
    journal_status: str = ""
    chain_balance: Optional[int] = None
    state: str = UNKNOWN_EXTERNAL_ACTIVITY
    reason: str = ""
    settlement_asset: str = ""
    is_sol_settlement: bool = False
    requires_operator: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def consistent(self) -> bool:
        return self.state in (CONSISTENT_OPEN, CONSISTENT_CLOSED)

    @property
    def fails_closed(self) -> bool:
        return self.state in FAIL_CLOSED_STATES

    def as_dict(self) -> dict[str, Any]:
        return {
            "mint": self.mint,
            "symbol": self.symbol,
            "journal_status": self.journal_status,
            "chain_balance": self.chain_balance,
            "state": self.state,
            "reason": self.reason,
            "settlement_asset": self.settlement_asset,
            "is_sol_settlement": self.is_sol_settlement,
            "requires_operator": self.requires_operator,
            "consistent": self.consistent,
            "fails_closed": self.fails_closed,
            "evidence": self.evidence,
        }


def classify_settlement_asset(asset_mint: str) -> tuple[str, bool]:
    """
    Return ``(state, is_sol)`` for a settlement asset.

    An empty asset means "not observed". It resolves to ``EXTERNALLY_CLOSED``
    with ``is_sol=False`` -- the close is real but the proceeds are UNSTATED
    until a signature is checked. It must never be read as ``SOL_received=0``.
    A SOL/WSOL asset is the clean case; anything else is ``SETTLED_NON_SOL``.
    """
    if not asset_mint:
        return EXTERNALLY_CLOSED, False
    if asset_mint in SOL_LIKE_SETTLEMENT_MINTS:
        return EXTERNALLY_CLOSED, True
    return SETTLED_NON_SOL, False


def classify_position_against_chain(
    *,
    mint: str,
    journal_status: str,
    chain_balance: Optional[int],
    symbol: str = "",
    settlement_asset: str = "",
    manual: bool = False,
    evidence: Optional[dict[str, Any]] = None,
) -> ReconciliationFinding:
    """
    Compare ONE journal position with the wallet's on-chain balance of it.

    ``journal_status`` is the position's status (``OPEN`` / ``CLOSED`` /
    ``AMBIGUOUS``). ``chain_balance`` is the RAW token balance owned by the
    configured wallet, or ``None`` when it could not be read.

    A ``settlement_asset`` may be supplied when the close was observed; it is
    what distinguishes a non-SOL settlement from a plain external close. It is
    NEVER inferred -- an unobserved settlement asset stays unnamed.
    """
    ev = dict(evidence or {})
    if chain_balance is None:
        return ReconciliationFinding(
            mint=mint, symbol=symbol, journal_status=journal_status,
            chain_balance=None, state=BALANCE_UNREADABLE,
            reason="the wallet's balance of this mint could not be read; an "
                   "unanswered read is a mismatch, not a match",
            settlement_asset=settlement_asset,
            requires_operator=True, evidence=ev,
        )

    balance = int(chain_balance)
    status = str(journal_status or "").upper()
    is_open = status == "OPEN"

    # -- journal OPEN, chain holds tokens: consistent ----------------------
    if is_open and balance > 0:
        return ReconciliationFinding(
            mint=mint, symbol=symbol, journal_status=status,
            chain_balance=balance, state=CONSISTENT_OPEN,
            reason="journal OPEN and the chain still holds the tokens",
            evidence=ev,
        )

    # -- journal CLOSED, chain empty: consistent ---------------------------
    if not is_open and balance == 0:
        return ReconciliationFinding(
            mint=mint, symbol=symbol, journal_status=status,
            chain_balance=0, state=CONSISTENT_CLOSED,
            reason="journal closed and the chain holds no such tokens",
            evidence=ev,
        )

    # -- journal OPEN, chain empty: the $SI shape --------------------------
    if is_open and balance == 0:
        if manual:
            state = MANUAL_SETTLEMENT
            reason = ("the position was closed by hand (outside the bot) and "
                      "the tokens have left this wallet")
        else:
            state, _is_sol = classify_settlement_asset(settlement_asset)
            if state == EXTERNALLY_CLOSED:
                reason = ("journal OPEN but the chain holds nothing; the tokens "
                          "left this wallet without the bot selling them")
            else:
                reason = ("journal OPEN but the chain holds nothing; the close "
                          "settled in a NON-SOL asset -- this is a real sale "
                          "with real proceeds, not SOL_received=0 and not a "
                          "failed sale")
        _state, is_sol = classify_settlement_asset(settlement_asset)
        return ReconciliationFinding(
            mint=mint, symbol=symbol, journal_status=status,
            chain_balance=0, state=state, reason=reason,
            settlement_asset=settlement_asset, is_sol_settlement=is_sol,
            requires_operator=True, evidence=ev,
        )

    # -- journal CLOSED, chain still holds: unattributable -----------------
    return ReconciliationFinding(
        mint=mint, symbol=symbol, journal_status=status,
        chain_balance=balance, state=UNKNOWN_EXTERNAL_ACTIVITY,
        reason=("journal says the position is closed but the chain still holds "
                f"{balance} raw units; refusing to guess whether this is a "
                "failed sale, a re-deposit, or a different position"),
        settlement_asset=settlement_asset,
        requires_operator=True, evidence=ev,
    )


def reconcile_journal_against_chain(
    positions: Iterable[Any],
    chain_balances: dict[str, Optional[int]],
    *,
    settlements: Optional[dict[str, str]] = None,
    manual_mints: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """
    Check every journal position against the chain and summarise the result.

    READ ONLY. ``positions`` is any iterable exposing ``mint``, ``symbol`` and
    ``status``. ``chain_balances`` maps mint -> raw balance (or ``None`` when
    unreadable). ``settlements`` optionally maps mint -> settlement asset for a
    close that was observed. ``manual_mints`` names mints the operator settled
    by hand.

    Returns a report. It deletes nothing and overwrites nothing: each finding
    references the position, and the underlying record is left intact.
    """
    settles = dict(settlements or {})
    manual = set(manual_mints or ())
    findings: list[ReconciliationFinding] = []
    for pos in positions or ():
        mint = str(getattr(pos, "mint", "") or "")
        if not mint:
            continue
        findings.append(
            classify_position_against_chain(
                mint=mint,
                symbol=str(getattr(pos, "symbol", "") or ""),
                journal_status=str(getattr(pos, "status", "") or ""),
                chain_balance=chain_balances.get(mint),
                settlement_asset=settles.get(mint, ""),
                manual=mint in manual,
            )
        )

    by_state: dict[str, int] = {}
    for f in findings:
        by_state[f.state] = by_state.get(f.state, 0) + 1
    actionable = [f for f in findings if f.requires_operator]
    consistent = all(f.consistent for f in findings)
    return {
        "checked": len(findings),
        "consistent": consistent,
        "by_state": by_state,
        "actionable": [f.as_dict() for f in actionable],
        "findings": [f.as_dict() for f in findings],
        "states": list(RECONCILIATION_STATES),
    }
