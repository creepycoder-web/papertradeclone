"""
Settlement classification: what asset did this exit ACTUALLY return?

THE PROBLEM THIS SOLVES
-----------------------
The exit path assumed every sell settled into native SOL. That assumption is
convenient and, on Solana, wrong. A Pump.fun or Jupiter sell CAN return:

* native SOL / WSOL        -- the expected case;
* USDC (or any other SPL)  -- a routed stablecoin settlement, which lands as a
  token balance in the seller's own wallet and does NOT exist as spendable SOL
  until it is converted;
* a token credited to a DIFFERENT account -- a bundled or forwarded settlement
  this bot does not control and can never spend;
* nothing at all -- a transfer, a burn, or a partial fill.

An exit that returned USDC was previously recorded as if it had returned SOL. The
position was closed, proceeds were attributed, and the money was simultaneously
missing and marked as received. That is the exact class of bug where the ledger
disagrees with the wallet and nothing catches it.

THE RULES
---------
1. THE CHAIN DECIDES. The output asset is read from the confirmed transaction's
   token balances, not from the request, the quote, or the provider's optimism.
2. ONLY THE SIGNER'S OWN WALLET COUNTS AS PROCEEDS. A credit to another account
   is recorded as UNCONTROLLED and is never treated as spendable money.
3. A NON-SOL SETTLEMENT IS NOT SILENTLY ACCEPTED. It is classified, reported, and
   -- when the asset is one this bot may convert -- handed to the follow-up
   normalization swap, which is a SEPARATE order with its own intent, quote,
   safety checks, simulation, signature, broadcast, confirmation and
   reconciliation. It is not a second leg of the sell.
4. FAIL CLOSED. Anything unrecognised, unreadable, or ambiguous is refused and
   left for the operator, never assumed to be SOL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

#: Wrapped SOL. Settles as a token balance but is spendable as SOL after
#: unwrap, so it is classified with SOL rather than as "some other SPL".
WSOL_MINT = "So11111111111111111111111111111111111111112"

#: USDC. The settlement that motivated this module.
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


class Settlement:
    """The settlement classes. Each has a different correct action."""

    #: Native SOL or WSOL reached the signer. Nothing more to do.
    SETTLED_SOL = "SETTLED_SOL"
    #: A recognized, signer-controlled SPL asset came back. Needs a follow-up
    #: conversion before the money is spendable.
    SETTLED_SPL_NEEDS_NORMALIZATION = "SETTLED_SPL_NEEDS_NORMALIZATION"
    #: The output went to an account this bot does not control. It is money that
    #: exists but is not ours to spend.
    UNCONTROLLED = "UNCONTROLLED_SETTLEMENT"
    #: The signer was credited nothing measurable. Not a sale.
    NO_PROCEEDS = "NO_PROCEEDS"
    #: The transaction could not be read back. Must not be classified by guess.
    UNREADABLE = "UNREADABLE"


#: Assets this bot is permitted to convert back to SOL itself. Anything not in
#: this set is reported and left to the operator: automatically swapping an
#: unrecognised token is how a bot ends up trading something nobody approved.
NORMALIZABLE_MINTS = frozenset({USDC_MINT})


@dataclass(frozen=True)
class SettlementReading:
    """What the chain says the exit returned, and what must happen next."""

    settlement: str
    output_mint: str = ""
    output_amount: int = 0
    #: True when the credited account is the configured signer's own.
    controlled: bool = False
    #: Mints credited to an account that is NOT the signer.
    external_mints: tuple[str, ...] = ()
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_settled_sol(self) -> bool:
        return self.settlement == Settlement.SETTLED_SOL

    @property
    def needs_normalization(self) -> bool:
        return self.settlement == Settlement.SETTLED_SPL_NEEDS_NORMALIZATION

    @property
    def is_problem(self) -> bool:
        """Anything that must not be treated as a clean, completed sale."""
        return self.settlement in (
            Settlement.UNCONTROLLED,
            Settlement.NO_PROCEEDS,
            Settlement.UNREADABLE,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "settlement": self.settlement,
            "output_mint": self.output_mint,
            "output_amount": self.output_amount,
            "controlled": self.controlled,
            "external_mints": list(self.external_mints),
            "reason": self.reason,
            "needs_normalization": self.needs_normalization,
            "is_problem": self.is_problem,
            "details": self.details,
        }


def classify_settlement(
    tx: Optional[dict],
    *,
    wallet: str,
    native_lamports_delta: int = 0,
    expected_output_mint: str = "",
    decimals_of: Optional[Any] = None,
) -> SettlementReading:
    """
    Read one confirmed exit transaction and state what it actually returned.

    ``tx`` is the parsed transaction. ``wallet`` is the configured signer's
    public address, used as the ONLY test for "ours". ``native_lamports_delta``
    is the wallet's own native SOL movement, passed in because the engine
    already computes it for other reasons.

    ORDER OF PRECEDENCE, and why: money the signer received natively is the
    cleanest evidence, so it is checked first; then a signer-owned token credit;
    then credits to other accounts; and only then "nothing". Reading them in the
    other order would report NO_PROCEEDS for a transaction that in fact paid out
    in native SOL, because native movement does not appear in token balances.
    """
    if tx is None:
        return SettlementReading(
            settlement=Settlement.UNREADABLE,
            reason="the confirmed transaction could not be read back, so the "
                   "output asset is unknown; refusing to assume it was SOL",
        )

    meta = tx.get("meta") or {}
    if meta.get("err"):
        return SettlementReading(
            settlement=Settlement.UNREADABLE,
            reason=f"the transaction reports an error ({meta.get('err')!r}); "
                   f"it is not a settlement",
        )

    keys = [str(k.get("pubkey") if isinstance(k, dict) else k)
            for k in (tx.get("transaction", {}).get("message", {})
                      .get("accountKeys") or [])]

    # --- token balance movements, correlated across pre/post arrays ---------
    post_by_index = {int(b.get("accountIndex", -1)): b
                     for b in (meta.get("postTokenBalances") or [])}
    deltas: list[tuple[str, str, int]] = []      # (mint, owner, delta)
    for pre in (meta.get("preTokenBalances") or []):
        idx = int(pre.get("accountIndex", -1))
        post = post_by_index.get(idx)
        if post is None:
            continue
        try:
            amount = int(post["uiTokenAmount"]["amount"]) - int(
                pre["uiTokenAmount"]["amount"])
        except (KeyError, TypeError, ValueError):
            continue
        owner = str(post.get("owner") or pre.get("owner") or "")
        mint = str(post.get("mint") or pre.get("mint") or "")
        deltas.append((mint, owner, amount))

    ours = [(m, a) for (m, o, a) in deltas if o == wallet and a > 0]
    theirs = sorted({m for (m, o, a) in deltas if o != wallet and a > 0})

    # --- 1. native SOL to the signer is the clean case ----------------------
    if native_lamports_delta > 0:
        # A token credit to someone else alongside it does not invalidate our
        # SOL, but it is reported so the operator knows it happened.
        return SettlementReading(
            settlement=Settlement.SETTLED_SOL,
            output_mint="SOL",
            output_amount=native_lamports_delta,
            controlled=True,
            external_mints=tuple(theirs),
            reason="native SOL credited to the configured signer",
            details={"native_lamports_delta": native_lamports_delta,
                     "token_deltas": deltas},
        )

    # --- 2. WSOL to the signer settles with SOL -----------------------------
    wsol = [a for (m, a) in ours if m == WSOL_MINT]
    if wsol:
        return SettlementReading(
            settlement=Settlement.SETTLED_SOL,
            output_mint=WSOL_MINT,
            output_amount=sum(wsol),
            controlled=True,
            external_mints=tuple(theirs),
            reason="wrapped SOL credited to the configured signer; it is "
                   "spendable as SOL after unwrap",
            details={"token_deltas": deltas},
        )

    # --- 3. a recognized, signer-controlled SPL came back -------------------
    if ours:
        mint, amount = ours[0]
        if mint in NORMALIZABLE_MINTS:
            return SettlementReading(
                settlement=Settlement.SETTLED_SPL_NEEDS_NORMALIZATION,
                output_mint=mint,
                output_amount=amount,
                controlled=True,
                external_mints=tuple(theirs),
                reason=(
                    f"the exit returned {amount} of {mint} to this wallet, NOT "
                    f"SOL. That asset is not spendable until it is converted, so "
                    f"the sale is NOT finished: a separate conversion order is "
                    f"required and must be reconciled on its own."),
                details={"token_deltas": deltas,
                         "normalizable": True},
            )
        return SettlementReading(
            settlement=Settlement.SETTLED_SPL_NEEDS_NORMALIZATION,
            output_mint=mint,
            output_amount=amount,
            controlled=True,
            external_mints=tuple(theirs),
            reason=(
                f"the exit returned {amount} of {mint} to this wallet. This "
                f"asset is not in the normalizable set, so it is reported and "
                f"left to the operator; the bot will not swap it "
                f"automatically."),
            details={"token_deltas": deltas, "normalizable": False},
        )

    # --- 4. money went somewhere we do not control --------------------------
    if theirs:
        return SettlementReading(
            settlement=Settlement.UNCONTROLLED,
            output_mint=theirs[0],
            output_amount=sum(a for (m, o, a) in deltas
                              if o != wallet and a > 0),
            controlled=False,
            external_mints=tuple(theirs),
            reason=(
                f"the exit credited {', '.join(theirs)} to an account that is "
                f"NOT the configured signer. This is not spendable money and is "
                f"never counted as proceeds; the operator must locate it."),
            details={"token_deltas": deltas},
        )

    # --- 5. nothing arrived --------------------------------------------------
    detail = ""
    if expected_output_mint:
        detail = (f" (the order expected {expected_output_mint} to be "
                  f"returned)")
    return SettlementReading(
        settlement=Settlement.NO_PROCEEDS,
        output_mint="",
        output_amount=0,
        controlled=False,
        external_mints=(),
        reason=("the signer was credited nothing by this transaction; it is a "
                "movement of tokens, not a sale" + detail),
        details={"token_deltas": deltas},
    )


def normalize_intent_for(
    reading: SettlementReading,
    *,
    source_mint: str,
    amount: int,
) -> Optional[dict[str, Any]]:
    """
    Describe the follow-up conversion order, or None when none is permitted.

    This RETURNS A DESCRIPTION and broadcasts nothing. The caller is expected to
    run it through the ordinary order pipeline -- intent reserve, quote, size
    gate, quote freshness, simulation, sign, broadcast, confirm, reconcile --
    exactly as for any other order. A conversion is a real trade with real
    slippage and a real fee, and treating it as bookkeeping would be how a bot
    loses money quietly.

    Refuses when the asset is not normalizable, so an unrecognised token can
    never be auto-swapped.
    """
    if not reading.needs_normalization:
        return None
    if source_mint not in NORMALIZABLE_MINTS:
        return None
    if amount <= 0:
        return None
    return {
        "kind": "SETTLEMENT_NORMALIZATION",
        "side": "ENTRY",                 # buy SOL with the token we received
        "input_mint": source_mint,
        "input_amount": amount,
        "output_mint": WSOL_MINT,
        "reason": ("the exit settled in a non-SOL asset; converting it back to "
                   "SOL is a separate order, not part of the sale"),
    }
