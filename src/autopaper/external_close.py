"""
EXTERNALLY_CLOSED reconciliation for a REAL position that left this wallet
without the bot selling it.

WHY THIS EXISTS
---------------
A REAL position can stop being held for a reason the bot did not choose: the
operator sold it in another wallet, a bundle moved it, or a third party acted.
When that happens the persisted position stays ``OPEN`` while the chain says
zero, and :meth:`autopaper.engine.PaperTrader.reconcile_real_state` correctly
records a mismatch and fails closed. That is the right behaviour for the
detection, but it is a dead end for the recovery: the mismatch can only be
cleared by an operator action that this repository did not have.

The $SI position of 2026-10-01 is the worked example. The bot's automatic exit
was refused, the operator then moved the tokens out of the configured wallet in
a separate transaction, and the runtime then reported
``recovery_consistent (unresolved reconciliation mismatch)`` on every REAL
action -- which, because the flag is process-wide, blocked REAL entries as well
as exits.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
It is an EXPLICIT, OPERATOR-INITIATED, FAIL-CLOSED state transition that records
what the chain already says. It is never automatic, and it never guesses:

* no sale price is invented. If the wallet received nothing, proceeds are 0;
* a transaction whose outcome cannot be confirmed is refused;
* a token delta that does not match the booked amount exactly is refused;
* a token movement not owned by the configured signer is refused;
* the audit trail is preserved, not overwritten -- the external signature is
  recorded on the position and in the returned report.

THE CHAIN IS THE AUTHORITY
--------------------------
Every fact used here is read from the chain at the moment of reconciliation,
not from local state and not from operator assertion. The operator supplies only
a signature to look up; the chain decides whether that signature is admissible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

#: The exit reason written onto a position closed this way. Distinct from every
#: bot-authored exit reason, so a reader can always tell an external closure
#: from one the strategy produced.
EXTERNAL_CLOSE_REASON = "EXTERNALLY_CLOSED"

#: Set when the output asset could not be attributed to this wallet.
UNCONTROLLED_SETTLEMENT = "EXTERNAL_UNCONTROLLED"

#: Mints that count as spendable-as-SOL for the purposes of "did we get paid".
_SOL_LIKE = frozenset({
    "SOL",
    "So11111111111111111111111111111111111111112",   # WSOL
})


def _decimals_of(meta: dict[str, Any], delta: TokenDelta) -> int:
    """The mint's own decimals, read from whichever balance row carries it."""
    for row in (meta.get("postTokenBalances") or meta.get("preTokenBalances")
                or []):
        if int(row.get("accountIndex", -1)) == delta.account_index:
            try:
                return int(row["uiTokenAmount"]["decimals"])
            except (KeyError, TypeError, ValueError):
                break
    return 0


class ExternalCloseRefused(Exception):
    """
    The requested reconciliation is not admissible.

    Raised rather than warned: every failure mode here means the chain does not
    say what the operator asked it to say, and recording the position as
    externally closed anyway would be writing a fact we cannot support.
    """


@dataclass(frozen=True)
class TokenDelta:
    """One token-account balance movement inside a confirmed transaction."""

    account_index: int
    mint: str
    owner: str
    pre: int
    post: int

    @property
    def delta(self) -> int:
        """Post minus pre. Negative means the account was debited."""
        return self.post - self.pre


@dataclass(frozen=True)
class WalletProceeds:
    """
    What the configured wallet actually received, measured from the chain.

    ``native_lamports_delta`` is the wallet's native SOL movement from the
    transaction, excluding any lamport movement on other accounts. Zero is a real
    measurement, not a default.
    """

    native_lamports_delta: int
    token_deltas: tuple[TokenDelta, ...] = ()
    debited_mints: tuple[str, ...] = ()

    @property
    def native_sol(self) -> float:
        return self.native_lamports_delta / 1_000_000_000


@dataclass(frozen=True)
class AssetCredit:
    """
    One asset credited to the configured signer by an external transaction.

    ADDED after the $SI reconciliation proved that native lamports are not the
    only way value can arrive. The $SI position was not transferred away for
    nothing: the counterparty sent 0.170950 USDC into this wallet in the same
    transaction. An accounting path that measured only native SOL concluded the
    position was a "TOTAL LOSS of its cost basis" -- when the wallet had in fact
    been paid in a different asset. A realized PnL computed from the assets that
    happen to be SOL is not a conservative estimate, it is a wrong number.
    """

    mint: str
    amount: int
    decimals: int = 0
    kind: str = "SOL"          # "SOL" | "SPL" | "NATIVE"

    @property
    def ui_amount(self) -> float:
        try:
            return self.amount / (10 ** int(self.decimals))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return 0.0


@dataclass(frozen=True)
class ExternalCloseVerification:
    """The outcome of checking one candidate external closure."""

    ok: bool
    mint: str
    signature: str
    reason: str = ""
    expected_amount: int = 0
    debited_amount: int = 0
    proceeds_sol: float = 0.0
    settlement: str = "CONTROLLED"
    destination_accounts: tuple[str, ...] = ()
    unknown_output_mints: tuple[str, ...] = ()
    #: EVERY asset the signer was credited, in any denomination.
    credited_assets: tuple[AssetCredit, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_multi_asset(self) -> bool:
        return len(self.credited_assets) > 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mint": self.mint,
            "signature": self.signature,
            "reason": self.reason,
            "expected_amount": self.expected_amount,
            "debited_amount": self.debited_amount,
            "proceeds_sol": self.proceeds_sol,
            "settlement": self.settlement,
            "destination_accounts": list(self.destination_accounts),
            "unknown_output_mints": list(self.unknown_output_mints),
            "credited_assets": [
                {"mint": a.mint, "amount": a.amount, "decimals": a.decimals,
                 "kind": a.kind, "ui_amount": a.ui_amount}
                for a in self.credited_assets
            ],
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# on-chain facts, injected so this module is testable without an RPC
# ---------------------------------------------------------------------------


class ChainFacts:
    """
    The chain reads reconciliation needs, behind an interface.

    Kept as an explicit protocol so the verification logic can be driven with
    fixtures in tests and can never accidentally reach for a network call in
    the middle of an assertion.
    """

    async def token_balance(self, owner: str, mint: str) -> Optional[int]:
        raise NotImplementedError

    async def get_transaction(self, signature: str) -> Optional[dict]:
        """Parsed transaction, or None when the chain does not have it."""
        raise NotImplementedError

    async def signature_status(self, signature: str) -> Optional[dict]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# parsing helpers
# ---------------------------------------------------------------------------


def token_deltas(meta: dict[str, Any], account_keys: list[str]) -> tuple[TokenDelta, ...]:
    """
    Extract every token-account balance movement from a parsed transaction.

    The chain reports this in three separate arrays that are correlated by
    ``accountIndex``; reading any one of them alone would miss either the debit
    or the credit.
    """
    post_by_index = {
        int(b.get("accountIndex", -1)): b
        for b in (meta.get("postTokenBalances") or [])
    }
    out: list[TokenDelta] = []
    for pre in (meta.get("preTokenBalances") or []):
        idx = int(pre.get("accountIndex", -1))
        post = post_by_index.get(idx)
        if post is None:
            continue
        try:
            pre_amt = int(pre["uiTokenAmount"]["amount"])
            post_amt = int(post["uiTokenAmount"]["amount"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append(TokenDelta(
            account_index=idx,
            mint=str(pre.get("mint") or ""),
            owner=str(post.get("owner") or pre.get("owner") or ""),
            pre=pre_amt,
            post=post_amt,
        ))
    return tuple(out)


def native_lamport_delta(meta: dict[str, Any], account_keys: list[str],
                         wallet: str) -> int:
    """
    The wallet's own native SOL movement inside the transaction.

    Measured only on the account index that IS the configured wallet, so SOL
    movements belonging to pools, vaults and other signers cannot be mistaken
    for proceeds. The transaction fee is charged to the fee payer, which may be
    a different account, so this does not silently absorb another party's fee.
    """
    try:
        idx = next(i for i, k in enumerate(account_keys) if k == wallet)
    except StopIteration:
        return 0
    pre = meta.get("preBalances") or []
    post = meta.get("postBalances") or []
    if idx >= len(pre) or idx >= len(post):
        return 0
    return int(post[idx]) - int(pre[idx])


# ---------------------------------------------------------------------------
# the verification itself
# ---------------------------------------------------------------------------


async def verify_external_close(
    chain: ChainFacts,
    *,
    wallet: str,
    mint: str,
    signature: str,
    expected_amount: int,
    require_zero_balance: bool = True,
) -> ExternalCloseVerification:
    """
    Decide whether ``signature`` proves this wallet's position in ``mint`` was
    closed externally. Fail closed on every doubt.

    The order matters and is deliberate: cheap local facts first, then the
    transaction, then ownership. Nothing is recorded until all of them agree.
    """
    # --- 1. the position must genuinely be gone from this wallet -----------
    if require_zero_balance:
        try:
            balance = await chain.token_balance(wallet, mint)
        except Exception as exc:                     # noqa: BLE001
            return ExternalCloseVerification(
                ok=False, mint=mint, signature=signature,
                reason=f"on-chain balance unreadable ({type(exc).__name__}); "
                       f"refusing to assume the position is gone",
            )
        if balance is None:
            return ExternalCloseVerification(
                ok=False, mint=mint, signature=signature,
                reason="on-chain balance unavailable; cannot verify the "
                       "position left the wallet",
            )
        if balance != 0:
            return ExternalCloseVerification(
                ok=False, mint=mint, signature=signature,
                reason=f"wallet still holds {balance} of {mint}; the position "
                       f"has NOT been closed externally",
                details={"balance": balance},
            )

    # --- 2. the signature must exist and be confirmed clean ----------------
    try:
        st = await chain.signature_status(signature)
    except Exception as exc:                         # noqa: BLE001
        return ExternalCloseVerification(
            ok=False, mint=mint, signature=signature,
            reason=f"signature lookup failed ({type(exc).__name__}); an "
                   f"unanswered lookup is not a pass",
        )
    if st is None:
        return ExternalCloseVerification(
            ok=False, mint=mint, signature=signature,
            reason="signature not found on chain",
        )
    if st.get("err"):
        return ExternalCloseVerification(
            ok=False, mint=mint, signature=signature,
            reason=f"the transaction FAILED on chain (err={st.get('err')!r})",
            details={"err": st.get("err")},
        )
    conf = st.get("confirmationStatus")
    if conf not in ("confirmed", "finalized"):
        return ExternalCloseVerification(
            ok=False, mint=mint, signature=signature,
            reason=f"transaction is not confirmed (status={conf!r})",
            details={"confirmationStatus": conf},
        )

    # --- 3. the transaction must debit EXACTLY the booked amount -------------
    tx = await chain.get_transaction(signature)
    if tx is None:
        return ExternalCloseVerification(
            ok=False, mint=mint, signature=signature,
            reason="transaction body unavailable; cannot verify the token delta",
        )
    meta = tx.get("meta") or {}
    keys = [str(k.get("pubkey") if isinstance(k, dict) else k)
            for k in (tx.get("transaction", {}).get("message", {})
                      .get("accountKeys") or [])]
    deltas = token_deltas(meta, keys)

    ours = [d for d in deltas
            if d.mint == mint and d.owner == wallet and d.delta < 0]
    debited = -sum(d.delta for d in ours)

    if not ours:
        return ExternalCloseVerification(
            ok=False, mint=mint, signature=signature,
            reason=f"the transaction does not debit {mint} from any token "
                   f"account owned by this wallet",
            details={"deltas": [d.__dict__ for d in deltas]},
        )
    if debited != int(expected_amount):
        return ExternalCloseVerification(
            ok=False, mint=mint, signature=signature,
            reason=f"token delta {debited} does not equal the booked position "
                   f"amount {expected_amount}; refusing to close a position "
                   f"this transaction did not fully settle",
            expected_amount=int(expected_amount), debited_amount=debited,
            details={"deltas": [d.__dict__ for d in ours]},
        )

    # --- 4. the destination must have received it ---------------------------
    destinations = tuple(
        d.account_index for d in deltas
        if d.mint == mint and d.delta > 0 and d.delta == debited
    )
    if not destinations:
        return ExternalCloseVerification(
            ok=False, mint=mint, signature=signature,
            reason=f"no token account credited {debited} of {mint}; the debit "
                   f"has no matching destination",
            expected_amount=int(expected_amount), debited_amount=debited,
        )

    # --- 5. proceeds are MEASURED IN EVERY ASSET, never assumed ------------
    # The $SI lesson: the counterparty paid 0.170950 USDC into this wallet in
    # the same transaction that removed the tokens. A native-lamports-only check
    # saw zero and concluded the position was a total loss. Value can arrive as
    # any asset, so every credit to the signer is collected and reported, and
    # NOTHING is converted to USD here -- pricing is the caller's job and an
    # assumed peg is not a measurement.
    lamports_delta = native_lamport_delta(meta, keys, wallet)
    credits: list[AssetCredit] = []
    if lamports_delta > 0:
        credits.append(AssetCredit(mint="SOL", amount=lamports_delta,
                                   decimals=9, kind="NATIVE"))
    for d in deltas:
        if d.delta > 0 and d.owner == wallet and d.mint != mint:
            dec = _decimals_of(meta, d)
            credits.append(AssetCredit(mint=d.mint, amount=d.delta,
                                       decimals=dec, kind="SPL"))

    non_sol = [c for c in credits if c.mint not in _SOL_LIKE]
    if not credits:
        settlement = "NO_PROCEEDS"
    elif non_sol and lamports_delta <= 0:
        settlement = "SETTLED_NON_SOL"
    elif non_sol:
        settlement = "SETTLED_MIXED"
    else:
        settlement = "CONTROLLED"

    return ExternalCloseVerification(
        ok=True,
        mint=mint,
        signature=signature,
        reason="external closure verified against the chain",
        expected_amount=int(expected_amount),
        debited_amount=debited,
        proceeds_sol=max(0, lamports_delta) / 1_000_000_000,
        settlement=settlement,
        destination_accounts=tuple(str(i) for i in destinations),
        unknown_output_mints=tuple(sorted({c.mint for c in non_sol})),
        credited_assets=tuple(credits),
        details={
            "token_deltas": [d.__dict__ for d in deltas],
            "native_lamports_delta": lamports_delta,
            "transaction_fee_lamports": (tx.get("meta") or {}).get("fee"),
        },
    )


# ---------------------------------------------------------------------------
# applying the verified closure
# ---------------------------------------------------------------------------


def apply_external_close(position, verification: ExternalCloseVerification,
                         *, proceeds_usd: Optional[float] = None,
                         asset_prices_usd: Optional[dict] = None,
                         now: Optional[float] = None) -> dict[str, Any]:
    """
    Record the verified external closure on an open REAL position.

    PROCEEDS ARE MEASURED IN EVERY ASSET. The first version valued only native
    lamports and therefore booked the $SI position as a "TOTAL LOSS of its cost
    basis" when the counterparty had in fact paid 0.170950 USDC into the same
    wallet. ``proceeds_usd`` is the caller's MEASURED value of everything the
    signer was credited; this function never assumes a peg and never assumes
    zero.

    AN UNVALUABLE CREDIT MAKES THE P&L UNSTATED, NOT ZERO AND NOT A LOSS. If
    ``proceeds_usd`` is None while credits exist, the position closes (the
    tokens are gone, which is a fact) and ``realized_pnl_usd`` is left None with
    the unpriceable asset named. The three available answers -- a loss, a gain,
    and "cannot be stated" -- are genuinely different, and the honest one is
    sometimes the third.

    The caller is responsible for having verified first; this function does not
    re-verify and does not perform I/O.
    """
    import time

    from .execution.price_layers import realized_return_pct
    from .real_account import (POSITION_CLOSED, RealFillRecord, cost_basis)

    if not verification.ok:
        raise ExternalCloseRefused(verification.reason)
    if position.status != "OPEN":
        raise ExternalCloseRefused(
            f"position is {position.status}, not OPEN; nothing to reconcile")

    ts = now if now is not None else time.time()
    cost = cost_basis(position)
    credits = verification.credited_assets
    prices = dict(asset_prices_usd or {})

    unvalued = tuple(c.mint for c in credits
                     if c.mint not in _SOL_LIKE and not prices.get(c.mint))
    pnl: Optional[float]
    if unvalued:
        # The position closed, but the P&L cannot be stated in USD.
        pnl = None
        value_note = (
            "credited assets with no readable USD price: "
            + ", ".join(unvalued)
            + ". realized PnL is NOT STATED.")
    elif credits or proceeds_usd is not None:
        pnl = float(proceeds_usd or 0.0) - cost
        value_note = ""
    else:
        # Genuinely nothing came back at all.
        pnl = -cost
        value_note = "no asset was credited to this wallet."

    proceeds_sol = float(verification.proceeds_sol or 0.0)

    # An audit record, NOT an invented fill: no execution price, no fabricated
    # output amount, and the side is recorded as EXTERNAL so it can never be
    # mistaken for a swap this bot broadcast.
    record = RealFillRecord(
        signature=verification.signature,
        side="EXTERNAL",
        commitment="finalized",
        slot=None,
        confirmed_at=ts,
        requested_lamports=None,
        actual_input_amount=verification.debited_amount,
        actual_output_amount=None,        # multi-asset: there is no single one
        execution_price=None,
        quote_price=None,
        price_impact_pct=None,
        fee_lamports=(verification.details.get("transaction_fee_lamports")
                      if verification.details.get("transaction_fee_lamports")
                      is not None else 0),
        actual_slippage=None,
        route_labels=tuple(f"CREDITED_{c.mint[:8]}" for c in credits) or
                     ("EXTERNAL_TRANSFER",),
    )

    position.exit = record
    position.status = POSITION_CLOSED
    position.exit_timestamp = ts
    # There is NO observed mark and NO swap venue here: nothing this bot traded.
    # ``exit_price_usd`` stays ``None`` -- an absent price, never a zero -- and
    # the kind is stated so no consumer can read a mark that does not exist.
    position.exit_price_usd = None
    position.exit_price_kind = "NOT_APPLICABLE_EXTERNAL_SETTLEMENT"
    position.exit_execution_price_usd = None
    position.exit_reason = EXTERNAL_CLOSE_REASON
    position.realized_pnl_usd = pnl
    # An external close has no swap leg to price, so there is no mark excursion
    # either. Both stay absent WITH a reason rather than being derived from a
    # mark that describes a pool the tokens never traded on.
    position.observed_mark_return_pct = None
    position.executable_return_pct = None
    position.exit_cost_components = {
        "settlement": verification.settlement,
        "proceeds_sol": proceeds_sol,
        "network_fee_lamports": record.fee_lamports,
        "note": "no swap leg: the tokens were credited by an external transfer",
    }
    position.exit_proceeds_usd = (None if pnl is None
                                  else float(proceeds_usd or 0.0))
    position.realized_return_pct = realized_return_pct(
        realized_pnl_usd=pnl, cost_basis_usd=cost)
    notes = []
    if value_note:
        notes.append(value_note)
    if credits:
        notes.append("credited: " + ", ".join(
            f"{c.ui_amount:,.6f} {c.mint[:8]}" for c in credits))
    else:
        notes.append("no asset was credited to this wallet")
    position.ambiguity_reason = "; ".join(n for n in notes if n)

    return {
        "mint": verification.mint,
        "signature": verification.signature,
        "reason": EXTERNAL_CLOSE_REASON,
        "proceeds_sol": proceeds_sol,
        "proceeds_usd": (None if pnl is None else float(proceeds_usd or 0.0)),
        "entry_cost_usd": cost,
        "realized_pnl_usd": pnl,
        "pnl_stated": pnl is not None,
        "settlement": verification.settlement,
        "unknown_output_mints": list(verification.unknown_output_mints),
        "credited_assets": [
            {"mint": c.mint, "amount": c.amount, "ui_amount": c.ui_amount,
             "kind": c.kind, "price_usd": prices.get(c.mint)}
            for c in credits
        ],
    }

# ---------------------------------------------------------------------------
# RPC-backed ChainFacts
# ---------------------------------------------------------------------------


class RpcChainFacts(ChainFacts):
    """
    :class:`ChainFacts` over the bot's existing RPC backend.

    READ ONLY. Every method here is a balance read or a transaction fetch; none
    builds, signs or broadcasts anything. It reuses the configured RPC client
    rather than opening a second one, so there is one node endpoint and one
    rate budget.
    """

    def __init__(self, rpc, wallet: str) -> None:
        self._rpc = rpc
        self._wallet = wallet

    async def token_balance(self, owner: str, mint: str) -> Optional[int]:
        return await self._rpc.get_token_account_balance(owner, mint)

    async def get_transaction(self, signature: str) -> Optional[dict]:
        return await self._rpc.get_transaction(signature)

    async def signature_status(self, signature: str) -> Optional[dict]:
        statuses = await self._rpc.get_signature_statuses(signature)
        return statuses[0] if statuses else None
