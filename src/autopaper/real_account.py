"""
REAL account accounting. Strictly separate from PAPER.

WHY A SEPARATE TYPE
-------------------
The previous runtime had exactly one account and it was simulated. Adding a
real one to the same class would mean the ledger, the dashboard and the
statistics could all read "cash" without knowing which account they were
reading, and the failure mode of that is a statistics page that mixes a
simulation's PnL with a wallet's.

So this is a different type, with different invariants:

* its cash is NOT tracked locally - the wallet IS the balance. Anything cached
  here is a last-observed snapshot with a timestamp and is never presented as
  current;
* its PnL is computed from ACTUAL on-chain amounts, never from a reference
  price;
* a position is only ever OPEN after a confirmed transaction;
* a position whose outcome could not be reconciled is AMBIGUOUS and holds no
  PnL at all, rather than being optimistically booked.

Nothing in :mod:`autopaper.account` imports this, and nothing here imports that.
The two accounts cannot be summed because neither can see the other.
"""

from __future__ import annotations

from .execution.price_layers import (
    executable_return_pct,
    mark_return_pct,
    realized_return_pct,
)
from .pricing import realized_execution_price

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .modes import TradeMode
from .safety import mask_address

#: A position's lifecycle. There is no UNFILLED, because a real order either
#: confirmed or it did not exist.
POSITION_OPEN = "OPEN"
POSITION_CLOSED = "CLOSED"
POSITION_AMBIGUOUS = "AMBIGUOUS"


@dataclass
class RealFillRecord:
    """
    The on-chain facts of one confirmed transaction.

    This is the record a real trade is written from. Every field is read back
    from the chain or the venue response. A field that could not be determined
    stays ``None``; it is never filled with the quote.
    """

    signature: str
    side: str = "ENTRY"
    commitment: str = ""
    slot: Optional[int] = None
    confirmed_at: Optional[float] = None

    #: What was asked for, in raw base units.
    requested_lamports: Optional[int] = None
    #: What actually went in / came out, in raw base units.
    actual_input_amount: Optional[int] = None
    actual_output_amount: Optional[int] = None
    #: WHAT ``actual_input_amount`` IS. ``"SWAP_LEG"`` (the amount that crossed
    #: into or out of the venue) or ``"WALLET_DEBIT"`` (the payer's whole SOL
    #: balance change, which additionally contains account-creation rent and the
    #: network fee).
    #:
    #: These are different quantities and mixing them is how the 2026-10-02
    #: $GOAT entry published ``actual_execution_price = 0.0004634453850018868``:
    #: a 3,497,651-lamport wallet debit -- 1,928,811 of swap principal plus
    #: 55,000 of fee plus 1,513,840 of token-account rent -- divided by
    #: 7,547,061,883 raw token units. That is not a price. The default is
    #: ``"SWAP_LEG"`` because that is what the field has always meant at a call
    #: site; the reconciler declares ``"WALLET_DEBIT"`` explicitly and
    #: additionally records ``swap_input_amount`` when it can derive one.
    input_amount_kind: str = "SWAP_LEG"
    #: The venue-side input leg, when it could be separated from the wallet
    #: balance delta. ``None`` means "not established" and no per-token price
    #: may be derived from the wallet delta instead.
    swap_input_amount: Optional[int] = None
    #: The payer's whole SOL balance change for this transaction.
    wallet_debit_lamports: Optional[int] = None
    #: Lamports of account-creation rent the transaction paid, i.e. the increase
    #: in accounts that previously held nothing. Itemised so the cost basis can
    #: state it rather than silently absorbing it.
    rent_lamports: Optional[int] = None
    #: The token mint's decimals, read from chain. ``None`` means no per-token
    #: price may be stated at all.
    token_decimals: Optional[int] = None
    #: The quote asset of this leg (wSOL / USDC / ...).
    quote_mint: Optional[str] = None
    #: Venue identity of the fill: provider, venue label, pool/curve address and
    #: the on-chain program that actually executed. Present so a market mark
    #: describing a different pool is detectable.
    provider: Optional[str] = None
    venue: Optional[str] = None
    pool: Optional[str] = None
    program_id: Optional[str] = None
    #: actual_input / actual_output, in quote units per output token.
    execution_price: Optional[float] = None
    #: The quote's price, kept alongside so slippage is measurable.
    quote_price: Optional[float] = None
    #: Price impact reported by the venue.
    price_impact_pct: Optional[float] = None
    #: actual fee, in lamports.
    fee_lamports: Optional[int] = None
    #: (actual / quoted) - 1. Positive means the fill was worse than quoted.
    actual_slippage: Optional[float] = None
    route_labels: list[str] = field(default_factory=list)

    @property
    def fee_sol(self) -> Optional[float]:
        if self.fee_lamports is None:
            return None
        return self.fee_lamports / 1_000_000_000

    @property
    def filled(self) -> bool:
        return self.commitment in ("confirmed", "finalized")

    def as_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "side": self.side,
            "commitment": self.commitment,
            "slot": self.slot,
            "confirmed_at": self.confirmed_at,
            "requested_lamports": self.requested_lamports,
            "actual_input_amount": self.actual_input_amount,
            "actual_output_amount": self.actual_output_amount,
            "input_amount_kind": self.input_amount_kind,
            "swap_input_amount": self.swap_input_amount,
            "wallet_debit_lamports": self.wallet_debit_lamports,
            "rent_lamports": self.rent_lamports,
            "token_decimals": self.token_decimals,
            "quote_mint": self.quote_mint,
            "provider": self.provider,
            "venue": self.venue,
            "pool": self.pool,
            "program_id": self.program_id,
            "execution_price": self.execution_price,
            "quote_price": self.quote_price,
            "price_impact_pct": self.price_impact_pct,
            "fee_lamports": self.fee_lamports,
            "fee_sol": self.fee_sol,
            "actual_slippage": self.actual_slippage,
            "route_labels": list(self.route_labels),
            "filled": self.filled,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RealFillRecord":
        return cls(
            signature=str(raw.get("signature") or ""),
            side=str(raw.get("side") or "ENTRY"),
            commitment=str(raw.get("commitment") or ""),
            slot=raw.get("slot"),
            confirmed_at=raw.get("confirmed_at"),
            requested_lamports=raw.get("requested_lamports"),
            actual_input_amount=raw.get("actual_input_amount"),
            actual_output_amount=raw.get("actual_output_amount"),
            input_amount_kind=raw.get("input_amount_kind") or "SWAP_LEG",
            swap_input_amount=raw.get("swap_input_amount"),
            wallet_debit_lamports=raw.get("wallet_debit_lamports"),
            rent_lamports=raw.get("rent_lamports"),
            token_decimals=raw.get("token_decimals"),
            quote_mint=raw.get("quote_mint"),
            provider=raw.get("provider"),
            venue=raw.get("venue"),
            pool=raw.get("pool"),
            program_id=raw.get("program_id"),
            execution_price=raw.get("execution_price"),
            quote_price=raw.get("quote_price"),
            price_impact_pct=raw.get("price_impact_pct"),
            fee_lamports=raw.get("fee_lamports"),
            actual_slippage=raw.get("actual_slippage"),
            route_labels=list(raw.get("route_labels") or []),
        )

    @classmethod
    def from_onchain_fill(cls, fill) -> "RealFillRecord":
        """Build from an :class:`autopaper.execution.real.OnchainFill`."""
        slippage = None
        if (
            fill.quote_price
            and fill.execution_price
            and fill.quote_price > 0
        ):
            slippage = fill.execution_price / fill.quote_price - 1.0
        return cls(
            signature=fill.signature,
            side=fill.side,
            commitment=fill.commitment,
            slot=fill.slot,
            confirmed_at=fill.confirmed_at,
            requested_lamports=fill.requested_lamports,
            actual_input_amount=fill.input_amount,
            actual_output_amount=fill.output_amount,
            input_amount_kind=getattr(fill, "input_amount_kind", "SWAP_LEG"),
            swap_input_amount=getattr(fill, "swap_input_amount", None),
            wallet_debit_lamports=getattr(fill, "wallet_debit_lamports", None),
            rent_lamports=getattr(fill, "rent_lamports", None),
            token_decimals=getattr(fill, "token_decimals", None),
            quote_mint=getattr(fill, "quote_mint", None),
            provider=getattr(fill, "provider", None),
            venue=getattr(fill, "venue", None),
            pool=getattr(fill, "pool", None),
            program_id=getattr(fill, "program_id", None),
            execution_price=fill.execution_price,
            quote_price=fill.quote_price,
            price_impact_pct=fill.price_impact_pct,
            fee_lamports=fill.fee_lamports,
            actual_slippage=slippage,
            route_labels=list(fill.route_labels),
        )


@dataclass
class RealPosition:
    """An open real position, backed by a confirmed on-chain fill."""

    mint: str
    symbol: str
    entry_signature: str
    entry: RealFillRecord
    token_amount: int
    #: The entry REFERENCE price. Kept because MFE/MAE are excursions of the
    #: observed mark series against it, and that definition is preserved on
    #: purpose. ``entry_price_kind`` says what it is.
    #:
    #: On 2026-10-02 this field held a DexScreener mark while every consumer
    #: read it as the price the position was opened at. That is the defect the
    #: three-layer split fixes, and it is why there is now a separate
    #: ``entry_execution_price_usd`` which is the authoritative fill price.
    entry_price_usd: float
    size_usd: float
    opened_at: float
    status: str = POSITION_OPEN
    exit: Optional[RealFillRecord] = None
    #: WHAT ``entry_price_usd`` IS. ``"OBSERVED_MARK"`` when it came from a
    #: third-party indexer, ``"ACTUAL_EXECUTION"`` when it was derived from
    #: settled amounts. Never left implicit.
    entry_price_kind: str = "OBSERVED_MARK"
    #: THE price this position was actually opened at, per WHOLE token, in USD,
    #: derived from the settled entry amounts (quote spent / tokens received)
    #: with the token's own decimals applied once. ``None`` when the settlement
    #: cannot support it -- a legitimate state, and NOT a zero.
    #:
    #: This is the only entry price that may appear in realized PnL.
    entry_execution_price_usd: Optional[float] = None
    #: The token mint's decimals, read from chain. ``None`` when unknown, in
    #: which case no per-token price may be stated at all.
    entry_token_decimals: Optional[int] = None
    #: The asset the entry was paid in (wSOL / USDC / ...).
    entry_quote_mint: Optional[str] = None
    #: Venue identity of the entry fill: provider, venue, pool. Present so a
    #: mark from a different pool is detectable rather than invisible.
    entry_provider: Optional[str] = None
    entry_venue: Optional[str] = None
    entry_pool: Optional[str] = None
    #: Raw settled amounts of the entry SWAP leg, distinct from the wallet
    #: balance delta. ``entry_swap_input_lamports`` is what went to the venue;
    #: ``entry_wallet_debit_lamports`` is what actually left the wallet, and the
    #: difference is rent and fees. Conflating the two is what produced a
    #: journalled "execution price" of 0.0004634 on 2026-10-02.
    entry_swap_input_lamports: Optional[int] = None
    entry_wallet_debit_lamports: Optional[int] = None
    #: Lamports of token-account rent the entry paid. Itemised, never netted.
    entry_rent_lamports: int = 0
    #: Lamports of network fee the entry paid (``meta.fee``).
    entry_fee_lamports: Optional[int] = None
    #: The full itemised cost record for the entry. Serialisable dict so it
    #: survives persistence without importing the pricing types here.
    entry_cost_components: dict = field(default_factory=dict)
    #: Why the cost basis is an estimate rather than a settled figure. ``None``
    #: when it is settled. A missing reason here means "we know and it is
    #: exact", which is the only way that value may be absent.
    entry_cost_basis_reason: Optional[str] = None
    #: Cost basis EXCLUDING rent that is still locked in an account, i.e. the
    #: trading-only figure. Retained beside the all-in basis so the difference
    #: is visible rather than implicit.
    entry_cost_usd_trading_only: float = 0.0
    #: MFE/MAE measured against the ACTUAL fill price rather than the entry
    #: mark. Both exist; they answer different questions and neither is a
    #: realized result.
    mfe_vs_fill_pct: float = 0.0
    mae_vs_fill_pct: float = 0.0
    #: Latest observed price. ``None`` means "we do not have one", and the
    #: position is shown as DATA_UNCERTAIN rather than valued at a guess.
    current_price_usd: Optional[float] = None
    last_observed_at: Optional[float] = None
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    realized_pnl_usd: Optional[float] = None
    #: Realized return as a PERCENTAGE of the all-in cost basis. Derived from
    #: settlement on both legs. This is the only percentage that may be shown
    #: to an operator as a result.
    realized_return_pct: Optional[float] = None
    #: The mark-to-mark return. An OBSERVATION about a third-party index, kept
    #: under this name so it can never again be printed as "ผลลัพธ์". This is the
    #: +71.56% arithmetic from 2026-10-02.
    observed_mark_return_pct: Optional[float] = None
    #: The quote-to-quote return, when both an entry and an exit executable
    #: quote were captured.
    executable_return_pct: Optional[float] = None
    exit_reason: str = ""
    exit_timestamp: Optional[float] = None
    #: The reference price at exit. Declared here because the old code assigned
    #: it dynamically and ``as_dict`` never reported it, so a closed position
    #: had an exit price nothing could read.
    exit_price_usd: Optional[float] = None
    #: WHAT KIND of price ``exit_price_usd`` is. It has historically been an
    #: OBSERVED_MARK from a third-party indexer, written into a field named as if
    #: it were the exit price -- which is how $BUTTHOLE published
    #: ``exit_price 2.043e-05`` for a trade that actually settled at
    #: ``4.99e-06``. The value is preserved (it is a real observation and MFE is
    #: built on it); its LABEL is corrected so no consumer can read it as an
    #: execution price without asking.
    exit_price_kind: str = "OBSERVED_MARK"
    #: The price actually realized per whole token, derived from the settled
    #: amounts. ``None`` when the settlement cannot support it -- which is a
    #: legitimate state, and is distinguished from a zero.
    exit_execution_price_usd: Optional[float] = None
    #: (mark - venue) / venue at exit, when a venue quote was available. This is
    #: the number that would have caught the divergence on the day.
    exit_mark_vs_venue_pct: Optional[float] = None
    #: Venue identity of the exit fill. Compared against the mark's pool, and
    #: against the entry's pool, so a cross-venue round trip is visible.
    exit_provider: Optional[str] = None
    exit_venue: Optional[str] = None
    exit_pool: Optional[str] = None
    #: The itemised costs of the exit leg (network fee, any venue fee, any rent
    #: that stayed locked). Serialisable dict.
    exit_cost_components: dict = field(default_factory=dict)
    #: Proceeds actually received, in USD. Retained separately from PnL so the
    #: two can be reconciled by eye.
    exit_proceeds_usd: Optional[float] = None
    #: True when the entry mark, the entry fill and/or the exit mark describe
    #: DIFFERENT pools. On 2026-10-02 a $GOAT entry and its exit filled on two
    #: different pools while the mark described a third address, and the system
    #: called them one market.
    cross_venue_detected: bool = False
    cross_venue_detail: Optional[str] = None
    #: The exit style the position was managed under, so the SAME exit policy
    #: that governs PAPER can govern REAL.
    exit_style: Any = None
    #: Realized cost basis in USD: the deployed notional plus the entry fee,
    #: both valued at the entry-time SOL price. PnL is measured against THIS,
    #: never against a SOL quantity.
    entry_cost_usd: float = 0.0
    #: Runner trailing state, mirroring the PAPER position so the same exit
    #: policy can be applied to a REAL position.
    runner_mode: bool = False
    runner_armed_at: float | None = None
    exit_hold_count: int = 0
    #: The SELECTABLE strategy profile this REAL position was opened under,
    #: stamped at entry and never re-read. Mirrors the PAPER field for the same
    #: reason: a REAL position must keep the exit policy it was opened under
    #: even if the operator changes the global selection while it is open.
    strategy_profile: str = "L25_CURRENT"
    #: Set when a real order was attempted and did not confirm.
    ambiguity_reason: str = ""

    @property
    def position_id(self) -> str:
        return f"REAL-{self.entry_signature[:8] or self.mint[:8]}"

    @property
    def entry_at(self) -> float:
        return self.opened_at

    @property
    def entry_timestamp(self) -> float:
        """Alias so the SAME :class:`~autopaper.exits.ExitPolicy` can assess a
        REAL position as a PAPER one. Consistency of concepts across the two
        accounts is a requirement, not a convenience."""
        return self.opened_at

    @property
    def peak_return_pct(self) -> float:
        return self.mfe_pct

    def return_at(self, price_usd: Optional[float]) -> Optional[float]:
        """Fractional return at ``price_usd`` (0.10 == +10%)."""
        if price_usd is None or price_usd <= 0 or self.entry_price_usd <= 0:
            return None
        return price_usd / self.entry_price_usd - 1.0

    @property
    def is_open(self) -> bool:
        return self.status == POSITION_OPEN

    @property
    def confirmed(self) -> bool:
        return bool(self.entry_signature) and self.entry.filled

    def mark(self, price_usd: Optional[float], now: float) -> None:
        """
        Update the mark and the excursion path.

        MFE/MAE are tracked on the same real price series the exit will be
        measured against, so the numbers shown for a real position mean the
        same thing they mean for a paper one.
        """
        self.last_observed_at = now
        if price_usd is None or price_usd <= 0 or self.entry_price_usd <= 0:
            return
        self.current_price_usd = price_usd
        ret = (price_usd / self.entry_price_usd - 1.0) * 100.0
        self.mfe_pct = max(self.mfe_pct, ret)
        self.mae_pct = min(self.mae_pct, ret)

    def unrealized_usd(self) -> Optional[float]:
        if self.current_price_usd is None or self.entry_price_usd <= 0:
            return None
        return (
            (self.current_price_usd / self.entry_price_usd - 1.0) * self.size_usd
        )

    def return_pct(self) -> Optional[float]:
        if self.current_price_usd is None or self.entry_price_usd <= 0:
            return None
        return (self.current_price_usd / self.entry_price_usd - 1.0) * 100.0

    def holding_seconds(self, now: Optional[float] = None) -> float:
        end = self.exit_timestamp or now or time.time()
        return max(0.0, end - self.opened_at)

    def to_state(self) -> dict[str, Any]:
        """Serialisable form, complete enough to RESTORE the position."""
        return {
            "mint": self.mint,
            "symbol": self.symbol,
            "entry_signature": self.entry_signature,
            "entry": self.entry.as_dict(),
            "exit": self.exit.as_dict() if self.exit else None,
            "token_amount": self.token_amount,
            "entry_price_usd": self.entry_price_usd,
            "size_usd": self.size_usd,
            "entry_cost_usd": self.entry_cost_usd,
            "opened_at": self.opened_at,
            "status": self.status,
            "current_price_usd": self.current_price_usd,
            "last_observed_at": self.last_observed_at,
            "mfe_pct": self.mfe_pct,
            "mae_pct": self.mae_pct,
            "realized_pnl_usd": self.realized_pnl_usd,
            "exit_reason": self.exit_reason,
            "exit_timestamp": self.exit_timestamp,
            "exit_price_usd": self.exit_price_usd,
            "exit_price_kind": self.exit_price_kind,
            "exit_execution_price_usd": self.exit_execution_price_usd,
            "exit_mark_vs_venue_pct": self.exit_mark_vs_venue_pct,
            "entry_price_kind": self.entry_price_kind,
            "entry_execution_price_usd": self.entry_execution_price_usd,
            "entry_token_decimals": self.entry_token_decimals,
            "entry_quote_mint": self.entry_quote_mint,
            "entry_provider": self.entry_provider,
            "entry_venue": self.entry_venue,
            "entry_pool": self.entry_pool,
            "entry_swap_input_lamports": self.entry_swap_input_lamports,
            "entry_wallet_debit_lamports": self.entry_wallet_debit_lamports,
            "entry_rent_lamports": self.entry_rent_lamports,
            "entry_fee_lamports": self.entry_fee_lamports,
            "entry_cost_components": dict(self.entry_cost_components),
            "entry_cost_basis_reason": self.entry_cost_basis_reason,
            "entry_cost_usd_trading_only": self.entry_cost_usd_trading_only,
            "mfe_vs_fill_pct": self.mfe_vs_fill_pct,
            "mae_vs_fill_pct": self.mae_vs_fill_pct,
            "realized_return_pct": self.realized_return_pct,
            "observed_mark_return_pct": self.observed_mark_return_pct,
            "executable_return_pct": self.executable_return_pct,
            "exit_provider": self.exit_provider,
            "exit_venue": self.exit_venue,
            "exit_pool": self.exit_pool,
            "exit_cost_components": dict(self.exit_cost_components),
            "exit_proceeds_usd": self.exit_proceeds_usd,
            "cross_venue_detected": self.cross_venue_detected,
            "cross_venue_detail": self.cross_venue_detail,
            "runner_mode": self.runner_mode,
            "runner_armed_at": self.runner_armed_at,
            "exit_hold_count": self.exit_hold_count,
            "ambiguity_reason": self.ambiguity_reason,
        }

    @classmethod
    def from_state(cls, raw: dict[str, Any]) -> "RealPosition":
        pos = cls(
            mint=str(raw.get("mint") or ""),
            symbol=str(raw.get("symbol") or "?"),
            entry_signature=str(raw.get("entry_signature") or ""),
            entry=RealFillRecord.from_dict(raw.get("entry") or {}),
            token_amount=int(raw.get("token_amount") or 0),
            entry_price_usd=float(raw.get("entry_price_usd") or 0.0),
            size_usd=float(raw.get("size_usd") or 0.0),
            opened_at=float(raw.get("opened_at") or 0.0),
            status=str(raw.get("status") or POSITION_OPEN),
        )
        if raw.get("exit"):
            pos.exit = RealFillRecord.from_dict(raw["exit"])
        pos.entry_cost_usd = float(raw.get("entry_cost_usd") or pos.size_usd)
        pos.current_price_usd = raw.get("current_price_usd")
        pos.last_observed_at = raw.get("last_observed_at")
        pos.mfe_pct = float(raw.get("mfe_pct") or 0.0)
        pos.mae_pct = float(raw.get("mae_pct") or 0.0)
        pos.realized_pnl_usd = raw.get("realized_pnl_usd")
        pos.exit_reason = str(raw.get("exit_reason") or "")
        pos.exit_timestamp = raw.get("exit_timestamp")
        pos.exit_price_usd = raw.get("exit_price_usd")
        pos.exit_price_kind = raw.get("exit_price_kind") or "OBSERVED_MARK"
        pos.exit_execution_price_usd = raw.get("exit_execution_price_usd")
        pos.exit_mark_vs_venue_pct = raw.get("exit_mark_vs_venue_pct")
        # -- three-layer price provenance and itemised costs ----------------
        # Absent keys on an older persisted row stay absent rather than
        # defaulting to a fabricated 0.0 or a fabricated mark.
        pos.entry_price_kind = raw.get("entry_price_kind") or "OBSERVED_MARK"
        pos.entry_execution_price_usd = raw.get("entry_execution_price_usd")
        dec = raw.get("entry_token_decimals")
        pos.entry_token_decimals = int(dec) if dec is not None else None
        pos.entry_quote_mint = raw.get("entry_quote_mint")
        pos.entry_provider = raw.get("entry_provider")
        pos.entry_venue = raw.get("entry_venue")
        pos.entry_pool = raw.get("entry_pool")
        pos.entry_swap_input_lamports = raw.get("entry_swap_input_lamports")
        pos.entry_wallet_debit_lamports = raw.get("entry_wallet_debit_lamports")
        pos.entry_rent_lamports = int(raw.get("entry_rent_lamports") or 0)
        pos.entry_fee_lamports = raw.get("entry_fee_lamports")
        pos.entry_cost_components = dict(raw.get("entry_cost_components") or {})
        pos.entry_cost_basis_reason = raw.get("entry_cost_basis_reason")
        pos.entry_cost_usd_trading_only = float(
            raw.get("entry_cost_usd_trading_only") or pos.entry_cost_usd)
        pos.mfe_vs_fill_pct = float(raw.get("mfe_vs_fill_pct") or 0.0)
        pos.mae_vs_fill_pct = float(raw.get("mae_vs_fill_pct") or 0.0)
        pos.realized_return_pct = raw.get("realized_return_pct")
        pos.observed_mark_return_pct = raw.get("observed_mark_return_pct")
        pos.executable_return_pct = raw.get("executable_return_pct")
        pos.exit_provider = raw.get("exit_provider")
        pos.exit_venue = raw.get("exit_venue")
        pos.exit_pool = raw.get("exit_pool")
        pos.exit_cost_components = dict(raw.get("exit_cost_components") or {})
        pos.exit_proceeds_usd = raw.get("exit_proceeds_usd")
        pos.cross_venue_detected = bool(raw.get("cross_venue_detected") or False)
        pos.cross_venue_detail = raw.get("cross_venue_detail")
        pos.runner_mode = bool(raw.get("runner_mode"))
        pos.runner_armed_at = raw.get("runner_armed_at")
        pos.exit_hold_count = int(raw.get("exit_hold_count") or 0)
        pos.ambiguity_reason = str(raw.get("ambiguity_reason") or "")
        return pos

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": TradeMode.REAL.value,
            "mint": self.mint,
            "symbol": self.symbol,
            "status": self.status,
            "confirmed": self.confirmed,
            "entry_signature": self.entry_signature,
            "entry_price_usd": self.entry_price_usd,
            "entry_price_kind": self.entry_price_kind,
            "entry_execution_price_usd": self.entry_execution_price_usd,
            "entry_venue": self.entry_venue,
            "entry_pool": self.entry_pool,
            "entry_cost_usd": self.entry_cost_usd,
            "entry_cost_usd_trading_only": self.entry_cost_usd_trading_only,
            "entry_rent_lamports": self.entry_rent_lamports,
            "entry_cost_components": dict(self.entry_cost_components),
            "current_price_usd": self.current_price_usd,
            "size_usd": self.size_usd,
            "token_amount": self.token_amount,
            "opened_at": self.opened_at,
            "unrealized_usd": self.unrealized_usd(),
            "return_pct": self.return_pct(),
            "realized_pnl_usd": self.realized_pnl_usd,
            "realized_return_pct": self.realized_return_pct,
            "observed_mark_return_pct": self.observed_mark_return_pct,
            "executable_return_pct": self.executable_return_pct,
            "mfe_pct": self.mfe_pct,
            "mae_pct": self.mae_pct,
            "mfe_vs_fill_pct": self.mfe_vs_fill_pct,
            "mae_vs_fill_pct": self.mae_vs_fill_pct,
            "exit_reason": self.exit_reason,
            "exit_timestamp": self.exit_timestamp,
            "exit_price_usd": self.exit_price_usd,
            "exit_price_kind": self.exit_price_kind,
            "exit_execution_price_usd": self.exit_execution_price_usd,
            "exit_mark_vs_venue_pct": self.exit_mark_vs_venue_pct,
            "exit_venue": self.exit_venue,
            "exit_pool": self.exit_pool,
            "exit_cost_components": dict(self.exit_cost_components),
            "exit_proceeds_usd": self.exit_proceeds_usd,
            "cross_venue_detected": self.cross_venue_detected,
            "cross_venue_detail": self.cross_venue_detail,
            "exit_style": (
                getattr(self.exit_style, "value", None)
                or (str(self.exit_style) if self.exit_style is not None else None)
            ),
            "runner_mode": self.runner_mode,
            "entry_fill": self.entry.as_dict(),
            "exit_fill": self.exit.as_dict() if self.exit else None,
            "ambiguity_reason": self.ambiguity_reason,
        }


@dataclass
class RealAccountSnapshot:
    """A last-observed view of the wallet. Never presented as live."""

    lamports: int
    sol: float
    masked_address: str
    observed_at: float
    sol_price_usd: Optional[float] = None

    @property
    def usd(self) -> Optional[float]:
        if self.sol_price_usd is None:
            return None
        return self.sol * self.sol_price_usd

    def as_dict(self) -> dict[str, Any]:
        return {
            "lamports": self.lamports,
            "sol": self.sol,
            "masked_address": self.masked_address,
            "observed_at": self.observed_at,
            "sol_price_usd": self.sol_price_usd,
            "usd": self.usd,
        }


class RealAccount:
    """
    The real account.

    Holds real positions and a cached wallet snapshot. It does NOT hold a cash
    balance, because the wallet is the balance; :attr:`wallet_snapshot` is
    explicitly a last-observed view with a timestamp, and
    :func:`equity_usd` refuses to report an equity figure derived from a
    snapshot older than its own staleness budget.
    """

    #: An equity figure older than this is not reported as current.
    SNAPSHOT_STALE_S = 30.0

    mode = TradeMode.REAL

    def __init__(self) -> None:
        self.positions: dict[str, RealPosition] = {}
        self.wallet_snapshot: Optional[RealAccountSnapshot] = None
        #: Realized PnL booked from CLOSED positions only. An ambiguous exit
        #: contributes nothing.
        self.realized_pnl_usd: float = 0.0
        self.fees_paid_sol: float = 0.0
        self.closed_count: int = 0

    # -- wallet ------------------------------------------------------------

    def observe_wallet(
        self,
        *,
        lamports: int,
        address: str,
        sol_price_usd: Optional[float] = None,
        now: Optional[float] = None,
    ) -> RealAccountSnapshot:
        snap = RealAccountSnapshot(
            lamports=int(lamports),
            sol=int(lamports) / 1_000_000_000,
            masked_address=mask_address(address),
            observed_at=now if now is not None else time.time(),
            sol_price_usd=sol_price_usd,
        )
        self.wallet_snapshot = snap
        return snap

    def wallet_is_fresh(self, now: Optional[float] = None) -> bool:
        if self.wallet_snapshot is None:
            return False
        t = now if now is not None else time.time()
        return (t - self.wallet_snapshot.observed_at) <= self.SNAPSHOT_STALE_S

    def equity_usd(self, now: Optional[float] = None) -> Optional[float]:
        """
        Real equity in USD, or ``None`` when it cannot be stated honestly.

        ``None`` when the wallet snapshot is missing, stale, or has no SOL
        price. The caller then refuses to size, which is the correct behaviour:
        a real order sized off a stale balance is not a real order, it is a
        guess with real money attached.
        """
        if not self.wallet_is_fresh(now):
            return None
        assert self.wallet_snapshot is not None
        usd = self.wallet_snapshot.usd
        if usd is None:
            return None
        return usd

    # -- positions ---------------------------------------------------------

    def open_positions(self) -> list[RealPosition]:
        return [p for p in self.positions.values() if p.is_open]

    def has_mint(self, mint: str) -> bool:
        return any(p.mint == mint and p.is_open for p in self.positions.values())

    def register_entry(
        self,
        *,
        mint: str,
        symbol: str,
        fill: RealFillRecord,
        token_amount: int,
        entry_price_usd: float,
        size_usd: float,
        now: Optional[float] = None,
        sol_price_usd: Optional[float] = None,
        exit_style: Any = None,
        strategy_profile: str = "L25_CURRENT",
        entry_price_kind: str = "OBSERVED_MARK",
        entry_mark: Optional[Any] = None,
        token_decimals: Optional[int] = None,
        quote_mint: Optional[str] = None,
        provider: Optional[str] = None,
        venue: Optional[str] = None,
        pool: Optional[str] = None,
        rent_lamports: Optional[int] = None,
        wallet_debit_lamports: Optional[int] = None,
        entry_quote: Optional[Any] = None,
    ) -> RealPosition:
        """
        Open a real position from a CONFIRMED fill.

        Raises unless the fill is confirmed and carries a signature. A real
        position with an unconfirmed fill is precisely the state this whole
        design exists to prevent.

        WHAT ``entry_price_usd`` MEANS HERE
        ----------------------------------
        It is the entry REFERENCE price and ``entry_price_kind`` says which kind.
        The caller is expected to pass an ``OBSERVED_MARK`` (that is what the
        market layer produces) plus the venue quote, and this method derives
        ``entry_execution_price_usd`` -- the price actually paid -- from the
        settled amounts. Realized PnL is measured against the settled figures,
        never against ``entry_price_usd``.

        THE COST BASIS IS ITEMISED, AND RENT IS NOT A SILENT LOSS
        -----------------------------------------------------------
        $NOTIFY (2026-10-02) and $GOAT (2026-10-02) both left 1,513,840 lamports
        of token-account rent in an account and had it silently omitted from the
        cost basis, which made a losing round trip book as a small profit. This
        method therefore records:

        * ``entry_cost_usd`` -- swap principal + network fee + rent that is
          LOCKED (paid and not provably returned). This is the honest basis.
        * ``entry_cost_usd_trading_only`` -- the same figure without the locked
          rent, retained so the difference is visible.
        * ``entry_cost_components`` -- every leg, itemised.

        Rent is only excluded from the all-in basis when the system can point at
        a proof it came back. Until then it is a cost, because the lamports are
        not in the trading wallet.
        """
        if not fill.signature:
            raise ValueError(
                "refusing to open a REAL position without a transaction "
                "signature"
            )
        if not fill.filled:
            # An unconfirmed transaction must not become a position. The
            # alternatives are worse: guessing that it filled, or retrying and
            # possibly double-buying.
            raise ValueError(
                f"refusing to open a REAL position from an unconfirmed "
                f"transaction (commitment={fill.commitment!r}, "
                f"signature={fill.signature[:12]!r}...)"
            )
        if self.has_mint(mint):
            raise ValueError(f"already holding a REAL position in {mint}")

        pos = RealPosition(
            mint=mint,
            symbol=symbol,
            entry_signature=fill.signature,
            entry=fill,
            token_amount=int(token_amount),
            entry_price_usd=float(entry_price_usd),
            size_usd=float(size_usd),
            opened_at=now if now is not None else time.time(),
            exit_style=exit_style,
            strategy_profile=strategy_profile,
            entry_price_kind=entry_price_kind,
            entry_token_decimals=(int(token_decimals)
                                  if token_decimals is not None else None),
            entry_quote_mint=quote_mint,
            entry_provider=provider,
            entry_venue=venue,
            entry_pool=pool,
        )

        # -- the ACTUAL entry price, from settlement ------------------------
        # The price comes from the SWAP leg only. A wallet balance delta also
        # contains account-creation rent and the network fee, and dividing that
        # by a token count is not a price -- it is how 2026-10-02 published an
        # "execution price" of 0.0004634. When the swap leg cannot be separated,
        # no per-token entry price is stated at all.
        swap_input = _entry_swap_leg(fill)
        pos.entry_swap_input_lamports = swap_input
        pos.entry_fee_lamports = fill.fee_lamports
        # Fall back to the FILL's own reading when the caller does not restate
        # it. The reconciler already measured both from the same transaction, and
        # a cost basis that silently defaults rent to zero when the caller
        # forgets a keyword is precisely how 1,513,840 lamports of it went
        # missing on 2026-10-02.
        pos.entry_rent_lamports = int(
            rent_lamports if rent_lamports is not None
            else (fill.rent_lamports or 0))
        pos.entry_wallet_debit_lamports = (
            int(wallet_debit_lamports) if wallet_debit_lamports is not None
            else fill.wallet_debit_lamports)

        if fill.fee_lamports is not None:
            self.fees_paid_sol += fill.fee_sol or 0.0
        if (pos.entry_token_decimals is not None
                and swap_input and pos.entry_token_decimals >= 0):
            pos.entry_execution_price_usd = realized_execution_price(
                side="BUY",
                input_amount=swap_input,
                output_amount=pos.token_amount,
                sol_price_usd=sol_price_usd,
                token_decimals=pos.entry_token_decimals,
            )

        # -- the cost basis, itemised ---------------------------------------
        px = sol_price_usd if (sol_price_usd and sol_price_usd > 0) else None
        entry_fee_usd = (pos.entry_fee_lamports or 0) / 1e9 * (px or 0.0)
        if fill.fee_lamports is not None and px is None:
            # The fee is a real lamport quantity we simply cannot value right
            # now. Recorded, and the reason recorded with it.
            pos.entry_cost_basis_reason = (
                "SOL/USD unavailable at entry, so the network fee could not be "
                "valued in USD; the cost basis is a lamport figure only"
            )
        principal_lamports = swap_input
        basis_lamports = 0
        basis_reason = pos.entry_cost_basis_reason
        if principal_lamports:
            basis_lamports += int(principal_lamports)
        basis_lamports += int(pos.entry_fee_lamports or 0)
        basis_lamports += int(pos.entry_rent_lamports or 0)
        trading_only_lamports = (
            int(principal_lamports or 0) + int(pos.entry_fee_lamports or 0))

        if px is not None:
            pos.entry_cost_usd = basis_lamports / 1e9 * px
            pos.entry_cost_usd_trading_only = trading_only_lamports / 1e9 * px
        else:
            # No SOL/USD: the basis cannot be stated in USD. This is the same
            # honesty the exit leg already has -- an unstated figure beats a
            # fabricated one -- and the reason is recorded so it is not silently
            # treated as settled.
            pos.entry_cost_usd = 0.0
            pos.entry_cost_usd_trading_only = 0.0
            basis_reason = (
                (basis_reason + "; ") if basis_reason else ""
            ) + "no SOL/USD at entry: the cost basis is not stated in USD"

        pos.entry_cost_components = {
            "swap_input_lamports": int(principal_lamports or 0),
            "network_fee_lamports": int(pos.entry_fee_lamports or 0),
            "account_rent_lamports": int(pos.entry_rent_lamports or 0),
            "rent_recovered_lamports": 0,
            "rent_locked_lamports": int(pos.entry_rent_lamports or 0),
            "wallet_debit_lamports": pos.entry_wallet_debit_lamports,
            "sol_price_usd": px,
            "basis_lamports": basis_lamports,
            "trading_only_lamports": trading_only_lamports,
        }
        pos.entry_cost_basis_reason = basis_reason

        # The mark's own identity, so a later comparison can say whether the
        # entry mark and the entry fill described the same pool.
        mark = entry_mark
        if mark is not None and getattr(mark, "pool", None):
            pos.cross_venue_detected = bool(
                pool and str(mark.pool) != str(pool))
            if pos.cross_venue_detected:
                pos.cross_venue_detail = (
                    f"entry mark described pool {mark.pool} but the fill used "
                    f"{pool}")

        self.positions[pos.mint] = pos
        return pos

    def mark(self, mint: str, price_usd: Optional[float], now: float) -> None:
        pos = self.positions.get(mint)
        if pos is not None and pos.is_open:
            pos.mark(price_usd, now)

    def close_position(
        self,
        *,
        mint: str,
        fill: RealFillRecord,
        exit_price_usd: float,
        reason: str,
        now: Optional[float] = None,
        sol_price_usd: Optional[float] = None,
        output_mint: str = "",
        output_decimals: int = 9,
        output_usd_price: Optional[float] = None,
        token_amount_sold: Optional[int] = None,
        token_decimals: Optional[int] = None,
        exit_mark: Optional[Any] = None,
        exit_quote: Optional[Any] = None,
        provider: Optional[str] = None,
        venue: Optional[str] = None,
        pool: Optional[str] = None,
        rent_lamports: Optional[int] = None,
    ) -> RealPosition:
        """
        Close a real position from a CONFIRMED fill.

        THE EXIT PRICE IS A SETTLEMENT FACT, NOT A MARK
        -------------------------------------------------
        ``exit_price_usd`` remains the OBSERVED MARK the exit decision was taken
        on -- it is preserved, and ``exit_price_kind`` labels it. It is NOT used
        for PnL. ``exit_execution_price_usd`` is derived from the settled
        amounts with ``token_amount_sold`` as the input leg; that value is what
        a reader must use to know what the position was worth when it was sold.

        On 2026-10-02 ``token_amount_sold`` was not a parameter at all, so the
        input leg was whatever the reconciler happened to have left on the fill
        (which is ``None`` for a native-SOL unwind). The exit execution price
        was therefore unmeasurable for exactly the trades it mattered most, and
        the mark/venue divergence helper silently returned ``None``. Passing the
        quantity the position actually held closes that hole at the source.

        PnL is computed from the actual exit fill in ONE currency:

            proceeds_usd = SOL received x SOL/USD
            pnl_usd      = proceeds_usd - cost basis (USD)

        The previous implementation subtracted a USD notional from a SOL
        quantity, so a $50 position that exited at break-even booked about
        -49.875. There is no code path here that can subtract SOL from USD.

        Nothing is booked when the exit cannot be reconciled, and a SOL/USD
        price is REQUIRED - without it the PnL cannot be stated in USD and the
        caller must keep the position open rather than invent a number.

        THE PROCEEDS MIGHT NOT BE SOL. A confirmed sell returns whatever the
        venue settles in, which may be USDC or another SPL asset. ``output_mint``
        says which, and the amount is then valued in THAT asset's own terms:

        * SOL or WSOL -> lamports x SOL/USD, as before;
        * another SPL with a readable USD price -> amount x that price;
        * another SPL with NO price -> the position is still CLOSED, because the
          tokens really were sold, but ``realized_pnl_usd`` is left ``None``
          rather than being computed from a unit price that was never observed.
          An unstated P&L is honest; a P&L derived from an assumed price is a
          fabrication that later reads as a measurement.

        A confirmed exit is NEVER blocked by this method. Refusing to book a sell
        the chain already settled would leave the position OPEN forever while its
        tokens are gone, which is a worse lie than an unstated P&L.
        """
        pos = self.positions.get(mint)
        if pos is None:
            raise KeyError(f"no real position in {mint}")
        if not fill.signature:
            raise ValueError(
                "refusing to close a REAL position without a transaction "
                "signature"
            )
        ts = now if now is not None else time.time()
        if not fill.filled:
            # The order is in an unknown state. The position stays OPEN and is
            # flagged, because a confirmed-on-chain-but-unreconciled sell is
            # still a sell in flight and reporting it as closed would be a lie.
            pos.ambiguity_reason = (
                fill.ambiguity_reason or "exit transaction not confirmed"
            )
            return pos

        is_sol_settlement = (not output_mint
                             or output_mint in (SOL_SETTLEMENT_MINT,
                                                WSOL_SETTLEMENT_MINT))

        if is_sol_settlement:
            if not sol_price_usd or sol_price_usd <= 0:
                raise ValueError(
                    "refusing to realize REAL PnL without a SOL/USD price: a "
                    "SOL proceeds amount cannot be converted to USD without one"
                )
        if exit_price_usd is None or exit_price_usd <= 0:
            raise ValueError(
                "refusing to close a REAL position without a positive exit price"
            )

        # ``actual_output_amount`` for an EXIT is net SOL received (the fee has
        # already been deducted from the wallet balance), in lamports.
        raw_out = float(fill.actual_output_amount or 0)
        pnl: Optional[float]
        if is_sol_settlement:
            proceeds_sol = raw_out / 1_000_000_000
            proceeds_usd = proceeds_sol * float(sol_price_usd)
            pnl = proceeds_usd - cost_basis(pos)
        elif output_usd_price is not None and float(output_usd_price) > 0:
            proceeds_usd = (raw_out / (10 ** max(0, int(output_decimals)))) \
                * float(output_usd_price)
            pnl = proceeds_usd - cost_basis(pos)
        else:
            # Non-SOL settlement we cannot price. The sale is real, so the
            # position closes; the P&L is simply not stated.
            pnl = None
            pos.ambiguity_reason = (
                f"the exit settled in {output_mint}, whose USD price was not "
                f"readable, so realized PnL is not stated. The sale is "
                f"confirmed; the proceeds are not spendable until converted."
            )

        if fill.fee_lamports is not None:
            self.fees_paid_sol += fill.fee_sol or 0.0

        # -- the ACTUAL exit price, from the settled amounts ----------------
        # The input leg is the quantity this position actually held. It comes
        # from the POSITION, not from the fill: on a native-SOL unwind the
        # reconciler sees no token-input delta (the chain records the token
        # movement as a balance change on an account, not as an SPL transfer
        # from the fee payer), so the fill's input leg is absent by construction.
        # Before this was a parameter the exit execution price was therefore
        # unmeasurable for exactly the trades that needed it.
        sold_raw = (int(token_amount_sold)
                    if token_amount_sold is not None
                    else fill.actual_input_amount)
        dec = (int(token_decimals) if token_decimals is not None
               else (pos.entry_token_decimals if not is_sol_settlement
                     else 6))
        pos.exit_execution_price_usd = realized_execution_price(
            side="EXIT",
            input_amount=sold_raw,
            output_amount=fill.actual_output_amount,
            sol_price_usd=(sol_price_usd
                           if (not is_sol_settlement or
                               (sol_price_usd or 0) > 0) else None),
            token_decimals=dec if dec is not None else 6,
        )
        pos.exit_provider = provider or fill.provider
        pos.exit_venue = venue or fill.venue
        pos.exit_pool = pool or fill.pool
        pos.exit_cost_components = {
            "swap_output_amount": int(fill.actual_output_amount or 0),
            "network_fee_lamports": int(fill.fee_lamports or 0),
            "account_rent_lamports": int(rent_lamports or 0),
            "settled_in": (output_mint or SOL_SETTLEMENT_MINT),
            "token_amount_sold": sold_raw,
            "sol_price_usd": sol_price_usd if (sol_price_usd or 0) > 0 else None,
        }
        pos.exit_proceeds_usd = (proceeds_usd if pnl is not None else None)

        pos.exit = fill
        pos.status = POSITION_CLOSED
        pos.exit_price_usd = float(exit_price_usd)
        # PROVENANCE, recorded at the moment of booking. ``exit_price_usd`` is
        # the OBSERVED MARK the exit decision was taken on -- it is preserved,
        # because MFE/MAE and the operator's review both need it -- but it is
        # now LABELLED, so it can no longer be read as the price the position was
        # sold at. The realized price is derived from the settled amounts and is
        # kept separately, and may legitimately be None.
        pos.exit_price_kind = "OBSERVED_MARK"
        pos.exit_timestamp = ts
        pos.exit_reason = reason
        pos.realized_pnl_usd = pnl
        # The PERCENTAGE an operator sees is derived from the settlement, on the
        # all-in cost basis. It is not a ratio of two marks: that arithmetic is
        # the +71.56% from 2026-10-02 and it lives, named, in
        # ``observed_mark_return_pct``.
        pos.realized_return_pct = realized_return_pct(
            realized_pnl_usd=pnl, cost_basis_usd=cost_basis(pos))
        pos.observed_mark_return_pct = mark_return_pct(
            exit_mark=pos.exit_price_usd,
            entry_mark=pos.entry_price_usd)
        pos.current_price_usd = float(exit_price_usd)

        # -- cross-venue detection ------------------------------------------
        # A mark that describes a different pool from the one that filled is
        # not wrong; it is about somewhere else. Recording that is the whole
        # point: on 2026-10-02 a $GOAT entry and exit filled on two pools while
        # the mark described a third, and the system called it one market.
        mark_pool = getattr(exit_mark, "pool", None)
        fill_pool = pos.exit_pool
        entry_pool = pos.entry_pool
        notes = []
        if pos.cross_venue_detail:
            notes.append(pos.cross_venue_detail)
        if (mark_pool and fill_pool and str(mark_pool) != str(fill_pool)):
            notes.append(f"exit mark described pool {mark_pool} but the exit "
                         f"filled on {fill_pool}")
        if (entry_pool and fill_pool and str(entry_pool) != str(fill_pool)
                and pos.entry_venue and pos.exit_venue
                and str(pos.entry_venue) != str(pos.exit_venue)):
            notes.append(f"entry filled on {entry_pool} ({pos.entry_venue}) "
                         f"and the exit filled on {fill_pool} "
                         f"({pos.exit_venue})")
        if notes:
            pos.cross_venue_detected = True
            pos.cross_venue_detail = "; ".join(notes)

        # -- the executable return, when both quotes were captured ----------
        entry_q = getattr(pos, "_entry_quote_obj", None)
        if entry_q is not None and exit_quote is not None:
            pos.executable_return_pct = executable_return_pct(
                entry_quote=entry_q, exit_quote=exit_quote)

        if pnl is not None:
            self.realized_pnl_usd += pnl
        self.closed_count += 1
        return pos

    def close_externally(self, mint: str, verification: Any, *,
                         proceeds_usd: Optional[float] = None,
                         asset_prices_usd: Optional[dict] = None,
                         now: Optional[float] = None) -> tuple["RealPosition",
                                                                dict]:
        """
        Close a position that left this wallet without the bot selling it.

        THE ACCOUNT AGGREGATES ARE UPDATED HERE, not by the caller. An earlier
        version applied the closure to the position object directly, which left
        ``realized_pnl_usd`` and ``closed_count`` untouched: the individual
        position reported +$0.0567 while the account total said $0.00. A ledger
        whose rows disagree with its total is worse than no ledger, because
        every report that reads the total is quietly wrong.

        Delegating to the one :func:`apply_external_close` implementation keeps
        the state mutation in a single place; this method only adds the
        bookkeeping that belongs to the account.
        """
        from .external_close import apply_external_close

        pos = self.positions.get(mint)
        if pos is None:
            raise KeyError(f"no real position in {mint}")
        out = apply_external_close(pos, verification,
                                   proceeds_usd=proceeds_usd,
                                   asset_prices_usd=asset_prices_usd,
                                   now=now)
        pnl = out.get("realized_pnl_usd")
        if pnl is not None:
            # only a STATED pnl enters the total; an unstated one is not zero
            self.realized_pnl_usd += float(pnl)
        self.closed_count += 1
        return pos, out

    # -- views -------------------------------------------------------------

    def open_exposure_usd(self) -> float:
        return sum(p.size_usd for p in self.open_positions())

    def unrealized_usd(self) -> float:
        return sum(v for v in (p.unrealized_usd() for p in self.open_positions()) if v)

    # -- persistence -------------------------------------------------------

    def to_state(self) -> dict[str, Any]:
        """
        The full REAL account, serialisable to disk.

        The wallet snapshot is deliberately NOT persisted: equity is a live
        quantity and a restored snapshot would present a stale balance as
        current. Only positions and their realized outcomes survive a restart.
        """
        return {
            "positions": {
                mint: p.to_state() for mint, p in self.positions.items()
            },
            "realized_pnl_usd": self.realized_pnl_usd,
            "fees_paid_sol": self.fees_paid_sol,
            "closed_count": self.closed_count,
        }

    @classmethod
    def from_state(cls, raw: dict[str, Any]) -> "RealAccount":
        acct = cls()
        for mint, prow in (raw.get("positions") or {}).items():
            try:
                acct.positions[str(mint)] = RealPosition.from_state(prow)
            except Exception:  # noqa: BLE001 - one bad row must not lose the rest
                continue
        acct.realized_pnl_usd = float(raw.get("realized_pnl_usd") or 0.0)
        acct.fees_paid_sol = float(raw.get("fees_paid_sol") or 0.0)
        acct.closed_count = int(raw.get("closed_count") or 0)
        return acct

    def as_dict(self) -> dict[str, Any]:
        snap = self.wallet_snapshot
        return {
            "mode": TradeMode.REAL.value,
            "wallet": snap.as_dict() if snap else None,
            "wallet_snapshot_fresh": self.wallet_is_fresh(),
            "open_positions": len(self.open_positions()),
            "open_exposure_usd": self.open_exposure_usd(),
            "unrealized_usd": self.unrealized_usd(),
            "realized_pnl_usd": self.realized_pnl_usd,
            "fees_paid_sol": self.fees_paid_sol,
            "closed_trades": self.closed_count,
            "equity_usd": self.equity_usd(),
        }


#: Settlement mints that count as "the proceeds are spendable as SOL". A WSOL
#: payout is SOL that has not been unwrapped yet; native SOL is represented by
#: the empty string so a caller that never looked still gets the old behaviour.
SOL_SETTLEMENT_MINT = "SOL"
WSOL_SETTLEMENT_MINT = "So11111111111111111111111111111111111111112"


def _entry_swap_leg(fill: RealFillRecord) -> Optional[int]:
    """
    The ENTRY input leg that actually crossed into the venue, in lamports.

    Prefers an explicitly separated ``swap_input_amount``. Falls back to
    ``actual_input_amount`` ONLY when the fill declares that field to be the
    swap leg. A fill that declares ``WALLET_DEBIT`` yields ``None``, and no
    per-token price is derived from a wallet balance change.
    """
    if fill.swap_input_amount is not None:
        return int(fill.swap_input_amount)
    if str(fill.input_amount_kind or "").upper() == "SWAP_LEG":
        return (int(fill.actual_input_amount)
                if fill.actual_input_amount is not None else None)
    return None


def cost_basis(pos: "RealPosition") -> float:
    """What a position cost, preferring the recorded entry cost over the size."""
    return float(pos.entry_cost_usd or pos.size_usd or 0.0)
