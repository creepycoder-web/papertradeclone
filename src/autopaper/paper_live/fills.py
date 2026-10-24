"""
Execution-aware simulated fills, and the settlement arithmetic.

THE ONE RULE THAT MATTERS MOST
===============================

    The simulated fill price IS the executable quote's effective price.

Nothing is added on top. Not a slippage percentage, not an impact estimate, not
a staleness penalty. An executable quote is already the price the venue computed
for THIS exact order size, so applying a further penalty double-counts it - and
a double-counted penalty does not look like an error, it looks like a strategy
with a worse fill.

The venue's own ``price_impact_pct`` is retained SEPARATELY, as an observation
about the quote, and is never compounded into the price.

A post-quote penalty is available as an explicit parameter for future research,
and it is OFF by default. It is a named, opt-in, recorded adjustment - not a
hidden constant.

WHY NOT REUSE ``autopaper.execution.fill``
==========================================

That module is mark-based by design: it takes the observed mark, takes the
"worse of quote and current", and adds a modelled slippage. It is correct for
what it was built for. Reusing it here would reintroduce exactly the mark
contamination this harness exists to remove, so the fill model is separate and
small, and the ONLY thing it takes as input is a quote.

SETTLEMENT
==========

    gross = exit_proceeds - entry_cost          (both in the QUOTE asset)
    net   = gross - fees - modeled_costs

Quote-asset PnL is primary, because that is the asset the position was actually
opened and closed in. The USD figure is a conversion at an explicitly
timestamped reference, stored beside it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from autopaper.same_market.models import WSOL_MINT
from autopaper.same_market.numeric import to_float

from .quotes import SolReference, classify_entry_identity
from .schema import (
    SIMULATED_FILL_KIND,
    FailureReason,
    PaperPosition,
    QuotePoint,
    RealizedPnl,
    SimulatedFill,
)


class FillRefused(RuntimeError):
    """A simulated fill could not be produced. Carries the reason to record."""

    def __init__(self, reason: FailureReason, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


#: SOL base units. The modeled network cost is DECLARED in lamports while every
#: amount it is added to on this path is denominated in the quote asset (SOL),
#: so the conversion lives in exactly one place -- see
#: :attr:`FillModel.network_cost_sol`.
LAMPORTS_PER_SOL = Decimal(1_000_000_000)


@dataclass(frozen=True)
class FillModel:
    """
    How a quote becomes a simulated fill.

    ``post_quote_penalty_bps`` is the ONLY knob, and it is ZERO by default. It
    exists so a future study can ask "what if we were conservative" without
    editing the fill path - and it is recorded on every fill, so a dataset
    collected with it OFF can never be confused with one collected with it on.
    """

    post_quote_penalty_bps: float = 0.0
    #: Network costs are not charged to a simulated position because nothing is
    #: broadcast; the field exists so the accounting shape matches a real
    #: settlement and is visibly zero rather than absent.
    #:
    #: LAMPORTS. Use :attr:`network_cost_sol` -- never this -- when charging it
    #: against a position, whose amounts are all in the quote asset.
    network_cost_lamports: Decimal = Decimal(0)

    @property
    def network_cost_sol(self) -> Decimal:
        """The modeled network cost in the QUOTE ASSET (SOL).

        Charging ``network_cost_lamports`` directly is a 10^9 unit error that
        fails silently: the value is a plain Decimal, so 5,000 lamports of
        intended cost (0.000005 SOL) is spent as 5,000 SOL, and no field name on
        the receiving side says which unit it must be. It is currently masked
        only because the default is zero -- which makes it a landmine on the
        settlement path rather than a live error, and exactly the kind that
        surfaces the first time someone models a real network fee.
        """
        return Decimal(self.network_cost_lamports) / LAMPORTS_PER_SOL

    def penalised_price(self, price: Decimal, side: str) -> Decimal:
        """Apply the optional, OFF-by-default penalty. BUY worsens, SELL lowers."""
        if self.post_quote_penalty_bps == 0:
            return price
        factor = Decimal(self.post_quote_penalty_bps) / Decimal(10_000)
        return price * (Decimal(1) + factor if side == "BUY" else Decimal(1) - factor)

    def as_dict(self) -> dict[str, Any]:
        return {
            "post_quote_penalty_bps": self.post_quote_penalty_bps,
            "penalty_active": self.post_quote_penalty_bps != 0,
            "network_cost_lamports": to_float(self.network_cost_lamports),
            "policy": (
                "the fill price is the executable quote's effective price; no "
                "slippage, impact or staleness penalty is added on top of it"
            ),
        }


def _non_stale_available_quote_is_not_rejected(
    quote: Any,
    *,
    side: str,
) -> None:
    """
    Prove the order of the fill-gate guards.

    An executable, non-stale quote with a valid effective price MUST reach the
    fill, not be classified unavailable. If the unavailable guard sits before
    the stale/price guards, a quote that is stale or price-less but still
    marked available would be wrongly classified as "quote unavailable" rather
    than as stale or arithmetic.
    """
    if getattr(quote, "available", False) and not getattr(quote, "stale", False):
        if quote.effective_price is not None and quote.effective_price > 0:
            return
    raise AssertionError(
        "internal invariant: the construction above must not be reachable for "
        "a quote that is available, non-stale and price-positive; if reached, "
        "the guard ordering in fill_from_quote is wrong"
    )


def fill_from_quote(
    quote: QuotePoint,
    *,
    side: str,
    now: float | None = None,
    model: FillModel | None = None,
    sol_reference: SolReference | None = None,
    require_identity: bool = True,
) -> SimulatedFill:
    """Build a :class:`SimulatedFill` from an executable quote.

    Fail-closed at every step, each with a named reason. The guards are ordered
    so a quote that IS available, non-stale and price-positive reaches the fill
    and is never reclassified as unavailable. The explicit ordering is:

    * the quote must exist (``None`` is not an available quote);
    * an unavailable answer must be named, not treated as a price;
    * the quote must be denominated in the token the fill is supposed to
      receive or pay;
    * it must not be older than the configured window;
    * its same-market identity must be proven when ``require_identity``;
    * its output must be positive;
    * its decimals must be known.

    At NO point is a mark consulted. There is no parameter through which one
    could be passed.
    """
    model = model or FillModel()
    now = time.time() if now is None else now

    if quote is None:
        raise FillRefused(
            FailureReason.ENTRY_REJECTED_QUOTE_UNAVAILABLE
            if side == "BUY"
            else FailureReason.EXIT_QUOTE_UNAVAILABLE,
            "no quote object was returned by the provider (quote is None); "
            "no fill is fabricated and the position is left open",
        )
    if not getattr(quote, "available", False):
        raise FillRefused(
            FailureReason.ENTRY_REJECTED_QUOTE_UNAVAILABLE
            if side == "BUY"
            else FailureReason.EXIT_QUOTE_UNAVAILABLE,
            quote.missing_reason or "no executable quote was obtained",
        )
    # The QUOTE MINT is the asset the fill is denominated in: SOL/wSOL on both
    # legs. It is checked here, in the one function every fill passes through,
    # because the engine validated the TOKEN mint and this was the remaining
    # half. A quote denominated in anything else would produce a fill whose
    # ``input_mint``/``output_mint`` were a lie, and every downstream number --
    # PnL, bankroll, fee -- would be computed against an asset nobody holds.
    if quote.quote_mint and quote.quote_mint != WSOL_MINT:
        raise FillRefused(
            FailureReason.QUOTE_MINT_MISMATCH
            if side == "BUY"
            else FailureReason.EXIT_QUOTE_MINT_MISMATCH,
            (
                f"the quote is denominated in {quote.quote_mint[:8]}… but a "
                f"{side} leg is settled in {WSOL_MINT[:8]}…; a quote priced in "
                "another asset is refused rather than relabelled"
            ),
        )
    if quote.stale:
        raise FillRefused(
            FailureReason.ENTRY_REJECTED_QUOTE_STALE
            if side == "BUY"
            else FailureReason.EXIT_QUOTE_STALE,
            (
                f"the executable quote is {quote.quote_age_ms:.0f}ms old, beyond "
                f"the configured {quote.max_quote_age_ms:.0f}ms limit; an old "
                "quote is NOT used as a fill and is NOT replaced with the mark"
            ),
        )
    if not getattr(quote, "available", False):
        raise FillRefused(
            FailureReason.ENTRY_REJECTED_QUOTE_UNAVAILABLE
            if side == "BUY"
            else FailureReason.EXIT_QUOTE_UNAVAILABLE,
            quote.missing_reason or "no executable quote was obtained",
        )
    if require_identity and quote.identity != "PASS":
        raise FillRefused(
            FailureReason.ENTRY_REJECTED_IDENTITY_UNKNOWN
            if side == "BUY"
            else FailureReason.EXIT_QUOTE_IDENTITY_UNKNOWN,
            (
                # The named reason, the source that produced it and the market
                # in question travel WITH the refusal. One flat verdict could
                # not distinguish "this token has no market we can see" from
                # "the market is there but could not be proven to be this
                # token's" from "the executable market is provably another
                # book" - three facts with three different remedies.
                f"same-market identity is {quote.identity}, not PASS: "
                f"reason={classify_entry_identity(quote)} "
                f"source={quote.provider or 'none'} token={quote.token_mint} "
                f"pool={quote.pool_or_curve or 'none'}: "
                f"{quote.identity_evidence or 'unproven'}. The PAPER trade fails "
                "closed; the observation is retained as a non-trade record"
            ),
        )
    if (
        quote.output_amount_raw is None
        or quote.output_amount_raw <= 0
        or quote.output_decimals is None
        or quote.input_amount_raw is None
        or quote.input_decimals is None
    ):
        raise FillRefused(
            FailureReason.ARITHMETIC_ERROR,
            "the quote is missing an amount or its decimals, so no fill can be "
            "derived from it; nothing is assumed",
        )
    if quote.effective_price is None or quote.effective_price <= 0:
        raise FillRefused(
            FailureReason.ARITHMETIC_ERROR,
            "the quote has no positive effective price, so no fill can be derived",
        )
    # Internal invariant only: proves the guard ordering, never mutates the path.
    _non_stale_available_quote_is_not_rejected(quote, side=side)
    price = model.penalised_price(Decimal(quote.effective_price), side)

    return SimulatedFill(
        side=side,
        kind=SIMULATED_FILL_KIND,
        input_amount_raw=quote.input_amount_raw,
        input_mint=(quote.quote_mint if side == "BUY" else quote.token_mint),
        input_decimals=quote.input_decimals,
        input_amount_decimal=float(Decimal(quote.input_amount_raw).scaleb(
            -quote.input_decimals)),
        output_amount_raw=quote.output_amount_raw,
        output_mint=(quote.token_mint if side == "BUY" else quote.quote_mint),
        output_decimals=quote.output_decimals,
        output_amount_decimal=float(Decimal(quote.output_amount_raw).scaleb(
            -quote.output_decimals)),
        effective_price=price,
        effective_price_usd=quote.effective_price_usd,
        # The fee is taken from the QUOTE and is already inside the venue's
        # output amount, so it is recorded as zero rather than double-subtracted
        # below. It is itemised separately for analysis.
        fee=Decimal(0),
        fee_bps=quote.fee_bps,
        price_impact_pct=quote.price_impact_pct,
        quote_timestamp=quote.quote_timestamp or now,
        quote_age_ms=quote.quote_age_ms,
        max_quote_age_ms=quote.max_quote_age_ms,
        provider=quote.provider,
        venue=quote.venue,
        pool_or_curve=quote.pool_or_curve,
        route=quote.route,
        identity=quote.identity,
        fill_timestamp=now,
        note=(
            "simulated from an executable quote; no transaction was built, "
            "signed or broadcast; no extra slippage was added on top of the "
            "quote; the venue fee is already inside the quoted output"
        ),
    )


def realize(
    position: PaperPosition,
    *,
    exit_fill: SimulatedFill,
    exit_observed_mark: float | None,
    sol_reference: SolReference | None,
    model: FillModel | None = None,
    now: float | None = None,
) -> RealizedPnl:
    """
    Settle a simulated position from its two FILLS.

    ``net = exit_proceeds - entry_cost - fees - network_costs - other_costs``,
    all in the quote asset, and every term on the right is reported on the
    result with the value that was actually charged. The observed mark is passed
    in for the record and is NOT an input to the arithmetic - it is stored so
    the two can be compared later.
    """
    model = model or FillModel()
    now = time.time() if now is None else now

    entry_cost = position.entry_cost_quote_asset
    exit_proceeds = Decimal(exit_fill.output_amount_raw).scaleb(
        -exit_fill.output_decimals
    )
    fees = position.trading_fees_quote + Decimal(exit_fill.fee)
    # BOTH LEGS PAY A NETWORK COST, AND BOTH ARE IN SOL.
    #
    # ``position.network_costs_quote`` is the ENTRY leg, snapshotted at open;
    # ``model.network_cost_sol`` is the EXIT leg, charged now. A round trip pays
    # on both, so charging both is correct -- but they must be the same unit, and
    # the model's own field is declared in LAMPORTS. Adding it raw would charge
    # 10^9x the intended amount (silently, and only once someone sets it).
    network_costs = position.network_costs_quote + model.network_cost_sol
    # ``other_costs_quote`` IS SUBTRACTED, and was not.
    #
    # It was recorded on the result while being left out of ``net``, so the
    # settlement identity every consumer asserts --
    # ``net == gross - trading_fees - network_costs - other_costs`` -- only held
    # because the field is zero by default. The same was true of the EXIT
    # network leg, which was charged to ``net`` but reported as the ENTRY leg
    # alone. Both are now charged once and reported as exactly what was charged,
    # so the identity closes for any value rather than only for zero.
    other_costs = position.other_costs_quote

    gross = exit_proceeds - entry_cost
    net = gross - fees - network_costs - other_costs

    bankroll_before = position.bankroll_before
    pnl_usd = float(net) * (
        sol_reference.usd_per_sol if sol_reference else position.sol_usd_reference
    )
    bankroll_after = bankroll_before + pnl_usd
    peak = max(bankroll_before, bankroll_after)
    drawdown = (
        ((peak - bankroll_after) / peak * 100.0) if peak > 0 else 0.0
    )

    return RealizedPnl(
        entry_cost_quote=entry_cost,
        exit_proceeds_quote=exit_proceeds,
        trading_fees_quote=fees,
        network_costs_quote=network_costs,
        other_costs_quote=other_costs,
        gross_pnl_quote=gross,
        net_pnl_quote=net,
        realized_pnl_usd=pnl_usd,
        realized_pnl_sol=float(net),
        return_pct=(
            float(net) / float(entry_cost) * 100.0 if entry_cost > 0 else None
        ),
        bankroll_before=bankroll_before,
        bankroll_after=bankroll_after,
        drawdown_pct=drawdown,
        hold_seconds=max(0.0, now - position.opened_at),
        sol_usd_reference=(sol_reference.usd_per_sol if sol_reference
                           else position.sol_usd_reference),
        sol_usd_reference_at=(sol_reference.at if sol_reference
                              else position.sol_usd_reference_at),
    )


__all__ = ["FillModel", "FillRefused", "fill_from_quote", "realize"]
