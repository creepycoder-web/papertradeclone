"""
WHAT ONE EXECUTABLE QUOTE ACTUALLY COST, DECOMPOSED FROM INTEGERS.

WHY THIS EXISTS
===============
The entry notification used to print a single line::

    Gap: +1.26%

and a reader could reasonably have taken it for "the mark was wrong by
1.26%", or for "the fee was 1.25% and the rest is dust". It is neither. That
number is the execution-native gap -- the executable average price divided by
THIS MARKET's own on-chain marginal price, from the same slot -- which is
``fee + size impact``, measured against a reference that is deliberately not
the observed mark. The mark premium is a different quantity entirely, and on
the population this system trades it is an order of magnitude larger.

Everything here exists so that a log line can be re-run by hand:

* every price is derived from raw integer legs, never from a rounded float;
* ``fee + impact`` composes into ``total cost`` MULTIPLICATIVELY, exactly,
  because ``(1 + fee) * (1 + impact) = effective / marginal`` and both sides
  of that identity are ratios of the same integers;
* the observed mark is carried alongside, so "expensive to execute" and
  "the mark lagged the curve" can never be conflated again.

NOTHING HERE DECIDES ANYTHING
=============================
No guard reads this module. It is arithmetic over a quote that has already
been produced, and it is read-only: it cannot produce, refresh or mutate a
quote. If a required leg is missing the field is ``None`` -- never ``0``,
because a zero would render as a finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .schema import QuotePoint

#: One percent of a relative quantity expressed in basis points.
BPS = Decimal(10000)
#: One whole, for percent conversion.
HUNDRED = Decimal(100)

DASH = "-"


# ---------------------------------------------------------------------------
# small exact helpers
# ---------------------------------------------------------------------------

def _scaled(raw: int | None, decimals: int | None) -> Decimal | None:
    """A raw integer in whole units. ``None`` propagates; it never becomes 0."""
    if raw is None or decimals is None:
        return None
    return Decimal(raw).scaleb(-decimals)


def _ratio_bps(part: Decimal | None, whole: Decimal | None) -> Decimal | None:
    """``(part / whole - 1)`` in basis points. Undefined inputs give ``None``."""
    if part is None or whole is None or whole <= 0 or part <= 0:
        return None
    return (part / whole - Decimal(1)) * BPS


def _relative_pct(part: Decimal | None, whole: Decimal | None) -> Decimal | None:
    """``(part / whole - 1)`` in percent. Undefined inputs give ``None``."""
    if part is None or whole is None or whole <= 0 or part <= 0:
        return None
    return (part / whole - Decimal(1)) * HUNDRED


def _usd_decimal(value: float | Decimal | None) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


# ---------------------------------------------------------------------------
# the breakdown
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuoteCostBreakdown:
    """
    Fee, size impact and mark premium for ONE executable quote.

    Units
    -----
    Every ``*_quote`` quantity is in the QUOTE ASSET (for this system, SOL)
    unless the field name says ``_usd`` or ``_tokens``. Every ``*_raw``
    quantity is the verbatim integer leg, so the arithmetic can be repeated
    from a log line alone.

    The three bps fields
    --------------------
    * ``fee_cost_bps`` -- what the venue took, relative to the PRE-FEE
      quote-asset leg: ``(fee_amount / amount_after_fee) * 1e4``. Same sign
      on both sides (always positive in normal operation): on a SELL the fee
      REDUCES proceeds, and that direction lives in
      ``output < amount_after_fee``, not in this rate.
    * ``impact_bps`` -- PURE size impact: what moving the book for this order
      costs, at constant fees: ``(pre_fee_price / marginal_before - 1) * 1e4``.
      This is the quantity Jupiter means by ``priceImpactPct``; the direct
      Pump ``DirectQuoteResult.price_impact_pct`` does NOT mean it.
    * ``total_cost_bps`` -- the execution-native gap, in bps:
      ``(effective_price / marginal_before - 1) * 1e4``, where
      ``effective_price`` is the executable price AFTER the venue's fee
      mechanics -- it equals ``quote.effective_price`` on both sides.

    They compose exactly, with the fee's direction taken by side::

        BUY : (1 + fee/1e4) * (1 + impact/1e4) = 1 + total/1e4
        SELL: (1 - fee/1e4) * (1 + impact/1e4) = 1 + total/1e4

    because every term is a ratio of integers from the same quote. A test
    asserts both identities; this docstring does not establish them.
    """

    side: str
    provider: str
    identity: str = "UNKNOWN"
    quote_decimals: int | None = None
    token_decimals: int | None = None

    # -- the raw integer legs, verbatim -----------------------------------
    #: The two quote-asset legs, ordered so that
    #: ``quote_gross_raw - quote_net_raw`` is always the fee:
    #:   BUY : gross = SOL SENT (the fee is taken out of it),
    #:         net = ``amount_after_fee`` (what the curve priced against)
    #:   SELL: gross = ``amount_after_fee`` (pre-fee proceeds),
    #:         net = SOL RECEIVED
    quote_gross_raw: int | None = None
    quote_net_raw: int | None = None
    #: The token amount: tokens received on a BUY, tokens sent on a SELL.
    token_amount_raw: int | None = None
    amount_after_fee_raw: int | None = None
    #: numerator / denominator of the same-slot native reference.
    quote_reserves_raw: int | None = None
    token_reserves_raw: int | None = None

    # -- money ---------------------------------------------------------------
    fee_bps: int | None = None               # the venue's own schedule, in bps
    fee_amount_quote: Decimal | None = None  # quote asset actually extracted
    input_amount: Decimal | None = None
    output_amount: Decimal | None = None
    #: marginal (spot) price of this market BEFORE the order, same slot
    marginal_before_quote: Decimal | None = None
    #: marginal price AFTER this order has been applied to those reserves
    marginal_after_quote: Decimal | None = None
    #: average price for this size, before the fee
    pre_fee_price: Decimal | None = None
    #: average price for this size, after the fee == quote.effective_price
    effective_price: Decimal | None = None

    fee_cost_bps: Decimal | None = None
    impact_bps: Decimal | None = None
    total_cost_bps: Decimal | None = None
    #: the provider's own ``price_impact_pct`` field, whatever it means there
    provider_impact_pct: float | None = None

    observed_mark_usd: float | None = None
    effective_price_usd: float | None = None
    #: ``(effective_price_usd / observed_mark - 1) * 100``. THE quantity a
    #: reader first assumes ``Gap:`` was showing, and the one it never was.
    mark_premium_pct: Decimal | None = None

    quote_age_ms: float | None = None
    venue: str | None = None
    pool_or_curve: str | None = None

    def as_dict(self) -> dict[str, Any]:
        def f(value: Decimal | None) -> float | None:
            return float(value) if value is not None else None

        return {
            "side": self.side,
            "provider": self.provider,
            "identity": self.identity,
            "quote_decimals": self.quote_decimals,
            "token_decimals": self.token_decimals,
            "quote_gross_raw": self.quote_gross_raw,
            "quote_net_raw": self.quote_net_raw,
            "token_amount_raw": self.token_amount_raw,
            "amount_after_fee_raw": self.amount_after_fee_raw,
            "quote_reserves_raw": self.quote_reserves_raw,
            "token_reserves_raw": self.token_reserves_raw,
            "fee_bps": self.fee_bps,
            "fee_amount_quote": f(self.fee_amount_quote),
            "input_amount": f(self.input_amount),
            "output_amount": f(self.output_amount),
            "marginal_before_quote": f(self.marginal_before_quote),
            "marginal_after_quote": f(self.marginal_after_quote),
            "pre_fee_price": f(self.pre_fee_price),
            "effective_price": f(self.effective_price),
            "fee_cost_bps": f(self.fee_cost_bps),
            "impact_bps": f(self.impact_bps),
            "total_cost_bps": f(self.total_cost_bps),
            "provider_impact_pct": self.provider_impact_pct,
            "observed_mark_usd": self.observed_mark_usd,
            "effective_price_usd": self.effective_price_usd,
            "mark_premium_pct": f(self.mark_premium_pct),
            "quote_age_ms": self.quote_age_ms,
            "venue": self.venue,
            "pool_or_curve": self.pool_or_curve,
        }


def breakdown_of(
    quote: QuotePoint,
    *,
    observed_mark_usd: float | None = None,
) -> QuoteCostBreakdown:
    """
    Decompose one executable quote into fee, size impact and mark premium.

    READ-ONLY. Uses only integers already on the quote plus the same-slot
    native reference the guard compared against. Where a leg is missing --
    Jupiter does not disclose an after-fee leg, an unavailable quote has no
    legs at all -- the derived field is ``None``.
    """
    side = (quote.side or "").upper()
    buy = side == "BUY"
    quote_dec = quote.input_decimals if buy else quote.output_decimals
    token_dec = quote.output_decimals if buy else quote.input_decimals

    native = quote.execution_native_reference
    quote_reserves_raw = native.numerator_raw if native else None
    token_reserves_raw = native.denominator_raw if native else None

    # The quote-asset leg charged, and that same leg BEFORE the fee.
    #   BUY : gross = input (SOL sent)                   net = amount_after_fee
    #   SELL: gross = amount_after_fee (SOL out before fee)      net = output
    if buy:
        quote_gross_raw = quote.input_amount_raw
        quote_net_raw = quote.amount_after_fee_raw
        token_amount_raw = quote.output_amount_raw
    else:
        quote_gross_raw = quote.amount_after_fee_raw
        quote_net_raw = quote.output_amount_raw
        token_amount_raw = quote.input_amount_raw

    fee_amount_quote: Decimal | None = None
    if quote_gross_raw is not None and quote_net_raw is not None:
        # NOT clipped at zero: a negative "fee" would mean the venue paid the
        # trader to trade, and that is a finding, not a formatting problem.
        fee_amount_quote = _scaled(quote_gross_raw - quote_net_raw, quote_dec)

    input_amount = _scaled(quote.input_amount_raw, quote.input_decimals)
    output_amount = _scaled(quote.output_amount_raw, quote.output_decimals)

    marginal_before = _marginal_price(native, quote_dec, token_dec)
    marginal_after = _marginal_after(
        quote_reserves_raw, token_reserves_raw,
        quote_net_raw, token_amount_raw,
        quote_dec=quote_dec, token_dec=token_dec, buy=buy,
    )

    gross = _scaled(quote_gross_raw, quote_dec)
    net = _scaled(quote_net_raw, quote_dec)
    tokens = _scaled(token_amount_raw, token_dec)

    # WHICH LEG IS "EXECUTABLE" AND WHICH IS "PRE-FEE" FLIPS BY SIDE, and
    # getting it backwards stays silent: the resulting numbers still look
    # plausible, they are just the wrong quantity.
    #   BUY : you SEND ``input`` and the fee is already inside it, so the
    #         executable price is ``input / tokens``.
    #   SELL: the curve models ``amount_after_fee`` BEFORE the fee and you
    #         RECEIVE ``output``, so the executable price is
    #         ``output / tokens``.
    # On both sides that is exactly ``quote.effective_price`` -- the field
    # contract of QuotePoint. The PRE-FEE leg is ``amount_after_fee_raw`` on
    # both sides: what the curve saw on a BUY, what the curve produced before
    # the fee on a SELL. ``fee_amount_quote`` above stays ``gross - net``:
    # input - amount_after_fee on a BUY, amount_after_fee - output on a SELL.
    effective_leg = gross if buy else net
    pre_fee_leg = net if buy else gross

    pre_fee_price: Decimal | None = None
    if pre_fee_leg is not None and tokens is not None and tokens > 0:
        pre_fee_price = pre_fee_leg / tokens

    effective_price: Decimal | None = None
    if effective_leg is not None and tokens is not None and tokens > 0:
        effective_price = effective_leg / tokens

    # What the venue took, relative to the PRE-FEE quote-asset leg. Always
    # non-negative in normal operation -- on a SELL the fee REDUCES proceeds,
    # so the sign lives in ``output < amount_after_fee``, not in this rate.
    fee_cost_bps: Decimal | None = None
    if (
        fee_amount_quote is not None
        and pre_fee_leg is not None and pre_fee_leg > 0
    ):
        fee_cost_bps = fee_amount_quote / pre_fee_leg * BPS

    return QuoteCostBreakdown(
        side=side or "?",
        provider=quote.provider or "?",
        identity=quote.identity,
        quote_decimals=quote_dec,
        token_decimals=token_dec,
        quote_gross_raw=quote_gross_raw,
        quote_net_raw=quote_net_raw,
        token_amount_raw=token_amount_raw,
        amount_after_fee_raw=quote.amount_after_fee_raw,
        quote_reserves_raw=quote_reserves_raw,
        token_reserves_raw=token_reserves_raw,
        fee_bps=quote.fee_bps,
        fee_amount_quote=fee_amount_quote,
        input_amount=input_amount,
        output_amount=output_amount,
        marginal_before_quote=marginal_before,
        marginal_after_quote=marginal_after,
        pre_fee_price=pre_fee_price,
        effective_price=effective_price,
        fee_cost_bps=fee_cost_bps,
        impact_bps=_ratio_bps(pre_fee_price, marginal_before),
        total_cost_bps=_ratio_bps(effective_price, marginal_before),
        provider_impact_pct=quote.price_impact_pct,
        observed_mark_usd=observed_mark_usd,
        effective_price_usd=quote.effective_price_usd,
        mark_premium_pct=_relative_pct(
            _usd_decimal(quote.effective_price_usd),
            _usd_decimal(observed_mark_usd),
        ),
        quote_age_ms=quote.quote_age_ms,
        venue=quote.venue,
        pool_or_curve=quote.pool_or_curve,
    )


def _marginal_price(
    native: Any,
    quote_dec: int | None,
    token_dec: int | None,
) -> Decimal | None:
    """The same-slot marginal price, from the reserve integers where possible."""
    if native is None:
        return None
    numerator = getattr(native, "numerator_raw", None)
    denominator = getattr(native, "denominator_raw", None)
    if (
        numerator is not None and denominator not in (None, 0)
        and quote_dec is not None and token_dec is not None
    ):
        return Decimal(numerator).scaleb(token_dec - quote_dec) / Decimal(denominator)
    price = getattr(native, "reference_price", None)
    if price:
        return Decimal(str(price))
    return None


def _marginal_after(
    quote_reserves_raw: int | None,
    token_reserves_raw: int | None,
    quote_leg_raw: int | None,
    token_leg_raw: int | None,
    *,
    quote_dec: int | None,
    token_dec: int | None,
    buy: bool,
) -> Decimal | None:
    """
    The same market's marginal price AFTER this order.

    Constant product: a BUY adds the post-fee quote to the quote reserve and
    removes the tokens bought; a SELL does the inverse. Nothing in the
    accounting uses it -- it is the number that makes the size impact
    checkable by hand against the reserves printed in the log.
    """
    if (
        quote_reserves_raw is None or token_reserves_raw is None
        or quote_leg_raw is None or token_leg_raw is None
        or quote_dec is None or token_dec is None
    ):
        return None
    quote_reserves = Decimal(quote_reserves_raw)
    token_reserves = Decimal(token_reserves_raw)
    quote_leg = Decimal(quote_leg_raw)
    token_leg = Decimal(token_leg_raw)

    if buy:
        new_quote = quote_reserves + quote_leg
        new_token = token_reserves - token_leg
    else:
        new_quote = quote_reserves - quote_leg
        new_token = token_reserves + token_leg
    if new_token <= 0 or new_quote <= 0:
        return None
    return new_quote.scaleb(token_dec - quote_dec) / new_token


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _fmt_price(value: Decimal | None) -> str:
    """A per-token price that never silently rounds to zero."""
    if value is None or value <= 0:
        return DASH
    if value >= Decimal("0.000001"):
        text = f"${value:.10f}".rstrip("0").rstrip(".")
        return DASH if text == "$0" else text
    return f"${value:.6e}"


def _fmt_sol_price(value: Decimal | None) -> str:
    if value is None or value <= 0:
        return DASH
    if value < Decimal("0.000001"):
        return f"{value:.8e}"
    return f"{value:.9f}".rstrip("0").rstrip(".")


def _fmt_units(value: Decimal | None, decimals: int | None) -> str:
    """An amount at EXACTLY its own decimals, so it can be re-integer'd."""
    if value is None:
        return DASH
    return f"{value:.{max(0, decimals or 0)}f}"


def _fmt_bps(value: Decimal | None, *, signed: bool) -> str:
    if value is None:
        return DASH
    if signed:
        return f"{value:+.2f} bps"
    integral = value == value.to_integral_value()
    return f"{int(value)} bps" if integral else f"{value:.2f} bps"


def _fmt_pct(value: Decimal | None) -> str:
    return DASH if value is None else f"{value:+.2f}%"


def cost_block(
    breakdown: QuoteCostBreakdown,
    *,
    sol_usd: float | None = None,
) -> list[str]:
    """
    The audit lines for one quote: reference, legs, fee, impact, premium.

    Plain text with no markup, so the Telegram bridge and the console CLI
    render the SAME numbers. A line whose input is missing shows ``-`` rather
    than being dropped, so an absent field is visible as absent.

    The observed mark is NOT reprinted here: both callers already show it,
    directly above this block, which is what makes the last line -- the
    premium of the executable price OVER that mark -- readable in context.
    """
    quote_side = "SOL" if breakdown.quote_decimals == 9 else "quote"
    buy = breakdown.side == "BUY"

    marginal_sol = breakdown.marginal_before_quote
    marginal_text = _fmt_sol_price(marginal_sol)
    if marginal_sol is not None and sol_usd and sol_usd > 0:
        marginal_text = (
            f"{_fmt_price(marginal_sol * Decimal(str(sol_usd)))}"
            f" ({marginal_text} SOL)"
        )
    elif marginal_sol is not None:
        marginal_text = f"{marginal_text} SOL"

    def raw(value: int | None) -> int | str:
        return value if value is not None else DASH

    lines = [
        f"Curve mid/marginal: {marginal_text}",
        f"Reserves: quote {raw(breakdown.quote_reserves_raw)}"
        f" / token {raw(breakdown.token_reserves_raw)} (raw)",
    ]
    if buy:
        lines += [
            f"SOL input: {_fmt_units(breakdown.input_amount, breakdown.quote_decimals)}"
            f" {quote_side} ({raw(breakdown.quote_gross_raw)} raw)",
            f"Tokens out: {_fmt_units(breakdown.output_amount, breakdown.token_decimals)}"
            f" ({raw(breakdown.token_amount_raw)} raw)",
        ]
    else:
        lines += [
            f"Tokens in: {_fmt_units(breakdown.input_amount, breakdown.token_decimals)}"
            f" ({raw(breakdown.token_amount_raw)} raw)",
            # The raw beside the SOL-out figure is the RECEIVED leg
            # (quote_net_raw), not the pre-fee proceeds: a raw integer that
            # does not scale to the decimal printed next to it is worse than
            # no raw integer at all.
            f"SOL out: {_fmt_units(breakdown.output_amount, breakdown.quote_decimals)}"
            f" {quote_side} ({raw(breakdown.quote_net_raw)} raw)",
        ]

    scheduled = (
        Decimal(breakdown.fee_bps) if breakdown.fee_bps is not None else None
    )
    # The SCHEDULED rate and the rate actually extracted can differ by the
    # integer ceiling on a small order; both are shown rather than reconciled.
    lines += [
        f"Trading fee: {_fmt_bps(scheduled, signed=False)}"
        f" (extracted {_fmt_bps(breakdown.fee_cost_bps, signed=False)})",
        f"Fee amount: {_fmt_units(breakdown.fee_amount_quote, breakdown.quote_decimals)}"
        f" {quote_side}",
        f"Price impact: {_fmt_bps(breakdown.impact_bps, signed=True)}",
        f"Total quote cost: {_fmt_bps(breakdown.total_cost_bps, signed=True)}",
        f"Mark → executable premium: {_fmt_pct(breakdown.mark_premium_pct)}",
    ]
    return lines


__all__ = ["QuoteCostBreakdown", "breakdown_of", "cost_block"]
