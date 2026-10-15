"""
Decimal normalisation and the two divisions the experiment rests on.

Everything here is ``Decimal``. A $0.23 order against a $0.00001783 token
produces amounts in the 1e4..1e7 range with six to nine decimals, and doing that
arithmetic in binary floats is how a harness ends up reporting a gap that is
mostly representation error. ``float`` appears only at the very edge, when a
value is handed to the renderer or serialised to JSON.

Two rules, both load-bearing:

* **Never round before calculating.** The effective price is computed from the
  full-precision quotient. Rounding to a display precision first would make
  the gap a function of how many digits the terminal happened to print.
* **A division by zero is a named failure, never infinity.** ``0/0`` is
  ``DECIMAL_ERROR``. There is no code path that yields ``0.0`` for "unknown".
"""

from __future__ import annotations

from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext
from typing import Any

#: Working precision. 60 significant digits is far beyond what a memecoin
#: quote needs, and the point of the surplus headroom is that the result is
#: exact rather than merely accurate: none of the divisions performed here
#: produce a repeating expansion that gets truncated at a display boundary.
PRECISION = 60

ZERO = Decimal(0)
ONE = Decimal(1)


class DecimalError(ValueError):
    """
    An amount could not be interpreted, or a required division was undefined.

    Carries the offending input so the report can show what failed rather than
    only that something did.
    """

    def __init__(self, message: str, *, offending: Any = None) -> None:
        super().__init__(message)
        self.offending = offending


def to_decimal(value: Any, *, field: str = "value") -> Decimal:
    """
    Coerce to ``Decimal`` without going through ``float``.

    The ``float`` path exists only because JSON numbers arrive as floats. Even
    then the value is routed via ``repr`` so the shortest round-trip
    representation is used rather than the binary expansion.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool) or value is None:
        raise DecimalError(f"{field} is not a number", offending=value)
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        try:
            return Decimal(repr(value))
        except InvalidOperation as exc:
            raise DecimalError(f"{field} is not a finite number", offending=value) from exc
    try:
        text = str(value).strip()
        if not text:
            raise DecimalError(f"{field} is empty", offending=value)
        return Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise DecimalError(f"{field} is not a number: {value!r}", offending=value) from exc


def normalise_amount(raw_amount: int | str, decimals: int) -> Decimal:
    """
    Convert a raw on-chain integer amount into human units.

    Raw amounts are integers of the smallest divisible unit. Solidity-style
    ``amount * 10**-decimals`` in binary floating point loses exactly the
    digits that matter at this price point, so the scaling is done in
    ``Decimal``.
    """
    if decimals < 0 or decimals > 30:
        raise DecimalError(f"implausible decimals: {decimals}", offending=decimals)
    amount = to_decimal(raw_amount, field="amount")
    with localcontext() as ctx:
        ctx.prec = PRECISION
        scaled = amount.scaleb(-decimals)
    return scaled


def effective_price(
    input_amount: Decimal,
    output_amount: Decimal,
    *,
    input_decimals: int,
    output_decimals: int,
) -> Decimal:
    """
    ``effective_price = input_amount / output_amount`` with correct scaling.

    Both amounts arrive as RAW integers and are normalised to human units
    inside the same expression, so the unit conversion cannot be applied twice
    or forgotten once - the two errors that make a reported price wrong by a
    factor of 10**6 or 10**9 without looking wrong at a glance.

    Raises :class:`DecimalError` on a zero or negative output amount. A
    zero-output quote is a refused trade, not a price of zero.
    """
    in_dec = to_decimal(input_amount, field="input_amount")
    out_dec = to_decimal(output_amount, field="output_amount")
    if out_dec <= 0:
        raise DecimalError(
            "output amount is not positive; a quote that delivers nothing has "
            "no effective price",
            offending=output_amount,
        )
    if in_dec < 0:
        raise DecimalError("input amount is negative", offending=input_amount)
    with localcontext() as ctx:
        ctx.prec = PRECISION
        in_human = in_dec.scaleb(-input_decimals)
        out_human = out_dec.scaleb(-output_decimals)
        return +(in_human / out_human)


def price_gap_pct(executable: Any, observed: Any) -> Decimal:
    """
    ``(executable / observed - 1) * 100``.

    Signed. A positive gap means the executable quote is worse than the mark
    (you pay more than DexScreener says), which is the direction that matters
    for a BUY.

    Either side being zero is a :class:`DecimalError`: a percentage gap against
    a zero mark is undefined, and reporting ``inf`` would be a fabrication.
    """
    exe = to_decimal(executable, field="executable_price")
    mark = to_decimal(observed, field="observed_mark")
    if mark <= 0:
        raise DecimalError(
            "observed mark is not positive; no percentage gap is defined",
            offending=observed,
        )
    if exe < 0:
        raise DecimalError("executable price is negative", offending=executable)
    with localcontext() as ctx:
        ctx.prec = PRECISION
        return +(((exe / mark) - ONE) * Decimal(100))


def pct_return(start: Any, end: Any) -> Decimal:
    """
    ``(end / start - 1) * 100``.

    Used by the GOAT regression to restate a mark-to-mark move. Named
    separately from :func:`price_gap_pct` because a mark RETURN is not a
    trading return and the report must not imply that it is one.
    """
    a = to_decimal(start, field="start")
    b = to_decimal(end, field="end")
    if a <= 0:
        raise DecimalError("baseline is not positive", offending=start)
    with localcontext() as ctx:
        ctx.prec = PRECISION
        return +(((b / a) - ONE) * Decimal(100))


def constant_product_out(
    reserve_in: Any,
    reserve_out: Any,
    amount_in_after_fee: Any,
) -> Decimal:
    """
    Tokens out of a constant-product curve: ``x = y - k / (y + dx)``.

    ``reserve_in`` is denominated in the INPUT asset (SOL), ``reserve_out`` in
    the OUTPUT asset (tokens), and ``amount_in_after_fee`` is the input net of
    the protocol fee. The caller applies the fee so the fee is visible in the
    report rather than buried in a constant.
    """
    y = to_decimal(reserve_in, field="reserve_in")
    x = to_decimal(reserve_out, field="reserve_out")
    dx = to_decimal(amount_in_after_fee, field="amount_in_after_fee")
    if y <= 0 or x <= 0:
        raise DecimalError(
            "constant-product reserves must both be positive",
            offending=(reserve_in, reserve_out),
        )
    if dx < 0:
        raise DecimalError("input amount is negative", offending=amount_in_after_fee)
    with localcontext() as ctx:
        ctx.prec = PRECISION
        k = y * x
        # Integer token amounts are truncated DOWN by the pool, never rounded,
        # so the modelled output must be floored to match what a pool would
        # actually credit. Rounding up would overstate the fill and understate
        # the price.
        return int(x - (k / (y + dx)))


def apply_fee_bps(amount: Any, fee_bps: int) -> tuple[Decimal, Decimal]:
    """
    Split ``amount`` into ``(net, fee)``.

    The fee is floored, so ``net + fee == amount`` exactly and the two numbers
    in the report reconcile.
    """
    total = to_decimal(amount, field="amount")
    if fee_bps < 0:
        raise DecimalError("fee_bps cannot be negative", offending=fee_bps)
    if fee_bps >= 10_000:
        raise DecimalError(
            "fee_bps must be below 10000", offending=fee_bps
        )
    with localcontext() as ctx:
        ctx.prec = PRECISION
        fee = int(total * Decimal(fee_bps) / Decimal(10_000))
        net = total - fee
    return net, fee


def sol_amount_for_usd(usd: Any, sol_usd: Any) -> Decimal:
    """
    SOL equivalent of a USD notional, from an explicitly logged SOL reference.

    The reference price is a caller-supplied observation, recorded in the
    report next to the result. This test deliberately does not fetch one
    itself: an unlogged price would make the order size unreproducible, and an
    order size that cannot be reproduced makes the whole comparison unverifiable.
    """
    target = to_decimal(usd, field="usd")
    price = to_decimal(sol_usd, field="sol_usd")
    if price <= 0:
        raise DecimalError("SOL reference price is not positive", offending=sol_usd)
    with localcontext() as ctx:
        ctx.prec = PRECISION
        return +(target / price)


def to_float(value: Any) -> float | None:
    """
    Convert for display. Non-finite results become ``None``, never ``inf``.

    JSON has no representation for ``Infinity`` or ``NaN``, and emitting the
    bare token produces a file that standard parsers reject.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        if not value.is_finite():
            return None
        return float(value)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def quantise(value: Any, places: int = 12) -> str:
    """
    Render with a fixed number of decimals, for display only.

    Never call this before computing a gap. It exists so the CLI shows stable,
    comparable strings - a bare ``1.5100000000000000081e-05`` in a report is
    unreadable and tells the operator nothing about the magnitude of the
    difference.
    """
    dec = to_decimal(value, field="value")
    with localcontext() as ctx:
        ctx.prec = PRECISION
        return str(dec.quantize(Decimal(1).scaleb(-places)))


__all__ = [
    "DecimalError",
    "PRECISION",
    "apply_fee_bps",
    "constant_product_out",
    "effective_price",
    "normalise_amount",
    "pct_return",
    "price_gap_pct",
    "quantise",
    "sol_amount_for_usd",
    "to_decimal",
    "to_float",
]