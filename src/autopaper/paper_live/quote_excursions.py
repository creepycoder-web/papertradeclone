"""
Quote-based MFE / MAE: execution-aware excursions, with honest absence.

WHAT THE DIFFERENCE IS
----------------------
``PaperPosition.observed_mark_mfe`` / ``observed_mark_mae`` are MARK excursions.
They are already named so nobody can read them as executable returns, and they
answer "how far did the observed mark move".

``PaperPosition.quote_mfe_pct`` / ``quote_mae_pct`` answer the other question:
"what would this position have realised if liquidated at this instant". They are
computed from SELL quotes for this position's EXACT held token quantity, and each
observation carries its own provenance.

THE ABSENCE RULE (the reason this module exists)
------------------------------------------------
A missing measurement is NOT a zero. A zero MFE is a real finding -- the position
never went up at all -- and a missing MFE is the absence of evidence. Collapsing
the two would report positions that were never measured as positions that never
moved, which is precisely the error this whole data discipline is meant to
prevent.

So:

* ``quote_mfe_pct`` / ``quote_mae_pct`` are ``Optional[float]``. ``None`` means
  "not measured".
* :attr:`PaperPosition.quote_mfe_mae_basis` is set to ``"MARK_ONLY"`` once it is
  established that no usable SELL quote was ever obtained, so a reader can tell
  "never went up" from "was never measured" at a glance.
* Nothing here substitutes a mark for an executable quote. Where no quote exists,
  the excursion stays ``None`` and the record says so.
"""

from __future__ import annotations

from typing import Any, Optional

#: The value ``quote_mfe_mae_basis`` carries before any observation exists.
BASIS_NOT_MEASURED = "NOT_MEASURED"

#: Set once it is established that no usable SELL quote was ever obtained. The
#: mark excursions are still recorded -- they are simply labelled as mark.
BASIS_MARK_ONLY = "MARK_ONLY"

#: Set once at least one usable SELL-quote observation contributed.
BASIS_EXECUTABLE_SELL_QUOTES = "EXECUTABLE_SELL_QUOTES"


def sell_quote_return_pct(
    *, position: Any, quote: Any, net_liquidation_return_pct: float | None = None,
) -> Optional[float]:
    """
    The return a SELL at this quote would have realised, WITH COSTS.

    Preferred input is ``net_liquidation_return_pct``, already computed by the
    caller as ``(net_sell_proceeds - all_in_entry_cost) / all_in_entry_cost``
    from the quote's own numbers. That is the cost-aware basis: the quote's
    venue fee, price impact and slippage are already embedded in its output
    amount, so they are NOT added again here.

    Falls back to computing from the quote's USD price against the position's
    entry fill ONLY when no net figure is supplied, and the result is the same
    quantity on the price basis. ``None`` is returned for a missing or unusable
    input -- never ``0.0``.
    """
    if net_liquidation_return_pct is not None:
        try:
            value = float(net_liquidation_return_pct)
        except (TypeError, ValueError):
            return None
        return value if value == value else None  # NaN guard

    if quote is None:
        return None

    # The quote must be for the SIDE the position would close on. A BUY quote
    # is not evidence about an exit, and reading it as one is the reciprocal
    # error the rest of this codebase guards against.
    if getattr(quote, "side", None) not in (None, "SELL"):
        return None
    if getattr(quote, "available", False) is False:
        return None

    price = getattr(quote, "effective_price_usd", None)
    entry = getattr(position, "entry_fill_price_usd", None)
    if price is None or entry is None:
        return None
    try:
        price, entry = float(price), float(entry)
    except (TypeError, ValueError):
        return None
    if not (price > 0 and entry > 0):
        return None
    return (price / entry - 1.0) * 100.0


def record_quote_excursion(
    position: Any, *, quote: Any, net_liquidation_return_pct: float | None = None,
) -> Optional[float]:
    """
    Fold one SELL-quote observation into the position's quote-based excursions.

    Returns the recorded return, or ``None`` when this observation could not be
    used -- in which case the position is left UNCHANGED and nothing is counted.
    """
    ret = sell_quote_return_pct(
        position=position, quote=quote,
        net_liquidation_return_pct=net_liquidation_return_pct,
    )
    if ret is None:
        # Still record WHY there is nothing: a position that has seen only
        # unusable quotes converges on MARK_ONLY, so the absence is legible.
        if getattr(position, "quote_observation_count", 0) == 0:
            position.quote_mfe_mae_basis = BASIS_MARK_ONLY
        return None

    if position.quote_mfe_pct is None or ret > position.quote_mfe_pct:
        position.quote_mfe_pct = ret
    if position.quote_mae_pct is None or ret < position.quote_mae_pct:
        position.quote_mae_pct = ret
    position.quote_observation_count = int(
        getattr(position, "quote_observation_count", 0) or 0) + 1
    position.quote_mfe_mae_basis = BASIS_EXECUTABLE_SELL_QUOTES
    return ret


def mark_only(position: Any) -> Any:
    """
    Mark a position's quote-based excursions as MARK_ONLY, leaving them ``None``.

    Used when a monitoring pass produces a mark but no usable SELL quote: the
    mark excursions above remain valid for what they are, and this record makes
    the executable absence explicit rather than implied.
    """
    if getattr(position, "quote_observation_count", 0) == 0:
        position.quote_mfe_mae_basis = BASIS_MARK_ONLY
    return position


def describe(position: Any) -> dict[str, Any]:
    """
    The machine-readable excursion record, with absence spelled out.

    ``None`` stays ``None`` in the output. ``"N/A"`` is used for the derived
    text fields so a report never renders a missing measurement as ``0.0``.
    """
    mfe = getattr(position, "quote_mfe_pct", None)
    mae = getattr(position, "quote_mae_pct", None)
    count = int(getattr(position, "quote_observation_count", 0) or 0)
    basis = getattr(position, "quote_mfe_mae_basis", BASIS_NOT_MEASURED) \
        or BASIS_NOT_MEASURED
    return {
        "quote_mfe_pct": mfe,
        "quote_mae_pct": mae,
        "quote_mfe_pct_display": "N/A" if mfe is None else f"{mfe:+.2f}%",
        "quote_mae_pct_display": "N/A" if mae is None else f"{mae:+.2f}%",
        "quote_observation_count": count,
        "quote_mfe_mae_basis": basis,
        "measured": count > 0 and mfe is not None,
        # The mark-based excursions, carried alongside so the two are never
        # pooled and never confused for one another.
        "observed_mark_mfe": getattr(position, "observed_mark_mfe", None),
        "observed_mark_mae": getattr(position, "observed_mark_mae", None),
        "observed_mark_label": "MARK (not executable)",
    }


__all__ = [
    "BASIS_EXECUTABLE_SELL_QUOTES",
    "BASIS_MARK_ONLY",
    "BASIS_NOT_MEASURED",
    "describe",
    "mark_only",
    "record_quote_excursion",
    "sell_quote_return_pct",
]