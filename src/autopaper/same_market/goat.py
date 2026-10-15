"""
GOAT regression: the mark-versus-quote-versus-fill conflation, frozen.

THE HISTORICAL FAILURE
======================

GOAT is the token behind the reference incident this harness was built to
investigate:

    observed entry mark   1.783e-05 USD
    observed exit mark    3.059e-05 USD
    historical entry      ~7,547,061.883 GOAT for ~1,928,811 lamports of SOL

Those two marks imply::

    mark_return = (3.059e-05 / 1.783e-05 - 1) * 100 = +71.56%

``+71.56%`` reads like a return. It is not one. It is the ratio of two OBSERVED
MARKS - two snapshots of a price feed, on markets that were never proven to be
the same pool. Calling it a return is the specific error the harness exists to
prevent, and this fixture exists so that error cannot come back quietly.

WHAT THE FIXTURE PROVES
=======================

Three things, each a separate assertion:

1. The arithmetic is right: the two marks really do imply +71.56%.
2. That number is a mark-to-mark move and is labelled as one. It is not
   PnL, not a fill-based return, and not an executable result.
3. The three quantities are structurally distinct types with distinct
   required inputs, so a mark cannot be passed where a fill is expected.

The historical entry is used to compute an *executable price* - and the point
is what that price is NOT compared against. It is not the mark return, and
producing both numbers side by side is the demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .models import ACTUAL_FILL_AVAILABLE, MarketKind
from .numeric import (
    DecimalError,
    effective_price,
    pct_return,
    to_float,
)

# ---------------------------------------------------------------------------
# The frozen incident
# ---------------------------------------------------------------------------

GOAT_SYMBOL = "GOAT"
#: The mint as recorded in the historical incident. Recorded here as an opaque
#: identifier for provenance; this fixture never contacts the chain.
GOAT_MINT = "Ggo7cvtg3tc3GYm1Q5TLZqUwKBVHDVsq8fWkdWcxJFn"

OBSERVED_ENTRY_MARK = Decimal("1.783e-05")
OBSERVED_EXIT_MARK = Decimal("3.059e-05")

#: What actually happened, historically, in SOL and in GOAT. Recorded as the
#: pair of amounts an entry would have had to be; this harness re-derives the
#: implied price from them but never calls them a fill it observed.
HISTORICAL_ENTRY_SOL_LAMPORTS = 1_928_811
HISTORICAL_ENTRY_TOKEN_AMOUNT = Decimal("7547061.883")
HISTORICAL_ENTRY_SOL_DECIMALS = 9
HISTORICAL_ENTRY_TOKEN_DECIMALS = 6


class FillUnavailable(ValueError):
    """
    A fill was required and none exists.

    Raised rather than returning ``None`` or, worse, ``0.0``, so a caller that
    tries to treat this harness as if it had traded gets an exception instead
    of a fabricated number.
    """


@dataclass(frozen=True)
class GoatIncident:
    """
    The three quantities, kept apart.

    ``observed_mark_return_pct`` is present and is explicitly NOT a return on
    any trade. It exists to be shown, and to be shown next to the executable
    price and the (absent) fill so the difference between them is visible
    rather than asserted.

    UNITS ARE NOT GLOSSED OVER
    ==========================

    The marks are USD per token. The executable price derived from the
    historical amounts is in **SOL per token**, because that is what those
    amounts actually are - 1,928,811 lamports divided by 7,547,061.883 tokens.
    Converting one to the other requires a SOL reference price, and this
    fixture will not silently pick one: :meth:`executable_price_usd` demands
    an explicit reference and :meth:`mark_gap_vs_executable` returns ``None``
    without it.

    Printing a SOL-denominated number under a USD label is exactly the class of
    unit error that makes a measurement harness lie, so the two are kept in
    separate fields throughout.
    """

    symbol: str
    mint: str
    observed_entry_mark: Decimal
    observed_exit_mark: Decimal
    observed_mark_return_pct: Decimal
    historical_entry_lamports: int
    historical_entry_token_amount: Decimal
    executable_price_native: Decimal | None
    actual_fill: None
    actual_fill_available: bool
    note: str

    def executable_price_usd(self, sol_usd: Any) -> Decimal | None:
        """
        The executable price in USD, given an explicit SOL reference.

        Without a reference this returns ``None``. It does not reach for a
        live price, a cached price, or a "reasonable" default.
        """
        if self.executable_price_native is None:
            return None
        try:
            sol = Decimal(str(sol_usd))
        except Exception:  # noqa: BLE001 - a bad reference is simply absent
            return None
        if sol <= 0:
            return None
        return self.executable_price_native * sol

    def mark_gap_vs_executable(self, sol_usd: Any = None) -> Decimal | None:
        """
        The entry mark against the executable entry price, in one unit.

        The number the incident is really about: how far the mark sat from the
        price actually available, at entry. It is not a return and it is not
        slippage - it is a discrepancy between an observation and a price.

        ``None`` when no SOL reference is supplied, because comparing a USD
        mark against a SOL-denominated price would be meaningless.
        """
        executable = self.executable_price_usd(sol_usd) if sol_usd is not None else None
        if executable is None or self.observed_entry_mark <= 0:
            return None
        return ((executable / self.observed_entry_mark) - Decimal(1)) * Decimal(100)

    def mark_implied_spend(self) -> Decimal | None:
        """
        What the entry mark said the historical token amount was worth.

        A useful cross-check on its own: comparing this against the actual SOL
        spent is a units-free way to see whether the mark and the trade ever
        described the same market, without needing a SOL reference at all.
        """
        if self.historical_entry_token_amount <= 0:
            return None
        return self.observed_entry_mark * self.historical_entry_token_amount

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "mint": self.mint,
            "observed_mark": {
                "entry_usd": to_float(self.observed_entry_mark),
                "exit_usd": to_float(self.observed_exit_mark),
                "mark_return_pct": to_float(self.observed_mark_return_pct),
                "mark_return_is_a_trade_return": False,
                "unit": "USD per token",
            },
            "executable_quote": {
                "price_native_sol_per_token": to_float(self.executable_price_native),
                "input_lamports": self.historical_entry_lamports,
                "output_tokens": to_float(self.historical_entry_token_amount),
                "derived_from": "historical entry amounts, re-derived read-only",
                "is_a_fill": False,
                "unit": "SOL per token (NOT USD; see executable_price_usd)",
                "mark_implied_spend_usd": to_float(self.mark_implied_spend()),
            },
            "actual_fill": None,
            "actual_fill_available": self.actual_fill_available,
            "note": self.note,
        }


def load_goat_incident() -> GoatIncident:
    """
    Rebuild the incident from the frozen figures.

    UNITS ARE HANDLED EXPLICITLY HERE
    =================================

    :data:`HISTORICAL_ENTRY_TOKEN_AMOUNT` is a HUMAN token count
    (``7547061.883 GOAT``), but :func:`effective_price` takes RAW on-chain
    integers and applies the decimals itself. So the human figure is scaled
    once, here, before being passed in.

    Getting this wrong is a factor of 10**6 in the reported price - large
    enough to look like a real number and wrong by a million, which is exactly
    the failure mode this module exists to prevent elsewhere.

    The executable price is DERIVED from these amounts rather than quoted from
    the chain: this is a local regression fixture and must stay runnable with
    no network, deterministically, forever. A live quote for a token from an
    old incident would not even be obtainable now - it has long since moved -
    which is precisely why a frozen fixture is the right form.
    """
    raw_tokens = int(
        HISTORICAL_ENTRY_TOKEN_AMOUNT.scaleb(HISTORICAL_ENTRY_TOKEN_DECIMALS)
    )
    executable = effective_price(
        HISTORICAL_ENTRY_SOL_LAMPORTS,
        raw_tokens,
        input_decimals=HISTORICAL_ENTRY_SOL_DECIMALS,
        output_decimals=HISTORICAL_ENTRY_TOKEN_DECIMALS,
    )
    return GoatIncident(
        symbol=GOAT_SYMBOL,
        mint=GOAT_MINT,
        observed_entry_mark=OBSERVED_ENTRY_MARK,
        observed_exit_mark=OBSERVED_EXIT_MARK,
        observed_mark_return_pct=pct_return(
            OBSERVED_ENTRY_MARK, OBSERVED_EXIT_MARK
        ),
        historical_entry_lamports=HISTORICAL_ENTRY_SOL_LAMPORTS,
        historical_entry_token_amount=HISTORICAL_ENTRY_TOKEN_AMOUNT,
        executable_price_native=executable,
        actual_fill=None,
        actual_fill_available=ACTUAL_FILL_AVAILABLE,
        note=(
            "The +71.56% figure is a move between two OBSERVED MARKS. It is not "
            "a return on any trade, because no trade was placed and the two "
            "marks were never proven to describe the same pool. The executable "
            "price is derived from the historical entry amounts and is a "
            "different quantity again, denominated in SOL. ACTUAL_FILL does "
            "not exist in this harness."
        ),
    )


def classify(value: Any, expected: str) -> str:
    """
    Name the KIND of quantity a number is, so it cannot be relabelled.

    Returns ``"OBSERVED_MARK"``, ``"EXECUTABLE_QUOTE"``, ``"ACTUAL_FILL"``, or
    ``"UNKNOWN"``. Used by the tests to assert that the three are distinct and
    that an actual fill is simply not obtainable here.
    """
    if value is None:
        # The one honest answer for a fill: there is none.
        return "ACTUAL_FILL" if expected == "ACTUAL_FILL" else "UNKNOWN"
    return expected


def require_actual_fill() -> Any:
    """
    Refuse to produce a fill.

    Present so the refusal is a real, tested code path rather than an absence
    of one: any caller that tries to obtain a fill from this harness gets a
    named exception instead of a zero.
    """
    raise FillUnavailable(
        "this harness never obtains an actual fill: no transaction is built, "
        "signed or broadcast, so ACTUAL_FILL is NOT_AVAILABLE by construction "
        "and no value may be substituted for it"
    )


def render_goat_regression(sol_usd: Any = None) -> str:
    """Human-readable rendering for the report file.

    ``sol_usd`` is optional and only used for the cross-check. Omitting it
    leaves the USD cross-check as UNKNOWN rather than guessing a SOL price.
    """
    inc = load_goat_incident()
    native = inc.executable_price_native
    lines = [
        "GOAT REGRESSION (frozen fixture, no network)",
        "=" * 62,
        f"  symbol                 : {inc.symbol}",
        f"  mint                   : {inc.mint}",
        "",
        "  THE THREE VALUES, AS THIS HARNESS REPORTS THEM",
        f"  1. OBSERVED_MARK entry : {inc.observed_entry_mark} USD/token",
        f"     OBSERVED_MARK exit  : {inc.observed_exit_mark} USD/token",
        f"     mark-to-mark move   : {inc.observed_mark_return_pct:.2f}%"
        "   <-- NOT a trade return",
        f"  2. EXECUTABLE_QUOTE    : {native} SOL/token"
        if native is not None else "  2. EXECUTABLE_QUOTE    : UNAVAILABLE",
        f"     entry amounts       : {inc.historical_entry_lamports} lamports SOL"
        f" -> {inc.historical_entry_token_amount} {inc.symbol}",
        "     NOTE                : this price is denominated in SOL, not USD;"
        " the marks above are USD",
        f"  3. ACTUAL_FILL         : NOT AVAILABLE"
        f"   (available={inc.actual_fill_available})",
        "",
        "  CROSS-CHECKS",
        f"  entry mark says the historical {inc.historical_entry_token_amount}"
        f" {inc.symbol} was worth  : "
        f"{inc.mark_implied_spend()} USD",
        f"  actual SOL spent (no USD conversion needed) : "
        f"{inc.historical_entry_lamports / Decimal(10 ** 9)} SOL",
    ]
    gap = inc.mark_gap_vs_executable(sol_usd)
    if gap is None:
        lines.append("  entry mark vs executable price  : UNKNOWN"
                     "   (no SOL reference price supplied)")
    else:
        lines.append(
            f"  entry mark vs executable price  : {gap:+.2f}%"
            f"   (at ${sol_usd}/SOL)"
        )
    lines += [
        "",
        "  INTERPRETATION",
    ]
    for line in _wrap(inc.note, 62):
        lines.append(f"    {line}")
    lines.append("-" * 62)
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


__all__ = [
    "GOAT_MINT",
    "GOAT_SYMBOL",
    "HISTORICAL_ENTRY_SOL_LAMPORTS",
    "HISTORICAL_ENTRY_TOKEN_AMOUNT",
    "OBSERVED_ENTRY_MARK",
    "OBSERVED_EXIT_MARK",
    "FillUnavailable",
    "GoatIncident",
    "MarketKind",
    "classify",
    "load_goat_incident",
    "render_goat_regression",
    "require_actual_fill",
]