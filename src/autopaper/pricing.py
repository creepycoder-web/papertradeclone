"""
Price provenance: which of these numbers is a mark, which is a quote, and which
is money that actually moved.

WHY THIS MODULE EXISTS
----------------------
On 2026-10-01 the $BUTTHOLE REAL position reported::

    Entry   1.181000e-05
    Current 2.043000e-05
    PnL     +72.99%

and actually realized **-$0.071**. Every one of those four numbers came from the
same place -- a third-party indexed pool mark (``DexScreener.priceUsd``) -- while
the bot held an authoritative price from the venue it actually traded on and
never compared the two. The mark read 0.55x the venue price at entry and 4.09x
it at exit, and nothing in the system was capable of noticing.

The defect is not a wrong number. It is that **three different kinds of number
shared one field name**, so a spot mark could be read, stored and displayed as an
execution price without any type-level or accounting-level distinction.

THE THREE KINDS, AND WHY THEY MUST NOT BE CONFLATED
----------------------------------------------------
``OBSERVED_MARK``
    A third-party indexer's view: a last print, a derived spot, or a pool
    reserve ratio. Useful, cheap, continuous, and **not an offer**. It is not
    sized to your order, carries no impact, and cannot be transacted against.
``VENUE_QUOTE``
    What the execution venue will actually do for *your* size, right now. This
    is the only price that can become an order.
``ACTUAL_EXECUTION``
    What the chain says happened, derived from the settled amounts. This is the
    only price that may appear in realized PnL.

A position can show ``OBSERVED_MFE = +72.99%`` and realize ``-76%`` and both be
correct, because they answer different questions. The failure is not that MFE
exists; it is that MFE was *named and stored as though it were the exit price*.

WHAT THIS MODULE GUARANTEES
---------------------------
* An execution price is always derived from settled amounts, in the correct
  orientation for the side, and is ``None`` -- never ``0.0`` -- when the inputs
  cannot support it.
* A mark and a venue quote are never silently substituted for one another.
* A divergence between them is a first-class, comparable number rather than
  something a reader has to reconstruct.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

LAMPORTS_PER_SOL = 1_000_000_000


class PriceProvenance(str, Enum):
    """
    EXACTLY which of these five things produced a number.

    Added after the forensic reconciliation of the 26 divergent PAPER exits
    (``reports/research/l25_exec_forensic_divergent_exits.md``) established that
    22 of them were booked strictly BELOW the lowest price the runtime had observed
    at any point during the hold. A price outside the entire observed band cannot
    come from the monitored book, so every stored execution-side number must say
    where it came from -- otherwise the next reconciliation has nothing to test.

    ``MARK_ONLY``
        A third-party indexed mark and nothing else. **This is not a fill.** It may
        appear in a PnL only when the record says, in the same field, that the PnL
        is mark-to-mark.
    ``NATIVE_REFERENCE``
        A venue's own marginal spot price read from its on-chain state (for a
        pump.fun bonding curve, the curve's reserve ratio). It is that venue's
        price at its own size assumptions; it is not this order's fill.
    ``EXECUTABLE_QUOTE``
        What the execution venue will do for THIS order's size, right now. The
        only thing a simulated fill may be built from.
    ``PAPER_FILL``
        A SIMULATED execution derived from an ``EXECUTABLE_QUOTE`` by a cost model.
        It is the best answer available without a signature, and it must never be
        written into a field whose name says "fill" without this label travelling
        with it.
    ``ONCHAIN_FILL``
        Derived from settled on-chain amounts. The only provenance that may be
        described as an actual fill.
    """

    MARK_ONLY = "MARK_ONLY"
    NATIVE_REFERENCE = "NATIVE_REFERENCE"
    EXECUTABLE_QUOTE = "EXECUTABLE_QUOTE"
    PAPER_FILL = "PAPER_FILL"
    ONCHAIN_FILL = "ONCHAIN_FILL"


#: Provenances that may be described as an actual, signed execution.
ACTUAL_FILL_PROVENANCE: frozenset[str] = frozenset({PriceProvenance.ONCHAIN_FILL.value})

#: Provenances that are NOT an execution at all, in either direction.
NON_EXECUTION_PROVENANCE: frozenset[str] = frozenset({
    PriceProvenance.MARK_ONLY.value,
})


def is_actual_fill(provenance: Optional[str]) -> bool:
    """Only ``ONCHAIN_FILL`` may be called an actual fill. Never inferred."""
    return str(provenance or "") in ACTUAL_FILL_PROVENANCE


class PriceKind(str, Enum):
    """What a price number actually IS. Never inferred from context."""

    #: Third-party indexed mark. Not an offer.
    OBSERVED_MARK = "OBSERVED_MARK"
    #: The venue's price for this order's size, now.
    VENUE_QUOTE = "VENUE_QUOTE"
    #: Derived from settled on-chain amounts.
    ACTUAL_EXECUTION = "ACTUAL_EXECUTION"
    #: Not a price at all -- an error, a missing value, or an empty quote.
    UNAVAILABLE = "UNAVAILABLE"


def classify(price: Optional[float], *, known_kind: Optional[PriceKind] = None
             ) -> PriceKind:
    """
    Classify a price, refusing to guess.

    A caller that knows the provenance should pass ``known_kind``; that is the
    only way this returns a positive kind for a bare float, because a number
    alone carries no provenance. ``0.0`` and ``None`` are ``UNAVAILABLE`` in
    every case -- a zero price is a bug that once reached a journal as an
    execution price, and it must never be read as a real quote.
    """
    if known_kind is not None:
        return known_kind
    if price is None:
        return PriceKind.UNAVAILABLE
    try:
        p = float(price)
    except (TypeError, ValueError):
        return PriceKind.UNAVAILABLE
    if p != p or p <= 0 or p == float("inf"):
        return PriceKind.UNAVAILABLE
    return PriceKind.UNAVAILABLE  # a positive float alone is not provenance


# ---------------------------------------------------------------------------
# execution price, derived from settled amounts
# ---------------------------------------------------------------------------


def normalize_side(side: Optional[str]) -> Optional[str]:
    """
    Map a declared direction onto the canonical ``"BUY"`` / ``"SELL"`` pair.

    Two vocabularies exist in this codebase and they mean the same two things:
    the venue-facing one (``"BUY"`` / ``"SELL"``) and the accounting one
    (``"ENTRY"`` / ``"EXIT"``). ``ENTRY`` is a BUY (quote -> token) and ``EXIT``
    is a SELL (token -> quote), exactly.

    Anything else returns ``None``, and every caller below then refuses. That is
    the point: a direction that cannot be named exactly must not be guessed at,
    because guessing at it is what inverted a sell-side price on 2026-10-02.
    """
    s = (side or "").strip().upper()
    if s in ("BUY", "ENTRY"):
        return "BUY"
    if s in ("SELL", "EXIT"):
        return "SELL"
    return None


def execution_price_per_token(
    *, side: str, input_amount: Optional[int], output_amount: Optional[int],
) -> Optional[float]:
    """
    Price actually paid/received per RAW unit of the output asset, or ``None``.

    THE ORIENTATION IS THE POINT, AND IT IS TAKEN FROM ``side`` -- NEVER INFERRED
    from mints, from which field happens to be populated, or from sign. A BUY
    spends ``input`` to receive ``output``, so the price is ``input / output``. A
    SELL gives up ``input`` (tokens) and receives ``output`` (lamports), so the
    price is ``output / input``.

    One formula for both is exactly how the $BUTTHOLE exit published an execution
    price of ``0.0``: the shared expression ``(input_amount or 0) /
    output_amount`` was evaluated on the exit side, where the input leg is
    frequently absent for a native-SOL unwind, and the resulting zero was written
    to the journal as though it had been measured. A price of zero is not a price
    and is never returned as one.

    ``None`` is returned whenever the inputs cannot support the division: an
    unrecognised or missing direction, a missing leg, a zero, or a negative. A
    plausible-looking execution price is worse than an absent one, because an
    absent one cannot be booked.
    """
    side_u = normalize_side(side)
    if side_u is None:
        return None
    if input_amount is None or output_amount is None:
        return None
    try:
        i, o = int(input_amount), int(output_amount)
    except (TypeError, ValueError):
        return None
    if i <= 0 or o <= 0:
        return None
    return (i / o) if side_u == "BUY" else (o / i)


def realized_execution_price(
    *, side: str, input_amount: Optional[int], output_amount: Optional[int],
    sol_price_usd: Optional[float], token_decimals: int = 6,
) -> Optional[float]:
    """
    Realized execution price in USD per WHOLE token, or ``None``.

    Wraps :func:`execution_price_per_token` with the SOL conversion so callers
    never re-derive the inversion. ``token_decimals`` is the number of raw units
    in one whole token; it is a parameter rather than a lookup so the caller is
    forced to be explicit about which mint's decimals apply.

    On a SELL the token is the INPUT leg, so the token's decimals belong in
    ``input_decimals``. The parameter keeps its name for call-site
    compatibility, and both spellings are accepted.
    """
    side_u = normalize_side(side)
    if side_u is None:
        return None
    if sol_price_usd is None or sol_price_usd <= 0:
        return None
    per_raw = execution_price_per_token(
        side=side_u, input_amount=input_amount, output_amount=output_amount)
    if per_raw is None:
        return None
    try:
        scale = 10 ** max(0, int(token_decimals))
    except (TypeError, ValueError):
        return None
    return (per_raw / LAMPORTS_PER_SOL) * scale * float(sol_price_usd)



# ---------------------------------------------------------------------------
# divergence between what was marked and what was quotable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Divergence:
    """
    How far a mark sat from the venue's own price for the same instant.

    ``pct`` is ``(mark - venue) / venue``. A POSITIVE value means the mark was
    ABOVE the executable price, which flatters an exit: the position looks more
    valuable than it is sellable. A NEGATIVE value means the mark was BELOW,
    which flatters an entry: the position looks cheaper to buy than it is.

    Both directions are dangerous and they do not cancel. An entry that fills
    worse than its mark silently shrinks the position; an exit that marks better
    than its fill silently inflates the result.
    """

    mark: float
    venue: float
    pct: float
    kind: str

    @property
    def mark_overstates(self) -> bool:
        return self.pct > 0

    def as_dict(self) -> dict:
        return {"mark": self.mark, "venue": self.venue, "pct": self.pct,
                "kind": self.kind, "mark_overstates": self.mark_overstates}


def divergence(mark: Optional[float], venue: Optional[float], *,
               kind: str = "UNSPECIFIED") -> Optional[Divergence]:
    """
    Mark-versus-venue divergence, or ``None`` when either side is unusable.

    Returning ``None`` for an unusable pair is deliberate: a divergence computed
    from a zero or a missing quote is a fabricated number, and a fabricated
    ratio is exactly the class of thing this audit exists to remove.
    """
    if mark is None or venue is None:
        return None
    try:
        m, v = float(mark), float(venue)
    except (TypeError, ValueError):
        return None
    if not (m > 0) or not (v > 0) or m != m or v != v:
        return None
    return Divergence(mark=m, venue=v, pct=(m - v) / v, kind=kind)


# ---------------------------------------------------------------------------
# observed excursions, explicitly NOT realizable
# ---------------------------------------------------------------------------


def observed_return_pct(price: Optional[float], entry_price: Optional[float]
                       ) -> Optional[float]:
    """
    Percent move of an OBSERVED mark from an OBSERVED entry mark.

    This is an excursion statistic. It is deliberately a function of two marks and
    nothing else -- it never reads a fill, an order, or a balance -- so it cannot
    silently acquire the authority of an execution price. The docstring on
    :class:`PriceKind` is the contract: ``OBSERVED_MARK`` in, excursion out.
    """
    if price is None or entry_price is None:
        return None
    try:
        p, e = float(price), float(entry_price)
    except (TypeError, ValueError):
        return None
    if not (p > 0) or not (e > 0):
        return None
    return (p / e - 1.0) * 100.0


def label_for(kind: PriceKind) -> str:
    """Operator-facing wording, so a screen cannot imply more than it knows."""
    return {
        PriceKind.OBSERVED_MARK: "observed market mark (not an executable quote)",
        PriceKind.VENUE_QUOTE: "venue quote for this size",
        PriceKind.ACTUAL_EXECUTION: "actual on-chain execution",
        PriceKind.UNAVAILABLE: "unavailable",
    }[kind]
