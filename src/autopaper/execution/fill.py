"""
Paper fill model.

This is a SIMULATION and is documented as such. It does not claim to reproduce
any real historical execution. What it does is refuse to be optimistic:

    quoted_price      = the price visible at decision time
    fill_price        = the price visible at (decision_time + decision_latency),
                        i.e. after the market has had a real chance to move
    fill_price       += slippage, composed of
                          - a base component for a thin memecoin book
                          - size impact, sqrt(notional / liquidity)
                          - a staleness penalty if the print is not fresh
    notional          = quantity * fill_price
    fee               = notional * fee_bps
    cash_out          = notional + fee

Failure modes are first-class, not exceptional:

* no quote within the timeout        -> ``UNFILLED / no_quote`` (recorded, not hidden)
* quote older than ``max_quote_age`` -> ``UNFILLED / stale_quote``
* token disappeared                  -> ``UNFILLED / market_unavailable``
* price moved more than the limit    -> filled at the WORSE of the two prices

There is no path in this module that selects the better of two prices, and no
path that uses a future candle as the fill. The engine calls ``simulate_fill``
at a moment in wall-clock time and only the observation available at that moment
can be used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

from ..config import ExecutionModel
from ..logging_setup import log
from ..models import DataState, FillResult, Observation
from ..pricing import PriceProvenance
from ..safety import AGENT_ID


@dataclass(frozen=True)
class FillInputs:
    """
    Everything the model is allowed to see at fill time.

    THE THREE LAYERS ARE THREE FIELDS, NOT ONE
    -----------------------------------------
    ``observation`` is an OBSERVED_MARK: a third-party indexer price. ``quote``
    is an EXECUTABLE_QUOTE: what a venue would do for this size. This function
    produces a SIMULATED_FILL, which is a third kind of number again.

    They are separate because on 2026-10-02 a REAL position was reported with a
    percentage computed from two marks and a dollar computed from the chain, on
    one line. Keeping the layers apart here means a PAPER trade carries the same
    distinction a REAL one does, instead of pretending PAPER has no such problem
    because PAPER never settles.
    """

    side: str                       # "entry" | "exit"
    quantity: float                 # token units for entry, units to sell for exit
    notional_hint_usd: float        # target USD size (used for impact only)
    quoted_price: float | None      # price at decision time
    decision_time: float
    observation: Observation | None  # the OBSERVED_MARK available AT fill time
    observed_at: float
    latency_ms: float | None = None
    #: An EXECUTABLE_QUOTE for this size, when one could be obtained. ``None``
    #: means "not established" and is recorded as missing WITH A REASON, never
    #: as zero and never silently replaced by the mark.
    quote: Any = None
    #: Why ``quote`` is ``None``. Empty string when a quote was present.
    quote_missing_reason: str = ""


#: How the liquidity used for impact was obtained. Surfaced on every fill so a
#: reader can tell a measured impact from a conservative stand-in.
LIQUIDITY_OBSERVED = "OBSERVED"
LIQUIDITY_UNKNOWN = "UNKNOWN"


def liquidity_source(liquidity_usd: float | None) -> str:
    """
    Whether ``liquidity_usd`` is a real number or an absence.

    An absent pool liquidity is a fact about the world, not a licence to invent
    one. Callers use this to label the fill rather than to hide it.
    """
    return LIQUIDITY_OBSERVED if (liquidity_usd or 0) > 0 else LIQUIDITY_UNKNOWN


def impact_bps(notional_usd: float, liquidity_usd: float | None,
               model: ExecutionModel) -> float:
    """
    Deterministic size-impact approximation.

    ``sqrt(notional / liquidity)`` scaled so that consuming 1% of a pool's
    liquidity costs ~9 bps and 25% of it costs ~45 bps. Capped at
    ``impact_max_bps``.

    UNKNOWN LIQUIDITY IS NOT SUBSTITUTED.
    ---------------------------
    This used to fall back to ``model.assume_sol_usd`` ($60,000 by default),
    which was worse than useless twice over:

    * it was a fabricated number presented in the fill's own ``detail`` string
      as if it had been read off the venue, and
    * it made impact collapse to ~1 bp on a $12 order, i.e. it silently
      **flattered** the fill -- the opposite of the "deliberately small pool"
      comment that justified it. A real $17k pool on the same order is ~2.4 bp
      and a $2k pool is ~12 bp, so one constant cannot be conservative for all
      of them.

    When liquidity cannot be read, the honest price is "I cannot compute this",
    so we charge the configured worst case (``impact_unknown_bps``, defaulting
    to the ``impact_max_bps`` cap) and report ``liquidity_source=UNKNOWN`` so
    the fill is labelled lower-fidelity rather than presented as measured.
    """
    if notional_usd <= 0:
        return 0.0
    if (liquidity_usd or 0) <= 0:
        # No liquidity: charge the worst case we are willing to model, and say so.
        unknown = getattr(model, "impact_unknown_bps", None)
        return float(model.impact_max_bps if unknown is None else unknown)
    frac = notional_usd / float(liquidity_usd)
    raw = 100.0 * math.sqrt(max(0.0, frac)) * 0.9
    return min(model.impact_max_bps, raw)


def staleness_bps(quote_age_ms: float, model: ExecutionModel) -> float:
    if quote_age_ms <= 0:
        return 0.0
    penalty = (quote_age_ms / 1000.0) * model.staleness_penalty_bps_per_sec
    return min(model.staleness_penalty_max_bps, penalty)


def total_slippage_bps(
    *, side: str, notional_usd: float, liquidity_usd: float | None,
    quote_age_ms: float, model: ExecutionModel
) -> tuple[float, float, float]:
    """Returns ``(total_bps, base_bps, impact_bps)``; staleness folded into total."""
    base = model.entry_slippage_bps if side == "entry" else model.exit_slippage_bps
    imp = impact_bps(notional_usd, liquidity_usd, model)
    stale = staleness_bps(quote_age_ms, model)
    return base + imp + stale, base, imp


def _quote_price_of(quote: Any) -> Optional[float]:
    """
    An EXECUTABLE_QUOTE's price per whole token IN THE QUOTE ASSET, or ``None``.

    ``None`` for anything that is not a quote -- including ``None`` itself. A mark
    must never be able to arrive here wearing a quote's name, because that is the
    substitution that made the "worse of quote and current" guard inert.

    THE UNIT IS THE QUOTE ASSET. A SOL-quoted leg yields SOL per whole token, NOT
    USD. See :func:`_quote_price_usd_of` for the USD form, which is the only one
    that may be compared against an ``Observation.price_usd``.
    """
    if quote is None:
        return None
    if getattr(quote, "is_executable", None) is False:
        return None
    price = getattr(quote, "effective_price_per_whole_token", None)
    try:
        price = float(price)
    except (TypeError, ValueError):
        return None
    return price if price > 0 else None


def _quote_price_usd_of(quote: Any) -> Optional[float]:
    """The quote's USD per whole token, or ``None`` when not convertible."""
    if quote is None:
        return None
    try:
        px = float(getattr(quote, "price_usd_per_whole_token", None))
    except (TypeError, ValueError):
        return None
    return px if px > 0 else None



def fill_books_reconcile(fill: Any, *, quantity: float | None = None,
                        tol: float = 1e-6) -> tuple[bool, str]:
    """
    Do a fill's OWN books add up?

    ``(exit_fill_price - entry_fill_price) * qty - fees == cash_delta`` holds for any
    pair of fills, INCLUDING a pair that lies: a fill that claims one reference price
    and another executed price while charging no slippage for the difference is
    internally consistent with itself and with the cash movement.

    That lie used to be caught by the account, because the account derived gross
    from the REFERENCE prices. Realized PnL is now derived from the FILLS -- which is
    the correct basis and the only one that can equal the cash movement exactly --
    so the account can no longer see it. The check moves here, to the only place
    that can still see it: the fill's own slippage must equal the price difference
    it claims to have charged.

    ``quantity`` is the quantity actually being transacted, which the caller knows
    even when the fill object does not carry it. Passing it matters: a fill that
    omits its quantity is exactly the shape that would otherwise be uncheckable.

    Returns ``(ok, reason)``. ``ok`` is True when the fill cannot be checked at all
    (no quantity, or no reference price to compare against), because an uncheckable
    fill is not a proven liar.
    """
    qty = quantity if quantity is not None else getattr(fill, "quantity", None)
    ref = getattr(fill, "price_at_quote", None)
    px = getattr(fill, "fill_price", None)
    slip = getattr(fill, "slippage_usd", None)
    try:
        qty = float(qty) if qty is not None else 0.0
        ref = float(ref) if ref is not None else None
        px = float(px) if px is not None else None
        slip = float(slip) if slip is not None else None
    except (TypeError, ValueError):
        return True, "fill price fields are not numeric; nothing to reconcile"
    if qty <= 0 or ref is None or px is None or slip is None or ref <= 0 or px <= 0:
        return True, "fill carries no quantity/reference, so no slippage is claimed"
    implied = qty * abs(px - ref)
    if abs(implied - slip) <= max(1e-9, tol * max(1.0, implied)):
        return True, ""
    return False, (
        f"fill claims a reference of {ref:.12g} and an execution of {px:.12g} over "
        f"{qty:.6g} units, which implies {implied:.10f} of slippage, but books "
        f"{slip:.10f}; a close that does not reconcile its own books is refused rather "
        f"than booked")

def simulate_fill(
    inputs: FillInputs,
    model: ExecutionModel,
    *,
    data_state: DataState = DataState.LIVE,
    wait_for_quote: Callable[[float], Observation | None] | None = None,
    timeout_ms: float | None = None,
) -> FillResult:
    """
    Simulate one paper fill.

    ``wait_for_quote`` is how live mode supplies the *current* observation once
    the simulated execution timestamp has actually arrived. It is a callback on
    wall-clock time, not a lookup into history, which is what keeps this honest:
    at the moment of the call, no future information exists yet.
    """
    timeout = model.fill_quote_timeout_ms if timeout_ms is None else timeout_ms
    side = inputs.side
    fee_bps = model.entry_fee_bps if side == "entry" else model.exit_fee_bps
    sign = 1.0 if side == "entry" else -1.0
    quoted = inputs.quoted_price
    placed_at = inputs.observed_at
    latency = (
        inputs.latency_ms
        if inputs.latency_ms is not None
        else (inputs.observed_at - inputs.decision_time) * 1000.0
    )

    obs = inputs.observation
    if obs is None and wait_for_quote is not None:
        # Blocked upstream (or the caller wants us to source it ourselves).
        if side == "exit":
            obs = inputs.observation
        else:
            obs = wait_for_quote(timeout)

    # -- no market ---------------------------------------------------------
    if obs is None or obs.price_usd is None or obs.price_usd <= 0:
        return FillResult(
            filled=False,
            reason="no_quote" if obs is None else "market_unavailable",
            quoted_price=quoted,
            fill_price=None,
            slippage_bps=0.0,
            slippage_usd=0.0,
            fee_bps=fee_bps,
            fee_usd=0.0,
            latency_ms=latency,
            placed_at=placed_at,
            fill_at=inputs.observed_at,
            observed_at=obs.observed_at if obs else None,
            price_at_quote=obs.price_usd if obs else None,
            data_state=data_state,
            notional_usd=0.0,
            quantity=0.0,
            detail="no usable price at the simulated execution timestamp; trade NOT filled",
        )

    price = obs.price_usd
    quote_age_ms = max(0.0, (inputs.observed_at - obs.observed_at) * 1000.0)
    if quote_age_ms > model.max_quote_age_ms:
        return FillResult(
            filled=False,
            reason="stale_quote",
            quoted_price=quoted,
            fill_price=None,
            slippage_bps=0.0,
            slippage_usd=0.0,
            fee_bps=fee_bps,
            fee_usd=0.0,
            latency_ms=latency,
            placed_at=placed_at,
            fill_at=inputs.observed_at,
            observed_at=obs.observed_at,
            price_at_quote=price,
            data_state=data_state,
            detail=(
                f"best available print is {quote_age_ms:.0f}ms old "
                f"(limit {model.max_quote_age_ms:.0f}ms); refusing to fill"
            ),
        )

    # -- price: the worse of the EXECUTABLE quote and the observed mark ----
    #
    # For a BUY the worse price is higher; for a SELL it is lower. Getting this
    # backwards would hand the paper account free slippage on every exit.
    #
    # AN EXECUTABLE QUOTE IS A REAL SECOND OPINION. When one is available the
    # selection is ``worse(quote, mark)``, which is a genuine guard. Before this
    # the "quote" leg was ``position.mark_price`` -- the SAME DexScreener mark,
    # microseconds apart -- so the selection was a no-op and the guard was inert
    # on every PAPER exit while appearing to protect them. That inertness is
    # recorded on the result (``quote_guard_effective``) and the executable leg
    # is now a separate, named field.
    is_entry = side == "entry"
    # THE UNITS MUST MATCH BEFORE THE NUMBERS ARE COMPARED.
    #
    # ``price`` above is ``Observation.price_usd``: USD per token. A quote's
    # ``effective_price_per_whole_token`` is in the QUOTE ASSET -- SOL per token for
    # a SOL-quoted leg. Comparing them directly is a unit error, and for a SELL it
    # is a one-directional one: ``min(USD, SOL)`` is essentially always the SOL
    # figure, so the "worse of quote and mark" guard silently became "settle at the
    # quote-asset figure". The forensic reconciliation of the historical exits found
    # 22 of 26 divergent exits booked strictly below the lowest price ever observed
    # for that token, which is the signature of a value from outside the USD book.
    #
    # So the USD form is required. A quote that cannot be converted to USD is NOT
    # usable in this selection; it is recorded as present-but-unconvertible and the
    # mark is used alone, which is the honest outcome rather than a silent unit mix.
    exec_price = _quote_price_usd_of(inputs.quote)
    quote_present_quote_asset = _quote_price_of(inputs.quote)
    if exec_price is not None and exec_price > 0:
        entry_ref = max(price, exec_price) if is_entry else min(price, exec_price)
        quote_source = "EXECUTABLE_QUOTE"
    elif quote_present_quote_asset is not None:
        # present, executable, but with no USD conversion available
        entry_ref = price
        quote_source = "MARK_ONLY_QUOTE_NOT_CONVERTIBLE"
        unusable_reason = (
            "an executable quote was obtained but carried no quote-asset USD price, "
            "so it cannot be compared against a USD mark; the mark was used alone "
            "rather than mixing units")
    elif quoted is not None and quoted > 0:
        entry_ref = max(price, quoted) if is_entry else min(price, quoted)
        quote_source = "MARK_DUPLICATE"
    else:
        entry_ref = price
        quote_source = "MARK_ONLY"

    if quote_source == "MARK_ONLY_QUOTE_NOT_CONVERTIBLE" and not inputs.quote_missing_reason:
        inputs = replace(inputs, quote_missing_reason=unusable_reason)

    notional = inputs.quantity * entry_ref if inputs.quantity > 0 else inputs.notional_hint_usd
    # ---------------------------------------------------------------------------
    # AN EXECUTABLE QUOTE ALREADY CONTAINS THE SIZE IMPACT OF THIS ORDER.
    # Charging a modelled ``impact_bps`` on top of it counts the same cost twice
    # and makes a REAL quote look worse than the synthetic path it was supposed
    # to replace. So the impact term is applied only when the reference is a
    # mark we are modelling our way through, never on top of a real quote.
    # Fee and staleness still apply: the venue fee is inside the quoted OUTPUT
    # amount, but the model's fee ledger is a separate accounting record.
    quote_is_executable = quote_source == "EXECUTABLE_QUOTE"
    slip_total, base_bps, imp_bps = total_slippage_bps(
        side=side,
        notional_usd=notional,
        liquidity_usd=None if quote_is_executable else obs.liquidity_usd,
        quote_age_ms=quote_age_ms,
        model=model,
    )
    if quote_is_executable:
        imp_bps = 0.0
        slip_total = base_bps + staleness_bps(quote_age_ms, model)
    liq_src = liquidity_source(
        None if quote_is_executable else obs.liquidity_usd)
    # slippage always works against us: up when buying, down when selling
    fill_price = entry_ref * (1 + sign * slip_total / 10_000.0)

    quantity = inputs.quantity if inputs.quantity > 0 else 0.0
    if quantity > 0:
        final_notional = quantity * fill_price
        slip_usd = quantity * abs(fill_price - entry_ref)
    else:
        quantity = inputs.notional_hint_usd / fill_price
        final_notional = inputs.notional_hint_usd
        slip_usd = final_notional * (slip_total / 10_000.0)
    fee_usd = final_notional * fee_bps / 10_000.0

    # THE QUOTE GUARD CAN BE INERT, AND THAT IS WORTH RECORDING. ``entry_ref``
    # above is the worse of ``quoted`` and the observation, which is a real
    # protection only when they are independent numbers. The PAPER exit path
    # passes ``quoted_price=position.mark_price`` alongside an observation taken
    # from the SAME DexScreener mark microseconds later, so the two are the same
    # number and the "worse of" selection cannot move the fill. Detected rather
    # than assumed, so the condition is visible in every fill record.
    guard_effective = quote_source == "EXECUTABLE_QUOTE" and abs(
        float(exec_price) - price) > 1e-15

    result = FillResult(
        filled=True,
        reason="filled",
        quoted_price=quoted,
        fill_price=fill_price,
        slippage_bps=slip_total,
        slippage_usd=slip_usd,
        fee_bps=fee_bps,
        fee_usd=fee_usd,
        latency_ms=latency,
        placed_at=placed_at,
        fill_at=inputs.observed_at,
        observed_at=obs.observed_at,
        price_at_quote=entry_ref,
        data_state=data_state,
        notional_usd=final_notional,
        quantity=quantity,
        liquidity_usd=obs.liquidity_usd,
        liquidity_source=liq_src,
        execution_fidelity=(
            "PAPER/EXECUTION_AWARE" if quote_is_executable
            else ("PAPER/MARK_BASED" if liq_src == LIQUIDITY_OBSERVED
                  else "PAPER/LOW_FIDELITY_NO_LIQUIDITY")),
        detail=(
            (f"base={base_bps:.0f}bp impact=0bp(exec quote already contains it) "
             f"stale={quote_age_ms:.0f}ms liq=n/a(quote-priced)"
             if quote_is_executable else
             (f"base={base_bps:.0f}bp impact={imp_bps:.0f}bp stale={quote_age_ms:.0f}ms "
              f"liq=${obs.liquidity_usd:,.0f}"
              if liq_src == LIQUIDITY_OBSERVED else
              f"base={base_bps:.0f}bp impact={imp_bps:.0f}bp "
              f"liq=UNKNOWN -> charged configured worst case; "
              f"NOT a measured impact and NOT priced off a real pool"))
        ) + ("" if guard_effective else
             f" | quote guard INERT ({quote_source}): no independent "
             f"executable quote was available, so no worse-of-venue-price "
             f"protection applied"),
        price_kind="SIMULATED_FILL",
        # WHAT PRODUCED THE REFERENCE PRICE, and WHAT PRODUCED THE FILL. Recorded
        # separately because they can differ: the reference may be a bare mark while
        # the fill is a PAPER_FILL derived from it, and a reader must never have to
        # guess which number they are looking at.
        reference_price_provenance=(
            PriceProvenance.EXECUTABLE_QUOTE.value
            if quote_source == "EXECUTABLE_QUOTE"
            else PriceProvenance.MARK_ONLY.value),
        fill_price_provenance=PriceProvenance.PAPER_FILL.value,
        quote_guard_effective=guard_effective,
        observed_mark_price=price,
        executable_quote_price=exec_price,
        executable_quote_price_usd=(
            _quote_price_usd_of(inputs.quote)
            if _quote_price_usd_of(inputs.quote) is not None else None),
        executable_quote_kind=(
            str(getattr(inputs.quote, "kind", "VENUE_QUOTE").value
                if hasattr(getattr(inputs.quote, "kind", None), "value")
                else getattr(inputs.quote, "kind", "VENUE_QUOTE"))
            if exec_price is not None else None),
        executable_quote_missing_reason=(
            "" if exec_price is not None
            else (inputs.quote_missing_reason
                  or "no executable quote was supplied to the fill model")),
        executable_quote_at=(float(getattr(inputs.quote, "received_at", 0.0) or 0.0)
                             if exec_price is not None else None),
        executable_quote_provider=(getattr(inputs.quote, "provider", None)
                                   if exec_price is not None else None),
        executable_quote_venue=(getattr(inputs.quote, "venue", None)
                                if exec_price is not None else None),
        executable_quote_pool=(getattr(inputs.quote, "pool", None)
                               if exec_price is not None else None),
        quote_price_source=quote_source,
        token_decimals=(
            getattr(inputs.quote, "output_decimals", None)
            if exec_price is not None else None),
    )
    log(
        "PAPER_FILL",
        "%s qty=%.6g notional=$%.2f quote=%s fill=$%.3e slip=%.0fbp fee=$%.4f "
        "lat=%.0fms [%s]",
        side.upper(),
        quantity,
        final_notional,
        f"${quoted:.3e}" if quoted else "n/a",
        fill_price,
        slip_total,
        fee_usd,
        latency,
        result.detail,
    )
    return result
