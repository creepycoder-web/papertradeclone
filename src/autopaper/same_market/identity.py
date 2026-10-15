"""
Same-market verification: is the executable quote from the pool the mark
describes?

This is the experiment. Everything else in the harness exists to feed it.

WHY PRICE COMPARISON IS NOT ENOUGH
==================================

The naive check is "DexScreener price versus Jupiter price, and they are close,
so they are the same market". That inference is invalid. Two venues can price
the same token far apart, and two numbers agreeing to two decimal places while
describing different books would look like a pass. So the harness verifies
IDENTITY first and only then reports a gap.

The five checks
===============

``TOKEN_MINT_MATCH``
    The signal's mint is the token actually being priced.

``QUOTE_MINT_MATCH``
    The quote side is the intended asset. A pool quoted in USDC while the order
    would spend SOL is a different book, and the two are not directly
    comparable.

``CHAIN_MATCH``
    Solana.

``VENUE_MATCH``
    The execution provider routed through a venue consistent with the observed
    pair. Matched on canonical venue name, because DexScreener says ``pumpfun``
    and Jupiter says ``Pumpfun`` and a naive string compare would call that a
    mismatch.

``POOL_MATCH``
    The load-bearing one. Where the provider discloses a pool address, it is
    compared against the DexScreener ``pairAddress``. Equal is PASS, different
    is FAIL, and unavailable is UNKNOWN.

THE UNKNOWN RULE
================

``UNKNOWN`` never becomes ``PASS``, and ``UNKNOWN`` means REJECT. There is no
path in this module that upgrades an unverifiable identity to a pass, because
an unverifiable identity is the finding, not an obstacle to it. Reporting it
as a pass would defeat the experiment; reporting it as a fail would claim a
mismatch that was never established.
"""

from __future__ import annotations

from typing import Sequence

from .dexscreener import normalise_venue
from .jupiter import JupiterQuoteResult
from .models import (
    SOL_DECIMALS,
    WSOL_MINT,
    CheckState,
    Decision,
    DexPair,
    ExecutableQuote,
    FailureMode,
    IdentityChecks,
    SameMarketVerdict,
)
from .numeric import DecimalError, price_gap_pct, to_float

#: Quote mints accepted for a SOL-denominated test order. wSOL is the natural
#: one; USDC and USDT are accepted because a venue may quote the order in a
#: stable and the harness must be able to SAY so rather than fail silently.
ACCEPTABLE_QUOTE_MINTS: frozenset[str] = frozenset({
    WSOL_MINT,
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
})

#: Venues whose DexScreener label and Jupiter label are known to differ for the
#: same underlying pool. Pump.fun curve tokens that have graduated are the
#: classic case: DexScreener keeps ``pumpfun`` while the route uses PumpSwap.
VENUE_EQUIVALENCES: tuple[frozenset[str], ...] = (
    frozenset({"pumpfun", "pumpswap"}),
    frozenset({"raydium", "raydiumclmm"}),
    frozenset({"meteora", "meteora-dlmm", "meteora-dbc"}),
)


def _venues_equivalent(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    na, nb = normalise_venue(a), normalise_venue(b)
    if na == nb:
        return True
    return any(na in group and nb in group for group in VENUE_EQUIVALENCES)


def verify_identity(
    *,
    expected_mint: str,
    pair: DexPair | None,
    quote: ExecutableQuote | None,
    expected_quote_mint: str = WSOL_MINT,
    jupiter_result: JupiterQuoteResult | None = None,
    marked_pool_is_identified: bool = True,
) -> IdentityChecks:
    """
    Run all five checks and report each one.

    Every check is independent and reported separately, so a reader can see
    which specific link in the chain broke rather than only that the chain did.

    ``marked_pool_is_identified`` carries the DexScreener selection's verdict.
    When the marked pair was itself ambiguous, comparing the executable route
    against it would produce a ``POOL_MATCH = PASS`` for a pool nobody actually
    identified - a confident answer to the wrong question. In that case the
    comparison is downgraded to ``UNKNOWN``.
    """
    notes: list[str] = []

    # -- token mint --------------------------------------------------------
    if quote is None:
        token_mint = CheckState.UNKNOWN
        notes.append("no executable quote, so the token actually priced is unknown")
    elif quote.output_mint == expected_mint:
        token_mint = CheckState.PASS
    elif not quote.output_mint:
        token_mint = CheckState.UNKNOWN
    else:
        token_mint = CheckState.FAIL
        notes.append(
            f"the quote delivered {quote.output_mint[:8]}… but the signal's "
            f"token is {expected_mint[:8]}…"
        )

    # -- quote mint --------------------------------------------------------
    if quote is None or not quote.input_mint:
        quote_mint = CheckState.UNKNOWN
    elif quote.input_mint == expected_quote_mint:
        quote_mint = CheckState.PASS
    elif quote.input_mint in ACCEPTABLE_QUOTE_MINTS:
        quote_mint = CheckState.PASS
        notes.append(
            f"the order is quoted in {quote.input_mint[:8]}… rather than the "
            "configured input asset; accepted because it is a liquid quote asset, "
            "and the denominating asset differs from the mark's"
        )
    else:
        quote_mint = CheckState.FAIL
        notes.append(
            f"the quote is denominated in {quote.input_mint[:8]}…, which is not "
            "a quote asset this harness accepts"
        )

    # -- chain -------------------------------------------------------------
    if pair is None:
        chain = CheckState.UNKNOWN
    elif pair.chain_id == "solana":
        chain = CheckState.PASS
    else:
        chain = CheckState.FAIL
        notes.append(f"the observed pair is on {pair.chain_id!r}, not solana")

    # -- venue -------------------------------------------------------------
    venue_state = CheckState.UNKNOWN
    route_labels: tuple[str, ...] = tuple(quote.route) if quote else ()
    if pair is not None and route_labels:
        if any(_venues_equivalent(pair.dex_id, label) for label in route_labels):
            venue_state = CheckState.PASS
        else:
            venue_state = CheckState.FAIL
            notes.append(
                f"the observed pair is on {pair.dex_id!r} but the executable "
                f"route used {list(route_labels)}; the venues are not the same "
                "book"
            )
    elif pair is not None and quote is not None and not route_labels:
        notes.append(
            "the quote provider did not name a venue, so venue equivalence "
            "cannot be checked"
        )

    # -- pool --------------------------------------------------------------
    pool_state, pool_notes = verify_pool_identity(
        pair=pair,
        quote=quote,
        jupiter_result=jupiter_result,
    )
    notes.extend(pool_notes)

    # -- can the execution provider identify this market at all? ----------
    identifiable: bool | None = None
    if quote is not None:
        if quote.failure is not None:
            identifiable = False
            notes.append(
                f"the executable quote failed ({quote.failure.value}), so no "
                "execution provider identified this market"
            )
        elif quote.pool_address:
            identifiable = True
        elif route_labels:
            # A route exists but no pool key was disclosed. The provider can
            # route, but cannot be shown to use THIS pool.
            identifiable = False
            notes.append(
                f"the route ({', '.join(route_labels)}) names venues but "
                "discloses no pool address, so it cannot be shown to be the "
                "pool the mark describes"
            )
        else:
            identifiable = False
            notes.append("the quote names neither a pool nor a venue")

    return IdentityChecks(
        token_mint_match=token_mint,
        quote_mint_match=quote_mint,
        chain_match=chain,
        venue_match=venue_state,
        pool_match=pool_state,
        execution_can_identify_market=identifiable,
        notes=tuple(notes),
    )


def verify_pool_identity(
    *,
    pair: DexPair | None,
    quote: ExecutableQuote | None,
    jupiter_result: JupiterQuoteResult | None,
) -> tuple[CheckState, list[str]]:
    """
    Compare the executable pool against the observed pool.

    Three outcomes, and the distinction between the last two is the whole
    point:

    * equal addresses -> PASS
    * both known and different -> FAIL
    * either side unknown -> UNKNOWN, and UNKNOWN is not a pass
    """
    notes: list[str] = []
    if pair is None or not pair.pair_address:
        notes.append(
            "no observed pair address, so the executable pool cannot be "
            "compared against anything"
        )
        return CheckState.UNKNOWN, notes

    observed = pair.pair_address

    # A direct pump path quotes against a pool address we know from the chain.
    execution_pool = quote.pool_address if quote else None

    # Jupiter discloses pool keys per hop. Single-hop with a key is comparable;
    # multi-hop has no single pool to compare, which is an honest UNKNOWN.
    keys: tuple[str, ...] = ()
    if jupiter_result is not None:
        keys = jupiter_result.pool_keys
    hops = len(jupiter_result.route_plan) if jupiter_result else 0

    if keys and len(keys) == 1 and hops == 1:
        if keys[0] == observed:
            return CheckState.PASS, [
                f"the executable route uses pool {keys[0][:8]}…, which is the "
                "same pool DexScreener marked"
            ]
        return CheckState.FAIL, [
            f"the executable route uses pool {keys[0][:8]}… but DexScreener "
            f"marked pool {observed[:8]}…; the two prices describe DIFFERENT "
            "books, so the gap between them is not a price discrepancy"
        ]

    if hops > 1:
        notes.append(
            f"the route spans {hops} hops, so there is no single pool to compare "
            "against the marked pool; a multi-hop price is an aggregate of books "
            "and cannot be attributed to one"
        )
        return CheckState.UNKNOWN, notes

    if execution_pool:
        if execution_pool == observed:
            return CheckState.PASS, [
                f"the quote was taken against pool {execution_pool[:8]}…, the "
                "same pool DexScreener marked"
            ]
        return CheckState.FAIL, [
            f"the quote was taken against pool {execution_pool[:8]}… but "
            f"DexScreener marked {observed[:8]}…"
        ]

    notes.append(
        "the quote provider disclosed no pool address, so same-market identity "
        "is UNKNOWN rather than assumed"
    )
    return CheckState.UNKNOWN, notes


def compare_mark_to_executable(
    *,
    observed_mark: float | None,
    executable_price: float | None,
    threshold_pct: float | None,
    identity: IdentityChecks | None = None,
    failures: Sequence[FailureMode] = (),
    mark_freshness: "Freshness | None" = None,
    quote_freshness: "Freshness | None" = None,
) -> SameMarketVerdict:
    """
    The verdict.

    Order matters, and it is fail-closed:

    1. A measurement that could not be taken -> REJECT. This is checked FIRST,
       ahead of the threshold, because a run that obtained no mark and no quote
       has measured nothing - reporting that as ``INFORMATIVE`` would dress an
       absence up as a finding.
    2. Genuinely stale mark or quote -> REJECT.
    3. Unverified market identity -> REJECT.
    4. No threshold configured -> ``INFORMATIVE``, not PASS. Inventing a
       threshold to manufacture a verdict is the behaviour under test.
    5. Otherwise compare the gap to the threshold.

    An UNCONFIGURED freshness window (``Freshness.UNKNOWN``) is reported and
    noted but does not itself reject: the operator simply did not ask for a
    bound. What it must never do is read as FRESH, which is why the report
    carries the literal string ``UNKNOWN`` in the freshness field.
    """
    from .models import Freshness

    failure_list = list(failures)

    # 1. Nothing measured at all.
    if observed_mark is None or not observed_mark > 0:
        return SameMarketVerdict(
            observed_mark=observed_mark,
            executable_price=executable_price,
            gap_pct=None,
            abs_gap_pct=None,
            threshold_pct=threshold_pct,
            decision=Decision.REJECT,
            reason=(
                "REJECT: no observed mark was obtained, so there is nothing to "
                "compare the executable quote against"
            ),
        )

    if executable_price is None or not executable_price > 0:
        return SameMarketVerdict(
            observed_mark=observed_mark,
            executable_price=executable_price,
            gap_pct=None,
            abs_gap_pct=None,
            threshold_pct=threshold_pct,
            decision=Decision.REJECT,
            reason=(
                "REJECT: no executable quote was obtained. The observed mark on "
                "its own proves nothing about what could actually have been "
                "bought at that size"
            ),
        )

    # 2. Genuinely stale only.
    unconfigured: list[str] = []
    for stamp, mode in (
        (mark_freshness, "MARK"),
        (quote_freshness, "QUOTE"),
    ):
        if stamp is Freshness.STALE:
            failure_list.append(
                FailureMode.MARK_STALE if mode == "MARK"
                else FailureMode.QUOTE_STALE
            )
        elif stamp is Freshness.UNKNOWN:
            unconfigured.append(mode)

    if failure_list:
        # The threshold sentence must not assume a threshold exists. This
        # branch is the FAIL-CLOSED path, and it used to raise
        # TypeError when a measurement failed while no --max-mark-gap-pct was
        # configured - i.e. it crashed precisely when a signal could not be
        # measured, turning a reportable REJECT into a lost run. A graduated
        # pump token reaches it every time, because the (correctly) reported
        # CURVE_UNAVAILABLE is a failure.
        threshold_note = (
            f". threshold {threshold_pct:g}% is not applied"
            if threshold_pct is not None
            else ". no threshold was configured, so none is claimed"
        )
        return SameMarketVerdict(
            observed_mark=observed_mark,
            executable_price=executable_price,
            gap_pct=_safe_gap(observed_mark, executable_price),
            abs_gap_pct=_safe_abs_gap(observed_mark, executable_price),
            threshold_pct=threshold_pct,
            decision=Decision.REJECT,
            reason=(
                "REJECT on "
                + ", ".join(f.value for f in dict.fromkeys(failure_list))
                + threshold_note
            ),
        )

    # 3. Identity.
    if identity is not None and identity.market_match is not Decision.PASS:
        return SameMarketVerdict(
            observed_mark=observed_mark,
            executable_price=executable_price,
            gap_pct=_safe_gap(observed_mark, executable_price),
            abs_gap_pct=_safe_abs_gap(observed_mark, executable_price),
            threshold_pct=threshold_pct,
            decision=Decision.REJECT,
            reason=(
                "REJECT: market identity is not proven "
                f"(TOKEN_MINT_MATCH={identity.token_mint_match.value}, "
                f"QUOTE_MINT_MATCH={identity.quote_mint_match.value}, "
                f"CHAIN_MATCH={identity.chain_match.value}, "
                f"VENUE_MATCH={identity.venue_match.value}, "
                f"POOL_MATCH={identity.pool_match.value}). A gap between an "
                "unproven pair of prices is not evidence about either price"
            ),
        )

    notes = ""
    if unconfigured:
        notes = (
            f" (no freshness window was configured for {'/'.join(unconfigured)}; "
            "freshness is UNKNOWN, not assumed good)"
        )

    # 4. No threshold.
    if threshold_pct is None:
        return SameMarketVerdict(
            observed_mark=observed_mark,
            executable_price=executable_price,
            gap_pct=_safe_gap(observed_mark, executable_price),
            abs_gap_pct=_safe_abs_gap(observed_mark, executable_price),
            threshold_pct=None,
            decision=Decision.INFORMATIVE,
            reason=(
                "no --max-mark-gap-pct was configured, so the gap is reported "
                "for diagnosis and NO pass/fail verdict is claimed. Supply a "
                "threshold to get one" + notes
            ),
        )

    # 5. The comparison itself.
    try:
        gap = price_gap_pct(executable_price, observed_mark)
    except DecimalError as exc:
        return SameMarketVerdict(
            observed_mark=observed_mark,
            executable_price=executable_price,
            gap_pct=None,
            abs_gap_pct=None,
            threshold_pct=threshold_pct,
            decision=Decision.REJECT,
            reason=f"REJECT: the gap is undefined: {exc}",
        )

    absolute = abs(gap)
    passed = absolute <= threshold_pct
    direction = "worse than" if gap > 0 else "better than"
    return SameMarketVerdict(
        observed_mark=observed_mark,
        executable_price=executable_price,
        gap_pct=to_float(gap),
        abs_gap_pct=to_float(absolute),
        threshold_pct=threshold_pct,
        decision=Decision.PASS if passed else Decision.REJECT,
        reason=(
            (
                f"market identity proven and |gap| {to_float(absolute):.4f}% is "
                f"within the configured {threshold_pct:g}% threshold"
            )
            if passed else
            (
                f"market identity proven, but the executable quote is "
                f"{to_float(absolute):.4f}% {direction} the observed mark, "
                f"outside the configured {threshold_pct:g}% threshold"
            )
        ) + notes,
    )


def _safe_gap(observed: float | None, executable: float | None) -> float | None:
    if observed is None or executable is None:
        return None
    if observed <= 0:
        return None
    try:
        return to_float(price_gap_pct(executable, observed))
    except DecimalError:
        return None


def _safe_abs_gap(observed: float | None, executable: float | None) -> float | None:
    gap = _safe_gap(observed, executable)
    return None if gap is None else abs(gap)


__all__ = [
    "ACCEPTABLE_QUOTE_MINTS",
    "SOL_DECIMALS",
    "VENUE_EQUIVALENCES",
    "compare_mark_to_executable",
    "verify_identity",
    "verify_pool_identity",
]