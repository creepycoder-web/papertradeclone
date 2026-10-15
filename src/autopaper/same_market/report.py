"""
Human-readable rendering of a :class:`~autopaper.same_market.pipeline.SignalReport`.

The rendering is deliberately blunt about one thing: the three values are
never on the same line. ``OBSERVED MARK``, ``EXECUTABLE QUOTE`` and
``ACTUAL FILL: NOT AVAILABLE`` each get their own labelled section, and the
report ends with the non-execution notice. A reader skimming the output cannot
come away thinking a fill happened.

Credential hygiene: nothing here reads environment variables, and every value
rendered comes from a report field rather than from raw config. The Telegram
session, the API hash and the bot token have no path into this output.
"""

from __future__ import annotations

from typing import Any

from ..clock import iso
from .models import ACTUAL_FILL_AVAILABLE, Decision
from .numeric import quantise, to_float

#: Display precision for USD-denominated prices. Enough to show a 0.1% gap on a
#: sub-cent token without turning into noise.
PRICE_PLACES = 12


def _ms(value: float | None, *, approximate: bool = False) -> str:
    if value is None:
        return "UNKNOWN"
    suffix = "~" if approximate else ""
    if value >= 1000.0:
        return f"{value / 1000.0:.2f}s"
    return f"{value:.0f}ms{suffix}"


def _price(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${quantise(value, PRICE_PLACES)}"


def _usd(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def _pct(value: float | None, signed: bool = True) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}%" if signed else f"{value:.4f}%"


def _mask(address: str | None, head: int = 6, tail: int = 4) -> str:
    if not address:
        return "n/a"
    if len(address) <= head + tail + 1:
        return address
    return f"{address[:head]}…{address[-tail:]}"


def render_report(report: Any) -> str:
    """Render one signal's report to text."""
    out: list[str] = []
    add = out.append
    if getattr(report, "direct", None) is not None:
        body = render_direct(report.direct)
        out.append(body)
        add = out.append

    add("=" * 72)
    add("SAME-MARKET QUOTE TEST  (READ-ONLY / NO EXECUTION)")
    add("=" * 72)

    # -- signal ------------------------------------------------------------
    add("")
    add("NEW SIGNAL")
    add(f"  Channel            : @{report.channel.lstrip('@')}")
    add(f"  Message ID         : {report.message_id if report.message_id is not None else 'UNKNOWN'}")
    add(f"  Telegram time      : {iso(report.telegram_timestamp) or 'UNKNOWN'}")
    add(f"  Local receive      : {iso(report.local_receive_timestamp) or 'UNKNOWN'}")
    add(f"  Telegram latency   : {_ms(report.receive_latency_ms, approximate=True)}"
        f"   (two clocks + 1s Telegram resolution)")
    if report.latency.ordering_violations:
        add("  !! stage ordering violations: " + "; ".join(report.latency.ordering_violations))

    # -- token -------------------------------------------------------------
    add("")
    add("TOKEN")
    token = report.token
    if token.ambiguous:
        add(f"  Extraction         : AMBIGUOUS - {len(token.candidates)} distinct mints")
        for m in token.candidates[:8]:
            add(f"                      {_mask(m)}")
        add("  STOPPED: refusing to guess which token the signal refers to.")
    elif not token.mint:
        add("  Extraction         : NO MINT FOUND")
        add("  STOPPED: no token to price.")
    else:
        add(f"  Symbol             : {token.symbol or 'n/a'}")
        add(f"  Mint               : {token.mint}")
        add(f"  Extraction         : OK ({len(token.candidates)} distinct mint(s))")
    if token.venue_hint:
        add(f"  Signal venue hint  : {token.venue_hint}")
    for w in token.warnings:
        add(_field_note("note", w))

    # -- test order --------------------------------------------------------
    add("")
    add("TEST BUY SIZE (HYPOTHETICAL)")
    add(f"  TEST BUY SIZE      : ${report.test_order_usd:.6f} USD equivalent")
    add(f"  Size in lamports   : "
        f"{report.test_order_lamports if report.test_order_lamports is not None else 'UNKNOWN'}")
    add(f"  SOL reference      : "
        f"{_usd(report.sol_reference_usd) if report.sol_reference_usd is not None else 'NOT SUPPLIED'}"
        f"   (logged so the notional is reproducible)")
    add("  NO MONEY IS SPENT. NO TRANSACTION IS SUBMITTED.")

    # -- dexscreener -------------------------------------------------------
    add("")
    add("DEXSCREENER  (OBSERVATION ONLY - NOT AN EXECUTION PRICE)")
    sel = report.selection
    add(f"  Candidates seen    : {sel.considered}")
    if report.pairs:
        for p in report.pairs[:10]:
            liq = _usd(to_float(p.liquidity_usd))
            # DexScreener omits "dexId" on some pairs, and the probe passes the
            # missing key through as None. Formatting that with a width spec
            # raises TypeError, which killed the read loop on the first such
            # pair - the report is how the harness shows its work, so it must
            # render a missing venue name rather than abort the measurement.
            dex = p.dex_id if p.dex_id else "unknown"
            add(f"    - {_mask(p.pair_address)}  dex={dex:<14} "
                f"{_price(to_float(p.price_usd))}  liq={liq}")
        if len(report.pairs) > 10:
            add(f"    ... and {len(report.pairs) - 10} more")
    mark = report.observed_mark
    add(f"  Selected pair      : {_mask(sel.pair.pair_address) if sel.pair else 'n/a'}")
    add(f"  Selection rule     : {sel.rule}")
    add(f"  Ambiguous          : {'YES' if sel.ambiguous else 'no'}")
    add("")
    add(f"  OBSERVED_MARK      : {_price(to_float(mark.value_usd))}")
    add(f"    source           : {mark.source}")
    add(f"    pair             : {_mask(mark.pair_address)}")
    add(f"    dex              : {mark.dex_id}")
    add(f"    timestamp        : {iso(mark.timestamp)}")
    add(f"    request latency  : {_ms(mark.request_latency_ms)}")
    add(f"    age              : {_ms(report.latency.receive_to_mark_ms)} since receive")
    add(f"    freshness        : {report.mark_freshness}")
    add("")
    add("  PAIR SELECTION REASON")
    for line in _wrap(sel.reason, 68):
        add(f"    {line}")

    # -- identity ----------------------------------------------------------
    add("")
    add("IDENTITY CHECK")
    ident = report.identity
    add(f"  TOKEN_MINT_MATCH   = {ident.token_mint_match.value}")
    add(f"  QUOTE_MINT_MATCH   = {ident.quote_mint_match.value}")
    add(f"  CHAIN_MATCH        = {ident.chain_match.value}")
    add(f"  VENUE_MATCH        = {ident.venue_match.value}")
    add(f"  POOL_MATCH         = {ident.pool_match.value}")
    add(f"  MARKET_MATCH       = {ident.market_match.value}")
    add(f"  Provider identifies market : "
        f"{ident.execution_can_identify_market}")
    for note in ident.notes:
        add(_bullet(note, 66))

    # -- execution market --------------------------------------------------
    add("")
    add("EXECUTION MARKET")
    add(f"  Type               : {report.market_kind.value}")
    if report.pool_state is not None:
        ps = report.pool_state
        add(f"  Curve / pool       : {_mask(ps.address)}")
        add(f"  Program            : {_mask(ps.program_id)}")
        add(f"  Identity verified  : {ps.identity_verified}")
        add(f"  State timestamp    : {iso(ps.timestamp)}")
        add(f"  State freshness    : {report.state_freshness}")
        for k, v in (ps.detail or {}).items():
            if k in ("account_len",):
                continue
            add(f"    {k:<18}: {v}")
    else:
        add("  Curve / pool       : (no direct chain state read)")
        add(f"  State freshness    : {report.state_freshness}")
    if report.jupiter is not None:
        add(f"  Jupiter route      : {', '.join(report.jupiter.venue_labels) or 'n/a'}")
        for hop in report.jupiter.route_plan:
            add(f"    - {hop.label or '?':<22} pool={_mask(hop.amm_key)}")

    # -- quote -------------------------------------------------------------
    add("")
    add("HYPOTHETICAL BUY QUOTE  (READ-ONLY - NOT A FILL)")
    q = report.quote
    if q.failure is not None:
        add(f"  QUOTE UNAVAILABLE  : {q.failure.value}")
        if q.detail:
            for line in _wrap(q.detail, 66):
                add(f"    {line}")
    else:
        add(f"  Input mint         : {q.input_mint}")
        add(f"  Output mint        : {q.output_mint}")
        add(f"  Input amount       : {q.input_amount_raw} "
            f"(decimals {q.input_decimals})")
        add(f"  Output amount      : {q.output_amount_raw} "
            f"(decimals {q.output_decimals})")
        add(f"  EXECUTABLE_QUOTE   : {_price(to_float(q.effective_price))}")
        add(f"    (effective = input / output, full precision, no pre-rounding)")
        add(f"  Price impact       : {_pct(to_float(q.price_impact_pct))}")
        add(f"  Fee                : "
            f"{q.fee_bps if q.fee_bps is not None else 'n/a'} bps"
            f"{'' if q.fee_amount is None else f'  ({q.fee_amount:.9f} SOL)'}")
        add(f"  Venue              : {q.venue or 'n/a'}")
        add(f"  Pool / curve       : {_mask(q.pool_address)}")
        add(f"  Quote timestamp    : {iso(q.quote_timestamp)}")
        add(f"  Quote latency      : {_ms(q.quote_latency_ms)}")
        add(f"  Quote freshness    : {report.quote_freshness}")
        if q.detail:
            add("  note:")
            for line in _wrap(q.detail, 66):
                add(f"    {line}")

    # -- comparison --------------------------------------------------------
    add("")
    add("MARK VS EXECUTABLE")
    v = report.verdict
    add(f"  OBSERVED_MARK      : {_price(to_float(v.observed_mark))}")
    add(f"  EXECUTABLE_QUOTE   : {_price(to_float(v.executable_price))}")
    add(f"  Gap (signed)       : {_pct(to_float(v.gap_pct))}")
    add(f"  Gap (absolute)     : {_pct(to_float(v.abs_gap_pct), signed=False)}")
    add(f"  Threshold          : "
        f"{_pct(v.threshold_pct, signed=False) if v.threshold_pct is not None else 'NOT CONFIGURED'}")
    add(f"  Decision           : {v.decision.value}")
    for line in _wrap(v.reason, 68):
        add(f"    {line}")

    # -- latency -----------------------------------------------------------
    add("")
    add("LATENCY")
    lat = report.latency
    add(f"  Telegram -> receive        : {_ms(lat.telegram_to_receive_ms, approximate=True)}")
    add(f"  Receive  -> mark           : {_ms(lat.receive_to_mark_ms)}")
    add(f"  Mark     -> pool state     : {_ms(lat.mark_to_state_ms)}")
    add(f"  State    -> quote          : {_ms(lat.state_to_quote_ms)}")
    add(f"  Quote    -> decision       : {_ms(lat.quote_to_decision_ms)}")
    add(f"  TOTAL signal -> decision   : {_ms(lat.total_signal_to_decision_ms)}")
    add("  detail:")
    add(f"    parsing                  : {_ms(lat.parsing_ms)}")
    add(f"    dexscreener              : {_ms(lat.dexscreener_ms)}")
    add(f"    solana rpc               : {_ms(lat.rpc_ms)}")
    add(f"    pool state               : {_ms(lat.pool_state_ms)}")
    add(f"    quote                    : {_ms(lat.quote_ms)}")
    add(f"    jupiter                  : {_ms(lat.jupiter_ms)}")
    add("  (durations are monotonic-clock differences; wall clock is used only")
    add("   for the Telegram span, which crosses two machines)")

    # -- failures ----------------------------------------------------------
    if report.failures:
        add("")
        add("FAILURE MODES")
        for f in report.failures:
            add(f"  {f.value}")

    # -- the three values, one last time -----------------------------------
    add("")
    add("THE THREE VALUES")
    add(f"  1. OBSERVED_MARK     : {_price(to_float(report.observed_mark.value_usd))}"
        f"   (DexScreener, pair {_mask(report.observed_mark.pair_address)})")
    add(f"  2. EXECUTABLE_QUOTE  : {_price(to_float(report.quote.effective_price))}"
        f"   (read-only, {report.quote.provider})")
    add(f"  3. ACTUAL_FILL       : NOT AVAILABLE"
        f"   (available={ACTUAL_FILL_AVAILABLE})")
    add("  A quote is not a fill. Nothing was signed and nothing was sent.")

    add("")
    add("EXECUTION")
    add("  NOT EXECUTED")
    add("  NO TRANSACTION BUILT, SIGNED OR BROADCAST")
    add("-" * 72)
    return "\n".join(out)


def render_direct(direct: Any) -> str:
    """
    Render the four-way measurement.

    Prints each source under its own heading, keeps the four labels visibly
    distinct, and prints the cross-check and identity blocks. The final four
    lines are a standing statement that no transaction was produced - a
    read-only harness that does not say so on every page invites the reader to
    assume a fill happened.
    """
    out: list[str] = []
    add = out.append
    add("")
    add("#" * 72)
    add("# FOUR-WAY EXECUTABLE QUOTE VERIFICATION (READ-ONLY)")
    add("#" * 72)

    classification = direct.classification
    if classification is not None:
        add("")
        add("EXECUTION MARKET")
        add(f"Type          : {classification.market.value}")
        add(f"Curve         : {_mask(classification.curve_address, 10, 6) or 'none'}")
        add(f"Pool          : {_mask(classification.pool_address, 10, 6) or 'none'}")
        add(f"Graduated     : {classification.graduated}")
        add(f"Evidence      : {classification.evidence}")
        for note in classification.notes:
            add(_bullet(note))

    add("")
    add("OBSERVED MARK  (DexScreener - an observation, NOT a fill)")
    add(f"DexScreener   : {_price(direct.mark.price_usd_per_token)}"
        f"   via {_mask(direct.mark.pool_or_curve, 10, 6) or 'n/a'}"
        f" ({direct.mark.venue or 'n/a'})")
    add(f"Mark freshness: {direct.mark_freshness.value}")

    for title, quote in (
        ("DIRECT PUMP  (pump.fun bonding curve, priced from on-chain state)",
         direct.pump_direct),
        ("DIRECT PUMPSWAP  (AMM pool, priced from on-chain state)",
         direct.pumpswap_direct),
    ):
        add("")
        add(title)
        if quote is None:
            add("  not attempted for this signal")
            continue
        if not quote.available:
            add(f"  {quote.provider:<18} : UNKNOWN  [{quote.failure.value if quote.failure else '-'}]")
            for line in _wrap(quote.missing_reason or "", 64):
                add(f"    {line}")
            continue
        detail = quote.quote.as_dict()
        snap = quote.snapshot
        add(f"  Curve / pool   : {_mask(quote.curve_address or quote.pool_address, 10, 6)}")
        add(f"  Input          : {detail['input_amount_raw']} raw "
            f"({detail['input_decimals']} decimals) = "
            f"{detail['input_amount_decimal']:.9f} SOL"
            f"   (a SOL amount; the USD equivalent is the logged reference "
            f"multiplied out below)")
        add(f"  Output         : {detail['output_amount_raw']} raw "
            f"({detail['output_decimals']} decimals) = "
            f"{detail['output_amount_decimal']:.9f} tokens")
        add(f"  Effective      : SOL/token {to_float(quote.quote.effective_price):.12g}")
        add(f"  Effective USD  : {_price(direct_mark_price(direct, quote))}   "
            f"(at the logged {_usd(direct.sol_reference_usd)}/SOL)")
        add(f"  Price impact   : {_pct(to_float(quote.quote.price_impact_pct))}")
        add(f"  Fee            : {quote.quote.fee_bps} bps "
            f"(lp {quote.quote.fees.lp_fee_bps} / protocol "
            f"{quote.quote.fees.protocol_fee_bps} / creator "
            f"{quote.quote.fees.creator_fee_bps})")
        add(f"  State slot     : {snap.slot if snap else 'UNKNOWN'}")
        add(f"  State age      : {_ms(snap.state_age_ms() if snap else None)}")
        add(f"  State freshness: {direct.state_freshness.value}")
        add(f"  Quote latency  : {_ms(quote.quote_latency_ms)}")
        for note in quote.quote.notes:
            add(_bullet(note))

    add("")
    add("JUPITER  (independent aggregator cross-check)")
    jp = direct.jupiter
    if jp.available:
        add(f"Output         : {jp.raw_output} raw ({jp.output_decimals} decimals)")
        add(f"Effective      : SOL/token {to_float(jp.price_sol_per_token):.12g}")
        add(f"Effective USD  : {_price(jp.price_usd_per_token)}")
        add(f"Route          : {', '.join(jp.route) or 'unnamed'}")
        add(f"Pool keys      : {', '.join(_mask(k, 10, 6) for k in [jp.pool_or_curve] if k) or 'none disclosed'}")
        add(f"Quote latency  : {_ms(jp.quote_latency_ms)}")
    else:
        add(f"  UNKNOWN  [{jp.missing_reason}]")

    add("")
    add(direct.cross.render())
    add("")
    add(direct.identity.render())
    for note in direct.identity.notes:
        add(_bullet(note))

    add("")
    add("LATENCY")
    lat = direct.latency.as_dict()
    for key in (
        "telegram_to_receive_ms", "receive_to_dexscreener_ms",
        "dexscreener_to_rpc_ms", "rpc_to_pump_direct_ms",
        "rpc_to_pumpswap_direct_ms", "jupiter_quote_ms",
        "identity_comparison_ms", "total_signal_to_final_decision_ms",
    ):
        if key in lat:
            add(f"  {key:<36} {_ms(lat[key])}")

    if direct.notes:
        add("")
        add("NOTES")
        for note in direct.notes:
            add(_bullet(note))

    add("")
    add("-" * 72)
    add("NO TRANSACTION BUILT")
    add("NO TRANSACTION SIGNED")
    add("NO TRANSACTION BROADCAST")
    add("ACTUAL_FILL = NOT AVAILABLE (this harness never executes)")
    return "\n".join(out)


def direct_mark_price(direct: Any, quote: Any) -> float | None:
    """The direct quote's USD price at the logged SOL reference."""
    if quote.quote is None or direct.sol_reference_usd is None:
        return None
    return to_float(quote.quote.effective_price) * direct.sol_reference_usd


def _wrap(text: str, width: int) -> list[str]:
    """Simple greedy wrap. Preserves the ability to read the whole reason."""
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _field_note(label: str, text: str, width: int = 52) -> str:
    """A labelled, wrapped note line, aligned under its label."""
    lines = _wrap(text, width)
    if not lines:
        return f"  {label:<19}:"
    pad = " " * (len(label) + 5)
    out = [f"  {label:<19}: {lines[0]}"]
    out.extend(f"{pad}{line}" for line in lines[1:])
    return "\n".join(out)


def _bullet(text: str, width: int = 52) -> str:
    """
    Render one note as an indented bullet.

    The marker goes on the FIRST line only. Repeating it on every wrapped line
    makes a single sentence look like a list of separate findings, which
    overstates how many things went wrong.
    """
    lines = _wrap(text, width)
    if not lines:
        return ""
    out = [f"    - {lines[0]}"]
    out.extend(f"      {line}" for line in lines[1:])
    return "\n".join(out)


def render_summary(reports: list[Any]) -> str:
    """A compact roll-up across several signals."""
    out: list[str] = []
    add = out.append
    add("=" * 72)
    add("SAME-MARKET QUOTE TEST - SUMMARY")
    add("=" * 72)
    add(f"  signals processed : {len(reports)}")
    counts: dict[str, int] = {}
    for r in reports:
        counts[r.verdict.decision.value] = counts.get(r.verdict.decision.value, 0) + 1
    for key in (Decision.PASS.value, Decision.REJECT.value,
                Decision.INFORMATIVE.value):
        add(f"  {key:<18}: {counts.get(key, 0)}")

    proven = [r for r in reports if r.identity.pool_match.value == "PASS"]
    add(f"  same-market proven : {len(proven)}")
    unproven = [r for r in reports if r.identity.pool_match.value == "UNKNOWN"]
    add(f"  identity UNKNOWN   : {len(unproven)}")
    mismatched = [r for r in reports if r.identity.pool_match.value == "FAIL"]
    add(f"  identity MISMATCH  : {len(mismatched)}")

    gaps = [
        (abs(r.verdict.gap_pct), r)
        for r in reports if r.verdict.gap_pct is not None
    ]
    if gaps:
        gaps.sort(key=lambda pair: pair[0])
        add("")
        add("  gap distribution (absolute, by signal):")
        for gap, r in gaps:
            add(f"    {_gap_label(gap):>10}  {r.token.symbol or '(no symbol)':<12}"
                f" {r.market_kind.value:<22} pool_match={r.identity.pool_match.value}")
        add(f"    median {_gap_label(_median([g for g, _ in gaps]))}"
            f"   max {_gap_label(gaps[-1][0])}")

    latency = [r.latency.total_signal_to_decision_ms for r in reports
               if r.latency.total_signal_to_decision_ms is not None]
    if latency:
        latency.sort()
        add("")
        add(f"  total signal->decision ms: min {_ms(latency[0])}"
            f"  median {_ms(_median(latency))}  max {_ms(latency[-1])}")

    if unproven:
        add("")
        add("  CASES WHERE EXACT MARKET IDENTITY COULD NOT BE PROVEN:")
        for r in unproven:
            add(f"    - {_mask(r.observed_mark.pair_address)}"
                f"  reason: {r.identity.pool_match.value}"
                f" / {r.verdict.decision.value}")

    no_quote = [r for r in reports if r.quote.failure is not None]
    if no_quote:
        add("")
        add("  CASES WHERE NO QUOTE WAS AVAILABLE:")
        for r in no_quote:
            add(f"    - {_mask(r.token.mint)}  {r.quote.failure.value}")

    add("")
    add("  NO TRANSACTION WAS BROADCAST. NO FUNDS MOVED.")
    add("-" * 72)
    return "\n".join(out)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _gap_label(value: float) -> str:
    return f"{value:.3f}%" if value < 10 else f"{value:.2f}%"


__all__ = ["PRICE_PLACES", "render_direct", "render_report", "render_summary"]