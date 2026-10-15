"""Compact, cost-aware closed-position reporting.

WHAT WAS WRONG
==============
The closed-position summary was long, and its MFE/MAE line was both misleading
and indistinguishable from a result. It read::

    MFE: +377.9% | MAE: +0.0%   (observed mark)

Two separate defects in one line:

  * ``MAE +0.0%`` on a real loser came from seeding the excursion at the entry
    and only ever moving it OUTWARD on a mark that happened not to cross the
    entry print -- so "never went below entry on the aggregator" printed as a
    confident zero;
  * ``MFE +377.9%`` was a mark excursion, which is a different book from the one
    the position was actually exposed to and is NOT what selling the whole held
    quantity would have paid, next to a realised PnL that was negative.

WHAT THIS DOES
==============
One block per position, in the order a reader needs it, with every number
labelled by what it actually is::

    ALTERNET 🔴 -$0.090 (-26.3%) · held 4m44s
    Exit: data_invalid · no fresh executable quote for 125s
    Quote-MFE: N/A · Quote-MAE: N/A · coverage N/A

or, with a valid quote history::

    Quote-MFE: +8.4% @ 21s · Quote-MAE: -14.1% @ 46s · coverage 62%

THE FOUR CATEGORIES ARE NEVER MIXED
===================================
A result is exactly one of:

  1. ``SETTLED``     -- canonical realised PnL from the accounting settlement.
  2. ``PAPER_SIM``   -- PAPER simulated PnL from executable entry/exit quotes.
  3. ``MARK_ONLY``   -- analytics measured on the aggregator mark, which is not
                        an executable price and is never a PnL.
  4. ``UNAVAILABLE`` -- not computable from the data held, with the reason.

A quote is an EXECUTION ESTIMATE, never a fill. Nothing here presents a mark as
an achievable price, and nothing here claims an exit rule avoids losses.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from .quote_excursions import TELEMETRY_REASONS

#: The four result categories. One per position, never blended.
CATEGORY_SETTLED = "SETTLED"
CATEGORY_PAPER_SIM = "PAPER_SIM"
CATEGORY_MARK_ONLY = "MARK_ONLY"
CATEGORY_UNAVAILABLE = "UNAVAILABLE"

#: Human-readable labels, so a rendered line carries its own category.
CATEGORY_LABELS = {
    CATEGORY_SETTLED: "settled",
    CATEGORY_PAPER_SIM: "paper simulated",
    CATEGORY_MARK_ONLY: "mark-only analytics",
    CATEGORY_UNAVAILABLE: "unavailable",
}

#: Exit reasons that are NOT a strategy exit. They are named so a data failure
#: or an operator action is never mistaken for a trade decision.
_NON_STRATEGY_EXITS = frozenset({
    "data_invalid",
    "flatten",
    "stop_command",
    "emergency_stop",
    "unsellable",
    "kill",
})


def classify_result(row: Mapping[str, Any]) -> tuple[str, str]:
    """
    The single category this closed position belongs to, and why.

    The category is DERIVED from the recorded provenance, not chosen by the
    caller, so a caller cannot label a mark-only figure as settled.
    """
    basis = str(row.get("pnl_basis") or "").strip().upper()
    provenance = str(row.get("exit_price_provenance") or "").strip().upper()
    settlement = str(row.get("settlement_status") or "").strip().upper()
    mode = str(row.get("trade_mode") or row.get("mode") or "").strip().upper()

    if basis in {"SETTLED", "CANONICAL_SETTLEMENT", "ACTUAL_FILL"}:
        return CATEGORY_SETTLED, basis
    if settlement in {"SETTLED", "SETTLED_CANONICAL"}:
        return CATEGORY_SETTLED, settlement
    if provenance == "MARK_ONLY" or basis == "MARK_ONLY":
        return CATEGORY_MARK_ONLY, provenance or basis
    if not row.get("realized_pnl_usd") and row.get("net_pnl_usd") is None:
        return CATEGORY_UNAVAILABLE, "no realised PnL is recorded for this row"
    if mode == "PAPER" or basis in {"PAPER", "PAPER_SIM", "SIMULATED_FILL",
                                    "EXECUTABLE_QUOTE", "QUOTE_BASED"}:
        return CATEGORY_PAPER_SIM, basis or "PAPER"
    if basis:
        return CATEGORY_MARK_ONLY, basis
    return CATEGORY_UNAVAILABLE, "the row records no price provenance"


def compact_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    """
    Everything the compact block needs, already resolved and labelled.

    Every field is ``None`` when the data does not support it, so a renderer
    can never turn a missing value into a real number by defaulting.
    """
    symbol = _symbol(row)
    pnl = _float(row.get("realized_pnl_usd"))
    if pnl is None:
        pnl = _float(row.get("net_pnl_usd"))
    ret = _float(row.get("return_pct"))
    hold_s = _float(row.get("holding_seconds"))
    if hold_s is None:
        hold_s = _float(row.get("hold_s"))
    category, reason = classify_result(row)

    return {
        "symbol": symbol,
        "category": category,
        "category_label": CATEGORY_LABELS[category],
        "category_reason": reason,
        "pnl_usd": pnl,
        "return_pct": ret,
        "hold_s": hold_s,
        "exit_reason": _text(row.get("exit_reason")),
        "exit_reason_code": _text(row.get("exit_reason_code")
                                 or row.get("exit_trigger")),
        # Quote-based excursion. ``None`` means NOT MEASURED, never zero.
        "quote_mfe_pct": _float(row.get("quote_mfe_pct")),
        "quote_mae_pct": _float(row.get("quote_mae_pct")),
        "quote_basis": _text(row.get("quote_mfe_mae_basis")) or "NOT_MEASURED",
        "quote_unavailable_reason": _text(row.get("quote_unavailable_reason")),
        "quote_mfe_elapsed_ms": _float(row.get("quote_mfe_elapsed_ms")),
        "quote_mae_elapsed_ms": _float(row.get("quote_mae_elapsed_ms")),
        "quote_coverage_pct": _float(row.get("quote_coverage_pct")),
        "quote_sampling_interval_s": _float(row.get("quote_sampling_interval_s")),
        "quote_max_sampling_gap_ms": _float(row.get("quote_max_sampling_gap_ms")),
        "quote_mfe_source": _text(row.get("quote_mfe_source")),
        "quote_mae_source": _text(row.get("quote_mae_source")),
        # HOW MUCH EVIDENCE the pair rests on, and WHAT KIND of absence it is.
        # Both are facts about the record, carried so the renderer never has to
        # invent a reason or guess between "never quoted" and "never sampled".
        "quote_observation_count": _int(row.get("quote_observation_count")),
        "quote_rejection_count": _int(row.get("quote_rejection_count")),
        "quote_last_rejection_reason": _text(
            row.get("quote_last_rejection_reason")),
        "quote_settlement_sample": bool(row.get("quote_settlement_sample")),
        "quote_settlement_identity": _text(
            row.get("quote_settlement_identity")),
        "quote_settlement_source": _text(row.get("quote_settlement_source")),
        "quote_settlement_status": _text(row.get("quote_settlement_status")),
        "quote_telemetry_state": _text(row.get("quote_telemetry_state")),
        # Mark-only excursion, kept separate and labelled.
        "mark_mfe_pct": _float(row.get("mfe_pct")),
        "mark_mae_pct": _float(row.get("mae_pct")),
        "mark_basis": _text(row.get("mfe_basis") or "OBSERVED_MARK"),
        # Data-quality warnings that change how the numbers should be read.
        "mark_age_ms": _float(row.get("mark_age_ms")),
        "data_quality_reason": _text(row.get("data_quality_reason")),
        "route": _text(row.get("route")),
        "mint": _text(row.get("ca") or row.get("mint")),
        "trade_id": _text(row.get("trade_id")),
        "pnl_basis": _text(row.get("pnl_basis")),
    }


def render_compact(row: Mapping[str, Any]) -> str:
    """
    The compact block for ONE closed position.

    Line 1 carries the token, the outcome and the holding time so a reader can
    scan a list without opening anything. Line 2 says WHY it exited. Line 3 is
    the executable excursion, with the coverage that bounds it. Mark-only
    analytics and data-quality notes appear only when they change the reading.
    """
    s = compact_summary(row)
    lines = [f"{_headline(s)}"]

    exit_reason = s["exit_reason"] or s["exit_reason_code"] or "unknown"
    exit_line = f"Exit: {_short(exit_reason, 64)}"
    if _is_data_failure(s["exit_reason_code"] or s["exit_reason"]):
        detail = (s["data_quality_reason"]
                  or _no_fresh_quote_detail(s["mark_age_ms"]))
        if detail:
            exit_line += f" · {detail}"
    lines.append(exit_line)

    excursion = render_excursion(s)
    if excursion:
        lines.append(excursion)

    warnings = data_quality_warnings(s)
    for warning in warnings:
        lines.append(warning)

    # The category is always stated, so a mark-only or unavailable figure can
    # never be read as a settled result.
    if s["category"] != CATEGORY_SETTLED:
        lines.append(f"Result: {s['category_label']} ({s['category_reason']})")
    return "\n".join(lines)


def render_excursion(summary: Mapping[str, Any]) -> str:
    """
    The Quote-MFE/MAE line, or the explicitly-labelled Mark-only equivalent.

    ``N/A`` plus the EXACT reason when the data does not exist. Never a zero
    standing in for a missing measurement, and never a guess: the reason comes
    from the record's own telemetry state, so "never quoted", "quoted at
    settlement but never sampled" and "predates the metric" stay three
    different statements.

    A measured pair is rendered with the number of SELL-quote observations
    behind it, because ``MFE == MAE`` on a single sample is not a bound and
    reads exactly like one.
    """
    basis = str(summary.get("quote_basis") or "NOT_MEASURED").upper()
    if basis == "QUOTE_BASED":
        mfe = _signed_pct(summary.get("quote_mfe_pct"))
        mae = _signed_pct(summary.get("quote_mae_pct"))
        line = f"Quote-MFE: {mfe} @ {_age(summary.get('quote_mfe_elapsed_ms'))}"
        line += f" · Quote-MAE: {mae} @ {_age(summary.get('quote_mae_elapsed_ms'))}"
        samples = _int(summary.get("quote_observation_count"))
        if samples is not None:
            line += f" · {samples} quote sample{'' if samples == 1 else 's'}"
        coverage = summary.get("quote_coverage_pct")
        if coverage is not None:
            line += f" · coverage {float(coverage):.0f}%"
        gap = summary.get("quote_max_sampling_gap_ms")
        if gap is not None and float(gap) > 0:
            line += f" · max gap {float(gap) / 1000.0:.0f}s"
        # ONE OBSERVATION CANNOT BOUND A RANGE. The extrema of a single quote
        # are that quote, so saying "MFE -17.1% · MAE -17.1%" without this
        # invites the reader to treat it as the range the hold actually spanned.
        if samples is not None and samples <= 1:
            line += " · single observation, not a bound on the hold"
        elif not summary.get("quote_settlement_sample"):
            line += " · exit quote not sampled"
        return line

    reason = summary.get("quote_unavailable_reason") or _excursion_reason(summary)
    if summary.get("quote_rejection_count"):
        rejections = int(summary["quote_rejection_count"])
        last = summary.get("quote_last_rejection_reason")
        reason += (f" · {rejections} quote observation{'' if rejections == 1 else 's'}"
                   f" rejected{(' (' + str(last) + ')') if last else ''}")
    coverage = summary.get("quote_coverage_pct")
    if coverage is not None:
        reason += f" · coverage {float(coverage):.0f}%"
    return f"Quote-MFE: N/A · Quote-MAE: N/A · {reason}"


def _excursion_reason(summary: Mapping[str, Any]) -> str:
    """
    The reason for an absent excursion, derived from the record's own state.

    The old fallback -- "no valid same-market SELL quote was obtained for this
    position" -- was a GUESS, and it was wrong for every record that had in fact
    settled on a valid executable quote while never being sampled for one. It
    is now only ever reached when the row itself states why.
    """
    state = str(summary.get("quote_telemetry_state") or "").upper()
    if state:
        reason = TELEMETRY_REASONS.get(state)
        if reason:
            return reason
    # No state at all on the row: the caller built it without the excursion
    # block. Say THAT, rather than asserting a fact about quoting.
    return "this row carries no quote-excursion telemetry"


def render_mark_only(summary: Mapping[str, Any]) -> Optional[str]:
    """The mark pair, ONLY when it exists, and only ever under its own label."""
    mfe = summary.get("mark_mfe_pct")
    mae = summary.get("mark_mae_pct")
    if mfe is None and mae is None:
        return None
    label = str(summary.get("mark_basis") or "observed mark")
    return (f"Mark-only MFE: {_signed_pct(mfe)} · "
            f"Mark-only MAE: {_signed_pct(mae)} "
            f"({label}; NOT an executable price and NOT a PnL)")


def data_quality_warnings(summary: Mapping[str, Any]) -> list[str]:
    """
    Short warnings that change how the numbers above should be read.

    Only warnings that actually apply are emitted; a list with none of them is
    a list with no warning lines.
    """
    out = []
    mark_age = summary.get("mark_age_ms")
    quote_measured = str(summary.get("quote_basis") or "").upper() == "QUOTE_BASED"
    # A stale mark is only a warning when it was the price the exit actually
    # used. On a position whose exit was validated by a fresh executable quote,
    # a stale aggregator observation is a diagnostic, not a defect in the result.
    if mark_age is not None and float(mark_age) > 15_000 and not quote_measured:
        out.append(f"Data: mark observation was {float(mark_age) / 1000.0:.0f}s "
                   "old at the exit attempt")
    if summary.get("quote_mfe_pct") is None:
        reason = summary.get("data_quality_reason")
        if reason:
            out.append(f"Data: {_short(reason, 90)}")
    coverage = summary.get("quote_coverage_pct")
    if coverage is not None and float(coverage) < 50.0:
        out.append(f"Coverage: only {float(coverage):.0f}% of sampled ticks had "
                   "a usable quote; the excursion does not cover the whole hold")
    gap = summary.get("quote_max_sampling_gap_ms")
    if gap is not None and float(gap) > 30_000:
        out.append(f"Sampling: longest gap between usable quotes "
                   f"{float(gap) / 1000.0:.0f}s; intrasecond moves are NOT "
                   "captured by a sampled path")
    return out


# ---------------------------------------------------------------------------
# helpers -- every one refuses a missing value rather than inventing one
# ---------------------------------------------------------------------------


def _headline(summary: Mapping[str, Any]) -> str:
    """
    ``ALTERNET 🔴 -$0.0900 (-26.3%) · held 4m44s``.

    The PnL is shown with its sign and its magnitude so a minus is never
    mistaken for a formatting artefact, and ``N/A`` is printed when no PnL is
    recorded at all -- a position with no realised PnL does not get a $0.00.
    """
    symbol = summary.get("symbol") or "?"
    pnl = summary.get("pnl_usd")
    ret = summary.get("return_pct")
    if pnl is None:
        badge = "🔴 PnL unavailable"
    else:
        value = float(pnl)
        dot = "🔴" if value < 0 else ("🟢" if value > 0 else "⚪")
        sign = "+" if value >= 0 else "-"
        badge = f"{dot} {sign}${abs(value):.4f}"
        if ret is not None:
            badge += f" ({float(ret):+.1f}%)"
    hold = _duration(summary.get("hold_s"))
    tail = f" · held {hold}" if hold else ""
    if pnl is None:
        tail += " · PnL unavailable"
    return f"{symbol} {badge}{tail}"


def _short(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if not value:
        return "unknown"
    value = value.replace("\n", " ")
    return value if len(value) <= limit else value[:limit - 1] + "…"


def _symbol(row: Mapping[str, Any]) -> str:
    for key in ("symbol", "token_symbol", "name"):
        value = _text(row.get(key))
        if value:
            return value.upper()
    return "?"


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float(value: Any) -> Optional[float]:
    """A finite float or ``None``. A zero is kept; a NaN is dropped."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def _int(value: Any) -> Optional[int]:
    """A finite int or ``None``. A zero is kept -- zero observations is a fact."""
    out = _float(value)
    return None if out is None else int(out)


def _signed_pct(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):+.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _duration(seconds: Any) -> Optional[str]:
    if seconds is None:
        return None
    try:
        total = float(seconds)
    except (TypeError, ValueError):
        return None
    if total < 0:
        return None
    if total < 60:
        return f"{total:.0f}s"
    if total < 3600:
        return f"{total / 60:.0f}m{total % 60:.0f}s"
    return f"{total / 3600:.1f}h"


def _age(ms: Any) -> str:
    if ms is None:
        return "n/a"
    try:
        seconds = float(ms) / 1000.0
    except (TypeError, ValueError):
        return "n/a"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _is_data_failure(reason: Any) -> bool:
    token = str(reason or "").strip().lower()
    if not token:
        return False
    return any(name in token for name in _NON_STRATEGY_EXITS) or "invalid" in token


def _no_fresh_quote_detail(mark_age_ms: Any) -> Optional[str]:
    """Say how long the market data was unusable, when that is known."""
    if mark_age_ms is None:
        return None
    try:
        seconds = float(mark_age_ms) / 1000.0
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return f"no fresh executable quote for {seconds:.0f}s"


def render_compact_list(rows: Sequence[Mapping[str, Any]],
                        *, total: Optional[int] = None,
                        offset: int = 0) -> str:
    """The list version, with the totals stated so a page cannot fake a book."""
    from ..telegram.ui import EMPTY_CLOSED_POSITIONS
    if not rows:
        return EMPTY_CLOSED_POSITIONS
    total = len(rows) if total is None else int(total)
    out = [f"CLOSED POSITIONS  ({offset + 1}-{offset + len(rows)} of {total})",
           ""]
    for index, row in enumerate(rows):
        if index:
            out.append("")
        out.append(render_compact(row))
    out.append("")
    out.append(f"<i>{total} closed in the ledger. Detail fields (CA, trade id, "
               "route, latency, identity) stay in the per-trade detail view.</i>")
    return "\n".join(out)


__all__ = [
    "CATEGORY_SETTLED",
    "CATEGORY_PAPER_SIM",
    "CATEGORY_MARK_ONLY",
    "CATEGORY_UNAVAILABLE",
    "CATEGORY_LABELS",
    "classify_result",
    "compact_summary",
    "data_quality_warnings",
    "render_compact",
    "render_compact_list",
    "render_excursion",
    "render_mark_only",
]
