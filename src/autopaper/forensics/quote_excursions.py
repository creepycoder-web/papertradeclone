"""
Quote-based excursions: MFE and MAE measured on what a SELL could have paid.

WHY MARK-BASED MFE/MAE IS MISLEADING
====================================
The aggregator observation (``mark``) is a DIFFERENT BOOK from the one a
position is exposed to. On a pump.fun bonding curve the aggregator mark sits
materially away from the curve's own virtual-reserve spot, and it is a mark for
ONE token -- not the amount that selling the WHOLE HELD quantity would pay. A
mark therefore misstates excursions in both directions at once:

  * it can report an extreme MFE on a position whose realisable maximum was far
    lower -- so a winner looks bigger than it was; and
  * it can report MAE as ``+0.0%`` on a real loser, because the mark happened
    not to cross the entry print even though the executable book did.

Both have been observed in production. Neither is a measurement of a trade.

WHAT THIS MODULE MEASURES INSTEAD
=================================
At each observation, the HOLD path obtains a SELL quote for the exact
``remaining_token_amount_raw`` actually held, through the same pinned-market,
same-identity rules the exit uses. The number that matters is:

    net_quote_pnl_pct = (estimated_net_sell_proceeds - all_in_entry_cost)
                        / all_in_entry_cost * 100

where the quoted output is already net of the venue's fee, price impact and
any hop costs the route carries -- so those costs are NOT added a second time
here. ``all_in_entry_cost`` is the position's recorded entry cost in the same
currency. ``sol_usd_reference`` and each quote's own timestamp give the
timestamped SOL/USD conversion where the figures are in SOL.

    quote_mfe = max over valid observations of net_quote_pnl_pct
    quote_mae = min over valid observations of net_quote_pnl_pct

SAMPLED, NOT COMPLETE, AND NEVER A FILL
=======================================
This is a SAMPLED path: observations arrive on a monitoring cadence, so
intrasecond extremes are not captured. ``coverage_pct``, ``sampling_interval_s``
and ``max_sampling_gap_ms`` are reported so a reader can see how much of the
hold the extrema cover.

A SELL quote is an EXECUTION ESTIMATE. It is not a guarantee that a submitted
transaction would fill at that output amount, and nothing here presents it as
one. Reported ``quote_mfe_*``/``quote_mae_*`` are never substituted for a
canonical realised settlement: ``realized_pnl_usd`` continues to come from
``realize()`` on the actual entry and exit fills.

UNAVAILABLE MEANS UNAVAILABLE
=============================
When there is no valid sell-quote observation, the metric is NONE and the
summary says so with the reason. It is never coerced to 0.0%.

WHY IT IS MISSING IS A FACT, NOT A DEFAULT
==========================================
"There is no Quote-MFE" has five genuinely different causes, and collapsing them
under one sentence is how a reader ends up believing a measured trade was never
quoted at all. :func:`excursion_telemetry_state` separates them:

  ``MEASURED``              usable SELL-quote observations exist.
  ``NO_INTRATRADE_SAMPLES`` a settlement SELL quote WAS accepted, but no
                            intratrade observation was recorded for the hold.
  ``SAMPLES_REJECTED``      observations were taken and every one failed the
                            identity / freshness / quantity / route gate.
  ``NO_USABLE_QUOTES``      telemetry exists and recorded nothing usable.
  ``NO_QUOTE_TELEMETRY``    the record predates quote telemetry entirely: it
                            carries no telemetry field at all. Nothing was
                            measured and nothing claimed to be.
  ``MODELLED_RECORD``       the settlement was booked off a mark by the legacy
                            modelled path, so there was never an executable
                            quote to observe.

A settlement that is labelled quote-based REQUIRES an accepted SELL quote, and
that invariant is unchanged by any of this: :func:`excursion_telemetry_state`
only explains a metric that is absent, it never grants one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

#: Basis labels. Quote-based and mark-based excursions are never mixed under one
#: label, so a reader can always tell which one a figure came from.
BASIS_QUOTE = "QUOTE_BASED"
BASIS_MARK_ONLY = "MARK_ONLY"
BASIS_UNAVAILABLE = "UNAVAILABLE"

#: A quote observation is usable for the return basis only when ALL of these
#: hold. Anything else is recorded as an observation that was REJECTED, with the
#: reason, so the coverage figure is honest rather than flattering.
_USABLE_STATUS = frozenset({"VALID"})

#: The two COST-AWARE vs TRAVEL bases for a quote excursion, and why both exist.
#:
#: ``PNL_VS_ENTRY_COST`` is the cost-aware PnL the task specifies:
#: ``(net_sell_proceeds - all_in_entry_cost) / all_in_entry_cost``. It answers
#: "what would this position have realised net, including what I paid to get in
#: and what it costs to get out".
#:
#: ``TRAVEL_VS_ENTRY_QUOTE`` is a pure price-travel metric: the move measured
#: against the position-sized NET SELL quote taken AT ENTRY. It excludes the
#: entry execution cost entirely, so it is NOT a PnL -- the two differ by
#: roughly the round-trip slippage and fees. It is kept because
#: ``var/paper_hold_path.jsonl``'s pre-existing ``EXEC_REFERENCE_*`` summary
#: fields are on this basis, and silently reporting them as cost-aware PnL
#: would be the misread this module exists to prevent.
EXCURSION_BASIS_PNL = "PNL_VS_ENTRY_COST"
EXCURSION_BASIS_TRAVEL = "TRAVEL_VS_ENTRY_QUOTE"


@dataclass(frozen=True)
class QuoteExcursionPoint:
    """One SELL-quote observation, reduced to what an excursion needs."""

    at: Optional[float]
    elapsed_ms: Optional[float]
    net_proceeds: Optional[float]
    gross_proceeds: Optional[float]
    all_in_entry_cost: Optional[float]
    return_pct: Optional[float]
    net_proceeds_raw: Optional[float]
    source: str = ""
    identity: str = ""
    quote_age_ms: Optional[float] = None
    usable: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "elapsed_ms": self.elapsed_ms,
            "net_proceeds": self.net_proceeds,
            "net_proceeds_raw": self.net_proceeds_raw,
            "gross_proceeds": self.gross_proceeds,
            "all_in_entry_cost": self.all_in_entry_cost,
            "return_pct": self.return_pct,
            "source": self.source,
            "identity": self.identity,
            "quote_age_ms": self.quote_age_ms,
            "usable": self.usable,
            "reason": self.reason,
        }


def net_quote_return_pct(*, net_proceeds: Any,
                         all_in_entry_cost: Any) -> Optional[float]:
    """
    ``(estimated_net_sell_proceeds - all_in_entry_cost) / all_in_entry_cost``.

    Both in the SAME currency. ``None`` when either is missing, zero,
    non-numeric or negative -- a return computed against a cost of zero is not
    a return, and manufacturing one is exactly the invented number this module
    exists to prevent.

    Costs already inside the quote (venue fee, price impact, hop fees) are NOT
    added again here; doing so would double-count them.
    """
    try:
        proceeds = float(net_proceeds)
        cost = float(all_in_entry_cost)
    except (TypeError, ValueError):
        return None
    if proceeds != proceeds or cost != cost:
        return None
    if cost <= 0 or proceeds < 0:
        return None
    return (proceeds - cost) / cost * 100.0


def point_from_hold_record(
    record: Mapping[str, Any],
    *,
    all_in_entry_cost: Any = None,
    entry_notional_usd: Any = None,
    require_identity_pass: bool = True,
    max_quote_age_ms: Optional[float] = None,
) -> QuoteExcursionPoint:
    """
    Reduce one ``paper_hold_path`` OBSERVATION to an excursion point.

    ``all_in_entry_cost`` may be supplied by the caller (the position's recorded
    entry cost in the decision currency); when absent it falls back to the
    record's own ``entry_notional_usd`` -- which is only sound when the sell
    quote is USD-denominated, so the fallback is used for the USD path only.

    A record is usable only when the quote itself says VALID, the identity
    check passed (unless the caller relaxes it), and the quote's age is inside
    the caller's bound. Every rejection carries its reason.
    """
    status = str(record.get("quote_status") or "").strip().upper()
    identity = str(record.get("identity_check_status") or "").strip().upper()
    source = str(record.get("sell_quote_source") or "")
    raw = record.get("sell_quote_net_raw")
    gross = record.get("sell_quote_net_usd")
    usd = record.get("sell_quote_net_usd")
    return_pct = record.get("return_pct")
    age_ms = record.get("quote_age_ms")

    # The caller may declare the all-in entry cost. Otherwise the record's own
    # ``entry_notional_usd`` is used, which is only sound for the USD path -- a
    # SOL-denominated comparison must not silently reuse a USD notional.
    cost_usd = all_in_entry_cost
    if cost_usd is None:
        cost_usd = record.get("entry_notional_usd")
    net = usd
    cost = cost_usd

    reason = ""
    if status and status not in _USABLE_STATUS:
        reason = f"quote_status={status}"
    elif raw is None and net is None:
        reason = "no sell quote was obtained for this observation"
    elif status not in _USABLE_STATUS:
        reason = f"quote_status={status or 'MISSING'}"
    elif require_identity_pass and identity and identity != "PASS":
        reason = f"identity_check_status={identity}"
    elif max_quote_age_ms is not None and age_ms is not None:
        try:
            if float(age_ms) > float(max_quote_age_ms):
                reason = (f"quote_age_ms={float(age_ms):.0f} exceeds "
                          f"{float(max_quote_age_ms):.0f}")
        except (TypeError, ValueError):
            reason = "quote_age_ms is not a number"

    computed = net_quote_return_pct(net_proceeds=net, all_in_entry_cost=cost)
    if computed is None and reason == "":
        reason = ("net sell proceeds or all-in entry cost is unavailable or "
                  "non-positive")
    # A REJECTED observation carries no return at all. Computing one anyway
    # would let an unavailable or stale quote contribute to an excursion, which
    # is exactly the fabricated number this module exists to prevent -- and it
    # would be invisible, because ``usable`` alone is easy to overlook.
    pct = None if reason else computed

    return QuoteExcursionPoint(
        at=_as_epoch(record.get("quote_timestamp_utc")
                    or record.get("observed_at_utc")),
        elapsed_ms=_as_float(record.get("elapsed_ms")),
        net_proceeds=_as_float(net),
        net_proceeds_raw=_as_float(raw),
        gross_proceeds=_as_float(gross),
        all_in_entry_cost=_as_float(cost),
        return_pct=pct,
        source=source,
        identity=identity,
        quote_age_ms=_as_float(age_ms),
        usable=(reason == "" and pct is not None),
        reason=reason,
    )


def excursions(
    points: Iterable[QuoteExcursionPoint],
    *,
    travel_basis: bool = False,
    entry_reference_raw: Any = None,
) -> dict[str, Any]:
    """
    Quote MFE / MAE from the valid points, with honest coverage and gaps.

    ``travel_basis=True`` recomputes the percentages relative to the ENTRY SELL
    QUOTE (the position-sized estimated liquidation value at entry) instead of
    the all-in entry COST, and labels the result
    ``TRAVEL_VS_ENTRY_QUOTE`` so it can never be read as a PnL.
    ``entry_reference_raw`` supplies that entry reference explicitly when the
    caller has it recorded (as ``paper_hold_path``'s SUMMARY does); otherwise
    the earliest usable observation is used, which is the best available
    estimate and is labelled as such.

    Returns an UNAVAILABLE basis with a reason when no point is usable -- never
    a zero. Coverage, sampling interval and the longest gap are computed from
    the timestamps that ARE present, and stay ``None`` when they cannot be.
    """
    if travel_basis:
        return _excursions(points, basis=EXCURSION_BASIS_TRAVEL,
                           entry_reference_raw=entry_reference_raw)
    return _excursions(points, basis=EXCURSION_BASIS_PNL)


def _excursions(
    points: Iterable[QuoteExcursionPoint],
    *,
    basis: str,
    entry_reference_raw: Any = None,
) -> dict[str, Any]:
    usable = [p for p in points if p.usable and p.return_pct is not None]
    reference = None
    if basis == EXCURSION_BASIS_TRAVEL:
        # Travel is measured against the ENTRY SELL QUOTE: the position-sized
        # estimated liquidation value at entry. Every later observation's return
        # is rebased onto it, so entry slippage and fees cancel and the number
        # is a pure move -- explicitly NOT a PnL.
        #
        # The declared reference is preferred when the caller recorded it (as
        # ``var/paper_hold_path``'s SUMMARY does), because it is the value the
        # position was actually opened at rather than an estimate from the
        # earliest surviving observation.
        declared = _as_float(entry_reference_raw)
        if declared not in (None, 0):
            base = declared
        else:
            entries = [p for p in points if p.usable and p.at is not None]
            reference = min(entries, key=lambda p: p.at) if entries else None
            if reference is None or reference.net_proceeds in (None, 0):
                return {
                    "basis": BASIS_UNAVAILABLE,
                    "excursion_basis": basis,
                    "unavailable_reason": "no usable ENTRY sell quote to rebase "
                                         "the travel basis on",
                    "quote_mfe_pct": None, "quote_mae_pct": None,
                }
            base = (reference.net_proceeds_raw
                    if reference.net_proceeds_raw not in (None, 0)
                    else reference.net_proceeds)
        # The TRAVEL basis stays in ONE currency end to end: the raw quote-asset
        # amount, which is what the declared reference is denominated in.
        # Mixing raw and USD across observations of one trade is a unit error.
        usable = [
            p if p is reference else _rebased(p, base)
            for p in usable
        ]
    if not usable:
        return {
            "basis": BASIS_UNAVAILABLE,
            "excursion_basis": basis,
            "quote_mfe_pct": None,
            "quote_mae_pct": None,
            "mfe_at": None,
            "mae_at": None,
            "mfe_elapsed_ms": None,
            "mae_elapsed_ms": None,
            "mfe_source": None,
            "mae_source": None,
            "valid_quotes": 0,
            "observations_total": None,
            "coverage_pct": None,
            "sampling_interval_s": None,
            "max_sampling_gap_ms": None,
            "unavailable_reason": "no valid SELL quote observation exists for "
                                 "this position",
        }

    mfe_point = max(usable, key=lambda p: p.return_pct)
    mae_point = min(usable, key=lambda p: p.return_pct)

    total = len(list(points))
    stamps = sorted(p.at for p in usable if p.at is not None)
    gaps = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
    max_gap_ms = (max(gaps) * 1000.0) if gaps else None
    # A cadence estimate: the MEDIAN gap is used rather than the mean, because
    # a single long outage must not be averaged into "the interval".
    interval_s = None
    if gaps:
        ordered = sorted(gaps)
        mid = len(ordered) // 2
        interval_s = (ordered[mid] if len(ordered) % 2
                      else (ordered[mid - 1] + ordered[mid]) / 2.0)

    coverage_pct = (100.0 * len(usable) / total) if total else None

    return {
        "basis": BASIS_QUOTE,
        "excursion_basis": basis,
        "quote_mfe_pct": round(mfe_point.return_pct, 6),
        "quote_mae_pct": round(mae_point.return_pct, 6),
        "mfe_at": mfe_point.at,
        "mae_at": mae_point.at,
        "mfe_elapsed_ms": mfe_point.elapsed_ms,
        "mae_elapsed_ms": mae_point.elapsed_ms,
        "mfe_source": mfe_point.source,
        "mae_source": mae_point.source,
        "valid_quotes": len(usable),
        "observations_total": total,
        "coverage_pct": (None if coverage_pct is None
                         else round(coverage_pct, 3)),
        "sampling_interval_s": (None if interval_s is None
                                else round(interval_s, 3)),
        "max_sampling_gap_ms": (None if max_gap_ms is None
                                else round(max_gap_ms, 1)),
        "unavailable_reason": None,
    }


def _rebased(point: QuoteExcursionPoint, base: float) -> QuoteExcursionPoint:
    """The same observation, with its return measured against ``base``."""
    value = (point.net_proceeds_raw
             if point.net_proceeds_raw not in (None, 0)
             else point.net_proceeds)
    if value is None or not base:
        return point
    pct = (value - base) / base * 100.0
    return QuoteExcursionPoint(
        at=point.at, elapsed_ms=point.elapsed_ms,
        net_proceeds=point.net_proceeds,
        net_proceeds_raw=point.net_proceeds_raw,
        gross_proceeds=point.gross_proceeds,
        all_in_entry_cost=point.all_in_entry_cost, return_pct=pct,
        source=point.source, identity=point.identity,
        quote_age_ms=point.quote_age_ms, usable=True, reason=point.reason)


def mark_only_excursions(
    *, mark_mfe_pct: Any = None, mark_mae_pct: Any = None, observations: Any = None
) -> dict[str, Any]:
    """
    The LEGACY mark-based pair, explicitly labelled.

    Returned under ``BASIS_MARK_ONLY`` so it can never be printed as an
    executable excursion. ``None`` values stay ``None``: a mark excursion with
    no observations is not a 0.0% result.
    """
    return {
        "basis": BASIS_MARK_ONLY,
        "quote_mfe_pct": None,
        "quote_mae_pct": None,
        "mark_mfe_pct": _as_float(mark_mfe_pct),
        "mark_mae_pct": _as_float(mark_mae_pct),
        "mark_observations": _as_int(observations),
        "unavailable_reason": None,
    }


# ---------------------------------------------------------------------------
# WHY AN EXCURSION IS ABSENT
# ---------------------------------------------------------------------------

#: The telemetry states. ``MEASURED`` is the only one that produces a number.
TELEMETRY_MEASURED = "MEASURED"
TELEMETRY_NO_INTRATRADE_SAMPLES = "NO_INTRATRADE_SAMPLES"
TELEMETRY_SAMPLES_REJECTED = "SAMPLES_REJECTED"
TELEMETRY_NO_USABLE_QUOTES = "NO_USABLE_QUOTES"
TELEMETRY_ABSENT = "NO_QUOTE_TELEMETRY"
TELEMETRY_MODELLED = "MODELLED_RECORD"

#: How the settlement itself was priced, as far as the record states it.
SETTLEMENT_ACCEPTED = "ACCEPTED"
SETTLEMENT_NOT_ACCEPTED = "NOT_ACCEPTED"
SETTLEMENT_MODELLED = "MODELLED"
SETTLEMENT_UNKNOWN = "UNKNOWN"

#: One sentence per state, in the order a reader needs them. Each names the
#: FACT, and never implies a quote was or was not executable when the record does
#: not say.
TELEMETRY_REASONS = {
    TELEMETRY_NO_INTRATRADE_SAMPLES: (
        "a valid settlement SELL quote was accepted, but no intratrade "
        "SELL-quote observation was recorded for this hold, so the excursion "
        "is not measured"),
    TELEMETRY_SAMPLES_REJECTED: (
        "SELL-quote observations were taken while this position was open and "
        "every one was rejected (identity, freshness, quantity or route); "
        "none was usable"),
    TELEMETRY_NO_USABLE_QUOTES: (
        "quote-excursion telemetry is present for this position but recorded "
        "no usable same-market SELL quote"),
    TELEMETRY_ABSENT: (
        "this record carries no quote-excursion telemetry at all -- it was "
        "written before the metric existed, or by a path that never sampled "
        "quotes, so no excursion was ever measured for it"),
    TELEMETRY_MODELLED: (
        "this settlement was booked off an observed mark by the modelled "
        "backtest path, not an executable quote, so there was never an "
        "executable quote to observe"),
}

#: Pnl bases that mean the record is a legacy modelled backtest, never a
#: quote-based settlement.
_MODELLED_BASES = frozenset({
    "V2_FLOOR_BETA_MODELED_BACKTEST", "MODELED_BACKTEST_BASIS", "MARK_ONLY",
    "MODELED",
})


def settlement_quote_state(record: Mapping[str, Any]) -> str:
    """
    Whether the record states that a valid settlement SELL quote was accepted.

    Read from what the record actually persisted, never inferred from the P&L:
    a positive P&L is not evidence of an executable quote, and neither is its
    absence evidence of a failed one. ``UNKNOWN`` when the record does not say.
    """
    basis = str(record.get("pnl_basis") or "").strip().upper()
    if basis in _MODELLED_BASES:
        return SETTLEMENT_MODELLED
    status = str(record.get("sell_execution_status") or "").strip().upper()
    if status == "SETTLED" or str(
            record.get("settlement_status") or "").strip().upper() in {
                "SETTLED", "SETTLED_CANONICAL"}:
        return SETTLEMENT_ACCEPTED
    exit_basis = str(record.get("exit_basis") or "").strip().upper()
    if exit_basis and exit_basis not in _MODELLED_BASES:
        return SETTLEMENT_ACCEPTED
    if status == "NO_EXECUTABLE_SELL":
        return SETTLEMENT_NOT_ACCEPTED
    if basis:
        return SETTLEMENT_ACCEPTED if basis.endswith(
            ("QUOTE_SIMULATED_FILL", "FILL_VS_FILL")) else SETTLEMENT_MODELLED
    return SETTLEMENT_UNKNOWN


def excursion_telemetry_state(
    record: Mapping[str, Any],
    *,
    basis: Any = None,
    observation_count: Any = None,
    telemetry_persisted: Optional[bool] = None,
    rejection_count: Any = None,
    settlement: Any = None,
) -> str:
    """
    Why this record does (or does) have a sampled quote excursion.

    ``telemetry_persisted`` is the decisive fact for historical records and must
    be passed explicitly by the reader: it is true only when the record carries a
    quote-telemetry field at all. A record with no telemetry field was never
    measured; a record WITH one that says ``NOT_MEASURED`` was measured and
    found nothing. Reporting the second as the first hides a real sampling
    failure, and reporting the first as the second invents a fault.
    """
    if basis is None:
        basis = record.get("quote_basis") or record.get("quote_mfe_mae_basis")
    if str(basis or "").strip().upper() == BASIS_QUOTE:
        return TELEMETRY_MEASURED
    if telemetry_persisted is False:
        return TELEMETRY_ABSENT
    if settlement is None:
        settlement = settlement_quote_state(record)
    if settlement == SETTLEMENT_MODELLED:
        return TELEMETRY_MODELLED
    if settlement == SETTLEMENT_ACCEPTED:
        return TELEMETRY_NO_INTRATRADE_SAMPLES
    rejected = _as_int(rejection_count) or 0
    if rejected > 0:
        return TELEMETRY_SAMPLES_REJECTED
    if observation_count is None:
        observation_count = record.get("quote_observation_count")
    if (_as_int(observation_count) or 0) == 0:
        return TELEMETRY_NO_USABLE_QUOTES
    return TELEMETRY_NO_USABLE_QUOTES


def excursion_unavailable_reason(record: Mapping[str, Any], **kwargs: Any) -> str:
    """The one-sentence reason for an absent excursion, or ``""`` when measured."""
    state = excursion_telemetry_state(record, **kwargs)
    if state == TELEMETRY_MEASURED:
        return ""
    explicit = record.get("quote_unavailable_reason")
    if explicit and state in (TELEMETRY_NO_USABLE_QUOTES, TELEMETRY_SAMPLES_REJECTED):
        # The collector recorded a specific, per-observation reason. It is more
        # precise than the generic sentence, so it wins -- but only for the
        # states where a reason was actually recorded.
        return str(explicit)
    return TELEMETRY_REASONS.get(state, TELEMETRY_REASONS[TELEMETRY_NO_USABLE_QUOTES])


# ---------------------------------------------------------------------------
# presentation
# ---------------------------------------------------------------------------


def format_excursion_line(result: Mapping[str, Any]) -> str:
    """
    One compact line for the closed-position report.

    ``Quote-MFE: +8.4% @ 21s · Quote-MAE: -14.1% @ 46s`` when the data exists,
    and ``Quote-MFE: N/A · Quote-MAE: N/A`` plus the reason when it does not.
    Never a bare 0.0%.
    """
    basis = str(result.get("basis") or BASIS_UNAVAILABLE)
    if basis == BASIS_MARK_ONLY:
        mfe = _signed_pct(result.get("mark_mfe_pct"))
        mae = _signed_pct(result.get("mark_mae_pct"))
        return f"Mark-only MFE: {mfe} · Mark-only MAE: {mae}  (not executable)"
    if basis != BASIS_QUOTE:
        reason = str(result.get("unavailable_reason") or "no valid quote path")
        return f"Quote-MFE: N/A · Quote-MAE: N/A  ({reason})"

    mfe = _signed_pct(result.get("quote_mfe_pct"))
    mae = _signed_pct(result.get("quote_mae_pct"))
    mfe_at = _duration_s(result.get("mfe_elapsed_ms"))
    mae_at = _duration_s(result.get("mae_elapsed_ms"))
    line = f"Quote-MFE: {mfe} @ {mfe_at} · Quote-MAE: {mae} @ {mae_at}"
    coverage = result.get("coverage_pct")
    if coverage is not None:
        line += f" · coverage {float(coverage):.0f}%"
    return line


def _signed_pct(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):+.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _duration_s(ms: Any) -> str:
    if ms is None:
        return "n/a"
    try:
        seconds = float(ms) / 1000.0
    except (TypeError, ValueError):
        return "n/a"
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds / 60.0:.1f}m"


# ---------------------------------------------------------------------------
# small coercion helpers: a missing value is NEVER silently zero
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def _as_int(value: Any) -> Optional[int]:
    out = _as_float(value)
    return None if out is None else int(out)


def _as_epoch(value: Any) -> Optional[float]:
    """
    A wall-clock epoch from a UTC string or a numeric stamp, or ``None``.

    ``None`` when the source gave nothing usable: a fabricated timestamp here
    would move an excursion to the wrong moment without anyone seeing it.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        out = float(value)
        return out if out > 0 else None
    text = str(value).strip()
    if not text:
        return None
    from datetime import datetime, timezone
    candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


__all__ = [
    "BASIS_QUOTE",
    "BASIS_MARK_ONLY",
    "BASIS_UNAVAILABLE",
    "EXCURSION_BASIS_PNL",
    "EXCURSION_BASIS_TRAVEL",
    "SETTLEMENT_ACCEPTED",
    "SETTLEMENT_MODELLED",
    "SETTLEMENT_NOT_ACCEPTED",
    "SETTLEMENT_UNKNOWN",
    "TELEMETRY_ABSENT",
    "TELEMETRY_MEASURED",
    "TELEMETRY_MODELLED",
    "TELEMETRY_NO_INTRATRADE_SAMPLES",
    "TELEMETRY_NO_USABLE_QUOTES",
    "TELEMETRY_REASONS",
    "TELEMETRY_SAMPLES_REJECTED",
    "QuoteExcursionPoint",
    "excursion_telemetry_state",
    "excursion_unavailable_reason",
    "excursions",
    "format_excursion_line",
    "mark_only_excursions",
    "net_quote_return_pct",
    "point_from_hold_record",
    "settlement_quote_state",
]
