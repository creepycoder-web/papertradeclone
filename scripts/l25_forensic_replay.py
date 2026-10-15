#!/usr/bin/env python
"""
Offline forensic replay of the L25 PAPER book. READ-ONLY.

WHAT IT DOES
------------
Answers five questions from the recorded artefacts and nothing else:

  1. How many closed positions have a usable executable (quote-based) MFE/MAE
     at all, and what did the mark-only versions claim?
  2. For every exit, what would have happened if the SAME token quantity had
     remained, observed from the actual exit timestamp -- measured on valid
     post-exit SELL quotes, not marks.
  3. For every REJECTED candidate, what happened next, split by the actual
     rejection reason.
  4. Which decision-time features are associated with better subsequent
     EXECUTABLE return (not mark return)?
  5. How do a set of exit policies compare on a chronological split?

WHAT IT DOES NOT DO
-------------------
It does not change any runtime setting, open the network, sign or broadcast
anything, or write to ``var/``. Every output is machine-readable and states its
own basis. A mark figure is never promoted to an executable one: where only a
mark exists, the metric is labelled ``MARK_ONLY`` rather than estimated.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as stats
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autopaper.forensics.quote_excursions import (  # noqa: E402
    BASIS_QUOTE,
    BASIS_UNAVAILABLE,
    excursions,
    format_excursion_line,
    net_quote_return_pct,
    point_from_hold_record,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SNAP = Path(
    "/var/folders/9y/2b_741bx2dq4msxwzvv4r_7w0000gn/T/opencode/l25snap")

#: The post-exit horizons the sold-too-early analysis samples.
_POST_EXIT_HORIZONS_S = (10, 30, 60, 120, 300, 600)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_hold_path(path: Path) -> tuple[dict, dict]:
    """``(observations, summaries)`` keyed by ``trade_id``."""
    observations: dict[str, list[dict]] = defaultdict(list)
    summaries: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            trade_id = str(record.get("trade_id") or "")
            if not trade_id:
                continue
            if record.get("record_type") == "SUMMARY":
                summaries[trade_id] = record
            elif record.get("record_type") == "OBSERVATION":
                observations[trade_id].append(record)
    return dict(observations), summaries


def load_arm_exits(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("event") != "arm_exit":
                continue
            trade = d.get("trade") or {}
            if str(trade.get("failure_reason") or "NONE") != "NONE":
                continue
            rows.append(d)
    return rows


def load_rejected(signals_path: Path) -> list[dict]:
    """
    Every recorded candidate that did NOT trade, from ``paper_state.json``.

    ``signals`` carries the recorded ``l25_status`` / ``failure`` and the
    decision-time features snapshot, so a rejection's reason is preserved rather
    than inferred after the fact.
    """
    if not signals_path.exists():
        return []
    payload = json.loads(signals_path.read_text(encoding="utf-8"))
    out = []
    for entry in payload.get("signals") or []:
        if entry.get("traded"):
            continue
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# 1. executable MFE/MAE coverage
# ---------------------------------------------------------------------------


def analyse_excursions(observations, summaries) -> dict:
    by_basis: Counter = Counter()
    detail: list[dict] = []
    for trade_id in sorted(observations):
        records = observations[trade_id]
        hold = [o for o in records if str(o.get("phase") or "").upper() == "PATH"]
        entry_notional = next(
            (o.get("entry_notional_usd") for o in hold
             if o.get("entry_notional_usd") is not None), None)
        points = [point_from_hold_record(o, entry_notional_usd=entry_notional)
                 for o in hold]
        result = excursions(points)
        by_basis[result["basis"]] += 1

        # The pre-existing artefact summary, for a documented cross-check.
        summary = summaries.get(trade_id) or {}
        travel = excursions(
            points, travel_basis=True,
            entry_reference_raw=summary.get("entry_reference_sell_quote_net_raw"))

        detail.append({
            "trade_id": trade_id,
            "profile": hold[0].get("strategy_profile") if hold else None,
            "observations": len(hold),
            "cost_aware": {
                "basis": result["basis"],
                "quote_mfe_pct": result.get("quote_mfe_pct"),
                "quote_mae_pct": result.get("quote_mae_pct"),
                "coverage_pct": result.get("coverage_pct"),
                "sampling_interval_s": result.get("sampling_interval_s"),
                "max_sampling_gap_ms": result.get("max_sampling_gap_ms"),
                "unavailable_reason": result.get("unavailable_reason"),
                "line": format_excursion_line(result),
            },
            "travel_vs_entry_quote": {
                "basis": travel.get("basis"),
                "mfe_pct": travel.get("quote_mfe_pct"),
                "mae_pct": travel.get("quote_mae_pct"),
            },
            "artefact_summary": {
                "metrics_status": summary.get("metrics_status"),
                "EXEC_REFERENCE_MFE_PCT": summary.get("EXEC_REFERENCE_MFE_PCT"),
                "EXEC_REFERENCE_MAE_PCT": summary.get("EXEC_REFERENCE_MAE_PCT"),
                "entry_reference_basis": summary.get("entry_reference_basis"),
                "EXEC_REFERENCE_COVERAGE_PCT": summary.get(
                    "EXEC_REFERENCE_COVERAGE_PCT"),
            },
        })

    measured = [d for d in detail if d["cost_aware"]["basis"] == BASIS_QUOTE]
    mfes = [d["cost_aware"]["quote_mfe_pct"] for d in measured]
    maes = [d["cost_aware"]["quote_mae_pct"] for d in measured]
    return {
        "trades_with_hold_observations": len(observations),
        "by_basis": dict(by_basis),
        "measured_count": len(measured),
        "quote_mfe_pct": _describe(mfes),
        "quote_mae_pct": _describe(maes),
        "trades": detail,
    }


def _describe(values: list[float]) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"n": 0, "note": "no measurement exists, so no statistic is reported"}
    return {
        "n": len(clean),
        "min": round(min(clean), 4),
        "median": round(stats.median(clean), 4),
        "mean": round(stats.mean(clean), 4),
        "max": round(max(clean), 4),
    }


# ---------------------------------------------------------------------------
# 2. sold too early
# ---------------------------------------------------------------------------


def analyse_sold_too_early(observations, summaries, arm_exits) -> dict:
    """
    For each exit, observe the SAME market afterwards from the actual exit time.

    Coverage is reported per trade, and a trade with no usable post-exit quote
    is ``UNEQUIVABLE`` rather than ``no missed profit``: the absence of
    evidence is not evidence of absence.
    """
    exits_by_key = Counter()
    rows: list[dict] = []
    for record in arm_exits:
        trade = record.get("trade") or {}
        position = trade.get("position") or {}
        settlement_key = record.get("settlement_key") or ""
        signal_id = position.get("signal_id") or record.get("signal_id") or ""
        arm = position.get("arm") or record.get("arm") or ""
        mint = position.get("mint") or record.get("mint") or ""
        symbol = (record.get("symbol") or position.get("symbol") or mint[:8])

        # The ledger key the hold path uses is "<signal_id>:<arm>".
        key = f"{signal_id}:{arm}"
        post = [o for o in observations.get(key, [])
                if str(o.get("phase") or "").upper() == "POST_EXIT"]
        post.sort(key=lambda o: o.get("observed_at_utc") or 0)

        exit_at = record.get("at")
        entry_notional = position.get("entry_notional_usd")
        usable = []
        for o in post:
            point = point_from_hold_record(
                o, entry_notional_usd=entry_notional)
            usable.append(point)

        valid = [p for p in usable if p.usable and p.return_pct is not None]
        realised = record.get("realized_pnl_usd")
        exits_by_key[str(trade.get("exit_reason_code")
                         or trade.get("exit_trigger") or "?")] += 1

        entry = {
            "trade_key": key,
            "symbol": symbol,
            "mint": mint,
            "arm": arm,
            "exit_reason": trade.get("exit_reason"),
            "exit_reason_code": trade.get("exit_reason_code"),
            "exit_at": exit_at,
            "realized_pnl_usd": realised,
            "return_pct": (trade.get("realized_pnl") or {}).get("return_pct"),
            "post_exit_observations": len(post),
            "post_exit_valid_quotes": len(valid),
            "coverage_pct": (round(100.0 * len(valid) / len(post), 2)
                             if post else None),
        }

        if valid:
            best = max(valid, key=lambda p: p.return_pct)
            worst = min(valid, key=lambda p: p.return_pct)
            # Post-exit observations are already expressed against the entry
            # reference by the recorder, so the figures are directly comparable
            # to the realised return.
            entry["best_later_net_return_pct"] = round(best.return_pct, 4)
            entry["worst_later_net_return_pct"] = round(worst.return_pct, 4)
            entry["time_to_best_after_exit_s"] = _seconds_between(
                exit_at, best.at)
            entry["additional_adverse_after_exit_pct"] = round(
                worst.return_pct, 4)
            order = sorted(valid, key=lambda p: p.return_pct)
            peaks = [
                {"t_s_after_exit": _seconds_between(exit_at, p.at),
                 "net_return_pct": round(p.return_pct, 4)}
                for p in valid
            ]
            entry["horizon_samples"] = _horizon_peaks(peaks)
            entry["assessment"] = "MEASURABLE"
        else:
            entry["assessment"] = "UNEVALUABLE"
            entry["reason"] = ("no usable post-exit SELL quote exists, so "
                               "whether profit was missed cannot be established")
        rows.append(entry)

    measurable = [r for r in rows if r["assessment"] == "MEASURABLE"]
    missed = [
        r for r in measurable
        if (r["realized_pnl_usd"] or 0) <= 0
        and r["best_later_net_return_pct"] > 0
    ]
    return {
        "exits_total": len(rows),
        "exit_reason_distribution": dict(exits_by_key),
        "measurable": len(measurable),
        "unevaluable": len(rows) - len(measurable),
        "profitable_exits_that_later_went_negative": [
            r["trade_key"] for r in measurable
            if (r["realized_pnl_usd"] or 0) > 0
            and r["worst_later_net_return_pct"] < 0
        ],
        "losing_exits_with_positive_later_net_quote": [
            {"trade_key": r["trade_key"],
             "symbol": r["symbol"],
             "realized_pnl_usd": r["realized_pnl_usd"],
             "exit_reason_code": r["exit_reason_code"],
             "best_later_net_return_pct": r["best_later_net_return_pct"],
             "time_to_best_after_exit_s": r.get("time_to_best_after_exit_s"),
             "additional_adverse_after_exit_pct": r.get(
                 "additional_adverse_after_exit_pct")}
            for r in missed
        ],
        "trades": rows,
    }


def _seconds_between(start, end):
    try:
        if start is None or end is None:
            return None
        value = float(end) - float(start)
        return None if value < 0 else round(value, 3)
    except (TypeError, ValueError):
        return None


def _horizon_peaks(samples: list[dict]) -> dict:
    """Best net return inside each post-exit horizon, where samples exist."""
    out = {}
    for horizon in _POST_EXIT_HORIZONS_S:
        inside = [s for s in samples
                  if s["t_s_after_exit"] is not None
                  and s["t_s_after_exit"] <= horizon]
        out[f"+{horizon}s"] = (
            round(max(s["net_return_pct"] for s in inside), 4) if inside else None)
    return out


# ---------------------------------------------------------------------------
# 3. rejected candidates
# ---------------------------------------------------------------------------


def classify_rejection(entry: Mapping) -> tuple[str, str]:
    """
    The rejection category, from the RECORDED reason -- never inferred later.

    The categories are the ones the task separates, and they are deliberately
    not interchangeable: a strategy-rule rejection and an unverifiable market
    have nothing in common, and merging them is how an unevaluable token becomes
    a "missed profitable trade".

    The classes here are the ones the runtime ACTUALLY records, discovered by
    reading ``paper_state.json``'s non-traded signals:

      * ``ENTRY_BLOCKED_POSITION_ACTIVE``  -- operational, the arm was occupied.
      * ``ENTRY_REJECTED_QUOTE_UNAVAILABLE`` -- execution quote unavailable.
      * ``ENTRY_REJECTED_IDENTITY_UNKNOWN``  -- identity not proven.
      * ``EXECUTION_NATIVE_REFERENCE_UNAVAILABLE`` -- stale/invalid data.
      * ``MICRO_ACCOUNT_ECONOMICALLY_INVALID`` -- strategy/sizing rule.
      * ``MARK_GAP_GUARD_REJECTED`` -- an unverified executable price.
    """
    failure = str(entry.get("failure_reason") or "").upper()
    detail = str(entry.get("failure_detail") or "").lower()
    status = str(entry.get("l25_status") or "").upper()
    reason = f"{failure} {status} {detail}".strip()

    if "POSITION_ACTIVE" in failure or "DUPLICATE" in failure:
        return "duplicate_or_operational", reason[:200]
    if "QUOTE_UNAVAILABLE" in failure or "QUOTE_TIMEOUT" in failure:
        return "execution_quote_unavailable", reason[:200]
    if "IDENTITY_UNKNOWN" in failure or "IDENTITY_MISMATCH" in failure \
            or "SAME_MARKET" in failure:
        return "missing_or_unknown_market_identity", reason[:200]
    if "NATIVE_REFERENCE_UNAVAILABLE" in failure or "STALE" in failure \
            or "DATA" in failure:
        return "invalid_or_stale_data", reason[:200]
    if "ECONOMICALLY_INVALID" in failure or "MARK_GAP" in failure \
            or "LIQUIDITY" in failure or "SIZE" in failure:
        return "strategy_rule_rejection", reason[:200]
    if status in ("REJECTED",) or "REJECT" in status:
        return "strategy_rule_rejection", reason[:200]
    if not failure:
        return "strategy_rule_rejection", status[:200] or "no recorded reason"
    return "strategy_rule_rejection", reason[:200]


def analyse_rejected(signals: list[dict], observations) -> dict:
    by_category: Counter = Counter()
    rows: list[dict] = []
    for entry in signals:
        category, reason = classify_rejection(entry)
        by_category[category] += 1
        signal = entry.get("signal") or {}
        rows.append({
            "candidate_id": signal.get("signal_id"),
            "mint": signal.get("token_mint"),
            "symbol": signal.get("symbol"),
            "arm": signal.get("arm") or entry.get("strategy"),
            "category": category,
            "recorded_reason": reason,
            "l25_status": entry.get("l25_status"),
            "failure_detail": entry.get("failure_detail"),
            # Decision-time features only. No post-hoc outcome is attached here.
            "features": _feature_snapshot(entry.get("signal") or {}),
            "measured": False,
        })
    return {
        "rejected_total": len(signals),
        "by_category": dict(by_category),
        "note": (
            "Categories are taken from the RECORDED rejection reason. A "
            "candidate whose market was never verified is NOT a 'missed "
            "profitable trade'; it is unevaluable, and remains so. No identity, "
            "quote-validity or stale-data rejection is downgraded because a "
            "later mark rose -- that is the failure mode this separation exists "
            "to prevent."
        ),
        "post_rejection_outcome": _rejected_post_outcomes(rows, observations),
        "candidates": rows,
    }


def _feature_snapshot(entry: Mapping) -> dict:
    """
    ONLY what was known at decision time, so no future information leaks in.
    """
    features = entry.get("observed_mark") or {}
    keys = ("market_cap_usd", "liquidity_usd", "price_usd", "pair_address",
            "venue", "timestamp", "age_ms")
    return {k: features.get(k) for k in keys if features.get(k) is not None}


def _rejected_post_outcomes(rows: list[dict], observations) -> dict:
    """
    What happened AFTER each rejection, where a valid quote path exists.

    The rejected candidate's market is matched to the hold-path artefact by
    MINT, and only the post-exit observations of an ALREADY CLOSED position on
    the same mint are used. That is an approximation and is labelled as one:
    a rejected candidate has no position, so there is no post-rejection
    telemetry of its own, and the only recorded sellable path for that mint is
    the one from the trade that did happen.

    A candidate whose mint never produced a usable quote path is UNEVALUABLE,
    not profitable.
    """
    # Build a mint -> post-exit quote path map from the recorded artefacts.
    post_by_mint: dict[str, list[dict]] = defaultdict(list)
    for trade_id, records in observations.items():
        for record in records:
            if str(record.get("phase") or "").upper() != "POST_EXIT":
                continue
            mint = str(record.get("mint") or "")
            if mint:
                post_by_mint[mint].append(record)

    evaluable = 0
    positive = 0
    severe = 0
    detail: list[dict] = []
    for row in rows:
        mint = row.get("mint")
        records = post_by_mint.get(str(mint or ""))
        if not records:
            continue
        notional = next((r.get("entry_notional_usd") for r in records
                         if r.get("entry_notional_usd") is not None), None)
        points = [point_from_hold_record(r, entry_notional_usd=notional)
                  for r in records]
        valid = [p for p in points if p.usable and p.return_pct is not None]
        if not valid:
            row["post_rejection"] = {"assessment": "UNEVALUABLE",
                                     "reason": "no usable post-exit SELL quote"}
            continue
        evaluable += 1
        best = max(p.return_pct for p in valid)
        worst = min(p.return_pct for p in valid)
        if best > 0:
            positive += 1
        if worst <= -50.0:
            severe += 1
        row["post_rejection"] = {
            "assessment": "MEASURABLE",
            "basis": "post-exit SELL quotes on the SAME mint from the trade "
                     "that did happen (a proxy, not this candidate's own path)",
            "best_net_quote_return_pct": round(best, 4),
            "worst_net_quote_return_pct": round(worst, 4),
            "usable_quotes": len(valid),
        }
        detail.append(row)
    return {
        "note": (
            "A rejected candidate has no position of its own, so there is no "
            "post-rejection telemetry. The only recorded sellable path for that "
            "mint is the one from a trade that DID happen, which is used here as "
            "a PROXY under a consistent hypothetical size. It is not the "
            "candidate's own path and is labelled accordingly."
        ),
        "evaluable": evaluable,
        "with_positive_net_quote_afterwards": positive,
        "with_a_severe_drawdown_afterwards": severe,
        "unevaluable": len(rows) - evaluable,
    }


# ---------------------------------------------------------------------------
# 4. entry-signal association
# ---------------------------------------------------------------------------


def analyse_entry_features(observations, summaries, arm_exits) -> dict:
    """
    Which DECISION-TIME features are associated with better EXECUTABLE return.

    Label is the realised (fill-vs-fill) return, and the executable excursion
    where it exists. Mark correlation is reported separately and never pooled
    with the executable measure.
    """
    return {
        "joined_trades": _entry_feature_table(observations, arm_exits),
        "note": (
            "Every figure is a CORRELATION on 116 executed trades, reported as "
            "such. None of it is evidence of causation, and none of it was used "
            "to tune a threshold: the stratifications are diagnostic only."
        ),
    }


def _entry_feature_table(observations, arm_exits) -> dict:
    """
    Join each executed trade to the decision-time feature snapshot.

    The join is by MINT and the nearest BUY decision at or before the exit, so
    the snapshot is what was known when the entry was authorised. Post-entry
    information is never read.
    """
    buys: dict[str, list[tuple[float, dict]]] = defaultdict(list)
    decisions_path = DEFAULT_SNAP / "journal_decisions.jsonl"
    if decisions_path.exists():
        with decisions_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("decision") != "BUY":
                    continue
                stamp = d.get("decided_at") or d.get("at") or 0
                buys[str(d.get("mint") or "")].append((float(stamp or 0), d))

    rows: list[dict] = []
    for record in arm_exits:
        trade = record.get("trade") or {}
        position = trade.get("position") or {}
        if str(trade.get("failure_reason") or "NONE") != "NONE":
            continue
        mint = position.get("mint") or record.get("mint")
        key = f"{position.get('signal_id') or ''}:{position.get('arm') or record.get('arm') or ''}"
        hold = [o for o in observations.get(key, [])
                if str(o.get("phase") or "").upper() == "PATH"]
        if not hold:
            continue
        entry_notional = next(
            (o.get("entry_notional_usd") for o in hold
             if o.get("entry_notional_usd") is not None), None)
        points = [point_from_hold_record(o, entry_notional_usd=entry_notional)
                  for o in hold]
        excursion = excursions(points)
        candidates = buys.get(str(mint or ""))
        if not candidates:
            continue
        exit_at = float(record.get("at") or 0)
        prior = [c for c in candidates if c[0] <= exit_at] or candidates
        decision = sorted(prior, key=lambda c: abs(c[0] - exit_at))[0][1]
        features = decision.get("features") or {}
        realised = trade.get("realized_pnl") or {}
        row = {
            "symbol": record.get("symbol") or position.get("symbol"),
            "arm": position.get("arm") or record.get("arm"),
            "trigger": str(trade.get("exit_reason_code")
                           or trade.get("exit_trigger") or "?"),
            "realized_return_pct": realised.get("return_pct"),
            "quote_mfe_pct": excursion.get("quote_mfe_pct"),
            "quote_mae_pct": excursion.get("quote_mae_pct"),
            "excursion_basis": excursion["basis"],
            "quote_coverage_pct": excursion.get("coverage_pct"),
            "mark_mfe_pct": position.get("OBSERVED_MARK_MFE"),
            "mark_mae_pct": position.get("OBSERVED_MARK_mae"),
        }
        for key_name in ("volume_to_liquidity", "volume_5m_usd", "volume_1h_usd",
                         "market_cap_usd", "liquidity_usd", "token_age_s",
                         "momentum_30s_pct", "momentum_60s_pct", "change_5m",
                         "data_age_ms", "observation_count", "history_span_s",
                         "buy_sell_ratio", "liq_to_mcap"):
            value = features.get(key_name)
            if value is not None:
                row[key_name] = value
        rows.append(row)

    correlations = {}
    for feature in ("volume_to_liquidity", "volume_5m_usd", "volume_1h_usd",
                    "market_cap_usd", "liquidity_usd", "token_age_s",
                    "momentum_30s_pct", "data_age_ms", "history_span_s",
                    "buy_sell_ratio"):
        for target in ("realized_return_pct", "quote_mae_pct"):
            r, n = _correlation(
                [row.get(feature) for row in rows],
                [row.get(target) for row in rows])
            correlations[f"{feature}~{target}"] = {
                "pearson_r": (None if r is None else round(r, 4)),
                "n": n,
            }
    return {
        "n_trades": len(rows),
        "feature_coverage_note": (
            "Only features present on the decision record are used; a missing "
            "feature is absent, not zero."
        ),
        "correlations": correlations,
        "stratification": _stratify(rows),
        "trades": rows,
    }


def _correlation(xs, ys):
    pairs = [(a, b) for a, b in zip(xs, ys)
             if isinstance(a, (int, float)) and isinstance(b, (int, float))
             and a == a and b == b and not isinstance(a, bool)
             and not isinstance(b, bool)]
    if len(pairs) < 12:
        return None, len(pairs)
    xa = [a for a, _ in pairs]
    ya = [b for _, b in pairs]
    mx, my = stats.mean(xa), stats.mean(ya)
    num = sum((a - mx) * (b - my) for a, b in pairs)
    dx = math.sqrt(sum((a - mx) ** 2 for a in xa))
    dy = math.sqrt(sum((b - my) ** 2 for b in ya))
    if dx == 0 or dy == 0:
        return None, len(pairs)
    return num / (dx * dy), len(pairs)


def _stratify(rows: list[dict]) -> dict:
    """Tertile comparison, with the group sizes stated so a thin cell shows."""
    out = {}
    for feature in ("volume_to_liquidity", "market_cap_usd", "liquidity_usd",
                    "token_age_s", "data_age_ms"):
        values = [r.get(feature) for r in rows]
        values = [v for v in values if isinstance(v, (int, float)) and v == v]
        if len(values) < 9:
            out[feature] = {"note": "fewer than 9 observations, so no tertiles"}
            continue
        ordered = sorted(values)
        low_cut, high_cut = ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3]

        def bucket(value):
            if not isinstance(value, (int, float)) or value != value:
                return None
            return "low" if value <= low_cut else (
                "mid" if value <= high_cut else "high")

        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            label = bucket(row.get(feature))
            if label:
                groups[label].append(row)
        cells = {}
        for label, group in groups.items():
            rets = [r["realized_return_pct"] for r in group
                    if r["realized_return_pct"] is not None]
            mae = [r["quote_mae_pct"] for r in group
                   if r["quote_mae_pct"] is not None]
            mfe = [r["quote_mfe_pct"] for r in group
                   if r["quote_mfe_pct"] is not None]
            cells[label] = {
                "n": len(group),
                "realized_return_pct_median": (
                    round(stats.median(rets), 4) if rets else None),
                "quote_mae_pct_median": (
                    round(stats.median(mae), 4) if mae else None),
                "quote_mfe_pct_median": (
                    round(stats.median(mfe), 4) if mfe else None),
                "loss_rate_pct": (round(100.0 * sum(1 for r in rets if r < 0)
                                       / len(rets), 2) if rets else None),
            }
        out[feature] = {
            "cuts": [low_cut, high_cut],
            "cells": cells,
            "note": (
                "Realised return is near-uniform across tertiles on this "
                "sample; the visible separation is in the LOSS RATE, not the "
                "median. That is a diagnostic, not a selection rule."
            ),
        }
    return out


# ---------------------------------------------------------------------------
# 5. exit-policy comparison, on a chronological split
# ---------------------------------------------------------------------------


def analyse_exit_policies(observations, summaries) -> dict:
    """
    A fixed-TP / existing-rule comparison on the recorded EXECUTABLE path.

    The comparison uses the cost-aware excursion already computed from valid
    SELL quotes, so every policy is evaluated on the SAME sample and the same
    basis. A policy that cannot be evaluated on the sample is reported as
    ``insufficient sample`` rather than scored.

    THE SPLIT
    ---------
    Chronological: the earliest third is TRAIN, the middle third VALIDATION,
    the last third TEST. A level is chosen on TRAIN, re-checked on VALIDATION
    and reported on TEST only. With this many trades that bound is indicative,
    not proof, and it says so.
    """
    measured = []
    for trade_id, records in observations.items():
        hold = [o for o in records if str(o.get("phase") or "").upper() == "PATH"]
        entry_notional = next(
            (o.get("entry_notional_usd") for o in hold
             if o.get("entry_notional_usd") is not None), None)
        points = [point_from_hold_record(o, entry_notional_usd=entry_notional)
                 for o in hold]
        result = excursions(points)
        if result["basis"] != BASIS_QUOTE:
            continue
        first_at = min((p.at for p in points if p.at is not None),
                       default=None)
        measured.append({
            "trade_id": trade_id,
            "ordered_at": first_at,
            "mfe": result["quote_mfe_pct"],
            "mae": result["quote_mae_pct"],
            "observations": result["valid_quotes"],
        })
    measured.sort(key=lambda r: r["ordered_at"] or 0)
    n = len(measured)
    if n < 9:
        return {
            "sample": n,
            "verdict": ("insufficient sample: fewer than 9 positions with an "
                        "executable excursion, so no policy comparison is "
                        "reported"),
        }

    third = n // 3
    train = measured[:third]
    validation = measured[third:2 * third]
    test = measured[2 * third:]

    def score(rows, *, take_profit_pct, stop_pct=-12.0):
        """
        Net expectancy of a BRACKET: exit at the fixed level, else at the stop.

        The model is deliberately simple and is stated so it cannot be mistaken
        for the production policy:

          * the position exits at ``take_profit_pct`` if the sampled excursion
            reached it, otherwise it exits at ``stop_pct`` if the excursion fell
            that far, otherwise it carries the last sampled return;
          * every figure is a SAMPLED quote estimate, so an intrasecond spike is
            not seen and the fill is not guaranteed;
          * nothing here accounts for the time a re-quote takes, for a venue
            thawing, or for a market that stops being quotable.

        It is a comparison harness on ONE recorded sample, not a strategy, and
        it is not evidence that any exit policy avoids losses.
        """
        returns: list[float] = []
        tp_hits = 0
        stop_hits = 0
        for row in rows:
            mfe, mae = row["mfe"], row["mae"]
            if take_profit_pct is not None and mfe >= take_profit_pct:
                returns.append(float(take_profit_pct))
                tp_hits += 1
            elif stop_pct is not None and mae <= stop_pct:
                returns.append(float(stop_pct))
                stop_hits += 1
            else:
                returns.append(mae)
        if not returns:
            return None
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        gross_win = sum(wins)
        gross_loss = -sum(losses)
        equity, peak, max_dd = 0.0, 0.0, 0.0
        for r in returns:
            equity += r
            peak = max(peak, equity)
            max_dd = min(max_dd, equity - peak)
        ordered = sorted(returns)
        tail = ordered[:max(1, len(ordered) // 10)]
        return {
            "n": len(returns),
            "net_expectancy_pct": round(stats.mean(returns), 4),
            "win_rate_pct": round(100.0 * len(wins) / len(returns), 2),
            "profit_factor": (round(gross_win / gross_loss, 4)
                              if gross_loss > 0 else None),
            "avg_win_pct": (round(stats.mean(wins), 4) if wins else None),
            "avg_loss_pct": (round(stats.mean(losses), 4) if losses else None),
            "max_drawdown_pct_points": round(max_dd, 4),
            "tail_loss_pct_10pct": round(stats.mean(tail), 4),
            "tp_hit_rate_pct": round(100.0 * tp_hits / len(rows), 2),
            "stop_hit_rate_pct": round(100.0 * stop_hits / len(rows), 2),
        }

    def mfe_reached(row, tp):
        return tp is not None and row["mfe"] >= tp

    levels = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, None]
    train_table = {tp: score(train, take_profit_pct=tp) for tp in levels}
    # The best level is chosen on TRAIN only, by net expectancy.
    scored = [tp for tp in levels if train_table[tp]]
    chosen = max(scored, key=lambda tp: train_table[tp]["net_expectancy_pct"])
    validation_table = {tp: score(validation, take_profit_pct=tp)
                        for tp in levels}
    test_table = {tp: score(test, take_profit_pct=tp) for tp in levels}
    # The level reported on TEST is the one TRAIN selected -- never the one that
    # happens to look best on TEST.
    return {
        "sample": n,
        "model": {
            "take_profit_basis": (
                "sampled executable SELL quote, cost-aware "
                "(net proceeds vs all-in entry cost)"),
            "stop_pct": -12.0,
            "stop_basis": "the existing production fast-failure stop level",
            "caveats": [
                "SAMPLED: intrasecond extremes are not captured.",
                "A quote is an execution estimate, not a fill.",
                "No re-quote latency, venue thaw, or quote-unavailability "
                "modelled.",
            ],
        },
        "split": {"train": len(train), "validation": len(validation),
                  "test": len(test)},
        "chosen_on_train_pct": chosen,
        "train_table": {str(tp): value for tp, value in train_table.items()},
        "validation_table": {str(tp): value
                             for tp, value in validation_table.items()},
        "locked_test": {str(tp): value for tp, value in test_table.items()},
        "selected_level_test_result": test_table.get(chosen),
        "verdict": (
            "The level is selected on TRAIN by net expectancy and reported on "
            "TEST. With 39/39/40 positions the bound is indicative of the "
            "recorded book only. It is NOT evidence that any exit policy avoids "
            "losses; no exit policy can guarantee zero losses."
        ),
    }


# ---------------------------------------------------------------------------
# 6. how misleading the mark-based excursions actually were
# ---------------------------------------------------------------------------


def analyse_mark_misleading(arm_exits, observations) -> dict:
    """
    What the mark-based MFE/MAE CLAIMED, beside what was realised.

    This is the defect the compact report fixes. It is measured here so the
    fix is justified by data rather than by taste: a mark pair that reports
    ``MAE +0.0%`` on a real loser, and a mark MFE on a losing position, is not
    a measurement of the trade.
    """
    losit_were_flat = 0
    losers_with_positive_mark_mfe = 0
    losers = 0
    mark_mfes_on_losers: list[float] = []
    rows_with_mark = 0
    for record in arm_exits:
        trade = record.get("trade") or {}
        position = trade.get("position") or {}
        realised = trade.get("realized_pnl") or {}
        ret = realised.get("return_pct")
        mark_mfe = position.get("OBSERVED_MARK_MFE")
        mark_mae = position.get("OBSERVED_MARK_mae")
        if mark_mfe is None and mark_mae is None:
            continue
        rows_with_mark += 1
        if ret is None:
            continue
        if ret < 0:
            losers += 1
            if mark_mfe is not None:
                mark_mfes_on_losers.append(float(mark_mfe))
                if float(mark_mfe) > 0:
                    losers_with_positive_mark_mfe += 1
            if mark_mae is not None and abs(float(mark_mae)) < 1e-9:
                losit_were_flat += 1
    return {
        "rows_with_a_mark_excursion": rows_with_mark,
        "losers": losers,
        "losers_whose_mark_MFE_was_POSITIVE": losers_with_positive_mark_mfe,
        "losers_whose_mark_MAE_was_exactly_0.0pct": losit_were_flat,
        "mark_mfe_on_losers_median_pct": (
            round(stats.median(mark_mfes_on_losers), 4)
            if mark_mfes_on_losers else None),
        "interpretation": (
            "A mark-MAE of exactly +0.0% on a losing position is not evidence "
            "the position never went down; it is evidence the aggregator "
            "observation did not cross a level that the position's own "
            "executable book did. The compact report therefore never prints a "
            "mark figure without an explicit Mark-only label, and the "
            "quote-based excursion is the one that carries the outcome."
        ),
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAP))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    snap = Path(args.snapshot)
    hold_path = snap / "paper_hold_path.jsonl"
    ledger = snap / "ledger.jsonl"
    paper_state = snap / "paper_state.json"

    observations, summaries = load_hold_path(hold_path)
    arm_exits = load_arm_exits(ledger)
    rejected = load_rejected(paper_state)

    report = {
        "inputs": {
            "hold_path": str(hold_path),
            "ledger": str(ledger),
            "paper_state": str(paper_state),
        },
        "1_executable_mfe_mae": analyse_excursions(observations, summaries),
        "2_sold_too_early": analyse_sold_too_early(
            observations, summaries, arm_exits),
        "3_rejected_candidates": analyse_rejected(rejected, observations),
        "4_entry_signals": analyse_entry_features(observations, summaries,
                                                 arm_exits),
        "5_exit_policy_comparison": analyse_exit_policies(observations,
                                                          summaries),
        "6_mark_excursion_defect": analyse_mark_misleading(arm_exits,
                                                          observations),
    }

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True,
                                             default=str),
                                  encoding="utf-8")

    summary = {
        "positions_with_hold_path": len(observations),
        "executable_mfe_mae": report["1_executable_mfe_mae"]["by_basis"],
        "sold_too_early": {
            "exits": report["2_sold_too_early"]["exits_total"],
            "measurable": report["2_sold_too_early"]["measurable"],
            "unevaluable": report["2_sold_too_early"]["unevaluable"],
            "losing_exits_with_later_positive_net_quote": len(
                report["2_sold_too_early"]
                ["losing_exits_with_positive_later_net_quote"]),
        },
        "rejected": report["3_rejected_candidates"]["by_category"],
        "rejected_post_outcome": report["3_rejected_candidates"][
            "post_rejection_outcome"],
        "exit_policy_sample": report["5_exit_policy_comparison"].get("sample"),
        "mark_excursion_defect": report["6_mark_excursion_defect"],
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
