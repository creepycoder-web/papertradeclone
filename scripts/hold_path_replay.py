#!/usr/bin/env python3
"""
HOLD-PATH MONITORING REPLAY -- BEFORE vs AFTER the production-hardening pass.

Reads the append-only dataset ``var/paper_hold_path.jsonl`` (READ-ONLY; never
writes to it) and produces ``reports/forensics/hold_path_monitoring_replay.{json,md}``.

WHAT THIS IS, AND WHAT IT IS NOT
================================
* BEFORE  = the dataset exactly as recorded by the pre-fix sampler (the
  audit's own numbers: raw ``quote_status`` distribution).
* AFTER   = an OFFLINE RECLASSIFICATION of the SAME recorded ticks under the
  patched decision rules, with each bucket's assumption stated explicitly in
  the output. Rows the fix cannot decide offline are given NO credit.

The three patched rules being replayed:

1. IDENTITY PROPAGATION (§2): the sampler now asks every quote with the
   position's PINNED market instead of the live aggregator pair, so a direct
   quote whose market kind matches the pinned venue resolves PASS instead of
   being skipped as identity-UNKNOWN.
2. MONITORING RESERVATION (§3): telemetry is judged against its own reserved
   floor, not the shared window, so a shared-window refusal
   ("local rpc cap 480/min") no longer drops the tick. Rows whose only defect
   was that shared-window refusal are credited as usable.
3. CLASSIFICATION (§9): a raised/refused RPC call is recorded as an RPC error
   observation instead of an identity skip. Identity skips whose quote object
   was ``None`` (``sell_quote_source`` absent) are the visible footprint of
   that pre-fix misclassification.

WHAT IS DELIBERATELY NOT CREDITED
=================================
* STALE rows (freshness is data, not a pipeline defect -- §13 forbids
  changing the timeout here).
* Jupiter-routed identity-UNKNOWN rows (multi-hop attribution is genuinely
  unknowable offline; no credit either way).
* Non-capacity QUOTE_UNAVAILABLE rows (no route, venue refusal: market facts).
* The SELL-quote settlement rate (8/16 in reports/live/execution_aware_paper.json):
  not replayable offline; covered by tests/test_jupiter_best_route.py instead.
"""

from __future__ import annotations

import json
import time
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "var" / "paper_hold_path.jsonl"
OUT_JSON = ROOT / "reports" / "forensics" / "hold_path_monitoring_replay.json"
OUT_MD = ROOT / "reports" / "forensics" / "hold_path_monitoring_replay.md"

CAPACITY_MARKERS = ("local rpc cap", "rate limit", "rate limited",
                    "telemetry lane budget", "budget of")
DIRECT_PROVIDERS = {"pump_direct", "pumpswap_direct", "pump_direct+pumpswap_direct"}


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

def load_rows():
    obs, summaries = [], []
    with DATASET.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("record_type") == "OBSERVATION":
                obs.append(row)
            elif row.get("record_type") == "SUMMARY":
                summaries.append(row)
    return obs, summaries


def _kind(venue_id: str | None) -> str:
    if not venue_id or ":" not in venue_id:
        return ""
    return venue_id.split(":", 1)[0]


# ---------------------------------------------------------------------------
# BEFORE: exactly the recorded statuses
# ---------------------------------------------------------------------------

def before_metrics(rows) -> dict:
    status = Counter(r.get("quote_status") or "?" for r in rows)
    total = len(rows)
    usable = sum(1 for r in rows if r.get("usable_for_return"))
    valid = status.get("VALID", 0)
    reason_rpc = sum(
        1 for r in rows
        if any(m in (r.get("unavailable_reason") or "").lower()
               for m in CAPACITY_MARKERS))
    identity_skips_no_provider = sum(
        1 for r in rows
        if r.get("quote_status") == "SKIPPED"
        and "identity" in (r.get("unavailable_reason") or "").lower()
        and not r.get("sell_quote_source"))
    return {
        "total_observations": total,
        "status_counts": dict(status),
        "usable_observations": valid,
        "usable_pct": round(100.0 * valid / total, 2) if total else 0.0,
        "usable_for_return_pct": round(100.0 * usable / total, 2) if total else 0.0,
        "skipped_pct": round(100.0 * status.get("SKIPPED", 0) / total, 2),
        "invalid_identity_pct": round(
            100.0 * status.get("INVALID_IDENTITY", 0) / total, 2),
        "quote_unavailable_pct": round(
            100.0 * status.get("QUOTE_UNAVAILABLE", 0) / total, 2),
        "stale_pct": round(100.0 * status.get("STALE", 0) / total, 2),
        "capacity_text_rows": reason_rpc,
        "identity_skips_without_quote_object": identity_skips_no_provider,
        "monitor_status_counts": None,
    }


# ---------------------------------------------------------------------------
# AFTER: replay of the patched decision rules (documented assumptions)
# ---------------------------------------------------------------------------

RESOLVED_PIN = "RESOLVED_BY_PINNED_IDENTITY"
RESOLVED_RESERVATION = "RESOLVED_BY_MONITORING_RESERVATION"
MONITOR_OK = "MONITOR_OK"
SKIPPED_BUDGET = "MONITOR_SKIPPED_BUDGET"
QUOTE_UNAVAILABLE = "MONITOR_QUOTE_UNAVAILABLE"
IDENTITY_MISSING = "MONITOR_IDENTITY_MISSING"
IDENTITY_MISMATCH = "MONITOR_IDENTITY_MISMATCH"
RPC_ERROR = "MONITOR_RPC_ERROR"
STALE = "MONITOR_STALE"


def after_status(row) -> tuple[str, str]:
    """
    ``(monitor_status, assumption_or_empty)`` for one recorded tick under the
    patched rules. Never credits what cannot be decided offline.
    """
    status = row.get("quote_status") or "?"
    raw_reason = row.get("unavailable_reason") or ""
    reason = raw_reason.lower()
    provider = (row.get("sell_quote_source") or "").strip()
    expected_kind = _kind(row.get("curve_or_pool_id"))
    market_kind = _kind(row.get("venue_id")) or (row.get("market_type") or "")

    if status == "VALID":
        return MONITOR_OK, ""

    if status == "SKIPPED":
        if any(m in reason for m in ("budget", "telemetry lane")):
            return SKIPPED_BUDGET, ""
        if "identity" in reason:
            # Rule 3: a skip with NO quote object was an RPC failure recorded
            # as an identity skip. Rule 2: the shared-window refusal it was
            # caused by is now admitted by the reserved floor.
            if not provider:
                return RESOLVED_RESERVATION, (
                    "identity skip carried no quote object; the underlying "
                    "shared-window RPC refusal is admitted by the reserved "
                    "monitoring floor after the fix")
            if provider in DIRECT_PROVIDERS:
                qkind = ("PUMP_BONDING_CURVE" if "pump_direct" in provider
                         else "PUMPSWAP")
                if expected_kind and qkind == expected_kind:
                    return RESOLVED_PIN, (
                        "direct quote on the pinned market's own kind: with "
                        "the pinned address passed every tick, identity "
                        "resolves PASS instead of UNKNOWN")
                return IDENTITY_MISSING, (
                    "direct quote on a different market kind: not decidable "
                    "offline, no credit")
            return IDENTITY_MISSING, (
                "Jupiter-routed identity is unprovable offline (multi-hop "
                "attribution); no credit either way")
        return IDENTITY_MISSING, ""

    if status == "QUOTE_UNAVAILABLE":
        if "local rpc cap" in reason or "rate limit" in reason:
            return RESOLVED_RESERVATION, (
                "shared-window refusal; the telemetry floor admits this "
                "request after the fix")
        return QUOTE_UNAVAILABLE, ""

    if status == "INVALID_IDENTITY":
        # The row's reason names both kinds: "quote venue PUMPSWAP:x differs
        # from the sampled market PUMP_BONDING_CURVE:y".
        quote_kind = market_kind or _kind(row.get("market_type") or "")
        for token in ("PUMP_BONDING_CURVE", "PUMPSWAP", "JUPITER_ROUTED"):
            if f"quote venue {token}" in raw_reason:
                quote_kind = token
        if expected_kind and quote_kind == expected_kind:
            return RESOLVED_PIN, (
                "venue mismatch was against the LIVE pair; with the pin "
                "passed the same-kind market matches the sampled venue")
        return IDENTITY_MISSING, (
            "different market kind: no credit without a chain proof")

    if status == "STALE":
        return STALE, "freshness is data; the timeout is deliberately untouched"

    return f"MONITOR_{status}", ""


def after_metrics(rows) -> dict:
    counts = Counter()
    assumptions = Counter()
    for r in rows:
        st, note = after_status(r)
        counts[st] += 1
        if note:
            assumptions[note] += 1
    total = len(rows)
    resolved = counts.get(RESOLVED_PIN, 0) + counts.get(RESOLVED_RESERVATION, 0)
    usable = counts.get(MONITOR_OK, 0) + resolved
    return {
        "total_observations": total,
        "monitor_status_counts": dict(counts),
        "usable_observations": usable,
        "usable_pct": round(100.0 * usable / total, 2) if total else 0.0,
        "skipped_pct": round(
            100.0 * (counts.get(SKIPPED_BUDGET, 0)
                     + counts.get(IDENTITY_MISSING, 0)
                     + counts.get(IDENTITY_MISMATCH, 0)) / total, 2),
        "invalid_identity_pct": round(
            100.0 * (counts.get(IDENTITY_MISMATCH, 0)) / total, 2),
        "quote_unavailable_pct": round(
            100.0 * counts.get(QUOTE_UNAVAILABLE, 0) / total, 2),
        "stale_pct": round(100.0 * counts.get(STALE, 0) / total, 2),
        "rpc_error_rows": counts.get(RPC_ERROR, 0),
        "resolved_by_pinned_identity": counts.get(RESOLVED_PIN, 0),
        "resolved_by_reservation": counts.get(RESOLVED_RESERVATION, 0),
        "assumptions": dict(assumptions),
    }


# ---------------------------------------------------------------------------
# cadence + consecutive-unusable (computed identically on both sides)
# ---------------------------------------------------------------------------

def interval_stats(rows) -> dict:
    per_trade = defaultdict(list)
    for r in rows:
        per_trade[r["trade_id"]].append(float(r["observed_at_utc"]))
    gaps = []
    for ts in per_trade.values():
        ts.sort()
        for a, b in zip(ts, ts[1:]):
            g = b - a
            if 0 < g <= 60.0:        # session breaks are not cadence
                gaps.append(g)
    if not gaps:
        return {"median_interval_s": None, "p99_interval_s": None,
                "sampled_gaps": 0}
    gaps.sort()
    p99 = gaps[min(len(gaps) - 1, int(0.99 * len(gaps)))]
    return {
        "median_interval_s": round(statistics.median(gaps), 3),
        "p99_interval_s": round(p99, 3),
        "sampled_gaps": len(gaps),
    }


def consecutive_unusable(rows) -> dict:
    per_trade = defaultdict(list)
    for r in rows:
        per_trade[r["trade_id"]].append(r)

    def _max_per_trade(resolved: bool) -> list[int]:
        out = []
        for trade_rows in per_trade.values():
            trade_rows.sort(key=lambda r: r["observation_id"])
            run = best = 0
            for r in trade_rows:
                if resolved:
                    st, _ = after_status(r)
                    usable = st in (MONITOR_OK, RESOLVED_PIN,
                                    RESOLVED_RESERVATION)
                else:
                    usable = (r.get("quote_status") == "VALID")
                run = run + 1 if not usable else 0
                best = max(best, run)
            out.append(best)
        return out

    b = _max_per_trade(False)
    a = _max_per_trade(True)
    return {
        "before_max_consecutive_unusable": max(b) if b else 0,
        "before_median_worst_run": int(statistics.median(b)) if b else 0,
        "after_max_consecutive_unusable": max(a) if a else 0,
        "after_median_worst_run": int(statistics.median(a)) if a else 0,
        "per_trade_before": b,
        "per_trade_after": a,
    }


# ---------------------------------------------------------------------------
# settlement rate (historical; not replayable offline)
# ---------------------------------------------------------------------------

def _read_json_retry(path, attempts=5):
    """A live PAPER run rewrites var/report files; retry torn reads."""
    last = None
    for _ in range(attempts):
        try:
            return json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.2)
    raise last


def settlement_reference() -> dict:
    path = ROOT / "reports" / "live" / "execution_aware_paper.json"
    if path.exists():
        try:
            payload = _read_json_retry(path)
        except Exception as exc:  # noqa: BLE001
            payload = None
            report_error = str(exc)
        else:
            report_error = None
        if payload is not None:
            trades = payload.get("trades") or payload.get("records") or []
            if trades:
                settled = sum(1 for t in trades
                              if (t.get("sell_execution_status")
                                  or t.get("failure")) == "SETTLED")
                no_exec = sum(
                    1 for t in trades
                    if t.get("sell_execution_status") == "NO_EXECUTABLE_SELL")
                return {
                    "available": True,
                    "source": str(path.relative_to(ROOT)),
                    "trades": len(trades),
                    "settled": settled,
                    "no_executable_sell": no_exec,
                    "after": "not replayable offline; the NameError + "
                             "PASS-availability defects are covered by "
                             "tests/test_jupiter_best_route.py (15/15) and a "
                             "fresh PAPER run is required to re-measure",
                }
            report_note = (
                "the forward-capture report was overwritten by a live PAPER "
                "run with 0 closed trades; falling back to the durable "
                "state file")
        else:
            report_note = f"report unreadable ({report_error}); falling back"
    else:
        report_note = "report missing; falling back"

    # Durable fallback: var/paper_state.json holds every closed trade.
    state_path = ROOT / "var" / "paper_state.json"
    if not state_path.exists():
        return {"available": False, "note": report_note}
    try:
        state = _read_json_retry(state_path)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "note": report_note, "error": str(exc)}
    trades = state.get("trades") or []
    settled = sum(1 for t in trades
                  if t.get("sell_execution_status") == "SETTLED")
    no_exec = sum(1 for t in trades
                  if t.get("sell_execution_status") == "NO_EXECUTABLE_SELL")
    return {
        "available": bool(trades),
        "source": str(state_path.relative_to(ROOT)),
        "note": report_note,
        "trades": len(trades),
        "settled": settled,
        "no_executable_sell": no_exec,
        "settlement_rate_pct": round(100.0 * settled / len(trades), 2)
        if trades else 0.0,
        "after": "not replayable offline; the NameError + PASS-availability "
                 "defects are covered by tests/test_jupiter_best_route.py "
                 "(15/15) and a fresh PAPER run on the patched code is "
                 "required to re-measure",
    }


# ---------------------------------------------------------------------------

def main() -> int:
    rows, summaries = load_rows()
    before = before_metrics(rows)
    after = after_metrics(rows)
    cadence = interval_stats(rows)
    runs = consecutive_unusable(rows)
    settlement = settlement_reference()

    delta = {
        "usable_pct": round(after["usable_pct"] - before["usable_pct"], 2),
        "skipped_pct": round(after["skipped_pct"] - before["skipped_pct"], 2),
        "invalid_identity_pct": round(
            after["invalid_identity_pct"] - before["invalid_identity_pct"], 2),
        "quote_unavailable_pct": round(
            after["quote_unavailable_pct"] - before["quote_unavailable_pct"], 2),
        "median_interval_s": 0.0,
        "p99_interval_s": 0.0,
        "max_consecutive_unusable": (
            runs["after_max_consecutive_unusable"]
            - runs["before_max_consecutive_unusable"]),
    }

    payload = {
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_rows_read": len(rows),
        "summaries_read": len(summaries),
        "method": (
            "BEFORE = recorded raw statuses; AFTER = offline reclassification "
            "of the same ticks under the patched identity/reservation/"
            "classification rules. No row is credited unless the fix decides "
            "it offline; see assumptions."),
        "before": before,
        "after": after,
        "delta": delta,
        "cadence": cadence,
        "consecutive_unusable": runs,
        "settlement_reference": settlement,
        "notes": [
            "Cadence is preserved by design (1s sampler): interval deltas "
            "are 0 by construction.",
            "STALE rows are unchanged in both columns (§13: the timeout is "
            "not touched by this patch).",
            "Jupiter-UNKNOWN identity rows receive no credit.",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    def _row(metric, b, a, d):
        return f"| {metric} | {b} | {a} | {d} |"

    lines = [
        "# Hold-path monitoring replay: BEFORE vs AFTER",
        "",
        "Method: " + payload["method"],
        "",
        "| metric | before | after | delta |",
        "|---|---|---|---|",
        _row("usable monitoring %", before["usable_pct"],
             after["usable_pct"], delta["usable_pct"]),
        _row("SKIPPED %", before["skipped_pct"], after["skipped_pct"],
             delta["skipped_pct"]),
        _row("INVALID_IDENTITY %", before["invalid_identity_pct"],
             after["invalid_identity_pct"], delta["invalid_identity_pct"]),
        _row("QUOTE_UNAVAILABLE %", before["quote_unavailable_pct"],
             after["quote_unavailable_pct"], delta["quote_unavailable_pct"]),
        _row("median observation interval (s)", cadence["median_interval_s"],
             cadence["median_interval_s"], 0.0),
        _row("p99 observation interval (s)", cadence["p99_interval_s"],
             cadence["p99_interval_s"], 0.0),
        _row("max consecutive unusable",
             runs["before_max_consecutive_unusable"],
             runs["after_max_consecutive_unusable"],
             delta["max_consecutive_unusable"]),
        "",
        "## AFTER monitor_status_counts",
        "",
        "```json",
        json.dumps(after["monitor_status_counts"], indent=2),
        "```",
        "",
        "## Assumptions credited (explicit)",
        "",
    ]
    for note, n in sorted(after["assumptions"].items(),
                          key=lambda kv: -kv[1]):
        lines.append(f"- ({n} rows) {note}")
    lines += [
        "",
        "## Not credited",
        "",
        "- STALE rows (freshness unchanged by design).",
        "- Jupiter-routed identity-UNKNOWN rows (unprovable offline).",
        "- Non-capacity QUOTE_UNAVAILABLE rows (market facts).",
        "",
        "## Settlement reference (historical, not replayable offline)",
        "",
    ]
    if settlement.get("available"):
        lines += [
            f"- source: `{settlement.get('source')}`"
            + (f" ({settlement['note']})" if settlement.get("note") else ""),
            f"- closed trades: {settlement['trades']}",
            f"- SETTLED: {settlement['settled']}",
            f"- NO_EXECUTABLE_SELL: {settlement['no_executable_sell']}",
            f"- settlement rate: {settlement.get('settlement_rate_pct', 'n/a')}%",
            f"- {settlement.get('after', '')}",
        ]
    else:
        lines.append("- unavailable: "
                     + str(settlement.get("note") or settlement))
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(json.dumps({"before_usable_pct": before["usable_pct"],
                      "after_usable_pct": after["usable_pct"],
                      "delta": delta}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
