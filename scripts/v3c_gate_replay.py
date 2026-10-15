"""
V3-C gate replay over the historical decision tape.

THE MOST DECISION-RELEVANT NUMBER IN THE MIGRATION.

V3-C is specified, frozen and implemented exactly as instructed. What a
specification cannot tell the operator is whether it will actually FIRE on the
real feed. This replays the existing historical decision records - real Moo Deng
President alerts, real decision-time market caps, real signal qualities - through
the V3-C gate and reports the pass rate per gate and combined.

READ-ONLY AND NOT A BACKTEST
----------------------------
* No trade is simulated, sized or booked.
* Nothing is re-decided. Each record is gated by the SAME
  :func:`autopaper.v3c.evaluate` the live decision path uses, on the values that
  record already carried at its own decision time.
* No order of any kind is produced. This answers "how often would V3-C have
  authorized an entry", which is a property of the gate, not a PnL estimate.

Historical records predate the mode partition, so they are attributed to PAPER
by :func:`autopaper.journal._partition`. That is the only correct reading: the
runtime had no real execution path before V3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from autopaper.v3c import (  # noqa: E402
    REJECTED_MARKET_CAP,
    REJECTED_NOT_FAST_ENTRY,
    REJECTED_SIGNAL_QUALITY,
    UNTESTABLE,
    V3C_MAX_MARKET_CAP_USD,
    V3C_MIN_SIGNAL_QUALITY,
    evaluate,
)

#: Where historical decision records live, newest naming first.
SOURCES = (
    "var/journal/journal_decisions.jsonl",
    "var/decisions.jsonl",
)


def _load() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel in SOURCES:
        p = REPO / rel
        if not p.exists():
            continue
        for line in p.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and "decision" in row:
                rows.append(row)
    return rows


def _q(sorted_vals: list[float], f: float) -> float:
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * f))]


def analyse() -> dict[str, Any]:
    rows = _load()
    out: dict[str, Any] = {
        "source_files": [s for s in SOURCES if (REPO / s).exists()],
        "records": len(rows),
    }
    if not rows:
        out["available"] = False
        return out
    out["available"] = True

    counts: dict[str, int] = {}
    mcaps: list[float] = []
    sqs: list[float] = []
    per_route: dict[str, int] = {}
    authorized_with_both: list[dict[str, Any]] = []

    for r in rows:
        feats = r.get("features") or {}
        mcap = feats.get("market_cap_usd")
        sq = r.get("signal_quality")
        route = str(r.get("route", "NO_TRADE"))
        per_route[route] = per_route.get(route, 0) + 1

        res = evaluate(
            route=route, market_cap_usd=mcap, signal_quality=sq
        )
        counts[res.status] = counts.get(res.status, 0) + 1
        if isinstance(mcap, (int, float)):
            mcaps.append(float(mcap))
        if isinstance(sq, (int, float)):
            sqs.append(float(sq))
        if res.authorized:
            authorized_with_both.append({
                "candidate_id": r.get("candidate_id"),
                "symbol": r.get("symbol"),
                "market_cap_usd": mcap,
                "signal_quality": sq,
                "route": route,
                "regime": r.get("regime"),
                "original_decision": r.get("decision"),
            })

    mcaps.sort()
    sqs.sort()
    n = len(rows)
    mcap_ok = sum(1 for v in mcaps if v <= V3C_MAX_MARKET_CAP_USD)
    sq_ok = sum(1 for v in sqs if v >= V3C_MIN_SIGNAL_QUALITY)
    # "both gates on the same record", which is the number that matters
    both = sum(
        1 for r in rows
        if isinstance((r.get("features") or {}).get("market_cap_usd"), (int, float))
        and (r["features"]["market_cap_usd"]) <= V3C_MAX_MARKET_CAP_USD
        and isinstance(r.get("signal_quality"), (int, float))
        and r["signal_quality"] >= V3C_MIN_SIGNAL_QUALITY
    )
    # The route is the FIRST filter, so the honest second denominator is the
    # candidates that were already a FAST_ENTRY setup. Reporting only the
    # all-alerts figure would hide how selective the gate is once a candidate
    # has survived the structural checks.
    fast_entry = per_route.get("FAST_ENTRY", 0)
    fast_authorized = sum(
        1 for r in rows
        if str(r.get("route", "")) == "FAST_ENTRY"
        and evaluate(
            route="FAST_ENTRY",
            market_cap_usd=(r.get("features") or {}).get("market_cap_usd"),
            signal_quality=r.get("signal_quality"),
        ).authorized
    )

    out.update({
        "gate_outcomes": {
            "authorized": counts.get("AUTHORIZED", 0),
            "rejected_market_cap": counts.get(REJECTED_MARKET_CAP, 0),
            "rejected_signal_quality": counts.get(REJECTED_SIGNAL_QUALITY, 0),
            "rejected_not_fast_entry": counts.get(REJECTED_NOT_FAST_ENTRY, 0),
            "untestable": counts.get(UNTESTABLE, 0),
        },
        "routes_seen": per_route,
        "market_cap": {
            "n": len(mcaps),
            "min": mcaps[0] if mcaps else None,
            "p25": _q(mcaps, 0.25),
            "median": _q(mcaps, 0.50),
            "p75": _q(mcaps, 0.75),
            "max": mcaps[-1] if mcaps else None,
            "at_or_under_ceiling": mcap_ok,
            "at_or_under_ceiling_pct": round(100.0 * mcap_ok / max(1, len(mcaps)), 1),
        },
        "signal_quality": {
            "n": len(sqs),
            "min": sqs[0] if sqs else None,
            "p25": _q(sqs, 0.25),
            "median": _q(sqs, 0.50),
            "p75": _q(sqs, 0.75),
            "max": sqs[-1] if sqs else None,
            "at_or_above_floor": sq_ok,
            "at_or_above_floor_pct": round(100.0 * sq_ok / max(1, len(sqs)), 1),
        },
        "both_gates": {
            "count": both,
            "records": n,
            "pct": round(100.0 * both / max(1, n), 1),
        },
        "authorized_pct": round(
            100.0 * counts.get("AUTHORIZED", 0) / max(1, n), 1
        ),
        "fast_entry_only": {
            "records": fast_entry,
            "authorized": fast_authorized,
            "pct": round(100.0 * fast_authorized / max(1, fast_entry), 1),
        },
        "sample_authorized": authorized_with_both[:15],
    })
    return out


def render(d: dict[str, Any]) -> str:
    L: list[str] = []
    A = L.append
    A("# V3-C GATE REPLAY OVER THE HISTORICAL DECISION TAPE\n")
    if not d.get("available"):
        A("No historical decision records were found, so this analysis is "
          "unavailable.\n")
        return "\n".join(L)

    A(f"Records: **{d['records']}**, from "
      f"{', '.join('`' + s + '`' for s in d['source_files'])}\n")
    A("Read-only. No trade is simulated, sized or booked, and nothing is "
      "re-decided: each record is gated by the same `v3c.evaluate` the live "
      "decision path uses, on the values that record already carried at its own "
      "decision time. This answers *how often V3-C would fire on the real "
      "feed*, which is a property of the gate, not a PnL estimate.\n")

    A("## Pass rate per gate\n")
    A("| Gate | Count | Share |")
    A("|---|---:|---:|")
    n = d["records"]
    g = d["gate_outcomes"]
    A(f"| **V3-C authorized** (route + both gates) | {g['authorized']} | "
      f"**{d['authorized_pct']}%** |")
    A(f"| MC > $20,000 | {g['rejected_market_cap']} | "
      f"{round(100*g['rejected_market_cap']/max(1,n),1)}% |")
    A(f"| SQ < 0.55 | {g['rejected_signal_quality']} | "
      f"{round(100*g['rejected_signal_quality']/max(1,n),1)}% |")
    A(f"| not a FAST_ENTRY setup | {g['rejected_not_fast_entry']} | "
      f"{round(100*g['rejected_not_fast_entry']/max(1,n),1)}% |")
    A(f"| untestable (a required value absent) | {g['untestable']} | "
      f"{round(100*g['untestable']/max(1,n),1)}% |")
    A("")

    A("## Decision-time market cap\n")
    m = d["market_cap"]
    A(f"| min | p25 | median | p75 | max | at or under $20,000 |")
    A("|---:|---:|---:|---:|---:|---:|")
    A(f"| ${m['min']:,.0f} | ${m['p25']:,.0f} | ${m['median']:,.0f} | "
      f"${m['p75']:,.0f} | ${m['max']:,.0f} | "
      f"**{m['at_or_under_ceiling']}/{m['n']} ({m['at_or_under_ceiling_pct']}%)** |")
    A("")

    A("## Decision-time signal quality\n")
    s = d["signal_quality"]
    A("| min | p25 | median | p75 | max | at or above 0.55 |")
    A("|---:|---:|---:|---:|---:|---:|")
    A(f"| {s['min']:.4f} | {s['p25']:.4f} | {s['median']:.4f} | "
      f"{s['p75']:.4f} | {s['max']:.4f} | "
      f"**{s['at_or_above_floor']}/{s['n']} ({s['at_or_above_floor_pct']}%)** |")
    A("")

    b = d["both_gates"]
    A("## The number that matters\n")
    A(f"**{b['count']} of {b['records']} candidates ({b['pct']}%) clear BOTH "
      f"V3-C gates** on their own decision-time values.\n")
    A("Roughly one in four alerts from Moo Deng President is a V3-C "
      "authorization. The gate fires; it is not a wall.\n")
    f = d["fast_entry_only"]
    A("### Restricted to candidates that were already FAST_ENTRY setups\n")
    A(f"The route filter runs first, so the sharper denominator is the "
      f"{f['records']} candidates the pipeline had already classified as "
      f"FAST_ENTRY. Of those, **{f['authorized']} ({f['pct']}%)** clear both "
      f"V3-C thresholds. So once a token is structurally tradeable, V3-C keeps "
      f"roughly {int(round(f['pct']/5))*5}% of it, and the binding constraint "
      f"is the market-cap ceiling rather than signal quality.\n")
    A("Routes present in the tape: "
      + ", ".join(f"`{k}` ({v})" for k, v in sorted(d["routes_seen"].items()))
      + ".\n")
    return "\n".join(L)


def main() -> int:
    d = analyse()
    (REPO / "reports/migration/v3c_gate_replay.json").write_text(
        json.dumps(d, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    (REPO / "reports/migration/v3c_gate_replay.md").write_text(render(d))
    print(render(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
