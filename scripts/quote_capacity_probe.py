"""
Quote-capacity and decision-latency probe (READ-ONLY).

Measures exactly what the quote-capacity fix has to be judged on:

  * executable quote availability
  * quote latency (p50/p95/p99)
  * signal -> executable-decision latency
  * how often capacity exhaustion is reported as "no executable quote"
    instead of the honest RPC classification

It trades nothing, signs nothing and broadcasts nothing. It only READS chain
state and writes a JSON report.

Run it BEFORE and AFTER a change; the two reports are the evidence:

    PYTHONPATH=src .venv/bin/python scripts/quote_capacity_probe.py \
        --label before --pace 0
    PYTHONPATH=src .venv/bin/python scripts/quote_capacity_probe.py \
        --label after  --pace 0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
import time
from pathlib import Path

from autopaper.config import build_config
from autopaper.engine import PaperTrader

CANDIDATES = Path("var/candidates.jsonl")
REPORTS = Path("reports/research")


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return round(s[k], 1)


def dist(values: list[float]) -> dict[str, float | None]:
    return {
        "n": len(values),
        "p50": pct(values, 0.50),
        "p95": pct(values, 0.95),
        "p99": pct(values, 0.99),
        "max": round(max(values), 1) if values else None,
        "mean": round(st.mean(values), 1) if values else None,
    }


def mints_from(report: str, limit: int) -> list[dict]:
    rows = json.loads(Path(report).read_text())["rows"]
    return [{"mint": r["mint"], "symbol": r.get("symbol") or "?"}
            for r in rows if r.get("mint")][:limit]


def recent_mints(limit: int) -> list[dict]:
    rows = [json.loads(l) for l in CANDIDATES.read_text().splitlines() if l.strip()]
    seen, out = set(), []
    for r in reversed(rows):
        m = r.get("mint")
        if not m or m in seen:
            continue
        seen.add(m)
        out.append({"mint": m, "symbol": r.get("symbol") or "?"})
        if len(out) >= limit:
            break
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--pace", type=float, default=0.0,
                    help="seconds between candidates; 0 = as fast as possible")
    ap.add_argument("--from-report", default=None,
                    help="reuse the exact candidate list from a previous "
                         "report, so before/after are apples-to-apples")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="candidates quoted concurrently (1 = the old "
                         "sequential behaviour)")
    args = ap.parse_args()

    trader = PaperTrader(build_config())
    engine = await trader._paper_bridge_instance()
    ref = engine.sol_reference
    sol_lamports = int(0.50 / float(ref.usd_per_sol) * 1e9)

    qsvc = engine.quote_service
    reader = qsvc._reader
    rpc = reader.rpc

    print(f"[{args.label}] SOL/USD={float(ref.usd_per_sol):.2f} "
          f"rpc_url={getattr(rpc, 'url', '?')}")
    print(f"[{args.label}] endpoint(s)={len(getattr(rpc, 'urls', [getattr(rpc,'url','')])) or 1} "
          f"max_concurrent={getattr(rpc, 'max_concurrent', 'n/a')} "
          f"pace={args.pace}s concurrency={args.concurrency}")

    rows: list[dict] = []
    queue = (mints_from(args.from_report, args.n) if args.from_report
             else recent_mints(args.n))
    rpc_calls_before = getattr(rpc, "calls", 0)

    async def one(c: dict) -> dict:
        rec: dict = {"symbol": c["symbol"], "mint": c["mint"]}
        t_signal = time.monotonic()
        try:
            mark = await engine.observe(c["mint"])
            rec["mark_ms"] = round((time.monotonic() - t_signal) * 1000, 1)
            rec["mark_available"] = bool(mark.available)
            if not mark.available or not mark.value_usd:
                rec["outcome"] = "NO_MARK"
                rec["reason"] = mark.missing_reason
                return rec

            t0 = time.monotonic()
            try:
                dec = await reader.mint_decimals(c["mint"])
            except Exception as exc:  # noqa: BLE001
                rec["outcome"] = "DECIMALS_RAISED"
                rec["reason"] = f"{type(exc).__name__}: {exc}"[:200]
                rec["decision_ms"] = round((time.monotonic() - t_signal) * 1000, 1)
                return rec
            rec["decimals_ms"] = round((time.monotonic() - t0) * 1000, 1)
            rec["decimals_error"] = getattr(reader, "last_mint_decimals_error", "")
            if not dec:
                # A mint account that does not exist YET is a market fact; a
                # throttled read is a capacity fact. They must not be pooled.
                cause = (rec["decimals_error"] or "").lower()
                rec["outcome"] = ("DECIMALS_CAPACITY"
                                  if ("rate" in cause or "timeout" in cause
                                      or "rpc" in cause)
                                  else "DECIMALS_ABSENT")
                return rec

            t1 = time.monotonic()
            quote, cands = await qsvc.quote_buy_best(
                mint=c["mint"], sol_lamports=sol_lamports, token_decimals=dec,
                observed_pair_address=mark.pair_address)
            rec["quote_ms"] = round((time.monotonic() - t1) * 1000, 1)
            rec["outcome"] = "QUOTED" if quote.available else "NO_QUOTE"
            rec["provider"] = quote.provider
            rec["identity"] = quote.identity
            rec["venues_enumerated"] = len(cands)
            rec["reason"] = quote.missing_reason
            # PER-SOURCE outcome table: which venue answered, which failed,
            # and with which code -- the candidate-level diagnostic itself.
            try:
                rec["candidate_summary"] = qsvc.candidate_summary(
                    cands, mint=c["mint"])
            except Exception as exc:  # noqa: BLE001 - probe must not die
                rec["candidate_summary"] = f"unavailable: {type(exc).__name__}"
            if quote.available and quote.effective_price_usd and mark.value_usd:
                rec["gap_pct"] = round(
                    (quote.effective_price_usd / mark.value_usd - 1) * 100, 2)
            rec["decision_ms"] = round((time.monotonic() - t_signal) * 1000, 1)
        except Exception as exc:  # noqa: BLE001 - the probe must not die
            rec["outcome"] = "EXCEPTION"
            rec["reason"] = f"{type(exc).__name__}: {exc}"[:200]
            rec["decision_ms"] = round((time.monotonic() - t_signal) * 1000, 1)
        return rec

    if args.concurrency > 1:
        # Bounded fan-out: never unbounded, never a sequential queue.
        sem = asyncio.Semaphore(args.concurrency)

        async def guarded(c: dict) -> dict:
            async with sem:
                return await one(c)

        rows = list(await asyncio.gather(*(guarded(c) for c in queue)))
    else:
        for c in queue:
            rows.append(await one(c))
            if args.pace:
                await asyncio.sleep(args.pace)

    rpc_calls = getattr(rpc, "calls", 0) - rpc_calls_before
    quoted = [r for r in rows if r.get("outcome") == "QUOTED"]
    # Availability CONDITIONAL on the token having a readable mint account: the
    # pool that the quote path is actually responsible for.
    eligible = [r for r in rows if r.get("outcome") not in (
        "NO_MARK", "DECIMALS_ABSENT", "DECIMALS_CAPACITY", "DECIMALS_RAISED")]
    cond = round(100 * len(quoted) / max(1, len(eligible)), 1)
    noquote = [r for r in rows if r.get("outcome") == "NO_QUOTE"]
    rl = [r for r in rows if "rate limited" in str(r.get("reason", "")).lower()]

    summary = {
        "label": args.label,
        "probed": len(rows),
        "pace_s": args.pace,
        "candidate_concurrency": args.concurrency,
        "rpc_calls": rpc_calls,
        "rpc_calls_per_candidate": round(rpc_calls / max(1, len(rows)), 2),
        "quote_availability_pct": round(100 * len(quoted) / max(1, len(rows)), 1),
        "eligible_after_decimals": len(eligible),
        "quote_availability_conditional_pct": cond,
        "decimals_absent": sum(1 for r in rows
                               if r.get("outcome") == "DECIMALS_ABSENT"),
        "decimals_capacity": sum(1 for r in rows
                                 if r.get("outcome") == "DECIMALS_CAPACITY"),
        "no_quote": len(noquote),
        "rpc_rate_limited": len(rl),
        "rpc_rate_limit_reported_as_no_quote_pct": round(
            100 * len(rl) / max(1, len(noquote)), 1) if noquote else 0.0,
        "quote_latency_ms": dist([r["quote_ms"] for r in rows if "quote_ms" in r]),
        "decision_latency_ms": dist(
            [r["decision_ms"] for r in rows if "decision_ms" in r]),
        "mark_latency_ms": dist([r["mark_ms"] for r in rows if "mark_ms" in r]),
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"quote_capacity_{args.label}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2))

    print(f"\n=== [{args.label}] SUMMARY ===")
    for k in ("probed", "rpc_calls", "rpc_calls_per_candidate",
              "quote_availability_pct", "eligible_after_decimals",
              "quote_availability_conditional_pct", "decimals_absent",
              "decimals_capacity", "no_quote", "rpc_rate_limited",
              "rpc_rate_limit_reported_as_no_quote_pct"):
        print(f"  {k:44} {summary[k]}")
    for k in ("quote_latency_ms", "decision_latency_ms", "mark_latency_ms"):
        print(f"  {k:44} {summary[k]}")
    print(f"  report -> {REPORTS / f'quote_capacity_{args.label}.json'}")

    if rl:
        print("\n  capacity exhaustion still reported as 'no executable quote':")
        for r in rl[:4]:
            print(f"    {r['symbol'][:10]:10} {str(r.get('reason'))[:90]}")


if __name__ == "__main__":
    asyncio.run(main())