"""
READ-ONLY mark-vs-executable forensics.

Answers the one question the incident report cannot: is the 10% guard refusing
GENUINELY bad execution, or is the mark/market matching producing FALSE
rejections?

For each live candidate this measures, on the SAME pair and the SAME order
size, at (almost) the same instant:
  observed DexScreener mark  vs  the executable quote

Nothing is traded, signed or broadcast. It only READS chain state and a public
price API, and writes a JSON report.
"""

from __future__ import annotations

import asyncio
import json
import statistics as st
import time
from pathlib import Path

from autopaper.config import build_config
from autopaper.engine import PaperTrader

NOTIONAL_USD = 0.50
#: 16 rpm is the production bridge's own ceiling (public_rpc_max_concurrency
#: 4 * 4). ~8s between candidates keeps a multi-call probe under it.
PACE_S = float(__import__("os").environ.get("PROBE_PACE_S", "9"))
CANDIDATES = Path("var/candidates.jsonl")
OUT = Path("reports/research/mark_gap_forensics.json")


def recent_mints(limit: int = 8) -> list[dict]:
    rows = [json.loads(l) for l in CANDIDATES.read_text().splitlines() if l.strip()]
    seen, out = set(), []
    for r in reversed(rows):
        m = r.get("mint")
        if not m or m in seen:
            continue
        seen.add(m)
        out.append({"mint": m, "symbol": r.get("symbol") or "?",
                    "fields": r.get("fields") or {}})
        if len(out) >= limit:
            break
    return out


async def main() -> None:
    # The PRODUCTION bridge, so this measures the real wiring rather than a
    # hand-built engine that could differ from what actually trades.
    trader = PaperTrader(build_config())
    engine = await trader._paper_bridge_instance()
    snap_usd = float(engine.sol_reference.usd_per_sol)
    sol_lamports = int(NOTIONAL_USD / snap_usd * 1e9)
    print(f"  SOL/USD reference: {snap_usd:.3f}")

    rows = []
    for c in recent_mints():
        rec = {"symbol": c["symbol"], "mint": c["mint"]}
        try:
            mark = await engine.observe(c["mint"])
            rec["mark_price"] = mark.value_usd
            rec["mark_pair"] = mark.pair_address
            rec["mark_dex"] = mark.dex_id
            rec["mark_mcap"] = mark.market_cap_usd
            rec["mark_ts"] = mark.timestamp
            if not mark.available or not mark.value_usd:
                rec["error"] = f"mark unavailable: {mark.missing_reason}"
                rows.append(rec)
                await asyncio.sleep(PACE_S)
                continue

            reader = engine.quote_service._reader
            dec = await reader.mint_decimals(c["mint"])
            rec["decimals"] = dec
            t0 = time.time()
            quote, cands = await engine.quote_service.quote_buy_best(
                mint=c["mint"], sol_lamports=sol_lamports, token_decimals=dec,
                observed_pair_address=mark.pair_address)
            rec["quote_age_ms"] = round((time.time() - t0) * 1000, 1)
            rec["provider"] = quote.provider
            rec["market_type"] = quote.market_type
            rec["identity"] = quote.identity
            rec["exec_pool"] = quote.pool_or_curve
            rec["exec_price"] = quote.effective_price_usd
            rec["price_impact_pct"] = quote.price_impact_pct
            rec["venues_enumerated"] = len(cands)
            rec["quote_missing_reason"] = quote.missing_reason
            rec["n_identity_pass"] = sum(
                1 for x in cands if x.identity == "PASS" and x.available)
            rec["same_pool"] = (quote.pool_or_curve == mark.pair_address)
            rec["age_seconds"] = (c["fields"] or {}).get("age_seconds")

            if (not quote.available) and quote.missing_reason:
                rec["unquotable"] = True
            if quote.available and quote.effective_price_usd and mark.value_usd:
                gap = (quote.effective_price_usd / mark.value_usd - 1) * 100
                rec["gap_pct"] = round(gap, 4)
                rec["guard_10pct"] = "REJECT" if gap > 10.0 else "PASS"
            # The alert's OWN market cap is an independent price witness:
            # supply for a pump.fun launch is ~1e9, so alert_mcap/1e9 is a
            # price that never passed through our mark path.
            mc = (c["fields"] or {}).get("market_cap_usd")
            if mc and mark.value_usd:
                rec["alert_mcap"] = mc
                rec["alert_implied_supply"] = round(mc / mark.value_usd)
                if rec.get("gap_pct") is not None:
                    rec["exec_vs_alertimplied_pct"] = round(
                        (rec["exec_price"] / (mc / 1e9) - 1) * 100, 2)
                    rec["mark_vs_alertimplied_pct"] = round(
                        (mark.value_usd / (mc / 1e9) - 1) * 100, 2)
        except Exception as exc:  # noqa: BLE001 - forensics must not die
            rec["error"] = f"{type(exc).__name__}: {exc}"[:160]
        # Respect the engine's OWN public-RPC limiter. Unpaced, the probe trips
        # it and every later read fails as "rate limited" -- which would be
        # misread as "this market is unquotable".
        await asyncio.sleep(PACE_S)
        rows.append(rec)
        print(f"  {c['symbol'][:10]:10} gap={rec.get('gap_pct')} "
              f"ident={rec.get('identity')} same_pool={rec.get('same_pool')} "
              f"prov={rec.get('provider')} {rec.get('error','')}"[:150])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))

    gaps = [r["gap_pct"] for r in rows if r.get("gap_pct") is not None]
    print("\n=== SUMMARY ===")
    print(f"  probed            : {len(rows)}")
    print(f"  gap measured      : {len(gaps)}")
    print(f"  would REJECT >10% : {sum(1 for g in gaps if g > 10)}")
    if gaps:
        s = sorted(gaps)
        print(f"  median {st.median(s):.2f}%  max {max(s):.2f}%  min {min(s):.2f}%")
    rej = [r for r in rows if r.get("guard_10pct") == "REJECT"]
    print(f"\n=== REJECTED: is the MARK or the EXEC price the outlier? ===")
    for r in rej:
        print(f"  {r['symbol']:10} exec_vs_alert={r.get('exec_vs_alertimplied_pct')}% "
              f"mark_vs_alert={r.get('mark_vs_alertimplied_pct')}% "
              f"same_pool={r.get('same_pool')} liq={r.get('alert_mcap')}")
    unq=[r for r in rows if r.get("unquotable")]
    print(f"\n=== QUOTE AVAILABILITY (the real bottleneck) ===")
    print(f"  no executable quote : {len(unq)}/{len(rows)}")
    for r in unq[:6]:
        print(f"    {r['symbol'][:10]:10} age={r.get('age_seconds')}s :: "
              f"{str(r.get('quote_missing_reason'))[:95]}")
    print(f"\n  report -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
