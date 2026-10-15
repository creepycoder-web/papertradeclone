"""
LIVE read-only verification of the promoted execution-native guard.

Trades nothing, signs nothing, broadcasts nothing. Reads chain state and the
public price API, and writes a JSON report. No threshold is modified here.

Verifies, on live candidates:
  * direct same-market quoting works (no crash)
  * identity == PASS  (observed market IS the executable market)
  * the native gap and the legacy gap side by side on the SAME record
  * quote latency percentiles
"""
import asyncio, json, statistics as st, sys, time
sys.path.insert(0, "src")
from autopaper.clock import SystemClock
from autopaper.config import build_config
from autopaper.engine import PaperTrader
from autopaper.market import MarketDataService
from autopaper.same_market import pumpdirect as PD

GUARD = 10.0


async def main():
    rows = [json.loads(l) for l in open("var/candidates.jsonl") if l.strip()]
    seen, mints = set(), []
    for r in reversed(rows):
        if r["mint"] not in seen:
            seen.add(r["mint"]); mints.append((r["mint"], r.get("symbol") or "?"))
        if len(mints) >= 16:
            break

    tr = PaperTrader(build_config())
    eng = await tr._paper_bridge_instance()
    qs, rd = eng.quote_service, eng.quote_service._reader
    md = MarketDataService(build_config().market, clock=SystemClock())
    sol = float(eng.sol_reference.usd_per_sol)
    L = int(0.50 / sol * 1e9)

    out = []
    for mint, sym in mints:
        rec = {"symbol": sym, "mint": mint}
        try:
            t0 = time.monotonic()
            dec = await rd.mint_decimals(mint)
            if not dec:
                rec["outcome"] = "DECIMALS_UNKNOWN"; out.append(rec); continue
            mark = await md.quote_async(mint)
            obs = mark.observation
            mp = getattr(obs, "price_usd", None)
            # Pass the OBSERVED pair exactly as production does: identity is
            # proven by observed_pair_address == market_address, so with None
            # it can never be PASS and the probe would under-report.
            pair = getattr(obs, "pair_address", None) or getattr(obs, "pool_address", None)
            rec["observed_pair"] = pair
            rec["mark_age_status"] = getattr(mark.observation, "age_status", None)
            q = await qs.quote_buy(mint=mint, sol_lamports=L, token_decimals=dec,
                                   observed_pair_address=pair)
            rec["quote_ms"] = round((time.monotonic() - t0) * 1000, 1)
            rec["provider"] = q.provider
            rec["market_type"] = q.market_type
            rec["identity"] = q.identity
            rec["pool_or_curve"] = q.pool_or_curve
            rec["venue_id"] = q.venue_id
            rec["price_impact_pct"] = (float(q.price_impact_pct)
                                       if q.price_impact_pct is not None else None)
            rec["quote_age_ms"] = q.quote_age_ms
            rec["quote_latency_ms"] = q.quote_latency_ms
            rec["mark_price_usd"] = mp
            nat = getattr(q, "execution_native_reference", None)
            rec["native_present"] = nat is not None
            if nat is not None:
                rec["native_reference_price"] = float(nat.reference_price)
                rec["native_market_id"] = nat.market_id
                rec["native_method"] = nat.calculation_method
                rec["same_market"] = (nat.market_id == q.venue_id)
                rec["exec_price"] = float(q.effective_price)
                rec["native_gap_pct"] = round(
                    (float(q.effective_price) / float(nat.reference_price) - 1) * 100, 3)
                rec["would_reject_native"] = rec["native_gap_pct"] > GUARD
            if q.available and q.effective_price_usd and mp:
                rec["legacy_gap_pct"] = round(
                    (q.effective_price_usd / mp - 1) * 100, 3)
                rec["would_reject_legacy"] = rec["legacy_gap_pct"] > GUARD
            rec["outcome"] = "QUOTED" if q.available else "NO_QUOTE"
        except Exception as exc:
            rec["outcome"] = "EXCEPTION"
            rec["error"] = f"{type(exc).__name__}: {exc}"[:150]
        out.append(rec)
        await asyncio.sleep(0.6)

    json.dump(out, open("reports/research/native_gap_live_probe.json", "w"),
              indent=2, default=str)

    q = [r for r in out if r["outcome"] == "QUOTED"]
    ident = [r for r in q if r["identity"] == "PASS"]
    lat = sorted(r["quote_ms"] for r in out if r.get("quote_ms"))
    ng = sorted(r["native_gap_pct"] for r in q if r.get("native_gap_pct") is not None)
    lg = sorted(r["legacy_gap_pct"] for r in q if r.get("legacy_gap_pct") is not None)

    def pc(v, k):
        return round(v[min(len(v) - 1, int(k * (len(v) - 1)))], 2) if v else None

    print(f"\n  probed {len(out)} | quoted {len(q)} | identity PASS {len(ident)}"
          f" | errors {sum(1 for r in out if r['outcome']=='EXCEPTION')}")
    if q:
        print(f"  quote latency ms : p50 {pc(lat,.5)}  p95 {pc(lat,.95)}  max {round(max(lat),1)}")
    if ng:
        print(f"  NATIVE gap  %     : p50 {pc(ng,.5)}  p90 {pc(ng,.9)}  max {round(max(ng),2)}"
              f"   >{GUARD}%: {sum(1 for x in ng if x>GUARD)}/{len(ng)}")
    if lg:
        print(f"  LEGACY gap  %     : p50 {pc(lg,.5)}  p90 {pc(lg,.9)}  max {round(max(lg),2)}"
              f"   >{GUARD}%: {sum(1 for x in lg if x>GUARD)}/{len(lg)}")
    agree = sum(1 for r in q if r.get("would_reject_native") == r.get("would_reject_legacy"))
    print(f"  native/legacy verdict agreement: {agree}/{len(q)}")
    print("  -> reports/research/native_gap_live_probe.json")

asyncio.run(main())
