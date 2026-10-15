"""READ-ONLY quote round-trip probe: BUY, then SELL the BUY's own output.

WHAT THIS MEASURES
==================

For each ELIGIBLE real market candidate it does exactly what the V2 FLOOR (BETA)
ALL-IN entry/exit pair would do, and nothing more:

1. read the observed mark and the token's decimals;
2. request an executable **BUY** quote for the deployable notional the ALL-IN
   rule would size (available cash MINUS the outside-swap reserve);
3. request an executable **SELL** quote for the EXACT raw token quantity that
   BUY quote says it would acquire -- through the SAME
   ``quote_sell_pinned``/``quote_sell`` selection the beta exit uses;
4. validate the pair and record what it says.

WHAT IT DOES **NOT** DO
=======================

It builds NOTHING, signs NOTHING, simulates NOTHING, submits NOTHING and
broadcasts NOTHING. It never calls a send method. It writes to an ISOLATED
temporary state root, so the operator's ``var/`` is not touched.

The result is a **QUOTE-ONLY ESTIMATE, NOT A FILLED TRADE.** A quote is a
provider's statement about the price it would charge for a size at the instant
it was asked. It is not an execution, it is not a settlement, and the round-trip
return it implies is arithmetic on two quotes -- it is not P&L.

WHY THE VALIDATION FIELDS EXIST
===============================

A round trip is only meaningful if both legs describe the SAME market for the
SAME token and the SELL asks for the quantity the BUY would actually have
acquired. So each record carries, explicitly: the venue/venue id and pool of
both legs, both mints, the exact raw amounts, both legs' freshness, both legs'
measured latency, both legs' price impact against the limit the INTENDED
transaction would apply, the intended slippage/minimum-output settings (which
PAPER never applies, because it builds no transaction), and the SOL/USD
reference with its source and timestamp.

Usage:

    PYTHONPATH=src .venv/bin/python scripts/quote_roundtrip_probe.py \\
        --label allin_roundtrip --n 6 --pace 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from autopaper.config import build_config
from autopaper.engine import PaperTrader
from autopaper.paper_live.execution_costs import (
    deployable_sol_lamports,
    outside_swap_costs,
    slippage_boundary,
)

REPO = Path(__file__).resolve().parents[1]
CANDIDATES = REPO / "var" / "candidates.jsonl"
REPORTS = REPO / "reports" / "research"

#: Every record is stamped with this, so a consumer can never mistake a
#: quote-only estimate for a filled trade.
KIND = "QUOTE_ONLY_ESTIMATE_NOT_A_FILL"


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
        "max": round(max(values), 1) if values else None,
        "mean": round(st.mean(values), 1) if values else None,
    }


def _load(line: str) -> dict | None:
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None
    mint = str(rec.get("mint") or rec.get("token_mint") or "")
    if not mint:
        return None
    rec["mint"] = mint
    rec["symbol"] = rec.get("symbol") or "?"
    return rec


def recent_mints(limit: int) -> list[dict]:
    """The most recent DISTINCT mints from the candidate feed (READ-ONLY)."""
    if not CANDIDATES.exists():
        return []
    lines = CANDIDATES.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: list[dict] = []
    seen: set[str] = set()
    for line in reversed(lines):
        rec = _load(line)
        if rec is None or rec["mint"] in seen:
            continue
        seen.add(rec["mint"])
        out.append(rec)
        if len(out) >= limit:
            break
    return out


def _iso_config():
    """A config whose WRITE paths point at a throwaway directory.

    The RPC endpoint and every other read-only setting still come from the
    operator's ``.env.local`` (the probe reads chain state); nothing can be
    written near the live state, and the beta experiment the bridge builds sees
    an EMPTY store instead of the operator's 51 historical trades.
    """
    cfg = build_config()
    iso = Path(tempfile.mkdtemp(prefix="quote_roundtrip_probe_"))
    return replace(
        cfg,
        var_dir=iso,
        log_dir=iso / "logs",
        state_file=iso / "paper_state.json",
        ledger_file=iso / "ledger.jsonl",
        decisions_file=iso / "decisions.jsonl",
        updates_file=iso / "position_updates.jsonl",
        control_file=iso / "control.json",
        candidates_file=CANDIDATES,
    )


def _leg(point) -> dict:
    """One quote leg, verbatim, for the report."""
    return {
        "available": bool(getattr(point, "available", False)),
        "missing_reason": getattr(point, "missing_reason", None),
        "side": getattr(point, "side", None),
        "provider": getattr(point, "provider", None),
        "market_type": getattr(point, "market_type", None),
        "venue": getattr(point, "venue", None),
        "venue_id": getattr(point, "venue_id", None),
        "pool_or_curve": getattr(point, "pool_or_curve", None),
        "route": list(getattr(point, "route", ()) or ()),
        "token_mint": getattr(point, "token_mint", None),
        "quote_mint": getattr(point, "quote_mint", None),
        "identity": getattr(point, "identity", None),
        "input_amount_raw": getattr(point, "input_amount_raw", None),
        "input_decimals": getattr(point, "input_decimals", None),
        "input_amount_decimal": (
            None if getattr(point, "input_amount_decimal", None) is None
            else float(point.input_amount_decimal)),
        "output_amount_raw": getattr(point, "output_amount_raw", None),
        "output_decimals": getattr(point, "output_decimals", None),
        "output_amount_decimal": (
            None if getattr(point, "output_amount_decimal", None) is None
            else float(point.output_amount_decimal)),
        "price_impact_pct": getattr(point, "price_impact_pct", None),
        "fee_bps": getattr(point, "fee_bps", None),
        "quote_timestamp": getattr(point, "quote_timestamp", None),
        "quote_age_ms": getattr(point, "quote_age_ms", None),
        "max_quote_age_ms": getattr(point, "max_quote_age_ms", None),
        "stale": bool(getattr(point, "stale", False)),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="roundtrip")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--pace", type=float, default=1.0,
                    help="seconds between candidates")
    ap.add_argument("--notional-usd", type=float, default=10.0,
                    help="the ALL-IN arm's available cash (the beta default)")
    ap.add_argument("--max-price-impact-pct", type=float, default=5.0,
                    help="the limit the INTENDED transaction would apply")
    args = ap.parse_args()

    trader = PaperTrader(_iso_config())
    engine = await trader._paper_bridge_instance()

    # PAPER-ONLY, PROVED RATHER THAN ASSUMED.
    from autopaper.paper_live.guards import assert_paper_only
    audit = assert_paper_only()
    forbidden = [n for n in ("sendTransaction", "sign_transaction",
                             "send_raw_transaction", "submit", "broadcast",
                             "send_transaction")
                 if hasattr(engine, n)]
    if forbidden:  # pragma: no cover - a send line here would abort the probe
        raise SystemExit(f"REFUSING TO CONTINUE: engine exposes {forbidden}")

    ref = engine.sol_reference
    sol_usd = float(ref.usd_per_sol)
    costs = outside_swap_costs(sol_usd=sol_usd,
                               sol_usd_source=str(getattr(ref, "source", "")),
                               sol_usd_at=getattr(ref, "at", None))
    sol_lamports, why = deployable_sol_lamports(
        available_usd=args.notional_usd, sol_usd=sol_usd, costs=costs)

    qsvc = engine.quote_service
    reader = qsvc._reader
    rpc = reader.rpc

    print(f"[{args.label}] KIND={KIND}")
    print(f"[{args.label}] SOL/USD={sol_usd:.2f} source={getattr(ref,'source','')!r} "
          f"at={getattr(ref,'at',None)} endpoint={getattr(rpc,'url','?')}")
    print(f"[{args.label}] available=${args.notional_usd:.4f} "
          f"reserve={costs.total_reserve_lamports} lamports "
          f"deployable={sol_lamports} lamports ({why or 'ok'})")
    print(f"[{args.label}] paper_audit={audit} "
          f"forbidden_attrs={forbidden or 'none'}\n")

    rows: list[dict] = []
    rpc_before = getattr(rpc, "calls", 0)
    for cand in recent_mints(args.n):
        rec: dict = {"symbol": cand["symbol"], "mint": cand["mint"],
                     "kind": KIND}
        try:
            t0 = time.monotonic()
            mark = await engine.observe(cand["mint"])
            rec["mark_ms"] = round((time.monotonic() - t0) * 1000, 1)
            rec["mark_available"] = bool(mark.available)
            rec["mark_usd"] = getattr(mark, "value_usd", None)
            rec["mark_pair_address"] = getattr(mark, "pair_address", None)
            rec["mark_timestamp"] = getattr(mark, "timestamp", None)
            rec["mark_reason"] = getattr(mark, "missing_reason", None)
            if not mark.available or not mark.value_usd:
                rec["outcome"] = "NO_MARK"
                rows.append(rec)
                if args.pace:
                    await asyncio.sleep(args.pace)
                continue

            dec = await reader.mint_decimals(cand["mint"])
            rec["token_decimals"] = dec
            if not dec:
                rec["outcome"] = "DECIMALS_ABSENT"
                rows.append(rec)
                if args.pace:
                    await asyncio.sleep(args.pace)
                continue

            # -- LEG 1: THE BUY, at the ALL-IN deployable size --------------
            t1 = time.monotonic()
            buy, cands = await qsvc.quote_buy_best(
                mint=cand["mint"], sol_lamports=sol_lamports,
                token_decimals=int(dec),
                observed_pair_address=mark.pair_address)
            rec["buy_request_ms"] = round((time.monotonic() - t1) * 1000, 1)
            rec["buy"] = _leg(buy)
            rec["buy_candidates_enumerated"] = len(cands)
            rec["outcome"] = "BUY_QUOTED" if buy.available else "BUY_UNAVAILABLE"
            if not buy.available or not buy.output_amount_raw:
                rows.append(rec)
                if args.pace:
                    await asyncio.sleep(args.pace)
                continue

            # -- LEG 2: THE SELL for the EXACT acquired raw quantity -------
            # THE SAME SELECTION THE BETA EXIT USES. ``quote_sell_pinned``
            # returns (chosen, counterfactual); the CHOSEN quote is the leg.
            sell_raw = int(buy.output_amount_raw)
            t2 = time.monotonic()
            pinned = getattr(qsvc, "quote_sell_pinned", None)
            if buy.pool_or_curve and pinned is not None:
                result = await pinned(
                    mint=cand["mint"], token_amount_raw=sell_raw,
                    token_decimals=int(dec), pinned_venue_id=None,
                    pinned_pool_address=buy.pool_or_curve,
                    observed_pair_address=buy.pool_or_curve)
                sell = result[0] if isinstance(result, (tuple, list)) else result
                rec["sell_path"] = "quote_sell_pinned"
                rec["sell_returned_pair"] = isinstance(result, (tuple, list))
            else:
                sell = await qsvc.quote_sell(
                    mint=cand["mint"], token_amount_raw=sell_raw,
                    token_decimals=int(dec),
                    observed_pair_address=buy.pool_or_curve)
                rec["sell_path"] = "quote_sell"
                rec["sell_returned_pair"] = False
            rec["sell_request_ms"] = round((time.monotonic() - t2) * 1000, 1)
            rec["sell"] = _leg(sell)
            rec["outcome"] = "ROUND_TRIP_QUOTED" if sell.available else "SELL_UNAVAILABLE"

            # -- VALIDATION ------------------------------------------------
            v: dict = {
                "sell_asks_the_exact_acquired_quantity": (
                    sell.available
                    and int(sell.input_amount_raw or -1) == sell_raw),
                "same_token_mint": (
                    bool(buy.token_mint) and buy.token_mint == sell.token_mint),
                "quote_asset_is_the_quote_asset": (
                    bool(buy.quote_mint) and buy.quote_mint == sell.quote_mint),
                "same_venue_id": (
                    bool(buy.venue_id) and buy.venue_id == sell.venue_id),
                "same_pool_or_curve": (
                    bool(buy.pool_or_curve)
                    and buy.pool_or_curve == sell.pool_or_curve),
                "buy_identity": buy.identity,
                "sell_identity": getattr(sell, "identity", None),
                "buy_direction_is_BUY": buy.side == "BUY",
                "sell_direction_is_SELL": (
                    sell.available and sell.side == "SELL"),
                "buy_fresh": not bool(getattr(buy, "stale", False)),
                "sell_fresh": not bool(getattr(sell, "stale", False)),
            }
            if buy.available and sell.available:
                sol_in = float(buy.input_amount_decimal or 0.0)
                sol_out = float(sell.output_amount_decimal or 0.0)
                v["sol_in"] = sol_in
                v["sol_out"] = sol_out
                v["gross_round_trip_return_pct"] = (
                    round((sol_out / sol_in - 1.0) * 100.0, 6)
                    if sol_in > 0 else None)
                impacts = [abs(float(x.price_impact_pct))
                           for x in (buy, sell)
                           if x.price_impact_pct is not None]
                v["max_abs_price_impact_pct"] = (
                    round(max(impacts), 6) if impacts else None)
                v["would_breach_intended_impact_limit"] = (
                    None if not impacts else
                    max(impacts) > args.max_price_impact_pct)
            v["same_market_ok"] = all(bool(v[k]) for k in (
                "sell_asks_the_exact_acquired_quantity", "same_token_mint",
                "quote_asset_is_the_quote_asset", "same_venue_id",
                "same_pool_or_curve", "buy_direction_is_BUY",
                "sell_direction_is_SELL", "buy_fresh", "sell_fresh"))
            # THE INTENDED BOUNDARY, AND THE FACT THAT PAPER APPLIES NONE.
            v["slippage_and_minimum_output"] = slippage_boundary(
                measured_price_impact_pct=v.get("max_abs_price_impact_pct"),
                max_price_impact_pct=args.max_price_impact_pct)
            v["sol_usd"] = sol_usd
            v["sol_usd_source"] = getattr(ref, "source", "")
            v["sol_usd_at"] = getattr(ref, "at", None)
            v["sol_usd_age_ms"] = ref.age_ms()
            v["no_transaction_built"] = True
            v["no_transaction_signed"] = True
            v["no_transaction_submitted"] = True
            v["no_transaction_broadcast"] = True
            v["is_a_filled_trade"] = False
            rec["validation"] = v
        except Exception as exc:  # noqa: BLE001 - the probe must not die
            rec["outcome"] = "EXCEPTION"
            rec["reason"] = f"{type(exc).__name__}: {exc}"[:200]
        rows.append(rec)
        if args.pace:
            await asyncio.sleep(args.pace)

    rpc_calls = getattr(rpc, "calls", 0) - rpc_before
    buys = [r for r in rows if r.get("outcome") in
            ("BUY_QUOTED", "SELL_UNAVAILABLE", "ROUND_TRIP_QUOTED")]
    rts = [r for r in rows if r.get("outcome") == "ROUND_TRIP_QUOTED"]
    ok_market = [r for r in rts if r["validation"]["same_market_ok"]]
    returns = [r["validation"]["gross_round_trip_return_pct"] for r in rts
               if r["validation"].get("gross_round_trip_return_pct") is not None]
    buy_lat = [r["buy_request_ms"] for r in rows if "buy_request_ms" in r]
    sell_lat = [r["sell_request_ms"] for r in rows if "sell_request_ms" in r]
    summary = {
        "kind": KIND,
        "label": args.label,
        "quote_only_estimate_not_a_fill": True,
        "transaction_built": False,
        "transaction_signed": False,
        "transaction_submitted": False,
        "transaction_broadcast": False,
        "probed": len(rows),
        "rpc_calls": rpc_calls,
        "all_in_notional_usd": args.notional_usd,
        "deployable_sol_lamports": sol_lamports,
        "outside_swap_reserve_lamports": costs.total_reserve_lamports,
        "outside_swap_charged_lamports": costs.non_recoverable_lamports,
        "sol_usd": sol_usd,
        "sol_usd_source": getattr(ref, "source", ""),
        "sol_usd_at": getattr(ref, "at", None),
        "sol_usd_age_ms_at_finish": ref.age_ms(),
        "buy_quoted": len(buys),
        "round_trips_quoted": len(rts),
        "sell_unavailable": sum(1 for r in rows
                                if r.get("outcome") == "SELL_UNAVAILABLE"),
        "same_market_validated": len(ok_market),
        "sell_pinned_path_used": sum(1 for r in rows
                                     if r.get("sell_path") == "quote_sell_pinned"),
        "sell_pair_unwrapped": sum(1 for r in rows
                                   if r.get("sell_returned_pair")),
        "buy_latency_ms": dist(buy_lat),
        "sell_latency_ms": dist(sell_lat),
        "gross_round_trip_return_pct": dist(returns),
        "outcomes": {k: sum(1 for r in rows if r.get("outcome") == k)
                     for k in sorted({str(r.get("outcome")) for r in rows})},
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"quote_roundtrip_{args.label}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print(f"=== [{args.label}] SUMMARY (QUOTE-ONLY ESTIMATE, NOT A FILL) ===")
    for k, val in summary.items():
        if k in ("kind", "label"):
            continue
        print(f"  {k:34} {val}")
    print(f"  report -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
