"""
Single-mint market-identity diagnostic (READ-ONLY).

Answers, for ONE token, the exact question the ENTRY identity gate asks:

    mint
      -> what did the candidate disclose (pool_or_curve per source)?
      -> what does provider-native discovery derive from the mint?
      -> is the derived market chain-verified for this mint?
      -> does the observed (DexScreener) pair equal the executable market?
      -> what is the BUY quote / identity verdict?

It trades nothing, signs nothing and broadcasts nothing. It only READS chain
state and writes a JSON report. Used to decide whether an
``ENTRY_REJECTED_IDENTITY_UNKNOWN`` was:

    A. the canonical market genuinely could not be found, or
    B. the canonical market exists but the candidate did not carry it.

    PYTHONPATH=src .venv/bin/python scripts/market_identity_probe.py \
        --mint <MINT> --label <name> [--sol 0.005]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from autopaper.config import build_config
from autopaper.engine import PaperTrader
from autopaper.same_market.models import WSOL_MINT
from autopaper.same_market.reader import derive_bonding_curve, derive_canonical_pool

REPORTS = Path("reports/research")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mint", required=True)
    ap.add_argument("--label", default="identity")
    ap.add_argument("--sol", type=float, default=0.005)
    ap.add_argument("--observed-pair", default=None,
                    help="force the observed pair (e.g. the retired bonding "
                         "curve) to reproduce the historical decision)")
    args = ap.parse_args()
    mint = args.mint

    trader = PaperTrader(build_config())
    engine = await trader._paper_bridge_instance()
    ref = engine.sol_reference
    lamports = int(args.sol / float(ref.usd_per_sol) * 1e9)
    qsvc = engine.quote_service
    reader = qsvc._reader
    jupiter = getattr(qsvc, "jupiter", None)

    out: dict = {
        "mint": mint,
        "label": args.label,
        "sol_lamports": lamports,
        "sol_usd": float(ref.usd_per_sol),
        "jupiter_configured": jupiter is not None,
    }

    # -- 1. observed mark (the layer that supplies observed_pair_address) ---
    t0 = time.monotonic()
    mark = await engine.observe(mint)
    out["observed_mark"] = {
        "available": bool(mark.available),
        "value_usd": mark.value_usd,
        "pair_address": mark.pair_address,
        "dex_id": mark.dex_id,
        "liquidity_usd": mark.liquidity_usd,
        "missing_reason": mark.missing_reason,
        "latency_ms": round((time.monotonic() - t0) * 1000, 1),
    }

    # -- 2. provider-native discovery, purely from the mint -----------------
    curve_addr, _ = derive_bonding_curve(mint)
    pool_addr, _ = derive_canonical_pool(mint, WSOL_MINT)
    out["derived"] = {"bonding_curve": curve_addr, "canonical_pool": pool_addr}

    decimals = await reader.mint_decimals(mint)
    out["mint_decimals"] = decimals
    out["mint_decimals_error"] = getattr(reader, "last_mint_decimals_error", "")

    try:
        curve, snap = await reader.read_bonding_curve(mint)
        out["bonding_curve"] = {
            "found": curve is not None,
            "address": curve.address if curve else None,
            "graduated": bool(getattr(curve, "graduated", False)) if curve else None,
            "detail": snap.detail,
        }
    except Exception as exc:  # noqa: BLE001 - probe must not die
        out["bonding_curve"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}

    try:
        pool, snap2, address = await reader.read_pumpswap_pool(
            mint, pool_address=out["observed_mark"]["pair_address"])
        out["pumpswap_pool"] = {
            "found": pool is not None,
            "address": address,
            "verified_for_mint": bool(pool is not None),
            "detail": snap2.detail if pool is None else None,
            "equals_derived": address == pool_addr,
        }
    except Exception as exc:  # noqa: BLE001 - probe must not die
        out["pumpswap_pool"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}

    # -- 3. the raw Jupiter route: what pool key does it actually disclose? -
    if jupiter is not None and decimals:
        try:
            res = await jupiter.quote_buy(
                token_mint=mint, quote_asset_mint=WSOL_MINT,
                in_amount=lamports, output_decimals=decimals)
            out["jupiter_route"] = {
                "hops": len(res.route_plan),
                "labels": [h.label for h in res.route_plan],
                "amm_keys": list(res.pool_keys),
                "out_amount": int(res.out_amount),
            }
        except Exception as exc:  # noqa: BLE001 - probe must not die
            out["jupiter_route"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}

    # -- 4. the full candidate enumeration the ENTRY gate actually sees ------
    observed = args.observed_pair or out["observed_mark"]["pair_address"]
    out["observed_pair_used"] = observed
    if decimals:
        quote, cands = await qsvc.quote_buy_best(
            mint=mint, sol_lamports=lamports, token_decimals=decimals,
            observed_pair_address=observed)
        out["buy_quote"] = {
            "available": bool(quote.available),
            "provider": quote.provider,
            "market_type": quote.market_type,
            "pool_or_curve": quote.pool_or_curve,
            "identity": quote.identity,
            "identity_evidence": quote.identity_evidence,
            "observed_pair_address": quote.observed_pair_address,
            "missing_reason": quote.missing_reason,
            "effective_price_usd": quote.effective_price_usd,
        }
        out["candidates"] = [
            {
                "provider": c.provider,
                "available": c.available,
                "pool_or_curve": c.pool_or_curve,
                "venue_id": c.venue_id,
                "identity": c.identity,
                "identity_evidence": c.identity_evidence,
                "missing_reason": c.missing_reason,
            }
            for c in cands
        ]
        try:
            out["candidate_summary"] = qsvc.candidate_summary(cands, mint=mint)
        except Exception as exc:  # noqa: BLE001
            out["candidate_summary"] = f"unavailable: {type(exc).__name__}"

    # -- 5. the A-vs-B verdict, computed from facts, not guessed ------------
    pool_found = bool(out.get("pumpswap_pool", {}).get("verified_for_mint"))
    out["verdict"] = {
        "observed_pair_is_derived_curve": observed == curve_addr,
        "observed_pair_is_derived_pool": observed == pool_addr,
        "canonical_market_exists": pool_found or out["bonding_curve"].get("found", False),
        "canonical_pool_derived_from_mint": pool_addr,
        "canonical_pool_chain_verified": pool_found,
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"market_identity_{args.label}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nreport -> {path}")


if __name__ == "__main__":
    asyncio.run(main())
