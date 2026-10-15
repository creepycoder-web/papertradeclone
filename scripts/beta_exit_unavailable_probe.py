#!/usr/bin/env python
"""
READ-ONLY diagnosis of the V2 FLOOR (BETA) ``no executable SELL quote`` refusal.

WHAT IT ANSWERS, from the PROVIDER, not from the word "unavailable":

    1. does an executable SELL quote actually exist for the exact held quantity?
    2. which candidate venue produced it, and what identity verdict did the
       engine's own selection give it?
    3. is the token's entry bonding curve retired (graduated), and does the
       canonical PumpSwap pool derived from the mint carry the quote?
    4. does the existing graduated-exit migration resolve the same mint to a
       PASS-identified executable SELL?

It builds the SAME stack the PAPER runtime builds (SolanaReadOnlyRpc ->
DirectPumpReader -> QuoteService + JupiterQuoteClient) and only calls read
methods. It builds, signs and broadcasts nothing; there is no signer anywhere on
this path.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/beta_exit_unavailable_probe.py \
        [--state var/v2_floor_beta_state.json]

The RPC URL is read from the environment (RPC_URL / PUBLIC_RPC_URL) and is never
printed: only whether one was configured.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from autopaper.paper_live.quotes import QuoteService, SolReference
from autopaper.same_market.jupiter import JupiterQuoteClient
from autopaper.same_market.pumpdirect import derive_bonding_curve, derive_canonical_pool
from autopaper.same_market.rpc import SolanaReadOnlyRpc

WSOL = "So11111111111111111111111111111111111111112"


def _positions(state_path: Path) -> list[dict]:
    data = json.loads(state_path.read_text())
    out = []
    for rec in data.get("open_positions") or []:
        if rec.get("status") in ("unresolved", "open"):
            out.append(rec)
    return out


def _fmt(v, width: int = 0) -> str:
    s = "" if v is None else str(v)
    s = " ".join(s.split())
    if width and len(s) > width:
        s = s[: width - 1] + "\u2026"
    return s


def _show_candidate(label: str, point) -> None:
    if point is None:
        print(f"    {label:<20} NONE (no quote object)")
        return
    print(
        f"    {label:<20} available={bool(getattr(point,'available',False))} "
        f"stale={bool(getattr(point,'stale',False))} "
        f"identity={_fmt(getattr(point,'identity',None))} "
        f"out={_fmt(getattr(point,'output_amount_decimal',None))} "
        f"pool={_fmt(getattr(point,'pool_or_curve',None))} "
        f"provider={_fmt(getattr(point,'provider',None))}"
    )
    reason = getattr(point, "missing_reason", None)
    if reason:
        print(f"      missing_reason: {_fmt(reason, 200)}")
    evidence = getattr(point, "identity_evidence", None)
    if evidence:
        print(f"      identity_evidence: {_fmt(evidence, 200)}")


async def probe(state_path: Path, iterations: int, max_rpm: int) -> int:
    rpc_url = (os.environ.get("RPC_URL") or os.environ.get("PUBLIC_RPC_URL") or "").strip()
    print(f"RPC configured: {bool(rpc_url)} (url not printed)")
    rpc = SolanaReadOnlyRpc(rpc_url or None, timeout_s=10.0, max_rpm=max_rpm)
    jupiter = JupiterQuoteClient(timeout_s=12.0, cache_ttl_s=0.0)
    service = QuoteService(
        rpc=rpc,
        sol_reference=SolReference(150.0, time.time(), "probe-placeholder"),
        jupiter=jupiter,
        max_quote_age_ms=30_000.0,
        allow_jupiter_fallback=True,
    )

    positions = _positions(state_path)
    print(f"open/unresolved positions in {state_path}: {len(positions)}\n")

    for rec in positions:
        mint = rec["mint"]
        symbol = rec.get("symbol")
        arm = rec.get("arm")
        tokens_raw = int(rec.get("entry_tokens_raw") or 0)
        decimals = int(rec.get("entry_token_decimals") or 6)
        entry_curve = rec.get("entry_pool_or_curve")
        entry_venue = (rec.get("entry_provenance") or {}).get("venue_id") or ""
        derived_curve, _ = derive_bonding_curve(mint)
        derived_pool, _ = derive_canonical_pool(mint, WSOL)

        print("=" * 86)
        print(f"ARM {arm}  {symbol}  mint={mint}")
        print(f"  held quantity      : {tokens_raw} raw  ({decimals} decimals)")
        print(f"  entry market (rec) : {entry_curve}")
        print(f"  derived curve PDA  : {derived_curve}  (match={derived_curve == entry_curve})")
        print(f"  derived canon pool : {derived_pool}")
        print(f"  recorded venue_id  : {entry_venue or '(none)'}")

        # -- 1. the chain state of the entry market -------------------------
        try:
            curve, _snap = await service._reader.read_bonding_curve(mint)
        except Exception as exc:  # noqa: BLE001 - diagnostic
            curve = None
            print(f"  read_bonding_curve : ERROR {type(exc).__name__}: {_fmt(exc, 120)}")
        if curve is not None:
            print(
                f"  bonding curve      : address={curve.address} complete="
                f"{getattr(curve,'complete',None)} graduated={getattr(curve,'graduated',None)} "
                f"virtual_token_reserves={getattr(curve,'virtual_token_reserves',None)}"
            )
        else:
            print("  bonding curve      : none returned")

        # -- 2. the candidate enumeration the engine performs --------------
        print("  --- quote_candidates(SELL) AS THE ENGINE CALLS IT ---")
        for i in range(iterations):
            if i:
                print("  --- retry ---")
            cands = await service.quote_candidates(
                side="SELL", mint=mint, amount_raw=tokens_raw,
                token_decimals=decimals, observed_pair_address=entry_curve)
            for c in cands:
                _show_candidate(_fmt(getattr(c, "provider", "?"), 20), c)

            # -- 3. the settlement selection the engine performs -----------
            chosen, counter = service._select_best_sell_executable(
                cands, pinned_venue_id=None, mint=mint)
            print(f"    chosen             : {_fmt(getattr(chosen,'provider',None))} "
                  f"identity={_fmt(getattr(chosen,'identity',None))} "
                  f"available={bool(getattr(chosen,'available',False))}")
            if counter is not None:
                print(f"    counterfactual     : {_fmt(getattr(counter,'provider',None))} "
                      f"identity={_fmt(getattr(counter,'identity',None))}")

            # -- 4. the actual beta-path call ------------------------------
            pair = await service.quote_sell_pinned(
                mint=mint, token_amount_raw=tokens_raw, token_decimals=decimals,
                pinned_venue_id=None, pinned_pool_address=entry_curve,
                observed_pair_address=entry_curve)
            q = pair[0] if isinstance(pair, (tuple, list)) else pair
            print("  --- quote_sell_pinned(AS _beta_exit_fill CALLS IT) ---")
            _show_candidate("pinned[0]", q)

        # -- 5. the direct migrated-market route ---------------------------
        print("  --- resolve_graduated_exit_venue (the EXISTING migration) ---")
        migrated = await service.resolve_graduated_exit_venue(
            mint=mint, token_amount_raw=tokens_raw, token_decimals=decimals,
            entry_venue_id=f"PUMP_BONDING_CURVE:{entry_curve}",
            entry_pool_address=entry_curve, observed_pair_address=entry_curve)
        _show_candidate("migrated", migrated)
        vm = getattr(migrated, "venue_migration", None)
        if vm is not None:
            print(f"      venue_migration: {_fmt(getattr(vm,'reason',None))} "
                  f"from={_fmt(getattr(vm,'from_venue_id',None))} "
                  f"to={_fmt(getattr(vm,'to_venue_id',None))} "
                  f"verified={_fmt(getattr(vm,'verified',None), 160)}")

        # -- 6. a direct canonical-pool SELL, independent of the above -----
        direct = await service._reader.quote_pumpswap_sell(
            mint, token_amount_raw=tokens_raw, pool_address=derived_pool)
        print("  --- direct quote_pumpswap_sell(derived canonical pool) ---")
        print(f"    available={direct.available} provider={direct.provider} "
              f"pool={_fmt(direct.pool_address)}")
        if not direct.available:
            print(f"      missing_reason: {_fmt(direct.missing_reason, 200)}")

    await jupiter.aclose()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="var/v2_floor_beta_state.json")
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--max-rpm", type=int, default=30)
    args = ap.parse_args()
    return asyncio.run(probe(Path(args.state), max(1, args.iterations),
                             max(1, args.max_rpm)))


if __name__ == "__main__":
    sys.exit(main())
