"""
READ-ONLY PAPER SELL diagnostic for one persisted PAPER position.

THIS SCRIPT CANNOT TRADE.
It builds the SAME execution-aware quote service the runtime builds, calls the
SELL quote for the EXACT persisted token balance, and prints the result. It
never calls ``fill_from_quote``, never settles, never writes durable state, and
never touches the ledger, the account or the position object. There is no
``sign`` and no ``broadcast`` in the call graph: the quote service is read-only
by construction.

Usage:
    python -m scripts.diag_tits_sell_readonly [MINT]

Exit code 0 = a valid executable SELL quote exists and would pass the identity
and freshness gates. Exit code 2 = no usable quote (the exact refusal is
printed). Any other non-zero = the diagnostic itself failed.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autopaper.config import build_config  # noqa: E402
from autopaper.execution.solana import WSOL_MINT  # noqa: E402
from autopaper.paper_live.fills import FillRefused, fill_from_quote  # noqa: E402
from autopaper.paper_live.quotes import QuoteService, SolReference  # noqa: E402
from autopaper.paper_live.schema import (  # noqa: E402
    FailureReason,
    ObservedMarkPoint,
    PaperPosition,
    SimulatedFill,
)
from autopaper.same_market.jupiter import JupiterQuoteClient  # noqa: E402
from autopaper.same_market.reader import DirectPumpReader  # noqa: E402
from autopaper.same_market.rpc import SolanaReadOnlyRpc  # noqa: E402


def load_position(mint: str) -> dict:
    state = json.loads((ROOT / "var" / "paper_state.json").read_text())
    for record in state.get("open_positions", []):
        if record.get("mint") == mint:
            return record
    raise SystemExit(f"no open PAPER position for mint {mint}")


def rebuild(record: dict) -> PaperPosition:
    """Rebuild the position exactly the way ``rehydrate`` does."""
    from autopaper.paper_live.engine import _position_from_dict

    position = _position_from_dict(record)
    if position is None:
        raise SystemExit("the persisted record cannot be rebuilt into a position")
    return position


def build_quote_service(config) -> QuoteService:
    real = config.real
    rpc = SolanaReadOnlyRpc(
        url=real.public_rpc_url or None,
        urls=tuple(u.strip() for u in (real.public_rpc_urls or "").split(",")
                   if u.strip()),
        max_rpm=max(1, int(round(float(real.public_rpc_rate_per_s) * 60.0))),
        max_concurrent=max(1, int(real.public_rpc_max_concurrency)),
        limiter_wait_s=float(real.public_rpc_limiter_wait_s),
        timeout_s=float(real.public_rpc_timeout_s),
        telemetry_max_per_minute=max(
            0, int(real.public_rpc_telemetry_max_per_minute)),
        cache_ttl_s=0.0,
    )
    market_cfg = config.market
    return QuoteService(
        rpc=rpc,
        sol_reference=SolReference(0.0, time.time(), "diagnostic"),
        jupiter=JupiterQuoteClient(
            timeout_s=float(getattr(market_cfg, "http_timeout_s", 8.0)),
            cache_ttl_s=float(getattr(market_cfg, "cache_ttl_s", 2.0)),
        ),
        max_quote_age_ms=30_000.0,
        allow_jupiter_fallback=True,
    )


def field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


async def main(mint: str) -> int:
    config = build_config()
    record = load_position(mint)
    position = rebuild(record)

    service = build_quote_service(config)
    reader = service._reader  # noqa: SLF001 - same object the runtime uses

    print("=" * 78)
    print("READ-ONLY PAPER SELL DIAGNOSTIC  (no sign, no broadcast, no fill)")
    print("=" * 78)
    print(f"position_id:            paperlive_{position.signal_id}_{position.arm}")
    print(f"arm:                    {position.arm}")
    print(f"mint:                   {position.mint}")
    print(f"tokens_raw:             {position.tokens_raw}")
    print(f"token_decimals:         {position.token_decimals}")
    print(f"stored market identity: entry_venue_id="
          f"{position.entry_venue_id}")
    print(f"                         entry_pool_address="
          f"{getattr(position, 'entry_pool_address', None)}")
    print(f"                         entry_market_type="
          f"{getattr(position, 'entry_market_type', None)}")
    print(f"                         exit_venue_id="
          f"{getattr(position, 'exit_venue_id', None)}")
    print(f"entry fill (USD/token): "
          f"{float(position.entry_fill.effective_price_usd):.12g}")
    print(f"entry fill provider:    {position.entry_fill.provider}")

    # -- current authoritative discovery: the mint account itself ---------
    decimals = await reader.mint_decimals(position.mint)
    print(f"\nlive mint decimals:     {decimals} (read from the mint account)")

    curve = None
    try:
        curve, curve_snapshot = await reader.read_bonding_curve(position.mint)
    except Exception as exc:  # noqa: BLE001 - diagnostic must report, not raise
        print(f"bonding curve read:     ERROR {type(exc).__name__}: {exc}")
    if curve is not None:
        print(f"bonding curve:          {curve.address}")
        print(f"  complete:             {curve.complete}")
        print(f"  graduated:            {curve.graduated}")
        print(f"  virtual_token_reserves: {curve.virtual_token_reserves}")
        print(f"  mint matches:         {curve.mint == position.mint}")
    else:
        print("bonding curve:          UNREADABLE / ABSENT")

    # -- the execution-aware SELL quote ----------------------------------
    pinned_pool = (
        getattr(position, "exit_venue_id", None) or ""
    )
    pinned_address = None
    if pinned_pool and ":" in str(pinned_pool):
        pinned_address = str(pinned_pool).split(":")[-1]
    pinned_address = (
        pinned_address
        or getattr(position, "entry_pool_address", None)
    )

    # EVERY candidate, not only the selected one: "no usable quote" is only
    # actionable when the operator can see WHICH venue refused and why.
    print("\n--- quote_candidates (every venue, for this exact size) ---")
    started = time.monotonic()
    try:
        candidates = await service.quote_candidates(
            side="SELL", mint=position.mint,
            amount_raw=position.tokens_raw,
            token_decimals=position.token_decimals,
            observed_pair_address=pinned_address,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"quote_candidates RAISED {type(exc).__name__}: {exc}")
        candidates = []
    print(f"candidate count:        {len(candidates)} "
          f"({(time.monotonic() - started) * 1000.0:.0f}ms)")
    usable = service._usable_candidates(candidates, side="SELL",
                                        mint=position.mint)  # noqa: SLF001
    print(f"usable candidates:      {len(usable)}")
    for c in candidates:
        print(f"  - provider={field(c, 'provider')!r} "
              f"available={field(c, 'available')} "
              f"stale={field(c, 'stale')} "
              f"identity={field(c, 'identity')!r} "
              f"venue_id={field(c, 'venue_id')!r}")
        print(f"      out_raw={field(c, 'output_amount_raw')} "
              f"out_dec={field(c, 'output_amount_decimal')} "
              f"token_mint={str(field(c, 'token_mint'))[:8]}… "
              f"quote_mint={str(field(c, 'quote_mint'))[:8]}…")
        ev = field(c, 'identity_evidence')
        mr = field(c, 'missing_reason')
        if ev:
            print(f"      identity_evidence: {str(ev)[:300]}")
        if mr:
            print(f"      missing_reason:    {str(mr)[:300]}")

    # THE VALIDATED MIGRATION FALLBACK. ``_close_for_arm`` consults it whenever
    # the pinned book yields nothing and the entry venue is a bonding curve, so
    # the diagnostic must exercise it too -- otherwise it reports the pinned
    # path's refusal as the whole story.
    if str(position.entry_venue_id).startswith("PUMP_BONDING_CURVE:"):
        print("\n--- resolve_graduated_exit_venue (the ONLY permitted migration) ---")
        started = time.monotonic()
        migrated = await service.resolve_graduated_exit_venue(
            mint=position.mint,
            token_amount_raw=position.tokens_raw,
            token_decimals=position.token_decimals,
            entry_venue_id=position.entry_venue_id,
            entry_pool_address=getattr(position, "entry_pool_address", None),
            observed_pair_address=None,
        )
        print(f"latency_ms:             {(time.monotonic() - started) * 1000.0:.0f}")
        print(f"provider:               {field(migrated, 'provider')}")
        print(f"resolved market:        venue_id={field(migrated, 'venue_id')}")
        print(f"                         market_type="
              f"{field(migrated, 'market_type')}")
        print(f"                         pool_or_curve="
              f"{field(migrated, 'pool_or_curve')}")
        print(f"identity:               {field(migrated, 'identity')}")
        print(f"identity_evidence:      "
              f"{str(field(migrated, 'identity_evidence'))[:400]}")
        print(f"available:              {field(migrated, 'available')}")
        print(f"quote age (ms):         {field(migrated, 'quote_age_ms')}")
        print(f"executable output raw:  {field(migrated, 'output_amount_raw')}")
        print(f"executable output dec:  {field(migrated, 'output_amount_decimal')}")
        print(f"effective price USD:    {field(migrated, 'effective_price_usd')}")
        print(f"missing_reason:         "
              f"{str(field(migrated, 'missing_reason'))[:400]}")

    print("\n--- quote_sell_pinned (the canonical PAPER SELL quote path) ---")
    started = time.monotonic()
    try:
        quote, counterfactual = await service.quote_sell_pinned(
            mint=position.mint,
            token_amount_raw=position.tokens_raw,
            token_decimals=position.token_decimals,
            pinned_venue_id=position.effective_exit_venue_id or None,
            pinned_pool_address=pinned_address,
            observed_pair_address=None,
            cache=None,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"quote_sell_pinned RAISED {type(exc).__name__}: {exc}")
        return 3
    latency_ms = (time.monotonic() - started) * 1000.0

    print(f"latency_ms:             {latency_ms:.0f}")
    print(f"provider:               {field(quote, 'provider')}")
    print(f"resolved market:        venue_id={field(quote, 'venue_id')}")
    print(f"                         market_type={field(quote, 'market_type')}")
    print(f"                         pool_or_curve={field(quote, 'pool_or_curve')}")
    print(f"identity:               {field(quote, 'identity')}")
    print(f"identity_evidence:      {str(field(quote, 'identity_evidence'))[:400]}")
    print(f"SELL quote available:   {bool(field(quote, 'available'))}")
    print(f"quote age (ms):         {field(quote, 'quote_age_ms')}")
    print(f"max quote age (ms):     {field(quote, 'max_quote_age_ms')}")
    print(f"stale:                  {field(quote, 'stale')}")
    print(f"executable output raw:  {field(quote, 'output_amount_raw')}")
    print(f"executable output dec:  {field(quote, 'output_amount_decimal')}")
    print(f"effective price USD:    {field(quote, 'effective_price_usd')}")
    print(f"missing_reason:         {str(field(quote, 'missing_reason'))[:400]}")
    if counterfactual is not None:
        print(f"counterfactual venue:   {field(counterfactual, 'venue_id')}")

    # -- the settlement gate, evaluated but NEVER applied -----------------
    print("\n--- fill gate (evaluated on a THROWAWAY fill model; nothing settled) ---")
    try:
        fill = fill_from_quote(
            quote, side="SELL", model=None, sol_reference=None,
            require_identity=True,
        )
    except FillRefused as exc:
        print(f"exact rejection:        {exc.reason.value}")
        print(f"exact detail:           {exc.detail[:400]}")
        print("\nVERDICT: NO EXECUTABLE SELL QUOTE. Position must stay OPEN.")
        return 2
    print(f"gate verdict:           PASS (a fill would be constructible)")
    print(f"would-be fill price:    {float(fill.effective_price_usd):.12g} USD/token")
    print(f"would-be WSOL out:      {fill.output_amount_decimal}")
    print("\nVERDICT: an executable SELL quote exists and passes identity+freshness.")
    print("         Nothing was filled, settled or written by this script.")
    return 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else \
        "GD66KG85oaZZicz7eD8hriuqABGMhhtgZPhghZ8xpump"
    raise SystemExit(asyncio.run(main(target)))