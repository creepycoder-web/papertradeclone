"""
Reproduce the TITS CLOSE ALL failure on the REAL quote service, in memory only.

This constructs the same TelegramPaperBridge the runtime builds, rehydrates the
REAL persisted TITS record, and calls the canonical ``_close_for_arm`` exactly as
``close_all_paper`` does. It writes NOTHING: no ledger append, no state store, no
report artefact. It exists to answer one question -- what does the canonical
PAPER SELL path actually return for this position right now?

``DRY`` mode stops before settlement so no in-memory bankroll is even moved.
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
from autopaper.exits import ExitDecision, ExitStyle  # noqa: E402
from autopaper.paper_live.arms import parse_arms  # noqa: E402
from autopaper.paper_live.bridge import TelegramPaperBridge  # noqa: E402
from autopaper.paper_live.engine import _position_from_dict  # noqa: E402
from autopaper.paper_live.fills import FillModel  # noqa: E402
from autopaper.paper_live.quotes import QuoteService, SolReference  # noqa: E402
from autopaper.paper_live.schema import ObservedMarkPoint  # noqa: E402
from autopaper.same_market.jupiter import JupiterQuoteClient  # noqa: E402
from autopaper.same_market.reader import DirectPumpReader  # noqa: E402
from autopaper.same_market.rpc import SolanaReadOnlyRpc  # noqa: E402

MINT = sys.argv[1] if len(sys.argv) > 1 else \
    "GD66KG85oaZZicz7eD8hriuqABGMhhtgZPhghZ8xpump"


def build(config):
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
    reader = DirectPumpReader(rpc)
    service = QuoteService(
        rpc=rpc,
        sol_reference=SolReference(0.0, time.time(), "diagnostic"),
        jupiter=JupiterQuoteClient(
            timeout_s=float(getattr(config.market, "http_timeout_s", 8.0)),
            cache_ttl_s=float(getattr(config.market, "cache_ttl_s", 2.0)),
        ),
        max_quote_age_ms=30_000.0,
        allow_jupiter_fallback=True,
    )
    bridge = TelegramPaperBridge(
        arms=parse_arms(["L25_CURRENT"]),
        decision_engine=_NoDecision(),
        exit_policy=_NoPolicy(),
        quote_service=service,
        market=_NoMarket(),
        token_decimals_for=reader.mint_decimals,
        sol_reference=SolReference(0.0, time.time(), "diagnostic"),
        fill_model=FillModel(),
        starting_bankroll_usd=5.00,
        root=ROOT,
    )
    return bridge


class _NoDecision:
    def decide(self, **kw):
        raise AssertionError("not reached")


class _NoPolicy:
    def assess(self, *a, **kw):
        raise AssertionError("not reached")


class _NoMarket:
    async def quote_async(self, mint):
        raise AssertionError("not reached")


async def main() -> int:
    config = build_config()
    state = json.loads((ROOT / "var" / "paper_state.json").read_text())
    record = next(r for r in state["open_positions"] if r["mint"] == MINT)
    position = _position_from_dict(record)
    if position is None:
        print("the persisted record cannot be rebuilt")
        return 3

    bridge = build(config)
    spec = bridge.arms[0]
    account = bridge.accounts[spec.name.value]
    account.open_position(position)

    now = time.time()
    mark = ObservedMarkPoint(
        value_usd=2.397000e-06, pair_address=None, dex_id=None,
        liquidity_usd=None, volume_5m_usd=None, market_cap_usd=None,
        timestamp=now, available=True, missing_reason="",
    )
    decision = ExitDecision(
        action="EXIT", reason="EMERGENCY_KILL_ALL",
        style=ExitStyle.FLATTEN, trigger="EMERGENCY_KILL_ALL",
    )

    print("calling the canonical _close_for_arm (no persistence follows this)")
    trade = await bridge._close_for_arm(
        spec=spec, position=position, mark=mark, decision=decision)

    print(f"failure:           {trade.failure}")
    print(f"failure_detail:    {str(trade.failure_detail)[:500]}")
    print(f"exit_fill:         {type(trade.exit_fill).__name__}")
    print(f"sell_execution_status: {trade.sell_execution_status}")
    print(f"sell_quote_source: {trade.sell_quote_source}")
    print(f"sell_market_id:    {trade.sell_market_id}")
    print(f"sell_quote_price:  {trade.sell_quote_price}")
    print(f"sell_quote_age_ms: {trade.sell_quote_age_ms}")
    print(f"sell_quote_failure_cause: {trade.sell_quote_failure_cause}")
    print(f"sell_quote_retryable:     {trade.sell_quote_retryable}")
    print(f"sell_quote_attempts:      {trade.sell_quote_attempts}")
    print(f"account.position still set: {account.position is not None}")
    print(f"account.bankroll_usd:      {account.bankroll_usd}")

    # NOW the exact reporting expression close_all_paper uses.
    failure = getattr(trade, "failure", None)
    exit_fill = getattr(trade, "exit_fill", None)
    print("\n--- what close_all_paper reports for this trade ---")
    print(f"branch taken (failure is not None or exit_fill is None): "
          f"{(failure is not None or exit_fill is None)}")
    reason = str(
        getattr(trade, "failure_detail", None)
        or getattr(failure, "value", failure)
        or "NO_EXECUTABLE_SELL_QUOTE"
    )[:400]
    print(f"reported reason: {reason!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))