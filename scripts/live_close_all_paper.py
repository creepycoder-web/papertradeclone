"""
Run the EXISTING PAPER-only CLOSE ALL operator action, once, against the real
persisted book.

THIS IS NOT A NEW EXECUTION PATH. ``close_all_paper`` is the exact method the
Telegram CLOSE ALL PAPER confirm button calls (it is wired in as the
``close_all_paper`` extra and invoked by ``ButtonHandler``); this script only
supplies the runtime that the button would have run inside. There is no second
close implementation, no direct ledger write and no hand-edited fill.

PAPER ONLY. The runtime is constructed from the operator's own configuration,
which starts OFF / PAPER / REAL-not-armed; ``startup()`` asserts that before
anything else and cannot arm REAL. No signer, transaction builder or broadcaster
is reachable from this path -- ``assert_paper_only`` proves it structurally.

Refuses to run unless the operator has stopped the bot, because two processes
must never hold the same account.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autopaper.config import build_config  # noqa: E402
from autopaper.engine import PaperTrader  # noqa: E402
from autopaper.paper_live.guards import assert_paper_only  # noqa: E402
from autopaper.telegram.ui import render_close_all_paper_result  # noqa: E402


async def main() -> int:
    audit = assert_paper_only().as_dict()
    print(f"paper-only audit: built={audit['transaction_built']} "
          f"signed={audit['transaction_signed']} "
          f"broadcast={audit['transaction_broadcast']}")

    config = build_config(ROOT)
    trader = PaperTrader(config)
    trader.startup()
    print(f"mode={trader.state.trade_mode.value} "
          f"real_armed={trader.state.real_armed} "
          f"real_trading_enabled={trader.state.real_trading_enabled}")
    if trader.state.real_armed or trader.state.trade_mode.value != "PAPER":
        print("REFUSING: this runtime is not PAPER-only")
        return 4

    bridge = await trader._paper_bridge_instance()
    print("\nopen positions BEFORE:")
    for row in trader.open_positions_view():
        if row.get("source") == "execution_aware_paper":
            print(f"  {row['position_id']} {row['symbol']} {row['mint']}")

    result = await trader.close_all_paper()

    print("\n" + render_close_all_paper_result(result))
    print("\nraw result:")
    print(json.dumps(result, indent=2, default=str))

    print("\nopen positions AFTER:")
    for row in trader.open_positions_view():
        if row.get("source") == "execution_aware_paper":
            print(f"  {row['position_id']} {row['symbol']} {row['mint']}")
    if not any(r.get("source") == "execution_aware_paper"
               for r in trader.open_positions_view()):
        print("  (none)")

    # Release ownership, exactly as a normal shutdown does. A shutdown must
    # never close anything by itself, and this runtime never started the trading
    # loop, so there is nothing here but the lock and the account heartbeat.
    await trader.shutdown()
    return 0 if result.get("success") else 5


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))