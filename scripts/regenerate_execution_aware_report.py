"""
Regenerate ``reports/live/execution_aware_paper.*`` from the AUTHORITATIVE book,
through the existing canonical report-generation path.

WHY THIS EXISTS
---------------
``reports/live/execution_aware_paper.json`` is a DERIVED per-run dataset: the
execution-aware engine's in-memory ``signal_records`` / ``trade_records`` plus
its live account status, written by
:meth:`autopaper.paper_live.bridge.TelegramPaperBridge.persist`. The durable
history lives elsewhere -- ``var/paper_state.json`` (append-and-reconcile) and
``var/ledger.jsonl`` (append-only settlements).

An offline test once wrote into that path, so the report on disk described a
fixture position that does not exist in the operator's book. The authoritative
files were repaired; the report is regenerated here so the two agree again.

WHAT IT CALLS
-------------
Exactly the production path, unchanged:

    PaperTrader._paper_bridge_instance()   # builds the bridge, rehydrates
    TelegramPaperBridge.persist()          # writes the four reports/live artefacts

It is NOT a hand-edited JSON, and it does not re-implement the writer.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
* It never calls ``startup()``. That acquires the account lock, and the live
  runtime (a PAPER process the operator started) holds it. A second writer on the
  account is the exact hazard the lock exists to prevent.
* It never calls ``persist_state()``. That writes ``var/paper_state.json`` --
  the AUTHORITATIVE book -- and a second writer on the durable state is the same
  hazard one layer down. This script only writes the derived report.
* It never trades, signs, broadcasts, or touches REAL.
* It does not fabricate a settlement to make the report look consistent.

The two authoritative files are digested before and after, and the script fails
if either changed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autopaper.config import build_config  # noqa: E402
from autopaper.engine import PaperTrader  # noqa: E402
from autopaper.paper_live.guards import assert_paper_only  # noqa: E402

#: The authoritative files. Read-only to this script, and verified unchanged.
AUTHORITATIVE = (
    ROOT / "var" / "paper_state.json",
    ROOT / "var" / "ledger.jsonl",
)

REPORT = ROOT / "reports" / "live" / "execution_aware_paper.json"


def digest(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot() -> dict[Path, str | None]:
    return {p: digest(p) for p in AUTHORITATIVE}


def before_report() -> dict:
    if not REPORT.exists():
        return {}
    d = json.loads(REPORT.read_text())
    st = d.get("status", {})
    acc = (st.get("accounts") or {}).get("L25_CURRENT", {})
    arm = (st.get("arms") or {}).get("L25_CURRENT", {})
    return {
        "generated_at": d.get("generated_at"),
        "generated_at_iso": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(d.get("generated_at") or 0)),
        "signals": len(d.get("signals") or []),
        "trades": len(d.get("trades") or []),
        "accounts.has_position": acc.get("has_position"),
        "accounts.open_position_mint": acc.get("open_position_mint"),
        "accounts.open_exposure_usd": acc.get("open_exposure_usd"),
        "accounts.bankroll_usd": acc.get("bankroll_usd"),
        "accounts.cash_usd": acc.get("cash_usd"),
        "accounts.realized_trades": acc.get("realized_trades"),
        "arms.open_position": arm.get("open_position"),
        "arms.realized_pnl_usd": arm.get("realized_pnl_usd"),
        "arms.exits": arm.get("exits"),
        "arms.entry_blocked_position_active": arm.get(
            "entry_blocked_position_active"),
    }


async def main() -> int:
    audit = assert_paper_only().as_dict()
    print(f"paper-only audit: built={audit['transaction_built']} "
          f"signed={audit['transaction_signed']} "
          f"broadcast={audit['transaction_broadcast']}")

    before_auth = snapshot()
    before = before_report()
    print("\n=== BEFORE (stale report) ===")
    for k, v in before.items():
        print(f"  {k:34s} = {v!r}")

    config = build_config(ROOT)
    # NO startup(): the live runtime holds the account lock.
    trader = PaperTrader(config)
    if trader.owns_account:
        print("REFUSING: this process took the account lock")
        return 4
    print("\naccount lock NOT taken (the running runtime owns it)")

    bridge = await trader._paper_bridge_instance()
    restored = trader.rehydrate_paper_state()
    print(f"rehydrated from the authoritative book: {restored}")
    for account in bridge.accounts.values():
        print(f"  arm {account.arm}: has_position={account.has_position} "
              f"bankroll_usd={account.bankroll_usd:.6f} "
              f"realized_trades={account.closed_trades}")

    # THE CANONICAL REPORT-GENERATION CALL. Writes only reports/live/*.
    written = bridge.persist()
    print("\nregenerated via TelegramPaperBridge.persist():")
    for name, path in written.items():
        print(f"  {name:11s} -> {path}")

    after_auth = snapshot()
    print("\n=== authoritative files unchanged? ===")
    ok = True
    for p in AUTHORITATIVE:
        same = before_auth[p] == after_auth[p]
        ok = ok and same
        print(f"  {p.name:22s} {'UNCHANGED' if same else '*** CHANGED ***'}  "
              f"{str(after_auth[p])[:16]}")
    if not ok:
        print("\nREFUSING TO REPORT SUCCESS: a durable file was modified")
        return 5

    print("\n=== AFTER (regenerated report) ===")
    for k, v in before_report().items():
        print(f"  {k:34s} = {v!r}")

    # -- agreement with the authoritative book -----------------------
    d = json.loads(REPORT.read_text())
    st = d["status"]
    acc = st["accounts"]["L25_CURRENT"]
    book = json.loads((ROOT / "var" / "paper_state.json").read_text())
    n_open = len(book.get("open_positions") or [])

    print("\n=== agreement with the authoritative book ===")
    mints = {s["signal"]["token_mint"] for s in (d.get("signals") or [])}
    fixture_mints = mints & {"APxxh2tWCh95tbHitFJchcdbMsctSBgRS2Zzju9Bpump"}
    print(f"  report open position   : {acc['open_position_mint']}")
    print(f"  paper_state open count : {n_open}")
    print(f"  agree (both flat)      : "
          f"{acc['open_position_mint'] is None and n_open == 0}")
    print(f"  fixture mints in report: {len(fixture_mints)}")
    print(f"  safety                 : "
          f"{ {k: st['safety'][k] for k in ('ok', 'transaction_built', 'transaction_signed', 'transaction_broadcast')} }")
    return 0 if (acc["open_position_mint"] is None and n_open == 0
                 and not fixture_mints and ok) else 7


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))