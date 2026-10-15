"""
Reset the PAPER account to a clean $100 with a fresh statistics window.

WHY THIS DOES NOT TRUNCATE THE LEDGER
-------------------------------------
An earlier version of this script emptied ``var/ledger.jsonl``. That was wrong,
and it broke three tests:

* ``test_replay_reproduces_the_frozen_universe`` dropped to 0 events;
* ``test_snapshot_is_a_prefix_of_the_live_ledger`` failed, because the frozen V1
  snapshot is required to remain a PREFIX of the live ledger;
* ``test_historical_records_are_preserved`` failed - it exists precisely to
  assert "the historical ledger must not be emptied".

The ledger is append-only AND is the frozen research universe the L25 replay
validates against. So a reset must not touch it. Instead the account records a
``stats_epoch_started_at`` and the statistics report only trades at or after it:
the REPORTING WINDOW moves, the HISTORY stays.

WHAT CHANGES
------------
var/paper_account.json : cash $100, realized $0, peak $100, run_count 1,
                         no open positions, caches cleared, new stats epoch
var/ledger.jsonl      : UNTOUCHED (append-only, research input)
var/decisions.jsonl   : UNTOUCHED
var/candidates.jsonl  : UNTOUCHED

WHAT IS NEVER TOUCHED
---------------------
var/forward_validation/  the deployment boundary is IMMUTABLE and must not be
                         reset - doing so would let pre-reset data count as
                         prospective
var/emergency_stop.json  a latched halt is a safety state, not a statistic
research/, reports/, source, config, .env.local

Refuses to run while an engine holds the account lock, because that process
would overwrite the reset on its next persist.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VAR = ROOT / "var"
ARCHIVE = VAR / "archive"

#: Never truncated, and why.
APPEND_ONLY = ("ledger.jsonl", "decisions.jsonl", "candidates.jsonl")


def live_locks() -> list[Path]:
    """Lock files whose owning process is still alive."""
    live = []
    for lock in VAR.glob("*.lock"):
        try:
            pid = int(json.loads(lock.read_text()).get("pid", 0))
            os.kill(pid, 0)
        except (ProcessLookupError, ValueError, TypeError, OSError):
            continue
        live.append(lock)
    return live


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Reset the PAPER account to $100 (ledger untouched).")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--yes", action="store_true",
                    help="skip the interactive confirmation")
    args = ap.parse_args()
    root = Path(args.root)
    var = root / "var"
    state = var / "paper_account.json"

    if not state.exists():
        print(f"REFUSED: {state} does not exist; nothing to reset.")
        return 1

    locks = live_locks()
    if locks:
        print("REFUSED: an engine is running and holds the account lock:")
        for lock in locks:
            print(f"  {lock}")
        print("Stop it first, or it will overwrite this reset on its next "
              "persist.")
        return 1

    before = json.loads(state.read_text())
    acct = before.get("account", {})
    old_epoch = acct.get("stats_epoch_started_at")

    n_exit = 0
    ledger = var / "ledger.jsonl"
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            if not line.strip():
                continue
            try:
                if json.loads(line).get("event") == "exit":
                    n_exit += 1
            except json.JSONDecodeError:
                continue

    print("PAPER ACCOUNT RESET  (account only - the ledger is not touched)")
    print("=" * 68)
    print(f"  before : cash ${float(acct.get('cash_usd', 0)):,.2f}   "
          f"realized ${float(acct.get('realized_pnl_usd', 0)):,.2f}   "
          f"run #{acct.get('run_count')}")
    print(f"           stats window: "
          f"{'all history (no epoch)' if old_epoch is None else f'since {old_epoch}'}"
          f"   closed trades in ledger: {n_exit}")
    print(f"  after  : cash $100.00   realized $0.00   run #1")
    print(f"           stats window: trades from NOW on   closed trades shown: 0")
    print()
    print("  ledger.jsonl / decisions.jsonl / candidates.jsonl : UNTOUCHED")
    print("    (append-only, and the frozen research universe the L25 replay")
    print("     validates against - truncating it breaks 3 provenance tests)")
    print("  var/forward_validation/  : UNTOUCHED (boundary is immutable)")
    print("  var/emergency_stop.json : UNTOUCHED (a halt is a safety state)")
    print()

    if not args.yes:
        print("The previous account state is ARCHIVED, so this is reversible.")
        if input("Type 'reset' to proceed: ").strip().lower() != "reset":
            print("aborted; nothing changed.")
            return 1

    stamp = time.strftime("%Y%m%d-%H%M%S")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    archived = ARCHIVE / f"paper_account-pre-reset-{stamp}.json"
    shutil.copy2(state, archived)

    from autopaper.ledger import PaperAccountState, save_json_atomic

    now = time.time()
    fresh = PaperAccountState(
        starting_bankroll_usd=100.0,
        cash_usd=100.0,
        realized_pnl_usd=0.0,
        fees_paid_usd=0.0,
        slippage_paid_usd=0.0,
        created_at=now,
        updated_at=now,
        run_count=1,
        peak_equity_usd=100.0,
        realized_at_peak=0,
        # the new statistics window
        stats_epoch_started_at=now,
    )
    payload = {
        "account_id": before.get("account_id"),
        "agent": before.get("agent"),
        "state_path": before.get("state_path"),
        "saved_at": now,
        "owner_pid": None,
        "account": fresh.as_dict(),
        "open_positions": [],
        "counters": {"candidates_seen": 0, "buys": 0, "rejects": 0, "waits": 0},
        "unfilled_attempts": 0,
        "recently_exited": {},
        "first_valid_prices": {},
        "seen_messages": [],
        "seen_candidates": {},
    }
    save_json_atomic(state, payload)

    cleared = []
    for lock in var.glob("*.lock"):
        lock.unlink()
        cleared.append(lock.name)

    print("RESET COMPLETE")
    print("-" * 68)
    print(f"  archived  {archived.relative_to(root)}  "
          f"({archived.stat().st_size:,} bytes)")
    for name in cleared:
        print(f"  cleared stale lock  {name}")
    print()
    print("  cash $100.00 | realized $0.00 | run #1 | stats window: from now")
    print(f"  ledger untouched: {n_exit} historical trades still on disk")
    print(f"  to restore: cp {archived.relative_to(root)} var/paper_account.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
