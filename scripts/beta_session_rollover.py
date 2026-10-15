#!/usr/bin/env python
"""
Open a NEW isolated V2 FLOOR (BETA) session, PRESERVING the previous one.

WHAT IT DOES
------------
    arm A -> $10.00
    arm B -> $10.00
    arm C -> $10.00
    total experimental capital -> $30.00

...on three INDEPENDENT ledgers, with ALL-IN 100%-of-deployable sizing, the
affordability checks and the unresolved-position block all unchanged (they live
in code, not in the state file, and this script does not touch them).

WHAT IT DOES NOT DO
-------------------
* It does NOT flatten, settle or price anything. A position whose exit never got
  an executable quote is carried over UNRESOLVED, with no exit price and no P&L,
  because inventing one is the single thing this harness exists to prevent.
* It does NOT delete history. The previous session file is archived verbatim --
  every historical modelled trade, every incident-recovery record and every
  unresolved position -- under a timestamped sibling name, and the forward store
  is archived the same way.
* It does NOT touch the L25 account, the ledger, the frozen research universe,
  ``.env.local``, or any other profile's state. Only the two V2 FLOOR (BETA)
  state files are rolled.
* It does NOT enable REAL execution. Nothing on this path can build, sign or
  broadcast a transaction.

SAFETY
------
Refuses to run while another process owns the PAPER account lock, because that
process would overwrite the new session on its next persist -- and would be
mid-write on the very file being archived.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/beta_session_rollover.py \
        [--root var] [--reason "..."] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autopaper import account_lock  # noqa: E402
from autopaper.paper_live.v2_floor_beta import (  # noqa: E402
    BETA_ARMS,
    BETA_SIZING_RULE,
    BETA_STARTING_BANKROLL_USD,
    FORWARD_STATE_FILENAME,
    BetaExperiment,
)

BETA_STATE_FILENAME = "v2_floor_beta_state.json"


def _counts(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"unreadable": True}
    return payload.get("counts") or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "var"))
    ap.add_argument("--reason", default="operator session rollover")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--bankroll", type=float, default=BETA_STARTING_BANKROLL_USD)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    beta_state = root / BETA_STATE_FILENAME
    forward_state = root / FORWARD_STATE_FILENAME

    print(f"state root           : {root}")
    print(f"beta session file    : {beta_state}")
    print(f"forward session file : {forward_state}")
    print(f"before: beta counts    = {_counts(beta_state)}")
    print(f"before: forward counts = {_counts(forward_state)}")

    if args.dry_run:
        print("DRY RUN: nothing written, no rollover performed.")
        return 0

    # THE SINGLE-WRITER GUARD. The runtime holds an OS-level flock on its account
    # state; if that lock is already taken, an engine is live and this rollover
    # must not run -- it would both race the writer and be overwritten by it.
    account_state = root / "paper_account.json"
    try:
        lock = account_lock.acquire(account_state, kind="beta_session_rollover")
    except account_lock.AccountAlreadyLocked as exc:
        print("\nREFUSED: a PAPER process still owns the account lock, so its "
              "state is being written right now.", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        print("  Stop the engine gracefully, confirm it exited, then run this "
              "again.", file=sys.stderr)
        return 2
    except account_lock.AccountLockUnavailable as exc:
        print(f"\nREFUSED: {exc}", file=sys.stderr)
        return 2

    try:
        exp = BetaExperiment(
            starting_bankroll_usd=float(args.bankroll),
            state_root=root, runtime=True)
        print(f"\nloaded session       : {exp.session_id}")
        print(f"forward session      : {exp.forward_session_id}")
        print("bankroll per arm BEFORE: "
              + ", ".join(f"{a}=${exp.accounts[a].bankroll_usd:,.2f}"
                          for a in BETA_ARMS))
        out = exp.rollover_session(reason=args.reason)
    finally:
        account_lock.release(lock)

    print("\n" + "=" * 72)
    print("ROLLOVER COMPLETE")
    print("=" * 72)
    print(f"reason                     : {out['reason']}")
    print(f"prior session id           : {out['prior_session_id']}")
    print(f"new session id             : {out['new_session_id']}")
    print(f"prior forward session id   : {out['prior_forward_session_id']}")
    print(f"new forward session id     : {out['new_forward_session_id']}")
    print(f"archived session           : {out['archived_to']}")
    print(f"archived forward session   : {out['archived_forward_to']}")
    print(f"flattened anything         : {out['flattened']}")
    print("bankroll per arm AFTER     : "
          + ", ".join(f"{a}=${v:,.2f}"
                      for a, v in sorted(out["bankroll_per_arm_after"].items())))
    print(f"total starting capital     : ${out['total_starting_capital_usd']:,.2f}")
    print(f"outstanding carried over   : {len(out['outstanding_carried'])} "
          "(UNRESOLVED incidents, no price, no P&L)")
    for row in out["outstanding_carried"]:
        print(f"  - arm {row['arm']} {row['symbol'] or '?'} "
              f"({row['parent_signal_id']}) attempts={row['exit_attempts']} "
              f"status={row['status']}")
    print(f"prior counts               : {out['prior_summary'].get('counts')}")
    print(f"ALL-IN sizing rule         : {BETA_SIZING_RULE}")
    print("after: beta counts         = " + str(_counts(beta_state)))
    print("after: forward counts      = " + str(_counts(forward_state)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
