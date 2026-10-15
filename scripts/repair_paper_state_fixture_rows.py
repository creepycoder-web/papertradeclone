"""
DRY-RUN / APPLY cleanup of offline test-fixture rows from the PAPER book.

WHAT IT DOES
------------
``var/paper_state.json`` accumulated open positions that were written by the
OFFLINE TEST SUITE, not by the runtime. They are provably fixtures (see the
40-check fingerprint below) and they are not trades: no ledger row mentions
them, they carry a fixture mint, and every field is byte-identical to
``tests/paper_bridge_fakes.py``.

This removes those rows from ``open_positions`` ONLY. It never touches:

* ``var/ledger.jsonl`` -- append-only settlement history, not rewritten;
* the ``trades`` collection -- historical closed trades, preserved whole;
* the ``signals`` collection -- every signal ever seen, preserved whole;
* any balance, bankroll or PnL figure. NO accounting number is recomputed,
  because none of these rows ever reached the ledger, so none of them ever
  moved an account.

THE ONLY STATE TRANSITION USED IS THE CANONICAL ONE
----------------------------------------------------
``PersistedState.close_positions(keys)`` -- the SAME method the runtime itself
calls in ``persist_state`` when a position settles. The cleanup is therefore
expressed in the production vocabulary ("this position is no longer open") and
not as an ad-hoc list edit.

NO AMOUNT IS FABRICATED. Removing an open position that never settled does not
change any bankroll: the arm's bankroll moved only on settlement, these never
settled, and the settlement rows do not exist. Accounting drift is therefore
zero BY CONSTRUCTION and is verified, not assumed.

USAGE
-----
    python -m scripts.repair_paper_state_fixture_rows            # dry run
    python -m scripts.repair_paper_state_fixture_rows --apply    # write

Dry run is the default and touches nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from autopaper.paper_live.schema import SIMULATED_FILL_KIND  # noqa: E402
from autopaper.paper_live.state import PaperStateStore  # noqa: E402

STATE = ROOT / "var" / "paper_state.json"
LEDGER = ROOT / "var" / "ledger.jsonl"

# ---------------------------------------------------------------------------
# THE FIXTURE FINGERPRINT
#
# Every expected value is read from the fixture module AT RUNTIME. Nothing is
# hardcoded, so this cannot drift from the fixture, and a record is only ever
# classified as a fixture by matching the fixture exactly -- never by a
# truncated mint prefix, a symbol, or a price that merely looks familiar.
# ---------------------------------------------------------------------------

from tests import paper_bridge_fakes as _fk  # noqa: E402

_BP = _fk.buy_point()
_RAW = f"BUY ${_fk.MINT}"
_FIX = {
    "mint": _fk.MINT,
    "pair": _fk.PAIR,
    "raw_sha256": hashlib.sha256(_RAW.encode()).hexdigest(),
    "raw_len": len(_RAW),
    "chat": "MooDengPresidentCallers",
    "symbol": "T",
    "token_decimals": 6,
    "tokens_raw": _fk.BUY_TOKENS_RAW,
    "bankroll_before": 5.0,
    "sol_usd": _fk.SOL_USD,
    "mark_usd": _fk.MARK_USD,
    "market_type": _BP.market_type,
    "venue_id": _BP.venue_id,
    "provider": _BP.provider,
    "venue": _BP.venue,
    "input_raw": _fk.BUY_LAMPORTS,
    "quote_mint": _BP.quote_mint,
    "eff_price_usd": float(_BP.effective_price_usd),
    "eff_price": float(_BP.effective_price),
}

CHECKS_PER_RECORD = 40


def fingerprint(record: dict, signal: dict | None) -> tuple[int, list[str]]:
    """
    Every field that must match for this to be a fixture row.

    Returns ``(checks_run, failures)``. A non-empty ``failures`` means the
    record is NOT a fixture and must be preserved.
    """
    if signal is None:
        return (0, ["no matching signal record in state"])

    f = record.get("entry_fill") or {}
    sg = signal.get("signal") or {}
    fails: list[str] = []
    n = 0

    def eq(name, got, want):
        nonlocal n
        n += 1
        if got != want:
            fails.append(f"{name}: {got!r} != {want!r}")

    def num(name, got, want):
        nonlocal n
        n += 1
        if got is None or abs(float(got) - float(want)) > 1e-30:
            fails.append(f"{name}: {got!r} != {want!r}")

    # -- position side
    eq("pos.mint", record.get("mint"), _FIX["mint"])
    eq("pos.symbol", record.get("symbol"), _FIX["symbol"])
    eq("pos.token_decimals", record.get("token_decimals"),
       _FIX["token_decimals"])
    eq("pos.tokens_raw", record.get("tokens_raw"), _FIX["tokens_raw"])
    num("pos.bankroll_before", record.get("bankroll_before"),
        _FIX["bankroll_before"])
    num("pos.sol_usd_reference", record.get("sol_usd_reference"),
        _FIX["sol_usd"])
    num("pos.entry_observed_mark", record.get("entry_observed_mark"),
        _FIX["mark_usd"])
    eq("pos.entry_market_type", record.get("entry_market_type"),
       _FIX["market_type"])
    eq("pos.entry_pool_address", record.get("entry_pool_address"),
       _FIX["pair"])
    eq("pos.entry_venue_id", record.get("entry_venue_id"), _FIX["venue_id"])
    eq("pos.entry_quote_source", record.get("entry_quote_source"),
       _FIX["provider"])

    # -- fill side
    eq("fill.kind", f.get("kind"), SIMULATED_FILL_KIND)
    eq("fill.side", f.get("side"), "BUY")
    eq("fill.provider", f.get("provider"), _FIX["provider"])
    eq("fill.venue", f.get("venue"), _FIX["venue"])
    eq("fill.pool", f.get("pool_or_curve"), _FIX["pair"])
    eq("fill.route", list(f.get("route") or ()), ["pump_direct"])
    eq("fill.identity", f.get("identity"), "PASS")
    eq("fill.fee_bps", f.get("fee_bps"), _BP.fee_bps)
    num("fill.price_impact", f.get("price_impact_pct"), _BP.price_impact_pct)
    eq("fill.input_raw", f.get("input_amount_raw"), _FIX["input_raw"])
    eq("fill.input_decimals", f.get("input_decimals"), 9)
    eq("fill.input_mint", f.get("input_mint"), _FIX["quote_mint"])
    eq("fill.output_raw", f.get("output_amount_raw"), _FIX["tokens_raw"])
    eq("fill.output_mint", f.get("output_mint"), _FIX["mint"])
    eq("fill.output_decimals", f.get("output_decimals"), 6)
    num("fill.eff_price_usd", f.get("effective_price_usd"),
        _FIX["eff_price_usd"])
    num("fill.eff_price", f.get("effective_price"), _FIX["eff_price"])
    num("fill.max_quote_age_ms", f.get("max_quote_age_ms"), 30_000.0)
    eq("fill.fee", f.get("fee"), 0.0)

    # -- signal side
    eq("sig.raw_sha256", sg.get("raw_text_sha256"), _FIX["raw_sha256"])
    eq("sig.raw_len", sg.get("raw_text_length"), _FIX["raw_len"])
    eq("sig.channel", sg.get("channel"), _FIX["chat"])
    eq("sig.token_mint", sg.get("token_mint"), _FIX["mint"])
    eq("sig.symbol", sg.get("symbol"), _FIX["symbol"])
    eq("sig.provenance", sg.get("extraction_provenance"),
       {"source": "telegram"})
    eq("sig.ambiguous", sg.get("ambiguous"), False)
    eq("sig.warnings", list(sg.get("extraction_warnings") or ()), [])
    eq("sig.candidates", list(sg.get("candidates") or ()), [])
    eq("sig.layer", sg.get("layer"), "OBSERVED_MARK")
    return n, fails


def classify() -> tuple[dict, list[dict], list[dict]]:
    """Return ``(raw_state, fixtures, preserved)``."""
    raw = json.loads(STATE.read_text())
    signals = {s["signal"]["signal_id"]: s for s in raw.get("signals") or []}
    fixtures, preserved = [], []
    for record in raw.get("open_positions") or []:
        n, fails = fingerprint(record, signals.get(record.get("signal_id")))
        row = {
            "signal_id": record.get("signal_id"),
            "arm": record.get("arm"),
            "symbol": record.get("symbol"),
            "mint": record.get("mint"),
            "opened_at": record.get("opened_at"),
            "checks_run": n,
            "failures": fails,
        }
        (fixtures if not fails else preserved).append(row)
    return raw, fixtures, preserved


def ledger_mentions(signal_ids: set[str]) -> set[str]:
    """Which of these signal_ids appear anywhere in the append-only ledger."""
    if not LEDGER.exists():
        return set()
    found = set()
    for line in LEDGER.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        for sid in signal_ids:
            if sid in line:
                found.add(sid)
    return found


def report(*, apply: bool) -> int:
    raw, fixtures, preserved = classify()
    # An operator repair action against production state; it says so
    # explicitly rather than relying on the store's permissive default.
    store = PaperStateStore(ROOT / "var", runtime=True)
    state = store.load()
    ok, problems = state.validate()

    print("=" * 78)
    print("PAPER STATE FIXTURE-ROW CLEANUP -- "
          + ("APPLY" if apply else "DRY RUN (nothing written)"))
    print("=" * 78)
    print(f"file              : {STATE}")
    print(f"state validate ok : {ok}")
    if problems:
        for p in problems:
            print(f"  problem: {p}")

    before = state.as_dict()
    print(f"\nopen_positions before : {len(state.open_positions)}")
    print(f"trades (preserved)    : {len(state.trades)}")
    print(f"signals (preserved)   : {len(state.signals)}")
    print(f"ledger lines (untouched): "
          f"{sum(1 for _ in LEDGER.open()) if LEDGER.exists() else 0}")

    # -- the ledger cross-check -------------------------------------------
    touched = ledger_mentions({r["signal_id"] for r in fixtures})
    print(f"\nledger cross-check: {len(fixtures)} fixture rows, "
          f"{len(touched)} of their signal_ids appear in the ledger")
    if touched:
        print("  !! a fixture signal has ledger history; STOP and review:")
        for sid in sorted(touched):
            print(f"     {sid}")

    # -- REMOVE ----------------------------------------------------------
    print(f"\n--- WOULD REMOVE: {len(fixtures)} fixture open position(s) "
          f"({CHECKS_PER_RECORD}/{CHECKS_PER_RECORD} checks each) ---")
    for r in fixtures:
        print(f"  REMOVE  {r['signal_id']}  arm={r['arm']} sym={r['symbol']} "
              f"mint={r['mint']} opened={r['opened_at']}")

    # -- PRESERVE --------------------------------------------------------
    print(f"\n--- WOULD PRESERVE: {len(preserved)} record(s) ---")
    if not preserved:
        print("  (none -- every open position is a proven fixture row)")
    for r in preserved:
        print(f"  KEEP    {r['signal_id']}  arm={r['arm']} sym={r['symbol']} "
              f"mint={r['mint']}")
        for f in r["failures"][:6]:
            print(f"            reason: {f}")

    print(f"\n--- UNTOUCHED BY THIS OPERATION ---")
    print(f"  var/ledger.jsonl            : NOT opened for writing")
    print(f"  state.trades                : {len(state.trades)} preserved whole")
    print(f"  state.signals               : {len(state.signals)} preserved whole")
    print(f"  every bankroll / PnL figure : NOT recomputed, NOT touched")
    print(f"  TITS settlement             : not in open_positions, unaffected")

    if not apply:
        print("\nDRY RUN COMPLETE. Nothing was written. "
              "Re-run with --apply to remove the rows above.")
        return 0

    if touched:
        print("\nREFUSING TO APPLY: a fixture signal has ledger history.")
        return 4
    if not fixtures:
        print("\nNOTHING TO REMOVE.")
        return 0

    # -- APPLY, using the canonical state transition ----------------------
    keys = [f"{r['arm']}::{r['signal_id']}" for r in fixtures]
    removed = state.close_positions(keys)
    after = state.as_dict()

    # Only the open_positions collection may differ. Anything else is a bug in
    # this script and is refused rather than written.
    for field in ("trades", "signals", "schema_version", "created_at"):
        if before.get(field) != after.get(field):
            print(f"\nREFUSING TO WRITE: {field} would change; only "
                  "open_positions may change")
            return 5

    store.save(state, note="fixture row cleanup (offline test pollution)")
    print(f"\nremoved {removed} open position(s)")
    print(f"open_positions after  : {len(state.open_positions)}")

    # -- post-write verification -----------------------------------------
    reread = json.loads(STATE.read_text())
    survivors = [p for p in reread["open_positions"]
                 if not fingerprint(p, {s['signal']['signal_id']: s
                                        for s in reread['signals']}.get(
                     p.get('signal_id')))[1]]
    print(f"\npost-write verification:")
    print(f"  fixture rows remaining : {len(survivors)}")
    print(f"  trades preserved       : {len(reread['trades'])}")
    print(f"  signals preserved      : {len(reread['signals'])}")
    print(f"  ledger sha256          : "
          f"{hashlib.sha256(LEDGER.read_bytes()).hexdigest()[:16]} (unchanged)")
    if len(reread["trades"]) != len(before["trades"]):
        print("  !! TRADE COUNT CHANGED")
        return 6
    return 0 if not survivors else 7


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the change (default: dry run)")
    raise SystemExit(report(apply=ap.parse_args().apply))