#!/usr/bin/env python
"""
Offline replay of ``var/paper_hold_path.jsonl`` into quote-based excursions.

READ-ONLY. It never touches the runtime, never opens the network, never signs or
broadcasts anything, and never writes to ``var/``. It reads a SNAPSHOT of the
hold path and reports per-trade quote-based MFE/MAE with honest coverage.

The point of this script is to establish, from data rather than assertion, how
many closed positions actually have a usable executable excursion path at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from autopaper.forensics.quote_excursions import (
    BASIS_MARK_ONLY,
    BASIS_QUOTE,
    BASIS_UNAVAILABLE,
    excursions,
    format_excursion_line,
    point_from_hold_record,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(
    "/var/folders/9y/2b_741bx2dq4msxwzvv4r_7w0000gn/T/opencode/l25snap"
    "/paper_hold_path.jsonl")


def load(path: Path) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Return (summaries, observations-by-trade) from the hold-path artefact."""
    summaries: dict[str, dict] = {}
    observations: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            kind = record.get("record_type")
            trade_id = str(record.get("trade_id") or "")
            if not trade_id:
                continue
            if kind == "SUMMARY":
                summaries[trade_id] = record
            elif kind == "OBSERVATION":
                observations[trade_id].append(record)
    return summaries, dict(observations)


def replay(path: Path, max_quote_age_ms: float | None = None) -> dict:
    summaries, observations = load(path)
    by_basis: dict[str, list[str]] = defaultdict(list)
    rows: list[dict] = []
    for trade_id in sorted(observations):
        # HOLD-phase observations only. POST_EXIT is a separate, later question
        # (the sold-too-early analysis) and must not be folded into the
        # in-position excursion.
        hold = [o for o in observations[trade_id]
                if str(o.get("phase") or "").upper() == "PATH"]
        cost = None
        entry = None
        for o in hold:
            if o.get("entry_notional_usd") is not None:
                entry = o.get("entry_notional_usd")
                break
        points = [
            point_from_hold_record(
                o, all_in_entry_cost=cost, entry_notional_usd=entry,
                max_quote_age_ms=max_quote_age_ms,
            ) for o in hold
        ]
        result = excursions(points)
        by_basis[result["basis"]].append(trade_id)
        rows.append({
            "trade_id": trade_id,
            "symbol": (hold[0].get("strategy_profile") if hold else None),
            "profile": (hold[0].get("strategy_profile") if hold else None),
            "observations": len(hold),
            "excursion": result,
        })
    return {
        "input": str(path),
        "trades_with_observations": len(observations),
        "summaries": len(summaries),
        "by_basis": {k: len(v) for k, v in by_basis.items()},
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT))
    parser.add_argument("--max-quote-age-ms", type=float, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    result = replay(Path(args.input), args.max_quote_age_ms)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2,
                                              sort_keys=True),
                                  encoding="utf-8")
    print(json.dumps({
        "input": result["input"],
        "trades_with_observations": result["trades_with_observations"],
        "by_basis": result["by_basis"],
    }, indent=2))

    # A few worked examples, one per basis, so the labels can be read directly.
    seen = set()
    for row in result["rows"]:
        basis = row["excursion"]["basis"]
        if basis in seen or basis == BASIS_QUOTE:
            continue
        seen.add(basis)
        print(f"\n[{basis}] {row['trade_id']}")
        print("  ", format_excursion_line(row["excursion"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
