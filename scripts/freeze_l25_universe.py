"""
Freeze the L25 research universe into an IMMUTABLE, committed artifact.

WHY THIS EXISTS
===============
``scripts/l25_replay_check.py`` is documented as verifying the runtime L25 gate
against "the frozen 276-event universe". It was not. It called
``v3search.load_universe()``, which reads the LIVE ``var/decisions.jsonl`` and
``var/ledger.jsonl`` -- files the running bot appends to forever. So the
"frozen" universe grew to 278 while the bot traded, and a correctness check
started failing on data that had nothing to do with the code under test.

THE BOUNDARY IS NOT GUESSED
===========================
The cut is read from the research report that produced the reference numbers,
not chosen by searching for a count that works:

    reports/research/v3c_random_order.md line 165
      "All 276 events predate the L25 cut (epoch 1790735204)."

Bounding the live universe by that epoch independently reproduces four
documented numbers at once -- 276 events, the 253/23 route mix, V3-C 154 and
L25 116. Four mutually independent figures agreeing is the corroboration; the
cut was not fitted to any of them.

This script is the ONE place that is allowed to read live data, and it exists
to be run once. The artifact it writes is what the replay check consumes from
then on, so the replay check has no live dependency at all.

Run:  python scripts/freeze_l25_universe.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "research"))

#: The L25 cut, quoted from reports/research/v3c_random_order.md line 165.
L25_CUT_EPOCH = 1_790_735_204

#: What the research report says the reference population was.
DOCUMENTED = {
    "universe_events": 276,
    "route_mix": {"FAST_ENTRY": 253, "PRECISION_ENTRY": 23},
    "V2": 253,
    "V3C": 154,
    "L25": 116,
}

#: Every field any consumer of this artifact reads. ``f_liq_to_mcap`` is kept
#: verbatim because the reference predicate ``randorder.lmc()`` consumes the
#: PRECOMPUTED value while the runtime recomputes liquidity/market_cap. Keeping
#: both sides' inputs intact is what makes the comparison meaningful.
FIELDS = (
    "t_entry",
    "t_exit",
    "route",
    "f_market_cap_usd",
    "f_signal_quality",
    "f_liquidity_usd",
    "f_liq_to_mcap",
)

OUT_DIR = REPO / "research" / "frozen_l25"
OUT_UNIVERSE = OUT_DIR / "universe.json"
OUT_MANIFEST = OUT_DIR / "MANIFEST.json"


def build() -> dict[str, Any]:
    import v3search as S

    universe, _rejected, _trades, _dropped, _data = S.load_universe()
    frozen = [t for t in universe if float(t["t_entry"]) <= L25_CUT_EPOCH]
    frozen.sort(key=lambda t: (float(t["t_entry"]), float(t["t_exit"])))

    rows = [{k: t.get(k) for k in FIELDS} for t in frozen]

    route_mix: dict[str, int] = {}
    for r in rows:
        route_mix[r["route"]] = route_mix.get(r["route"], 0) + 1

    from autopaper.l25 import evaluate as evaluate_l25

    v3c = sum(
        1
        for r in rows
        if (
            lambda x: x.v3c is not None and x.v3c.authorized
        )(
            evaluate_l25(
                route=str(r["route"]),
                market_cap_usd=r["f_market_cap_usd"],
                signal_quality=r["f_signal_quality"],
                liquidity_usd=r["f_liquidity_usd"],
            )
        )
    )
    l25 = sum(
        1
        for r in rows
        if evaluate_l25(
            route=str(r["route"]),
            market_cap_usd=r["f_market_cap_usd"],
            signal_quality=r["f_signal_quality"],
            liquidity_usd=r["f_liquidity_usd"],
        ).authorized
    )
    v2 = sum(
        1
        for r in rows
        if r["route"] == "FAST_ENTRY"
        and not (
            r["f_market_cap_usd"] is not None
            and float(r["f_market_cap_usd"]) > 25_372.71
        )
    )

    observed = {
        "universe_events": len(rows),
        "route_mix": route_mix,
        "V2": v2,
        "V3C": v3c,
        "L25": l25,
    }
    agrees = observed == DOCUMENTED
    # the first mismatch, named, so a failure says WHICH number drifted
    diffs = [
        f"{k}: observed {observed[k]!r} != documented {DOCUMENTED[k]!r}"
        for k in DOCUMENTED
        if observed[k] != DOCUMENTED[k]
    ]
    return {
        "rows": rows,
        "observed": observed,
        "agrees": agrees,
        "diffs": diffs,
        "live_universe_events": len(universe),
    }


def main() -> int:
    r = build()
    print(f"live universe (unbounded) : {r['live_universe_events']}")
    print(f"bounded by L25 cut       : {r['observed']['universe_events']}")
    for k in ("V2", "V3C", "L25"):
        print(f"  {k:<4}: {r['observed'][k]:>4}   (documented {DOCUMENTED[k]})")
    print(f"  route mix: {r['observed']['route_mix']}")
    print(f"  (documented {DOCUMENTED['route_mix']})")

    if not r["agrees"]:
        print("\nREFUSING TO WRITE: the bounded universe does not reproduce the "
              "documented research population.", file=sys.stderr)
        for d in r["diffs"]:
            print(f"  {d}", file=sys.stderr)
        print("\nThe documented numbers are the contract. If they no longer "
              "reproduce, the research is not reproducible and this artifact "
              "must not be created -- fixing the number instead would destroy "
              "the only evidence that the drift happened.", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    body = json.dumps(r["rows"], indent=2, sort_keys=True) + "\n"
    OUT_UNIVERSE.write_text(body)
    digest = hashlib.sha256(body.encode()).hexdigest()

    manifest = {
        "description": (
            "The frozen L25 research universe. Immutable: this is the input the "
            "L25 replay check consumes, so the check no longer reads live "
            "var/decisions.jsonl and cannot drift as the bot trades."
        ),
        "l25_cut_epoch": L25_CUT_EPOCH,
        "cut_source": (
            "reports/research/v3c_random_order.md line 165: 'All 276 events "
            "predate the L25 cut (epoch 1790735204).'"
        ),
        "derived_from": "v3search.load_universe() over var/ at freeze time",
        "regenerate_with": "python scripts/freeze_l25_universe.py",
        "universe_file": "research/frozen_l25/universe.json",
        "universe_sha256": digest,
        "n_events": r["observed"]["universe_events"],
        "fields": list(FIELDS),
        "observed_counts": r["observed"],
        "documented_counts": DOCUMENTED,
        "counts_agree_with_documented": True,
        "live_universe_events_at_freeze": r["live_universe_events"],
        "note": (
            "The live universe was "
            f"{r['live_universe_events']} at freeze time, i.e. "
            f"{r['live_universe_events'] - r['observed']['universe_events']} "
            "event(s) had been appended by the running bot AFTER the L25 cut. "
            "Those post-cut events are outside the research population and are "
            "excluded by the documented boundary, not by dropping rows to reach "
            "a number."
        ),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {OUT_UNIVERSE.relative_to(REPO)}")
    print(f"  sha256 {digest}")
    print(f"wrote {OUT_MANIFEST.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
