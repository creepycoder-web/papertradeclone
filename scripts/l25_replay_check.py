"""
Deterministic replay of the ACTIVE strategy against the repository's own
research dataset.

THIS IS NOT NEW RESEARCH
========================
It derives nothing, tunes nothing and produces no PnL. It is a correctness
check that the runtime's L25 gate reproduces the frozen research definition on
the same 276 events that produced the reference numbers.

THE INPUT IS FROZEN
===================
This reads ``research/frozen_l25/universe.json`` -- a committed, sha256-pinned
artifact -- and NEVER ``var/decisions.jsonl``. It used to call
``v3search.load_universe()``, which reads the live runtime files that the bot
appends to forever, so the "frozen" 276-event universe grew to 278 while the
bot traded and this correctness check began failing on data that had nothing to
do with the code under test. The artifact is frozen at the L25 cut epoch
1790735204, quoted from reports/research/v3c_random_order.md line 165. See
``scripts/freeze_l25_universe.py``.

    reference, reports/research/v3c_random_order.md §2
        route mix  FAST_ENTRY 253 / PRECISION_ENTRY 23
        V2 253,  V3C 154,  L25 116

The reference predicates are read from ``research/randorder.py`` - the
repository's own corrected spec implementation - rather than restated here, so
this script cannot accidentally agree with itself.

The counts are a VERIFICATION REFERENCE. Nothing about them is hardcoded in the
runtime; if the working dataset ever changes, this script explains the delta
rather than forcing the old numbers.

Run:  python scripts/l25_replay_check.py
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

from autopaper.l25 import (  # noqa: E402
    ACTIVE_STRATEGY,
    L25_DERIVED_QUANTILE_EXACT,
    L25_MIN_LIQ_TO_MCAP,
    L25_RESEARCH_CLASSIFICATION,
    evaluate as evaluate_l25,
)
from autopaper.v3c import (  # noqa: E402
    V3C_MAX_MARKET_CAP_USD,
    V3C_MIN_SIGNAL_QUALITY,
    evaluate as evaluate_v3c,
)

#: The reference counts from reports/research/v3c_random_order.md §2, used only
#: to REPORT agreement or explain a delta.
REFERENCE = {"V2": 253, "V3C": 154, "L25": 116}
REFERENCE_ROUTE_MIX = {"FAST_ENTRY": 253, "PRECISION_ENTRY": 23}

#: The frozen, committed input. Immutable by construction: it is sha256-pinned
#: in its manifest and verified on every run.
FROZEN_DIR = REPO / "research" / "frozen_l25"
FROZEN_UNIVERSE = FROZEN_DIR / "universe.json"
FROZEN_MANIFEST = FROZEN_DIR / "MANIFEST.json"


def load_frozen_universe() -> list[dict[str, Any]]:
    """
    Read the frozen universe and verify it has not been edited.

    A hash check is the point: without it the artifact would be no more
    trustworthy than the live file it replaced, just smaller.
    """
    if not FROZEN_UNIVERSE.exists():
        raise SystemExit(
            f"missing {FROZEN_UNIVERSE}\n"
            "Regenerate it once with: python scripts/freeze_l25_universe.py"
        )
    body = FROZEN_UNIVERSE.read_text()
    digest = hashlib.sha256(body.encode()).hexdigest()
    pinned = json.loads(FROZEN_MANIFEST.read_text())["universe_sha256"]
    if digest != pinned:
        raise SystemExit(
            "the frozen L25 universe no longer matches its pinned sha256\n"
            f"  expected {pinned}\n  observed {digest}\n"
            "Someone edited the frozen artifact. That is the one thing it must "
            "never be used for."
        )
    return json.loads(body)


def _reference_predicates() -> dict[str, Any]:
    """
    Import the repository's own corrected spec predicates.

    ``research/randorder.py`` holds the authoritative, non-duplicated
    implementation. Importing it rather than restating it is the point: if the
    research ever changes, this check follows it.
    """
    import randorder as R

    return {
        "module": "research/randorder.py",
        "V3C_MC_MAX": R.V3C_MC_MAX,
        "V3C_SQ_MIN": R.V3C_SQ_MIN,
        "L25_LIQ_MC": R.L25_LIQ_MC,
        "V2A_CEILING": R.V2A_CEILING,
        "in_v2": R.in_v2,
        "in_v3c": R.in_v3c,
        "in_l25": R.in_l25,
    }


def run() -> dict[str, Any]:
    ref = _reference_predicates()
    universe = load_frozen_universe()

    thresholds_agree = (
        ref["V3C_MC_MAX"] == V3C_MAX_MARKET_CAP_USD
        and ref["V3C_SQ_MIN"] == V3C_MIN_SIGNAL_QUALITY
        and ref["L25_LIQ_MC"] == L25_MIN_LIQ_TO_MCAP
    )

    n = len(universe)
    got = {"V2": 0, "V3C": 0, "L25": 0}
    ref_counts = {"V2": 0, "V3C": 0, "L25": 0}
    route_mix: dict[str, int] = {}
    mismatches: list[dict[str, Any]] = []
    # Reproduce the documented Round-2 population to explain the delta. The
    # Round-2 search omitted route == FAST_ENTRY AND compared the ratio at the
    # full-precision quantile, so both differences are measured separately.
    without_route_test = {"V3C": 0, "L25_at_4dp": 0, "L25_at_full_precision": 0}
    gap_events: list[dict[str, Any]] = []

    for t in universe:
        route = str(t.get("route"))
        mcap = t.get("f_market_cap_usd")
        sq = t.get("f_signal_quality")
        liq = t.get("f_liquidity_usd")

        route_mix[route] = route_mix.get(route, 0) + 1

        impl_l25 = evaluate_l25(
            route=route, market_cap_usd=mcap, signal_quality=sq, liquidity_usd=liq
        )
        impl_v3c = impl_l25.v3c
        impl_v2 = (
            route == "FAST_ENTRY"
            and not (mcap is not None and float(mcap) > ref["V2A_CEILING"])
        )

        got["L25"] += 1 if impl_l25.authorized else 0
        got["V3C"] += 1 if (impl_v3c is not None and impl_v3c.authorized) else 0
        got["V2"] += 1 if impl_v2 else 0

        r_l25 = bool(ref["in_l25"](t))
        r_v3c = bool(ref["in_v3c"](t))
        r_v2 = bool(ref["in_v2"](t))
        ref_counts["L25"] += 1 if r_l25 else 0
        ref_counts["V3C"] += 1 if r_v3c else 0
        ref_counts["V2"] += 1 if r_v2 else 0

        if impl_l25.authorized != r_l25 or (
            impl_v3c is not None and impl_v3c.authorized
        ) != r_v3c:
            mismatches.append({
                "t_entry": t.get("t_entry"),
                "route": route,
                "f_market_cap_usd": mcap,
                "f_liquidity_usd": liq,
                "f_signal_quality": sq,
                "impl_l25": impl_l25.authorized,
                "ref_l25": r_l25,
            })

        # Reproduce the documented Round-2 population to show where 159/119
        # came from, and decompose the delta.
        if (mcap is not None and float(mcap) <= 20000.0
                and float(sq or 0.0) >= 0.55):
            without_route_test["V3C"] += 1
            ratio = (float(liq) / float(mcap)) if (liq and mcap) else None
            if ratio is not None:
                if ratio >= L25_MIN_LIQ_TO_MCAP:
                    without_route_test["L25_at_4dp"] += 1
                if ratio >= L25_DERIVED_QUANTILE_EXACT:
                    without_route_test["L25_at_full_precision"] += 1
                    if ratio < L25_MIN_LIQ_TO_MCAP:
                        gap_events.append({
                            "route": route,
                            "f_market_cap_usd": mcap,
                            "f_liquidity_usd": liq,
                            "f_liq_to_mcap": ratio,
                            "note": (
                                "equals the TRAIN quantile exactly: passes the "
                                "full-precision threshold, fails the 4 dp one"
                            ),
                        })

    return {
        "active_strategy": ACTIVE_STRATEGY,
        "l25_min_liq_to_mcap": L25_MIN_LIQ_TO_MCAP,
        "research_classification": L25_RESEARCH_CLASSIFICATION,
        "universe_events": n,
        "universe_source": (
            "research/frozen_l25/universe.json (frozen, sha256-pinned; NOT live "
            "var/decisions.jsonl)"
        ),
        "l25_cut_epoch": json.loads(FROZEN_MANIFEST.read_text())["l25_cut_epoch"],
        "reference_source": ref["module"],
        "thresholds_match_reference": thresholds_agree,
        "implementation_counts": got,
        "reference_predicate_counts": ref_counts,
        "documented_reference_counts": REFERENCE,
        "counts_match_documented_reference": got == REFERENCE,
        "route_mix_observed": route_mix,
        "route_mix_documented": REFERENCE_ROUTE_MIX,
        "route_mix_matches": route_mix == REFERENCE_ROUTE_MIX,
        "per_event_mismatches": mismatches,
        "round2_without_route_test": without_route_test,
        "delta_decomposition": {
            "v3c": (
                f"{without_route_test['V3C']} -> {got['V3C']}: "
                f"{without_route_test['V3C'] - got['V3C']} PRECISION_ENTRY "
                "events removed by the route test"
            ),
            "l25": (
                f"{without_route_test['L25_at_full_precision']} -> "
                f"{got['L25']}: "
                f"{without_route_test['L25_at_full_precision'] - got['L25']} total, "
                "composed of the PRECISION_ENTRY events removed by the route "
                "test plus the events removed by comparing the ratio at the "
                "frozen 4 dp threshold instead of the full-precision quantile"
            ),
        },
        "threshold_precision_note": (
            f"The Round-2 ledger compared f_liq_to_mcap against the "
            f"full-precision TRAIN quantile {L25_DERIVED_QUANTILE_EXACT!r}. "
            f"The corrected spec (research/randorder.py) and this runtime use "
            f"the 4 dp form {L25_MIN_LIQ_TO_MCAP!r}. "
            f"{len(gap_events)} event(s) sit in that interval."
        ),
        "gap_events": gap_events,
    }


def render(d: dict[str, Any]) -> str:
    L: list[str] = []
    A = L.append
    A("# L25 REPLAY CHECK\n")
    A(f"**Active strategy: `{d['active_strategy']}`**  ")
    A(f"**f_liq_to_mcap threshold: `{d['l25_min_liq_to_mcap']:.4f}`**  ")
    A(f"**Research status: `{d['research_classification']}`**\n")
    A("Not new research. Derives nothing, tunes nothing, produces no PnL. It "
      "checks that the runtime gate reproduces the frozen research definition on "
      "the repository's own research universe, using the repository's own "
      "corrected spec predicates as the reference.\n")

    A(f"- Universe: **{d['universe_events']} events** (the frozen V2 universe)")
    A(f"- Source: `{d['universe_source']}`")
    A(f"- L25 cut epoch: `{d['l25_cut_epoch']}` "
      f"(`reports/research/v3c_random_order.md` line 165)")
    A(f"- Reference predicates: `{d['reference_source']}`")
    A(f"- Thresholds match the reference: **{d['thresholds_match_reference']}**\n")

    A("## Filtering relationship\n")
    A("| Strategy | Runtime | Reference predicates | Documented |")
    A("|---|---:|---:|---:|")
    A(f"| V2 | {d['implementation_counts']['V2']} | "
      f"{d['reference_predicate_counts']['V2']} | "
      f"{d['documented_reference_counts']['V2']} |")
    A(f"| V3-C | {d['implementation_counts']['V3C']} | "
      f"{d['reference_predicate_counts']['V3C']} | "
      f"{d['documented_reference_counts']['V3C']} |")
    A(f"| **L25** | **{d['implementation_counts']['L25']}** | "
      f"{d['reference_predicate_counts']['L25']} | "
      f"{d['documented_reference_counts']['L25']} |")
    A("")
    A(f"Matches the documented frozen universe: "
      f"**{d['counts_match_documented_reference']}**  ")
    A(f"Route mix matches: **{d['route_mix_matches']}** "
      f"({d['route_mix_observed']})  ")
    A(f"Per-event mismatches against the reference predicates: "
      f"**{len(d['per_event_mismatches'])}**\n")

    A("## Where 159/119 became 154/116\n")
    A("Two differences, not one. Both are documented consequences of the "
      "correction, and the runtime implements the corrected side of both.\n")
    A(f"- **V3-C:** {d['delta_decomposition']['v3c']}")
    A(f"- **L25:** {d['delta_decomposition']['l25']}\n")
    A(f"Threshold precision: {d['threshold_precision_note']}\n")
    if d["gap_events"]:
        A("Event(s) inside the precision interval:\n")
        A("```")
        for g in d["gap_events"]:
            A(f"{g}")
        A("```")
        A("")
    A("The route omission is the documented one in "
      "`reports/research/v3c_random_order.md` §2. The threshold-precision "
      "difference is a second, separate effect that the same section's "
      "corrected numbers reflect: the Round-2 ledger compared against the "
      "full-precision quantile, while the spec implementation and this runtime "
      "use the 4 dp constant. The runtime uses `0.9647` because that is the "
      "value in the repository's corrected spec implementation and the value "
      "that reproduces the documented 116.\n")

    if d["per_event_mismatches"]:
        A("## Mismatching events\n")
        A("```")
        for m in d["per_event_mismatches"][:20]:
            A(f"{m}")
        A("```\n")

    A("## What this does not establish\n")
    A(f"`L25` remains `{d['research_classification']}`. There is still no "
      "post-derivation data, so this check confirms implementation fidelity "
      "only. It is not evidence that L25 generalises to unseen data, not a PnL "
      "estimate, and not a claim that L25 is better than V3-C.\n")
    return "\n".join(L)


def main() -> int:
    d = run()
    out = REPO / "reports/migration/l25_replay_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=2, default=str) + "\n")
    (REPO / "reports/migration/l25_replay_check.md").write_text(render(d))
    print(render(d))
    return 0 if (
        d["counts_match_documented_reference"]
        and not d["per_event_mismatches"]
        and d["thresholds_match_reference"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
