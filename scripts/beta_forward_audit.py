"""Forward-experiment + delivery/recovery audit for V2 FLOOR (BETA). READ-ONLY.

WHAT IT DOES
============

Two things, and neither of them settles, deletes or rewrites anything.

**``--init-forward``** initializes the FORWARD experiment's own store,
``var/v2_floor_beta_forward_state.json``, from the live experiment's CURRENT
ledger. It writes ONLY that new file: the main store
(``v2_floor_beta_state.json``) is hashed before and after and the audit FAILS
if its bytes changed. No bankroll is reset, no record is deleted, no historical
trade is moved or re-based, and no session is started.

**default (audit)** copies the live store into a temporary root, restores it
with the REAL :class:`BetaExperiment`, and renders every operator-facing
surface from that copy -- Status (``stats()``), Closed Positions
(``PaperTrader._beta_closed_rows``), the legacy PAPER page and the Telegram
message builders -- then checks that all of them describe the SAME settlements.

WHY A COPY AND NOT THE LIVE FILE
================================

Rendering must never be able to write. Restoring from a COPY in an isolated root
means the real rendering code runs against the real data while the operator's
file is only ever read.

WHAT "AGREE" MEANS HERE, AND WHAT IT CANNOT MEAN
===============================================

It checks: the number and identity of settled arms, the realized P&L, the exit
reasons, the accounting basis, and that an UNRESOLVED arm is never presented as
a closed/sold position on any surface.

It CANNOT check that a Telegram message was DELIVERED. There is no persisted
send ledger in this codebase (``TelegramNotifier`` counts queued/sent/failed in
memory only), so delivery is unverifiable offline and is reported as such rather
than asserted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import dataclasses  # noqa: E402

from autopaper.engine import PaperTrader  # noqa: E402
from autopaper.paper_live.engine import PaperLiveEngine  # noqa: E402
from autopaper.paper_live.v2_floor_beta import (  # noqa: E402
    EXECUTABLE_QUOTE_BASIS,
    FORWARD_STATE_FILENAME,
    BetaExperiment,
    BetaPosition,
    basis_bucket,
)
from autopaper.telegram import ui  # noqa: E402

VAR = REPO / "var"
MAIN = VAR / "v2_floor_beta_state.json"
FORWARD = VAR / FORWARD_STATE_FILENAME
REPORTS = REPO / "reports" / "research"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Notifier(PaperLiveEngine):
    """The REAL engine's notification methods, with only the experiment wired.

    ``render_beta_exits`` calls its own ``_beta_trade_persisted``,
    ``_beta_exit_message`` and ``_beta_exit_arm_lines``, so the message surface
    has to be exercised through a real instance method -- an unbound call on a
    ``SimpleNamespace`` is exactly the kind of stand-in that would have hidden
    the pinned-quote defect. Only ``beta_experiment`` is set: these builders
    read nothing else, and nothing here can reach a ledger.
    """

    def __init__(self, exp: BetaExperiment) -> None:
        self.beta_experiment = exp


def _bridge(exp: BetaExperiment) -> SimpleNamespace:
    return SimpleNamespace(beta_experiment=exp)


def closed_rows(exp: BetaExperiment) -> list[dict]:
    return PaperTrader._beta_closed_rows(  # type: ignore[misc]
        SimpleNamespace(_paper_bridge=_bridge(exp)))


def as_position(rec: dict) -> BetaPosition:
    """A CLOSED trade record reconstructed as the position the builders expect.

    The notification builders take ``BetaPosition`` objects. A closed record is
    NOT restored into ``exp.positions`` (only outstanding arms are), so the
    surface check rebuilds the object from the SAME already-persisted fields
    rather than re-settling anything. No number is recomputed.
    """
    names = {f.name for f in dataclasses.fields(BetaPosition)}
    kw = {k: rec[k] for k in names if k in rec}
    if isinstance(kw.get("entry_route"), list):
        kw["entry_route"] = tuple(kw["entry_route"])
    kw.setdefault("entry_route", ())
    return BetaPosition(**kw)


def init_forward() -> int:
    """Create/populate the forward store. Writes ONLY the forward file."""
    if not MAIN.exists():
        print(f"REFUSED: {MAIN} does not exist; nothing to initialize from")
        return 2
    before_hash = sha256(MAIN)
    before_size = MAIN.stat().st_size
    existed = FORWARD.exists()
    before_forward = sha256(FORWARD) if existed else None

    # runtime=True is the OPERATOR-SIDE permission to touch this directory. The
    # experiment loads the live ledger; only ``_persist_forward`` is called, so
    # the main store is never written.
    exp = BetaExperiment(state_root=VAR, runtime=True)
    prog_before = exp.forward_progress()
    exp._persist_forward()  # noqa: SLF001 - the single forward writer
    exp2 = BetaExperiment(state_root=VAR, runtime=True)
    prog = exp2.forward_progress()

    after_hash = sha256(MAIN)
    after_size = MAIN.stat().st_size
    ok = before_hash == after_hash
    print(f"main store before : {before_hash[:16]} ({before_size} B)")
    print(f"main store after  : {after_hash[:16]} ({after_size} B)")
    print(f"MAIN STORE BYTE-IDENTICAL: {ok}")
    print(f"forward store     : {prog['store']}")
    print(f"forward store existed before this run: {existed}")
    if before_forward:
        print(f"forward store hash before: {before_forward[:16]}")
    print(f"forward store hash after : {sha256(FORWARD)[:16]}")
    print(f"forward session   : {prog['session_id']}")
    print(f"eligibility       : {prog['eligibility']}")
    print(f"excludes          : {prog['excludes']}")
    print(f"unique parents    : {prog['unique_parent_signals']} "
          f"/ {prog['target_unique_parent_signals']} "
          f"(remaining {prog['remaining_to_target']}, "
          f"{prog['progress_pct']}%)")
    print(f"completed trades  : {prog['completed_trades']}")
    print(f"unresolved arms   : {prog['unresolved_arms']}")
    print(f"performance claim : {prog['performance_claim']}")
    print(f"progress before this call: {prog_before['unique_parent_signals']} "
          f"parents / {prog_before['completed_trades']} trades")
    for g in prog["grouped_by_parent"]:
        print(f"  parent {g['parent_signal_id']}  settled={g['arms_settled']} "
              f"unresolved={sorted(k for k, v in g['unresolved'].items() if v)}")
    return 0 if ok else 1


def audit() -> int:
    if not MAIN.exists():
        print(f"REFUSED: {MAIN} does not exist")
        return 2
    live = json.loads(MAIN.read_text())
    tmp = Path(tempfile.mkdtemp(prefix="beta_forward_audit_"))
    shutil.copy2(MAIN, tmp / "v2_floor_beta_state.json")
    if FORWARD.exists():
        shutil.copy2(FORWARD, tmp / FORWARD_STATE_FILENAME)
    exp = BetaExperiment(state_root=tmp)

    trades = [t for t in exp.completed if t.get("status") == "closed"]
    persisted_quote = [t for t in trades if basis_bucket(t) == "quote"]
    persisted_modeled = [t for t in trades if basis_bucket(t) == "modeled"]
    stats = exp.stats()
    rows = closed_rows(exp)
    prog = exp.forward_progress()

    open_arms = [p for pid in exp.positions.values() for p in pid.values()
                 if p.status in ("open", "unresolved")]
    unresolved = [p for p in open_arms if p.status == "unresolved"]

    # -- the surfaces -------------------------------------------------------
    arm_stats = stats["arms"]
    status_closed = sum(int(a["completed_trades"]) for a in arm_stats.values())
    status_unresolved = sum(int(a["unresolved"]) for a in arm_stats.values())
    status_realized = round(
        sum(float(a["realized_pnl_usd"]) for a in arm_stats.values()), 6)
    persisted_realized = round(
        sum(float(t.get("net_pnl_usd") or 0.0) for t in trades), 6)
    rows_realized = round(
        sum(float(r.get("realized_pnl_usd") or 0.0) for r in rows), 6)
    rows_quote = sum(1 for r in rows
                     if r.get("pnl_basis", "").endswith("QUOTE_SIMULATED_FILL"))

    # -- the Telegram builders, on the SAME records -------------------------
    exp_render = BetaExperiment(state_root=tmp)
    notifier = _Notifier(exp_render)
    settled_positions = [as_position(t) for t in exp_render.completed
                         if t.get("status") == "closed"]
    sale_msgs = notifier.render_beta_exits(settled_positions)
    blocked_msgs = notifier.render_beta_exit_blocked()
    # A message names its arms in its OWN identity (``dedupe_key``), which is
    # derived from exactly the arms it reports -- so the arms each built message
    # describes are read from the message, not assumed from a count that is not
    # on it.
    def arms_in(msg: dict) -> list[str]:
        tail = str(msg.get("dedupe_key") or "").split("::")[-1]
        return [a for a in tail.split(",") if a]

    sale_arms: list[tuple[str, str]] = []
    modeled_arms: list[tuple[str, str]] = []
    for m in sale_msgs:
        pairs = [(str(m.get("parent_signal_id")), a) for a in arms_in(m)]
        if m.get("kind") == "beta_exit":
            sale_arms += pairs
        elif m.get("kind") == "beta_exit_modeled":
            modeled_arms += pairs
    quote_pairs = {(str(t.get("parent_signal_id")), str(t.get("arm")))
                   for t in persisted_quote}
    modeled_pairs = {(str(t.get("parent_signal_id")), str(t.get("arm")))
                     for t in persisted_modeled}

    # PER-ARM EXACTNESS, not a loose total. ``arm_stats`` rounds each arm's
    # realized figure to 4 decimals for display, so a TOTAL built from three
    # rounded arms can sit a few 1e-5 away from the exact sum. That is display
    # rounding, and it is MEASURED here rather than hidden behind a wide
    # tolerance: per arm the check is exact, and the total's residual is
    # reported as its own number.
    status_per_arm_exact = all(
        abs(float(a["realized_pnl_usd"])
            - round(sum(float(t.get("net_pnl_usd") or 0.0) for t in trades
                        if t.get("arm") == arm), 4)) < 1e-9
        for arm, a in arm_stats.items())
    row_counts_by_arm = {
        arm: sum(1 for t in trades if t.get("arm") == arm)
        for arm in arm_stats}
    rows_by_arm = {arm: sum(1 for r in rows if r.get("arm") == arm)
                   for arm in arm_stats}

    # THE PROPERTY THAT MATTERS: no CLOSED POSITION row may describe an arm that
    # is still outstanding. (A closed-position row has no ``status`` key at all
    # -- it is a row in a closed view -- so the check is on the IDENTITY of the
    # (parent, arm) pairs, which is what "shown as closed" actually means.)
    unresolved_pairs = {(p.parent_signal_id, p.arm) for p in unresolved}
    row_pairs = {(str(r.get("parent_signal_id")), str(r.get("arm")))
                 for r in rows}
    total_residual = round(status_realized - persisted_realized, 9)

    checks = {
        "status_closed_matches_persisted": status_closed == len(trades),
        "status_per_arm_realized_is_exact": status_per_arm_exact,
        "status_per_arm_counts_match_persisted": (
            all(int(a["completed_trades"]) == row_counts_by_arm[arm]
                for arm, a in arm_stats.items())),
        "status_total_within_display_rounding": abs(total_residual) <= 2e-4,
        "closed_rows_match_persisted": len(rows) == len(trades),
        "closed_rows_realized_matches_persisted": (
            abs(rows_realized - persisted_realized) < 1e-9),
        "closed_rows_count_per_arm_matches": rows_by_arm == row_counts_by_arm,
        "closed_rows_quote_count_matches": rows_quote == len(persisted_quote),
        "status_unresolved_matches_outstanding": status_unresolved == len(unresolved),
        "no_unresolved_arm_appears_as_closed": not (row_pairs & unresolved_pairs),
        "every_closed_row_carries_realized_pnl": all(
            r.get("realized_pnl_usd") is not None for r in rows),
        "unresolved_never_announced_as_a_sale": not (
            set(sale_arms) & unresolved_pairs),
        "sale_messages_describe_exactly_the_quote_settlements": (
            set(sale_arms) == quote_pairs),
        "modeled_messages_describe_exactly_the_modeled_settlements": (
            set(modeled_arms) == modeled_pairs),
        "modeled_messages_never_claim_a_quote_sale": all(
            "executable SELL quote" not in m["text"]
            and "MODELED" in m["text"]
            for m in sale_msgs if m.get("kind") == "beta_exit_modeled"),
        "quote_sale_messages_stay_labelled_a_quote_fill": all(
            "executable SELL quote" in m["text"]
            for m in sale_msgs if m.get("kind") == "beta_exit"),
        "exit_blocked_messages_book_no_sale": all(
            "NO SALE WAS BOOKED" in m["text"]
            and "net PAPER P&L" not in m["text"]
            for m in blocked_msgs),
    }

    # -- render the real operator surfaces (read-only) ----------------------
    status_text = ui.render_beta_stats({"execution_aware_paper": {
        "v2_floor_beta": stats}})
    closed_text = ui.render_closed_positions(rows, total=len(rows))

    report = {
        "kind": "READ_ONLY_AUDIT",
        "main_store": str(MAIN),
        "main_store_sha256": sha256(MAIN),
        "forward_store": str(FORWARD),
        "forward_store_present": FORWARD.exists(),
        "session_id_loaded": exp.session_id,
        "sessions_in_store": sorted({str(t.get("session_id"))
                                     for t in trades}),
        "persisted": {
            "closed_trades": len(trades),
            "quote_basis": len(persisted_quote),
            "modeled_basis": len(persisted_modeled),
            "unique_parents": len({str(t.get("parent_signal_id"))
                                   for t in trades}),
            "realized_pnl_usd": persisted_realized,
            "exit_reasons": {r: sum(1 for t in trades
                                    if t.get("exit_reason") == r)
                             for r in ("TP", "SL", "TIME", "FLATTEN")},
        },
        "outstanding": {
            "arms": len(open_arms),
            "unresolved": len(unresolved),
            "parents": sorted({p.parent_signal_id for p in open_arms}),
            "reasons": sorted({(p.exit_blocked_reason or "")[:120]
                               for p in unresolved}),
            "max_exit_attempts": max((int(p.exit_attempts or 0)
                                      for p in open_arms), default=0),
        },
        "status_surface": {
            "closed_trades": status_closed,
            "unresolved_arms": status_unresolved,
            "realized_pnl_usd": status_realized,
            "realized_residual_vs_persisted_usd": total_residual,
            "residual_explained_by": (
                "arm_stats rounds each arm's realized figure to 4 decimals for "
                "display; the residual is that rounding summed over three arms, "
                "and the per-arm check is exact"),
            "per_arm": {k: {"completed_trades": v["completed_trades"],
                            "unresolved": v["unresolved"],
                            "realized_pnl_usd": v["realized_pnl_usd"],
                            "bankroll_usd": v["bankroll_usd"],
                            "outside_swap_cost_model": v["outside_swap_cost_model"]}
                        for k, v in arm_stats.items()},
        },
        "closed_positions_surface": {
            "rows": len(rows),
            "realized_pnl_usd": rows_realized,
            "quote_basis_rows": rows_quote,
            "rows_per_arm": rows_by_arm,
            "unresolved_pairs_shown_as_closed": sorted(
                f"{p}::{a}" for p, a in (row_pairs & unresolved_pairs)),
        },
        "telegram_surface": {
            "sale_messages": sum(1 for m in sale_msgs
                                 if m.get("kind") == "beta_exit"),
            "modeled_messages": sum(1 for m in sale_msgs
                                    if m.get("kind") == "beta_exit_modeled"),
            "sale_arms_described": len(sale_arms),
            "modeled_arms_described": len(modeled_arms),
            "exit_blocked_messages": len(blocked_msgs),
            "parents_named_by_messages": sorted({m["parent_signal_id"]
                                                 for m in sale_msgs}),
            "delivery_ledger_present": False,
            "delivery_verifiable_offline": False,
            "note": (
                "messages are BUILT from persisted records only, and an arm is "
                "described as a SALE only when its exit basis is an executable "
                "SELL quote AND its settlement is on the durable book. There is "
                "NO persisted send ledger in this codebase, so actual Telegram "
                "DELIVERY cannot be verified offline and is not claimed"),
        },
        "forward_experiment": prog,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "v2_floor_beta_forward_delivery_audit.json"
    out.write_text(json.dumps(report, indent=2, default=str))

    print(f"main store        : {MAIN}")
    print(f"forward store     : {FORWARD} (present={FORWARD.exists()})")
    print(f"persisted closed  : {len(trades)} "
          f"(quote {len(persisted_quote)} / modeled {len(persisted_modeled)}) "
          f"parents {report['persisted']['unique_parents']} "
          f"realized ${persisted_realized}")
    print(f"outstanding arms  : {len(open_arms)} (unresolved {len(unresolved)}) "
          f"max attempts {report['outstanding']['max_exit_attempts']}")
    print(f"Status            : closed {status_closed} / unresolved "
          f"{status_unresolved} / realized ${status_realized}")
    print(f"Closed Positions  : rows {len(rows)} / realized ${rows_realized} "
          f"/ quote rows {rows_quote}")
    print(f"Telegram (built)  : sale msgs {report['telegram_surface']['sale_messages']}"
          f" / modeled msgs {report['telegram_surface']['modeled_messages']}"
          f" / exit-blocked {report['telegram_surface']['exit_blocked_messages']}")
    print(f"forward progress  : {prog['unique_parent_signals']} parents "
          f"/ {prog['completed_trades']} trades / "
          f"{prog['unresolved_arms']} unresolved / target "
          f"{prog['target_unique_parent_signals']}")
    print()
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\nALL CHECKS PASS: {report['all_checks_pass']}")
    print(f"report -> {out}")
    if status_text:
        print("\n--- Status block (first 12 lines) ---")
        print("\n".join(status_text.splitlines()[:12]))
    print(f"\n--- Closed Positions text length: {len(closed_text)} chars ---")
    return 0 if report["all_checks_pass"] else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-forward", action="store_true",
                    help="initialize/populate ONLY the forward store, then stop")
    args = ap.parse_args()
    raise SystemExit(init_forward() if args.init_forward else audit())


if __name__ == "__main__":
    main()
