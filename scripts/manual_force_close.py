#!/usr/bin/env python3
"""
One-off ADMINISTRATIVE force-close of the currently-open PAPER position(s).

WHAT THIS IS
============
This is NOT a strategy exit. It is an operator action that takes the PAPER book
flat so the execution-aware engine is no longer blocked by
``POSITION_ACTIVE``. Nothing here is a trading decision, and nothing here is
attributed to a strategy's exit logic.

EXECUTION PRIORITY (per the operator's instruction)
===================================================
    A. a currently valid EXECUTABLE SELL QUOTE on the position's own pinned
       venue;
    B. never a DexScreener mark dressed up as an executable fill;
    C. a clearly defined PAPER manual/emergency price source -- if one exists;
    D. otherwise close as ``FORCE_CLOSED_NO_EXECUTABLE_QUOTE`` with realized
       PnL recorded as UNKNOWN rather than fabricated.

There is no ``PAPER_MANUAL_PRICE`` source anywhere in this repository, so (C)
cannot be selected and the outcome is (A) or (D). The subtype recorded is
therefore ``EXECUTABLE_QUOTE`` or ``NO_EXECUTABLE_QUOTE``.

SAFETY
======
* PAPER only. No transaction is built, signed or broadcast; this script cannot
  reach a signer at all.
* It never touches ``var/real_account_state.json`` or ``.env.local``.
* It never changes strategy/entry/exit logic or any flag.
* It never alters or deletes historical records. The ledger is append-only and
  is only ever appended to. Only the ``open_positions`` list of
  ``var/paper_state.json`` is edited, and the file is backed up first.
* Dry-run is the DEFAULT. Pass ``--apply`` to actually mutate state.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from autopaper.paper_live.state import (  # noqa: E402
    PaperStateStore,
    PersistedState,
)
from autopaper.paper_live.quotes import QuoteService, SolReference  # noqa: E402
from autopaper.paper_live.fills import (  # noqa: E402
    FillModel,
    FillRefused,
    fill_from_quote,
    realize,
)
from autopaper.execution.chainstate import DEFAULT_PUBLIC_RPC_URL  # noqa: E402
from autopaper.same_market.rpc import SolanaReadOnlyRpc  # noqa: E402
from autopaper.same_market.jupiter import JupiterQuoteClient  # noqa: E402
from autopaper.ledger import Ledger  # noqa: E402

CLOSE_REASON = "MANUAL_FORCE_CLOSE"
SUBTYPE_EXECUTABLE = "EXECUTABLE_QUOTE"
SUBTYPE_MANUAL = "PAPER_MANUAL_PRICE"
SUBTYPE_NONE = "NO_EXECUTABLE_QUOTE"


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _fingerprint(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "bytes": st.st_size,
        "mtime": st.st_mtime,
        "sha256": _sha256(path),
    }


# ---------------------------------------------------------------------------
# quote probe
# ---------------------------------------------------------------------------


async def _probe(
    record: dict, *, rpc_url: str, require_identity: bool, allow_jupiter: bool,
) -> dict:
    """Ask the real quote stack for an executable SELL on this exact position."""
    rpc = SolanaReadOnlyRpc(rpc_url, timeout_s=10.0, max_rpm=120, cache_ttl_s=1.0)
    jupiter = JupiterQuoteClient(timeout_s=10.0, cache_ttl_s=30.0)

    sol_ref: SolReference | None = None
    sol_note = None
    try:
        from autopaper.same_market.sol_price import (
            JupiterSolPriceProvider,
            resolve_sol_usd_async,
        )

        provider = JupiterSolPriceProvider(timeout_s=10.0, ttl_s=60.0, max_rpm=20)
        try:
            snap = await resolve_sol_usd_async(provider=provider)
            sol_ref = SolReference(snap.usd_per_sol, snap.timestamp, snap.source)
        finally:
            await provider.aclose()
    except Exception as exc:  # noqa: BLE001 - reference is display-only here
        sol_note = f"{type(exc).__name__}: {exc}"

    qs = QuoteService(
        rpc=rpc,
        sol_reference=sol_ref,
        jupiter=jupiter,
        max_quote_age_ms=30_000.0,
        allow_jupiter_fallback=allow_jupiter,
        require_identity=require_identity,
    )

    mint = record["mint"]
    tokens_raw = int(record["tokens_raw"])
    decimals = int(record["token_decimals"])
    pinned_venue = record.get("entry_venue_id") or None
    pinned_pool = record.get("entry_pool_address") or None

    # The identity anchor we hold on disk is the market the position was opened
    # against. Production compares the live mark's pair to the chain-verified
    # market; we have no live mark here, so the pinned market is used as the
    # anchor. This is recorded explicitly so the choice is auditable.
    try:
        pinned, counterfactual = await qs.quote_sell_pinned(
            mint=mint,
            token_amount_raw=tokens_raw,
            token_decimals=decimals,
            pinned_venue_id=pinned_venue,
            pinned_pool_address=pinned_pool,
            observed_pair_address=pinned_pool,
        )
    except Exception as exc:  # noqa: BLE001
        pinned, counterfactual = None, None
        pinned_error = f"{type(exc).__name__}: {exc}"
    else:
        pinned_error = None

    # Any executable venue at all -- evidence about graduation, never a
    # settlement.
    try:
        generic = await qs.quote_sell(
            mint=mint, token_amount_raw=tokens_raw, token_decimals=decimals,
            observed_pair_address=None,
        )
    except Exception as exc:  # noqa: BLE001
        generic, generic_error = None, f"{type(exc).__name__}: {exc}"
    else:
        generic_error = None

    return {
        "rpc_url": rpc_url,
        "sol_reference": sol_ref.as_dict() if sol_ref else None,
        "sol_reference_error": sol_note,
        "pinned_venue_id": pinned_venue,
        "observed_pair_address_used": pinned_pool,
        "pinned_quote": pinned.as_dict() if hasattr(pinned, "as_dict") else None,
        "pinned_error": pinned_error,
        "counterfactual_quote": (
            counterfactual.as_dict()
            if hasattr(counterfactual, "as_dict") else None
        ),
        "any_venue_quote": (
            generic.as_dict() if hasattr(generic, "as_dict") else None
        ),
        "any_venue_error": generic_error,
        "_pinned": pinned,
        "_generic": generic,
        "_sol_ref": sol_ref,
        "_jupiter": jupiter,
    }


def _executable(quote, *, require_identity: bool) -> bool:
    if quote is None or not getattr(quote, "available", False):
        return False
    if getattr(quote, "stale", False):
        return False
    if require_identity and getattr(quote, "identity", None) != "PASS":
        return False
    return True


def _close_record(
    record: dict, probe: dict, *, require_identity: bool, now: float,
) -> dict:
    """Build the audit record for one administrative close."""
    pinned = probe["_pinned"]
    generic = probe["_generic"]
    sol_ref = probe["_sol_ref"]

    entry_fill = record.get("entry_fill") or {}
    opened_at = record.get("opened_at")
    hold_s = (max(0.0, now - float(opened_at))
              if isinstance(opened_at, (int, float)) else None)

    executable = _executable(pinned, require_identity=require_identity)
    subtype = SUBTYPE_EXECUTABLE if executable else SUBTYPE_NONE
    disposition = {
        SUBTYPE_EXECUTABLE: "CLOSED_AT_EXECUTABLE_QUOTE",
        SUBTYPE_MANUAL: "CLOSED_AT_PAPER_MANUAL_PRICE",
        SUBTYPE_NONE: "FORCE_CLOSED_NO_EXECUTABLE_QUOTE",
    }[subtype]

    # Graduation evidence: the pinned bonding curve is complete and a DIFFERENT
    # venue (a PumpSwap pool) is the one holding the migrated liquidity.
    pinned_reason = " ".join(str(x) for x in (
        getattr(pinned, "missing_reason", None) if pinned is not None else None,
        probe.get("pinned_error"),
    ) if x)
    any_venue = generic if hasattr(generic, "available") else None
    curve_complete = ("curve is complete" in pinned_reason
                      or "no curve to sell into" in pinned_reason)
    migrated_pool = (getattr(any_venue, "pool_or_curve", None)
                     if any_venue is not None else None)
    any_venue_is_swap = (any_venue is not None
                         and getattr(any_venue, "available", False)
                         and getattr(any_venue, "market_type", None) == "PUMPSWAP")
    graduated = bool(
        str(record.get("entry_market_type") or "") == "PUMP_BONDING_CURVE"
        and not executable
        and (curve_complete or any_venue_is_swap)
    )

    out: dict = {
        "close_reason": CLOSE_REASON,
        "close_subtype": subtype,
        "close_disposition": disposition,
        "administrative": True,
        "not_a_strategy_exit": True,
        "paper_only": True,
        "closed_at": now,
        "closed_at_iso": _iso(now),

        "arm": record.get("arm"),
        "signal_id": record.get("signal_id"),
        "symbol": record.get("symbol"),
        "mint": record.get("mint"),
        "token_decimals": record.get("token_decimals"),
        "tokens_raw": record.get("tokens_raw"),
        "tokens_decimal": record.get("tokens_decimal"),

        "entry_timestamp": opened_at,
        "entry_timestamp_iso": _iso(opened_at if isinstance(opened_at, (int, float)) else None),
        "entry_fill_price_sol": entry_fill.get("effective_price"),
        "entry_fill_price_usd": entry_fill.get("effective_price_usd"),
        "entry_cost_quote_asset_sol": record.get("entry_cost_quote_asset"),
        "entry_notional_usd": record.get("usd_notional"),
        "production_profile": record.get("production_profile"),

        "hold_duration_seconds": hold_s,
        "hold_duration_hours": (hold_s / 3600.0) if hold_s is not None else None,

        "pinned_venue_id": record.get("entry_venue_id"),
        "execution_venue": record.get("entry_market_type"),
        "venue": record.get("entry_fill", {}).get("venue")
        if isinstance(record.get("entry_fill"), dict) else None,
        "entry_pool_address": record.get("entry_pool_address"),

        "position_state_before": {
            "exit_status": record.get("exit_status"),
            "exit_attempts": record.get("exit_attempts"),
            "last_exit_error": record.get("last_exit_error"),
            "exit_trigger_reason": record.get("exit_trigger_reason"),
            "rent_locked": record.get("rent_locked"),
            "reclaim_status": record.get("reclaim_status"),
            "reclaim_blocker": record.get("reclaim_blocker"),
        },

        "exit_price_sol": None,
        "exit_price_usd": None,
        "exit_price_source": None,
        "exit_proceeds_quote_sol": None,
        "quote_timestamp": None,
        "quote_age_ms": None,
        "quote_identity": None,
        "realized_pnl_usd": None,
        "realized_pnl_sol": None,
        "realized_pnl_basis": None,

        "graduated_or_unexecutable": not executable,
        "graduated": graduated,
        "graduated_pool_address": migrated_pool if graduated else None,
        "unexecutable_reason": (None if executable else (
            getattr(pinned, "missing_reason", None) if pinned is not None
            else probe.get("pinned_error") or "no pinned quote object")),

        # The any-venue quote is EVIDENCE of graduation only. It is a different
        # book, so selling there would be a different trade; it is never the
        # settlement. This flag makes that explicit on the record.
        "counterfactual_any_venue": {
            "used_for_settlement": False,
            "rule": (
                "a sell routed to a venue other than the pinned entry venue is "
                "a different book and therefore a different trade; it is "
                "recorded here as evidence and never as the settlement"
            ),
            "market_type": (getattr(generic, "market_type", None)
                            if any_venue is not None else None),
            "provider": (getattr(generic, "provider", None)
                         if any_venue is not None else None),
            "pool_or_curve": migrated_pool,
            "available": (getattr(generic, "available", None)
                          if any_venue is not None else None),
            "effective_price_sol": (
                getattr(generic, "effective_price", None)
                if any_venue is not None else None),
        },

        "quote_probe": {
            "rpc_url": probe.get("rpc_url"),
            "observed_pair_address_used": probe.get("observed_pair_address_used"),
            "sol_reference": probe.get("sol_reference"),
            "sol_reference_error": probe.get("sol_reference_error"),
            "pinned_quote": probe.get("pinned_quote"),
            "pinned_error": probe.get("pinned_error"),
            "counterfactual_quote": probe.get("counterfactual_quote"),
            "any_venue_quote": probe.get("any_venue_quote"),
            "any_venue_error": probe.get("any_venue_error"),
            "jupiter_quote_sell_available": hasattr(qs_jupiter, "quote_sell")
            if (qs_jupiter := probe.get("_jupiter")) is not None else None,
        },
    }

    if executable:
        out["exit_price_sol"] = getattr(pinned, "effective_price", None)
        out["exit_price_usd"] = getattr(pinned, "effective_price_usd", None)
        out["exit_price_source"] = (
            f"executable sell quote for the exact size held; provider="
            f"{getattr(pinned, 'provider', None)} venue="
            f"{getattr(pinned, 'venue_id', None) or getattr(pinned, 'venue', None)}"
        )
        out["quote_timestamp"] = getattr(pinned, "quote_timestamp", None)
        out["quote_age_ms"] = getattr(pinned, "quote_age_ms", None)
        out["quote_identity"] = getattr(pinned, "identity", None)
        out["realized_pnl_basis"] = (
            "simulated settlement between the stored entry SIMULATED_FILL and "
            "a fresh exit SIMULATED_FILL from the executable sell quote; never "
            "exit_mark - entry_mark"
        )
        out["_exit_fill_plan"] = True
    return out


def _settle_pnl(record: dict, probe: dict, close: dict) -> None:
    """Fill in realized PnL when an executable quote exists."""
    from autopaper.paper_live.engine import _position_from_dict

    position = _position_from_dict(record)
    if position is None:
        close["realized_pnl_basis"] += (
            " [NOT COMPUTED: the stored position could not be rebuilt]"
        )
        close["realized_pnl_usd"] = None
        close["realized_pnl_sol"] = None
        return
    pinned = probe["_pinned"]
    sol_ref = probe["_sol_ref"]
    try:
        exit_fill = fill_from_quote(
            pinned, side="SELL", now=close["closed_at"],
            model=FillModel(), sol_reference=sol_ref, require_identity=True,
        )
    except FillRefused as exc:
        close["realized_pnl_basis"] += f" [NOT COMPUTED: {exc}]"
        return
    pnl = realize(
        position, exit_fill=exit_fill, exit_observed_mark=None,
        sol_reference=sol_ref, model=FillModel(), now=close["closed_at"],
    )
    close["exit_proceeds_quote_sol"] = float(pnl.exit_proceeds_quote)
    close["realized_pnl_usd"] = float(pnl.realized_pnl_usd)
    close["realized_pnl_sol"] = float(pnl.realized_pnl_sol)
    close["return_pct"] = float(pnl.return_pct) if pnl.return_pct is not None else None
    close["bankroll_before"] = float(pnl.bankroll_before)
    close["bankroll_after"] = float(pnl.bankroll_after)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-dir", default=str(ROOT / "var"),
                    help=("directory holding paper_state.json "
                          "(default: the repository's own var/)"))
    ap.add_argument("--ledger-dir", default=str(ROOT / "var"))
    ap.add_argument("--reports-dir", default=str(ROOT / "reports/forensics"))
    ap.add_argument("--rpc-url", default=DEFAULT_PUBLIC_RPC_URL)
    ap.add_argument("--no-jupiter", action="store_true",
                    help="do not consult Jupiter as a fallback route")
    ap.add_argument("--allow-identity-unknown", action="store_true",
                    help="treat an UNKNOWN same-market identity as executable")
    ap.add_argument("--apply", action="store_true",
                    help="actually mutate state (default is dry-run)")
    args = ap.parse_args(argv)

    require_identity = not args.allow_identity_unknown
    # RESOLVED ONCE, explicitly. The default used to be the bare string "var",
    # so running this from any other directory silently read -- and with
    # --apply, WROTE -- a different state file than the one the runtime uses.
    # The runtime state root is now named, not inherited from the CWD.
    state_dir = Path(args.state_dir).expanduser().resolve()
    # ``runtime=True``: this IS an operator action against production state, and
    # it says so. A test or probe constructing the same store without that flag
    # is refused by ``PaperStateStore``.
    store = PaperStateStore(state_dir, runtime=True)
    state_path = store.path

    real_path = state_dir / "real_account_state.json"
    legacy_path = state_dir / "paper_account.json"
    env_path = ROOT / ".env.local"

    before = {
        "paper_state": _fingerprint(state_path),
        "real_account_state": _fingerprint(real_path),
        "legacy_paper_account": _fingerprint(legacy_path),
        "env_local": _fingerprint(env_path),
    }

    state: PersistedState = store.load()
    ok, problems = state.validate()

    open_positions = list(state.open_positions.values())
    print("=" * 78)
    print(f"ADMINISTRATIVE PAPER FORCE-CLOSE   mode={'APPLY' if args.apply else 'DRY-RUN'}")
    print("=" * 78)
    print(f"paper_state       : {state_path}")
    print(f"schema_version    : {state.schema_version}  validate_ok={ok}")
    if problems:
        for p in problems:
            print(f"  - validation: {p}")
    print(f"counts            : {state.as_dict()['counts']}")
    print(f"open PAPER positions in THIS store: {len(open_positions)}")

    # The legacy engine's account is a different store; report it, never touch.
    legacy_open = None
    if legacy_path.exists():
        try:
            legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
            legacy_open = len(legacy.get("open_positions") or [])
        except Exception as exc:  # noqa: BLE001
            legacy_open = f"unreadable: {exc}"
    print(f"open positions in LEGACY paper_account.json: {legacy_open}")
    print()

    if not open_positions:
        print("Nothing to close: no open PAPER positions.")
        return 0

    now = time.time()
    records: list[dict] = []
    for rec in open_positions:
        print("-" * 78)
        print(f"POSITION  {rec.get('arm')}  {rec.get('symbol')}  {rec.get('signal_id')}")
        print(f"  mint              : {rec.get('mint')}")
        print(f"  pinned venue      : {rec.get('entry_venue_id')}")
        print(f"  entry fill price  : {((rec.get('entry_fill') or {}).get('effective_price'))} SOL/token "
              f"({((rec.get('entry_fill') or {}).get('effective_price_usd'))} USD/token)")
        print(f"  opened_at         : {rec.get('opened_at')}  ({_iso(rec.get('opened_at'))})")
        print(f"  exit_status       : {rec.get('exit_status')}  attempts={rec.get('exit_attempts')}")
        print(f"  last_exit_error   : {str(rec.get('last_exit_error'))[:160]}")
        print("  probing executable SELL on the pinned venue ...")
        probe = asyncio.run(_probe(
            rec, rpc_url=args.rpc_url, require_identity=require_identity,
            allow_jupiter=not args.no_jupiter))
        close = _close_record(rec, probe, require_identity=require_identity, now=now)
        if close["close_subtype"] == SUBTYPE_EXECUTABLE:
            _settle_pnl(rec, probe, close)
        print(f"  -> subtype        : {close['close_subtype']}")
        if close["close_subtype"] == SUBTYPE_EXECUTABLE:
            print(f"     exit price    : {close['exit_price_sol']} SOL/token")
            print(f"     realized PnL  : ${close['realized_pnl_usd']}")
        else:
            print(f"     realized PnL  : UNKNOWN (no executable quote; not fabricated)")
            print(f"     reason        : {str(close['unexecutable_reason'])[:200]}")
        records.append(close)

    report = {
        "action": CLOSE_REASON,
        "mode": "APPLY" if args.apply else "DRY_RUN",
        "generated_at": now,
        "generated_at_iso": _iso(now),
        "paper_only": True,
        "not_a_strategy_exit": True,
        "close_reason": CLOSE_REASON,
        "state_file": str(state_path),
        "open_positions_before": len(open_positions),
        "positions": records,
        "fingerprints_before": before,
        "applied": False,
    }

    reports_dir = Path(args.reports_dir).expanduser().resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if not args.apply:
        out_path = reports_dir / f"manual_force_close_{stamp}.dryrun.json"
        out_path.write_text(json.dumps(report, indent=2, default=str),
                            encoding="utf-8")
        print()
        print(f"DRY-RUN. No state was changed. Plan written to {out_path}")
        print("Re-run with --apply to execute the administrative close.")
        return 0

    # -- APPLY --------------------------------------------------------------
    # 1. Back up the state file (additive; nothing is deleted or rewritten).
    backup = state_path.with_name(
        f"{state_path.name}.pre_manual_force_close.{stamp}.bak")
    shutil.copy2(state_path, backup)
    print()
    print(f"backup            : {backup}")

    # 2. Append the audit records to the append-only ledger FIRST, so the intent
    #    to close exists in the audit trail even if the state write is killed.
    ledger = Ledger(
        Path(args.ledger_dir) / "ledger.jsonl",
        Path(args.ledger_dir) / "decisions.jsonl",
        Path(args.ledger_dir) / "candidates.jsonl",
        Path(args.ledger_dir) / "position_updates.jsonl",
    )
    for close in records:
        ledger.note(
            "MANUAL_FORCE_CLOSE (administrative; NOT a strategy exit)",
            **close,
        )
    ledger.note(
        "MANUAL_FORCE_CLOSE batch summary",
        close_reason=CLOSE_REASON,
        administrative=True,
        not_a_strategy_exit=True,
        closed=len(records),
        subtypes=[c["close_subtype"] for c in records],
        positions=[c["signal_id"] for c in records],
    )

    # 3. Drop the closed positions from durable state and save atomically.
    keys = [f"{c['arm']}::{c['signal_id']}" for c in records]
    removed = state.close_positions(keys)
    written = store.save(
        state,
        note=(f"{CLOSE_REASON}: removed {removed} open PAPER position(s) "
              f"administratively; " +
              ", ".join(f"{c['signal_id']}={c['close_subtype']}" for c in records)),
    )

    after = {
        "paper_state": _fingerprint(state_path),
        "real_account_state": _fingerprint(real_path),
        "legacy_paper_account": _fingerprint(legacy_path),
        "env_local": _fingerprint(env_path),
    }

    # 4. Re-load and prove nothing will be resumed.
    reloaded = store.load()
    resumable, problems = store.rehydrate(reloaded)
    ok2, problems2 = reloaded.validate()

    report.update({
        "applied": True,
        "removed": removed,
        "save_result": written,
        "fingerprints_after": after,
        "verify": {
            "open_positions_after": len(reloaded.open_positions),
            "validate_ok": ok2,
            "validate_problems": problems2,
            "rehydrate_restored": [f"{p.get('arm')}/{p.get('signal_id')}" for p in resumable],
            "rehydrate_problems": problems,
            "real_state_unchanged": (
                before["real_account_state"].get("sha256")
                == after["real_account_state"].get("sha256")),
            "env_local_unchanged": (
                before["env_local"].get("sha256")
                == after["env_local"].get("sha256")
                and before["env_local"].get("mtime")
                == after["env_local"].get("mtime")),
            "ledger_appended_only": True,
        },
    })

    out_path = reports_dir / f"manual_force_close_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str),
                        encoding="utf-8")
    _write_markdown(reports_dir / f"manual_force_close_{stamp}.md", report)

    print()
    print("=" * 78)
    print(f"APPLIED. removed={removed}  open_positions_after={len(reloaded.open_positions)}")
    print(f"real state unchanged : {report['verify']['real_state_unchanged']}")
    print(f".env.local unchanged : {report['verify']['env_local_unchanged']}")
    print(f"rehydrate restores   : {report['verify']['rehydrate_restored']}")
    print(f"report               : {out_path}")
    print("=" * 78)
    return 0


def _write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Administrative PAPER force-close",
        "",
        f"- action: `{report['action']}`",
        f"- generated_at: `{report['generated_at_iso']}`",
        "- **this is NOT a strategy exit** — it is an operator cleanup",
        f"- open positions before: `{report['open_positions_before']}`",
        f"- open positions after: `{report['verify']['open_positions_after']}`",
        f"- REAL state unchanged: `{report['verify']['real_state_unchanged']}`",
        f"- `.env.local` unchanged: `{report['verify']['env_local_unchanged']}`",
        "",
        "| arm | symbol | mint | subtype | exit price (SOL) | realized PnL (USD) | hold (s) | venue | graduated/unexecutable |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in report["positions"]:
        lines.append(
            "| {arm} | {sym} | `{mint}` | {sub} | {px} | {pnl} | {hold} | `{ven}` | {gu} |".format(
                arm=c.get("arm"),
                sym=c.get("symbol"),
                mint=(c.get("mint") or "")[:12] + "…",
                sub=c.get("close_subtype"),
                px=("n/a" if c.get("exit_price_sol") is None
                    else f"{c['exit_price_sol']:.6g}"),
                pnl=("UNKNOWN" if c.get("realized_pnl_usd") is None
                     else f"{c['realized_pnl_usd']:.6f}"),
                hold=("n/a" if c.get("hold_duration_seconds") is None
                      else f"{c['hold_duration_seconds']:.0f}"),
                ven=c.get("pinned_venue_id"),
                gu=c.get("graduated_or_unexecutable"),
            ))
    lines += ["", "## Unexecutable reasons", ""]
    for c in report["positions"]:
        lines.append(f"- `{c.get('signal_id')}` ({c.get('symbol')}): "
                     f"[{c.get('close_disposition')}] "
                     f"graduated={c.get('graduated')} "
                     f"migrated_pool={c.get('graduated_pool_address')} :: "
                     f"{c.get('unexecutable_reason')}")
    lines += ["", "## Quote probe (pinned vs any-venue)", ""]
    for c in report["positions"]:
        qp = c.get("quote_probe") or {}
        lines.append(f"- `{c.get('signal_id')}`:")
        lines.append(f"  - pinned: {json.dumps(qp.get('pinned_quote'))[:400]}")
        lines.append(f"  - pinned_error: {qp.get('pinned_error')}")
        lines.append(f"  - any_venue: {json.dumps(qp.get('any_venue_quote'))[:400]}")
        lines.append(f"  - any_venue_error: {qp.get('any_venue_error')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
