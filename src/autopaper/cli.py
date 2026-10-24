"""
Command line interface.

    python -m autopaper.cli run          start the bot (does NOT trade yet)
    python -m autopaper.cli status       one-shot status + safety audit
    python -m autopaper.cli preflight    read-only REAL preflight
    python -m autopaper.cli stop         STOP: no new positions, keep managing
    python -m autopaper.cli flatten      FLATTEN: close open paper positions now
    python -m autopaper.cli login        one-time Telegram session creation
    python -m autopaper.cli config       show effective configuration
    python -m autopaper.cli report       full paper performance report
    python -m autopaper.cli verify       migration + safety audit
    python -m autopaper.cli audit-deps   dependency + import audit
    python -m autopaper.cli same-market-test
                                        READ-ONLY harness: measures whether a
                                        signal's market is the market an
                                        execution would use, and how close the
                                        executable quote is to the observed
                                        mark. Broadcasts nothing.

STARTUP IS SAFE
---------------
``run`` initializes the data services and the Telegram listener and then waits.
It does not open a position and it does not arm REAL. Candidate processing
begins only when the operator presses the Telegram start button, and REAL only
on a confirmed press plus a passing preflight.

There is deliberately NO CLI command that arms REAL. The CLI can inspect the
real path (``preflight``) but cannot authorize it, because authorization is an
operator action in Telegram and a terminal is not that.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import time
from pathlib import Path

from . import __version__
from .account_lock import (
    AccountAlreadyLocked, AccountLockUnavailable, LegacyWriterSuspected,
    assert_no_legacy_writer, describe_holder, identity_for, pid_is_alive,
)
from .config import ENV_FILENAME, Config, build_config, describe
from .ledger import Ledger, load_json, save_json_atomic
from .logging_setup import configure as configure_logging
from .logging_setup import log
from .memory import ResearchMemory
from .run_isolation import RunPathViolation, assert_run_path_isolated
from .safety import AGENT_ID, STRATEGY, audit
from .strategy_profile import label as _sp_label, max_hold_s as _sp_max_hold_s
from .strategy_profile import parse_profile as _sp_parse

CMD_STOP = "STOP"
CMD_FLATTEN = "FLATTEN"


def _ledger(cfg: Config) -> Ledger:
    return Ledger(
        cfg.ledger_file, cfg.decisions_file, cfg.candidates_file, cfg.updates_file
    )


def _trader(cfg: Config):
    from .engine import PaperTrader
    from .nar import NarService
    from .telegram import TelegramListener

    listener = TelegramListener(cfg.telegram, on_alert=None)
    nar = NarService(cfg.nar)
    trader = PaperTrader(cfg, nar=nar)
    trader.listener = listener
    listener.on_alert = trader.on_alert
    # restore the dedupe window from the ledger so a restart cannot replay
    listener.restore_seen(_ledger(cfg).seen_message_keys())
    return trader


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    configure_logging(cfg.log_dir, args.log_level)
    # CONTAINMENT: refuse to start if real-execution capability is live.
    #
    # G1 (var/audit.md) established that `solders` is reachable in the import
    # graph from this function, via engine.py:90 -> execution/real.py:63 ->
    # execution/solana.py:327, the last two being lazy in-function imports.
    # That graph cannot be cleaned without a refactor nobody asked for, so what
    # is enforced here instead is the fact the paper deployment's safety
    # actually rests on: the capability-bearing distributions are NOT INSTALLED.
    #
    # Checked BEFORE the engine is constructed, so a machine that has gained the
    # capability never reaches the alert, decision, sizing or exit paths. It is
    # a startup refusal, not a trading rule: nothing below this line behaves
    # differently because of it.
    try:
        assert_run_path_isolated()
    except RunPathViolation as exc:
        log("SYSTEM", "refusing to start: %s", exc)
        print(f"\nREFUSED: {exc}\n", file=sys.stderr)
        return 6
    # --cli MUST reach the config before the engine is constructed.
    #
    # PaperTrader builds the Telegram bot (transport, notifier, poller, queue)
    # inside __init__. Setting trader.cli_only afterwards was too late: the
    # outbound machinery already existed and its worker was already running, so
    # `--cli` disabled nothing. Config is frozen, so rebuild it.
    if bool(getattr(args, "cli", False)) != cfg.cli_only:
        cfg = dataclasses.replace(cfg, cli_only=bool(args.cli))
    trader = _trader(cfg)
    if args.reset_control:
        save_json_atomic(cfg.control_file, {"command": "", "at": 0.0})
    try:
        asyncio.run(_run_with_flags(trader, args))
    except KeyboardInterrupt:
        log("SYSTEM", "interrupted")
        if trader.owns_account:
            trader.persist()
    except AccountAlreadyLocked as exc:
        # The whole point of the lock: this process never reached the alert,
        # decision, sizing or exit paths, so it cannot have written anything.
        log("ACCOUNT", "%s", exc)
        print(f"\n{exc}\n", file=sys.stderr)
        if exc.holder:
            print(f"  holder : pid={exc.holder.get('pid')} host={exc.holder.get('host')} "
                  f"since={exc.holder.get('acquired_at')}", file=sys.stderr)
        print(f"  account: {exc.identity.label if exc.identity else cfg.state_file}",
              file=sys.stderr)
        print("  this process made no ledger, state or trading mutation.\n"
              "  If the holder is genuinely gone, no cleanup is required: the\n"
              "  operating system releases the lock when a process dies.\n",
              file=sys.stderr)
        return 3
    except LegacyWriterSuspected as exc:
        log("ACCOUNT", "%s", exc)
        print(f"\n{exc}\n", file=sys.stderr)
        return 5
    except AccountLockUnavailable as exc:
        log("ACCOUNT", "refusing to run: %s", exc)
        print(f"\nREFUSED: {exc}\n", file=sys.stderr)
        return 4
    return 0


async def _run_with_flags(trader, args: argparse.Namespace) -> None:
    """
    ``PaperTrader.run`` owns startup, so the adopt flag is threaded through the
    instance rather than through run()'s signature - the public call sites and
    the existing tests keep working unchanged.
    """
    trader.adopt_legacy_account = bool(getattr(args, "adopt_legacy_account", False))
    # The engine was already built with this value (cmd_run folds --cli into the
    # config first). Kept as a read so a test that constructs a trader directly
    # still works, but it is NOT the mechanism.
    if trader.cli_only != bool(getattr(args, "cli", False)):
        raise SystemExit(
            "internal: --cli must be applied to the config before _trader()"
        )
    if trader.cli_only:
        log("SYSTEM", "CLI-only mode: Telegram outbound disabled; "
                      "channel ingestion unchanged")
    await trader.run()


def cmd_status(cfg: Config, args: argparse.Namespace) -> int:
    """
    One-shot status.

    Prints the SAME status message the Telegram UI renders, from the SAME view
    object, so the two surfaces cannot report different things. A discrepancy
    between a terminal and a chat is exactly what this avoids.
    """
    configure_logging(None, args.log_level)
    state = load_json(cfg.state_file)
    ledger = _ledger(cfg)
    events = ledger.summary()
    report = audit()
    from .engine import PaperTrader
    from .modes import RuntimeState
    from .telegram import ui

    view = _offline_status_view(cfg, state)
    print(f"{AGENT_ID} v{__version__}")
    print("-" * 64)
    print("TELEGRAM VIEW")
    print("-" * 64)
    print(ui.render_status(view))
    print("-" * 64)
    print("CLI VIEW (same canonical state)")
    print("-" * 64)
    if state:
        acct = state.get("account", {})
        print(f"bankroll (starting)  ${acct.get('starting_bankroll_usd', 0):.2f}")
        print(f"cash                 ${acct.get('cash_usd', 0):.2f}")
        print(f"realized pnl         {acct.get('realized_pnl_usd', 0):+.2f}")
        print(f"fees paid            ${acct.get('fees_paid_usd', 0):.2f}")
        print(f"slippage paid        ${acct.get('slippage_paid_usd', 0):.2f}")
        print(f"open positions       {len(state.get('open_positions', []))}")
        print(f"run count            {acct.get('run_count', 0)}")
        print(f"counters             {state.get('counters', {})}")
        print(f"last heartbeat       {acct.get('last_heartbeat')}")
        print(f"account id           {state.get('account_id', 'legacy (unstamped)')}")
        ident = identity_for(cfg.state_file)
        holder = describe_holder(ident.lock_path)
        print(f"lock                 {ident.lock_path}")
        print(f"lock holder          "
              f"pid={holder.get('pid')} (alive={pid_is_alive(holder.get('pid'))})")
    else:
        print("no persisted paper account yet (run `autopaper run` once)")
    print("-" * 64)
    print(f"ledger events        {events}")
    print(f"control              {load_json(cfg.control_file) or 'none'}")
    print("-" * 64)
    print("MIGRATION AUDIT")
    print(report.render())
    print("-" * 64)
    print("JOURNAL")
    print(json.dumps(_journal_summary(cfg), indent=2))
    print("-" * 64)
    print("REAL")
    print(f"  trading_enabled       {cfg.real.trading_enabled}  "
          "(always False from a fresh config; arming is a Telegram action)")
    print(f"  env request ignored   {cfg.real.env_trading_enabled_request}")
    print(f"  missing capabilities  {cfg.real.missing_capabilities() or 'none'}")
    return 0 if report.ok else 1


def _offline_status_view(cfg: Config, state: dict | None):
    """
    A StatusView assembled WITHOUT constructing the engine.

    The engine acquires an account lock, so building it just to print a status
    would take the single-writer lock away from a running bot. The state read
    here comes from the persisted file, and the mode is read from the persisted
    emergency latch - which is the only mode fact that survives a restart,
    because nothing else is persistent by design.
    """
    from .ledger import load_json as _load
    from .modes import RuntimeState
    from .telegram.ui import StatusView

    s = RuntimeState()
    latch = _load(cfg.var_dir / "emergency_stop.json") or {}
    s.restore_emergency_stop(bool(latch.get("emergency_stop")))
    acct = (state or {}).get("account", {}) if state else {}
    # EQUITY IS None WHEN UNMEASURED, NOT 0.0.
    #
    # ``StatusView.equity_usd`` documents ``None`` as UNKNOWN and the renderer
    # prints an explicit "not measured" for it, precisely so an unmeasured
    # equity is never shown as a measured zero. This producer then did
    # ``float(acct.get("equity_usd") or 0.0)``, which both coerced a missing key
    # to ``0.0`` AND coerced a real ``0.0`` through the same path -- so
    # ``autopaper status`` reported ``Equity $0.0000`` for a state file with no
    # equity field at all. A default that claims a measurement was made.
    equity = acct.get("equity_usd")
    return StatusView(
        state=s,
        channel=cfg.telegram.CANDIDATE_SOURCE,
        chat=cfg.telegram.chat,
        wallet_masked="(not configured)",
        balance_sol=None,
        balance_usd=None,
        equity_usd=(float(equity) if equity is not None else None),
        cash_usd=acct.get("cash_usd"),
        open_positions=len((state or {}).get("open_positions", []) or []),
        executor_available=(
            cfg.real.wallet_configured and cfg.real.rpc_configured
        ),
        real_preflight_ok=None,
        emergency_stop=s.emergency_stop,
        # The persisted signal-source SELECTION, not the dataclass default. It
        # was omitted, so ``autopaper status`` printed ``SOURCE: AUTO`` whatever
        # the operator had actually chosen -- the offline surface disagreeing
        # with the running bot on the one field that decides which feed may
        # open a position.
        signal_source=(
            str((_load(cfg.var_dir / "signal_source_selection.json") or {})
                .get("signal_source") or "AUTO")
        ),
    )


def _journal_summary(cfg: Config) -> dict:
    from .journal import Journal

    return Journal(cfg.var_dir / "journal").summary()


SETUP_WSOL_PHRASE = "I UNDERSTAND THIS CREATES A REAL ON-CHAIN ACCOUNT"


def cmd_setup_wsol(cfg: Config, args: argparse.Namespace) -> int:
    """
    Report, and optionally prepare, the wallet's wrapped-SOL account.

    WHY THIS EXISTS
    ---------------
    A SOL-denominated BUY used to be routed through Jupiter with
    ``wrapAndUnwrapSol: true``, which makes Jupiter create, initialise and
    CLOSE a temporary wrapped-SOL account inside the swap. Those three extra
    instructions plus a dozen extra account keys are what push the serialized
    transaction past Solana's hard 1232-byte limit on a thin L25 token, where
    the node rejects it as unserializable and the BUY dies at
    ``simulateTransaction`` with -32602.

    Trading through an account that already exists removes that trio. This
    command is how an operator provisions it.

    SAFETY
    ------
    * Default is READ-ONLY. It prints and returns; it changes nothing.
    * Creating the account on chain requires ``--confirm`` AND an exact typed
      phrase. Without both it refuses.
    * Nothing here is wired into ``run``. There is no automatic provisioning
      path: the account is created once, deliberately, by a human.
    * No private key, seed or secret is ever printed. The command shows the
      wallet ADDRESS and the derived ACCOUNT address, both public.
    * It does not arm REAL and cannot.
    """
    import asyncio

    from .execution.solana import (SolanaRpc, SolanaWallet, WSOL_MINT,
                                   wsol_ata_address)

    def _resolve(pubkey) -> str:
        return str(pubkey)

    async def _work() -> int:
        wallet = SolanaWallet(cfg.real.wallet_keypair_path)
        owner = _resolve(wallet.pubkey)
        rpc = SolanaRpc(cfg.real.rpc_url)
        try:
            ata = wsol_ata_address(owner, WSOL_MINT)
            health = await rpc.get_health()
            print()
            print("  wrapped-SOL provisioning")
            print("  " + "-" * 58)
            print(f"  wallet address      {owner}")
            print(f"  WSOL mint           {WSOL_MINT}")
            print(f"  WSOL ATA address    {ata}")
            print(f"  RPC reachable       {getattr(health, 'reachable', False)}")
            print()
            if not getattr(health, "reachable", False):
                print("  WSOL_ATA_PRESENT    unknown (RPC unreachable)")
                print("  REFUSING: cannot verify state, will not guess.")
                return 2
            ready = await rpc.wsol_readiness(owner, 1)
            print(f"  WSOL_ATA_PRESENT    {ready.ata_present}")
            print(f"  WSOL_BALANCE        {ready.balance_lamports} lamports")
            print()
            if ready.ata_present:
                print("  Nothing to do - the account already exists.")
                if not ready.sufficient:
                    print("  It holds no balance, so a wrapAndUnwrapSol=false BUY")
                    print("  would still fail. Wrap native SOL into it first.")
                return 0
            print("  This account does NOT exist yet.")
            print()
            print("  Creating it would send a real transaction and cost rent")
            print("  (~0.00203942 SOL). It is deliberately NOT automatic and is")
            print("  not wired into `run`.")
            print()
            print("  If you are certain, re-run with BOTH:")
            print(f'      --confirm "{SETUP_WSOL_PHRASE}"')
            return 1
        finally:
            await rpc.aclose()

    # ON-CHAIN CREATION IS INTENTIONALLY NOT IMPLEMENTED HERE.
    #
    # Building the create-ATA transaction requires care that this inspection
    # did not verify (rent sizing, idempotency handling, and confirming the
    # derived address against the program's own validation). Shipping a
    # half-verified transaction builder behind a --confirm flag would be worse
    # than shipping nothing: an operator would reasonably assume it had been
    # tested. The command therefore reports, and tells the operator exactly
    # what is missing. Nothing is broadcast by this code path.
    if args.confirm:
        if (args.confirm or "").strip() != SETUP_WSOL_PHRASE:
            print("  REFUSING: confirmation phrase does not match exactly.")
            return 2
        print()
        print("  on-chain creation is NOT implemented in this build.")
        print("  This command is diagnostic only and broadcasts nothing.")
        print()
        return 3
    return asyncio.run(_work())


def cmd_preflight(cfg: Config, args: argparse.Namespace) -> int:
    """
    Run the real preflight. READ-ONLY, and it cannot arm anything.

    Every check it performs is a capability probe: the wallet loads, signing
    works offline against a throwaway instruction, the RPC answers, the quote
    provider is reachable, the sizing policy is verified as a pure function. No
    transaction is built, signed for the network, or broadcast, and the
    emergency stop is not touched.
    """
    configure_logging(cfg.log_dir, args.log_level)
    from .execution.real import RealExecutor

    missing = cfg.real.missing_capabilities()
    if missing:
        print("PREFLIGHT: cannot run - missing capabilities: "
              + ", ".join(missing), file=sys.stderr)
        print("  Set them in .env.local and re-run. This command never arms "
              "REAL.", file=sys.stderr)
        return 2

    ex = RealExecutor(cfg)

    async def _run():
        # One event loop for the probe AND the teardown. Closing an httpx client
        # from a different loop than the one that created it raises
        # "Event loop is closed", which would bury the preflight result in a
        # traceback and leave the operator unable to read the checks.
        try:
            return await ex.preflight()
        finally:
            await ex.aclose()

    result = asyncio.run(_run())

    print("REAL PREFLIGHT (read-only, nothing broadcast)")
    print("=" * 64)
    print(result.render())
    print("=" * 64)
    print(f"result: {'PASS' if result.ok else 'FAIL'}")
    print("NOTE: a passing preflight does NOT arm REAL. Arming requires the")
    print("      operator to press the REAL button in Telegram and confirm.")
    return 0 if result.ok else 1


def cmd_recent(cfg: Config, args: argparse.Namespace) -> int:
    """Print recently closed paper trades. Read-only; places no order."""
    from .engine import PaperTrader
    from .telegram.ui import render_recent_orders

    trader = PaperTrader(cfg)
    try:
        rows = trader.recent_orders(limit=args.limit)
    finally:
        trader._release_ownership()
    print(render_recent_orders(rows))
    return 0


def cmd_real_equity(cfg: Config, args: argparse.Namespace) -> int:
    """
    Print the measured REAL equity.

    Read-only: reads a balance and a price, signs nothing and broadcasts
    nothing. An unreadable balance or price is reported as UNAVAILABLE with its
    reason, never as $0.00.
    """
    import asyncio

    from .engine import PaperTrader

    trader = PaperTrader(cfg)
    try:
        if not trader.real_capable():
            print("REAL capabilities are not configured "
                  f"(missing: {trader.real_executor.missing or 'wallet/rpc'}).")
            return 1
        result = asyncio.run(trader.real_equity())
    finally:
        trader._release_ownership()

    if args.json:
        print(json.dumps(result.as_dict(), indent=2, default=str))
        return 0 if result.available else 1

    print("REAL EQUITY (measured)")
    print("=" * 62)
    print(f"  wallet          : {result.wallet_masked}")
    print(f"  native SOL      : {result.sol:.9f}")
    print(f"  SOL/USD         : "
          f"{result.sol_price_usd:,.4f}" if result.sol_price_usd
          else "  SOL/USD         : unavailable")
    if result.available:
        print(f"  equity          : ${result.equity_usd:,.4f}")
    else:
        print(f"  equity          : UNAVAILABLE")
        print(f"  reason          : {result.reason}")
        print()
        print("  REAL entry sizing is BLOCKED until equity is available.")
    print("=" * 62)
    return 0 if result.available else 1


def _cleanup_inputs(trader) -> tuple[list[str], list[str], list[str], dict]:
    """
    Every mint the runtime still needs, split by the rule it trips.

    Returns ``(open_mints, pending_mints, reserved_mints, detail)``.

    ``reserved_mints`` is the AMBIGUITY union required by the cleanup contract:
    a mint that any single layer still references -- an open REAL position, an
    open PAPER position, a non-terminal execution intent, or a position whose
    on-chain balance is zero while local state still calls it open (an
    unresolved external close) -- is refused as ``SKIPPED_AMBIGUOUS`` rather
    than closed because one layer (open positions) happened to be empty.

    Every read is individually guarded. A layer that raises contributes nothing
    rather than silently shrinking the reserved set, which is why the caller is
    told which layers answered and which did not.
    """
    detail: dict[str, object] = {}
    open_mints: list[str] = []
    pending_mints: list[str] = []
    reserved: set[str] = set()

    trader.load_real_state()
    detail["real_state_loaded"] = True

    try:
        real_open = [p.mint for p in trader.real_account.open_positions()
                     if getattr(p, "mint", None)]
        open_mints.extend(real_open)
        reserved.update(real_open)
        detail["real_open_positions"] = len(real_open)
    except Exception as exc:  # noqa: BLE001
        detail["real_open_positions"] = f"unreadable ({type(exc).__name__})"
        raise

    try:
        # PAPER open positions live in the in-memory account the engine loads
        # from disk; the on-disk snapshot is the fallback when it is empty.
        paper_open: list[str] = []
        try:
            paper_open = [str(p.mint) for p in trader.account.open_positions()
                          if getattr(p, "mint", None)]
        except Exception:  # noqa: BLE001
            paper_open = []
        if not paper_open:
            raw = (load_json(trader.config.state_file) or {}).get(
                "open_positions") or []
            paper_open = [str(p.get("mint")) for p in raw
                          if isinstance(p, dict) and p.get("mint")]
            # The engine persists on a heartbeat, so a snapshot read from a
            # SECOND process can lag a live in-memory position. Reporting the age
            # is the difference between "no paper position" and "the paper
            # position file I could see is this old".
            snapshot = (load_json(trader.config.state_file) or {})
            beat = (snapshot.get("account") or {}).get("last_heartbeat")
            if beat:
                detail["paper_state_snapshot_age_s"] = round(
                    max(0.0, trader.clock.now() - float(beat)), 1)
        reserved.update(paper_open)
        detail["paper_open_positions"] = len(paper_open)
    except Exception as exc:  # noqa: BLE001
        detail["paper_open_positions"] = f"unreadable ({type(exc).__name__})"

    try:
        unresolved = [str(i.mint) for i in trader.execution_intents.unresolved()
                      if getattr(i, "mint", None)]
        pending_mints.extend(unresolved)
        reserved.update(unresolved)
        detail["unresolved_intents"] = len(unresolved)
    except Exception as exc:  # noqa: BLE001
        detail["unresolved_intents"] = f"unreadable ({type(exc).__name__})"

    try:
        active = [str(i.mint) for i in trader.execution_intents.active()
                  if getattr(i, "mint", None)]
        reserved.update(active)
        detail["non_terminal_intents"] = len(active)
    except Exception as exc:  # noqa: BLE001
        detail["non_terminal_intents"] = f"unreadable ({type(exc).__name__})"

    # A position the runtime still calls open but which is NOT confirmed is an
    # unresolved exit: its token account may still be needed.
    try:
        unconfirmed = [p.mint for p in trader.real_account.positions.values()
                       if getattr(p, "is_open", False)
                       and not getattr(p, "confirmed", True)
                       and getattr(p, "mint", None)]
        reserved.update(unconfirmed)
        detail["unconfirmed_real_positions"] = len(unconfirmed)
    except Exception as exc:  # noqa: BLE001
        detail["unconfirmed_real_positions"] = (
            f"unreadable ({type(exc).__name__})")

    reserved.update(open_mints)
    reserved.update(pending_mints)
    return open_mints, pending_mints, sorted(reserved), detail


def cmd_wallet_cleanup(cfg: Config, args: argparse.Namespace) -> int:
    """
    Recover rent from EMPTY token accounts. DRY RUN by default.

    Read-only unless ``--execute`` is given WITH the exact confirmation token.
    The safety rules (owner, zero balance, close authority, no delegate, known
    token program, no open position, no pending transaction, no ambiguity, rent
    destination) are enforced by the planner; every candidate is read from the
    configured RPC first, so the plan is a statement about the CHAIN rather than
    about a cached row.
    """
    import asyncio

    from .engine import PaperTrader
    from .wallet_assets import program_label
    from .token_cleanup import (
        AWAITING_SIMULATION,
        CONFIRM_TOKEN,
        ELIGIBLE,
        audit_close_instructions,
        build_close_instructions,
        confirm_closures,
        execute_cleanup,
        plan_token_account_cleanup,
        probe_closability,
        read_on_chain_states,
        read_transaction_fee,
        reconcile_balance,
        resolve_broadcast_outcome,
    )

    trader = PaperTrader(cfg)
    try:
        if not trader.real_capable():
            print("REAL capabilities are not configured; cannot inspect the "
                  "wallet.", file=sys.stderr)
            return 2
        backend = trader.real_executor.backend
        open_mints, pending_mints, reserved_mints, runtime_detail = (
            _cleanup_inputs(trader))

        async def _run():
            """
            One event loop for the whole command.

            The RPC client binds its HTTP connection to the loop that first uses
            it, so a second ``asyncio.run`` in the same process would fail with a
            closed-loop error and every read after it would look like an
            unreadable chain. Discovery, the simulation proofs, the broadcast and
            the post-broadcast confirmation therefore share one loop.
            """
            listed = await backend.rpc.token_balances(backend.pubkey)
            balance_before = await reconcile_balance(
                backend.rpc, backend.pubkey)
            states = await read_on_chain_states(
                backend.rpc, [str(r.get("account") or "") for r in listed])
            accounts = []
            for row in listed:
                row = dict(row)
                row["chain_state"] = states.get(str(row.get("account") or ""))
                accounts.append(row)

            def _plan(closability=None):
                return plan_token_account_cleanup(
                    accounts, owner=backend.pubkey, open_mints=open_mints,
                    pending_mints=pending_mints,
                    reserved_mints=reserved_mints,
                    # The rent must return to the wallet that owns the accounts.
                    # Asserted, not assumed: a differing destination refuses the
                    # whole plan.
                    rent_destination=backend.pubkey,
                    closability=closability,
                    wallet_masked=backend.wallet.masked)

            # Pass one decides everything the account's own fields decide.
            # Accounts whose closability the fields cannot settle are HELD until
            # the node itself answers -- which pass two asks for by simulation.
            # Simulation is read-only; nothing is broadcast.
            pre = _plan()
            proofs: dict = {}
            if pre.simulation_required and (args.verify or args.execute):
                proofs = await probe_closability(
                    backend.rpc, backend.wallet, backend.pubkey,
                    pre.simulation_required)
            plan = _plan(proofs or None)

            if args.verify:
                # READ-ONLY forensic verification: every candidate carries its
                # on-chain state, its runtime verdict and the node's own
                # simulation verdict. Nothing is broadcast. This is the evidence
                # behind a decision, not a second cleanup path.
                verdicts = []
                for row in accounts:
                    address = str(row.get("account") or "")
                    hit = next((c for c in plan.safe_to_close
                                if c.token_account == address), None)
                    blocked = next((b for b in plan.blocked
                                    if b.token_account == address), None)
                    held = next((c for c in plan.simulation_required
                                 if c.token_account == address), None)
                    probe = proofs.get(address) or {}
                    verdicts.append({
                        "token_account": address,
                        "mint": str(row.get("mint") or ""),
                        "program_label": program_label(str(row.get("program"))),
                        "on_chain": row.get("chain_state"),
                        "eligibility": (ELIGIBLE if hit else
                                        blocked.status if blocked else
                                        AWAITING_SIMULATION if held
                                        else "UNCLASSIFIED"),
                        "reason": (hit.reason if hit else
                                   blocked.reason if blocked else
                                   "awaiting simulation proof"),
                        "lamports": (hit or blocked or held).lamports,
                        "chain_would_close": probe.get("closable"),
                        "chain_detail": probe.get("detail"),
                    })
                return {
                    "wallet": backend.wallet.masked,
                    "rpc_configured": bool(cfg.real.rpc_url),
                    "balance_before": balance_before,
                    "runtime_state": runtime_detail,
                    "plan": plan.as_dict(),
                    "chain_probes": verdicts,
                    "broadcast_attempted": False,
                    "eligible_count": len(plan.safe_to_close),
                    "blocked_count": len(plan.blocked),
                    "held_for_proof_count": len(plan.simulation_required),
                    "rendered": plan.render(),
                }, 0

            if not args.execute:
                if args.json:
                    return plan.as_dict(), 0
                print("WALLET CLEANUP (DRY RUN -- nothing broadcast)")
                print("=" * 64)
                print(plan.render())
                print("=" * 64)
                print("To actually close, re-run with --execute --confirm "
                      f"{CONFIRM_TOKEN}")
                return None, 0

            if (args.confirm or "") != CONFIRM_TOKEN:
                print("REFUSED: --execute requires --confirm "
                      f"{CONFIRM_TOKEN}", file=sys.stderr)
                return None, 3
            if not plan.has_safe_accounts:
                print("nothing safe to close; no transaction broadcast.")
                return None, 0

            # The instruction set is audited BEFORE anything is signed or sent,
            # so a cleanup transaction can only ever contain the planned closes.
            audit = audit_close_instructions(
                build_close_instructions(plan), plan)
            if audit:
                print("REFUSED: instruction audit failed:\n  "
                      + "\n  ".join(audit), file=sys.stderr)
                return None, 4

            result = await execute_cleanup(
                backend.rpc, backend.wallet, plan, confirm=args.confirm,
                simulate=True)
            result["balance_before"] = balance_before
            if not result.get("broadcast_attempted"):
                result["balance_after"] = balance_before
                result["note"] = ("nothing was broadcast; the balance is "
                                  "unchanged by definition")
                result["wallet"] = backend.wallet.masked
                return result, 0 if result.get("executed") else 1
            signature = result.get("signature") or ""
            if signature:
                # The confirmation pass is reported only when the chain agrees
                # with it. A rate-limited status poll makes that pass fail open
                # (an empty status list plus a slot comparison against a block
                # from a different node), which can report a finalized cleanup
                # as "expired" -- so the chain is asked directly what happened.
                outcome = await resolve_broadcast_outcome(
                    backend.rpc, signature,
                    first_verdict=str(result.get("status") or ""))
                result["authoritative_outcome"] = outcome
                if outcome["authoritative_status"] in ("confirmed", "finalized"):
                    result["executed"] = True
                    result["status"] = outcome["authoritative_status"]
                elif outcome["authoritative_status"] == "failed":
                    result["executed"] = False
            result["closures"] = await confirm_closures(
                backend.rpc, result.get("closed") or ())
            result["fee_lamports"] = (
                await read_transaction_fee(backend.rpc, signature)
                if signature else None)
            result["balance_after"] = await reconcile_balance(
                backend.rpc, backend.pubkey)
            result["wallet"] = backend.wallet.masked
            return result, 0 if result.get("executed") else 1

        payload, code = asyncio.run(_run())
    finally:
        trader._release_ownership()

    if payload is not None and (args.verify or args.json or args.execute):
        rendered = payload.pop("rendered", None) if isinstance(payload, dict) else None
        print(json.dumps(payload, indent=2, default=str))
        if rendered:
            print(rendered)
    return code


def cmd_wallet_scan(cfg: Config, args: argparse.Namespace) -> int:
    """
    Full, honest read of the configured wallet across BOTH token programs.

    READ-ONLY: it reads a native balance, token accounts and prices, and signs
    and broadcasts nothing. Every detected asset is reported, including ones
    that cannot be priced (``asset_detected=true, valuation=unavailable``).
    Token-2022 holdings are enumerated explicitly, which the old classic-only
    scan could not do.
    """
    import asyncio

    from .engine import PaperTrader
    from .wallet_assets import scan_wallet, summarize_scan

    trader = PaperTrader(cfg)
    try:
        if not trader.real_capable():
            print("REAL capabilities are not configured; cannot scan a wallet.",
                  file=sys.stderr)
            return 2
        backend = trader.real_executor.backend

        async def _run():
            sol_price = await trader._backend_sol_price()
            prices: dict = {}
            for row in await backend.rpc.token_balances(backend.pubkey) or ():
                mint = str(row.get("mint") or "")
                if not mint or int(row.get("amount") or 0) == 0:
                    continue
                px = await trader._real_asset_price_usd(
                    mint, str(row.get("symbol") or ""))
                if px is not None:
                    prices[mint] = px
            return await scan_wallet(
                backend.rpc, backend.pubkey, sol_price_usd=sol_price,
                price_of=lambda mint, symbol: prices.get(mint),
                now=trader.clock.now(), wallet_masked=backend.wallet.masked)

        scan = asyncio.run(_run())
    finally:
        trader._release_ownership()

    if args.json:
        print(json.dumps(scan.as_dict(), indent=2, default=str))
    else:
        summary = summarize_scan(scan)
        print("WALLET SCAN (read-only; nothing broadcast)")
        print("=" * 64)
        print(f"  wallet           : {scan.wallet_masked}")
        print(f"  native SOL       : "
              + (f"{scan.sol:.9f}" if scan.sol is not None else "UNAVAILABLE"))
        print(f"  equity           : "
              + (f"${scan.equity_usd:,.4f}" if scan.available else "UNAVAILABLE"))
        if not scan.available and scan.reason:
            print(f"  reason           : {scan.reason}")
        if scan.unpriced_mints:
            print(f"  unpriced assets  : {', '.join(scan.unpriced_mints)}")
            print("      (asset_detected=true, valuation=unavailable -- not counted "
                  "as zero)")
        print(f"  SPL balances     : {summary['spl_count']}")
        print(f"  Token-2022       : {summary['token_2022_count']}")
        print(f"  empty accounts   : {summary['empty_account_count']}")
        print(f"  rent recoverable : {scan.rent_recoverable_lamports:,} lamports "
              f"({scan.rent_recoverable_sol:.9f} SOL)")
        if scan.assets:
            print("-" * 64)
            for a in scan.assets:
                flag = " [Token-2022]" if a.is_token_2022 else ""
                if a.is_empty:
                    print(f"  {a.mint[:24]:<24} empty{flag}"
                          f"  rent={a.lamports or 0:,}")
                else:
                    val = (f"${a.value_usd:,.6f}" if a.value_usd is not None
                           else "valuation=unavailable")
                    print(f"  {a.mint[:24]:<24} {a.ui_amount:,.6f}{flag}  {val}")
        print("=" * 64)
    return 0 if scan.balance_readable else 1


def cmd_reconcile_external_close(cfg: Config, args: argparse.Namespace) -> int:
    """
    Record a REAL position as EXTERNALLY_CLOSED after verifying the chain.

    OPERATOR-INITIATED and FAIL CLOSED. It reads the chain; it never signs and
    never broadcasts. Every refusal reason is printed verbatim, because "the
    chain does not say what you asked it to say" must never look like success.

    The position is only closed when the wallet's on-chain balance is zero AND
    the supplied signature is confirmed with err==null AND that transaction
    debited exactly the booked token amount from a token account owned by the
    configured signer AND a destination was credited the same amount. Proceeds
    are the wallet's measured native SOL gain; zero proceeds is recorded as
    zero, and no exit price is invented.
    """
    import asyncio

    from .engine import PaperTrader
    from .external_close import (
        EXTERNAL_CLOSE_REASON,
        RpcChainFacts,
        verify_external_close,
    )

    mint = (args.mint or "").strip()
    signature = (args.signature or "").strip()
    if not mint or not signature:
        print("both --mint and --signature are required; refusing to guess "
              "either.", file=sys.stderr)
        return 2

    trader = PaperTrader(cfg)
    try:
        trader.load_real_state()
        pos = trader.real_account.positions.get(mint)
        if pos is None:
            print(f"no REAL position for mint {mint} in persisted state",
                  file=sys.stderr)
            return 2
        if pos.status != "OPEN":
            print(f"position is {pos.status}, not OPEN; nothing to reconcile",
                  file=sys.stderr)
            return 2
        if not trader.real_capable():
            print("REAL capabilities are not configured; cannot verify against "
                  "the chain.", file=sys.stderr)
            return 2

        backend = trader.real_executor.backend
        facts = RpcChainFacts(backend.rpc, backend.pubkey)

        async def _run():
            """
            Everything async happens in ONE event loop, on purpose.

            The RPC client is an httpx.AsyncClient bound to whichever loop
            first used it. Calling ``asyncio.run`` a second time builds a new
            loop, and the shared client's connections are already closed against
            the old one -- so the second call raises "Event loop is closed",
            which the price reader correctly but unhelpfully reports as an
            unreadable price. The result was a reconciliation that silently
            could not price USDC and withheld a PnL it had the data for.
            """
            verification = await verify_external_close(
                facts,
                wallet=backend.pubkey,
                mint=mint,
                signature=signature,
                expected_amount=int(pos.token_amount),
            )
            if not verification.ok:
                return verification, None, {}, []

            # PRICE EVERYTHING THE WALLET WAS CREDITED, measured. The $SI
            # reconciliation proved why this cannot be native-lamports-only: the
            # counterparty paid 0.170950 USDC into this wallet, and a SOL-only
            # accounting booked the position as a total loss when it had in fact
            # been paid. An asset whose price cannot be read leaves the PnL
            # UNSTATED rather than guessing a peg.
            sol_price = await trader._backend_sol_price()
            prices: dict = {}
            unpriced: list[str] = []
            total = 0.0
            for c in verification.credited_assets:
                px = (sol_price if c.mint == "SOL"
                      else await trader._real_asset_price_usd(c.mint, c.mint[:8]))
                if px and px > 0:
                    prices[c.mint] = float(px)
                    total += c.ui_amount * float(px)
                else:
                    unpriced.append(c.mint)
            return verification, (None if unpriced else total), prices, unpriced

        verification, proceeds_usd, prices, unpriced = asyncio.run(_run())
        if not verification.ok:
            print("REFUSED. Not recording the position as externally closed.",
                  file=sys.stderr)
            print(f"  reason: {verification.reason}", file=sys.stderr)
            if args.json:
                print(json.dumps(verification.as_dict(), indent=2, default=str))
            return 1

        # through the ACCOUNT, not the position object, so realized_pnl_usd and
        # closed_count stay consistent with the row they came from
        _closed, applied = trader.real_account.close_externally(
            mint, verification, proceeds_usd=proceeds_usd,
            asset_prices_usd=prices)
        # the normal state writer, so persistence stays one code path
        trader.save_real_state()

        # the flag that was blocking REAL is derived from the chain at startup;
        # clearing the stale OPEN row is what clears it.
        trader._real_state_consistent = True
    finally:
        trader._release_ownership()

    print("EXTERNALLY_CLOSED (verified against the chain)")
    print("=" * 68)
    print(f"  mint             : {mint}")
    print(f"  signature        : {signature}")
    print(f"  debited (raw)    : {verification.debited_amount:,}")
    print(f"  settlement       : {applied['settlement']}")
    print()
    print("  ASSETS CREDITED TO THIS WALLET (measured from the chain):")
    if applied["credited_assets"]:
        for c in applied["credited_assets"]:
            px = c.get("price_usd")
            val = ("" if not px
                   else f"  = ${c['ui_amount'] * px:,.6f}")
            print(f"      {c['ui_amount']:>18,.6f} {c['mint'][:20]}"
                  f"  @ ${px:,.6f}{val}" if px
                  else f"      {c['ui_amount']:>18,.6f} {c['mint'][:20]}"
                       f"  (price UNREADABLE)")
    else:
        print("      (none) nothing was credited to this wallet")
    print()
    print(f"  entry cost (USD) : ${applied['entry_cost_usd']:,.6f}")
    if applied["pnl_stated"]:
        print(f"  proceeds (USD)   : ${applied['proceeds_usd']:,.6f}")
        pnl = applied["realized_pnl_usd"]
        print(f"  realized PnL     : ${pnl:+,.6f}  "
              f"({'GAIN' if pnl > 0 else 'LOSS' if pnl < 0 else 'FLAT'})")
    else:
        print("  proceeds (USD)   : NOT STATED")
        print("  realized PnL     : NOT STATED")
        print(f"      unreadable price for: {', '.join(unpriced)}")
        print("      A PnL is not invented from an unobserved price, and a")
        print("      zero here would also be a guess: value WAS received, in")
        print("      an asset whose price could not be read.")
    print(f"  exit reason      : {EXTERNAL_CLOSE_REASON}")
    print("=" * 68)
    if not applied["credited_assets"]:
        print("  No asset reached this wallet: this was a transfer away, not a")
        print("  sale, so the cost basis is recorded as a total loss.")
    print("State written through the normal REAL state path.")
    print("Restart the runtime so recovery_consistent is recomputed at startup.")
    return 0


def cmd_real_intent_status(cfg: Config, args: argparse.Namespace) -> int:
    """
    Show every REAL execution intent and, for the unresolved ones, the evidence.

    READ ONLY. An unresolved intent is the one state that blocks all REAL
    trading, so the operator needs to see WHICH intent and WHY it is stuck
    before deciding anything. This prints the recorded state, whether a
    signature exists, and -- for a stuck entry -- whether the chain shows the
    position was actually created.
    """
    import asyncio

    from .engine import PaperTrader
    from .execution.intents import UNRESOLVED_STATES

    trader = PaperTrader(cfg)
    try:
        rows = []
        for store_intent in trader.execution_intents.all_intents():
            rows.append(store_intent)
        if args.json:
            print(json.dumps(
                [getattr(r, "__dict__", str(r)) for r in rows],
                indent=2, default=str))
            return 0

        print("REAL EXECUTION INTENTS")
        print("=" * 78)
        if not rows:
            print("  (none recorded)")
        for r in rows:
            state = getattr(r, "state", "?")
            stuck = state in UNRESOLVED_STATES
            print(f"  [{'STUCK' if stuck else 'ok   '}] {getattr(r,'side','?'):<5} "
                  f"{str(getattr(r,'mint',''))[:14]:<15} {state:<10} "
                  f"sig={(getattr(r,'signature','') or '(NONE)')[:20]}")
            if getattr(r, "detail", ""):
                print(f"          detail: {r.detail}")
            if stuck:
                print("          ^^^ THIS ONE BLOCKS EVERY REAL ENTRY")
        print("=" * 78)
        return 0
    finally:
        trader._release_ownership()


def cmd_reconcile_unresolved_intent(cfg: Config, args: argparse.Namespace) -> int:
    """
    Clear a stuck REAL intent, but ONLY after the chain says nothing happened.

    WHY THIS EXISTS. A broadcast that the node refused during its own preflight
    simulation (``skipPreflight: False``, RPC ``-32002``) was recorded as
    UNKNOWN, because "the RPC said no" is not the same claim as "the network
    never saw it". That is right for a LOST RESPONSE and wrong for a structured
    refusal: the node simulates, watches the transaction fail, and does not
    submit it. One such intent on 8HJY9 blocked twenty REAL entries over three
    hours while the mint's token account did not exist on chain at all.

    THE REFUSALS ARE THE POINT. This command will not clear an intent that:

      * has a signature -- a signature means the node DID submit it;
      * is not in a non-terminal state -- a confirmed or failed intent is not
        stuck and rewriting history would corrupt the audit trail;
      * was refused for a reason that is not provably non-submission, i.e. a
        transport failure or an unrecognised code, where submission is possible;
      * shows on chain that the position WAS created -- if the tokens are there,
        the trade happened and this is a RECOVERY, not a cleanup.

    A missing token account is treated as proof of non-execution only together
    with the absent signature and the recorded node refusal. Any one of those
    alone is not enough, and the command says which condition failed.
    """
    import asyncio

    from .engine import PaperTrader
    from .execution.intents import UNRESOLVED_STATES

    key = (args.logical_key or "").strip()
    if not key:
        print("--logical-key is required", file=sys.stderr)
        return 2

    trader = PaperTrader(cfg)
    try:
        intent = trader.execution_intents.find(key)
        if intent is None:
            print(f"no intent with logical_key {key}", file=sys.stderr)
            return 2

        state = getattr(intent, "state", "")
        sig = getattr(intent, "signature", "") or ""
        detail = getattr(intent, "detail", "") or ""
        side = getattr(intent, "side", "")
        mint = getattr(intent, "mint", "")

        # -- the refusals, in order, each with its own reason ---------------
        if sig:
            print("REFUSED: this intent HAS a signature, so the node submitted "
                  "it.\n  This is a RECOVERY problem, not a cleanup. Do not "
                  "clear it here.", file=sys.stderr)
            print(f"  signature: {sig}", file=sys.stderr)
            return 1
        if state not in UNRESOLVED_STATES:
            print(f"REFUSED: intent is {state}, which is not unresolved; "
                  f"nothing to clear.", file=sys.stderr)
            return 1
        if "BROADCAST_REJECTED" not in detail:
            print("REFUSED: the recorded failure is not a broadcast rejection, "
                  "so non-submission is not provable from the log.",
                  file=sys.stderr)
            print(f"  detail: {detail}", file=sys.stderr)
            return 1
        if "RPC error" not in detail:
            print("REFUSED: the rejection has no RPC code, so it is a "
                  "transport failure where submission remains possible.",
                  file=sys.stderr)
            print(f"  detail: {detail}", file=sys.stderr)
            return 1

        # -- the chain gets the last word -----------------------------------
        backend = trader.real_executor.backend

        async def _check():
            import re as _re
            from .execution.solana import (
                RpcError, send_failure_may_have_submitted)

            # Pull the code back out of the recorded detail. A naive
            # whitespace split yields "-32002:" -- colon included -- which is not
            # an int, so the code came back None and the command correctly
            # refused as "unknown". That refusal was the fail-closed design
            # catching its own bug, which is the behaviour worth keeping; the
            # parse is what needed fixing.
            m = _re.search(r"RPC error (-?\d+)", detail)
            code = int(m.group(1)) if m else None
            still_unknown = send_failure_may_have_submitted(
                RpcError(detail, code=code))
            balance = None
            if mint:
                balance = await backend.rpc.get_token_account_balance(
                    backend.pubkey, mint)
            return still_unknown, balance

        still_unknown, balance = asyncio.run(_check())

        if still_unknown:
            print("REFUSED: this rejection is one where the transaction MAY "
                  "have been submitted\n  (transport failure or an "
                  "unrecognised code). Clearing it could double the position.",
                  file=sys.stderr)
            return 1

        if side == "ENTRY" and balance:
            print(f"REFUSED: the chain shows {balance} of {mint} in this "
                  f"wallet,\n  so the entry DID execute. This is a RECOVERY, "
                  f"not a stuck intent.", file=sys.stderr)
            return 1

        trader.execution_intents.transition(
            intent.execution_id, "FAILED",
            detail=("operator cleared after chain verification: node refused "
                    "at preflight with no signature"
                    + (f"; {mint} token account does not exist"
                       if side == "ENTRY" and not balance else "")),
        )
        trader.save_real_state()
    finally:
        trader._release_ownership()

    print("CLEARED (verified against the chain)")
    print("=" * 78)
    print(f"  logical_key   : {key}")
    print(f"  side          : {side}")
    print(f"  signature     : (none) the node never submitted it")
    print(f"  recorded cause: {detail}")
    if side == "ENTRY":
        print(f"  chain check   : {mint} balance = "
              f"{balance if balance is not None else 'no token account'}"
              f"  -> the entry never executed")
    print("  new state     : FAILED (terminal)")
    print("=" * 78)
    print("REAL entries are unblocked. Restart the runtime to pick up the "
          "change if it is already running.")
    return 0


def cmd_telegram_status(cfg: Config, args: argparse.Namespace) -> int:
    """
    Report whether the Telegram control channel is usable, and nothing more.

    Booleans, counters and the runtime's own state. The token and the
    allow-listed chat id are deliberately omitted: knowing which chat is
    authorised would help an attacker target it, and neither is needed to
    operate the bot. The configured chat appears only as a boolean.
    """
    from .engine import PaperTrader

    trader = PaperTrader(cfg)
    try:
        status = trader.bot_status()
    finally:
        trader._release_ownership()

    if args.json:
        print(json.dumps(status, indent=2, default=str))
        return 0

    print("TELEGRAM TRANSPORT")
    print("=" * 62)
    for key in ("transport_initialized", "control_ready", "notifier_attached",
                "token_configured", "chat_configured",
                "forward_validation_active", "run_state", "trade_mode",
                "real_trading_enabled"):
        print(f"  {key:<28}: {status.get(key)}")
    control = status.get("control") or {}
    if control:
        print(f"  {'polling':<28}: {control.get('polling')}")
        print("-" * 62)
        print(f"  updates seen              : {control.get('updates_seen')}")
        print(f"  authorized                : {control.get('updates_authorized')}")
        print(f"  refused (chat)            : "
              f"{control.get('updates_refused_chat')}")
        print(f"  refused (replay)          : "
              f"{control.get('updates_refused_replay')}")
        print(f"  refused (malformed)       : "
              f"{control.get('updates_malformed')}")
        print(f"  last error                : "
              f"{control.get('last_error') or '(none)'}")
    notifier = status.get("notifier") or {}
    if notifier:
        print("-" * 62)
        print(f"  notifications sent        : {notifier.get('sent')}")
        print(f"  notifications failed      : {notifier.get('failed')}")
        print(f"  dropped (queue full)      : "
              f"{notifier.get('dropped_queue_full')}")
    return 0


def cmd_forward_report(cfg: Config, args: argparse.Namespace) -> int:
    """
    Print the forward PAPER validation report.

    Reads only the prospective store. This command cannot validate L25 by
    itself: it reports the status the pre-declared policy allows, and says so
    when that status is anything other than CONSISTENT.
    """
    from .forward_validation import (
        ForwardRecorder, build_report, load_or_create_deployment,
    )
    from .l25 import ACTIVE_STRATEGY, L25_MIN_LIQ_TO_MCAP

    root = cfg.var_dir / "forward_validation"
    dep = load_or_create_deployment(
        root, strategy=ACTIVE_STRATEGY,
        l25_min_liq_to_mcap=cfg.decision.l25_min_liq_to_mcap,
        config_source=cfg.config_source,
    )
    records = ForwardRecorder(root, dep).all_records()
    report = build_report(records, deployment=dep)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, default=str))
    else:
        print(report.render())
    return 0


def cmd_config_source(cfg: Config, args: argparse.Namespace) -> int:
    """
    Safe configuration diagnostic.

    Reports WHERE settings came from and the non-secret safety booleans. It
    never prints a value from .env.local, a credential, a path, or a URL - only
    key NAMES and the source that supplied each one. This is the command to run
    when someone asks "what config is this thing actually using?".
    """
    payload = {
        "config_source": cfg.config_source,
        "env_file": ENV_FILENAME,
        "env_file_present": (cfg.root / ENV_FILENAME).exists(),
        "precedence": [ENV_FILENAME, "environment", "default"],
        "origin_by_key": dict(cfg.env_origin),
        "safety": {
            "real_trading_enabled": cfg.real.trading_enabled,
            "env_real_request_ignored": cfg.real.env_trading_enabled_request,
            "reentry_enabled": cfg.risk.reentry_enabled,
            "max_open_positions": cfg.risk.max_simultaneous_positions,
            "wallet_configured": cfg.real.wallet_configured,
            "rpc_configured": cfg.real.rpc_configured,
        },
        "strategy": {
            "active": STRATEGY,
            "l25_min_liq_to_mcap": cfg.decision.l25_min_liq_to_mcap,
            "v3c_max_market_cap_usd": cfg.decision.v3c_max_market_cap_usd,
            "v3c_min_signal_quality": cfg.decision.v3c_min_signal_quality,
            "v2a_ceiling_usd": cfg.decision.v2a_ceiling_usd,
        },
    }
    if args.json:
        # JSON ONLY. A machine-readable flag that has to be parsed out of
        # decorated prose is not machine-readable.
        print(json.dumps(payload, indent=2))
        return 0

    print("CONFIG SOURCE")
    print("=" * 62)
    print(f"  source            : {cfg.config_source}")
    print(f"  env file          : {ENV_FILENAME} "
          f"({'present' if (cfg.root / ENV_FILENAME).exists() else 'absent'})")
    print(f"  precedence        : {ENV_FILENAME} > environment > built-in default")
    print()
    print("  per-key origin (NAMES only, never values):")
    by_origin: dict[str, list[str]] = {}
    for key, origin in sorted(cfg.env_origin.items()):
        by_origin.setdefault(origin, []).append(key)
    for origin in (ENV_FILENAME, "environment", "default"):
        keys = by_origin.get(origin, [])
        if keys:
            shown = ", ".join(keys[:12]) + (f" ... (+{len(keys)-12})" if len(keys) > 12 else "")
            print(f"    from {origin:<14}: {shown}")
    print()
    print("  SAFETY (non-secret booleans):")
    print(f"    REAL trading enabled   : {cfg.real.trading_enabled}")
    print(f"    env REAL request       : {cfg.real.env_trading_enabled_request} "
          "(reported only; authorizes nothing)")
    print(f"    risk acknowledged     : {cfg.real.risk_acknowledged}")
    print(f"    re-entry enabled      : {cfg.risk.reentry_enabled}")
    print(f"    max open positions    : {cfg.risk.max_simultaneous_positions}")
    print(f"    wallet configured     : {cfg.real.wallet_configured}")
    print(f"    rpc configured        : {cfg.real.rpc_configured}")
    print(f"    missing capabilities  : "
          f"{', '.join(cfg.real.missing_capabilities()) or 'none'}")
    print(f"    active strategy       : {STRATEGY}")
    # The selected PROFILE, distinct from the strategy identity above: every
    # profile trades L25, and the profile says which exit policy variant. Not a
    # secret, so the value is shown.
    _cap = _sp_max_hold_s(_sp_parse(cfg.decision.strategy_profile))
    print(f"    strategy profile      : {cfg.decision.strategy_profile} "
          f"({_sp_label(_sp_parse(cfg.decision.strategy_profile))})")
    print(f"    profile max hold      : "
          f"{'none' if _cap is None else f'{_cap:g}s'}")
    print(f"    L25 ratio threshold   : {cfg.decision.l25_min_liq_to_mcap:.4f}")
    print(f"    V3-C market cap <=    : {cfg.decision.v3c_max_market_cap_usd:,.0f}")
    print(f"    V3-C signal quality   : {cfg.decision.v3c_min_signal_quality:.2f}")
    print(f"    V2-A ceiling          : {cfg.decision.v2a_ceiling_usd:,.2f}")
    print("=" * 62)
    return 0


def cmd_audit_deps(cfg: Config, args: argparse.Namespace) -> int:
    """
    Dependency + import audit.

    Proves the runtime imports no ML, Laya or shadow library, that none is
    installed-and-loaded, and lists what the real executor actually needs so an
    operator can tell required from optional.
    """
    configure_logging(None, args.log_level)
    import sys as _sys

    from .safety import (
        FORBIDDEN_MODULES,
        MODEL_CALLS,
        EXTERNAL_MODEL_IMPORTS,
        SHADOW_CAPITAL,
        SHADOW_ENTRIES,
        audit_loaded_modules,
        loaded_forbidden_modules,
    )

    print("DEPENDENCY + IMPORT AUDIT")
    print("=" * 64)
    print("forbidden modules (must never be importable):")
    for m in sorted(FORBIDDEN_MODULES):
        print(f"  - {m}")
    print()
    loaded = loaded_forbidden_modules()
    print(f"forbidden modules currently LOADED: {loaded or 'none'}")
    violations = audit_loaded_modules()
    print(f"ML/shadow symbols in autopaper.*  : {violations or 'none'}")
    print(f"EXTERNAL_MODEL_IMPORTS            : {EXTERNAL_MODEL_IMPORTS}")
    print(f"MODEL_CALLS                       : {MODEL_CALLS}")
    print(f"SHADOW_ENTRIES                    : {SHADOW_ENTRIES}")
    print(f"SHADOW_CAPITAL                    : {SHADOW_CAPITAL}")
    print()
    print("modules loaded under autopaper.*:")
    for name in sorted(n for n in _sys.modules if n.startswith("autopaper")):
        print(f"  {name}")
    print()
    report = audit()
    print(report.render())
    return 0 if (report.ok and not loaded and not violations) else 1


def cmd_stop(cfg: Config, args: argparse.Namespace) -> int:
    payload = {"command": CMD_STOP, "at": __import__("time").time(), "pid": None}
    existing = load_json(cfg.control_file) or {}
    existing.update(payload)
    save_json_atomic(cfg.control_file, existing)
    print(
        "STOP issued.\n"
        "  - no new paper positions will be opened\n"
        "  - existing positions continue to be managed until their exit logic fires\n"
        "  - use `flatten` if you want them closed now\n"
        "  - positions are NOT silently closed by a stop"
    )
    return 0


def cmd_flatten(cfg: Config, args: argparse.Namespace) -> int:
    payload = {"command": CMD_FLATTEN, "at": __import__("time").time()}
    existing = load_json(cfg.control_file) or {}
    existing.update(payload)
    save_json_atomic(cfg.control_file, existing)
    print("FLATTEN issued: the running engine will close open paper positions now.")
    return 0


def cmd_resume(cfg: Config, args: argparse.Namespace) -> int:
    save_json_atomic(cfg.control_file, {"command": "", "at": 0.0, "acked_at": None})
    print("control cleared: the engine will open new positions again.")
    return 0


def cmd_unlatch(cfg: Config, args: argparse.Namespace) -> int:
    """
    Clear the emergency-stop latch from the terminal.

    The latch is designed to survive restarts, and the bot's command set is
    five commands with no /clear_emergency. /resume clears it, but if the bot is
    not running -- which is the usual reason you are at a terminal -- the latch
    has to be clearable from here. Otherwise the only recovery is deleting a
    file by hand.

    Refuses without --yes, because this re-opens entry authorization.
    """
    path = cfg.var_dir / "emergency_stop.json"
    latch = load_json(path) or {}
    if not bool(latch.get("emergency_stop")):
        print("no emergency-stop latch is set; nothing to clear.")
        return 0
    if not bool(getattr(args, "yes", False)):
        print("EMERGENCY STOP is latched. Clearing it re-opens entry "
              "authorization.")
        print(f"  latch: {path}")
        print("  re-run with --yes to clear it.")
        return 1
    save_json_atomic(path, {"emergency_stop": False, "at": time.time()})
    save_json_atomic(cfg.control_file, {"command": "", "at": 0.0,
                                        "acked_at": None})
    print("emergency-stop latch cleared. The engine will open new positions "
          "again on its next start.")
    return 0


def cmd_config(cfg: Config, args: argparse.Namespace) -> int:
    print(describe(cfg))
    print(json.dumps(cfg.as_dict(), indent=2, default=str))
    return 0


def cmd_verify(cfg: Config, args: argparse.Namespace) -> int:
    """The migration + safety audit."""
    report = audit()
    print(report.render())
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    return 0 if report.ok else 1


def cmd_reconcile(cfg: Config, args: argparse.Namespace) -> int:
    """Read-only PAPER account/ledger reconciliation gate."""
    from .engine import _trade_from
    from .paper_accounting import reconcile

    ledger = _ledger(cfg)
    state = load_json(cfg.state_file) or {}
    account = dict(state.get("account") or {})
    trades = [
        trade for trade in (_trade_from(row) for row in ledger.trades())
        if trade is not None
    ]
    historical_report = reconcile(
        trades,
        starting_equity=float(
            account.get("starting_bankroll_usd", cfg.risk.starting_bankroll_usd)
            or cfg.risk.starting_bankroll_usd
        ),
        account_book=account,
    )
    boundary = dict(state.get("accounting_boundary") or {})
    historical_unreconciled = bool(
        boundary.get("historical_unreconciled")
        or historical_report.drift_status not in ("OK", "NO_TRADES")
    )
    # A pre-existing ledger/account mismatch is never repaired by rewriting
    # either file.  It is quarantined from the active production scope.  The
    # active account starts with no trusted historical settlements, so its
    # canonical reconciliation is against zero active trades and the persisted
    # account book.  Historical evidence remains fully reported below.
    active_trades = [] if historical_unreconciled else trades
    report = reconcile(
        active_trades,
        starting_equity=float(
            account.get("starting_bankroll_usd", cfg.risk.starting_bankroll_usd)
            or cfg.risk.starting_bankroll_usd
        ),
        account_book=account,
    )
    result = report.as_dict()
    result["open_positions"] = len(state.get("open_positions") or [])
    result["ledger_trade_count"] = len(trades)
    result["strategy_profiles"] = sorted({
        str(getattr(trade, "strategy_profile", ""))
        for trade in trades
        if getattr(trade, "strategy_profile", None)
    })
    result["scope"] = "ACTIVE_PRODUCTION"
    result["historical_unreconciled"] = historical_unreconciled
    result["historical_report"] = historical_report.as_dict()
    result["historical_trade_count"] = len(trades) if historical_unreconciled else 0
    result["active_trade_count"] = len(active_trades)
    print(json.dumps(result, indent=2, default=str))
    invariant_failure = any(
        "INVARIANT VIOLATION" in note for note in report.notes
    )
    return 0 if report.drift_status in ("OK", "NO_TRADES") and not invariant_failure else 1


def cmd_report(cfg: Config, args: argparse.Namespace) -> int:
    from .stats import attribution, compute, format_summary

    ledger = _ledger(cfg)
    from .engine import _trade_from

    trades = [t for t in (_trade_from(r) for r in ledger.trades()) if t is not None]
    memory = ResearchMemory(trades)
    state = load_json(cfg.state_file) or {}
    acct = state.get("account", {})
    s = compute(
        trades,
        starting_equity=float(acct.get("starting_bankroll_usd", 100.0) or 100.0),
    )
    print(f"{AGENT_ID} - PAPER REPORT")
    print("=" * 64)
    if not trades:
        print("no closed paper trades yet")
    for line in format_summary(s):
        print(" ", line)
    for label, table in (
        ("by route", s.by_route),
        ("by regime", s.by_regime),
        ("by exit style", s.by_exit_style),
        ("by age bucket", s.by_age_bucket),
        ("by mcap bucket", s.by_mc_bucket),
        ("by liquidity bucket", s.by_liq_bucket),
    ):
        if table:
            print(f"\n  {label}:")
            for k, v in table.items():
                print(
                    f"    {k:<22} n={v['n']:<4} wr={v['win_rate'] * 100:5.1f}% "
                    f"pnl={v['net_pnl_usd']:+8.2f} avg_ret={v['avg_return_pct']:+7.1f}%"
                )
    print("\n  funnel:", attribution(ledger.all_decisions()))
    print("\n  paper memory lessons (as of now):")
    for line in memory.lessons(10 ** 12) or ["  (no completed trades to learn from)"]:
        print("   ", line)
    print("=" * 64)
    return 0


def cmd_login(cfg: Config, args: argparse.Namespace) -> int:
    from .telegram import interactive_login

    if not cfg.telegram.has_credentials:
        print("TELEGRAM_API_ID / TELEGRAM_API_HASH missing from .env.local", file=sys.stderr)
        return 2
    print("One-time Telegram user login. This is a READ-ONLY account session used")
    print("to receive alerts; it has no wallet and no trading capability.")
    value = asyncio.run(
        interactive_login(cfg.telegram.api_id, cfg.telegram.api_hash, cfg.telegram.session or "")
    )
    print()
    print("Add this line to .env.local (do not paste it into chat):")
    print()
    print(f"TELEGRAM_SESSION={value}")
    print()
    print("Then leave the running engine alone - it will pick the session up on the")
    print("next retry without a restart.")
    return 0


def cmd_same_market_test(cfg: Config, args: argparse.Namespace) -> int:
    """
    Measure whether a signal's marked market is the market that would trade.

    THIS COMMAND CANNOT TRADE. It is a measurement harness with no capability
    to sign, submit or broadcast anything: the Solana client it builds exposes
    read methods only, the Jupiter client builds quotes and not transactions,
    and an import barrier refuses every module in this repository that
    contains a signer or a broadcaster. The handler asserts that property
    before it constructs a single client.

    It also cannot modify behaviour. It does not touch L25, the strategy
    thresholds, the risk settings, or any wallet state, and it never reaches
    the trading engine - so a result from this command cannot become a
    strategy change by accident.

    Exit codes: 0 on a completed measurement (a REJECT verdict is a completed
    measurement), 2 when the harness refuses to run or cannot measure.
    """
    from .same_market.cli import run_same_market_test

    return run_same_market_test(cfg, args)


def cmd_paper_live(cfg: Config, args: argparse.Namespace) -> int:
    """
    Execution-aware PAPER forward capture.

    THIS COMMAND CANNOT TRADE. It has no signer, no transaction builder and no
    broadcaster, and the handler asserts that structurally before it constructs
    a single client. Every fill it writes is labelled
    SIMULATED_FILL_FROM_EXECUTABLE_QUOTE; the ACTUAL_FILL column is always
    null.

    What it does change relative to the historical PAPER model is WHERE THE
    PRICE COMES FROM. Entry and exit are priced from fresh executable quotes
    read off chain, not from DexScreener observed marks, so the realized
    result can no longer be a mark-to-mark ratio. It does not modify L25, the
    thresholds, the exits, the risk settings or any wallet state.

    Exit codes: 0 on a completed run, 2 when the harness refuses to start.
    """
    from .paper_live.cli import run

    return run(cfg, args)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autopaper",
        description="MEMEBOT V3 - V3-C Solana memecoin bot, PAPER or REAL. "
                    "REAL is OFF by default and cannot be armed from here.",
    )
    p.add_argument("--root", default=None, help="project root (default: package root)")
    p.add_argument("--log-level", default=None,
                   help="DEBUG | INFO | WARNING | ERROR")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="start or resume the autonomous paper loop")
    r.add_argument(
        "--cli",
        action="store_true",
        help="CLI-only mode. Trading and the upstream Telegram CHANNEL listener "
             "run normally, but every OUTBOUND Telegram bot call is disabled so "
             "nothing the bot does can delay a trade.",
    )
    r.add_argument("--reset-control", action="store_true",
                   help="clear any pending STOP/FLATTEN before starting")
    r.add_argument("--adopt-legacy-account", action="store_true",
                   help="adopt a state file last written by a pre-lock process "
                        "(one-time, after confirming that process has stopped)")
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("status", help="one-shot status")
    s.set_defaults(func=cmd_status)
    st = sub.add_parser("stop", help="STOP: stop opening new paper positions")
    st.set_defaults(func=cmd_stop)
    fl = sub.add_parser("flatten", help="FLATTEN: close open paper positions now")
    fl.set_defaults(func=cmd_flatten)
    rs = sub.add_parser("resume", help="clear a pending STOP/FLATTEN")
    rs.set_defaults(func=cmd_resume)
    ul = sub.add_parser(
        "unlatch",
        help="clear the emergency-stop latch (survives restarts; needs --yes)")
    ul.add_argument("--yes", action="store_true",
                    help="actually clear the latch")
    ul.set_defaults(func=cmd_unlatch)
    c = sub.add_parser("config", help="show effective configuration")
    c.set_defaults(func=cmd_config)
    v = sub.add_parser("verify", help="migration + safety audit")
    v.add_argument("--json", action="store_true")
    v.set_defaults(func=cmd_verify)
    pf = sub.add_parser(
        "preflight",
        help="read-only REAL preflight (does NOT arm REAL, broadcasts nothing)",
    )
    pf.set_defaults(func=cmd_preflight)
    sw = sub.add_parser(
        "setup-wsol",
        help="report the wallet's wrapped-SOL account (read-only by default; "
             "broadcasts nothing)",
    )
    sw.add_argument(
        "--confirm",
        default=None,
        help=f"required phrase to prepare on chain; must be exactly "
             f"{SETUP_WSOL_PHRASE!r}. Not wired into run.",
    )
    sw.set_defaults(func=cmd_setup_wsol)
    ad = sub.add_parser("audit-deps", help="dependency + import audit")
    ad.set_defaults(func=cmd_audit_deps)
    cs = sub.add_parser(
        "config-source",
        help="safe config diagnostic: which source supplied each setting "
             "(names only, never values)",
    )
    cs.add_argument("--json", action="store_true")
    cs.set_defaults(func=cmd_config_source)
    rep = sub.add_parser("report", help="full paper performance report")
    rep.set_defaults(func=cmd_report)
    rec = sub.add_parser(
        "reconcile",
        help="read-only PAPER account/ledger reconciliation (fails on drift)",
    )
    rec.set_defaults(func=cmd_reconcile)
    ro = sub.add_parser(
        "recent",
        help="recently CLOSED paper trades, newest first (read-only)",
    )
    ro.add_argument("--limit", type=int, default=10)
    ro.set_defaults(func=cmd_recent)
    re_ = sub.add_parser(
        "real-equity",
        help="measured REAL equity from the configured wallet and RPC "
             "(read-only; reports UNAVAILABLE rather than $0.00 on failure)",
    )
    re_.add_argument("--json", action="store_true")
    re_.set_defaults(func=cmd_real_equity)

    ws = sub.add_parser(
        "wallet-scan",
        help="full read of the configured wallet across SPL and Token-2022 "
             "(read-only; reports unpriced assets as detected-but-unvalued, "
             "never as zero)",
    )
    ws.add_argument("--json", action="store_true")
    ws.set_defaults(func=cmd_wallet_scan)

    wc = sub.add_parser(
        "wallet-cleanup",
        help="recover rent from EMPTY token accounts. DRY RUN by default; "
             "closing requires --execute with an explicit confirmation token",
    )
    wc.add_argument("--dry-run", action="store_true", default=True,
                    help="report what would close; broadcast nothing (default)")
    wc.add_argument("--verify", action="store_true",
                    help="read-only forensic verification: read every candidate "
                         "from the chain and ask the node, by simulation, "
                         "whether the close would succeed. Broadcasts nothing.")
    wc.add_argument("--execute", action="store_true",
                    help="actually close the safe accounts (requires --confirm)")
    wc.add_argument("--confirm", default=None,
                    help="must equal the cleanup confirmation token")
    wc.add_argument("--json", action="store_true")
    wc.set_defaults(func=cmd_wallet_cleanup)

    ec = sub.add_parser(
        "reconcile-external-close",
        help="record a REAL position as EXTERNALLY_CLOSED after verifying on "
             "chain that the wallet no longer holds it (read-only; never signs, "
             "never broadcasts; fail-closed on any doubt)",
    )
    ec.add_argument("--mint", required=True,
                    help="the mint of the position that left the wallet")
    ec.add_argument("--signature", required=True,
                    help="the external transaction that moved it out")
    ec.add_argument("--json", action="store_true")
    ec.set_defaults(func=cmd_reconcile_external_close)

    ris = sub.add_parser(
        "real-intent-status",
        help="list every REAL execution intent and mark the ones that are "
             "blocking trading (read-only)",
    )
    ris.add_argument("--json", action="store_true")
    ris.set_defaults(func=cmd_real_intent_status)

    rui = sub.add_parser(
        "reconcile-unresolved-intent",
        help="clear a stuck REAL intent, but only after verifying on chain that "
             "nothing was submitted; refuses when a signature exists or the "
             "position was actually created (read-only, never broadcasts)",
    )
    rui.add_argument("--logical-key", required=True,
                     help="e.g. ENTRY:<mint> or EXIT:<mint>")
    rui.add_argument("--json", action="store_true")
    rui.set_defaults(func=cmd_reconcile_unresolved_intent)
    ts = sub.add_parser(
        "telegram-status",
        help="sanitised Telegram transport diagnostics: booleans and counters "
             "only, never the token or the allow-listed chat",
    )
    ts.add_argument("--json", action="store_true")
    ts.set_defaults(func=cmd_telegram_status)
    fv = sub.add_parser(
        "forward-report",
        help="forward PAPER validation status: prospective data only, with an "
             "explicit verdict about whether L25 is validated (it is not, until "
             "the pre-declared sample size is reached)",
    )
    fv.add_argument("--json", action="store_true")
    fv.set_defaults(func=cmd_forward_report)
    lg = sub.add_parser("login", help="one-time Telegram session creation")
    lg.set_defaults(func=cmd_login)

    smt = sub.add_parser(
        "same-market-test",
        help="READ-ONLY test harness. For each new signal from a channel, "
             "reports whether the DexScreener-marked market is provably the "
             "same market an execution would use, and how far a read-only BUY "
             "quote for the configured test size sits from the observed mark. "
             "BROADCASTS NOTHING, SIGNS NOTHING, BUYS NOTHING",
    )
    smt.add_argument(
        "--channel",
        default="MooDengPresidentCallers",
        help="the one Telegram channel to read (default: %(default)s)",
    )
    smt.add_argument(
        "--order-usd",
        type=float,
        default=0.23,
        help="hypothetical BUY size in USD equivalent (default: %(default)s). "
             "No money is spent; no transaction is submitted",
    )
    smt.add_argument(
        "--sol-usd-override",
        dest="sol_usd_override",
        type=float,
        default=None,
        help="OFFLINE/TEST-ONLY deterministic SOL/USD reference, used to size "
             "--order-usd in lamports. Supply it to reproduce a run with no "
             "network call. Omit it in normal use: the reference is fetched "
             "once from the free keyless Jupiter price API and recorded in the "
             "report. If it cannot be obtained the harness REFUSES rather "
             "than guessing",
    )
    smt.add_argument(
        "--sol-usd",
        type=float,
        default=None,
        help="DEPRECATED: alias for --sol-usd-override, kept so existing "
             "commands keep working. No longer required in normal use",
    )
    smt.add_argument(
        "--max-mark-gap-pct",
        type=float,
        default=None,
        help="absolute percentage gap between the executable quote and the "
             "observed mark that counts as PASS. If omitted, results are "
             "reported as INFORMATIVE and NO pass/fail verdict is claimed",
    )
    smt.add_argument("--max-mark-age-ms", type=float, default=None,
                     help="freshness window for the observed mark; beyond it "
                          "the signal is REJECTed as MARK_STALE")
    smt.add_argument("--max-state-age-ms", type=float, default=None,
                     help="freshness window for pool/curve state")
    smt.add_argument("--max-quote-age-ms", type=float, default=None,
                     help="freshness window for the executable quote")
    smt.add_argument("--quote-mint", default="So11111111111111111111111111111111111111112",
                     help="input asset for the hypothetical BUY (default: wSOL)")
    smt.add_argument("--slippage-bps", type=int, default=1000,
                     help="slippage sent to the quote provider (default: %(default)s). "
                          "A quote is not sent anywhere, so this only affects "
                          "the returned minimum-output field")
    # The RPC default is written as a literal rather than imported from
    # ``same_market.rpc``. Importing it here would install that package's
    # no-broadcast import barrier into EVERY CLI invocation, including
    # ``run``, which legitimately needs the execution modules. The harness's
    # barrier must apply to the harness alone.
    smt.add_argument("--rpc-url", default="https://api.mainnet-beta.solana.com",
                     help="read-only Solana RPC (default: the free public node)")
    smt.add_argument("--rpc-timeout-s", type=float, default=8.0)
    smt.add_argument("--rpc-max-rpm", type=int, default=20,
                     help="client-side cap on public RPC requests per minute")
    smt.add_argument("--provider-timeout-s", type=float, default=8.0)
    smt.add_argument("--cache-ttl-s", type=float, default=1.0)
    smt.add_argument("--poll-interval-s", type=float, default=1.5)
    smt.add_argument("--allow-backlog", action="store_true",
                     help="also process messages already in the channel. OFF by "
                          "default: the harness is meant to measure NEW signals")
    smt.add_argument("--message", default=None,
                     help="measure this message text instead of listening. "
                          "Use '-' to read stdin. No Telegram connection is made")
    smt.add_argument(
        "--divergence-pct",
        type=float,
        default=None,
        help="tolerance used to label a pairwise price gap CLOSE or "
             "DIVERGENT. If omitted - the default - every gap is reported with "
             "its raw percentage and classified UNKNOWN, because inventing a "
             "band to manufacture a verdict is the behaviour under test",
    )
    smt.add_argument(
        "--no-direct",
        action="store_true",
        help="skip the direct on-chain Pump.fun / PumpSwap quote stage. Only "
             "then is there no DIRECT_EXECUTABLE_QUOTE column, and it is "
             "reported as absent rather than substituted",
    )
    smt.add_argument("--no-rpc", action="store_true",
                     help="skip all Solana reads; quotes then come only from "
                          "the venue and chain state is UNKNOWN")
    smt.add_argument("--no-pump", action="store_true",
                     help="skip the Pump.fun bonding-curve and PumpSwap paths, "
                          "including the direct on-chain quote stage")
    smt.add_argument("--no-jupiter", action="store_true",
                     help="skip Jupiter quoting entirely")
    smt.add_argument("--json", action="store_true",
                     help="emit the machine-readable JSON report to stdout "
                          "instead of the rendered text")
    smt.set_defaults(func=cmd_same_market_test)

    # ---------------------------------------------------------------------
    # paper-live: the execution-aware PAPER forward-capture harness.
    #
    # The parser is built by the harness module so its argument surface lives
    # next to the code that consumes it. The heavy imports (the RPC client, the
    # market service, the Telegram listener) stay inside the handler for the
    # same reason they do for every other command: ``autopaper --help`` must
    # not drag in a signer-adjacent import graph.
    # ---------------------------------------------------------------------
    pliv = sub.add_parser(
        "paper-live",
        help="PAPER ONLY - NO TRANSACTION EXECUTION. For each NEW signal from a "
             "channel, runs the existing L25 decision unchanged, takes a fresh "
             "EXECUTABLE quote, and simulates the fill from that quote rather "
             "than from the DexScreener observed mark. Writes a "
             "forward-validation dataset. Buys nothing, signs nothing, sends "
             "nothing",
    )
    from .paper_live.cli import build_parser_args as _paper_live_args

    _paper_live_args(pliv)
    pliv.set_defaults(func=cmd_paper_live)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = build_config(args.root)
    return int(args.func(cfg, args))


if __name__ == "__main__":
    raise SystemExit(main())
