"""
``autopaper paper-live`` - the execution-aware PAPER forward-capture command.

READ-ONLY WITH RESPECT TO MONEY. It listens to a channel, runs the production
L25 decision, takes fresh executable quotes, simulates fills and writes a
dataset. It has no signer, no transaction builder and no broadcaster; the
safety audit runs before anything else and refuses to start if that is not true.

THE SAME COMMAND IN EVERY INVOCATION STYLE
==========================================

* listener (the normal case): waits for NEW signals and processes each one
* ``--message`` : measures one message inline, with no Telegram connection
* ``--max-signals N`` : bounds the run, which is what makes a smoke test
  possible on a machine with no session

NO MARK-VERSUS-QUOTE THRESHOLD
===============================

Deliberately absent as a blocking option for this run. The point is to collect
the empirical distribution of mark-to-executable gaps, so a threshold would
destroy the thing being measured. Every gap is recorded; none is filtered on.
"""

from __future__ import annotations

from .arms import arm_names

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from ..clock import iso
from ..config import DecisionConfig, ExitConfig
from .arms import parse_arms, selectable_arm_names
from .engine import PaperLiveEngine, build_signal_snapshot
from .fills import FillModel
from .guards import audit_paper_capability
from .quotes import QuoteService, SolReference
from .quote_diagnostics import breakdown_of, cost_block
from .report import NO_EXECUTION_NOTE, DatasetWriter
from .schema import FailureReason

BANNER = "PAPER ONLY — NO TRANSACTION EXECUTION"


def build_parser_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The argument surface, kept next to the code that consumes it."""
    parser.add_argument(
        "--channel", default="MooDengPresidentCallers",
        help="the one Telegram channel to read (default: %(default)s)",
    )
    parser.add_argument(
        "--bankroll-usd", type=float, default=5.0,
        help="STARTING virtual bankroll per arm (default: %(default)s). Each arm "
             "has its own and they cannot affect each other. Position size is "
             "the PRODUCTION formula applied to this number: 10%% at or below "
             "$50, 12.5%% above",
    )
    parser.add_argument(
        # THE DEFAULT NAMES THE REAL ARMS. This used to read
        # "L25_CURRENT,L25_60S,H2_CAP60", and ``L25_60S`` is a PROFILE, not an
        # arm -- it was renamed to ``L25_MAX_HOLD`` in the profile vocabulary
        # and never existed in ``ResearchArm``. So the default could not parse:
        # ``autopaper paper-live`` with no ``--strategies`` refused to start at
        # all, which is a bug that hides itself by looking like a config problem.
        #
        # ``selectable_arm_names()`` and NOT ``arm_names()``: the V2 FLOOR
        # (BETA) experiment is an EXPLICIT research experiment with its own
        # session, ledgers and reset, so it is opt-in by name. Deriving the
        # default from the group token (or from ``parse_arms([])``) would start
        # three extra ledgers on every invocation that never asked for them.
        "--strategies",
        default=",".join(selectable_arm_names()),
        help="comma-separated research arms (default: %(default)s); to run the "
             "V2 FLOOR (BETA) three-arm exit experiment pass V2_FLOOR_BETA",
    )
    parser.add_argument(
        "--sol-usd-override",
        dest="sol_usd_override",
        type=float,
        default=None,
        help="OFFLINE/TEST-ONLY deterministic SOL/USD reference. Supply this to "
             "reproduce a run exactly with no network call. Omit it in normal "
             "use: the reference is fetched once from the free keyless Jupiter "
             "price API, and the run is REFUSED if it cannot be obtained",
    )
    parser.add_argument(
        "--sol-usd", type=float, default=None,
        help="DEPRECATED: alias for --sol-usd-override, kept so existing "
             "commands keep working. No longer required in normal use",
    )
    parser.add_argument(
        "--max-quote-age-ms", type=float, default=30_000.0,
        help="a simulated fill is refused if its quote is older than this "
             "(default: %(default)s). An old quote is NEVER replaced with the "
             "DexScreener mark",
    )
    parser.add_argument(
        "--rpc-url", default="https://api.mainnet-beta.solana.com",
        help="read-only Solana RPC (default: the free public node)",
    )
    parser.add_argument(
        "--rpc-max-rpm", type=int, default=15,
        help="client-side cap on public RPC requests per minute",
    )
    parser.add_argument(
        "--provider-timeout-s", type=float, default=8.0,
    )
    parser.add_argument(
        "--cache-ttl-s", type=float, default=1.0,
        help="quote/observation cache TTL. Set > 0 and two arms asking at the "
             "same moment share one request rather than hammering the provider",
    )
    parser.add_argument(
        "--monitor-interval-s", type=float, default=5.0,
        help="how often open positions are re-marked and re-assessed for exit",
    )
    parser.add_argument(
        "--post-quote-penalty-bps", type=float, default=0.0,
        help="OPTIONAL extra penalty applied AFTER a quote. OFF by default: a "
             "quote already prices this exact size, so a default penalty would "
             "double-count slippage",
    )
    parser.add_argument(
        "--allow-identity-unknown", action="store_true",
        help="permit a simulated trade when same-market identity cannot be "
             "PROVEN. OFF by default: the trade fails closed and the "
             "observation is retained as a non-trade record",
    )
    parser.add_argument(
        "--message", default=None,
        help="measure this message text instead of listening. No Telegram "
             "connection is made",
    )
    parser.add_argument(
        "--max-signals", type=int, default=0,
        help="stop after N signals (0 = unlimited). Makes a bounded smoke run "
             "possible",
    )
    parser.add_argument(
        "--poll-interval-s", type=float, default=1.5,
    )
    parser.add_argument(
        "--allow-backlog", action="store_true",
        help="also process messages already in the channel. OFF by default: "
             "this harness measures NEW signals",
    )
    parser.add_argument("--json", action="store_true",
                        help="print the JSON dataset to stdout at the end")
    parser.add_argument("--root", default=None, help="project root for reports")
    parser.add_argument("--log-level", default=None)
    return parser


def _fmt_usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:,.8f}"


def _fmt_sol(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.12f}"


def render_entry(record: Any) -> str:
    """The real-time console block for one accepted simulated trade."""
    q = record.buy_quote
    lat = record.latency.as_dict()
    lines = [
        "",
        "-" * 60,
        "PAPER ENTRY",
        f"Strategy: {record.arm}",
        f"Token: {record.signal.symbol or '?'}",
        f"Mint: {record.signal.token_mint}",
        "",
        f"Bankroll before: ${record.bankroll_before:.4f}",
        f"Position size:   ${record.position_size_usd:.4f}"
        f"  ({record.position_size_pct:.1f}%)",
        "",
        f"Observed mark:   {_fmt_usd(record.observed_mark.value_usd)}",
        f"Executable BUY:  {_fmt_usd(q.effective_price_usd)}",
    ]
    # The same cost block the Telegram bridge renders: reference price, raw
    # legs, fee, size impact, total quote cost and the premium over the mark.
    # One implementation, so the console and the alert cannot drift apart.
    lines += [f"  {line}" for line in cost_block(
        breakdown_of(q, observed_mark_usd=record.observed_mark.value_usd),
    )]
    lines += [
        "",
        f"Provider: {q.provider}",
        f"Venue:    {q.venue}",
        f"Pool:     {q.pool_or_curve}",
        f"Route:    {' → '.join(q.route) or 'n/a'}",
        f"Identity: {q.identity}",
        "",
        f"Input:  {q.input_amount_raw} raw ({q.input_decimals} dec) "
        f"= {q.input_amount_decimal} SOL",
        f"Output: {q.output_amount_raw} raw ({q.output_decimals} dec) "
        f"= {q.output_amount_decimal} tokens",
        # NAMED FOR WHAT IT IS. For a direct Pump quote this field is
        # fee + size impact, not the pure impact Jupiter means by the same
        # words -- see QuoteCostBreakdown for the split.
        f"Provider price_impact_pct: "
        + (f"{q.price_impact_pct:+.4f}%" if q.price_impact_pct is not None
           else "n/a"),
        f"Quote age: {_fmt_ms(q.quote_age_ms)}",
        f"Signal → entry: {_fmt_ms(lat.get('signal_to_simulated_entry_ms'))}",
        "",
        "SIMULATED FILL  (SIMULATED_FILL_FROM_EXECUTABLE_QUOTE)",
        "## NO TRANSACTION SENT",
    ]
    return "\n".join(lines)


def _fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}ms"


def render_exit(trade: Any) -> str:
    """The real-time console block for one settled simulated trade."""
    pnl = trade.pnl
    lines = [
        "",
        "-" * 60,
        "PAPER EXIT",
        f"Strategy: {trade.position.arm}",
        f"Token:   {trade.position.symbol or '?'}  {trade.position.mint}",
        "",
        f"Observed exit mark: {_fmt_usd(trade.exit_observed_mark)}",
        f"Executable SELL:    {_fmt_usd(trade.sell_quote.effective_price_usd)}",
    ]
    lines += [f"  {line}" for line in cost_block(
        breakdown_of(trade.sell_quote, observed_mark_usd=trade.exit_observed_mark),
    )]
    lines += [""]
    if pnl is None:
        lines += [
            "REALIZED PAPER PNL",
            f"  NOT SETTLED - {trade.failure.value}",
            f"  {trade.failure_detail}",
            "",
            "  The position is LEFT OPEN. It is not closed at the mark and not",
            "  force-closed at cost; a transient quote outage must not become a",
            "  realised loss or a phantom gain.",
        ]
    else:
        lines += [
            "REALIZED PAPER PNL",
            f"  Entry cost (quote asset): {_fmt_sol(float(pnl.entry_cost_quote))}",
            f"  Exit proceeds:           {_fmt_sol(float(pnl.exit_proceeds_quote))}",
            f"  Fees:                    {_fmt_sol(float(pnl.trading_fees_quote))}",
            f"  Net SOL:                 {_fmt_sol(pnl.realized_pnl_sol)}",
            f"  Net USD:                 {_fmt_usd(pnl.realized_pnl_usd)}",
            f"  Return:                  "
            + ("n/a" if pnl.return_pct is None else f"{pnl.return_pct:+.4f}%"),
            f"  Bankroll:                ${pnl.bankroll_before:.4f} → ${pnl.bankroll_after:.4f}",
            f"  Hold:                    {pnl.hold_seconds:.1f}s",
            "",
            "  Basis: simulated settlement between two SIMULATED_FILLs.",
            "  Never exit_mark / entry_mark - 1.",
        ]
    lines += [
        f"Exit reason: {trade.exit_trigger}",
        "",
        "## NO TRANSACTION SENT",
    ]
    return "\n".join(lines)


def render_reject(record: Any) -> str:
    """One line per non-trade. Rejected opportunities are data, not noise."""
    return (
        f"  [no trade] {record.arm:<12} {record.failure.value:<34} "
        f"{record.failure_detail[:90]}"
    )


def _build_engine(cfg: Any, args: argparse.Namespace) -> tuple[PaperLiveEngine, Any, Any]:
    """
    Construct the engine and its read-only clients.

    The RPC client is the harness's own five-read-method client, which refuses
    any other method name at call time. Nothing here can broadcast, and the
    audit has already run.
    """
    from ..market.service import MarketDataService  # imported late: heavy
    from ..same_market.jupiter import JupiterQuoteClient
    from ..same_market.rpc import SolanaReadOnlyRpc
    from ..analysis.decision import DecisionEngine
    from ..exits import ExitPolicy
    from ..reader import reader_for  # type: ignore[attr-defined]

    return None, None, None  # pragma: no cover - replaced below


async def _resolve_sol_reference(args: argparse.Namespace) -> Any:
    """
    Obtain the SOL/USD reference for this run, or refuse.

    Resolution order, and nothing else:

    1. ``--sol-usd-override`` (or the deprecated ``--sol-usd``) -- the
       offline/test-only deterministic input;
    2. one fetch from the free keyless Jupiter price API, cached for the run.

    There is no third option. ``SolPriceUnavailable`` propagates to ``run()``,
    which refuses, because a guessed SOL price would silently mis-size every
    notional in every arm.

    This value is ONLY a USD->SOL conversion reference. It never touches token
    pricing, quoting, fills or PnL.
    """
    from ..same_market.sol_price import JupiterSolPriceProvider, resolve_sol_usd_async

    override = getattr(args, "sol_usd_override", None)
    deprecated = getattr(args, "sol_usd", None)
    if override is not None or deprecated is not None:
        # An override makes no network call, so no client is ever built.
        return await resolve_sol_usd_async(override=override,
                                           deprecated=deprecated)

    provider = JupiterSolPriceProvider(
        timeout_s=float(getattr(args, "provider_timeout_s", 8.0)),
        ttl_s=float(getattr(args, "cache_ttl_s", 60.0)),
        max_rpm=int(getattr(args, "sol_price_max_rpm", 20)),
    )
    try:
        return await resolve_sol_usd_async(provider=provider)
    finally:
        await provider.aclose()


def run(cfg: Any, args: argparse.Namespace) -> int:
    """Entry point for ``autopaper paper-live``."""
    audit = audit_paper_capability()
    if not audit.ok:
        print("REFUSED: the PAPER harness has an execution capability; this is a "
              "bug, not a configuration problem.", file=sys.stderr)
        for note in audit.notes:
            print(f"  - {note}", file=sys.stderr)
        return 2

    print(BANNER)
    print("=" * 72)
    print(audit.render())
    print()

    try:
        arms = parse_arms([s for s in str(args.strategies).split(",")])
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    # The SOL/USD reference is a USD->SOL conversion input for sizing only. It
    # is obtained before any client is built, and its absence refuses the run
    # rather than falling back to a guessed price.
    from ..same_market.sol_price import SolPriceUnavailable, describe_snapshot

    try:
        sol_snapshot = asyncio.run(_resolve_sol_reference(args))
    except SolPriceUnavailable as exc:
        print("REFUSED: no SOL/USD reference could be obtained. Position sizing "
              "is specified in USD while execution happens in SOL, so this "
              "number governs every notional and it will not be guessed.",
              file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        print("  For an offline deterministic run pass --sol-usd-override "
              "<price>.", file=sys.stderr)
        return 2
    sol_reference = SolReference(
        sol_snapshot.usd_per_sol, sol_snapshot.timestamp, sol_snapshot.source,
    )

    from ..analysis.decision import DecisionEngine
    from ..exits import ExitPolicy
    from ..market import MarketDataService
    from ..same_market.jupiter import JupiterQuoteClient
    from ..same_market.rpc import SolanaReadOnlyRpc
    from ..same_market.reader import DirectPumpReader

    rpc = SolanaReadOnlyRpc(
        args.rpc_url, timeout_s=float(args.rpc_timeout_s if hasattr(args, "rpc_timeout_s") else 8.0),
        max_rpm=int(args.rpc_max_rpm), cache_ttl_s=float(args.cache_ttl_s),
    )
    jupiter = JupiterQuoteClient(
        timeout_s=float(args.provider_timeout_s),
        cache_ttl_s=float(args.cache_ttl_s),
    )
    reader = DirectPumpReader(rpc)

    async def decimals_for(mint: str) -> int | None:
        return await reader.mint_decimals(mint)

    quote_service = QuoteService(
        rpc=rpc,
        sol_reference=sol_reference,
        jupiter=jupiter,
        max_quote_age_ms=float(args.max_quote_age_ms),
        allow_jupiter_fallback=True,
        require_identity=not args.allow_identity_unknown,
    )

    # The PRODUCTION market observation service, constructed from the
    # production MarketConfig. It is the observation layer only: nothing it
    # returns is ever used as a fill.
    from ..config import MarketConfig

    market = MarketDataService(MarketConfig(
        http_timeout_s=float(args.provider_timeout_s),
        cache_ttl_s=float(args.cache_ttl_s),
    ))

    engine = PaperLiveEngine(
        arms=arms,
        decision_engine=DecisionEngine(DecisionConfig(), ExitConfig()),
        exit_policy=ExitPolicy(ExitConfig()),
        quote_service=quote_service,
        market=market,
        token_decimals_for=decimals_for,
        sol_reference=sol_reference,
        fill_model=FillModel(post_quote_penalty_bps=float(args.post_quote_penalty_bps)),
        starting_bankroll_usd=float(args.bankroll_usd),
        require_identity=not args.allow_identity_unknown,
    )

    print(f"BANKROLL          : ${float(args.bankroll_usd):.4f} per arm (virtual)")
    print(f"ARMS              : {', '.join(s.name.value for s in arms)}")
    print(f"SOL/USD reference : {describe_snapshot(sol_snapshot)}")
    print(f"SOL/USD metadata  : sol_usd={sol_snapshot.usd_per_sol!r} "
          f"source={sol_snapshot.source!r} "
          f"timestamp={sol_snapshot.timestamp!r} "
          f"age_ms={sol_snapshot.age_ms():.0f}")
    print(f"MAX QUOTE AGE     : {float(args.max_quote_age_ms):.0f}ms "
          f"(an older quote is refused, never replaced with the mark)")
    print(f"MARK GAP GUARD    : NOT CONFIGURED - gaps are collected, not filtered")
    print(f"CHANNEL           : @{str(args.channel).lstrip('@')}")
    print("-" * 72)

    return asyncio.run(
        _drive(engine, args, arms)
    )


async def _drive(engine: PaperLiveEngine, args: argparse.Namespace, arms) -> int:
    """Listen or measure one message, then write the dataset."""
    processed = 0
    stop = asyncio.Event()

    async def on_record(records) -> None:
        for record in records:
            if record.traded:
                print(render_entry(record))
            else:
                print(render_reject(record))

    async def on_trade(trade) -> None:
        print(render_exit(trade))

    if args.message:
        from ..same_market.signals import extract_token

        text = str(args.message)
        candidate = extract_token(text)
        signal = build_signal_snapshot(
            channel=str(args.channel), text=text, message_id=None,
            telegram_timestamp=None, mint=candidate.mint,
            symbol=candidate.symbol, provenance=(
                {"mint": "extract_token"} if candidate.mint else {}
            ),
            warnings=candidate.warnings, ambiguous=candidate.ambiguous,
            candidates=candidate.candidates,
        )
        await on_record(await engine.on_signal(signal))
        # Let the position age through one monitor pass so the exit path runs.
        await asyncio.sleep(0.05)
        for trade in await engine.monitor_once():
            await on_trade(trade)
    else:
        from ..config import TelegramConfig
        from ..same_market.listener import listen

        telegram = TelegramConfig(
            api_id=None, api_hash=None, session=None, chat=args.channel,
            allow_backlog=bool(args.allow_backlog),
            poll_interval_s=float(args.poll_interval_s),
        )
        cfg = getattr(args, "_cfg", None)
        if cfg is not None:
            telegram = TelegramConfig(
                api_id=cfg.telegram.api_id, api_hash=cfg.telegram.api_hash,
                session=cfg.telegram.session, chat=args.channel,
                allow_backlog=bool(args.allow_backlog),
                poll_interval_s=float(args.poll_interval_s),
            )
        if not telegram.has_credentials or not telegram.has_session:
            print("REFUSED: no authenticated Telegram session available.", file=sys.stderr)
            print("  The harness reuses this repository's existing session and "
                  "will not log in itself. Use --message to measure a message "
                  "without Telegram.", file=sys.stderr)
            return 2

        print(f"Waiting for NEW messages from @{str(args.channel).lstrip('@')}. "
              f"Ctrl+C to stop.")

        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(__import__("signal").SIGINT, stop.set)
        except (NotImplementedError, RuntimeError):
            pass

        async def on_alert(snapshot, parsed) -> None:
            nonlocal processed
            try:
                records = await engine.on_signal(
                    build_signal_snapshot(
                        channel=str(args.channel),
                        text=str(getattr(snapshot, "raw_text", "")),
                        message_id=getattr(snapshot, "message_id", None),
                        telegram_timestamp=getattr(snapshot, "source_timestamp", None),
                        mint=getattr(snapshot, "mint", None),
                        symbol=getattr(snapshot, "symbol", None),
                        provenance={"mint": "telegram_alert_snapshot"},
                    )
                )
                await on_record(records)
                processed += 1
                for trade in await engine.monitor_once():
                    await on_trade(trade)
            except Exception as exc:  # noqa: BLE001 - one signal must not kill the run
                print(f"  [signal failed, continuing] {type(exc).__name__}: {exc}"[:200])
            if args.max_signals and processed >= args.max_signals:
                stop.set()

        async def monitor_loop() -> None:
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=float(args.monitor_interval_s))
                except asyncio.TimeoutError:
                    pass
                try:
                    for trade in await engine.monitor_once():
                        await on_trade(trade)
                except Exception as exc:  # noqa: BLE001 - keep monitoring
                    print(f"  [monitor pass failed] {type(exc).__name__}: {exc}"[:200])

        monitor = asyncio.ensure_future(monitor_loop())
        try:
            await listen(telegram, on_alert, stop=stop)
        finally:
            stop.set()
            monitor.cancel()

    writer = DatasetWriter(Path(args.root) if args.root else None, arms=arms)
    json_path = writer.write_json(
        signals=engine.signal_records, trades=engine.trade_records,
        status=engine.status(),
    )
    signals_csv = writer.write_signals_csv(engine.signal_records)
    trades_csv = writer.write_trades_csv(engine.trade_records)
    md_path = writer.write_markdown(
        signals=engine.signal_records, trades=engine.trade_records,
        status=engine.status(),
    )
    print("")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {signals_csv}")
    print(f"wrote {trades_csv}")
    if args.json:
        print(json.dumps(engine.status(), indent=2, default=str))
    print("")
    print(NO_EXECUTION_NOTE)
    return 0


__all__ = ["BANNER", "build_parser_args", "render_entry", "render_exit", "run"]
