"""
CLI wiring for the same-market quote harness.

Separated from :mod:`autopaper.cli` because the heavy imports are deferred and
the command needs a lot of argument surface; keeping it in its own module means
the main CLI file gains a handler and a parser entry and nothing else.

Read-only. Constructs no engine, acquires no account lock, arms nothing, and
broadcasts nothing. The one thing it does assert before it starts is that it
has no execution capability - see :func:`build_pipeline`.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from ..clock import iso
from .dexscreener import DexScreenerProbe
from .guard import audit_no_execution_capability
from .jupiter import JupiterQuoteClient
from .models import StalenessPolicy
from .pipeline import SameMarketPipeline, SignalReport, TestConfig
from .report import render_report, render_summary
from .rpc import DEFAULT_RPC_URL, SolanaReadOnlyRpc

#: Where machine-readable and human-readable output land. Under ``reports/``
#: because that is where this repository keeps its research artifacts, and
#: because ``src/autopaper`` never writes into ``reports/`` on its own.
REPORT_DIR = Path("reports") / "tests" / "same_market_quote"

#: Marks this a test harness in every rendered output.
BANNER = "SAME-MARKET QUOTE TEST  (READ-ONLY / NO EXECUTION)"


def _reports_root(root: Path | None) -> Path:
    return (root or Path.cwd()) / REPORT_DIR


def write_report_files(
    reports: list[SignalReport],
    *,
    root: Path | None = None,
    sol_usd: float | None = None,
    sol_metadata: dict[str, Any] | None = None,
) -> tuple[list[Path], Path]:
    """
    Write one JSON file per signal, plus a rolling ``latest.md``.

    Credentials are not reachable from here: the serialised payload is built
    from the report dataclasses, which contain no environment values, no
    session data and no key material. Telegram session bytes are never part of
    a :class:`SignalReport` at all - the listener hands over text and timing,
    nothing else.
    """
    base = _reports_root(root)
    base.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    stamp = iso(time.time()) or "unknown"
    file_stamp = stamp.replace(":", "").replace("-", "").replace(".", "")

    # Additive only: the SOL/USD snapshot metadata is MERGED into the existing
    # payload. No existing key is renamed, removed or re-valued, so every
    # downstream reader of these files keeps working unchanged.
    sol_meta = sol_metadata or {}

    for report in reports:
        mint = (report.token.mint or "no-token")[:44]
        payload = report.as_dict()
        for key, value in sol_meta.items():
            payload.setdefault(key, value)
        path = base / f"{file_stamp}_{mint}.json"
        path.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        written.append(path)

    latest = base / "latest.md"
    parts = [render_goat_section(sol_usd)]
    for report in reports:
        parts.append(render_report(report))
    parts.append(render_summary(reports))
    latest.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return written, latest


def render_goat_section(sol_usd: float | None = None) -> str:
    from .goat import render_goat_regression

    return render_goat_regression(sol_usd)


def build_pipeline(args: Any) -> tuple[SameMarketPipeline, TestConfig, SolanaReadOnlyRpc | None]:
    """
    Construct the pipeline from CLI arguments.

    The no-execution audit runs here, before any client exists, so a misconfigured
    process refuses rather than starting a measurement it should not be taking.
    """
    audit = audit_no_execution_capability()
    if not audit.ok:
        print("REFUSED: the same-market harness has an execution capability; "
              "this is a bug, not a configuration problem.", file=sys.stderr)
        for note in audit.notes:
            print(f"  - {note}", file=sys.stderr)
        raise SystemExit(2)

    config = TestConfig(
        order_usd=float(args.order_usd),
        sol_reference_usd=config_sol(args),
        quote_asset_mint=args.quote_mint,
        max_mark_gap_pct=(float(args.max_mark_gap_pct)
                          if args.max_mark_gap_pct is not None else None),
        staleness=StalenessPolicy(
            max_mark_age_ms=args.max_mark_age_ms,
            max_state_age_ms=args.max_state_age_ms,
            max_quote_age_ms=args.max_quote_age_ms,
        ),
        slippage_bps=int(args.slippage_bps),
        prefer_pump=not args.no_pump,
        divergence_pct=(float(args.divergence_pct)
                        if getattr(args, "divergence_pct", None) is not None
                        else None),
        direct_quotes=not getattr(args, "no_direct", False),
    )

    dexscreener = DexScreenerProbe(
        timeout_s=float(args.provider_timeout_s),
        cache_ttl_s=float(args.cache_ttl_s),
    )

    rpc: SolanaReadOnlyRpc | None = None
    if not args.no_rpc:
        rpc = SolanaReadOnlyRpc(
            args.rpc_url,
            timeout_s=float(args.rpc_timeout_s),
            max_rpm=int(args.rpc_max_rpm),
            cache_ttl_s=float(args.cache_ttl_s),
        )

    jupiter = None
    if not args.no_jupiter:
        jupiter = JupiterQuoteClient(
            timeout_s=float(args.provider_timeout_s),
            slippage_bps=int(args.slippage_bps),
            cache_ttl_s=float(args.cache_ttl_s),
        )

    pipeline = SameMarketPipeline(
        config=config, dexscreener=dexscreener, rpc=rpc, jupiter=jupiter,
    )
    return pipeline, config, rpc


async def _run_listener(args: Any, pipeline: SameMarketPipeline) -> int:
    """
    Wait for NEW signals from the configured channel and measure each one.

    Runs until Ctrl+C. Uses the repository's existing authenticated Telegram
    session; never prints it, and never prints the API hash or the chat id.
    """
    from dataclasses import replace

    from ..config import TelegramConfig
    from .listener import listen

    audit = audit_no_execution_capability()

    telegram = TelegramConfig(
        api_id=None, api_hash=None, session=None, chat=args.channel,
        allow_backlog=bool(args.allow_backlog),
        poll_interval_s=float(args.poll_interval_s),
    )
    # Credentials come from the operator's own config, are attached to the
    # listener object and are never rendered. ``cfg`` is not consulted for
    # anything else: this command must not touch trading configuration.
    cfg = getattr(args, "_cfg", None)
    if cfg is not None:
        telegram = replace(
            telegram,
            api_id=cfg.telegram.api_id,
            api_hash=cfg.telegram.api_hash,
            session=cfg.telegram.session,
        )

    if not telegram.has_credentials or not telegram.has_session:
        print("REFUSED: no authenticated Telegram session available.", file=sys.stderr)
        print("  The harness reuses the repository's existing session and will "
              "not log in itself.", file=sys.stderr)
        return 2

    print(BANNER)
    print("=" * 72)
    print(audit.render())
    print()
    print(f"TEST BUY SIZE: ${float(args.order_usd):.6f} (READ-ONLY / NO EXECUTION)")
    _snap = getattr(args, "_sol_snapshot", None)
    if _snap is not None:
        print(f"SOL reference : ${_snap.usd_per_sol:,.4f}  "
              f"(recorded, so the notional is reproducible)")
    print(f"Gap threshold : "
          f"{args.max_mark_gap_pct}%" if args.max_mark_gap_pct is not None
          else "NOT CONFIGURED - results will be INFORMATIVE, never PASS")
    print(f"Channel       : @{str(args.channel).lstrip('@')}")
    print("Waiting for NEW messages. Ctrl+C to stop.")
    print("-" * 72)

    reports: list[SignalReport] = []

    async def on_message(
        text: str,
        message_id: int | None,
        telegram_ts: float | None,
        received_at: float,
    ) -> None:
        report = await pipeline.run(
            raw_text=text,
            channel=str(args.channel),
            message_id=message_id,
            telegram_timestamp=telegram_ts,
        )
        reports.append(report)
        print()
        print(render_report(report))
        print()
        print(render_summary(reports))

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    try:
        loop.add_signal_handler(__import__("signal").SIGINT, stop.set)
    except (NotImplementedError, RuntimeError):
        pass

    try:
        result = await listen(telegram, on_message, stop=stop)
        print(f"\nlistener stopped: {result.stopped_reason}")
    except asyncio.CancelledError:
        pass
    finally:
        if rpc_is_set(pipeline):
            await pipeline.rpc.aclose()
        if pipeline.jupiter is not None:
            await pipeline.jupiter.aclose()

    if reports:
        written, latest = write_report_files(
            reports, root=_root(args), sol_usd=config_sol(args),
            sol_metadata=config_sol_metadata(args),
        )
        print(f"\nwrote {len(written)} JSON report(s) and {latest}")
    print("NO TRANSACTION WAS BROADCAST. NO FUNDS MOVED.")
    return 0


def rpc_is_set(pipeline: SameMarketPipeline) -> bool:
    return pipeline.rpc is not None


def config_sol(args: Any) -> float | None:
    """
    The SOL/USD reference for this run, or ``None`` if it was never resolved.

    Prefers the resolved snapshot (fetched or overridden) so every downstream
    consumer -- sizing, report rendering, the goat section -- reads ONE number.
    """
    snap = getattr(args, "_sol_snapshot", None)
    if snap is not None:
        return float(snap.usd_per_sol)
    override = getattr(args, "sol_usd_override", None)
    if override is not None:
        return float(override)
    deprecated = getattr(args, "sol_usd", None)
    return float(deprecated) if deprecated is not None else None


def config_sol_metadata(args: Any) -> dict[str, Any]:
    """
    The SOL/USD snapshot metadata block recorded beside results.

    Exactly the reported keys: ``sol_usd``, ``sol_usd_source``,
    ``sol_usd_timestamp``, ``sol_usd_age_ms`` (plus the provenance extras this
    module already records). No token price, executable quote, fill or PnL
    field is touched.
    """
    snap = getattr(args, "_sol_snapshot", None)
    if snap is not None:
        return snap.as_dict()
    from .sol_price import override_snapshot

    raw = config_sol(args)
    if raw is None:
        return {"sol_usd": None, "sol_usd_source": None,
                "sol_usd_timestamp": None, "sol_usd_age_ms": None}
    try:
        return override_snapshot(raw, source="operator (cli)").as_dict()
    except Exception:  # noqa: BLE001 - metadata must never break a report
        return {"sol_usd": raw, "sol_usd_source": "operator (cli)",
                "sol_usd_timestamp": None, "sol_usd_age_ms": None}


async def _resolve_sol_snapshot(args: Any) -> Any:
    """
    Obtain the SOL/USD reference for this run, or refuse.

    Resolution order, and nothing else:

    1. ``--sol-usd-override`` (or the deprecated ``--sol-usd``) -- the
       offline/test-only deterministic input, which makes no network call;
    2. one fetch from the free keyless Jupiter price API, cached for the run.

    There is no third option and no default. The harness REFUSES rather than
    guessing a price, because an unlogged reference makes the order size
    unreproducible, and an unreproducible order size makes the whole comparison
    unverifiable -- which is why the resolved snapshot is always recorded.
    """
    from .sol_price import JupiterSolPriceProvider, resolve_sol_usd_async

    override = getattr(args, "sol_usd_override", None)
    deprecated = getattr(args, "sol_usd", None)
    if override is not None or deprecated is not None:
        return await resolve_sol_usd_async(override=override,
                                          deprecated=deprecated)

    provider = JupiterSolPriceProvider(
        timeout_s=float(getattr(args, "provider_timeout_s", 8.0)),
        ttl_s=float(getattr(args, "cache_ttl_s", 60.0)),
    )
    try:
        return await resolve_sol_usd_async(provider=provider)
    finally:
        await provider.aclose()


def _root(args: Any) -> Path:
    root = getattr(args, "root", None)
    return Path(root) if root else Path.cwd()


async def _run_offline(args: Any, pipeline: SameMarketPipeline) -> int:
    """
    Measure a message supplied inline, without touching Telegram.

    Useful for a deterministic check of the whole chain against a captured
    message, and it is how the harness can be exercised end to end on a
    machine with no session.
    """
    text = sys.stdin.read() if args.message == "-" else (args.message or "")
    if not text.strip():
        print("REFUSED: no message to measure. Pass --message '<text>' or "
              "--message - to read stdin.", file=sys.stderr)
        return 2

    report = await pipeline.run(
        raw_text=text,
        channel=str(args.channel),
        message_id=None,
        telegram_timestamp=None,
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, default=str))
    else:
        print(render_report(report))

    written, latest = write_report_files(
        [report], root=_root(args), sol_usd=config_sol(args),
        sol_metadata=config_sol_metadata(args),
    )
    print(f"\nwrote {written[0]} and {latest}")
    print("NO TRANSACTION WAS BROADCAST. NO FUNDS MOVED.")
    if pipeline.rpc is not None:
        await pipeline.rpc.aclose()
    if pipeline.jupiter is not None:
        await pipeline.jupiter.aclose()
    return 0


def run_same_market_test(cfg: Any, args: Any) -> int:
    """
    Entry point for ``autopaper same-market-test``.

    Returns an exit code: 0 on a completed measurement, 2 when the harness
    refused to run. A REJECT verdict is a successful measurement and exits 0 -
    the command did its job and the answer was "no".
    """
    args._cfg = cfg

    # Resolve the SOL/USD reference BEFORE any client exists, so a run that
    # cannot obtain one refuses instead of half-starting.
    from .sol_price import SolPriceUnavailable, describe_snapshot

    try:
        args._sol_snapshot = asyncio.run(_resolve_sol_snapshot(args))
    except SolPriceUnavailable as exc:
        print("REFUSED: no SOL/USD reference could be obtained, and this "
              "harness will not guess one. Sizing --order-usd in lamports "
              "requires it, and an unlogged reference would make the whole "
              "comparison unverifiable.", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        print("  For an offline deterministic run pass --sol-usd-override "
              "<price>.", file=sys.stderr)
        return 2
    print(f"SOL reference : {describe_snapshot(args._sol_snapshot)}")
    print("                 (USD->SOL conversion only; not a token price and "
          "not used for pricing)")

    pipeline, _config, _rpc = build_pipeline(args)
    try:
        if args.message:
            return asyncio.run(_run_offline(args, pipeline))
        return asyncio.run(_run_listener(args, pipeline))
    except KeyboardInterrupt:
        print("\ninterrupted. NO TRANSACTION WAS BROADCAST.")
        return 0
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - reported, not dumped as a traceback
        print(f"ERROR: {exc}", file=sys.stderr)
        print("NO TRANSACTION WAS BROADCAST.", file=sys.stderr)
        return 2


__all__ = [
    "BANNER",
    "DEFAULT_RPC_URL",
    "REPORT_DIR",
    "build_pipeline",
    "render_goat_section",
    "run_same_market_test",
    "write_report_files",
]