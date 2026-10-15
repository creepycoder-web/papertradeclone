"""
One-shot, READ-ONLY Solana WebSocket probe for the native market monitor.

WHAT IT DOES
------------
    connect -> subscribe once -> wait a BOUNDED time for one notification
            -> unsubscribe -> close -> report -> exit

It answers the only question an operator has after a run that logged
``market data unusable for 123s``: is the configured ``wss://`` endpoint
actually serving the PubSub feed the native monitor intends to use, and how
fresh is it?

    PYTHONPATH=src python scripts/native_monitor_ws_probe.py \
        --url wss://api.mainnet.solana.com --mode slot --timeout 10

    # the real monitoring subscription, against one specific venue account
    PYTHONPATH=src python scripts/native_monitor_ws_probe.py \
        --url wss://api.mainnet.solana.com --mode account --account <CURVE>

WHAT IT CANNOT DO
-----------------
It is not the bot. It never starts, imports or touches a trading engine, a
position, the ledger or ``var/``. It sends exactly two JSON-RPC messages and
nothing else: one ``accountSubscribe``/``slotSubscribe`` and, if the first one
worked, one ``accountUnsubscribe``/``slotUnsubscribe``. It holds no keypair, so
it cannot sign, broadcast or simulate anything, and it has no exit-code path
that loops. There is no retry loop: one connection, one attempt, one timeout,
one exit.

The endpoint is a VALUE the operator supplies or the environment provides; it is
never guessed, and a URL carrying a token is never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
from pathlib import Path

# The repo src/ layout, so this runs without an install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autopaper.same_market.ws import (  # noqa: E402
    WebSocketClosed,
    WebSocketConnection,
    WebSocketUnavailable,
    parse_ws_url,
)

#: The commitment the native monitor subscribes with. ``processed`` is the
#: freshest commitment a public endpoint serves for a push feed; an equivalent
#: ``confirmed`` subscription is available via ``--commitment``.
DEFAULT_COMMITMENT = "processed"

#: The documented Solana public mainnet HTTP endpoint. Its PubSub twin is
#: ``wss://api.mainnet.solana.com``. The two are separate services on the same
#: cluster: this default is a place to START, not a proof that either works, and
#: a private/provider endpoint should be used for production monitoring.
PUBLIC_HTTP_URL = "https://api.mainnet.solana.com"

#: Per-read cap. A notification stream must not be able to hang the probe, and
#: it is small on purpose: the probe is bounded in TIME by ``--timeout`` anyway.
READ_TIMEOUT_S = 20.0


def _endpoint_from_env() -> str:
    """``NATIVE_MONITOR_WS_URL`` if it is set, else the documented public one."""
    return (os.environ.get("NATIVE_MONITOR_WS_URL") or "").strip() or (
        f"wss://{PUBLIC_HTTP_URL.split('//', 1)[1]}")


async def probe(url: str, mode: str, account: str, timeout_s: float,
                commitment: str) -> int:
    """
    One bounded round trip. Returns a process exit code; never raises.
    """
    started = time.monotonic()
    report: dict[str, object] = {
        "endpoint_host": "", "mode": mode, "commitment": commitment,
        "connected": False, "subscribed": False, "notifications": 0,
        "latency_ms": {}, "error": "",
    }
    try:
        target = parse_ws_url(url)
    except WebSocketUnavailable as exc:
        report["error"] = f"not a websocket endpoint: {exc}"
        print(json.dumps(report, indent=2))
        return 2
    # The HOST is reported; the URL is not, so a token in a query string cannot
    # reach a log or a terminal.
    report["endpoint_host"] = target.host

    connection = WebSocketConnection(target, open_timeout_s=min(10.0, timeout_s))
    handshake_started = time.monotonic()
    try:
        await connection.connect()
    except (WebSocketUnavailable, OSError) as exc:
        report["latency_ms"]["handshake"] = round(
            (time.monotonic() - handshake_started) * 1000.0, 1)
        report["error"] = f"{type(exc).__name__}: {exc}"[:300]
        print(json.dumps(report, indent=2))
        return 3
    report["connected"] = True
    report["latency_ms"]["handshake"] = round(
        (time.monotonic() - handshake_started) * 1000.0, 1)

    # ---- the subscription request -------------------------------------
    request_id = 1
    if mode == "account":
        params = [account, {"encoding": "base64", "commitment": commitment}]
        method = "accountSubscribe"
    else:
        # ``slotSubscribe`` is the minimum safe diagnostic feed: it is the same
        # connection, handshake, id-matching and framing path the monitor uses,
        # it needs no account, and it carries no market data at all.
        #
        # It takes NO parameters on the current protocol version. Passing a
        # config object is answered with ``-32602 Invalid parameters: No
        # parameters were expected`` -- verified against the public endpoint,
        # 2026-10-11 -- so the empty array is the correct request, not a
        # simplification.
        params = []
        method = "slotSubscribe"
    request = {"jsonrpc": "2.0", "id": request_id, "method": method,
               "params": params}
    subscription_id: int | None = None

    async def wait_for_subscription(deadline: float) -> int | None:
        """Read frames until this id is answered, erroring, or the deadline."""
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                raw = await asyncio.wait_for(connection.recv_text(),
                                             timeout=max(0.1, remaining))
            except asyncio.TimeoutError:
                return None
            except WebSocketUnavailable as exc:
                report["error"] = f"stream failed: {exc}"[:300]
                return None
            try:
                message = json.loads(raw)
            except ValueError:
                # A non-JSON frame is a transport anomaly, not a subscription.
                report["error"] = "endpoint sent a non-JSON frame"
                continue
            if not isinstance(message, dict) or message.get("id") != request_id:
                # Not the answer to THIS request. Ignored: a probe must not
                # confuse another subscription's notification with its own.
                continue
            if isinstance(message.get("error"), dict):
                report["error"] = (
                    f"subscription REJECTED: {message['error'].get('code')} "
                    f"{message['error'].get('message')}")[:300]
                return None
            result = message.get("result")
            if not isinstance(result, int):
                report["error"] = (
                    f"subscription answer was not an id: {result!r}")[:300]
                return None
            return result
        return None

    deadline = time.monotonic() + timeout_s
    first_notification_at: float | None = None
    try:
        await connection.send_text(json.dumps(request))
        subscription_id = await wait_for_subscription(deadline)
    except WebSocketUnavailable as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"[:300]
        subscription_id = None
    report["subscribed"] = subscription_id is not None

    # ---- one bounded read window --------------------------------------
    if subscription_id is not None:
        read_deadline = time.monotonic() + timeout_s
        while time.monotonic() < read_deadline:
            try:
                raw = await asyncio.wait_for(
                    connection.recv_text(),
                    timeout=max(0.1, read_deadline - time.monotonic()))
            except asyncio.TimeoutError:
                break
            except WebSocketUnavailable as exc:
                report["error"] = f"stream failed: {exc}"[:300]
                break
            try:
                message = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(message, dict):
                continue
            if mode == "account":
                is_ours = (
                    message.get("method") == "accountNotification"
                    and isinstance(message.get("params"), dict)
                    and message["params"].get("subscription") == subscription_id)
            else:
                is_ours = message.get("method") == "slotNotification" or (
                    "result" in message)
            if not is_ours:
                continue
            if first_notification_at is None:
                first_notification_at = time.monotonic()
            report["notifications"] = int(report["notifications"]) + 1
            if int(report["notifications"]) >= 1:
                # One is enough. A probe is a liveness check, not a measurement,
                # and staying attached any longer than that is an unbounded
                # connection loop by another name.
                break
    report["latency_ms"]["time_to_first_notification"] = (
        None if first_notification_at is None
        else round((first_notification_at - handshake_started) * 1000.0, 1))
    report["latency_ms"]["total"] = round((time.monotonic() - started) * 1000.0, 1)

    # ---- tidy up -------------------------------------------------------
    if subscription_id is not None:
        unsubscribe = {"jsonrpc": "2.0", "id": request_id + 1,
                       "method": ("accountUnsubscribe" if mode == "account"
                                  else "slotUnsubscribe"),
                       "params": [subscription_id]}
        with contextlib.suppress(Exception):
            await connection.send_text(json.dumps(unsubscribe))
    await connection.aclose()

    ok = bool(report["connected"]) and bool(report["subscribed"]) \
        and int(report["notifications"]) >= 1
    report["verdict"] = (
        "OK: the endpoint served the subscription the monitor sends"
        if ok else
        "UNUSABLE: no notification arrived within the bound, so this endpoint "
        "cannot currently feed the native monitor")
    print(json.dumps(report, indent=2))
    return 0 if ok else 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="One-shot read-only Solana WebSocket probe. It subscribes, "
                    "reads one notification and disconnects. It never trades, "
                    "signs or broadcasts.")
    parser.add_argument(
        "--url", default=None,
        help="wss:// endpoint. Defaults to $NATIVE_MONITOR_WS_URL, then the "
             "documented public mainnet PubSub endpoint.")
    parser.add_argument("--mode", choices=("slot", "account"), default="slot",
                        help="slot = the minimum safe diagnostic feed "
                             "(default); account = the monitor's own "
                             "accountSubscribe against one venue account")
    parser.add_argument("--account", default=None,
                        help="base58 pubkey of the venue account to subscribe "
                             "to. Required with --mode account")
    parser.add_argument("--commitment", default=DEFAULT_COMMITMENT)
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="bounded seconds for the subscription AND the "
                             "notification wait. Default 10.")
    args = parser.parse_args(argv)

    if args.mode == "account" and not args.account:
        parser.error("--mode account requires --account <BASE58_PUBKEY>")
    if args.timeout <= 0:
        parser.error("--timeout must be positive; the probe is bounded by "
                     "design")
    return asyncio.run(probe(
        args.url or _endpoint_from_env(), args.mode, args.account or "",
        args.timeout, args.commitment))


if __name__ == "__main__":
    raise SystemExit(main())
