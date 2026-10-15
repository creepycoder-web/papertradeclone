"""
The WebSocket path of native market monitoring, from configuration to price.

WHAT THIS FILE IS FOR
---------------------
The push channel was already built -- :mod:`autopaper.same_market.ws` is the
transport, :mod:`autopaper.paper_live.native_monitor` owns subscriptions -- but
``build_config`` never read ``NATIVE_MONITOR_WS_URL``, so ``MarketConfig`` always
carried the dataclass default ``""`` and the engine's own gate
(``native_monitor_enabled AND a non-empty URL``) was never satisfied. Every
monitoring tick therefore took the RPC polling recovery path, and the shared
request budget was spent on reads the push channel exists to avoid.

These tests pin the WHOLE chain, not the config alone:

    configuration -> MonitorConfig -> websocket connect -> accountSubscribe
      -> accountNotification -> local price -> freshness -> identity -> engine

They use a LOCAL fake websocket endpoint (a real RFC 6455 handshake over
127.0.0.1, driven by the production ``WebSocketConnection``), so the connection,
framing, id-matching and reconnect code are the shipped code -- only the socket
is local. No test here touches a network, a wallet, a live position or ``var/``.

WHAT IS NOT TESTED HERE, DELIBERATELY: that any particular provider serves
websockets. That is a question about an endpoint, not about this code, and the
read-only probe answers it (``scripts/native_monitor_ws_probe.py``).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import struct
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_paper_live import (  # noqa: E402
    MINT,
    PAIR,
    FakeQuoteService,
    make_engine,
    run,
    sell_point,
    signal,
)

from autopaper.config import ExitConfig, build_config  # noqa: E402
from autopaper.exits import ExitPolicy  # noqa: E402
from autopaper.paper_live.arms import parse_arms  # noqa: E402
from autopaper.paper_live.native_monitor import (  # noqa: E402
    MonitorConfig,
    NativeMarketMonitor,
    NativeMonitorRegistry,
    StaticVenueIdentity,
    SOURCE_PUSH,
    STATE_HEALTHY,
    STATE_STALE,
    STATE_STOPPED,
)
from autopaper.same_market import pumpdirect as pd  # noqa: E402
from autopaper.same_market.pda import b58encode  # noqa: E402
from autopaper.same_market.pumpdirect import (  # noqa: E402
    MarketKind,
    decode_bonding_curve,
    derive_bonding_curve,
)
from autopaper.same_market.ws import WebSocketUnavailable  # noqa: E402

try:
    from tests import pumpdirect_accounts as _acct
except ImportError:
    import pumpdirect_accounts as _acct


CURVE_ADDRESS, _CURVE_BUMP = derive_bonding_curve(MINT)
ROLE_CURVE = "CURVE"
_HANDSHAKE_GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


# ---------------------------------------------------------------------------
# a local RFC 6455 endpoint, good enough for Solana's JSON-RPC-over-WS
# ---------------------------------------------------------------------------


def _server_frame(payload: bytes) -> bytes:
    """One UNMASKED server-to-client text frame (servers must not mask)."""
    header = bytearray([0x81])                       # FIN + TEXT
    size = len(payload)
    if size < 126:
        header.append(size)
    elif size < (1 << 16):
        header.append(126)
        header.extend(struct.pack("!H", size))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", size))
    return bytes(header) + payload


async def _read_client_frame(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    """Read one (possibly masked) client frame. Enough of RFC 6455 to talk."""
    head = await reader.readexactly(2)
    opcode = head[0] & 0x0F
    size = head[1] & 0x7F
    if size == 126:
        size = struct.unpack("!H", await reader.readexactly(2))[0]
    elif size == 127:
        size = struct.unpack("!Q", await reader.readexactly(8))[0]
    mask = await reader.readexactly(4) if head[1] & 0x80 else b""
    payload = bytearray(await reader.readexactly(size))
    for index in range(size):
        payload[index] ^= mask[index % 4]
    return opcode, bytes(payload)


def _notification(subscription_id: int, raw: bytes, *, slot: int = 4_242) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "accountNotification",
        "params": {
            "subscription": subscription_id,
            "result": {
                "context": {"slot": slot},
                "value": {
                    "data": [base64.b64encode(raw).decode(), "base64"],
                    "owner": pd.PUMP_FUN_PROGRAM,
                },
            },
        },
    }


class FakeWsEndpoint:
    """
    A local websocket endpoint that speaks Solana's PubSub subset.

    It performs the real handshake (so the production transport's
    ``Sec-WebSocket-Accept`` check is exercised), answers
    ``accountSubscribe`` with a real integer subscription id, and pushes
    ``accountNotification`` frames built from the curve fixtures. It records
    every subscribe request it received, per connection, which is what makes
    "a reconnect did not duplicate my subscriptions" assertable.
    """

    def __init__(self, *, reject_subscriptions: bool = False,
                 push_interval_s: float = 0.02,
                 close_after_first: bool = False,
                 reserve_step: int = 0) -> None:
        self.reject_subscriptions = reject_subscriptions
        self.push_interval_s = push_interval_s
        self.close_after_first = close_after_first
        #: When non-zero, the Nth pushed curve carries base + N*step quote
        #: reserves, so every push is a strictly different price. A fixed
        #: fixture pair cannot answer "did the freshness timestamp move": the
        #: first read can already be the final value.
        self.reserve_step = reserve_step
        self.curves: list[bytes] = [_acct.bonding_curve_bytes()]
        #: ``{connection_index: [request dicts]}``
        self.subscribe_requests: dict[int, list[dict]] = {}
        self.connections = 0
        self.url = ""
        self._server: asyncio.AbstractServer | None = None
        self._handlers: list[asyncio.Task] = []

    async def start(self) -> "FakeWsEndpoint":
        self._server = await asyncio.start_server(
            self._handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}"
        return self

    async def aclose(self) -> None:
        for task in self._handlers:
            task.cancel()
        for task in self._handlers:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._handlers.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    @property
    def total_subscribes(self) -> int:
        return sum(len(v) for v in self.subscribe_requests.values())

    async def _handle(self, reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter) -> None:
        # ---- the handshake, byte for byte what the client expects ----
        lines: list[bytes] = []
        while True:
            line = await reader.readline()
            if not line or line in (b"\r\n", b"\n"):
                break
            lines.append(line)
        headers = {}
        for line in lines:
            name, _, value = line.decode("latin-1").partition(":")
            headers[name.strip().lower()] = value.strip()
        key = headers.get("sec-websocket-key", "")
        accept = base64.b64encode(
            hashlib.sha1(key.encode() + _HANDSHAKE_GUID).digest()).decode()
        writer.write(
            b"HTTP/1.1 101 Switching Protocols\r\n"
            b"Upgrade: websocket\r\n"
            b"Connection: Upgrade\r\n"
            b"Sec-WebSocket-Accept: " + accept.encode() + b"\r\n"
            b"\r\n")
        await writer.drain()

        index = self.connections
        self.connections += 1
        self.subscribe_requests.setdefault(index, [])
        subscription_id = 100 + index * 10
        try:
            sent = 0
            while True:
                opcode, payload = await _read_client_frame(reader)
                if opcode != 0x1:                     # not a TEXT frame
                    continue
                message = json.loads(payload.decode("utf-8"))
                if message.get("method") == "accountSubscribe":
                    self.subscribe_requests[index].append(message)
                    if self.reject_subscriptions:
                        writer.write(_server_frame(json.dumps({
                            "jsonrpc": "2.0", "id": message.get("id"),
                            "error": {"code": -32602,
                                      "message": "Invalid parameters"},
                        }).encode()))
                        await writer.drain()
                        continue
                    subscription_id += 1
                    writer.write(_server_frame(json.dumps({
                        "jsonrpc": "2.0", "id": message.get("id"),
                        "result": subscription_id,
                    }).encode()))
                    await writer.drain()
                    await self._push(writer, subscription_id, sent)
                    sent += 1
                    if self.close_after_first:
                        writer.close()
                        return
                elif message.get("method") == "accountUnsubscribe":
                    writer.write(_server_frame(json.dumps({
                        "jsonrpc": "2.0", "id": message.get("id"),
                        "result": True,
                    }).encode()))
                    await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionResetError,
                asyncio.CancelledError):
            return
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001 - best effort on a dying socket
                pass

    async def _push(self, writer: asyncio.StreamWriter,
                    subscription_id: int, index: int) -> None:
        """Push the fixture curves, then keep the stream alive until closed."""
        while True:
            if self.reserve_step:
                raw = _acct.bonding_curve_bytes(
                    virtual_quote_reserves=100_562_804_716
                    + index * self.reserve_step)
            else:
                raw = self.curves[min(index, len(self.curves) - 1)]
            writer.write(_server_frame(json.dumps(
                _notification(subscription_id, raw,
                              slot=4_242 + index)).encode()))
            await writer.drain()
            index += 1
            if self.close_after_first:
                # Only ONE notification, then the socket dies: the reconnect
                # fixture. Returning (rather than sleeping inside an infinite
                # loop) is what makes the close reachable at all.
                return
            await asyncio.sleep(self.push_interval_s)


def _identity(**kw) -> StaticVenueIdentity:
    base = dict(
        mint=MINT, market_type=MarketKind.PUMP_BONDING_CURVE.value,
        market_address=CURVE_ADDRESS, base_mint=MINT,
        quote_mint="So11111111111111111111111111111111111111112",
        token_decimals=6,
    )
    base.update(kw)
    return StaticVenueIdentity(**base)


def _monitor_config(url: str, **kw) -> MonitorConfig:
    base = dict(ws_url=url, stale_after_ms=5_000.0,
                open_timeout_s=5.0, reconnect_min_s=0.05,
                reconnect_max_s=0.2)
    base.update(kw)
    return MonitorConfig(**base)


async def _wait_for_price(monitor: NativeMarketMonitor, *,
                          timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        point = monitor.latest()
        if point is not None and point.usable:
            return point
        await asyncio.sleep(0.01)
    return None


def _tmp_config(tmp_path: Path, body: str):
    (tmp_path / ".env.local").write_text(body)
    return build_config(tmp_path)



# ===========================================================================
# 1. CONFIGURATION: the setting reaches MarketConfig, or is diagnosed
# ===========================================================================


def test_the_ws_url_survives_the_config_loader(tmp_path: Path):
    """
    THE ROOT CAUSE, as a test.

    ``MarketConfig`` declared ``native_monitor_ws_url`` from the moment the push
    channel shipped, and ``build_config`` never passed it, so the value stayed
    ``""`` no matter what the operator wrote. This asserts the whole transport
    of the setting: env file -> MarketConfig.
    """
    cfg = _tmp_config(tmp_path, "NATIVE_MONITOR_WS_URL=wss://example.invalid\n")
    assert cfg.market.native_monitor_ws_url == "wss://example.invalid"
    assert cfg.market.native_monitor_enabled is True
    assert cfg.config_warnings == ()


def test_the_other_monitor_settings_survive_too(tmp_path: Path):
    cfg = _tmp_config(
        tmp_path,
        "NATIVE_MONITOR_WS_URL=wss://example.invalid\n"
        "NATIVE_MONITOR_ENABLED=false\n"
        "NATIVE_MONITOR_STALE_AFTER_MS=9000\n"
        "NATIVE_MONITOR_OPEN_TIMEOUT_S=3\n")
    assert cfg.market.native_monitor_enabled is False
    assert cfg.market.native_monitor_stale_after_ms == 9000.0
    assert cfg.market.native_monitor_open_timeout_s == 3.0


def test_the_process_environment_can_supply_the_endpoint(tmp_path: Path,
                                                         monkeypatch):
    """Env is the fallback, so a CI or shell-exported value still works."""
    monkeypatch.setenv("NATIVE_MONITOR_WS_URL", "wss://env.example.invalid")
    cfg = _tmp_config(tmp_path, "NATIVE_MONITOR_STALE_AFTER_MS=7000\n")
    assert cfg.market.native_monitor_ws_url == "wss://env.example.invalid"


def test_an_empty_url_is_the_documented_state_and_only_that(tmp_path: Path):
    """
    No endpoint is NOT an error: the polling recovery path remains correct, it
    is simply expensive. The diagnostic says so instead of staying silent.
    """
    cfg = _tmp_config(tmp_path, "")
    assert cfg.market.native_monitor_ws_url == ""
    assert len(cfg.config_warnings) == 1
    assert "NATIVE_MONITOR_WS_URL" in cfg.config_warnings[0]
    assert "polling" in cfg.config_warnings[0]


def test_an_http_url_is_refused_without_being_converted(tmp_path: Path):
    """
    ``RPC_URL`` is an HTTP endpoint and must NOT silently become a websocket.

    The Solana HTTP and PubSub services are separate. Rewriting the scheme would
    hand the operator a monitor pointed at a host that may not serve websockets,
    precisely while reporting that monitoring is enabled.
    """
    cfg = _tmp_config(tmp_path, "NATIVE_MONITOR_WS_URL=https://api.mainnet.solana.com\n")
    assert cfg.market.native_monitor_ws_url == "", (
        "an HTTP URL was accepted as a websocket endpoint")
    assert cfg.config_warnings and "NATIVE_MONITOR_WS_URL" in cfg.config_warnings[0]


def test_a_url_with_no_scheme_at_all_is_refused(tmp_path: Path):
    cfg = _tmp_config(tmp_path, "NATIVE_MONITOR_WS_URL=api.mainnet.solana.com\n")
    assert cfg.market.native_monitor_ws_url == ""
    assert cfg.config_warnings, "a scheme-less URL produced no diagnostic"


def test_a_host_only_url_is_refused(tmp_path: Path):
    cfg = _tmp_config(tmp_path, "NATIVE_MONITOR_WS_URL=wss://\n")
    assert cfg.market.native_monitor_ws_url == ""
    assert cfg.config_warnings, "a hostless URL produced no diagnostic"


def test_the_diagnostic_never_carries_the_value(tmp_path: Path):
    """
    A hosted provider's endpoint routinely carries a token. A diagnostic that
    echoed it would put a credential in the log.
    """
    secret = "s3cr3t-token-value"
    cfg = _tmp_config(tmp_path, f"NATIVE_MONITOR_WS_URL=wss://rpc.example/{secret}\n")
    assert cfg.market.native_monitor_ws_url == "wss://rpc.example/s3cr3t-token-value"
    for warning in cfg.config_warnings:
        assert secret not in warning


def test_the_configuration_reaches_the_engine_gate():
    """The engine's own gate must be satisfied by a configured endpoint."""
    from tests.paper_bridge_fakes import bridge

    engine = bridge(native_monitor_enabled=True,
                    native_monitor_ws_url="ws://127.0.0.1:1")
    assert engine._effective_native_monitor_url() == "ws://127.0.0.1:1"

    armed = bridge(native_monitor_enabled=True,
                    native_monitor_ws_url="https://api.mainnet.solana.com")
    assert armed._effective_native_monitor_url() == ""
    assert armed._ensure_native_monitor(
        SimpleNamespace(mint=MINT, token_decimals=6, entry_pool_address=None,
                        entry_venue_id="", cached_identity=None),
        SimpleNamespace(market_type="PUMP_BONDING_CURVE", pool_or_curve="")) is None


# ===========================================================================
# 2. THE LIVE PATH: connect -> accountSubscribe -> notification -> price
# ===========================================================================


def test_a_configured_endpoint_yields_a_subscription_and_a_price():
    """
    THE END-TO-END ASSERTION.

    A real socket is opened to a local endpoint through the production
    ``WebSocketConnection``; the production ``SharedSubscriptionStream`` sends
    ``accountSubscribe`` on it; the endpoint answers with a subscription id and
    pushes a curve account; the production monitor decodes it and serves a
    FRESH price. If any link in that chain is broken, this fails.
    """

    async def _run():
        endpoint = await FakeWsEndpoint().start()
        try:
            config = _monitor_config(endpoint.url)
            registry = NativeMonitorRegistry()
            monitor = registry.acquire(_identity(), config)
            try:
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline and endpoint.total_subscribes < 1:
                    await asyncio.sleep(0.01)
                assert endpoint.total_subscribes >= 1, (
                    "the stream connected but sent no accountSubscribe")
                request = endpoint.subscribe_requests[0][0]["params"]
                assert request[0] == CURVE_ADDRESS, (
                    "the monitor subscribed to the wrong account")
                assert request[1]["encoding"] == "base64"
                assert request[1]["commitment"] == "processed"

                point = await _wait_for_price(monitor)
                assert point is not None, "no usable price arrived from the push"
                assert point.source == SOURCE_PUSH
                expected = decode_bonding_curve(
                    endpoint.curves[0], curve_address=CURVE_ADDRESS, mint=MINT)
                assert point.price_sol_per_token is not None
                assert point.price_sol_per_token > 0
                assert expected.virtual_quote_reserves > 0
                assert monitor.state == STATE_HEALTHY
            finally:
                await registry.aclose()
        finally:
            await endpoint.aclose()

    asyncio.run(_run())


def test_the_pushed_price_updates_the_freshness_timestamp():
    """A new push must be a NEW freshness timestamp, not a reused one."""

    async def _run():
        endpoint = await FakeWsEndpoint(
            push_interval_s=0.01, reserve_step=1_000).start()
        try:
            registry = NativeMonitorRegistry()
            monitor = registry.acquire(_identity(), _monitor_config(endpoint.url))
            try:
                first = await _wait_for_price(monitor)
                assert first is not None
                previous = first.price_sol_per_token
                deadline = time.monotonic() + 3.0
                moved = False
                while time.monotonic() < deadline:
                    point = monitor.latest()
                    if point is not None and point.usable and \
                            point.price_sol_per_token != previous:
                        moved = True
                        assert point.age_ms < 1_000.0, (
                            "a new push was served with a stale age")
                        break
                    await asyncio.sleep(0.005)
                assert moved, "the pushed price never moved with the reserves"
                assert monitor.stats.updates >= 2
            finally:
                await registry.aclose()
        finally:
            await endpoint.aclose()

    asyncio.run(_run())


def test_a_rejected_subscription_is_reported_not_faked():
    """
    A ``-32602`` answer must leave the monitor with NO price, never with a
    manufactured one. The stream records the refusal and the monitor goes STALE.
    """

    async def _run():
        endpoint = await FakeWsEndpoint(reject_subscriptions=True).start()
        try:
            registry = NativeMonitorRegistry()
            monitor = registry.acquire(_identity(), _monitor_config(endpoint.url))
            try:
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline and endpoint.total_subscribes < 1:
                    await asyncio.sleep(0.01)
                assert endpoint.total_subscribes == 1
                # Give the refusal time to be dispatched and recorded.
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline and \
                        not monitor.stats.last_error:
                    await asyncio.sleep(0.01)
                assert monitor.latest() is None, (
                    "a price was served for a subscription the endpoint refused")
                assert monitor.state == STATE_STALE, monitor.state
                assert "refused" in monitor.stats.last_error
                assert monitor.stats.subscription_ids == {}, (
                    "the monitor believes it holds a subscription it was refused")
            finally:
                await registry.aclose()
        finally:
            await endpoint.aclose()

    asyncio.run(_run())


def test_a_malformed_notification_cannot_become_a_price():
    """
    An unusable envelope yields NO bytes, which is a decode failure -- not a
    zero-reserve curve, and not a price.
    """

    async def _run():
        endpoint = await FakeWsEndpoint().start()
        endpoint.curves = [b""]                      # the wrong envelope
        try:
            registry = NativeMonitorRegistry()
            monitor = registry.acquire(_identity(), _monitor_config(endpoint.url))
            try:
                await asyncio.sleep(0.3)
                assert monitor.stats.updates == 0
                assert monitor.stats.decode_failures > 0
                assert monitor.latest() is None
            finally:
                await registry.aclose()
        finally:
            await endpoint.aclose()

    asyncio.run(_run())


def test_a_silent_stream_is_never_reported_as_fresh():
    """
    THE FAILURE MODE THAT PRODUCED "market data unusable for 123s".

    An open socket proves nothing. Once the staleness bound passes with no push,
    ``latest()`` must refuse the last number rather than serve it as current.
    """
    config = MonitorConfig(ws_url="ws://unused", stale_after_ms=50.0,
                           healthy_after_updates=1)
    monitor = NativeMarketMonitor(_identity(), config=config)
    from autopaper.paper_live.native_monitor import SharedSubscriptionStream

    stream = SharedSubscriptionStream("ws://unused", config=config)
    monitor._stream = stream
    stream._monitors.append(monitor)
    for address, role in monitor.identity.subscription_addresses():
        stream._routes[7] = (monitor, address, role)
    monitor.stream_resubscribed({7: (CURVE_ADDRESS, ROLE_CURVE)})
    monitor.deliver(ROLE_CURVE, _acct.bonding_curve_bytes(), 10, CURVE_ADDRESS)
    assert monitor.latest() is not None and monitor.latest().usable

    time.sleep(0.2)
    point = monitor.latest()
    assert point is not None and not point.usable
    assert "staleness bound" in point.reason


def test_a_disconnect_triggers_a_bounded_reconnect_and_no_duplicate_subscriptions():
    """
    RECONNECT BEHAVIOUR, MEASURED.

    The endpoint closes the connection after the first subscription. The stream
    must reconnect, re-subscribe ONCE per market, and not accumulate routes -- a
    duplicate subscription would double the push rate for the same bytes, which
    is the waste the shared connection exists to prevent.
    """

    async def _run():
        endpoint = await FakeWsEndpoint(close_after_first=True).start()
        try:
            registry = NativeMonitorRegistry()
            monitor = registry.acquire(_identity(), _monitor_config(endpoint.url))
            try:
                deadline = time.monotonic() + 4.0
                while time.monotonic() < deadline and endpoint.connections < 2:
                    await asyncio.sleep(0.01)
                assert endpoint.connections >= 2, (
                    "the stream never reconnected after the socket closed")
                # One market, one account, one subscription per connection.
                for connection_index, requests in \
                        endpoint.subscribe_requests.items():
                    assert len(requests) == 1, (
                        f"connection {connection_index} sent "
                        f"{len(requests)} accountSubscribe calls for ONE market")
                assert len(monitor._subscriptions) == 1, (
                    "reconnect left duplicate subscription ids on the monitor")
                assert len(monitor.stats.subscription_ids) == 1
                assert monitor.state in (STATE_HEALTHY, STATE_STALE)
            finally:
                await registry.aclose()
        finally:
            await endpoint.aclose()

    asyncio.run(_run())


def test_a_notification_for_an_unknown_subscription_is_dropped():
    """Routing is by subscription id; a straggler must not price another market."""
    from autopaper.paper_live.native_monitor import SharedSubscriptionStream

    async def _run():
        endpoint = await FakeWsEndpoint().start()
        try:
            registry = NativeMonitorRegistry()
            monitor = registry.acquire(_identity(), _monitor_config(endpoint.url))
            try:
                await asyncio.sleep(0.2)
                stream = monitor._stream
                assert stream is not None
                before = monitor.stats.updates
                stream._dispatch(_notification(999_999, _acct.bonding_curve_bytes()))
                assert monitor.stats.updates == before
            finally:
                await registry.aclose()
        finally:
            await endpoint.aclose()

    asyncio.run(_run())


# ===========================================================================
# 3. IDENTITY AND FRESHNESS ARE NOT RELAXED FOR THE PUSH PATH
# ===========================================================================


def test_a_pushed_price_for_another_market_is_rejected_by_the_engine():
    """
    THE SAME-MARKET GATE, APPLIED TO PUSHED DATA.

    A position pinned to one curve must refuse a price the push channel computed
    from a different market. This is the identity rule; relaxing it for the new
    source is exactly the defect the gate exists to prevent.
    """
    from autopaper.paper_live.engine import PaperLiveEngine

    position = SimpleNamespace(
        entry_pool_address=CURVE_ADDRESS, entry_venue_id="",
        exit_venue_id="", sol_usd_reference=122.36, mint=MINT,
        signal_id="sig", arm="L25_CURRENT", token_decimals=6,
        cached_identity=None)
    engine = make_engine(
        arms=parse_arms(["L25_CURRENT"]), quotes=FakeQuoteService(sell=sell_point()))
    engine._position_monitors[engine._exit_retry_key(position)] = SimpleNamespace(
        latest=lambda: SimpleNamespace(
            price_sol_per_token=0.000001, market_address="9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin",
            venue_id="PUMP_BONDING_CURVE:other", age_ms=12.0, usable=True,
            reason=""),
        state=STATE_HEALTHY)
    usd, provenance = engine._monitor_reference(position, None)
    assert usd is None
    assert provenance["reference_identity"] == "MISMATCH"


def test_an_identity_mismatch_stays_a_mismatch_when_configuring_the_url(tmp_path:
                                                                        Path):
    """
    Configuring the TRANSPORT must not touch the identity vocabulary.

    The five identity checks are the same objects with and without a websocket
    endpoint, so a market that is not the position's own can never be accepted
    just because the monitor is now connected.
    """
    from autopaper.same_market.identity import verify_identity
    from autopaper.same_market.models import CheckState
    from test_same_market_identity import MINT as IDENTITY_MINT
    from test_same_market_identity import _pair_obj, _quote

    cfg = _tmp_config(tmp_path, "NATIVE_MONITOR_WS_URL=wss://example.invalid\n")
    assert cfg.market.native_monitor_ws_url.startswith("wss://")

    same = verify_identity(expected_mint=IDENTITY_MINT, pair=_pair_obj(),
                           quote=_quote())
    assert same.pool_match is CheckState.PASS

    # The marked pair names a DIFFERENT pool than the executable quote does.
    # That is the mismatch the whole same-market layer exists to catch.
    other = verify_identity(expected_mint=IDENTITY_MINT,
                            pair=_pair_obj(address=PAIR),
                            quote=_quote())
    assert other.pool_match is CheckState.FAIL, (
        "a different market was accepted as the same market")


def test_unknown_identity_fails_closed_inside_the_engine():
    """``identity=UNKNOWN`` is a refusal, never a usable observation."""
    from dataclasses import replace
    from test_paper_live import native_sell_point

    engine = make_engine(
        arms=parse_arms(["L25_CURRENT"]), quotes=FakeQuoteService(sell=sell_point()))
    run(engine.on_signal(signal()))
    position = engine.accounts["L25_CURRENT"].position
    engine.quote_service = FakeQuoteService(sell=replace(sell_point(), identity="UNKNOWN"))
    usd, provenance = run(engine._native_trigger_reference(position, engine.observed_marks[MINT]))
    assert usd is None
    assert "identity" in (provenance["unavailable_reason"] or "")


def test_a_stale_pushed_price_is_not_reused_as_fresh():
    """
    The last price must never be recycled. ``latest()`` returns the point with
    ``usable=False``, so the caller falls back -- the number is still visible
    for diagnostics, which is the honest state.
    """
    config = MonitorConfig(ws_url="ws://unused", stale_after_ms=30.0)
    monitor = NativeMarketMonitor(_identity(), config=config)
    from autopaper.paper_live.native_monitor import SharedSubscriptionStream

    stream = SharedSubscriptionStream("ws://unused", config=config)
    monitor._stream = stream
    stream._monitors.append(monitor)
    stream._routes[7] = (monitor, CURVE_ADDRESS, ROLE_CURVE)
    monitor.stream_resubscribed({7: (CURVE_ADDRESS, ROLE_CURVE)})
    monitor.deliver(ROLE_CURVE, _acct.bonding_curve_bytes(), 10, CURVE_ADDRESS)
    monitor.stream_connected(False)                     # the socket dies
    point = monitor.latest()
    assert point is not None and not point.usable
    assert point.reason


# ===========================================================================
# 4. EXITS, RETRIES AND THE 123s BOUND
# ===========================================================================


def test_a_failed_exit_quote_does_not_stop_monitoring():
    """
    Monitoring must survive an exit that cannot be quoted. The position stays
    OPEN, the failure is scheduled, and the NEXT tick still runs.
    """
    from autopaper.exits import ExitPolicy

    engine = make_engine(
        arms=parse_arms(["L25_CURRENT"]),
        exits=ExitPolicy(ExitConfig()),
        quotes=FakeQuoteService(sell=replace(sell_point(),
                                            execution_native_reference=None)))
    run(engine.on_signal(signal()))
    position = engine.accounts["L25_CURRENT"].position
    assert position is not None
    run(engine.monitor_once())
    assert engine.accounts["L25_CURRENT"].has_position
    run(engine.monitor_once())
    assert engine.accounts["L25_CURRENT"].has_position
    assert engine._monitor_diag["MONITOR_TICKS"] >= 2


def test_the_exit_retry_budget_is_bounded():
    from autopaper.paper_live import engine as engine_module

    assert engine_module._EXIT_MAX_ATTEMPTS == 8, (
        "the exit retry budget moved; a failed exit must not be able to spend "
        "the capacity monitoring needs")
    assert engine_module._EXIT_BACKOFF_CAP_S > engine_module._EXIT_BACKOFF_BASE_S


def test_the_data_invalid_bound_is_unchanged():
    """
    The existing 123-second safety behaviour: continuously unevaluable triggers
    exit to a known state after the configured bound. Enabling the push channel
    must not lengthen it, and this is the guard against that.
    """
    assert ExitConfig().data_invalid_exit_s == 120.0
    engine = make_engine(
        arms=parse_arms(["L25_CURRENT"]), exits=ExitPolicy(ExitConfig()),
        quotes=FakeQuoteService(sell=replace(sell_point(),
                                            execution_native_reference=None)))
    run(engine.on_signal(signal()))
    position = engine.accounts["L25_CURRENT"].position
    run(engine.monitor_once())
    started = position.native_reference_unavailable_since
    assert started is not None, (
        "the fail-closed timer never started, so 'fail closed' would mean "
        "'hold forever'")
    assert time.time() - started < 60.0


def test_the_ws_disconnected_counter_separates_cause_from_symptom():
    """
    A run can only be diagnosed if a disconnect is distinguishable from a rate
    limit. The counter is a ROOT-CAUSE counter, so a tick that was both
    disconnected AND throttled increments both.
    """
    engine = make_engine(
        arms=parse_arms(["L25_CURRENT"]), quotes=FakeQuoteService(sell=sell_point()))
    run(engine.on_signal(signal()))
    position = engine.accounts["L25_CURRENT"].position
    provenance = {"reference_identity": "UNVERIFIED",
                  "unavailable_reason": "solana rpc rate limited"}
    engine._position_monitors[engine._exit_retry_key(position)] = SimpleNamespace(
        _stream=None, state=STATE_STALE, age_ms=lambda: -1.0)
    engine._classify_monitor_tick(position, None, provenance)
    assert engine._monitor_diag["MONITOR_WS_DISCONNECTED"] == 1
    assert engine._monitor_diag["MONITOR_RPC_ERROR"] == 1


def test_a_budget_refusal_is_counted_as_capacity_not_as_a_defect():
    """The counter that makes 'budget exhaustion' separable from an RPC fault."""
    engine = make_engine(
        arms=parse_arms(["L25_CURRENT"]), quotes=FakeQuoteService(sell=sell_point()))
    run(engine.on_signal(signal()))
    position = engine.accounts["L25_CURRENT"].position
    provenance = {"reference_identity": "UNVERIFIED",
                  "unavailable_reason": "solana rpc telemetry lane budget spent (60/m)"}
    engine._classify_monitor_tick(position, None, provenance)
    assert engine._monitor_diag["MONITOR_SKIPPED_BUDGET"] == 1
    assert position.monitor_skipped_budget_ticks == 1
    assert engine._monitor_diag["MONITOR_RPC_ERROR"] == 0


# ===========================================================================
# 5. NO EXECUTION CAPABILITY ANYWHERE ON THIS PATH
# ===========================================================================


def test_the_push_path_names_no_execution_capability():
    """
    A websocket that could order would be a second executor. Asserted by
    scanning the transport, the monitor and the probe for the method names that
    would make it one, the same way the existing audit test does.
    """
    import ast

    for path in ("src/autopaper/same_market/ws.py",
                 "src/autopaper/paper_live/native_monitor.py",
                 "scripts/native_monitor_ws_probe.py"):
        tree = ast.parse(Path(path).read_text())
        methods = {n.value for n in ast.walk(tree)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        for banned in ("sendTransaction", "sendRawTransaction",
                       "signTransaction", "simulateTransaction",
                       "requestAirdrop"):
            assert banned not in methods, f"{path} names {banned}"
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "solders" not in imported, f"{path} imports solders"
        banned_modules = {"autopaper.execution", "autopaper.engine"}
        assert not (imported & banned_modules), (
            f"{path} imports a trading module: {imported & banned_modules}")


def test_the_probe_is_read_only_by_construction():
    """
    The diagnostic probe is the only network-facing addition. It must not touch
    a position, the ledger or ``var/``, and it must not be able to loop.
    """
    source = Path("scripts/native_monitor_ws_probe.py").read_text()
    for banned in ("autopaper.engine", "autopaper.execution", "paper_state",
                   "paper_bridge", "open_position", "buy", "sell", "order"):
        assert banned not in source, f"the probe mentions {banned}"
    assert "while True" not in source.split("def probe")[0], (
        "the probe contains an unbounded loop")


# ===========================================================================
# 6. THE REQUEST BUDGET: what one monitoring tick actually costs
# ===========================================================================


class _JsonRpcClient:
    """
    An httpx-shaped transport double that counts JSON-RPC requests.

    Answering from a fixture table, so the reader under test takes the real code
    path and the REQUEST COUNT is the measurement rather than a mock's
    recollection of what should have happened.
    """

    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.requests: list[dict] = []

    async def post(self, url, json=None, headers=None, timeout=None):
        body = json or {}
        self.requests.append({"url": url, "body": body})
        method = body.get("method")
        key = f"{method}"
        if key not in self.responses:
            return SimpleNamespace(status_code=200, headers={},
                                   json=lambda: {"jsonrpc": "2.0",
                                                 "id": body.get("id"),
                                                 "result": None})
        result = self.responses[key]
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(status_code=200, headers={},
                               json=lambda: {"jsonrpc": "2.0",
                                             "id": body.get("id"),
                                             "result": result})

    async def aclose(self) -> None:
        return None


def _quote_service(accounts: dict[str, object]):
    """A real QuoteService over a counting transport, with no Jupiter leg."""
    from autopaper.paper_live.quotes import QuoteService
    from autopaper.same_market.rpc import SolanaReadOnlyRpc

    client = _JsonRpcClient(accounts)
    rpc = SolanaReadOnlyRpc(url="https://rpc.example", client=client,
                            cache_ttl_s=0.0, max_rpm=600)
    return QuoteService(rpc=rpc, jupiter=None, allow_jupiter_fallback=False), client


def _curve_position() -> SimpleNamespace:
    """The minimum a position must carry for the monitoring path to read it."""
    return SimpleNamespace(
        mint=MINT, symbol="T", signal_id="sig", arm="L25_CURRENT",
        position_id="sig:L25_CURRENT", token_decimals=6,
        tokens_raw=5_000_000_000, entry_pool_address=CURVE_ADDRESS,
        entry_venue_id=f"PUMP_BONDING_CURVE:{CURVE_ADDRESS}",
        entry_market_type="PUMP_BONDING_CURVE", exit_venue_id="",
        sol_usd_reference=122.36, cached_identity=None, opened_at=time.time(),
        entry_fill=SimpleNamespace(effective_price_usd=0.001, price_usd=0.001),
        strategy_profile="L25_CURRENT",
        exit_style=__import__("autopaper.models", fromlist=["ExitStyle"])
        .ExitStyle.FAST_FAILURE,
        native_reference_unavailable_since=None,
    )


def test_the_polling_recovery_path_costs_rpc_reads():
    """
    THE COST OF ONE RECOVERY TICK, MEASURED.

    This is the number the push channel exists to remove: without it the trigger
    reference is obtained by READING the venue, and those reads come out of the
    shared request budget the exit quote also needs.
    """
    from test_paper_live import run as run_coro

    curve = _acct.bonding_curve_bytes()
    quote_service, client = _quote_service({
        "getSlot": 452_000_000,
        "getMultipleAccounts": [
            {"data": [base64.b64encode(curve).decode(), "base64"],
             "owner": pd.PUMP_FUN_PROGRAM, "lamports": 1},
        ],
        "getTokenSupply": {"value": {"uiAmount": 1_000_000.0,
                                     "decimals": 6, "amount": "1000000000000000"}},
    })
    engine = make_engine(
        arms=parse_arms(["L25_CURRENT"]), quotes=quote_service, market=None)
    engine.native_monitor_enabled = False
    position = _curve_position()

    before = len(client.requests)
    usd, provenance = run_coro(engine._native_trigger_reference(
        position, SimpleNamespace(value_usd=0.001, available=True,
                                  pair_address=None)))
    spent = len(client.requests) - before
    assert spent >= 1, "the recovery path should have spent at least one read"
    assert provenance["trigger_reference_source"] == "EXECUTION_NATIVE_SAME_MARKET"


def test_the_push_channel_costs_zero_rpc_reads():
    """
    THE SAME TICK THROUGH THE PUSH CHANNEL costs zero requests.

    This is the whole argument for the configuration change. The pushed account
    update is priced LOCALLY, so the reads the recovery path spends are not
    spent at all -- that is the capacity the exit quote needs, and the thing the
    123-second data-invalid exits were being priced against.
    """

    async def _run():
        endpoint = await FakeWsEndpoint(push_interval_s=0.02).start()
        try:
            curve = _acct.bonding_curve_bytes()
            quote_service, client = _quote_service({
                "getSlot": 452_000_000,
                "getMultipleAccounts": [
                    {"data": [base64.b64encode(curve).decode(), "base64"],
                     "owner": pd.PUMP_FUN_PROGRAM, "lamports": 1},
                ],
            })
            engine = make_engine(
                arms=parse_arms(["L25_CURRENT"]), quotes=quote_service, market=None)
            engine.native_monitor_enabled = True
            engine.native_monitor_ws_url = endpoint.url
            engine.native_monitor_stale_after_ms = 5_000.0
            engine.native_monitor_open_timeout_s = 5.0
            position = _curve_position()

            monitor = engine._ensure_native_monitor(position, None)
            assert monitor is not None, (
                "a configured endpoint did not reach the monitor registry")
            point = await _wait_for_price(monitor)
            assert point is not None, "no price arrived from the push channel"

            before = len(client.requests)
            usd, provenance = engine._monitor_reference(position, None)
            spent = len(client.requests) - before
            assert spent == 0, (
                f"the push channel spent {spent} RPC reads; a pushed price must "
                f"cost nothing")
            assert usd is not None and usd > 0, (
                "a usable pushed price did not resolve into a reference")
            assert provenance["reference_identity"] == "PASS"
            assert provenance["monitor_source"] == SOURCE_PUSH
            await engine.native_monitors.aclose()
        finally:
            await endpoint.aclose()

    run(_run())


def test_the_monitor_diagnostics_name_the_path_and_the_budget():
    """
    "Why was market data unusable" must be answerable from the diagnostics
    block alone: which path the run was on, whether the push channel was ever
    connected, and how much of the shared request budget was left when it
    stopped being usable.
    """
    from autopaper.same_market.rpc import SolanaReadOnlyRpc

    quote_service, client = _quote_service({"getSlot": 1})
    engine = make_engine(
        arms=parse_arms(["L25_CURRENT"]), quotes=quote_service, market=None)
    engine.native_monitor_enabled = False
    diag = engine._monitor_diagnostics()
    assert diag["MONITOR_WS_DISCONNECTED"] == 0
    assert diag["monitoring_ws_disconnected_pct"] == 0.0
    assert diag["native_monitor_ws_url_configured"] is False, (
        "an unconfigured push channel must not be reported as configured")

    engine.native_monitor_enabled = True
    engine.native_monitor_ws_url = "wss://example.invalid"
    diag = engine._monitor_diagnostics()
    assert diag["native_monitor_ws_url_configured"] is True

    snapshot = engine._shared_budget_snapshot()
    assert set(snapshot) >= {
        "telemetry_lane_used_per_min", "telemetry_lane_ceiling_per_min",
        "telemetry_lane_remaining_per_min", "rpc_rejections", "blocked_for_s"}
    assert snapshot["telemetry_lane_remaining_per_min"] <= \
        snapshot["telemetry_lane_ceiling_per_min"]


def test_a_budget_snapshot_is_empty_without_a_limiter_rather_than_zero():
    """A zero remaining budget must not be invented for an engine with no RPC."""
    from tests.paper_bridge_fakes import bridge

    engine = bridge()
    assert engine._shared_budget_snapshot() == {}
