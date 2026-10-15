"""
A minimal, dependency-free Solana WebSocket transport.

WHY THIS EXISTS RATHER THAN A LIBRARY
-------------------------------------
``pyproject.toml`` declares exactly three runtime dependencies -- ``telethon``,
``httpx`` and ``solders`` -- and ``solders`` is keys/transaction codec only. Neither
``websockets`` nor ``solana-py`` is installed, and this engine's whole read
surface is the closed tuple
:data:`autopaper.same_market.rpc.READ_METHODS`. Adding a dependency to obtain a
push channel for an existing read is a larger change than the channel itself, so
the RFC 6455 subset Solana needs is implemented here:

  * the HTTP/1.1 ``Upgrade`` handshake,
  * text frames (Solana's JSON-RPC-over-WS is text),
  * ping/pong and close,
  * client-to-server masking, which RFC 6455 requires and servers enforce.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It is a transport, not a client. It does not know about subscriptions, RPC
methods, slots or rate limiting -- :mod:`autopaper.paper_live.native_monitor`
owns all of that. Nothing here reads chain state, and nothing here decides what a
quote is worth. If this module cannot be understood in isolation, the split is
wrong.

SAFETY
------
A websocket carries observations, never authority. No transaction is ever
signed, simulated or broadcast through this path, and :mod:`autopaper.safety`
still gates the REAL executor independently of anything connected here.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import ssl
import struct
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlsplit

#: The GUID RFC 6455 fixes for the ``Sec-WebSocket-Accept`` computation.
_HANDSHAKE_GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

#: Opcodes. Only the ones Solana actually sends are named.
_OP_CONTINUATION = 0x0
_OP_TEXT = 0x1
_OP_BINARY = 0x2
_OP_CLOSE = 0x8
_OP_PING = 0x9
_OP_PONG = 0xA

#: A control frame's payload is capped at 125 bytes including the header, so
#: anything longer is a protocol violation rather than a large message.
_MAX_CONTROL_PAYLOAD = 125

#: Guard against an unbounded continuation from a hostile or broken endpoint.
_MAX_MESSAGE_BYTES = 8 * 1024 * 1024


class WebSocketUnavailable(RuntimeError):
    """
    The websocket could not be established, or the stream failed.

    Distinct from a JSON-RPC ``error`` RESPONSE on purpose: this means there is
    no channel at all, which is a transport condition the monitor must recover
    from rather than a market condition it can price around.
    """


class WebSocketClosed(WebSocketUnavailable):
    """The peer closed the stream, cleanly or not."""


@dataclass(frozen=True)
class WsTarget:
    """A parsed ``wss://`` endpoint."""

    host: str
    port: int
    path: str
    #: Injected so a test can drive the stream without a real TLS endpoint.
    ssl_context: Optional[ssl.SSLContext] = field(default=None, compare=False)

    @property
    def host_header(self) -> str:
        default = 443
        return self.host if self.port == default else f"{self.host}:{self.port}"

    @property
    def request_target(self) -> str:
        return self.path or "/"


def parse_ws_url(url: str, *, ssl_context: Optional[ssl.SSLContext] = None) -> WsTarget:
    """
    Parse a ``wss://`` URL into a connectable target.

    ``ws://`` is accepted and mapped to an unencrypted socket so a local test
    endpoint works, but Solana endpoints are always ``wss://``.
    """
    parts = urlsplit(str(url).strip())
    scheme = (parts.scheme or "").lower()
    if scheme not in ("wss", "ws"):
        raise WebSocketUnavailable(
            f"unsupported websocket scheme {parts.scheme!r}; expected wss://")
    if not parts.hostname:
        raise WebSocketUnavailable(f"websocket URL {url!r} has no host")
    if scheme == "wss" and ssl_context is None:
        ssl_context = ssl.create_default_context()
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    return WsTarget(
        host=parts.hostname,
        port=int(parts.port or 443),
        path=path,
        ssl_context=ssl_context,
    )


def _accept_token(key: str) -> str:
    """``base64(sha1(key + GUID))`` -- the handshake's proof of understanding."""
    digest = hashlib.sha1(key.encode("ascii") + _HANDSHAKE_GUID).digest()
    return base64.b64encode(digest).decode("ascii")


def encode_frame(payload: bytes, *, opcode: int = _OP_TEXT) -> bytes:
    """
    A single, unfragmented, MASKED client frame.

    Masking is mandatory for client-to-server frames (RFC 6455 §5.3) and Solana's
    endpoint rejects unmasked ones, so this is not optional decoration.
    """
    if opcode in (_OP_CLOSE, _OP_PING, _OP_PONG) and \
            len(payload) > _MAX_CONTROL_PAYLOAD:
        raise ValueError("a control frame payload may not exceed 125 bytes")

    header = bytearray()
    header.append(0x80 | (opcode & 0x0F))          # FIN + opcode
    mask_bit = 0x80                                 # client frames are masked
    size = len(payload)
    if size < 126:
        header.append(mask_bit | size)
    elif size < (1 << 16):
        header.append(mask_bit | 126)
        header.extend(struct.pack("!H", size))
    else:
        header.append(mask_bit | 127)
        header.extend(struct.pack("!Q", size))

    mask = os.urandom(4)
    header.extend(mask)
    masked = bytearray(payload)
    for index in range(size):
        masked[index] ^= mask[index % 4]
    return bytes(header) + bytes(masked)


async def _read_exactly(reader: asyncio.StreamReader, count: int) -> bytes:
    """Read exactly ``count`` bytes, or raise. Partial reads are normal."""
    if count == 0:
        return b""
    try:
        return await reader.readexactly(count)
    except (asyncio.IncompleteReadError, ConnectionResetError) as exc:
        raise WebSocketClosed(f"stream ended mid-frame: {exc}") from exc


class WebSocketConnection:
    """
    One websocket connection: connect, send JSON, read JSON.

    Not reusable after close, and not thread-safe. A monitor owns one connection
    and reconnects by constructing another -- which is why
    :class:`~autopaper.paper_live.native_monitor.NativeMarketMonitor` can treat a
    failure as "build a new one" instead of trying to repair a half-dead socket.
    """

    def __init__(self, target: WsTarget, *, open_timeout_s: float = 10.0) -> None:
        self.target = target
        self.open_timeout_s = open_timeout_s
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._open = False
        #: Monotonic time of the last successful read. Interval arithmetic uses
        #: this, never wall time, so an NTP step cannot invent or hide a gap.
        self.last_read_monotonic: float = 0.0
        self.connected_at_monotonic: float = 0.0

    @property
    def is_open(self) -> bool:
        return self._open and self._writer is not None and \
            not self._writer.is_closing()

    async def connect(self) -> None:
        """Open the stream and complete the handshake, or raise."""
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(
                    self.target.host, self.target.port,
                    ssl=self.target.ssl_context,
                    server_hostname=self.target.host if self.target.ssl_context
                    else None,
                ),
                timeout=self.open_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise WebSocketUnavailable(
                f"websocket connect to {self.target.host}:{self.target.port} "
                f"timed out after {self.open_timeout_s:.1f}s") from exc
        except (OSError, ssl.SSLError) as exc:
            raise WebSocketUnavailable(
                f"websocket connect to {self.target.host}:{self.target.port} "
                f"failed: {type(exc).__name__}: {exc}") from exc

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {self.target.request_target} HTTP/1.1\r\n"
            f"Host: {self.target.host_header}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        assert self._writer is not None
        self._writer.write(request.encode("ascii"))
        await self._writer.drain()

        assert self._reader is not None
        try:
            status_line = await asyncio.wait_for(
                self._reader.readline(), timeout=self.open_timeout_s)
        except asyncio.TimeoutError as exc:
            await self.aclose()
            raise WebSocketUnavailable(
                "websocket handshake: no response line before the timeout") from exc
        if not status_line:
            await self.aclose()
            raise WebSocketClosed("websocket handshake: endpoint closed the TCP "
                                  "connection without a response")
        parts = status_line.decode("latin-1").strip().split(" ", 2)
        if len(parts) < 2 or parts[1] != "101":
            await self.aclose()
            raise WebSocketUnavailable(
                f"websocket handshake refused: {status_line.decode('latin-1').strip()!r}")

        accept: Optional[str] = None
        while True:
            line = await self._reader.readline()
            if not line or line in (b"\r\n", b"\n"):
                break
            name, _, value = line.decode("latin-1").partition(":")
            if name.strip().lower() == "sec-websocket-accept":
                accept = value.strip()
        if accept != _accept_token(key):
            await self.aclose()
            raise WebSocketUnavailable(
                "websocket handshake failed the Sec-WebSocket-Accept check, so "
                "this is not a conformant websocket endpoint")

        self._open = True
        self.last_read_monotonic = loop.time()
        self.connected_at_monotonic = self.last_read_monotonic
        del started

    async def send_text(self, payload: str) -> None:
        """Send one text frame."""
        if not self.is_open:
            raise WebSocketClosed("cannot send on a websocket that is not open")
        assert self._writer is not None
        try:
            self._writer.write(encode_frame(payload.encode("utf-8"), opcode=_OP_TEXT))
            await self._writer.drain()
        except (OSError, ConnectionResetError) as exc:
            self._open = False
            raise WebSocketClosed(f"websocket send failed: {exc}") from exc

    async def recv_text(self) -> str:
        """
        Read the next TEXT message, transparently handling ping, pong, close
        and fragmentation.

        A binary message is refused rather than coerced: Solana's JSON-RPC over
        websocket is text, and a binary frame means the endpoint is not the
        protocol this module was written against.
        """
        if self._reader is None:
            raise WebSocketClosed("cannot read from a websocket that was never opened")
        loop = asyncio.get_running_loop()
        chunks: list[bytes] = []
        total = 0
        message_opcode: Optional[int] = None

        while True:
            head = await _read_exactly(self._reader, 2)
            first, second = head[0], head[1]
            fin = bool(first & 0x80)
            opcode = first & 0x0F
            masked = bool(second & 0x80)
            size = second & 0x7F
            if size == 126:
                size = struct.unpack("!H", await _read_exactly(self._reader, 2))[0]
            elif size == 127:
                size = struct.unpack("!Q", await _read_exactly(self._reader, 8))[0]
            if size > _MAX_MESSAGE_BYTES:
                self._open = False
                raise WebSocketUnavailable(
                    f"websocket frame declares {size} bytes, above the "
                    f"{_MAX_MESSAGE_BYTES} byte ceiling")
            mask = await _read_exactly(self._reader, 4) if masked else b""
            payload = bytearray(await _read_exactly(self._reader, size))
            if masked:
                for index in range(size):
                    payload[index] ^= mask[index % 4]

            if opcode == _OP_CLOSE:
                self._open = False
                raise WebSocketClosed(
                    f"websocket closed by peer (code {int.from_bytes(payload[:2], 'big') if len(payload) >= 2 else 0})")
            if opcode == _OP_PING:
                await self._pong(bytes(payload))
                continue
            if opcode == _OP_PONG:
                continue

            if opcode in (_OP_TEXT, _OP_BINARY):
                if message_opcode is not None:
                    raise WebSocketUnavailable(
                        "a new message started before the previous one finished")
                message_opcode = opcode
            elif opcode != _OP_CONTINUATION:
                raise WebSocketUnavailable(
                    f"unknown websocket opcode {opcode:#x}")

            chunks.append(bytes(payload))
            total += len(payload)
            if total > _MAX_MESSAGE_BYTES:
                self._open = False
                raise WebSocketUnavailable(
                    f"websocket message exceeded {_MAX_MESSAGE_BYTES} bytes")
            if fin:
                self.last_read_monotonic = loop.time()
                if message_opcode == _OP_BINARY:
                    raise WebSocketUnavailable(
                        "websocket delivered a binary frame; Solana JSON-RPC over "
                        "websocket is text and this transport will not guess")
                try:
                    return b"".join(chunks).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise WebSocketUnavailable(
                        f"websocket text frame was not utf-8: {exc}") from exc

    async def _pong(self, payload: bytes) -> None:
        if self.is_open:
            assert self._writer is not None
            try:
                self._writer.write(
                    encode_frame(payload[:_MAX_CONTROL_PAYLOAD], opcode=_OP_PONG))
                await self._writer.drain()
            except (OSError, ConnectionResetError):
                self._open = False

    async def aclose(self) -> None:
        """Close the stream. Never raises: a failing close must not mask a
        real error the caller is already handling."""
        if self._writer is None:
            self._open = False
            return
        try:
            if self.is_open:
                self._writer.write(encode_frame(b"\x03\xe8", opcode=_OP_CLOSE))
                await self._writer.drain()
        except Exception:  # noqa: BLE001 - best effort on a dying socket
            pass
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except Exception:  # noqa: BLE001 - same
            pass
        self._open = False
        self._writer = None
        self._reader = None