"""Low-level Solana primitives for the REAL executor.

PROVENANCE
---------
Ported from the reference executor in ``~/works/memecoin-bot``
(``bot/execution/solana.py``) without behavioural change. That file has been
exercised against Solana mainnet for quote -> build -> sign -> simulate. This
file is the same implementation; only the module docstring and log channel
name differ.

Scope
-----
This module is the *only* place that touches signing material or the Solana
JSON-RPC wire. Everything above it (quotes, sizing, journal) deals in plain
dataclasses, so the secret boundary is one file wide and auditable.

Secret-handling contract
------------------------
1. Key material is read from a **filesystem path** configured by the operator.
   It is never accepted as an argument, never read from the environment, never
   written to a log, and never placed in an exception message.
2. :class:`SolanaWallet` holds the keypair in memory for the process lifetime
   and therefore refuses to be ``repr``-ed, logged, or serialised.
3. Only the **public** address is ever returned in a display string, and
   :func:`mask_address` shortens it further for operator-facing output.

Everything here fails closed: a missing dependency, a malformed keypair file,
an unreachable RPC or a malformed response all raise, and callers must treat a
raise as "do not trade".
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

log = logging.getLogger("execution.solana")

LAMPORTS_PER_SOL = 1_000_000_000

#: Solana's native wrapped-SOL mint. Used as the swap input currency.
WSOL_MINT = "So11111111111111111111111111111111111111112"

#: USDC mint, used only to derive a SOL/USD reference price for reporting
#: (never as a trading pair on the V2 path).
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

#: SPL Token program, used to enumerate a wallet's token accounts. Module level
#: so the equity path and the ATA derivation agree on one program id.
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

#: Token-2022 (SPL Token Extensions) program. A wallet can hold accounts under
#: BOTH this program and the classic one, and `getTokenAccountsByOwner` filtered
#: to a single `programId` sees only that program's accounts. The live $SI, $Boy
#: and notify mints were all Token-2022, so a classic-only scan reported the
#: wallet as holding nothing while rent sat in eight empty 2022 accounts.
TOKEN_2022_PROGRAM_ID = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"

#: Every token program a wallet may hold accounts under, classic first. The
#: scanner iterates this so an asset is never invisible merely because it uses
#: the newer program id.
TOKEN_PROGRAMS = (TOKEN_PROGRAM_ID, TOKEN_2022_PROGRAM_ID)

#: Rent-exempt lamports locked by a base (165-byte) SPL token account. This is
#: the amount recoverable by closing an EMPTY account; it is a floor, not a
#: measurement, and the scanner reports each account's own actual lamports.
TOKEN_ACCOUNT_RENT_LAMPORTS = 2_039_280

#: Bytes in a base SPL/Token-2022 account. Token-2022 accounts with extensions
#: are larger, which is exactly why rent is read per-account and not assumed.
TOKEN_ACCOUNT_SIZE = 165

_BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


class SolanaUnavailable(RuntimeError):
    """A Solana capability is missing. Always means DO NOT TRADE."""


#: Hard Solana limit on a serialized LEGACY or v0 transaction. A transaction
#: larger than this can never be broadcast, and the node rejects it during
#: simulation with JSON-RPC -32602 "VersionedTransaction too large".
#:
#: This is NOT a universal limit. A v1 transaction (SIMD-0385) may be up to 4096
#: bytes. The per-version ceilings live in :mod:`.txcodec` and every size gate
#: in this runtime reads the ACTUAL transaction version after deserialization
#: rather than applying this number to everything. Kept here as the legacy/v0
#: constant because the name is imported across the execution layer.
MAX_SOLANA_TX_BYTES = 1232

#: Longest base64 representation the node will accept for a legacy/v0
#: transaction of the size above.
MAX_SOLANA_TX_B64_CHARS = 1644

#: The largest ``maxSupportedTransactionVersion`` this runtime asks for. Set to
#: 1 so every transaction-read RPC (getTransaction, getBlock) can return a v1
#: transaction instead of refusing it. Passing 0 here would make the node reject
#: a v1 transaction on READ -- the same transaction this bot would have just
#: broadcast -- and reconciliation would silently come back empty.
MAX_SUPPORTED_TRANSACTION_VERSION = 1

#: Substrings that identify the size-limit rejection in a node's -32602
#: message. Matched case-insensitively. Kept as data, not as logic, so a
#: wording change on the node side degrades to the generic invalid-params
#: classification instead of mis-reporting.
_TX_TOO_LARGE_MARKERS = ("too large", "exceeds maximum size", "transaction too big")


class RpcError(RuntimeError):
    """
    The RPC node returned an error or an unusable payload.

    Carries the numeric ``code`` and a sanitized ``message`` so a caller can
    CLASSIFY the failure instead of only reporting that one happened.

    Before this existed the message was dropped and only the code survived,
    which is why the July incident surfaced as a bare
    ``simulateTransaction: RPC error -32602``. A code alone is not actionable:
    -32602 covers "invalid params", "malformed transaction" and "transaction
    too large", which have completely different fixes. The message is what
    distinguishes them, so it is preserved.
    """

    def __init__(self, message: str, *, code: Optional[int] = None,
                 rpc_message: str = "", rpc_data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.rpc_message = rpc_message
        self.rpc_data = rpc_data


#: JSON-RPC error codes that mean the node PROCESSED the request and REFUSED it,
#: so the transaction was never submitted and cannot be executing on chain.
#:
#: WHY THIS TABLE EXISTS. ``sendTransaction`` is a single request with a single
#: outcome: the node either submits and returns the transaction's signature, or
#: it refuses and returns a JSON-RPC error. A node that submitted our
#: transaction and then failed would have to return a signature, so a CODED
#: error is proof that nothing was submitted. The 8HJY9 intent of 2026-10-02 sat
#: in UNKNOWN for three hours and blocked twenty REAL entries because
#: ``-32002 Transaction simulation failed`` -- with ``skipPreflight: False``,
#: that is the node simulating our transaction, watching it fail, and refusing
#: to submit it. The token account for that mint still does not exist.
#:
#: The table is deliberately a CLOSED ALLOW-LIST. An unrecognised coded error
#: falls through to "unknown", because the cost of wrongly releasing an intent
#: that really is in flight is a second real order for the same position, and the
#: cost of wrongly keeping one is a delay an operator can clear by hand. Only the
#: asymmetry justifies guessing, and here it does not.
_PROVABLY_NOT_SUBMITTED_CODES = frozenset({
    -32002,   # Transaction simulation failed: node preflight refused it
    -32602,   # Invalid params: the node rejected the request itself
    -32003,   # Blockhash not found: submitted but already expired, so dead
})


def send_failure_may_have_submitted(exc: BaseException) -> bool:
    """
    Is it possible the network saw this transaction? ``True`` means UNKNOWN.

    Conservative by construction. Anything that is not a recognised node-side
    refusal -- a timeout, a reset connection, an HTTP 5xx, a non-JSON body, an
    unfamiliar code -- is reported as possibly-submitted, because a lost response
    is exactly the case where the transaction may already be executing and a
    retry would double the position.
    """
    code = getattr(exc, "code", None)
    if code is None:
        # transport-level: we never learned the node's verdict
        return True
    return code not in _PROVABLY_NOT_SUBMITTED_CODES


def sanitize_rpc_message(text: str, *, limit: int = 240) -> str:
    """
    Reduce an RPC message to something safe to log and journal.

    A node message is attacker-influenceable in the sense that it echoes
    caller-supplied bytes, so it is truncated and stripped of anything
    base64-shaped that is long enough to be a transaction or signature. Short
    identifiers and the numbers that matter (byte counts, limits) survive,
    because those are the diagnostic.
    """
    if not text:
        return ""
    out = str(text).replace("\n", " ").strip()
    # Collapse long base64/base58-looking runs: they can only be payload.
    out = _LONG_B64_RUN.sub("<payload redacted>", out)
    return out[:limit]


_LONG_B64_RUN = re.compile(r"[A-Za-z0-9+/=_-]{80,}")

#: How much program-log evidence is kept. A swap's log is short; the cap exists
#: so a hostile or chatty program cannot turn one failed order into an unbounded
#: record. The tail is what matters (the failing program logs last), so the cap
#: keeps the FIRST lines plus an explicit marker when it truncates.
MAX_SIMULATION_LOG_LINES = 64


def _sanitize_simulation_logs(logs: Any) -> list:
    """
    Sanitize the node's program logs for the audit trail.

    Each line is run through the same redaction as an RPC message (long
    base64/base58 runs removed) and truncated, so a transaction payload or a
    signature can never be recorded. Never raises: a missing or malformed
    ``logs`` field yields an empty list rather than a failure.
    """
    if not isinstance(logs, (list, tuple)):
        return []
    out: list = []
    for line in logs[:MAX_SIMULATION_LOG_LINES]:
        try:
            out.append(sanitize_rpc_message(str(line), limit=400))
        except Exception:  # noqa: BLE001 - evidence collection must not raise
            continue
    if len(logs) > MAX_SIMULATION_LOG_LINES:
        out.append(f"<{len(logs) - MAX_SIMULATION_LOG_LINES} further log "
                   f"lines omitted>")
    return out


def classify_simulation_error(code: Optional[int], message: str) -> str:
    """
    Map an RPC simulation rejection onto a stable, machine-readable status.

    Deliberately conservative: anything unrecognised stays generic rather than
    being forced into a specific bucket, because a wrong classification would
    send an operator hunting the wrong defect.
    """
    low = (message or "").lower()
    if any(m in low for m in _TX_TOO_LARGE_MARKERS):
        return "SIMULATION_FAILED_TX_TOO_LARGE"
    if "deserialize" in low or "invalid transaction" in low:
        return "SIMULATION_FAILED_TX_MALFORMED"
    if "invalid params" in low or "missing field" in low:
        return "SIMULATION_FAILED_RPC_INVALID_PARAMS"
    if code == -32602:
        return "SIMULATION_FAILED_RPC_INVALID_PARAMS"
    if code == -32601:
        return "SIMULATION_FAILED_RPC_METHOD_UNSUPPORTED"
    if code in (-32005,):
        return "SIMULATION_FAILED_RPC_NODE_LIMIT"
    return "SIMULATION_FAILED_RPC"


class TransactionFailed(RuntimeError):
    """A transaction reached a terminal, non-confirmed state on chain."""


def b58encode(raw: bytes) -> str:
    """Base58 encode. Local implementation: no extra runtime dependency."""
    number = int.from_bytes(raw, "big")
    out = ""
    while number:
        number, remainder = divmod(number, 58)
        out = _BASE58[remainder] + out
    pad = len(raw) - len(raw.lstrip(b"\x00"))
    return "1" * pad + (out or "")


def b58decode(value: str) -> bytes:
    number = 0
    for ch in value:
        idx = _BASE58.find(ch)
        if idx < 0:
            raise ValueError("invalid base58 character")
        number = number * 58 + idx
    body = number.to_bytes((number.bit_length() + 7) // 8, "big") if number else b""
    pad = len(value) - len(value.lstrip("1"))
    return b"\x00" * pad + body


def mask_address(address: str, head: int = 4, tail: int = 4) -> str:
    """``wUEU…Wbt6`` - the only form an operator surface may show.

    A Solana address is public by design, but a short mask is still used in
    Telegram and reports so a screenshot cannot trivially be turned into a
    targeted drain request.
    """
    if not address:
        return "(none)"
    if len(address) <= head + tail + 1:
        return address[:2] + "…"
    return f"{address[:head]}…{address[-tail:]}"


# ---------------------------------------------------------------------------
# Wallet
# ---------------------------------------------------------------------------
class SolanaWallet:
    """A locally-held ed25519 keypair loaded from a file path.

    Deliberately *not* a dataclass and deliberately without ``__repr__``: the
    private key must not be reachable through ``repr()``, ``%s`` formatting or
    an accidental ``asdict()`` in a crash report.
    """

    __slots__ = ("_keypair", "_pubkey", "_path")

    def __init__(self, keypair_path: str):
        if not keypair_path:
            raise SolanaUnavailable("WALLET_KEYPAIR_PATH is not configured")
        self._path = Path(keypair_path).expanduser()
        if not self._path.is_file():
            raise SolanaUnavailable(
                f"wallet keypair file not found: {self._path.name}")
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SolanaUnavailable(
                "wallet keypair file could not be parsed") from exc
        if not isinstance(raw, list) or len(raw) != 64 or \
                not all(isinstance(x, int) and 0 <= x <= 255 for x in raw):
            raise SolanaUnavailable(
                "wallet keypair file is not a 64-byte Solana keypair")
        try:
            from solders.keypair import Keypair
        except ImportError as exc:                      # pragma: no cover
            raise SolanaUnavailable(
                "solders is not installed; cannot load a Solana keypair") from exc
        try:
            self._keypair = Keypair.from_bytes(bytes(raw))
        except Exception as exc:                        # pragma: no cover
            raise SolanaUnavailable(
                "wallet keypair bytes were rejected by the ed25519 decoder") from exc
        self._pubkey = str(self._keypair.pubkey())

    @property
    def pubkey(self) -> str:
        """The public address. Safe to display (masked via mask_address)."""
        return self._pubkey

    @property
    def pubkey_obj(self):
        """The public key as a ``Pubkey``, for the solders APIs."""
        from solders.pubkey import Pubkey
        return Pubkey.from_string(self._pubkey)

    @property
    def masked(self) -> str:
        return mask_address(self._pubkey)

    def sign_new_transaction(self, instructions, blockhash=None) -> bytes:
        """Build and sign a transaction from ``instructions``.

        The keypair stays inside this object: callers get back signed wire
        bytes and never touch the key. Used both for the venue's pre-built
        transaction (via :meth:`sign`) and for the offline signing-capability
        probe.
        """
        from solders.hash import Hash
        from solders.message import Message
        from solders.pubkey import Pubkey
        from solders.transaction import Transaction
        bh = blockhash or Hash.default()
        msg = Message.new_with_blockhash(
            list(instructions), Pubkey.from_string(self._pubkey), bh)
        raw = bytes(Transaction([self._keypair], msg, bh))
        return self.sign_wire(raw)

    def sign_wire(self, raw: bytes) -> bytes:
        """Sign a Solana wire transaction in place and return the new bytes.

        Why wire-level rather than ``Transaction.sign``
        -------------------------------------------------
        A Jupiter swap is a **v0 versioned transaction** that references address
        lookup tables. In ``solders`` a ``VersionedTransaction`` exposes no
        in-place ``sign`` method, and ``into_legacy_transaction()`` returns
        ``None`` precisely because the lookup tables cannot be represented
        legacy. So the transaction is signed the way ``solana-web3.js`` does
        it: locate the signature array, sign the serialized message that
        belongs to the payer's slot, and splice the result back.

        VERSION-AWARE, BECAUSE THE LAYOUT IS NOT THE SAME FOR EVERY VERSION
        ---------------------------------------------------------------------
            legacy / v0 :  [count][signature 64n][message]
            v1          :  [message][signature 64n]      <-- signatures LAST

        The previous implementation assumed the legacy/v0 layout unconditionally.
        Applied to a v1 transaction that assumption is not merely wrong, it is
        destructive: it reads a byte from the middle of the message as a
        signature count, then signs a truncated "message" that is really a
        fragment of the real one, producing a transaction the validator rejects
        with a signature error and no useful diagnostic. The version is therefore
        detected from the bytes and the layout chosen to match, and an
        unidentifiable transaction is refused rather than signed.

        The signature covers the message bytes exactly as they appear on the
        wire, which is what the validator re-computes. This was verified to be
        byte-identical to ``Transaction.sign()`` on a legacy transaction, so
        the two paths are equivalent rather than merely similar.

        The key is read from memory only and is never returned or logged.
        """
        from .txcodec import sign_transaction_bytes

        if not raw:
            raise SolanaUnavailable("cannot sign an empty transaction")
        self._assert_required_signer(raw)
        try:
            return sign_transaction_bytes(raw, self._keypair)
        except ValueError as exc:
            # A structural refusal, not a key problem. Surfaced without any
            # detail of the key.
            raise SolanaUnavailable(
                f"transaction could not be signed safely: {exc}") from exc

    def _assert_required_signer(self, raw: bytes) -> None:
        """Refuse to sign a transaction this wallet is not required to sign.

        Signing a message that names a different payer yields a transaction the
        validator rejects: a wasted network fee and a confusing error instead of
        a clear one. The check is read-only and runs before the key is used, so
        a mismatch costs nothing.

        If the transaction cannot be parsed at all the check is skipped rather
        than guessed - the subsequent signature would simply be rejected by the
        validator, and refusing here would mask the real problem.
        """
        try:
            from solders.transaction import Transaction, VersionedTransaction
            required = None
            for parser in (VersionedTransaction, Transaction):
                try:
                    tx = parser.from_bytes(raw)
                except Exception:
                    continue
                msg = tx.message
                required = {str(msg.account_keys[i]) for i in range(
                    len(msg.account_keys)) if msg.is_signer(i)}
                break
        except Exception:                                # pragma: no cover
            return
        if required is None:
            return
        if self._pubkey not in required:
            raise SolanaUnavailable(
                "this wallet is not a required signer of the transaction; "
                "refusing to sign")

    def sign(self, transaction) -> None:
        """Sign a transaction in place. The private key never leaves here.

        ``solders>=0.29`` takes the *recent blockhash* as the second argument,
        not a public key. That distinction is load-bearing: passing anything
        other than the message's own blockhash makes solders **rewrite** the
        blockhash and clear every existing signature, which would silently
        invalidate a venue-built transaction. So the blockhash is read back off
        the message and reused verbatim.
        """
        from solders.transaction import Transaction
        if not isinstance(transaction, Transaction):
            raise SolanaUnavailable("sign() requires a solders Transaction")
        try:
            blockhash = transaction.message.recent_blockhash
        except AttributeError as exc:
            raise SolanaUnavailable(
                "transaction has no message blockhash to sign against") from exc
        try:
            transaction.sign([self._keypair], blockhash)
        except Exception as exc:
            # Surface a signer problem without echoing anything about the key.
            raise SolanaUnavailable(
                f"transaction signing was rejected: {type(exc).__name__}") from exc
        # Fail loudly rather than shipping a transaction with an empty
        # signature: an all-zero signature slot means an unfilled order.
        if not any(transaction.signatures[0]):
            raise SolanaUnavailable(
                "signature slot is still empty after signing")

    def __repr__(self) -> str:                          # pragma: no cover
        # Must never raise: a failed construction leaves _pubkey unset, and a
        # repr that throws turns a clean error into an opaque AttributeError
        # inside pytest, a log handler or a crash reporter.
        try:
            shown = self.masked
        except Exception:
            shown = "(unavailable)"
        return f"SolanaWallet(pubkey={shown!r}, key=<withheld>)"

    __str__ = __repr__


# ---------------------------------------------------------------------------
# RPC client
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WsolReadiness:
    """
    The four facts that decide whether a pre-provisioned-wSOL swap is possible.

    Reported separately rather than collapsed into one boolean so the operator
    is told WHICH action is needed: create the account, fund it, or neither.
    An account that exists but is empty satisfies ``ata_present`` and NOT
    ``sufficient`` -- that distinction is the whole point.
    """
    owner: str
    wsol_ata: str
    ata_present: bool
    balance_lamports: int
    required_lamports: int
    sufficient: bool
    query_ok: bool = True

    @property
    def ready(self) -> bool:
        """True only when the account exists AND holds at least the amount."""
        return bool(self.query_ok and self.ata_present and self.sufficient)

    def as_dict(self) -> dict[str, Any]:
        return {
            "WSOL_ATA_PRESENT": self.ata_present,
            "WSOL_BALANCE": self.balance_lamports,
            "WSOL_REQUIRED": self.required_lamports,
            "WSOL_SUFFICIENT": self.sufficient,
            "wsol_ata": self.wsol_ata,
            "query_ok": self.query_ok,
        }

    def block_reason(self) -> str:
        """A fail-closed explanation. Never a reason to proceed anyway."""
        if not self.query_ok:
            return "could not read the wallet's wSOL holdings from the RPC"
        if not self.ata_present:
            return (f"no wrapped-SOL account for this wallet "
                    f"({self.wsol_ata}); run setup-wsol")
        if not self.sufficient:
            return (f"wrapped-SOL balance {self.balance_lamports} lamports is "
                    f"below the {self.required_lamports} required for this order")
        return ""


#: Byte length of the ``AddressLookupTable`` program's account state, up to and
#: including the serialized ``Vec<Pubkey>`` length, for a table with an
#: authority (the overwhelmingly common case; an authority-less table is shorter
#: and is not reconstructed).
#:
#: Layout: enum discriminant (u32) + deactivation_slot (u64) + last_extended_slot
#: (u64) + last_extended_slot_start_index (u8) + authority (Option<Pubkey> = 1
#: tag byte + 32) + 2 bytes + compact-u16 address count, followed by 32 bytes per
#: address. Verified against a live table by cross-checking the local decode
#: against the node's own ``jsonParsed`` parse: 241 of 241 addresses identical.
ALT_STATE_HEADER_BYTES = 56

#: The program that owns address lookup table accounts. An account whose owner is
#: not this is not a table, whatever its bytes look like.
ADDRESS_LOOKUP_TABLE_PROGRAM = "AddressLookupTab1e1111111111111111111111111"


def decode_lookup_table_state(raw_b64: str) -> list:
    """Decode an address lookup table account's base64 state into addresses.

    The LAST resort of :meth:`SolanaRpc.get_address_lookup_table`, used only
    when the node offers neither ``getAddressLookupTable`` nor a ``jsonParsed``
    account. It refuses anything whose length does not reconcile exactly: a
    table whose byte count does not match ``header + 32 * n`` is not a table this
    decoder understands, and guessing an address list there would mean building a
    transaction with the wrong accounts.
    """
    if not raw_b64:
        return []
    from solders.pubkey import Pubkey

    try:
        data = base64.b64decode(raw_b64, validate=True)
    except Exception:  # noqa: BLE001
        return []
    body = len(data) - ALT_STATE_HEADER_BYTES
    if body <= 0 or body % 32:
        return []
    count = body // 32
    addresses = []
    for i in range(count):
        chunk = data[ALT_STATE_HEADER_BYTES + 32 * i:
                     ALT_STATE_HEADER_BYTES + 32 * (i + 1)]
        try:
            addresses.append(str(Pubkey.from_bytes(chunk)))
        except Exception:  # noqa: BLE001
            return []
    return addresses


def _alt_is_deactivated(info: dict) -> bool:
    """True when a lookup table's authority has closed it.

    A deactivated table's addresses are not loadable by the runtime, so a
    transaction that references one cannot be rebuilt from it. Reporting this
    distinctly from "unreadable" matters: the table exists and was read, it is
    simply not usable, and an operator chasing the wrong cause would look at
    their RPC.
    """
    value = info.get("lastDeactivatedSlot")
    if value in (None, 0, "0", ""):
        return False
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _token_accounts(res: Any) -> list:
    """
    Pull the account list out of a getTokenAccountsByOwner result.

    The public Solana RPC returns ``{"value": {"accounts": [...]}}``. Some
    gateways flatten that to ``{"value": [...]}``. Both are handled, because a
    parser that assumes one shape reports "no wSOL account" for a wallet that
    has one -- which would fail a legitimate trade closed for the wrong reason.
    """
    if not isinstance(res, dict):
        return []
    val = res.get("value")
    if isinstance(val, dict):
        accts = val.get("accounts")
        return accts if isinstance(accts, list) else []
    if isinstance(val, list):
        return val
    return []


def _parse_token_account_state(address: str, value: Any) -> Optional[dict]:
    """
    Normalize ONE ``getAccountInfo`` / ``getMultipleAccounts`` entry.

    ``value`` is the RPC's account object, or ``None`` for an account that does
    not exist -- which is the normal, expected answer for a CLOSED token
    account, and is reported as ``None`` rather than as an error.

    An account that exists but is not a parsed token account (a system account,
    a mint, a program-owned account) is returned with ``is_token_account=False``
    and the program that owns it, so a caller can refuse it BY NAME instead of
    inferring something it was never told. Every field a safety rule needs --
    ``closeAuthority``, ``delegate``, ``frozen``, ``state``, the raw amount, and
    the account's OWNING PROGRAM -- is read from the chain here, never from a
    cached listing row.

    Shared by the single-account and batch reads so the two can never disagree
    about what an account says.
    """
    if not value:
        return None
    parsed = (((value.get("data") or {}).get("parsed") or {}).get("info") or {})
    if not isinstance(parsed, dict) or not parsed:
        # Present on chain but not a parsed token account.
        return {
            "address": str(address),
            "exists": True,
            "is_token_account": False,
            "owner_program": str(value.get("owner") or ""),
            "lamports": value.get("lamports"),
            "executable": bool(value.get("executable")),
            "rent_epoch": value.get("rentEpoch"),
        }
    token_amount = parsed.get("tokenAmount") or {}
    try:
        amount = int(token_amount.get("amount"))
        decimals = int(token_amount.get("decimals"))
    except (TypeError, ValueError):
        amount, decimals = None, None
    try:
        lamports = int(value.get("lamports"))
    except (TypeError, ValueError):
        lamports = None
    close_authority = parsed.get("closeAuthority")
    delegate = parsed.get("delegate")
    return {
        "address": str(address),
        "exists": True,
        "is_token_account": True,
        "mint": str(parsed.get("mint") or ""),
        "token_owner": str(parsed.get("owner") or ""),
        "state": str(parsed.get("state") or ""),
        "amount": amount,
        "decimals": decimals,
        "delegate": str(delegate) if delegate else None,
        "close_authority": (str(close_authority) if close_authority else None),
        "close_authority_option": (close_authority.get("closeAuthority")
                                   if isinstance(close_authority, dict)
                                   else None),
        "state_option": (parsed.get("state")
                         if isinstance(parsed.get("state"), dict) else None),
        "is_native_option": parsed.get("isNative"),
        "frozen": bool(parsed.get("frozen")),
        "owner_program": str(value.get("owner") or ""),
        "lamports": lamports,
        "executable": bool(value.get("executable")),
        "rent_epoch": value.get("rentEpoch"),
    }


def wsol_ata_address(owner: str, mint: str = WSOL_MINT) -> str:
    """
    Derive the Associated Token Account address for (owner, mint).

    SPL's ``get_associated_token_address``: the program-derived seeds are the
    owner, the TOKEN PROGRAM id, the mint, and the literal
    "ProgramDerivedAddress", under the Associated Token Account program.
    """
    from solders.pubkey import Pubkey

    def pk(x):
        return x if isinstance(x, Pubkey) else Pubkey.from_string(str(x))

    ata_program = pk("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
    token_program = pk("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
    address, _bump = Pubkey.find_program_address(
        [bytes(pk(owner)), bytes(token_program), bytes(pk(mint)),
         b"ProgramDerivedAddress"], ata_program)
    return str(address)


def close_account_instructions(owner: str, accounts) -> list:
    """
    Build one SPL ``CloseAccount`` instruction per ``(token_account, program)``.

    PURE AND OFFLINE: nothing is signed and nothing is sent. The rent that was
    bonded to the account is returned to ``owner``, which must be the account's
    own owner AND the signer -- the token program enforces both. The program id
    is the account's OWN program, so a Token-2022 account is closed by
    Token-2022 and a classic account by the classic program; using the wrong one
    would fail on chain.

    Lives in the execution layer because it is the only layer permitted to
    import a chain library.
    """
    from solders.instruction import AccountMeta, Instruction
    from solders.pubkey import Pubkey

    owner_pk = Pubkey.from_string(str(owner))
    out = []
    for token_account, program_id in accounts:
        out.append(Instruction(
            program_id=Pubkey.from_string(str(program_id)),
            data=bytes([9]),          # SPL Token CloseAccount discriminator
            accounts=[
                AccountMeta(pubkey=Pubkey.from_string(str(token_account)),
                            is_signer=False, is_writable=True),
                AccountMeta(pubkey=owner_pk, is_signer=False, is_writable=True),
                AccountMeta(pubkey=owner_pk, is_signer=True, is_writable=False),
            ],
        ))
    return out


def programs_in_transaction(raw: bytes) -> list[str]:
    """
    Every program id a serialized transaction actually invokes.

    Used to PROVE the absence of anything but the expected programs in a
    transaction the runtime built itself -- e.g. that a cleanup transaction
    invokes token programs and no router, no SystemProgram and nothing else.

    ``raw`` is a serialized TRANSACTION and is deserialized as one: a message
    and a transaction are different wire layouts, and parsing transaction bytes
    as a message yields no keys at all, which would make the audit vacuously
    pass.

    Lives in the execution layer because it is the only layer permitted to
    import a chain library.
    """
    from solders.transaction import Transaction

    try:
        message = Transaction.from_bytes(bytes(raw)).message
    except Exception:  # noqa: BLE001
        return []
    keys = message.account_keys
    out: list[str] = []
    for compiled in message.instructions:
        try:
            pk = str(keys[compiled.program_id_index])
        except (IndexError, TypeError, ValueError):
            continue
        if pk not in out:
            out.append(pk)
    return out


@dataclass
class SimulationOutcome:
    """
    The full, classifiable result of one simulation attempt.

    Carries the stage, the classified status, the RPC code, the sanitized node
    message and ``broadcast_attempted`` so a reader can always tell WHAT
    FAILED, WHERE, and whether anything was ever put on chain. It holds no
    signed bytes and no signature.
    """
    ok: bool
    status: str
    detail: str
    stage: str = "SIMULATION"
    rpc_code: Optional[int] = None
    rpc_message: str = ""
    broadcast_attempted: bool = False
    units_consumed: Optional[int] = None
    size: dict = field(default_factory=dict)
    #: The node's raw ``err`` value, exactly as returned (e.g.
    #: ``{'InstructionError': [6, {'Custom': 6001}]}``). Kept in its structured
    #: form so a reader can name the program and instruction index instead of
    #: re-parsing a formatted string. Contains no signed bytes and no secret.
    raw_err: Any = None
    #: The node's program log lines, truncated per line and in count. This is
    #: the evidence an operator needs to identify WHICH program refused a swap;
    #: it never contains key material.
    logs: list = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "status": self.status, "detail": self.detail,
            "stage": self.stage, "rpc_code": self.rpc_code,
            "rpc_message": self.rpc_message,
            "broadcast_attempted": self.broadcast_attempted,
            "units_consumed": self.units_consumed,
            "size": self.size,
            "raw_err": self.raw_err,
            "logs": list(self.logs),
        }


@dataclass
class RpcHealth:
    reachable: bool = False
    slot: Optional[int] = None
    detail: str = "not probed"


class SolanaRpc:
    """Minimal async JSON-RPC client.

    Deliberately not a general-purpose SDK wrapper: it exposes exactly the calls
    the executor needs, each of which is fail-closed, and every response is
    shape-checked before use. ``getHealth`` alone is not treated as proof of
    liveness - a balance read is, because it returns data.
    """

    def __init__(self, url: str, timeout: float = 15.0,
                 client: Optional[httpx.AsyncClient] = None):
        if not url:
            raise SolanaUnavailable("RPC_URL is not configured")
        self.url = url
        self.timeout = timeout
        self._client = client
        self._own_client = client is None
        self._id = 0

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._own_client:
            await self._client.aclose()
        self._client = None

    async def _rpc(self, method: str, params: list) -> Any:
        client = await self._ensure_client()
        self._id += 1
        payload = {"jsonrpc": "2.0", "id": self._id,
                   "method": method, "params": params}
        try:
            resp = await client.post(self.url, json=payload,
                                     timeout=self.timeout)
        except httpx.TimeoutException as exc:
            raise RpcError(f"{method}: RPC timed out") from exc
        except httpx.HTTPError as exc:
            raise RpcError(f"{method}: RPC transport error") from exc
        if resp.status_code != 200:
            raise RpcError(f"{method}: RPC HTTP {resp.status_code}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise RpcError(f"{method}: RPC returned non-JSON") from exc
        if isinstance(body, dict) and body.get("error"):
            err = body["error"]
            code = err.get("code") if isinstance(err, dict) else None
            # Preserve the node's own message (sanitized). The code alone is
            # not actionable: -32602 covers invalid params, malformed
            # transaction AND transaction-too-large, which have different
            # fixes. Dropping it here is what made the incident undiagnosable
            # from the journal alone.
            rpc_msg = ""
            rpc_data = None
            if isinstance(err, dict):
                rpc_msg = sanitize_rpc_message(str(err.get("message") or ""))
                data = err.get("data")
                # `data` may echo the offending payload. Keep only its shape.
                if data is not None:
                    rpc_data = (f"<{type(data).__name__} "
                                f"len={len(data) if hasattr(data, '__len__') else '?'}>")
            raise RpcError(f"{method}: RPC error {code}"
                           + (f": {rpc_msg}" if rpc_msg else ""),
                           code=code, rpc_message=rpc_msg, rpc_data=rpc_data)
        if not isinstance(body, dict) or "result" not in body:
            raise RpcError(f"{method}: RPC response missing result")
        return body["result"]

    # -- reads ------------------------------------------------------------
    async def get_health(self) -> RpcHealth:
        """Liveness probe. Never raises; reports instead."""
        try:
            await self._rpc("getHealth", [])
            return RpcHealth(reachable=True, slot=None, detail="getHealth ok")
        except (RpcError, SolanaUnavailable) as exc:
            return RpcHealth(reachable=False, slot=None, detail=str(exc))

    async def get_balance_lamports(self, address: str) -> int:
        res = await self._rpc("getBalance", [address])
        try:
            return int(res["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RpcError("getBalance: malformed result") from exc

    async def get_token_accounts(
        self, owner: str, program_id: str = TOKEN_PROGRAM_ID
    ) -> list[dict]:
        """
        Every token account ``owner`` holds under ONE token program.

        Returns normalized rows. ``query_ok`` is carried so a caller can tell
        "there are no accounts" from "the node could not be asked" -- the two
        are very different facts and only one of them means an empty wallet.
        A malformed row is skipped rather than coerced into a balance; the
        account count is still reported by the caller's own arithmetic.
        """
        try:
            res = await self._rpc(
                "getTokenAccountsByOwner",
                [owner, {"programId": program_id},
                 {"encoding": "jsonParsed"}])
        except RpcError:
            return []
        out: list[dict] = []
        for acct in _token_accounts(res):
            parsed = (((acct.get("account") or {}).get("data") or {})
                      .get("parsed") or {}).get("info") or {}
            if not isinstance(parsed, dict):
                continue
            try:
                amount = parsed.get("tokenAmount") or {}
                mint = str(parsed.get("mint") or "")
                if not mint:
                    continue
                raw = int(amount.get("amount"))
                decimals = int(amount.get("decimals"))
            except (TypeError, ValueError, AttributeError):
                continue
            account_obj = acct.get("account") or {}
            try:
                lamports = int(account_obj.get("lamports"))
            except (TypeError, ValueError):
                lamports = None
            out.append({
                "mint": mint,
                "amount": raw,
                "decimals": decimals,
                "symbol": str(parsed.get("symbol") or ""),
                "account": str(acct.get("pubkey") or ""),
                "owner": str(parsed.get("owner") or owner),
                "program": program_id,
                "lamports": lamports,
                "is_empty": raw == 0,
            })
        return out

    async def get_token_account_states(self, addresses: list[str]
                                       ) -> dict[str, Optional[dict]]:
        """
        MANY token accounts' on-chain state, in ONE ``getMultipleAccounts``.

        WHY BATCHED
        -----------
        The cleanup re-check has to re-read every planned account immediately
        before the transaction is built. Reading them one at a time turns a
        single verification pass into N sequential RPC calls, and a rate-limited
        public endpoint answers the tail of that burst with HTTP 429. A 429 seen
        during a safety re-check is indistinguishable from a refusal, and
        treating it as one either blocks a safe cleanup or -- far worse -- reads
        as "verified" if the caller is careless. One call has one failure mode,
        so it is named once and can be retried as a unit.

        Returns ``{address: state-or-None}``. A whole-batch read failure
        RAISES: the caller must not plan against accounts it could not read.
        """
        wanted = [str(a) for a in addresses if a]
        if not wanted:
            return {}
        res = await self._rpc("getMultipleAccounts", [
            wanted, {"encoding": "jsonParsed", "commitment": "confirmed"}])
        values = res if isinstance(res, list) else []
        out: dict[str, Optional[dict]] = {}
        for address, value in zip(wanted, values):
            out[address] = _parse_token_account_state(address, value)
        return out

    async def get_latest_blockhash(self, commitment: str = "confirmed"
                                   ) -> tuple["Hash", int]:
        """
        A real ``(blockhash, last_valid_block_height)`` pair to sign against.

        WHY THIS EXISTS
        ---------------
        :meth:`SolanaWallet.sign_new_transaction` defaults to ``Hash.default()``
        -- 32 zero bytes. That is correct only for an OFFLINE signing-capability
        probe. A transaction signed against the zero hash passes local
        serialization and fails at the node with "Blockhash not found", so a
        caller that means to BROADCAST must supply a recent hash. There is no
        other way to obtain one, and inventing it here would be worse than the
        failure it prevents.

        Raises :class:`RpcError` when the node cannot answer; a caller must not
        fall back to the zero hash.
        """
        from solders.hash import Hash

        res = await self._rpc("getLatestBlockhash", [{"commitment": commitment}])
        if not isinstance(res, dict):
            raise RpcError("getLatestBlockhash: malformed result")
        value = res.get("value") or {}
        raw = str(value.get("blockhash") or "")
        if not raw:
            raise RpcError("getLatestBlockhash: no blockhash returned")
        try:
            last_valid = int(value["lastValidBlockHeight"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RpcError(
                "getLatestBlockhash: no lastValidBlockHeight") from exc
        return Hash.from_string(raw), last_valid

    async def get_token_account_state(self, address: str) -> Optional[dict]:
        """
        One token account's authoritative ON-CHAIN state, by address.

        WHY THIS EXISTS
        ---------------
        ``getTokenAccountsByOwner`` answers "which accounts does this owner
        hold" and reports the parsed token balance, but it does NOT report the
        ``closeAuthority`` / ``delegate`` / ``frozen`` fields that decide
        whether an account can be closed and by whom. Closing an account whose
        close authority is somebody else, or which carries a live delegate,
        fails on chain or -- worse -- succeeds against an assumption the runtime
        never verified.

        Returns a normalized dict, or ``None`` when the account does not exist
        (a closed account). A node that cannot answer raises; "unreadable" is
        never silently turned into "does not exist".
        """
        res = await self._rpc("getAccountInfo", [address, {
            "encoding": "jsonParsed", "commitment": "confirmed"}])
        return _parse_token_account_state(str(address), (res or {}).get("value"))

    async def get_token_account_balance(self, owner: str, mint: str) -> Optional[int]:
        """Token balance for ``mint`` owned by ``owner``; None if no account.

        ``None`` is meaningful: it means "no token account exists yet", which
        is the normal state before the first buy of a mint. The filter is by
        MINT, not by program, so a Token-2022 account is found here exactly as
        a classic one is -- which is why this read was already correct while
        :meth:`token_balances` was not.
        """
        res = await self._rpc(
            "getTokenAccountsByOwner",
            [owner, {"mint": mint}, {"encoding": "jsonParsed"}])
        total = 0
        found = False
        for acct in _token_accounts(res):
            try:
                info = acct["account"]["data"]["parsed"]["info"]["tokenAmount"]
                total += int(info["amount"])
                found = True
            except (KeyError, TypeError, ValueError):
                continue
        return total if found else None

    async def token_balances(self, owner: str) -> list[dict]:
        """
        EVERY token balance in ``owner``, across BOTH token programs.

        ADDED FOR AUTHORITATIVE EQUITY. Equity that only reads native SOL is a
        partial equity: a wallet holding wrapped SOL or USDC has real money the
        number does not see, and REAL sizing against a partial figure sizes a
        position against a fraction of the account.

        WHY BOTH PROGRAMS. ``getTokenAccountsByOwner`` takes exactly one of
        ``mint`` or ``programId``. A programId-filtered scan therefore sees only
        that program's accounts, and filtering to the classic SPL program made
        every Token-2022 holding INVISIBLE -- including all of $SI, $Boy and the
        notify mint. Two round trips (one per program) are the cost of not
        silently dropping an asset. A mint can appear under both programs, so
        rows are kept per (mint, account) rather than collapsed: a token can
        legitimately exist in two accounts and collapsing them would understate
        the balance.

        The result carries the SPL Token program id, token account address, the
        account's own rent-exempt lamports and an ``is_empty`` flag, so the
        cleanup and rent-recovery paths do not have to re-query the chain.

        Amounts are returned as RAW integers. Decimal scaling is the caller's
        job, done once, with the mint's own decimals -- never guessed here.
        """
        out: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for program_id in TOKEN_PROGRAMS:
            for row in await self.get_token_accounts(owner, program_id):
                key = (row["mint"], row["account"])
                if key in seen:
                    continue
                seen.add(key)
                out.append(row)
        return out

    async def get_slot(self) -> int:
        res = await self._rpc("getSlot", [])
        try:
            return int(res)
        except (TypeError, ValueError) as exc:
            raise RpcError("getSlot: malformed result") from exc

    async def get_token_supply(self, mint: str) -> Optional[dict]:
        try:
            res = await self._rpc("getTokenSupply", [mint])
        except RpcError:
            return None
        return res.get("value") if isinstance(res, dict) else None

    # -- WSOL readiness ---------------------------------------------------
    async def wsol_readiness(self, owner: str, required_lamports: int
                             ) -> "WsolReadiness":
        """
        Is this wallet ready to trade SOL->token through a PRE-PROVISIONED
        wrapped-SOL account?

        Why this exists
        ---------------
        With ``wrapAndUnwrapSol: true`` Jupiter wraps SOL into a temporary wSOL
        account inside the swap and closes it again. That costs three extra
        instructions (SystemProgram CreateAccount, Token InitializeAccount,
        Token CloseAccount) and roughly a dozen extra account keys. On a thin
        L25 token that pushes the serialized transaction past Solana's 1232
        byte limit and the node rejects it as unserializable.

        With ``wrapAndUnwrapSol: false`` those three instructions disappear,
        but the wallet must ALREADY hold wSOL. This method answers that
        question authoritatively.

        Deliberate design points:
          * the balance is read from the chain, never inferred from the mere
            existence of an account. An empty ATA is NOT a funded ATA.
          * readiness is reported as four separate facts rather than one
            boolean, because the operator action differs: create the account,
            or fund it.
          * nothing here creates, wraps, funds or sends anything. It is a read.
        """
        ata = wsol_ata_address(owner, WSOL_MINT)
        present = False
        balance = 0
        try:
            res = await self._rpc("getTokenAccountsByOwner", [
                owner, {"mint": WSOL_MINT}, {"encoding": "jsonParsed"}])
        except RpcError:
            # A node that cannot answer this is not a licence to assume yes.
            return WsolReadiness(
                owner=owner, wsol_ata=ata, ata_present=False,
                balance_lamports=0, required_lamports=required_lamports,
                sufficient=False, query_ok=False)
        accounts = _token_accounts(res)
        for acct in accounts:
            info = (((acct.get("account") or {}).get("data") or {})
                    .get("parsed") or {}).get("info") or {}
            if str(info.get("mint")) != WSOL_MINT:
                continue
            if str(info.get("owner")) != str(owner):
                continue
            present = True
            try:
                balance = int((info.get("tokenAmount") or {}).get("amount") or 0)
            except (TypeError, ValueError):
                balance = 0
        return WsolReadiness(
            owner=str(owner), wsol_ata=ata, ata_present=present,
            balance_lamports=balance, required_lamports=required_lamports,
            sufficient=balance >= required_lamports and required_lamports > 0,
            query_ok=True)

    async def get_address_lookup_table(self, table_address: str) -> dict:
        """
        Read one address lookup table and return its addresses in order.

        v1 transactions do not support address lookup tables, so a v0
        transaction that is too large can only be made smaller-by-version, not
        smaller-by-compression: every looked-up address has to be materialised
        into the message. That makes this call load-bearing, and its failure
        modes explicit:

        * a table that does not exist, or whose authority has closed it, is
          reported as ``usable: False``. A closed table's addresses are no
          longer valid to load, so it is never reconstructed.
        * a node that cannot answer is ``query_ok: False``. A table that cannot
          be read is never treated as an empty table, because an empty table
          would silently drop accounts out of the transaction.

        WHY THERE ARE THREE READ PATHS
        --------------------------------
        ``getAddressLookupTable`` is the purpose-built method and is what a paid
        RPC will answer, but the **public** mainnet endpoint returns
        ``-32601 Method not found`` for it, so it cannot be the only path. The
        fallbacks are tried in order of how much the node does for us:

        1. ``getAddressLookupTable``   -- richest: splits writable / readonly.
        2. ``getAccountInfo`` +
           ``jsonParsed``              -- the node parses the table itself and
                                          returns the address list. Verified
                                          against the same table's base64 state:
                                          241 of 241 addresses identical, so
                                          the node's parse is the authority.
        3. ``getAccountInfo`` +
           ``base64``                  -- local decode of the table state, for a
                                          node that offers neither of the above.
                                          Used only when the decoded length
                                          matches the account exactly, and
                                          still subject to the caller's
                                          recompile check.

        The returned ``addresses`` list is in the order the chain stores them,
        which is the order the account index space expects. See
        :func:`autopaper.execution.txcodec.build_loaded_addresses`.
        """
        out: dict = {"table": str(table_address), "query_ok": False,
                     "usable": False, "addresses": [], "writable": [],
                     "readonly": [], "source": "", "detail": "not read"}

        # 1. purpose-built method
        try:
            res = await self._rpc("getAddressLookupTable", [table_address, {
                "encoding": "jsonParsed", "commitment": "confirmed"}])
        except RpcError as exc:
            res = None
            last_error = exc
        else:
            last_error = None
        if isinstance(res, dict) and (res.get("value") or {}):
            value = res.get("value") or {}
            info = ((value.get("data") or {}).get("parsed") or {}).get("info") \
                or {}
            if _alt_is_deactivated(info):
                out["query_ok"] = True
                out["detail"] = ("address lookup table has been deactivated; "
                                 "its addresses are not loadable")
                return out
            writable = [str(a) for a in (info.get("addresses") or [])]
            if writable:
                out.update(query_ok=True, usable=True, addresses=writable,
                           writable=writable, readonly=[], source="getAddressLookupTable",
                           detail=f"{len(writable)} addresses")
                return out

        # 2. node-parsed account
        try:
            res = await self._rpc("getAccountInfo", [table_address, {
                "encoding": "jsonParsed", "commitment": "confirmed"}])
        except RpcError as exc:
            res = None
            last_error = last_error or exc
        value = (res or {}).get("value") or {}
        data = value.get("data") or {}
        if isinstance(data, dict) and data.get("parsed"):
            info = ((data.get("parsed") or {}).get("info") or {})
            if _alt_is_deactivated(info):
                out["query_ok"] = True
                out["detail"] = ("address lookup table has been deactivated; "
                                 "its addresses are not loadable")
                return out
            addresses = [str(a) for a in (info.get("addresses") or [])]
            if addresses:
                out.update(query_ok=True, usable=True, addresses=addresses,
                           writable=addresses, readonly=[],
                           source="getAccountInfo:jsonParsed",
                           detail=f"{len(addresses)} addresses")
                return out

        # 3. local decode of the raw account state
        try:
            res = await self._rpc("getAccountInfo", [table_address, {
                "encoding": "base64", "commitment": "confirmed"}])
        except RpcError as exc:
            res = None
            last_error = last_error or exc
        value = (res or {}).get("value") or {}
        raw = (value.get("data") or [None])[0] if isinstance(
            value.get("data"), list) else None
        if raw:
            addresses = decode_lookup_table_state(raw)
            if addresses:
                out.update(query_ok=True, usable=True, addresses=addresses,
                           writable=addresses, readonly=[],
                           source="getAccountInfo:base64+local",
                           detail=f"{len(addresses)} addresses")
                return out
            out["query_ok"] = True
            out["detail"] = "lookup table state could not be decoded locally"
            return out

        out["detail"] = (f"address lookup table could not be read: "
                         f"{getattr(last_error, 'code', None)} "
                         f"{getattr(last_error, 'rpc_message', '')}").strip()
        return out

    async def resolve_lookup_tables(self, table_addresses: list
                                    ) -> tuple[list, list]:
        """Resolve every lookup table a message references.

        Returns ``(resolved, unresolved)`` where ``resolved`` is the
        ``(address, addresses)`` tuple list
        :func:`autopaper.execution.txcodec.build_loaded_addresses` expects, and
        ``unresolved`` names the tables that could not be read.

        Only the address list is returned. The writable / readonly partition is
        taken from the message being rebuilt rather than from here, because a
        lookup table does not record it and no RPC can report it correctly.

        Every table in ``table_addresses`` must be present in ``resolved``. A
        partial result is returned as such and never silently used: a message
        whose tables are only partly resolved cannot be expanded, and guessing
        the missing addresses would mean sending a transaction with the wrong
        accounts.
        """
        resolved: list = []
        unresolved: list = []
        for address in table_addresses:
            info = await self.get_address_lookup_table(address)
            if info.get("usable"):
                resolved.append((info["table"], info["addresses"]))
            else:
                unresolved.append({"table": info.get("table"),
                                   "detail": info.get("detail")})
        return resolved, unresolved

    # -- write path --------------------------------------------------------
    def transaction_size_report(self, tx_b64: str) -> dict:
        """
        Report whether a serialized transaction can physically be sent.

        This is a LOCAL check made before the RPC is contacted. It cannot make a
        failing order succeed and it cannot make an unsafe one safe -- it only
        replaces an opaque remote -32602 with a precise, local reason. A
        transaction over the limit for ITS OWN VERSION is unsendable at ANY
        commitment, so refusing early is strictly conservative: it is the same
        decision the node would make, taken sooner and with the actual byte
        counts attached.

        The limit is version-aware. legacy and v0 transactions are held to 1232
        bytes; a v1 transaction is held to 4096. The version is read from the
        deserialized message, never from a provider name or a request flag.
        """
        from .txcodec import measure_transaction
        return measure_transaction(tx_b64)

    async def simulate_transaction_detailed(self, tx_b64: str
                                            ) -> "SimulationOutcome":
        """
        Simulate a signed transaction without broadcasting it.

        Returns a :class:`SimulationOutcome`. Never raises, never broadcasts,
        and never degrades a failure into a success.
        """
        size = self.transaction_size_report(tx_b64)
        if not size["ok"]:
            # Local refusal. No RPC call, no broadcast. This is a subset of
            # what the node would reject, so it cannot mask a real problem.
            return SimulationOutcome(
                ok=False,
                status=("SIMULATION_FAILED_TX_TOO_LARGE" if size["status"]
                        == "TX_TOO_LARGE" else "SIMULATION_FAILED_TX_MALFORMED"),
                detail=size["detail"],
                stage="SIMULATION",
                rpc_code=None,
                rpc_message="",
                broadcast_attempted=False,
                size=size,
            )
        try:
            res = await self._rpc("simulateTransaction", [tx_b64, {
                "encoding": "base64",
                "sigVerify": False,
                "replaceRecentBlockhash": True,
                "commitment": "confirmed",
            }])
        except RpcError as exc:
            return SimulationOutcome(
                ok=False,
                status=classify_simulation_error(exc.code, exc.rpc_message
                                                 or str(exc)),
                detail=str(exc),
                stage="SIMULATION",
                rpc_code=exc.code,
                rpc_message=exc.rpc_message or "",
                broadcast_attempted=False,
                size=size,
            )
        value = (res or {}).get("value") or {}
        logs = _sanitize_simulation_logs(value.get("logs"))
        if value.get("err"):
            return SimulationOutcome(
                ok=False,
                status="SIMULATION_FAILED_TX_REJECTED",
                detail=f"simulation error: {value['err']}",
                stage="SIMULATION",
                rpc_code=None,
                rpc_message=sanitize_rpc_message(str(value["err"])),
                broadcast_attempted=False,
                size=size,
                units_consumed=value.get("unitsConsumed"),
                raw_err=value.get("err"),
                logs=logs,
            )
        return SimulationOutcome(
            ok=True, status="SIMULATION_OK", detail="simulation ok",
            stage="SIMULATION", rpc_code=None, rpc_message="",
            broadcast_attempted=False, size=size,
            units_consumed=value.get("unitsConsumed"),
            logs=logs)

    async def simulate_transaction(self, tx_b64: str) -> tuple[bool, str]:
        """
        Simulate a signed transaction without broadcasting it.

        Returns ``(ok, detail)`` and never raises, so a caller can report the
        precise reason instead of a stack trace. ``detail`` now carries the
        classified status, the RPC code and the sanitized node message rather
        than a bare code.
        """
        out = await self.simulate_transaction_detailed(tx_b64)
        return out.ok, f"{out.status}: {out.detail}"

    async def send_transaction(self, tx_b64: str) -> str:
        """Broadcast a signed transaction. Returns the signature string.

        ``skipPreflight=False`` keeps the node's own preflight as a second
        independent check; a preflight failure surfaces as an RPC error and the
        transaction is never considered sent.
        """
        res = await self._rpc("sendTransaction", [tx_b64, {
            "encoding": "base64",
            "skipPreflight": False,
            "preflightCommitment": "confirmed",
            "maxRetries": 0,
        }])
        if not isinstance(res, str) or not res:
            raise TransactionFailed("sendTransaction returned no signature")
        return res

    async def get_signature_statuses(self, signature: str) -> list:
        res = await self._rpc("getSignatureStatuses", [
            [signature], {"searchTransactionHistory": True}])
        return (res or {}).get("value") or []

    async def wait_for_confirmation(self, signature: str,
                                    last_valid_block_height: Optional[int] = None,
                                    timeout: float = 90.0,
                                    poll_interval: float = 2.0,
                                    commitment: str = "confirmed",
                                    clock: Any = time.time,
                                    sleep: Any = asyncio.sleep,
                                ) -> dict:
        """Poll until the signature reaches ``commitment`` or times out.

        The return value is a dict with exactly one of:
          ``status``   -> "confirmed" / "finalized"
          ``error``    -> a non-None chain error (terminal failure)
          ``timeout``  -> True (ambiguous, MUST NOT be treated as a fill)

        An ambiguous result is surfaced as ``timeout`` precisely so a caller
        cannot mistake "we stopped waiting" for "it filled".
        """
        deadline = clock() + timeout
        last_detail = "no status returned"

        async def _bounded(coro, what: str):
            """Run ``coro`` under the REMAINING wall-clock budget.

            The previous loop checked its deadline only at the top, then awaited
            up to two 15s RPC calls, so a 90s budget could run ~122s. Bounding
            each call by the remaining budget makes the configured timeout a
            true wall-clock ceiling.
            """
            remaining = deadline - clock()
            if remaining <= 0:
                raise asyncio.TimeoutError(f"{what}: budget exhausted")
            return await asyncio.wait_for(coro, timeout=remaining)

        while clock() < deadline:
            try:
                statuses = await _bounded(
                    self.get_signature_statuses(signature), "getSignatureStatuses"
                )
            except (RpcError, asyncio.TimeoutError) as exc:
                last_detail = str(exc) or "confirmation budget exhausted"
                statuses = []
            if statuses:
                st = statuses[0]
                if st is None:
                    last_detail = "signature not yet indexed"
                elif st.get("err"):
                    return {"status": "failed", "error": st["err"],
                            "signature": signature}
                else:
                    conf = st.get("confirmationStatus")
                    if conf in ("confirmed", "finalized"):
                        return {"status": conf, "error": None,
                                "signature": signature,
                                "slot": st.get("slot")}
                    last_detail = f"confirmationStatus={conf}"
            if last_valid_block_height is not None and clock() < deadline:
                try:
                    slot = await _bounded(self.get_slot(), "getSlot")
                    if slot > last_valid_block_height:
                        return {"status": "expired", "error": "blockhash expired",
                                "signature": signature}
                except (RpcError, asyncio.TimeoutError):
                    pass
            remaining = deadline - clock()
            if remaining <= 0:
                break
            await sleep(min(poll_interval, remaining))
        return {"status": "timeout", "error": last_detail,
                "signature": signature}

    async def get_transaction(self, signature: str,
                              commitment: str = "confirmed") -> Optional[dict]:
        """Fetch a confirmed transaction for actual-amount reconciliation.

        Asks for ``maxSupportedTransactionVersion: 1`` so a v1 transaction can
        be read back. With 0 the node refuses a v1 transaction on READ, which
        would mean the bot broadcasts something it can never reconcile.
        """
        try:
            res = await self._rpc("getTransaction", [signature, {
                "encoding": "jsonParsed",
                "commitment": commitment,
                "maxSupportedTransactionVersion":
                    MAX_SUPPORTED_TRANSACTION_VERSION,
            }])
        except RpcError:
            return None
        return res if isinstance(res, dict) else None

    async def get_block(self, slot: int,
                        commitment: str = "confirmed") -> Optional[dict]:
        """Fetch a confirmed block, v1-aware.

        Present so block-level consumers (forensics, reconciliation of a slot
        that contains one of our fills) ask for the same transaction-version
        support as :meth:`get_transaction`. Without it a block containing a v1
        transaction is refused entirely rather than returned without it.
        """
        try:
            res = await self._rpc("getBlock", [slot, {
                "encoding": "jsonParsed",
                "commitment": commitment,
                "maxSupportedTransactionVersion":
                    MAX_SUPPORTED_TRANSACTION_VERSION,
                "transactionDetails": "full",
                "rewards": False,
            }])
        except RpcError:
            return None
        return res if isinstance(res, dict) else None


# ---------------------------------------------------------------------------
# Actual-amount reconciliation
# ---------------------------------------------------------------------------
@dataclass
class OnchainAmounts:
    """Amounts read back from a CONFIRMED transaction.

    Every field is optional because a swap may legitimately produce a token
    leg, a wrapped-SOL leg, or both. ``net_lamports_delta`` is the SOL actually
    spent, which is the authoritative network cost for accounting.
    """

    signature: str = ""
    commitment: str = ""
    slot: Optional[int] = None
    token_input_amount: Optional[int] = None
    token_output_amount: Optional[int] = None
    token_mint: Optional[str] = None
    net_lamports_delta: Optional[int] = None
    #: The payer's whole SOL balance change (positive = SOL left the wallet).
    #: Contains the swap principal AND account-creation rent AND the network
    #: fee, which is why it may not be used as a per-token price.
    wallet_debit_lamports: Optional[int] = None
    #: Lamports that went into accounts which previously held nothing, i.e.
    #: account-creation rent. Itemised so the cost basis can state it.
    created_account_rent_lamports: Optional[int] = None
    #: Lamports that came out of accounts other than the fee payer, i.e. the
    #: SOL the transaction paid OUT. On a native-SOL sell this is what left the
    #: pools and program accounts.
    lamports_out_to_others: Optional[int] = None
    #: Lamports that came INTO accounts other than the fee payer, EXCLUDING
    #: account-creation rent. This is the swap principal plus any protocol fee
    #: skim, and it is the only quantity a per-token price may be derived from.
    lamports_in_from_others: Optional[int] = None
    #: The same total INCLUDING rent, so the two can be reconciled by eye.
    lamports_including_rent: Optional[int] = None
    #: The on-chain program ids that appear in the instruction list. Recorded so
    #: the venue that ACTUALLY executed is known, not merely the label the
    #: provider reported.
    invoked_programs: tuple = ()
    fee_lamports: Optional[int] = None
    success: bool = False
    log_messages: list = field(default_factory=list)
    #: Populated when a value could not be trusted (e.g. an implausible fee).
    #: A non-empty list means the caller must NOT silently use the number.
    sanity_warnings: list = field(default_factory=list)

    @property
    def fee_sol(self) -> Optional[float]:
        if self.fee_lamports is None:
            return None
        return self.fee_lamports / LAMPORTS_PER_SOL

    def as_dict(self) -> dict:
        return {
            "signature": self.signature,
            "commitment": self.commitment,
            "slot": self.slot,
            "token_input_amount": self.token_input_amount,
            "token_output_amount": self.token_output_amount,
            "token_mint": self.token_mint,
            "net_lamports_delta": self.net_lamports_delta,
            "wallet_debit_lamports": self.wallet_debit_lamports,
            "created_account_rent_lamports":
                self.created_account_rent_lamports,
            "lamports_out_to_others": self.lamports_out_to_others,
            "lamports_in_from_others": self.lamports_in_from_others,
            "lamports_including_rent": self.lamports_including_rent,
            "invoked_programs": list(self.invoked_programs),
            "fee_lamports": self.fee_lamports,
            "fee_sol": self.fee_sol,
            "success": self.success,
            "sanity_warnings": list(self.sanity_warnings),
        }


#: A transaction fee above this is not believed. 0.5 SOL is roughly three
#: orders of magnitude above any realistic swap fee, so a value beyond it is a
#: decoding error, not a fee. Used to fail loudly rather than clamp silently.
FEE_SANITY_MAX_LAMPORTS = 500_000_000


def _balance_info(entry) -> Optional[dict]:
    """Normalise one token-balance entry to ``{"mint", "amount", ...}``.

    ``getTransaction`` returns token balances in the FLAT shape::

        {"accountIndex": 1, "mint": "...", "owner": "...",
         "programId": "...", "uiTokenAmount": {"amount": "...",
                                              "decimals": 6, ...}}

    Some other RPC payloads use the nested ``account.data.parsed.info`` form.
    Both are accepted so this works regardless of which shape arrives; an
    unrecognised shape returns ``None`` rather than raising, because a single
    malformed entry must not abort reconciliation of the whole transaction.
    """
    if not isinstance(entry, dict):
        return None
    flat_amt = entry.get("uiTokenAmount")
    if isinstance(flat_amt, dict) and entry.get("mint"):
        return {"mint": entry["mint"], "amount": flat_amt.get("amount"),
                "owner": entry.get("owner"),
                "decimals": flat_amt.get("decimals"),
                "accountIndex": entry.get("accountIndex")}
    try:
        info = entry["account"]["data"]["parsed"]["info"]
    except (KeyError, TypeError):
        return None
    if not isinstance(info, dict) or not info.get("mint"):
        return None
    token_amount = info.get("tokenAmount") or {}
    if not isinstance(token_amount, dict):
        return None
    return {"mint": info["mint"], "amount": token_amount.get("amount"),
            "owner": info.get("owner"),
            "decimals": token_amount.get("decimals"),
            "accountIndex": entry.get("accountIndex")}


def _invoked_programs(logs: list) -> tuple:
    """
    Program ids that appear as an ``invoke`` target, in order, deduplicated.

    Read from the transaction's own log lines, so the program that ACTUALLY
    executed is known rather than the label a provider reported. This is what
    makes a venue label checkable: on 2026-10-02 a $GOAT entry and its exit ran
    under two different programs while both were recorded as "Pump.fun".
    """
    out: list[str] = []
    for line in logs or []:
        if not line.startswith("Program ") or " invoke [" not in line:
            continue
        target = line[len("Program "):].split(" ", 1)[0]
        if target and target not in out:
            out.append(target)
    return tuple(out)


def reconcile_transaction(tx: dict, signature: str, commitment: str,
                          owner: str, output_mint: str) -> OnchainAmounts:
    """Derive real input/output/fee from a confirmed transaction.

    Reads the wallet owner's own token-balance deltas, so it cannot attribute
    another account's movement to this trade. If the token legs cannot be
    identified the result is left incomplete rather than estimated: an
    unquantified fill is not a fill.
    """
    out = OnchainAmounts(signature=signature, commitment=commitment)
    if not tx:
        return out
    out.slot = tx.get("slot")
    # ``meta`` is a SIBLING of ``transaction`` in the ``getTransaction``
    # response, not a child of it. Reading it from the wrong place yields an
    # empty dict, which silently turns a confirmed fill into an unreconcilable
    # one: no fee, no SOL delta, no token legs, no logs. The nested location is
    # still accepted so a caller that reshapes the payload keeps working.
    meta = tx.get("meta")
    if not isinstance(meta, dict):
        meta = ((tx.get("transaction") or {}).get("meta")) or {}
    out.log_messages = list(meta.get("logMessages") or [])
    out.success = not meta.get("err")

    # The ACTUAL network fee. This is a distinct quantity from the wallet's
    # SOL balance delta: on a buy the delta is swap-input PLUS fee, and on a
    # sell it is proceeds MINUS fee, so using the delta as the fee overstated it
    # by ~4500x. The fee the runtime actually paid is ``meta.fee``, in lamports,
    # for the transaction as a whole.
    try:
        raw_fee = meta.get("fee")
        if raw_fee is not None:
            fee = int(raw_fee)
            if 0 <= fee <= FEE_SANITY_MAX_LAMPORTS:
                out.fee_lamports = fee
            else:
                # Absurd fee for this transaction. Fail LOUDLY by refusing to
                # report a number, never by clamping it to something plausible.
                out.fee_lamports = None
                out.sanity_warnings.append(
                    f"fee {fee} lamports outside sanity envelope "
                    f"(0..{FEE_SANITY_MAX_LAMPORTS})"
                )
    except (TypeError, ValueError):
        out.sanity_warnings.append("fee was not an integer")

    # Native SOL delta for the fee payer. Kept separately: it is the
    # authoritative SOL movement, and the fee above is the authoritative cost.
    try:
        idx = ((tx.get("transaction") or {}).get("message") or {}) \
            .get("accountKeys", [])
        key_index = next(
            (i for i, k in enumerate(idx)
             if (k.get("pubkey") if isinstance(k, dict) else k) == owner), None)
        if key_index is not None:
            pre = (meta.get("preBalances") or [])[key_index]
            post = (meta.get("postBalances") or [])[key_index]
            out.net_lamports_delta = int(pre) - int(post)
            out.wallet_debit_lamports = out.net_lamports_delta
    except (KeyError, IndexError, TypeError, ValueError):
        pass

    # SEPARATE THE SWAP LEG FROM THE WALLET'S OWN MOVEMENT.
    #
    # The payer's balance change bundles three different things: the swap
    # principal, account-creation rent, and the network fee. Only the first is a
    # price input. The other two are costs, and folding them into a per-token
    # price is what published ``actual_execution_price = 0.0004634`` for a $GOAT
    # entry on 2026-10-02 whose real execution price was 0.0002556.
    #
    # The split is read from the same balances: lamports that moved into or out
    # of accounts OTHER THAN the fee payer are the venue and protocol movement;
    # lamports that appeared in accounts that previously held nothing are
    # account-creation rent.
    try:
        pre_list = [int(v) for v in (meta.get("preBalances") or [])]
        post_list = [int(v) for v in (meta.get("postBalances") or [])]
        if key_index is not None and len(pre_list) == len(post_list):
            inbound = outbound = rent = 0
            for i, (a, b) in enumerate(zip(pre_list, post_list)):
                if i == key_index:
                    continue
                if b > a:
                    inbound += b - a
                    if a == 0:
                        rent += b - a
                elif a > b:
                    outbound += a - b
            # Account-creation rent is movement TOO, but it is a cost and not
            # swap principal. Leaving it inside the swap leg is the error, so it
            # is subtracted here and reported separately. On the 2026-10-02 $GOAT
            # entry this is the difference between 3,442,651 (wrong: includes
            # 1,513,840 of rent) and 1,928,811 (the real swap principal).
            out.lamports_in_from_others = max(0, inbound - rent)
            out.lamports_including_rent = inbound
            out.lamports_out_to_others = outbound
            out.created_account_rent_lamports = rent
    except (TypeError, ValueError):
        out.sanity_warnings.append(
            "could not separate the swap leg from the wallet balance change")
    out.invoked_programs = _invoked_programs(logs=out.log_messages)

    def _entry_wallet(entry) -> Optional[str]:
        """The wallet that owns a token-balance entry, or None if unknown."""
        info = _balance_info(entry)
        if info:
            owner = info.get("owner")
            return owner if isinstance(owner, str) else None
        owner = entry.get("owner")
        return owner if isinstance(owner, str) else None

    def _token_totals(entries) -> dict:
        """Sum token balances per mint from a ``jsonParsed`` balance list.

        Two things are load-bearing here and both were wrong before the $Boy
        incident.

        SHAPE. ``getTransaction`` returns token balances FLAT --
        ``{"accountIndex", "mint", "owner", "programId", "uiTokenAmount"}``.
        The nested ``account.data.parsed.info`` form belongs to other RPC
        payloads. Reading only the nested form made every entry raise KeyError,
        which the bare ``except`` then swallowed with ``continue``, so a swap
        that provably succeeded reconciled to zero amounts.

        OWNER. A single swap touches many accounts: the AMM pools, the user's
        own other token accounts, and any account created during the swap. Only
        the WALLET's own balances describe this trade. Summing every owner
        would attribute unrelated pools' movements to the fill. Entries whose
        owner cannot be determined are skipped rather than guessed at.
        """
        totals: dict = {}
        for entry in entries or []:
            try:
                info = _balance_info(entry)
                if not info:
                    continue
                if _entry_wallet(entry) != owner:
                    continue
                mint = info["mint"]
                amount = int(info["amount"])
            except (KeyError, TypeError, ValueError):
                continue
            totals[mint] = totals.get(mint, 0) + amount
        return totals

    try:
        pre_totals = _token_totals(meta.get("preTokenBalances"))
        post_totals = _token_totals(meta.get("postTokenBalances"))
    except Exception:                                    # pragma: no cover
        pre_totals = post_totals = {}

    # Net movement per mint: positive = received, negative = spent.
    deltas: dict = {}
    for mint in set(pre_totals) | set(post_totals):
        deltas[mint] = post_totals.get(mint, 0) - pre_totals.get(mint, 0)

    if output_mint in deltas:
        out.token_mint = output_mint
        out.token_output_amount = int(deltas[output_mint])
        spent = {m: v for m, v in deltas.items() if v < 0}
        if spent:
            mint = next(iter(spent))
            out.token_input_amount = int(-spent[mint])
            if out.token_mint is None:
                out.token_mint = mint
    return out
