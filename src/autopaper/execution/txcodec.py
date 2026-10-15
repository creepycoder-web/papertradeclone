"""Version-aware Solana transaction codec: legacy, v0 and v1 (SIMD-0385).

WHY THIS EXISTS
---------------
The bot used to carry ONE global constant::

    MAX_SOLANA_TX_BYTES = 1232

and applied it to every transaction regardless of what the transaction actually
was. That is correct for a legacy transaction and for a v0 transaction, and it
is WRONG for a v1 transaction. Solana mainnet now accepts v1 transactions, whose
serialized size ceiling is 4096 bytes, not 1232. A 1250-byte v0 transaction is
genuinely unsendable; a 3500-byte v1 transaction is perfectly sendable. Judging
both by the same number refuses trades that would have worked and reports them
as if they were malformed.

So the size gate is now a function of the *actual* transaction version, read
after deserialization. It is never inferred from a provider name, from a URL, or
from a request parameter -- only from the bytes.

WHAT WAS VERIFIED, NOT ASSUMED
------------------------------
Every claim this module encodes was checked against live Solana mainnet before
being written down (see ``reports/forensics/transaction_v1_upgrade.md``):

* ``solders==0.29.0`` (the only Solana dependency this runtime has) carries
  ``solders.message.MessageV1``, ``solders.message.TransactionConfig`` and
  ``solders.message.v1`` with ``MAX_TRANSACTION_SIZE == 4096``. It parses,
  compiles, serializes, signs and round-trips a v1 message.
* Mainnet (``solana-core 4.3.0``, feature set 3383571666) answers
  ``getFeeForMessage`` and ``simulateTransaction`` for a v1 transaction.
* ``getTransaction`` / ``getBlock`` accept ``maxSupportedTransactionVersion: 1``.

THE V1 WIRE FORMAT IS NOT THE V0 ONE
------------------------------------
This is the single most dangerous fact in this module, so it is stated plainly:

    legacy / v0 :  [signature-count][signature 64n][message]
    v1          :  [message][signature 64n]          <-- signatures come LAST

A v1 transaction therefore begins with ``0x81`` (``V1_PREFIX``) and carries its
signatures AFTER the message. The pre-existing
:meth:`autopaper.execution.solana.SolanaWallet.sign_wire` locates the signature
array by counting shortvec signatures from byte 0 and then signs whatever
follows them -- which is correct for legacy/v0 and **silently corrupts** a v1
transaction. :func:`sign_transaction_bytes` below is version-aware and refuses
rather than guessing.

SCOPE AND TRUST
---------------
This module is pure: it decodes bytes, resolves nothing over the network, and
never signs with anything but a keypair handed to it by the caller. Address
lookup tables are resolved by the caller (:meth:`SolanaRpc.get_address_lookup_table`)
and passed in as plain address lists, so the "resolve every table" step is
explicit, auditable and fails closed when a table cannot be read.

Nothing here decides whether an order is *desirable* -- only whether a
transaction is structurally sendable and whether a reconstruction preserved the
original semantics exactly.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

log = logging.getLogger("execution.txcodec")


# ---------------------------------------------------------------------------
# Versions and size ceilings
# ---------------------------------------------------------------------------
VERSION_LEGACY = "legacy"
VERSION_V0 = "v0"
VERSION_V1 = "v1"

#: Size ceiling per version. These are protocol constants, not tunables.
#: legacy and v0 are capped by the packet buffer at 1232 bytes; v1 carries its
#: own larger limit. Read them from solders where possible so a library upgrade
#: cannot silently disagree with this table.
MAX_BYTES_LEGACY = 1232
MAX_BYTES_V0 = 1232
MAX_BYTES_V1 = 4096


def _solders_v1_max_bytes() -> Optional[int]:
    """The v1 ceiling as the installed Solana library defines it."""
    try:
        from solders.message.v1 import MAX_TRANSACTION_SIZE
        return int(MAX_TRANSACTION_SIZE)
    except Exception:  # noqa: BLE001 - a missing constant must not crash a trade
        return None


_lib_v1_max = _solders_v1_max_bytes()
if _lib_v1_max:
    MAX_BYTES_V1 = _lib_v1_max

MAX_BYTES_BY_VERSION: dict[str, int] = {
    VERSION_LEGACY: MAX_BYTES_LEGACY,
    VERSION_V0: MAX_BYTES_V0,
    VERSION_V1: MAX_BYTES_V1,
}

#: Retained for callers that still import it. It is the legacy/v0 ceiling and
#: must NOT be used as a universal gate: use :func:`max_bytes_for_version`, or
#: the version-aware :func:`measure_transaction` which reads the real version.
MAX_SOLANA_TX_BYTES = MAX_BYTES_LEGACY

#: The v1 message prefix (a shortvec-tagged version marker). A transaction
#: whose first byte is this is a v1 transaction, whose message starts at byte 0
#: and whose signatures are appended at the end.
V1_PREFIX = 0x81

#: The v0 message prefix, which appears immediately after the signature array.
V0_PREFIX = 0x80

#: A signature is always 64 bytes.
SIGNATURE_SIZE = 64

#: ComputeBudget program. v1 moves resource configuration out of these
#: instructions and into the message's own config, so they are read for their
#: VALUES and then dropped during a v1 reconstruction.
COMPUTE_BUDGET_PROGRAM = "ComputeBudget111111111111111111111111111111"

# ComputeBudget instruction discriminants (a single leading byte).
_CB_REQUEST_HEAP_FRAME = 0x01
_CB_SET_COMPUTE_UNIT_LIMIT = 0x02
_CB_SET_COMPUTE_UNIT_PRICE = 0x03
_CB_SET_LOADED_ACCOUNTS_DATA_SIZE_LIMIT = 0x04


#: Substrings that identify a node refusal to LOAD an address lookup table
#: referenced by a versioned transaction. Matched case-insensitively.
#:
#: This is a distinct condition from "transaction too large" and it is a real,
#: reproducible one: a Raptor-built v0 swap at 824 bytes -- well inside the
#: 1232-byte limit -- was rejected by live mainnet with
#:
#:     -32602 invalid transaction: Transaction loads an address table account
#:     that doesn't exist
#:
#: while ``getAccountInfo`` on every one of those tables answered normally
#: (246 / 248 / 129 / 236 addresses). The tables exist; the node will not load
#: them for a simulation. Public RPC endpoints restrict this.
#:
#: A v1 transaction has no address lookup tables at all, so the same trade
#: re-encoded as v1 simulates clean. That makes the v1 reconstruction the fix
#: for this failure too, which is why it is classified here rather than left to
#: be discovered as an opaque -32602.
_ALT_LOAD_REFUSAL_MARKERS = (
    "address table account that doesn't exist",
    "address table account that does not exist",
    "address lookup table account",
    "failed to load address lookup table",
)


def is_alt_load_refusal(text: str) -> bool:
    """True when the node refused to load a transaction's lookup tables."""
    low = (text or "").lower()
    return any(marker in low for marker in _ALT_LOAD_REFUSAL_MARKERS)


#: Substrings that identify an exceeded loaded-accounts budget. Distinct from
#: the ALT case above, and answered differently: the fix is a larger budget in
#: the v1 resource config, not a different transaction.
_LOADED_ACCOUNTS_MARKERS = (
    "maxloadedaccountsdatasizeexceeded",
    "loaded accounts data size limit exceeded",
)


def is_loaded_accounts_refusal(text: str) -> bool:
    """True when the node refused on the loaded-accounts data budget."""
    low = (text or "").lower()
    return any(marker in low for marker in _LOADED_ACCOUNTS_MARKERS)


def max_bytes_for_version(version: Optional[str]) -> int:
    """The serialized-size ceiling that applies to ``version``.

    An unknown or missing version is treated as the legacy/v0 ceiling, i.e. the
    most conservative answer. Guessing "v1" for a transaction we failed to
    identify would turn an identification bug into an unsendable broadcast.
    """
    return MAX_BYTES_BY_VERSION.get(version or VERSION_LEGACY,
                                    MAX_BYTES_LEGACY)


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TxIdentity:
    """What the bytes say this transaction actually is."""

    version: str
    signature_count: int = 0
    instruction_count: int = 0
    account_count: int = 0
    #: Address lookup tables referenced by the message. v1 must be zero.
    lookup_table_count: int = 0
    static_key_count: int = 0

    def as_dict(self) -> dict:
        return {
            "transaction_version": self.version,
            "transaction_version_num": (0 if self.version == VERSION_LEGACY
                                        else int(self.version[1:])),
            "signature_count": self.signature_count,
            "instruction_count": self.instruction_count,
            "account_count": self.account_count,
            "lookup_table_count": self.lookup_table_count,
            "static_key_count": self.static_key_count,
        }


def detect_version(raw: bytes) -> Optional[str]:
    """Read the transaction version off the wire, or ``None`` if unparseable.

    The version is taken from the DESERIALIZED message type, not guessed from
    the first byte alone, and never from anything the caller told us about the
    provider. A ``0x81`` first byte is used only as a fast reject: if the
    library agrees the message is v1 we return v1, and if the library says
    otherwise we believe the library.
    """
    if not raw:
        return None
    try:
        from solders.transaction import VersionedTransaction
        message = VersionedTransaction.from_bytes(raw).message
    except Exception:  # noqa: BLE001 - unparseable is a normal outcome here
        return None
    name = type(message).__name__
    if name == "MessageV1":
        return VERSION_V1
    if name == "MessageV0":
        return VERSION_V0
    if name == "Message":
        return VERSION_LEGACY
    return None


def identify(raw: bytes) -> Optional[TxIdentity]:
    """Full structural identity of a serialized transaction."""
    version = detect_version(raw)
    if version is None:
        return None
    try:
        from solders.transaction import VersionedTransaction
        tx = VersionedTransaction.from_bytes(raw)
        message = tx.message
        keys = list(message.account_keys)
        tables = (list(message.address_table_lookups)
                  if version == VERSION_V0 else [])
        return TxIdentity(
            version=version,
            signature_count=len(list(tx.signatures)),
            instruction_count=len(message.instructions),
            account_count=len(keys),
            lookup_table_count=len(tables),
            static_key_count=len(keys),
        )
    except Exception:  # noqa: BLE001
        # The version was readable; the rest is best-effort metadata.
        return TxIdentity(version=version)


# ---------------------------------------------------------------------------
# Instruction -> program resolution
# ---------------------------------------------------------------------------
def instruction_program_ids(
    transaction_b64: str, *, loaded_addresses: Optional[list] = None
) -> list:
    """The program id of every instruction, in instruction order.

    WHY THIS EXISTS
    ---------------
    A simulation rejection is reported by the node as::

        {'InstructionError': [6, {'Custom': 6001}]}

    ``6`` is an instruction index, not an error code, and a ``Custom`` code is
    defined by the program that EMITTED it (Anchor offsets a program's own
    ``#[error_code]`` enum by 6000). To say what ``6001`` means, the emitting
    program must first be named, and it is named by resolving instruction 6's
    program id out of the transaction that was simulated.

    Returns a list positionally matching ``message.instructions``. An entry is
    ``None`` when the program id lives in an address lookup table that was not
    supplied: the caller then reports the index as unresolved rather than
    guessing a program. A transaction that cannot be decoded yields ``[]``.

    Pure and offline: no network, no signing, no broadcast. ``loaded_addresses``
    is the runtime-order address list produced by
    :func:`build_loaded_addresses` (static keys ++ loaded writable ++ loaded
    readonly); a legacy/v1 message needs none because its keys are materialised.
    """
    try:
        raw = base64.b64decode(transaction_b64, validate=True)
        from solders.transaction import VersionedTransaction
        message = VersionedTransaction.from_bytes(raw).message
    except Exception:  # noqa: BLE001 - an undecodable transaction has no answer
        return []

    static = [str(k) for k in message.account_keys]
    space = static + [str(a) for a in (loaded_addresses or [])]
    out: list = []
    for compiled in message.instructions:
        try:
            index = int(compiled.program_id_index)
        except Exception:  # noqa: BLE001
            out.append(None)
            continue
        out.append(space[index] if 0 <= index < len(space) else None)
    return out


# ---------------------------------------------------------------------------
# Size measurement
# ---------------------------------------------------------------------------
def measure_transaction(transaction_b64: str) -> dict:
    """Version-aware size verdict for a serialized transaction.

    Deserializes, identifies the real version, re-serializes, and compares the
    measured byte count against the ceiling for THAT version. Purely local: no
    RPC call, no signature, no broadcast.

    The returned mapping always carries the four numbers an operator needs to
    act on the answer::

        transaction_version  raw_bytes  max_bytes  over_by
    """
    try:
        raw = base64.b64decode(transaction_b64, validate=True)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": "TX_MALFORMED_ENCODING",
                "detail": f"transaction is not valid base64: "
                          f"{type(exc).__name__}",
                "transaction_version": None, "raw_bytes": None,
                "max_bytes": None, "over_by": None}

    version = detect_version(raw)
    if version is None:
        return {"ok": False, "status": "TX_MALFORMED",
                "detail": "transaction could not be deserialized; version "
                          "unknown so no size verdict is possible",
                "transaction_version": None, "raw_bytes": len(raw),
                "max_bytes": None, "over_by": None}

    limit = max_bytes_for_version(version)
    # The measurement is the number of bytes that would actually be put on the
    # wire -- i.e. the bytes the provider handed us -- not a re-serialization of
    # them. A node measures what it is sent, so that is what has to fit. A
    # re-serialized length could differ for a malformed or padded payload and
    # would understate the real cost.
    n = len(raw)
    over = n - limit
    return {
        "ok": over <= 0,
        "status": "TX_SIZE_OK" if over <= 0 else "TX_TOO_LARGE",
        "transaction_version": version,
        "raw_bytes": n,
        "max_bytes": limit,
        "over_by_bytes": over if over > 0 else 0,
        "over_by": over if over > 0 else 0,
        "b64_chars": len(transaction_b64),
        "limit_bytes": limit,
        "detail": (f"TX_SIZE {n} <= TX_LIMIT {limit} (version {version})"
                   if over <= 0 else
                   f"TX_SIZE {n} > TX_LIMIT {limit} OVER_BY {over} "
                   f"(version {version})"),
    }


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------
def message_bytes_and_signature_slots(raw: bytes) -> tuple[bytes, int, str]:
    """Split a serialized transaction into ``(message, signature_count, version)``.

    Raises ``ValueError`` for anything it cannot split with certainty. It never
    guesses: a wrong answer here would produce a signature over the wrong bytes,
    which the validator rejects as an invalid transaction.

    HOW THE VERSION IS DECIDED
    --------------------------
    ``v1`` is decided ONLY by a successful v1 deserialization. A leading
    ``0x81`` on its own is not enough, because ``0x81`` is also how a
    shortvec signature count above 127 is encoded -- a layout this module must
    still be able to sign.

    Everything else takes the legacy/v0 path, where the layout is unambiguous
    from byte 0: a shortvec signature count, then that many 64-byte signatures,
    then the message. The shape is validated (at least one signature, a
    non-empty message after the signature array) rather than parsed, so a
    structurally valid transaction is signed whether or not a full deserializer
    happens to accept it.
    """
    if not raw:
        raise ValueError("empty transaction")

    if detect_version(raw) == VERSION_V1:
        count = _count_trailing_signatures(raw)
        if count < 1:
            raise ValueError("v1 transaction carries no signature")
        start = len(raw) - SIGNATURE_SIZE * count
        if start <= 0:
            raise ValueError("v1 message region is empty")
        return raw[:start], count, VERSION_V1

    # legacy / v0: a shortvec signature count, then the signatures, then the
    # message.
    count, offset = _read_shortvec(raw, 0)
    if count < 1:
        raise ValueError("transaction has no signature slot")
    start = offset + SIGNATURE_SIZE * count
    if start >= len(raw):
        raise ValueError("transaction has no message to sign")
    message = raw[start:]
    version = VERSION_V0 if message[:1] == bytes([V0_PREFIX]) else VERSION_LEGACY
    return message, count, version


def _read_shortvec(raw: bytes, offset: int) -> tuple[int, int]:
    """Read a shortvec at ``offset``. Returns ``(value, next_offset)``."""
    if offset >= len(raw):
        raise ValueError("shortvec past end of buffer")
    first = raw[offset]
    if first < 0x80:
        return first, offset + 1
    extra = first & 0x7F
    if extra == 0 or extra > 8:
        # A multi-byte shortvec cannot describe a transaction this small, and
        # the previous byte pattern has already been ruled out. Refusing is
        # correct: guessing here would splice a signature into the wrong place.
        raise ValueError("unsupported shortvec encoding")
    value = 0
    for i in range(extra):
        byte = raw[offset + 1 + i]
        value |= (byte & 0x7F) << (7 * i)
    return value, offset + 1 + extra


def _count_trailing_signatures(raw: bytes) -> int:
    """How many signatures are appended to a v1 transaction.

    Read back from the parsed transaction rather than assumed, because the whole
    point of the v1 layout is that the signature array is not where a legacy
    reader expects it. A parse failure is fatal here: without it the split point
    is a guess, and a guess writes a signature into the middle of a message.
    """
    from solders.transaction import VersionedTransaction
    return len(list(VersionedTransaction.from_bytes(raw).signatures))


def signature_slot_offsets(raw: bytes) -> tuple[int, int, int]:
    """``(first_signature_offset, message_length, count)`` for splicing."""
    if detect_version(raw) == VERSION_V1:
        message, count, _ = message_bytes_and_signature_slots(raw)
        return len(raw) - SIGNATURE_SIZE * count, len(message), count
    count, offset = _read_shortvec(raw, 0)
    return offset, len(raw) - offset - SIGNATURE_SIZE * count, count


def sign_transaction_bytes(raw: bytes, keypair: Any) -> bytes:
    """Sign a wire transaction of ANY version and return the signed bytes.

    Version-aware by construction:

    * legacy / v0 -- the signature array precedes the message; the first slot is
      the payer's and is what this function fills.
    * v1 -- the signature array FOLLOWS the message; the payer's slot is the
      first 64 bytes after it.

    The message bytes that get signed are exactly the bytes the validator
    re-computes: the serialized message, unmodified. The key never leaves this
    call, and the version is read from the bytes rather than declared by the
    caller.
    """
    if not raw:
        raise ValueError("cannot sign an empty transaction")
    message, count, version = message_bytes_and_signature_slots(raw)
    if count < 1:
        raise ValueError("transaction has no signature slot to fill")
    signature = bytes(keypair.sign_message(message))
    if version == VERSION_V1:
        # Signatures are appended: overwrite the FIRST 64 bytes of the tail.
        first = len(raw) - SIGNATURE_SIZE * count
        return raw[:first] + signature + raw[first + SIGNATURE_SIZE:]
    # legacy / v0: the payer's slot is the first, immediately after the count.
    _, offset = _read_shortvec(raw, 0)
    return raw[:offset] + signature + raw[offset + SIGNATURE_SIZE:]


def is_unsigned(raw: bytes) -> bool:
    """True when every signature slot is still all zeroes."""
    try:
        from solders.transaction import VersionedTransaction
        return all(bytes(s) == bytes(SIGNATURE_SIZE)
                   for s in VersionedTransaction.from_bytes(raw).signatures)
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Message expansion (used by the v1 reconstruction)
# ---------------------------------------------------------------------------
def expand_message(message: Any, loaded_addresses: Iterable[str]
                   ) -> list:
    """Turn a compiled message into resolved ``Instruction`` objects.

    ``message`` is a ``solders`` ``MessageV0`` or ``MessageV1``;
    ``loaded_addresses`` are the addresses pulled out of the message's address
    lookup tables, already in the canonical order (see
    :func:`build_loaded_addresses`).

    Every account index in the message is resolved against
    ``static keys + loaded addresses`` -- exactly the order a Solana runtime
    uses -- and each account's signer / writable flags are read off the message
    header rather than assumed. An index that does not resolve is an error, not
    a silent zero: producing a transaction with the wrong accounts is exactly
    the failure mode this module exists to prevent.
    """
    from solders.instruction import AccountMeta, Instruction

    # Held as Pubkey objects, not raw bytes: an account index space holds
    # pubkeys, and handing AccountMeta a bytes object produces a wrong-looking
    # transaction rather than an error.
    space = [_as_pubkey(k) for k in message.account_keys]
    space += [_as_pubkey(a) for a in loaded_addresses]

    out = []
    for compiled in message.instructions:
        program_index = int(compiled.program_id_index)
        if program_index >= len(space):
            raise ValueError(
                f"instruction program index {program_index} is out of range "
                f"({len(space)} addresses available)")
        program_id = space[program_index]
        metas = []
        for index in compiled.accounts:
            index = int(index)
            if index >= len(space):
                raise ValueError(
                    f"instruction account index {index} is out of range "
                    f"({len(space)} addresses available)")
            metas.append(AccountMeta(
                pubkey=space[index],
                is_signer=bool(message.is_signer(index)),
                is_writable=bool(message.is_maybe_writable(index)),
            ))
        out.append(Instruction(program_id=program_id, accounts=metas,
                               data=bytes(compiled.data)))
    return out


def fee_payer_of(message: Any):
    """The fee payer of any message version.

    A v1 message exposes ``fee_payer()``; a legacy or v0 message does not, and
    its payer is by definition the first account key. Normalising here means the
    equivalence check can compare payers across versions instead of branching on
    the version at every call site.
    """
    payer = getattr(message, "fee_payer", None)
    if callable(payer):
        return payer()
    keys = list(message.account_keys)
    if not keys:
        raise ValueError("message has no account keys, so it has no fee payer")
    return keys[0]


def _pubkey_bytes(value: Any) -> bytes:
    """Raw 32 bytes of a pubkey given as a Pubkey, raw bytes, or base58."""
    from solders.pubkey import Pubkey
    if isinstance(value, Pubkey):
        return bytes(value)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return bytes(Pubkey.from_string(str(value)))


def _as_pubkey(value: Any):
    """A ``Pubkey`` from a Pubkey, raw bytes, or a base58 string."""
    from solders.pubkey import Pubkey
    if isinstance(value, Pubkey):
        return value
    return Pubkey(_pubkey_bytes(value))


# ---------------------------------------------------------------------------
# ComputeBudget extraction
# ---------------------------------------------------------------------------
@dataclass
class ResourcePlan:
    """Resource settings lifted out of a v0 transaction's ComputeBudget set.

    A v1 transaction carries these in the message's own
    :class:`solders.message.TransactionConfig` rather than as instructions
    (SIMD-0385). This dataclass is the hand-off: the reconstruction reads the
    original v0 values here, writes them into the v1 config, and the verifier
    proves the mapping was faithful.
    """

    compute_unit_limit: Optional[int] = None
    compute_unit_price_micro_lamports: Optional[int] = None
    loaded_accounts_data_size_limit: Optional[int] = None
    heap_size: Optional[int] = None
    #: Number of ComputeBudget instructions found and consumed.
    compute_budget_instruction_count: int = 0

    def as_dict(self) -> dict:
        return {
            "compute_unit_limit": self.compute_unit_limit,
            "compute_unit_price_micro_lamports":
                self.compute_unit_price_micro_lamports,
            "loaded_accounts_data_size_limit":
                self.loaded_accounts_data_size_limit,
            "heap_size": self.heap_size,
            "compute_budget_instruction_count":
                self.compute_budget_instruction_count,
        }


def read_compute_budget(instructions: Iterable[Any]) -> ResourcePlan:
    """Read the resource settings a v0 transaction carries as instructions.

    Only the ComputeBudget program is read. Every other instruction is returned
    untouched by the caller, so a swap's instructions are never interpreted,
    reordered or discarded.
    """
    plan = ResourcePlan()
    for ix in instructions:
        if str(ix.program_id) != COMPUTE_BUDGET_PROGRAM:
            continue
        data = bytes(ix.data)
        if not data:
            continue
        kind = data[0]
        body = data[1:]
        try:
            if kind == _CB_SET_COMPUTE_UNIT_LIMIT and len(body) >= 4:
                plan.compute_unit_limit = int.from_bytes(body[:4], "little")
            elif kind == _CB_SET_COMPUTE_UNIT_PRICE and len(body) >= 8:
                plan.compute_unit_price_micro_lamports = int.from_bytes(
                    body[:8], "little")
            elif kind == _CB_SET_LOADED_ACCOUNTS_DATA_SIZE_LIMIT and len(body) >= 4:
                plan.loaded_accounts_data_size_limit = int.from_bytes(
                    body[:4], "little")
            elif kind == _CB_REQUEST_HEAP_FRAME and len(body) >= 4:
                plan.heap_size = int.from_bytes(body[:4], "little")
            else:
                continue
        except (TypeError, ValueError):  # pragma: no cover
            continue
        plan.compute_budget_instruction_count += 1
    return plan


#: Defaults applied ONLY where the source transaction said nothing.
#:
#: The loaded-accounts budget is the one that matters and it is NOT a guess. v1
#: supports no address lookup tables, so a transaction rescued from v0 has every
#: looked-up address materialised as an explicit account key -- which means the
#: runtime has to load far more account data than the v0 original did. Measured
#: on mainnet against the incident token ($AI, a Whirlpool -> Pump.fun route):
#:
#:     128 KiB .. 16 MiB -> MaxLoadedAccountsDataSizeExceeded
#:     24 MiB            -> MaxLoadedAccountsDataSizeExceeded
#:     32 MiB            -> SIMULATION_OK
#:     64 MiB            -> SIMULATION_OK
#:
#: 64 MiB is Solana's maximum for this field and is what a correct v1
#: transaction carries. It is a CAP, not a reservation, so setting it costs
#: nothing: the transaction still pays the base fee plus the priority fee. The
#: per-transaction compute and priority settings are untouched.
DEFAULT_LOADED_ACCOUNTS_DATA_SIZE_LIMIT = 64 * 1024 * 1024
DEFAULT_COMPUTE_UNIT_LIMIT = 400_000
DEFAULT_HEAP_SIZE = 131_072
#: A v1 priority fee is a FLAT lamport amount, not a micro-lamports-per-unit
#: price. Verified against mainnet: a config with ``priority_fee=777`` bills
#: ``getFeeForMessage == 5777`` (5000 base + 777), and 250000 bills 255000.
DEFAULT_PRIORITY_FEE_LAMPORTS = 50_000

#: Solana's own maximum for the loaded-accounts budget, used as the escalation
#: target if a caller ever needs more than the default.
MAX_LOADED_ACCOUNTS_DATA_SIZE_LIMIT = 64 * 1024 * 1024

#: v1 protocol bounds, from ``solders.message.v1``.
V1_MIN_HEAP_SIZE = 32_768
V1_MAX_HEAP_SIZE = 262_144
V1_MAX_ADDRESSES = 64
V1_MAX_INSTRUCTIONS = 64
V1_MAX_SIGNATURES = 12


def build_v1_config(plan: ResourcePlan, *,
                    priority_fee_lamports: Optional[int] = None
                    ) -> tuple[Any, dict]:
    """Build a v1 :class:`TransactionConfig` from a v0 transaction's resources.

    ``priority_fee_lamports`` is preferred when the caller knows the venue's
    real priority-fee budget (Jupiter and Raptor both report it). Otherwise it
    is derived from the v0 per-unit price so the reconstructed transaction pays
    the same priority the original asked for, rather than silently paying
    nothing.

    Returns ``(config, description)``. Every applied value, and the value it was
    derived from, is reported so the choice is auditable after the fact.
    """
    from solders.message import TransactionConfig

    heap = plan.heap_size or DEFAULT_HEAP_SIZE
    heap = max(V1_MIN_HEAP_SIZE, min(int(heap), V1_MAX_HEAP_SIZE))
    cu_limit = int(plan.compute_unit_limit or DEFAULT_COMPUTE_UNIT_LIMIT)
    lads = int(plan.loaded_accounts_data_size_limit
               or DEFAULT_LOADED_ACCOUNTS_DATA_SIZE_LIMIT)

    if priority_fee_lamports is not None:
        priority_fee = int(priority_fee_lamports)
        priority_source = "provider_reported_lamports"
    elif plan.compute_unit_price_micro_lamports is not None:
        # v0 priced compute in micro-lamports per unit. v1 charges a flat
        # lamport amount, so the equivalent budget is price * limit / 1e6.
        priority_fee = int(plan.compute_unit_price_micro_lamports * cu_limit
                           // 1_000_000)
        priority_source = "derived_from_v0_compute_unit_price"
    else:
        priority_fee = DEFAULT_PRIORITY_FEE_LAMPORTS
        priority_source = "default_lamports"

    priority_fee = max(0, int(priority_fee))

    # A config with every field null is invalid on both sides, so the mask is
    # always populated with at least one field. In practice all four are set.
    config = TransactionConfig(
        compute_unit_limit=cu_limit,
        priority_fee=priority_fee,
        loaded_accounts_data_size_limit=lads,
        heap_size=heap,
    )
    description = {
        "compute_unit_limit": cu_limit,
        "compute_unit_limit_source": ("from_v0_compute_budget_instruction"
                                      if plan.compute_unit_limit else
                                      "default"),
        "loaded_accounts_data_size_limit": lads,
        "loaded_accounts_data_size_limit_source": (
            "from_v0_compute_budget_instruction"
            if plan.loaded_accounts_data_size_limit else "default"),
        "heap_size": heap,
        "heap_size_source": ("from_v0_compute_budget_instruction"
                             if plan.heap_size else "default"),
        "priority_fee_lamports": priority_fee,
        "priority_fee_source": priority_source,
        "v0_compute_unit_price_micro_lamports":
            plan.compute_unit_price_micro_lamports,
        "v0_compute_budget_instruction_count":
            plan.compute_budget_instruction_count,
    }
    return config, description


# ---------------------------------------------------------------------------
# ALT resolution
# ---------------------------------------------------------------------------
def build_loaded_addresses(message: Any, resolved_tables: list) -> list:
    """Flatten resolved lookup tables into the runtime's account-index order.

    A Solana v0 message numbers its accounts as::

        static keys
        ++ loaded writable addresses, by ascending loaded position
        ++ loaded readonly addresses, by ascending loaded position

    Getting this order wrong yields a transaction whose accounts are silently
    permuted, which is the most dangerous possible outcome of a "size fix", so
    it is built here explicitly and nowhere else.

    THE SPLIT AND THE POSITION COME FROM THE MESSAGE, NOT FROM THE TABLE
    --------------------------------------------------------------------
    A lookup table stores one flat ``Vec<Pubkey>``. It does not record which
    entries are writable, nor the order the runtime loads them in. Both come
    from the message's own lookup entry, which in ``solders`` is exposed as
    ``writable_indexes`` / ``readonly_indexes`` -- a byte array read as::

        entry[loaded_position] = index into that table's address list

    (It is the values of Solana's ``IndexMap<Pubkey, u8>`` in key order, which
    is why the positions run 0..n-1 while the values are a permutation.) The
    message being rebuilt is therefore the only authority for the mapping, and
    :func:`verify_address_resolution` proves the result by recompiling the
    original message and requiring byte-identical output.

    ``resolved_tables`` is an ordered list of ``(table_address, addresses)``,
    positionally matched to the message's lookup list.
    """
    lookups = list(message.address_table_lookups)
    if len(lookups) != len(resolved_tables):
        raise ValueError(
            f"message references {len(lookups)} lookup table(s) but "
            f"{len(resolved_tables)} were resolved")
    writable: list = []
    readonly: list = []
    for lookup, entry in zip(lookups, resolved_tables):
        table_address, addresses = entry[0], entry[1]
        for positions, sink, kind in (
                (lookup.writable_indexes, writable, "writable"),
                (lookup.readonly_indexes, readonly, "readonly")):
            for position, table_index in enumerate(bytes(positions)):
                table_index = int(table_index)
                if table_index >= len(addresses):
                    raise ValueError(
                        f"lookup table {table_address} holds "
                        f"{len(addresses)} addresses but the message's "
                        f"{kind} index {position} points at {table_index}")
                sink.append(addresses[table_index])
    return writable + readonly


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------
@dataclass
class Reconstruction:
    """The outcome of one attempt to re-encode a transaction as v1."""

    ok: bool
    status: str
    detail: str
    transaction_b64: Optional[str] = None
    transaction_version: Optional[str] = None
    raw_bytes: Optional[int] = None
    max_bytes: Optional[int] = None
    over_by: Optional[int] = None
    #: Semantic-preservation proof. ``ok`` here is what authorises the
    #: reconstruction; a false value means DO NOT USE the rebuilt transaction.
    verification: dict = field(default_factory=dict)
    resource_config: dict = field(default_factory=dict)
    resolved_tables: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "status": self.status, "detail": self.detail,
            "transaction_version": self.transaction_version,
            "raw_bytes": self.raw_bytes, "max_bytes": self.max_bytes,
            "over_by": self.over_by, "verification": self.verification,
            "resource_config": self.resource_config,
            "resolved_tables": self.resolved_tables,
        }


def verify_semantics_preserved(source: Any, rebuilt: Any,
                            loaded_addresses: Optional[list] = None,
                            expected: Optional[list] = None) -> dict:
    """Prove a rebuilt message means exactly what the original meant.

    This is the gate that makes a reconstruction safe. It compares the two
    messages on every axis that changes what the chain will do:

    * the fee payer
    * the blockhash / lifetime specifier
    * the number of instructions and their ORDER
    * each instruction's program id and instruction data, byte for byte
    * each instruction's account list, in order, with the same signer and
      writable flags

    It deliberately does not compare the account *set* by identity: a v0 message
    keeps its static keys while a v1 message materialises address lookup table
    entries into explicit keys, so the key lists differ by construction even
    when the instructions are identical. Comparing instructions is the
    semantically meaningful check.

    ``expected`` overrides which instructions the rebuild is expected to carry.
    It exists for one honest reason: a v1 transaction expresses its resource
    settings in the message config rather than as ComputeBudget instructions, so
    the rebuild legitimately omits those two instructions. The caller therefore
    passes the non-budget instruction list, and the CHECK still proves that every
    instruction that changes what the chain does survived, in order, byte for
    byte. Dropping a *swap* instruction is still a failure.
    """
    from solders.pubkey import Pubkey

    try:
        source_ixs = (list(expected) if expected is not None
                      else expand_message(source, loaded_addresses or []))
        rebuilt_ixs = expand_message(rebuilt, [])
    except Exception as exc:  # noqa: BLE001
        return {"ok": False,
                "detail": f"could not expand messages: {exc}"}

    problems: list[str] = []
    if str(fee_payer_of(source)) != str(fee_payer_of(rebuilt)):
        problems.append("fee payer changed")
    if bytes(source.recent_blockhash) != bytes(rebuilt.recent_blockhash):
        problems.append("recent blockhash changed")
    if len(source_ixs) != len(rebuilt_ixs):
        problems.append(
            f"instruction count changed ({len(source_ixs)} -> "
            f"{len(rebuilt_ixs)})")
    else:
        for i, (a, b) in enumerate(zip(source_ixs, rebuilt_ixs)):
            if str(a.program_id) != str(b.program_id):
                problems.append(f"instruction {i} program id changed")
            if bytes(a.data) != bytes(b.data):
                problems.append(f"instruction {i} data changed")
            if len(a.accounts) != len(b.accounts):
                problems.append(f"instruction {i} account count changed")
                continue
            for j, (ma, mb) in enumerate(zip(a.accounts, b.accounts)):
                if str(ma.pubkey) != str(mb.pubkey):
                    problems.append(
                        f"instruction {i} account {j} address changed")
                if bool(ma.is_signer) != bool(mb.is_signer):
                    problems.append(
                        f"instruction {i} account {j} signer flag changed")
                if bool(ma.is_writable) != bool(mb.is_writable):
                    problems.append(
                        f"instruction {i} account {j} writable flag changed")
            if problems:
                break

    return {
        "ok": not problems,
        "instruction_count": len(source_ixs),
        "fee_payer": str(fee_payer_of(rebuilt)),
        "recent_blockhash": str(rebuilt.recent_blockhash),
        "problems": problems,
        "detail": "instruction program ids, data, account order and "
                  "signer/writable flags are identical"
                  if not problems else "; ".join(problems[:4]),
    }


def verify_address_resolution(source: Any, resolved_tables: list,
                             source_version: str) -> dict:
    """Prove the resolved lookup tables are STRUCTURALLY consistent with the
    message's account index space.

    The proof: recompile the ORIGINAL v0 message from the expanded instructions
    and the resolved tables, then compare the serialized bytes with the original
    message's bytes. The compiler is deterministic, so an identical result means
    the expansion maps back to the same index structure.

    WHAT THIS PROVES, PRECISELY
    ---------------------------
    A v0 message records account **indices**, not addresses. So a byte-identical
    recompile establishes that:

    * every table's writable / readonly partition matches the message's lookup
      entry, and the two partition sizes are exactly right;
    * the resolved address count is exactly the count the message needs;
    * every instruction's account-index mapping round-trips through the tables
      unchanged, so nothing was dropped, duplicated or shifted.

    Measured detection rates over 12 independently generated fixtures:

        two loaded values swapped ................ 12/12 detected
        a loaded address removed .................. 12/12 detected
        a pair of loaded slots reordered .......... 12/12 detected
        one loaded value replaced by a new value .. 11/12 detected
        an extra UNREFERENCED address appended ....  0/12, and correctly so
        the correct table (control) ..............  0/12, and correctly so

    WHAT IT DOES NOT PROVE
    ---------------------
    It cannot detect a *uniformly substituted* address. If the resolved table
    were wrong everywhere in the same way, the message would still recompile to
    the same bytes, because the message never contains the addresses to
    disagree with. That is not a gap that can be closed here: nothing in the
    transaction records what a table contains, so the on-chain table IS the
    authority and the bot has no independent statement to check it against.
    What the bot DOES control -- that the expansion and the rebuild agree
    instruction for instruction, account for account, flag for flag -- is proven
    by :func:`verify_semantics_preserved`.

    An unreferenced extra address being accepted is not a miss: the runtime only
    loads the addresses a message references, so an extra unused entry cannot
    change what the chain does.

    Returns ``{"ok": bool, "detail": str}``. ``ok=False`` means DO NOT REBUILD.
    A source with no lookup tables is trivially self-consistent and passes.
    """
    if source_version != VERSION_V0:
        return {"ok": True,
                "detail": "legacy source has no address lookup tables"}
    try:
        from solders.address_lookup_table_account import AddressLookupTableAccount
        from solders.message import MessageV0
        from solders.pubkey import Pubkey
    except Exception as exc:  # noqa: BLE001  # pragma: no cover
        return {"ok": False,
                "detail": f"solders could not be imported: {type(exc).__name__}"}

    lookups = list(source.address_table_lookups)
    if not lookups:
        return {"ok": True, "detail": "message references no lookup tables"}

    try:
        tables_in_order = [AddressLookupTableAccount(
            lookup.account_key,
            [Pubkey.from_string(a) for a in entry[1]])
            for lookup, entry in zip(lookups, resolved_tables)]
        loaded = build_loaded_addresses(source, resolved_tables)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False,
                "detail": f"could not regroup the resolved addresses: {exc}"}

    try:
        instructions = expand_message(source, loaded)
        rebuilt = MessageV0.try_compile(fee_payer_of(source), instructions,
                                        tables_in_order, source.recent_blockhash)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False,
                "detail": f"recompiling the v0 message failed: "
                          f"{type(exc).__name__}"}

    if bytes(rebuilt) != bytes(source):
        return {"ok": False,
                "detail": (f"recompiled v0 message is {len(bytes(rebuilt))} "
                           f"bytes but the original is {len(bytes(source))}; "
                           f"the resolved addresses do not reproduce the "
                           f"original account index space")}
    return {"ok": True,
            "detail": (f"recompiling the v0 message from the resolved tables "
                       f"reproduces the original {len(bytes(source))} bytes "
                       f"exactly")}


def reconstruct_as_v1(
    transaction_b64: str,
    *,
    resolved_tables: Optional[list] = None,
    priority_fee_lamports: Optional[int] = None,
) -> Reconstruction:
    """Re-encode a legacy/v0 transaction as a v1 transaction.

    The pipeline, in order, with no step skipped:

        decode -> resolve ALT -> materialise addresses -> rebuild as v1
        -> apply v1 resource config -> serialize -> verify semantics -> size

    The caller must have resolved every address lookup table already and pass
    them as ``resolved_tables``: a list of ``(table_address, addresses)``, in
    the same order as the message's lookup list. Passing fewer tables than the
    message references is an error, not something to work around, because a
    partially-resolved message cannot be expanded at all.

    The writable / readonly split is taken from the MESSAGE, not from the
    resolved table -- see :func:`build_loaded_addresses` for why the table
    cannot supply it.

    Returns a :class:`Reconstruction`. A result with ``ok=False`` must never be
    used: the caller falls back to reporting the token as un-executable at this
    size rather than sending something whose semantics were not proven.
    """
    from solders.transaction import VersionedTransaction

    limit = max_bytes_for_version(VERSION_V1)
    try:
        raw = base64.b64decode(transaction_b64, validate=True)
    except Exception as exc:  # noqa: BLE001
        return Reconstruction(False, "TX_MALFORMED_ENCODING",
                              f"not valid base64: {type(exc).__name__}",
                              transaction_version=VERSION_V1, max_bytes=limit)

    try:
        source_tx = VersionedTransaction.from_bytes(raw)
    except Exception as exc:  # noqa: BLE001
        return Reconstruction(False, "TX_MALFORMED",
                              f"could not deserialize: {type(exc).__name__}",
                              transaction_version=VERSION_V1, max_bytes=limit)

    source = source_tx.message
    source_version = detect_version(raw)
    if source_version == VERSION_V1:
        # Already v1. Nothing to rescue: report the measurement instead of
        # rebuilding a transaction that is already in the target format.
        measured = measure_transaction(transaction_b64)
        return Reconstruction(
            measured["ok"], "ALREADY_V1",
            "transaction is already v1; no reconstruction needed",
            transaction_b64=transaction_b64,
            transaction_version=VERSION_V1,
            raw_bytes=measured.get("raw_bytes"),
            max_bytes=limit,
            over_by=measured.get("over_by"))
    if source_version not in (VERSION_LEGACY, VERSION_V0):
        return Reconstruction(False, "TX_MALFORMED",
                              "source transaction version is unknown",
                              transaction_version=VERSION_V1, max_bytes=limit)

    # -- resolve + materialise ------------------------------------------------
    lookups = (list(source.address_table_lookups)
               if source_version == VERSION_V0 else [])
    tables = list(resolved_tables or [])
    if len(tables) != len(lookups):
        return Reconstruction(
            False, "ALT_UNRESOLVED",
            f"message references {len(lookups)} address lookup table(s) but "
            f"{len(tables)} were resolved; refusing to rebuild a partially "
            f"resolved transaction",
            transaction_version=source_version)

    try:
        loaded = build_loaded_addresses(source, tables)
        source_ixs = expand_message(source, loaded)
    except Exception as exc:  # noqa: BLE001
        return Reconstruction(False, "ALT_RESOLUTION_FAILED",
                              f"could not materialise addresses: {exc}",
                              transaction_version=VERSION_V1, max_bytes=limit)

    # Hard gate: the resolved tables must reproduce the original message byte
    # for byte. This runs BEFORE the v1 rebuild so a bad resolution can never
    # produce a transaction that looks fine and swaps the wrong accounts.
    resolution = verify_address_resolution(source, tables, source_version)
    if not resolution["ok"]:
        return Reconstruction(
            False, "ALT_RESOLUTION_UNVERIFIED",
            f"address lookup tables could not be verified: "
            f"{resolution['detail']}",
            transaction_version=VERSION_V1, max_bytes=limit,
            verification=resolution)

    # -- v1 resource configuration -------------------------------------------
    # Read the v0 ComputeBudget instructions for their VALUES, then leave them
    # out of the v1 message: in v1 those settings live in the message config, and
    # copying the instructions across as well would be exactly the "blindly copy
    # V0 ComputeBudget instructions into V1" mistake. Verified on mainnet: a v1
    # message's config governs the priority fee even when a
    # SetComputeUnitPrice instruction is also present.
    plan = read_compute_budget(source_ixs)
    swap_ixs = [ix for ix in source_ixs
                if str(ix.program_id) != COMPUTE_BUDGET_PROGRAM]
    dropped_cb = len(source_ixs) - len(swap_ixs)
    config, config_desc = build_v1_config(
        plan, priority_fee_lamports=priority_fee_lamports)

    if len(swap_ixs) > V1_MAX_INSTRUCTIONS:
        return Reconstruction(
            False, "V1_INSTRUCTION_LIMIT",
            f"transaction has {len(swap_ixs)} non-budget instructions; v1 "
            f"allows at most {V1_MAX_INSTRUCTIONS}",
            transaction_version=VERSION_V1, max_bytes=limit,
            resource_config=config_desc)

    # -- rebuild --------------------------------------------------------------
    try:
        from solders.message import MessageV1
        rebuilt = MessageV1.try_compile(
            fee_payer_of(source), swap_ixs, source.recent_blockhash, config)
    except Exception as exc:  # noqa: BLE001
        return Reconstruction(
            False, "V1_COMPILE_FAILED",
            f"could not compile a v1 message: {type(exc).__name__}: {exc}",
            transaction_version=VERSION_V1, max_bytes=limit,
            resource_config=config_desc)

    # The compiler materialises one key per distinct address, so the address
    # count is the real v1 limit to check -- and it is a hard protocol bound,
    # not a size heuristic.
    if len(rebuilt.account_keys) > V1_MAX_ADDRESSES:
        return Reconstruction(
            False, "V1_ADDRESS_LIMIT",
            f"materialising the addresses needs {len(rebuilt.account_keys)} "
            f"keys; v1 allows at most {V1_MAX_ADDRESSES}. v1 does not support "
            f"address lookup tables, so the addresses cannot be compressed",
            transaction_version=VERSION_V1, max_bytes=limit,
            resource_config=config_desc)

    # -- verify, THEN serialize ----------------------------------------------
    verification = verify_semantics_preserved(source, rebuilt, loaded,
                                             expected=swap_ixs)
    if not verification["ok"]:
        return Reconstruction(
            False, "SEMANTICS_NOT_PRESERVED",
            f"rebuilt message does not match the original: "
            f"{verification.get('detail')}",
            transaction_version=VERSION_V1, max_bytes=limit,
            verification=verification, resource_config=config_desc)

    # The signature count comes from the compiled header, never from a guess: a
    # v1 transaction appends one 64-byte slot per required signature.
    from solders.transaction import VersionedTransaction as _VT
    try:
        num_required = int(rebuilt.header.num_required_signatures)
    except Exception:  # noqa: BLE001  # pragma: no cover
        num_required = 1
    if num_required < 1 or num_required > V1_MAX_SIGNATURES:
        return Reconstruction(
            False, "V1_SIGNATURE_LIMIT",
            f"compiled message needs {num_required} signatures; v1 allows "
            f"between 1 and {V1_MAX_SIGNATURES}",
            transaction_version=VERSION_V1, max_bytes=limit,
            verification=verification, resource_config=config_desc)

    try:
        # ``NullSigner`` is what makes this an UNSIGNED transaction: it fills
        # each required signature slot with zeroes, which is exactly the shape
        # an aggregator hands a client and exactly the shape the bot's own
        # wire-level signer then replaces. No key material is involved here, and
        # ``solders`` will not build a versioned transaction from raw byte
        # blobs, so this is the only way to emit the envelope without signing.
        #
        # ONE NullSigner PER REQUIRED SIGNATURE, each bound to that signer's own
        # pubkey. A single signer is not enough ("not enough signers") and
        # repeating the payer's is actively wrong ("keypair-pubkey mismatch"),
        # so the list is built from the message's own header and account keys.
        from solders.null_signer import NullSigner
        keys = list(rebuilt.account_keys)
        signers = [NullSigner(keys[i]) for i in range(num_required)]
        rebuilt_tx = _VT(rebuilt, signers)
        raw_v1 = bytes(rebuilt_tx)
    except Exception as exc:  # noqa: BLE001
        return Reconstruction(
            False, "V1_SERIALIZE_FAILED",
            f"could not serialize the v1 transaction: {type(exc).__name__}",
            transaction_version=VERSION_V1, max_bytes=limit,
            verification=verification, resource_config=config_desc)

    over = len(raw_v1) - limit
    b64 = base64.b64encode(raw_v1).decode("ascii")

    # Round-trip: a transaction that cannot be re-parsed as v1 is not a v1
    # transaction, whatever our own encoder thought it produced.
    if detect_version(raw_v1) != VERSION_V1:
        return Reconstruction(
            False, "V1_ROUNDTRIP_FAILED",
            "the rebuilt bytes do not deserialize as a v1 transaction",
            transaction_version=VERSION_V1, max_bytes=limit,
            verification=verification, resource_config=config_desc)

    detail = (f"rebuilt {source_version} ({len(raw)} bytes) as v1 "
              f"({len(raw_v1)} bytes, limit {limit}); "
              f"{len(swap_ixs)} instructions preserved, "
              f"{len(tables)} lookup table(s) materialised and verified, "
              f"{dropped_cb} ComputeBudget instruction(s) moved into the v1 "
              f"resource config")
    verification["address_resolution"] = resolution["detail"]
    return Reconstruction(
        ok=over <= 0,
        status="V1_RECONSTRUCTED" if over <= 0 else "V1_STILL_TOO_LARGE",
        detail=detail,
        transaction_b64=b64,
        transaction_version=VERSION_V1,
        raw_bytes=len(raw_v1),
        max_bytes=limit,
        over_by=over if over > 0 else 0,
        verification=verification,
        resource_config=config_desc,
        resolved_tables=[{"table": t[0], "address_count": len(t[1])}
                         for t in tables],
    )
