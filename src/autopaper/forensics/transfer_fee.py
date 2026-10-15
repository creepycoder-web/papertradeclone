"""
READ-ONLY token transfer-fee / Token-2022 inspection.

PURPOSE
=======
Answer one question with evidence: *does this token have an observable
transfer-fee mechanism, and if so what does it charge?*

WHAT THIS REFUSES TO DO
=======================
* It does NOT assume every memecoin has a tax. Most do not.
* It does NOT fabricate a value. There is no default, no estimate, no
  "typical memecoin tax".
* It does NOT claim to detect an arbitrary custom "buy tax" or "sell tax" from
  the token program alone. A custom program can do anything; the honest answer
  for one is ``UNKNOWN``, never ``NONE``.
* It does NOT touch PnL, L25, sizing or paper-live. It is a reader.

THE THREE STATES, AND WHY THEY ARE DISTINCT
==========================================
``NONE``
    A recognised, fully-decodable token state was inspected and it provably
    carries no transfer fee. This is a positive finding.

``PRESENT``
    A transfer fee was decoded from on-chain account data and the numbers are
    reported. For Token-2022 ``TransferFeeConfig`` this is authoritative: the fee
    is an extension of the token mint itself.

``UNKNOWN``
    The token is not a program this module can decode, the account could not be
    fetched, the extension data is malformed, or the fee configuration could not
    be read. **Unknown is not "no fee".** It is the absence of evidence, and it is
    reported as such.

The distinction between ``NONE`` and ``UNKNOWN`` is the entire value of this
module: collapsing them would let a custom taxed token read as fee-free.

FEE SEMANTICS, STATED PRECISELY
===============================
Token-2022's ``TransferFeeConfig`` defines ONE fee, expressed in basis points of
the transfer amount, applied on transfer. It is symmetric: there is no
buy/sell asymmetry in the extension itself. So:

* ``transfer_fee_bps`` is the decoded extension value.
* ``buy_fee_bps`` / ``sell_fee_bps`` are reported as equal to it ONLY when the
  configuration is provably the plain symmetric Token-2022 transfer fee.
* If a token is a custom program with asymmetric behaviour, buy/sell are
  reported as ``None`` and the status is ``UNKNOWN`` -- because guessing which
  side is taxed is exactly the fabrication this module exists to avoid.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, Sequence

# ---------------------------------------------------------------------------
# well-known addresses (public, constant, no credential involved)
# ---------------------------------------------------------------------------

#: SPL Token program.
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
#: Token-2022 program.
TOKEN_2022_PROGRAM_ID = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
#: SPL Associated Token Account program.
ATA_PROGRAM_ID = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"

# ---------------------------------------------------------------------------
# Token-2022 account geometry
#
# Verified against the reference implementation
# (solana-program/token-2022, extension/mod.rs) rather than guessed:
#
#   byte 0   ..82  : Mint body
#   byte 82  ..165 : zero padding -- mints are padded up to Account::LEN so the
#                    type byte sits at the same offset for mints and accounts
#   byte 165       : account-type discriminator (1 = Mint, 2 = Account)
#   byte 166 ..    : TLV stream, u16 type LE || u16 length LE || value
#
# An unextended mint is EXACTLY 82 bytes and has no type byte and no TLV.
# ---------------------------------------------------------------------------

TOKEN_2022_MINT_BASE_LEN = 82
#: spl-token-2022 `BASE_ACCOUNT_LENGTH` == `Account::LEN` == 165.
BASE_ACCOUNT_LEN = 165
ACCOUNT_TYPE_OFFSET = 165
ACCOUNT_TYPE_MINT = 1
ACCOUNT_TYPE_ACCOUNT = 2
#: TLV entries begin immediately after the account-type byte.
EXTENSION_START = ACCOUNT_TYPE_OFFSET + 1          # == 166
DISCRIMINATOR_SIZE = 4                             # u16 type + u16 length
EXTENSION_TYPE_UNINITIALIZED = 0
EXTENSION_TYPE_TRANSFER_FEE_CONFIG = 1

# ---------------------------------------------------------------------------
# TransferFeeConfig payload -- exactly 108 bytes, fixed width, little-endian.
#
#   0  ..32  transfer_fee_config_authority  OptionalNonZeroPubkey, NO tag byte
#   32 ..64  withdraw_withheld_authority    OptionalNonZeroPubkey, NO tag byte
#   64 ..72  withheld_amount                u64 LE
#   72 ..90  older_transfer_fee             TransferFee (18 bytes)
#   90 ..108 newer_transfer_fee             TransferFee (18 bytes)
#
# and TransferFee is #[repr(C)] { epoch u64, maximum_fee u64,
# transfer_fee_basis_points u16 } == 18 bytes, ALL fixed width.
# There is no Borsh length prefix anywhere in this struct: it is Pod, not
# Borsh. A variable-length read here would silently decode garbage.
# ---------------------------------------------------------------------------

TRANSFER_FEE_CONFIG_SIZE = 108
AUTHORITY_OFFSET = 0
WITHDRAWN_AUTHORITY_OFFSET = 32
WITHHELD_AMOUNT_OFFSET = 64
OLDER_FEE_OFFSET = 72
NEWER_FEE_OFFSET = 90

#: Within a TransferFee: epoch @+0, maximum_fee @+8, bps @+16.
FEE_EPOCH_OFFSET = 0
FEE_MAX_FEE_OFFSET = 8
FEE_BPS_OFFSET = 16
FEE_SIZE = 18

#: Absolute offsets of the OLDER config's fields, kept as named constants so
#: the two configs are never confused with each other.
OLDER_EPOCH_OFFSET = OLDER_FEE_OFFSET + FEE_EPOCH_OFFSET            # 72
OLDER_MAX_FEE_OFFSET = OLDER_FEE_OFFSET + FEE_MAX_FEE_OFFSET        # 80
OLDER_BPS_OFFSET = OLDER_FEE_OFFSET + FEE_BPS_OFFSET                # 88
NEWER_EPOCH_OFFSET = NEWER_FEE_OFFSET + FEE_EPOCH_OFFSET            # 90
NEWER_MAX_FEE_OFFSET = NEWER_FEE_OFFSET + FEE_MAX_FEE_OFFSET        # 98
NEWER_BPS_OFFSET = NEWER_FEE_OFFSET + FEE_BPS_OFFSET                # 106

#: Upper bound on the transfer fee basis points, from spl-token-2022
#: `MAX_FEE_BASIS_POINTS`. `calculate_fee` returns None above this, so a
#: larger decoded value is a protocol impossibility, not a huge fee.
MAX_FEE_BASIS_POINTS = 10_000

#: Basis-point denominator. 10_000 bps == 100%. Alias of the authoritative
#: spl-token-2022 limit, kept as the name used throughout the report code.
BPS_DENOMINATOR = MAX_FEE_BASIS_POINTS

#: Refuse to decode an account larger than this. A Solana account cannot be
#: larger than 10 KB; anything bigger is not a mint and must not be parsed.
MAX_ACCOUNT_BYTES = 10_240


class TransferFeeStatus(str, Enum):
    NONE = "NONE"
    PRESENT = "PRESENT"
    UNKNOWN = "UNKNOWN"


class TokenProgram(str, Enum):
    LEGACY_SPL = "LEGACY_SPL_TOKEN"
    TOKEN_2022 = "TOKEN_2022"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# the injected reader
# ---------------------------------------------------------------------------


class AccountReader(Protocol):
    """
    The ONLY way this module reaches the chain.

    Injected, so every test is fully offline. A production caller would supply
    something that wraps a free read-only RPC; nothing here opens a socket.
    """

    def get_account_info(self, address: str) -> Optional[dict[str, Any]]:
        """
        Raw account, or ``None`` if it does not exist.

        The dict must carry ``data`` (base64 string or bytes) and SHOULD carry
        ``owner`` -- the program that owns the account -- because the owner is
        what identifies SPL Token vs Token-2022 vs a custom program. A reader
        that cannot supply ``owner`` yields ``UNKNOWN``, never ``NONE``.
        """
        ...


class StubAccountReader:
    """In-memory reader. Used by tests and by offline report generation."""

    def __init__(self,
                 accounts: Optional[dict[str, bytes]] = None,
                 owners: Optional[dict[str, str]] = None) -> None:
        self._accounts = dict(accounts or {})
        self._owners = dict(owners or {})
        self.calls: list[str] = []

    def set(self, address: str, data: bytes, owner: Optional[str] = None) -> None:
        self._accounts[address] = data
        if owner is not None:
            self._owners[address] = owner

    def get_account_info(self, address: str) -> Optional[dict[str, Any]]:
        self.calls.append(address)
        raw = self._accounts.get(address)
        if raw is None:
            return None
        out: dict[str, Any] = {
            "data": [base64.b64encode(raw).decode("ascii"), "base64"]}
        if address in self._owners:
            out["owner"] = self._owners[address]
        return out


# ---------------------------------------------------------------------------
# decoding helpers, each independently testable
# ---------------------------------------------------------------------------


def decode_account(
    reader: AccountReader, address: str
) -> tuple[Optional[bytes], Optional[str], Optional[str]]:
    """
    Fetch and base64-decode one account, and read its owning program.

    Returns ``(data, error, owner)``. Any failure is an ERROR STRING, never an
    exception: an unreadable account is an UNKNOWN finding, not a crash. The
    owner is separate from the error because a perfectly readable account can
    still be owned by a program this module cannot classify.
    """
    if not address or not isinstance(address, str):
        return None, "no account address supplied", None
    try:
        info = reader.get_account_info(address)
    except Exception as exc:  # noqa: BLE001 - RPC unavailable is UNKNOWN
        return None, f"account fetch failed: {type(exc).__name__}: {exc}", None
    if not info:
        return None, "account not found", None
    owner = info.get("owner")
    owner = owner if isinstance(owner, str) and owner else None
    data = info.get("data")
    if isinstance(data, str):
        try:
            return base64.b64decode(data, validate=True), None, owner
        except (binascii.Error, ValueError) as exc:
            return None, f"account data is not valid base64: {exc}", owner
    if isinstance(data, (list, tuple)) and data:
        blob = data[0]
        if isinstance(blob, str):
            try:
                return base64.b64decode(blob, validate=True), None, owner
            except (binascii.Error, ValueError) as exc:
                return None, f"account data is not valid base64: {exc}", owner
        if isinstance(blob, bytes):
            return blob, None, owner
    return None, ("account data is missing or in an unrecognised encoding",
                  owner)


def identify_token_program(owner: Optional[str]) -> TokenProgram:
    """Map the account owner to a token program. Unknown owner -> UNKNOWN."""
    if not isinstance(owner, str) or not owner:
        return TokenProgram.UNKNOWN
    if owner == TOKEN_PROGRAM_ID:
        return TokenProgram.LEGACY_SPL
    if owner == TOKEN_2022_PROGRAM_ID:
        return TokenProgram.TOKEN_2022
    return TokenProgram.UNKNOWN


def normalise_bps(raw: Any) -> Optional[int]:
    """
    Coerce a decoded basis-point count to a non-negative int.

    Returns ``None`` for anything that is not a plain non-negative integer.
    The wire type is a raw little-endian ``u16``, but a value above
    ``MAX_FEE_BASIS_POINTS`` (10_000) is a protocol impossibility -- the
    program itself refuses to compute a fee above it -- so it is treated as
    malformed rather than clamped to 100%.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if not isinstance(raw, int):
        return None
    if raw < 0 or raw > MAX_FEE_BASIS_POINTS:
        return None
    return raw


def normalise_lamports(raw: Any) -> Optional[int]:
    """Coerce a decoded lamport amount to a non-negative int, else ``None``."""
    if raw is None or isinstance(raw, bool) or not isinstance(raw, int):
        return None
    return raw if raw >= 0 else None


def find_transfer_fee_extension(
    mint_data: bytes,
) -> tuple[Optional[bytes], Optional[str]]:
    """
    Locate the ``TransferFeeConfig`` extension inside a Token-2022 mint account.

    Returns ``(payload, error)``. The three outcomes are kept distinct because
    they mean very different things:

    * ``(payload, None)`` -- the extension is present and readable.
    * ``(None, None)``    -- the mint was inspected and provably has NO
      ``TransferFeeConfig``. This is the only path to a ``NONE`` verdict.
    * ``(None, error)``   -- the account could not be interpreted. This is
      ``UNKNOWN``, never ``NONE``.

    The account-type byte is checked rather than trusted. Without that check a
    165-byte legacy token ACCOUNT could be misread as a mint, and its trailing
    bytes could be parsed as a fabricated fee configuration.
    """
    n = len(mint_data)
    if n > MAX_ACCOUNT_BYTES:
        return None, (
            f"account is {n} bytes, larger than a Solana account can be; "
            f"refusing to parse it as a mint"
        )

    # An unextended mint is exactly 82 bytes: provably no extension space.
    if n == TOKEN_2022_MINT_BASE_LEN:
        return None, None

    # The account-type byte is checked FIRST, before any length or TLV
    # reasoning, so the diagnostic names the actual problem. Rejecting a token
    # ACCOUNT that merely happens to be long enough is the whole point.
    if n <= ACCOUNT_TYPE_OFFSET:
        return None, (
            f"account is {n} bytes, too short to carry the account-type byte "
            f"at {ACCOUNT_TYPE_OFFSET}, yet is not the "
            f"{TOKEN_2022_MINT_BASE_LEN}-byte unextended mint shape"
        )
    atype = mint_data[ACCOUNT_TYPE_OFFSET]
    if atype != ACCOUNT_TYPE_MINT:
        kind = ("account" if atype == ACCOUNT_TYPE_ACCOUNT
                else f"unknown type {atype}")
        return None, (
            f"account-type byte at {ACCOUNT_TYPE_OFFSET} is {atype} ({kind}), "
            f"not {ACCOUNT_TYPE_MINT} (mint); refusing to parse a non-mint as a "
            f"mint"
        )

    # Valid type byte but not one byte of TLV: there is no extension list, so
    # there is provably no TransferFeeConfig.
    if n == EXTENSION_START:
        return None, None

    # A mint must be zero-padded between 82 and 165. Checking it costs nothing
    # and rejects an account whose bytes merely happen to line up.
    pad = mint_data[TOKEN_2022_MINT_BASE_LEN:ACCOUNT_TYPE_OFFSET]
    if any(pad):
        return None, (
            "non-zero bytes in the mint padding region (82..165); this is not "
            "a well-formed extended Token-2022 mint"
        )

    pos = EXTENSION_START
    while pos + DISCRIMINATOR_SIZE <= n:
        etype = int.from_bytes(mint_data[pos:pos + 2], "little")
        elen = int.from_bytes(mint_data[pos + 2:pos + 4], "little")
        body = pos + DISCRIMINATOR_SIZE
        if etype == EXTENSION_TYPE_UNINITIALIZED and elen == 0:
            # Zero padding after the last real entry. End of the list.
            return None, None
        if body + elen > n:
            return None, (
                f"extension type {etype} declares {elen} bytes but only "
                f"{n - body} remain; the account is truncated or malformed"
            )
        if etype == EXTENSION_TYPE_TRANSFER_FEE_CONFIG:
            if elen != TRANSFER_FEE_CONFIG_SIZE:
                return None, (
                    f"TransferFeeConfig declares {elen} bytes but the extension "
                    f"is fixed at {TRANSFER_FEE_CONFIG_SIZE}; a different size "
                    f"means this is not the layout this module can decode"
                )
            return mint_data[body:body + elen], None
        pos = body + elen
    return None, None


def _u64(payload: bytes, off: int) -> int:
    return int.from_bytes(payload[off:off + 8], "little")


def _decode_fee_slot(payload: bytes, base: int) -> dict[str, Any]:
    """One 18-byte ``TransferFee``: epoch, maximum_fee, basis points."""
    epoch = _u64(payload, base + FEE_EPOCH_OFFSET)
    max_fee = normalise_lamports(_u64(payload, base + FEE_MAX_FEE_OFFSET))
    raw_bps = int.from_bytes(
        payload[base + FEE_BPS_OFFSET:base + FEE_BPS_OFFSET + 2], "little")
    return {
        "epoch": epoch,
        "maximum_fee_lamports": max_fee,
        "transfer_fee_basis_points": normalise_bps(raw_bps),
        "basis_points_raw": raw_bps,
    }


def active_fee_slot(dec: dict[str, Any], epoch: int) -> Optional[str]:
    """
    Which of the two configurations is live at ``epoch``.

    This mirrors spl-token-2022's ``get_epoch_fee``: the newer configuration
    takes effect once the cluster reaches its epoch, and the older one applies
    until then. Reporting only the newer configuration -- the intuitive but
    wrong reading -- overstates the fee during a pending change and understates
    it the moment a change is scheduled and still in its notice period.
    """
    newer = dec.get("newer") or {}
    older = dec.get("older") or {}
    new_epoch = newer.get("epoch")
    if new_epoch is None or epoch >= new_epoch:
        return "newer"
    return "older"


def other_config_note(dec: dict[str, Any], active: str) -> str:
    """
    Describe the configuration that is NOT active, so a rate reader can see
    whether the rate is about to change.

    This is the difference between reporting the fee the chain is charging and
    reporting the fee somebody intends to charge later.
    """
    other = "older" if active == "newer" else "newer"
    slot = dec.get(other) or {}
    bps = slot.get("transfer_fee_basis_points")
    epoch = slot.get("epoch")
    if bps is None or epoch is None:
        return ""
    if active == "older":
        return (f" A HIGHER-OR-DIFFERENT configuration is already scheduled: "
                f"{other}={bps} bps takes effect at epoch {epoch}.")
    return f" The {other} configuration is {bps} bps and is no longer applied."


def decode_transfer_fee_config(payload: bytes) -> dict[str, Any]:
    """
    Decode a ``TransferFeeConfig`` extension payload.

    Returns a dict that always has ``ok``. On ``ok=False`` the reason is
    ``error`` and every numeric field is ``None`` -- a malformed extension must
    never yield a half-decoded fee that could later be used.

    BOTH epoch configurations are returned. Selecting between them requires the
    cluster epoch, which is not available from account data alone, so the choice
    is left to the caller (see :func:`active_fee_slot`) rather than guessed.
    """
    if len(payload) < TRANSFER_FEE_CONFIG_SIZE:
        return {"ok": False, "error": (
            f"TransferFeeConfig payload is {len(payload)} bytes, expected "
            f"exactly {TRANSFER_FEE_CONFIG_SIZE}"),
            "transfer_fee_bps": None, "maximum_fee_lamports": None}

    older = _decode_fee_slot(payload, OLDER_FEE_OFFSET)
    newer = _decode_fee_slot(payload, NEWER_FEE_OFFSET)

    for label, slot in (("older", older), ("newer", newer)):
        if slot["transfer_fee_basis_points"] is None:
            return {"ok": False, "error": (
                f"{label} transfer fee basis points "
                f"{slot['basis_points_raw']} is outside the valid range "
                f"0..{MAX_FEE_BASIS_POINTS}; treating the extension as "
                f"malformed"),
                "transfer_fee_bps": None, "maximum_fee_lamports": None}
        if slot["maximum_fee_lamports"] is None:
            return {"ok": False, "error": (
                f"{label} maximum_fee is not a valid u64; treating the "
                f"extension as malformed"),
                "transfer_fee_bps": None, "maximum_fee_lamports": None}

    withheld = normalise_lamports(_u64(payload, WITHHELD_AMOUNT_OFFSET))
    cfg_auth = _decode_optional_pubkey(payload, AUTHORITY_OFFSET)
    wdr_auth = _decode_optional_pubkey(payload, WITHDRAWN_AUTHORITY_OFFSET)

    return {
        "ok": True,
        "error": None,
        "older": older,
        "newer": newer,
        "transfer_fee_bps": newer["transfer_fee_basis_points"],
        "maximum_fee_lamports": newer["maximum_fee_lamports"],
        "maximum_fee_sol": newer["maximum_fee_lamports"] / 1e9,
        "withheld_amount_lamports": withheld,
        "transfer_fee_config_authority": cfg_auth,
        "withdraw_withheld_authority": wdr_auth,
        "note": (
            "transfer_fee_bps/maximum_fee_lamports above default to the NEWER "
            "configuration for readability only; they are NOT necessarily the "
            "active rate. Use active_fee_slot(dec, epoch) to select correctly."),
    }


_B58_ALPHABET = ("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")


def _b58_encode(raw: bytes) -> str:
    """
    Bitcoin/Solana base58, for DISPLAY only.

    Implemented here rather than imported so this module carries no chain-library
    dependency: a read-only forensic reader must not be able to reach the
    Solana execution layer, and ``tests/test_safety.py`` enforces exactly that.
    The raw bytes remain the source of truth -- this is a rendering.
    """
    leading = len(raw) - len(raw.lstrip(b"\x00"))
    num = int.from_bytes(raw, "big")
    out = ""
    while num > 0:
        num, rem = divmod(num, 58)
        out = _B58_ALPHABET[rem] + out
    return "1" * leading + (out or "")


def _decode_optional_pubkey(payload: bytes, off: int) -> Optional[str]:
    """
    An ``OptionalNonZeroPubkey``: 32 bytes with NO tag byte, where all-zero
    means None. Rendered base58 for display; the raw bytes are what matter.
    """
    raw = payload[off:off + 32]
    if len(raw) < 32 or not any(raw):
        return None
    return _b58_encode(raw)


# ---------------------------------------------------------------------------
# the public result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransferFeeReport:
    """
    The structured answer. Field names match the requested contract.

    ``buy_fee_bps`` / ``sell_fee_bps`` are populated ONLY when a symmetric fee
    is provable. A custom program leaves them ``None`` and sets
    ``asymmetric_unprovable`` so the distinction survives serialisation.
    """

    mint: str
    transfer_fee_status: TransferFeeStatus
    token_program: TokenProgram
    transfer_fee_bps: Optional[int]
    buy_fee_bps: Optional[int]
    sell_fee_bps: Optional[int]
    max_fee_lamports: Optional[int]
    max_fee_sol: Optional[float]
    evidence_source: str
    state_slot: Optional[int]
    timestamp: Optional[float]
    reason: str
    asymmetric_unprovable: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)

    # -- Token-2022 epoch-gated configurations ---------------------------
    #: Both configurations are always retained, because only the cluster epoch
    #: says which one is live.
    older_transfer_fee_bps: Optional[int] = None
    newer_transfer_fee_bps: Optional[int] = None
    older_max_fee_lamports: Optional[int] = None
    newer_max_fee_lamports: Optional[int] = None
    active_config: Optional[str] = None
    active_config_epoch: Optional[int] = None

    # -- readable on-chain state -----------------------------------------
    transfer_fee_config_authority: Optional[str] = None
    withdraw_withheld_authority: Optional[str] = None
    withheld_amount_lamports: Optional[int] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mint": self.mint,
            "transfer_fee_status": self.transfer_fee_status.value,
            "token_program": self.token_program.value,
            "transfer_fee_bps": self.transfer_fee_bps,
            "buy_fee_bps": self.buy_fee_bps,
            "sell_fee_bps": self.sell_fee_bps,
            "max_fee_lamports": self.max_fee_lamports,
            "max_fee_sol": self.max_fee_sol,
            "evidence_source": self.evidence_source,
            "state_slot": self.state_slot,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "asymmetric_unprovable": self.asymmetric_unprovable,
            "older_transfer_fee_bps": self.older_transfer_fee_bps,
            "newer_transfer_fee_bps": self.newer_transfer_fee_bps,
            "older_max_fee_lamports": self.older_max_fee_lamports,
            "newer_max_fee_lamports": self.newer_max_fee_lamports,
            "active_config": self.active_config,
            "active_config_epoch": self.active_config_epoch,
            "transfer_fee_config_authority": self.transfer_fee_config_authority,
            "withdraw_withheld_authority": self.withdraw_withheld_authority,
            "withheld_amount_lamports": self.withheld_amount_lamports,
            "warnings": list(self.warnings),
            "is_not_a_tax_estimate": (
                "this is on-chain evidence only; it is not an estimate of any "
                "custom buy/sell tax, and UNKNOWN must not be read as zero"
            ),
        }


def inspect_transfer_fee(
    mint: str,
    reader: AccountReader,
    *,
    now: Optional[float] = None,
    state_slot: Optional[int] = None,
    epoch: Optional[int] = None,
) -> TransferFeeReport:
    """
    Inspect one mint, read-only, and report what the chain actually says.

    Never raises. Every failure path is an explicit state, because a detector
    that crashes on a malformed account is worse than one that says UNKNOWN.

    ``epoch`` is the cluster epoch. It is REQUIRED to report a single active
    fee rate, because Token-2022 stores two configurations and the newer one
    only takes effect at its own epoch. Without ``epoch`` this returns both and
    leaves ``transfer_fee_bps`` as ``None`` rather than picking the one that
    happens to be listed first.
    """
    # 1. resolve the mint account
    data, err, owner = decode_account(reader, mint)
    if err is not None:
        return TransferFeeReport(
            mint=mint, transfer_fee_status=TransferFeeStatus.UNKNOWN,
            token_program=TokenProgram.UNKNOWN, transfer_fee_bps=None,
            buy_fee_bps=None, sell_fee_bps=None, max_fee_lamports=None,
            max_fee_sol=None, evidence_source="rpc_getAccountInfo",
            state_slot=state_slot, timestamp=now,
            reason=f"mint account unreadable: {err}",
            warnings=("the token may still have a transfer fee; this is the "
                      "absence of evidence, not evidence of absence",),
        )

    program = identify_token_program(owner)

    if program is TokenProgram.LEGACY_SPL:
        # A legacy SPL mint has no extension space at all, so there is provably
        # no Token-2022 transfer fee. This is a POSITIVE finding.
        return TransferFeeReport(
            mint=mint, transfer_fee_status=TransferFeeStatus.NONE,
            token_program=program, transfer_fee_bps=None, buy_fee_bps=None,
            sell_fee_bps=None, max_fee_lamports=None, max_fee_sol=None,
            evidence_source="rpc_getAccountInfo.owner",
            state_slot=state_slot, timestamp=now,
            reason=("legacy SPL Token mint: the program defines no transfer-fee "
                    "extension, so no on-chain transfer fee exists"),
        )

    if program is TokenProgram.TOKEN_2022:
        payload, ext_err = find_transfer_fee_extension(data)
        if ext_err is not None:
            return TransferFeeReport(
                mint=mint, transfer_fee_status=TransferFeeStatus.UNKNOWN,
                token_program=program, transfer_fee_bps=None,
                buy_fee_bps=None, sell_fee_bps=None, max_fee_lamports=None,
                max_fee_sol=None, evidence_source="rpc_getAccountInfo.extension",
                state_slot=state_slot, timestamp=now,
                reason=f"extension list unreadable: {ext_err}",
                warnings=("a Token-2022 mint with an unreadable extension list "
                          "cannot be reported as fee-free",),
            )
        if payload is None:
            return TransferFeeReport(
                mint=mint, transfer_fee_status=TransferFeeStatus.NONE,
                token_program=program, transfer_fee_bps=None,
                buy_fee_bps=None, sell_fee_bps=None, max_fee_lamports=None,
                max_fee_sol=None,
                evidence_source="rpc_getAccountInfo.extension",
                state_slot=state_slot, timestamp=now,
                reason=("Token-2022 mint inspected and carries NO "
                        "TransferFeeConfig extension"),
            )
        dec = decode_transfer_fee_config(payload)
        if not dec["ok"]:
            return TransferFeeReport(
                mint=mint, transfer_fee_status=TransferFeeStatus.UNKNOWN,
                token_program=program, transfer_fee_bps=None,
                buy_fee_bps=None, sell_fee_bps=None, max_fee_lamports=None,
                max_fee_sol=None, evidence_source="rpc_getAccountInfo.extension",
                state_slot=state_slot, timestamp=now,
                reason=f"TransferFeeConfig malformed: {dec['error']}",
                warnings=("a malformed fee extension must not be read as a zero "
                          "fee; reported UNKNOWN",),
            )
        older, newer = dec["older"], dec["newer"]
        common = dict(
            mint=mint, token_program=program,
            older_transfer_fee_bps=older["transfer_fee_basis_points"],
            newer_transfer_fee_bps=newer["transfer_fee_basis_points"],
            older_max_fee_lamports=older["maximum_fee_lamports"],
            newer_max_fee_lamports=newer["maximum_fee_lamports"],
            transfer_fee_config_authority=dec["transfer_fee_config_authority"],
            withdraw_withheld_authority=dec["withdraw_withheld_authority"],
            withheld_amount_lamports=dec["withheld_amount_lamports"],
            evidence_source="rpc_getAccountInfo.extension",
            state_slot=state_slot, timestamp=now)

        # Both configs are zero -> provably fee-free regardless of epoch.
        if (older["transfer_fee_basis_points"] == 0
                and newer["transfer_fee_basis_points"] == 0):
            return TransferFeeReport(
                transfer_fee_status=TransferFeeStatus.NONE,
                transfer_fee_bps=0, buy_fee_bps=0, sell_fee_bps=0,
                max_fee_lamports=newer["maximum_fee_lamports"],
                max_fee_sol=newer["maximum_fee_lamports"] / 1e9,
                reason=("Token-2022 TransferFeeConfig present but BOTH the "
                        "older and newer basis-points fees are 0, so no fee is "
                        "charged in any epoch"),
                **common)

        # Which config is live depends on the cluster epoch. Without it, do not
        # guess -- report both and say so.
        if epoch is None:
            return TransferFeeReport(
                transfer_fee_status=TransferFeeStatus.PRESENT,
                transfer_fee_bps=None, buy_fee_bps=None, sell_fee_bps=None,
                max_fee_lamports=None, max_fee_sol=None,
                reason=(
                    f"Token-2022 TransferFeeConfig decoded: older="
                    f"{older['transfer_fee_basis_points']} bps "
                    f"(epoch {older['epoch']}, cap "
                    f"{older['maximum_fee_lamports']} lamports), newer="
                    f"{newer['transfer_fee_basis_points']} bps "
                    f"(epoch {newer['epoch']}, cap "
                    f"{newer['maximum_fee_lamports']} lamports). Which one is "
                    "ACTIVE depends on the cluster epoch, which was not "
                    "supplied, so no single rate is reported."),
                warnings=(
                    "two epoch-gated configurations are present; the active "
                    "rate needs the cluster epoch (pass epoch=)",
                    "no cluster epoch supplied: transfer_fee_bps is left None "
                    "rather than defaulting to the newer configuration",
                ),
                **common)

        slot = active_fee_slot(dec, epoch)
        live = dec[slot]
        bps = live["transfer_fee_basis_points"]
        max_fee = live["maximum_fee_lamports"]
        pending = other_config_note(dec, slot)
        return TransferFeeReport(
            transfer_fee_status=TransferFeeStatus.PRESENT,
            transfer_fee_bps=bps,
            # Symmetric by construction: Token-2022 defines ONE transfer fee
            # and no side asymmetry, so both sides equal the decoded value.
            buy_fee_bps=bps, sell_fee_bps=bps,
            max_fee_lamports=max_fee, max_fee_sol=max_fee / 1e9,
            active_config=slot, active_config_epoch=epoch,
            reason=(f"Token-2022 TransferFeeConfig decoded: {bps} bps "
                    f"({bps / BPS_DENOMINATOR:.2%}) per transfer, capped at "
                    f"{max_fee} lamports, from the {slot} configuration "
                    f"(epoch {live['epoch']} <= cluster epoch {epoch}). "
                    f"Symmetric: the extension defines one fee with no buy/sell "
                    f"asymmetry.{pending}"),
            **common)

    # Unknown owner: a custom program. It MAY tax transfers; nothing in the
    # program identity says either way.
    return TransferFeeReport(
        mint=mint, transfer_fee_status=TransferFeeStatus.UNKNOWN,
        token_program=TokenProgram.UNKNOWN, transfer_fee_bps=None,
        buy_fee_bps=None, sell_fee_bps=None, max_fee_lamports=None,
        max_fee_sol=None, evidence_source="rpc_getAccountInfo.owner",
        state_slot=state_slot, timestamp=now,
        reason=(f"account owner {owner!r} is not a recognised SPL Token or "
                "Token-2022 program; custom programs can impose any transfer "
                "behaviour, including an asymmetric buy/sell tax, and that "
                "cannot be determined from program identity alone"),
        asymmetric_unprovable=True,
        warnings=("a custom token program may charge an arbitrary or "
                  "asymmetric tax; treat as UNKNOWN, never as zero",),
    )


def inspect_many(
    mints: Sequence[str],
    reader: AccountReader,
    *,
    now: Optional[float] = None,
    state_slot: Optional[int] = None,
    epoch: Optional[int] = None,
) -> list[TransferFeeReport]:
    """Batch, in input order. One token's failure never affects another's."""
    return [inspect_transfer_fee(m, reader, now=now, state_slot=state_slot,
                                 epoch=epoch)
            for m in mints]
