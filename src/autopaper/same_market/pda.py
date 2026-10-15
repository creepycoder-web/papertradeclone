"""
Base58 and program-derived addresses, with no third-party dependency.

WHY THIS EXISTS HERE RATHER THAN A SDK
======================================

Deriving a Pump.fun bonding curve from a mint requires ``findProgramAddress``.
The official SDKs do offer it, but they are TypeScript packages, and this
repository is Python-only (no ``package.json``, no ``node_modules``, no JS
build). The task for this harness explicitly says not to introduce a
cross-language build system just to run a measurement.

So the ~30 lines are written out here, in Python, from the public Solana
program-derived-address definition:

    PDA = the first 32 bytes of
          sha256( seed_1 || ... || seed_n || program_id || "ProgramDerivedAddress" )
    that is NOT a valid ed25519 point ("off curve")

There is no secret here and no shortcut: a PDA is a public function of its
seeds and its program. What matters is getting it EXACTLY right, so the
implementation is verified against the official SDK in two independent ways,
both recorded in
``reports/tests/same_market_quote/direct_pump_verification.json``:

* live mainnet account lookups - a wrong derivation returns an account that
  does not exist, or one owned by the wrong program;
* fixed test vectors captured from the official SDK's own
  ``bondingCurvePda`` / ``poolPda`` outputs, asserted in the test suite.

WHY REIMPLEMENT base58 INSTEAD OF IMPORTING ONE
===============================================

``base58`` and ``bs58`` are not dependencies of this project, and adding a
dependency to a read-only measurement harness is a poor trade: it widens the
supply chain of a tool whose entire value proposition is being able to state
exactly what it did and did not touch. The implementation below is the
standard Bitcoin/Solana alphabet and is exercised against the SDK's own
outputs in the tests.

WHAT THIS MODULE CANNOT DO
==========================

It cannot sign, send, or simulate anything. It reads no files, opens no
sockets and holds no keys. It is arithmetic over byte strings.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

#: The Bitcoin/Solana base58 alphabet: no 0, O, I or l.
B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

_B58_INDEX = {ch: i for i, ch in enumerate(B58_ALPHABET)}

#: The marker a program-derived address hashes, per the Solana spec. Getting
#: this wrong yields a plausible-looking address that is not a PDA at all.
_PDA_MARKER = b"ProgramDerivedAddress"

#: A valid ed25519 public key is a point on the curve; ~50% of 32-byte strings
#: are. The canonical on-curve test (rejection sampling by decompression) is
#: what @solana/web3.js implements. Reproducing the full curve maths is
#: unnecessary here: the addresses this harness derives are checked against the
#: official SDK and against live mainnet, so a subtle difference would show up
#: as a failing test rather than as a wrong price. See ``on_curve_note``.
_ON_CURVE_NOTE = (
    "on-curve detection uses the standard field-order/decompression validity "
    "check; every address this module derives is additionally verified against "
    "the official SDK and against a live account owner, so a divergence is a "
    "test failure rather than a silent mis-derivation"
)


class B58Error(ValueError):
    """A base58 string was not decodable."""


class PdaError(ValueError):
    """A program-derived address could not be derived."""


def b58encode(raw: bytes) -> str:
    """Encode bytes as base58."""
    n = int.from_bytes(raw, "big") if raw else 0
    out: list[str] = []
    while n > 0:
        n, rem = divmod(n, 58)
        out.append(B58_ALPHABET[rem])
    pad = 0
    for byte in raw:
        if byte == 0:
            pad += 1
        else:
            break
    return "1" * pad + "".join(reversed(out))


def b58decode(text: str) -> bytes:
    """
    Decode base58 to bytes.

    Raises rather than returning a truncated result: a partially decoded pubkey
    would silently address the wrong account, and this harness reads account
    DATA.
    """
    if not text:
        raise B58Error("empty base58 string")
    n = 0
    for ch in text:
        digit = _B58_INDEX.get(ch)
        if digit is None:
            raise B58Error(f"{ch!r} is not a base58 character")
        n = n * 58 + digit
    body = n.to_bytes((n.bit_length() + 7) // 8, "big")
    pad = 0
    for ch in text:
        if ch == "1":
            pad += 1
        else:
            break
    # A canonical base58 pubkey is 32 bytes. The decoded length is fixed by the
    # decoded number plus leading zeros; if it is not 32 the string is not a
    # pubkey and is refused rather than padded into one.
    out = b"\x00" * pad + body
    return out


def b58decode_pubkey(text: str, *, field: str = "address") -> bytes:
    """Decode and assert exactly 32 bytes."""
    raw = b58decode(text)
    if len(raw) != 32:
        raise PdaError(
            f"{field} {text!r} decodes to {len(raw)} bytes, not the 32 a pubkey "
            "must be"
        )
    return raw


#: ed25519 field prime.
_FIELD_P = 2 ** 255 - 19
#: ed25519 curve constant ``d = -121665/121666 mod p``.
_CURVE_D = (-121665 * pow(121666, -1, _FIELD_P)) % _FIELD_P


def is_on_curve(raw: bytes) -> bool:
    """
    Is this 32-byte value a valid ed25519 point (i.e. NOT a PDA)?

    A program-derived address is defined as a hash that fails the curve
    membership test, so this predicate is what separates a PDA from a normal
    wallet or mint address.

    This follows ``curve25519-dalek``'s ``CompressedEdwardsY::decompress``,
    which is what the reference ``@solana/web3.js`` calls, rather than a
    shortcut. The shortcut was tried first and was WRONG: it accepted
    candidates that dalek rejects, which produced plausible-looking PDAs at
    bump 255 that did not exist on chain. A derivation that is wrong but looks
    right is the single worst failure mode available to this harness, so the
    arithmetic is done properly and pinned to the SDK's own outputs in the
    tests.

    The last bit of a compressed Edwards point is the sign of ``x``, not part
    of ``y``, so it is masked off before the membership test.
    """
    if len(raw) != 32:
        raise PdaError(f"cannot test a {len(raw)}-byte value for curve membership")
    value = int.from_bytes(raw, "little")
    y = value & ((1 << 255) - 1)  # drop the sign bit
    if y >= _FIELD_P:
        return False
    # x^2 = (y^2 - 1) / (d*y^2 + 1); the point is on the curve iff that
    # quotient is a square, i.e. iff the computed root verifies.
    yy = (y * y) % _FIELD_P
    u = (yy - 1) % _FIELD_P
    v = (_CURVE_D * yy + 1) % _FIELD_P
    v3 = (v * v % _FIELD_P) * v % _FIELD_P
    v7 = (v3 * v3 % _FIELD_P) * v % _FIELD_P
    # dalek's ``pow_p58`` is exponent (p-5)/8, NOT (p+3)/8. Using (p+3)/8 here
    # produces a root of the wrong group and rejects valid curve points, which
    # silently inflates the bump and yields a PDA that does not exist on chain.
    x = (u * v3 % _FIELD_P) * pow((u * v7) % _FIELD_P, (_FIELD_P - 5) // 8, _FIELD_P) % _FIELD_P
    check = (v * x % _FIELD_P) * x % _FIELD_P
    if check == u or check == (-u) % _FIELD_P:
        return True
    return False


def find_program_address(seeds: Sequence[bytes], program_id: bytes) -> tuple[bytes, int]:
    """
    Derive a program-derived address and its bump.

    ``findProgramAddress`` in the reference implementation: walk the bump from
    255 downwards and take the first seed set that hashes OFF curve. Returns
    ``(address_bytes, bump)``.
    """
    if len(program_id) != 32:
        raise PdaError(f"program id must be 32 bytes, got {len(program_id)}")
    for bump in range(255, -1, -1):
        candidate = hashlib.sha256()
        for seed in seeds:
            candidate.update(seed)
        candidate.update(bytes([bump]))
        candidate.update(program_id)
        candidate.update(_PDA_MARKER)
        digest = candidate.digest()
        if not is_on_curve(digest):
            return digest, bump
    raise PdaError("no viable bump; the address space is exhausted")


def find_program_address_b58(
    seeds_b58: Sequence[bytes],
    program_id_b58: str,
) -> tuple[str, int]:
    """
    Derive a PDA from base58 program id and raw seed bytes, returning base58.
    """
    program = b58decode_pubkey(program_id_b58, field="program_id")
    address, bump = find_program_address(seeds_b58, program)
    return b58encode(address), bump


__all__ = [
    "B58_ALPHABET",
    "B58Error",
    "PdaError",
    "b58decode",
    "b58decode_pubkey",
    "b58encode",
    "find_program_address",
    "find_program_address_b58",
    "is_on_curve",
]
