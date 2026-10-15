"""Deterministic Solana transaction fixtures for the version-aware tests.

Everything here is OFFLINE and built with ``solders``: no network, no RPC, no
funded key. Ephemeral in-memory keypairs only.

Why this module exists
----------------------
A serialized transaction is now judged by its DESERIALIZED version, so a test
that hands the executor an arbitrary string like ``"AQAB"`` is no longer
modelling anything real -- the size gate must refuse it, correctly. These
helpers build transactions that are genuinely valid, so the tests exercise the
production paths rather than tripping over a fixture that could never occur.

The builders are parameterised by VERSION and by TARGET SIZE, because the whole
point of the V1 work is that a transaction's verdict depends on which of those
it is:

    legacy  -> 1232 byte ceiling
    v0      -> 1232 byte ceiling
    v1      -> 4096 byte ceiling
"""

from __future__ import annotations

import base64
from typing import Optional

from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import (
    Message,
    MessageV0,
    MessageV1,
    TransactionConfig,
)
from solders.pubkey import Pubkey
from solders.system_program import TransferParams, transfer
from solders.transaction import Transaction, VersionedTransaction

from autopaper.execution.txcodec import (
    MAX_BYTES_LEGACY,
    MAX_BYTES_V1,
    VERSION_LEGACY,
    VERSION_V0,
    VERSION_V1,
)

#: A deterministic stand-in for a pool/mint account. Any 32-byte value works;
#: nothing here is funded or expected to exist on chain.
FAKE_ACCOUNT = "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"
FAKE_ACCOUNT_2 = "2q5g2nx9gJdpxCK6JtnKvq9rPAa6fQLkRCQMFvm1STNK"
MEMO_PROGRAM = "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"

ZERO_SIGNATURE = bytes(64)


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _deterministic_pubkey(seed: str) -> Pubkey:
    """A stable, non-funded pubkey derived from ``seed``.

    Deterministic rather than random so a failing size assertion is
    reproducible, and a SHA-256 digest rather than a real curve point so
    nothing can be mistaken for an account that exists on chain.
    """
    import hashlib
    return Pubkey.from_bytes(hashlib.sha256(seed.encode()).digest())


def filler_instructions(count: int, *, accounts_per_ix: int = 1,
                        data_len: int = 8) -> list:
    """``count`` distinct, non-ComputeBudget instructions of a chosen size.

    Each instruction carries its own unique account keys and a unique data blob,
    so the serialized size grows predictably and no two instructions collapse
    onto the same account key. Memo instructions are used because the program
    id is a single shared constant, keeping the account list predictable.

    They are never EXECUTED: this is a byte-shape fixture for size and codec
    tests, and nothing here is funded.
    """
    out = []
    for i in range(count):
        metas = [AccountMeta(_deterministic_pubkey(f"fx-{i}-{j}"), False, True)
                 for j in range(accounts_per_ix)]
        out.append(Instruction(
            program_id=Pubkey.from_string(MEMO_PROGRAM),
            data=bytes([1, i & 0xFF, (i >> 8) & 0xFF, 0, 0])
                  + bytes((i + j) & 0xFF for j in range(max(0, data_len - 5))),
            accounts=metas,
        ))
    return out


def padded_instructions(target_bytes: int) -> list:
    """Instructions whose combined serialized size approaches ``target_bytes``.

    Searches over instruction count and per-instruction size. Used so boundary
    tests do not hard-code a byte count that a library upgrade could move.
    """
    if target_bytes <= 0:
        return filler_instructions(1)
    for accounts_per_ix in (1, 2, 3):
        for data_len in (8, 32, 96, 192, 256):
            for count in range(1, 60):
                candidate = filler_instructions(
                    count, accounts_per_ix=accounts_per_ix, data_len=data_len)
                # 40 bytes/ix is a lower bound on the per-instruction cost; stop
                # widening this shape once the bound is already past the target.
                if 40 * count > target_bytes:
                    break
                yield_probe = candidate
                if sum(32 * accounts_per_ix + data_len + 6
                       for _ in range(count)) >= target_bytes:
                    return yield_probe
    return filler_instructions(1, accounts_per_ix=3, data_len=256)


def zero_signatures(raw: bytes) -> bytes:
    """Blank every signature slot, leaving an UNSIGNED transaction.

    This is how an aggregator hands a transaction to a client: built and signed
    by the venue's throwaway key, then the signature bytes are replaced with
    zeroes so only the wallet's own key can produce a valid transaction.
    ``solders`` will not construct a versioned transaction from raw byte blobs,
    so the fixtures are signed with the in-memory keypair and blanked here.

    Works for all three layouts because it is driven by the version detected
    from the bytes, which is exactly the property under test elsewhere.
    """
    from autopaper.execution.txcodec import (
        SIGNATURE_SIZE,
        message_bytes_and_signature_slots,
    )
    _message, count, version = message_bytes_and_signature_slots(raw)
    if version == VERSION_V1:
        first = len(raw) - SIGNATURE_SIZE * count
        return raw[:first] + ZERO_SIGNATURE * count
    # legacy / v0: the payer's slot is the first, after the shortvec count.
    first = 1 if raw[0] < 0x80 else 1 + (raw[0] & 0x7F)
    return raw[:first] + ZERO_SIGNATURE * count + raw[first + SIGNATURE_SIZE * count:]


def build_legacy(payer: Keypair, *, instructions: Optional[list] = None,
                 blockhash: Optional[Hash] = None) -> bytes:
    """A real, unsigned LEGACY transaction."""
    ixs = instructions if instructions is not None else filler_instructions(2)
    message = Message.new_with_blockhash(
        ixs, payer.pubkey(), blockhash or Hash.default())
    return bytes(Transaction.new_unsigned(message))


def build_v0(payer: Keypair, *, instructions: Optional[list] = None,
             blockhash: Optional[Hash] = None) -> bytes:
    """A real, unsigned v0 transaction with no address lookup tables."""
    ixs = instructions if instructions is not None else filler_instructions(2)
    message = MessageV0.try_compile(payer.pubkey(), ixs, [],
                                    blockhash or Hash.default())
    return zero_signatures(bytes(VersionedTransaction(message, [payer])))


def build_v1(payer: Keypair, *, instructions: Optional[list] = None,
             blockhash: Optional[Hash] = None, config=None) -> bytes:
    """A real, unsigned v1 transaction (SIMD-0385).

    A v1 message REQUIRES a non-empty resource config: solders rejects an
    all-null one with "invalid transaction config mask" and the node rejects a
    swap whose loaded-accounts budget is unset with
    ``MaxLoadedAccountsDataSizeExceeded``. Both were verified against mainnet.
    """
    ixs = instructions if instructions is not None else filler_instructions(2)
    cfg = config or TransactionConfig(
        compute_unit_limit=200_000,
        priority_fee=10_000,
        loaded_accounts_data_size_limit=64 * 1024,
        heap_size=32_768,
    )
    message = MessageV1.try_compile(payer.pubkey(), ixs,
                                    blockhash or Hash.default(), cfg)
    return zero_signatures(bytes(VersionedTransaction(message, [payer])))


def build(version: str, payer: Optional[Keypair] = None,
          *, instructions: Optional[list] = None,
          blockhash: Optional[Hash] = None) -> bytes:
    """Build an unsigned transaction of ``version``."""
    kp = payer or Keypair()
    if version == VERSION_LEGACY:
        return build_legacy(kp, instructions=instructions, blockhash=blockhash)
    if version == VERSION_V0:
        return build_v0(kp, instructions=instructions, blockhash=blockhash)
    if version == VERSION_V1:
        return build_v1(kp, instructions=instructions, blockhash=blockhash)
    raise ValueError(f"unknown transaction version {version!r}")


def build_v0_with_alt(payer: Keypair, table_addresses: dict,
                      *, blockhash: Optional[Hash] = None,
                      n_instructions: Optional[int] = None) -> bytes:
    """A real, unsigned v0 transaction that REFERENCES an address lookup table.

    ``table_addresses`` maps a lookup-table address to the address list stored
    in that table, exactly as ``getAddressLookupTable`` (or the ``jsonParsed``
    account parse) would return it. The addresses are pushed into a table far
    enough up the account list that the compiler is forced to compress them,
    which is the only way to obtain a v0 message whose indices genuinely point
    into loaded addresses.

    The table account itself is an ordinary in-memory
    :class:`~solders.address_lookup_table_account.AddressLookupTableAccount`; it
    is never fetched, funded or created, so nothing here touches the network.
    """
    from solders.address_lookup_table_account import AddressLookupTableAccount

    ixs: list = []
    for table_addr, table_list in table_addresses.items():
        # Reference the TABLE's own addresses as account metas, so the compiler
        # has real compression work to do: these keys are the ones that end up
        # inside the lookup table rather than in the static key list. The first
        # ``n_instructions`` entries are referenced; any tail is deliberately
        # left unreferenced, so a test can corrupt an address that CANNOT change
        # what the chain does and prove the check is not fooled by it.
        usable = (table_list if n_instructions is None
                  else table_list[:n_instructions])
        for slot, address in enumerate(usable):
            ixs.append(Instruction(
                program_id=Pubkey.from_string(MEMO_PROGRAM),
                data=bytes([9, slot & 0xFF]),
                accounts=[AccountMeta(Pubkey.from_string(address), False, True)],
            ))
    if not ixs:
        raise AssertionError("a lookup table fixture needs at least one address")
    accounts = [AddressLookupTableAccount(
        Pubkey.from_string(addr),
        [Pubkey.from_string(a) for a in addrs])
        for addr, addrs in table_addresses.items()]

    message = MessageV0.try_compile(payer.pubkey(), ixs, accounts,
                                    blockhash or Hash.default())
    raw = bytes(VersionedTransaction(message, [payer]))
    lookups = list(message.address_table_lookups)
    if not lookups:
        raise AssertionError(
            "the compiler compressed nothing; the fixture did not exercise an "
            "address lookup table")
    return zero_signatures(raw)


def alt_accounts(table_addresses: dict) -> list:
    """The :class:`AddressLookupTableAccount` list for ``table_addresses``."""
    from solders.address_lookup_table_account import AddressLookupTableAccount
    return [AddressLookupTableAccount(
        Pubkey.from_string(addr),
        [Pubkey.from_string(a) for a in addrs])
        for addr, addrs in table_addresses.items()]


def build_exact(version: str, nbytes: int,
                payer: Optional[Keypair] = None) -> bytes:
    """A real transaction of ``version`` whose serialization is exactly
    ``nbytes``.

    The byte-size boundary tests need a transaction that sits exactly ON a
    limit, not merely near it: "1232 accepted" and "1233 rejected" are different
    claims, and a fixture that overshoots by a whole account key cannot
    distinguish a boundary from a coincidence.

    One filler instruction's data blob is swept one byte at a time, which moves
    the serialized size by one byte per step, so every reachable size is hit
    exactly. Sizes below the minimum a single instruction can produce are raised
    with a clear error rather than silently rounded.
    """
    kp = payer or Keypair()
    smallest = len(build(version, kp, instructions=filler_instructions(
        1, data_len=1)))
    if nbytes < smallest:
        raise AssertionError(
            f"a {version} transaction cannot be {nbytes} bytes; the smallest "
            f"is {smallest}")
    for data_len in range(1, nbytes + 8):
        raw = build(version, kp,
                    instructions=filler_instructions(1, data_len=data_len))
        if len(raw) == nbytes:
            return raw
        if len(raw) > nbytes:
            break
    # Second pass: add accounts/instructions to close a gap the data sweep
    # cannot reach (a compact-u16 length prefix can jump a few bytes at 128).
    for accounts_per_ix in (1, 2, 3):
        for count in range(1, 62):
            for data_len in range(1, 300):
                raw = build(version, kp, instructions=filler_instructions(
                    count, accounts_per_ix=accounts_per_ix, data_len=data_len))
                if len(raw) == nbytes:
                    return raw
                if len(raw) > nbytes:
                    break
    raise AssertionError(
        f"could not build a {version} transaction of exactly {nbytes} bytes")


def build_wide_data(version: str, target_bytes: int,
                    payer: Optional[Keypair] = None) -> bytes:
    """A transaction that is large because of instruction DATA, not accounts.

    v1 caps distinct keys at 64, so a fixture that reaches a few thousand bytes
    by adding accounts trips the address bound instead of the size bound. This
    one grows a single instruction's data blob instead, so a test can exceed a
    size ceiling without conflating the two.
    """
    kp = payer or Keypair()
    smallest = len(build(version, kp, instructions=filler_instructions(
        1, data_len=1)))
    if target_bytes < smallest:
        raise AssertionError(
            f"a {version} transaction cannot be {target_bytes} bytes; the "
            f"smallest is {smallest}")
    for data_len in range(1, target_bytes + 8):
        raw = build(version, kp,
                    instructions=filler_instructions(1, data_len=data_len))
        if len(raw) >= target_bytes:
            return raw
    raise AssertionError(
        f"could not build a {version} transaction of {target_bytes} bytes")


def build_of_size(version: str, target_bytes: int,
                  payer: Optional[Keypair] = None) -> bytes:
    """The smallest transaction of ``version`` that is at least ``target_bytes``.

    Fine granularity first: a single filler instruction whose data blob is swept
    one byte at a time, so a transaction can be placed exactly on a boundary
    rather than overshooting it by a whole account key. Only if that cannot
    reach the target does it fall back to adding instructions and accounts,
    which move the size in ~34-byte steps.

    Used so the boundary tests do not hard-code a byte count that a library
    upgrade could move.
    """
    kp = payer or Keypair()
    # Phase 1: one-byte resolution via a single variable-length data blob.
    for data_len in range(1, 4200):
        raw = build(version, kp,
                    instructions=filler_instructions(1, data_len=data_len))
        if len(raw) >= target_bytes:
            return raw
        if len(raw) + 4 < target_bytes and data_len > 8 and \
                _phase1_plateau(version, kp, data_len):
            break
    # Phase 2: coarse, by instruction and account count.
    best: Optional[bytes] = None
    for accounts_per_ix in (1, 2, 3):
        for data_len in (8, 32, 64, 128, 192, 256):
            for count in range(1, 62):
                raw = build(version, kp, instructions=filler_instructions(
                    count, accounts_per_ix=accounts_per_ix, data_len=data_len))
                if len(raw) >= target_bytes:
                    if best is None or len(raw) < len(best):
                        best = raw
                    break
            if best is not None and len(best) - target_bytes < 64:
                return best
    if best is not None:
        return best
    raise AssertionError(
        f"cannot build a {version} transaction of {target_bytes} bytes")


def _phase1_plateau(version: str, payer: Keypair, data_len: int) -> bool:
    """True when the data blob has stopped changing the serialized size.

    A compact-u16 length prefix means a blob past 127 bytes still grows the
    message, so the plateau only appears if a hard per-instruction cap is hit.
    Detecting it lets phase 1 stop before an O(n^2) scan of huge blobs.
    """
    a = build(version, payer,
              instructions=filler_instructions(1, data_len=data_len))
    b = build(version, payer,
              instructions=filler_instructions(1, data_len=data_len + 1))
    return len(b) == len(a)


def simple_transfer(payer: Keypair, lamports: int = 1,
                    to: Optional[str] = None) -> Instruction:
    return transfer(TransferParams(
        from_pubkey=payer.pubkey(),
        to_pubkey=Pubkey.from_string(to or FAKE_ACCOUNT),
        lamports=lamports,
    ))


def signer_meta(pubkey: Pubkey, writable: bool = True) -> AccountMeta:
    return AccountMeta(pubkey, True, writable)


def read_only_meta(pubkey: Pubkey) -> AccountMeta:
    return AccountMeta(pubkey, False, False)


__all__ = [
    "FAKE_ACCOUNT", "FAKE_ACCOUNT_2", "MEMO_PROGRAM", "ZERO_SIGNATURE",
    "MAX_BYTES_LEGACY", "MAX_BYTES_V1", "VERSION_LEGACY", "VERSION_V0",
    "VERSION_V1", "b64", "build", "build_legacy", "build_v0", "build_v1",
    "build_of_size", "filler_instructions", "simple_transfer",
    "zero_signatures", "padded_instructions", "build_exact",
    "build_wide_data",
    "build_v0_with_alt", "alt_accounts",
    "signer_meta", "read_only_meta",
]
