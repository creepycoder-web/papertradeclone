"""
Direct, read-only Pump.fun and PumpSwap executable quotes.

WHY THIS MODULE EXISTS
======================

The harness previously had a "direct Pump quote" that could not work against
real data, for two independent reasons:

1. ``derive_bonding_curve_address`` returned ``None`` and said so in a
   docstring. With no PDA derivation there is no way to go from a mint to its
   curve, so the bonding-curve path was only reachable if the operator
   hand-supplied addresses.
2. ``decode_pumpswap_pool`` read no reserves at all, and
   ``quote_pumpswap_buy`` was then handed ``detail.get("base_reserve", 0)``.
   That is a zero reserve, so it always raised. The PumpSwap path was
   structurally dead.

Separately, the account layout the old decoder used was wrong. It looked for a
``mint`` field at offset 40 and used the discriminator
``189bc628ac2ffd72``. The real ``BondingCurve`` account has **no ``mint``
field** - the mint is only recoverable from the PDA seeds - and the real
discriminator is ``17b7f83760d8ac60``. So the old decoder would have rejected
every real curve on chain, in addition to never being able to find one.

This module replaces all of that with layouts and quote arithmetic taken from
the official SDKs and checked against live mainnet. See
``reports/tests/same_market_quote/direct_pump_verification.json`` for the
evidence, including an exact-match check of this arithmetic against
``@pump-fun/pump-swap-sdk``.

WHERE THE NUMBERS COME FROM
===========================

Nothing in this module is remembered from memory. Each fact below is taken
from the IDL that ships inside the official npm packages, and each is asserted
in ``tests/test_same_market_pumpdirect.py`` against both the documented offset
and a live-decoded account:

===========================  =====================  =========================
account                      discriminator          size(s)
===========================  =====================  =========================
``BondingCurve``             ``17b7f83760d8ac60``   125 legacy / 151 current
``Global``                   ``a7e8e8b1c86c727f``   1087
``FeeConfig`` (both progs)   ``8f3492bbdb7b4c9b``   2512 / 4073 / 4097
``Pool`` (PumpSwap)          ``f19a6d0411b16dbc``   270 / 300
``GlobalConfig`` (PumpSwap)  ``95089ccaa0fcb0d9``   949
===========================  =====================  =========================

THE TWO RULES THAT MATTER MOST
==============================

**1. Reserves are never guessed.** A PumpSwap quote reserve is the pool's quote
vault balance *plus* ``Pool.virtual_quote_reserves``, a **signed i128** that
has been negative since 2025-09-30. Using the raw vault balance alone is the
specific mistake that produces a confidently wrong price today and a
catastrophically wrong one later, so the virtual component is always added and
the base side is always the base vault balance.

**2. A missing field is a named failure, never a zero.** Every decoder raises
:class:`StateUnavailable` rather than returning a partially populated state.
A quote built on a zero reserve looks exactly like a real quote in a report,
and that is the failure this harness exists to prevent.

THIS MODULE CANNOT EXECUTE
==========================

There is no signer, no transaction builder, no ``send``, and no RPC method
other than the read methods in :mod:`autopaper.same_market.rpc`. The quote
functions below are pure arithmetic over state that the caller already holds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Awaitable, Callable, Mapping, Sequence

from .models import (
    SOL_DECIMALS,
    WSOL_MINT,
    ExecutableQuote,
    FailureMode,
    Freshness,
    MarketKind,
    PoolState,
)
from .numeric import DecimalError, effective_price, to_decimal, to_float
from .pda import b58decode_pubkey, b58encode, find_program_address_b58

# ---------------------------------------------------------------------------
# Programs and PDAs (all verified against the official SDKs)
# ---------------------------------------------------------------------------

#: Pump.fun bonding-curve program.
PUMP_FUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
#: PumpSwap AMM program.
PUMP_SWAP_PROGRAM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
#: pump-fees program, which owns the live fee configuration.
PUMP_FEE_PROGRAM = "pfeeUxB6jkeY1Hxd7CsFCAjcbHA9rWtchMGdZ6VojVZ"

#: The canonical pool index every migrated pump coin uses.
CANONICAL_POOL_INDEX = 0

#: The all-zeros pubkey. Pump.fun stores it as the ``quote_mint`` of a
#: SOL-quoted curve, and uses it to mean "no creator fee payable", so it has to
#: be recognised rather than treated as a real mint.
DEFAULT_PUBKEY = "11111111111111111111111111111111"

#: Quote mints the protocol treats as SOL, from the SDKs' own sets.
SOL_LIKE_QUOTE_MINTS: frozenset[str] = frozenset({
    DEFAULT_PUBKEY,
    WSOL_MINT,
    "9pan9bMn5HatX4EJdBwg9VgCa7Uz5HL8N1m5D3NdXejP",
})
#: Quote mints the protocol treats as a stable.
STABLE_QUOTE_MINTS: frozenset[str] = frozenset({
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
})

#: The fixed supply used for a mayhem-mode curve/pool's market cap.
PUMP_TOTAL_SUPPLY = 1_000_000_000_000_000


def derive_bonding_curve(mint: str) -> tuple[str, int]:
    """``findProgramAddress([b"bonding-curve", mint], PUMP)``."""
    return find_program_address_b58(
        [b"bonding-curve", b58decode_pubkey(mint, field="mint")], PUMP_FUN_PROGRAM
    )


def derive_pool_authority(mint: str) -> str:
    """``pumpPda(["pool-authority", mint])`` - under the PUMP program."""
    address, _ = find_program_address_b58(
        [b"pool-authority", b58decode_pubkey(mint, field="mint")], PUMP_FUN_PROGRAM
    )
    return address


def derive_canonical_pool(
    mint: str, quote_mint: str = WSOL_MINT
) -> tuple[str, int]:
    """
    ``poolPda(index=0, creator=pumpPoolAuthorityPda(mint), baseMint, quoteMint)``.

    This is the pool a migrated pump coin trades on. It is DERIVED, then the
    account's owner and discriminator are checked before its bytes are read -
    a derivation is a claim, and the chain is what confirms it.
    """
    authority = derive_pool_authority(mint)
    index = CANONICAL_POOL_INDEX.to_bytes(2, "little")
    return find_program_address_b58(
        [b"pool", index, b58decode_pubkey(authority, field="authority"),
         b58decode_pubkey(mint, field="mint"),
         b58decode_pubkey(quote_mint, field="quote_mint")],
        PUMP_SWAP_PROGRAM,
    )


#: Static PDA constants, verified against the official SDKs.
GLOBAL_PDA = find_program_address_b58([b"global"], PUMP_FUN_PROGRAM)[0]
PUMP_FEE_CONFIG_PDA = find_program_address_b58(
    [b"fee_config", b58decode_pubkey(PUMP_FUN_PROGRAM)], PUMP_FEE_PROGRAM
)[0]
AMM_GLOBAL_CONFIG_PDA = find_program_address_b58(
    [b"global_config"], PUMP_SWAP_PROGRAM
)[0]
AMM_FEE_CONFIG_PDA = find_program_address_b58(
    [b"fee_config", b58decode_pubkey(PUMP_SWAP_PROGRAM)], PUMP_FEE_PROGRAM
)[0]

# ---------------------------------------------------------------------------
# Discriminators, straight from the shipped IDLs
# ---------------------------------------------------------------------------

CURVE_DISCRIMINATOR = bytes([23, 183, 248, 55, 96, 216, 172, 96])        # 17b7f83760d8ac60
GLOBAL_DISCRIMINATOR = bytes([167, 232, 232, 177, 200, 108, 114, 127])   # a7e8e8b1c86c727f
FEE_CONFIG_DISCRIMINATOR = bytes([143, 52, 146, 187, 219, 123, 76, 155])  # 8f3492bbdb7b4c9b
POOL_DISCRIMINATOR = bytes([241, 154, 109, 4, 17, 177, 109, 188])        # f19a6d0411b16dbc
AMM_GLOBAL_CONFIG_DISCRIMINATOR = bytes([149, 8, 156, 202, 160, 252, 176, 217])  # 95089ccaa0fcb0d9

#: Discriminators are also reproducible as ``sha256("account:<Name>")[0:8]``,
#: which is asserted in the tests. Two errors were caught this way while
#: writing this module - a mistyped byte in the FeeConfig discriminator and
#: another in the Pool one - which is precisely why the check is not a
#: convenience but a load-bearing guard: a mis-transcribed constant would
#: otherwise have silently rejected every real account, or accepted a wrong
#: account type and read its bytes as reserves.

#: Minimum byte lengths. Accounts are VARIABLE length: Anchor grew the curve
#: account 125 -> 151 and the pool 270 -> 300, and an older account simply does
#: not carry the later fields. So the decoder reads what is present and treats
#: a missing tail field as absent - never as a value it invented.
CURVE_MIN_LEN = 84          # through quote_mint
CURVE_CREATOR_FEE_LEN = 123  # through creator_fee_bps
POOL_MIN_LEN = 245          # through virtual_quote_reserves
POOL_CREATOR_FEE_LEN = 269  # through creator_fee_bps

#: Offsets. Each is the documented field's byte position, counted from the
#: start of the account INCLUDING the 8-byte discriminator.
_OFF = {
    "curve.virtual_token_reserves": 8,
    "curve.virtual_quote_reserves": 16,
    "curve.real_token_reserves": 24,
    "curve.real_quote_reserves": 32,
    "curve.token_total_supply": 40,
    "curve.complete": 48,
    "curve.creator": 49,
    "curve.is_mayhem_mode": 81,
    "curve.is_cashback_coin": 82,
    "curve.quote_mint": 83,
    "curve.creator_fee_bps": 115,
    "curve.can_edit_creator_fee": 123,
    "global.token_total_supply": 97,
    "global.fee_basis_points": 105,
    "global.creator_fee_basis_points": 154,
    "global.creator_fee_configurable": 1045,
    "global.max_configurable_creator_fee_bps": 1046,
    "pool.pool_bump": 8,
    "pool.index": 9,
    "pool.creator": 11,
    "pool.base_mint": 43,
    "pool.quote_mint": 75,
    "pool.lp_mint": 107,
    "pool.pool_base_token_account": 139,
    "pool.pool_quote_token_account": 171,
    "pool.lp_supply": 203,
    "pool.coin_creator": 211,
    "pool.is_mayhem_mode": 243,
    "pool.is_cashback_coin": 244,
    "pool.virtual_quote_reserves": 245,
    "pool.creator_fee_bps": 261,
    "amm_global.lp_fee_basis_points": 40,
    "amm_global.protocol_fee_basis_points": 48,
    "amm_global.coin_creator_fee_basis_points": 313,
    "amm_global.creator_fee_configurable": 940,
    "amm_global.max_configurable_creator_fee_bps": 941,
}

#: SPL layouts. ``supply``/``decimals`` live on the MINT, not the token
#: account; the token account only holds a balance.
MINT_SUPPLY_OFFSET = 36
MINT_DECIMALS_OFFSET = 44
TOKEN_ACCOUNT_AMOUNT_OFFSET = 64


class StateUnavailable(RuntimeError):
    """
    On-chain state could not be read, or does not mean what was assumed.

    Carries the :class:`FailureMode` the report should show. Never degraded to a
    zero reserve or a default fee: a fabricated-but-plausible reserve is the
    one thing that would make this harness lie.
    """

    def __init__(self, message: str, *, failure: FailureMode = FailureMode.STATE_UNAVAILABLE) -> None:
        super().__init__(message)
        self.failure = failure


# ---------------------------------------------------------------------------
# Byte readers
# ---------------------------------------------------------------------------


def _require_len(raw: bytes, minimum: int, what: str) -> None:
    if len(raw) < minimum:
        raise StateUnavailable(
            f"{what} account is {len(raw)} bytes; the documented layout needs "
            f"at least {minimum}, so this is not the account it claims to be"
        )


def _u8(raw: bytes, offset: int, what: str) -> int:
    _require_len(raw, offset + 1, what)
    return raw[offset]


def _u16(raw: bytes, offset: int, what: str) -> int:
    _require_len(raw, offset + 2, what)
    return int.from_bytes(raw[offset : offset + 2], "little")


def _u64(raw: bytes, offset: int, what: str) -> int:
    _require_len(raw, offset + 8, what)
    return int.from_bytes(raw[offset : offset + 8], "little")


def _i128(raw: bytes, offset: int, what: str) -> int:
    """
    A SIGNED 128-bit little-endian integer.

    ``virtual_quote_reserves`` went negative in September 2025, so reading it
    unsigned would make a pool look like it holds an absurd reserve. This is
    read as signed on purpose, and the quote adds it to the vault balance
    exactly as the protocol does.
    """
    _require_len(raw, offset + 16, what)
    value = int.from_bytes(raw[offset : offset + 16], "little")
    if value >= 1 << 127:
        value -= 1 << 128
    return value


def _bool(raw: bytes, offset: int, what: str) -> bool:
    return _u8(raw, offset, what) == 1


def _pubkey(raw: bytes, offset: int, what: str) -> str:
    _require_len(raw, offset + 32, what)
    return b58encode(raw[offset : offset + 32])


def _check_discriminator(raw: bytes, expected: bytes, what: str) -> None:
    """
    Is this account the type it claims to be?

    Both halves matter: the owner check says "this program owns these bytes",
    the discriminator says "and these bytes are the account type I think they
    are". An account owned by PumpSwap that happens to be long enough is not a
    pool.
    """
    actual = raw[:8]
    if actual != expected:
        raise StateUnavailable(
            f"{what} account discriminator is {actual.hex()} but the official "
            f"IDL says {expected.hex()}; these bytes are a different account "
            "type, so they are not read as if they were a market"
        )


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BondingCurve:
    """A decoded ``BondingCurve``, with the mint it belongs to attached."""

    address: str
    mint: str
    virtual_token_reserves: int
    virtual_quote_reserves: int
    real_token_reserves: int
    real_quote_reserves: int
    token_total_supply: int
    complete: bool
    creator: str
    is_mayhem_mode: bool
    is_cashback_coin: bool
    quote_mint: str
    creator_fee_bps: int | None
    can_edit_creator_fee: bool | None
    account_len: int
    slot: int | None = None
    timestamp: float = 0.0

    @property
    def quote_mint_is_sol_like(self) -> bool:
        return self.quote_mint in SOL_LIKE_QUOTE_MINTS

    @property
    def graduated(self) -> bool:
        """
        Graduated == the curve has been drained and migration has happened.

        ``complete`` alone is the protocol's own flag; a drained curve is the
        observable proof, and a graduated curve quotes zero output, so the
        quote functions refuse it rather than model a market that has stopped.
        """
        return self.complete or self.virtual_token_reserves == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "mint": self.mint,
            "virtual_token_reserves": self.virtual_token_reserves,
            "virtual_quote_reserves": self.virtual_quote_reserves,
            "real_token_reserves": self.real_token_reserves,
            "real_quote_reserves": self.real_quote_reserves,
            "token_total_supply": self.token_total_supply,
            "complete": self.complete,
            "creator": self.creator,
            "creator_fee_bps": self.creator_fee_bps,
            "quote_mint": self.quote_mint,
            "quote_mint_is_sol_like": self.quote_mint_is_sol_like,
            "is_mayhem_mode": self.is_mayhem_mode,
            "is_cashback_coin": self.is_cashback_coin,
            "graduated": self.graduated,
            "account_len": self.account_len,
            "slot": self.slot,
            "timestamp": self.timestamp,
        }


def decode_bonding_curve(
    raw: bytes,
    *,
    curve_address: str,
    mint: str,
    slot: int | None = None,
    timestamp: float | None = None,
) -> BondingCurve:
    """
    Decode a ``BondingCurve`` account.

    IDENTITY IS FROM THE PDA, NOT THE ACCOUNT. The current layout has no
    ``mint`` field - the mint lives in the address's seeds and nowhere in the
    data - so the check that this account is the curve for ``mint`` is made by
    re-deriving the PDA and comparing addresses. :func:`load_bonding_curve`
    does exactly that, and refuses a curve whose address does not re-derive
    from the mint under test. This is not a weaker check than reading a stored
    mint; it is a stronger one, because the address is what the protocol itself
    routes through.
    """
    _check_discriminator(raw, CURVE_DISCRIMINATOR, "bonding curve")
    _require_len(raw, CURVE_MIN_LEN, "bonding curve")
    o = _OFF
    return BondingCurve(
        address=curve_address,
        mint=mint,
        virtual_token_reserves=_u64(raw, o["curve.virtual_token_reserves"], "bonding curve"),
        virtual_quote_reserves=_u64(raw, o["curve.virtual_quote_reserves"], "bonding curve"),
        real_token_reserves=_u64(raw, o["curve.real_token_reserves"], "bonding curve"),
        real_quote_reserves=_u64(raw, o["curve.real_quote_reserves"], "bonding curve"),
        token_total_supply=_u64(raw, o["curve.token_total_supply"], "bonding curve"),
        complete=_bool(raw, o["curve.complete"], "bonding curve"),
        creator=_pubkey(raw, o["curve.creator"], "bonding curve"),
        is_mayhem_mode=_bool(raw, o["curve.is_mayhem_mode"], "bonding curve"),
        is_cashback_coin=_bool(raw, o["curve.is_cashback_coin"], "bonding curve"),
        quote_mint=_pubkey(raw, o["curve.quote_mint"], "bonding curve"),
        creator_fee_bps=(
            _u64(raw, o["curve.creator_fee_bps"], "bonding curve")
            if len(raw) >= CURVE_CREATOR_FEE_LEN
            else None
        ),
        can_edit_creator_fee=(
            _bool(raw, o["curve.can_edit_creator_fee"], "bonding curve")
            if len(raw) >= CURVE_CREATOR_FEE_LEN + 1
            else None
        ),
        account_len=len(raw),
        slot=slot,
        timestamp=time.time() if timestamp is None else timestamp,
    )


@dataclass(frozen=True)
class PumpGlobal:
    """Pump's ``Global`` account - the fallback fee schedule."""

    fee_basis_points: int
    creator_fee_basis_points: int
    token_total_supply: int
    creator_fee_configurable: bool
    max_configurable_creator_fee_bps: int | None
    account_len: int
    timestamp: float = 0.0


def decode_global(raw: bytes, *, timestamp: float | None = None) -> PumpGlobal:
    _check_discriminator(raw, GLOBAL_DISCRIMINATOR, "pump global")
    o = _OFF
    return PumpGlobal(
        fee_basis_points=_u64(raw, o["global.fee_basis_points"], "pump global"),
        creator_fee_basis_points=_u64(raw, o["global.creator_fee_basis_points"], "pump global"),
        token_total_supply=_u64(raw, o["global.token_total_supply"], "pump global"),
        creator_fee_configurable=_bool(raw, o["global.creator_fee_configurable"], "pump global"),
        max_configurable_creator_fee_bps=(
            _u64(raw, o["global.max_configurable_creator_fee_bps"], "pump global")
            if len(raw) >= o["global.max_configurable_creator_fee_bps"] + 8
            else None
        ),
        account_len=len(raw),
        timestamp=time.time() if timestamp is None else timestamp,
    )


@dataclass(frozen=True)
class Fees:
    """One fee schedule. Basis points, not fractions."""

    lp_fee_bps: int
    protocol_fee_bps: int
    creator_fee_bps: int

    @property
    def total(self) -> int:
        return self.lp_fee_bps + self.protocol_fee_bps + self.creator_fee_bps

    def as_dict(self) -> dict[str, int]:
        return {
            "lp_fee_bps": self.lp_fee_bps,
            "protocol_fee_bps": self.protocol_fee_bps,
            "creator_fee_bps": self.creator_fee_bps,
            "total_bps": self.total,
        }


@dataclass(frozen=True)
class FeeTier:
    market_cap_lamports_threshold: int
    fees: Fees


@dataclass(frozen=True)
class FeeConfig:
    """The live fee schedule. Tiers are market-cap bands, not flat rates."""

    flat_fees: Fees
    fee_tiers: tuple[FeeTier, ...]
    stable_fee_tiers: tuple[FeeTier, ...]
    exotic_flat_fees: Fees
    account_len: int
    timestamp: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "flat_fees": self.flat_fees.as_dict(),
            "exotic_flat_fees": self.exotic_flat_fees.as_dict(),
            "fee_tiers": [
                {"market_cap_lamports_threshold": t.market_cap_lamports_threshold,
                 "fees": t.fees.as_dict()}
                for t in self.fee_tiers
            ],
            "stable_fee_tiers": [
                {"market_cap_lamports_threshold": t.market_cap_lamports_threshold,
                 "fees": t.fees.as_dict()}
                for t in self.stable_fee_tiers
            ],
            "account_len": self.account_len,
        }


def decode_fee_config(raw: bytes, *, timestamp: float | None = None) -> FeeConfig:
    """
    Decode a ``FeeConfig``.

    Layout: discriminator, ``bump``, ``admin``, ``flat_fees``, then two
    ``vec<FeeTier>`` (a u32 length prefix followed by 40-byte entries of a u128
    threshold plus three u64s), then ``exotic_flat_fees``.
    """
    _check_discriminator(raw, FEE_CONFIG_DISCRIMINATOR, "fee config")
    cursor = 41  # 8 discriminator + 1 bump + 32 admin

    def read_fees() -> Fees:
        nonlocal cursor
        if cursor + 24 > len(raw):
            raise StateUnavailable("fee config is truncated inside a Fees struct")
        value = Fees(
            lp_fee_bps=_u64(raw, cursor, "fee config"),
            protocol_fee_bps=_u64(raw, cursor + 8, "fee config"),
            creator_fee_bps=_u64(raw, cursor + 16, "fee config"),
        )
        cursor += 24
        return value

    def read_tiers() -> tuple[FeeTier, ...]:
        nonlocal cursor
        if cursor + 4 > len(raw):
            raise StateUnavailable("fee config is truncated at a tier vector header")
        count = int.from_bytes(raw[cursor : cursor + 4], "little")
        cursor += 4
        if count < 0 or cursor + count * 40 > len(raw):
            raise StateUnavailable(
                f"fee config declares {count} tiers needing {count * 40} bytes "
                f"but only {max(0, len(raw) - cursor)} remain; the length prefix "
                "is not trustworthy and the tiers are NOT read"
            )
        tiers = []
        for _ in range(count):
            threshold = int.from_bytes(raw[cursor : cursor + 16], "little")
            cursor += 16  # the u128 threshold occupies its own 16 bytes
            tiers.append(FeeTier(threshold, read_fees()))
        return tuple(tiers)

    flat = read_fees()
    tiers = read_tiers()
    stable = read_tiers()
    exotic = read_fees()
    return FeeConfig(
        flat_fees=flat,
        fee_tiers=tiers,
        stable_fee_tiers=stable,
        exotic_flat_fees=exotic,
        account_len=len(raw),
        timestamp=time.time() if timestamp is None else timestamp,
    )


@dataclass(frozen=True)
class PumpSwapPool:
    """A decoded PumpSwap ``Pool`` plus the reserves read from its vaults."""

    address: str
    pool_bump: int
    index: int
    creator: str
    base_mint: str
    quote_mint: str
    lp_mint: str
    pool_base_token_account: str
    pool_quote_token_account: str
    lp_supply: int
    coin_creator: str
    is_mayhem_mode: bool
    is_cashback_coin: bool
    virtual_quote_reserves: int
    creator_fee_bps: int | None
    base_decimals: int
    base_supply: int
    base_reserve: int
    quote_reserve: int
    account_len: int
    slot: int | None = None
    timestamp: float = 0.0

    @property
    def effective_quote_reserve(self) -> int:
        """
        The protocol's own quote reserve: the vault balance PLUS the signed
        virtual component. This is what the SDK quotes against, and it is not
        the vault balance.
        """
        return self.quote_reserve + self.virtual_quote_reserves

    @property
    def is_canonical_pump_pool(self) -> bool:
        """A canonical pool's creator is the mint's pool-authority PDA."""
        try:
            return self.creator == derive_pool_authority(self.base_mint)
        except ValueError:
            return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "pool_bump": self.pool_bump,
            "index": self.index,
            "creator": self.creator,
            "base_mint": self.base_mint,
            "quote_mint": self.quote_mint,
            "lp_mint": self.lp_mint,
            "pool_base_token_account": self.pool_base_token_account,
            "pool_quote_token_account": self.pool_quote_token_account,
            "lp_supply": self.lp_supply,
            "coin_creator": self.coin_creator,
            "virtual_quote_reserves": self.virtual_quote_reserves,
            "base_reserve_raw": self.base_reserve,
            "quote_reserve_raw": self.quote_reserve,
            "effective_quote_reserve_raw": self.effective_quote_reserve,
            "base_decimals": self.base_decimals,
            "base_supply": self.base_supply,
            "is_canonical_pump_pool": self.is_canonical_pump_pool,
            "is_mayhem_mode": self.is_mayhem_mode,
            "is_cashback_coin": self.is_cashback_coin,
            "creator_fee_bps": self.creator_fee_bps,
            "account_len": self.account_len,
            "slot": self.slot,
            "timestamp": self.timestamp,
        }


def decode_pool(raw: bytes, *, pool_address: str, slot: int | None = None,
                timestamp: float | None = None) -> dict[str, Any]:
    """
    Decode the pool account's own fields.

    Reserves are NOT decoded here, because they are not in this account. They
    live in the two token accounts the pool points at, so they arrive as
    arguments to :func:`decode_pumpswap_pool`.
    """
    _check_discriminator(raw, POOL_DISCRIMINATOR, "pumpswap pool")
    _require_len(raw, POOL_MIN_LEN, "pumpswap pool")
    o = _OFF
    return {
        "pool_bump": _u8(raw, o["pool.pool_bump"], "pumpswap pool"),
        "index": _u16(raw, o["pool.index"], "pumpswap pool"),
        "creator": _pubkey(raw, o["pool.creator"], "pumpswap pool"),
        "base_mint": _pubkey(raw, o["pool.base_mint"], "pumpswap pool"),
        "quote_mint": _pubkey(raw, o["pool.quote_mint"], "pumpswap pool"),
        "lp_mint": _pubkey(raw, o["pool.lp_mint"], "pumpswap pool"),
        "pool_base_token_account": _pubkey(raw, o["pool.pool_base_token_account"], "pumpswap pool"),
        "pool_quote_token_account": _pubkey(raw, o["pool.pool_quote_token_account"], "pumpswap pool"),
        "lp_supply": _u64(raw, o["pool.lp_supply"], "pumpswap pool"),
        "coin_creator": _pubkey(raw, o["pool.coin_creator"], "pumpswap pool"),
        "is_mayhem_mode": _bool(raw, o["pool.is_mayhem_mode"], "pumpswap pool"),
        "is_cashback_coin": _bool(raw, o["pool.is_cashback_coin"], "pumpswap pool"),
        "virtual_quote_reserves": _i128(raw, o["pool.virtual_quote_reserves"], "pumpswap pool"),
        "creator_fee_bps": (
            _u64(raw, o["pool.creator_fee_bps"], "pumpswap pool")
            if len(raw) >= POOL_CREATOR_FEE_LEN else None
        ),
        "account_len": len(raw),
        "slot": slot,
        "timestamp": time.time() if timestamp is None else timestamp,
    }


def decode_spl_mint(raw: bytes, *, what: str = "mint") -> tuple[int, int]:
    """``(decimals, supply)`` from an SPL Mint account."""
    _require_len(raw, MINT_DECIMALS_OFFSET + 1, what)
    return raw[MINT_DECIMALS_OFFSET], _u64(raw, MINT_SUPPLY_OFFSET, what)


def decode_token_account_amount(raw: bytes, *, what: str = "token account") -> int:
    """The balance held by an SPL Token account."""
    return _u64(raw, TOKEN_ACCOUNT_AMOUNT_OFFSET, what)


def decode_pumpswap_pool(
    raw: bytes,
    *,
    pool_address: str,
    base_mint_account: bytes,
    base_token_account: bytes,
    quote_token_account: bytes,
    slot: int | None = None,
    timestamp: float | None = None,
) -> PumpSwapPool:
    """
    Assemble a :class:`PumpSwapPool` from the pool and the accounts it names.

    Three accounts are needed and all three are verified: the pool says which
    vault accounts hold its liquidity, and those vaults are read directly. A
    reserve read from anywhere else - a stale cache, a guessed balance, an
    unrelated account - would not be this pool's reserve.
    """
    fields = decode_pool(raw, pool_address=pool_address, slot=slot, timestamp=timestamp)
    decimals, supply = decode_spl_mint(base_mint_account, what="base mint")
    return PumpSwapPool(
        address=pool_address,
        base_decimals=decimals,
        base_supply=supply,
        base_reserve=decode_token_account_amount(base_token_account),
        quote_reserve=decode_token_account_amount(quote_token_account),
        **fields,
    )


@dataclass(frozen=True)
class AmmGlobalConfig:
    lp_fee_basis_points: int
    protocol_fee_basis_points: int
    coin_creator_fee_basis_points: int
    creator_fee_configurable: bool
    max_configurable_creator_fee_bps: int | None
    account_len: int
    timestamp: float = 0.0


def decode_amm_global_config(raw: bytes, *, timestamp: float | None = None) -> AmmGlobalConfig:
    _check_discriminator(raw, AMM_GLOBAL_CONFIG_DISCRIMINATOR, "amm global config")
    o = _OFF
    return AmmGlobalConfig(
        lp_fee_basis_points=_u64(raw, o["amm_global.lp_fee_basis_points"], "amm global config"),
        protocol_fee_basis_points=_u64(raw, o["amm_global.protocol_fee_basis_points"], "amm global config"),
        coin_creator_fee_basis_points=_u64(raw, o["amm_global.coin_creator_fee_basis_points"], "amm global config"),
        creator_fee_configurable=_bool(raw, o["amm_global.creator_fee_configurable"], "amm global config"),
        max_configurable_creator_fee_bps=(
            _u64(raw, o["amm_global.max_configurable_creator_fee_bps"], "amm global config")
            if len(raw) >= o["amm_global.max_configurable_creator_fee_bps"] + 8
            else None
        ),
        account_len=len(raw),
        timestamp=time.time() if timestamp is None else timestamp,
    )


# ---------------------------------------------------------------------------
# Fee selection - the part that quietly changes prices
# ---------------------------------------------------------------------------


def ceil_fee(amount: int, fee_bps: int) -> int:
    """
    ``ceil(amount * bps / 10000)``.

    CEIL, deliberately. The SDK's inline comments say "floor" while its code
    ceils; the code is what runs in production, so the code is what is
    reproduced here. Flooring every fee understates the cost by up to one
    lamport per leg, which is small - which is exactly why it survives review.
    """
    if fee_bps <= 0 or amount <= 0:
        return 0
    return -((-amount * fee_bps) // 10_000)


def _select_tier(tiers: Sequence[FeeTier], market_cap: int) -> Fees:
    if not tiers:
        raise StateUnavailable("the fee schedule has no tiers for this quote asset")
    if market_cap < tiers[0].market_cap_lamports_threshold:
        return tiers[0].fees
    for tier in reversed(tiers):
        if market_cap >= tier.market_cap_lamports_threshold:
            return tier.fees
    return tiers[0].fees


def curve_market_cap(curve: BondingCurve, mint_supply: int) -> int:
    """
    The curve's market-cap basis for fee-tier selection.

    Refuses on a graduated curve. Its virtual token reserves are zero once
    migration has drained it, so this would raise ``ZeroDivisionError`` - which
    is not a named failure and escapes the fail-closed contract that every other
    path in this module honours. A token that graduates between being observed
    and being priced must produce ``UNKNOWN``, not a crash.
    """
    if curve.virtual_token_reserves <= 0:
        raise StateUnavailable(
            "the curve's virtual token reserves are "
            f"{curve.virtual_token_reserves}, so no market cap can be computed; "
            "the curve is drained and is no longer the market",
            failure=FailureMode.CURVE_UNAVAILABLE,
        )
    return (curve.virtual_quote_reserves * mint_supply) // curve.virtual_token_reserves


def curve_fee_schedule(
    curve: BondingCurve,
    *,
    global_config: PumpGlobal | None,
    fee_config: FeeConfig | None,
    mint_supply: int,
) -> Fees:
    """
    The curve's fee schedule.

    Read from the live ``FeeConfig`` when it exists - which on mainnet it does -
    and only falling back to ``Global`` when the fee config is absent. The
    ``Global`` numbers are a FALLBACK in the protocol's own code, not the
    primary source, so preferring them would quote the wrong price.
    """
    if fee_config is not None:
        market_cap = curve_market_cap(curve, mint_supply)
        if curve.quote_mint_is_sol_like or curve.quote_mint in STABLE_QUOTE_MINTS:
            tiers = (
                fee_config.fee_tiers
                if curve.quote_mint_is_sol_like
                else (fee_config.stable_fee_tiers or fee_config.fee_tiers)
            )
            schedule = _select_tier(tiers, market_cap)
        elif fee_config.exotic_flat_fees.total:
            schedule = fee_config.exotic_flat_fees
        else:
            schedule = fee_config.flat_fees
    else:
        if global_config is None:
            raise StateUnavailable(
                "neither a FeeConfig nor a Global account could be read, so the "
                "protocol fee is unknown; it is NOT defaulted"
            )
        schedule = Fees(0, global_config.fee_basis_points,
                        global_config.creator_fee_basis_points)

    creator_bps = schedule.creator_fee_bps
    if (
        global_config is not None
        and global_config.creator_fee_configurable
        and curve.creator_fee_bps
    ):
        creator_bps = curve.creator_fee_bps
    return Fees(schedule.lp_fee_bps, schedule.protocol_fee_bps, creator_bps)


def amm_fee_schedule(
    pool: PumpSwapPool,
    *,
    global_config: AmmGlobalConfig | None,
    fee_config: FeeConfig | None,
    circulating_supply: int,
) -> Fees:
    """
    The pool's fee schedule.

    A CANONICAL pump pool and an arbitrary third-party pool on the same program
    are charged differently - canonical pools use market-cap tiers, others get
    the flat rate. So the canonical test (``creator == pool_authority_pda(mint)``)
    is part of pricing, not a nicety.
    """
    if fee_config is not None:
        market_cap = (pool.effective_quote_reserve * circulating_supply) // pool.base_reserve
        if not pool.is_canonical_pump_pool:
            schedule = fee_config.flat_fees
        elif pool.quote_mint in SOL_LIKE_QUOTE_MINTS:
            schedule = _select_tier(fee_config.fee_tiers, market_cap)
        elif pool.quote_mint in STABLE_QUOTE_MINTS:
            schedule = _select_tier(
                fee_config.stable_fee_tiers or fee_config.fee_tiers, market_cap
            )
        elif fee_config.exotic_flat_fees.total:
            schedule = fee_config.exotic_flat_fees
        else:
            schedule = fee_config.flat_fees
    else:
        if global_config is None:
            raise StateUnavailable(
                "neither a FeeConfig nor a GlobalConfig could be read, so the "
                "pool fee is unknown; it is NOT defaulted"
            )
        schedule = Fees(
            global_config.lp_fee_basis_points,
            global_config.protocol_fee_basis_points,
            global_config.coin_creator_fee_basis_points,
        )
    creator_bps = schedule.creator_fee_bps
    if (
        global_config is not None
        and global_config.creator_fee_configurable
        and pool.creator_fee_bps
    ):
        creator_bps = pool.creator_fee_bps
    return Fees(schedule.lp_fee_bps, schedule.protocol_fee_bps, creator_bps)


# ---------------------------------------------------------------------------
# The quote arithmetic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectQuoteResult:
    """Raw integer output of a quote, before it is dressed up for a report."""

    input_amount_raw: int
    output_amount_raw: int
    input_decimals: int
    output_decimals: int
    fee_bps: int
    fees: Fees
    effective_price: Decimal
    price_impact_pct: Decimal | None = None
    amount_after_fee: int | None = None
    quote_timestamp: float = 0.0
    #: ``True`` for a BUY (input is the quote asset, output is the token) and
    #: ``False`` for a SELL (input is the token, output is the quote asset).
    #: Recorded rather than inferred so a caller cannot read the two sides the
    #: wrong way round - the mistake this whole module exists to prevent.
    quote_asset_input: bool = True
    notes: tuple[str, ...] = ()
    #: THE EXECUTION-NATIVE REFERENCE PRICE.
    #:
    #: The SAME market's current price, derived from the SAME decoded account
    #: and at the SAME slot that produced ``effective_price`` above, in the same
    #: units (quote asset per WHOLE token). It is the marginal/spot price: what
    #: this market charges before this particular order is layered on.
    #:
    #: It used to be computed inside :func:`_impact_pct` and then DISCARDED,
    #: leaving only the ratio. That ratio is precisely
    #: ``(effective_price / spot_price - 1) * 100`` -- so ``price_impact_pct``
    #: was ALREADY an execution-native, same-market, same-moment comparison,
    #: and the entry guard was throwing it away in favour of an external
    #: aggregate's lagging snapshot. It is surfaced here so the guard can
    #: compare against the market itself instead.
    #:
    #: ``None`` when no positive spot could be derived, which the caller must
    #: treat as "reference unavailable" and never as zero.
    spot_price: Decimal | None = None
    #: The exact reserve ratio the spot came from, for reproducibility. For a
    #: bonding curve this is ``virtual_quote_reserves /
    #: virtual_token_reserves``; for a PumpSwap pool it is
    #: ``effective_quote_reserve / base_reserve`` where the effective reserve is
    #: the vault balance PLUS the virtual reserve -- the reserve the protocol
    #: actually prices against, not the vault alone.
    spot_numerator_raw: int | None = None
    spot_denominator_raw: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_amount_raw": self.input_amount_raw,
            "output_amount_raw": self.output_amount_raw,
            "input_amount_decimal": float(
                to_decimal(self.input_amount_raw).scaleb(-self.input_decimals)
            ),
            "output_amount_decimal": float(
                to_decimal(self.output_amount_raw).scaleb(-self.output_decimals)
            ),
            "input_decimals": self.input_decimals,
            "output_decimals": self.output_decimals,
            # SOL per token. NOT USD: the quote was taken with a SOL input, and
            # labelling this "usd" is exactly the SOL-vs-USD mislabelling that
            # produced a gap of several hundred percent in this repository. The
            # USD figure is computed by the caller against a logged reference.
            "effective_price_sol_per_token": float(self.effective_price),
            "fee_bps": self.fee_bps,
            "fees": self.fees.as_dict(),
            "price_impact_pct": (
                float(self.price_impact_pct) if self.price_impact_pct is not None
                else None
            ),
            "execution_native_reference_price": (
                float(self.spot_price) if self.spot_price is not None else None
            ),
            "spot_numerator_raw": self.spot_numerator_raw,
            "spot_denominator_raw": self.spot_denominator_raw,
            "amount_after_fee_raw": self.amount_after_fee,
            "quote_timestamp": self.quote_timestamp,
            "direction": "BUY" if self.quote_asset_input else "SELL",
            "side": (
                f"{'quote_asset' if self.quote_asset_input else 'token'}"
                f"({self.input_decimals}) -> "
                f"{'token' if self.quote_asset_input else 'quote_asset'}"
                f"({self.output_decimals})"
            ),
            "notes": list(self.notes),
        }


def quote_bonding_curve_buy(
    curve: BondingCurve,
    *,
    sol_amount_lamports: int,
    token_decimals: int,
    fees: Fees,
    mint_supply: int,
) -> DirectQuoteResult:
    """
    A BUY against the bonding curve: ``SOL/wSOL -> TOKEN``.

    This reproduces ``getBuyTokenAmountFromSolAmount`` exactly:

        input  = (amount - 1) * 10000 // (10000 + totalFeeBps)
        output = input * virtualTokenReserves // (virtualQuoteReserves + input)
        output = min(output, realTokenReserves)

    Three details are easy to get wrong and each changes the price:

    * the fee is taken from the INPUT, before the curve, not added on top of
      the output (the base-in direction adds it instead - mixing the two
      under-quotes by about the fee);
    * ``real_token_reserves`` CAPS the output, so the curve cannot sell tokens
      it does not hold;
    * a graduated curve returns ZERO output, and this raises rather than
      reporting a zero-output quote.
    """
    if sol_amount_lamports <= 0:
        raise StateUnavailable(
            "the buy amount must be positive", failure=FailureMode.QUOTE_UNAVAILABLE
        )
    if curve.graduated:
        raise StateUnavailable(
            "the bonding curve is complete (virtual token reserves are "
            f"{curve.virtual_token_reserves}), so it is no longer the market and "
            "prices nothing; quote the PumpSwap pool instead",
            failure=FailureMode.CURVE_UNAVAILABLE,
        )
    if curve.virtual_quote_reserves <= 0:
        raise StateUnavailable(
            "the curve's virtual quote reserve is zero, so no price is defined",
            failure=FailureMode.CURVE_UNAVAILABLE,
        )

    total_fee_bps = fees.protocol_fee_bps
    if curve.creator != DEFAULT_PUBKEY:
        total_fee_bps += fees.creator_fee_bps
    if total_fee_bps >= 10_000:
        raise StateUnavailable(
            f"the total fee is {total_fee_bps} bps, which is not a price",
            failure=FailureMode.QUOTE_UNAVAILABLE,
        )

    net_input = ((sol_amount_lamports - 1) * 10_000) // (10_000 + total_fee_bps)
    if net_input <= 0:
        raise StateUnavailable(
            f"{sol_amount_lamports} lamports is below the minimum this curve "
            "will price once the fee is taken",
            failure=FailureMode.QUOTE_UNAVAILABLE,
        )

    out = (net_input * curve.virtual_token_reserves) // (
        curve.virtual_quote_reserves + net_input
    )
    capped = min(out, curve.real_token_reserves)
    if capped <= 0:
        raise StateUnavailable(
            "the curve prices zero output for this size, so there is no quote",
            failure=FailureMode.QUOTE_UNAVAILABLE,
        )

    try:
        price = effective_price(
            sol_amount_lamports, capped,
            input_decimals=SOL_DECIMALS, output_decimals=token_decimals,
        )
    except DecimalError as exc:
        raise StateUnavailable(
            f"cannot compute an effective price: {exc}",
            failure=FailureMode.DECIMAL_ERROR,
        ) from exc

    quoted_at = time.time()
    spot = _spot_price(
        numerator=curve.virtual_quote_reserves,
        denominator=curve.virtual_token_reserves,
        input_decimals=SOL_DECIMALS,
        output_decimals=token_decimals,
    )
    impact = _impact_pct(
        spot_price_numerator=curve.virtual_quote_reserves,
        spot_price_denominator=curve.virtual_token_reserves,
        executed_price=price,
        input_decimals=SOL_DECIMALS,
        output_decimals=token_decimals,
    )
    return DirectQuoteResult(
        input_amount_raw=sol_amount_lamports,
        output_amount_raw=capped,
        input_decimals=SOL_DECIMALS,
        output_decimals=token_decimals,
        fee_bps=total_fee_bps,
        fees=fees,
        effective_price=price,
        price_impact_pct=impact,
        # The execution-native reference: this curve's own virtual reserves,
        # read in the same account fetch, at the same slot, as `price`.
        spot_price=spot,
        spot_numerator_raw=curve.virtual_quote_reserves,
        spot_denominator_raw=curve.virtual_token_reserves,
        amount_after_fee=net_input,
        quote_timestamp=quoted_at,
        notes=(
            "quote-in direction: the fee is taken from the input before the "
            "curve, matching getBuyTokenAmountFromSolAmount",
            "output capped at real_token_reserves",
            *(() if out <= curve.real_token_reserves
              else ("the constant-product output exceeded real_token_reserves "
                    "and was capped",)),
        ),
    )


def quote_pumpswap_buy(
    pool: PumpSwapPool,
    *,
    sol_amount_lamports: int,
    fees: Fees,
    circulating_supply: int,
) -> DirectQuoteResult:
    """
    A BUY against a PumpSwap pool: ``SOL/wSOL -> TOKEN``.

    This reproduces ``buyQuoteInput`` exactly:

        effectiveQuote = quote * 10000 // (10000 + lpFee + protocolFee + creatorFee)
        [clamped so the ceil'd fees still fit inside the input]
        input          = effectiveQuote - 1
        output         = baseReserve * input // (effectiveQuoteReserve + input)

    where ``effectiveQuoteReserve = quoteVaultBalance + virtualQuoteReserves``
    - the vault balance alone is NOT the reserve the protocol prices against.
    """
    if sol_amount_lamports <= 0:
        raise StateUnavailable(
            "the buy amount must be positive", failure=FailureMode.QUOTE_UNAVAILABLE
        )
    if pool.base_reserve <= 0:
        raise StateUnavailable(
            f"pool base reserve is {pool.base_reserve}; a pool with no base "
            "liquidity prices nothing and is not modelled with a substitute",
            failure=FailureMode.STATE_UNAVAILABLE,
        )
    effective_quote_reserve = pool.effective_quote_reserve
    if effective_quote_reserve <= 0:
        raise StateUnavailable(
            "the pool's effective quote reserve (vault balance "
            f"{pool.quote_reserve} plus virtual {pool.virtual_quote_reserves}) "
            "is not positive",
            failure=FailureMode.STATE_UNAVAILABLE,
        )

    creator_fee_bps = fees.creator_fee_bps if pool.coin_creator != DEFAULT_PUBKEY else 0
    total_fee_bps = fees.lp_fee_bps + fees.protocol_fee_bps + creator_fee_bps
    if total_fee_bps >= 10_000:
        raise StateUnavailable(
            f"the total fee is {total_fee_bps} bps, which is not a price",
            failure=FailureMode.QUOTE_UNAVAILABLE,
        )

    effective_quote = (sol_amount_lamports * 10_000) // (10_000 + total_fee_bps)
    lp_fee = ceil_fee(effective_quote, fees.lp_fee_bps)
    protocol_fee = ceil_fee(effective_quote, fees.protocol_fee_bps)
    cc_fee = ceil_fee(effective_quote, creator_fee_bps)
    total_with_fees = effective_quote + lp_fee + protocol_fee + cc_fee
    if total_with_fees > sol_amount_lamports:
        # Small orders can round up past the input once fees are ceiled. The
        # protocol backs the fee out of the trade; so does this.
        effective_quote -= total_with_fees - sol_amount_lamports

    net_input = effective_quote - 1
    if net_input <= 0:
        raise StateUnavailable(
            f"{sol_amount_lamports} lamports is below this pool's minimum trade "
            "size once fees are taken",
            failure=FailureMode.QUOTE_UNAVAILABLE,
        )

    out = (pool.base_reserve * net_input) // (effective_quote_reserve + net_input)
    if out <= 0:
        raise StateUnavailable(
            "the pool prices zero output for this size, so there is no quote",
            failure=FailureMode.QUOTE_UNAVAILABLE,
        )

    try:
        price = effective_price(
            sol_amount_lamports, out,
            input_decimals=SOL_DECIMALS, output_decimals=pool.base_decimals,
        )
    except DecimalError as exc:
        raise StateUnavailable(
            f"cannot compute an effective price: {exc}",
            failure=FailureMode.DECIMAL_ERROR,
        ) from exc

    quoted_at = time.time()
    spot = _spot_price(
        numerator=effective_quote_reserve,
        denominator=pool.base_reserve,
        input_decimals=SOL_DECIMALS,
        output_decimals=pool.base_decimals,
    )
    impact = _impact_pct(
        spot_price_numerator=effective_quote_reserve,
        spot_price_denominator=pool.base_reserve,
        executed_price=price,
        input_decimals=SOL_DECIMALS,
        output_decimals=pool.base_decimals,
    )
    return DirectQuoteResult(
        input_amount_raw=sol_amount_lamports,
        output_amount_raw=out,
        input_decimals=SOL_DECIMALS,
        output_decimals=pool.base_decimals,
        fee_bps=total_fee_bps,
        fees=fees,
        effective_price=price,
        price_impact_pct=impact,
        # The execution-native reference: this POOL's own reserves, using the
        # effective quote reserve (vault balance PLUS virtual reserve), which is
        # the reserve the protocol prices against. The vault balance alone is
        # not the market.
        spot_price=spot,
        spot_numerator_raw=effective_quote_reserve,
        spot_denominator_raw=pool.base_reserve,
        amount_after_fee=effective_quote,
        quote_timestamp=quoted_at,
        notes=(
            "quote-in direction: the fee is taken from the input, matching "
            "buyQuoteInput",
            f"effective quote reserve = vault {pool.quote_reserve} + virtual "
            f"{pool.virtual_quote_reserves}",
            "fees are ceiled, matching the SDK (whose inline comments say floor)",
            "canonical pump pool" if pool.is_canonical_pump_pool
            else "non-canonical pool: flat fee schedule, not the tiered one",
        ),
    )


def _spot_price(
    *, numerator: int, denominator: int, input_decimals: int,
    output_decimals: int,
) -> Decimal | None:
    """
    The market's OWN current price, from its own reserves.

    ONE definition, used by every market type and by every consumer, so the
    reference the entry guard compares against and the ``price_impact_pct``
    already on the quote cannot drift apart. Computed in the SAME decimal
    scaling as the executed price, so it cannot be off by 10**6 or 10**9
    relative to it.
    """
    if numerator <= 0 or denominator <= 0:
        return None
    spot = to_decimal(numerator).scaleb(
        output_decimals - input_decimals
    ) / to_decimal(denominator)
    return spot if spot > 0 else None


def _impact_pct(
    *, spot_price_numerator: int, spot_price_denominator: int,
    executed_price: Decimal, input_decimals: int, output_decimals: int,
) -> Decimal | None:
    """
    Price impact against the pool's own spot price.

    Computed from the SAME decimal scaling as the executed price, so it cannot
    be off by 10**6 or 10**9 relative to it.
    """
    spot = _spot_price(
        numerator=spot_price_numerator,
        denominator=spot_price_denominator,
        input_decimals=input_decimals,
        output_decimals=output_decimals,
    )
    if spot is None:
        return None
    return (executed_price / spot - 1) * 100


# ---------------------------------------------------------------------------
# Market classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketClassification:
    """
    Which of A/B/C/D this token is, as a FACT about the chain.

    ``dex_id`` is recorded for contrast and is never the deciding input: a
    graduated coin keeps a ``pumpfun`` label on one aggregator while trading
    on PumpSwap through another, and the ticker says nothing at all.
    """

    market: MarketKind
    curve_address: str | None
    pool_address: str | None
    dex_id: str | None
    evidence: str
    graduated: bool | None = None
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "market": self.market.value,
            "curve_address": self.curve_address,
            "pool_address": self.pool_address,
            "dex_id_hint_not_used_for_classification": self.dex_id,
            "graduated": self.graduated,
            "evidence": self.evidence,
            "notes": list(self.notes),
        }


def classify(
    *,
    curve: BondingCurve | None,
    pool: PumpSwapPool | None,
    dex_id: str | None = None,
) -> MarketClassification:
    """
    Classify from on-chain state alone.

    A: a curve exists, is owned by the pump program, is not complete and still
    holds reserves.
    B: the curve is complete (graduated) or a verified PumpSwap pool exists.
    C: neither, and the provider had a pair on some other venue.
    D: nothing could be established.
    """
    if curve is not None and not curve.graduated:
        return MarketClassification(
            market=MarketKind.PUMP_BONDING_CURVE,
            curve_address=curve.address,
            pool_address=pool.address if pool else None,
            dex_id=dex_id,
            evidence=(
                f"account {curve.address} is owned by the pump.fun program, "
                f"carries the BondingCurve discriminator, is complete="
                f"{str(curve.complete).lower()} and holds "
                f"{curve.virtual_token_reserves} virtual token reserves"
            ),
            graduated=False,
            notes=(
                "classification came from the account's owner and "
                "discriminator, not from the aggregator's dexId label",
            ),
        )
    if curve is not None and curve.graduated:
        return MarketClassification(
            market=MarketKind.PUMPSWAP,
            curve_address=curve.address,
            pool_address=pool.address if pool else None,
            dex_id=dex_id,
            evidence=(
                f"the bonding curve at {curve.address} is complete="
                f"{str(curve.complete).lower()} with virtual token reserves "
                f"{curve.virtual_token_reserves}; a completed curve is no longer "
                "the market"
                + (
                    f" and the canonical PumpSwap pool {pool.address} exists"
                    if pool else
                    " and no canonical PumpSwap pool was found"
                )
            ),
            graduated=True,
            notes=(
                "a token that graduated keeps its pump suffix and often its "
                "pumpfun label, so the label alone would say the wrong thing",
            ),
        )
    if pool is not None:
        return MarketClassification(
            market=MarketKind.PUMPSWAP,
            curve_address=None,
            pool_address=pool.address,
            dex_id=dex_id,
            evidence=(
                f"account {pool.address} is owned by the PumpSwap program, "
                f"carries the Pool discriminator, and its base mint matches "
                "the token under test"
            ),
            graduated=True,
        )
    return MarketClassification(
        market=MarketKind.UNKNOWN,
        curve_address=None,
        pool_address=None,
        dex_id=dex_id,
        evidence=(
            "no pump.fun bonding curve and no PumpSwap pool could be "
            "established for this mint on chain"
            + (
                f"; the aggregator's label was {dex_id!r}, which was not used "
                "as evidence"
                if dex_id else ""
            )
        ),
        graduated=None,
        notes=(
            "classification is UNKNOWN, which is not the same as C (another "
            "market): a missing account and an unreadable account are different "
            "findings and only the chain can tell them apart",
        ),
    )


__all__ = [
    "AMM_FEE_CONFIG_PDA",
    "AMM_GLOBAL_CONFIG_DISCRIMINATOR",
    "AMM_GLOBAL_CONFIG_PDA",
    "AmmGlobalConfig",
    "BondingCurve",
    "CURVE_DISCRIMINATOR",
    "DEFAULT_PUBKEY",
    "DirectQuoteResult",
    "FeeConfig",
    "FeeTier",
    "Fees",
    "GLOBAL_DISCRIMINATOR",
    "GLOBAL_PDA",
    "PUMP_FEE_CONFIG_PDA",
    "PUMP_FEE_PROGRAM",
    "PUMP_FUN_PROGRAM",
    "PUMP_SWAP_PROGRAM",
    "POOL_DISCRIMINATOR",
    "PumpGlobal",
    "PumpSwapPool",
    "MarketClassification",
    "SOL_LIKE_QUOTE_MINTS",
    "STABLE_QUOTE_MINTS",
    "StateUnavailable",
    "ceil_fee",
    "classify",
    "curve_fee_schedule",
    "decode_amm_global_config",
    "decode_bonding_curve",
    "decode_fee_config",
    "decode_global",
    "decode_pool",
    "decode_pumpswap_pool",
    "decode_spl_mint",
    "decode_token_account_amount",
    "derive_bonding_curve",
    "derive_canonical_pool",
    "derive_pool_authority",
    "amm_fee_schedule",
    "quote_bonding_curve_buy",
    "quote_bonding_curve_sell",
    "quote_pumpswap_buy",
    "quote_pumpswap_sell",
]


# ---------------------------------------------------------------------------
# SELL direction - TOKEN -> SOL/wSOL
# ---------------------------------------------------------------------------
#
# SELL is the mirror of BUY and the fee side INVERTS. On a buy the fee comes out
# of the INPUT before the curve; on a sell it comes out of the OUTPUT after it.
# Using the buy formula for a sell does not produce an obviously wrong number -
# it produces one of the right order of magnitude, which is the most dangerous
# kind.
#
# Both functions below are transcribed from the official SDK and verified
# against it on live mainnet state.


def quote_bonding_curve_sell(
    curve: BondingCurve,
    *,
    token_amount_raw: int,
    token_decimals: int,
    fees: Fees,
) -> DirectQuoteResult:
    """
    A SELL against the bonding curve: ``TOKEN -> SOL/wSOL``.

    Reproduces ``getSellSolAmountFromTokenAmount`` exactly::

        gross = amount * virtualQuoteReserves // (virtualTokenReserves + amount)
        net   = gross - protocolFee(gross) - creatorFee(gross)

    Both fees are CEILED, and the creator fee is charged only when the curve has
    a creator. A graduated curve returns zero in the protocol and is refused
    here.

    ``token_decimals`` is a PARAMETER, not something read off the curve: the
    ``BondingCurve`` layout has no decimals field, so the only correct source is
    the mint account, which the caller reads.
    """
    if token_amount_raw <= 0:
        raise StateUnavailable(
            "the sell amount must be positive",
            failure=FailureMode.QUOTE_UNAVAILABLE,
        )
    if curve.graduated:
        raise StateUnavailable(
            "the bonding curve is complete, so it is no longer the market and "
            "there is nothing to sell into on it",
            failure=FailureMode.CURVE_UNAVAILABLE,
        )
    if curve.virtual_quote_reserves <= 0:
        raise StateUnavailable(
            "the curve's virtual quote reserve is zero, so no sell price is defined",
            failure=FailureMode.CURVE_UNAVAILABLE,
        )

    gross = (token_amount_raw * curve.virtual_quote_reserves) // (
        curve.virtual_token_reserves + token_amount_raw
    )
    if gross <= 0:
        raise StateUnavailable(
            "the curve models zero proceeds for this size, so there is no quote",
            failure=FailureMode.QUOTE_UNAVAILABLE,
        )

    protocol_fee = ceil_fee(gross, fees.protocol_fee_bps)
    creator_fee = (
        ceil_fee(gross, fees.creator_fee_bps)
        if curve.creator != DEFAULT_PUBKEY
        else 0
    )
    charged_creator_bps = fees.creator_fee_bps if curve.creator != DEFAULT_PUBKEY else 0
    net = gross - protocol_fee - creator_fee
    if net <= 0:
        raise StateUnavailable(
            f"fees ({protocol_fee + creator_fee}) consume the entire gross "
            f"proceeds ({gross}); there is no quote",
            failure=FailureMode.QUOTE_UNAVAILABLE,
        )

    try:
        # SOL received per whole token.
        price = effective_price(
            net, token_amount_raw,
            input_decimals=SOL_DECIMALS, output_decimals=token_decimals,
        )
    except DecimalError as exc:
        raise StateUnavailable(
            f"cannot compute an effective sell price: {exc}",
            failure=FailureMode.DECIMAL_ERROR,
        ) from exc

    quoted_at = time.time()
    spot = _spot_price(
        numerator=curve.virtual_quote_reserves,
        denominator=curve.virtual_token_reserves,
        input_decimals=SOL_DECIMALS,
        output_decimals=token_decimals,
    )
    impact = _impact_pct(
        spot_price_numerator=curve.virtual_quote_reserves,
        spot_price_denominator=curve.virtual_token_reserves,
        executed_price=price,
        input_decimals=SOL_DECIMALS,
        output_decimals=token_decimals,
    )
    return DirectQuoteResult(
        input_amount_raw=token_amount_raw,
        output_amount_raw=net,
        input_decimals=token_decimals,
        output_decimals=SOL_DECIMALS,
        quote_asset_input=False,
        fee_bps=fees.protocol_fee_bps + charged_creator_bps,
        fees=fees,
        effective_price=price,
        price_impact_pct=impact,
        # THE EXECUTION-NATIVE REFERENCE, SELL SIDE. It is the SAME curve
        # reserves, read in the SAME account fetch, at the SAME slot, that
        # ``price`` was computed from -- exactly as the BUY result already
        # carries them. Omitting it here meant every SELL quote arrived at
        # ``QuoteService`` with no ``spot_price``, so
        # ``_execution_native_reference`` returned ``None`` for the whole
        # hold path: the exit trigger had no same-market price to compare
        # against, started its data-invalid timer, and closed a position
        # whose market was readable, identified and quoted the entire time.
        spot_price=spot,
        spot_numerator_raw=curve.virtual_quote_reserves,
        spot_denominator_raw=curve.virtual_token_reserves,
        amount_after_fee=gross,
        quote_timestamp=quoted_at,
        notes=(
            "SELL direction: the fee is taken from the OUTPUT after the curve, "
            "matching getSellSolAmountFromTokenAmount - the mirror image of the "
            "buy path",
            f"gross proceeds {gross}, fees {protocol_fee + creator_fee}, net {net}",
        ),
    )


def quote_pumpswap_sell(
    pool: PumpSwapPool,
    *,
    token_amount_raw: int,
    fees: Fees,
) -> DirectQuoteResult:
    """
    A SELL against a PumpSwap pool: ``TOKEN -> SOL/wSOL``.

    Reproduces ``sellBaseInput`` exactly::

        gross = effectiveQuoteReserve * base // (baseReserve + base)
        net   = gross - lpFee(gross) - protocolFee(gross) - creatorFee(gross)

    All three fees are CEILED off the OUTPUT. The SDK additionally refuses when
    the raw quote vault cannot cover the output net of the LP fee; that check
    is reproduced, because it is what stops a sell being quoted against reserves
    the pool would not actually pay out of.
    """
    if token_amount_raw <= 0:
        raise StateUnavailable(
            "the sell amount must be positive",
            failure=FailureMode.QUOTE_UNAVAILABLE,
        )
    if pool.base_reserve <= 0:
        raise StateUnavailable(
            f"pool base reserve is {pool.base_reserve}; nothing to sell into",
            failure=FailureMode.STATE_UNAVAILABLE,
        )
    effective_quote_reserve = pool.effective_quote_reserve
    if effective_quote_reserve <= 0:
        raise StateUnavailable(
            "the pool's effective quote reserve is not positive",
            failure=FailureMode.STATE_UNAVAILABLE,
        )

    gross = (effective_quote_reserve * token_amount_raw) // (
        pool.base_reserve + token_amount_raw
    )
    if gross <= 0:
        raise StateUnavailable(
            "the pool models zero proceeds for this size, so there is no quote",
            failure=FailureMode.QUOTE_UNAVAILABLE,
        )

    creator_fee_bps = fees.creator_fee_bps if pool.coin_creator != DEFAULT_PUBKEY else 0
    lp_fee = ceil_fee(gross, fees.lp_fee_bps)
    protocol_fee = ceil_fee(gross, fees.protocol_fee_bps)
    creator_fee = ceil_fee(gross, creator_fee_bps)

    if pool.quote_reserve < gross - lp_fee:
        raise StateUnavailable(
            "the pool's real quote reserves cannot cover this sell "
            f"(vault {pool.quote_reserve} < net of LP fee {gross - lp_fee}); "
            "the SDK refuses this trade and so does this quote",
            failure=FailureMode.STATE_UNAVAILABLE,
        )

    net = gross - lp_fee - protocol_fee - creator_fee
    if net <= 0:
        raise StateUnavailable(
            "fees consume the entire gross proceeds; there is no quote",
            failure=FailureMode.QUOTE_UNAVAILABLE,
        )

    try:
        price = effective_price(
            net, token_amount_raw,
            input_decimals=SOL_DECIMALS, output_decimals=pool.base_decimals,
        )
    except DecimalError as exc:
        raise StateUnavailable(
            f"cannot compute an effective sell price: {exc}",
            failure=FailureMode.DECIMAL_ERROR,
        ) from exc

    quoted_at = time.time()
    spot = _spot_price(
        numerator=effective_quote_reserve,
        denominator=pool.base_reserve,
        input_decimals=SOL_DECIMALS,
        output_decimals=pool.base_decimals,
    )
    impact = _impact_pct(
        spot_price_numerator=effective_quote_reserve,
        spot_price_denominator=pool.base_reserve,
        executed_price=price,
        input_decimals=SOL_DECIMALS,
        output_decimals=pool.base_decimals,
    )
    return DirectQuoteResult(
        input_amount_raw=token_amount_raw,
        output_amount_raw=net,
        input_decimals=pool.base_decimals,
        output_decimals=SOL_DECIMALS,
        quote_asset_input=False,
        fee_bps=fees.lp_fee_bps + fees.protocol_fee_bps + creator_fee_bps,
        fees=fees,
        effective_price=price,
        price_impact_pct=impact,
        # THE EXECUTION-NATIVE REFERENCE, SELL SIDE -- same pool, same fetch,
        # same slot as ``price``, mirroring ``quote_pumpswap_buy``.
        spot_price=spot,
        spot_numerator_raw=effective_quote_reserve,
        spot_denominator_raw=pool.base_reserve,
        amount_after_fee=gross,
        quote_timestamp=quoted_at,
        notes=(
            "SELL direction: lp, protocol and creator fees are ceiled off the "
            "OUTPUT, matching sellBaseInput",
            f"effective quote reserve = vault {pool.quote_reserve} + virtual "
            f"{pool.virtual_quote_reserves}",
            "the vault-covers-output check is reproduced, so a sell is never "
            "quoted against reserves the pool would not pay from",
        ),
    )
