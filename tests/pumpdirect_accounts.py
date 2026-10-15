"""
Account fixtures, encoded from the DOCUMENTED layouts field by field.

WHY THIS EXISTS INSTEAD OF A HEX BLOB
=====================================

If the fixture bytes and the decoder were both produced by the same mistake -
one offset shifted in the builder and the same offset shifted in the decoder -
every test would pass and the harness would price the wrong thing. So the
fixtures below are assembled field by field from the same field lists the
official IDL describes, in order, and the tests additionally assert the offsets
directly.

Every length here was confirmed against live mainnet:

* ``BondingCurve`` is 125 bytes (legacy) or 151 (current);
* ``Pool`` is 270 (legacy) or 300 (current);
* ``FeeConfig`` is 2512 / 4073 / 4097 depending on which tier vectors exist;
* ``Global`` is 1087; ``GlobalConfig`` is 949.

The variable-length cases matter: a decoder that assumes the largest size will
read past the end of an older account, and one that assumes the smallest will
report a missing field for a newer one. Both are exercised.
"""

from __future__ import annotations

from autopaper.same_market import pumpdirect as pd
from autopaper.same_market.pda import b58decode_pubkey, b58encode

TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"


def _u64(value: int) -> bytes:
    return int(value).to_bytes(8, "little")


def _u128(value: int) -> bytes:
    return int(value).to_bytes(16, "little")


def _i128(value: int) -> bytes:
    return int(value).to_bytes(16, "little", signed=True)


def _u16(value: int) -> bytes:
    return int(value).to_bytes(2, "little")


def _pk(address: str) -> bytes:
    return b58decode_pubkey(address)


def _pad(data: bytes, size: int) -> bytes:
    """Zero-extend to ``size``. Anchor grew these accounts over time."""
    return data + b"\x00" * max(0, size - len(data))


def bonding_curve_bytes(
    *,
    virtual_token_reserves: int = 320_098_471_659_904,
    virtual_quote_reserves: int = 100_562_804_716,
    real_token_reserves: int = 40_198_471_659_904,
    real_quote_reserves: int = 0,
    token_total_supply: int = 1_000_000_000_000_000,
    complete: bool = False,
    creator: str = "5L5k7gtNLbeXdzpvNrFshg1E1id1ceUDfc6vPUTxp98q",
    is_mayhem_mode: bool = False,
    is_cashback_coin: bool = False,
    quote_mint: str = pd.DEFAULT_PUBKEY,
    creator_fee_bps: int = 0,
    size: int = 151,
) -> bytes:
    """
    A ``BondingCurve`` in the documented field order.

    NOTE WHAT IS NOT HERE: a ``mint`` field. The current layout has no such
    field - the mint is recoverable only from the PDA seeds. A fixture that
    included one would have hidden the single most important fact about this
    account.
    """
    body = (
        pd.CURVE_DISCRIMINATOR
        + _u64(virtual_token_reserves)
        + _u64(virtual_quote_reserves)
        + _u64(real_token_reserves)
        + _u64(real_quote_reserves)
        + _u64(token_total_supply)
        + bytes([1 if complete else 0])
        + _pk(creator)
        + bytes([1 if is_mayhem_mode else 0])
        + bytes([1 if is_cashback_coin else 0])
        + _pk(quote_mint)
        + _u64(creator_fee_bps)
        + bytes([0])  # can_edit_creator_fee
    )
    return _pad(body, size)


def global_bytes(
    *,
    fee_basis_points: int = 95,
    creator_fee_basis_points: int = 5,
    token_total_supply: int = 1_000_000_000_000_000,
    creator_fee_configurable: bool = False,
    size: int = 1087,
) -> bytes:
    """``Global``. Only the fee fields this harness reads are meaningful."""
    body = bytearray(b"\x00" * size)
    body[0:8] = pd.GLOBAL_DISCRIMINATOR
    body[8] = 1  # initialized
    body[97:105] = _u64(token_total_supply)
    body[105:113] = _u64(fee_basis_points)
    body[154:162] = _u64(creator_fee_basis_points)
    body[1045] = 1 if creator_fee_configurable else 0
    body[1046:1054] = _u64(500)
    return bytes(body)


def _fees(lp: int, protocol: int, creator: int) -> bytes:
    return _u64(lp) + _u64(protocol) + _u64(creator)


def _tier(threshold: int, lp: int, protocol: int, creator: int) -> bytes:
    return _u128(threshold) + _fees(lp, protocol, creator)


def fee_config_bytes(
    *,
    flat: tuple[int, int, int] = (25, 5, 0),
    tiers: list[tuple[int, int, int, int]] | None = None,
    stable_tiers: list[tuple[int, int, int, int]] | None = None,
    exotic: tuple[int, int, int] = (20, 5, 5),
) -> bytes:
    """
    A ``FeeConfig``.

    The default reproduces the live AMM schedule read off mainnet on
    2026-10-02: flat 25/5/0, and canonical tiers of 125 bps below 420 SOL then
    stepping down. Reproducing it exactly matters, because reading the FLAT
    rate where a tier belongs would overstate a canonical pool's fee by an
    order of magnitude in basis points.
    """
    tiers = tiers if tiers is not None else [
        (0, 2, 93, 30),
        (420_000_000_000, 20, 5, 95),
        (1_470_000_000_000, 20, 5, 90),
    ]
    stable_tiers = stable_tiers if stable_tiers is not None else [
        (59_000_000, 2, 93, 30),
    ]
    body = (
        pd.FEE_CONFIG_DISCRIMINATOR
        + bytes([253])                       # bump
        + _pk("FFWtrEQ4B4PKQoVuHYzZq8FabGkVatYzDpEVHsK5rrhF")  # admin
        + _fees(*flat)
        + len(tiers).to_bytes(4, "little")
        + b"".join(_tier(*t) for t in tiers)
        + len(stable_tiers).to_bytes(4, "little")
        + b"".join(_tier(*t) for t in stable_tiers)
        + _fees(*exotic)
    )
    return _pad(body, 4097)


def curve_fee_config_bytes() -> bytes:
    """
    The curve ``FeeConfig`` as read from mainnet: one tier, 95/30.

    The `Global` fallback says 95 + 5, so reading the fallback where the live
    schedule belongs would understate the creator fee by 25 bps.
    """
    return fee_config_bytes(
        flat=(0, 95, 30),
        tiers=[(0, 0, 95, 30)],
        stable_tiers=[(0, 0, 95, 30)],
        exotic=(0, 95, 30),
    )


def amm_global_config_bytes(
    *,
    lp_fee_basis_points: int = 20,
    protocol_fee_basis_points: int = 5,
    coin_creator_fee_basis_points: int = 0,
    creator_fee_configurable: bool = False,
    size: int = 949,
) -> bytes:
    """``GlobalConfig``. ``coin_creator_fee_basis_points`` is at 313, not 57."""
    body = bytearray(b"\x00" * size)
    body[0:8] = pd.AMM_GLOBAL_CONFIG_DISCRIMINATOR
    body[8:40] = _pk("11111111111111111111111111111111")
    body[40:48] = _u64(lp_fee_basis_points)
    body[48:56] = _u64(protocol_fee_basis_points)
    body[56] = 0  # disable_flags
    body[313:321] = _u64(coin_creator_fee_basis_points)
    body[940] = 1 if creator_fee_configurable else 0
    body[941:949] = _u64(500)
    return bytes(body)


def pool_bytes(
    *,
    pool_bump: int = 254,
    index: int = 0,
    creator: str,
    base_mint: str,
    quote_mint: str = "So11111111111111111111111111111111111111112",
    lp_mint: str = "6dpnPD6UWDw5hbJEuPQwnCCMba1JYwHANKuL6GQ6otAH",
    pool_base_token_account: str,
    pool_quote_token_account: str,
    lp_supply: int = 4_194_352_106_721,
    coin_creator: str = "5L5k7gtNLbeXdzpvNrFshg1E1id1ceUDfc6vPUTxp98q",
    is_mayhem_mode: bool = False,
    is_cashback_coin: bool = False,
    virtual_quote_reserves: int = 0,
    creator_fee_bps: int = 0,
    size: int = 300,
) -> bytes:
    """A PumpSwap ``Pool`` in the documented field order."""
    body = (
        pd.POOL_DISCRIMINATOR
        + bytes([pool_bump])
        + _u16(index)
        + _pk(creator)
        + _pk(base_mint)
        + _pk(quote_mint)
        + _pk(lp_mint)
        + _pk(pool_base_token_account)
        + _pk(pool_quote_token_account)
        + _u64(lp_supply)
        + _pk(coin_creator)
        + bytes([1 if is_mayhem_mode else 0])
        + bytes([1 if is_cashback_coin else 0])
        + _i128(virtual_quote_reserves)
        + _u64(creator_fee_bps)
        + bytes([0])  # can_edit_creator_fee
    )
    return _pad(body, size)


def mint_bytes(*, decimals: int = 6, supply: int = 1_000_000_000_000_000) -> bytes:
    """An SPL ``Mint``: supply at 36, decimals at 44."""
    body = bytearray(b"\x00" * 82)
    body[0:4] = (0).to_bytes(4, "little")   # no mint authority
    body[36:44] = _u64(supply)
    body[44] = decimals
    body[45] = 1                          # initialized
    return bytes(body)


def token_account_bytes(*, amount: int, mint: str) -> bytes:
    """An SPL ``Token`` account. The balance is at offset 64."""
    body = bytearray(b"\x00" * 165)
    body[0:32] = _pk(mint)
    body[32:64] = _pk("11111111111111111111111111111111")
    body[64:72] = _u64(amount)
    return bytes(body)


#: A real, live mainnet token that has NOT graduated, with the exact addresses
#: confirmed on 2026-10-02. Used as the "ordinary case" so the tests exercise a
#: genuine shape rather than a contrived one.
LIVE_CURVE_MINT = "APxxh2tWCh95tbHitFJchcdbMsctSBgRS2Zzju9Bpump"
LIVE_CURVE_ADDRESS = "9LCvns4zZsYAinZcEWRdzEEz7P6V84K3c5ZYCLT6vzMb"
#: A real, live mainnet token that HAS graduated to PumpSwap.
LIVE_AMM_MINT = "7LSsEoJGhLeZzGvDofTdNg7M3JttxQqGWNLo6vWMpump"
#: The same mint's bonding curve. Present but COMPLETE - a graduated coin keeps
#: its curve account, drained to zero, which is why "no curve" is not the test
#: for graduation.
LIVE_OTHER_CURVE_ADDRESS = "3MUkKMbuornHohtAtzrToSzqkj1gEEhQqYVz8sZnmQg1"
LIVE_POOL_ADDRESS = "GseMAnNDvntR5uFePZ51yZBXzNSn7GdFPkfHwfr6d77J"

#: Reserves read from that live pool on 2026-10-02, and the SDK's own answer
#: for a 1,879,699-lamport BUY: 9,466,908,943 base units. Jupiter returned the
#: identical figure, so this is a three-way exact agreement.
LIVE_POOL_BASE_RESERVE = 649_904_538_581_704
LIVE_POOL_QUOTE_RESERVE = 127_446_425_099
LIVE_POOL_SUPPLY = 999_211_843_792_757
LIVE_BUY_LAMPORTS = 1_879_699          # $0.23 at $122.36/SOL
LIVE_SDK_POOL_OUTPUT = 9_466_908_943
#: The official SDK's answer for EXACTLY the reserves the fixture above holds,
#: for the same 1,879,699-lamport BUY: 5,909,232,175. Verified by replaying
#: @pump-fun/pump-sdk's getBuyTokenAmountFromSolAmount against those numbers.
#:
#: An earlier snapshot of the same curve produced 5,176,254,910 - the curve
#: trades continuously, so the two figures are both correct for their own
#: state. Pairing the answer with the wrong reserves would make this a test
#: that can never pass.
LIVE_SDK_CURVE_OUTPUT = 5_909_232_175


__all__ = [
    "LIVE_AMM_MINT",
    "LIVE_BUY_LAMPORTS",
    "LIVE_CURVE_ADDRESS",
    "LIVE_CURVE_MINT",
    "LIVE_OTHER_CURVE_ADDRESS",
    "LIVE_POOL_ADDRESS",
    "LIVE_POOL_BASE_RESERVE",
    "LIVE_POOL_QUOTE_RESERVE",
    "LIVE_POOL_SUPPLY",
    "LIVE_SDK_CURVE_OUTPUT",
    "LIVE_SDK_POOL_OUTPUT",
    "TOKEN_PROGRAM",
    "amm_global_config_bytes",
    "bonding_curve_bytes",
    "curve_fee_config_bytes",
    "fee_config_bytes",
    "global_bytes",
    "mint_bytes",
    "pool_bytes",
    "token_account_bytes",
]
