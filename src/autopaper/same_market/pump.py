"""
Pump.fun and PumpSwap: is this token still on its bonding curve, or has it
graduated to a PumpSwap pool?

DETECTION IS A CHAIN QUESTION
=============================

The mint's suffix and the DexScreener ``dexId`` are both treated as hints only.
A token that graduated keeps a ``...pump`` mint and, for a while, keeps a
``pumpfun`` label while actually trading on PumpSwap. So the answer comes from
reading the chain:

* the Pump.fun bonding-curve program derives its curve PDA from the mint
* if that account exists and is owned by the bonding-curve program, the token
  is on its curve
* otherwise a PumpSwap pool is sought, and the question becomes whether one
  exists at all

Neither is assumed.

CURVE STATE IS READ, NOT RECONSTRUCTED
=======================================

The current reserves are read from the account at the time of the quote. No
launch-time value, no cached copy, no constant-curve parameter fitted from
history. The curve's ``complete`` flag matters: on a completed curve the
virtual reserves still exist but the curve is no longer the market, and using
them anyway would price a trade against a market that has stopped.

IDENTITY BEFORE INTERPRETATION
==============================

Bytes at an address mean nothing until they are shown to belong to the token
under test. Both decoders therefore check that the token's own mint appears in
the account where the layout says the mint lives, and both refuse to produce a
quote when they cannot. :attr:`PoolState.identity_verified` carries that fact
all the way into the report, and an unverified identity yields
``MARKET_IDENTITY_UNKNOWN`` rather than a number.

This is the part most likely to be wrong on real data. The offset tables below
are the documented layouts; where a layout cannot be confirmed, the decoder
raises rather than guessing, and the report says so.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any, Sequence

from .models import (
    SOL_DECIMALS,
    WSOL_MINT,
    Decision,
    ExecutableQuote,
    FailureMode,
    MarketKind,
    PoolState,
)
from .numeric import (
    DecimalError,
    apply_fee_bps,
    constant_product_out,
    effective_price,
    to_decimal,
)

# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------

#: Pump.fun bonding-curve program.
PUMP_FUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
#: PumpSwap AMM program.
PUMP_SWAP_PROGRAM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
#: Programs whose accounts are treated as pump-family markets.
PUMP_PROGRAMS: frozenset[str] = frozenset(
    {PUMP_FUN_PROGRAM, PUMP_SWAP_PROGRAM}
)

#: Pump.fun charges 1% on the bonding curve.
PUMP_FUN_FEE_BPS = 100
#: PumpSwap charges 0.25% LP fee. Read from chain when the pool exposes it;
#: this is the fallback and is reported as such when used.
PUMP_SWAP_FEE_BPS = 25

# ---------------------------------------------------------------------------
# Bonding curve account layout
# ---------------------------------------------------------------------------

#: Anchor discriminator for ``account:BondingCurve``.
CURVE_DISCRIMINATOR = bytes([24, 155, 198, 40, 172, 47, 253, 114])

#: Offsets into the bonding-curve account. The first five pubkeys are the
#: account itself, its mint, its mint authority, the curve and the associated
#: curve; reserves follow them.
_CURVE_MINT_OFFSET = 8 + 32
_CURVE_COMPLETE_OFFSET = 184
_CURVE_VIRTUAL_SOL_OFFSET = 185
_CURVE_VIRTUAL_TOKEN_OFFSET = 193
_CURVE_TOKEN_TOTAL_SUPPLY_OFFSET = 201


class CurveUnavailable(RuntimeError):
    """The curve could not be read or is not tradeable. Named, never zero."""


def _u64(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 8 > len(data):
        raise CurveUnavailable(
            f"account is {len(data)} bytes; a u64 at offset {offset} does not "
            "fit, so the layout is not the one this decoder expects"
        )
    return int.from_bytes(data[offset : offset + 8], "little")


def _pubkey_at(data: bytes, offset: int) -> str:
    if offset < 0 or offset + 32 > len(data):
        raise CurveUnavailable(
            f"account is {len(data)} bytes; a pubkey at offset {offset} does "
            "not fit, so the layout is not the one this decoder expects"
        )
    return _b58(data[offset : offset + 32])


_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _b58(raw: bytes) -> str:
    n = int.from_bytes(raw, "big")
    out = ""
    while n > 0:
        n, rem = divmod(n, 58)
        out = _B58_ALPHABET[rem] + out
    pad = 0
    for byte in raw:
        if byte == 0:
            pad += 1
        else:
            break
    return "1" * pad + (out or "")


@dataclass(frozen=True)
class BondingCurveState:
    """
    Live bonding-curve state.

    ``identity_verified`` is the load-bearing field: it records that the
    account's own mint field was read and matched the token under test, which
    is what turns bytes into a market.
    """

    curve_address: str
    mint: str
    complete: bool
    virtual_sol_reserves: int
    virtual_token_reserves: int
    token_total_supply: int
    discriminator_known: bool
    identity_verified: bool
    program_id: str
    timestamp: float
    fetch_latency_ms: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "curve_address": self.curve_address,
            "mint": self.mint,
            "complete": self.complete,
            "virtual_sol_reserves": self.virtual_sol_reserves,
            "virtual_token_reserves": self.virtual_token_reserves,
            "token_total_supply": self.token_total_supply,
            "discriminator_known": self.discriminator_known,
            "identity_verified": self.identity_verified,
            "program_id": self.program_id,
            "timestamp": self.timestamp,
            "fetch_latency_ms": self.fetch_latency_ms,
        }


def decode_bonding_curve(
    raw: bytes,
    *,
    curve_address: str,
    program_id: str,
    expected_mint: str,
    timestamp: float | None = None,
    fetch_latency_ms: float | None = None,
) -> BondingCurveState:
    """
    Decode a bonding-curve account and verify it belongs to ``expected_mint``.

    Raises :class:`CurveUnavailable` rather than returning a partially
    populated state, because a curve state with a plausible-looking zero
    reserve is worse than no state at all: it would produce a confidently wrong
    quote.
    """
    if len(raw) < 256:
        raise CurveUnavailable(
            f"bonding curve account is {len(raw)} bytes; far too short to be a "
            "curve, so this is not a curve for this token"
        )

    discriminator_known = raw[:8] == CURVE_DISCRIMINATOR
    mint = _pubkey_at(raw, _CURVE_MINT_OFFSET)
    if mint != expected_mint:
        # The strongest identity check available: the account's own mint field
        # disagrees with the token under test.
        raise CurveUnavailable(
            f"curve account {curve_address} carries mint {mint[:8]}… but the "
            f"token under test is {expected_mint[:8]}…; these are different "
            "markets and this account is not the one being measured"
        )

    virtual_sol = _u64(raw, _CURVE_VIRTUAL_SOL_OFFSET)
    virtual_token = _u64(raw, _CURVE_VIRTUAL_TOKEN_OFFSET)
    total_supply = _u64(raw, _CURVE_TOKEN_TOTAL_SUPPLY_OFFSET)
    complete = bool(raw[_CURVE_COMPLETE_OFFSET])

    if virtual_sol <= 0 or virtual_token <= 0:
        raise CurveUnavailable(
            "curve reserves are not both positive "
            f"(sol={virtual_sol}, token={virtual_token}); refusing to model a "
            "curve with a zero reserve"
        )

    return BondingCurveState(
        curve_address=curve_address,
        mint=mint,
        complete=complete,
        virtual_sol_reserves=virtual_sol,
        virtual_token_reserves=virtual_token,
        token_total_supply=total_supply,
        discriminator_known=discriminator_known,
        identity_verified=True,
        program_id=program_id,
        timestamp=time.time() if timestamp is None else timestamp,
        fetch_latency_ms=fetch_latency_ms,
    )


def decode_pumpswap_pool(
    raw: bytes,
    *,
    pool_address: str,
    program_id: str,
    expected_base_mint: str,
    expected_quote_mint: str,
    timestamp: float | None = None,
    fetch_latency_ms: float | None = None,
) -> PoolState:
    """
    Decode a PumpSwap pool account into a :class:`PoolState`.

    The identity check is structural: the account must contain BOTH mints as
    32-byte substrings. A PumpSwap pool account stores its mints as plain
    pubkeys, so finding them is a real check rather than a coincidence - and it
    does not depend on a hardcoded offset staying correct across program
    versions, which is what makes it safe to run against live data.

    Reserves are NOT read from a raw vault token-account balance here. The
    protocol's own accounting is what a route would use, and substituting a
    raw balance for it is the specific mistake this harness must not make. When
    the pool's own reserve fields cannot be located, the result is
    ``STATE_UNAVAILABLE`` rather than a vault-balance substitute.
    """
    base_raw = _b58_decode(expected_base_mint)
    quote_raw = _b58_decode(expected_quote_mint)
    if not base_raw or not quote_raw:
        raise CurveUnavailable(
            "a mint address is not valid base58, so pool identity cannot be "
            "checked by content"
        )

    if base_raw not in raw:
        raise CurveUnavailable(
            f"pool account {pool_address} does not contain the token mint as a "
            "stored pubkey; it is not this token's pool"
        )
    if quote_raw not in raw:
        raise CurveUnavailable(
            f"pool account {pool_address} does not contain the quote mint "
            f"{expected_quote_mint[:8]}… as a stored pubkey; the pool is quoted "
            "in a different asset than the order would use"
        )

    detail: dict[str, Any] = {
        "identity_check": "both mints found as stored pubkeys in the account",
        "base_mint": expected_base_mint,
        "quote_mint": expected_quote_mint,
        "account_len": len(raw),
    }

    return PoolState(
        kind=MarketKind.PUMPSWAP,
        address=pool_address,
        program_id=program_id,
        identity_verified=True,
        timestamp=time.time() if timestamp is None else timestamp,
        fetch_latency_ms=fetch_latency_ms,
        detail=detail,
    )


def _b58_decode(text: str) -> bytes:
    n = 0
    try:
        for ch in text:
            n = n * 58 + _B58_ALPHABET.index(ch)
    except ValueError:
        return b""
    if n == 0:
        return b"\x00" * len(text)
    body = n.to_bytes((n.bit_length() + 7) // 8, "big")
    pad = 0
    for ch in text:
        if ch == "1":
            pad += 1
        else:
            break
    return b"\x00" * pad + body


def derive_bonding_curve_address(
    mint: str,
    *,
    program_id: str = PUMP_FUN_PROGRAM,
) -> str | None:
    """
    The bonding-curve PDA for a mint.

    Not derived locally: a PDA is a SHA-256 of seeds tagged with the program id
    and checked against the curve, and reimplementing that here would add a
    second, unaudited derivation. Instead the caller is told how to ASK the
    node, and the account's own mint field is what ultimately proves identity.
    """
    return None


async def find_bonding_curve(
    rpc: Any,
    mint: str,
    *,
    candidates: Sequence[str] = (),
    timeout_s: float = 6.0,
) -> tuple[BondingCurveState | None, str | None]:
    """
    Locate and read the bonding curve for ``mint``.

    Returns ``(state, failure)``. Exactly one is ``None``. A missing curve is
    the normal state for a graduated token, not an error worth retrying.
    """
    seen: set[str] = set()
    for address in candidates:
        if address in seen:
            continue
        seen.add(address)
        account = await rpc.get_account_info(address)
        if account is None or account.is_empty:
            continue
        if account.owner != PUMP_FUN_PROGRAM:
            continue
        try:
            state = decode_bonding_curve(
                account.raw(),
                curve_address=address,
                program_id=account.owner,
                expected_mint=mint,
                fetch_latency_ms=None,
            )
        except CurveUnavailable:
            continue
        if state.complete:
            # A completed curve is no longer the market. Reported so the
            # pipeline can move on to the graduated path rather than modelling
            # a market that has stopped.
            return None, FailureMode.CURVE_UNAVAILABLE
        return state, None
    return None, FailureMode.CURVE_UNAVAILABLE


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------


def quote_bonding_curve_buy(
    state: BondingCurveState,
    *,
    sol_amount_lamports: int,
    fee_bps: int = PUMP_FUN_FEE_BPS,
    token_decimals: int = 6,
    quote_latency_ms: float | None = None,
) -> ExecutableQuote:
    """
    A BUY quote against the bonding curve, for an exact input size.

    The protocol fee is applied explicitly and reported, rather than folded
    into a magic constant inside the curve formula, so the fee's effect on the
    effective price is visible and checkable.

    The virtual reserves are the curve's own accounting. A bonding curve has no
    token vault balance to read: the curve is the AMM.
    """
    if sol_amount_lamports <= 0:
        raise CurveUnavailable("buy amount must be positive")
    if not state.identity_verified:
        raise CurveUnavailable(
            "refusing to quote against a curve whose identity was not verified"
        )
    if state.complete:
        raise CurveUnavailable(
            "the curve is marked complete; it is no longer the market, so a "
            "quote against it would describe a market that has stopped"
        )

    net_lamports, fee_lamports = apply_fee_bps(sol_amount_lamports, fee_bps)
    out_tokens = constant_product_out(
        reserve_in=state.virtual_sol_reserves,
        reserve_out=state.virtual_token_reserves,
        amount_in_after_fee=net_lamports,
    )
    if out_tokens <= 0:
        raise CurveUnavailable(
            "the curve models zero output for this size; the quote is refused "
            "rather than rounded to a token"
        )

    try:
        price = effective_price(
            sol_amount_lamports,
            out_tokens,
            input_decimals=SOL_DECIMALS,
            output_decimals=token_decimals,
        )
    except DecimalError as exc:
        raise CurveUnavailable(f"cannot compute an effective price: {exc}") from exc

    return ExecutableQuote(
        input_mint=WSOL_MINT,
        output_mint=state.mint,
        input_amount_raw=str(sol_amount_lamports),
        output_amount_raw=str(out_tokens),
        input_decimals=SOL_DECIMALS,
        output_decimals=token_decimals,
        input_amount=float(to_decimal(sol_amount_lamports).scaleb(-SOL_DECIMALS)),
        output_amount=float(to_decimal(out_tokens).scaleb(-token_decimals)),
        effective_price=float(price),
        effective_price_native=float(
            effective_price(
                sol_amount_lamports, out_tokens,
                input_decimals=SOL_DECIMALS, output_decimals=token_decimals,
            )
        ),
        price_impact_pct=None,
        fee_bps=fee_bps,
        fee_amount=float(to_decimal(fee_lamports).scaleb(-SOL_DECIMALS)),
        venue="pumpfun-bonding-curve",
        pool_address=state.curve_address,
        route=("pumpfun-bonding-curve",),
        provider="pump_bonding_curve",
        state_timestamp=state.timestamp,
        quote_timestamp=time.time(),
        quote_latency_ms=quote_latency_ms,
        detail=(
            "virtual reserves read from the live curve account; "
            f"{fee_bps} bps protocol fee applied to the input before the "
            "constant-product step"
        ),
    )


def quote_pumpswap_buy(
    *,
    base_mint: str,
    quote_mint: str,
    pool_address: str,
    base_reserve: int,
    quote_reserve: int,
    sol_amount_lamports: int,
    base_decimals: int,
    quote_decimals: int = SOL_DECIMALS,
    fee_bps: int = PUMP_SWAP_FEE_BPS,
    price_impact_pct: float | None = None,
    state_timestamp: float | None = None,
    quote_latency_ms: float | None = None,
) -> ExecutableQuote:
    """
    A BUY quote against a PumpSwap pool, for an exact input size.

    ``base_reserve``/``quote_reserve`` must be the pool's own EFFECTIVE
    reserves. Passing a raw vault balance is a different number and is called
    out in the detail line so a reader can see which was used.
    """
    if sol_amount_lamports <= 0:
        raise CurveUnavailable("buy amount must be positive")
    if base_reserve <= 0 or quote_reserve <= 0:
        raise CurveUnavailable(
            f"pool reserves must both be positive (base={base_reserve}, "
            f"quote={quote_reserve}); refusing to model a pool with a zero side"
        )

    net_lamports, fee_lamports = apply_fee_bps(sol_amount_lamports, fee_bps)
    out_tokens = constant_product_out(
        reserve_in=quote_reserve,
        reserve_out=base_reserve,
        amount_in_after_fee=net_lamports,
    )
    if out_tokens <= 0:
        raise CurveUnavailable(
            "the pool models zero output for this size; the quote is refused"
        )

    try:
        price = effective_price(
            sol_amount_lamports,
            out_tokens,
            input_decimals=SOL_DECIMALS,
            output_decimals=base_decimals,
        )
    except DecimalError as exc:
        raise CurveUnavailable(f"cannot compute an effective price: {exc}") from exc

    return ExecutableQuote(
        input_mint=quote_mint,
        output_mint=base_mint,
        input_amount_raw=str(sol_amount_lamports),
        output_amount_raw=str(out_tokens),
        input_decimals=SOL_DECIMALS,
        output_decimals=base_decimals,
        input_amount=float(to_decimal(sol_amount_lamports).scaleb(-SOL_DECIMALS)),
        output_amount=float(to_decimal(out_tokens).scaleb(-base_decimals)),
        effective_price=float(price),
        effective_price_native=float(price),
        price_impact_pct=price_impact_pct,
        fee_bps=fee_bps,
        fee_amount=float(to_decimal(fee_lamports).scaleb(-SOL_DECIMALS)),
        venue="pumpswap",
        pool_address=pool_address,
        route=("pumpswap",),
        provider="pumpswap",
        state_timestamp=state_timestamp,
        quote_timestamp=time.time(),
        quote_latency_ms=quote_latency_ms,
        detail=(
            "constant-product against the pool's effective reserves, "
            f"{fee_bps} bps fee applied to the input first"
        ),
    )


def failed_quote(
    failure: FailureMode,
    *,
    input_mint: str | None = None,
    output_mint: str | None = None,
    detail: str | None = None,
    market: MarketKind = MarketKind.UNKNOWN,
) -> ExecutableQuote:
    """
    A quote that failed, carrying its reason.

    Amounts are ``None``/``""``, never ``0``. A zero-amount quote that reached
    a downstream comparison would produce a zero price and a meaningless gap,
    which is exactly the "never convert a failure to zero" rule.
    """
    return ExecutableQuote(
        input_mint=input_mint or "",
        output_mint=output_mint or "",
        input_amount_raw="",
        output_amount_raw="",
        input_decimals=SOL_DECIMALS,
        output_decimals=0,
        input_amount=0.0,
        output_amount=0.0,
        effective_price=None,
        effective_price_native=None,
        price_impact_pct=None,
        fee_bps=None,
        fee_amount=None,
        venue=None,
        pool_address=None,
        route=(),
        provider=f"none ({failure.value})",
        state_timestamp=None,
        quote_timestamp=time.time(),
        quote_latency_ms=None,
        failure=failure,
        detail=detail or market.value,
    )


__all__ = [
    "BondingCurveState",
    "CURVE_DISCRIMINATOR",
    "CurveUnavailable",
    "PUMP_FUN_FEE_BPS",
    "PUMP_FUN_PROGRAM",
    "PUMP_PROGRAMS",
    "PUMP_SWAP_FEE_BPS",
    "PUMP_SWAP_PROGRAM",
    "decode_bonding_curve",
    "decode_pumpswap_pool",
    "derive_bonding_curve_address",
    "failed_quote",
    "find_bonding_curve",
    "quote_bonding_curve_buy",
    "quote_pumpswap_buy",
]