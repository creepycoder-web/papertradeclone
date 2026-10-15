"""
Pump.fun bonding curve and PumpSwap (test items 8, 9, 10 and 11).

Account bytes are SYNTHESISED here at the documented offsets rather than
captured from mainnet, so the suite is deterministic and offline. The decoders
are written to fail closed when a layout does not match, and several of these
tests exist precisely to prove they do: a decoder that reads a zero reserve and
returns it would produce a confidently wrong quote.
"""


from __future__ import annotations

import pytest

from autopaper.same_market.models import WSOL_MINT, FailureMode, MarketKind
from autopaper.same_market.pump import (
    CURVE_DISCRIMINATOR,
    PUMP_FUN_FEE_BPS,
    PUMP_FUN_PROGRAM,
    PUMP_SWAP_PROGRAM,
    CurveUnavailable,
    decode_bonding_curve,
    decode_pumpswap_pool,
    failed_quote,
    quote_bonding_curve_buy,
    quote_pumpswap_buy,
)
from autopaper.same_market.pump import _b58, _b58_decode

pytestmark = pytest.mark.same_market

WSOL = WSOL_MINT
MINT = "H9XQrbw5MANzYVgHjXJbScFxWqyGw9CCAec5XDTgpump"
CURVE = "9rPYCUjkgH3rMCbB9fB7bF4nUqNQ6EqFNzVvCGrXvpS"
POOL = "2wT8Yq49kHgDzXuPxZSaeLaH1qbmGXtEyPy64bL7aD3c"

_VIRTUAL_SOL = 30_000_000_000      # 30 SOL
_VIRTUAL_TOKEN = 800_000_000_000   # 800,000,000 tokens (6dp)
_SUPPLY = 1_000_000_000_000


def _curve_bytes(
    *,
    mint: str = MINT,
    complete: bool = False,
    virtual_sol: int = _VIRTUAL_SOL,
    virtual_token: int = _VIRTUAL_TOKEN,
    discriminator: bytes = CURVE_DISCRIMINATOR,
    length: int = 300,
) -> bytes:
    """A bonding-curve account laid out as the decoder expects."""
    buf = bytearray(length)
    buf[0:8] = discriminator
    buf[8:40] = _b58_decode(CURVE)
    buf[40:72] = _b58_decode(mint)          # <-- the identity field
    buf[72:104] = _b58_decode(WSOL)
    buf[168:176] = (123_456).to_bytes(8, "little")   # price
    buf[176:184] = (1_000_000).to_bytes(8, "little")  # supply
    buf[184] = 1 if complete else 0
    buf[185:193] = virtual_sol.to_bytes(8, "little")
    buf[193:201] = virtual_token.to_bytes(8, "little")
    buf[201:209] = _SUPPLY.to_bytes(8, "little")
    return bytes(buf)


def _decode(raw: bytes | None = None, **kw):
    return decode_bonding_curve(
        raw if raw is not None else _curve_bytes(),
        curve_address=CURVE,
        program_id=PUMP_FUN_PROGRAM,
        expected_mint=kw.pop("expected_mint", MINT),
        timestamp=1000.0,
        **kw,
    )


# ---------------------------------------------------------------------------
# 8. Pump bonding curve detection
# ---------------------------------------------------------------------------

def test_a_live_curve_decodes_with_verified_identity():
    state = _decode()
    assert state.mint == MINT
    assert state.identity_verified is True
    assert state.discriminator_known is True
    assert state.complete is False
    assert state.virtual_sol_reserves == _VIRTUAL_SOL
    assert state.virtual_token_reserves == _VIRTUAL_TOKEN
    assert state.token_total_supply == _SUPPLY


def test_a_curve_whose_own_mint_disagrees_is_refused():
    """
    The strongest identity check available. Accepting this would mean pricing
    one token's curve and reporting it against another's mark.
    """
    raw = _curve_bytes(mint="6uPEdkU2xPg1iyfLzV1ssUwvve8nndA2sd6nBFrb1n58")
    with pytest.raises(CurveUnavailable) as exc:
        _decode(raw)
    assert "different markets" in str(exc.value)


def test_an_account_too_short_to_be_a_curve_is_refused():
    with pytest.raises(CurveUnavailable) as exc:
        _decode(_curve_bytes()[:100])
    assert "too short" in str(exc.value)


def test_a_zero_reserve_is_refused_rather_than_modelled():
    with pytest.raises(CurveUnavailable) as exc:
        _decode(_curve_bytes(virtual_sol=0))
    assert "zero reserve" in str(exc.value)


def test_an_unrecognised_discriminator_is_reported_but_not_fatal_on_its_own():
    """
    The mint check is the authoritative one, so an unfamiliar discriminator is
    surfaced as a flag for the report rather than aborting - but it must be
    visible, because it is the signal that the layout may have moved.
    """
    state = _decode(_curve_bytes(discriminator=b"\x00" * 8))
    assert state.discriminator_known is False
    assert state.identity_verified is True


def test_a_completed_curve_is_flagged():
    assert _decode(_curve_bytes(complete=True)).complete is True


def test_b58_round_trips():
    assert _b58(_b58_decode(MINT)) == MINT
    assert _b58_decode("not!base58") == b""


# ---------------------------------------------------------------------------
# 9. Pump bonding curve quote
# ---------------------------------------------------------------------------

def test_a_curve_buy_quote_reports_both_amounts_and_the_price():
    state = _decode()
    lamports = 1_150_000
    q = quote_bonding_curve_buy(state, sol_amount_lamports=lamports)

    assert q.input_mint == WSOL
    assert q.output_mint == MINT
    assert int(q.input_amount_raw) == lamports
    assert int(q.output_amount_raw) > 0
    assert q.effective_price == pytest.approx(
        float(q.input_amount) / float(q.output_amount), rel=1e-9
    )
    assert q.fee_bps == PUMP_FUN_FEE_BPS
    assert q.fee_amount == pytest.approx(lamports * 0.01 / 1e9, rel=1e-6)
    assert q.pool_address == CURVE
    assert q.venue == "pumpfun-bonding-curve"
    assert q.failure is None


def test_the_protocol_fee_is_reported_and_makes_the_quote_worse_than_free():
    lamports = 1_150_000
    with_fee = quote_bonding_curve_buy(_decode(), sol_amount_lamports=lamports,
                                       fee_bps=PUMP_FUN_FEE_BPS)
    without = quote_bonding_curve_buy(_decode(), sol_amount_lamports=lamports,
                                      fee_bps=0)
    assert int(without.output_amount_raw) > int(with_fee.output_amount_raw)
    assert with_fee.effective_price > without.effective_price


def test_a_larger_buy_gets_a_worse_price_on_the_curve():
    small = quote_bonding_curve_buy(_decode(), sol_amount_lamports=200_000)
    large = quote_bonding_curve_buy(_decode(), sol_amount_lamports=5_000_000)
    assert large.effective_price > small.effective_price


def test_quoting_a_completed_curve_is_refused():
    """
    A completed curve is no longer the market. Modelling it would price a
    trade against something that has stopped.
    """
    state = _decode(_curve_bytes(complete=True))
    with pytest.raises(CurveUnavailable) as exc:
        quote_bonding_curve_buy(state, sol_amount_lamports=1_150_000)
    assert "no longer the market" in str(exc.value)


def test_a_curve_quote_needs_a_verified_identity():
    state = _decode()
    object.__setattr__(state, "identity_verified", False)
    with pytest.raises(CurveUnavailable):
        quote_bonding_curve_buy(state, sol_amount_lamports=1_150_000)


def test_a_zero_sized_buy_is_refused():
    with pytest.raises(CurveUnavailable):
        quote_bonding_curve_buy(_decode(), sol_amount_lamports=0)


# ---------------------------------------------------------------------------
# 10. PumpSwap detection
# ---------------------------------------------------------------------------

def _pool_bytes(base: str = MINT, quote: str = WSOL) -> bytes:
    buf = bytearray(200)
    buf[40:72] = _b58_decode(base)
    buf[72:104] = _b58_decode(quote)
    buf[104] = 6   # base decimals
    buf[105] = 9   # quote decimals
    return bytes(buf)


def test_a_pool_containing_both_mints_verifies_identity():
    state = decode_pumpswap_pool(
        _pool_bytes(), pool_address=POOL, program_id=PUMP_SWAP_PROGRAM,
        expected_base_mint=MINT, expected_quote_mint=WSOL, timestamp=1000.0,
    )
    assert state.identity_verified is True
    assert state.kind is MarketKind.PUMPSWAP
    assert state.address == POOL
    assert "both mints found" in state.detail["identity_check"]


def test_a_pool_that_does_not_contain_the_token_is_refused():
    raw = _pool_bytes(base="6uPEdkU2xPg1iyfLzV1ssUwvve8nndA2sd6nBFrb1n58")
    with pytest.raises(CurveUnavailable) as exc:
        decode_pumpswap_pool(raw, pool_address=POOL,
                             program_id=PUMP_SWAP_PROGRAM,
                             expected_base_mint=MINT, expected_quote_mint=WSOL)
    assert "not this token's pool" in str(exc.value)


def test_a_pool_quoted_in_another_asset_is_refused():
    """
    Substitution hazard: the pool exists and holds the token, but it is quoted
    in a different asset than the order would spend.
    """
    raw = _pool_bytes(quote="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
    with pytest.raises(CurveUnavailable) as exc:
        decode_pumpswap_pool(raw, pool_address=POOL,
                             program_id=PUMP_SWAP_PROGRAM,
                             expected_base_mint=MINT, expected_quote_mint=WSOL)
    assert "different asset" in str(exc.value)


def test_identity_is_checked_by_content_not_by_a_brittle_offset():
    """
    The mints are located anywhere in the account. A decoder that hardcoded an
    offset would break on a program version bump and silently read zeros.
    """
    raw = bytearray(_pool_bytes())
    shifted = bytearray(200)
    shifted[10:42] = raw[40:72]    # base mint moved
    shifted[42:74] = raw[72:104]   # quote mint moved
    state = decode_pumpswap_pool(
        bytes(shifted), pool_address=POOL, program_id=PUMP_SWAP_PROGRAM,
        expected_base_mint=MINT, expected_quote_mint=WSOL,
    )
    assert state.identity_verified is True


def test_an_invalid_mint_address_is_refused_before_any_decoding():
    with pytest.raises(CurveUnavailable):
        decode_pumpswap_pool(_pool_bytes(), pool_address=POOL,
                             program_id=PUMP_SWAP_PROGRAM,
                             expected_base_mint="not base58!!",
                             expected_quote_mint=WSOL)


# ---------------------------------------------------------------------------
# 11. PumpSwap quote
# ---------------------------------------------------------------------------

def test_a_pumpswap_buy_quote_reports_its_pool_and_price():
    q = quote_pumpswap_buy(
        base_mint=MINT, quote_mint=WSOL, pool_address=POOL,
        base_reserve=800_000_000_000, quote_reserve=30_000_000_000,
        sol_amount_lamports=1_150_000, base_decimals=6,
        state_timestamp=1000.0,
    )
    assert q.input_mint == WSOL
    assert q.output_mint == MINT
    assert q.pool_address == POOL
    assert q.venue == "pumpswap"
    assert int(q.output_amount_raw) > 0
    assert q.effective_price == pytest.approx(
        float(q.input_amount) / float(q.output_amount), rel=1e-9
    )
    assert "effective reserves" in q.detail


def test_a_zero_reserve_is_refused_not_filled_from_a_vault_balance():
    """
    The specific mistake this harness must not make: substituting a raw vault
    balance for the protocol's effective quote reserves.
    """
    with pytest.raises(CurveUnavailable) as exc:
        quote_pumpswap_buy(
            base_mint=MINT, quote_mint=WSOL, pool_address=POOL,
            base_reserve=800_000_000_000, quote_reserve=0,
            sol_amount_lamports=1_150_000, base_decimals=6,
        )
    assert "zero side" in str(exc.value)


def test_the_pumpswap_quote_is_directionally_a_buy():
    """Input is the quote asset, output is the token. Never reversed."""
    q = quote_pumpswap_buy(
        base_mint=MINT, quote_mint=WSOL, pool_address=POOL,
        base_reserve=800_000_000_000, quote_reserve=30_000_000_000,
        sol_amount_lamports=1_150_000, base_decimals=6,
    )
    assert q.input_mint == WSOL != q.output_mint == MINT
    assert q.input_decimals == 9
    assert q.output_decimals == 6


def test_a_larger_pumpswap_buy_gets_a_worse_price():
    def eff(size: int) -> float:
        q = quote_pumpswap_buy(
            base_mint=MINT, quote_mint=WSOL, pool_address=POOL,
            base_reserve=800_000_000_000, quote_reserve=30_000_000_000,
            sol_amount_lamports=size, base_decimals=6,
        )
        return q.effective_price

    assert eff(5_000_000) > eff(200_000)


# ---------------------------------------------------------------------------
# failed quotes never carry a zero price
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("failure", list(FailureMode))
def test_a_failed_quote_has_no_price_and_no_amounts(failure):
    q = failed_quote(failure)
    assert q.failure is failure
    assert q.effective_price is None
    assert q.effective_price_native is None
    assert q.input_amount_raw == ""
    assert q.output_amount_raw == ""
    assert q.pool_address is None
    assert failure.value in q.provider