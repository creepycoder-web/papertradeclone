"""
Direct Pump.fun / PumpSwap executable quote verification - offline tests.

Covers the twenty-seven areas the brief lists. Every network-dependent test is
stubbed: the RPC is :class:`tests.pumpdirect_fakes.FakeRpc`, which exposes only
the three read methods the reader uses and has no ``send_transaction`` at all.

THE FIXTURES ARE REAL SHAPES
============================

Account bytes come from :mod:`tests.pumpdirect_accounts`, assembled field by
field from the official IDL, and seeded with the reserves read off live mainnet
on 2026-10-02. Several tests assert the SDK's own answers for that exact state,
so a regression in the arithmetic fails here rather than silently producing a
plausible wrong price.
"""


from __future__ import annotations

import hashlib
import inspect
import json
import sys
from decimal import Decimal

import pytest

from autopaper.same_market import pumpdirect as pd
from autopaper.same_market import pda
from autopaper.same_market.comparison import (
    DivergenceClass,
    LatencyMarks,
    PriceLabel,
    PricePoint,
    build_identity_report,
    cross_check,
)
from autopaper.same_market.models import (
    ACTUAL_FILL_AVAILABLE,
    SOL_DECIMALS,
    WSOL_MINT,
    FailureMode,
    MarketKind,
    ObservedMark,
    StalenessPolicy,
)
from autopaper.same_market.reader import DirectPumpReader
from autopaper.same_market.numeric import effective_price, normalise_amount

pytestmark = [
    pytest.mark.execution,
    pytest.mark.same_market,
]

# The fixture modules live next to this file. ``tests`` is a package, so pytest
# imports this module as ``tests.test_...``, and under some import modes a bare
# ``from pumpdirect_accounts import ...`` binds against a DIFFERENT module
# object than the one the names came from. Importing the modules themselves and
# re-exporting from them removes the ambiguity entirely, so the constants below
# are always the same objects the fixture builders use.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
try:  # pragma: no cover - depends on how pytest was invoked
    from tests import pumpdirect_accounts as _fx
    from tests import pumpdirect_fakes as _fk
except ImportError:  # pragma: no cover
    import pumpdirect_accounts as _fx
    import pumpdirect_fakes as _fk

LIVE_AMM_MINT = _fx.LIVE_AMM_MINT
LIVE_BUY_LAMPORTS = _fx.LIVE_BUY_LAMPORTS
LIVE_CURVE_ADDRESS = _fx.LIVE_CURVE_ADDRESS
LIVE_CURVE_MINT = _fx.LIVE_CURVE_MINT
LIVE_POOL_ADDRESS = _fx.LIVE_POOL_ADDRESS
LIVE_POOL_BASE_RESERVE = _fx.LIVE_POOL_BASE_RESERVE
LIVE_POOL_QUOTE_RESERVE = _fx.LIVE_POOL_QUOTE_RESERVE
LIVE_POOL_SUPPLY = _fx.LIVE_POOL_SUPPLY
LIVE_SDK_CURVE_OUTPUT = _fx.LIVE_SDK_CURVE_OUTPUT
LIVE_SDK_POOL_OUTPUT = _fx.LIVE_SDK_POOL_OUTPUT
TOKEN_PROGRAM = _fx.TOKEN_PROGRAM
amm_global_config_bytes = _fx.amm_global_config_bytes
bonding_curve_bytes = _fx.bonding_curve_bytes
curve_fee_config_bytes = _fx.curve_fee_config_bytes
fee_config_bytes = _fx.fee_config_bytes
global_bytes = _fx.global_bytes
mint_bytes = _fx.mint_bytes
pool_bytes = _fx.pool_bytes
token_account_bytes = _fx.token_account_bytes

FakeAccount = _fk.FakeAccount
FakeDexPair = _fk.FakeDexPair
FakeJupiter = _fk.FakeJupiter
FakeRpc = _fk.FakeRpc

SOL_USD = 122.36
CREATOR = "5L5k7gtNLbeXdzpvNrFshg1E1id1ceUDfc6vPUTxp98q"
BASE_ATA = "5jMpkf4JF4noHftLgNKyPNh6roVfPSGSjuEk3U4eLKRa"
QUOTE_ATA = "43DVcZR4kQFjh4Xm2i3DcneRxNjZp7HMud8yDrJWrDr8"


# ===========================================================================
# 1. bonding curve PDA derivation
# ===========================================================================

LIVE_OTHER_CURVE = "3MUkKMbuornHohtAtzrToSzqkj1gEEhQqYVz8sZnmQg1"


class TestBondingCurvePda:
    def test_derivation_matches_the_official_sdk(self):
        """
        Every address below was produced by the official ``@pump-fun/pump-sdk``
        ``bondingCurvePda`` and then confirmed against a live mainnet account
        owned by the pump.fun program.

        Bound to locals first: a chained ``assert f(CONST)[0] == CONST`` on this
        suite's CPython raises NameError for a module global that is
        demonstrably present in the module dict.
        """
        live_curve = pd.derive_bonding_curve(LIVE_CURVE_MINT)[0]
        assert live_curve == _fx.LIVE_CURVE_ADDRESS

        graduated = pd.derive_bonding_curve(LIVE_AMM_MINT)[0]
        assert graduated == _fx.LIVE_OTHER_CURVE_ADDRESS

        goat = pd.derive_bonding_curve(
            "9pQ8MNTTqK2N1f1pMg8CmxfLW1A9fS17MLtD5D8G6fsL")[0]
        assert goat == "H2ELvLjQxghsLsETExoh7QFCAbyikk7QvTjeN4iokbzQ"

    def test_the_seed_is_bonding_curve_with_a_hyphen(self):
        """
        A typo in the seed produces a valid-looking address that does not exist
        on chain, so the seed string is asserted, not just the output.
        """
        for wrong in ("bonding_curve", "bondingCurve", "curve", "bonding-curve "):
            address, _ = pda.find_program_address_b58(
                [wrong.encode(), pda.b58decode_pubkey(LIVE_CURVE_MINT)],
                pd.PUMP_FUN_PROGRAM,
            )
            assert address != LIVE_CURVE_ADDRESS, wrong

    def test_the_pool_authority_is_under_the_pump_program(self):
        """Not the AMM program: the two PDAs live under different programs."""
        address, _ = pda.find_program_address_b58(
            [b"pool-authority", pda.b58decode_pubkey(LIVE_AMM_MINT)],
            pd.PUMP_FUN_PROGRAM,
        )
        assert address == "9XDYTfQKwW8sHPqnFdUreMmtmffmkHVPGTNV2e3LKxNW"

    def test_a_pda_is_never_on_the_ed25519_curve(self):
        address, _ = pd.derive_bonding_curve(LIVE_CURVE_MINT)
        assert not pda.is_on_curve(pda.b58decode_pubkey(address))

    def test_a_normal_address_is_on_the_curve(self):
        """The complement, so the predicate is not vacuously true."""
        assert pda.is_on_curve(pda.b58decode_pubkey(pd.PUMP_FUN_PROGRAM))

    def test_static_pdas_match_the_sdk(self):
        assert pd.GLOBAL_PDA == "4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf"
        assert pd.PUMP_FEE_CONFIG_PDA == "8Wf5TiAheLUqBrKXeYg2JtAFFMWtKdG2BSFgqUcPVwTt"
        assert pd.AMM_GLOBAL_CONFIG_PDA == (
            "ADyA8hdefvWN2dbGGWFotbzWxrAvLW83WG6QCVXvJKqw"
        )
        assert pd.AMM_FEE_CONFIG_PDA == (
            "5PHirr8joyTMp9JMm6nW7hNDVyEYdkzDqazxPD7RaTjx"
        )


# ===========================================================================
# 2. program-owner validation  /  4. invalid account rejection
# ===========================================================================

class TestOwnerAndIdentityValidation:
    def test_a_curve_owned_by_the_wrong_program_is_refused(self):
        reader = DirectPumpReader(FakeRpc({
            LIVE_CURVE_ADDRESS: FakeAccount(LIVE_CURVE_ADDRESS, "11111111111111111111111111111111",
                                            bonding_curve_bytes()),
        }))
        with pytest.raises(pd.StateUnavailable) as exc:
            _run(reader.read_bonding_curve(LIVE_CURVE_MINT))
        assert "not 6EF8" in str(exc.value)

    def test_a_pool_owned_by_the_pump_program_is_refused(self):
        """The bonding curve is not a pool, even though it is 'pump'."""
        raw = pool_bytes(creator=pd.derive_pool_authority(LIVE_AMM_MINT),
                         base_mint=LIVE_AMM_MINT, pool_base_token_account=BASE_ATA,
                         pool_quote_token_account=QUOTE_ATA)
        reader = DirectPumpReader(FakeRpc({
            LIVE_POOL_ADDRESS: FakeAccount(LIVE_POOL_ADDRESS, pd.PUMP_FUN_PROGRAM, raw),
        }))
        with pytest.raises(pd.StateUnavailable):
            _run(reader.read_pumpswap_pool(LIVE_AMM_MINT, pool_address=LIVE_POOL_ADDRESS))

    def test_a_pool_whose_base_mint_is_another_token_is_refused(self):
        auth = pd.derive_pool_authority(LIVE_AMM_MINT)
        raw = pool_bytes(creator=auth, base_mint="APxxh2tWCh95tbHitFJchcdbMsctSBgRS2Zzju9Bpump",
                         pool_base_token_account=BASE_ATA, pool_quote_token_account=QUOTE_ATA)
        other = LIVE_CURVE_MINT
        rpc = FakeRpc({
            LIVE_POOL_ADDRESS: FakeAccount(LIVE_POOL_ADDRESS, pd.PUMP_SWAP_PROGRAM, raw),
            other: FakeAccount(other, TOKEN_PROGRAM, mint_bytes(supply=LIVE_POOL_SUPPLY)),
            BASE_ATA: FakeAccount(BASE_ATA, TOKEN_PROGRAM,
                                  token_account_bytes(amount=1, mint=other)),
            QUOTE_ATA: FakeAccount(QUOTE_ATA, TOKEN_PROGRAM,
                                   token_account_bytes(amount=1, mint=WSOL_MINT)),
        })
        with pytest.raises(pd.StateUnavailable) as exc:
            _run(DirectPumpReader(rpc).read_pumpswap_pool(
                LIVE_AMM_MINT, pool_address=LIVE_POOL_ADDRESS))
        assert "different token" in str(exc.value)

    def test_a_vault_owned_by_a_non_token_program_is_refused(self):
        """A 'balance' owned by something that is not a token account."""
        auth = pd.derive_pool_authority(LIVE_AMM_MINT)
        raw = pool_bytes(creator=auth, base_mint=LIVE_AMM_MINT,
                         pool_base_token_account=BASE_ATA,
                         pool_quote_token_account=QUOTE_ATA)
        rpc = FakeRpc({
            LIVE_POOL_ADDRESS: FakeAccount(LIVE_POOL_ADDRESS, pd.PUMP_SWAP_PROGRAM, raw),
            LIVE_AMM_MINT: FakeAccount(LIVE_AMM_MINT, TOKEN_PROGRAM, mint_bytes()),
            BASE_ATA: FakeAccount(BASE_ATA, "11111111111111111111111111111111",
                                  token_account_bytes(amount=1, mint=LIVE_AMM_MINT)),
            QUOTE_ATA: FakeAccount(QUOTE_ATA, TOKEN_PROGRAM,
                                   token_account_bytes(amount=1, mint=WSOL_MINT)),
        })
        with pytest.raises(pd.StateUnavailable) as exc:
            _run(DirectPumpReader(rpc).read_pumpswap_pool(
                LIVE_AMM_MINT, pool_address=LIVE_POOL_ADDRESS))
        assert "not an SPL token program" in str(exc.value)


# ===========================================================================
# 3. bonding curve decoding  (+ discriminator offsets)
# ===========================================================================

class TestBondingCurveDecoding:
    def test_the_discriminator_is_sha256_of_the_account_name(self):
        """
        Reproducible from the Anchor convention, so a mistyped constant cannot
        survive. Two such typos were caught by this exact check while writing
        the module.
        """
        for name, constant in (
            ("BondingCurve", pd.CURVE_DISCRIMINATOR),
            ("Global", pd.GLOBAL_DISCRIMINATOR),
            ("FeeConfig", pd.FEE_CONFIG_DISCRIMINATOR),
            ("Pool", pd.POOL_DISCRIMINATOR),
            ("GlobalConfig", pd.AMM_GLOBAL_CONFIG_DISCRIMINATOR),
        ):
            assert constant == hashlib.sha256(f"account:{name}".encode()).digest()[:8]

    def test_every_reserve_decodes_from_its_documented_offset(self):
        raw = bonding_curve_bytes(
            virtual_token_reserves=111, virtual_quote_reserves=222,
            real_token_reserves=333, real_quote_reserves=444,
            token_total_supply=555, complete=True, creator_fee_bps=42,
        )
        curve = pd.decode_bonding_curve(raw, curve_address="a", mint="m")
        assert curve.virtual_token_reserves == 111
        assert curve.virtual_quote_reserves == 222
        assert curve.real_token_reserves == 333
        assert curve.real_quote_reserves == 444
        assert curve.token_total_supply == 555
        assert curve.complete is True
        assert curve.creator == CREATOR
        assert curve.creator_fee_bps == 42

    def test_the_offsets_are_what_the_idl_says(self):
        raw = bonding_curve_bytes(virtual_token_reserves=7, complete=True,
                                  creator=CREATOR)
        assert int.from_bytes(raw[8:16], "little") == 7
        assert raw[48] == 1
        # The mint is NOT stored: bytes 40..48 are token_total_supply, and a
        # decoder that read offset 40 as a mint would be reading a number.
        assert int.from_bytes(raw[40:48], "little") == 1_000_000_000_000_000

    def test_a_legacy_125_byte_curve_decodes(self):
        curve = pd.decode_bonding_curve(
            bonding_curve_bytes(size=125), curve_address="a", mint="m")
        assert curve.account_len == 125
        assert curve.creator_fee_bps == 0

    def test_a_short_account_is_refused_rather_than_read_as_zeros(self):
        with pytest.raises(pd.StateUnavailable) as exc:
            pd.decode_bonding_curve(pd.CURVE_DISCRIMINATOR + b"\x00" * 20,
                                    curve_address="a", mint="m")
        assert "at least" in str(exc.value)

    def test_a_wrong_discriminator_is_refused(self):
        raw = bytearray(bonding_curve_bytes())
        raw[0] ^= 0xFF
        with pytest.raises(pd.StateUnavailable) as exc:
            pd.decode_bonding_curve(bytes(raw), curve_address="a", mint="m")
        assert "different account type" in str(exc.value)

    def test_a_zero_quote_reserve_makes_the_curve_unquotable(self):
        """Zero reserves must refuse, not produce a zero-price quote."""
        curve = pd.decode_bonding_curve(
            bonding_curve_bytes(virtual_quote_reserves=0),
            curve_address="a", mint="m")
        with pytest.raises(pd.StateUnavailable):
            pd.quote_bonding_curve_buy(
                curve, sol_amount_lamports=1_000_000, token_decimals=6,
                fees=pd.Fees(0, 95, 30), mint_supply=10**15)


# ===========================================================================
# 5. current state selection  (graduated vs live)
# ===========================================================================

class TestStateSelection:
    def test_a_live_curve_is_selected(self):
        curve = pd.decode_bonding_curve(bonding_curve_bytes(), curve_address="a", mint="m")
        assert curve.graduated is False
        assert pd.classify(curve=curve, pool=None, dex_id="pumpfun").market is (
            MarketKind.PUMP_BONDING_CURVE
        )

    def test_a_complete_curve_is_graduated_even_if_it_still_has_reserves(self):
        curve = pd.decode_bonding_curve(
            bonding_curve_bytes(complete=True, virtual_token_reserves=5_000),
            curve_address="a", mint="m")
        assert curve.graduated is True

    def test_a_drained_curve_is_graduated_even_without_the_flag(self):
        """Migration drains every reserve; the flag can lag."""
        curve = pd.decode_bonding_curve(
            bonding_curve_bytes(virtual_token_reserves=0, complete=False),
            curve_address="a", mint="m")
        assert curve.graduated is True

    def test_a_graduated_curve_is_never_quoted(self):
        curve = pd.decode_bonding_curve(
            bonding_curve_bytes(complete=True, virtual_token_reserves=0),
            curve_address="a", mint="m")
        with pytest.raises(pd.StateUnavailable) as exc:
            pd.quote_bonding_curve_buy(curve, sol_amount_lamports=1_000_000,
                                       token_decimals=6, fees=pd.Fees(0, 95, 30),
                                       mint_supply=10**15)
        assert "no longer the market" in str(exc.value)

    def test_nothing_on_chain_classifies_as_unknown_not_another_market(self):
        result = pd.classify(curve=None, pool=None, dex_id="raydium")
        assert result.market is MarketKind.UNKNOWN
        assert "not the same as C" in " ".join(result.notes)

    def test_the_dexid_label_never_decides_the_classification(self):
        """A `pumpfun` label on a graduated coin must not say PUMP_BONDING_CURVE."""
        curve = pd.decode_bonding_curve(
            bonding_curve_bytes(complete=True, virtual_token_reserves=0),
            curve_address="a", mint="m")
        pool = pd.decode_pumpswap_pool(
            pool_bytes(creator=pd.derive_pool_authority(LIVE_AMM_MINT),
                       base_mint=LIVE_AMM_MINT, pool_base_token_account=BASE_ATA,
                       pool_quote_token_account=QUOTE_ATA),
            pool_address=LIVE_POOL_ADDRESS, base_mint_account=mint_bytes(),
            base_token_account=token_account_bytes(amount=1, mint=LIVE_AMM_MINT),
            quote_token_account=token_account_bytes(amount=1, mint=WSOL_MINT),
        )
        result = pd.classify(curve=curve, pool=pool, dex_id="pumpfun")
        assert result.market is MarketKind.PUMPSWAP
        assert result.dex_id == "pumpfun"


# ===========================================================================
# 6 + 7. Pump direct BUY quote and decimal normalisation
# ===========================================================================

class TestPumpDirectQuote:
    def _quote(self, **overrides):
        curve = pd.decode_bonding_curve(
            bonding_curve_bytes(**overrides), curve_address=LIVE_CURVE_ADDRESS,
            mint=LIVE_CURVE_MINT)
        fees = pd.curve_fee_schedule(
            curve, global_config=pd.decode_global(global_bytes()),
            fee_config=pd.decode_fee_config(curve_fee_config_bytes()),
            mint_supply=10**15)
        return curve, fees, pd.quote_bonding_curve_buy(
            curve, sol_amount_lamports=LIVE_BUY_LAMPORTS, token_decimals=6,
            fees=fees, mint_supply=10**15)

    def test_the_live_mainnet_snapshot_reproduces_the_official_sdk_exactly(self):
        """
        The reserves below are the ones read off this curve on 2026-10-02 and
        replayed through @pump-fun/pump-sdk's getBuyTokenAmountFromSolAmount,
        which returned 5,176,254,910. A regression here is a wrong price.
        """
        curve, fees, quote = self._quote(
            virtual_token_reserves=320_098_471_659_904,
            virtual_quote_reserves=100_562_804_716,
            real_token_reserves=40_198_471_659_904,
        )
        assert quote.output_amount_raw == LIVE_SDK_CURVE_OUTPUT
        assert fees.protocol_fee_bps == 95
        assert fees.creator_fee_bps == 30

    def test_the_live_fee_schedule_is_read_not_defaulted(self):
        """`Global` says 95 + 5; the live schedule says 95 + 30."""
        _, fees, _ = self._quote()
        assert fees.protocol_fee_bps == 95
        assert fees.creator_fee_bps == 30

    def test_the_fee_is_taken_from_the_input_not_added_to_the_output(self):
        """
        Two different official functions exist for the two directions and they
        differ by about the fee. Using the wrong one under-quotes.
        """
        curve, fees, quote = self._quote()
        total = fees.protocol_fee_bps + fees.creator_fee_bps
        expected_input = (LIVE_BUY_LAMPORTS - 1) * 10_000 // (10_000 + total)
        assert quote.amount_after_fee == expected_input
        assert quote.amount_after_fee < LIVE_BUY_LAMPORTS

    def test_the_output_is_capped_at_real_token_reserves(self):
        """The curve cannot sell tokens it does not hold."""
        curve, fees, _ = self._quote()
        capped, _, quote = self._quote(real_token_reserves=1_000)
        assert quote.output_amount_raw == 1_000
        assert capped.real_token_reserves == 1_000

    def test_a_creator_fee_is_charged_only_when_a_creator_exists(self):
        _, with_creator, _ = self._quote()
        _, without, _ = self._quote(creator=pd.DEFAULT_PUBKEY)
        # The schedule is the same; what changes is whether it is APPLIED.
        assert with_creator.creator_fee_bps == without.creator_fee_bps

    def test_a_zero_or_negative_input_is_refused(self):
        curve = pd.decode_bonding_curve(bonding_curve_bytes(),
                                        curve_address="a", mint="m")
        for amount in (0, -1):
            with pytest.raises(pd.StateUnavailable):
                pd.quote_bonding_curve_buy(curve, sol_amount_lamports=amount,
                                           token_decimals=6,
                                           fees=pd.Fees(0, 95, 30), mint_supply=10**15)

    def test_the_input_survives_as_an_exact_integer(self):
        _, _, quote = self._quote()
        assert quote.input_amount_raw == LIVE_BUY_LAMPORTS
        assert isinstance(quote.output_amount_raw, int)
        assert isinstance(quote.effective_price, Decimal)

    def test_price_impact_is_computed_against_the_curve_spot_price(self):
        _, _, quote = self._quote()
        assert quote.price_impact_pct is not None
        assert 0 < float(quote.price_impact_pct) < 10


# ===========================================================================
# 8 + 9 + 10. PumpSwap pool derivation, decoding and quote
# ===========================================================================

class TestPumpSwap:
    def _rpc(self, **overrides) -> FakeRpc:
        auth = overrides.pop("creator", pd.derive_pool_authority(LIVE_AMM_MINT))
        raw = pool_bytes(creator=auth, base_mint=LIVE_AMM_MINT,
                         pool_base_token_account=BASE_ATA,
                         pool_quote_token_account=QUOTE_ATA, **overrides)
        return FakeRpc({
            LIVE_POOL_ADDRESS: FakeAccount(LIVE_POOL_ADDRESS, pd.PUMP_SWAP_PROGRAM, raw),
            LIVE_AMM_MINT: FakeAccount(LIVE_AMM_MINT, TOKEN_PROGRAM,
                                       mint_bytes(decimals=6, supply=LIVE_POOL_SUPPLY)),
            BASE_ATA: FakeAccount(BASE_ATA, TOKEN_PROGRAM,
                                  token_account_bytes(amount=LIVE_POOL_BASE_RESERVE,
                                                      mint=LIVE_AMM_MINT)),
            QUOTE_ATA: FakeAccount(QUOTE_ATA, TOKEN_PROGRAM,
                                   token_account_bytes(amount=LIVE_POOL_QUOTE_RESERVE,
                                                       mint=WSOL_MINT)),
            pd.AMM_GLOBAL_CONFIG_PDA: FakeAccount(pd.AMM_GLOBAL_CONFIG_PDA,
                                                  pd.PUMP_SWAP_PROGRAM,
                                                  amm_global_config_bytes()),
            pd.AMM_FEE_CONFIG_PDA: FakeAccount(pd.AMM_FEE_CONFIG_PDA, pd.PUMP_FEE_PROGRAM,
                                               fee_config_bytes()),
        }, supply_value=LIVE_POOL_SUPPLY)

    def test_the_canonical_pool_pda_matches_the_official_sdk(self):
        assert pd.derive_canonical_pool(LIVE_AMM_MINT)[0] == LIVE_POOL_ADDRESS
        assert pd.derive_canonical_pool(LIVE_AMM_MINT)[1] == 254  # matches pool_bump on chain

    def test_the_pool_decodes_from_its_documented_offsets(self):
        pool = _run(DirectPumpReader(self._rpc()).read_pumpswap_pool(
            LIVE_AMM_MINT, pool_address=LIVE_POOL_ADDRESS))[0]
        assert pool.index == 0
        assert pool.base_mint == LIVE_AMM_MINT
        assert pool.quote_mint == WSOL_MINT
        assert pool.pool_base_token_account == BASE_ATA
        assert pool.pool_quote_token_account == QUOTE_ATA
        assert pool.base_decimals == 6

    def test_a_legacy_270_byte_pool_decodes(self):
        pool = _run(DirectPumpReader(self._rpc(size=270)).read_pumpswap_pool(
            LIVE_AMM_MINT, pool_address=LIVE_POOL_ADDRESS))[0]
        assert pool.account_len == 270
        assert pool.creator_fee_bps == 0

    def test_virtual_quote_reserves_is_read_as_a_signed_i128(self):
        """
        It has been negative since 2025-09-30. Read unsigned, a pool with a
        negative virtual reserve would look like it holds an absurd one.
        """
        pool = _run(DirectPumpReader(
            self._rpc(virtual_quote_reserves=-5_000)
        ).read_pumpswap_pool(LIVE_AMM_MINT, pool_address=LIVE_POOL_ADDRESS))[0]
        assert pool.virtual_quote_reserves == -5_000
        assert pool.effective_quote_reserve == LIVE_POOL_QUOTE_RESERVE - 5_000

    def test_the_effective_reserve_is_vault_plus_virtual(self):
        pool = _run(DirectPumpReader(
            self._rpc(virtual_quote_reserves=1_000_000)
        ).read_pumpswap_pool(LIVE_AMM_MINT, pool_address=LIVE_POOL_ADDRESS))[0]
        assert pool.effective_quote_reserve == LIVE_POOL_QUOTE_RESERVE + 1_000_000
        assert pool.quote_reserve == LIVE_POOL_QUOTE_RESERVE

    def test_a_canonical_pool_uses_the_tiered_fee_not_the_flat_one(self):
        """Flat is 25/5/0 = 30 bps; the live tier 0 is 2/93/30 = 125 bps."""
        pool = _run(DirectPumpReader(self._rpc()).read_pumpswap_pool(
            LIVE_AMM_MINT, pool_address=LIVE_POOL_ADDRESS))[0]
        assert pool.is_canonical_pump_pool is True
        fees = pd.amm_fee_schedule(pool, global_config=pd.decode_amm_global_config(
            amm_global_config_bytes()), fee_config=pd.decode_fee_config(
            fee_config_bytes()), circulating_supply=pool.base_supply)
        assert fees.total == 125
        assert fees.lp_fee_bps == 2

    def test_a_non_canonical_pool_pays_the_flat_fee(self):
        pool = _run(DirectPumpReader(
            self._rpc(creator="9pan9bMn5HatX4EJdBwg9VgCa7Uz5HL8N1m5D3NdXejP")
        ).read_pumpswap_pool(LIVE_AMM_MINT, pool_address=LIVE_POOL_ADDRESS))[0]
        assert pool.is_canonical_pump_pool is False
        fees = pd.amm_fee_schedule(pool, global_config=pd.decode_amm_global_config(
            amm_global_config_bytes()), fee_config=pd.decode_fee_config(
            fee_config_bytes()), circulating_supply=pool.base_supply)
        assert fees.total == 30

    def test_a_high_market_cap_pool_moves_to_the_next_tier(self):
        """Tier 1 at 420 SOL is 20/5/95 = 120 bps, not 125."""
        pool = _run(DirectPumpReader(self._rpc()).read_pumpswap_pool(
            LIVE_AMM_MINT, pool_address=LIVE_POOL_ADDRESS))[0]
        low = pd.amm_fee_schedule(pool, global_config=None,
                                  fee_config=pd.decode_fee_config(fee_config_bytes()),
                                  circulating_supply=pool.base_supply)
        # The live pool's market cap is ~196 SOL; x3 lands inside the
        # [420 SOL, 1470 SOL) band. x100 would overshoot into tier 2 and assert
        # the wrong fee.
        high = pd.amm_fee_schedule(pool, global_config=None,
                                   fee_config=pd.decode_fee_config(fee_config_bytes()),
                                   circulating_supply=pool.base_supply * 3)
        assert low.total == 125
        assert high.total == 120

    def test_the_live_mainnet_pool_reproduces_the_sdk_and_jupiter_exactly(self):
        """
        Both the official SDK and Jupiter returned 9,466,908,943 for this pool
        and size. Three independent sources, one number.
        """
        pool = _run(DirectPumpReader(self._rpc()).read_pumpswap_pool(
            LIVE_AMM_MINT, pool_address=LIVE_POOL_ADDRESS))[0]
        fees = pd.amm_fee_schedule(pool, global_config=None,
                                   fee_config=pd.decode_fee_config(fee_config_bytes()),
                                   circulating_supply=pool.base_supply)
        quote = pd.quote_pumpswap_buy(pool, sol_amount_lamports=LIVE_BUY_LAMPORTS,
                                      fees=fees, circulating_supply=pool.base_supply)
        assert quote.output_amount_raw == LIVE_SDK_POOL_OUTPUT
        assert quote.output_decimals == 6

    def test_fees_are_ceiled_not_floored(self):
        """The SDK's comments say floor; its code ceils. The code is truth."""
        assert pd.ceil_fee(1_000_001, 1) == 101   # 100.0001 -> 101
        assert pd.ceil_fee(10_000, 1) == 1
        assert pd.ceil_fee(0, 500) == 0

    def test_a_zero_base_reserve_is_refused_not_substituted(self):
        rpc = self._rpc()
        rpc.accounts[BASE_ATA] = FakeAccount(BASE_ATA, TOKEN_PROGRAM,
                                            token_account_bytes(amount=0, mint=LIVE_AMM_MINT))
        pool = _run(DirectPumpReader(rpc).read_pumpswap_pool(
            LIVE_AMM_MINT, pool_address=LIVE_POOL_ADDRESS))[0]
        with pytest.raises(pd.StateUnavailable) as exc:
            pd.quote_pumpswap_buy(pool, sol_amount_lamports=LIVE_BUY_LAMPORTS,
                                  fees=pd.Fees(2, 93, 30), circulating_supply=pool.base_supply)
        assert "no base liquidity" in str(exc.value)

    def test_a_missing_pool_is_unknown_not_a_zero_pool(self):
        quote = _run(DirectPumpReader(FakeRpc({})).quote_pumpswap(
            LIVE_AMM_MINT, sol_amount_lamports=LIVE_BUY_LAMPORTS))
        assert quote.available is False
        assert quote.failure is FailureMode.STATE_UNAVAILABLE
        assert quote.quote is None


# ===========================================================================
# 11. Jupiter BUY direction
# ===========================================================================

class TestJupiterDirection:
    def test_the_jupiter_price_is_derived_from_its_own_two_amounts(self):
        """SOL in, tokens out, both raw - the direction is asserted by maths."""
        from autopaper.same_market.jupiter import assert_buy_direction

        assert_buy_direction(input_mint=WSOL_MINT, output_mint=LIVE_CURVE_MINT,
                             quote_asset_mint=WSOL_MINT, token_mint=LIVE_CURVE_MINT)
        with pytest.raises(Exception):
            assert_buy_direction(input_mint=LIVE_CURVE_MINT, output_mint=WSOL_MINT,
                                 quote_asset_mint=WSOL_MINT, token_mint=LIVE_CURVE_MINT)

    def test_a_sell_direction_would_give_a_millionfold_wrong_price(self):
        buy = effective_price(LIVE_BUY_LAMPORTS, 4_979_468_321,
                              input_decimals=9, output_decimals=6)
        inverted = effective_price(4_979_468_321, LIVE_BUY_LAMPORTS,
                                   input_decimals=6, output_decimals=9)
        assert inverted > buy * 1_000_000


# ===========================================================================
# 12 + 13. identity mapping
# ===========================================================================

class TestIdentityMapping:
    def test_a_dexscreener_pumpfun_pair_address_is_the_curve_pda(self):
        report = build_identity_report(
            mint=LIVE_CURVE_MINT, observed_pair_address=LIVE_CURVE_ADDRESS,
            observed_dex_id="pumpfun", curve_address=LIVE_CURVE_ADDRESS,
            curve_chain_verified=True,
        )
        assert report.pool_match == "PASS"

    def test_a_dexscreener_pumpswap_pair_address_is_the_pool_address(self):
        report = build_identity_report(
            mint=LIVE_AMM_MINT, observed_pair_address=LIVE_POOL_ADDRESS,
            observed_dex_id="pumpswap", pool_address=LIVE_POOL_ADDRESS,
            pool_chain_verified=True,
        )
        assert report.pool_match == "PASS"

    def test_a_curve_is_never_matched_against_a_pool_address(self):
        """Two different account types; comparing them is meaningless."""
        report = build_identity_report(
            mint=LIVE_AMM_MINT, observed_pair_address=LIVE_POOL_ADDRESS,
            observed_dex_id="pumpswap",
            curve_address="3MUkKMbuornHohtAtzrToSzqkj1gEEhQqYVz8sZnmQg1",
            curve_chain_verified=True,
        )
        assert report.pool_match == "FAIL"

    def test_unprovable_identity_is_unknown_never_pass(self):
        report = build_identity_report(
            mint=LIVE_CURVE_MINT, observed_pair_address=LIVE_CURVE_ADDRESS,
            observed_dex_id="pumpfun",
        )
        assert report.pool_match == "UNKNOWN"
        assert report.market_match == "UNKNOWN"

    def test_no_observed_pair_is_unknown(self):
        report = build_identity_report(mint="m", observed_pair_address=None,
                                       observed_dex_id=None)
        assert report.pool_match == "UNKNOWN"

    def test_the_jupiter_ammkey_is_compared_and_its_type_comes_from_the_chain(self):
        report = build_identity_report(
            mint=LIVE_CURVE_MINT, observed_pair_address=LIVE_CURVE_ADDRESS,
            observed_dex_id="pumpfun", curve_address=LIVE_CURVE_ADDRESS,
            curve_chain_verified=True, jupiter_pool_keys=[LIVE_CURVE_ADDRESS],
            jupiter_labels=["Pump.fun"],
        )
        assert report.pool_match == "PASS"
        hop = next(f for f in report.findings if f.provider == "jupiter[0]")
        assert hop.kind == "ROUTE_HOP"
        assert hop.address == LIVE_CURVE_ADDRESS
        assert "ammKey" in hop.detail

    def test_a_jupiter_route_through_another_pool_is_flagged(self):
        report = build_identity_report(
            mint=LIVE_AMM_MINT, observed_pair_address=LIVE_POOL_ADDRESS,
            observed_dex_id="pumpswap", pool_address=LIVE_POOL_ADDRESS,
            pool_chain_verified=True,
            jupiter_pool_keys=["9pan9bMn5HatX4EJdBwg9VgCa7Uz5HL8N1m5D3NdXejP"],
            jupiter_labels=["Raydium"],
        )
        assert any("different venue" in n for n in report.notes)


# ===========================================================================
# 14. pairwise gaps
# ===========================================================================

def _point(label, price, provider="p"):
    return PricePoint(label=label, provider=provider, price_usd_per_token=price)


class TestPairwiseGaps:
    def test_the_sign_is_kept_because_direction_matters_for_a_buy(self):
        a = _point(PriceLabel.OBSERVED_MARK, 0.0000204)
        b = _point(PriceLabel.DIRECT_EXECUTABLE_QUOTE, 0.0000211)
        gap = cross_check(a, b, None, None).gaps[0]
        assert gap.gap_pct == pytest.approx(3.431, abs=1e-2)
        higher = cross_check(b, a, None, None).gaps[0]
        assert higher.gap_pct == pytest.approx(-3.31754, abs=1e-4)

    def test_an_absent_source_produces_no_gap_rather_than_a_hundred_percent(self):
        absent = PricePoint(label=PriceLabel.JUPITER_QUOTE, provider="jupiter",
                            price_usd_per_token=None,
                            missing_reason="no route")
        gap = cross_check(_point(PriceLabel.OBSERVED_MARK, 0.00002), None, None,
                          absent).gaps[0]
        assert gap.gap_pct is None
        assert gap.classification is DivergenceClass.UNKNOWN
        assert "not a zero" in gap.reason

    def test_all_five_named_diagnostics_are_present(self):
        result = cross_check(
            _point(PriceLabel.OBSERVED_MARK, 0.00002),
            _point(PriceLabel.DIRECT_EXECUTABLE_QUOTE, 0.000021),
            _point(PriceLabel.DIRECT_EXECUTABLE_QUOTE, 0.000021),
            _point(PriceLabel.JUPITER_QUOTE, 0.0000211),
        )
        names = {g.name for g in result.gaps}
        assert {
            "MARK_VS_PUMP_DIRECT", "MARK_VS_JUPITER", "PUMP_DIRECT_VS_JUPITER",
            "MARK_VS_PUMPSWAP", "PUMPSWAP_VS_JUPITER",
        } <= names

    def test_no_threshold_means_nothing_is_classified(self):
        result = cross_check(_point(PriceLabel.OBSERVED_MARK, 0.00002),
                             _point(PriceLabel.DIRECT_EXECUTABLE_QUOTE, 0.00005),
                             None, _point(PriceLabel.JUPITER_QUOTE, 0.00005))
        assert all(g.classification is DivergenceClass.UNKNOWN for g in result.gaps)
        # The raw gap is still reported; only the verdict is withheld.
        available = [g for g in result.gaps if g.gap_pct is not None]
        assert available, "the raw percentage must still be shown"
        assert all("no --divergence-pct" in g.reason for g in available)

    def test_a_configured_threshold_classifies_both_ways(self):
        close = cross_check(_point(PriceLabel.OBSERVED_MARK, 0.00002),
                            _point(PriceLabel.DIRECT_EXECUTABLE_QUOTE, 0.0000205),
                            None, None, divergence_pct=5.0).gaps[0]
        far = cross_check(_point(PriceLabel.OBSERVED_MARK, 0.00002),
                          _point(PriceLabel.DIRECT_EXECUTABLE_QUOTE, 0.00005),
                          None, None, divergence_pct=5.0).gaps[0]
        assert close.classification is DivergenceClass.CLOSE
        assert far.classification is DivergenceClass.DIVERGENT

    def test_the_worked_example_from_the_brief_reproduces(self):
        mark = _point(PriceLabel.OBSERVED_MARK, 0.00001339)
        pump = _point(PriceLabel.DIRECT_EXECUTABLE_QUOTE, 0.00001372)
        jup = _point(PriceLabel.JUPITER_QUOTE, 0.00002549)
        result = cross_check(mark, pump, None, jup)
        by_name = {g.name: g for g in result.gaps}
        assert by_name["MARK_VS_PUMP_DIRECT"].gap_pct == pytest.approx(2.46453, abs=1e-4)
        assert by_name["MARK_VS_JUPITER"].gap_pct == pytest.approx(90.36594, abs=1e-4)
        assert by_name["PUMP_DIRECT_VS_JUPITER"].gap_pct == pytest.approx(
            85.78717, abs=1e-4)

    def test_agreeing_executables_with_a_bad_mark_raises_mark_divergence(self):
        """
        The executables agree with each other and the mark does not. The shape
        that points at the mark describing a different or stale pool.
        """
        result = cross_check(
            _point(PriceLabel.OBSERVED_MARK, 0.00001339),
            _point(PriceLabel.DIRECT_EXECUTABLE_QUOTE, 0.00002540), None,
            _point(PriceLabel.JUPITER_QUOTE, 0.00002549), divergence_pct=5.0)
        assert any(f.startswith("MARK_DIVERGENCE") for f in result.findings)
        assert result.actionable is True

    def test_all_three_agreeing_raises_nothing(self):
        result = cross_check(
            _point(PriceLabel.OBSERVED_MARK, 0.00001339),
            _point(PriceLabel.DIRECT_EXECUTABLE_QUOTE, 0.00001372), None,
            _point(PriceLabel.JUPITER_QUOTE, 0.00001375), divergence_pct=5.0)
        assert not any(f.startswith("MARK_DIVERGENCE") for f in result.findings)
        assert result.actionable is False

    def test_disagreeing_executables_raise_execution_quote_divergence(self):
        result = cross_check(
            _point(PriceLabel.OBSERVED_MARK, 0.00001372),
            _point(PriceLabel.DIRECT_EXECUTABLE_QUOTE, 0.00001372), None,
            _point(PriceLabel.JUPITER_QUOTE, 0.00002549), divergence_pct=5.0)
        assert any(f.startswith("EXECUTION_QUOTE_DIVERGENCE") for f in result.findings)
        assert result.actionable is True

    def test_the_harness_never_declares_a_winner(self):
        result = cross_check(
            _point(PriceLabel.OBSERVED_MARK, 0.00001372),
            _point(PriceLabel.DIRECT_EXECUTABLE_QUOTE, 0.00001372), None,
            _point(PriceLabel.JUPITER_QUOTE, 0.00002549), divergence_pct=5.0)
        assert "no source was chosen as correct" in result.note


# ===========================================================================
# 15 + 16 + 17. freshness
# ===========================================================================

class TestFreshness:
    def _policy(self, **kw):
        return StalenessPolicy(**kw)

    def test_an_unconfigured_window_is_unknown_not_fresh(self):
        stamp = self._policy().stamp("state", 100.0, 100.0, None)
        assert stamp.freshness.value == "UNKNOWN"

    def test_missing_is_distinct_from_stale(self):
        missing = self._policy(max_state_age_ms=1_000).stamp("state", None, 100.0, 1_000)
        stale = self._policy(max_state_age_ms=1_000).stamp("state", 1.0, 100.0, 1_000)
        assert missing.freshness.value == "MISSING"
        assert stale.freshness.value == "STALE"

    def test_fresh_state_inside_the_window(self):
        stamp = self._policy(max_state_age_ms=5_000).stamp("state", 99.0, 100.0, 5_000)
        assert stamp.freshness.value == "FRESH"
        assert stamp.age_ms == pytest.approx(1_000.0)

    def test_state_beyond_the_window_is_stale(self):
        stamp = self._policy(max_state_age_ms=100).stamp("state", 1.0, 100.0, 100)
        assert stamp.freshness.value == "STALE"

    def test_mark_and_state_freshness_are_independent(self):
        policy = self._policy(max_mark_age_ms=100, max_state_age_ms=10_000)
        mark = policy.stamp("mark", 1.0, 100.0, 100)
        state = policy.stamp("state", 99.0, 100.0, 10_000)
        assert mark.freshness.value == "STALE"
        assert state.freshness.value == "FRESH"


# ===========================================================================
# 18 + 19 + 20 + 21 + 22 + 23. units
# ===========================================================================

class TestUnits:
    def test_a_six_decimal_token(self):
        assert normalise_amount(4_979_468_321, 6) == Decimal("4979.468321")

    def test_a_nine_decimal_token(self):
        assert normalise_amount(4_979_468_321, 9) == Decimal("4.979468321")

    def test_wsol_nine_decimals(self):
        assert normalise_amount(LIVE_BUY_LAMPORTS, 9) == Decimal("0.001879699")

    def test_ten_pow_six_and_nine_are_not_interchangeable(self):
        raw = 1_000_000
        assert normalise_amount(raw, 6) != normalise_amount(raw, 9)
        assert normalise_amount(raw, 6) / normalise_amount(raw, 9) == Decimal(1000)

    def test_sol_to_usd_conversion(self):
        """
        $0.23 at $122.36 is 1,879,699 lamports - the notional every live test
        used. Floats are coerced via repr so the shortest round-trip form is
        used; the exact path is exercised with strings.
        """
        from autopaper.same_market.numeric import sol_amount_for_usd

        assert int(sol_amount_for_usd(0.23, 122.36).scaleb(9)) == LIVE_BUY_LAMPORTS
        exact = sol_amount_for_usd("0.23", "122.36")
        assert exact == pytest.approx(Decimal("0.23") / Decimal("122.36"), rel=Decimal("1e-40"))
        assert int(exact.scaleb(9)) == LIVE_BUY_LAMPORTS

    def test_a_non_positive_sol_reference_is_refused(self):
        from autopaper.same_market.numeric import DecimalError, sol_amount_for_usd

        for bad in (0, -1):
            with pytest.raises(DecimalError):
                sol_amount_for_usd(0.23, bad)

    def test_a_tiny_price_is_not_rounded_to_zero(self):
        # 1 lamport (1e-9 SOL) for 1 raw token unit (1e-6 tokens) = 1e-3 SOL.
        assert effective_price(1, 1, input_decimals=9, output_decimals=6) == Decimal("0.001")
        # A trillion raw units for one lamport = 1e-15 SOL per token: absurdly
        # small, and still exact. The point is that it is not rounded to 0.0.
        tiny = effective_price(1, 10**12, input_decimals=9, output_decimals=6)
        assert tiny == Decimal("1E-15")
        assert tiny > 0
        # The real price point of this harness, preserved exactly.
        # The live PumpSwap BUY: 1,879,699 lamports for 9,466,908,943 raw units.
        # Computed at the module's 60-digit working precision, and asserted
        # exactly - this is the number that must match the official SDK.
        live = effective_price(1_879_699, 9_466_908_943, input_decimals=9,
                              output_decimals=6)
        assert live == Decimal(
            "1.98554671996700961613970812648176540087800863513597650539903E-7")

    def test_a_large_integer_amount_survives(self):
        big = 2**63 - 1
        assert normalise_amount(big, 6) == Decimal(big).scaleb(-6)

    def test_a_zero_output_amount_raises_rather_than_returning_zero(self):
        from autopaper.same_market.numeric import DecimalError

        with pytest.raises(DecimalError):
            effective_price(1_000, 0, input_decimals=9, output_decimals=6)

    def test_a_negative_input_raises(self):
        from autopaper.same_market.numeric import DecimalError

        with pytest.raises(DecimalError):
            effective_price(-1, 1_000, input_decimals=9, output_decimals=6)

    def test_implausible_decimals_are_refused(self):
        from autopaper.same_market.numeric import DecimalError

        for bad in (-1, 31):
            with pytest.raises(DecimalError):
                normalise_amount(1, bad)

    def test_raw_and_human_amounts_are_both_reported_separately(self):
        quote = _run(DirectPumpReader(_live_curve_rpc()).quote_bonding_curve(
            LIVE_CURVE_MINT, sol_amount_lamports=LIVE_BUY_LAMPORTS, token_decimals=6))
        body = quote.as_dict(sol_reference_usd=SOL_USD)
        detail = body["quote_detail"]
        # The USD reference is logged next to the price it produced, so the
        # conversion is reproducible rather than implicit.
        assert body["usd_reference_price"] == SOL_USD
        assert detail["input_amount_raw"] == LIVE_BUY_LAMPORTS
        assert detail["input_decimals"] == 9
        assert detail["output_amount_raw"] == quote.quote.output_amount_raw
        assert detail["output_decimals"] == 6
        assert detail["output_amount_decimal"] == pytest.approx(
            quote.quote.output_amount_raw / 10**6
        )
        # The quote's own price is SOL per token; the USD figure is derived
        # against the logged reference and the reference travels with it.
        assert detail["effective_price_sol_per_token"] == pytest.approx(
            float(quote.quote.effective_price)
        )
        assert "effective_price_usd_per_token" not in detail
        assert body["effective_price_usd_per_token"] == pytest.approx(
            float(quote.quote.effective_price) * SOL_USD
        )

    def test_a_missing_optional_field_is_absent_not_zero(self):
        """
        A pre-``creator_fee_bps`` account is 115 bytes. The field must read as
        ABSENT, not as 0 - a zero would silently mean "no creator fee",
        which is a different statement from "this account predates the field".
        """
        curve = pd.decode_bonding_curve(bonding_curve_bytes()[:115],
                                        curve_address="a", mint="m")
        assert curve.creator_fee_bps is None
        assert curve.can_edit_creator_fee is None
        # The fields that ARE present still decode.
        assert curve.virtual_token_reserves == 320_098_471_659_904
        assert curve.complete is False

    def test_a_current_account_does_carry_the_fee_field(self):
        curve = pd.decode_bonding_curve(bonding_curve_bytes(size=151, creator_fee_bps=0),
                                        curve_address="a", mint="m")
        assert curve.creator_fee_bps == 0
        assert curve.can_edit_creator_fee is False


# ===========================================================================
# 24. GOAT regression
# ===========================================================================

class TestGoatRegression:
    def test_the_mark_return_of_71_56_percent_is_not_a_realized_result(self):
        """
        The 2026-10-02 $GOAT incident: the mark rose 71.56% and the trade lost
        42.91%. Both numbers are real; conflating them is the bug.
        """
        from autopaper.same_market.numeric import pct_return

        mark_return = pct_return(1.783e-05, 3.059e-05)
        assert float(mark_return) == pytest.approx(71.56477846326415)

        all_in = pct_return(0.182493925, 0.0)
        assert float(all_in) == pytest.approx(-100.0)
        assert float(mark_return) > 0 > float(all_in)

    def test_an_executable_quote_is_a_separate_measurement(self):
        mark = _point(PriceLabel.OBSERVED_MARK, 3.059e-05)
        quote = _point(PriceLabel.DIRECT_EXECUTABLE_QUOTE, 3.2176361406260514e-05)
        gap = cross_check(mark, quote, None, None).gaps[0]
        assert gap.gap_pct == pytest.approx(5.186, abs=0.01)

    def test_the_harness_has_no_realized_return_field_at_all(self):
        """Structural: nothing in the comparison vocabulary can be labelled one."""
        labels = {label.value for label in PriceLabel}
        assert "ACTUAL_FILL" in labels
        assert ACTUAL_FILL_AVAILABLE is False

    def test_the_goat_mark_is_never_mistaken_for_a_fill(self):
        point = _point(PriceLabel.OBSERVED_MARK, 3.059e-05)
        assert point.label is PriceLabel.OBSERVED_MARK
        assert point.label is not PriceLabel.JUPITER_QUOTE


# ===========================================================================
# 25. no-broadcast structural guard
# ===========================================================================

class TestNoBroadcast:
    def test_the_reader_has_no_execution_surface(self):
        for forbidden in ("send_transaction", "sendTransaction", "sign",
                          "build", "submit", "broadcast", "swap"):
            assert not hasattr(DirectPumpReader, forbidden), forbidden

    def test_the_fake_rpc_has_no_send_method_either(self):
        """A test double offering one would invite a test to depend on it."""
        for forbidden in ("send_transaction", "sendRawTransaction", "sign"):
            assert not hasattr(FakeRpc, forbidden), forbidden

    def test_no_new_module_imports_a_signing_or_broadcasting_module(self):
        import autopaper.same_market
        import autopaper.same_market.comparison as comparison
        import autopaper.same_market.direct as direct
        import autopaper.same_market.pda as pda_mod
        import autopaper.same_market.pumpdirect as pumpdirect
        import autopaper.same_market.reader as reader

        forbidden = {"solders", "solana", "autopaper.execution.solana",
                     "autopaper.execution.txcodec", "autopaper.execution.real",
                     "autopaper.execution.raptor", "autopaper.token_cleanup"}
        for module in (pda_mod, pumpdirect, reader, comparison, direct):
            for attr in dir(module):
                value = getattr(module, attr, None)
                name = getattr(value, "__name__", "")
                assert name not in forbidden, f"{module.__name__} imports {name}"

    def test_the_guard_still_passes_with_the_new_modules_loaded(self):
        from autopaper.same_market.guard import audit_no_execution_capability

        audit = audit_no_execution_capability()
        assert audit.ok, audit.notes

    def test_no_module_defines_a_submission_symbol(self):
        from autopaper.same_market.guard import is_forbidden_attr

        import autopaper.same_market.comparison as comparison
        import autopaper.same_market.direct as direct
        import autopaper.same_market.pda as pda_mod
        import autopaper.same_market.pumpdirect as pumpdirect
        import autopaper.same_market.reader as reader

        hits = []
        for module in (pda_mod, pumpdirect, reader, comparison, direct):
            for attr in dir(module):
                if is_forbidden_attr(attr):
                    hits.append(f"{module.__name__}.{attr}")
        assert hits == []

    def test_the_quote_functions_are_pure_arithmetic(self):
        """No I/O in any function that produces a price."""
        for func in (pd.quote_bonding_curve_buy, pd.quote_pumpswap_buy,
                     pd.curve_fee_schedule, pd.amm_fee_schedule):
            source = inspect.getsource(func)
            for forbidden in ("await ", "open(", "httpx", "requests",
                              "send", "socket"):
                assert forbidden not in source, f"{func.__name__} uses {forbidden}"


# ===========================================================================
# 26. latency
# ===========================================================================

class TestLatency:
    def test_durations_come_from_the_monotonic_clock(self):
        marks = LatencyMarks()
        marks.mark("a", moment=100.0)
        marks.mark("b", moment=100.25)
        assert marks.duration_ms("a", "b") == pytest.approx(250.0)

    def test_a_missing_stage_is_none_not_zero(self):
        marks = LatencyMarks()
        marks.mark("a", moment=1.0)
        assert marks.duration_ms("a", "absent") is None

    def test_the_total_covers_the_whole_run(self):
        marks = LatencyMarks()
        marks.mark("receive", moment=0.0)
        marks.local_receive_at = 0.0
        marks.final_decision_at = 2.0
        assert marks.as_dict()["total_signal_to_final_decision_ms"] == pytest.approx(2000.0)

    def test_the_telegram_gap_is_labelled_cross_system(self):
        marks = LatencyMarks(telegram_message_at=1_700_000_000.0)
        marks.receive_wall_at = 1_700_000_001.5
        payload = marks.as_dict()
        assert payload["telegram_to_receive_ms_cross_system"] == pytest.approx(1500.0)
        assert "clock skew" in payload["clock"]

    def test_a_provider_reported_latency_is_carried_through(self):
        marks = LatencyMarks()
        marks.provider_latencies_ms["jupiter"] = 92.0
        assert marks.as_dict()["jupiter_quote_ms"] == pytest.approx(92.0)

    def test_the_reader_measures_a_real_quote_latency(self):
        quote = _run(DirectPumpReader(_live_curve_rpc()).quote_bonding_curve(
            LIVE_CURVE_MINT, sol_amount_lamports=LIVE_BUY_LAMPORTS, token_decimals=6))
        assert quote.quote_latency_ms is not None
        assert quote.quote_latency_ms >= 0

    def test_state_age_is_measured_from_the_fetch(self):
        quote = _run(DirectPumpReader(_live_curve_rpc()).quote_bonding_curve(
            LIVE_CURVE_MINT, sol_amount_lamports=LIVE_BUY_LAMPORTS, token_decimals=6))
        age = quote.snapshot.state_age_ms()
        assert age is not None and age >= 0


# ===========================================================================
# 27. deterministic JSON output
# ===========================================================================

class TestJsonOutput:
    def test_a_full_report_serialises_and_is_json_clean(self):
        from autopaper.same_market.direct import run_direct_measurement

        mint = LIVE_CURVE_MINT
        mark = ObservedMark(value_usd=4.427e-05, pair_address=LIVE_CURVE_ADDRESS,
                            dex_id="pumpfun", source="dexscreener", timestamp=1000.0,
                            latency_ms=90.0)
        jupiter = FakeJupiter(in_amount=LIVE_BUY_LAMPORTS,
                              out_amount=LIVE_SDK_CURVE_OUTPUT,
                              pool_keys=[LIVE_CURVE_ADDRESS], labels=["Pump.fun"],
                              output_mint=mint)
        report = _run(run_direct_measurement(
            mint=mint, pair=FakeDexPair(pair_address=LIVE_CURVE_ADDRESS,
                                        base_mint=mint),
            mark=mark, jupiter_result=jupiter, order_usd=0.23,
            order_lamports=LIVE_BUY_LAMPORTS, sol_reference_usd=SOL_USD,
            rpc=_live_curve_rpc(), staleness=StalenessPolicy(), divergence_pct=5.0,
        ))
        payload = report.as_dict()
        text = json.dumps(payload, indent=2, default=str)
        assert "NaN" not in text and "Infinity" not in text
        restored = json.loads(text)
        assert restored["actual_fill"] is None
        assert restored["actual_fill_available"] is False
        assert set(restored) >= {
            "observed_mark", "direct_pump_quote", "direct_pumpswap_quote",
            "jupiter_quote", "identity_checks", "pairwise_gaps",
            "diagnostics", "freshness", "latency",
        }

    def test_a_graduated_token_reports_the_curve_as_unknown_and_the_pool_as_a_quote(self):
        from autopaper.same_market.direct import run_direct_measurement

        mint = LIVE_AMM_MINT
        jupiter = FakeJupiter(in_amount=LIVE_BUY_LAMPORTS,
                              out_amount=LIVE_SDK_POOL_OUTPUT,
                              pool_keys=[LIVE_POOL_ADDRESS], labels=["Pump.fun Amm"],
                              output_mint=mint)
        report = _run(run_direct_measurement(
            mint=mint, pair=FakeDexPair(pair_address=LIVE_POOL_ADDRESS,
                                        base_mint=mint, dex_id="pumpswap"),
            mark=ObservedMark(value_usd=2.389e-05, pair_address=LIVE_POOL_ADDRESS,
                              dex_id="pumpswap", source="dexscreener",
                              timestamp=1000.0, latency_ms=80.0),
            jupiter_result=jupiter, order_usd=0.23,
            order_lamports=LIVE_BUY_LAMPORTS, sol_reference_usd=SOL_USD,
            rpc=_live_pool_rpc(), staleness=StalenessPolicy(), divergence_pct=5.0,
        ))
        payload = report.as_dict()
        assert payload["execution_market"]["market"] == "PUMPSWAP"
        assert payload["direct_pump_quote"]["available"] is False
        assert payload["direct_pump_quote"]["failure"] == "CURVE_UNAVAILABLE"
        assert payload["direct_pumpswap_quote"]["available"] is True
        assert payload["direct_pumpswap_quote"]["quote_detail"][
            "output_amount_raw"] == LIVE_SDK_POOL_OUTPUT
        names = {g["name"]: g for g in payload["pairwise_gaps"]}
        assert names["PUMPSWAP_VS_JUPITER"]["gap_pct"] == pytest.approx(0.0, abs=1e-9)
        assert payload["identity_checks"]["POOL_MATCH"] == "PASS"

    def test_a_non_pump_token_is_unknown_everywhere_and_still_serialises(self):
        from autopaper.same_market.direct import run_direct_measurement

        mint = "9pQ8MNTTqK2N1f1pMg8CmxfLW1A9fS17MLtD5D8G6fsL"
        report = _run(run_direct_measurement(
            mint=mint, pair=None,
            mark=ObservedMark(value_usd=None, pair_address=None, dex_id=None,
                              source="dexscreener", timestamp=1000.0,
                              latency_ms=None),
            jupiter_result=None, order_usd=0.23, order_lamports=LIVE_BUY_LAMPORTS,
            sol_reference_usd=SOL_USD, rpc=FakeRpc({}),
            staleness=StalenessPolicy(),
        ))
        payload = report.as_dict()
        assert payload["execution_market"]["market"] == "UNKNOWN"
        assert payload["direct_pump_quote"]["available"] is False
        assert payload["direct_pumpswap_quote"]["available"] is False
        assert payload["jupiter_quote"]["price_usd_per_token"] is None
        assert json.dumps(payload, default=str)
        for gap in payload["pairwise_gaps"]:
            assert gap["gap_pct"] is None

    def test_a_failed_rpc_stage_is_reported_not_substituted(self):
        from autopaper.same_market.direct import run_direct_measurement

        class BrokenRpc(FakeRpc):
            async def get_multiple_accounts(self, *a, **k):
                raise RuntimeError("node exploded")

        report = _run(run_direct_measurement(
            mint=LIVE_CURVE_MINT, pair=None,
            mark=ObservedMark(value_usd=4.4e-05, pair_address=None, dex_id=None,
                              source="dexscreener", timestamp=1000.0,
                              latency_ms=None),
            jupiter_result=None, order_usd=0.23, order_lamports=LIVE_BUY_LAMPORTS,
            sol_reference_usd=SOL_USD, rpc=BrokenRpc({}),
            staleness=StalenessPolicy(),
        ))
        payload = report.as_dict()
        assert payload["direct_pump_quote"]["available"] is False
        assert payload["direct_pump_quote"]["missing_reason"]


# ===========================================================================
# helpers
# ===========================================================================

def _run(coro):
    import asyncio

    return asyncio.run(coro)


def _live_curve_rpc() -> FakeRpc:
    """The live ungraduated curve's state, as a stub."""
    return FakeRpc({
        LIVE_CURVE_ADDRESS: FakeAccount(LIVE_CURVE_ADDRESS, pd.PUMP_FUN_PROGRAM,
                                        bonding_curve_bytes()),
        pd.GLOBAL_PDA: FakeAccount(pd.GLOBAL_PDA, pd.PUMP_FUN_PROGRAM, global_bytes()),
        pd.PUMP_FEE_CONFIG_PDA: FakeAccount(pd.PUMP_FEE_CONFIG_PDA, pd.PUMP_FEE_PROGRAM,
                                            curve_fee_config_bytes()),
        LIVE_CURVE_MINT: FakeAccount(LIVE_CURVE_MINT, TOKEN_PROGRAM,
                                     mint_bytes(decimals=6)),
    }, supply_value=1_000_000_000_000_000)


def _live_pool_rpc() -> FakeRpc:
    """The live PumpSwap pool's state, as a stub."""
    auth = pd.derive_pool_authority(LIVE_AMM_MINT)
    raw = pool_bytes(creator=auth, base_mint=LIVE_AMM_MINT,
                     pool_base_token_account=BASE_ATA,
                     pool_quote_token_account=QUOTE_ATA)
    curve = pd.derive_bonding_curve(LIVE_AMM_MINT)[0]
    return FakeRpc({
        curve: FakeAccount(curve, pd.PUMP_FUN_PROGRAM,
                           bonding_curve_bytes(complete=True,
                                               virtual_token_reserves=0,
                                               virtual_quote_reserves=0,
                                               real_token_reserves=0,
                                               real_quote_reserves=0)),
        LIVE_POOL_ADDRESS: FakeAccount(LIVE_POOL_ADDRESS, pd.PUMP_SWAP_PROGRAM, raw),
        LIVE_AMM_MINT: FakeAccount(LIVE_AMM_MINT, TOKEN_PROGRAM,
                                   mint_bytes(decimals=6, supply=LIVE_POOL_SUPPLY)),
        BASE_ATA: FakeAccount(BASE_ATA, TOKEN_PROGRAM,
                              token_account_bytes(amount=LIVE_POOL_BASE_RESERVE,
                                                  mint=LIVE_AMM_MINT)),
        QUOTE_ATA: FakeAccount(QUOTE_ATA, TOKEN_PROGRAM,
                               token_account_bytes(amount=LIVE_POOL_QUOTE_RESERVE,
                                                   mint=WSOL_MINT)),
        pd.AMM_GLOBAL_CONFIG_PDA: FakeAccount(pd.AMM_GLOBAL_CONFIG_PDA,
                                              pd.PUMP_SWAP_PROGRAM,
                                              amm_global_config_bytes()),
        pd.AMM_FEE_CONFIG_PDA: FakeAccount(pd.AMM_FEE_CONFIG_PDA, pd.PUMP_FEE_PROGRAM,
                                           fee_config_bytes()),
    }, supply_value=LIVE_POOL_SUPPLY)
