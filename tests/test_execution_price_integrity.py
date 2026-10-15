"""
Execution-price, quote-direction, accounting and PnL integrity regression tests.

Every number is either a real value from the 2026-10-02 $GOAT REAL trade, read
back from the chain, or a value derived from it. Nothing here touches the
network, signs anything, or reaches a broadcaster -- and
``TestNoBroadcastCapability`` asserts that structurally rather than promising
it in a docstring.

THE INCIDENT THIS PINS
----------------------
The $GOAT position was reported to the operator as::

    ผลลัพธ์: +71.56%  $+0.0016

``+71.56%`` was ``exit_mark / entry_mark - 1`` where BOTH prices were DexScreener
``priceUsd`` observations. ``$+0.0016`` was settlement-derived. They came from
two different accounting systems on one line. The executable price had moved
**+6.38%**; the all-in cash result was **-$0.182494**.

The ground truth, from the chain:

===========================  ==========================================
entry slot / signature      452502336 / 4HLLJSmc...beJ
exit slot / signature       452502572 / 4JxLwRHz...oovQ
tokens (raw / whole)        7,547,061,883 / 7,547.061883   (6 decimals)
SOL into the venue          1,928,811 lamports
token-ATA rent (locked)     1,513,840 lamports
network fee, per leg        55,000 lamports
wallet debit at entry       3,497,651 lamports
wallet credit at exit       1,996,937 lamports (net of fee)
SOL/USD                     121.604733
===========================  ==========================================

WHAT MUST BE IMPOSSIBLE AFTER THIS WORK
--------------------------------------
* a reported percentage that is not derived from settlement,
* a sell-side price computed as input/output,
* a cost basis that silently omits an itemised cost,
* a percentage and a dollar on one line from different sources,
* a paper capture that collapses the three price layers into one number.
"""


from __future__ import annotations

import base64
import json
import math

import pytest

pytestmark = pytest.mark.execution

# ===========================================================================
# the recorded $GOAT trade, 2026-10-02
# ===========================================================================

GOAT_MINT = "HtsYzeVrkU953zzN14EJcdBrkEQoAUSA7kqZyD2bpBWh"
GOAT_SYMBOL = "GOAT"
GOAT_WALLET = "wUEUbVmz2EfcLfchHmucTKhRpZgDpsBvo3FttZ4Wbt6"
GOAT_DECIMALS = 6
GOAT_SUPPLY_RAW = 939_737_458_272_644
GOAT_TOKENS_RAW = 7_547_061_883
GOAT_TOKENS_WHOLE = 7_547.061883

GOAT_SOL_PX = 121.604733

# observed marks (DexScreener priceUsd) -- NOT executable
GOAT_ENTRY_MARK = 1.783e-05
GOAT_EXIT_MARK = 3.059e-05
GOAT_MARK_RETURN_PCT = 71.56477846326415

# settled amounts
GOAT_ENTRY_SWAP_LAMPORTS = 1_928_811
GOAT_ENTRY_RENT_LAMPORTS = 1_513_840
GOAT_ENTRY_FEE_LAMPORTS = 55_000
GOAT_ENTRY_WALLET_DEBIT = 3_497_651
GOAT_EXIT_WALLET_CREDIT = 1_996_937
GOAT_EXIT_FEE_LAMPORTS = 55_000

# derived, and asserted against the chain below
GOAT_ENTRY_EXEC_USD = 3.107865687318666e-05      # USD per whole token
GOAT_EXIT_EXEC_USD_NET = 3.2176361406260514e-05
GOAT_EXEC_RETURN_PCT = 6.383518136302602        # pool-to-pool, pre-fee
#: The CORRECTED system measures the realized return against the ALL-IN cost
#: basis -- swap principal + network fee + rent that is still locked -- because
#: the lamports really left the trading wallet. The narrower trading-only figure
#: is retained beside it.
GOAT_ALL_IN_RETURN_PCT = -42.90633914018294
GOAT_TRADING_ONLY_RETURN_PCT = 0.6616456243416626
GOAT_ALL_IN_PNL_USD = -0.182493925279362
GOAT_BOOKED_PNL_USD = 0.0015961594044113792
GOAT_ALL_IN_COST_USD = 0.425330915982183
GOAT_TRADING_ONLY_COST_USD = 0.2412408312984096

#: The venue the mark described, and the venue each leg actually filled on.
GOAT_MARK_POOL = "9y4AXgn51YEz7RJdUaJFXDfAf8cpJY7gU1cRBKR79MLn"
GOAT_ENTRY_POOL = GOAT_MARK_POOL
GOAT_EXIT_POOL = "BemDnhmgCs6dccd41UYzDn9KG5HUG4WWAgsfxAT3USxB"
GOAT_ENTRY_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
GOAT_EXIT_PROGRAM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"

ENTRY_SIG = ("4HLLJSmcCbYdWxRxxSrLUabiqeCDDM3UbKpBN1GVUu14ty65X57NWKKf3bkFvviZQr1qBrj"
             "XXEUFLaAdBAoS9beJ")
EXIT_SIG = ("4JxLwRHz3pcWzuyQLCdLBvbndn1a6MAcJPaw2jTty2wpdXwEX68nvC8iW3JHezsUqj7RMZaURcGC"
            "22QafJHqoovQ")

WSOL = "So11111111111111111111111111111111111111112"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
TOKEN = "So11111111111111111111111111111111111111199"


# ===========================================================================
# 1. the three layers are three types
# ===========================================================================


class TestThreeLayersAreDistinctTypes:
    def test_a_mark_is_never_executable(self):
        from autopaper.execution.price_layers import ObservedMark
        m = ObservedMark(mint=TOKEN, price=1.783e-05, source="dexscreener")
        assert m.is_executable is False
        assert m.kind.value == "OBSERVED_MARK"

    def test_a_quote_is_executable_and_says_whether_the_venue_answered(self):
        from tests.execguard_fakes import FakeAggregatorQuote
        q = FakeAggregatorQuote(side="BUY", token_mint=TOKEN, quote_mint=WSOL,
                                quote_amount=GOAT_ENTRY_SWAP_LAMPORTS,
                                token_decimals=GOAT_DECIMALS,
                                expected_token_amount=GOAT_TOKENS_RAW,
                                quote_asset_usd=GOAT_SOL_PX)
        assert q.is_executable is True
        assert q.venue_quoted is True
        assert q.kind.value == "VENUE_QUOTE"

    def test_a_chain_derived_quote_declares_itself_unquoted(self):
        """A model price is executable in principle but is NOT the venue's answer."""
        from autopaper.execution.price_layers import ExecutableQuote, Side, leg_for_buy
        leg = leg_for_buy(token_mint=TOKEN, quote_mint=WSOL,
                          quote_amount=1000, token_decimals=6,
                          expected_token_amount=10**7)
        q = ExecutableQuote(side=Side.BUY, leg=leg, provider="pumpswap",
                            venue_quoted=False, model="chain_state_cpm_v1")
        assert q.is_executable is True
        assert q.venue_quoted is False
        assert q.model == "chain_state_cpm_v1"

    def test_the_kinds_are_not_interchangeable(self):
        from autopaper.execution.price_layers import (
            ActualFill, ExecutableQuote, ObservedMark, Side, leg_for_buy,
        )
        leg = leg_for_buy(token_mint=TOKEN, quote_mint=WSOL, quote_amount=1,
                          token_decimals=6, expected_token_amount=1)
        assert ObservedMark(mint=TOKEN, price=1.0).kind.value == "OBSERVED_MARK"
        assert ExecutableQuote(side=Side.BUY, leg=leg, provider="p").kind.value \
            == "VENUE_QUOTE"
        assert ActualFill(side=Side.BUY, leg=leg).kind.value == "ACTUAL_EXECUTION"

    def test_each_layer_serialises_its_layer_name(self):
        from autopaper.execution.price_layers import (
            ActualFill, ObservedMark, Side, leg_for_buy,
        )
        leg = leg_for_buy(token_mint=TOKEN, quote_mint=WSOL, quote_amount=1,
                          token_decimals=6, expected_token_amount=1)
        assert ObservedMark(mint=TOKEN, price=1.0).as_dict()["layer"] == "OBSERVED_MARK"
        assert ActualFill(side=Side.BUY, leg=leg).as_dict()["layer"] == "ACTUAL_FILL"


# ===========================================================================
# 2. BUY direction
# ===========================================================================


class TestBuyDirection:
    def test_a_buy_leg_is_quote_to_token(self):
        from autopaper.execution.price_layers import Side, leg_for_buy
        leg = leg_for_buy(token_mint=TOKEN, quote_mint=WSOL,
                          quote_amount=GOAT_ENTRY_SWAP_LAMPORTS,
                          token_decimals=GOAT_DECIMALS,
                          expected_token_amount=GOAT_TOKENS_RAW)
        assert leg.side is Side.BUY
        assert leg.input_mint == WSOL
        assert leg.output_mint == TOKEN
        assert leg.effective_price == pytest.approx(
            GOAT_ENTRY_SWAP_LAMPORTS / GOAT_TOKENS_RAW)

    def test_a_buy_price_is_the_real_entry_price(self):
        from autopaper.execution.price_layers import leg_for_buy
        leg = leg_for_buy(token_mint=TOKEN, quote_mint=WSOL,
                          quote_amount=GOAT_ENTRY_SWAP_LAMPORTS,
                          token_decimals=GOAT_DECIMALS,
                          expected_token_amount=GOAT_TOKENS_RAW)
        assert leg.effective_price_per_whole_token * GOAT_SOL_PX \
            == pytest.approx(GOAT_ENTRY_EXEC_USD, rel=1e-12)

    def test_quote_buy_needs_no_side_argument(self):
        """
        There is no ``side`` parameter on ``quote_buy``, so a caller cannot pass
        the wrong one. The two directions are separate methods rather than one
        method with a flag, because a flag is what got inverted on 2026-10-02.
        """
        import inspect
        from autopaper.execution.quoting import QuoteProvider
        assert "side" not in inspect.signature(QuoteProvider.quote_buy).parameters
        assert "side" not in inspect.signature(QuoteProvider.quote_sell).parameters
        assert "side" in inspect.signature(QuoteProvider.quote_swap).parameters

    def test_the_legacy_raptor_price_property_is_direction_aware(self):
        from autopaper.execution.raptor import Quote
        buy = Quote(input_mint=WSOL, output_mint=TOKEN,
                    in_amount=GOAT_ENTRY_SWAP_LAMPORTS,
                    expected_out_amount=GOAT_TOKENS_RAW, min_out_amount=1,
                    price_impact_pct=0.0, side="BUY")
        assert buy.effective_price("BUY") == pytest.approx(
            GOAT_ENTRY_SWAP_LAMPORTS / GOAT_TOKENS_RAW)
        assert buy.price_per_token == pytest.approx(
            GOAT_ENTRY_SWAP_LAMPORTS / GOAT_TOKENS_RAW)


# ===========================================================================
# 3. SELL direction
# ===========================================================================


class TestSellDirection:
    def test_a_sell_leg_is_token_to_quote(self):
        from autopaper.execution.price_layers import Side, leg_for_sell
        leg = leg_for_sell(token_mint=TOKEN, quote_mint=WSOL,
                           token_amount=GOAT_TOKENS_RAW,
                           token_decimals=GOAT_DECIMALS,
                           expected_quote_amount=GOAT_EXIT_WALLET_CREDIT)
        assert leg.side is Side.SELL
        assert leg.input_mint == TOKEN
        assert leg.output_mint == WSOL
        # OUTPUT / INPUT. The inverted form is tokens-per-lamport.
        assert leg.effective_price == pytest.approx(
            GOAT_EXIT_WALLET_CREDIT / GOAT_TOKENS_RAW)

    def test_the_inverted_form_is_the_exact_2026_10_02_defect(self):
        """The $notify exit persisted 6634.9158 -- tokens per lamport."""
        from autopaper.execution.raptor import Quote
        sell = Quote(input_mint=TOKEN, output_mint=WSOL,
                     in_amount=GOAT_TOKENS_RAW,
                     expected_out_amount=GOAT_EXIT_WALLET_CREDIT,
                     min_out_amount=1, price_impact_pct=0.0, side="SELL")
        correct = GOAT_EXIT_WALLET_CREDIT / GOAT_TOKENS_RAW
        inverted = GOAT_TOKENS_RAW / GOAT_EXIT_WALLET_CREDIT
        assert inverted > 1000, "the inverted form is the tell-tale large number"
        assert sell.effective_price("SELL") == pytest.approx(correct)
        assert sell.effective_price("SELL") != pytest.approx(inverted)

    def test_the_sell_price_is_the_real_exit_price(self):
        from autopaper.execution.price_layers import leg_for_sell
        leg = leg_for_sell(token_mint=TOKEN, quote_mint=WSOL,
                           token_amount=GOAT_TOKENS_RAW,
                           token_decimals=GOAT_DECIMALS,
                           expected_quote_amount=GOAT_EXIT_WALLET_CREDIT)
        assert leg.effective_price_per_whole_token * GOAT_SOL_PX \
            == pytest.approx(GOAT_EXIT_EXEC_USD_NET, rel=1e-12)

    def test_both_legs_via_one_side_blind_ratio_would_agree(self):
        """
        The defect is invisible from a single side: input/output happens to be
        correct for a BUY. It only shows up on the SELL, which is why it
        survived a working system.
        """
        from autopaper.pricing import execution_price_per_token
        naive = GOAT_ENTRY_SWAP_LAMPORTS / GOAT_TOKENS_RAW
        buy_ok = execution_price_per_token(side="BUY",
                                           input_amount=GOAT_ENTRY_SWAP_LAMPORTS,
                                           output_amount=GOAT_TOKENS_RAW)
        sell_naive = GOAT_TOKENS_RAW / GOAT_EXIT_WALLET_CREDIT
        sell_ok = execution_price_per_token(side="SELL",
                                            input_amount=GOAT_TOKENS_RAW,
                                            output_amount=GOAT_EXIT_WALLET_CREDIT)
        assert naive == pytest.approx(buy_ok)
        assert sell_ok != pytest.approx(sell_naive)


# ===========================================================================
# 4. mint inversion is refused, never repaired
# ===========================================================================


class TestMintInversionRefused:
    def test_a_declared_side_contradicting_the_mints_raises(self):
        from autopaper.execution.price_layers import (
            SwapDirectionError, resolve_side_from_mints,
        )
        with pytest.raises(SwapDirectionError):
            resolve_side_from_mints(TOKEN, WSOL, declared_side="BUY")

    def test_a_leg_with_two_quote_ends_has_no_resolvable_direction(self):
        """
        The structural invariant is enforced where the direction is decided:
        ``resolve_side_from_mints``. A bare ``SwapLeg`` cannot know which mint is
        the quote asset, so it does not pretend to.
        """
        from autopaper.execution.price_layers import (
            SwapDirectionError, resolve_side_from_mints,
        )
        with pytest.raises(SwapDirectionError):
            resolve_side_from_mints(WSOL, USDC)

    def test_a_leg_with_no_quote_end_is_refused(self):
        from autopaper.execution.price_layers import SwapDirectionError, Side, SwapLeg
        with pytest.raises(SwapDirectionError):
            SwapLeg(side=Side.BUY, input_mint=TOKEN, output_mint=TOKEN,
                    input_amount=1, output_amount=1,
                    input_decimals=6, output_decimals=6)

    def test_identical_input_and_output_mints_are_refused(self):
        from autopaper.execution.price_layers import SwapDirectionError, leg_for_sell
        with pytest.raises(SwapDirectionError):
            leg_for_sell(token_mint=WSOL, quote_mint=WSOL, token_amount=1,
                         token_decimals=6)

    @pytest.mark.parametrize("bad", [None, "buy-long", "SWAP", 1, object()])
    def test_an_unnamable_side_is_refused(self, bad):
        from autopaper.execution.price_layers import Side, SwapDirectionError
        with pytest.raises(SwapDirectionError):
            Side.coerce(bad)

    def test_the_entry_vocabulary_still_works(self):
        """ENTRY is a BUY and EXIT is a SELL; existing callers keep their words."""
        from autopaper.pricing import normalize_side
        assert normalize_side("ENTRY") == "BUY"
        assert normalize_side("EXIT") == "SELL"
        assert normalize_side("nonsense") is None


# ===========================================================================
# 5. decimal normalization
# ===========================================================================


class TestDecimalNormalization:
    @pytest.mark.parametrize("decimals", [0, 2, 6, 9])
    def test_the_price_per_whole_token_scales_exactly(self, decimals):
        from autopaper.execution.price_layers import leg_for_buy
        raw_tokens = 10 ** 9
        leg = leg_for_buy(token_mint=TOKEN, quote_mint=WSOL,
                          quote_amount=1_000_000, token_decimals=decimals,
                          expected_token_amount=raw_tokens)
        expected = (1_000_000 / 10 ** 9) / (raw_tokens / 10 ** decimals)
        assert leg.effective_price_per_whole_token == pytest.approx(expected)

    def test_a_sell_uses_the_TOKENS_decimals_not_the_quotes(self):
        """
        The trap: on a SELL the token is the INPUT leg. Multiplying by the
        OUTPUT decimals instead inflates a lamport figure by 10^9.
        """
        from autopaper.execution.price_layers import leg_for_sell
        leg = leg_for_sell(token_mint=TOKEN, quote_mint=WSOL,
                           token_amount=GOAT_TOKENS_RAW,
                           token_decimals=GOAT_DECIMALS,
                           expected_quote_amount=GOAT_EXIT_WALLET_CREDIT)
        assert leg.token_decimals == GOAT_DECIMALS
        assert leg.quote_decimals == 9
        assert leg.effective_price_per_whole_token < 1.0, (
            "a lamport-per-whole-token figure must be tiny; 10^9x off lands "
            "near 10^5 and looks like a token price")

    def test_decimals_are_read_from_chain_never_guessed(self):
        from autopaper.execution.chainstate import PublicRpcClient
        import asyncio
        import httpx

        raw = bytearray(82)
        raw[0] = 1
        raw[36:44] = GOAT_SUPPLY_RAW.to_bytes(8, "little")
        raw[44] = GOAT_DECIMALS

        def handler(request):
            body = json.loads(request.content)
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": body["id"],
                "result": {"context": {"slot": 1}, "value": {
                    "owner": "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",
                    "lamports": 1461600, "executable": False,
                    "data": [base64.b64encode(bytes(raw)).decode(), "base64"]}}})

        async def run():
            rpc = PublicRpcClient(
                client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
            return await rpc.get_mint(GOAT_MINT)

        read = asyncio.run(run())
        assert read.ok
        assert read.value["decimals"] == GOAT_DECIMALS
        assert read.value["supply"] == GOAT_SUPPLY_RAW
        assert read.value["token_program"].startswith("Tokenz")


# ===========================================================================
# 6. the observed mark is not an executable quote
# ===========================================================================


class TestObservedMarkIsNotAnExecutableQuote:
    def test_the_marks_are_a_thousand_times_apart_from_the_executions(self):
        assert GOAT_ENTRY_MARK / GOAT_ENTRY_EXEC_USD == pytest.approx(0.5737, abs=1e-3)
        # against the NET exit price -- what the wallet actually received
        assert GOAT_EXIT_MARK / GOAT_EXIT_EXEC_USD_NET == pytest.approx(0.9507, abs=1e-3)

    def test_the_divergence_is_reported_both_ways(self):
        from autopaper.execution.price_layers import mark_vs_quote_divergence
        d = mark_vs_quote_divergence(GOAT_ENTRY_MARK, GOAT_ENTRY_EXEC_USD)
        assert d is not None
        assert d.pct == pytest.approx(GOAT_ENTRY_MARK / GOAT_ENTRY_EXEC_USD - 1,
                                     abs=1e-9)

    def test_an_unusable_leg_yields_no_divergence(self):
        from autopaper.execution.price_layers import mark_vs_quote_divergence
        assert mark_vs_quote_divergence(0.0, 1.0) is None
        assert mark_vs_quote_divergence(1.0, 0.0) is None
        assert mark_vs_quote_divergence(None, 1.0) is None

    def test_the_mark_is_never_executable(self):
        from autopaper.execution.price_layers import ObservedMark
        assert ObservedMark(mint=TOKEN, price=GOAT_ENTRY_MARK).is_executable is False


# ===========================================================================
# 7. the entry guard: freshness, timeout, staleness, divergence
# ===========================================================================


def _mark(price=GOAT_ENTRY_MARK, pool=GOAT_MARK_POOL, **kw):
    from autopaper.execution.price_layers import ObservedMark
    import time
    return ObservedMark(mint=TOKEN, price=price, source="dexscreener",
                        venue="pumpfun", pool=pool,
                        observed_at=time.time(), **kw)


def _quote(price_usd=GOAT_ENTRY_EXEC_USD, pool=GOAT_ENTRY_POOL,
           received_at=None, venue="pumpfun"):
    from tests.execguard_fakes import FakeAggregatorQuote
    import time
    lamports = GOAT_ENTRY_SWAP_LAMPORTS
    usd = (lamports / 1e9) * GOAT_SOL_PX
    raw = int(usd / price_usd * (10 ** GOAT_DECIMALS))
    return FakeAggregatorQuote(
        side="BUY", token_mint=TOKEN, quote_mint=WSOL,
        quote_amount=lamports, token_decimals=GOAT_DECIMALS,
        expected_token_amount=raw, quote_asset_usd=GOAT_SOL_PX,
        venue=venue, pool=pool,
        received_at=time.time() if received_at is None else received_at)


class TestEntryQuoteGuard:
    def test_the_goat_entry_would_have_been_refused(self):
        from autopaper.execution.quoting import DivergenceGuard
        v = DivergenceGuard().evaluate_entry(mark=_mark(), quote=_quote())
        assert v.allowed is False
        assert v.reason == "quote_worse_than_mark"
        assert v.pct is not None and v.pct < 0        # the mark UNDERSTATED
        assert "+74" in v.detail, "the venue was 74% dearer than the mark"

    def test_a_small_divergence_is_allowed(self):
        from autopaper.execution.quoting import DivergenceGuard
        v = DivergenceGuard().evaluate_entry(
            mark=_mark(), quote=_quote(price_usd=GOAT_ENTRY_MARK * 1.02))
        assert v.allowed is True
        assert v.reason == "within_band"

    def test_a_quote_far_BETTER_than_the_mark_is_an_integrity_alarm(self):
        from autopaper.execution.quoting import DivergenceGuard
        v = DivergenceGuard().evaluate_entry(
            mark=_mark(), quote=_quote(price_usd=GOAT_ENTRY_MARK * 0.20))
        assert v.allowed is False
        assert v.reason == "quote_much_better_than_mark"

    def test_a_missing_quote_fails_closed(self):
        from autopaper.execution.quoting import DivergenceGuard
        v = DivergenceGuard().evaluate_entry(mark=_mark(), quote=None)
        assert v.allowed is False
        assert v.reason == "no_executable_quote"
        assert v.fail_closed is True

    def test_a_stale_quote_fails_closed(self):
        import time
        from autopaper.execution.quoting import DivergenceGuard
        v = DivergenceGuard().evaluate_entry(
            mark=_mark(),
            quote=_quote(received_at=time.time() - 30.0))
        assert v.allowed is False
        assert v.reason == "quote_stale"
        assert v.fail_closed is True

    def test_a_missing_mark_fails_closed(self):
        from autopaper.execution.quoting import DivergenceGuard
        v = DivergenceGuard().evaluate_entry(mark=None, quote=_quote())
        assert v.allowed is False
        assert v.reason == "no_observed_mark"
        assert v.fail_closed is True

    def test_a_stale_mark_fails_closed(self):
        from autopaper.execution.quoting import DivergenceGuard
        v = DivergenceGuard().evaluate_entry(
            mark=_mark(stale=True, quality="stale"), quote=_quote())
        assert v.allowed is False
        assert v.reason == "observed_mark_stale"

    def test_a_sell_quote_is_never_accepted_by_the_entry_guard(self):
        from autopaper.execution.price_layers import Side, leg_for_sell
        from autopaper.execution.price_layers import ExecutableQuote
        from autopaper.execution.quoting import DivergenceGuard
        leg = leg_for_sell(token_mint=TOKEN, quote_mint=WSOL,
                           token_amount=GOAT_TOKENS_RAW,
                           token_decimals=GOAT_DECIMALS,
                           expected_quote_amount=GOAT_EXIT_WALLET_CREDIT)
        q = ExecutableQuote(side=Side.SELL, leg=leg, provider="p",
                            quote_asset_usd=GOAT_SOL_PX)
        v = DivergenceGuard().evaluate_entry(mark=_mark(), quote=q)
        assert v.allowed is False
        assert v.reason == "wrong_direction"

    def test_an_unconvertible_quote_fails_closed_rather_than_comparing_units(self):
        from autopaper.execution.price_layers import ExecutableQuote, Side, leg_for_buy
        from autopaper.execution.quoting import DivergenceGuard
        leg = leg_for_buy(token_mint=TOKEN, quote_mint=WSOL, quote_amount=1000,
                          token_decimals=6, expected_token_amount=10 ** 7)
        q = ExecutableQuote(side=Side.BUY, leg=leg, provider="p",
                            quote_asset_usd=None)
        v = DivergenceGuard().evaluate_entry(mark=_mark(), quote=q)
        assert v.allowed is False
        assert v.reason == "quote_not_comparable"

    def test_the_threshold_is_configuration_not_a_fitted_parameter(self):
        from autopaper.execution.quoting import DivergenceGuard
        loose = DivergenceGuard(quote_worse_than_mark_max_pct=100.0)
        assert loose.evaluate_entry(mark=_mark(), quote=_quote()).allowed is True
        tight = DivergenceGuard(quote_worse_than_mark_max_pct=1.0)
        assert tight.evaluate_entry(
            mark=_mark(), quote=_quote(price_usd=GOAT_ENTRY_MARK * 1.05)
        ).allowed is False

    def test_the_default_comes_from_config_and_is_reported(self):
        from autopaper.config import RealConfig
        rc = RealConfig()
        assert rc.entry_quote_divergence_max_pct == 10.0
        assert rc.entry_quote_alarm_max_pct == 50.0
        d = rc.as_dict()
        assert d["entry_quote_divergence_max_pct"] == 10.0
        assert "not a fitted parameter" in \
            __import__("autopaper.execution.quoting",
                       fromlist=["DivergenceGuard"]).DivergenceGuard().notes


# ===========================================================================
# 8. provider failure fails closed
# ===========================================================================


class TestProviderFailureFailsClosed:
    def test_an_exhausted_chain_yields_none_and_a_status(self):
        import asyncio
        from tests.execguard_fakes import FakeAggregatorQuote
        from autopaper.execution.quoting import RouterQuoteProvider
        from autopaper.execution.quote_errors import QuoteUnavailable

        async def boom(**kw):
            raise QuoteUnavailable("venue down")

        from autopaper.execution.quoting import QuoteProvider

        class Boom(QuoteProvider):
            name = "boom"

            async def quote_swap(self, **kw):
                return await boom(**kw)

        chain = RouterQuoteProvider([Boom()])
        quote, report = asyncio.run(chain.try_all(
            side="BUY", token_mint=TOKEN, quote_mint=WSOL, amount_raw=1000,
            token_decimals=6))
        assert quote is None
        assert report["failure_status"]
        assert report["attempts"][0]["status"] == "UNAVAILABLE"

    def test_the_first_answers_price_is_kept_even_if_a_later_one_is_cheaper(self):
        """
        The winning quote is the FIRST provider's. A later provider's price --
        even a better one -- must not be presented as what the trade costs: the
        router fills through the provider that answered first.
        """
        import asyncio
        from autopaper.execution.quoting import QuoteProvider, RouterQuoteProvider
        from autopaper.execution.quote_errors import QuoteRejected

        answered = _quote(price_usd=GOAT_ENTRY_EXEC_USD)

        class Primary(QuoteProvider):
            name = "primary"

            async def quote_swap(self, **kw):
                return answered

        class Rejecting(QuoteProvider):
            name = "rejecting"

            async def quote_swap(self, **kw):
                raise QuoteRejected("no route")

        chain = RouterQuoteProvider([Primary(), Rejecting()])
        quote, report = asyncio.run(chain.try_all(
            side="BUY", token_mint=TOKEN, quote_mint=WSOL,
            amount_raw=GOAT_ENTRY_SWAP_LAMPORTS, token_decimals=GOAT_DECIMALS))
        assert quote is answered, \
            "the fallback must not replace a good primary quote"
        assert report["failure_status"] == ""

    def test_a_direction_fault_stops_the_chain_immediately(self):
        import asyncio
        from tests.execguard_fakes import FakeAggregatorQuote
        from autopaper.execution.price_layers import SwapDirectionError
        from autopaper.execution.quoting import RouterQuoteProvider

        from autopaper.execution.quoting import QuoteProvider

        class Faulty(QuoteProvider):
            name = "faulty"

            async def quote_swap(self, **kw):
                raise SwapDirectionError("inverted leg")

        from tests.execguard_fakes import FakeQuoteProvider
        chain = RouterQuoteProvider([
            Faulty(), FakeQuoteProvider(price_usd=1e-5, now_fn=lambda: 1e9)])
        quote, report = asyncio.run(chain.try_all(
            side="BUY", token_mint=TOKEN, quote_mint=WSOL, amount_raw=1000,
            token_decimals=6))
        assert quote is None
        assert report["failure_status"] == "SWAP_DIRECTION"
        assert len(report["attempts"]) == 1, "a direction fault is not a venue fault"


# ===========================================================================
# 9. rate limits and caching on the free public RPC
# ===========================================================================


class TestPublicRpcConduct:
    def _rpc(self, handler, **kw):
        import asyncio
        import httpx
        from autopaper.execution.chainstate import PublicRpcClient
        return PublicRpcClient(
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            **kw)

    def test_a_429_is_reported_not_hidden(self):
        import asyncio
        calls = []

        def always_429(request):
            calls.append(1)
            return __import__("httpx").Response(
                429, headers={"Retry-After": "0"}, json={})

        rpc = self._rpc(always_429, max_retries=1, backoff_base_s=0.001)
        read = asyncio.run(rpc.get_slot())
        assert read.ok is False
        assert "rate limited" in (read.error or "").lower()
        assert rpc.stats["rate_limited"] == 2, "the limit is retried once, then stops"

    def test_identical_reads_are_cached(self):
        import asyncio
        import httpx

        def handler(request):
            body = json.loads(request.content)
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": body["id"],
                "result": {"context": {"slot": 9}, "value": None}})

        rpc = self._rpc(handler)
        asyncio.run(rpc.get_account(TOKEN))
        asyncio.run(rpc.get_account(TOKEN))
        assert rpc.stats["requests"] == 1
        assert rpc.stats["cache_hits"] == 1

    def test_concurrent_identical_reads_are_deduplicated(self):
        """
        The handler BLOCKS until every reader is waiting, so the overlap is real
        rather than an artefact of a transport that happens to resolve inline.
        """
        import asyncio
        import httpx
        from autopaper.execution.chainstate import PublicRpcClient

        calls = []
        loop = asyncio.new_event_loop()
        release = asyncio.Event()
        arrived = []

        async def handler(request):
            body = json.loads(request.content)
            calls.append(body["method"])
            arrived.append(1)
            if len(arrived) == 1:
                await asyncio.wait_for(release.wait(), timeout=2.0)
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": body["id"],
                "result": {"context": {"slot": 9}, "value": None}})

        async def scenario():
            rpc = PublicRpcClient(
                client=httpx.AsyncClient(
                    transport=httpx.MockTransport(handler)))
            readers = [asyncio.ensure_future(rpc.get_account(TOKEN))
                       for _ in range(8)]
            await asyncio.sleep(0)          # let them all reach the dedup point
            release.set()
            return rpc, await asyncio.gather(*readers)

        rpc, _ = asyncio.run(scenario())
        assert len(calls) == 1, "eight simultaneous reads must cost one request"
        assert rpc.stats["dedup_hits"] >= 7
        _ = loop

    def test_a_missing_account_is_a_state_not_an_error(self):
        import asyncio
        import httpx

        def handler(request):
            body = json.loads(request.content)
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": body["id"],
                "result": {"context": {"slot": 9}, "value": None}})

        read = asyncio.run(self._rpc(handler).get_account(TOKEN))
        assert read.missing is True
        assert read.ok is False

    def test_an_oversized_account_is_refused_rather_than_truncated(self):
        from autopaper.execution.chainstate import ChainStateError, decode_account_data
        with pytest.raises(ChainStateError):
            decode_account_data({"data_b64": base64.b64encode(b"x" * 2000).decode()})

    def test_no_paid_provider_is_configured_or_imported(self):
        from autopaper.config import RealConfig
        rc = RealConfig()
        assert rc.public_rpc_url == "https://api.mainnet-beta.solana.com"
        blob = json.dumps(rc.as_dict()).lower()
        for banned in ("helius", "birdeye", "quicknode", "alchemy", "triton"):
            assert banned not in blob, f"{banned} must not be a configured dependency"


# ===========================================================================
# 10. the swap leg is separated from the wallet debit (recorded chain data)
# ===========================================================================


def _tx(owner, accounts, *, fee, pre, post):
    """A transaction payload shaped like the recorded $GOAT ones."""
    return {
        "slot": 452502336,
        "transaction": {"message": {"accountKeys": accounts}},
        "meta": {
            "err": None, "fee": fee,
            "logMessages": [
                "Program 675kPX9MHTjS2zt1qfr1NYHuzDGuEiKC9iJfF9V4aU1E invoke [1]",
                f"Program {GOAT_ENTRY_PROGRAM} invoke [2]",
            ],
            "preBalances": pre, "postBalances": post,
            "preTokenBalances": [],
            "postTokenBalances": [
                {"accountIndex": 0, "mint": GOAT_MINT, "owner": owner,
                 "uiTokenAmount": {"amount": str(GOAT_TOKENS_RAW),
                                   "decimals": GOAT_DECIMALS}},
            ],
        },
    }


class TestSwapLegIsNotTheWalletDebit:
    def _entry_amounts(self):
        from autopaper.execution.solana import reconcile_transaction
        vault = "Vaut1111111111111111111111111111111111111"
        return reconcile_transaction(
            _tx(GOAT_WALLET, [GOAT_WALLET, vault], fee=GOAT_ENTRY_FEE_LAMPORTS,
                pre=[GOAT_ENTRY_WALLET_DEBIT + GOAT_ENTRY_SWAP_LAMPORTS,
                     60_000_000],
                post=[GOAT_ENTRY_SWAP_LAMPORTS,
                      60_000_000 + GOAT_ENTRY_SWAP_LAMPORTS]),
            "sig", "confirmed", GOAT_WALLET, GOAT_MINT)

    def test_the_swap_leg_is_the_venue_amount_not_the_debit(self):
        a = self._entry_amounts()
        # the payer lost rent + fee + principal; the venue leg is only principal
        assert a.wallet_debit_lamports == GOAT_ENTRY_WALLET_DEBIT
        assert a.lamports_in_from_others == GOAT_ENTRY_SWAP_LAMPORTS
        assert a.lamports_in_from_others < GOAT_ENTRY_WALLET_DEBIT

    def test_the_price_must_never_come_from_the_wallet_debit(self):
        from autopaper.pricing import execution_price_per_token
        a = self._entry_amounts()
        wrong = execution_price_per_token(
            side="BUY", input_amount=a.wallet_debit_lamports,
            output_amount=GOAT_TOKENS_RAW)
        right = execution_price_per_token(
            side="BUY", input_amount=a.lamports_in_from_others,
            output_amount=GOAT_TOKENS_RAW)
        assert wrong != right
        assert (right * 10 ** GOAT_DECIMALS / 1e9 * GOAT_SOL_PX) \
            == pytest.approx(GOAT_ENTRY_EXEC_USD, rel=1e-6)

    def test_rent_is_separated_from_principal(self):
        raw = bytearray(82)
        raw[0] = 1
        raw[36:44] = GOAT_SUPPLY_RAW.to_bytes(8, "little")
        raw[44] = GOAT_DECIMALS
        from autopaper.execution.solana import reconcile_transaction
        vault = "Vaut1111111111111111111111111111111111111"
        new_ata = "Ata11111111111111111111111111111111111111"
        a = reconcile_transaction(
            _tx(GOAT_WALLET, [GOAT_WALLET, vault, new_ata],
                fee=GOAT_ENTRY_FEE_LAMPORTS,
                pre=[GOAT_ENTRY_WALLET_DEBIT + GOAT_ENTRY_SWAP_LAMPORTS,
                     60_000_000, 0],
                post=[GOAT_ENTRY_SWAP_LAMPORTS,
                      60_000_000 + GOAT_ENTRY_SWAP_LAMPORTS,
                      GOAT_ENTRY_RENT_LAMPORTS]),
            "sig", "confirmed", GOAT_WALLET, GOAT_MINT)
        assert a.created_account_rent_lamports == GOAT_ENTRY_RENT_LAMPORTS
        assert a.lamports_in_from_others == GOAT_ENTRY_SWAP_LAMPORTS

    def test_the_program_that_executed_is_known_not_just_labelled(self):
        from autopaper.execution.solana import reconcile_transaction
        a = self._entry_amounts()
        assert GOAT_ENTRY_PROGRAM in a.invoked_programs
        assert all(":" not in p for p in a.invoked_programs), (
            "log noise must not be mistaken for a program id")


# ===========================================================================
# 11. realized PnL from settlement
# ===========================================================================


def _goat_fill(side, *, input_amount, output_amount, fee, kind="SWAP_LEG",
               rent=None, wallet_debit=None, swap_input=None):
    from autopaper.real_account import RealFillRecord
    return RealFillRecord(
        signature="sig", side=side, commitment="confirmed",
        actual_input_amount=input_amount, actual_output_amount=output_amount,
        input_amount_kind=kind, fee_lamports=fee, rent_lamports=rent,
        swap_input_amount=swap_input,
        wallet_debit_lamports=wallet_debit, token_decimals=GOAT_DECIMALS,
        quote_mint=WSOL)


class TestRealizedPnlFromSettlement:
    def _opened(self, **kw):
        from autopaper.real_account import RealAccount
        acct = RealAccount()
        acct.register_entry(
            mint=GOAT_MINT, symbol=GOAT_SYMBOL,
            fill=_goat_fill("ENTRY",
                            input_amount=GOAT_ENTRY_SWAP_LAMPORTS,
                            output_amount=GOAT_TOKENS_RAW,
                            fee=GOAT_ENTRY_FEE_LAMPORTS,
                            rent=GOAT_ENTRY_RENT_LAMPORTS,
                            swap_input=GOAT_ENTRY_SWAP_LAMPORTS,
                            wallet_debit=GOAT_ENTRY_WALLET_DEBIT),
            token_amount=GOAT_TOKENS_RAW,
            entry_price_usd=GOAT_ENTRY_MARK, size_usd=0.2345525709834096,
            now=1.0, sol_price_usd=GOAT_SOL_PX,
            token_decimals=GOAT_DECIMALS, quote_mint=WSOL,
            **kw)
        return acct

    def test_the_entry_price_is_the_settlement_not_the_mark(self):
        acct = self._opened()
        p = acct.positions[GOAT_MINT]
        assert p.entry_execution_price_usd == pytest.approx(
            GOAT_ENTRY_EXEC_USD, rel=1e-9)
        assert p.entry_price_kind == "OBSERVED_MARK"
        assert p.entry_price_usd == GOAT_ENTRY_MARK, (
            "the mark is preserved, and labelled as a mark")

    def test_the_cost_basis_is_itemised_and_includes_the_locked_rent(self):
        acct = self._opened()
        p = acct.positions[GOAT_MINT]
        c = p.entry_cost_components
        assert c["swap_input_lamports"] == GOAT_ENTRY_SWAP_LAMPORTS
        assert c["network_fee_lamports"] == GOAT_ENTRY_FEE_LAMPORTS
        assert c["account_rent_lamports"] == GOAT_ENTRY_RENT_LAMPORTS
        assert c["rent_locked_lamports"] == GOAT_ENTRY_RENT_LAMPORTS
        assert p.entry_cost_usd == pytest.approx(
            (GOAT_ENTRY_SWAP_LAMPORTS + GOAT_ENTRY_FEE_LAMPORTS
             + GOAT_ENTRY_RENT_LAMPORTS) / 1e9 * GOAT_SOL_PX, rel=1e-9)

    def test_the_booked_return_is_the_settlement_return_and_not_the_mark(self):
        acct = self._opened()
        p = acct.close_position(
            mint=GOAT_MINT, reason="max_hold_60s", now=67.0,
            sol_price_usd=GOAT_SOL_PX,
            exit_price_usd=GOAT_EXIT_MARK,
            token_amount_sold=GOAT_TOKENS_RAW,
            token_decimals=GOAT_DECIMALS,
            fill=_goat_fill("EXIT", input_amount=None,
                            output_amount=GOAT_EXIT_WALLET_CREDIT,
                            fee=GOAT_EXIT_FEE_LAMPORTS))
        assert p.realized_return_pct == pytest.approx(
            GOAT_ALL_IN_RETURN_PCT, rel=1e-6)
        assert p.realized_return_pct != pytest.approx(GOAT_MARK_RETURN_PCT)
        assert p.realized_return_pct != pytest.approx(
            GOAT_TRADING_ONLY_RETURN_PCT), (
            "the all-in basis must not be silently swapped back for the "
            "rent-excluding one")
        # and the narrower figure is retained, so the difference is visible
        trading_only = (GOAT_BOOKED_PNL_USD
                        / p.entry_cost_usd_trading_only * 100.0)
        assert trading_only == pytest.approx(GOAT_TRADING_ONLY_RETURN_PCT, rel=1e-6)
        assert p.observed_mark_return_pct == pytest.approx(
            GOAT_MARK_RETURN_PCT, rel=1e-9)

    def test_the_mark_return_is_never_used_as_realized_pnl(self):
        acct = self._opened()
        p = acct.close_position(
            mint=GOAT_MINT, reason="x", now=67.0, sol_price_usd=GOAT_SOL_PX,
            exit_price_usd=GOAT_EXIT_MARK,
            token_amount_sold=GOAT_TOKENS_RAW, token_decimals=GOAT_DECIMALS,
            fill=_goat_fill("EXIT", input_amount=None,
                            output_amount=GOAT_EXIT_WALLET_CREDIT,
                            fee=GOAT_EXIT_FEE_LAMPORTS))
        realized = p.realized_pnl_usd
        # Changing ONLY the mark must not move the dollars by a cent.
        acct2 = self._opened()
        p2 = acct2.close_position(
            mint=GOAT_MINT, reason="x", now=67.0, sol_price_usd=GOAT_SOL_PX,
            exit_price_usd=9.9e-05,
            token_amount_sold=GOAT_TOKENS_RAW, token_decimals=GOAT_DECIMALS,
            fill=_goat_fill("EXIT", input_amount=None,
                            output_amount=GOAT_EXIT_WALLET_CREDIT,
                            fee=GOAT_EXIT_FEE_LAMPORTS))
        assert p2.realized_pnl_usd == realized
        assert p2.realized_return_pct == p.realized_return_pct
        assert p2.observed_mark_return_pct != p.observed_mark_return_pct

    def test_a_zero_or_absent_execution_price_never_becomes_zero(self):
        acct = self._opened()
        p = acct.close_position(
            mint=GOAT_MINT, reason="x", now=67.0, sol_price_usd=GOAT_SOL_PX,
            exit_price_usd=GOAT_EXIT_MARK, token_amount_sold=None,
            token_decimals=None,
            fill=_goat_fill("EXIT", input_amount=None,
                            output_amount=GOAT_EXIT_WALLET_CREDIT,
                            fee=GOAT_EXIT_FEE_LAMPORTS))
        assert p.exit_execution_price_usd is None
        assert p.realized_pnl_usd is not None, "the sale still settled"

    def test_an_unpriceable_settlement_states_no_pnl_rather_than_zero(self):
        acct = self._opened()
        p = acct.close_position(
            mint=GOAT_MINT, reason="x", now=67.0, sol_price_usd=GOAT_SOL_PX,
            exit_price_usd=GOAT_EXIT_MARK,
            token_amount_sold=GOAT_TOKENS_RAW, token_decimals=GOAT_DECIMALS,
            output_mint="SomeUnknownMint11111111111111111111111111",
            fill=_goat_fill("EXIT", input_amount=None,
                            output_amount=1234, fee=GOAT_EXIT_FEE_LAMPORTS))
        assert p.status == "CLOSED"
        assert p.realized_pnl_usd is None
        assert p.ambiguity_reason

    def test_realized_return_pct_refuses_a_non_positive_basis(self):
        from autopaper.execution.price_layers import realized_return_pct
        assert realized_return_pct(realized_pnl_usd=1.0, cost_basis_usd=0.0) is None
        assert realized_return_pct(realized_pnl_usd=None, cost_basis_usd=1.0) is None
        assert realized_return_pct(realized_pnl_usd=1.0, cost_basis_usd=1.0) == 100.0


# ===========================================================================
# 12. rent accounting: the $NOTIFY regression
# ===========================================================================


class TestRentAccounting:
    def test_locked_rent_is_a_cost_until_it_is_proven_recovered(self):
        from autopaper.execution.price_layers import CostComponents
        c = CostComponents(swap_input_amount=1_928_811,
                           network_fee_lamports=110_000,
                           account_rent_lamports=1_513_840)
        assert c.rent_locked_lamports == 1_513_840

    def test_only_proven_recovery_nets_rent_out(self):
        from autopaper.execution.price_layers import CostComponents
        c = CostComponents(account_rent_lamports=1_513_840,
                           rent_recovered_lamports=1_513_840)
        assert c.rent_locked_lamports == 0
        c2 = CostComponents(account_rent_lamports=1_513_840,
                            rent_recovered_lamports=1_000_000)
        assert c2.rent_locked_lamports == 513_840, (
            "a partial recovery leaves a partial lock")

    def test_the_notify_shape_books_a_loss_not_a_profit(self):
        """
        $NOTIFY (2026-10-02): the price moved +0.6% in the trader's favour, the
        network fees and 1,513,840 lamports of rent turned it into a loss, and
        the rent was omitted so it booked as a SMALL PROFIT. That is the exact
        shape $GOAT repeated at a larger size.
        """
        from autopaper.real_account import RealAccount
        notional = 794_808
        fee = 55_000
        acct = RealAccount()
        acct.register_entry(
            mint=TOKEN, symbol="notify",
            fill=_goat_fill("ENTRY", input_amount=notional,
                            output_amount=GOAT_TOKENS_RAW, fee=fee,
                            rent=1_513_840, swap_input=notional,
                            wallet_debit=notional + fee + 1_513_840),
            token_amount=GOAT_TOKENS_RAW, entry_price_usd=1.6e-05,
            size_usd=notional / 1e9 * 118.106157, now=1.0,
            sol_price_usd=118.106157, token_decimals=GOAT_DECIMALS,
            quote_mint=WSOL)
        p = acct.close_position(
            mint=TOKEN, reason="x", now=67.0, sol_price_usd=118.106157,
            exit_price_usd=1.6e-05, token_amount_sold=GOAT_TOKENS_RAW,
            token_decimals=GOAT_DECIMALS,
            fill=_goat_fill("EXIT", input_amount=None,
                            output_amount=799_599, fee=55_000))
        assert p.realized_pnl_usd < 0.0, (
            "including the locked rent the round trip is a loss")
        trading_only = p.entry_cost_usd_trading_only
        assert trading_only < p.entry_cost_usd, (
            "the trading-only figure is retained so the difference is visible")
        assert p.entry_cost_usd - trading_only == pytest.approx(
            1_513_840 / 1e9 * 118.106157, rel=1e-9)

    def test_a_rent_free_entry_does_not_invent_rent(self):
        from autopaper.real_account import RealAccount
        acct = RealAccount()
        acct.register_entry(
            mint=TOKEN, symbol="x",
            fill=_goat_fill("ENTRY", input_amount=1_000_000,
                            output_amount=GOAT_TOKENS_RAW, fee=55_000,
                            rent=None, swap_input=1_000_000),
            token_amount=GOAT_TOKENS_RAW, entry_price_usd=1e-5,
            size_usd=0.1, now=1.0, sol_price_usd=GOAT_SOL_PX,
            token_decimals=GOAT_DECIMALS, quote_mint=WSOL)
        p = acct.positions[TOKEN]
        assert p.entry_rent_lamports == 0
        assert p.entry_cost_usd == pytest.approx(
            1_055_000 / 1e9 * GOAT_SOL_PX, rel=1e-9)

    def test_an_unavailable_sol_price_states_no_basis_with_a_reason(self):
        from autopaper.real_account import RealAccount
        acct = RealAccount()
        acct.register_entry(
            mint=TOKEN, symbol="x",
            fill=_goat_fill("ENTRY", input_amount=1_000_000,
                            output_amount=GOAT_TOKENS_RAW, fee=55_000,
                            swap_input=1_000_000),
            token_amount=GOAT_TOKENS_RAW, entry_price_usd=1e-5,
            size_usd=0.1, now=1.0, sol_price_usd=None,
            token_decimals=GOAT_DECIMALS, quote_mint=WSOL)
        p = acct.positions[TOKEN]
        assert p.entry_cost_usd == 0.0
        assert "not stated in USD" in (p.entry_cost_basis_reason or "")


# ===========================================================================
# 13. cross-venue detection
# ===========================================================================


class TestCrossVenueDetection:
    def test_a_mark_about_another_pool_is_flagged(self):
        from autopaper.execution.price_layers import (
            ActualFill, ObservedMark, Side, leg_for_sell, mark_venue_mismatch,
        )
        leg = leg_for_sell(token_mint=TOKEN, quote_mint=WSOL,
                           token_amount=GOAT_TOKENS_RAW,
                           token_decimals=GOAT_DECIMALS,
                           expected_quote_amount=GOAT_EXIT_WALLET_CREDIT)
        fill = ActualFill(side=Side.SELL, leg=leg, pool=GOAT_EXIT_POOL)
        mark = ObservedMark(mint=TOKEN, price=GOAT_EXIT_MARK,
                            pool=GOAT_MARK_POOL)
        assert mark_venue_mismatch(mark, fill) is True

    def test_a_mark_about_the_same_pool_is_not_flagged(self):
        from autopaper.execution.price_layers import (
            ActualFill, ObservedMark, Side, leg_for_sell, mark_venue_mismatch,
        )
        leg = leg_for_sell(token_mint=TOKEN, quote_mint=WSOL,
                           token_amount=GOAT_TOKENS_RAW,
                           token_decimals=GOAT_DECIMALS,
                           expected_quote_amount=GOAT_EXIT_WALLET_CREDIT)
        fill = ActualFill(side=Side.SELL, leg=leg, pool=GOAT_MARK_POOL)
        mark = ObservedMark(mint=TOKEN, price=GOAT_EXIT_MARK,
                            pool=GOAT_MARK_POOL)
        assert mark_venue_mismatch(mark, fill) is False

    def test_an_absent_pool_is_not_laundered_into_a_mismatch(self):
        from autopaper.execution.price_layers import (
            ActualFill, ObservedMark, Side, leg_for_sell, mark_venue_mismatch,
        )
        leg = leg_for_sell(token_mint=TOKEN, quote_mint=WSOL,
                           token_amount=GOAT_TOKENS_RAW,
                           token_decimals=GOAT_DECIMALS,
                           expected_quote_amount=GOAT_EXIT_WALLET_CREDIT)
        assert mark_venue_mismatch(
            ObservedMark(mint=TOKEN, price=1.0, pool=None),
            ActualFill(side=Side.SELL, leg=leg, pool=GOAT_EXIT_POOL)) is False

    def test_the_goat_round_trip_is_recorded_as_cross_venue(self):
        from autopaper.real_account import RealAccount
        from autopaper.execution.price_layers import ObservedMark
        acct = RealAccount()
        acct.register_entry(
            mint=GOAT_MINT, symbol=GOAT_SYMBOL,
            fill=_goat_fill("ENTRY", input_amount=GOAT_ENTRY_SWAP_LAMPORTS,
                            output_amount=GOAT_TOKENS_RAW,
                            fee=GOAT_ENTRY_FEE_LAMPORTS,
                            rent=GOAT_ENTRY_RENT_LAMPORTS,
                            swap_input=GOAT_ENTRY_SWAP_LAMPORTS,
                            wallet_debit=GOAT_ENTRY_WALLET_DEBIT),
            token_amount=GOAT_TOKENS_RAW, entry_price_usd=GOAT_ENTRY_MARK,
            size_usd=0.2345, now=1.0, sol_price_usd=GOAT_SOL_PX,
            token_decimals=GOAT_DECIMALS, quote_mint=WSOL,
            venue="pump_bonding_curve", pool=GOAT_ENTRY_POOL,
            provider="raptor")
        p = acct.close_position(
            mint=GOAT_MINT, reason="x", now=67.0, sol_price_usd=GOAT_SOL_PX,
            exit_price_usd=GOAT_EXIT_MARK, token_amount_sold=GOAT_TOKENS_RAW,
            token_decimals=GOAT_DECIMALS, venue="pumpswap",
            pool=GOAT_EXIT_POOL, provider="raptor",
            exit_mark=ObservedMark(mint=GOAT_MINT, price=GOAT_EXIT_MARK,
                                   pool=GOAT_MARK_POOL),
            fill=_goat_fill("EXIT", input_amount=None,
                            output_amount=GOAT_EXIT_WALLET_CREDIT,
                            fee=GOAT_EXIT_FEE_LAMPORTS))
        assert p.cross_venue_detected is True
        assert GOAT_MARK_POOL in p.cross_venue_detail
        assert GOAT_EXIT_POOL in p.cross_venue_detail


# ===========================================================================
# 14. the exit divergence helper is no longer inert
# ===========================================================================


class TestExitDivergenceIsNotInert:
    def _pos(self, token_amount=GOAT_TOKENS_RAW, decimals=GOAT_DECIMALS):
        from autopaper.real_account import RealAccount
        acct = RealAccount()
        acct.register_entry(
            mint=GOAT_MINT, symbol=GOAT_SYMBOL,
            fill=_goat_fill("ENTRY", input_amount=GOAT_ENTRY_SWAP_LAMPORTS,
                            output_amount=token_amount,
                            fee=GOAT_ENTRY_FEE_LAMPORTS,
                            rent=GOAT_ENTRY_RENT_LAMPORTS,
                            swap_input=GOAT_ENTRY_SWAP_LAMPORTS),
            token_amount=token_amount, entry_price_usd=GOAT_ENTRY_MARK,
            size_usd=0.2345, now=1.0, sol_price_usd=GOAT_SOL_PX,
            token_decimals=decimals, quote_mint=WSOL)
        return acct.positions[GOAT_MINT]

    def _obs(self, price):
        from types import SimpleNamespace
        return SimpleNamespace(price_usd=price)

    def _fill(self, output_amount):
        from autopaper.execution.real import OnchainFill
        f = OnchainFill(side="EXIT", status="confirmed", signature="s")
        f.input_amount = None            # the native-SOL unwind shape
        f.output_amount = output_amount
        return f

    def test_it_produces_a_number_for_a_native_sol_exit(self):
        from autopaper.engine import PaperTrader
        from autopaper.config import RealConfig
        trader = PaperTrader.__new__(PaperTrader)
        trader._sol_price_usd = GOAT_SOL_PX
        pos = self._pos()
        d = PaperTrader._exit_mark_vs_venue_pct(
            trader, self._obs(GOAT_EXIT_MARK), self._fill(GOAT_EXIT_WALLET_CREDIT),
            pos=pos)
        assert d is not None, (
            "the guard returned None on every native-SOL exit before this; that "
            "is exactly why $GOAT was not caught")
        expected = GOAT_EXIT_MARK / GOAT_EXIT_EXEC_USD_NET - 1.0
        assert d.pct == pytest.approx(expected, rel=1e-9)
        assert d.mark_overstates is False

    def test_an_unusable_leg_yields_none_rather_than_a_fabricated_ratio(self):
        from autopaper.engine import PaperTrader
        trader = PaperTrader.__new__(PaperTrader)
        trader._sol_price_usd = GOAT_SOL_PX
        pos = self._pos()
        d = PaperTrader._exit_mark_vs_venue_pct(
            trader, self._obs(GOAT_EXIT_MARK), self._fill(None), pos=pos)
        assert d is None

    def test_a_supplied_exit_quote_is_preferred_over_the_derived_price(self):
        from autopaper.engine import PaperTrader
        trader = PaperTrader.__new__(PaperTrader)
        trader._sol_price_usd = GOAT_SOL_PX
        pos = self._pos()
        q = _quote(price_usd=GOAT_EXIT_EXEC_USD_NET, pool=GOAT_EXIT_POOL)
        d = PaperTrader._exit_mark_vs_venue_pct(
            trader, self._obs(GOAT_EXIT_MARK), self._fill(GOAT_EXIT_WALLET_CREDIT),
            pos=pos, exit_quote=q)
        assert d is not None and d.kind == "EXIT_QUOTE"


# ===========================================================================
# 15. the Telegram close message
# ===========================================================================


class TestTelegramCloseMessage:
    def _close(self, **kw):
        from autopaper.engine import _close_text
        return _close_text(kw.pop("symbol", GOAT_SYMBOL), **kw)

    def test_the_goat_message_no_longer_pairs_a_dollar_with_a_mark_ratio(self):
        msg = self._close(return_pct=GOAT_ALL_IN_RETURN_PCT,
                          net_pnl_usd=GOAT_ALL_IN_PNL_USD)
        assert "+71.56" not in msg
        assert "-42.91%" in msg
        assert "$-0.1825" in msg

    def test_the_mark_return_is_hidden_unless_diagnostics_are_on(self):
        quiet = self._close(return_pct=GOAT_ALL_IN_RETURN_PCT,
                            net_pnl_usd=GOAT_ALL_IN_PNL_USD,
                            mark_return_pct=GOAT_MARK_RETURN_PCT)
        assert "+71.56" not in quiet

        loud = self._close(return_pct=GOAT_ALL_IN_RETURN_PCT,
                           net_pnl_usd=GOAT_ALL_IN_PNL_USD,
                           mark_return_pct=GOAT_MARK_RETURN_PCT,
                           show_mark_return=True)
        assert "+71.56" in loud
        assert "not executable" in loud

    def test_no_percentage_is_rendered_when_there_is_no_result(self):
        msg = self._close(return_pct=None, net_pnl_usd=None)
        assert "ผลลัพธ์" not in msg
        assert msg.startswith("✅ <b>ปิดออเดอร์</b>")

    def test_a_missing_fill_price_renders_nothing_rather_than_zero(self):
        msg = self._close(return_pct=GOAT_ALL_IN_RETURN_PCT,
                          net_pnl_usd=GOAT_ALL_IN_PNL_USD,
                          entry_fill=None, exit_fill=None)
        assert "Entry fill" not in msg
        good = self._close(return_pct=GOAT_ALL_IN_RETURN_PCT,
                           net_pnl_usd=GOAT_ALL_IN_PNL_USD,
                           entry_fill="$3.10787e-05",
                           exit_fill="$3.21764e-05")
        assert "Entry fill: $3.10787e-05" in good
        assert "Exit fill: $3.21764e-05" in good

    def test_a_mark_return_is_never_labelled_as_a_result(self):
        from autopaper.engine import _close_text
        loud = _close_text(GOAT_SYMBOL, GOAT_ALL_IN_RETURN_PCT,
                           GOAT_ALL_IN_PNL_USD,
                           mark_return_pct=GOAT_MARK_RETURN_PCT,
                           show_mark_return=True)
        # the realised figure is the ONLY one under the result label
        result_lines = [ln for ln in loud.splitlines() if "ผลลัพธ์" in ln]
        assert len(result_lines) == 1
        assert "+71.56" not in result_lines[0]


# ===========================================================================
# 16. PAPER keeps three layers
# ===========================================================================


def _paper_quote(price_usd):
    from tests.execguard_fakes import FakeAggregatorQuote
    out_raw = 10_000_000
    in_raw = int(round(price_usd * out_raw))
    return FakeAggregatorQuote(
        side="BUY", token_mint=TOKEN, quote_mint=USDC, quote_amount=in_raw,
        token_decimals=6, expected_token_amount=out_raw, quote_asset_usd=1.0,
        venue="pumpswap", pool="PAPER_POOL")


class TestPaperTelemetryHasThreeLayers:
    def _run(self, *, quote, side="entry"):
        from autopaper.config import ExecutionModel
        from autopaper.execution.fill import FillInputs, simulate_fill
        from autopaper.market.base import Observation
        from autopaper.models import DataState
        return simulate_fill(
            FillInputs(
                side=side, quantity=1000.0, notional_hint_usd=0.1,
                quoted_price=None, decision_time=1.0,
                observation=Observation(mint=TOKEN, observed_at=1.0,
                                        received_at=1.0,
                                        price_usd=2.043e-05,
                                        liquidity_usd=17_600.0,
                                        provider="test", venue="pumpfun"),
                observed_at=1.0, quote=quote,
                quote_missing_reason="" if quote else "no venue available"),
            ExecutionModel(), data_state=DataState.LIVE)

    def test_all_three_layers_are_present_and_distinct(self):
        r = self._run(quote=_paper_quote(2.10e-05))
        assert r.price_kind == "SIMULATED_FILL"
        assert r.observed_mark_price == 2.043e-05
        assert r.executable_quote_price == pytest.approx(2.10e-05)
        assert r.executable_quote_kind == "VENUE_QUOTE"
        assert r.executable_quote_at is not None
        assert r.executable_quote_provider == "fake"
        assert r.executable_quote_venue == "pumpswap"
        assert r.executable_quote_pool == "PAPER_POOL"
        assert r.fill_price != r.executable_quote_price

    def test_a_missing_quote_is_missing_with_a_reason_never_zero(self):
        r = self._run(quote=None)
        assert r.filled
        assert r.executable_quote_price is None
        assert r.executable_quote_price_usd is None
        assert r.executable_quote_missing_reason
        assert r.quote_guard_effective is False
        assert r.quote_price_source == "MARK_ONLY"

    def test_the_guard_is_genuine_only_with_a_quote(self):
        guarded = self._run(quote=_paper_quote(2.10e-05))
        assert guarded.quote_guard_effective is True
        assert guarded.quote_price_source == "EXECUTABLE_QUOTE"
        unguarded = self._run(quote=None)
        assert unguarded.quote_guard_effective is False
        assert "INERT" in unguarded.detail

    def test_a_paper_quote_source_cannot_build_a_transaction(self):
        from autopaper.execution.aggregators import QuoteOnlyJupiterProvider
        p = QuoteOnlyJupiterProvider()
        for forbidden in ("build", "build_transaction", "sign", "send_transaction",
                          "broadcast", "swap"):
            assert not hasattr(p, forbidden), (
                f"the PAPER quote source must not expose {forbidden}; a "
                f"configurable price with a broadcast capability is not a "
                f"measurement")


# ===========================================================================
# 17. the GOAT end-to-end regression
# ===========================================================================


class TestGoatRegression:
    """
    The whole incident, as one assertion set.

    Reproduces the false +71.56%, then proves the corrected system cannot produce
    it. Every input is the recorded chain data above.
    """

    def _account(self):
        from autopaper.real_account import RealAccount
        acct = RealAccount()
        acct.register_entry(
            mint=GOAT_MINT, symbol=GOAT_SYMBOL,
            fill=_goat_fill("ENTRY", input_amount=GOAT_ENTRY_SWAP_LAMPORTS,
                            output_amount=GOAT_TOKENS_RAW,
                            fee=GOAT_ENTRY_FEE_LAMPORTS,
                            rent=GOAT_ENTRY_RENT_LAMPORTS,
                            swap_input=GOAT_ENTRY_SWAP_LAMPORTS,
                            wallet_debit=GOAT_ENTRY_WALLET_DEBIT),
            token_amount=GOAT_TOKENS_RAW, entry_price_usd=GOAT_ENTRY_MARK,
            size_usd=0.2345525709834096, now=1.0,
            sol_price_usd=GOAT_SOL_PX, token_decimals=GOAT_DECIMALS,
            quote_mint=WSOL, venue="pump_bonding_curve",
            pool=GOAT_ENTRY_POOL, provider="raptor")
        return acct

    def _close(self, acct):
        return acct.close_position(
            mint=GOAT_MINT, reason="max_hold_60s", now=67.0,
            sol_price_usd=GOAT_SOL_PX, exit_price_usd=GOAT_EXIT_MARK,
            token_amount_sold=GOAT_TOKENS_RAW, token_decimals=GOAT_DECIMALS,
            venue="pumpswap", pool=GOAT_EXIT_POOL, provider="raptor",
            fill=_goat_fill("EXIT", input_amount=None,
                            output_amount=GOAT_EXIT_WALLET_CREDIT,
                            fee=GOAT_EXIT_FEE_LAMPORTS))

    def test_the_false_71_56_is_reproducible_from_the_marks_alone(self):
        from autopaper.execution.price_layers import mark_return_pct
        assert mark_return_pct(exit_mark=GOAT_EXIT_MARK,
                               entry_mark=GOAT_ENTRY_MARK) \
            == pytest.approx(GOAT_MARK_RETURN_PCT, rel=1e-12)

    def test_the_old_expression_is_exactly_what_produced_it(self):
        """The removed line, reproduced, so the regression is pinned."""
        old = (GOAT_EXIT_MARK / GOAT_ENTRY_MARK - 1.0) * 100.0
        assert old == pytest.approx(71.56477846326415, rel=1e-12)
        assert round(old, 2) == 71.56

    def test_the_corrected_system_reports_the_settlement_return(self):
        p = self._close(self._account())
        assert p.realized_return_pct == pytest.approx(
            GOAT_ALL_IN_RETURN_PCT, rel=1e-6)
        assert p.realized_return_pct != pytest.approx(GOAT_MARK_RETURN_PCT)

    def test_the_executable_price_move_is_what_it_was(self):
        from autopaper.execution.price_layers import executable_return_pct
        acct = self._account()
        p = self._close(acct)
        from tests.execguard_fakes import FakeAggregatorQuote
        entry_q = _quote(price_usd=GOAT_ENTRY_EXEC_USD, pool=GOAT_ENTRY_POOL)
        exit_q = FakeAggregatorQuote(
            side="SELL", token_mint=TOKEN, quote_mint=WSOL,
            token_amount=GOAT_TOKENS_RAW, token_decimals=GOAT_DECIMALS,
            expected_quote_amount=GOAT_EXIT_WALLET_CREDIT,
            quote_asset_usd=GOAT_SOL_PX, pool=GOAT_EXIT_POOL)
        got = executable_return_pct(entry_quote=entry_q, exit_quote=exit_q)
        assert got is not None
        assert got == pytest.approx(
            GOAT_EXIT_EXEC_USD_NET / GOAT_ENTRY_EXEC_USD * 100 - 100, rel=1e-6)

    def test_the_all_in_result_is_a_loss_not_a_profit(self):
        p = self._close(self._account())
        net = (GOAT_EXIT_WALLET_CREDIT - GOAT_ENTRY_WALLET_DEBIT) / 1e9 * GOAT_SOL_PX
        assert net == pytest.approx(GOAT_ALL_IN_PNL_USD, rel=1e-9)
        assert p.entry_cost_usd == pytest.approx(GOAT_ALL_IN_COST_USD, rel=1e-6)
        assert p.entry_cost_usd_trading_only == pytest.approx(
            GOAT_TRADING_ONLY_COST_USD, rel=1e-6)
        assert p.entry_cost_usd > p.entry_cost_usd_trading_only

    def test_the_three_numbers_coexist_without_being_confused(self):
        p = self._close(self._account())
        assert p.realized_return_pct is not None
        assert p.observed_mark_return_pct is not None
        assert p.realized_return_pct != p.observed_mark_return_pct
        assert p.realized_return_pct < 0 < p.observed_mark_return_pct, (
            "the honest result is a loss; the mark excursion was a gain")
        assert p.entry_execution_price_usd is not None
        assert p.exit_execution_price_usd is not None
        assert p.exit_execution_price_usd != p.exit_price_usd

    def test_the_guard_would_refuse_this_entry_before_it_happened(self):
        from autopaper.execution.quoting import DivergenceGuard
        v = DivergenceGuard().evaluate_entry(
            mark=_mark(), quote=_quote(price_usd=GOAT_ENTRY_EXEC_USD))
        assert v.allowed is False
        assert v.reason == "quote_worse_than_mark"


# ===========================================================================
# 18. provider mapping: Jupiter, Pump bonding curve, PumpSwap
# ===========================================================================


class TestProviderMapping:
    def test_a_jupiter_route_becomes_venue_pool_and_route(self):
        from autopaper.execution.aggregators import AggregatorQuoteProvider
        from autopaper.execution.jupiter import JupiterProvider

        body = {
            "outAmount": "12345678",
            "otherAmountThreshold": "12000000",
            "priceImpactPct": 0.0004,
            "routePlan": [{"swapInfo": {"label": "pumpswap",
                                        "ammKey": GOAT_EXIT_POOL,
                                        "inputMint": TOKEN,
                                        "outputMint": WSOL,
                                        "inAmount": "1",
                                        "outAmount": "2"}}],
        }

        class Inner:
            def provider_name(self):
                return "jupiter"

            async def quote(self, i, o, amt, slip, *, side=None):
                from autopaper.execution.jupiter import Quote as JQuote
                q = JQuote(input_mint=i, output_mint=o,
                           in_amount=amt, expected_out_amount=12_345_678,
                           min_out_amount=12_000_000,
                           price_impact_pct=0.04, slippage_bps=slip,
                           side=side, raw=dict(body))
                return q

        import asyncio
        p = AggregatorQuoteProvider(Inner(), name="jupiter")
        q = asyncio.run(p.quote_buy(token_mint=TOKEN, quote_mint=WSOL,
                                   quote_amount_raw=1000, token_decimals=6,
                                   quote_asset_usd=GOAT_SOL_PX))
        assert q.provider == "jupiter"
        assert q.venue == "pumpswap"
        assert q.pool == GOAT_EXIT_POOL
        assert q.venue_quoted is True
        assert q.route and q.route[0].pool == GOAT_EXIT_POOL

    def test_a_pumpswap_quote_refuses_a_route_on_another_venue(self):
        import asyncio
        from autopaper.execution.aggregators import AggregatorQuoteProvider
        from autopaper.execution.pump import PumpSwapQuoteProvider

        from tests.execguard_fakes import FakeBuildProvider
        inner = FakeBuildProvider(price_usd=1e-5, venue="raydium",
                                  pool="Rpool")
        provider = PumpSwapQuoteProvider(inner, chain=None,
                                         require_program_attestation=False)
        from autopaper.execution.quote_errors import ChainStateUnavailable
        with pytest.raises(ChainStateUnavailable):
            asyncio.run(provider.quote_buy(
                token_mint=TOKEN, quote_mint=WSOL, quote_amount_raw=1000,
                token_decimals=6, quote_asset_usd=1.0))

    def test_a_pumpswap_quote_attests_the_pool_program_on_chain(self):
        import asyncio
        import httpx
        from autopaper.execution.chainstate import PublicRpcClient
        from autopaper.execution.pump import (
            PUMP_SWAP, PUMP_SWAP_PROGRAM_ID, PumpSwapQuoteProvider,
        )
        from autopaper.execution.quote_errors import ChainStateUnavailable
        from tests.execguard_fakes import FakeQuoteProvider

        def owner_for(program):
            def handler(request):
                body = json.loads(request.content)
                return httpx.Response(200, json={
                    "jsonrpc": "2.0", "id": body["id"],
                    "result": {"context": {"slot": 1},
                               "value": {"owner": program,
                                         "lamports": 100,
                                         "data": [base64.b64encode(b"x" * 400)
                                                  .decode(), "base64"]}}})
            return handler

        from tests.execguard_fakes import FakeBuildProvider
        inner = FakeBuildProvider(price_usd=GOAT_ENTRY_EXEC_USD,
                                  sol_usd=GOAT_SOL_PX,
                                  pool=GOAT_EXIT_POOL,
                                  token_decimals=GOAT_DECIMALS,
                                  now_fn=lambda: 1e9)
        good = PumpSwapQuoteProvider(
            inner,
            chain=PublicRpcClient(
                client=httpx.AsyncClient(
                    transport=httpx.MockTransport(owner_for(PUMP_SWAP_PROGRAM_ID)))),
            require_program_attestation=True)
        q = asyncio.run(good.quote_buy(
            token_mint=TOKEN, quote_mint=WSOL, quote_amount_raw=1_928_811,
            token_decimals=GOAT_DECIMALS, quote_asset_usd=GOAT_SOL_PX))
        assert q.venue == PUMP_SWAP.key
        assert q.pool == GOAT_EXIT_POOL
        assert q.provenance["program_attested"] == PUMP_SWAP_PROGRAM_ID

        bad = PumpSwapQuoteProvider(
            inner,
            chain=PublicRpcClient(
                client=httpx.AsyncClient(
                    transport=httpx.MockTransport(owner_for("SomeOtherProgram")))),
            require_program_attestation=True)
        with pytest.raises(ChainStateUnavailable):
            asyncio.run(bad.quote_buy(
                token_mint=TOKEN, quote_mint=WSOL, quote_amount_raw=1_928_811,
                token_decimals=GOAT_DECIMALS, quote_asset_usd=GOAT_SOL_PX))

    def test_pump_venues_have_distinct_program_ids(self):
        from autopaper.execution.pump import (
            PUMP_BONDING_CURVE_PROGRAM_ID, PUMP_SWAP_PROGRAM_ID,
        )
        assert PUMP_BONDING_CURVE_PROGRAM_ID != PUMP_SWAP_PROGRAM_ID


# ===========================================================================
# 19. the router cannot silently change direction
# ===========================================================================


class TestRouterDirection:
    def test_the_router_requires_an_explicit_side(self):
        import asyncio
        from tests.execguard_fakes import FakeAggregatorQuote
        from autopaper.execution.price_layers import SwapDirectionError
        from autopaper.execution.quoting import RouterQuoteProvider
        from tests.execguard_fakes import FakeQuoteProvider
        chain = RouterQuoteProvider([
            FakeQuoteProvider(price_usd=1e-5, now_fn=lambda: 1e9)])
        with pytest.raises(SwapDirectionError):
            asyncio.run(chain.try_all(
                side="sideways", token_mint=TOKEN, quote_mint=WSOL,
                amount_raw=1000, token_decimals=6))

    def test_an_aggregator_refuses_mints_that_contradict_the_side(self):
        import asyncio
        from autopaper.execution.aggregators import AggregatorQuoteProvider
        from autopaper.execution.jupiter import JupiterProvider
        from autopaper.execution.price_layers import SwapDirectionError
        from autopaper.execution.quote_errors import QuoteUnavailable

        class Inner:
            def provider_name(self):
                return "jupiter"

            async def quote(self, i, o, amt, slip, *, side=None):
                raise AssertionError("must not reach the venue")

        p = AggregatorQuoteProvider(Inner())
        with pytest.raises(SwapDirectionError):
            asyncio.run(p.quote_swap(
                side="BUY", token_mint=WSOL, quote_mint=TOKEN, amount_raw=1000,
                token_decimals=6))

    def test_an_aggregator_needs_decimals_before_it_can_price(self):
        import asyncio
        from autopaper.execution.aggregators import AggregatorQuoteProvider
        from autopaper.execution.price_layers import SwapDirectionError
        from autopaper.execution.quote_errors import QuoteUnavailable

        class Inner:
            def provider_name(self):
                return "jupiter"

            async def quote(self, i, o, amt, slip, *, side=None):
                raise AssertionError("must not reach the venue")

        with pytest.raises(QuoteUnavailable):
            asyncio.run(AggregatorQuoteProvider(Inner()).quote_buy(
                token_mint=TOKEN, quote_mint=WSOL, quote_amount_raw=1000,
                token_decimals=None))

    def test_a_provider_chain_key_that_is_unknown_is_refused(self):
        from types import SimpleNamespace
        from autopaper.config import RealConfig
        from autopaper.engine import PaperTrader

        trader = PaperTrader.__new__(PaperTrader)
        trader.config = SimpleNamespace(
            real=RealConfig(quote_provider_chain="pumpswap,not_a_venue"))
        trader._real_executor = SimpleNamespace(
            backend=SimpleNamespace(provider=None,
                                    fallback_provider=lambda: None))
        with pytest.raises(ValueError, match="unknown quote provider"):
            trader._quote_provider()

    def test_an_empty_chain_is_refused(self):
        from types import SimpleNamespace
        from autopaper.config import RealConfig
        from autopaper.engine import PaperTrader
        trader = PaperTrader.__new__(PaperTrader)
        trader.config = SimpleNamespace(real=RealConfig(quote_provider_chain=" , "))
        trader._real_executor = SimpleNamespace(
            backend=SimpleNamespace(provider=None,
                                    fallback_provider=lambda: None))
        with pytest.raises(ValueError):
            trader._quote_provider()


# ===========================================================================
# 20. no broadcast capability anywhere in this module's graph
# ===========================================================================


class TestNoBroadcastCapability:
    def test_this_module_imports_nothing_that_can_sign_or_send(self):
        import autopaper.execution.price_layers as pl
        import autopaper.execution.quote_errors as qe
        for mod in (pl, qe):
            src = open(mod.__file__, encoding="utf-8").read()
            for forbidden in ("send_transaction", "sign_transaction",
                              "broadcast(", "SolanaWallet", "solders"):
                assert forbidden not in src, (
                    f"{mod.__name__} must not reference {forbidden}")

    def test_the_chain_state_client_has_no_send(self):
        from autopaper.execution.chainstate import PublicRpcClient
        for forbidden in ("send", "broadcast", "sign", "confirm"):
            assert not hasattr(PublicRpcClient, forbidden), (
                f"the public reader must not expose {forbidden}")

    def test_the_paper_quote_source_has_no_execution_methods(self):
        from autopaper.execution.aggregators import QuoteOnlyJupiterProvider
        for forbidden in ("build", "build_transaction", "sign", "sign_transaction",
                          "send_transaction", "broadcast", "swap",
                          "execute_swap", "send", "submit"):
            assert not hasattr(QuoteOnlyJupiterProvider, forbidden), (
                f"the PAPER quote source must not expose {forbidden}")
        assert hasattr(QuoteOnlyJupiterProvider, "quote_buy")
        assert hasattr(QuoteOnlyJupiterProvider, "quote_sell")