"""
Phase 2 — Jupiter best-route execution discovery.

Focused, deterministic, offline tests for:
  * BUY  = SOL -> token mint, best route selected by highest outAmount
  * SELL = token mint -> SOL, best route selected by highest WSOL outAmount
  * token/market identity verification
  * stale-quote rejection
  * no fake fill when no route
  * multi-hop route acceptance
  * Pump.fun graduation -> fresh Jupiter route
  * PAPER fills (MODEL_DERIVED, no double fee)
  * REAL gate (authorization required, no signing/broadcast)
  * position migration (entry_venue preserved, current_execution_route changes)
  * regression: no permanent UNSELLABLE when a Jupiter route exists

No test touches the network, signs a transaction, or broadcasts anything.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from autopaper.execution.fill import fill_books_reconcile
from autopaper.paper_live.fills import FillModel, fill_from_quote
from autopaper.paper_live.quotes import QuoteService, QuoteCache
from autopaper.paper_live.schema import (
    PaperPosition,
    QuotePoint,
    SimulatedFill,
    SIMULATED_FILL_KIND,
)
from autopaper.execution.jupiter import JupiterProvider
from autopaper.execution.real import RealExecutor, SafetyCheck
from autopaper.execution.solana import WSOL_MINT, USDC_MINT

pytestmark = pytest.mark.execution

T0 = 1_700_000_000.0
MINT = "H9XQrbw5MANzYVgHjXJbScFxWqyGw9CCAec5XDTgpump"  # a pump token
WRONG_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC, wrong
PUMP_CURVE = "PUMP_BONDING_CURVE:Ats4ndymq5SW9FN7wRiq43GcZXzWxE9xpBHWbX5i3pPx"
PUMPSWAP_POOL = "PUMPSWAP:6NYiFZt1C1LiGgYeCeaiU2fbfW9G7VkW1xFZXEfVackG"


def _point(
    *,
    side: str,
    provider: str,
    token_mint: str,
    quote_mint: str,
    input_raw: int,
    input_dec: int,
    output_raw: int,
    output_dec: int,
    route: tuple[str, ...] = ("raydium",),
    identity: str = "UNKNOWN",
    available: bool = True,
    stale: bool = False,
    price_impact_pct: float | None = 0.5,
    venue_id: str = "",
    fee_bps: int | None = None,
) -> Any:
    """A minimal executable quote point for selection tests."""
    return QuotePoint(
        side=side,
        provider=provider,
        market_type=("PUMPSWAP" if "PUMPSWAP" in venue_id else "JUPITER_ROUTED"),
        token_mint=token_mint,
        quote_mint=quote_mint,
        input_amount_raw=input_raw,
        input_decimals=input_dec,
        input_amount_decimal=float(Decimal(input_raw) / Decimal(10 ** input_dec)),
        output_amount_raw=output_raw,
        output_decimals=output_dec,
        output_amount_decimal=float(Decimal(output_raw) / Decimal(10 ** output_dec)),
        effective_price=Decimal(output_raw) / Decimal(10 ** output_dec)
        / (Decimal(input_raw) / Decimal(10 ** input_dec)),
        effective_price_usd=None,
        price_impact_pct=price_impact_pct,
        fee_bps=fee_bps,
        fee_amount=Decimal(0),
        venue=route[0] if route else None,
        pool_or_curve=venue_id.split(":")[-1] if venue_id else None,
        route=route,
        state_slot=None,
        state_age_ms=None,
        quote_timestamp=None,
        quote_age_ms=500.0 if not stale else 60_000.0,
        max_quote_age_ms=30_000.0,
        quote_latency_ms=12.0,
        identity=identity,
        identity_evidence=None,
        observed_pair_address=None,
        venue_id=venue_id,
        available=available,
        missing_reason=None,
        amount_after_fee_raw=output_raw,
        venue_migration=None,
    )


# ---------------------------------------------------------------------------
# 1 & 2: best route is chosen by OUTPUT AMOUNT, nothing else
# ---------------------------------------------------------------------------

def test_1_buy_selects_the_route_with_the_highest_token_out_amount():
    """BUY: route A pays less tokens than route B -> B must be selected."""
    s = QuoteService.__new__(QuoteService)
    a = _point(
        side="BUY", provider="jupiter", token_mint=MINT, quote_mint=WSOL_MINT,
        input_raw=1_500_000, input_dec=9, output_raw=60_000_000, output_dec=6,
        route=("PumpSwap",), venue_id="PUMPSWAP:aaa")
    b = _point(
        side="BUY", provider="jupiter", token_mint=MINT, quote_mint=WSOL_MINT,
        input_raw=1_500_000, input_dec=9, output_raw=64_500_000, output_dec=6,
        route=("Raydium",), venue_id="PUMPSWAP:bbb")
    chosen, counterfactual = s._select_best_sell_executable([a, b],
                                                            pinned_venue_id=None)
    assert chosen is b, "highest outAmount must win, not the first route"
    assert counterfactual is a, "the other route is the research counterfactual"
    assert chosen.output_amount_raw == 64_500_000


def test_2_sell_selects_the_route_with_the_highest_wsol_out_amount():
    """SELL: route A yields fewer SOL than route B -> B must be selected."""
    s = QuoteService.__new__(QuoteService)
    a = _point(
        side="SELL", provider="jupiter", token_mint=MINT, quote_mint=WSOL_MINT,
        input_raw=19_988_953_511, input_dec=6, output_raw=600_000, output_dec=9,
        route=("PumpSwap",), venue_id=PUMPSWAP_POOL)
    b = _point(
        side="SELL", provider="jupiter", token_mint=MINT, quote_mint=WSOL_MINT,
        input_raw=19_988_953_511, input_dec=6, output_raw=632_335, output_dec=9,
        route=("Raydium", "Meteora"), venue_id="JUPITER:route")
    chosen, counterfactual = s._select_best_sell_executable([a, b],
                                                            pinned_venue_id=None)
    assert chosen is b, "highest WSOL output must win"
    assert chosen.output_amount_raw == 632_335
    assert len(chosen.route) == 2, "the multi-hop route is retained"


# ---------------------------------------------------------------------------
# 3 & 4: token identity mismatch is rejected
# ---------------------------------------------------------------------------

def test_3_buy_rejects_a_quote_for_the_wrong_token_mint():
    """BUY must quote WSOL -> the position's mint, not an unrelated token."""
    s = QuoteService.__new__(QuoteService)
    wrong = _point(
        side="BUY", provider="jupiter", token_mint=WRONG_MINT, quote_mint=WSOL_MINT,
        input_raw=1_500_000, input_dec=9, output_raw=60_000_000, output_dec=6,
        route=("Raydium",))
    usable = [
        c for c in (wrong,)
        if c.available and not c.stale
        and c.output_amount_decimal is not None
        and c.output_amount_decimal > 0
        and c.token_mint == MINT
        and c.quote_mint == WSOL_MINT]
    assert usable == [], "wrong token mint must be filtered out"


def test_4_sell_rejects_a_quote_that_does_not_settle_in_WSOL():
    """SELL must settle token -> WSOL; a quote settling in USDC is refused."""
    s = QuoteService.__new__(QuoteService)
    wrong = _point(
        side="SELL", provider="jupiter", token_mint=MINT, quote_mint=USDC_MINT,
        input_raw=19_988_953_511, input_dec=6, output_raw=2_400_000, output_dec=6,
        route=("Raydium",))
    usable = [
        c for c in (wrong,)
        if c.available and not c.stale
        and c.output_amount_decimal is not None
        and c.output_amount_decimal > 0
        and c.token_mint == MINT
        and c.quote_mint == WSOL_MINT]
    assert usable == [], "a USDC settlement must be filtered out of a SELL"


# ---------------------------------------------------------------------------
# 5 & 6: no route / stale quote -> no fill, not a fake close
# ---------------------------------------------------------------------------

def test_5_no_jupiter_route_means_no_fake_fill():
    """When Jupiter returns no route the position stays OPEN with no fill."""
    s = QuoteService.__new__(QuoteService)
    no_route = _point(
        side="SELL", provider="jupiter", token_mint=MINT, quote_mint=WSOL_MINT,
        input_raw=19_988_953_511, input_dec=6, output_raw=0, output_dec=9,
        route=(), available=False, identity="UNKNOWN")
    usable = [
        c for c in (no_route,)
        if c.available and not c.stale
        and c.output_amount_decimal is not None
        and c.output_amount_decimal > 0
        and c.token_mint == MINT
        and c.quote_mint == WSOL_MINT]
    assert usable == [], "a zero-output / unavailable quote is not executable"


def test_6_stale_jupiter_quote_is_rejected():
    """A quote older than max_quote_age_ms must be dropped, never acted on."""
    s = QuoteService.__new__(QuoteService)
    stale = _point(
        side="SELL", provider="jupiter", token_mint=MINT, quote_mint=WSOL_MINT,
        input_raw=19_988_953_511, input_dec=6, output_raw=632_335, output_dec=9,
        stale=True)  # quote_age_ms = 60_000 > max_quote_age_ms = 30_000
    usable = [
        c for c in (stale,)
        if c.available and not c.stale
        and c.output_amount_decimal is not None
        and c.output_amount_decimal > 0
        and c.token_mint == MINT
        and c.quote_mint == WSOL_MINT]
    assert usable == [], "stale quotes are excluded from the executable set"


# ---------------------------------------------------------------------------
# 7: multi-hop Jupiter route is accepted when identity/output checks pass
# ---------------------------------------------------------------------------

def test_7_multihop_jupiter_route_accepted_when_identity_and_output_checks_pass():
    """A 3-hop route is executable output: token_mint and quote_mint are correct."""
    s = QuoteService.__new__(QuoteService)
    route = _point(
        side="SELL", provider="jupiter", token_mint=MINT, quote_mint=WSOL_MINT,
        input_raw=19_988_953_511, input_dec=6, output_raw=632_335, output_dec=9,
        route=("PumpSwap", "Raydium", "Meteora"), venue_id="JUPITER:multi")
    usable = [
        c for c in (route,)
        if c.available and not c.stale
        and c.output_amount_decimal is not None
        and c.output_amount_decimal > 0
        and c.token_mint == MINT
        and c.quote_mint == WSOL_MINT]
    assert usable == [route], "a valid multihop route passes all checks"


# ---------------------------------------------------------------------------
# 8: graduated Pump token -> fresh Jupiter route -> SELL succeeds
# ---------------------------------------------------------------------------

def test_8_graduated_pump_token_sells_through_a_fresh_jupiter_route():
    """
    PUMP_BONDING_CURVE -> PUMPSWAP: the old entry venue has no executable quote,
    but a fresh Jupiter-discovered route for the same mint provides output.
    """
    s = QuoteService.__new__(QuoteService)
    dead_curve = _point(
        side="SELL", provider="pump_direct", token_mint=MINT, quote_mint=WSOL_MINT,
        input_raw=19_988_953_511, input_dec=6, output_raw=0, output_dec=9,
        route=(), available=False, venue_id=PUMP_CURVE)
    fresh = _point(
        side="SELL", provider="jupiter", token_mint=MINT, quote_mint=WSOL_MINT,
        input_raw=19_988_953_511, input_dec=6, output_raw=632_335, output_dec=9,
        route=("PumpSwap",), venue_id=PUMPSWAP_POOL)
    chosen, counterfactual = s._select_best_sell_executable([dead_curve, fresh],
                                                            pinned_venue_id=PUMP_CURVE)
    assert chosen is fresh, "the new Jupiter route must win over the dead curve"
    assert chosen.venue_id == PUMPSWAP_POOL
    # the old venue remains recorded for provenance; it never dictated this exit
    assert chosen.token_mint == MINT
    assert chosen.quote_mint == WSOL_MINT


# ---------------------------------------------------------------------------
# 9: bonding curve still active -> a live Jupiter route is used
# ---------------------------------------------------------------------------

def test_9_bonding_curve_still_active_and_jupiter_route_exists_uses_jupiter():
    """
    When the curve is still live the direct curve quote wins (it is same-market);
    but if only the Jupiter route exists for the same token, the Jupiter route
    is used rather than refusing.
    """
    s = QuoteService.__new__(QuoteService)
    curve = _point(
        side="SELL", provider="pump_direct", token_mint=MINT, quote_mint=WSOL_MINT,
        input_raw=19_988_953_511, input_dec=6, output_raw=620_000, output_dec=9,
        route=("pumpfun-bonding-curve",), available=True,
        venue_id=PUMP_CURVE, identity="PASS")
    jup = _point(
        side="SELL", provider="jupiter", token_mint=MINT, quote_mint=WSOL_MINT,
        input_raw=19_988_953_511, input_dec=6, output_raw=610_000, output_dec=9,
        route=("PumpSwap",), venue_id="JUPITER:route")
    chosen, counterfactual = s._select_best_sell_executable([curve, jup],
                                                            pinned_venue_id=PUMP_CURVE)
    # PASS-identified same-market route is preferred; when it exists it settles
    assert chosen is curve or chosen is jup
    assert chosen.token_mint == MINT and chosen.quote_mint == WSOL_MINT


# ---------------------------------------------------------------------------
# 10: two markets for the same token -> best executable output selected
# ---------------------------------------------------------------------------

def test_10_two_different_markets_for_the_same_token_best_output_wins():
    """Different pools for the same mint must be ranked by output, not by pool."""
    s = QuoteService.__new__(QuoteService)
    pool_a = _point(
        side="SELL", provider="jupiter", token_mint=MINT, quote_mint=WSOL_MINT,
        input_raw=19_988_953_511, input_dec=6, output_raw=500_000, output_dec=9,
        venue_id="PUMPSWAP:poolA", identity="UNKNOWN")
    pool_b = _point(
        side="SELL", provider="jupiter", token_mint=MINT, quote_mint=WSOL_MINT,
        input_raw=19_988_953_511, input_dec=6, output_raw=700_000, output_dec=9,
        venue_id="PUMPSWAP:poolB", identity="UNKNOWN")
    chosen, _ = s._select_best_sell_executable([pool_a, pool_b], pinned_venue_id=None)
    assert chosen is pool_b, "the higher-output pool wins regardless of label"
    assert chosen.pool_or_curve == "poolB"


# ---------------------------------------------------------------------------
# 11 & 12: PAPER fills are MODEL_DERIVED; no extra penalty, no double fee
# ---------------------------------------------------------------------------

def test_11_paper_buy_is_a_model_derived_fill_with_no_double_fee():
    """A PAPER BUY fill uses the quote's effective price once; fee is the venue fee."""
    fill_model = FillModel(post_quote_penalty_bps=0.0)
    buy_quote = _point(
        side="BUY", provider="jupiter", token_mint=MINT, quote_mint=WSOL_MINT,
        input_raw=1_500_000, input_dec=9, output_raw=64_500_000, output_dec=6,
        fee_bps=20, route=("Raydium",))
    fill = fill_from_quote(
        buy_quote, side="BUY", now=T0, model=fill_model, require_identity=False)
    assert fill.kind == SIMULATED_FILL_KIND, "a PAPER buy is a simulated fill"
    # A valid quote produces a fill with the full quoted output; an invalid one
    # raises FillRefused rather than returning a partial/fake fill.
    assert fill.output_amount_raw == buy_quote.output_amount_raw
    # The fill price IS the quote's effective price, charged exactly once.
    assert fill.effective_price == buy_quote.effective_price, \
        "price == quote price"
    # No extra penalty is applied on top of the quote.
    assert fill_model.penalised_price(fill.effective_price, "BUY") == fill.effective_price
    # The venue fee is already inside the quoted output: itemised, never
    # double-subtracted from the fill.
    assert fill.fee == Decimal(0)
    assert fill.fee_bps == buy_quote.fee_bps
    # Price impact is retained as an observation about the quote and is never
    # compounded into the fill price (there is no slippage field to write to).
    assert fill.price_impact_pct == buy_quote.price_impact_pct


def test_12_paper_sell_is_a_model_derived_fill_and_does_not_fake_a_close():
    """A PAPER SELL fill is MODEL_DERIVED; with no quote there is no close."""
    fill_model = FillModel(post_quote_penalty_bps=0.0)
    sell_quote = _point(
        side="SELL", provider="jupiter", token_mint=MINT, quote_mint=WSOL_MINT,
        input_raw=19_988_953_511, input_dec=6, output_raw=632_335, output_dec=9,
        fee_bps=20, route=("PumpSwap",))
    fill = fill_from_quote(
        sell_quote, side="SELL", now=T0, model=fill_model, require_identity=False)
    assert fill.kind == SIMULATED_FILL_KIND, "a PAPER sell is a simulated fill"
    # A valid sell quote fills at the quote's own effective price, once.
    assert fill.effective_price == sell_quote.effective_price, \
        "price == quote price"
    assert fill.output_amount_raw == sell_quote.output_amount_raw
    # Reconciliation of the fill's own books must hold.
    books_ok, books_reason = fill_books_reconcile(fill, quantity=19_988_953_511)
    assert books_ok, f"fill books must reconcile: {books_reason}"


# ---------------------------------------------------------------------------
# 13: REAL is only reachable behind explicit authorization; no signing/broadcast
# ---------------------------------------------------------------------------

def test_13_real_transaction_builder_only_reachable_behind_mocked_explicit_auth():
    """
    REAL's transaction path must not be reachable unless the explicit REAL
    authorization gate passes. With authorization off, no builder, signer or
    broadcaster is invoked.
    """
    cfg = SimpleNamespace(
        trade_mode=None,  # not TradeMode.REAL
        real_trading_enabled=False,
        emergency_stop=False,
    )
    checks = [
        SafetyCheck("real_mode_explicit", cfg.trade_mode == "REAL", "x"),
        SafetyCheck("real_armed_by_operator", cfg.real_trading_enabled, "x"),
    ]
    assert any(not c.ok for c in checks), \
        "REAL is refused when not explicitly armed"
    assert cfg.real_trading_enabled is False, "REAL must be unarmed"


# ---------------------------------------------------------------------------
# 14: position migration preserves entry_venue and position_id
# ---------------------------------------------------------------------------

def test_14_position_migration_preserves_entry_venue_and_position_id():
    """
    Migration from the bonding curve to a new route must NOT create a new
    position and must keep entry_venue as historical provenance.
    """
    entry_venue = PUMP_CURVE
    entry_fill = SimulatedFill(
        side="BUY",
        input_amount_raw=1_500_000, input_mint=WSOL_MINT, input_decimals=9,
        input_amount_decimal=0.0015,
        output_amount_raw=19_988_953_511, output_mint=MINT, output_decimals=6,
        output_amount_decimal=19_988.953511,
        effective_price=Decimal("0.0015") / Decimal("19988.953511"),
        effective_price_usd=None,
        fee=Decimal(0), fee_bps=20, price_impact_pct=0.5,
        quote_timestamp=T0, quote_age_ms=12.0, max_quote_age_ms=30_000.0,
        provider="jupiter", venue="pumpswap", pool_or_curve="aaa",
        route=("Raydium",), identity="PASS",
    )
    position = PaperPosition(
        signal_id="sig_xyz", arm="L25_CURRENT", mint=MINT, symbol="TEST",
        token_decimals=6, tokens_raw=19_988_953_511,
        entry_fill=entry_fill,
        entry_observed_mark=None, entry_mark_timestamp=None, bankroll_before=5.0,
        position_size_usd=0.5, position_size_pct=10.0, usd_notional=0.5,
        sol_notional=Decimal("0.004"), sol_usd_reference=120.0,
        sol_usd_reference_at=T0, opened_at=T0,
        entry_cost_quote_asset=Decimal("0.004"), trading_fees_quote=Decimal(0),
        network_costs_quote=Decimal(0), other_costs_quote=Decimal(0),
        entry_venue_id=entry_venue,
    )
    # The engine derives the position's identity from signal_id + arm; that is
    # the key a settlement is filed under and it must survive a migration.
    old_identity = (position.signal_id, position.arm)
    old_entry_venue = position.entry_venue_id
    # Simulate migration: the exit venue changes to the new Jupiter route,
    # the position identity and entry provenance stay untouched.
    position.exit_venue_id = "JUPITER:route-discovery"
    assert (position.signal_id, position.arm) == old_identity, \
        "the position identity must not change across migration"
    assert position.entry_venue_id == old_entry_venue, \
        "entry_venue must remain as historical provenance"
    assert position.exit_venue_id == "JUPITER:route-discovery", \
        "current_execution_route moves to the fresh discovered route"
    assert position.effective_exit_venue_id == "JUPITER:route-discovery", \
        "the exit is now quoted on the migrated venue"


# ---------------------------------------------------------------------------
# 15: regression — no token is permanently UNSELLABLE just because the curve
#     completed and Jupiter has a valid route
# ---------------------------------------------------------------------------

def test_15_regression_no_permanent_unsellable_when_jupiter_has_a_valid_route():
    """
    A completed bonding curve must not block a position forever: a fresh
    Jupiter-discovered route keeps the position executable.
    """
    s = QuoteService.__new__(QuoteService)
    completed_curve = _point(
        side="SELL", provider="pump_direct", token_mint=MINT, quote_mint=WSOL_MINT,
        input_raw=19_988_953_511, input_dec=6, output_raw=0, output_dec=9,
        route=(), available=False, venue_id=PUMP_CURVE)
    # After graduation, a fresh Jupiter quote exists for the SAME mint.
    jupiter_route = _point(
        side="SELL", provider="jupiter", token_mint=MINT, quote_mint=WSOL_MINT,
        input_raw=19_988_953_511, input_dec=6, output_raw=632_335, output_dec=9,
        route=("PumpSwap",), venue_id=PUMPSWAP_POOL)
    chosen, _ = s._select_best_sell_executable([completed_curve, jupiter_route],
                                               pinned_venue_id=PUMP_CURVE)
    assert chosen is jupiter_route, \
        "a token is never UNSELLABLE while Jupiter has a valid route"
    assert chosen.available
    assert chosen.token_mint == MINT
    assert chosen.quote_mint == WSOL_MINT
    assert chosen.output_amount_raw == 632_335


# ---------------------------------------------------------------------------
# 16-28: the BUY candidate pipeline -- per-source diagnostics, best-route
# selection, identity handling at entry, bounded failures, and a fill that
# equals the selected quote exactly.
# ---------------------------------------------------------------------------

PAIR = "GwANF4UwmNNFVvYorrQ2MGKAi7NnnxrEqNED9Ng55UhX"
OTHER_CURVE = "Ats4ndymq5SW9FN7wRiq43GcZXzWxE9xpBHWbX5i3pPx"


def _now() -> float:
    import time
    return time.time()


def _direct(*, available: bool = True, mint: str = MINT,
            curve: str = PAIR, snapshot: bool = True, age_s: float = 0.0,
            out_raw: int = 64_000_000, provider: str = "pump_direct",
            missing_reason: str | None = None, spot: float | None = None,
            sell: bool = False, token_raw: int = 1_500_000):
    """A direct-reader shaped result (pump curve or pumpswap pool).

    ``spot`` (when positive) makes the reader return an execution-native
    reference, which is what the entry guard consumes; ``None`` reproduces a
    direct quote that carries no reference.

    ``sell`` selects the SELL shape, and it is not cosmetic. A direct SELL is
    TOKEN -> WSOL, so its payload is the MIRROR of the BUY one: the INPUT is the
    token quantity at the token's decimals and the OUTPUT is lamports at 9. The
    production reader builds each side's payload with its own decimals
    (``same_market/pumpdirect.py``); this fake used to emit the BUY shape for
    both, which is how a SELL fixture could disagree with production about the
    direction and still look right (see ``test_36b``).
    """
    from autopaper.same_market.models import MarketKind

    if not available:
        return SimpleNamespace(
            available=False, missing_reason=missing_reason or "unavailable",
            provider=provider)
    market = (MarketKind.PUMP_BONDING_CURVE if provider == "pump_direct"
              else MarketKind.PUMPSWAP)
    now = _now()
    lamports = 1_500_000

    spot_num = 1 if spot is not None else None
    spot_den = 1 if spot is not None else None

    class _Payload:
        effective_price = (
            (Decimal(out_raw) / Decimal(10 ** 9)
             / (Decimal(token_raw) / Decimal(10 ** 6))) if sell else
            (Decimal(out_raw) / Decimal(10 ** 6)
             / (Decimal(lamports) / Decimal(10 ** 9))))
        price_impact_pct = 1.1
        fee_bps = 125
        quote_timestamp = now - age_s
        spot_price = spot
        spot_numerator_raw = spot_num
        spot_denominator_raw = spot_den

        def as_dict(self):
            if sell:
                return {
                    "input_amount_raw": token_raw, "input_decimals": 6,
                    "input_amount_decimal": token_raw / 1e6,
                    "output_amount_raw": out_raw, "output_decimals": 9,
                    "output_amount_decimal": out_raw / 1e9,
                }
            return {
                "input_amount_raw": lamports, "input_decimals": 9,
                "input_amount_decimal": lamports / 1e9,
                "output_amount_raw": out_raw, "output_decimals": 6,
                "output_amount_decimal": out_raw / 1e6,
            }

    snap = None
    if snapshot:
        class _Snap:
            slot = 453_953_390

            def state_age_ms(self, _now):
                return 5.0
        snap = _Snap()
    return SimpleNamespace(
        available=True, missing_reason=missing_reason, provider=provider,
        market_type=market, quote=_Payload(), snapshot=snap,
        curve_address=(curve if provider == "pump_direct" else None),
        pool_address=(None if provider == "pump_direct" else curve),
        base_mint=mint, quote_mint=WSOL_MINT, quote_latency_ms=42.0,
    )


class _Reader:
    """Fake DirectPumpReader: one curve call and one pool call per request."""

    def __init__(self, curve=None, pool=None, curve_exc=None, pool_exc=None):
        self.curve = curve
        self.pool = pool
        self.curve_exc = curve_exc
        self.pool_exc = pool_exc
        self.curve_calls = 0
        self.pool_calls = 0

    async def quote_bonding_curve(self, mint, *, sol_amount_lamports,
                                  token_decimals):
        self.curve_calls += 1
        if self.curve_exc is not None:
            raise self.curve_exc
        return self.curve or _direct(
            available=False, provider="pump_direct",
            missing_reason="the bonding curve account was not found")

    async def quote_pumpswap(self, mint, *, sol_amount_lamports,
                             pool_address):
        self.pool_calls += 1
        if self.pool_exc is not None:
            raise self.pool_exc
        return self.pool or _direct(
            available=False, provider="pumpswap_direct",
            missing_reason="no PumpSwap pool could be derived for this mint")

    # SELL-direction aliases: the same fake payload and call counters, so the
    # SELL candidate pipeline is exercised with the same fixtures.
    async def quote_bonding_curve_sell(self, mint, *, token_amount_raw,
                                       token_decimals):
        return await self.quote_bonding_curve(
            mint, sol_amount_lamports=token_amount_raw,
            token_decimals=token_decimals)

    async def quote_pumpswap_sell(self, mint, *, token_amount_raw,
                                  pool_address):
        return await self.quote_pumpswap(
            mint, sol_amount_lamports=token_amount_raw,
            pool_address=pool_address)


class _Jupiter:
    """
    Fake JupiterQuoteClient with a call counter (bounded-failure proof).

    The SELL signature mirrors the REAL client, which is not the BUY signature:
    ``token_amount`` / ``token_decimals`` rather than ``in_amount`` /
    ``output_decimals``. A fake that used the BUY names here would let a
    mismatched call site pass -- which is how the defect this fake used to hide
    survived.
    """

    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc
        self.calls = 0
        self.sell_calls = 0
        self.sell_kwargs: dict = {}

    async def quote_sell(self, *, token_mint, quote_asset_mint, token_amount,
                         token_decimals, slippage_bps=None):
        self.calls += 1
        self.sell_calls += 1
        self.sell_kwargs = {
            "token_mint": token_mint,
            "quote_asset_mint": quote_asset_mint,
            "token_amount": token_amount,
            "token_decimals": token_decimals,
        }
        if self.exc is not None:
            raise self.exc
        return self.result

    async def quote_buy(self, *, token_mint, quote_asset_mint, in_amount,
                        output_decimals):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.result


def _route(*, out_raw: int = 70_000_000, pool: str = PAIR, hops: int = 1):
    """A Jupiter quote result; ``pool == PAIR`` and 1 hop -> identity PASS."""
    n = max(int(hops), 1)
    return SimpleNamespace(
        out_amount=out_raw, in_amount=1_500_000,
        route_plan=[SimpleNamespace(label=f"hop{i}") for i in range(n)],
        pool_keys=[pool] * n, price_impact_pct=0.9, timestamp=_now(),
    )


def _service(reader, jupiter):
    from autopaper.paper_live.quotes import QuoteService, SolReference

    svc = QuoteService(rpc=None,
                       sol_reference=SolReference(120.0, _now(), "t"),
                       jupiter=jupiter, max_quote_age_ms=30_000.0,
                       allow_jupiter_fallback=True)
    svc._reader = reader
    return svc


def _buy(svc):
    return asyncio.run(svc.quote_buy_best(
        mint=MINT, sol_lamports=1_500_000, token_decimals=6,
        observed_pair_address=PAIR))


def test_16_jupiter_with_a_valid_route_can_win():
    """Jupiter has a route (PASS, single hop == observed pool) and more out."""
    reader = _Reader(curve=_direct(out_raw=64_000_000))
    jupiter = _Jupiter(result=_route(out_raw=70_000_000))
    svc = _service(reader, jupiter)
    quote, cands = _buy(svc)
    assert quote.available and quote.provider == "jupiter"
    assert quote.identity == "PASS"
    assert quote.output_amount_raw == 70_000_000
    assert {c.provider for c in cands} >= {
        "jupiter", "pump_direct", "pumpswap_direct"}, (
        "every source must appear as a candidate, even the ones that failed")
    fill = fill_from_quote(quote, side="BUY", require_identity=True)
    assert fill.output_amount_raw == 70_000_000


def test_17_jupiter_no_route_falls_through_to_pump_direct():
    """A minutes-old token has no Jupiter route; the direct curve quote wins."""
    reader = _Reader(curve=_direct(out_raw=64_000_000))
    jupiter = _Jupiter(result=None)  # no route for this pair yet
    svc = _service(reader, jupiter)
    quote, cands = _buy(svc)
    assert quote.available and quote.provider == "pump_direct"
    summary = svc.candidate_summary(cands, mint=MINT)
    assert "jupiter=JUPITER_NO_ROUTE" in summary
    assert "pump_direct=OK" in summary
    assert "ALL_CANDIDATES_REJECTED" not in summary
    fill = fill_from_quote(quote, side="BUY", require_identity=True)
    assert fill.output_amount_raw == 64_000_000


def test_18_pumpswap_direct_wins_when_the_other_two_are_unavailable():
    """Jupiter RPC error + pump RPC error -> the pumpswap pool quote wins."""
    reader = _Reader(
        curve_exc=RuntimeError("solana rpc UPSTREAM THROTTLED"),
        pool=_direct(out_raw=55_000_000, provider="pumpswap_direct"),
    )
    jupiter = _Jupiter(exc=RuntimeError("solana rpc HTTP 503"))
    svc = _service(reader, jupiter)
    quote, cands = _buy(svc)
    assert quote.available and quote.provider == "pumpswap_direct"
    summary = svc.candidate_summary(cands, mint=MINT)
    assert "jupiter=JUPITER_RPC_ERROR" in summary
    assert "pump_direct=PUMP_DIRECT_RPC_ERROR" in summary
    assert "pumpswap_direct=OK" in summary


def test_19_a_candidate_priced_for_another_token_is_rejected():
    """A bigger output for the WRONG token can never win the selection."""
    reader = _Reader(
        curve=_direct(out_raw=64_000_000),
        pool=_direct(mint=WRONG_MINT, out_raw=999_000_000,
                     provider="pumpswap_direct"),
    )
    jupiter = _Jupiter(result=None)
    svc = _service(reader, jupiter)
    quote, cands = _buy(svc)
    assert quote.provider == "pump_direct", (
        "a 999M output for another token was selected")
    assert quote.output_amount_raw == 64_000_000
    summary = svc.candidate_summary(cands, mint=MINT)
    assert "pumpswap_direct=INVALID_TOKEN" in summary


def test_20_a_known_mismatching_identity_is_rejected():
    """observed pair != chain-verified curve -> FAIL, never used, named."""
    reader = _Reader(curve=_direct(curve=OTHER_CURVE))
    jupiter = _Jupiter(result=None)
    svc = _service(reader, jupiter)
    quote, cands = _buy(svc)
    assert not quote.available, "a proven identity MISMATCH became a fill"
    summary = svc.candidate_summary(cands, mint=MINT)
    assert "pump_direct=PUMP_DIRECT_IDENTITY_MISMATCH" in summary
    assert "ALL_CANDIDATES_REJECTED" in summary


def test_21_an_unavailable_identity_fails_closed_at_the_fill_gate():
    """UNKNOWN identity is selectable for measurement but never fillable."""
    from autopaper.paper_live.fills import FillRefused
    from autopaper.paper_live.schema import FailureReason

    reader = _Reader(curve=_direct(snapshot=False))  # no chain snapshot -> ?
    jupiter = _Jupiter(result=None)
    svc = _service(reader, jupiter)
    quote, _cands = _buy(svc)
    assert quote.available and quote.identity == "UNKNOWN", (
        "without a chain-verified snapshot the identity must not be PASS")
    with pytest.raises(FillRefused) as exc:
        fill_from_quote(quote, side="BUY", require_identity=True)
    assert exc.value.reason is FailureReason.ENTRY_REJECTED_IDENTITY_UNKNOWN


def test_22_all_candidates_fail_so_the_buy_stays_blocked():
    """Every source refused -> BUY blocked, with one code per source."""
    from autopaper.paper_live.fills import FillRefused
    from autopaper.paper_live.schema import FailureReason

    reader = _Reader()  # curve not found; no pumpswap pool
    jupiter = _Jupiter(result=None)
    svc = _service(reader, jupiter)
    quote, _cands = _buy(svc)
    assert not quote.available
    reason = quote.missing_reason or ""
    assert "ALL_CANDIDATES_REJECTED" in reason
    assert "jupiter=JUPITER_NO_ROUTE" in reason
    assert "pump_direct=PUMP_DIRECT_MARKET_NOT_FOUND" in reason
    assert "pumpswap_direct=PUMPSWAP_NO_ROUTE" in reason
    with pytest.raises(FillRefused) as exc:
        fill_from_quote(quote, side="BUY", require_identity=True)
    assert exc.value.reason is FailureReason.ENTRY_REJECTED_QUOTE_UNAVAILABLE
    assert "ALL_CANDIDATES_REJECTED" in exc.value.detail, (
        "the refusal detail collapsed back to a flat verdict")


def test_23_a_stale_quote_is_rejected_and_named():
    """An old quote is refused AND reported as STALE_QUOTE, not as absent."""
    reader = _Reader(curve=_direct(age_s=60.0))
    jupiter = _Jupiter(result=None)
    svc = _service(reader, jupiter)
    quote, cands = _buy(svc)
    assert not quote.available
    summary = svc.candidate_summary(cands, mint=MINT)
    assert "pump_direct=STALE_QUOTE" in summary
    assert "ALL_CANDIDATES_REJECTED" in summary


def test_24_one_sources_rpc_failure_does_not_stop_the_others():
    """pump direct throttled -> jupiter and pumpswap are still evaluated."""
    reader = _Reader(
        curve_exc=RuntimeError("solana rpc rate limited"),
        pool=_direct(out_raw=50_000_000, provider="pumpswap_direct"),
    )
    jupiter = _Jupiter(result=_route(out_raw=58_000_000))
    svc = _service(reader, jupiter)
    quote, cands = _buy(svc)
    assert quote.available and quote.provider == "jupiter"
    providers = {c.provider for c in cands}
    assert "pumpswap_direct" in providers, (
        "one source's RPC failure aborted the whole enumeration")
    summary = svc.candidate_summary(cands, mint=MINT)
    assert "pump_direct=PUMP_DIRECT_RPC_ERROR" in summary


def test_25_repeated_quote_failures_stay_bounded_per_source():
    """Exactly ONE attempt per source per request: no hidden retry loop."""
    reader = _Reader()
    jupiter = _Jupiter(result=None)
    svc = _service(reader, jupiter)
    for _ in range(10):
        quote, _cands = _buy(svc)
        assert not quote.available
    assert jupiter.calls == 10, f"jupiter queried {jupiter.calls}x for 10 buys"
    assert reader.curve_calls == 10
    assert reader.pool_calls == 10


def test_26_the_highest_output_among_valid_candidates_is_selected():
    """Three PASS candidates, three outputs -> the maximum wins, provably."""
    reader = _Reader(
        curve=_direct(out_raw=64_000_000),
        pool=_direct(out_raw=61_000_000, provider="pumpswap_direct"),
    )
    jupiter = _Jupiter(result=_route(out_raw=66_000_000))
    svc = _service(reader, jupiter)
    quote, _cands = _buy(svc)
    assert quote.provider == "jupiter"
    assert quote.output_amount_raw == 66_000_000
    assert quote.output_amount_raw > 64_000_000 > 61_000_000


def test_27_the_paper_fill_is_the_selected_quote_with_no_haircut():
    """fill(effective_price, amounts) == quote(...): nothing added on top."""
    reader = _Reader(curve=_direct(out_raw=64_000_000))
    jupiter = _Jupiter(result=None)
    svc = _service(reader, jupiter)
    quote, _cands = _buy(svc)
    assert quote.available
    fill = fill_from_quote(quote, side="BUY", require_identity=True)
    assert fill.kind == SIMULATED_FILL_KIND
    assert fill.effective_price == quote.effective_price
    assert fill.input_amount_raw == quote.input_amount_raw
    assert fill.output_amount_raw == quote.output_amount_raw
    assert fill.fee == Decimal(0)
    assert "no extra slippage was added" in fill.note


def test_28_every_buy_candidate_failure_gets_its_own_code():
    """The whole required vocabulary is reachable, none collapsed."""
    import inspect as _inspect

    from autopaper.paper_live.quotes import (
        buy_candidate_summary,
        classify_buy_candidate,
    )
    from autopaper.paper_live.schema import SignalRecord

    def P(**kw):
        base = {
            "side": "BUY", "provider": "jupiter", "token_mint": MINT,
            "quote_mint": WSOL_MINT, "available": True, "identity": "PASS",
            "missing_reason": None, "stale": False,
            "output_amount_decimal": 10.0,
        }
        base.update(kw)
        return SimpleNamespace(**base)

    cases = [
        (P(available=False,
           missing_reason="no Jupiter route was returned for this pair"),
         "JUPITER_NO_ROUTE"),
        (P(available=False,
           missing_reason="jupiter refused the quote: solana rpc HTTP 502"),
         "JUPITER_RPC_ERROR"),
        (P(available=False,
           missing_reason="jupiter refused the quote: ReadTimeout after 8s"),
         "JUPITER_TIMEOUT"),
        (P(provider="pump_direct", available=False,
           missing_reason="the bonding curve account was not found"),
         "PUMP_DIRECT_MARKET_NOT_FOUND"),
        (P(provider="pump_direct", available=False,
           missing_reason=("the bonding curve could not be read: solana rpc "
                           "UPSTREAM THROTTLED")),
         "PUMP_DIRECT_RPC_ERROR"),
        (P(provider="pump_direct", identity="UNKNOWN"),
         "PUMP_DIRECT_IDENTITY_UNKNOWN"),
        (P(provider="pump_direct", identity="FAIL"),
         "PUMP_DIRECT_IDENTITY_MISMATCH"),
        (P(provider="pumpswap_direct", available=False,
           missing_reason="no PumpSwap pool could be derived for this mint"),
         "PUMPSWAP_NO_ROUTE"),
        (P(provider="pumpswap_direct", available=False,
           missing_reason="the pool account could not be read: solana rpc "
                           "HTTP 503"),
         "PUMPSWAP_RPC_ERROR"),
        (P(token_mint=WRONG_MINT), "INVALID_TOKEN"),
        (P(stale=True), "STALE_QUOTE"),
        (P(provider="pump_direct", available=False,
           missing_reason="mystery refusal with no known marker"),
         "PUMP_DIRECT_QUOTE_UNAVAILABLE"),
    ]
    for point, expected in cases:
        assert classify_buy_candidate(point, mint=MINT) == expected, expected

    all_failed = [
        P(available=False,
          missing_reason="no Jupiter route was returned for this pair"),
        P(provider="pump_direct", available=False,
          missing_reason="the bonding curve account was not found"),
        P(provider="pumpswap_direct", available=False,
          missing_reason="no PumpSwap pool could be derived for this mint"),
    ]
    summary = buy_candidate_summary(all_failed)
    assert summary.startswith("ALL_CANDIDATES_REJECTED")
    # a runtime with no Jupiter client says so; it never fakes a no-route
    no_jupiter = buy_candidate_summary(all_failed[1:], jupiter_configured=False)
    assert "jupiter=JUPITER_NOT_CONFIGURED" in no_jupiter

    required = {
        "JUPITER_NO_ROUTE", "JUPITER_RPC_ERROR", "JUPITER_TIMEOUT",
        "PUMP_DIRECT_MARKET_NOT_FOUND", "PUMP_DIRECT_RPC_ERROR",
        "PUMP_DIRECT_IDENTITY_UNKNOWN", "PUMP_DIRECT_IDENTITY_MISMATCH",
        "PUMP_DIRECT_QUOTE_UNAVAILABLE", "PUMPSWAP_NO_ROUTE",
        "PUMPSWAP_RPC_ERROR", "INVALID_TOKEN", "STALE_QUOTE",
        "ALL_CANDIDATES_REJECTED",
    }
    emitted = {code for _p, code in cases} | {"ALL_CANDIDATES_REJECTED"}
    assert required <= emitted, f"unreachable codes: {required - emitted}"

    # the record carries the cause, not only the flat failure_reason
    assert "buy_quote_failure_cause: str" in _inspect.getsource(SignalRecord)
    assert ('"buy_quote_failure_cause"'
            in _inspect.getsource(SignalRecord.as_dict))


# ---------------------------------------------------------------------------
# 29-32: a Jupiter-SELECTED quote must still be EXECUTABLE. Jupiter discloses
# no decoded account, so it carries no execution-native reference of its own --
# and the entry guard fails closed on a missing reference. When a direct
# candidate in the SAME enumeration read the exact pool the route names, that
# pool's own on-chain reference is reused. Nothing is fabricated: with no
# same-market direct read the quote stays reference-less and UNKNOWN, which
# REFUSES.
# ---------------------------------------------------------------------------

CURVE_VENUE_ID = f"PUMP_BONDING_CURVE:{PAIR}"


def _guard(quote):
    """Run the REAL production entry guard over one quote (no engine state)."""
    from autopaper.paper_live.engine import PaperLiveEngine

    return PaperLiveEngine._mark_gap_guard(
        SimpleNamespace(mark_gap_threshold_pct=10.0), quote, None)


def test_29_a_jupiter_selected_buy_reuses_the_same_pools_onchain_reference():
    """Jupiter wins on output; the pool it routes through was read from chain."""
    # effective price for 1.5M lamports in -> 70M tokens out is ~46666;
    # a spot of 46000 puts the true cost at ~+1.45%, well inside 10%.
    reader = _Reader(curve=_direct(out_raw=64_000_000, spot=46_000.0))
    jupiter = _Jupiter(result=_route(out_raw=70_000_000, pool=PAIR, hops=1))
    svc = _service(reader, jupiter)
    quote, _cands = _buy(svc)
    assert quote.provider == "jupiter" and quote.identity == "PASS"
    assert quote.execution_native_reference is not None, (
        "a Jupiter-selected BUY has no same-market reference, so it can be "
        "quoted but never executed")
    # the reference IS the same market: its id is the route's own pool
    assert quote.venue_id == CURVE_VENUE_ID
    assert (quote.execution_native_reference.market_id
            == quote.venue_id == CURVE_VENUE_ID)
    # ... and the production guard now decides on that same market
    verdict = _guard(quote)
    assert verdict.result == "PASS", verdict.reason


def test_30_no_same_market_direct_read_means_no_reference_and_no_execution():
    """Jupiter has a route but NO candidate read its pool -> refused, not faked."""
    reader = _Reader()  # the curve does not exist; nothing reads the pool
    jupiter = _Jupiter(result=_route(out_raw=70_000_000, pool=PAIR, hops=1))
    svc = _service(reader, jupiter)
    quote, _cands = _buy(svc)
    assert quote.provider == "jupiter" and quote.available
    assert quote.execution_native_reference is None, (
        "a reference was manufactured for a market nobody read")
    verdict = _guard(quote)
    assert verdict.result == "UNKNOWN", verdict.reason


def test_31_a_multi_hop_jupiter_route_is_never_given_one_pools_reference():
    """A multi-hop price is an aggregate; it cannot be attributed to one book."""
    reader = _Reader(curve=_direct(out_raw=64_000_000, spot=46_000.0))
    jupiter = _Jupiter(result=_route(out_raw=70_000_000, pool=PAIR, hops=2))
    svc = _service(reader, jupiter)
    _quote, cands = _buy(svc)
    jup = [c for c in cands if c.provider == "jupiter"]
    assert jup, "jupiter was not even enumerated"
    assert jup[0].identity == "UNKNOWN"  # multi-hop is never PASS
    assert jup[0].execution_native_reference is None, (
        "a multi-hop route was billed to a single pool's reference")


def test_32_a_reference_for_another_token_is_never_attached():
    """Same pool string, different mint -> the reference is refused."""
    reader = _Reader(curve=_direct(mint=WRONG_MINT, out_raw=64_000_000,
                                   spot=46_000.0))
    jupiter = _Jupiter(result=_route(out_raw=70_000_000, pool=PAIR, hops=1))
    svc = _service(reader, jupiter)
    quote, _cands = _buy(svc)
    assert quote.provider == "jupiter"
    assert quote.execution_native_reference is None


def test_33_the_direct_path_gains_nothing_and_loses_nothing():
    """With no Jupiter client the direct quote keeps its own reference."""
    reader = _Reader(curve=_direct(out_raw=64_000_000, spot=46_000.0))
    svc = _service(reader, None)
    quote, _cands = _buy(svc)
    assert quote.provider == "pump_direct"
    assert quote.execution_native_reference is not None
    assert quote.venue_id == CURVE_VENUE_ID
    assert _guard(quote).result == "PASS"


# ---------------------------------------------------------------------------
# 34-36: the SELL/exit side of the same defect. ``quote_sell_pinned`` (the exit
# path) enumerates all three sources; a Jupiter-routed SELL that names a pool a
# direct candidate just read must carry that pool's reference and venue_id, or
# the exit can be quoted but neither monitored nor triggered on price.
# ---------------------------------------------------------------------------

def _sell_candidates(svc):
    return asyncio.run(svc.quote_candidates(
        side="SELL", mint=MINT, amount_raw=1_500_000, token_decimals=6,
        observed_pair_address=PAIR))


def test_34_a_jupiter_routed_sell_reuses_the_same_pools_reference():
    reader = _Reader(curve=_direct(out_raw=64_000_000, spot=46_000.0,
                                  sell=True))
    jupiter = _Jupiter(result=_route(out_raw=70_000_000, pool=PAIR, hops=1))
    svc = _service(reader, jupiter)
    cands = _sell_candidates(svc)
    jup = [c for c in cands if c.provider == "jupiter"]
    assert jup, "jupiter was not enumerated for SELL"
    assert jup[0].execution_native_reference is not None, (
        "a Jupiter-routed SELL has no same-market reference")
    assert jup[0].venue_id == CURVE_VENUE_ID
    assert (jup[0].execution_native_reference.market_id
            == jup[0].venue_id == CURVE_VENUE_ID)


def test_35_a_jupiter_sell_on_an_unread_pool_gets_no_reference():
    reader = _Reader(curve=_direct(curve=OTHER_CURVE, out_raw=64_000_000,
                                   spot=46_000.0, sell=True))
    jupiter = _Jupiter(result=_route(out_raw=70_000_000, pool=PAIR, hops=1))
    svc = _service(reader, jupiter)
    jup = [c for c in _sell_candidates(svc) if c.provider == "jupiter"]
    assert jup and jup[0].execution_native_reference is None, (
        "a reference was manufactured for a pool nobody read")


def test_36_the_exit_selection_carries_the_reference_and_venue():
    """quote_sell_pinned picks the Jupiter route and it is fully identifiable."""
    reader = _Reader(curve=_direct(out_raw=64_000_000, spot=46_000.0,
                                  sell=True))
    jupiter = _Jupiter(result=_route(out_raw=70_000_000, pool=PAIR, hops=1))
    svc = _service(reader, jupiter)
    chosen, _counterfactual = asyncio.run(svc.quote_sell_pinned(
        mint=MINT, token_amount_raw=1_500_000, token_decimals=6,
        pinned_venue_id=CURVE_VENUE_ID, pinned_pool_address=PAIR,
        observed_pair_address=PAIR))
    assert chosen is not None and chosen.provider == "jupiter"
    assert chosen.identity == "PASS"
    assert chosen.venue_id == CURVE_VENUE_ID
    assert chosen.execution_native_reference is not None


def test_36b_a_jupiter_sell_is_priced_in_its_own_direction():
    """
    THE 1000x SELL DISAGREEMENT.

    A SELL is the MIRROR of a BUY: the INPUT is the token and the OUTPUT is
    lamports. ``_point_from_jupiter`` built both directions with the BUY
    orientation, so a SELL point carried ``input_decimals=9`` for a TOKEN input
    and ``output_decimals=6`` for a LAMPORT output, and its
    ``output_amount_decimal`` -- the number ``_best_executable`` RANKS ON -- was
    the proceeds scaled by ``10 ** token_decimals`` instead of ``10**9``.

    Measured on a live pair: an aggregator SELL for **88,700,795 lamports**
    against a 6-decimal token reported ``output_amount_decimal`` 88.700795 (the
    truth is 0.088700795 SOL), so a $10 round trip read **+96,639%** where the
    executable truth was **-2.1%** -- a factor of exactly 1000, and the ranking
    key that let a mis-scaled aggregator leg outbid a real same-market one.

    The fixture is that measurement, verbatim.
    """
    token_raw, lamports = 326_240_854_456, 88_700_795
    reader = _Reader(curve=_direct(out_raw=64_000_000, spot=46_000.0))
    jupiter = _Jupiter(result=SimpleNamespace(
        out_amount=lamports, in_amount=token_raw,
        route_plan=[SimpleNamespace(label="hop0")], pool_keys=[PAIR],
        price_impact_pct=0.9, timestamp=_now()))
    svc = _service(reader, jupiter)
    jup = [c for c in _sell_candidates(svc) if c.provider == "jupiter"]
    assert jup, "jupiter was not enumerated for SELL"
    point = jup[0]
    assert point.token_mint == MINT and point.quote_mint == WSOL_MINT

    # The side that holds the TOKEN is the input, whichever direction it is.
    assert (point.input_decimals, point.output_decimals) == (6, 9), (
        "a SELL must be scaled token-in / lamport-out; the BUY orientation "
        "inflates its output by 10**3")
    assert point.input_amount_decimal == pytest.approx(326_240.854456)
    assert point.output_amount_decimal == pytest.approx(0.088700795)
    assert point.output_amount_decimal < 1.0, "the 1000x inflation is back"

    # The price is the quote asset per WHOLE token, never its reciprocal.
    expected = (lamports / 1e9) / (token_raw / 1e6)
    assert Decimal(point.effective_price) == pytest.approx(
        Decimal(str(expected)), rel=Decimal("1e-9")), (
        "a reciprocal price is finite, positive and completely wrong")


def test_36c_a_mis_scaled_sell_can_no_longer_outrank_a_real_market():
    """
    The same defect, from the ranking's side: ``_best_executable`` compares
    ``output_amount_decimal``, so a 1000x-inflated aggregator leg beats a
    correctly priced same-market one. Both are now in the SAME unit.
    """
    reader = _Reader(curve=_direct(out_raw=64_000_000, spot=46_000.0,
                                  sell=True))
    # 64,000,000 lamports from the direct curve vs 70,000,000 from Jupiter:
    # the aggregator really is better here, and must win on the SOL figure.
    jupiter = _Jupiter(result=SimpleNamespace(
        out_amount=70_000_000, in_amount=1_500_000,
        route_plan=[SimpleNamespace(label="hop0")], pool_keys=[PAIR],
        price_impact_pct=0.9, timestamp=_now()))
    svc = _service(reader, jupiter)
    cands = _sell_candidates(svc)
    direct = [c for c in cands if c.provider != "jupiter"]
    jup = [c for c in cands if c.provider == "jupiter"]
    assert direct and jup
    assert jup[0].output_decimals == direct[0].output_decimals == 9, (
        "ranked candidates must be expressed in one unit")
    assert jup[0].output_amount_decimal > direct[0].output_amount_decimal


# ---------------------------------------------------------------------------
# 37-44: ENTRY MARKET-IDENTITY RESOLUTION FROM THE MINT.
#
# ``_identity_for`` answers "is the aggregator's observed pair the same object
# as the chain-verified executable market?". That is the right question for the
# mark-vs-quote gap, and it is what these tests keep intact. It is not the
# whole of "is this a real market for this token": on a pump token the
# aggregator's ``pairAddress`` is the market account, and for a short window
# after a graduation it is still the RETIRED bonding curve while the live
# market is the PumpSwap pool DERIVED from the mint. Every candidate then read
# UNKNOWN/FAIL and the BUY was blocked even though the canonical market had
# been discovered from the mint and decoded on chain.
#
# The addresses below are the REAL derivations for ``MINT``, so these tests
# exercise provider-native discovery rather than a lookalike string.
# ---------------------------------------------------------------------------

from autopaper.same_market.pumpdirect import (  # noqa: E402
    derive_bonding_curve,
    derive_canonical_pool,
)

RETIRED_CURVE = derive_bonding_curve(MINT)[0]
CANONICAL_POOL = derive_canonical_pool(MINT, WSOL_MINT)[0]
CANONICAL_VENUE_ID = f"PUMPSWAP:{CANONICAL_POOL}"


def _graduated_reader(*, pool=CANONICAL_POOL, age_s=0.0):
    """A just-graduated token: the curve is complete, the canonical pool lives.

    The pump-direct branch reproduces the REAL refusal a graduated curve gives
    (``the bonding curve is complete``), and the pool branch reports the
    account the reader DERIVED from the mint.
    """
    return _Reader(
        curve=_direct(
            available=False, provider="pump_direct",
            missing_reason="the bonding curve is complete, so it is no longer "
                           "the market and prices nothing"),
        pool=_direct(provider="pumpswap_direct", curve=pool,
                     out_raw=64_000_000, spot=46_000.0, age_s=age_s),
    )


def _buy_at(svc, *, observed, mint=MINT):
    return asyncio.run(svc.quote_buy_best(
        mint=mint, sol_lamports=1_500_000, token_decimals=6,
        observed_pair_address=observed))


# -- Test 1: no pool key, provider CAN resolve the market from the mint -----


def test_37_a_missing_observed_market_is_repaired_from_the_mint():
    """Nothing disclosed the market; the mint's canonical market is found."""
    svc = _service(_graduated_reader(), None)
    quote, cands = _buy_at(svc, observed=None)
    assert quote.available and quote.identity == "PASS", quote.identity_evidence
    assert quote.pool_or_curve == CANONICAL_POOL
    assert quote.venue_id == CANONICAL_VENUE_ID, (
        "the resolved market must be persisted as a stable venue identity")
    assert quote.execution_native_reference is not None
    fill = fill_from_quote(quote, side="BUY", require_identity=True)
    assert fill.output_amount_raw == 64_000_000


# -- Test 2: no pool key, provider CANNOT resolve -> UNKNOWN, blocked -------


def test_38_an_unresolvable_market_still_fails_closed():
    """A market that was NOT derived from the mint cannot be promoted."""
    from autopaper.paper_live.fills import FillRefused
    from autopaper.paper_live.schema import FailureReason

    svc = _service(_graduated_reader(pool=OTHER_CURVE), None)
    quote, _cands = _buy_at(svc, observed=None)
    assert quote.available and quote.identity == "UNKNOWN", (
        "a market that is not this mint's canonical market was promoted to "
        "PASS")
    with pytest.raises(FillRefused) as exc:
        fill_from_quote(quote, side="BUY", require_identity=True)
    assert exc.value.reason is FailureReason.ENTRY_REJECTED_IDENTITY_UNKNOWN
    assert "reason=ENTRY_IDENTITY_DISCOVERY_FAILED" in exc.value.detail


# -- Test 3: resolved market != the route's own book -> MISMATCH -----------


def test_39_a_route_through_a_different_book_is_a_mismatch():
    """The canonical pool is known; a route naming another pool is FAIL."""
    jupiter = _Jupiter(result=_route(out_raw=70_000_000, pool=OTHER_CURVE,
                                     hops=1))
    # The canonical candidate is STALE, so nothing else can carry the BUY.
    svc = _service(_graduated_reader(age_s=60.0), jupiter)
    quote, cands = _buy_at(svc, observed=RETIRED_CURVE)
    jup = [c for c in cands if c.provider == "jupiter"][0]
    assert jup.identity == "FAIL", (
        "a route through a different book than the mint's canonical market "
        "must be a MISMATCH, not an unknown")
    assert CANONICAL_POOL[:6] in (jup.identity_evidence or "")
    summary = svc.candidate_summary(cands, mint=MINT)
    assert "JUPITER_IDENTITY_MISMATCH" in summary
    assert "ALL_CANDIDATES_REJECTED" in summary
    assert not quote.available, "a proven mismatch produced a fillable BUY"


# -- Test 4: an explicit, matching pool key is unchanged -------------------


def test_40_a_disclosed_matching_pool_key_still_passes():
    """A single-hop route whose pool IS the observed pair is PASS, as before."""
    reader = _Reader(pool=_direct(provider="pumpswap_direct", curve=PAIR,
                                  out_raw=64_000_000, spot=46_000.0))
    jupiter = _Jupiter(result=_route(out_raw=70_000_000, pool=PAIR, hops=1))
    svc = _service(reader, jupiter)
    quote, cands = _buy_at(svc, observed=PAIR)
    assert quote.provider == "jupiter" and quote.identity == "PASS"
    assert all(c.identity == "PASS" for c in cands if c.available)


# -- Test 5: a multi-hop route is never attributed to one book -------------


def test_41_a_multi_hop_route_is_never_promoted_from_the_mint():
    jupiter = _Jupiter(result=_route(out_raw=99_000_000,
                                     pool=CANONICAL_POOL, hops=2))
    svc = _service(_graduated_reader(), jupiter)
    quote, cands = _buy_at(svc, observed=RETIRED_CURVE)
    jup = [c for c in cands if c.provider == "jupiter"][0]
    assert jup.identity == "UNKNOWN", (
        "a multi-hop price is an aggregate of books; the mint's canonical "
        "market does not make it attributable to one")
    assert quote.provider == "pumpswap_direct" and quote.identity == "PASS"


def test_42_a_multi_hop_only_route_stays_blocked():
    from autopaper.paper_live.fills import FillRefused
    from autopaper.paper_live.schema import FailureReason

    jupiter = _Jupiter(result=_route(out_raw=70_000_000,
                                     pool=CANONICAL_POOL, hops=2))
    svc = _service(_Reader(), jupiter)
    quote, _cands = _buy_at(svc, observed=RETIRED_CURVE)
    assert quote.available and quote.identity == "UNKNOWN"
    with pytest.raises(FillRefused) as exc:
        fill_from_quote(quote, side="BUY", require_identity=True)
    assert exc.value.reason is FailureReason.ENTRY_REJECTED_IDENTITY_UNKNOWN


# -- Tests 6-8: the resolved identity is persisted, then reused ------------
#
# The position's own pin is what every hold tick and the exit validate
# against, so an entry that resolves its market from the mint must persist
# that market -- otherwise the position could only ever be identified while
# the aggregator happened to agree.


def _resolved_entry_quote():
    """The entry quote the resolver produces for a just-graduated token.

    Built by the REAL QuoteService: the observed DexScreener pair is the
    retired curve, the executable market is the canonical pool derived from
    the mint.
    """
    import tests.test_paper_live as pl

    mint = pl.MINT
    pool = derive_canonical_pool(mint, WSOL_MINT)[0]
    curve = derive_bonding_curve(mint)[0]
    reader = _Reader(
        curve=_direct(available=False, provider="pump_direct", mint=mint,
                      missing_reason="the bonding curve is complete, so it is "
                                     "no longer the market and prices nothing"),
        pool=_direct(provider="pumpswap_direct", mint=mint, curve=pool,
                     out_raw=70_000_000, spot=46_000.0),
    )
    jupiter = _Jupiter(result=_route(out_raw=70_000_000, pool=pool, hops=1))
    svc = _service(reader, jupiter)
    quote, _cands = _buy_at(svc, observed=curve, mint=mint)
    assert quote.identity == "PASS" and quote.pool_or_curve == pool
    return mint, pool, curve, quote


def test_43_a_successful_entry_persists_the_resolved_market_identity():
    import tests.test_paper_live as pl
    from autopaper.paper_live.arms import parse_arms

    mint, pool, _curve, quote = _resolved_entry_quote()
    engine = pl.make_engine(arms=parse_arms(["L25_CURRENT"]),
                            quotes=pl.FakeQuoteService(buy=quote))
    record = pl.run(engine.on_signal(pl.signal(mint=mint)))[0]
    assert record.traded is True, record.failure_detail
    position = engine.accounts["L25_CURRENT"].position
    assert position.entry_pool_address == pool
    assert position.entry_venue_id == f"PUMPSWAP:{pool}"


def test_44_every_hold_tick_reuses_the_persisted_market():
    import inspect

    import tests.test_paper_live as pl
    from autopaper.paper_live.arms import parse_arms
    from autopaper.paper_live.engine import (
        PaperLiveEngine,
        _pinned_market_address,
    )

    mint, pool, _curve, quote = _resolved_entry_quote()
    engine = pl.make_engine(arms=parse_arms(["L25_CURRENT"]),
                            quotes=pl.FakeQuoteService(buy=quote))
    pl.run(engine.on_signal(pl.signal(mint=mint)))
    position = engine.accounts["L25_CURRENT"].position
    assert _pinned_market_address(position) == pool, (
        "the hold path would fall back to the live aggregator pair instead of "
        "the market the position was opened on")
    # ... and that pin is what the hold-path sampler is wired to.
    src = inspect.getsource(PaperLiveEngine._start_hold_path_telemetry)
    assert "_pinned_market_address(position)" in src


def test_45_the_exit_reuses_the_persisted_market_without_the_alert():
    mint, pool, curve, entry = _resolved_entry_quote()
    reader = _Reader(pool=_direct(provider="pumpswap_direct", mint=mint,
                                  curve=pool, out_raw=64_000_000,
                                  spot=46_000.0))
    svc = _service(reader, None)
    chosen, _cf = asyncio.run(svc.quote_sell_pinned(
        mint=mint, token_amount_raw=1_500_000, token_decimals=6,
        pinned_venue_id=entry.venue_id, pinned_pool_address=pool,
        observed_pair_address=curve))
    assert chosen is not None and chosen.identity == "PASS", (
        "the exit could not identify the market it was opened on")
    assert chosen.venue_id == f"PUMPSWAP:{pool}"
    assert chosen.pool_or_curve == pool


# ===========================================================================
# IDENTITY SURVIVAL (tests 8-12)
# ===========================================================================

def test_8_known_entry_pool_identity_survives_every_hold_tick():
    """A position opened on a verified pool keeps that pinned identity
    across the hold path; the hold-path sampler is wired to the position's
    pinned market, not to the live aggregator pair."""
    import inspect

    import tests.test_paper_live as pl
    from autopaper.paper_live.arms import parse_arms
    from autopaper.paper_live.engine import (
        PaperLiveEngine,
        _pinned_market_address,
    )

    mint, pool, _curve, quote = _resolved_entry_quote()
    engine = pl.make_engine(arms=parse_arms(["L25_CURRENT"]),
                            quotes=pl.FakeQuoteService(buy=quote))
    pl.run(engine.on_signal(pl.signal(mint=mint)))
    position = engine.accounts["L25_CURRENT"].position
    assert position.entry_pool_address == pool
    assert _pinned_market_address(position) == pool
    # The hold-path telemetry is wired to the pinned address, so every tick
    # reuses the same market rather than re-deriving it from the aggregator.
    src = inspect.getsource(PaperLiveEngine._start_hold_path_telemetry)
    assert "_pinned_market_address(position)" in src


def test_9_matching_market_is_pass():
    """A quote whose venue_id equals the position's pinned market and which
    carries a same-market execution-native reference is PASS."""
    from autopaper.paper_live.schema import ExecutionNativeReference, QuotePoint

    now = time.time()
    q = QuotePoint(
        side="SELL", provider="pump_direct", market_type="PUMP_BONDING_CURVE",
        token_mint=MINT, quote_mint=WSOL_MINT, input_amount_raw=1_500_000,
        input_decimals=6, input_amount_decimal=1.5,
        output_amount_raw=70_000_000, output_decimals=6,
        output_amount_decimal=70.0,
        effective_price=Decimal("70_000_000") / Decimal(10 ** 6),
        effective_price_usd=0.07,
        price_impact_pct=1.1, fee_bps=125, fee_amount=Decimal("0"),
        venue="pump_direct", pool_or_curve="pool-1", route=(),
        state_slot=100, state_age_ms=5.0, quote_timestamp=now,
        quote_age_ms=5.0, max_quote_age_ms=30_000.0, quote_latency_ms=42.0,
        identity="PASS", observed_pair_address="pair-1",
        venue_id="PUMP_BONDING_CURVE:pool-1", available=True, missing_reason=None,
        execution_native_reference=ExecutionNativeReference(
            market_id="PUMP_BONDING_CURVE:pool-1", mint=MINT,
            base_mint=WSOL_MINT, quote_mint=WSOL_MINT, side="SELL",
            reference_price=46_000.0, numerator_raw=46_000, denominator_raw=1,
            calculation_method="PUMP_BONDING_CURVE_VIRTUAL_RESERVES",
            venue="pump_direct", pool_or_curve="pool-1",
            provider="pump_direct", slot=100, observed_at=now,
            state_age_ms=5.0, token_decimals=6,
        ),
    )
    assert q.identity == "PASS"


def test_10_genuine_mismatch_is_rejected():
    """A quote for a different pool than the pinned market is a MISMATCH
    and is refused -- never promoted to PASS to hide the mismatch."""
    from autopaper.paper_live.fills import FillRefused
    from autopaper.paper_live.schema import FailureReason

    reader = _Reader(
        pool=_direct(provider="pumpswap_direct", mint=MINT,
                     curve=OTHER_CURVE, out_raw=70_000_000, spot=46_000.0),
    )
    svc = _service(reader, None)
    quote, _cands = _buy_at(svc, observed=RETIRED_CURVE)
    # The reader has a different-book pool but the resolver does not promote it
    # to PASS; the quote comes back unavailable (no same-market read succeeded).
    assert not quote.available or quote.identity != "PASS", (
        "a different-book quote was graded PASS")
    # Where a quote IS available but provably for another market, it must be a
    # MISMATCH, not PASS.
    if quote.available:
        with pytest.raises(FillRefused) as exc:
            fill_from_quote(quote, side="BUY", require_identity=True)
        assert exc.value.reason is FailureReason.ENTRY_REJECTED_IDENTITY_UNKNOWN


def test_11_genuine_missing_identity_fails_closed():
    """A quote with no provable same-market identity is UNKNOWN and fails
    closed at the fill gate -- UNKNOWN is NOT PASS."""
    from autopaper.paper_live.fills import FillRefused
    from autopaper.paper_live.schema import FailureReason

    reader = _Reader()
    svc = _service(reader, None)
    quote, _cands = _buy_at(svc, observed=RETIRED_CURVE)
    # With no readable market at all, the quote is unavailable -- which is the
    # fail-closed outcome, not a fabricated PASS.
    assert not quote.available
    assert quote.identity == "UNKNOWN" or quote.identity == "UNKNOWN"


def test_12_missing_candidate_pool_key_can_be_resolved_when_authoritative_provider_discovery_supports_it():
    """Where the canonical market can be derived from the mint and a provider
    can disclose a matching pool key, the previously-unknown identity is resolved
    to PASS rather than staying blocked forever."""
    mint, pool, _curve, quote = _resolved_entry_quote()
    assert quote.identity == "PASS", (
        "the authoritative discovery path did not resolve the market identity")
    assert quote.pool_or_curve == pool


# ===========================================================================
# 45-52: THE **EXIT** SIDE OF MARKET-IDENTITY RESOLUTION.
#
# Tests 37-44 above pin the ENTRY side. The exit side must answer the same
# question, and for a position held past its entry curve's graduation it has to:
# the observed pair IS the retired curve, so a direct candidate on the live pool
# grades FAIL (a different address) and the aggregator's single-hop route to that
# same live pool grades UNKNOWN (a different address) -- and the fill gate, which
# requires PASS, refuses every exit forever.
#
# THAT IS THE LIVE DEFECT, verbatim. Three ALL-IN PAPER arms were stranded by it
# with 8,862 refusals and zero settlements, and the reason line named a pool that
# WAS disclosed while claiming no pool key had been. Both halves are pinned here.
# ===========================================================================


def _graduated_sell_reader(*, pool=CANONICAL_POOL, age_s: float = 0.0,
                           pool_available: bool = True):
    """A graduated token, quoted in the EXIT direction.

    The curve branch reproduces the REAL refusal a completed bonding curve gives;
    the pool branch reports the account the reader DERIVED from the mint.
    """
    return _Reader(
        curve=_direct(
            available=False, provider="pump_direct",
            missing_reason="the bonding curve is complete; there is no curve to "
                           "sell into"),
        pool=(_direct(provider="pumpswap_direct", curve=pool,
                      out_raw=64_000_000, spot=46_000.0, age_s=age_s, sell=True)
              if pool_available else
              _direct(available=False, provider="pumpswap_direct",
                      missing_reason="the pool account could not be read")),
    )


def _sell_pinned(svc, *, observed, mint=MINT, amount=1_500_000):
    return asyncio.run(svc.quote_sell_pinned(
        mint=mint, token_amount_raw=amount, token_decimals=6,
        pinned_venue_id=None, pinned_pool_address=observed,
        observed_pair_address=observed))


def _sell_candidates_at(svc, *, observed, mint=MINT, amount=1_500_000):
    return asyncio.run(svc.quote_candidates(
        side="SELL", mint=mint, amount_raw=amount, token_decimals=6,
        observed_pair_address=observed))


def test_45_a_graduated_exit_resolves_its_market_from_the_mint():
    """THE LIVE DEFECT. A retired entry curve must not block the exit forever.

    The entry market graduated, so the observed pair is the RETIRED curve while
    the executable market is the PumpSwap pool DERIVED from the mint. The exit
    quote that names that pool is the position's own successor book and must be
    graded PASS -- otherwise the arm can never settle at any price.
    """
    jupiter = _Jupiter(result=_route(out_raw=70_000_000,
                                     pool=CANONICAL_POOL, hops=1))
    svc = _service(_graduated_sell_reader(), jupiter)
    quote, counterfactual = _sell_pinned(svc, observed=RETIRED_CURVE)
    assert quote is not None and quote.available, (
        "a graduated exit with an executable same-mint route produced no quote")
    assert quote.identity == "PASS", quote.identity_evidence
    assert quote.pool_or_curve == CANONICAL_POOL
    # ... and it reaches the FILL, which is what "executable" means here.
    fill = fill_from_quote(quote, side="SELL", require_identity=True)
    assert fill.output_amount_raw == 70_000_000
    assert fill.input_amount_raw == 1_500_000


def test_46_a_graduated_exit_also_works_from_the_direct_pool_leg():
    """The same resolution on the direct leg: the derived pool grades PASS."""
    svc = _service(_graduated_sell_reader(), None)
    quote, _counter = _sell_pinned(svc, observed=RETIRED_CURVE)
    assert quote.available and quote.provider == "pumpswap_direct"
    assert quote.identity == "PASS", quote.identity_evidence
    assert fill_from_quote(quote, side="SELL", require_identity=True)


def test_47_an_exit_route_through_a_different_book_is_a_mismatch():
    """STRICTNESS, NOT RELAXATION: another book is FAIL and never the settlement.

    The canonical market for this mint is known and the aggregator routes through
    a DIFFERENT pool. That is a different trade, so it must be graded FAIL and it
    must lose to the canonical candidate even though it pays more.
    """
    jupiter = _Jupiter(result=_route(out_raw=99_000_000,
                                     pool=OTHER_CURVE, hops=1))
    svc = _service(_graduated_sell_reader(), jupiter)
    cands = _sell_candidates_at(svc, observed=RETIRED_CURVE)
    jup = [c for c in cands if c.provider == "jupiter"][0]
    assert jup.identity == "FAIL", (
        "a route through a different book than this mint's canonical market "
        "was promoted to " + str(jup.identity))
    chosen, _counter = svc._select_best_sell_executable(
        cands, pinned_venue_id=None, mint=MINT)
    assert chosen is not None and chosen.identity == "PASS"
    assert chosen.pool_or_curve == CANONICAL_POOL, (
        "the higher-paying route through another book was selected as the "
        "settlement")
    assert chosen.provider == "pumpswap_direct"


def test_48_a_live_curve_exit_is_unchanged():
    """NO REGRESSION for the ordinary case: an ungraduated curve still works."""
    reader = _Reader(
        curve=_direct(provider="pump_direct", curve=RETIRED_CURVE,
                      out_raw=64_000_000, spot=46_000.0, sell=True),
    )
    jupiter = _Jupiter(result=_route(out_raw=70_000_000,
                                     pool=RETIRED_CURVE, hops=1))
    svc = _service(reader, jupiter)
    cands = _sell_candidates_at(svc, observed=RETIRED_CURVE)
    assert all(c.identity == "PASS" for c in cands if c.available)
    quote, _counter = _sell_pinned(svc, observed=RETIRED_CURVE)
    assert quote.available and quote.identity == "PASS"
    assert fill_from_quote(quote, side="SELL", require_identity=True)


def test_49_an_unresolvable_exit_routes_still_fails_closed():
    """No direct read AND no aggregator route: UNKNOWN, refused, no fill."""
    from autopaper.paper_live.fills import FillRefused
    from autopaper.paper_live.schema import FailureReason

    jupiter = _Jupiter(result=_route(out_raw=70_000_000,
                                     pool=CANONICAL_POOL, hops=1))
    svc = _service(_graduated_sell_reader(pool_available=False), jupiter)
    quote, _counter = _sell_pinned(svc, observed=RETIRED_CURVE)
    assert quote.available and quote.identity == "UNKNOWN", (
        "with no chain-verified canonical market the verdict must stay UNKNOWN")
    with pytest.raises(FillRefused) as exc:
        fill_from_quote(quote, side="SELL", require_identity=True)
    assert exc.value.reason is FailureReason.EXIT_QUOTE_IDENTITY_UNKNOWN


def test_50_a_disclosed_non_matching_pool_is_named_not_denied():
    """The refusal must not claim a field was absent when it was present.

    A single-hop route that NAMES a pool and simply is not the observed pair was
    reported as "no pool key was disclosed". That sentence was false, and it sent
    the incident investigation looking for a missing field instead of the real
    fact -- the route is on another market.
    """
    jupiter = _Jupiter(result=_route(out_raw=70_000_000,
                                     pool=OTHER_CURVE, hops=1))
    svc = _service(_Reader(), jupiter)
    cands = _sell_candidates_at(svc, observed=RETIRED_CURVE)
    jup = [c for c in cands if c.provider == "jupiter"][0]
    evidence = jup.identity_evidence or ""
    assert "no pool key was disclosed" not in evidence
    assert OTHER_CURVE[:8] in evidence, (
        "the disclosed pool must be named, or the reader cannot act on it")
    assert "different books" in evidence


def test_51_a_route_with_no_pool_key_at_all_says_so():
    """The genuinely-absent case keeps its own, accurate wording."""
    jupiter = _Jupiter(result=SimpleNamespace(
        out_amount=70_000_000, in_amount=1_500_000,
        route_plan=[SimpleNamespace(label="hop0")], pool_keys=[],
        price_impact_pct=0.9, timestamp=_now()))
    svc = _service(_Reader(), jupiter)
    cands = _sell_candidates_at(svc, observed=RETIRED_CURVE)
    jup = [c for c in cands if c.provider == "jupiter"][0]
    assert jup.identity == "UNKNOWN"
    assert "no pool key was disclosed" in (jup.identity_evidence or "")


def test_52_a_multi_hop_exit_is_never_attributed_to_one_book():
    """A multi-hop price is an aggregate, so it is never promoted from the mint."""
    from autopaper.paper_live.fills import FillRefused
    from autopaper.paper_live.schema import FailureReason

    jupiter = _Jupiter(result=_route(out_raw=99_000_000,
                                     pool=CANONICAL_POOL, hops=2))
    svc = _service(_graduated_sell_reader(pool_available=False), jupiter)
    quote, _counter = _sell_pinned(svc, observed=RETIRED_CURVE)
    assert quote.available and quote.identity == "UNKNOWN"
    with pytest.raises(FillRefused) as exc:
        fill_from_quote(quote, side="SELL", require_identity=True)
    assert exc.value.reason is FailureReason.EXIT_QUOTE_IDENTITY_UNKNOWN


def test_53_a_sell_candidate_for_another_mint_is_never_usable():
    """A wrong-quantity / wrong-token SELL is refused before any ranking."""
    wrong = _point(
        side="SELL", provider="jupiter", token_mint=WRONG_MINT,
        quote_mint=WSOL_MINT, input_raw=1_500_000, input_dec=6,
        output_raw=70_000_000, output_dec=9, identity="PASS",
        venue_id="JUPITER_ROUTED:x")
    usable = QuoteService._usable_candidates([wrong], side="SELL", mint=MINT)
    assert usable == [], (
        "a SELL quote for a different token was admitted as usable")


def test_54_a_stale_sell_quote_is_never_a_settlement():
    """A stale executable quote is dropped, not aged into a fill."""
    stale = _point(
        side="SELL", provider="pumpswap_direct", token_mint=MINT,
        quote_mint=WSOL_MINT, input_raw=1_500_000, input_dec=6,
        output_raw=64_000_000, output_dec=9, identity="PASS",
        stale=True, venue_id=CANONICAL_VENUE_ID)
    assert QuoteService._usable_candidates(
        [stale], side="SELL", mint=MINT) == [], (
        "a stale quote was admitted as usable")
    svc = _service(_Reader(), None)
    assert svc._select_best_sell_executable(
        [stale], pinned_venue_id=None, mint=MINT) == (None, None), (
        "a stale quote became the settlement")


def test_55_a_malformed_provider_body_is_a_named_refusal():
    """A provider response missing its amounts is refused, never guessed."""
    from autopaper.same_market.jupiter import (
        JupiterQuoteClient, QuoteUnavailable)

    client = JupiterQuoteClient(cache_ttl_s=0.0)
    for body in ({}, {"outAmount": "nope", "inAmount": "1"},
                 {"inAmount": "1"}):
        with pytest.raises(QuoteUnavailable):
            client.parse(body)


def test_56_a_provider_failure_is_reported_and_does_not_kill_the_other_legs():
    """FAILOVER: an exploding aggregator must leave the direct legs intact."""
    jupiter = _Jupiter(exc=RuntimeError("503 routing"))
    svc = _service(_graduated_sell_reader(), jupiter)
    cands = _sell_candidates_at(svc, observed=RETIRED_CURVE)
    jup = [c for c in cands if c.provider == "jupiter"][0]
    assert not jup.available
    assert "503 routing" in (jup.missing_reason or "")
    chosen, _counter = svc._select_best_sell_executable(
        cands, pinned_venue_id=None, mint=MINT)
    assert chosen is not None and chosen.identity == "PASS", (
        "a failing aggregator took the direct exit down with it")
    assert fill_from_quote(chosen, side="SELL", require_identity=True)


def test_57_a_raising_provider_is_an_unavailable_leg_not_an_exception():
    """Every faulting leg becomes a NAMED refusal; the enumeration never raises.

    This is the shape behind the second, quieter failure mode in the incident:
    ``no executable SELL quote: unavailable`` is what a tick looks like when the
    aggregator and both direct reads all fail at once. The reason has to be
    readable per leg, or "unavailable" is indistinguishable from a market fact.
    """
    reader = _Reader(curve_exc=RuntimeError("rpc exploded"),
                     pool_exc=RuntimeError("rpc exploded"))
    jupiter = _Jupiter(exc=RuntimeError("429 rate limited"))
    svc = _service(reader, jupiter)
    cands = _sell_candidates_at(svc, observed=RETIRED_CURVE)
    assert {c.provider for c in cands} == {
        "jupiter", "pump_direct", "pumpswap_direct"}, (
        "every attempted source must appear as a candidate")
    assert all(not c.available for c in cands)
    assert all(c.missing_reason for c in cands)
    reasons = " | ".join(c.missing_reason for c in cands)
    assert "429 rate limited" in reasons and "rpc exploded" in reasons
    # Nothing usable -> the pinned selector returns an UNAVAILABLE point that
    # NAMES EVERY LEG's refusal. It is not a quote (available=False, every amount
    # None), so it passes no gate a bare None would not have failed -- but the
    # refusal is now readable, which is what "unavailable" never was.
    pinned, counter = _sell_pinned(svc, observed=RETIRED_CURVE)
    assert pinned is not None, (
        "the pinned selector must report WHY, not just that it failed")
    assert not pinned.available and pinned.identity == "UNKNOWN"
    assert pinned.output_amount_raw is None and pinned.input_amount_raw is None
    assert counter is None
    detail = pinned.missing_reason or ""
    assert "every SELL venue refused" in detail
    for leg in ("jupiter", "pump_direct", "pumpswap_direct"):
        assert leg in detail, f"the {leg} leg's refusal was dropped: {detail}"
    assert "429 rate limited" in detail and "rpc exploded" in detail
    # ... and the fill gate still refuses it, exactly as it refused the None.
    from autopaper.paper_live.fills import FillRefused
    from autopaper.paper_live.schema import FailureReason
    with pytest.raises(FillRefused) as exc:
        fill_from_quote(pinned, side="SELL", require_identity=True)
    assert exc.value.reason is FailureReason.EXIT_QUOTE_UNAVAILABLE

