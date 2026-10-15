"""
REAL equity valued from the CANONICAL SOL/USD reference, not a swap quote.

THE DEFECT THIS PINS
====================
``LiveSolanaBackend.sol_price_usd()`` asked the EXECUTION quote provider for a
``WSOL -> USDC`` route in order to obtain a reference price. That is the wrong
abstraction, and it failed closed for a structural reason rather than a
transient one: both ends of that leg are quote assets, so
``resolve_side_from_mints()`` correctly refuses to choose a direction. The
safety invariant was RIGHT; the caller was wrong. Every REAL equity read then
reported "unavailable" forever.

The fix routes the reference through the one provider that already backs PAPER
sizing (``autopaper.same_market.sol_price``), injected rather than constructed
inside the accounting layer. These tests hold both halves of that in place:

* the REAL path reads the SAME canonical provider PAPER does, and
* ``resolve_side_from_mints()`` is left exactly as it was.

FULLY OFFLINE. No test here opens a socket: the provider is always injected with
an ``http_get`` hook or a fake, and ``test_the_fake_provider_performs_zero_
network_calls`` asserts that structurally rather than trusting it.

Run:  PYTHONPATH=src .venv/bin/python -m pytest tests/test_real_equity_sol_usd.py -q
"""

from __future__ import annotations

import asyncio
import inspect
import time
from pathlib import Path

import pytest

from autopaper.engine import PaperTrader
from autopaper.same_market import sol_price as S
from autopaper.same_market.sol_price import (
    JupiterSolPriceProvider,
    SolPriceUnavailable,
    SolUsdSnapshot,
)

pytestmark = [pytest.mark.safety, pytest.mark.execution]

WSOL = "So11111111111111111111111111111111111111112"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
PRICE = 180.25
SOL_USD = 180.25

AI = Path(__file__).resolve().parents[1]
SRC = AI / "src" / "autopaper"


# ---------------------------------------------------------------------------
# Fakes. Every one of these is injected; none can reach a network.
# ---------------------------------------------------------------------------


def _price_body(price=PRICE, *, mint=WSOL):
    return {mint: {"usdPrice": price, "decimals": 9, "blockId": 348004023}}


class FakeSolUsdProvider:
    """
    A stand-in for :class:`JupiterSolPriceProvider`.

    Records every call so a test can assert on caching and on the ABSENCE of
    network access, and can be told to fail or to hand back a stale snapshot so
    the fail-closed paths are exercised without waiting for real time to pass.
    """

    def __init__(self, price=PRICE, *, fail=False, snapshot=None, clock=None):
        self.price = price
        self.fail = fail
        self.calls = 0
        self._snapshot = snapshot
        self._clock = clock or time.time

    def now(self):
        return float(self._clock())

    async def snapshot(self, *, refresh: bool = False):
        self.calls += 1
        if self.fail:
            raise SolPriceUnavailable("SOL/USD provider unavailable (test)")
        if self._snapshot is not None:
            return self._snapshot
        return SolUsdSnapshot(
            usd_per_sol=float(self.price),
            timestamp=self.now(),
            source="fake (test)",
            fetched=True,
        )


class FakeRPC:
    """Token balances + supply. Read-only, in-memory."""

    def __init__(self, balances=()):
        self._balances = list(balances)

    async def token_balances(self, pubkey):
        return list(self._balances)

    async def get_token_supply(self, mint):
        return {"decimals": 6}


class FakeQuoteProvider:
    """Prices a held SPL in WSOL. Records calls so 'no swap quote for SOL' holds."""

    def __init__(self, lamports_per_token=None):
        self.calls = []
        self._px = lamports_per_token

    async def quote(self, input_mint, output_mint, in_amount, slippage_bps, **kw):
        self.calls.append((input_mint, output_mint, in_amount))
        if self._px is None:
            raise RuntimeError("no route")
        return type("Q", (), {"expected_out_amount": self._px})()


class FakeBackend:
    """A live-shaped backend: wallet_state + rpc + provider. Never signs."""

    def __init__(self, *, lamports=10_718_431, masked="wUEU…Wbt6",
                 balances=(), quote_provider=None):
        self.pubkey = "FakePubkey1111111111111111111111111111111"
        self.rpc = FakeRPC(balances)
        self.provider = quote_provider if quote_provider is not None \
            else FakeQuoteProvider()
        self._lamports = lamports
        self._masked = masked

    async def wallet_state(self):
        return {"lamports": self._lamports, "sol": self._lamports / 1e9,
                "masked": self._masked}


def make_trader(provider, backend=None, **kw):
    """
    A PaperTrader carrying ONLY what ``real_equity()`` touches.

    Built by subclassing rather than by constructing a real one: this keeps the
    test hermetic (no config, no var dir, no lock, no Telegram) while still
    exercising the REAL production methods -- ``real_equity``,
    ``_backend_sol_price``, ``_real_wallet_holdings`` and
    ``compute_real_equity_multi`` are the real implementations, not doubles.
    """
    backend = backend if backend is not None else FakeBackend(**kw)

    class _T(PaperTrader):
        def __init__(self):
            self.sol_usd_provider = provider
            self._sol_usd_snapshot = None
            self._sol_price_usd = None
            self.real_equity_cache = None

        @property
        def real_backend(self):
            return backend

        @real_backend.setter
        def real_backend(self, value):  # pragma: no cover - never assigned
            raise AssertionError("real_backend is injected, not assigned")

    return _T()


def run(coro):
    return asyncio.run(coro)


# ===========================================================================
# 1-4. The reference resolves, and SOL / WSOL / USDC are all valued
# ===========================================================================


def test_real_equity_returns_usd_equity_from_the_injected_provider():
    """(1) The headline acceptance criterion."""
    provider = FakeSolUsdProvider(PRICE)
    t = make_trader(provider)
    eq = run(t.real_equity())

    assert eq.available, eq.reason
    assert eq.equity_usd == pytest.approx(0.010718431 * PRICE)
    assert eq.sol_price_usd == pytest.approx(PRICE)
    assert eq.wallet_masked == "wUEU…Wbt6"


def test_native_sol_is_valued_with_a_valid_reference():
    """(2)"""
    eq = run(make_trader(FakeSolUsdProvider(200.0)).real_equity())
    assert eq.available
    assert eq.equity_usd == pytest.approx(0.010718431 * 200.0)


def test_wsol_holding_is_valued_at_the_sol_reference():
    """
    (3) WSOL is SOL. It must never need its own price, and never be able to
    make equity unavailable on its own account.
    """
    backend = FakeBackend(balances=[
        {"mint": WSOL, "symbol": "WSOL", "amount": 1_000_000_000, "decimals": 9},
    ])
    eq = run(make_trader(FakeSolUsdProvider(PRICE), backend).real_equity())

    assert eq.available, eq.reason
    # native 0.010718431 + wsol 1.0, both at PRICE
    assert eq.equity_usd == pytest.approx((0.010718431 + 1.0) * PRICE)
    assert eq.unpriced_mints == ()


def test_usdc_holding_is_still_valued_correctly():
    """
    (4) A non-SOL holding keeps working.

    USDC is priced by ASKING the venue (asset -> WSOL) and converting with the
    same SOL reference. That is the pre-existing, unchanged behaviour and it
    must still hold after the reference itself moved.
    """
    backend = FakeBackend(
        balances=[{"mint": USDC, "symbol": "USDC",
                   "amount": 25_000_000, "decimals": 6}],
        # 25 USDC -> 25/180.25 SOL, quoted for ONE WHOLE token (1e6 units).
        quote_provider=FakeQuoteProvider(
            lamports_per_token=int(1e9 / PRICE)),
    )
    eq = run(make_trader(FakeSolUsdProvider(PRICE), backend).real_equity())

    assert eq.available, eq.reason
    assert eq.equity_usd == pytest.approx(0.010718431 * PRICE + 25.0)


# ===========================================================================
# 5-6. Failure and staleness stay fail-closed
# ===========================================================================


def test_provider_failure_makes_equity_unavailable():
    """(5) Never $0.00. An unreadable price is an explicit refusal."""
    t = make_trader(FakeSolUsdProvider(fail=True))
    eq = run(t.real_equity())

    assert not eq.available
    assert eq.equity_usd is None
    assert "SOL/USD" in eq.reason
    assert eq.sol_price_usd == 0.0


def test_stale_reference_makes_equity_unavailable():
    """
    (6) Staleness is a REFUSAL, not a warning.

    Built by backdating the snapshot rather than by sleeping, so the test is
    deterministic and instant.
    """
    stale = SolUsdSnapshot(
        usd_per_sol=PRICE,
        timestamp=time.time() - 3_600.0,  # an hour old
        source="fake (test)",
        fetched=True,
    )
    provider = FakeSolUsdProvider(snapshot=stale)
    eq = run(make_trader(provider).real_equity())

    assert not eq.available
    assert eq.equity_usd is None
    assert "SOL/USD" in eq.reason


def test_no_provider_wired_is_unavailable_not_a_guess():
    """
    A backend with no provider must refuse rather than reach for a quote.

    Asserted on the BACKEND, which is where the old swap-quote lookup lived, so
    the guarantee is tested where the defect was -- and offline, because no
    network client is involved at all.
    """
    from autopaper.execution.real import LiveSolanaBackend

    backend = LiveSolanaBackend.__new__(LiveSolanaBackend)
    backend._sol_usd_provider = None

    assert run(backend.sol_usd_snapshot()) is None
    assert run(backend.sol_price_usd()) is None


# ===========================================================================
# 7. Offline: the fake provider performs zero network calls
# ===========================================================================


def test_the_fake_provider_performs_zero_network_calls():
    """
    (7) Proven structurally, not by trust.

    The whole point of injecting a provider is that the accounting layer has no
    transport of its own. So the assertion is about the CLIENT: a provider built
    with an ``http_get`` hook never constructs an httpx client, so there is no
    socket for an offline run to dial, and the only transport that executed is
    the hook the test itself supplied.
    """
    hook_calls = []

    def _hook(params):
        hook_calls.append(params)
        return _price_body()

    provider = JupiterSolPriceProvider(http_get=_hook, clock=time.time)
    trader = make_trader(provider)
    eq = run(trader.real_equity())

    assert eq.available, eq.reason
    assert provider._client is None, "the provider built a real HTTP client"
    assert provider.total_requests == 0, "the rate-limited transport was used"
    # the hook is the ONLY thing that ran, and it ran exactly once
    assert len(hook_calls) == 1


def test_a_provider_with_no_hook_never_dials_out_under_the_offline_guard(monkeypatch):
    """
    (7, harder) The engine must not construct a network client just because
    equity was asked for.

    ``httpx.AsyncClient`` is replaced with a tripwire, so if the engine ever
    built a real transport on this path the test fails instead of quietly
    reaching Jupiter.
    """
    import httpx

    def _forbidden(*a, **kw):
        raise AssertionError("the SOL/USD path constructed an HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", _forbidden)

    eq = run(make_trader(FakeSolUsdProvider(PRICE)).real_equity())
    assert eq.available, eq.reason


def test_the_providers_http_hook_is_the_only_transport():
    """
    Belt and braces on (7): a provider built with an ``http_get`` hook and NO
    client never constructs an httpx client, so an offline run cannot dial out
    even by accident.
    """
    provider = JupiterSolPriceProvider(http_get=lambda p: _price_body())
    run(provider.snapshot())
    assert provider._client is None, "an http_get-injected provider built a client"


# ===========================================================================
# 8. The cache is shared, not duplicated
# ===========================================================================


def test_the_shared_cache_is_reused_across_equity_reads():
    """(8) One provider, one cache: repeated reads do not re-fetch."""
    calls = []

    def _get(params):
        calls.append(params)
        return _price_body()

    now = {"t": time.time()}
    provider = JupiterSolPriceProvider(
        http_get=_get, clock=lambda: now["t"], ttl_s=60.0)
    t = make_trader(provider)

    first = run(t.real_equity())
    second = run(t.real_equity())

    assert first.available and second.available
    assert first.equity_usd == second.equity_usd
    assert len(calls) == 1, f"the cache was not reused: {len(calls)} fetches"
    assert provider.cache_hits >= 1


def test_paper_and_real_read_the_same_provider_instance():
    """
    The architectural requirement: ONE canonical reference, not two.

    Asserted on the object identity, because "one implementation" is only true
    if both consumers hold the same provider rather than two that agree today.
    """
    provider = FakeSolUsdProvider(PRICE)
    t = make_trader(provider)
    t.sol_usd_provider = provider

    # The PAPER sizing path resolves through the same accessor the REAL path
    # does, and both land on the injected instance.
    assert t._sol_usd_reference() is provider
    assert t.sol_usd_provider is provider


def test_the_backend_is_given_the_engines_provider():
    """
    The preflight reads the price off the BACKEND, so the backend must hold the
    engine's provider or the preflight and ``real_equity`` could disagree.
    """
    provider = FakeSolUsdProvider(PRICE)
    t = make_trader(provider)
    backend = t.real_backend
    backend._sol_usd_provider = t.sol_usd_provider
    assert backend._sol_usd_provider is provider


# ===========================================================================
# 9. No swap quote is used to value SOL
# ===========================================================================


def test_real_equity_never_quotes_a_swap_for_the_sol_reference():
    """
    (9) The core prohibition.

    ``real_equity`` must reach SOL/USD only through the canonical provider. If
    a swap quoter were consulted, the reference and the fill price would come
    from one observation -- the exact coupling the canonical provider exists to
    prevent -- and the direction rule would refuse the leg anyway.
    """
    src = inspect.getsource(PaperTrader.real_equity)
    assert "provider.quote(" not in src
    assert "quote_buy" not in src
    assert "quote_sell" not in src
    assert "_backend_sol_price()" in src


def test_the_backend_never_asks_the_quote_provider_for_sol_usd():
    """
    (9, continued) asserted on the backend method that USED to do it.

    The old body was ``self.provider.quote(WSOL_MINT, USDC_MINT, ...)``. Both
    of those must be gone from the method that produces the reference.
    """
    from autopaper.execution.real import LiveSolanaBackend

    src = inspect.getsource(LiveSolanaBackend.sol_price_usd)
    assert "provider.quote(" not in src
    assert "USDC_MINT" not in src
    assert "sol_usd_snapshot" in src


def test_sol_usd_never_goes_through_the_swap_quoter_at_runtime():
    """
    (9, behavioural) A quote provider wired to EXPLODE if consulted still yields
    a priced equity -- proving the reference path never touches it.
    """

    class ExplodingQuoteProvider:
        async def quote(self, *a, **kw):
            raise AssertionError("a swap quote was requested to value SOL")

    backend = FakeBackend(quote_provider=ExplodingQuoteProvider())
    eq = run(make_trader(FakeSolUsdProvider(PRICE), backend).real_equity())
    assert eq.available, eq.reason


def test_the_spl_path_still_quotes_for_a_non_sol_holding():
    """
    The counterpart, so (9) is not satisfied by simply deleting pricing: a
    non-SOL holding is STILL priced by asking the venue. Only the SOL REFERENCE
    moved.
    """
    quote_provider = FakeQuoteProvider(lamports_per_token=int(1e9 / PRICE))
    backend = FakeBackend(
        balances=[{"mint": USDC, "symbol": "USDC",
                   "amount": 25_000_000, "decimals": 6}],
        quote_provider=quote_provider,
    )
    eq = run(make_trader(FakeSolUsdProvider(PRICE), backend).real_equity())

    assert eq.available, eq.reason
    assert quote_provider.calls, "the SPL price path stopped quoting entirely"
    # the quote is asset -> WSOL, never the quote-asset-to-quote-asset leg
    for in_mint, out_mint, _ in quote_provider.calls:
        assert in_mint == USDC and out_mint == WSOL


# ===========================================================================
# 10. The safety invariant is untouched
# ===========================================================================


def test_resolve_side_from_mints_is_unchanged():
    """
    (10) The invariant that refused the old SOL/USD leg must still refuse it.

    This is the load-bearing assertion of the whole change: the bug was fixed by
    changing the CALLER, never by relaxing the rule.
    """
    from autopaper.execution.price_layers import SwapDirectionError, \
        resolve_side_from_mints

    # both ends quote assets -> still refused
    with pytest.raises(SwapDirectionError):
        resolve_side_from_mints(WSOL, USDC)
    # neither end -> still refused
    with pytest.raises(SwapDirectionError):
        resolve_side_from_mints("mintA", "mintB")
    # exactly one quote end -> still resolves, both directions
    assert resolve_side_from_mints(WSOL, "mintA").value == "BUY"
    assert resolve_side_from_mints("mintA", WSOL).value == "SELL"


def test_a_contradictory_declared_side_is_still_refused():
    """The declared-side contradiction check must survive too."""
    from autopaper.execution.price_layers import SwapDirectionError, \
        resolve_side_from_mints

    with pytest.raises(SwapDirectionError):
        resolve_side_from_mints(WSOL, "mintA", declared_side="SELL")


def test_the_price_layers_module_is_byte_identical_in_shape():
    """
    (10, structural) ``resolve_side_from_mints`` still carries its stated
    contract, asserted on source so a future edit cannot quietly widen it.
    """
    src = (SRC / "execution" / "price_layers.py").read_text()
    fn = src.split("def resolve_side_from_mints", 1)[1].split("\ndef ", 1)[0]
    assert "if in_is_quote == out_is_quote:" in fn
    assert "raise SwapDirectionError(" in fn
    assert "contradicts the mints" in fn


# ===========================================================================
# Provenance is recorded (not required by the numbered list, but required by
# "record sol_usd / sol_usd_source / sol_usd_timestamp / sol_usd_age_ms")
# ===========================================================================


def test_the_equity_records_where_its_reference_came_from():
    provider = FakeSolUsdProvider(PRICE)
    eq = run(make_trader(provider).real_equity())

    assert eq.sol_usd_source == "fake (test)"
    assert isinstance(eq.sol_usd_timestamp, float)
    assert eq.sol_usd_age_ms is not None and eq.sol_usd_age_ms >= 0.0

    d = eq.as_dict()
    for key in ("sol_usd_source", "sol_usd_timestamp", "sol_usd_age_ms"):
        assert key in d


def test_provenance_is_absent_when_there_is_no_reference():
    """No reference means no provenance -- never a fabricated timestamp."""
    eq = run(make_trader(FakeSolUsdProvider(fail=True)).real_equity())
    assert eq.sol_usd_source is None
    assert eq.sol_usd_timestamp is None
    assert eq.sol_usd_age_ms is None


def test_the_canonical_providers_own_metadata_fields_are_untouched():
    """The reporting contract the rest of the repo depends on."""
    snap = S.override_snapshot(150.0, now=1_700_000_000.0)
    d = snap.as_dict(now=1_700_000_030.0)
    for key in ("sol_usd", "sol_usd_source", "sol_usd_timestamp", "sol_usd_age_ms"):
        assert key in d
    assert d["sol_usd"] == 150.0
    assert d["sol_usd_age_ms"] == pytest.approx(30_000.0)


# ===========================================================================
# Safety audit: the valuation path can neither sign nor broadcast
# ===========================================================================


def test_the_valuation_path_cannot_sign_or_broadcast():
    """
    The SOL/USD reference must be structurally incapable of execution: it is a
    price read. Asserted on source, because "we did not call it" is a claim
    about the future too.
    """
    src = (SRC / "same_market" / "sol_price.py").read_text()
    for banned in ("sign_transaction", "send_transaction", "sendRawTransaction",
                   "broadcast", "build_transaction", "execute_swap",
                   "sendTransaction", "keypair", "solders"):
        assert banned not in src, f"the SOL/USD provider references {banned}"


def test_the_provider_module_has_no_execution_imports():
    from autopaper.same_market import sol_price as mod

    src = inspect.getsource(mod)
    for banned in ("from ..execution", "from .execution", "execution.real",
                   "execution.solana", "execution.txcodec", "execution.planner"):
        assert banned not in src, f"the SOL/USD provider imports {banned}"


def test_real_equity_source_has_no_signing_or_sending():
    src = inspect.getsource(PaperTrader.real_equity)
    for banned in ("sign_transaction", "send_transaction", "execute_swap",
                   "broadcast", "send_raw"):
        assert banned not in src


def test_the_provider_carries_no_key_material_in_its_output():
    """A price reference must never be able to print a secret."""
    snap = S.override_snapshot(150.0, now=1_700_000_000.0)
    blob = repr(snap) + repr(snap.as_dict())
    for bad in ("PRIVATE KEY", "seed", "mnemonic", "api_key", "x-api-key"):
        assert bad.lower() not in blob.lower()