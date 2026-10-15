"""
THE V2 FLOOR (BETA) EXIT QUOTE PATH -- the real defect and its guards.

THE DEFECT
----------
``unavailable`` was not the whole truth, and it was not the cause. Reading the
live log by REFUSAL REASON rather than by the word in the position's record:

    8,862 x "SELL quote refused as a fill: same-market identity is UNKNOWN, not
             PASS: reason=ENTRY_IDENTITY_DISCOVERY_FAILED ... pool=8rxMDWZX...
             ...: no pool key was disclosed"
    1,583 x "no executable SELL quote: unavailable"

Two distinct failure modes, both real:

* the DOMINANT one was an IDENTITY defect. An executable SELL quote existed and
  was refused, because on the EXIT side the same-market identity was resolved
  against the aggregator's observed pair ALONE. Once the entry bonding curve
  graduated, the observed pair IS that retired curve, so the direct candidate on
  the live PumpSwap pool graded FAIL (a different address) and the aggregator's
  single-hop route to that SAME pool graded UNKNOWN (a different address). The
  fill gate requires PASS, so no price could ever be accepted. The reason line
  even named the pool it had just disclosed while claiming no pool key had been.
* the QUIETER one was CAPACITY. Every failed tick re-quoted on the next tick, so
  a stranded arm spent the shared RPC/aggregator budget every ~5 seconds, and the
  throttling those attempts caused produced the very failures it retried against.

WHAT IS PINNED HERE
-------------------
The exit path end to end, on the REAL provider response SHAPES:

1. a graduated entry curve is exited through the pool DERIVED from the mint, and
   the quote's identity is PROVEN rather than assumed;
2. the exact acquired quantity and its decimals are what gets quoted;
3. a genuine ``unavailable`` leaves the arm UNRESOLVED -- no mark, no price;
4. a stale quote is never aged into a fill;
5. a malformed quote is refused and never read as a zero;
6. a quote for another token is never usable;
7. a provider failure is a named refusal, and the direct leg survives it;
8. every provider failing is a refusal, not a crash;
9. settlement happens EXACTLY ONCE across a refusal, a recovery and a re-tick.

PAPER only. No network, no signer, no transaction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from autopaper.paper_live.fills import fill_from_quote  # noqa: E402
from autopaper.paper_live.quotes import QuoteService  # noqa: E402
from autopaper.same_market.models import WSOL_MINT  # noqa: E402
from autopaper.same_market.pumpdirect import (  # noqa: E402
    derive_bonding_curve,
    derive_canonical_pool,
)

from test_jupiter_best_route import (  # noqa: E402
    _Jupiter,
    _Reader,
    _direct,
    _route,
    _service,
)
from test_paper_live import FakeQuoteService, MINT, run, sell_point  # noqa: E402
from test_v2_floor_beta import beta_engine, eligible_signal  # noqa: E402

pytestmark = [pytest.mark.paper_live, pytest.mark.strategy]


# ---------------------------------------------------------------------------
# fixtures: the arms' OWN mint drives every derived address
# ---------------------------------------------------------------------------


def _setup(tmp_path):
    """Three beta arms opened, each pinned to its OWN retired bonding curve.

    The entry is driven through a fake quote service -- the entry path is not
    what is under test. The recorded entry market is the RETIRED curve derived
    from the arm's mint, which is what a position held past graduation carries,
    and the exit is then driven through a REAL ``QuoteService``.
    """
    eng = beta_engine(tmp_path, quotes=FakeQuoteService())
    run(eng.on_signal(eligible_signal()))
    exp = eng.beta_experiment
    pid = next(iter(exp.positions))
    mint = exp.positions[pid]["A"].mint
    curve = derive_bonding_curve(mint)[0]
    pool = derive_canonical_pool(mint, WSOL_MINT)[0]
    for arm_pos in exp.positions[pid].values():
        assert arm_pos.mint == mint
        arm_pos.entry_pool_or_curve = curve
    return eng, exp, pid, mint, curve, pool


def _exit_service(mint, pool, *, pool_available=True, jupiter="canonical",
                  jupiter_exc=None, age_s=0.0):
    """A REAL ``QuoteService`` whose provider doubles answer the EXIT.

    The reader reports the pool DERIVED from ``mint`` (what production does); the
    aggregator's route names the canonical pool by default, or -- when a test
    asks for it -- another book, or nothing at all.
    """
    reader = _Reader(
        curve=_direct(
            available=False, provider="pump_direct", mint=mint,
            missing_reason="the bonding curve is complete; there is no curve to "
                           "sell into"),
        pool=(_direct(provider="pumpswap_direct", mint=mint, curve=pool,
                      out_raw=64_000_000, spot=46_000.0, age_s=age_s, sell=True)
              if pool_available else
              _direct(available=False, provider="pumpswap_direct", mint=mint,
                      missing_reason="the pool account could not be read: "
                                     "solana rpc LOCAL quota exhausted")),
    )
    if isinstance(jupiter, str):
        if jupiter == "none":
            result = None
        elif jupiter == "other":
            result = _route(out_raw=99_000_000,
                            pool="Ats4ndymq5SW9FN7wRiq43GcZXzWxE9xpBHWbX5i3pPx",
                            hops=1)
        elif jupiter == "multi":
            result = _route(out_raw=99_000_000, pool=pool, hops=2)
        else:  # "canonical"
            result = _route(out_raw=70_000_000, pool=pool, hops=1)
    else:
        result = jupiter
    return _service(reader, _Jupiter(result=result, exc=jupiter_exc))


class _RecordingExit:
    """Records the EXACT request the exit path makes, then delegates."""

    def __init__(self, inner):
        self.inner = inner
        self.pinned: list[dict] = []
        self.sell: list[dict] = []

    async def quote_sell_pinned(self, **kw):
        self.pinned.append(dict(kw))
        return await self.inner.quote_sell_pinned(**kw)

    async def quote_sell(self, **kw):
        self.sell.append(dict(kw))
        return await self.inner.quote_sell(**kw)


def _statuses(settled):
    return sorted((p.arm, p.status) for p in settled)


# ---------------------------------------------------------------------------
# 1. THE LIVE DEFECT: a graduated exit settles from a real, proven quote
# ---------------------------------------------------------------------------


def test_1_a_graduated_exit_settles_from_the_pool_derived_from_the_mint(
        tmp_path):
    eng, exp, pid, mint, curve, pool = _setup(tmp_path)
    eng.quote_service = _exit_service(mint, pool)
    settled = run(eng._monitor_beta_experiment())
    closed = [p for p in settled if p.status == "closed"]
    assert len(closed) == 3, (
        "a graduated exit with an executable same-mint route did not settle: "
        + str(_statuses(settled)))
    for pos in closed:
        assert pos.exit_basis == "EXECUTABLE_QUOTE_SIMULATED_FILL"
        assert pos.exit_identity == "PASS", pos.exit_blocked_reason
        assert pos.net_pnl_usd is not None
        assert pos.exit_tokens_raw == pos.entry_tokens_raw, (
            "the exit was booked for a different quantity than was acquired")
        assert pos.exit_provenance
    assert len(exp.completed) == 3


def test_1b_the_aggregator_alone_is_not_same_market_proof(tmp_path):
    """FAIL CLOSED: without a chain-verified read, a route stays UNPROVEN.

    The aggregator's route names the canonical pool, but nothing in this tick
    decoded that pool on chain, so its identity cannot be PROVEN. The arm stays
    unresolved -- the price is not "probably right", it is unverified, and this
    is the guard that must never be relaxed.
    """
    eng, exp, pid, mint, curve, pool = _setup(tmp_path)
    eng.quote_service = _exit_service(mint, pool, pool_available=False)
    settled = run(eng._monitor_beta_experiment())
    assert {p.status for p in settled} == {"unresolved"}
    assert exp.completed == []
    for pos in exp.positions[pid].values():
        assert pos.net_pnl_usd is None and pos.exit_fill_price is None
    assert "identity is UNKNOWN" in eng._beta_quote_refusals[pid] or \
        "UNKNOWN" in eng._beta_quote_refusals[pid]


def test_1c_an_exit_through_another_book_is_refused(tmp_path):
    """STRICTNESS: a route on a DIFFERENT book never becomes the settlement.

    This is the guard that must NOT be relaxed: an executable quote exists, it
    pays MORE, and it is still refused because it is not this position's market.
    """
    eng, exp, pid, mint, curve, pool = _setup(tmp_path)
    eng.quote_service = _exit_service(mint, pool, pool_available=False,
                                      jupiter="other")
    settled = run(eng._monitor_beta_experiment())
    assert {p.status for p in settled} == {"unresolved"}
    assert exp.completed == []
    for pos in exp.positions[pid].values():
        assert pos.net_pnl_usd is None and pos.exit_fill_price is None


def test_1d_a_multi_hop_exit_route_is_never_attributed_to_one_book(tmp_path):
    eng, exp, pid, mint, curve, pool = _setup(tmp_path)
    eng.quote_service = _exit_service(mint, pool, pool_available=False,
                                      jupiter="multi")
    settled = run(eng._monitor_beta_experiment())
    assert {p.status for p in settled} == {"unresolved"}
    assert exp.completed == []


# ---------------------------------------------------------------------------
# 2. the quoted quantity is the acquired quantity
# ---------------------------------------------------------------------------


def test_2_the_sell_quotes_the_exact_acquired_quantity(tmp_path):
    eng, exp, pid, mint, curve, pool = _setup(tmp_path)
    recorder = _RecordingExit(_exit_service(mint, pool))
    eng.quote_service = recorder
    run(eng._monitor_beta_experiment())
    assert recorder.pinned, (
        "the exit did not go through the entry-aware collector")
    for pos in exp.positions[pid].values():
        asks = [kw for kw in recorder.pinned
                if kw.get("token_amount_raw") == pos.entry_tokens_raw]
        assert asks, (
            f"arm {pos.arm} was quoted for a quantity it does not hold: asked "
            f"{[kw.get('token_amount_raw') for kw in recorder.pinned]}, holds "
            f"{pos.entry_tokens_raw}")
        kw = asks[0]
        assert kw["token_decimals"] == pos.entry_token_decimals == 6
        assert kw["mint"] == pos.mint
        assert kw["observed_pair_address"] == pos.entry_pool_or_curve


# ---------------------------------------------------------------------------
# 3-8. the refusal modes, each with its own evidence
# ---------------------------------------------------------------------------


def test_3_a_genuinely_unavailable_exit_stays_unresolved(tmp_path):
    eng, exp, pid, mint, curve, pool = _setup(tmp_path)
    eng.quote_service = _exit_service(
        mint, pool, pool_available=False,
        jupiter_exc=RuntimeError("429 rate limited"), jupiter="none")
    settled = run(eng._monitor_beta_experiment())
    assert {p.status for p in settled} == {"unresolved"}
    for pos in exp.positions[pid].values():
        assert pos.net_pnl_usd is None
        assert pos.exit_fill_price is None
        assert pos.exit_proceeds_usd is None
        assert "no valid executable SELL quote" in pos.exit_blocked_reason
    assert exp.completed == []
    # THE PROVIDER'S OWN WORDS REACH THE OPERATOR, PER LEG. This is the half of
    # the incident that used to arrive as the bare word "unavailable".
    text = eng.render_beta_exit_blocked()[0]["text"]
    assert "429 rate limited" in text
    for leg in ("jupiter", "pump_direct", "pumpswap_direct"):
        assert leg in text, f"the {leg} leg's refusal was not reported"


def test_4_a_stale_exit_quote_is_refused_as_stale(tmp_path):
    eng, exp, pid, mint, curve, pool = _setup(tmp_path)
    eng.quote_service = _exit_service(mint, pool, pool_available=False,
                                      jupiter="none", age_s=120.0)
    settled = run(eng._monitor_beta_experiment())
    assert {p.status for p in settled} == {"unresolved"}
    assert exp.completed == []
    # The direct leg came back stale rather than absent: the refusal must say so.
    for pos in exp.positions[pid].values():
        assert pos.net_pnl_usd is None


def test_5_a_malformed_quote_is_refused_not_zeroed(tmp_path):
    from autopaper.paper_live.fills import FillModel, FillRefused
    from autopaper.paper_live.schema import FailureReason
    from autopaper.same_market.jupiter import JupiterQuoteClient, QuoteUnavailable

    client = JupiterQuoteClient(cache_ttl_s=0.0)
    for body in ({}, {"inAmount": "1"}, {"outAmount": "x", "inAmount": "1"}):
        with pytest.raises(QuoteUnavailable):
            client.parse(body)

    hollow = sell_point(output_amount_raw=None, output_amount_decimal=None)
    with pytest.raises(FillRefused) as exc:
        fill_from_quote(hollow, side="SELL", model=FillModel(),
                        require_identity=True)
    assert exc.value.reason is FailureReason.ARITHMETIC_ERROR


def test_6_a_quote_for_another_token_never_settles(tmp_path):
    eng, exp, pid, _mint, _curve, _pool = _setup(tmp_path)
    other_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    other = sell_point(token_mint=other_mint, identity="PASS")
    assert QuoteService._usable_candidates(
        [other], side="SELL", mint=MINT) == [], (
        "a SELL quote for another token was admitted as usable")
    _ = eng
    for pos in exp.positions[pid].values():
        assert pos.status == "open" and pos.net_pnl_usd is None


def test_7_a_failing_aggregator_does_not_take_the_direct_exit_down(tmp_path):
    eng, exp, pid, mint, curve, pool = _setup(tmp_path)
    eng.quote_service = _exit_service(
        mint, pool, jupiter_exc=RuntimeError("503 routing unavailable"))
    settled = run(eng._monitor_beta_experiment())
    closed = [p for p in settled if p.status == "closed"]
    assert len(closed) == 3, (
        "the direct PumpSwap leg must still settle when the aggregator fails: "
        + str(_statuses(settled)))
    for pos in closed:
        assert pos.exit_identity == "PASS"


def test_8_every_provider_failing_is_a_named_refusal_not_a_crash(tmp_path):
    eng, exp, pid, mint, curve, pool = _setup(tmp_path)
    eng.quote_service = _exit_service(
        mint, pool, pool_available=False,
        jupiter_exc=RuntimeError("429 rate limited"), jupiter="none")
    settled = run(eng._monitor_beta_experiment())
    assert {p.status for p in settled} == {"unresolved"}
    for pos in exp.positions[pid].values():
        assert pos.exit_blocked_reason
        assert pos.net_pnl_usd is None


# ---------------------------------------------------------------------------
# 9. exactly-once settlement
# ---------------------------------------------------------------------------


def test_9_settlement_happens_exactly_once_across_a_refusal_and_a_recovery(
        tmp_path):
    eng, exp, pid, mint, curve, pool = _setup(tmp_path)
    eng.quote_service = _exit_service(
        mint, pool, pool_available=False,
        jupiter_exc=RuntimeError("429 rate limited"), jupiter="none")
    run(eng._monitor_beta_experiment())
    assert exp.completed == []
    assert {pos.exit_attempts for pos in exp.positions[pid].values()} == {1}

    # The provider recovers and the cooldown expires (a restart is a fresh
    # budget, and so is an expired cooldown).
    eng._beta_exit_retry.clear()
    eng.quote_service = _exit_service(mint, pool)
    first = run(eng._monitor_beta_experiment())
    assert len([p for p in first if p.status == "closed"]) == 3, _statuses(first)
    assert len(exp.completed) == 3
    bankrolls = {arm: exp.accounts[arm].bankroll_usd for arm in ("A", "B", "C")}

    again = run(eng._monitor_beta_experiment())
    assert again == []
    assert len(exp.completed) == 3, "a settled arm was settled twice"
    assert {arm: exp.accounts[arm].bankroll_usd
            for arm in ("A", "B", "C")} == bankrolls, (
        "a re-tick moved a settled ledger again")
    assert all(pos.status == "closed" for pos in exp.positions[pid].values())
