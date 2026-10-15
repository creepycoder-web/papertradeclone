"""
REAL equity: native SOL balance x SOL/USD, or an explicit refusal.

The bug this pins: the armed-REAL message rendered "Equity: $0.0000" for a
wallet holding SOL, because the only caller passed a literal placeholder and
omitted the equity entirely. So nothing measured a balance or a price - the
number was never a reading.

Every test here uses injected values or a GENERATED keypair. The production
wallet is never opened, no key material is read, and nothing is broadcast.
"""

from __future__ import annotations

import pytest

from autopaper.real_equity import (
    LAMPORTS_PER_SOL,
    EquityUnavailable,
    RealEquity,
    compute_real_equity,
    lamports_to_sol,
)

# The balance the operator reported: 0.010718431 SOL
REPORTED_LAMPORTS = 10_718_431
REPORTED_SOL = 0.010718431


# ---------------------------------------------------------------------------
# lamports -> SOL
# ---------------------------------------------------------------------------


def test_the_reported_balance_converts_exactly():
    assert lamports_to_sol(REPORTED_LAMPORTS) == REPORTED_SOL


def test_lamports_per_sol_is_the_standard_value():
    assert LAMPORTS_PER_SOL == 1_000_000_000


def test_zero_lamports_is_a_real_zero():
    assert lamports_to_sol(0) == 0.0


@pytest.mark.parametrize("bad", [
    None, True, False, -1, -1_000, "abc", "100", 1.5, [], {}, object(),
])
def test_unusable_balances_are_refused_not_coerced(bad):
    assert lamports_to_sol(bad) is None


def test_a_whitespace_padded_integer_string_is_refused():
    """Deliberate: a balance arriving as a string is a bug upstream, not a
    value to be salvaged."""
    assert lamports_to_sol(" 100 ") is None


# ---------------------------------------------------------------------------
# equity
# ---------------------------------------------------------------------------


def test_the_reported_wallet_produces_a_real_equity():
    r = compute_real_equity(REPORTED_LAMPORTS, 180.0, "wUEU…Wbt6")
    assert r.available is True
    assert r.sol == REPORTED_SOL
    assert r.equity_usd == pytest.approx(REPORTED_SOL * 180.0)
    assert r.equity_usd > 0, "a funded wallet must never read as $0"


def test_equity_is_the_product_not_a_constant():
    """Two different prices must give two different equities."""
    a = compute_real_equity(REPORTED_LAMPORTS, 100.0).equity_usd
    b = compute_real_equity(REPORTED_LAMPORTS, 200.0).equity_usd
    assert a == pytest.approx(REPORTED_SOL * 100.0)
    assert b == pytest.approx(REPORTED_SOL * 200.0)
    assert a != b


def test_a_genuinely_empty_wallet_is_a_true_zero():
    """Zero balance and a GOOD price is a real, measured $0.00."""
    r = compute_real_equity(0, 180.0)
    assert r.available is True
    assert r.equity_usd == 0.0
    assert r.reason == ""


def test_a_funded_wallet_never_becomes_zero_by_initialisation():
    """The regression in one assertion: non-zero balance, good price -> > 0."""
    for price in (0.01, 1.0, 50.0, 180.0, 10_000.0):
        r = compute_real_equity(REPORTED_LAMPORTS, price)
        assert r.available, price
        assert r.equity_usd > 0, f"price {price} produced {r.equity_usd}"


# ---------------------------------------------------------------------------
# fail closed
# ---------------------------------------------------------------------------


def test_a_missing_price_is_unavailable_not_zero():
    r = compute_real_equity(REPORTED_LAMPORTS, None)
    assert r.available is False
    assert r.equity_usd is None
    assert "SOL/USD" in r.reason


def test_a_zero_price_is_unavailable_not_zero():
    """A price of 0 is not a price; multiplying by it would fake a $0.00."""
    r = compute_real_equity(REPORTED_LAMPORTS, 0)
    assert r.available is False
    assert r.equity_usd is None


@pytest.mark.parametrize("bad", ["abc", float("nan"), float("inf"), -5.0, [1]])
def test_unusable_prices_are_unavailable(bad):
    r = compute_real_equity(REPORTED_LAMPORTS, bad)
    assert r.available is False
    assert r.equity_usd is None
    assert r.reason


def test_an_unreadable_balance_is_unavailable():
    r = compute_real_equity(None, 180.0)
    assert r.available is False
    assert "balance" in r.reason


def test_failure_is_distinguishable_from_a_real_zero():
    """The whole point: these two must never look the same."""
    real_zero = compute_real_equity(0, 180.0)
    failed = compute_real_equity(None, 180.0)
    assert real_zero.available is True and real_zero.equity_usd == 0.0
    assert failed.available is False and failed.equity_usd is None
    assert real_zero.as_dict() != failed.as_dict()


def test_no_default_price_is_ever_substituted():
    """No hardcoded fallback price, anywhere."""
    r = compute_real_equity(REPORTED_LAMPORTS, None)
    assert r.sol_price_usd == 0.0 and not r.available
    # a real price is passed through unchanged
    assert compute_real_equity(1, 123.45).sol_price_usd == 123.45


def test_the_sentinel_is_always_falsy():
    assert bool(EquityUnavailable("x")) is False


def test_as_dict_is_json_safe():
    import json
    for r in (compute_real_equity(1, 2.0), compute_real_equity(None, None)):
        json.dumps(r.as_dict())


def test_the_wallet_mask_survives():
    assert compute_real_equity(1, 2.0, "wUEU…Wbt6").wallet_masked == "wUEU…Wbt6"
    assert compute_real_equity(1, 2.0).wallet_masked == "(unavailable)"


def test_the_result_is_immutable():
    r = compute_real_equity(1, 2.0)
    with pytest.raises(Exception):
        r.equity_usd = 99.0     # type: ignore[misc]
