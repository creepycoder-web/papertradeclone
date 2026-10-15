"""
Freshness, missing state and rate limits (test items 19, 20, 21 and 23).

The theme: an observation with no bound on its age is not a current
observation, and a failure is never quietly rendered as a zero or as "fresh".
"""


from __future__ import annotations

import asyncio

import pytest

from autopaper.same_market.identity import compare_mark_to_executable
from autopaper.same_market.jupiter import (
    JupiterQuoteClient,
    ProviderRateLimited,
    QuoteUnavailable,
)
from autopaper.same_market.rate_limit import RateLimiterProbe
from autopaper.same_market.models import (
    WSOL_MINT,
    CheckState,
    Decision,
    FailureMode,
    Freshness,
    IdentityChecks,
    StalenessPolicy,
)
from autopaper.same_market.rpc import RpcRateLimited, SolanaReadOnlyRpc

pytestmark = pytest.mark.same_market

WSOL = WSOL_MINT
MINT = "H9XQrbw5MANzYVgHjXJbScFxWqyGw9CCAec5XDTgpump"
MARK = 1.783e-05
NOW = 1_000.000


def _all_pass() -> IdentityChecks:
    return IdentityChecks(CheckState.PASS, CheckState.PASS, CheckState.PASS,
                          CheckState.PASS, CheckState.PASS, True)


# ---------------------------------------------------------------------------
# freshness stamps
# ---------------------------------------------------------------------------

def test_a_fresh_observation_is_fresh():
    policy = StalenessPolicy(max_mark_age_ms=1_000.0)
    stamp = policy.stamp("mark", NOW - 0.2, NOW, policy.max_mark_age_ms)
    assert stamp.freshness is Freshness.FRESH
    assert stamp.age_ms == pytest.approx(200.0)


def test_an_old_observation_is_stale():
    policy = StalenessPolicy(max_mark_age_ms=1_000.0)
    stamp = policy.stamp("mark", NOW - 5.0, NOW, policy.max_mark_age_ms)
    assert stamp.freshness is Freshness.STALE
    assert stamp.age_ms == pytest.approx(5_000.0)


def test_an_unconfigured_window_is_unknown_not_fresh():
    """
    The distinction that matters: the operator did not ask for a bound, so the
    harness does not know. Reporting UNKNOWN as FRESH would be the failure this
    harness exists to catch.
    """
    policy = StalenessPolicy()
    stamp = policy.stamp("mark", NOW, NOW, policy.max_mark_age_ms)
    assert stamp.freshness is Freshness.UNKNOWN
    assert stamp.freshness is not Freshness.FRESH


def test_a_missing_observation_is_missing_not_zero():
    policy = StalenessPolicy(max_mark_age_ms=1_000.0)
    stamp = policy.stamp("mark", None, NOW, policy.max_mark_age_ms)
    assert stamp.freshness is Freshness.MISSING
    assert stamp.age_ms is None


def test_the_window_is_configurable_per_observation():
    policy = StalenessPolicy(max_mark_age_ms=1_000.0,
                             max_state_age_ms=10_000.0,
                             max_quote_age_ms=500.0)
    stamp = policy.stamp("state", NOW - 5.0, NOW, policy.max_state_age_ms)
    assert stamp.freshness is Freshness.FRESH


# ---------------------------------------------------------------------------
# 19. stale mark rejection
# ---------------------------------------------------------------------------

def test_a_stale_mark_rejects_and_names_itself():
    verdict = compare_mark_to_executable(
        observed_mark=MARK, executable_price=MARK,
        threshold_pct=5.0, identity=_all_pass(),
        mark_freshness=Freshness.STALE,
    )
    assert verdict.decision is Decision.REJECT
    assert FailureMode.MARK_STALE.value in verdict.reason


# ---------------------------------------------------------------------------
# 20. stale quote rejection
# ---------------------------------------------------------------------------

def test_a_stale_quote_rejects_and_names_itself():
    verdict = compare_mark_to_executable(
        observed_mark=MARK, executable_price=MARK,
        threshold_pct=5.0, identity=_all_pass(),
        quote_freshness=Freshness.STALE,
    )
    assert verdict.decision is Decision.REJECT
    assert FailureMode.QUOTE_STALE.value in verdict.reason


def test_both_stale_is_still_one_clean_rejection():
    verdict = compare_mark_to_executable(
        observed_mark=MARK, executable_price=MARK,
        threshold_pct=5.0, identity=_all_pass(),
        mark_freshness=Freshness.STALE, quote_freshness=Freshness.STALE,
    )
    assert verdict.decision is Decision.REJECT
    assert FailureMode.MARK_STALE.value in verdict.reason
    assert FailureMode.QUOTE_STALE.value in verdict.reason


def test_unconfigured_freshness_does_not_reject_but_is_reported():
    verdict = compare_mark_to_executable(
        observed_mark=MARK, executable_price=MARK,
        threshold_pct=5.0, identity=_all_pass(),
        mark_freshness=Freshness.UNKNOWN, quote_freshness=Freshness.UNKNOWN,
    )
    assert verdict.decision is Decision.PASS
    assert "freshness is UNKNOWN, not assumed good" in verdict.reason


def test_a_fresh_pair_with_a_configured_threshold_passes():
    verdict = compare_mark_to_executable(
        observed_mark=MARK, executable_price=MARK * 1.01,
        threshold_pct=5.0, identity=_all_pass(),
        mark_freshness=Freshness.FRESH, quote_freshness=Freshness.FRESH,
    )
    assert verdict.decision is Decision.PASS


# ---------------------------------------------------------------------------
# 21. missing state rejection
# ---------------------------------------------------------------------------

def test_a_missing_mark_is_a_rejection_with_no_gap():
    verdict = compare_mark_to_executable(
        observed_mark=None, executable_price=MARK,
        threshold_pct=5.0, identity=_all_pass(),
        mark_freshness=Freshness.MISSING,
    )
    assert verdict.decision is Decision.REJECT
    assert verdict.gap_pct is None
    assert "no observed mark" in verdict.reason


def test_a_missing_quote_is_a_rejection_with_no_gap():
    verdict = compare_mark_to_executable(
        observed_mark=MARK, executable_price=None,
        threshold_pct=5.0, identity=_all_pass(),
        quote_freshness=Freshness.MISSING,
    )
    assert verdict.decision is Decision.REJECT
    assert verdict.gap_pct is None
    assert "proves nothing" in verdict.reason


def test_a_missing_measurement_is_reject_even_with_no_threshold():
    for mark, price in ((None, None), (None, MARK), (MARK, None)):
        verdict = compare_mark_to_executable(
            observed_mark=mark, executable_price=price, threshold_pct=None,
            identity=_all_pass(),
        )
        assert verdict.decision is Decision.REJECT


# ---------------------------------------------------------------------------
# 23. provider rate-limit handling
# ---------------------------------------------------------------------------

def test_jupiter_rate_limit_surfaces_as_a_named_failure():
    client = JupiterQuoteClient(http_get=lambda p: (_ for _ in ()).throw(
        ProviderRateLimited("429")))
    with pytest.raises(ProviderRateLimited):
        asyncio.run(client.quote_buy(token_mint=MINT, quote_asset_mint=WSOL,
                                     in_amount=1_150_000, output_decimals=6))


def test_a_venue_rate_limit_is_not_retried_into_a_wrong_answer():
    """Bounded retries: the harness backs off, it does not hammer."""
    attempts: list[int] = []

    def _get(params):
        attempts.append(1)
        raise ProviderRateLimited("429")

    client = JupiterQuoteClient(http_get=_get, max_retries=1)
    with pytest.raises(ProviderRateLimited):
        asyncio.run(client.quote_buy(token_mint=MINT, quote_asset_mint=WSOL,
                                     in_amount=1_150_000, output_decimals=6))
    assert 1 <= len(attempts) <= 3, "retries must be bounded, not a hot loop"


def test_a_venue_refusing_the_size_is_a_named_failure():
    client = JupiterQuoteClient(
        http_get=lambda p: (_ for _ in ()).throw(QuoteUnavailable("TOO_SMALL")))
    with pytest.raises(QuoteUnavailable) as exc:
        asyncio.run(client.quote_buy(token_mint=MINT, quote_asset_mint=WSOL,
                                     in_amount=1_150_000, output_decimals=6))
    assert "TOO_SMALL" in str(exc.value)


def test_the_rpc_rate_limit_is_a_failure_mode_distinct_from_a_provider_one():
    """
    They have different causes and different remedies: the shared public node
    being overloaded versus a data provider throttling us. One label would hide
    which knob to turn.
    """
    from autopaper.same_market.pipeline import _classify_provider_error

    rpc_error = RpcRateLimited("429")
    rpc_error.retry_after_s = 5.0
    assert _classify_provider_error(rpc_error) is FailureMode.RPC_RATE_LIMIT
    assert (_classify_provider_error(ProviderRateLimited("429"))
            is FailureMode.PROVIDER_RATE_LIMIT)
    assert (FailureMode.RPC_RATE_LIMIT
            is not FailureMode.PROVIDER_RATE_LIMIT)


def test_the_local_rate_cap_is_a_named_condition_not_a_silent_drop():
    probe = RateLimiterProbe()
    assert probe.rate_limited_is_named()
    assert "rate cap" in probe.reason


def test_a_provider_backoff_is_honoured_locally():
    """A Retry-After from a provider blocks locally until it has elapsed."""
    probe = RateLimiterProbe(max_per_minute=60)
    probe.note_retry_after(60.0)          # blocked until t=1060
    assert probe.allow_after_backoff(1000.0) is False
    assert probe.allow_after_backoff(1059.0) is False
    assert probe.allow_after_backoff(1060.0) is True


def test_the_rpc_client_refuses_a_non_read_method():
    """
    A belt-and-braces assertion on top of the class simply not having one.
    """
    rpc = SolanaReadOnlyRpc("https://example.invalid")
    assert "sendTransaction" not in rpc.read_methods
    assert "sendTransaction" not in dir(rpc)

    from autopaper.same_market.rpc import RpcUnavailable

    with pytest.raises(RpcUnavailable) as exc:
        asyncio.run(rpc._rpc("sendTransaction", []))
    assert "not a read method" in str(exc.value)


def test_an_rpc_rate_limit_is_an_exception_not_a_zero():
    error = RpcRateLimited("rate limited")
    error.retry_after_s = 12.0
    assert isinstance(error, Exception)
    assert error.retry_after_s == 12.0


def test_the_rpc_client_honours_a_local_rate_cap():
    rpc = SolanaReadOnlyRpc("https://example.invalid", max_rpm=2)
    with pytest.raises(Exception) as exc:
        for _ in range(10):
            rpc.limiter.acquire(1000.0)
    assert "rate cap" in str(exc.value)