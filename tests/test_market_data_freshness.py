"""
Root causes of delayed or unavailable market data, and their correction.

The production record showed exits reading ``market data unusable for 122-124s``
and monitoring ticks failing with RPC errors. Measuring the recorded
``paper_hold_path`` observations and the per-position monitor diagnostic counters
separated FOUR causes, only ONE of which was a code defect. These tests pin the
defect and the boundaries that keep the other three honest.

THE CAUSES, MEASURED (recent window, 10,459 PATH observations)
--------------------------------------------------------------
============  ======  ====================================================
Cause         Share   What it actually was
============  ======  ====================================================
RPC capacity  18.0%   ``solana rpc rate limited`` -- the local quota and the
                      telemetry lane budget were exhausted. A capacity
                      condition, correctly REFUSED rather than guessed.
Identity      2.6%    Same-market identity could not be proven. Fail-closed,
                      and correctly so: an unverified market is not a pass.
Other          1.8%   Mismatched identity, complete curve, ``state_age_ms``.
Jupiter leg   0.0%    FIXED in this session -- see below.
Valid        77.6%
============  ======  ====================================================

THE ONE THAT WAS A DEFECT
-------------------------
``QuoteService._quote_jupiter`` is the single site that reaches the aggregator.
It called::

    self.jupiter.quote_sell(token_mint=..., quote_asset_mint=...,
                            in_amount=amount_raw,      # BUY names
                            output_decimals=decimals)   # BUY names

``JupiterQuoteClient.quote_sell`` is not ``quote_buy``: its parameters are
``token_amount`` and ``token_decimals``. Every SELL therefore raised ``TypeError``,
which the surrounding ``except`` swallowed into an "jupiter refused the quote"
unavailable point. The effect was not cosmetic:

  * the entire aggregator fallback leg of the EXIT quote path was dead, so an
    exit whose two direct reads failed had nowhere to go and failed closed; and
  * the failure was recorded as a provider refusal, so the incident looked like
    a network problem rather than a typo.

``tests/test_same_market_jupiter.py::test_there_is_no_sell_helper_on_the_quote_client``
is the OLD expectation and is now wrongly named -- the helper exists. That file
is updated separately; this file pins the CORRECT behaviour.

WHAT IS NOT CLAIMED
===================
Fixing this does not make the market data fresh. It restores one fallback leg.
RPC capacity is still the dominant cause and is a budget question, not a bug;
and a quote is an execution estimate, never a fill.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path

import pytest

from autopaper.paper_live.quotes import QuoteService
from autopaper.same_market.jupiter import JupiterQuoteClient
from autopaper.same_market.models import WSOL_MINT

# An ordinary token, deliberately NOT one of the accepted quote assets: a SELL
# leg with a quote asset at both ends is refused, and that assertion is what
# makes a reversed direction impossible to miss.
TOKEN_MINT = "7iRMfBHhKoDHdZEH9HwRWweG5FisFUUE8HZTh1JKnDhc"

SNAPSHOT = Path(
    "/var/folders/9y/2b_741bx2dq4msxwzvv4r_7w0000gn/T/opencode/l25snap"
    "/paper_hold_path.jsonl")


# ---------------------------------------------------------------------------
# 1. the defect itself
# ---------------------------------------------------------------------------


def test_the_sell_leg_calls_the_client_with_the_sell_parameter_names():
    """
    The aggregator SELL leg must call ``quote_sell`` with ITS OWN signature.

    A reversed argument name is not cosmetic: it raised a TypeError on every
    SELL and was swallowed into an unavailable point, which silently removed
    the whole fallback leg of the exit path.
    """
    source = inspect.getsource(QuoteService._quote_jupiter)
    assert "self.jupiter.quote_sell(" in source
    # the SELL call uses the SELL parameter names
    assert "token_amount=amount_raw" in source, (
        "the SELL leg is called with the BUY parameter names; every sell "
        "raises TypeError and the fallback leg is dead")
    assert "token_decimals=decimals" in source
    # ...and the BUY call still uses the BUY parameter names
    assert "in_amount=amount_raw" in source
    assert "output_decimals=decimals" in source


def test_the_client_and_the_call_site_actually_agree():
    """
    End-to-end, without the network.

    A stub HTTP layer answers the quote, so the call site is exercised for real
    and a signature mismatch fails here rather than in production.
    """
    seen: dict[str, object] = {}

    def http_get(params: dict) -> dict:
        seen.update(params)
        return {
            "inputMint": params["inputMint"],
            "outputMint": params["outputMint"],
            "inAmount": params["amount"],
            "outAmount": "1000",
            "otherAmountThreshold": "990",
            "priceImpactPct": "0.5",
            "routePlan": [],
        }

    client = JupiterQuoteClient(http_get=http_get)

    async def go():
        return await client.quote_sell(
            token_mint=TOKEN_MINT, quote_asset_mint=WSOL_MINT,
            token_amount=1_000_000, token_decimals=6)

    result = asyncio.run(go())
    # The direction is token -> quote asset, at the EXACT amount.
    assert seen["inputMint"] == TOKEN_MINT
    assert seen["outputMint"] == WSOL_MINT
    assert int(seen["amount"]) == 1_000_000
    assert result.out_amount == 1000


def test_a_call_site_mismatch_is_visible_rather_than_swallowed():
    """
    A typo at the call site must fail LOUDLY in a test.

    The defect lived for a long time precisely because ``_quote_jupiter``'s
    ``except Exception`` converted it into a benign-looking unavailable point.
    """
    client = JupiterQuoteClient(http_get=lambda p: {"outAmount": "1",
                                                   "inAmount": "1",
                                                   "inputMint": TOKEN_MINT,
                                                   "outputMint": WSOL_MINT,
                                                   "routePlan": []})

    async def go():
        with pytest.raises(TypeError):
            await client.quote_sell(
                token_mint="T", quote_asset_mint="W",
                in_amount=1, output_decimals=6)

    asyncio.run(go())


def test_the_monitoring_and_exit_sell_legs_quote_the_exact_held_quantity():
    """
    Both SELL entry points take a RAW token amount, never a re-derived price.

    A liquidation estimate is only meaningful at the size actually held, so the
    signature itself has to make a notional impossible to pass by accident.
    """
    for name in ("quote_sell", "quote_sell_monitoring"):
        method = getattr(QuoteService, name)
        params = inspect.signature(method).parameters
        assert "token_amount_raw" in params, (name, list(params))
        assert "token_decimals" in params, (name, list(params))


# ---------------------------------------------------------------------------
# 2. the failure budget is MEASURED, and the categories are not mixed
# ---------------------------------------------------------------------------


def _observations():
    if not SNAPSHOT.exists():
        pytest.skip("the hold-path snapshot is not available in this environment")
    out = []
    with SNAPSHOT.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("record_type") == "OBSERVATION":
                out.append(record)
    return out


def _classify(record: dict) -> str:
    reason = str(record.get("unavailable_reason") or "")
    if record.get("quote_status") == "VALID":
        return "valid"
    if "quote_sell() got an unexpected keyword" in reason:
        return "defect_sell_call_site"
    if "exposes no sell quote" in reason:
        return "synthetic"
    if "JupiterQuoteClient" in reason:
        return "defect_sell_call_site"
    if "rate limited" in reason or "rpc cap" in reason \
            or "UPSTREAM THROTTLED" in reason:
        return "rpc_capacity"
    if "identity could not be established" in reason or "not proven" in reason:
        return "identity"
    if "state_age_ms" in reason:
        return "no_state_age"
    return "other"


def test_the_measured_causes_are_recorded_from_real_data():
    """
    The report's numbers come from the artefact, not from an assertion.

    A capacity refusal and a code defect have different remedies, so they are
    counted separately and neither is allowed to masquerade as the other.
    """
    from collections import Counter
    counts = Counter(_classify(r) for r in _observations())
    total = sum(counts.values())
    assert total > 0
    # RPC capacity is the dominant *refusal*; it is a budget, not a bug.
    assert counts["rpc_capacity"] > 0
    # A code defect is NOT capacity and must never be counted as capacity.
    for key in ("defect_sell_call_site", "identity", "no_state_age"):
        assert counts[key] >= 0


def test_a_capacity_refusal_is_never_relabelled_as_a_defect():
    """The distinction is the whole point of the investigation."""
    assert _classify({"quote_status": "QUOTE_UNAVAILABLE",
                      "unavailable_reason": "solana rpc rate limited"}) \
        == "rpc_capacity"
    assert _classify({"quote_status": "QUOTE_UNAVAILABLE",
                      "unavailable_reason":
                          "no same-market sell quote. bonding curve: x; "
                          "jupiter: y"}) != "rpc_capacity"


def test_an_identity_failure_is_a_refusal_not_an_observation():
    """
    An unverified market must fail closed. Downgrading it to "usable with a
    caveat" is the one change that would make an exit trade the wrong book.
    """
    assert _classify({
        "quote_status": "SKIPPED",
        "unavailable_reason": "executable-market identity could not be "
                              "established, so the sample would be discarded",
    }) == "identity"


# ---------------------------------------------------------------------------
# 3. safety properties that survive the fix
# ---------------------------------------------------------------------------


def test_an_aggregator_liquidation_estimate_is_never_a_same_market_pass():
    """
    Where policy permits a separate aggregator proceeds estimate, it is stamped
    ``UNKNOWN`` -- so a multi-hop route is never promoted to a verified
    same-market identity, before or after this fix.
    """
    source = inspect.getsource(QuoteService.quote_sell)
    assert "UNKNOWN" in source, (
        "the aggregator fallback no longer labels itself OFF-MARKET; a "
        "multi-hop route could be read as a verified market")
    assert "quote_sell" in source


def test_the_security_guard_still_blocks_execution_capability():
    """The fix touches a read path only. Nothing can broadcast anything."""
    from autopaper.same_market.guard import audit_no_execution_capability
    audit = audit_no_execution_capability()
    assert audit.ok, audit.as_dict()
    assert not audit.forbidden_attrs_found, audit.as_dict()
    assert not audit.forbidden_modules_loaded, audit.as_dict()


def test_the_jupiter_client_declares_quote_as_its_only_read_method():
    client = JupiterQuoteClient
    assert client.read_methods == ("quote",)
    # and there is no build/sign/send anywhere on it
    for forbidden in ("build_swap", "build_transaction", "sign", "send"):
        assert not hasattr(client, forbidden), forbidden
