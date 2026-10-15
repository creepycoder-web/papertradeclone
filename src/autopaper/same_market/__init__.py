"""
Same-market quote harness - a standalone, read-only measurement tool.

WHAT THIS IS
============

A CLI harness that answers one question about one Telegram signal:

    Is the price DexScreener marks for this token the SAME market that an
    execution would actually trade against, and how close is the executable
    quote to that mark for a small, fixed, hypothetical order?

WHAT THIS IS NOT
================

Not a trader. It has no capability to buy, sell, swap, transfer, sign or
broadcast. It does not modify L25, the strategy thresholds, the risk settings,
or any wallet or account state. It reads a channel, reads the chain, and prints
numbers.

The no-broadcast property is structural rather than procedural:

* :mod:`autopaper.same_market.rpc` exposes five read methods and no
  ``sendTransaction``
* :mod:`autopaper.same_market.jupiter` exposes a quote and no transaction
  builder
* :mod:`autopaper.same_market.guard` refuses to import any module containing a
  signer or a broadcaster

THE THREE VALUES, KEPT APART
============================

``OBSERVED_MARK`` is what DexScreener says. ``EXECUTABLE_QUOTE`` is what a
read-only quote against the real market says. ``ACTUAL_FILL`` does not exist
here and is reported as ``NOT AVAILABLE`` rather than being approximated by one
of the other two.

Importing this package installs the no-broadcast barrier, before any provider
module is loaded.
"""

from __future__ import annotations

from .guard import (
    NoBroadcastViolation,
    assert_no_execution_capability,
    audit_no_execution_capability,
    install_no_broadcast_guard,
)

# Armed at import time, before any provider module can be loaded.
install_no_broadcast_guard()

from .models import (  # noqa: E402
    ACTUAL_FILL_AVAILABLE,
    CheckState,
    Decision,
    ExecutableQuote,
    FailureMode,
    Freshness,
    IdentityChecks,
    MarketKind,
    ObservedMark,
    PoolState,
    SameMarketVerdict,
    StalenessPolicy,
    TokenCandidate,
)
from .numeric import DecimalError, effective_price, price_gap_pct  # noqa: E402
from .signals import extract_token  # noqa: E402
from .trace import LatencyBreakdown, StageTrace, build_breakdown  # noqa: E402

__all__ = [
    "ACTUAL_FILL_AVAILABLE",
    "CheckState",
    "DecimalError",
    "Decision",
    "ExecutableQuote",
    "FailureMode",
    "Freshness",
    "IdentityChecks",
    "LatencyBreakdown",
    "MarketKind",
    "NoBroadcastViolation",
    "ObservedMark",
    "PoolState",
    "SameMarketVerdict",
    "StageTrace",
    "StalenessPolicy",
    "TokenCandidate",
    "assert_no_execution_capability",
    "audit_no_execution_capability",
    "build_breakdown",
    "effective_price",
    "extract_token",
    "install_no_broadcast_guard",
    "price_gap_pct",
]