"""
EXECUTION-AWARE LIVE PAPER mode. PAPER ONLY.

WHAT THIS IS
============

A forward-validation harness that consumes NEW Telegram signals and simulates
trades using FRESH EXECUTABLE QUOTES instead of DexScreener observed marks, so
that the resulting dataset can later support an execution-aware backtest of L25
and the neighbouring research strategies.

WHY IT EXISTS
=============

The historical PAPER model was mark-based. It bought at an observed
``priceUsd`` and "sold" at another observed ``priceUsd``, which is how the
2026-10-02 $GOAT position was reported as a ``+71.56%`` gain when its all-in
cash result was ``-$0.1825``. A mark is an observation about a pool; it is not
a price anyone can trade at, and it does not contain slippage, fees or the exact
order size.

THE FOUR VALUES, KEPT APART
===========================

==================  =====================================================
label               meaning
==================  =====================================================
OBSERVED_MARK       DexScreener's price. An observation. NEVER a fill.
EXECUTABLE_QUOTE    a read-only price computed from on-chain state for THIS
                    exact order size. Still not a fill.
SIMULATED_FILL      what this harness assumes happened, derived ONLY from an
                    executable quote.
ACTUAL_FILL         does not exist. No transaction is ever built or sent.
==================  =====================================================

The realized PAPER return is ``exit_fill / entry_fill`` over simulated
settlements. It is NEVER ``exit_mark / entry_mark - 1``.

ONE SOURCE OF TRUTH FOR QUOTES
===============================

All executable quote logic lives in :mod:`autopaper.same_market` - the
oracle-verified read-only engine whose bonding-curve and PumpSwap quotes were
checked against the official ``@pump-fun`` SDKs and against Jupiter on live
mainnet, to the raw integer. :mod:`autopaper.paper_live.quotes` is a thin
adapter over it. There is deliberately NO second quote implementation here,
because two implementations of the same formula is exactly how the two drift
apart without anyone noticing.

THE PRODUCTION STRATEGY IS NOT REWRITTEN
========================================

This module calls, unmodified:

* :func:`autopaper.l25.evaluate` - the frozen, hash-verified entry gate
* :class:`autopaper.analysis.decision.DecisionEngine` - the full decision
* :func:`autopaper.analysis.features.build_features` - the feature build
* :class:`autopaper.exits.ExitPolicy` - the exit precedence
* :func:`autopaper.sizing.compute_position_size` - the sizing formula
* :mod:`autopaper.strategy_profile` - the max-hold profiles

No threshold, gate, precedence or size is redefined here. What changes is only
WHERE THE PRICE COMES FROM.

SAFETY
======

This package has no signer, no transaction builder, no broadcaster and no wallet.
Importing it installs the read-only import barrier from
:mod:`autopaper.same_market.guard`, and
:func:`autopaper.paper_live.guards.audit` fails closed if any submission
capability is ever reachable from it.
"""

from __future__ import annotations

# Armed at import time, before any provider module is loaded, so the barrier is
# in place before a client can be constructed.
from autopaper.same_market.guard import (  # noqa: E402
    NoBroadcastViolation,
    audit_no_execution_capability,
    install_no_broadcast_guard,
)

install_no_broadcast_guard()

from autopaper.paper_live.arms import (  # noqa: E402
    ARMS,
    ResearchArm,
    arm_by_name,
    describe_arm,
)
from autopaper.paper_live.guards import (  # noqa: E402
    PaperAudit,
    assert_paper_only,
    audit_paper_capability,
)
from autopaper.paper_live.schema import (  # noqa: E402
    ACTUAL_FILL_AVAILABLE,
    FailureReason,
    LatencyMarks,
    ObservedMarkPoint,
    PaperPosition,
    PriceLayer,
    QuotePoint,
    RealizedPnl,
    SignalSnapshot,
    SimulatedFill,
    SignalRecord,
    TradeRecord,
)

__all__ = [
    "ACTUAL_FILL_AVAILABLE",
    "ARMS",
    "FailureReason",
    "LatencyMarks",
    "NoBroadcastViolation",
    "ObservedMarkPoint",
    "PaperAudit",
    "PaperPosition",
    "PriceLayer",
    "QuotePoint",
    "RealizedPnl",
    "ResearchArm",
    "SignalRecord",
    "SignalSnapshot",
    "SimulatedFill",
    "TradeRecord",
    "arm_by_name",
    "assert_paper_only",
    "audit_no_execution_capability",
    "audit_paper_capability",
    "describe_arm",
    "install_no_broadcast_guard",
]
