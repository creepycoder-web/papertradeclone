"""Canonical quote failure taxonomy.

WHY A SEPARATE MODULE
---------------------
The failure classes were originally declared in :mod:`autopaper.execution.jupiter`
and imported by every other provider, which made Jupiter look like the owner of
the contract when it is only one implementation of it. A provider that cannot
import Jupiter must still be able to raise and be caught by the same ``except``
clauses.

This module owns the vocabulary. :mod:`autopaper.execution.jupiter`,
:mod:`autopaper.execution.raptor` and :mod:`autopaper.execution.quoting` all
re-export these exact class objects, so every existing ``except`` clause keeps
working and ``isinstance`` relationships are unchanged.

THE VOCABULARY IS SAFETY-CRITICAL
---------------------------------
* :class:`QuoteUnavailable` -- no usable price. ALWAYS means DO NOT TRADE.
* :class:`QuoteRejected` -- the venue refused the ORDER (no route, bad token).
  A second venue may legitimately serve it.
* :class:`MicroOrderUnsupported` -- the venue will not accept the NOTIONAL. NOT
  fallback-eligible: another venue's minimum is not a fix for this one, and
  enlarging the order is never the answer.
* :class:`QuoteStale` -- a price exists but is outside its freshness budget.
  Fails closed rather than being used.
* :class:`QuoteDirectionError` -- the direction was wrong or unstated. Never
  repaired; see :mod:`autopaper.execution.price_layers`.

A provider outage and a venue refusal must stay distinguishable, because only
one of them may trigger a fallback. Collapsing them is how a provider outage
turns into a double attempt on a live order.
"""

from __future__ import annotations


class QuoteUnavailable(RuntimeError):
    """No usable quote. Always means DO NOT TRADE."""

    status = "QUOTE_UNAVAILABLE"


class QuoteRejected(QuoteUnavailable):
    """The venue refused the order (too small, no route, bad token)."""

    status = "QUOTE_REJECTED"


class MicroOrderUnsupported(QuoteUnavailable):
    """
    The venue will not support this notional.

    The ``MICRO_REAL_EXECUTION_UNSUPPORTED`` condition. A distinct exception so
    the caller stops cleanly instead of retrying or enlarging the order.
    """

    status = "MICRO_REAL_EXECUTION_UNSUPPORTED"


class QuoteStale(QuoteUnavailable):
    """
    A price exists but is older than its freshness budget.

    Separate from :class:`QuoteRejected` because a stale price is not a refusal:
    the trade may be perfectly fine 200 ms later. The caller decides whether to
    re-quote, and the staleness is recorded either way.
    """

    status = "QUOTE_STALE"

    def __init__(self, message: str, *, age_ms: float = 0.0,
                 budget_ms: float = 0.0, **kwargs) -> None:
        super().__init__(message)
        self.age_ms = age_ms
        self.budget_ms = budget_ms


class ChainStateUnavailable(QuoteUnavailable):
    """
    On-chain state could not be read or failed its structural validation.

    Raised by the direct chain-state providers. It is a refusal to state a price
    we cannot prove, never a fallback to some unrelated market source: a
    PumpSwap pool whose reserves cannot be read must not be priced from a
    DexScreener mark wearing its name.
    """

    status = "CHAIN_STATE_UNAVAILABLE"


__all__ = [
    "ChainStateUnavailable",
    "MicroOrderUnsupported",
    "QuoteRejected",
    "QuoteStale",
    "QuoteUnavailable",
]