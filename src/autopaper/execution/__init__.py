"""
Execution layer.

Three implementations of the same idea, deliberately kept distinct:

* :mod:`fill`  - the PAPER simulator. A documented, deterministic model of
  fees, slippage and market impact. It is a SIMULATION and is never used on the
  REAL path.
* :mod:`solana` - wallet loading, wire-level signing, the JSON-RPC client,
  broadcast, confirmation and on-chain reconciliation. Ported from the
  reference executor, unchanged in behaviour.
* :mod:`jupiter` - quote and unsigned-transaction construction from the venue.
* :mod:`real`  - the REAL execution path: quote -> build -> sign -> simulate ->
  broadcast -> confirm -> reconcile, behind the full pre-trade safety sequence.

Position SIZING IS NOT HERE.

It lives in exactly one place, :mod:`autopaper.sizing`, which implements the
frozen V3-C policy: 10% of equity at or below $50, 12.5% above. It is a pure
function of equity with no overrides, no floors and no venue minimum, so there
is no second sizing path to drift.
"""

from .fill import (
    FillInputs,
    fill_books_reconcile,
    impact_bps,
    simulate_fill,
    staleness_bps,
    total_slippage_bps,
)

__all__ = [
    "FillInputs",
    "simulate_fill",
    "impact_bps",
    "staleness_bps",
    "total_slippage_bps",
    "fill_books_reconcile",
]
