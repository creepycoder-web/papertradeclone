"""
Read-only research and observability modules.

ISOLATION CONTRACT
------------------
Nothing in this package may:

* trade, sign, construct or broadcast a transaction
* import an execution, wallet, signing or settlement module
* write to ``var/``, ``.env.local``, or any credential or wallet path
* modify ``autopaper.paper_live`` or any production strategy component

Every module here is a READER. Where a module needs data it takes it as an
argument, or reads through an explicitly injected client, so it can be tested
fully offline with stubs.

``tests/research/test_forensics_isolation.py`` enforces the contract.
"""

from __future__ import annotations

__all__ = [
    "telegram_guard",
    "transfer_fee",
    "market_timing",
    "cost_forensics",
    "min_trade",
]
