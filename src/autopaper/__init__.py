"""
MEMEBOT V3 runtime.

Moo Deng President -> AI-paper-trade data pipeline -> V3-C -> risk/sizing ->
executor (paper or Solana/Jupiter real) -> Telegram UI -> PAPER or REAL.

Startup is ALWAYS safe::

    RUNNING = false
    MODE     = PAPER
    REAL     = false

``REAL_TRADING_ENABLED=false`` is the default and an environment variable is
NOT sufficient to change it. REAL is armed only by an explicit, confirmed
Telegram operator action plus a passing preflight; see :mod:`autopaper.modes`.

The ML/Laya/shadow import guard is installed at import time, before any other
submodule can be loaded.
"""

from __future__ import annotations

from .safety import (  # noqa: F401  (import order is load-bearing)
    AGENT_ID,
    STRATEGY,
    AuditReport,
    SafetyViolation,
    assert_real_trading_disabled_at_startup,
    audit,
    install_import_guard,
)

install_import_guard()

__version__ = "3.0.0"
__all__ = [
    "AGENT_ID",
    "STRATEGY",
    "AuditReport",
    "SafetyViolation",
    "assert_real_trading_disabled_at_startup",
    "audit",
    "install_import_guard",
    "__version__",
]
