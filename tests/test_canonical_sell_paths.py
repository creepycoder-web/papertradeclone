"""Production sell-path invariants.

These are intentionally small source/contract tests. They protect the safety
boundary that a future refactor could otherwise bypass while leaving the more
extensive execution-aware bridge tests unchanged.
"""

from __future__ import annotations

import inspect

from autopaper.engine import PaperTrader


def test_flatten_delegates_to_force_close_without_mark_pricing():
    source = inspect.getsource(PaperTrader.flatten)
    assert "close_all_paper" in source
    assert "paper_legacy_mark_fill_rollback" in source
    assert "account.close_position" not in source


def test_force_close_has_no_legacy_mark_fill_fallback():
    source = inspect.getsource(PaperTrader.close_all_paper)
    assert "self.market.quote(" not in source
    assert "quote.observation" not in source
    assert "account.close_position" not in source
    assert "LEGACY_POSITION_REQUIRES_EXECUTION_AWARE_SELL" in source
