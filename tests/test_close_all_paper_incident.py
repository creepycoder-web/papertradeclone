"""
The TITS CLOSE ALL incident, pinned offline.

THE INCIDENT
------------
A PAPER position on mint ``GD66KG85…`` (TITS) was reported to the operator as:

    CLOSE ALL PAPER RESULT: Targeted 1 / Closed 0 / Failed 1 / Remaining 1
    Errors: NONE

...on every press of the kill switch, while the position itself never closed and
kept blocking the single L25 slot.

TWO DEFECTS, BOTH REQUIRED FOR THAT OUTCOME
-------------------------------------------
1. ``close_all_paper`` decided "did it settle?" with ``failure is not None``.
   ``FailureReason.NONE`` is an enum member and is NOT ``None``, so a position
   that HAD been sold, filled and settled took the FAILURE branch every time and
   was reported with the reason string ``"NONE"`` -- a word an operator cannot
   act on. This is where ``Errors: NONE`` came from.
2. ``close_all_paper`` never committed the settlement. It called
   ``_close_for_arm`` (which moves the arm's bankroll in memory) and returned.
   No ``arm_exit`` row reached the ledger and ``paper_state.json`` was never
   rewritten, so the resumable book still listed the position OPEN and the next
   restart rehydrated it -- closing, then un-closing, forever.

EVERYTHING HERE IS OFFLINE. No network, no RPC, no Telegram, no wallet. The
quote doubles are the repository's own, and no test fabricates a successful live
trade: a "closed" assertion here means a simulated fill was constructed from an
injected executable quote, which is the production PAPER definition of a close.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from tests import paper_bridge_fakes as _fk
except ImportError:  # pragma: no cover - direct invocation
    import paper_bridge_fakes as _fk  # type: ignore[no-redef]

from autopaper.config import build_config
from autopaper.engine import PaperTrader
from autopaper.paper_live.arms import arm_names
from autopaper.paper_live.quotes import SolReference
from autopaper.paper_live.schema import (
    ARM_CLOSE_NO_FILL_CONTRADICTION,
    ARM_CLOSE_SETTLEMENT_FAILED,
    FailureReason,
)
from autopaper.telegram.ui import render_close_all_paper_result

alert, real_report, run = _fk.alert, _fk.real_report, _fk.run

#: The exact persisted shape of the incident: a bonding-curve entry whose curve
#: has since graduated, so its own book cannot be sold into.
TITS_LIKE_MINT = "GD66KG85oaZZicz7eD8hriuqABGMhhtgZPhghZ8xpump"


def _async_value(value):
    async def _inner():
        return value
    return _inner()


@pytest.fixture(autouse=True)
def _sol_reference(monkeypatch):
    """
    PAPER self-resolves SOL/USD through the canonical provider.

    Offline, that provider is unreachable, so the runtime would refuse every
    entry with ``SOL_REFERENCE_MISSING`` and no fixture position would ever
    open. Pinning the resolver keeps these tests on the CLOSE path, which is
    what is under test; it is the same substitution the existing incident suite
    uses.
    """
    monkeypatch.setattr(
        PaperTrader, "_backend_sol_price", lambda self: _async_value(119.55))


def _trader(tmp_path, *, quotes=None) -> PaperTrader:
    """A PAPER runtime with an INJECTED quote service (so nothing is online)."""
    return PaperTrader(
        build_config(tmp_path),
        market=_fk.FakeMarket(),
        quote_service=quotes if quotes is not None else _fk.FakeQuoteService(),
    )


def _open_one(tmp_path, *, quotes=None, mint=_fk.MINT, symbol="T"):
    """Open exactly ONE execution-aware PAPER position and return the trader."""
    trader = _trader(tmp_path, quotes=quotes)
    run(trader._execute_paper_buy_execution_aware(
        real_report(mint=mint, symbol=symbol), alert(mint=mint, symbol=symbol),
        None))
    assert trader.last_execution is not None, "the fixture did not open a fill"
    assert trader.last_execution["executed"] is True, (
        f"fixture did not open a position: {trader.last_execution}")
    bridge = trader._paper_bridge
    account = bridge.accounts[arm_names()[0]]
    assert account.position is not None
    return trader


def _arm_exit_rows(trader: PaperTrader) -> list[dict]:
    """The arm settlements on the ledger, read back from the file itself."""
    path = trader.ledger.ledger_path
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "arm_exit":
            rows.append(row)
    return rows


def _open_mints(trader: PaperTrader) -> list[str]:
    return [row["mint"] for row in trader.open_positions_view()
            if row.get("source") == "execution_aware_paper"]


#: A second, distinct mint. Re-entry into the FIRST mint is separately blocked by
#: the cooldown, so using it would measure the wrong guard.
_SECOND_MINT = "7LSsEoJGhLeZzGvDofTdNg7M3JttxQqGWNLo6vWMpump"


def _try_second_entry(trader: PaperTrader) -> str | None:
    """Attempt a second entry on the same arm; return why it was refused."""
    run(trader._execute_paper_buy_execution_aware(
        real_report(mint=_SECOND_MINT, symbol="T2"),
        alert(mint=_SECOND_MINT, symbol="T2", message_id=9999), None))
    engine = trader._paper_bridge
    for record in reversed(engine.signal_records):
        if not record.traded:
            return str(getattr(record.failure, "value", record.failure))
    return None


# ===========================================================================
# 1. the successful close is reported as a close
# ===========================================================================


class TestSuccessfulCloseIsReportedAsClosed:
    def test_a_valid_sell_quote_closes_and_is_counted(self, tmp_path):
        """
        THE REPORTING DEFECT.

        Before the fix this returned ``Closed: 0 / Failed: 1 / Errors: NONE``
        for a position that had been sold.
        """
        trader = _open_one(tmp_path)
        result = run(trader.close_all_paper())

        assert result["success"] is True
        assert result["targeted"] == 1
        assert result["closed"] == 1, result
        assert result["failed"] == 0, result
        assert result["remaining"] == 0, result
        assert result["errors"] == [], result
        assert result["details"][0]["outcome"] == "CLOSED"

    def test_a_settled_close_never_reports_the_reason_none(self, tmp_path):
        """``Errors: NONE`` is the specific unreadable output being banned."""
        trader = _open_one(tmp_path)
        result = run(trader.close_all_paper())
        rendered = render_close_all_paper_result(result)
        assert "NONE" not in rendered, rendered
        # Nothing anywhere in the payload may carry the bare verdict either: a
        # reason list is what an operator reads when the close does not happen.
        assert not [e for e in result["errors"] if str(e).strip() == "NONE"]

    def test_the_position_is_gone_from_the_operator_view(self, tmp_path):
        trader = _open_one(tmp_path)
        run(trader.close_all_paper())
        assert _open_mints(trader) == []

    def test_the_l25_slot_is_released(self, tmp_path):
        """
        A closed position must stop holding the one-position slot.

        Asserted by BEHAVIOUR -- a second entry on the same arm is no longer
        refused with ``ENTRY_BLOCKED_POSITION_ACTIVE`` -- rather than by reading
        a counter, which is a tally of refusals and not a state.
        """
        trader = _open_one(tmp_path)
        arm = arm_names()[0]
        assert trader._paper_bridge.accounts[arm].has_position

        run(trader.close_all_paper())

        assert not trader._paper_bridge.accounts[arm].has_position
        blocked = _try_second_entry(trader)
        assert blocked != "ENTRY_BLOCKED_POSITION_ACTIVE", blocked


# ===========================================================================
# 2. the settlement is DURABLE (defect 2)
# ===========================================================================


class TestSettlementIsDurable:
    def test_the_ledger_records_exactly_one_settlement(self, tmp_path):
        trader = _open_one(tmp_path)
        run(trader.close_all_paper())
        rows = _arm_exit_rows(trader)
        assert len(rows) == 1, rows
        assert rows[0]["pnl_basis"] == "FILL_VS_FILL"

    def test_the_resumable_book_no_longer_lists_the_position(self, tmp_path):
        """
        THE RESTART BUG.

        Before the fix ``paper_state.json`` still listed the position OPEN, so
        every restart rehydrated it and the operator's close was undone.
        """
        trader = _open_one(tmp_path)
        assert any(_fk.MINT in json.dumps(p) for p in
                   json.loads(trader.paper_state_path().read_text())
                   ["open_positions"]), "fixture did not persist the position"

        run(trader.close_all_paper())

        state = json.loads(trader.paper_state_path().read_text())
        assert not any(_fk.MINT in json.dumps(p) for p in state["open_positions"])

    def test_repeated_close_all_does_not_double_settle(self, tmp_path):
        trader = _open_one(tmp_path)
        first = run(trader.close_all_paper())
        second = run(trader.close_all_paper())

        assert first["closed"] == 1
        assert second["targeted"] == 0, second
        assert second["closed"] == 0, second
        assert len(_arm_exit_rows(trader)) == 1

    def test_a_failed_booking_is_not_reported_as_a_close(self, tmp_path,
                                                        monkeypatch):
        """
        A settlement whose closed record cannot be written is a persistence
        incident. Presenting it as a completed sale is the one thing this must
        never do.
        """
        trader = _open_one(tmp_path)
        monkeypatch.setattr(
            trader, "_record_arm_trades", lambda trades: [])
        result = run(trader.close_all_paper())

        assert result["closed"] == 0, result
        assert result["failed"] == 1, result
        assert result["success"] is False
        codes = [d.get("failure_reason") for d in result["details"]]
        assert ARM_CLOSE_SETTLEMENT_FAILED in codes, result
        assert any(ARM_CLOSE_SETTLEMENT_FAILED in e for e in result["errors"])
        assert "NONE" not in render_close_all_paper_result(result)


# ===========================================================================
# 3. a failed close stays OPEN and says WHY
# ===========================================================================


class TestFailedCloseIsActionable:
    def _assert_still_open_and_named(self, trader, result, *, code):
        arm = arm_names()[0]
        assert trader._paper_bridge.accounts[arm].position is not None, \
            "a failed close must leave the position OPEN"
        assert result["closed"] == 0, result
        assert result["failed"] == 1, result
        assert result["remaining"] == 1, result
        assert result["success"] is False
        codes = [d.get("failure_reason") for d in result["details"]]
        assert code in codes, result
        assert _fk.MINT in _open_mints(trader), "the position disappeared"
        rendered = render_close_all_paper_result(result)
        assert "NONE" not in rendered, rendered
        assert code in rendered, rendered

    def test_no_quote_leaves_it_open_with_a_specific_code(self, tmp_path):
        unavailable = _fk.FakeQuoteService(
            sell=_fk.sell_point(
                available=False, missing_reason="no executable route"))
        trader = _open_one(tmp_path, quotes=unavailable)
        result = run(trader.close_all_paper())
        self._assert_still_open_and_named(
            trader, result, code=FailureReason.EXIT_QUOTE_UNAVAILABLE.value)

    def test_a_none_quote_cannot_raise_or_report_none(self, tmp_path):
        """
        THE ``None`` QUOTE.

        A provider that returns no quote object at all must produce a named
        refusal -- never an unhandled exception, and never ``Errors: NONE``.
        """
        trader = _open_one(tmp_path, quotes=_NoneSellQuoteService())
        result = run(trader.close_all_paper())
        self._assert_still_open_and_named(
            trader, result, code=FailureReason.EXIT_QUOTE_UNAVAILABLE.value)

    def test_a_quote_that_raises_is_named_not_crashed(self, tmp_path):
        trader = _open_one(tmp_path, quotes=_RaisingSellQuoteService())
        result = run(trader.close_all_paper())
        self._assert_still_open_and_named(
            trader, result, code=FailureReason.EXIT_QUOTE_UNAVAILABLE.value)

    def test_an_unproven_identity_is_still_refused(self, tmp_path):
        unknown = _fk.FakeQuoteService(sell=_fk.sell_point(identity="UNKNOWN"))
        trader = _open_one(tmp_path, quotes=unknown)
        result = run(trader.close_all_paper())
        self._assert_still_open_and_named(
            trader, result,
            code=FailureReason.EXIT_QUOTE_IDENTITY_UNKNOWN.value)

    def test_a_mismatched_identity_is_still_refused(self, tmp_path):
        mismatch = _fk.FakeQuoteService(sell=_fk.sell_point(identity="FAIL"))
        trader = _open_one(tmp_path, quotes=mismatch)
        result = run(trader.close_all_paper())
        self._assert_still_open_and_named(
            trader, result,
            code=FailureReason.EXIT_QUOTE_IDENTITY_UNKNOWN.value)

    def test_a_stale_quote_is_still_refused(self, tmp_path):
        stale = _fk.FakeQuoteService(sell=_fk.sell_point(
            quote_age_ms=90_000.0, max_quote_age_ms=30_000.0))
        trader = _open_one(tmp_path, quotes=stale)
        result = run(trader.close_all_paper())
        self._assert_still_open_and_named(
            trader, result, code=FailureReason.EXIT_QUOTE_STALE.value)

    def test_a_failed_close_keeps_blocking_the_next_buy(self, tmp_path):
        """The blocker is only ever suppressed on a REAL close."""
        unavailable = _fk.FakeQuoteService(
            sell=_fk.sell_point(
                available=False, missing_reason="no executable route"))
        trader = _open_one(tmp_path, quotes=unavailable)
        result = run(trader.close_all_paper())
        assert result["remaining"] == 1, result

        assert _try_second_entry(trader) == "ENTRY_BLOCKED_POSITION_ACTIVE"


class _NoneSellQuoteService:
    """A provider that answers a SELL with no quote object at all."""

    def __init__(self):
        self._buy = _fk.FakeQuoteService()

    async def quote_buy(self, **kw):
        return await self._buy.quote_buy(**kw)

    async def quote_sell(self, **kw):
        return None


class _RaisingSellQuoteService:
    """A provider whose SELL raises, as an upstream throttle surface can."""

    def __init__(self):
        self._buy = _fk.FakeQuoteService()

    async def quote_buy(self, **kw):
        return await self._buy.quote_buy(**kw)

    async def quote_sell(self, **kw):
        raise RuntimeError("upstream 429")


# ===========================================================================
# 4. the classification itself
# ===========================================================================


class TestCloseFailureClassification:
    def test_a_settled_trade_is_not_misread_as_a_failure(self):
        """
        The regression in one line: ``FailureReason.NONE`` is not ``None``.

        Every other assertion in this module is downstream of this one.
        """
        trade = _SettledTrade()
        assert trade.failure is FailureReason.NONE
        assert trade.failure is not None, "the enum member must not be None"

    def test_a_contradictory_trade_is_named_not_none(self):
        """
        A record that claims ``NONE`` and holds no fill is the contradiction
        this fix exists to name. It must never reach the operator as ``NONE``.
        """
        from autopaper.engine import _arm_close_failure_code

        code = _arm_close_failure_code(_ContradictoryTrade())
        assert code == ARM_CLOSE_NO_FILL_CONTRADICTION
        assert code != "NONE"

    def test_a_real_reason_is_preserved_verbatim(self):
        from autopaper.engine import (
            _arm_close_failure_code,
            _arm_close_failure_reason,
        )

        trade = _FailedTrade(
            failure=FailureReason.EXIT_QUOTE_UNAVAILABLE,
            detail="the bonding curve is complete; there is no curve to sell into",
        )
        assert _arm_close_failure_code(trade) == \
            FailureReason.EXIT_QUOTE_UNAVAILABLE.value
        reason = _arm_close_failure_reason(trade)
        assert reason.startswith(FailureReason.EXIT_QUOTE_UNAVAILABLE.value)
        assert "bonding curve is complete" in reason

    def test_the_renderer_never_shows_an_empty_failure(self):
        """A count that says something failed must never print no cause."""
        rendered = render_close_all_paper_result({
            "targeted": 1, "closed": 0, "failed": 1, "remaining": 1,
            "details": [{"outcome": "ERROR"}], "errors": [],
        })
        assert ARM_CLOSE_NO_FILL_CONTRADICTION in rendered


class _SettledTrade:
    """The shape of a trade that DID settle: ``NONE`` and a fill."""

    def __init__(self):
        self.failure = FailureReason.NONE
        self.failure_detail = ""
        self.exit_fill = object()


class _ContradictoryTrade:
    """A success verdict with no fill on it. Never reportable as ``NONE``."""

    def __init__(self):
        self.failure = FailureReason.NONE
        self.failure_detail = ""
        self.exit_fill = None


class _FailedTrade:
    def __init__(self, failure, detail=""):
        self.failure = failure
        self.failure_detail = detail
        self.exit_fill = None


# ===========================================================================
# 5. safety invariants the incident must not have cost
# ===========================================================================


class TestSafetyInvariants:
    def test_real_stays_off(self, tmp_path):
        trader = _open_one(tmp_path)
        assert trader.state.real_armed is False
        assert trader.state.real_trading_enabled is False
        assert trader.state.trade_mode.value == "PAPER"

    def test_a_close_never_builds_signs_or_broadcasts(self, tmp_path):
        """Structural: the PAPER arm has no transaction capability to reach."""
        from autopaper.paper_live.guards import assert_paper_only

        audit = assert_paper_only().as_dict()
        assert audit["transaction_built"] == 0
        assert audit["transaction_signed"] == 0
        assert audit["transaction_broadcast"] == 0
        assert audit["forbidden_attrs_found"] == []
        assert audit["forbidden_modules_found"] == []

    def test_no_mark_price_fallback_was_introduced(self):
        import inspect

        source = inspect.getsource(PaperTrader.close_all_paper)
        assert "self.market.quote(" not in source
        assert "quote.observation" not in source
        assert "current_mark_usd" not in source, \
            "a mark must never stand in for an executable SELL quote"