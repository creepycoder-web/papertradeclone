"""
OPEN POSITIONS / CLOSED POSITIONS: the two operator views added on 2026-09-30.

WHAT THESE VIEWS ARE NOT
------------------------
They are NOT a second position store. Both read the objects the accounting layer
reads:

* OPEN   -> ``PaperTrader.account`` / ``RealAccount`` (the same objects whose
  ``cash`` / ``open_exposure`` / ``equity`` produce the identity check)
* CLOSED -> the ledger's ``exit`` events (the same records
  ``statistics_by_mode`` and ``ResearchMemory`` aggregate)

So ``/balance``, ``/status``, OPEN POSITIONS and CLOSED POSITIONS cannot
disagree, because there is only one copy of each fact. Every test that compares
two surfaces compares them against the account, not against each other.

They are also on-demand only. There is no poller, no subscription and no
per-price message: pressing a button calls a pure renderer with rows the engine
already had.
"""


from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from autopaper.clock import FakeClock
from autopaper.engine import CLOSED_PAGE_SIZE, PaperTrader
from autopaper.modes import RunState, TradeMode
from autopaper.telegram import ui
from autopaper.telegram.actions import ButtonHandler, OperatorActions

from .conftest import MINT, MINT2
from .test_incident_paper_buy import (
    alert,
    force_buy,
    make_trader,
    notifications,
)

pytestmark = pytest.mark.strategy

T0 = 1_700_000_000.0


@pytest.fixture(autouse=True)
def config(config):
    """
    These views are the LEGACY book's views, so they need the legacy path.

    The module docstring says so: OPEN reads ``PaperTrader.account`` and CLOSED
    reads the ledger's ``exit`` events -- not the execution-aware arms. The
    default PAPER entry goes through the executable-quote bridge, which needs a
    real chain client and is guarded offline. The rollback flag is the ONE
    documented switch to the legacy path (see
    ``tests/test_incident_paper_buy.py`` for the same switch's coverage), and
    setting it explicitly here is how this module states which book it tests
    instead of quietly depending on a default.
    """
    object.__setattr__(config.real, "paper_legacy_mark_fill_rollback", True)
    yield config


def press_open(handler: ButtonHandler):
    return asyncio.run(handler.handle(ui.CB_OPEN))


def press_closed(handler: ButtonHandler, offset: int = 0):
    return asyncio.run(handler.handle(ui.CB_CLOSED))


def handler_for(trader: PaperTrader) -> ButtonHandler:
    return ButtonHandler(trader.state, trader._operator_actions())


# ---------------------------------------------------------------------------
# 1. the buttons exist and are wired to real readers
# ---------------------------------------------------------------------------


def _state():
    from autopaper.modes import RuntimeState

    st = RuntimeState()
    st.start()
    return st


def test_both_position_buttons_are_on_the_menu():
    labels = {b["text"] for row in ui.operator_keyboard(_state()) for b in row}
    assert ui.BTN_OPEN_POSITIONS in labels
    assert ui.BTN_CLOSED_POSITIONS in labels
    payloads = {b["callback_data"]: b["text"]
                for row in ui.operator_keyboard(_state()) for b in row}
    assert payloads[ui.CB_OPEN] == ui.BTN_OPEN_POSITIONS
    assert payloads[ui.CB_CLOSED] == ui.BTN_CLOSED_POSITIONS


def test_the_two_views_read_the_engines_authoritative_views(config, clock):
    """The wiring is to the engine's readers, not to a Telegram-side list."""
    trader = make_trader(config, clock)
    trader.startup()
    extras = trader._operator_actions().extras
    assert extras["open_positions"] == trader.open_positions_view
    assert extras["closed_positions"] == trader.closed_positions_view
    trader._release_ownership()


# ---------------------------------------------------------------------------
# 2. the empty states
# ---------------------------------------------------------------------------


def test_open_positions_says_no_open_positions_explicitly():
    text = ui.render_open_positions([])
    assert "No open positions." in text
    assert "OPEN POSITIONS" in text
    # No zeros invented for a position that does not exist.
    assert "$0.0000" not in text


def test_closed_positions_says_no_closed_positions_explicitly():
    text = ui.render_closed_positions([], total=0)
    assert "No closed positions yet." in text
    assert "CLOSED POSITIONS" in text


def test_the_buttons_render_the_empty_state_through_the_handler(config, clock):
    trader = make_trader(config, clock)
    trader.startup()
    h = handler_for(trader)
    before = h.state.run_state
    assert "No open positions." in press_open(h).text
    assert "No closed positions yet." in press_closed(h).text
    assert h.state.run_state is before, "a read must not change state"
    trader._release_ownership()


# ---------------------------------------------------------------------------
# 3. OPEN POSITIONS: every requested field, from the account
# ---------------------------------------------------------------------------


def test_open_positions_reports_every_requested_field(config, clock):
    trader = make_trader(config, clock)
    trader.startup()
    trader.state.start()
    force_buy(trader)
    notifications(trader)
    asyncio.run(trader.on_alert(alert(clock), None))

    rows = trader.open_positions_view()
    assert len(rows) == 1
    row = rows[0]
    for field in ("symbol", "ca", "entry_at_iso", "entry_price", "current_price",
                  "size_usd", "pnl_usd", "pnl_pct", "hold_s", "mfe_pct",
                  "mae_pct", "tp_sl", "thesis", "exit_monitor", "exit_state"):
        assert field in row, f"the open-position view is missing {field}"
    assert row["ca"] == MINT
    assert row["size_usd"] > 0
    assert row["tp_sl"] != "not declared"

    text = ui.render_open_positions(rows)
    # token, CA, entry time, entry price, current price, size, PnL $ and %,
    # hold time, MFE, MAE, TP/SL and the monitor state all reach the chat.
    assert "Bittersweet" in text
    assert MINT in text
    assert "Entry:" in text and "Current:" in text
    assert "PnL:" in text and "Size:" in text
    assert "Held:" in text
    assert "MFE:" in text and "MAE:" in text
    assert "TP/SL:" in text
    assert "Monitor:" in text
    assert "Thesis:" in text
    assert row["entry_at_iso"].startswith("20")
    trader._release_ownership()


def test_open_positions_matches_the_account_exactly(config, clock):
    """
    The view and the accounting must be the same numbers, not two renderings.
    """
    trader = make_trader(config, clock)
    trader.startup()
    trader.state.start()
    force_buy(trader)
    notifications(trader)
    asyncio.run(trader.on_alert(alert(clock), None))

    position = trader.account.open_positions()[0]
    row = trader.open_positions_view()[0]
    marks = trader._refresh_marks()
    snapshot = trader.account.snapshot(marks)

    assert snapshot["open_positions"] == 1
    assert row["position_id"] == position.position_id
    assert row["size_usd"] == pytest.approx(position.cash_out_usd)
    assert row["pnl_usd"] == pytest.approx(
        marks[position.position_id].unrealized
    )
    assert row["pnl_pct"] == pytest.approx(position.current_return_pct)
    assert row["entry_price"] == pytest.approx(
        position.entry_reference_price or position.fill_price
    )
    assert row["mfe_pct"] == pytest.approx(position.mfe_pct)
    assert row["mae_pct"] == pytest.approx(position.mae_pct)
    # ...and the same identity the dashboard shows.
    assert snapshot["open_exposure_usd"] == pytest.approx(row["size_usd"], abs=1e-4)
    trader._release_ownership()


def test_open_positions_reads_no_network(config, clock):
    """
    On demand only: pressing the button must not fetch. Marks come from the
    engine's own cache.
    """
    trader = make_trader(config, clock)
    trader.startup()
    trader.state.start()
    force_buy(trader)
    notifications(trader)
    asyncio.run(trader.on_alert(alert(clock), None))
    calls_before = trader.market.request_count
    h = handler_for(trader)
    press_open(h)
    press_open(h)
    assert trader.market.request_count == calls_before, (
        "the OPEN POSITIONS view must not place a market request"
    )
    trader._release_ownership()


def test_open_positions_never_shows_a_secret(config, clock):
    """
    Distinctive credentials are configured on purpose: the shared fixture uses a
    one-character api_hash, which would "appear" in any sentence and prove
    nothing.
    """
    import dataclasses

    from autopaper.safety import find_secrets_in_text

    config = dataclasses.replace(
        config,
        telegram=dataclasses.replace(
            config.telegram,
            api_hash="TEST_API_HASH_MUST_NOT_LEAK",
            bot_token="123456:TEST_BOT_TOKEN_MUST_NOT_LEAK",
            session="TEST_TELEGRAM_SESSION_MUST_NOT_LEAK",
        ),
        real=dataclasses.replace(
            config.real, wallet_keypair_path="/tmp/TEST_KEYPAIR_MUST_NOT_LEAK.json"
        ),
    )
    trader = make_trader(config, clock)
    trader.startup()
    trader.state.start()
    force_buy(trader)
    notifications(trader)
    asyncio.run(trader.on_alert(alert(clock), None))
    rows = trader.open_positions_view()
    text = ui.render_open_positions(rows)
    # Whatever this fixture happens to carry, none of it reaches the message.
    for secret in (trader.config.telegram.bot_token,
                   trader.config.telegram.api_hash,
                   trader.config.telegram.session,
                   trader.config.real.wallet_keypair_path):
        if secret:
            assert secret not in text, f"a configured credential leaked: {secret}"
    # ...and no secret PATTERN appears.
    assert find_secrets_in_text(text) == []
    # The wallet is never part of a paper-position row at all.
    assert "wallet_masked" not in text
    trader._release_ownership()


def test_the_open_position_view_renders_missing_fields_as_absent(config, clock):
    """A dash, never a fabricated zero."""
    rows = [{
        "mode": "PAPER", "position_id": "pos_x", "symbol": "S", "ca": MINT,
        "entry_price": None, "current_price": None, "size_usd": None,
        "pnl_usd": None, "pnl_pct": None, "mfe_pct": None, "mae_pct": None,
        "entry_at": None, "hold_s": None,
    }]
    text = ui.render_open_positions(rows)
    assert "unavailable (no mark)" in text
    assert "$0.00" not in text
    assert "0.00%" not in text


# ---------------------------------------------------------------------------
# 4. CLOSED POSITIONS: recent window, reverse chronological, paginated
# ---------------------------------------------------------------------------


def test_closed_positions_are_reverse_chronological(config, clock):
    from autopaper.exits import ExitDecision
    from autopaper.models import ExitStyle

    trader = make_trader(config, clock)
    trader.startup()
    trader.state.start()

    closed_ids: list[str] = []
    for i in range(3):
        force_buy(trader)
        notifications(trader)
        asyncio.run(trader.on_alert(alert(clock, mid=100 + i, symbol=f"T{i}"),
                                    None))
        position = trader.account.open_positions()[0]
        closed_ids.append(position.position_id)
        clock.advance(120.0)
        asyncio.run(trader._do_exit(
            position,
            ExitDecision("EXIT", "test", ExitStyle.FAST_FAILURE, trigger="test"),
            clock.now(),
        ))

    rows, total = trader.closed_positions_view()
    assert total == 3
    assert [r["position_id"] for r in rows] == list(reversed(closed_ids)), (
        "closed positions must be newest first"
    )
    exits = [r["exit_at"] for r in rows]
    assert exits == sorted(exits, reverse=True)

    text = ui.render_closed_positions(rows, total=total)
    assert "CLOSED POSITIONS" in text
    assert "newest first" in text
    trader._release_ownership()


def test_closed_positions_report_every_requested_field(config, clock):
    from autopaper.exits import ExitDecision
    from autopaper.models import ExitStyle

    trader = make_trader(config, clock)
    trader.startup()
    trader.state.start()
    force_buy(trader)
    notifications(trader)
    asyncio.run(trader.on_alert(alert(clock), None))
    position = trader.account.open_positions()[0]
    clock.advance(90.0)
    asyncio.run(trader._do_exit(
        position,
        ExitDecision("EXIT", "forced test exit", ExitStyle.FAST_FAILURE,
                     trigger="fast_failure"),
        clock.now(),
    ))

    rows, total = trader.closed_positions_view()
    assert total == 1
    row = rows[0]
    for field in ("symbol", "ca", "entry_at_iso", "exit_at_iso", "entry_price",
                  "exit_price", "realized_pnl_usd", "return_pct", "hold_s",
                  "exit_reason", "mfe_pct", "mae_pct", "trade_id"):
        assert field in row, f"the closed-position view is missing {field}"
    # The compact default carries what a reader needs to scan the list: the
    # token, the outcome, the hold, the exit reason and the labelled excursion.
    compact = ui.render_closed_positions(rows, total=total)
    assert "BITTERSWEET" in compact
    assert "held 2m30s" in compact
    assert "Exit:" in compact
    assert "-2.4%" in compact
    # The recorded reason is the exit's trigger, as the ledger stores it.
    assert "fast_failure" in compact
    # The per-trade detail (CA, trade id, route) is NOT repeated on every row of
    # the list, because a list with forty trades cannot read as forty blocks of
    # boilerplate -- but it is still reachable, from the detail view.
    assert "CA:" not in compact, (
        "the compact list repeats the CA on every row; it belongs in the "
        "detail view")
    assert "Trade:" not in compact
    detail = ui.render_closed_position_detail(row)
    for label in ("CA:", "Trade:", "Route:", "Entry:", "Exit:"):
        assert label in detail, f"{label} is missing from the detail view"
    assert row["trade_id"] in detail
    assert MINT in detail
    trader._release_ownership()


def test_closed_positions_paginate_without_spamming(config, clock):
    """
    A ledger with many trades must not become one message per trade.
    """
    from autopaper.exits import ExitDecision
    from autopaper.models import ExitStyle

    trader = make_trader(config, clock)
    trader.startup()
    trader.state.start()

    for i in range(CLOSED_PAGE_SIZE + 4):
        force_buy(trader)
        notifications(trader)
        asyncio.run(trader.on_alert(alert(clock, mid=200 + i, symbol=f"S{i}"),
                                    None))
        position = trader.account.open_positions()[0]
        clock.advance(60.0)
        asyncio.run(trader._do_exit(
            position,
            ExitDecision("EXIT", "t", ExitStyle.FAST_FAILURE, trigger="t"),
            clock.now(),
        ))

    total = CLOSED_PAGE_SIZE + 4
    rows, seen_total = trader.closed_positions_view()
    assert seen_total == total
    assert len(rows) == CLOSED_PAGE_SIZE
    text = ui.render_closed_positions(rows, total=seen_total,
                                      page_size=CLOSED_PAGE_SIZE)
    assert f"of {total}" in text
    # A second window, not a repeat of the first.
    rows2, _ = trader.closed_positions_view(offset=CLOSED_PAGE_SIZE)
    assert rows2 and rows2[0]["trade_id"] != rows[0]["trade_id"]
    assert len({r["trade_id"] for r in rows} &
               {r["trade_id"] for r in rows2}) == 0
    trader._release_ownership()


def test_closed_positions_matches_the_statistics_source(config, clock):
    """
    The view and the statistics are computed from the SAME ledger rows.
    """
    from autopaper.exits import ExitDecision
    from autopaper.models import ExitStyle

    trader = make_trader(config, clock)
    trader.startup()
    trader.state.start()
    for i in range(2):
        force_buy(trader)
        notifications(trader)
        asyncio.run(trader.on_alert(alert(clock, mid=300 + i), None))
        position = trader.account.open_positions()[0]
        clock.advance(30.0)
        asyncio.run(trader._do_exit(
            position,
            ExitDecision("EXIT", "t", ExitStyle.FAST_FAILURE, trigger="t"),
            clock.now(),
        ))

    rows, total = trader.closed_positions_view()
    paper_stats, _real_stats = trader.statistics_by_mode()
    assert total == paper_stats["total_trades"], (
        "CLOSED POSITIONS and /status must count the same closed trades"
    )
    # ...and the PnL sums agree.
    assert sum(r["realized_pnl_usd"] for r in rows) == pytest.approx(
        trader.account.realized, abs=1e-6
    )
    trader._release_ownership()


def test_closed_positions_never_shows_a_secret(config, clock):
    import dataclasses

    from autopaper.safety import find_secrets_in_text

    config = dataclasses.replace(
        config,
        telegram=dataclasses.replace(
            config.telegram,
            api_hash="TEST_API_HASH_MUST_NOT_LEAK",
            bot_token="123456:TEST_BOT_TOKEN_MUST_NOT_LEAK",
        ),
    )
    trader = make_trader(config, clock)
    trader.startup()
    rows, total = trader.closed_positions_view()
    text = ui.render_closed_positions(rows, total=total)
    assert find_secrets_in_text(text) == []
    for secret in (trader.config.telegram.api_hash,
                   trader.config.telegram.bot_token):
        if secret:
            assert secret not in text
    trader._release_ownership()


# ---------------------------------------------------------------------------
# 5. safety: the views cannot move anything
# ---------------------------------------------------------------------------


def test_the_position_views_cannot_change_any_state(config, clock):
    """
    Read-only, proved by comparison: every observable is identical before and
    after both presses, including cash, exposure, the mode and the emergency
    latch.
    """
    trader = make_trader(config, clock)
    trader.startup()
    trader.state.start()
    force_buy(trader)
    notifications(trader)
    asyncio.run(trader.on_alert(alert(clock), None))

    def snapshot():
        return (
            round(trader.account.cash, 10),
            round(trader.account.open_exposure(), 10),
            round(trader.account.realized, 10),
            trader.buys,
            trader.blocks,
            len(trader.account.open_positions()),
            trader.state.trade_mode,
            trader.state.run_state,
            trader.state.real_armed,
            trader.state.emergency_stop,
            trader.account.unfilled_attempts,
        )

    before = snapshot()
    h = handler_for(trader)
    for _ in range(3):
        press_open(h)
        press_closed(h)
    assert snapshot() == before, "a read-only view changed runtime state"


def test_real_trading_is_still_off(config, clock):
    trader = make_trader(config, clock)
    trader.startup()
    assert trader.state.real_trading_enabled is False
    assert trader.config.real.trading_enabled is False
    assert trader.config.real.env_trading_enabled_request is False
    h = handler_for(trader)
    # Pressing a position view cannot arm REAL, and ARM REAL still needs the
    # confirmed two-press flow with a passing preflight.
    result = asyncio.run(h.handle(ui.CB_REAL))
    assert "ยืนยัน" in result.text
    assert trader.state.real_trading_enabled is False
    press_open(h)
    assert trader.state.real_trading_enabled is False
    assert trader.state.trade_mode is TradeMode.PAPER
    trader._release_ownership()


def test_kill_remains_effective_alongside_the_new_buttons(config, clock):
    trader = make_trader(config, clock)
    trader.startup()
    h = handler_for(trader)
    press_open(h)
    press_closed(h)
    result = asyncio.run(h.handle(ui.CB_KILL))
    assert result.handled
    assert trader.state.emergency_stop is True
    assert trader.state.can_open_new_entries()[0] is False
    trader._release_ownership()


def test_an_unwired_reader_still_renders_the_empty_state(config, clock):
    """
    The buttons must work even before the extras are attached, rather than
    raising or reporting a position list nobody can vouch for.
    """
    from autopaper.modes import RuntimeState

    st = RuntimeState()
    st.start()
    h = ButtonHandler(st, OperatorActions())
    assert "No open positions." in press_open(h).text
    assert "No closed positions yet." in press_closed(h).text


def test_an_unknown_callback_is_still_refused(config, clock):
    from autopaper.modes import RuntimeState

    st = RuntimeState()
    st.start()
    h = ButtonHandler(st, OperatorActions())
    result = asyncio.run(h.handle("positions_please"))
    assert result.handled is False
    assert result.error == "unrecognised_callback"
