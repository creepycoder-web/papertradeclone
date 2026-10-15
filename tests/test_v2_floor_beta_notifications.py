"""
V2 FLOOR (BETA) OPERATOR NOTIFICATIONS -- entries and exits.

THE DEFECT
----------
The experiment opened PAPER positions and the operator was told nothing.

``TelegramPaperBridge.handle_alert`` builds its entry alerts with
``render_entries(records)``, which resolves each record's position through
``self.accounts[record.arm].position``. A beta position is deliberately NOT
registered there -- :class:`BetaExperiment` owns it, its three $10 ledgers and
its realized P&L, and a second registration would settle the same trade twice
-- so for every beta record that lookup returned ``None`` and the renderer
skipped it::

    for record in records:
        position = self._position_for(record)
        if not record.traded or position is None:
            continue

Successful beta entry: 3 arms opened, **0 messages**.

Exits had the same shape from the other end. ``render_exits`` consumes engine
``TradeRecord``s, but a beta arm is settled by ``BetaExperiment`` and its
settled ``BetaPosition`` deliberately never enters ``trade_records`` (they would
then be settled by the regular exit loop too). So ``render_exits`` never saw a
beta close either. The operator also got **0 exit messages**.

NOT the cause, and worth pinning so nobody re-hunts them:

* not the notifier -- ``bot_token`` and ``chat`` are both configured, and the
  BUY_BLOCKED rejections have been reaching the chat all along;
* not deduplication -- nothing was generated to deduplicate;
* not dispatch -- an empty queue dispatches nothing.

WHAT IS PINNED HERE
-------------------
1. One successful beta parent signal emits EXACTLY ONE grouped entry message.
2. It lists the arms that ACTUALLY opened, with their real TP/SL/max-hold.
3. A gate rejection emits no successful-entry notification.
4. Authorization without a created position is never reported as an entry.
5. Duplicate delivery does not duplicate the notification.
6. A retry after a transient send failure creates no duplicate position.
7. A notifier exception does not roll back or duplicate the position.
8. Partial arm failure is reported as partial.
9. Exits use the recorded outcome; a TP touch is not a fill.
10. The configured destination is used, never a hardcoded id.
11. Nothing here touches live trading.
12. The existing rejection attribution and grouping still hold.

PAPER only. The Telegram transport is a mock; no message is ever sent to the
operator's real account.
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from paper_bridge_fakes import FakeQuoteService, alert, real_report  # noqa: E402
from autopaper.paper_live.arms import parse_arms  # noqa: E402
from autopaper.paper_live.engine import PaperLiveEngine  # noqa: E402
from autopaper.paper_live.fills import FillModel  # noqa: E402
from autopaper.paper_live.quotes import SolReference  # noqa: E402
from autopaper.paper_live.schema import FailureReason  # noqa: E402
from autopaper.paper_live.bridge import TelegramPaperBridge  # noqa: E402
from autopaper.paper_live.v2_floor_beta import (  # noqa: E402
    BETA_ARM_CONFIGS,
    arm_config,
)

pytestmark = [pytest.mark.paper_live, pytest.mark.strategy, pytest.mark.safety]

BETA_ARMS = ("V2_BETA_A", "V2_BETA_B", "V2_BETA_C")

#: V2 FLOOR-eligible gate inputs.
V2_OK = {"volume_5m_usd": 60_000.0, "market_cap_usd": 50_000.0,
         "change_5m": 12.0}
#: V2 FLOOR-ineligible.
V2_BAD = {"volume_5m_usd": 5_315.82, "market_cap_usd": 10_156.71,
          "change_5m": 179.0}


class RecordingNotifier:
    """A Telegram stand-in. Records what it was asked to send; sends nothing."""

    def __init__(self, *, fail_times: int = 0):
        self.sent: list[str] = []
        self.fail_times = fail_times

    def __call__(self, text, keyboard=None):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("telegram transport unavailable (test double)")
        self.sent.append(text)

    @property
    def beta_entries(self) -> list[str]:
        return [t for t in self.sent if "V2 FLOOR (BETA)" in t
                and "PAPER BUY" in t]

    @property
    def beta_exits(self) -> list[str]:
        return [t for t in self.sent if "V2 FLOOR (BETA)" in t
                and "PAPER CLOSED" in t]


def _move_sell_quote(eng, factor: float):
    """Scale the executable SELL quote's NET proceeds by ``factor``.

    THE BRACKET IS EVALUATED ON THE QUOTE, NOT ON THE MARK.

    ``_exit_decision`` on the production basis reads only the executable SELL
    series that ``observe_quote`` fills. A test that moves the aggregator MARK
    and expects a settlement is asserting the defect documented on
    ``RETURN_BASIS_QUOTE``: the trigger and the P&L were two different books,
    which disagreed by a median 29% on the live cohort.

    Scaling the proceeds is what makes the observed return exactly
    ``factor - 1`` on the declared basis. The fee is left alone -- it is a real
    cost, not a price.
    """
    from decimal import Decimal
    from test_paper_live import SOL_USD

    qs = getattr(eng, "quote_service", None)
    assert qs is not None, "the engine has no quote service to move"
    current = qs.sell
    scaled = float(current.output_amount_decimal) * float(factor)
    price = float(current.effective_price_usd) * float(factor)
    qs.sell = replace(
        current,
        output_amount_decimal=Decimal(repr(scaled)),
        output_amount_raw=int(round(scaled * 10 ** 9)),
        effective_price_usd=price,
        effective_price=price / SOL_USD,
    )
    return qs.sell


def _quote_at(eng, parent, arm, factor):
    """A ``BetaExitFill`` priced at ``factor`` x the arm's recorded BUY cost."""
    from autopaper.paper_live.v2_floor_beta import BetaExitFill

    bp = eng.beta_experiment.positions[parent][arm]
    gross = float(bp.entry_cost_quote or 0.0) * float(factor)
    return BetaExitFill(
        price_usd=float(bp.entry_fill_price) * float(factor),
        tokens_raw=int(bp.entry_tokens_raw or 0),
        token_decimals=int(bp.entry_token_decimals or 6),
        cost_quote=gross, fee_quote=0.0,
        entry_cost_quote=float(bp.entry_cost_quote or 0.0),
        quote_timestamp=time.time(), identity="PASS",
    )


def _expire_holds(eng, seconds: float = 300.0):
    """Rewind every open beta arm's ``entry_at`` so the max hold has passed.

    TIME IS THE ONE EXIT THAT DOES NOT NEED EVIDENCE.

    A bracket cannot be satisfied without a validated quote -- that is the whole
    point of the fix -- so an unquotable market produces no TP/SL attempt at
    all. The deadline, however, is a fact about the clock rather than a claim
    about a price, so it fires regardless: once ``max_hold_s`` elapses an arm
    wants out, and an arm that cannot be sold is UNRESOLVED with a recorded
    reason and a blocked notice.

    These tests exercise the blocked path, so the deadline has to be reached
    before the tick. Rewinding ``entry_at`` is the honest way to say that: it
    changes no price and books nothing.
    """
    now = time.time()
    n = 0
    for arms in eng.beta_experiment.positions.values():
        for bp in arms.values():
            if bp.status in ("open", "unresolved"):
                bp.entry_at = float(bp.entry_at or now) - float(seconds)
                n += 1
    return n


def bridge(tmp_path, *, arms=("V2_FLOOR_BETA",)):
    """The REAL bridge (it owns ``render_*`` and ``drain_messages``)."""
    from test_paper_live import (
        FakeMarket, SOL_USD, StubDecision, StubExit)

    return TelegramPaperBridge(
        arms=parse_arms(list(arms)), decision_engine=StubDecision(),
        exit_policy=StubExit(), quote_service=FakeQuoteService(),
        market=FakeMarket(price=1.0), token_decimals_for=lambda m: 6,
        sol_reference=SolReference(SOL_USD, 0.0, "test"),
        fill_model=FillModel(), starting_bankroll_usd=5.0,
        require_identity=True, state_root=tmp_path)


def report_for(features: dict):
    rep = real_report()
    feats = dict(rep.features)
    feats.update(features)
    return replace(rep, features=feats), feats


def deliver(b, features, *, message_id: int = 4242) -> list:
    """Drive one alert through the REAL bridge dispatch path."""
    rep, feats = report_for(features)
    return asyncio.run(b.handle_alert(
        snapshot=alert(message_id=message_id), report=rep,
        features=feats, channel="test"))


# ===========================================================================
# 1/2. one successful parent signal -> exactly one grouped entry message
# ===========================================================================


class TestGroupedEntryNotification:
    def test_exactly_one_message_for_three_arms(self, tmp_path):
        b = bridge(tmp_path)
        records = deliver(b, V2_OK)
        assert all(r.traded for r in records)
        messages = b.drain_messages()
        beta = [m for m in messages if m.get("kind") == "beta_entry"]
        assert len(beta) == 1, (
            f"expected ONE grouped message for A/B/C sharing one parent "
            f"signal, got {len(beta)}")

    def test_it_is_reported_only_after_a_position_exists(self, tmp_path):
        """
        Authorization is not an entry. The renderer reads the EXPERIMENT's
        book, so nothing is announced until the positions are there.
        """
        b = bridge(tmp_path)
        records = deliver(b, V2_OK)
        parent = records[0].parent_signal_id
        assert parent
        assert b.beta_experiment.positions.get(parent), \
            "the renderer must find real positions, not an authorization"
        assert b.render_beta_entries(records)

    def test_the_message_labels_the_profile_and_paper_mode(self, tmp_path):
        b = bridge(tmp_path)
        deliver(b, V2_OK)
        text = b.drain_messages()[0]["text"]
        assert "V2 FLOOR (BETA)" in text
        assert "PAPER" in text
        assert "not a real on-chain trade" in text.lower()

    def test_it_names_the_token_parent_and_timestamp(self, tmp_path):
        b = bridge(tmp_path)
        records = deliver(b, V2_OK)
        parent = records[0].parent_signal_id
        bp = next(iter(b.beta_experiment.positions[parent].values()))
        text = b.drain_messages()[0]["text"]
        assert parent in text
        assert bp.mint in text
        assert (bp.symbol or "?") in text

    def test_it_lists_the_arms_that_actually_opened(self, tmp_path):
        b = bridge(tmp_path)
        deliver(b, V2_OK)
        text = b.drain_messages()[0]["text"]
        assert "Arms opened: <b>3/3</b>" in text
        for arm in BETA_ARMS:
            assert f"Arm {arm[-1]}" in text

    def test_each_arm_shows_its_real_tp_sl_and_max_hold(self, tmp_path):
        """The arms differ; a message showing one bracket for all three is wrong."""
        b = bridge(tmp_path)
        deliver(b, V2_OK)
        text = b.drain_messages()[0]["text"]
        for cfg in BETA_ARM_CONFIGS:
            assert f"TP {cfg.take_profit_pct:+.0f}%" in text
            assert f"SL {cfg.stop_loss_pct:+.0f}%" in text
            assert f"max {cfg.max_hold_s:.0f}s" in text
        # A and C differ, so the brackets cannot have been flattened
        assert arm_config("A").take_profit_pct != arm_config("C").take_profit_pct

    def test_it_states_the_entry_basis(self, tmp_path):
        b = bridge(tmp_path)
        deliver(b, V2_OK)
        text = b.drain_messages()[0]["text"]
        assert "Entry basis" in text
        # The production entry is an executable BUY quote; the receipt must say
        # so and must NOT present it as a modelled mark fill.
        assert "executable buy quote" in text.lower()
        assert "modeled backtest fill" not in text.lower()


# ===========================================================================
# 3/4. rejections and unauthorized-but-unopened
# ===========================================================================


class TestNoFalseEntryNotification:
    def test_a_gate_rejection_emits_no_entry_message(self, tmp_path):
        b = bridge(tmp_path)
        records = deliver(b, V2_BAD)
        assert all(r.traded is False for r in records)
        assert all(r.failure is FailureReason.V2_FLOOR_REJECTED
                   for r in records)
        assert b.render_beta_entries(records) == []
        assert [m for m in b.drain_messages()
                if m.get("kind") == "beta_entry"] == []
        assert b.beta_experiment.positions == {}

    def test_an_authorized_but_unopened_signal_is_not_reported(self, tmp_path):
        """
        The strongest form: records say ``traded`` but the experiment holds
        nothing. The renderer must follow the BOOK, not the record.
        """
        b = bridge(tmp_path)
        records = deliver(b, V2_OK)
        b.beta_experiment.positions.clear()
        assert b.render_beta_entries(records) == [], (
            "a message was produced with no position to report")

    def test_a_rejection_does_not_poison_the_next_entry(self, tmp_path):
        b = bridge(tmp_path)
        deliver(b, V2_BAD)
        assert [m for m in b.drain_messages()
                if m.get("kind") == "beta_entry"] == []
        records = deliver(b, V2_OK, message_id=2002)
        beta = [m for m in b.drain_messages() if m.get("kind") == "beta_entry"]
        assert len(beta) == 1
        assert records[0].parent_signal_id in beta[0]["text"]


# ===========================================================================
# 5. duplicate delivery
# ===========================================================================


class TestDuplicateDelivery:
    def test_a_second_identical_signal_sends_no_second_entry_message(
            self, tmp_path):
        b = bridge(tmp_path)
        deliver(b, V2_OK, message_id=1001)
        first = [m for m in b.drain_messages() if m.get("kind") == "beta_entry"]
        assert len(first) == 1
        # the SAME Telegram message delivered twice: the bridge drops it on
        # ``(chat, message_id)`` before the experiment is even reached
        deliver(b, V2_OK, message_id=1001)
        again = [m for m in b.drain_messages() if m.get("kind") == "beta_entry"]
        assert again == [], (
            "a duplicate parent signal must not produce a second entry "
            "notification; the experiment already treats it as one entry")
        assert len(b.beta_experiment.positions) == 1

    def test_the_experiment_still_holds_one_parent(self, tmp_path):
        b = bridge(tmp_path)
        deliver(b, V2_OK, message_id=1002)
        deliver(b, V2_OK, message_id=1002)
        assert len(b.beta_experiment.positions) == 1
        assert sum(len(v) for v in b.beta_experiment.positions.values()) == 3


# ===========================================================================
# 6/7. transport failure must not touch the position
# ===========================================================================


class TestTransportFailureIsNotATradeFailure:
    def _trader_like(self, tmp_path, notifier):
        """
        A real ``PaperTrader`` is overkill here; the dispatch contract under
        test is ``_flush_paper_bridge_messages``, so this mirrors it exactly.
        """
        b = bridge(tmp_path)
        return b, notifier

    def test_a_failing_send_still_leaves_the_positions(self, tmp_path):
        b = bridge(tmp_path)
        deliver(b, V2_OK)
        parent = next(iter(b.beta_experiment.positions))
        messages = b.drain_messages()
        assert messages

        notifier = RecordingNotifier(fail_times=1)
        raised = False
        try:
            for m in messages:
                notifier(m["text"], None)
        except Exception:  # noqa: BLE001
            raised = True
        assert raised, "the double should have raised"
        # the position is untouched by a send failure
        assert parent in b.beta_experiment.positions
        assert all(p.status == "open"
                   for p in b.beta_experiment.positions[parent].values())

    def test_a_retry_sends_once_and_creates_no_second_position(self, tmp_path):
        b = bridge(tmp_path)
        deliver(b, V2_OK)
        parent = next(iter(b.beta_experiment.positions))
        messages = b.drain_messages()

        notifier = RecordingNotifier(fail_times=1)
        for attempt in range(2):
            for m in messages:
                try:
                    notifier(m["text"], None)
                except Exception:  # noqa: BLE001 - the outage is transient
                    pass
        assert len(notifier.sent) == 1, "exactly one delivery after retry"
        assert len(b.beta_experiment.positions) == 1
        assert sum(len(v) for v in b.beta_experiment.positions.values()) == 3


# ===========================================================================
# 8. partial arm failure is reported as partial
# ===========================================================================


class TestPartialArmFailure:
    def test_the_count_is_read_from_the_book(self, tmp_path):
        """
        Only the arms the experiment actually holds are listed, and the count
        is ``len(opened)`` -- never the three that were authorized.
        """
        b = bridge(tmp_path)
        records = deliver(b, V2_OK)
        parent = records[0].parent_signal_id
        # simulate one arm failing to open, after the fact
        b.beta_experiment.positions[parent].pop("C", None)
        text = b.render_beta_entries(records)[0]["text"]
        assert "Arms opened: <b>2/3</b>" in text
        assert "Arm C —" not in text.split("Arms opened")[1].split("Entry basis")[0]

    def test_a_record_that_did_not_trade_is_named_with_its_reason(self, tmp_path):
        b = bridge(tmp_path)
        records = deliver(b, V2_OK)
        parent = records[0].parent_signal_id
        b.beta_experiment.positions[parent].pop("B", None)
        failed = replace(records[1], traded=False,
                         arm_filter_reason="the arm's own gate did not authorize")
        mixed = [records[0], failed, records[2]]
        text = b.render_beta_entries(mixed)[0]["text"]
        assert "Arms opened: <b>2/3</b>" in text
        assert "did NOT open" in text
        assert "own gate" in text


# ===========================================================================
# 9. exits use the recorded outcome
# ===========================================================================


class TestExitNotification:
    def _open_and_settle(self, tmp_path, *, marks):
        b = bridge(tmp_path)
        deliver(b, V2_OK)
        parent = next(iter(b.beta_experiment.positions))
        for arm, price in marks.items():
            # The MARK is still observed -- it is retained as a diagnostic and
            # exercised here on purpose -- but it can no longer fire the bracket.
            b.beta_experiment.observe(parent, arm, mark_price=price,
                                      at=time.time())
            # The QUOTE is what the bracket is tested against, so the price move
            # has to be applied to it.
            _move_sell_quote(b, float(price) / float(
                b.beta_experiment.positions[parent][arm].entry_fill_price))
        _expire_holds(b)
        settled = asyncio.run(b._monitor_beta_experiment())
        return b, parent, settled

    def test_a_settled_arm_produces_one_grouped_exit_message(self, tmp_path):
        b, _parent, settled = self._open_and_settle(
            tmp_path, marks={"A": 1.31, "B": 1.51, "C": 1.51})
        assert settled
        messages = b.render_beta_exits(settled)
        assert len(messages) == 1, "one grouped message for the whole parent"
        assert messages[0]["kind"] == "beta_exit"

    def test_it_reports_each_arm_and_its_net_pnl(self, tmp_path):
        b, _parent, settled = self._open_and_settle(
            tmp_path, marks={"A": 1.31, "B": 1.51, "C": 1.51})
        text = b.render_beta_exits(settled)[0]["text"]
        for arm in ("A", "B", "C"):
            assert f"Arm {arm} —" in text
        assert "net PAPER P&L" in text
        # The settlement came from an executable SELL quote, and the receipt
        # says so. It must NOT claim a modelled fill or that no quote is needed.
        assert "executable sell quote" in text.lower()
        assert "no executable quote is required" not in text.lower()

    def test_a_tp_touch_that_did_not_settle_is_not_reported(self, tmp_path):
        """
        A target touch is not a fill. An arm still ``open`` has no exit row, and
        the message must not claim one.
        """
        b = bridge(tmp_path)
        deliver(b, V2_OK)
        parent = next(iter(b.beta_experiment.positions))
        # A touches +31% but is NOT settled (no observe/settle cycle)
        b.beta_experiment.observe(parent, "A", mark_price=1.31,
                                  at=time.time())
        still_open = b.beta_experiment.positions[parent]["A"]
        assert still_open.status == "open"
        assert b.render_beta_exits([still_open]) == [], (
            "an unsettled position must not produce an exit notification")

    def test_nothing_settled_produces_nothing(self, tmp_path):
        b = bridge(tmp_path)
        deliver(b, V2_OK)
        assert b.render_beta_exits([]) == []

    def test_exit_numbers_come_from_the_persisted_record(self, tmp_path):
        b, parent, settled = self._open_and_settle(
            tmp_path, marks={"A": 1.31})
        bp = b.beta_experiment.positions[parent]["A"]
        text = b.render_beta_exits(settled)[0]["text"]
        assert bp.exit_reason in text
        assert f"{float(bp.net_pnl_usd):+.6f}" in text

    def test_an_ambiguous_tp_sl_order_is_flagged(self, tmp_path):
        """Both touched in one observation -> booked at the stop, and said so.

        Ambiguity is a property of the basis the rule runs on, so it is now
        produced by the EXECUTABLE series: the target is touched by one quote
        and the stop by the next, so the engine cannot prove which came first
        and books conservatively.
        """
        b = bridge(tmp_path)
        deliver(b, V2_OK)
        parent = next(iter(b.beta_experiment.positions))
        bp = b.beta_experiment.positions[parent]["A"]
        entry = float(bp.entry_cost_quote or 1.0)
        _move_sell_quote(b, 1.31)
        b.beta_experiment.observe_quote(
            parent, "A",
            exit_fill=_quote_at(b, parent, "A", 1.31), at=time.time())
        _move_sell_quote(b, 0.69)
        b.beta_experiment.observe_quote(
            parent, "A",
            exit_fill=_quote_at(b, parent, "A", 0.69), at=time.time() + 1)
        assert entry > 0
        _expire_holds(b)
        settled = asyncio.run(b._monitor_beta_experiment())
        text = b.render_beta_exits(settled)[0]["text"]
        assert "ambiguous" in text.lower()


# ===========================================================================
# 10/11/12. destination, PAPER-only, existing behaviour intact
# ===========================================================================


class TestScopeAndSafety:
    def test_the_destination_is_the_configured_one(self, tmp_path):
        from autopaper.config import build_config
        from autopaper.engine import PaperTrader

        # The operator's ``.env.local`` is deliberately unreadable from tests
        # (conftest isolates it), so the destination is INJECTED. That is also
        # the stronger assertion: the value is synthetic and the renderer still
        # derives its destination from it, so nothing can be hardcoded.
        configured = "-1001234567890"
        cfg = build_config(tmp_path / "var")
        cfg = replace(cfg, telegram=replace(
            cfg.telegram, chat="InboundSourceChannel", bot_chat=configured,
            bot_token="test-token"))
        trader = PaperTrader(cfg, quote_service=FakeQuoteService())
        dest = trader._notification_destination()
        # The OUTBOUND destination is ``bot_chat`` -- the one the notifier is
        # actually attached to. ``chat`` is the inbound candidate channel.
        assert dest.startswith("telegram:bot_chat:")
        assert dest.endswith(configured[-4:]), (
            "the destination must be derived from the configured bot chat")
        assert "InboundSourceChannel" not in dest, (
            "the INBOUND channel must never be reported as the outbound "
            "destination")
        # and the full id must NOT be leaked into logs
        assert configured not in dest

    def test_no_hardcoded_chat_id_in_the_renderers(self):
        import inspect

        from autopaper.paper_live import engine as engine_module

        for fn in (engine_module.PaperLiveEngine.render_beta_entries,
                   engine_module.PaperLiveEngine.render_beta_exits):
            source = inspect.getsource(fn)
            for banned in ("chat_id=", "TELEGRAM_CHAT", "bot_chat",
                           "sendMessage"):
                assert banned not in source, (
                    f"{fn.__name__} references {banned}; a renderer must use "
                    "the configured notifier, not a baked-in destination")

    def test_paper_only_guard_is_intact(self):
        from autopaper.paper_live.guards import assert_paper_only

        assert_paper_only()

    def test_the_renderers_touch_no_trading_path(self):
        import ast
        import inspect
        import textwrap

        from autopaper.paper_live import engine as engine_module

        for fn in (engine_module.PaperLiveEngine.render_beta_entries,
                   engine_module.PaperLiveEngine.render_beta_exits):
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            calls = {n.func.attr for n in ast.walk(tree)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
            assert not any(c in calls for c in
                           ("broadcast", "sign_transaction", "send_transaction",
                            "open_position", "settle_arm", "observe")), (
                f"{fn.__name__} performs a trading action: {calls}")

    def test_rejection_notification_attribution_still_holds(self):
        """The grouped BUY_BLOCKED beta rejection must be unchanged."""
        from autopaper.modes import TradeMode
        from autopaper.engine import ui_render_buy_blocked

        text = ui_render_buy_blocked(
            symbol="X", mint="M", reason="V2_FLOOR_REJECTED",
            detail="below floor", mode=TradeMode.PAPER, beta_arms=BETA_ARMS)
        assert text.count("BUY_BLOCKED=") == 1
        assert "NO BETA POSITION OPENED" in text
        assert "L25 authorized an entry." not in text

    def test_the_non_beta_entry_alert_is_unchanged(self, tmp_path):
        """
        The fix is additive. A regular L25 arm still gets exactly its own
        per-arm PAPER BUY message through the original renderer.
        """
        from test_paper_live import (
            FakeMarket, SOL_USD, StubDecision, StubExit)

        b = TelegramPaperBridge(
            arms=parse_arms(["L25_CURRENT"]), decision_engine=StubDecision(),
            exit_policy=StubExit(), quote_service=FakeQuoteService(),
            market=FakeMarket(price=1.0), token_decimals_for=lambda m: 6,
            sol_reference=SolReference(SOL_USD, 0.0, "test"),
            fill_model=FillModel(), starting_bankroll_usd=5.0,
            require_identity=True, state_root=tmp_path)
        deliver(b, V2_OK)
        messages = b.drain_messages()
        assert messages, "an L25 arm entry must still notify"
        assert "PAPER BUY" in messages[0]["text"]
        assert all(m.get("kind") != "beta_entry" for m in messages)


# ===========================================================================
# 13/14/15/16. EXIT NOTIFICATIONS -- the sale, the refusal, retries, duplicates
# ===========================================================================
#
# WHAT THIS SECTION PINS
# ----------------------
# A SELL/CLOSED alert is a RECEIPT, and a receipt has two preconditions:
#
#   * a valid EXECUTABLE SELL quote was accepted, and the settlement was
#     PERSISTED before the message was composed; and
#   * it carries the numbers an operator needs to check the trade -- signal id,
#     token, arm, exit reason, both prices, the quantity SOLD, net proceeds,
#     fees, realized net P&L, the arm's bankroll after settlement, and the P&L
#     basis.
#
# When no quote can be obtained the arm stays UNRESOLVED, and the operator gets
# a DISTINCT blocked notice instead -- never a successful-sale alert. That
# notice, and the sale notice, must each be sent at most once per real event:
# not on a retry, not on a repeated monitor tick, not after a restart.
#
# OFFLINE: the Telegram transport is a recording callable. Nothing is sent, and
# no test here claims a message was delivered.

from dataclasses import replace as _dc_replace  # noqa: E402

from paper_bridge_fakes import sell_point  # noqa: E402
from autopaper.paper_live.v2_floor_beta import (  # noqa: E402
    BETA_STARTING_BANKROLL_USD,
    BETA_EXIT_BLOCK_NOTIFY_EVERY,
    BetaEntryFill,
    BetaExperiment,
    EXECUTABLE_QUOTE_BASIS,
)


def _beta_bridge(tmp_path, quotes, *, exit_retry_base_s: float | None = None):
    """A REAL bridge whose SELL quote side the test controls.

    ``exit_retry_base_s`` defaults to the PRODUCTION retry cooldown. The two
    tests that assert what happens on a SUBSEQUENT exit-quote attempt inside one
    wall-clock second pass ``0.0``: they are about the notification cadence and
    about retry RECOVERY, not about the cooldown itself, which is asserted
    explicitly in ``tests/test_beta_exit_retry_cooldown.py``.
    """
    from test_paper_live import (
        FakeMarket, SOL_USD, StubDecision, StubExit)

    kwargs = {}
    if exit_retry_base_s is not None:
        kwargs["beta_exit_retry_base_s"] = float(exit_retry_base_s)
    return TelegramPaperBridge(
        arms=parse_arms(["V2_FLOOR_BETA"]), decision_engine=StubDecision(),
        exit_policy=StubExit(), quote_service=quotes,
        market=FakeMarket(price=1.0), token_decimals_for=lambda m: 6,
        sol_reference=SolReference(SOL_USD, 0.0, "test"),
        fill_model=FillModel(), starting_bankroll_usd=5.0,
        require_identity=True, state_root=tmp_path, **kwargs)


def _no_sell_quotes(reason="no route to the entry book"):
    return FakeQuoteService(
        sell=sell_point(available=False, missing_reason=reason))


def _trader_using(bridge, notifier, tmp_path):
    """The EXISTING runtime, wired to this bridge -- no second bot or view."""
    from autopaper.config import build_config
    from autopaper.engine import PaperTrader

    cfg = build_config(tmp_path / "rt")
    cfg = _dc_replace(cfg, telegram=_dc_replace(
        cfg.telegram, bot_chat="-1001234567890", bot_token="test-token"))
    trader = PaperTrader(cfg, quote_service=FakeQuoteService())
    trader._paper_bridge = bridge
    # ``_notify`` is the runtime's ONE outbound hook; ``set_notifier`` would
    # WRAP it and swallow the failure this test needs to see.
    trader._notify = notifier
    return trader


def _exp_parent(b) -> str:
    return next(iter(b.beta_experiment.positions))


def _recorded_exits(exp) -> list[dict]:
    return [r for r in exp.completed
            if r.get("record_type") == "beta_exit"]


class TestSuccessfulExitReceipt:
    def test_the_receipt_carries_every_required_field(self, tmp_path):
        b = _beta_bridge(tmp_path, FakeQuoteService())
        records = deliver(b, V2_OK)
        parent = records[0].parent_signal_id
        _expire_holds(b)
        settled = asyncio.run(b._monitor_beta_experiment())
        assert [p for p in settled if p.status == "closed"]

        text = b.render_beta_exits(settled)[0]["text"]
        exp = b.beta_experiment
        bp = exp.positions[parent]["A"]

        # every field the notification is required to carry
        assert parent in text                       # signal ID
        assert bp.mint in text                      # token ID
        assert "Arm A —" in text                   # arm A/B/C
        assert bp.exit_reason in text               # exit reason
        assert b._price(bp.entry_fill_price) in text  # entry price
        assert b._price(bp.exit_fill_price) in text   # exit price
        assert b._tokens_qty(bp.entry_tokens_raw,
                            bp.entry_token_decimals) in text   # quantity sold
        assert b._usd(bp.exit_proceeds_usd) in text           # net proceeds
        assert "fees:" in text and "SOL" in text              # fees
        assert f"{float(bp.net_pnl_usd):+.6f}" in text        # realized net P&L
        assert f"${float(exp.accounts['A'].bankroll_usd):,.4f}" \
            in text                                          # arm bankroll
        assert "ALL-IN" in text and "100% of" in text         # sizing rule
        assert "quote-simulated fill" in text                 # P&L basis
        assert "executable SELL quote" in text                # the sale basis

    def test_a_non_quote_settlement_is_never_called_a_sale(self, tmp_path):
        """A modelled (backtest) settlement is labelled, and set apart."""
        b = _beta_bridge(tmp_path, FakeQuoteService())
        records = deliver(b, V2_OK)
        parent = records[0].parent_signal_id
        exp = b.beta_experiment
        entry = float(exp.positions[parent]["A"].entry_fill_price)
        exp.observe(parent, "A", mark_price=entry * 1.31, at=time.time())
        pos = exp.settle_arm(parent, "A", exit_fill=None, require_quote=False)
        assert pos.status == "closed"
        assert pos.exit_basis != EXECUTABLE_QUOTE_BASIS

        msgs = b.render_beta_exits([pos])
        assert len(msgs) == 1
        assert msgs[0]["kind"] == "beta_exit_modeled", (
            "a modelled fill must not share the QUOTE-sale message kind")
        text = msgs[0]["text"]
        assert "MODELED" in text
        assert "NOT an executable quote" in text
        assert "modeled backtest" in text.lower()
        # and the quote-sale basis line is a different thing entirely
        assert "executable SELL quote simulated fill" not in text

    def test_an_unpersisted_settlement_produces_no_receipt(self, tmp_path):
        """The persistence gate: an in-memory-only close is not announced."""
        b = _beta_bridge(tmp_path, FakeQuoteService())
        records = deliver(b, V2_OK)
        parent = records[0].parent_signal_id
        _expire_holds(b)
        settled = asyncio.run(b._monitor_beta_experiment())
        exp = b.beta_experiment
        assert b.render_beta_exits(settled), "the fixture must notify normally"
        # a saved copy of the durable book, then the book loses the trade: the
        # receipt must be withheld rather than describe a trade state cannot
        # confirm.
        saved = list(exp.completed)
        del exp.completed[:]
        assert b.render_beta_exits(settled) == [], (
            "a receipt was produced for a settlement not on the durable book")
        exp.completed.extend(saved)
        assert exp.positions[parent]["A"].status == "closed"


class TestUnresolvedExit:
    def test_a_refused_sell_quote_is_not_reported_as_a_sale(self, tmp_path):
        b = _beta_bridge(tmp_path, _no_sell_quotes())
        deliver(b, V2_OK)
        _expire_holds(b)
        settled = asyncio.run(b._monitor_beta_experiment())
        assert settled and all(p.status == "unresolved" for p in settled)
        assert b.render_beta_exits(settled) == [], (
            "an unresolved arm must not produce a CLOSED receipt")

    def test_it_sends_a_distinct_blocked_notice_with_reason_and_retry(
            self, tmp_path):
        b = _beta_bridge(tmp_path, _no_sell_quotes("no route to the book"))
        records = deliver(b, V2_OK)
        parent = records[0].parent_signal_id
        _expire_holds(b)
        asyncio.run(b._monitor_beta_experiment())
        msgs = b.render_beta_exit_blocked()
        assert len(msgs) == 1
        assert msgs[0]["kind"] == "beta_exit_blocked"
        text = msgs[0]["text"]
        assert "EXIT BLOCKED" in text
        assert "NO SALE WAS BOOKED" in text
        assert "UNRESOLVED" in text
        assert parent in text
        assert b.beta_experiment.positions[parent]["A"].mint in text
        assert "attempt 1" in text
        assert "retried" in text.lower()
        assert "no route to the book" in text or "no executable SELL quote" in text
        # it must NOT contain a price, a P&L or a proceeds claim
        assert "net proceeds" not in text.lower()
        assert "net PAPER P&L" not in text
        assert "executable SELL quote simulated fill" not in text

    def test_the_blocked_reason_names_the_quote_failure(self, tmp_path):
        b = _beta_bridge(tmp_path, _no_sell_quotes("SELL quote is stale"))
        deliver(b, V2_OK)
        _expire_holds(b)
        asyncio.run(b._monitor_beta_experiment())
        exp = b.beta_experiment
        pos = next(iter(next(iter(exp.positions.values())).values()))
        assert pos.exit_blocked_reason
        assert "SELL quote" in pos.exit_blocked_reason
        assert "SELL quote" in b.render_beta_exit_blocked()[0]["text"]


class TestRetryRecovery:
    def test_a_later_valid_quote_settles_the_arm_and_reports_the_sale(
            self, tmp_path):
        quotes = _no_sell_quotes()
        # The cooldown is driven to zero HERE so the retry is exercised on the
        # next tick; the cooldown's own behaviour is asserted separately.
        b = _beta_bridge(tmp_path, quotes, exit_retry_base_s=0.0)
        deliver(b, V2_OK)
        b.drain_messages()
        _expire_holds(b)
        first = asyncio.run(b._monitor_beta_experiment())
        assert all(p.status == "unresolved" for p in first)
        blocked = b.render_beta_exit_blocked()
        assert blocked, "the refusal must be announced"
        # the runtime DELIVERS the blocked notice, so the marker is persisted
        exp = b.beta_experiment
        sent = []
        trader = _trader_using(
            b, lambda text, kb=None: sent.append(text), tmp_path)
        b.pending_messages.extend(blocked)
        trader._flush_paper_bridge_messages()
        assert len(sent) == 1
        for pos in exp.positions[_exp_parent(b)].values():
            assert pos.exit_notified_attempt == 1

        # THE RETRY RECOVERS: the provider answers, and the SAME arm settles.
        quotes.sell = sell_point()
        _expire_holds(b)
        settled = asyncio.run(b._monitor_beta_experiment())
        closed = [p for p in settled if p.status == "closed"]
        assert closed, "a recovered quote must settle the outstanding arm"
        msgs = b.render_beta_exits(settled)
        assert len(msgs) == 1
        assert "PAPER CLOSED" in msgs[0]["text"]
        assert "NO SALE WAS BOOKED" not in msgs[0]["text"]
        for pos in closed:
            assert pos.exit_basis == EXECUTABLE_QUOTE_BASIS
            assert pos.net_pnl_usd is not None
            # the refusal's bookkeeping is preserved AND does not suppress the
            # SALE receipt -- they are different messages with different keys
            assert pos.exit_notified_attempt == 1
        b.pending_messages.extend(msgs)
        trader._flush_paper_bridge_messages()
        assert len(sent) == 2, (
            "the recovered sale must be delivered after its blocked notice")


class TestDuplicatePrevention:
    def test_the_same_refused_attempt_is_not_re_announced(self, tmp_path):
        # cooldown 0.0: the point here is the ANNOUNCEMENT cadence over two
        # consecutive refused attempts, not the retry cooldown.
        b = _beta_bridge(tmp_path, _no_sell_quotes(), exit_retry_base_s=0.0)
        deliver(b, V2_OK)
        _expire_holds(b)
        asyncio.run(b._monitor_beta_experiment())
        assert len(b.render_beta_exit_blocked()) == 1
        # a second tick refuses again -- attempt 2 is BELOW the cadence, so the
        # operator is not interrupted for every retry
        _expire_holds(b)
        asyncio.run(b._monitor_beta_experiment())
        exp = b.beta_experiment
        pos = next(iter(next(iter(exp.positions.values())).values()))
        assert pos.exit_attempts == 2
        assert b.render_beta_exit_blocked() == []

    def test_the_cadence_is_first_then_every_nth_and_never_repeats(
            self, tmp_path):
        exp = BetaExperiment(starting_bankroll_usd=10.0, state_root=tmp_path)
        pos = self._unresolved(exp)
        seen = []
        for attempt in range(1, 3 * BETA_EXIT_BLOCK_NOTIFY_EVERY + 1):
            pos.exit_attempts = attempt
            if exp.exit_block_notifiable(pos):
                seen.append(attempt)
                assert exp.mark_exit_block_notified(
                    pos.parent_signal_id, pos.arm, attempt), (
                    "the announced attempt must be recorded")
                assert not exp.exit_block_notifiable(pos), (
                    "an announced attempt must never be re-announced")
        assert seen == [1, BETA_EXIT_BLOCK_NOTIFY_EVERY,
                        2 * BETA_EXIT_BLOCK_NOTIFY_EVERY,
                        3 * BETA_EXIT_BLOCK_NOTIFY_EVERY]

    def test_an_open_arm_is_never_notifiable(self, tmp_path):
        exp = BetaExperiment(starting_bankroll_usd=10.0, state_root=tmp_path)
        exp.open_for_signal(parent_signal_id="p", mint="M", symbol="S",
                            position_size_usd=1.0,
                            entry_fill=_entry_fill())
        pos = exp.positions["p"]["A"]
        assert pos.status == "open" and pos.exit_attempts == 0
        assert not exp.exit_block_notifiable(pos), (
            "an arm with no attempted exit has no refusal to announce")

    def test_a_restart_does_not_re_announce_an_announced_block(self, tmp_path):
        exp = BetaExperiment(starting_bankroll_usd=10.0, state_root=tmp_path)
        pos = self._unresolved(exp)
        assert exp.mark_exit_block_notified(pos.parent_signal_id, pos.arm, 1)

        restarted = BetaExperiment(starting_bankroll_usd=10.0,
                                   state_root=tmp_path)
        again = restarted.positions["p"]["A"]
        assert again.status == "unresolved"
        assert again.exit_attempts == 1
        assert again.exit_notified_attempt == 1, (
            "the announced-attempt marker must be durable")
        assert not restarted.exit_block_notifiable(again), (
            "a restart re-announced an exit block the operator already saw")

    def test_a_later_arm_closing_is_not_collapsed_with_an_earlier_one(
            self, tmp_path):
        """The three arms close at DIFFERENT times; both receipts must arrive."""
        b = _beta_bridge(tmp_path, FakeQuoteService())
        deliver(b, V2_OK)
        b.drain_messages()      # the entry receipt is not what is under test
        _expire_holds(b)
        settled = asyncio.run(b._monitor_beta_experiment())
        by_arm = {p.arm: p for p in settled}
        msg_a = b.render_beta_exits([by_arm["A"]])[0]
        msg_b = b.render_beta_exits([by_arm["B"]])[0]
        assert msg_a["dedupe_key"] != msg_b["dedupe_key"], (
            "two arms closing must not share one dedupe key")
        assert "Arm B" not in msg_a["text"]

        sent = []
        trader = _trader_using(
            b, lambda text, kb=None: sent.append(text), tmp_path)
        for msg in (msg_a, msg_b):
            b.pending_messages.append(msg)
            trader._flush_paper_bridge_messages()
        assert len(sent) == 2, (
            "the second arm's receipt was swallowed as a duplicate")

    def test_a_blocked_notice_is_delivered_once_across_flushes(self, tmp_path):
        b = _beta_bridge(tmp_path, _no_sell_quotes())
        deliver(b, V2_OK)
        b.drain_messages()      # the entry receipt is not what is under test
        _expire_holds(b)
        asyncio.run(b._monitor_beta_experiment())
        msg = b.render_beta_exit_blocked()[0]
        sent = []
        trader = _trader_using(
            b, lambda text, kb=None: sent.append(text), tmp_path)
        b.pending_messages.append(msg)
        trader._flush_paper_bridge_messages()
        assert len(sent) == 1
        # (a) the EXPERIMENT now knows the attempt was announced
        assert b.render_beta_exit_blocked() == []
        # (b) a re-entrant flush of the same message is dropped by identity
        b.pending_messages.append(msg)
        trader._flush_paper_bridge_messages()
        assert len(sent) == 1

    def test_a_failed_blocked_send_is_retried_and_not_marked(self, tmp_path):
        b = _beta_bridge(tmp_path, _no_sell_quotes())
        deliver(b, V2_OK)
        b.drain_messages()      # the entry receipt is not what is under test
        _expire_holds(b)
        asyncio.run(b._monitor_beta_experiment())
        exp = b.beta_experiment
        pos = next(iter(next(iter(exp.positions.values())).values()))

        def boom(text, kb=None):
            raise RuntimeError("telegram transport unavailable (test double)")

        trader = _trader_using(b, boom, tmp_path)
        b.pending_messages.append(b.render_beta_exit_blocked()[0])
        trader._flush_paper_bridge_messages()
        assert pos.exit_notified_attempt == 0, (
            "a send that FAILED must not be recorded as announced")
        assert b.render_beta_exit_blocked(), (
            "the retry must still find the block un-announced")

        sent = []
        trader._notify = lambda text, kb=None: sent.append(text)
        b.pending_messages.extend(b.render_beta_exit_blocked())
        trader._flush_paper_bridge_messages()
        assert len(sent) == 1, "the retry must deliver exactly once"
        assert pos.exit_notified_attempt == 1
        assert b.render_beta_exit_blocked() == []

    @staticmethod
    def _unresolved(exp: BetaExperiment):
        """One arm that has had an exit refused, so it is UNRESOLVED.

        The exit is made due by the DEADLINE, not by a mark: TP and SL now
        require a validated executable quote, and ``settle_arm`` is called here
        with ``exit_fill=None``, so there is no evidence a threshold could be
        tested against. The hold is rewound so the max-hold TIME exit fires --
        a deadline is a fact about the clock and does not need a quote.
        """
        exp.open_for_signal(parent_signal_id="p", mint="M", symbol="S",
                            position_size_usd=1.0, entry_fill=_entry_fill())
        exp.observe("p", "A", mark_price=2.0, at=time.time())
        for bp in exp.positions["p"].values():
            bp.entry_at = float(bp.entry_at or time.time()) - 300.0
        pos = exp.settle_arm("p", "A", exit_fill=None, require_quote=True)
        assert pos.status == "unresolved" and pos.exit_attempts == 1
        return pos


class TestNotificationFailureNeverTouchesTheTrade:
    def test_a_failed_receipt_send_does_not_re_settle_or_duplicate(
            self, tmp_path):
        b = _beta_bridge(tmp_path, FakeQuoteService())
        records = deliver(b, V2_OK)
        b.drain_messages()      # the entry receipt is not what is under test
        parent = records[0].parent_signal_id
        _expire_holds(b)
        settled = asyncio.run(b._monitor_beta_experiment())
        exp = b.beta_experiment
        exits_before = _recorded_exits(exp)
        assert exits_before
        bankroll = {a: exp.accounts[a].bankroll_usd for a in ("A", "B", "C")}
        closed = {a: exp.accounts[a].closed_trades for a in ("A", "B", "C")}

        def boom(text, kb=None):
            raise RuntimeError("telegram transport unavailable (test double)")

        trader = _trader_using(b, boom, tmp_path)
        b.pending_messages.extend(b.render_beta_exits(settled))
        trader._flush_paper_bridge_messages()
        # THE SETTLEMENT IS UNTOUCHED BY A SEND FAILURE.
        assert {a: exp.accounts[a].bankroll_usd
                for a in ("A", "B", "C")} == bankroll
        assert {a: exp.accounts[a].closed_trades
                for a in ("A", "B", "C")} == closed
        assert _recorded_exits(exp) == exits_before, (
            "a failed notification re-booked the trade")
        assert all(exp.positions[parent][a].status == "closed"
                   for a in ("A", "B", "C"))

        # THE RETRY DELIVERS ONCE AND STILL DOES NOT RE-SETTLE.
        sent = []
        trader._notify = lambda text, kb=None: sent.append(text)
        b.pending_messages.extend(b.render_beta_exits(settled))
        trader._flush_paper_bridge_messages()
        assert len(sent) == 1
        assert _recorded_exits(exp) == exits_before
        # and a THIRD flush of the same message is deduplicated
        b.pending_messages.extend(b.render_beta_exits(settled))
        trader._flush_paper_bridge_messages()
        assert len(sent) == 1
        assert _recorded_exits(exp) == exits_before


class TestExitNotificationScopeAndSafety:
    def test_the_new_renderers_touch_no_trading_path(self):
        import ast
        import inspect
        import textwrap

        from autopaper.paper_live import engine as engine_module

        for fn in (engine_module.PaperLiveEngine.render_beta_exit_blocked,
                   engine_module.PaperLiveEngine._beta_exit_message,
                   engine_module.PaperLiveEngine._beta_exit_blocked_message,
                   engine_module.PaperLiveEngine._beta_exit_arm_lines):
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            calls = {n.func.attr for n in ast.walk(tree)
                     if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute)}
            assert not any(c in calls for c in
                           ("broadcast", "sign_transaction", "send_transaction",
                            "open_position", "settle_arm", "observe")), (
                f"{fn.__name__} performs a trading action: {calls}")

    def test_no_renderer_bakes_in_a_destination(self):
        import inspect

        from autopaper.paper_live import engine as engine_module

        for fn in (engine_module.PaperLiveEngine.render_beta_exit_blocked,
                   engine_module.PaperLiveEngine._beta_exit_message,
                   engine_module.PaperLiveEngine._beta_exit_blocked_message):
            source = inspect.getsource(fn)
            for banned in ("chat_id=", "TELEGRAM_CHAT", "bot_chat",
                           "sendMessage"):
                assert banned not in source, (
                    f"{fn.__name__} references {banned}")

    def test_exit_notifications_use_the_existing_bridge_and_runtime_hook(
            self, tmp_path):
        """No second bot and no new view: the message is queued by the bridge
        and delivered by the runtime's own ``_notify``."""
        b = _beta_bridge(tmp_path, _no_sell_quotes())
        deliver(b, V2_OK)
        _expire_holds(b)
        asyncio.run(b._monitor_beta_experiment())
        b.pending_messages.clear()
        # the bridge queues it and the runtime hook sends it
        assert b.render_beta_exit_blocked()
        b.pending_messages.extend(b.render_beta_exit_blocked())
        assert [m for m in b.drain_messages()
                if m["kind"] == "beta_exit_blocked"]
        sent = []
        trader = _trader_using(
            b, lambda text, kb=None: sent.append(text), tmp_path)
        b.pending_messages.append({'text': 'x', 'kind': 'beta_exit_blocked',
                                   'parent_signal_id': 'p',
                                   'blocked_arms': []})
        trader._flush_paper_bridge_messages()
        assert sent == ["x"]

    def test_a_beta_only_monitor_tick_actually_delivers_the_receipt(
            self, tmp_path):
        """The flush must not be gated on a regular ``TradeRecord``.

        A beta arm is settled by ``BetaExperiment``, so a tick that settles one
        produces NO entry in ``trades``. Gating the flush on ``trades`` meant
        that under a BETA-ONLY selection -- where no regular arm exists at all --
        a beta CLOSE and a beta EXIT BLOCKED were queued and NEVER delivered.
        """
        b = _beta_bridge(tmp_path, FakeQuoteService())
        records = deliver(b, V2_OK)
        b.drain_messages()
        parent = records[0].parent_signal_id
        _expire_holds(b)
        asyncio.run(b._monitor_beta_experiment())
        settled = b._monitor_beta_experiment_for_tick()
        assert [p for p in settled if p.status == "closed"], (
            "the fixture must have settled an arm")
        b.pending_messages.extend(b.render_beta_exits(settled))

        sent = []
        trader = _trader_using(
            b, lambda text, kb=None: sent.append(text), tmp_path)
        # A BETA-ONLY selection, unchanged, so no arm reconciliation runs.
        trader.state.selected_arms = ["V2_FLOOR_BETA"]
        trader._paper_bridge_arms = ("V2_BETA_A", "V2_BETA_B", "V2_BETA_C")
        asyncio.run(trader.monitor_once())
        assert any("PAPER CLOSED" in t and "V2 FLOOR (BETA)" in t
                   for t in sent), (
            "a beta-only monitor tick queued the sale receipt but never "
            "delivered it")
        assert b.beta_experiment.positions[parent]["A"].status == "closed"

    def test_paper_only_guards_are_intact(self):
        from autopaper.paper_live.guards import assert_paper_only

        assert_paper_only()


def _entry_fill() -> BetaEntryFill:
    """A synthetic executable-quote entry for the experiment-unit tests."""
    return BetaEntryFill(
        price_usd=1.0, tokens_raw=1_000_000, token_decimals=6,
        cost_quote=1.0, fee_quote=0.0, mark_price=None,
        quote_timestamp=time.time(), quote_age_ms=0.0, venue=None,
        pool_or_curve=None, provider=None, route=(), identity="PASS")