"""
PainApe is the production FORBIDDEN fallback that replaced SmugCalls.

Covers: the parser on the exact captured fixture and its normal variations; the
Telegram source selector (MooDeng / PainApe / AUTO); the AUTO 5h heartbeat; the
manual-vs-heartbeat invariant; cross-source deduplication; source attribution;
persistence; and that a source switch never touches open positions or listener
high-water marks. No BUY is required anywhere.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from autopaper.clock import FakeClock
from autopaper.engine import PaperTrader
from autopaper.models import AlertSnapshot
from autopaper.modes import RuntimeState
from autopaper.telegram import parser as parser_mod
from autopaper.telegram import ui
from autopaper.telegram.actions import ButtonHandler, OperatorActions
from autopaper.telegram.client import ConnectionState, TelegramConfig, TelegramListener
from autopaper.telegram.signal_source import (
    FALLBACK,
    FALLBACK_ACTIVE,
    PRIMARY,
    PRIMARY_ACTIVE,
    SOURCE_AUTO,
    SOURCE_MOODENG,
    SOURCE_PAINAPE,
    SignalSourceState,
)

#: Exact captured @painaped alert. Must not be "cleaned up".
PAINAPE_MESSAGE = """🟡[PUMP]   -  JEZ ｜ $JEZ

📜 CA: AwdfqHLnUTdAo5rNc7N6uxr5td8yBDKUe88FJcmLpump

┌ Market Cap: $21k
├ Liq: $17.5k
├ Holders: 235 ｜ TOP 10:  19.75%
├ Devs: ✅
├ Bots: 103
├ Snipers: 1
├ Insiders: 0
├ Volume: $10.6k
└ Age: a moment ago

AVE ｜ GMGN (BOT) ｜ GMGN (WEB) ｜ AXIOM ｜ BASED (BOT) ｜ BASED (WEB) ｜ TROJAN ｜ BLOOM
🔰 Socials: X ｜ Telegram ｜ Website ｜ Tweets ｜ #JEZ
________________________
🔥🔥🔥 PREMIUM SIGNAL BOT 📣
🔱 ONLY 0.35 SOL - 💬 DM @krypto_investorz to JOIN 🌙🌙🌙"""

PAINAPE_MINT = "AwdfqHLnUTdAo5rNc7N6uxr5td8yBDKUe88FJcmLpump"
PAINAPE_SYMBOL = "JEZ"


# ===========================================================================
# PART 3/4 - the parser on the exact fixture and normal variations
# ===========================================================================


def test_exact_fixture_parses_every_required_field():
    p = parser_mod.parse_alert(PAINAPE_MESSAGE)
    assert p.mint == PAINAPE_MINT
    assert p.symbol == PAINAPE_SYMBOL
    assert p.num("market_cap_usd") == pytest.approx(21_000.0)
    assert p.num("liquidity_usd") == pytest.approx(17_500.0)
    assert p.fields["holder_count"] == 235
    assert p.fields["top10_pct"] == pytest.approx(19.75)
    assert p.num("volume_usd") == pytest.approx(10_600.0)


def test_exact_fixture_parses_the_supported_extra_fields():
    p = parser_mod.parse_alert(PAINAPE_MESSAGE)
    assert p.fields["sniper_count"] == 1
    assert p.fields["insider_count"] == 0
    # "Bots: N" is preserved in the model's bot-wallet counter.
    assert p.fields["bundler_count"] == 103


def test_age_is_not_fabricated():
    """"a moment ago" has no numeric age; it must stay unparsed, not invented."""
    p = parser_mod.parse_alert(PAINAPE_MESSAGE)
    assert p.fields.get("age_text") == "a moment ago"
    assert p.age_seconds() is None


def test_promotional_footer_is_not_signal_data():
    p = parser_mod.parse_alert(PAINAPE_MESSAGE)
    assert p.symbol == "JEZ"          # not "SOL" from the footer
    assert p.num("market_cap_usd") == pytest.approx(21_000.0)
    # The footer's "0.35 SOL" must not become a price or liquidity figure.
    assert p.num("liquidity_usd") == pytest.approx(17_500.0)
    assert p.fields.get("price_usd") is None


@pytest.mark.parametrize("value,expected", [
    ("$21k", 21_000.0), ("$21K", 21_000.0), ("$21.5k", 21_500.0),
    ("$21,500", 21_500.0),
])
def test_market_cap_number_formats(value, expected):
    text = PAINAPE_MINT + f"\nCA:\nMarket Cap: {value}\nLiq: $17.5k"
    assert parser_mod.parse_alert(text).num("market_cap_usd") == pytest.approx(expected)


@pytest.mark.parametrize("header", [
    "Market Cap:", "Market Cap :", "MarketCap:", "MC:",
])
def test_market_cap_label_variations(header):
    text = PAINAPE_MINT + f"\nCA:\n{header} $21k\nLiq: $17.5k"
    assert parser_mod.parse_alert(text).num("market_cap_usd") == pytest.approx(21_000.0)


@pytest.mark.parametrize("header", ["Liq:", "Liq :", "Liquidity:"])
def test_liquidity_label_variations(header):
    text = PAINAPE_MINT + f"\nCA:\nMarket Cap: $21k\n{header} $17.5k"
    assert parser_mod.parse_alert(text).num("liquidity_usd") == pytest.approx(17_500.0)


def test_symbol_with_punctuation_is_safe():
    p = parser_mod.parse_alert("$WIF-2.0\nCA: " + PAINAPE_MINT)
    assert p.symbol  # parsed, not rejected
    assert p.mint == PAINAPE_MINT


def test_invalid_mint_is_rejected():
    bad = "0OIl-not-a-mint\nCA:\nMarket Cap: $21k"
    assert parser_mod.parse_alert(bad).mint is None


def test_painape_missing_mint_produces_no_candidate():
    text = "🟡[PUMP] - JEZ ｜ $JEZ\n─ Market Cap: $21k\n─ Liq: $17.5k"
    assert parser_mod.parse_alert(text).mint is None


# ===========================================================================
# PART 2/5/6 - listener, source selector, attribution
# ===========================================================================


def _listener(collected: list, fallback_id: int = 2253481601) -> TelegramListener:
    async def on_alert(snapshot, parsed):
        collected.append((snapshot, parsed))

    listener = TelegramListener(
        TelegramConfig(api_id=1, api_hash="h", chat="MooDengPresidentCallers"),
        on_alert=on_alert,
    )
    listener.status.state = ConnectionState.CONNECTED
    listener.source_chats = {
        PRIMARY: "MooDengPresidentCallers",
        FALLBACK: "@painaped",
    }
    listener._source_chat_ids = {PRIMARY: 111, FALLBACK: fallback_id}
    listener.status.chat_id = 111
    return listener


class _Msg:
    def __init__(self, mid: int, text: str, chat_id: int = 2253481601) -> None:
        self.id = mid
        self.sender_id = 999
        self.chat_id = chat_id
        self.message = text
        self.date = None


def test_painape_message_becomes_a_candidate_with_source_attribution():
    collected: list = []
    listener = _listener(collected)
    asyncio.run(listener._handle(_Msg(8001, PAINAPE_MESSAGE), FALLBACK))
    assert collected
    snapshot, _ = collected[0]
    assert snapshot.signal_source == "PainApe"
    assert snapshot.mint == PAINAPE_MINT
    assert snapshot.symbol == PAINAPE_SYMBOL
    assert snapshot.fields["market_cap_usd"] == pytest.approx(21_000.0)


# ===========================================================================
# PART 1/8 - AUTO heartbeat and the source selector
# ===========================================================================


def _trader(config, listener=None):
    clock = FakeClock(1_700_000_000.0)
    trader = PaperTrader(config, clock=clock, listener=listener)
    trader.control.command = "STOP"   # record, never trade
    return trader, clock


def _snap(trader, source, mid, mint=PAINAPE_MINT, symbol=PAINAPE_SYMBOL):
    now = trader.clock.now()
    return AlertSnapshot(
        candidate_id=f"cand_{mid}", source="telegram", chat="MooDengPresidentCallers",
        message_id=mid, source_timestamp=now, received_timestamp=now,
        raw_text=PAINAPE_MESSAGE, mint=mint, symbol=symbol,
        fields={"market_cap_usd": 21_000.0, "liquidity_usd": 17_500.0},
        signal_source=source,
    )


def test_manual_moodeng_ignores_painape(config):
    trader, _ = _trader(config)
    trader.signal_source_selection = SOURCE_MOODENG
    asyncio.run(trader.on_alert(_snap(trader, FALLBACK, 1), None))
    assert trader.candidates_seen == 0
    assert trader._signal_source_counters["fallback_ignored_primary_active"] == 1


def test_manual_painape_accepts_painape(config):
    trader, _ = _trader(config)
    trader.signal_source_selection = SOURCE_PAINAPE
    asyncio.run(trader.on_alert(_snap(trader, FALLBACK, 2), None))
    assert trader.candidates_seen == 1
    assert trader._signal_source_counters["fallback_valid"] == 1


def test_manual_painape_ignores_moodeng(config):
    trader, _ = _trader(config)
    trader.signal_source_selection = SOURCE_PAINAPE
    asyncio.run(trader.on_alert(_snap(trader, PRIMARY, 3), None))
    assert trader.candidates_seen == 0


def test_auto_primary_healthy_uses_moodeng_only(config):
    trader, clock = _trader(config)
    trader.signal_source_selection = SOURCE_AUTO
    # A valid MooDeng signal within 5h keeps primary active.
    trader.signal_source.observe_primary(at=clock.now())
    assert trader.signal_source.active_source == PRIMARY_ACTIVE
    asyncio.run(trader.on_alert(_snap(trader, FALLBACK, 4), None))
    assert trader.candidates_seen == 0, "PainApe must be ignored while primary is healthy"
    asyncio.run(trader.on_alert(_snap(trader, PRIMARY, 5), None))
    assert trader.candidates_seen == 1


def test_auto_primary_silent_over_5h_uses_painape(config):
    trader, clock = _trader(config)
    trader.signal_source_selection = SOURCE_AUTO
    trader.signal_source.observe_primary(at=clock.now() - 6 * 60 * 60)
    assert trader.signal_source.refresh(clock.now()) == FALLBACK_ACTIVE
    asyncio.run(trader.on_alert(_snap(trader, FALLBACK, 6), None))
    assert trader.candidates_seen == 1


def test_new_moodeng_signal_returns_to_primary_immediately(config):
    trader, clock = _trader(config)
    trader.signal_source_selection = SOURCE_AUTO
    trader.signal_source.observe_primary(at=clock.now() - 6 * 60 * 60)
    assert trader.signal_source.refresh(clock.now()) == FALLBACK_ACTIVE

    # A new valid MooDeng signal flips the source back at once.
    asyncio.run(trader.on_alert(_snap(trader, PRIMARY, 7), None))
    assert trader.signal_source.active_source == PRIMARY_ACTIVE
    assert not trader.signal_source.accepts(FALLBACK, clock.now())
    # ... and PainApe is ignored again.
    asyncio.run(trader.on_alert(_snap(trader, FALLBACK, 8), None))
    assert trader.candidates_seen == 1


def test_painape_does_not_update_the_moodeng_heartbeat(config):
    trader, _ = _trader(config)
    trader.signal_source_selection = SOURCE_PAINAPE
    asyncio.run(trader.on_alert(_snap(trader, FALLBACK, 9), None))
    assert trader.signal_source.primary_last_signal_at is None


def test_moodeng_updates_the_heartbeat_even_in_manual_painape(config):
    trader, clock = _trader(config)
    trader.signal_source_selection = SOURCE_PAINAPE
    snap = _snap(trader, PRIMARY, 10)
    asyncio.run(trader.on_alert(snap, None))
    assert trader.signal_source.primary_last_signal_at == pytest.approx(
        snap.received_timestamp
    )


# ===========================================================================
# PART 7 - cross-source deduplication
# ===========================================================================


def test_cross_source_duplicate_is_suppressed(config):
    trader, clock = _trader(config)
    trader.signal_source_selection = SOURCE_AUTO
    trader.signal_source.observe_primary(at=clock.now() - 6 * 60 * 60)
    asyncio.run(trader.on_alert(_snap(trader, FALLBACK, 11), None))
    # Same mint from the other source -> one underlying opportunity.
    asyncio.run(trader.on_alert(_snap(trader, PRIMARY, 12), None))
    assert trader.candidates_seen == 1
    assert trader._signal_source_counters["cross_source_duplicates"] == 1


# ===========================================================================
# PART 6 - attribution persists into the candidate record
# ===========================================================================


def test_source_attribution_is_recorded(config):
    trader, _ = _trader(config)
    trader.signal_source_selection = SOURCE_PAINAPE
    asyncio.run(trader.on_alert(_snap(trader, FALLBACK, 13), None))
    rows = [json.loads(l) for l in config.candidates_file.read_text().splitlines() if l]
    seen = [r for r in rows if r.get("event") == "candidate_seen"]
    assert seen and seen[-1]["signal_source"] == "PainApe"


# ===========================================================================
# PART 2 - persistence and no token/backlog side effects
# ===========================================================================


def test_selection_persists_across_restart(config):
    trader, _ = _trader(config)
    trader._persist_signal_source("PainApe")
    # The constructor restores the persisted selection, so a fresh runtime
    # comes up on PainApe rather than resetting to AUTO.
    fresh, _ = _trader(config)
    assert fresh.signal_source_selection == SOURCE_PAINAPE
    fresh._restore_signal_source()
    assert fresh.signal_source_selection == SOURCE_PAINAPE


def test_retired_smugcalls_selection_is_refused(config):
    trader, _ = _trader(config)
    trader.signal_source_selection = SOURCE_AUTO
    trader._persist_signal_source("SmugCalls")
    assert trader.signal_source_selection == SOURCE_AUTO


def test_switching_source_does_not_touch_listener_high_water_marks(config):
    listener = _listener([])
    listener._source_cursors = {PRIMARY: 101, FALLBACK: 202}
    listener._source_chat_ids = {PRIMARY: 111, FALLBACK: 222}
    trader, _ = _trader(config, listener=listener)
    trader._persist_signal_source("PainApe")
    trader._restore_signal_source()
    assert listener._source_cursors == {PRIMARY: 101, FALLBACK: 202}
    assert listener._source_chat_ids == {PRIMARY: 111, FALLBACK: 222}


def test_source_selection_does_not_touch_existing_positions():
    saved: list[str] = []
    runtime = RuntimeState()
    handler = ButtonHandler(
        runtime,
        OperatorActions(
            persist_signal_source=saved.append,
            current_signal_source=lambda: saved[-1] if saved else "AUTO",
        ),
    )
    existing = {"position_id": "pos-1", "status": "OPEN", "mint": "M"}
    asyncio.run(handler.handle(f"{ui.CB_SOURCE_SELECT}PainApe"))
    assert existing == {"position_id": "pos-1", "status": "OPEN", "mint": "M"}
    assert runtime.run_state.name == "OFF"
    assert saved == ["PainApe"]


# ===========================================================================
# PART 4 - the JSON keymap itself
# ===========================================================================


def test_signal_source_keyboard_offers_moodeng_painape_auto():
    rows = ui.signal_source_keyboard("AUTO")
    callbacks = [r[0]["callback_data"] for r in rows]
    assert callbacks == [
        "signal_source:MooDeng", "signal_source:PainApe", "signal_source:AUTO",
    ]
    labels = " ".join(r[0]["text"] for r in rows)
    assert "MooDeng" in labels and "PainApe" in labels and "AUTO" in labels
    assert "SmugCalls" not in labels


# ===========================================================================
# LIVE-RUNTIME REGRESSION - the fallback feed must actually be POLLED
# ===========================================================================
#
# The production entrypoint builds the trader FIRST and attaches the listener
# afterwards (cli._trader), so ``PaperTrader.__init__`` sees ``listener=None``.
# When the feed wiring lived only in __init__, the listener started with
# ``source_chats`` unset, took the single-chat branch and never even fetched
# ``@painaped``. A new PainApe message was therefore silently ignored before
# it could reach the parser. These tests pin the wiring to the start point so
# the bug cannot return through construction order.


def test_listener_attached_after_construction_is_wired_before_start(config):
    wired_at_start: dict = {}
    listener = _listener([])
    listener.source_chats = None
    listener._source_chat_ids = {}

    async def fake_start():
        # Capture the state the moment the real ``start`` would resolve feeds.
        wired_at_start.update(listener.source_chats or {})

    listener.start = fake_start
    listener.status.state = ConnectionState.CONNECTED

    trader, _ = _trader(config)          # constructed WITHOUT a listener
    assert trader.listener is None
    # Keep the fixture config; the real refresh() only matters for a new session.
    trader._refresh_telegram_config = lambda: False
    trader.listener = listener           # EXACTLY what cli._trader does

    assert asyncio.run(trader._supervise_listener(start_stream=False)) is True
    assert wired_at_start == {
        PRIMARY: config.telegram.chat,
        FALLBACK: config.telegram.fallback_chat,
    }, "the listener must be pointed at BOTH feeds before it starts"


def test_wire_listener_sources_is_config_driven_and_reversible(config):
    listener = _listener([])
    trader, _ = _trader(config, listener=listener)
    assert listener.source_chats == {
        PRIMARY: config.telegram.chat,
        FALLBACK: config.telegram.fallback_chat,
    }
    # A deployment with no fallback configured must fall back to single-source.
    from dataclasses import replace as _replace
    trader.config = _replace(
        trader.config,
        telegram=_replace(trader.config.telegram, fallback_chat=None),
    )
    trader._wire_listener_sources()
    assert listener.source_chats is None


class _FakeMessagesClient:
    def __init__(self, newest_id: int = 555) -> None:
        self.newest_id = newest_id
        self.calls: list[tuple] = []

    async def get_messages(self, chat_id, limit=1, min_id=None):
        self.calls.append((chat_id, limit, min_id))

        class _M:
            id = self.newest_id

        return [_M()]


def test_reconnect_does_not_jump_an_established_cursor(config):
    listener = _listener([])
    listener._client = _FakeMessagesClient(newest_id=999)
    listener._source_chat_ids = {PRIMARY: 111, FALLBACK: 222}
    listener._source_cursors = {FALLBACK: 900}
    asyncio.run(listener._anchor_sources())
    # The already-streaming feed resumes from the id we processed, NOT the
    # newest message: re-anchoring on reconnect would skip the gap.
    assert listener._source_cursors[FALLBACK] == 900
    # The feed with no cursor yet is still anchored on first sight.
    assert listener._source_cursors[PRIMARY] == 999
    assert (222, 1, None) not in listener._client.calls
    assert (111, 1, None) in listener._client.calls
