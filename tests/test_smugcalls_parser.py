"""
SmugCalls PARSER regression - NOT a production source.

``@SmugCalls`` is no longer the production fallback (that is ``@painaped`` now,
see ``tests/test_painape_source.py``). The historical SmugCalls message FORMAT
is still supported by the parser, and this module keeps it working: the fixture
below is the real captured alert, and these tests must not be deleted.

The original defect: the parser silently dropped ``MC ~$10K`` and truncated the
symbol ``$401si(k)`` to ``401si``. The mint, ``Liq $6K`` and the candidate path
already worked. This module pins the fix and its safety. It does NOT require,
request or assert a BUY.
"""

from __future__ import annotations

import asyncio

import pytest

from autopaper.analysis import DecisionEngine, build_features
from autopaper.config import DecisionConfig, ExitConfig, TelegramConfig
from autopaper.models import AlertSnapshot, ObservationSet
from autopaper.nar import NullNarSource
from autopaper.telegram.client import ConnectionState, TelegramListener
from autopaper.telegram.parser import parse_alert

from .conftest import obs

# The EXACT message supplied for the investigation. Do not "clean" it up.
EXACT_SMUG_MESSAGE = """😏 SMUG CALL

$401si(k)
EbqrTQLVrieK2GmiBSffTuMc8CgcVViot2H7ePtCJir2

💊 MC ~$10K  ·  Liq $6K
⚡ Axiom  ·  📱 FOMO  ·  📈 GMGN
🧬 creator: 24 grads / 2507 launches

aped early. DYOR, fam. 🔥"""

SMUG_MINT = "EbqrTQLVrieK2GmiBSffTuMc8CgcVViot2H7ePtCJir2"
SMUG_SYMBOL = "401si(k)"
SMUG_MC = 10_000.0
SMUG_LIQ = 6_000.0

#: A historical source label. SmugCalls is NOT a configured production source
#: anymore, so this is just the tag these parser tests carry.
SMUG_SOURCE = "SmugCalls"


# ---------------------------------------------------------------------------
# 1. the exact message against the real parser
# ---------------------------------------------------------------------------


def test_exact_message_parses_the_symbol_including_parentheses():
    p = parse_alert(EXACT_SMUG_MESSAGE)
    assert p.symbol == SMUG_SYMBOL
    assert p.name == SMUG_SYMBOL


def test_exact_message_parses_the_mint():
    p = parse_alert(EXACT_SMUG_MESSAGE)
    assert p.mint == SMUG_MINT
    assert "no base58 mint" not in " ".join(p.warnings)


def test_exact_message_parses_the_approximation_market_cap():
    """``MC ~$10K`` must read 10_000 - the ``~`` is an approximation marker."""
    p = parse_alert(EXACT_SMUG_MESSAGE)
    assert p.num("market_cap_usd") == pytest.approx(SMUG_MC)


def test_exact_message_parses_liquidity():
    p = parse_alert(EXACT_SMUG_MESSAGE)
    assert p.num("liquidity_usd") == pytest.approx(SMUG_LIQ)


def test_exact_message_does_not_fabricate_unprovided_fields():
    p = parse_alert(EXACT_SMUG_MESSAGE)
    assert p.fields.get("volume_usd") is None
    assert p.fields.get("holder_count") is None
    assert p.fields.get("age_text") is None


# ---------------------------------------------------------------------------
# 2. the message through the LISTENER becomes a candidate
# ---------------------------------------------------------------------------


def _smug_listener(collected: list) -> TelegramListener:
    async def on_alert(snapshot, parsed):
        collected.append((snapshot, parsed))

    listener = TelegramListener(
        TelegramConfig(api_id=1, api_hash="h", chat="MooDengPresidentCallers"),
        on_alert=on_alert,
    )
    listener.status.state = ConnectionState.CONNECTED
    listener.source_chats = {SMUG_SOURCE: "@SmugCalls"}
    listener._source_chat_ids = {SMUG_SOURCE: 2253481601}
    listener.status.chat_id = 2253481601
    return listener


class _Msg:
    def __init__(self, mid: int, text: str, chat_id: int = 2253481601) -> None:
        self.id = mid
        self.sender_id = 999
        self.chat_id = chat_id
        self.message = text
        self.date = None


def test_telegram_message_becomes_a_candidate():
    collected: list = []
    listener = _smug_listener(collected)
    asyncio.run(listener._handle(_Msg(7001, EXACT_SMUG_MESSAGE), SMUG_SOURCE))

    assert collected, "the SmugCalls alert produced no candidate"
    snapshot, _parsed = collected[0]
    assert snapshot.signal_source == SMUG_SOURCE
    assert snapshot.mint == SMUG_MINT
    assert snapshot.symbol == SMUG_SYMBOL
    assert snapshot.fields["market_cap_usd"] == pytest.approx(SMUG_MC)
    assert snapshot.fields["liquidity_usd"] == pytest.approx(SMUG_LIQ)


# ---------------------------------------------------------------------------
# 3. the candidate continues into the EXISTING decision pipeline
# ---------------------------------------------------------------------------


def _decision_for(source: str):
    parsed = parse_alert(EXACT_SMUG_MESSAGE)
    now = 1_700_000_000.0
    snapshot = AlertSnapshot(
        candidate_id="cand_smug", source="telegram", chat="@SmugCalls",
        message_id=7001, source_timestamp=now - 1.0, received_timestamp=now,
        raw_text=EXACT_SMUG_MESSAGE, mint=parsed.mint, symbol=parsed.symbol,
        name=parsed.name, fields=dict(parsed.fields),
        provenance=dict(parsed.provenance), signal_source=source,
    )
    history = ObservationSet([
        obs(now - 30, 1.0e-5, mint=parsed.mint, market_cap_usd=None,
            liquidity_usd=None),
        obs(now - 15, 1.2e-5, mint=parsed.mint, market_cap_usd=None,
            liquidity_usd=None),
        obs(now, 1.4e-5, mint=parsed.mint, market_cap_usd=None,
            liquidity_usd=None),
    ])
    nar = NullNarSource().fetch(parsed.mint or "M")
    features = build_features(
        candidate_id="cand_smug", mint=parsed.mint or "M", decision_time=now,
        history=history, alert=snapshot, nar=nar,
        stale_after_ms=15_000.0, unusable_after_ms=45_000.0,
    )
    report = DecisionEngine(DecisionConfig(), ExitConfig()).decide(
        features=features, alert=snapshot, nar=nar, analysis_start=now,
        analysis_complete=now + 0.05,
    )
    return features, report


def test_parsed_values_reach_the_feature_set():
    features, _ = _decision_for(SMUG_SOURCE)
    assert features.market_cap_usd == pytest.approx(SMUG_MC)
    assert features.liquidity_usd == pytest.approx(SMUG_LIQ)


def test_candidate_produces_a_decision_without_error():
    """The pipeline runs end to end; a BUY is explicitly NOT required."""
    _features, report = _decision_for(SMUG_SOURCE)
    assert report.decision.value in ("BUY", "WAIT", "REJECT")
    assert report.mint == SMUG_MINT
    assert report.thesis


# ---------------------------------------------------------------------------
# 4. malformed SmugCalls messages still fail safely
# ---------------------------------------------------------------------------


def test_no_mint_produces_no_candidate():
    collected: list = []
    listener = _smug_listener(collected)
    bad = "😏 SMUG CALL\n$401si(k)\n\n💊 MC ~$10K  ·  Liq $6K"
    assert parse_alert(bad).mint is None
    asyncio.run(listener._handle(_Msg(7101, bad), SMUG_SOURCE))
    assert not collected
    assert listener.status.unparsed_dropped == 1


def test_invalid_mint_produces_no_candidate():
    collected: list = []
    listener = _smug_listener(collected)
    bad = "😏 SMUG CALL\n$401si(k)\n0OIl-not-a-mint\n\n💊 MC ~$10K · Liq $6K"
    assert parse_alert(bad).mint is None
    asyncio.run(listener._handle(_Msg(7102, bad), SMUG_SOURCE))
    assert not collected


def test_missing_market_cap_is_not_invented():
    p = parse_alert("😏 SMUG CALL\n$401si(k)\n" + SMUG_MINT + "\n\n💊 Liq $6K")
    assert p.mint == SMUG_MINT
    assert p.fields.get("market_cap_usd") is None
    assert p.fields.get("liquidity_usd") == pytest.approx(SMUG_LIQ)


def test_garbled_text_produces_no_candidate():
    collected: list = []
    listener = _smug_listener(collected)
    for bad in ("", "   ", "%%%$$$", "aped early. DYOR, fam. 🔥"):
        asyncio.run(listener._handle(_Msg(7200, bad), SMUG_SOURCE))
    assert not collected


def test_duplicate_smugcalls_message_is_dropped_once():
    collected: list = []
    listener = _smug_listener(collected)
    msg = _Msg(7301, EXACT_SMUG_MESSAGE)
    asyncio.run(listener._handle(msg, SMUG_SOURCE))
    asyncio.run(listener._handle(msg, SMUG_SOURCE))
    assert len(collected) == 1, "the duplicate must not create a second candidate"
    assert listener.status.duplicates_dropped == 1
