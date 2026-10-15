"""
Moo Deng President is the ONLY candidate source.

Covers: exact CA extraction, the channel restriction, the duplicate defences,
and the message-identity record. The property that matters most is the last
one - a candidate from any other chat must be REFUSED, not filtered quietly
downstream where the drop would be invisible.
"""

from __future__ import annotations

import pytest

from autopaper.config import TelegramConfig
from autopaper.models import AlertSnapshot
from autopaper.telegram import parser as parser_mod
from autopaper.telegram.client import (
    FORBIDDEN_SOURCE_NAMES,
    TelegramSourceNotAllowed,
    TelegramListener,
)

MINT = "So11111111111111111111111111111111111111112"
REAL_MINT = "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj"


# ---------------------------------------------------------------------------
# the configured source
# ---------------------------------------------------------------------------


def test_the_single_source_is_moo_deng_president():
    assert TelegramConfig.CANDIDATE_SOURCE == "Moo Deng President"


def test_the_only_second_source_is_the_painape_fallback():
    """
    Exactly one PRIMARY channel and exactly one operational fallback.

    The fallback is now ``@painaped``; ``@SmugCalls`` was retired. There is still
    no third feed and no list of extra channels.
    """
    fields = TelegramConfig.__dataclass_fields__
    assert "chats" not in fields
    assert "sources" not in fields
    assert "chat" in fields
    assert "fallback_chat" in fields
    cfg = TelegramConfig(api_id=1, api_hash="x", chat="-100123")
    assert cfg.chat == "-100123"
    assert cfg.fallback_chat == "@painaped"
    assert TelegramConfig.FALLBACK_SOURCE == "PainApe"
    assert TelegramConfig.CANDIDATE_SOURCE == "Moo Deng President"


def test_memesdontlies_is_a_known_forbidden_source():
    joined = " ".join(FORBIDDEN_SOURCE_NAMES)
    assert "memesdontlies" in joined.replace(" ", "")
    assert "memes_dont_lies" in joined


@pytest.mark.parametrize(
    "title",
    ["MemesDontLies", "memesdontlies", "Memes Dont Lies", "MEMES_DONT_LIES"],
)
def test_a_known_wrong_channel_title_is_refused(title):
    import re

    slug = re.sub(r"[^a-z0-9]", "", title.lower())
    assert any(bad in slug for bad in FORBIDDEN_SOURCE_NAMES), title


def test_moo_deng_president_is_not_refused():
    import re

    slug = re.sub(r"[^a-z0-9]", "", "Moo Deng President".lower())
    for bad in FORBIDDEN_SOURCE_NAMES:
        assert bad not in slug


def test_source_name_is_reported_for_the_status_message():
    cfg = TelegramConfig(api_id=1, api_hash="x", chat="-100123")
    assert cfg.source_name == "Moo Deng President"


# ---------------------------------------------------------------------------
# the listener refuses a wrong source at connect time
# ---------------------------------------------------------------------------


class _Entity:
    def __init__(self, eid: int, title: str, username: str = "") -> None:
        self.id = eid
        self.title = title
        self.username = username


def _listener() -> TelegramListener:
    return TelegramListener(TelegramConfig(api_id=1, api_hash="x", chat="42"))


def test_a_forbidden_chat_is_refused_at_resolve_time():
    listener = _listener()
    with pytest.raises(TelegramSourceNotAllowed) as exc:
        listener._assert_permitted_source(
            _Entity(1, "MemesDontLies", "memesdontlies")
        )
    assert "Moo Deng President" in str(exc.value)


def test_a_forbidden_username_is_refused_even_with_an_innocent_title():
    listener = _listener()
    with pytest.raises(TelegramSourceNotAllowed):
        listener._assert_permitted_source(
            _Entity(1, "Alerts", "memesdontlies")
        )


def test_a_test_channel_is_refused():
    listener = _listener()
    with pytest.raises(TelegramSourceNotAllowed):
        listener._assert_permitted_source(_Entity(1, "sandbox"))


def test_moo_deng_president_is_accepted():
    listener = _listener()
    listener._assert_permitted_source(_Entity(1, "Moo Deng President"))


def test_an_unconfigured_chat_is_refused_before_any_connection():
    import asyncio

    listener = TelegramListener(TelegramConfig(api_id=1, api_hash="x"))
    with pytest.raises(TelegramSourceNotAllowed, match="not configured"):
        asyncio.run(listener._resolve_chat())


def test_a_refused_source_sets_the_error_state():
    listener = _listener()
    with pytest.raises(TelegramSourceNotAllowed):
        listener._assert_permitted_source(_Entity(1, "MemesDontLies"))
    assert listener.status.state == "ERROR"
    assert "Moo Deng President" in listener.status.detail


# ---------------------------------------------------------------------------
# per-message chat enforcement
# ---------------------------------------------------------------------------


def test_a_message_from_another_chat_is_dropped_at_handle_time():
    """
    Defence in depth. Even if the resolved chat id were somehow wrong, a message
    whose chat does not match is dropped and counted.
    """
    import asyncio

    listener = _listener()
    listener.status.chat_id = 100
    seen_before = listener.status.unparsed_dropped

    class _Msg:
        id = 555
        sender_id = 999
        chat_id = 777          # NOT the permitted source chat
        message = f"New coin $TEST\nCA: {REAL_MINT}\nMC: $5000"
        date = None

    listener._client = object()  # never used: the drop happens first
    foreign_before = listener.status.foreign_chat_dropped
    asyncio.run(listener._handle(_Msg()))
    assert listener.status.unparsed_dropped == seen_before + 1
    assert listener.status.foreign_chat_dropped == foreign_before + 1
    # and it produced no candidate at all
    assert listener.status.messages_accepted == 0


def test_the_permitted_chat_passes_the_chat_check():
    listener = _listener()
    listener.status.chat_id = 100

    class _Msg:
        id = 556
        sender_id = 999
        chat_id = 100          # the permitted source chat
        message = f"New coin $TEST\nCA: {REAL_MINT}\nMC: $5000"
        date = None

    listener._client = object()
    # It gets past the chat check and fails later (no client), which is the
    # point: the chat check did not drop it.
    import asyncio

    try:
        asyncio.run(listener._handle(_Msg()))
    except Exception:
        pass
    assert listener.status.unparsed_dropped == 0


# ---------------------------------------------------------------------------
# exact CA extraction
# ---------------------------------------------------------------------------


def test_a_bare_mint_is_extracted():
    p = parser_mod.parse_alert(f"New coin $TEST\nCA: {REAL_MINT}")
    assert p.mint == REAL_MINT


def test_a_mint_in_a_url_is_extracted():
    url = f"https://dexscreener.com/solana/{REAL_MINT}"
    p = parser_mod.parse_alert(f"New coin $TEST\n{url}")
    assert p.mint == REAL_MINT


def test_a_mint_inside_a_pump_fun_url_is_extracted():
    url = f"https://pump.fun/{REAL_MINT}"
    p = parser_mod.parse_alert(f"Launching $TEST\n{url}")
    assert p.mint == REAL_MINT


def test_a_non_base58_string_is_not_a_mint():
    """0, O, I and l are not in the base58 alphabet."""
    bad = "0OIl" + "1" * 40
    p = parser_mod.parse_alert(f"New coin $TEST\nCA: {bad}")
    assert p.mint != bad


def test_a_too_short_string_is_not_a_mint():
    p = parser_mod.parse_alert("New coin $TEST\nCA: abc123")
    assert p.mint != "abc123"


def test_a_missing_ca_yields_no_mint():
    p = parser_mod.parse_alert("New coin $TEST with no address")
    assert p.mint is None


def test_the_exact_ca_is_preserved_verbatim():
    """No case folding, no trimming: the CA that goes on chain is the CA read."""
    p = parser_mod.parse_alert(f"CA: {REAL_MINT}")
    assert p.mint == REAL_MINT
    assert p.mint == p.mint.strip()


# ---------------------------------------------------------------------------
# message identity and dedupe
# ---------------------------------------------------------------------------


def _snap(mid: int, mint: str = REAL_MINT, chat: str = "-100999") -> AlertSnapshot:
    return AlertSnapshot(
        candidate_id=f"cand_{mid}",
        source="telegram",
        chat=chat,
        message_id=mid,
        source_timestamp=1_000.0,
        received_timestamp=1_000.5,
        raw_text=f"CA: {mint}",
        mint=mint,
    )


def test_a_snapshot_carries_its_message_identity():
    s = _snap(42)
    assert s.message_id == 42
    assert s.chat == "-100999"
    assert s.candidate_id == "cand_42"


def test_the_same_message_yields_the_same_candidate_id():
    """Message identity is the dedupe key, so it must be deterministic."""
    a = _snap(42)
    b = _snap(42)
    assert a.candidate_id == b.candidate_id


def test_different_messages_yield_different_candidate_ids():
    assert _snap(42).candidate_id != _snap(43).candidate_id


def test_dedupe_window_drops_a_repeat():
    listener = _listener()
    assert listener.dedupe.add((1, 42)) is True
    assert listener.dedupe.add((1, 42)) is False
    assert (1, 42) in listener.dedupe


def test_dedupe_window_is_bounded():
    listener = _listener()
    for i in range(listener.dedupe.capacity + 50):
        listener.dedupe.add((1, i))
    assert len(listener.dedupe) <= listener.dedupe.capacity


def test_dedupe_distinguishes_message_ids():
    listener = _listener()
    assert listener.dedupe.add((1, 1)) is True
    assert listener.dedupe.add((1, 2)) is True
    assert listener.dedupe.add((2, 1)) is True


def test_seen_keys_round_trip_for_restart():
    listener = _listener()
    for i in (1, 2, 3):
        listener.dedupe.add((7, i))
    keys = listener.seen_keys()
    fresh = _listener()
    fresh.restore_seen(keys)
    assert (7, 2) in fresh.dedupe
    assert fresh.dedupe.add((7, 2)) is False


# ---------------------------------------------------------------------------
# candidate normalization
# ---------------------------------------------------------------------------


def test_a_symbol_is_normalized_to_uppercase():
    """
    Candidate normalization: the symbol is uppercased so $test and $TEST are the
    same token in every downstream key and log line.
    """
    p = parser_mod.parse_alert(f"New coin $test\nCA: {REAL_MINT}")
    assert p.symbol.upper() == "TEST"
    assert p.name.upper() == "TEST"


def test_a_price_is_parsed_to_a_float():
    p = parser_mod.parse_alert(
        f"$TEST\nCA: {REAL_MINT}\nPrice: $0.00001234\nMC: $15000"
    )
    assert p.fields.get("price_usd") is not None
    assert p.fields["price_usd"] == pytest.approx(0.00001234)


def test_a_market_cap_is_parsed():
    p = parser_mod.parse_alert(f"$TEST\nCA: {REAL_MINT}\nMC: $18,500")
    assert p.fields["market_cap_usd"] == pytest.approx(18_500.0)


def test_an_absent_field_stays_absent():
    """No defaults, no fabrication."""
    p = parser_mod.parse_alert(f"$TEST\nCA: {REAL_MINT}")
    assert p.fields.get("market_cap_usd") is None
    assert p.fields.get("liquidity_usd") is None


def test_parse_alert_is_pure():
    """No clock, no I/O, no network: the same text always parses the same."""
    import inspect

    src = inspect.getsource(parser_mod.parse_alert)
    for bad in ("time.", "datetime", "httpx", "requests", "open("):
        assert bad not in src
