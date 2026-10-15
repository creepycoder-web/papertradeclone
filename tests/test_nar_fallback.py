"""
Narrative provider fallback: Phenes (primary) -> tokenscan (fallback).

The fallback exists for ONE reason: the primary narrative bot stops answering.
It is NOT a content switch. A delivered-but-unfavourable narrative is a usable
answer, never an outage, and switching providers because the prose was negative
would be switching on content - which is explicitly forbidden.

The state machine is intentionally simple and self-healing:

    Phenes OK                     -> counter = 0, provider = Phenes
    failure #1, #2                -> still Phenes, counter climbs
    failure #3                    -> provider = tokenscan
    Phenes answers again          -> counter = 0, provider = Phenes

Every attempt queries the PRIMARY FIRST, which is the only way a recovery can
be observed, so the source cannot stay stranded on the fallback.

``NarSnapshot`` is frozen; these tests pin that attribution is applied by
rebuilding the snapshot rather than mutating it.
"""

from __future__ import annotations

import asyncio

import pytest

from autopaper.clock import FakeClock
from autopaper.config import NarConfig
from autopaper.models import NarSnapshot
from autopaper.nar.phanes import PhanesNarSource

pytestmark = pytest.mark.strategy

T0 = 1_700_000_000.0
MINT = "AwdfqHLnUTdAo5rNc7N6uxr5td8yBDKUe88FJcmLpump"

REAL_REPLY = (
    "Narrative for JEZ\n"
    "\n"
    "The JEZ token launched in 2024 with a viral meme campaign and a 2025 "
    "roadmap; 40% of supply is unlocked.\n"
    "\n"
    "— Accuracy not guaranteed."
)

NEGATIVE_REPLY = (
    "Narrative for JEZ\n"
    "\n"
    "The JEZ token launched in 2024 as a copy of an earlier project and has no "
    "working product; holders report it as abandoned.\n"
    "\n"
    "— Accuracy not guaranteed."
)


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _ok(mint=MINT, narrative="A 2023 launch with a roadmap.", **kw):
    return NarSnapshot(
        mint=mint, fetched_at=T0, source="phanes_bot", available=True,
        narrative=narrative, **kw,
    )


def _timeout(mint=MINT):
    return NarSnapshot(
        mint=mint, fetched_at=T0, source="phanes_bot", available=False,
        error="NAR_UNAVAILABLE: no bot reply within 2.5s (narrative budget exceeded)",
    )


def _placeholder(mint=MINT):
    return NarSnapshot(
        mint=mint, fetched_at=T0, source="phanes_bot", available=False,
        error="NAR_PLACEHOLDER: bot has no narrative for this token",
    )


# ---------------------------------------------------------------------------
# A scripted source: controls each provider's answer without a network client.
# ---------------------------------------------------------------------------


class ScriptedSource(PhanesNarSource):
    """``PhanesNarSource`` whose providers answer from a per-bot script."""

    def __init__(self, script, **kw):
        super().__init__(clock=FakeClock(T0), client=object(), **kw)
        # script: username -> a NarSnapshot, a callable, an Exception to raise,
        # or a list consumed left-to-right (the last entry repeats).
        self.script = script
        self.calls: list[tuple[str, str]] = []

    async def _query(self, bot_username, mint):
        self.calls.append((bot_username, mint))
        entry = self.script.get(bot_username, _timeout)
        if callable(entry):
            entry = entry()
        if isinstance(entry, list):
            entry = entry.pop(0) if len(entry) > 1 else entry[0]
        if isinstance(entry, Exception):
            raise entry
        return entry


def fetch(src, mint=MINT):
    return run_async(src.fetch_async(mint))


# ---------------------------------------------------------------------------
# 1-6. the state machine
# ---------------------------------------------------------------------------


def test_phenes_success_uses_phenes_and_never_touches_fallback():
    src = ScriptedSource({"Phanes_bot": _ok(), "tokenscan": _ok()})
    snap = fetch(src)
    assert snap.available is True
    assert snap.source == "phanes_bot", "attribution must name the primary"
    assert src.consecutive_failures == 0
    assert src.using_fallback is False
    assert [b for b, _ in src.calls] == ["Phanes_bot"]
    assert src.health()["narrative_source"] == "phanes_bot"


def test_first_failure_increments_counter_and_stays_on_phenes():
    src = ScriptedSource({"Phanes_bot": _timeout(), "tokenscan": _ok()})
    snap = fetch(src)
    assert snap.available is False
    assert src.consecutive_failures == 1
    assert src.using_fallback is False
    assert [b for b, _ in src.calls] == ["Phanes_bot"], "no fallback before 3"
    assert src.health()["consecutive_failures"] == 1


def test_second_failure_increments_counter_and_stays_on_phenes():
    src = ScriptedSource({"Phanes_bot": _timeout(), "tokenscan": _ok()})
    fetch(src)
    fetch(src)
    assert src.consecutive_failures == 2
    assert src.using_fallback is False
    assert [b for b, _ in src.calls] == ["Phanes_bot", "Phanes_bot"]


def test_third_failure_activates_tokenscan():
    src = ScriptedSource(
        {"Phanes_bot": _timeout(), "tokenscan": _ok(narrative="tokenscan data")}
    )
    fetch(src)
    fetch(src)
    snap = fetch(src)
    assert src.consecutive_failures == 3
    assert src.using_fallback is True
    assert snap.available is True
    assert snap.source == "tokenscan"
    assert snap.narrative == "tokenscan data"
    assert [b for b, _ in src.calls] == ["Phanes_bot", "Phanes_bot", "Phanes_bot",
                                         "tokenscan"]
    assert src.health()["narrative_source"] == "tokenscan"


def test_phenes_recovery_resets_counter_and_returns_to_primary():
    src = ScriptedSource({
        "Phanes_bot": [_timeout(), _timeout(), _timeout(), _ok()],
        "tokenscan": _ok(narrative="tokenscan data"),
    })
    fetch(src)
    fetch(src)
    third = fetch(src)
    assert third.source == "tokenscan" and src.using_fallback is True

    recovered = fetch(src)
    assert recovered.available is True
    assert recovered.source == "phanes_bot"
    assert src.consecutive_failures == 0
    assert src.using_fallback is False
    assert src.health()["narrative_source"] == "phanes_bot"


def test_fallback_does_not_stay_permanently_after_one_recovery():
    """After recovery a single fresh failure must not re-enter fallback."""
    src = ScriptedSource({
        "Phanes_bot": [_timeout(), _timeout(), _timeout(), _ok(), _timeout()],
        "tokenscan": _ok(narrative="tokenscan data"),
    })
    fetch(src)
    fetch(src)
    fetch(src)                       # -> fallback
    fetch(src)                       # -> recovered, counter 0
    snap = fetch(src)                # one more failure
    assert src.consecutive_failures == 1
    assert src.using_fallback is False
    assert snap.source == "phanes_bot"


# ---------------------------------------------------------------------------
# 5. the fallback speaks the SAME protocol: /nar <CA>
# ---------------------------------------------------------------------------


class FakeEntity:
    def __init__(self, username):
        self.username = username
        self.id = abs(hash(username)) % 100000 + 1


class FakeMessage:
    def __init__(self, id, text="", reply_to=None, out=False):
        self.id = id
        self.message = text
        self.text = text
        self.reply_to_msg_id = reply_to
        self.out = out


class FakeBotClient:
    """Tracks which bot received each command, and answers per-bot."""

    def __init__(self, replies):
        self.replies = replies
        self.sent = []  # (username, text)
        self._entities = {u: FakeEntity(u) for u in replies}
        self._user_of = {id(e): u for u, e in self._entities.items()}
        self._pending = {u: [] for u in replies}
        self._next_id = 1000

    async def get_entity(self, username):
        if username not in self._entities:
            raise ValueError(f"unknown bot: {username}")
        return self._entities[username]

    async def get_messages(self, bot, limit=1, min_id=None):
        u = self._user_of[id(bot)]
        if min_id:
            return [m for m in self._pending[u] if m.id > min_id]
        return [FakeMessage(1, "hi")]

    async def send_message(self, bot, text):
        u = self._user_of[id(bot)]
        self.sent.append((u, text))
        sent = FakeMessage(self._next_id, text, out=True)
        self._next_id += 1
        reply = self.replies.get(u)
        if reply is not None:
            answer = FakeMessage(self._next_id, reply, reply_to=sent.id, out=False)
            self._next_id += 1
            self._pending[u].append(answer)
        return sent


def test_tokenscan_is_queried_with_the_identical_nar_command():
    client = FakeBotClient({"Phanes_bot": None, "tokenscan": REAL_REPLY})
    src = PhanesNarSource(
        clock=FakeClock(T0), client=client, collect_extra_s=0.005,
        min_gap_s=0.0, timeout_s=0.05,
    )
    for _ in range(3):
        run_async(src.fetch_async(MINT))
    snap = run_async(src.fetch_async(MINT))
    assert snap.available is True
    assert snap.source == "tokenscan"
    # The 3rd fetch crosses the threshold and the 4th stays on the fallback:
    # the fallback is queried once per request from then on.
    tokenscan_sent = [t for u, t in client.sent if u == "tokenscan"]
    assert tokenscan_sent == [f"/nar {MINT}"] * 2
    # primary was always attempted first, with the same command
    phanes_sent = [t for u, t in client.sent if u == "Phanes_bot"]
    assert phanes_sent == [f"/nar {MINT}"] * 4


def test_timeout_against_the_real_client_counts_as_a_failure():
    client = FakeBotClient({"Phanes_bot": None, "tokenscan": REAL_REPLY})
    src = PhanesNarSource(
        clock=FakeClock(T0), client=client, collect_extra_s=0.005,
        min_gap_s=0.0, timeout_s=0.05,
    )
    snap = run_async(src.fetch_async(MINT))
    assert snap.available is False
    assert "no bot reply" in (snap.error or "")
    assert src._is_operational_failure(snap) is True
    assert src.consecutive_failures == 1


# ---------------------------------------------------------------------------
# 7. content is never a failure
# ---------------------------------------------------------------------------


def test_valid_negative_response_is_not_a_failure():
    src = ScriptedSource({"Phanes_bot": _ok(narrative="it is abandoned and dead")})
    snap = fetch(src)
    assert snap.available is True
    assert src._is_operational_failure(snap) is False
    assert src.consecutive_failures == 0
    assert src.using_fallback is False


def test_placeholder_response_is_not_an_operational_failure():
    """'Bot has no narrative' is a delivered answer, not an outage."""
    src = ScriptedSource({"Phanes_bot": _placeholder()})
    snap = fetch(src)
    assert snap.available is False
    assert src._is_operational_failure(snap) is False
    assert src.consecutive_failures == 0
    assert src.using_fallback is False


def test_negative_prose_parsed_through_the_parser_is_not_a_failure():
    from autopaper.nar.phanes import parse_nar_response

    snap = parse_nar_response(NEGATIVE_REPLY, MINT, T0)
    src = PhanesNarSource(clock=FakeClock(T0), client=object())
    assert snap.available is True
    assert src._is_operational_failure(snap) is False


# ---------------------------------------------------------------------------
# 8-9. real operational failures
# ---------------------------------------------------------------------------


def test_transport_error_counts_as_a_failure():
    src = ScriptedSource({
        "Phanes_bot": ConnectionResetError("connection reset by peer"),
        "tokenscan": _ok(),
    })
    snap = fetch(src)
    assert snap.available is False
    assert "NAR_UNAVAILABLE" in (snap.error or "")
    assert src.consecutive_failures == 1
    assert src.using_fallback is False


# ---------------------------------------------------------------------------
# 10, 13. both providers down -> honest unavailability, nothing fabricated
# ---------------------------------------------------------------------------


def test_both_providers_failing_keeps_the_safe_unavailable_result():
    src = ScriptedSource({"Phanes_bot": _timeout(), "tokenscan": _timeout()})
    fetch(src)
    fetch(src)
    snap = fetch(src)
    assert snap.available is False
    assert snap.narrative is None
    assert snap.narrative_score is None
    assert snap.holder_count is None
    assert snap.security_flags is None
    assert "NAR_UNAVAILABLE" in (snap.error or "")
    assert src.using_fallback is True
    assert src.health()["narrative_source"] == "tokenscan"


def test_fallback_never_fabricates_availability():
    """An unavailable fallback must not be turned into a pseudo-narrative."""
    src = ScriptedSource({"Phanes_bot": _timeout(), "tokenscan": _timeout()})
    for _ in range(3):
        snap = fetch(src)
        assert snap.available is False
        assert snap.narrative is None
    assert src.health()["narrative_source"] == "tokenscan"


# ---------------------------------------------------------------------------
# 11. attribution
# ---------------------------------------------------------------------------


def test_attribution_survives_into_the_snapshot_and_health():
    ok_src = ScriptedSource({"Phanes_bot": _ok()})
    assert fetch(ok_src).source == "phanes_bot"
    assert ok_src.health()["narrative_source"] == "phanes_bot"
    assert ok_src.health()["bot"] == "Phanes_bot"

    fb_src = ScriptedSource(
        {"Phanes_bot": _timeout(), "tokenscan": _ok()}
    )
    for _ in range(3):
        snap = fetch(fb_src)
    assert snap.source == "tokenscan"
    assert fb_src.health()["narrative_source"] == "tokenscan"
    assert fb_src.health()["fallback_bot"] == "tokenscan"


def test_attribution_is_a_rebuild_not_a_mutation():
    """``NarSnapshot`` is frozen; attribution must not raise FrozenInstanceError."""
    src = ScriptedSource({"Phanes_bot": _ok()})
    snap = fetch(src)
    assert snap.source == "phanes_bot"
    with pytest.raises(Exception):
        snap.source = "other"  # noqa: B010 - the point IS that this is frozen


# ---------------------------------------------------------------------------
# 12. no retry storm
# ---------------------------------------------------------------------------


def test_no_retry_storm_one_send_per_attempt():
    client = FakeBotClient({"Phanes_bot": None, "tokenscan": None})
    src = PhanesNarSource(
        clock=FakeClock(T0), client=client, collect_extra_s=0.005,
        min_gap_s=0.0, timeout_s=0.05,
    )
    # Two failing calls: strictly two sends, no hidden retries.
    run_async(src.fetch_async(MINT))
    run_async(src.fetch_async(MINT))
    assert len(client.sent) == 2
    assert all(u == "Phanes_bot" for u, _ in client.sent)


def test_switching_to_fallback_does_not_retry_the_primary_twice():
    src = ScriptedSource({"Phanes_bot": _timeout(), "tokenscan": _ok()})
    for _ in range(3):
        fetch(src)
    # third call: primary once, fallback once. Never primary twice per call.
    assert [b for b, _ in src.calls] == [
        "Phanes_bot", "Phanes_bot", "Phanes_bot", "tokenscan"
    ]


# ---------------------------------------------------------------------------
# config + 14. existing narrative / L25 semantics unchanged
# ---------------------------------------------------------------------------


def test_config_defaults_are_the_documented_ones():
    cfg = NarConfig()
    assert cfg.fallback_bot_username == "tokenscan"
    assert cfg.fallback_after_failures == 3
    assert cfg.bot_username == "Phanes_bot"
    assert cfg.bot_command == "/nar"


def test_fallback_can_be_disabled_entirely():
    src = ScriptedSource({"Phanes_bot": _timeout()}, fallback_bot_username=None)
    for _ in range(5):
        snap = fetch(src)
    assert src.using_fallback is False
    assert src.fallback_bot_username is None
    assert [b for b, _ in src.calls] == ["Phanes_bot"] * 5
    assert snap.available is False


def test_existing_narrative_parsing_semantics_unchanged():
    from autopaper.nar.phanes import parse_nar_response

    snap = parse_nar_response(REAL_REPLY, MINT, T0)
    assert snap.available is True
    assert "launched in 2024" in snap.narrative
    assert "Accuracy not guaranteed" not in snap.narrative
    for field in ("holder_count", "top10_holders_pct", "bundler_count",
                  "sniper_count", "insider_count", "narrative_score"):
        assert getattr(snap, field) is None, field


def test_l25_and_v3c_thresholds_are_untouched():
    from autopaper.config import DecisionConfig
    from autopaper.l25 import L25_MIN_LIQ_TO_MCAP
    from autopaper.v3c import V3C_MAX_MARKET_CAP_USD, V3C_MIN_SIGNAL_QUALITY

    dc = DecisionConfig()
    assert dc.l25_min_liq_to_mcap == L25_MIN_LIQ_TO_MCAP
    assert dc.v3c_max_market_cap_usd == V3C_MAX_MARKET_CAP_USD == 20_000.0
    assert dc.v3c_min_signal_quality == V3C_MIN_SIGNAL_QUALITY == 0.55
