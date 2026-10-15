"""
Focused tests for the shared safety/ratelimit helper contracts (the audit).

WHY THIS FILE EXISTS
====================
``src/autopaper/safety.py`` and ``src/autopaper/market/ratelimit.py`` are the two
shared files every other module reaches into for redaction, masking and
rate-limit classification. They are also the two files that were, at one point,
rewritten down to a handful of stub helpers whose only job was to make imports
resolve. The stubs LOOKED right and had the right names, but several of them had
the wrong semantics, so this file pins the behaviour rather than the presence of
a symbol.

Behaviour that the stubs got wrong, and that is asserted below:

* ``REDACTED`` was ``"[redacted]"``; the real marker is ``"***REDACTED***"``.
* ``mask_telegram_id`` kept 2 leading characters and silently shortened an
  already-short id; the real mask keeps 3 and returns a short id unchanged.
* ``redact_mapping`` was flat; the real one recurses into nested mappings.
* ``find_secrets_in_text`` used invented patterns (bare 64-hex-char strings, a
  ``<id>:<hash>`` Telegram shape); the real patterns are a mnemonic run, a
  base-58-style integer array private key, a ``Bearer`` token, a ``Telethon
  session`` marker and a ``private_key:``/``seed_phrase:`` block.

Real callers depend on these: ``telegram/ui.py`` and ``real_account.py`` import
``mask_address``/``mask_telegram_id``, ``journal.py`` imports ``redact_mapping``
on every write, ``paper_live/guards.py`` and ``same_market/guard.py`` subclass
``SafetyViolation``, and ``same_market/rpc.py`` plus ``paper_live/engine.py``
import the five lane/reason constants.
"""

from __future__ import annotations

import pytest

from autopaper.safety import (
    REDACTED,
    SafetyViolation,
    find_secrets_in_text,
    is_secret_key,
    mask_address,
    mask_telegram_id,
    redact_mapping,
)

MINT = "So11111111111111111111111111111111111111112"


# ---------------------------------------------------------------------------
# address masking
# ---------------------------------------------------------------------------

def test_address_masking_hides_the_middle_and_keeps_the_ends():
    masked = mask_address(MINT)
    assert masked == "So11…1112"
    assert masked.startswith("So11")
    assert masked.endswith("1112")
    assert len(masked) < 20


def test_address_masking_never_returns_the_full_address():
    masked = mask_address(MINT)
    assert MINT not in masked
    assert all(len(part) <= 4 for part in masked.split("…"))


def test_address_masking_is_idempotent():
    """
    Renderers mask defensively even when the value is already masked.

    Without idempotence a second pass shortens ``So11…1112`` to ``So…``, so the
    operator sees a different (and useless) mask depending on how many layers
    happened to run.
    """
    once = mask_address(MINT)
    assert mask_address(once) == once


def test_address_masking_does_not_shorten_a_short_value_into_nothing():
    assert mask_address("") == "(none)"
    assert mask_address("ab") == "ab…"


def test_address_masking_agrees_with_the_execution_layer_copy():
    """
    ``execution/solana.py`` carries its own ``mask_address``.

    Two implementations of one mask is a drift risk, so the operator surfaces
    (``telegram/ui.py`` via ``safety``, ``execution/real.py`` via ``solana``)
    must not disagree about what a masked address looks like.
    """
    from autopaper.execution.solana import mask_address as execution_mask

    for value in (MINT, "", "ab", "abcdefghijklm", "x" * 44):
        assert mask_address(value) == execution_mask(value), value


# ---------------------------------------------------------------------------
# telegram identifier masking
# ---------------------------------------------------------------------------

def test_telegram_id_masking_hides_the_identifier():
    masked = mask_telegram_id("-1001234567890")
    assert "-1001234567890" not in masked
    assert "…" in masked
    assert len(masked) < len("-1001234567890")


def test_telegram_id_masking_keeps_enough_to_recognise_the_chat():
    """
    The operator must be able to confirm WHICH chat is configured from the
    status line, so the leading characters survive -- enough to recognise, not
    enough to impersonate.
    """
    assert mask_telegram_id("-1001234567890") == "-10…90"
    assert mask_telegram_id("-1001234567890").startswith("-10")


def test_telegram_id_masking_of_a_short_id_leaves_it_alone():
    """A 2-character id has nothing to hide, and truncating it would be a lie."""
    assert mask_telegram_id("99") == "99"


def test_telegram_id_masking_of_an_absent_id_says_so():
    assert mask_telegram_id("") == "(not configured)"


# ---------------------------------------------------------------------------
# secret detection in free text
# ---------------------------------------------------------------------------

#: The complete vocabulary ``find_secrets_in_text`` may return. A new name here
#: is a new secret class, which is a deliberate decision, not a typo.
SECRET_NAMES = frozenset({
    "mnemonic",
    "solana_private_key",
    "bearer",
    "telegram_session",
    "private_key_block",
})


def test_a_seed_phrase_is_detected_without_being_returned():
    hits = find_secrets_in_text("recovery: " + " ".join(["alpha"] * 12))
    assert "mnemonic" in hits
    assert set(hits) <= SECRET_NAMES


def test_a_numeric_private_key_array_is_detected():
    """The shape a JSON-serialised keypair is written in."""
    hits = find_secrets_in_text("[" + ",".join(["7"] * 64) + "]")
    assert "solana_private_key" in hits


def test_a_bearer_token_is_detected():
    assert "bearer" in find_secrets_in_text(
        "Authorization: Bearer " + "A" * 30)


def test_a_telethon_session_marker_is_detected():
    assert "telegram_session" in find_secrets_in_text(
        "Telethon session file .session present")


def test_a_labelled_key_block_is_detected():
    assert "private_key_block" in find_secrets_in_text("private_key = abc")
    assert "private_key_block" in find_secrets_in_text("seed_phrase: words")


def test_detected_names_never_contain_the_secret_itself():
    """
    The scan exists to REPORT hardcoded credentials without publishing them, so
    the return value must never carry the value it matched.
    """
    secret = " ".join(["alpha"] * 12)
    hits = find_secrets_in_text(f"recovery: {secret}")
    assert hits, "a 12-word mnemonic must be detected"
    assert all(secret not in name for name in hits)
    assert all(name in SECRET_NAMES for name in hits)


def test_ordinary_telegram_signal_text_is_not_flagged():
    """
    A false positive here is not harmless: this runs against message text and
    template files, and one spurious hit would refuse a legitimate signal.
    """
    text = (
        "New call: $WIF mint " + MINT + "\n"
        "MC 42k, liq 60k, 5m vol 1.2k, 3 green candles, buy score 7.4\n"
        "Token age 6m. Do your own research."
    )
    assert find_secrets_in_text(text) == []


def test_an_empty_or_absent_text_is_not_scanned():
    assert find_secrets_in_text("") == []


# ---------------------------------------------------------------------------
# mapping redaction
# ---------------------------------------------------------------------------

def test_the_redaction_marker_is_the_shared_constant():
    assert REDACTED == "***REDACTED***"


def test_redaction_replaces_every_credential_shaped_key():
    out = redact_mapping({
        "api_hash": "0123456789abcdef0123456789abcdef",
        "bot_token": "1234:AAaaBBbbCCcc",
        "seed_phrase": "one two three four five six seven",
        "private_key": "x" * 64,
        "password": "hunter2",
    })
    assert set(out.values()) == {REDACTED}


def test_redaction_leaves_a_public_address_and_plain_config_alone():
    """
    Over-redaction is not "safe": a journal whose record reads
    ``***REDACTED***`` where a decision was is worse than no journal.
    """
    out = redact_mapping({
        "public_address": MINT,
        "mint": MINT,
        "STRATEGY_PROFILE": "L25_60S",
        "mint_count": 3,
    })
    assert out["public_address"] == MINT
    assert out["mint"] == MINT
    assert out["STRATEGY_PROFILE"] == "L25_60S"
    assert out["mint_count"] == 3


def test_redaction_recurses_into_nested_mappings():
    out = redact_mapping({"outer": {"api_hash": "0123456789abcdef"}, "mint": "M"})
    assert out["outer"]["api_hash"] == REDACTED
    assert out["mint"] == "M"


def test_redaction_does_not_mutate_the_caller_mapping():
    original = {"api_hash": "secret-value", "mint": MINT}
    out = redact_mapping(original)
    assert original["api_hash"] == "secret-value", "the caller's dict was mutated"
    assert out["api_hash"] == REDACTED


def test_an_absent_value_stays_absent_rather_than_becoming_a_marker():
    """``None`` means 'not supplied'; redacting it would invent a value."""
    assert redact_mapping({"api_hash": None})["api_hash"] is None


def test_is_secret_key_classifies_names_not_substrings_of_them():
    assert is_secret_key("STRATEGY_PROFILE") is False
    assert is_secret_key("mint") is False
    assert is_secret_key("authorized") is False
    assert is_secret_key("api_hash") is True
    assert is_secret_key("bot_token") is True
    assert is_secret_key("private_key") is True


def test_the_journal_redacts_on_write():
    """The real consumer: a persisted event must not carry the credential."""
    import tempfile
    from pathlib import Path

    from autopaper.journal import Journal
    from autopaper.modes import TradeMode

    with tempfile.TemporaryDirectory() as tmp:
        journal = Journal(Path(tmp))
        journal.event("audit", TradeMode.PAPER, api_hash="deadbeef", mint="M")
        row = journal.all_events()[-1]
    assert row["api_hash"] == REDACTED
    assert row["mint"] == "M"


# ---------------------------------------------------------------------------
# exception vocabulary
# ---------------------------------------------------------------------------

def test_safety_violation_is_a_runtime_error():
    assert issubclass(SafetyViolation, RuntimeError)


def test_the_guards_own_violations_are_safety_violations():
    """
    Both guard modules subclass ``SafetyViolation`` so one ``except`` in the
    caller cannot miss a violation raised by the other.
    """
    from autopaper.paper_live.guards import PaperCapabilityViolation
    from autopaper.same_market.guard import NoBroadcastViolation

    assert issubclass(PaperCapabilityViolation, SafetyViolation)
    assert issubclass(NoBroadcastViolation, SafetyViolation)


# ---------------------------------------------------------------------------
# request lanes and limiter budget classification
# ---------------------------------------------------------------------------

def test_the_lane_and_reason_constants_are_distinct():
    """
    These are compared for equality all over the RPC client and the limiter. Two
    that shared a value would make a telemetry refusal indistinguishable from an
    execution one, which is the exact confusion the constants exist to prevent.
    """
    from autopaper.market.ratelimit import (
        LANE_EXECUTION,
        LANE_TELEMETRY,
        LOCAL_WINDOW,
        TELEMETRY_BUDGET,
        UPSTREAM_BLOCKED,
    )

    values = [LANE_EXECUTION, LANE_TELEMETRY, LOCAL_WINDOW,
              TELEMETRY_BUDGET, UPSTREAM_BLOCKED]
    assert len(set(values)) == len(values)


def test_the_telemetry_lane_is_not_the_execution_lane():
    from autopaper.market.ratelimit import LANE_EXECUTION, LANE_TELEMETRY
    from autopaper.same_market.rpc import current_rpc_lane, rpc_lane

    assert current_rpc_lane() == LANE_EXECUTION
    with rpc_lane(LANE_TELEMETRY):
        assert current_rpc_lane() == LANE_TELEMETRY
    assert current_rpc_lane() == LANE_EXECUTION, "the lane leaked out of its block"


def test_a_refusal_defaults_to_the_local_window_and_classifies_as_local():
    from autopaper.market.ratelimit import LOCAL_WINDOW
    from autopaper.same_market.rpc import RpcRateLimited

    exc = RpcRateLimited("local cap")
    assert exc.reason == LOCAL_WINDOW
    assert exc.is_local_quota is True
    assert exc.is_upstream is False


def test_an_upstream_block_classifies_as_upstream_not_local():
    from autopaper.market.ratelimit import UPSTREAM_BLOCKED
    from autopaper.same_market.rpc import RpcRateLimited

    exc = RpcRateLimited("node said 429", reason=UPSTREAM_BLOCKED, retry_after_s=2.0)
    assert exc.is_upstream is True
    assert exc.is_local_quota is False
    assert exc.retry_after_s == 2.0


def test_an_upstream_throttle_is_reported_as_upstream_by_the_limiter():
    """
    The limiter must name WHICH condition refused.

    An upstream provider throttle and a local quota were once collapsed into one
    "local rate cap reached" message, which points the operator at the wrong
    system and therefore invites the wrong fix.
    """
    from autopaper.market.ratelimit import (
        LOCAL_WINDOW,
        UPSTREAM_BLOCKED,
        RateLimiter,
    )

    limiter = RateLimiter(60, name="test")
    limiter.note_retry_after(100.0, 5.0, provider="dexscreener")
    allowed, wait, reason = limiter.check(101.0)
    assert allowed is False
    assert reason == UPSTREAM_BLOCKED
    assert reason != LOCAL_WINDOW
    assert 0 < wait <= 5.0
    assert limiter.blocked_by_upstream is True


def test_a_spent_telemetry_budget_is_not_reported_as_a_market_failure():
    """
    The classification the hold-path sampler depends on.

    A spent telemetry lane means "we chose not to ask"; a market failure means
    "we asked and could not read it". Pooling them inflated QUOTE_UNAVAILABLE
    coverage numbers, so the two must stay distinguishable by the reason
    constant and never by the message text.
    """
    from autopaper.market.ratelimit import TELEMETRY_BUDGET, UPSTREAM_BLOCKED
    from autopaper.paper_live.engine import _is_telemetry_skip
    from autopaper.same_market.rpc import RpcRateLimited

    skip = RpcRateLimited("telemetry sub-budget spent", reason=TELEMETRY_BUDGET)
    market = RpcRateLimited("node refused", reason=UPSTREAM_BLOCKED)
    assert _is_telemetry_skip(skip) is True
    assert _is_telemetry_skip(market) is False
    assert _is_telemetry_skip(RuntimeError("unrelated")) is False


def test_the_reader_re_raises_a_spent_budget_but_passes_a_real_failure():
    """
    ``same_market/reader.py`` re-raises only the telemetry-skip case.

    Asserted through the module's own predicate rather than by calling the
    network path: the point is the classification, not the read.
    """
    from autopaper.market.ratelimit import TELEMETRY_BUDGET, UPSTREAM_BLOCKED
    from autopaper.same_market.reader import _reraise_if_telemetry_skip
    from autopaper.same_market.rpc import RpcRateLimited

    with pytest.raises(RpcRateLimited):
        _reraise_if_telemetry_skip(
            RpcRateLimited("spent", reason=TELEMETRY_BUDGET))

    # A non-telemetry refusal passes THROUGH the predicate so the caller can
    # wrap it as an ordinary read failure.
    _reraise_if_telemetry_skip(
        RpcRateLimited("node refused", reason=UPSTREAM_BLOCKED))


# ---------------------------------------------------------------------------
# the limiter itself
# ---------------------------------------------------------------------------

def test_the_request_limiter_enforces_its_configured_window():
    from autopaper.market.ratelimit import MarketDataError, RateLimiter

    limiter = RateLimiter(2, name="test")
    limiter.acquire(100.0)
    limiter.acquire(101.0)
    with pytest.raises(MarketDataError) as exc:
        limiter.acquire(102.0)
    assert exc.value.rate_limited is True
    assert exc.value.retry_after_s and exc.value.retry_after_s > 0
    assert limiter.rejections == 1


def test_the_request_limiter_forgets_calls_outside_its_window():
    from autopaper.market.ratelimit import RateLimiter

    limiter = RateLimiter(1, name="test")
    limiter.acquire(100.0)
    # 61s later the earlier call has aged out of the 60s window.
    limiter.acquire(161.0)
    assert limiter.rejections == 0
    assert limiter.total_requests == 2


def test_a_retry_after_is_obeyed_rather_than_guessed():
    from autopaper.market.ratelimit import RateLimiter

    limiter = RateLimiter(60, name="test")
    limiter.note_retry_after(100.0, 5.0)
    allowed, wait = limiter.allow(101.0)
    assert allowed is False
    assert 0 < wait <= 5.0


def test_the_two_lanes_partition_the_budget_without_raising_the_total():
    """
    The reservation is a PARTITION, not an increase.

    A telemetry floor is cut out of the execution ceiling so the two sum to the
    configured total exactly: adding a background sampler must not multiply the
    provider request rate.
    """
    from autopaper.market.ratelimit import RateLimiter

    limiter = RateLimiter(10, name="test", telemetry_max_per_minute=3)
    assert limiter.telemetry_ceiling() == 3
    assert limiter.execution_ceiling() == 7
    assert (limiter.telemetry_ceiling() + limiter.execution_ceiling()
            == limiter.max_per_minute)


def test_execution_traffic_cannot_refuse_a_monitoring_request():
    """
    The invariant that a blocked SELL must not disable OPEN POSITION monitoring.

    A telemetry acquire is judged only against its own sub-budget, so exit and
    quote retries filling the execution window cannot refuse the sampler.
    """
    from autopaper.market.ratelimit import LANE_EXECUTION, LANE_TELEMETRY, RateLimiter

    limiter = RateLimiter(3, name="test", telemetry_max_per_minute=1)
    for i in range(limiter.execution_ceiling()):
        limiter.acquire(100.0 + i, LANE_EXECUTION)

    spent, _wait, reason = limiter.check(100.0, LANE_EXECUTION)
    assert spent is False, "the execution lane should be at its ceiling"

    allowed, _wait, _reason = limiter.check(100.0, LANE_TELEMETRY)
    assert allowed is True, (
        "a full execution window must not refuse a telemetry request")


def test_a_spent_telemetry_lane_names_its_own_reason():
    from autopaper.market.ratelimit import (
        LANE_TELEMETRY,
        TELEMETRY_BUDGET,
        RateLimiter,
    )

    limiter = RateLimiter(5, name="test", telemetry_max_per_minute=1)
    limiter.acquire(100.0, LANE_TELEMETRY)
    allowed, _wait, reason = limiter.check(100.1, LANE_TELEMETRY)
    assert allowed is False
    assert reason == TELEMETRY_BUDGET
    assert limiter.rejections_by_reason[TELEMETRY_BUDGET] == 0, (
        "check() must not count a rejection; only acquire() records one")


def test_the_template_env_file_carries_no_secret():
    """
    The check ``tests/test_config.py`` runs: the shipped example must not itself
    look like key material.
    """
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / ".env.example"
    if not example.exists():
        pytest.skip(".env.example is not present in this checkout")
    assert find_secrets_in_text(example.read_text(encoding="utf-8")) == []
