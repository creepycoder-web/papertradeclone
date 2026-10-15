"""
Signal parsing and token extraction (test items 1 and 2).

Exercised against the REAL captured messages from ``@MooDengPresidentCallers``
in ``tests/fixtures/real_alerts.json`` rather than invented ones, because the
formats that break a parser are the ones the channel actually sends: a
narrative post with no contract address, a mint that ends in ``pump``, a
``DEX:`` line whose value is a subscription marker rather than a venue.
"""


from __future__ import annotations

import json
from pathlib import Path

import pytest

from autopaper.same_market.signals import (
    extract_token,
    launchpad_for_mint,
    looks_like_mint,
    multiple_mints,
)

pytestmark = pytest.mark.same_market

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "real_alerts.json"


@pytest.fixture(scope="module")
def real_alerts() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Telegram signal parsing
# ---------------------------------------------------------------------------

def test_every_captured_message_is_classified_without_raising(real_alerts):
    for alert in real_alerts:
        result = extract_token(alert["text"])
        assert result.mint is None or looks_like_mint(result.mint)
        assert isinstance(result.ambiguous, bool)


def test_alerts_with_a_contract_address_yield_exactly_that_mint(real_alerts):
    with_mint = [a for a in real_alerts if multiple_mints(a["text"])]
    assert with_mint, "the fixture should contain messages with mints"
    for alert in with_mint:
        result = extract_token(alert["text"])
        assert not result.ambiguous
        assert result.mint in multiple_mints(alert["text"])


def test_narrative_posts_are_reported_as_having_no_mint(real_alerts):
    """A ticker and figures but no contract address is not a token signal."""
    narrative = [
        a for a in real_alerts
        if not multiple_mints(a["text"]) and "$" in a["text"]
    ]
    assert narrative, "the fixture should contain narrative posts"
    for alert in narrative:
        result = extract_token(alert["text"])
        assert result.mint is None
        assert not result.ambiguous
        assert any("no base58 mint" in w for w in result.warnings)


def test_empty_and_blank_messages_are_handled():
    for text in ("", "   ", "\n\n"):
        result = extract_token(text)
        assert result.mint is None
        assert result.candidates == ()
        assert result.warnings


def test_the_channel_promo_line_is_not_mistaken_for_a_venue(real_alerts):
    """
    The channel emits "DEX:  Paid❌ Ads❌ 0⚡".

    "Paid" is the channel's subscription tier, not a venue. Reading it as a
    venue would produce a confident wrong hint, and because an unmatched hint
    is reported as VENUE_MISMATCH it would then fail the identity check for
    entirely the wrong reason.
    """
    paid_posts = [a for a in real_alerts if "Paid" in a["text"] and "DEX" in a["text"]]
    assert paid_posts, "the fixture should contain the promo format"
    for alert in paid_posts:
        assert extract_token(alert["text"]).venue_hint is None


def test_a_genuine_venue_hint_is_still_read():
    text = (
        "🔥 $TOKEN\n🎯 CA: H9XQrbw5MANzYVgHjXJbScFxWqyGw9CCAec5XDTgpump\n"
        "DEX: Raydium\nMC: $18k"
    )
    assert extract_token(text).venue_hint == "Raydium"


def test_fullwidth_punctuation_from_the_channel_is_handled(real_alerts):
    """The channel mixes fullwidth colons; the shared normaliser folds them."""
    for alert in real_alerts:
        result = extract_token(alert["text"])
        assert result is not None


# ---------------------------------------------------------------------------
# 2. token extraction, including the ambiguity rule
# ---------------------------------------------------------------------------

def test_two_distinct_mints_are_ambiguous_and_neither_is_chosen():
    text = (
        "buy A: H9XQrbw5MANzYVgHjXJbScFxWqyGw9CCAec5XDTgpump\n"
        "or B: 6uPEdkU2xPg1iyfLzV1ssUwvve8nndA2sd6nBFrb1n58"
    )
    result = extract_token(text)
    assert result.ambiguous is True
    assert result.mint is None, "an ambiguous signal must not acquire a mint"
    assert len(result.candidates) == 2
    assert any("refusing to guess" in w for w in result.warnings)


def test_one_mint_repeated_many_times_is_not_ambiguous():
    mint = "H9XQrbw5MANzYVgHjXJbScFxWqyGw9CCAec5XDTgpump"
    text = f"CA: {mint}\nMC $18k\nagain {mint}\nand {mint}"
    result = extract_token(text)
    assert result.ambiguous is False
    assert result.mint == mint
    # Labelled, so the label is the reason it is unambiguous.
    assert any("explicit 'CA' field" in w for w in result.warnings)


def test_one_mint_repeated_many_times_without_a_label_is_not_ambiguous():
    """The unlabelled twin: repetition alone is still one token."""
    mint = "H9XQrbw5MANzYVgHjXJbScFxWqyGw9CCAec5XDTgpump"
    text = f"MC $18k\n{mint}\nand {mint}\nand {mint}"
    result = extract_token(text)
    assert result.ambiguous is False
    assert result.mint == mint
    assert any("not ambiguous" in w for w in result.warnings)


def test_a_three_mint_message_reports_all_three():
    mints = (
        "H9XQrbw5MANzYVgHjXJbScFxWqyGw9CCAec5XDTgpump "
        "6uPEdkU2xPg1iyfLzV1ssUwvve8nndA2sd6nBFrb1n58 "
        "8HnLRKJ6XkeGq1tpmKPqn3yB5ruqbSR5LR5nVCNAshop"
    )
    result = extract_token(f"three tokens {mints}")
    assert result.ambiguous is True
    assert len(result.candidates) == 3


@pytest.mark.parametrize("value", ["", "tooshort", "x" * 45, "!!!", "0" * 32])
def test_non_mint_strings_are_not_mints(value):
    assert looks_like_mint(value) is False


def test_mint_shape_accepts_real_solana_addresses(real_alerts):
    mints = [m for a in real_alerts for m in multiple_mints(a["text"])]
    assert mints
    assert all(32 <= len(m) <= 44 for m in mints)


def test_launchpad_suffix_is_a_hint_only():
    assert launchpad_for_mint("H9XQrbw5MANzYVgHjXJbScFxWqyGw9CCAec5XDTgpump") == "pumpfun"
    assert launchpad_for_mint("6uPEdkU2xPg1iyfLzV1ssUwvve8nndA2sd6nBFrb1n58") is None
    # A mint that is nothing but a suffix is not evidence of a launchpad.
    assert launchpad_for_mint("pump") is None
    assert launchpad_for_mint(None) is None