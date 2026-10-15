"""
CANONICAL PARSER FIXTURE: the ``$AI`` signal from @MooDengPresidentCallers.

Why this message is the fixture
-------------------------------

It carries the three things that break a signal parser, in one capture:

1. an explicit ``CA:`` field, which is authoritative for the token;
2. a **two-character ticker** (``$AI``) that must never be inferred as a mint;
3. a promo footer (``AVE ｜ GMGN … #AI``) dense with decoy text.

The expected mint is fixed in ``fixtures/same_market_ai_signal.json``. Anything
inferred from the ticker is a failure, not a near-miss.

THE RULE THESE TESTS PIN DOWN
=============================

A labelled contract-address field is authoritative. A second address in the
footer does NOT make the signal ambiguous.

That rule was added because of a real defect found while building this fixture:
before it, a correctly-formatted signal whose footer mentioned a second token
was declared ``AMBIGUOUS`` and **silently dropped**. On a busy caller channel
that loses real signals. The mirror-image risk - letting a footer override the
label - is what this module exists to prevent, so both directions are asserted.
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

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "same_market_ai_signal.json"

CANONICAL_MINT = "A5MdMEt99PzbYKsfPaQ5Ee1jFZEsYu3zBvdpDdyX2sXQ"
OTHER_MINT = "6uPEdkU2xPg1iyfLzV1ssUwvve8nndA2sd6nBFrb1n58"


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def canonical(spec) -> str:
    return spec["messages"][0]["text"]


# ---------------------------------------------------------------------------
# the canonical message
# ---------------------------------------------------------------------------

def test_the_canonical_fixture_pins_the_expected_mint(spec):
    assert spec["expected_mint"] == CANONICAL_MINT
    assert spec["must_not_infer_mint_from"] == "$AI"


def test_the_ca_field_yields_exactly_the_expected_mint(canonical):
    result = extract_token(canonical)
    assert result.mint == CANONICAL_MINT
    assert result.ambiguous is False


def test_the_ticker_is_recorded_as_a_symbol_not_as_a_mint(canonical):
    """
    ``$AI`` is two characters. A mint is 32..44 base58 characters. Inferring the
    token from the ticker would be inferring it from the ticker.
    """
    result = extract_token(canonical)
    assert result.symbol == "AI"
    assert result.mint != "AI"
    assert result.mint not in result.symbol


def test_the_mint_is_the_only_address_shaped_string_in_the_message(canonical):
    found = multiple_mints(canonical)
    assert found == (CANONICAL_MINT,), (
        f"the canonical message should yield exactly one address, got {found}"
    )


def test_the_canonical_message_is_not_flagged_ambiguous(canonical, spec):
    assert extract_token(canonical).ambiguous is spec["expected_ambiguous"] is False


def test_the_promo_dex_line_is_not_read_as_a_venue(canonical, spec):
    """``DEX:  Paid❌ Ads❌ 0⚡`` is a subscription marker, not a venue."""
    assert extract_token(canonical).venue_hint == spec["expected_venue_hint"] is None


def test_the_venue_must_come_from_chain_state_not_the_mint_suffix(canonical):
    """
    This mint ends in ``X2sXQ``, so no launchpad can be inferred from it. That
    is a legitimate UNKNOWN and is reported as such rather than guessed.
    """
    assert launchpad_for_mint(CANONICAL_MINT) is None
    result = extract_token(canonical)
    assert any("venue must be established from chain state" in w
               for w in result.warnings)


def test_the_canonical_message_reaches_the_pipeline_as_a_single_token(canonical):
    """End of the parser's responsibility: one unambiguous mint, ready to price."""
    result = extract_token(canonical)
    assert result.ambiguous is False
    assert result.mint == CANONICAL_MINT
    assert looks_like_mint(result.mint)
    assert len(result.candidates) == 1


# ---------------------------------------------------------------------------
# the CA: field is authoritative
# ---------------------------------------------------------------------------

def test_a_second_address_in_the_footer_does_not_create_ambiguity(spec):
    case = spec["messages"][1]
    result = extract_token(case["text"])
    assert result.ambiguous is False
    assert result.mint == case["expected_mint"] == CANONICAL_MINT
    assert any("explicit 'CA' field" in w for w in result.warnings)
    assert any("outranks" in w for w in result.warnings)


def test_the_ignored_footer_address_is_still_reported_not_silently_dropped(spec):
    """A reader must be able to see what was set aside and why."""
    result = extract_token(spec["messages"][1]["text"])
    assert len(result.candidates) == 2, "both addresses remain visible"
    assert OTHER_MINT in result.candidates


@pytest.mark.parametrize("label", [
    "CA", "CA:", "Mint", "mint", "Contract", "contract address", "Token Address",
])
def test_every_contract_address_alias_is_authoritative(label):
    text = f"{label}: {CANONICAL_MINT}\nsee also {OTHER_MINT}"
    result = extract_token(text)
    assert result.ambiguous is False
    assert result.mint == CANONICAL_MINT


def test_without_a_label_two_addresses_stay_ambiguous(spec):
    """
    The CA: rule is about an explicit field, not a licence to guess.
    """
    case = spec["messages"][2]
    result = extract_token(case["text"])
    assert result.ambiguous is True
    assert result.mint is None
    assert result.ambiguous is case["expected_ambiguous"]
    assert any("names no contract-address field" in w for w in result.warnings)


def test_a_labelled_field_naming_two_mints_disambiguates_nothing(spec):
    """
    ``CA: X or Y`` names two tokens, so it cannot be resolved to either.

    The shared key/value extractor does not treat ``X or Y`` as a single
    labelled value, so this resolves through the no-label path instead - and
    the fallback refuses it for the same reason. What matters is that neither
    path picks one, so both are asserted.
    """
    case = spec["messages"][3]
    result = extract_token(case["text"])
    assert result.ambiguous is True
    assert result.mint is None
    assert any("refusing to guess" in w or "does not disambiguate" in w
               for w in result.warnings)


def test_a_single_label_never_resolves_to_just_its_first_mint():
    """
    The production parser's ``_first_mint`` would take X from ``CA: X or Y``.

    The invariant this harness must hold is not "return both" but "never return
    exactly one". Either the label is recognised and yields both mints, or the
    shared extractor does not recognise it as a labelled value at all and
    yields none - and the no-label path then refuses. Collapsing it to a single
    mint is the one outcome that would be a guess.
    """
    from autopaper.same_market.signals import _labelled_mints

    mints = [m for _label, group in _labelled_mints(
        f"CA: {CANONICAL_MINT} or {OTHER_MINT}") for m in group]
    assert len(mints) != 1, (
        "the labelled value names two tokens; collapsing it to one would be a "
        "guess presented as a fact"
    )
    # ...and whatever the extractor made of it, the signal is still refused.
    result = extract_token(f"CA: {CANONICAL_MINT} or {OTHER_MINT}")
    assert result.mint is None
    assert result.ambiguous is True


def test_a_label_never_overrides_itself_with_a_footer_mint():
    """
    The dangerous direction: the most frequent base58 string is NOT a better
    answer than an explicit label, however many times it appears.
    """
    text = f"CA: {CANONICAL_MINT}\n" + f"{OTHER_MINT} " * 5
    result = extract_token(text)
    assert result.mint == CANONICAL_MINT


def test_the_canonical_mint_repeated_many_times_is_not_ambiguous():
    text = f"CA: {CANONICAL_MINT}\n" + f"{CANONICAL_MINT} " * 4
    result = extract_token(text)
    assert result.mint == CANONICAL_MINT
    assert result.ambiguous is False


# ---------------------------------------------------------------------------
# non-signals
# ---------------------------------------------------------------------------

def test_a_ticker_only_post_is_not_a_token_signal(spec):
    case = spec["messages"][4]
    result = extract_token(case["text"])
    assert result.mint is None
    assert result.ambiguous is False
    assert result.symbol == "AI", "the ticker is still a symbol, just not a mint"
    assert any("no base58 mint" in w for w in result.warnings)


def test_an_empty_message_yields_no_mint_and_no_ambiguity():
    result = extract_token("")
    assert result.mint is None
    assert result.ambiguous is False


# ---------------------------------------------------------------------------
# the fixture stays honest about what it is
# ---------------------------------------------------------------------------

def test_the_fixture_is_the_exact_capture_not_a_trimmed_one(spec):
    """
    A trimmed fixture would hide exactly the decoy text this exists to test.
    Asserted so a future edit that "tidies" it fails loudly.
    """
    text = spec["messages"][0]["text"]
    for fragment in (
        "Anti Intelligence", "$AI", "CA: " + CANONICAL_MINT,
        "Market Cap", "Pool Age", "Liquid", "Volume",
        "DEX:  Paid", "Scans", "Holders", "Bundles",
        "PREMIUM CHANNEL", "Owner @k_vlde", "Dev:", "Sold:",
        "GMGN", "AXIOM", "BASED", "TROJAN", "BLOOM", "#AI",
    ):
        assert fragment in text, f"the capture is missing {fragment!r}"


def test_every_fixture_case_behaves_as_declared(spec):
    for case in spec["messages"]:
        result = extract_token(case["text"])
        assert result.ambiguous == case["expected_ambiguous"], case["id"]
        assert result.mint == case["expected_mint"], case["id"]

# ---------------------------------------------------------------------------
# unit normalisation of the executable quote (found by live data)
# ---------------------------------------------------------------------------

def test_a_sol_denominated_quote_is_compared_in_usd_not_sol():
    """
    A live run exposed this: the executable quote was SOL per token and the
    mark was USD per token, and comparing them directly produced a gap of
    -99.16% that read like a catastrophic mispricing.

    It was a unit error, and it was SILENT - both numbers were finite and
    positive, so nothing looked wrong. The fix is that the pipeline converts
    with the same logged ``--sol-usd`` reference that sized the order, and the
    verdict compares the USD figure.
    """
    from autopaper.same_market.models import ExecutableQuote, WSOL_MINT

    from autopaper.same_market.pipeline import SameMarketPipeline, TestConfig

    quote = ExecutableQuote(
        input_mint=WSOL_MINT, output_mint=CANONICAL_MINT,
        input_amount_raw="1150000", output_amount_raw="12921348000",
        input_decimals=9, output_decimals=6,
        input_amount=0.00115, output_amount=12921.348,
        effective_price=0.00115 / 12921.348,   # SOL per token
        effective_price_native=0.00115 / 12921.348,
    )
    pipeline = SameMarketPipeline(
        config=TestConfig(sol_reference_usd=200.0), dexscreener=None,
    )
    out = pipeline._express_quote_in_usd(quote)
    assert out.effective_price_usd == pytest.approx(
        (0.00115 / 12921.348) * 200.0
    )
    assert out.usd_reference_price == 200.0
    # The SOL figure is retained, never overwritten.
    assert out.effective_price == pytest.approx(0.00115 / 12921.348)
    assert "converted from SOL to USD" in out.detail


def test_without_a_sol_reference_no_usd_price_is_invented():
    """No reference means no USD price, and therefore no comparison."""
    from autopaper.same_market.models import ExecutableQuote, WSOL_MINT

    from autopaper.same_market.pipeline import SameMarketPipeline, TestConfig

    quote = ExecutableQuote(
        input_mint=WSOL_MINT, output_mint=CANONICAL_MINT,
        input_amount_raw="1150000", output_amount_raw="1",
        input_decimals=9, output_decimals=6,
        input_amount=0.00115, output_amount=1e-6,
        effective_price=1150.0, effective_price_native=1150.0,
    )
    out = SameMarketPipeline(
        config=TestConfig(sol_reference_usd=None), dexscreener=None,
    )._express_quote_in_usd(quote)
    assert out.effective_price_usd is None


def test_a_usdc_denominated_quote_is_already_usd_and_is_not_converted():
    from autopaper.same_market.models import ExecutableQuote

    from autopaper.same_market.pipeline import SameMarketPipeline, TestConfig

    quote = ExecutableQuote(
        input_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        output_mint=CANONICAL_MINT,
        input_amount_raw="230000", output_amount_raw="12921",
        input_decimals=6, output_decimals=6,
        input_amount=0.23, output_amount=0.012921,
        effective_price=17.8, effective_price_native=17.8,
    )
    out = SameMarketPipeline(
        config=TestConfig(sol_reference_usd=200.0), dexscreener=None,
    )._express_quote_in_usd(quote)
    assert out.effective_price_usd == 17.8
    assert out.usd_reference_price is None, "no conversion was applied"


def test_the_jupiter_pumpswap_label_matches_the_dexscreener_venue():
    """
    Jupiter labels the PumpSwap AMM ``Pump.fun Amm`` while DexScreener calls the
    same pool ``pumpswap``. Failing that pair as a VENUE_MISMATCH is a false
    alarm on precisely the case the harness exists to settle.
    """
    from autopaper.same_market.dexscreener import normalise_venue

    assert normalise_venue("Pump.fun Amm") == normalise_venue("pumpswap")
    assert normalise_venue("Pump.fun Amm") == "pumpswap"
    assert normalise_venue("raydium-clmm") == "raydium"
