"""
Telegram alert parsing.

The alert format of the configured chat is unknown ahead of time, so these
tests cover several plausible shapes plus the failure modes that matter:
no mint, ambiguous mints, missing fields, and unit parsing.
"""

from __future__ import annotations

import pytest

from autopaper.telegram.parser import looks_like_alert, normalise, parse_alert

MINT = "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ2E"
OTHER = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


def test_looks_like_alert():
    assert looks_like_alert(f"NEW TOKEN $PEPE\nCA: {MINT}\nMC: $45,000")
    assert not looks_like_alert("")
    assert not looks_like_alert("good morning chat")


def test_label_style():
    text = f"""
<b>NEW TOKEN ALERT</b>
Name: Doge Rocket
Symbol: $DROKE
CA: {MINT}
Price: $0.0000421
Market Cap: $412,000
Liquidity: $38,500
Volume 24h: $210,000
Age: 4m
1H Buys: 412
1H Sells: 188
ATH: $0.0000610
Fresh: yes
Top 10: 22%
Dev Tokens: 1.2%
Bundler: 0
Sniper: 2
Insider: 0
DEX Paid: no
Website: https://doge.example
X: https://x.com/dogerocket
"""
    p = parse_alert(text)
    assert p.mint == MINT
    assert p.symbol == "DROKE"
    assert p.fields["price_usd"] == pytest.approx(0.0000421)
    assert p.fields["market_cap_usd"] == pytest.approx(412_000)
    assert p.fields["liquidity_usd"] == pytest.approx(38_500)
    assert p.fields["volume_24h_usd"] == pytest.approx(210_000)
    assert p.fields["buys_1h"] == 412
    assert p.fields["sells_1h"] == 188
    assert p.fields["ath_usd"] == pytest.approx(0.0000610)
    assert p.fields["fresh"] is True
    assert p.fields["dex_paid"] is False
    assert p.fields["top10_pct"] == pytest.approx(22.0)
    assert p.age_seconds() == pytest.approx(240.0)
    assert p.links["twitter"] == "https://x.com/dogerocket"
    assert p.provenance["liquidity_usd"].startswith("alert_label:")
    assert p.format in ("standard", "rich")


def test_inline_pipe_style():
    text = (
        f"🚀 $MOON | {MINT} | MC $88,000 | LIQ $12,400 | "
        f"5m Buys 210 | 5m Sells 95 | Vol $64,000 | Age 90s"
    )
    p = parse_alert(text)
    assert p.mint == MINT
    assert p.symbol == "MOON"
    assert p.fields["market_cap_usd"] == pytest.approx(88_000)
    assert p.fields["liquidity_usd"] == pytest.approx(12_400)
    assert p.fields["buys_5m"] == 210
    assert p.fields["sells_5m"] == 95
    assert p.age_seconds() == pytest.approx(90.0)


def test_json_payload():
    text = (
        '{"mint": "' + MINT + '", "symbol": "JSON", "price": 0.5, '
        '"market_cap": 1000, "liquidity": 900, "buys_1h": 5, "sells_1h": 4, '
        '"holder_count": 33}'
    )
    p = parse_alert(text)
    assert p.format == "json"
    assert p.mint == MINT
    assert p.symbol == "JSON"
    assert p.fields["price_usd"] == pytest.approx(0.5)
    assert p.fields["holder_count"] == 33


def test_derived_buy_sell_ratio():
    p = parse_alert(f"SYM\nCA: {MINT}\n1H Buys: 300\n1H Sells: 100")
    assert p.fields["buy_sell_ratio_1h"] == pytest.approx(3.0)
    assert p.fields["txns_1h"] == 400


def test_zero_sells_does_not_fabricate_infinite_ratio():
    p = parse_alert(f"SYM\nCA: {MINT}\n1H Buys: 300\n1H Sells: 0")
    assert p.fields["buy_sell_ratio_1h"] is None
    assert any("ratio not computed" in w for w in p.warnings)


def test_extension_derived_from_supplied_first_price():
    p = parse_alert(
        f"SYM\nCA: {MINT}\nInitial Price: $0.000010\nPrice: $0.000016"
    )
    assert p.fields["first_valid_price"] == pytest.approx(1e-5)
    assert p.fields["extension_from_first_valid_price"] == pytest.approx(0.6)


def test_drawdown_from_ath_derived():
    p = parse_alert(f"SYM\nCA: {MINT}\nATH: $0.000020\nPrice: $0.000015")
    assert p.fields["from_ath_pct"] == pytest.approx(-25.0)


def test_missing_fields_stay_none():
    p = parse_alert(f"just a token {MINT} with no numbers")
    assert p.mint == MINT
    assert p.fields.get("liquidity_usd") is None
    assert p.fields.get("buys_1h") is None
    assert p.fields.get("dex_paid") is None
    assert "holder_count" not in p.fields
    assert p.format in ("empty", "sparse")
    # a bare mint with no numbers must not acquire invented metrics
    assert set(p.fields) <= {"name", "symbol"}


def test_no_mint_is_reported_not_guessed():
    p = parse_alert("Price: $0.001\nMarket Cap: $5000\nno address here")
    assert p.mint is None
    assert any("no base58 mint" in w for w in p.warnings)
    assert not looks_like_alert("Price: $0.001\nMarket Cap: $5000\nno address here")


def test_invalid_base58_string_is_not_a_mint():
    # 0, O, I and l are not in the base58 alphabet
    p = parse_alert("CA: 0OIl-not-base58-at-all-really-32chars")
    assert p.mint is None


def test_labelled_mint_wins_over_incidental_base58():
    text = f"contract: {MINT}  |  solana: {OTHER}"
    p = parse_alert(text)
    assert p.mint == MINT


def test_ambiguous_multiple_mints_warns():
    text = f"first {MINT}\nsecond {OTHER}"
    p = parse_alert(text)
    assert p.mint in (MINT, OTHER)
    assert any("multiple base58" in w for w in p.warnings)


def test_html_and_zero_width_normalisation():
    text = f"<b>ALERT</b> {MINT} &amp; friends\n<b>Price</b>: $0.1"
    p = parse_alert(text)
    assert p.mint == MINT
    assert p.fields["price_usd"] == pytest.approx(0.1)


def test_ticker_variants():
    assert parse_alert(f"$DOGE {MINT}").symbol == "DOGE"
    assert parse_alert(f"Ticker: FROG\nCA: {MINT}").symbol == "FROG"
    assert parse_alert(f"Name: Real Name\nSymbol: RN\nCA: {MINT}").fields[
        "name"
    ] == "Real Name"


def test_age_units():
    for text, expected in (("Age: 30s", 30.0), ("Age: 3m", 180.0),
                           ("Age: 2h", 7200.0), ("Age: 1d", 86400.0)):
        p = parse_alert(f"CA: {MINT}\n{text}")
        assert p.age_seconds() == pytest.approx(expected), text


def test_garbage_input_is_survivable():
    for text in ("", "   ", "\n\n", "<b></b>", "%%%$$$", "1"):
        p = parse_alert(text)
        assert isinstance(p.as_dict(), dict)


def test_normalise_strips_markup():
    # tags become a space so they cannot weld two words together
    assert normalise("<b>a</b>&amp;b") == "a &b"
    assert normalise("<code>1H Buys: 5</code>").strip() == "1H Buys: 5"
