"""
Telegram alert parsing.

Deliberately a PURE function: ``parse_alert(text)`` has no I/O, no clock and no
network, so every rule below is unit-testable and the same code path is used
offline in tests and live in production.

Design constraints:

* The exact alert format of the configured chat is not known ahead of time, so
  the parser is generic and multi-strategy rather than hard-coded to one bot.
* Source values are preserved verbatim in ``raw_fields``; parsed numbers live
  separately in ``fields`` with provenance. We never overwrite a raw value.
* Anything not present stays ``None``. No defaults, no fabrication.
* A Solana mint is validated by base58 alphabet and length; a Solana address is
  32-44 chars. A non-base58 or wrong-length string is rejected as a mint.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
MINT_RE = re.compile(rf"[1-9A-HJ-NP-Za-km-z]{{32,44}}")
URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)
# A ticker is 2-15 chars and must contain at least one letter. Without the
# letter rule, the "$0" in "$0.001" is read as a symbol.
TICKER_RE = re.compile(
    r"(?<![A-Za-z0-9$])\$([A-Za-z0-9_]{2,15})(?![A-Za-z0-9_])"
)
# A line whose ENTIRE content is one ``$``-prefixed token. SmugCalls prints the
# symbol on its own line ("$401si(k)"), and the general ticker rule above stops
# at the "(" - so the symbol would be read as "401si" and the identity would
# drift. A standalone token line is authoritative for the symbol and may carry
# unusual characters, including parentheses. Only a WHOLE line is accepted, so
# the MooDeng header ("... ｜ $AI") is never captured here.
STANDALONE_TICKER_RE = re.compile(
    r"^[ \t]*\$([A-Za-z0-9][A-Za-z0-9_().\-]{0,29})[ \t]*$", re.MULTILINE
)

# label -> canonical field. Order matters: first match wins, so put the more
# specific labels ("market cap") before the generic ones ("cap").
LABEL_ALIASES: dict[str, str] = {
    # identity
    "mint": "mint", "token address": "mint", "address": "mint",
    "ca": "mint", "contract": "mint", "contract address": "mint",
    "token": "symbol", "ticker": "symbol", "sym": "symbol", "symbol": "symbol",
    "coin": "symbol",
    "name": "name", "token name": "name", "project": "name",
    # price
    "price": "price_usd", "price usd": "price_usd", "usd price": "price_usd",
    "current price": "price_usd", "last price": "price_usd", "spot": "price_usd",
    "entry price": "price_usd", "trigger price": "price_usd",
    "initial price": "first_valid_price", "start price": "first_valid_price",
    "launch price": "first_valid_price", "first price": "first_valid_price",
    "ath": "ath_usd", "all time high": "ath_usd", "high": "ath_usd",
    "atl": "atl_usd", "all time low": "atl_usd",
    # size
    "market cap": "market_cap_usd", "marketcap": "market_cap_usd",
    "mcap": "market_cap_usd", "m cap": "market_cap_usd", "cap": "market_cap_usd",
    "mc": "market_cap_usd", "mc usd": "market_cap_usd",
    "fdv": "fdv_usd", "fully diluted": "fdv_usd",
    "liquidity": "liquidity_usd", "liq": "liquidity_usd", "liquidity usd": "liquidity_usd",
    "liquid": "liquidity_usd", "liquid usd": "liquidity_usd", "pool liquidity": "liquidity_usd",
    "pool liquidity": "liquidity_usd", "pool": "liquidity_usd",
    "volume": "volume_usd", "vol": "volume_usd", "volume 24h": "volume_24h_usd",
    "scans": "scan_count",
    "vol 24h": "volume_24h_usd", "volume 1h": "volume_1h_usd", "vol 1h": "volume_1h_usd",
    "volume 5m": "volume_5m_usd", "vol 5m": "volume_5m_usd",
    "mcap/l": "mcap_liquidity_ratio",
    # activity
    "age": "age_text", "token age": "age_text", "pair age": "age_text",
    "pool age": "age_text",
    "created": "created_text",
    "1h buys": "buys_1h", "buys 1h": "buys_1h", "1h buy": "buys_1h",
    "buy 1h": "buys_1h", "buys (1h)": "buys_1h",
    "1h sells": "sells_1h", "sells 1h": "sells_1h", "1h sell": "sells_1h",
    "sell 1h": "sells_1h", "sells (1h)": "sells_1h",
    "5m buys": "buys_5m", "buys 5m": "buys_5m",
    "5m sells": "sells_5m", "sells 5m": "sells_5m",
    "24h buys": "buys_24h", "buys 24h": "buys_24h",
    "24h sells": "sells_24h", "sells 24h": "sells_24h",
    "txns": "txns_5m", "transactions": "txns_5m", "txns 5m": "txns_5m",
    "holders": "holder_count", "holders count": "holder_count",
    "top 10": "top10_pct", "top10": "top10_pct", "top 10 holders": "top10_pct",
    "top holders": "top10_pct", "top10 holders": "top10_pct",
    "dev tokens": "dev_holdings_pct", "dev holdings": "dev_holdings_pct",
    "dev": "dev_holdings_pct", "dev holding": "dev_holdings_pct",
    "bundler": "bundler_count", "bundlers": "bundler_count",
    # PainApe reports bot-wallet activity as "Bots: N". The candidate model has
    # no dedicated bot counter; bundler_count is its bot-wallet count, so the
    # value is preserved there rather than dropped or invented under a new name.
    "bots": "bundler_count", "bot": "bundler_count",
    "bundles": "bundle_count",
    "bundled": "bundled_pct", "sold": "sold_pct",
    "sniper": "sniper_count", "snipers": "sniper_count",
    "insider": "insider_count", "insiders": "insider_count",
    "dex paid": "dex_paid", "dexpay": "dex_paid", "paid promo": "dex_paid",
    "dex": "dex_status",
    "fresh": "fresh", "is fresh": "fresh",
    "bundle": "bundle_count", "bundles": "bundle_count",
    "mint revokes": "mint_revoked", "revoked": "mint_revoked",
    "freeze": "freeze_authority", "freeze authority": "freeze_authority",
    "lp burned": "lp_burned", "lp burn": "lp_burned", "burned lp": "lp_burned",
    "rug score": "rug_score", "risk score": "risk_score", "score": "risk_score",
    "ath change": "from_ath_pct", "drawdown from ath": "from_ath_pct",
    "ath multiple": "ath_multiple", "x ath": "ath_multiple",
    "twitter followers": "twitter_followers", "x followers": "twitter_followers",
    "socials": "social_count",
    "ads": "dex_ads_paid", "ads paid": "dex_ads_paid",
    "owner": "alert_owner",
}

# fields that should be coerced to numbers
NUMERIC_FIELDS = {
    "price_usd", "first_valid_price", "ath_usd", "atl_usd", "market_cap_usd",
    "fdv_usd", "liquidity_usd", "volume_usd", "volume_5m_usd", "volume_1h_usd",
    "volume_24h_usd", "mcap_liquidity_ratio", "buys_1h", "sells_1h", "buys_5m",
    "sells_5m", "buys_24h", "sells_24h", "txns_5m", "holder_count", "top10_pct",
    "dev_holdings_pct", "bundler_count", "sniper_count", "insider_count",
    "bundle_count", "rug_score", "risk_score", "from_ath_pct", "ath_multiple",
    "twitter_followers", "social_count", "scan_count", "bundled_pct", "sold_pct",
}

INT_FIELDS = {
    "buys_1h", "sells_1h", "buys_5m", "sells_5m", "buys_24h", "sells_24h",
    "txns_5m", "holder_count", "bundler_count", "sniper_count", "insider_count",
    "bundle_count", "twitter_followers", "social_count", "scan_count",
}

BOOL_FIELDS = {
    "dex_paid", "fresh", "mint_revoked", "freeze_authority", "lp_burned",
    "dex_ads_paid",
}

AGE_UNITS = {
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "m": 60.0, "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
    "d": 86400.0, "day": 86400.0, "days": 86400.0,
    "w": 604800.0, "week": 604800.0, "weeks": 604800.0,
}

NUMBER_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

SOCIAL_HOSTS = {
    "twitter": "twitter", "x.com": "twitter", "t.me": "telegram",
    "telegram": "telegram", "discord": "discord", "dexscreener": "dexscreener",
    "geckoterminal": "geckoterminal", "pump.fun": "pumpfun",
    "birdeye": "birdeye", "rugcheck": "rugcheck", "solscan": "solscan",
}


@dataclass
class ParsedAlert:
    mint: str | None = None
    symbol: str | None = None
    name: str | None = None
    fields: dict[str, Any] = field(default_factory=dict)
    raw_fields: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)
    links: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    format: str = "generic"
    raw_text: str = ""

    def num(self, key: str) -> float | None:
        value = self.fields.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    def has(self, key: str) -> bool:
        return self.fields.get(key) is not None

    def age_seconds(self) -> float | None:
        """``age_text`` normalised to seconds. ``None`` when absent/unparsable."""
        if isinstance(self.fields.get("age_seconds"), (int, float)):
            return float(self.fields["age_seconds"])
        text = self.fields.get("age_text")
        if not text:
            return None
        m = re.search(r"(\d+(?:\.\d+)?)\s*([a-zA-Z]+)", str(text))
        if not m:
            return None
        try:
            value = float(m.group(1))
        except ValueError:
            return None
        unit = AGE_UNITS.get(m.group(2).lower().rstrip("s") or m.group(2).lower())
        if unit is None:
            unit = AGE_UNITS.get(m.group(2).lower())
        if unit is None:
            self.warnings.append(f"unparsed age unit: {m.group(2)!r}")
            return None
        return value * unit

    def as_dict(self) -> dict[str, Any]:
        return {
            "mint": self.mint,
            "symbol": self.symbol,
            "name": self.name,
            "format": self.format,
            "raw_text": self.raw_text,
            "fields": dict(self.fields),
            "raw_fields": dict(self.raw_fields),
            "provenance": dict(self.provenance),
            "links": dict(self.links),
            "warnings": list(self.warnings),
            "age_seconds": self.age_seconds(),
        }


# ---------------------------------------------------------------------------
# text normalisation
# ---------------------------------------------------------------------------


_TAG_RE = re.compile(r"<[^>]+>")
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\ufeff]")

# The alert bot in the configured channel uses fullwidth punctuation throughout
# (｜ U+FF5C as the field separator, ： U+FF1A as the label colon). Folding these
# to ASCII is what lets one set of rules parse the real feed instead of a
# hand-written approximation of it.
_FULLWIDTH_MAP = {
    "\uff5c": "|",   # ｜
    "\uff1a": ":",   # ：
    "\uff0c": ",",   # ，
    "\uff08": "(", "\uff09": ")",
    "\u3001": " ", "\u00a0": " ",
    "\u2192": "->",  # →
}


def normalise(text: str) -> str:
    """
    Unescape Telegram HTML, drop zero-width noise, collapse whitespace.

    Telegram wraps most alert fields in ``<b>``/``<code>``/``<pre>``, so tags are
    removed rather than left sitting between a label and its value.
    """
    if not text:
        return ""
    out = html.unescape(text)
    out = _TAG_RE.sub(" ", out)
    out = _ZERO_WIDTH.sub("", out)
    for wide, narrow in _FULLWIDTH_MAP.items():
        out = out.replace(wide, narrow)
    out = out.replace("", "").replace("‌", "").replace("‍", "")
    out = out.replace("\xa0", " ")
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


# ---------------------------------------------------------------------------
# value coercion
# ---------------------------------------------------------------------------


# The bot abbreviates money: "$51.3K", "$37.2k", "$1.2M", "$900". Dropping the
# suffix is a 1000x error, so it is read and applied rather than ignored.
# No \b after the number: "203.8 SOL" must read 203.8, not backtrack to 203
# because there is no word boundary between "8" and "S".
_SUFFIX_RE = re.compile(
    r"[-+]?\$?([0-9][0-9,]*(?:\.[0-9]+)?)\s*([KkMmBbTt])?(?![0-9])"
)
_SUFFIX_MULT = {
    "k": 1e3,
    "m": 1e6,
    "b": 1e9,
    "t": 1e12,
}


def _first_number(text: str) -> float | None:
    """
    First numeric token in ``text``, with any K/M/B/T magnitude suffix applied.

    Returns ``None`` rather than a partial number when the token cannot be read.
    """
    m = _SUFFIX_RE.search(text or "")
    if not m:
        return None
    try:
        value = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    suffix = m.group(2)
    if suffix:
        return value * _SUFFIX_MULT[suffix.lower()]
    return value


# Emoji markers the bot uses, and what they mean.
# Line-level patterns for percentage fields, because the value the label
# captures stops at the "|" before the percentage.
_PCT_LINE_PATTERNS = {
    "dev_holdings_pct": r"Dev\s*[:：][^\n]*?([0-9][0-9,]*(?:\.[0-9]+)?\s*%)",
    "top10_pct": r"TOP\s*10\s*[:：][^\n]*?([0-9][0-9,]*(?:\.[0-9]+)?\s*%)",
    "bundled_pct": r"Bundled\s*[:：][^\n]*?([0-9][0-9,]*(?:\.[0-9]+)?\s*%)",
    "sold_pct": r"Sold\s*[:：][^\n]*?([0-9][0-9,]*(?:\.[0-9]+)?\s*%)",
}

_ATH_MARKERS = ("🔝", "🔜")     # high-water / target
_NA_MARKERS = ("❌", "❎", "✖", "✗")


_YES_MARKS = ("✅", "✔", "☑")
_NO_MARKS = ("❌", "❎", "✖", "✗", "❌")


def _marked_flag(text: str, word: str) -> bool | None:
    """
    Read ``<word><mark>`` from a line, e.g. "Paid❌" -> False, "Paid✅" -> True.

    ``None`` when the word is not present at all. When the word is present with
    no mark, the bare word itself decides ("Paid" -> True, "Ads" -> False), which
    is what the bot means by naming the field.
    """
    lowered = (text or "").lower()
    idx = lowered.find(word.lower())
    if idx < 0:
        return None
    tail = (text or "")[idx + len(word):]
    tail = tail.lstrip()
    if tail:
        for mark in _NO_MARKS:
            if tail.startswith(mark):
                return False
        for mark in _YES_MARKS:
            if tail.startswith(mark):
                return True
    if word.lower().startswith("ads"):
        # an "Ads" mention with no mark is not a claim of a paid promotion
        return None
    return True


def _first_percent(text: str) -> float | None:
    """First explicit ``N%`` in the text. Prefers an explicit percentage."""
    m = re.search(r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*%", text or "")
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _coerce(key: str, value: str) -> Any:
    value = value.strip().strip("|").strip().strip("*").strip()
    if key in ("symbol", "name") and value.startswith("$"):
        value = value[1:].strip()
    value = value.strip("\u2022").strip()
    if value.startswith("$"):
        value = value[1:].strip()
    if value in {"", "-", "n/a", "N/A", "null", "None", "?"}:
        return None

    if key in BOOL_FIELDS:
        # "DEX: Paid❌ Ads❌" is a FALSE, and the cross must be stripped before
        # the check or the field is read as present-but-unknown.
        has_na = any(m in value for m in _NA_MARKERS)
        low = value.lower()
        if has_na or low in {"no", "false", "n", "0", "unpaid", "not burned", "off"}:
            return False
        if low in {"yes", "true", "y", "1", "✅", "✔", "✔️", "paid", "burned",
                   "revoked", "fresh", "on"}:
            return True
        return None

    # "Dev: 203.8 SOL | 0%" - the percentage is the holdings figure, the SOL
    # amount is just the raw size. Reading the first number would report 203.8%
    # of supply for a dev holding 0%, which is a catastrophic misread that would
    # block or pass a trade on pure noise.
    if key in ("dev_holdings_pct", "bundled_pct", "sold_pct", "top10_pct"):
        pct = _first_percent(value)
        if pct is not None:
            return pct
    if key in NUMERIC_FIELDS:
        n = _first_number(value)
        if n is None:
            return None
        if key in INT_FIELDS:
            return int(n)
        if key.endswith("_pct") and "%" not in value and abs(n) > 1.0:
            # "top 10 12" is ambiguous; keep the number and flag nothing here
            # because bots publish both "12" and "12%".
            return n
        return n
    return value


# ---------------------------------------------------------------------------
# key-value extraction
# ---------------------------------------------------------------------------

# Hard separators between fields. NOTE: comma is deliberately NOT a separator -
# in alert bots "$412,000" is one number, and splitting on the comma silently
# truncates every figure with a thousands separator to its first three digits.
# That is a correctness bug, not a style preference.
_KV_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"([A-Za-z0-9][A-Za-z0-9 _()./%+\-]{0,28}?)"
    r"[ \u00a0]*[:=\-\u2013][ \u00a0]*"
    r"([^|;\n\u2022\u00b7\t]*?)"
    r"[ \u00a0]*(?=\n|\||;|\u2022|\u00b7|\t|$)",
    re.MULTILINE,
)

_EJECT = {"http", "https", "com", "org", "io", "t", "co"}


def _label_to_key(label: str) -> str | None:
    key = LABEL_ALIASES.get(label)
    if key:
        return key
    norm = re.sub(r"\s+", " ", label).strip(" :*-·()[]#").lower()
    norm = re.sub(r"\s*\(\d+[hm]\)\s*", " ", norm).strip()
    if norm in LABEL_ALIASES:
        return LABEL_ALIASES[norm]
    return None


def _extract_kv(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for m in _KV_RE.finditer(text):
        label = m.group(1).strip()
        value = m.group(2).strip().strip("$").strip()
        if not value or len(value) > 64:
            continue
        if label.lower() in _EJECT or label.lower() in {"http", "https"}:
            continue
        out.append((label, value))
    return out


# ---------------------------------------------------------------------------
# whitespace-separated "LABEL value" pass
# ---------------------------------------------------------------------------
#
# Very common in alert bots:  "MC $88,000 | LIQ $12,400 | 5m Buys 210" with no
# colon at all. Only UNAMBIGUOUS metric labels are accepted here, and only at a
# segment boundary. Identity words (token, name, address, ca, project) are
# deliberately excluded: without a separator, "address here" would turn "here"
# into a mint and "TOKEN ALERT" would become a ticker.

BARE_LABELS: tuple[str, ...] = (
    "market cap", "marketcap", "mcap", "m cap", "mc", "mc usd", "cap",
    "liquid", "liquidity", "liq", "liquidity usd", "pool liquidity",
    "pool age", "age", "token age", "pair age",
    "volume", "vol", "scans", "holders", "bundles", "top 10", "top10",
    "dev", "dev tokens", "dev holdings", "dex", "dex paid",
    "price usd", "price", "usd price", "current price", "last price", "spot",
    "entry price", "trigger price",
    "initial price", "start price", "launch price", "first price",
    "ath", "all time high", "atl", "all time low",
    "market cap", "marketcap", "mcap", "m cap", "mc", "cap", "mc usd",
    "fdv", "fully diluted",
    "liquidity usd", "liquidity", "liq", "pool liquidity",
    "volume 24h", "vol 24h", "volume 1h", "vol 1h", "volume 5m", "vol 5m",
    "volume", "vol",
    "age", "token age", "pair age",
    "1h buys", "buys 1h", "1h buy", "buy 1h", "buys (1h)",
    "1h sells", "sells 1h", "1h sell", "sell 1h", "sells (1h)",
    "5m buys", "buys 5m", "5m sells", "sells 5m",
    "24h buys", "buys 24h", "24h sells", "sells 24h",
    "txns", "transactions", "txns 5m",
    "holders", "holders count",
    "top 10", "top10", "top 10 holders", "top holders", "top10 holders",
    "dev tokens", "dev holdings", "dev", "dev holding",
    "bundler", "bundlers", "sniper", "snipers", "insider", "insiders",
    "dex paid", "dexpay", "paid promo", "fresh", "is fresh",
    "bundle", "bundles", "rug score", "risk score", "score",
    "ath change", "drawdown from ath", "ath multiple", "x ath",
    "lp burned", "lp burn", "burned lp", "mint revokes", "revoked",
    "twitter followers", "x followers",
)

_BARE_ALTERNATION = "|".join(
    re.escape(a).replace(r"\ ", r"[ \u00a0]+") for a in sorted(BARE_LABELS, key=len, reverse=True)
)
# The separator between a bare label and its value may be a colon/equals/dash,
# spaces, or an APPROXIMATION glyph. SmugCalls writes "MC ~$10K", and without
# the "~" in this class the value never matches and the market cap is silently
# lost - which stalls the L25 liquidity gate on a candidate that did carry it.
_BARE_KV_RE = re.compile(
    rf"(?:^|[|,\u2022\n]|\b)({_BARE_ALTERNATION})"
    rf"[ \u00a0]*(?:[~\u2248][ \u00a0]*)?[:=\-]?[ \u00a0]*"
    rf"(\$?[A-Za-z0-9][A-Za-z0-9_,.()%/+-]{{0,40}})",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_bare_kv(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for m in _BARE_KV_RE.finditer(text):
        label, value = m.group(1), m.group(2).strip()
        key = _label_to_key(label)
        if key and value:
            out.append((label, value))
    return out


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------


def parse_alert(raw_text: str) -> ParsedAlert:
    text = normalise(raw_text)
    parsed = ParsedAlert(raw_text=text)
    if not text:
        parsed.warnings.append("empty message")
        return parsed

    # 0) JSON payloads first - some bots emit them
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            import json

            payload = json.loads(stripped)
        except Exception:  # noqa: BLE001
            payload = None
        if isinstance(payload, Mapping):
            parsed.format = "json"
            _absorb_json(parsed, payload)
            _finalise(parsed, text)
            return parsed

    # 1) links / socials
    for url in URL_RE.findall(text):
        low = url.lower()
        for host, key in SOCIAL_HOSTS.items():
            if host in low:
                parsed.links.setdefault(key, url)
                break
    if "twitter" in parsed.links:
        parsed.links["twitter"] = parsed.links.pop("twitter")
    elif "x.com" in text:
        m = re.search(r"https?://(?:www\.)?(?:x|twitter)\.com/[^\s]+", text, re.IGNORECASE)
        if m:
            parsed.links["twitter"] = m.group(0)

    # 2) mint
    mints = [m for m in MINT_RE.findall(text)]
    # prefer the address that follows a mint-ish label
    labelled = None
    for label, value in _extract_kv(text):
        if _label_to_key(label) == "mint":
            cand = _first_mint(value)
            if cand:
                labelled = cand
                break
    if labelled:
        parsed.mint = labelled
    elif mints:
        parsed.mint = max(set(mints), key=mints.count)
        if len(set(mints)) > 1:
            parsed.warnings.append("multiple base58 strings; chose most frequent")

    # 3) ticker
    # A standalone "$TICKER" line is authoritative (SmugCalls puts the symbol on
    # its own line) and may carry characters the general rule stops at, such as
    # parentheses. Checked first so "$401si(k)" is read whole, not as "401si".
    standalone = STANDALONE_TICKER_RE.search(text)
    if standalone:
        parsed.symbol = standalone.group(1)
    else:
        tickers = [t for t in TICKER_RE.findall(text) if re.search(r"[A-Za-z]", t)]
        if tickers:
            parsed.symbol = tickers[0]
        upper_head = re.search(r"^\s*([A-Z0-9]{2,15})\s*[|/—–-]", text)
        if upper_head and upper_head.group(1) not in {"DEX", "USD", "SOL", "API", "ATH"}:
            parsed.symbol = upper_head.group(1)

    # 4) labelled fields
    seen: set[str] = set()
    pairs = _extract_kv(text) + _extract_bare_kv(text)
    # Percentage fields first, from the whole LINE.
    #
    # The bot writes "Dev:  203.8 SOL | 0%" - the holdings percentage is a
    # sibling segment AFTER the separator, while the value the label captured
    # stops at it. Reading the captured value reports 203.8% of supply for a dev
    # holding 0%, which is a catastrophic misread built on the wrong number.
    for key, pattern in _PCT_LINE_PATTERNS.items():
        if key in seen:
            continue
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            continue
        value = _first_percent(m.group(0))
        if value is None:
            continue
        seen.add(key)
        parsed.fields[key] = value
        parsed.raw_fields[key] = m.group(0).strip()
        parsed.provenance[key] = "alert_label:line"

    for label, value in pairs:
        key = _label_to_key(label)
        if not key or key in seen:
            continue
        if key == "dex_status":
            # "DEX:  Paid❌ Ads❌ 0⚡" is three facts on one line. Coercing the
            # whole line to one boolean loses the difference between DEX Paid
            # and Ads, and a single stray cross would make both false.
            # Word scanning uses string operations, not a regex character class.
            # These are multi-codepoint symbols and a bare run of them inside a
            # pattern is a literal SEQUENCE, not a class - which silently matches
            # nothing and turns every "Paid<cross>" into a paid promotion.
            for sub_key, word in (("dex_paid", "paid"), ("dex_ads_paid", "ads")):
                if sub_key in seen:
                    continue
                found = _marked_flag(value, word)
                if found is None:
                    continue
                seen.add(sub_key)
                parsed.fields[sub_key] = found
                parsed.raw_fields[sub_key] = value
                parsed.provenance[sub_key] = f"alert_label:DEX/{word}"
            continue
        coerced = _coerce(key, value)
        if coerced is None:
            continue
        seen.add(key)
        parsed.fields[key] = coerced
        parsed.raw_fields[key] = value
        parsed.provenance[key] = f"alert_label:{label.strip()}"

    # explicit "name" that is not just the symbol
    name = parsed.fields.get("name")
    if isinstance(name, str) and name and name.lower() not in {"", "n/a"}:
        if parsed.symbol and name.upper() == parsed.symbol.upper():
            parsed.fields.pop("name", None)
        else:
            parsed.name = name
    parsed.symbol = parsed.fields.get("symbol") or parsed.symbol

    # 5) links into fields
    if "twitter" in parsed.links and not parsed.fields.get("twitter"):
        parsed.fields["twitter"] = parsed.links["twitter"]
        parsed.provenance["twitter"] = "url"
    if "telegram" in parsed.links and not parsed.fields.get("telegram"):
        parsed.fields["telegram"] = parsed.links["telegram"]
        parsed.provenance["telegram"] = "url"
    if "website" in parsed.links:
        parsed.fields.setdefault("website", parsed.links["website"])
        parsed.provenance.setdefault("website", "url")

    # 6) derive what can only be derived
    _derive(parsed)

    _finalise(parsed, text)
    return parsed


def _first_mint(value: str) -> str | None:
    m = MINT_RE.search(value.replace(" ", ""))
    return m.group(0) if m else None


def _absorb_json(parsed: ParsedAlert, payload: Mapping[str, Any]) -> None:
    def take(key: str, target: str) -> None:
        if key in payload and payload[key] is not None:
            parsed.fields[target] = payload[key]
            parsed.provenance[target] = "alert_json"

    for src, dst in (
        ("mint", "mint"), ("address", "mint"), ("tokenAddress", "mint"),
        ("token_address", "mint"), ("ca", "mint"),
        ("symbol", "symbol"), ("ticker", "symbol"), ("name", "name"),
        ("price", "price_usd"), ("price_usd", "price_usd"), ("priceUsd", "price_usd"),
        ("market_cap", "market_cap_usd"), ("marketCap", "market_cap_usd"),
        ("mcap", "market_cap_usd"), ("liquidity", "liquidity_usd"),
        ("volume", "volume_usd"), ("volume_24h", "volume_24h_usd"),
        ("buys_1h", "buys_1h"), ("sells_1h", "sells_1h"),
        ("holder_count", "holder_count"), ("age", "age_text"),
        ("ath", "ath_usd"), ("twitter", "twitter"), ("telegram", "telegram"),
        ("website", "website"),
    ):
        take(src, dst)
    if "mint" in parsed.fields:
        parsed.mint = str(parsed.fields["mint"])


# The bot appends the high-water / target figure after an emoji marker, e.g.
# "Market Cap:  $51.3K |^$85.1K" or "MC:$18k -> $42.8k". It is a real reported
# number, so it is captured as its own field rather than discarded.
_ATH_LINE_RE = re.compile(
    r"(?:Market Cap|MC)\s*[:：][^\n]*?[🔝🔜]\s*\$?\s*"
    r"([0-9][0-9,]*(?:\.[0-9]+)?\s*[KkMmBbTt])?"
)
_ATH_MULTI_RE = re.compile(
    r"MC\s*[:：]\s*\$?\s*[0-9][0-9,.]*\s*[KkMmBbTt]?"
    r"\s*(?:🔜|->|→)\s*\$?\s*([0-9][0-9,.]*\s*[KkMmBbTt])"
)


def _derive(parsed: ParsedAlert) -> None:
    f = parsed.fields
    raw = parsed.raw_fields
    # buys/sells -> ratio only when both are actually present
    buys, sells = f.get("buys_1h"), f.get("sells_1h")
    if isinstance(buys, (int, float)) and isinstance(sells, (int, float)):
        f["txns_1h"] = int(buys) + int(sells)
        f["buy_sell_ratio_1h"] = (buys / sells) if sells else None
        if sells == 0 and buys > 0:
            f["buy_sell_ratio_1h"] = None
            parsed.warnings.append("sells_1h==0: ratio not computed (undefined)")
    if parsed.symbol and "name" not in f:
        f["name"] = parsed.symbol
        parsed.provenance["name"] = "derived:symbol"
    if parsed.name and "symbol" not in f:
        f["symbol"] = parsed.name[:15]
        parsed.provenance["symbol"] = "derived:name"
    # high-water / target figure reported by the bot
    if "ath_usd" not in f:
        m = _ATH_LINE_RE.search(
            raw.get("market_cap_usd", "")
        ) or _ATH_MULTI_RE.search(parsed.raw_text or "")
        if m:
            ath = _first_number(m.group(1) or "")
            if ath:
                f["ath_usd"] = ath
                parsed.provenance["ath_usd"] = "alert_label:ath_marker"
    if "x_ath" not in f and f.get("market_cap_usd") and f.get("ath_usd"):
        f["x_ath"] = f["ath_usd"] / f["market_cap_usd"]
        parsed.provenance["x_ath"] = "derived"

    # drawdown from ATH only when both exist
    price, ath = f.get("price_usd"), f.get("ath_usd")
    if isinstance(price, (int, float)) and isinstance(ath, (int, float)) and ath > 0:
        f["from_ath_pct"] = (price - ath) / ath * 100.0
        parsed.provenance["from_ath_pct"] = "derived:price_vs_ath"
    # extension from first valid price when the alert supplies both
    fvp = f.get("first_valid_price")
    if isinstance(price, (int, float)) and isinstance(fvp, (int, float)) and fvp > 0:
        f["extension_from_first_valid_price"] = (price - fvp) / fvp
        parsed.provenance["extension_from_first_valid_price"] = "derived:price_vs_fvp"
    # mcap / liquidity
    mc, liq = f.get("market_cap_usd"), f.get("liquidity_usd")
    if isinstance(mc, (int, float)) and isinstance(liq, (int, float)) and liq > 0:
        f["mcap_liquidity_ratio"] = mc / liq
        parsed.provenance["mcap_liquidity_ratio"] = "derived"


def _finalise(parsed: ParsedAlert, text: str) -> None:
    # Identity may have arrived via the JSON path or a labelled field. A mint
    # MUST look like base58 of the right length - anything else is discarded
    # with a warning rather than becoming a tradeable identifier.
    if not parsed.mint and isinstance(parsed.fields.get("mint"), str):
        candidate = MINT_RE.search(str(parsed.fields["mint"]).replace(" ", ""))
        if candidate:
            parsed.mint = candidate.group(0)
        else:
            parsed.warnings.append(
                f"rejected non-base58 mint value {str(parsed.fields['mint'])[:48]!r}"
            )
            parsed.fields.pop("mint", None)
    if not parsed.symbol and isinstance(parsed.fields.get("symbol"), str):
        parsed.symbol = str(parsed.fields["symbol"]).lstrip("$").strip() or None
    if not parsed.name and isinstance(parsed.fields.get("name"), str):
        parsed.name = str(parsed.fields["name"]).strip() or None
    if not parsed.mint:
        parsed.warnings.append("no base58 mint found")
    if parsed.format == "generic":
        labels = len(parsed.fields)
        parsed.format = (
            "rich" if labels >= 8 else
            "standard" if labels >= 4 else
            "sparse" if labels >= 1 else
            "empty"
        )
    if not parsed.fields:
        parsed.warnings.append("no recognised fields; relying on market data only")


def is_token_candidate(text: str) -> tuple[bool, str]:
    """
    Is this a tradeable token alert, and if not, why?

    The configured channel carries more than alerts: narrative and promo posts
    ("fric | $fric / #fric 2.4X MC:$18k") have prices and tickers but no
    contract address. Those are not candidates, and conflating "not a token
    alert" with "failed to parse a token alert" would make the log lie about how
    well the feed is being understood.
    """
    parsed = parse_alert(text)
    if parsed.mint:
        return True, "mint present"
    if TICKER_RE.search(normalise(text)) or any(
        k in parsed.fields for k in ("market_cap_usd", "liquidity_usd")
    ):
        return False, "narrative/promo post: ticker or figures but no contract address"
    return False, "no contract address found"


def looks_like_alert(text: str) -> bool:
    """
    Cheap pre-filter so ordinary chat does not become a candidate.

    Accepted: any base58 mint, or a ticker plus at least two recognised labels.
    A bare "price/market cap" message with no address is NOT an alert - there is
    nothing to analyse without a mint.
    """
    if not text or not text.strip():
        return False
    clean = normalise(text)
    if MINT_RE.search(clean):
        return True
    parsed = parse_alert(clean)
    return bool(parsed.symbol and len(parsed.fields) >= 2)
