"""
Phanes narrative source over Telegram.

``@Phanes_bot`` answers ``/nar <CA>`` with a narrative paragraph, e.g.::

    Narrative for USDC

    The USDC token originated from announcements by Circle about launching a
    stablecoin pegged to the US dollar, with the background being its 2018
    debut as a digital dollar alternative ...

    — Accuracy not guaranteed.

What this source therefore DOES provide: ``narrative`` text, and whether a
narrative exists at all.

What it does NOT provide, and what is therefore left ``None`` everywhere:
holder counts, buy/sell counts, liquidity, security flags, bundler/sniper/
insider counts, wallet flow, holder concentration, and any "score". The bot does
not emit them. The agent must not reverse-engineer them out of prose.

Two design points worth being explicit about:

1. **The narrative is untrusted external content.** It is prose written by
   somebody else and delivered over the network. It is parsed into a string and
   a couple of coverage facts, and nothing else. It can never select the mint,
   the symbol, the route, or the decision - those come from the alert and the
   market data. ``tests/test_phanes_nar.py`` asserts this by feeding the parser
   prompt-injection-shaped text.

2. **The bot must not be spammed.** Queries are serialised behind a minimum
   gap with a bounded queue. If the queue is saturated the source reports
   NAR_UNAVAILABLE and the candidate proceeds downgraded - the experiment
   degrades its own evidence, it does not hammer somebody else's service.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

from ..clock import Clock, SystemClock
from ..logging_setup import log
from ..models import NarSnapshot

DEFAULT_BOT = "Phanes_bot"
COMMAND = "/nar"
#: The narrative FALLBACK bot. Reuses the identical ``/nar <CA>`` protocol; it is
#: only consulted after the primary bot fails several consecutive OPERATIONAL
#: times, and the primary is restored as soon as it answers again.
DEFAULT_FALLBACK_BOT = "tokenscan"

# ---------------------------------------------------------------------------
# The real /nar reply for a memecoin is a STATS CARD, not prose. Parsed below.
# ---------------------------------------------------------------------------

SUBSCRIPT = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
# "$0.0₄1322" - the subscript digit is the count of leading zeros after the
# decimal point, and the following digits are the significant figures. So the
# value is digits * 10^-(zeros + len(digits)):
#   0.0 + 4 zeros + "1322"  ->  1322e-8  =  1.322e-5   (0.00001322)
# Verified against the card's own MC/supply: 1.322e-5 * 1e9 = $13.2k.
_SUBSCRIPT_RE = re.compile(
    r"\$?0\.0([₀₁₂₃₄₅₆₇₈₉]+)([0-9]+)"
)

_CARD_HEAD_RE = re.compile(
    r"^[^\S\n]*(?:\U0001FA99|🪙)?[^\S\n]*([^\n($]+?)\s*\(\s*\$?([A-Za-z0-9_.]{1,15})\s*\)"
)
_AGE_RE = re.compile(r"🌱\s*(\d+(?:\.\d+)?)\s*([smhdw])", re.IGNORECASE)
_HOLDERS_EYE_RE = re.compile(r"👁️?\s*(\d+)")
_BUYS_RE = re.compile(r"🅑\s*([0-9,]+)")
_SELLS_RE = re.compile(r"Ⓢ\s*([0-9,]+)")
_FRESH_RE = re.compile(r"Fresh\s*([0-9.]+)%\s*1D\s*\|\s*([0-9.]+)%\s*7D")
_TOP10_RE = re.compile(r"Top\s*10\s*([0-9.]+)%\s*\|\s*([0-9,]+)")
_DEV_TOKS_RE = re.compile(r"Dev\s*Toks\s*([0-9,]+)\s*\|\s*(\d+)")
_BUNDLER_RE = re.compile(r"Bundler\s*([0-9,]+)\s*\(([0-9.]+)%\)")
_SNIPER_RE = re.compile(r"Sniper\s*([0-9,]+)\s*\(([0-9.]+)%\)")
_ATH_RE = re.compile(
    r"ATH\s*\$?([0-9,.]+[KkMmBb]?)\s*\(\s*(-?[0-9.]+)%\s*(?:/\s*([0-9]+[smhdw]))?\s*\)"
)
_SUPPLY_RE = re.compile(r"Sup\s*([0-9.,]+[KkMmBb]?)\s*/")
_SOCIAL_RE = re.compile(r"𝕏\s*•?\s*\[?([A-Za-z0-9_.\-]{1,32})\]?")
_UNIT_S = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0}


def _dur(value: str | None, unit: str | None) -> float | None:
    if not value or not unit:
        return None
    try:
        base = float(value)
    except ValueError:
        return None
    return base * _UNIT_S.get(unit.lower(), 1.0)


def parse_price_subscript(text: str) -> float | None:
    """``$0.0₄1322`` -> ``1.322e-05``."""
    m = _SUBSCRIPT_RE.search(text or "")
    if not m:
        return None
    # The subscript is a DIGIT whose value is the zero count, not a run of
    # characters: "\u2084" is the character 4, so len() would be 1 and the price
    # would come out 1000x too high.
    try:
        zeros = int(m.group(1).translate(SUBSCRIPT))
        digits = m.group(2)
        return int(digits) * (10.0 ** -(zeros + len(digits)))
    except (ValueError, OverflowError):
        return None


def _mag(text: str | None) -> float | None:
    if not text:
        return None
    from ..telegram.parser import _first_number

    return _first_number(text)


def parse_stats_card(text: str, mint: str, now: float) -> NarSnapshot:
    """
    Parse the ``/nar`` stats card into a NarSnapshot.

    This is a real data source - price, 1h buys/sells, ATH, holder
    concentration, dev supply, bundler and sniper counts, DEX-paid status. Where
    the card omits a field it stays ``None``; nothing is back-filled from prose.

    The card is cross-checked against itself: price x supply must land near the
    stated market cap. A card that fails its own arithmetic is reported as
    unavailable rather than traded on.
    """
    lines = (text or "").splitlines()
    body = "\n".join(lines)
    first = lines[0] if lines else ""

    head = _CARD_HEAD_RE.search(first)
    token_name = head.group(1).strip() if head else None
    token_symbol = head.group(2).strip() if head else None

    price = parse_price_subscript(body)
    price_change = None
    m = re.search(r"\(([+-]?[0-9.]+)%\)", first)
    if m:
        try:
            price_change = float(m.group(1))
        except ValueError:
            price_change = None

    def row(label: str) -> str | None:
        m = re.search(
            rf"^\s*[\u250c\u251c\u2514\u2500\u2510\u252c\u2534\u253c\u2502]?"
            rf"\s*{re.escape(label)}\s+(.+?)\s*$", body, re.MULTILINE
        )
        return m.group(1).strip() if m else None

    mcap = _mag(row("MC"))
    volume = _mag(row("Vol"))
    liquidity = _mag(row("LP"))

    supply = None
    m = _SUPPLY_RE.search(body)
    if m:
        supply = _mag(m.group(1))

    change_1h = None
    m = re.search(r"([+-]?[0-9.]+)%\s*🅑", body)
    if m:
        try:
            change_1h = float(m.group(1))
        except ValueError:
            change_1h = None

    buys = sells = None
    m = _BUYS_RE.search(body)
    if m:
        buys = int(m.group(1).replace(",", ""))
    m = _SELLS_RE.search(body)
    if m:
        sells = int(m.group(1).replace(",", ""))
    ratio = None
    if buys is not None and sells:
        ratio = buys / sells

    ath_row = row("ATH")
    ath = _mag(ath_row) if ath_row else None
    from_ath = ath_age = None
    if ath_row:
        inner = re.search(r"\(([^)]*)\)", ath_row)
        inner = inner.group(1) if inner else ""
        m = re.search(r"(-?[0-9.]+)%", inner)
        if m:
            try:
                from_ath = float(m.group(1))
            except ValueError:
                from_ath = None
        m = re.search(r"(\d+(?:\.\d+)?)\s*([smhdw])", inner)
        if m:
            ath_age = _dur(m.group(1), m.group(2))

    age_s = None
    m = _AGE_RE.search(body)   # the seed-age marker is on the second line
    if m:
        age_s = _dur(m.group(1), m.group(2))

    holders = top10 = fresh_1d = fresh_7d = None
    m = _TOP10_RE.search(body)
    if m:
        top10 = float(m.group(1))
        holders = int(m.group(2).replace(",", ""))
    if holders is None:
        m = _HOLDERS_EYE_RE.search(body)
        if m:
            holders = int(m.group(1))
    m = _FRESH_RE.search(body)
    if m:
        fresh_1d, fresh_7d = float(m.group(1)), float(m.group(2))

    dev_tokens = dev_large = None
    m = _DEV_TOKS_RE.search(body)
    if m:
        dev_tokens = int(m.group(1).replace(",", ""))
        dev_large = int(m.group(2))

    bundler = bundled_pct = None
    m = _BUNDLER_RE.search(body)
    if m:
        bundler = int(m.group(1).replace(",", ""))
        bundled_pct = float(m.group(2))

    sniper = sniper_pct = None
    m = _SNIPER_RE.search(body)
    if m:
        sniper = int(m.group(1).replace(",", ""))
        sniper_pct = float(m.group(2))

    dex_paid = None
    if re.search(r"DEX\s*Paid", body, re.IGNORECASE):
        dex_paid = "\U0001f534" in body or "🔴" in body

    social = _SOCIAL_RE.search(body)
    twitter = social.group(1) if social else None

    flags: list[str] = []
    if dev_tokens is not None and dev_tokens > 1000:
        flags.append("large_dev_token_count")
    if bundled_pct is not None and bundled_pct >= 30:
        flags.append("heavy_bundling")
    if dex_paid:
        flags.append("dex_paid")

    consistency = None
    if price and supply and mcap and mcap > 0:
        consistency = (price * supply) / mcap
        if not (0.5 <= consistency <= 2.0):
            return NarSnapshot(
                mint=mint, fetched_at=now, source="phanes_bot",
                available=False,
                error=(
                    "NAR_UNAVAILABLE: card failed its own consistency check "
                    f"(price x supply is {consistency:.2f}x the stated market cap)"
                ),
            )

    return NarSnapshot(
        mint=mint,
        fetched_at=now,
        source="phanes_bot",
        available=True,
        narrative=None,
        social_activity=change_1h,
        holder_count=holders,
        top10_holders_pct=top10,
        dev_holdings_pct=None,
        bundler_count=bundler,
        sniper_count=sniper,
        insider_count=None,
        security_flags=tuple(flags) or None,
        wallet_flow_net=None,
        dex_paid=dex_paid,
        website=None,
        twitter=twitter,
        telegram=None,
        error=None,
        price_usd=price,
        price_change_pct=price_change,
        market_cap_usd=mcap,
        volume_usd=volume,
        liquidity_usd=liquidity,
        supply=supply,
        change_1h_pct=change_1h,
        buys_1h=buys,
        sells_1h=sells,
        buy_sell_ratio_1h=ratio,
        from_ath_pct=from_ath,
        ath_age_s=ath_age,
        age_s=age_s,
        fresh_1d_pct=fresh_1d,
        fresh_7d_pct=fresh_7d,
        dev_token_count=dev_tokens,
        dev_large_wallets=dev_large,
        bundled_pct=bundled_pct,
        sniper_pct=sniper_pct,
        card_consistency=consistency,
        token_name=token_name,
        token_symbol=token_symbol,
    )


# The bot's own framing lines. Everything before the first one of these is the
# narrative; everything after is boilerplate.
_HEADER_RE = re.compile(r"^\s*Narrative\s+for\s+(?P<subject>.+?)\s*$", re.IGNORECASE)
_FOOTER_RE = re.compile(
    r"^\s*[—\-–]\s*Accuracy not guaranteed\.?\s*$", re.IGNORECASE
)

# Phrases the bot uses when it has nothing. Treated as a placeholder, never as a
# narrative, and never as evidence in either direction.
_PLACEHOLDER_MARKERS = (
    "no narrative",
    "not available",
    "no information",
    "unable to find",
    "could not find",
    "no data",
    "unknown token",
    "n/a",
)

# Concrete, checkable detail in prose: a year, a percentage, a proper noun-ish
# capitalised token, a number with a unit. Used only to distinguish "a real
# narrative" from "a sentence of nothing", never to score hype.
_SPECIFIC_RE = re.compile(
    r"(\b(19|20)\d{2}\b)"          # years
    r"|(\b\d+(?:\.\d+)?\s*%)"      # percentages
    r"|(\b[A-Z][a-zA-Z]{3,}\b)"    # capitalised words
)

MAX_NARRATIVE_CHARS = 1200


def parse_nar_response(text: str, mint: str, now: float) -> NarSnapshot:
    """
    Parse a ``/nar`` reply into a NarSnapshot.

    Deterministic and side-effect free. Returns all-unset fields when the
    response is empty, malformed, or a placeholder.
    """
    if not text or not text.strip():
        return NarSnapshot(
            mint=mint, fetched_at=now, source="phanes_bot",
            available=False, error="NAR_UNAVAILABLE: empty bot response",
        )

    lines = text.splitlines()
    subject: str | None = None
    body: list[str] = []
    for line in lines:
        if subject is None:
            m = _HEADER_RE.match(line)
            if m:
                subject = m.group("subject").strip()
                continue
            if _FOOTER_RE.match(line):
                continue
            body.append(line)
            continue
        if _FOOTER_RE.match(line):
            break
        body.append(line)

    narrative = "\n".join(body).strip()
    if not narrative:
        return NarSnapshot(
            mint=mint, fetched_at=now, source="phanes_bot",
            available=False,
            error="NAR_UNAVAILABLE: bot replied with no narrative body",
        )

    lowered = narrative.lower()
    placeholder = any(marker in lowered for marker in _PLACEHOLDER_MARKERS)
    # The bot's footer disclaimer is not evidence of anything; it is always on.
    specifics = len(set(m.group(0).lower() for m in _SPECIFIC_RE.finditer(narrative)))

    return NarSnapshot(
        mint=mint,
        fetched_at=now,
        source="phanes_bot",
        # A placeholder is "the bot answered but has nothing", which is reported
        # as unavailable rather than presented as a narrative.
        available=not placeholder,
        narrative=narrative[:MAX_NARRATIVE_CHARS],
        narrative_score=None,          # the bot does not provide one
        social_activity=None,
        holder_count=None,
        top10_holders_pct=None,
        dev_holdings_pct=None,
        bundler_count=None,
        sniper_count=None,
        insider_count=None,
        security_flags=None,
        wallet_flow_net=None,
        dex_paid=None,
        website=None,
        twitter=None,
        telegram=None,
        error=(
            "NAR_PLACEHOLDER: bot has no narrative for this token"
            if placeholder else None
        ),
    )


@dataclass
class NarQueryStats:
    sent: int = 0
    answered: int = 0
    empty: int = 0
    placeholder: int = 0
    timeouts: int = 0
    errors: int = 0
    queue_full: int = 0
    last_error: str | None = None


class PhanesNarSource:
    """
    Queries ``@Phanes_bot`` over the existing Telethon connection.

    Shares the alert listener's client: one session, one connection, no second
    login. The engine attaches it after the listener connects.
    """

    name = "nar:phanes_bot"

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        client: Any = None,
        bot_username: str = DEFAULT_BOT,
        command: str = COMMAND,
        #: Default equal to NarConfig.timeout_s. 12.0 was a latent fallback: a
        #: caller that forgot to pass the configured budget got 12s, not 2.5s.
        timeout_s: float = 2.5,
        min_gap_s: float = 4.0,
        max_queue: int = 8,
        collect_extra_s: float = 1.5,
        #: Fallback bot, used only after ``fallback_after_failures`` consecutive
        #: PRIMARY operational failures. ``None`` disables the fallback entirely.
        fallback_bot_username: str | None = DEFAULT_FALLBACK_BOT,
        fallback_after_failures: int = 3,
    ) -> None:
        self.clock = clock or SystemClock()
        self.client = client
        self.bot_username = bot_username
        self.command = command
        self.timeout_s = timeout_s
        self.min_gap_s = min_gap_s
        self.max_queue = max_queue
        self.collect_extra_s = collect_extra_s
        self.fallback_bot_username = fallback_bot_username or None
        self.fallback_after_failures = max(1, int(fallback_after_failures))
        #: Consecutive OPERATIONAL primary failures, and whether the fallback bot
        #: is currently the provider. A successful primary response resets both.
        self.consecutive_failures = 0
        self.using_fallback = False
        self.stats = NarQueryStats()
        #: One resolved entity per bot: the primary and the fallback do not share
        #: an entity, and resolving one must not evict the other.
        self._bot_entities: dict[str, Any] = {}
        self._gate = asyncio.Lock()
        self._last_sent = 0.0

    # -- availability ------------------------------------------------------

    @property
    def available(self) -> bool:
        return self.client is not None

    # -- querying ----------------------------------------------------------

    async def fetch_async(self, mint: str) -> NarSnapshot:
        if self.client is None:
            return NarSnapshot(
                mint=mint, fetched_at=self.clock.now(), source=None,
                available=False, error="NAR_UNAVAILABLE: telegram not connected",
            )
        # Bounded queue: if the bot is already behind, say so and move on rather
        # than building an unbounded backlog that delays every decision.
        if self._gate.locked():
            self.stats.queue_full += 1
            return NarSnapshot(
                mint=mint, fetched_at=self.clock.now(), source=self.name,
                available=False,
                error="NAR_UNAVAILABLE: narrative query already in flight",
            )
        try:
            async with self._gate:
                return await self._resolve(mint)
        except Exception as exc:  # noqa: BLE001
            self.stats.errors += 1
            self.stats.last_error = f"{type(exc).__name__}: {exc}"
            log("NAR", "phanes query failed for %s: %r", mint[:8], exc)
            return NarSnapshot(
                mint=mint, fetched_at=self.clock.now(), source=self.name,
                available=False, error=f"NAR_UNAVAILABLE: {exc!r}",
            )

    def fetch(self, mint: str) -> NarSnapshot:
        """
        Synchronous interface for the NarService contract.

        The engine awaits ``fetch_async`` directly; this exists so the source
        satisfies the same protocol as the others and can be exercised
        synchronously in tests. It refuses rather than blocking the event loop.
        """
        return NarSnapshot(
            mint=mint, fetched_at=self.clock.now(), source=self.name,
            available=False,
            error="NAR_UNAVAILABLE: use fetch_async from the event loop",
        )

    # -- provider selection + fallback state machine -----------------------

    def _is_operational_failure(self, snap: NarSnapshot) -> bool:
        """
        Count ONLY real operational failures.

        A response the provider actually delivered is never a failure, even when
        its content is unfavourable: a placeholder ("no narrative for this
        token") is a usable answer, not an outage. Only an absent/broken response
        - timeout, no reply, transport error, malformed body - advances the
        counter. Switching providers because the answer was negative would be
        switching on content, which is explicitly not allowed.
        """
        if snap.available:
            return False
        err = str(snap.error or "")
        if "PLACEHOLDER" in err:
            return False
        return True

    async def _resolve(self, mint: str) -> NarSnapshot:
        """
        Query the primary, then the fallback once the failure threshold is met.

        The primary is ALWAYS attempted first. That is deliberate: it is the only
        way a recovery can be detected, so the source cannot stay permanently on
        the fallback after the primary comes back. Each attempt is a single
        bounded query - the existing timeout/retry policy is reused and no extra
        retries are manufactured.
        """
        primary = await self._query_provider(
            self.bot_username, mint, label="phanes_bot"
        )
        if not self._is_operational_failure(primary):
            self.consecutive_failures = 0
            if self.using_fallback:
                self.using_fallback = False
                log("NAR", "NARRATIVE_SOURCE=PHENES reason=PHENES_RECOVERED")
            return primary

        self.consecutive_failures += 1
        if (
            self.fallback_bot_username
            and self.consecutive_failures >= self.fallback_after_failures
        ):
            if not self.using_fallback:
                self.using_fallback = True
                log(
                    "NAR",
                    "NARRATIVE_SOURCE=TOKENSCAN "
                    "reason=PHENES_%d_CONSECUTIVE_FAILURES",
                    self.consecutive_failures,
                )
            return await self._query_provider(
                self.fallback_bot_username, mint, label="tokenscan"
            )
        return primary

    async def _query_provider(
        self, bot_username: str, mint: str, *, label: str
    ) -> NarSnapshot:
        """One bounded query against one provider, attributed to ``label``."""
        try:
            snap = await self._query(bot_username, mint)
        except Exception as exc:  # noqa: BLE001 - a provider fault is a failure
            self.stats.errors += 1
            self.stats.last_error = f"{type(exc).__name__}: {exc}"
            snap = NarSnapshot(
                mint=mint, fetched_at=self.clock.now(), source=self.name,
                available=False, error=f"NAR_UNAVAILABLE: {exc!r}",
            )
        # Attribution: the snapshot names the provider that actually answered
        # this request, so it survives into the candidate's diagnostics.
        # ``NarSnapshot`` is frozen, so this is a replace, not a mutation: it
        # rebuilds the same value with only the source label changed.
        return replace(snap, source=label)

    async def _query(self, bot_username: str, mint: str) -> NarSnapshot:
        """
        The WHOLE query, inside the configured budget.

        The previous version applied ``self.timeout_s`` to the POLLING LOOP only.
        Everything before it - ``get_entity``, the anchor ``get_messages``, the
        polite ``min_gap_s`` sleep (4.0s by default) and ``send_message`` - sat
        outside the deadline entirely, so the real worst case was
        ``min_gap_s + n_rpc_calls + timeout_s`` and the "2.5s budget" was a
        floor, not a bound. The engine's ``asyncio.wait_for`` was the only thing
        that made it finite, and it could not attribute the overshoot.

        Now one deadline covers all of it, so ``timeout_s`` is a real bound: the
        source returns at or before it, and reports a timeout that says so.
        """
        try:
            async with asyncio.timeout(self.timeout_s):
                return await self._query_inner(
                    bot_username, mint, deadline=time.monotonic() + self.timeout_s
                )
        except TimeoutError:
            self.stats.timeouts += 1
            log("NAR", "phanes query exceeded the %.1fs budget for %s",
                self.timeout_s, mint[:8])
            return NarSnapshot(
                mint=mint, fetched_at=self.clock.now(), source=self.name,
                available=False,
                error=(
                    f"NAR_UNAVAILABLE: no bot reply within "
                    f"{self.timeout_s:.1f}s (narrative budget exceeded)"
                ),
            )

    async def _query_inner(
        self, bot_username: str, mint: str, *, deadline: float
    ) -> NarSnapshot:
        """
        One attempt, inside ``deadline``.

        Every wait below - the polite gap and the poll interval - is clamped to
        the time actually remaining, so the steps cannot queue up behind each
        other and overrun the budget they share.
        """
        now = self.clock.now()
        bot = self._bot_entities.get(bot_username)
        if bot is None:
            bot = await self.client.get_entity(bot_username)
            self._bot_entities[bot_username] = bot

        # keep a handle on the current newest message so we only read replies
        # that arrive after our command
        anchor = 0
        try:
            newest = await self.client.get_messages(bot, limit=1)
            if newest:
                anchor = max((m.id or 0) for m in newest)
        except Exception:  # noqa: BLE001
            anchor = 0

        # polite gap between queries, clamped to the budget that is LEFT
        since = self.clock.now() - self._last_sent
        if self._last_sent and since < self.min_gap_s:
            await asyncio.sleep(
                min(self.min_gap_s - since, max(0.0, deadline - time.monotonic()))
            )

        self._last_sent = self.clock.now()
        self.stats.sent += 1
        sent = await self.client.send_message(bot, f"{self.command} {mint}")
        sent_id = getattr(sent, "id", 0)

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # Cap the sleep to what is LEFT, rather than sleeping a whole
            # collect interval past the deadline. With collect_extra_s=1.5 and a
            # 2.5s budget the loop previously overshot to ~3.0s.
            await asyncio.sleep(min(self.collect_extra_s, remaining))
            try:
                messages = await self.client.get_messages(
                    bot, limit=6, min_id=anchor
                )
            except Exception as exc:  # noqa: BLE001
                self.stats.errors += 1
                return NarSnapshot(
                    mint=mint, fetched_at=self.clock.now(), source=self.name,
                    available=False, error=f"NAR_UNAVAILABLE: {exc!r}",
                )
            for message in messages or []:
                if getattr(message, "id", 0) == sent_id:
                    continue
                # prefer an explicit reply to our command
                if getattr(message, "reply_to_msg_id", None) != sent_id:
                    # also accept any new inbound message from the bot after our
                    # command: some bots answer without threading
                    if getattr(message, "out", False):
                        continue
                text = getattr(message, "message", None) or getattr(
                    message, "text", ""
                ) or ""
                if not text.strip():
                    continue
                self.stats.answered += 1
                snap = self.parse_any(text, mint)
                if snap.error and "PLACEHOLDER" in snap.error:
                    self.stats.placeholder += 1
                if not snap.available and not snap.narrative:
                    self.stats.empty += 1
                log(
                    "NAR",
                    "phanes %s %s (%d chars, %d specifics) in %.1fs",
                    "placeholder" if snap.error and "PLACEHOLDER" in snap.error
                    else "narrative",
                    mint[:8],
                    len(snap.narrative or ""),
                    len(set(m.group(0).lower() for m in
                            _SPECIFIC_RE.finditer(snap.narrative or ""))),
                    self.clock.now() - self._last_sent,
                )
                return snap

        self.stats.timeouts += 1
        return NarSnapshot(
            mint=mint, fetched_at=self.clock.now(), source=self.name,
            available=False,
            error=f"NAR_UNAVAILABLE: no bot reply within {self.timeout_s:.0f}s",
        )

    @staticmethod
    def parse_any(text: str, mint: str, now: float | None = None) -> NarSnapshot:
        """
        The bot answers in two shapes: a stats card for a token it recognises,
        and prose for anything else (USDC, for example). Try the card first, fall
        back to the narrative parser, and treat an unrecognised shape as
        unavailable rather than guessing.
        """
        from ..clock import SystemClock

        now = SystemClock().now() if now is None else now
        if _CARD_HEAD_RE.search((text or "").splitlines()[0] if text else ""):
            card = parse_stats_card(text, mint, now)
            if card.available:
                return card
            # a card that failed its own consistency check is not a narrative
            return card
        return parse_nar_response(text, mint, now)

    def health(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bot": self.bot_username,
            "fallback_bot": self.fallback_bot_username,
            "narrative_source": (
                "tokenscan" if self.using_fallback else "phanes_bot"
            ),
            "consecutive_failures": self.consecutive_failures,
            "available": self.available,
            **vars(self.stats),
        }
