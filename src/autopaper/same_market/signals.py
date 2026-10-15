"""
Token extraction from a Telegram signal, with ambiguity treated as a result.

THE EXPLICIT ``CA:`` FIELD IS AUTHORITATIVE
============================================

A signal that carries a labelled contract-address field (``CA:``, ``mint``,
``contract``, ``token address``…) has said which token it is about. That field
wins outright.

This matters in both directions. Without it, a message whose promo footer
happens to quote a second real mint is declared AMBIGUOUS and the signal is
**silently dropped** - a real, correctly-formatted signal lost to an incidental
mention in a footer. And a harness that let a footer override the labelled field
would be measuring whichever token happened to appear twice, which is precisely
the guess this module exists to refuse.

So the rules are:

* a labelled field yielding exactly one valid mint is THE mint. Other base58
  strings in the message are recorded as a note, and do not create ambiguity.
* a labelled field yielding several distinct mints is still AMBIGUOUS. The field
  itself does not disambiguate, so nothing can.
* with no labelled field, fall back to collecting every base58 string; more than
  one distinct is AMBIGUOUS.

AMBIGUITY IS STILL A FIRST-CLASS OUTCOME
=========================================

The production parser (:mod:`autopaper.telegram.parser`) resolves a multi-mint
message by taking the most frequent base58 string. That is the right call for a
live bot, which must act on something. It is the wrong call for this harness: an
action taken against the wrong token produces a price comparison that describes
the wrong asset while looking entirely plausible. So
:func:`extract_token` reports ambiguity as a result and the pipeline stops.

* **Structural** - two or more *distinct* base58 strings where nothing identifies
  which one is meant.
* **Repeated** - the same mint appearing several times is NOT ambiguous. One
  token mentioned three times is still one token.

A ticker is never a mint candidate. ``$AI`` is a symbol; the address is the
address.

Venue hints are read but never acted on blindly. A signal saying "DEX: Moonshot"
is a hint that must be corroborated against the pair, not a selection rule.
"""

from __future__ import annotations

import re
from typing import Iterable

from ..telegram.parser import (
    MINT_RE,
    TICKER_RE,
    _extract_kv,
    _label_to_key,
    normalise,
)
from .models import TokenCandidate

#: Solana pubkeys are 32 bytes; base58 of 32 bytes is 32..44 characters. The
#: shared parser's regex is reused so both agree on what "a mint" means.
MIN_MINT_LEN = 32
MAX_MINT_LEN = 44

#: Trailing markers that identify the launchpad from the mint itself. Pump.fun
#: mints end in ``pump``; the others are launchpads seen on this channel.
LAUNCHPAD_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("pump", "pumpfun"),
    ("bonk", "letsbonk"),
    ("moonit", "moonit"),
    ("boop", "boop"),
    ("believe", "believe"),
    ("letsbonkfun", "letsbonk"),
)

#: Channel formatting that names a venue, e.g. "DEX: PumpFun".
_VENUE_RE = re.compile(
    r"\bDEX\b\s*[:：]\s*([A-Za-z0-9_./+-]{2,32})", re.IGNORECASE
)

#: Words that follow "DEX:" in this channel's format but are NOT venues. The
#: promo line is "DEX:  Paid❌ Ads❌ 0⚡", where "Paid" and "Ads" describe the
#: channel's own subscription tiers. Reading one as a venue would produce a
#: confident, wrong venue hint and - because an unmatched hint is a reported
#: VENUE_MISMATCH - would then fail the identity check for the wrong reason.
_NON_VENUE_HINTS: frozenset[str] = frozenset({
    "paid", "ads", "ad", "free", "premium", "none", "yes", "no", "0", "1",
    "true", "false", "on", "off", "x", "ok", "n/a",
})


def _venue_hint(text: str) -> str | None:
    """
    Read a venue name from the message, or ``None``.

    Returns ``None`` rather than a guess when the token after "DEX:" is one of
    the channel's subscription markers. A wrong hint is worse than no hint:
    selection treats an unmatched hint as a genuine disagreement with the
    venue, so a misparse would manufacture a VENUE_MISMATCH.
    """
    match = _VENUE_RE.search(text)
    if not match:
        return None
    raw = match.group(1).strip()
    if not raw or raw.lower() in _NON_VENUE_HINTS:
        return None
    return raw


def looks_like_mint(value: str) -> bool:
    """Is this string shaped like a Solana mint?"""
    if not value:
        return False
    text = value.strip()
    if not (MIN_MINT_LEN <= len(text) <= MAX_MINT_LEN):
        return False
    return MINT_RE.fullmatch(text) is not None


def launchpad_for_mint(mint: str | None) -> str | None:
    """
    Launchpad implied by the mint's own suffix, or ``None``.

    This is a hint, not a proof. It is recorded and then corroborated against
    chain state; nothing downstream treats it as authoritative, because a mint
    ending in ``pump`` says where the token was created and not where it trades
    now.
    """
    if not mint:
        return None
    low = mint.lower()
    for suffix, venue in LAUNCHPAD_SUFFIXES:
        if low.endswith(suffix) and len(low) > len(suffix):
            return venue
    return None


def _symbols(text: str) -> tuple[str | None, ...]:
    return tuple(t for t in TICKER_RE.findall(text) if re.search(r"[A-Za-z]", t))


def extract_token(raw_text: str) -> TokenCandidate:
    """
    Extract the token mint from one signal message.

    Returns a :class:`TokenCandidate` in every case, including the failure
    cases, so the caller always has a structured reason rather than a ``None``
    it has to interpret. The three outcomes are:

    * exactly one distinct mint -> ``ambiguous=False``, ``mint`` set
    * several distinct mints -> ``ambiguous=True``, ``mint=None``, all of them
      listed in ``candidates`` so the report can show what was seen
    * no mint at all -> ``ambiguous=False``, ``mint=None``, ``TOKEN_NOT_FOUND``
      signalled by a warning
    """
    if not raw_text or not raw_text.strip():
        return TokenCandidate(
            mint=None,
            symbol=None,
            ambiguous=False,
            candidates=(),
            venue_hint=None,
            warnings=("message is empty",),
        )

    text = normalise(raw_text)

    # Collect every base58-looking token, preserving position so a mint in the
    # body and a mint in a URL are both seen. Duplicates are collapsed, but
    # only AFTER collection - de-duplicating first would make a repeated mint
    # indistinguishable from a single one, which is the opposite of what the
    # ambiguity check needs.
    found: list[str] = []
    for match in MINT_RE.finditer(text):
        token = match.group(0)
        if looks_like_mint(token):
            found.append(token)

    distinct = tuple(dict.fromkeys(found))
    venue_hint = _venue_hint(text)
    symbols = _symbols(text)
    symbol = symbols[0] if symbols else None

    warnings: list[str] = []

    if not distinct:
        return TokenCandidate(
            mint=None,
            symbol=symbol,
            ambiguous=False,
            candidates=(),
            venue_hint=venue_hint,
            warnings=("no base58 mint found in the message",),
        )

    labelled = _labelled_mints(text)
    labelled_mints = tuple(dict.fromkeys(
        m for _label, mints in labelled for m in mints
    ))

    mint: str | None = None
    source_label: str | None = None
    if len(labelled_mints) == 1:
        mint = labelled_mints[0]
        source_label = labelled[0][0]

    # A labelled field that itself names several mints disambiguates nothing.
    if labelled_mints and len(labelled_mints) > 1:
        masked = ", ".join(f"{m[:6]}…{m[-4:]}" for m in labelled_mints[:6])
        warnings.append(
            f"the labelled contract-address field names {len(labelled_mints)} "
            f"distinct mints ({masked}); the field does not disambiguate, so "
            "nothing can"
        )
        return TokenCandidate(
            mint=None,
            symbol=symbol,
            ambiguous=True,
            candidates=tuple(dict.fromkeys((*labelled_mints, *distinct))),
            venue_hint=venue_hint,
            warnings=tuple(warnings),
        )

    if mint is None and len(distinct) > 1:
        # No labelled field, and the message names two tokens. It does not say
        # which one is meant, and picking one would make every downstream
        # number confidently wrong.
        masked = ", ".join(f"{m[:6]}…{m[-4:]}" for m in distinct[:6])
        warnings.append(
            f"message contains {len(distinct)} distinct mints ({masked}) and "
            "names no contract-address field; refusing to guess which token the "
            "signal refers to"
        )
        return TokenCandidate(
            mint=None,
            symbol=symbol,
            ambiguous=True,
            candidates=distinct,
            venue_hint=venue_hint,
            warnings=tuple(warnings),
        )

    if mint is None:
        # Sole distinct base58 string, and no labelled field to point at it.
        mint = distinct[0]

    if source_label is not None:
        # The labelled field decided it. Always record that, so the report
        # shows the PROVENANCE of the decision rather than only its outcome.
        # Anything else address-shaped in the message is incidental - a footer,
        # a related link, a comparison - and is named explicitly so a reader
        # can see it was considered and set aside, not overlooked.
        others = [m for m in distinct if m != mint]
        if others:
            masked = ", ".join(f"{m[:6]}…{m[-4:]}" for m in others[:6])
            warnings.append(
                f"took the mint from the explicit '{source_label}' field; "
                f"ignored {len(others)} other address-shaped string(s) in the "
                f"message ({masked}), which the label outranks"
            )
        else:
            warnings.append(
                f"took the mint from the explicit '{source_label}' field, which "
                "is the only address-shaped string in the message"
            )
    elif len(found) > 1:
        # Same mint, several mentions, no label. Still one token.
        warnings.append(
            f"mint appears {len(found)} times in the message; a single distinct "
            "token, so not ambiguous"
        )

    if not launchpad_for_mint(mint) and not venue_hint:
        warnings.append(
            "mint carries no recognised launchpad suffix and the message names "
            "no venue; venue must be established from chain state"
        )

    return TokenCandidate(
        mint=mint,
        symbol=symbol,
        ambiguous=False,
        candidates=distinct,
        venue_hint=venue_hint,
        warnings=tuple(warnings),
    )


def _labelled_mints(text: str) -> list[tuple[str, tuple[str, ...]]]:
    """
    Mints taken from an explicit contract-address label, as
    ``(label, mints_in_that_value)``.

    Reuses the production parser's own label table and key/value extractor, so
    there is exactly one definition of which words mean "contract address"
    (``CA``, ``mint``, ``contract``, ``token address``, …) in this repository.

    Every mint in a labelled value is collected, not just the first. The
    production parser takes ``_first_mint``, which is right for a bot that must
    act but wrong here: a field reading ``CA: X or Y`` names two tokens and
    resolving it to X is a guess presented as a fact.
    """
    pairs: list[tuple[str, tuple[str, ...]]] = []
    for label, value in _extract_kv(text):
        if _label_to_key(label) != "mint":
            continue
        mints = tuple(dict.fromkeys(
            m.group(0) for m in MINT_RE.finditer(value)
            if looks_like_mint(m.group(0))
        ))
        if mints:
            pairs.append((label.strip(), mints))
    return pairs


def multiple_mints(text: str) -> tuple[str, ...]:
    """Every distinct mint in the text. Convenience for assertions."""
    return tuple(dict.fromkeys(
        m.group(0) for m in MINT_RE.finditer(normalise(text or ""))
        if looks_like_mint(m.group(0))
    ))


def summarise_mints(mints: Iterable[str], *, head: int = 6, tail: int = 4) -> str:
    """Masked listing for a human-facing report line."""
    parts = [f"{m[:head]}…{m[-tail:]}" if len(m) > head + tail else m for m in mints]
    return ", ".join(parts)


__all__ = [
    "LAUNCHPAD_SUFFIXES",
    "MAX_MINT_LEN",
    "MIN_MINT_LEN",
    "extract_token",
    "launchpad_for_mint",
    "looks_like_mint",
    "multiple_mints",
    "summarise_mints",
]