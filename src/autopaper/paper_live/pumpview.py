"""
PumpView (pumpview.fun) *Amarih* signal adapter -- read-only ingestion.

WHAT THIS IS
============
A monitoring adapter that turns genuine PumpView **Amarih** signal events into
normalised :class:`AmarihSignal` records that the existing PAPER entry pipeline
can consume. It is a *source*, nothing more: it re-derives no strategy rule,
issues no quote, opens no position and submits no transaction.

WHY THIS SOURCE
===============
Amarih is a **community (user-defined) strategy** on PumpView, not an admin
strategy and not a label invented here. Verified by direct observation of the
public ``/api/signal-history`` and ``/api/strategy-detail`` endpoints, which the
pumpview.fun web app itself calls on page load with *optional* auth:

    * each event carries ``signal_type == "Amarih"`` -- so Amarih events are
      distinguishable from every other strategy in the same stream;
    * each event carries a stable ``id`` (signal id), ``ca`` (token mint),
      ``fired_at_ts`` (signal timestamp) and ``mc_signal`` (entry market cap);
    * the same ``/api/strategy-detail?name=Amarih`` endpoint returns the 24h
      outcome of every fired signal, **including losers** (``cur_pct < 0``) and
      zero-ATH signals, so there is no survivor bias in the record.

See ``reports/research/amarih_pumpview_integration.md`` for the full evidence,
confidence classification and the decision.

MONITORING ROUTE (chosen)
=========================
**Public REST poll of ``GET /api/signal-history?limit=N&offset=M``** -- the
exact request the site's own front-end issues. It needs **no authentication**
(the site sends a Bearer token only when one is present, and the read still
succeeds anonymously), imposes no observed rate-limit headers, and returns the
full per-signal payload. It is polled here with :mod:`httpx`, which is already a
declared dependency -- so no new dependency is introduced.

A WebSocket (``wss://pumpview.fun/ws``) also pushes ``sig_hist`` /
``sig_hist_delta`` / ``signal_alert`` frames carrying the same fields, but the
project declares **no** WebSocket client dependency, so wiring that live-push
route would mean adding one. That is a deliberate *not-done*: this adapter does
not silently substitute the poll for a live push -- it uses the poll, which is
what is verified reliable with the dependencies actually present.

WHAT IT WILL NOT DO
===================
* It will not fabricate a market cap, a score or an outcome. Missing fields
  stay ``None``.
* It will not mint a fill from a mark. The entry market cap it preserves is the
  strategy's own ``mc_signal`` -- an *observation* for the record, never a fill.
* It will not bypass auth, rate limits or access controls, and it never carries
  a credential. ``token`` is an optional caller-supplied Bearer for an operator
  who has their own PumpView session; the adapter never reads, stores or logs
  one beyond passing it straight through on the request.

DEDUPLICATION / STALENESS
=========================
* A signal ``id`` is processed at most once per process (``seen_ids``), so the
  same event arriving in two consecutive polls, twice within one batch, or out
  of order after a reconnect, produces exactly one :class:`AmarihSignal`.
* A signal older than ``max_age_s`` at receipt is **rejected** with reason
  ``STALE`` and recorded, never traded -- a late poll of a long-gone signal must
  not become an entry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

import httpx

#: The PumpView base. Overridable for a test double; never a credential.
DEFAULT_BASE_URL = "https://pumpview.fun"

#: The strategy this adapter is scoped to. A signal whose ``signal_type`` is not
#: exactly this is ignored -- Amarih events must not be conflated with another
#: strategy that happens to appear in the same stream.
STRATEGY_NAME = "Amarih"

#: Default maximum signal age, in seconds. A signal fired more than this long
#: before we *receive* it is stale and refused. Amarih is a fast, high-frequency
#: momentum strategy (tens of thousands of signals over ~3 weeks), so the
#: default is tight; the operator can widen it, but a wide window on a strategy
#: whose edge decays in seconds manufactures bad fills.
#:
#: 15.0 SECONDS, NOT 120.0. This default was 120.0 s in both this constant and
#: the engine's constructor fallback, and every runtime that did not set
#: ``PUMPVIEW_AMARIH_MAX_S`` therefore judged staleness against a 120 s window
#: while the strategy's own design -- and the operator's stated intent -- is a
#: 15 s window. The effect was not partial: a signal 29 s old is 29 s old
#: against a 15 s gate and 29 s fresh against a 120 s one, so every signal in
#: the 15 s .. 120 s band was accepted, and 118 s-old calls traded as if they
#: had just fired. Age at entry is the single largest determinant of an entry
#: fill on a strategy whose edge decays in seconds, so this silently moved the
#: whole entry population.
DEFAULT_MAX_AGE_S = 15.0

#: How far AHEAD of the poll clock a source timestamp may sit, in seconds.
#:
#: ``fired_at_ts`` is produced by PumpView's own clock, which is not this
#: process's clock: a genuine clock skew between the two would otherwise make a
#: freshly fired signal look like it arrived from the future, and rejecting it
#: for that reason would be a false negative with no market cause. Tolerance
#: here is deliberately tiny compared with the freshness window -- 2 s out of a
#: 15 s gate -- so it can only absorb skew and can never excuse a stale signal.
FUTURE_TOLERANCE_S = 2.0

#: Reasons an incoming event is not turned into a tradeable signal. Recorded
#: verbatim so a refused signal is auditable, exactly like an accepted one.
REJECT_STALE = "STALE"
REJECT_NO_MINT = "NO_MINT"
REJECT_DUPLICATE = "DUPLICATE"
REJECT_WRONG_STRATEGY = "WRONG_STRATEGY"
#: A missing, non-numeric or otherwise uninterpretable ``fired_at_ts``. Separate
#: from ``NO_MINT`` because it is a different defect with a different fix, and
#: because the two were previously collapsed into one reason string -- which is
#: how "no mint" was the recorded cause for a signal that had a perfectly good
#: mint and no usable timestamp.
REJECT_NO_TIMESTAMP = "NO_TIMESTAMP"
#: A ``fired_at_ts`` more than ``FUTURE_TOLERANCE_S`` ahead of the poll clock.
#: A signal from the future cannot be aged honestly, so it is refused rather
#: than assigned a negative age and judged fresh.
REJECT_FUTURE_TIMESTAMP = "FUTURE_TIMESTAMP"


@dataclass(frozen=True)
class AmarihSignal:
    """One genuine, de-duplicated, non-stale Amarih signal event.

    Every field is preserved from the source with its provenance. ``None``
    means the source did not supply it -- nothing is defaulted or invented.
    """

    #: PumpView signal id. The stable identity used for de-duplication.
    signal_id: str
    #: The strategy name, preserved verbatim (``"Amarih"``).
    strategy: str
    #: Token mint (``ca``). Validated base58 upstream; ``None`` if absent.
    mint: Optional[str]
    symbol: Optional[str]
    #: Unix seconds the strategy fired the signal (``fired_at_ts``), NORMALISED to
    #: seconds. See :func:`_normalise_source_ts` for the accepted shapes.
    fired_at: float
    #: Unix seconds we received it (the poll wall-clock at parse time).
    received_at: float
    #: The strategy's entry market cap in USD (``mc_signal``). An OBSERVATION.
    entry_market_cap_usd: Optional[float]
    #: Buy Score at fire time, when supplied.
    buy_score: Optional[float]
    #: Price at fire time in USD, when supplied.
    price_signal_usd: Optional[float]
    #: The owner wallet that defined the strategy (provenance only).
    owner_wallet: Optional[str]
    #: The full raw event, preserved for audit.
    raw: Mapping[str, Any] = field(default_factory=dict)
    #: ``received_at - fired_at`` in ms. The AUTHORITY for the freshness
    #: decision: computed from the SOURCE timestamp, never from receipt or poll
    #: time. Recorded so a wrongly-accepted signal leaves a trace of what its
    #: age actually was.
    signal_age_ms: Optional[float] = None

    @property
    def signal_to_receipt_latency_ms(self) -> float:
        """Signal fire -> our receipt, in ms. A positive number = we were late."""
        return round((self.received_at - self.fired_at) * 1000.0, 2)

    def as_dict(self) -> dict[str, Any]:
        # THE FRESHNESS DIAGNOSTICS, in full.
        #
        # These are the fields the operator needs to tell a genuinely stale
        # signal from a clock problem from a unit mismatch, and none of them was
        # previously recorded: the age was computed, compared and thrown away, so
        # a wrongly-accepted signal left no trace of what its age actually was.
        # No secret and no full provider payload is included.
        return {
            "signal_id": self.signal_id,
            "strategy": self.strategy,
            "mint": self.mint,
            "symbol": self.symbol,
            "fired_at": self.fired_at,
            "received_at": self.received_at,
            "signal_to_receipt_latency_ms": self.signal_to_receipt_latency_ms,
            "calculated_age_ms": self.signal_age_ms,
            "entry_market_cap_usd": self.entry_market_cap_usd,
            "buy_score": self.buy_score,
            "price_signal_usd": self.price_signal_usd,
            "owner_wallet": self.owner_wallet,
        }


@dataclass
class IngestReport:
    """What one poll produced: accepted signals and every refusal, in order."""

    accepted: list[AmarihSignal] = field(default_factory=list)
    #: ``(signal_id_or_raw_id, reason)`` for each event not turned into a signal.
    refused: list[tuple[str, str]] = field(default_factory=list)
    #: Total raw events examined (before strategy filter).
    events_seen: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "events_seen": self.events_seen,
            "accepted": [s.as_dict() for s in self.accepted],
            "refused": [{"id": i, "reason": r} for i, r in self.refused],
        }


def _num(value: Any) -> Optional[float]:
    """Coerce to a finite float or ``None``. Never returns NaN/inf."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


#: A numeric field is MILLISECONDS, not seconds, above this magnitude.
#:
#: ``fired_at_ts`` is Unix seconds in PumpView's current API, but a provider
#: (or a proxy, or a field rename) that emits milliseconds is indistinguishable
#: from a seconds value by type alone -- both are floats. Reading milliseconds as
#: seconds gives an age of roughly ``now - 1.79e12`` = a large NEGATIVE number,
#: which then passes every ``age > max_age`` test in the codebase. Nothing looks
#: wrong: the age is finite, the signal is "fresh", and the entry is priced
#: against a token that fired hours ago.
#:
#: 1e11 is chosen deliberately: it is far above any seconds-POSIX-timestamp that
#: will exist in this century's operation (2e9 is year 2033) and far below any
#: plausible milliseconds value for the same window (2e12). A value below 1e11
#: is therefore seconds, above it milliseconds.
_MS_THRESHOLD = 1e11


def _normalise_source_ts(value: Any) -> Optional[float]:
    """
    Interpret PumpView's ``fired_at_ts`` as Unix SECONDS, or ``None``.

    Handles the three representations the field can plausibly take and refuses
    anything else, because the age calculation is only meaningful against a
    timestamp whose unit is known:

    * **Unix seconds, int or float** -- the current API shape
      (``1791975031.198767``). Used as-is.
    * **Unix milliseconds** -- detected by magnitude (see ``_MS_THRESHOLD``) and
      divided by 1000. A seconds-shaped value is never divided.
    * **ISO-8601 string, with or without a ``Z`` / ``+07:00`` offset** -- parsed
      as an aware datetime and converted to POSIX seconds, so a UTC offset is
      normalised rather than silently treated as local time. A string with a
      trailing ``Z`` is UTC; one with an explicit offset is honoured; one with
      neither is read as UTC, which is the convention PumpView documents.

    Returns ``None`` for anything missing, non-numeric, non-parseable, or out of
    finite range. The caller refuses the signal rather than guessing, and the
    recorded age stays ``None`` -- never ``0``, which would read as "just
    fired".
    """
    if value is None or isinstance(value, bool):
        return None

    # Numeric: seconds, or milliseconds by magnitude.
    if isinstance(value, (int, float)):
        seconds = _num(value)
        if seconds is None:
            return None
        if abs(seconds) >= _MS_THRESHOLD:
            seconds = seconds / 1000.0
        return seconds if _num(seconds) is not None else None

    text = str(value).strip()
    if not text:
        return None

    # Numeric STRING ("1791975031.198767" / "1791975031198767"). Handled before
    # ISO parsing because it is the common case and is cheaper to normalise.
    try:
        seconds = float(text)
    except ValueError:
        pass
    else:
        if seconds == seconds and abs(seconds) != float("inf"):
            if abs(seconds) >= _MS_THRESHOLD:
                seconds = seconds / 1000.0
            return seconds if _num(seconds) is not None else None

    # ISO-8601, with or without a fractional part and an offset.
    iso = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # No offset means UTC in this feed's convention. Saying so explicitly
        # matters: a naive datetime is LOCAL time in Python, and treating a UTC
        # timestamp as local would shift the age by the operator's UTC offset
        # (here, +7h) -- which is a 25,200-second error that reads as a very
        # stale signal and refuses a live one.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return float(parsed.timestamp())


class PumpViewAmarihSource:
    """Read-only poller/normaliser for the PumpView Amarih strategy.

    Deliberately transport-only: :meth:`poll_once` performs the HTTP read and
    :meth:`ingest_events` normalises an already-fetched payload. Keeping the
    normaliser free of I/O makes it unit-testable offline against a fixture and
    is the same code path the live poll uses.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        token: Optional[str] = None,
        max_age_s: float = DEFAULT_MAX_AGE_S,
        strategy: str = STRATEGY_NAME,
        clock: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_age_s = float(max_age_s)
        self.strategy = strategy
        #: Injectable clock (tests pass a fake); ``time.time`` in production.
        self._now = clock.now if clock is not None else time.time
        #: Signal ids already emitted. Process-local de-duplication; the caller
        #: persists accepted ids across restarts if it wants cross-restart dedup.
        self.seen_ids: set[str] = set()
        #: Optional operator Bearer token. Passed straight through on the
        #: request; never stored beyond this attribute and never logged.
        self._token = token

    # -- transport --------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def poll_once(
        self,
        *,
        limit: int = 200,
        offset: int = 0,
        timeout_s: float = 30.0,
        client: Optional[httpx.Client] = None,
    ) -> IngestReport:
        """Fetch one page of signal history and normalise the Amarih events.

        Uses ``httpx`` (a declared dependency). No auth is required for the
        public read; a caller-supplied token is only attached when present.
        """
        url = f"{self.base_url}/api/signal-history"
        params = {"limit": int(limit), "offset": int(offset)}

        def _do(c: httpx.Client) -> dict[str, Any]:
            resp = c.get(url, params=params, headers=self._headers(),
                         timeout=timeout_s)
            resp.raise_for_status()
            return resp.json()

        if client is not None:
            payload = _do(client)
        else:
            with httpx.Client(timeout=timeout_s) as c:
                payload = _do(c)
        return self.ingest_events(payload)


    # -- normalisation (pure) --------------------------------------------

    def ingest_events(
        self, payload: Mapping[str, Any], *, now: Optional[float] = None
    ) -> IngestReport:
        """Turn a raw ``/api/signal-history`` payload into an ingest report.

        Pure: no I/O, no clock unless ``now`` is omitted. Handles duplicate and
        out-of-order events, wrong-strategy events, missing mints and stale
        events. The same method is what the tests exercise against the fixture.
        """
        now = self._now() if now is None else float(now)
        report = IngestReport()
        events = payload.get("signals") if isinstance(payload, Mapping) else None
        if not isinstance(events, Sequence):
            return report

        for ev in events:
            if not isinstance(ev, Mapping):
                continue
            report.events_seen += 1

            # Only Amarih. Another strategy in the same stream is not ours.
            if str(ev.get("signal_type") or "") != self.strategy:
                continue

            raw_id = ev.get("id")
            signal_id = str(raw_id) if raw_id is not None else ""

            # De-duplicate: the same id may recur across polls or within a
            # batch, and may arrive out of order after a reconnect. One id in,
            # one signal out.
            if signal_id and signal_id in self.seen_ids:
                report.refused.append((signal_id or "?", REJECT_DUPLICATE))
                continue

            mint = ev.get("ca")
            if not (isinstance(mint, str) and mint.strip()):
                # No mint means no tradeable identity. Recorded, never traded.
                if signal_id:
                    self.seen_ids.add(signal_id)
                report.refused.append((signal_id or "?", REJECT_NO_MINT))
                continue

            # THE SOURCE TIMESTAMP IS NORMALISED, THEN VALIDATED, THEN AGED.
            #
            # Order matters, and it is the order the age calculation is only
            # meaningful in: interpret the field first (so the unit is known),
            # then reject what cannot be aged or is from the future (so a bogus
            # value cannot become a small or negative age), and only then compare
            # against the freshness window.
            #
            # MISSING OR MALFORMED IS REFUSED, NOT DEFAULTED. Falling back to
            # receipt time would age every timestamp-less signal as 0 ms -- fresh
            # by construction -- and turn "we could not read the clock" into "the
            # signal just fired". That is the one substitution this adapter must
            # never make, so the signal is refused and stays in the denominator.
            fired_at = _normalise_source_ts(ev.get("fired_at_ts"))
            if fired_at is None:
                if signal_id:
                    self.seen_ids.add(signal_id)
                report.refused.append((signal_id or "?", REJECT_NO_TIMESTAMP))
                continue

            age = now - fired_at
            if age < -FUTURE_TOLERANCE_S:
                # From the future. Cannot be aged honestly, and the tolerance
                # above only absorbs clock skew between PumpView and this process.
                if signal_id:
                    self.seen_ids.add(signal_id)
                report.refused.append((signal_id or "?", REJECT_FUTURE_TIMESTAMP))
                continue

            age_ms = round(max(0.0, age) * 1000.0, 3)

            if age > self.max_age_s:
                # Stale: a long-gone signal must not become an entry. Mark seen
                # so a later poll does not re-surface it, and record the refusal.
                if signal_id:
                    self.seen_ids.add(signal_id)
                report.refused.append((signal_id or "?", REJECT_STALE))
                continue

            if signal_id:
                self.seen_ids.add(signal_id)

            report.accepted.append(
                AmarihSignal(
                    signal_id=signal_id or f"pv_{int(fired_at * 1000)}",
                    strategy=str(ev.get("signal_type")),
                    mint=mint.strip(),
                    symbol=(str(ev["sym"]) if ev.get("sym") else None),
                    fired_at=fired_at,
                    received_at=now,
                    signal_age_ms=age_ms,
                    entry_market_cap_usd=_num(ev.get("mc_signal")),
                    buy_score=_num(ev.get("buy_score")),
                    price_signal_usd=_num(ev.get("price_signal")),
                    owner_wallet=(str(ev["owner_wallet"])
                                  if ev.get("owner_wallet") else None),
                    raw=dict(ev),
                )
            )

        return report


__all__ = [
    "AmarihSignal",
    "IngestReport",
    "PumpViewAmarihSource",
    "STRATEGY_NAME",
    "DEFAULT_MAX_AGE_S",
    "DEFAULT_BASE_URL",
    "REJECT_STALE",
    "REJECT_NO_MINT",
    "REJECT_NO_TIMESTAMP",
    "REJECT_FUTURE_TIMESTAMP",
    "REJECT_DUPLICATE",
    "REJECT_WRONG_STRATEGY",
    "FUTURE_TOLERANCE_S",
]

