"""
Price freshness: the timestamp of every price an entry touches, and its age.

WHY THIS EXISTS
---------------
The live forensics found that on the one fully-measured REAL round trip the
decision-time price read was ~2.07 s old and the market moved ~+30% between the
decision and the quote ~0.9 s later -- while the freshness gate was set to 15 s.
The move that cost money happened BEFORE the quote existed, so a gate that only
bounds the quote's age cannot see it.

THE FOUR TIMESTAMPS
-------------------
For every entry the runtime records:

    signal timestamp     when the price that triggered the candidate was observed
    decision timestamp   when L25 authorised the entry
    quote timestamp      when the venue quote was taken
    execution timestamp  when the fill happened

and derives:

    price_age_at_decision   decision - signal
    price_age_at_quote      quote - signal
    price_change_before_fill  (fill - signal) / signal, when prices are known

A missing timestamp is left ``None`` -- never zero. A zero age would read as
"perfectly fresh", which is the opposite of "unknown".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


def _ms(later: Optional[float], earlier: Optional[float]) -> Optional[float]:
    if later is None or earlier is None:
        return None
    return max(0.0, (float(later) - float(earlier)) * 1000.0)


def _pct(after: Optional[float], before: Optional[float]) -> Optional[float]:
    if after is None or before is None or before == 0:
        return None
    return (float(after) / float(before) - 1.0) * 100.0


@dataclass(frozen=True)
class PriceTrace:
    """One entry's price provenance, and the ages it implies."""

    signal_at: Optional[float] = None
    decision_at: Optional[float] = None
    quote_at: Optional[float] = None
    execution_at: Optional[float] = None
    signal_price: Optional[float] = None
    decision_price: Optional[float] = None
    quote_price: Optional[float] = None
    fill_price: Optional[float] = None

    # -- derived ages ------------------------------------------------------

    @property
    def price_age_at_decision_ms(self) -> Optional[float]:
        return _ms(self.decision_at, self.signal_at)

    @property
    def price_age_at_quote_ms(self) -> Optional[float]:
        return _ms(self.quote_at, self.signal_at)

    @property
    def decision_to_quote_ms(self) -> Optional[float]:
        return _ms(self.quote_at, self.decision_at)

    @property
    def quote_to_fill_ms(self) -> Optional[float]:
        return _ms(self.execution_at, self.quote_at)

    @property
    def price_change_before_fill_pct(self) -> Optional[float]:
        """Signal -> fill, when both prices are known. Stated, never inferred."""
        return _pct(self.fill_price, self.signal_price)

    @property
    def analysis_to_quote_pct(self) -> Optional[float]:
        return _pct(self.quote_price, self.decision_price)

    def is_stale(self, budget_ms: float) -> Optional[bool]:
        """
        Whether the decision-time price age exceeds ``budget_ms``.

        ``None`` when the age is unknown -- an unknown age is not "fresh".
        """
        age = self.price_age_at_decision_ms
        if age is None:
            return None
        return age > budget_ms

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal_at": self.signal_at,
            "decision_at": self.decision_at,
            "quote_at": self.quote_at,
            "execution_at": self.execution_at,
            "signal_price": self.signal_price,
            "decision_price": self.decision_price,
            "quote_price": self.quote_price,
            "fill_price": self.fill_price,
            "price_age_at_decision_ms": self.price_age_at_decision_ms,
            "price_age_at_quote_ms": self.price_age_at_quote_ms,
            "decision_to_quote_ms": self.decision_to_quote_ms,
            "quote_to_fill_ms": self.quote_to_fill_ms,
            "price_change_before_fill_pct": self.price_change_before_fill_pct,
            "analysis_to_quote_pct": self.analysis_to_quote_pct,
        }

    @classmethod
    def from_stamps(
        cls,
        stamps: Mapping[str, float],
        *,
        signal_price: Optional[float] = None,
        decision_price: Optional[float] = None,
        quote_price: Optional[float] = None,
        fill_price: Optional[float] = None,
    ) -> "PriceTrace":
        """
        Build from a :class:`~autopaper.latency.StageRecorder` stamp mapping.

        ``quote_at`` prefers the moment the quote was REQUESTED (``quote_start``)
        so the age measures the price the quote was built from, falling back to
        ``quote_end``. ``execution_at`` is the ``fill`` stamp.
        """
        s = stamps or {}
        return cls(
            signal_at=s.get("alert_received") or s.get("market_start"),
            decision_at=s.get("l25_end") or s.get("risk_end"),
            quote_at=s.get("quote_start") or s.get("quote_end"),
            execution_at=s.get("fill") or s.get("submit_end"),
            signal_price=signal_price,
            decision_price=decision_price,
            quote_price=quote_price,
            fill_price=fill_price,
        )
