"""
Authoritative REAL equity: native SOL balance x SOL/USD, or an explicit refusal.

WHY THIS EXISTS
---------------
The armed-REAL message used to render::

    Wallet: —…
    Equity: $0.0000
    Next Position: —

and none of that was a measurement. The only call site passed a literal em-dash
for the wallet and omitted ``equity_usd`` entirely, so the renderer's default of
``0.0`` was displayed. The wallet was never read and the price was never
fetched, because nothing in that code path asked for either. The balance code
itself was correct and simply unused.

THE RULE THIS ENFORCES
----------------------
A balance or a price that could not be read is **unavailable**, never zero.
Collapsing an error into ``0.0000`` is the actual defect: it looks like a
measured result, and it makes sizing refuse for the wrong reason. So:

* a real, readable balance of zero yields a genuine ``$0.00``;
* an unreadable balance, or a missing SOL/USD price, yields
  ``equity_usd = None`` plus a reason, and REAL sizing must not proceed.

Equity is computed ONCE here and consumed by the status message, the preflight
and the sizing guard, so the three cannot disagree.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Iterable, Optional

#: Lamports per SOL. Used for the single conversion, so the rate lives in one
#: place and cannot drift between the balance reader and the valuation.
LAMPORTS_PER_SOL = 1_000_000_000

#: Wrapped SOL. Recognised by mint so a caller that labels the asset "WSOL" and
#: a caller that leaves the symbol blank still agree it is worth the SOL price.
WSOL_MINT = "So11111111111111111111111111111111111111112"


class EquityUnavailable:
    """Sentinel: equity could not be determined, and this is why."""

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __bool__(self) -> bool:          # always False: it is not a number
        return False

    def __repr__(self) -> str:           # pragma: no cover - debugging aid
        return f"EquityUnavailable({self.reason!r})"


@dataclass(frozen=True)
class RealEquity:
    """
    The calculated REAL equity, or an explicit statement that it is unavailable.

    ``equity_usd`` is ``None`` if and only if the value is genuinely unknown.
    A wallet holding exactly nothing yields ``0.0``, which is a real answer.
    """

    sol: float
    sol_price_usd: float
    equity_usd: Optional[float]
    wallet_masked: str
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.equity_usd is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sol": self.sol,
            "sol_price_usd": self.sol_price_usd,
            "equity_usd": self.equity_usd,
            "wallet_masked": self.wallet_masked,
            "reason": self.reason,
            "available": self.available,
        }


def lamports_to_sol(lamports: Any) -> Optional[float]:
    """
    Lamports -> SOL, or ``None`` if the input is not a usable integer.

    Rejects rather than coercing: a bool, a negative value, a float or a
    non-numeric string must not become a plausible-looking balance.
    """
    if isinstance(lamports, bool) or lamports is None:
        return None
    if isinstance(lamports, float):
        if lamports != int(lamports) or lamports < 0:
            return None
        lamports = int(lamports)
    if not isinstance(lamports, int):
        return None
    if lamports < 0:
        return None
    return lamports / LAMPORTS_PER_SOL


@dataclass(frozen=True)
class AssetHolding:
    """
    One non-native asset measured in a REAL wallet.

    ``amount`` is the RAW on-chain integer, not a float: a balance of
    ``1_000_000_000`` with ``decimals=9`` is exactly 1 token, and doing that
    division in float first is how a rounding error becomes a wrong equity.

    ``price_usd`` is ``None`` when no price could be read. That is the important
    case: a held asset with no price makes total equity UNAVAILABLE, because a
    partial total is a number that looks complete.
    """

    mint: str
    symbol: str
    amount: int
    decimals: int = 0
    price_usd: Optional[float] = None

    @property
    def ui_amount(self) -> float:
        try:
            return self.amount / (10 ** int(self.decimals))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return 0.0

    @property
    def value_usd(self) -> Optional[float]:
        if self.price_usd is None:
            return None
        return self.ui_amount * float(self.price_usd)

    @property
    def is_wsol(self) -> bool:
        """WSOL is a wrapper around SOL, so it is worth the SOL price."""
        return self.symbol.upper() in ("WSOL", "WRAPPED SOL") or self.mint == WSOL_MINT


@dataclass(frozen=True)
class MultiAssetEquity:
    """
    REAL equity across every supported asset, or an explicit refusal.

    EQUITY IS THE SUM, OR IT IS NOTHING. The previous version valued native SOL
    only. On a wallet holding a wrapped-SOL or USDC balance that number silently
    excluded real money, and sizing a position off it would size against a
    fraction of the account. So a non-zero holding with no readable price
    invalidates the whole figure rather than being dropped from it.
    """

    sol: float
    sol_price_usd: float
    equity_usd: Optional[float]
    wallet_masked: str
    assets: tuple[AssetHolding, ...] = ()
    reason: str = ""
    unpriced_mints: tuple[str, ...] = ()
    #: PROVENANCE of the SOL/USD reference the figure was computed from. These
    #: are ADDITIVE: no existing field is renamed or removed. ``sol_usd_source``
    #: says which reference answered and ``sol_usd_timestamp``/``sol_usd_age_ms``
    #: say when, so a valuation is auditable rather than a bare float that looks
    #: like a measurement. All three are ``None`` when no usable reference was
    #: obtained, which is exactly the case that must not be mistaken for a
    #: reading.
    sol_usd_source: Optional[str] = None
    sol_usd_timestamp: Optional[float] = None
    sol_usd_age_ms: Optional[float] = None

    @property
    def available(self) -> bool:
        return self.equity_usd is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sol": self.sol,
            "sol_price_usd": self.sol_price_usd,
            "equity_usd": self.equity_usd,
            "wallet_masked": self.wallet_masked,
            "available": self.available,
            "reason": self.reason,
            "unpriced_mints": list(self.unpriced_mints),
            "sol_usd_source": self.sol_usd_source,
            "sol_usd_timestamp": self.sol_usd_timestamp,
            "sol_usd_age_ms": self.sol_usd_age_ms,
            "assets": [
                {
                    "mint": a.mint,
                    "symbol": a.symbol,
                    "amount": a.amount,
                    "ui_amount": a.ui_amount,
                    "price_usd": a.price_usd,
                    "value_usd": a.value_usd,
                }
                for a in self.assets
            ],
        }


def compute_real_equity_multi(
    lamports: Any,
    sol_price_usd: Any,
    holdings: Iterable[AssetHolding] = (),
    wallet_masked: str = "",
    *,
    sol_usd_source: Optional[str] = None,
    sol_usd_timestamp: Optional[float] = None,
    sol_usd_age_ms: Optional[float] = None,
) -> MultiAssetEquity:
    """
    Equity across native SOL and every supported SPL holding, or a refusal.

    Three rules, in order:

    1. Native SOL and WSOL are the same asset twice; WSOL is valued at the SOL
       price automatically and can never be the reason equity is unavailable.
    2. A held, non-zero asset with no readable price makes equity UNAVAILABLE
       and names the mint. It is not valued at zero and not left out.
    3. A ZERO balance needs no price. Nothing is at stake in valuing nothing,
       and requiring a price for an empty balance would make an idle wallet
       permanently unsizable.

    The three keyword-only provenance fields are carried through to the result
    so the caller can record WHERE the SOL/USD reference came from. They are
    passed through, never invented here: this function does no I/O and has no
    opinion about which reference provider answered.
    """
    provenance = {
        "sol_usd_source": sol_usd_source,
        "sol_usd_timestamp": sol_usd_timestamp,
        "sol_usd_age_ms": sol_usd_age_ms,
    }
    mask = wallet_masked or "(unavailable)"
    assets = tuple(holdings or ())

    sol = lamports_to_sol(lamports)
    if sol is None:
        return MultiAssetEquity(
            sol=0.0, sol_price_usd=0.0, equity_usd=None, wallet_masked=mask,
            assets=assets,
            reason=("native SOL balance unreadable" if lamports is None
                    else f"native SOL balance invalid: {lamports!r}"),
            **provenance,
        )

    # --- the SOL price, validated once and reused for SOL and WSOL ---------
    if sol_price_usd is None:
        price = None
        sol_reason = "SOL/USD valuation unavailable"
    else:
        try:
            candidate = float(sol_price_usd)
        except (TypeError, ValueError):
            candidate, sol_reason = None, f"SOL/USD valuation invalid: {sol_price_usd!r}"
        else:
            if not (candidate > 0) or candidate != candidate or candidate == float("inf"):
                candidate = None
                sol_reason = f"SOL/USD valuation not usable: {sol_price_usd!r}"
            else:
                sol_reason = ""
    price = candidate if sol_price_usd is not None or sol_reason == "" else None

    # WSOL is SOL. Fill its price in when the caller left it unset, so a caller
    # that does not special-case the wrapper still gets a correct total.
    priced: list[AssetHolding] = []
    for h in assets:
        if price is not None and h.is_wsol and h.price_usd is None:
            priced.append(dataclasses.replace(h, price_usd=price))
        else:
            priced.append(h)
    assets = tuple(priced)

    # --- unpriced NON-ZERO holdings invalidate the total -------------------
    unpriced = tuple(h.mint for h in assets if h.amount != 0 and h.value_usd is None)
    if unpriced:
        symbols = ", ".join(
            (h.symbol or h.mint) for h in assets
            if h.amount != 0 and h.value_usd is None)
        return MultiAssetEquity(
            sol=sol,
            sol_price_usd=float(price or 0.0),
            equity_usd=None,
            wallet_masked=mask,
            assets=assets,
            unpriced_mints=unpriced,
            reason=(f"held asset(s) with no readable USD price: {symbols}. "
                    f"Equity is UNAVAILABLE rather than a partial sum."),
            **provenance,
        )

    if price is None:
        return MultiAssetEquity(
            sol=sol, sol_price_usd=0.0, equity_usd=None, wallet_masked=mask,
            assets=assets, reason=sol_reason,
            **provenance,
        )

    total = sol * price
    for h in assets:
        v = h.value_usd
        if v:
            total += v

    return MultiAssetEquity(
        sol=sol,
        sol_price_usd=float(price),
        equity_usd=total,
        wallet_masked=mask,
        assets=assets,
        **provenance,
    )


def compute_real_equity(
    lamports: Any,
    sol_price_usd: Any,
    wallet_masked: str = "",
) -> RealEquity:
    """
    Combine a native balance and a SOL/USD price into equity, or explain why not.

    Every failure is explicit. There is no default price, no fallback balance and
    no path that yields a ``0.0`` equity that was not actually measured.

    This is the native-SOL-only view, kept for callers that genuinely only care
    about SOL. :func:`compute_real_equity_multi` is the one a REAL wallet should
    use, because a REAL wallet may hold more than SOL.
    """
    r = compute_real_equity_multi(lamports, sol_price_usd, (), wallet_masked)
    return RealEquity(
        sol=r.sol,
        sol_price_usd=r.sol_price_usd,
        equity_usd=r.equity_usd,
        wallet_masked=r.wallet_masked,
        reason=r.reason,
    )
