"""
Read-only wallet asset discovery across BOTH Solana token programs.

WHY THIS EXISTS
---------------
``SolanaRpc.token_balances`` filtered ``getTokenAccountsByOwner`` to the classic
SPL Token program. On Solana a wallet may hold accounts under the classic
program AND under Token-2022, and a programId-filtered scan sees only one of
them. Every live L25 mint in this account's history -- $SI, $Boy, notify -- was
Token-2022, so a classic-only scan reported the wallet as holding nothing while
rent sat locked in eight empty Token-2022 accounts.

This module is the single read path that states, explicitly and completely, what
a wallet holds:

    native SOL
    SPL token balances (classic program)
    Token-2022 balances
    each account's mint, decimals, owning program and rent-exempt lamports
    a USD valuation where a price is available
    an UNKNOWN/unpriced bucket that is named, never silently dropped

THE RULES
---------
1. AN ASSET IS NEVER SILENTLY IGNORED. Every token account is reported. A
   non-zero balance with no readable price keeps ``asset_detected=True`` and
   ``valuation="unavailable"``; it is NOT converted to zero and NOT omitted.
2. EQUITY IS THE SUM, OR IT IS NOT STATED. If any non-zero holding cannot be
   priced the whole equity is ``None`` and the mint is named -- a partial total
   that looks complete is the defect, not the fix.
3. RENT IS MEASURED PER ACCOUNT. Empty accounts' own lamports are summed into
   ``rent_recoverable_lamports``; the number comes from the chain, not from a
   constant.
4. AN UNREADABLE BALANCE IS UNAVAILABLE, NOT ZERO.

This module performs no signing and no broadcast. It reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from .execution.solana import (
    TOKEN_2022_PROGRAM_ID,
    TOKEN_PROGRAM_ID,
    TOKEN_PROGRAMS,
    WSOL_MINT,
)

LAMPORTS_PER_SOL = 1_000_000_000

#: A valuation verdict. ``unavailable`` is a real answer, not a failure.
VALUATION_PRICED = "priced"
VALUATION_UNAVAILABLE = "unavailable"

#: Human labels for the owning program, so a report can distinguish them.
PROGRAM_LABEL = {
    TOKEN_PROGRAM_ID: "SPL_TOKEN",
    TOKEN_2022_PROGRAM_ID: "TOKEN_2022",
}


def program_label(program_id: str) -> str:
    return PROGRAM_LABEL.get(program_id, f"UNKNOWN({program_id[:12]})" if program_id else "UNKNOWN")


@dataclass(frozen=True)
class WalletAsset:
    """One token balance in a wallet, with its program, account and valuation."""

    mint: str
    symbol: str
    amount_raw: int
    decimals: int
    program: str = TOKEN_PROGRAM_ID
    token_account: str = ""
    lamports: Optional[int] = None
    price_usd: Optional[float] = None

    @property
    def is_token_2022(self) -> bool:
        return self.program == TOKEN_2022_PROGRAM_ID

    @property
    def is_empty(self) -> bool:
        return self.amount_raw == 0

    @property
    def ui_amount(self) -> float:
        try:
            return self.amount_raw / (10 ** int(self.decimals))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return 0.0

    @property
    def value_usd(self) -> Optional[float]:
        if self.price_usd is None:
            return None
        return self.ui_amount * float(self.price_usd)

    @property
    def valuation(self) -> str:
        """
        ``priced`` or ``unavailable``. An EMPTY balance is never priced and is
        never ``unavailable`` in a way that matters: nothing is at stake.
        """
        if self.is_empty:
            return VALUATION_PRICED
        return VALUATION_PRICED if self.value_usd is not None else VALUATION_UNAVAILABLE

    @property
    def asset_detected(self) -> bool:
        """The asset exists in the wallet, whether or not it can be valued."""
        return self.amount_raw != 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "mint": self.mint,
            "symbol": self.symbol,
            "amount_raw": self.amount_raw,
            "decimals": self.decimals,
            "ui_amount": self.ui_amount,
            "program": self.program,
            "program_label": program_label(self.program),
            "is_token_2022": self.is_token_2022,
            "token_account": self.token_account,
            "lamports": self.lamports,
            "is_empty": self.is_empty,
            # These two facts are what the brief asks to be reported separately.
            "asset_detected": self.asset_detected,
            "valuation": self.valuation,
            "price_usd": self.price_usd,
            "value_usd": self.value_usd,
        }


@dataclass
class WalletScan:
    """The complete, honest view of a wallet at one instant."""

    wallet_masked: str
    observed_at: float
    native_lamports: Optional[int]
    sol_price_usd: Optional[float]
    assets: tuple[WalletAsset, ...] = ()
    reason: str = ""
    balance_readable: bool = True

    # -- derived -----------------------------------------------------------

    @property
    def sol(self) -> Optional[float]:
        if self.native_lamports is None:
            return None
        return self.native_lamports / LAMPORTS_PER_SOL

    @property
    def unpriced_mints(self) -> tuple[str, ...]:
        """Non-zero holdings with no readable USD price. Named, never dropped."""
        return tuple(
            a.mint for a in self.assets if a.amount_raw != 0 and a.value_usd is None
        )

    @property
    def unknown_assets(self) -> tuple[WalletAsset, ...]:
        return tuple(
            a for a in self.assets if a.amount_raw != 0 and a.value_usd is None
        )

    @property
    def token_2022_assets(self) -> tuple[WalletAsset, ...]:
        return tuple(a for a in self.assets if a.is_token_2022 and a.amount_raw != 0)

    @property
    def empty_accounts(self) -> tuple[WalletAsset, ...]:
        return tuple(a for a in self.assets if a.is_empty)

    @property
    def rent_recoverable_lamports(self) -> int:
        """Rent locked in EMPTY accounts, measured from the chain."""
        return sum(int(a.lamports or 0) for a in self.assets if a.is_empty)

    @property
    def equity_usd(self) -> Optional[float]:
        """
        The sum, or ``None``.

        ``None`` when the native balance is unreadable, the SOL price is missing,
        or any non-zero holding cannot be priced. Never a partial total.
        """
        if self.native_lamports is None or self.sol_price_usd is None:
            return None
        try:
            price = float(self.sol_price_usd)
        except (TypeError, ValueError):
            return None
        if not (price > 0) or price != price:
            return None
        if self.unpriced_mints:
            return None
        total = (self.native_lamports / LAMPORTS_PER_SOL) * price
        for a in self.assets:
            v = a.value_usd
            if v:
                total += v
        return total

    @property
    def available(self) -> bool:
        return self.equity_usd is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "wallet_masked": self.wallet_masked,
            "observed_at": self.observed_at,
            "native_lamports": self.native_lamports,
            "sol": self.sol,
            "sol_price_usd": self.sol_price_usd,
            "balance_readable": self.balance_readable,
            "equity_usd": self.equity_usd,
            "available": self.available,
            "reason": self.reason,
            "unpriced_mints": list(self.unpriced_mints),
            "token_2022_count": len(self.token_2022_assets),
            "empty_account_count": len(self.empty_accounts),
            "rent_recoverable_lamports": self.rent_recoverable_lamports,
            "rent_recoverable_sol": self.rent_recoverable_lamports / LAMPORTS_PER_SOL,
            "assets": [a.as_dict() for a in self.assets],
        }


#: A price callback: ``(mint, symbol) -> Optional[float]``. ``None`` means the
#: price could not be read, which is a verdict, not a zero.
PriceOf = Callable[[str, str], Optional[float]]


async def scan_wallet(
    rpc: Any,
    address: str,
    *,
    sol_price_usd: Optional[float] = None,
    price_of: Optional[PriceOf] = None,
    now: float = 0.0,
    wallet_masked: str = "",
) -> WalletScan:
    """
    Read every asset in ``address`` across both token programs.

    ``rpc`` is anything exposing ``get_balance_lamports`` and ``token_balances``
    (both are read-only). Nothing here signs or broadcasts.

    A failure to READ native SOL yields ``native_lamports=None`` and an
    unreadable balance, never a zero. A failure to list token accounts is NOT
    treated as "no holdings": it raises, because silently returning an empty
    asset list would understate the wallet in exactly the way this module exists
    to prevent. The caller decides how to handle a partial read.
    """
    try:
        lamports = await rpc.get_balance_lamports(address)
    except Exception as exc:  # noqa: BLE001
        return WalletScan(
            wallet_masked=wallet_masked,
            observed_at=now,
            native_lamports=None,
            sol_price_usd=sol_price_usd,
            reason=f"native SOL balance unreadable ({type(exc).__name__})",
            balance_readable=False,
        )

    rows = await rpc.token_balances(address)
    assets: list[WalletAsset] = []
    for row in rows or ():
        try:
            mint = str(row.get("mint") or "")
            amount = int(row.get("amount"))
            decimals = int(row.get("decimals") or 0)
        except (TypeError, ValueError, AttributeError):
            continue
        if not mint:
            continue
        symbol = str(row.get("symbol") or "")
        program = str(row.get("program") or TOKEN_PROGRAM_ID)
        lamports_row = row.get("lamports")
        try:
            lamports_row = int(lamports_row) if lamports_row is not None else None
        except (TypeError, ValueError):
            lamports_row = None

        price: Optional[float] = None
        if amount != 0:
            if mint == WSOL_MINT:
                # WSOL is wrapped SOL: worth the SOL price, never routed.
                price = sol_price_usd
            elif price_of is not None:
                try:
                    price = price_of(mint, symbol)
                except Exception:  # noqa: BLE001 - an unreadable price is None
                    price = None
        assets.append(
            WalletAsset(
                mint=mint,
                symbol=symbol,
                amount_raw=amount,
                decimals=decimals,
                program=program,
                token_account=str(row.get("account") or ""),
                lamports=lamports_row,
                price_usd=price,
            )
        )

    return WalletScan(
        wallet_masked=wallet_masked,
        observed_at=now,
        native_lamports=int(lamports),
        sol_price_usd=sol_price_usd,
        assets=tuple(assets),
    )


def summarize_scan(scan: WalletScan) -> dict[str, Any]:
    """A compact, operator-facing summary of one scan."""
    return {
        "wallet": scan.wallet_masked,
        "native_sol": scan.sol,
        "sol_price_usd": scan.sol_price_usd,
        "spl_count": len([a for a in scan.assets
                          if a.program == TOKEN_PROGRAM_ID and a.amount_raw != 0]),
        "token_2022_count": len(scan.token_2022_assets),
        "unknown_asset_count": len(scan.unknown_assets),
        "unpriced_mints": list(scan.unpriced_mints),
        "empty_account_count": len(scan.empty_accounts),
        "rent_recoverable_lamports": scan.rent_recoverable_lamports,
        "rent_recoverable_sol": scan.rent_recoverable_lamports / LAMPORTS_PER_SOL,
        "equity_usd": scan.equity_usd,
        "available": scan.available,
        "reason": scan.reason,
    }
