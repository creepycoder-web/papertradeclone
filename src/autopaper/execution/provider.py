"""Unified execution-provider interface for the REAL swap-build layer.

WHY THIS EXISTS
---------------
The REAL path must not care *which* aggregator builds a transaction. It needs
one shape:

    quote()             -> a priced, safety-checked quote
    build_transaction() -> an UNSIGNED transaction ready for local signing
    provider_name()     -> "raptor" / "jupiter" (for reporting + fallback)
    route_metadata()    -> sanitized route info (no payload, no secrets)

Raptor is the primary provider; Jupiter is the fallback for routes Raptor
cannot serve. Both are ordinary implementations of this interface, so the L25 /
strategy layer never sees a provider detail, and the executor can run BOTH
through the exact same safety pipeline.

WHAT A PROVIDER IS NOT ALLOWED TO DO
------------------------------------
* hold or receive a private key, seed phrase, or signed transaction
* sign
* broadcast
* decide position size, slippage, freshness, or route preference by PnL

Those stay in the bot. A provider only quotes and builds.

SOL / WSOL OWNERSHIP
--------------------
A SOL-denominated BUY has no single right answer for how SOL becomes the
input leg. Raptor manages native SOL itself (it wraps and unwraps inside the
swap and needs no pre-provisioned wSOL account); a provider may instead prefer
the wallet to already hold wSOL. That decision belongs to the PROVIDER, not to
the generic execution layer, so it is declared here as data:

    requires_preprovisioned_wsol   -- True when the provider cannot serve a
                                      SOL input without a funded wSOL ATA
    sol_input_mode                 -- how it turns native SOL into the input
                                      leg, for reporting

The generic executor must NEVER read ``requires_preprovisioned_wsol`` to gate
a SOL BUY on a pre-provisioned wSOL account. Only the provider's own native
path (e.g. Jupiter's ``execute_swap`` planner) may impose that requirement. A
provider that manages SOL automatically must work from native balance alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class ExecutionProvider(ABC):
    """The one interface every swap-build provider implements."""

    #: True only when the provider's native SOL path REQUIRES the wallet to
    #: already hold a funded wrapped-SOL account. Raptor: False (it manages
    #: native SOL itself). Descriptive metadata -- the generic executor never
    #: gates on it.
    requires_preprovisioned_wsol: bool = False

    #: How this provider forms a native-SOL input leg. One of
    #: ``"auto_wrap_unwrap"`` (provider wraps/unwraps internally),
    #: ``"preprovision_capable"`` (can consume an existing wSOL account), or
    #: ``"unknown"``. Reporting only.
    sol_input_mode: str = "unknown"

    @abstractmethod
    def provider_name(self) -> str:
        """Stable lower-case provider id: ``"raptor"`` / ``"jupiter"``."""
        raise NotImplementedError

    @abstractmethod
    async def quote(self, input_mint: str, output_mint: str, in_amount: int,
                    slippage_bps: int) -> Any:
        """Price a swap. Raises a provider-classified error on failure."""
        raise NotImplementedError

    @abstractmethod
    async def build_transaction(
        self,
        quote: Any,
        user_public_key: str,
        *,
        wrap_and_unwrap_sol: bool = True,
        priority_fee: Any = "auto",
        only_direct_routes: Optional[bool] = None,
        dynamic_compute_unit_limit: Optional[bool] = None,
        tx_version: Optional[str] = None,
    ) -> Any:
        """Build an UNSIGNED transaction for ``quote``.

        The returned object must expose ``transaction_b64`` (base64 of an
        unsigned transaction). Implementations must never sign.
        """
        raise NotImplementedError

    @abstractmethod
    def route_metadata(self, quote: Any) -> dict:
        """Sanitized route projection. Never a transaction payload."""
        raise NotImplementedError

    async def health(self) -> dict:
        """Readiness probe. Never raises; returns ``reachable``."""
        return {"provider": self.provider_name(), "reachable": True,
                "detail": "no probe implemented"}

    async def aclose(self) -> None:
        return None
