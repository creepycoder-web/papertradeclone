"""
Read-only chain access for the direct Pump / PumpSwap quotes.

WHAT THIS DOES
==============

Turns a mint into a verified :class:`~autopaper.same_market.pumpdirect.PumpSwapPool`
or :class:`~autopaper.same_market.pumpdirect.BondingCurve`, and a market into an
:class:`~autopaper.same_market.pumpdirect.DirectQuoteResult`, using nothing but
the read methods on
:class:`~autopaper.same_market.rpc.SolanaReadOnlyRpc`.

THE ORDER OF OPERATIONS IS THE POINT
=====================================

    derive -> fetch -> CHECK OWNER -> CHECK DISCRIMINATOR -> decode -> verify identity

Every step that could produce a confident wrong answer has a check in front of
it, and each check is one that has already caught a real mistake in this
codebase:

* **derive is a claim.** A PDA derivation says what the address *should* be.
  The account's ``owner`` is what the chain says it *is*. A seed typo produces
  a valid-looking address whose account does not exist, or worse, exists and is
  some other account.
* **owner is not enough.** An account owned by a program is not automatically
  the account type wanted, so the Anchor discriminator is checked too.
* **identity is separate from existence.** For a curve the mint is recovered
  from the PDA seeds (the account itself no longer stores it), so the check is
  that the derived address re-derives from the mint under test.
* **absent is not empty.** A missing account yields ``None``, which the caller
  turns into ``UNKNOWN``. It is never read as "zero reserves".

FRESHNESS IS RECORDED PER QUOTE, NOT PER RUN
=============================================

A bonding curve moves on every trade. A quote computed from state that is
four seconds old describes a market that has already traded, so
:attr:`StateSnapshot.state_age_ms` and the slot are carried into the report and
the caller applies its own staleness policy. ``UNKNOWN`` freshness is never
presented as fresh.

NO EXECUTION SURFACE
====================

There is no ``send``, ``sign`` or transaction builder here, and this module
imports nothing that has one. The RPC object's ``_rpc`` method refuses any
method outside its five-name read list, so a write cannot be expressed even by
mistake.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: How long the PumpSwap AMM program-config accounts may be reused. These are
#: governance constants, not a price feed; the window is short enough that a
#: genuine fee change still takes effect promptly.
AMM_FEE_CACHE_TTL_S = 30.0

from .models import WSOL_MINT, FailureMode, MarketKind
from .numeric import to_float
from .pumpdirect import (
    AMM_FEE_CONFIG_PDA,
    AMM_GLOBAL_CONFIG_PDA,
    GLOBAL_PDA,
    PUMP_FEE_CONFIG_PDA,
    PUMP_FEE_PROGRAM,
    PUMP_FUN_PROGRAM,
    PUMP_SWAP_PROGRAM,
    BondingCurve,
    DirectQuoteResult,
    FeeConfig,
    MarketClassification,
    PumpGlobal,
    PumpSwapPool,
    StateUnavailable,
    amm_fee_schedule,
    classify,
    curve_fee_schedule,
    decode_amm_global_config,
    decode_bonding_curve,
    decode_fee_config,
    decode_global,
    decode_pool,
    decode_pumpswap_pool,
    decode_spl_mint,
    derive_bonding_curve,
    derive_canonical_pool,
    quote_bonding_curve_buy,
    quote_bonding_curve_sell,
    quote_pumpswap_buy,
    quote_pumpswap_sell,
)
from .pda import b58encode
from .rpc import RpcRateLimited, RpcUnavailable, is_token_program


@dataclass(frozen=True)
class StateSnapshot:
    """Provenance of one set of on-chain reads."""

    slot: int | None
    fetched_at: float
    fetch_latency_ms: float | None
    accounts_read: int
    owner_checks: tuple[tuple[str, str], ...] = ()
    failure: FailureMode | None = None
    detail: str | None = None

    def state_age_ms(self, now: float | None = None) -> float | None:
        if self.slot is None and self.fetched_at <= 0:
            return None
        return max(0.0, ((now or time.time()) - self.fetched_at) * 1000.0)

    def as_dict(self, *, now: float | None = None) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "state_fetch_timestamp": self.fetched_at,
            "state_age_ms": self.state_age_ms(now),
            "fetch_latency_ms": self.fetch_latency_ms,
            "accounts_read": self.accounts_read,
            "owner_checks": [
                {"address": a, "owner": o} for a, o in self.owner_checks
            ],
            "failure": self.failure.value if self.failure else None,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class DirectQuote:
    """
    A verified direct quote, with everything needed to reproduce it.

    ``missing_reason`` is the field that keeps this honest: when no direct
    quote could be produced, the object still exists (so the report can show
    the comparison was ATTEMPTED) and says why it is empty, rather than being
    absent from the report as though it were never looked for.
    """

    provider: str
    market_type: MarketKind
    quote: DirectQuoteResult | None
    state: Any | None
    snapshot: StateSnapshot | None
    curve_address: str | None = None
    pool_address: str | None = None
    base_mint: str | None = None
    quote_mint: str | None = None
    missing_reason: str | None = None
    failure: FailureMode | None = None
    identity_evidence: str | None = None
    #: Wall-to-wall time for the chain reads plus the arithmetic, so the report
    #: can separate "the node was slow" from "we spent time thinking".
    quote_latency_ms: float | None = None

    @property
    def available(self) -> bool:
        return self.quote is not None

    def as_dict(self, *, sol_reference_usd: float | None = None,
                now: float | None = None) -> dict[str, Any]:
        """
        Serialise, keeping every unit distinct.

        The brief for this work asks for raw amount, decimals, human amount,
        quote-asset amount and USD reference all stored SEPARATELY, because
        the historical bugs in this repository were 10**6 and 10**9 mistakes
        that looked fine. So nothing here is pre-combined into a single number.
        """
        price_usd = None
        if self.quote is not None and sol_reference_usd:
            price_usd = float(self.quote.effective_price) * float(sol_reference_usd)
        body: dict[str, Any] = {
            "provider": self.provider,
            "market_type": self.market_type.value,
            "available": self.available,
            "curve_address": self.curve_address,
            "pool_address": self.pool_address,
            "base_mint": self.base_mint,
            "quote_mint": self.quote_mint,
            "identity_evidence": self.identity_evidence,
            "missing_reason": self.missing_reason,
            "failure": self.failure.value if self.failure else None,
            "usd_reference_price": sol_reference_usd,
            "state": (
                self.state.as_dict() if self.state is not None else None
            ),
            "state_read": (
                self.snapshot.as_dict(now=now) if self.snapshot is not None else None
            ),
            "quote_latency_ms": self.quote_latency_ms,
            "quote_detail": (
                self.quote.as_dict() if self.quote is not None else None
            ),
        }
        if self.quote is not None:
            body["effective_price_usd_per_token"] = (
                to_float(self.quote.effective_price * _ref(sol_reference_usd))
                if sol_reference_usd else None
            )
            body["effective_price_sol_per_token"] = to_float(
                self.quote.effective_price
            )
        else:
            body["effective_price_usd_per_token"] = None
            body["effective_price_sol_per_token"] = None
        return body


def _ref(value: float | None):
    from decimal import Decimal

    return Decimal(str(value)) if value else Decimal(1)


def _reraise_if_telemetry_skip(exc: BaseException) -> None:
    """
    Let the telemetry lane's OWN refusal escape as itself.

    A spent reserved telemetry budget is a decision this side made -- "we chose
    not to ask" -- and NOT a fact about the market. Wrapping it in
    ``StateUnavailable`` erased that distinction: it reached the hold-path
    sampler as "the bonding curve could not be read", so 73% of a position's
    monitoring ticks were filed as QUOTE_UNAVAILABLE (a market failure) when the
    truth was SKIPPED_BUDGET (our own reservation). Re-raising it lets the
    sampler's existing skip predicate classify it correctly, and keeps a market
    failure and a capacity decision from ever being pooled again.
    """
    from ..market.ratelimit import TELEMETRY_BUDGET

    if (isinstance(exc, RpcRateLimited)
            and getattr(exc, "reason", None) == TELEMETRY_BUDGET):
        raise exc


class DirectPumpReader:
    """
    Read-only reader for pump.fun bonding curves and PumpSwap pools.

    Collaborators are injected so the whole thing is drivable from fixtures with
    no network at all. The RPC object only needs ``getMultipleAccounts``,
    ``getSlot`` and ``getTokenSupply``.
    """

    def __init__(
        self,
        rpc: Any,
        *,
        token_decimals_for: Any = None,
        commitment: str = "confirmed",
    ) -> None:
        self.rpc = rpc
        self.commitment = commitment
        self._decimals_for = token_decimals_for

    # -- low-level ---------------------------------------------------------

    async def _slot(self) -> int | None:
        try:
            return await self.rpc.get_slot()
        except Exception:  # noqa: BLE001 - the slot is context, not the subject
            return None

    async def _accounts(self, addresses: Sequence[str]) -> list[Any]:
        return await self.rpc.get_multiple_accounts(
            list(addresses),
            encoding="base64",
            commitment=self.commitment,
        )

    @staticmethod
    def _raw(account: Any, what: str) -> bytes:
        if account is None or account.is_empty:
            raise StateUnavailable(
                f"the {what} account does not exist at that address, so its "
                "state is UNKNOWN rather than zero"
            )
        return account.raw()

    @staticmethod
    def _verify_owner(account: Any, expected: str, address: str,
                      what: str) -> None:
        if account.owner != expected:
            raise StateUnavailable(
                f"the {what} account at {address} is owned by {account.owner}, "
                f"not {expected}; these are not this program's bytes, so they "
                "are not decoded as a market"
            )

    async def _mint_supply(self, mint: str) -> int | None:
        try:
            return await self.rpc.get_token_supply(mint)
        except Exception:  # noqa: BLE001 - optional context
            return None

    async def mint_decimals(self, mint: str) -> int | None:
        """
        The mint's own ``decimals``, read from the mint account.

        Not assumed. A 10**6 error here would scale every price in the report
        by a million while every individual number stayed plausible, which is
        precisely the class of bug this harness exists to catch. So the value is
        read, and when it cannot be read the caller is told ``None`` rather than
        being handed a default of 6.
        """
        # The CAUSE is recorded. This method returns ``None`` for "could not
        # read", and a caller that only sees ``None`` cannot distinguish a
        # genuinely absent mint account from a transient RPC failure -- which is
        # the difference between a correct refusal and a broken runtime.
        self.last_mint_decimals_error = ""
        try:
            account = (await self._accounts([mint]))[0]
        except Exception as exc:  # noqa: BLE001 - named as absent
            self.last_mint_decimals_error = (
                f"{type(exc).__name__}: {exc}")[:200]
            return None
        if account is None or account.is_empty:
            self.last_mint_decimals_error = (
                "the mint account is absent or empty")
            return None
        if not is_token_program(account.owner):
            self.last_mint_decimals_error = (
                "the mint account is not owned by a recognised token program")
            return None
        try:
            decimals, _supply = decode_spl_mint(account.raw(), what=f"mint {mint[:8]}")
        except StateUnavailable:
            return None
        return decimals

    # -- shared fee state --------------------------------------------------

    async def _pump_fee_state(self) -> tuple[PumpGlobal | None, FeeConfig | None]:
        """
        ``Global`` and the live ``FeeConfig``.

        Both are read because the protocol's own code prefers the FeeConfig and
        falls back to Global. Reading only Global would quote the fallback fee
        on every trade, which is wrong today (95 bps + 30 bps live, versus 95 +
        5 in Global).
        """
        try:
            accounts = await self._accounts([GLOBAL_PDA, PUMP_FEE_CONFIG_PDA])
        except (RpcRateLimited, RpcUnavailable):
            raise
        except Exception as exc:  # noqa: BLE001 - named below
            raise StateUnavailable(f"could not read the pump fee accounts: {exc}") from exc

        glob: PumpGlobal | None = None
        fee_config: FeeConfig | None = None
        global_account, fee_account = accounts[0], accounts[1]
        if global_account is not None and not global_account.is_empty:
            self._verify_owner(global_account, PUMP_FUN_PROGRAM, GLOBAL_PDA, "global")
            glob = decode_global(global_account.raw())
        if fee_account is not None and not fee_account.is_empty:
            self._verify_owner(fee_account, PUMP_FEE_PROGRAM, PUMP_FEE_CONFIG_PDA,
                               "fee config")
            fee_config = decode_fee_config(fee_account.raw())
        return glob, fee_config

    async def _amm_fee_state(self) -> tuple[Any | None, FeeConfig | None]:
        """
        The PumpSwap AMM global config and fee config.

        These are PROGRAM CONFIG accounts, not market data: they define the fee
        schedule and change on a governance cadence, not per trade. They were
        being re-read on every single quote -- two extra round trips per
        candidate, purely to rediscover constants.

        They are cached for a bounded window and the TTL is deliberately short
        enough that a fee change still lands promptly. Nothing that PRICES a
        trade is cached here; only the constants a fee is computed FROM.
        """
        now = time.monotonic()
        cached = getattr(self, "_amm_fee_cache", None)
        if cached is not None and (now - cached[0]) < AMM_FEE_CACHE_TTL_S:
            return cached[1], cached[2]
        accounts = await self._accounts(
            [AMM_GLOBAL_CONFIG_PDA, AMM_FEE_CONFIG_PDA]
        )
        global_config = None
        fee_config = None
        global_account, fee_account = accounts[0], accounts[1]
        if global_account is not None and not global_account.is_empty:
            self._verify_owner(global_account, PUMP_SWAP_PROGRAM,
                               AMM_GLOBAL_CONFIG_PDA, "amm global config")
            global_config = decode_amm_global_config(global_account.raw())
        if fee_account is not None and not fee_account.is_empty:
            self._verify_owner(fee_account, PUMP_FEE_PROGRAM, AMM_FEE_CONFIG_PDA,
                               "amm fee config")
            fee_config = decode_fee_config(fee_account.raw())
        self._amm_fee_cache = (now, global_config, fee_config)
        return global_config, fee_config

    # -- bonding curve -----------------------------------------------------

    async def read_bonding_curve(self, mint: str) -> tuple[BondingCurve | None, StateSnapshot]:
        """
        Derive the curve PDA for ``mint``, read it, and verify it belongs there.

        Returns ``(None, snapshot)`` when the account does not exist, which is
        the NORMAL state for a graduated or non-pump token and is not an error.
        """
        started = time.monotonic()
        curve_address, _bump = derive_bonding_curve(mint)
        try:
            slot = await self._slot()
            account = (await self._accounts([curve_address]))[0]
        except (RpcRateLimited, RpcUnavailable) as exc:
            _reraise_if_telemetry_skip(exc)
            raise StateUnavailable(
                f"the bonding curve could not be read: {exc}",
                failure=FailureMode.RPC_RATE_LIMIT
                if isinstance(exc, RpcRateLimited)
                else FailureMode.STATE_UNAVAILABLE,
            ) from exc
        except Exception as exc:  # noqa: BLE001 - any transport fault is one finding
            raise StateUnavailable(
                f"reading the bonding curve at {curve_address} failed: "
                f"{type(exc).__name__}: {exc}",
                failure=FailureMode.STATE_UNAVAILABLE,
            ) from exc
        fetched = time.time()
        latency = (time.monotonic() - started) * 1000.0

        if account is None or account.is_empty:
            return None, StateSnapshot(
                slot=slot, fetched_at=fetched, fetch_latency_ms=latency,
                accounts_read=1,
                detail=(
                    f"no account at the derived bonding-curve PDA {curve_address}; "
                    "the token is either graduated or not a pump.fun launch"
                ),
            )
        self._verify_owner(account, PUMP_FUN_PROGRAM, curve_address, "bonding curve")
        curve = decode_bonding_curve(
            account.raw(), curve_address=curve_address, mint=mint, slot=slot,
            timestamp=fetched,
        )
        return curve, StateSnapshot(
            slot=slot, fetched_at=fetched, fetch_latency_ms=latency, accounts_read=1,
            owner_checks=((curve_address, account.owner),),
        )

    async def quote_bonding_curve(
        self,
        mint: str,
        *,
        sol_amount_lamports: int,
        token_decimals: int,
    ) -> DirectQuote:
        """
        A direct ``SOL -> TOKEN`` quote against the mint's bonding curve.

        Fails closed: every unavailable state becomes ``PUMP_DIRECT = UNKNOWN``
        with a reason, never a zero.
        """
        provider = "pump_direct"
        try:
            curve, snapshot = await self.read_bonding_curve(mint)
        except StateUnavailable as exc:
            return DirectQuote(
                provider=provider, market_type=MarketKind.PUMP_BONDING_CURVE,
                quote=None, state=None, snapshot=None, base_mint=mint,
                quote_mint=WSOL_MINT, missing_reason=str(exc),
                failure=exc.failure,
            )
        if curve is None:
            return DirectQuote(
                provider=provider, market_type=MarketKind.PUMP_BONDING_CURVE,
                quote=None, state=None, snapshot=snapshot, base_mint=mint,
                quote_mint=WSOL_MINT,
                missing_reason=(
                    f"no bonding curve exists for this mint ({snapshot.detail})"
                ),
                failure=FailureMode.CURVE_UNAVAILABLE,
            )
        if curve.graduated:
            return DirectQuote(
                provider=provider, market_type=MarketKind.PUMP_BONDING_CURVE,
                quote=None, state=curve, snapshot=snapshot,
                curve_address=curve.address, base_mint=mint, quote_mint=WSOL_MINT,
                missing_reason=(
                    "the bonding curve is complete, so it is no longer the "
                    "market and prices nothing"
                ),
                failure=FailureMode.CURVE_UNAVAILABLE,
                identity_evidence=(
                    f"account {curve.address} is owned by the pump.fun program "
                    "and carries the BondingCurve discriminator"
                ),
            )

        started = time.monotonic()
        try:
            glob, fee_config = await self._pump_fee_state()
            supply = await self._mint_supply(mint)
            if supply is None:
                raise StateUnavailable(
                    "the mint's circulating supply could not be read, and the "
                    "curve's fee tier depends on it; the tier is NOT guessed"
                )
            fees = curve_fee_schedule(
                curve, global_config=glob, fee_config=fee_config, mint_supply=supply
            )
            quote = quote_bonding_curve_buy(
                curve, sol_amount_lamports=sol_amount_lamports,
                token_decimals=token_decimals, fees=fees, mint_supply=supply,
            )
        except StateUnavailable as exc:
            return DirectQuote(
                provider=provider, market_type=MarketKind.PUMP_BONDING_CURVE,
                quote=None, state=curve, snapshot=snapshot,
                curve_address=curve.address, base_mint=mint, quote_mint=WSOL_MINT,
                missing_reason=str(exc), failure=exc.failure,
                identity_evidence=(
                    f"account {curve.address} is owned by the pump.fun program "
                    "and carries the BondingCurve discriminator"
                ),
            )
        return DirectQuote(
            provider=provider, market_type=MarketKind.PUMP_BONDING_CURVE,
            quote=quote, state=curve, snapshot=snapshot,
            quote_latency_ms=(time.monotonic() - started) * 1000.0,
            curve_address=curve.address, base_mint=mint,
            quote_mint=WSOL_MINT if curve.quote_mint_is_sol_like else curve.quote_mint,
            identity_evidence=(
                f"the account at the PDA derived from seeds "
                f"['bonding-curve', {mint}] is owned by the pump.fun program "
                f"and carries the BondingCurve discriminator, so these bytes "
                "are the curve for this mint"
            ),
        )

    # -- pumpswap ----------------------------------------------------------

    async def read_market_by_address(
        self,
        address: str,
        *,
        mint: str,
        quote_mint: str = WSOL_MINT,
    ) -> tuple[BondingCurve | PumpSwapPool | None, StateSnapshot, str, str]:
        """
        Read a market (bonding curve or PumpSwap pool) by its known address.

        Returns ``(market, snapshot, market_type, provider)`` where market_type is
        "PUMP_BONDING_CURVE" or "PUMPSWAP" and provider is "pump_direct" or
        "pumpswap_direct".

        Used by monitoring to fetch static identity without re-deriving addresses.
        """
        started = time.monotonic()
        try:
            slot = await self._slot()
            account = (await self._accounts([address]))[0]
        except (RpcRateLimited, RpcUnavailable) as exc:
            _reraise_if_telemetry_skip(exc)
            raise StateUnavailable(
                f"the market account could not be read: {exc}",
                failure=FailureMode.RPC_RATE_LIMIT
                if isinstance(exc, RpcRateLimited)
                else FailureMode.STATE_UNAVAILABLE,
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise StateUnavailable(
                f"reading the market at {address} failed: "
                f"{type(exc).__name__}: {exc}",
                failure=FailureMode.STATE_UNAVAILABLE,
            ) from exc
        fetched = time.time()
        latency = (time.monotonic() - started) * 1000.0

        if account is None or account.is_empty:
            return None, StateSnapshot(
                slot=slot, fetched_at=fetched, fetch_latency_ms=latency,
                accounts_read=1,
                detail=f"no account at {address}"
            ), "UNKNOWN", "unknown"

        # Try bonding curve first
        if account.owner == PUMP_FUN_PROGRAM:
            self._verify_owner(account, PUMP_FUN_PROGRAM, address, "bonding curve")
            curve = decode_bonding_curve(
                account.raw(), curve_address=address, mint=mint, slot=slot,
                timestamp=fetched,
            )
            return curve, StateSnapshot(
                slot=slot, fetched_at=fetched, fetch_latency_ms=latency, accounts_read=1,
                owner_checks=((address, account.owner),),
            ), "PUMP_BONDING_CURVE", "pump_direct"

        # Try PumpSwap pool
        if account.owner == PUMP_SWAP_PROGRAM:
            self._verify_owner(account, PUMP_SWAP_PROGRAM, address, "pumpswap pool")
            raw = self._raw(account, "pumpswap pool")
            fields = decode_pool(raw, pool_address=address)
            base_ata = fields["pool_base_token_account"]
            quote_ata = fields["pool_quote_token_account"]
            base_mint_on_chain = fields["base_mint"]

            try:
                mint_account, base_ata_account, quote_ata_account = await self._accounts(
                    [base_mint_on_chain, base_ata, quote_ata]
                )
            except Exception as exc:  # noqa: BLE001
                raise StateUnavailable(
                    "the pool's own vault accounts could not be read "
                    f"({type(exc).__name__}: {exc}), so its reserves are UNKNOWN "
                    "rather than guessed",
                    failure=FailureMode.STATE_UNAVAILABLE,
                ) from exc
            for addr, acct, what in (
                (base_mint_on_chain, mint_account, "base mint"),
                (base_ata, base_ata_account, "pool base vault"),
                (quote_ata, quote_ata_account, "pool quote vault"),
            ):
                if acct is None or acct.is_empty:
                    raise StateUnavailable(
                        f"the {what} account {addr} named by pool {address} does not "
                        "exist, so the pool's reserves are UNKNOWN"
                    )
                if not is_token_program(acct.owner):
                    raise StateUnavailable(
                        f"the {what} account {addr} is owned by {acct.owner}, not an "
                        "SPL token program, so its bytes are not a token balance"
                    )

            if base_mint_on_chain != mint:
                raise StateUnavailable(
                    f"pool {address} has base mint {base_mint_on_chain[:8]}…, not "
                    f"{mint[:8]}…; it is a different token's pool and is refused"
                )
            if fields["quote_mint"] != quote_mint:
                raise StateUnavailable(
                    f"pool {address} is quoted in {fields['quote_mint'][:8]}…, not "
                    f"{quote_mint[:8]}…; it is a different book"
                )

            pool = decode_pumpswap_pool(
                raw,
                pool_address=address,
                base_mint_account=mint_account.raw(),
                base_token_account=base_ata_account.raw(),
                quote_token_account=quote_ata_account.raw(),
                slot=slot,
                timestamp=fetched,
            )
            return pool, StateSnapshot(
                slot=slot, fetched_at=fetched, fetch_latency_ms=latency, accounts_read=4,
                owner_checks=(
                    (address, account.owner),
                    (base_ata, base_ata_account.owner),
                    (quote_ata, quote_ata_account.owner),
                ),
            ), "PUMPSWAP", "pumpswap_direct"

        return None, StateSnapshot(
            slot=slot, fetched_at=fetched, fetch_latency_ms=latency, accounts_read=1,
            detail=f"account at {address} is not a recognized market type"
        ), "UNKNOWN", "unknown"

    async def read_pumpswap_pool(
        self,
        mint: str,
        quote_mint: str = WSOL_MINT,
        pool_address: str | None = None,
    ) -> tuple[PumpSwapPool | None, StateSnapshot, str, str]:
        """
        Locate, read and verify a PumpSwap pool for ``mint``.

        The DERIVED canonical pool is tried first. A caller-supplied address
        (from DexScreener or a Jupiter route) is used only as a fallback, and it
        is then VERIFIED against the chain rather than trusted.

        Order matters. On a pump venue the aggregator's pair address is the
        market account itself - the curve PDA before graduation, the pool after
        - so for an UNGRADUATED token the supplied address is the bonding curve.
        Probing that first would read a curve as a pool and report a confusing
        "wrong discriminator" instead of the true "this token has not graduated,
        so it has no pool yet".
        """
        started = time.monotonic()
        candidates: list[str] = []
        derived, _bump = derive_canonical_pool(mint, quote_mint)
        candidates.append(derived)
        if pool_address and pool_address != derived:
            candidates.append(pool_address)

        first_failure: StateUnavailable | None = None
        for index, address in enumerate(candidates):
            try:
                return await self._load_pool(mint, address, quote_mint)
            except StateUnavailable as exc:
                if index == 0:
                    first_failure = exc
                continue
        assert first_failure is not None
        raise first_failure

    async def _load_pool(
        self, mint: str, address: str, quote_mint: str,
    ) -> tuple[PumpSwapPool | None, StateSnapshot, str | None]:
        started = time.monotonic()
        try:
            slot = await self._slot()
            account = (await self._accounts([address]))[0]
        except (RpcRateLimited, RpcUnavailable) as exc:
            _reraise_if_telemetry_skip(exc)
            raise StateUnavailable(
                f"the pool account could not be read: {exc}",
                failure=FailureMode.RPC_RATE_LIMIT
                if isinstance(exc, RpcRateLimited)
                else FailureMode.STATE_UNAVAILABLE,
            ) from exc
        except Exception as exc:  # noqa: BLE001 - named, never escapes
            raise StateUnavailable(
                f"reading the pool at {address} failed: "
                f"{type(exc).__name__}: {exc}",
                failure=FailureMode.STATE_UNAVAILABLE,
            ) from exc
        fetched = time.time()
        latency = (time.monotonic() - started) * 1000.0

        if account is None or account.is_empty:
            return None, StateSnapshot(
                slot=slot, fetched_at=fetched, fetch_latency_ms=latency,
                accounts_read=1,
                detail=(
                    f"no account at {address}; a non-canonical or non-pump "
                    "token has no PumpSwap pool"
                ),
            ), address
        self._verify_owner(account, PUMP_SWAP_PROGRAM, address, "pumpswap pool")

        # The pool names its own vault accounts; they are then read directly
        # rather than inferred from the pool account.
        raw = self._raw(account, "pumpswap pool")
        fields = decode_pool(raw, pool_address=address)
        base_ata = fields["pool_base_token_account"]
        quote_ata = fields["pool_quote_token_account"]
        base_mint_on_chain = fields["base_mint"]

        try:
            mint_account, base_ata_account, quote_ata_account = await self._accounts(
                [base_mint_on_chain, base_ata, quote_ata]
            )
        except Exception as exc:  # noqa: BLE001 - named, never escapes
            raise StateUnavailable(
                "the pool's own vault accounts could not be read "
                f"({type(exc).__name__}: {exc}), so its reserves are UNKNOWN "
                "rather than guessed",
                failure=FailureMode.STATE_UNAVAILABLE,
            ) from exc
        for addr, acct, what in (
            (base_mint_on_chain, mint_account, "base mint"),
            (base_ata, base_ata_account, "pool base vault"),
            (quote_ata, quote_ata_account, "pool quote vault"),
        ):
            if acct is None or acct.is_empty:
                raise StateUnavailable(
                    f"the {what} account {addr} named by pool {address} does not "
                    "exist, so the pool's reserves are UNKNOWN"
                )
            if not is_token_program(acct.owner):
                raise StateUnavailable(
                    f"the {what} account {addr} is owned by {acct.owner}, not an "
                    "SPL token program, so its bytes are not a token balance"
                )

        if base_mint_on_chain != mint:
            raise StateUnavailable(
                f"pool {address} has base mint {base_mint_on_chain[:8]}…, not "
                f"{mint[:8]}…; it is a different token's pool and is refused"
            )
        if fields["quote_mint"] != quote_mint:
            raise StateUnavailable(
                f"pool {address} is quoted in {fields['quote_mint'][:8]}…, not "
                f"{quote_mint[:8]}…; it is a different book"
            )

        pool = decode_pumpswap_pool(
            raw,
            pool_address=address,
            base_mint_account=mint_account.raw(),
            base_token_account=base_ata_account.raw(),
            quote_token_account=quote_ata_account.raw(),
            slot=slot,
            timestamp=fetched,
        )
        return pool, StateSnapshot(
            slot=slot, fetched_at=fetched, fetch_latency_ms=latency, accounts_read=4,
            owner_checks=(
                (address, account.owner),
                (base_ata, base_ata_account.owner),
                (quote_ata, quote_ata_account.owner),
            ),
        ), address

    async def quote_pumpswap(
        self,
        mint: str,
        *,
        sol_amount_lamports: int,
        pool_address: str | None = None,
        quote_mint: str = WSOL_MINT,
    ) -> DirectQuote:
        """A direct ``SOL -> TOKEN`` quote against a verified PumpSwap pool."""
        provider = "pumpswap_direct"
        try:
            pool, snapshot, address = await self.read_pumpswap_pool(
                mint, pool_address=pool_address, quote_mint=quote_mint
            )
        except StateUnavailable as exc:
            return DirectQuote(
                provider=provider, market_type=MarketKind.PUMPSWAP, quote=None,
                state=None, snapshot=None, pool_address=pool_address,
                base_mint=mint, quote_mint=quote_mint, missing_reason=str(exc),
                failure=exc.failure,
            )
        if pool is None:
            return DirectQuote(
                provider=provider, market_type=MarketKind.PUMPSWAP, quote=None,
                state=None, snapshot=snapshot, pool_address=address,
                base_mint=mint, quote_mint=quote_mint,
                missing_reason=f"no PumpSwap pool found ({snapshot.detail})",
                failure=FailureMode.STATE_UNAVAILABLE,
            )
        started = time.monotonic()
        try:
            global_config, fee_config = await self._amm_fee_state()
            fees = amm_fee_schedule(
                pool, global_config=global_config, fee_config=fee_config,
                circulating_supply=pool.base_supply,
            )
            quote = quote_pumpswap_buy(
                pool, sol_amount_lamports=sol_amount_lamports, fees=fees,
                circulating_supply=pool.base_supply,
            )
        except StateUnavailable as exc:
            return DirectQuote(
                provider=provider, market_type=MarketKind.PUMPSWAP, quote=None,
                state=pool, snapshot=snapshot, pool_address=pool.address,
                base_mint=mint, quote_mint=pool.quote_mint,
                missing_reason=str(exc), failure=exc.failure,
                identity_evidence=_amm_identity(pool, pool_address, address),
            )
        return DirectQuote(
            provider=provider, market_type=MarketKind.PUMPSWAP, quote=quote,
            state=pool, snapshot=snapshot,
            quote_latency_ms=(time.monotonic() - started) * 1000.0,
            pool_address=pool.address,
            base_mint=pool.base_mint, quote_mint=pool.quote_mint,
            identity_evidence=_amm_identity(pool, pool_address, address),
        )

    # -- SELL direction ----------------------------------------------------

    async def quote_bonding_curve_sell(
        self,
        mint: str,
        *,
        token_amount_raw: int,
        token_decimals: int,
    ) -> DirectQuote:
        """
        A direct ``TOKEN -> SOL`` quote against the mint's bonding curve.

        Mirror of :meth:`quote_bonding_curve`, and it fails closed the same way:
        an unavailable curve yields ``UNKNOWN`` with a reason, never a zero and
        never a mark.
        """
        provider = "pump_direct"
        try:
            curve, snapshot = await self.read_bonding_curve(mint)
        except StateUnavailable as exc:
            return DirectQuote(
                provider=provider, market_type=MarketKind.PUMP_BONDING_CURVE,
                quote=None, state=None, snapshot=None, base_mint=mint,
                quote_mint=WSOL_MINT, missing_reason=str(exc), failure=exc.failure,
            )
        if curve is None:
            return DirectQuote(
                provider=provider, market_type=MarketKind.PUMP_BONDING_CURVE,
                quote=None, state=None, snapshot=snapshot, base_mint=mint,
                quote_mint=WSOL_MINT,
                missing_reason="no bonding curve exists for this mint",
                failure=FailureMode.CURVE_UNAVAILABLE,
            )
        if curve.graduated:
            return DirectQuote(
                provider=provider, market_type=MarketKind.PUMP_BONDING_CURVE,
                quote=None, state=curve, snapshot=snapshot,
                curve_address=curve.address, base_mint=mint, quote_mint=WSOL_MINT,
                missing_reason=(
                    "the bonding curve is complete; there is no curve to sell into"
                ),
                failure=FailureMode.CURVE_UNAVAILABLE,
            )
        started = time.monotonic()
        try:
            glob, fee_config = await self._pump_fee_state()
            supply = await self._mint_supply(mint)
            if supply is None:
                raise StateUnavailable(
                    "the mint's circulating supply could not be read and the "
                    "curve's fee tier depends on it; the tier is NOT guessed"
                )
            fees = curve_fee_schedule(
                curve, global_config=glob, fee_config=fee_config, mint_supply=supply
            )
            quote = quote_bonding_curve_sell(
                curve, token_amount_raw=token_amount_raw,
                token_decimals=token_decimals, fees=fees,
            )
        except StateUnavailable as exc:
            return DirectQuote(
                provider=provider, market_type=MarketKind.PUMP_BONDING_CURVE,
                quote=None, state=curve, snapshot=snapshot,
                curve_address=curve.address, base_mint=mint, quote_mint=WSOL_MINT,
                missing_reason=str(exc), failure=exc.failure,
            )
        return DirectQuote(
            provider=provider, market_type=MarketKind.PUMP_BONDING_CURVE,
            quote=quote, state=curve, snapshot=snapshot,
            quote_latency_ms=(time.monotonic() - started) * 1000.0,
            curve_address=curve.address, base_mint=mint,
            quote_mint=WSOL_MINT if curve.quote_mint_is_sol_like else curve.quote_mint,
            identity_evidence=(
                f"the account at the PDA derived from seeds "
                f"['bonding-curve', {mint}] is owned by the pump.fun program "
                "and carries the BondingCurve discriminator"
            ),
        )

    async def quote_pumpswap_sell(
        self,
        mint: str,
        *,
        token_amount_raw: int,
        pool_address: str | None = None,
        quote_mint: str = WSOL_MINT,
    ) -> DirectQuote:
        """A direct ``TOKEN -> SOL`` quote against a verified PumpSwap pool."""
        provider = "pumpswap_direct"
        try:
            pool, snapshot, address = await self.read_pumpswap_pool(
                mint, pool_address=pool_address, quote_mint=quote_mint
            )
        except StateUnavailable as exc:
            return DirectQuote(
                provider=provider, market_type=MarketKind.PUMPSWAP, quote=None,
                state=None, snapshot=None, pool_address=pool_address,
                base_mint=mint, quote_mint=quote_mint, missing_reason=str(exc),
                failure=exc.failure,
            )
        if pool is None:
            return DirectQuote(
                provider=provider, market_type=MarketKind.PUMPSWAP, quote=None,
                state=None, snapshot=snapshot, pool_address=address,
                base_mint=mint, quote_mint=quote_mint,
                missing_reason="no PumpSwap pool found",
                failure=FailureMode.STATE_UNAVAILABLE,
            )
        started = time.monotonic()
        try:
            global_config, fee_config = await self._amm_fee_state()
            fees = amm_fee_schedule(
                pool, global_config=global_config, fee_config=fee_config,
                circulating_supply=pool.base_supply,
            )
            quote = quote_pumpswap_sell(
                pool, token_amount_raw=token_amount_raw, fees=fees,
            )
        except StateUnavailable as exc:
            return DirectQuote(
                provider=provider, market_type=MarketKind.PUMPSWAP, quote=None,
                state=pool, snapshot=snapshot, pool_address=pool.address,
                base_mint=mint, quote_mint=pool.quote_mint,
                missing_reason=str(exc), failure=exc.failure,
                identity_evidence=_amm_identity(pool, pool_address, address),
            )
        return DirectQuote(
            provider=provider, market_type=MarketKind.PUMPSWAP, quote=quote,
            state=pool, snapshot=snapshot,
            quote_latency_ms=(time.monotonic() - started) * 1000.0,
            pool_address=pool.address, base_mint=pool.base_mint,
            quote_mint=pool.quote_mint,
            identity_evidence=_amm_identity(pool, pool_address, address),
        )

    # -- combined ----------------------------------------------------------

    async def measure(
        self,
        mint: str,
        *,
        sol_amount_lamports: int,
        token_decimals: int,
        pool_address: str | None = None,
        dex_id: str | None = None,
        mark: Any = None,
    ) -> tuple[MarketClassification, DirectQuote, DirectQuote]:
        """
        Classify the market and take BOTH direct quotes where applicable.

        Returns the classification, the Pump curve quote and the PumpSwap pool
        quote. Both are attempted even after the market is classified, because
        "PUMPSWAP" as a label can be wrong (a graduated coin still carries a
        ``pumpfun`` label) and the classification here comes from the chain.

        The caller decides which one is the market; this reports both so the
        comparison is possible.
        """
        if mark is not None:
            mark("pump_direct")
        try:
            curve, curve_snapshot = await self.read_bonding_curve(mint)
        except StateUnavailable as exc:
            # measure() reports; it does not raise. An unreadable market must
            # not abort the OTHER sources, or one dead endpoint would cost the
            # whole comparison.
            curve, curve_snapshot = None, StateSnapshot(
                slot=None, fetched_at=time.time(), fetch_latency_ms=None,
                accounts_read=0, failure=exc.failure, detail=str(exc),
            )
        curve_failure: DirectQuote | None = None
        if curve is not None and not curve.graduated:
            curve_failure = await self.quote_bonding_curve(
                mint, sol_amount_lamports=sol_amount_lamports,
                token_decimals=token_decimals,
            )
        else:
            curve_failure = DirectQuote(
                provider="pump_direct", market_type=MarketKind.PUMP_BONDING_CURVE,
                quote=None, state=curve, snapshot=curve_snapshot,
                curve_address=curve.address if curve else None, base_mint=mint,
                missing_reason=(
                    "the bonding curve is complete; it is no longer the market"
                    if curve is not None
                    else "no bonding curve exists for this mint"
                ),
                failure=FailureMode.CURVE_UNAVAILABLE,
            )

        if mark is not None:
            mark("pump_direct")
            mark("pumpswap_direct")
        amm = await self.quote_pumpswap(
            mint, sol_amount_lamports=sol_amount_lamports,
            pool_address=pool_address,
        )  # already returns a failed DirectQuote rather than raising
        if mark is not None:
            mark("pumpswap_direct")
        classification = classify(
            curve=curve,
            pool=amm.state if isinstance(amm.state, PumpSwapPool) else None,
            dex_id=dex_id,
        )
        return classification, curve_failure, amm


def _amm_identity(pool: PumpSwapPool, supplied: str | None, resolved: str | None) -> str:
    parts = [
        f"account {pool.address} is owned by the PumpSwap program and carries "
        "the Pool discriminator"
    ]
    if supplied and resolved and supplied == resolved:
        parts.append(
            "its address is the one the aggregator named, so the aggregator's "
            "pool and this pool are the same account"
        )
    elif supplied:
        parts.append(
            f"the aggregator named {supplied[:8]}… but the verified pool is "
            f"{pool.address[:8]}…"
        )
    parts.append(
        f"its base mint field equals the token under test and its two vault "
        "accounts were read directly for the reserves"
    )
    return "; ".join(parts)


__all__ = ["DirectPumpReader", "DirectQuote", "StateSnapshot"]
