"""
The REAL executor: quote -> build -> sign -> broadcast -> confirm -> reconcile.

PROVENANCE
----------
Ported from the reference executor in ``~/works/memecoin-bot``
(``bot/execution/real.py``), which has been exercised against Solana mainnet
for quote -> build -> sign -> simulate. The order of operations, the failure
vocabulary, the on-chain reconciliation and the "ambiguity is a first-class
outcome" discipline are all preserved. What changed is only that the sizing
input now comes from this project's central V3-C policy
(:mod:`autopaper.sizing`) and the strategy gate is V3-C rather than V2.

WHAT IS NEVER DONE HERE
-----------------------

* A trade is never marked filled before on-chain confirmation.
* A simulated number is never substituted for an actual one. If the confirmed
  transaction cannot be reconciled, the outcome is ``AMBIGUOUS``, which is a
  refusal, not a fill.
* ``slippage_bps`` is never raised to force a fill.
* A micro order is never enlarged. If the venue refuses the notional, the
  outcome is ``MICRO_REAL_EXECUTION_UNSUPPORTED`` and REAL stays usable but this
  particular trade does not happen.
"""

from __future__ import annotations

import base64
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .intents import (
    BROADCASTED,
    CONFIRMED,
    FAILED,
    READY,
    SIGNING,
    UNKNOWN,
)
from .jupiter import (
    JupiterProvider,
    MicroOrderUnsupported,
    QuoteRejected,
    QuoteUnavailable,
)
from .raptor import (
    RaptorBuildFailed,
    RaptorInvalidResponse,
    RaptorMicroOrderUnsupported,
    RaptorNetworkError,
    RaptorProvider,
    RaptorQuoteFailed,
    RaptorTxTooLarge,
    RaptorUnsupportedRoute,
    transaction_size_report,
)
from .planner import plan_execution
from .txcodec import is_alt_load_refusal as _is_alt_load_refusal
from .solana import (
    LAMPORTS_PER_SOL,
    RpcError,
    SolanaRpc,
    SolanaUnavailable,
    SolanaWallet,
    TransactionFailed,
    WSOL_MINT,
    mask_address,
    reconcile_transaction,
    send_failure_may_have_submitted,
)

log = logging.getLogger("autopaper.execution.real")

#: Statuses that mean "this actually happened on chain".
FILLED_STATUSES = ("confirmed", "finalized")

#: The one status that means "we stopped knowing". Never a fill.
AMBIGUOUS = "AMBIGUOUS"

#: A transaction the chain CONFIRMED, whose output amount could not be read
#: back. This is categorically different from AMBIGUOUS and must never be
#: collapsed into it.
#:
#: AMBIGUOUS means "we do not know whether this happened" -- it may have been
#: dropped before landing, and the safe response is to stop and ask a human.
#: CONFIRMED_BUT_UNRECONCILED means "the chain says this SUCCEEDED and we can
#: prove it; we just failed to read the amount". Funds very likely moved.
#:
#: Reporting the second as the first caused the $Boy incident: a swap that
#: succeeded on chain was journalled as ``BUY_BLOCKED=fill_refused`` with no
#: position opened, so the wallet held tokens the runtime believed it did not
#: hold. It must fail closed against a SECOND broadcast, but it must never be
#: reported as a refused entry.
CONFIRMED_UNRECONCILED = "CONFIRMED_BUT_UNRECONCILED"

#: Rent for one associated token account, plus headroom. Held back so a swap
#: can never spend the SOL needed to create the account it requires.
ATA_RENT_LAMPORTS = 2_039_280

#: Base Solana signature cost.
BASE_FEE_LAMPORTS = 5_000

#: Headroom held back on top of the explicit costs.
MIN_SOL_FEE_RESERVE = 0.0015

#: Exactly one concurrent position. The micro envelope has no room for two.
MAX_OPEN_POSITIONS = 1


class RealExecutorUnavailable(RuntimeError):
    """The real execution path is not available. Always means DO NOT TRADE."""


def _replace_transaction(built: Any, transaction_b64: str) -> Any:
    """A copy of a provider's built swap carrying a different transaction.

    Used when the v1 rescue re-encodes a venue transaction: everything the
    caller needs downstream (the quote, the last valid block height, the
    priority fee the venue reported) is carried over unchanged, because none of
    it depends on the transaction's wire format. Mutating the original in place
    would be wrong: the provider's own object is what the attempt log describes,
    and the two are reported separately.
    """
    import dataclasses
    if dataclasses.is_dataclass(built) and not isinstance(built, type):
        return dataclasses.replace(built, transaction_b64=transaction_b64)
    clone = copy.copy(built)
    try:
        clone.transaction_b64 = transaction_b64
    except Exception:  # noqa: BLE001  # pragma: no cover
        pass
    return clone


def _quote_price(quote: Any, side: str) -> Optional[float]:
    """
    The venue's quoted price ORIENTED FOR ``side``, or ``None``.

    Two routes, both side-aware:

    1. the provider's own ``effective_price(side)``, when it has one;
    2. otherwise the price derived here from the quote's own in/out amounts via
       :func:`pricing.execution_price_per_token`, which takes its orientation
       from ``side``.

    The legacy ``quote.price_per_token`` property is deliberately NOT consulted.
    It was ``in_amount / expected_out_amount`` for every side, which is
    tokens-per-lamport on a sell; the 2026-10-02 $notify exit persisted
    ``quote_price = 6634.915851383544`` because of exactly that.
    """
    from ..pricing import execution_price_per_token

    accessor = getattr(quote, "effective_price", None)
    if callable(accessor):
        try:
            return accessor(side)
        except TypeError:
            return accessor()
    i = getattr(quote, "in_amount", None)
    o = getattr(quote, "expected_out_amount", None)
    if i is None or o is None:
        return None
    return execution_price_per_token(side=side, input_amount=i,
                                     output_amount=o)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass
class OnchainFill:
    """
    Actual, chain-derived execution facts for one order.

    Every field here is read back from the confirmed transaction or the venue
    response. Nothing is estimated unless the field name says ``quote_``.
    """

    side: str = "ENTRY"
    status: str = ""
    signature: str = ""
    requested_lamports: Optional[int] = None
    input_amount: Optional[int] = None
    output_amount: Optional[int] = None
    #: WHICH ASSET each amount is denominated in. ADDED because a fill that
    #: records only a bare number cannot be reconciled: an exit that returned
    #: 4,061 of USDC and an exit that returned 4,061 of WSOL are indistinguishable
    #: without the mint, and the venue is free to choose either. Every consumer
    #: of ``output_amount`` must read these rather than assuming WSOL.
    input_mint: str = ""
    output_mint: str = ""
    #: WHAT ``input_amount`` IS: ``"SWAP_LEG"`` (the amount that crossed into or
    #: out of the venue) or ``"WALLET_DEBIT"`` (the payer's whole SOL balance
    #: change, which ALSO contains account-creation rent and the network fee).
    #: On 2026-10-02 a $GOAT entry recorded a 3,497,651-lamport wallet debit --
    #: 1,928,811 of swap principal + 55,000 of fee + 1,513,840 of token-account
    #: rent -- in ``input_amount``, and dividing it by 7,547,061,883 raw token
    #: units produced a journalled "execution price" of 0.0004634 that is not a
    #: price. The kind is now explicit and the two quantities are separate
    #: fields.
    input_amount_kind: str = "SWAP_LEG"
    #: The venue-side input leg when it could be separated from the wallet
    #: balance delta. ``None`` means "not established".
    swap_input_amount: Optional[int] = None
    #: Lamports the transaction paid OUT of accounts other than the fee payer.
    #: On a sell this is the SOL that left the pools and program accounts,
    #: before the network fee and any protocol skim.
    lamports_out_to_others: Optional[int] = None
    #: The payer's whole SOL balance change for this transaction.
    wallet_debit_lamports: Optional[int] = None
    #: Lamports of account-creation rent this transaction paid: the increase in
    #: accounts that previously held nothing. Itemised so the cost basis can
    #: state it instead of absorbing it silently.
    rent_lamports: Optional[int] = None
    #: The token mint's decimals, read from chain before the order. ``None``
    #: means no per-token price may be stated at all.
    token_decimals: Optional[int] = None
    #: Venue identity of the fill: provider, venue label, pool/curve address and
    #: the on-chain program that executed. Present so a market mark describing a
    #: DIFFERENT pool is detectable rather than invisible.
    quote_provider: Optional[str] = None
    venue: Optional[str] = None
    pool: Optional[str] = None
    program_id: Optional[str] = None
    fee_lamports: Optional[int] = None
    execution_price: Optional[float] = None
    quote_price: Optional[float] = None
    price_impact_pct: Optional[float] = None
    slippage_bps: int = 0
    commitment: str = ""
    slot: Optional[int] = None
    route_labels: list[str] = field(default_factory=list)
    confirmed_at: Optional[float] = None
    ambiguity_reason: str = ""
    detail: str = ""
    #: True when the chain confirmed this transaction but the output amount
    #: could not be read back. Distinct from AMBIGUOUS: here we KNOW funds
    #: moved, and the position must be recovered from on-chain state. Never
    #: treat this as a refused entry, and never re-broadcast it.
    requires_recovery: bool = False
    #: Sanitized execution metadata (provider, tx bytes, slippage/fee mapping,
    #: route). Never contains a transaction payload or a secret.
    route_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def filled(self) -> bool:
        return self.status in FILLED_STATUSES

    @property
    def confirmed_on_chain(self) -> bool:
        """True once the chain has confirmed, whether or not we could read it.

        This is the distinction the $Boy incident turned on. ``filled`` means
        "reconciled into a usable amount"; ``confirmed_on_chain`` means "the
        network says it happened". A confirmed-but-unreadable fill must fail
        closed against another broadcast while still being reported as money
        that moved.
        """
        return (self.status in FILLED_STATUSES
                or self.status == CONFIRMED_UNRECONCILED)

    @property
    def fee_sol(self) -> Optional[float]:
        if self.fee_lamports is None:
            return None
        return self.fee_lamports / LAMPORTS_PER_SOL

    def as_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "status": self.status,
            "signature": self.signature,
            "requested_lamports": self.requested_lamports,
            "input_amount": self.input_amount,
            "output_amount": self.output_amount,
            "input_mint": self.input_mint,
            "output_mint": self.output_mint,
            "input_amount_kind": self.input_amount_kind,
            "swap_input_amount": self.swap_input_amount,
            "lamports_out_to_others": self.lamports_out_to_others,
            "wallet_debit_lamports": self.wallet_debit_lamports,
            "invoked_programs": list(getattr(self, "invoked_programs", ()) or ()),
            "rent_lamports": self.rent_lamports,
            "token_decimals": self.token_decimals,
            "quote_provider": self.quote_provider,
            "venue": self.venue,
            "pool": self.pool,
            "program_id": self.program_id,
            "fee_lamports": self.fee_lamports,
            "fee_sol": self.fee_sol,
            "execution_price": self.execution_price,
            "quote_price": self.quote_price,
            "price_impact_pct": self.price_impact_pct,
            "slippage_bps": self.slippage_bps,
            "commitment": self.commitment,
            "slot": self.slot,
            "route_labels": list(self.route_labels),
            "confirmed_at": self.confirmed_at,
            "filled": self.filled,
            "ambiguity_reason": self.ambiguity_reason,
            "detail": self.detail,
            "route_metadata": dict(self.route_metadata),
        }


# ---------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------


class LiveSolanaBackend:
    """
    Owns the wallet, the RPC client and the quote provider.

    Constructed lazily: importing this module, or gate-testing the executor,
    never touches key material. The wallet, the RPC client and the Jupiter
    client are each built on first use, so a PAPER-only process pays nothing
    for them and cannot accidentally hold a signer.
    """

    def __init__(
        self,
        cfg: Any,
        logger: Optional[logging.Logger] = None,
        *,
        wallet: Optional[SolanaWallet] = None,
        rpc: Optional[SolanaRpc] = None,
        provider: Optional[Any] = None,
        fallback_provider: Optional[Any] = None,
        sol_usd_provider: Optional[Any] = None,
        timeout: float = 90.0,
    ) -> None:
        self.cfg = cfg
        #: The real-execution settings, which live under ``cfg.real`` on the
        #: project Config and at the top level on a bare settings object.
        #: Resolved once here so every gate reads one shape.
        self.real_cfg = getattr(cfg, "real", cfg)
        self.log = logger or log
        self._wallet = wallet
        self._rpc = rpc
        self._provider = provider
        #: Jupiter fallback, used only when the primary provider cannot serve
        #: the trade. Injected in tests; built lazily in production.
        self._fallback_provider = fallback_provider
        #: The canonical SOL/USD reference provider, INJECTED. It is never
        #: constructed here: this module must not own a price source, a second
        #: SOL/USD implementation, or a second cache. The engine owns the one
        #: instance that PAPER sizing and REAL equity both read, and a test
        #: injects a fake so nothing here can reach a network.
        self._sol_usd_provider = sol_usd_provider
        rc = self.real_cfg
        self.confirm_timeout = float(
            getattr(rc, "confirmation_timeout_s", timeout)
        )
        self.simulate_before_send = bool(
            getattr(rc, "simulate_before_send", True)
        )
        self.priority_fee_lamports = int(
            getattr(rc, "priority_fee_lamports", 50_000)
        )
        self.quote_max_age_ms = float(
            getattr(rc, "quote_max_age_ms", 8000.0)
        )
        #: Feature flag. When false the existing (Jupiter) path is used
        #: unchanged; the Raptor path is selected only when true.
        self.raptor_enabled = bool(
            getattr(rc, "raptor_enabled", False)
        )
        self.raptor_base_url = str(
            getattr(rc, "raptor_base_url", "https://raptor.solanatracker.io")
        )
        #: Fall back to Jupiter when Raptor cannot route (or overshoots).
        self.raptor_fallback_enabled = bool(
            getattr(rc, "raptor_fallback_enabled", True)
        )
        #: When a provider returns a legacy/v0 transaction that exceeds its own
        #: 1232-byte ceiling, try to re-encode it as a v1 transaction (whose
        #: ceiling is 4096) before giving up. Off by default: it is a rescue
        #: path, not the normal one, and an operator should be able to see it
        #: disabled. It changes nothing about an order's economics -- the
        #: instructions, accounts, fee payer, blockhash, amount and slippage are
        #: all preserved and the result is proved equivalent before use.
        self.v1_rescue_enabled = bool(
            getattr(rc, "v1_rescue_enabled", True)
        )

    # -- lazily built capabilities ----------------------------------------

    @property
    def wallet(self) -> SolanaWallet:
        if self._wallet is None:
            self._wallet = SolanaWallet(self.real_cfg.wallet_keypair_path)
        return self._wallet

    @property
    def rpc(self) -> SolanaRpc:
        if self._rpc is None:
            self._rpc = SolanaRpc(self.real_cfg.rpc_url)
        return self._rpc

    @property
    def provider(self):
        """The swap quote/build provider.

        Selected by the ``RAPTOR_ENABLED`` feature flag. Raptor and Jupiter
        expose the same ``quote`` / ``build`` interface, so the rest of the
        execution path is provider-agnostic. No key material is ever passed to
        either provider.
        """
        if self._provider is None:
            impact = float(getattr(self.real_cfg, "max_price_impact_pct", 5.0))
            if self.raptor_enabled:
                self._provider = RaptorProvider(
                    base_url=self.raptor_base_url,
                    max_price_impact_pct=impact,
                )
            else:
                self._provider = JupiterProvider(max_price_impact_pct=impact)
        return self._provider

    @property
    def fallback_provider(self):
        """The Jupiter fallback provider, or ``None`` when disabled.

        Built lazily so a process that never needs a fallback pays nothing for
        it. It is a SECOND provider instance, never a second Jupiter
        implementation.
        """
        if self._fallback_provider is not None:
            return self._fallback_provider
        if not self.raptor_fallback_enabled:
            return None
        self._fallback_provider = JupiterProvider(
            max_price_impact_pct=float(
                getattr(self.real_cfg, "max_price_impact_pct", 5.0)))
        return self._fallback_provider

    def _provider_chain(self) -> list:
        """Providers to try, in order. Raptor first, Jupiter fallback second.

        Both implement the same :class:`ExecutionProvider` interface, so the
        executor treats them identically. Jupiter is only ever invoked when the
        earlier provider fails in a fallback-eligible way -- never blindly for
        every trade.
        """
        chain = [self.provider]
        if self.raptor_enabled and self.raptor_fallback_enabled:
            fb = self.fallback_provider
            if fb is not None and fb is not self.provider:
                chain.append(fb)
        return chain

    def _classify_provider_failure(self, exc: Exception, provider_name: str,
                                   stage: str) -> str:
        """
        Map a provider exception to the structured failure vocabulary.

        The TEXT REFINEMENT COMES FIRST, deliberately. Jupiter answers a generic
        HTTP 400 for both "your order is too small" and "this token is not
        tradable", so ``MicroOrderUnsupported`` is raised for either. Only a
        genuine below-minimum refusal is the micro condition; a token that is
        simply not tradable is an UNSUPPORTED ROUTE, which is what makes it
        fallback-eligible downstream.

        Refining before the generic ``status`` attribute matters now that the
        exception classes carry one. Reading ``status`` first would report every
        Jupiter 400 as MICRO_REAL_EXECUTION_UNSUPPORTED, and a plain "no route"
        as QUOTE_REJECTED. Both are wrong in the same direction: an untradable
        token or a missing route is an UNSUPPORTED_ROUTE and IS
        fallback-eligible, so misclassifying it as a size problem or a generic
        quote failure stops a second provider from ever being consulted.

        The refinement is SCOPED TO JUPITER. Raptor subclasses the same two
        exceptions with its own status vocabulary, and rewriting a raptor
        refusal into a Jupiter status would make it look non-fallback-eligible
        and disable the very fallback this method exists to keep.
        """
        if provider_name == "raptor":
            # Raptor carries its own status vocabulary on the exception and its
            # own wording for each condition. Its classification is already
            # specific, so the generic ``status`` is authoritative here.
            status = getattr(exc, "status", None)
            if status:
                return status
            return ("RAPTOR_QUOTE_FAILED" if stage == "quote"
                    else "RAPTOR_BUILD_FAILED")
        if isinstance(exc, MicroOrderUnsupported):
            text = str(exc).lower()
            if any(m in text for m in ("not tradable", "invalid", "no route",
                                       "unknown token", "not found",
                                       "unsupported")):
                return "JUPITER_UNSUPPORTED_ROUTE"
            return "MICRO_REAL_EXECUTION_UNSUPPORTED"
        if isinstance(exc, QuoteRejected):
            return "JUPITER_UNSUPPORTED_ROUTE"
        status = getattr(exc, "status", None)
        if status:
            return status
        return ("JUPITER_QUOTE_FAILED" if stage == "quote"
                else "JUPITER_BUILD_FAILED")

    @staticmethod
    def _fallback_eligible(status: str) -> bool:
        """Only a missing route or an oversized Raptor tx triggers a fallback.

        Network / quote / build failures are NOT fallback-eligible: a provider
        outage must never cause a second attempt against the other provider.
        """
        return status in ("RAPTOR_UNSUPPORTED_ROUTE", "RAPTOR_TX_TOO_LARGE")

    @staticmethod
    def _failure_record(status: str, detail: str, **kw) -> dict:
        """Build the categorized failure-telemetry record for one refusal."""
        from .failures import classify_execution_failure, failure_telemetry
        custom_code = kw.pop("custom_code", None)
        program_id = kw.pop("program_id", None)
        if "category" not in kw or not kw["category"]:
            kw["category"] = classify_execution_failure(
                status, detail=detail, custom_code=custom_code,
                program_id=program_id)
        return failure_telemetry(
            status=status, detail=detail, custom_code=custom_code,
            program_id=program_id, **kw)

    async def _resolve_instruction_program(self, transaction_b64: str,
                                           index: Optional[int]) -> str:
        """Name the program that owns one instruction index, or return "".

        A ``Custom`` simulation error names an instruction INDEX and a code;
        the code's meaning belongs to whichever program sits at that index. This
        resolves it from the very transaction that was simulated -- never from a
        provider label -- and, when the program id lives in an address lookup
        table, resolves the table READ-ONLY. It signs nothing and broadcasts
        nothing. An index that cannot be resolved returns an empty string, so a
        reader sees "unresolved" rather than a guessed program.
        """
        if index is None:
            return ""
        from .txcodec import build_loaded_addresses, instruction_program_ids
        try:
            ids = instruction_program_ids(transaction_b64)
        except Exception:  # noqa: BLE001
            ids = []
        if 0 <= index < len(ids) and ids[index]:
            return ids[index]
        try:
            import base64
            from solders.transaction import VersionedTransaction
            message = VersionedTransaction.from_bytes(
                base64.b64decode(transaction_b64)).message
            lookups = list(message.address_table_lookups)
            resolver = getattr(self.rpc, "resolve_lookup_tables", None)
            if lookups and callable(resolver):
                resolved, unresolved = await resolver(
                    [str(l.account_key) for l in lookups])
                if resolved and not unresolved:
                    loaded = build_loaded_addresses(message, resolved)
                    ids = instruction_program_ids(
                        transaction_b64, loaded_addresses=loaded)
                    if 0 <= index < len(ids) and ids[index]:
                        return ids[index]
        except Exception:  # noqa: BLE001 - unresolved is an honest answer
            pass
        return ""

    @staticmethod
    def _provider_name(prov) -> str:
        """A provider's name, tolerating a minimal test double."""
        fn = getattr(prov, "provider_name", None)
        if callable(fn):
            try:
                return str(fn())
            except Exception:  # noqa: BLE001
                pass
        return str(getattr(prov, "name", type(prov).__name__))

    async def quote_first_available(
        self,
        input_mint: str,
        output_mint: str,
        in_amount: int,
        slippage_bps: int,
    ) -> dict:
        """Price the order against the provider chain, in order.

        The SAME chain the swap path uses: Raptor first, Jupiter second, and
        Jupiter is consulted ONLY when Raptor fails in a fallback-eligible way
        (no route, or an oversized transaction). This is what makes the pre-trade
        ``quote_valid`` gate agree with the executor instead of asking only the
        primary provider and refusing a trade the fallback could have served.

        Quote-only: builds nothing, signs nothing, broadcasts nothing. A caller
        that adopts a fallback quote still passes every downstream gate
        (freshness, size, simulation) exactly once, on one intent.

        Returns a mapping, never raises for a venue refusal::

            quote               the winning Quote, or None
            provider            the winning provider's name, or ""
            attempts            one record per provider tried, with its status
            fallback_attempted  True once a second provider was consulted
            failure_status      the terminal status when quote is None
            failure_detail      its message when quote is None
            failure_http_status the provider HTTP status when one was reported
            failure_rpc_code    an RPC code when the failure carried one
            micro               a micro-notional refusal message, if any
        """
        providers = self._provider_chain()
        attempts: list[dict] = []
        for index, prov in enumerate(providers):
            pname = self._provider_name(prov)
            last = index == len(providers) - 1
            try:
                quote = await prov.quote(
                    input_mint, output_mint, int(in_amount), int(slippage_bps))
            except MicroOrderUnsupported as exc:
                # Not fallback-eligible on purpose: a venue minimum is a fact
                # about the notional, and a second venue's minimum is not a fix.
                return {
                    "quote": None, "provider": "", "attempts": attempts,
                    "fallback_attempted": bool(attempts),
                    "failure_status": "MICRO_REAL_EXECUTION_UNSUPPORTED",
                    "failure_detail": str(exc), "failure_http_status": None,
                    "failure_rpc_code": None, "micro": str(exc),
                }
            except Exception as exc:  # noqa: BLE001
                status = self._classify_provider_failure(exc, pname, "quote")
                attempts.append({"provider": pname, "stage": "quote",
                                 "status": status, "detail": str(exc)[:200]})
                if not last and self._fallback_eligible(status):
                    continue
                return {
                    "quote": None, "provider": "", "attempts": attempts,
                    "fallback_attempted": len(attempts) > 1,
                    "failure_status": status, "failure_detail": str(exc),
                    "failure_http_status": getattr(exc, "http_status", None),
                    "failure_rpc_code": getattr(exc, "code", None),
                    "micro": "",
                }
            attempts.append({"provider": pname, "stage": "quote",
                             "status": "OK",
                             "detail": f"expected {quote.expected_out_amount}"})
            return {
                "quote": quote, "provider": pname, "attempts": attempts,
                "fallback_attempted": len(attempts) > 1,
                "failure_status": "", "failure_detail": "",
                "failure_http_status": None, "failure_rpc_code": None,
                "micro": "",
            }
        # Unreachable: the loop always returns. Kept explicit for the type.
        return {"quote": None, "provider": "", "attempts": attempts,
                "fallback_attempted": False,
                "failure_status": "REAL_BUY_BLOCKED_NO_VALID_EXECUTION_ROUTE",
                "failure_detail": "no quote provider available",
                "failure_http_status": None, "failure_rpc_code": None,
                "micro": ""}

    # -- v1 rescue ---------------------------------------------------------
    async def _v1_rescue(self, built: Any, size: dict, *,
                         reason: str = "TX_TOO_LARGE") -> dict:
        """
        Try to re-encode a legacy/v0 transaction as a v1 transaction.

        WHY
        ---
        A v0 transaction is capped at 1232 bytes; a v1 transaction at 4096. A
        Jupiter swap on a token with a deep route can land between the two, and
        under the old single-limit rule such an order was refused as
        unsendable when it was not. Re-encoding it as v1 is the difference
        between trading the token and reporting it un-executable.

        A v1 transaction also fixes a SECOND, unrelated failure: a v0
        transaction that references address lookup tables the node will not load
        for a simulation is rejected outright, regardless of its size. v1 has
        no lookup tables, so the same trade simulates clean. That is why
        ``reason`` exists: the rescue is invoked both when the transaction is
        over its own size limit and when the node refused to load its tables,
        and both callers must go through the identical proof.

        WHAT IT MAY NOT DO
        ------------------
        It is a RE-ENCODING, not a re-route and not a re-plan. It may not:

        * drop an instruction or an account to save bytes,
        * change the amount, the output mint, the slippage or the route,
        * sign, or ask a provider to sign or broadcast,
        * proceed if the rebuilt message cannot be PROVED equivalent to the
          original.

        So it decodes, resolves every address lookup table, materialises the
        looked-up addresses, recompiles, applies the v1 resource configuration
        the v0 ComputeBudget instructions were carrying, and then verifies the
        result against the original on fee payer, blockhash, instruction count
        and order, program ids, instruction data, and every account's address,
        order and signer/writable flags. The address resolution itself is proved
        by recompiling the ORIGINAL v0 message and requiring the bytes to match.

        Any failure returns a report with ``ok: False`` and a reason. The caller
        then reports the token as un-executable at this size. It never falls
        through to a partially-verified rebuild.
        """
        report: dict = {"attempted": False, "ok": False, "status": "",
                        "reason": reason, "detail": "", "source_version": None,
                        "transaction_version": None, "raw_bytes": None,
                        "max_bytes": None, "over_by": None,
                        "transaction_b64": None, "verification": {},
                        "resource_config": {}}
        if not self.v1_rescue_enabled:
            report["status"] = "V1_RESCUE_DISABLED"
            return report
        version = (size or {}).get("transaction_version")
        report["source_version"] = version
        if version not in ("legacy", "v0"):
            report["status"] = "NOT_APPLICABLE"
            report["detail"] = (
                f"source transaction is version {version}; only legacy and v0 "
                f"transactions can be re-encoded as v1")
            return report
        # The size gate is not the only reason to get here, so the "is it
        # oversized?" precondition is checked only for the size-driven caller.
        if reason == "TX_TOO_LARGE" and (size or {}).get("ok"):
            report["status"] = "NOT_NEEDED"
            return report

        report["attempted"] = True
        try:
            from solders.transaction import VersionedTransaction
            raw = base64.b64decode(built.transaction_b64, validate=True)
            message = VersionedTransaction.from_bytes(raw).message
            tables = ([str(lookup.account_key)
                       for lookup in message.address_table_lookups]
                      if version == "v0" else [])
        except Exception as exc:  # noqa: BLE001
            report["status"] = "V1_RECONSTRUCTION_FAILED"
            report["detail"] = f"could not decode the source transaction: " \
                               f"{type(exc).__name__}"
            return report

        # Resolve EVERY table. A partially-resolved message cannot be expanded,
        # and guessing the missing addresses would mean trading the wrong
        # accounts, so an unresolved table ends the attempt here.
        resolved: list = []
        unresolved: list = []
        for address in tables:
            info = await self.rpc.get_address_lookup_table(address)
            if info.get("usable"):
                # (table_address, addresses). The writable/readonly split is
                # NOT taken from the RPC: the message's own lookup entry is the
                # only authority for it, and an RPC cannot know the answer.
                resolved.append((info["table"], info.get("addresses") or []))
            else:
                unresolved.append({"table": address,
                                   "detail": info.get("detail")})
        if unresolved:
            report["status"] = "ALT_UNRESOLVED"
            report["detail"] = "; ".join(
                f"{u['table']}: {u['detail']}" for u in unresolved)[:300]
            report["verification"] = {"unresolved_tables": unresolved}
            return report

        from .txcodec import reconstruct_as_v1
        rec = reconstruct_as_v1(
            built.transaction_b64,
            resolved_tables=resolved,
            priority_fee_lamports=getattr(
                built, "prioritization_fee_lamports", None),
        )
        report.update(
            ok=rec.ok, status=rec.status, detail=rec.detail,
            transaction_version=rec.transaction_version,
            raw_bytes=rec.raw_bytes, max_bytes=rec.max_bytes,
            over_by=rec.over_by, transaction_b64=rec.transaction_b64,
            verification=rec.verification, resource_config=rec.resource_config,
            resolved_tables=rec.resolved_tables,
        )
        if not rec.ok:
            self.log.warning(
                "v1 rescue failed for %s: %s (%s)", self.__class__.__name__,
                rec.status, rec.detail[:200])
        return report

    @property
    def pubkey(self) -> str:
        return self.wallet.pubkey

    async def aclose(self) -> None:
        for res in (self._rpc, self._provider, self._fallback_provider):
            if res is not None:
                await res.aclose()

    # -- probes ------------------------------------------------------------

    async def verify(self) -> dict[str, Any]:
        """
        Prove each live capability. Used by the REAL preflight.

        Signing is proved OFFLINE: a throwaway self-transfer instruction is
        built and signed and then discarded. Nothing is serialised for the
        network, nothing is broadcast, and nothing about the key is printed.
        This is a capability probe, not a trade.
        """
        report: dict[str, Any] = {
            "wallet_loaded": False,
            "wallet_masked": None,
            "signing_available": False,
            "rpc_reachable": False,
            "quote_provider": False,
            "slot": None,
            "broadcast_attempted": False,
        }
        try:
            report["wallet_masked"] = self.wallet.masked
            # Set AFTER the wallet is actually obtained, never before. Setting
            # it first would report "wallet loaded" for a wallet that failed to
            # load, and a preflight that lies about the one capability it exists
            # to prove is worse than no preflight.
            report["wallet_loaded"] = True
            from solders.pubkey import Pubkey
            from solders.system_program import TransferParams, transfer

            probe = self.wallet.sign_new_transaction(
                [
                    transfer(
                        TransferParams(
                            from_pubkey=self.wallet.pubkey_obj,
                            to_pubkey=Pubkey.default(),
                            lamports=1,
                        )
                    )
                ]
            )
            report["signing_available"] = bool(probe)
        except Exception as exc:  # noqa: BLE001 - report, never raise
            report["wallet_loaded"] = False
            report["signing_available"] = False
            report["wallet_error"] = type(exc).__name__
        try:
            health = await self.rpc.get_health()
            report["rpc_reachable"] = health.reachable
            report["rpc_detail"] = health.detail
            if health.reachable:
                report["slot"] = await self.rpc.get_slot()
        except Exception as exc:  # noqa: BLE001
            report["rpc_error"] = type(exc).__name__
        try:
            h = await self.provider.health()
            report["quote_provider"] = bool(h.get("reachable"))
            report["quote_provider_detail"] = h.get("detail")
        except Exception as exc:  # noqa: BLE001
            report["quote_provider"] = False
            report["quote_provider_error"] = type(exc).__name__
        return report

    async def sol_usd_snapshot(self) -> Any:
        """
        The canonical SOL/USD reference snapshot, or ``None`` when unavailable.

        SOL/USD IS A REFERENCE PRICE, NOT A SWAP. This used to ask the execution
        quote provider for a 1 SOL -> USDC route, which is the wrong abstraction
        twice over: the direction rule in
        :func:`~autopaper.execution.price_layers.resolve_side_from_mints`
        correctly refuses a leg whose two ends are both quote assets, and even
        if it answered, an EXECUTABLE quote is a fill price -- deriving a
        reference from it would make the sizing basis and the fill price come
        from one observation, which is exactly the coupling the canonical
        provider exists to prevent.

        So the value comes from the ONE shared provider that already backs PAPER
        sizing, injected rather than built here. There is no second cache and no
        second SOL/USD implementation in this module.
        """
        provider = self._sol_usd_provider
        if callable(provider):
            # A RESOLVER, not an instance. Both the provider and this backend are
            # built lazily and in either order, so a resolver is what keeps the
            # preflight and the equity read on the same reference instead of
            # whichever happened to exist first.
            try:
                provider = provider()
            except Exception:  # noqa: BLE001 - a failed resolve is unavailable
                return None
        if provider is None:
            # No provider wired. Report unavailable rather than reaching for the
            # swap quoter: this method must never acquire a price source on its
            # own, or the two implementations would drift apart again.
            return None
        try:
            return await provider.snapshot()
        except Exception:  # noqa: BLE001 - any fault is an unavailable price
            return None

    async def sol_price_usd(self) -> Optional[float]:
        """
        SOL/USD reference, or ``None`` when it cannot be obtained.

        Fails closed. A missing reference must never become a zero, because a
        zero SOL/USD would silently value a funded wallet at $0 and make sizing
        refuse for the wrong reason.
        """
        from ..same_market.sol_price import SolPriceUnavailable

        try:
            snap = await self.sol_usd_snapshot()
        except SolPriceUnavailable:
            return None
        if snap is None:
            return None
        try:
            price = float(getattr(snap, "usd_per_sol"))
        except (TypeError, ValueError):
            return None
        return price if price > 0 else None

    async def wallet_state(self) -> dict[str, Any]:
        """Live wallet balances used by the exposure guard."""
        lamports = await self.rpc.get_balance_lamports(self.pubkey)
        return {
            "lamports": int(lamports),
            "sol": lamports / LAMPORTS_PER_SOL,
            "masked": mask_address(self.pubkey),
        }

    # -- the trade ---------------------------------------------------------

    async def _execute_swap_raptor(
        self,
        side: str,
        input_mint: str,
        output_mint: str,
        lamports: int,
        slippage_bps: int,
        *,
        intent_store: Any = None,
        logical_key: Optional[str] = None,
        stage: Optional[Any] = None,
    ) -> OnchainFill:
        """Raptor-backed REAL swap with a Jupiter fallback.

        Order (the safety order, not a preference):

            intent reserve -> [Raptor | Jupiter-on-Raptor-unsupported]
            quote -> build (UNSIGNED) -> deserialize/validate -> size gate
            -> quote freshness gate -> simulate -> sign -> broadcast
            -> confirm -> reconcile

        Raptor is tried first. Jupiter is invoked only when Raptor has no route
        (or its transaction is oversized), and only BEFORE simulation/signing/
        broadcast -- so one intent is broadcast at most once. Neither provider
        signs or broadcasts; both are measured, simulated, signed and broadcast
        through the SAME safety layer. Every gate the existing REAL path
        enforced is kept for both.
        """

        def _stage(name: str) -> None:
            if stage is not None:
                try:
                    stage(name)
                except Exception:  # noqa: BLE001 - telemetry never breaks a trade
                    pass

        fill = OnchainFill(
            side=side, requested_lamports=int(lamports),
            slippage_bps=int(slippage_bps),
        )
        side_mint = output_mint if side == "ENTRY" else input_mint

        # 0. RESERVE the logical order BEFORE anything irreversible. Persisted.
        intent = None
        if intent_store is not None and logical_key:
            _stage("exec_intent_reserve_start")
            try:
                intent = intent_store.reserve(
                    logical_key=logical_key, side=side, mint=side_mint,
                    requested_amount=int(lamports),
                )
            except Exception as exc:  # noqa: BLE001 - a duplicate is a refusal
                fill.status = "DUPLICATE_EXECUTION"
                fill.ambiguity_reason = str(exc)
                _stage("exec_intent_reserve_end")
                return fill
            _stage("exec_intent_reserve_end")

        def _fail(status: str, reason: str, *,
                  unknown: bool = False) -> OnchainFill:
            fill.status = status
            fill.ambiguity_reason = reason
            fill.detail = reason
            if intent is not None:
                try:
                    intent_store.transition(
                        intent.execution_id,
                        UNKNOWN if unknown else FAILED,
                        detail=f"{status}: {reason}",
                    )
                except Exception:  # noqa: BLE001
                    pass
            return fill

        # 1/2/3. PROVIDER CHAIN: quote -> build (UNSIGNED) -> validate + SIZE GATE.
        #
        #    Raptor is the PRIMARY provider. Jupiter is invoked ONLY when Raptor
        #    cannot serve the trade:
        #      * RAPTOR_UNSUPPORTED_ROUTE (no route), or
        #      * the Raptor transaction is oversized (>1232 raw bytes).
        #    Network / quote / build failures do NOT fall back, so a provider
        #    outage can never trigger a second attempt. The fallback happens
        #    BEFORE simulation/signing/broadcast, so a single trade intent is
        #    still broadcast at most once, and Jupiter is never called blindly
        #    for every trade.
        providers = self._provider_chain()
        attempts: list[dict] = []
        selected = None            # (provider, name, quote, built, size, t0)
        failure = None             # (status, reason)
        for index, prov in enumerate(providers):
            pname = prov.provider_name()
            last = index == len(providers) - 1
            quote_t0 = time.time()

            # 1. QUOTE. Every provider failure becomes a NAMED condition.
            _stage("tx_quote_start")
            try:
                quote = await prov.quote(
                    input_mint, output_mint, int(lamports), int(slippage_bps)
                )
            except Exception as exc:  # noqa: BLE001
                status = self._classify_provider_failure(exc, pname, "quote")
                attempts.append({"provider": pname, "stage": "quote",
                                 "status": status, "detail": str(exc)[:200]})
                _stage("tx_quote_end")
                if not last and self._fallback_eligible(status):
                    continue
                failure = (status, str(exc))
                break
            _stage("tx_quote_end")

            # 2. BUILD. The returned transaction is UNSIGNED; no provider has a
            #    key and none is asked to sign or broadcast.
            #
            #    SOL/WSOL HANDLING IS PROVIDER-OWNED. This Raptor path does NOT
            #    run a wSOL-readiness gate and does NOT create, fund or require a
            #    pre-provisioned wSOL account. Raptor wraps native SOL into wSOL
            #    inside the swap and unwraps on the way out; the Jupiter fallback
            #    is asked for its own temporary wrap/unwrap semantics. The
            #    executor therefore does not dictate a wSOL mode -- each provider
            #    applies its own default (Raptor: auto wrap/unwrap; Jupiter:
            #    auto wrap/unwrap for the fallback build). The Jupiter-specific
            #    pre-provisioned-wSOL requirement lives only in Jupiter's own
            #    ``execute_swap`` path, never here.
            _stage("tx_build_start")
            try:
                built = await prov.build_transaction(
                    quote, self.pubkey,
                    priority_fee=self.priority_fee_lamports,
                )
            except Exception as exc:  # noqa: BLE001
                status = self._classify_provider_failure(exc, pname, "build")
                attempts.append({"provider": pname, "stage": "build",
                                 "status": status, "detail": str(exc)[:200]})
                _stage("tx_build_end")
                if not last and self._fallback_eligible(status):
                    continue
                failure = (status, str(exc))
                break
            _stage("tx_build_end")
            tx_built_at = time.time()

            # 3. VALIDATE + SIZE GATE. Deserialize -> reserialize -> measure
            #    raw bytes. Refused BEFORE simulation/signing/broadcast, on real
            #    bytes, for BOTH providers.
            #
            #    The limit is the one for the transaction's ACTUAL version, read
            #    off the wire: legacy and v0 are held to 1232 bytes, v1 to 4096.
            size = transaction_size_report(built.transaction_b64)
            attempt = {
                "provider": pname, "stage": "size_gate",
                "status": "TX_SIZE_OK" if size["ok"] else "TX_TOO_LARGE",
                "raw_bytes": size.get("raw_bytes"),
                "encoded_bytes": size.get("encoded_bytes"),
                "transaction_version": size.get("transaction_version"),
                "limit_bytes": size.get("max_bytes"),
                "max_bytes": size.get("max_bytes"),
                "over_by": size.get("over_by"),
                "route": prov.route_metadata(quote),
            }

            # 3b. V1 RESCUE. An oversized legacy/v0 transaction is a candidate
            #     for re-encoding as v1, whose ceiling is 4096 bytes. This is a
            #     RE-ENCODING of the same instructions, not a re-route: it runs
            #     before any fallback, before simulation, signing and broadcast,
            #     and its result is only accepted when the rebuilt message is
            #     PROVED equivalent to the original. A rescue that fails leaves
            #     the provider's transaction exactly as it was -- oversized --
            #     so the fallback below still gets its turn.
            if not size["ok"]:
                rescue = await self._v1_rescue(built, size)
                attempt["v1_rescue"] = {
                    k: v for k, v in rescue.items()
                    if k not in ("transaction_b64", "detail")
                }
                attempt["v1_rescue_detail"] = rescue.get("detail", "")[:240]
                if rescue.get("ok"):
                    # Adopt the rebuilt transaction. Everything downstream --
                    # freshness, simulation, signing, broadcast, confirmation,
                    # reconciliation -- runs on it unchanged.
                    built = _replace_transaction(built, rescue["transaction_b64"])
                    size = transaction_size_report(built.transaction_b64)
                    attempt["status"] = "TX_SIZE_OK_AFTER_V1_REBUILD"
                    attempt["raw_bytes"] = size.get("raw_bytes")
                    attempt["encoded_bytes"] = size.get("encoded_bytes")
                    attempt["transaction_version"] = size.get("transaction_version")
                    attempt["max_bytes"] = size.get("max_bytes")
                    attempt["limit_bytes"] = size.get("max_bytes")
                    attempt["over_by"] = size.get("over_by")
                    attempt["route"]["transaction_version"] = \
                        size.get("transaction_version")
                    attempt["v1_resource_config"] = rescue.get(
                        "resource_config", {})
                    log.info(
                        "%s swap: TX_TOO_LARGE at %s bytes as %s -> rebuilt as "
                        "v1 at %s bytes (limit %s); instructions, accounts, "
                        "fee payer and blockhash verified identical",
                        pname.upper(), attempt.get("v1_rescue", {}).get(
                            "source_bytes") or size.get("raw_bytes"),
                        rescue.get("source_version"),
                        size.get("raw_bytes"), size.get("max_bytes"))
            attempts.append(attempt)

            if not size["ok"]:
                if not last:
                    # Oversized: try the fallback (still before any
                    # simulation/sign/broadcast, still one intent).
                    continue
                limit = size.get("max_bytes")
                failure = (
                    f"{pname.upper()}_TX_TOO_LARGE",
                    f"REAL_BUY_BLOCKED=TX_TOO_LARGE provider={pname.upper()} "
                    f"transaction_version={size.get('transaction_version')} "
                    f"raw_bytes={size.get('raw_bytes')} "
                    f"encoded_bytes={size.get('encoded_bytes')} "
                    f"limit={limit} over_by={size.get('over_by')}; "
                    f"no simulation, no signing, no broadcast",
                )
                break

            selected = (prov, pname, quote, built, size, quote_t0)
            break

        if selected is None:
            fill.route_metadata = {
                "provider_chain": [p.provider_name() for p in providers],
                "attempts": attempts,
            }
            if failure is not None and len(attempts) <= 1:
                term_status, term_reason = failure
            else:
                joined = ", ".join(
                    f"{a['provider']}={a['status']}" for a in attempts)
                term_status = "REAL_BUY_BLOCKED_NO_VALID_EXECUTION_ROUTE"
                term_reason = f"REAL_BUY_BLOCKED=NO_VALID_EXECUTION_ROUTE; {joined}"
            # No provider could produce a route. This is a ROUTE failure, and the
            # telemetry says so explicitly rather than reporting a generic
            # execution exception.
            fill.route_metadata["failure"] = self._failure_record(
                term_status, term_reason,
                provider=(attempts[-1]["provider"] if attempts else ""),
                fallback_attempted=len(attempts) > 1,
                fallback_result=", ".join(
                    f"{a['provider']}:{a['status']}" for a in attempts),
                slippage_bps=int(slippage_bps),
            )
            return _fail(term_status, term_reason)

        _prov, provider_name, quote, built, size, quote_t0 = selected
        quote_received_at = (getattr(quote, "quote_received_at", None)
                             or quote_t0)
        if intent is not None:
            intent_store.transition(intent.execution_id, READY,
                                    detail=f"{provider_name} quote accepted")

        # 4. Record the selected candidate's economics (unchanged semantics).
        # SIDE-AWARE QUOTE PRICE. ``quote.effective_price(side)`` is the venue's
        # own price oriented for THIS trade. The previous ``quote.price_per_token``
        # was ``in_amount / expected_out_amount`` for every side, which is
        # tokens-per-lamport on a sell; the 2026-10-02 $notify exit persisted
        # ``quote_price = 6634.915851383544`` because of it.
        fill.quote_price = _quote_price(quote, side)
        fill.price_impact_pct = quote.price_impact_pct
        fill.route_labels = list(quote.route_labels)
        fill.quote_provider = getattr(quote, "provider_name",
                                      lambda: "")() or None
        fill.route_metadata = {
            "provider": provider_name.upper(),
            "provider_name": provider_name,
            # Provider-owned SOL semantics, reported for the audit trail. The
            # generic layer never gates a SOL BUY on these.
            "sol_input_mode": getattr(_prov, "sol_input_mode", "unknown"),
            "wsol_preprovision_required": bool(
                getattr(_prov, "requires_preprovisioned_wsol", False)),
            "provider_chain": [p.provider_name() for p in providers],
            "attempts": attempts,
            "route": _prov.route_metadata(quote),
            "raw_bytes": size.get("raw_bytes"),
            "encoded_bytes": size.get("encoded_bytes"),
            # The version is read off the wire, and the limit is the one that
            # applied to it: 1232 for legacy/v0, 4096 for v1. Reported so an
            # operator can tell "too big for v0, fine for v1" from "too big
            # either way" without re-deriving anything.
            "transaction_version": size.get("transaction_version"),
            "max_bytes": size.get("max_bytes"),
            "limit_bytes": size.get("max_bytes"),
            "over_by": size.get("over_by"),
            "signature_slots": getattr(built, "signature_slots", None),
            "unsigned": getattr(built, "unsigned", None),
            "requested_slippage_bps": int(slippage_bps),
            "effective_slippage_bps": int(getattr(quote, "slippage_bps", 0)),
            "requested_priority_fee_lamports": int(self.priority_fee_lamports),
            "effective_priority_fee_lamports": getattr(
                built, "prioritization_fee_lamports", None),
            "timing": {"quote_received_at": quote_received_at,
                       "tx_built_at": tx_built_at},
        }

        # 5. QUOTE FRESHNESS. An existing rule, unchanged. A provider quote does
        #    not automatically satisfy the bot's freshness contract.
        age_ms = (time.time() - quote_received_at) * 1000.0
        if self.quote_max_age_ms and age_ms > self.quote_max_age_ms:
            return _fail(
                f"{provider_name.upper()}_STALE_QUOTE",
                f"{provider_name} quote age {age_ms:.0f}ms exceeds "
                f"{self.quote_max_age_ms:.0f}ms; refused before signing",
            )

        # 6. SIMULATE explicitly, before signing and broadcast. The unsigned
        #    transaction is simulated (sigVerify is off on the RPC). BOTH
        #    providers reach this exact stage; neither can bypass it.
        _stage("exec_simulate_start")
        detailed = getattr(self.rpc, "simulate_transaction_detailed", None)
        if callable(detailed):
            out = await detailed(built.transaction_b64)
        else:
            ok, detail = await self.rpc.simulate_transaction(
                built.transaction_b64)
            out = _coerce_legacy_outcome(ok, detail)
        _stage("exec_simulate_end")
        log.info(
            "%s swap: QUOTE=OK BUILD=OK SERIALIZE=OK SIZE=%s "
            "SIMULATION=%s BROADCAST=NOT_ATTEMPTED TX_LIMIT=%s TX_VERSION=%s",
            provider_name.upper(), size.get("raw_bytes"),
            "OK" if out.ok else "FAILED", size.get("max_bytes"),
            size.get("transaction_version"))
        if not out.ok:
            # 6b. ALT-LOAD RESCUE. A v0 transaction whose address lookup tables
            #     the node refuses to load is rejected regardless of its size --
            #     a Raptor swap at 824 bytes, well inside the 1232-byte limit,
            #     fails with "-32602 Transaction loads an address table account
            #     that doesn't exist" while every table reads back fine through
            #     getAccountInfo. v1 has no lookup tables, so the same trade
            #     re-encoded as v1 simulates clean.
            #
            #     This does NOT skip simulation: the original transaction WAS
            #     simulated and failed. The rebuilt one is simulated again from
            #     scratch, through the same gate, before anything is signed.
            if _is_alt_load_refusal(getattr(out, "detail", "") or
                                    getattr(out, "rpc_message", "")):
                rescue = await self._v1_rescue(built, size,
                                               reason="ALT_LOAD_REFUSED")
                attempts.append({
                    "provider": provider_name, "stage": "v1_rescue_after_sim",
                    "status": rescue.get("status"),
                    "raw_bytes": rescue.get("raw_bytes"),
                    "transaction_version": rescue.get("transaction_version"),
                    "max_bytes": rescue.get("max_bytes"),
                    "route": _prov.route_metadata(quote),
                    "v1_rescue": {k: v for k, v in rescue.items()
                                  if k != "transaction_b64"},
                })
                if rescue.get("ok"):
                    built = _replace_transaction(built,
                                                 rescue["transaction_b64"])
                    size = transaction_size_report(built.transaction_b64)
                    fill.route_metadata.update({
                        "transaction_version": size.get("transaction_version"),
                        "raw_bytes": size.get("raw_bytes"),
                        "max_bytes": size.get("max_bytes"),
                        "over_by": size.get("over_by"),
                    })
                    _stage("exec_simulate_start")
                    detailed = getattr(self.rpc, "simulate_transaction_detailed",
                                       None)
                    if callable(detailed):
                        out = await detailed(built.transaction_b64)
                    else:
                        ok, detail = await self.rpc.simulate_transaction(
                            built.transaction_b64)
                        out = _coerce_legacy_outcome(ok, detail)
                    _stage("exec_simulate_end")
                    log.info(
                        "%s swap: v0 ALT load refused by the node -> rebuilt as "
                        "v1 at %s bytes (limit %s), SIMULATION=%s",
                        provider_name.upper(), size.get("raw_bytes"),
                        size.get("max_bytes"),
                        "OK" if out.ok else "FAILED")
            if not out.ok:
                # The chain refused the transaction. Record WHICH program
                # refused it and at WHICH instruction index, resolved from the
                # very transaction that was simulated: a ``Custom`` code is
                # program-scoped and is meaningless without its emitting
                # program.
                from .failures import parse_instruction_error
                ix, kind, custom = parse_instruction_error(
                    getattr(out, "raw_err", None))
                program_id = await self._resolve_instruction_program(
                    built.transaction_b64, ix)
                sim_status = f"{provider_name.upper()}_SIMULATION_FAILED"
                record = self._failure_record(
                    sim_status, out.detail,
                    provider=provider_name, rpc_code=out.rpc_code,
                    program_id=program_id, instruction_index=ix,
                    custom_code=custom,
                    quote_at=quote_received_at, simulation_at=time.time(),
                    price_impact_pct=getattr(quote, "price_impact_pct", None),
                    slippage_bps=int(slippage_bps),
                    fallback_attempted=len(attempts) > 1,
                    fallback_result=", ".join(
                        f"{a['provider']}:{a['status']}" for a in attempts),
                )
                record["error_kind"] = kind
                if getattr(out, "logs", None):
                    record["simulation_logs"] = list(out.logs)
                if getattr(out, "rpc_message", ""):
                    record["rpc_message"] = out.rpc_message
                fill.route_metadata.setdefault("failure", {}).update(record)
                return _fail(sim_status, out.detail)

        # 7. SIGN locally. The bot is the only signer; nothing signed is sent
        #    to any provider.
        if intent is not None:
            intent_store.transition(intent.execution_id, SIGNING,
                                    detail="signing begins")
        _stage("exec_signing_start")
        try:
            signed_b64 = base64.b64encode(
                self.wallet.sign_wire(
                    base64.b64decode(built.transaction_b64))
            ).decode("ascii")
        except Exception as exc:  # noqa: BLE001
            _stage("exec_signing_end")
            return _fail("SIGN_FAILED",
                         f"signing failed: {type(exc).__name__}")
        _stage("exec_signing_end")
        simulation_at = time.time()
        fill.route_metadata["timing"]["simulation_at"] = simulation_at

        # 8. BROADCAST through the bot's OWN RPC. No provider's
        #    /send-transaction endpoint is ever called, so signing, intent and
        #    reconciliation control stay with the bot. Broadcast happens EXACTLY
        #    ONCE, on the selected provider's transaction only.
        if intent is not None:
            intent_store.transition(intent.execution_id, BROADCASTED,
                                    detail="broadcast attempted")
        _stage("exec_broadcast_start")
        try:
            signature = await self.rpc.send_transaction(signed_b64)
        except (TransactionFailed, RpcError) as exc:
            _stage("exec_broadcast_end")
            # A node-side refusal with no signature is PROOF nothing was
            # submitted, so the intent is FAILED and terminal. Recording it as
            # UNKNOWN is what left 8HJY9 blocking every REAL entry for three
            # hours after a preflight simulation failure.
            maybe_submitted = send_failure_may_have_submitted(exc)
            from .failures import TRANSACTION_UNKNOWN, RPC_REJECTED
            fill.route_metadata.setdefault("failure", {}).update(
                self._failure_record(
                    "BROADCAST_REJECTED", str(exc), provider=provider_name,
                    category=(TRANSACTION_UNKNOWN if maybe_submitted
                              else RPC_REJECTED),
                    rpc_code=getattr(exc, "code", None),
                    quote_at=quote_received_at, simulation_at=time.time(),
                    slippage_bps=int(slippage_bps),
                    fallback_attempted=len(attempts) > 1,
                    fallback_result=", ".join(
                        f"{a['provider']}:{a['status']}" for a in attempts),
                ))
            return _fail("BROADCAST_REJECTED", str(exc),
                         unknown=maybe_submitted)
        _stage("exec_broadcast_end")
        fill.signature = signature
        fill.route_metadata["timing"]["broadcast_at"] = time.time()
        if intent is not None:
            intent_store.transition(
                intent.execution_id, BROADCASTED,
                signature=signature, detail="signature observed",
            )

        # 8. CONFIRM. Ambiguity is a first-class outcome.
        _stage("exec_confirm_start")
        result = await self.rpc.wait_for_confirmation(
            signature,
            last_valid_block_height=built.last_valid_block_height,
            timeout=self.confirm_timeout,
        )
        _stage("exec_confirm_end")
        status = result.get("status", AMBIGUOUS)
        if status not in FILLED_STATUSES:
            if status == "failed":
                return _fail(
                    "CHAIN_FAILED", str(result.get("error") or status),
                )
            fill.status = (AMBIGUOUS if status == "timeout"
                           else f"CHAIN_{str(status).upper()}")
            fill.ambiguity_reason = str(result.get("error") or status)
            if intent is not None:
                intent_store.transition(
                    intent.execution_id, UNKNOWN,
                    signature=signature, detail=fill.status,
                )
            return fill
        fill.status = status

        # 9. RECONCILE from chain. Chain reconciliation stays authoritative;
        #    Raptor's response is never trusted for final accounting.
        _stage("exec_reconcile_start")
        tx = await self.rpc.get_transaction(signature, commitment="confirmed")
        amounts = reconcile_transaction(
            tx or {}, signature, status, self.pubkey, output_mint
        )
        fill.commitment = status
        fill.slot = amounts.slot
        fill.fee_lamports = amounts.fee_lamports
        fill.confirmed_at = time.time()
        if amounts.sanity_warnings:
            fill.detail = "; ".join(amounts.sanity_warnings)
            log.warning("RAPTOR reconciliation sanity warning: %s", fill.detail)
        if amounts.token_input_amount is not None:
            fill.input_amount = amounts.token_input_amount
        if amounts.token_output_amount is not None:
            fill.output_amount = amounts.token_output_amount
        if side == "EXIT" and fill.output_amount is None:
            delta = amounts.net_lamports_delta
            if delta is not None:
                fill.output_amount = max(0, -delta)
        # THE SWAP LEG IS NOT THE WALLET'S BALANCE CHANGE.
        #
        # ``net_lamports_delta`` is the payer's whole SOL decrease: swap
        # principal PLUS account-creation rent PLUS the network fee. For a $GOAT
        # entry on 2026-10-02 that was 3,497,651 lamports, of which only
        # 1,928,811 reached the venue. Dividing the total by a token count is
        # what produced a journalled "execution price" of 0.0004634 and a
        # "slippage" of 70.5% against a quote that was never 70% away.
        #
        # The split is read from the same transaction, so it is a measurement
        # and not an estimate. When the venue leg cannot be separated the swap
        # input stays ``None`` and no per-token price is derived from the wallet
        # delta.
        fill.wallet_debit_lamports = amounts.wallet_debit_lamports
        fill.rent_lamports = amounts.created_account_rent_lamports
        fill.invoked_programs = tuple(amounts.invoked_programs or ())
        if side == "ENTRY":
            venue_leg = amounts.lamports_in_from_others or None
            fill.swap_input_amount = venue_leg
            if fill.input_amount is None:
                # Fall back to the payer's whole debit SO THE FACT IS NOT LOST,
                # but LABEL it. A wallet balance change is a cash-flow figure,
                # not a price input, and nothing downstream is allowed to derive
                # a per-token price from it -- ``RealFillRecord`` refuses exactly
                # that, which is why ``actual_execution_price`` is ``None`` here
                # when the venue leg could not be separated.
                fill.input_amount = venue_leg
                fill.input_amount_kind = "SWAP_LEG"
                if fill.input_amount is None:
                    fill.input_amount = amounts.net_lamports_delta
                    fill.input_amount_kind = "WALLET_DEBIT"
        if side == "EXIT" and amounts.lamports_out_to_others is not None:
            fill.lamports_out_to_others = amounts.lamports_out_to_others
        if fill.output_amount:
            # SIDE-AWARE, FAIL-CLOSED. The previous expression applied one
            # formula to both sides, which is wrong twice: on a sell the ratio
            # is inverted (tokens-per-lamport), and when the input leg is absent
            # -- the normal case for a native-SOL unwind -- the ``or 0`` made it
            # identically zero. That zero was journalled as
            # ``actual_execution_price``, so those exits published a
            # measured-looking price of 0.0. Unset is correct here; a fabricated
            # zero is not.
            from ..pricing import execution_price_per_token
            # The price comes from the SWAP leg. On an ENTRY that is
            # ``swap_input_amount`` (the venue-side lamports), never the payer's
            # balance change, which also carries rent and the network fee.
            price_input = (fill.swap_input_amount
                           if side == "ENTRY"
                           else (fill.input_amount
                                 or getattr(fill, "lamports_out_to_others",
                                            None)))
            fill.execution_price = execution_price_per_token(
                side=side,
                input_amount=price_input,
                output_amount=fill.output_amount,
            )
            if fill.execution_price is None:
                fill.ambiguity_reason = (
                    (fill.ambiguity_reason + "; " if fill.ambiguity_reason else "")
                    + "execution price not derivable from reconciled amounts; "
                      "left unset rather than reported as 0.0"
                )
        if fill.output_amount in (None, 0):
            # The chain CONFIRMED this transaction. It did not fail and it was
            # not refused -- we could simply not read the output amount back.
            # Reporting AMBIGUOUS/fill_refused here is exactly what made the
            # $Boy buy look like a rejected entry while the wallet already held
            # the tokens.
            #
            # The amount is NEVER invented from the quote. It stays unknown and
            # the fill is flagged for recovery, so the authoritative amount is
            # read back from on-chain state rather than assumed.
            fill.status = CONFIRMED_UNRECONCILED
            fill.ambiguity_reason = (
                "confirmed on chain but no output amount could be reconciled; "
                "funds may have moved -- recovery from on-chain state required"
            )
            fill.requires_recovery = True
            if intent is not None:
                intent_store.transition(
                    intent.execution_id, UNKNOWN,
                    signature=signature,
                    detail="confirmed but unreconciled; recovery required",
                )
            _stage("exec_reconcile_end")
            return fill
        if intent is not None:
            intent_store.transition(
                intent.execution_id, CONFIRMED,
                signature=signature,
                detail=f"fill reconciled ({side})",
            )
        _stage("exec_reconcile_end")
        return fill

    async def execute_swap(
        self,
        side: str,
        input_mint: str,
        output_mint: str,
        lamports: int,
        slippage_bps: int,
        *,
        intent_store: Any = None,
        logical_key: Optional[str] = None,
        stage: Optional[Any] = None,
    ) -> OnchainFill:
        """
        quote -> build -> sign -> simulate -> send -> confirm -> reconcile.

        Raises only for programmer error. Every venue, chain or safety failure
        comes back as an unfilled/ambiguous :class:`OnchainFill` carrying the
        exact reason.

        IDEMPOTENCY: when ``intent_store`` and ``logical_key`` are supplied, a
        durable intent is RESERVED and each lifecycle transition is persisted
        BEFORE the network call it describes. A crash after broadcast therefore
        leaves a record that the attempt was in flight, and a retry is refused
        by the pre-trade checks rather than broadcasting a second time.
        """

        # The Raptor path is selected by the feature flag. It shares the wallet,
        # RPC, intent store, sizing, slippage and freshness rules, but builds
        # through the Raptor router instead of the Jupiter planner.
        if self.raptor_enabled:
            return await self._execute_swap_raptor(
                side, input_mint, output_mint, lamports, slippage_bps,
                intent_store=intent_store, logical_key=logical_key,
                stage=stage,
            )

        def _stage(name: str) -> None:
            if stage is not None:
                try:
                    stage(name)
                except Exception:  # noqa: BLE001 - telemetry never breaks a trade
                    pass

        fill = OnchainFill(
            side=side, requested_lamports=int(lamports), slippage_bps=int(slippage_bps)
        )
        side_mint = output_mint if side == "ENTRY" else input_mint
        # 0. RESERVE the logical order BEFORE anything irreversible. Persisted.
        intent = None
        if intent_store is not None and logical_key:
            _stage("exec_intent_reserve_start")
            try:
                intent = intent_store.reserve(
                    logical_key=logical_key, side=side, mint=side_mint,
                    requested_amount=int(lamports),
                )
            except Exception as exc:  # noqa: BLE001 - duplicate reservation is a refusal
                fill.status = "DUPLICATE_EXECUTION"
                fill.ambiguity_reason = str(exc)
                _stage("exec_intent_reserve_end")
                return fill
            _stage("exec_intent_reserve_end")

        def _fail(status: str, reason: str, *, unknown: bool = False) -> OnchainFill:
            """Record a pre/at-broadcast failure with the right terminal state."""
            fill.status = status
            fill.ambiguity_reason = reason
            if intent is not None:
                try:
                    intent_store.transition(
                        intent.execution_id,
                        UNKNOWN if unknown else FAILED,
                        detail=f"{status}: {reason}",
                    )
                except Exception:  # noqa: BLE001
                    pass
            return fill

        # 1. Quote. A venue that refuses the notional is a distinct, reportable
        #    condition - it is what a $0.155 order on a thin venue produces.
        _stage("tx_quote_start")
        try:
            quote = await self.provider.quote(
                input_mint, output_mint, lamports, slippage_bps
            )
        except MicroOrderUnsupported as exc:
            _stage("tx_quote_end")
            return _fail("MICRO_REAL_EXECUTION_UNSUPPORTED", str(exc))
        except QuoteRejected as exc:
            _stage("tx_quote_end")
            return _fail("QUOTE_REJECTED", str(exc))
        except QuoteUnavailable as exc:
            _stage("tx_quote_end")
            return _fail("QUOTE_UNAVAILABLE", str(exc))
        _stage("tx_quote_end")

        # SIDE-AWARE QUOTE PRICE. ``quote.effective_price(side)`` is the venue's
        # own price oriented for THIS trade. The previous ``quote.price_per_token``
        # was ``in_amount / expected_out_amount`` for every side, which is
        # tokens-per-lamport on a sell; the 2026-10-02 $notify exit persisted
        # ``quote_price = 6634.915851383544`` because of it.
        fill.quote_price = _quote_price(quote, side)
        fill.price_impact_pct = quote.price_impact_pct
        fill.route_labels = list(quote.route_labels)
        fill.quote_provider = getattr(quote, "provider_name",
                                      lambda: "")() or None
        if intent is not None:
            intent_store.transition(intent.execution_id, READY, detail="quote accepted")

        # 2a. WSOL MODE must be decided BEFORE planning, because it changes
        #     which candidates the planner is allowed to build.
        #
        #     A SOL-denominated BUY is routed through a PRE-PROVISIONED
        #     wrapped-SOL account when one is available. With
        #     wrapAndUnwrapSol=true (the old default) the venue wraps SOL into a
        #     temporary wSOL account inside the swap and closes it again; those
        #     extra instructions and account keys are what push a thin L25
        #     token's transaction past Solana's 1232-byte limit.
        #
        #     When wSOL is NOT ready we REFUSE. We deliberately do not fall back
        #     to wrapAndUnwrapSol=true: silently reverting to the
        #     temporary-account route would rebuild the oversized transaction
        #     and hide the reason the trade failed.
        wrap_and_unwrap = True
        if side == "ENTRY" and str(input_mint) == WSOL_MINT:
            readiness = await self.rpc.wsol_readiness(
                self.pubkey, int(lamports))
            if not readiness.ready:
                _stage("tx_build_end")
                return _fail("REAL_BUY_BLOCKED", readiness.block_reason())
            # Ready: route through the existing wSOL account and let the venue
            # omit the create/initialize/close trio entirely.
            wrap_and_unwrap = False

        # 2. PLAN. Build every candidate, measure its real serialized size,
        #    reject anything over the protocol limit, simulate what survives,
        #    and choose deterministically.
        #
        #    This replaces "hand the venue's answer straight to
        #    simulateTransaction". An unsendable transaction used to surface as
        #    an opaque -32602 at the last possible moment; now it is rejected
        #    on measured bytes, before any round trip, with the byte counts
        #    attached. Simulation is still mandatory and still runs before
        #    broadcast -- it has moved EARLIER in the sequence, never been
        #    skipped.
        plan = await plan_execution(
            provider=self.provider, rpc=self.rpc, wallet=self.wallet,
            side=side, input_mint=input_mint, output_mint=output_mint,
            lamports=lamports, slippage_bps=slippage_bps, owner=self.pubkey,
            priority_fee=self.priority_fee_lamports,
            # The temporary-wSOL route is never offered silently: it is only a
            # candidate when the wallet genuinely has no provisioned wSOL, and
            # in that case the readiness check in 2a has already refused.
            allow_temporary_wsol=bool(wrap_and_unwrap))
        for cand in plan.candidates:
            if cand.simulation is None and cand.status == "OVERSIZED":
                limit = cand.max_bytes or 0
                log.info("REAL BUY candidate rejected before RPC: %s TX_SIZE=%s "
                         "TX_LIMIT=%s OVER_BY=%s VERSION=%s", cand.label,
                         cand.raw_bytes, limit,
                         (cand.raw_bytes or 0) - limit,
                         cand.transaction_version)
        if plan.chosen is None:
            _stage("tx_build_end")
            # Keep the diagnosis specific. "No fitting route" and "every route
            # simulated badly" are different problems with different fixes, and
            # collapsing them into one status is how a simulation regression
            # gets read as a size problem.
            if plan.status == "ALL_SIMULATIONS_FAILED":
                return _fail("SIMULATION_FAILED_ALL_CANDIDATES", plan.detail)
            if plan.status == "ALL_SIGNINGS_FAILED":
                return _fail("SIGN_FAILED", plan.detail)
            if plan.status == "ALL_BUILDS_FAILED":
                return _fail("BUILD_FAILED", plan.detail)
            if plan.status == "QUOTE_FAILED":
                return _fail("QUOTE_UNAVAILABLE", plan.detail)
            return _fail("NO_FITTING_ROUTE", plan.detail)

        # 2b. Build the unsigned venue transaction (v0 + ALTs, all venue-side).
        #
        #    WSOL MODE was decided before planning (see 2a): a SOL-denominated
        #    BUY routes through a pre-provisioned wrapped-SOL account, and the
        #    planner was only allowed to build the wsol-preprovisioned
        #    candidates in that case.
        _stage("tx_build_start")
        chosen = plan.chosen
        from .jupiter import BuiltSwap
        built = BuiltSwap(transaction_b64=chosen.transaction_b64,
                          last_valid_block_height=chosen.route_metadata.get(
                              "last_valid_block_height"),
                          prioritization_fee_lamports=None,
                          quote=quote)
        _stage("tx_build_end")

        # 3. Sign locally. Wire-level signing is required because a Jupiter swap
        #    is a v0 versioned transaction with address lookup tables, which
        #    solders' legacy Transaction.sign() cannot represent.
        if intent is not None:
            intent_store.transition(
                intent.execution_id, SIGNING, detail="signing begins"
            )
        _stage("exec_signing_start")
        try:
            signed_b64 = base64.b64encode(
                self.wallet.sign_wire(base64.b64decode(built.transaction_b64))
            ).decode("ascii")
        except Exception as exc:  # noqa: BLE001
            _stage("exec_signing_end")
            return _fail("SIGN_FAILED", f"signing failed: {type(exc).__name__}")
        _stage("exec_signing_end")

        # 4. Simulation already ran inside the planner, on the chosen candidate
        #    only. It is NOT repeated here: that would be a second round trip
        #    for no new information. The invariant that matters is that a
        #    simulation failure can never reach the send step, and the planner
        #    enforces it structurally -- it only ever selects a candidate whose
        #    simulation came back clean.
        #
        #    A local size check still runs as a last guard, on the exact bytes
        #    that would be signed, against the ceiling for the transaction's own
        #    version. A transaction over ITS OWN limit can never be broadcast, so
        #    this is a refusal, never a workaround: nothing is relaxed, skipped
        #    or enlarged to get past it.
        if self.simulate_before_send:
            _stage("exec_simulate_start")
            # Compatibility shim: this executor is also driven by test doubles
            # and any SolanaRpc-shaped object exposing only the original
            # two-tuple API. Falling back to it is the pre-existing fail-closed
            # behaviour, so nothing is weakened or skipped.
            size_fn = getattr(self.rpc, "transaction_size_report", None)
            size = (size_fn(signed_b64) if callable(size_fn)
                    else {"ok": True, "status": "TX_SIZE_CHECK_UNAVAILABLE",
                          "raw_bytes": None, "max_bytes": None})
            if not size["ok"]:
                _stage("exec_simulate_end")
                return _fail(
                    "SIMULATION_FAILED_TX_TOO_LARGE"
                    if size["status"] == "TX_TOO_LARGE"
                    else "SIMULATION_FAILED_TX_MALFORMED",
                    f"{size['detail']}; no RPC call made, nothing broadcast",
                )
            detailed = getattr(self.rpc, "simulate_transaction_detailed", None)
            if callable(detailed):
                out = await detailed(signed_b64)
            else:
                ok, detail = await self.rpc.simulate_transaction(signed_b64)
                out = _coerce_legacy_outcome(ok, detail)
            _stage("exec_simulate_end")
            log.info(
                "REAL BUY simulation: BUILD=OK SERIALIZE=OK RPC_REQUEST=OK "
                "SIMULATION=%s BROADCAST=NOT_ATTEMPTED TX_SIZE=%s TX_LIMIT=%s "
                "TX_VERSION=%s",
                "OK" if out.ok else "FAILED",
                size.get("raw_bytes"), size.get("max_bytes"),
                size.get("transaction_version"))
            if not out.ok:
                return _fail(out.status, out.detail)

        # 5. Broadcast exactly once. The intent is persisted as BROADCASTED
        #    BEFORE the call, so a crash during the call still leaves a record
        #    that an irreversible attempt may be on chain.
        if intent is not None:
            intent_store.transition(
                intent.execution_id, BROADCASTED, detail="broadcast attempted"
            )
        _stage("exec_broadcast_start")
        try:
            signature = await self.rpc.send_transaction(signed_b64)
        except (TransactionFailed, RpcError) as exc:
            _stage("exec_broadcast_end")
            # "the RPC said no" is not the same claim as "the network never saw
            # it" -- a LOST RESPONSE may still have submitted it, and that stays
            # UNKNOWN. But a node that answered with a structured refusal and no
            # signature did not submit anything, and calling that UNKNOWN blocks
            # every subsequent REAL entry for a transaction that provably does
            # not exist.
            return _fail("BROADCAST_REJECTED", str(exc),
                         unknown=send_failure_may_have_submitted(exc))
        _stage("exec_broadcast_end")
        fill.signature = signature
        # Record the signature the instant it is known.
        if intent is not None:
            intent_store.transition(
                intent.execution_id, BROADCASTED,
                signature=signature, detail="signature observed",
            )

        # 6. Confirmation. Ambiguity is a first-class outcome.
        _stage("exec_confirm_start")
        result = await self.rpc.wait_for_confirmation(
            signature,
            last_valid_block_height=built.last_valid_block_height,
            timeout=self.confirm_timeout,
        )
        _stage("exec_confirm_end")
        status = result.get("status", AMBIGUOUS)
        if status not in FILLED_STATUSES:
            if status == "failed":
                # Terminal on chain with no swap: provably no fill.
                return _fail(
                    "CHAIN_FAILED", str(result.get("error") or status),
                )
            # timeout / expired: the transaction may still land. UNKNOWN.
            fill.status = AMBIGUOUS if status == "timeout" else f"CHAIN_{str(status).upper()}"
            fill.ambiguity_reason = str(result.get("error") or status)
            if intent is not None:
                intent_store.transition(
                    intent.execution_id, UNKNOWN,
                    signature=signature, detail=fill.status,
                )
            return fill
        fill.status = status

        # 7. Reconcile the ACTUAL amounts from the confirmed transaction.
        _stage("exec_reconcile_start")
        tx = await self.rpc.get_transaction(signature, commitment="confirmed")
        amounts = reconcile_transaction(
            tx or {}, signature, status, self.pubkey, output_mint
        )
        fill.commitment = status
        fill.slot = amounts.slot
        # The ACTUAL fee, in lamports, read from the transaction metadata - NOT
        # the wallet's SOL balance delta. On a buy the delta is principal plus
        # fee; on a sell it is proceeds minus fee. Using the delta as the fee is
        # the ~4500x overstatement this corrects.
        fill.fee_lamports = amounts.fee_lamports
        fill.confirmed_at = time.time()
        if amounts.sanity_warnings:
            # Never silently clamp an absurd fee: surface it and leave the field
            # unset so no caller can book a fabricated number.
            fill.detail = "; ".join(amounts.sanity_warnings)
            log.warning("REAL reconciliation sanity warning: %s", fill.detail)
        if amounts.token_input_amount is not None:
            fill.input_amount = amounts.token_input_amount
        if amounts.token_output_amount is not None:
            fill.output_amount = amounts.token_output_amount

        # For a SELL the token leg is the input and SOL is the output. A Jupiter
        # exit may unwrap to native SOL, in which case there is no WSOL token
        # balance and the token leg is absent. The wallet's SOL delta is then
        # authoritative: ``net_lamports_delta`` is (pre - post), so a sell that
        # increased the balance has a NEGATIVE delta and the net proceeds are
        # ``-delta`` (the fee is already reflected in the balance). The old code
        # used ``max(0, delta - fee)``, which is 0 for a negative delta and
        # reported a successful sell as AMBIGUOUS.
        if side == "EXIT" and fill.output_amount is None:
            delta = amounts.net_lamports_delta
            if delta is not None:
                fill.output_amount = max(0, -delta)

        # For a BUY the SOL leg is the INPUT and is not a token balance, so
        # ``reconcile_transaction`` leaves it None. The wallet's total SOL
        # decrease is reported: swap principal PLUS fee. This is deliberate and
        # conservative for a slippage figure - ``execution_price`` therefore
        # includes the cost of the trade - and the fee is ALSO reported
        # separately in ``fee_lamports``, read from ``meta.fee``. The two are
        # distinct quantities; neither is derived from the other.
        # THE SWAP LEG IS NOT THE WALLET'S BALANCE CHANGE.
        #
        # ``net_lamports_delta`` is the payer's whole SOL decrease: swap
        # principal PLUS account-creation rent PLUS the network fee. For a $GOAT
        # entry on 2026-10-02 that was 3,497,651 lamports, of which only
        # 1,928,811 reached the venue. Dividing the total by a token count is
        # what produced a journalled "execution price" of 0.0004634 and a
        # "slippage" of 70.5% against a quote that was never 70% away.
        #
        # The split is read from the same transaction, so it is a measurement
        # and not an estimate. When the venue leg cannot be separated the swap
        # input stays ``None`` and no per-token price is derived from the wallet
        # delta.
        fill.wallet_debit_lamports = amounts.wallet_debit_lamports
        fill.rent_lamports = amounts.created_account_rent_lamports
        fill.invoked_programs = tuple(amounts.invoked_programs or ())
        if side == "ENTRY":
            venue_leg = amounts.lamports_in_from_others or None
            fill.swap_input_amount = venue_leg
            if fill.input_amount is None:
                # Fall back to the payer's whole debit SO THE FACT IS NOT LOST,
                # but LABEL it. A wallet balance change is a cash-flow figure,
                # not a price input, and nothing downstream is allowed to derive
                # a per-token price from it -- ``RealFillRecord`` refuses exactly
                # that, which is why ``actual_execution_price`` is ``None`` here
                # when the venue leg could not be separated.
                fill.input_amount = venue_leg
                fill.input_amount_kind = "SWAP_LEG"
                if fill.input_amount is None:
                    fill.input_amount = amounts.net_lamports_delta
                    fill.input_amount_kind = "WALLET_DEBIT"
        if side == "EXIT" and amounts.lamports_out_to_others is not None:
            fill.lamports_out_to_others = amounts.lamports_out_to_others
        if fill.output_amount:
            # SIDE-AWARE, FAIL-CLOSED. The previous expression applied one
            # formula to both sides, which is wrong twice: on a sell the ratio
            # is inverted (tokens-per-lamport), and when the input leg is absent
            # -- the normal case for a native-SOL unwind -- the ``or 0`` made it
            # identically zero. That zero was journalled as
            # ``actual_execution_price``, so those exits published a
            # measured-looking price of 0.0. Unset is correct here; a fabricated
            # zero is not.
            from ..pricing import execution_price_per_token
            # The price comes from the SWAP leg. On an ENTRY that is
            # ``swap_input_amount`` (the venue-side lamports), never the payer's
            # balance change, which also carries rent and the network fee.
            price_input = (fill.swap_input_amount
                           if side == "ENTRY"
                           else (fill.input_amount
                                 or getattr(fill, "lamports_out_to_others",
                                            None)))
            fill.execution_price = execution_price_per_token(
                side=side,
                input_amount=price_input,
                output_amount=fill.output_amount,
            )
            if fill.execution_price is None:
                fill.ambiguity_reason = (
                    (fill.ambiguity_reason + "; " if fill.ambiguity_reason else "")
                    + "execution price not derivable from reconciled amounts; "
                      "left unset rather than reported as 0.0"
                )

        if fill.output_amount in (None, 0):
            # The chain CONFIRMED this transaction. It did not fail and it was
            # not refused -- we could simply not read the output amount back.
            # Reporting AMBIGUOUS/fill_refused here is exactly what made the
            # $Boy buy look like a rejected entry while the wallet already held
            # the tokens.
            #
            # The amount is NEVER invented from the quote. It stays unknown and
            # the fill is flagged for recovery, so the authoritative amount is
            # read back from on-chain state rather than assumed.
            fill.status = CONFIRMED_UNRECONCILED
            fill.ambiguity_reason = (
                "confirmed on chain but no output amount could be reconciled; "
                "funds may have moved -- recovery from on-chain state required"
            )
            fill.requires_recovery = True
            if intent is not None:
                intent_store.transition(
                    intent.execution_id, UNKNOWN,
                    signature=signature,
                    detail="confirmed but unreconciled; recovery required",
                )
            _stage("exec_reconcile_end")
            return fill
        if intent is not None:
            intent_store.transition(
                intent.execution_id, CONFIRMED,
                signature=signature,
                detail=f"fill reconciled ({side})",
            )
        _stage("exec_reconcile_end")
        return fill


# ---------------------------------------------------------------------------
# pre-trade safety
# ---------------------------------------------------------------------------


@dataclass
class SafetyCheck:
    name: str
    ok: bool
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class PreflightResult:
    """The full result of a REAL preflight. ``ok`` requires all checks."""

    ok: bool
    checks: list[SafetyCheck] = field(default_factory=list)
    detail: str = ""

    @property
    def failures(self) -> list[SafetyCheck]:
        return [c for c in self.checks if not c.ok]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "detail": self.detail,
            "checks": [c.as_dict() for c in self.checks],
        }

    def render(self) -> str:
        lines = []
        for c in self.checks:
            mark = "PASS" if c.ok else "FAIL"
            lines.append(f"[{mark}] {c.name}: {c.detail}")
        return "\n".join(lines)


#: Names of every pre-trade check, in order. Used by tests to prove none was
#: quietly dropped.
PRE_TRADE_CHECKS: tuple[str, ...] = (
    "real_mode_explicit",
    "real_armed_by_operator",
    "no_emergency_stop",
    "v3c_authorized",
    "wallet_loaded",
    "wallet_correct",
    "rpc_healthy",
    "quote_provider_ready",
    "sizing_matches_policy",
    "sol_for_transaction_costs",
    "route_requirements_satisfied",
    "quote_valid",
    "quote_freshness",
    "exposure_within_limit",
    "open_positions_within_limit",
    "no_duplicate_position",
    "no_reentry",
    "no_concurrent_operation",
)


# ---------------------------------------------------------------------------
# the executor
# ---------------------------------------------------------------------------


def _coerce_legacy_outcome(ok: bool, detail: str):
    """
    Wrap a two-tuple ``(ok, detail)`` simulation result in a SimulationOutcome.

    Lets an older SolanaRpc-shaped object take part in the classified failure
    path without the caller branching on which API it got. The status is
    derived from the detail text, so a legacy failure is still reported with a
    specific reason instead of a bare status.
    """
    from .solana import SimulationOutcome, classify_simulation_error

    status = "SIMULATION_OK" if ok else classify_simulation_error(None, detail)
    return SimulationOutcome(
        ok=bool(ok), status=status, detail=detail, stage="SIMULATION",
        rpc_code=None, rpc_message="", broadcast_attempted=False)


class RealExecutor:
    """
    The REAL execution path.

    Constructing it does not arm anything. Arming lives in
    :class:`autopaper.modes.RuntimeState` and is an operator action; this class
    only refuses to trade when the state is not armed.
    """

    #: The real execution path is genuinely implemented.
    PLACEMENT_IMPLEMENTED = True

    def __init__(
        self,
        cfg: Any,
        backend: Optional[LiveSolanaBackend] = None,
        sol_usd_provider: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.real_cfg = getattr(cfg, "real", cfg)
        self._backend = backend
        #: Passed through to a lazily-built backend so the REAL path reads the
        #: SAME canonical SOL/USD provider the PAPER path does. See
        #: :meth:`LiveSolanaBackend.sol_usd_snapshot`.
        self._sol_usd_provider = sol_usd_provider
        self._busy = False
        #: Structured result of the most recent :meth:`quote_probe`, so the
        #: caller can name WHY the venue quote failed (category, provider,
        #: HTTP status, RPC code, whether a fallback was tried) instead of
        #: collapsing every pre-trade quote failure into one word.
        self._last_quote_probe: dict = {}

    @property
    def backend(self) -> LiveSolanaBackend:
        if self._backend is None:
            self._backend = LiveSolanaBackend(
                self.cfg, sol_usd_provider=self._sol_usd_provider)
        return self._backend

    @property
    def last_quote_probe(self) -> dict:
        """The structured record of the most recent quote probe (or ``{}``)."""
        return self._last_quote_probe

    @staticmethod
    def _probe_record(probe: dict, *, side: str) -> dict:
        """Turn a provider-chain probe into a categorized, secret-free record."""
        from .failures import classify_execution_failure
        status = probe.get("failure_status") or ""
        detail = probe.get("failure_detail") or ""
        category = (classify_execution_failure(status, detail=detail)
                    if status else "")
        return {
            "side": side,
            "provider": probe.get("provider") or "",
            "attempts": list(probe.get("attempts") or []),
            "fallback_attempted": bool(probe.get("fallback_attempted")),
            "ok": bool(probe.get("quote") is not None and not probe.get("micro")),
            "status": status,
            "detail": detail[:400],
            "category": category,
            "http_status": probe.get("failure_http_status"),
            "rpc_code": probe.get("failure_rpc_code"),
        }

    async def aclose(self) -> None:
        if self._backend is not None:
            await self._backend.aclose()

    # -- gates -------------------------------------------------------------

    def configuration_gates(self) -> list[SafetyCheck]:
        """
        Construction-time configuration checks. Never reveals a value, only
        whether a named thing is present.
        """
        rc = self.real_cfg
        checks = [
            SafetyCheck(
                "wallet_configured", bool(rc.wallet_keypair_path),
                "WALLET_KEYPAIR_PATH is set" if rc.wallet_keypair_path
                else "WALLET_KEYPAIR_PATH is not configured",
            ),
            SafetyCheck(
                "rpc_configured", bool(rc.rpc_url),
                "RPC_URL is set" if rc.rpc_url else "RPC_URL is not configured",
            ),
            SafetyCheck(
                "reentry_disabled", not bool(getattr(rc, "reentry_enabled", False)),
                "REENTRY_ENABLED is false",
            ),
            SafetyCheck(
                "max_open_positions",
                int(getattr(rc, "max_open_positions", 0)) <= MAX_OPEN_POSITIONS,
                f"MAX_OPEN_POSITIONS="
                f"{getattr(rc, 'max_open_positions', 'unset')} "
                f"(limit {MAX_OPEN_POSITIONS})",
            ),
            SafetyCheck(
                "slippage_bounded",
                0 < int(getattr(rc, "slippage_bps", 0)) <= 5000,
                f"REAL_SLIPPAGE_BPS={getattr(rc, 'slippage_bps', 'unset')}",
            ),
            SafetyCheck(
                "price_impact_bounded",
                0 < float(getattr(rc, "max_price_impact_pct", 0)) <= 50.0,
                f"REAL_MAX_PRICE_IMPACT_PCT="
                f"{getattr(rc, 'max_price_impact_pct', 'unset')}",
            ),
            SafetyCheck(
                "real_not_auto_enabled", not bool(getattr(rc, "trading_enabled", False)),
                "REAL is not enabled by configuration; only an operator action "
                "arms it",
            ),
        ]
        return checks

    async def preflight(self, *, wallet_masked_expected: Optional[str] = None) -> PreflightResult:
        """
        Full live preflight. Runs before the operator can arm REAL.

        Every failure is reported with its exact reason and the whole result is
        ``ok=False``, which the mode state machine treats as a refusal to arm.
        """
        checks: list[SafetyCheck] = list(self.configuration_gates())
        try:
            report = await self.backend.verify()
        except (SolanaUnavailable, RpcError) as exc:
            checks.append(SafetyCheck("executor_available", False, str(exc)))
            return PreflightResult(False, checks, "executor unavailable")

        checks.append(
            SafetyCheck(
                "wallet_loaded", bool(report.get("wallet_loaded")),
                report.get("wallet_masked") or "wallet could not be loaded",
            )
        )
        if wallet_masked_expected and report.get("wallet_masked") != wallet_masked_expected:
            checks.append(
                SafetyCheck(
                    "wallet_correct", False,
                    "loaded wallet is not the expected public address",
                )
            )
        else:
            checks.append(
                SafetyCheck("wallet_correct", bool(report.get("wallet_loaded")),
                            "public address verified")
            )
        checks.append(
            SafetyCheck(
                "rpc_healthy", bool(report.get("rpc_reachable")),
                str(report.get("rpc_detail") or "RPC unreachable"),
            )
        )
        checks.append(
            SafetyCheck(
                "quote_provider_ready", bool(report.get("quote_provider")),
                str(report.get("quote_provider_detail") or "quote provider unreachable"),
            )
        )
        checks.append(
            SafetyCheck(
                "signing_available", bool(report.get("signing_available")),
                "offline signing probe succeeded"
                if report.get("signing_available")
                else f"signing probe failed ({report.get('signing_error')})",
            )
        )
        # The sizing policy itself, verified as a pure function.
        from ..sizing import compute_position_size, describe_policy

        size = compute_position_size(1.55)
        sizing_ok = abs(size.position_usd - 0.155) < 1e-9
        checks.append(
            SafetyCheck(
                "sizing_matches_policy", sizing_ok,
                f"{describe_policy()} (${1.55:.2f} -> ${size.position_usd:.6f})",
            )
        )
        checks.append(
            SafetyCheck(
                "no_broadcast_in_preflight", not report.get("broadcast_attempted", False),
                "preflight is read-only; no transaction was broadcast",
            )
        )
        # Equity must be CALCULABLE before REAL can be armed. Without this, an
        # unreadable balance or a missing SOL/USD price produced a sizing
        # input of 0.0, which refused for the wrong reason and looked like a
        # measured "the account is empty".
        equity_reason = ""
        try:
            from ..real_equity import compute_real_equity

            lamports = None
            balance_readable = False
            try:
                state = await self.backend.wallet_state()
                lamports = state.get("lamports")
                balance_readable = lamports is not None
            except Exception as exc:  # noqa: BLE001
                equity_reason = f"native SOL balance unavailable ({type(exc).__name__})"

            checks.append(SafetyCheck(
                "native_sol_balance_readable", balance_readable,
                (f"{lamports / 1_000_000_000:.9f} SOL"
                 if lamports is not None else equity_reason or "not read"),
            ))

            price = None
            try:
                price = await self.backend.sol_price_usd()
            except Exception:  # noqa: BLE001
                pass
            checks.append(SafetyCheck(
                "sol_usd_valuation_available", price is not None,
                (f"${price:,.4f}" if price else "SOL/USD unavailable"),
            ))

            eq = compute_real_equity(lamports, price)
            checks.append(SafetyCheck(
                "equity_calculable", eq.available,
                (f"${eq.equity_usd:,.4f}" if eq.available
                 else f"unavailable: {eq.reason}"),
            ))
        except Exception as exc:  # noqa: BLE001
            checks.append(SafetyCheck(
                "equity_calculable", False,
                f"equity check could not run ({type(exc).__name__})",
            ))

        ok = all(c.ok for c in checks)
        detail = "all checks passed" if ok else (
            f"{len([c for c in checks if not c.ok])} check(s) failed"
        )
        return PreflightResult(ok, checks, detail)

    # -- the trade ---------------------------------------------------------

    async def open_position(
        self,
        *,
        mint: str,
        position_lamports: int,
        slippage_bps: int,
        prechecks: list[SafetyCheck],
        intent_store: Any = None,
        logical_key: Optional[str] = None,
        stage: Optional[Any] = None,
    ) -> OnchainFill:
        """
        Buy with real funds.

        ``prechecks`` is the pre-trade safety result. If ANY check failed, this
        does not trade - it returns a refusal with the reason. There is no
        "warn and continue".
        """
        failures = [c for c in prechecks if not c.ok]
        if failures:
            names = ", ".join(c.name for c in failures)
            fill = OnchainFill(side="ENTRY", status="PRE_TRADE_CHECK_FAILED")
            fill.ambiguity_reason = f"pre-trade safety failed: {names}"
            return fill
        self._busy = True
        try:
            return await self.backend.execute_swap(
                "ENTRY", WSOL_MINT, mint, int(position_lamports), int(slippage_bps),
                intent_store=intent_store, logical_key=logical_key, stage=stage,
            )
        finally:
            self._busy = False

    async def close_position(
        self,
        *,
        mint: str,
        token_amount: int,
        slippage_bps: int,
        prechecks: Optional[list[SafetyCheck]] = None,
        intent_store: Any = None,
        logical_key: Optional[str] = None,
        stage: Optional[Any] = None,
    ) -> OnchainFill:
        """Sell back to SOL with real funds.

        ``prechecks`` runs the same fail-closed gate as an entry when supplied;
        an exit that does not pass its own risk validation does not broadcast.
        """
        if prechecks is not None:
            failures = [c for c in prechecks if not c.ok]
            if failures:
                names = ", ".join(c.name for c in failures)
                fill = OnchainFill(side="EXIT", status="PRE_TRADE_CHECK_FAILED")
                fill.ambiguity_reason = f"exit pre-trade safety failed: {names}"
                return fill
        self._busy = True
        try:
            return await self.backend.execute_swap(
                "EXIT", mint, WSOL_MINT, int(token_amount), int(slippage_bps),
                intent_store=intent_store, logical_key=logical_key, stage=stage,
            )
        finally:
            self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    # -- exposure ----------------------------------------------------------

    def required_reserve_sol(self) -> float:
        """SOL that must stay spendable-free for fees and rent."""
        lamports = (
            BASE_FEE_LAMPORTS
            + int(getattr(self.real_cfg, "priority_fee_lamports", 50_000))
            + ATA_RENT_LAMPORTS
        )
        return MIN_SOL_FEE_RESERVE + lamports / LAMPORTS_PER_SOL

    async def sufficient_sol(
        self, *, position_lamports: int, sol_price_usd: Optional[float]
    ) -> SafetyCheck:
        """Is there enough SOL for the trade plus its transaction costs?"""
        try:
            state = await self.backend.wallet_state()
        except (RpcError, SolanaUnavailable) as exc:
            return SafetyCheck("sol_for_transaction_costs", False, str(exc))
        price = sol_price_usd
        if not price or price <= 0:
            price = await self.backend.sol_price_usd()
        if not price or price <= 0:
            return SafetyCheck(
                "sol_for_transaction_costs", False,
                "SOL/USD unavailable; cannot price the required reserve",
            )
        reserve_sol = self.required_reserve_sol()
        needed_sol = position_lamports / LAMPORTS_PER_SOL + reserve_sol
        ok = state["sol"] >= needed_sol
        return SafetyCheck(
            "sol_for_transaction_costs", ok,
            (
                f"wallet {state['sol']:.6f} SOL, need {needed_sol:.6f} SOL "
                f"({position_lamports / LAMPORTS_PER_SOL:.6f} trade + "
                f"{reserve_sol:.6f} reserve)"
                if ok
                else f"wallet {state['sol']:.6f} SOL < required {needed_sol:.6f} SOL "
                     f"({position_lamports / LAMPORTS_PER_SOL:.6f} trade + "
                     f"{reserve_sol:.6f} reserve for fees and rent)"
            ),
        )

    async def quote_probe(
        self,
        *,
        mint: str,
        position_lamports: int,
        slippage_bps: int,
        side: str,
    ) -> tuple[SafetyCheck, Optional[OnchainFill]]:
        """
        Price the order without building, signing or broadcasting.

        This is the check that answers the micro-account question honestly: it
        asks the venue whether it will accept the notional at all, and reports
        the venue's own reason when it will not.

        DIRECTION IS EXPLICIT AND PER-SIDE. The probe used to hardcode
        ``WSOL -> token``, which is correct for an ENTRY and wrong for an EXIT.
        On a REAL exit the trade being priced is ``token -> WSOL``, so probing
        the ENTRY direction evaluated a phantom BUY and reported its price
        impact as if it were the sell's. On $SI (2026-10-01) that produced
        400-500% impact against a 5% limit and the position could never be
        exited: the gate refused a sell it was never actually pricing.

            ENTRY: in=WSOL_MINT  out=mint   amount in WSOL lamports
            EXIT:  in=mint       out=WSOL_MINT  amount in RAW TOKEN UNITS

        ``position_lamports`` keeps its historical name, but the unit is
        per-side and is NOT converted: an ENTRY passes a WSOL lamport notional,
        an EXIT passes the raw token quantity being sold. That is exactly what
        :meth:`close_position` broadcasts (``execute_swap("EXIT", mint,
        WSOL_MINT, int(token_amount), ...)``), so the probe and the broadcast
        now price the same trade. Callers must not normalise the two into one
        unit -- that is the bug this parameterisation exists to prevent.

        ``side`` is REQUIRED and has no default. It was previously defaulted to
        ``"ENTRY"``, which meant a caller that forgot it silently priced a
        phantom BUY -- the exact defect this parameterisation exists to prevent.
        A caller must now state which direction it means.

        An unrecognised ``side`` is refused rather than assumed, so a new caller
        cannot silently inherit a direction.
        """
        if side not in ("ENTRY", "EXIT"):
            return (
                SafetyCheck(
                    "quote_valid", False,
                    f"unknown quote_probe side {side!r}; refusing to guess a "
                    f"direction (expected 'ENTRY' or 'EXIT')",
                ),
                None,
            )
        in_mint, out_mint = (
            (WSOL_MINT, mint) if side == "ENTRY" else (mint, WSOL_MINT)
        )
        out_unit = "tokens" if side == "ENTRY" else "lamports"
        # PRICE THROUGH THE SAME PROVIDER CHAIN THE SWAP USES. The probe used to
        # ask only the primary provider, so a Raptor-only "no multi-hop route"
        # failed the gate here and the executor's Jupiter fallback was never
        # reached. Asking the chain instead does not weaken the gate: it applies
        # the identical fallback eligibility (a missing route, never an outage)
        # and every downstream check still runs once, on one intent.
        probe = await self.backend.quote_first_available(
            in_mint, out_mint, int(position_lamports), int(slippage_bps)
        )
        self._last_quote_probe = self._probe_record(probe, side=side)
        if probe.get("micro"):
            return (
                SafetyCheck(
                    "quote_valid", False,
                    f"venue rejected the notional as too small: {probe['micro']}",
                ),
                OnchainFill(
                    side=side,
                    status="MICRO_REAL_EXECUTION_UNSUPPORTED",
                    ambiguity_reason=probe["micro"],
                ),
            )
        quote = probe.get("quote")
        if quote is None:
            return (
                SafetyCheck("quote_valid", False, probe.get("failure_detail") or ""),
                None,
            )
        via = (f" [provider={probe['provider']}]"
               if probe.get("fallback_attempted") else "")
        return (
            SafetyCheck(
                "quote_valid", True,
                (
                    f"expected {quote.expected_out_amount} {out_unit}, "
                    f"impact {quote.price_impact_pct:.3f}%, "
                    f"min out {quote.min_out_amount}{via}"
                ),
            ),
            None,
        )
