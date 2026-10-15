"""The three price layers, as three different types.

WHY THIS MODULE EXISTS
----------------------
On 2026-10-02 the $GOAT REAL position was reported to the operator as::

    ✅ ปิดออเดอร์
    GOAT
    ผลลัพธ์: +71.56%  $+0.0016

Both halves of that one line came from two different accounting systems, and
neither was the trade's return:

* ``+71.56%`` was ``exit_mark / entry_mark - 1`` where BOTH prices were
  DexScreener ``priceUsd`` observations -- not executable, not realizable.
* ``$+0.0016`` was a genuine settlement-derived dollar PnL.

The actual executable price had moved ``+6.38%``. The trade's true all-in cash
result was ``-$0.182494`` (-42.91%), because 1,513,840 lamports of token
account rent left the wallet on entry and was never returned.

The defect was not a wrong number. It was that three different kinds of number
shared one field name, so a spot mark could be read, stored, divided and
displayed as an execution price with no type-level distinction anywhere.

THE THREE LAYERS
----------------
``OBSERVED_MARK``
    A third-party indexer's view of some pool. Useful for features, signal,
    liquidity, market cap. It is not sized to this order, carries no impact and
    cannot be transacted against.

``EXECUTABLE_QUOTE``
    What a named venue will actually do for *this* size, right now. This is the
    only price that can become an order, and it is the only price a rejection
    may be based on.

``ACTUAL_FILL``
    What the chain says happened, derived from settled on-chain amounts. This is
    the only price that may appear in realized PnL.

WHAT THIS MODULE GUARANTEES
---------------------------
* A direction is declared, never inferred. ``BUY`` is always
  ``quote -> token`` and ``SELL`` is always ``token -> quote``; a call that
  tries to build the other one raises rather than quietly quoting the wrong
  trade (:class:`SwapDirectionError`).
* An effective price is always derived from two raw amounts in the correct
  orientation for the declared side, and is ``None`` -- never ``0.0`` -- when
  the amounts cannot support the division.
* Decimal normalization is explicit at every boundary. A price is quoted per
  WHOLE token and per RAW unit separately, and both are derived, never
  hand-written.
* Mark returns, executable returns and realized returns are three different
  numbers with three different names. :class:`PriceReturns` holds all three and
  they are never summed or substituted for one another.
* Provenance travels with the value. Every layer records provider, venue,
  pool/curve identity, mints, timestamps and model, so a downstream consumer
    cannot read one layer as another.

THIS MODULE COMPUTES NOTHING OBSERVABLE. It is arithmetic, types and invariants
only. See :mod:`autopaper.execution.quoting` for the providers that produce the
values, and :mod:`autopaper.pricing` for the shared low-level formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..pricing import (
    LAMPORTS_PER_SOL,
    PriceKind,
    divergence,
    execution_price_per_token,
    observed_return_pct,
    realized_execution_price,
)

#: Native SOL and wrapped SOL are the same spendable asset for every purpose in
#: this system. The executor's legs are always expressed in wSOL so that a
#: provider sees a concrete SPL mint; the empty string remains accepted wherever
#: older code passed it.
WSOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

#: Decimals for the quote assets this system settles in. Token decimals are
#: ALWAYS read from chain and never assumed from this table.
QUOTE_ASSET_DECIMALS = {WSOL_MINT: 9, USDC_MINT: 6}


class SwapDirectionError(ValueError):
    """A swap leg was built against the wrong direction.

    Raised, never repaired. A direction that is "fixed" by swapping the mints
    silently is a direction that was never checked.
    """

    def __init__(self, message: str, *, side: Any = None,
                 input_mint: str = "", output_mint: str = "") -> None:
        super().__init__(message)
        self.side = side
        self.input_mint = input_mint
        self.output_mint = output_mint

    def as_dict(self) -> dict:
        return {"error": "swap_direction", "detail": str(self),
                "side": str(getattr(self.side, "value", self.side)),
                "input_mint": self.input_mint,
                "output_mint": self.output_mint}


class Side(str, Enum):
    """Trade direction. Declared, never inferred.

    ``BUY``  -- always ``quote -> token``. We spend the quote asset to acquire
                the token. ``input_mint`` MUST be the quote asset.
    ``SELL`` -- always ``token -> quote``. We give up the token to receive the
                quote asset. ``output_mint`` MUST be the quote asset.
    """

    BUY = "BUY"
    SELL = "SELL"

    @classmethod
    def coerce(cls, value: Any) -> "Side":
        """Normalise ``"BUY"`` / ``"buy"`` / :class:`Side` to a :class:`Side`.

        Refuses anything else. A caller that cannot state its direction does not
        get a guessed one.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            upper = value.strip().upper()
            for member in cls:
                if member.value == upper:
                    return member
        raise SwapDirectionError(
            f"unknown swap side {value!r}; a direction must be declared as "
            f"'BUY' or 'SELL' and is never inferred from context",
            side=value,
        )

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


# ---------------------------------------------------------------------------
# legs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwapLeg:
    """One declared leg of a swap, in RAW integer units.

    Raw integers only. A float token amount cannot survive a mint with 9
    decimals and must never be constructed here.
    """

    side: Side
    input_mint: str
    output_mint: str
    input_amount: int
    output_amount: int
    input_decimals: int
    output_decimals: int

    def __post_init__(self) -> None:
        assert_side_leg(self)

    @property
    def input_amount_whole(self) -> float:
        return self.input_amount / (10 ** self.input_decimals)

    @property
    def output_amount_whole(self) -> float:
        return self.output_amount / (10 ** self.output_decimals)

    @property
    def token_decimals(self) -> int:
        """Decimals of the TOKEN asset, whichever side it happens to be on.

        A BUY has the token on the output leg and a SELL has it on the input
        leg. Any price-per-token figure must use the TOKEN's decimals, so this
        is the only place that lookup is written down.
        """
        return (self.output_decimals if self.side is Side.BUY
                else self.input_decimals)

    @property
    def quote_decimals(self) -> int:
        """Decimals of the QUOTE asset, whichever side it happens to be on."""
        return (self.input_decimals if self.side is Side.BUY
                else self.output_decimals)

    @property
    def effective_price(self) -> Optional[float]:
        """Quote RAW units per RAW output-asset unit, oriented for ``side``.

        ``None`` when the amounts cannot support the division. Never ``0.0``:
        a zero price is a missing value that once reached a journal wearing a
        measurement's name.
        """
        return execution_price_per_token(
            side=self.side.value,
            input_amount=self.input_amount,
            output_amount=self.output_amount,
        )

    @property
    def effective_price_per_whole_token(self) -> Optional[float]:
        """
        WHOLE quote units per WHOLE token, with both decimal scales applied once.

        This is the only per-token figure a human should ever be shown. It is
        asset-agnostic: SOL-per-token for a SOL-quoted leg, USDC-per-token for a
        USDC-quoted leg, and ``None`` when either amount is missing.

        Getting this wrong is not cosmetic. Multiplying the SELL leg by the
        OUTPUT decimals instead of the TOKEN decimals inflates a lamport-per-
        token figure by 10^9, which is precisely the class of unit error this
        module exists to make impossible.
        """
        per_raw = self.effective_price
        if per_raw is None:
            return None
        scale = (10 ** self.token_decimals) / (10 ** self.quote_decimals)
        return per_raw * scale

    def as_dict(self) -> dict:
        return {
            "side": self.side.value,
            "input_mint": self.input_mint,
            "output_mint": self.output_mint,
            "input_amount": self.input_amount,
            "output_amount": self.output_amount,
            "input_decimals": self.input_decimals,
            "output_decimals": self.output_decimals,
            "effective_price": self.effective_price,
            "effective_price_per_whole_token": self.effective_price_per_whole_token,
        }


def resolve_side_from_mints(input_mint: str, output_mint: str, *,
                            declared_side: Any = None,
                            quote_mints: Optional[set] = None) -> Side:
    """
    Establish a swap's direction from its mints, with a stated invariant.

    THE INVARIANT
    -------------
    Exactly one end of a swap leg is the quote asset -- the one we pay in or get
    out in. So:

    * input is the quote asset, output is not  -> ``BUY``  (quote -> token)
    * output is the quote asset, input is not  -> ``SELL`` (token -> quote)
    * both are quote assets, or neither is     -> refuse

    This is not "guessing a direction from context". It is a checked rule with
    a fail-closed answer, and it is applied to the two mints a caller already
    had to name explicitly. It exists so that the older positional provider
    signature ``quote(input_mint, output_mint, amount, slippage)`` -- which
    predates any ``side`` argument -- can be given a direction safely instead of
    either (a) carrying none, or (b) dividing input by output on a SELL.

    A caller that KNOWS its direction should pass ``declared_side``, and a
    declared side that contradicts the mints is a bug in the caller: it raises
    rather than being overridden, because silently "fixing" a caller's
    direction is how a swap gets sent the wrong way round.
    """
    qm = set(quote_mints) if quote_mints else {WSOL_MINT, USDC_MINT}
    in_is_quote = input_mint in qm
    out_is_quote = output_mint in qm
    if in_is_quote == out_is_quote:
        raise SwapDirectionError(
            "a swap leg must have exactly one quote-asset end "
            f"(got input={input_mint!r} output={output_mint!r}); refusing to "
            f"choose a direction",
            input_mint=input_mint, output_mint=output_mint)
    derived = Side.BUY if in_is_quote else Side.SELL
    if declared_side is not None:
        declared = Side.coerce(declared_side)
        if declared is not derived:
            raise SwapDirectionError(
                f"declared side {declared.value} contradicts the mints: the "
                f"quote asset is on the "
                f"{'input' if in_is_quote else 'output'} side, so this is a "
                f"{derived.value}",
                side=declared, input_mint=input_mint, output_mint=output_mint)
        return declared
    return derived


def assert_side_leg(leg: "SwapLeg") -> None:
    """
    Structural validity of a leg. The direction rule itself lives in
    :func:`leg_for_buy` / :func:`leg_for_sell`, which are the only sanctioned
    constructors, because deciding which mint is "the token" requires the caller
    to say so.

    What is enforced here is what can be checked from the leg alone: the side is
    a real :class:`Side`, both mints are named and distinct, decimals are sane,
    and every amount that is present is strictly positive.
    """
    side = leg.side
    if not isinstance(side, Side):
        raise SwapDirectionError("leg side must be a Side",
                                 side=side,
                                 input_mint=leg.input_mint,
                                 output_mint=leg.output_mint)
    if not leg.input_mint or not leg.output_mint:
        raise SwapDirectionError(
            "a swap leg must name both an input and an output mint",
            side=side, input_mint=leg.input_mint,
            output_mint=leg.output_mint)
    if leg.input_mint == leg.output_mint:
        raise SwapDirectionError(
            "a swap leg cannot have the same input and output mint",
            side=side, input_mint=leg.input_mint,
            output_mint=leg.output_mint)
    if leg.input_decimals < 0 or leg.output_decimals < 0:
        raise SwapDirectionError(
            "leg decimals must be non-negative",
            side=side, input_mint=leg.input_mint,
            output_mint=leg.output_mint)
    if leg.input_amount < 0 or leg.output_amount < 0:
        raise SwapDirectionError(
            "leg amounts are raw integers and may not be negative",
            side=side, input_mint=leg.input_mint,
            output_mint=leg.output_mint)


def require_token_decimals(token_decimals: Any) -> int:
    """
    Validate the token's decimals before any price is derived from them.

    A per-token price computed from an UNKNOWN scale is wrong by a power of
    ten, and it looks entirely plausible. On 2026-10-02 the 6-decimal scale of a
    Token-2022 mint had to be reconstructed by hand because nothing in the
    runtime recorded it. An absent value is therefore refused here rather than
    defaulted to 6 or 9.
    """
    if token_decimals is None:
        raise ValueError(
            "the token's decimals are unknown; no per-token price may be "
            "stated. They must be read from the mint on chain, never assumed")
    try:
        dec = int(token_decimals)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"token decimals are not an integer: {token_decimals!r}") \
            from exc
    if dec < 0 or dec > 18:
        raise ValueError(f"token decimals out of range: {dec}")
    return dec


def leg_for_buy(*, token_mint: str, quote_mint: str, quote_amount: int,
                quote_decimals: Optional[int] = None,
                token_decimals: Any, expected_token_amount: Optional[int] = None,
                ) -> SwapLeg:
    """Build a ``quote -> token`` leg, with the direction enforced."""
    if not token_mint or not quote_mint:
        raise SwapDirectionError("a BUY leg needs both a token and a quote mint",
                                 side=Side.BUY)
    if token_mint == quote_mint:
        raise SwapDirectionError("token and quote mint are identical",
                                 side=Side.BUY)
    qdec = (QUOTE_ASSET_DECIMALS.get(quote_mint, 9)
            if quote_decimals is None else int(quote_decimals))
    if quote_amount <= 0:
        raise SwapDirectionError("a BUY leg needs a positive quote amount",
                                 side=Side.BUY)
    return SwapLeg(side=Side.BUY, input_mint=quote_mint, output_mint=token_mint,
                   input_amount=int(quote_amount),
                   output_amount=int(expected_token_amount or 0),
                   input_decimals=qdec,
                   output_decimals=require_token_decimals(token_decimals))


def leg_for_sell(*, token_mint: str, quote_mint: str, token_amount: int,
                 quote_decimals: Optional[int] = None,
                 token_decimals: Any, expected_quote_amount: Optional[int] = None,
                 ) -> SwapLeg:
    """Build a ``token -> quote`` leg, with the direction enforced."""
    if not token_mint or not quote_mint:
        raise SwapDirectionError("a SELL leg needs both a token and a quote mint",
                                 side=Side.SELL)
    if token_mint == quote_mint:
        raise SwapDirectionError("token and quote mint are identical",
                                 side=Side.SELL)
    qdec = (QUOTE_ASSET_DECIMALS.get(quote_mint, 9)
            if quote_decimals is None else int(quote_decimals))
    if token_amount <= 0:
        raise SwapDirectionError("a SELL leg needs a positive token amount",
                                 side=Side.SELL)
    return SwapLeg(side=Side.SELL, input_mint=token_mint, output_mint=quote_mint,
                   input_amount=int(token_amount),
                   output_amount=int(expected_quote_amount or 0),
                   input_decimals=require_token_decimals(token_decimals),
                   output_decimals=qdec)


# ---------------------------------------------------------------------------
# layer 1: observed mark
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservedMark:
    """A third-party observation of a pool. Never an offer.

    This is what DexScreener/GeckoTerminal produce and it is the RIGHT input for
    features, liquidity, market cap, momentum and L25's market read. It is
    categorically not an input to realized PnL, and it may not be used as an
    entry or exit price.
    """

    mint: str
    price: float
    #: Human label for the number's unit, e.g. ``"USD/whole_token"``.
    price_unit: str = "USD/whole_token"
    quote_mint: str = USDC_MINT
    #: Who produced it: ``"dexscreener"``, ``"geckoterminal"``, ...
    source: str = "unknown"
    #: The venue label the source reported, e.g. ``"pumpfun"``, ``"pumpswap"``.
    venue: Optional[str] = None
    #: The pool/pair identity the observation actually describes. This is the
    #: field that makes a cross-venue mismatch detectable: a mark whose pool
    #: differs from the executed pool is a mark about somewhere else.
    pool: Optional[str] = None
    quote_mint_source_mint: Optional[str] = None
    observed_at: float = 0.0
    #: The source's own timestamp when it reports one, else ``None``. ``None``
    #: means "not reported" and is never replaced with the receive time.
    source_timestamp: Optional[float] = None
    received_at: Optional[float] = None
    liquidity_usd: Optional[float] = None
    market_cap_usd: Optional[float] = None
    #: Free-form, sanitised provenance for the record.
    provenance: dict = field(default_factory=dict)
    #: Whether the provider considers its own figure current.
    stale: bool = False
    #: ``"live"`` / ``"stale"`` / ``"unusable"`` / ``"missing"``
    quality: str = "live"

    kind: PriceKind = PriceKind.OBSERVED_MARK

    @property
    def is_executable(self) -> bool:
        """Always ``False``. An observed mark is not an offer for any size."""
        return False

    @property
    def pool_key(self) -> str:
        return f"{self.venue or '?'}:{self.pool or '?'}"

    def as_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "layer": "OBSERVED_MARK",
            "mint": self.mint,
            "quote_mint": self.quote_mint,
            "price": self.price,
            "price_unit": self.price_unit,
            "source": self.source,
            "venue": self.venue,
            "pool": self.pool,
            "observed_at": self.observed_at,
            "source_timestamp": self.source_timestamp,
            "received_at": self.received_at,
            "liquidity_usd": self.liquidity_usd,
            "market_cap_usd": self.market_cap_usd,
            "stale": self.stale,
            "quality": self.quality,
            "provenance": dict(self.provenance),
        }

    # -- construction from the existing market layer -----------------------

    @classmethod
    def from_observation(cls, obs: Any, *, now: Optional[float] = None,
                         source: str = "unknown") -> "ObservedMark":
        """Wrap an existing :class:`autopaper.models.Observation`.

        The market layer already carries the pool identity in
        ``obs.extra['pairAddress']`` and the venue in ``obs.venue``. Preserving
        them here is what allows a later comparison against the pool that
        actually executed.
        """
        extra = dict(getattr(obs, "extra", None) or {})
        price = getattr(obs, "price_usd", None)
        observed_at = float(getattr(obs, "observed_at", 0.0) or 0.0)
        age_ms = None
        if now is not None and observed_at:
            age_ms = max(0.0, (now - observed_at) * 1000.0)
        stale = bool(getattr(obs, "stale", False))
        return cls(
            mint=getattr(obs, "mint", "") or "",
            price=float(price) if price is not None else float("nan"),
            quote_mint=extra.get("quote_mint") or USDC_MINT,
            source=source,
            venue=getattr(obs, "venue", None),
            pool=extra.get("pairAddress"),
            observed_at=observed_at,
            received_at=float(getattr(obs, "received_at", observed_at) or 0.0),
            liquidity_usd=getattr(obs, "liquidity_usd", None),
            market_cap_usd=getattr(obs, "market_cap_usd", None),
            provenance={
                "dex_id": extra.get("dex_id"),
                "price_native": getattr(obs, "price_native", None),
                "pair_created_at": extra.get("pairCreatedAt"),
                "age_ms": age_ms,
            },
            stale=stale,
            quality="stale" if stale else "live",
        )


# ---------------------------------------------------------------------------
# layer 2: executable quote
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteHop:
    """One hop of an executable route. Sanitized: no payloads, no signatures."""

    venue: str = ""
    program_id: str = ""
    pool: str = ""
    input_mint: str = ""
    output_mint: str = ""
    amount_in: str = ""
    amount_out: str = ""
    percent: float = 0.0

    def as_dict(self) -> dict:
        return {
            "venue": self.venue, "program_id": self.program_id,
            "pool": self.pool, "input_mint": self.input_mint,
            "output_mint": self.output_mint, "amount_in": self.amount_in,
            "amount_out": self.amount_out, "percent": self.percent,
        }


@dataclass(frozen=True)
class ExecutableQuote:
    """What a named venue will do for THIS size, right now.

    The only price an entry or exit decision may be refused on, and the only
    price (besides settlement) that may be compared against an observed mark.

    ``venue_quoted`` distinguishes a quote the venue itself returned from a
    quote this system derived from on-chain state. Both are executable; only the
    first is the venue's own answer. Callers that need a venue's own quote
    (``venue_quoted=True``) must check the flag rather than assume.
    """

    side: Side
    leg: SwapLeg
    #: ``"jupiter"`` / ``"raptor"`` / ``"pump_bonding_curve"`` /
    #: ``"pumpswap"``
    provider: str
    #: Venue label as the provider reports it, e.g. ``"pumpswap"``, ``"pumpfun"``.
    venue: Optional[str] = None
    #: Pool / curve / route identity. ``None`` means the provider did not report
    #: one, which is recorded rather than invented.
    pool: Optional[str] = None
    program_id: Optional[str] = None
    route: tuple = ()
    price_impact_pct: float = 0.0
    slippage_bps: int = 0
    #: Venue-side minimum output protection. ``None`` when the venue gave none.
    min_output_amount: Optional[int] = None
    quoted_at: float = 0.0
    received_at: float = 0.0
    context_slot: Optional[int] = None
    #: USD per WHOLE quote-asset unit (e.g. ``121.6`` for SOL). ``None`` means
    #: "not known"; USD conversions then return ``None`` rather than guessing.
    quote_asset_usd: Optional[float] = None
    #: True when the venue itself answered. False for a chain-derived model.
    venue_quoted: bool = True
    #: The model name when the price was derived rather than returned.
    model: Optional[str] = None
    stale: bool = False
    quality: str = "live"
    provenance: dict = field(default_factory=dict)

    kind: PriceKind = PriceKind.VENUE_QUOTE

    # -- derived, never stored twice ---------------------------------------

    @property
    def is_executable(self) -> bool:
        return not self.stale and self.quality != "unusable"

    @property
    def effective_price(self) -> Optional[float]:
        return self.leg.effective_price

    @property
    def effective_price_per_whole_token(self) -> Optional[float]:
        return self.leg.effective_price_per_whole_token

    @property
    def price_usd_per_whole_token(self) -> Optional[float]:
        """USD per WHOLE token, or ``None`` when the quote-asset USD price is
        unknown. A quote with no USD conversion is still usable in quote-asset
        units; it is simply not convertible, and saying so is correct."""
        per_whole = self.effective_price_per_whole_token
        if per_whole is None or self.quote_asset_usd is None:
            return None
        if self.quote_asset_usd <= 0:
            return None
        return per_whole * self.quote_asset_usd

    @property
    def age_s(self) -> Optional[float]:
        if not self.received_at:
            return None
        import time as _time
        return max(0.0, _time.time() - self.received_at)

    @property
    def pool_key(self) -> str:
        return f"{self.venue or self.provider}:{self.pool or '?'}"

    def as_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "layer": "EXECUTABLE_QUOTE",
            "side": self.side.value,
            "provider": self.provider,
            "venue": self.venue,
            "pool": self.pool,
            "program_id": self.program_id,
            "route": [h.as_dict() if isinstance(h, RouteHop) else dict(h)
                      for h in self.route],
            "price_impact_pct": self.price_impact_pct,
            "slippage_bps": self.slippage_bps,
            "min_output_amount": self.min_output_amount,
            "quoted_at": self.quoted_at,
            "received_at": self.received_at,
            "context_slot": self.context_slot,
            "quote_asset_usd": self.quote_asset_usd,
            "venue_quoted": self.venue_quoted,
            "model": self.model,
            "stale": self.stale,
            "quality": self.quality,
            "leg": self.leg.as_dict(),
            "effective_price": self.effective_price,
            "effective_price_per_whole_token":
                self.effective_price_per_whole_token,
            "price_usd_per_whole_token": self.price_usd_per_whole_token,
            "provenance": dict(self.provenance),
        }


# ---------------------------------------------------------------------------
# layer 3: actual fill
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostComponents:
    """
    Every explicit cost of a round trip, itemised.

    Nothing is netted away. A cost that is unrecovered, a cost that was
    refunded, and a cost that is still locked in an account are three different
    facts, and the $NOTIFY/$GOAT shape (1,513,840 lamports of token-account rent
    silently omitted from the cost basis) is exactly what happens when they are
    collapsed into one number.
    """

    #: Swap principal actually committed to the venue, in quote-asset raw units.
    swap_input_amount: int = 0
    #: Network fees (``meta.fee``) in lamports, entry + exit.
    network_fee_lamports: int = 0
    #: Token-account / account-creation rent in lamports.
    account_rent_lamports: int = 0
    #: Venue/protocol fees already netted inside the settled output.
    venue_fee_amount: Optional[int] = None
    #: Rent the system has PROVEN was returned to the wallet (closed account
    #: confirmed on chain). Only this may be netted out of the cost basis.
    rent_recovered_lamports: int = 0
    #: Any other explicit, itemised cost the caller wants recorded.
    other_lamports: int = 0
    #: Free-text reason a component could not be established, so an unknown
    #: stays unknown instead of becoming a zero.
    unreconciled_reason: Optional[str] = None

    @property
    def rent_locked_lamports(self) -> int:
        """Rent that left the wallet and has not been proven to come back."""
        return max(0, int(self.account_rent_lamports)
                   - int(self.rent_recovered_lamports))

    def as_dict(self) -> dict:
        return {
            "swap_input_amount": self.swap_input_amount,
            "network_fee_lamports": self.network_fee_lamports,
            "account_rent_lamports": self.account_rent_lamports,
            "venue_fee_amount": self.venue_fee_amount,
            "rent_recovered_lamports": self.rent_recovered_lamports,
            "rent_locked_lamports": self.rent_locked_lamports,
            "other_lamports": self.other_lamports,
            "unreconciled_reason": self.unreconciled_reason,
        }


@dataclass(frozen=True)
class ActualFill:
    """What the chain says happened. The ONLY source of realized PnL.

    ``input_amount`` / ``output_amount`` are the SWAP legs (what went to and
    from the venue), never a wallet balance delta. The wallet delta is recorded
    separately on :class:`CostComponents` because it also carries rent and fees,
    and mixing the two is what produced a journalled "execution price" of
    ``0.0004634`` on 2026-10-02 -- lamports-per-token computed from a debit that
    included an account's rent.
    """

    side: Side
    leg: SwapLeg
    signature: str = ""
    slot: Optional[int] = None
    block_time: Optional[float] = None
    confirmed_at: Optional[float] = None
    fee_lamports: Optional[int] = None
    venue: Optional[str] = None
    pool: Optional[str] = None
    program_id: Optional[str] = None
    provider: Optional[str] = None
    #: The asset the proceeds actually arrived in. ``WSOL_MINT`` for native SOL.
    settlement_mint: Optional[str] = None
    settlement_decimals: int = 9
    costs: CostComponents = field(default_factory=CostComponents)
    provenance: dict = field(default_factory=dict)

    kind: PriceKind = PriceKind.ACTUAL_EXECUTION

    @property
    def is_executable(self) -> bool:
        return True

    @property
    def effective_price(self) -> Optional[float]:
        return self.leg.effective_price

    @property
    def effective_price_per_whole_token(self) -> Optional[float]:
        return self.leg.effective_price_per_whole_token

    @property
    def execution_price_usd(self) -> Optional[float]:
        """USD per WHOLE token, derived from settled amounts in the correct
        orientation for ``side`` and with the TOKEN's decimals applied once."""
        return realized_execution_price(
            side=self.side.value,
            input_amount=self.leg.input_amount,
            output_amount=self.leg.output_amount,
            sol_price_usd=self.provenance.get("quote_asset_usd"),
            token_decimals=self.leg.token_decimals,
        )

    def as_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "layer": "ACTUAL_FILL",
            "side": self.side.value,
            "signature": self.signature,
            "slot": self.slot,
            "block_time": self.block_time,
            "confirmed_at": self.confirmed_at,
            "fee_lamports": self.fee_lamports,
            "venue": self.venue,
            "pool": self.pool,
            "program_id": self.program_id,
            "provider": self.provider,
            "settlement_mint": self.settlement_mint,
            "settlement_decimals": self.settlement_decimals,
            "leg": self.leg.as_dict(),
            "effective_price": self.effective_price,
            "effective_price_per_whole_token":
                self.effective_price_per_whole_token,
            "execution_price_usd_per_whole_token": self.execution_price_usd,
            "costs": self.costs.as_dict(),
            "provenance": dict(self.provenance),
        }


# ---------------------------------------------------------------------------
# returns: three numbers, three names, never interchangeable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriceReturns:
    """
    The three returns of a position, kept apart on purpose.

    ``realized_pct`` is the only one that may be presented to an operator as
    profit. ``executable_pct`` is what the venue said the trade was worth.
    ``observed_mark_pct`` is what a third-party index thought happened; it is
    neither realizable nor a result, and it is named so it cannot be printed
    under a profit label by accident.

    Any of the three may be ``None``. A ``None`` means "not established", which
    is a legitimate and often correct state -- it is never replaced by ``0.0``.
    """

    realized_pct: Optional[float] = None
    executable_pct: Optional[float] = None
    observed_mark_pct: Optional[float] = None
    #: Why any of them is None. Never empty when one of them is None.
    reasons: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "realized_pct": self.realized_pct,
            "executable_pct": self.executable_pct,
            "observed_mark_pct": self.observed_mark_pct,
            "reasons": dict(self.reasons),
        }


def realized_return_pct(*, realized_pnl_usd: Optional[float],
                        cost_basis_usd: Optional[float]) -> Optional[float]:
    """``realized_pnl_usd / cost_basis_usd * 100``, or ``None``.

    Both legs are settlement-derived USD figures. There is deliberately no
    parameter for a price: this function cannot be handed a mark, because the
    whole $GOAT incident was a mark being used as if it were here.
    """
    if realized_pnl_usd is None or cost_basis_usd is None:
        return None
    try:
        pnl = float(realized_pnl_usd)
        basis = float(cost_basis_usd)
    except (TypeError, ValueError):
        return None
    if not (basis > 0) or pnl != pnl:
        return None
    return pnl / basis * 100.0


def mark_return_pct(*, exit_mark: Optional[float],
                    entry_mark: Optional[float]) -> Optional[float]:
    """Mark-to-mark return. Explicitly NOT a result.

    This is the ``+71.56%`` arithmetic from 2026-10-02, kept and named. It is
    retained because an operator comparing the bot's screen against an exchange
    screen needs to see what the bot was looking at -- and it is retained under
    this name precisely so it can never again be labelled "ผลลัพธ์".
    """
    return observed_return_pct(exit_mark, entry_mark)


def executable_return_pct(*, entry_quote: Optional[ExecutableQuote],
                          exit_quote: Optional[ExecutableQuote]) -> Optional[float]:
    """Quote-to-quote return, both legs executable and both sides correct.

    An ENTRY quote's price is the price we PAID and an EXIT quote's price is the
    price we RECEIVED, so the ratio is directly comparable. Returns ``None``
    when either leg is missing, unusable, or not in quote-asset units.
    """
    if entry_quote is None or exit_quote is None:
        return None
    if entry_quote.side is not Side.BUY or exit_quote.side is not Side.SELL:
        return None
    a = entry_quote.effective_price_per_whole_token
    b = exit_quote.effective_price_per_whole_token
    if a is None or b is None or a <= 0:
        return None
    return (b / a - 1.0) * 100.0


def mark_vs_quote_divergence(mark: Optional[float],
                             quote: Optional[float], *,
                             kind: str = "ENTRY_MARK_VS_QUOTE"
                             ) -> Any:
    """``(mark - quote) / quote`` for an executable pair, else ``None``.

    This is the comparison the 2026-10-02 audit needed and never performed: the
    observed mark against a price that could actually have been transacted at,
    on the same side, at the same moment.
    """
    return divergence(mark, quote, kind=kind)


def mark_venue_mismatch(mark: Optional[ObservedMark],
                        fill_or_quote: Any) -> bool:
    """
    True when the mark describes a DIFFERENT pool than the one that executed.

    ``fill_or_quote`` may be an :class:`ActualFill` or an
    :class:`ExecutableQuote`. A mark about another pool is not wrong, it is
    about somewhere else -- and presenting it as this position's price is the
    defect this predicate exists to make visible.

    Returns ``False`` (not ``True``) when either side does not report a pool:
    absence of an identity is not evidence of a mismatch, and it must not be
    laundered into one.
    """
    if mark is None or mark.pool is None or mark.pool == "":
        return False
    executed_pool = getattr(fill_or_quote, "pool", None)
    if executed_pool is None or executed_pool == "":
        return False
    return str(executed_pool) != str(mark.pool)


__all__ = [
    "CostComponents",
    "ExecutableQuote",
    "ObservedMark",
    "ActualFill",
    "PriceReturns",
    "RouteHop",
    "Side",
    "SwapDirectionError",
    "SwapLeg",
    "QUOTE_ASSET_DECIMALS",
    "USDC_MINT",
    "WSOL_MINT",
    "assert_side_leg",
    "executable_return_pct",
    "require_token_decimals",
    "resolve_side_from_mints",
    "leg_for_buy",
    "leg_for_sell",
    "mark_return_pct",
    "mark_venue_mismatch",
    "mark_vs_quote_divergence",
    "realized_return_pct",
    "LAMPORTS_PER_SOL",
]