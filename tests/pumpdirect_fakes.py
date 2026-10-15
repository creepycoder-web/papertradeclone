"""
Offline test doubles for the direct Pump / PumpSwap quote path.

EVERYTHING HERE IS A FIXTURE. No test in this file touches the network.

The account bytes are built by :mod:`tests.pumpdirect_fixtures`, which encodes
the DOCUMENTED layouts field by field rather than pasting a hex blob. A hex
blob would pass even if the decoder's offsets were shifted, because the fixture
and the decoder would be wrong in the same direction; building the bytes from
the same field list the IDL describes makes an offset error show up as a
decoding failure instead.
"""

from __future__ import annotations

import base64
from typing import Any, Sequence

from autopaper.same_market.pda import b58decode_pubkey, b58encode


class FakeAccount:
    """One account, shaped like the RPC client's ``RpcAccount``."""

    def __init__(
        self,
        address: str,
        owner: str,
        data: bytes = b"",
        *,
        lamports: int = 1_000_000,
    ) -> None:
        self.address = address
        self.owner = owner
        self.data_b64 = base64.b64encode(data).decode() if data else ""
        self.lamports = lamports
        self.executable = False
        self.rent_epoch = 0

    @property
    def is_empty(self) -> bool:
        return not self.data_b64

    def raw(self) -> bytes:
        return base64.b64decode(self.data_b64) if self.data_b64 else b""

    def as_dict(self) -> dict[str, Any]:
        return {"address": self.address, "owner": self.owner, "lamports": self.lamports}


class FakeRpc:
    """
    A read-only RPC stand-in.

    Only the three methods the reader uses exist. There is deliberately no
    ``send_transaction``: a fake that offered one would be inviting a test to
    depend on an execution capability the real harness does not have.
    """

    def __init__(
        self,
        accounts: dict[str, FakeAccount] | None = None,
        *,
        slot: int = 452_000_000,
        supply: int = 1_000_000_000_000_000,
        fail_addresses: Sequence[str] = (),
        supply_value: int | None = None,
    ) -> None:
        self.accounts = dict(accounts or {})
        self.slot = slot
        self.supply = supply_value if supply_value is not None else supply
        self.fail_addresses = set(fail_addresses)
        self.calls: list[tuple[str, Any]] = []
        self.multi_calls = 0

    async def get_slot(self) -> int:
        self.calls.append(("get_slot", None))
        return self.slot

    async def get_token_supply(self, mint: str) -> int:
        self.calls.append(("get_token_supply", mint))
        if mint in self.fail_addresses:
            raise RuntimeError("supply unavailable")
        return self.supply

    async def get_multiple_accounts(
        self, addresses: list[str], *, encoding: str = "base64",
        commitment: str = "confirmed",
    ) -> list[Any]:
        self.multi_calls += 1
        self.calls.append(("get_multiple_accounts", tuple(addresses)))
        return [self.accounts.get(a) for a in addresses]

    async def get_account_info(self, address: str, **kwargs: Any) -> Any:
        self.calls.append(("get_account_info", address))
        return self.accounts.get(address)

    # -- intent helpers ----------------------------------------------------

    @property
    def total_accounts_read(self) -> int:
        return sum(len(a) for name, a in self.calls if name == "get_multiple_accounts")

    def methods_used(self) -> set[str]:
        return {name for name, _ in self.calls}


class FakeJupiter:
    """A Jupiter result stand-in with the fields the stage reads."""

    def __init__(
        self,
        *,
        in_amount: int,
        out_amount: int,
        pool_keys: Sequence[str] = (),
        labels: Sequence[str] = (),
        timestamp: float = 1_000.0,
        latency_ms: float | None = 12.0,
        input_mint: str = "So11111111111111111111111111111111111111112",
        output_mint: str = "",
    ) -> None:
        from autopaper.same_market.jupiter import RouteHop

        self.input_mint = input_mint
        self.output_mint = output_mint
        self.in_amount = in_amount
        self.out_amount = out_amount
        self.other_amount_threshold = out_amount
        self.price_impact_pct = 0.01
        self.route_plan = tuple(
            RouteHop(
                label=labels[i] if i < len(labels) else None,
                amm_key=pool_keys[i] if i < len(pool_keys) else None,
                input_mint=input_mint,
                output_mint=output_mint,
                in_amount=in_amount,
                out_amount=out_amount,
                fee_amount=None,
                percent=100.0,
            )
            for i in range(max(len(pool_keys), len(labels)))
        )
        self.raw: dict[str, Any] = {}
        self.request_latency_ms = latency_ms
        self.timestamp = timestamp

    @property
    def pool_keys(self) -> tuple[str, ...]:
        return tuple(h.amm_key for h in self.route_plan if h.amm_key)


class FakeDexPair:
    """A DexScreener pair stand-in."""

    def __init__(
        self,
        *,
        pair_address: str,
        base_mint: str,
        dex_id: str = "pumpfun",
        price_usd: float | None = 4.4e-05,
        quote_mint: str = "So11111111111111111111111111111111111111112",
        chain_id: str = "solana",
    ) -> None:
        self.chain_id = chain_id
        self.dex_id = dex_id
        self.pair_address = pair_address
        self.base_token_address = base_mint
        self.base_token_symbol = "TEST"
        self.quote_token_address = quote_mint
        self.quote_token_symbol = "SOL"
        self.price_usd = price_usd
        self.price_native = None
        self.liquidity_usd = None
        self.volume_24h_usd = None
        self.fdv_usd = None
        self.market_cap_usd = None
        self.pair_created_at = None
        self.txns_5m = None
        self.txns_1h = None
        self.buys_5m = None
        self.sells_5m = None
        self.buys_1h = None
        self.sells_1h = None
        self.url = None


__all__ = ["FakeAccount", "FakeDexPair", "FakeJupiter", "FakeRpc"]
