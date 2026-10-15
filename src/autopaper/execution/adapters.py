"""Legacy emergency-close adapter compatibility.

The active runtime does not import this module. It remains importable for old
fixtures, with signing and broadcasting deliberately unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RealSellIntent:
    position_id: str
    mint: str
    market_identity: Dict[str, Any]
    route: List[str]
    input_amount: int
    expected_output: int
    quote_timestamp: float
    quote_age_ms: float
    slippage_bps: int
    provider: str
    transaction_b64: str
    last_valid_block_height: Optional[int] = None


class PaperExecutionAdapter:
    def __init__(self, *, sell_quote: Any = None, close_one: Any = None,
                 config: Any = None, intent_store: Any = None) -> None:
        self._sell_quote = sell_quote
        self._close_one = close_one
        self.config = config
        self._intent_store = intent_store

    async def quote_sell_best(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if self._sell_quote is None:
            return {"quote": None, "provider": "PAPER", "age_ms": 0.0,
                    "error": "NO_QUOTE_PROVIDER"}
        try:
            from autopaper.emergency_close import SellQuoteRequest
            quote = await self._sell_quote(SellQuoteRequest(
                position_id=request.get("position_id", ""), mint=request.get("mint", ""),
                tokens_raw=request.get("tokens_raw", 0),
                token_decimals=request.get("token_decimals", 0),
                venue_id=request.get("venue_id"), pool_address=request.get("pool_address")))
            return {"quote": quote, "provider": "PAPER", "age_ms": 0.0,
                    "error": None if quote is not None else "NO_QUOTE"}
        except Exception as exc:
            return {"quote": None, "provider": "PAPER", "age_ms": 0.0,
                    "error": f"{type(exc).__name__}: {exc}"}

    async def build_transaction_intent(self, position: Dict[str, Any],
                                       quote_info: Dict[str, Any]) -> RealSellIntent:
        if quote_info.get("quote") is None:
            raise RuntimeError(quote_info.get("error") or "NO_EXECUTABLE_QUOTE")
        return RealSellIntent(
            position_id=str(position.get("position_id") or position.get("signal_id") or ""),
            mint=str(position.get("mint") or ""),
            market_identity={"provider": quote_info.get("provider", "PAPER")},
            route=[], input_amount=int(position.get("tokens_raw") or 0), expected_output=0,
            quote_timestamp=0.0, quote_age_ms=0.0, slippage_bps=0,
            provider=quote_info.get("provider", "PAPER"), transaction_b64="")

    async def run_prebroadcast_safety_checks(self, position: Dict[str, Any],
                                             intent: RealSellIntent) -> None:
        if int(position.get("tokens_raw") or 0) != intent.input_amount:
            raise RuntimeError("INPUT_AMOUNT_MISMATCH")

    async def attempt_sign_and_broadcast(self, intent: RealSellIntent) -> Tuple[bool, str]:
        return True, "PAPER_PROCESSED"

    async def emergency_sell_position(self, position: Dict[str, Any]) -> Dict[str, Any]:
        try:
            quote = await self.quote_sell_best(position)
            intent = await self.build_transaction_intent(position, quote)
            await self.run_prebroadcast_safety_checks(position, intent)
            success, reason = await self.attempt_sign_and_broadcast(intent)
            return {"success": success, "reason": reason, "position_id": position.get("position_id"),
                    "mint": position.get("mint"), "intent": intent,
                    "error": None if success else reason}
        except Exception as exc:
            return {"success": False, "reason": str(exc), "position_id": position.get("position_id"),
                    "mint": position.get("mint"), "intent": None, "error": str(exc)}


class RealEmergencyAdapter:
    def __init__(self, config: Any, runtime_state: Any, intent_store: Any,
                 real_executor: Any = None) -> None:
        self.config = config
        self.runtime_state = runtime_state
        self.intent_store = intent_store
        self._real_executor = real_executor

    @property
    def is_real_trading_enabled(self) -> bool:
        return bool(getattr(self.runtime_state, "real_trading_enabled", False))

    def _enforce_real_gate(self, operation_name: str) -> None:
        if not self.is_real_trading_enabled:
            raise RuntimeError("REAL_TRADING_DISABLED")

    async def attempt_sign_and_broadcast(self, intent: RealSellIntent) -> Tuple[bool, str]:
        self._enforce_real_gate("sign_and_broadcast")
        return False, "REAL_TRADING_DISABLED"
