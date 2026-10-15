"""
Shared emergency close service that works with both PAPER and REAL adapters.

This service provides a strategy-independent orchestration layer for emergency
close operations, delegating to either PAPER or REAL execution adapters based
on configuration.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from autopaper.config import Config
from autopaper.execution.intents import ExecutionIntentStore


class EmergencyCloseService:
    """
    Strategy-independent emergency close service.
    
    This service owns sequencing, state, idempotency and reporting for emergency
    close operations, delegating the actual market/settlement logic to injected
    adapters (PAPER or REAL).
    """

    def __init__(
        self,
        *,
        config: Config,
        adapter: Any,  # PAPER or REAL execution adapter
        intent_store: ExecutionIntentStore,
        list_open: Any,  # Callable[[], Sequence[dict[str, Any]]]
        pause_entries: Optional[Any] = None,  # Callable[[bool], None]
        max_attempts: int = 3,
        backoff_s: float = 0.0,
        sleep: Any = time.sleep,
        now: Any = time.time,
        clock: Any = time.time,
    ) -> None:
        """
        Initialize the emergency close service.
        
        Args:
            config: Runtime configuration
            adapter: Execution adapter (PAPER or REAL) that implements:
                - quote_sell_best(request) -> quote info
                - build_transaction_intent(position, quote_info) -> intent
                - run_prebroadcast_safety_checks(position, intent) -> None
                - attempt_sign_and_broadcast(intent) -> (success, error_msg)
            intent_store: For idempotency and duplicate intent detection
            list_open: Function to list open positions
            pause_entries: Optional function to pause/unpause new entries
            max_attempts: Maximum retry attempts per position
            backoff_s: Backoff seconds between attempts
            sleep: Sleep function (injectable for testing)
            now: Current time function (injectable for testing)
            clock: Clock function (injectable for testing)
        """
        self.config = config
        self._adapter = adapter
        self._intent_store = intent_store
        self._list_open = list_open
        self._pause_entries = pause_entries
        self._max_attempts = max_attempts
        self._backoff_s = backoff_s
        self._sleep = sleep
        self._now = now
        self._clock = clock

    async def run_emergency_close(self) -> Dict[str, Any]:
        """
        Execute emergency close for all open positions.
        
        Returns:
            Dictionary with overall results:
                {
                    "success": bool,  # True if all positions processed
                    "timestamp": float,
                    "positions_processed": int,
                    "positions_closed": int,
                    "positions_failed": int,
                    "positions_skipped": int,
                    "details": List[Dict[str, Any]],  # Per-position results
                    "errors": List[str]  # Overall errors
                }
        """
        start_time = self._now()
        
        # Pause new entries if pause_entries callback provided
        if self._pause_entries is not None:
            self._pause_entries(True)
        
        try:
            # Get list of open positions
            open_positions = await self._list_open() if callable(self._list_open) else list(self._list_open)
            
            results = []
            positions_processed = 0
            positions_closed = 0
            positions_failed = 0
            positions_skipped = 0
            
            # Process each position
            for position in open_positions:
                position_id = str(position.get("position_id") or position.get("signal_id") or "unknown")
                mint = str(position.get("mint") or "unknown")
                
                # Process position with retry logic
                position_result = await self._process_position_with_retry(position)
                results.append(position_result)
                
                positions_processed += 1
                if position_result["success"]:
                    positions_closed += 1
                elif position_result.get("reason") == "POSITION_ALREADY_CLOSED":
                    positions_skipped += 1
                else:
                    positions_failed += 1
                
                # Backoff between positions (except after last)
                if self._backoff_s > 0 and position != open_positions[-1]:
                    await self._sleep(self._backoff_s)
            
            # Determine overall success
            overall_success = positions_failed == 0
            
            return {
                "success": overall_success,
                "timestamp": start_time,
                "positions_processed": positions_processed,
                "positions_closed": positions_closed,
                "positions_failed": positions_failed,
                "positions_skipped": positions_skipped,
                "details": results,
                "errors": [],  # Could collect specific errors here if needed
            }
            
        finally:
            # Always resume entries when done
            if self._pause_entries is not None:
                self._pause_entries(False)

            # Always resume entries when done
            if self._pause_entries is not None:
                self._pause_entries(False)

    async def _process_position_with_retry(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single position with retry logic.
        
        Args:
            position: Position dictionary to process
            
        Returns:
            Dictionary with position-level result
        """
        position_id = str(position.get("position_id") or position.get("signal_id") or "unknown")
        mint = str(position.get("mint") or "unknown")
        
        last_error = None
        
        for attempt in range(self._max_attempts):
            try:
                # Attempt to sell the position
                result = await self._adapter.emergency_sell_position(position)
                
                # If successful or position already closed, return immediately
                if result["success"] or result.get("reason") == "POSITION_ALREADY_CLOSED":
                    return result
                
                # Otherwise, store error and retry if we have attempts left
                last_error = result.get("reason", "Unknown error")
                if attempt < self._max_attempts - 1:
                    await self._sleep(self._backoff_s)
                    
            except Exception as exc:
                # Handle unexpected exceptions
                last_error = f"UNEXPECTED_ERROR: {exc}"
                if attempt < self._max_attempts - 1:
                    await self._sleep(self._backoff_s)
        
        # All attempts exhausted
        return {
            "success": False,
            "reason": last_error or "Max attempts exceeded",
            "position_id": position_id,
            "mint": mint,
            "attempts": self._max_attempts,
            "error": last_error,
        }
