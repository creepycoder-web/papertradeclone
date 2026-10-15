"""Legacy REAL emergency adapter import compatibility."""
from __future__ import annotations

from typing import Tuple

from .adapters import RealEmergencyAdapter as _BaseRealEmergencyAdapter
from .adapters import RealSellIntent


class RealEmergencyAdapter(_BaseRealEmergencyAdapter):
    async def attempt_sign_and_broadcast(self, intent: RealSellIntent) -> Tuple[bool, str]:
        self._enforce_real_gate("sign_and_broadcast")
        return False, "REAL_TRADING_DISABLED"
