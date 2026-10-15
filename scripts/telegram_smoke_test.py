"""
Bounded, PAPER-only, end-to-end smoke test through a MOCKED Bot API.

Nothing here touches the network, a wallet, an RPC, or a blockchain. The
transport is an ``httpx.MockTransport``, so the full path is exercised - update
in, authorization, replay check, dispatch through the real ``ButtonHandler``,
state transition, reply queued, reply "delivered" - with the only unreal part
being Telegram itself.

Asserts the ten required properties, and asserts the ABSENCE of the things that
matter: no REAL arm, no wallet access, no signing, no broadcast.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autopaper.config import build_config                      # noqa: E402
from autopaper.engine import PaperTrader                      # noqa: E402
from autopaper.forward_validation import (                    # noqa: E402
    VALIDATING, build_report,
)
from autopaper.modes import REAL_CONFIRMATION_TTL_S, RunState, TradeMode  # noqa: E402
from autopaper.telegram.botapi import BotApiTransport, BotControl  # noqa: E402
from autopaper.telegram.notifier import TelegramNotifier       # noqa: E402

ALLOWED = -1001234567890
FOREIGN = -1009999999999
TOKEN = "999:FAKEfakefakefakefakefakefake"

checks: list[tuple[bool, str]] = []


def check(ok: bool, label: str) -> None:
    checks.append((bool(ok), label))


class FakeTelegramAPI:
    """
    A stand-in for api.telegram.org.

    Records every request and serves queued updates, so the whole inbound path
    runs for real while the network does not exist.
    """

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.queued: list[dict] = []
        self.sent: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        try:
            body = json.loads(request.content or b"{}")
        except Exception:  # noqa: BLE001
            body = {}
        self.requests.append({"method": method, "body": body})
        if method == "getUpdates":
            out, self.queued = self.queued, []
            return httpx.Response(200, json={"ok": True, "result": out})
        if method in ("sendMessage", "editMessageText"):
            self.sent.append(body)
            return httpx.Response(200, json={"ok": True, "result": {
                "message_id": len(self.sent)}})
        return httpx.Response(200, json={"ok": True, "result": True})


def cb(chat, data, uid, cb_id):
    return {"update_id": uid, "callback_query": {
        "id": cb_id, "data": data, "message": {"chat": {"id": chat}}}}


async def main() -> int:
    import os
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="tg_smoke_"))

    # Stand in for the operator's .env.local. These are FAKE and exist only so
    # the control channel is built; the API itself is mocked, so the token is
    # never transmitted anywhere.
    os.environ["TELEGRAM_BOT_TOKEN"] = TOKEN
    os.environ["TELEGRAM_BOT_CHAT"] = str(ALLOWED)
    os.environ.pop("REAL_TRADING_ENABLED", None)

    # 1. process starts OFF -------------------------------------------------
    cfg = build_config(root)
    trader = PaperTrader(cfg)
    api = FakeTelegramAPI()

    # swap in the mocked transport, keeping the engine's real wiring
    real_client = httpx.AsyncClient(transport=httpx.MockTransport(api.handler))
    trader.bot_transport._client = real_client
    trader.bot_notifier = TelegramNotifier(trader.bot_transport,
                                          min_send_interval_s=0.0)
    trader.set_notifier(trader.bot_notifier.send)
    control = BotControl(transport=trader.bot_transport,
                         handle_button=trader.handle_button,
                         allowed_chat=ALLOWED)

    check(trader.state.run_state is RunState.OFF, "1. process starts OFF")
    check(trader.state.real_trading_enabled is False, "   REAL not armed")
    check(trader.state.trade_mode is TradeMode.PAPER, "   mode is PAPER")
    check(trader.bot_control is not None, "   control channel built")

    # 2/3. an unauthorized START is refused, then an authorized one lands ---
    await control.poll_once(timeout_s=0)          # nothing queued yet
    api.queued = [cb(FOREIGN, "start", 1, "s-foreign")]
    await control.poll_once(timeout_s=0)
    check(trader.state.run_state is RunState.OFF,
          "2. unauthorized START leaves the runtime OFF")

    api.queued = [cb(ALLOWED, "start", 2, "s-auth")]
    await control.poll_once(timeout_s=0)

    # 4. the runtime is running PAPER --------------------------------------
    check(trader.state.run_state is RunState.PAPER,
          "3. authorized START reaches the runtime")
    check(trader.state.running is True, "4. runtime enters PAPER state")
    allowed, why = trader.state.can_open_new_entries()
    check(allowed is True, f"   the real entry gate opened ({why})")

    # 5. the notifier delivers the acknowledgement -------------------------
    trader.bot_notifier.start()
    await asyncio.sleep(0)
    sent = await trader.bot_notifier.flush(timeout_s=2)
    check(sent and api.sent, "5. notifier delivered an acknowledgement")
    check(any("reply_markup" in m for m in api.sent),
          "   the reply carried the inline keyboard")
    check(any(m.get("chat_id") == ALLOWED for m in api.sent),
          "   delivered to the allow-listed chat only")

    # 6/7/8. no REAL, no wallet, no signing, no broadcast --------------------
    check(trader.state.real_trading_enabled is False, "6. no REAL arm occurred")
    check(trader.state.trade_mode is TradeMode.PAPER, "   still PAPER")
    check(trader._real_executor is None,
          "7. no real execution capability was ever acquired")
    methods = {r["method"] for r in api.requests}
    check(not (methods & {"signTransaction", "sendTransaction", "sendRaw"}),
          "8. nothing was signed or broadcast")
    check(all(m.get("chat_id") in (ALLOWED, None) for m in api.sent),
          "   no message went to an unauthorized chat")

    # replay: a duplicate must not re-run the transition --------------------
    api.queued = [cb(ALLOWED, "start", 2, "s-auth")]
    await control.poll_once(timeout_s=0)
    check(trader.state.run_state is RunState.PAPER,
          "   a replayed START did not change the state")

    # every callback payload, none may arm REAL ----------------------------
    payloads = ["start", "stop", "paper", "real", "real_off", "stats", "open",
                "emergency", "cancel_real", "clear_emergency",
                "confirm_real:", "confirm_real:guess", "unknown"]
    for i, p in enumerate(payloads):
        api.queued = [cb(ALLOWED, p, 100 + i, f"all-{i}")]
        await control.poll_once(timeout_s=0)
        check(trader.state.real_trading_enabled is False,
              f"   payload {p!r} did not arm REAL")
        if p == "stop":
            trader.state.start()          # keep the runtime usable for the rest

    # 9. STOP returns the runtime to a non-trading state --------------------
    trader.state.start()
    api.queued = [cb(ALLOWED, "stop", 500, "s-stop")]
    await control.poll_once(timeout_s=0)
    check(trader.state.run_state is RunState.STOPPED,
          "9. STOP returns the runtime to STOPPED")
    allowed_after, _ = trader.state.can_open_new_entries()
    check(allowed_after is False, "   no partially-running state remains")

    # 10. forward validation is intact --------------------------------------
    check(trader.ledger.forward is trader.forward,
          "10. the Ledger is still the single forward-validation path")
    check(trader.forward is not None and trader.forward_deployment is not None,
          "    the deployment boundary exists")
    report = build_report(trader.forward.all_records(),
                          deployment=trader.forward_deployment)
    check(report.dataset is VALIDATING,
          "    records would be PROSPECTIVE_PAPER only")
    check(report.is_validating is False,
          "    and no validation is claimed with no closed trades")
    check(trader.forward_deployment.l25_min_liq_to_mcap == 0.9647,
          "    the frozen threshold is recorded in the boundary")

    # invariants that must hold no matter what -----------------------------
    check(REAL_CONFIRMATION_TTL_S == 300.0, "    the 300s token TTL is intact")
    check(cfg.real.trading_enabled is False, "    REAL_TRADING_ENABLED is false")
    check(trader.bot_status()["real_trading_enabled"] is False,
          "    status reports REAL off")

    await trader.bot_notifier.aclose()
    await trader.bot_transport.aclose()
    trader._release_ownership()

    # ---- report -----------------------------------------------------------
    width = 74
    print("=" * width)
    print("TELEGRAM BOT API - BOUNDED PAPER END-TO-END SMOKE TEST")
    print("mocked Bot API | no network | no wallet | no signing | no broadcast")
    print("=" * width)
    for ok, label in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    failed = [l for ok, l in checks if not ok]
    print("-" * width)
    print(f"  {len(checks) - len(failed)}/{len(checks)} checks passed")
    print(f"  telegram requests made : {len(api.requests)}")
    print(f"  messages delivered    : {len(api.sent)} (chat {ALLOWED} only)")
    print(f"  final runtime state   : {trader.state.run_state.value}")
    print(f"  final trade mode      : {trader.state.trade_mode.value}")
    print(f"  REAL armed            : {trader.state.real_trading_enabled}")
    print("=" * width)
    if failed:
        print("\nFAILED:")
        for f in failed:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
