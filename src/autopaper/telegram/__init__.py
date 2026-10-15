"""
Telegram package: the candidate listener, the alert parser, and the operator UI.

Three distinct jobs, kept apart on purpose:

* :mod:`parser` - pure text -> structured alert. No I/O, no clock, no network.
* :mod:`client` - MTProto read side. Produces candidates. Never accepts commands.
* :mod:`ui` / :mod:`actions` - the operator surface. The only place a button
  press can change the runtime state.
"""

from .actions import ButtonHandler, ButtonResult, OperatorActions
from .client import (
    FORBIDDEN_SOURCE_NAMES,
    ConnectionState,
    DedupeWindow,
    ListenerStatus,
    TelegramListener,
    TelegramSourceNotAllowed,
    interactive_login,
)
from .parser import ParsedAlert, looks_like_alert, normalise, parse_alert
from .ui import (
    StatusView,
    consistency_check,
    main_keyboard,
    operator_keyboard,
    render_status,
    render_strategy_menu,
    strategy_keyboard,
)

__all__ = [
    # listener
    "ConnectionState",
    "DedupeWindow",
    "ListenerStatus",
    "TelegramListener",
    "TelegramSourceNotAllowed",
    "FORBIDDEN_SOURCE_NAMES",
    "interactive_login",
    # parser
    "ParsedAlert",
    "looks_like_alert",
    "normalise",
    "parse_alert",
    # operator UI
    "StatusView",
    "ButtonHandler",
    "ButtonResult",
    "OperatorActions",
    "consistency_check",
    "main_keyboard",
    "operator_keyboard",
    "render_strategy_menu",
    "strategy_keyboard",
    "render_status",
]
