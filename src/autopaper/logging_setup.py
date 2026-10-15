"""
Tagged logging: every line is prefixed with a role so the ledger of a live run
is readable at a glance.

[ALERT] [ANALYSIS] [BUY] [WAIT] [REJECT] [PAPER_FILL] [HOLD] [EXIT]
[ACCOUNT] [ERROR] [MARKET] [TELEGRAM] [NAR] [SYSTEM] [SAFETY]
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

TAGS = (
    "ALERT",
    "ANALYSIS",
    "BUY",
    "WAIT",
    "REJECT",
    "PAPER_FILL",
    "HOLD",
    "EXIT",
    "ACCOUNT",
    "ERROR",
    "MARKET",
    "TELEGRAM",
    "NAR",
    "SYSTEM",
    "SAFETY",
    "RECOVERY",
    "CONTROL",
)

_LOGGERS: dict[str, logging.Logger] = {}
_HANDLER: logging.Handler | None = None
_CONFIGURED = False

LEVEL_BY_TAG = {
    "ERROR": logging.ERROR,
    "SAFETY": logging.WARNING,
    "RECOVERY": logging.WARNING,
    "CONTROL": logging.WARNING,
}


class _TagFormatter(logging.Formatter):
    def __init__(self, color: bool) -> None:
        super().__init__("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, self.datefmt)
        tag = getattr(record, "tag", "SYSTEM")
        if self.color:
            level = record.levelno
            if level >= logging.ERROR:
                colour = "\033[31m"
            elif tag in ("BUY", "EXIT", "PAPER_FILL"):
                colour = "\033[36m"
            elif tag == "REJECT":
                colour = "\033[33m"
            elif tag in ("HOLD", "WAIT"):
                colour = "\033[35m"
            elif tag in ("SAFETY", "CONTROL", "RECOVERY"):
                colour = "\033[31;1m"
            elif tag == "ACCOUNT":
                colour = "\033[32m"
            else:
                colour = "\033[37m"
            return f"{ts} \033[90m{record.levelname:<5}\033[0m {colour}[{tag:<11}]\033[0m {record.getMessage()}"
        return f"{ts} {record.levelname:<5} [{tag:<11}] {record.getMessage()}"


def configure(log_dir: Path | None = None, level: str | None = None) -> None:
    global _HANDLER, _CONFIGURED
    root = logging.getLogger("autopaper")
    if _CONFIGURED:
        root.setLevel(_resolve_level(level))
        return
    root.setLevel(_resolve_level(level))
    root.propagate = False
    color = sys.stderr.isatty() and os.environ.get("NO_COLOR") is None
    formatter = _TagFormatter(color)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    _HANDLER = stream

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "autopaper.log", encoding="utf-8")
        fh.setFormatter(_TagFormatter(color=False))
        root.addHandler(fh)

    _CONFIGURED = True


def _resolve_level(level: str | None) -> int:
    name = (level or os.environ.get("AUTOPAPER_LOG_LEVEL") or "INFO").upper()
    return getattr(logging, name, logging.INFO)


def get(tag: str) -> logging.Logger:
    if tag not in TAGS:
        tag = "SYSTEM"
    if tag not in _LOGGERS:
        _LOGGERS[tag] = logging.getLogger(f"autopaper.{tag.lower()}")
    return _LOGGERS[tag]


def log(tag: str, message: str, *args: Any, level: int | None = None) -> None:
    """
    Tagged log. Use %-style args so the formatter is lazy.

    The tag travels on the record (not on the logger) so two loggers at
    different levels cannot disagree about what a line is.
    """
    logger = get(tag)
    lvl = level if level is not None else LEVEL_BY_TAG.get(tag, logging.INFO)
    logger.log(lvl, message, *args, extra={"tag": tag})
