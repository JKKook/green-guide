"""로깅 — `print("[prefix] ...")` 655곳을 대체하는 최소 API.

    from greenguide_common.logging import get_logger, fail_open
    log = get_logger(__name__)
    with fail_open(log, "Supabase 기록"):
        ...  # 예외는 traceback 과 함께 warning 으로 기록되고 삼켜진다
"""
from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

_FORMAT = "%(levelname)s %(name)s: %(message)s"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)


@contextmanager
def fail_open(log: logging.Logger, what: str) -> Iterator[None]:
    """best-effort 블록: 실패해도 흐름을 막지 않되, traceback 은 남긴다."""
    try:
        yield
    except Exception:  # noqa: BLE001 — fail-open 이 이 컨텍스트의 목적
        log.warning("%s 실패 (무시하고 계속)", what, exc_info=True)
