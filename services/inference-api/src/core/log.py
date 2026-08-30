"""공통 로거. 모듈마다 `log = get_logger(__name__)` 한 줄로 사용.

포맷의 로거 이름(src.xxx)이 이전 print 접두어 `[xxx]` 역할을 대신한다.
"""
from __future__ import annotations

import logging

_FORMAT = "[%(levelname)s] %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=_FORMAT)
    return logging.getLogger(name)
