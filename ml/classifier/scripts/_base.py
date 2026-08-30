"""scripts/ 공통 베이스 — sys.path 설정 + 경로 상수 + make_parser 재수출.

`python scripts/x.py` 로 실행하면 scripts/ 가 sys.path[0] 이므로 `from _base import ...` 로
쓴다. 프로젝트 루트를 sys.path 에 넣어 `from greenguide_classifier import ...` 가 동작하게 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

from greenguide_common import settings
from greenguide_common.cli import make_parser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PREPROCESSOR_ROOT = settings.PREPROCESSOR_ROOT
RAW_DIR = PREPROCESSOR_ROOT / "data" / "raw" / "garbage-classification"

__all__ = ["PREPROCESSOR_ROOT", "PROJECT_ROOT", "RAW_DIR", "make_parser"]
