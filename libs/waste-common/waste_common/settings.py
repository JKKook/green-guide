"""공통 설정 — 환경변수·형제 프로젝트 경로·Supabase 이름의 단일 출처.

.env 탐색은 여기서 한 번만 한다: (1) 현재 작업 디렉터리 기준 기본 탐색,
(2) waste-preprocessor/.env (역사적으로 세 프로젝트가 공유해 온 위치). 이미 설정된
환경변수는 덮어쓰지 않는다.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

WASTE_ROOT: Path = Path(__file__).resolve().parents[3]  # libs/waste-common/waste_common/ → 레포 루트


def _root(env_name: str, default_dirname: str) -> Path:
    return Path(os.getenv(env_name, WASTE_ROOT / default_dirname)).resolve()


PREPROCESSOR_ROOT: Path = _root("WASTE_PREPROCESSOR_ROOT", "ml/preprocessor")
CLASSIFIER_ROOT: Path = _root("WASTE_CLASSIFIER_ROOT", "ml/classifier")
API_ROOT: Path = _root("WASTE_API_ROOT", "services/inference-api")


def load_env() -> None:
    """.env 를 한 번 로드. 여러 번 호출해도 무해(override=False)."""
    load_dotenv()
    load_dotenv(PREPROCESSOR_ROOT / ".env")


load_env()

# ── Supabase ────────────────────────────────────────────────────────────────
SUPABASE_TABLE_ITEMS: str = os.getenv("SUPABASE_TABLE", "items")
BUCKET_RAW_IMAGES: str = os.getenv("SUPABASE_BUCKET", "raw-images")
BUCKET_USER_UPLOADS: str = "user-uploads"
BUCKET_MODELS: str = "models"


def supabase_credentials() -> tuple[str | None, str | None]:
    """호출 시점의 환경변수를 읽는다 (테스트에서 monkeypatch 가능)."""
    return os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
