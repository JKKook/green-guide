"""파이프라인 전역 설정 — 경로 상수는 여기, 공통값(이미지·Supabase)은 greenguide_common 에서 가져온다."""
from __future__ import annotations

from pathlib import Path

from greenguide_common import settings
from greenguide_common.imaging import (  # noqa: F401 — 하위호환 re-export
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SUPPORTED_EXTENSIONS,
)
from greenguide_common.taxonomy import LEGACY_LABELS

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
VECTORS_DIR: Path = PROCESSED_DIR / "vectors"
MANIFEST_PATH: Path = PROCESSED_DIR / "manifest.json"

DATASET_NAME: str = "garbage-classification"
DATASET_DIR: Path = RAW_DIR / DATASET_NAME
# Kaggle 시드 6클래스 — collect.dataset_present 의 존재 검증 전용.
# 현행 분류체계 정본은 greenguide_common.taxonomy 이며, catalog 는 폴더 auto-discover 로 실제 클래스를 결정한다.
CLASS_LABELS: tuple[str, ...] = LEGACY_LABELS

VECTOR_DIM: int = IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS  # 150,528

HASH_SIZE: int = 8

SUPABASE_BUCKET: str = settings.BUCKET_RAW_IMAGES
SUPABASE_TABLE: str = settings.SUPABASE_TABLE_ITEMS


def ensure_directories() -> None:
    """파이프라인 실행 전 필요한 디렉토리 생성."""
    for directory in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, VECTORS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
