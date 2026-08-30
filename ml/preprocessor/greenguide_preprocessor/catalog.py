"""Phase 2: 카탈로그.

raw 디렉토리를 스캔해 (id, label, path) 목록을 만든다. 메타데이터 1차 JSON.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TypedDict

from greenguide_common.logging import get_logger
from greenguide_preprocessor import config

log = get_logger(__name__)


class CatalogItem(TypedDict):
    id: str
    label: str
    source_path: str
    filename: str


class Catalog(TypedDict):
    dataset: str
    classes: list[str]
    count: int
    items: list[CatalogItem]


def _iter_class_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_EXTENSIONS
    )


def _stringify_path(path: Path) -> str:
    """가능하면 PROJECT_ROOT 기준 상대 경로, 아니면 절대 경로."""
    try:
        return str(path.relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_catalog(dataset_dir: Path | None = None) -> Catalog:
    dataset_dir = dataset_dir if dataset_dir is not None else config.DATASET_DIR
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")

    # 하드코딩된 CLASS_LABELS 우선 (Kaggle 기본 6개 보장 순서),
    # 그 외 raw 폴더에 존재하는 모든 서브디렉토리를 추가 클래스로 인식.
    base_labels = list(config.CLASS_LABELS)
    extra_labels = sorted(
        d.name for d in dataset_dir.iterdir()
        if d.is_dir() and d.name not in base_labels
    )
    all_labels = base_labels + extra_labels

    items: list[CatalogItem] = []
    seen_labels: list[str] = []
    for label in all_labels:
        class_dir = dataset_dir / label
        if not class_dir.is_dir():
            continue
        images = _iter_class_images(class_dir)
        if not images:
            continue
        seen_labels.append(label)
        for image_path in images:
            items.append(
                CatalogItem(
                    id=uuid.uuid4().hex[:12],
                    label=label,
                    source_path=_stringify_path(image_path),
                    filename=image_path.name,
                )
            )

    return Catalog(
        dataset=config.DATASET_NAME,
        classes=seen_labels,
        count=len(items),
        items=items,
    )


def save_catalog(catalog: Catalog, path: Path | None = None) -> Path:
    output = path or (config.INTERIM_DIR / "catalog.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    log.info("%d items → %s", catalog["count"], output)
    return output


def load_catalog(path: Path | None = None) -> Catalog:
    src = path or (config.INTERIM_DIR / "catalog.json")
    with src.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    catalog = build_catalog()
    save_catalog(catalog)
