"""Phase 7: 전체 파이프라인 오케스트레이션.

수집 → 카탈로그 → 클렌징 → (각 이미지) 전처리 → 벡터화 → (선택) Supabase 업로드
→ manifest.json 출력.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from src import config
from src.catalog import build_catalog, save_catalog
from src.cleanse import cleanse
from src.collect import ensure_dataset
from src.preprocess import image_stats, preprocess_image
from src.vectorize import flatten, save_vector
from waste_common.logging import get_logger

if TYPE_CHECKING:
    from src.storage import SupabaseStore

log = get_logger(__name__)


def _relative_to_root(path: Path) -> str:
    if path.is_absolute() and config.PROJECT_ROOT in path.parents:
        return str(path.relative_to(config.PROJECT_ROOT))
    return str(path)


def _process_item(item: dict[str, Any], *, vectorize: bool, store: "SupabaseStore | None") -> dict[str, Any]:
    """catalog item 하나 → manifest record. vectorize=False 면 핵심 필드만."""
    abs_path = config.PROJECT_ROOT / item["source_path"]
    record: dict[str, Any] = {
        "id": item["id"],
        "label": item["label"],
        "source_path": item["source_path"],
        "filename": item["filename"],
    }
    if vectorize:
        arr = preprocess_image(abs_path)
        vector_path = save_vector(flatten(arr), item_id=item["id"])
        record["vector_path"] = _relative_to_root(vector_path)
        record["stats"] = image_stats(arr)
    if store is not None:
        record["original_url"] = store.upload_image(abs_path, item["label"], item["id"])
    return record


def run(upload_to_supabase: bool = False, vectorize: bool = True) -> Path:
    """전체 파이프라인 실행. manifest.json 경로 반환.

    Args:
        upload_to_supabase: 원본 이미지 Supabase 업로드 여부.
        vectorize: True 면 각 이미지를 .npz 로 벡터화(디스크 사용 多).
                   False 면 manifest 만 생성 (id/label/source_path) — CNN 학습은
                   raw 이미지를 직접 로드하므로 벡터 불필요. 디스크 대폭 절약.
    """
    config.ensure_directories()

    ensure_dataset()

    catalog = build_catalog()
    save_catalog(catalog)

    cleansed, _ = cleanse(catalog)
    save_catalog(cleansed, config.INTERIM_DIR / "catalog.cleansed.json")

    store = None
    if upload_to_supabase:
        from src.storage import SupabaseStore
        store = SupabaseStore()

    records: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []

    for item in tqdm(cleansed["items"], desc="[pipeline]"):
        try:
            records.append(_process_item(item, vectorize=vectorize, store=store))
        except Exception as exc:  # noqa: BLE001 — 한 장 실패가 전체를 막지 않도록 fail-open
            log.warning("item %s (%s) 처리 실패: %s", item["id"], item["source_path"], exc)
            failed.append({"id": item["id"], "path": item["source_path"], "error": str(exc)})

    if store is not None and records:
        upserted = store.upsert_many(records)
        log.info("upserted %d rows to Supabase", upserted)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "image_size": config.IMAGE_SIZE,
            "vector_dim": config.VECTOR_DIM,
            "normalize": "imagenet",
            "channels": config.IMAGE_CHANNELS,
        },
        "counts": {
            "input": catalog["count"],
            "cleansed": cleansed["count"],
            "processed": len(records),
            "failed": len(failed),
        },
        "items": records,
        "failed": failed,
    }

    config.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log.info(
        "done. input=%d cleansed=%d processed=%d failed=%d — manifest → %s",
        catalog["count"], cleansed["count"], len(records), len(failed), config.MANIFEST_PATH,
    )
    return config.MANIFEST_PATH
