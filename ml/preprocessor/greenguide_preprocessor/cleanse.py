"""Phase 3: 클렌징.

손상된 이미지를 거르고, perceptual hash 로 중복을 제거한다.
"""
from __future__ import annotations

from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from greenguide_preprocessor import config
from greenguide_preprocessor.catalog import Catalog, CatalogItem
from greenguide_common.logging import get_logger

log = get_logger(__name__)


def is_corrupt(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        return True
    return False


def perceptual_hash(path: Path) -> str | None:
    try:
        with Image.open(path) as img:
            return str(imagehash.phash(img, hash_size=config.HASH_SIZE))
    except (UnidentifiedImageError, OSError, SyntaxError):
        return None


def cleanse(catalog: Catalog) -> tuple[Catalog, dict[str, int]]:
    """손상·중복 제거된 새 catalog 와 통계 반환."""
    kept: list[CatalogItem] = []
    seen_hashes: dict[str, str] = {}
    stats = {"input": catalog["count"], "corrupt": 0, "duplicate": 0, "kept": 0}

    for item in tqdm(catalog["items"], desc="[cleanse]"):
        abs_path = config.PROJECT_ROOT / item["source_path"]

        if is_corrupt(abs_path):
            stats["corrupt"] += 1
            continue

        phash = perceptual_hash(abs_path)
        if phash is None:
            stats["corrupt"] += 1
            continue
        if phash in seen_hashes:
            stats["duplicate"] += 1
            continue
        seen_hashes[phash] = item["id"]

        kept.append(item)

    stats["kept"] = len(kept)
    cleaned = Catalog(
        dataset=catalog["dataset"],
        classes=catalog["classes"],
        count=len(kept),
        items=kept,
    )
    log.info("in=%d corrupt=%d duplicate=%d kept=%d",
             stats["input"], stats["corrupt"], stats["duplicate"], stats["kept"])
    return cleaned, stats


if __name__ == "__main__":
    from greenguide_preprocessor.catalog import load_catalog, save_catalog
    catalog = load_catalog()
    cleaned, _ = cleanse(catalog)
    save_catalog(cleaned, config.INTERIM_DIR / "catalog.cleansed.json")
