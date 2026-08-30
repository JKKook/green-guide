"""Phase 2: train/val/test 분할.

sklearn 의 stratified split 으로 라벨 비율 보존. 인덱스 JSON 저장으로 재현성 확보.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split
from waste_common.logging import get_logger

from src import config

log = get_logger(__name__)


def stratified_split(
    items: list[dict[str, Any]],
    ratios: dict[str, float] = config.SPLIT_RATIOS,
    seed: int = config.SPLIT_SEED,
) -> dict[str, list[int]]:
    """전체 items 을 인덱스 기준으로 train/val/test 분할."""
    n = len(items)
    indices = list(range(n))
    labels = [it["label"] for it in items]

    # 1) train vs (val+test)
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=ratios["val"] + ratios["test"],
        stratify=labels,
        random_state=seed,
    )

    # 2) val vs test (holdout 내에서 다시 stratified)
    holdout_labels = [labels[i] for i in holdout_idx]
    val_ratio_within = ratios["val"] / (ratios["val"] + ratios["test"])
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=1 - val_ratio_within,
        stratify=holdout_labels,
        random_state=seed,
    )

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def save_splits(splits: dict[str, list[int]], path: Path | None = None) -> Path:
    out = path or (config.SPLITS_DIR / "splits.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(
            {k: list(map(int, v)) for k, v in splits.items()},
            f, ensure_ascii=False, indent=2,
        )
    log.info(f"saved → {out}  "
          f"(train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])})")
    return out


def load_splits(path: Path | None = None) -> dict[str, list[int]]:
    src = path or (config.SPLITS_DIR / "splits.json")
    with src.open("r", encoding="utf-8") as f:
        return json.load(f)


def subset_items(items: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    return [items[i] for i in indices]
