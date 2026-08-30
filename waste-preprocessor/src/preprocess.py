"""Phase 4: 이미지 전처리 — 구현은 waste_common.imaging (학습·서빙과 동일 체인).

순서: load → EXIF 방향 보정 → RGB 통일 → 리사이즈(IMAGE_SIZE²) → [0,1] → ImageNet 정규화.
출력: (H, W, C) float32 numpy 배열.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src import config
from waste_common import imaging
from waste_common.imaging import resize_square, to_normalized_array  # noqa: F401 — 하위호환 re-export


def load_rgb(path: Path) -> Image.Image:
    return imaging.decode_rgb(path)


def preprocess_image(path: Path) -> np.ndarray:
    """단일 이미지 → (IMAGE_SIZE, IMAGE_SIZE, 3) float32."""
    return imaging.preprocess(path, size=config.IMAGE_SIZE)


def image_stats(arr: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def run_sample() -> None:
    """데이터셋의 첫 이미지로 전처리를 시연 (`main.py --step preprocess`)."""
    sample = next(
        (p for p in config.DATASET_DIR.rglob("*") if p.suffix.lower() in config.SUPPORTED_EXTENSIONS),
        None,
    )
    if sample is None:
        print("no sample image found.")
        return
    arr = preprocess_image(sample)
    print(f"sample: {sample.name} → shape={arr.shape}, dtype={arr.dtype}")
    print(f"stats: {image_stats(arr)}")


if __name__ == "__main__":
    run_sample()
