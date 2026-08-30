"""Phase 5: 벡터화.

전처리된 (H, W, C) → (H*W*C,) 1D 로 flatten 후 float16 으로 다운캐스트하여
gzip 압축된 .npz 로 저장 (~75% 디스크 절감, ImageNet 정규화 범위에서 정확도 손실 거의 없음).
load 시 자동으로 float32 로 복원.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from greenguide_preprocessor import config

STORAGE_DTYPE = np.float16
COMPUTE_DTYPE = np.float32


def flatten(arr: np.ndarray) -> np.ndarray:
    """전처리 결과를 1차원 벡터로."""
    if arr.dtype != COMPUTE_DTYPE:
        arr = arr.astype(COMPUTE_DTYPE, copy=False)
    vector = arr.reshape(-1)
    if vector.shape[0] != config.VECTOR_DIM:
        raise ValueError(
            f"unexpected vector dim: got {vector.shape[0]}, expected {config.VECTOR_DIM}"
        )
    return vector


def save_vector(vector: np.ndarray, item_id: str, base_dir: Path | None = None) -> Path:
    """vector 를 float16 으로 다운캐스트하여 압축 저장. .npz 경로 반환."""
    base_dir = base_dir if base_dir is not None else config.VECTORS_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    out = base_dir / f"{item_id}.npz"
    np.savez_compressed(out, vector=vector.astype(STORAGE_DTYPE, copy=False))
    return out


def load_vector(item_id: str, base_dir: Path | None = None) -> np.ndarray:
    """저장된 float16 벡터를 float32 로 복원해 반환."""
    base_dir = base_dir if base_dir is not None else config.VECTORS_DIR
    with np.load(base_dir / f"{item_id}.npz", allow_pickle=False) as npz:
        return npz["vector"].astype(COMPUTE_DTYPE, copy=False)


def run_sample() -> None:
    """데이터셋의 첫 이미지를 벡터화해 저장 (`main.py --step vectorize`)."""
    from greenguide_preprocessor.preprocess import preprocess_image

    sample = next(
        (p for p in config.DATASET_DIR.rglob("*")
         if p.suffix.lower() in config.SUPPORTED_EXTENSIONS),
        None,
    )
    if sample is None:
        print("no sample image found.")
        return
    vec = flatten(preprocess_image(sample))
    path = save_vector(vec, item_id="demo")
    print(f"vector shape={vec.shape}, saved to {path}")


if __name__ == "__main__":
    run_sample()
