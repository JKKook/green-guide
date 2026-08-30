"""공통 fixture."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src import config


@pytest.fixture()
def fake_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict], Path]:
    """클래스당 8장 가짜 manifest — .npz(MLP용) + 실제 소형 JPEG(CNN raw 로딩용).

    주의: CLASS_LABELS 는 실제 manifest 로 동적 갱신되므로(현재 13개)
    총 개수를 하드코딩하지 말 것 (len(CLASS_LABELS) * 8).
    """
    from PIL import Image

    vectors_dir = tmp_path / "vectors"
    vectors_dir.mkdir()
    imgs_dir = tmp_path / "imgs"
    imgs_dir.mkdir()

    rng = np.random.default_rng(0)
    items = []
    for label in config.CLASS_LABELS:
        for k in range(8):  # stratified split (70/15/15) 가 가능한 최소 수
            item_id = f"{label}_{k}"
            vec = rng.standard_normal(config.INPUT_DIM, dtype=np.float32)
            np.savez_compressed(vectors_dir / f"{item_id}.npz",
                                vector=vec.astype(np.float16))
            # CNN raw 로딩용 실제 이미지 (작게 — 로더가 224 로 리사이즈)
            arr = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
            Image.fromarray(arr, "RGB").save(imgs_dir / f"{item_id}.jpg")
            items.append({
                "id": item_id,
                "label": label,
                "vector_path": str(vectors_dir / f"{item_id}.npz"),
                "source_path": f"imgs/{item_id}.jpg",
                "filename": f"{item_id}.jpg",
            })

    manifest_path = tmp_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({"items": items}, f)

    monkeypatch.setattr(config, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(config, "VECTORS_DIR", vectors_dir)
    # source_path 는 PREPROCESSOR_ROOT 기준 상대경로 (raw 직접 로딩)
    monkeypatch.setattr(config, "PREPROCESSOR_ROOT", tmp_path)
    return items, tmp_path
