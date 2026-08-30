"""pytest 공통 fixture."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from greenguide_preprocessor import config


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    """800x600 RGB JPEG 임시 파일 생성."""
    arr = (np.random.default_rng(0).random((600, 800, 3)) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    path = tmp_path / "sample.jpg"
    img.save(path, format="JPEG", quality=90)
    return path


@pytest.fixture()
def fake_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """6개 클래스 폴더에 각각 이미지 2장씩 들어있는 가짜 데이터셋."""
    dataset_dir = tmp_path / "garbage-classification"
    rng = np.random.default_rng(42)

    for label in config.CLASS_LABELS:
        class_dir = dataset_dir / label
        class_dir.mkdir(parents=True)
        for i in range(2):
            arr = (rng.random((100, 100, 3)) * 255).astype(np.uint8)
            Image.fromarray(arr).save(class_dir / f"{label}_{i}.jpg")

    monkeypatch.setattr(config, "DATASET_DIR", dataset_dir)
    return dataset_dir
