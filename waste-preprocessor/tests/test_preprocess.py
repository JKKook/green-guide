"""전처리 모듈 테스트."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src import config
from src.preprocess import (
    image_stats,
    load_rgb,
    preprocess_image,
    resize_square,
    to_normalized_array,
)


def test_load_rgb_converts_non_rgb(tmp_path: Path) -> None:
    gray = Image.fromarray(np.zeros((50, 50), dtype=np.uint8), mode="L")
    path = tmp_path / "gray.png"
    gray.save(path)

    with load_rgb(path) as img:
        assert img.mode == "RGB"


def test_resize_square_outputs_target_size(sample_image: Path) -> None:
    with load_rgb(sample_image) as img:
        resized = resize_square(img, size=config.IMAGE_SIZE)
    assert resized.size == (config.IMAGE_SIZE, config.IMAGE_SIZE)


def test_to_normalized_array_dtype_and_range(sample_image: Path) -> None:
    with load_rgb(sample_image) as img:
        arr = to_normalized_array(resize_square(img))
    assert arr.dtype == np.float32
    # ImageNet 정규화 후 대략 -2.5 ~ +2.5 범위
    assert -3.0 < float(arr.min()) < 3.0
    assert -3.0 < float(arr.max()) < 3.0


def test_preprocess_image_shape(sample_image: Path) -> None:
    arr = preprocess_image(sample_image)
    assert arr.shape == (config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNELS)
    assert arr.dtype == np.float32


def test_preprocess_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        preprocess_image(tmp_path / "no-such-file.jpg")


def test_image_stats_keys(sample_image: Path) -> None:
    arr = preprocess_image(sample_image)
    stats = image_stats(arr)
    assert set(stats.keys()) == {"mean", "std", "min", "max"}
