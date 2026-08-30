"""preprocess 모듈 테스트."""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src.core import config
from src.preprocess import (
    ImageDecodeError,
    decode_image,
    preprocess,
    to_model_input,
    to_normalized_array,
)


def test_decode_image_returns_rgb(sample_image_bytes: bytes) -> None:
    img = decode_image(sample_image_bytes)
    assert img.mode == "RGB"


def test_decode_image_converts_grayscale() -> None:
    arr = np.zeros((100, 100), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    img = decode_image(buf.getvalue())
    assert img.mode == "RGB"


def test_decode_image_rejects_invalid_bytes() -> None:
    with pytest.raises(ImageDecodeError):
        decode_image(b"not an image")


def test_to_normalized_array_shape_and_dtype(sample_image_bytes: bytes) -> None:
    img = decode_image(sample_image_bytes)
    arr = to_normalized_array(img)
    assert arr.shape == (config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNELS)
    assert arr.dtype == np.float32


def test_to_normalized_array_value_range(sample_image_bytes: bytes) -> None:
    img = decode_image(sample_image_bytes)
    arr = to_normalized_array(img)
    # ImageNet 정규화 후 대략 -2.5 ~ +2.5 범위
    assert -3.0 < float(arr.min()) < 3.0
    assert -3.0 < float(arr.max()) < 3.0


def test_to_model_input_mlp_shape() -> None:
    arr = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNELS), dtype=np.float32)
    out = to_model_input(arr, "mlp")
    assert out.shape == (1, config.IMAGE_SIZE * config.IMAGE_SIZE * config.IMAGE_CHANNELS)


def test_to_model_input_cnn_shape() -> None:
    arr = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNELS), dtype=np.float32)
    out = to_model_input(arr, "cnn")
    assert out.shape == (1, config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)


def test_to_model_input_invalid_arch_raises() -> None:
    arr = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNELS), dtype=np.float32)
    with pytest.raises(ValueError):
        to_model_input(arr, "invalid")


def test_preprocess_end_to_end_mlp(sample_image_bytes: bytes) -> None:
    out = preprocess(sample_image_bytes, "mlp")
    assert out.shape == (1, config.IMAGE_SIZE * config.IMAGE_SIZE * config.IMAGE_CHANNELS)


def test_preprocess_end_to_end_cnn(sample_image_bytes: bytes) -> None:
    out = preprocess(sample_image_bytes, "cnn")
    assert out.shape == (1, config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
