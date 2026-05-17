"""이미지 bytes → 모델 입력 텐서.

waste-preprocessor의 preprocess.py 와 동일한 변환:
  1. RGB 변환
  2. 224×224 리사이즈 (bilinear)
  3. [0,1] 스케일
  4. ImageNet 정규화

마지막에 arch 에 맞는 shape 으로 reshape.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from src import config


_MEAN = np.array(config.IMAGENET_MEAN, dtype=np.float32)
_STD = np.array(config.IMAGENET_STD, dtype=np.float32)


class ImageDecodeError(Exception):
    """이미지 디코딩 실패."""


def decode_image(raw: bytes) -> Image.Image:
    """bytes → Pillow Image (RGB)."""
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageDecodeError(f"이미지를 디코딩할 수 없음: {exc}") from exc

    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def to_normalized_array(img: Image.Image) -> np.ndarray:
    """Pillow Image → (H, W, C) float32 (ImageNet 정규화)."""
    resized = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return (arr - _MEAN) / _STD


def to_model_input(arr: np.ndarray, arch: str) -> np.ndarray:
    """(H, W, C) → 모델 입력 shape (batch=1 포함)."""
    if arch == "mlp":
        # (H, W, C) → (1, H*W*C)
        return arr.reshape(1, -1)
    if arch == "cnn":
        # (H, W, C) → (1, C, H, W)
        chw = np.ascontiguousarray(arr.transpose(2, 0, 1))
        return chw.reshape(1, config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
    raise ValueError(f"unsupported arch={arch!r}")


def preprocess(raw: bytes, arch: str) -> np.ndarray:
    """원본 bytes → 모델 입력 텐서 (한 줄 헬퍼)."""
    img = decode_image(raw)
    arr = to_normalized_array(img)
    return to_model_input(arr, arch)
