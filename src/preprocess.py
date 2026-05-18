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


def to_edge_input(arr: np.ndarray) -> np.ndarray:
    """(H, W, C) RGB normalized → (1, 3, H, W) Sobel edge map (3채널 복제, ImageNet 재정규화).
    waste-classifier 의 WasteEdgeDataset 와 동일 변환.
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # 1) 역정규화 → [0,1] RGB
    rgb = (arr * std + mean).clip(0.0, 1.0)
    # 2) Grayscale
    gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114

    # 3) Sobel (numpy 구현)
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    # 패딩 후 convolve (간단 구현)
    padded = np.pad(gray, 1, mode="edge")
    h, w = gray.shape
    edge_x = np.zeros_like(gray)
    edge_y = np.zeros_like(gray)
    for i in range(3):
        for j in range(3):
            edge_x += sx[i, j] * padded[i:i + h, j:j + w]
            edge_y += sy[i, j] * padded[i:i + h, j:j + w]
    edge = np.sqrt(edge_x ** 2 + edge_y ** 2)

    # 4) [0,1] 정규화 + 3채널 복제
    if edge.max() > 1e-6:
        edge = edge / edge.max()
    edge_3c = np.stack([edge, edge, edge], axis=0)  # (3, H, W)

    # 5) ImageNet 정규화 → (1, 3, H, W)
    mean_chw = mean.reshape(3, 1, 1)
    std_chw = std.reshape(3, 1, 1)
    normalized = (edge_3c - mean_chw) / std_chw
    return normalized[np.newaxis, ...].astype(np.float32)


def preprocess_both(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """원본 bytes → (color_tensor, edge_tensor) — ensemble 용."""
    img = decode_image(raw)
    arr = to_normalized_array(img)
    color_input = to_model_input(arr, "cnn")
    edge_input = to_edge_input(arr)
    return color_input, edge_input
