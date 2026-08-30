"""이미지 전처리 — 21파일에 복제돼 있던 ImageNet 정규화 체인의 단일 구현.

표준 체인: decode(RGB, EXIF 보정) → resize_square(224, BILINEAR) → /255 → (x-mean)/std.
학습(classifier)·서빙(api)·데이터 파이프라인(preprocessor)이 모두 이 함수를 써야
train/serve 분포가 일치한다.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE: int = 224
IMAGE_CHANNELS: int = 3
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

MEAN_ARRAY: np.ndarray = np.array(IMAGENET_MEAN, dtype=np.float32)  # (3,) — HWC 브로드캐스트용
STD_ARRAY: np.ndarray = np.array(IMAGENET_STD, dtype=np.float32)
MEAN_CHW: np.ndarray = MEAN_ARRAY.reshape(3, 1, 1)                  # (3,1,1) — CHW 브로드캐스트용
STD_CHW: np.ndarray = STD_ARRAY.reshape(3, 1, 1)

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

_CONTENT_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def content_type(path: Path | str) -> str:
    return _CONTENT_TYPES.get(Path(path).suffix.lower(), "application/octet-stream")


def decode_rgb(src: Path | str | bytes | Image.Image, *, exif: bool = True) -> Image.Image:
    """경로/bytes/PIL 이미지 → RGB PIL 이미지. exif=True 면 촬영 방향 보정."""
    if isinstance(src, Image.Image):
        img = src
    elif isinstance(src, (bytes, bytearray)):
        img = Image.open(io.BytesIO(src))
    else:
        img = Image.open(src)
    if exif:
        img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def resize_square(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    return img.resize((size, size), Image.Resampling.BILINEAR)


def to_unit_array(img: Image.Image) -> np.ndarray:
    """PIL → (H, W, C) float32, [0, 1]. 정규화 전."""
    return np.asarray(img, dtype=np.float32) / 255.0


def normalize_hwc(arr: np.ndarray) -> np.ndarray:
    """(H, W, C) [0,1] → ImageNet 정규화."""
    return (arr - MEAN_ARRAY) / STD_ARRAY


def normalize_chw(arr: np.ndarray) -> np.ndarray:
    """(C, H, W) 또는 (N, C, H, W) [0,1] → ImageNet 정규화."""
    return (arr - MEAN_CHW) / STD_CHW


def to_normalized_array(img: Image.Image) -> np.ndarray:
    """PIL(RGB) → (H, W, C) float32 ImageNet 정규화 배열."""
    return normalize_hwc(to_unit_array(img))


def hwc_to_chw(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def preprocess(
    src: Path | str | bytes | Image.Image,
    *,
    size: int = IMAGE_SIZE,
    exif: bool = True,
    layout: str = "hwc",
) -> np.ndarray:
    """표준 체인 일괄 실행. layout: 'hwc' | 'chw' | 'nchw'."""
    arr = to_normalized_array(resize_square(decode_rgb(src, exif=exif), size))
    if layout == "hwc":
        return arr
    chw = hwc_to_chw(arr)
    if layout == "chw":
        return chw
    if layout == "nchw":
        return chw[None]
    raise ValueError(f"unknown layout={layout!r}")


def recompress_for_storage(
    src: Path | str | bytes | Image.Image,
    *,
    max_side: int,
    fmt: str = "JPEG",
    quality: int = 90,
) -> bytes:
    """긴 변을 max_side 로 축소(확대는 하지 않음)해 재인코딩한 bytes."""
    img = decode_rgb(src)
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return buf.getvalue()
