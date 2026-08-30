"""CAM (Class Activation Map) → 사용자가 볼 수 있는 heatmap overlay PNG.

흐름:
  1. CAM (H, W) np.ndarray (보통 7×7) ← inference.predict(want_cam=True)
  2. ReLU + min-max 정규화 → [0, 1]
  3. 원본 입력 사이즈로 bilinear resize
  4. jet colormap → RGB heatmap (PIL Image)
  5. 원본 이미지와 alpha-blend → overlay
  6. PNG 인코딩 → base64 문자열 (data URI 형식)
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


# matplotlib jet colormap 을 numpy 로 미리 구현해두면 matplotlib 의존성을 피할 수 있음.
# 256-entry RGB lookup table.
def _jet_lut() -> np.ndarray:
    """matplotlib 의 'jet' colormap 을 256 entry RGB (uint8) 로 prebuild."""
    n = 256
    x = np.linspace(0, 1, n)

    def _piecewise(x: np.ndarray, *segments: tuple[float, float, float, float]) -> np.ndarray:
        """[(x_lo, x_hi, y_lo, y_hi), ...] 구간별 선형 보간."""
        y = np.zeros_like(x)
        for x_lo, x_hi, y_lo, y_hi in segments:
            mask = (x >= x_lo) & (x <= x_hi)
            t = (x[mask] - x_lo) / max(x_hi - x_lo, 1e-9)
            y[mask] = y_lo + t * (y_hi - y_lo)
        return y

    # 표준 jet: r/g/b 별 piecewise (matplotlib 원형 근사)
    r = _piecewise(x,
                   (0.0, 0.35, 0.0, 0.0),
                   (0.35, 0.66, 0.0, 1.0),
                   (0.66, 0.89, 1.0, 1.0),
                   (0.89, 1.0, 1.0, 0.5))
    g = _piecewise(x,
                   (0.0, 0.125, 0.0, 0.0),
                   (0.125, 0.375, 0.0, 1.0),
                   (0.375, 0.64, 1.0, 1.0),
                   (0.64, 0.91, 1.0, 0.0),
                   (0.91, 1.0, 0.0, 0.0))
    b = _piecewise(x,
                   (0.0, 0.11, 0.5, 1.0),
                   (0.11, 0.34, 1.0, 1.0),
                   (0.34, 0.65, 1.0, 0.0),
                   (0.65, 1.0, 0.0, 0.0))
    lut = (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)
    return lut


_JET_LUT: np.ndarray | None = None


def _apply_jet(values: np.ndarray) -> np.ndarray:
    """values in [0, 1] (any shape) → RGB uint8 (shape + (3,))."""
    global _JET_LUT
    if _JET_LUT is None:
        _JET_LUT = _jet_lut()
    idx = np.clip((values * 255).astype(int), 0, 255)
    return _JET_LUT[idx]


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    """ReLU + min-max → [0, 1]. all-zero 이면 그대로 0."""
    cam = np.maximum(cam, 0.0)
    cam_min = cam.min()
    cam_max = cam.max()
    rng = cam_max - cam_min
    if rng < 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    return ((cam - cam_min) / rng).astype(np.float32)


def render_overlay_png_base64(
    image_bytes: bytes,
    cam: np.ndarray,
    alpha: float = 0.45,
    max_long_side: int = 720,
) -> str:
    """원본 이미지 위에 CAM heatmap 을 overlay → PNG → base64 (data URI 형식).

    Args:
        image_bytes: 원본 입력 이미지 bytes (UploadFile.read() 결과)
        cam: (H, W) np.ndarray — inference 가 반환한 raw CAM (정규화 전)
        alpha: heatmap 의 투명도 (0.45 = heatmap 45% + 원본 55%)
        max_long_side: 응답 크기 줄이기 위한 다운스케일 한도 (긴 변 기준 px)

    Returns:
        "data:image/png;base64,..." 문자열 — Flutter <Image.network> 또는
        base64 디코드 후 Image.memory 로 바로 표시 가능.
    """
    # 1) 원본 디코드
    orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 2) 너무 크면 다운스케일 (네트워크 응답 크기 절약)
    if max(orig.size) > max_long_side:
        scale = max_long_side / max(orig.size)
        new_size = (int(orig.size[0] * scale), int(orig.size[1] * scale))
        orig = orig.resize(new_size, Image.BILINEAR)

    w, h = orig.size

    # 3) CAM 정규화 + 원본 크기로 resize
    cam_norm = normalize_cam(cam)
    cam_pil = Image.fromarray((cam_norm * 255).astype(np.uint8))
    cam_resized = np.array(cam_pil.resize((w, h), Image.BILINEAR)) / 255.0

    # 4) jet colormap 적용
    heatmap = _apply_jet(cam_resized)  # (h, w, 3) uint8

    # 5) alpha blend
    orig_arr = np.array(orig).astype(np.float32)
    blended = (alpha * heatmap.astype(np.float32) + (1 - alpha) * orig_arr)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    overlay_pil = Image.fromarray(blended)

    # 6) PNG 인코딩 → base64
    buf = io.BytesIO()
    overlay_pil.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"
