"""객체 누끼(saliency segmentation) — u2netp ONNX 직접 사용.

rembg 라이브러리(numpy 2.x·scikit-image·numba 등 무거운 의존성) 없이
u2netp.onnx(4.4MB) 만 기존 onnxruntime 으로 구동. rembg 의 u2net 전/후처리 재현.

용도: 캡처 이미지에서 주요 객체의 mask + bbox 추출 → 앱이 배경 dim + 라벨 오버레이.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from src import config


_U2NET_MEAN = (0.485, 0.456, 0.406)
_U2NET_STD = (0.229, 0.224, 0.225)
_SIZE = 320
_MASK_THRESHOLD = 64   # 0-255, bbox 추출 시 객체로 간주할 alpha 하한


def _model_path() -> Path:
    bundled = config.PROJECT_ROOT / "models" / "u2netp.onnx"
    return bundled


class Segmenter:
    """u2netp saliency 모델 wrapper."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or _model_path()
        self.available = self.model_path.exists()
        self.session: ort.InferenceSession | None = None
        if self.available:
            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        im = img.convert("RGB").resize((_SIZE, _SIZE), Image.LANCZOS)
        ary = np.array(im).astype(np.float64)
        mx = ary.max()
        if mx > 0:
            ary = ary / mx
        tmp = np.zeros((_SIZE, _SIZE, 3), dtype=np.float64)
        for c in range(3):
            tmp[:, :, c] = (ary[:, :, c] - _U2NET_MEAN[c]) / _U2NET_STD[c]
        chw = tmp.transpose((2, 0, 1))[np.newaxis, ...].astype(np.float32)
        return chw

    def object_mask_grid(self, image_bytes: bytes, grid: int) -> np.ndarray:
        """u2netp saliency → grid×grid 객체 점유 비율 (0~1). 없으면 전부 1."""
        if not self.available or self.session is None:
            return np.ones((grid, grid), dtype=np.float32)
        orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inp = self._preprocess(orig)
        out = self.session.run(None, {self.input_name: inp})[0][0, 0]
        mi, ma = float(out.min()), float(out.max())
        out = (out - mi) / (ma - mi + 1e-8)
        m = Image.fromarray((out * 255).astype(np.uint8)).resize(
            (grid, grid), Image.BILINEAR,
        )
        return np.array(m).astype(np.float32) / 255.0

    def segment(self, image_bytes: bytes) -> dict:
        """이미지 → {cutout_base64, bbox_norm, object_ratio}.

        - cutout_base64: 객체만 남기고 배경을 투명하게 한 RGBA PNG (data URI),
          긴 변 최대 512. 앱이 [dim 원본] 위에 이 cutout 을 겹쳐 객체를 부각.
        - bbox_norm: [x0, y0, x1, y1] 0~1 정규화 (라벨 위치용, 없으면 None).
        - object_ratio: 객체가 차지하는 면적 비율.
        """
        if not self.available or self.session is None:
            return {"cutout_base64": None, "bbox_norm": None, "object_ratio": 0.0}

        orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        ow, oh = orig.size

        inp = self._preprocess(orig)
        out = self.session.run(None, {self.input_name: inp})[0]  # (1,1,320,320)
        pred = out[0, 0, :, :]
        mi, ma = float(pred.min()), float(pred.max())
        if ma - mi > 1e-8:
            pred = (pred - mi) / (ma - mi)
        else:
            pred = np.zeros_like(pred)

        # 응답 크기 위해 긴 변 512 제한
        long_side = max(ow, oh)
        scale = min(1.0, 512 / long_side)
        out_w, out_h = int(ow * scale), int(oh * scale)

        alpha = Image.fromarray((pred * 255).astype(np.uint8), mode="L").resize(
            (out_w, out_h), Image.LANCZOS,
        )
        rgb = orig.resize((out_w, out_h), Image.LANCZOS)

        # cutout — RGBA (배경 alpha=saliency)
        cutout = rgb.convert("RGBA")
        cutout.putalpha(alpha)

        # bbox (정규화)
        m = np.array(alpha)
        ys, xs = np.where(m > _MASK_THRESHOLD)
        bbox_norm = None
        object_ratio = 0.0
        if len(xs) > 0:
            bbox_norm = [
                float(xs.min() / out_w), float(ys.min() / out_h),
                float(xs.max() / out_w), float(ys.max() / out_h),
            ]
            object_ratio = float((m > _MASK_THRESHOLD).mean())

        buf = io.BytesIO()
        cutout.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return {
            "cutout_base64": f"data:image/png;base64,{b64}",
            "bbox_norm": bbox_norm,
            "object_ratio": round(object_ratio, 4),
        }


_segmenter: Segmenter | None = None


def get_segmenter() -> Segmenter:
    global _segmenter
    if _segmenter is None:
        _segmenter = Segmenter()
    return _segmenter


def component_bbox_at(
    image_bytes: bytes, tap_x: float, tap_y: float,
    threshold: float = 0.4, search_radius_frac: float = 0.08,
) -> list[float] | None:
    """탭 지점(정규화 0~1)이 속한 saliency 연결 성분의 bbox_norm 반환.

    탭-투-셀렉트용: 혼재 장면에서 사용자가 지목한 객체만 분리.
    1) u2netp 마스크(320²) → threshold 이진화
    2) 탭 지점이 배경이면 주변 반경에서 가장 가까운 객체 픽셀 탐색
    3) BFS flood-fill 로 해당 연결 성분 추출 → bbox 정규화
    실패(마스크 없음/성분 없음) 시 None — 호출부가 window-crop fallback.
    """
    seg = get_segmenter()
    if not seg.available or seg.session is None:
        return None
    orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inp = seg._preprocess(orig)
    out = seg.session.run(None, {seg.input_name: inp})[0][0, 0]
    mi, ma = float(out.min()), float(out.max())
    if ma - mi < 1e-8:
        return None
    binary = ((out - mi) / (ma - mi)) > threshold

    tx = min(max(int(tap_x * _SIZE), 0), _SIZE - 1)
    ty = min(max(int(tap_y * _SIZE), 0), _SIZE - 1)

    # 탭 지점이 배경이면 반경 내 최근접 객체 픽셀로 스냅
    if not binary[ty, tx]:
        r = max(1, int(_SIZE * search_radius_frac))
        ys, xs = np.where(
            binary[max(0, ty - r):ty + r + 1, max(0, tx - r):tx + r + 1])
        if len(xs) == 0:
            return None
        d2 = (ys - min(ty, r)) ** 2 + (xs - min(tx, r)) ** 2
        k = int(d2.argmin())
        ty = max(0, ty - r) + int(ys[k])
        tx = max(0, tx - r) + int(xs[k])

    # BFS flood fill (scipy 없이 — 320² 는 충분히 가벼움)
    from collections import deque
    visited = np.zeros_like(binary, dtype=bool)
    q = deque([(ty, tx)])
    visited[ty, tx] = True
    x0, y0, x1, y1 = tx, ty, tx, ty
    while q:
        cy, cx = q.popleft()
        x0, x1 = min(x0, cx), max(x1, cx)
        y0, y1 = min(y0, cy), max(y1, cy)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if (0 <= ny < _SIZE and 0 <= nx < _SIZE
                    and binary[ny, nx] and not visited[ny, nx]):
                visited[ny, nx] = True
                q.append((ny, nx))

    # 너무 작은 성분(노이즈)은 무시
    if (x1 - x0) < _SIZE * 0.03 or (y1 - y0) < _SIZE * 0.03:
        return None
    return [x0 / _SIZE, y0 / _SIZE, (x1 + 1) / _SIZE, (y1 + 1) / _SIZE]


def all_component_bboxes(
    image_bytes: bytes, threshold: float = 0.4,
    min_side_frac: float = 0.06, max_n: int = 5,
) -> list[list[float]]:
    """u2netp saliency 의 모든 연결 성분 bbox_norm 목록 (면적 내림차순, 최대 max_n).

    탐지-후-분류용: 혼재 장면의 각 객체 후보를 분리한다.
    saliency 는 인스턴스 세그가 아니므로 붙어있는 객체는 병합될 수 있음 —
    그 한계는 탭-투-셀렉트가 보완.
    """
    seg = get_segmenter()
    if not seg.available or seg.session is None:
        return []
    orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inp = seg._preprocess(orig)
    out = seg.session.run(None, {seg.input_name: inp})[0][0, 0]
    mi, ma = float(out.min()), float(out.max())
    if ma - mi < 1e-8:
        return []
    binary = ((out - mi) / (ma - mi)) > threshold

    from collections import deque
    visited = np.zeros_like(binary, dtype=bool)
    comps: list[tuple[int, list[float]]] = []
    min_side = _SIZE * min_side_frac

    for sy in range(_SIZE):
        for sx in range(_SIZE):
            if not binary[sy, sx] or visited[sy, sx]:
                continue
            # BFS
            q = deque([(sy, sx)])
            visited[sy, sx] = True
            x0, y0, x1, y1 = sx, sy, sx, sy
            area = 0
            while q:
                cy, cx = q.popleft()
                area += 1
                x0, x1 = min(x0, cx), max(x1, cx)
                y0, y1 = min(y0, cy), max(y1, cy)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < _SIZE and 0 <= nx < _SIZE
                            and binary[ny, nx] and not visited[ny, nx]):
                        visited[ny, nx] = True
                        q.append((ny, nx))
            if (x1 - x0) >= min_side and (y1 - y0) >= min_side:
                comps.append((area, [x0 / _SIZE, y0 / _SIZE,
                                     (x1 + 1) / _SIZE, (y1 + 1) / _SIZE]))

    comps.sort(key=lambda t: -t[0])
    return [bb for _, bb in comps[:max_n]]
