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
