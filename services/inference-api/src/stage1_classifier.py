"""Stage 1 — Binary waste/non-waste classifier (MobileNetV3-Small).

Two-stage Cascade 의 1단계. /predict-centered, /predict-with-regions 에서 호출.
- waste: 분리수거 대상 (cardboard, plastic, etc 등 12 클래스)
- non_object: 손/배경/잡 객체 (재촬영 안내)

OOD 입력을 명시적으로 차단 → 2단계 13-class 분류기의 confident-wrong 우회.
"""
from __future__ import annotations

import io
import threading

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.core import config
from src.core.log import get_logger
from src.core.singleton import lazy_singleton

log = get_logger(__name__)


_STAGE1_PATH = config.PROJECT_ROOT / "cache" / "stage1_binary.onnx"
_FALLBACK_PATH = config.PROJECT_ROOT / "models" / "stage1_binary.onnx"
_INPUT_SIZE = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Stage1Classifier:
    """MobileNetV3-Small ONNX wrapper — image → (is_waste, confidence)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: ort.InferenceSession | None = None
        self._input_name = ""
        self._load()

    def _load(self) -> None:
        for path in (_STAGE1_PATH, _FALLBACK_PATH):
            if path.exists():
                try:
                    self._session = ort.InferenceSession(
                        str(path), providers=["CPUExecutionProvider"],
                    )
                    self._input_name = self._session.get_inputs()[0].name
                    log.info(f"loaded: {path}")
                    return
                except Exception as exc:  # noqa: BLE001
                    log.info(f"load failed {path}: {exc}")
        log.info(f"WARN: no stage1 model found (checked {_STAGE1_PATH}, {_FALLBACK_PATH})")

    @property
    def available(self) -> bool:
        return self._session is not None

    def _preprocess(self, raw: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize(
            (_INPUT_SIZE, _INPUT_SIZE), Image.BILINEAR,
        )
        arr = (np.array(img, dtype=np.float32) / 255.0 - _MEAN) / _STD
        return arr.transpose(2, 0, 1)[None].astype(np.float32)

    def predict(self, raw: bytes) -> tuple[bool, float]:
        """이미지 → (is_waste, waste_prob). 모델 없으면 항상 True (cascade 비활성)."""
        if self._session is None:
            return True, 1.0
        try:
            inp = self._preprocess(raw)
        except Exception as exc:  # noqa: BLE001
            log.info(f"preprocess failed: {exc}")
            return True, 1.0
        with self._lock:
            logits = self._session.run(None, {self._input_name: inp})[0][0]
        # softmax
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        # index 0 = non_object, index 1 = waste (train_stage1_binary.py 와 일치)
        waste_prob = float(probs[1])
        return waste_prob >= 0.5, waste_prob


@lazy_singleton
def get_stage1_classifier() -> Stage1Classifier:
    return Stage1Classifier()
