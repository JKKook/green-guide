"""DINOv2-small + linear head — cloud 13-class 분류기 (ResNet18 의 ensemble/fallback).

Cascade Stage 2 의 second opinion. ResNet18 이 confident-wrong (OOD 입력에 대해
높은 confidence 로 잘못된 분류) 인 케이스를 잡기 위해 distribution-robust SSL 표현을
사용. CLS 토큰 384-dim → Linear(384, 13) → softmax.

사용:
    cls = get_dinov2_classifier()
    if cls.available:
        result = cls.predict(raw_bytes)   # {"predicted_class": "...", "confidences": {...}}
"""
from __future__ import annotations

import io
import json
import threading

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.core import config
from src.core.log import get_logger
from src.core.singleton import lazy_singleton

log = get_logger(__name__)


_DINOV2_PATH = config.PROJECT_ROOT / "cache" / "dinov2_classifier.onnx"
_FALLBACK_PATH = config.PROJECT_ROOT / "models" / "dinov2_classifier.onnx"
_LABELS_PATH = config.PROJECT_ROOT / "models" / "dinov2_labels.json"
_INPUT_SIZE = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DINOv2Classifier:
    """DINOv2 + linear head ONNX wrapper."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: ort.InferenceSession | None = None
        self._input_name = ""
        self._labels: list[str] = []
        self._load()

    def _load(self) -> None:
        # ONNX 모델 로드
        for path in (_DINOV2_PATH, _FALLBACK_PATH):
            if path.exists():
                try:
                    self._session = ort.InferenceSession(
                        str(path), providers=["CPUExecutionProvider"],
                    )
                    self._input_name = self._session.get_inputs()[0].name
                    log.info(f"loaded: {path}")
                    break
                except Exception as exc:  # noqa: BLE001
                    log.info(f"load failed {path}: {exc}")
        else:
            log.info(f"WARN: no model (checked {_DINOV2_PATH}, {_FALLBACK_PATH})")
            return

        # labels 로드
        if _LABELS_PATH.exists():
            try:
                self._labels = json.loads(_LABELS_PATH.read_text())
                log.info(f"labels: {self._labels}")
            except Exception as exc:  # noqa: BLE001
                log.info(f"labels load failed: {exc}")
                self._session = None
        else:
            log.info(f"WARN: labels.json missing at {_LABELS_PATH}")
            self._session = None

    @property
    def available(self) -> bool:
        return self._session is not None and len(self._labels) > 0

    def _preprocess(self, raw: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize(
            (_INPUT_SIZE, _INPUT_SIZE), Image.BILINEAR,
        )
        arr = (np.array(img, dtype=np.float32) / 255.0 - _MEAN) / _STD
        return arr.transpose(2, 0, 1)[None].astype(np.float32)

    def predict(self, raw: bytes) -> dict | None:
        """이미지 → {predicted_class, confidences{...}}. 실패시 None."""
        if not self.available:
            return None
        try:
            inp = self._preprocess(raw)
        except Exception as exc:  # noqa: BLE001
            log.info(f"preprocess failed: {exc}")
            return None
        with self._lock:
            logits = self._session.run(None, {self._input_name: inp})[0][0]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        confidences = {label: float(probs[i]) for i, label in enumerate(self._labels)}
        top_idx = int(np.argmax(probs))
        return {
            "predicted_class": self._labels[top_idx],
            "confidences": confidences,
        }


@lazy_singleton
def get_dinov2_classifier() -> DINOv2Classifier:
    return DINOv2Classifier()
