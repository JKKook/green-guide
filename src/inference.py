"""ONNX 모델 추론 wrapper.

서버 시작 시 한 번 로드해서 메모리에 보관, 매 요청마다 재사용.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from src import config


def _softmax(logits: np.ndarray) -> np.ndarray:
    """(B, C) logits → 확률 (numerically stable)."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class WasteClassifier:
    """ONNX 모델 + 메타데이터를 묶은 추론 클래스."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or config.MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX 모델 파일을 찾을 수 없음: {self.model_path}\n"
                f"waste-classifier 의 학습·export 를 먼저 완료해주세요."
            )

        self.arch = config.get_model_arch()
        self.input_name = config.get_input_name(self.arch)
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

    def predict(self, model_input: np.ndarray) -> dict[str, Any]:
        """preprocess 결과 (1, …) → 예측 dict."""
        t0 = time.perf_counter()
        logits = self.session.run(
            [config.ONNX_OUTPUT_NAME],
            {self.input_name: model_input},
        )[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        probs = _softmax(logits)[0]
        idx = int(probs.argmax())
        return {
            "predicted_class": config.CLASS_LABELS[idx],
            "predicted_index": idx,
            "confidence": float(probs[idx]),
            "all_probabilities": {
                label: float(probs[i])
                for i, label in enumerate(config.CLASS_LABELS)
            },
            "model_arch": self.arch,
            "inference_ms": round(elapsed_ms, 2),
        }


_classifier: WasteClassifier | None = None


def get_classifier() -> WasteClassifier:
    """전역 싱글톤 — FastAPI lifespan에서 초기화."""
    global _classifier
    if _classifier is None:
        _classifier = WasteClassifier()
    return _classifier


def reset_classifier() -> None:
    """테스트용: 싱글톤 초기화."""
    global _classifier
    _classifier = None
