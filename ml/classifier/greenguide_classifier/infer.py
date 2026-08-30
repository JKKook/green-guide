"""추론 공통 — ONNX 세션 생성·softmax·torch device 선택.

스크립트마다 복붙되던 세 조각을 모은 것. 도메인 로직 없음.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

CPU_PROVIDERS: tuple[str, ...] = ("CPUExecutionProvider",)


def load_session(path: Path | str, providers: Sequence[str] = CPU_PROVIDERS) -> ort.InferenceSession:
    """ONNX Runtime 세션. 기본은 CPU EP (배포 환경과 동일)."""
    return ort.InferenceSession(str(path), providers=list(providers))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """수치 안정 softmax (max-shift). 1-D 는 axis=-1, (B,C) 는 axis=1, (C,h,w) 셀별은 axis=0."""
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def pick_device() -> torch.device:
    """cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
