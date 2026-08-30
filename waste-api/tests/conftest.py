"""공통 fixture."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture()
def sample_image_bytes() -> bytes:
    """랜덤 RGB 224x224 이미지를 JPEG bytes 로 생성."""
    arr = (np.random.default_rng(0).random((224, 224, 3)) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture()
def real_sample_image_bytes() -> bytes | None:
    """waste-preprocessor의 실제 cardboard 샘플을 bytes 로 반환 (있으면)."""
    candidate = (
        Path(__file__).resolve().parent.parent.parent
        / "waste-preprocessor" / "data" / "raw" / "garbage-classification"
        / "cardboard" / "cardboard1.jpg"
    )
    if not candidate.exists():
        return None
    return candidate.read_bytes()


@pytest.fixture()
def client() -> TestClient:
    """TestClient — 실제 ONNX 모델 로드함 (waste-classifier 학습 완료가 전제)."""
    from src.api import app
    with TestClient(app) as c:
        yield c
