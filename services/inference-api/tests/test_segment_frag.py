"""/predict-objects 근접 촬영 오분할 플래그 — saliency 를 스텁해 네트워크·모델 없이 검증."""
from __future__ import annotations

import numpy as np
import pytest

from src import segment
from src.core import config


def _speckle() -> np.ndarray:
    """큰 블롭 2개(각 ≥6% 변) + 작은 점 30개 → 성분 32개."""
    p = np.zeros((320, 320), dtype=np.float32)
    p[20:120, 20:120] = 1.0
    p[180:300, 180:300] = 1.0
    for i in range(30):
        y, x = 140 + (i // 10) * 10, 10 + (i % 10) * 15
        p[y:y + 3, x:x + 3] = 1.0
    return p


@pytest.mark.parametrize("on,n", [(False, 2), (True, 1)])
def test_frag_whole(monkeypatch, on, n) -> None:
    monkeypatch.setattr(segment, "_saliency_prob", lambda _b: _speckle())
    monkeypatch.setattr(config, "OBJECTS_FRAG_WHOLE", on)
    bbs = segment.all_component_bboxes(b"")
    assert len(bbs) == n
    if on:
        assert bbs == [[0.0, 0.0, 1.0, 1.0]]


def test_frag_whole_keeps_few_components(monkeypatch) -> None:
    p = _speckle()
    p[140:170, :] = 0.0                       # 점 제거 → 성분 2개(진짜 다중 물건)
    monkeypatch.setattr(segment, "_saliency_prob", lambda _b: p)
    monkeypatch.setattr(config, "OBJECTS_FRAG_WHOLE", True)
    assert len(segment.all_component_bboxes(b"")) == 2
