"""Characterization test — 리팩토링 전 전처리 출력 고정 (Phase 0).

의도된 동작 변경(예: EXIF 처리 도입) 시에만 스냅샷 값을 갱신하고, 커밋 메시지에 명시한다.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from greenguide_preprocessor.preprocess import image_stats, preprocess_image


# 결정적 입력: seed 0, 600x800 RGB — conftest.sample_image 와 동일 생성식이지만
# JPEG 인코딩 의존을 피하기 위해 PNG 로 저장.
def _sample(tmp_path: Path) -> Path:
    arr = (np.random.default_rng(0).random((600, 800, 3)) * 255).astype(np.uint8)
    path = tmp_path / "sample.png"
    Image.fromarray(arr, mode="RGB").save(path, format="PNG")
    return path


def test_preprocess_output_snapshot(tmp_path: Path) -> None:
    arr = preprocess_image(_sample(tmp_path))
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.float32
    digest = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    stats = image_stats(arr)
    # 아래 두 값은 Phase 0 시점 실제 출력. 바뀌면 전처리 동작이 바뀐 것.
    assert digest == "9211f3e2a22e2c1571281c2389bae9a024adb6caca80c711aa69c46b8d1c975d", digest
    assert round(stats["mean"], 6) == 0.218798, stats
