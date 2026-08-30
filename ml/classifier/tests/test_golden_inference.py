"""Characterization test — 현행 ONNX 모델의 출력을 고정(golden)해 리팩토링 안전망으로 쓴다.

입력은 seed 고정 난수 텐서 3장(실제 이미지 불필요, 재현성 보장). 모델 파일(*.onnx)은
git 미추적이므로 없으면 skip. golden 재생성: `python -m tests.test_golden_inference --update`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from src import config

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "golden_logits.json"
# worktree 등 outputs/ 가 없는 체크아웃에서는 env 로 모델 위치를 지정
MODELS_DIR = Path(os.getenv("WASTE_GOLDEN_MODELS_DIR", config.MODELS_DIR))
MODELS = {
    "cnn": MODELS_DIR / "cnn" / "classifier.onnx",
    "cnn_hier": MODELS_DIR / "cnn_hier" / "classifier.onnx",
}


def fixed_inputs() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((3, 3, 224, 224), dtype=np.float32)


def run_logits(onnx_path: Path) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return sess.run(["logits"], {"image": fixed_inputs()})[0]


@pytest.mark.parametrize("name", sorted(MODELS))
def test_logits_match_golden(name: str) -> None:
    path = MODELS[name]
    if not path.exists():
        pytest.skip(f"{path} 없음 (모델 파일은 git 미추적)")
    golden = json.loads(GOLDEN_PATH.read_text())
    if name not in golden:
        pytest.skip(f"golden 에 {name} 없음 — --update 로 생성")
    got = run_logits(path)
    exp = np.asarray(golden[name], dtype=np.float32)
    assert got.shape == exp.shape
    np.testing.assert_allclose(got, exp, rtol=1e-4, atol=1e-4)


if __name__ == "__main__" and "--update" in sys.argv:
    out = {n: run_logits(p).tolist() for n, p in MODELS.items() if p.exists()}
    GOLDEN_PATH.parent.mkdir(exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(out))
    print(f"golden 갱신: {list(out)} → {GOLDEN_PATH}")
