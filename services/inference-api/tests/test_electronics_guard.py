"""W6 electronics 편향 완화 플래그 — 기본 꺼짐 + bias 가 electronics 로짓만 낮추는지."""
from __future__ import annotations

import numpy as np
import pytest

from src import hier_inference as hi
from src.core import config

FINE = ["metal", "pet", "electronics", "non_object"]
COARSE = ["metal", "plastic", "electronics", "non_object"]


class _Sess:
    def __init__(self, logits: list[float]) -> None:
        self.logits = np.array([logits], dtype=np.float32)

    def run(self, outputs, feed):  # noqa: ARG002
        return [self.logits]


def _clf(logits: list[float]) -> hi.HierWasteClassifier:
    c = object.__new__(hi.HierWasteClassifier)
    c.fine_labels, c.coarse_labels = FINE, COARSE
    c.f2c_idx = [0, 1, 2, 3]
    c.gate = {"fine_min_confidence": 0.6, "fine_min_margin": 0.15,
              "coarse_min_confidence": 0.55}
    c.taxonomy = {"version": "test"}
    c.session, c.dino_session, c.ood_protos = _Sess(logits), None, None
    c.ood_tau_soft = c.ood_tau_hard = float("inf")
    return c


X = np.zeros((1, 3, 224, 224), dtype=np.float32)


def test_flags_default_off() -> None:
    assert config.ELEC_LOGIT_BIAS == 0
    assert config.ELEC_CLIP_PRIOR is False
    assert config.VLM_ARBITRATE_CLASSES == frozenset()


def test_bias_lowers_only_electronics(monkeypatch: pytest.MonkeyPatch) -> None:
    logits = [1.0, 0.5, 3.0, 0.0]
    base = _clf(logits).predict(X)
    monkeypatch.setattr(config, "ELEC_LOGIT_BIAS", 3.0)
    out = _clf(logits).predict(X)
    assert base["coarse_class"] == "electronics"
    assert out["coarse_class"] == "metal"
    p_b, p_o = base["coarse_probabilities"], out["coarse_probabilities"]
    assert p_o["electronics"] < p_b["electronics"]
    # 나머지 클래스 간 상대 비율은 보존
    assert p_o["metal"] / p_o["plastic"] == pytest.approx(p_b["metal"] / p_b["plastic"], rel=1e-4)
