"""W6 electronics 편향 완화 — 확정 기본값 + 각 장치가 의도한 경우에만 작동하는지."""
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


def test_confirmed_defaults() -> None:
    """실사용 51·35장 전후 측정으로 확정한 값 (W6 수정 ②~⑤)."""
    assert config.SCENE_AUTOCROP is True
    assert config.NONOBJ_RAW_GUARD == 0.5
    assert config.ELEC_LOGIT_BIAS == 0.5
    assert config.ELEC_COARSE_GATE == 0.7
    assert config.ELEC_CLIP_MIN == 0.3
    # 실험 전용은 꺼짐
    assert config.ELEC_CLIP_PRIOR is False
    assert config.VLM_ARBITRATE_CLASSES == frozenset()


@pytest.fixture(autouse=True)
def _flags_off(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """메커니즘 테스트는 장치를 하나씩 켜서 본다 (기본값 테스트는 제외)."""
    if request.node.name == "test_confirmed_defaults":
        return
    for name in ("ELEC_LOGIT_BIAS", "ELEC_COARSE_GATE", "NONOBJ_RAW_GUARD"):
        monkeypatch.setattr(config, name, 0.0)


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


def test_elec_coarse_gate_rejects_low_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    logits = [1.0, 0.5, 2.0, 0.0]          # electronics ≈ 0.6 → 기본 게이트(0.55) 통과
    assert _clf(logits).predict(X)["display_level"] != "reject"
    monkeypatch.setattr(config, "ELEC_COARSE_GATE", 0.7)
    assert _clf(logits).predict(X)["display_level"] == "reject"
    # electronics 가 아닌 판정에는 영향 없음
    assert _clf([3.0, 0.5, 1.0, 0.0]).predict(X)["display_level"] != "reject"


def test_nonobj_raw_guard_rejects_instead_of_redistribution(
        monkeypatch: pytest.MonkeyPatch) -> None:
    logits = [0.0, 0.0, 2.0, 4.0]          # raw non_object ≈ 0.86 → 마스킹 시 electronics 로 재분배
    assert _clf(logits).predict(X, mask_non_object=True)["coarse_class"] == "electronics"
    monkeypatch.setattr(config, "NONOBJ_RAW_GUARD", 0.5)
    assert _clf(logits).predict(X, mask_non_object=True)["display_level"] == "reject"
    # 마스킹하지 않는 호출(게이트 미통과 경로)에는 적용 안 함
    assert _clf(logits).predict(X)["coarse_class"] == "non_object"
