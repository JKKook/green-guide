"""PRODUCTION_READINESS 결정 5·7·8 재현 테스트 (W2 모델·추론 묶음)."""
from __future__ import annotations

import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import clip_identity as ci
from src import hier_inference as hi
from src.core import config
from src.services.regions_service import displayable_evidence

F2C = {"pet": "plastic", "plastic_other": "plastic", "clothes": "clothes",
       "non_object": "non_object"}


# ── 결정 8: OOD 보정값 가드 ────────────────────────────────────────────────

@pytest.fixture()
def no_env_tau(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "OOD_TAU_SOFT", None)
    monkeypatch.setattr(config, "OOD_TAU_HARD", None)


def test_ood_calibrated_sidecar_uses_values(no_env_tau, caplog) -> None:
    tax = {"ood": {"tau_soft": 0.558, "tau_hard": 0.674}}
    with caplog.at_level(logging.WARNING, logger="src.hier_inference"):
        soft, hard, ok = hi.check_ood_calibration(tax)
    assert (soft, hard, ok) == (0.558, 0.674, True)
    assert not caplog.records


def test_ood_missing_fails_open_with_one_warning(no_env_tau, caplog) -> None:
    """ood 키가 없으면 구 기본값(0.30/0.40)이 아니라 끔 + 경고 1줄."""
    with caplog.at_level(logging.WARNING, logger="src.hier_inference"):
        soft, hard, ok = hi.check_ood_calibration({}, "x/taxonomy.json")
    assert soft == float("inf") and hard == float("inf") and ok is False
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1 and warns[0].name == "src.hier_inference"


def test_ood_legacy_tau_only_is_partial(no_env_tau, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="src.hier_inference"):
        soft, hard, ok = hi.check_ood_calibration({"ood": {"tau": 0.68}})
    assert hard == 0.68 and soft == float("inf") and ok is False
    assert "tau_soft" in caplog.records[0].getMessage()


def test_ood_env_overrides_sidecar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "OOD_TAU_SOFT", 0.5)
    monkeypatch.setattr(config, "OOD_TAU_HARD", 0.7)
    assert hi.check_ood_calibration({}) == (0.5, 0.7, True)


# ── 결정 5: identity 칩 정책 ───────────────────────────────────────────────

def _ident(slug: str, score: float = 0.7) -> dict:
    return {"type": "identity", "token": slug, "matched_text": slug,
            "mapped_class": slug, "score": score}


def test_chip_dropped_when_non_object() -> None:
    assert displayable_evidence([_ident("non_object")], "plastic", F2C) == []


def test_chip_dropped_when_coarse_differs() -> None:
    assert displayable_evidence([_ident("clothes")], "plastic", F2C) == []


def test_chip_kept_when_coarse_matches_and_other_types_kept() -> None:
    ocr = {"type": "ocr", "token": "PET", "matched_text": "PET",
           "mapped_class": "clothes", "score": 2.0}
    out = displayable_evidence([_ident("pet"), ocr], "plastic", F2C)
    assert out == [_ident("pet"), ocr]


def test_chip_threshold_equals_decision_threshold() -> None:
    """판정 임계(0.30) 미만 정체는 칩으로도 나오지 않는다 (구 0.25)."""
    eng = object.__new__(ci.ClipIdentity)
    eng.slugs = ["pet", "clothes"]
    eng.phrases = ["a pet bottle", "a shirt"]
    eng.phrases_ko = ["페트병", "셔츠"]
    old_scene = ci.SCENE_WEIGHT
    try:
        ci.SCENE_WEIGHT = 0.0
        # top1 0.27: 구 표시 임계(0.25)는 넘지만 판정 임계(0.30) 미만
        _, ev = eng.evidence_prior(np.array([0.27, 0.26]), ["pet", "clothes"])
        assert ev == []
        _, ev = eng.evidence_prior(np.array([0.31, 0.69]), ["pet", "clothes"])
        assert [e["mapped_class"] for e in ev] == ["clothes"]
    finally:
        ci.SCENE_WEIGHT = old_scene


# ── 결정 7: /predict-with-regions = 계층 모델 ─────────────────────────────

def test_regions_uses_hier_model(client: TestClient, sample_image_bytes: bytes,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        clf = hi.get_hier_classifier()
    except FileNotFoundError:
        pytest.skip("계층 모델 미배치")
    from src.routers import inference as ri
    monkeypatch.setattr(ri, "stage1_gate", lambda raw: None)
    res = client.post("/predict-with-regions",
                      files={"image": ("x.jpg", sample_image_bytes, "image/jpeg")})
    assert res.status_code == 200, res.text
    d = res.json()
    assert d["model_arch"].startswith("cnn_hier")
    vocab = set(clf.fine_labels) | set(clf.coarse_labels) | {"etc"}
    assert d["predicted_class"] in vocab
    assert set(d["all_probabilities"]) == set(clf.coarse_labels)
