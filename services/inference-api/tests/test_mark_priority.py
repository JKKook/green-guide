"""분리배출 표시 최우선 판정 (feature/api-mark-priority, 2026-10-05).

스마트촬영(capture_mode=smart)은 확신도와 무관하게 OCR 을 돌리고, 몸체 표시가
읽히면 모델 결과를 그 재질로 교체한다. 갤러리 경로·부속 표기(캡:PP)는 교체하지 않는다.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.routers.inference import apply_mark_override
from src.semantic_evidence import mark_override, match_evidence

FINE_TO_COARSE = {"pet": "plastic", "plastic_other": "plastic", "metal": "metal",
                  "trash_other": "trash", "carton": "paper_pack"}


def _t(text: str, score: float = 0.9) -> dict[str, Any]:
    return {"text": text, "score": score, "bbox_norm": [0.1, 0.1, 0.3, 0.2]}


# ── 어휘·부속 표기 ──────────────────────────────────────────────────────────

def test_other_mark_maps_to_plastic_other() -> None:
    ev = match_evidence([_t("OTHER")])
    assert [(e["type"], e["mapped_class"], e["attachment"]) for e in ev] == [
        ("mark", "plastic_other", False)]


def test_attachment_mark_flagged_and_not_primary() -> None:
    ev = match_evidence([_t("캡:PP")])
    assert ev and ev[0]["mapped_class"] == "plastic_other" and ev[0]["attachment"] is True
    assert mark_override(ev) is None          # 부속 표기만으론 교체 안 함


def test_override_picks_body_mark_with_highest_score() -> None:
    ev = match_evidence([_t("캡:PP", 0.95), _t("무색페트", 0.7), _t("OTHER", 0.8)])
    best = mark_override(ev)
    assert best is not None and best["token"] == "other" and best["primary"] is True
    assert sum(e["primary"] for e in ev) == 1


def test_override_respects_min_score() -> None:
    ev = match_evidence([_t("PET", 0.5)])
    assert mark_override(ev, min_score=0.6) is None
    assert mark_override(ev, min_score=0.4) is not None


def test_new_2021_marks() -> None:
    assert match_evidence([_t("도포·첩합")])[0]["mapped_class"] == "trash_other"
    assert match_evidence([_t("알미늄")])[0]["mapped_class"] == "metal"
    assert match_evidence([_t("일반팩")])[0]["mapped_class"] == "carton"


def test_ambiguous_tokens_need_mark_context() -> None:
    # 일반 문장 — 매칭 자체를 하지 않는다 (교체 판정 오탐 방지)
    def marks(text: str) -> list[dict]:
        return [e for e in match_evidence([_t(text)]) if e["type"] == "mark"]
    assert marks("얇게 도포하세요") == []
    assert marks("발포정 비타민") == []              # '비타민' 정체어(text)만 남는다
    assert marks("and other items") == []
    # 표시 줄 모양 — 매칭
    assert match_evidence([_t("발포스티렌")])[0]["mapped_class"] == "styrofoam"
    both = {(e["token"], e["attachment"]) for e in marks("OTHER 캡:PP")}
    assert ("other", False) in both and ("pp", True) in both


def test_weak_token_never_overrides() -> None:
    ev = match_evidence([_t("철분 함유", 0.95)])      # '철' 은 약화 토큰 → 보조 증거만
    assert ev and ev[0]["mapped_class"] == "metal"
    assert mark_override(ev) is None


def test_body_and_attachment_on_one_line() -> None:
    ev = match_evidence([_t("몸체:PET 라벨:PP 뚜껑:HDPE")])
    by = {(e["token"], e["attachment"]) for e in ev if e["type"] == "mark"}
    assert ("pet", False) in by and ("pp", True) in by and ("hdpe", True) in by
    assert mark_override(ev)["token"] == "pet"


def test_attachment_line_before_body_line_still_overrides() -> None:
    ev = match_evidence([_t("뚜껑:PP", 0.9), _t("PP", 0.8)])
    assert mark_override(ev)["token"] == "pp" and mark_override(ev)["attachment"] is False


def test_longer_token_wins_same_line() -> None:
    ev = match_evidence([_t("종이팩")])
    assert mark_override(ev)["mapped_class"] == "carton"


# ── 결과 교체 ──────────────────────────────────────────────────────────────

BASE = {"display_level": "fine", "display_class": "metal", "coarse_class": "metal",
        "coarse_confidence": 0.9, "fine_class": "metal", "fine_confidence": 0.9,
        "fine_margin": 0.5, "coarse_probabilities": {"metal": 0.9, "plastic": 0.05},
        "fine_top5": [], "model_arch": "test", "inference_ms": 1.0}


def test_apply_override_fine_target_other_class_uses_ocr_score() -> None:
    mark = {"mapped_class": "pet", "score": 0.8, "token": "pet"}
    out = apply_mark_override(dict(BASE), mark, FINE_TO_COARSE)
    assert (out["display_level"], out["display_class"], out["coarse_class"]) == ("fine", "pet", "plastic")
    assert out["fine_confidence"] == pytest.approx(0.8)        # metal 0.9 를 승계하지 않음
    assert out["coarse_confidence"] == pytest.approx(0.8)
    assert out["coarse_probabilities"]["plastic"] == pytest.approx(0.8)
    assert sum(out["coarse_probabilities"].values()) == pytest.approx(1.0)
    assert "mark:pet" in out["model_arch"]


def test_apply_override_same_class_keeps_higher_confidence() -> None:
    mark = {"mapped_class": "metal", "score": 0.7, "token": "캔류"}
    out = apply_mark_override(dict(BASE), mark, FINE_TO_COARSE)
    assert out["coarse_confidence"] == pytest.approx(0.9)
    assert sum(out["coarse_probabilities"].values()) == pytest.approx(1.0)


def test_apply_override_coarse_target() -> None:
    mark = {"mapped_class": "glass", "score": 0.7, "token": "유리"}
    out = apply_mark_override(dict(BASE), mark, FINE_TO_COARSE)
    assert (out["display_level"], out["display_class"], out["fine_class"]) == ("coarse", "glass", None)
    assert out["coarse_confidence"] == pytest.approx(0.7)
    assert out["coarse_probabilities"]["glass"] == pytest.approx(0.7)


# ── 라우터 배선: smart 는 항상 OCR + 교체, gallery 는 기존 가드 ───────────────

HIER_RESULT = dict(BASE)


@pytest.fixture()
def stub(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {"ocr_calls": 0}
    import src.hier_inference as hi
    import src.routers.inference as ri
    import src.semantic_evidence as se

    def fake_predict_rotations(clf, raw, degs, **kw):
        return dict(HIER_RESULT), np.zeros((1, 3, 448, 448), dtype=np.float32)

    class FakeOCR:
        available = True

        def read_texts(self, image_bytes):
            captured["ocr_calls"] += 1
            return [_t("OTHER", 0.9)]

    from types import SimpleNamespace
    stub_clf = SimpleNamespace(
        taxonomy={"fine_to_coarse": FINE_TO_COARSE},
        fine_labels=list(FINE_TO_COARSE),
        predict=lambda *a, **kw: dict(HIER_RESULT),
    )
    monkeypatch.setattr(hi, "get_hier_classifier", lambda: stub_clf)
    monkeypatch.setattr(hi, "predict_rotations", fake_predict_rotations)
    monkeypatch.setattr(ri, "non_object_gate", lambda raw: None)
    monkeypatch.setattr(ri, "record_safely", lambda *a, **kw: "test-upload-id")
    monkeypatch.setattr(se, "get_evidence_engine", lambda: FakeOCR())
    return captured


def _post(client: TestClient, image: bytes, **form: str) -> dict:
    res = client.post("/predict-hier", files={"image": ("x.jpg", image, "image/jpeg")}, data=form)
    assert res.status_code == 200, res.text
    return res.json()


def test_smart_capture_mark_overrides_confident_model(client: TestClient, sample_image_bytes: bytes,
                                                      stub: dict) -> None:
    body = _post(client, sample_image_bytes, capture_mode="smart")
    assert stub["ocr_calls"] == 1                       # 확신 0.9 여도 OCR 실행
    assert body["display_class"] == "plastic_other" and body["coarse_class"] == "plastic"
    primary = [e for e in body["evidence"] if e["primary"]]
    assert len(primary) == 1 and primary[0]["token"] == "other"


def test_gallery_keeps_confidence_gate(client: TestClient, sample_image_bytes: bytes,
                                       stub: dict) -> None:
    body = _post(client, sample_image_bytes, capture_mode="gallery")
    assert stub["ocr_calls"] == 0                       # 확신 0.9 ≥ 0.75 → 스킵 (기존 동작)
    assert body["display_class"] == "metal"
