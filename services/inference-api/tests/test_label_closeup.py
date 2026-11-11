"""라벨 근접 오분류 대응 A·B·C (11-11 40e4a32a 페트병 라벨 → 종이 단정). 네트워크 무관."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.core import config
from src.semantic_evidence import cosmetic_prior, has_cosmetic_text
from tests.test_mark_priority import FINE_TO_COARSE, _post, _t, stub  # noqa: F401

F2C = {**FINE_TO_COARSE, "paper_other": "paper", "glass_clear": "glass"}
FINE = list(F2C)


def test_cosmetic_prior_scales_paper_down() -> None:
    p = cosmetic_prior([_t("모이스처 트리트먼트 500ml")], FINE, F2C)
    assert p is not None
    assert p[FINE.index("paper_other")] == 0.5 and p[FINE.index("pet")] == 1.5
    assert p[FINE.index("glass_clear")] == 1.2 and p[FINE.index("metal")] == 1.0
    assert cosmetic_prior([_t("영수증 승인번호")], FINE, F2C) is None


def test_has_cosmetic_text_ignores_spaces() -> None:
    assert has_cosmetic_text([_t("헤어 트리 트먼트")]) and not has_cosmetic_text([_t("가 나")])


class _Ocr:
    available = True

    def __init__(self, texts):
        self.texts = texts

    def read_texts(self, image_bytes):
        return self.texts


def _vlm(monkeypatch, slug: str, conf: float):
    import src.vlm_fallback as vf

    class V:
        def classify(self, *a, **k):
            return {"slug": slug, "item_name": None, "stream": None, "condition": None,
                    "confidence": conf, "reason": "페트병 라벨로 보임"}
    monkeypatch.setattr(vf, "get_vlm_fallback", lambda: V())


@pytest.mark.parametrize("on", [True, False])
def test_disagree_confirm(client: TestClient, sample_image_bytes: bytes, stub: dict,  # noqa: F811
                          monkeypatch: pytest.MonkeyPatch, on: bool) -> None:
    import src.semantic_evidence as se
    monkeypatch.setattr(se, "get_evidence_engine", lambda: _Ocr([]))
    monkeypatch.setattr(config, "VLM_DISAGREE_CONFIRM", on)
    monkeypatch.setattr(config, "VLM_TRIGGER_CONF", 1.01)
    _vlm(monkeypatch, "pet", 0.6)                     # CNN metal 0.9 와 다른 대분류, 0.8 미달
    body = _post(client, sample_image_bytes, capture_mode="smart")
    assert body["display_class"] == "metal"            # 결과 재질은 그대로
    if on:
        assert body["needs_confirm"] is True
        assert body["alt_candidate"] == {"class": "pet", "coarse_class": "plastic",
                                         "confidence": 0.6, "reason": "페트병 라벨로 보임"}
    else:
        assert body["needs_confirm"] is None and body["alt_candidate"] is None


def test_disagree_confirm_not_when_same_coarse_or_adopted(
        client: TestClient, sample_image_bytes: bytes, stub: dict,  # noqa: F811
        monkeypatch: pytest.MonkeyPatch) -> None:
    import src.semantic_evidence as se
    monkeypatch.setattr(se, "get_evidence_engine", lambda: _Ocr([]))
    monkeypatch.setattr(config, "VLM_DISAGREE_CONFIRM", True)
    monkeypatch.setattr(config, "VLM_TRIGGER_CONF", 1.01)
    _vlm(monkeypatch, "pet", 0.9)                      # 채택되는 판정 — 확인 필요 아님
    assert _post(client, sample_image_bytes, capture_mode="smart")["needs_confirm"] is None


@pytest.mark.parametrize("coarse,texts,hint", [
    ("paper", [_t("모이스처 트리트먼트")], "whole_object"),
    ("paper", [_t("트리트먼트"), _t("PET", 0.9)], None),   # 분리배출 표시가 있으면 신호 없음
    ("paper", [_t("가" * 70)], None),                      # 글자만 많은 종이 문서 — 신호 없음
    ("metal", [_t("트리트먼트")], None),                   # 종이 판정이 아니면 신호 없음
])
def test_retake_hint(client: TestClient, sample_image_bytes: bytes, stub: dict,  # noqa: F811
                     monkeypatch: pytest.MonkeyPatch, coarse, texts, hint) -> None:
    import src.semantic_evidence as se
    import tests.test_mark_priority as tmp
    monkeypatch.setitem(tmp.HIER_RESULT, "coarse_class", coarse)
    monkeypatch.setattr(se, "get_evidence_engine", lambda: _Ocr(texts))
    monkeypatch.setattr(config, "RETAKE_TEXT_HINT", True)
    monkeypatch.setattr(config, "MARK_PRIORITY_SMART", False)   # 표시 교체와 분리해 신호만 본다
    monkeypatch.setattr(config, "OCR_SKIP_CONFIDENCE", 1.01)
    assert _post(client, sample_image_bytes, capture_mode="smart")["retake_hint"] == hint


def test_cosmetic_prior_router(client: TestClient, sample_image_bytes: bytes, stub: dict,  # noqa: F811
                               monkeypatch: pytest.MonkeyPatch) -> None:
    import src.semantic_evidence as se
    seen: dict = {}
    import src.hier_inference as hi
    clf = hi.get_hier_classifier()

    def pred(*a, fine_prior=None, **k):
        seen["prior"] = fine_prior
        from tests.test_mark_priority import HIER_RESULT
        return dict(HIER_RESULT)
    monkeypatch.setattr(clf, "predict", pred)
    monkeypatch.setattr(se, "get_evidence_engine", lambda: _Ocr([_t("헤어 트리트먼트")]))
    monkeypatch.setattr(config, "COSMETIC_PRIOR", True)
    _post(client, sample_image_bytes, capture_mode="smart")
    assert seen["prior"] is not None and np.isclose(
        seen["prior"][list(FINE_TO_COARSE).index("pet")], 1.5 * seen["prior"][
            list(FINE_TO_COARSE).index("metal")])


def test_retake_hint_survives_prior_repredict(client: TestClient, sample_image_bytes: bytes,
                                              stub: dict,  # noqa: F811
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """B prior 재예측이 result 를 갈아끼워도 C 신호가 남는다 (40e4a32a 실측 회귀)."""
    import src.semantic_evidence as se
    import tests.test_mark_priority as tmp
    monkeypatch.setitem(tmp.HIER_RESULT, "coarse_class", "paper")
    monkeypatch.setattr(se, "get_evidence_engine", lambda: _Ocr([_t("핸드트리트먼트")]))
    monkeypatch.setattr(config, "COSMETIC_PRIOR", True)
    monkeypatch.setattr(config, "RETAKE_TEXT_HINT", True)
    assert _post(client, sample_image_bytes, capture_mode="smart")["retake_hint"] == "whole_object"
