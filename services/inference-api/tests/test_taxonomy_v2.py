"""분류 체계 v2 서빙 — VLM fine_v2·state·mark 파싱, 규칙표 어댑터, v1 하위호환 (네트워크 무관)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from src import disposal_v2, vlm_fallback
from src import hier_inference as hi
from src.core import config
from src.schemas import PredictionHierResponse
from src.vlm_fallback import VlmFallback

FINE = ["paper_other", "pet", "glass_clear", "metal", "food_waste", "etc", "non_object"]
F2C = {"paper_other": "paper", "pet": "plastic", "glass_clear": "glass",
       "metal": "metal", "food_waste": "food_waste", "etc": "etc",
       "non_object": "non_object"}

needs_v2 = pytest.mark.skipif(not disposal_v2.AVAILABLE,
                              reason="greenguide_common.disposal 미설치")


def _vlm(**kw: object) -> str:
    return json.dumps({"confidence": 0.9, "reason": "r", **kw})


@pytest.fixture()
def mode(monkeypatch: pytest.MonkeyPatch):
    def _set(m: str) -> None:
        monkeypatch.setattr(config, "VLM_TAXONOMY", m)
    return _set


@pytest.fixture(autouse=True)
def _single_by_default(request: pytest.FixtureRequest, mode) -> None:
    # 이 파일의 파서·프롬프트 테스트는 single 기준 — 모드별 테스트는 각자 덮어쓴다
    mode("single")


# ── VLM 파서 ──────────────────────────────────────────────────────────────

@needs_v2
def test_parse_v2_axes_metal_can() -> None:
    out = VlmFallback._parse_response(
        _vlm(slug="metal", fine_v2="metal_can", state="dirty", mark="can"), FINE, F2C)
    assert out is not None
    assert (out["slug"], out["coarse"], out["fine_v2"], out["state"], out["mark"]) == \
        ("metal", "metal", "metal_can", "dirty", "can")


@needs_v2
def test_parse_v2_invalid_axes_become_null_but_material_kept() -> None:
    # 목록 밖 값은 그 축만 버린다 — 재질 판정은 유지(72% 재질 경로 보존)
    out = VlmFallback._parse_response(
        _vlm(slug="pet", fine_v2="metal_can", state="wet", mark="aluminum"), FINE, F2C)
    assert out is not None
    assert out["slug"] == "pet"
    assert (out["fine_v2"], out["state"], out["mark"]) == (None, None, None)


@needs_v2
def test_parse_v2_fine_v2_only_for_split_coarse() -> None:
    out = VlmFallback._parse_response(
        _vlm(slug="food_waste", fine_v2="food_inedible", state="clean"), FINE, F2C)
    assert out["fine_v2"] == "food_inedible"
    out = VlmFallback._parse_response(
        _vlm(slug="glass_clear", fine_v2="food_inedible", state="broken"), FINE, F2C)
    assert out["fine_v2"] is None and out["state"] == "broken"


@needs_v2
def test_parse_v1_response_still_accepted() -> None:
    # 구 프롬프트 형식 답(v2 필드 없음)도 그대로 채택
    out = VlmFallback._parse_response(_vlm(slug="pet"), FINE, F2C)
    assert out is not None and out["slug"] == "pet"
    assert out["state"] is None and out["fine_v2"] is None and out["mark"] is None


def test_parse_generated_item_has_no_v2_axes() -> None:
    out = VlmFallback._parse_response(
        _vlm(slug=None, item_name="소파", stream="bulky_waste"), FINE, F2C)
    assert out is not None and "state" not in out


V1_KEYS = {"slug", "item_name", "stream", "confidence", "reason", "condition"}


def test_v2_disabled_keeps_v1_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(disposal_v2, "AVAILABLE", False)   # 패키지 없음 → single 이어도 off
    out = VlmFallback._parse_response(_vlm(slug="pet", state="dirty"), FINE, F2C)
    assert set(out) == V1_KEYS
    prompt = VlmFallback._taxonomy_prompt(FINE, F2C)
    assert "fine_v2" not in prompt and "5. 반드시 JSON 만 출력" in prompt


@pytest.mark.parametrize("m", ["off", "two_stage", "bogus"])
def test_non_single_modes_use_v1_prompt_verbatim(mode, m: str) -> None:
    # off·two_stage 1회차·잘못된 값 → v1 프롬프트 원문(재질 경로 불변)
    mode("off")
    v1 = VlmFallback._taxonomy_prompt(FINE, F2C)
    mode(m)
    assert VlmFallback._taxonomy_prompt(FINE, F2C) == v1
    out = VlmFallback._parse_response(_vlm(slug="pet", state="dirty"), FINE, F2C)
    assert set(out) == V1_KEYS


def test_default_mode_is_off() -> None:
    import importlib
    import os
    if "GREENGUIDE_VLM_TAXONOMY" in os.environ:
        pytest.skip("환경변수로 모드 지정됨")
    assert importlib.reload(config).VLM_TAXONOMY == "off"


# ── 프롬프트 ──────────────────────────────────────────────────────────────

@needs_v2
def test_prompt_keeps_material_rules_and_adds_v2() -> None:
    p = VlmFallback._taxonomy_prompt(FINE, F2C)
    for rule in ("1. 사진의 가장 주된 폐기물 하나만 판정",
                 "2. 재질도 품목도 확실하지 않으면 slug=etc",
                 "3. 폐기물이 아예 없으면(손·배경만) slug=non_object"):
        assert rule in p
    for token in ("metal_can", "metal_scrap", "food_inedible", "composite",
                  "sterile_pack", '"fine_v2"', '"state"', '"mark"'):
        assert token in p


# ── 규칙표 어댑터 ─────────────────────────────────────────────────────────

@needs_v2
@pytest.mark.parametrize(("coarse", "fine", "state", "mark", "category", "source"), [
    ("paper", "paper_other", "dirty", None, "general", "rule"),      # 기름 묻은 피자 상자
    ("glass", "glass_clear", "broken", None, "general", "rule"),     # 깨진 병
    ("metal", "metal_can", "clean", None, "can", "rule"),
    ("metal", "metal_scrap", None, None, "scrap_metal", "rule"),
    ("food_waste", "food_inedible", None, None, "general", "rule"),  # 조개 껍데기
    ("plastic", "plastic_other", None, "pet", "pet", "mark"),        # 마크 우선
])
def test_resolve_cases(coarse: str, fine: str, state: str | None, mark: str | None,
                       category: str, source: str) -> None:
    d = disposal_v2.resolve(coarse, fine, state, mark)
    assert d is not None
    assert (d["category"], d["source"]) == (category, source)
    assert isinstance(d["notes"], list)


@needs_v2
def test_resolve_metal_without_fine_v2_is_pending() -> None:
    # v14 CNN 은 metal 세부를 모른다 → 판정 보류(null), 캔/고철을 추측하지 않는다
    assert disposal_v2.resolve("metal", "metal", None, None) is None
    assert disposal_v2.resolve("non_object", None, None, None) is None


def test_resolve_disabled_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(disposal_v2, "_rules", None)
    assert disposal_v2.resolve("paper", None, None, None) is None


# ── 응답 하위호환 ─────────────────────────────────────────────────────────

def _base_result() -> dict[str, object]:
    return {"display_level": "fine", "display_class": "pet", "coarse_class": "plastic",
            "coarse_confidence": 0.9, "fine_class": "pet", "fine_confidence": 0.9,
            "fine_margin": 0.5, "coarse_probabilities": {"plastic": 0.9},
            "fine_top5": [{"slug": "pet", "prob": 0.9}], "model_arch": "cnn_hier",
            "inference_ms": 10.0}


def test_response_without_v2_fields_is_valid() -> None:
    r = PredictionHierResponse(**_base_result())
    assert r.disposal is None and r.state is None and r.fine_v2 is None


def test_response_with_disposal_serializes() -> None:
    r = PredictionHierResponse(**_base_result(), state="dirty",
                               disposal={"category": "plastic",
                                         "notes": ["rinse_or_general"], "source": "rule"})
    d = r.model_dump()
    assert d["disposal"] == {"category": "plastic", "notes": ["rinse_or_general"],
                             "source": "rule"}
    assert d["display_class"] == "pet"   # 기존 필드 불변


# ── two_stage 호출 흐름 (가짜 클라이언트 — 네트워크 없음) ─────────────────────

class _FakeMessages:
    def __init__(self, replies: list[object]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, object]] = []

    def create(self, **kw: object) -> object:
        self.calls.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text=r)])


def _fallback(monkeypatch: pytest.MonkeyPatch, tmp_path, replies: list[object]):
    monkeypatch.setattr(vlm_fallback, "_COUNTER_PATH", tmp_path / "calls.json")
    monkeypatch.setattr(vlm_fallback, "_LOG_PATH", tmp_path / "labels.jsonl")
    monkeypatch.setattr(vlm_fallback, "_CANDIDATES_PATH", tmp_path / "cand.jsonl")
    fb = VlmFallback.__new__(VlmFallback)
    fb.available, fb.key_status = True, "ok"
    fb._client = SimpleNamespace(messages=_FakeMessages(replies))
    return fb


@needs_v2
def test_two_stage_second_call_fills_axes(monkeypatch, tmp_path, mode,
                                          sample_image_bytes) -> None:
    mode("off")
    v1_prompt = VlmFallback._taxonomy_prompt(FINE, F2C)
    mode("two_stage")
    fb = _fallback(monkeypatch, tmp_path, [
        _vlm(slug="metal"),
        '{"fine_v2": "metal_can", "state": "dirty", "mark": null}'])
    out = fb.classify(sample_image_bytes, FINE, F2C)
    calls = fb._client.messages.calls
    assert len(calls) == 2 and out["vlm_calls"] == 2
    assert calls[0]["system"][0]["text"] == v1_prompt   # 1회차 = v1 원문
    assert calls[1]["messages"][0]["content"][1]["text"] == "재질 판정: metal (대분류 metal)"
    assert (out["slug"], out["fine_v2"], out["state"], out["mark"]) == \
        ("metal", "metal_can", "dirty", None)
    assert VlmFallback._calls_today() == 2   # 일일 상한은 호출 단위로 센다


@needs_v2
@pytest.mark.parametrize("second", [RuntimeError("boom"), "not json", '{"state": "wet"}'])
def test_two_stage_second_failure_nulls_axes_only(monkeypatch, tmp_path, mode,
                                                  sample_image_bytes, second) -> None:
    mode("two_stage")
    fb = _fallback(monkeypatch, tmp_path, [_vlm(slug="pet"), second])
    out = fb.classify(sample_image_bytes, FINE, F2C)
    assert out is not None and out["slug"] == "pet" and out["confidence"] == 0.9
    assert (out["fine_v2"], out["state"], out["mark"]) == (None, None, None)


@needs_v2
@pytest.mark.parametrize("first", [
    _vlm(slug="non_object"),
    json.dumps({"slug": "pet", "confidence": 0.5, "reason": "r"}),  # 채택 임계 미달
    _vlm(slug=None, item_name="소파", stream="bulky_waste"),
])
def test_two_stage_skips_second_call(monkeypatch, tmp_path, mode,
                                     sample_image_bytes, first) -> None:
    mode("two_stage")
    fb = _fallback(monkeypatch, tmp_path, [first])
    out = fb.classify(sample_image_bytes, FINE, F2C)
    assert len(fb._client.messages.calls) == 1 and out["vlm_calls"] == 1


@needs_v2
def test_two_stage_respects_daily_cap(monkeypatch, tmp_path, mode,
                                      sample_image_bytes) -> None:
    mode("two_stage")
    monkeypatch.setattr(vlm_fallback, "DAILY_CAP", 1)
    fb = _fallback(monkeypatch, tmp_path, [_vlm(slug="pet")])
    out = fb.classify(sample_image_bytes, FINE, F2C)
    assert len(fb._client.messages.calls) == 1
    assert out["slug"] == "pet" and out["state"] is None


@needs_v2
def test_single_mode_makes_one_call(monkeypatch, tmp_path, mode, sample_image_bytes) -> None:
    mode("single")
    fb = _fallback(monkeypatch, tmp_path, [_vlm(slug="metal", fine_v2="metal_scrap")])
    out = fb.classify(sample_image_bytes, FINE, F2C)
    assert len(fb._client.messages.calls) == 1 and out["fine_v2"] == "metal_scrap"


# ── CNN v2 상태 헤드 (ONNX state_logits 계약) ─────────────────────────────────

V2_TAX = {"state_labels": ["clean", "dirty", "broken", "composite"],
          "state_gate": {"min_confidence": 0.6}}


@pytest.mark.parametrize(("tax", "outputs", "on"), [
    (V2_TAX, {"logits", "state_logits"}, True),
    ({}, {"logits"}, False),                                   # v14 — 그대로
    (V2_TAX, {"logits"}, False),                               # 출력 없음
    ({}, {"logits", "state_logits"}, False),                   # 라벨 없음
])
def test_state_head_contract(tax: dict, outputs: set[str], on: bool) -> None:
    labels, _ = hi._state_head(tax, outputs)
    assert (labels is not None) == on


@needs_v2
def test_state_head_rejects_unknown_labels() -> None:
    labels, _ = hi._state_head({"state_labels": ["clean", "wet"]}, {"state_logits"})
    assert labels is None


class _FakeSession:
    def __init__(self, state_logits: list[float]) -> None:
        self.state_logits = np.array([state_logits], dtype=np.float32)
        self.asked: list[str] = []

    def run(self, outputs: list[str], feeds: dict) -> list[np.ndarray]:
        self.asked = list(outputs)
        table = {"logits": np.array([[3.0, 0.0, 0.0]], dtype=np.float32),
                 "state_logits": self.state_logits}
        return [table[o] for o in outputs]


def _tiny_clf(state_logits: list[float] | None) -> hi.HierWasteClassifier:
    clf = hi.HierWasteClassifier.__new__(hi.HierWasteClassifier)
    clf.fine_labels = ["pet", "glass_clear", "metal"]
    clf.coarse_labels = ["plastic", "glass", "metal"]
    clf.f2c_idx = [0, 1, 2]
    clf.gate = {"fine_min_confidence": 0.5, "fine_min_margin": 0.1, "coarse_min_confidence": 0.5}
    clf.taxonomy = {"version": "test"}
    clf.dino_session, clf.dino_weight, clf.ood_protos = None, 0.0, None
    clf.session = _FakeSession(state_logits or [0, 0, 0, 0])
    if state_logits is None:
        clf.state_labels, clf.state_min_conf = None, 1.0
    else:
        clf.state_labels, clf.state_min_conf = V2_TAX["state_labels"], 0.6
    return clf


def test_cnn_state_head_emits_state() -> None:
    clf = _tiny_clf([0.0, 4.0, 0.0, 0.0])
    r = clf.predict(np.zeros((1, 3, 224, 224), dtype=np.float32))
    assert clf.session.asked == ["logits", "state_logits"]
    assert r["state"] == "dirty" and r["state_confidence"] > 0.9
    assert r["display_class"] == "pet"   # 재질 경로 불변


def test_cnn_state_below_gate_is_none() -> None:
    clf = _tiny_clf([0.1, 0.0, 0.0, 0.0])   # 거의 균등 → 임계 0.6 미달
    r = clf.predict(np.zeros((1, 3, 224, 224), dtype=np.float32))
    assert r["state"] is None and r["state_confidence"] < 0.6


def test_v14_model_has_no_state_keys() -> None:
    clf = _tiny_clf(None)
    r = clf.predict(np.zeros((1, 3, 224, 224), dtype=np.float32))
    assert clf.session.asked == ["logits"] and "state" not in r


# ── /predict-hier 상태 우선순위 (VLM > CNN) ─────────────────────────────────

def _hier_stub(monkeypatch: pytest.MonkeyPatch, result: dict, vlm: dict | None) -> None:
    import src.routers.inference as ri
    import src.vlm_fallback as vf
    best = np.zeros((1, 3, 224, 224), dtype=np.float32)
    stub_clf = SimpleNamespace(
        taxonomy={"fine_to_coarse": F2C}, fine_labels=FINE,
        predict=lambda *a, **kw: dict(result))
    monkeypatch.setattr(hi, "get_hier_classifier", lambda: stub_clf)
    monkeypatch.setattr(hi, "predict_rotations", lambda *a, **kw: (dict(result), best))
    monkeypatch.setattr(ri, "non_object_gate", lambda raw: None)
    monkeypatch.setattr(ri, "record_safely", lambda *a, **kw: None)
    monkeypatch.setattr(vf, "get_vlm_fallback",
                        lambda: SimpleNamespace(classify=lambda *a, **kw: vlm))


def _hier(**kw: object) -> dict[str, object]:
    return {"display_level": "fine", "display_class": "paper_other", "coarse_class": "paper",
            "coarse_confidence": 0.95, "fine_class": "paper_other", "fine_confidence": 0.95,
            "fine_margin": 0.9, "coarse_probabilities": {"paper": 0.95}, "fine_top5": [],
            "model_arch": "test", "inference_ms": 1.0, **kw}


def _post_hier(client, image: bytes) -> dict:
    res = client.post("/predict-hier", files={"image": ("x.jpg", image, "image/jpeg")})
    assert res.status_code == 200, res.text
    return res.json()


@needs_v2
def test_endpoint_uses_cnn_state_when_no_vlm(monkeypatch, client, sample_image_bytes) -> None:
    _hier_stub(monkeypatch, _hier(state="dirty", state_confidence=0.9), vlm=None)
    body = _post_hier(client, sample_image_bytes)
    assert (body["state"], body["state_source"]) == ("dirty", "cnn")
    assert body["disposal"]["category"] == "general"   # 오염 종이 → 종량제


@needs_v2
def test_endpoint_vlm_state_overrides_cnn(monkeypatch, client, sample_image_bytes) -> None:
    vlm = {"slug": "paper_other", "item_name": None, "stream": None, "confidence": 0.95,
           "reason": "r", "condition": None, "fine_v2": None, "state": "clean", "mark": None}
    _hier_stub(monkeypatch, _hier(coarse_confidence=0.3, state="dirty"), vlm=vlm)
    body = _post_hier(client, sample_image_bytes)
    assert (body["state"], body["state_source"]) == ("clean", "vlm")
    assert body["disposal"]["category"] == "paper"


@needs_v2
def test_endpoint_cnn_fine_v2_label_feeds_disposal(monkeypatch, client, sample_image_bytes) -> None:
    # CNN v2 가 세부로 metal_can 을 직접 내면 VLM 없이도 캔류
    _hier_stub(monkeypatch, _hier(display_class="metal_can", coarse_class="metal",
                                  fine_class="metal_can"), vlm=None)
    body = _post_hier(client, sample_image_bytes)
    assert body["fine_v2"] == "metal_can" and body["disposal"]["category"] == "can"


# ── 장난감 → trash 고정 (GREENGUIDE_VLM_TOY, 기본 off) ─────────────────────────

@pytest.fixture()
def toy(monkeypatch: pytest.MonkeyPatch):
    def _set(on: bool) -> None:
        monkeypatch.setattr(config, "VLM_TOY", on)
    return _set


@needs_v2
def test_toy_off_keeps_measured_axes_prompt(toy) -> None:
    toy(False)
    p = VlmFallback._axes_prompt()
    assert '"toy"' not in p and "4. 반드시 JSON 만 출력" in p
    toy(True)
    assert '"toy": true|false' in VlmFallback._axes_prompt()


@needs_v2
@pytest.mark.parametrize(("raw", "expect"), [
    ('{"state": "clean", "toy": true}', True),
    ('{"state": "clean", "toy": "true"}', False),   # 문자열은 불인정
    ('{"state": "clean"}', False),
])
def test_parse_axes_toy_flag(toy, raw: str, expect: bool) -> None:
    toy(True)
    assert VlmFallback._parse_axes(raw, "etc", F2C)["toy"] is expect
    toy(False)
    assert "toy" not in VlmFallback._parse_axes(raw, "etc", F2C)


@needs_v2
@pytest.mark.parametrize(("coarse", "fine", "applies"), [
    ("plastic", "plastic_other", True),    # 레고·피규어
    ("etc", "etc", True),
    ("metal", "metal", True),              # 금속 장난감 자동차
    ("plastic", "pet", False),             # 캐릭터 음료병 — 오탐 방지
    ("glass", "glass_clear", False),       # 캐릭터 유리병
    ("paper_pack", "carton", False),       # 캐릭터 우유팩
    ("electronics", "electronics", False),  # 전자 장난감은 폐가전
])
def test_toy_applies_guards_containers(coarse: str, fine: str, applies: bool) -> None:
    assert disposal_v2.toy_applies(coarse, fine) is applies


def _vlm_hit(**kw: object) -> dict[str, object]:
    return {"slug": "plastic_other", "item_name": None, "stream": None, "confidence": 0.95,
            "reason": "r", "condition": None, "fine_v2": None, "state": "clean", "mark": None,
            **kw}


FINE_T = FINE + ["plastic_other", "trash_other"]
F2C_T = {**F2C, "plastic_other": "plastic", "trash_other": "trash"}


def _hier_stub_t(monkeypatch, result: dict, vlm: dict) -> None:
    _hier_stub(monkeypatch, result, vlm)
    stub = hi.get_hier_classifier()
    stub.fine_labels, stub.taxonomy = FINE_T, {"fine_to_coarse": F2C_T}


@needs_v2
def test_endpoint_toy_fixes_trash(monkeypatch, client, sample_image_bytes) -> None:
    _hier_stub_t(monkeypatch, _hier(coarse_confidence=0.3), _vlm_hit(toy=True))
    body = _post_hier(client, sample_image_bytes)
    assert (body["coarse_class"], body["display_class"]) == ("trash", "trash_other")
    assert body["disposal"]["category"] == "general" and "toy" in body["disposal"]["notes"]


@needs_v2
def test_endpoint_toy_flag_on_bottle_ignored(monkeypatch, client, sample_image_bytes) -> None:
    _hier_stub_t(monkeypatch, _hier(coarse_confidence=0.3), _vlm_hit(slug="pet", toy=True))
    body = _post_hier(client, sample_image_bytes)
    assert body["coarse_class"] == "plastic" and body["disposal"]["category"] == "pet"


@needs_v2
def test_endpoint_no_toy_key_unchanged(monkeypatch, client, sample_image_bytes) -> None:
    # GREENGUIDE_VLM_TOY off → 파서가 toy 키를 안 만든다 → 재질 그대로
    _hier_stub_t(monkeypatch, _hier(coarse_confidence=0.3), _vlm_hit())
    body = _post_hier(client, sample_image_bytes)
    assert body["coarse_class"] == "plastic" and body["disposal"]["category"] == "plastic"


# ── 중재 R1·1회차 재질 규칙 (11-04 gate A 오답 분석, 둘 다 기본 꺼짐) ─────────────

def _low_vlm(slug: str) -> dict[str, object]:
    return _vlm_hit(slug=slug, confidence=0.6)   # VLM_MIN_CONF 0.8 미달


@pytest.mark.parametrize(("on", "slug", "adopted"), [
    (True, "plastic_other", True),    # CNN 1위와 같은 재질 → 확신 무관 채택
    (False, "plastic_other", False),  # 플래그 꺼짐 → 기존 0.8 임계
    (True, "trash_other", False),     # 재질이 다르면 R1 미적용
])
def test_agree_accept(monkeypatch, client, sample_image_bytes, on: bool, slug: str,
                      adopted: bool) -> None:
    monkeypatch.setattr(config, "VLM_AGREE_ACCEPT", on)
    _hier_stub_t(monkeypatch, _hier(display_level="reject", display_class="etc",
                                    coarse_class="plastic", coarse_confidence=0.5,
                                    fine_class=None), _low_vlm(slug))
    body = _post_hier(client, sample_image_bytes)
    assert (body["display_class"] == slug) is adopted
    assert body["display_level"] == ("fine" if adopted else "reject")


@pytest.mark.parametrize("on", [True, False])
def test_material_rule_prompt(monkeypatch, on: bool) -> None:
    monkeypatch.setattr(config, "VLM_MATERIAL_RULE", on)
    p = VlmFallback._taxonomy_prompt(FINE, F2C)
    assert ("4-1. slug 는 물체의 재질" in p) is on
    assert "반드시 JSON 만 출력" in p   # 출력 규칙은 그대로
    if on:
        assert "영수증" in p and "장난감" in p


@pytest.mark.parametrize("model,expected", [
    ("claude-sonnet-5-5", 0.6), ("claude-opus-5-5", 0.8)])
def test_sonnet_min_conf(monkeypatch, model, expected) -> None:
    import importlib

    from src.core import config
    monkeypatch.setenv("GREENGUIDE_VLM_SONNET_MIN_CONF", "0.6")
    monkeypatch.setenv("VLM_MODEL", model)
    monkeypatch.delenv("VLM_MIN_CONF", raising=False)
    try:
        assert importlib.reload(config).VLM_MIN_CONF == expected
    finally:
        monkeypatch.undo()
        importlib.reload(config)
