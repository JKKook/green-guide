"""스트림 닫힌 목록 + VLM 폴백 파싱·검증 단위 테스트 (네트워크 무관)."""
from __future__ import annotations

import pytest

from src.streams import STREAMS, is_valid_stream, prompt_lines, to_api_dict
from src.vlm_fallback import VlmFallback

FINE = ["pet", "glass_clear", "etc", "non_object"]


# ── streams 레지스트리 ──────────────────────────────────────────────────────

def test_streams_registry_core_slugs_exist() -> None:
    for slug in ("general_trash", "incombustible_bag", "bulky_waste",
                 "collection_battery", "collection_fluorescent",
                 "collection_medicine", "retailer_takeback", "deposit_return"):
        assert is_valid_stream(slug), slug


def test_hazardous_destinations_are_distinct_streams() -> None:
    # 건전지·형광등·의약품은 물리적 목적지가 달라 하나로 묶으면 안 됨
    assert len({"collection_battery", "collection_fluorescent",
                "collection_medicine"} & set(STREAMS)) == 3


def test_prompt_lines_exclude_reserved() -> None:
    lines = prompt_lines()
    assert "farm_waste" not in lines
    assert "bulky_waste" in lines


def test_to_api_dict_shape() -> None:
    d = to_api_dict("bulky_waste")
    assert d is not None
    assert d["display_name"] == "대형폐기물 신고 배출"
    assert isinstance(d["how_to"], list)
    assert to_api_dict("no_such_stream") is None


# ── VLM 응답 파싱·검증 ─────────────────────────────────────────────────────

def test_parse_fine_slug_path() -> None:
    out = VlmFallback._parse_response(
        '{"slug": "pet", "confidence": 0.9, "reason": "투명 음료병"}', FINE)
    assert out == {"slug": "pet", "item_name": None, "stream": None,
                   "confidence": 0.9, "reason": "투명 음료병", "condition": None}


def test_parse_generated_item_path() -> None:
    out = VlmFallback._parse_response(
        '{"slug": null, "item_name": "소파", "stream": "bulky_waste", '
        '"condition": null, "confidence": 0.85, "reason": "3인용 패브릭 소파"}',
        FINE)
    assert out is not None
    assert out["slug"] is None
    assert out["item_name"] == "소파"
    assert out["stream"] == "bulky_waste"


def test_parse_rejects_invented_stream() -> None:
    # VLM 이 목록 밖 스트림을 발명하면 무시 (닫힌 목록의 요체)
    out = VlmFallback._parse_response(
        '{"slug": null, "item_name": "소파", "stream": "sofa_bin", '
        '"confidence": 0.9, "reason": "x"}', FINE)
    assert out is None


def test_parse_rejects_unknown_fine_slug_without_item() -> None:
    out = VlmFallback._parse_response(
        '{"slug": "sofa", "confidence": 0.9, "reason": "x"}', FINE)
    assert out is None


def test_parse_item_with_condition_and_fence() -> None:
    out = VlmFallback._parse_response(
        '```json\n{"slug": null, "item_name": "화분", '
        '"stream": "incombustible_bag", '
        '"condition": "도자기면 불연 마대, 플라스틱이면 재활용", '
        '"confidence": 0.7, "reason": "도자기 화분"}\n```', FINE)
    assert out is not None
    assert out["stream"] == "incombustible_bag"
    assert "도자기" in out["condition"]


def test_parse_invalid_json_raises() -> None:
    with pytest.raises(Exception):
        VlmFallback._parse_response("정답은 소파입니다", FINE)


def test_prompt_contains_both_tiers() -> None:
    p = VlmFallback._taxonomy_prompt(FINE, {"pet": "plastic"})
    assert "pet" in p and "bulky_waste" in p
    assert "item_name" in p and "stream" in p
