import pytest

from greenguide_common.disposal import DISPOSALS, STATES, resolve
from greenguide_common.taxonomy import TAXONOMY


@pytest.mark.parametrize("coarse,fine,state,expected", [
    ("paper", "cardboard", "dirty", "general"),          # 기름 묻은 치킨 상자
    ("paper", "paper_other", "composite", "general"),    # 영수증·코팅지
    ("glass", "glass_brown", "broken", "general"),       # 깨진 병
    ("glass", "glass_clear", "clean", "glass"),
    ("metal", "metal_can", "clean", "can"),
    ("metal", "metal_scrap", "clean", "scrap_metal"),
    ("food_waste", "food_inedible", "clean", "general"),  # 조개껍데기
    ("food_waste", "food_waste", "dirty", "food"),
    ("plastic", "plastic_other", "composite", "general"),  # 칫솔
    ("plastic", "pet", "clean", "pet"),
    ("hazardous", "battery", "clean", "hazardous"),       # 보조배터리
    ("clothes", "clothes", "dirty", "clothes_bin"),
])
def test_rules(coarse, fine, state, expected):
    assert resolve(coarse, fine, state).category == expected


def test_mark_overrides_rule():
    d = resolve("paper", "paper_other", "dirty", mark="general_pack")
    assert (d.category, d.source) == ("general_pack", "mark")


def test_undecidable():
    assert resolve("non_object") is None
    assert resolve("metal", "metal") is None      # v14 세부 — 캔/고철 미확정
    assert resolve("food_waste", None) is None


def test_every_v14_coarse_resolves_or_defers():
    for coarse, fines in TAXONOMY.items():
        for fine in fines:
            for st in STATES:
                d = resolve(coarse, fine, st)
                assert d is None or d.category in DISPOSALS


def test_bad_state():
    with pytest.raises(ValueError):
        resolve("paper", None, "wet")
