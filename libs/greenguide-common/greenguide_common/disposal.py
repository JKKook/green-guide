"""분류 체계 v2 — 재질 × 상태 → 분리배출 카테고리 규칙표 (docs/plans/TAXONOMY_V2.md §1~§3, 결정 32).

배출 카테고리는 모델이 예측하지 않고 이 규칙표로 도출한다. 제품의 분리배출 마크를 읽었으면 마크가 우선한다.
`taxonomy.TAXONOMY`(v14 모델 출력 공간)는 건드리지 않는다 — v2 세부(metal_can 등)는 여기서만 정의한다.
"""
from __future__ import annotations

from typing import NamedTuple

STATES: tuple[str, ...] = ("clean", "dirty", "broken", "composite")

# v2 에서 갈라지는 세부. v14 의 'metal'·'food_waste' 세부는 미확정(None)으로 들어올 수 있다.
FINE_V2_SPLITS: dict[str, tuple[str, ...]] = {
    "metal": ("metal_can", "metal_scrap"),
    "food_waste": ("food_waste", "food_inedible"),
}

# 분리배출 축 15 = 표시 마크 9 + 배출 경로 6 (+ 고철은 마크가 아니라 지역 수거함 경로)
MARKS: tuple[str, ...] = ("paper", "general_pack", "sterile_pack", "can", "glass",
                          "pet", "plastic", "vinyl", "styrofoam")
ROUTES: tuple[str, ...] = ("clothes_bin", "food", "e_waste", "hazardous", "bulky", "general")
SCRAP = "scrap_metal"
DISPOSALS: tuple[str, ...] = MARKS + ROUTES + (SCRAP,)


class Disposal(NamedTuple):
    category: str          # DISPOSALS 중 하나
    notes: tuple[str, ...]  # 사용자 안내 문구 키(앱·서버가 문장으로 바꾼다)
    source: str            # "mark" | "rule"


# (재질 키, 상태) → (배출, 안내). 재질 키는 v2 세부가 있으면 세부, 없으면 대분류.
_RULES: dict[str, dict[str, tuple[str, tuple[str, ...]]]] = {
    "paper":       {"clean": ("paper", ()), "dirty": ("general", ("contaminated",)),
                    "broken": ("paper", ()), "composite": ("general", ("non_recyclable_paper",))},
    "paper_pack":  {"clean": ("general_pack", ()), "dirty": ("general", ("contaminated",)),
                    "broken": ("general_pack", ()), "composite": ("general", ())},
    "glass":       {"clean": ("glass", ()), "dirty": ("glass", ("rinse",)),
                    "broken": ("general", ("wrap_broken_glass",)), "composite": ("general", ("not_bottle_glass",))},
    "metal_can":   {"clean": ("can", ()), "dirty": ("can", ("rinse",)),
                    "broken": ("can", ()), "composite": ("can", ("separate_parts",))},
    "metal_scrap": {s: (SCRAP, ()) for s in STATES},
    "plastic":     {"clean": ("plastic", ()), "dirty": ("plastic", ("rinse_or_general",)),
                    "broken": ("plastic", ()), "composite": ("general", ("mixed_material",))},
    "vinyl":       {"clean": ("vinyl", ()), "dirty": ("general", ("contaminated",)),
                    "broken": ("vinyl", ()), "composite": ("general", ())},
    "styrofoam":   {"clean": ("styrofoam", ()), "dirty": ("general", ("contaminated",)),
                    "broken": ("styrofoam", ()), "composite": ("general", ())},
    "food_waste":  {s: ("food", ()) for s in STATES},
    "food_inedible": {s: ("general", ("inedible_food",)) for s in STATES},
}
# 상태와 무관하게 경로가 고정인 재질
_FIXED: dict[str, str] = {
    "clothes": "clothes_bin", "electronics": "e_waste", "hazardous": "hazardous",
    "trash": "general", "etc": "general",
}
# 세부 단위 예외(재질 규칙보다 우선)
_FINE_OVERRIDES: dict[str, tuple[str, tuple[str, ...]]] = {
    "pet": ("pet", ()),
    "light_bulb": ("general", ()),  # 백열·LED 전구는 종량제, 형광등은 hazardous 로 라벨
}


def material_key(coarse: str, fine: str | None) -> str | None:
    """규칙표 조회 키. v2 분기 재질인데 세부를 모르면 None(판정 보류)."""
    if coarse in FINE_V2_SPLITS:
        return fine if fine in FINE_V2_SPLITS[coarse] else None
    return coarse


def resolve(coarse: str, fine: str | None = None, state: str | None = None,
            mark: str | None = None) -> Disposal | None:
    """배출 카테고리 도출. 판정할 수 없으면 None(non_object, 세부 미상 등)."""
    if mark is not None:
        if mark not in MARKS:
            raise ValueError(f"unknown mark: {mark}")
        notes = ("rinse",) if state == "dirty" else ()
        return Disposal(mark, notes, "mark")
    if coarse == "non_object":
        return None
    if state is not None and state not in STATES:
        raise ValueError(f"unknown state: {state}")
    st = state or "clean"
    if fine in _FINE_OVERRIDES and st == "clean":
        cat, notes = _FINE_OVERRIDES[fine]
        return Disposal(cat, notes, "rule")
    if coarse in _FIXED:
        notes = ("wrap_broken",) if st == "broken" and coarse == "hazardous" else ()
        return Disposal(_FIXED[coarse], notes, "rule")
    key = material_key(coarse, fine)
    if key is None or key not in _RULES:
        return None
    cat, notes = _RULES[key][st]
    return Disposal(cat, notes, "rule")
