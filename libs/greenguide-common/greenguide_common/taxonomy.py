"""계층 taxonomy — 대분류(coarse) → 세부품목(fine) 정의의 단일 진실.

GREENGUIDE_BLUEPRINT.md §1 의 코드화. 모델의 fine 출력공간과
대분류 롤업(P(coarse) = Σ P(fine children)) 매핑을 제공한다.

감독(supervision) 매핑 규칙:
- 기존 manifest 라벨이 특정 fine 하나에 대응되면 → fine 감독 (예: cardboard)
- 여러 fine 이 섞여 있으면 → coarse 감독 (예: glass 는 색상 미상 → 롤업 CE)
- fine-staging 디렉터리 라벨은 전부 fine 감독

이 모듈은 각 프로젝트의 config.CLASS_LABELS (flat, manifest 기반 동적) 를 건드리지 않는다.
greenguide-classifier(학습)·waste-api(서빙)·greenguide-preprocessor 가 모두 여기서 import 한다.
"""
from __future__ import annotations

# Kaggle 시드 데이터셋의 6클래스 — flat 파이프라인 fallback / 데이터셋 존재 검증 전용.
LEGACY_LABELS: tuple[str, ...] = ("cardboard", "glass", "metal", "paper", "plastic", "trash")

# ── 계층 정의: coarse slug → fine children (순서 고정 = 재현성) ──────────────
TAXONOMY: dict[str, tuple[str, ...]] = {
    "paper":      ("paper_other", "cardboard"),
    "paper_pack": ("carton", "paper_cup"),
    "glass":      ("glass_brown", "glass_green", "glass_clear",
                   "glass_deposit", "glass_etc"),
    "metal":      ("metal",),
    "plastic":    ("pet", "plastic_other"),
    "vinyl":      ("vinyl_clean", "vinyl_dirty"),
    "styrofoam":  ("styrofoam_white", "styrofoam_color", "styrofoam_dirty"),
    "clothes":    ("clothes",),
    "food_waste": ("food_waste",),
    "electronics": ("electronics",),
    "hazardous":  ("battery",),               # 향후: fluorescent, medicine
    # 일반쓰레기 — 기존 DB slug(trash) 와 정렬 (blueprint 의 'general' 대응)
    "trash":      ("trash_other", "light_bulb"),  # light_bulb ⚠️전구≠형광등→일반
    # 내부 신호 (사용자 비노출)
    "etc":        ("etc",),
    "non_object": ("non_object",),
}

COARSE_LABELS: tuple[str, ...] = tuple(TAXONOMY.keys())
FINE_LABELS: tuple[str, ...] = tuple(f for children in TAXONOMY.values() for f in children)

COARSE_TO_INDEX: dict[str, int] = {c: i for i, c in enumerate(COARSE_LABELS)}
FINE_TO_INDEX: dict[str, int] = {f: i for i, f in enumerate(FINE_LABELS)}
FINE_TO_COARSE: dict[str, str] = {
    f: c for c, children in TAXONOMY.items() for f in children
}
# fine index → coarse index (롤업 텐서 구성용)
FINE_IDX_TO_COARSE_IDX: tuple[int, ...] = tuple(
    COARSE_TO_INDEX[FINE_TO_COARSE[f]] for f in FINE_LABELS
)

NUM_FINE: int = len(FINE_LABELS)
NUM_COARSE: int = len(COARSE_LABELS)

# ── 감독 매핑 ────────────────────────────────────────────────────────────────
# 기존 manifest 라벨 → ("fine"|"coarse", 대상 slug)
LEGACY_LABEL_SUPERVISION: dict[str, tuple[str, str]] = {
    "cardboard":  ("fine", "cardboard"),
    "paper":      ("fine", "paper_other"),   # cardboard 가 별도였으므로 나머지=paper_other
    "glass":      ("coarse", "glass"),        # 색상 미상 → 롤업 감독
    "metal":      ("fine", "metal"),
    "plastic":    ("coarse", "plastic"),      # PET+기타 혼재 → 롤업 감독
    "trash":      ("fine", "trash_other"),
    "vinyl":      ("fine", "vinyl_clean"),    # 기존 수집분은 분리배출 가능 비닐 위주
    "styrofoam":  ("coarse", "styrofoam"),    # 흰/컬러/오염 미상 → 롤업 감독
    "clothes":    ("fine", "clothes"),
    "food_waste": ("fine", "food_waste"),
    "electronics": ("fine", "electronics"),
    "etc":        ("fine", "etc"),
    "non_object": ("fine", "non_object"),
}

# fine-staging 디렉터리명 → ("fine"|"coarse", 대상 slug)
STAGING_DIR_SUPERVISION: dict[str, tuple[str, str]] = {
    "battery":         ("fine", "battery"),
    "paper_pack":      ("fine", "carton"),
    "paper_cup":       ("fine", "paper_cup"),
    "glass_brown":     ("fine", "glass_brown"),
    "glass_green":     ("fine", "glass_green"),
    "glass_clear":     ("fine", "glass_clear"),
    "glass_deposit":   ("fine", "glass_deposit"),
    "glass_etc":       ("fine", "glass_etc"),
    "pet":             ("fine", "pet"),
    "styrofoam_white": ("fine", "styrofoam_white"),
    "styrofoam_color": ("fine", "styrofoam_color"),
    "styrofoam_dirty": ("fine", "styrofoam_dirty"),
    "light_bulb":      ("fine", "light_bulb"),
    "vinyl_dirty":     ("fine", "vinyl_dirty"),
    "metal_boost":     ("fine", "metal"),
    "paper_boost":     ("fine", "paper_other"),
    "plastic_boost":   ("coarse", "plastic"),  # 오염 플라스틱: PET 여부 미상
    "electronics_boost": ("fine", "electronics"),
    "cardboard_boost": ("fine", "cardboard"),      # 140 포장상자 수혈  # 140 개인기기(폰 등) 실사용 보강
    # 실내 배경 전용 네거티브 — 합성 실험(v9)용이었으나 접근 폐기됨. 매핑 잔존 무해.
    "non_object_boost": ("fine", "non_object"),
    # TACO (CC BY 4.0, 실환경 in-context) — 합성 폐기 후 실데이터 노선 (2026-07-20).
    # 색·오염도 미상 카테고리는 coarse 감독 (glass/vinyl/styrofoam)
    "taco_glass":     ("coarse", "glass"),
    "taco_vinyl":     ("coarse", "vinyl"),
    "taco_styrofoam": ("coarse", "styrofoam"),
    "taco_trash":     ("fine", "trash_other"),
    "taco_food":      ("fine", "food_waste"),
}


# ── 안내-동일 그룹 (guidance groups) ─────────────────────────────────────────
# 같은 그룹 내 형제 혼동은 사용자 배출 안내가 동일 → guidance-safe 지표에서 정답 처리.
# 근거: blueprint 원칙 "배출법이 실제로 달라질 때만 쪼갠다" + 혼동 분석
# (carton↔paper_cup 상호 199건, glass 색상 간 14% — 전부 같은 수거함 안내).
# 주의: glass_deposit(보증금 반환)은 안내가 달라 색상 그룹에서 제외.
GUIDANCE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"carton", "paper_cup"}),                                  # 종이팩 수거함
    frozenset({"glass_brown", "glass_green", "glass_clear", "glass_etc"}),  # 유리병 수거함
)

_GUIDANCE_GROUP_OF: dict[str, frozenset[str]] = {
    slug: group for group in GUIDANCE_GROUPS for slug in group
}


def same_guidance(a: str, b: str) -> bool:
    """두 fine slug 가 동일한 배출 안내를 공유하는가 (자기 자신 포함)."""
    if a == b:
        return True
    return _GUIDANCE_GROUP_OF.get(a) is not None and _GUIDANCE_GROUP_OF.get(a) == _GUIDANCE_GROUP_OF.get(b)


def supervision_index(kind: str, slug: str) -> int:
    """감독 대상 slug → 해당 공간의 정수 인덱스."""
    if kind == "fine":
        return FINE_TO_INDEX[slug]
    if kind == "coarse":
        return COARSE_TO_INDEX[slug]
    raise ValueError(f"unknown supervision kind={kind!r}")


def rollup_fine_probs(fine_probs):
    """(B, NUM_FINE) 확률 → (B, NUM_COARSE) 롤업 확률. torch/numpy 겸용."""
    import numpy as np
    try:
        import torch
        if isinstance(fine_probs, torch.Tensor):
            out = torch.zeros(*fine_probs.shape[:-1], NUM_COARSE,
                              dtype=fine_probs.dtype, device=fine_probs.device)
            for fi, ci in enumerate(FINE_IDX_TO_COARSE_IDX):
                out[..., ci] += fine_probs[..., fi]
            return out
    except ImportError:
        pass
    arr = np.asarray(fine_probs)
    out = np.zeros((*arr.shape[:-1], NUM_COARSE), dtype=arr.dtype)
    for fi, ci in enumerate(FINE_IDX_TO_COARSE_IDX):
        out[..., ci] += arr[..., fi]
    return out
