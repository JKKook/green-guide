"""분류 체계 v2 어댑터 — libs/greenguide-common `greenguide_common.disposal` 규칙표를 서빙에 연결.

규칙표는 복사하지 않고 공통 패키지를 import 한다(정본 = docs/plans/TAXONOMY_V2.md, 결정 32).
패키지가 설치돼 있지 않으면 AVAILABLE=False — v2 축(fine_v2·state·mark·disposal)만 꺼지고
기존 응답·VLM 프롬프트는 v1 그대로 동작한다(fail-open, 하위호환).
"""
from __future__ import annotations

from typing import Any

from src.core.log import get_logger

log = get_logger(__name__)

try:
    from greenguide_common import disposal as _rules
    AVAILABLE = True
    STATES: tuple[str, ...] = _rules.STATES
    MARKS: tuple[str, ...] = _rules.MARKS
    FINE_V2_SPLITS: dict[str, tuple[str, ...]] = _rules.FINE_V2_SPLITS
except ImportError:  # 공통 패키지 미설치 — v2 축 비활성
    _rules = None
    AVAILABLE = False
    STATES, MARKS, FINE_V2_SPLITS = (), (), {}
    log.info("greenguide_common.disposal 없음 — 분류 체계 v2 축 비활성")


def resolve(coarse: str, fine: str | None, state: str | None,
            mark: str | None) -> dict[str, Any] | None:
    """재질·상태·마크 → {category, notes, source} | None(판정 보류·비활성)."""
    if _rules is None:
        return None
    try:
        d = _rules.resolve(coarse, fine, state, mark)
    except ValueError as exc:   # 파서가 검증하므로 정상 경로에선 오지 않는다
        log.warning(f"disposal 규칙표 입력 거부: {exc}")
        return None
    if d is None:
        return None
    return {"category": d.category, "notes": list(d.notes), "source": d.source}


# 장난감 → 재질 trash 고정에서 제외하는 대분류 — 전자 장난감은 폐가전(명세), 병·팩·유리 용기는
# 캐릭터 인쇄가 있어도 용기라 오탐을 막는다. 세부 pet(음료병)도 같은 이유로 제외.
_TOY_EXEMPT_COARSE = frozenset({"electronics", "hazardous", "food_waste", "glass", "paper_pack"})
_TOY_EXEMPT_FINE = frozenset({"pet"})


def toy_applies(coarse: str, fine: str | None) -> bool:
    """VLM 이 toy=true 라고 답했을 때 재질을 trash 로 고정할지 (TAXONOMY_V2 §3 장난감)."""
    return coarse not in _TOY_EXEMPT_COARSE and fine not in _TOY_EXEMPT_FINE
