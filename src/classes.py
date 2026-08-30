"""waste_classes 레지스트리 — Supabase 에서 동적 로드.

이 모듈이 단일 진실. config.py 의 하드코딩 CLASS_LABELS 는 fallback 으로만 유지.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.uploads import _client as _supabase_client
from src.core.log import get_logger

log = get_logger(__name__)


@dataclass
class WasteClass:
    slug: str
    sort_order: int
    display_name: str
    summary: str | None
    bin: str | None
    how_to: list[str]
    caution: list[str]
    color_hex: str | None
    icon_name: str | None
    trained_in_model: bool
    active: bool
    # 계층 (migration 008) — 구 스키마 행에는 없을 수 있어 기본값 유지
    level: int = 1
    parent_slug: str | None = None
    is_negative_guidance: bool = False

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "WasteClass":
        return cls(
            slug=row["slug"],
            sort_order=row.get("sort_order") or 100,
            display_name=row["display_name"],
            summary=row.get("summary"),
            bin=row.get("bin"),
            how_to=list(row.get("how_to") or []),
            caution=list(row.get("caution") or []),
            color_hex=row.get("color_hex"),
            icon_name=row.get("icon_name"),
            trained_in_model=bool(row.get("trained_in_model")),
            active=bool(row.get("active", True)),
            level=int(row.get("level") or 1),
            parent_slug=row.get("parent_slug"),
            is_negative_guidance=bool(row.get("is_negative_guidance")),
        )

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "sort_order": self.sort_order,
            "display_name": self.display_name,
            "summary": self.summary,
            "bin": self.bin,
            "how_to": self.how_to,
            "caution": self.caution,
            "color_hex": self.color_hex,
            "icon_name": self.icon_name,
            "trained_in_model": self.trained_in_model,
            "level": self.level,
            "parent_slug": self.parent_slug,
            "is_negative_guidance": self.is_negative_guidance,
        }


class ClassRegistry:
    """애플리케이션 라이프타임 동안 한 번 로드 후 메모리에 캐싱.
    재로드 필요 시 reload() 호출."""

    _classes: list[WasteClass] | None = None

    @classmethod
    def load(cls) -> list[WasteClass]:
        if cls._classes is None:
            cls.reload()
        return cls._classes or []

    @classmethod
    def reload(cls) -> None:
        try:
            client = _supabase_client()
            res = (
                client.table("waste_classes")
                .select("*")
                .eq("active", True)
                .order("sort_order")
                .execute()
            )
            cls._classes = [WasteClass.from_row(r) for r in (res.data or [])]
        except Exception as exc:  # noqa: BLE001
            # Supabase 불가(쿼터 제한·네트워크 등) — taxonomy 사이드카 기반
            # 오프라인 폴백. 피드백 라벨 검증·/labels 가 인프라 장애에 죽지
            # 않게 한다 (2026-07-21 제한 사태 중 도입).
            log.info(f"Supabase 불가 → taxonomy 폴백: {str(exc)[:80]}")
            cls._classes = cls._fallback_classes()

    @classmethod
    def _fallback_classes(cls) -> list[WasteClass]:
        """오프라인 폴백 — 서빙 taxonomy.json 의 대분류+세부를 최소 메타로."""
        import json
        from pathlib import Path
        sidecar = Path(__file__).resolve().parent.parent / "models" / "taxonomy.json"
        out: list[WasteClass] = []
        try:
            tax = json.loads(sidecar.read_text(encoding="utf-8"))
            f2c = tax.get("fine_to_coarse", {})
            for i, slug in enumerate(tax.get("coarse_labels", [])):
                if slug == "non_object":
                    continue
                out.append(WasteClass.from_row({
                    "slug": slug, "sort_order": i * 10,
                    "display_name": slug, "trained_in_model": True,
                    "active": True, "level": 1}))
            for i, slug in enumerate(tax.get("fine_labels", [])):
                if slug in ("non_object", "etc"):
                    continue
                out.append(WasteClass.from_row({
                    "slug": slug, "sort_order": 500 + i,
                    "display_name": slug, "trained_in_model": True,
                    "active": True, "level": 2,
                    "parent_slug": f2c.get(slug)}))
        except Exception as exc:  # noqa: BLE001
            log.info(f"폴백 로드 실패: {exc}")
        return out

    @classmethod
    def all_slugs(cls) -> list[str]:
        return [c.slug for c in cls.load()]

    @classmethod
    def color_map(cls) -> dict[str, str | None]:
        """slug → color_hex (다중재질 빗금 색상용)."""
        return {c.slug: c.color_hex for c in cls.load()}

    @classmethod
    def trained_slugs(cls) -> list[str]:
        """현재 ONNX 모델이 출력하는 클래스 (sort_order 순)."""
        return [c.slug for c in cls.load() if c.trained_in_model]

    @classmethod
    def is_valid_label(cls, slug: str) -> bool:
        return slug in cls.all_slugs()
