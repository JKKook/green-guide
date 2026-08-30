#!/usr/bin/env python3
"""계층 세부품목 활성화 승격 — v2 평가의 ready 판정을 waste_classes 에 반영.

전제: migrations/008_hierarchy.sql 이 Supabase 에 적용되어
      세부품목 행들이 존재해야 함 (level=2, active=false 시드 상태).

동작 (idempotent):
- evaluation.json 의 fine_activation 에서 ready=true 인 세부품목
  → trained_in_model=true, active=true
- ready=false (carton/paper_cup/glass_clear 등)
  → trained_in_model=true 만 (모델은 출력하지만 노출은 롤업 유지)
- 신규 대분류 (paper_pack, hazardous)
  → 자식이 하나라도 학습됐으면 trained_in_model=true, active=true

사용:
    .venv/bin/python scripts/apply_hier_activation.py --dry-run
    .venv/bin/python scripts/apply_hier_activation.py
"""
from __future__ import annotations

import argparse
import json

from _base import PROJECT_ROOT

EVAL_PATH = PROJECT_ROOT / "outputs" / "logs" / "cnn_hier" / "evaluation.json"

# legacy 라벨과 이름이 같은 fine 은 이미 waste_classes 에 level=1 행으로 존재
# (metal/clothes/...) — 이 스크립트는 신규 level=2 행만 다룬다.
NEW_FINE_SLUGS = {
    "carton", "paper_cup",
    "glass_brown", "glass_green", "glass_clear", "glass_deposit", "glass_etc",
    "pet", "plastic_other",
    "vinyl_clean", "vinyl_dirty",
    "styrofoam_white", "styrofoam_color", "styrofoam_dirty",
    "battery", "light_bulb",
}
NEW_COARSE_SLUGS = ("paper_pack", "hazardous")


def _client():
    from waste_common.supabase import get_client

    return get_client()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ev = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    activation = ev["fine_activation"]

    plan: list[tuple[str, dict]] = []
    for slug in sorted(NEW_FINE_SLUGS):
        a = activation.get(slug, {})
        ready = bool(a.get("ready"))
        patch = {"trained_in_model": True, "active": ready}
        plan.append((slug, patch))
    ready_any = any(p["active"] for s, p in plan)
    for slug in NEW_COARSE_SLUGS:
        plan.append((slug, {"trained_in_model": True, "active": ready_any}))

    print(f"{'[DRY-RUN] ' if args.dry_run else ''}승격 계획 (evaluation: "
          f"coarse {ev['coarse_accuracy']:.4f} / fine {ev['fine_accuracy_on_fine_items']:.4f}):")
    for slug, patch in plan:
        f1 = activation.get(slug, {}).get("f1", "-")
        print(f"  {slug:18} trained=True active={patch['active']!s:5} (f1={f1})")

    if args.dry_run:
        return

    sb = _client()
    updated, missing = 0, []
    for slug, patch in plan:
        res = sb.table("waste_classes").update(patch).eq("slug", slug).execute()
        if res.data:
            updated += 1
        else:
            missing.append(slug)
    print(f"\n갱신 {updated}건 완료.")
    if missing:
        print(f"! 행 없음 (migration 008 미적용?): {missing}")


if __name__ == "__main__":
    main()
