#!/usr/bin/env python3
"""신규 Supabase 프로젝트 데이터 부트스트랩 — 이사(옵션 C) 자동화.

선행 조건:
  1) 사용자가 새 프로젝트 생성 + SQL Editor 에서
     migrations/_bootstrap_new_project.sql 실행 (스키마+시드)
  2) waste-api/.env 의 SUPABASE_URL / SUPABASE_KEY 를 새 프로젝트 값으로 교체

이 스크립트가 하는 일:
  [1] 버킷 생성: user-uploads(공개), models(공개), raw-images(비공개)
  [2] 로컬 캐시된 피드백 51장 업로드 + user_uploads 행 복원
      (라벨 = raw/<label>/user_<id>.jpg 디렉토리명, feedback_status=corrected)
  [3] 지역 배출 규정 재적재 (load_region_rules 재사용 — 공공 API에서 933행)
  [4] 검증 출력

이후 수동 단계 (스크립트가 마지막에 안내):
  - waste-classifier publish_hier_version.py --apply (모델 레지스트리 재발행)
  - 계층 활성화 상태 복원: apply_hier_activation.py
  - HF Spaces 시크릿(SUPABASE_URL/KEY) 교체 → 재빌드

실행: .venv/bin/python scripts/bootstrap_new_supabase.py
"""
from __future__ import annotations

import mimetypes
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

RAW = Path("/Users/ethan/practice/waste/waste-preprocessor/data/raw/garbage-classification")


def main() -> None:
    from supabase import create_client
    url, key = os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"]
    sb = create_client(url, key)
    print(f"[bootstrap] 대상: {url}")

    # ── [0] 스키마 확인 ────────────────────────────────────────────────────
    try:
        n = len(sb.table("waste_classes").select("slug").execute().data or [])
        print(f"[0] waste_classes {n}행 — 스키마 OK")
        if n < 6:
            raise SystemExit("시드가 비어 있음 — _bootstrap_new_project.sql 먼저 실행하세요")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"스키마 미준비: {exc}\n→ SQL Editor 에서 "
                         "migrations/_bootstrap_new_project.sql 실행 후 재시도") from exc

    # ── [1] 버킷 ──────────────────────────────────────────────────────────
    for name, public in (("user-uploads", True), ("models", True), ("raw-images", False)):
        try:
            sb.storage.create_bucket(name, options={"public": public})
            print(f"[1] 버킷 생성: {name} (public={public})")
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "already exists" in msg or "Duplicate" in msg:
                print(f"[1] 버킷 존재: {name}")
            else:
                raise

    # ── [2] 피드백 51장 복원 ──────────────────────────────────────────────
    files = sorted(RAW.glob("*/user_*.jpg")) + sorted(RAW.glob("*/user_*.png"))
    restored = skipped = 0
    for f in files:
        label = f.parent.name
        upload_id = f.stem.replace("user_", "")
        ext = f.suffix
        storage_path = f"{label}/{upload_id}{ext}"
        ct = mimetypes.guess_type(f.name)[0] or "image/jpeg"
        try:
            sb.storage.from_("user-uploads").upload(
                path=storage_path, file=f.read_bytes(),
                file_options={"upsert": "true", "content-type": ct})
            image_url = sb.storage.from_("user-uploads").get_public_url(storage_path)
            sb.table("user_uploads").upsert({
                "id": upload_id,
                "image_url": image_url,
                "storage_path": storage_path,
                "predicted_class": label,      # 원 예측값은 소실 — 라벨로 대체 (평가엔 feedback_label 만 사용)
                "predicted_confidence": None,
                "model_arch": "restored_from_local_cache",
                "feedback_status": "corrected",
                "feedback_label": label,
            }, on_conflict="id").execute()
            restored += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {f.name}: {str(exc)[:80]}")
            skipped += 1
    print(f"[2] 피드백 복원 {restored}건 (실패 {skipped})")

    # ── [3] 지역 규정 재적재 ──────────────────────────────────────────────
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "load_region_rules.py")],
        capture_output=True, text=True, timeout=1800)
    tail = (r.stdout or r.stderr).strip().splitlines()[-1:]
    print(f"[3] 지역 규정: {' '.join(tail)} (rc={r.returncode})")

    # ── [4] 검증 ──────────────────────────────────────────────────────────
    fb = sb.table("user_uploads").select("id").in_(
        "feedback_status", ["confirmed", "corrected"]).execute().data or []
    rr = sb.table("region_waste_rules").select("id").limit(1).execute()
    print(f"[4] 검증 — 피드백 {len(fb)}건, region_rules 접근 OK")

    print("\n다음 수동 단계:")
    print("  1) waste-classifier: .venv/bin/python scripts/apply_hier_activation.py "
          "(계층 활성화/승격 상태 복원)")
    print("  2) waste-classifier: publish_hier_version.py --apply (레지스트리 재발행 — 운영 배포 승인 필요)")
    print("  3) HF Spaces Settings → Secrets: SUPABASE_URL / SUPABASE_KEY 교체 → 재시작")


if __name__ == "__main__":
    main()
