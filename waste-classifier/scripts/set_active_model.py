"""model_versions 의 active 버전을 전환 (롤백/승급).

사용:
    .venv/bin/python scripts/set_active_model.py --version 20260520_013134
    .venv/bin/python scripts/set_active_model.py --list   # 버전 목록만
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

PREPROCESSOR_ROOT = Path(__file__).resolve().parent.parent.parent / "waste-preprocessor"


def _client():
    load_dotenv(PREPROCESSOR_ROOT / ".env")
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        sys.exit("ERROR: SUPABASE_URL / SUPABASE_KEY 미설정")
    return create_client(url, key)


def main() -> int:
    parser = argparse.ArgumentParser(description="active 모델 버전 전환")
    parser.add_argument("--version", type=str, help="active 로 설정할 version 문자열")
    parser.add_argument("--list", action="store_true", help="버전 목록만 출력")
    args = parser.parse_args()

    client = _client()

    res = (
        client.table("model_versions")
        .select("version,test_accuracy,is_active,num_classes,created_at")
        .order("created_at", desc=True)
        .limit(10)
        .execute()
    )
    print("model_versions (최근 10):")
    for r in res.data:
        flag = "★ ACTIVE" if r["is_active"] else ""
        acc = r.get("test_accuracy")
        print(f"  v{r['version']}  acc={acc}  classes={r['num_classes']}  {flag}")

    if args.list or not args.version:
        return 0

    # 대상 버전 존재 확인
    target = [r for r in res.data if r["version"] == args.version]
    if not target:
        sys.exit(f"\nERROR: version {args.version!r} 가 model_versions 에 없음")

    print(f"\n전환: → v{args.version} (acc={target[0].get('test_accuracy')}) 활성화")
    client.table("model_versions").update({"is_active": False}).eq("is_active", True).execute()
    client.table("model_versions").update({"is_active": True}).eq("version", args.version).execute()

    check = (
        client.table("model_versions")
        .select("version,test_accuracy")
        .eq("is_active", True)
        .execute()
    )
    print(f"✓ 현재 active: v{check.data[0]['version']} ({check.data[0].get('test_accuracy')})")
    print("\nwaste-api 즉시 반영:")
    print("  curl -X POST https://ethandev92-waste-api.hf.space/admin/reload-model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
