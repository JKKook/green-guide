#!/usr/bin/env python3
"""Supabase 스토리지 사용량 점검 — 무료 티어 1GB 쿼터 가드.

배경: 2026-07-21 구 프로젝트가 models 버킷 누적(구버전 25개, 981MB)으로
쿼터 초과 → 서비스 전면 제한 → 프로젝트 이사까지 간 사태의 재발 방지.

사용자 지시(상시): 1GB 초과 위험(80% 이상) 시 사전 경고할 것.
- 단독 실행: 현황 출력 + 경고 판정 (rc=1 이면 위험 수위)
- 라이브러리: check_storage(client, incoming_bytes) — 업로드 전 사전 검사용

실행: .venv/bin/python scripts/storage_usage.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

QUOTA_BYTES = 1_000_000_000          # Free 티어 1GB
WARN_RATIO = 0.80                    # 80% 이상이면 사전 경고


def bucket_usage(client, bucket: str) -> tuple[int, int]:
    """버킷 총 (파일수, bytes) — 폴더 재귀."""
    total_n = total_b = 0
    stack = [""]
    while stack:
        prefix = stack.pop()
        try:
            entries = client.storage.from_(bucket).list(
                prefix, {"limit": 1000, "offset": 0})
        except Exception:  # noqa: BLE001
            return total_n, total_b
        for e in entries:
            meta = e.get("metadata")
            name = f"{prefix}/{e['name']}" if prefix else e["name"]
            if meta and meta.get("size") is not None:
                total_n += 1
                total_b += int(meta["size"])
            else:
                stack.append(name)   # 폴더
    return total_n, total_b


def storage_report(client) -> tuple[int, dict[str, tuple[int, int]]]:
    """전 버킷 사용량 → (총 bytes, {bucket: (n, bytes)})."""
    per: dict[str, tuple[int, int]] = {}
    total = 0
    for b in client.storage.list_buckets():
        bid = b.id if hasattr(b, "id") else b["id"]
        n, size = bucket_usage(client, bid)
        per[bid] = (n, size)
        total += size
    return total, per


def check_storage(client, incoming_bytes: int = 0) -> bool:
    """True = 안전. False = (예정 업로드 포함) 80% 초과 — 호출부는 사용자 경고 필요."""
    total, per = storage_report(client)
    projected = total + incoming_bytes
    ratio = projected / QUOTA_BYTES
    for bid, (n, size) in sorted(per.items()):
        print(f"  {bid:14s} {n:4d}개 {size/1e6:7.1f} MB")
    print(f"  {'합계':14s} {total/1e6:7.1f} MB"
          + (f" (+예정 {incoming_bytes/1e6:.1f} MB → {projected/1e6:.1f} MB)"
             if incoming_bytes else ""))
    if ratio >= WARN_RATIO:
        print(f"⚠️  경고: 쿼터 {ratio*100:.0f}% — 1GB 초과 위험. "
              "구버전/불필요 파일 정리 필요 (사용자에게 보고할 것)")
        return False
    print(f"  쿼터 사용률 {ratio*100:.0f}% — 안전")
    return True


def main() -> None:
    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    ok = check_storage(sb)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
