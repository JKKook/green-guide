#!/usr/bin/env python3
"""지역별 배출 규정 적재 — 공공데이터포털 → Supabase region_waste_rules.

원천: 행정안전부_생활쓰레기배출정보 조회서비스 (data.go.kr/data/15155080/openapi.do)
- REST, JSON, 일간 갱신, 시도·시군구 검색. 무료지만 **활용신청으로 서비스키 필요**.
- 키 준비: data.go.kr 회원 → 해당 API '활용신청'(자동승인) → 일반 인증키(Decoding)
  를 waste-api/.env 에 DATA_GO_KR_KEY=... 로 저장.

실행: .venv/bin/python scripts/load_region_rules.py [--sido 서울특별시]
      (인자 없으면 전국 전체 페이지네이션 적재 — upsert)
주기 갱신: 크론 등에서 주 1회면 충분 (규정 변경은 드묾).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

# 표준 조회서비스 엔드포인트 — 활용신청 페이지의 상세기능 명세 기준.
# (승인 후 실제 요청 URL/파라미터 명세가 마이페이지에 표시됨 — 상이하면 여기 수정)
BASE_URL = "http://apis.data.go.kr/1741000/HouseholdWasteDischargeInfo"
LIST_OP = "getHouseholdWasteDischargeInfoList"

# 표준데이터 필드 → 테이블 컬럼 매핑
FIELD_MAP = {
    "ctprvnNm": "sido",
    "signguNm": "sigungu",
    "mngZoneNm": "district",
    "emsnPlcType": "emit_place_type",
    "emsnPlc": "emit_place",
    "lifeWasteEmsnMthd": "method_general",
    "foodWasteEmsnMthd": "method_food",
    "rcyclEmsnMthd": "method_recycle",
    "tmprLqtyWasteEmsnMthd": "method_bulk",
    "lifeWasteEmsnDow": "days_general",
    "foodWasteEmsnDow": "days_food",
    "rcyclEmsnDow": "days_recycle",
    "emsnTime": "emit_time",
    "uncollectDay": "no_collect_day",
    "mngDeptNm": "managing_dept",
    "phoneNumber": "phone",
    "referenceDate": "data_date",
}


def fetch_rows(key: str, sido: str | None, page: int, rows: int = 500) -> list[dict]:
    params = {
        "serviceKey": key,
        "pageNo": page,
        "numOfRows": rows,
        "type": "json",
    }
    if sido:
        params["ctprvnNm"] = sido
    r = httpx.get(f"{BASE_URL}/{LIST_OP}", params=params, timeout=60)
    r.raise_for_status()
    body = r.json()
    # data.go.kr 표준 응답 구조 방어적 파싱
    items = (
        body.get("response", {}).get("body", {}).get("items")
        or body.get("HouseholdWasteDischargeInfo", [{}, {}])[-1].get("row")
        or []
    )
    if isinstance(items, dict):
        items = items.get("item", [])
    return items if isinstance(items, list) else [items]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sido", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    key = os.getenv("DATA_GO_KR_KEY")
    if not key:
        raise SystemExit(
            "DATA_GO_KR_KEY 미설정 — data.go.kr 에서 '행정안전부_생활쓰레기배출정보 "
            "조회서비스' 활용신청 후 .env 에 키를 추가하세요.")

    from supabase import create_client  # noqa: PLC0415
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    total = 0
    page = 1
    while True:
        items = fetch_rows(key, args.sido, page)
        if not items:
            break
        payload = []
        for it in items:
            row = {col: it.get(src) for src, col in FIELD_MAP.items()}
            if not (row.get("sido") and row.get("sigungu")):
                continue
            payload.append(row)
        if payload and not args.dry_run:
            sb.table("region_waste_rules").upsert(
                payload, on_conflict="sido,sigungu,district").execute()
        total += len(payload)
        print(f"[region] page {page}: {len(payload)}행 (누적 {total})")
        if len(items) < 500:
            break
        page += 1
    print(f"[region] 완료 — 총 {total:,}행 적재")


if __name__ == "__main__":
    main()
