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

# 실측 확인된 엔드포인트 (2026-07-15, https 필수 — http 는 401)
BASE_URL = "https://apis.data.go.kr/1741000/household_waste_info"
LIST_OP = "info"

# 실측 응답 필드 → 테이블 컬럼 매핑 (totalCount 10,175 / 2026-07 기준)
FIELD_MAP = {
    "CTPV_NM": "sido",
    "SGG_NM": "sigungu",
    "MNG_ZONE_NM": "district",
    "EMSN_PLC_TYPE": "emit_place_type",
    "EMSN_PLC": "emit_place",
    "LF_WST_EMSN_MTHD": "method_general",
    "FOD_WST_EMSN_MTHD": "method_food",
    "RCYCL_EMSN_MTHD": "method_recycle",
    "TMPRY_BULK_WASTE_EMSN_MTHD": "method_bulk",
    "LF_WST_EMSN_DOW": "days_general",
    "FOD_WST_EMSN_DOW": "days_food",
    "RCYCL_EMSN_DOW": "days_recycle",
    "UNCLLT_DAY": "no_collect_day",
    "MNG_DEPT_NM": "managing_dept",
    "MNG_DEPT_TELNO": "phone",
    "DAT_CRTR_YMD": "data_date",
}


def fetch_rows(key: str, sido: str | None, page: int, rows: int = 100) -> list[dict]:
    params = {
        "serviceKey": key,
        "pageNo": page,
        "numOfRows": rows,
        "type": "json",
    }
    # (이 API 는 지역 필터 파라미터 미제공 — 전체 페이지네이션 후 적재)
    del sido
    r = httpx.get(f"{BASE_URL}/{LIST_OP}", params=params, timeout=60)
    r.raise_for_status()
    items = r.json().get("response", {}).get("body", {}).get("items") or {}
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
            row["district"] = row.get("district") or ""
            # 배출 시간대: 생활쓰레기 시작~종료 시각 합성 (품목별 시간이 대부분 동일)
            bgn, end = it.get("LF_WST_EMSN_BGNG_TM"), it.get("LF_WST_EMSN_END_TM")
            row["emit_time"] = f"{bgn}~{end}" if bgn and end else None
            payload.append(row)
        if payload and not args.dry_run:
            sb.table("region_waste_rules").upsert(
                payload, on_conflict="sido,sigungu,district").execute()
        total += len(payload)
        if page % 10 == 0 or len(items) < 100:
            print(f"[region] page {page}: 누적 {total}행")
        if len(items) < 100:   # 서버가 페이지당 100행 캡 — 미만이면 마지막 페이지
            break
        page += 1
    print(f"[region] 완료 — 총 {total:,}행 적재")


if __name__ == "__main__":
    main()
