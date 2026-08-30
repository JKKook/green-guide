-- 011: 지역별 생활쓰레기 배출 규정 (전국생활쓰레기배출정보 표준데이터 적재)
--
-- 대한민국 조례 특성상 지자체마다 배출 요일·시간·방법이 다름 — 앱의 지역 선택
-- 시나리오 근거 테이블. 데이터 원천: 공공데이터포털 표준데이터(매일 갱신,
-- 7,398 관리구역) — scripts/load_region_rules.py 가 적재/갱신.

create table if not exists region_waste_rules (
  id bigint generated always as identity primary key,
  sido text not null,                    -- 시도명 (예: 서울특별시)
  sigungu text not null,                 -- 시군구명 (예: 강남구)
  district text not null default '',     -- 관리구역명 (동/구역 단위, 없으면 '')
  emit_place_type text,                  -- 배출장소 유형 (문전/거점 등)
  emit_place text,                       -- 배출장소 상세
  method_general text,                   -- 생활쓰레기 배출방법
  method_food text,                      -- 음식물쓰레기 배출방법
  method_recycle text,                   -- 재활용품 배출방법
  method_bulk text,                      -- 일시적 다량(대형)폐기물 배출방법
  days_general text,                     -- 생활쓰레기 배출요일
  days_food text,                        -- 음식물 배출요일
  days_recycle text,                     -- 재활용품 배출요일
  emit_time text,                        -- 배출 시간대
  no_collect_day text,                   -- 미수거일
  managing_dept text,                    -- 관리부서명
  phone text,                            -- 연락처
  data_date date,                        -- 데이터 기준일자
  updated_at timestamptz not null default now()
);

create index if not exists idx_region_rules_region
  on region_waste_rules (sido, sigungu);

-- 적재 upsert 기준 (scripts/load_region_rules.py on_conflict)
create unique index if not exists uq_region_rules_key
  on region_waste_rules (sido, sigungu, district);

comment on table region_waste_rules is
  '지자체별 생활쓰레기 배출 규정 — 공공데이터포털 전국생활쓰레기배출정보 표준데이터';
