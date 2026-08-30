-- 008: waste_classes 계층 확장 — 대분류(level 1) → 세부품목(level 2)
-- GREENGUIDE_BLUEPRINT.md §4. Supabase SQL Editor 에 붙여넣고 Run.
-- 하위호환: 기존 행은 level=1 기본값으로 유지, 기존 컬럼 변경 없음.

ALTER TABLE waste_classes ADD COLUMN IF NOT EXISTS level SMALLINT NOT NULL DEFAULT 1;
ALTER TABLE waste_classes ADD COLUMN IF NOT EXISTS parent_slug TEXT REFERENCES waste_classes(slug);
ALTER TABLE waste_classes ADD COLUMN IF NOT EXISTS disposal_stream TEXT;
ALTER TABLE waste_classes ADD COLUMN IF NOT EXISTS is_negative_guidance BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE waste_classes ADD COLUMN IF NOT EXISTS min_samples_to_activate INT NOT NULL DEFAULT 300;
ALTER TABLE waste_classes ADD COLUMN IF NOT EXISTS min_frozen_to_activate INT NOT NULL DEFAULT 30;

CREATE INDEX IF NOT EXISTS idx_waste_classes_parent ON waste_classes(parent_slug);
CREATE INDEX IF NOT EXISTS idx_waste_classes_level ON waste_classes(level, sort_order);

-- ── 신규 대분류 (level 1) ────────────────────────────────────────────────
INSERT INTO waste_classes (slug, level, display_name, summary, bin, how_to, caution,
  color_hex, icon_name, sort_order, trained_in_model, active)
VALUES
  ('paper_pack', 1, '종이팩',
   '우유팩·두유팩 등 종이팩은 일반 종이와 별도 스트림으로 배출합니다.',
   '종이팩 전용 수거함 (없으면 주민센터 교환)',
   '["내용물을 비우고 물로 헹구기","펼쳐서 말리기","일반 종이류와 섞지 않기"]'::jsonb,
   '["종이팩을 종이류에 넣으면 재활용 불가"]'::jsonb,
   '#8D6E63', 'local_drink', 15, false, false),
  ('hazardous', 1, '유해폐기물',
   '건전지·형광등·폐의약품은 일반쓰레기로 버리면 안 되는 유해폐기물입니다.',
   '전용 수거함 (주민센터·아파트 단지)',
   '["종류별 전용 수거함에 배출","일반쓰레기·재활용품에 절대 혼입 금지"]'::jsonb,
   '["환경오염·화재 위험"]'::jsonb,
   '#D32F2F', 'warning', 85, false, false)
ON CONFLICT (slug) DO NOTHING;

-- ── 세부품목 (level 2, 데이터 게이트 통과 전 active=false) ────────────────
INSERT INTO waste_classes (slug, level, parent_slug, display_name, summary, bin,
  how_to, caution, is_negative_guidance, color_hex, icon_name, sort_order,
  trained_in_model, active)
VALUES
  -- paper_pack 세부
  ('carton', 2, 'paper_pack', '종이팩(우유팩)', '우유팩·주스팩 등 종이팩.',
   '종이팩 전용 수거함',
   '["헹궈서 펼쳐 말리기","빨대·비닐 제거"]'::jsonb, '[]'::jsonb,
   false, '#8D6E63', 'local_drink', 16, false, false),
  ('paper_cup', 2, 'paper_pack', '종이컵', '종이컵은 종이팩류로 배출합니다.',
   '종이팩 전용 수거함',
   '["내용물 비우고 헹구기","여러 개는 겹쳐서"]'::jsonb,
   '["오염이 심하면 일반쓰레기"]'::jsonb,
   false, '#A1887F', 'coffee', 17, false, false),
  -- glass 세부
  ('glass_brown', 2, 'glass', '갈색 유리병', '갈색 유리병(맥주병 등 색유리).',
   '유리병 수거함', '["뚜껑 분리","내용물 비우기"]'::jsonb, '[]'::jsonb,
   false, '#795548', 'liquor', 31, false, false),
  ('glass_green', 2, 'glass', '녹색 유리병', '녹색 유리병.',
   '유리병 수거함', '["뚜껑 분리","내용물 비우기"]'::jsonb, '[]'::jsonb,
   false, '#2E7D32', 'liquor', 32, false, false),
  ('glass_clear', 2, 'glass', '무색 유리병', '무색(백색) 유리병.',
   '유리병 수거함', '["뚜껑 분리","내용물 비우기"]'::jsonb, '[]'::jsonb,
   false, '#90A4AE', 'liquor', 33, false, false),
  ('glass_deposit', 2, 'glass', '보증금 반환병', '소주병·맥주병은 빈용기 보증금 대상입니다.',
   '소매점 반환 (보증금 환급)',
   '["뚜껑 닫아 소매점에 반환","병당 70~130원 환급"]'::jsonb,
   '["깨진 병은 반환 불가 → 유리병 수거함"]'::jsonb,
   false, '#00838F', 'currency_exchange', 30, false, false),
  ('glass_etc', 2, 'glass', '기타 유리', '색상 무관 기타 유리 용기.',
   '유리병 수거함', '["뚜껑 분리","내용물 비우기"]'::jsonb, '[]'::jsonb,
   false, '#B0BEC5', 'liquor', 34, false, false),
  -- plastic 세부
  ('pet', 2, 'plastic', '페트병', '투명·유색 PET 음료병.',
   '투명페트 별도 배출 (지역별 확인)',
   '["라벨 완전 제거","내용물 비우고 압착","뚜껑 닫아 배출"]'::jsonb,
   '["라벨 부착 시 재활용 등급 하락"]'::jsonb,
   false, '#0288D1', 'water_drop', 41, false, false),
  ('plastic_other', 2, 'plastic', '기타 플라스틱', 'PET 외 플라스틱 용기류.',
   '플라스틱 수거함', '["내용물 비우고 헹구기"]'::jsonb, '[]'::jsonb,
   false, '#42A5F5', 'recycling', 42, false, false),
  -- vinyl 세부
  ('vinyl_clean', 2, 'vinyl', '비닐(깨끗한)', '이물질 없는 비닐·필름류.',
   '비닐 수거함', '["이물질 없이 모아서"]'::jsonb, '[]'::jsonb,
   false, '#7E57C2', 'shopping_bag', 51, false, false),
  ('vinyl_dirty', 2, 'vinyl', '오염 비닐', '음식물 등이 묻은 비닐은 재활용이 안 됩니다.',
   '종량제 봉투 (일반쓰레기)',
   '["오염이 심하면 일반쓰레기로","가볍게 헹궈지면 헹군 후 비닐 수거함"]'::jsonb,
   '["오염 비닐 혼입 시 전체 재활용 불가"]'::jsonb,
   true, '#9575CD', 'delete', 52, false, false),
  -- styrofoam 세부
  ('styrofoam_white', 2, 'styrofoam', '흰색 스티로폼', '흰색 완충재·포장 스티로폼.',
   '스티로폼 전용 수거함',
   '["테이프·스티커 제거","이물질 제거"]'::jsonb, '[]'::jsonb,
   false, '#ECEFF1', 'inventory_2', 61, false, false),
  ('styrofoam_color', 2, 'styrofoam', '컬러 스티로폼', '색깔 있는 스티로폼.',
   '지역별 상이 (다수 지역 일반쓰레기)',
   '["지역 규정 확인","불가 지역은 종량제 봉투"]'::jsonb,
   '["색상 스티로폼은 재활용 불가 지역 많음"]'::jsonb,
   true, '#FFB74D', 'inventory_2', 62, false, false),
  ('styrofoam_dirty', 2, 'styrofoam', '오염 스티로폼', '음식물이 묻은 스티로폼(컵라면 용기 등).',
   '종량제 봉투 (일반쓰레기)',
   '["오염 제거가 어려우면 일반쓰레기"]'::jsonb,
   '["오염된 채 배출 시 전체 오염"]'::jsonb,
   true, '#FF8A65', 'delete', 63, false, false),
  -- hazardous 세부
  ('battery', 2, 'hazardous', '폐건전지', '건전지·배터리는 전용 수거함으로.',
   '폐건전지 전용 수거함 (주민센터·아파트)',
   '["전용 수거함에 배출","테이프로 단자 절연 권장"]'::jsonb,
   '["일반쓰레기 혼입 시 화재 위험"]'::jsonb,
   false, '#F57F17', 'battery_alert', 86, false, false),
  -- trash(일반쓰레기) 세부 (⚠️오분리방지) — 기존 slug 'trash' 를 부모로 사용
  ('light_bulb', 2, 'trash', '전구(LED·백열)', 'LED·백열전구는 형광등이 아니라 일반쓰레기입니다.',
   '종량제 봉투 (일반쓰레기)',
   '["신문지에 싸서 종량제 봉투에","형광등 수거함에 넣지 않기"]'::jsonb,
   '["형광등만 전용 수거함 대상 (수은 함유)","전구 혼입 시 형광등 재활용 오염"]'::jsonb,
   true, '#FDD835', 'lightbulb', 91, false, false)
ON CONFLICT (slug) DO NOTHING;

-- 기존 대분류 행에 level/스트림 명시 (이미 존재하는 행 갱신)
UPDATE waste_classes SET level = 1 WHERE parent_slug IS NULL AND level IS DISTINCT FROM 1;
