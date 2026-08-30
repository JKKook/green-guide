-- ═══════════════════════════════════════════════════════════════════════
-- 그린가이드 신규 Supabase 프로젝트 부트스트랩 (2026-07-21)
-- 구 프로젝트 스토리지 쿼터 초과 → 이사. SQL Editor 에 전체 붙여넣고 Run 1회.
-- 구성: [0] 기반 테이블(초기 대시보드 생성분 복원) + 기반 6클래스 시드
--       [1~11] 기존 마이그레이션 전체 재생
-- ═══════════════════════════════════════════════════════════════════════

-- ── [0] 기반 테이블 ──────────────────────────────────────────────────────
create table if not exists public.waste_classes (
  slug text primary key,
  sort_order int not null default 100,
  display_name text not null,
  summary text,
  bin text,
  how_to jsonb not null default '[]'::jsonb,
  caution jsonb not null default '[]'::jsonb,
  color_hex text,
  icon_name text,
  trained_in_model boolean not null default false,
  active boolean not null default true,
  created_at timestamptz not null default now()
);
alter table public.waste_classes enable row level security;

create table if not exists public.user_uploads (
  id text primary key,
  uploaded_at timestamptz not null default now(),
  image_url text not null,
  storage_path text not null,
  predicted_class text not null,
  predicted_confidence double precision,
  all_probabilities jsonb,
  model_arch text,
  inference_ms double precision,
  feedback_status text not null default 'pending',
  feedback_label text,
  feedback_at timestamptz
);
alter table public.user_uploads enable row level security;

-- ── [0b] 기반 6클래스 시드 (구 프로젝트 초기 대시보드 입력분 — 앱 fallback 기준 복원) ──
insert into public.waste_classes
  (slug, sort_order, display_name, summary, bin, how_to, caution,
   color_hex, icon_name, trained_in_model, active)
values
  ('paper', 10, '종이류', '신문지·책·종이상자 등 종이류는 이물질 없이 배출합니다.',
   '종이류 수거함',
   '["테이프·스프링 등 이물질 제거","비에 젖지 않게 배출","오염된 종이는 일반쓰레기"]'::jsonb,
   '["코팅된 영수증·컵라면 용기는 재활용 불가"]'::jsonb,
   '#8D6E63', 'description', true, true),
  ('cardboard', 20, '종이상자', '택배 상자 등 골판지는 펼쳐서 종이류로 배출합니다.',
   '종이류 수거함',
   '["테이프·운송장 스티커 제거","상자를 펼쳐 부피 줄이기"]'::jsonb,
   '["음식물 오염 시(피자박스 기름) 일반쓰레기"]'::jsonb,
   '#A1887F', 'inventory_2', true, true),
  ('glass', 30, '유리류', '유리병은 내용물을 비우고 유리류로 배출합니다.',
   '유리류 수거함',
   '["내용물 비우고 헹구기","뚜껑은 재질별 분리","소주·맥주병은 보증금 환급 대상"]'::jsonb,
   '["깨진 유리는 신문지에 싸서 일반쓰레기(불연)로"]'::jsonb,
   '#26A69A', 'wine_bar', true, true),
  ('metal', 40, '캔·고철', '음료캔·통조림 등 금속류는 캔류로 배출합니다.',
   '캔류 수거함',
   '["내용물 비우고 헹구기","가능하면 압착","부탄·스프레이는 구멍 뚫어 배출"]'::jsonb,
   '["유해물질 캔은 내용물 완전히 비우기"]'::jsonb,
   '#78909C', 'blender', true, true),
  ('plastic', 50, '플라스틱', '페트병·플라스틱 용기는 라벨 제거 후 배출합니다.',
   '플라스틱 수거함',
   '["내용물 비우고 헹구기","라벨·뚜껑 분리","투명 페트병은 별도 배출"]'::jsonb,
   '["음식물 오염이 심하면 일반쓰레기"]'::jsonb,
   '#42A5F5', 'recycling', true, true),
  ('trash', 60, '일반쓰레기', '재활용이 불가한 폐기물은 종량제 봉투로 배출합니다.',
   '종량제 봉투',
   '["지자체 종량제 봉투 사용","배출 요일·장소 준수"]'::jsonb,
   '["재활용 가능 자원 혼입 금지"]'::jsonb,
   '#9E9E9E', 'delete', true, true)
on conflict (slug) do nothing;

-- ── [0c] 13클래스 시대 추가분 시드 (대시보드 생성분 복원 — 008 의 parent 참조 대상) ──
insert into public.waste_classes
  (slug, sort_order, display_name, summary, bin, how_to, caution,
   color_hex, icon_name, trained_in_model, active)
values
  ('vinyl', 55, '비닐류', '비닐봉지·포장 필름류는 비닐 수거함으로 배출합니다.',
   '비닐 수거함',
   '["이물질 없이 모아서 배출","오염된 비닐은 일반쓰레기"]'::jsonb,
   '["음식물 묻은 비닐 혼입 시 전체 재활용 불가"]'::jsonb,
   '#7E57C2', 'shopping_bag', true, true),
  ('styrofoam', 58, '스티로폼', '흰색 스티로폼은 전용 수거함으로 배출합니다.',
   '스티로폼 전용 수거함',
   '["테이프·스티커·이물질 제거","흰색만 재활용 (색상·오염은 지역 규정 확인)"]'::jsonb,
   '["오염 스티로폼은 일반쓰레기"]'::jsonb,
   '#ECEFF1', 'inventory_2', true, true),
  ('clothes', 70, '의류·원단', '옷·이불 등 원단류는 의류수거함으로 배출합니다.',
   '의류수거함',
   '["세탁 후 배출 권장","젖지 않게 배출"]'::jsonb,
   '["오염·훼손 심한 원단은 일반쓰레기"]'::jsonb,
   '#EC407A', 'checkroom', true, true),
  ('food_waste', 75, '음식물쓰레기', '음식물은 물기를 제거하고 음식물 수거함으로 배출합니다.',
   '음식물쓰레기 수거함/봉투',
   '["물기 최대한 제거","이물질(비닐·이쑤시개) 혼입 금지"]'::jsonb,
   '["뼈·조개껍데기·달걀껍질은 일반쓰레기"]'::jsonb,
   '#8BC34A', 'restaurant', true, true)
on conflict (slug) do nothing;


-- ═══ 재생: 001_model_versions.sql ═══
-- Active Learning: 모델 버전 관리 테이블.
-- 새 ONNX 가 학습될 때마다 row 가 추가되고, is_active=true 인 row 가
-- "production" 모델. waste-api / Flutter 앱은 부팅 시 이 row 를 조회해서
-- 자신의 캐시 버전과 비교 후 더 새 게 있으면 다운로드.
--
-- 사용:
--   Supabase Dashboard → SQL Editor 에 붙여넣고 실행.

-- ─────────────────────────────────────────────────────────────────
-- 1. model_versions 테이블
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_versions (
    id BIGSERIAL PRIMARY KEY,
    version TEXT NOT NULL UNIQUE,            -- "20260519_153021" timestamp 형식
    color_storage_path TEXT NOT NULL,        -- 예: "v20260519_153021/classifier.onnx"
    edge_storage_path TEXT,                  -- 선택: edge stream (없으면 NULL)
    color_sha256 CHAR(64) NOT NULL,          -- 다운로드 무결성 검증
    edge_sha256 CHAR(64),
    color_url TEXT NOT NULL,                 -- public download URL
    edge_url TEXT,
    test_accuracy NUMERIC(6, 4),             -- 예: 0.9261
    num_classes INT NOT NULL,
    class_labels JSONB NOT NULL,             -- ["cardboard", "glass", ...]
    feedback_count INT NOT NULL DEFAULT 0,   -- 이 버전 학습에 사용된 사용자 피드백 수
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_model_versions_created
    ON model_versions(created_at DESC);

-- 한 번에 하나의 active version 만 허용 (partial unique index).
CREATE UNIQUE INDEX IF NOT EXISTS uq_model_versions_single_active
    ON model_versions(is_active) WHERE is_active = TRUE;


-- ─────────────────────────────────────────────────────────────────
-- 2. models Storage 버킷 (public read).
--    bucket 자체는 Supabase Dashboard 의 Storage 메뉴에서도 만들 수 있으나
--    SQL 로도 동일하게 가능.
-- ─────────────────────────────────────────────────────────────────
INSERT INTO storage.buckets (id, name, public)
VALUES ('models', 'models', TRUE)
ON CONFLICT (id) DO UPDATE SET public = TRUE;

-- 익명 read 허용 (public bucket 이지만 policy 도 명시).
-- 이미 존재하면 무시.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE schemaname = 'storage'
          AND tablename = 'objects'
          AND policyname = 'models_public_read'
    ) THEN
        CREATE POLICY models_public_read
            ON storage.objects FOR SELECT
            USING (bucket_id = 'models');
    END IF;
END $$;

-- service_role 만 write 가능 (anon 은 download 만, retrain.py 가 service_role 로 upload).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE schemaname = 'storage'
          AND tablename = 'objects'
          AND policyname = 'models_service_write'
    ) THEN
        CREATE POLICY models_service_write
            ON storage.objects FOR INSERT
            WITH CHECK (bucket_id = 'models' AND auth.role() = 'service_role');
    END IF;
END $$;


-- ═══ 재생: 002_etc_class.sql ═══
-- "기타/분류 불가" 클래스 추가.
--
-- 목적:
--   현재 6대 카테고리(cardboard/glass/metal/paper/plastic/trash)에 해당하지
--   않는 객체(우레탄, 손, 가구, 사람, 빈 배경 등 OOD) 에 대해 사용자가
--   "기타" 라벨로 corrected 피드백을 줄 수 있도록 함.
--
-- 흐름:
--   1. 사용자가 사진 찍음 → 모델이 어색하게 "cardboard" 등으로 분류
--   2. 사용자가 결과 모달의 피드백 카드에서 "기타" 로 corrected 라벨링
--   3. user_uploads.feedback_label = 'etc' 로 기록
--   4. 다음 retrain.py 실행 시 'etc' 폴더에 이미지 수집 + 7개 클래스로 학습
--   5. 새 모델은 OOD 입력에 대해 'etc' 로 답할 수 있게 됨
--
-- 사용:
--   Supabase Dashboard → SQL Editor 에서 실행.

INSERT INTO public.waste_classes
  (slug, sort_order, display_name, summary, bin, how_to, caution,
   color_hex, icon_name, trained_in_model, active)
VALUES
  ('etc', 65,
   '기타 / 분류 불가',
   '6대 분리수거 카테고리에 속하지 않는 모든 것 (우레탄·천·손·가구 등)',
   '해당 없음 — 재질 따로 확인 필요',
   '["이 사진이 6대 분리수거 카테고리(종이상자/유리/캔/종이/플라스틱/일반쓰레기) 중 어디에도 해당하지 않을 때 선택해주세요.",
     "사용자 피드백이 누적되면 다음 모델 학습 때 ''기타'' 클래스로 정식 학습됩니다.",
     "특정 재질(예: 우레탄·의류·전자제품)을 자주 마주치시면 알려주세요 — 별도 클래스로 분리할 수 있습니다."]'::jsonb,
   '["이 라벨은 아직 모델이 학습하지 않았습니다 (피드백 수집 단계 — UI 에 ''NEW'' 배지로 표시됨).",
     "사용자가 ''이건 6개 카테고리 어디에도 안 맞아'' 라고 알려주는 용도입니다."]'::jsonb,
   '#9E9E9E', 'help_outline',
   FALSE,  -- 아직 학습 전
   TRUE    -- active = 피드백·UI 에서 선택 가능
)
ON CONFLICT (slug) DO NOTHING;


-- ═══ 재생: 003_update_etc_metadata.sql ═══
-- "기타/분류 불가"(etc) 클래스 메타데이터 최신화.
--
-- 이유: 기존 설명이 "6대 카테고리" 기준 + "아직 학습 안 됨" 인데,
-- 현재는 11개 클래스 + etc 도 trained_in_model=true (데이터 적어 정확도만 낮음).
--
-- 사용: Supabase SQL Editor 에 붙여넣고 Run.

UPDATE public.waste_classes SET
  summary = '어느 분리수거 카테고리에도 맞지 않는 물건 (복합재질·전자제품·생활용품 등)',
  bin = '재질 확인 후 배출 — 대부분 일반쓰레기 또는 전용 수거함',
  how_to = '[
    "분리수거 대상이 아니거나 재질을 특정하기 어려운 물건일 때 선택하세요.",
    "여러 재질이 섞인 복합 제품(예: 칫솔·장난감·소형가전)은 대부분 일반쓰레기입니다.",
    "자주 마주치는 특정 품목은 피드백으로 알려주시면 별도 클래스로 추가됩니다."
  ]'::jsonb,
  caution = '[
    "아직 학습 데이터가 적어 자동 분류 정확도가 낮습니다 — 피드백이 쌓일수록 정확해집니다.",
    "배터리·형광등·의약품 등 유해 폐기물은 전용 수거함을 이용하세요 (일반쓰레기 아님)."
  ]'::jsonb
WHERE slug = 'etc';


-- ═══ 재생: 004_model_diagnostics.sql ═══
-- 모델 진단 이력 — diagnose.py 가 재학습마다 한 행씩 기록.
--
-- 목적: 버전 간 per-class 정확도/혼동쌍을 시간축으로 추적해 회귀(특정 클래스
--       하락)를 감지하고, 운영 대시보드/앱에서 조회 가능하게 한다.
--       (레포 outputs/logs/diagnosis/ JSONL 과 이중 저장.)
--
-- 사용: Supabase SQL Editor 에 붙여넣고 Run.

CREATE TABLE IF NOT EXISTS public.model_diagnostics (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    version         TEXT NOT NULL,
    arch            TEXT NOT NULL DEFAULT 'cnn',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    test_size       INT,
    num_classes     INT,
    accuracy        REAL,
    macro_f1        REAL,
    per_class       JSONB,    -- [{label, precision, recall, f1, support}]
    confusion_pairs JSONB,    -- [{true, pred, count, frac_of_true}]
    weak_classes    JSONB,    -- ["paper", ...]
    needs_data      JSONB,    -- ["paper", "vinyl", ...]
    regressions     JSONB,    -- [{label, prev_recall, new_recall, drop_pp}]
    gate_pass       BOOLEAN,
    gate_reasons    JSONB     -- ["전체 정확도 ...", ...]
);

CREATE INDEX IF NOT EXISTS idx_model_diagnostics_version
    ON public.model_diagnostics (version);
CREATE INDEX IF NOT EXISTS idx_model_diagnostics_created
    ON public.model_diagnostics (created_at DESC);

-- 진단은 운영자(service_role)만 기록. 공개 읽기는 필요 시 별도 정책으로.
ALTER TABLE public.model_diagnostics ENABLE ROW LEVEL SECURITY;


-- ═══ 재생: 005_etc_clusters.sql ═══
-- etc 자동 발견 군집 리뷰 큐 — etc_queue.py 가 Stage2 에서 한 행씩 기록.
--
-- 목적: 어느 기존 클래스에도 안 붙는 etc 이미지들이 뭉친 '신규 클래스 후보'를
--       운영자가 검토/명명하는 큐. pseudo-class(waste_classes.active=false)와 1:1.
--       운영자가 이름·배출법을 넣고 active=true 로 승격하면 정식 클래스가 된다.
--
-- 사용: Supabase SQL Editor 에 붙여넣고 Run.

CREATE TABLE IF NOT EXISTS public.etc_clusters (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    slug              TEXT NOT NULL UNIQUE,         -- etc_auto_<ts>_<k> (waste_classes.slug 와 동일)
    size              INT NOT NULL,                 -- 군집 크기
    sample_upload_ids JSONB,                        -- 대표 샘플 upload_id (검토용)
    status            TEXT NOT NULL DEFAULT 'pending_review',  -- pending_review | promoted | discarded
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_at       TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_etc_clusters_status
    ON public.etc_clusters (status);

ALTER TABLE public.etc_clusters ENABLE ROW LEVEL SECURITY;


-- ═══ 재생: 006_electronics_class.sql ═══
-- 전자제품(electronics) 클래스 추가 — AI Hub 생활폐기물(140) '전자제품' 데이터로 학습.
--
-- 목적: 마우스 등 소형 전자제품이 의류/플라스틱으로 오분류되는 문제 해결.
--       전자제품은 일반쓰레기가 아니라 소형 전자폐기물(전용 수거) 대상.
--
-- 순서: 이 INSERT 를 retrain 전에 실행해야 /labels 메타·표시가 정상.
--       trained_in_model 은 retrain 후 _mark_trained_classes 가 자동 true 로 변경.
--
-- 사용: Supabase SQL Editor 에 붙여넣고 Run.

INSERT INTO public.waste_classes
  (slug, sort_order, display_name, summary, bin, how_to, caution,
   color_hex, icon_name, trained_in_model, active)
VALUES
  ('electronics', 115,
   '전자제품',
   '소형 전자제품·생활가전 (일반쓰레기 아님 — 전용 수거 대상)',
   '소형: 전용 수거함 / 대형: 무상방문수거 신청',
   '[
     "소형 가전(마우스·충전기·이어폰 등)은 주민센터·행정복지센터의 소형 폐가전 수거함에 배출하세요.",
     "대형 가전(냉장고·세탁기·TV 등)은 폐가전 무상방문수거(1599-0903)를 신청하면 무료로 수거합니다.",
     "분리 가능한 부품(전선·배터리)은 가능하면 분리해 배출하세요."
   ]'::jsonb,
   '[
     "배터리·충전지가 들어있으면 분리해 전용 수거함에 버리세요 (발화 위험).",
     "휴대폰·PC 등 저장장치가 있는 기기는 개인정보를 먼저 삭제하세요.",
     "일반쓰레기 종량제 봉투에 넣어 버리면 안 됩니다."
   ]'::jsonb,
   '#5C6BC0', 'devices',
   FALSE,     -- 학습 후 retrain 의 _mark_trained_classes 가 true 로 변경
   TRUE)      -- active=true 여야 /labels·앱에 노출
ON CONFLICT (slug) DO NOTHING;

-- 확인
SELECT slug, display_name, trained_in_model, active
FROM public.waste_classes WHERE slug = 'electronics';


-- ═══ 재생: 007_non_object_class.sql ═══
-- non_object 클래스 추가 — 손/신체/배경 등 '폐기물이 아닌 것' 인식용. (Tier 2-1)
--
-- 목적: 손에 든 마우스 등이 cardboard 로 과신 오분류되는 문제 해결.
--       모델이 'non_object' 로 분류하면 앱이 배출카드 대신 "다시 촬영" 안내.
--
-- 특수성: 이건 사용자 배출 카테고리가 아니라 모델 내부 '재촬영' 신호다.
--   - active = FALSE  → /labels·피드백 목록·배출카드에 노출 안 됨
--   - 앱은 predicted_class == 'non_object' 를 특수 처리(재촬영 안내)
--   - trained_in_model 은 retrain 후 동기화 SQL 이 true 로 변경
--
-- 사용: Supabase SQL Editor 에 붙여넣고 Run.

INSERT INTO public.waste_classes
  (slug, sort_order, display_name, summary, bin, how_to, caution,
   color_hex, icon_name, trained_in_model, active)
VALUES
  ('non_object', 200,
   '분류 대상 아님',
   '폐기물이 화면에 없거나(손·배경만), 인식이 어려운 경우',
   NULL,
   '["폐기물만 화면 가운데에 담아 다시 촬영해주세요.","손·배경이 너무 많이 나오지 않게 해주세요."]'::jsonb,
   '[]'::jsonb,
   '#90A4AE', 'help_outline',
   FALSE,
   FALSE)   -- active=false: 사용자 카테고리 아님(모델 내부 재촬영 신호)
ON CONFLICT (slug) DO NOTHING;

SELECT slug, display_name, trained_in_model, active
FROM public.waste_classes WHERE slug = 'non_object';


-- ═══ 재생: 008_hierarchy.sql ═══
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


-- ═══ 재생: 009_model_versions_hier.sql ═══
-- 009: model_versions 계층 메타 — 온디바이스 대분류 라벨 + taxonomy 스냅샷
-- Supabase SQL Editor 에 붙여넣고 Run.

ALTER TABLE model_versions ADD COLUMN IF NOT EXISTS coarse_labels JSONB;
ALTER TABLE model_versions ADD COLUMN IF NOT EXISTS fine_to_coarse JSONB;
ALTER TABLE model_versions ADD COLUMN IF NOT EXISTS taxonomy_hash TEXT;

COMMENT ON COLUMN model_versions.coarse_labels IS
  '온디바이스 대분류 모델 라벨 순서. class_labels(세부) 와 별개.';
COMMENT ON COLUMN model_versions.fine_to_coarse IS
  '{"fine_slug": "coarse_slug"} 롤업 매핑 — 클라이언트가 세부→대분류 표시에 사용.';
COMMENT ON COLUMN model_versions.taxonomy_hash IS
  'waste_classes 계층 스냅샷 해시 — 앱 캐시 무효화 신호.';


-- ═══ 재생: 010_diagnostics_hier.sql ═══
-- 010: model_diagnostics 레벨별 지표 — 대분류 회귀 감시 강화
-- Supabase SQL Editor 에 붙여넣고 Run.
-- 게이트 원칙 (blueprint §7): 대분류 recall 회귀는 강하게 차단(-5pp),
-- 세부품목은 활성화 임계(f1>=0.80 & frozen>=30) 충족 시에만 승격.

ALTER TABLE model_diagnostics ADD COLUMN IF NOT EXISTS coarse_accuracy REAL;
ALTER TABLE model_diagnostics ADD COLUMN IF NOT EXISTS fine_accuracy REAL;
ALTER TABLE model_diagnostics ADD COLUMN IF NOT EXISTS per_fine JSONB;
ALTER TABLE model_diagnostics ADD COLUMN IF NOT EXISTS fine_activation JSONB;

COMMENT ON COLUMN model_diagnostics.coarse_accuracy IS '대분류 정확도 (fine 롤업 후, 전체 test)';
COMMENT ON COLUMN model_diagnostics.fine_accuracy IS '세부 정확도 (fine-감독 test 아이템만)';
COMMENT ON COLUMN model_diagnostics.per_fine IS '[{label, precision, recall, f1, support}] 세부품목별';
COMMENT ON COLUMN model_diagnostics.fine_activation IS '{fine_slug: {f1, support, ready}} 활성화 판정';


-- ═══ 재생: 011_region_waste_rules.sql ═══
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
