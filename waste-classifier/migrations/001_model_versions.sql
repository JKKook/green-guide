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
