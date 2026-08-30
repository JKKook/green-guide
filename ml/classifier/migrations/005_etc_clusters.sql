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
