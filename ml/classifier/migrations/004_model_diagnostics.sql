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
