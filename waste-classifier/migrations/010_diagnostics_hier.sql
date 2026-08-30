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
