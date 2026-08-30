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
