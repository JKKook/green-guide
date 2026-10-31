-- 014: 접근 제어 보강 — RLS(model_versions·region_waste_rules) + user-uploads 버킷 비공개
-- 근거: docs/plans/PRODUCTION_READINESS.md 결정 9 (2026-10-31)
--
-- 현재 상태: 두 테이블은 RLS 가 꺼져 있어 anon 키로 PostgREST 읽기·쓰기·삭제가 모두 열려 있다.
-- user-uploads 버킷은 public=true 라 경로만 알면 누구나 사진을 내려받는다.
--
-- 영향 범위:
--   · 서버(inference-api)·학습 스크립트는 service_role 키를 쓰므로 RLS 를 우회한다 → 동작 변화 없음.
--   · 앱은 Supabase 를 직접 호출하지 않는다(서버 경유) → 동작 변화 없음.
--   · 버킷 비공개 후 user_uploads.image_url(공개 URL)은 400/404 가 된다.
--     image_url 로 내려받는 ml/classifier 스크립트는 storage_path + service_role 다운로드로 바꿔야 한다
--     (revalidate.py · etc_queue.py · feedback_monitor.py · scripts/realworld_eval*.py · greenguide_classifier/ood.py).
--     retrain.py 는 storage_path 다운로드가 먼저라 영향 없음.
--
-- 적용: 리더 확인 뒤 Supabase SQL Editor 에서 실행. 한 트랜잭션이라 중간 실패 시 전부 되돌아간다.
-- 되돌리기: 파일 끝 "ROLLBACK 절차" 참고.

begin;

-- ── 1. model_versions: anon/authenticated 는 읽기만, 쓰기는 service_role(RLS 우회)만 ──────
alter table public.model_versions enable row level security;

drop policy if exists model_versions_public_read on public.model_versions;
create policy model_versions_public_read
  on public.model_versions
  for select
  to anon, authenticated
  using (true);

-- ── 2. region_waste_rules: 같은 규칙 (공공데이터라 읽기 공개, 적재는 load_region_rules.py service_role) ──
alter table public.region_waste_rules enable row level security;

drop policy if exists region_waste_rules_public_read on public.region_waste_rules;
create policy region_waste_rules_public_read
  on public.region_waste_rules
  for select
  to anon, authenticated
  using (true);

-- ── 3. user-uploads 버킷 비공개 ─────────────────────────────────────────────────────────
-- 비공개 버킷은 storage.objects RLS 를 따른다. user-uploads 에 anon 정책을 두지 않으면
-- service_role(서버)만 읽고 쓸 수 있다. 공개 URL(/object/public/...)은 즉시 거부된다.
update storage.buckets
   set public = false
 where id = 'user-uploads';

commit;


-- ═══════════════════════════════════════════════════════════════════════════════════════
-- 적용 전 확인 쿼리 (읽기 전용 — 먼저 실행해 결과를 리더에게 올린다)
-- ═══════════════════════════════════════════════════════════════════════════════════════
--
-- (a) 013 적용 여부: 1행이면 적용됨, 0행이면 013 먼저 적용
--   select column_name, data_type
--     from information_schema.columns
--    where table_schema = 'public' and table_name = 'user_uploads'
--      and column_name = 'ai_training_opt_in';
--
-- (b) 현재 RLS 상태: relrowsecurity = false 인 테이블이 이번 대상
--   select relname, relrowsecurity
--     from pg_class
--    where relnamespace = 'public'::regnamespace
--      and relname in ('model_versions', 'region_waste_rules', 'user_uploads',
--                      'waste_classes', 'model_diagnostics', 'etc_clusters')
--    order by relname;
--
-- (c) 버킷 공개 여부
--   select id, public from storage.buckets order by id;
--
-- (d) user-uploads 에 걸린 기존 storage 정책 (anon 읽기 정책이 있으면 비공개 효과가 없다 → 이름 확인 후 drop)
--   select policyname, cmd, roles, qual
--     from pg_policies
--    where schemaname = 'storage' and tablename = 'objects'
--      and (qual like '%user-uploads%' or with_check like '%user-uploads%');
--
-- ═══════════════════════════════════════════════════════════════════════════════════════
-- 적용 후 검증 (성공 기준: anon 쓰기 거부 · 버킷 공개 URL 거부)
-- ═══════════════════════════════════════════════════════════════════════════════════════
--
-- 1) 정책 확인
--   select tablename, policyname, cmd, roles
--     from pg_policies
--    where schemaname = 'public'
--      and tablename in ('model_versions', 'region_waste_rules');
--
-- 2) anon 쓰기 거부 (셸, anon 키 사용 — 기대: 401/403 또는 "new row violates row-level security policy")
--   curl -s -o /dev/null -w '%{http_code}\n' -X POST "$SUPABASE_URL/rest/v1/model_versions" \
--     -H "apikey: $SUPABASE_ANON_KEY" -H "Authorization: Bearer $SUPABASE_ANON_KEY" \
--     -H "Content-Type: application/json" -d '{"version":"rls_probe"}'
--
-- 3) anon 읽기 유지 (기대: 200)
--   curl -s -o /dev/null -w '%{http_code}\n' "$SUPABASE_URL/rest/v1/region_waste_rules?select=id&limit=1" \
--     -H "apikey: $SUPABASE_ANON_KEY" -H "Authorization: Bearer $SUPABASE_ANON_KEY"
--
-- 4) 버킷 공개 URL 거부 (기존 image_url 하나로 — 기대: 400 또는 404, 200 이면 실패)
--   curl -s -o /dev/null -w '%{http_code}\n' "<user_uploads.image_url 한 건>"
--
-- 5) 서버 동작 유지: 배포 서버 /predict-hier 1회 → user_uploads 1행 증가 확인
--
-- ═══════════════════════════════════════════════════════════════════════════════════════
-- ROLLBACK 절차 (문제 시)
-- ═══════════════════════════════════════════════════════════════════════════════════════
--   update storage.buckets set public = true where id = 'user-uploads';
--   drop policy if exists model_versions_public_read on public.model_versions;
--   drop policy if exists region_waste_rules_public_read on public.region_waste_rules;
--   alter table public.model_versions disable row level security;
--   alter table public.region_waste_rules disable row level security;
