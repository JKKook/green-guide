-- 013: user_uploads AI 학습 활용 동의 (feature/repo-legal-privacy)
-- 앱 선택 약관 "촬영 사진 AI 학습 활용" 동의값을 /predict-hier 폼 필드
-- ai_training_opt_in 으로 받아 기록한다. null = 미전송(구버전 앱) → 미동의로 취급.
-- 학습 데이터셋 추출 시 `where ai_training_opt_in is true` 로 필터한다.
-- 적용은 사용자 승인 후 수동으로. 컬럼 미배포 상태에서도 서버는 기본 컬럼만으로
-- 기록을 계속한다 (uploads._remote_record 의 재시도 fail-open).
alter table public.user_uploads
  add column if not exists ai_training_opt_in boolean;
