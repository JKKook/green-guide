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
