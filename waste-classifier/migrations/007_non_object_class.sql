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
