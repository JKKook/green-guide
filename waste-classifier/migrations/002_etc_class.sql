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
