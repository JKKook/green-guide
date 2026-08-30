# 시맨틱 증거 융합 (OCR·CLIP·CAM)

> 소스: `docs/plans/SEMANTIC_FUSION_PLAN.md`, `waste-api/src/semantic_evidence.py`·`clip_identity.py`, `docs/plans/CAM_MATERIAL_UPGRADE_PLAN.md`, `docs/greenguide_model_methods.html` §5 (2026-08-13 탐색)

"VLM처럼 판단하기" — CNN의 텍스처 통계 한계를 넘어 글자·정체·재질 증거를 fine 확률에 **log-linear prior**로 융합: `log p_fused = log p + Σ w_e·log prior_e`. 원칙: **증거 없으면 기존과 완전 동일**(별도 override 경로 없음, 게이트 이전 확률에 적용 → 강한 증거 시 자연히 reject 해제).

## 신호별 현행 상태

- **OCR** (RapidOCR PP-OCRv5 korean, onnxruntime): 어휘 2계층 — **A급 분리배출표시/재질어 boost ×6**(무색페트, HDPE, 종이팩…), **B급 정체어 ×2.5**(소주→glass_deposit, 영수증→paper_other…). 요청당 원본에서 1회만 실행, 확신 ≥0.75면 스킵(운영 -2~4s, [model-versions-accuracy](model-versions-accuracy.md) B1). 탭 경로는 항상 실행.
- **CLIP 제로샷** (이미지 인코더 INT8 84MB + 사전계산 66컨셉 `clip_concepts.npz`): **경로를 가린다** — 장면(풀프레임) 적용은 역효과(검은 기기를 'computer mouse' 0.99)라 기각, **탭 고립 crop만 +4건(w=0.5)**. 초기 설계의 '조용한 클래스 벌점'은 -3건이라 **부스트 전용 + 확신 임계 0.30 + 우도비 상한 8**로 재설계.
- **CAM 융합**: 448² 고해상 CAM(25클래스 14×14)의 탭 bbox 평균 활성을 prior로. **자기강화 위험 실증**(w≥0.4에서 -1~-4) → **w=0.15(무해 실측), 탭 경로만**. env `WASTE_API_CAM_W`.
- 융합 후 베스트 회전 텐서 1장만 재예측.

## CAM 다중재질 트랙 (CAM_MATERIAL_UPGRADE_PLAN)

`/predict-with-regions`의 재질 지도 정밀화. **Stage 1 완료(2026-07-13)**: 448² 재forward로 셀 16px, hier 25 CAM 전환, non_object/etc 셀 경쟁 제외, **zoom-and-verify(CAM은 제안자, 분류기가 심판)**, saliency 가중 스코어. Stage 2(합성 마스크 aux seg head — 합성은 픽셀 정답이 공짜) / Stage 3(SAM 의사라벨 → 경량 seg)는 미착수.

## 실측 이력과 주의

Phase 1+TTA+풀프레임 적용으로 51장 23→37(72.5%) 기록 — 단, **이 51장 지표는 이후 train 오염 판명으로 무효** ([retrain-loop](retrain-loop.md) 평가 무결성). OCR 융합 자체는 플립 0건, 확신 보정 효과만 실측(약국 영수증 paper 0.45→0.82).

부수 발견 2건이 더 큰 레버였음: ① EXIF 방향 분포 어긋남(AI-Hub 크롭은 센서 방향) → 회전 TTA ② u2 자동 크롭 역효과 → 장면 풀프레임 전환.

관련: [waste-api](waste-api.md) · [model-versions-accuracy](model-versions-accuracy.md) · [planning-docs](planning-docs.md)
