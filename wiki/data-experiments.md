# 데이터 실험 대장 — 가설 검증·증강 A/B

> 소스: `waste-preprocessor/AIHUB_PAPER_HYPOTHESIS_TEST.md`, `DATA_AUGMENTATION_DESIGN.md`, `DATA_AUGMENTATION_RESULTS.md` (2026-08-13 탐색)

## AI-Hub paper 노이즈 가설 — **기각** (2026-05-30)

가설: "AI-Hub paper 8,353장의 facility-style noise crop이 spurious feature를 학습시켜 실사용 over-prediction 유발."

- 사전 근거는 강력해 보였음: 라벨 정합성 40%, 활성 모델 자가진단 99.5% paper(확증편향), CLIP 외부 심판 0% paper.
- 그러나 제거(Test B) 결과: frozen +0.74pp인데 **realworld -17.4pp** — 예측과 정반대. paper와 무관한 클래스까지 광범위 악화(clothes -100pp, glass -67pp).
- **교훈: "라벨 정합성"과 "학습 기여도"는 별개.** 노이즈 크롭도 폐기물 일반 시각 표현(반사·재질 텍스처·클러터) 학습에 기여한다. AI-Hub paper 유지 확정.
- 도메인 갭의 진짜 원인 후보: 학습 분포(facility+studio) vs 사용자 분포(실내 가정·폰 시점) 격차.

## 실내 합성·TACO 증강 A/B — Test C1 **미채택** (2026-05-31)

설계(DESIGN v2): 4 도메인 갭 축(실내 배경/손/폰 시점/결합)을 합성으로 직접 생성. TACO 1,145 + MIT Indoor 배경 287 + 자체 합성 1,825 추가. WebP는 실측 후 포기(JPG에 lossless 4.5배 역효과).

결과: frozen -0.34pp(게이트 PASS), realworld 전체 **-2.2pp** / 70%크롭 **+4.4pp**. 경계선 zone이라 모델 미채택, 다만 크롭 신호는 smart capture 측에 즉시 활용. 합성 인프라는 보존.

앱 측 Fix 1 시리즈(reject 임계 0.55, entropy reject, non_object 강제 reject 등)로 사용자 케이스 2/6→4/6. 잔여 실패는 모델이 0.75+ 확신하는 오답 → **"앱 측 lever 한계, 데이터 lever 필요"**.

## 이후 계보

이 실험들의 결론이 [planning-docs](planning-docs.md)의 방향 전환을 만들었다: 합성 실내 접근은 v8·v9에서 재시도 후 배경 편향으로 최종 폐기, TACO 실데이터 노선(2026-07-20~)과 AI-Hub 71385 대량 크롭([dataset-staging](dataset-staging.md))으로 전환. 평가 표본 부족(n=20)이 모든 학습 레버 판정을 막는 현 상황은 [model-versions-accuracy](model-versions-accuracy.md) 참조.
