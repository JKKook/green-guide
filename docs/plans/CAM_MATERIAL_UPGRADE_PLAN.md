# CAM 재질 유추 정밀화 플랜

> 작성: 2026-07-13. [GREENGUIDE_BLUEPRINT.md](GREENGUIDE_BLUEPRINT.md) 부속 —
> 다중재질 검출(`/predict-with-regions`)의 정확도 고도화 트랙.
> 현재 한계: 7×7 해상도(셀=32px) · 수용영역 번짐 · flat 13클래스 CAM 사용 ·
> CAM 추측을 검증 없이 노출.

## 원리 요약
`CAM_c(h,w) = Σₖ fc.weight[c,k] × feature_k(h,w)` — 분류기 최종 feature map 에
클래스별 fc 가중치를 투영한 "증거 지도". 셀별 argmax = 재질 지도.
u2netp(객체 위치, 클래스 무관)와 결합해 배경 헛증거를 걷어낸다.

---

## Stage 1 — 재학습 불필요 (즉시)

| # | 항목 | 방법 | 기대 효과 |
|---|---|---|---|
| 1-1 | **고해상 CAM** | ResNet 은 GAP 까지 fully-convolutional → **입력 448²로 한 번 더 forward** 하면 CAM 이 (25,14,14) 로 — 셀 16px. ONNX 를 동적 H/W 로 재export | 해상도 4배 (셀 면적 기준) |
| 1-2 | **계층 CAM 전환** | regions 를 flat 13 → hier 25 CAM 으로 — pet 몸통 vs 비닐 라벨 같은 **세부 재질 구분** | 재질 어휘 2배 |
| 1-3 | **재질 후보 제한** | 셀 경쟁에서 non_object/etc 제외 (재질이 아님) | 헛영역 감소 |
| 1-4 | **영역 재검증 (zoom-and-verify)** ★핵심 | CAM 이 제안한 각 영역을 **크롭해 풀 분류로 확정** — CAM 은 제안자, 분류기가 심판. 불일치 시 재분류 결과 채택 + 확신 미달 영역 폐기 | CAM 추측 → 검증된 판정 |
| 1-5 | saliency 가중 스코어 | 셀 확신 × 객체 점유도로 avg_conf 재정의 | 경계 셀 과대평가 억제 |

비용: 추가 forward 1(448) + 영역 수 N(≤4)회 재분류 ≈ +60~120ms — regions 는
비동기 병렬 호출이라 UX 영향 미미.

## Stage 2 — 재학습 1회 (합성 마스크 보조 head)

핵심 관찰: **다중객체 합성(synthesize_multiobject)은 붙여넣은 위치·알파를 알고
있다 = 픽셀 정답 마스크가 공짜**.
- 합성 시 `synmo_*_mask.png` (클래스별 인스턴스 마스크) 동시 산출
- ResNet layer3(14×14) 위에 경량 seg head(1×1 conv → 25ch) 추가,
  합성 마스크로 감독 + 실데이터는 CAM-distillation 으로 보조
- 출력: 학습된 28×28 재질 맵 — CAM 대비 경계 선명, 번짐 감소
- 게이트: 분류 head 성능 회귀 없음 조건 (aux loss 가중 0.1~0.3 탐색)

## Stage 3 — P6 (실데이터 픽셀 라벨)

탭-라벨·피드백이 쌓이면 SAM 으로 bbox/탭 → 마스크 의사라벨 생성 →
경량 인스턴스 세그(예: YOLO-seg nano 급) 학습. Stage 2 의 seg head 가
초기화·distill 교사로 재사용된다.

## 검증 계획
- 정량: 합성 홀드아웃(마스크 있음)에서 재질 맵 mIoU — Stage 별 비교
- 실전: realworld 다중재질 케이스(PET+라벨 등) 수집 → region slug 정답률
- 회귀: 분류 지표는 기존 게이트가 보호

## 진행 상태
- [x] Stage 1-1~1-5 구현·실검증 완료 (2026-07-13) — v5 ONNX 로 14×14 CAM 가동,
  혼재장면 battery 0.999+glass_deposit 0.873 분리 확인. flat-가드 계층 예외 처리 포함
- [ ] Stage 2 합성 마스크 산출 + aux head
- [ ] Stage 3 (P6)
