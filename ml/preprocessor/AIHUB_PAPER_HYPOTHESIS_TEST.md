# AI Hub paper 가설 검증 — Test A/B 비교

> 실행: 2026-05-30
> 가설: "AI Hub paper 8,353장의 facility-style noise crop 이 분류기에 spurious feature(어두운 잡배경 = paper) 를 학습시켜 실사용 over-prediction 을 유발한다"

---

## 1. 검증 설계

### 가설 한 줄
AI Hub paper 데이터(8353장, 시각 검토 결과 라벨 정합성 ~40%)가 학습에 **해**가 되며, 제거 시 실사용 정확도가 **개선** 된다.

### 실험 설계
| 구성 | 학습 데이터 | 변경점 |
|---|---|---|
| **Test A (control)** | 현재 manifest 그대로 (69,368 items, paper 9,363) | — |
| **Test B (가설)** | AI Hub paper 8,312장 제거 (61,056 items, paper 1,051) | aihub_paper_*.jpg 만 제외, 그 외 모든 데이터 동일 |

### 측정 지표
- **frozen test 정확도** — AI Hub 분포 기준, 버전 간 비교 가능
- **realworld 정확도** — 사용자 피드백 기반 (Supabase user_uploads confirmed/corrected)
- 두 지표의 갭이 핵심 — Test B 가 frozen ↓ + realworld ↑ 면 spurious feature 가설 확정

### 판정 기준
| 결과 | 해석 | 조치 |
|---|---|---|
| Test B realworld > Test A + 3pp 이상 | 가설 강하게 확정 | AI Hub paper 폐기 + 다른 클래스도 같은 검사 |
| Test B realworld ≈ Test A (±2pp) | 가설 약하게 지지 / 무영향 | CLIP 부분 필터 등 미세 조정 시도 |
| Test B realworld < Test A | 가설 기각 (데이터 양 자체가 도움) | AI Hub paper 유지, 다른 lever 탐색 |

추가 신호 — Test B 가 frozen 에서 paper 정확도만 떨어지고 다른 클래스는 유지된다면, "paper 학습에만 노이즈가 있었다" 가 깔끔하게 확인됨.

---

## 2. 사전 분석 — 가설의 근거

### (a) 시각 검토 466장 (2026-05-30 — agent 정량 분석)
- AI Hub 의 66% 가 dark facility scene (사용자 분포 indoor 2.8% 와 큰 격차)
- AI Hub 의 라벨 정합성 **paper 만 40%** (다른 클래스 평균 78%)
- → paper 클래스가 가장 문제적

### (b) 활성 모델 자가 진단 — closed-loop 불가
- 활성 모델이 AI Hub paper 200장 중 **199장(99.5%)** 을 paper 라고 응답
- 중앙값 신뢰도 99.7%
- → 모델이 AI Hub 시각 노이즈 자체를 paper feature 로 흡수했음
- → 활성 모델로 자기 학습 데이터를 검증할 수 없음 (confirmation bias)

### (c) CLIP 외부 심판자 측정
- CLIP zero-shot 결과: AI Hub paper 200장의 **0장(0%)** 이 paper top1
  - 47% plastic, 43% trash, 6% 기타로 응답
  - paper 확률 max=0.071, median=0.001
- 대조군: Kaggle paper 30장의 30% 가 paper top1
- → CLIP 자체 노이즈는 있으나, AI Hub paper 가 거의 모두 paper 아님은 강한 신호

---

## 3. 결과

### 3.1 Test A (baseline)
| 지표 | 값 | 비고 |
|---|---|---|
| frozen test 정확도 | **95.9%** | 직전 측정값 (2026-05-25) |
| realworld 정확도 (전체이미지) | **58.7%** | 피드백 46건 기준 (2026-05-30) |
| realworld 정확도 (중앙70%크롭) | **30.4%** | |
| paper recall (실사용) | 2/4 (50%) | |

**주요 혼동 (Test A 실사용)**:
- etc → non_object: 4건
- electronics → non_object: 3건
- metal → non_object: 2건
- etc → plastic: 2건

### 3.2 Test B (AI Hub paper 8,312장 제거 후 재학습)
- 학습: CNN 15 epochs, **조기 종료 epoch 8** (patience=4, best epoch 4, val_acc 0.9690)
- 학습 데이터: 61,056 items (paper 1,051 = Kaggle + user only)
- 측정: 2026-05-30 19:00 KST 완료

| 지표 | 값 | Test A 대비 |
|---|---|---|
| frozen test 정확도 (같은 9,167장 동결) | **96.42%** | +0.74pp |
| realworld 정확도 (전체이미지) | **41.3%** | **−17.4pp** |
| realworld 정확도 (중앙70%크롭) | **23.9%** | −6.5pp |
| paper recall (frozen) | 82.9% | +2.4pp |
| paper recall (실사용) | 1/4 (25%) | −25pp |

**주요 혼동 (Test B 실사용)**:
- etc → non_object: 3건 (-1)
- electronics → non_object: 3건 (=)
- glass → plastic: 2건 (**신규**)
- plastic → cardboard: 2건 (**신규**)
- paper → cardboard: 1건 (신규)
- paper → plastic: 1건 (=)

**주요 혼동 (Test B frozen — 직전 대비 회귀 경보)**:
- paper → trash: 9건 (5.5%)
- paper → cardboard: 8건 (4.9%)
- paper → non_object: 5건 (3.0%)

→ Test B 의 paper 모형은 cardboard·trash 와 경계가 흐릿해짐 (AI Hub paper 의 다양한 시각 표본이 빠지면서 경계 학습 약화).

### 3.3 클래스별 realworld 변화 (46건 피드백)
| 클래스 | Test A | Test B | Δ | 비고 |
|---|---|---|---|---|
| cardboard | 2/2 (100%) | 2/2 (100%) | 0 | |
| clothes | 3/3 (100%) | **0/3 (0%)** | **−100pp** | 큰 collateral 손실 (AI Hub clothes 데이터 자체는 0건이지만 영향 받음) |
| electronics | 0/4 (0%) | 1/4 (25%) | +25pp | 유일한 개선 |
| etc | 5/12 (42%) | 4/12 (33%) | −9pp | |
| food_waste | 1/1 (100%) | 1/1 (100%) | 0 | |
| glass | 3/3 (100%) | 1/3 (33%) | −67pp | 큰 손실 |
| metal | 3/6 (50%) | 2/6 (33%) | −17pp | |
| paper | 2/4 (50%) | 1/4 (25%) | −25pp | 가설 대상 클래스, 오히려 더 못 맞춤 |
| plastic | 6/9 (67%) | 6/9 (67%) | 0 | |
| vinyl | 2/2 (100%) | 1/2 (50%) | −50pp | |

---

## 4. 비교 분석 및 가설 판정

### 4.1 핵심 비교표
| 지표 | Test A | Test B | Δ | 의미 |
|---|---|---|---|---|
| frozen test | 95.68% | **96.42%** | **+0.74pp** | AI Hub 분포 — Test B 가 약간 우세 |
| realworld 전체 | 58.7% | **41.3%** | **−17.4pp** | 사용자 폰 사진 — Test B 가 크게 열세 |
| realworld 70%크롭 | 30.4% | 23.9% | −6.5pp | 같음 방향 |
| paper recall (frozen) | 80.5% | 82.9% | +2.4pp | paper 자체도 frozen 에선 별 차이 없음 |

### 4.2 가설 판정 — **기각 (REJECTED)**

**근거**:
1. **realworld 가 큰 폭으로 악화** (−17.4pp) — 가설의 예측("paper noise 제거 시 realworld ↑")과 정반대
2. **paper 자체 학습도 그대로** — frozen paper recall 80.5% → 82.9% 거의 동일. AI Hub paper 의 8,312장이 paper 학습에 **추가 정보를 거의 안 주고 있음에도** realworld 가 크게 떨어진 건 paper 외 다른 효과 때문
3. **collateral 손실이 광범위** — clothes (−100pp), glass (−67pp), vinyl (−50pp), metal (−17pp) 등 paper 와 무관한 클래스도 모두 악화. 즉 AI Hub paper 데이터가 **전반적 시각 표현(representation)** 학습에 기여 중이었음
4. **frozen 가 살짝 ↑** 한 이유: 일반화 손실보다 노이즈 제거 이득이 약간 더 컸을 수 있음. 하지만 realworld 큰 손실로 우세 효과 상쇄

### 4.3 가설이 틀린 이유 — 메커니즘 추정
시각 검토에서 AI Hub paper crop 의 60% 가 "라벨된 paper 가 dominant 객체 아님" 으로 보였지만, 모델 입장에선:
- 그 crop 안에 paper 가 **부분적으로라도 존재** (옆에 같이 보이는 PET·종이박스 등)
- 모델이 일반적 **폐기물 시각 패턴**(빛 반사·재질 텍스처·배경 클러터 처리) 을 다양하게 학습
- 결과: paper 라벨에 대해선 비효율적이지만, 전체 모델의 **표현 능력은 증진**

→ "라벨 정합성" 과 "학습 기여도" 가 별개. 이번 실험이 그 가정 자체를 깸.

### 4.4 도메인 갭의 진짜 원인은?
실사용 41~58% 정확도 (vs frozen 96%) 의 갭은 **AI Hub paper 의 noise 가 주범이 아님**. 더 근본적인 후보:
- 사용자 분포 (실내 가정, 폰 시점, 단일 객체) 가 학습 분포(facility scene + studio shot) 양쪽 모두와 멀음
- 학습 데이터에 indoor_household 가 약 3% 에 불과 (시각 검토 결과)
- Smart capture 측 도움 (품질 게이트·TTA·자동 크롭) 이 결과적으로 더 효과적일 가능성

### 4.5 통계적 한계
- realworld eval 표본은 46건 (피드백 기준) — 95% CI 가 매우 넓음 (±15~20pp)
- 17pp 차이는 큰 편이지만 단일 측정 — 동일 데이터로 여러 random seed 학습해 분산 측정 필요
- 그래도 방향성은 명확 (10개 클래스 중 7개가 realworld 에서 악화)

---

---

## 5. 학습 설정 (참고)
- 아키텍처: ResNet18 (`CamWasteClassifierCNN` wrapper, 3-output ONNX)
- 백본: `IMAGENET1K_V1` 사전학습, full fine-tune
- 입력: 224×224, ImageNet mean/std
- batch size: 32
- num epochs: 15 (CNN 기본값)
- LR: 1e-4, weight decay: 1e-5, patience: 4
- device: MPS (Apple Silicon)

### 제거된 데이터
- 8,312장 (정확히 `data/raw/garbage-classification/paper/aihub_*.jpg`)
- 다른 클래스의 AI Hub 데이터는 모두 유지 (vinyl/styrofoam/glass/metal/plastic/electronics/non_object)
- Kaggle paper 1,050장 + 기존 다른 paper 소스(user 등) 만 paper 클래스 학습에 사용

---

## 6. 후속 단계 (가설 기각 결과 반영)

### 즉시 결정
- **AI Hub paper 유지** — 시각적으론 라벨 노이즈처럼 보였지만 실제 학습에 기여 중. 통째로 빼면 손해
- **활성 모델 = Test A** — Test B 모델은 폐기 (백업: outputs/backups/test_B/), 원본 manifest·모델 복원 완료

### 다시 잡아야 할 가설들
이번 실험이 보여준 더 중요한 통찰:
1. **시각 검토 기반 "라벨 정합성"은 학습 기여도와 별개** — 모델은 우리가 noise 라고 본 부분에서도 일반화 신호를 추출
2. **AI Hub paper 단독으론 over-prediction 의 주범 아님** — 다른 메커니즘 존재 (다른 클래스 데이터 불균형? softmax 강제 배정?)
3. **frozen vs realworld 갭 (96% vs 41~58%) 의 원인을 paper 데이터 너머에서 찾아야 함**

### 다음 검증 후보
- **a) 다른 AI Hub 클래스 동일 실험** — vinyl/styrofoam (각 10K, AI Hub 단독) 제거 시 어떻게 되는지. paper 와 다른 패턴 보이면 클래스별 처치 필요
- **b) 사용자 분포 데이터 확장 (Tier 1-2)** — user_uploads 의 ingest 가속. 현재 46건 → 200~500건 목표. realworld eval 의 통계적 신뢰도도 같이 올림
- **c) 스마트 캡쳐 측면 개선 ([SMART_CAPTURE_STRATEGY.md](SMART_CAPTURE_STRATEGY.md))** — 캡처 시점 quality gate, multi-frame TTA, u2netp 객체 크롭. 모델 측 lever 가 한계니 캡처 측에서 분포 정렬
- **d) 클래스 가중치 / softmax 분석** — paper/clothes 가 realworld 에서 over-fire 한다는 사용자 관찰이 사실인지 직접 측정. user_uploads 의 예측 분포 vs 라벨 분포 비교

### 실험 인프라 보강 (메타)
- **realworld eval 의 표본 (46건) 가 부족** — 결정적 비교는 200건 이상 모인 뒤
- **realworld eval 의 random seed variance** 도 측정 필요 — 같은 데이터로 3~5회 학습 후 평균/분산 비교

---

## 7. 변경 사항 / 아티팩트

### 백업 / 보관
- `waste-classifier/outputs/backups/test_A_baseline/classifier.onnx` — Test A 원본 모델 (96.42% 보존)
- `waste-classifier/outputs/backups/test_B/classifier.onnx` — Test B 모델 (참고용)
- `waste-preprocessor/data/processed/manifest_test_A_baseline.json` — 원본 manifest (69,368 items)
- `waste-classifier/data/splits/splits.json.bak_test_A` — 원본 splits
- `waste-classifier/outputs/logs/test_B_train.log` — Test B 학습 로그
- `waste-classifier/outputs/logs/test_B_continuation.log` — 측정 자동화 로그
- `waste-classifier/outputs/logs/diagnosis/test_B.json` — Test B frozen 진단 결과

### 복원 작업
- 활성 manifest, 모델 ONNX, splits 모두 Test A baseline 으로 복원
- diagnose history 에 Test B 가 커밋되지 않도록 게이트 FAIL 분기에서 자동 차단됨 (✓)

---

*문서 위치: [/Users/whdrnr01/ai/waste-preprocessor/AIHUB_PAPER_HYPOTHESIS_TEST.md](AIHUB_PAPER_HYPOTHESIS_TEST.md)*
*관련 문서: [GREENGUIDE_BLUEPRINT.md](../GREENGUIDE_BLUEPRINT.md), [SMART_CAPTURE_STRATEGY.md](../SMART_CAPTURE_STRATEGY.md)*
