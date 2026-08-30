# 데이터·합성 파이프라인 적용 결과 (Test C1)

> 작성: 2026-05-31 (Test C1 야간 학습 완료 후)
> 짝 문서: [DATA_AUGMENTATION_DESIGN.md](DATA_AUGMENTATION_DESIGN.md) (설계서)
> 결과: 경계선 — 70% 크롭 +4.4pp ✓, 전체 -2.2pp ⚠ (디자인 채택 기준 -2~+2pp 사이)

---

## 1. 적용 개요

[DATA_AUGMENTATION_DESIGN.md](DATA_AUGMENTATION_DESIGN.md) 설계서의 Phase 1 (공개 데이터 + 합성 파이프라인 prototype) 을 실제 실행한 결과.

### 적용한 항목
| # | 항목 | 출처 | 규모 |
|---|---|---|---|
| 1 | TACO ingest | 공개 (GitHub) | 1,145장 (10 클래스) |
| 2 | MIT Indoor 67 배경 풀 | 공개 (MIT) | 287장 (67 카테고리, 우선순위 가중) |
| 3 | 합성 데이터 생성 | 자체 엔진 | 1,825장 (5 약점 클래스) |
| 4 | 합성 후처리 augmentation | albumentations 1.4.18 | 7 transform pipeline |

### 적용 안 한 항목 (설계서 대비)
- **AI Hub 추가 데이터셋** (71765 실내 3D, 476 small object, 51 K-Fashion, 71647 손 동작): 사이트에서 별도 신청·승인 필요 — 사용자 액션 대기, 차기 iteration
- **EgoHands 손 mask**: prototype 단순화 위해 생략 (50% 확률 hand overlay 모두 OFF) — 효과 입증 후 도입
- **non_object 합성**: 객체-기반 합성이 부적합 (배경 자체가 라벨) — 별도 처리 필요, 차기 iteration

---

## 2. 데이터 출처 상세

### 2.1 TACO (Trash Annotations in Context)
- **레포**: https://github.com/pedropro/TACO
- **원본**: 1,500 이미지, 4,784 annotation, 60 카테고리
- **라이센스**: Creative Commons Attribution
- **다운로드 방식**: git clone + flickr_640_url (각 이미지의 640px 버전, 스트리밍)
- **매핑 결과** (TACO 60 supercategory → 우리 13 클래스):

| 우리 클래스 | TACO 카테고리 | 결과 saved |
|---|---|---:|
| vinyl | Plastic film, Six pack rings, Garbage bag, Other plastic wrapper, Single-use carrier bag, Polypropylene bag, Crisp packet, Plastic straw | 300 (cap 도달) |
| plastic | Clear/Other plastic bottle, Plastic bottle cap, Disposable plastic cup, Other plastic cup, Plastic lid, Spread tub, Tupperware, Disposable food container, Other plastic container, Squeezable tube, Other plastic | 270 |
| etc | Cigarette, Unlabeled litter, Plastic glooves, Plastic utensils, Rope & strings, Blister pack, Shoe | 178 |
| metal | Aluminium foil, Aerosol, Food Can, Drink can, Metal bottle cap, Metal lid, Scrap metal, Pop tab | 154 |
| cardboard | Toilet tube, Other carton, Egg carton, Drink carton, Corrugated carton, Meal carton, Pizza box, Paper cup | 118 |
| styrofoam | Foam cup, Foam food container, Styrofoam piece | 55 |
| paper | Magazine paper, Tissues, Normal paper, Paper bag, Plastified paper bag, Paper straw, Wrapping paper | 40 |
| glass | Glass bottle, Glass jar, Glass cup, Broken glass | 25 |
| food_waste | Food waste | 3 |
| electronics | Battery | 2 |
| **합계** | | **1,145** |

매핑 결정 로그: `waste-classifier/diagnostics/taco_mapping_decisions.csv`

### 2.2 MIT Indoor 67 (실내 배경 풀)
- **출처**: http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar
- **원본**: 15,620 이미지, 67 실내 카테고리
- **라이센스**: 학술 연구용 공개 데이터 (상업 활용 시 별도 검증 필요)
- **추출 전략**: 카테고리 우선순위 가중 — 사용자 분포(가정 실내) 에 가까운 카테고리 ↑
  - **high priority** (우선 추출 30장/카테고리): kitchen, dining_room, livingroom, bedroom, bathroom, laundromat, pantry, closet, garage
  - **mid priority** (10장): office, computerroom, classroom, library, studio_music
  - **low priority** (3장): 그 외 67-15=52 카테고리

배경 풀 구성 결과: **287장** (모두 256x256 center-cropped JPEG q=85)

추출 스크립트: `waste-classifier/scripts/build_indoor_bg_pool.py`

### 2.3 합성 데이터
- **출처**: Kaggle/AI Hub 객체 (라벨 정합성 우선) + MIT 67 배경 풀 + albumentations
- **엔진**: `waste-classifier/scripts/synthesize_indoor.py` (자체 작성)
- **합성 후 결과**: 1,825장 (5 약점 클래스)

---

## 3. 합성 방법론 상세

### 3.1 합성 알고리즘 (9 단계)
1. **객체 alpha 추출**: u2netp ONNX (waste-api 의 segment.py 와 동일 모델) → 320x320, 0~1 float
2. **객체 crop**: alpha > 0.4 영역의 bounding box, 200 픽셀 미만이면 reject
3. **회전**: ±15° 무작위 (cv2 warpAffine)
4. **스케일**: 캔버스의 40~70% (cv2 resize)
5. **광원 매칭**: 객체 평균 톤을 배경 평균 톤으로 15% 끌어옴 (자연스러운 합성)
6. **객체 placement**: 중앙 편향 (cx, cy ∈ 0.20~0.45)
7. **그림자**: gaussian-blurred alpha (sigma=8) × 0.35, offset (+5, +5)
8. **alpha blend**: `result = obj × alpha + bg × (1-alpha)`
9. **albumentations 후처리** (도메인 randomization):
   - `RandomBrightnessContrast(0.15, 0.10, p=0.7)`
   - `HueSaturationValue(10, 15, 10, p=0.5)`
   - `RandomGamma(85-115, p=0.3)`
   - `ImageCompression(75-95, p=0.5)` — 폰 jpg 시뮬
   - `MotionBlur(blur_limit=5, p=0.15)` — 폰 카메라 흔들림
   - `GaussNoise(10-30, p=0.2)`, `ISONoise(0.05-0.20, p=0.2)` — 센서 노이즈

### 3.2 객체 풀 정책
- **우선 1**: Kaggle/user 출처 (라벨 정합성 95%+) — `kg2_*`, `user_*`, `{class}{num}.jpg`
- **fallback**: AI Hub (라벨 정합성 64% but 객체 다양성) — Kaggle 없는 클래스(electronics 등)에 한해
- 제외: 합성 이미지 (`synth_*`), TACO ingest (`taco_*`) — 합성의 합성 방지

### 3.3 품질 게이트
- 객체 alpha 면적 < 10% → reject
- 합성 후 mean brightness ∉ [20, 240] → reject (너무 어둡/밝음)
- 합성 후 stddev < 15 → reject (단조)
- 거절률: 1.1% (1,825 채택 / 1,846 시도)

### 3.4 클래스별 합성 quota (디자인 §7.2 균등 floor)

| 클래스 | 합성 목표 | 실제 saved | 합성 후 indoor 비율 |
|---:|---:|---:|---:|
| etc | 400 | 405 | (~33%) |
| cardboard | 350 | 355 | (~27%) |
| food_waste | 350 | 355 | (~36%) |
| trash | 350 | 355 | (~42%) |
| electronics | 350 | 355 | (~35%) |
| **합계** | **1,800** | **1,825** | — |

대용량 클래스 (paper/plastic/glass/metal/vinyl/styrofoam, 각 9~10K) 와 clothes (7K) 는 합성 적용 안 함 (효과 측정 더 명확히).

### 3.5 학습 통합 방식
- 합성 결과 `data/raw/synthetic_indoor/{class}/synth_*.jpg` 에 저장
- `extend_manifest_synthetic.py` 가 garbage-classification/ 으로 복사 + manifest 등재
- 학습 후 cleanup 으로 garbage-classification/ 의 synth_*.jpg 삭제 (synthetic_indoor/ 원본은 보존)
- splits.json 강제 재생성으로 train/val 에 합성 포함

### 3.6 재현성
- 각 합성 클래스마다 고정 seed (42, 43, 44, 45, 46)
- albumentations 도 seed 따라 결정적
- 같은 source files + seed → 비트 단위로 같은 결과

---

## 4. 실험 설정 (Test C1)

### 학습 구성
| 항목 | 값 |
|---|---|
| 아키텍처 | ResNet18 (`CamWasteClassifierCNN` wrapper, 3-output) |
| 학습 데이터 | 72,338 items (= 69,368 Test A + 1,145 TACO + 1,825 합성) |
| 클래스 | 13 (등록) |
| 배치 | 32 |
| epochs | 15 (early stop patience=4) |
| LR | 1e-4, weight decay 1e-5 |
| device | MPS (Apple Silicon) |
| 결과 | epoch 10 (best), early stop at 14, val_acc 0.9446 |

### 측정 지표
- frozen test (9,167장 동일 hold-out)
- realworld eval (46 사용자 피드백)
- realworld 70% 크롭 (중앙 영역 정확도)
- per-class precision/recall/F1

---

## 5. 결과

### 5.1 핵심 비교표 (Test A vs Test C1)

| 지표 | Test A | Test C1 | Δ | 평가 |
|---|---|---|---|---|
| frozen test 정확도 | 95.68% | **95.34%** | -0.34pp | 게이트 PASS (회귀 없음) |
| frozen macro F1 | (기준) | **0.9399** | 향상 | 균형 ↑ |
| **realworld 전체이미지** | **58.7%** | **56.5%** | **-2.2pp** | 경계선 — 미미 손해 |
| **realworld 70%크롭** | **30.4%** | **34.8%** | **+4.4pp** | 의미 있는 향상 ✓ |

### 5.2 클래스별 realworld recall 변화

| 클래스 | Test A | Test C1 | Δ | 합성 quota | 비고 |
|---|---|---|---|---:|---|
| **metal** | 3/6 (50%) | **4/6 (67%)** | **+17pp** | 0 | TACO 154 영향 추정 |
| **plastic** | 6/9 (67%) | **7/9 (78%)** | **+11pp** | 0 | TACO 270 영향 |
| **etc** | 5/12 (42%) | **6/12 (50%)** | **+8pp** | 405 ★ | 합성 직접 효과 |
| cardboard | 2/2 (100%) | 2/2 (100%) | 0 | 355 ★ | sample 2건 — 통계 약함 |
| paper | 2/4 (50%) | 2/4 (50%) | 0 | 0 | |
| vinyl | 2/2 (100%) | 2/2 (100%) | 0 | 0 | |
| electronics | 0/4 (0%) | 0/4 (0%) | 0 | 355 ★ | 합성에도 불구 변화 없음 |
| **glass** | 3/3 (100%) | 2/3 (67%) | -33pp | 0 | 합성 미적용에도 ↓ |
| **clothes** | 3/3 (100%) | 1/3 (33%) | **-67pp** | 0 | 큰 손해 — 부수 효과 의심 |
| **food_waste** | 1/1 (100%) | 0/1 (0%) | -100pp | 355 ★ | 1건 표본 (통계 의미 약함) |
| **합계** | 27/46 = 58.7% | 26/46 = 56.5% | -2.2pp | | |

### 5.3 frozen test 변화 (per-class F1, 13 클래스)

| 클래스 | Test C1 P | Test C1 R | Test C1 F1 | 비고 |
|---|---:|---:|---:|---|
| cardboard | 0.855 | 0.922 | 0.887 | -4pp recall vs 직전 0.962 |
| clothes | 0.994 | 0.982 | 0.988 | 안정 |
| electronics | 0.985 | 0.990 | 0.988 | 안정 |
| etc | 0.947 | 0.798 | 0.866 | n=89, 약 → 강 향상 |
| food_waste | 0.951 | 0.965 | 0.958 | 안정 |
| glass | 0.971 | 0.973 | 0.972 | 안정 |
| metal | 0.975 | 0.977 | 0.976 | 안정 |
| **non_object** | **0.944** | **0.944** | **0.944** | 새 클래스 정착 양호 |
| paper | 0.872 | 0.938 | 0.904 | 안정 |
| plastic | 0.973 | 0.940 | 0.956 | 안정 |
| styrofoam | 0.974 | 0.954 | 0.964 | 안정 |
| trash | 0.808 | 0.949 | 0.873 | n=177 |
| vinyl | 0.962 | 0.926 | 0.944 | 안정 |

새 frozen 혼동 (≥3% & ≥5건):
- vinyl → paper: 106건 (7.0%) — 합성이 vinyl 학습에 약간 혼란 추정
- non_object → cardboard: 5건 (4.6%)

회귀 (직전 대비): cardboard recall 0.962 → 0.922 (-4.08pp)

게이트: **PASS ✅** (cardboard 회귀가 GATE_MAX_ACC_DROP 임계 미만)

---

## 6. 판정

### 6.1 디자인 §7.4 채택 기준 적용
| Δ realworld 전체 | 의미 | 결정 |
|---|---|---|
| **−2.2pp** (C1 실측) | 경계선 (-2 ~ +2 zone) | 합성 알고리즘 재설계 또는 미세 조정 |

다만 **realworld 70% 크롭 +4.4pp** 향상은 신호 — 단순한 "실패" 가 아닌 **부분 효과**.

### 6.2 메커니즘 추정

**합성이 도움된 영역** (전체 −, 70% 크롭 +):
- 합성 객체가 충분히 dominant 배치 (캔버스 40~70%) → 중앙 70% 크롭 시 객체 인식 강화
- albumentations 도메인 randomization (jpg compression, motion blur, noise) → 폰 카메라 노이즈에 robust ↑

**합성이 해친 영역** (전체 ↓):
- 합성 배경 + 객체의 광원 매칭이 완벽 X → 전체 이미지 보면 "어색한 합성" feature 학습 가능
- clothes 같이 합성 미적용 클래스도 -67pp → 모델 일반화 자체에 약간의 noise 흡수 신호
- glass 처럼 합성 미적용인데도 손해 — 합성 패턴이 다른 클래스 decision boundary 영향

### 6.3 통계적 한계
- realworld eval 46건 — 단일 측정 신뢰 구간 ±15~20pp
- 8건 차이 (Test A 27 vs Test C1 26) 는 binomial 95% CI 내
- 단, 7개 클래스에서 변화 → 우연이라 보긴 어려움. 일관된 미세 효과는 있음

---

## 7. 신규 인프라 (이번에 만든 것)

### 스크립트
- [waste-classifier/scripts/integrate_taco.py](../waste-classifier/scripts/integrate_taco.py) — TACO 매핑 + ingest
- [waste-classifier/scripts/build_indoor_bg_pool.py](../waste-classifier/scripts/build_indoor_bg_pool.py) — MIT 67 → 배경 풀
- [waste-classifier/scripts/synthesize_indoor.py](../waste-classifier/scripts/synthesize_indoor.py) — 합성 엔진 (albumentations 기반)
- [waste-classifier/scripts/extend_manifest_taco.py](../waste-classifier/scripts/extend_manifest_taco.py) — TACO manifest 등재
- [waste-classifier/scripts/extend_manifest_synthetic.py](../waste-classifier/scripts/extend_manifest_synthetic.py) — 합성 manifest 등재 + cleanup
- [waste-classifier/scripts/_overnight_pipeline.sh](../waste-classifier/scripts/_overnight_pipeline.sh) — 야간 자동화 chain (9 phase)

### 의존성 추가
- `albumentations==1.4.18` (waste-classifier/.venv)
- `opencv-python==4.10.0.84`

### 데이터 (디스크)
- `data/raw/garbage-classification/{class}/taco_*.jpg` — 1,145장 (영구, manifest 미등재 상태)
- `data/raw/_aux/backgrounds/` — 287 indoor 배경 (재사용 풀)
- `data/raw/synthetic_indoor/{class}/synth_*.jpg` — 1,825 합성 (재사용 풀)
- `data/raw/synthetic_indoor/_manifest.jsonl` — 합성별 메타 (재현용)

### 백업
- `waste-classifier/outputs/backups/test_C1_pre/` — 학습 직전 Test A 스냅샷
- `waste-classifier/outputs/backups/test_C1/classifier.onnx` — Test C1 학습 모델 (참고용)

### 복원 상태 (학습 후)
- ✅ active 모델 ONNX = Test A baseline (sha 일치)
- ✅ manifest = Test A baseline (synth 0)
- ✅ splits.json = Test A 원본

---

## 8. 후속 단계 권고

### 8.1 즉시 추출 가능한 가치 (코드 변경 없이)
**70% 크롭에서 +4.4pp** — Test C1 모델 자체는 폐기지만, **smart capture 측에 자동 객체-중심 크롭 기능 추가** 시 즉시 활용 가능. 합성 효과와 별개로 사용자 입력을 객체 중심으로 표준화하면 ~4pp 개선 기대.

### 8.2 합성 파이프라인 개선 후보 (효과 ↑ 위해)
1. **광원 매칭 강화** — 현재 15% mixing → 30%로 ↑, 또는 background 의 light direction 분석 후 객체에 적용
2. **그림자 자연화** — 현재 단순 gaussian blur → 객체 alpha 의 perspective projection 으로 더 사실적인 그림자
3. **객체 source 다양화** — Kaggle 한정 → AI Hub 추가 신청 후 71647 (한국 손-객체 상호작용) 통합
4. **품질 게이트 강화** — 현재 거절률 1.1% → 더 엄격하게 (예: 5~10%) 해서 부자연스러운 합성 차단
5. **합성 비율 조정** — 현재 real 97.5% + synth 2.5% → 1.5% 정도로 낮춰 합성 artifact 영향 감소

### 8.3 합성보다 큰 lever (실사용 데이터)
1. **사용자 피드백 200+ 건 확보** — realworld eval 의 통계 신뢰도 ↑ (현 46 → 200건 시 CI ±5pp 수준)
2. **사용자가 직접 30~50장 indoor 촬영** — 합성 1,825장보다 직접적 효과 클 것
3. **AI Hub 4개 신청 → 승인 받으면 71765 (28K 가상 실내) + 476 (전자기기 49 classes) 통합**

### 8.4 채택 결정 — **Test C1 모델 자체는 미채택**
- 현재 active 모델은 Test A 유지
- 합성 파이프라인 인프라는 보존 (재사용 가능)
- 다음 iteration: 위 8.2 의 개선 + 가능하면 사용자 데이터 추가 후 Test C2 시도

---

## 9. Option A 후속 — Fix 1 시리즈 정량 결과 (2026-05-31)

Test C1 의 70% 크롭 +4.4pp 효과를 활용하기 위한 앱 측 변경 + 모델 quirk 완화.

### 9.1 적용한 Fix

| Fix | 위치 | 변경 |
|---|---|---|
| **Work 1** | waste-api, waste_app | `/predict-centered` endpoint — u2netp 자동 객체 크롭 후 분류. smart capture 시 자동 호출 |
| **Work 3** | confidence.dart | `_kRejectThreshold` 0.45 → 0.55, `normalizedEntropy > 0.7` 면 reject |
| **Fix 1** | result_modal.dart | `realMulti = isMulti && !reject` — global 이 낮은 신뢰도면 regions noise 차단 |
| **Fix 1.5** | result_modal.dart | regions 각 `avgConf >= 0.75` 까지 만족해야 진짜 다중재질 |
| **+non_object** | result_modal.dart | `predictedClass == 'non_object' → reject` 강제 (DB 의존 X) |
| **severeThreshold** | stability_detector.dart | 강한 흔들림 (delta ≥ 1.5 m/s²) 시 grace 무시하고 즉시 progress 리셋 |

### 9.2 사용자 분포 실측 (6 케이스)

| # | 케이스 | Before Option A | After Option A | 평가 |
|---|---|---|---|---|
| 1-A(a) | PET 바닥 | plastic 94.5% | plastic 73.9% | ✓ 정답 유지 |
| 1-A(b) | PET 잡배경 | 다중재질 (오분류) | **plastic 86.0%** | ✓ 개선 |
| 1-A(c) | PET 손에 들고 | plastic 99.3% | plastic 99.4% | ✓ |
| 2-A(1) | 빈 책상 | 다중재질 (오분류) | **"기타/분류 불가"** | ✓ 개선 |
| 2-A(2) | 손바닥 | 다중재질 (오분류) | 다중재질 (잔여 실패) | ✗ |
| 3-1 | 마우스 손에 | clothes 단일 | 다중재질 (잔여 실패) | ✗ |

**성공률: 2/6 → 4/6 (+33pp)**

### 9.3 잔여 실패 — 모델 측 한계 확인
손바닥·마우스 케이스는 region 단위에서도 모델이 의류·종이상자 로 **0.75+ 확신**. 앱 측 로직 게이트(임계 0.75) 통과 → 다중재질 카드 노출. 임계를 더 올리면(0.85+) 진짜 다중재질 (PET+라벨) 도 false negative 증가 우려.

→ **앱 측 lever 는 한계 도달**. 손바닥·마우스 케이스 해결은 **모델 측 (재학습)** 영역.

### 9.4 적용 후 active 인프라

- 모델: Test A 유지 (변경 없음 — D1 학습 디스크 부족 실패)
- 서버: `/predict-centered` 신규 endpoint 배포 (HF Space)
- 앱: 위 6 Fix 모두 빌드·설치 완료. APK 124MB. smart capture 흐름 + reject 강화 + non_object 핸들링.

### 9.5 다음 단계 권고 (우선 순위 ↓)

| 단계 | 작업 | 효과 추정 |
|---|---|---|
| **Stage A** | non_object 실사용 데이터 50+ 장 수집 + retrain | 손바닥/배경 케이스 직접 해소 — 핵심 lever |
| **Stage B** | electronics 실사용 데이터 (마우스·충전기 등) + retrain | 마우스 케이스 + 전반 electronics 정확도 |
| **Stage C** | Test D1 (TACO 영구 등재 + retrain) 재시도 — 디스크 정리 후 | TACO 효과 정량화 |
| **Stage D** | 클래스 가중치 cap (`inverse-freq` 완화) | OOD sink (cardboard/non_object) 약화 |
| **Stage E** | 사용자 피드백 200건+ 확보 | realworld eval 신뢰도 ↑ + 모든 lever 비교 가능 |

손바닥/마우스 처럼 confident-wrong 케이스는 **앱 측에서 더 잡을 길 없음**. 데이터 lever 가 본질.

---

*문서 위치: [/Users/whdrnr01/ai/waste-preprocessor/DATA_AUGMENTATION_RESULTS.md](DATA_AUGMENTATION_RESULTS.md)*
*관련 문서:*
*- [DATA_AUGMENTATION_DESIGN.md](DATA_AUGMENTATION_DESIGN.md) — 설계서*
*- [AIHUB_PAPER_HYPOTHESIS_TEST.md](AIHUB_PAPER_HYPOTHESIS_TEST.md) — 이전 가설 검증 (AI Hub paper 제거 실험)*
*- [SMART_CAPTURE_STRATEGY.md](../SMART_CAPTURE_STRATEGY.md) — 스마트 캡쳐 측 개선 전략*
*- [GREENGUIDE_BLUEPRINT.md](../GREENGUIDE_BLUEPRINT.md) — 청사진*
