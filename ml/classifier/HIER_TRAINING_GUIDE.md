# 계층 CNN 학습 가이드 — 클래스별 데이터가 모델이 되기까지

> 작성: 2026-07-13 (활성 모델 v4 = `v20260710_194710` 기준)
> [lab.md](lab.md)(구 6클래스 flat 교육문서)의 계층판 후속.
> 관련: [GREENGUIDE_BLUEPRINT.md](../GREENGUIDE_BLUEPRINT.md) · [DIAGNOSIS_PROCESS.md](../DIAGNOSIS_PROCESS.md)

이 문서는 **"어떤 클래스가, 어떤 데이터로, 어떻게 학습되는가"** 를 실제 수치·코드
위치와 함께 추적하고, 마지막에 **실사용 정확도가 낮은 클래스와 데이터 보강
우선순위**를 정리한다.

---

## 0. 구조 한 장 요약

```
출력 공간: fine 25클래스 (모델이 실제 예측)
롤업 공간: coarse 14대분류 — P(대분류) = Σ P(자식 세부)   ← 학습·서빙 공용
감독 종류: ① fine 감독 (라벨이 세부까지 확실)  → 표준 CE loss
          ② coarse 감독 (대분류만 확실: glass/plastic/styrofoam 혼재 데이터)
                                              → 롤업 logsumexp NLL loss
```

정의 위치: [greenguide_classifier/taxonomy.py](greenguide_classifier/taxonomy.py) — `TAXONOMY`(계층), `LEGACY_LABEL_SUPERVISION`
(구 manifest 라벨 → 감독), `STAGING_DIR_SUPERVISION`(fine-staging 폴더 → 감독),
`GUIDANCE_GROUPS`(안내-동일 형제).

---

## 1. 클래스별 학습 데이터 현황 (v4 실측)

### 세부(fine) 25클래스

| fine | 대분류 | train | (합성) | frozen test | f1 | guidance-safe f1 | 활성 |
|---|---|---:|---:|---:|---:|---:|:--:|
| metal | metal | 12,319 | 0 | 2,333 | 0.955 | 0.955 | ✅ |
| pet | plastic | 9,538 | 658 | 1,687 | 0.962 | 0.962 | ✅ |
| glass_deposit | glass | 9,114 | 668 | 1,544 | 0.927 | 0.927 | ✅ |
| paper_other | paper | 8,872 | 0 | 1,632 | 0.902 | 0.902 | ✅ |
| glass_brown | glass | 7,667 | 623 | 1,319 | 0.854 | 0.921 | ✅ |
| vinyl_clean | vinyl | 7,124 | 0 | 1,526 | 0.949 | 0.949 | ✅ |
| light_bulb ⚠️ | trash | 6,259 | 667 | 1,200 | 0.993 | 0.993 | ✅ |
| paper_cup | paper_pack | 6,200 | 658 | 1,053 | 0.808 | 0.900 | ✅ |
| glass_clear | glass | 6,190 | 661 | 1,057 | 0.774 | 0.877 | ✅ |
| glass_green | glass | 5,993 | 643 | 1,000 | 0.908 | 0.953 | ✅ |
| glass_etc | glass | 5,531 | 0 | 1,059 | 0.865 | 0.955 | ✅ |
| clothes | clothes | 5,106 | 0 | 1,094 | 0.987 | 0.987 | ✅ |
| **battery** | hazardous | 4,283 | 667 | 670 | 0.937 | 0.937 | ✅ |
| electronics | electronics | 4,229 | 0 | 873 | 0.989 | 0.989 | ✅ |
| styrofoam_white | styrofoam | 4,013 | 652 | 630 | 0.865 | 0.865 | ✅ |
| **carton** | paper_pack | 3,559 | 650 | 558 | 0.647 | **0.805** | 🔒 롤업 |
| styrofoam_dirty ⚠️ | styrofoam | 3,566 | 0 | 671 | 0.889 | 0.889 | ✅ |
| styrofoam_color | styrofoam | 3,518 | 0 | 739 | 0.991 | 0.991 | ✅ |
| vinyl_dirty ⚠️ | vinyl | 2,664 | 0 | 494 | 0.814 | 0.814 | ✅ |
| **cardboard** | paper | **678** | 0 | 183 | 0.881 | 0.881 | ✅* |
| **food_waste** | food_waste | **661** | 0 | 183 | 0.967 | 0.967 | ✅* |
| **trash_other** | trash | **550** | 0 | 161 | 0.908 | 0.908 | ✅* |
| non_object | non_object | 504 | 0 | 108 | 0.871 | 0.871 | (내부) |
| **etc** | etc | **131** | 0 | 31 | 0.528 | 0.528 | 🔒 |
| plastic_other | plastic | 0 | 0 | 0 | — | — | 🔒 설계상 |

*✅\* = frozen f1 은 좋지만 **표본이 적어 실사용 일반화가 약한 클래스** (아래 §6).

### 대분류 감독(롤업 전용) — 세부 미상 혼재 데이터
| 라벨 | train | 의미 |
|---|---:|---|
| glass | 6,654 | 색상 미상 구 데이터 → 유리 5형제 롤업으로만 감독 |
| plastic | 9,093 | PET 여부 미상 → {pet, plastic_other} 롤업 감독 |
| styrofoam | 6,970 | 흰/컬러/오염 미상 → 3형제 롤업 감독 |

**핵심**: 이 2.3만 장의 "세부 미상" 데이터도 버려지지 않고 롤업 loss 로 세부 head 를
함께 학습시킨다 (아래 §4).

---

## 2. 데이터 소스 → 클래스 매핑 (어디서 온 데이터인가)

| 소스 | 규모 | 채우는 클래스 |
|---|---|---|
| 구 manifest (Kaggle+AI-Hub 71362/140+TACO) | ~7.5만 | 13개 legacy 라벨 → §0 감독 매핑으로 변환 |
| **AI-Hub 71385** (생활폐기물 활용·환류) bbox 크롭 | ~11만 크롭 | battery·carton·paper_cup·유리4색·pet·스티로폼 등 신규 16라벨 + 조건(clean/라벨부착/오염) |
| **AI-Hub 140** 품목 zip | 1.3만 | light_bulb(LED·백열), glass_deposit(소주·맥주병), electronics(폰 5천) |
| **다중객체 합성** (`scripts/synthesize_multiobject.py`) | 8,000 | 10개 클래스 ×800 — 이웃 파편 포함 크롭(서빙 분포 정렬) + carton↔유리 하드네거티브 |
| 사용자 피드백 (user_uploads) | 51 | 정정 라벨별 |

물리 위치: 구 데이터 `greenguide-preprocessor/data/raw/garbage-classification/<라벨>/`,
신규 `.../data/raw/fine-staging/<staging라벨>/` (폴더명→감독은
`STAGING_DIR_SUPERVISION`). 조건은 파일명에 보존: `aihub385_<cond>__…`,
합성은 `synmo_*` (test 진입 금지).

---

## 3. 파이프라인 단계별 (코드 추적)

### 3-1. 아이템 수집 — [greenguide_classifier/hier_dataset.py](greenguide_classifier/hier_dataset.py) `build_hier_items()`
manifest 아이템 + fine-staging 파일을 합쳐 각 아이템에
`{source_path, sup_kind(fine|coarse), sup_slug, sup_idx}` 를 부여.
미지의 라벨이 fine/coarse 공간에 있으면 자동 수용(피드백 fine slug 지원).

### 3-2. 분할 — `build_hier_splits()` (경로 기반, v2 포맷)
- **test 동결**: legacy `frozen_test.json` + `hier_frozen_test.json` 의
  source_path 는 영구 test. 신규 클래스는 클래스당 ≥30 (실데이터만) 동결 충원.
- **합성(synmo_\*) test 금지** — 실데이터 잣대 유지.
- 나머지 감독그룹별 stratified train/val.
- 저장은 **경로 목록**(`format: paths_v2`) — 아이템 목록이 변해도 멤버십 불변.
  (⚠️ v3 사고 교훈: 학습 도중 데이터 추가 → 인덱스 밀림 → 게이트가 잡아 롤백.
   **사이클 중 fine-staging 수정 금지**)

### 3-3. 로딩·증강 — `HierImageDataset`
PIL→RGB→224² bilinear→[0,1]→ImageNet 정규화. train 증강: 좌우 flip 50%,
**grayscale 20%**(색 편향 억제), color jitter 70%. 반환 `(x, is_fine, sup_idx)`.

### 3-4. 모델 — [greenguide_classifier/model.py](greenguide_classifier/model.py) `WasteClassifierCNN(num_classes=25)`
ImageNet 사전학습 ResNet18, fc 만 25-way 교체, 전층 fine-tune.

### 3-5. 손실 — [greenguide_classifier/hier_train.py](greenguide_classifier/hier_train.py) `HierarchicalLoss`
```
fine 아이템:   CE( logits, fine_idx ) × w_fine[fine_idx]
coarse 아이템: -log P(coarse_idx) × w_coarse[coarse_idx]
              P(c) = Σ_{f∈children(c)} softmax(logits)_f   (logsumexp, scatter)
배치 loss = 두 항의 아이템 평균
```
가중치: 감독 공간별 inverse-freq, **median×4 cap**
(coarse 아이템은 자식 수로 나눠 fine 빈도에 근사 반영).

### 3-6. 학습 — `train_hier()`
batch 32 / 최대 15 epoch / Adam 1e-4 / patience 4 / seed 42 / MPS.
체크포인트 선택 점수 = **val 대분류 acc + 0.2×세부 acc** (대분류 우선 원칙).
v4: 에포크 13 best, train 14.1만.

### 3-7. 평가·활성화 — [greenguide_classifier/hier_evaluate.py](greenguide_classifier/hier_evaluate.py)
- 대분류 acc(전체 test, 롤업) / 세부 acc(fine 감독 test 만)
- **활성화 판정**: `test≥30 AND (f1≥0.80 OR guidance_safe_f1≥0.85)`
  — guidance-safe 는 **배출 안내가 같은 형제 혼동을 정답 처리**
  (carton↔paper_cup, 유리 4색; 단 보증금병은 안내 달라 제외).
  근거: 혼동분석 결과 형제 혼동은 사용자 피해 0.

### 3-8. 게이트·루프 — [retrain_hier.py](retrain_hier.py)
```
피드백 수집 → 격리 → 백업 → preprocessor → splits 재생성(frozen 유지)
→ 학습 → 평가 → 게이트(대분류 acc -2pp / 대분류별 recall -5pp 초과 하락 시 FAIL→자동 롤백)
→ PASS: ONNX(3-output)+OOD 프로토타입 export → 자동 승격/강등(Supabase)
→ model_diagnostics 기록 → 실사용 평가(realworld_eval_hier) 자동 실행
```
baseline: `outputs/logs/diagnosis/hier_history.jsonl` (게이트 통과분만 누적).

### 3-9. 서빙까지
export: `outputs/models/cnn_hier/{classifier.onnx, taxonomy.json, ood.npz}` →
waste-api `/predict-hier`(손감지→Stage1→u2netp 크롭→분류(non_object 마스킹)→2단 OOD→신뢰도 게이트)
+ 앱 온디바이스 번들(`assets/models/classifier_hier.onnx`, 동일 롤업·게이트).

---

## 4. 딥러닝 학습 4단계 루프 — 원리 상세

한 배치(32장)가 모델을 실제로 "가르치는" 과정. 이 루프가 v4 기준
**에포크당 4,406 스텝 × 13 에포크 ≈ 5.7만 번** 반복됐다.

```
┌─→ [1] 데이터 입력·예측 (forward)     x(32,3,224,224) → logits(32,25)
│   [2] 평가·채점 (loss)              logits + 정답 → 스칼라 loss
│   [3] 원인 분석 (backpropagation)    loss → 1,120만 개 파라미터별 기여도(gradient)
└── [4] 수정 (optimizer step)          gradient 반대 방향으로 가중치 미세 조정
```

### [1] 데이터 입력 및 예측 (Forward Pass)

**입력 배치** — DataLoader 가 셔플된 train 에서 32장을 뽑아 변환:
```
x:       (32, 3, 224, 224) float32   ← 증강 + ImageNet 정규화 후
is_fine: (32,) {0,1}                 ← 감독 종류 마스크
sup_idx: (32,) int64                 ← fine 인덱스(0..24) 또는 coarse 인덱스(0..13)
```

**증강이 하는 일** (train 에만): 같은 사진이 에포크마다 조금씩 다르게 보인다 —
좌우 flip(50%)은 "방향은 정체성이 아니다", grayscale(20%)은 "색만 보지 말고
형태를 봐라", color jitter(70%)는 "조명이 달라도 같은 물건이다"를 gradient 로
강제하는 장치. 14.1만 장이 실질적으로 수십만 장처럼 작동한다.

**ResNet18 내부에서 벌어지는 일** — 텐서 크기 추적:
```
(32, 3, 224, 224)
 → Conv7×7 stride2 (64필터) + BN + ReLU   → (32,  64, 112, 112)  엣지·색 blob
 → MaxPool3×3 stride2                     → (32,  64,  56,  56)
 → Layer1: BasicBlock×2 (64→64)           → (32,  64,  56,  56)  질감·모서리 조합
 → Layer2: BasicBlock×2 (64→128, /2)      → (32, 128,  28,  28)  부분 패턴 (라벨, 뚜껑)
 → Layer3: BasicBlock×2 (128→256, /2)     → (32, 256,  14,  14)  객체 부품 (병목, 캔뚜껑)
 → Layer4: BasicBlock×2 (256→512, /2)     → (32, 512,   7,   7)  객체 수준 의미
 → GlobalAvgPool                          → (32, 512)            "이 이미지의 요약" = embedding
 → fc: Linear(512 → 25)                   → (32, 25)             fine logits
```
- **BasicBlock** = Conv3×3→BN→ReLU→Conv3×3→BN + **skip connection**(입력을 출력에 더함).
  skip 덕에 18층 깊이에서도 gradient 가 소실되지 않고 앞층까지 전달된다.
- **BatchNorm** 은 각 채널 활성값을 배치 통계로 정규화 — 층이 깊어도 분포가
  안정되어 lr 1e-4 로도 빠르게 수렴.
- **pretrained 의 의미**: 이 1,120만 파라미터는 0에서 시작하지 않는다. ImageNet
  128만 장으로 이미 "세상의 시각 어휘"(엣지→질감→부품)를 배운 상태에서, 우리는
  마지막 어휘 조합만 폐기물 용으로 재조정한다. lab.md 실증: 같은 데이터에서
  MLP 35% vs pretrained CNN 92%.

**출력 logits 예시** (소주병 이미지 1장, v4 모델):
```
logits[glass_deposit]=9.1, [glass_brown]=3.2, [pet]=1.0, ... [battery]=-4.7
```
아직 확률이 아니고, 크기 비교만 의미 있는 "점수". 확률화는 채점 단계에서.

### [2] 평가 및 채점 (Loss)

**softmax — 점수를 확률로**:
```
p_i = exp(z_i) / Σ_j exp(z_j)
위 예: p[glass_deposit] = e^9.1 / (e^9.1 + e^3.2 + …) ≈ 0.997
```

**케이스 A — fine 감독** (정답이 세부까지 확실, 예: 정답 glass_deposit):
```
loss = -log p[정답]  ×  w_fine[정답]
     = -log(0.997) ≈ 0.003          ← 거의 완벽한 예측 → 벌점 미미
반대로 p[정답]=0.05 였다면 -log(0.05) ≈ 3.0  ← 큰 벌점
```
-log 의 성질: 확신하고 틀리면(p→0) 벌점이 무한대로 폭증 — "자신 있는 오답"을
가장 세게 처벌한다.

**케이스 B — coarse 감독** (정답이 "유리"까지만, 색상 미상):
```
P(glass) = p[glass_brown]+p[glass_green]+p[glass_clear]+p[glass_deposit]+p[glass_etc]
loss = -log P(glass) × w_coarse[glass]
```
5형제 중 무엇이라 했는지는 안 묻고, **합이 크기만 하면 통과** — 부분 정보를
정확히 부분만큼만 요구한다. 구현은 수치 안정을 위해 logsumexp
([greenguide_classifier/hier_train.py](greenguide_classifier/hier_train.py) `coarse_log_probs`, scatter_add 로 파이썬 루프 없이).

**클래스 가중치** — 불균형 보정:
```
w[c] = N_total / (25 × N_c)  를 median(w)×4 로 상한
예: metal(12,319장) w≈0.46  vs  trash_other(550장) w≈10.3→cap≈4.6
```
희소 클래스의 실수를 더 크게 채점해 "다수 클래스만 잘 맞히는 게으른 해"를 차단.
cap 이 없으면 etc(131장) 같은 극소 클래스가 학습 전체를 흔든다.

**배치 loss** = 32개 아이템 벌점의 평균 → 스칼라 하나.
v4 실측 궤적: epoch1 train 0.581 → epoch6 0.250 → epoch13 0.148 (val 0.345 부근
정체 = 과적합 경계 → early stop).

### [3] 원인 분석 (Backpropagation)

`loss.backward()` 한 줄이 하는 일: **미분의 연쇄법칙으로 1,120만 개 파라미터
각각에 "네가 이 벌점에 얼마나 기여했나"를 계산**.

출발점 — softmax+CE 의 gradient 는 놀랍도록 단순하다:
```
∂loss/∂z_i = p_i - y_i        (y = 정답 one-hot)

예: 정답 glass_deposit, p=[0.997(dep), 0.002(brown), …]
    ∂L/∂z[deposit] = 0.997-1 = -0.003   ← "deposit 점수 조금만 올려"
    ∂L/∂z[brown]   = 0.002-0 = +0.002   ← "brown 점수 조금 내려"
```
**롤업 감독의 gradient** (이 설계의 핵심 성질):
```
∂(-log P(glass))/∂z_f = p_f - p_f/P(glass)   (f ∈ 유리 형제)
                      = p_f × (1 - 1/P(glass))  < 0
∂/∂z_g = p_g                                  (g ∉ 유리)  > 0
```
→ 유리 형제들의 점수는 **현재 확률 비율대로** 함께 올리고, 비유리는 내린다.
색상 미상 데이터가 "유리다움"은 가르치되 형제 서열엔 간섭하지 않는 이유.

이 출발 gradient 가 fc→GAP→Layer4→…→Conv1 로 거꾸로 흐르며(각 층의 국소 미분과
곱해지며) 모든 conv 필터·BN 파라미터에 도달한다. skip connection 은 이 역류의
고속도로 — gradient 소실 방지의 실체다. PyTorch autograd 가 forward 때 기록한
연산 그래프를 따라 자동 수행.

### [4] 수정 (Optimizer Step — Adam)

gradient 는 "방향"만 준다. 얼마나 움직일지는 Adam 이 파라미터별로 결정:
```
m ← 0.9·m + 0.1·g            (1차 모멘트 — 최근 gradient 의 이동평균 = 관성)
v ← 0.999·v + 0.001·g²       (2차 모멘트 — gradient 크기의 이동평균)
θ ← θ - lr · m̂ / (√v̂ + ε)    (lr=1e-4, m̂·v̂은 bias 보정값)
```
- **관성(m)**: 배치마다 요동치는 gradient 를 평활 — 일관된 방향만 누적 반영
- **적응 보폭(1/√v)**: 자주·크게 흔들리는 파라미터는 조심히, 조용한 파라미터는
  과감히 — 1,120만 개가 각자 다른 보폭으로 움직인다
- **weight decay 1e-5**: 매 스텝 θ를 0.000001% 씩 0으로 잡아당김 — 특정 가중치가
  비대해져 훈련데이터를 암기하는 것(과적합)을 억제
- pretrained fine-tune 에 lr 1e-4 인 이유: 이미 좋은 지점 근처라 크게 움직이면
  ImageNet 지식이 파괴된다(catastrophic forgetting)

한 스텝의 규모감: 파라미터당 이동량 ~1e-5 수준의 미세 조정 × 5.7만 스텝의
누적이 "유리병과 우유팩을 구분하는 능력"이 된다.

### 루프 제어 — 언제 멈추는가

- **에포크 끝 val 평가**: `model.eval()` + `no_grad` — dropout 끔, BN 은 학습된
  통계 사용, gradient 계산 안 함. **val 은 채점만 하지 가르치지 않는다** (그래서
  val 로 모델을 고르는 건 되지만 val 이 train 에 새면 안 됨 — frozen 규율의 이유).
- **체크포인트 선택**: score = val 대분류 acc + 0.2×세부 acc 가 갱신될 때만 저장
  → "대분류 절대 우선" 원칙이 모델 선택 기준에 내장.
- **Early stopping**: patience 4 — val 개선이 4에포크 없으면 중단. v4 는 train acc
  93%↑ 진행 중에도 val 이 멈춰(과적합 개시) 에포크 13에서 종료.
- **재현성**: seed 42 (random/numpy/torch), 결정적 split — 같은 데이터면 같은 결과.

---

## 5. 기술 스택 · 전처리 · 학습 기법 카탈로그

### 5-1. 사용 라이브러리 (실제 버전)

| 영역 | 라이브러리 | 버전 | 역할 |
|---|---|---|---|
| 학습 프레임워크 | **PyTorch** | 2.4.1 | 텐서·autograd·optimizer·DataLoader |
| 모델 zoo | **torchvision** | 0.19.1 | `resnet18(IMAGENET1K_V1)` 사전학습 가중치 |
| 수치 | numpy | 1.26.4 | 배열 연산 공용 |
| 이미지 I/O | **Pillow** | 10.4.0 | 디코드·리사이즈·크롭 (전 구간 표준) |
| 분할·지표 | scikit-learn | 1.5.2 | stratified split, P/R/F1/혼동행렬, (etc_queue) HDBSCAN |
| 중복 제거 | **imagehash** | 4.3.1 | perceptual hash(pHash, 8×8) 근사중복 제거 |
| 모델 교환 | onnx / **onnxruntime** | 1.17.0 / 1.19.2 | export·등가성 검증·서빙/스테이징 추론 |
| 서버 | FastAPI + uvicorn | 0.115.0 | `/predict-hier` 등 |
| 손 감지 | **MediaPipe** | 0.10.18 | 서빙 게이트 0단 (손 면적 ≥0.5 → non_object) |
| 시각화 | opencv-python | 4.10.0.84 | 서버 CAM/영역 렌더 보조 |
| DB/수집 | supabase-py, requests | 2.7.4 / 2.32.3 | 피드백 루프·모델 레지스트리 |
| 앱 추론 | **onnxruntime (Flutter)** | ^1.4.1 | 온디바이스 계층 추론 |
| 앱 이미지 | image (Dart) | ^4.3.0 | 온디바이스 전처리(디코드·리사이즈) |
| 진행/테스트 | tqdm / pytest | 4.66.5 / 8.3.3 | |

**의도적으로 안 쓰는 것**: torchvision.transforms(증강은 텐서 연산 자체구현 —
아래 5-3), albumentations(구 합성 실험에만 사용, 현 파이프라인 무의존),
TensorFlow/TFLite(교환 포맷은 ONNX 단일).

### 5-2. 전처리기(preprocessor) 계보 — 3계층

**(a) 원천 정제 — [greenguide-preprocessor](../greenguide-preprocessor/)** (구 manifest 생성)
```
collect (Kaggle CLI 자동 다운로드)
→ catalog (클래스 폴더 스캔 + 12자리 UUID 부여)
→ cleanse: PIL Image.verify() 손상 제거 + imagehash pHash(hash_size=8) 근사중복 제거
→ manifest.json (라벨·경로 인덱스)
```
학습은 이 manifest 의 `source_path` 로 **원본 JPEG 를 직접** 읽는다
(.npz 벡터 캐시는 구 MLP 시절 유산).

**(b) bbox 크롭 스테이징 — [aihub_71385_staging/](../aihub_71385_staging/)** (신규 데이터)
AI-Hub tar 스트림 → zip 파트 병합없이 직접 읽기(MultiPartFile) → (끊김 시
PK 헤더 스캔 salvage) → 라벨 bbox + **8% 패딩** 크롭 → **min side 64px 필터**
→ 긴 변 256 thumbnail → JPEG q90 저장. 조건(clean/라벨부착/오염)은 폴더/파일명으로 보존.

**(c) 학습 시점 변환 — [greenguide_classifier/hier_dataset.py](greenguide_classifier/hier_dataset.py) `_load_rgb_chw01` + `HierImageDataset`**
```
PIL.open → convert("RGB") → resize(224², BILINEAR) → np/255 → (3,224,224) float32
→ [train만] 증강(5-3) → (x - mean)/std   # ImageNet mean(0.485,0.456,0.406) std(0.229,0.224,0.225)
```
왜 ImageNet 통계인가: ResNet18 사전학습이 이 정규화를 전제로 학습됐다.
같은 분포로 넣어야 사전학습 필터가 의도대로 반응한다. **서빙(waste-api
preprocess.py)·온디바이스(local_inference.dart)도 동일 수식** — train/serve
skew 를 없애는 단일 계약이며, 테스트로 고정돼 있다.

**(＋) 서빙 전용 전처리 — [waste-api/src/preprocess.py](../waste-api/src/preprocess.py)**
`normalize_orientation`: EXIF 회전을 픽셀에 굽고 **GPS 등 메타데이터 제거**
(privacy) → u2netp 객체-인지 크롭(§5-4) → (c) 와 동일 변환.

### 5-3. 증강(augmentation) 기법 — 자체 구현 텐서 연산

torchvision transforms 대신 `[0,1]` 텐서 위 직접 연산
([greenguide_classifier/dataset.py](greenguide_classifier/dataset.py) `_apply_augmentation`, hier 가 재사용):

| 기법 | 확률 | 구현 | 가르치는 불변성 |
|---|---|---|---|
| Horizontal flip | 50% | `torch.flip(dims=[2])` | 좌우 방향 무관 |
| **Random grayscale** | 20% | ITU-R 가중합(0.299R+0.587G+0.114B) 3채널 복제 | **색이 아닌 형태·질감 강제** (색편향 대응 핵심) |
| Color jitter | 70% | brightness ×(1±0.25) → contrast(평균 중심 스케일) → saturation(회색과 보간, 50%) | 조명·카메라 색감 변동 |

val/test 는 증강 없음(잣대 고정). 회전·크롭 증강을 안 쓰는 이유: 입력이 이미
bbox 크롭이라 기하 변형이 라벨 경계를 훼손할 수 있음 — 대신 기하 다양성은
**다중객체 합성(5-4)** 이 담당.

### 5-4. 데이터 수준 기법 (증강의 상위 레벨)

- **u2netp saliency 컷아웃**: 4.4MB ONNX(입력 320², U²-Net 경량판). "두드러진
  객체 vs 배경" 픽셀 마스크 — 합성의 알파 추출과 서빙의 객체-인지 크롭에 공용.
- **다중객체 장면 합성** ([scripts/synthesize_multiobject.py](scripts/synthesize_multiobject.py)):
  실내 배경 471장 + 컷아웃 2~3개 알파합성(+가우시안 그림자) → 각 객체 bbox+**25%
  패딩** 크롭 = **이웃 파편이 섞인 학습 샘플** → 서빙 crop 분포와 정렬.
- **하드 네거티브 페어링**: 합성 장면의 60%를 혼동쌍(carton↔유리색상↔종이컵)
  조합으로 강제 — 경계 학습 집중.
- **합성 test 격리**: `synmo_*` 는 frozen test 진입 금지 (실데이터 잣대).

### 5-5. 학습 기법 요약표

| 기법 | 위치 | 효과 (실측) |
|---|---|---|
| Transfer learning (ImageNet→전층 fine-tune) | model.py | MLP 35%→CNN 92% (lab.md) |
| **계층 롤업 loss** (약감독 활용) | hier_train.py | 세부미상 2.3만 장 재활용 |
| Inverse-freq class weight + median×4 cap | hier_train.py | 극소클래스 붕괴/폭주 방지 |
| Early stopping + 대분류-우선 체크포인트 | hier_train.py | 과적합 컷 (v4: ep13) |
| **Guidance-safe 활성화 지표** | hier_evaluate.py | paper_cup/glass_clear 구제 |
| 회귀 게이트 + 자동 롤백 | retrain_hier.py | v3 오염 차단 실증 |
| OOD 프로토타입(임베딩 metric) 2단 reject | ood + hier_inference | 노이즈→clothes 0.999 차단 |
| non_object 마스킹 (캐스케이드 정합) | hier_inference | 실사용 +5.9pp |
| ONNX 등가성 검증 (torch↔ort <1e-4) | hier_export.py | train/serve 동일성 보장 |

---

## 6. 왜 "혼재 라벨"도 버리지 않는가 (롤업 감독의 가치)

구 glass 데이터 6,654장은 색상을 모른다. flat 체계라면 (a) 버리거나 (b) 오염 라벨로
쓰는 수밖에 없지만, 롤업 loss 는 "**정답이 유리 5형제 중 하나**"라는 부분 정보를
그대로 활용한다 — 5형제 확률의 합을 키우는 gradient 가 세부 head 에 흐른다.
세부 직접 감독(색상별 6~9천 장)이 형제 간 경계를 세우고, 롤업 감독이 "유리다움"을
보강하는 분업.

---

## 7. 클래스 추가 시 체크리스트

1. 데이터: `fine-staging/<새라벨>/` 에 크롭 적재 (조건은 파일명 prefix)
2. [greenguide_classifier/taxonomy.py](greenguide_classifier/taxonomy.py): `TAXONOMY` 에 자식 추가 + `STAGING_DIR_SUPERVISION` 매핑
3. (안내 동일 형제면) `GUIDANCE_GROUPS` 갱신
4. migration: `waste_classes` 에 level=2 행 시드 (`active=false`)
5. `pytest tests/test_hierarchy.py` (분할 무결성 자동 검증)
6. `retrain_hier.py` 1사이클 — 활성화 판정은 자동

---

## 8. 정직한 진단: 실사용 정확도가 낮은 진짜 이유

현재 실사용 대분류 **52.9%** (frozen 93.8%). 갭의 구조:

| 원인 | 증거 | 대응 상태 |
|---|---|---|
| **극소 클래스** | cardboard 678 / food_waste 661 / trash_other 550 / **etc 131** — frozen f1 은 높아 보여도 (test 도 같은 분포라서) 실사용 일반화 불가 | ❌ **최우선 보강 대상** |
| 실내·손·잡배경 분포 부재 | 학습 대부분이 시설/스튜디오/bbox 크롭 | 부분 대응 (합성 8k, u2netp 크롭, non_object 마스킹 +5.9pp) |
| carton 역혼동 | 유리병→carton 오인 (안내 다름) | 하드네거티브로 gs 0.805 까지 (임계 0.85 미달) |
| etc 캐치올의 본질적 모호함 | 실사용 오답의 12/51 이 etc 진실 | 구조적 한계 (open-set) |

### 데이터 보강 우선순위 (다음 사이클)

1. **cardboard / food_waste / trash_other 증량** — 목표 각 3천+
   - cardboard: AI-Hub 140 `포장상자` 계열 zip (기승인, 소형)
   - food_waste: 140 음식물 계열 재확인 / 합성(음식물은 배경 의존 큼)
   - trash_other: 실사용 수집이 정석 (잡동사니는 공개셋 희소)
2. **carton 마감**: 하드네거티브 2라운드(비율↑) 또는 140 우유팩 유사품목 재탐색 → gs 0.85 돌파 시 자동 승격
3. **실사용자 데이터 루프 가동** (배포 후): 탭-라벨·피드백이 §3-8 루프에 자동 유입 —
   **53→85% 는 결국 이 루프가 닫는다** (모든 설계 문서의 일관된 결론)

### 재학습 실행 (전 과정 자동)
```bash
cd greenguide-classifier
.venv/bin/python retrain_hier.py --dry-run     # 피드백 현황
.venv/bin/python retrain_hier.py               # 풀 사이클 (게이트·승격·실사용평가 포함)
```
