# greenguide-classifier

GreenGuide AI 의 두 번째 서브 프로젝트. 자매 프로젝트 [`greenguide-preprocessor`](../greenguide-preprocessor) 가 만든 메타데이터(manifest.json) 와 전처리된 벡터(.npz) 를 입력으로 받아, 6-class 폐기물 분류기를 학습하는 PyTorch 기반 파이프라인.

**두 가지 아키텍처를 모두 지원** — Fully-Connected NN (baseline) 과 ResNet18 기반 CNN. 같은 데이터·평가 파이프라인으로 두 모델을 직접 비교 가능. 학습된 모델은 ONNX 로 export 되어 클라우드 API · Flutter on-device 등 다양한 배포 환경에 사용 가능하다.

| 모델 | Test Accuracy | 파일 크기 | 학습 시간 |
|---|---:|---:|---:|
| MLP (Fully-Connected) | 39.84% | 147 MB | ~90초 |
| **CNN (ResNet18 pretrained)** | **92.35%** | **43 MB** | ~12분 |

---

## 목차

1. [프로젝트 위치](#프로젝트-위치)
2. [핵심 결정 사항](#핵심-결정-사항)
3. [아키텍처](#아키텍처)
4. [모듈 구성](#모듈-구성)
5. [설치 및 환경 설정](#설치-및-환경-설정)
6. [사용법](#사용법)
7. [실행 결과](#실행-결과)
8. [모델 구조와 하이퍼파라미터](#모델-구조와-하이퍼파라미터)
9. [출력물 위치](#출력물-위치)
10. [테스트](#테스트)
11. [알려진 한계와 향후 개선](#알려진-한계와-향후-개선)
12. [트러블슈팅](#트러블슈팅)
13. [프로젝트 구조](#프로젝트-구조)

---

## 프로젝트 위치

```
GreenGuide AI
├── greenguide-preprocessor   (1) 수집·전처리·벡터화         완성
├── greenguide-classifier     (2) 지도학습 분류기 + ONNX     현재
├── 추론 API             (3) FastAPI/Cloud 배포        예정
├── Flutter 클라이언트   (4) 모바일 on-device 추론       예정
└── ReAct 에이전트화     (5) LLM 결합                  장기
```

이 프로젝트는 **모델 학습 + ONNX export** 까지만 다룬다. 실제 서빙은 별도 프로젝트.

---

## 핵심 결정 사항

| 분야 | 선택 | 이유 |
|---|---|---|
| 프레임워크 | **PyTorch + torchvision** | numpy 친화적 API, 디버깅 용이, HuggingFace 등 LLM 생태계와 일관, pretrained 모델 풍부 |
| 모델 (baseline) | FC NN (3 layer) | flatten 1D 벡터 입력으로 학습 원리 파악, 향후 비교 기준선 |
| 모델 (실용) | **ResNet18 (ImageNet pretrained) fine-tuning** | 적은 데이터로도 높은 정확도, 학습 시간 짧음 |
| 배포 포맷 | **ONNX** | Framework lock-in 회피, Flutter ONNX Runtime · Cloud 모두 호환 |
| 데이터 분할 | 70/15/15 stratified | 표준 비율, 클래스 분포 유지. MLP·CNN 동일 split 사용 (재현성) |
| 학습 전략 | Adam + early stopping | 안전한 기본값, 과적합 자동 방지 |
| 평가 지표 | Accuracy + per-class P/R/F1 + Confusion Matrix | 클래스별 편향 파악 가능 |
| 시각화 | matplotlib (Agg 백엔드) | headless 환경(서버) 호환 |
| 산출물 구조 | `outputs/<category>/<arch>/...` | MLP·CNN 산출물 격리, 비교·재실행 용이 |

---

## 아키텍처

```
+----------------------------------------------------------+
|  greenguide-preprocessor (sibling)                            |
|    data/processed/                                       |
|      manifest.json    <- 메타데이터                      |
|      vectors/<id>.npz <- float16 압축 벡터               |
+--------------------------+-------------------------------+
                           |
                           | 상대 경로 read
                           v
+----------------------------------------------------------+
|  greenguide-classifier                                        |
|                                                          |
|  greenguide_classifier/dataset.py   --> PyTorch Dataset                    |
|  greenguide_classifier/split.py     --> stratified train/val/test          |
|  greenguide_classifier/model.py     --> WasteClassifierMLP (3 layer FC)    |
|  greenguide_classifier/train.py     --> training loop + early stopping     |
|  greenguide_classifier/evaluate.py  --> metrics on test set                |
|  greenguide_classifier/visualize.py --> loss curve + confusion matrix      |
|  greenguide_classifier/export.py    --> torch -> ONNX + equivalence check  |
|                                                          |
|  outputs/                                                |
|    checkpoints/best.pt        <- 학습 중간 산출물        |
|    models/classifier.onnx     <- 배포용 최종 산출물      |
|    logs/training_log.json     <- epoch 별 손실/정확도     |
|    logs/evaluation.json       <- test set 성능 보고서    |
|    plots/training_curves.png                             |
|    plots/confusion_matrix.png                            |
+----------------------------------------------------------+
```

---

## 모듈 구성

| 모듈 | 책임 | 핵심 함수 |
|---|---|---|
| `config.py` | 경로·하이퍼파라미터 상수 | - |
| `dataset.py` | manifest+npz lazy 로드 | `load_manifest()`, `WasteDataset` |
| `split.py` | 재현 가능한 stratified 분할 | `stratified_split()`, `save/load_splits()` |
| `model.py` | FC NN 정의 | `WasteClassifierMLP`, `count_parameters` |
| `train.py` | 학습 루프 + early stopping | `train()`, `set_seed`, `pick_device` |
| `evaluate.py` | 테스트셋 평가 보고서 | `evaluate()`, `collect_predictions` |
| `visualize.py` | 그래프 PNG 저장 | `plot_training_curves`, `plot_confusion_matrix` |
| `export.py` | ONNX export + 등가성 검증 | `export_onnx()` |
| `main.py` | CLI 진입점 | `train/evaluate/visualize/export/all` |

---

## 설치 및 환경 설정

```bash
cd /Users/whdrnr01/ai/greenguide-classifier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 데이터 준비

greenguide-preprocessor 가 먼저 실행되어 있어야 한다.

```bash
ls ../greenguide-preprocessor/data/processed/manifest.json    # 존재 확인
ls ../greenguide-preprocessor/data/processed/vectors/ | head  # *.npz 파일 확인
```

만약 없다면 `../greenguide-preprocessor/README.md` 참조하여 먼저 실행.

---

## 사용법

### CLI 명령

`--arch` 플래그로 모델 선택 (default: `mlp`).

```bash
# 전체 흐름 (학습 -> 평가 -> 시각화 -> ONNX export)
python main.py all --arch cnn       # 추천: 정확도 92%+
python main.py all --arch mlp       # baseline: 정확도 ~40%

# 개별 단계
python main.py train --arch cnn
python main.py evaluate --arch cnn
python main.py visualize --arch cnn
python main.py export --arch cnn
```

### Active Learning Retrain

waste-api 가 수집하고 사용자가 피드백한 새 이미지로 모델을 재학습:

```bash
# 어떤 피드백이 있는지 확인만 (실제 학습 안 함)
python retrain.py --dry-run

# 실제 재학습 (다운로드 + 전처리 + 학습 + ONNX export, 약 15-20분)
python retrain.py

# 이미지 다운로드는 이미 됐고 전처리·학습만 다시
python retrain.py --skip-preprocessor
```

`retrain.py` 가 하는 일:
1. Supabase `user_uploads` 에서 `confirmed`/`corrected` 피드백 조회
2. 이미지를 `../greenguide-preprocessor/data/raw/garbage-classification/<label>/user_<id>.jpg` 로 다운로드
3. 기존 모델·평가 결과를 `outputs/backups/cnn_YYYYMMDD_HHMMSS/` 로 백업
4. greenguide-preprocessor 재실행 (전처리·벡터화)
5. 기존 `splits.json` 삭제 → 새 데이터 포함 재분할
6. greenguide-classifier 재학습 + 평가 + ONNX export
7. 새 vs 이전 정확도 비교, **악화 시 복원 안내** (수동 복원)

재학습 후 waste-api 가 새 모델을 쓰려면 서버 재시작 필요.

### Python API

```python
# 학습
from greenguide_classifier.train import train
ckpt_path = train()

# 평가
from greenguide_classifier.evaluate import evaluate
report = evaluate()
print(report["accuracy"])

# 학습된 모델로 추론 (ONNX Runtime)
import numpy as np
import onnxruntime as ort
sess = ort.InferenceSession("outputs/models/classifier.onnx")
vec = np.random.randn(1, 150528).astype(np.float32)
logits = sess.run(["logits"], {"vector": vec})[0]
predicted_class = logits.argmax(axis=1)
```

### 추론 예시 (학습된 모델 + 실제 벡터)
```python
import numpy as np
import onnxruntime as ort

# 1) ONNX 모델 로드
sess = ort.InferenceSession("outputs/models/classifier.onnx")

# 2) greenguide-preprocessor가 만든 벡터 하나 로드
with np.load("../greenguide-preprocessor/data/processed/vectors/<some_id>.npz") as data:
    vec = data["vector"].astype(np.float32).reshape(1, -1)

# 3) 예측
logits = sess.run(["logits"], {"vector": vec})[0]
probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
class_idx = int(probs.argmax(axis=1)[0])
labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
print(f"{labels[class_idx]}: {probs[0, class_idx]:.2%}")
```

---

## 실행 결과

2026-05-17 실측 기준. 두 아키텍처 모두 동일한 splits.json 으로 학습·평가.

### 공통: 데이터 분할
| Split | 크기 | 비율 |
|---|---:|---:|
| Train | 1,765 | 70% |
| Val | 378 | 15% |
| Test | 379 | 15% |

학습 기기: Apple Silicon (MPS 백엔드)

---

### MLP (Fully-Connected NN) — Baseline

| 항목 | 값 |
|---|---|
| 파라미터 수 | 38,552,262 (38.5M) |
| 배치 크기 | 64 |
| 학습 시간 | ~90초 (8 epoch 후 early stopping) |
| Best epoch | 3 |
| Best val accuracy | **35.19%** |
| **Test accuracy** | **39.84%** |

학습 곡선 요약 (best epoch 굵게):

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---:|---:|---:|---:|---:|
| 1 | 17.95 | 0.228 | 5.35 | 0.286 |
| 2 | 14.69 | 0.249 | 3.77 | 0.331 |
| **3** | **14.00** | **0.264** | **2.96** | **0.352** |
| 4-8 | 4.30~9.56 | 0.25~0.30 | 2.07~2.71 | 0.21~0.28 |

Per-class F1: cardboard 0.56 / glass 0.24 / metal 0.24 / paper 0.44 / plastic 0.54 / **trash 0.00**

---

### CNN (ResNet18 pretrained) — 실용 모델

| 항목 | 값 |
|---|---|
| 파라미터 수 | 11,179,590 (11.2M) — MLP 보다 **3.5배 적음** |
| 배치 크기 | 32 |
| 학습 시간 | ~12분 (12 epoch 후 early stopping) |
| Best epoch | 8 |
| Best val accuracy | **92.86%** |
| **Test accuracy** | **92.35%** |

학습 곡선 요약 (best epoch 굵게):

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---:|---:|---:|---:|---:|
| 1 | 0.724 | 0.750 | 0.353 | 0.876 |
| 2 | 0.175 | 0.956 | 0.298 | 0.892 |
| 3 | 0.053 | 0.997 | 0.234 | 0.907 |
| 7 | 0.014 | 0.999 | 0.211 | 0.913 |
| **8** | **0.013** | **0.999** | **0.208** | **0.929** |
| 9 | 0.005 | 1.000 | 0.201 | 0.926 |
| 10-12 | ~0.005 | 1.000 | 0.22~0.25 | 0.90~0.92 |

> **인사이트**: epoch 1 만에 val acc 87.57% 도달 — ImageNet pretrained 표현이 폐기물 분류에 이미 거의 충분.

Per-class F1: cardboard 0.97 / glass 0.89 / metal 0.92 / paper 0.96 / plastic 0.89 / **trash 0.89**

---

### MLP vs CNN 직접 비교

| Metric | MLP | CNN | 차이 |
|---|---:|---:|---:|
| Test Accuracy | 39.84% | **92.35%** | **+52.51pp** |
| Macro F1 | 0.34 | **0.92** | +0.58 |
| 파라미터 | 38.5M | 11.2M | 3.5× ↓ |
| 모델 파일 크기 | 147 MB | 43 MB | 3.5× ↓ |
| 학습 시간 | 90초 | 12분 | 8× ↑ |
| 학습 epoch 수 | 8 | 12 | +4 |
| ONNX export 오차 | 1.57e-06 | 2.38e-06 | 둘 다 거의 0 |

**Per-class F1 비교** (가장 극적인 변화는 trash):

| Class | MLP F1 | CNN F1 | 개선 |
|---|---:|---:|---:|
| cardboard | 0.56 | **0.97** | +0.41 |
| glass | 0.24 | **0.89** | +0.65 |
| metal | 0.24 | **0.92** | +0.68 |
| paper | 0.44 | **0.96** | +0.52 |
| plastic | 0.54 | **0.89** | +0.35 |
| **trash** | **0.00** | **0.89** | **+0.89** |

MLP 가 완전히 실패했던 trash 클래스에서 CNN 은 F1=0.89 달성. CNN 이 단순히 "전체 성능 좋음" 이 아니라 **약한 클래스를 균등하게 끌어올림** — 데이터 불균형 문제까지 일부 흡수.

---

### ONNX Export 검증

| 모델 | 파일 | 크기 | 최대 절대 오차 (PyTorch vs ONNX Runtime) |
|---|---|---:|---:|
| MLP | `outputs/models/mlp/classifier.onnx` | 147 MB | 1.57e-06 |
| **CNN** | `outputs/models/cnn/classifier.onnx` | **43 MB** | 2.38e-06 |

둘 다 tol 1e-4 통과. Dynamic batch axis 지원. Flutter `onnxruntime_flutter` plugin / Python `onnxruntime` 등 ONNX 지원 환경 어디서든 동일 결과.

### 시각화
- `outputs/plots/mlp/training_curves.png`, `outputs/plots/mlp/confusion_matrix.png`
- `outputs/plots/cnn/training_curves.png`, `outputs/plots/cnn/confusion_matrix.png`

### 결과 해석

| 관찰 | 시사점 |
|---|---|
| CNN val acc epoch 1 부터 87.57% | ImageNet pretrained 표현이 폐기물 분류에 이미 거의 충분 — transfer learning 의 위력 |
| CNN train acc 1.000 vs val acc 0.929 | 약간 overfitting 이지만 generalization 양호. 데이터 증강·regularization 으로 추가 개선 여지 |
| CNN 이 trash 클래스도 F1=0.89 | conv 특징이 적은 샘플(20개) 에서도 일반화. MLP 와 결정적 차이 |
| **MLP 가 CNN 보다 3.5× 큼** | 첫 Linear(150528→256) 가 38M 점유. 입력 차원이 큰 것이 본질적 비효율 |
| 학습 시간 8× | conv 연산이 더 무거움. 그러나 MLP 도 GPU 효율은 좋지 않음 (메모리 bandwidth bound) |

---

## 모델 구조와 하이퍼파라미터

### MLP — WasteClassifierMLP
```
Input  : (B, 150528) float32  ← flatten 1D 벡터
  ├─ Linear(150528 → 256)     # 첫 layer 가 파라미터 99.96% 차지
  ├─ ReLU
  ├─ Dropout(0.5)
  ├─ Linear(256 → 64)
  ├─ ReLU
  ├─ Dropout(0.3)
  └─ Linear(64 → 6)            # logits
Output : (B, 6)

총 파라미터: 38,552,262 (~38.5M)
```

### CNN — WasteClassifierCNN
```
Input  : (B, 3, 224, 224) float32  ← (C, H, W) 채널 우선 이미지 텐서
  │
  └─ torchvision.models.resnet18(weights=IMAGENET1K_V1)
       │  (Conv → BN → ReLU → MaxPool → 4 stages of BasicBlock)
       │   사전학습된 표현 활용
       │
       └─ self.backbone.fc = Linear(512 → 6)   # 마지막 fc 만 교체
Output : (B, 6)

총 파라미터: 11,179,590 (~11.2M)
freeze_backbone=True 면 마지막 fc 만 학습 (3,078 params)
```

### 하이퍼파라미터 (`greenguide_classifier/config.py`)

| 항목 | MLP | CNN |
|---|---:|---:|
| BATCH_SIZE | 64 | 32 |
| NUM_EPOCHS | 30 (early stop) | 15 (early stop) |
| LEARNING_RATE | 1e-4 | 1e-4 |
| WEIGHT_DECAY | 1e-5 | 1e-5 |
| EARLY_STOPPING_PATIENCE | 5 | 4 |
| 추가 옵션 | HIDDEN_DIMS=(256,64), DROPOUT=(0.5,0.3) | FREEZE_BACKBONE=False |

공통: `SPLIT_SEED = RANDOM_SEED = 42` 로 재현성 보장.

---

## 출력물 위치

산출물은 `outputs/<category>/<arch>/...` 구조로 격리되어 MLP·CNN 동시 비교가 가능하다.

```
outputs/
├── checkpoints/
│   ├── mlp/best.pt              # 147 MB
│   └── cnn/best.pt              # 43 MB
├── models/
│   ├── mlp/classifier.onnx      # 147 MB (배포용)
│   └── cnn/classifier.onnx      # 43 MB  (배포용 - 추천)
├── logs/
│   ├── mlp/training_log.json    # epoch 별 (train_loss, train_acc, val_loss, val_acc)
│   ├── mlp/evaluation.json      # test set metrics + confusion matrix
│   ├── cnn/training_log.json
│   └── cnn/evaluation.json
└── plots/
    ├── mlp/training_curves.png
    ├── mlp/confusion_matrix.png
    ├── cnn/training_curves.png
    └── cnn/confusion_matrix.png
```

`data/splits/splits.json` 에는 재현 가능한 train/val/test 인덱스가 저장된다. **MLP·CNN 이 같은 split 을 사용**해 비교 가능성을 보장한다. 같은 분할로 재학습하려면 이 파일을 보존.

---

## 테스트

```bash
.venv/bin/python -m pytest
# ruff + pytest 한 번에 (커밋 전 검증)
scripts/check.sh
```

총 **24개 테스트**:
- `test_dataset.py` (4): MLP dataset - manifest 누락, 길이, shape/dtype, 라벨 일관성
- `test_dataset_cnn.py` (5): CNN image dataset - shape, MLP 와 데이터 일관성 (펼침/접힘 등가), build_dataset factory
- `test_split.py` (4): 분할 크기, 겹침 없음, 클래스 분포 보존, JSON roundtrip
- `test_model.py` (11): MLP/CNN forward·backward·파라미터, freeze_backbone, build_model factory

---

## 알려진 한계와 향후 개선

| 항목 | 현재 상태 | 개선 방향 |
|---|---|---|
| ~~모델 표현력~~ | ~~FC NN 한계~~ → **CNN 도입 완료 (test acc 92.35%)** | (해결) |
| ~~입력 형태~~ | ~~flatten 1D 만~~ → **WasteImageDataset 으로 (C,H,W) 지원** | (해결) |
| 데이터 증강 | 없음 | 학습 시 RandomFlip, ColorJitter, RandomResizedCrop 등 추가 → 추가 +2~5pp 기대 |
| 클래스 불균형 | trash 137개 vs paper 592개 (~4.3배). CNN 이 어느 정도 흡수했지만 여전히 trash recall 0.85 | class weights, oversampling, focal loss |
| 모델 경량화 | CNN 43MB → 모바일 on-device 에 적당하지만 더 작게 가능 | int8 quantization, MobileNetV3, EfficientNet-Lite |
| 추적 도구 | JSON 로그만 | wandb / tensorboard 통합 |
| 배포 | ONNX export 만 | FastAPI 추론 서버, Docker 이미지화 |
| 메타데이터 query | manifest.json 직접 read | Supabase 에서 라벨/필터 조회 (production 서빙 시) |

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `manifest not found` | greenguide-preprocessor 미실행 | 자매 프로젝트 먼저 완료 |
| `least populated class has only N member` | 데이터셋이 너무 작아 stratified 분할 실패 | 클래스당 최소 8개 이상 확보 |
| `Out of memory` | MPS/CUDA 메모리 부족 | `BATCH_SIZE` 를 32 또는 16 으로 |
| 학습이 매우 느림 | CPU 만 사용 중 | Mac 의 경우 자동으로 MPS 가 잡혀야 함. `pick_device()` 결과 확인 |
| ONNX 변환 후 결과 불일치 | opset 버전 미스매치 | `export_onnx(opset=...)` 변경하여 재시도 |
| `tflite_flutter`/ Flutter 연동 어려움 | 본 프로젝트 범위 밖 | ONNX Runtime Flutter plugin 사용 또는 Cloud API 패스 |

---

## 프로젝트 구조

```
greenguide-classifier/
├── .gitignore
├── README.md
├── pytest.ini
├── requirements.txt
├── main.py
├── data/
│   └── splits/splits.json      # 재현 가능한 분할
├── outputs/
│   ├── checkpoints/
│   ├── models/
│   ├── logs/
│   └── plots/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── split.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── export.py
└── tests/
    ├── conftest.py
    ├── test_dataset.py          # MLP dataset
    ├── test_dataset_cnn.py      # CNN image dataset
    ├── test_split.py
    └── test_model.py            # MLP + CNN 모델
```
