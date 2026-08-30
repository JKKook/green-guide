# Lab: 학습 데이터가 모델까지 가는 길

waste-classifier 의 학습 데이터가 어떤 형태로 만들어지고, 어떤 변환을 거쳐 **두 가지 모델(MLP / CNN)** 입력이 되는지를 **실제 값과 코드 위치**로 추적한 문서. 2026-05-17 실행 기준.

추적 대상 샘플 ID: `b2dfb128a3ad` (cardboard 클래스의 첫 번째 이미지)

> 처음 작성될 당시엔 MLP 만 있었으나, 이후 CNN 이 추가되면서 단계 3 (Tensor 변환) 과 단계 5 (DataLoader 배칭) 에서 분기가 생긴다. 각 단계의 "MLP" / "CNN" 표지를 따라가면 두 흐름을 동시에 비교 가능.

---

## 단계 0. 데이터의 출발점

학습 데이터는 자매 프로젝트 **waste-preprocessor** 가 만들어 둔 산출물이다. waste-classifier 는 두 종류의 파일만 읽는다.

```
../waste-preprocessor/data/processed/
├── manifest.json              <- 모든 메타데이터 (2,522 items)
└── vectors/
    ├── b2dfb128a3ad.npz       <- 개별 이미지의 1D 벡터
    ├── 004cfe236bcf.npz
    └── ... (총 2,522개)
```

**참조 코드**: [src/config.py:9-12](src/config.py#L9-L12)
```python
PREPROCESSOR_ROOT: Path = PROJECT_ROOT.parent / "waste-preprocessor"
MANIFEST_PATH: Path = PREPROCESSOR_ROOT / "data" / "processed" / "manifest.json"
VECTORS_DIR: Path = PREPROCESSOR_ROOT / "data" / "processed" / "vectors"
```

Supabase Postgres·Storage 에도 동일한 데이터의 사본이 있지만, **학습 시에는 사용하지 않는다** (속도 때문에 로컬 파일 우선).

---

## 단계 1. Manifest 파싱

`manifest.json` 은 전체 데이터셋의 인덱스 역할. 한 item 은 다음과 같이 생겼다.

```json
{
  "id": "b2dfb128a3ad",
  "label": "cardboard",
  "source_path": "data/raw/garbage-classification/cardboard/cardboard1.jpg",
  "filename": "cardboard1.jpg",
  "vector_path": "data/processed/vectors/b2dfb128a3ad.npz",
  "stats": {
    "mean": 0.7368,
    "std": 0.6166,
    "min": -1.9467,
    "max": 2.1290
  },
  "original_url": "https://hzcljbtarkvztxjpdfch.supabase.co/storage/v1/object/public/raw-images/cardboard/b2dfb128a3ad.jpg"
}
```

| 필드 | 학습에서의 역할 |
|---|---|
| `id` | `.npz` 파일을 찾는 키 |
| `label` | 정답 클래스 (지도학습 타겟) |
| `stats` | 전처리 결과 검증용 메타. 학습 자체엔 미사용 |
| `original_url` | 학습 후 디버깅 시각 검증에 유용 |
| `source_path`, `vector_path`, `filename` | 추적·디버깅용 메타 |

전체 manifest 는 다음과 같이 load 한다.

**참조 코드**: [src/dataset.py:11-19](src/dataset.py#L11-L19)
```python
def load_manifest(path: Path | None = None) -> list[dict[str, Any]]:
    path = path if path is not None else config.MANIFEST_PATH
    if not path.exists():
        raise FileNotFoundError(...)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)["items"]
```

**산출물**: Python list 길이 2,522 (= cleansed item 개수)

---

## 단계 2. 개별 벡터 (.npz) 로드

`b2dfb128a3ad.npz` 파일의 내부 구조:

```
파일 크기 (디스크)       : 79,977 bytes   (~78 KB, gzip 압축)
내부 array key            : ["vector"]
shape                     : (150528,)      = 224 × 224 × 3
저장 dtype                : float16        (RAM 절약, 압축률 ↑)
메모리 점유 (decompressed): 301,056 bytes  (~294 KB)
값 범위                   : -1.9463 ~ +2.1289
평균                      : 0.7368         (ImageNet 정규화 후 시프트)
```

처음 5개 / 마지막 5개 값:
```
처음   : [+2.0098, +1.4834, +1.0889, +2.0430, +1.5186]   <- 밝은 영역 (cardboard 윗부분)
마지막 : [-0.4775, -0.6367, -0.1486, -0.4602, -0.6191]   <- 어두운 영역
```

**참조 코드**: [src/dataset.py:33-37](src/dataset.py#L33-L37)
```python
def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
    item = self.items[idx]
    npz_path = self.vectors_dir / f"{item['id']}.npz"
    with np.load(npz_path, allow_pickle=False) as data:
        vec = data["vector"].astype(np.float32, copy=False)
    return torch.from_numpy(vec), config.LABEL_TO_INDEX[item["label"]]
```

이 코드는:
1. `.npz` 압축 파일을 메모리 매핑으로 연다
2. `"vector"` 키의 float16 배열을 꺼낸다
3. **float16 → float32 로 다시 캐스팅** (모델·CPU/GPU 연산이 float32 기본)
4. numpy → `torch.Tensor` 무복사 변환

---

## 단계 3. PyTorch Tensor + 라벨 인덱스 반환 (MLP / CNN 분기)

여기서 두 아키텍처가 처음으로 갈린다.

### 3a. MLP — `WasteDataset[0]`
```
반환값:
  ├─ x: Tensor, shape (150528,), dtype torch.float32   ← 1D flatten 그대로
  └─ y: int, value 0 = "cardboard"
```

**참조 코드**: [src/dataset.py:32-37](src/dataset.py#L32-L37)

### 3b. CNN — `WasteImageDataset[0]`
```
반환값:
  ├─ x: Tensor, shape (3, 224, 224), dtype torch.float32   ← CHW 채널 우선
  └─ y: int, value 0 = "cardboard"

내부 변환:
  1) np.load → (150528,) float16  → astype(float32)
  2) .reshape(224, 224, 3)         → HWC (preprocessor 가 저장한 그대로)
  3) .transpose(2, 0, 1)           → CHW (PyTorch CNN 표준)
  4) np.ascontiguousarray(...)     → 메모리 연속성 보장 (Tensor 변환 안정성)
  5) torch.from_numpy(...)
```

**참조 코드**: [src/dataset.py:50-64](src/dataset.py#L50-L64)

> CNN 도 같은 `.npz` 파일을 읽는다. 데이터는 동일하고 **모양만 바뀐다**. 테스트 `test_image_dataset_consistency_with_flatten` 가 이 등가성을 검증.

### 공통 — 라벨 매핑

| 정수 인덱스 | 클래스 이름 |
|:---:|---|
| 0 | cardboard |
| 1 | glass |
| 2 | metal |
| 3 | paper |
| 4 | plastic |
| 5 | trash |

**참조 코드**: [src/config.py:28-29](src/config.py#L28-L29). 라벨을 정수 인덱스로 바꾸는 이유: `nn.CrossEntropyLoss` 가 `int64` 타겟을 요구하기 때문.

### Factory 함수 — `build_dataset(arch, items)`
`arch` 문자열 ("mlp" / "cnn") 로 적절한 Dataset 인스턴스를 생성. **참조 코드**: [src/dataset.py:67-72](src/dataset.py#L67-L72)

---

## 단계 4. train / val / test 분할

2,522 개를 **70 / 15 / 15 stratified split** 으로 나눈다. stratified 의 핵심은 **모든 split 에 클래스 비율을 보존**.

| Split | 크기 | 비율 | 클래스별 분포 |
|---|---:|---:|---|
| train | 1,765 | 70% | cardboard=282, glass=351, metal=286, paper=414, plastic=336, trash=96 |
| val | 378 | 15% | cardboard=60, glass=75, metal=61, paper=89, plastic=72, trash=21 |
| test | 379 | 15% | cardboard=61, glass=75, metal=62, paper=89, plastic=72, trash=20 |

분할 결과는 `data/splits/splits.json` 에 인덱스로 저장되어 **재실행해도 동일한 분할**이 보장된다.

```json
{
  "train": [1492, 2103, 0, 1840, 815, ...],   // 1,765개
  "val":   [22, 1893, 504, ...],              // 378개
  "test":  [1701, 935, 12, ...]               // 379개
}
```

**참조 코드**: [src/split.py:18-43](src/split.py#L18-L43)
```python
def stratified_split(items, ratios=..., seed=42):
    # 1) train vs (val+test)
    train_idx, holdout_idx = train_test_split(
        indices, test_size=ratios["val"]+ratios["test"],
        stratify=labels, random_state=seed,
    )
    # 2) holdout 을 val vs test 로 다시 stratified split
    ...
```

`random_state=42` 로 고정. 다른 seed 로 분할하려면 `config.SPLIT_SEED` 변경 또는 `splits.json` 삭제 후 재실행.

---

## 단계 5. DataLoader 가 배칭 (MLP / CNN 분기)

PyTorch `DataLoader` 는 Dataset 을 batch_size 단위로 묶어 (옵션으로) shuffle 한다. `arch` 별 batch_size 가 다르다.

### 5a. MLP — batch_size=64
```
총 배치 수   : 1,765 / 64 = 28 batches (마지막 배치는 1,765 % 64 = 5 items)

x_batch: shape (64, 150528)
         dtype torch.float32
         메모리 ~36.8 MB / batch

y_batch: shape (64,)
         dtype torch.int64

  배치 내 라벨 분포 예시:
    Counter({3 (paper)=19, 1 (glass)=16, 0 (cardboard)=11,
             2 (metal)=9, 4 (plastic)=8, 5 (trash)=1})
```

### 5b. CNN — batch_size=32
```
총 배치 수   : 1,765 / 32 = 56 batches

x_batch: shape (32, 3, 224, 224)
         dtype torch.float32
         메모리 ~18.4 MB / batch (값의 총량은 MLP 와 동일 - 32×150528 = 64×75264)

y_batch: shape (32,)
         dtype torch.int64
```

> **왜 CNN 만 batch_size 가 작은가?** MLP 는 첫 Linear 의 weight matrix (38M params × 4 byte) 가 메모리의 대부분이라 activations 가 큰 batch 를 써도 부담 적음. CNN 은 conv layer 마다 큰 feature map (예: 64×112×112) 을 보존해야 해 batch 가 클수록 activation 메모리가 폭증. 실측 MPS GPU 메모리 한도에 맞춰 32 로 설정.

**참조 코드**: [src/train.py:120-122](src/train.py#L120-L122)
```python
train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=hp.batch_size, shuffle=False, num_workers=0)
```
`hp.batch_size` 는 [src/train.py:42-58](src/train.py#L42-L58) 의 `get_hyperparams(arch)` 에서 결정.

---

## 단계 6. 모델 입력 → 출력 (MLP / CNN 분기)

### 6a. MLP — WasteClassifierMLP
```
Input  : (64, 150528) float32
   │
   ▼  Linear(150528 → 256)  weights: (256, 150528) = 38.5M params
   ▼  ReLU
   ▼  Dropout(0.5)           training 시 50% 무작위 0
   │
   ▼  Linear(256 → 64)       weights: (64, 256) = 16K
   ▼  ReLU
   ▼  Dropout(0.3)
   │
   ▼  Linear(64 → 6)         weights: (6, 64) = 384
   │
Output : (64, 6) float32   <- 6개 클래스의 raw logits (softmax 안 됨)

총 trainable parameters: 38,552,262
```

**참조 코드**: [src/model.py:11-39](src/model.py#L11-L39)

### 6b. CNN — WasteClassifierCNN
```
Input  : (32, 3, 224, 224) float32
   │
   ▼  ResNet18 backbone (사전학습 ImageNet)
   │   Conv7x7(3→64) → BN → ReLU → MaxPool3x3
   │   Layer1: 2× BasicBlock (64→64)
   │   Layer2: 2× BasicBlock (64→128, stride 2)
   │   Layer3: 2× BasicBlock (128→256, stride 2)
   │   Layer4: 2× BasicBlock (256→512, stride 2)
   │   AdaptiveAvgPool2d(1, 1) → (32, 512)
   │
   ▼  fc: Linear(512 → 6)   <- 마지막 layer 만 6-class 로 교체
   │
Output : (32, 6) float32

총 trainable parameters: 11,179,590
  - backbone: 11,176,512  (사전학습 weights 시작점)
  - fc layer: 3,078
```

**참조 코드**: [src/model.py:42-69](src/model.py#L42-L69)

> **핵심 차이**: MLP 가 38.5M params 인데도 CNN(11.2M) 보다 못한 이유는, MLP 의 거의 모든 params 가 **첫 Linear(150528→256)** 에 쏟아져 픽셀 위치별 weight 를 학습하는 데 소모되기 때문. CNN 의 Conv2d 는 **동일 weight 를 이미지 전체에 슬라이딩** 하면서 위치 불변 특징을 학습 — 파라미터 효율 압도적.

### 공통: build_model factory
```python
from src.model import build_model
model = build_model("cnn")  # 또는 "mlp"
```
**참조 코드**: [src/model.py:72-78](src/model.py#L72-L78)

logits 예시는 둘 다 같은 형태 — 가장 큰 값을 가진 인덱스가 모델의 예측 클래스.

---

## 단계 7. 손실 계산 + 역전파

`nn.CrossEntropyLoss` 가 softmax + negative log likelihood 를 한 번에 처리한다.

```
loss = CrossEntropyLoss(logits, y_batch)

  내부 계산:
    1. softmax: logits → 확률 (각 행 합=1)
    2. -log(정답 클래스의 확률)
    3. batch 평균

예시 (한 샘플):
  logits = [-0.32, 0.18, -0.05, 0.41, -0.12, 0.27]
  softmax = [0.09, 0.14, 0.11, 0.18, 0.10, 0.15]
  y_true = 0 (cardboard)
  → loss = -log(0.09) ≈ 2.41
```

실제 학습 첫 epoch 의 loss 값 비교 (같은 데이터·split):

| | MLP | CNN |
|---|---:|---:|
| epoch 1 train_loss | 17.95 | **0.72** |
| epoch 1 val_loss | 5.35 | **0.35** |
| epoch 1 val_acc | 0.286 | **0.876** |

CNN 은 epoch 1 부터 이미 val acc 87.6% 달성. **ImageNet pretrained 표현이 폐기물 분류에도 매우 잘 맞기 때문**.

3 epoch 후 (MLP best):
```
MLP: train_loss=14.00, val_loss=2.96, val_acc=0.352  (35.2%)
CNN: train_loss=0.05,  val_loss=0.23, val_acc=0.907  (90.7%)
```

8 epoch 후 (CNN best):
```
CNN: train_loss=0.013, val_loss=0.208, val_acc=0.929  (92.9%)
```

loss 가 낮아지면서 점차 정답 클래스에 더 높은 확률을 부여하게 된다.

역전파:
```python
optimizer.zero_grad(set_to_none=True)  # 이전 gradient 초기화
loss.backward()                         # 모든 파라미터에 gradient 계산
optimizer.step()                        # gradient 의 반대 방향으로 가중치 업데이트
```

Adam optimizer (lr=1e-4) 가 38.5M 개의 파라미터 각각에 대해 learning rate 와 momentum 을 적응적으로 조정.

**참조 코드**: [src/train.py:65-71](src/train.py#L65-L71)

---

## 단계 8. Epoch 단위 평가 → 체크포인트 → Early Stopping

매 epoch 종료 후:

1. **Validation** 셋에서 무작위 dropout 없이 forward 만 수행
2. val_accuracy 계산
3. 이전 best 보다 좋으면 **`outputs/checkpoints/<arch>/best.pt`** 에 모델 저장
4. patience epoch 동안 개선 없으면 **early stopping**

### MLP 학습 진행 (patience=5)

| epoch | val_acc | best 갱신? | patience |
|:---:|:---:|:---:|:---:|
| 1 | 0.286 | ✓ | 0 |
| 2 | 0.331 | ✓ | 0 |
| **3** | **0.352** | **✓ ← best** | 0 |
| 4 | 0.280 | ✗ | 1 |
| 5 | 0.259 | ✗ | 2 |
| 6 | 0.243 | ✗ | 3 |
| 7 | 0.209 | ✗ | 4 |
| 8 | 0.217 | ✗ | 5 → **early stop** |

3 epoch 만에 best 도달 후 5 epoch 동안 개선 없어 종료. **FC NN 의 표현력 한계**.

### CNN 학습 진행 (patience=4)

| epoch | val_acc | best 갱신? | patience |
|:---:|:---:|:---:|:---:|
| 1 | 0.876 | ✓ | 0 |
| 2 | 0.892 | ✓ | 0 |
| 3 | 0.907 | ✓ | 0 |
| 4 | 0.902 | ✗ | 1 |
| 5 | 0.902 | ✗ | 2 |
| 6 | 0.907 | ✗ | 3 |
| 7 | 0.913 | ✓ | 0 |
| **8** | **0.929** | **✓ ← best** | 0 |
| 9 | 0.926 | ✗ | 1 |
| 10 | 0.915 | ✗ | 2 |
| 11 | 0.918 | ✗ | 3 |
| 12 | 0.905 | ✗ | 4 → **early stop** |

CNN 은 train acc 가 epoch 3 부터 99%+ (epoch 9 부터 100%) 인데도 val acc 는 90%+ 유지 → 약간의 overfitting 이지만 generalization 양호. 데이터 증강 도입하면 추가 개선 여지.

---

## 단계 9. 학습된 모델로 추론 흐름

### MLP — `outputs/models/mlp/classifier.onnx` (147 MB)
```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("outputs/models/mlp/classifier.onnx")

with np.load("../waste-preprocessor/data/processed/vectors/b2dfb128a3ad.npz") as data:
    vec = data["vector"].astype(np.float32).reshape(1, 150528)

logits = sess.run(["logits"], {"vector": vec})[0]          # (1, 6)
probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
predicted_idx = int(probs.argmax(1)[0])
print(f"MLP 예측: {['cardboard','glass','metal','paper','plastic','trash'][predicted_idx]} "
      f"({probs[0, predicted_idx]:.2%})")
```

### CNN — `outputs/models/cnn/classifier.onnx` (43 MB)
같은 벡터를 **(1, 3, 224, 224)** 로 reshape 해서 입력. input 이름도 `"vector"` → `"image"`.

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("outputs/models/cnn/classifier.onnx")

with np.load("../waste-preprocessor/data/processed/vectors/b2dfb128a3ad.npz") as data:
    vec = data["vector"].astype(np.float32)
    # 단계 3b 와 동일한 reshape
    hwc = vec.reshape(224, 224, 3)
    chw = np.ascontiguousarray(hwc.transpose(2, 0, 1))
    img = chw.reshape(1, 3, 224, 224)

logits = sess.run(["logits"], {"image": img})[0]           # (1, 6)
probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
predicted_idx = int(probs.argmax(1)[0])
print(f"CNN 예측: {['cardboard','glass','metal','paper','plastic','trash'][predicted_idx]} "
      f"({probs[0, predicted_idx]:.2%})")
```

두 모델 모두 같은 원본 데이터에서 출발하지만 **CNN 이 약 2.3배 높은 정확도** 로 예측한다.

---

## 데이터 흐름 한 줄 요약

```
manifest.json (2,522 items)
   ├─> stratified split → train/val/test (1,765 / 378 / 379)  ← 양쪽 모델 공통
   │
   └─> 각 item: .npz (78 KB, float16) → np.load + astype(float32)
        │
        ├─ MLP path:
        │   torch.Tensor (150528,) → DataLoader batches → (64, 150528)
        │   → WasteClassifierMLP (38.5M params) → (64, 6) logits
        │   → CrossEntropyLoss + backward + Adam
        │   → 8 epoch 후 early stop, best val acc 35.2%
        │   → ONNX export (147 MB)
        │
        └─ CNN path:
            .reshape(224,224,3).transpose(2,0,1) → torch.Tensor (3,224,224)
            → DataLoader batches → (32, 3, 224, 224)
            → WasteClassifierCNN (ResNet18 pretrained, 11.2M params) → (32, 6)
            → CrossEntropyLoss + backward + Adam
            → 12 epoch 후 early stop, best val acc 92.9%
            → ONNX export (43 MB)
```

---

## 직접 실행해서 확인하기

```bash
# Dataset 한 개 샘플 보기
.venv/bin/python -c "
from src.dataset import WasteDataset, load_manifest
items = load_manifest()
ds = WasteDataset(items)
x, y = ds[0]
print(f'shape={x.shape}, dtype={x.dtype}, label_idx={y}')
print(f'item meta: {items[0][\"id\"]}, {items[0][\"label\"]}')
"

# Split 결과 확인
.venv/bin/python -c "
from src.split import load_splits
s = load_splits()
print({k: len(v) for k, v in s.items()})
"

# 한 batch 모양 확인
.venv/bin/python -c "
from torch.utils.data import DataLoader
from src.dataset import WasteDataset, load_manifest
from src.split import load_splits, subset_items
items = load_manifest()
splits = load_splits()
loader = DataLoader(WasteDataset(subset_items(items, splits['train'])),
                    batch_size=64, shuffle=True)
x, y = next(iter(loader))
print(f'x: {x.shape} {x.dtype}, y: {y.shape} {y.dtype}')
print(f'y values: {y.tolist()}')
"
```

---

## 참고: 왜 FC NN 은 한계가 있는가 → CNN 이 어떻게 해결했나

### MLP 의 본질적 한계

이 lab 의 단계 3a 에서 보듯 **이미지가 1D 로 펼쳐진 시점에서 공간 구조(spatial structure)가 완전히 사라진다**. 즉:

- 픽셀 `(100, 100)` 과 `(100, 101)` 이 옆에 있다는 정보 사라짐
- 같은 색이 모여 있다는 정보 사라짐
- 모서리·질감 같은 local pattern 사라짐

FC NN 은 평탄해진 150,528 차원에서 **각 픽셀 위치를 독립 feature 로** 학습. 그래서 색 평균·전체 밝기 같은 **global statistics** 는 잡지만 (cardboard 가 갈색이라 잘 잡힘), 형태·질감(glass, metal)은 잡지 못한다.

### CNN 이 이를 어떻게 해결하는가

CNN 은 단계 3b 에서 `.reshape + .transpose` 로 공간 구조를 복원한 (3, 224, 224) 텐서를 입력으로 받는다. 그 후 Conv2d 가:

1. **Local connectivity**: 작은 kernel (예: 3×3) 이 인접 픽셀만 본다 → 모서리·질감 같은 local pattern 직접 학습
2. **Weight sharing**: 같은 kernel 이 이미지 전체에 슬라이딩 → 위치 불변 (translation invariance)
3. **Hierarchical features**: shallow layer = edge/color blob, deep layer = 객체 부분 (귀, 잎, 라벨…)
4. **Pretrained representation**: ImageNet 1.28M 장으로 학습된 표현이 폐기물 분류에도 강력하게 전이됨

### 결과 비교 (test set)

| Class | MLP F1 | CNN F1 | 차이 | 해석 |
|---|---:|---:|---:|---|
| cardboard | 0.56 | 0.97 | +0.41 | MLP 도 갈색은 잡았지만 CNN 이 모양까지 인식 |
| glass | 0.24 | 0.89 | +0.65 | 투명도·반사 같은 spatial pattern 핵심 |
| metal | 0.24 | 0.92 | +0.68 | 광택·금속 질감 인식 |
| paper | 0.44 | 0.96 | +0.52 | 종이 결·접힘 같은 texture |
| plastic | 0.54 | 0.89 | +0.35 | 색·투명도 혼합 |
| **trash** | **0.00** | **0.89** | **+0.89** | 20개로 가장 적은 클래스 — pretrained 의 일반화가 결정적 |

`trash` 클래스의 변화가 가장 극적. MLP 는 데이터가 적은 클래스를 아예 학습하지 못했지만, CNN 은 pretrained backbone 의 일반화 능력으로 20개만 보고도 F1 0.89 달성.

### 핵심 교훈

> **모델 크기가 아니라 모델 구조가 중요하다.**
> MLP 38.5M params << CNN 11.2M params 인데도 정확도는 39.84% << 92.35%.
> "데이터에 맞는 inductive bias" 가 capacity 보다 훨씬 중요하다는 실증.
