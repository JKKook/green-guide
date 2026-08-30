"""전역 설정 — 경로·하이퍼파라미터·메타."""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# 자매 프로젝트인 waste-preprocessor의 산출물을 직접 참조
PREPROCESSOR_ROOT: Path = PROJECT_ROOT.parent / "waste-preprocessor"
MANIFEST_PATH: Path = PREPROCESSOR_ROOT / "data" / "processed" / "manifest.json"
VECTORS_DIR: Path = PREPROCESSOR_ROOT / "data" / "processed" / "vectors"

# 산출물 경로
DATA_DIR: Path = PROJECT_ROOT / "data"
SPLITS_DIR: Path = DATA_DIR / "splits"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR: Path = OUTPUTS_DIR / "checkpoints"
MODELS_DIR: Path = OUTPUTS_DIR / "models"
LOGS_DIR: Path = OUTPUTS_DIR / "logs"
PLOTS_DIR: Path = OUTPUTS_DIR / "plots"

# [역사적 유물] 구 6클래스 fallback — flat 파이프라인 하위호환 전용.
# 계층 학습의 정본은 src/taxonomy.py (대분류 14 × 세부 25) 이며 이 목록과 무관.
# 모듈 import 시 manifest 가 있으면 동적으로 갱신.
_DEFAULT_LABELS: tuple[str, ...] = (
    "cardboard", "glass", "metal", "paper", "plastic", "trash",
)
CLASS_LABELS: tuple[str, ...] = _DEFAULT_LABELS
NUM_CLASSES: int = len(CLASS_LABELS)
LABEL_TO_INDEX: dict[str, int] = {label: i for i, label in enumerate(CLASS_LABELS)}
INDEX_TO_LABEL: dict[int, str] = {i: label for i, label in enumerate(CLASS_LABELS)}


def refresh_classes_from_manifest() -> None:
    """manifest.json 의 실제 클래스 분포를 읽어 CLASS_LABELS 갱신.
    retrain.py 가 새 데이터로 manifest 를 갱신한 직후 호출하면
    train/evaluate/export 가 자동으로 새 클래스 수로 동작."""
    import json
    global CLASS_LABELS, NUM_CLASSES, LABEL_TO_INDEX, INDEX_TO_LABEL
    if not MANIFEST_PATH.exists():
        return
    try:
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        labels = sorted({item["label"] for item in manifest.get("items", [])})
        if not labels:
            return
        CLASS_LABELS = tuple(labels)
        NUM_CLASSES = len(CLASS_LABELS)
        LABEL_TO_INDEX = {label: i for i, label in enumerate(CLASS_LABELS)}
        INDEX_TO_LABEL = {i: label for i, label in enumerate(CLASS_LABELS)}
        print(f"[config] classes refreshed: {NUM_CLASSES} classes — {list(CLASS_LABELS)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[config] failed to refresh classes from manifest: {exc}")


# import 시 자동 시도
refresh_classes_from_manifest()

# 입력 차원 (preprocessor 의 VECTOR_DIM 과 일치해야 함)
IMAGE_SIZE: int = 224
IMAGE_CHANNELS: int = 3
INPUT_DIM: int = IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS  # 150,528

# 데이터 분할 비율
SPLIT_RATIOS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED: int = 42

# 재현성
RANDOM_SEED: int = 42

# 지원하는 아키텍처
# - mlp: flatten 1D 입력
# - cnn: (3, 224, 224) RGB 입력 (color stream)
# - cnn_edge: (3, 224, 224) Sobel edge map (edge stream, ensemble 용)
SUPPORTED_ARCHS: tuple[str, ...] = ("mlp", "cnn", "cnn_edge")

# MLP (Fully-Connected NN) 하이퍼파라미터
MLP_HIDDEN_DIMS: tuple[int, ...] = (256, 64)
MLP_DROPOUT_RATES: tuple[float, ...] = (0.5, 0.3)
MLP_BATCH_SIZE: int = 64
MLP_NUM_EPOCHS: int = 30
MLP_LEARNING_RATE: float = 1e-4
MLP_WEIGHT_DECAY: float = 1e-5
MLP_EARLY_STOPPING_PATIENCE: int = 5

# CNN (ResNet18 pretrained fine-tuning) 하이퍼파라미터
# CNN 은 사전학습된 표현을 이미 갖고 있어 epoch·LR 모두 더 작아도 충분
CNN_BATCH_SIZE: int = 32
CNN_NUM_EPOCHS: int = 15
CNN_LEARNING_RATE: float = 1e-4
CNN_WEIGHT_DECAY: float = 1e-5
CNN_EARLY_STOPPING_PATIENCE: int = 4
CNN_FREEZE_BACKBONE: bool = False  # True 면 마지막 FC layer 만 학습
# inverse-frequency 클래스 가중치 상한 = median * 이 배수. (Stage D, B-1.1)
# 낮출수록 rare 클래스 과가중 완화 → OOD-sink(cardboard/non_object) 약화.
# 현 train(49,240): rare 6클래스가 전부 이 천장에 붙음. 4.0→2.97, 2.5→1.85, 2.0→1.48.
# 기본 4.0(기존 동작 보존). 실험 시 env WASTE_CNN_WEIGHT_CAP 로 override.
CNN_CLASS_WEIGHT_CAP: float = float(os.getenv("WASTE_CNN_WEIGHT_CAP", "4.0"))

# 기존 MLP-only 코드가 참조하던 짧은 이름들 (alias)
HIDDEN_DIMS = MLP_HIDDEN_DIMS
DROPOUT_RATES = MLP_DROPOUT_RATES
BATCH_SIZE = MLP_BATCH_SIZE
NUM_EPOCHS = MLP_NUM_EPOCHS
LEARNING_RATE = MLP_LEARNING_RATE
WEIGHT_DECAY = MLP_WEIGHT_DECAY
EARLY_STOPPING_PATIENCE = MLP_EARLY_STOPPING_PATIENCE


def arch_subdir(category_dir: Path, arch: str) -> Path:
    """outputs/<category>/<arch>/ 경로 반환 + 생성."""
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"unsupported arch={arch!r} (supported: {SUPPORTED_ARCHS})")
    d = category_dir / arch
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_directories() -> None:
    for d in (SPLITS_DIR, CHECKPOINTS_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for arch in SUPPORTED_ARCHS:
        for d in (CHECKPOINTS_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR):
            (d / arch).mkdir(parents=True, exist_ok=True)
