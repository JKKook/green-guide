"""waste-api 전역 설정."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # 아래 os.getenv 전에 .env 반영 (이미 설정된 env 는 덮어쓰지 않음)


PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]  # src/core/config.py → 레포 루트

# 자매 프로젝트의 ONNX 모델 직접 참조 (로컬 개발용)
CLASSIFIER_ROOT: Path = PROJECT_ROOT.parent / "waste-classifier"
DEFAULT_MODEL_ARCH: str = "cnn"  # mlp | cnn


def _resolve_model_path() -> Path:
    """Color stream 모델 — 우선순위:
    1. WASTE_API_MODEL_PATH 환경변수
    2. waste-api/models/classifier.onnx  (배포 패키지 내 번들 — Docker 등)
    3. ../waste-classifier/outputs/models/cnn/classifier.onnx  (로컬 sibling)
    """
    env_path = os.getenv("WASTE_API_MODEL_PATH")
    if env_path:
        return Path(env_path)

    bundled = PROJECT_ROOT / "models" / "classifier.onnx"
    if bundled.exists():
        return bundled

    return CLASSIFIER_ROOT / "outputs" / "models" / DEFAULT_MODEL_ARCH / "classifier.onnx"


def _resolve_edge_model_path() -> Path | None:
    """Edge stream 모델 (선택). 없으면 ensemble 비활성."""
    env_path = os.getenv("WASTE_API_EDGE_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        return p if p.exists() else None

    bundled = PROJECT_ROOT / "models" / "classifier_edge.onnx"
    if bundled.exists():
        return bundled

    sibling = CLASSIFIER_ROOT / "outputs" / "models" / "cnn_edge" / "classifier.onnx"
    if sibling.exists():
        return sibling
    return None


MODEL_PATH: Path = _resolve_model_path()
EDGE_MODEL_PATH: Path | None = _resolve_edge_model_path()

# Ensemble 가중치 (color weight)
# 0.8 이 test set 에서 최적 (92.61% vs color 단독 91.82%)
ENSEMBLE_COLOR_WEIGHT: float = float(
    os.getenv("WASTE_API_ENSEMBLE_COLOR_WEIGHT", "0.8"),
)

# 클래스 정의 (waste-preprocessor·waste-classifier와 동일 순서)
CLASS_LABELS: tuple[str, ...] = (
    "cardboard", "glass", "metal", "paper", "plastic", "trash",
)
NUM_CLASSES: int = len(CLASS_LABELS)

# 이미지 전처리 — preprocessor와 동일해야 함
IMAGE_SIZE: int = 224
IMAGE_CHANNELS: int = 3
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# 업로드 제한
MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
SUPPORTED_CONTENT_TYPES: tuple[str, ...] = (
    "image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp",
)

# 사용자 사진 수집 (active learning loop) 활성 여부.
# False 면 Supabase 호출 자체를 안 함 → 오프라인 추론만 동작.
COLLECT_USER_UPLOADS: bool = (
    os.getenv("WASTE_API_COLLECT_UPLOADS", "true").lower() in ("1", "true", "yes")
)

# 서버
API_TITLE: str = "GreenGuide Waste Classifier API"
API_VERSION: str = "0.1.0"
CORS_ORIGINS: tuple[str, ...] = ("*",)  # 개발 단계 — 배포 시 좁힐 것

# 추론 시 input/output 이름 (export.py 와 일치해야 함)
ONNX_INPUT_NAME_MLP: str = "vector"
ONNX_INPUT_NAME_CNN: str = "image"
ONNX_OUTPUT_NAME: str = "logits"


def get_model_arch() -> str:
    """모델 경로로부터 arch 추정 (.../models/cnn/... → 'cnn')."""
    parent_name = MODEL_PATH.parent.name
    if parent_name in ("mlp", "cnn"):
        return parent_name
    return DEFAULT_MODEL_ARCH


def get_input_name(arch: str) -> str:
    return {"mlp": ONNX_INPUT_NAME_MLP, "cnn": ONNX_INPUT_NAME_CNN}[arch]


# ── 외부 서비스 자격 ──
SUPABASE_URL: str | None = os.getenv("SUPABASE_URL")
SUPABASE_KEY: str | None = os.getenv("SUPABASE_KEY")
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")

# ── 계층 분류(hier) ──
HIER_MODEL_PATH_ENV: str | None = os.getenv("WASTE_API_HIER_MODEL_PATH")
DINO_WEIGHT: float = float(os.getenv("WASTE_API_DINO_W", "0"))       # 0 = DINOv2 앙상블 비활성
CAM_PRIOR_WEIGHT: float = float(os.getenv("WASTE_API_CAM_W", "0.15"))

# ── 증거 엔진 (CLIP 정체 / OCR) ──
CLIP_ENABLED: bool = os.getenv("WASTE_API_CLIP", "1") != "0"
CLIP_PRIOR_WEIGHT: float = float(os.getenv("WASTE_API_CLIP_W", "0.5"))
CLIP_SCENE_WEIGHT: float = float(os.getenv("WASTE_API_CLIP_SCENE_W", "0.2"))
OCR_ENABLED: bool = os.getenv("WASTE_API_OCR", "1") != "0"
OCR_SKIP_CONFIDENCE: float = float(os.getenv("WASTE_API_OCR_SKIP_CONF", "0.75"))

# ── VLM 폴백 ──
VLM_MODEL: str = os.getenv("VLM_MODEL", "claude-haiku-4-5-20251001")
VLM_DAILY_CAP: int = int(os.getenv("VLM_DAILY_CAP", "200"))
VLM_MIN_CONF: float = float(os.getenv("VLM_MIN_CONF", "0.8"))            # fine 판정 채택 임계
VLM_ITEM_MIN_CONF: float = float(os.getenv("VLM_ITEM_MIN_CONF", "0.6"))  # 품목 생성 채택 임계
