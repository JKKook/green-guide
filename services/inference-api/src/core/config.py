"""waste-api 전역 설정."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # 아래 os.getenv 전에 .env 반영 (이미 설정된 env 는 덮어쓰지 않음)


PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]  # src/core/config.py → 레포 루트



def monorepo_root(project_root: Path) -> Path:
    """services/inference-api → 모노레포 루트. 컨테이너(/app)처럼 부모가 하나뿐인
    얕은 경로면 자기 자신을 돌려준다 — parents[1] 이 IndexError 로 import 를
    죽이던 HF Space 배포 회귀(2026-09-11) 방지. sibling 경로는 그 경우 어차피 없다."""
    parents = project_root.parents
    return parents[1] if len(parents) > 1 else project_root


MONOREPO_ROOT: Path = monorepo_root(PROJECT_ROOT)

# 자매 프로젝트의 ONNX 모델 직접 참조 (로컬 개발용)
# services/inference-api → <레포 루트>/ml/classifier
CLASSIFIER_ROOT: Path = MONOREPO_ROOT / "ml" / "classifier"
DEFAULT_MODEL_ARCH: str = "cnn"  # mlp | cnn


def _resolve_model_path() -> Path:
    """Color stream 모델 — 우선순위:
    1. GREENGUIDE_API_MODEL_PATH 환경변수
    2. services/inference-api/models/classifier.onnx  (배포 패키지 내 번들 — Docker 등)
    3. ../../ml/classifier/outputs/models/cnn/classifier.onnx  (로컬 sibling)
    """
    env_path = os.getenv("GREENGUIDE_API_MODEL_PATH")
    if env_path:
        return Path(env_path)

    bundled = PROJECT_ROOT / "models" / "classifier.onnx"
    if bundled.exists():
        return bundled

    return CLASSIFIER_ROOT / "outputs" / "models" / DEFAULT_MODEL_ARCH / "classifier.onnx"


def _resolve_edge_model_path() -> Path | None:
    """Edge stream 모델 (선택). 없으면 ensemble 비활성."""
    env_path = os.getenv("GREENGUIDE_API_EDGE_MODEL_PATH")
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
    os.getenv("GREENGUIDE_API_ENSEMBLE_COLOR_WEIGHT", "0.8"),
)

# 클래스 정의 (greenguide-preprocessor·greenguide-classifier와 동일 순서)
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
    os.getenv("GREENGUIDE_API_COLLECT_UPLOADS", "true").lower() in ("1", "true", "yes")
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
HIER_MODEL_PATH_ENV: str | None = os.getenv("GREENGUIDE_API_HIER_MODEL_PATH")
DINO_WEIGHT: float = float(os.getenv("GREENGUIDE_API_DINO_W", "0"))       # 0 = DINOv2 앙상블 비활성
CAM_PRIOR_WEIGHT: float = float(os.getenv("GREENGUIDE_API_CAM_W", "0.15"))
# OOD 거부 임계 재정의 (없으면 taxonomy.json 의 보정값 → 과거 기본값 순)
OOD_TAU_SOFT: float | None = (
    float(v) if (v := os.getenv("GREENGUIDE_API_OOD_TAU_SOFT")) else None)
OOD_TAU_HARD: float | None = (
    float(v) if (v := os.getenv("GREENGUIDE_API_OOD_TAU_HARD")) else None)

# ── 증거 엔진 (CLIP 정체 / OCR) ──
CLIP_ENABLED: bool = os.getenv("GREENGUIDE_API_CLIP", "1") != "0"
CLIP_PRIOR_WEIGHT: float = float(os.getenv("GREENGUIDE_API_CLIP_W", "0.5"))
CLIP_SCENE_WEIGHT: float = float(os.getenv("GREENGUIDE_API_CLIP_SCENE_W", "0.2"))
OCR_ENABLED: bool = os.getenv("GREENGUIDE_API_OCR", "1") != "0"
OCR_SKIP_CONFIDENCE: float = float(os.getenv("GREENGUIDE_API_OCR_SKIP_CONF", "0.75"))
# 분리배출 표시 최우선 판정 (스마트촬영): 항상 OCR + 몸체 표시가 읽히면 결과 교체
MARK_PRIORITY_SMART: bool = os.getenv("GREENGUIDE_API_MARK_PRIORITY", "1") != "0"
MARK_OVERRIDE_MIN_SCORE: float = float(os.getenv("GREENGUIDE_API_MARK_MIN_SCORE", "0.6"))
MARK_SHORT_MIN_SCORE: float = float(os.getenv("GREENGUIDE_API_MARK_SHORT_MIN_SCORE", "0.75"))  # 2글자 표시(pp·ps) 교체 임계
# 서로 다른 재질 표시가 함께 읽히면(복합 포장) 표시 교체 대신 VLM 중재 — 11-08 b6437a2f
MARK_CONFLICT_VLM: bool = os.getenv("GREENGUIDE_API_MARK_CONFLICT_VLM", "1") == "1"

# ── VLM 폴백 ──
VLM_MODEL: str = os.getenv("VLM_MODEL", "claude-sonnet-5-5")  # 사용자 결정 11-06
VLM_DAILY_CAP: int = int(os.getenv("VLM_DAILY_CAP", "200"))
VLM_MIN_CONF: float = float(os.getenv("VLM_MIN_CONF", "0.8"))            # fine 판정 채택 임계
# Sonnet 전용 채택 임계(기본 없음 = 위 값) — Sonnet 은 확신을 낮게 매긴다 (11-06 gate A 비교)
_SONNET_MIN_CONF = os.getenv("GREENGUIDE_VLM_SONNET_MIN_CONF")
if _SONNET_MIN_CONF and "sonnet" in VLM_MODEL:
    VLM_MIN_CONF = float(_SONNET_MIN_CONF)
VLM_ITEM_MIN_CONF: float = float(os.getenv("VLM_ITEM_MIN_CONF", "0.6"))  # 품목 생성 채택 임계
# 중재 발동 임계 — 대분류 확신도가 이 값 미만이면 VLM 호출(reject·증거 불일치는 항상).
# >1 이면 표시 판정 외 전 장 중재 — 기본 상시(사용자 결정 11-06, 이전 0.55)
VLM_TRIGGER_CONF: float = float(os.getenv("GREENGUIDE_API_VLM_TRIGGER_CONF", "1.01"))
# 중재 R1(기본 켜짐, 11-06): VLM 재질 대분류가 CNN 1위와 같으면 확신 무관 채택 (11-04 gate A 분석)
VLM_AGREE_ACCEPT: bool = os.getenv("GREENGUIDE_API_VLM_AGREE_ACCEPT", "1") == "1"
# 1회차 프롬프트 재질 규칙(기본 켜짐, 11-06): slug 칸에 배출 방법(trash) 대신 재질 (TAXONOMY_V2 §3)
VLM_MATERIAL_RULE: bool = os.getenv("GREENGUIDE_VLM_MATERIAL_RULE", "1") == "1"
# 분류 체계 v2 축(fine_v2·state·mark) 질의 방식
#   off(v1) | single(한 프롬프트) | two_stage(재질 v1 원문 + 축 별도 호출, 기본 11-06)
# GG_VLM 실측(qwen): single 재질 66→54/79 회귀, two_stage 66 유지
# → eval_v2 gate 79 로 셋 비교 후 채택
VLM_TAXONOMY: str = os.getenv("GREENGUIDE_VLM_TAXONOMY", "two_stage")
# two_stage 2회차에 장난감 여부(toy) 질의 → 재질 trash 고정(TAXONOMY_V2 §3)
# 기본 꺼짐 — CNN v2 와 함께 gate 65 측정 후 채택
VLM_TOY: bool = os.getenv("GREENGUIDE_VLM_TOY", "0") == "1"

# /predict-objects 근접 촬영 오분할 방지(기본 꺼짐, 11-07): saliency 성분이 잘게 흩어지면
# (질감·근접 — 키보드 키 등) 후보를 나누지 않고 전체 사진 1개로
OBJECTS_FRAG_WHOLE: bool = os.getenv("GREENGUIDE_API_OBJECTS_FRAG_WHOLE", "0") == "1"

# ── electronics 편향 완화 (W6 수정, ELECTRONICS_BIAS_ANALYSIS.md E절) ──
# 실사용 피드백 51(사용자 라벨)·테스터 35(GG_CNN 라벨) 전후 측정으로 값 확정 — 끄려면 0/빈값
SCENE_AUTOCROP: bool = os.getenv("GREENGUIDE_API_SCENE_AUTOCROP", "1") == "1"       # 장면 경로 u2netp 물체 크롭(실패=전체 프레임)
NONOBJ_RAW_GUARD: float = float(os.getenv("GREENGUIDE_API_NONOBJ_RAW_GUARD", "0.5"))  # 마스킹 전 non_object ≥ 이 값이면 재분배 대신 reject
ELEC_LOGIT_BIAS: float = float(os.getenv("GREENGUIDE_API_ELEC_LOGIT_BIAS", "0.5"))   # electronics 로짓 감산 — 테스터 35로 보정(recall 무손실 최대)
ELEC_COARSE_GATE: float = float(os.getenv("GREENGUIDE_API_ELEC_COARSE_GATE", "0.7"))  # electronics 전용 대분류 임계, 미달=reject→VLM 중재
ELEC_CLIP_MIN: float = float(os.getenv("GREENGUIDE_API_ELEC_CLIP_MIN", "0.3"))       # 장면 electronics 판정에 CLIP 전자제품 컨셉 < 이 값이면 억제
# 실험만(기본 꺼짐): 장면 CLIP 부스트 prior — 51장 효과 미미(오탐 16→14) / 클래스별 상시 VLM 중재 — 호출 3→24/51
ELEC_CLIP_PRIOR: bool = os.getenv("GREENGUIDE_API_ELEC_CLIP_PRIOR", "0") == "1"
VLM_ARBITRATE_CLASSES: frozenset[str] = frozenset(
    c for c in os.getenv("GREENGUIDE_API_VLM_ARBITRATE_CLASSES", "").split(",") if c)
