"""Pydantic 요청·응답 스키마."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# pydantic 2.x 는 "model_" 로 시작하는 필드명을 보호 namespace 로 경고
# 도메인 용어상 "model" 사용이 자연스러우므로 비활성화
_ALLOW_MODEL_FIELDS = ConfigDict(protected_namespaces=())


class ServiceInfo(BaseModel):
    model_config = _ALLOW_MODEL_FIELDS

    name: str
    version: str
    model_arch: str
    model_path: str
    class_labels: list[str]
    max_upload_size_bytes: int


class HealthResponse(BaseModel):
    status: str = "ok"


class LabelsResponse(BaseModel):
    labels: list[str]
    count: int
    classes: list[dict] | None = Field(
        default=None,
        description="(선택) 클래스 전체 메타데이터 — display_name, color, icon, "
                    "how_to, caution, bin, trained_in_model 등",
    )


class PredictionResponse(BaseModel):
    model_config = _ALLOW_MODEL_FIELDS

    predicted_class: str = Field(..., description="가장 높은 확률의 클래스 이름")
    predicted_index: int = Field(..., ge=0, description="가장 높은 확률의 클래스 인덱스")
    confidence: float = Field(..., ge=0.0, le=1.0, description="예측 클래스의 확률")
    all_probabilities: dict[str, float] = Field(..., description="6개 클래스 전체 확률 분포")
    model_arch: str = Field(..., description="사용된 모델 아키텍처 (mlp | cnn)")
    inference_ms: float = Field(..., description="ONNX 추론 소요 시간 (밀리초)")
    upload_id: str | None = Field(
        default=None,
        description="Supabase 에 기록된 업로드 ID. /feedback 에서 참조. 수집 비활성 시 null.",
    )


class FeedbackRequest(BaseModel):
    upload_id: str = Field(..., description="/predict 응답의 upload_id")
    confirmed: bool = Field(..., description="예측이 맞으면 true, 틀리면 false")
    corrected_label: str | None = Field(
        default=None,
        description="confirmed=false 일 때 사용자가 지정한 올바른 클래스",
    )


class FeedbackResponse(BaseModel):
    upload_id: str
    feedback_status: str
    feedback_label: str


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


class ModelVersionResponse(BaseModel):
    """현재 active 모델 버전 메타데이터 — 클라이언트(앱) 가 자체 캐시와 비교용."""
    model_config = _ALLOW_MODEL_FIELDS

    version: str | None = Field(
        default=None,
        description="active 버전 문자열. null 이면 서버가 fallback(번들/sibling) 사용 중.",
    )
    color_url: str | None = None
    edge_url: str | None = None
    color_sha256: str | None = None
    edge_sha256: str | None = None
    test_accuracy: float | None = None
    num_classes: int | None = None
    class_labels: list[str] | None = None
    feedback_count: int | None = None
    is_fallback: bool = Field(
        ...,
        description="true 면 Supabase active row 가 없어 로컬 fallback 모델 사용 중",
    )


class ReloadModelResponse(BaseModel):
    model_config = _ALLOW_MODEL_FIELDS

    reloaded: bool
    previous_version: str | None = None
    new_version: str | None = None
    is_fallback: bool


class PredictionWithCamResponse(PredictionResponse):
    """`/predict-with-cam` 응답 — 예측 + heatmap overlay (base64 PNG data URI)."""

    cam_base64: str | None = Field(
        default=None,
        description="원본 이미지 + heatmap alpha-blend 한 PNG (data:image/png;base64,…). "
                    "현재 모델이 cam-aware ONNX 가 아니면 null.",
    )
    cam_available: bool = Field(
        ...,
        description="false 면 서버가 단일 출력 ONNX 사용 중이라 CAM 생성 불가",
    )


class PredictionWithMaskResponse(PredictionResponse):
    """`/predict-with-mask` 응답 — 예측 + 객체 누끼(cutout)."""

    cutout_base64: str | None = Field(
        default=None,
        description="객체만 남기고 배경 투명 처리한 RGBA PNG (data URI). "
                    "앱이 [dim 원본] 위에 겹쳐 객체 부각 + 라벨 오버레이.",
    )
    bbox_norm: list[float] | None = Field(
        default=None,
        description="객체 bounding box [x0,y0,x1,y1] 0~1 정규화. 라벨 위치용. 없으면 null.",
    )
    object_ratio: float = Field(
        default=0.0,
        description="객체가 프레임에서 차지하는 면적 비율 (0~1).",
    )
