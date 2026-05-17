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
