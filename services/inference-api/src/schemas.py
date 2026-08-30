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
    # 진단용 — 어느 Supabase 프로젝트를 보는지 (호스트는 공개 정보, 키 아님)
    supabase_host: str | None = None


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
    all_probabilities: dict[str, float] = Field(..., description="전체 클래스 확률 분포 (동적 N개)")
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


class MaterialRegion(BaseModel):
    """다중재질 분석에서 검출된 한 재질 영역."""
    slug: str
    bbox_norm: list[float] = Field(..., description="[x0,y0,x1,y1] 0~1 (라벨 위치용)")
    avg_conf: float
    cell_count: int


class PredictionWithRegionsResponse(PredictionResponse):
    """`/predict-with-regions` 응답 — 예측 + 다중재질 영역 + 빗금 오버레이."""

    overlay_base64: str | None = Field(
        default=None,
        description="원본 이미지에 영역별 빗금을 그린 JPEG (data URI). 누끼 이미지 대신 표시.",
    )
    regions: list[MaterialRegion] = Field(
        default_factory=list,
        description="검출된 재질 영역들 (확실히 다른 재질만). 1개면 단일재질, 2+면 다중재질.",
    )
    grid_h: int = 0
    grid_w: int = 0


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


class FineTopEntry(BaseModel):
    """세부 클래스 확률 상위 항목."""

    slug: str
    prob: float


class PredictionHierResponse(BaseModel):
    """`/predict-hier` 응답 — 계층(대분류→세부) 분류.

    display_level 로 표시 깊이를 알린다:
    - "fine":   세부까지 확신 (fine_class 사용)
    - "coarse": 대분류만 확신 (coarse_class 사용, fine_class 는 null)
    - "reject": 대분류도 불확실 → display_class="etc" (재촬영/캐치올)
    """

    display_level: str = Field(description='"fine" | "coarse" | "reject"')
    display_class: str = Field(description="사용자에게 표시할 클래스 slug (게이트 적용 결과)")
    coarse_class: str = Field(description="대분류 slug (롤업 top1) — 항상 존재")
    coarse_confidence: float = Field(description="대분류 확신도 (children 확률 합)")
    fine_class: str | None = Field(default=None, description="세부 slug — 게이트 통과 시에만")
    fine_confidence: float = Field(description="세부 top1 확률 (게이트 무관 참고값)")
    fine_margin: float = Field(description="세부 top1-top2 격차")
    coarse_probabilities: dict[str, float] = Field(description="대분류 전체 확률 분포")
    fine_top5: list[FineTopEntry] = Field(description="세부 상위 5개 (참고용)")
    model_arch: str
    inference_ms: float
    upload_id: str | None = Field(default=None, description="user_uploads 기록 id")
    ood_distance: float | None = Field(
        default=None,
        description="최근접 prototype 임베딩 cosine 거리 (낮을수록 in-distribution)")
    ood_reject: bool = Field(
        default=False,
        description="임베딩이 학습 분포 밖 → softmax 무관 reject 처리됨")
    evidence: list["EvidenceItem"] | None = Field(
        default=None,
        description="시맨틱 증거 — 이미지 텍스트에서 인식된 재질/정체 단서 (융합 반영됨)")
    generated_item: "GeneratedItem | None" = Field(
        default=None,
        description="VLM 이 재질 taxonomy 밖에서 인식한 품목 — 재질 필드와 별개의 "
                    "스트림 안내 (사전 밖 품목 자유 생성 + 닫힌 스트림)")


class StreamInfo(BaseModel):
    """배출 스트림(목적지) 안내 — src/streams.py 닫힌 목록의 한 항목."""

    slug: str
    display_name: str
    summary: str
    how_to: list[str]


class GeneratedItem(BaseModel):
    """VLM 자유 생성 품목 판정 (예: 소파 → 대형폐기물 신고)."""

    item_name: str = Field(description="VLM 이 인식한 품목명 (자유 텍스트)")
    stream: StreamInfo = Field(description="배출 스트림 안내 (닫힌 목록에서 선택됨)")
    condition: str | None = Field(
        default=None, description="안내가 갈리는 상태 조건 (재질·파손·오염 등)")
    confidence: float


class EvidenceItem(BaseModel):
    """OCR 기반 시맨틱 증거 한 건 (SEMANTIC_FUSION_PLAN 신호①)."""

    type: str = Field(description='"mark"=분리배출표시/재질어, "text"=정체어')
    token: str = Field(description="매칭된 어휘 토큰 (예: 무색페트)")
    matched_text: str = Field(description="OCR 이 읽은 원문")
    mapped_class: str = Field(description="증거가 가리키는 클래스 slug")
    score: float = Field(description="OCR 인식 확신도")


class TaxonomyResponse(BaseModel):
    """`/taxonomy` 응답 — 계층 구조 메타 (앱이 롤업·표시에 사용)."""

    version: str
    fine_labels: list[str]
    coarse_labels: list[str]
    fine_to_coarse: dict[str, str]
    gate: dict[str, float]


class ObjectCandidate(BaseModel):
    """`/predict-objects` 의 객체 후보 하나 — 계층 분류 결과 포함."""

    bbox_norm: list[float] = Field(description="[x0,y0,x1,y1] 0~1 정규화")
    display_level: str = Field(description='"fine" | "coarse" | "reject"')
    display_class: str
    coarse_class: str
    coarse_confidence: float
    fine_class: str | None = None
    fine_confidence: float = 0.0
    coarse_probabilities: dict[str, float] = Field(default_factory=dict)


class PredictObjectsResponse(BaseModel):
    """`/predict-objects` 응답 — 혼재 장면의 객체 후보들 (면적 내림차순).

    saliency 성분 기반이라 붙은 객체는 병합될 수 있음 — 탭-투-셀렉트 병용.
    후보가 1개면 단일 객체 장면.
    """

    objects: list[ObjectCandidate]
    count: int
    inference_ms: float
