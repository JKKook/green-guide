"""FastAPI app 정의."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from src import config
from src.classes import ClassRegistry
from src.inference import get_classifier, reset_classifier
from src.preprocess import ImageDecodeError, preprocess
from src.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    LabelsResponse,
    PredictionResponse,
    ServiceInfo,
)
from src.uploads import get_recorder, reset_recorder


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier = get_classifier()
    ClassRegistry.load()
    print(f"[startup] loaded model arch={classifier.arch} from {classifier.model_path}")
    print(f"[startup] class registry: "
          f"{len(ClassRegistry.all_slugs())} total "
          f"({len(ClassRegistry.trained_slugs())} trained)")
    print(f"[startup] user upload collection: "
          f"{'ENABLED' if config.COLLECT_USER_UPLOADS else 'disabled'}")
    yield
    reset_classifier()
    reset_recorder()


app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.CORS_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=ServiceInfo, tags=["meta"])
def root() -> ServiceInfo:
    classifier = get_classifier()
    return ServiceInfo(
        name=config.API_TITLE,
        version=config.API_VERSION,
        model_arch=classifier.arch,
        model_path=str(classifier.model_path),
        class_labels=ClassRegistry.trained_slugs(),
        max_upload_size_bytes=config.MAX_UPLOAD_SIZE_BYTES,
    )


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/labels", response_model=LabelsResponse, tags=["meta"])
def labels() -> LabelsResponse:
    """전체 클래스 목록 (학습된 것 + 신규 미학습) + 메타데이터."""
    classes = ClassRegistry.load()
    return LabelsResponse(
        labels=[c.slug for c in classes],
        count=len(classes),
        classes=[c.to_api_dict() for c in classes],
    )


@app.post("/reload-classes", tags=["admin"])
def reload_classes() -> dict[str, int]:
    """레지스트리 강제 리로드 (관리자용)."""
    ClassRegistry.reload()
    return {
        "total": len(ClassRegistry.all_slugs()),
        "trained": len(ClassRegistry.trained_slugs()),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    image: UploadFile = File(..., description="분류할 폐기물 이미지"),
) -> PredictionResponse:
    if image.content_type not in config.SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"지원하지 않는 파일 형식: {image.content_type}. "
                   f"지원 형식: {', '.join(config.SUPPORTED_CONTENT_TYPES)}",
        )

    raw = await image.read()
    if len(raw) > config.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"파일이 너무 큼: {len(raw):,} bytes > "
                   f"{config.MAX_UPLOAD_SIZE_BYTES:,} bytes",
        )
    if len(raw) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="빈 파일이 업로드됨",
        )

    classifier = get_classifier()
    try:
        model_input = preprocess(raw, classifier.arch)
    except ImageDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    result = classifier.predict(model_input)

    upload_id: str | None = None
    if config.COLLECT_USER_UPLOADS:
        try:
            upload_id = get_recorder().record_prediction(
                image_bytes=raw,
                content_type=image.content_type or "application/octet-stream",
                prediction=result,
            )
        except Exception as exc:  # noqa: BLE001
            # 수집 실패는 추론 자체를 막지 않도록 — 로그만 남기고 응답은 정상
            print(f"[warn] upload collection failed: {exc}")

    return PredictionResponse(**result, upload_id=upload_id)


@app.post("/feedback", response_model=FeedbackResponse, tags=["learning"])
def feedback(req: FeedbackRequest) -> FeedbackResponse:
    if not config.COLLECT_USER_UPLOADS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="사용자 업로드 수집이 비활성화됨 (WASTE_API_COLLECT_UPLOADS=false)",
        )

    if not req.confirmed and req.corrected_label is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="confirmed=false 일 때는 corrected_label 이 필요합니다",
        )
    if req.confirmed and req.corrected_label is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="confirmed=true 일 때는 corrected_label 을 지정하지 마세요",
        )
    if req.corrected_label is not None and not ClassRegistry.is_valid_label(req.corrected_label):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"유효하지 않은 라벨: {req.corrected_label!r}. "
                   f"지원: {', '.join(ClassRegistry.all_slugs())}",
        )

    try:
        row = get_recorder().record_feedback(
            upload_id=req.upload_id,
            confirmed=req.confirmed,
            corrected_label=req.corrected_label,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return FeedbackResponse(
        upload_id=req.upload_id,
        feedback_status=row["feedback_status"],
        feedback_label=row["feedback_label"],
    )
