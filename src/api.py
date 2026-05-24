"""FastAPI app 정의."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from src import config
from src.cam_renderer import render_overlay_png_base64
from src.classes import ClassRegistry
from src.inference import get_active_meta, get_classifier, reset_classifier
from src.preprocess import ImageDecodeError, preprocess_both
from src.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    LabelsResponse,
    ModelVersionResponse,
    PredictionResponse,
    PredictionWithCamResponse,
    PredictionWithMaskResponse,
    ReloadModelResponse,
    ServiceInfo,
)
from src.segment import get_segmenter
from src.uploads import get_recorder, reset_recorder


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier = get_classifier()
    meta = get_active_meta()
    ClassRegistry.load()
    print(f"[startup] color model: {classifier.model_path}")
    print(f"[startup] edge model: {classifier.edge_model_path or '(disabled)'}")
    print(f"[startup] inference mode: "
          f"{'ensemble (color+edge)' if classifier.has_edge_stream else 'single (color)'}")
    if meta is not None:
        print(f"[startup] remote model version: v{meta.version} "
              f"(accuracy={meta.test_accuracy}, feedback={meta.feedback_count})")
    else:
        print("[startup] remote model version: (fallback — Supabase 에 active row 없음)")
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


@app.get("/model/latest", response_model=ModelVersionResponse, tags=["meta"])
def model_latest() -> ModelVersionResponse:
    """현재 서비스가 사용 중인 모델 버전 메타데이터.

    Flutter 앱이 부팅 시 호출 → 자신의 캐시 버전과 비교 → 더 새 게 있으면
    color_url / edge_url 로 직접 다운로드.
    """
    meta = get_active_meta()
    if meta is None:
        return ModelVersionResponse(is_fallback=True)
    return ModelVersionResponse(
        version=meta.version,
        color_url=meta.color_url,
        edge_url=meta.edge_url,
        color_sha256=meta.color_sha256,
        edge_sha256=meta.edge_sha256,
        test_accuracy=meta.test_accuracy,
        num_classes=meta.num_classes,
        class_labels=meta.class_labels,
        feedback_count=meta.feedback_count,
        is_fallback=False,
    )


@app.post("/admin/reload-model", response_model=ReloadModelResponse, tags=["admin"])
def reload_model() -> ReloadModelResponse:
    """모델 강제 재로드 — Supabase 의 최신 active 버전을 다시 fetch.

    retrain.py 가 새 ONNX 를 publish 한 직후 호출하면 즉시 반영됨
    (그렇지 않으면 다음 서버 재시작까지 옛 모델 그대로).
    """
    prev_meta = get_active_meta()
    prev_version = prev_meta.version if prev_meta else None

    reset_classifier()
    get_classifier()  # 재로드 트리거 — model_loader.resolve_model_paths() 다시 호출됨
    new_meta = get_active_meta()
    new_version = new_meta.version if new_meta else None

    return ReloadModelResponse(
        reloaded=True,
        previous_version=prev_version,
        new_version=new_version,
        is_fallback=new_meta is None,
    )


async def _read_and_validate_image(image: UploadFile) -> bytes:
    """공통 헬퍼 — 업로드 검증 + bytes 반환."""
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
    return raw


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    image: UploadFile = File(..., description="분류할 폐기물 이미지"),
) -> PredictionResponse:
    raw = await _read_and_validate_image(image)

    classifier = get_classifier()
    try:
        color_input, edge_input = preprocess_both(raw)
    except ImageDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    result = classifier.predict(color_input, edge_input)

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


@app.post(
    "/predict-with-cam",
    response_model=PredictionWithCamResponse,
    tags=["inference"],
)
async def predict_with_cam(
    image: UploadFile = File(..., description="분류할 폐기물 이미지"),
) -> PredictionWithCamResponse:
    """`/predict` + heatmap PNG (base64 data URI).

    응답의 `cam_base64` 를 그대로 `<img src=...>` / Flutter Image.memory 로 표시.
    모델이 cam-aware ONNX 가 아니면 `cam_available=false` + `cam_base64=null`.
    """
    raw = await _read_and_validate_image(image)

    classifier = get_classifier()
    try:
        color_input, edge_input = preprocess_both(raw)
    except ImageDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    result = classifier.predict(color_input, edge_input, want_cam=True)
    cam_array = result.pop("cam", None)

    cam_b64: str | None = None
    if cam_array is not None:
        try:
            cam_b64 = render_overlay_png_base64(raw, cam_array)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] CAM rendering failed: {exc}")

    # /predict 와 동일하게 upload 기록 (active learning 데이터로 동등하게 누적)
    upload_id: str | None = None
    if config.COLLECT_USER_UPLOADS:
        try:
            upload_id = get_recorder().record_prediction(
                image_bytes=raw,
                content_type=image.content_type or "application/octet-stream",
                prediction=result,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] upload collection failed: {exc}")

    return PredictionWithCamResponse(
        **result,
        upload_id=upload_id,
        cam_base64=cam_b64,
        cam_available=classifier.has_cam_output,
    )


@app.post(
    "/predict-with-mask",
    response_model=PredictionWithMaskResponse,
    tags=["inference"],
)
async def predict_with_mask(
    image: UploadFile = File(..., description="분류할 폐기물 이미지"),
) -> PredictionWithMaskResponse:
    """`/predict` + 객체 누끼(saliency mask + bbox).

    앱이 mask 로 배경을 dim 하고 객체 위에 단일 재질 라벨을 오버레이.
    grid(9타일) 방식 대체 — 객체 하나에 라벨 하나로 깔끔하게.
    """
    raw = await _read_and_validate_image(image)

    classifier = get_classifier()
    try:
        color_input, edge_input = preprocess_both(raw)
    except ImageDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    result = classifier.predict(color_input, edge_input)

    # 누끼 (saliency segmentation → cutout)
    seg = {"cutout_base64": None, "bbox_norm": None, "object_ratio": 0.0}
    try:
        seg = get_segmenter().segment(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] segmentation failed: {exc}")

    upload_id: str | None = None
    if config.COLLECT_USER_UPLOADS:
        try:
            upload_id = get_recorder().record_prediction(
                image_bytes=raw,
                content_type=image.content_type or "application/octet-stream",
                prediction=result,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] upload collection failed: {exc}")

    return PredictionWithMaskResponse(
        **result,
        upload_id=upload_id,
        cutout_base64=seg["cutout_base64"],
        bbox_norm=seg["bbox_norm"],
        object_ratio=seg["object_ratio"],
    )


@app.post("/segment", tags=["inference"])
async def segment(
    image: UploadFile = File(..., description="누끼할 이미지"),
) -> dict:
    """객체 누끼만 — 분류 없이 cutout + bbox 반환 (앱이 분류와 병렬 호출)."""
    raw = await _read_and_validate_image(image)
    try:
        return get_segmenter().segment(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] segmentation failed: {exc}")
        return {"cutout_base64": None, "bbox_norm": None, "object_ratio": 0.0}


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
