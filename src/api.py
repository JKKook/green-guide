"""FastAPI app 정의."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from src import config
from src.cam_renderer import render_overlay_png_base64
from src.classes import ClassRegistry
from src.inference import get_active_meta, get_classifier, reset_classifier
from src.preprocess import ImageDecodeError, normalize_orientation, preprocess_both
from src.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    LabelsResponse,
    MaterialRegion,
    ModelVersionResponse,
    PredictionResponse,
    PredictionWithCamResponse,
    PredictionWithMaskResponse,
    PredictionWithRegionsResponse,
    ReloadModelResponse,
    ServiceInfo,
)
from src.hand_detector import get_hand_detector
from src.regions import extract_regions, render_hatching
from src.segment import get_segmenter
from src.stage1_classifier import get_stage1_classifier
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
    # EXIF 회전 태그를 픽셀에 적용 — Flutter 표시(태그 적용)와 서버 처리
    # (분류·CAM·빗금·누끼) 의 방향을 일치시킴.
    return normalize_orientation(raw)


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


def _auto_crop_to_object(raw: bytes, expand: float = 0.10) -> bytes:
    """u2netp 으로 객체 bbox 검출 → bbox + padding 으로 크롭 → JPEG bytes 반환.

    bbox 검출 실패 또는 크롭 너무 작으면 원본 그대로. /predict-centered 와
    /predict-with-regions 가 공통 사용. 객체 중심 입력으로 표준화 → 잡배경 영향 ↓.
    """
    import io  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    try:
        seg = get_segmenter().segment(raw)
        bbox_norm = seg.get("bbox_norm")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] segment for auto-crop failed: {exc}")
        return raw

    if not bbox_norm:
        return raw

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        W, H = img.size
        x0 = max(0, int((bbox_norm[0] - expand) * W))
        y0 = max(0, int((bbox_norm[1] - expand) * H))
        x1 = min(W, int((bbox_norm[2] + expand) * W))
        y1 = min(H, int((bbox_norm[3] + expand) * H))
        if x1 - x0 < 64 or y1 - y0 < 64:
            return raw  # 너무 작은 크롭은 의미 없음 — 원본
        buf = io.BytesIO()
        img.crop((x0, y0, x1, y1)).save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] bbox crop failed: {exc}")
        return raw


def _force_non_object_result(reason: str) -> dict:
    """모델 호출 없이 non_object 결과 dict 반환 (손 지배 등 OOD 강제 분기).

    [classifier.predict 와 동일 schema] — predicted_class/index/confidence/
    all_probabilities/model_arch/inference_ms.
    """
    from src.classes import ClassRegistry  # noqa: PLC0415
    labels = list(ClassRegistry.all_slugs())
    probs = {l: 0.0 for l in labels}
    if "non_object" in labels:
        non_idx = labels.index("non_object")
        probs["non_object"] = 1.0
        cls = "non_object"
    else:
        # fallback — non_object 가 DB 에 없으면 etc 로
        non_idx = labels.index("etc") if "etc" in labels else 0
        probs[labels[non_idx]] = 1.0
        cls = labels[non_idx]
    return {
        "predicted_class": cls,
        "predicted_index": non_idx,
        "confidence": 1.0,
        "all_probabilities": probs,
        "model_arch": f"hand-detected: {reason}",
        "inference_ms": 0.0,
    }


@app.post("/predict-centered", response_model=PredictionResponse, tags=["inference"])
async def predict_centered(
    image: UploadFile = File(..., description="분류할 폐기물 이미지 (객체 자동 크롭 후 분류)"),
) -> PredictionResponse:
    """객체 자동 크롭 → 분류. Smart capture 가 사용.

    Two-stage Cascade 파이프라인:
      Stage 0 (MediaPipe Hands): 손 50%+ → 모델 호출 없이 non_object
      Stage 1 (MobileNetV3-Small binary): waste/non_object 이진 판정
      Stage 2 (ResNet18 13-class): waste 면 정밀 분류
    """
    raw = await _read_and_validate_image(image)

    # ─ Stage 0: 손 dominance 체크 ──────────────────────
    try:
        hand_area = get_hand_detector().hand_area_ratio(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] hand detection failed: {exc}")
        hand_area = 0.0

    if hand_area >= 0.50:
        result = _force_non_object_result(f"hand area {hand_area:.2f} >= 0.50")
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
        return PredictionResponse(**result, upload_id=upload_id)

    # ─ Stage 1: binary classifier — waste/non-waste 판정 ─
    try:
        is_waste, waste_prob = get_stage1_classifier().predict(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] stage1 failed: {exc}")
        is_waste, waste_prob = True, 1.0   # fail-open: stage2 로 위임

    if not is_waste:
        result = _force_non_object_result(f"stage1 waste_prob={waste_prob:.3f} < 0.50")
        upload_id = None
        if config.COLLECT_USER_UPLOADS:
            try:
                upload_id = get_recorder().record_prediction(
                    image_bytes=raw,
                    content_type=image.content_type or "application/octet-stream",
                    prediction=result,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] upload collection failed: {exc}")
        return PredictionResponse(**result, upload_id=upload_id)

    # ─ Stage 2: 자동 크롭 + 13-class 분류 ─────────────
    cropped_raw = _auto_crop_to_object(raw)

    classifier = get_classifier()
    try:
        color_input, edge_input = preprocess_both(cropped_raw)
    except ImageDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc),
        ) from exc

    result = classifier.predict(color_input, edge_input)

    # upload 기록 (원본 이미지 — 사용자 피드백·재학습은 원본 기준)
    upload_id = None
    if config.COLLECT_USER_UPLOADS:
        try:
            upload_id = get_recorder().record_prediction(
                image_bytes=raw,
                content_type=image.content_type or "application/octet-stream",
                prediction=result,
            )
        except Exception as exc:  # noqa: BLE001
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


@app.post(
    "/predict-with-regions",
    response_model=PredictionWithRegionsResponse,
    tags=["inference"],
)
async def predict_with_regions(
    image: UploadFile = File(..., description="분류할 폐기물 이미지"),
) -> PredictionWithRegionsResponse:
    """`/predict` + 다중재질 영역 분석 (Cascade + CAM-argmax + u2netp + 손 제외).

    파이프라인:
      Stage 1 (binary): waste 아니면 → non_object 응답 (regions 분석 skip)
      Stage 2 (regions): waste 면 객체 크롭 + 손 mask 제외 + CAM 분석
    """
    raw_orig = await _read_and_validate_image(image)

    # Stage 1: binary waste/non-waste 판정
    try:
        is_waste, waste_prob = get_stage1_classifier().predict(raw_orig)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] stage1 failed: {exc}")
        is_waste, waste_prob = True, 1.0

    if not is_waste:
        result = _force_non_object_result(f"stage1 waste_prob={waste_prob:.3f} < 0.50")
        upload_id_n: str | None = None
        if config.COLLECT_USER_UPLOADS:
            try:
                upload_id_n = get_recorder().record_prediction(
                    image_bytes=raw_orig,
                    content_type=image.content_type or "application/octet-stream",
                    prediction=result,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] upload collection failed: {exc}")
        return PredictionWithRegionsResponse(
            **result, upload_id=upload_id_n,
            overlay_base64=None, regions=[], grid_h=0, grid_w=0,
        )

    raw = _auto_crop_to_object(raw_orig)

    classifier = get_classifier()
    try:
        color_input, edge_input = preprocess_both(raw)
    except ImageDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc),
        ) from exc

    result, cam = classifier.region_cam(color_input)

    overlay_b64: str | None = None
    regions_out: list[MaterialRegion] = []
    grid_h = grid_w = 0
    if cam is not None:
        try:
            grid_h, grid_w = cam.shape[1], cam.shape[2]
            mask_grid = get_segmenter().object_mask_grid(raw, grid_h)
            # 손 mask 검출 → object mask 에서 손 영역 제외
            try:
                hand_grid = get_hand_detector().mask_grid(raw, grid_h)
                mask_grid = mask_grid * (1.0 - hand_grid).clip(0.0, 1.0)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] hand mask grid failed: {exc}")
            labels = list(classifier.labels)
            regions = extract_regions(cam, mask_grid, labels)
            if regions:
                overlay_b64 = render_hatching(
                    raw, regions, grid_h, grid_w, ClassRegistry.color_map(),
                )
                regions_out = [
                    MaterialRegion(
                        slug=r["slug"], bbox_norm=r["bbox_norm"],
                        avg_conf=r["avg_conf"], cell_count=len(r["cells"]),
                    )
                    for r in regions
                ]
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] region analysis failed: {exc}")

    # upload 기록은 원본 이미지 (사용자 피드백·재학습 일관성)
    upload_id: str | None = None
    if config.COLLECT_USER_UPLOADS:
        try:
            upload_id = get_recorder().record_prediction(
                image_bytes=raw_orig,
                content_type=image.content_type or "application/octet-stream",
                prediction=result,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] upload collection failed: {exc}")

    return PredictionWithRegionsResponse(
        **result,
        upload_id=upload_id,
        overlay_base64=overlay_b64,
        regions=regions_out,
        grid_h=grid_h,
        grid_w=grid_w,
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
