"""FastAPI app 정의."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
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
    ObjectCandidate,
    PredictionHierResponse,
    PredictionResponse,
    PredictObjectsResponse,
    PredictionWithCamResponse,
    PredictionWithMaskResponse,
    PredictionWithRegionsResponse,
    ReloadModelResponse,
    ServiceInfo,
    TaxonomyResponse,
)
from src.hand_detector import get_hand_detector
from src.regions import extract_regions, render_hatching
from src.segment import get_segmenter
from src.dinov2_classifier import get_dinov2_classifier
from src.stage1_classifier import get_stage1_classifier
from src.uploads import get_recorder, reset_recorder


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier = get_classifier()
    meta = get_active_meta()
    # Supabase 불가 시에도 부팅은 계속 — 레지스트리는 요청 시 재시도됨.
    # (설계 원칙: "API 는 항상 부팅한다" — model_loader 와 동일한 강건성)
    try:
        ClassRegistry.load()
    except Exception as exc:  # noqa: BLE001
        print(f"[startup][warn] class registry 로드 실패 (Supabase 미접속?): {exc}")
    print(f"[startup] color model: {classifier.model_path}")
    print(f"[startup] edge model: {classifier.edge_model_path or '(disabled)'}")
    print(f"[startup] inference mode: "
          f"{'ensemble (color+edge)' if classifier.has_edge_stream else 'single (color)'}")
    if meta is not None:
        print(f"[startup] remote model version: v{meta.version} "
              f"(accuracy={meta.test_accuracy}, feedback={meta.feedback_count})")
    else:
        print("[startup] remote model version: (fallback — Supabase 에 active row 없음)")
    try:
        print(f"[startup] class registry: "
              f"{len(ClassRegistry.all_slugs())} total "
              f"({len(ClassRegistry.trained_slugs())} trained)")
    except Exception:  # noqa: BLE001
        print("[startup] class registry: (미로드 — 요청 시 재시도)")
    print(f"[startup] user upload collection: "
          f"{'ENABLED' if config.COLLECT_USER_UPLOADS else 'disabled'}")
    # DINOv2 미리 로드 (첫 요청 지연 회피)
    dino = get_dinov2_classifier()
    print(f"[startup] dinov2 classifier: "
          f"{'ENABLED' if dino.available else 'disabled (model 없음)'}")
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


@app.get("/taxonomy", response_model=TaxonomyResponse, tags=["meta"])
def taxonomy() -> TaxonomyResponse:
    """계층 taxonomy 메타 — 대분류/세부 라벨, 롤업 매핑, 게이트 임계.

    계층 모델 미배치 시 404 (앱은 flat 모드로 fallback).
    """
    from src.hier_inference import get_hier_classifier  # noqa: PLC0415
    try:
        clf = get_hier_classifier()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    t = clf.taxonomy
    return TaxonomyResponse(
        version=t.get("version", "?"),
        fine_labels=t["fine_labels"],
        coarse_labels=t["coarse_labels"],
        fine_to_coarse=t["fine_to_coarse"],
        gate=t["gate"],
    )


@app.post("/predict-hier", response_model=PredictionHierResponse, tags=["inference"])
async def predict_hier(
    image: UploadFile = File(..., description="분류할 폐기물 이미지"),
    tap_x: float | None = Form(default=None, ge=0.0, le=1.0),
    tap_y: float | None = Form(default=None, ge=0.0, le=1.0),
) -> PredictionHierResponse:
    """계층 분류 — 대분류(항상) + 세부(신뢰도 게이트 통과 시).

    기존 /predict 와 독립적인 추가 엔드포인트 (하위호환 유지).
    display_level 로 표시 깊이 판단: fine → 세부 카드, coarse → 대분류만,
    reject → 재촬영/etc 안내.

    tap_x/tap_y (정규화 0~1, EXIF 적용 후 이미지 기준): 탭-투-셀렉트.
    혼재 장면에서 사용자가 지목한 객체의 saliency 성분만 크롭해 분류.
    """
    from src.hier_inference import (  # noqa: PLC0415
        get_hier_classifier, predict_best_rotation,
    )

    raw = await _read_and_validate_image(image)
    try:
        clf = get_hier_classifier()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"계층 모델 미배치: {exc}",
        ) from exc

    # ─ 검증된 캐스케이드 방어선 재사용 (predict-centered 와 동일) ─
    # Stage 0: 손 dominance / Stage 1: waste 이진 게이트 — 실물 비폐기물
    # (손바닥·마우스 등) 이 confident-wrong 으로 통과하는 것을 차단.
    forced_reason: str | None = None
    try:
        hand_area = get_hand_detector().hand_area_ratio(raw)
        if hand_area >= 0.50:
            forced_reason = f"hand area {hand_area:.2f} >= 0.50"
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] hand detection failed: {exc}")
    if forced_reason is None:
        try:
            is_waste, waste_prob = get_stage1_classifier().predict(raw)
            if not is_waste:
                forced_reason = f"stage1 waste_prob={waste_prob:.3f} < 0.50"
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] stage1 failed: {exc}")  # fail-open

    if forced_reason is not None:
        result = {
            "display_level": "reject",
            "display_class": "non_object",
            "coarse_class": "non_object",
            "coarse_confidence": 1.0,
            "fine_class": None,
            "fine_confidence": 0.0,
            "fine_margin": 0.0,
            "coarse_probabilities": {"non_object": 1.0},
            "fine_top5": [],
            "model_arch": f"cascade-gate: {forced_reason}",
            "inference_ms": 0.0,
        }
        return PredictionHierResponse(**result)

    # 장면 분류는 풀프레임 — v2 시절엔 u2 자동크롭이 +2pp 였으나 v6+회전TTA
    # 에선 역전 (실측 실사용 51장: TTA+풀 39 vs TTA+u2크롭 25). 크롭은 문맥을
    # 잃고 saliency 오검출 시 엉뚱한 영역을 자르는 위험이 TTA 이득을 상쇄함.
    # 탭 좌표가 오면 탭 지점의 saliency 성분만 크롭 (탭-투-셀렉트 — 기능상 필수).
    tap_region: list[float] | None = None
    if tap_x is not None and tap_y is not None:
        cropped_raw, tap_region = _crop_at_tap(raw, tap_x, tap_y)
    else:
        cropped_raw = raw

    # 시맨틱 증거 융합 (SEMANTIC_FUSION_PLAN §3) — 경로별 신호 구성 (51장 실측 근거):
    #   장면(풀프레임): OCR 만 — CLIP 은 혼재 장면 center-crop 정체 오인으로 역효과
    #   탭(고립 crop):  OCR × CLIP 정체(w0.5, 25→29) × CAM 영역 재질(w0.15, 신호④)
    # 실패는 각각 격리 — 증거 없으면 기존 경로와 완전 동일.
    from src.clip_identity import get_clip_identity  # noqa: PLC0415
    from src.semantic_evidence import (  # noqa: PLC0415
        evidence_prior, get_evidence_engine, match_evidence,
    )
    evidence: list[dict] = []
    prior = None

    def _mul(a, b):
        if b is None:
            return a
        return b if a is None else a * b

    try:
        texts = get_evidence_engine().read_texts(cropped_raw)
        evidence = match_evidence(texts)
        prior = _mul(prior, evidence_prior(
            evidence, clf.fine_labels, clf.taxonomy["fine_to_coarse"]))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] semantic evidence failed: {exc}")
    if tap_region is not None:
        try:
            clip_eng = get_clip_identity()
            probs = clip_eng.identity_probs(cropped_raw)
            if probs is not None:
                clip_prior, clip_ev = clip_eng.evidence_prior(
                    probs, clf.fine_labels)
                prior = _mul(prior, clip_prior)
                evidence.extend(clip_ev)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] clip identity failed: {exc}")
        try:
            from src.hier_inference import cam_region_prior  # noqa: PLC0415
            prior = _mul(prior, cam_region_prior(clf, raw, tap_region))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] cam region prior failed: {exc}")

    # 게이트를 통과했다 = stage1 이 '폐기물'로 판정 (또는 fail-open)
    # → 분류기의 non_object 는 모순된 답이므로 마스킹 (실측 +5.9pp)
    # 회전 TTA: 학습 크롭(센서 방향) vs 서빙(EXIF 세움) 분포 어긋남 흡수
    # (실측 실사용 51장: 26 → 37. hier_inference.predict_best_rotation 참고)
    try:
        result = predict_best_rotation(
            clf, cropped_raw, mask_non_object=True, fine_prior=prior)
    except ImageDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc),
        ) from exc
    if evidence:
        result["evidence"] = [
            {k: ev[k] for k in ("type", "token", "matched_text",
                                "mapped_class", "score")}
            for ev in evidence
        ]

    upload_id: str | None = None
    if config.COLLECT_USER_UPLOADS:
        try:
            # user_uploads 스키마와 호환되는 형태로 기록 (게이트 적용 결과 기준)
            upload_id = get_recorder().record_prediction(
                image_bytes=raw,
                content_type=image.content_type or "application/octet-stream",
                prediction={
                    "predicted_class": result["display_class"],
                    "confidence": (
                        result["fine_confidence"]
                        if result["display_level"] == "fine"
                        else result["coarse_confidence"]
                    ),
                    "all_probabilities": result["coarse_probabilities"],
                    "model_arch": result["model_arch"],
                    "inference_ms": result["inference_ms"],
                },
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] upload collection failed: {exc}")

    return PredictionHierResponse(**result, upload_id=upload_id)


@app.post("/predict-objects", response_model=PredictObjectsResponse, tags=["inference"])
async def predict_objects(
    image: UploadFile = File(..., description="혼재 장면 이미지"),
) -> PredictObjectsResponse:
    """탐지-후-분류 — 장면의 객체 후보들을 각각 계층 분류해 반환.

    u2netp saliency 연결 성분으로 객체 후보를 분리(면적 내림차순, 최대 5개),
    각 후보를 bbox+12% 크롭해 계층 분류. 혼재 장면에서 "단일 오답" 대신
    "보이는 물건 N개" 를 제시하는 근거 데이터.
    성분 미검출 시 전체 이미지 1개 후보로 fallback.
    """
    import io as _io  # noqa: PLC0415
    import time as _time  # noqa: PLC0415
    from PIL import Image as _Image  # noqa: PLC0415
    from src.hier_inference import (  # noqa: PLC0415
        get_hier_classifier, predict_best_rotation,
    )
    from src.segment import all_component_bboxes  # noqa: PLC0415

    raw = await _read_and_validate_image(image)
    try:
        clf = get_hier_classifier()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"계층 모델 미배치: {exc}",
        ) from exc

    t0 = _time.perf_counter()
    try:
        bboxes = all_component_bboxes(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] component split failed: {exc}")
        bboxes = []
    if not bboxes:
        bboxes = [[0.0, 0.0, 1.0, 1.0]]

    # 시맨틱 증거 — 전체 프레임 OCR 1회 후 텍스트 위치로 후보별 귀속
    # (후보마다 OCR 재실행 금지 — 비용. SEMANTIC_FUSION_PLAN §1 공간 귀속)
    from src.clip_identity import get_clip_identity  # noqa: PLC0415
    from src.semantic_evidence import (  # noqa: PLC0415
        evidence_prior, get_evidence_engine, match_evidence,
    )
    scene_evidence: list[dict] = []
    try:
        scene_evidence = match_evidence(get_evidence_engine().read_texts(raw))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] semantic evidence failed: {exc}")

    img = _Image.open(_io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    objects: list[ObjectCandidate] = []
    for bb in bboxes:
        x0, y0, x1, y1 = bb
        px, py = (x1 - x0) * 0.12, (y1 - y0) * 0.12
        box = (max(0, int((x0 - px) * w)), max(0, int((y0 - py) * h)),
               min(w, int((x1 + px) * w)), min(h, int((y1 + py) * h)))
        if box[2] - box[0] < 48 or box[3] - box[1] < 48:
            continue
        buf = _io.BytesIO()
        img.crop(box).save(buf, format="JPEG", quality=92)
        prior = evidence_prior(
            scene_evidence, clf.fine_labels,
            clf.taxonomy["fine_to_coarse"], region=bb,
        ) if scene_evidence else None
        # CLIP 정체 — 고립 crop 에서만 유효 (장면 전체는 실측 역효과)
        try:
            probs = get_clip_identity().identity_probs(buf.getvalue())
            if probs is not None:
                clip_prior, _ = get_clip_identity().evidence_prior(
                    probs, clf.fine_labels)
                prior = clip_prior if prior is None else prior * clip_prior
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] clip identity failed: {exc}")
        try:
            r = predict_best_rotation(
                clf, buf.getvalue(), mask_non_object=True, fine_prior=prior)
        except ImageDecodeError:
            continue
        objects.append(ObjectCandidate(
            bbox_norm=bb,
            display_level=r["display_level"],
            display_class=r["display_class"],
            coarse_class=r["coarse_class"],
            coarse_confidence=r["coarse_confidence"],
            fine_class=r["fine_class"],
            fine_confidence=r["fine_confidence"],
            coarse_probabilities=r["coarse_probabilities"],
        ))

    elapsed = (_time.perf_counter() - t0) * 1000
    return PredictObjectsResponse(
        objects=objects, count=len(objects), inference_ms=round(elapsed, 2),
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


def _verify_regions(raw: bytes, regions: list[dict], hier_clf) -> list[dict]:
    """CAM 제안 영역을 크롭 재분류로 확정 (zoom-and-verify, Stage 1-4).

    - reject(불확신) 영역 → 폐기 (스퓨리어스 차단)
    - CAM slug 와 재분류 slug 불일치 → 재분류 결과 채택 (분류기가 심판)
    - avg_conf 는 재분류 확신으로 교체 (검증된 수치)
    """
    import io as _io  # noqa: PLC0415
    from PIL import Image as _Image  # noqa: PLC0415
    from src.preprocess import preprocess_both as _pb  # noqa: PLC0415

    try:
        img = _Image.open(_io.BytesIO(raw)).convert("RGB")
    except Exception:  # noqa: BLE001
        return regions
    W, H = img.size
    verified: list[dict] = []
    for reg in regions[:4]:  # 상위 4개만 (비용 상한)
        x0, y0, x1, y1 = reg["bbox_norm"]
        pw, ph = (x1 - x0) * 0.15, (y1 - y0) * 0.15
        box = (max(0, int((x0 - pw) * W)), max(0, int((y0 - ph) * H)),
               min(W, int((x1 + pw) * W)), min(H, int((y1 + ph) * H)))
        if box[2] - box[0] < 40 or box[3] - box[1] < 40:
            continue
        buf = _io.BytesIO()
        img.crop(box).save(buf, format="JPEG", quality=90)
        try:
            ci, _ = _pb(buf.getvalue())
            r = hier_clf.predict(ci, mask_non_object=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] region verify failed: {exc}")
            verified.append(reg)
            continue
        if r["display_level"] == "reject":
            continue  # CAM 헛제안 폐기
        slug = r["fine_class"] or r["coarse_class"]
        conf = (r["fine_confidence"] if r["fine_class"]
                else r["coarse_confidence"])
        if slug != reg["slug"]:
            reg = {**reg, "slug": slug}
        reg["avg_conf"] = round(float(conf), 3)
        verified.append(reg)
    # 재검증 후 같은 slug 로 수렴한 영역 병합은 하지 않음 — 시각적으로
    # 분리된 영역은 분리 표시가 자연스러움 (동일 slug 2개 = 같은 재질 2곳)
    return verified


def _crop_at_tap(raw: bytes, tap_x: float, tap_y: float,
                 expand: float = 0.12) -> tuple[bytes, list[float] | None]:
    """탭 지점의 saliency 연결 성분 bbox 로 크롭 (탭-투-셀렉트).

    성분 미검출 시 탭 중심 window-crop (shortestSide 50%) fallback —
    사용자가 지목했다는 사실 자체가 '그 근처에 객체가 있다'는 신호이므로
    전역 크롭보다 탭 중심이 낫다.
    반환: (crop bytes, region bbox_norm|None) — bbox 는 CAM 재질 융합용.
    """
    import io  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from src.segment import component_bbox_at  # noqa: PLC0415

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        bbox = component_bbox_at(raw, tap_x, tap_y)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            px, py = (x1 - x0) * expand, (y1 - y0) * expand
            box = (max(0, int((x0 - px) * w)), max(0, int((y0 - py) * h)),
                   min(w, int((x1 + px) * w)), min(h, int((y1 + py) * h)))
            region = list(bbox)
        else:
            # window fallback: 탭 중심 정사각 (shortestSide 50%)
            side = int(min(w, h) * 0.5)
            cx, cy = int(tap_x * w), int(tap_y * h)
            x0 = min(max(0, cx - side // 2), w - side)
            y0 = min(max(0, cy - side // 2), h - side)
            box = (x0, y0, x0 + side, y0 + side)
            region = [box[0] / w, box[1] / h, box[2] / w, box[3] / h]
        if box[2] - box[0] < 48 or box[3] - box[1] < 48:
            return raw, None
        buf = io.BytesIO()
        img.crop(box).save(buf, format="JPEG", quality=92)
        return buf.getvalue(), region
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] tap crop failed: {exc}")
        return raw, None


def _ensemble_with_dinov2(
    resnet_result: dict, raw: bytes, w_dino: float = 0.7,
) -> dict:
    """ResNet18 결과 + DINOv2 확률 weighted average.

    ResNet18 이 OOD 입력 (예: 손 안의 객체) 에 confident-wrong 인 케이스를 보정.
    DINOv2 가 더 robust 한 표현이라 더 큰 가중치 (0.7) 부여. DINOv2 가 없거나
    실패하면 원본 resnet_result 그대로 반환.
    """
    dino_cls = get_dinov2_classifier()
    if not dino_cls.available:
        return resnet_result
    dino_out = dino_cls.predict(raw)
    if dino_out is None:
        return resnet_result

    resnet_probs = resnet_result.get("all_probabilities") or {}
    dino_probs = dino_out["confidences"]

    # 두 모델 라벨 union — non_object 가 ClassRegistry 에 없을 수 있어
    # ClassRegistry 만 쓰면 누락. 학습 라벨(manifest) 이 정본.
    all_labels = sorted(set(resnet_probs.keys()) | set(dino_probs.keys()))
    fused = {}
    for lbl in all_labels:
        r = float(resnet_probs.get(lbl, 0.0))
        d = float(dino_probs.get(lbl, 0.0))
        fused[lbl] = (1.0 - w_dino) * r + w_dino * d

    s = sum(fused.values())
    if s > 0:
        fused = {l: p / s for l, p in fused.items()}

    top_label = max(fused, key=fused.get)
    # predicted_index: ResNet 의 인덱스 체계 유지 (없으면 기존값)
    reg_labels = list(ClassRegistry.all_slugs())
    top_idx = (
        reg_labels.index(top_label) if top_label in reg_labels
        else resnet_result.get("predicted_index", 0)
    )

    return {
        **resnet_result,
        "predicted_class": top_label,
        "predicted_index": top_idx,
        "confidence": float(fused[top_label]),
        "all_probabilities": fused,
        "model_arch": f"{resnet_result.get('model_arch', '')}+dinov2-w{w_dino:.1f}",
    }


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

    # ─ Stage 2.5: DINOv2 ensemble — confident-wrong 보정 ─
    try:
        result = _ensemble_with_dinov2(result, cropped_raw)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] dinov2 ensemble failed: {exc}")

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
      Stage 2 (regions): waste 면 전체 이미지에 대해 CAM/u2netp 마스크/손 제외
                        후 셀별 argmax 로 재질 영역 추출. /predict-with-cam 과
                        같은 원본 입력 사용 — 둘의 영역 표시가 일치하도록.
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

    # auto_crop 제거 — /predict-with-cam 과 같은 원본 입력으로 일관성 확보.
    # 다중재질 분석은 전체 이미지가 본래 목적에 부합하고, region overlay 좌표가
    # cropped 좌표계로 떠서 CAM 과 시각적으로 어긋나는 문제도 해결됨.
    raw = raw_orig

    classifier = get_classifier()
    try:
        color_input, edge_input = preprocess_both(raw)
    except ImageDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc),
        ) from exc

    result, cam = classifier.region_cam(color_input)

    # DINOv2 ensemble — confident-wrong 보정 (regions 분석은 ResNet18 CAM 그대로)
    try:
        result = _ensemble_with_dinov2(result, raw)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] dinov2 ensemble failed: {exc}")

    # ── 계층 고해상 CAM 우선 (CAM_MATERIAL_UPGRADE_PLAN Stage 1) ─────────
    # 448² forward → CAM (25,14,14): 셀 16px, 세부 25클래스 재질 어휘.
    # 실패/구 ONNX 시 flat 7×7 CAM fallback (하위호환).
    labels = list(classifier.labels)
    allowed_indices: list[int] | None = None
    hier_clf = None
    try:
        from src.hier_inference import get_hier_classifier  # noqa: PLC0415
        from src.preprocess import color_tensor_at  # noqa: PLC0415
        hier_clf = get_hier_classifier()
        cam_hi = hier_clf.cam_hires(color_tensor_at(raw, 448))
        if cam_hi is not None:
            cam = cam_hi
            labels = list(hier_clf.fine_labels)
            allowed_indices = hier_clf.material_class_indices()
    except FileNotFoundError:
        pass  # 계층 모델 미배치 — flat CAM 유지
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] hier hi-res cam failed: {exc}")

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
            regions = extract_regions(cam, mask_grid, labels,
                                      allowed_indices=allowed_indices)

            # ── 영역 재검증 (Stage 1-4, zoom-and-verify) ────────────────
            # CAM 은 제안자, 분류기가 심판: 각 영역을 크롭해 풀 분류로 확정.
            # reject 영역은 폐기, 불일치 시 재분류 slug 채택.
            if hier_clf is not None and regions:
                regions = _verify_regions(raw, regions, hier_clf)
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

    # [flat 폴백 전용 가드] regions dominant 가 flat top-1 과 다르면 overlay 제거.
    # 계층 경로(hier_clf)에서는 영역이 zoom-verify(크롭 재분류)를 이미 통과했고
    # slug 공간도 세부(25)라 flat top-1 과의 문자열 비교가 무의미 — 가드 제외.
    if hier_clf is None and regions_out \
            and regions_out[0].slug != result["predicted_class"]:
        regions_out = []
        overlay_b64 = None
        grid_h = grid_w = 0

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
