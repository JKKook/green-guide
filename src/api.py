"""FastAPI app 정의."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from src.core import config
from src.core.errors import register_exception_handlers
from src.cam_renderer import render_overlay_png_base64
from src.classes import ClassRegistry
from src.inference import get_active_meta, get_classifier, reset_classifier
from src.preprocess import ImageDecodeError, preprocess_both
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
from src.services.cascade import (
    ensemble_with_dinov2, force_non_object_result, non_object_gate, run_cascade, stage1_gate,
)
from src.services.image_io import crop_at_tap, read_and_validate_image, read_validate_with_orientation
from src.services.recording import record_safely
from src.services.regions_service import evidence_conflicts, tap_silhouette_regions, verify_regions
from src.dinov2_classifier import get_dinov2_classifier
from src.uploads import get_recorder
from src.core.log import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier = get_classifier()
    meta = get_active_meta()
    # Supabase 불가 시에도 부팅은 계속 — 레지스트리는 요청 시 재시도됨.
    # (설계 원칙: "API 는 항상 부팅한다" — model_loader 와 동일한 강건성)
    try:
        ClassRegistry.load()
    except Exception as exc:  # noqa: BLE001
        log.warning(f"class registry 로드 실패 (Supabase 미접속?): {exc}")
    log.info(f"color model: {classifier.model_path}")
    log.info(f"edge model: {classifier.edge_model_path or '(disabled)'}")
    log.info(f"inference mode: "
          f"{'ensemble (color+edge)' if classifier.has_edge_stream else 'single (color)'}")
    if meta is not None:
        log.info(f"remote model version: v{meta.version} "
              f"(accuracy={meta.test_accuracy}, feedback={meta.feedback_count})")
    else:
        log.info("remote model version: (fallback — Supabase 에 active row 없음)")
    try:
        log.info(f"class registry: "
              f"{len(ClassRegistry.all_slugs())} total "
              f"({len(ClassRegistry.trained_slugs())} trained)")
    except Exception:  # noqa: BLE001
        log.info("class registry: (미로드 — 요청 시 재시도)")
    log.info(f"user upload collection: "
          f"{'ENABLED' if config.COLLECT_USER_UPLOADS else 'disabled'}")
    # DINOv2 미리 로드 (첫 요청 지연 회피)
    dino = get_dinov2_classifier()
    log.info(f"dinov2 classifier: "
          f"{'ENABLED' if dino.available else 'disabled (model 없음)'}")

    # 수집 정리 — 피드백 없는 업로드 7일 후 삭제 (무료 쿼터 지속성).
    # 기동 직후 1회 + 24시간 주기. 실패해도 부팅·서빙 무영향.
    async def _prune_loop() -> None:
        import asyncio  # noqa: PLC0415
        while True:
            try:
                from src.uploads import prune_stale_uploads  # noqa: PLC0415
                await asyncio.to_thread(prune_stale_uploads, 7)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"정리 실패(다음 주기 재시도): {str(exc)[:80]}")
            await asyncio.sleep(24 * 3600)

    prune_task = None
    if config.COLLECT_USER_UPLOADS:
        import asyncio  # noqa: PLC0415
        prune_task = asyncio.create_task(_prune_loop())
    yield
    if prune_task is not None:
        prune_task.cancel()
    reset_classifier()
    get_recorder.reset()


# 트랙 B1 — 1차 확신이 이 값 이상이면 장면 경로 OCR 스킵 (운영 지연 -2~4s)
OCR_SKIP_CONFIDENCE = config.OCR_SKIP_CONFIDENCE

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
register_exception_handlers(app)


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
    host = None
    try:
        from urllib.parse import urlparse  # noqa: PLC0415
        u = config.SUPABASE_URL or ""
        host = urlparse(u).hostname if u else None
    except Exception:  # noqa: BLE001
        host = None
    return HealthResponse(supabase_host=host)


@app.get("/design/tokens.json", tags=["meta"])
def design_tokens() -> dict:
    """디자인 토큰 (W3C Design Tokens draft) — 앱 실측값.

    출처: waste_app app_theme.dart · confidence.dart · waste_info.dart.
    디자인 도구(Figma Tokens/style-dictionary)·시안 문서가 URL 로 소비.
    """
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415
    p = Path(__file__).resolve().parent.parent / "design" / "tokens.json"
    return json.loads(p.read_text(encoding="utf-8"))


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
        degs_for_orientation, get_hier_classifier, predict_rotations,
    )

    raw, exif_tag = await read_validate_with_orientation(image)
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
    forced_reason = non_object_gate(raw)

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
        cropped_raw, tap_region = crop_at_tap(raw, tap_x, tap_y)
    else:
        cropped_raw = raw

    # ── 1차 패스: EXIF 태그 기반 축소 TTA (트랙 B2 — 3×→평균 1.7×) ──────────
    # 게이트를 통과했다 = stage1 이 '폐기물'로 판정 (또는 fail-open)
    # → 분류기의 non_object 는 모순된 답이므로 마스킹 (실측 +5.9pp)
    result, best_tensor = predict_rotations(
        clf, cropped_raw, degs_for_orientation(exif_tag),
        mask_non_object=True, ood_relax=tap_x is not None)

    # ── 시맨틱 증거 융합 (SEMANTIC_FUSION_PLAN §3 + 청사진 v2 트랙 B1) ──────
    #   OCR: 탭이거나 1차 확신이 낮을 때만 (고확신 장면은 스킵 — 운영 -2~4s)
    #   CLIP·CAM: 탭(고립 crop)에서만 (장면 적용은 51장 실측 역효과)
    # prior 가 생기면 베스트 회전 텐서 1장만 재예측 — TTA 전체 재실행 없음.
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

    need_ocr = (tap_region is not None) or (
        result["fine_confidence"] < OCR_SKIP_CONFIDENCE)
    if need_ocr:
        try:
            texts = get_evidence_engine().read_texts(cropped_raw)
            evidence = match_evidence(texts)
            prior = _mul(prior, evidence_prior(
                evidence, clf.fine_labels, clf.taxonomy["fine_to_coarse"]))
        except Exception as exc:  # noqa: BLE001
            log.warning(f"semantic evidence failed: {exc}")
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
            log.warning(f"clip identity failed: {exc}")
        try:
            from src.hier_inference import cam_region_prior  # noqa: PLC0415
            prior = _mul(prior, cam_region_prior(clf, raw, tap_region))
        except Exception as exc:  # noqa: BLE001
            log.warning(f"cam region prior failed: {exc}")

    if prior is not None:
        refined = clf.predict(best_tensor, mask_non_object=True, fine_prior=prior,
                              ood_relax=tap_x is not None)
        refined["tta_rotation"] = result.get("tta_rotation", 0)
        refined["inference_ms"] = round(
            result["inference_ms"] + refined["inference_ms"], 2)
        result = refined

    # ── VLM 폴백 (트랙 A2) — 융합 후에도 저확신이면 Claude 에 최종 판정 위임 ──
    # 키 미설정/한도초과/실패 시 자동 무시 (fail-open). 결과는 evidence 로 표면화.
    # 증거-불일치 중재: 강한 CLIP 정체 증거(≥0.6)가 CNN 과 다른 대분류를
    # 가리키면 확신도와 무관하게 중재 — 과확신 오답(confident-wrong)이 증거
    # 칩과 모순된 채 그대로 노출되던 이격(실사용: 음식물 사진→의류 85.8%) 처방.
    evidence_conflict = evidence_conflicts(
        evidence, result["coarse_class"], clf.taxonomy["fine_to_coarse"])
    if evidence_conflict:
        log.info(f"증거-불일치 중재 발동: CNN={result['coarse_class']}")
    if (result["display_level"] == "reject"
            or result["coarse_confidence"] < 0.55 or evidence_conflict):
        try:
            from src.vlm_fallback import get_vlm_fallback  # noqa: PLC0415
            v = get_vlm_fallback().classify(
                cropped_raw, clf.fine_labels, clf.taxonomy["fine_to_coarse"])
            # 과신 가드 3단 — 재질 교체 0.8: etc 잡동사니에 재질을 부여하는
            # 오버라이드가 홀드아웃 실측서 4건 중 2건 오답 / non_object 0.5:
            # 재촬영 신호라 보수적 방향 / 품목 생성 0.6: 재질 필드 미변경
            # + 스트림은 닫힌 목록이라 중위험.
            if v is not None and v["slug"] is None:
                min_conf = config.VLM_ITEM_MIN_CONF
            elif v is not None and v["slug"] == "non_object":
                min_conf = 0.5
            else:
                min_conf = config.VLM_MIN_CONF
            if v is not None and v["confidence"] >= min_conf:
                slug = v["slug"]
                if slug is None:
                    # 사전 밖 품목 생성 판정 — 재질 필드는 건드리지 않고
                    # (기존 클라이언트 하위호환) 스트림 안내를 별도 표면화.
                    from src.streams import to_api_dict  # noqa: PLC0415
                    stream_info = to_api_dict(v["stream"])
                    if stream_info is not None:
                        result["generated_item"] = {
                            "item_name": v["item_name"],
                            "stream": stream_info,
                            "condition": v["condition"],
                            "confidence": v["confidence"],
                        }
                        result["model_arch"] = result["model_arch"] + "+vlm"
                        evidence.append({
                            "type": "vlm",
                            "token": f'{v["item_name"]} → {stream_info["display_name"]}',
                            "matched_text": v["reason"],
                            "mapped_class": v["stream"],
                            "score": v["confidence"],
                        })
                else:
                    coarse = clf.taxonomy["fine_to_coarse"].get(slug, slug)
                    if slug == "non_object":
                        result["display_level"] = "reject"
                        result["display_class"] = "non_object"
                    else:
                        result["display_level"] = "fine" if slug != "etc" else "coarse"
                        result["display_class"] = slug if slug != "etc" else "etc"
                        result["fine_class"] = slug if slug != "etc" else None
                        result["coarse_class"] = coarse
                    result["model_arch"] = result["model_arch"] + "+vlm"
                    evidence.append({
                        "type": "vlm",
                        "token": v["reason"] or "AI 정밀 분석",
                        "matched_text": v["reason"],
                        "mapped_class": slug,
                        "score": v["confidence"],
                    })
        except Exception as exc:  # noqa: BLE001
            log.warning(f"vlm fallback failed: {exc}")
    if evidence:
        result["evidence"] = [
            {k: ev[k] for k in ("type", "token", "matched_text",
                                "mapped_class", "score")}
            for ev in evidence
        ]

    # user_uploads 스키마와 호환되는 형태로 기록 (게이트 적용 결과 기준)
    upload_id = record_safely(raw, image, {
        "predicted_class": result["display_class"],
        "confidence": (
            result["fine_confidence"]
            if result["display_level"] == "fine"
            else result["coarse_confidence"]
        ),
        "all_probabilities": result["coarse_probabilities"],
        "model_arch": result["model_arch"],
        "inference_ms": result["inference_ms"],
    })

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
        degs_for_orientation, get_hier_classifier, predict_best_rotation,
    )
    from src.segment import all_component_bboxes  # noqa: PLC0415

    raw, exif_tag = await read_validate_with_orientation(image)
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
        log.warning(f"component split failed: {exc}")
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
        log.warning(f"semantic evidence failed: {exc}")

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
            log.warning(f"clip identity failed: {exc}")
        try:
            r = predict_best_rotation(
                clf, buf.getvalue(), mask_non_object=True, fine_prior=prior,
                degs=degs_for_orientation(exif_tag))
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


@app.get("/region-info", tags=["meta"])
def region_info(sido: str, sigungu: str) -> dict:
    """지역별 생활쓰레기 배출 규정 — 앱 지역 선택 시나리오의 데이터 소스.

    Supabase region_waste_rules (공공데이터포털 전국생활쓰레기배출정보 표준데이터,
    scripts/load_region_rules.py 적재) 조회. 데이터 미적재/오프라인이어도
    빈 목록으로 응답 — 앱은 전국 공통 안내로 fallback.
    """
    try:
        from src.uploads import _client as _supabase_client  # noqa: PLC0415
        client = _supabase_client()
        res = (client.table("region_waste_rules")
               .select("*")
               .eq("sido", sido)
               .eq("sigungu", sigungu)
               .limit(50)
               .execute())
        rules = res.data or []
    except Exception as exc:  # noqa: BLE001
        log.warning(f"region-info 조회 실패 (빈 응답): {exc}")
        rules = []
    return {"sido": sido, "sigungu": sigungu, "count": len(rules), "rules": rules}


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


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    image: UploadFile = File(..., description="분류할 폐기물 이미지"),
) -> PredictionResponse:
    raw = await read_and_validate_image(image)

    classifier = get_classifier()
    color_input, edge_input = preprocess_both(raw)

    result = classifier.predict(color_input, edge_input)

    upload_id = record_safely(raw, image, result)

    return PredictionResponse(**result, upload_id=upload_id)


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
    raw = await read_and_validate_image(image)
    result = run_cascade(raw)
    # upload 기록 (원본 이미지 — 사용자 피드백·재학습은 원본 기준)
    upload_id = record_safely(raw, image, result)
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
    raw = await read_and_validate_image(image)

    classifier = get_classifier()
    color_input, edge_input = preprocess_both(raw)

    result = classifier.predict(color_input, edge_input, want_cam=True)
    cam_array = result.pop("cam", None)

    cam_b64: str | None = None
    if cam_array is not None:
        try:
            cam_b64 = render_overlay_png_base64(raw, cam_array)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"CAM rendering failed: {exc}")

    # /predict 와 동일하게 upload 기록 (active learning 데이터로 동등하게 누적)
    upload_id = record_safely(raw, image, result)

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
    raw = await read_and_validate_image(image)

    classifier = get_classifier()
    color_input, edge_input = preprocess_both(raw)

    result = classifier.predict(color_input, edge_input)

    # 누끼 (saliency segmentation → cutout)
    seg = {"cutout_base64": None, "bbox_norm": None, "object_ratio": 0.0}
    try:
        seg = get_segmenter().segment(raw)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"segmentation failed: {exc}")

    upload_id = record_safely(raw, image, result)

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
    tap_x: float | None = Form(default=None, ge=0.0, le=1.0),
    tap_y: float | None = Form(default=None, ge=0.0, le=1.0),
) -> PredictionWithRegionsResponse:
    """`/predict` + 다중재질 영역 분석 (Cascade + CAM-argmax + u2netp + 손 제외).

    파이프라인:
      Stage 1 (binary): waste 아니면 → non_object 응답 (regions 분석 skip)
      Stage 2 (regions): waste 면 전체 이미지에 대해 CAM/u2netp 마스크/손 제외
                        후 셀별 argmax 로 재질 영역 추출. /predict-with-cam 과
                        같은 원본 입력 사용 — 둘의 영역 표시가 일치하도록.
    """
    raw_orig = await read_and_validate_image(image)

    # Stage 1: binary waste/non-waste 판정
    reason = stage1_gate(raw_orig)
    if reason is not None:
        result = force_non_object_result(reason)
        # 업로드 기록 없음 — 앱은 같은 사진으로 /predict-hier 를 함께 호출하고
        # 그쪽 upload_id 로 피드백한다. 여기서도 저장하면 분석 1회당 사진이
        # 2장씩 쌓였음(2026-08-29 실기기 검증에서 확인).
        return PredictionWithRegionsResponse(
            **result, upload_id=None,
            overlay_base64=None, regions=[], grid_h=0, grid_w=0,
        )

    # auto_crop 제거 — /predict-with-cam 과 같은 원본 입력으로 일관성 확보.
    # 다중재질 분석은 전체 이미지가 본래 목적에 부합하고, region overlay 좌표가
    # cropped 좌표계로 떠서 CAM 과 시각적으로 어긋나는 문제도 해결됨.
    raw = raw_orig

    classifier = get_classifier()
    color_input, edge_input = preprocess_both(raw)

    result, cam = classifier.region_cam(color_input)

    # DINOv2 ensemble — confident-wrong 보정 (regions 분석은 ResNet18 CAM 그대로)
    try:
        result = ensemble_with_dinov2(result, raw)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"dinov2 ensemble failed: {exc}")

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
        log.warning(f"hier hi-res cam failed: {exc}")

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
                log.warning(f"hand mask grid failed: {exc}")

            # 탭-투-셀렉트 재분석 — 탭한 성분 bbox 밖 셀을 마스킹해 빗금·영역
            # 추출을 그 물건에 집중 (좌표계는 원본 유지 → 오버레이 정합).
            # "마커는 이동하는데 빗금은 안 움직인다" 사용자 리포트의 처방.
            tap_grabcut_ok = False
            if tap_x is not None and tap_y is not None:
                try:
                    # 1순위: GrabCut 전경 실루엣 — 탭한 물건의 픽셀 경계 점유.
                    # saliency(시선 지도)는 책상 경계·이웃 물체까지 밝아 빗금이
                    # 탭 지점과 어긋나던 문제의 처방.
                    from src.segment import grabcut_object_at  # noqa: PLC0415
                    gmask, gbox = grabcut_object_at(raw, tap_x, tap_y, grid_h)
                    if gmask is not None and (gmask >= 0.35).sum() >= 1:
                        mask_grid = gmask
                        tap_grabcut_ok = True
                        log.info(f"grabcut bbox={[round(v,2) for v in gbox]} "
                              f"cells={(gmask >= 0.35).sum()}")
                except Exception as exc:  # noqa: BLE001
                    log.warning(f"tap grabcut failed: {exc}")
            if tap_x is not None and tap_y is not None and not tap_grabcut_ok:
                try:
                    from src.segment import component_bbox_at  # noqa: PLC0415
                    tb = component_bbox_at(raw, tap_x, tap_y)
                    if tb is None:
                        s = 0.25  # 성분 미검출 — 탭 중심 50% 윈도우
                        tb = [max(0.0, tap_x - s), max(0.0, tap_y - s),
                              min(1.0, tap_x + s), min(1.0, tap_y + s)]
                    else:
                        # 성분이 파편(하이라이트 등)이면 최소 창 보장 — 저대비
                        # 물체는 saliency 성분이 조각나 창이 셀 몇 개로 줄어듦
                        _mh = 0.12
                        _cx, _cy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
                        if tb[2] - tb[0] < 2 * _mh:
                            tb[0], tb[2] = max(0.0, _cx - _mh), min(1.0, _cx + _mh)
                        if tb[3] - tb[1] < 2 * _mh:
                            tb[1], tb[3] = max(0.0, _cy - _mh), min(1.0, _cy + _mh)
                    import numpy as _np  # noqa: PLC0415
                    focus = _np.zeros_like(mask_grid)
                    r0 = max(0, int(tb[1] * grid_h)); r1 = min(grid_h, int(tb[3] * grid_h) + 1)
                    c0 = max(0, int(tb[0] * grid_w)); c1 = min(grid_w, int(tb[2] * grid_w) + 1)
                    focus[r0:r1, c0:c1] = 1.0
                    # 탭 = 객체 존재 신호: 창 안 약한 saliency(≥0.12) 셀은 점유
                    # 하한(0.35)을 보장 — 저대비 물체가 점유 필터에 전멸해 빗금이
                    # 안 나오는 문제 방지. saliency 가 거의 없는 셀은 그대로 제외.
                    mask_grid = _np.maximum(
                        mask_grid, 0.35 * (mask_grid >= 0.12)) * focus
                    log.info(f"bbox={[round(v,2) for v in tb]} grid=({r0}:{r1},{c0}:{c1})")
                except Exception as exc:  # noqa: BLE001
                    log.warning(f"tap focus mask failed: {exc}")
            if tap_x is not None and tap_y is not None:
                # 탭 경로: saliency 실루엣 기반 — 빗금이 탭한 물건 형태를 따라감
                # GrabCut 실루엣은 이미 탭 물건 성분만이라 반경 제한 불필요;
                # saliency fallback 은 번짐 방지 위해 반경 3 유지
                regions = tap_silhouette_regions(
                    cam, mask_grid, labels, allowed_indices,
                    tap_x, tap_y, grid_h, grid_w,
                    radius=max(grid_h, grid_w) if tap_grabcut_ok else 3)
                if not regions:  # 실루엣 실패 — 기존 CAM-argmax 방식 fallback
                    regions = extract_regions(cam, mask_grid, labels,
                                              allowed_indices=allowed_indices)
                log.info(f"tap=({tap_x:.2f},{tap_y:.2f}) "
                      f"extract={[(r['slug'], len(r['cells'])) for r in regions]}")
            else:
                regions = extract_regions(cam, mask_grid, labels,
                                          allowed_indices=allowed_indices)

            # ── 영역 재검증 (Stage 1-4, zoom-and-verify) ────────────────
            # CAM 은 제안자, 분류기가 심판: 각 영역을 크롭해 풀 분류로 확정.
            # reject 영역은 폐기, 불일치 시 재분류 slug 채택.
            if hier_clf is not None and regions:
                pre_verify = regions
                regions = verify_regions(raw, regions, hier_clf,
                                          ood_relax=tap_x is not None)
                if tap_x is not None:
                    log.info(f"verify={[(r['slug'], len(r['cells'])) for r in regions]}")
                    # 탭 맥락 = 사용자가 지목한 물건 — 빗금(선택 피드백)이 우선.
                    # 검증이 전멸시켜도 최상위 CAM 영역은 유지해 항상 표시.
                    if not regions and pre_verify:
                        regions = pre_verify[:1]
                        log.info("verify 전멸 → 탭 최상위 영역 유지")
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
            log.warning(f"region analysis failed: {exc}")

    # [flat 폴백 전용 가드] regions dominant 가 flat top-1 과 다르면 overlay 제거.
    # 계층 경로(hier_clf)에서는 영역이 zoom-verify(크롭 재분류)를 이미 통과했고
    # slug 공간도 세부(25)라 flat top-1 과의 문자열 비교가 무의미 — 가드 제외.
    if hier_clf is None and regions_out \
            and regions_out[0].slug != result["predicted_class"]:
        regions_out = []
        overlay_b64 = None
        grid_h = grid_w = 0

    # 업로드 기록 없음 — /predict-hier 가 같은 사진을 이미 저장·피드백 대상으로
    # 삼는다(중복 저장 방지, 2026-08-29).
    upload_id: str | None = None

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
    raw = await read_and_validate_image(image)
    try:
        return get_segmenter().segment(raw)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"segmentation failed: {exc}")
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
