"""추론 엔드포인트 — /predict* , /segment."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from src.cam_renderer import render_overlay_png_base64
from src.core import config
from src.core.log import get_logger
from src.inference import get_classifier
from src.preprocess import ImageDecodeError, preprocess_both
from src.schemas import (
    ObjectCandidate,
    PredictObjectsResponse,
    PredictionHierResponse,
    PredictionResponse,
    PredictionWithCamResponse,
    PredictionWithMaskResponse,
    PredictionWithRegionsResponse,
)
from src.segment import get_segmenter
from src.services.cascade import force_non_object_result, non_object_gate, run_cascade, stage1_gate
from src.services.image_io import (
    crop_at_tap,
    read_and_validate_image,
    read_validate_with_orientation,
)
from src.services.recording import record_safely
from src.services.regions_service import analyze_regions, evidence_conflicts

log = get_logger(__name__)
router = APIRouter()


@router.post("/predict-hier", response_model=PredictionHierResponse, tags=["inference"])
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
        result["fine_confidence"] < config.OCR_SKIP_CONFIDENCE)
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


@router.post("/predict-objects", response_model=PredictObjectsResponse, tags=["inference"])
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


@router.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    image: UploadFile = File(..., description="분류할 폐기물 이미지"),
) -> PredictionResponse:
    raw = await read_and_validate_image(image)

    classifier = get_classifier()
    color_input, edge_input = preprocess_both(raw)

    result = classifier.predict(color_input, edge_input)

    upload_id = record_safely(raw, image, result)

    return PredictionResponse(**result, upload_id=upload_id)


@router.post("/predict-centered", response_model=PredictionResponse, tags=["inference"])
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


@router.post(
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


@router.post(
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


@router.post(
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

    out = analyze_regions(raw_orig, tap_x, tap_y)

    # 업로드 기록 없음 — /predict-hier 가 같은 사진을 이미 저장·피드백 대상으로
    # 삼는다(중복 저장 방지, 2026-08-29).
    upload_id: str | None = None

    return PredictionWithRegionsResponse(
        **out["result"],
        upload_id=upload_id,
        overlay_base64=out["overlay_base64"],
        regions=out["regions"],
        grid_h=out["grid_h"],
        grid_w=out["grid_w"],
    )


@router.post("/segment", tags=["inference"])
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
