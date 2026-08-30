"""다중재질 영역 분석 — 탭 실루엣, 영역 재검증, 증거 충돌 판정."""
from __future__ import annotations

from src.classes import ClassRegistry
from src.core.log import get_logger
from src.hand_detector import get_hand_detector
from src.inference import get_classifier
from src.preprocess import preprocess_both
from src.regions import extract_regions, render_hatching
from src.schemas import MaterialRegion
from src.segment import get_segmenter
from src.services.cascade import ensemble_with_dinov2

log = get_logger(__name__)


def tap_silhouette_regions(
    cam_all, mask_grid, labels: list[str],
    allowed_indices: list[int] | None,
    tap_x: float, tap_y: float, grid_h: int, grid_w: int,
    radius: int = 3,
) -> list[dict]:
    """탭 물건의 saliency 실루엣을 빗금 영역으로 (탭 경로 전용).

    CAM argmax 셀은 '판별에 쓴 부위'만 밝혀 물건 형태와 어긋나고, 클래스별
    묶음이라 이웃 물건의 같은 클래스 셀까지 섞임 → 빗금이 탭 지점과 달라 보임
    (사용자 리포트). 대신: 탭 셀에서 saliency(점유≥0.35) 연결 성분을 그리드
    flood-fill 로 잡고 탭 반경 radius 셀로 제한 — 빗금이 탭한 물건 실루엣을
    따라감. 라벨은 그 셀들의 CAM argmax 를 클래스별로 묶어 부여 (≥2셀 클래스만
    분리, 아니면 다수결 단일 영역 = 다중재질 표시 유지).
    """
    import numpy as _np  # noqa: PLC0415
    from src.regions import _softmax0  # noqa: PLC0415

    tr = min(grid_h - 1, max(0, int(tap_y * grid_h)))
    tc = min(grid_w - 1, max(0, int(tap_x * grid_w)))
    sal = mask_grid >= 0.35

    # 시드: 탭 셀이 saliency 밖이면 반경 2 내 최근접 saliency 셀
    seed = None
    if sal[tr, tc]:
        seed = (tr, tc)
    else:
        best_d = None
        for r in range(max(0, tr - 2), min(grid_h, tr + 3)):
            for c in range(max(0, tc - 2), min(grid_w, tc + 3)):
                if sal[r, c]:
                    d = max(abs(r - tr), abs(c - tc))
                    if best_d is None or d < best_d:
                        best_d, seed = d, (r, c)
    if seed is None:
        return []

    # flood fill (4-이웃) + 탭 반경 제한
    comp: list[tuple[int, int]] = []
    seen = {seed}
    stack = [seed]
    while stack:
        r, c = stack.pop()
        comp.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < grid_h and 0 <= nc < grid_w
                    and (nr, nc) not in seen and sal[nr, nc]
                    and max(abs(nr - tr), abs(nc - tc)) <= radius):
                seen.add((nr, nc))
                stack.append((nr, nc))
    if not comp:
        return []

    # 셀 라벨: CAM argmax (재질 후보 제한)
    if allowed_indices is not None:
        masked = _np.full_like(cam_all, -1e9)
        masked[allowed_indices] = cam_all[allowed_indices]
        cam_all = masked
    probs = _softmax0(cam_all)
    cls = probs.argmax(axis=0)
    conf = probs.max(axis=0)

    by_class: dict[int, list[tuple[int, int]]] = {}
    for (r, c) in comp:
        by_class.setdefault(int(cls[r, c]), []).append((r, c))

    def _mk(ci: int, cells: list[tuple[int, int]]) -> dict:
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return {
            "class_index": ci,
            "slug": labels[ci] if ci < len(labels) else "etc",
            "cells": [[r, c] for r, c in cells],
            "bbox_norm": [min(cs) / grid_w, min(rs) / grid_h,
                          (max(cs) + 1) / grid_w, (max(rs) + 1) / grid_h],
            "avg_conf": round(float(_np.mean([conf[r, c] for r, c in cells])), 3),
        }

    # 탭 경로는 실루엣 전체 = 단일 영역 (다수결 라벨) — CAM argmax 노이즈가
    # 단일 물체를 유사-재질 조각으로 쪼개고 verify 가 조각을 떨궈 빗금이
    # 누더기·부분 커버가 되는 문제 방지. (다중재질 분리 표시는 첫 분류의
    # extract_regions 경로에 유지 — 탭의 목적은 '이 물건 선택' 피드백)
    maj = max(by_class, key=lambda ci: len(by_class[ci]))
    return [_mk(maj, comp)]


def verify_regions(raw: bytes, regions: list[dict], hier_clf,
                    ood_relax: bool = False) -> list[dict]:
    """CAM 제안 영역을 크롭 재분류로 확정 (zoom-and-verify, Stage 1-4).

    - reject(불확신) 영역 → 폐기 (스퓨리어스 차단)
    - CAM slug 와 재분류 slug 불일치 → 재분류 결과 채택 (분류기가 심판)
    - avg_conf 는 재분류 확신으로 교체 (검증된 수치)
    ood_relax: 탭-투-셀렉트 경로 True — 크롭은 OOD 거리가 튀어 하드 reject 로
    영역이 전부 폐기되는 문제(빗금 미표시) 방지. 탭 없는 경로는 기존 가드 유지.
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
            r = hier_clf.predict(ci, mask_non_object=True, ood_relax=ood_relax)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"region verify failed: {exc}")
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


def evidence_conflicts(
    evidence: list[dict], coarse_class: str, fine_to_coarse: dict[str, str],
    min_score: float = 0.6,
) -> bool:
    """강한 CLIP 정체 증거가 CNN 과 다른 대분류를 가리키는가.

    과확신 오답(confident-wrong)이 증거 칩과 모순된 채 노출되던 이격의 검출자
    — True 면 확신도와 무관하게 VLM 중재를 발동시킨다 (실사용 사례:
    음식물 사진 → CNN 의류 85.8% 인데 정체 증거는 음식물).
    identity(확률 0~1 스케일)만 대상 — OCR 계열 score 는 부스트 배수라 제외.
    """
    for ev in evidence:
        if ev.get("type") != "identity":
            continue
        if float(ev.get("score", 0)) < min_score:
            continue
        mapped = ev.get("mapped_class")
        ev_coarse = fine_to_coarse.get(mapped, mapped)
        if ev_coarse and ev_coarse != coarse_class:
            return True
    return False


def analyze_regions(raw: bytes, tap_x: float | None, tap_y: float | None) -> dict:
    """/predict-with-regions Stage 2: 분류 + CAM/u2netp 마스크/손 제외 → 재질 영역.

    반환: {"result", "overlay_base64", "regions", "grid_h", "grid_w"}
    """
    # auto_crop 제거 — /predict-with-cam 과 같은 원본 입력으로 일관성 확보.
    # 다중재질 분석은 전체 이미지가 본래 목적에 부합하고, region overlay 좌표가
    # cropped 좌표계로 떠서 CAM 과 시각적으로 어긋나는 문제도 해결됨.
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


    return {
        "result": result, "overlay_base64": overlay_b64,
        "regions": regions_out, "grid_h": grid_h, "grid_w": grid_w,
    }
