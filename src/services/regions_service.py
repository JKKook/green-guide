"""다중재질 영역 분석 — 탭 실루엣, 영역 재검증, 증거 충돌 판정."""
from __future__ import annotations

from src.core.log import get_logger

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
