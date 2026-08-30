"""업로드 이미지 읽기·검증·크롭 — 추론 엔드포인트 공통 입력 처리."""
from __future__ import annotations

from fastapi import HTTPException, UploadFile, status

from src.core import config
from src.core.log import get_logger
from src.preprocess import normalize_orientation
from src.segment import get_segmenter

log = get_logger(__name__)


async def read_validate_with_orientation(image: UploadFile) -> tuple[bytes, int]:
    """업로드 검증 + (EXIF 정규화 bytes, 원본 Orientation 태그) 반환.

    태그는 회전 TTA 축소(청사진 v2 트랙 B2)에 사용 — 학습 데이터가 센서
    방향이므로 "어느 회전이 유효 후보인지"를 태그가 알려준다.
    """
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
    orientation = 1
    try:
        import io as _io  # noqa: PLC0415
        from PIL import Image as _Image  # noqa: PLC0415
        orientation = int(_Image.open(_io.BytesIO(raw)).getexif().get(274, 1))
    except Exception:  # noqa: BLE001
        pass
    # EXIF 회전 태그를 픽셀에 적용 — Flutter 표시(태그 적용)와 서버 처리
    # (분류·CAM·빗금·누끼) 의 방향을 일치시킴.
    return normalize_orientation(raw), orientation


async def read_and_validate_image(image: UploadFile) -> bytes:
    """공통 헬퍼 — 업로드 검증 + EXIF 정규화 bytes 반환."""
    raw, _ = await read_validate_with_orientation(image)
    return raw


def auto_crop_to_object(raw: bytes, expand: float = 0.10) -> bytes:
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
        log.warning(f"segment for auto-crop failed: {exc}")
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
        log.warning(f"bbox crop failed: {exc}")
        return raw


def crop_at_tap(raw: bytes, tap_x: float, tap_y: float,
                 expand: float = 0.12) -> tuple[bytes, list[float] | None]:
    """탭 지점의 saliency 연결 성분 bbox 로 크롭 (탭-투-셀렉트).

    성분 미검출 시 탭 중심 window-crop (shortestSide 50%) fallback —
    사용자가 지목했다는 사실 자체가 '그 근처에 객체가 있다'는 신호이므로
    전역 크롭보다 탭 중심이 낫다.
    반환: (crop bytes, region bbox_norm|None) — bbox 는 CAM 재질 융합용.
    """
    import io  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from src.segment import component_bbox_at, grabcut_object_at  # noqa: PLC0415

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        # 1순위 GrabCut(픽셀 경계 실루엣) — 맞닿은 물체도 탭 물건만 크롭.
        # 실패 시 saliency 성분 fallback.
        bbox = None
        try:
            _, bbox = grabcut_object_at(raw, tap_x, tap_y, 14)
        except Exception:  # noqa: BLE001
            bbox = None
        if bbox is None:
            bbox = component_bbox_at(raw, tap_x, tap_y)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            # 파편 성분(하이라이트 조각 등) 보정 — 크롭 최소 변 35% 보장.
            # 저대비 물체는 성분이 조각나 sliver 크롭이 되면 분류가 망가짐.
            min_side = 0.35 * min(w, h)
            cx, cy = (x0 + x1) / 2 * w, (y0 + y1) / 2 * h
            bw, bh = max((x1 - x0) * w, min_side), max((y1 - y0) * h, min_side)
            x0, y0 = (cx - bw / 2) / w, (cy - bh / 2) / h
            x1, y1 = (cx + bw / 2) / w, (cy + bh / 2) / h
            px, py = (x1 - x0) * expand, (y1 - y0) * expand
            box = (max(0, int((x0 - px) * w)), max(0, int((y0 - py) * h)),
                   min(w, int((x1 + px) * w)), min(h, int((y1 + py) * h)))
            region = [max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)]
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
        log.warning(f"tap crop failed: {exc}")
        return raw, None
