"""사용자 업로드 기록 (active learning 데이터 수집) — 엔드포인트 공통."""
from __future__ import annotations

from typing import Any

from fastapi import UploadFile

from src.core import config
from src.core.log import get_logger
from src.uploads import get_recorder

log = get_logger(__name__)


def record_safely(raw: bytes, image: UploadFile, prediction: dict[str, Any]) -> str | None:
    """예측 결과와 원본 이미지를 기록하고 upload_id 를 돌려준다.

    fail-open: 수집이 꺼져 있거나(COLLECT_USER_UPLOADS=false) Supabase 기록이
    실패해도 추론 응답은 정상 반환해야 하므로 None 을 돌려주고 경고만 남긴다.
    """
    if not config.COLLECT_USER_UPLOADS:
        return None
    try:
        return get_recorder().record_prediction(
            image_bytes=raw,
            content_type=image.content_type or "application/octet-stream",
            prediction=prediction,
        )
    except Exception as exc:  # noqa: BLE001 — fail-open (위 docstring)
        log.warning(f"upload collection failed: {exc}")
        return None
