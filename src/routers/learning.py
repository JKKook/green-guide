"""학습 루프 — 사용자 피드백 수집."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from src.classes import ClassRegistry
from src.core import config
from src.core.log import get_logger
from src.schemas import FeedbackRequest, FeedbackResponse
from src.uploads import get_recorder

log = get_logger(__name__)
router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse, tags=["learning"])
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
