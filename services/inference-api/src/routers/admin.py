"""관리 — 클래스 레지스트리·모델 리로드."""
from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, Header, HTTPException

from src.classes import ClassRegistry
from src.core import config
from src.inference import get_active_meta, get_classifier, reset_classifier
from src.schemas import ReloadModelResponse


def require_admin_token(x_admin_token: str | None = Header(default=None)) -> None:
    """X-Admin-Token 이 GREENGUIDE_ADMIN_TOKEN 과 일치해야 통과. 서버 토큰 미설정 시 전부 거부."""
    expected = config.ADMIN_TOKEN
    if not expected or not x_admin_token or not secrets.compare_digest(
        x_admin_token.encode(), expected.encode(),
    ):
        raise HTTPException(status_code=401, detail="admin token required")


router = APIRouter(dependencies=[Depends(require_admin_token)])


@router.post("/reload-classes", tags=["admin"])
def reload_classes() -> dict[str, int]:
    """레지스트리 강제 리로드 (관리자용)."""
    ClassRegistry.reload()
    return {
        "total": len(ClassRegistry.all_slugs()),
        "trained": len(ClassRegistry.trained_slugs()),
    }


@router.post("/admin/reload-model", response_model=ReloadModelResponse, tags=["admin"])
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
