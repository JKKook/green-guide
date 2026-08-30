"""Supabase 접속 헬퍼 — 17곳에 흩어져 있던 `create_client` + 가드 블록의 단일화.

    from waste_common.supabase import get_client, Bucket
    cli = get_client()                       # 미설정 시 RuntimeError (메시지 통일)
    cli = try_get_client()                   # 미설정 시 None (best-effort 경로용)
    url = upload_and_get_url(Bucket.MODELS, "v1/model.onnx", path)
"""
from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from waste_common import settings
from waste_common.imaging import content_type

if TYPE_CHECKING:
    from supabase import Client

MISSING_CREDENTIALS_MESSAGE = "SUPABASE_URL / SUPABASE_KEY 가 설정되지 않음. .env 파일을 확인하세요."


class Bucket(StrEnum):
    RAW_IMAGES = settings.BUCKET_RAW_IMAGES
    USER_UPLOADS = settings.BUCKET_USER_UPLOADS
    MODELS = settings.BUCKET_MODELS


@lru_cache(maxsize=1)
def get_client() -> "Client":
    """프로세스당 하나의 클라이언트. 자격 증명 없으면 RuntimeError."""
    from supabase import create_client  # noqa: PLC0415 — 무거운 의존은 지연 import

    url, key = settings.supabase_credentials()
    if not url or not key:
        raise RuntimeError(MISSING_CREDENTIALS_MESSAGE)
    return create_client(url, key)


def try_get_client() -> "Client | None":
    """자격 증명이 없으면 None — 선택적 업로드/기록 경로용."""
    try:
        return get_client()
    except RuntimeError:
        return None


def reset_client() -> None:
    """테스트 격리용."""
    get_client.cache_clear()


def upload_and_get_url(
    bucket: Bucket | str,
    remote_path: str,
    src: Path | bytes,
    *,
    content_type_override: str | None = None,
    upsert: bool = True,
) -> str:
    """파일(경로 또는 bytes)을 업로드하고 public URL 을 반환."""
    storage = get_client().storage.from_(str(bucket))
    data = src.read_bytes() if isinstance(src, Path) else src
    ctype = content_type_override or (content_type(src) if isinstance(src, Path) else "application/octet-stream")
    storage.upload(
        path=remote_path,
        file=data,
        file_options={"upsert": "true" if upsert else "false", "content-type": ctype},
    )
    return storage.get_public_url(remote_path)


def download(bucket: Bucket | str, remote_path: str) -> bytes:
    return get_client().storage.from_(str(bucket)).download(remote_path)
