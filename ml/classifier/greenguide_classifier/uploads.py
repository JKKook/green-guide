"""user-uploads 사진 bytes 조회 — 비공개 버킷 대응 (PRODUCTION_READINESS 결정 9 선행).

공개 URL(image_url)은 버킷을 비공개로 돌리면 400 이 되므로 service key 로
storage API 를 쓴다. storage_path 가 없는 레거시 행만 image_url 로 받는다.
"""
from __future__ import annotations

from typing import Any

import requests
from greenguide_common.supabase import Bucket


def fetch_upload_bytes(client: Any, row: dict, timeout: float = 30) -> bytes:
    """row(storage_path·image_url) → 사진 bytes. 실패는 예외로 올린다."""
    sp = row.get("storage_path")
    if sp:
        return client.storage.from_(str(Bucket.USER_UPLOADS)).download(sp)
    resp = requests.get(row["image_url"], timeout=timeout)
    resp.raise_for_status()
    return resp.content
