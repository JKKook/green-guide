"""사용자 업로드 이미지 + 메타데이터를 Supabase 에 저장 + 피드백 기록."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from supabase import Client, create_client

from src import config


SUPABASE_URL: str | None = config.__dict__.get("SUPABASE_URL") or None
SUPABASE_KEY: str | None = config.__dict__.get("SUPABASE_KEY") or None


def _client() -> Client:
    """waste-classifier 의 .env 와 같은 Supabase 자격증명을 사용."""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL / SUPABASE_KEY 가 설정되지 않음. .env 파일을 확인하세요."
        )
    return create_client(url, key)


_UPLOAD_BUCKET = "user-uploads"
_UPLOAD_TABLE = "user_uploads"

# 모델 라벨 → MIME 확장자
_EXT_BY_CT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}


class UploadRecorder:
    """추론 결과를 user_uploads 에 기록하고 이미지를 Storage 에 업로드."""

    def __init__(self, client: Client | None = None) -> None:
        self.client = client or _client()

    def record_prediction(
        self,
        image_bytes: bytes,
        content_type: str,
        prediction: dict[str, Any],
    ) -> str:
        """업로드 + INSERT 후 upload_id 반환.

        prediction 은 inference.WasteClassifier.predict() 결과 dict.
        """
        upload_id = uuid.uuid4().hex[:16]
        ext = _EXT_BY_CT.get(content_type, ".bin")
        label = prediction["predicted_class"]
        storage_path = f"{label}/{upload_id}{ext}"

        # 1) Storage 업로드
        self.client.storage.from_(_UPLOAD_BUCKET).upload(
            path=storage_path,
            file=image_bytes,
            file_options={"upsert": "true", "content-type": content_type},
        )
        image_url = self.client.storage.from_(_UPLOAD_BUCKET).get_public_url(storage_path)

        # 2) Postgres INSERT
        row = {
            "id": upload_id,
            "image_url": image_url,
            "storage_path": storage_path,
            "predicted_class": prediction["predicted_class"],
            "predicted_confidence": prediction["confidence"],
            "all_probabilities": prediction["all_probabilities"],
            "model_arch": prediction["model_arch"],
            "inference_ms": prediction["inference_ms"],
            "feedback_status": "pending",
        }
        self.client.table(_UPLOAD_TABLE).insert(row).execute()
        return upload_id

    def record_feedback(
        self,
        upload_id: str,
        confirmed: bool,
        corrected_label: str | None,
    ) -> dict[str, Any]:
        """사용자 피드백을 user_uploads 에 기록.

        Note: 라벨 유효성은 api.py 의 endpoint 에서 ClassRegistry (Supabase 동적 라벨)
        로 이미 검증됨. 여기서는 confirmed/corrected_label 의 상호 배타성만 확인.
        """
        if confirmed and corrected_label is not None:
            raise ValueError("confirmed=True 면 corrected_label 은 None 이어야 함")
        if not confirmed and corrected_label is None:
            raise ValueError("confirmed=False 면 corrected_label 필수")

        # 현재 row 조회 — predicted_class 가 정답일 경우 feedback_label 로 그대로 저장
        existing = (
            self.client.table(_UPLOAD_TABLE)
            .select("predicted_class")
            .eq("id", upload_id)
            .single()
            .execute()
        )
        if not existing.data:
            raise LookupError(f"upload_id 없음: {upload_id!r}")

        status = "confirmed" if confirmed else "corrected"
        label = existing.data["predicted_class"] if confirmed else corrected_label

        update = {
            "feedback_status": status,
            "feedback_label": label,
            "feedback_at": datetime.now(timezone.utc).isoformat(),
        }
        result = (
            self.client.table(_UPLOAD_TABLE)
            .update(update)
            .eq("id", upload_id)
            .execute()
        )
        if not result.data:
            raise LookupError(f"업데이트 실패 — upload_id 없음: {upload_id!r}")
        return result.data[0]


_recorder: UploadRecorder | None = None


def get_recorder() -> UploadRecorder:
    global _recorder
    if _recorder is None:
        _recorder = UploadRecorder()
    return _recorder


def reset_recorder() -> None:
    global _recorder
    _recorder = None
