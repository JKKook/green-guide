"""사용자 업로드 이미지 + 메타데이터를 Supabase 에 저장 + 피드백 기록."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from supabase import Client, create_client

from src.core import config
from src.core.log import get_logger
from src.core.singleton import lazy_singleton

log = get_logger(__name__)


SUPABASE_URL: str | None = config.__dict__.get("SUPABASE_URL") or None
SUPABASE_KEY: str | None = config.__dict__.get("SUPABASE_KEY") or None


def _client() -> Client:
    """waste-classifier 의 .env 와 같은 Supabase 자격증명을 사용."""
    url, key = config.SUPABASE_URL, config.SUPABASE_KEY
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


# Supabase 불가 시 로컬 폴백 저장소 — A1(피드백 축적)이 인프라 장애에 멈추지
# 않게 한다 (2026-07-21 쿼터 제한 사태 중 도입). 로컬 서버(개발 모드)에서 유효;
# 운영 컨테이너 디스크는 휘발성이므로 어디까지나 보조 수단.
# 복구 후 scripts/sync_local_feedback.py 로 Supabase 에 승격.
_LOCAL_DIR = Path(__file__).resolve().parent.parent / "local_feedback"


# 저장 포맷: WebP 640px q75 — JPEG 720 q80(~35KB) 대비 ~55% 절감(~15KB/장).
# 재학습 입력(224~448px)에 충분한 해상도. 2026-08-29 전환, 기존 .jpg 행과 혼재 가능
# (소비처는 storage_path 확장자를 그대로 따르므로 문제 없음).
_STORE_MAX_SIDE = 640
_STORE_FORMAT = "WEBP"
_STORE_QUALITY = 75
_STORE_CONTENT_TYPE = "image/webp"


def _recompress_for_storage(
    image_bytes: bytes, content_type: str,
) -> tuple[bytes, str]:
    """저장용 재압축 — 긴 변 640px WebP q75. 실패 시 원본 그대로 (fail-open)."""
    try:
        import io  # noqa: PLC0415
        from PIL import Image, ImageOps  # noqa: PLC0415
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((_STORE_MAX_SIDE, _STORE_MAX_SIDE), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format=_STORE_FORMAT, quality=_STORE_QUALITY, method=4)
        out = buf.getvalue()
        # 재압축이 오히려 커지는 극단 케이스(이미 작은 저화질)는 원본 유지
        if len(out) < len(image_bytes):
            return out, _STORE_CONTENT_TYPE
        return image_bytes, content_type
    except Exception as exc:  # noqa: BLE001
        log.info(f"재압축 실패(원본 저장): {str(exc)[:60]}")
        return image_bytes, content_type


def prune_stale_uploads(days: int = 7) -> int:
    """피드백 없는 업로드를 N일 후 삭제 — 무료 쿼터 지속성 (수집 정책:
    라벨 가치가 확정된 사진만 장기 보관, 나머지는 휘발).

    대상: feedback_status 가 confirmed/corrected 가 아니고
          uploaded_at < now - days 인 행 + 스토리지 파일.
    반환: 삭제 건수. 실패는 개별 무시 (다음 주기 재시도).
    """
    from datetime import timedelta  # noqa: PLC0415
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    client = _client()
    rows = (
        client.table(_UPLOAD_TABLE)
        .select("id,storage_path,feedback_status")
        .lt("uploaded_at", cutoff)
        .execute().data or [])
    stale = [r for r in rows
             if r.get("feedback_status") not in ("confirmed", "corrected")]
    removed = 0
    for r in stale:
        try:
            sp = r.get("storage_path")
            if sp:
                client.storage.from_(_UPLOAD_BUCKET).remove([sp])
            client.table(_UPLOAD_TABLE).delete().eq("id", r["id"]).execute()
            removed += 1
        except Exception as exc:  # noqa: BLE001
            log.info(f"{r.get('id')} 삭제 실패(무시): {str(exc)[:60]}")
    if removed:
        log.info(f"피드백 없는 {days}일 경과 업로드 {removed}건 삭제")
    return removed


class UploadRecorder:
    """추론 결과를 user_uploads 에 기록하고 이미지를 Storage 에 업로드.

    Supabase 실패 시 로컬 폴백(_LOCAL_DIR)에 이미지+메타를 저장하고
    "local-" 접두 upload_id 를 반환 — 피드백도 같은 경로로 이어진다.
    """

    def __init__(self, client: Client | None = None) -> None:
        self.client = client or _client()

    # ── 로컬 폴백 ────────────────────────────────────────────────────────
    @staticmethod
    def _local_record(image_bytes: bytes, ext: str, prediction: dict[str, Any]) -> str:
        upload_id = "local-" + uuid.uuid4().hex[:12]
        _LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        (_LOCAL_DIR / f"{upload_id}{ext}").write_bytes(image_bytes)
        meta = {
            "id": upload_id,
            "predicted_class": prediction["predicted_class"],
            "predicted_confidence": prediction.get("confidence"),
            "model_arch": prediction.get("model_arch"),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "ext": ext,
            "feedback_status": "pending",
        }
        with (_LOCAL_DIR / "meta.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        return upload_id

    @staticmethod
    def _local_feedback(upload_id: str, confirmed: bool,
                        corrected_label: str | None) -> dict[str, Any]:
        rows = []
        meta_p = _LOCAL_DIR / "meta.jsonl"
        found = None
        for line in meta_p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["id"] == upload_id:
                status = "confirmed" if confirmed else "corrected"
                r["feedback_status"] = status
                r["feedback_label"] = (r["predicted_class"] if confirmed
                                        else corrected_label)
                r["feedback_at"] = datetime.now(timezone.utc).isoformat()
                found = r
            rows.append(r)
        if found is None:
            raise LookupError(f"upload_id 없음: {upload_id!r}")
        meta_p.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8")
        return found

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
        # 저장용 재압축 (무료 쿼터 지속성): 긴 변 640px·WebP q75 → 장당
        # ~220KB→~15KB. 재학습 입력(224/448)에 충분한 해상도.
        image_bytes, content_type = _recompress_for_storage(
            image_bytes, content_type)
        ext = _EXT_BY_CT.get(content_type, ".bin")
        label = prediction["predicted_class"]
        storage_path = f"{label}/{upload_id}{ext}"

        try:
            return self._remote_record(
                upload_id, ext, storage_path, image_bytes, content_type, prediction)
        except Exception as exc:  # noqa: BLE001
            log.info(f"Supabase 실패 → 로컬 폴백: {str(exc)[:80]}")
            return self._local_record(image_bytes, ext, prediction)

    def _remote_record(
        self, upload_id: str, ext: str, storage_path: str,
        image_bytes: bytes, content_type: str, prediction: dict[str, Any],
    ) -> str:
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

        if upload_id.startswith("local-"):
            return self._local_feedback(upload_id, confirmed, corrected_label)

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


@lazy_singleton
def get_recorder() -> UploadRecorder:
    return UploadRecorder()

