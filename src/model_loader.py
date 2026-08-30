"""Supabase model_versions 테이블 + Storage 에서 최신 ONNX 를 가져오는 로더.

흐름:
  - get_latest_active_version(): model_versions 에서 is_active=true row 조회
  - ensure_cached(meta): 캐시(`cache/models/v<version>/`)에 없으면 다운로드
  - resolve_model_paths(): 최신 active 모델 경로 또는 fallback (config) 반환

연결이 안 되거나 active row 가 없으면 silently fallback —
waste-api 는 항상 부팅 가능해야 함 (오프라인/초기 상태 대응).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from src.core import config
from src.core.log import get_logger

log = get_logger(__name__)


CACHE_ROOT: Path = config.PROJECT_ROOT / "cache" / "models"
_BUCKET = "models"


@dataclass(frozen=True)
class RemoteModelMeta:
    """Supabase model_versions 의 active row 한 줄."""
    version: str
    color_storage_path: str
    edge_storage_path: str | None
    color_sha256: str
    edge_sha256: str | None
    color_url: str
    edge_url: str | None
    test_accuracy: float | None
    num_classes: int
    class_labels: list[str]
    feedback_count: int

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "RemoteModelMeta":
        return cls(
            version=row["version"],
            color_storage_path=row["color_storage_path"],
            edge_storage_path=row.get("edge_storage_path"),
            color_sha256=row["color_sha256"],
            edge_sha256=row.get("edge_sha256"),
            color_url=row["color_url"],
            edge_url=row.get("edge_url"),
            test_accuracy=row.get("test_accuracy"),
            num_classes=row["num_classes"],
            class_labels=list(row.get("class_labels") or []),
            feedback_count=int(row.get("feedback_count") or 0),
        )


def _supabase_client():
    """Supabase client (없거나 env 미설정 시 None — fallback 트리거)."""
    try:
        from supabase import create_client
    except Exception:  # noqa: BLE001
        return None
    url, key = config.SUPABASE_URL, config.SUPABASE_KEY
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception as exc:  # noqa: BLE001
        log.info(f"supabase client init failed: {exc}")
        return None


def get_latest_active_version() -> RemoteModelMeta | None:
    """model_versions 에서 is_active=true row 1건 (없으면 None)."""
    client = _supabase_client()
    if client is None:
        return None
    try:
        res = (
            client.table("model_versions")
            .select("*")
            .eq("is_active", True)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        log.info(f"model_versions query failed: {exc}")
        return None
    rows = res.data or []
    if not rows:
        return None
    return RemoteModelMeta.from_row(rows[0])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, expected_sha: str | None) -> bool:
    """url → dest 스트리밍 다운로드. SHA 가 주어지면 검증. 실패 시 False."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with httpx.stream("GET", url, timeout=120.0, follow_redirects=True) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_bytes(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
    except Exception as exc:  # noqa: BLE001
        log.info(f"download failed {url}: {exc}")
        tmp.unlink(missing_ok=True)
        return False

    if expected_sha:
        got = _sha256(tmp)
        if got != expected_sha:
            log.info(f"sha256 mismatch: expected {expected_sha} got {got}")
            tmp.unlink(missing_ok=True)
            return False

    tmp.replace(dest)
    return True


def ensure_cached(meta: RemoteModelMeta) -> tuple[Path | None, Path | None]:
    """캐시에 ONNX 가 있으면 그대로 반환, 없거나 검증 실패면 다운로드.

    Returns: (color_path, edge_path or None). color 다운로드 실패 시 (None, None).
    """
    version_dir = CACHE_ROOT / f"v{meta.version}"
    color_path = version_dir / "classifier.onnx"
    edge_path = version_dir / "classifier_edge.onnx"

    # Color
    if not color_path.exists() or _sha256(color_path) != meta.color_sha256:
        log.info(f"downloading color ONNX v{meta.version}...")
        ok = _download(meta.color_url, color_path, meta.color_sha256)
        if not ok:
            return None, None

    # Edge (있을 때만)
    edge_out: Path | None = None
    if meta.edge_url and meta.edge_storage_path:
        if not edge_path.exists() or (
            meta.edge_sha256 and _sha256(edge_path) != meta.edge_sha256
        ):
            log.info(f"downloading edge ONNX v{meta.version}...")
            ok = _download(meta.edge_url, edge_path, meta.edge_sha256)
            if ok:
                edge_out = edge_path
        else:
            edge_out = edge_path

    return color_path, edge_out


def resolve_model_paths() -> tuple[Path, Path | None, RemoteModelMeta | None]:
    """현재 사용해야 할 ONNX 경로 결정.

    1. Supabase 의 active 버전이 있으면 → 캐시(또는 다운로드) 후 그 경로.
    2. 아니면 → config 의 기본 경로 (번들 / sibling).

    Returns: (color_path, edge_path or None, remote_meta or None).
    remote_meta 가 None 이면 fallback 으로 동작 중인 것.
    """
    meta = get_latest_active_version()
    if meta is not None:
        color, edge = ensure_cached(meta)
        if color is not None:
            log.info(f"using remote v{meta.version} (accuracy={meta.test_accuracy})")
            return color, edge, meta
        log.info("remote fetch failed → falling back to local")

    # Fallback
    log.info(f"using local fallback: {config.MODEL_PATH}")
    return config.MODEL_PATH, config.EDGE_MODEL_PATH, None
