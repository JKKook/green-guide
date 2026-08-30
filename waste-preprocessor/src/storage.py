"""Phase 6: Supabase 연동 — 접속은 waste_common.supabase, 여기는 items 테이블·raw-images 버킷 규약만.

- Storage: 원본 이미지를 bucket 에 업로드 → public URL 획득
- Postgres: items 테이블에 메타데이터 upsert

테이블 스키마 (Supabase Dashboard 또는 아래 SQL 로 미리 생성):

    create table public.items (
        id              text primary key,
        label           text not null,
        original_url    text not null,
        vector_path     text not null,
        source_path     text not null,
        filename        text not null,
        stats           jsonb,
        created_at      timestamptz default now()
    );
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src import config
from waste_common import supabase
from waste_common.imaging import content_type

if TYPE_CHECKING:
    from supabase import Client


class SupabaseStore:
    """Storage + Postgres 헬퍼."""

    def __init__(self, client: "Client | None" = None) -> None:
        self.client = client or supabase.get_client()
        self.bucket = config.SUPABASE_BUCKET
        self.table = config.SUPABASE_TABLE

    def upload_image(self, local_path: Path, label: str, item_id: str) -> str:
        """이미지를 {label}/{item_id}{ext} 로 업로드하고 public URL 반환."""
        remote_path = f"{label}/{item_id}{local_path.suffix.lower()}"
        storage = self.client.storage.from_(self.bucket)
        with local_path.open("rb") as f:
            storage.upload(
                path=remote_path,
                file=f,
                file_options={"upsert": "true", "content-type": content_type(local_path)},
            )
        return storage.get_public_url(remote_path)

    def upsert_item(self, item: dict[str, Any]) -> dict[str, Any]:
        response = self.client.table(self.table).upsert(item).execute()
        return response.data[0] if response.data else {}

    def upsert_many(self, items: list[dict[str, Any]], chunk_size: int = 100) -> int:
        total = 0
        for start in range(0, len(items), chunk_size):
            chunk = items[start:start + chunk_size]
            response = self.client.table(self.table).upsert(chunk).execute()
            total += len(response.data or [])
        return total


if __name__ == "__main__":
    store = SupabaseStore()
    print(f"[storage] connected. bucket={store.bucket} table={store.table}")
