"""SupabaseStore — 클라이언트 주입(mock)으로 네트워크 없이 규약 검증."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from greenguide_preprocessor.storage import SupabaseStore


def _store() -> tuple[SupabaseStore, MagicMock]:
    client = MagicMock()
    client.storage.from_.return_value.get_public_url.return_value = "https://x/public/glass/abc.jpg"
    client.table.return_value.upsert.return_value.execute.return_value = SimpleNamespace(data=[{"id": 1}])
    return SupabaseStore(client=client), client


def test_upload_image_path_and_content_type(sample_image: Path) -> None:
    store, client = _store()
    url = store.upload_image(sample_image, "glass", "abc")
    assert url == "https://x/public/glass/abc.jpg"
    kwargs = client.storage.from_.return_value.upload.call_args.kwargs
    assert kwargs["path"] == "glass/abc.jpg"
    assert kwargs["file_options"] == {"upsert": "true", "content-type": "image/jpeg"}


def test_upsert_many_chunks() -> None:
    store, client = _store()
    n = store.upsert_many([{"id": i} for i in range(250)], chunk_size=100)
    assert client.table.return_value.upsert.call_count == 3
    assert n == 3  # mock 은 chunk 당 1행 반환
