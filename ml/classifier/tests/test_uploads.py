"""fetch_upload_bytes — 비공개 버킷 대응 (결정 9 선행)."""
from __future__ import annotations

import pytest

from greenguide_classifier import uploads


class _Bucket:
    def __init__(self, log: list) -> None:
        self.log = log

    def download(self, path: str) -> bytes:
        self.log.append(path)
        return b"from-storage"


class _Storage:
    def __init__(self, log: list) -> None:
        self.log = log
        self.bucket = None

    def from_(self, name: str) -> _Bucket:
        self.bucket = name
        return _Bucket(self.log)


class _Client:
    def __init__(self) -> None:
        self.log: list = []
        self.storage = _Storage(self.log)


def test_storage_path_uses_storage_api_not_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_http(*a, **k):
        raise AssertionError("공개 URL 을 호출하면 안 된다")
    monkeypatch.setattr(uploads.requests, "get", _no_http)
    cli = _Client()
    row = {"storage_path": "2026/10/x.jpg", "image_url": "https://public/x.jpg"}
    assert uploads.fetch_upload_bytes(cli, row) == b"from-storage"
    assert cli.log == ["2026/10/x.jpg"]
    assert cli.storage.bucket == str(uploads.Bucket.USER_UPLOADS)


def test_legacy_row_without_storage_path_uses_url(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        content = b"from-url"

        def raise_for_status(self) -> None:
            pass
    seen = []
    monkeypatch.setattr(uploads.requests, "get",
                        lambda url, timeout: seen.append(url) or _Resp())
    cli = _Client()
    assert uploads.fetch_upload_bytes(cli, {"image_url": "https://legacy"}) == b"from-url"
    assert seen == ["https://legacy"] and cli.log == []
