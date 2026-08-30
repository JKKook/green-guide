from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from greenguide_preprocessor import config, pipeline


def test_run_writes_manifest(fake_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 데이터셋은 fake_dataset(6클래스×2장) — source_path 가 PROJECT_ROOT 기준이 되도록 루트 교체
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "INTERIM_DIR", tmp_path / "interim")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "VECTORS_DIR", tmp_path / "processed" / "vectors")
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "processed" / "manifest.json")
    monkeypatch.setattr(pipeline, "ensure_dataset", lambda: fake_dataset)

    out = pipeline.run(upload_to_supabase=False, vectorize=True)
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["counts"]["processed"] == manifest["counts"]["cleansed"] > 0
    assert manifest["counts"]["failed"] == 0
    first = manifest["items"][0]
    assert {"id", "label", "source_path", "filename", "vector_path", "stats"} <= set(first)
    assert "original_url" not in first
    assert (tmp_path / first["vector_path"]).exists()


def test_process_item_raw_mode_with_store(fake_dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "PROJECT_ROOT", fake_dataset.parent)
    img = next(fake_dataset.rglob("*.jpg"))
    item = {"id": "x1", "label": img.parent.name, "source_path": str(img.relative_to(fake_dataset.parent)),
            "filename": img.name}
    store = MagicMock()
    store.upload_image.return_value = "https://x/x1.jpg"
    rec = pipeline._process_item(item, vectorize=False, store=store)
    assert rec == {**item, "original_url": "https://x/x1.jpg"}
    store.upload_image.assert_called_once()
