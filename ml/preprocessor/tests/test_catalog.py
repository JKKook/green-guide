"""카탈로그 모듈 테스트."""
from __future__ import annotations

from pathlib import Path

import pytest

from greenguide_preprocessor import config
from greenguide_preprocessor.catalog import build_catalog, load_catalog, save_catalog


def test_build_catalog_counts_all_items(fake_dataset: Path) -> None:
    catalog = build_catalog(fake_dataset)
    assert catalog["count"] == len(config.CLASS_LABELS) * 2
    assert set(catalog["classes"]) == set(config.CLASS_LABELS)


def test_build_catalog_items_have_required_fields(fake_dataset: Path) -> None:
    catalog = build_catalog(fake_dataset)
    for item in catalog["items"]:
        assert {"id", "label", "source_path", "filename"} <= set(item.keys())
        assert item["label"] in config.CLASS_LABELS


def test_build_catalog_ids_are_unique(fake_dataset: Path) -> None:
    catalog = build_catalog(fake_dataset)
    ids = [item["id"] for item in catalog["items"]]
    assert len(ids) == len(set(ids))


def test_build_catalog_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_catalog(tmp_path / "nonexistent")


def test_save_and_load_catalog_roundtrip(fake_dataset: Path, tmp_path: Path) -> None:
    catalog = build_catalog(fake_dataset)
    out = tmp_path / "catalog.json"
    save_catalog(catalog, out)
    loaded = load_catalog(out)
    assert loaded["count"] == catalog["count"]
    assert loaded["items"][0]["id"] == catalog["items"][0]["id"]
