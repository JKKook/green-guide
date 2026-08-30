"""WasteDataset 테스트."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src import config
from src.dataset import WasteDataset, load_manifest


def test_load_manifest_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError):
        load_manifest()


def test_dataset_len(fake_dataset: tuple[list[dict], Path]) -> None:
    items, _ = fake_dataset
    ds = WasteDataset(items)
    # CLASS_LABELS 는 동적 (manifest 기준) — 하드코딩 대신 파생값으로 검증
    assert len(ds) == len(items) == len(config.CLASS_LABELS) * 8


def test_dataset_getitem_shape_and_dtype(fake_dataset: tuple[list[dict], Path]) -> None:
    items, _ = fake_dataset
    ds = WasteDataset(items)
    vec, label = ds[0]
    assert isinstance(vec, torch.Tensor)
    assert vec.shape == (config.INPUT_DIM,)
    assert vec.dtype == torch.float32
    assert isinstance(label, int)
    assert 0 <= label < config.NUM_CLASSES


def test_dataset_label_mapping_consistent(fake_dataset: tuple[list[dict], Path]) -> None:
    items, _ = fake_dataset
    ds = WasteDataset(items)
    for i in range(len(ds)):
        _, idx = ds[i]
        assert config.INDEX_TO_LABEL[idx] == items[i]["label"]
