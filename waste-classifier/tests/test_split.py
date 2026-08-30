"""split 테스트."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from src import config
from src.split import load_splits, save_splits, stratified_split


def test_stratified_split_sizes(fake_dataset: tuple[list[dict], Path]) -> None:
    items, _ = fake_dataset
    splits = stratified_split(items, ratios={"train": 0.5, "val": 0.25, "test": 0.25}, seed=0)
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == len(items)


def test_stratified_split_no_overlap(fake_dataset: tuple[list[dict], Path]) -> None:
    items, _ = fake_dataset
    splits = stratified_split(items, seed=0)
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_stratified_split_preserves_label_distribution(
    fake_dataset: tuple[list[dict], Path],
) -> None:
    items, _ = fake_dataset
    splits = stratified_split(items, seed=0)
    # 각 split 에 모든 클래스가 최소 1개씩 포함
    for name in ("train", "val", "test"):
        labels_in_split = {items[i]["label"] for i in splits[name]}
        assert labels_in_split == set(config.CLASS_LABELS), (
            f"{name} split 에 일부 클래스 누락: {labels_in_split}"
        )


def test_save_and_load_roundtrip(fake_dataset: tuple[list[dict], Path], tmp_path: Path) -> None:
    items, _ = fake_dataset
    splits = stratified_split(items, seed=0)
    out = tmp_path / "splits.json"
    save_splits(splits, out)
    loaded = load_splits(out)
    assert loaded["train"] == splits["train"]
    assert loaded["val"] == splits["val"]
    assert loaded["test"] == splits["test"]
