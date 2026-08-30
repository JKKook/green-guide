"""WasteImageDataset (CNN 입력) 테스트."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src import config
from src.dataset import WasteImageDataset, build_dataset


def test_image_dataset_shape(fake_dataset: tuple[list[dict], Path]) -> None:
    items, _ = fake_dataset
    ds = WasteImageDataset(items)
    x, y = ds[0]
    assert x.shape == (config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert x.dtype == torch.float32
    assert 0 <= y < config.NUM_CLASSES


def test_image_dataset_matches_manual_pipeline(
    fake_dataset: tuple[list[dict], Path],
) -> None:
    """CNN raw 로딩이 수동 재계산(resize→[0,1]→ImageNet 정규화)과 일치.

    (과거 npz-flatten 등가성 테스트는 raw 직접 로딩 전환으로 전제가 사라져
    현재 파이프라인 검증으로 대체.)
    """
    import numpy as np
    from PIL import Image

    items, root = fake_dataset
    img_ds = WasteImageDataset(items)  # augment=False
    x, y = img_ds[0]

    with Image.open(root / items[0]["source_path"]) as im:
        im = im.convert("RGB").resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    chw = torch.from_numpy(arr.transpose(2, 0, 1).copy())
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    expected = (chw - mean) / std

    assert y == config.LABEL_TO_INDEX[items[0]["label"]]
    assert torch.allclose(x, expected, atol=1e-5)


@pytest.mark.parametrize("arch,expected_shape", [
    ("mlp", (config.INPUT_DIM,)),
    ("cnn", (config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)),
])
def test_build_dataset_shape(
    fake_dataset: tuple[list[dict], Path],
    arch: str,
    expected_shape: tuple[int, ...],
) -> None:
    items, _ = fake_dataset
    ds = build_dataset(arch, items)
    x, _ = ds[0]
    assert tuple(x.shape) == expected_shape


def test_build_dataset_invalid_arch(fake_dataset: tuple[list[dict], Path]) -> None:
    items, _ = fake_dataset
    with pytest.raises(ValueError):
        build_dataset("invalid", items)
