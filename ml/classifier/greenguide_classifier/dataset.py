"""Phase 1: PyTorch Dataset.

greenguide-preprocessor 의 manifest.json 을 읽고, raw JPEG 이미지를 학습 시점에
직접 로드 (on-the-fly). 과거엔 .npz 로 벡터화된 캐시(수 GB)를 읽었으나,
디스크 효율 + 표준 파이프라인을 위해 raw 직접 로드로 전환.

manifest item 의 `source_path` (preprocessor 루트 기준 상대경로) 를 사용.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from greenguide_common import imaging

from greenguide_classifier import config


def load_manifest(path: Path | None = None) -> list[dict[str, Any]]:
    path = path if path is not None else config.MANIFEST_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"manifest not found: {path}\n"
            f"greenguide-preprocessor 를 먼저 실행해주세요."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)["items"]


def _load_rgb_chw01(item: dict[str, Any]) -> torch.Tensor:
    """manifest item → (3, 224, 224) float32 텐서, [0,1] 범위 (정규화 전).

    source_path 는 preprocessor 루트 기준 상대경로.
    """
    src = item.get("source_path")
    if src is None:
        raise KeyError(f"item {item.get('id')} 에 source_path 없음")
    abs_path = config.PREPROCESSOR_ROOT / src
    with Image.open(abs_path) as im:
        im = im.convert("RGB").resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR,
        )
        arr = np.asarray(im, dtype=np.float32) / 255.0  # (H, W, C)
    chw = np.ascontiguousarray(arr.transpose(2, 0, 1))   # (C, H, W)
    return torch.from_numpy(chw)


class WasteDataset(Dataset):
    """MLP 입력용: flatten 1D 벡터 + 라벨 인덱스."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        vectors_dir: Path | None = None,
    ) -> None:
        self.items = items
        self.vectors_dir = vectors_dir if vectors_dir is not None else config.VECTORS_DIR

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.items[idx]
        npz_path = self.vectors_dir / f"{item['id']}.npz"
        with np.load(npz_path, allow_pickle=False) as data:
            vec = data["vector"].astype(np.float32, copy=False)
        return torch.from_numpy(vec), config.LABEL_TO_INDEX[item["label"]]


class WasteImageDataset(Dataset):
    """CNN 입력용: raw JPEG 을 (C, H, W) = (3, 224, 224) 로 로드 + ImageNet 정규화.

    Args:
        items: manifest items 리스트 (source_path 필요)
        augment: 학습 시 True — 색 편향 줄이는 augmentation 적용
                 (eval/test 에선 False)
    """

    def __init__(
        self,
        items: list[dict[str, Any]],
        augment: bool = False,
    ) -> None:
        self.items = items
        self.augment = augment
        # ImageNet 정규화
        self._mean = torch.tensor(list(imaging.IMAGENET_MEAN)).view(3, 1, 1)
        self._std = torch.tensor(list(imaging.IMAGENET_STD)).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.items[idx]
        x = _load_rgb_chw01(item)  # (3, 224, 224), [0,1]
        if self.augment:
            x = self._apply_augmentation(x)
        # ImageNet 정규화
        x = (x - self._mean) / self._std
        return x, config.LABEL_TO_INDEX[item["label"]]

    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """[0,1] RGB 텐서에 색 편향 완화 + 강건성 augmentation."""
        import random

        # 1) Horizontal flip (50%) — 형태는 좌우 무관
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])

        # 2) Random grayscale (20%) — 형태에 집중하도록 강제 ⭐ 색 편향 핵심 대응
        if random.random() < 0.20:
            gray = (x[0] * 0.299 + x[1] * 0.587 + x[2] * 0.114).unsqueeze(0)
            x = gray.expand(3, -1, -1).clone()

        # 3) Color jitter — brightness/contrast/saturation 무작위 변동
        if random.random() < 0.7:
            brightness_factor = 1.0 + (random.random() - 0.5) * 0.5
            x = x * brightness_factor
            contrast_factor = 1.0 + (random.random() - 0.5) * 0.5
            mean_val = x.mean(dim=(1, 2), keepdim=True)
            x = (x - mean_val) * contrast_factor + mean_val
            if random.random() < 0.5:
                saturation_factor = 1.0 + (random.random() - 0.5) * 0.6
                gray = (x[0] * 0.299 + x[1] * 0.587 + x[2] * 0.114).unsqueeze(0)
                x = gray + (x - gray) * saturation_factor

        return torch.clamp(x, 0.0, 1.0)


class WasteEdgeDataset(Dataset):
    """Edge stream 학습용 — Sobel edge magnitude 를 CNN 입력으로.

    Color 정보를 의도적으로 제거하고 **모양/윤곽** 에만 집중하게 학습.
    같은 CNN 아키텍처를 사용하지만 입력 데이터가 다름.

    파이프라인:
      .npz (정규화된 RGB) → 역정규화 → grayscale → Sobel 필터 → 3채널 복제
      → ImageNet 정규화 → 모델 입력
    """

    def __init__(
        self,
        items: list[dict[str, Any]],
        augment: bool = False,
    ) -> None:
        self.items = items
        self.augment = augment
        self._mean = torch.tensor(list(imaging.IMAGENET_MEAN)).view(3, 1, 1)
        self._std = torch.tensor(list(imaging.IMAGENET_STD)).view(3, 1, 1)
        # Sobel 커널
        self._sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                      dtype=torch.float32).view(1, 1, 3, 3)
        self._sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                      dtype=torch.float32).view(1, 1, 3, 3)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        import torch.nn.functional as F
        item = self.items[idx]
        x = _load_rgb_chw01(item)  # (3, 224, 224), [0,1] RGB

        # Grayscale
        gray = (x[0] * 0.299 + x[1] * 0.587 + x[2] * 0.114).unsqueeze(0).unsqueeze(0)
        # shape: (1, 1, H, W)

        # 3) Sobel edge
        edge_x = F.conv2d(gray, self._sobel_x, padding=1)
        edge_y = F.conv2d(gray, self._sobel_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2).squeeze(0)  # (1, H, W)

        # 4) 정규화 (0-1 범위로) + 3채널 복제
        edge_max = edge.max()
        if edge_max > 1e-6:
            edge = edge / edge_max
        edge_3c = edge.expand(3, -1, -1).clone()

        # 5) Augmentation (학습 시)
        if self.augment:
            import random
            if random.random() < 0.5:
                edge_3c = torch.flip(edge_3c, dims=[2])

        # 6) ImageNet 정규화 (ResNet pretrained 와 호환)
        return (edge_3c - self._mean) / self._std, config.LABEL_TO_INDEX[item["label"]]


def build_dataset(
    arch: str,
    items: list[dict[str, Any]],
    augment: bool = False,
    input_mode: str = "color",
) -> Dataset:
    """arch 별 적절한 Dataset 인스턴스 생성.

    Args:
        arch: "mlp" | "cnn" | "cnn_edge"
        augment: True 면 학습용 (Color/Grayscale/Flip)
        input_mode: "color" (기본 RGB) | "edge" (Sobel edge map) — CNN 전용
    """
    if arch == "mlp":
        return WasteDataset(items)
    if arch == "cnn_edge":
        return WasteEdgeDataset(items, augment=augment)
    if arch == "cnn":
        if input_mode == "edge":
            return WasteEdgeDataset(items, augment=augment)
        return WasteImageDataset(items, augment=augment)
    raise ValueError(f"unsupported arch={arch!r}")
