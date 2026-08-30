"""DINOv2 기반 분류기 구축 (cloud fallback 용).

흐름:
  1. facebook/dinov2-small 로드 (HF Hub, ~88MB)
  2. 학습 데이터 70K 의 DINOv2 embedding (384-dim) 추출 + npz 저장
  3. Linear classifier (384 → 13) 학습 (CE loss, 10 epochs)
  4. 통합 ONNX export (image → logits) — waste-api 에서 직접 사용

산출:
  outputs/models/dinov2_classifier/
    embeddings.npz       — (N, 384) features + labels (재사용)
    linear_head.pt       — torch checkpoint
    dinov2_classifier.onnx — 통합 (DINOv2 backbone + linear head)

사용:
  .venv/bin/python scripts/build_dinov2_classifier.py [--skip-extract] [--skip-train]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from _base import PREPROCESSOR_ROOT, PROJECT_ROOT

# DINOv2 의 interpolate_pos_encoding 이 MPS 미지원 op (upsample_bicubic2d) 사용 →
# 해당 연산만 CPU 로 폴백. 매 forward 마다 한 번 호출되므로 약간 느려지지만 작동.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn as nn
from greenguide_common import imaging
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from transformers import AutoModel

from greenguide_classifier.infer import pick_device

ImageFile.LOAD_TRUNCATED_IMAGES = True

MANIFEST_PATH = PREPROCESSOR_ROOT / "data" / "processed" / "manifest.json"
OUT_DIR = PROJECT_ROOT / "outputs" / "models" / "dinov2_classifier"

DINOV2_NAME = "facebook/dinov2-small"
EMBED_DIM = 384      # ViT-S/14 의 CLS token dim
INPUT_SIZE = 224

# ImageNet 정규화 (DINOv2 가 기대하는 입력)
_MEAN = list(imaging.IMAGENET_MEAN)
_STD = list(imaging.IMAGENET_STD)


# ─── DINOv2 wrapper (ONNX export 호환) ──────────────────────────
class DINOv2FeatureExtractor(nn.Module):
    """CLS token 만 반환 — ONNX export 시 dict output 회피."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # last_hidden_state: [B, num_patches+1, 384], CLS token = [:,0]
        out = self.model(pixel_values=pixel_values)
        return out.last_hidden_state[:, 0]  # [B, 384]


class DINOv2Classifier(nn.Module):
    """DINOv2 backbone + linear head — 통합 ONNX export 용."""

    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(EMBED_DIM, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.backbone(pixel_values)
        return self.head(features)


# ─── 1. Embedding 추출 ──────────────────────────────────────────
class ImageItemDataset(Dataset):
    def __init__(self, items: list[dict], transform):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        path = PREPROCESSOR_ROOT / it["source_path"]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, idx


def extract_embeddings(items: list[dict], device: torch.device,
                       batch_size: int = 32) -> np.ndarray:
    """DINOv2 로 모든 item 의 embedding 추출 → (N, 384)."""
    print(f"[dinov2] loading {DINOV2_NAME}...")
    backbone = AutoModel.from_pretrained(DINOV2_NAME).eval()
    extractor = DINOv2FeatureExtractor(backbone).to(device).eval()

    tf = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])
    ds = ImageItemDataset(items, tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    embeds = np.zeros((len(items), EMBED_DIM), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i, (x, indices) in enumerate(loader):
            x = x.to(device)
            feat = extractor(x).cpu().numpy()
            for j, idx in enumerate(indices):
                embeds[int(idx)] = feat[j]
            if (i + 1) % 50 == 0:
                done = (i + 1) * batch_size
                elapsed = time.time() - t0
                eta = elapsed * (len(items) - done) / max(done, 1)
                print(f"  [{done}/{len(items)}] elapsed={elapsed:.0f}s eta={eta:.0f}s")
    print(f"[dinov2] embeddings extracted ({len(items)}, {EMBED_DIM}) "
          f"in {time.time()-t0:.0f}s")
    return embeds


# ─── 2. Linear head 학습 ────────────────────────────────────────
def train_linear_head(
    embeds: np.ndarray, labels: np.ndarray, label_names: list[str],
    device: torch.device, epochs: int = 10, lr: float = 1e-3,
) -> nn.Linear:
    num_classes = len(label_names)
    print(f"[linear] training {EMBED_DIM} → {num_classes} ({epochs} epochs)")

    # 80/20 split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(embeds))
    cut = int(len(embeds) * 0.8)
    train_idx, val_idx = idx[:cut], idx[cut:]

    X_train = torch.tensor(embeds[train_idx]).to(device)
    y_train = torch.tensor(labels[train_idx], dtype=torch.long).to(device)
    X_val = torch.tensor(embeds[val_idx]).to(device)
    y_val = torch.tensor(labels[val_idx], dtype=torch.long).to(device)

    # 클래스 가중치 (불균형 보정)
    class_counts = np.bincount(labels[train_idx], minlength=num_classes)
    class_weights = torch.tensor(
        np.median(class_counts) / np.maximum(class_counts, 1),
        dtype=torch.float32,
    ).to(device)

    head = nn.Linear(EMBED_DIM, num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_acc = 0.0
    best_state = None
    batch = 512
    for epoch in range(1, epochs + 1):
        # train (단일 epoch에 mini-batch)
        head.train()
        perm = torch.randperm(len(X_train))
        loss_sum = correct = total = 0
        for i in range(0, len(X_train), batch):
            sl = perm[i:i + batch]
            optimizer.zero_grad()
            logits = head(X_train[sl])
            loss = criterion(logits, y_train[sl])
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(sl)
            correct += (logits.argmax(1) == y_train[sl]).sum().item()
            total += len(sl)

        head.eval()
        with torch.no_grad():
            val_logits = head(X_val)
            val_acc = (val_logits.argmax(1) == y_val).float().mean().item()
            # per-class recall
            per_class_recall = []
            for c in range(num_classes):
                mask = y_val == c
                if mask.sum() > 0:
                    r = (val_logits[mask].argmax(1) == y_val[mask]).float().mean().item()
                    per_class_recall.append((label_names[c], r, int(mask.sum())))

        train_acc = correct / total
        train_loss = loss_sum / total
        print(f"  epoch {epoch}/{epochs}: train_acc={train_acc:.4f} "
              f"loss={train_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = head.state_dict().copy()

    print(f"\n[linear] best val_acc={best_acc:.4f}")
    print("[linear] per-class recall (final epoch):")
    for name, r, n in sorted(per_class_recall, key=lambda x: -x[2]):
        print(f"    {name:<14} recall={r:.4f}  (n={n})")

    head.load_state_dict(best_state)
    return head


# ─── 3. 통합 ONNX export ────────────────────────────────────────
def export_combined_onnx(
    head: nn.Linear, label_names: list[str], onnx_path: Path,
) -> None:
    print(f"[onnx] exporting combined model to {onnx_path}")
    backbone = AutoModel.from_pretrained(DINOV2_NAME).eval()
    extractor = DINOv2FeatureExtractor(backbone).eval()
    full = DINOv2Classifier(extractor, len(label_names)).eval()
    full.head.load_state_dict(head.state_dict())

    full = full.to("cpu")
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        full, dummy, str(onnx_path),
        input_names=["pixel_values"], output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14, do_constant_folding=True,
    )
    print(f"[onnx] ✓ {onnx_path.stat().st_size // (1024*1024)} MB")

    # 라벨 매핑 저장 (waste-api 가 인덱스 → 라벨 매핑에 사용)
    labels_path = onnx_path.parent / "labels.json"
    labels_path.write_text(json.dumps(label_names, ensure_ascii=False))
    print(f"[onnx] ✓ labels: {labels_path}")


# ─── Main ───────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extract", action="store_true",
                    help="기존 embeddings.npz 재사용")
    ap.add_argument("--skip-train", action="store_true",
                    help="기존 linear_head.pt 재사용 (ONNX export 만)")
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    if not MANIFEST_PATH.exists():
        sys.exit(f"manifest 없음: {MANIFEST_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m = json.load(MANIFEST_PATH.open())
    items = m["items"]
    print(f"[main] manifest items: {len(items)}")

    label_names = sorted({it["label"] for it in items})
    label_to_idx = {n: i for i, n in enumerate(label_names)}
    print(f"[main] {len(label_names)} classes: {label_names}")

    device = pick_device()
    print(f"[main] device: {device}")

    # 1. Embedding 추출
    emb_path = OUT_DIR / "embeddings.npz"
    if args.skip_extract and emb_path.exists():
        data = np.load(emb_path)
        embeds = data["embeddings"]
        labels = data["labels"]
        print(f"[main] reused embeddings: {embeds.shape}")
    else:
        embeds = extract_embeddings(items, device)
        labels = np.array([label_to_idx[it["label"]] for it in items], dtype=np.int64)
        np.savez_compressed(emb_path, embeddings=embeds, labels=labels,
                            label_names=np.array(label_names))
        print(f"[main] ✓ embeddings saved: {emb_path}")

    print(f"[main] label distribution: {Counter(label_names[i] for i in labels[:1000])} (sample)")

    # 2. Linear head 학습
    head_path = OUT_DIR / "linear_head.pt"
    if args.skip_train and head_path.exists():
        head = nn.Linear(EMBED_DIM, len(label_names))
        head.load_state_dict(torch.load(head_path, weights_only=True))
        print(f"[main] reused linear head: {head_path}")
    else:
        head = train_linear_head(embeds, labels, label_names, device, args.epochs)
        torch.save(head.state_dict(), head_path)
        print(f"[main] ✓ linear head saved: {head_path}")

    # 3. 통합 ONNX export
    onnx_path = OUT_DIR / "dinov2_classifier.onnx"
    export_combined_onnx(head, label_names, onnx_path)

    print(f"\n[main] 다음: cp {onnx_path} ../waste-api/models/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
