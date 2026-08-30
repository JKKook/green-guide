"""Stage 1 — Binary (waste vs non_object) classifier 학습.

Cascade 1단계: 사용자 입력이 '폐기물 분류 대상' 인지 OOD (손/배경) 인지 판정.
- positive (waste): non_object 제외 12 클래스 모두
- negative (non_object): 손/배경/잡 객체

작은 모델 (MobileNetV3-Small) — 모바일 추론 빠르고 binary 라 충분히 정확.

사용:
    .venv/bin/python scripts/train_stage1_binary.py [--epochs 5] [--batch-size 64]
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from _base import PREPROCESSOR_ROOT, PROJECT_ROOT, make_parser
from greenguide_common import imaging
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from greenguide_classifier.infer import pick_device

ImageFile.LOAD_TRUNCATED_IMAGES = True

MANIFEST_PATH = PREPROCESSOR_ROOT / "data" / "processed" / "manifest.json"
OUT_DIR = PROJECT_ROOT / "outputs" / "models" / "stage1_binary"
INPUT_SIZE = 224

_MEAN = list(imaging.IMAGENET_MEAN)
_STD = list(imaging.IMAGENET_STD)


class BinaryWasteDataset(Dataset):
    """manifest → (image, 0/1). 1 = waste, 0 = non_object."""

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
        y = 0 if it["label"] == "non_object" else 1
        return x, y


def build_model() -> nn.Module:
    m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # 마지막 classifier FC 를 2-output 으로 교체
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, 2)
    return m


def train(args) -> int:
    if not MANIFEST_PATH.exists():
        sys.exit(f"manifest 없음: {MANIFEST_PATH}")

    m = json.load(MANIFEST_PATH.open())
    items = m["items"]
    label_counts = Counter("non_object" if it["label"] == "non_object" else "waste" for it in items)
    n_non = label_counts["non_object"]
    n_waste = label_counts["waste"]
    print(f"[stage1] dataset: non_object={n_non}, waste={n_waste} "
          f"(ratio 1:{n_waste/max(n_non,1):.1f})")

    # 80/20 split (stratified)
    rng = np.random.default_rng(args.seed)
    by_label = {"non_object": [], "waste": []}
    for it in items:
        lbl = "non_object" if it["label"] == "non_object" else "waste"
        by_label[lbl].append(it)
    train_items, val_items = [], []
    for lbl, lst in by_label.items():
        idx = rng.permutation(len(lst))
        cut = int(len(lst) * 0.8)
        train_items += [lst[i] for i in idx[:cut]]
        val_items += [lst[i] for i in idx[cut:]]
    print(f"  train: {len(train_items)} (non={sum(1 for x in train_items if x['label']=='non_object')}), "
          f"val: {len(val_items)} (non={sum(1 for x in val_items if x['label']=='non_object')})")

    # transforms
    train_tf = T.Compose([
        T.Resize((INPUT_SIZE + 16, INPUT_SIZE + 16)),
        T.RandomCrop(INPUT_SIZE),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])
    val_tf = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])
    train_ds = BinaryWasteDataset(train_items, train_tf)
    val_ds = BinaryWasteDataset(val_items, val_tf)

    # WeightedRandomSampler — non_object 가 96배 적으므로 balanced batch.
    weights = []
    for it in train_items:
        if it["label"] == "non_object":
            weights.append(1.0 / max(sum(1 for x in train_items if x["label"] == "non_object"), 1))
        else:
            weights.append(1.0 / max(sum(1 for x in train_items if x["label"] != "non_object"), 1))
    # 위 계산 비효율 — 한 번에 계산
    n_train_non = sum(1 for x in train_items if x["label"] == "non_object")
    n_train_waste = len(train_items) - n_train_non
    weights = [1.0/n_train_non if it["label"] == "non_object" else 1.0/n_train_waste
               for it in train_items]
    sampler = WeightedRandomSampler(weights=weights, num_samples=min(len(train_items), 2*n_train_waste),
                                    replacement=True)

    device = pick_device()
    print(f"[stage1] device: {device}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=False)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_correct = train_total = 0
        train_loss_sum = 0.0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.size(0)
            train_loss_sum += loss.item() * y.size(0)
            if i % 50 == 0:
                print(f"  [epoch {epoch} iter {i}/{len(train_loader)}] "
                      f"loss={loss.item():.4f} acc={train_correct/train_total:.4f}")

        train_acc = train_correct / max(train_total, 1)
        train_loss = train_loss_sum / max(train_total, 1)

        # validation
        model.eval()
        val_correct = val_total = 0
        val_non_correct = val_non_total = 0
        val_waste_correct = val_waste_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                preds = model(x).argmax(1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
                # per-class breakdown
                non_mask = (y == 0)
                waste_mask = (y == 1)
                val_non_correct += ((preds == y) & non_mask).sum().item()
                val_non_total += non_mask.sum().item()
                val_waste_correct += ((preds == y) & waste_mask).sum().item()
                val_waste_total += waste_mask.sum().item()

        val_acc = val_correct / max(val_total, 1)
        non_recall = val_non_correct / max(val_non_total, 1)
        waste_recall = val_waste_correct / max(val_waste_total, 1)
        dt = time.time() - t0
        print(f"[stage1] epoch {epoch}/{args.epochs} done "
              f"({dt:.0f}s) train acc={train_acc:.4f} loss={train_loss:.4f} | "
              f"val acc={val_acc:.4f} non_recall={non_recall:.4f} waste_recall={waste_recall:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [best] val_acc={val_acc:.4f}")

    if best_state is None:
        sys.exit("[stage1] no best state captured")

    # save best
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_DIR / "best.pt"
    torch.save({"model_state": best_state, "val_acc": best_val_acc,
                "epoch": args.epochs}, ckpt_path)
    print(f"[stage1] ✓ best.pt saved ({best_val_acc:.4f}): {ckpt_path}")

    # ONNX export
    model.load_state_dict(best_state)
    model.eval().to("cpu")
    onnx_path = OUT_DIR / "stage1_binary.onnx"
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14, do_constant_folding=True,
    )
    print(f"[stage1] ✓ ONNX exported: {onnx_path} "
          f"({onnx_path.stat().st_size // 1024} KB)")

    return 0


def main() -> int:
    ap = make_parser("train_stage1_binary", "Stage 1 binary classifier")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
