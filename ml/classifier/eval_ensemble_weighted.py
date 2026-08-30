"""Weighted ensemble — color weight grid search.

ensemble probs = w * color_probs + (1-w) * edge_probs
w 를 0.1 ~ 0.9 까지 변경하며 최적값 찾기.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from greenguide_common.logging import get_logger
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from greenguide_classifier import config
from greenguide_classifier.dataset import build_dataset, load_manifest
from greenguide_classifier.split import load_splits, subset_items

log = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
COLOR_ONNX = PROJECT_ROOT / "outputs" / "models" / "cnn" / "classifier.onnx"
EDGE_ONNX = PROJECT_ROOT / "outputs" / "models" / "cnn_edge" / "classifier.onnx"


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def collect_probs():
    sc = ort.InferenceSession(str(COLOR_ONNX), providers=["CPUExecutionProvider"])
    se = ort.InferenceSession(str(EDGE_ONNX), providers=["CPUExecutionProvider"])
    color_in, edge_in = sc.get_inputs()[0].name, se.get_inputs()[0].name

    items = load_manifest()
    splits = load_splits()
    test_items = subset_items(items, splits["test"])
    color_ds = build_dataset("cnn", test_items, input_mode="color")
    edge_ds = build_dataset("cnn", test_items, input_mode="edge")
    color_loader = DataLoader(color_ds, batch_size=32, shuffle=False)
    edge_loader = DataLoader(edge_ds, batch_size=32, shuffle=False)

    ys, color_probs, edge_probs = [], [], []
    for (xc, yc), (xe, _) in zip(color_loader, edge_loader):
        logits_c = sc.run(["logits"], {color_in: xc.numpy()})[0]
        logits_e = se.run(["logits"], {edge_in: xe.numpy()})[0]
        color_probs.append(_softmax(logits_c))
        edge_probs.append(_softmax(logits_e))
        ys.append(yc.numpy())
    return (
        np.concatenate(ys),
        np.concatenate(color_probs),
        np.concatenate(edge_probs),
    )


def main():
    log.info("Loading probs...")
    y_true, p_color, p_edge = collect_probs()
    labels = list(range(config.NUM_CLASSES))
    names = list(config.CLASS_LABELS)

    print()
    print(f"{'color_weight':>12} {'acc':>8} {'macro_f1':>10}")
    print("-" * 32)
    best = (-1.0, None, None)
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        p = w * p_color + (1 - w) * p_edge
        preds = p.argmax(axis=1)
        acc = (preds == y_true).mean()
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, preds, labels=labels, zero_division=0,
        )
        macro_f1 = f1.mean()
        marker = " ★" if acc > best[0] else ""
        print(f"{w:>12.2f} {acc*100:>7.2f}% {macro_f1:>10.3f}{marker}")
        if acc > best[0]:
            best = (acc, w, macro_f1)

    print()
    print(f"Best: w={best[1]:.2f} → acc {best[0]*100:.2f}%, macro F1 {best[2]:.3f}")

    # 최적 weight 의 per-class F1
    w = best[1]
    p = w * p_color + (1 - w) * p_edge
    preds = p.argmax(axis=1)
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, preds, labels=labels, zero_division=0,
    )
    print()
    print(f"Per-class F1 @ w={w}:")
    for i, name in enumerate(names):
        print(f"  {name:10}: {f1[i]:.3f}")

    # 기준: color 단독
    preds_color = p_color.argmax(axis=1)
    acc_color = (preds_color == y_true).mean()
    _, _, f1_color, _ = precision_recall_fscore_support(
        y_true, preds_color, labels=labels, zero_division=0,
    )
    print()
    print("vs color-only baseline:")
    print(f"  acc:      {acc_color*100:.2f}% → {best[0]*100:.2f}% "
          f"({(best[0]-acc_color)*100:+.2f}pp)")


if __name__ == "__main__":
    main()
