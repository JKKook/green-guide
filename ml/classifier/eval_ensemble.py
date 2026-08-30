"""Color + Edge ensemble 평가 — single 대비 성능 비교.

세 가지 비교:
  1. Color 모델만 (현재 production)
  2. Edge 모델만 (shape-focused, baseline)
  3. Color + Edge ensemble (산술 평균)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from greenguide_common.logging import get_logger
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from greenguide_classifier import config
from greenguide_classifier.dataset import build_dataset, load_manifest
from greenguide_classifier.infer import load_session, softmax
from greenguide_classifier.split import load_splits, subset_items

log = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
COLOR_ONNX = PROJECT_ROOT / "outputs" / "models" / "cnn" / "classifier.onnx"
EDGE_ONNX = PROJECT_ROOT / "outputs" / "models" / "cnn_edge" / "classifier.onnx"


def evaluate_ensemble():
    log.info(f"color model: {COLOR_ONNX}")
    log.info(f"edge  model: {EDGE_ONNX}")

    sess_color = load_session(COLOR_ONNX)
    sess_edge = load_session(EDGE_ONNX)
    color_input = sess_color.get_inputs()[0].name
    edge_input = sess_edge.get_inputs()[0].name

    items = load_manifest()
    splits = load_splits()
    test_items = subset_items(items, splits["test"])

    color_ds = build_dataset("cnn", test_items, input_mode="color")
    edge_ds = build_dataset("cnn", test_items, input_mode="edge")

    color_loader = DataLoader(color_ds, batch_size=32, shuffle=False)
    edge_loader = DataLoader(edge_ds, batch_size=32, shuffle=False)

    y_true, y_color, y_edge, y_ensemble = [], [], [], []

    for (xc, yc), (xe, ye) in zip(color_loader, edge_loader, strict=False):
        assert (yc == ye).all(), "label 순서 다름"
        xc_np = xc.numpy()
        xe_np = xe.numpy()

        logits_c = sess_color.run(["logits"], {color_input: xc_np})[0]
        logits_e = sess_edge.run(["logits"], {edge_input: xe_np})[0]

        probs_c = softmax(logits_c, axis=1)
        probs_e = softmax(logits_e, axis=1)
        probs_avg = (probs_c + probs_e) / 2.0

        y_true.extend(yc.tolist())
        y_color.extend(probs_c.argmax(axis=1).tolist())
        y_edge.extend(probs_e.argmax(axis=1).tolist())
        y_ensemble.extend(probs_avg.argmax(axis=1).tolist())

    y_true = np.array(y_true)
    labels = list(range(config.NUM_CLASSES))
    names = list(config.CLASS_LABELS)

    def summary(name, preds):
        preds = np.array(preds)
        acc = (preds == y_true).mean()
        p, r, f1, sup = precision_recall_fscore_support(
            y_true, preds, labels=labels, zero_division=0,
        )
        return acc, {names[i]: f1[i] for i in range(len(names))}

    acc_c, f1_c = summary("color", y_color)
    acc_e, f1_e = summary("edge", y_edge)
    acc_s, f1_s = summary("ensemble", y_ensemble)

    print()
    print("=" * 70)
    print(f"{'':12} {'Color':>10} {'Edge':>10} {'Ensemble':>10}")
    print("-" * 70)
    print(f"{'Accuracy':12} {acc_c*100:>9.2f}% {acc_e*100:>9.2f}% {acc_s*100:>9.2f}%")
    print()
    print("Per-class F1:")
    for name in names:
        print(f"  {name:10} "
              f"{f1_c[name]:>9.3f}  {f1_e[name]:>9.3f}  {f1_s[name]:>9.3f}")
    print()
    # Δ (ensemble − color)
    print(f"{'Δ (ens-clr)':12} {'':10} {'':10} "
          f"{(acc_s-acc_c)*100:>+9.2f}pp")

    # Save full report
    report = {
        "color": {"accuracy": float(acc_c), "f1_per_class": f1_c},
        "edge": {"accuracy": float(acc_e), "f1_per_class": f1_e},
        "ensemble": {"accuracy": float(acc_s), "f1_per_class": f1_s},
    }
    out = PROJECT_ROOT / "outputs" / "logs" / "ensemble_comparison.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    evaluate_ensemble()
