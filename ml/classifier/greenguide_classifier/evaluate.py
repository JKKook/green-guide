"""Phase 5: 평가 (MLP/CNN 공용).

best checkpoint 를 로드해서 test set 에서 accuracy / per-class metrics / confusion matrix.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from greenguide_common.logging import get_logger
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from greenguide_classifier import config
from greenguide_classifier.dataset import build_dataset, load_manifest
from greenguide_classifier.model import build_model
from greenguide_classifier.split import load_splits, subset_items
from greenguide_classifier.train import _input_mode, get_hyperparams, model_kind, pick_device

log = get_logger(__name__)


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(preds)


def evaluate(arch: str = "mlp") -> dict[str, Any]:
    if arch not in config.SUPPORTED_ARCHS:
        raise ValueError(f"unsupported arch={arch!r}")

    config.ensure_directories()
    device = pick_device()
    hp = get_hyperparams(arch)

    ckpt_path = config.arch_subdir(config.CHECKPOINTS_DIR, arch) / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    log.info(f"[{arch}] loaded checkpoint epoch {ckpt['epoch']} "
          f"(val_acc={ckpt['val_acc']:.4f})")

    model = build_model(model_kind(arch)).to(device)
    model.load_state_dict(ckpt["model_state"])

    items = load_manifest()
    splits = load_splits()
    test_ds = build_dataset(arch, subset_items(items, splits["test"]),
                              input_mode=_input_mode(arch))
    test_loader = DataLoader(test_ds, batch_size=hp.batch_size, shuffle=False)
    log.info(f"[{arch}] test size: {len(test_ds):,}")

    y_true, y_pred = collect_predictions(model, test_loader, device)

    accuracy = float((y_true == y_pred).mean())
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(config.NUM_CLASSES)), zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))

    report = {
        "arch": arch,
        "test_size": int(len(test_ds)),
        "accuracy": accuracy,
        "per_class": [
            {
                "label": config.CLASS_LABELS[i],
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(config.NUM_CLASSES)
        ],
        "confusion_matrix": cm.tolist(),
        "class_labels": list(config.CLASS_LABELS),
    }

    report_path = config.arch_subdir(config.LOGS_DIR, arch) / "evaluation.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print(classification_report(
        y_true, y_pred,
        labels=list(range(config.NUM_CLASSES)),
        target_names=list(config.CLASS_LABELS),
        zero_division=0,
    ))
    log.info(f"[{arch}] accuracy: {accuracy:.4f}")
    log.info(f"[{arch}] report   → {report_path}")
    return report


if __name__ == "__main__":
    import sys
    arch = sys.argv[1] if len(sys.argv) > 1 else "mlp"
    evaluate(arch=arch)
