"""Phase 6: 시각화 (MLP/CNN 공용)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from waste_common.logging import get_logger

from src import config

log = get_logger(__name__)


def plot_training_curves(arch: str = "mlp") -> Path:
    log_dir = config.arch_subdir(config.LOGS_DIR, arch)
    plot_dir = config.arch_subdir(config.PLOTS_DIR, arch)
    log_path = log_dir / "training_log.json"
    output_path = plot_dir / "training_curves.png"

    with log_path.open("r", encoding="utf-8") as f:
        train_log = json.load(f)

    history = train_log["history"]
    epochs = [h["epoch"] for h in history]
    tr_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    tr_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, tr_loss, label="train", marker="o", markersize=3)
    axes[0].plot(epochs, val_loss, label="val", marker="s", markersize=3)
    axes[0].set_title(f"{arch.upper()} Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, tr_acc, label="train", marker="o", markersize=3)
    axes[1].plot(epochs, val_acc, label="val", marker="s", markersize=3)
    axes[1].axhline(y=train_log["best_val_acc"], color="red", linestyle="--",
                    alpha=0.5, label=f"best val={train_log['best_val_acc']:.4f}")
    axes[1].set_title(f"{arch.upper()} Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[{arch}] training curves → {output_path}")
    return output_path


def plot_confusion_matrix(arch: str = "mlp", normalize: bool = True) -> Path:
    log_dir = config.arch_subdir(config.LOGS_DIR, arch)
    plot_dir = config.arch_subdir(config.PLOTS_DIR, arch)
    report_path = log_dir / "evaluation.json"
    output_path = plot_dir / "confusion_matrix.png"

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    cm = np.array(report["confusion_matrix"], dtype=np.float64)
    labels = report["class_labels"]
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{arch.upper()} Confusion Matrix "
                 f"({'normalized' if normalize else 'counts'})  "
                 f"accuracy={report['accuracy']:.4f}")

    fmt = ".2f" if normalize else "d"
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=10,
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[{arch}] confusion matrix → {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    arch = sys.argv[1] if len(sys.argv) > 1 else "mlp"
    plot_training_curves(arch)
    plot_confusion_matrix(arch)
