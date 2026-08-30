"""greenguide-classifier CLI.

사용 예:
    python main.py train --arch mlp
    python main.py train --arch cnn
    python main.py all --arch cnn
    python main.py evaluate --arch cnn
"""
from __future__ import annotations

import argparse

from greenguide_classifier import config


def cmd_train(arch: str) -> int:
    from greenguide_classifier.train import train
    train(arch=arch)
    return 0


def cmd_evaluate(arch: str) -> int:
    from greenguide_classifier.evaluate import evaluate
    evaluate(arch=arch)
    return 0


def cmd_visualize(arch: str) -> int:
    from greenguide_classifier.visualize import plot_confusion_matrix, plot_training_curves
    plot_training_curves(arch=arch)
    plot_confusion_matrix(arch=arch)
    return 0


def cmd_export(arch: str) -> int:
    from greenguide_classifier.export import export_onnx
    export_onnx(arch=arch)
    return 0


def cmd_all(arch: str) -> int:
    for fn in (cmd_train, cmd_evaluate, cmd_visualize, cmd_export):
        rc = fn(arch)
        if rc != 0:
            return rc
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="greenguide-classifier")
    parser.add_argument(
        "command",
        choices=["train", "evaluate", "visualize", "export", "all"],
    )
    parser.add_argument(
        "--arch",
        choices=list(config.SUPPORTED_ARCHS),
        default="mlp",
        help="모델 아키텍처 (default: mlp)",
    )
    args = parser.parse_args()

    return {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "visualize": cmd_visualize,
        "export": cmd_export,
        "all": cmd_all,
    }[args.command](args.arch)


if __name__ == "__main__":
    raise SystemExit(main())
