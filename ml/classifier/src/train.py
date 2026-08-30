"""Phase 4: 학습 루프 (MLP/CNN 공용).

train/val loop, early stopping, best checkpoint 저장, JSON 로그.
arch 파라미터로 모델 종류 분기.
"""
from __future__ import annotations

import json
import random
import statistics
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from waste_common.logging import get_logger

from src import config
from src.dataset import build_dataset, load_manifest
from src.frozen_test import build_splits
from src.model import build_model, count_parameters
from src.split import load_splits, save_splits, subset_items

log = get_logger(__name__)


def inverse_freq_weights(
    counts: dict[int, int], n_classes: int, cap_multiplier: float,
) -> torch.Tensor:
    """클래스 빈도 → inverse-frequency 가중치 (감독 공간 일반화).

    weight[c] = total / (n_classes * count[c]), median 의 cap_multiplier 배로 상한.
    상한이 없으면 etc(10장) 같은 극소 클래스가 과도한 가중치로 학습을 불안정하게 만듦.
    """
    total = sum(counts.values()) or 1
    raw = [total / (n_classes * counts.get(i, 1)) for i in range(n_classes)]
    med = statistics.median(raw) if raw else 1.0
    return torch.tensor(
        [min(w, med * cap_multiplier) for w in raw], dtype=torch.float32,
    )


def _compute_class_weights(
    train_items: list[dict[str, Any]],
    device: torch.device,
    cap_multiplier: float = 4.0,
) -> torch.Tensor:
    """학습 split 의 클래스 빈도로 inverse-frequency 가중치 계산."""
    counts = Counter(config.LABEL_TO_INDEX[it["label"]] for it in train_items)
    return inverse_freq_weights(counts, config.NUM_CLASSES, cap_multiplier).to(device)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    elapsed_sec: float


@dataclass
class ArchHyperparams:
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    patience: int


def get_hyperparams(arch: str) -> ArchHyperparams:
    if arch == "mlp":
        return ArchHyperparams(
            batch_size=config.MLP_BATCH_SIZE,
            num_epochs=config.MLP_NUM_EPOCHS,
            learning_rate=config.MLP_LEARNING_RATE,
            weight_decay=config.MLP_WEIGHT_DECAY,
            patience=config.MLP_EARLY_STOPPING_PATIENCE,
        )
    if arch in ("cnn", "cnn_edge"):
        return ArchHyperparams(
            batch_size=config.CNN_BATCH_SIZE,
            num_epochs=config.CNN_NUM_EPOCHS,
            learning_rate=config.CNN_LEARNING_RATE,
            weight_decay=config.CNN_WEIGHT_DECAY,
            patience=config.CNN_EARLY_STOPPING_PATIENCE,
        )
    raise ValueError(f"unsupported arch={arch!r}")


def model_kind(arch: str) -> str:
    """arch → 실제 모델 클래스 키 (cnn_edge 는 cnn 모델 사용)."""
    return "cnn" if arch == "cnn_edge" else arch


def _input_mode(arch: str) -> str:
    return "edge" if arch == "cnn_edge" else "color"


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    desc: str,
    step: Callable[..., tuple[torch.Tensor, ...]],
) -> tuple[float, int, list[float]]:
    """공용 epoch 러너 — optimizer 가 주어지면 train, 아니면 eval.

    step(logits, *targets) -> (loss, *stats). 반환: (loss 합, 샘플 수, stats 합).
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_count = 0
    sums: list[float] | None = None

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = [t.to(device, non_blocking=True) for t in batch]
            logits = model(batch[0])
            loss, *stats = step(logits, *batch[1:])

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = batch[0].size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
            sums = stats if sums is None else [a + b for a, b in zip(sums, stats, strict=True)]

    return total_loss, total_count, sums or []


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "",
) -> tuple[float, float]:
    """optimizer 가 주어지면 train, 아니면 eval."""
    def step(logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, int]:
        return criterion(logits, y), (logits.argmax(dim=1) == y).sum().item()

    total_loss, total_count, (total_correct,) = run_epoch(
        model, loader, device, optimizer, desc, step,
    )
    return total_loss / total_count, total_correct / total_count


def train(arch: str = "mlp") -> Path:
    """학습 실행. 최종 best checkpoint 경로 반환."""
    if arch not in config.SUPPORTED_ARCHS:
        raise ValueError(f"unsupported arch={arch!r}")

    config.ensure_directories()
    set_seed()
    device = pick_device()
    hp = get_hyperparams(arch)
    log.info(f"[{arch}] device={device}, hyperparams={hp}")

    # 1) 데이터
    items = load_manifest()
    log.info(f"[{arch}] manifest items: {len(items):,}")

    splits_path = config.SPLITS_DIR / "splits.json"
    if splits_path.exists():
        splits = load_splits()
        log.info(f"[{arch}] using existing splits at {splits_path}")
    else:
        # 고정 held-out test set 기준 분할 — test 멤버는 source_path 로 동결되어
        # 버전 간 정확도가 같은 잣대로 비교 가능 (frozen_test.py 참고).
        splits = build_splits(items)
        save_splits(splits)

    # 학습은 augmentation 활성, val/test 는 비활성
    # cnn_edge 는 같은 cnn 모델 클래스를 쓰지만 입력이 Sobel edge
    mode = _input_mode(arch)
    train_ds = build_dataset(arch, subset_items(items, splits["train"]),
                              augment=True, input_mode=mode)
    val_ds = build_dataset(arch, subset_items(items, splits["val"]),
                            augment=False, input_mode=mode)

    # raw JPEG 디코딩이 병목 → worker 병렬화로 대폭 가속.
    # 70K 이미지 규모에선 num_workers=0 이면 epoch 당 수십 분.
    _nw = 6
    train_loader = DataLoader(
        train_ds, batch_size=hp.batch_size, shuffle=True,
        num_workers=_nw, persistent_workers=True, prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=hp.batch_size, shuffle=False,
        num_workers=_nw, persistent_workers=True, prefetch_factor=4,
    )

    # 2) 모델 (cnn_edge 도 동일한 CNN 클래스 사용)
    model = build_model(model_kind(arch)).to(device)
    log.info(f"[{arch}] trainable params: {count_parameters(model):,}")

    # 클래스 불균형 보정 — inverse frequency 가중치.
    # clothes(7305) vs trash(834) vs etc(10) 처럼 편차가 크면 다수 클래스로
    # 편향됨. 가중치로 소수 클래스에 더 집중. 단, etc 같은 극소 클래스가
    # 학습을 망치지 않도록 median 의 N배로 cap (config.CNN_CLASS_WEIGHT_CAP, Stage D).
    class_weights = _compute_class_weights(
        subset_items(items, splits["train"]), device,
        cap_multiplier=config.CNN_CLASS_WEIGHT_CAP,
    )
    log.info(f"[{arch}] class weights: "
          f"{ {config.INDEX_TO_LABEL[i]: round(float(w), 2) for i, w in enumerate(class_weights)} }")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hp.learning_rate, weight_decay=hp.weight_decay,
    )

    # 3) 학습 루프
    history: list[EpochMetrics] = []
    best_val_acc = -1.0
    best_epoch = -1
    patience_counter = 0
    ckpt_dir = config.arch_subdir(config.CHECKPOINTS_DIR, arch)
    log_dir = config.arch_subdir(config.LOGS_DIR, arch)
    best_ckpt_path = ckpt_dir / "best.pt"

    for epoch in range(1, hp.num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _run_epoch(
            model, train_loader, criterion, device, optimizer,
            desc=f"epoch {epoch} train",
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, device, None,
            desc=f"epoch {epoch}  val ",
        )
        elapsed = time.time() - t0

        m = EpochMetrics(epoch, tr_loss, tr_acc, val_loss, val_acc, elapsed)
        history.append(m)
        log.info(
            f"[{arch} epoch {epoch:3d}] "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "arch": arch,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
            }, best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= hp.patience:
                log.info(f"[{arch}] early stopping at epoch {epoch} "
                      f"(no improvement for {hp.patience} epochs)")
                break

    # 4) 로그 저장
    log_path = log_dir / "training_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump({
            "arch": arch,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "hyperparams": asdict(hp),
            "history": [asdict(m) for m in history],
        }, f, ensure_ascii=False, indent=2)

    log.info(f"[{arch}] done. best epoch={best_epoch}, val_acc={best_val_acc:.4f}")
    log.info(f"[{arch}] checkpoint → {best_ckpt_path}")
    log.info(f"[{arch}] log        → {log_path}")
    return best_ckpt_path


if __name__ == "__main__":
    import sys
    arch = sys.argv[1] if len(sys.argv) > 1 else "mlp"
    train(arch=arch)
