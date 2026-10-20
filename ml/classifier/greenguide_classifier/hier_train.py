"""계층 학습 — fine head + 대분류 롤업 loss.

핵심 아이디어 (GREENGUIDE_BLUEPRINT.md §2):
- 모델 출력 = fine 공간 logits (NUM_FINE)
- fine 라벨 아이템:   CE(logits, fine_idx)
- coarse 라벨 아이템: NLL( log P(coarse), coarse_idx ),
    log P(coarse) = logsumexp( log_softmax(logits)[children] )
  → 기존 대분류-라벨 데이터(glass/plastic/styrofoam 등 수만 장)도
    fine head 학습에 그대로 기여한다.

기존 flat 파이프라인(greenguide_classifier/train.py)은 건드리지 않는다.
실행: .venv/bin/python -m greenguide_classifier.hier_train
"""
from __future__ import annotations

import json
import os as _os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from greenguide_common.logging import get_logger
from greenguide_common.taxonomy import (
    COARSE_LABELS,
    FINE_IDX_TO_COARSE_IDX,
    FINE_LABELS,
    NUM_COARSE,
    NUM_FINE,
)
from torch.utils.data import DataLoader

from greenguide_classifier import config
from greenguide_classifier.hier_dataset import (
    HierImageDataset,
    build_hier_items,
    load_or_build_hier_splits,
)
from greenguide_classifier.model import build_hier_model, count_parameters
from greenguide_classifier.train import (
    ArchHyperparams,
    inverse_freq_weights,
    pick_device,
    run_epoch,
    set_seed,
)

log = get_logger(__name__)

ARCH = "cnn_hier"

BACKBONE = _os.getenv("GREENGUIDE_HIER_BACKBONE", "resnet18")
LABEL_SMOOTH = float(_os.getenv("GREENGUIDE_HIER_LABEL_SMOOTH", "0.0"))
# 1 이면 last.pt(매 epoch 저장)에서 이어 학습 — 수면·재부팅으로 끊긴 사이클 재개용.
# v14 는 세 번(10-10 ×2, 10-18) 모두 중간에 죽어 매번 처음부터 돌았다.
RESUME = _os.getenv("GREENGUIDE_HIER_RESUME", "0") == "1"
CKPT_DIR = config.CHECKPOINTS_DIR / ARCH
LOG_DIR = config.LOGS_DIR / ARCH
LAST_CKPT_NAME = "last.pt"


def get_hier_hyperparams() -> ArchHyperparams:
    """CNN 하이퍼파라미터 재사용 (동일 백본)."""
    return ArchHyperparams(
        batch_size=config.CNN_BATCH_SIZE,
        num_epochs=config.CNN_NUM_EPOCHS,
        learning_rate=config.CNN_LEARNING_RATE,
        weight_decay=config.CNN_WEIGHT_DECAY,
        patience=config.CNN_EARLY_STOPPING_PATIENCE,
    )


class HierarchicalLoss(nn.Module):
    """fine CE + coarse 롤업 NLL 혼합 (아이템 단위 마스크).

    class weight 는 fine/coarse 공간 각각 inverse-freq(cap) 로 계산해 적용.
    """

    def __init__(
        self,
        fine_weights: torch.Tensor,
        coarse_weights: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("fine_w", fine_weights)
        self.register_buffer("coarse_w", coarse_weights)
        # fine → coarse 매핑 (롤업 인덱스)
        self.register_buffer(
            "f2c", torch.tensor(FINE_IDX_TO_COARSE_IDX, dtype=torch.long),
        )

    def coarse_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """(B, NUM_FINE) logits → (B, NUM_COARSE) log P(coarse).

        log P(c) = logsumexp_{f∈children(c)} log_softmax(logits)_f
        scatter 기반 — 클래스 수가 늘어도 파이썬 루프 없음.
        """
        logp = F.log_softmax(logits, dim=1)                     # (B, F)
        # per-coarse logsumexp: exp 후 scatter_add → log (수치 안정 위해 max-shift)
        m = logp.max(dim=1, keepdim=True).values                # (B, 1)
        exp_shift = (logp - m).exp()                            # (B, F)
        summed = torch.zeros(
            logp.size(0), NUM_COARSE, dtype=logp.dtype, device=logp.device,
        ).scatter_add_(1, self.f2c.expand(logp.size(0), -1), exp_shift)  # (B, C)
        return summed.clamp_min(1e-12).log() + m

    def forward(
        self,
        logits: torch.Tensor,       # (B, NUM_FINE)
        is_fine: torch.Tensor,      # (B,) 1=fine, 0=coarse
        sup_idx: torch.Tensor,      # (B,) fine 또는 coarse 인덱스
    ) -> torch.Tensor:
        losses = torch.zeros(logits.size(0), dtype=logits.dtype, device=logits.device)

        fine_mask = is_fine.bool()
        if fine_mask.any():
            fl = F.cross_entropy(
                logits[fine_mask], sup_idx[fine_mask],
                weight=self.fine_w, reduction="none",
                label_smoothing=LABEL_SMOOTH,
            )
            losses[fine_mask] = fl

        coarse_mask = ~fine_mask
        if coarse_mask.any():
            clogp = self.coarse_log_probs(logits[coarse_mask])   # (Bc, C)
            ci = sup_idx[coarse_mask]
            nll = -clogp.gather(1, ci.unsqueeze(1)).squeeze(1)
            losses[coarse_mask] = nll * self.coarse_w[ci]

        return losses.mean()


def compute_hier_weights(
    train_items: list[dict[str, Any]], cap_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fine/coarse 감독 빈도 각각으로 가중치 계산.

    coarse 아이템은 fine 빈도에 균등 분배로 반영 (근사) — fine 가중치가
    coarse 데이터 규모를 무시하고 폭주하지 않도록.
    """
    fine_counts: Counter[int] = Counter()
    coarse_counts: Counter[int] = Counter()
    for it in train_items:
        if it["sup_kind"] == "fine":
            fine_counts[it["sup_idx"]] += 1
        else:
            coarse_counts[it["sup_idx"]] += 1
            # 균등 분배 근사: coarse 1장 = 각 child 에 1/n 장
            children = [
                fi for fi, ci in enumerate(FINE_IDX_TO_COARSE_IDX)
                if ci == it["sup_idx"]
            ]
            for fi in children:
                fine_counts[fi] += 1 / len(children)

    fine_int = {k: max(1, round(v)) for k, v in fine_counts.items()}
    fine_w = inverse_freq_weights(fine_int, NUM_FINE, cap_multiplier)
    coarse_w = inverse_freq_weights(dict(coarse_counts), NUM_COARSE, cap_multiplier)
    return fine_w, coarse_w


def _run_hier_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: HierarchicalLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "",
) -> tuple[float, float, float]:
    """반환: (loss, fine_acc — fine 아이템만, coarse_acc — 전체 롤업)."""
    f2c = torch.tensor(FINE_IDX_TO_COARSE_IDX, dtype=torch.long, device=device)

    def step(
        logits: torch.Tensor, is_fine: torch.Tensor, sup_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int, int]:
        loss = criterion(logits, is_fine, sup_idx)
        pred_fine = logits.argmax(dim=1)
        pred_coarse = f2c[pred_fine]
        fm = is_fine.bool()
        fine_correct = fine_count = 0
        if fm.any():
            fine_correct = (pred_fine[fm] == sup_idx[fm]).sum().item()
            fine_count = int(fm.sum())
        # coarse 정답: fine 아이템은 롤업, coarse 아이템은 그대로
        true_coarse = torch.where(fm, f2c[sup_idx.clamp(max=NUM_FINE - 1)], sup_idx)
        coarse_correct = (pred_coarse == true_coarse).sum().item()
        return loss, fine_correct, fine_count, coarse_correct

    total_loss, total_count, (fine_correct, fine_count, coarse_correct) = run_epoch(
        model, loader, device, optimizer, desc, step,
    )
    return (
        total_loss / max(total_count, 1),
        fine_correct / max(fine_count, 1),
        coarse_correct / max(total_count, 1),
    )


def save_last_checkpoint(
    path: Path, *, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer,
    history: list[dict], best_score: float, best_epoch: int, patience_counter: int,
) -> None:
    """epoch 끝 상태 전체 저장 — 모델·옵티마이저·이력·조기종료 카운터.

    best.pt 와 달리 '가장 좋은' 것이 아니라 '가장 최근' 상태. 임시 파일에 쓰고
    교체해 저장 도중 죽어도 이전 last.pt 가 남게 한다.
    """
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "arch": ARCH,
        "backbone": BACKBONE,
        "label_smooth": LABEL_SMOOTH,
        "num_fine": NUM_FINE,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "patience_counter": patience_counter,
    }, tmp)
    tmp.replace(path)


def load_last_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device | str,
) -> dict | None:
    """last.pt 를 모델·옵티마이저에 복원하고 재개 상태를 돌려준다.

    백본·label smoothing·세부 클래스 수가 지금 설정과 다르면 이어 받을 수 없으므로
    None (호출부는 처음부터 학습). 파일이 없어도 None.
    """
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mismatch = {
        k: (ckpt.get(k), v)
        for k, v in (("backbone", BACKBONE), ("label_smooth", LABEL_SMOOTH), ("num_fine", NUM_FINE))
        if ckpt.get(k) != v
    }
    if mismatch:
        log.warning(f"[{ARCH}] last.pt 설정 불일치 — 처음부터 학습: {mismatch}")
        return None
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return {
        "epoch": int(ckpt["epoch"]),
        "history": list(ckpt.get("history", [])),
        "best_score": float(ckpt.get("best_score", -1.0)),
        "best_epoch": int(ckpt.get("best_epoch", -1)),
        "patience_counter": int(ckpt.get("patience_counter", 0)),
    }


def train_hier() -> Path:
    config.ensure_directories()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    set_seed()
    device = pick_device()
    hp = get_hier_hyperparams()
    log.info(f"[{ARCH}] device={device}, fine={NUM_FINE}, coarse={NUM_COARSE}")

    items = build_hier_items()
    splits = load_or_build_hier_splits(items)
    train_items = [items[i] for i in splits["train"]]
    val_items = [items[i] for i in splits["val"]]
    log.info(f"[{ARCH}] items: train={len(train_items):,} val={len(val_items):,} "
          f"test={len(splits['test']):,}")

    sup_dist = Counter((it["sup_kind"], it["sup_slug"]) for it in train_items)
    log.info(f"[{ARCH}] 감독 분포(train): {dict(sup_dist.most_common(10))} ...")

    train_loader = DataLoader(
        HierImageDataset(train_items, augment=True),
        batch_size=hp.batch_size, shuffle=True,
        num_workers=6, persistent_workers=True, prefetch_factor=4,
    )
    val_loader = DataLoader(
        HierImageDataset(val_items, augment=False),
        batch_size=hp.batch_size, shuffle=False,
        num_workers=6, persistent_workers=True, prefetch_factor=4,
    )

    model = build_hier_model(NUM_FINE, BACKBONE).to(device)
    log.info(f"[{ARCH}] backbone={BACKBONE}, label_smooth={LABEL_SMOOTH}")
    log.info(f"[{ARCH}] trainable params: {count_parameters(model):,}")

    fine_w, coarse_w = compute_hier_weights(
        train_items, cap_multiplier=config.CNN_CLASS_WEIGHT_CAP,
    )
    criterion = HierarchicalLoss(fine_w, coarse_w).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hp.learning_rate, weight_decay=hp.weight_decay,
    )

    history: list[dict] = []
    best_score = -1.0      # 선택 기준: coarse_acc (대분류 견고성 우선) + fine_acc 보조
    best_epoch = -1
    patience_counter = 0
    best_ckpt_path = CKPT_DIR / "best.pt"
    last_ckpt_path = CKPT_DIR / LAST_CKPT_NAME

    start_epoch = 1
    if RESUME:
        state = load_last_checkpoint(last_ckpt_path, model, optimizer, device)
        if state is None:
            log.info(f"[{ARCH}] 재개 요청됐으나 이어 받을 last.pt 없음 — 처음부터")
        else:
            history = state["history"]
            best_score, best_epoch = state["best_score"], state["best_epoch"]
            patience_counter = state["patience_counter"]
            start_epoch = state["epoch"] + 1
            log.info(f"[{ARCH}] last.pt 에서 재개 — epoch {start_epoch} 부터 "
                     f"(best epoch={best_epoch}, score={best_score:.4f})")
            # 데이터 로더 셔플 RNG 는 복원하지 않는다 — epoch 순서가 바뀔 뿐 결과 동등성엔 무해.

    for epoch in range(start_epoch, hp.num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_facc, tr_cacc = _run_hier_epoch(
            model, train_loader, criterion, device, optimizer,
            desc=f"epoch {epoch} train",
        )
        val_loss, val_facc, val_cacc = _run_hier_epoch(
            model, val_loader, criterion, device, None,
            desc=f"epoch {epoch}  val ",
        )
        elapsed = time.time() - t0
        # 대분류 우선 + 세부 보조 (blueprint: "대분류는 절대 회귀 금지")
        score = val_cacc + 0.2 * val_facc

        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_fine_acc": tr_facc,
            "train_coarse_acc": tr_cacc, "val_loss": val_loss,
            "val_fine_acc": val_facc, "val_coarse_acc": val_cacc,
            "elapsed_sec": elapsed,
        })
        log.info(
            f"[{ARCH} epoch {epoch:3d}] "
            f"train loss={tr_loss:.4f} f_acc={tr_facc:.4f} c_acc={tr_cacc:.4f} | "
            f"val loss={val_loss:.4f} f_acc={val_facc:.4f} c_acc={val_cacc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "arch": ARCH,
                "backbone": BACKBONE,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_fine_acc": val_facc,
                "val_coarse_acc": val_cacc,
                "fine_labels": list(FINE_LABELS),
                "coarse_labels": list(COARSE_LABELS),
                "fine_to_coarse_idx": list(FINE_IDX_TO_COARSE_IDX),
            }, best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= hp.patience:
                log.info(f"[{ARCH}] early stopping at epoch {epoch}")
                break

        # 매 epoch 최근 상태 저장 — 중단돼도 GREENGUIDE_HIER_RESUME=1 로 여기서 이어간다
        save_last_checkpoint(
            last_ckpt_path, epoch=epoch, model=model, optimizer=optimizer, history=history,
            best_score=best_score, best_epoch=best_epoch, patience_counter=patience_counter,
        )

    # 정상 완료 — 다음 사이클이 옛 상태를 이어 받지 않도록 제거
    last_ckpt_path.unlink(missing_ok=True)

    log_path = LOG_DIR / "training_log.json"
    log_path.write_text(json.dumps({
        "arch": ARCH,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "hyperparams": asdict(hp),
        "fine_labels": list(FINE_LABELS),
        "coarse_labels": list(COARSE_LABELS),
        "history": history,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info(f"[{ARCH}] done. best epoch={best_epoch}")
    log.info(f"[{ARCH}] checkpoint → {best_ckpt_path}")
    return best_ckpt_path


if __name__ == "__main__":
    train_hier()
