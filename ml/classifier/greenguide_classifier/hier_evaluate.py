"""계층 모델 레벨별 평가 — 대분류/세부 정확도를 분리 산출.

blueprint §7 측정 원칙:
- 대분류 정확도: 전체 test (fine 라벨은 롤업해서 coarse 정답으로)
- 세부 f1: fine-감독 test 아이템만 (coarse-감독 아이템은 세부 정답이 없음)
- fine 클래스별 f1 < 활성화 임계(0.80) 는 "롤업 유지" 신호

실행: .venv/bin/python -m greenguide_classifier.hier_evaluate
출력: outputs/logs/cnn_hier/evaluation.json
"""
from __future__ import annotations

import json

import torch
from greenguide_common.logging import get_logger
from greenguide_common.taxonomy import (
    COARSE_LABELS,
    FINE_IDX_TO_COARSE_IDX,
    FINE_LABELS,
    NUM_FINE,
    same_guidance,
)
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from greenguide_classifier import config
from greenguide_classifier.hier_dataset import (
    HierImageDataset,
    build_hier_items,
    load_or_build_hier_splits,
)
from greenguide_classifier.hier_train import ARCH, CKPT_DIR, LOG_DIR
from greenguide_classifier.model import build_hier_model
from greenguide_classifier.train import pick_device

log = get_logger(__name__)

FINE_ACTIVATION_F1 = 0.80          # blueprint §5: 세부품목 활성화 임계
GUIDANCE_SAFE_ACTIVATION_F1 = 0.85  # 안내-동일 형제 혼동을 정답 처리한 보조 임계
# (같은 수거함 안내를 공유하는 형제로의 혼동은 사용자 피해가 없음 — 혼동분석 근거.
#  완화된 지표인 만큼 임계는 더 엄격하게.)


def _guidance_safe_f1(fine_true: list[int], fine_pred: list[int], slug: str) -> float:
    """안내-동일 혼동을 정답으로 취급한 f1 (특정 slug 기준)."""
    idx = FINE_LABELS.index(slug)
    tp = fp = fn = 0
    for t, p in zip(fine_true, fine_pred, strict=False):
        t_ok = t == idx
        # 예측/정답이 slug 와 '안내상 동일' 하면 매치로 간주
        p_match = same_guidance(FINE_LABELS[p], slug)
        t_match = same_guidance(FINE_LABELS[t], slug)
        if t_ok:
            if p_match:
                tp += 1
            else:
                fn += 1
        elif p == idx and not t_match:
            fp += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def evaluate_hier() -> dict:
    device = pick_device()
    ckpt_path = CKPT_DIR / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 없음: {ckpt_path} — greenguide_classifier.hier_train 먼저 실행")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_hier_model(NUM_FINE, ckpt.get("backbone", "resnet18")).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    items = build_hier_items()
    splits = load_or_build_hier_splits(items)
    test_items = [items[i] for i in splits["test"]]
    log.info(f"[{ARCH}] test items: {len(test_items):,}")

    loader = DataLoader(
        HierImageDataset(test_items, augment=False),
        batch_size=config.CNN_BATCH_SIZE, shuffle=False,
        num_workers=6, persistent_workers=True, prefetch_factor=4,
    )

    f2c = torch.tensor(FINE_IDX_TO_COARSE_IDX, dtype=torch.long, device=device)
    all_pred_fine: list[int] = []
    all_pred_coarse: list[int] = []
    all_is_fine: list[int] = []
    all_sup: list[int] = []

    with torch.no_grad():
        for x, is_fine, sup_idx in loader:
            logits = model(x.to(device, non_blocking=True))
            pf = logits.argmax(dim=1)
            all_pred_fine.extend(pf.cpu().tolist())
            all_pred_coarse.extend(f2c[pf].cpu().tolist())
            all_is_fine.extend(is_fine.tolist())
            all_sup.extend(sup_idx.tolist())

    # ── 대분류 정확도 (전체) ─────────────────────────────────────────────
    true_coarse = [
        FINE_IDX_TO_COARSE_IDX[s] if f else s
        for f, s in zip(all_is_fine, all_sup, strict=False)
    ]
    coarse_report = classification_report(
        true_coarse, all_pred_coarse,
        labels=list(range(len(COARSE_LABELS))),
        target_names=list(COARSE_LABELS), output_dict=True, zero_division=0,
    )
    coarse_acc = coarse_report["accuracy"]

    # ── 세부 (fine-감독 아이템만) ────────────────────────────────────────
    fine_true = [s for f, s in zip(all_is_fine, all_sup, strict=False) if f]
    fine_pred = [p for f, p in zip(all_is_fine, all_pred_fine, strict=False) if f]
    fine_report = classification_report(
        fine_true, fine_pred,
        labels=list(range(len(FINE_LABELS))),
        target_names=list(FINE_LABELS), output_dict=True, zero_division=0,
    )
    fine_acc = fine_report["accuracy"] if fine_true else 0.0

    # 활성화 판정: (f1 ≥ 0.80 OR guidance_safe_f1 ≥ 0.85) + test 표본 ≥ 30
    activation = {}
    for name in FINE_LABELS:
        r = fine_report.get(name, {})
        support = int(r.get("support", 0))
        f1 = float(r.get("f1-score", 0.0))
        gs_f1 = _guidance_safe_f1(fine_true, fine_pred, name)
        activation[name] = {
            "f1": round(f1, 4),
            "guidance_safe_f1": round(gs_f1, 4),
            "support": support,
            "ready": bool(
                support >= 30
                and (f1 >= FINE_ACTIVATION_F1 or gs_f1 >= GUIDANCE_SAFE_ACTIVATION_F1)
            ),
        }

    cm = confusion_matrix(
        true_coarse, all_pred_coarse, labels=list(range(len(COARSE_LABELS))),
    ).tolist()

    result = {
        "arch": ARCH,
        "test_size": len(test_items),
        "coarse_accuracy": round(coarse_acc, 4),
        "fine_accuracy_on_fine_items": round(fine_acc, 4),
        "fine_items_in_test": len(fine_true),
        "coarse_report": coarse_report,
        "fine_report": fine_report,
        "fine_activation": activation,
        "coarse_confusion_matrix": cm,
        "coarse_labels": list(COARSE_LABELS),
        "fine_labels": list(FINE_LABELS),
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / "evaluation.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info(f"[{ARCH}] 대분류 정확도: {coarse_acc:.4f}")
    log.info(f"[{ARCH}] 세부 정확도(fine 아이템 {len(fine_true):,}건): {fine_acc:.4f}")
    ready = [k for k, v in activation.items() if v["ready"]]
    log.info(f"[{ARCH}] 활성화 준비된 세부품목({len(ready)}): {ready}")
    log.info(f"[{ARCH}] → {out_path}")
    return result


if __name__ == "__main__":
    evaluate_hier()
