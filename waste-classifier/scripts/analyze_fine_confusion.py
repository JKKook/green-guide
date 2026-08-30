#!/usr/bin/env python3
"""미달 세부품목(carton/paper_cup/glass_clear) 혼동 분석.

v2 체크포인트로 test set 을 재추론해 fine 혼동행렬을 만들고,
미달 품목의 오류가 어느 클래스로 새는지 + 조건(cond)별 오류율을 뽑는다.
→ 데이터 레버(재균형/병합/증강) 결정 근거.

실행: .venv/bin/python scripts/analyze_fine_confusion.py
출력: outputs/logs/cnn_hier/fine_confusion_report.json + 콘솔
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import _base  # noqa: F401 — sys.path 설정
import torch
from torch.utils.data import DataLoader
from waste_common.taxonomy import FINE_LABELS, NUM_FINE

from src import config
from src.hier_dataset import (
    HierImageDataset,
    build_hier_items,
    load_or_build_hier_splits,
)
from src.hier_train import CKPT_DIR, LOG_DIR
from src.model import WasteClassifierCNN
from src.train import pick_device

TARGETS = ("carton", "paper_cup", "glass_clear")


def main() -> None:
    device = pick_device()
    ckpt = torch.load(CKPT_DIR / "best.pt", map_location=device, weights_only=False)
    model = WasteClassifierCNN(num_classes=NUM_FINE).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    items = build_hier_items()
    splits = load_or_build_hier_splits(items)
    test_items = [items[i] for i in splits["test"]]
    # fine 감독 + 타깃 클래스 관련만 (타깃이 true 이거나, 예측이 타깃일 수 있는 전체)
    fine_items = [it for it in test_items if it["sup_kind"] == "fine"]
    print(f"fine test items: {len(fine_items):,}")

    loader = DataLoader(
        HierImageDataset(fine_items, augment=False),
        batch_size=config.CNN_BATCH_SIZE, shuffle=False,
        num_workers=6, persistent_workers=True, prefetch_factor=4,
    )

    preds: list[int] = []
    with torch.no_grad():
        for x, _, _ in loader:
            logits = model(x.to(device, non_blocking=True))
            preds.extend(logits.argmax(dim=1).cpu().tolist())

    # 혼동 집계 (true → pred), 조건별 오류
    conf: dict[str, Counter] = defaultdict(Counter)
    cond_err: dict[str, Counter] = defaultdict(Counter)   # target → cond 별 (err, total)
    cond_tot: dict[str, Counter] = defaultdict(Counter)
    for it, p in zip(fine_items, preds):
        t_slug = it["sup_slug"]
        p_slug = FINE_LABELS[p]
        conf[t_slug][p_slug] += 1
        if t_slug in TARGETS:
            # 파일명에서 조건 추출: aihub385_<cond>__... (fine-staging 파일 규약)
            name = Path(it["source_path"]).name
            cond = "unknown"
            if name.startswith("aihub385_") or name.startswith("aihub140_"):
                part = name.split("__")[0]
                cond = part.split("_", 1)[1] if "_" in part else "unknown"
            cond_tot[t_slug][cond] += 1
            if p_slug != t_slug:
                cond_err[t_slug][cond] += 1

    report = {}
    print("\n=== 미달 품목 혼동 (true → 어디로 새는가) ===")
    for t in TARGETS:
        row = conf[t]
        total = sum(row.values())
        correct = row[t]
        print(f"\n[{t}] recall {correct}/{total} = {correct / max(total,1):.3f}")
        leaks = [(k, v) for k, v in row.most_common() if k != t][:6]
        for k, v in leaks:
            print(f"    → {k:16} {v:>4} ({v / max(total,1):.1%})")
        print("  조건별 오류율:")
        for cond, tot in cond_tot[t].most_common():
            err = cond_err[t].get(cond, 0)
            print(f"    {cond:12} {err}/{tot} = {err / max(tot,1):.1%}")
        # 역방향: 무엇이 t 로 잘못 들어오나 (precision 저하 원인)
        incoming = Counter()
        for src, row2 in conf.items():
            if src != t and row2.get(t):
                incoming[src] = row2[t]
        print(f"  ← 역혼동 (남의 것이 {t} 로): {dict(incoming.most_common(4))}")
        report[t] = {
            "recall": correct / max(total, 1),
            "leaks": dict(leaks),
            "incoming": dict(incoming.most_common(6)),
            "cond_error": {c: [cond_err[t].get(c, 0), n] for c, n in cond_tot[t].items()},
        }

    out = LOG_DIR / "fine_confusion_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
