#!/usr/bin/env python3
"""구 6클래스 라벨의 '세부 오염' 교차감사.

가설: 구 paper/vinyl/trash 를 fine 감독으로 승격한 것이 신규 세부 클래스
(carton/paper_cup/vinyl_dirty/light_bulb)와 감독 충돌을 일으킨다.

방법: v4 모델로 해당 legacy train 아이템을 추론 → 충돌 후보 클래스로
고확신(≥0.7) 예측되는 비율 = 오염률 추정. (모델 자기참조 편향이 있으므로
하한 추정치로 해석 — 충돌 라벨로 '학습됐음에도' 반대로 예측할 정도면 강한 신호)
"""
from __future__ import annotations

from collections import Counter

import _base  # noqa: F401 — sys.path 설정
import numpy as np
import onnxruntime as ort
from greenguide_common.taxonomy import FINE_LABELS
from torch.utils.data import DataLoader

from greenguide_classifier import config
from greenguide_classifier.hier_dataset import (
    HierImageDataset,
    build_hier_items,
    load_or_build_hier_splits,
)

TARGETS = {
    "paper_other": ["carton", "paper_cup", "cardboard"],
    "vinyl_clean": ["vinyl_dirty"],
    "trash_other": ["light_bulb", "battery"],
}
CONF = 0.70

def main() -> None:
    sess = ort.InferenceSession(
        str(config.MODELS_DIR / "cnn_hier" / "classifier.onnx"),
        providers=["CPUExecutionProvider"])
    items = build_hier_items()
    splits = load_or_build_hier_splits(items)
    train = set(splits["train"])

    for src_slug, suspects in TARGETS.items():
        idxs = [i for i in splits["train"]
                if items[i]["sup_kind"] == "fine" and items[i]["sup_slug"] == src_slug
                and "fine-staging" not in items[i]["source_path"]][:4000]  # legacy 만
        subset = [items[i] for i in idxs]
        if not subset:
            print(f"{src_slug}: legacy 표본 없음"); continue
        loader = DataLoader(HierImageDataset(subset, augment=False),
                            batch_size=64, num_workers=4)
        hits = Counter(); n = 0
        for x, _, _ in loader:
            (lg,) = sess.run(["logits"], {"image": x.numpy()})
            e = np.exp(lg - lg.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
            top = p.argmax(axis=1); conf = p.max(axis=1)
            for t, c in zip(top, conf):
                n += 1
                slug = FINE_LABELS[int(t)]
                if slug in suspects and c >= CONF:
                    hits[slug] += 1
        total_hits = sum(hits.values())
        print(f"\n[{src_slug}] legacy {n:,}장 중 충돌클래스 고확신 예측: "
              f"{total_hits:,} ({total_hits/max(n,1):.1%})")
        for s, v in hits.most_common():
            print(f"    → {s:14} {v:,}")
    print("CONTAM_AUDIT_DONE")

if __name__ == "__main__":
    main()
