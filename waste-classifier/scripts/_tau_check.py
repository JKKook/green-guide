#!/usr/bin/env python3
"""OOD τ 선택 근거 — val 거리 퍼센타일 + 후보 τ별 오거부율 (일회성 분석)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort

from scripts.build_hier_prototypes import ONNX_PATH, OOD_PATH, _extract_embeddings
from src.hier_dataset import build_hier_items, load_or_build_hier_splits


def main() -> None:
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    P = np.load(OOD_PATH, allow_pickle=False)["prototypes"]

    items = build_hier_items()
    splits = load_or_build_hier_splits(items)
    rng = np.random.default_rng(43)  # 보정과 다른 시드
    val_items = [items[i] for i in splits["val"] if items[i]["sup_kind"] == "fine"]
    sample = [val_items[i] for i in rng.permutation(len(val_items))[:1500]]
    embs = _extract_embeddings(sess, sample)
    dist = 1.0 - (embs @ P.T).max(axis=1)

    print("val in-distribution 거리 퍼센타일:")
    for p in (50, 80, 90, 95, 97.5, 99):
        print(f"  p{p:<5} {np.percentile(dist, p):.4f}")
    for tau in (0.26, 0.28, 0.30, 0.32, 0.35):
        print(f"  τ={tau:.2f} → val 오거부율 {(dist > tau).mean():.2%}")


if __name__ == "__main__":
    main()
