#!/usr/bin/env python3
"""클래스 로짓 bias 재보정 근거 — 검증셋(val)에서 bias 별 클래스 recall 변화를 잰다.

서빙 플래그 GREENGUIDE_API_ELEC_LOGIT_BIAS(대상 fine 로짓에서 b 를 뺌)의 부작용
(진짜 electronics recall 손실·다른 대분류 정확도 변화)을 학습 분포 쪽에서 측정한다.
실사용 효과는 realworld_eval_golden.py 로 따로 잰다.

표본: val 중 대상 클래스 전수(상한 --max-target) + 다른 대분류별 --per-class 장.
서빙과 같이 non_object 로짓은 마스킹(Stage1 통과 가정), 회전 TTA 는 없음(0°).

실행: arch -arm64 .venv/bin/python scripts/calib_class_bias.py --target electronics
출력: outputs/logs/calib_class_bias/<target>.json (+ 로짓 캐시 .npz)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import _base  # noqa: F401 — sys.path 설정
import numpy as np
import onnxruntime as ort
from greenguide_common import imaging

from greenguide_classifier import config
from greenguide_classifier.dataset import _load_rgb_chw01
from greenguide_classifier.hier_dataset import build_hier_items, load_or_build_hier_splits

ONNX_PATH = config.MODELS_DIR / "cnn_hier" / "classifier.onnx"
OUT_DIR = config.OUTPUTS_DIR / "logs" / "calib_class_bias"


def sample_val(items: list[dict], val_idx: list[int], f2c: dict[str, str], target: str,
               per_class: int, max_target: int, seed: int) -> list[dict]:
    by_coarse: dict[str, list[dict]] = defaultdict(list)
    for i in val_idx:
        it = items[i]
        c = f2c.get(it["sup_slug"], it["sup_slug"]) if it["sup_kind"] == "fine" else it["sup_slug"]
        by_coarse[c].append({**it, "coarse": c})
    rng = random.Random(seed)
    out = []
    for c, lst in sorted(by_coarse.items()):
        rng.shuffle(lst)
        out.extend(lst[:max_target if c == target else per_class])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="electronics")
    ap.add_argument("--onnx", type=Path, default=ONNX_PATH)
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--max-target", type=int, default=600)
    ap.add_argument("--biases", default="0,0.5,1,1.5,2,2.5,3,4,5")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tax = json.loads((args.onnx.parent / "taxonomy.json").read_text(encoding="utf-8"))
    fine, coarse_labels = tax["fine_labels"], tax["coarse_labels"]
    f2c_idx = np.array(tax["fine_idx_to_coarse_idx"])
    f2c = tax["fine_to_coarse"]
    ti = fine.index(args.target)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / f"{args.target}_logits.npz"
    items = build_hier_items()
    splits = load_or_build_hier_splits(items)
    sample = sample_val(items, splits["val"], f2c, args.target,
                        args.per_class, args.max_target, args.seed)
    if cache.exists() and len(np.load(cache)["logits"]) == len(sample):
        logits = np.load(cache)["logits"]
    else:
        sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        rows = []
        for k, it in enumerate(sample):
            x = (_load_rgb_chw01(it).numpy() - imaging.MEAN_CHW) / imaging.STD_CHW
            (lg,) = sess.run(["logits"], {"image": x[None].astype(np.float32)})
            rows.append(lg[0])
            if k % 200 == 0:
                print(f"{k}/{len(sample)}", flush=True)
        logits = np.stack(rows)
        np.savez_compressed(cache, logits=logits)

    truth = np.array([coarse_labels.index(it["coarse"]) for it in sample])
    tgt_c = coarse_labels.index(f2c[args.target])
    no = fine.index("non_object") if "non_object" in fine else None
    results = []
    for b in [float(v) for v in args.biases.split(",")]:
        lg = logits.copy()
        lg[:, ti] -= b
        if no is not None:
            lg[:, no] = -1e9
        p = np.exp(lg - lg.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        cp = np.zeros((len(p), len(coarse_labels)))
        for fi, ci in enumerate(f2c_idx):
            cp[:, ci] += p[:, fi]
        pred = cp.argmax(1)
        recall = {}
        for ci, c in enumerate(coarse_labels):
            m = truth == ci
            if m.any():
                recall[c] = round(float((pred[m] == ci).mean()), 4)
        others = truth != tgt_c
        results.append({
            "bias": b,
            "target_recall": recall.get(f2c[args.target]),
            "others_acc": round(float((pred[others] == truth[others]).mean()), 4),
            "others_pred_as_target": int((pred[others] == tgt_c).sum()),
            "macro_recall": round(float(np.mean(list(recall.values()))), 4),
            "recall": recall,
        })
        print(f"b={b:<4} {args.target} recall {results[-1]['target_recall']:.3f} · "
              f"기타 정확도 {results[-1]['others_acc']:.3f} · 기타→{args.target} "
              f"{results[-1]['others_pred_as_target']} · macro {results[-1]['macro_recall']:.3f}")
    out = OUT_DIR / f"{args.target}.json"
    out.write_text(json.dumps({
        "onnx": str(args.onnx), "n": len(sample),
        "n_target": int((truth == tgt_c).sum()), "seed": args.seed,
        "per_class": args.per_class, "results": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
