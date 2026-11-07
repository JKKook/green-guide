#!/usr/bin/env python3
"""전처리 통계 리포트 — 사이클마다 학습 데이터의 구성·품질을 한 파일로 남긴다.

지표(데이터 품질 관리 "완전성·공정성" 상시 계측):
1. 클래스 폴더 × split 장수 (train/val/test) — 불균형 비율
2. 출처(source_key) × split 장수 — 도메인 구성
3. 격리 폴더(quarantine_*) 장수 — 학습에서 빠진 것
4. 표본 품질: 해상도 분위수, 흐림·어두움 비율(앱과 같은 임계), 출처별

실행: .venv/bin/python scripts/data_report.py [--sample 1500] [--seed 7]
출력: outputs/logs/data_report.json + 콘솔 요약
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import _base  # noqa: F401 — sys.path 설정
import numpy as np
from PIL import Image, UnidentifiedImageError

from greenguide_classifier import config
from greenguide_classifier.data_quality import image_quality, quality_issues, source_key
from greenguide_classifier.hier_dataset import build_hier_items, load_or_build_hier_splits

OUT_PATH = config.LOGS_DIR / "data_report.json"
RAW_DIR = config.PREPROCESSOR_ROOT / "data" / "raw"


def _class_folder(source_path: str) -> str:
    parts = Path(source_path).parts
    # data/raw/<set>/<class>/file — fine-staging 과 legacy 모두 뒤에서 두 번째
    return parts[-2] if len(parts) >= 2 else "?"


def _quarantine_counts() -> dict[str, int]:
    out: dict[str, int] = {}
    for q in sorted(RAW_DIR.glob("quarantine_*")):
        n = sum(1 for p in q.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
        out[q.name] = n
    return out


def _sample_quality(items: list[dict], train_idx: list[int], n: int, seed: int) -> dict:
    rng = random.Random(seed)
    picks = rng.sample(train_idx, min(n, len(train_idx)))
    widths, heights = [], []
    per_src: dict[str, Counter] = defaultdict(Counter)
    for i in picks:
        it = items[i]
        key = source_key(it["source_path"])
        try:
            with Image.open(config.PREPROCESSOR_ROOT / it["source_path"]) as im:
                im.load()
                w, h = im.size
                b, s = image_quality(im)
        except (UnidentifiedImageError, OSError):
            per_src[key]["unreadable"] += 1
            continue
        widths.append(w)
        heights.append(h)
        per_src[key]["n"] += 1
        for issue in quality_issues(b, s):
            per_src[key][issue] += 1
    q = lambda a, p: int(np.percentile(a, p)) if a else None  # noqa: E731
    return {
        "sample": len(picks),
        "resolution": {
            "width_p10_p50_p90": [q(widths, 10), q(widths, 50), q(widths, 90)],
            "height_p10_p50_p90": [q(heights, 10), q(heights, 50), q(heights, 90)],
        },
        "by_source": {
            k: {
                "n": c["n"],
                "too_dark_rate": round(c["too_dark"] / max(c["n"], 1), 4),
                "too_blurry_rate": round(c["too_blurry"] / max(c["n"], 1), 4),
                "unreadable": c["unreadable"],
            }
            for k, c in sorted(per_src.items(), key=lambda kv: -kv[1]["n"])
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=1500, help="품질·해상도 표본 장수 (train)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    items = build_hier_items()
    splits = load_or_build_hier_splits(items)

    by_class: dict[str, Counter] = defaultdict(Counter)
    by_source: dict[str, Counter] = defaultdict(Counter)
    for split, idx in splits.items():
        for i in idx:
            it = items[i]
            by_class[_class_folder(it["source_path"])][split] += 1
            by_source[source_key(it["source_path"])][split] += 1
    train_counts = {k: c["train"] for k, c in by_class.items() if c["train"]}
    imbalance = (max(train_counts.values()) / max(min(train_counts.values()), 1)) if train_counts else None

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "items_total": len(items),
        "splits": {k: len(v) for k, v in splits.items()},
        "class_imbalance_ratio_train": round(imbalance, 1) if imbalance else None,
        "by_class": {k: dict(c) for k, c in sorted(by_class.items(), key=lambda kv: -kv[1]["train"])},
        "by_source": {k: dict(c) for k, c in sorted(by_source.items(), key=lambda kv: -kv[1]["train"])},
        "quarantine": _quarantine_counts(),
        "quality": _sample_quality(items, splits["train"], args.sample, args.seed),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[report] items {len(items):,} · splits {report['splits']} · 불균형(train max/min) {report['class_imbalance_ratio_train']}x")
    print("[report] 출처별 train:", ", ".join(f"{k}={c['train']:,}" for k, c in report["by_source"].items()))
    print("[report] 격리:", report["quarantine"] or "없음")
    qs = report["quality"]
    print(f"[report] 표본 {qs['sample']}장 해상도 w p50={qs['resolution']['width_p10_p50_p90'][1]} "
          f"h p50={qs['resolution']['height_p10_p50_p90'][1]}")
    for k, v in qs["by_source"].items():
        print(f"[report]   {k:16s} n={v['n']:4d} 어두움 {v['too_dark_rate']:.1%} 흐림 {v['too_blurry_rate']:.1%}")
    print(f"→ {OUT_PATH}")


if __name__ == "__main__":
    main()
