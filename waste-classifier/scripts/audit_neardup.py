#!/usr/bin/env python3
"""fine-staging 근사중복 감사 — 클래스 내 중복률 + train↔frozen test 누수 측정.

pHash(8×8, 64bit) 기준:
- 거리 0  = 사실상 동일 이미지 (강한 중복)
- 거리 ≤4 = 근사중복 (연속 프레임/미세 변화) — 4-밴드 LSH 로 후보 축소 후 검증
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import _base  # noqa: F401 — sys.path 설정
import imagehash
from PIL import Image

from src import config
from src.hier_dataset import build_hier_items, load_or_build_hier_splits


def phash(p: Path):
    try:
        with Image.open(p) as im:
            return imagehash.phash(im, hash_size=8)
    except Exception:
        return None

def main() -> None:
    items = build_hier_items()
    splits = load_or_build_hier_splits(items)
    train_i, test_i = set(splits["train"]), set(splits["test"])

    # 대상: fine-staging 아이템 전체 + (누수 검사용) frozen test 전체
    hashes: dict[int, imagehash.ImageHash] = {}
    for i, it in enumerate(items):
        sp = it["source_path"]
        if ("fine-staging" in sp) or (i in test_i):
            h = phash(config.PREPROCESSOR_ROOT / sp)
            if h is not None:
                hashes[i] = h
    print(f"해시 계산: {len(hashes):,}장")

    # ── 1) 클래스 내 강한 중복(거리 0) ──
    by_class_exact: dict[str, Counter] = defaultdict(Counter)
    for i, h in hashes.items():
        if "fine-staging" in items[i]["source_path"]:
            by_class_exact[items[i]["sup_slug"]][str(h)] += 1
    print("\n=== fine-staging 클래스 내 '동일 pHash' 중복 (거리 0) ===")
    total_files = total_dups = 0
    for slug, cnt in sorted(by_class_exact.items()):
        n = sum(cnt.values()); dups = n - len(cnt)
        total_files += n; total_dups += dups
        if dups:
            print(f"  {slug:18} {n:>7,}장 중 중복 {dups:>6,} ({dups/n:.1%})")
    print(f"  {'합계':18} {total_files:>7,}장 중 중복 {total_dups:>6,} ({total_dups/max(total_files,1):.1%})")

    # ── 2) 근사중복(거리 ≤4) — 4밴드 LSH 후보 → 검증 ──
    band_buckets: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(4)]
    for i, h in hashes.items():
        v = int(str(h), 16)
        for b in range(4):
            band_buckets[b][(v >> (16 * b)) & 0xFFFF].append(i)
    near_pairs_within = 0
    leak_pairs = []          # (train_i, test_i, dist)
    seen = set()
    for b in range(4):
        for bucket in band_buckets[b].values():
            if len(bucket) < 2 or len(bucket) > 200:
                continue
            for a in range(len(bucket)):
                for c in range(a + 1, len(bucket)):
                    i, j = bucket[a], bucket[c]
                    if (i, j) in seen: continue
                    seen.add((i, j))
                    d = hashes[i] - hashes[j]
                    if d <= 4:
                        ti, tj = i in train_i, j in train_i
                        si, sj = i in test_i, j in test_i
                        if (ti and sj) or (tj and si):
                            leak_pairs.append((i, j, d))
                        elif ti and tj:
                            near_pairs_within += 1
    print("\n=== 근사중복 (거리≤4) ===")
    print(f"  train 내부 근사중복 쌍: {near_pairs_within:,}")
    print(f"  ★ train↔frozen-test 누수 쌍: {len(leak_pairs):,}")
    leak_by_class = Counter(items[i]['sup_slug'] for i, j, d in leak_pairs)
    for slug, n in leak_by_class.most_common(10):
        print(f"      {slug:18} {n:,}")
    print("AUDIT_DONE")

if __name__ == "__main__":
    main()
