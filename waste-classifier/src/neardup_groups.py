"""근사중복 그룹 계산 — 프레임 상관 데이터의 그룹-인지 분할용.

배경: AI-Hub 71385 는 같은 물체의 연속 촬영 프레임이 많아 bbox 크롭 간
근사중복이 발생. 무작위 분할 시 train↔test 로 갈라져 frozen 지표가
과대평가됨 (2026-07-13 감사: 누수 3,617쌍 실측).

해법: pHash(8×8) 거리 ≤ NEARDUP_DIST 인 이미지들을 union-find 로 묶어
"그룹"을 만들고, 분할은 그룹을 원자 단위로 수행한다.

- 대상: fine-staging 아이템 (legacy 는 preprocessor cleanse 를 이미 통과)
- 캐시: data/splits/phash_cache.json — 신규 경로만 증분 계산
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from src import config

PHASH_CACHE = config.SPLITS_DIR / "phash_cache.json"
NEARDUP_DIST = 4
_LSH_BANDS = 4  # 64bit / 4 = 16bit 밴드 — 거리≤4 후보의 재현율 확보


def _load_cache() -> dict[str, str]:
    if PHASH_CACHE.exists():
        try:
            return json.loads(PHASH_CACHE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _compute_missing(paths: list[str], cache: dict[str, str]) -> dict[str, str]:
    """캐시에 없는 경로만 pHash 계산 (증분)."""
    import imagehash
    from PIL import Image

    missing = [p for p in paths if p not in cache]
    if not missing:
        return cache
    print(f"[neardup] pHash 증분 계산: {len(missing):,}장")
    for p in missing:
        try:
            with Image.open(config.PREPROCESSOR_ROOT / p) as im:
                cache[p] = str(imagehash.phash(im, hash_size=8))
        except Exception:  # noqa: BLE001
            cache[p] = ""  # 손상 — 그룹화 제외
    PHASH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PHASH_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return cache


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def compute_groups(items: list[dict[str, Any]]) -> dict[str, int]:
    """source_path → group_id. 그룹 = pHash 거리 ≤ NEARDUP_DIST 연결 성분.

    fine-staging 아이템만 그룹화 대상 (그 외는 자기 자신이 단독 그룹 —
    반환 dict 에 포함하지 않음; 호출측은 miss 를 단독 취급).
    """
    import imagehash

    targets = [it["source_path"] for it in items if "fine-staging" in it["source_path"]]
    if not targets:
        return {}
    cache = _compute_missing(targets, _load_cache())

    hashes: list[imagehash.ImageHash | None] = []
    for p in targets:
        hx = cache.get(p, "")
        hashes.append(imagehash.hex_to_hash(hx) if hx else None)

    uf = _UnionFind(len(targets))
    buckets: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(_LSH_BANDS)]
    for i, h in enumerate(hashes):
        if h is None:
            continue
        v = int(str(h), 16)
        for b in range(_LSH_BANDS):
            buckets[b][(v >> (16 * b)) & 0xFFFF].append(i)
    checked: set[tuple[int, int]] = set()
    for b in range(_LSH_BANDS):
        for bucket in buckets[b].values():
            if len(bucket) < 2 or len(bucket) > 300:
                continue
            for x in range(len(bucket)):
                for y in range(x + 1, len(bucket)):
                    i, j = bucket[x], bucket[y]
                    key = (i, j) if i < j else (j, i)
                    if key in checked:
                        continue
                    checked.add(key)
                    hi, hj = hashes[i], hashes[j]
                    if hi is not None and hj is not None and (hi - hj) <= NEARDUP_DIST:
                        uf.union(i, j)

    groups: dict[str, int] = {}
    n_multi = 0
    root_seen: dict[int, int] = {}
    for i, p in enumerate(targets):
        r = uf.find(i)
        gid = root_seen.setdefault(r, len(root_seen))
        groups[p] = gid
    sizes = defaultdict(int)
    for g in groups.values():
        sizes[g] += 1
    n_multi = sum(1 for s in sizes.values() if s > 1)
    print(f"[neardup] 그룹 {len(sizes):,}개 (다원소 그룹 {n_multi:,})")
    return groups
