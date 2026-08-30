"""고정 held-out test set — source_path 기준으로 동결.

문제: 기존 split 은 manifest 의 **위치 인덱스**(`range(n)`)로 나눠서, 데이터가
추가되면 같은 인덱스가 다른 이미지를 가리킨다. seed 가 있어도 버전 간 test
accuracy 가 비교 불가이고, 이전 train 이미지가 다음 버전 test 로 새는 누수도 생긴다.

해결: test 멤버를 안정 키(`source_path`)로 **동결**한다. 한 번 test 에 들어간
이미지는 계속 test, 신규 데이터는 train/val 로만 들어간다. 신규 클래스는 첫
등장 시 일부를 test 에 시드한다. → 버전 간 숫자가 같은 잣대로 비교된다.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split
from greenguide_common.logging import get_logger

from greenguide_classifier import config

log = get_logger(__name__)

FROZEN_PATH: Path = config.SPLITS_DIR / "frozen_test.json"

_TEST_FRACTION: float = config.SPLIT_RATIOS["test"]   # 0.15
_MIN_TEST_PER_CLASS: int = 20   # 클래스당 최소 test 장수 (데이터 적으면 가능한 만큼)


def _key(item: dict[str, Any]) -> str:
    """안정 식별 키 — source_path (재빌드돼도 동일 이미지는 같은 경로)."""
    return item["source_path"]


def load_frozen_keys() -> set[str]:
    if FROZEN_PATH.exists():
        data = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
        return set(data.get("keys", []))
    return set()


def ensure_frozen_test(
    items: list[dict[str, Any]], seed: int = config.SPLIT_SEED,
) -> set[str]:
    """frozen test 키 집합 보장 — 사라진 키 제거, 신규/부족 클래스 보충 후 저장.

    불변식:
      - 이미 frozen 인 (그리고 아직 존재하는) 이미지는 계속 test 에 남는다.
      - 클래스별 frozen test 수가 목표치 미만이면 비-frozen 에서만 추가한다.
        (기존 test 멤버를 빼지 않으므로 기존 클래스 비교성은 보존)
    """
    rng = random.Random(seed)
    frozen = load_frozen_keys()
    present = {_key(it) for it in items}

    # 1) manifest 에서 사라진 키 제거 (파일 삭제/이동)
    frozen &= present

    # 2) 클래스별 그룹 + 현재 frozen 수
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        by_label[it["label"]].append(it)
    frozen_count: Counter[str] = Counter(
        it["label"] for it in items if _key(it) in frozen
    )

    # 3) 목표치 미달 클래스는 비-frozen 에서 보충
    for label, group in by_label.items():
        # 15% 를 기본으로, 작은 클래스는 최소치까지 끌어올리되 절반은 학습용으로 보존
        # (예: etc 10장 → 전부 test 로 가져가면 학습 0장이 되는 문제 방지)
        half = len(group) // 2
        target = round(len(group) * _TEST_FRACTION)
        target = min(max(target, _MIN_TEST_PER_CLASS), half)
        need = target - frozen_count[label]
        if need <= 0:
            continue
        candidates = [it for it in group if _key(it) not in frozen]
        rng.shuffle(candidates)
        for it in candidates[:need]:
            frozen.add(_key(it))

    _save(frozen, items)
    return frozen


def _save(frozen: set[str], items: list[dict[str, Any]]) -> None:
    FROZEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    per_class = Counter(it["label"] for it in items if _key(it) in frozen)
    FROZEN_PATH.write_text(
        json.dumps(
            {
                "keys": sorted(frozen),
                "per_class": dict(sorted(per_class.items())),
                "total": len(frozen),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info(f"{len(frozen):,}장 동결 → {FROZEN_PATH.name} "
          f"({len(per_class)} 클래스)")


def build_splits(
    items: list[dict[str, Any]], seed: int = config.SPLIT_SEED,
) -> dict[str, list[int]]:
    """frozen test 기준 index split 반환 (기존 splits.json 과 동일 shape).

    test = frozen 멤버, 나머지 = stratified train/val.
    """
    frozen = ensure_frozen_test(items, seed)

    test_idx: list[int] = []
    rest_idx: list[int] = []
    rest_labels: list[str] = []
    for i, it in enumerate(items):
        if _key(it) in frozen:
            test_idx.append(i)
        else:
            rest_idx.append(i)
            rest_labels.append(it["label"])

    # 나머지 → train/val stratified
    val_frac = config.SPLIT_RATIOS["val"] / (
        config.SPLIT_RATIOS["train"] + config.SPLIT_RATIOS["val"]
    )
    counts = Counter(rest_labels)
    stratify = (
        rest_labels
        if len(counts) > 1 and all(v >= 2 for v in counts.values())
        else None
    )
    train_idx, val_idx = train_test_split(
        rest_idx, test_size=val_frac, stratify=stratify, random_state=seed,
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}
