"""계층 학습용 Dataset — manifest(legacy) + fine-staging(세부) 혼합.

각 item 은 supervision (kind, idx) 를 갖는다:
- kind="fine":   fine 공간 CE loss
- kind="coarse": 롤업 P(coarse)=Σ P(children) 에 대한 NLL loss

split 은 hier_splits.json 으로 저장 (경로 안정키 기반 test 동결 —
frozen_test.py 의 원칙을 계층 데이터에 적용).
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src import config
from src.dataset import WasteImageDataset, _load_rgb_chw01, load_manifest
from src.taxonomy import (
    COARSE_TO_INDEX,
    FINE_TO_INDEX,
    LEGACY_LABEL_SUPERVISION,
    STAGING_DIR_SUPERVISION,
    supervision_index,
)

FINE_STAGING_DIR: Path = config.PREPROCESSOR_ROOT / "data" / "raw" / "fine-staging"
HIER_SPLITS_PATH: Path = config.SPLITS_DIR / "hier_splits.json"
HIER_FROZEN_PATH: Path = config.SPLITS_DIR / "hier_frozen_test.json"

# legacy frozen test (기존 flat 파이프라인의 동결 test) — 누수 방지 위해 그대로 계승
LEGACY_FROZEN_PATH: Path = config.SPLITS_DIR / "frozen_test.json"

SPLIT_RATIOS = config.SPLIT_RATIOS
SEED = config.SPLIT_SEED
MIN_FROZEN_PER_CLASS = 30   # 신규 fine 클래스 frozen test 최소 확보량


def build_hier_items() -> list[dict[str, Any]]:
    """전체 계층 학습 아이템 목록 (결정적 순서).

    item: {source_path(preprocessor 루트 상대), sup_kind, sup_slug, sup_idx, origin}
    """
    items: list[dict[str, Any]] = []

    # 1) legacy manifest — 라벨별 감독 매핑
    for it in load_manifest():
        # 사용자 피드백(user_*)은 학습에서 영구 제외 — 순수 실사용 평가셋 예약.
        # (v11 진단에서 피드백 51장 중 31장이 train 에 흡수돼 실사용 지표가
        #  부풀려졌던 오염 발견 → 평가 무결성 원칙: 피드백 = 홀드아웃 전용)
        if Path(it["source_path"]).name.startswith("user_"):
            continue
        label = it["label"]
        sup = LEGACY_LABEL_SUPERVISION.get(label)
        if sup is None:
            # 신규 피드백은 fine slug 로 정정될 수 있음 (예: raw/battery/user_x.jpg)
            # → taxonomy 의 fine/coarse 공간에 있으면 직접 감독으로 수용
            if label in FINE_TO_INDEX:
                sup = ("fine", label)
            elif label in COARSE_TO_INDEX:
                sup = ("coarse", label)
            else:
                # etc_auto_* pseudo-class 등 미정의 라벨은 계층에서 제외
                continue
        kind, slug = sup
        items.append({
            "source_path": it["source_path"],
            "sup_kind": kind,
            "sup_slug": slug,
            "sup_idx": supervision_index(kind, slug),
            "origin": "legacy",
        })

    # 2) fine-staging — 디렉터리명 = 감독 매핑
    if FINE_STAGING_DIR.exists():
        for d in sorted(FINE_STAGING_DIR.iterdir()):
            if not d.is_dir():
                continue
            sup = STAGING_DIR_SUPERVISION.get(d.name)
            if sup is None:
                print(f"[hier_dataset] 미정의 staging 라벨 스킵: {d.name}")
                continue
            kind, slug = sup
            idx = supervision_index(kind, slug)
            rel_base = d.relative_to(config.PREPROCESSOR_ROOT)
            for f in sorted(d.glob("*.jpg")):
                items.append({
                    "source_path": str(rel_base / f.name),
                    "sup_kind": kind,
                    "sup_slug": slug,
                    "sup_idx": idx,
                    "origin": "staging",
                })

    # 경로 기준 결정적 정렬 (splits 인덱스 안정성)
    items.sort(key=lambda x: x["source_path"])
    return items


def _load_frozen_paths() -> set[str]:
    """legacy frozen test 의 source_path 집합 (없으면 빈 집합)."""
    if not LEGACY_FROZEN_PATH.exists():
        return set()
    try:
        data = json.loads(LEGACY_FROZEN_PATH.read_text(encoding="utf-8"))
        # frozen_test.json 포맷: {"keys": [source_path...], "per_class": ..., "total": ...}
        if isinstance(data, dict) and "keys" in data:
            return set(data["keys"])
        if isinstance(data, list):
            return set(data)
    except Exception as exc:  # noqa: BLE001
        print(f"[hier_dataset] legacy frozen 로드 실패: {exc}")
    return set()


def build_hier_splits(items: list[dict[str, Any]]) -> dict[str, list[int]]:
    """계층 splits 생성 (test 동결 원칙 계승).

    - legacy frozen test 멤버 → 무조건 test (기존 누수 방지 규율 유지)
    - hier_frozen_test.json 에 이미 동결된 경로 → test
    - 신규 fine 클래스는 클래스당 최소 MIN_FROZEN_PER_CLASS 를 test 로 동결
      (단, 클래스 크기의 절반 상한 — 소수 클래스 학습량 보존)
    - 나머지: supervision(kind, slug) 기준 stratified train/val
    """
    rng = random.Random(SEED)
    legacy_frozen = _load_frozen_paths()
    hier_frozen: set[str] = set()
    if HIER_FROZEN_PATH.exists():
        hier_frozen = set(json.loads(HIER_FROZEN_PATH.read_text(encoding="utf-8"))["paths"])

    # ── 근사중복 그룹 (프레임 상관) — 그룹은 분할의 원자 단위 ──────────────
    # 연속 촬영 크롭이 train/test 로 갈라지는 누수 차단 (2026-07-13 감사:
    # 누수 3,617쌍). frozen 멤버가 속한 그룹은 통째로 test 로 끌려간다.
    from src.neardup_groups import compute_groups
    path_group = compute_groups(items)          # fine-staging 만; miss=단독
    group_members: dict[int, list[int]] = {}
    for i, it in enumerate(items):
        g = path_group.get(it["source_path"])
        if g is not None:
            group_members.setdefault(g, []).append(i)

    frozen_all = legacy_frozen | hier_frozen
    # frozen 멤버를 포함한 그룹 전체 → test 강제 (train 누수 차단)
    group_forced_test: set[int] = set()
    for g, members in group_members.items():
        if any(items[i]["source_path"] in frozen_all for i in members):
            group_forced_test.update(members)

    def _is_synthetic(i: int) -> bool:
        return Path(items[i]["source_path"]).name.startswith("synmo_")

    test_idx: list[int] = []
    dropped_synth = 0
    pool_by_class: dict[tuple[str, str], list[int]] = {}
    for i, it in enumerate(items):
        if (it["source_path"] in frozen_all) or (i in group_forced_test):
            # 합성은 test 금지 — 그룹이 test 로 끌려가면 합성 멤버는 이번
            # 빌드에서 미사용 (train 에 남기면 근사중복 경유 누수)
            if _is_synthetic(i):
                dropped_synth += 1
            else:
                test_idx.append(i)
        else:
            pool_by_class.setdefault((it["sup_kind"], it["sup_slug"]), []).append(i)
    if dropped_synth:
        print(f"[hier_dataset] test-그룹 소속 합성 {dropped_synth}장 미사용 처리")

    # ── 그룹 단위 헬퍼: 클래스 pool 을 (그룹 → 멤버들) 리스트로 ──────────────
    def _group_units(pool: list[int]) -> list[list[int]]:
        by_g: dict[int, list[int]] = {}
        singles: list[list[int]] = []
        for i in pool:
            g = path_group.get(items[i]["source_path"])
            if g is None:
                singles.append([i])
            else:
                by_g.setdefault(g, []).append(i)
        return list(by_g.values()) + singles

    # 신규 클래스 frozen 충원 — **그룹 원자 단위** + 합성 test 금지
    test_class_count = Counter((items[i]["sup_kind"], items[i]["sup_slug"]) for i in test_idx)
    new_frozen_paths: list[str] = []
    for key, pool in pool_by_class.items():
        have = test_class_count.get(key, 0)
        n_real = sum(1 for i in pool if not _is_synthetic(i))
        want = min(
            max(MIN_FROZEN_PER_CLASS, int(n_real * SPLIT_RATIOS["test"])),
            n_real // 2,
        )
        need = max(0, want - have)
        if need <= 0:
            continue
        units = _group_units(pool)
        rng.shuffle(units)
        taken: set[int] = set()
        got = 0
        for unit in units:
            if got >= need:
                break
            real = [i for i in unit if not _is_synthetic(i)]
            if not real:
                continue  # 합성 전용 그룹은 frozen 후보 아님
            # 그룹 통째로 test: 실데이터 → test+frozen, 합성 멤버 → 미사용
            taken.update(unit)
            test_idx.extend(real)
            new_frozen_paths.extend(items[i]["source_path"] for i in real)
            got += len(real)
        pool_by_class[key] = [i for i in pool if i not in taken]

    # train/val — 그룹 원자 단위 stratified
    train_idx: list[int] = []
    val_idx: list[int] = []
    val_frac = SPLIT_RATIOS["val"] / (SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"])
    for key, pool in pool_by_class.items():
        units = _group_units(pool)
        rng.shuffle(units)
        n_val_target = int(len(pool) * val_frac) if len(pool) >= 4 else 0
        v = 0
        for unit in units:
            if v < n_val_target:
                val_idx.extend(unit)
                v += len(unit)
            else:
                train_idx.extend(unit)

    # hier frozen 갱신 저장 (동결 누적)
    all_frozen = sorted(hier_frozen | set(new_frozen_paths))
    HIER_FROZEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    HIER_FROZEN_PATH.write_text(
        json.dumps({"paths": all_frozen}, ensure_ascii=False, indent=1), encoding="utf-8",
    )

    # ★ 경로 기반 저장 — 아이템 목록이 변해도(파일 추가 등) 멤버십이 밀리지 않음.
    #   (v3 사이클 사고의 교훈: 인덱스 splits + 학습 중 파일 추가 → train/test
    #    인덱스 전체가 밀려 오염. 게이트가 잡아 롤백했지만 구조적으로 차단.)
    splits_paths = {
        "format": "paths_v2",
        "train": sorted(items[i]["source_path"] for i in train_idx),
        "val": sorted(items[i]["source_path"] for i in val_idx),
        "test": sorted(items[i]["source_path"] for i in test_idx),
    }
    HIER_SPLITS_PATH.write_text(json.dumps(splits_paths), encoding="utf-8")
    return {"train": sorted(train_idx), "val": sorted(val_idx), "test": sorted(test_idx)}


def load_or_build_hier_splits(items: list[dict[str, Any]]) -> dict[str, list[int]]:
    """경로 기반 splits 로드 → 현재 items 에 대한 인덱스로 변환.

    - splits 에 없는 신규 경로: 어느 split 에도 배정하지 않음 (다음 rebuild 때 합류)
      → 사이클 중간에 파일이 추가돼도 test 가 절대 오염되지 않음.
    - 구버전(인덱스 포맷) 파일이면 폐기하고 재생성.
    """
    if HIER_SPLITS_PATH.exists():
        data = json.loads(HIER_SPLITS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("format") == "paths_v2":
            path_to_idx = {it["source_path"]: i for i, it in enumerate(items)}
            out: dict[str, list[int]] = {}
            missing = 0
            for split in ("train", "val", "test"):
                idxs = []
                for p in data[split]:
                    i = path_to_idx.get(p)
                    if i is None:
                        missing += 1
                    else:
                        idxs.append(i)
                out[split] = idxs
            assigned = sum(len(v) for v in out.values())
            unassigned = len(items) - assigned
            if missing or unassigned:
                print(f"[hier_dataset] splits 매핑: 소실 {missing}, "
                      f"미배정 신규 {unassigned} (rebuild 전까지 학습 제외)")
            return out
        print("[hier_dataset] 구버전(인덱스) splits 감지 → 재생성")
        HIER_SPLITS_PATH.unlink()
    return build_hier_splits(items)


# 90° 단위 회전 증강 (v7) — AI-Hub 크롭은 센서 방향(EXIF 미적용)이라 모델이
# '눕힌' 객체 분포를 학습했고, 서빙(EXIF 세움) 입력이 분포 밖이 되는 문제의
# 근본 해결 (실측: 세움 26 vs 눕힘 36/51 → 서빙은 회전 TTA 로 임시 흡수 중.
# 방향 불변 모델이 되면 TTA 제거 가능 — 추론 1/3). SEMANTIC_FUSION_PLAN 참고.
ROT90_AUG = os.getenv("WASTE_HIER_ROT90_AUG", "0") == "1"


class HierImageDataset(Dataset):
    """(x, sup_kind_flag, sup_idx) 반환.

    sup_kind_flag: 1=fine, 0=coarse — collate 후 loss 에서 마스크로 사용.
    변환(리사이즈/정규화/증강)은 기존 WasteImageDataset 과 동일
    (+ WASTE_HIER_ROT90_AUG=1 이면 90° 단위 회전 무작위 적용).
    """

    def __init__(self, items: list[dict[str, Any]], augment: bool = False) -> None:
        self.items = items
        self.augment = augment
        # 증강/정규화 로직 재사용 (빈 dataset 을 transform 헬퍼로)
        self._helper = WasteImageDataset([], augment=augment)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        it = self.items[idx]
        x = _load_rgb_chw01(it)  # (3,224,224) [0,1]
        if self.augment:
            x = self._helper._apply_augmentation(x)
            if ROT90_AUG:
                k = random.randrange(4)
                if k:
                    x = torch.rot90(x, k, dims=(1, 2))
        x = (x - self._helper._mean) / self._helper._std
        return x, int(it["sup_kind"] == "fine"), it["sup_idx"]
