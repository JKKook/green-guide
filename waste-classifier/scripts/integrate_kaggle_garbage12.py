"""Kaggle `mostafaabla/garbage-classification` (12 classes, 15K+ images) 통합.

흐름:
  1. 사전: kaggle datasets download mostafaabla/garbage-classification --unzip
     (이 스크립트는 다운로드는 안 함. 압축 해제된 폴더 위치만 인자로 받음)
  2. 클래스 매핑 (KAGGLE_TO_OURS):
     - biological → food_waste
     - brown/green/white-glass → glass (3종 모두)
     - shoes → clothes
     - battery → skip (현재 우리 분류에 없음, 별도 추가 필요)
     - 나머지는 1:1
  3. 매핑된 이미지를 `data/raw/garbage-classification/{our_class}/` 에
     `kg2_<원본명>` prefix 로 복사. 기존 데이터와 충돌 없이 누적.
  4. 다음 retrain.py 가 이 데이터를 자동으로 학습에 포함.

사용:
    cd waste-classifier
    .venv/bin/python scripts/integrate_kaggle_garbage12.py --src /tmp/kaggle_garbage12 --dry-run
    .venv/bin/python scripts/integrate_kaggle_garbage12.py --src /tmp/kaggle_garbage12
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSOR_ROOT = PROJECT_ROOT.parent / "waste-preprocessor"
TARGET_DIR = PREPROCESSOR_ROOT / "data" / "raw" / "garbage-classification"

# 데이터셋 클래스명 → 우리 클래스 slug.
# None = skip (해당 클래스를 우리는 아직 다루지 않음).
KAGGLE_TO_OURS: dict[str, str | None] = {
    "battery": None,           # 폐건전지 — 별도 클래스 추가 시 활성화
    "biological": "food_waste",
    "brown-glass": "glass",
    "cardboard": "cardboard",
    "clothes": "clothes",
    "green-glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "shoes": "clothes",        # 의류 통합 (별도 'shoes' 클래스 미존재)
    "trash": "trash",
    "white-glass": "glass",
}

VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _find_dataset_root(src: Path) -> Path:
    """압축 해제된 경로 중 12 클래스 폴더를 갖는 디렉토리 찾기.

    Kaggle 데이터셋은 압축 해제 시 'garbage_classification/' 같은 single subdir
    안에 클래스 폴더들이 있는 경우가 흔함. 그 경우 자동 탐색.
    """
    if not src.exists():
        sys.exit(f"ERROR: {src} 가 없음")

    expected = set(KAGGLE_TO_OURS.keys())

    # 1) src 자체가 클래스 폴더들을 갖는지
    direct = {p.name for p in src.iterdir() if p.is_dir()}
    if expected.issubset(direct):
        return src

    # 2) 한 단계 더 들어가서 찾기
    for sub in src.iterdir():
        if sub.is_dir():
            inner = {p.name for p in sub.iterdir() if p.is_dir()}
            if expected.issubset(inner):
                return sub

    sys.exit(
        f"ERROR: {src} 안에서 12 클래스 폴더를 찾지 못함.\n"
        f"  기대한 클래스: {sorted(expected)}\n"
        f"  실제 상위 폴더: {sorted(direct)}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Kaggle garbage-12 데이터셋 통합")
    parser.add_argument(
        "--src", type=Path, required=True,
        help="압축 해제된 데이터셋 폴더 (예: /tmp/kaggle_garbage12)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="실제 복사 없이 통계만 출력",
    )
    parser.add_argument(
        "--max-per-source-class", type=int, default=None,
        help="원본 클래스별 최대 이미지 수 (테스트용; 미지정 시 전체)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Kaggle garbage-classification (12 classes) → 통합")
    print("=" * 60)

    root = _find_dataset_root(args.src)
    print(f"\n[plan] 데이터셋 root: {root}")

    # 클래스별 이미지 수 + 매핑 결과 미리 계산
    plan: dict[str, tuple[str | None, list[Path]]] = {}
    for src_class, target_class in KAGGLE_TO_OURS.items():
        class_dir = root / src_class
        if not class_dir.exists():
            print(f"  [warn] {src_class} 폴더 없음 — skip")
            continue
        imgs = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_IMG_EXT
        )
        if args.max_per_source_class:
            imgs = imgs[: args.max_per_source_class]
        plan[src_class] = (target_class, imgs)

    # 통계 출력
    print(f"\n[plan] 매핑 결과:")
    by_target: dict[str, int] = Counter()
    skipped_total = 0
    for src_class, (target, imgs) in sorted(plan.items()):
        if target is None:
            print(f"  {src_class:14s} ({len(imgs):>5d} 장) → SKIP")
            skipped_total += len(imgs)
        else:
            print(f"  {src_class:14s} ({len(imgs):>5d} 장) → {target}")
            by_target[target] += len(imgs)

    print(f"\n[plan] 우리 클래스별 누적 (이번 통합으로 추가될 양):")
    for tgt, n in sorted(by_target.items(), key=lambda kv: -kv[1]):
        # 기존 폴더에 이미 있는 양도 같이 보여줌
        existing_dir = TARGET_DIR / tgt
        existing = (
            sum(1 for p in existing_dir.iterdir()
                if p.is_file() and p.suffix.lower() in VALID_IMG_EXT)
            if existing_dir.exists() else 0
        )
        print(f"  {tgt:14s} 기존 {existing:>5d}장 + 추가 {n:>5d}장 = {existing + n}장")

    print(f"\n[plan] 통합 안 함: {skipped_total}장 (battery 등)")

    total_to_copy = sum(len(imgs) for _, imgs in plan.values()
                        if plan and plan.get(src_class := list(plan.keys())[0])[0]) if False else \
                    sum(len(imgs) for tgt, imgs in plan.values() if tgt is not None)
    print(f"[plan] 총 복사할 이미지: {total_to_copy:,}장")

    if args.dry_run:
        print("\n[dry-run] 실제 복사 안 함. --dry-run 빼고 재실행하세요.")
        return 0

    # 실제 복사
    print(f"\n[copy] 시작...")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped_existing = 0
    failed = 0

    for src_class, (target, imgs) in plan.items():
        if target is None:
            continue
        dst_dir = TARGET_DIR / target
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            dst_name = f"kg2_{src_class}_{img.name}"
            dst = dst_dir / dst_name
            if dst.exists():
                skipped_existing += 1
                continue
            try:
                shutil.copy2(img, dst)
                copied += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  [fail] {img} → {dst}: {exc}")
                failed += 1
        print(f"  {src_class:14s} → {target}: 완료")

    print(f"\n[copy] 결과:")
    print(f"  copied={copied:,}  skipped(existing)={skipped_existing}  failed={failed}")
    print(f"\n다음 단계:")
    print(f"  cd ~/ai/waste-classifier")
    print(f"  .venv/bin/python retrain.py")
    print(f"\nretrain.py 가 새 데이터를 자동 감지해서 학습합니다 (시간 ~30분 예상).")
    print(f"학습 완료 후 ONNX 가 Supabase 에 publish 되고 앱·서버가 자동 갱신됩니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
