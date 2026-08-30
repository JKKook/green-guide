"""MIT Indoor 67 → 실내 배경 풀 추출.

흐름:
1. /tmp/indoorCVPR_09.tar 에서 tar 추출 (메모리 스트리밍)
2. 각 카테고리 별 무작위 N장 샘플링 — 균형 (주방·거실·식당·침실 등 우선)
3. 256x256 리사이즈 + JPG q=85 저장
4. data/raw/_aux/backgrounds/

카테고리 우선순위 (우리 사용자 분포에 가까운 것):
  high: kitchen, dining_room, livingroom, bedroom, bathroom, laundromat
  mid: office, classroom, gym, garage, pantry, closet
  low: 그 외 상업 시설 (대기실·로비·강당 등)

사용:
    .venv/bin/python scripts/build_indoor_bg_pool.py [--per-category 30]
"""
from __future__ import annotations

import io
import random
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from _base import PREPROCESSOR_ROOT, make_parser
from PIL import Image

BG_DIR = PREPROCESSOR_ROOT / "data" / "raw" / "_aux" / "backgrounds"
TAR_PATH = Path("/tmp/indoorCVPR_09.tar")
TARGET_SIZE = 256
JPEG_QUALITY = 85

# 카테고리 우선순위 (낮을수록 더 많이)
CATEGORY_PRIORITY = {
    # 가정 — 높은 우선순위 (사용자 분포의 핵심)
    "kitchen": 1, "dining_room": 1, "livingroom": 1, "bedroom": 1,
    "bathroom": 1, "laundromat": 1, "pantry": 1, "closet": 1, "garage": 1,
    # 사무실·교실 — 사용자가 종종 찍는 환경
    "office": 2, "computerroom": 2, "classroom": 2, "library": 2, "studio_music": 2,
    # 그 외 — 낮은 우선순위
}

DEFAULT_PRIORITY = 3


def extract_and_save(args) -> int:
    if not TAR_PATH.exists():
        sys.exit(f"tar 없음: {TAR_PATH}\n"
                 f"다운로드: curl -L -o {TAR_PATH} "
                 f"http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar")

    BG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[bg-pool] tar 열기: {TAR_PATH} ({TAR_PATH.stat().st_size//(1024**2)} MB)")

    # 카테고리별 멤버 수집 (메모리 효율 위해 tar walk)
    cat_files: dict[str, list[tarfile.TarInfo]] = defaultdict(list)
    with tarfile.open(TAR_PATH, "r") as tf:
        for ti in tf:
            if not ti.isfile():
                continue
            n = ti.name
            if not n.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            # 경로: Images/{category}/{file}.jpg
            parts = Path(n).parts
            if len(parts) >= 3 and parts[0].lower().startswith("image"):
                cat = parts[1]
                cat_files[cat].append(ti)
    print(f"  카테고리 {len(cat_files)}개 발견")

    # 우선순위별 분류
    rng = random.Random(args.seed)
    per_cat_high = args.per_category
    per_cat_mid = max(5, args.per_category // 3)
    per_cat_low = max(2, args.per_category // 10)

    selected: list[tuple[tarfile.TarInfo, str]] = []
    for cat, members in sorted(cat_files.items()):
        pri = CATEGORY_PRIORITY.get(cat, DEFAULT_PRIORITY)
        n = {1: per_cat_high, 2: per_cat_mid, 3: per_cat_low}[pri]
        sampled = rng.sample(members, min(n, len(members)))
        for m in sampled:
            selected.append((m, cat))

    print(f"  추출 대상: {len(selected)}장")
    print(f"    high priority (가정/실내): {sum(1 for _, c in selected if CATEGORY_PRIORITY.get(c, 3) == 1)}")
    print(f"    mid priority (사무·교실): {sum(1 for _, c in selected if CATEGORY_PRIORITY.get(c, 3) == 2)}")
    print(f"    low priority (기타): {sum(1 for _, c in selected if CATEGORY_PRIORITY.get(c, 3) == 3)}")

    if args.dry_run:
        print("\n(dry-run) 저장 안 함")
        return 0

    # 실제 추출
    saved = 0
    skipped = 0
    cat_count: Counter = Counter()
    with tarfile.open(TAR_PATH, "r") as tf:
        for ti, cat in selected:
            try:
                f = tf.extractfile(ti)
                if f is None:
                    skipped += 1
                    continue
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
                # 작은 이미지 거절 (해상도 부족)
                if min(img.size) < 200:
                    skipped += 1
                    continue
                # center crop to square, then resize
                w, h = img.size
                s = min(w, h)
                img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
                img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
                out_name = f"mit_indoor_{cat}_{saved:05d}.jpg"
                img.save(BG_DIR / out_name, "JPEG", quality=JPEG_QUALITY)
                saved += 1
                cat_count[cat] += 1
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {ti.name}: {e}")
                skipped += 1

    print(f"\n✓ {saved}장 추출 → {BG_DIR}")
    print(f"  skipped: {skipped}")
    print("\n카테고리별 카운트 (상위 10):")
    for c, n in cat_count.most_common(10):
        print(f"  {c:<25} {n}장 (priority={CATEGORY_PRIORITY.get(c, 3)})")
    return 0


def main() -> int:
    ap = make_parser("build_indoor_bg_pool", "MIT Indoor 67 → 실내 배경 풀")
    ap.add_argument("--per-category", type=int, default=15,
                    help="high-priority 카테고리당 추출 장수 (기본 15)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return extract_and_save(args)


if __name__ == "__main__":
    sys.exit(main())
