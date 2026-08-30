"""로컬 폴더의 사진을 학습 raw 데이터로 인제스트 (실사용 사진 수동 추가용).

사용자가 폰으로 찍어 Mac 폴더로 옮긴 사진을 → 256px 로 정규화해 raw/{class}/ 에 저장.
EXIF/회전 보정 + 메타 제거. 2-1(손→non_object) 및 1-2(실물→각 클래스) 공용.

사용:
    python scripts/ingest_folder.py --our-class non_object --src ~/Desktop/hands
    python scripts/ingest_folder.py --our-class plastic --src ~/Desktop/plastic_photos --cap 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _base import RAW_DIR
from PIL import Image, ImageOps, UnidentifiedImageError

STORE_SIZE = 256
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic")


def _count(our_class: str) -> int:
    d = RAW_DIR / our_class
    return sum(1 for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.exists() else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="로컬 폴더 → raw/{class} 인제스트")
    ap.add_argument("--our-class", required=True, help="저장할 클래스 slug (예: non_object)")
    ap.add_argument("--src", required=True, help="사진이 든 로컬 폴더 (재귀 탐색)")
    ap.add_argument("--cap", type=int, default=100000, help="이번 인제스트 최대 장수")
    ap.add_argument("--prefix", default="realworld", help="파일명 접두 (출처 구분용)")
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    if not greenguide_classifier.exists():
        sys.exit(f"ERROR: 소스 폴더 없음: {src}")

    out_dir = RAW_DIR / args.our_class
    out_dir.mkdir(parents=True, exist_ok=True)
    start = _count(args.our_class)
    n = start

    files = [p for p in greenguide_classifier.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT]
    print(f"[ingest] {src} 에서 이미지 {len(files)}개 발견 → {args.our_class} (현재 {start}장)")

    added = 0
    for p in files:
        if added >= args.cap:
            break
        try:
            img = Image.open(p)
            img = ImageOps.exif_transpose(img).convert("RGB")  # 회전 보정 + EXIF 제거
            img = img.resize((STORE_SIZE, STORE_SIZE), Image.BILINEAR)
            img.save(out_dir / f"{args.prefix}_{args.our_class}_{n:06d}.jpg", quality=88)
            n += 1
            added += 1
        except (UnidentifiedImageError, OSError) as exc:
            print(f"  [skip] {p.name}: {exc}")

    print(f"✓ {added}장 추가 → {args.our_class} 총 {n}장")
    print("  다음: (다른 클래스도 인제스트 후) retrain.py 실행")
    return 0


if __name__ == "__main__":
    sys.exit(main())
