"""data/raw/synthetic_indoor/{class}/synth_*.jpg 를 garbage-classification 으로 복사 +
manifest.json 에 등록 (학습이 인식하도록).

retrain 후 정리: garbage-classification 의 synth_*.jpg 삭제 + manifest 복원.

사용:
    .venv/bin/python scripts/extend_manifest_synthetic.py
    .venv/bin/python scripts/extend_manifest_synthetic.py --cleanup
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSOR_ROOT = PROJECT_ROOT.parent / "waste-preprocessor"
SYNTH_DIR = PREPROCESSOR_ROOT / "data" / "raw" / "synthetic_indoor"
RAW_DIR = PREPROCESSOR_ROOT / "data" / "raw" / "garbage-classification"
MANIFEST = PREPROCESSOR_ROOT / "data" / "processed" / "manifest.json"
MANIFEST_BACKUP = PREPROCESSOR_ROOT / "data" / "processed" / "manifest_pre_synthetic.json"


def add_synthetic() -> int:
    if not SYNTH_DIR.exists():
        sys.exit(f"synthetic_indoor 없음: {SYNTH_DIR}")
    if not MANIFEST.exists():
        sys.exit(f"manifest 없음: {MANIFEST}")

    # 1) 백업 (한 번만)
    if not MANIFEST_BACKUP.exists():
        shutil.copy(MANIFEST, MANIFEST_BACKUP)
        print(f"✓ manifest 백업: {MANIFEST_BACKUP}")

    # 2) manifest 읽기
    m = json.load(MANIFEST.open())
    existing_filenames = {it["filename"] for it in m["items"]}

    added = 0
    per_class: Counter = Counter()
    for cls_dir in sorted(SYNTH_DIR.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name.startswith("_"):
            continue
        target_dir = RAW_DIR / cls_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        for f in sorted(cls_dir.glob("synth_*.jpg")):
            target = target_dir / f.name
            # 1차 복사 (없으면)
            if not target.exists():
                shutil.copy(f, target)
            # 2차 manifest 등재 (없으면)
            if f.name in existing_filenames:
                continue
            file_id = hashlib.md5(f.name.encode()).hexdigest()[:12]
            m["items"].append({
                "id": file_id,
                "label": cls_dir.name,
                "source_path": f"data/raw/garbage-classification/{cls_dir.name}/{f.name}",
                "filename": f.name,
            })
            added += 1
            per_class[cls_dir.name] += 1

    # 3) manifest 저장
    with MANIFEST.open("w") as fh:
        json.dump(m, fh, ensure_ascii=False)
    print(f"\n✓ manifest 확장: 총 items {len(m['items'])} (synth 추가 {added}장)")
    print("클래스별 합성 추가:")
    for cls, n in sorted(per_class.items()):
        print(f"  {cls:<14} +{n}")
    return 0


def cleanup() -> int:
    """garbage-classification 의 synth_*.jpg 삭제 + manifest 복원."""
    if MANIFEST_BACKUP.exists():
        shutil.copy(MANIFEST_BACKUP, MANIFEST)
        MANIFEST_BACKUP.unlink()
        print(f"✓ manifest 복원")
    else:
        print("⚠ manifest backup 없음 — skip 복원")

    removed = 0
    for cls_dir in RAW_DIR.iterdir():
        if not cls_dir.is_dir():
            continue
        for f in cls_dir.glob("synth_*.jpg"):
            f.unlink()
            removed += 1
    print(f"✓ {removed} synth_*.jpg 삭제 (synthetic_indoor/ 원본은 보존)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleanup", action="store_true",
                    help="retrain 후 synth 파일 + manifest 복원")
    args = ap.parse_args()
    return cleanup() if args.cleanup else add_synthetic()


if __name__ == "__main__":
    sys.exit(main())
