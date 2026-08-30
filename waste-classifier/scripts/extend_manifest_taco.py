"""data/raw/garbage-classification/{class}/taco_*.jpg 를 manifest.json 에 등재.

TACO ingest 후 한 번만 실행하면 됨. 학습이 이 파일들을 인식 가능.

사용:
    .venv/bin/python scripts/extend_manifest_taco.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSOR_ROOT = PROJECT_ROOT.parent / "waste-preprocessor"
RAW_DIR = PREPROCESSOR_ROOT / "data" / "raw" / "garbage-classification"
MANIFEST = PREPROCESSOR_ROOT / "data" / "processed" / "manifest.json"
MANIFEST_BACKUP_TACO = PREPROCESSOR_ROOT / "data" / "processed" / "manifest_pre_taco.json"


def main() -> int:
    if not MANIFEST.exists():
        sys.exit(f"manifest 없음: {MANIFEST}")
    if not MANIFEST_BACKUP_TACO.exists():
        shutil.copy(MANIFEST, MANIFEST_BACKUP_TACO)
        print(f"✓ manifest 백업: {MANIFEST_BACKUP_TACO}")

    m = json.load(MANIFEST.open())
    existing = {it["filename"] for it in m["items"]}

    added = 0
    per_class: Counter = Counter()
    for cls_dir in sorted(RAW_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        for f in sorted(cls_dir.glob("taco_*.jpg")):
            if f.name in existing:
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

    with MANIFEST.open("w") as fh:
        json.dump(m, fh, ensure_ascii=False)
    print(f"\n✓ manifest 확장: 총 items {len(m['items'])} (taco 추가 {added}장)")
    print("클래스별 TACO 추가:")
    for cls, n in sorted(per_class.items()):
        print(f"  {cls:<14} +{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
