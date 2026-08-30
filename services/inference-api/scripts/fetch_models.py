#!/usr/bin/env python3
"""로컬 개발용 — HF Hub ethanDev92/waste-models/serving/ 을 waste-api/models/ 로 내려받는다.
Docker 빌드(Dockerfile)와 같은 원본을 쓴다. 실행: .venv/bin/python scripts/fetch_models.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "models"


def main() -> None:
    tmp = snapshot_download("ethanDev92/waste-models", allow_patterns=["serving/**"])
    src = Path(tmp) / "serving"
    DEST.mkdir(exist_ok=True)
    n = 0
    for f in src.rglob("*"):
        if f.is_file():
            out = DEST / f.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
            n += 1
    print(f"[fetch_models] {n} files → {DEST}")


if __name__ == "__main__":
    sys.exit(main())
