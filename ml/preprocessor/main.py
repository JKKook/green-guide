"""greenguide-preprocessor CLI.

사용 예:
    python main.py                         # 전체 파이프라인 (로컬만)
    python main.py --upload                # Supabase 업로드 포함
    python main.py --step collect          # 특정 단계만
"""
from __future__ import annotations

import argparse
import sys

from greenguide_preprocessor import config


def _run_pipeline(upload: bool, vectorize: bool = True) -> int:
    from greenguide_preprocessor.pipeline import run
    run(upload_to_supabase=upload, vectorize=vectorize)
    return 0


def _run_step(step: str) -> int:
    if step == "collect":
        from greenguide_preprocessor.collect import ensure_dataset
        ensure_dataset()
        return 0
    if step == "catalog":
        from greenguide_preprocessor.catalog import build_catalog, save_catalog
        save_catalog(build_catalog())
        return 0
    if step == "cleanse":
        from greenguide_preprocessor.catalog import load_catalog, save_catalog
        from greenguide_preprocessor.cleanse import cleanse
        cleansed, _ = cleanse(load_catalog())
        save_catalog(cleansed, config.INTERIM_DIR / "catalog.cleansed.json")
        return 0
    if step == "preprocess":
        from greenguide_preprocessor.preprocess import run_sample
        run_sample()
        return 0
    if step == "vectorize":
        from greenguide_preprocessor.vectorize import run_sample
        run_sample()
        return 0
    if step == "supabase-check":
        from greenguide_preprocessor.storage import SupabaseStore
        store = SupabaseStore()
        print(f"[supabase] connected. bucket={store.bucket} table={store.table}")
        return 0
    print(f"unknown step: {step}", file=sys.stderr)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(prog="greenguide-preprocessor")
    parser.add_argument(
        "--step",
        choices=["collect", "catalog", "cleanse", "preprocess", "vectorize", "supabase-check"],
        help="개별 단계 실행 (기본: 전체 파이프라인)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Supabase 에 원본 이미지 업로드 + 메타데이터 INSERT",
    )
    parser.add_argument(
        "--no-vectorize",
        action="store_true",
        help="npz 벡터화 생략 — manifest 만 생성 (CNN 은 raw 직접 로드). 디스크 절약.",
    )
    args = parser.parse_args()

    config.ensure_directories()
    if args.step:
        return _run_step(args.step)
    return _run_pipeline(upload=args.upload, vectorize=not args.no_vectorize)


if __name__ == "__main__":
    raise SystemExit(main())
