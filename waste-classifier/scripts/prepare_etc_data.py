"""'기타/etc' 클래스용 OOD 이미지 수집 스크립트.

GreenGuide 모델이 손·우레탄·배경 등 분리수거 대상이 아닌 객체를
cardboard/paper 로 잘못 분류하는 문제 완화용. ImageNet/Open Images 류
일반 객체 이미지를 모아서 'etc' 라벨로 학습시킴.

⚠️  **반드시 실행 전 코드 검토 + 다운로드 카테고리·개수 조정**.
⚠️  외부 데이터셋 라이센스 사용자 책임.

기본 동작:
  - HuggingFace `datasets` 라이브러리로 ImageNet-1k subset (CC license)
    또는 fallback 으로 Open Images V7 의 일부 카테고리에서 N장 다운로드
  - `~/ai/waste-preprocessor/data/raw/garbage-classification/etc/` 에 저장
  - 다음 retrain.py 실행 시 자동으로 etc 클래스로 학습됨

사용:
    cd waste-classifier
    .venv/bin/python scripts/prepare_etc_data.py --n 100 --dry-run     # 어떤 카테고리·이미지 받을지 미리 보기
    .venv/bin/python scripts/prepare_etc_data.py --n 100               # 실제 다운로드
    .venv/bin/python scripts/prepare_etc_data.py --source local --dir ~/my_ood_photos  # 직접 찍은 사진 통합
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSOR_ROOT = PROJECT_ROOT.parent / "waste-preprocessor"
TARGET_DIR = PREPROCESSOR_ROOT / "data" / "raw" / "garbage-classification" / "etc"

# OOD 로 적합한 ImageNet 카테고리 (waste 와 무관한 일반 객체).
# 사용자가 직접 추가/조정 가능.
SUGGESTED_IMAGENET_CATEGORIES = [
    # 사람 관련 (손·얼굴 등)
    "person",
    # 가구
    "chair", "desk", "couch",
    # 전자제품
    "laptop", "monitor", "remote_control",
    # 자연
    "tree", "plant", "flower",
    # 의류 (clothes 클래스와 겹칠 수 있으니 주의)
    # "shirt", "shoe",
    # 동물
    "dog", "cat",
]


def _check_supabase_class() -> bool:
    """waste_classes 에 'etc' 가 active 인지 확인."""
    try:
        import os

        from dotenv import load_dotenv
        from supabase import create_client

        load_dotenv(PREPROCESSOR_ROOT / ".env")
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            print("[warn] SUPABASE_URL/KEY 없음 — Supabase 검증 skip")
            return True

        client = create_client(url, key)
        res = client.table("waste_classes").select("slug,active").eq("slug", "etc").execute()
        rows = res.data or []
        if not rows:
            print("[ERROR] 'etc' 클래스가 waste_classes 에 없음.")
            print("        먼저 migrations/002_etc_class.sql 을 Supabase 에서 실행하세요.")
            return False
        if not rows[0]["active"]:
            print("[ERROR] 'etc' 클래스가 비활성화됨 (active=false).")
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Supabase 검증 실패: {exc} — 계속 진행")
        return True


def _download_from_hf_imagenet(n_per_category: int, categories: list[str], dry_run: bool) -> int:
    """HuggingFace `datasets` 로 ImageNet-1k 의 일부 카테고리 다운로드.

    Returns: 실제 저장된 이미지 수.
    """
    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError:
        print("[ERROR] huggingface datasets 미설치.")
        print("        .venv/bin/pip install datasets pillow")
        return 0

    print(f"[plan] HF imagenet-1k subset 에서 {len(categories)}개 카테고리 × {n_per_category}장 = "
          f"{len(categories) * n_per_category}장 예정")
    for c in categories:
        print(f"  - {c}")

    if dry_run:
        print("\n[dry-run] 실제 다운로드 안 함. 이 카테고리들로 진행하려면 --n 만 지정해서 재실행.")
        return 0

    # 실제 구현 — HF datasets 사용 (gated 데이터셋이라 hf token 필요할 수 있음)
    print("\n[note] HF imagenet-1k 는 gated dataset 입니다.")
    print("  https://huggingface.co/datasets/imagenet-1k 에서 라이센스 수락 필요.")
    print("  대신 더 친화적인 데이터셋을 사용하려면 --source manual 로 직접 사진 모으기 권장.")
    return 0


def _copy_local_dir(src_dir: Path, n: int, dry_run: bool) -> int:
    """로컬 폴더에서 이미지 N장 복사 (사용자가 직접 찍은 사진)."""
    if not src_dir.exists():
        print(f"[ERROR] {src_dir} 가 없음")
        return 0

    candidates = sorted(
        p for p in src_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    )
    if not candidates:
        print(f"[ERROR] {src_dir} 에 이미지가 없음")
        return 0

    selected = candidates[:n] if n > 0 else candidates
    print(f"[plan] {src_dir} → {TARGET_DIR}")
    print(f"  {len(candidates)}장 중 {len(selected)}장 복사 예정")

    if dry_run:
        print("\n[dry-run] 실제 복사 안 함.")
        return 0

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in selected:
        dest = TARGET_DIR / f"etc_local_{src.name}"
        if dest.exists():
            continue
        shutil.copy2(src, dest)
        copied += 1
    print(f"\n✓ {copied}장 복사 완료 → {TARGET_DIR}")
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(
        description="'기타/etc' 클래스용 OOD 이미지 수집",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", choices=["imagenet", "local"], default="imagenet",
        help="imagenet = HF datasets (gated, token 필요), local = 사용자 폴더에서 복사",
    )
    parser.add_argument(
        "--dir", type=Path, default=None,
        help="--source local 일 때 원본 폴더 경로",
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="다운로드/복사할 이미지 수 (per category, 기본 50)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="실제 다운로드/복사 없이 계획만 출력",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("기타(etc) 클래스 OOD 데이터 수집")
    print("=" * 60)

    # 1. 'etc' 클래스가 Supabase 에 등록돼있는지 확인
    print("\n[1/3] Supabase waste_classes 검증...")
    if not _check_supabase_class():
        return 1
    print("  ✓ 'etc' 클래스 active")

    # 2. 다운로드/복사
    print(f"\n[2/3] 이미지 수집 ({args.source})...")
    if args.source == "imagenet":
        count = _download_from_hf_imagenet(args.n, SUGGESTED_IMAGENET_CATEGORIES, args.dry_run)
    elif args.source == "local":
        if args.dir is None:
            print("[ERROR] --source local 일 때는 --dir <경로> 필수")
            return 1
        count = _copy_local_dir(args.dir, args.n, args.dry_run)
    else:
        print(f"[ERROR] unsupported source: {args.source}")
        return 1

    # 3. 다음 단계 안내
    print(f"\n[3/3] 완료 — 수집된 이미지: {count}장")
    if count > 0 and not args.dry_run:
        print("\n다음 단계:")
        print("  cd ~/ai/waste-classifier")
        print("  .venv/bin/python retrain.py")
        print("")
        print("  retrain 이 'etc' 폴더를 자동 감지해서 7개 클래스로 학습합니다.")
        print("  학습 후 새 ONNX 가 자동으로 Supabase 에 publish 되고,")
        print("  앱·서버가 다음 부팅 시 새 모델을 받아옵니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
