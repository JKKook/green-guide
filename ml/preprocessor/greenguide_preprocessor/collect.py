"""Phase 1: 데이터 수집.

Kaggle "Garbage Classification" 데이터셋을 확보한다.
- 이미 받아져 있으면 그대로 사용
- kaggle CLI 가 설치 + 인증되어 있으면 자동 다운로드
- 둘 다 안 되면 수동 다운로드 안내 후 종료
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from greenguide_preprocessor import config
from greenguide_common.logging import get_logger

log = get_logger(__name__)

KAGGLE_DATASET_SLUG = "asdasdasasdas/garbage-classification"


def dataset_present(dataset_dir: Path | None = None) -> bool:
    """6개 클래스 폴더가 모두 존재하고 이미지가 들어있는지 검사."""
    dataset_dir = dataset_dir if dataset_dir is not None else config.DATASET_DIR
    if not dataset_dir.exists():
        return False
    for label in config.CLASS_LABELS:
        class_dir = dataset_dir / label
        if not class_dir.is_dir():
            return False
        has_image = any(
            p.suffix.lower() in config.SUPPORTED_EXTENSIONS for p in class_dir.iterdir()
        )
        if not has_image:
            return False
    return True


def _kaggle_available() -> bool:
    return shutil.which("kaggle") is not None


def _download_via_kaggle(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET_SLUG,
            "-p", str(target_dir),
            "--unzip",
        ],
        check=True,
    )


def _flatten_nested_structure(dataset_dir: Path) -> None:
    """Kaggle zip 의 중첩 구조(Garbage classification/Garbage classification/<class>) 평탄화."""
    if all((dataset_dir / label).is_dir() for label in config.CLASS_LABELS):
        return

    candidates = [p for p in dataset_dir.rglob("*") if p.is_dir() and p.name in config.CLASS_LABELS]
    if not candidates:
        return

    by_label: dict[str, Path] = {}
    for path in candidates:
        by_label.setdefault(path.name, path)

    for label, src in by_label.items():
        dst = dataset_dir / label
        if greenguide_preprocessor.resolve() == dst.resolve():
            continue
        if dst.exists():
            for f in greenguide_preprocessor.iterdir():
                shutil.move(str(f), str(dst / f.name))
            greenguide_preprocessor.rmdir()
        else:
            shutil.move(str(src), str(dst))


def _print_manual_instructions() -> None:
    print(
        "\n[수동 다운로드 안내]\n"
        "1) https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification 접속\n"
        "2) Download 버튼으로 zip 받기\n"
        f"3) 압축 풀고 6개 폴더(cardboard, glass, metal, paper, plastic, trash)를\n"
        f"   다음 경로에 위치: {config.DATASET_DIR}\n"
        "4) 완료 후 다시 실행하세요.\n",
        file=sys.stderr,
    )


def ensure_dataset() -> Path:
    """데이터셋 준비를 보장하고 데이터셋 루트 경로를 반환."""
    config.ensure_directories()

    if dataset_present():
        log.info("dataset already present at %s", config.DATASET_DIR)
        return config.DATASET_DIR

    if _kaggle_available():
        log.info("downloading via kaggle CLI → %s", config.DATASET_DIR)
        try:
            _download_via_kaggle(config.DATASET_DIR)
            _flatten_nested_structure(config.DATASET_DIR)
        except subprocess.CalledProcessError as exc:
            log.error("kaggle download failed: %s", exc)
            _print_manual_instructions()
            raise SystemExit(1) from exc

        if dataset_present():
            log.info("dataset ready at %s", config.DATASET_DIR)
            return config.DATASET_DIR

        log.error("download finished but expected class folders are missing.")
        _print_manual_instructions()
        raise SystemExit(1)

    log.error("kaggle CLI not found.")
    _print_manual_instructions()
    raise SystemExit(1)


if __name__ == "__main__":
    ensure_dataset()
