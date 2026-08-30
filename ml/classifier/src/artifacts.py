"""재학습 사이클용 아티팩트 백업/롤백 (retrain.py / retrain_hier.py 공용)."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from waste_common.logging import get_logger

log = get_logger(__name__)


def backup_artifacts(
    artifacts: list[tuple[Path, str]], backup_root: Path, prefix: str,
) -> Path | None:
    """artifacts [(src, name)] 를 backup_root/<prefix>_<timestamp>/ 로 복사.
    Returns: 백업 폴더 경로 또는 None (복사된 게 없으면).
    """
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest = backup_root / f"{prefix}_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src, name in artifacts:
        if src.exists():
            shutil.copy2(src, dest / name)
            copied += 1
    log.info(f"{copied}개 → {dest}")
    return dest if copied else None


def rollback_artifacts(
    artifacts: list[tuple[Path, str]], backup: Path | None, *,
    failed_root: Path | None = None, prefix: str = "",
) -> bool:
    """백업에서 artifacts 복원. failed_root 가 주어지면 복원 전 실패 산출물을 보존.
    Returns True if 복원됨.
    """
    if backup is not None and failed_root is not None:
        # 복원 전, 실패 사이클의 산출물을 진단용으로 보존 — v8~v10 에서 롤백이
        # 실패 모델을 지워버려 "etc 가 어디로 새는지" 사후 분석이 불가능했던 문제.
        stamp = time.strftime("%Y%m%d_%H%M%S")
        fail_dir = failed_root / f"{prefix}_{stamp}"
        fail_dir.mkdir(parents=True, exist_ok=True)
        kept = 0
        for src, name in artifacts:
            if src.exists():
                shutil.copy2(src, fail_dir / name)
                kept += 1
        log.warning(f"실패 산출물 {kept}개 보존 → {fail_dir}")

    if backup is None or not backup.exists():
        log.warning("백업이 없어 복원 불가 (첫 학습이었을 수 있음)")
        return False
    restored: list[str] = []
    for dst, name in artifacts:
        b = backup / name
        if b.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(b, dst)
            restored.append(name)
    log.info(f"{backup} 에서 복원: {restored}")
    return True
