"""스크립트 공통 argparse 팩토리."""
from __future__ import annotations

import argparse


def make_parser(
    prog: str,
    description: str | None = None,
    *,
    seed: int | None = 42,
    dry_run: bool = False,
    cap: bool = False,
) -> argparse.ArgumentParser:
    """공통 플래그(--seed/--dry-run/--cap)를 갖춘 파서. 필요한 것만 켠다."""
    parser = argparse.ArgumentParser(prog=prog, description=description)
    if seed is not None:
        parser.add_argument("--seed", type=int, default=seed, help=f"난수 시드 (기본 {seed})")
    if dry_run:
        parser.add_argument("--dry-run", action="store_true", help="쓰기 없이 계획만 출력")
    if cap:
        parser.add_argument("--cap", type=int, default=None, help="처리 상한 (기본: 무제한)")
    return parser
