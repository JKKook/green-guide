"""uvicorn 진입점.

사용:
    python main.py                              # 기본 (0.0.0.0:8000)
    python main.py --host 127.0.0.1 --port 9000
    python main.py --reload                      # 개발 모드 (코드 변경 시 자동 재시작)
"""
from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    import os
    parser = argparse.ArgumentParser(prog="waste-api")
    parser.add_argument("--host", default="0.0.0.0")
    # PORT 환경변수가 있으면 우선 (Render·HF Spaces·Cloud Run 등은 PORT 로 주입)
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
    )
    parser.add_argument("--reload", action="store_true", help="dev: 코드 변경 시 자동 재시작")
    parser.add_argument("--workers", type=int, default=1, help="prod: worker 개수")
    args = parser.parse_args()

    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
