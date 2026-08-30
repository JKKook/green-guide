"""도메인 예외 → HTTP 응답 매핑 (엔드포인트마다 try/except 반복 제거)."""
from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from src.preprocess import ImageDecodeError


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ImageDecodeError)
    async def _image_decode_error(_: Request, exc: ImageDecodeError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)},
        )
