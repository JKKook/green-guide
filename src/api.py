"""FastAPI app 정의."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core import config
from src.core.errors import register_exception_handlers
from src.routers import admin, inference, learning, meta
from src.classes import ClassRegistry
from src.inference import get_active_meta, get_classifier, reset_classifier
from src.dinov2_classifier import get_dinov2_classifier
from src.uploads import get_recorder
from src.core.log import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier = get_classifier()
    meta = get_active_meta()
    # Supabase 불가 시에도 부팅은 계속 — 레지스트리는 요청 시 재시도됨.
    # (설계 원칙: "API 는 항상 부팅한다" — model_loader 와 동일한 강건성)
    try:
        ClassRegistry.load()
    except Exception as exc:  # noqa: BLE001
        log.warning(f"class registry 로드 실패 (Supabase 미접속?): {exc}")
    log.info(f"color model: {classifier.model_path}")
    log.info(f"edge model: {classifier.edge_model_path or '(disabled)'}")
    log.info(f"inference mode: "
          f"{'ensemble (color+edge)' if classifier.has_edge_stream else 'single (color)'}")
    if meta is not None:
        log.info(f"remote model version: v{meta.version} "
              f"(accuracy={meta.test_accuracy}, feedback={meta.feedback_count})")
    else:
        log.info("remote model version: (fallback — Supabase 에 active row 없음)")
    try:
        log.info(f"class registry: "
              f"{len(ClassRegistry.all_slugs())} total "
              f"({len(ClassRegistry.trained_slugs())} trained)")
    except Exception:  # noqa: BLE001
        log.info("class registry: (미로드 — 요청 시 재시도)")
    log.info(f"user upload collection: "
          f"{'ENABLED' if config.COLLECT_USER_UPLOADS else 'disabled'}")
    # DINOv2 미리 로드 (첫 요청 지연 회피)
    dino = get_dinov2_classifier()
    log.info(f"dinov2 classifier: "
          f"{'ENABLED' if dino.available else 'disabled (model 없음)'}")

    # 수집 정리 — 피드백 없는 업로드 7일 후 삭제 (무료 쿼터 지속성).
    # 기동 직후 1회 + 24시간 주기. 실패해도 부팅·서빙 무영향.
    async def _prune_loop() -> None:
        import asyncio  # noqa: PLC0415
        while True:
            try:
                from src.uploads import prune_stale_uploads  # noqa: PLC0415
                await asyncio.to_thread(prune_stale_uploads, 7)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"정리 실패(다음 주기 재시도): {str(exc)[:80]}")
            await asyncio.sleep(24 * 3600)

    prune_task = None
    if config.COLLECT_USER_UPLOADS:
        import asyncio  # noqa: PLC0415
        prune_task = asyncio.create_task(_prune_loop())
    yield
    if prune_task is not None:
        prune_task.cancel()
    reset_classifier()
    get_recorder.reset()


app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.CORS_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
register_exception_handlers(app)
for _r in (meta, inference, admin, learning):
    app.include_router(_r.router)
