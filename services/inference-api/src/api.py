"""FastAPI app 정의."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from limits import parse as parse_rate
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter

from src.classes import ClassRegistry
from src.core import config
from src.core.errors import register_exception_handlers
from src.core.log import get_logger
from src.dinov2_classifier import get_dinov2_classifier
from src.inference import get_active_meta, get_classifier, reset_classifier
from src.routers import admin, inference, learning, meta
from src.uploads import get_recorder

log = get_logger(__name__)


def _init_sentry() -> bool:
    """GREENGUIDE_SENTRY_DSN 있으면 Sentry 활성 (FastAPI·Starlette·logging 통합은 자동).
    SENTRY_WARNING_LOGGERS 의 WARNING 도 이벤트로 올린다 (기본 logging 통합은 ERROR 만)."""
    if not config.SENTRY_DSN:
        log.info("sentry: disabled (GREENGUIDE_SENTRY_DSN 없음)")
        return False
    import sentry_sdk  # noqa: PLC0415
    from sentry_sdk.integrations.logging import EventHandler  # noqa: PLC0415

    sentry_sdk.init(
        dsn=config.SENTRY_DSN,
        environment=config.SENTRY_ENVIRONMENT,
        release=f"greenguide-api@{config.API_VERSION}",
        traces_sample_rate=config.SENTRY_TRACES_SAMPLE_RATE,
        send_default_pii=False,
    )
    for name in config.SENTRY_WARNING_LOGGERS:
        logging.getLogger(name).addHandler(EventHandler(level=logging.WARNING))
    log.info(f"sentry: ENABLED (env={config.SENTRY_ENVIRONMENT})")
    return True


_init_sentry()


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
    # 계층 모델 사전 로드 — 생성자가 check_ood_calibration() 을 부르므로 OOD 보정값 누락
    # 경고가 첫 요청이 아닌 기동 시점에 뜬다 (logger src.hier_inference → Sentry 로도 나감)
    try:
        from src.hier_inference import get_hier_classifier  # noqa: PLC0415
        get_hier_classifier()
    except Exception as exc:  # noqa: BLE001
        log.warning(f"hier classifier 사전 로드 실패(요청 시 재시도): {exc}")

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

# 추론 엔드포인트 IP 기준 속도 제한 — 라우터(W2 영역)를 건드리지 않도록 경로 접두어 미들웨어.
# 메모리 저장소 = 프로세스(worker) 단위 계수. 현 배포는 worker 1.
_rate_item = parse_rate(config.RATE_LIMIT_INFERENCE) if config.RATE_LIMIT_INFERENCE else None
_rate_limiter = MovingWindowRateLimiter(MemoryStorage())


@app.middleware("http")
async def _rate_limit_inference(request: Request, call_next):
    if (
        _rate_item is not None
        and request.method == "POST"
        and request.url.path.startswith(config.RATE_LIMITED_PATHS)
    ):
        ip = request.client.host if request.client else "unknown"
        if not _rate_limiter.hit(_rate_item, "inference", ip):
            return JSONResponse(
                status_code=429,
                content={"detail": f"rate limit exceeded: {config.RATE_LIMIT_INFERENCE}"},
                headers={"Retry-After": str(_rate_item.get_expiry())},
            )
    return await call_next(request)


# 앱은 CORS 무관 — 허용 출처는 명시 목록만(기본 빈 목록), 자격증명 미허용.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.CORS_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
register_exception_handlers(app)
for _r in (meta, inference, admin, learning):
    app.include_router(_r.router)
