"""메타 정보 — 서비스/헬스/라벨/택소노미/지역 규칙/모델 버전."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from src.classes import ClassRegistry
from src.core import config
from src.core.log import get_logger
from src.inference import get_active_meta, get_classifier
from src.schemas import (
    HealthResponse,
    LabelsResponse,
    ModelVersionResponse,
    ServiceInfo,
    TaxonomyResponse,
)

log = get_logger(__name__)
router = APIRouter()


@router.get("/", response_model=ServiceInfo, tags=["meta"])
def root() -> ServiceInfo:
    classifier = get_classifier()
    return ServiceInfo(
        name=config.API_TITLE,
        version=config.API_VERSION,
        model_arch=classifier.arch,
        model_path=str(classifier.model_path),
        class_labels=ClassRegistry.trained_slugs(),
        max_upload_size_bytes=config.MAX_UPLOAD_SIZE_BYTES,
    )


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    host = None
    try:
        from urllib.parse import urlparse  # noqa: PLC0415
        u = config.SUPABASE_URL or ""
        host = urlparse(u).hostname if u else None
    except Exception:  # noqa: BLE001
        host = None
    return HealthResponse(supabase_host=host)


@router.get("/design/tokens.json", tags=["meta"])
def design_tokens() -> dict:
    """디자인 토큰 (W3C Design Tokens draft) — 앱 실측값.

    출처: waste_app app_theme.dart · confidence.dart · waste_info.dart.
    디자인 도구(Figma Tokens/style-dictionary)·시안 문서가 URL 로 소비.
    """
    import json  # noqa: PLC0415
    p = config.PROJECT_ROOT / "design" / "tokens.json"
    return json.loads(p.read_text(encoding="utf-8"))


@router.get("/labels", response_model=LabelsResponse, tags=["meta"])
def labels() -> LabelsResponse:
    """전체 클래스 목록 (학습된 것 + 신규 미학습) + 메타데이터."""
    classes = ClassRegistry.load()
    return LabelsResponse(
        labels=[c.slug for c in classes],
        count=len(classes),
        classes=[c.to_api_dict() for c in classes],
    )


@router.get("/taxonomy", response_model=TaxonomyResponse, tags=["meta"])
def taxonomy() -> TaxonomyResponse:
    """계층 taxonomy 메타 — 대분류/세부 라벨, 롤업 매핑, 게이트 임계.

    계층 모델 미배치 시 404 (앱은 flat 모드로 fallback).
    """
    from src.hier_inference import get_hier_classifier  # noqa: PLC0415
    try:
        clf = get_hier_classifier()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    t = clf.taxonomy
    return TaxonomyResponse(
        version=t.get("version", "?"),
        fine_labels=t["fine_labels"],
        coarse_labels=t["coarse_labels"],
        fine_to_coarse=t["fine_to_coarse"],
        gate=t["gate"],
    )


@router.get("/region-info", tags=["meta"])
def region_info(sido: str, sigungu: str) -> dict:
    """지역별 생활쓰레기 배출 규정 — 앱 지역 선택 시나리오의 데이터 소스.

    Supabase region_waste_rules (공공데이터포털 전국생활쓰레기배출정보 표준데이터,
    scripts/load_region_rules.py 적재) 조회. 데이터 미적재/오프라인이어도
    빈 목록으로 응답 — 앱은 전국 공통 안내로 fallback.
    """
    try:
        from src.uploads import _client as _supabase_client  # noqa: PLC0415
        client = _supabase_client()
        res = (client.table("region_waste_rules")
               .select("*")
               .eq("sido", sido)
               .eq("sigungu", sigungu)
               .limit(50)
               .execute())
        rules = res.data or []
    except Exception as exc:  # noqa: BLE001
        log.warning(f"region-info 조회 실패 (빈 응답): {exc}")
        rules = []
    return {"sido": sido, "sigungu": sigungu, "count": len(rules), "rules": rules}


@router.get("/model/latest", response_model=ModelVersionResponse, tags=["meta"])
def model_latest() -> ModelVersionResponse:
    """현재 서비스가 사용 중인 모델 버전 메타데이터.

    Flutter 앱이 부팅 시 호출 → 자신의 캐시 버전과 비교 → 더 새 게 있으면
    color_url / edge_url 로 직접 다운로드.
    """
    meta = get_active_meta()
    if meta is None:
        return ModelVersionResponse(is_fallback=True)
    return ModelVersionResponse(
        version=meta.version,
        color_url=meta.color_url,
        edge_url=meta.edge_url,
        color_sha256=meta.color_sha256,
        edge_sha256=meta.edge_sha256,
        test_accuracy=meta.test_accuracy,
        num_classes=meta.num_classes,
        class_labels=meta.class_labels,
        feedback_count=meta.feedback_count,
        is_fallback=False,
    )
