"""리팩토링 안전망 — 공개 인터페이스 동결 테스트.

api.py 분해(routers/services 추출) 전후로 엔드포인트 경로·태그·응답 필드가
변하지 않았음을 검증한다. 값(모델 출력)이 아니라 형태만 본다.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src import schemas

# 분해 전 스냅샷 — 경로 → (메서드, 태그). 바뀌면 의도된 인터페이스 변경인지 확인할 것.
EXPECTED_ROUTES: dict[str, tuple[str, str]] = {
    "/": ("get", "meta"),
    "/health": ("get", "meta"),
    "/design/tokens.json": ("get", "meta"),
    "/labels": ("get", "meta"),
    "/taxonomy": ("get", "meta"),
    "/region-info": ("get", "meta"),
    "/model/latest": ("get", "meta"),
    "/predict": ("post", "inference"),
    "/predict-hier": ("post", "inference"),
    "/predict-objects": ("post", "inference"),
    "/predict-centered": ("post", "inference"),
    "/predict-with-cam": ("post", "inference"),
    "/predict-with-mask": ("post", "inference"),
    "/predict-with-regions": ("post", "inference"),
    "/segment": ("post", "inference"),
    "/reload-classes": ("post", "admin"),
    "/admin/reload-model": ("post", "admin"),
    "/feedback": ("post", "learning"),
}


def test_openapi_routes_frozen(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    actual = {
        path: (method, ops["tags"][0])
        for path, methods in paths.items()
        for method, ops in methods.items()
    }
    assert actual == EXPECTED_ROUTES


@pytest.mark.parametrize(
    ("path", "model"),
    [
        ("/predict", schemas.PredictionResponse),
        ("/predict-centered", schemas.PredictionResponse),
        ("/predict-hier", schemas.PredictionHierResponse),
        ("/predict-with-regions", schemas.PredictionWithRegionsResponse),
        ("/predict-with-cam", schemas.PredictionWithCamResponse),
    ],
)
def test_predict_endpoints_response_shape(
    client: TestClient, sample_image_bytes: bytes, path: str, model: type,
) -> None:
    res = client.post(path, files={"image": ("x.jpg", sample_image_bytes, "image/jpeg")})
    assert res.status_code == 200, res.text
    assert set(res.json()) == set(model.model_fields)


def test_feedback_rejects_unconfirmed_without_label(client: TestClient) -> None:
    res = client.post(
        "/feedback",
        json={"upload_id": "00000000-0000-0000-0000-000000000000", "confirmed": False},
    )
    # 수집 비활성(503) 또는 검증 실패(400) — DB 없이도 도달 가능한 두 경로
    assert res.status_code in (400, 503), res.text


def test_project_root_points_to_repo() -> None:
    """config 가 src/core/ 로 옮겨진 뒤 PROJECT_ROOT 가 src/ 를 가리키던 회귀 방지."""
    from src.core import config
    assert (config.PROJECT_ROOT / "requirements.txt").is_file()
    assert (config.PROJECT_ROOT / "design" / "tokens.json").is_file()


def test_design_tokens_endpoint(client: TestClient) -> None:
    assert client.get("/design/tokens.json").status_code == 200
