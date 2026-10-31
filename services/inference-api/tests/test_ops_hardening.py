"""W3 서버 운영 — 관리 토큰·CORS·속도 제한·Sentry 비활성 경로.

lifespan(모델 로드)을 돌리지 않도록 `with` 없이 TestClient 를 만든다.
"""
from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient
from limits import parse as parse_rate

from src import api
from src.core import config
from src.routers import admin

TOKEN = "test-admin-token"


@pytest.fixture()
def bare_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(config, "ADMIN_TOKEN", TOKEN)
    monkeypatch.setattr(admin.ClassRegistry, "reload", classmethod(lambda cls: None))
    monkeypatch.setattr(admin.ClassRegistry, "all_slugs", classmethod(lambda cls: ["a", "b"]))
    monkeypatch.setattr(admin.ClassRegistry, "trained_slugs", classmethod(lambda cls: ["a"]))
    return TestClient(api.app)


@pytest.mark.parametrize("headers", [{}, {"X-Admin-Token": "wrong"}])
@pytest.mark.parametrize("path", ["/reload-classes", "/admin/reload-model"])
def test_admin_rejects_missing_or_wrong_token(
    bare_client: TestClient, path: str, headers: dict[str, str],
) -> None:
    assert bare_client.post(path, headers=headers).status_code == 401


def test_admin_accepts_valid_token(bare_client: TestClient) -> None:
    res = bare_client.post("/reload-classes", headers={"X-Admin-Token": TOKEN})
    assert res.status_code == 200
    assert res.json() == {"total": 2, "trained": 1}


def test_admin_rejects_all_when_server_token_unset(
    bare_client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "ADMIN_TOKEN", None)
    res = bare_client.post("/reload-classes", headers={"X-Admin-Token": ""})
    assert res.status_code == 401


def test_cors_preflight_has_no_wildcard_origin(bare_client: TestClient) -> None:
    res = bare_client.options(
        "/predict-hier",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert res.headers.get("access-control-allow-origin") != "*"
    assert "access-control-allow-credentials" not in res.headers


def test_inference_rate_limit_returns_429(
    bare_client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "_rate_item", parse_rate("2/minute"))
    api._rate_limiter.storage.reset()
    # 본문 없는 POST — 422 로 끝나도 미들웨어 계수는 그 전에 일어난다
    codes = [bare_client.post("/predict-hier").status_code for _ in range(3)]
    assert codes[:2] != [429, 429] and codes[2] == 429
    # 비대상 경로는 제한 없음
    assert bare_client.get("/health").status_code != 429
    api._rate_limiter.storage.reset()


def test_sentry_disabled_without_dsn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(config, "SENTRY_DSN", None)
    with caplog.at_level(logging.INFO, logger="src.api"):
        assert api._init_sentry() is False
    assert "sentry: disabled" in caplog.text
