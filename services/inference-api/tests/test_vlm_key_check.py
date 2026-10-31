"""VLM 폴백 키 점검 — 없음/무효/유효 3경로. 네트워크 없이 anthropic 클라이언트를 모의한다."""
from __future__ import annotations

import logging

import anthropic
import httpx
import pytest

from src import vlm_fallback as vf
from src.core import config


def _status_error(cls: type, code: int) -> Exception:
    req = httpx.Request("GET", "https://api.anthropic.com/v1/models")
    return cls("x", response=httpx.Response(code, request=req), body=None)


class _FakeClient:
    """models.list 결과만 바꿔 끼우는 최소 모의 클라이언트."""

    exc: Exception | None = None
    calls = 0

    def __init__(self, api_key: str) -> None:  # noqa: ARG002
        self.models = self

    def with_options(self, **_: object) -> _FakeClient:
        return self

    def list(self, **_: object) -> list:
        type(self).calls += 1
        if type(self).exc is not None:
            raise type(self).exc
        return []


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClient]:
    _FakeClient.exc, _FakeClient.calls = None, 0
    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")
    return _FakeClient


def test_missing_key_disables_without_check(monkeypatch: pytest.MonkeyPatch,
                                            fake: type[_FakeClient]) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    v = vf.VlmFallback()
    assert (v.available, v.key_status, fake.calls) == (False, "missing", 0)


def test_invalid_key_disables_with_warning(fake: type[_FakeClient],
                                           caplog: pytest.LogCaptureFixture) -> None:
    fake.exc = _status_error(anthropic.AuthenticationError, 401)
    with caplog.at_level(logging.WARNING):
        v = vf.VlmFallback()
    assert (v.available, v.key_status, fake.calls) == (False, "invalid", 1)
    assert any(r.levelno == logging.WARNING and "키 무효" in r.getMessage()
               for r in caplog.records)
    assert "test-key" not in caplog.text
    assert v.classify(b"", [], {}) is None


def test_valid_key_enables(fake: type[_FakeClient]) -> None:
    v = vf.VlmFallback()
    assert (v.available, v.key_status, fake.calls) == (True, "ok", 1)


def test_network_error_keeps_enabled_unverified(fake: type[_FakeClient]) -> None:
    fake.exc = httpx.ConnectError("down")
    v = vf.VlmFallback()
    assert (v.available, v.key_status) == (True, "unverified")


def test_health_exposes_vlm_status_without_key(client) -> None:  # noqa: ANN001
    body = client.get("/health").json()
    assert {"vlm_available", "vlm_model", "vlm_key_status"} <= body.keys()
    assert body["vlm_key_status"] in {"missing", "invalid", "unverified", "ok"}
    if config.ANTHROPIC_API_KEY:
        assert config.ANTHROPIC_API_KEY not in str(body)
