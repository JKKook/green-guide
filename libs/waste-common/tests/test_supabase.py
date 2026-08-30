from __future__ import annotations

import pytest

from waste_common import supabase


@pytest.fixture(autouse=True)
def _reset():
    supabase.reset_client()
    yield
    supabase.reset_client()


def test_missing_credentials_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SUPABASE_URL"):
        supabase.get_client()
    assert supabase.try_get_client() is None


def test_bucket_names() -> None:
    assert str(supabase.Bucket.USER_UPLOADS) == "user-uploads"
    assert str(supabase.Bucket.MODELS) == "models"
