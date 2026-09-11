"""/predict-hier want_cam — 결과 카드와 같은 텐서·prior 로 CAM 을 만들어 cam_base64 로 반환."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

HIER_RESULT = {
    "display_level": "fine", "display_class": "metal", "coarse_class": "metal",
    "coarse_confidence": 0.95, "fine_class": "metal", "fine_confidence": 0.95,
    "fine_margin": 0.9, "coarse_probabilities": {"metal": 0.95},
    "fine_top5": [], "model_arch": "test", "inference_ms": 1.0,
}


@pytest.fixture()
def hier_cam_stub(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {"predict_calls": []}
    import src.hier_inference as hi
    import src.routers.inference as ri

    best = np.zeros((1, 3, 224, 224), dtype=np.float32)

    def fake_predict_rotations(clf, raw, degs, **kw):
        return dict(HIER_RESULT), best

    def fake_predict(tensor, want_cam=False, **kw):
        captured["predict_calls"].append(
            {"want_cam": want_cam, "same_tensor": tensor is best, **kw})
        out = dict(HIER_RESULT)
        if want_cam:
            out["cam"] = np.random.default_rng(0).random((7, 7)).astype(np.float32)
        return out

    stub_clf = SimpleNamespace(
        taxonomy={"fine_to_coarse": {}}, fine_labels=[], predict=fake_predict)
    monkeypatch.setattr(hi, "get_hier_classifier", lambda: stub_clf)
    monkeypatch.setattr(hi, "predict_rotations", fake_predict_rotations)
    monkeypatch.setattr(ri, "non_object_gate", lambda raw: None)
    monkeypatch.setattr(ri, "record_safely", lambda *a, **kw: "test-upload-id")
    return captured


def _post(client: TestClient, image: bytes, **form: str) -> dict:
    res = client.post("/predict-hier", files={"image": ("x.jpg", image, "image/jpeg")}, data=form)
    assert res.status_code == 200, res.text
    return res.json()


def test_want_cam_returns_overlay_from_same_tensor(
        client: TestClient, sample_image_bytes: bytes, hier_cam_stub: dict[str, Any]) -> None:
    body = _post(client, sample_image_bytes, want_cam="true")
    assert body["cam_base64"] is not None and body["cam_base64"].startswith("data:image/png;base64,")
    cam_calls = [c for c in hier_cam_stub["predict_calls"] if c["want_cam"]]
    assert len(cam_calls) == 1 and cam_calls[0]["same_tensor"]
    # 카드 결과는 그대로 (CAM 계산이 표시 결과를 바꾸지 않음)
    assert body["display_class"] == "metal" and "cam" not in body


def test_without_want_cam_no_cam(client: TestClient, sample_image_bytes: bytes,
                                 hier_cam_stub: dict[str, Any]) -> None:
    body = _post(client, sample_image_bytes)
    assert body["cam_base64"] is None
    assert not any(c["want_cam"] for c in hier_cam_stub["predict_calls"])
