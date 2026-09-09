"""업로드 메타 수신·기록 (feature/api-upload-meta).

/predict-hier 의 orientation 폼 필드가 TTA 후보를 축소하고, 촬영 메타가
user_uploads 기록에 전달·저장(컬럼 미배포 시 fail-open)되는지 검증.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.uploads import UploadRecorder


# ── 라우터: orientation → degs 축소 + meta 전달 ──────────────────────────

HIER_RESULT = {
    "display_level": "fine", "display_class": "metal", "coarse_class": "metal",
    "coarse_confidence": 0.95, "fine_class": "metal", "fine_confidence": 0.95,
    "fine_margin": 0.9, "coarse_probabilities": {"metal": 0.95},
    "fine_top5": [], "model_arch": "test", "inference_ms": 1.0,
    "tta_rotation": 90,
}


@pytest.fixture()
def hier_stub(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """모델 없이 /predict-hier 배선만 검증 — 게이트·분류기·기록을 스텁."""
    captured: dict[str, Any] = {}
    import src.hier_inference as hi
    import src.routers.inference as ri

    def fake_predict_rotations(clf, raw, degs, **kw):
        captured["degs"] = degs
        return dict(HIER_RESULT), np.zeros((1, 3, 448, 448), dtype=np.float32)

    def fake_record_safely(raw, image, prediction, meta=None):
        captured["meta"] = meta
        return "test-upload-id"

    from types import SimpleNamespace
    stub_clf = SimpleNamespace(
        taxonomy={"fine_to_coarse": {}}, fine_labels=[],
        predict=lambda *a, **kw: dict(HIER_RESULT),  # prior 재예측 경로용
    )
    monkeypatch.setattr(hi, "get_hier_classifier", lambda: stub_clf)
    monkeypatch.setattr(hi, "predict_rotations", fake_predict_rotations)
    monkeypatch.setattr(ri, "non_object_gate", lambda raw: None)
    monkeypatch.setattr(ri, "record_safely", fake_record_safely)
    return captured


def _post_hier(client: TestClient, image: bytes, **form: str) -> dict:
    res = client.post("/predict-hier",
                      files={"image": ("x.jpg", image, "image/jpeg")}, data=form)
    assert res.status_code == 200, res.text
    return res.json()


def test_orientation_field_reduces_tta(client: TestClient, sample_image_bytes: bytes,
                                       hier_stub: dict) -> None:
    _post_hier(client, sample_image_bytes, orientation="6")
    assert hier_stub["degs"] == (0, 90)  # 태그 6 → 2후보


def test_orientation_absent_keeps_default(client: TestClient, sample_image_bytes: bytes,
                                          hier_stub: dict) -> None:
    _post_hier(client, sample_image_bytes)
    # 샘플엔 EXIF 없음(tag=1) → 전수 3방향 유지
    assert hier_stub["degs"] == (0, 90, 270)


def test_meta_reaches_recorder(client: TestClient, sample_image_bytes: bytes,
                               hier_stub: dict) -> None:
    body = _post_hier(client, sample_image_bytes, orientation="3",
                      capture_mode="smart", quality_blur="0.12",
                      quality_brightness="0.8", crop_applied="true",
                      crop_box="0.1,0.2,0.5,0.6", tap_x="0.4", tap_y="0.7")
    assert body["upload_id"] == "test-upload-id"
    meta = hier_stub["meta"]
    assert meta["orientation"] == 3
    assert meta["capture_mode"] == "smart"
    assert meta["quality_blur"] == pytest.approx(0.12)
    assert meta["quality_brightness"] == pytest.approx(0.8)
    assert meta["crop_applied"] is True
    assert meta["crop_box"] == "0.1,0.2,0.5,0.6"
    assert meta["exif_orientation"] == 1          # 서버가 읽은 태그 (샘플엔 없음)
    assert meta["tta_rotation"] == 90             # 채택 회전
    assert meta["tap_x"] == pytest.approx(0.4)
    assert meta["tap_y"] == pytest.approx(0.7)


def test_invalid_capture_mode_rejected(client: TestClient,
                                       sample_image_bytes: bytes) -> None:
    res = client.post("/predict-hier",
                      files={"image": ("x.jpg", sample_image_bytes, "image/jpeg")},
                      data={"capture_mode": "video"})
    assert res.status_code == 422


# ── 기록: meta 병합·컬럼 없음 fail-open ──────────────────────────────────

class _FakeTable:
    def __init__(self, log: list, fail_first: bool):
        self._log, self._fail_first = log, fail_first

    def insert(self, row):
        self._log.append(row)
        outer = self

        class _Exec:
            def execute(self):
                if outer._fail_first and len(outer._log) == 1:
                    raise RuntimeError("column \"orientation\" does not exist")
        return _Exec()


class _FakeClient:
    def __init__(self, inserts: list, fail_first: bool = False):
        self._inserts, self._fail_first = inserts, fail_first

    def table(self, name):
        return _FakeTable(self._inserts, self._fail_first)

    @property
    def storage(self):
        class _S:
            def from_(self, bucket):
                class _B:
                    def upload(self, **kw): pass
                    def get_public_url(self, p): return "http://x/" + p
                return _B()
        return _S()


def _recorder_with(client) -> UploadRecorder:
    r = UploadRecorder.__new__(UploadRecorder)
    r.client = client
    return r


PRED = {"predicted_class": "metal", "confidence": 0.9, "all_probabilities": {},
        "model_arch": "test", "inference_ms": 1.0}


def test_remote_record_merges_meta_and_drops_none() -> None:
    inserts: list[dict] = []
    r = _recorder_with(_FakeClient(inserts))
    r.record_prediction(b"img", "image/jpeg", dict(PRED),
                        meta={"orientation": 6, "capture_mode": None, "tap_x": 0.5})
    assert len(inserts) == 1
    assert inserts[0]["orientation"] == 6
    assert inserts[0]["tap_x"] == 0.5
    assert "capture_mode" not in inserts[0]  # None 은 저장하지 않음


def test_remote_record_retries_without_meta_columns() -> None:
    """컬럼 미배포(마이그레이션 전)면 기본 컬럼만으로 재시도 — 기록 자체는 성공."""
    inserts: list[dict] = []
    r = _recorder_with(_FakeClient(inserts, fail_first=True))
    upload_id = r.record_prediction(b"img", "image/jpeg", dict(PRED),
                                    meta={"orientation": 6})
    assert len(inserts) == 2
    assert "orientation" in inserts[0] and "orientation" not in inserts[1]
    assert not upload_id.startswith("local-")  # 로컬 폴백 아님


def test_local_record_includes_meta(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    import src.uploads as up
    monkeypatch.setattr(up, "_LOCAL_DIR", tmp_path)
    uid = UploadRecorder._local_record(b"img", ".jpg", dict(PRED),
                                       meta={"tta_rotation": 90, "crop_box": None})
    row = json.loads((tmp_path / "meta.jsonl").read_text().strip())
    assert row["id"] == uid
    assert row["tta_rotation"] == 90
    assert "crop_box" not in row
