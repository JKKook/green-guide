"""API 엔드포인트 통합 테스트.

실제 ONNX 모델을 로드하므로 waste-classifier 의 export 완료가 전제.
"""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_root_returns_service_info(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert "name" in data and "model_arch" in data
    assert data["class_labels"] == [
        "cardboard", "glass", "metal", "paper", "plastic", "trash",
    ]


def test_health_returns_ok(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_labels_returns_six_classes(client: TestClient) -> None:
    res = client.get("/labels")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 6
    assert "plastic" in data["labels"]


def test_predict_with_random_image(client: TestClient, sample_image_bytes: bytes) -> None:
    res = client.post(
        "/predict",
        files={"image": ("random.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["predicted_class"] in {
        "cardboard", "glass", "metal", "paper", "plastic", "trash",
    }
    assert 0.0 <= data["confidence"] <= 1.0
    assert abs(sum(data["all_probabilities"].values()) - 1.0) < 1e-4
    assert data["inference_ms"] > 0


def test_predict_with_real_cardboard_sample(
    client: TestClient, real_sample_image_bytes: bytes | None,
) -> None:
    """실제 cardboard 샘플은 cardboard 로 분류되어야 함 (CNN 92% 정확도 기준)."""
    if real_sample_image_bytes is None:
        import pytest
        pytest.skip("waste-preprocessor의 실제 샘플 이미지 없음")

    res = client.post(
        "/predict",
        files={"image": ("cardboard1.jpg", real_sample_image_bytes, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    # 100% 보장은 아니지만 CNN 이라면 cardboard 가 가장 높은 확률을 가질 것
    assert data["predicted_class"] == "cardboard", (
        f"기대: cardboard, 실제: {data['predicted_class']} "
        f"(probabilities: {data['all_probabilities']})"
    )


def test_predict_rejects_invalid_content_type(client: TestClient) -> None:
    res = client.post(
        "/predict",
        files={"image": ("file.txt", b"hello", "text/plain")},
    )
    assert res.status_code == 415


def test_predict_rejects_invalid_image_bytes(client: TestClient) -> None:
    res = client.post(
        "/predict",
        files={"image": ("fake.jpg", b"this is not an image", "image/jpeg")},
    )
    assert res.status_code == 400


def test_predict_rejects_empty_file(client: TestClient) -> None:
    res = client.post(
        "/predict",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert res.status_code == 400
