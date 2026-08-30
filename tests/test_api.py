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
    # 계층 레지스트리 기반 — 고정 목록 대신 구조만 검증 (플랫 fallback 라벨 포함)
    assert isinstance(data["class_labels"], list) and len(data["class_labels"]) >= 6


def test_health_returns_ok(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "supabase_host" in body  # 진단 필드 (호스트는 공개 정보)


def test_labels_returns_hierarchy(client: TestClient) -> None:
    res = client.get("/labels")
    assert res.status_code == 200
    data = res.json()
    # 계층 taxonomy(대분류+세부) — 레지스트리 오프라인 fallback 도 6개는 넘는다
    assert data["count"] >= 6
    assert "plastic" in data["labels"]


def test_predict_with_random_image(client: TestClient, sample_image_bytes: bytes) -> None:
    res = client.post(
        "/predict",
        files={"image": ("random.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert isinstance(data["predicted_class"], str) and data["predicted_class"]
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


# ── 증거-불일치 중재 판정 (_evidence_conflicts) ─────────────────────────────
# 실사용 이격 사례: 음식물 사진 → CNN 의류 85.8% 과확신인데 CLIP 정체 증거는
# 음식물 — 이때 확신도와 무관하게 VLM 중재가 발동해야 한다.

def test_evidence_conflict_triggers_on_strong_identity_mismatch():
    from src.services.regions_service import evidence_conflicts
    f2c = {"food_waste": "food_waste", "clothes": "clothes"}
    ev = [{"type": "identity", "mapped_class": "food_waste", "score": 0.84}]
    assert evidence_conflicts(ev, "clothes", f2c) is True


def test_evidence_conflict_ignores_agreement_and_weak_or_ocr():
    from src.services.regions_service import evidence_conflicts
    f2c = {"glass_deposit": "glass", "food_waste": "food_waste"}
    # 일치 → 중재 불필요
    assert evidence_conflicts(
        [{"type": "identity", "mapped_class": "glass_deposit", "score": 0.9}],
        "glass", f2c) is False
    # 약한 증거(<0.6) → 미발동
    assert evidence_conflicts(
        [{"type": "identity", "mapped_class": "food_waste", "score": 0.53}],
        "clothes", f2c) is False
    # OCR 계열(score 는 부스트 배수) → 제외
    assert evidence_conflicts(
        [{"type": "mark", "mapped_class": "food_waste", "score": 6.0}],
        "clothes", f2c) is False
