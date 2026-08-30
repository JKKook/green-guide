"""Cascade 추론 — 손 감지 → 이진 게이트 → 13-class 분류 (+DINOv2 보정)."""
from __future__ import annotations

from src.classes import ClassRegistry
from src.core.log import get_logger
from src.dinov2_classifier import get_dinov2_classifier
from src.hand_detector import get_hand_detector
from src.inference import get_classifier
from src.preprocess import preprocess_both
from src.services.image_io import auto_crop_to_object
from src.stage1_classifier import get_stage1_classifier

log = get_logger(__name__)


def force_non_object_result(reason: str) -> dict:
    """모델 호출 없이 non_object 결과 dict 반환 (손 지배 등 OOD 강제 분기).

    [classifier.predict 와 동일 schema] — predicted_class/index/confidence/
    all_probabilities/model_arch/inference_ms.
    """
    from src.classes import ClassRegistry  # noqa: PLC0415
    labels = list(ClassRegistry.all_slugs())
    probs = {l: 0.0 for l in labels}
    if "non_object" in labels:
        non_idx = labels.index("non_object")
        probs["non_object"] = 1.0
        cls = "non_object"
    else:
        # fallback — non_object 가 DB 에 없으면 etc 로
        non_idx = labels.index("etc") if "etc" in labels else 0
        probs[labels[non_idx]] = 1.0
        cls = labels[non_idx]
    return {
        "predicted_class": cls,
        "predicted_index": non_idx,
        "confidence": 1.0,
        "all_probabilities": probs,
        "model_arch": f"hand-detected: {reason}",
        "inference_ms": 0.0,
    }


def ensemble_with_dinov2(
    resnet_result: dict, raw: bytes, w_dino: float = 0.7,
) -> dict:
    """ResNet18 결과 + DINOv2 확률 weighted average.

    ResNet18 이 OOD 입력 (예: 손 안의 객체) 에 confident-wrong 인 케이스를 보정.
    DINOv2 가 더 robust 한 표현이라 더 큰 가중치 (0.7) 부여. DINOv2 가 없거나
    실패하면 원본 resnet_result 그대로 반환.
    """
    dino_cls = get_dinov2_classifier()
    if not dino_cls.available:
        return resnet_result
    dino_out = dino_cls.predict(raw)
    if dino_out is None:
        return resnet_result

    resnet_probs = resnet_result.get("all_probabilities") or {}
    dino_probs = dino_out["confidences"]

    # 두 모델 라벨 union — non_object 가 ClassRegistry 에 없을 수 있어
    # ClassRegistry 만 쓰면 누락. 학습 라벨(manifest) 이 정본.
    all_labels = sorted(set(resnet_probs.keys()) | set(dino_probs.keys()))
    fused = {}
    for lbl in all_labels:
        r = float(resnet_probs.get(lbl, 0.0))
        d = float(dino_probs.get(lbl, 0.0))
        fused[lbl] = (1.0 - w_dino) * r + w_dino * d

    s = sum(fused.values())
    if s > 0:
        fused = {l: p / s for l, p in fused.items()}

    top_label = max(fused, key=fused.get)
    # predicted_index: ResNet 의 인덱스 체계 유지 (없으면 기존값)
    reg_labels = list(ClassRegistry.all_slugs())
    top_idx = (
        reg_labels.index(top_label) if top_label in reg_labels
        else resnet_result.get("predicted_index", 0)
    )

    return {
        **resnet_result,
        "predicted_class": top_label,
        "predicted_index": top_idx,
        "confidence": float(fused[top_label]),
        "all_probabilities": fused,
        "model_arch": f"{resnet_result.get('model_arch', '')}+dinov2-w{w_dino:.1f}",
    }


# ── 게이트: 통과면 None, 거부면 사유 문자열. 감지기 오류는 fail-open (통과) ──

def hand_gate(raw: bytes) -> str | None:
    """Stage 0 (MediaPipe Hands): 손이 50%+ 면 모델 호출 없이 non_object."""
    try:
        hand_area = get_hand_detector().hand_area_ratio(raw)
    except Exception as exc:  # noqa: BLE001 — fail-open
        log.warning(f"hand detection failed: {exc}")
        return None
    return f"hand area {hand_area:.2f} >= 0.50" if hand_area >= 0.50 else None


def stage1_gate(raw: bytes) -> str | None:
    """Stage 1 (MobileNetV3-Small binary): waste/non_object 이진 판정."""
    try:
        is_waste, waste_prob = get_stage1_classifier().predict(raw)
    except Exception as exc:  # noqa: BLE001 — fail-open: stage2 로 위임
        log.warning(f"stage1 failed: {exc}")
        return None
    return None if is_waste else f"stage1 waste_prob={waste_prob:.3f} < 0.50"


def non_object_gate(raw: bytes) -> str | None:
    """Stage 0 → Stage 1 순서로 검사. 실물 비폐기물(손바닥·마우스 등)이
    confident-wrong 으로 통과하는 것을 차단하는 공통 방어선."""
    return hand_gate(raw) or stage1_gate(raw)


def run_cascade(raw: bytes) -> dict:
    """/predict-centered 파이프라인: 게이트 → 자동 크롭 → 13-class → DINOv2 보정."""
    reason = non_object_gate(raw)
    if reason is not None:
        return force_non_object_result(reason)

    cropped_raw = auto_crop_to_object(raw)
    color_input, edge_input = preprocess_both(cropped_raw)
    result = get_classifier().predict(color_input, edge_input)

    # Stage 2.5: DINOv2 ensemble — confident-wrong 보정
    try:
        result = ensemble_with_dinov2(result, cropped_raw)
    except Exception as exc:  # noqa: BLE001 — fail-open: 보정 없이 stage2 결과 사용
        log.warning(f"dinov2 ensemble failed: {exc}")
    return result
