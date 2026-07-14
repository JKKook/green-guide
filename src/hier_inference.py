"""계층(cnn_hier) ONNX 추론 — fine 예측 + 대분류 롤업 + 신뢰도 게이트.

waste-classifier 의 hier_export 산출물(classifier.onnx + taxonomy.json)을
로드한다. taxonomy 사이드카가 fine→coarse 매핑과 게이트 임계를 제공하므로
DB 없이도 계층 응답이 가능하다.

게이트 (GREENGUIDE_BLUEPRINT.md §2.3):
  세부 top1 ≥ fine_min_confidence AND (top1-top2) ≥ fine_min_margin
      → 세부까지 표시 (display=fine)
  elif 대분류 확률 ≥ coarse_min_confidence
      → 대분류만 표시 (display=coarse)
  else → reject (display=etc)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from src.inference import _softmax

# 모델 경로 해석: env → 번들 → 자매 레포 (기존 config.MODEL_PATH 관례와 동일)
_ENV_PATH = os.getenv("WASTE_API_HIER_MODEL_PATH")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CANDIDATES = [
    Path(_ENV_PATH) if _ENV_PATH else None,
    _PROJECT_ROOT / "models" / "classifier_hier.onnx",
    _PROJECT_ROOT.parent / "waste-classifier" / "outputs" / "models" / "cnn_hier" / "classifier.onnx",
]


def _resolve_hier_paths() -> tuple[Path, Path]:
    for cand in _CANDIDATES:
        if cand and cand.exists():
            sidecar = cand.parent / "taxonomy.json"
            if not sidecar.exists():
                raise FileNotFoundError(f"taxonomy.json 이 {cand.parent} 에 없음")
            return cand, sidecar
    raise FileNotFoundError(
        "계층 ONNX 를 찾을 수 없음 — waste-classifier 에서 "
        "`python -m src.hier_export` 를 먼저 실행하거나 "
        "WASTE_API_HIER_MODEL_PATH 를 설정하세요."
    )


class HierWasteClassifier:
    """fine logits → (fine, coarse 롤업, 게이트 판정)."""

    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path, sidecar_path = _resolve_hier_paths()
        else:
            sidecar_path = model_path.parent / "taxonomy.json"
        self.model_path = model_path
        self.taxonomy = json.loads(sidecar_path.read_text(encoding="utf-8"))
        self.fine_labels: list[str] = self.taxonomy["fine_labels"]
        self.coarse_labels: list[str] = self.taxonomy["coarse_labels"]
        self.f2c_idx: list[int] = self.taxonomy["fine_idx_to_coarse_idx"]
        self.gate: dict[str, float] = self.taxonomy["gate"]

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"],
        )

        # DINOv2 계층 앙상블 (선택) — build_dinov2_hier_head.py 산출물.
        # 실측: frozen +0.8pp, 실사용 52.9→60.8% (+7.8pp, confident-wrong 보정).
        self.dino_session: ort.InferenceSession | None = None
        self.dino_weight = 0.5   # frozen 그리드 탐색 최적값
        for cand in (model_path.parent / "dinov2_hier.onnx",
                     Path(__file__).resolve().parent.parent / "models" / "dinov2_hier.onnx"):
            if cand.exists():
                try:
                    self.dino_session = ort.InferenceSession(
                        str(cand), providers=["CPUExecutionProvider"])
                    print(f"[hier] dinov2 앙상블 활성: {cand.name} (w={self.dino_weight})")
                except Exception as exc:  # noqa: BLE001
                    print(f"[hier] dinov2 로드 실패(단독 모드): {exc}")
                break

        # OOD 프로토타입 (선택) — build_hier_prototypes.py 산출물.
        # softmax 는 OOD 에 과신하므로 임베딩 거리로 '학습된 무엇과도 안 닮음'을 잡는다.
        self.ood_protos: np.ndarray | None = None   # (C, 512) L2-normalized
        # 2단 임계 (τ 분석 근거: val p90=0.217/p95=0.320, noise=0.335, gray=0.353)
        #  - soft: 넘으면 세부 표시 억제(대분류 캡) — val 오거부 ~5.6%지만 안내는 유지
        #  - hard: 넘으면 완전 reject — 극단 OOD 만
        ood_cfg = self.taxonomy.get("ood") or {}
        self.ood_tau_soft: float = float(ood_cfg.get("tau_soft", 0.30))
        self.ood_tau_hard: float = float(ood_cfg.get("tau_hard", 0.40))
        ood_path = model_path.parent / "ood.npz"
        if ood_path.exists():
            data = np.load(ood_path, allow_pickle=False)
            self.ood_protos = data["prototypes"]

    def cam_hires(self, color_input_hi: np.ndarray) -> np.ndarray | None:
        """고해상 CAM — 448² 입력 forward 로 (C, 14, 14) 재질 증거 지도.

        분류(logits)는 학습 해상도 224 경로를 신뢰하고, 이 출력은 재질
        영역 분석 전용 (CAM_MATERIAL_UPGRADE_PLAN Stage 1-1).
        구버전(고정 224) ONNX 면 None — 호출부 7×7 fallback.
        """
        try:
            (cam,) = self.session.run(["cam"], {"image": color_input_hi})
            return cam[0]  # (C, h, w)
        except Exception as exc:  # noqa: BLE001
            print(f"[hier] hi-res cam 미지원(구 ONNX?): {exc}")
            return None

    def material_class_indices(self) -> list[int]:
        """재질 후보 fine 인덱스 — non_object/etc 는 재질이 아니므로 셀 경쟁 제외."""
        skip = {"non_object", "etc"}
        return [i for i, s in enumerate(self.fine_labels) if s not in skip]

    def _rollup(self, fine_probs: np.ndarray) -> np.ndarray:
        """(C_fine,) → (C_coarse,) 확률 합산."""
        coarse = np.zeros(len(self.coarse_labels), dtype=fine_probs.dtype)
        for fi, ci in enumerate(self.f2c_idx):
            coarse[ci] += fine_probs[fi]
        return coarse

    def predict(
        self,
        color_input: np.ndarray,
        want_cam: bool = False,
        mask_non_object: bool = False,
    ) -> dict[str, Any]:
        """(1,3,224,224) 입력 → 계층 예측 dict.

        mask_non_object: Stage1 이진 게이트가 이미 '폐기물'로 판정한 경우 True.
        non_object 는 게이트와 모순되는 답이므로 로짓에서 제외 — 실사용 잡배경
        사진이 non_object 로 새는 것을 차단 (실측 대분류 +5.9pp).
        """
        t0 = time.perf_counter()
        need_emb = self.ood_protos is not None
        outputs = ["logits"]
        if want_cam:
            outputs.append("cam")
        if need_emb:
            outputs.append("embedding")
        res = self.session.run(outputs, {"image": color_input})
        logits = res[0]

        # DINOv2 앙상블 — softmax 확률 가중합 (마스킹 전 단계에서 결합)
        dino_probs: np.ndarray | None = None
        if self.dino_session is not None:
            try:
                (dl,) = self.dino_session.run(["logits"], {"image": color_input})
                e = np.exp(dl - dl.max(axis=1, keepdims=True))
                dino_probs = e / e.sum(axis=1, keepdims=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[hier] dinov2 추론 실패(단독 진행): {exc}")
        if mask_non_object and "non_object" in self.fine_labels:
            logits = logits.copy()
            logits[:, self.fine_labels.index("non_object")] = -1e9
            if dino_probs is not None:
                # dino 확률에도 동일 마스킹 후 재정규화
                dino_probs = dino_probs.copy()
                dino_probs[:, self.fine_labels.index("non_object")] = 0.0
                dino_probs = dino_probs / dino_probs.sum(axis=1, keepdims=True)

        # OOD 거리 — 최근접 prototype cosine distance (2단 판정)
        ood_distance: float | None = None
        ood_soft = False   # 세부 억제 (대분류 캡)
        ood_reject = False  # 완전 reject
        if need_emb:
            emb = res[-1][0]
            emb = emb / max(float(np.linalg.norm(emb)), 1e-9)
            ood_distance = float(1.0 - (self.ood_protos @ emb).max())
            ood_soft = ood_distance > self.ood_tau_soft
            ood_reject = ood_distance > self.ood_tau_hard
        fine_probs = _softmax(logits)[0]                     # (C_fine,)
        if dino_probs is not None:
            w = self.dino_weight
            fine_probs = (1.0 - w) * fine_probs + w * dino_probs[0]
        coarse_probs = self._rollup(fine_probs)              # (C_coarse,)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        fi = int(fine_probs.argmax())
        ci = int(coarse_probs.argmax())
        fine_top1 = float(fine_probs[fi])
        fine_top2 = float(np.partition(fine_probs, -2)[-2])
        coarse_top1 = float(coarse_probs[ci])

        # 게이트: 표시 깊이 결정
        fine_ok = (
            fine_top1 >= self.gate["fine_min_confidence"]
            and (fine_top1 - fine_top2) >= self.gate["fine_min_margin"]
        )
        coarse_ok = coarse_top1 >= self.gate["coarse_min_confidence"]
        if ood_soft:
            # 분포 경계 밖 — 세부는 억제, 대분류 안내는 유지 (soft)
            fine_ok = False
        if ood_reject:
            # 극단 OOD — softmax 확신과 무관하게 완전 reject (hard)
            coarse_ok = False
        if fine_ok and coarse_ok:
            display_level, display_class = "fine", self.fine_labels[fi]
        elif coarse_ok:
            display_level, display_class = "coarse", self.coarse_labels[ci]
        else:
            display_level, display_class = "reject", "etc"

        result: dict[str, Any] = {
            "display_level": display_level,
            "display_class": display_class,
            "coarse_class": self.coarse_labels[ci],
            "coarse_confidence": round(coarse_top1, 4),
            "fine_class": self.fine_labels[fi] if fine_ok else None,
            "fine_confidence": round(fine_top1, 4),
            "fine_margin": round(fine_top1 - fine_top2, 4),
            "coarse_probabilities": {
                self.coarse_labels[i]: float(coarse_probs[i])
                for i in range(len(self.coarse_labels))
            },
            "fine_top5": [
                {"slug": self.fine_labels[i], "prob": float(fine_probs[i])}
                for i in np.argsort(fine_probs)[::-1][:5]
            ],
            "model_arch": f"cnn_hier ({self.taxonomy.get('version', '?')})",
            "inference_ms": round(elapsed_ms, 2),
            "ood_distance": round(ood_distance, 4) if ood_distance is not None else None,
            "ood_reject": ood_reject,
        }
        if want_cam:
            cam_all = res[1]  # (1, C_fine, 7, 7)
            result["cam"] = cam_all[0, fi]
        return result


_hier: HierWasteClassifier | None = None


def get_hier_classifier() -> HierWasteClassifier:
    global _hier
    if _hier is None:
        _hier = HierWasteClassifier()
    return _hier


def reset_hier_classifier() -> None:
    global _hier
    _hier = None
