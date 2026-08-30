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
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from src.inference import _softmax
from src.core import config
from src.core.log import get_logger
from src.core.singleton import lazy_singleton

log = get_logger(__name__)

# 모델 경로 해석: env → 번들 → 자매 레포 (기존 config.MODEL_PATH 관례와 동일)
_ENV_PATH = config.HIER_MODEL_PATH_ENV
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CANDIDATES = [
    Path(_ENV_PATH) if _ENV_PATH else None,
    _PROJECT_ROOT / "models" / "classifier_hier.onnx",
    _PROJECT_ROOT.parent / "waste-classifier" / "outputs" / "models" / "cnn_hier"
    / "classifier.onnx",
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

        # DINOv2 계층 앙상블 — 기본 비활성 (청사진 v2 트랙 B4, 2026-07-21).
        # 순수 홀드아웃 20장 실측: 기여 0 (solo 11/20 = 앙상블 11/20; 이전 +2는
        # 오염 표본 암기 효과). 원칙 "실측 이득 없으면 제거" — 추론 ~2× 단축.
        # 재활성: env WASTE_API_DINO_W=0.3 (홀드아웃 커지면 트랙 A3 재스윕)
        self.dino_session: ort.InferenceSession | None = None
        self.dino_weight = config.DINO_WEIGHT
        if self.dino_weight > 0:
            for cand in (model_path.parent / "dinov2_hier.onnx",
                         Path(__file__).resolve().parent.parent / "models" / "dinov2_hier.onnx"):
                if cand.exists():
                    try:
                        self.dino_session = ort.InferenceSession(
                            str(cand), providers=["CPUExecutionProvider"])
                        log.info(f"dinov2 앙상블 활성: {cand.name} (w={self.dino_weight})")
                    except Exception as exc:  # noqa: BLE001
                        log.info(f"dinov2 로드 실패(단독 모드): {exc}")
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
            log.info(f"hi-res cam 미지원(구 ONNX?): {exc}")
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
        fine_prior: np.ndarray | None = None,
        ood_relax: bool = False,
    ) -> dict[str, Any]:
        """(1,3,224,224) 입력 → 계층 예측 dict.

        ood_relax: 탭-투-셀렉트 경로 True. 탭 크롭은 물체 외곽이 잘려 텍스처만
        남기 쉬워 OOD 거리가 전체 장면 대비 크게 튄다 (실측 0.25→0.54, tau_hard
        0.40 초과 → 과잉 하드 reject). 사용자가 지목했다 = '여기에 물체가 있다'
        신호이므로 하드 reject 를 소프트 억제(대분류 표시)로 완화. non_object
        최종 판정은 VLM 폴백이 담당.

        mask_non_object: Stage1 이진 게이트가 이미 '폐기물'로 판정한 경우 True.
        non_object 는 게이트와 모순되는 답이므로 로짓에서 제외 — 실사용 잡배경
        사진이 non_object 로 새는 것을 차단 (실측 대분류 +5.9pp).
        fine_prior: (C_fine,) 시맨틱 증거 승수 (semantic_evidence.evidence_prior).
        log-linear 융합 — 앙상블 확률에 곱한 뒤 재정규화. 게이트 이전에 적용되므로
        강한 텍스트 증거는 자연히 reject 를 푼다 (SEMANTIC_FUSION_PLAN §3).
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
                log.info(f"dinov2 추론 실패(단독 진행): {exc}")
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
            ood_reject = (not ood_relax) and ood_distance > self.ood_tau_hard
        fine_probs = _softmax(logits)[0]                     # (C_fine,)
        if dino_probs is not None:
            w = self.dino_weight
            fine_probs = (1.0 - w) * fine_probs + w * dino_probs[0]
        if fine_prior is not None:
            fine_probs = fine_probs * fine_prior.astype(fine_probs.dtype)
            fine_probs = fine_probs / max(float(fine_probs.sum()), 1e-12)
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


# 탭 경로 CAM 융합 가중 — 실사용 51장(크롭 대리) 실측: 0.15 는 무해(29 유지),
# 0.2+ 부터 자기강화로 -1~-4. 같은 CNN 파생 신호라 보조 역할에 한정 (env 조정 가능).
CAM_PRIOR_WEIGHT = config.CAM_PRIOR_WEIGHT


def cam_region_prior(
    clf: HierWasteClassifier,
    raw: bytes,
    region: list[float],
    weight: float = CAM_PRIOR_WEIGHT,
    min_share: float = 0.08,
) -> np.ndarray | None:
    """탭 영역의 hi-res CAM 재질 증거 → fine prior (신호④, 탭 경로 전용).

    사용자가 지목한 영역에 대해 "전체 프레임 문맥에서 그 영역이 어떤 재질
    증거를 갖는가"를 crop 재분류(주 신호)에 보조로 곱한다. 같은 CNN 의 파생
    신호(자기강화 위험)이므로 부스트 전용 + 약한 가중으로 제한
    (SEMANTIC_FUSION_PLAN §2-1).

    region: [x0,y0,x1,y1] 정규화 bbox. 실패/미지원 시 None.
    """
    from src.preprocess import color_tensor_at  # noqa: PLC0415

    try:
        cam = clf.cam_hires(color_tensor_at(raw, 448))
        if cam is None:
            return None
        n_cls, gh, gw = cam.shape
        x0 = min(max(int(region[0] * gw), 0), gw - 1)
        x1 = min(max(int(region[2] * gw) + 1, x0 + 1), gw)
        y0 = min(max(int(region[1] * gh), 0), gh - 1)
        y1 = min(max(int(region[3] * gh) + 1, y0 + 1), gh)
        patch = cam[:, y0:y1, x0:x1].reshape(n_cls, -1).mean(axis=1)  # (C,)

        idxs = clf.material_class_indices()
        z = patch[idxs]
        z = z - z.max()
        p = np.exp(z)
        p = p / p.sum()                          # 재질 후보들 위의 분포

        prior = np.ones(n_cls, dtype=np.float64)
        uniform = 1.0 / len(idxs)
        for local_i, cls_i in enumerate(idxs):
            share = float(p[local_i])
            if share >= min_share and share > uniform:
                prior[cls_i] = min(share / uniform, 6.0) ** weight
        return prior
    except Exception as exc:  # noqa: BLE001
        log.info(f"cam region prior 실패 (증거 없이 진행): {exc}")
        return None


def degs_for_orientation(tag: int) -> tuple[int, ...]:
    """EXIF Orientation → 회전 TTA 후보 (청사진 v2 트랙 B2).

    학습 크롭이 센서 방향이라, 세워진 서빙 입력의 유효 후보는
    "원래 센서 방향으로 되돌린 회전"뿐이다. 태그가 방향을 알려줄 때만
    (정방향, 되돌림) 2개로 축소하고, 태그 없음(tag=1)은 정보가 없으므로
    전수 3방향 유지 — 실측: tag=1 을 1회로 줄이면 51장서 -3건 회귀.
    """
    return {6: (0, 90), 8: (0, 270), 3: (0, 180)}.get(tag, (0, 90, 270))


def predict_rotations(
    clf: HierWasteClassifier,
    raw: bytes,
    degs: tuple[int, ...],
    mask_non_object: bool = True,
    ood_relax: bool = False,
) -> tuple[dict[str, Any], "np.ndarray"]:
    """prior 없는 1차 패스 — 회전 후보 중 최고 확신 결과와 그 입력 텐서 반환.

    호출부가 (조건부 OCR/CLIP/CAM) prior 를 계산한 뒤, 반환된 텐서 1장만
    재예측하면 되므로 융합 비용이 TTA 전체 재실행에서 1회로 줄어든다 (트랙 B1).
    """
    from src.preprocess import color_tensor_rotations  # noqa: PLC0415

    best: dict[str, Any] | None = None
    best_tensor = None
    total_ms = 0.0
    for deg, ci in color_tensor_rotations(raw, degs):
        r = clf.predict(ci, mask_non_object=mask_non_object, ood_relax=ood_relax)
        total_ms += r["inference_ms"]
        r["tta_rotation"] = deg
        if best is None or r["fine_confidence"] > best["fine_confidence"]:
            best = r
            best_tensor = ci
    assert best is not None and best_tensor is not None
    best["inference_ms"] = round(total_ms, 2)
    return best, best_tensor


def predict_best_rotation(
    clf: HierWasteClassifier,
    raw: bytes,
    mask_non_object: bool = False,
    fine_prior: np.ndarray | None = None,
    degs: tuple[int, ...] = (0, 90, 270),
) -> dict[str, Any]:
    """회전 TTA — 각 회전으로 분류 후 세부 확신이 가장 높은 결과 채택.

    학습 크롭(센서 방향)과 서빙 입력(EXIF 세움)의 방향 분포 어긋남을 서빙에서
    흡수한다 (근본 해결은 v7 회전 증강 재학습 — preprocess.color_tensor_rotations 참고).
    실측(실사용 51장): 세움 26 → 3방향 TTA 37 (+11건). inference_ms 는 합산.
    """
    from src.preprocess import color_tensor_rotations  # noqa: PLC0415

    best: dict[str, Any] | None = None
    total_ms = 0.0
    for deg, ci in color_tensor_rotations(raw, degs):
        r = clf.predict(ci, mask_non_object=mask_non_object, fine_prior=fine_prior)
        total_ms += r["inference_ms"]
        r["tta_rotation"] = deg
        if best is None or r["fine_confidence"] > best["fine_confidence"]:
            best = r
    assert best is not None
    best["inference_ms"] = round(total_ms, 2)
    return best


@lazy_singleton
def get_hier_classifier() -> HierWasteClassifier:
    return HierWasteClassifier()

