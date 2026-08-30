"""CLIP 제로샷 정체 인식 — 신호② '물체 정체를 안다' (SEMANTIC_FUSION_PLAN Phase 2).

이미지 인코더(ONNX, INT8 84MB)만 서빙하고, 컨셉 텍스트 임베딩은
build_clip_concepts.py 가 사전계산 (clip_concepts.npz — 66컨셉, 컨셉→fine 매핑).
crop 임베딩 × 컨셉 행렬 cosine → 정체 분포 → fine prior 승수.

- 컨셉 추가/수정 = npz 재빌드만 (재학습·재배포 불필요, 인코더 동일)
- 실패 격리: 로드/추론 실패 시 available=False → 증거 없이 기존 경로
- 장면 맥락(신호③): 전체 프레임 정체 분포를 crop prior 에 약하게 혼합
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from src.core import config
from src.core.log import get_logger
from src.core.singleton import lazy_singleton

log = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CLIP_DIR = _PROJECT_ROOT / "models" / "clip"

# CLIP 전용 전처리 상수 (ImageNet 과 다름 — processor 설정에서 가져옴)
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(3, 1, 1)

# prior 변환 파라미터 — 실사용 51장 스윕으로 결정.
# 실측 교훈: 전체 장면 prior 는 역효과 (혼재 장면 center-crop 정체 오인) —
# 부스트 전용 + 확신 임계 + 크롭 경로 한정으로 재설계.
PRIOR_WEIGHT = config.CLIP_PRIOR_WEIGHT
SCENE_WEIGHT = config.CLIP_SCENE_WEIGHT
MIN_CONCEPT_PROB = 0.30     # 이 확신 미만의 정체는 노이즈로 보고 무시
_RATIO_MAX = 8.0            # 우도비 상한 — 단일 신호의 폭주 방지


class ClipIdentity:
    """CLIP 이미지 인코더 + 컨셉 매칭. 실패 시 available=False 로 격리."""

    def __init__(self) -> None:
        self.available = False
        if not config.CLIP_ENABLED:
            log.info("정체 인식 비활성 (WASTE_API_CLIP=0)")
            return
        onnx_path = _CLIP_DIR / "clip_image.onnx"
        npz_path = _CLIP_DIR / "clip_concepts.npz"
        if not (onnx_path.exists() and npz_path.exists()):
            log.info(f"자산 미배치 ({_CLIP_DIR}) — 정체 증거 없이 진행")
            return
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"])
            data = np.load(npz_path, allow_pickle=False)
            self.concept_embs: np.ndarray = data["embeddings"]      # (K,512) L2 정규화
            self.phrases: list[str] = [str(p) for p in data["phrases"]]
            # 한국어 표시명 (앱 증거 배지용) — 구버전 npz 는 영문 fallback
            self.phrases_ko: list[str] = (
                [str(p) for p in data["phrases_ko"]]
                if "phrases_ko" in data else list(self.phrases))
            self.slugs: list[str] = [str(s) for s in data["slugs"]]
            self.logit_scale = float(data["logit_scale"])
            self.available = True
            log.info(f"정체 인식 활성 (컨셉 {len(self.slugs)}개, "
                  f"w={PRIOR_WEIGHT}, scene_w={SCENE_WEIGHT})")
        except Exception as exc:  # noqa: BLE001
            log.info(f"초기화 실패 (정체 증거 없이 진행): {exc}")

    @staticmethod
    def _preprocess(img: Image.Image) -> np.ndarray:
        """CLIP 표준: 짧은 변 224 리사이즈 → 중앙 224 크롭 → CLIP 정규화."""
        w, h = img.size
        s = 224 / min(w, h)
        img = img.resize((max(224, round(w * s)), max(224, round(h * s))),
                         Image.BICUBIC)
        w, h = img.size
        left, top = (w - 224) // 2, (h - 224) // 2
        img = img.crop((left, top, left + 224, top + 224))
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return ((x - _MEAN) / _STD)[None]

    def identity_probs(self, image_bytes: bytes) -> np.ndarray | None:
        """이미지 → (K,) 컨셉 정체 분포. 실패 시 None."""
        if not self.available:
            return None
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            (emb,) = self.session.run(
                None, {"pixel_values": self._preprocess(img)})
            e = emb[0] / max(float(np.linalg.norm(emb[0])), 1e-9)
            sims = self.concept_embs @ e                     # (K,) cosine
            z = self.logit_scale * sims
            z = z - z.max()
            p = np.exp(z)
            return p / p.sum()
        except Exception as exc:  # noqa: BLE001
            log.info(f"추론 실패 (정체 증거 없이 진행): {exc}")
            return None

    def evidence_prior(
        self,
        crop_probs: np.ndarray,
        fine_labels: list[str],
        scene_probs: np.ndarray | None = None,
        weight: float | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """정체 분포 → (C_fine,) prior 승수 + 설명용 증거 목록.

        - 장면 맥락: scene_probs(전체 프레임)를 SCENE_WEIGHT 로 혼합 (신호③)
        - 클래스 확률 = 그 클래스로 매핑된 컨셉 확률 합
        - prior = (컨셉 있는 클래스들의 균등분포 대비 우도비)^weight, 상하한 클립
          컨셉이 없는 클래스(etc 등)는 1.0 (중립) — 벌점 없음
        """
        w = PRIOR_WEIGHT if weight is None else weight
        probs = crop_probs
        if scene_probs is not None:
            probs = (1.0 - SCENE_WEIGHT) * probs + SCENE_WEIGHT * scene_probs

        p_cls: dict[str, float] = {}
        for p, slug in zip(probs, self.slugs):
            p_cls[slug] = p_cls.get(slug, 0.0) + float(p)
        n_mapped = len(p_cls)

        # 부스트 전용 (벌점 없음): 확신 임계를 넘는 정체만 해당 클래스를 밀어올림.
        # (초기 설계의 '조용한 매핑 클래스 벌점'은 51장 실측에서 -3건 역효과)
        prior = np.ones(len(fine_labels), dtype=np.float64)
        for i, f in enumerate(fine_labels):
            p = p_cls.get(f, 0.0)
            if p >= MIN_CONCEPT_PROB:
                ratio = min(p * n_mapped, _RATIO_MAX)   # 균등 대비 우도비
                if ratio > 1.0:
                    prior[i] = ratio ** w

        # 설명용 — top1 컨셉 (임계 이상일 때만). token 은 앱 배지용 한국어.
        evidence: list[dict[str, Any]] = []
        top = int(np.argmax(probs))
        if float(probs[top]) >= 0.25:
            evidence.append({
                "type": "identity",
                "token": self.phrases_ko[top],
                "matched_text": self.phrases[top],
                "mapped_class": self.slugs[top],
                "score": round(float(probs[top]), 3),
            })
        return prior, evidence


@lazy_singleton
def get_clip_identity() -> ClipIdentity:
    return ClipIdentity()
