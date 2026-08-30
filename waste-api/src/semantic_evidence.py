"""시맨틱 증거 — 이미지 텍스트(OCR)에서 재질/정체 단서를 추출해 fine prior 로 변환.

SEMANTIC_FUSION_PLAN.md 신호① 구현. VLM 이 약통/화장품을 구분하는 첫 능력
"글자를 읽는다"를 이식한다: 한국 포장재에 인쇄된 분리배출 표시(무색페트·HDPE 등)는
사실상 정답지이고, 제품 정체어(캡슐·샴푸·소주)는 재질을 강하게 함의한다.

- 엔진: rapidocr v3 (onnxruntime 전용) + PP-OCRv5 korean mobile, models/ocr/ 번들.
- 요청당 원본(EXIF 정규화) 이미지에서 1회 실행 → bbox 로 물건 crop 에 귀속.
- 융합: log-linear — fused = softmax(log p + Σ log prior). 증거 없으면 prior 전부
  1.0 이라 기존 결과와 완전 동일 (안전 기본값).
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from src.core import config
from src.core.log import get_logger
from src.core.singleton import lazy_singleton

log = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OCR_DIR = _PROJECT_ROOT / "models" / "ocr"

# ── 어휘 2계층 ──────────────────────────────────────────────────────────────
# (패턴, 대상 slug(fine 또는 coarse), boost)
# A급 = 분리배출 표시·재질어: 포장재에 인쇄된 재질 표기 — 매우 강한 증거 (boost 6)
# B급 = 정체어: 물체 정체 → 재질 추론 (VLM식 세상지식) — 중간 증거 (boost 2.5)
_BOOST_MARK = 6.0
_BOOST_IDENTITY = 2.5

# 한글 패턴은 부분문자열 매칭(공백 제거 후), 라틴 패턴은 단어 경계 정규식.
_LEXICON: list[tuple[str, str, float]] = [
    # ── A급: 분리배출 표시/재질어 ──
    ("무색페트", "pet", _BOOST_MARK),
    ("페트", "pet", _BOOST_MARK),
    ("hdpe", "plastic_other", _BOOST_MARK),
    ("ldpe", "vinyl_clean", _BOOST_MARK),      # LDPE 마크는 대부분 비닐 포장
    ("pp", "plastic_other", _BOOST_MARK),
    ("ps", "plastic_other", _BOOST_MARK),
    ("pet", "pet", _BOOST_MARK),
    ("플라스틱", "plastic", _BOOST_MARK),
    ("비닐류", "vinyl", _BOOST_MARK),
    ("비닐", "vinyl", _BOOST_MARK),
    ("캔류", "metal", _BOOST_MARK),
    ("알루미늄", "metal", _BOOST_MARK),
    ("철", "metal", _BOOST_MARK * 0.5),         # 1글자급 오탐 여지 — 약화
    ("유리", "glass", _BOOST_MARK),
    ("종이팩", "carton", _BOOST_MARK),
    ("멸균팩", "carton", _BOOST_MARK),
    ("종이", "paper", _BOOST_MARK * 0.7),       # '종이팩' 보다 먼저 매칭되지 않게 아래 배치 유지
    ("스티로폼", "styrofoam", _BOOST_MARK),
    ("발포", "styrofoam", _BOOST_MARK),
    ("일반쓰레기", "trash", _BOOST_MARK),
    # ── B급: 정체어 ──
    # 의약/건강기능식품 용기 → 플라스틱 통
    ("캡슐", "plastic_other", _BOOST_IDENTITY),
    ("유산균", "plastic_other", _BOOST_IDENTITY),
    ("비타민", "plastic_other", _BOOST_IDENTITY),
    ("영양제", "plastic_other", _BOOST_IDENTITY),
    ("건강기능식품", "plastic_other", _BOOST_IDENTITY),
    ("약국", "paper", _BOOST_IDENTITY),          # 약봉투/처방전 맥락
    ("처방", "paper", _BOOST_IDENTITY),
    # 화장품 → 용기(펌프병 플라스틱 우세)
    ("샴푸", "plastic", _BOOST_IDENTITY),
    ("로션", "plastic", _BOOST_IDENTITY),
    ("토너", "plastic", _BOOST_IDENTITY),
    ("에센스", "plastic", _BOOST_IDENTITY),
    ("크림", "plastic", _BOOST_IDENTITY * 0.8),
    # 주류 병 → 보증금 유리병
    ("소주", "glass_deposit", _BOOST_IDENTITY),
    ("참이슬", "glass_deposit", _BOOST_IDENTITY),
    ("처음처럼", "glass_deposit", _BOOST_IDENTITY),
    ("진로", "glass_deposit", _BOOST_IDENTITY),
    ("카스", "glass_deposit", _BOOST_IDENTITY),
    ("테라", "glass_deposit", _BOOST_IDENTITY),
    ("맥주", "glass_deposit", _BOOST_IDENTITY),
    # 음료팩/우유
    ("우유", "carton", _BOOST_IDENTITY),
    ("두유", "carton", _BOOST_IDENTITY),
    # 영수증/문서 → 종이
    ("영수증", "paper_other", _BOOST_IDENTITY),
    ("승인번호", "paper_other", _BOOST_IDENTITY),
    ("사업자등록", "paper_other", _BOOST_IDENTITY),
    ("계산서", "paper_other", _BOOST_IDENTITY),
    # 전자제품 브랜드/용어
    ("benq", "electronics", _BOOST_IDENTITY),
    ("samsung", "electronics", _BOOST_IDENTITY),
    ("galaxy", "electronics", _BOOST_IDENTITY),
    ("lg", "electronics", _BOOST_IDENTITY * 0.8),
    ("모니터", "electronics", _BOOST_IDENTITY),
    ("충전", "electronics", _BOOST_IDENTITY * 0.8),
    # 건전지
    ("건전지", "battery", _BOOST_IDENTITY),
    ("배터리", "battery", _BOOST_IDENTITY),
    ("aaa", "battery", _BOOST_IDENTITY * 0.8),
    ("택배", "cardboard", _BOOST_IDENTITY),
    ("운송장", "cardboard", _BOOST_IDENTITY),
]

_LATIN_RE = {
    tok: re.compile(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])")
    for tok, _, _ in _LEXICON if tok.isascii()
}


class SemanticEvidence:
    """OCR 1회 실행 + 어휘 매칭 → fine prior 벡터. 실패 시 증거 없음으로 격리."""

    def __init__(self) -> None:
        self.available = False
        self._ocr = None
        if not config.OCR_ENABLED:
            log.info("OCR 비활성 (WASTE_API_OCR=0)")
            return
        det = _OCR_DIR / "det.onnx"
        rec = _OCR_DIR / "korean_rec_v5.onnx"
        if not (det.exists() and rec.exists()):
            log.info(f"OCR 모델 미배치 ({_OCR_DIR}) — 증거 없이 진행")
            return
        try:
            from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
            self._ocr = RapidOCR(params={
                "Det.model_path": str(det),
                "Rec.model_path": str(rec),
                "Rec.lang_type": LangRec.KOREAN,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.model_type": ModelType.MOBILE,
                "Global.use_cls": False,     # 회전 분류 생략 — 속도 (EXIF 는 이미 정규화됨)
            })
            self.available = True
            log.info("OCR 활성 (PP-OCRv5 korean)")
        except Exception as exc:  # noqa: BLE001
            log.info(f"OCR 초기화 실패 (증거 없이 진행): {exc}")

    def read_texts(self, image_bytes: bytes) -> list[dict[str, Any]]:
        """이미지 → [{text, score, bbox_norm}]. bbox 는 물건 crop 귀속용."""
        if not self.available or self._ocr is None:
            return []
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            w, h = img.size
            res = self._ocr(np.asarray(img))
            boxes = res.boxes if res.boxes is not None else []
            txts = res.txts if res.txts is not None else []
            scores = res.scores if res.scores is not None else []
            out = []
            for box, txt, score in zip(boxes, txts, scores):
                if not txt or float(score) < 0.55:
                    continue
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                out.append({
                    "text": txt,
                    "score": round(float(score), 3),
                    "bbox_norm": [min(xs) / w, min(ys) / h,
                                  max(xs) / w, max(ys) / h],
                })
            return out
        except Exception as exc:  # noqa: BLE001
            log.info(f"OCR 실행 실패 (증거 없이 진행): {exc}")
            return []


def match_evidence(texts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OCR 텍스트 → 어휘 매칭 목록 [{type, token, mapped_class, score, bbox_norm}]."""
    found: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for t in texts:
        norm = re.sub(r"\s+", "", t["text"]).lower()
        if len(norm) < 2:
            continue
        for tok, target, boost in _LEXICON:
            if tok.isascii():
                if not _LATIN_RE[tok].search(norm):
                    continue
            elif tok not in norm:
                continue
            key = (tok, target)
            if key in seen:
                continue
            seen.add(key)
            found.append({
                "type": "mark" if boost >= _BOOST_MARK * 0.5 else "text",
                "token": tok,
                "matched_text": t["text"],
                "mapped_class": target,
                "boost": boost,
                "score": t["score"],
                "bbox_norm": t["bbox_norm"],
            })
    return found


def evidence_prior(
    evidence: list[dict[str, Any]],
    fine_labels: list[str],
    fine_to_coarse: dict[str, str],
    region: list[float] | None = None,
) -> np.ndarray | None:
    """증거 목록 → (C_fine,) prior 승수 벡터. 증거 없으면 None (융합 생략).

    region 이 주어지면 (탭/후보 crop bbox_norm) 텍스트 중심이 그 안에 있는
    증거만 사용 — 장면의 다른 물건 라벨이 새는 것 방지 (신호③ 공간 귀속).
    """
    prior = np.ones(len(fine_labels), dtype=np.float64)
    used = False
    for ev in evidence:
        if region is not None:
            cx = (ev["bbox_norm"][0] + ev["bbox_norm"][2]) / 2
            cy = (ev["bbox_norm"][1] + ev["bbox_norm"][3]) / 2
            if not (region[0] <= cx <= region[2] and region[1] <= cy <= region[3]):
                continue
        target = ev["mapped_class"]
        boost = float(ev["boost"]) * float(ev["score"])   # OCR 확신으로 스케일
        if target in fine_labels:
            prior[fine_labels.index(target)] *= max(boost, 1.0)
            used = True
        else:
            # coarse 대상 — 자식 fine 전체에 분배 (√boost: 특정 자식 과잉확신 방지)
            child = [i for i, f in enumerate(fine_labels)
                     if fine_to_coarse.get(f) == target]
            for i in child:
                prior[i] *= max(boost, 1.0) ** 0.5
            used = used or bool(child)
    return prior if used else None


@lazy_singleton
def get_evidence_engine() -> SemanticEvidence:
    return SemanticEvidence()
