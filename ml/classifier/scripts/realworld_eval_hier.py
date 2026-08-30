#!/usr/bin/env python3
"""실사용 계층 평가 — 사용자 피드백(ground truth)으로 대분류/세부 정확도 측정.

realworld_eval.py(flat)의 계층 버전. blueprint §7 의 "실사용 대분류 ≥85%"
KPI 를 측정하는 하네스. 매 retrain_hier 사이클 후 실행 권장.

- truth 라벨 → taxonomy 감독 매핑 (legacy 라벨은 coarse/fine 자동 판별)
- coarse 정확도: 전체 / fine 정확도: fine-truth 아이템만
- guidance-safe 정확도도 함께 (안내 동일 형제는 정답)

실행: .venv/bin/python scripts/realworld_eval_hier.py
출력: outputs/logs/cnn_hier/realworld_eval.json
"""
from __future__ import annotations

import io
import json
from collections import Counter

import _base  # noqa: F401 — sys.path 설정
import numpy as np
import onnxruntime as ort
import requests
from greenguide_common import imaging
from greenguide_common.taxonomy import (
    COARSE_TO_INDEX,
    FINE_IDX_TO_COARSE_IDX,
    FINE_LABELS,
    FINE_TO_INDEX,
    LEGACY_LABEL_SUPERVISION,
    same_guidance,
)
from PIL import Image

from greenguide_classifier import config
from greenguide_classifier.hier_train import LOG_DIR
from retrain import fetch_feedback_rows

ONNX_PATH = config.MODELS_DIR / "cnn_hier" / "classifier.onnx"
OUT_PATH = LOG_DIR / "realworld_eval.json"

_MEAN = imaging.MEAN_CHW
_STD = imaging.STD_CHW


def _preprocess(raw: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(raw)).convert("RGB").resize(
        (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return ((arr - _MEAN) / _STD)[None]


def _truth_supervision(label: str) -> tuple[str, str] | None:
    sup = LEGACY_LABEL_SUPERVISION.get(label)
    if sup:
        return sup
    if label in FINE_TO_INDEX:
        return ("fine", label)
    if label in COARSE_TO_INDEX:
        return ("coarse", label)
    return None


def main() -> None:
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    rows = fetch_feedback_rows()
    print(f"피드백 ground truth: {len(rows)}건")

    n_eval = 0
    coarse_hit = 0
    fine_total = 0
    fine_hit = 0
    fine_guidance_hit = 0
    per_coarse: Counter = Counter()
    per_coarse_hit: Counter = Counter()
    errors: list[dict] = []
    skipped = 0

    for row in rows:
        sup = _truth_supervision(row["feedback_label"])
        if sup is None:
            skipped += 1
            continue
        kind, slug = sup
        try:
            raw = requests.get(row["image_url"], timeout=30).content
            x = _preprocess(raw)
        except Exception:  # noqa: BLE001
            skipped += 1
            continue

        (logits,) = sess.run(["logits"], {"image": x})
        fi = int(logits[0].argmax())
        pred_fine = FINE_LABELS[fi]
        # coarse 롤업 (확률 합산 argmax)
        e = np.exp(logits[0] - logits[0].max())
        probs = e / e.sum()
        coarse_probs = np.zeros(max(FINE_IDX_TO_COARSE_IDX) + 1)
        for f_idx, c_idx in enumerate(FINE_IDX_TO_COARSE_IDX):
            coarse_probs[c_idx] += probs[f_idx]
        pred_coarse_idx = int(coarse_probs.argmax())

        # truth coarse
        if kind == "fine":
            truth_coarse_idx = FINE_IDX_TO_COARSE_IDX[FINE_TO_INDEX[slug]]
        else:
            truth_coarse_idx = COARSE_TO_INDEX[slug]

        n_eval += 1
        truth_coarse = list(COARSE_TO_INDEX)[truth_coarse_idx]
        per_coarse[truth_coarse] += 1
        c_ok = pred_coarse_idx == truth_coarse_idx
        if c_ok:
            coarse_hit += 1
            per_coarse_hit[truth_coarse] += 1

        f_ok = None
        if kind == "fine":
            fine_total += 1
            f_ok = pred_fine == slug
            if f_ok:
                fine_hit += 1
            if same_guidance(pred_fine, slug):
                fine_guidance_hit += 1

        if not c_ok or f_ok is False:
            errors.append({
                "upload_id": row["id"], "truth": f"{kind}:{slug}",
                "pred_fine": pred_fine, "coarse_ok": c_ok,
            })

    coarse_acc = coarse_hit / max(n_eval, 1)
    result = {
        "n_eval": n_eval,
        "skipped": skipped,
        "coarse_accuracy": round(coarse_acc, 4),
        "fine_accuracy": round(fine_hit / max(fine_total, 1), 4),
        "fine_guidance_safe_accuracy": round(fine_guidance_hit / max(fine_total, 1), 4),
        "fine_total": fine_total,
        "per_coarse": {
            k: {"n": per_coarse[k], "recall": round(per_coarse_hit[k] / per_coarse[k], 3)}
            for k in per_coarse
        },
        "errors": errors,
        "kpi_note": "blueprint §7: 실사용 대분류 목표 ≥0.85 (표본 작음 — CI 주의)",
    }
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n실사용 대분류 정확도: {coarse_acc:.1%} ({coarse_hit}/{n_eval})  [KPI 목표 85%]")
    print(f"실사용 세부 정확도: {fine_hit}/{fine_total} "
          f"(guidance-safe {fine_guidance_hit}/{fine_total})")
    print("대분류별:")
    for k, v in sorted(per_coarse.items(), key=lambda kv: -kv[1]):
        print(f"  {k:12} {per_coarse_hit[k]}/{v}")
    print(f"→ {OUT_PATH}")


if __name__ == "__main__":
    main()
