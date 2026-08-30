"""실사용(real-world) 평가셋 — 사용자 피드백 사진으로 '진짜 정확도' 측정. (Tier 1-1)

frozen test 는 AI Hub 분포(길바닥 가전·깔끔한 크롭)라 실사용을 반영 못 한다.
user_uploads 의 confirmed/corrected 피드백(=사용자 검증 라벨)을 ground truth 로
현재 모델의 실사용 정확도·혼동을 측정한다.

사용: python realworld_eval.py
"""
from __future__ import annotations

import io
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime

import numpy as np
import requests
from greenguide_common import imaging
from greenguide_common.logging import get_logger
from greenguide_common.supabase import Bucket, get_client
from PIL import Image

from greenguide_classifier import config
from greenguide_classifier.infer import load_session, softmax

log = get_logger(__name__)

ONNX_PATH = config.MODELS_DIR / "cnn" / "classifier.onnx"
OUT_PATH = config.LOGS_DIR / "realworld_eval.json"
_MEAN = imaging.MEAN_ARRAY
_STD = imaging.STD_ARRAY


def _prep(img: Image.Image, center_frac: float | None = None) -> np.ndarray:
    if center_frac:
        w, h = img.size
        s = int(min(w, h) * center_frac)
        left, top = (w - s) // 2, (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
    im = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    return np.ascontiguousarray(((np.asarray(im, np.float32) / 255 - _MEAN) / _STD).transpose(2, 0, 1))[None]


def main() -> int:
    config.refresh_classes_from_manifest()
    labels = list(config.CLASS_LABELS)
    sess = load_session(ONNX_PATH)
    inp = sess.get_inputs()[0].name

    def classify(img, center_frac=None) -> tuple[str, float]:
        o = sess.run(["logits"], {inp: _prep(img, center_frac)})[0][0]
        p = softmax(o)
        i = int(p.argmax())
        return labels[i], float(p[i])

    cli = get_client()
    rows = (cli.table("user_uploads")
            .select("id,image_url,storage_path,feedback_label,feedback_status")
            .in_("feedback_status", ["confirmed", "corrected"]).execute().data) or []
    log.info(f"피드백 {len(rows)}건 수집")

    y_true, y_pred, y_pred_crop = [], [], []
    skipped_label = Counter()
    for r in rows:
        truth = r.get("feedback_label")
        if truth not in labels:   # paper_pack 등 모델 미출력 라벨
            skipped_label[truth] += 1
            continue
        try:
            # 비공개 버킷 대응: storage API 우선, 레거시 공개 URL fallback
            sp = r.get("storage_path")
            try:
                data = cli.storage.from_(str(Bucket.USER_UPLOADS)).download(sp) if sp else None
            except Exception:  # noqa: BLE001 — fail-open: 레거시 공개 URL 로 fallback
                data = None
            if data is None:
                data = requests.get(r["image_url"], timeout=20).content
            img = Image.open(io.BytesIO(data))
        except Exception:  # noqa: BLE001 — fail-open: 이미지 로드 실패 샘플 건너뜀
            continue
        pred, _ = classify(img)
        pred_c, _ = classify(img, center_frac=0.7)
        y_true.append(truth)
        y_pred.append(pred)
        y_pred_crop.append(pred_c)

    n = len(y_true)
    if n == 0:
        log.warning("평가 가능한 샘플 0 — 피드백 데이터 부족")
        return 0

    acc = sum(t == p for t, p in zip(y_true, y_pred, strict=False)) / n
    acc_crop = sum(t == p for t, p in zip(y_true, y_pred_crop, strict=False)) / n

    # per-class + 혼동
    per_class: dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0})
    confusions: Counter = Counter()
    for t, p in zip(y_true, y_pred, strict=False):
        per_class[t]["n"] += 1
        if t == p:
            per_class[t]["correct"] += 1
        else:
            confusions[f"{t}→{p}"] += 1

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": ONNX_PATH.name,
        "n_eval": n,
        "skipped_untrained_label": dict(skipped_label),
        "accuracy_fullimg": round(acc, 4),
        "accuracy_centercrop70": round(acc_crop, 4),
        "per_class": {k: {**v, "recall": round(v["correct"] / v["n"], 3)}
                      for k, v in sorted(per_class.items())},
        "top_confusions": confusions.most_common(10),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"실사용 정확도 (피드백 {n}건, 12클래스 모델)")
    print("=" * 60)
    print(f"  전체이미지: {acc*100:.1f}%   |  중앙70%크롭: {acc_crop*100:.1f}%")
    print("  (참고: frozen test(AI Hub 분포) = 95.9%)")
    print("  클래스별 (recall):")
    for k, v in report["per_class"].items():
        print(f"    {k:12} {v['correct']}/{v['n']} ({v['recall']*100:.0f}%)")
    if confusions:
        print("  주요 혼동:")
        for pair, c in confusions.most_common(8):
            print(f"    {pair:28} {c}건")
    if skipped_label:
        print(f"  제외(모델 미학습 라벨): {dict(skipped_label)}")
    print(f"  → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
