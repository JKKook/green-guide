"""재검증 패스 (B-2.2) — 새 모델로 pending 업로드를 재라벨링 → DRY-RUN 리포트.

라벨링 원칙(2026-06-15): 수동 라벨링 없음. 라벨은 학습된 모델이 매긴다.
일정량 피드백(active 이후 신규 RETRAIN_TRIGGER_NEW=100) 누적 → retrain → 이 패스로
새 모델이 pending 을 재라벨링하고, 이전(현 DB) 라벨과 무엇이 어떻게 바뀌는지 비교한다.

⚠️ DRY-RUN 전용 — Supabase 에 쓰지 않는다. 검토 후 반영 여부는 사람이 결정.

⚠️ 비교 기준 주의: DB 의 predicted_class 는 운영 경로(/predict)의 resnet+dinov2
   앙상블 결과다. --model 이 classifier.onnx(resnet 단독)면 flip 에는 '앙상블 vs 단독'
   차이가 섞인다. retrain 효과만 보려면 --baseline 에 구 classifier 를 주어
   '신 classifier vs 구 classifier' 단독끼리 비교하라(둘 다 동일 경로 → apples-to-apples).

사용:
    .venv/bin/python revalidate.py                  # active classifier.onnx, vs DB(앙상블)
    .venv/bin/python revalidate.py --limit 30       # 표본만(빠른 점검)
    .venv/bin/python revalidate.py --center-crop 0.7
    # retrain 효과 측정(권장): 신 모델 vs 구 모델 단독 비교
    .venv/bin/python revalidate.py --model outputs/models/cnn/classifier.onnx \
        --baseline outputs/backups/<prev>/classifier.onnx
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
from collections import Counter
from datetime import UTC, datetime

import numpy as np
import requests
from greenguide_common import imaging
from greenguide_common.logging import get_logger
from greenguide_common.supabase import get_client
from PIL import Image

from feedback_monitor import REJECT_THRESHOLD
from greenguide_classifier import config
from greenguide_classifier.infer import load_session, softmax

log = get_logger(__name__)

DEFAULT_MODEL = config.MODELS_DIR / "cnn" / "classifier.onnx"
OUT_PATH = config.LOGS_DIR / "cnn" / "revalidate.json"
_MEAN = imaging.MEAN_ARRAY
_STD = imaging.STD_ARRAY


def _prep(img: Image.Image, center_frac: float | None) -> np.ndarray:
    if center_frac:
        w, h = img.size
        s = int(min(w, h) * center_frac)
        l, t = (w - s) // 2, (h - s) // 2
        img = img.crop((l, t, l + s, t + s))
    im = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    return np.ascontiguousarray(((np.asarray(im, np.float32) / 255 - _MEAN) / _STD).transpose(2, 0, 1))[None]


def main() -> int:
    ap = argparse.ArgumentParser(prog="revalidate")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="재검증에 쓸 ONNX 경로(신 모델)")
    ap.add_argument("--baseline", default=None,
                    help="비교 기준 ONNX(구 모델). 주면 DB 대신 '신 vs 구' 단독 비교(apples-to-apples)")
    ap.add_argument("--limit", type=int, default=None, help="처리할 pending 개수 제한(표본 점검)")
    ap.add_argument("--center-crop", type=float, default=None, help="중앙 크롭 비율(예: 0.7)")
    args = ap.parse_args()

    config.refresh_classes_from_manifest()
    labels = list(config.CLASS_LABELS)

    sess = load_session(args.model)
    inp = sess.get_inputs()[0].name
    base_sess = (load_session(args.baseline)
                 if args.baseline else None)
    base_inp = base_sess.get_inputs()[0].name if base_sess else None

    def _run(s, iname, img: Image.Image) -> tuple[str, float, float]:
        o = s.run(["logits"], {iname: _prep(img, args.center_crop)})[0][0]
        p = softmax(o)
        i = int(p.argmax())
        nz = p[p > 0]
        h = float(-(nz * np.log(nz)).sum() / math.log(len(p))) if len(p) > 1 else 0.0
        return labels[i], float(p[i]), h

    def classify(img: Image.Image) -> tuple[str, float, float]:
        return _run(sess, inp, img)

    cli = get_client()
    rows = (cli.table("user_uploads")
            .select("id,image_url,predicted_class,predicted_confidence")
            .eq("feedback_status", "pending").execute().data) or []
    if args.limit:
        rows = rows[:args.limit]
    log.info(f"pending {len(rows)}건, 모델={os.path.basename(args.model)}"
          f"{f', center-crop={args.center_crop}' if args.center_crop else ''}")

    items, flips = [], Counter()
    new_dist, fetch_fail = Counter(), 0
    n_flip = n_reject = n_nonobj = 0
    for r in rows:
        try:
            img = Image.open(io.BytesIO(requests.get(r["image_url"], timeout=20).content))
        except Exception:  # noqa: BLE001 — fail-open: fetch 실패는 집계만 하고 건너뜀
            fetch_fail += 1
            continue
        old = _run(base_sess, base_inp, img)[0] if base_sess else r.get("predicted_class")
        new, conf, h = classify(img)
        rejected = conf < REJECT_THRESHOLD
        new_dist[new] += 1
        if new != old:
            n_flip += 1
            flips[f"{old}→{new}"] += 1
        if rejected:
            n_reject += 1
        if new == "non_object":
            n_nonobj += 1
        items.append({
            "id": r["id"], "old": old, "new": new,
            "new_confidence": round(conf, 4), "new_norm_entropy": round(h, 4),
            "rejected": rejected, "image_url": r.get("image_url"),
        })

    n = len(items)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": os.path.basename(args.model),
        "baseline": os.path.basename(args.baseline) if args.baseline else "DB(predicted_class, 앙상블)",
        "center_crop": args.center_crop,
        "dry_run": True,
        "n_pending": len(rows),
        "n_evaluated": n,
        "fetch_failed": fetch_fail,
        "flip_count": n_flip,
        "flip_rate": round(n_flip / n, 4) if n else None,
        "reject_count": n_reject,
        "reject_rate": round(n_reject / n, 4) if n else None,
        "non_object_count": n_nonobj,
        "new_label_distribution": dict(new_dist.most_common()),
        "top_flips": flips.most_common(15),
        "items": items,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 62)
    print(f"재검증 DRY-RUN  ({n}/{len(rows)}건 평가, fetch 실패 {fetch_fail})")
    print("=" * 62)
    if n:
        print(f"  라벨 변경(flip): {n_flip}건 ({n_flip/n*100:.1f}%) | "
              f"reject(<{REJECT_THRESHOLD}): {n_reject}건 ({n_reject/n*100:.1f}%) | "
              f"non_object: {n_nonobj}건")
        print("  새 라벨 분포:")
        for lbl, c in new_dist.most_common():
            print(f"     {lbl:12} {c}")
        if flips:
            print("  주요 변경(이전→새):")
            for pair, c in flips.most_common(10):
                print(f"     {pair:28} {c}건")
    print(f"  → {OUT_PATH}  (DB 미반영, dry-run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
