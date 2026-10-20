#!/usr/bin/env python3
"""라벨 노이즈 상시 계측 — CLIP 제로샷(독립 심판)으로 train 표본의 라벨 불일치율을 잰다.

왜 CLIP 인가: 우리 모델로 재판정하면 자기참조 편향(학습한 라벨을 다시 맞힘)이 생긴다.
CLIP 은 우리 데이터를 본 적이 없어 독립 신호다(2차 OI 채굴의 재질 라우팅과 같은 원칙).
단, CLIP 도 틀리므로 이 수치는 "의심 비율"이지 오류율 그 자체가 아니다 —
사이클 간 **변화**를 보는 지표로 쓴다.

판정: 컨셉 확률 softmax 의 top 컨셉을 대분류로 매핑 → 라벨 대분류와 다르고
top 확률 ≥ MIN_PROB 면 '불일치', top 확률 < MIN_PROB 면 '불확실'.

실행: .venv/bin/python scripts/audit_label_noise.py [--per-class 20] [--seed 7]
입력: outputs/models/clip/{clip_image.onnx, clip_concepts.npz} (build_clip_concepts.py)
출력: outputs/logs/cnn_hier/label_noise.json + 콘솔 요약
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime

import _base  # noqa: F401 — sys.path 설정
import numpy as np
from greenguide_common.taxonomy import COARSE_LABELS, FINE_IDX_TO_COARSE_IDX, FINE_LABELS
from PIL import Image, UnidentifiedImageError

from greenguide_classifier import config
from greenguide_classifier.data_quality import source_key
from greenguide_classifier.hier_dataset import build_hier_items, load_or_build_hier_splits
from greenguide_classifier.hier_train import LOG_DIR
from greenguide_classifier.infer import load_session

CLIP_DIR = config.MODELS_DIR / "clip"
OUT_PATH = LOG_DIR / "label_noise.json"
MIN_PROB = 0.35   # 2차 OI 채굴 CLIP_MIN_PROB 와 동일
# CLIP 표준 전처리 (waste-api clip_identity 와 동일)
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(3, 1, 1)

_FINE_TO_COARSE = {FINE_LABELS[i]: COARSE_LABELS[ci] for i, ci in enumerate(FINE_IDX_TO_COARSE_IDX)}


def _to_coarse(slug: str) -> str | None:
    if slug in _FINE_TO_COARSE:
        return _FINE_TO_COARSE[slug]
    if slug in COARSE_LABELS:
        return slug
    return None


def _preprocess(img: Image.Image) -> np.ndarray:
    """짧은 변 224 리사이즈 → 중앙 224 크롭 → CLIP 정규화 → (1,3,224,224)."""
    img = img.convert("RGB")
    w, h = img.size
    s = 224 / min(w, h)
    img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return ((x - _MEAN) / _STD)[None]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=20, help="세부 클래스당 train 표본")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    onnx_path, npz_path = CLIP_DIR / "clip_image.onnx", CLIP_DIR / "clip_concepts.npz"
    if not (onnx_path.exists() and npz_path.exists()):
        raise SystemExit(f"CLIP 자산 없음: {CLIP_DIR} — scripts/build_clip_concepts.py 먼저 실행")
    data = np.load(npz_path, allow_pickle=False)
    concept_embs = data["embeddings"].astype(np.float32)          # (K,512) L2 정규화
    concept_coarse = [_to_coarse(str(s)) for s in data["slugs"]]
    scale = float(data["logit_scale"]) if "logit_scale" in data.files else 100.0
    sess = load_session(onnx_path)

    items = build_hier_items()
    splits = load_or_build_hier_splits(items)
    rng = random.Random(args.seed)
    by_slug: dict[str, list[dict]] = defaultdict(list)
    for i in splits["train"]:
        it = items[i]
        if it["sup_kind"] == "fine":
            by_slug[it["sup_slug"]].append(it)

    per_class: dict[str, dict] = {}
    per_source: dict[str, Counter] = defaultdict(Counter)
    for slug, pool in sorted(by_slug.items()):
        label_coarse = _to_coarse(slug)
        if label_coarse is None:
            continue
        picks = rng.sample(pool, min(args.per_class, len(pool)))
        c = Counter()
        targets: Counter = Counter()
        for it in picks:
            try:
                with Image.open(config.PREPROCESSOR_ROOT / it["source_path"]) as im:
                    x = _preprocess(im)
            except (UnidentifiedImageError, OSError):
                c["unreadable"] += 1
                continue
            (emb,) = sess.run(None, {sess.get_inputs()[0].name: x})
            e = emb[0] / max(float(np.linalg.norm(emb[0])), 1e-9)
            logits = scale * (concept_embs @ e)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            top = int(p.argmax())
            src = source_key(it["source_path"])
            c["n"] += 1
            per_source[src]["n"] += 1
            if float(p[top]) < MIN_PROB or concept_coarse[top] is None:
                c["uncertain"] += 1
                per_source[src]["uncertain"] += 1
            elif concept_coarse[top] != label_coarse:
                c["disagree"] += 1
                per_source[src]["disagree"] += 1
                targets[concept_coarse[top]] += 1
        per_class[slug] = {
            "coarse": label_coarse,
            "n": c["n"],
            "disagree_rate": round(c["disagree"] / max(c["n"], 1), 4),
            "uncertain_rate": round(c["uncertain"] / max(c["n"], 1), 4),
            "disagree_to": dict(targets.most_common(3)),
            "unreadable": c["unreadable"],
        }

    total_n = sum(v["n"] for v in per_class.values())
    total_dis = sum(round(v["disagree_rate"] * v["n"]) for v in per_class.values())
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "judge": "clip_zero_shot",
        "min_prob": MIN_PROB,
        "per_class_sample": args.per_class,
        "overall": {"n": total_n, "disagree_rate": round(total_dis / max(total_n, 1), 4)},
        "per_class": per_class,
        "per_source": {
            k: {"n": c["n"], "disagree_rate": round(c["disagree"] / max(c["n"], 1), 4),
                "uncertain_rate": round(c["uncertain"] / max(c["n"], 1), 4)}
            for k, c in sorted(per_source.items(), key=lambda kv: -kv[1]["n"])
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[noise] 표본 {total_n:,}장 · CLIP 불일치 {report['overall']['disagree_rate']:.1%} (임계 {MIN_PROB})")
    worst = sorted(per_class.items(), key=lambda kv: -kv[1]["disagree_rate"])[:8]
    for slug, v in worst:
        print(f"[noise]   {slug:16s} n={v['n']:3d} 불일치 {v['disagree_rate']:.0%} 불확실 {v['uncertain_rate']:.0%} → {v['disagree_to']}")
    print("[noise] 출처별:", ", ".join(f"{k}={v['disagree_rate']:.0%}" for k, v in report["per_source"].items()))
    print(f"→ {OUT_PATH}")


if __name__ == "__main__":
    main()
