#!/usr/bin/env python3
"""실사용 셋 배포 게이트 — 새 계층 ONNX 를 실사용 사진으로 채점해 통과/실패를 낸다.

retrain_hier.py [7.5] 에서 export 직후 실행된다(exit 1 = FAIL → 롤백). 모델 자체를
재는 게이트라 서빙 부가 단계(OCR·CLIP·VLM·크롭)는 쓰지 않고, 서빙과 같은
회전 TTA(0/90/270 최고확신) + non_object 마스킹 + 사이드카 대분류 게이트만 적용한다.

셋: GG_DATA 통합셋 realworld_eval_set.jsonl(sha 중복 제거)을 source 칸으로 나눠 따로 채점
  feedback  — 사용자 피드백 라벨 51
  tester    — 테스터 고유 사진 35, GG_CNN 시각 라벨(label_source=agent_gg_cnn)
  golden_v3 — 사용자 검토 대기(label=null)라 라벨이 생길 때까지 건너뜀
기준 (ELECTRONICS_V15_DATA_PLAN §3 + W6 제안, 목표값은 GG_TRAIN 과 합의 대상):
  electronics 오탐률(정답≠electronics 중 electronics 판정) ≤ GATE_ELEC_FP_MAX
  electronics recall ≥ GATE_ELEC_RECALL_MIN (전자제품 정답 ≥ 5장인 셋만)
  대분류 정확도가 직전 PASS 기록 대비 GATE_ACC_MAX_DROP 초과 하락 금지
라벨 셋 파일이 없으면 FAIL, 라벨 행 0 인 셋은 건너뜀 (조용히 통과시키지 않는다). GREENGUIDE_REALWORLD_GATE=0 이면 기록만.

실행: .venv/bin/python scripts/realworld_gate.py [--onnx …] [--set name=path …]
출력: outputs/logs/cnn_hier/realworld_gate.json (+ PASS 시 realworld_gate_baseline.json 갱신)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import _base  # noqa: F401 — sys.path 설정
import numpy as np
from greenguide_common import imaging
from PIL import Image
from realworld_eval_golden import load_rows, score

from greenguide_classifier import config
from greenguide_classifier.hier_train import LOG_DIR

GATE_ELEC_FP_MAX = float(os.getenv("GREENGUIDE_GATE_ELEC_FP_MAX", "0.03"))
GATE_ELEC_RECALL_MIN = float(os.getenv("GREENGUIDE_GATE_ELEC_RECALL_MIN", "0.85"))
GATE_ACC_MAX_DROP = float(os.getenv("GREENGUIDE_GATE_REALWORLD_ACC_DROP", "0.02"))
RECALL_MIN_SUPPORT = 5

# GG_DATA 통합셋(피드백·테스터·골든 v3, sha 중복 제거) — 셋 이름 = source 칸
REALWORLD_SET = Path(os.getenv(
    "GREENGUIDE_REALWORLD_SET",
    str(Path.home() / "Downloads" / "greenguide-golden" / "realworld_eval_set.jsonl")))
DEFAULT_SETS = {"feedback": f"{REALWORLD_SET}#feedback", "tester": f"{REALWORLD_SET}#tester",
                "golden_v3": f"{REALWORLD_SET}#golden_v3"}
ONNX_PATH = config.MODELS_DIR / "cnn_hier" / "classifier.onnx"
OUT_PATH = LOG_DIR / "realworld_gate.json"
BASELINE_PATH = LOG_DIR / "realworld_gate_baseline.json"


def _tensor(im: Image.Image) -> np.ndarray:
    im = im.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return ((arr - imaging.MEAN_CHW) / imaging.STD_CHW)[None].astype(np.float32)


def predict(sess, tax: dict, path: Path) -> dict:
    """서빙 1차 패스와 같은 규칙: 회전 TTA 최고 fine 확신 + non_object 마스킹 + 대분류 게이트."""
    fine, coarse = tax["fine_labels"], tax["coarse_labels"]
    f2c = tax["fine_idx_to_coarse_idx"]
    no = fine.index("non_object") if "non_object" in fine else None
    with Image.open(path) as im:
        base = imaging.decode_rgb(im)
    best = None
    for deg in (0, 90, 270):
        (lg,) = sess.run(["logits"], {"image": _tensor(base.rotate(deg, expand=True))})
        lg = lg[0].astype(np.float64)
        if no is not None:
            lg[no] = -1e9
        p = np.exp(lg - lg.max())
        p /= p.sum()
        if best is None or p.max() > best.max():
            best = p
    cp = np.zeros(len(coarse))
    for fi, ci in enumerate(f2c):
        cp[ci] += best[fi]
    ci = int(cp.argmax())
    ok = cp[ci] >= tax["gate"]["coarse_min_confidence"]
    return {
        "display_level": "coarse" if ok else "reject",
        "display_class": coarse[ci] if ok else "etc",
        "coarse_class": coarse[ci],
        "coarse_confidence": round(float(cp[ci]), 4),
        "fine_class": None,
        "coarse_probabilities": {c: float(cp[i]) for i, c in enumerate(coarse)},
        "fine_top5": [{"slug": fine[i], "prob": float(best[i])}
                      for i in np.argsort(best)[::-1][:5]],
    }


def check(name: str, rep: dict, baseline: dict | None) -> list[str]:
    e = rep["electronics"]
    negatives = rep["n"] - e["support"]
    fp_rate = e["false_positive"] / negatives if negatives else 0.0
    rep["electronics"]["fp_rate"] = round(fp_rate, 4)
    reasons = []
    if fp_rate > GATE_ELEC_FP_MAX:
        reasons.append(f"{name}: electronics 오탐률 {fp_rate:.1%} > {GATE_ELEC_FP_MAX:.0%} "
                       f"({e['false_positive']}/{negatives})")
    if e["support"] >= RECALL_MIN_SUPPORT and (e["recall"] or 0) < GATE_ELEC_RECALL_MIN:
        reasons.append(f"{name}: electronics recall {e['recall']} < {GATE_ELEC_RECALL_MIN}")
    old = (baseline or {}).get(name, {}).get("coarse_acc")
    if old is not None and old - rep["coarse_acc"] > GATE_ACC_MAX_DROP:
        reasons.append(f"{name}: 대분류 {old:.3f}→{rep['coarse_acc']:.3f} "
                       f"(-{(old - rep['coarse_acc']) * 100:.1f}pp > {GATE_ACC_MAX_DROP * 100:.0f}pp)")
    return reasons


def main() -> int:
    import onnxruntime as ort  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, default=ONNX_PATH)
    ap.add_argument("--set", action="append", default=[], help="name=라벨 json (기본 셋 대체)")
    args = ap.parse_args()
    sets = dict(kv.split("=", 1) for kv in args.set) if args.set else DEFAULT_SETS
    tax = json.loads((args.onnx.parent / "taxonomy.json").read_text(encoding="utf-8"))
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    baseline = (json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
                if BASELINE_PATH.exists() else None)

    reasons: list[str] = []
    summary: dict[str, dict] = {}
    for name, p in sets.items():
        path_s, _, source = str(p).partition("#")
        path = Path(path_s)
        if not path.exists():
            reasons.append(f"{name}: 라벨 셋 없음 ({path})")
            continue
        rows = load_rows(path, source or None)
        if not rows:
            print(f"[{name}] 라벨 행 0 — 건너뜀 (사용자 검토 대기)")
            continue
        preds = [predict(sess, tax, path.parent / r["file"]) for r in rows]
        rep = score(rows, preds, tax)
        reasons += check(name, rep, baseline)
        e = rep["electronics"]
        summary[name] = {"n": rep["n"], "coarse_acc": rep["coarse_acc"],
                         "elec_fp": e["false_positive"], "elec_fp_rate": e["fp_rate"],
                         "elec_recall": e["recall"], "elec_support": e["support"],
                         "reject": rep["reject"], "label_source": _label_source(rows)}
        print(f"[{name}] 대분류 {rep['coarse_hit']}/{rep['n']} · electronics 오탐 "
              f"{e['false_positive']} ({e['fp_rate']:.1%}) · recall {e['recall']} (n={e['support']})")

    enforce = os.getenv("GREENGUIDE_REALWORLD_GATE", "1") != "0"
    passed = not reasons
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "onnx": str(args.onnx), "pass": passed, "enforced": enforce, "reasons": reasons,
        "thresholds": {"elec_fp_max": GATE_ELEC_FP_MAX, "elec_recall_min": GATE_ELEC_RECALL_MIN,
                       "acc_max_drop": GATE_ACC_MAX_DROP},
        "sets": summary}, ensure_ascii=False, indent=1), encoding="utf-8")
    if passed:
        BASELINE_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
        print("실사용 게이트 PASS")
        return 0
    print("실사용 게이트 FAIL:\n  - " + "\n  - ".join(reasons))
    return 1 if enforce else 0


def _label_source(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        src = r["feedback"].get("source") or "user"
        out[src] = out.get(src, 0) + 1
    return out


if __name__ == "__main__":
    sys.exit(main())
