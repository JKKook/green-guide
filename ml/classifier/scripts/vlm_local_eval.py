#!/usr/bin/env python3
"""로컬 VLM 라벨 평가 — 기준 라벨 대비 대분류 일치율·클래스별·확신 구간별 (W10).

기준(--ref) 세 형식:
  ai_eval_all.json   Opus 라벨 리스트(id·ai_coarse·ai_fine)
  *.jsonl            동결 평가셋 eval_v1(id·label·label_fine·split) — --split gate 로 게이트분만
  manifest.csv       하드 네거티브(file·clip_top) — CLIP 라우팅 라벨과의 불일치 비율
예측(--pred): vlm_local_label.py 출력 JSONL.

채택 게이트(--gate): 대분류 일치 ≥85% 또는 확신 ≥0.7 구간 ≥90%.

실행 예:
  .venv/bin/python scripts/vlm_local_eval.py --gate \
      --ref <scratchpad>/ai_eval_all.json --pred outputs/logs/vlm_local/qwen2.5vl_7b/labels.jsonl
출력: --pred 옆 eval_<ref 이름>.json + 콘솔 요약
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import _base
from greenguide_common.taxonomy import COARSE_LABELS, FINE_TO_COARSE

GATE_ALL = 0.85
GATE_CONF = 0.90
CONF_CUT = 0.7
BINS = ((0.9, 1.01), (0.7, 0.9), (0.5, 0.7), (0.0, 0.5))


def to_coarse(slug: str | None) -> str | None:
    if slug in COARSE_LABELS:
        return slug
    return FINE_TO_COARSE.get(slug)


def half_of(item_id: str) -> str:
    """과적합 교차 확인용 고정 반분 — sha256(id) 짝수 A, 홀수 B."""
    return "A" if int(hashlib.sha256(item_id.encode()).hexdigest(), 16) % 2 == 0 else "B"


def load_ref(path: Path, split: str | None = None) -> dict[str, dict]:
    """id → {coarse, fine?}. split 이 있으면 jsonl 의 split 칸이 같은 행만."""
    if path.suffix == ".csv":
        with path.open(newline="") as f:
            return {r["file"]: {"coarse": to_coarse(r["clip_top"]), "fine": None}
                    for r in csv.DictReader(f)}
    if path.suffix == ".jsonl":
        rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
        return {r["id"]: {"coarse": to_coarse(r["label"]), "fine": r.get("label_fine"),
                          **({"state": r["state"]} if r.get("state") else {})}
                for r in rows if split is None or r.get("split") == split}
    return {r["id"]: {"coarse": r["ai_coarse"], "fine": r.get("ai_fine")}
            for r in json.loads(path.read_text())}


def load_pred(path: Path) -> dict[str, dict]:
    """마지막 성공 행이 이긴다(재실행 누적 대비)."""
    preds: dict[str, dict] = {}
    for ln in path.read_text().splitlines():
        r = json.loads(ln)
        if "error" not in r:
            preds[r["id"]] = r
    return preds


def _acc(pairs: list[tuple[str, str]]) -> dict:
    n = len(pairs)
    k = sum(a == b for a, b in pairs)
    return {"n": n, "agree": k, "rate": round(k / n, 4) if n else None}


def eval_state(ref: dict[str, dict], pred: dict[str, dict]) -> dict | None:
    """상태 일치·'전부 clean' 기준선·dirty 정밀도/재현율 (기준·예측 둘 다 state 가 있는 행)."""
    ids = [i for i in ref if "state" in ref[i] and "state" in pred.get(i, {})]
    if not ids:
        return None
    pairs = [(ref[i]["state"], pred[i]["state"]) for i in ids]
    tp = sum(r == p == "dirty" for r, p in pairs)
    n_pred = sum(p == "dirty" for _, p in pairs)
    n_true = sum(r == "dirty" for r, _ in pairs)
    return {"acc": _acc(pairs), "all_clean": _acc([(r, "clean") for r, _ in pairs]),
            "dirty": {"tp": tp, "pred": n_pred, "true": n_true,
                      "precision": round(tp / n_pred, 4) if n_pred else None,
                      "recall": round(tp / n_true, 4) if n_true else None},
            "confusions": Counter(f"{r}→{p}" for r, p in pairs if r != p).most_common(10)}


def evaluate(ref: dict[str, dict], pred: dict[str, dict]) -> dict:
    ids = [i for i in ref if i in pred]
    pairs = [(ref[i]["coarse"], pred[i]["coarse"]) for i in ids]
    fine_ids = [i for i in ids if ref[i].get("fine")]
    per_class: dict[str, list] = defaultdict(list)
    for r, p in pairs:
        per_class[r].append((r, p))
    bins = {}
    for lo, hi in BINS:
        sub = [(ref[i]["coarse"], pred[i]["coarse"]) for i in ids
               if lo <= pred[i]["confidence"] < hi]
        bins[f"{lo:.1f}-{min(hi, 1.0):.1f}"] = _acc(sub)
    hi_conf = [(ref[i]["coarse"], pred[i]["coarse"]) for i in ids
               if pred[i]["confidence"] >= CONF_CUT]
    lat = [pred[i]["latency_s"] for i in ids if "latency_s" in pred[i]]
    conf_pairs = Counter(f"{r}→{p}" for r, p in pairs if r != p)
    elec_ref = [p for r, p in pairs if r == "electronics"]
    elec_fp = sum(1 for r, p in pairs if r != "electronics" and p == "electronics")
    return {
        "ref_n": len(ref), "pred_n": len(ids), "missing": len(ref) - len(ids),
        "coarse": _acc(pairs),
        "fine": _acc([(ref[i]["fine"], pred[i]["fine"]) for i in fine_ids]),
        "conf_ge_0.7": {**_acc(hi_conf), "coverage": round(len(hi_conf) / len(ids), 4)
                        if ids else None},
        "conf_bins": bins,
        "per_class": {c: _acc(v) for c, v in sorted(per_class.items())},
        "top_confusions": conf_pairs.most_common(10),
        "electronics": {"recall": _acc([("electronics", p) for p in elec_ref]),
                        "false_pos": elec_fp,
                        "non_elec_n": sum(1 for r, _ in pairs if r != "electronics")},
        "latency_s": ({"mean": round(statistics.mean(lat), 2),
                       "p50": round(statistics.median(lat), 2),
                       "p90": round(sorted(lat)[int(0.9 * (len(lat) - 1))], 2)}
                      if lat else None),
        "state": eval_state(ref, pred),
    }


def gate(res: dict) -> tuple[bool, str]:
    a = res["coarse"]["rate"] or 0.0
    c = res["conf_ge_0.7"]["rate"] or 0.0
    ok = a >= GATE_ALL or c >= GATE_CONF
    return ok, (f"{'PASS' if ok else 'FAIL'} — 전체 {a:.1%} (≥{GATE_ALL:.0%}) / "
                f"확신≥{CONF_CUT} {c:.1%} 커버 {res['conf_ge_0.7']['coverage']} "
                f"(≥{GATE_CONF:.0%})")


def main() -> None:
    p = _base.make_parser("vlm_local_eval", __doc__.splitlines()[0], seed=None)
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--gate", action="store_true", help="채택 게이트 판정 출력")
    p.add_argument("--split", help="jsonl 기준의 split 칸 필터 (gate | final_holdout)")
    p.add_argument("--half", choices=("A", "B"), help="sha256(id) 반분 필터(교차 확인)")
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    ref = load_ref(a.ref, a.split)
    if a.half:
        ref = {i: v for i, v in ref.items() if half_of(i) == a.half}
    res = evaluate(ref, load_pred(a.pred))
    if a.gate:
        res["gate"] = gate(res)[1]
    out = a.out or a.pred.parent / f"eval_{a.ref.stem}{'_' + a.split if a.split else ''}{'_' + a.half if a.half else ''}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    c = res["coarse"]
    print(f"대분류 일치 {c['agree']}/{c['n']} ({c['rate']}) · 세부 {res['fine']['rate']} · "
          f"누락 {res['missing']} · 지연 {res['latency_s']}")
    print(f"확신≥{CONF_CUT}: {res['conf_ge_0.7']}")
    for k, v in res["conf_bins"].items():
        print(f"  conf {k}: {v}")
    for k, v in res["per_class"].items():
        print(f"  {k:12s} {v['agree']}/{v['n']}")
    print("혼동 상위:", res["top_confusions"])
    print("electronics:", res["electronics"])
    if res["state"]:
        print("state:", res["state"])
    if a.gate:
        print("게이트:", res["gate"])
    print("→", out)


if __name__ == "__main__":
    main()
