#!/usr/bin/env python3
"""서빙 경로 실측(realworld_eval_golden run) vs 모델 1차 패스(realworld_gate.predict) 장별 비교.

입력: 같은 셋(.jsonl)으로 돌린 realworld_eval_golden run 결과 json 1개 이상.
1차 패스는 이 스크립트가 같은 ONNX 로 다시 계산한다(회전 TTA + non_object 마스킹 + 대분류 게이트).
출력: 셋 지표(대분류·elec 오탐/비전자·recall), 1차 대비 개선/악화 행, VLM 발동·토큰·비용.

비용 단가(USD/MTok, 기본 Opus 5.5): 입력 4 · 출력 20 · 캐시 읽기 0.4 · 캐시 쓰기 5.

사용:
  .venv/bin/python scripts/realworld_serving_compare.py --set <gate.jsonl> --onnx <v14 classifier.onnx> \
      --run B=<v14_B_novlm.json> --run B+VLM=<v14_B_vlm.json> --out <compare.json>
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import _base  # noqa: F401 — sys.path 설정
from realworld_eval_golden import load_rows, score
from realworld_gate import predict

PRICE = {"input": 4.0, "output": 20.0, "cache_read": 0.4, "cache_write": 5.0}


def summary(rep: dict) -> dict:
    e = rep["electronics"]
    neg = rep["n"] - e["support"]
    return {"coarse": f"{rep['coarse_hit']}/{rep['n']}", "elec_fp": f"{e['false_positive']}/{neg}",
            "elec_recall": f"{round((e['recall'] or 0) * e['support'])}/{e['support']}",
            "reject": rep["reject"], "vlm_triggered": rep.get("vlm_triggered")}


def vlm_cost(usage: list[dict]) -> dict:
    tok = Counter()
    for u in usage:
        tok.update({k: u[k] for k in PRICE})
    usd = sum(tok[k] * PRICE[k] for k in PRICE) / 1e6
    return {"calls": len(usage), "tokens": dict(tok), "usd": round(usd, 4),
            "usd_per_call": round(usd / len(usage), 4) if usage else None,
            "models": dict(Counter(u["model"] for u in usage))}


def main() -> None:
    import onnxruntime as ort  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, type=Path)
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--run", action="append", required=True, help="이름=realworld_eval_golden 결과 json")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    rows = load_rows(a.set)
    tax = json.loads((a.onnx.parent / "taxonomy.json").read_text(encoding="utf-8"))
    sess = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    first = score(rows, [predict(sess, tax, a.set.parent / r["file"]) for r in rows], tax)
    out = {"set": str(a.set), "onnx": str(a.onnx), "first_pass": summary(first), "runs": {}}
    print(f"[1차 패스] {out['first_pass']}")
    base_ok = {r["upload_id"]: r for r in first["rows"]}
    for kv in a.run:
        name, path = kv.split("=", 1)
        res = json.loads(Path(path).read_text(encoding="utf-8"))
        rep = score(rows, res["raw"], tax)
        rep["vlm_triggered"] = res.get("vlm_triggered")
        cur = {r["upload_id"]: r for r in rep["rows"]}
        diff = {"improved": [], "worsened": []}
        for uid, r in cur.items():
            b = base_ok[uid]
            if r["ok"] != b["ok"]:
                diff["improved" if r["ok"] else "worsened"].append(
                    {"id": uid, "truth": r["truth_coarse"], "first": b["pred_coarse"],
                     "serving": r["pred_coarse"], "vlm": bool(r["vlm_triggered"])})
        s = summary(rep) | {"improved": len(diff["improved"]), "worsened": len(diff["worsened"]),
                            "vlm_cost": vlm_cost(res.get("vlm_usage") or []),
                            "wall_ms_p50": rep["wall_ms_p50"]}
        out["runs"][name] = s | diff
        print(f"[{name}] {s}")
    if a.out:
        a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"→ {a.out}")


if __name__ == "__main__":
    main()
