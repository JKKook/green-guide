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


def v2_axes(truth: dict[str, dict], ids: list[str], raws: list[dict]) -> dict | None:
    """분류 체계 v2 축(state·fine_v2) — 셋에 정답이 있고 응답에 칸이 있을 때만. 가드 아님, 기록용."""
    if not any("state" in t for t in truth.values()) or not any("state" in r for r in raws):
        return None
    st_n = st_hit = st_pred = fv_n = fv_hit = 0
    for uid, r in zip(ids, raws, strict=True):
        t = truth[uid]
        if t.get("state"):
            st_n += 1
            if r.get("state"):
                st_pred += 1
                st_hit += r["state"] == t["state"]
        if t.get("label") in ("metal", "food_waste") and t.get("label_fine"):
            fv_n += 1
            fv_hit += r.get("fine_v2") == t["label_fine"]
    return {"state_hit": f"{st_hit}/{st_n}", "state_answered": f"{st_pred}/{st_n}",
            "fine_v2_hit": f"{fv_hit}/{fv_n}"}


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
    truth = {json.loads(line)["id"]: json.loads(line) for line in a.set.read_text(encoding="utf-8").splitlines()}
    tax = json.loads((a.onnx.parent / "taxonomy.json").read_text(encoding="utf-8"))
    sess = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    first = score(rows, [predict(sess, tax, a.set.parent / r["file"]) for r in rows], tax)
    out = {"set": str(a.set), "onnx": str(a.onnx), "first_pass": summary(first), "runs": {}}
    print(f"[1차 패스] {out['first_pass']}")
    base_ok = {r["upload_id"]: r for r in first["rows"]}
    for kv in a.run:
        name, path = kv.split("=", 1)
        res = json.loads(Path(path).read_text(encoding="utf-8"))
        # 셋에서 행이 빠졌을 수 있다(eval_v2 평가 제외) — 실행 당시 행 순서의 upload_id 로 맞춘다
        by_id = {r["upload_id"]: raw for r, raw in zip(res["rows"], res["raw"], strict=True)}
        raws = [by_id[r["upload_id"]] for r in rows]
        rep = score(rows, raws, tax)
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
                            "wall_ms_p50": rep["wall_ms_p50"],
                            "v2_axes": v2_axes(truth, [r["upload_id"] for r in rows], raws)}
        out["runs"][name] = s | diff
        print(f"[{name}] {s}")
    if a.out:
        a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"→ {a.out}")


if __name__ == "__main__":
    main()
