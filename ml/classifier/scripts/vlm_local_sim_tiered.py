#!/usr/bin/env python3
"""qwen 먼저 → 다르면 유료 VLM 단계형 서빙 시뮬레이션 — 기존 기록만 사용(API·ollama 호출 0, W10).

규칙1: CNN 1위 대분류 == qwen 대분류면 qwen 답으로 끝(무료), 아니면 유료 실행의 서빙 결과.
규칙2: 규칙1 + qwen 확신도 ≥0.95(재질 답)면 채택. qwen 품목 생성 답(slug 없음)은 불일치로 센다.
사진 대응: record 의 VLM 발동 행 순서 = order.json 키 순서 = 유료 vlmlog 순서(발동 id 순서 일치를 단언).

사용: python3 vlm_local_sim_tiered.py --qwen <serving_A_qwen7b> --trigger <eval_v2/trigger> \\
          --paid A_r1m_always --out <sim.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HI_CONF = 0.95


def load_rows(qdir: Path, tdir: Path, paid: str) -> list[dict]:
    f2c = json.loads((qdir / "labels.json").read_text())["fine_to_coarse"]
    rec = json.loads((qdir / "A_record_novlm.json").read_text())
    pay = json.loads((tdir / f"{paid}.json").read_text())
    plog = [json.loads(ln) for ln in (tdir / f"{paid}_vlmlog.jsonl").read_text().splitlines()]
    order = json.loads((qdir / "order.json").read_text())
    trig = [r for r in rec["rows"] if r["vlm_triggered"]]
    assert len(trig) == len(order) == len(plog), (len(trig), len(order), len(plog))
    assert [r["upload_id"] for r in pay["rows"] if r["vlm_triggered"]] == [
        r["upload_id"] for r in trig], "발동 순서 불일치"
    served = {r["upload_id"]: r["pred_coarse"] for r in pay["rows"]}
    cnn = {r["upload_id"]: x["coarse_class"] for r, x in zip(rec["rows"], rec["raw"], strict=True)}

    def coarse_of(v: dict | None) -> str | None:
        return None if v is None or v.get("slug") is None else f2c.get(v["slug"], v["slug"])

    out = []
    for r, k, pl in zip(trig, order, plog, strict=True):
        f = qdir / "results" / f"{k}.json"
        if not f.exists():            # qwen 미측정 사진은 제외
            continue
        q = json.loads(f.read_text())["result"]
        uid = r["upload_id"]
        out.append({"id": uid, "truth": r["truth_coarse"], "cnn": cnn[uid],
                    "cnn_conf": r["coarse_confidence"],
                    "qwen": coarse_of(q) or f"item:{q and q.get('item_name')}",
                    "qwen_conf": q and q["confidence"],
                    "paid_vlm": pl.get("coarse") or coarse_of(pl) or f"item:{pl.get('item_name')}",
                    "paid_served": served[uid]})
    return out


def final(o: dict, rule: int) -> tuple[str, str]:
    hi = (rule == 2 and not o["qwen"].startswith("item:")
          and (o["qwen_conf"] or 0) >= HI_CONF)
    if o["qwen"] == o["cnn"] or hi:
        return o["qwen"], "qwen"
    return o["paid_served"], "paid"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qwen", type=Path, required=True, help="vlm_local_serving 캐시 폴더")
    ap.add_argument("--trigger", type=Path, required=True, help="유료 실행 기록 폴더")
    ap.add_argument("--paid", default="A_r1m_always", help="유료 실행 이름(<이름>.json·_vlmlog.jsonl)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = load_rows(a.qwen, a.trigger, a.paid)
    summary: dict = {"paid_run": a.paid, "n": len(rows),
                     "cnn_only": sum(o["cnn"] == o["truth"] for o in rows),
                     "qwen_only_vlm": sum(o["qwen"] == o["truth"] for o in rows),
                     "paid_always_served": sum(o["paid_served"] == o["truth"] for o in rows)}
    for rule in (1, 2):
        picks = [final(o, rule) for o in rows]
        for o, (f, src) in zip(rows, picks, strict=True):
            o.update({f"r{rule}_final": f, f"r{rule}_src": src, f"r{rule}_ok": f == o["truth"]})
        summary[f"rule{rule}"] = {
            "hit": sum(o[f"r{rule}_ok"] for o in rows),
            "paid_calls": sum(src == "paid" for _, src in picks),
            "qwen_wrong_accepted": [o["id"] for o in rows
                                    if o[f"r{rule}_src"] == "qwen" and not o[f"r{rule}_ok"]]}
    a.out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
