#!/usr/bin/env python3
"""eval_v2 동결 — v1 gate/holdout jsonl + 라벨 확인 응답 → v2 jsonl.

응답: 로컬 review_v2.html 의 내보내기 JSON(review_v2_<날짜>.json)만 받는다(아티팩트 db 는 11-03 삭제).

분류 체계 v2(TAXONOMY_V2.md, 결정 32): 재질 대분류 14 · 세부 · 상태 4 · 마크 보임.
행 순서·split 은 v1 그대로 둔다(기존 서빙 run json 의 raw 순서와 맞춰 재호출 없이 재채점하기 위해).
확인 완료(confirmed)가 아닌 행이 하나라도 있으면 동결하지 않는다.

사용:
  python scripts/realworld_eval_v2_freeze.py --v1 <eval_v1_gate.jsonl> --v1 <eval_v1_final_holdout.jsonl> \
      --answers <review_v2_<날짜>.json> --out-dir <outputs/logs>
출력: split 별 eval_v2_*.jsonl(채점용) + 합본 realworld_eval_v2.jsonl(동결 정본)
      + realworld_eval_v2_excluded.json(평가 제외 id·split 별 분모 변화), 각 sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

STATES = {"clean", "dirty", "broken", "composite"}
REQUIRED_FINE = {"metal": {"metal_can", "metal_scrap"}, "food_waste": {"food_waste", "food_inedible"}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", action="append", required=True, type=Path)
    ap.add_argument("--answers", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--normalize", type=Path,
                    help="리더 정규화 목록 {id: {coarse, fine, reason, by}} — 응답 원본은 두고 동결본에만 반영·기록")
    a = ap.parse_args()

    j = json.loads(a.answers.read_text(encoding="utf-8"))
    if j.get("format") == "greenguide.review_v2":  # review_v2.html "JSON 내보내기" 파일
        ans = j["answers"]
    elif j and all(isinstance(v, dict) and "confirmed" in v for v in j.values()):
        ans = j  # 브라우저 localStorage 에서 복구한 원본(id → 답) — 11-03 사용자 채점분
    else:
        raise SystemExit("review_v2 응답 JSON 이 아님 — 내보내기 형식도 id→답 dict 도 아님")
    print(f"answers {a.answers} sha256 {hashlib.sha256(a.answers.read_bytes()).hexdigest()} n={len(ans)}")
    norm = json.loads(a.normalize.read_text(encoding="utf-8")) if a.normalize else {}
    if norm:
        print(f"normalize {a.normalize} sha256 {hashlib.sha256(a.normalize.read_bytes()).hexdigest()} n={len(norm)}")
    missing, outs, excluded, applied = [], {}, {}, []
    for src in a.v1:
        rows, exc = [], []
        for line in src.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            v = ans.get(r["id"])
            if v and v.get("excluded"):  # 사람 얼굴·신체 위주 사진 — 사용자가 평가셋에서 뺌(재질·상태 칸 비어도 됨)
                exc.append({"id": r["id"], "split": r.get("split"), "reason": v.get("exclude_reason")})
                continue
            ok = v and v.get("confirmed") and v.get("mark") is not None and (
                v["coarse"] == "non_object" or v.get("state") in STATES) and (
                v["coarse"] not in REQUIRED_FINE or v.get("fine") in REQUIRED_FINE[v["coarse"]])
            if not ok:
                missing.append(r["id"])
                continue
            if r["id"] in norm:
                nv = norm[r["id"]]
                v = v | {"coarse": nv["coarse"], "fine": nv.get("fine"),
                         "normalized": {"from": [v["coarse"], v.get("fine")], "reason": nv["reason"], "by": nv["by"]}}
                applied.append(r["id"])
            rows.append(r | {"label_v1": r["label"], "label_fine_v1": r["label_fine"],
                             "label": v["coarse"], "label_fine": v.get("fine"), "state": v.get("state"),
                             "mark_visible": bool(v["mark"]), "memo": v.get("memo") or None,
                             # 물체 수(single|multi)·판정 대상 점(표시 좌표 0~1) — 11-03 이후 페이지만 기록.
                             # multi 인데 점이 없으면 '화면에서 가장 큰 물체' 기준.
                             "objects": v.get("objects"), "target_xy": v.get("target_xy"),
                             "normalized": v.get("normalized"),
                             "label_source": "user_v2", "taxonomy": "v2"})
        name = src.name.replace("eval_v1", "eval_v2")
        outs[a.out_dir / name] = rows
        excluded[name] = {"n_v1": len(rows) + len(exc), "n_v2": len(rows), "excluded": exc}
    outs[a.out_dir / "realworld_eval_v2.jsonl"] = [r for rows in list(outs.values()) for r in rows]
    if set(norm) - set(applied):
        raise SystemExit(f"정규화 대상이 동결본에 없음(제외·미존재): {sorted(set(norm) - set(applied))}")
    if missing:
        raise SystemExit(f"확인 미완료 {len(missing)}장 — 동결 중단: {missing[:10]}")
    for path, rows in outs.items():
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        changed = sum(r["label"] != r["label_v1"] for r in rows)
        no_obj = sum(r["objects"] is None for r in rows)
        print(f"{path} n={len(rows)} 재질 변경 {changed} 물체 수 빈 장 {no_obj} sha256 {sha}")
    # 제외 목록 — 기준선·이후 모든 비교가 같은 분모를 쓰도록 함께 동결
    ex = a.out_dir / "realworld_eval_v2_excluded.json"
    ex.write_text(json.dumps(excluded, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, e in excluded.items():
        print(f"  {name}: {e['n_v1']} → {e['n_v2']} (제외 {len(e['excluded'])})")
    print(f"{ex} sha256 {hashlib.sha256(ex.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
