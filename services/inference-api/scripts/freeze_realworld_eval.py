"""실사용 평가셋 동결 — Opus 라벨 + 사용자 중재를 반영해 realworld_eval_v1.jsonl 로 고정한다.

라벨 우선순위: 사용자 중재(user_adjudicated) > Opus AI 라벨(ai_opus) > 기존.
  · 중재 행: pick=ai → AI 라벨, pick=prev → 기존 라벨, pick=other → other_label(대분류 또는 세분류 slug)
  · 중재 안 된 행: AI 라벨(= 기존과 대분류 일치가 전제) → label_source="ai_opus"
  · AI 대분류가 기존과 다른데 중재가 없는 행이 하나라도 있으면 실패(조용히 덮지 않는다)
  · 바뀐 행은 label_prev·label_prev_source 에 기존 값 보존

분할: final_holdout = sha256(id) 첫 바이트 % 5 == 0 (약 20%), 나머지 gate. 행마다 split 칸.
동결: 출력 파일을 읽기 전용(0444)으로 쓰고 <출력>.sha256 에 해시 기록. 이미 있으면 --force 없이는 거부.

입력:
  --ai     ai_eval_all.json   [{id, ai_coarse, ai_fine, ai_confidence(high|mid|low), note, ...}]
  --adj    중재 JSON          {id: {pick: ai|prev|other, other_label?}} 또는 [{id, pick, other_label?}]
  --dry-run 이면 파일을 쓰지 않고 요약만 출력(중재 파일 없이 점검할 때 --adj 생략 가능)

사용:
  python3 scripts/freeze_realworld_eval.py --ai <ai_eval_all.json> --adj <adjudication.json> --taxonomy <taxonomy.json>
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

GOLDEN = Path.home() / "Downloads" / "greenguide-golden"


def is_final_holdout(rid: str) -> bool:
    return hashlib.sha256(rid.encode()).digest()[0] % 5 == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=GOLDEN / "realworld_eval_set.jsonl")
    ap.add_argument("--ai", type=Path, required=True)
    ap.add_argument("--adj", type=Path)
    ap.add_argument("--taxonomy", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=GOLDEN / "realworld_eval_v1.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    tax = json.loads(args.taxonomy.read_text())
    f2c, coarse_set = tax["fine_to_coarse"], set(tax["coarse_labels"])

    def split_label(lab: str) -> tuple[str, str | None]:
        """slug → (대분류, 세분류|None)."""
        if lab in f2c:
            return f2c[lab], lab
        if lab in coarse_set:
            return lab, None
        raise SystemExit(f"taxonomy 에 없는 라벨: {lab}")

    rows = [json.loads(line) for line in args.src.expanduser().read_text().splitlines() if line.strip()]
    ai = {x["id"]: x for x in json.loads(args.ai.expanduser().read_text())}
    adj_raw = json.loads(args.adj.expanduser().read_text()) if args.adj else {}
    adj = {x["id"]: x for x in adj_raw} if isinstance(adj_raw, list) else adj_raw

    ids = {r["id"] for r in rows}
    errors = [f"AI 라벨 없음: {r['id']}" for r in rows if r["id"] not in ai]
    errors += [f"평가셋에 없는 중재 id: {k}" for k in adj if k not in ids]
    for k, a in adj.items():
        if a.get("pick") not in ("ai", "prev", "other"):
            errors.append(f"pick 값 오류 {k}: {a.get('pick')}")
        if a.get("pick") == "other" and not a.get("other_label"):
            errors.append(f"other_label 없음: {k}")
    for x in ai.values():
        if x.get("ai_fine") and split_label(x["ai_fine"])[0] != x["ai_coarse"]:
            errors.append(f"AI 세분류·대분류 불일치 {x['id']}: {x['ai_fine']}→{x['ai_coarse']}")
    unresolved = [r["id"] for r in rows if r["id"] in ai and ai[r["id"]]["ai_coarse"] != r["label"]
                  and r["id"] not in adj]
    if unresolved and not args.dry_run:
        errors.append(f"AI≠기존인데 중재 없음 {len(unresolved)}행: {unresolved}")
    if errors:
        print("동결 거부:\n  - " + "\n  - ".join(errors))
        return 1

    out_rows, changed = [], 0
    for r in rows:
        x = ai[r["id"]]
        prev_label, prev_fine, prev_src = r["label"], None, r["label_source"]
        if r.get("label_raw"):
            prev_fine = split_label(r["label_raw"])[1]
        a = adj.get(r["id"])
        if a:
            pick = a["pick"]
            if pick == "ai":
                label, fine = x["ai_coarse"], x.get("ai_fine")
            elif pick == "prev":
                label, fine = prev_label, prev_fine
            else:
                label, fine = split_label(a["other_label"])
            src, conf = "user_adjudicated", "user"
        else:
            label, fine, src, conf = x["ai_coarse"], x.get("ai_fine"), "ai_opus", x.get("ai_confidence")
        n = dict(r)
        n.update(label=label, label_fine=fine, label_source=src, label_confidence=conf,
                 adjudication=a.get("pick") if a else None,
                 split="final_holdout" if is_final_holdout(r["id"]) else "gate")
        n.pop("label_raw", None)
        if label != prev_label:
            changed += 1
        n["label_prev"], n["label_prev_source"] = prev_label, prev_src
        out_rows.append(n)

    split_n = collections.Counter(r["split"] for r in out_rows)
    summary = {
        "rows": len(out_rows),
        "label_source": dict(collections.Counter(r["label_source"] for r in out_rows)),
        "adjudication": dict(collections.Counter(r["adjudication"] for r in out_rows if r["adjudication"])),
        "label_changed_vs_prev": changed,
        "unresolved_ai_vs_prev": len(unresolved),
        "split": dict(split_n),
        "split_rule": "final_holdout = sha256(id)[0] % 5 == 0",
        "split_label_dist": {s: dict(collections.Counter(r["label"] for r in out_rows if r["split"] == s))
                             for s in split_n},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.dry_run:
        return 0

    out = args.out.expanduser()
    if out.exists() and not args.force:
        print(f"이미 동결됨: {out} — 덮으려면 --force")
        return 1
    if out.exists():
        os.chmod(out, 0o644)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows).encode()
    out.write_bytes(data)
    os.chmod(out, 0o444)
    digest = hashlib.sha256(data).hexdigest()
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n")
    print(f"동결 {out} sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
