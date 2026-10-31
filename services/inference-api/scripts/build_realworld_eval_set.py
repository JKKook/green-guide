"""실사용 평가셋 통합 — 피드백(golden v2) + 테스터 + 골든 v3 를 하나의 JSONL 로.

근거: docs/plans/PRODUCTION_READINESS.md 결정 4 — 모델 팀(GG_CNN·GG_TRAIN) 배포 게이트의 공통 입력.

행 스키마 (한 줄 = 사진 1장):
  {id, image_path(절대), source, label(대분류|null), label_raw, label_source, ai_label, ai_doubt,
   ambiguous, holdout, sha256, model_v14:{coarse, fine, conf, level}}

  · feedback : golden_v2.json 중 feedback.label 있는 행. label_source="user_feedback".
               ai_label = 리더 AI 판정(ai_judgments_*.json 의 ai_coarse), ai_doubt = ai_label ≠ label
  · tester   : tester_rows.json + imgs/<id>.jpg. 라벨은 GG_CNN 라벨 파일(--tester-labels, tester93_golden.json).
               feedback.source="user" 행은 "user_feedback", 나머지 "agent_gg_cnn". ambiguous 는 그대로 옮김
  · golden_v3: golden_v3.json. label=null, label_source="pending_user_review", ai_label = v3 AI 판정
  · 중복: 같은 id → sha256 → dHash(≤6) 순으로 묶고 묶음당 1행 (feedback > tester > golden_v3, 그다음 오래된 것)
  · model_v14: --local-api(COLLECT_UPLOADS=false, VLM 끔) /predict-hier 로 전 행 재판정

사용 (PIL 때문에 arm64 classifier venv):
  arch -arm64 ml/classifier/.venv/bin/python scripts/build_realworld_eval_set.py \\
      --tester-labels <GG_CNN 라벨 파일> --ai-judgments <리더 스크래치>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_golden_set import _dhash, is_holdout, post_image  # noqa: E402

GOLDEN = Path.home() / "Downloads" / "greenguide-golden"
REPO = Path(__file__).resolve().parents[3]
TESTER = REPO / "ml" / "classifier" / "outputs" / "logs" / "tester_eval"
PRIORITY = {"feedback": 0, "tester": 1, "golden_v3": 2}


def load_json_list(*paths: Path) -> list[dict]:
    out = []
    for p in paths:
        if p.exists():
            out += json.loads(p.read_text())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-dir", type=Path, default=GOLDEN)
    ap.add_argument("--tester-dir", type=Path, default=TESTER)
    ap.add_argument("--tester-labels", type=Path, required=True,
                    help="GG_CNN 라벨 파일 tester93_golden.json ({items:[{upload_id, feedback:{label, source, ambiguous}}]})")
    ap.add_argument("--ai-judgments", type=Path, required=True, help="ai_judgments_{a,b,v3_a,v3_b}.json 폴더")
    ap.add_argument("--taxonomy", type=Path, required=True, help="v14 taxonomy.json (fine_to_coarse)")
    ap.add_argument("--local-api", default="http://127.0.0.1:8765")
    ap.add_argument("--out", type=Path, default=GOLDEN / "realworld_eval_set.jsonl")
    ap.add_argument("--max-hamming", type=int, default=6)
    args = ap.parse_args()

    tax = json.loads(args.taxonomy.read_text())
    f2c, coarse_set = tax["fine_to_coarse"], set(tax["coarse_labels"])

    def to_coarse(label: str | None) -> str | None:
        if label is None:
            return None
        if label in f2c:
            return f2c[label]
        if label in coarse_set:
            return label
        raise SystemExit(f"대분류로 매핑 못 하는 라벨: {label}")

    gd, aj = args.golden_dir.expanduser(), args.ai_judgments.expanduser()
    ai_v2 = {x["upload_id"]: x["ai_coarse"] for x in load_json_list(aj / "ai_judgments_a.json", aj / "ai_judgments_b.json")}
    ai_v3 = {x["upload_id"]: x["ai_coarse"]
             for x in load_json_list(aj / "ai_judgments_v3_a.json", aj / "ai_judgments_v3_b.json")}

    cands: list[dict] = []
    for x in json.loads((gd / "golden_v2.json").read_text())["items"]:
        fb = x.get("feedback") or {}
        if not fb.get("label"):
            continue
        label = to_coarse(fb["label"])
        ai = ai_v2.get(x["upload_id"])
        cands.append({"id": x["upload_id"], "path": gd / x["file"], "source": "feedback", "at": x["uploaded_at"],
                      "label": label, "label_raw": fb["label"], "label_source": "user_feedback",
                      "ai_label": ai, "ai_doubt": ai is not None and ai != label, "server": True})

    td = args.tester_dir.expanduser()
    tlabel = {x["upload_id"]: x["feedback"] for x in json.loads(args.tester_labels.expanduser().read_text())["items"]}
    for r in json.loads((td / "tester_rows.json").read_text()):
        fb = tlabel.get(r["id"]) or {}
        user = fb.get("source") == "user"
        raw_label = fb.get("label")
        cands.append({"id": r["id"], "path": td / "imgs" / f"{r['id']}.jpg", "source": "tester", "at": r["uploaded_at"],
                      "label": to_coarse(raw_label), "label_raw": raw_label, "ambiguous": bool(fb.get("ambiguous")),
                      "label_source": "user_feedback" if user else ("agent_gg_cnn" if raw_label else "missing"),
                      "ai_label": None, "ai_doubt": None, "server": True})

    for x in json.loads((gd / "golden_v3.json").read_text())["items"]:
        cands.append({"id": x["upload_id"], "path": gd / x["file"], "source": "golden_v3", "at": x["uploaded_at"],
                      "label": None, "label_raw": None, "label_source": "pending_user_review",
                      "ai_label": ai_v3.get(x["upload_id"]), "ai_doubt": None, "server": x["source"] == "server"})

    for c in cands:
        data = c["path"].read_bytes()
        c["data"], c["sha256"], c["dhash"] = data, hashlib.sha256(data).hexdigest(), _dhash(data)

    parent = list(range(len(cands)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            if a["id"] == b["id"] or a["sha256"] == b["sha256"] \
                    or bin(a["dhash"] ^ b["dhash"]).count("1") <= args.max_hamming:
                parent[find(i)] = find(j)
    groups: dict[int, list[dict]] = {}
    for i, c in enumerate(cands):
        groups.setdefault(find(i), []).append(c)

    keep, dup_log = [], []
    for members in groups.values():
        members.sort(key=lambda c: (PRIORITY[c["source"]], c["at"]))
        k = members[0]
        # 유지 행에 라벨이 없고 묶음 안에 사용자 라벨이 있으면 승계
        if k["label"] is None:
            lab = next((m for m in members[1:] if m["label"] is not None), None)
            if lab:
                k.update(label=lab["label"], label_raw=lab["label_raw"], label_source=lab["label_source"],
                         ambiguous=lab.get("ambiguous", False))
        if len(members) > 1:
            dup_log.append({"keep": f"{k['source']}:{k['id']}",
                            "removed": [f"{m['source']}:{m['id']}" for m in members[1:]]})
        keep.append(k)
    keep.sort(key=lambda c: (PRIORITY[c["source"]], c["at"]))

    requests_ok = 0
    lines = []
    for i, c in enumerate(keep, 1):
        mime = "image/webp" if c["path"].suffix == ".webp" else "image/jpeg"
        res = post_image(args.local_api, "/predict-hier", c["path"].name, c["data"], mime)
        if res.get("upload_id"):
            raise SystemExit("로컬 서버가 업로드를 기록함 — GREENGUIDE_API_COLLECT_UPLOADS=false 로 다시 띄울 것")
        if "error" in res:
            mv = {"error": res["error"]}
        else:
            requests_ok += 1
            lvl = res["display_level"]
            mv = {"coarse": res["coarse_class"], "fine": res.get("fine_class"),
                  "conf": res["fine_confidence"] if lvl == "fine" else res["coarse_confidence"], "level": lvl}
        lines.append({
            "id": c["id"], "image_path": str(c["path"].resolve()), "source": c["source"],
            "label": c["label"], "label_raw": c["label_raw"], "label_source": c["label_source"],
            "ai_label": c["ai_label"], "ai_doubt": c["ai_doubt"], "ambiguous": c.get("ambiguous", False),
            "holdout": is_holdout(c["id"]) if c["server"] else False,
            "sha256": c["sha256"], "model_v14": mv,
        })
        print(f"[{i}/{len(keep)}] {c['source']} {c['id']} label={c['label']} v14={mv.get('coarse', 'err')}", flush=True)

    out = args.out.expanduser()
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
    (out.with_suffix(".dups.json")).write_text(json.dumps(dup_log, ensure_ascii=False, indent=1))
    by = {s: sum(1 for c in cands if c["source"] == s) for s in PRIORITY}
    print(f"in={by} total={len(cands)} dup_removed={len(cands) - len(keep)} final={len(keep)} "
          f"v14_ok={requests_ok} → {out}")
    return 0 if requests_ok == len(keep) else 1


if __name__ == "__main__":
    sys.exit(main())
