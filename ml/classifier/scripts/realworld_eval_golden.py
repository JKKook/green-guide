#!/usr/bin/env python3
"""골든셋 실사용 평가 — 피드백 라벨 행으로 대분류·세부 혼동행렬과 electronics 편향 계측.

입력: golden_v2.json(items 중 feedback.label 이 있는 행). 정답 = 사용자 피드백 라벨
(fine 이면 taxonomy 사이드카로 대분류 롤업).

모드
  replay  — 골든셋에 저장된 판정(hier_local)을 그대로 채점. 모델 실행 없음.
  run     — 서버 코드(/predict-hier)를 TestClient 로 in-process 실행해 채점.
            업로드 기록(record_safely)은 막고, VLM 은 --vlm 이 없으면 키를 비워 끈다.
            서버 플래그는 --env KEY=VAL 로 넘긴다 (설정 1개 = 프로세스 1개).

판정 규칙: display_level=reject 면 display_class(etc/non_object), 아니면 coarse_class.

실행 (api venv 는 x86_64 — Rosetta):
  arch -x86_64 services/inference-api/.venv/bin/python \
      ml/classifier/scripts/realworld_eval_golden.py replay
  arch -x86_64 services/inference-api/.venv/bin/python \
      ml/classifier/scripts/realworld_eval_golden.py run --api-root services/inference-api \
      --model-dir <onnx+taxonomy 폴더> --name base --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

GOLDEN = Path.home() / "Downloads" / "greenguide-golden" / "golden_v2.json"
TAXONOMY = (Path(__file__).resolve().parents[1]
            / "outputs" / "models" / "cnn_hier" / "taxonomy.json")
FOCUS = "electronics"


def load_rows(golden: Path, source: str | None = None) -> list[dict]:
    """라벨 행. golden_v*.json(items) 또는 GG_DATA 통합셋 .jsonl(한 줄 1장, label=null 제외).

    source: .jsonl 의 source 칸(feedback·tester·golden_v3) 필터.
    """
    if golden.suffix == ".jsonl":
        rows = []
        for line in golden.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("label") is None or (source and r["source"] != source):
                continue
            rows.append({"upload_id": r["id"], "file": r["image_path"],
                         "feedback": {"label": r["label"], "source": r["label_source"],
                                      "ambiguous": r.get("ambiguous", False)},
                         "meta": {}})
        return rows
    d = json.loads(golden.read_text(encoding="utf-8"))
    return [it for it in d["items"] if (it.get("feedback") or {}).get("label")]


def truth_of(label: str, f2c: dict[str, str], coarse: list[str]) -> tuple[str, str | None]:
    """피드백 라벨 → (대분류, 세부 또는 None)."""
    if label in f2c:
        return f2c[label], label
    if label in coarse:
        return label, None
    raise ValueError(f"taxonomy 에 없는 라벨: {label}")


def pred_of(res: dict) -> tuple[str, str | None]:
    if res.get("display_level") == "reject":
        return res.get("display_class") or "etc", None
    return res["coarse_class"], res.get("fine_class")


def score(rows: list[dict], preds: list[dict], tax: dict) -> dict:
    f2c, coarse = tax["fine_to_coarse"], tax["coarse_labels"]
    cm: Counter = Counter()
    fine_cm: Counter = Counter()
    per_row = []
    for row, res in zip(rows, preds, strict=True):
        tc, tf = truth_of(row["feedback"]["label"], f2c, coarse)
        pc, pf = pred_of(res)
        cm[(tc, pc)] += 1
        top1_fine = (res.get("fine_top5") or [{}])[0].get("slug")
        if tf is not None:
            fine_cm[(tf, pf or top1_fine or "-")] += 1
        per_row.append({
            "upload_id": row["upload_id"], "truth": row["feedback"]["label"],
            "truth_coarse": tc, "pred_coarse": pc, "pred_fine": pf,
            "display_level": res.get("display_level"),
            "coarse_confidence": res.get("coarse_confidence"),
            "elec_prob": (res.get("coarse_probabilities") or {}).get(FOCUS),
            "model_arch": res.get("model_arch"),
            "vlm_triggered": res.get("_vlm_triggered"),
            "ok": tc == pc,
            # 관대 채점: 정답이 etc/non_object 면 reject 계열(etc·non_object 표시)도 정답
            "ok_lenient": tc == pc or (tc in ("etc", "non_object")
                                       and res.get("display_level") == "reject"),
            "wall_ms": res.get("_wall_ms"),
        })
    n = len(per_row)
    hit = sum(r["ok"] for r in per_row)
    labels = sorted({t for t, _ in cm} | {p for _, p in cm},
                    key=lambda c: coarse.index(c) if c in coarse else 99)
    per_class = {}
    for c in labels:
        tp = cm[(c, c)]
        npred = sum(v for (_, p), v in cm.items() if p == c)
        ntrue = sum(v for (t, _), v in cm.items() if t == c)
        per_class[c] = {"support": ntrue, "predicted": npred, "tp": tp,
                        "precision": round(tp / npred, 3) if npred else None,
                        "recall": round(tp / ntrue, 3) if ntrue else None}
    errors = [r for r in per_row if not r["ok"]]
    elec_fp = [r for r in errors if r["pred_coarse"] == FOCUS]
    fine_rows = sum(fine_cm.values())
    ws = sorted(r["wall_ms"] for r in per_row if r["wall_ms"] is not None)
    return {
        "n": n, "coarse_hit": hit, "coarse_acc": round(hit / n, 4) if n else None,
        "errors": len(errors),
        "coarse_hit_lenient": sum(r["ok_lenient"] for r in per_row),
        "reject": sum(r["display_level"] == "reject" for r in per_row),
        "wall_ms_p50": ws[len(ws) // 2] if ws else None,
        "vlm_triggered": sum(bool(r["vlm_triggered"]) for r in per_row),
        "electronics": {
            "predicted": per_class.get(FOCUS, {}).get("predicted", 0),
            "false_positive": len(elec_fp),
            "share_of_errors": round(len(elec_fp) / len(errors), 3) if errors else None,
            "fp_by_truth": dict(Counter(r["truth_coarse"] for r in elec_fp)),
            "recall": per_class.get(FOCUS, {}).get("recall"),
            "support": per_class.get(FOCUS, {}).get("support", 0),
        },
        "error_pairs": dict(Counter(f"{r['truth_coarse']}->{r['pred_coarse']}"
                                    for r in errors).most_common()),
        "per_class": per_class,
        "confusion": {f"{t}->{p}": v for (t, p), v in sorted(cm.items())},
        "fine": {"n": fine_rows,
                 "hit": sum(v for (t, p), v in fine_cm.items() if t == p),
                 "confusion": {f"{t}->{p}": v for (t, p), v in sorted(fine_cm.items())}},
        "rows": per_row,
    }


def confusion_md(rep: dict, tax: dict) -> str:
    coarse = tax["coarse_labels"]
    cm = {tuple(k.split("->")): v for k, v in rep["confusion"].items()}
    labels = sorted({t for t, _ in cm} | {p for _, p in cm},
                    key=lambda c: coarse.index(c) if c in coarse else 99)
    head = "| 정답＼예측 | " + " | ".join(labels) + " | 계 |"
    lines = [head, "|" + "---|" * (len(labels) + 2)]
    for t in labels:
        cells = [str(cm.get((t, p), "")) for p in labels]
        lines.append(f"| {t} | " + " | ".join(cells)
                     + f" | {sum(cm.get((t, p), 0) for p in labels)} |")
    return "\n".join(lines)


def run_server(rows: list[dict], api_root: Path, golden_dir: Path, use_vlm: bool,
               send_meta: bool = True) -> list[dict]:
    """서버 /predict-hier 를 in-process 로 실행. 부수효과(업로드 기록)는 차단."""
    os.chdir(api_root)  # load_dotenv·상대경로 기준
    os.environ["SUPABASE_URL"] = ""
    os.environ["SUPABASE_KEY"] = ""
    if not use_vlm:
        os.environ["ANTHROPIC_API_KEY"] = ""
    sys.path.insert(0, str(api_root))
    import src.routers.inference as inf  # noqa: PLC0415
    import src.vlm_fallback as vf  # noqa: PLC0415
    from fastapi.testclient import TestClient  # noqa: PLC0415
    from src.api import app  # noqa: PLC0415

    inf.record_safely = lambda *a, **k: None  # 업로드 기록 금지
    vlm_calls: list[int] = [0]
    orig_classify = vf.VlmFallback.classify

    def counting_classify(self, *a, **k):  # 중재 발동 수 계측 (키 없으면 None 반환)
        vlm_calls[0] += 1
        return orig_classify(self, *a, **k)

    vf.VlmFallback.classify = counting_classify
    usage: list[dict] = []
    if use_vlm:  # 실호출 토큰 사용량 계측 (비용 추정용)
        import anthropic.resources.messages as am  # noqa: PLC0415

        orig_create = am.Messages.create

        def logging_create(self, *a, **k):
            msg = orig_create(self, *a, **k)
            u = msg.usage
            usage.append({"model": msg.model, "stop_reason": msg.stop_reason,
                          "input": u.input_tokens, "output": u.output_tokens,
                          "cache_read": u.cache_read_input_tokens or 0,
                          "cache_write": u.cache_creation_input_tokens or 0})
            return msg

        am.Messages.create = logging_create
    out = []
    with TestClient(app) as client:
        for row in rows:
            img = golden_dir / row["file"]
            meta = row.get("meta") or {}
            data = {k: str(meta[k]) for k in ("orientation", "capture_mode")
                    if send_meta and meta.get(k) is not None}
            before = vlm_calls[0]
            t0 = time.perf_counter()
            with img.open("rb") as fh:
                r = client.post("/predict-hier", data=data,
                                files={"image": (img.name, fh, "image/webp")})
            r.raise_for_status()
            res = r.json()
            res["_vlm_triggered"] = vlm_calls[0] > before
            res["_wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            out.append(res)
            print(".", end="", flush=True)
    print()
    if usage:
        out[0]["_vlm_usage"] = usage
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["replay", "run"])
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--taxonomy", type=Path, default=TAXONOMY)
    ap.add_argument("--api-root", type=Path)
    ap.add_argument("--model-dir", type=Path, help="classifier.onnx+taxonomy.json 폴더")
    ap.add_argument("--env", action="append", default=[], help="서버 플래그 KEY=VAL")
    ap.add_argument("--vlm", action="store_true", help="VLM 폴백 실제 호출 허용")
    ap.add_argument("--no-meta", action="store_true",
                    help="촬영 메타(orientation·capture_mode) 미전송 — 골든셋 hier_local 재추론 조건")
    ap.add_argument("--source", default=None, help=".jsonl 통합셋의 source 필터")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/logs/realworld_golden"))
    args = ap.parse_args()

    rows = load_rows(args.golden, args.source)
    tax_path = (args.model_dir / "taxonomy.json") if args.model_dir else args.taxonomy
    tax = json.loads(tax_path.read_text(encoding="utf-8"))
    out_dir = args.out.resolve()
    if args.mode == "replay":
        preds = [r["hier_local"] for r in rows]
    else:
        for kv in args.env:
            k, v = kv.split("=", 1)
            os.environ[k] = v
        if args.model_dir:
            os.environ["GREENGUIDE_API_HIER_MODEL_PATH"] = str(
                (args.model_dir / "classifier.onnx").resolve())
        preds = run_server(rows, args.api_root.resolve(), args.golden.parent, args.vlm,
                           send_meta=not args.no_meta)
    rep = score(rows, preds, tax)
    rep["config"] = {"mode": args.mode, "env": args.env, "vlm": args.vlm,
                     "model_dir": str(args.model_dir) if args.model_dir else None,
                     "golden": str(args.golden)}
    rep["raw"] = preds
    rep["vlm_usage"] = preds[0].get("_vlm_usage", []) if preds else []
    name = args.name or args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (out_dir / f"{name}_confusion.md").write_text(confusion_md(rep, tax) + "\n", encoding="utf-8")
    e = rep["electronics"]
    print(f"[{name}] 대분류 {rep['coarse_hit']}/{rep['n']} ({rep['coarse_acc']:.1%}) · "
          f"electronics 예측 {e['predicted']} / 오탐 {e['false_positive']} "
          f"(오분류 중 {e['share_of_errors']}) / recall {e['recall']} (n={e['support']}) · "
          f"세부 {rep['fine']['hit']}/{rep['fine']['n']} · VLM 중재 발동 {rep['vlm_triggered']} · "
          f"관대 {rep['coarse_hit_lenient']} · reject {rep['reject']} · p50 {rep['wall_ms_p50']}ms")
    print("오분류 쌍:", rep["error_pairs"])


if __name__ == "__main__":
    main()
