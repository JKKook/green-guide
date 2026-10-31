"""피드백 혼동쌍 집계 — user_uploads 의 사용자 정정(예측 → 정정 라벨) 상위 K 개.

근거: docs/plans/PRODUCTION_READINESS.md 결정 3 (혼동쌍별 VLM 중재 조건·임계 조정의 입력).

· 대상: feedback_status = 'corrected' 이고 예측 ≠ 정정인 행.
  2026-08-29 복원분(model_arch = restored_from_local_cache)은 predicted_class 에 정답 라벨이 들어 있어 제외한다.
· 세부(표시 클래스) 쌍과 대분류 쌍을 함께 낸다. 대분류 매핑은 배포 서버 /taxonomy 의 fine_to_coarse.
· 읽기 전용 — DB 를 바꾸지 않는다.

사용:
  arch -x86_64 .venv/bin/python scripts/confusion_pairs.py [--top 5] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import requests
from dotenv import load_dotenv
from supabase import create_client

API_BASE = "https://ethandev92-waste-api.hf.space"
RESTORED = "restored_from_local_cache"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--api", default=API_BASE)
    args = ap.parse_args()

    load_dotenv(Path(__file__).resolve().parents[1] / ".env") or load_dotenv(Path.cwd() / ".env")
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    rows = (sb.table("user_uploads")
            .select("id,uploaded_at,predicted_class,feedback_status,feedback_label,model_arch")
            .execute().data)
    f2c = requests.get(f"{args.api}/taxonomy", timeout=60).json().get("fine_to_coarse", {})

    live = [r for r in rows if (r.get("model_arch") or "") != RESTORED]
    status = Counter(r["feedback_status"] for r in live)
    wrong = [r for r in live
             if r["feedback_status"] == "corrected" and r["feedback_label"]
             and r["feedback_label"] != r["predicted_class"]]

    fine = Counter((r["predicted_class"], r["feedback_label"]) for r in wrong)
    coarse = Counter((f2c.get(r["predicted_class"], r["predicted_class"]),
                      f2c.get(r["feedback_label"], r["feedback_label"])) for r in wrong)
    coarse_wrong = sum(n for (p, t), n in coarse.items() if p != t)

    report = {
        "total_rows": len(rows),
        "restored_excluded": len(rows) - len(live),
        "live_rows": len(live),
        "feedback_status": dict(status),
        "corrections": len(wrong),
        "coarse_level_errors": coarse_wrong,
        "top_fine": [{"predicted": p, "corrected": t, "count": n} for (p, t), n in fine.most_common(args.top)],
        "top_coarse": [{"predicted": p, "corrected": t, "count": n}
                       for (p, t), n in coarse.most_common() if p != t][: args.top],
        "rows": [{"id": r["id"], "uploaded_at": r["uploaded_at"], "predicted": r["predicted_class"],
                  "corrected": r["feedback_label"], "model_arch": r["model_arch"]} for r in wrong],
    }

    print(f"전체 {len(rows)} · 복원분 제외 {report['restored_excluded']} · 실사용 {len(live)} "
          f"· 피드백 {dict(status)} · 정정 {len(wrong)} (대분류 오류 {coarse_wrong})")
    print("세부 혼동쌍 (예측 → 정정, 건수)")
    for x in report["top_fine"]:
        print(f"  {x['predicted']:>14} → {x['corrected']:<14} {x['count']}")
    print("대분류 혼동쌍")
    for x in report["top_coarse"]:
        print(f"  {x['predicted']:>14} → {x['corrected']:<14} {x['count']}")
    if args.json:
        args.json.expanduser().write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
