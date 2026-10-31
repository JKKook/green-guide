"""골든셋 추출 — user_uploads 최근 N장 + 모델 판정을 리뷰용 JSON 으로 묶는다.

근거: docs/plans/PRODUCTION_READINESS.md 결정 4 (1차 = 사진 전체 재질, 2차 = 물건별·영역별 후보).

수집 항목 (행마다):
  · upload_id·업로드 시각·원본 파일(로컬 저장 경로)·촬영 메타
  · hier  = 업로드 당시 /predict-hier 판정 (DB 기록값: 표시 클래스·확신·대분류 확률·model_arch)
            단 2026-08-29 복원분(model_arch=restored_from_local_cache)은 예측이 아니라 정답 라벨이 들어 있어
            hier_source="restored_label" 로 표시하고 판정으로 쓰지 않는다.
  · objects = 배포 서버 /predict-objects 재판정 (물건별 후보)
  · regions = 배포 서버 /predict-with-regions 재판정 (영역 라벨, 오버레이는 파일로 저장)
  · feedback = 사용자 피드백 상태·라벨 (있을 때)

두 재판정 엔드포인트는 user_uploads 에 기록하지 않는다(/predict-hier 만 기록) → DB 를 바꾸지 않는다.
Storage 다운로드는 storage_path + service_role 로 해서 버킷 비공개(014) 뒤에도 동작한다.

홀드아웃: sha256(upload_id) 첫 바이트 % 10 == 0 인 행은 제외하고 건수만 보고한다.

사용:
  arch -x86_64 .venv/bin/python scripts/build_golden_set.py --out ~/Downloads/greenguide-golden --n 100
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from supabase import create_client

API_BASE = "https://ethandev92-waste-api.hf.space"
BUCKET = "user-uploads"
META_COLS = ("capture_mode", "orientation", "exif_orientation", "tta_rotation",
             "crop_applied", "crop_box", "tap_x", "tap_y",
             "quality_blur", "quality_brightness", "ai_training_opt_in")


def is_holdout(upload_id: str) -> bool:
    return hashlib.sha256(upload_id.encode()).digest()[0] % 10 == 0


def post_image(base: str, endpoint: str, name: str, data: bytes, mime: str) -> dict:
    for attempt in range(3):
        try:
            r = requests.post(f"{base}{endpoint}", files={"image": (name, data, mime)}, timeout=240)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            if attempt == 2:
                return {"error": str(exc)[:200]}
            time.sleep(5 * (attempt + 1))
    return {"error": "unreachable"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--api", default=API_BASE)
    args = ap.parse_args()

    load_dotenv(Path(__file__).resolve().parents[1] / ".env") or load_dotenv(Path.cwd() / ".env")
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    rows = (sb.table("user_uploads").select("*")
            .order("uploaded_at", desc=True).execute().data)
    holdout = [r for r in rows if is_holdout(r["id"])]
    picked = [r for r in rows if not is_holdout(r["id"])][: args.n]

    out = args.out.expanduser()
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    taxonomy = requests.get(f"{args.api}/taxonomy", timeout=60).json()
    f2c = taxonomy.get("fine_to_coarse", {})

    items = []
    for i, r in enumerate(picked, 1):
        sp = r["storage_path"]
        ext = sp.rsplit(".", 1)[-1].lower()
        mime = "image/webp" if ext == "webp" else "image/jpeg"
        data = sb.storage.from_(BUCKET).download(sp)
        img_path = out / "images" / f"{r['id']}.{ext}"
        img_path.write_bytes(data)

        restored = (r.get("model_arch") or "") == "restored_from_local_cache"
        pc = r["predicted_class"]
        hier = None if restored else {
            "display_class": pc,
            "coarse_class": f2c.get(pc, pc),
            "confidence": r.get("predicted_confidence"),
            "coarse_probabilities": r.get("all_probabilities"),
            "model_arch": r.get("model_arch"),
            "inference_ms": r.get("inference_ms"),
        }

        objects = post_image(args.api, "/predict-objects", img_path.name, data, mime)
        regions = post_image(args.api, "/predict-with-regions", img_path.name, data, mime)
        overlay = regions.pop("overlay_base64", None) if isinstance(regions, dict) else None
        if overlay and "," in overlay:
            (out / "overlays" / f"{r['id']}.jpg").write_bytes(base64.b64decode(overlay.split(",", 1)[1]))
        if isinstance(regions, dict):
            regions.pop("all_probabilities", None)  # 27클래스 전체 확률 — 리뷰에 불필요, 크기만 키움

        fb_status = r.get("feedback_status")
        items.append({
            "upload_id": r["id"],
            "uploaded_at": r["uploaded_at"],
            "file": f"images/{img_path.name}",
            "overlay": f"overlays/{r['id']}.jpg" if overlay else None,
            "sha256": hashlib.sha256(data).hexdigest(),
            "hier_source": "restored_label" if restored else "upload_time_prediction",
            "hier": hier,
            "objects": objects,
            "regions": regions,
            "feedback": None if fb_status == "pending" else {
                "status": fb_status, "label": r.get("feedback_label"), "at": r.get("feedback_at"),
            },
            "meta": {k: r.get(k) for k in META_COLS},
            "review": {"photo_coarse": None, "photo_fine": None, "objects_ok": None, "note": None},
        })
        print(f"[{i}/{len(picked)}] {r['id']} hier={pc if not restored else '-'} "
              f"objects={objects.get('count', 'err')} regions={regions.get('predicted_class', 'err')}",
              flush=True)

    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_base": args.api,
        "requested_n": args.n,
        "total_uploads": len(rows),
        "holdout_excluded": len(holdout),
        "holdout_rule": "sha256(upload_id)[0] % 10 == 0",
        "count": len(items),
        "restored_rows": sum(1 for x in items if x["hier_source"] == "restored_label"),
        "items": items,
    }
    (out / "golden.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(f"total={len(rows)} holdout_excluded={len(holdout)} written={len(items)} → {out / 'golden.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
