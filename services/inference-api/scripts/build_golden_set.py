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


HIER_KEYS = ("display_level", "display_class", "coarse_class", "coarse_confidence",
             "fine_class", "fine_confidence", "fine_margin", "fine_top5",
             "coarse_probabilities", "ood_distance", "ood_reject", "model_arch", "inference_ms")


def fill_hier(golden: Path, local_api: str, out_file: Path) -> int:
    """기존 golden.json 의 모든 행에 로컬 서버 /predict-hier 판정을 채운다 (v2).

    local_api 는 GREENGUIDE_API_COLLECT_UPLOADS=false 로 띄운 로컬 서버여야 한다
    (프로덕션 /predict-hier 는 업로드를 기록해 user_uploads 를 오염시킨다).
    · 모든 행: hier_local = 로컬 재판정 전체 필드
    · 복원분(hier 없음): hier = hier_local 요약, hier_source = "local_reinference"
    · 실판정 행: hier(업로드 당시)는 그대로 두고 hier_source 유지
    """
    health = requests.get(f"{local_api}/health", timeout=30)
    health.raise_for_status()
    doc = json.loads(golden.read_text())
    root = golden.parent
    for i, it in enumerate(doc["items"], 1):
        p = root / it["file"]
        mime = "image/webp" if p.suffix == ".webp" else "image/jpeg"
        res = post_image(local_api, "/predict-hier", p.name, p.read_bytes(), mime)
        if "error" in res:
            print(f"[{i}] {it['upload_id']} ERROR {res['error']}", flush=True)
            it["hier_local"] = res
            continue
        if res.get("upload_id"):
            raise SystemExit("로컬 서버가 업로드를 기록함 — GREENGUIDE_API_COLLECT_UPLOADS=false 로 다시 띄울 것")
        hl = {k: res.get(k) for k in HIER_KEYS}
        hl["confidence"] = res["fine_confidence"] if res["display_level"] == "fine" else res["coarse_confidence"]
        it["hier_local"] = hl
        if it.get("hier") is None:
            it["hier"] = {k: hl[k] for k in ("display_class", "display_level", "coarse_class", "fine_class",
                                             "confidence", "coarse_confidence", "fine_confidence",
                                             "coarse_probabilities", "model_arch", "inference_ms")}
            it["hier_source"] = "local_reinference"
        print(f"[{i}/{len(doc['items'])}] {it['upload_id']} {hl['display_level']} {hl['display_class']} "
              f"({hl['confidence']:.2f})", flush=True)
    doc["version"] = 2
    doc["hier_local_api"] = local_api
    doc["hier_filled"] = sum(1 for x in doc["items"] if x.get("hier") and "error" not in (x.get("hier_local") or {}))
    out_file.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(f"hier 채움 {doc['hier_filled']}/{len(doc['items'])} → {out_file}")
    return 0 if doc["hier_filled"] == len(doc["items"]) else 1


def _dhash(data: bytes) -> int:
    """8×8 difference hash (9×8 그레이 축소 후 가로 이웃 비교 64비트). PIL 은 arm64 venv 에서."""
    import io

    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("L").resize((9, 8), Image.LANCZOS)
    px = list(im.getdata())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | (px[y * 9 + x] > px[y * 9 + x + 1])
    return bits


def build_v3(out: Path, local_api: str, local_dir: Path, cutoff: str, exclude_ids: set[str],
             max_hamming: int = 6) -> int:
    """골든셋 v3 — 피드백 없는(pending) 사진만, 서버+로컬 합집합, 중복 제거, 로컬 서버로 3종 판정.

    · 서버: user_uploads 중 feedback_status='pending', uploaded_at < cutoff 이거나 cutoff 이후라도
            exclude_ids 에 없는 행(테스트 노이즈는 id 로 뺀다). 홀드아웃은 빼지 않고 holdout 표시.
    · 로컬: local_dir/meta.jsonl 의 pending 행 + 실제 파일(<id>.jpg|.webp, meta 의 ext 무시).
    · 중복: sha256 일치 또는 dHash 해밍 ≤ max_hamming 을 union-find 로 묶고 묶음당 1장
            (서버 우선 → 오래된 것).
    · 판정: local_api(COLLECT_UPLOADS=false, ANTHROPIC_API_KEY 비움) /predict-hier·/predict-objects·
            /predict-with-regions. 프로덕션 호출 없음.
    """
    requests.get(f"{local_api}/health", timeout=30).raise_for_status()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env") or load_dotenv(Path.cwd() / ".env")
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    rows = sb.table("user_uploads").select("*").order("uploaded_at").execute().data
    server = [r for r in rows if r.get("feedback_status") == "pending" and r["id"] not in exclude_ids]
    excluded_test = [r["id"] for r in rows if r["id"] in exclude_ids]
    after_cutoff = [r["id"] for r in server if r["uploaded_at"] >= cutoff]

    cands = []
    for r in server:
        sp = r["storage_path"]
        ext = sp.rsplit(".", 1)[-1].lower()
        cands.append({"src": "server", "row": r, "id": r["id"], "at": r["uploaded_at"], "ext": ext,
                      "data": sb.storage.from_(BUCKET).download(sp)})
    n_server = len(cands)
    for line in (local_dir / "meta.jsonl").read_text().splitlines():
        m = json.loads(line)
        if m.get("feedback_status") != "pending":
            continue
        f = next(p for p in (local_dir / f"{m['id']}{e}" for e in (".jpg", ".webp", ".jpeg", ".png")) if p.exists())
        cands.append({"src": "local", "row": m, "id": m["id"], "at": m["uploaded_at"],
                      "ext": f.suffix.lstrip(".").lower(), "data": f.read_bytes()})
    n_local = len(cands) - n_server

    for c in cands:
        c["sha256"] = hashlib.sha256(c["data"]).hexdigest()
        c["dhash"] = _dhash(c["data"])

    parent = list(range(len(cands)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            if a["sha256"] == b["sha256"] or bin(a["dhash"] ^ b["dhash"]).count("1") <= max_hamming:
                parent[find(i)] = find(j)
    groups: dict[int, list[dict]] = {}
    for i, c in enumerate(cands):
        groups.setdefault(find(i), []).append(c)

    keep, dup_groups = [], []
    for members in groups.values():
        members.sort(key=lambda c: (c["src"] != "server", c["at"]))
        k = members[0]
        if len(members) > 1:
            gid = f"g{len(dup_groups) + 1:03d}"
            k["dup_group"] = gid
            dup_groups.append({
                "id": gid, "keep": k["id"],
                "removed": [{"id": c["id"], "source": c["src"],
                             "match": "sha256" if c["sha256"] == k["sha256"] else
                             f"dhash:{bin(c['dhash'] ^ k['dhash']).count('1')}"} for c in members[1:]],
            })
        keep.append(k)
    keep.sort(key=lambda c: c["at"])

    (out / "images_v3").mkdir(parents=True, exist_ok=True)
    (out / "overlays_v3").mkdir(parents=True, exist_ok=True)
    items = []
    for i, c in enumerate(keep, 1):
        r, data = c["row"], c["data"]
        mime = "image/webp" if c["ext"] == "webp" else "image/jpeg"
        img_path = out / "images_v3" / f"{c['id']}.{c['ext']}"
        img_path.write_bytes(data)

        res = post_image(local_api, "/predict-hier", img_path.name, data, mime)
        if res.get("upload_id"):
            raise SystemExit("로컬 서버가 업로드를 기록함 — GREENGUIDE_API_COLLECT_UPLOADS=false 로 다시 띄울 것")
        hl = res if "error" in res else {k: res.get(k) for k in HIER_KEYS}
        if "error" not in hl:
            hl["confidence"] = res["fine_confidence"] if res["display_level"] == "fine" else res["coarse_confidence"]
        objects = post_image(local_api, "/predict-objects", img_path.name, data, mime)
        regions = post_image(local_api, "/predict-with-regions", img_path.name, data, mime)
        overlay = regions.pop("overlay_base64", None)
        if overlay and "," in overlay:
            (out / "overlays_v3" / f"{c['id']}.jpg").write_bytes(base64.b64decode(overlay.split(",", 1)[1]))
        regions.pop("all_probabilities", None)

        restored = (r.get("model_arch") or "") == "restored_from_local_cache"
        items.append({
            "upload_id": c["id"],
            "source": c["src"],
            "holdout": is_holdout(c["id"]) if c["src"] == "server" else False,
            "dup_group": c.get("dup_group"),
            "uploaded_at": c["at"],
            "file": f"images_v3/{img_path.name}",
            "overlay": f"overlays_v3/{c['id']}.jpg" if overlay else None,
            "sha256": c["sha256"],
            "dhash": f"{c['dhash']:016x}",
            "hier_source": "restored_label" if restored else (
                "upload_time_prediction" if c["src"] == "server" else "local_cache_prediction"),
            "hier": None if restored else {
                "display_class": r.get("predicted_class"),
                "confidence": r.get("predicted_confidence"),
                "coarse_probabilities": r.get("all_probabilities"),
                "model_arch": r.get("model_arch"),
                "inference_ms": r.get("inference_ms"),
            },
            "hier_local": hl,
            "objects": objects,
            "regions": regions,
            "feedback": None,
            "meta": {k: r.get(k) for k in META_COLS},
            "review": {"photo_coarse": None, "photo_fine": None, "objects_ok": None, "note": None},
        })
        print(f"[{i}/{len(keep)}] {c['src']} {c['id']} hier={hl.get('display_class', 'err')} "
              f"objects={objects.get('count', 'err')} regions={regions.get('predicted_class', 'err')}", flush=True)

    ok = [x for x in items if "error" not in x["hier_local"] and "error" not in x["objects"]
          and "error" not in x["regions"]]
    doc = {
        "version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "local_api": local_api,
        "rule": "feedback_status=pending only; server+local union; sha256/dHash(≤%d) dedup, server first then oldest"
                % max_hamming,
        "server_total_rows": len(rows),
        "server_pending": n_server,
        "server_excluded_test_ids": excluded_test,
        "server_kept_after_cutoff": after_cutoff,
        "local_pending": n_local,
        "dup_removed": len(cands) - len(keep),
        "dup_groups": dup_groups,
        "count": len(items),
        "judged_ok": len(ok),
        "holdout_marked": sum(1 for x in items if x["holdout"]),
        "items": items,
    }
    (out / "golden_v3.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(f"server={n_server} local={n_local} dup_removed={doc['dup_removed']} final={len(items)} "
          f"judged_ok={len(ok)} → {out / 'golden_v3.json'}")
    return 0 if len(ok) == len(items) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--api", default=API_BASE)
    ap.add_argument("--fill-hier", type=Path, metavar="GOLDEN_JSON",
                    help="기존 golden.json 에 로컬 /predict-hier 판정을 채워 --out-file 로 저장")
    ap.add_argument("--local-api", default="http://127.0.0.1:8765")
    ap.add_argument("--out-file", type=Path)
    ap.add_argument("--v3", action="store_true", help="골든셋 v3 (pending 만·서버+로컬·중복 제거·로컬 판정)")
    ap.add_argument("--local-dir", type=Path, default=Path(__file__).resolve().parents[1] / "local_feedback")
    ap.add_argument("--cutoff", default="2026-10-31", help="이 시각 이후 서버 행은 결과에 따로 표시")
    ap.add_argument("--exclude-ids", default="", help="테스트 노이즈 upload_id (쉼표 구분)")
    args = ap.parse_args()
    if args.v3:
        if not args.out:
            ap.error("--out 필요")
        return build_v3(args.out.expanduser(), args.local_api, args.local_dir.expanduser(), args.cutoff,
                        {x for x in args.exclude_ids.split(",") if x})
    if args.fill_hier:
        out_file = args.out_file or args.fill_hier.with_name("golden_v2.json")
        return fill_hier(args.fill_hier.expanduser(), args.local_api, out_file.expanduser())
    if not args.out:
        ap.error("--out 필요")

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
