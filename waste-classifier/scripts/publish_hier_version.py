#!/usr/bin/env python3
"""계층 모델 버전 게시 — Supabase Storage 업로드 + model_versions 활성화.

retrain.py 의 publish_model_version 계층판. migration 009 의
coarse_labels / fine_to_coarse / taxonomy_hash 컬럼 사용.

⚠️ 운영 영향: is_active 를 뒤집으면 클라이언트가 새 모델을 받기 시작한다.
   --apply 없이는 dry-run (업로드·DB 변경 없음).

사용:
    .venv/bin/python scripts/publish_hier_version.py            # dry-run
    .venv/bin/python scripts/publish_hier_version.py --apply
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402

MODELS_DIR = config.MODELS_DIR / "cnn_hier"
ONNX = MODELS_DIR / "classifier.onnx"
SIDECAR = MODELS_DIR / "taxonomy.json"
OOD = MODELS_DIR / "ood.npz"
EVAL = config.LOGS_DIR / "cnn_hier" / "evaluation.json"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


HF_MODEL_REPO = "ethanDev92/waste-models"


def _hf_token() -> str:
    """HF 토큰 — git credential helper(키체인) 재사용. 파일에 저장하지 않음."""
    import subprocess  # noqa: PLC0415
    out = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=huggingface.co\n\n",
        capture_output=True, text=True, timeout=30).stdout
    tok = dict(l.split("=", 1) for l in out.strip().splitlines()
               if "=" in l).get("password")
    if not tok:
        sys.exit("HF 토큰 없음 — git credential(키체인) 확인")
    return tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="실제 업로드+활성화 (없으면 dry-run)")
    ap.add_argument("--storage", choices=("hf", "supabase"), default="hf",
                    help="모델 파일 호스팅: hf=HF Hub(기본, 대용량 무제한) / "
                         "supabase=스토리지(파일당 한도 주의 — 무료플랜 50MB 캡, "
                         "한도 해제 시 재시도용으로 유지)")
    args = ap.parse_args()

    for p in (ONNX, SIDECAR, OOD, EVAL):
        if not p.exists():
            sys.exit(f"필수 아티팩트 없음: {p}")

    tax = json.loads(SIDECAR.read_text(encoding="utf-8"))
    ev = json.loads(EVAL.read_text(encoding="utf-8"))
    version = time.strftime("%Y%m%d_%H%M%S")
    tax_hash = hashlib.sha256(
        json.dumps(tax["fine_to_coarse"], sort_keys=True).encode()).hexdigest()[:16]

    print(f"버전 v{version} (hier)")
    print(f"  onnx {ONNX.stat().st_size/1e6:.1f}MB sha={_sha256(ONNX)[:12]}…")
    print(f"  coarse={len(tax['coarse_labels'])} fine={len(tax['fine_labels'])} "
          f"taxonomy_hash={tax_hash}")
    print(f"  eval: coarse_acc={ev['coarse_accuracy']} fine_acc={ev['fine_accuracy_on_fine_items']}")

    if not args.apply:
        print("\n[dry-run] --apply 로 실제 게시. (운영 클라이언트에 즉시 영향)")
        return

    from supabase import create_client  # noqa: PLC0415

    from retrain import _load_supabase_env  # noqa: PLC0415
    url, key = _load_supabase_env()
    sb = create_client(url, key)

    base = f"v{version}"
    if args.storage == "hf":
        # ── HF Hub 호스팅 (기본) — Supabase 무료플랜 파일당 50MB 캡 우회.
        # 레지스트리(model_versions)는 그대로 Supabase — URL 만 HF 를 가리킴.
        from huggingface_hub import HfApi  # noqa: PLC0415
        api = HfApi(token=_hf_token())

        # ── HF 용량 가드 (사용자 상시 지시: 무료 보전 초과 금지) ────────────
        # Supabase 와 동일 원칙: 자체 상한 1GB 의 80% 초과 위험 시 발행 중단.
        # (정리 로직이 활성 버전만 남기므로 평시 ~95MB — 정리 실패 누적 방지용)
        _CAP = 1_000_000_000
        try:
            cur = sum(
                (getattr(f, "lfs", None).size if getattr(f, "lfs", None) else 0)
                or getattr(f, "size", 0) or 0
                for f in api.list_repo_tree(HF_MODEL_REPO, repo_type="model",
                                            recursive=True))
        except Exception as exc:  # noqa: BLE001
            print(f"[quota] HF 용량 조회 실패(보수적 진행): {str(exc)[:80]}")
            cur = 0
        incoming = (ONNX.stat().st_size + SIDECAR.stat().st_size
                    + OOD.stat().st_size)
        proj = cur + incoming
        print(f"[quota] HF {HF_MODEL_REPO}: 현재 {cur/1e6:.1f}MB "
              f"(+예정 {incoming/1e6:.1f}MB → {proj/1e6:.1f}MB, "
              f"자체 상한 {_CAP/1e9:.0f}GB 의 {proj/_CAP*100:.0f}%)")
        if proj > _CAP * 0.8:
            raise SystemExit(
                "발행 중단 — HF 저장소 자체 상한(1GB)의 80% 초과 위험. "
                "구버전 정리 실패 여부를 확인하고 사용자에게 보고하세요.")

        color_path = f"hf:{base}/classifier_hier.onnx"
        for local, name in ((ONNX, "classifier_hier.onnx"),
                            (SIDECAR, "taxonomy.json"), (OOD, "ood.npz")):
            api.upload_file(
                path_or_fileobj=str(local), path_in_repo=f"{base}/{name}",
                repo_id=HF_MODEL_REPO, repo_type="model",
                commit_message=f"publish {base}/{name}")
            print(f"  [hf] 업로드: {base}/{name}")
        color_url = (f"https://huggingface.co/{HF_MODEL_REPO}"
                     f"/resolve/main/{base}/classifier_hier.onnx")
    else:
        # ── Supabase 스토리지 호스팅 (한도 해제 후 재시도용 경로 — 유지) ──
        # 스토리지 쿼터 가드 (사용자 상시 지시: 1GB 초과 위험 사전 경고)
        # 2026-07-21 구버전 누적 981MB → 쿼터 초과 → 프로젝트 이사 사태 재발 방지
        sys.path.insert(0, "/Users/ethan/practice/waste/waste-api")
        from scripts.storage_usage import check_storage  # noqa: PLC0415
        incoming = ONNX.stat().st_size + SIDECAR.stat().st_size + OOD.stat().st_size
        print("[quota] 발행 전 스토리지 점검:")
        if not check_storage(sb, incoming_bytes=incoming):
            raise SystemExit(
                "발행 중단 — 스토리지 80% 초과 위험. 구버전 정리 후 재시도하거나 "
                "사용자에게 보고하세요.")

        color_path = f"{base}/classifier_hier.onnx"
        sb.storage.from_("models").upload(
            color_path, ONNX.read_bytes(),
            file_options={"content-type": "application/octet-stream", "upsert": "true"})
        sb.storage.from_("models").upload(
            f"{base}/taxonomy.json", SIDECAR.read_bytes(),
            file_options={"content-type": "application/json", "upsert": "true"})
        sb.storage.from_("models").upload(
            f"{base}/ood.npz", OOD.read_bytes(),
            file_options={"content-type": "application/octet-stream", "upsert": "true"})
        color_url = sb.storage.from_("models").get_public_url(color_path)

    sb.table("model_versions").update({"is_active": False}).eq("is_active", True).execute()
    sb.table("model_versions").insert({
        "version": version,
        "color_storage_path": color_path,
        "color_sha256": _sha256(ONNX),
        "color_url": color_url,
        "test_accuracy": ev["fine_accuracy_on_fine_items"],
        "num_classes": len(tax["fine_labels"]),
        "class_labels": tax["fine_labels"],
        "coarse_labels": tax["coarse_labels"],
        "fine_to_coarse": tax["fine_to_coarse"],
        "taxonomy_hash": tax_hash,
        "is_active": True,
        "notes": f"cnn_hier — coarse_acc={ev['coarse_accuracy']}",
    }).execute()
    print(f"게시 완료: v{version} (is_active=true)")

    # ── 구버전 파일 자동 삭제 — 활성 버전 파일만 유지 (쿼터 사태 근본 방지) ──
    # 경로 규약: Supabase 행="vXXX/..." / HF 행="hf:vXXX/..." — 각자 위치에서 정리
    rows = (sb.table("model_versions").select("version,color_storage_path,is_active")
            .execute().data or [])
    keep_prefix = f"v{version}/"
    removed = 0
    hf_api = None
    for r in rows:
        if r.get("is_active"):
            continue
        sp = r.get("color_storage_path") or ""
        is_hf = sp.startswith("hf:")
        old_base = sp.removeprefix("hf:").split("/")[0]
        if not old_base or old_base + "/" == keep_prefix:
            continue
        try:
            if is_hf:
                from huggingface_hub import HfApi  # noqa: PLC0415
                hf_api = hf_api or HfApi(token=_hf_token())
                for name in ("classifier_hier.onnx", "taxonomy.json", "ood.npz"):
                    hf_api.delete_file(f"{old_base}/{name}", repo_id=HF_MODEL_REPO,
                                       repo_type="model",
                                       commit_message=f"cleanup {old_base}")
                    removed += 1
            else:
                files = sb.storage.from_("models").list(old_base) or []
                paths = [f"{old_base}/{f['name']}" for f in files]
                if paths:
                    sb.storage.from_("models").remove(paths)
                    removed += len(paths)
        except Exception as exc:  # noqa: BLE001
            print(f"[cleanup] {old_base} 삭제 실패(무시): {str(exc)[:80]}")
    print(f"[cleanup] 구버전 파일 {removed}개 삭제 — 활성 버전만 유지")


if __name__ == "__main__":
    main()
