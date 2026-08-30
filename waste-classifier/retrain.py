"""Active Learning 재학습 스크립트.

흐름:
  1. Supabase user_uploads 에서 confirmed/corrected 피드백 수집
  2. 이미지를 waste-preprocessor의 raw 폴더로 다운로드
     (naming: user_<upload_id>.<ext>, 라벨별 폴더에 저장)
  3. 이전 모델 백업
  4. waste-preprocessor 재실행 (전처리·벡터화·manifest 갱신)
  5. waste-classifier 재학습·평가·ONNX export
  6. 새 ONNX 를 Supabase Storage 에 업로드 + model_versions row 등록
  7. 새 vs 기존 정확도 비교 및 보고

사용:
    cd waste-classifier
    .venv/bin/python retrain.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from supabase import create_client


PROJECT_ROOT: Path = Path(__file__).resolve().parent              # waste-classifier
PREPROCESSOR_ROOT: Path = PROJECT_ROOT.parent / "waste-preprocessor"
RAW_DIR: Path = PREPROCESSOR_ROOT / "data" / "raw" / "garbage-classification"

EVAL_PATH: Path = PROJECT_ROOT / "outputs" / "logs" / "cnn" / "evaluation.json"
CKPT_PATH: Path = PROJECT_ROOT / "outputs" / "checkpoints" / "cnn" / "best.pt"
ONNX_PATH: Path = PROJECT_ROOT / "outputs" / "models" / "cnn" / "classifier.onnx"
EDGE_ONNX_PATH: Path = PROJECT_ROOT / "outputs" / "models" / "cnn_edge" / "classifier.onnx"
SPLITS_PATH: Path = PROJECT_ROOT / "data" / "splits" / "splits.json"
BACKUP_DIR: Path = PROJECT_ROOT / "outputs" / "backups"

MODELS_BUCKET = "models"

# 기본 fallback — Supabase 조회 실패 시 사용 (정상 운영 시엔 항상 동적 조회).
_FALLBACK_LABELS = ("cardboard", "glass", "metal", "paper", "plastic", "trash")

# 70/15/15 stratified split 이 통과하려면 각 클래스가 holdout(30%)에서 ≥2개,
# 즉 전체 ≥7개 필요. 마진 두고 6으로 설정 — 그 이하면 quarantine.
MIN_SAMPLES_PER_CLASS = 6
QUARANTINE_DIR: Path = PREPROCESSOR_ROOT / "data" / "raw" / "quarantine_too_few_samples"


def _load_supabase_env() -> tuple[str, str]:
    # waste-preprocessor 의 .env 가 정본
    load_dotenv(PREPROCESSOR_ROOT / ".env")
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        sys.exit("ERROR: SUPABASE_URL / SUPABASE_KEY 가 설정되지 않음")
    return url, key


def fetch_feedback_rows():
    url, key = _load_supabase_env()
    client = create_client(url, key)
    res = (
        client.table("user_uploads")
        .select("*")
        .in_("feedback_status", ["confirmed", "corrected"])
        .execute()
    )
    return res.data or []


def fetch_active_labels() -> set[str]:
    """학습 대상 라벨 집합 — active=true 클래스 + 숨김 pseudo-class(etc_auto_*).

    pseudo-class 는 active=false(사용자에겐 숨김)지만 모델은 학습해야 하므로
    download 게이트에서 예외적으로 포함한다 (etc_queue.py 참고).
    """
    try:
        url, key = _load_supabase_env()
        client = create_client(url, key)
        res = client.table("waste_classes").select("slug,active").execute()
        slugs = {
            row["slug"] for row in (res.data or [])
            if row.get("active") or str(row["slug"]).startswith("etc_auto_")
        }
        if slugs:
            return slugs
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] waste_classes 조회 실패 → fallback: {exc}")
    return set(_FALLBACK_LABELS)


def download_to_raw(rows: list[dict], valid_labels: set[str]) -> tuple[int, int, int]:
    """피드백된 이미지를 라벨별 raw 폴더로 다운로드.
    Returns: (downloaded, skipped_existing, failed)
    """
    downloaded = 0
    skipped = 0
    failed = 0

    # 비공개 버킷 다운로드용 클라이언트 (service key)
    from supabase import create_client as _cc  # noqa: PLC0415
    _url, _key = _load_supabase_env()
    client = _cc(_url, _key)

    for row in rows:
        label = row["feedback_label"]
        if label not in valid_labels:
            print(f"  [skip] invalid label {label!r} for {row['id']}")
            failed += 1
            continue

        upload_id = row["id"]
        storage_path = row["storage_path"]
        ext = Path(storage_path).suffix or ".jpg"
        dest_dir = RAW_DIR / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"user_{upload_id}{ext}"

        if dest_file.exists():
            skipped += 1
            continue

        try:
            # 비공개 버킷 대응: service key 로 storage API 다운로드 우선,
            # 실패 시 image_url (레거시 공개 URL 행) fallback
            try:
                data = client.storage.from_("user-uploads").download(storage_path)
            except Exception:  # noqa: BLE001
                resp = requests.get(row["image_url"], timeout=30)
                resp.raise_for_status()
                data = resp.content
            dest_file.write_bytes(data)
            downloaded += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [fail] {upload_id}: {exc}")
            failed += 1

    return downloaded, skipped, failed


def quarantine_tiny_classes() -> dict[str, int]:
    """샘플 수 < MIN_SAMPLES_PER_CLASS 인 클래스 폴더를 quarantine 으로 이동.

    sklearn stratified_split 이 클래스당 ≥2 in holdout 을 요구하므로,
    너무 적은 클래스 (예: 사용자 피드백 첫 3장) 가 있으면 split 이 실패함.

    Returns: 격리된 클래스 → 이미지 수.
    """
    quarantined: dict[str, int] = {}
    if not RAW_DIR.exists():
        return quarantined

    for class_dir in sorted(RAW_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_dir.name.startswith("."):
            continue
        images = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        ]
        n = len(images)
        if n < MIN_SAMPLES_PER_CLASS:
            QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
            dest = QUARANTINE_DIR / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            # 이미지 이동 — 격리본에 같은 이름이 있으면 덮어쓰기 (중복 재다운로드 대비)
            for img in images:
                target = dest / img.name
                if target.exists():
                    target.unlink()
                shutil.move(str(img), str(target))
            # raw 의 클래스 디렉토리는 완전히 제거 (manifest 에서 빠지도록)
            shutil.rmtree(class_dir, ignore_errors=True)
            quarantined[class_dir.name] = n
    return quarantined


def backup_artifacts() -> Path | None:
    """현재 모델·평가 결과를 timestamp 폴더로 백업.
    Returns: 백업 폴더 경로 또는 None (백업할 게 없으면).
    """
    if not (CKPT_PATH.exists() or ONNX_PATH.exists()):
        return None

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = BACKUP_DIR / f"cnn_{ts}"
    target.mkdir()

    for src in (CKPT_PATH, ONNX_PATH, EVAL_PATH):
        if src.exists():
            shutil.copy2(src, target / src.name)
    return target


def rollback_artifacts(backup_path: Path | None) -> bool:
    """게이트 FAIL 시 백업에서 모델·평가 결과 복원 → 기존 active 모델 유지.
    Returns True if 복원됨.
    """
    if backup_path is None or not backup_path.exists():
        print("  [rollback] 백업이 없어 복원 불가 (첫 학습이었을 수 있음)")
        return False
    restored: list[str] = []
    for dst in (CKPT_PATH, ONNX_PATH, EVAL_PATH):
        src = backup_path / dst.name
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored.append(dst.name)
    print(f"  [rollback] 백업에서 복원: {restored}")
    return True


def get_current_accuracy() -> float | None:
    if not EVAL_PATH.exists():
        return None
    return float(json.loads(EVAL_PATH.read_text())["accuracy"])


def run_preprocessor() -> None:
    # --no-vectorize: npz 벡터 생략 (CNN 은 raw 직접 로드). 디스크 대폭 절약.
    subprocess.run(
        [str(PREPROCESSOR_ROOT / ".venv" / "bin" / "python"), "main.py", "--no-vectorize"],
        cwd=str(PREPROCESSOR_ROOT),
        check=True,
    )


def _mark_trained_classes() -> None:
    """현재 manifest 에 존재하는 모든 라벨을 waste_classes.trained_in_model = true 로."""
    manifest_path = PREPROCESSOR_ROOT / "data" / "processed" / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        labels = sorted({item["label"] for item in manifest.get("items", [])})
        url, key = _load_supabase_env()
        client = create_client(url, key)
        for slug in labels:
            client.table("waste_classes").update(
                {"trained_in_model": True},
            ).eq("slug", slug).execute()
        print(f"  waste_classes.trained_in_model = true → {labels}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] trained 갱신 실패: {exc}")


def _read_manifest_labels() -> list[str]:
    manifest_path = PREPROCESSOR_ROOT / "data" / "processed" / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return sorted({item["label"] for item in manifest.get("items", [])})


def publish_model_version(feedback_count: int, version: str | None = None) -> str | None:
    """새 ONNX 를 Supabase Storage 에 업로드 + model_versions 에 새 row 등록.

    - color (필수) + edge (선택) 둘 다 업로드.
    - 기존 active row 의 is_active 를 false 로 내리고 새 row 를 active=true 로.
    - version: 진단과 동일 태그를 쓰도록 외부에서 주입 가능 (없으면 새로 생성).
    - 반환: 생성된 version 문자열 (실패 시 None).
    """
    if not ONNX_PATH.exists():
        print(f"  [warn] {ONNX_PATH} 없음 — 모델 publish skip")
        return None

    url, key = _load_supabase_env()
    client = create_client(url, key)
    bucket = client.storage.from_(MODELS_BUCKET)
    version = version or datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Color ONNX
    color_bytes = ONNX_PATH.read_bytes()
    color_sha = hashlib.sha256(color_bytes).hexdigest()
    color_storage = f"v{version}/classifier.onnx"
    print(f"  ⬆ uploading color ONNX ({len(color_bytes) / 1024 / 1024:.1f} MB)...")
    bucket.upload(
        path=color_storage,
        file=color_bytes,
        file_options={"upsert": "true", "content-type": "application/octet-stream"},
    )
    color_url = bucket.get_public_url(color_storage)

    # 2) Edge ONNX (있을 때만)
    edge_storage: str | None = None
    edge_sha: str | None = None
    edge_url: str | None = None
    if EDGE_ONNX_PATH.exists():
        edge_bytes = EDGE_ONNX_PATH.read_bytes()
        edge_sha = hashlib.sha256(edge_bytes).hexdigest()
        edge_storage = f"v{version}/classifier_edge.onnx"
        print(f"  ⬆ uploading edge ONNX ({len(edge_bytes) / 1024 / 1024:.1f} MB)...")
        bucket.upload(
            path=edge_storage,
            file=edge_bytes,
            file_options={"upsert": "true", "content-type": "application/octet-stream"},
        )
        edge_url = bucket.get_public_url(edge_storage)
    else:
        print("  (edge ONNX 없음 — color 단독)")

    # 3) 메타데이터
    accuracy = get_current_accuracy()
    labels = _read_manifest_labels()

    # 4) 기존 active version 내리고 새 row 삽입
    client.table("model_versions").update({"is_active": False}).eq("is_active", True).execute()
    client.table("model_versions").insert({
        "version": version,
        "color_storage_path": color_storage,
        "edge_storage_path": edge_storage,
        "color_sha256": color_sha,
        "edge_sha256": edge_sha,
        "color_url": color_url,
        "edge_url": edge_url,
        "test_accuracy": accuracy,
        "num_classes": len(labels),
        "class_labels": labels,
        "feedback_count": feedback_count,
        "is_active": True,
    }).execute()

    print(f"  ✓ model_versions row inserted: version={version}, active=true")
    print(f"    color: {color_url}")
    if edge_url:
        print(f"    edge:  {edge_url}")
    return version


def run_classifier_full() -> None:
    # splits.json 을 지워 새 데이터까지 포함된 새 분할 생성
    if SPLITS_PATH.exists():
        SPLITS_PATH.unlink()
        print(f"  removed {SPLITS_PATH.name} → will regenerate with new data")

    subprocess.run(
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "main.py", "all", "--arch", "cnn",
        ],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Active learning retrain")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Supabase에서 피드백 데이터만 조회하고 다운로드·학습은 안 함",
    )
    parser.add_argument(
        "--skip-preprocessor", action="store_true",
        help="이미 raw 폴더에 이미지가 있다면 전처리 단계만 스킵",
    )
    parser.add_argument(
        "--publish-only", action="store_true",
        help="재학습 건너뛰고, 현재 outputs/ 의 ONNX 를 Supabase 에 그대로 publish "
             "(첫 seeding 또는 수동 publish 용)",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="전처리·학습을 건너뛰고 현재 best.pt/ONNX 로 진단→게이트→publish 만 수행 "
             "(학습은 됐는데 이후 단계가 실패한 경우 재학습 없이 복구용)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Active Learning Retrain")
    print("=" * 60)

    # --publish-only: 현재 있는 ONNX 만 Supabase 에 올림
    if args.publish_only:
        print("\n[publish-only] 재학습 없이 현재 ONNX 만 publish 합니다.")
        v = publish_model_version(feedback_count=0)
        if v is None:
            return 1
        print(f"\n✓ v{v} published. waste-api · Flutter 앱이 다음 부팅 시 자동 갱신됨.")
        return 0

    # 0) etc 큐 자동 처리 (open-set) — 임계 도달 시 기존 클래스 재배정 + 신규 후보 군집.
    #    feedback_label 을 갱신하므로 반드시 피드백 수집 전에 실행.
    print("\n[0] etc 큐 점검 (open-set 2단계)...")
    try:
        from etc_queue import maybe_process
        maybe_process()
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] etc 큐 처리 실패 (계속 진행): {exc}")

    # 1) Supabase에서 피드백 수집 + 유효 클래스 목록 (waste_classes active=true)
    print("\n[1/6] Supabase user_uploads 조회 + waste_classes 동적 라벨 셋...")
    valid_labels = fetch_active_labels()
    print(f"  active 클래스 ({len(valid_labels)}개): {sorted(valid_labels)}")
    rows = fetch_feedback_rows()
    confirmed_count = sum(1 for r in rows if r["feedback_status"] == "confirmed")
    corrected_count = sum(1 for r in rows if r["feedback_status"] == "corrected")
    print(f"  총 {len(rows)}건 (confirmed={confirmed_count}, corrected={corrected_count})")

    if not rows:
        print("\n재학습할 새 데이터가 없습니다. 종료.")
        return 0

    # 라벨별 분포 출력 — active 클래스 기준 (정렬)
    from collections import Counter
    label_dist = Counter(r["feedback_label"] for r in rows)
    print("  라벨별 분포:")
    for label in sorted(valid_labels):
        print(f"    {label:14s}: {label_dist.get(label, 0)}")
    # 알 수 없는 라벨이 있으면 같이 표시 (디버깅용)
    unknown = set(label_dist.keys()) - valid_labels
    if unknown:
        print(f"  [warn] active set 에 없는 라벨: {sorted(unknown)} — download 단계에서 reject 됨")

    if args.dry_run:
        print("\n[dry-run] 다운로드·학습 안 함. 종료.")
        return 0

    # 2) 이미지 다운로드
    print("\n[2/6] 이미지 다운로드...")
    downloaded, skipped, failed = download_to_raw(rows, valid_labels)
    print(f"  downloaded={downloaded} skipped(already)={skipped} failed={failed}")

    # 2.5) 너무 적은 클래스는 격리 (stratified split 통과 보장)
    quarantined = quarantine_tiny_classes()
    if quarantined:
        print(f"\n[격리] 샘플 < {MIN_SAMPLES_PER_CLASS}개인 클래스 → {QUARANTINE_DIR.name}/")
        for label, n in quarantined.items():
            print(f"  - {label}: {n}장 (다음 retrain 까지 학습 제외, 원복은 폴더 복원만 하면 됨)")

    # 3) 백업 + 기존 정확도 기록
    print("\n[3/6] 기존 모델 백업 + 정확도 기록...")
    old_acc = get_current_accuracy()
    backup_path = backup_artifacts()
    if old_acc is not None:
        print(f"  이전 test accuracy: {old_acc:.4f}")
    else:
        print("  이전 평가 결과 없음 (첫 학습)")
    if backup_path:
        print(f"  백업 위치: {backup_path}")

    # 4) waste-preprocessor 재실행
    if args.skip_train:
        print("\n[4-5] --skip-train: 전처리·학습 건너뜀 (현재 best.pt/ONNX 사용)")
    elif not args.skip_preprocessor:
        print("\n[4/6] waste-preprocessor 실행 (2-3분 소요)...")
        run_preprocessor()
    else:
        print("\n[4/6] preprocessor 스킵 (--skip-preprocessor)")

    # 5) waste-classifier 재학습 + 평가 + ONNX export
    if not args.skip_train:
        print("\n[5/7] waste-classifier 재학습 (10분 이상 소요)...")
        run_classifier_full()

    # 6) 진단 + 게이트 — 고정 held-out 으로 회귀 검사. 통과해야만 publish/activate.
    #    publish 와 동일 태그를 쓰도록 version 을 한 번만 생성해 공유.
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\n[6/7] 진단 + 게이트 (고정 held-out)...")
    from diagnose import run_diagnosis
    report = run_diagnosis(arch="cnn", version=version, commit_history=True)
    if not report["gate"]["pass"]:
        print("\n" + "=" * 60)
        print("❌ 게이트 FAIL — publish 취소, 백업에서 자동 롤백")
        print("=" * 60)
        for reason in report["gate"]["reasons"]:
            print(f"  - {reason}")
        rollback_artifacts(backup_path)
        print("\n  새 모델은 폐기됐고 기존 active 모델이 그대로 유지됩니다.")
        print(f"  실패 진단 상세: outputs/logs/diagnosis/{version}.json")
        return 1

    # 6.1) 게이트 통과 — trained_in_model 갱신 (manifest 의 모든 라벨)
    _mark_trained_classes()

    # 7) Supabase Storage 에 ONNX 업로드 + model_versions row 등록.
    #    waste-api · Flutter 앱이 부팅 시 이 row 를 조회해서 자동 갱신.
    print("\n[7/7] 새 ONNX → Supabase models/ 버킷 publish...")
    published_version = publish_model_version(feedback_count=len(rows), version=version)

    # 결과 비교
    new_acc = get_current_accuracy()
    print()
    print("=" * 60)
    print("재학습 결과")
    print("=" * 60)
    print(f"  이전 정확도: {old_acc:.4f}" if old_acc else "  이전 정확도: N/A")
    print(f"  새 정확도  : {new_acc:.4f}")

    if old_acc is not None and new_acc is not None:
        delta = new_acc - old_acc
        # 여기 도달했다면 게이트는 이미 통과 — 하락이 있어도 임계(-2pp) 이내.
        verdict = "IMPROVED" if delta >= 0 else "DEGRADED (게이트 임계 내 — 허용됨)"
        print(f"  변화      : {delta:+.4f} ({delta*100:+.2f}pp) {verdict}")

    print()
    print(f"  새 ONNX (로컬): {ONNX_PATH}")
    if published_version:
        print(f"  publish 완료 — waste-api · Flutter 앱은 다음 부팅 시 자동으로 v{published_version} 로 갱신됨.")
        print(f"  즉시 반영하려면:")
        print(f"    curl -X POST https://<waste-api>/admin/reload-model")
    else:
        print(f"  ⚠️  publish 실패 — waste-api / Flutter 앱은 옛 모델 그대로 사용.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
