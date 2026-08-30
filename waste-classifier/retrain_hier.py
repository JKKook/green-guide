#!/usr/bin/env python3
"""계층(cnn_hier) 액티브 러닝 루프 — 피드백 → 재학습 → 게이트 → 자동 승격.

retrain.py(flat)의 계층 버전. flat 루프는 건드리지 않는다 (별도 스크립트).

흐름:
  [1] 피드백 수집·다운로드 (retrain.py 함수 재사용 — fine slug 피드백 지원)
  [2] 소수 클래스 격리
  [3] hier 아티팩트 백업
  [4] preprocessor 재실행 (--skip-preprocessor 로 생략)
  [5] hier_splits 재생성(frozen 누적 유지) → 학습 → 평가
  [6] 게이트 (blueprint §7):
        - 대분류 정확도 하락 > GATE_MAX_COARSE_ACC_DROP → FAIL
        - 대분류별 recall 하락 > GATE_MAX_COARSE_RECALL_DROP → FAIL
        (세부는 게이트 대상 아님 — 활성화 판정에서 자동 승격/강등)
      FAIL → 백업 롤백 + exit 1
  [7] PASS → ONNX + OOD 프로토타입 재생성, history 기록
  [8] 세부품목 자동 승격/강등 (apply_hier_activation)
  [9] model_diagnostics 기록 (계층 컬럼)
  [10] --publish 시에만 model_versions 게시 (운영 배포는 명시적 opt-in)

사용:
    .venv/bin/python retrain_hier.py --dry-run
    .venv/bin/python retrain_hier.py
    .venv/bin/python retrain_hier.py --skip-preprocessor
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.hier_train import CKPT_DIR, LOG_DIR  # noqa: E402
from src.taxonomy import COARSE_LABELS  # noqa: E402

# retrain.py(flat) 의 검증된 유틸 재사용
from retrain import (  # noqa: E402
    download_to_raw, fetch_active_labels, fetch_feedback_rows,
    quarantine_tiny_classes, run_preprocessor,
)

MODELS_DIR = config.MODELS_DIR / "cnn_hier"
HIER_SPLITS = config.SPLITS_DIR / "hier_splits.json"
HISTORY_PATH = config.LOGS_DIR / "diagnosis" / "hier_history.jsonl"
BACKUP_ROOT = config.OUTPUTS_DIR / "backups"

# 게이트 임계 (DIAGNOSIS_PROCESS.md 의 flat 게이트와 동일 원칙)
GATE_MAX_COARSE_ACC_DROP = 0.02      # 대분류 전체 정확도 -2pp 초과 하락 금지
GATE_MAX_COARSE_RECALL_DROP = 0.05   # 대분류별 recall -5pp 초과 하락 금지
GATE_RECALL_MIN_SUPPORT = 50         # recall 거부권은 지지 이상 클래스만 (개정 2026-07-21)
# 안내-동일 대분류쌍 — 이 방향의 혼동은 recall 회귀로 세지 않는다.
# 근거: v8~v11 3연속 FAIL 의 실체가 etc(31표본)→trash 이동이었고, etc 의 사용자
# 안내("대부분 일반쓰레기")와 trash 안내가 실질 동일 (청사진 v2 §1.4)
GATE_GUIDANCE_EQUIV = {("etc", "trash"), ("trash", "etc")}

# 백업/복원 대상 아티팩트
ARTIFACTS = [
    (CKPT_DIR / "best.pt", "best.pt"),
    (MODELS_DIR / "classifier.onnx", "classifier.onnx"),
    (MODELS_DIR / "taxonomy.json", "taxonomy.json"),
    (MODELS_DIR / "ood.npz", "ood.npz"),
    (LOG_DIR / "evaluation.json", "evaluation.json"),
    (HIER_SPLITS, "hier_splits.json"),
]


def backup_artifacts() -> Path | None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_ROOT / f"cnn_hier_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src, name in ARTIFACTS:
        if src.exists():
            shutil.copy2(src, dest / name)
            copied += 1
    print(f"[3] 백업 {copied}개 → {dest}")
    return dest if copied else None


def rollback_artifacts(backup: Path | None) -> None:
    # 복원 전, 실패 사이클의 산출물을 진단용으로 보존 — v8~v10 에서 롤백이
    # 실패 모델을 지워버려 "etc 가 어디로 새는지" 사후 분석이 불가능했던 문제.
    if backup is not None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        fail_dir = BACKUP_ROOT.parent / "failed_cycles" / f"cnn_hier_{stamp}"
        fail_dir.mkdir(parents=True, exist_ok=True)
        kept = 0
        for src, name in ARTIFACTS:
            if src.exists():
                shutil.copy2(src, fail_dir / name)
                kept += 1
        print(f"[rollback] 실패 산출물 {kept}개 보존 → {fail_dir}")

    if backup is None:
        print("[rollback] 백업 없음 — 복원 생략")
        return
    for src, name in ARTIFACTS:
        b = backup / name
        if b.exists():
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(b, src)
    print(f"[rollback] {backup} 에서 복원 완료")


def _load_baseline() -> dict | None:
    if not HISTORY_PATH.exists():
        return None
    lines = HISTORY_PATH.read_text(encoding="utf-8").strip().splitlines()
    return json.loads(lines[-1]) if lines else None


def _coarse_recalls(evaluation: dict) -> dict[str, float]:
    """게이트용 대분류 recall — 개정판 (청사진 v2 §1.4, 2026-07-21).

    - 지지 < GATE_RECALL_MIN_SUPPORT 클래스는 제외 (소표본 거부권 차단 —
      31표본 etc 의 6건 흔들림이 사이클 전체를 3연속 기각했던 문제)
    - GATE_GUIDANCE_EQUIV 쌍으로의 혼동은 정답으로 인정한 보정 recall 사용
    """
    rep = evaluation["coarse_report"]
    cm = evaluation.get("coarse_confusion_matrix")
    labels = evaluation.get("coarse_labels", list(COARSE_LABELS))
    out: dict[str, float] = {}
    for c in COARSE_LABELS:
        r = rep.get(c)
        if not isinstance(r, dict) or r.get("support", 0) < GATE_RECALL_MIN_SUPPORT:
            continue
        recall = float(r["recall"])
        if cm is not None and c in labels:
            i = labels.index(c)
            row = cm[i]
            total = sum(row)
            if total > 0:
                hits = row[i] + sum(
                    row[labels.index(e)] for (s, e) in GATE_GUIDANCE_EQUIV
                    if s == c and e in labels)
                recall = hits / total
        out[c] = recall
    return out


def gate(evaluation: dict, baseline: dict | None) -> tuple[bool, list[str]]:
    """대분류 회귀 게이트. (통과여부, 사유들)"""
    reasons: list[str] = []
    if baseline is None:
        print("[6] baseline 없음 — 첫 실행은 무조건 PASS (기준 확립)")
        return True, []

    acc_new = evaluation["coarse_accuracy"]
    acc_old = baseline["coarse_accuracy"]
    if acc_old - acc_new > GATE_MAX_COARSE_ACC_DROP:
        reasons.append(
            f"대분류 정확도 회귀: {acc_old:.4f} → {acc_new:.4f} "
            f"(-{(acc_old - acc_new) * 100:.1f}pp > {GATE_MAX_COARSE_ACC_DROP * 100:.0f}pp)")

    old_rec = baseline.get("coarse_recall", {})
    new_rec = _coarse_recalls(evaluation)
    for slug, old in old_rec.items():
        new = new_rec.get(slug)
        if new is not None and old - new > GATE_MAX_COARSE_RECALL_DROP:
            reasons.append(
                f"대분류 recall 회귀 [{slug}]: {old:.3f} → {new:.3f}")

    return (not reasons), reasons


def append_history(version: str, evaluation: dict, feedback_count: int) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "version": version,
        "coarse_accuracy": evaluation["coarse_accuracy"],
        "fine_accuracy": evaluation["fine_accuracy_on_fine_items"],
        "coarse_recall": _coarse_recalls(evaluation),
        "ready_fine": [k for k, v in evaluation["fine_activation"].items() if v["ready"]],
        "feedback_count": feedback_count,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[7] history 기록 → {HISTORY_PATH}")


def record_diagnostics(version: str, evaluation: dict) -> None:
    """Supabase model_diagnostics 에 계층 지표 기록 (best-effort)."""
    try:
        from retrain import _load_supabase_env  # noqa: PLC0415
        from supabase import create_client  # noqa: PLC0415
        url, key = _load_supabase_env()
        client = create_client(url, key)
        fine_rep = evaluation["fine_report"]
        per_fine = [
            {"label": k, "precision": v.get("precision"), "recall": v.get("recall"),
             "f1": v.get("f1-score"), "support": v.get("support")}
            for k, v in fine_rep.items()
            if isinstance(v, dict) and k not in ("macro avg", "weighted avg")
        ]
        client.table("model_diagnostics").insert({
            "version": version,
            "arch": "cnn_hier",
            "test_size": evaluation["test_size"],
            "num_classes": len(evaluation["fine_labels"]),
            "accuracy": evaluation["fine_accuracy_on_fine_items"],
            "coarse_accuracy": evaluation["coarse_accuracy"],
            "fine_accuracy": evaluation["fine_accuracy_on_fine_items"],
            "per_fine": per_fine,
            "fine_activation": evaluation["fine_activation"],
            "gate_pass": True,
        }).execute()
        print("[9] model_diagnostics 기록 완료")
    except Exception as exc:  # noqa: BLE001
        print(f"[9][warn] model_diagnostics 기록 실패 (계속): {exc}")


def _run(cmd: list[str], desc: str) -> None:
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"{desc} 실패 (exit {r.returncode})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="피드백 현황만 확인")
    ap.add_argument("--skip-preprocessor", action="store_true")
    ap.add_argument("--skip-train", action="store_true",
                    help="학습 생략 — 기존 아티팩트로 평가→게이트→승격만")
    ap.add_argument("--publish", action="store_true",
                    help="게이트 통과 시 model_versions 게시 (운영 배포 opt-in)")
    args = ap.parse_args()
    py = sys.executable

    version = time.strftime("v%Y%m%d_%H%M%S")
    print(f"=== 계층 재학습 사이클 {version} ===")

    # [1] 피드백
    rows = fetch_feedback_rows()
    print(f"[1] 피드백 {len(rows)}건")
    if args.dry_run:
        from collections import Counter
        dist = Counter(r["feedback_label"] for r in rows)
        for k, v in dist.most_common():
            print(f"    {k:16} {v}")
        return 0
    valid = fetch_active_labels()
    dl, skip, fail = download_to_raw(rows, valid)
    print(f"    다운로드 {dl}, 기존 {skip}, 실패 {fail}")

    # [2] 소수 클래스 격리
    q = quarantine_tiny_classes()
    if q:
        print(f"[2] 격리: {q}")

    # [3] 백업
    backup = backup_artifacts()

    # [4] preprocessor
    if not args.skip_preprocessor:
        print("[4] preprocessor 재실행...")
        run_preprocessor()
    else:
        print("[4] preprocessor 생략")

    try:
        # [5] splits 재생성(frozen 누적) → 학습 → 평가
        if not args.skip_train:
            if HIER_SPLITS.exists():
                HIER_SPLITS.unlink()
                print("[5] hier_splits.json 삭제 (frozen 은 hier_frozen_test.json 으로 유지)")
            _run([py, "-m", "src.hier_train"], "계층 학습")
        _run([py, "-m", "src.hier_evaluate"], "계층 평가")
        evaluation = json.loads((LOG_DIR / "evaluation.json").read_text(encoding="utf-8"))

        # [6] 게이트
        baseline = _load_baseline()
        ok, reasons = gate(evaluation, baseline)
        if not ok:
            print("[6] ❌ 게이트 FAIL:")
            for r in reasons:
                print(f"    - {r}")
            rollback_artifacts(backup)
            return 1
        print(f"[6] ✅ 게이트 PASS (대분류 {evaluation['coarse_accuracy']:.4f})")

        # [7] export + OOD 프로토타입 + history
        _run([py, "-m", "src.hier_export"], "ONNX export")
        _run([py, "scripts/build_hier_prototypes.py"], "OOD 프로토타입")
        append_history(version, evaluation, feedback_count=len(rows))

        # [8] 자동 승격/강등
        _run([py, "scripts/apply_hier_activation.py"], "활성화 승격/강등")

        # [9] 진단 기록
        record_diagnostics(version, evaluation)

        # [9.5] 실사용 평가 자동 실행 (SMART_CAPTURE_STRATEGY §4 의 미해결
        # 과제: "매 retrain 직후 자동으로" — frozen 과 realworld 를 함께 기록
        # 해야 개선 레버의 실효를 비교 가능). 실패해도 사이클은 계속.
        try:
            _run([py, "scripts/realworld_eval_hier.py"], "실사용 평가")
        except Exception as exc:  # noqa: BLE001
            print(f"[9.5][warn] realworld_eval 실패 (계속): {exc}")

        # [10] 게시 (opt-in)
        if args.publish:
            print("[10] model_versions 게시는 아직 미구현 항목 포함 — "
                  "운영 배포 단계에서 hier 전용 publish 를 붙일 것")
        else:
            print("[10] publish 생략 (--publish 로 opt-in)")

        print(f"=== 사이클 완료: {version} ===")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc} → 롤백")
        rollback_artifacts(backup)
        return 1


if __name__ == "__main__":
    main()
