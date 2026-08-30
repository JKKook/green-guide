"""모델 진단 엔진 — 재학습마다 실행하는 반복 가능한 진단 프로세스.

산출:
  1. per-class precision / recall / f1 + 혼동행렬 (고정 held-out test 기준)
  2. 혼동 쌍 경보 (off-diagonal 임계 초과)
  3. 약한 클래스 + "데이터 더 필요" 우선순위
  4. 직전 버전 대비 클래스별 회귀 감지 (zlatan 사례 자동 탐지)
  5. 자동 루프용 PASS/FAIL 게이트 판정

저장:
  - 레포: outputs/logs/diagnosis/<version>.json + history.jsonl
  - Supabase: model_diagnostics 테이블 (supabase_sync 모듈, 선택)

사용:
    python diagnose.py --arch cnn --version v20260524_025122
    python diagnose.py            # arch=cnn, version=auto(timestamp)
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src import config
from src.dataset import build_dataset, load_manifest
from src.evaluate import collect_predictions
from src.frozen_test import ensure_frozen_test, load_frozen_keys
from src.model import build_model
from src.train import _input_mode, _model_kind, get_hyperparams, pick_device

# ── 임계값 ─────────────────────────────────────────────
WEAK_F1 = 0.85               # f1 < 0.85 → 약한 클래스
CONFUSION_PAIR_FRAC = 0.03   # off-diagonal ≥ source 클래스의 3% → 혼동쌍 경보
CONFUSION_PAIR_MIN = 5       # 그리고 절대 건수 ≥ 5

# ── 게이트 (자동 루프 안전판) ──────────────────────────
GATE_MAX_CLASS_RECALL_DROP = 0.05   # 기존 클래스 recall 5pp↑ 하락 → FAIL
GATE_MAX_ACC_DROP = 0.02            # 전체 정확도 2pp↑ 하락 → FAIL


DIAG_DIR: Path = config.LOGS_DIR / "diagnosis"
HISTORY_PATH: Path = DIAG_DIR / "history.jsonl"


def _sync_to_supabase(report: dict[str, Any]) -> None:
    """model_diagnostics 테이블에 한 행 기록 (best-effort — 실패해도 진단은 정상).

    테이블은 migrations/004_model_diagnostics.sql 로 미리 생성돼 있어야 함.
    """
    try:
        import os

        from dotenv import load_dotenv
        from supabase import create_client

        load_dotenv(config.PREPROCESSOR_ROOT / ".env")
        url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
        if not url or not key:
            print("[diagnose] Supabase env 없음 — 레포 파일에만 기록")
            return
        client = create_client(url, key)
        client.table("model_diagnostics").insert({
            "version": report["version"],
            "arch": report["arch"],
            "created_at": report["created_at"],
            "test_size": report["test_size"],
            "num_classes": report["num_classes"],
            "accuracy": report["accuracy"],
            "macro_f1": report["macro_f1"],
            "per_class": report["per_class"],
            "confusion_pairs": report["confusion_pairs"],
            "weak_classes": report["weak_classes"],
            "needs_data": report["needs_data"],
            "regressions": report["regressions"],
            "gate_pass": report["gate"]["pass"],
            "gate_reasons": report["gate"]["reasons"],
        }).execute()
        print("[diagnose] Supabase model_diagnostics += 1 row")
    except Exception as exc:  # noqa: BLE001
        print(f"[diagnose] Supabase 기록 실패 (무시): {exc}")


def _load_frozen_test_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = load_frozen_keys()
    if not keys:
        keys = ensure_frozen_test(items)
    return [it for it in items if it["source_path"] in keys]


def _load_model(arch: str, device: torch.device) -> torch.nn.Module:
    ckpt_path = config.arch_subdir(config.CHECKPOINTS_DIR, arch) / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(_model_kind(arch)).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[diagnose:{arch}] checkpoint epoch {ckpt.get('epoch')} "
          f"(val_acc={ckpt.get('val_acc', float('nan')):.4f})")
    return model


def _confusion_pairs(cm: np.ndarray, labels: list[str]) -> list[dict[str, Any]]:
    """off-diagonal 이 임계를 넘는 (실제→예측) 쌍 — 큰 순."""
    pairs: list[dict[str, Any]] = []
    for i, row in enumerate(cm):
        total = int(row.sum())
        if total == 0:
            continue
        for j, v in enumerate(row):
            v = int(v)
            if i == j or v < CONFUSION_PAIR_MIN:
                continue
            frac = v / total
            if frac >= CONFUSION_PAIR_FRAC:
                pairs.append({
                    "true": labels[i], "pred": labels[j],
                    "count": v, "frac_of_true": round(frac, 4),
                })
    pairs.sort(key=lambda p: p["count"], reverse=True)
    return pairs


def _prev_per_class() -> dict[str, dict[str, float]] | None:
    """history 의 가장 최근(직전) 엔트리의 per-class recall/f1 맵."""
    if not HISTORY_PATH.exists():
        return None
    last = None
    for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            last = line
    if not last:
        return None
    try:
        entry = json.loads(last)
        return {c["label"]: c for c in entry.get("per_class", [])}
    except Exception:  # noqa: BLE001
        return None


def run_diagnosis(
    arch: str = "cnn",
    version: str | None = None,
    sync_supabase: bool = True,
    commit_history: bool = True,
) -> dict[str, Any]:
    config.ensure_directories()
    # manifest 가 (preprocessor 재실행 등으로) 바뀌었을 수 있어 클래스 목록 재로드
    config.refresh_classes_from_manifest()
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    version = version or datetime.now(UTC).strftime("v%Y%m%d_%H%M%S")
    labels = list(config.CLASS_LABELS)

    device = pick_device()
    hp = get_hyperparams(arch)
    model = _load_model(arch, device)

    items = load_manifest()
    test_items = _load_frozen_test_items(items)
    test_ds = build_dataset(arch, test_items, input_mode=_input_mode(arch))
    loader = DataLoader(test_ds, batch_size=hp.batch_size, shuffle=False)
    print(f"[diagnose:{arch}] 고정 test: {len(test_ds):,}장 / {len(labels)} 클래스")

    y_true, y_pred = collect_predictions(model, loader, device)

    accuracy = float((y_true == y_pred).mean())
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(labels))), zero_division=0,
    )
    macro_f1 = float(np.mean(f1))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    per_class = [
        {
            "label": labels[i],
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i in range(len(labels))
    ]
    confusion_pairs = _confusion_pairs(cm, labels)

    # 약한 클래스 (support 있는 것 중 f1 낮음)
    weak = sorted(
        [c for c in per_class if c["support"] > 0 and c["f1"] < WEAK_F1],
        key=lambda c: c["f1"],
    )

    # 회귀 감지 (직전 버전 대비 recall 하락)
    prev = _prev_per_class()
    regressions: list[dict[str, Any]] = []
    if prev:
        for c in per_class:
            pc = prev.get(c["label"])
            if pc and c["support"] > 0:
                drop = pc["recall"] - c["recall"]
                if drop >= 0.03:
                    regressions.append({
                        "label": c["label"],
                        "prev_recall": pc["recall"],
                        "new_recall": c["recall"],
                        "drop_pp": round(drop * 100, 2),
                    })

    # "데이터 더 필요" 우선순위 — 약한 클래스 + 혼동쌍의 source
    needs_data = sorted({c["label"] for c in weak}
                        | {pr["true"] for pr in confusion_pairs})

    # 게이트 판정 (자동 루프 안전판)
    gate_reasons: list[str] = []
    prev_acc = None
    if HISTORY_PATH.exists():
        lines = [l for l in HISTORY_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            try:
                prev_acc = json.loads(lines[-1]).get("accuracy")
            except Exception:  # noqa: BLE001
                prev_acc = None
    if prev_acc is not None and (prev_acc - accuracy) > GATE_MAX_ACC_DROP:
        gate_reasons.append(
            f"전체 정확도 {prev_acc:.4f}→{accuracy:.4f} "
            f"({(accuracy-prev_acc)*100:+.2f}pp) — 임계 -{GATE_MAX_ACC_DROP*100:.0f}pp 초과"
        )
    for reg in regressions:
        if reg["drop_pp"] >= GATE_MAX_CLASS_RECALL_DROP * 100:
            gate_reasons.append(
                f"{reg['label']} recall {reg['prev_recall']:.3f}→{reg['new_recall']:.3f} "
                f"(-{reg['drop_pp']}pp)"
            )
    gate_pass = len(gate_reasons) == 0

    report: dict[str, Any] = {
        "version": version,
        "arch": arch,
        "created_at": datetime.now(UTC).isoformat(),
        "test_size": int(len(test_ds)),
        "num_classes": len(labels),
        "class_labels": labels,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_pairs": confusion_pairs,
        "weak_classes": [c["label"] for c in weak],
        "needs_data": needs_data,
        "regressions": regressions,
        "gate": {"pass": gate_pass, "reasons": gate_reasons},
    }

    _write_report(report)  # 버전별 상세 json — 항상 (실패 시도도 디버그용 보존)
    # 추적 이력(history + Supabase)은 게이트 통과 버전만 커밋 — 실패 시도가
    # 다음 회귀 비교의 baseline 을 오염시키지 않도록.
    if commit_history and report["gate"]["pass"]:
        _append_history(report)
        if sync_supabase:
            _sync_to_supabase(report)
    elif commit_history and not report["gate"]["pass"]:
        print("[diagnose] 게이트 FAIL — history/Supabase 커밋 생략 (baseline 보존)")
    _print_summary(report)
    return report


def _write_report(report: dict[str, Any]) -> None:
    """버전별 상세 리포트 json — 항상 기록 (실패 시도도 디버그용)."""
    full_path = DIAG_DIR / f"{report['version']}.json"
    full_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[diagnose] report → {full_path}")


def _append_history(report: dict[str, Any]) -> None:
    """history.jsonl 한 줄 요약 — 회귀 비교의 baseline. 게이트 통과 버전만."""
    summary = {
        "version": report["version"],
        "created_at": report["created_at"],
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_f1"],
        "num_classes": report["num_classes"],
        "per_class": report["per_class"],
        "weak_classes": report["weak_classes"],
        "gate_pass": report["gate"]["pass"],
    }
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"[diagnose] history += {HISTORY_PATH}")


def _print_summary(report: dict[str, Any]) -> None:
    print()
    print("=" * 60)
    print(f"진단 {report['version']}  acc={report['accuracy']:.4f}  "
          f"macroF1={report['macro_f1']:.4f}  test={report['test_size']:,}")
    print("=" * 60)
    if report["weak_classes"]:
        print(f"약한 클래스 (f1<{WEAK_F1}): {report['weak_classes']}")
    if report["confusion_pairs"]:
        print("혼동 쌍 (≥3% & ≥5건):")
        for pr in report["confusion_pairs"][:8]:
            print(f"  {pr['true']:11} → {pr['pred']:11} {pr['count']:4}건 "
                  f"({pr['frac_of_true']*100:.1f}%)")
    if report["regressions"]:
        print("⚠️  회귀 (직전 대비 recall 하락):")
        for reg in report["regressions"]:
            print(f"  {reg['label']:11} {reg['prev_recall']:.3f}→{reg['new_recall']:.3f} "
                  f"(-{reg['drop_pp']}pp)")
    if report["needs_data"]:
        print(f"데이터 보강 우선순위: {report['needs_data']}")
    g = report["gate"]
    print(f"게이트: {'PASS ✅' if g['pass'] else 'FAIL ❌'}")
    for reason in g["reasons"]:
        print(f"  - {reason}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="diagnose")
    ap.add_argument("--arch", default="cnn", choices=list(config.SUPPORTED_ARCHS))
    ap.add_argument("--version", default=None, help="모델 버전 태그 (기본: UTC timestamp)")
    ap.add_argument("--no-supabase", action="store_true", help="Supabase 기록 건너뛰기")
    args = ap.parse_args()
    rep = run_diagnosis(
        arch=args.arch, version=args.version, sync_supabase=not args.no_supabase,
    )
    raise SystemExit(0 if rep["gate"]["pass"] else 1)
