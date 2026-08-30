"""피드백 현황·retrain 준비도 모니터 (B-2.1) — Supabase READ-ONLY.

active learning 루프의 기계장치(collect→retrain→gate→publish)는 retrain.py 에
이미 있다. 빠진 건 '언제/무엇을 retrain 할지' 판단할 가시성. 이 스크립트는
user_uploads / model_versions 를 읽기만 해서 다음을 보고한다 (쓰기 없음):

  1. 수집량      — pending/confirmed/corrected, 라벨링 완료 총량
  2. retrain 준비도 — 클래스별 라벨 수 vs MIN_SAMPLES_PER_CLASS,
                      active 모델 이후 net-new 라벨 수
  3. 미학습 라벨   — 사용자가 고른 라벨인데 모델이 출력 못 하는 것(신규 클래스 후보)
  4. 드리프트 신호 — 예측 confidence / 정규화 entropy 분포, reject 임계 미만 비율
  5. 정정률       — corrected/(confirmed+corrected), 상위 혼동쌍(predicted→정답)
  6. 능동 샘플 큐  — 라벨 대기 중 가장 가치 높은(고엔트로피·저확신) 업로드 top-N

사용:  .venv/bin/python feedback_monitor.py [--top 15]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import UTC, datetime

from waste_common.supabase import get_client

from retrain import MIN_SAMPLES_PER_CLASS
from src import config

OUT_PATH = config.LOGS_DIR / "cnn" / "feedback_status.json"

# 앱(confidence.dart)의 _kRejectThreshold 와 동기화. 이 미만이면 앱이 reject.
REJECT_THRESHOLD = 0.55
# active 모델 이후 신규 피드백이 이만큼 쌓이면 retrain 트리거 (2026-06-15 확정).
RETRAIN_TRIGGER_NEW = 100


def _normalized_entropy(probs: dict[str, float]) -> float | None:
    """all_probabilities dict → 정규화 entropy(0~1). 높을수록 모델이 헷갈림."""
    vals = [float(v) for v in (probs or {}).values() if v is not None]
    vals = [v for v in vals if v > 0]
    if len(vals) < 2:
        return None
    h = -sum(v * math.log(v) for v in vals)
    return h / math.log(len(vals))


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(prog="feedback_monitor")
    ap.add_argument("--top", type=int, default=15, help="능동 샘플 큐 출력 개수")
    args = ap.parse_args()

    config.refresh_classes_from_manifest()
    labels = set(config.CLASS_LABELS)

    cli = get_client()

    rows = (cli.table("user_uploads")
            .select("id,image_url,predicted_class,predicted_confidence,"
                    "all_probabilities,feedback_status,feedback_label,uploaded_at,feedback_at")
            .execute().data) or []

    active = (cli.table("model_versions")
              .select("version,created_at,feedback_count,class_labels")
              .eq("is_active", True).limit(1).execute().data) or []
    active = active[0] if active else None
    active_since = _parse_dt(active.get("created_at")) if active else None

    # ── 1. 수집량 ────────────────────────────────────────────────
    status_counts = Counter(r.get("feedback_status") or "pending" for r in rows)
    labeled = [r for r in rows if r.get("feedback_status") in ("confirmed", "corrected")]
    pending = [r for r in rows if r.get("feedback_status") == "pending"]

    # ── 2. retrain 준비도 ───────────────────────────────────────
    per_class_labeled: Counter = Counter(
        r["feedback_label"] for r in labeled
        if r.get("feedback_label") in labels)
    below_floor = {c: per_class_labeled.get(c, 0)
                   for c in sorted(labels)
                   if per_class_labeled.get(c, 0) < MIN_SAMPLES_PER_CLASS}
    new_since_active = sum(
        1 for r in labeled
        if active_since and (_parse_dt(r.get("feedback_at") or r.get("uploaded_at"))
                             or active_since) > active_since)

    # ── 3. 미학습 라벨 backlog ──────────────────────────────────
    untrained = Counter(
        r["feedback_label"] for r in labeled
        if r.get("feedback_label") and r["feedback_label"] not in labels)

    # ── 4. 드리프트 신호 ────────────────────────────────────────
    confs = [float(r["predicted_confidence"]) for r in rows
             if r.get("predicted_confidence") is not None]
    entropies = [e for e in (_normalized_entropy(r.get("all_probabilities")) for r in rows)
                 if e is not None]
    below_reject = sum(1 for c in confs if c < REJECT_THRESHOLD)
    drift = {
        "n_pred": len(confs),
        "mean_confidence": round(sum(confs) / len(confs), 4) if confs else None,
        "pct_below_reject": round(below_reject / len(confs), 4) if confs else None,
        "mean_norm_entropy": round(sum(entropies) / len(entropies), 4) if entropies else None,
    }

    # ── 5. 정정률 + 혼동 ────────────────────────────────────────
    n_conf = status_counts.get("confirmed", 0)
    n_corr = status_counts.get("corrected", 0)
    correction_rate = round(n_corr / (n_conf + n_corr), 4) if (n_conf + n_corr) else None
    confusions: Counter = Counter()
    for r in labeled:
        if r.get("feedback_status") == "corrected" and r.get("predicted_class") and r.get("feedback_label"):
            confusions[f"{r['predicted_class']}→{r['feedback_label']}"] += 1

    # ── 6. 능동 샘플 큐 (라벨 대기 중 고엔트로피 우선) ──────────
    queue = []
    for r in pending:
        e = _normalized_entropy(r.get("all_probabilities"))
        queue.append({
            "id": r["id"], "predicted_class": r.get("predicted_class"),
            "confidence": r.get("predicted_confidence"),
            "norm_entropy": round(e, 4) if e is not None else None,
            "image_url": r.get("image_url"),
        })
    queue.sort(key=lambda x: (x["norm_entropy"] is None, -(x["norm_entropy"] or 0)))
    queue = queue[:args.top]

    # ── verdict ─────────────────────────────────────────────────
    ready = (not below_floor) and (new_since_active >= RETRAIN_TRIGGER_NEW or not active)

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "active_model": (active or {}).get("version"),
        "active_since": (active or {}).get("created_at"),
        "total_uploads": len(rows),
        "status_counts": dict(status_counts),
        "labeled_total": len(labeled),
        "new_labeled_since_active": new_since_active,
        "per_class_labeled": dict(sorted(per_class_labeled.items())),
        "below_min_samples": below_floor,
        "min_samples_per_class": MIN_SAMPLES_PER_CLASS,
        "untrained_label_backlog": dict(untrained),
        "drift": drift,
        "correction_rate": correction_rate,
        "top_confusions": confusions.most_common(10),
        "active_sampling_queue": queue,
        "retrain_ready": ready,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 콘솔 요약 ───────────────────────────────────────────────
    print("=" * 62)
    print(f"피드백 현황  (active 모델: {report['active_model'] or '미등록'})")
    print("=" * 62)
    print(f"  업로드 총 {len(rows)}건 | "
          f"pending {status_counts.get('pending', 0)} · "
          f"confirmed {n_conf} · corrected {n_corr}")
    print(f"  라벨링 완료 {len(labeled)}건 (active 이후 신규 {new_since_active}건)")
    if drift["mean_confidence"] is not None:
        print(f"  드리프트: 평균확신 {drift['mean_confidence']*100:.1f}% | "
              f"reject(<{REJECT_THRESHOLD}) 비율 {drift['pct_below_reject']*100:.1f}% | "
              f"평균엔트로피 {drift['mean_norm_entropy']}")
    if correction_rate is not None:
        print(f"  정정률(실사용 오류 proxy): {correction_rate*100:.1f}%")
    if below_floor:
        print(f"  ⚠ 표본 부족 클래스(<{MIN_SAMPLES_PER_CLASS}): {below_floor}")
    if untrained:
        print(f"  ⚠ 미학습 라벨(신규 클래스 후보): {dict(untrained)}")
    if confusions:
        print("  상위 혼동(예측→정답):")
        for pair, c in confusions.most_common(6):
            print(f"     {pair:28} {c}건")
    if queue:
        print(f"  능동 샘플 큐(라벨하면 가치 큰 대기건 top {len(queue)}):")
        for q in queue[:min(8, len(queue))]:
            print(f"     {q['id']}  {q['predicted_class']:12} "
                  f"conf={q['confidence']}  H={q['norm_entropy']}")
    print("-" * 62)
    print(f"  retrain 권장: {'예 ✅' if ready else '아직 ⏳'}  "
          f"(신규 {new_since_active}/{RETRAIN_TRIGGER_NEW}, "
          f"표본부족 {len(below_floor)}클래스)")
    print(f"  → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
