#!/usr/bin/env bash
# B-1.1 Stage D 실험: 클래스 가중치 cap=2.5 로 재학습 → 측정 (Supabase publish 없음).
# baseline 비교 기준: outputs/backups/test_A_baseline/classifier.onnx (현 active 와 sha 동일).
# baseline test 정확도: 0.9557 (outputs/logs/cnn/evaluation_capbaseline.json).
set -euo pipefail
cd "$(dirname "$0")/.."

export WASTE_CNN_WEIGHT_CAP="${WASTE_CNN_WEIGHT_CAP:-2.5}"
PY=.venv/bin/python
BASELINE=outputs/backups/test_A_baseline/classifier.onnx
TS() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[cap-exp $(TS)] START cap=${WASTE_CNN_WEIGHT_CAP}"

echo "[cap-exp $(TS)] 1/4 train+evaluate+export (main.py all --arch cnn)"
$PY main.py all --arch cnn

echo "[cap-exp $(TS)] 2/4 test accuracy (회귀 체크)"
$PY - <<'PY'
import json
new = json.load(open("outputs/logs/cnn/evaluation.json"))["accuracy"]
old = json.load(open("outputs/logs/cnn/evaluation_capbaseline.json"))["accuracy"]
print(f"  test acc: baseline {old:.4f} -> cap2.5 {new:.4f}  (Δ {(new-old)*100:+.2f}pp)")
PY

echo "[cap-exp $(TS)] 3/4 realworld_eval (Supabase read-only)"
$PY realworld_eval.py || echo "  [warn] realworld_eval 실패(피드백 부족 가능) — 계속"

echo "[cap-exp $(TS)] 4/4 revalidate vs baseline (apples-to-apples, dry-run)"
$PY revalidate.py --model outputs/models/cnn/classifier.onnx --baseline "$BASELINE"

echo "[cap-exp $(TS)] DONE — 모델 미publish. 결과 검토 후 채택 결정."
