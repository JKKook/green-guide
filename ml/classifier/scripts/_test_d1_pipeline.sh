#!/bin/bash
# Test D1 — TACO 1,145 영구 등재 + retrain.
# 합성 데이터 미포함 (Test C1 효과가 경계선이라 별도 결정).
# 결과는 비교용으로만 — 자동 promote 안 함, 사용자가 결정.
#
# Phase 1: manifest 등재 (extend_manifest_taco.py)
# Phase 2: Test A 백업
# Phase 3: splits 재생성 강제
# Phase 4: CNN 학습 (~4h, early stop)
# Phase 5: export + measure
# Phase 6: Test A 복원 (active 모델 보호)

cd "$(dirname "$0")/.."

LOG=outputs/logs/test_D1_pipeline.log
echo "[D1] start at $(date)" | tee -a "$LOG"

fail() {
  echo "[D1] FAILED at $1: $(date)" | tee -a "$LOG"
  touch outputs/logs/TEST_D1_FAILED
  exit 1
}
set -e

# ─── Phase 1: TACO manifest 등재 ─────────────────────
echo "[D1] Phase 1: TACO manifest 등재..." | tee -a "$LOG"
.venv/bin/python scripts/extend_manifest_taco.py >> "$LOG" 2>&1 || fail "Phase 1"

# ─── Phase 2: Test A 백업 ─────────────────────────────
echo "[D1] Phase 2: Test A 백업..." | tee -a "$LOG"
mkdir -p outputs/backups/test_D1_pre
cp outputs/models/cnn/classifier.onnx outputs/backups/test_D1_pre/classifier.onnx
cp ../preprocessor/data/processed/manifest.json outputs/backups/test_D1_pre/manifest.json
if [ -f data/splits/splits.json ]; then
  cp data/splits/splits.json outputs/backups/test_D1_pre/splits.json
fi

# ─── Phase 3: splits 재생성 강제 ──────────────────────
mv data/splits/splits.json data/splits/splits.json.bak_pre_D1 2>/dev/null || true

# ─── Phase 4: CNN 학습 ──────────────────────────────
echo "[D1] Phase 4: CNN 학습 시작..." | tee -a "$LOG"
.venv/bin/python main.py train --arch cnn >> outputs/logs/test_D1_train.log 2>&1 \
  || fail "Phase 4"

# ─── Phase 5: export + measure ──────────────────────
echo "[D1] Phase 5: export + measure..." | tee -a "$LOG"
.venv/bin/python main.py export --arch cnn >> "$LOG" 2>&1 || fail "Phase 5 export"
mkdir -p outputs/backups/test_D1
cp outputs/models/cnn/classifier.onnx outputs/backups/test_D1/classifier.onnx

.venv/bin/python realworld_eval.py > outputs/logs/test_D1_realworld.log 2>&1 || true
.venv/bin/python diagnose.py --arch cnn --version test_D1 > outputs/logs/test_D1_diagnose.log 2>&1 || true

# ─── Phase 6: Test A 복원 (active 보호) ──────────────
echo "[D1] Phase 6: Test A 복원..." | tee -a "$LOG"
cp outputs/backups/test_D1_pre/classifier.onnx outputs/models/cnn/classifier.onnx
cp outputs/backups/test_D1_pre/manifest.json ../preprocessor/data/processed/manifest.json
if [ -f data/splits/splits.json.bak_pre_D1 ]; then
  mv data/splits/splits.json.bak_pre_D1 data/splits/splits.json
fi

echo "[D1] ALL DONE at $(date)" | tee -a "$LOG"
touch outputs/logs/TEST_D1_DONE
