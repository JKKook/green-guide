#!/bin/bash
# 야간 통합 chain (재부팅 후 재시도 버전).
# - TACO 1,145 살아남은 그대로 사용 (재실행 안 함)
# - MIT 67 재다운로드 대기 → 배경 추출 → 합성 → 학습 → 측정 → 복원
# - caffeinate 로 sleep 방지 (호출자가 외부에서 캐피네이트 띄움)

cd "$(dirname "$0")/.."

LOG=outputs/logs/overnight_pipeline_v2.log
echo "[overnight v2] start at $(date)" | tee -a "$LOG"

fail() {
  echo "[overnight v2] FAILED at $1: $(date)" | tee -a "$LOG"
  touch outputs/logs/OVERNIGHT_FAILED
  exit 1
}
set -e

# ─── Phase 0: MIT 67 다운로드 완료 대기 ─────────────────
echo "[overnight v2] Phase 0: waiting for MIT Indoor 67 download..." | tee -a "$LOG"
while pgrep -f 'curl.*indoorCVPR' > /dev/null 2>&1; do
  sleep 30
done
MIT_SZ=$(stat -f%z /tmp/indoorCVPR_09.tar 2>/dev/null || echo 0)
echo "  MIT 67 final size: ${MIT_SZ} bytes" | tee -a "$LOG"
if [ "$MIT_SZ" -lt 2000000000 ]; then
  fail "Phase 0 (MIT 67 download incomplete: $MIT_SZ < 2GB)"
fi

# ─── Phase 1: TACO 살아남은 파일 manifest 등재 ─────────
echo "[overnight v2] Phase 1: extending manifest with surviving TACO files..." | tee -a "$LOG"
.venv/bin/python scripts/extend_manifest_taco.py >> "$LOG" 2>&1 \
  || fail "Phase 1 (TACO manifest extend)"

# ─── Phase 2: 배경 풀 추출 ─────────────────────────────
echo "[overnight v2] Phase 2: extracting indoor backgrounds (per-category=30)..." | tee -a "$LOG"
.venv/bin/python scripts/build_indoor_bg_pool.py --per-category 30 >> "$LOG" 2>&1 \
  || fail "Phase 2 (bg pool)"
BG_COUNT=$(ls ../preprocessor/data/raw/_aux/backgrounds/*.jpg 2>/dev/null | wc -l | tr -d ' ')
echo "  배경 풀: $BG_COUNT 장" | tee -a "$LOG"
if [ "$BG_COUNT" -lt 200 ]; then
  fail "Phase 2 (too few backgrounds: $BG_COUNT)"
fi

# ─── Phase 3: Sanity 합성 (5 클래스 × 5장) ──────────────
echo "[overnight v2] Phase 3: sanity synthesis (5 each)..." | tee -a "$LOG"
for cls in etc cardboard food_waste trash electronics; do
  .venv/bin/python scripts/synthesize_indoor.py --our-class "$cls" --n 5 --seed 1 >> "$LOG" 2>&1 \
    || fail "Phase 3 sanity ($cls)"
done

# ─── Phase 4: 본격 합성 (1,800장) ───────────────────────
echo "[overnight v2] Phase 4: full synthesis..." | tee -a "$LOG"
.venv/bin/python scripts/synthesize_indoor.py --our-class etc         --n 400 --seed 42 >> "$LOG" 2>&1 \
  || fail "Phase 4 etc"
.venv/bin/python scripts/synthesize_indoor.py --our-class cardboard   --n 350 --seed 43 >> "$LOG" 2>&1 \
  || fail "Phase 4 cardboard"
.venv/bin/python scripts/synthesize_indoor.py --our-class food_waste  --n 350 --seed 44 >> "$LOG" 2>&1 \
  || fail "Phase 4 food_waste"
.venv/bin/python scripts/synthesize_indoor.py --our-class trash       --n 350 --seed 45 >> "$LOG" 2>&1 \
  || fail "Phase 4 trash"
.venv/bin/python scripts/synthesize_indoor.py --our-class electronics --n 350 --seed 46 >> "$LOG" 2>&1 \
  || fail "Phase 4 electronics"

total=0
for cls in etc cardboard food_waste trash electronics; do
  n=$(ls ../preprocessor/data/raw/synthetic_indoor/"$cls"/synth_*.jpg 2>/dev/null | wc -l | tr -d ' ')
  echo "  $cls: $n synthesized" | tee -a "$LOG"
  total=$((total + n))
done
echo "  TOTAL: $total" | tee -a "$LOG"
[ "$total" -lt 1200 ] && fail "Phase 4 (total synth too few: $total)"

# ─── Phase 5: Test A baseline 백업 ─────────────────────
echo "[overnight v2] Phase 5: backup Test A baseline..." | tee -a "$LOG"
mkdir -p outputs/backups/test_C1_pre
cp outputs/models/cnn/classifier.onnx outputs/backups/test_C1_pre/classifier.onnx
cp ../preprocessor/data/processed/manifest.json outputs/backups/test_C1_pre/manifest.json
if [ -f data/splits/splits.json ]; then
  cp data/splits/splits.json outputs/backups/test_C1_pre/splits.json
fi

# ─── Phase 6: manifest 확장 (synth) + splits 재생성 ────
echo "[overnight v2] Phase 6: extend manifest with synth + reset splits..." | tee -a "$LOG"
.venv/bin/python scripts/extend_manifest_synthetic.py >> "$LOG" 2>&1 \
  || fail "Phase 6 (extend manifest)"
mv data/splits/splits.json data/splits/splits.json.bak_pre_C1 2>/dev/null || true

# ─── Phase 7: Test C1 학습 ──────────────────────────────
echo "[overnight v2] Phase 7: train Test C1 (CNN 15 epochs, early stop)..." | tee -a "$LOG"
.venv/bin/python main.py train --arch cnn >> outputs/logs/test_C1_train.log 2>&1 \
  || fail "Phase 7 (train)"

# ─── Phase 8: export + measure ──────────────────────────
echo "[overnight v2] Phase 8: export ONNX + measure..." | tee -a "$LOG"
.venv/bin/python main.py export --arch cnn >> "$LOG" 2>&1 \
  || fail "Phase 8 (export)"
mkdir -p outputs/backups/test_C1
cp outputs/models/cnn/classifier.onnx outputs/backups/test_C1/classifier.onnx

.venv/bin/python realworld_eval.py > outputs/logs/test_C1_realworld.log 2>&1 || true
.venv/bin/python diagnose.py --arch cnn --version test_C1 > outputs/logs/test_C1_diagnose.log 2>&1 || true

# ─── Phase 9: Test A 복원 + 합성 garbage-classification 정리 ─
echo "[overnight v2] Phase 9: restore Test A baseline + cleanup..." | tee -a "$LOG"
cp outputs/backups/test_C1_pre/classifier.onnx outputs/models/cnn/classifier.onnx
cp outputs/backups/test_C1_pre/manifest.json ../preprocessor/data/processed/manifest.json
if [ -f data/splits/splits.json.bak_pre_C1 ]; then
  mv data/splits/splits.json.bak_pre_C1 data/splits/splits.json
fi
.venv/bin/python scripts/extend_manifest_synthetic.py --cleanup >> "$LOG" 2>&1 || true

echo "[overnight v2] ALL DONE at $(date)" | tee -a "$LOG"
touch outputs/logs/OVERNIGHT_DONE
