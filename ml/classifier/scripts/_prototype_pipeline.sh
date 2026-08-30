#!/bin/bash
# 다운로드 → 배경 추출 → 합성 prototype 자동 실행.
set -e
cd "$(dirname "$0")/.."

LOG=outputs/logs/prototype_pipeline.log
echo "[pipeline] start at $(date)" | tee -a "$LOG"

# 1. MIT Indoor 67 다운로드 대기
echo "[pipeline] waiting for MIT Indoor 67 download..." | tee -a "$LOG"
while [ ! -f /tmp/indoorCVPR_09.tar ] || ! pgrep -f "curl.*indoorCVPR" > /dev/null 2>&1 && [ "$(stat -f%z /tmp/indoorCVPR_09.tar 2>/dev/null || echo 0)" -lt 2400000000 ]; do
  # 파일이 존재하고 다운로드 프로세스가 끝났으면 OK
  if [ -f /tmp/indoorCVPR_09.tar ] && ! pgrep -f "curl.*indoorCVPR" > /dev/null 2>&1; then
    sz=$(stat -f%z /tmp/indoorCVPR_09.tar)
    if [ "$sz" -gt 100000000 ]; then  # 100MB 이상 = OK 추정
      break
    fi
  fi
  sleep 30
done
SZ=$(stat -f%z /tmp/indoorCVPR_09.tar 2>/dev/null || echo 0)
echo "[pipeline] MIT 67 downloaded: ${SZ} bytes" | tee -a "$LOG"

# 2. 배경 추출 (high priority 카테고리당 30장)
echo "[pipeline] extracting backgrounds (per-category=30)..." | tee -a "$LOG"
.venv/bin/python scripts/build_indoor_bg_pool.py --per-category 30 >> "$LOG" 2>&1

# 3. TACO 완료 대기 (PID 또는 cap 도달 신호)
echo "[pipeline] waiting for TACO ingest..." | tee -a "$LOG"
while pgrep -f "integrate_taco.py" > /dev/null 2>&1; do
  sleep 30
done
echo "[pipeline] TACO done" | tee -a "$LOG"

# 4. 합성 prototype (3 클래스 × 50장 = 150장 smoke test)
echo "[pipeline] running synthesis prototype (3 classes × 50)..." | tee -a "$LOG"
for cls in etc cardboard food_waste; do
  .venv/bin/python scripts/synthesize_indoor.py \
    --our-class "$cls" --n 50 --seed 42 >> "$LOG" 2>&1 || true
done

# 5. 합성 결과 spot check
echo "[pipeline] synthesis counts:" | tee -a "$LOG"
for cls in etc cardboard food_waste; do
  cnt=$(ls ../preprocessor/data/raw/synthetic_indoor/"$cls"/ 2>/dev/null | wc -l)
  echo "  $cls: $cnt synthesized" | tee -a "$LOG"
done

echo "[pipeline] DONE at $(date)" | tee -a "$LOG"
touch outputs/logs/PROTOTYPE_DONE
