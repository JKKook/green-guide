#!/bin/bash
# Test B 학습 종료 후 → export + realworld + diagnose 를 순차 자동 실행.
# 학습 프로세스 PID 가 사라지면 시작.
set -e
cd "$(dirname "$0")/.."

TRAIN_PID="${1:-}"
if [ -z "$TRAIN_PID" ]; then
  echo "usage: $0 <train_pid>" >&2
  exit 1
fi

echo "[continuation] waiting for train PID $TRAIN_PID ..." | tee -a outputs/logs/test_B_continuation.log
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 30
done
echo "[continuation] training finished at $(date)" | tee -a outputs/logs/test_B_continuation.log

# Export ONNX (replaces outputs/models/cnn/classifier.onnx)
echo "[continuation] exporting ONNX..." | tee -a outputs/logs/test_B_continuation.log
.venv/bin/python main.py export --arch cnn >> outputs/logs/test_B_continuation.log 2>&1

# Backup the Test B classifier
mkdir -p outputs/backups/test_B
cp outputs/models/cnn/classifier.onnx outputs/backups/test_B/classifier.onnx
echo "[continuation] Test B classifier backed up to outputs/backups/test_B/" \
  | tee -a outputs/logs/test_B_continuation.log

# Realworld eval (uses outputs/models/cnn/classifier.onnx)
echo "[continuation] realworld eval..." | tee -a outputs/logs/test_B_continuation.log
.venv/bin/python realworld_eval.py >> outputs/logs/test_B_continuation.log 2>&1

# Diagnose (frozen test)
echo "[continuation] diagnose (frozen test)..." | tee -a outputs/logs/test_B_continuation.log
.venv/bin/python diagnose.py --arch cnn --version test_B >> outputs/logs/test_B_continuation.log 2>&1

echo "[continuation] ALL DONE at $(date)" | tee -a outputs/logs/test_B_continuation.log
touch outputs/logs/test_B_DONE
