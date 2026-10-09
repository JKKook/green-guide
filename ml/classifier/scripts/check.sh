#!/usr/bin/env bash
# 커밋 전 검증 한 번에: ruff → pytest. 실패 시 즉시 중단.
# 사용: ml/classifier 에서 scripts/check.sh   (Rosetta 셸이면 PY="arch -arm64 .venv/bin/python" scripts/check.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
echo "== ruff =="; $PY -m ruff check .
echo "== pytest =="; $PY -m pytest -q "$@"
