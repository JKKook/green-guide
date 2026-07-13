#!/bin/bash
# 운영 배포 준비 — 계층 모델 아티팩트를 waste-api 이미지 번들 위치로 복사.
# (HF Spaces 푸시는 별도 사용자 확인 후: git add/commit/push)
#
# 서버 코드의 경로 규약:
#   hier_inference._CANDIDATES → models/classifier_hier.onnx (+ 같은 폴더의
#   taxonomy.json, ood.npz). Dockerfile 이 models/ 를 통째로 COPY 하므로
#   여기 복사해두면 이미지에 포함된다.
set -euo pipefail
cd "$(dirname "$0")"

SRC=../waste-classifier/outputs/models/cnn_hier
for f in classifier.onnx taxonomy.json ood.npz; do
  [ -f "$SRC/$f" ] || { echo "누락: $SRC/$f"; exit 1; }
done

cp "$SRC/classifier.onnx" models/classifier_hier.onnx
cp "$SRC/taxonomy.json"   models/taxonomy.json
cp "$SRC/ood.npz"         models/ood.npz

echo "번들 완료:"
ls -lh models/classifier_hier.onnx models/taxonomy.json models/ood.npz
echo
echo "다음 단계 (사용자 확인 후 수동):"
echo "  1) 로컬 검증: WASTE_API_HIER_MODEL_PATH 미설정 상태로 서버 기동 → /taxonomy 200 확인"
echo "  2) git add models/ src/ && git commit && git push  (HF Spaces 자동 재배포)"
echo "  3) (선택) scripts/publish_hier_version.py --apply  (model_versions 활성화)"
