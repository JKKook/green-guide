# waste-classifier — 학습·ONNX export 레포

> 소스: `waste-classifier/README.md`, `waste-classifier/HIER_TRAINING_GUIDE.md`, `waste-classifier/lab.md`, `waste-classifier/src/`, `waste-classifier/scripts/` (2026-08-13 탐색)

GreenGuide AI 파이프라인의 학습 전담 서브프로젝트. **폐기물 이미지 분류 모델 학습 + ONNX export**를 담당하며, 서빙은 [waste-api](waste-api.md)(`/predict-hier`), 온디바이스는 [waste_app](waste-app.md)(onnxruntime)이 맡는다. git 저장소 아님 — 버전 이력은 `outputs/logs/diagnosis/*.jsonl`과 백업 폴더가 대신한다.

전체 흐름: `waste-preprocessor(수집·전처리) → waste-classifier(학습·ONNX) → waste-api(추론) → Flutter 앱`. → [architecture-overview](architecture-overview.md)

## 문서 3종 — 세대가 다름 (주의)

| 문서 | 시점 | 역할 | 현행성 |
|---|---|---|---|
| `README.md` | 2026-05 | 1세대: 6클래스 flat MLP vs CNN, CLI, 트러블슈팅 | 대부분 **구식** |
| `lab.md` | 2026-05 | 교육용 트레이스(샘플 1개가 manifest→ONNX까지 가는 전 과정) | 구 6클래스 기준, 원리는 유효 |
| `HIER_TRAINING_GUIDE.md` | 2026-07 | **현행 정본**: 계층 CNN 학습 전 과정 | 현재 기준 문서 |

## 모델 아키텍처

- **현행**: `src/model.py::build_hier_model()` — resnet18 / convnext_tiny / resnet50 선택(env `WASTE_HIER_BACKBONE`). **배포 아티팩트는 resnet50** (ONNX 94MB). 출력은 **fine 25클래스 단일 head**, 대분류는 확률 롤업 `P(coarse) = Σ P(fine children)`.
- CAM export 래퍼: `(logits, cam, embedding)` 3-output. CAM은 `fc.weight`를 1×1 conv로 적용(INT8 양자화 호환), embedding은 GAP 512d(OOD reject용 → [ood-openset](ood-openset.md)).
- 레거시 flat: MLP(39.84% — baseline 유물) / ResNet18 CNN(92.35%) / cnn_edge(Sobel 앙상블 실험). ConvNeXt는 MPS에서 ~40배 느려 실사용 제외.
- 실험 백본: DINOv2 head, CLIP concepts(→ [semantic-fusion](semantic-fusion.md)), Stage1 binary(6MB, 캐스케이드 전단).

분류 체계 상세: [hier-taxonomy](hier-taxonomy.md) · 학습 파이프라인: [hier-training-pipeline](hier-training-pipeline.md) · 재학습 루프: [retrain-loop](retrain-loop.md) · 성능/버전: [model-versions-accuracy](model-versions-accuracy.md)

## 주요 스크립트

| 파일 | 역할 |
|---|---|
| `main.py` | flat CLI (계층 학습은 `python -m src.hier_train`) |
| `retrain_hier.py` | **현행 재학습 루프** — [retrain-loop](retrain-loop.md) |
| `diagnose.py` | per-class 진단 + 회귀 게이트 + Supabase `model_diagnostics` 기록 |
| `etc_queue.py` | open-set 2단계: prototype 재배정 → HDBSCAN 군집 → 숨김 pseudo-class `etc_auto_*` |
| `feedback_monitor.py` | Supabase READ-ONLY 피드백 모니터 (`RETRAIN_TRIGGER_NEW=100`) |
| `scripts/realworld_eval_hier.py` | 실사용 평가 (user_uploads GT) — [model-versions-accuracy](model-versions-accuracy.md) |
| `scripts/publish_hier_version.py --apply` | hier 모델 게시 (retrain_hier `--publish`는 미구현, 수동) |
| `scripts/` 35개 | 데이터 통합·합성·품질감사·대체백본·운영 배치 |

## 클래스 추가 체크리스트 (가이드 §7)

fine-staging에 크롭 적재 → `src/taxonomy.py` 갱신 → 안내 동일 형제면 `GUIDANCE_GROUPS` → migration으로 `waste_classes` level=2 시드(active=false) → `pytest tests/test_hierarchy.py` → `retrain_hier.py` 1사이클(활성화 자동 판정).

## 알려진 이슈

핵심 이슈는 [model-versions-accuracy](model-versions-accuracy.md)의 실사용 갭 참조. 그 외:
- README/lab.md 문서 드리프트 (flat 6클래스 기준)
- `plastic_other` 학습 데이터 0 (설계상 롤업 전용)
- 학습 사이클 중 fine-staging 수정 금지 (v3 인덱스 밀림 사고 교훈; splits는 paths_v2로 완화)
- 모델 경량화(int8/MobileNetV3), 실험 추적 도구(wandb 등) 부재
