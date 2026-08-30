# 온디바이스 vs 클라우드 — 모델·학습데이터 구분

> 소스: `waste-api/models/`, `waste-api/src/api.py`, `waste_app/lib/services/{local_inference,remote_model_service,prediction_service}.dart`, `waste-classifier/HIER_TRAINING_GUIDE.md` §2, `docs/greenguide_model_methods.html` §1·§2·§6 (2026-08-13 질의로 생성)

핵심 원칙: **두 경로는 동일한 분류기 가중치를 공유한다.** 차이는 모델이 아니라 *파이프라인*(보조 모델·TTA·융합·VLM 유무)이다. 따라서 "온디바이스 전용 학습데이터"는 존재하지 않는다 — 학습은 전부 [waste-classifier](waste-classifier.md)에서 1회 이루어지고, 산출 ONNX가 양쪽에 배포된다.

## 클라우드 (waste-api, HF Spaces) — 모델 인벤토리

| 모델 | 크기 | 역할 | 학습 주체/데이터 |
|---|---|---|---|
| `classifier_hier.onnx` (ResNet50) | 90MB | **메인 계층 분류기** fine 25→coarse 14 롤업 | **자체 학습** — 아래 "학습데이터" 절 |
| `stage1_binary.onnx` (MobileNetV3-S) | 6MB | 폐기물/비폐기물 이진 게이트 | **자체 학습** (`scripts/train_stage1_binary.py`) |
| MediaPipe Hands | — | 손 면적 ≥0.5 → non_object | 사전학습 (Google, 프로젝트 데이터 무관) |
| `u2netp.onnx` | 4.4MB | saliency 누끼·탭 크롭 | 사전학습 (프로젝트 미학습) |
| CLIP 이미지 인코더 INT8 + `clip_concepts.npz` | 84MB | 제로샷 정체 prior (탭 경로만) | 사전학습 + 66컨셉 임베딩만 자체 사전계산 |
| RapidOCR det/rec (PP-OCRv5 korean) | 22MB | 텍스트 증거 (조건부) | 사전학습 |
| DINOv2 hier head | 84MB | second opinion — **기본 비활성**(W=0) | head만 자체 학습, 백본 사전학습 |
| Claude Haiku 4.5 (API) | — | VLM 폴백 (일 200회 캡) | 외부 API — 학습 무관 |
| flat color/edge ONNX (레거시) | 43MB×2 | `/predict` 구경로 | 자체 학습 (구 6~13클래스 세대) |

## 온디바이스 (waste_app, Flutter) — 모델

| 모델 | 전달 방식 | 비고 |
|---|---|---|
| `classifier_hier.onnx` — **클라우드와 동일 파일** | **OTA**: `/model/latest` → HF Hub `ethanDev92/waste-models` 다운로드 + sha256 검증 (번들 에셋 없음, 앱 204→88MB) | taxonomy.json 사이드카로 게이트·롤업을 온디바이스에서도 동일 적용 |
| (보조 모델 전부 없음) | — | u2netp·Stage1·CLIP·OCR·VLM·TTA·non_object 마스킹 미탑재 — **단일 패스** |

⚠️ BLUEPRINT의 "온디바이스 전용 경량 coarse 모델" 계획은 **미채택** — 단일 hier 가중치 공유로 귀결 ([planning-docs](planning-docs.md) 모순 3).

라우팅: 기본 클라우드. 온디바이스 모드에서 confidence <0.60이면 클라우드 재판정 폴백 ([waste-app](waste-app.md)). 실측(n=18 정직 서브셋)상 대분류 정답률은 양쪽 동률 — 클라우드의 가치는 보조 파이프라인(설명가능성·복합 장면) ([model-versions-accuracy](model-versions-accuracy.md) §7).

## 메인 분류기의 학습데이터 (양쪽 공통)

[hier-training-pipeline](hier-training-pipeline.md) 참조. 요약:

| 소스 | 규모 | 비고 |
|---|---|---|
| 구 manifest: Kaggle Garbage + AI-Hub 71362/140 + TACO | ~7.5만 | legacy 13라벨, coarse/fine 감독 혼합 |
| AI-Hub 71385 bbox 크롭 | ~11만 | 신규 세부 16라벨 ([dataset-staging](dataset-staging.md)) |
| AI-Hub 140 품목 | 1.3만 | light_bulb·glass_deposit·electronics |
| 다중객체 합성 `synmo_*` | 8천 | frozen test 진입 금지 |
| 사용자 피드백 `user_*` | 51 | **학습 영구 제외 — 순수 평가 전용** ([retrain-loop](retrain-loop.md)) |

MODEL_METHODS §4 주석: Kaggle류 스튜디오 데이터는 "노이즈 부재로 기각" 방향 — 현행 주력은 AI-Hub 71385 계층 크롭 + TACO 실외 + (평가용) 피드백.

## 서브프로젝트별 역할 한 줄

- [waste-preprocessor](waste-preprocessor.md): 위 학습데이터의 물리 보관처(`data/raw/` + `fine-staging/`)와 manifest. 모델 없음.
- [waste-classifier](waste-classifier.md): 유일한 학습 주체. hier·stage1·DINOv2 head·CLIP 컨셉·prototype 전부 여기서 산출.
- [waste-api](waste-api.md): 클라우드 서빙 — 위 표 전체를 조합한 캐스케이드.
- [waste-app](waste-app.md): 온디바이스 서빙 — hier 단일 패스 + OTA.
