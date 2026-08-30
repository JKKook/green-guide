# Wiki Index — waste (그린가이드 AI)

시작점: [architecture-overview](architecture-overview.md). 수치 인용 규칙: [planning-docs](planning-docs.md)의 "최신 사실 = MODEL_METHODS + ACCURACY_LATENCY §0" 참조.

## 개관
- [architecture-overview](architecture-overview.md) — 4 서브프로젝트 파이프라인, 이원 추론, 앱↔서버 계약 7가지, 인프라
- [planning-docs](planning-docs.md) — 루트 계획 문서 8종의 세대·역할·모순 5건 정리

## 서브프로젝트
- [waste-preprocessor](waste-preprocessor.md) — 수집·정제·manifest. raw-direct 모드, fine-staging 분리 이유
- [waste-classifier](waste-classifier.md) — 학습·ONNX export 레포. 문서 3종 세대, 스크립트 지도, 클래스 추가 절차
- [waste-api](waste-api.md) — FastAPI 추론 서버(HF Spaces). 엔드포인트 전량, /predict-hier 캐스케이드, VLM 폴백, 배출 스트림
- [waste-app](waste-app.md) — Flutter 앱(Android). setState 단일, OTA, 추론 라우팅, Trust UI

## 모델·학습
- [hier-taxonomy](hier-taxonomy.md) — 대분류 14 × 세부 25, 롤업, guidance 그룹, 데이터-게이트 활성화
- [hier-training-pipeline](hier-training-pipeline.md) — 데이터 소스, paths_v2 분할, rollup loss, export
- [retrain-loop](retrain-loop.md) — 재학습 사이클, 안전 게이트(개정사), etc 큐, 평가 오염 사고와 교훈
- [ood-openset](ood-openset.md) — prototype τ 거부, etc HDBSCAN pseudo-class
- [semantic-fusion](semantic-fusion.md) — OCR·CLIP·CAM log-linear 융합, CAM 다중재질 트랙
- [model-versions-accuracy](model-versions-accuracy.md) — **수치 SSOT 요약**: 활성 버전, frozen 96.4%/홀드아웃 55%, 지연 1.6s, 채택/기각 대장
- [ondevice-vs-cloud-models](ondevice-vs-cloud-models.md) — 배포 형태별 모델 인벤토리·학습데이터 구분 (동일 가중치 공유 원칙)

## 데이터·인프라
- [dataset-staging](dataset-staging.md) — AI-Hub 71385/140, TACO, Open Images, 합성 스테이징 현황표
- [data-experiments](data-experiments.md) — AI-Hub paper 가설 기각, 증강 A/B(C1 미채택), 교훈
- [supabase-infra](supabase-infra.md) — 테이블/버킷, 2026-07 쿼터 사태와 이사, 지속성 장치, local_feedback
