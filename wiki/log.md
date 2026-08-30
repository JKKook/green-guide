# Wiki Log (append-only)

## [2026-08-13] init | 위키 초기화
- schema.md / index.md / log.md 생성. `/llm-wiki init` 없이 첫 ingest 요청과 함께 부트스트랩.

## [2026-08-13] ingest | waste 프로젝트 전체(하위 레포 포함) 초기 적재
- 소스: 루트 계획 문서 8종(BLUEPRINT, ACCURACY_LATENCY, SEMANTIC_FUSION, CAM_MATERIAL, DIAGNOSIS, SMART_CAPTURE, UIUX_SPEC, docs/greenguide_model_methods.html) + 4 서브프로젝트(waste-preprocessor / waste-classifier / waste-api / waste_app) + 스테이징 6종 + bin/.
- 병렬 탐색 에이전트 4개 결과를 종합해 페이지 15개 생성: architecture-overview, planning-docs, waste-preprocessor, waste-classifier, waste-api, waste-app, hier-taxonomy, hier-training-pipeline, retrain-loop, ood-openset, semantic-fusion, model-versions-accuracy, dataset-staging, data-experiments, supabase-infra.
- 핵심 시사점: ① 문서 세대차 큼 — 수치 SSOT는 MODEL_METHODS(08-06)+ACCURACY_LATENCY §0 ② 실사용 51장 지표는 train 오염으로 무효(정직 홀드아웃 n=20, 55%) ③ frozen 96.4% vs 실사용 갭 ~30pp가 최대 이슈, 선결은 평가 표본 확보(A1) ④ README류(classifier/api/app)는 V1 시점이라 낡음 — 코드가 SSOT.

## [2026-08-13] query | 온디바이스 vs 클라우드 모델·학습데이터 구분
- 질의 답변을 ondevice-vs-cloud-models.md 로 재적재. 핵심: 두 경로는 동일 분류기 가중치 공유(OTA), 차이는 클라우드 전용 보조 파이프라인. 학습데이터는 waste-classifier 단일 출처.
