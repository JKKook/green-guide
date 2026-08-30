# 루트 계획 문서 지도 — 세대·모순 정리

> 소스: 루트 `*.md` 7종 + `docs/greenguide_model_methods.html` (2026-08-13 탐색)

시간축: SMART_CAPTURE(05-30) → DIAGNOSIS(05-24) → **BLUEPRINT v2**(07-07) → CAM_MATERIAL(07-13) → SEMANTIC_FUSION(07-15) → **ACCURACY_LATENCY v2**(07-21) → UIUX_SPEC(07-28) → **MODEL_METHODS**(08-06). **뒤 문서가 앞 문서를 뒤집는다 — 최신 사실 = MODEL_METHODS + ACCURACY_LATENCY §0.**

| 문서 | 성격 | 핵심 | 현행성 |
|---|---|---|---|
| `docs/plans/GREENGUIDE_BLUEPRINT.md` | 마스터 청사진 | flat 폐기 → 2단 계층([hier-taxonomy](hier-taxonomy.md)), 데이터-게이트 자동 세분화, KPI(실사용 대분류 ≥85%), 로드맵 P0~P5 | 허브. 로드맵 부분은 ACCURACY_LATENCY가 대체 |
| `docs/plans/DIAGNOSIS_PROCESS.md` | 프로세스 명세 | frozen test 동결·진단 엔진·안전 게이트·etc 큐 → [retrain-loop](retrain-loop.md) | 게이트 임계는 이후 2회 개정됨 |
| `docs/plans/SMART_CAPTURE_STRATEGY.md` | 촬영 개입 전략 | 품질 게이트 A~F, 갭 진단(frozen 95.9% vs 실사용 63.4%) | A/E/F는 앱에 구현, **B(u2 크롭)는 폐기** |
| `docs/plans/CAM_MATERIAL_UPGRADE_PLAN.md` | 부속 트랙 | 448² 고해상 CAM, zoom-and-verify → [semantic-fusion](semantic-fusion.md) | Stage 1 완료, 2·3 미착수 |
| `docs/plans/SEMANTIC_FUSION_PLAN.md` | 부속 트랙 | OCR·CLIP·CAM log-linear 융합 + 실측 | Phase 1·2 배포, 51장 수치는 사후 무효 |
| `docs/plans/ACCURACY_LATENCY_BLUEPRINT.md` | 청사진 v2 (대체) | **§0 = 수치 SSOT.** 정확도×지연 두 축 재설계, B1+B2+B4로 15s→1.6s | 현행. A1(평가 표본)이 선결 과제 |
| `docs/design/GREENGUIDE_UIUX_SPEC.md` | as-built 명세 | 앱 화면 7·위젯 8·토큰 v1.1.0 실측 → [waste-app](waste-app.md) | 현행 스냅샷 |
| `docs/greenguide_model_methods.html` (+PDF) | **최종 정답지** | v1.0(08-06), 대외 기술 문서. 전 계획의 채택/기각 확정 | **최신 사실 기준** |

## 문서 간 모순 5건 (인용 시 주의)

1. **실사용 51장 지표**: SEMANTIC_FUSION 72.5% → ACCURACY_LATENCY "31장 오염, 무효" → MODEL_METHODS는 정직 서브셋 n=18 분리 집계. 최신이 정답.
2. **클래스 수**: SMART_CAPTURE 12(flat) → BLUEPRINT 12+2 설계 → 구현은 **14×25**.
3. **온디바이스 모델**: BLUEPRINT는 별도 coarse 전용 ONNX 계획 → 실제는 **단일 hier 가중치 공유 + OTA**.
4. **u2netp 자동 크롭**: SMART_CAPTURE 최대 잠재력(+5~10pp) → 최종 기각(장면 경로), 탭은 GrabCut.
5. **DINOv2 앙상블**: BLUEPRINT·SEMANTIC_FUSION 상시 구성 → B4 이후 기본 비활성(W=0).

채택/기각 전체 대장: [model-versions-accuracy](model-versions-accuracy.md).
