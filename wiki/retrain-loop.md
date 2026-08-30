# 피드백 재학습 루프 + 안전 게이트

> 소스: `waste-classifier/retrain_hier.py`, `retrain.py`, `diagnose.py`, `docs/plans/DIAGNOSIS_PROCESS.md` (2026-08-13 탐색)

"더 많은 데이터 = 더 나음"이 아님을 코드로 강제하는 자동 사이클. 정확도를 버전 간 비교 가능한 숫자로 관리하고 나쁜 모델의 승격을 차단한다.

## 현행 사이클 (`retrain_hier.py`)

피드백 수집(Supabase `user_uploads` confirmed/corrected) → 소수클래스 격리(<6장 quarantine) → 백업 → preprocessor 재실행 → splits 재생성(frozen 유지) → `src.hier_train` → `src.hier_evaluate` → **게이트** → PASS: export + prototype 재빌드 + 활성화 자동 승격/강등 + `model_diagnostics` 기록 + 실사용 평가 자동 실행 / FAIL: **실패 산출물을 `outputs/failed_cycles/`에 보존 후** 백업 롤백 + exit 1.

- publish는 `scripts/publish_hier_version.py --apply` 수동 (`--publish` 플래그는 미구현 안내문만).
- baseline은 `hier_history.jsonl` 마지막 줄 — **게이트 통과 버전만 이력에 커밋**(실패가 baseline 오염 못 함).

## 게이트 상수 (2026-07-21 개정판)

```
GATE_MAX_COARSE_ACC_DROP    = 0.02   # 대분류 acc -2pp 초과 → FAIL
GATE_MAX_COARSE_RECALL_DROP = 0.05   # 대분류별 recall -5pp 초과 → FAIL
GATE_RECALL_MIN_SUPPORT     = 50     # 소표본 거부권 차단
GATE_GUIDANCE_EQUIV = {etc↔trash}    # 안내 동일 이동은 회귀로 안 셈
```

개정 사유: v8~v11 3연속 FAIL의 실체가 31표본 etc→trash 이동(안내 동일)이었음. 그럼에도 v11은 `etc recall 0.742→0.161` 실질 회귀로 FAIL → **현재 활성 아티팩트는 2026-07-15 계보**([model-versions-accuracy](model-versions-accuracy.md)).

## etc open-set 처리 (`etc_queue.py`)

etc 피드백 30건 누적 시 retrain 초입 자동 실행. Stage 1: 임베딩 prototype 거리 **+** softmax 병행(softmax 단독 금지 — OOD 과신) 기존 클래스 재배정. Stage 2: 잔여 HDBSCAN 군집 → 숨김 pseudo-class `etc_auto_*`(active=false). **사람이 하는 유일한 일 = 이름·배출법 입력 후 active=true 승격.** → [ood-openset](ood-openset.md)

## 평가 무결성 (사고와 교훈)

- **피드백 오염 사고**: 구 retrain이 피드백을 raw로 다운로드해 학습에 흡수 → 실사용 51장 중 31장이 train에 들어가 "76.5%"가 암기 수치로 판명(무효). 교정: `hier_dataset`에서 `user_*` 영구 제외 → **피드백 = 순수 평가 전용**.
- frozen test 도입 전 학습된 구 모델의 98%대 수치도 누수로 부풀려진 값(정직한 값 96%대).
- 오프라인 평가와 서빙 API 경로를 **같은 조건으로** 맞출 것 — 경로가 달라 회귀가 숨었던 전례([semantic-fusion](semantic-fusion.md)).

## 트리거·모니터

`feedback_monitor.py`(READ-ONLY): 수집량, 재학습 준비도, 드리프트(confidence entropy). `RETRAIN_TRIGGER_NEW=100`.
