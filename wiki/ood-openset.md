# OOD 거부 · open-set 처리

> 소스: `waste-classifier/src/ood.py`, `scripts/build_hier_prototypes.py`, `etc_queue.py`, `docs/plans/DIAGNOSIS_PROCESS.md` §⑤ (2026-08-13 탐색)

원칙: **"softmax는 최선만 고르고, 임베딩 거리는 닮았는가를 본다"** — 신경망은 OOD 입력에 과신하므로(노이즈→clothes 0.999 사례) softmax 단독 판정 금지.

## Prototype 기반 거부

- 학습 후 `build_hier_prototypes.py`: 클래스별 L2정규화 평균 embedding = prototype, val 최근접 cosine distance **97.5퍼센타일 = τ** → `ood.npz`로 모델과 함께 배포.
- 서빙([waste-api](waste-api.md) `/predict-hier`): 2단 reject — τ_soft 0.30 / τ_hard 0.40. 탭 경로는 완화(`ood_relax` — 사용자가 지목한 영역이므로).
- 캐스케이드의 다른 거부 장치와 협업: MediaPipe 손감지(면적≥0.5), Stage1 이진 게이트, non_object 마스킹, 신뢰도 게이트([hier-taxonomy](hier-taxonomy.md)).

## etc 캐치올 자동 처리 (`etc_queue.py`)

etc 피드백 30건 누적 시 2단계: ① prototype 거리+softmax 병행으로 기존 클래스 재배정 ② 잔여를 HDBSCAN 군집 → 숨김 pseudo-class `etc_auto_*`(active=false) 등록, noise는 etc 유지. 사람은 이름·배출법만 지어 승격. [retrain-loop](retrain-loop.md)의 게이트가 pseudo-class 부작용 방어.

## 한계

etc는 open-set의 구조적 모호함 그 자체 — frozen recall .742(최저), 실사용 오답의 최다 진원지. v8~v11 게이트 반복 교란의 원인이기도 함 ([model-versions-accuracy](model-versions-accuracy.md)).
