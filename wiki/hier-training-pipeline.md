# 계층 학습 파이프라인

> 소스: `waste-classifier/src/hier_*.py`, `waste-classifier/HIER_TRAINING_GUIDE.md` (2026-08-13 탐색)

[hier-taxonomy](hier-taxonomy.md)의 fine 25클래스를 학습하는 현행 파이프라인. 정본 문서는 `HIER_TRAINING_GUIDE.md`(2026-07).

## 데이터 소스

| 소스 | 규모 | 채우는 클래스 |
|---|---|---|
| 구 manifest (Kaggle + AI-Hub 71362/140 + TACO) | ~7.5만 | legacy 13라벨 (coarse/fine 감독 혼합) |
| AI-Hub 71385 bbox 크롭 | ~11만 | 신규 16라벨 + 조건(clean/multi/dirty) |
| AI-Hub 140 품목 | 1.3만 | light_bulb, glass_deposit, electronics |
| 다중객체 합성 (`synthesize_multiobject.py`) | 8,000 | 이웃 파편 + carton↔유리 하드네거티브 |
| 사용자 피드백 | 51 | **학습 제외 — 순수 평가 전용** (오염 사고 후 `user_*` 영구 제외) |

## 핵심 설계

- **분할 (`build_hier_splits`)**: 경로 기반 `paths_v2` — frozen test는 `source_path`로 영구 동결(`hier_frozen_test.json`), 합성(`synmo_*`)은 test 진입 금지, 나머지는 감독그룹별 stratified. v3 사고 교훈: 학습 중 데이터 추가 → 인덱스 밀림 → 게이트 롤백 → "사이클 중 fine-staging 수정 금지".
- **전처리**: PIL→RGB→224² bilinear→ImageNet 정규화. **학습/서버/온디바이스 3곳 동일 수식**(train-serve skew 제거, 테스트로 고정).
- **증강**: 자체 텐서 연산 — hflip 50%, grayscale 20%(색편향 억제), color jitter 70%, ROT90. 회전·크롭 증강 없음(입력이 이미 bbox 크롭) — 기하 다양성은 합성이 담당.
- **손실 (`HierarchicalLoss`)**: fine 아이템 CE + coarse 아이템 `-log Σ_children softmax` (rollup loss). 가중치 inverse-freq에 **median×4 cap**(극소클래스 붕괴 방지).
- **학습**: batch 32, ≤15 epoch, Adam 1e-4, patience 4, seed 42, MPS. 체크포인트 선택 = `val_coarse_acc + 0.2×val_fine_acc` (대분류 우선 원칙 내장).
- **평가 (`evaluate_hier`)**: 대분류 acc(롤업) / 세부 acc(fine 감독 test만) / guidance-safe f1. 활성화 판정 `support≥30 AND (f1≥0.80 OR gs_f1≥0.85)`.
- **export (`hier_export`)**: opset 17, 3-output ONNX(logits, cam, embedding) + `taxonomy.json` 사이드카(게이트 임계 포함). CAM은 1×1 conv 재구성(INT8 호환).
- **OOD prototype**: `build_hier_prototypes.py` — 클래스별 평균 임베딩 + val 97.5퍼센타일 τ → `ood.npz`. → [ood-openset](ood-openset.md)

백본 선택은 env `WASTE_HIER_BACKBONE` (현행 배포 resnet50). 실행: `python -m src.hier_train`, 전체 사이클은 [retrain-loop](retrain-loop.md).

## 검증된 기법 효과 (가이드 §5-5 실측)

Transfer learning MLP 35%→CNN 92% · rollup loss로 세부미상 2.3만 재활용 · inverse-freq+cap · guidance-safe로 paper_cup/glass_clear 구제 · 회귀게이트+자동롤백 · OOD 2단 reject · **non_object 마스킹 실사용 +5.9pp** · ONNX 등가성 <1e-4.
