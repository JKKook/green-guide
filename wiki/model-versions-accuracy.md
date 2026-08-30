# 모델 버전 · 정확도 현황

> 소스: `waste-classifier/outputs/`, `docs/plans/ACCURACY_LATENCY_BLUEPRINT.md` §0, `docs/greenguide_model_methods.html` §4·§7 (2026-08-13 탐색)

**수치의 단일 출처(SSOT)는 ACCURACY_LATENCY_BLUEPRINT §0 + MODEL_METHODS(2026-08-06).** 문서마다 실사용 수치가 다른 이유는 평가 오염 사고([retrain-loop](retrain-loop.md)) 때문 — 51장 중 31장이 train 오염이라 51장 지표는 무효, 정직한 홀드아웃은 n=18~20.

## 활성 아티팩트 (2026-07-15 계보, hier_v2)

- `outputs/models/cnn_hier/` — classifier.onnx 94MB (**ResNet50**), taxonomy.json, ood.npz. 레지스트리 v20260805_152341, 서버 facc879, 앱 88MB 슬림.
- 이력(게이트 통과분만): v20260713 coarse .9478 → v20260714 .9658 → **v20260715 .9637 / fine .9315 (현 baseline)**. v11(7/21)은 etc recall 붕괴로 FAIL → failed_cycles 보존.

## 정확도 좌표 (정직한 값)

| 지표 | 값 | 신뢰도 |
|---|---|---|
| frozen test 33.1k — 대분류 | **96.4%** | 높음 (그룹-어웨어 동결) |
| frozen test — 세부 | 92.7~93.2% | 높음 |
| 실사용 순수 홀드아웃 n=20 | **55%** | 표본 부족 (±20pp) |
| (무효) 실사용 51장 | 64.7~76.5% | 31장 train 오염 — 인용 금지 |

**frozen↔실사용 갭 ~30pp+가 최대 이슈.** 원인 구조: ① 극소 클래스(etc 131, trash_other 550, food_waste 661, cardboard 678 train — 목표 각 3,000+) ② 실내·손·잡배경 분포 부재(학습이 시설/스튜디오/bbox 크롭) ③ etc 캐치올의 본질적 모호함 ④ carton↔유리 역혼동. **n=20으로는 어떤 학습 레버도 판정 불가 → 정확도 트랙의 선결은 평가 표본 확보(A1).**

## 지연 좌표 (/predict-hier)

최적화 전 로컬 ~1.0s / 운영(HF Spaces 무료 CPU) ~15s → **B1(조건부 OCR)+B2(EXIF 기반 TTA 축소)+B4(DINOv2 제거) 적용 후 ~0.5s / ~1.6s (9.4× 단축, 예산 p50≤2s 달성)**. 온디바이스 0.1–0.3s.

## 채택/기각 결정 대장

| 레버 | 판정 | 근거 |
|---|---|---|
| 회전 TTA (EXIF 축소) | ✅ 채택 | AI-Hub 크롭의 방향 분포 어긋남 흡수 |
| non_object 마스킹 | ✅ 채택 | 실사용 +5.9pp |
| 조건부 OCR (확신≥0.75 스킵) | ✅ 채택 | -2~4s, 플립 0건 |
| VLM 폴백 (Claude Haiku) | ✅ 구현 | reject·저확신·증거-불일치만 위임 |
| **INT8 양자화** | ❌ 기각 | 3변형 모두 정확도 -1~-5pp, Apple 로컬 속도 역효과 |
| **장면 u2netp 자동 크롭** | ❌ 기각 | v6+TTA에서 역전(39→25). 탭 경로는 GrabCut |
| **DINOv2 앙상블** | ❌ 기본 비활성 | 순수 홀드아웃 기여 0 (이전 +2는 오염 암기) |
| CLIP 장면(풀프레임) 적용 | ❌ 기각 | 정체 오인 역효과. 탭 크롭만 +4 |
| 합성 실내 배경 (v8·v9) | ❌ 폐기 | 배경 편향 |
| 외부 실데이터 1.2만 (v10·v11) | ⏸ 미채택 | frozen 무손실이나 홀드아웃 동률 — n 부족으로 판정 불가 |

## 온디바이스 vs 클라우드 (2026-08-06, §7)

정직 서브셋 n=18에서 대분류 정답률 **동률**. 클라우드 겉보기 열세는 보수적 reject 정책 탓(의도된 동작). 결론: 속도·오프라인=온디바이스, 설명가능성·복합 장면=클라우드, 현행 기본값(클라우드+저확신 교차 폴백) 타당.

관련: [hier-training-pipeline](hier-training-pipeline.md) · [semantic-fusion](semantic-fusion.md) · [planning-docs](planning-docs.md)
