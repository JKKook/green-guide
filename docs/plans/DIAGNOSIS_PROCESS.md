# 모델 진단 프로세스 (GreenGuide)

재학습마다 자동으로 도는 **반복 가능한 진단 + 안전 게이트 + etc 자동 처리** 파이프라인.
정확도를 "느낌"이 아니라 **버전 간 비교 가능한 숫자**로 관리하고, 나쁜 모델이
프로덕션에 올라가는 걸 막고, 분류 불가(etc) 데이터를 자동으로 정리한다.

> 위치: `waste-classifier/` (학습 레포)
> 관련 문서: [ADDING_A_NEW_CLASS.md](ADDING_A_NEW_CLASS.md)

---

## 구성 요소

### ① 고정 held-out test set — `src/frozen_test.py`
- test 멤버를 manifest **위치 인덱스가 아니라 `source_path`(안정 키)로 동결**.
  → `data/splits/frozen_test.json`
- 한 번 test 에 들어간 이미지는 계속 test, **신규 데이터는 train/val 로만** 들어감.
  → 버전이 바뀌어도 **같은 잣대**로 정확도를 비교 (회귀 감지의 토대).
- 작은 클래스는 절반만 test 로 (학습용 샘플 보존). `train.py` 가 이걸 사용.

### ② 진단 엔진 — `diagnose.py`
고정 test 로 다음을 산출:
- per-class **precision / recall / f1** + **혼동행렬**
- **혼동 쌍 경보**: off-diagonal ≥ source 클래스의 3% & ≥5건 (예: `paper→vinyl`)
- **약한 클래스**: f1 < 0.85
- **needs-data**: 약한 클래스 + 혼동쌍 source → "데이터 보강 우선순위"
- **회귀 감지**: 직전 버전 대비 클래스별 recall 하락
- **PASS/FAIL 게이트** (아래 ④)

```bash
python diagnose.py --arch cnn --version v20260524_025122   # 단독 실행
python diagnose.py --no-supabase                            # Supabase 기록 생략
```

### ③ 저장 (이중)
- **레포**: `outputs/logs/diagnosis/<version>.json` (상세) + `history.jsonl` (회귀 비교 baseline)
- **Supabase**: `model_diagnostics` 테이블 ([migration 004](waste-classifier/migrations/004_model_diagnostics.sql))
- **게이트 통과 버전만** 이력에 커밋 → 실패 시도가 baseline 을 오염시키지 않음.

### ④ retrain 안전 게이트 — `retrain.py`
```
학습 → [진단] → PASS: trained 갱신 + publish/activate
              → FAIL: 백업에서 자동 롤백 + publish 취소 + exit 1
```
- **FAIL 임계**: 전체 정확도 **-2pp** 초과 하락 OR 기존 클래스 recall **-5pp** 초과 하락.
- 진단과 publish 가 **같은 version 태그** 공유.
- "더 많은 데이터 = 더 나음"이 아님(과거 사례)을 코드로 강제.

### ⑤ etc 큐 자동 처리 (open-set 2단계) — `etc_queue.py`
etc('기타/분류 불가') 피드백이 **30건** 쌓이면 retrain 초입에서 자동 실행:

1. **Stage 1 — 기존 클래스 재배정**
   etc 이미지를 현 모델로 다시 보고, **임베딩이 기존 클래스 prototype 에 가깝고
   + softmax 도 확신**하면 그 클래스로 재배정 (애매해서 '기타' 누른 케이스 회수).
   ※ 신경망은 OOD 에 과신 → softmax 단독이 아니라 **임베딩 거리도 함께** 판정.
2. **Stage 2 — 신규 클래스 후보**
   어디에도 안 붙는 것들끼리 **HDBSCAN 클러스터** → 뭉치면 **숨김 pseudo-class**
   (`slug=etc_auto_*`, `active=false`), 흩어진 noise 는 etc 유지.
3. 이후 정상 retrain → ④게이트. pseudo-class 가 기존 클래스를 망치면 자동 롤백.

**사람이 하는 단 한 가지**: 픽셀 군집에서 올바른 한국어 이름·배출법은 자동 생성
불가 → pseudo-class 는 `active=false`(사용자에게 숨김)로 두고, `etc_clusters`
리뷰 테이블([migration 005](waste-classifier/migrations/005_etc_clusters.sql))을 보고
운영자가 이름·배출법을 넣어 `active=true` 로 승격 (→ [ADDING_A_NEW_CLASS.md](ADDING_A_NEW_CLASS.md)).

```bash
python etc_queue.py            # dry-run (분석만, 변경 없음)
python etc_queue.py --apply    # 실제 적용
```

---

## 임계값 (운영하며 보정)

| 상수 | 위치 | 기본값 | 의미 |
|---|---|---|---|
| `WEAK_F1` | diagnose.py | 0.85 | 약한 클래스 판정 |
| `CONFUSION_PAIR_FRAC` | diagnose.py | 0.03 | 혼동쌍 경보 (source 의 3%) |
| `GATE_MAX_ACC_DROP` | diagnose.py | 0.02 | 게이트: 전체 정확도 하락 한계 |
| `GATE_MAX_CLASS_RECALL_DROP` | diagnose.py | 0.05 | 게이트: 클래스 recall 하락 한계 |
| `ETC_QUEUE_TRIGGER` | etc_queue.py | 30 | etc 자동 처리 발동 |
| `REASSIGN_SOFTMAX` | etc_queue.py | 0.85 | Stage1 재배정 확신 |
| `REASSIGN_MAX_COSDIST` | etc_queue.py | 0.40 | Stage1 prototype 거리 상한 |
| `CLUSTER_MIN_SIZE` | etc_queue.py | 5 | Stage2 신규 후보 최소 크기 |

---

## 운영 흐름 (재학습 한 사이클)

```
retrain.py
  [0] etc_queue.maybe_process()      # ≥30 이면 재배정 + 신규 후보
  [1] 피드백 수집 (갱신된 라벨 반영)
  [2-4] 다운로드 + 격리 + preprocessor
  [5] 학습 (frozen test 제외하고 train/val)
  [6] diagnose → 게이트
        FAIL → 자동 롤백, 종료 (옛 모델 유지)
  [7] publish + activate (waste-api·앱 자동 갱신)
```

## 첫 사용 / 셋업 체크리스트

- [ ] Supabase 에서 `004_model_diagnostics.sql` 실행 (완료)
- [ ] Supabase 에서 `005_etc_clusters.sql` 실행
- [ ] 첫 `retrain.py` 1회 — 정직한(누수 없는) baseline 확립.
      (현재 모델은 freeze 이전 학습이라 frozen-test 숫자가 부풀려져 있음 → 첫
       정식 retrain 부터가 비교 가능한 baseline)

## 알아둘 점

- **현재 baseline 캐비엇**: 기존 모델은 frozen test 도입 전 학습돼서 일부 test
  이미지를 학습 때 봄(누수) → 지금 frozen-test 정확도(98%대)는 부풀려짐.
  정직한 값은 96%대(구 held-out). **다음 정식 retrain 부터 정확**.
- etc 큐 임계(30) 미달이면 `maybe_process()` 는 그냥 skip — 안전.
- pseudo-class 는 절대 사용자에게 자동 노출되지 않음 (active=false).
