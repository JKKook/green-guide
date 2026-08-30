# GreenGuide 고도화 청사진 (v2) — 계층형 프로덕션 분류 체계

> 재정립: 2026-07-07
> 기준 시스템: 활성 모델 13클래스(flat softmax) / ResNet18+DINOv2 앙상블 / 하이브리드(온디바이스+클라우드)
> 대체 대상: 기존 문서들이 참조했으나 실재하지 않던 `GREENGUIDE_BLUEPRINT.md` 를 이 문서로 확정.
> 연계 문서: [DIAGNOSIS_PROCESS.md](DIAGNOSIS_PROCESS.md) · [SMART_CAPTURE_STRATEGY.md](SMART_CAPTURE_STRATEGY.md) · [waste-preprocessor/DATA_AUGMENTATION_RESULTS.md](waste-preprocessor/DATA_AUGMENTATION_RESULTS.md) · [AIHUB_PAPER_HYPOTHESIS_TEST.md](waste-preprocessor/AIHUB_PAPER_HYPOTHESIS_TEST.md)

---

## 0. 문제 재정의 — 왜 "낱개 6→13 flat 확장"을 멈춰야 하는가

지금까지의 확장 방식은 **flat softmax 에 클래스를 하나씩 추가**(etc → electronics → non_object → clothes/…)하는 것이었다. 이 방식의 구조적 한계:

1. **클래스를 늘릴수록 클래스당 데이터가 부족해진다.** 현재도 이미 극심한 불균형 — vinyl 10,176장 vs **etc 189 / non_object 720 / trash 827 / food_waste 985 / electronics 1,002 / cardboard 1,006**. 여기서 "프로덕션급"으로 품목을 더 쪼개면(PET 무색/유색, 우유팩/멸균팩, 갈색/녹색 유리…) 세부 품목당 수백 장 확보도 어렵다.
2. **flat softmax 는 "애매하면 대분류로만 답하기"를 못 한다.** PET인지 PP인지 헷갈리면 그냥 틀린 세부 라벨을 확신 있게 뱉는다. 사용자에겐 "플라스틱함에 넣으세요"만 맞아도 충분한데, 그걸 표현할 구조가 없다.
3. **실사용 갭(frozen 95.9% vs 실사용 63.4%, −32.5pp)은 클래스를 늘린다고 좁혀지지 않는다.** [SMART_CAPTURE_STRATEGY.md](SMART_CAPTURE_STRATEGY.md) · [AIHUB_PAPER_HYPOTHESIS_TEST.md](waste-preprocessor/AIHUB_PAPER_HYPOTHESIS_TEST.md) 의 일관된 결론.

### 해법의 뼈대 (3축 결정 반영)

- **계층형 taxonomy**: 대분류(재질/배출스트림, 데이터 풍부·항상 견고) → 세부품목(데이터 차면 활성화). 데이터 부족을 **구조로 흡수**한다.
- **AI-Hub 우선 데이터 수집**: 세부품목 cold-start(폭)를 AI-Hub 대량 데이터로 채운다. HD/메모리 확보됨.
- **하이브리드 유지**: 온디바이스 = **대분류**(견고·오프라인·즉답), 클라우드 = **세부품목 + 앙상블**(정밀). 하이브리드 결정과 계층이 그대로 맞물린다.

> 원칙 한 줄: **"항상 대분류는 맞힌다. 세부는 데이터가 허락하는 만큼만, 확신할 때만."**

---

## 1. 계층형 Taxonomy (2-level)

### Level 1 — 대분류 (배출 스트림, 항상 활성 · 온디바이스)

한국 분리배출 표준의 실제 배출함 단위. 데이터가 풍부해 견고하게 학습 가능. **이 레벨만으로도 프로덕션에서 유용**("플라스틱류입니다 → 플라스틱함").

| # | slug | 한글 | 배출 스트림 | 현재 상태 |
|--|--|--|--|--|
| 1 | `paper` | 종이류 | 종이 분리수거 | 있음 |
| 2 | `paper_pack` | 종이팩 | **종이팩 별도 스트림**(우유팩/멸균팩) | 신규(現 cardboard/paper에 섞임) |
| 3 | `glass` | 유리류 | 유리병 전용 | 있음 |
| 4 | `metal` | 캔·고철 | 캔류/고철 | 있음 |
| 5 | `plastic` | 플라스틱 | 플라스틱 용기·완구 | 있음 |
| 6 | `vinyl` | 비닐류 | 비닐/필름 | 있음 |
| 7 | `styrofoam` | 스티로폼 | 발포합성수지 전용 | 있음 |
| 8 | `clothes` | 의류 | 의류수거함 | 있음 |
| 9 | `food_waste` | 음식물 | 음식물 전용 | 있음 |
| 10 | `electronics` | 소형가전 | 폐가전 수거(1599-0903) | 있음 |
| 11 | `hazardous` | 유해폐기물 | **건전지/형광등/폐의약품 전용함** | 신규(現 etc/electronics에 섞임) |
| 12 | `general` | 일반쓰레기 | 종량제봉투 | = 現 trash |

내부 신호(사용자 비노출): `non_object`(재촬영), `etc`(대분류도 애매 → 캐치올). 총 **대분류 12 + 내부 2**.

### Level 2 — 세부품목 (데이터-게이트 활성화 · 클라우드)

각 대분류 아래 "배출법이 실제로 달라지는" 품목만. **데이터 임계 넘긴 것만 `active=true`**, 나머지는 부모 대분류로 롤업.

```
plastic ─┬ pet_clear      무색 PET (라벨 제거·압착·뚜껑 분리)
         ├ pet_colored    유색 PET
         ├ pp_pe_container PP·PE 용기류
         └ plastic_complex 복합재질 플라스틱(완구 등) → 일반 안내

paper ───┬ newspaper_book  신문·책·노트
         ├ office_paper    사무용지
         └ receipt         영수증(감열지) → 일반쓰레기 안내

paper_pack ┬ carton_normal 일반팩(우유)
           └ carton_asept  멸균팩(두유·주스)

glass ───┬ glass_clear/brown/green  무색/갈색/녹색 병
         └ glass_deposit           보증금 반환 병(소주·맥주) → 반환 안내

metal ───┬ can_aluminum   알루미늄캔
         ├ can_steel      철캔
         ├ aerosol        부탄가스·스프레이(내용물 비움·구멍)
         └ scrap_metal    고철

vinyl ───┬ vinyl_clean    이물질 없는 비닐 → 분리배출
         └ vinyl_dirty    오염 비닐 → 일반쓰레기 안내

hazardous ┬ battery       건전지
          ├ light_bulb    형광등
          └ medicine      폐의약품 → 약국 수거

electronics ┬ small_appliance 소형가전
            └ large_appliance 대형가전 → 무상방문수거 안내
```

세부품목 규칙:
- **모든 세부품목은 정확히 하나의 부모 대분류로 롤업**된다(`parent_slug`). → 세부가 불확실해도 대분류는 항상 도출.
- **배출법이 대분류와 동일하면 세부품목을 만들지 않는다.** (분류를 위한 분류 금지 — 오직 "안내가 달라질 때"만 쪼갠다.)
- 일부 세부품목은 "이건 사실 일반쓰레기예요" 안내용(영수증·오염비닐·복합플라스틱) — **오분리 방지**가 목적.

---

## 2. 모델 아키텍처 — 계층 = 온디바이스/클라우드 분업

기존 단일 softmax 를 버리지 않고, **세부 softmax + 결정적 롤업(deterministic rollup)** 으로 계층을 구현한다. 별도 계층 학습(brittle) 대신 taxonomy 테이블의 `parent_slug` 로 계층을 표현.

### 2.1 온디바이스 = 대분류 분류기 (견고·오프라인·즉답)

- 활성 대분류(~12)만 출력하는 경량 모델. 대분류는 데이터가 많아 **실사용에서도 견고**.
- 현행 ResNet18/MobileNet 유지 가능. `assets/models/classifier_coarse.onnx`.
- 오프라인/저신뢰 클라우드 fallback 시에도 **"대분류는 답한다"** 보장. → 하이브리드의 안전 바닥.

### 2.2 클라우드 = 세부품목 분류기 + 앙상블 (정밀)

- 활성 세부품목 전체 softmax → **부모 대분류별 확률 합산**으로 대분류 확률도 동시 산출.
  - `P(대분류 c) = Σ_{fine ∈ c} P(fine)` — 대분류는 세부의 결정적 롤업. 세부가 흩어져도 대분류는 견고.
- 기존 캐스케이드 그대로 재사용: MediaPipe 손감지 → Stage1 이진 게이트 → 세부 분류기(ResNet18) → **DINOv2 앙상블**(신뢰도 보정). [waste-api/src/api.py](waste-api/src/api.py) `predict_centered`.
- DINOv2 임베딩은 **few-shot 신규 세부품목**의 프로토타입 근거로도 재사용(3.3 참고).

### 2.3 표현 깊이 = 신뢰도 게이트 (핵심 UX 규칙)

```
if 세부 top1 ≥ τ_fine and (top1−top2) ≥ margin:   → 세부품목까지 안내 ("무색 PET, 라벨 떼고 압착")
elif 대분류 확률 ≥ τ_coarse:                        → 대분류만 안내 ("플라스틱류, 플라스틱함")
else:                                              → etc/non_object (재촬영 or 캐치올)
```

이 규칙이 "flat softmax 가 애매해도 확신 있게 틀리던 문제"를 구조적으로 제거한다. 기존 [waste_app/lib/data/confidence.dart](waste_app/lib/data/confidence.dart)(top1<0.55 or entropy>0.7 reject)를 **레벨별 임계**로 일반화.

---

## 3. 데이터 전략 — AI-Hub 우선 수집 (선택된 1순위 레버)

### 3.1 왜 AI-Hub가 세부품목 확장에 맞는가 (그리고 한계)

- AI-Hub 재활용/생활폐기물 세트는 **품목이 세분화·박스 라벨링**되어 있어, flat 6클래스로는 못 쓰던 세부 라벨을 **바로 세부품목 cold-start** 에 매핑 가능. HD 확보됨 → 대량 수용 가능.
- **정직한 한계(문서로 검증됨)**: AI-Hub는 스튜디오/시설 분포라 **실사용 갭을 혼자 못 닫는다**([AIHUB_PAPER_HYPOTHESIS_TEST.md](waste-preprocessor/AIHUB_PAPER_HYPOTHESIS_TEST.md): "라벨 정확도 ≠ 학습 기여", 진짜 문제는 분포 미스매치). → AI-Hub는 **"폭(breadth)/세부품목을 존재하게 하는" 레버**이고, **"실사용 정확도(depth)"는 크롭·도메인 랜덤화 + 실데이터 수집과 병행**해야 한다. 이 청사진은 AI-Hub를 1순위로 하되 이 병행을 명시한다.

### 3.2 AI-Hub 수집·통합 파이프라인 (기존 스크립트 재사용·확장)

이미 있는 통합 인프라를 세부품목 매핑으로 확장:
- [waste-classifier/scripts/integrate_aihub_140.py](waste-classifier/scripts/integrate_aihub_140.py) — AI-Hub `CLASS`(예: `전자제품`) → `--our-class` 박스 크롭 매핑. **여기에 세부품목 매핑 테이블 추가**.
- [scripts/filter_aihub_by_quality.py](waste-classifier/scripts/filter_aihub_by_quality.py) — 품질 필터(어두운 시설 컷 제거) 강화.
- 이미 확보/스테이징된 세트: AI-Hub 71362(재활용품), 140(생활폐기물), **71647(손동작 3D — `aihub_71647_staging/` 에 대기, 손 마스크용)**.

**수집 대상 선정 기준**(품목이 아니라 "부족한 대분류/세부품목 순"):
1. 신규 대분류 `paper_pack`·`hazardous` 를 먼저 견고화(현재 데이터 0에 가까움).
2. 세부품목 중 배출법 임팩트 큰 것 우선: PET 무색/유색, 우유팩/멸균팩, 캔 알루미늄/철, 유리 색상.
3. 각 세부품목 **활성화 임계(3.4)** 를 채우는 최소 수량 목표로 역산해 수집량 결정.

> 실행 시: AI-Hub 데이터셋별 라이선스·다운로드 승인 상태를 먼저 확인(71647은 "사이트 승인 대기" 이력 있음). 다운로드 후 `integrate_aihub_140.py` 확장본으로 세부품목 폴더(`data/raw/garbage-classification/<fine_slug>/`)에 크롭 적재 → preprocessor manifest 자동 반영.

### 3.3 세부품목 cold-start 을 위한 few-shot 보조 (재학습 없이 등록)

AI-Hub로도 즉시 못 채우는 롱테일 세부품목은 **DINOv2 임베딩 프로토타입**(이미 서빙 중)으로 few-shot 등록:
- 세부품목별 수십 장 임베딩 평균 → prototype. [waste-classifier/src/ood.py](waste-classifier/src/ood.py) 프로토타입 인프라 재사용.
- 재학습 전까지 **클라우드 retrieval 로 잠정 세부 판정**(active=false 상태로 A/B 관찰) → 데이터 차면 정식 head 로 승격.

### 3.4 병행 필수 레버 (AI-Hub만으로 부족한 부분)

- **크롭 정합(domain match)**: 학습 데이터에 u2netp 자동 크롭 버전 혼합(B-1) — AI-Hub 박스 크롭과 실사용 `/predict-centered` 크롭 분포를 맞춘다. [SMART_CAPTURE_STRATEGY.md](SMART_CAPTURE_STRATEGY.md) §3-B.
- **실사용 데이터 수집 루프 상시 가동**: in-app 피드백 + `etc_clusters` 승격은 계속. AI-Hub가 폭을, 실데이터가 정확도를 담당.

---

## 4. DB·스키마 개편 (하위호환 유지)

기존 테이블을 깨지 않고 계층 컬럼만 추가. migration 008~(수동 Supabase SQL, 기존 관례).

### `waste_classes` 확장 (migration 008)
```sql
ALTER TABLE waste_classes ADD COLUMN level        SMALLINT NOT NULL DEFAULT 1;  -- 1=대분류, 2=세부품목
ALTER TABLE waste_classes ADD COLUMN parent_slug  TEXT REFERENCES waste_classes(slug);  -- 세부→대분류 롤업
ALTER TABLE waste_classes ADD COLUMN min_samples_to_activate INT DEFAULT 300;   -- 활성화 임계(train)
ALTER TABLE waste_classes ADD COLUMN min_frozen_to_activate  INT DEFAULT 30;    -- 활성화 임계(frozen test)
-- 기존 active/trained_in_model 은 그대로: active=false 세부품목은 부모로 롤업되어 노출.
```
- 대분류 12행: `level=1, parent_slug=NULL`.
- 세부품목 N행: `level=2, parent_slug=<대분류>`, 초기 `active=false`.
- `non_object`/`etc` 는 `level=1, active=false`(내부 신호) 유지.

### `model_versions` 확장 (migration 009)
```sql
ALTER TABLE model_versions ADD COLUMN coarse_labels JSONB;   -- 온디바이스 대분류 라벨
ALTER TABLE model_versions ADD COLUMN taxonomy_hash TEXT;    -- 계층 스냅샷 해시(앱 캐시 무효화)
-- 기존 class_labels 는 세부(클라우드) 라벨로 계속 사용.
```

### `model_diagnostics` — 레벨별 게이트 (migration 010)
```sql
ALTER TABLE model_diagnostics ADD COLUMN coarse_accuracy REAL;    -- 대분류 정확도(롤업 후)
ALTER TABLE model_diagnostics ADD COLUMN per_fine JSONB;          -- 세부품목별 지표
-- 게이트: 대분류 recall 회귀는 강하게 차단(사용자 체감 직결), 세부는 신규 활성화만 허용.
```

### `etc_clusters` — 승격 대상에 parent 지정
운영자 리뷰 시 신규 클러스터를 **어느 대분류 아래 세부품목으로** 승격할지 지정(기존 자동발굴 흐름 [DIAGNOSIS_PROCESS.md](DIAGNOSIS_PROCESS.md) §5 를 계층에 맞춤).

---

## 5. 데이터-게이트 세부품목 활성화 (자동 세분화 메커니즘)

"낱개 수동 확장"을 대체하는 핵심. 세부품목은 **조건 충족 시에만 자동 노출**:

```
세부품목 fine 이 active=true 되는 조건 (retrain.py 게이트에 추가):
  ① train 표본 ≥ min_samples_to_activate (기본 300, AI-Hub+실데이터 합산)
  ② frozen test 표본 ≥ min_frozen_to_activate (기본 30) — 정직한 측정 가능
  ③ diagnose 게이트 통과: fine 추가가 부모 대분류 recall 을 −5pp 초과 떨어뜨리지 않음
  ④ fine 자체 f1 ≥ 0.80 (frozen test)
  → 모두 충족 시 active=true 승격, 아니면 active=false 유지(부모로 롤업)
```

- [waste-classifier/src/frozen_test.py](waste-classifier/src/frozen_test.py) 의 안정키 동결을 **세부품목 단위로 확장** → 세부품목별 회귀도 버전 간 비교 가능.
- [DIAGNOSIS_PROCESS.md](DIAGNOSIS_PROCESS.md) 의 PASS/FAIL 게이트에 "대분류 recall 은 절대 회귀 금지 / 세부는 승격만" 규칙 추가.
- 결과: **taxonomy 는 전체 스키마로 미리 정의해두고, 모델은 데이터가 차는 순서대로 자동으로 세분화**된다. 사람은 클러스터 이름·배출법만 넣는다(기존과 동일).

---

## 6. 앱 UX 개편 — 계층 표현

- **결과 화면 2단 표시**: 상단 대분류 배지(항상, 큰 배출함 아이콘) + 하단 세부품목 카드(확신 시에만). 세부 불확실 → 대분류만.
- **배출 가이드 상속**: 세부품목 카드는 부모 대분류 기본 안내 + 세부 특화 안내(라벨 제거/압착/색상 분리)를 덧붙임. `WasteInfo` 에 `parentSlug` 추가([waste_app/lib/data/waste_info.dart](waste_app/lib/data/waste_info.dart)).
- **오분리 방지 안내**: "영수증/오염비닐/복합플라스틱 → 일반쓰레기" 같은 세부품목은 눈에 띄게 경고 스타일.
- **온디바이스 즉답 → 클라우드 정밀 보강**: 온디바이스가 먼저 대분류를 즉시 보여주고, 클라우드 응답 도착 시 세부품목으로 자연스럽게 채워짐(현행 milestone 로더 [waste_app/lib/widgets/result_modal.dart](waste_app/lib/widgets/result_modal.dart) 확장).
- `/labels` 응답에 `level`/`parent_slug` 포함 → 앱이 계층 렌더(기존 동적 레지스트리 [class_loader.dart](waste_app/lib/services/class_loader.dart) 확장).

---

## 7. 측정·거버넌스 (계층 인지)

레벨별로 성공 기준을 분리한다.

| 지표 | 대상 | 목표 |
|--|--|--|
| **대분류 정확도(frozen)** | Level 1 롤업 | ≥ 97% (회귀 절대 차단) |
| **대분류 정확도(실사용)** | user_uploads 피드백 | 63% → **≥ 85%** (최우선 KPI) |
| **세부품목 커버리지** | active=true 세부 수 | 데이터 차는 순서대로 점증(수치 목표 아님) |
| **세부품목 f1(frozen)** | active 세부품목 | 각 ≥ 0.80 (미달 시 롤업) |
| **오분리율** | "일반쓰레기 안내" 세부품목 | 하락 추적 |

- [waste-classifier/realworld_eval.py](waste-classifier/realworld_eval.py) 를 **레벨별(대분류/세부) 동시 산출**로 확장하고 매 retrain 자동 실행([SMART_CAPTURE_STRATEGY.md](SMART_CAPTURE_STRATEGY.md) §4 의 미해결 과제 해소).
- **핵심 원칙**: 세부품목을 늘리려다 대분류를 망치지 않는다. 대분류 회귀는 즉시 자동 롤백(기존 게이트 [retrain.py](waste-classifier/retrain.py)).

---

## 8. 단계별 실행 로드맵

| Phase | 내용 | 산출물 | 리스크 |
|--|--|--|--|
| **P0. Taxonomy 확정** | 대분류 12 + 세부품목 스키마 확정, migration 008~010, waste_classes 시딩(세부는 active=false) | 스키마·시드 SQL | 낮음(하위호환) |
| **P1. 대분류 견고화** | AI-Hub로 신규 대분류(`paper_pack`,`hazardous`) 채우고 온디바이스 대분류 분류기 학습·배포 | `classifier_coarse.onnx`, 대분류 frozen ≥97% | 중(신규 대분류 데이터) |
| **P2. 계층 서빙** | 세부 softmax + 부모 롤업, 레벨별 신뢰도 게이트, 앱 2단 UX | `/predict*` 계층 응답, 앱 계층 렌더 | 중 |
| **P3. AI-Hub 세부품목 수집** | 임팩트순 세부품목 AI-Hub 크롭 적재 + 크롭 정합(B-1) 혼합 | 세부품목별 표본 축적 | 중(분포 갭) |
| **P4. 데이터-게이트 자동 세분화** | 활성화 임계·게이트 가동, 차는 순서대로 세부품목 자동 노출 | active 세부품목 점증 | 낮음(게이트 보호) |
| **P5. 실사용 정확도 depth** | 크롭 앙상블 + 실데이터 루프 상시화, 레벨별 realworld eval 자동화 | 실사용 대분류 ≥85% | 높음(근본: 실데이터) |

권장 순서 원칙: **대분류 먼저 완전히 견고하게(P1) → 그 위에 세부를 데이터 차는 대로(P3-P4).** 세부를 서두르지 않는다.

---

## 9. 솔직한 한계 (변하지 않는 결론)

- 계층형은 **"데이터 부족을 우아하게 다루는 구조"**이지, 데이터를 만들어내진 않는다. 세부품목 정확도는 결국 표본 수에 비례.
- AI-Hub는 **세부품목을 존재하게(폭)** 하지만, **실사용 정확도(깊이)는 크롭 정합 + 실사용자 데이터** 없이는 못 올라간다(문서 3종의 반복된 실증).
- 그래서 이 청사진의 성공 여부는 **"대분류를 실사용에서 85%까지 견고하게 만들고(P1·P5), 세부는 게이트로 안전하게 점증"** 시키는 데 달렸다. 대분류만 견고해도 이미 프로덕션에서 쓸 만하다 — 세부는 보너스로 쌓인다.

---

## 부록 A. 기존 자산 → 청사진 매핑 (버릴 것 없음)

| 기존 자산 | v2에서의 역할 |
|--|--|
| `waste_classes.active/trained_in_model` | 세부품목 데이터-게이트 활성화 플래그로 승격 |
| `model_versions.class_labels` + frozen test | 세부(클라우드) 라벨 + 레벨별 회귀 비교 |
| diagnose PASS/FAIL 게이트 | "대분류 회귀 금지 / 세부 승격만" 규칙으로 확장 |
| `etc_queue` + `etc_clusters` | 신규 세부품목 자동 발굴 → parent 지정 승격 |
| DINOv2 앙상블 + `ood.py` 프로토타입 | 세부품목 few-shot cold-start + open-set reject |
| `integrate_aihub_140.py` / `filter_aihub_by_quality.py` | 세부품목 매핑·품질필터로 확장 |
| 하이브리드 추론(온디바이스/클라우드) | 대분류/세부 분업으로 자연스럽게 재배치 |
| `/predict-centered` u2netp 크롭 | 크롭 정합(B-1) 학습 혼합의 실사용측 짝 |
