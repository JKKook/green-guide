# waste-preprocessor — 데이터 파이프라인

> 소스: `waste-preprocessor/README.md`, `waste-preprocessor/src/`, `waste-preprocessor/data/` (2026-08-13 탐색)

첫 서브프로젝트: 이미지 **수집 → 정제 → 전처리 → manifest 생성**. 학습은 안 함(→ [waste-classifier](waste-classifier.md)). git 아님.

## 파이프라인 (`src/pipeline.py`)

수집(`collect.py`, Kaggle CLI) → 카탈로그(`catalog.py`, 폴더 auto-discover + uuid 12hex) → 클렌징(`cleanse.py`, PIL verify + phash 중복 제거) → 전처리(224² bilinear + ImageNet 정규화) → 벡터화(선택) → Supabase 업로드(선택) → `data/processed/manifest.json`.

- **현재 실질 운영 모드는 `--no-vectorize` raw-direct** — CNN이 raw JPEG를 직접 로드하므로 `.npz` 벡터(150,528-dim float16, MLP 시절 유산)는 생략. `data/processed/vectors/`는 비어 있음.
- `CLASS_LABELS`(구 6클래스)는 역사적 시드 — **현재 정본 taxonomy는 `waste-classifier/src/taxonomy.py`** ([hier-taxonomy](hier-taxonomy.md)).

## data/raw 구조 (2026-07-20 manifest: input 73,313 / cleansed 70,524)

- `garbage-classification/` — 대분류 13클래스 학습 정본 입력 (~7.3만장: glass·metal·paper·plastic·styrofoam·vinyl 각 ~1만, clothes 7.3천, **극소: etc 190 · non_object 720 · trash 835 · food_waste 989 · electronics 1,008 · cardboard 1,414**)
- `fine-staging/` — 세부 24라벨 ~15.4만장 (계층 taxonomy용, 기존 raw와 분리 저장. 이유: 평면으로 섞으면 감독 신호 충돌). 유입 경로는 [dataset-staging](dataset-staging.md)
- `synthetic_indoor/` — 합성 1,825장 + `_manifest.jsonl` (재현 가능: 같은 seed+source면 비트 단위 동일)
- `quarantine_too_few_samples/` — MIN_SAMPLES 미달 격리

## 실험 문서 2종 (레포 내)

- `AIHUB_PAPER_HYPOTHESIS_TEST.md` — AI-Hub paper 노이즈 가설 **기각** 실험 → [data-experiments](data-experiments.md)
- `DATA_AUGMENTATION_DESIGN.md` / `DATA_AUGMENTATION_RESULTS.md` — 실내 합성·TACO A/B (Test C1 미채택) → [data-experiments](data-experiments.md)

## 알려진 한계 (README)

비결정적 UUID(재실행 시 고아 npz), resumability 없음, 시각화·린트 부재, Kaggle 잔여 파일.
