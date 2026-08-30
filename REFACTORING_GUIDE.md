# 리팩토링 작업 가이드 — 코드 품질 최적화 + 공통단(waste-common) 구성

작성일: 2026-08-30 · 대상: `waste-preprocessor` (955 LOC) · `waste-classifier` (10.6k LOC) · `waste-api` (5.6k LOC)

이 문서는 "무엇을, 어떤 순서로, 어떻게 검증하며" 리팩토링할지를 정한다. 모든 수치·위치는 2026-08-30 기준 실제 코드에서 측정한 값이다.

---

## 0. 한 줄 요약

세 프로젝트가 **같은 설정·같은 이미지 전처리·같은 Supabase 접속·같은 분류체계**를 각자 복사해 갖고 있다. 이를 설치형 패키지 `waste-common` 하나로 모으고, 그 위에서 `api.py`(1,463줄)와 `*_hier` 포크를 정리한다. **동작 변경 없음**이 원칙이며, 각 단계는 테스트 통과로 닫는다.

---

## 1. 현황 진단 (근거)

| # | 문제 | 규모 | 대표 위치 |
|---|---|---|---|
| D1 | `config.py` 3벌 + 로딩 방식 제각각 (dotenv 위치 다름, 미로딩) | 3 파일 + 6-클래스 튜플 4벌 | `*/src/config.py`, `waste-classifier/retrain.py:47` |
| D2 | ImageNet 정규화 체인(RGB→224→/255→mean/std) 재구현 | **21 파일**에 mean/std 하드코딩 | `waste-api/src/preprocess.py:67`, `waste-classifier/src/dataset.py:43`, `dinov2_classifier.py:79` … |
| D3 | 분류체계 SSOT는 `waste-classifier/src/taxonomy.py`인데 api가 import 못 해 JSON 재파싱 3경로 | 3 경로 + 매핑 테이블 6벌 | `waste-api/src/classes.py:82-126`, `hier_inference.py:60` |
| D4 | `create_client()` 직접 호출 | **17 곳**, 가드 블록 6벌 복붙 | `preprocessor/src/storage.py:34`, `api/src/uploads.py:32`, `classifier/retrain.py:69,87,109,248,279` … |
| D5 | ONNX 세션 생성/모델 경로 해석 반복 | 6 곳 | `api/src/config.py:15-46` ↔ `hier_inference.py:29-48` |
| D6 | `waste-api/src/api.py` 라우팅+비즈니스+IO 혼재 | 1,463줄, `predict_hier` **234줄**, `predict_with_regions` **204줄**, 함수 내부 import 41개 | `api.py:185`, `api.py:1201` |
| D7 | `train`/`hier_train`, `retrain`/`retrain_hier` 포크 | train 쌍 26% 동일, `backup/rollback_artifacts` 2벌 | `src/train.py:117` ↔ `src/hier_train.py:168` |
| D8 | 스크립트 보일러플레이트 | `sys.path.insert` 16 파일, 경로 상수 3줄 ~12 파일, 절대경로 하드코딩 | `scripts/publish_hier_version.py:141` (`/Users/ethan/...`) |
| D9 | 로깅 부재 | `import logging` **0건**, `print()` 655회, `except Exception` 121곳 (`str(exc)[:80]` 절단) | 전 프로젝트 |
| D10 | 테스트 공백 | 최고 위험 3모듈 `api.py`·`retrain.py`·`segment.py`에 직접 테스트 없음 | — |

**preprocessor 자체 이슈 (직접 확인):**
- `main.py:40,44` — `from src.preprocess import __main__` 은 모듈 속성이 아니므로 `--step preprocess/vectorize`는 ImportError. 죽은 경로.
- `pipeline.py:66-93` — `vectorize` 분기 두 개가 record 생성/업로드를 각각 중복. `_processed_record`와 raw 분기 dict를 하나로.
- `pipeline.py:97` — `{k:v ... if k != "stats"} | {"stats": r["stats"]}` 는 no-op.
- `preprocess.py:load_rgb` — EXIF 회전 미처리. 서빙(`api/src/preprocess.py:37`)은 `exif_transpose` 적용 → **학습/서빙 분포 불일치**. `hier_inference`가 회전 TTA로 보상 중. 공통 `imaging`으로 통일 시 자연 해소.
- `config.py` 모듈 상수를 함수 기본값으로 사용(`dataset_dir: Path = config.DATASET_DIR`) → 테스트가 `monkeypatch.setattr(config, ...)`에 의존하는 취약 패턴. 기본값을 `None`으로 두고 함수 안에서 해석.

**구조적 제약:** `waste-api/.git`이 별도 저장소다. 공통 패키지는 상대 import가 아니라 **`pip install -e ../waste-common`** 방식이어야 한다. 이것이 `sys.path.insert` 16곳과 절대경로 하드코딩도 함께 없앤다.

---

## 2. 원칙 (작업 규칙)

1. **동작 불변.** 리팩토링 PR에 기능 변경을 섞지 않는다. 출력이 바뀌면 그건 버그 수정이며 별도 커밋으로 분리한다.
2. **테스트 먼저 잠근다.** 옮길 코드에 테스트가 없으면 *옮기기 전에* 현재 동작을 고정하는 characterization test를 쓴다. (예: 이미지 1장 → 정규화 배열의 해시값)
3. **한 PR = 한 추출.** "Supabase 클라이언트 통합" 같은 단위로 자른다. 17곳을 한 번에 바꿔도 되지만, 두 주제를 섞지 않는다.
4. **대체 후 삭제.** 공통 함수를 만들고 → 호출부를 바꾸고 → 원본을 지운다. 둘 다 남겨두는 상태로 PR을 닫지 않는다 (`grep`으로 잔존 0 확인이 PR 체크리스트).
5. **인접 코드 손대지 않기.** 추출 중 눈에 띈 다른 개선은 이 문서 §6 백로그에 적고 넘어간다.
6. **검증 명령을 PR 설명에 적는다.** 각 단계의 `verify:`가 통과 기준이다.

---

## 3. 목표 구조

```
waste/
  waste-common/                 # 신규. pip install -e 로 세 프로젝트에 설치
    pyproject.toml
    waste_common/
      __init__.py
      settings.py               # D1  pydantic-settings: Supabase, 버킷/테이블, 이미지 상수, 형제 프로젝트 루트
      taxonomy.py               # D3  waste-classifier/src/taxonomy.py 를 그대로 이동 (+ tests/test_hierarchy.py 동반)
      imaging.py                # D2  decode_rgb, exif_upright, to_normalized_array, to_model_input, recompress
      supabase.py               # D4  get_client()(memoized), Bucket enum, upload_and_get_url, download
      onnx.py                   # D5  OnnxModel(경로 후보 해석 + 세션 + softmax), ModelArtifact 매니페스트
      logging.py                # D9  get_logger(name), fail_open(ctx) — traceback 포함 warning
      cli.py                    # D8  make_parser(common flags: --seed/--dry-run/--cap)
    tests/
  waste-preprocessor/           # waste_common 소비. src/config.py 는 경로 상수만 남김
  waste-classifier/             # src/taxonomy.py 삭제 → re-export shim 1줄 → 최종 제거
  waste-api/
    src/
      routers/{meta,inference,admin,learning}.py   # D6  얇은 어댑터 (요청 파싱 → 서비스 호출 → 응답)
      services/pipeline.py                         # D6  hand→stage1→TTA→evidence fusion 캐스케이드 (순수 함수)
      services/recording.py                        # D6  record_prediction 단일 훅
```

**공통단에 넣지 않는 것:** torch 의존 코드(모델 정의·학습 루프)는 classifier에 남긴다. `waste-common`은 numpy/Pillow/supabase/onnxruntime/pydantic 이상을 의존하지 않는다 — api 런타임을 무겁게 만들지 않기 위함.

---

## 4. 단계별 계획 (순서 = 효과/비용 순)

각 단계: `[작업] → verify: [확인 방법]`. 한 단계가 한 PR.

### Phase 0 — 안전망 (반나절)
- [ ] `waste-common` 스켈레톤 생성 (`pyproject.toml`, 빈 패키지, pytest 설정). 세 프로젝트 `requirements.txt`에 `-e ../waste-common` 추가.
  → verify: 세 venv에서 `python -c "import waste_common"` 성공
- [ ] 루트에 `ruff.toml` 추가 (현재 ruff 미설치, pyright만 있음). 규칙은 최소: `E,F,I` + `PLC0415`(함수 내 import) 는 **경고만**. 기존 코드 위반은 baseline으로 `--add-noqa` 하지 말고 그냥 통과 기준을 "새 위반 0"으로.
  → verify: `ruff check .` 가 실행되고 baseline 위반 수를 README에 기록
- [ ] characterization test 3개: (a) 샘플 이미지 → `to_normalized_array` 결과 sha256, (b) `taxonomy` 전체 매핑 스냅샷 JSON, (c) api `/predict-hier` 응답 스키마 스냅샷 (모델 로드 mock).
  → verify: `pytest` 3/3 통과, 이후 모든 Phase의 회귀 기준

### Phase 1 — Supabase 클라이언트 통합 (D4, 17곳, 기계적)
- [ ] `waste_common/supabase.py`: `get_client()`(lru_cache, 없으면 `RuntimeError` 단일 메시지), `class Bucket(StrEnum): RAW_IMAGES, USER_UPLOADS, MODELS`, `upload_and_get_url(bucket, remote_path, local_path)`, `download(bucket, remote_path) -> bytes`.
- [ ] 17개 호출부 교체. `preprocessor/src/storage.py:_client`, `api/src/uploads.py:_client`, `model_loader.py:71`, classifier 11곳.
- [ ] 콘텐츠타입 맵 2벌(`storage.py:69`, `uploads.py:39`) → `imaging.content_type(path)` 하나로 (Phase 2 선행 배치 가능).
  → verify: `grep -rn "create_client(" waste-*/ --include=*.py | grep -v waste-common | wc -l` == 0; 각 프로젝트 `pytest` 통과; `python main.py --step supabase-check`(preprocessor) 성공

### Phase 2 — 이미지 전처리 통합 (D2, 21파일)
- [ ] `waste_common/imaging.py`: `decode_rgb(src: Path|bytes, *, exif=True)`, `resize_square(img, size)`, `to_normalized_array(img) -> HWC float32`, `to_model_input(arr, layout="chw")`, `recompress_for_storage(img, max_side, fmt, quality)`.
- [ ] mean/std 상수는 `waste_common.imaging.IMAGENET_MEAN/STD` 단일 정의. 21파일의 로컬 `_MEAN/_STD` 삭제.
- [ ] preprocessor `load_rgb`에 EXIF 적용 — **이건 동작 변경**이므로 별도 커밋 + 커밋 메시지에 명시. 기존 `.npz` 벡터 재생성 필요 여부를 README에 기록.
- [ ] `api/src/preprocess.py:160 color_tensor_at` 인라인 재구현 → 공통 함수 호출.
  → verify: Phase 0(a) 해시 테스트 통과 (EXIF 커밋 제외); `grep -rn "0.485" waste-*/ --include=*.py | grep -v waste-common | wc -l` == 0

### Phase 3 — 설정 + 분류체계 (D1, D3)
- [ ] `waste_common/settings.py`: `class Settings(BaseSettings)` — `supabase_url/key`, 버킷/테이블명, `image_size=224`, `preprocessor_root/classifier_root/api_root`(env 오버라이드 가능, 기본은 형제 디렉토리). `.env` 탐색은 여기서 한 번만. classifier의 `load_dotenv(PREPROCESSOR_ROOT/".env")` 8곳 제거.
- [ ] `taxonomy.py`를 `waste_common`으로 **파일 이동**(git mv) + 테스트 동반. classifier에는 `from waste_common.taxonomy import *` shim을 1 릴리즈 유지 후 삭제.
- [ ] api `ClassRegistry` fallback(`classes.py:100-126`)을 JSON 재파싱 → `waste_common.taxonomy` import로 교체. `hier_export`는 sidecar JSON을 taxonomy 모듈에서 생성.
- [ ] 6-클래스 레거시 튜플 4벌 → `taxonomy.LEGACY_6` 하나. `preprocessor/collect.py:dataset_present`가 이걸 쓰는 것은 유지(Kaggle 데이터셋 검증 목적이므로 정당).
- [ ] `classifier/src/config.py:36-60` import-time 전역 mutate 제거 → `taxonomy`에서 함수로 조회. (`model.py:73-77` 주석의 11/12-클래스 버그 근본 원인)
- [ ] `api/src/uploads.py:15-16` 죽은 코드(항상 None) 삭제.
  → verify: Phase 0(b) 스냅샷 동일; `pytest waste-common/tests/test_hierarchy.py` 통과; `grep -rn "load_dotenv" waste-classifier --include=*.py | wc -l` == 0

### Phase 4 — api.py 분해 (D6, 최대 위험 감소)
- [ ] `services/pipeline.py`로 `predict_hier`(234줄) 본문을 **순서대로 잘라** 함수화: `gate_hand(img) -> HandResult`, `gate_stage1(arr) -> bool`, `run_tta(model, img) -> probs`, `fuse_evidence(probs, ocr, clip, cam) -> Fused`, `maybe_vlm_fallback(fused) -> Fused`. 각 함수는 이미지·확률만 받고 FastAPI 객체를 모른다.
- [ ] 라우트 모듈을 `routers/` 4개로 분리. 핸들러는 15줄 내외 목표(업로드 검증 → 서비스 → 응답 모델).
- [ ] `get_recorder().record_prediction` 7곳 → `services/recording.py`의 단일 `after_predict()` 훅.
- [ ] 임계값 상수(`OCR_SKIP_CONFIDENCE`, `hand_area 0.50`, `stage1 0.50`)를 `services/thresholds.py` 로.
- [ ] 함수 내부 import 41개: 서비스 분리 후 순환이 사라지면 모듈 상단으로. 남는 것은 "startup 비용" 사유를 주석에 명시.
- [ ] `lifespan` 안의 prune 루프 → `services/prune.py`.
  → verify: Phase 0(c) 응답 스키마 스냅샷 동일; `api.py` 200줄 이하(app 생성 + include_router만); 새로 생긴 `services/*` 함수마다 최소 1 단위테스트(mock 모델); `tests/test_api.py` 기존 3개 통과

### Phase 5 — classifier 포크 정리 + 스크립트 베이스 (D7, D8)
- [ ] `backup_artifacts/rollback_artifacts` 2벌 → `src/artifacts.py` 하나.
- [ ] `_compute_class_weights`/`_capped_inverse_freq` → 하나의 `inverse_freq_weights(counts, cap)`.
- [ ] `_run_epoch` vs `_run_hier_epoch`: loss 계산만 콜백으로 받는 `run_epoch(model, loader, loss_fn, ...)` 하나로. `hier_dataset.py`가 `dataset.py`를 compose하는 방식이 정답 — 같은 패턴 적용.
- [ ] `scripts/_base.py`: 경로 상수 + `make_parser()`. `sys.path.insert` 16곳 제거(설치형 패키지로 대체). `publish_hier_version.py:141` 절대경로 삭제.
- [ ] `integrate_aihub.py` vs `integrate_aihub_140.py`(~85% 동일) → 라벨 파서만 인자로 받는 하나.
- [ ] `export.py:12`의 private import(`from src.train import _model_kind`) → public 함수로 승격.
  → verify: `grep -rn "sys.path.insert" waste-classifier --include=*.py | wc -l` == 0; `pytest waste-classifier` 통과; `retrain_hier.py --dry-run` 실행 가능

### Phase 6 — 로깅 + 예외 (D9)
- [ ] `waste_common/logging.py`: `get_logger(name)` (포맷 `%(levelname)s %(name)s: %(message)s`, env `LOG_LEVEL`), `fail_open(logger, msg)` 컨텍스트 매니저 — 예외를 `logger.warning(..., exc_info=True)` 로 기록 후 삼킴.
- [ ] `print("[prefix] ...")` 655곳을 **모듈 단위로** 교체 (한 PR에 한 프로젝트). 대괄호 접두어는 logger name으로 대체되므로 제거.
- [ ] `except Exception: print(str(exc)[:80])` 패턴 → `with fail_open(log, "..."):`. 절단 없이 traceback 보존.
  → verify: `grep -rn "print(" waste-api/src | wc -l` == 0 (CLI 출력 목적의 `main.py`는 예외); 테스트에서 `caplog`로 warning 캡처 가능

### Phase 7 — preprocessor 소규모 정리 (독립, 언제든)
- [ ] `main.py:40,44` 죽은 `__main__` import → `preprocess`/`vectorize`에 `run_sample()` 함수 노출 후 호출.
- [ ] `pipeline.run` 두 분기 통합: record dict를 하나 만들고 `vectorize`일 때만 `vector_path/stats` 추가. no-op dict 연산(`:97`) 삭제.
- [ ] 함수 기본값의 `config.X` 참조 → `None` + 내부 해석. `conftest.py`의 `monkeypatch.setattr(config, ...)` 를 인자 전달로 교체.
- [ ] `storage.py`, `cleanse.py`, `pipeline.py` 테스트 추가 (Supabase는 `Client` 주입 mock).
  → verify: `pytest` 통과, `python main.py --step preprocess` 정상 출력

---

## 5. PR 체크리스트 (모든 리팩토링 PR 공통)

- [ ] 제목이 `refactor(<scope>): <추출/이동 단위>` 형식, 동작 변경 커밋은 `fix:`/`feat:`로 분리됨
- [ ] 원본 중복 코드가 **삭제**되었고 `grep` 잔존 수를 PR 본문에 적음
- [ ] 해당 Phase의 `verify:` 명령과 결과를 PR 본문에 붙임
- [ ] Phase 0 characterization test 3개 통과 (의도된 동작 변경이면 스냅샷 갱신 커밋을 따로)
- [ ] 세 프로젝트 `pytest` 모두 통과 (한 프로젝트만 건드렸어도 — 공통단 변경 파급 확인)
- [ ] `ruff check` 새 위반 0, `pyright` 새 오류 0
- [ ] 리팩토링 중 발견한 무관한 문제는 고치지 않고 §6에 추가

---

## 6. 백로그 (이번 범위 밖, 발견 시 여기에)

- `waste-api` 별도 `.git` → 모노레포 통합 여부 결정 (통합하면 `pip install -e` 대신 workspace 방식 가능)
- `.venv_broken_whdrnr01/`(preprocessor, 이전 계정 잔재) 삭제
- seed 4곳(`SPLIT_SEED`, `RANDOM_SEED`, 스크립트 `--seed 42` ×3) 단일화 — Phase 5 `make_parser` 이후
- U2-Net saliency 래퍼 5벌(`segment.py`, `synthesize_*.py` ×3, `filter_aihub_by_quality.py`) → `waste_common.onnx.U2NetSaliency` (Phase 2/5 사이, onnx 의존 확정 후)
- DINOv2 producer(`build_dinov2_classifier.py`)/consumer(`dinov2_classifier.py`) 간 입력 계약(`INPUT_SIZE`, `EMBED_DIM`) → `ModelArtifact` 매니페스트로 (Phase 4 이후)

---

## 7. 완료 정의 (DoD)

1. `waste_common` 7개 모듈이 존재하고 세 프로젝트가 모두 이를 import한다.
2. §1 표의 D1~D9 각 항목에 대해 "잔존 0" grep 이 PR에 기록되어 있다.
3. `api.py` ≤ 200줄, 가장 긴 함수 ≤ 60줄.
4. 세 프로젝트 + `waste-common`의 `pytest` 전부 통과, characterization 스냅샷이 Phase 0 시점과 동일(의도된 EXIF 변경 1건 제외).
5. `sys.path.insert`, `/Users/` 절대경로, `load_dotenv(형제 경로)` 가 코드베이스에 없다.

---

## 8. 진행 상태 (2026-08-30)

**범위 조정:** `waste-api`는 다른 Claude 세션이 자체 가이드(`waste-api/docs/REFACTORING_GUIDE.md`, `src/core`·`services`·`routers` 분해)로 동시 작업 중이라 **이번 작업에서 제외**. 아래는 `waste-common` + `waste-preprocessor` + `waste-classifier` 결과. api를 `waste-common`에 연결하는 것은 그쪽 작업 완료 후 후속.

| Phase | 상태 | 비고 |
|---|---|---|
| 0 안전망 | ✅ | `waste-common` 스켈레톤(+editable 설치), 루트 `ruff.toml`(기준선 53), characterization 2개(전처리 sha256, taxonomy 스냅샷) |
| 1 Supabase | ✅ | `create_client` 17→0 (`waste_common.supabase.get_client/try_get_client/Bucket`), `load_dotenv` 22→0 |
| 2 이미지 | ✅ | ImageNet 상수 21파일→0 (`waste_common.imaging`), preprocessor `load_rgb`에 **EXIF 보정 도입(동작 변경, 의도됨)** — 기존 `.npz` 벡터는 EXIF 있는 원본에 한해 값이 달라질 수 있음 |
| 3 설정/분류체계 | ✅(부분) | `taxonomy.py` → `waste_common`(shim 유지, 호출부 전부 전환 완료), 6-클래스 튜플 4→1(`LEGACY_LABELS`), 형제 경로→`settings`. **미적용:** classifier `config.refresh_classes_from_manifest()` import-time 갱신 제거 — 20파일 54곳 의존이라 별도 작업으로(백로그) |
| 4 api.py 분해 | ⏸ | 다른 세션 담당 |
| 5 classifier 포크 | ✅ | `src/artifacts.py`, `inverse_freq_weights`, `run_epoch` 통합(train/hier 래퍼), `scripts/_base.py`(sys.path.insert 16→1 +cross-repo 1), `integrate_aihub --dataset {71362,140}` 병합, `model_kind` 공개화. `paste_object` 2벌은 본문이 달라 미병합 |
| 6 로깅 | ✅(범위 한정) | classifier `src/`+최상위: print 255→118(남은 118은 리포트/표 출력), `except Exception` 25→14(11 → `fail_open`, 나머지 `# fail-open:` 주석). `scripts/`(1회성 CLI) 미적용. preprocessor 모듈 print → logging |
| 7 preprocessor | ✅ | `main.py` 죽은 `__main__` import 수정, `pipeline` 분기 통합, 기본값 `None` 패턴, storage/cleanse/pipeline 테스트 추가(16→23) |

**검증:** preprocessor 23 / waste-common 12 / classifier 35 passed · ruff 53→43 (신규 위반 0, E402 9→1) · 잔존 grep: `create_client`/`load_dotenv`/`0.485`/절대경로 모두 0.

**환경 메모:** Apple Silicon에서 셸이 Rosetta(x86_64)로 뜨면 venv(arm64 휠)와 충돌 — `arch -arm64 .venv/bin/python …`으로 실행. classifier `.venv/bin/ruff`는 x86_64 바이너리라 preprocessor venv의 ruff 사용.

**커밋:** 상위 저장소가 초기 커밋만 있고 소스 대부분이 untracked 상태라 Phase별 커밋 대신 **미커밋 상태로 둠** — 커밋 단위는 사용자 결정.

**Phase 5·6 편차:** `scripts/_base.py`가 `sys.path.insert` 1회를 수행(패키지 설치 방식 대신) — `src`를 배포 패키지명으로 설치하는 것이 더 어색하다고 판단. 71362 데이터셋 "zip 못 찾음" 경로가 `sys.exit` → `RuntimeError`로 통일됨.
