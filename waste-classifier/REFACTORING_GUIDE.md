# waste-classifier 리팩토링 작업 가이드

초점: **코드 품질 최적화 + 공통단(common layer) 구성**.
동작 변경 없음(behavior-preserving)이 대전제 — 모델 정확도·산출물 경로·CLI 인터페이스는 리팩토링 전후 동일해야 한다.

---

## 0. 현황 진단 (2026-08-30 grep 기준)

| 중복 패턴 | 규모 | 대표 위치 |
|---|---|---|
| `sys.path.insert(...)` 경로 해킹 | 15 파일 | `scripts/*` 거의 전부, `retrain_hier.py` |
| `ort.InferenceSession(x, providers=["CPUExecutionProvider"])` | 18 파일 / 25회 | `revalidate.py`, `eval_ensemble*.py`, `scripts/build_*`, `scripts/synthesize_*` |
| ImageNet mean/std 상수 직접 기술 | 16 파일 | `src/dataset.py`, `src/ood.py`, `visualize_*.py`, `scripts/*` |
| `_softmax` 자체 구현 | 5곳 | `revalidate`, `eval_ensemble`, `eval_ensemble_weighted`, `etc_queue`, `visualize_multimaterial` |
| `create_client(os.getenv("SUPABASE_URL"), ...)` | 8곳 | `retrain.py`(4회), `revalidate`, `realworld_eval`, `retrain_hier`, `etc_queue` |
| PIL 이미지 로드/URL 다운로드 | 24 파일 | `etc_queue._download_image`, `scripts/integrate_taco.download_image` 등 |
| device 선택(cuda/mps/cpu) | 7 파일 | `src/train.py`, `visualize_cam.py`, `scripts/build_dinov2_*` |
| 진입점 스크립트 (각자 argparse) | 루트 13 + scripts/ 27 = 40개 | — |
| 테스트 커버리지 | `src/` 5개 모듈만 (dataset/split/model/hierarchy) | 루트·scripts 는 0 |
| git | **83 파일 untracked** | baseline 없음 |

결론: "src 는 라이브러리, 루트·scripts 는 실험 스크립트" 라는 의도는 있으나 공통 인프라(추론·I/O·외부 서비스)가 스크립트마다 재구현되어 있다. 공통단은 **새 추상화를 발명하는 게 아니라 이미 5~25번 복붙된 코드를 한 곳으로 모으는 작업**이다.

---

## 1. 작업 원칙

1. **Baseline 먼저** — untracked 83개를 커밋해 리팩토링 diff 가 분리되게 한다. 커밋 없이는 "동작 동일" 검증 불가.
2. **Characterization test 먼저, 이동은 나중** — 옮기려는 코드의 현재 출력을 고정(golden)해 두고 옮긴다.
3. **Surgical** — 한 PR/커밋 = 한 중복 패턴. 공통화하면서 로직 "개선" 금지. 기존 버그를 발견하면 별도 이슈로 기록만.
4. **호출부까지 교체해야 완료** — 공통 함수 추가만 하고 원본을 남기면 중복이 늘어난 것. 원본 삭제까지가 한 단위.
5. **YAGNI** — 2곳 미만에서 쓰이는 코드는 공통단으로 올리지 않는다. 실험 1회용 스크립트(`scripts/_tau_check.py` 등)는 손대지 않거나 `scripts/archive/` 로 이동만.
6. **검증 게이트** — 매 단계 `pytest` + `pyright` + (해당 시) golden 비교 통과 후 다음 단계.

---

## 2. 목표 구조 (공통단)

```
src/
├── config.py            # 유지. 경로·하이퍼파라미터만 (Supabase 등 외부 의존 X)
├── taxonomy.py          # 유지 (계층 정본)
├── common/              # ★ 신설 — 5회 이상 복붙된 인프라 코드만
│   ├── __init__.py
│   ├── device.py        # get_device() -> torch.device  (cuda > mps > cpu)
│   ├── image.py         # IMAGENET_MEAN/STD, load_image(path|bytes), download_image(url, retries), to_input_tensor(img) -> np/torch
│   ├── onnx.py          # load_session(path) , run(sess, x) -> logits, softmax(logits, axis=-1)
│   ├── supabase.py      # get_client() 1회 생성 캐시 + 환경변수 검증(누락 시 명확한 에러), 자주 쓰는 쿼리 헬퍼(user_uploads 조회)
│   ├── paths.py         # (선택) outputs/ 하위 경로 조립 — config.arch_subdir 와 통합 검토
│   └── cli.py           # 공통 argparse 부모 파서 (--model, --device, --limit, --dry-run)
├── dataset.py …         # 기존 모듈은 common 을 import 하도록 교체
scripts/                 # python -m scripts.<name> 으로 실행. sys.path 해킹 제거
pyproject.toml           # ★ 신설 — src 패키지 editable 설치 + ruff 설정
```

**넣지 말 것**: 모델 정의(`model.py`에 있음), 학습 루프, 도메인 로직(taxonomy). common 은 "어느 ML 프로젝트에 옮겨도 그대로 쓰이는 코드"만.

---

## 3. 단계별 계획

각 단계는 `[작업] → verify: [확인 방법]` 형식. 순서는 의존성 순.

### Phase 0 — Baseline & 도구 (0.5일) ✅ 2026-08-30 완료
- [x] baseline 커밋 `634025c` — waste-classifier/ 디렉터리만(89 파일). monorepo 의 waste_app·waste-api·wiki 는 범위 밖이라 untracked 유지
- [x] `pyproject.toml` + `pip install -e .` (`5cfc193`) — `scripts/__init__.py` 추가, ruff 규칙 `E,F,I,B,UP` (E501 은 ignore)
- [x] `ruff --fix` 안전 수정 97건 적용 (147 → 50건 잔여). 잔여 50건 내역: E702 14 / B905 12 / E402 9(sys.path 해킹 → 2-1 에서 해소) / B007 6 / E741 5 / F841 3 / E701 1
- [x] pytest 32 passed, **pyright 기준선: 205 errors, 9 warnings** (`pyright src scripts *.py tests`)

> 환경 메모: Rosetta(x86_64) 셸에서 `.venv/bin/python` 을 실행하면 universal 바이너리의 x86_64 슬라이스가 선택돼 numpy(arm64) import 가 실패한다.
> Claude Code Bash 세션 등 i386 셸에서는 `arch -arm64 .venv/bin/python -m pytest` 로 실행. 일반 터미널(arm64)은 영향 없음.

**Phase 0 검증 기록**
- ruff 가 제거한 import 20개 이름 → 각 파일에서 잔여 참조 0건 (부수효과 import 없음, `timezone` 은 `UTC` 로 대체)
- 진입점 39개 `--help` 스모크: 37 통과 / 2 실패 — 두 건 모두 baseline 에서 동일 재현되는 **기존 버그** (아래)
- pytest 32 passed

**리팩토링 중 발견한 기존 버그 (범위 밖, 별도 수정 필요)**
- `scripts/filter_aihub_by_quality.py:222` — argparse help 문자열의 `18%)` 가 `%` 포맷으로 해석돼 `--help` 가 ValueError. `%%` 로 이스케이프 필요
- `scripts/synthesize_indoor.py` — `albumentations` 가 requirements.txt 에 없어 import 실패 (실험용이면 Phase 3 에서 archive 이동 대상)

### Phase 1 — Characterization tests (1일)
공통화 대상 5개 패턴의 현재 동작을 고정한다. 이 테스트가 이후 모든 단계의 안전망.
- [ ] `tests/test_golden_inference.py`: 고정 이미지 3장(클래스 다름) → 현행 ONNX 모델 logits 를 `tests/fixtures/golden_logits.npz` 로 저장. 테스트는 `np.allclose(atol=1e-5)`
  → verify: 테스트 통과, fixture 커밋
- [ ] `_softmax` 5개 구현 각각에 같은 입력 → 동일 출력인지 확인하는 임시 스크립트 실행 (차이 있으면 **여기서 발견**, 공통화 전에 어떤 걸 정본으로 할지 결정)
- [ ] 전처리 16곳 중 resize 방식(`BILINEAR`/`BICUBIC`), center-crop 유무가 다른 곳을 표로 정리 → 정본 1개 + 예외 목록
  → verify: 예외가 있으면 `to_input_tensor(img, *, crop=...)` 처럼 **명시 옵션**으로 남기고, 조용히 통일하지 않는다
- [ ] Supabase 헬퍼는 네트워크 필요 → `tests/conftest.py` 에 `create_client` 를 monkeypatch 하는 fake 추가

### Phase 2 — 공통단 추출 (2~3일, 패턴당 1커밋)
우선순위 = 중복 횟수 × 변경 위험 낮음 순.

| 순서 | 패턴 | 작업 | verify |
|---|---|---|---|
| 2-1 | `sys.path` 해킹 15곳 | Phase 0 editable 설치 덕에 삭제만. `scripts/__init__.py` 추가, README 실행법을 `python -m scripts.x` 로 | 각 스크립트 `--help` 실행 성공 |
| 2-2 | ONNX 세션 + softmax | `common/onnx.py` 작성 → 25회 호출부 교체, `_softmax` 5개 삭제 | golden 테스트 + 삭제된 함수 grep 0건 |
| 2-3 | ImageNet 상수/전처리 | `common/image.py` → 16곳 교체. `src/dataset.py` 는 torchvision transform 을 유지하되 상수만 import | golden 테스트, `test_dataset_cnn` 통과 |
| 2-4 | 이미지 로드/다운로드 | `download_image(url, retries, timeout)` 하나로 (`etc_queue`·`integrate_taco` 버전 중 retry 있는 쪽 채택) | fake requests 로 retry 단위테스트 |
| 2-5 | Supabase client | `common/supabase.py` `get_client()` + `fetch_user_uploads(status=...)`. `retrain.py` 4회 → 1회 | conftest fake 로 테스트, `grep -rn create_client` 가 common 1곳만 |
| 2-6 | device 선택 | `common/device.py` → 7곳 교체 | `src/train.py` smoke (1 epoch, tiny subset) |
| 2-7 | argparse 공통 옵션 | `common/cli.py` 부모 파서 — **옵션 이름이 이미 같은 스크립트만** 적용. 인터페이스 변경 금지 | 각 `--help` 출력 diff 없음 |

### Phase 3 — 스크립트 정리 (1일)
- [ ] 루트 13개 진입점 분류: 운영 파이프라인(`main`, `retrain*`, `revalidate`, `feedback_monitor`, `etc_queue`) / 분석 도구(`diagnose`, `visualize_*`, `eval_ensemble*`, `realworld_eval`)
  → 운영은 루트 유지, 분석은 `scripts/` 로 이동. `git mv` 사용(히스토리 보존)
- [ ] `scripts/` 27개 중 데이터 통합 완료된 1회용(`integrate_*`, `extend_manifest_*`, `extract_bg_140`) → `scripts/archive/` 이동. 삭제 X (재현성)
- [ ] `eval_ensemble.py` vs `eval_ensemble_weighted.py` — 가중치 인자 하나로 합칠 수 있으면 통합, 아니면 그대로
  → verify: README/HIER_TRAINING_GUIDE 의 실행 명령 전부 갱신 후 실제 실행

### Phase 4 — 품질 규칙 고정 (0.5일)
- [ ] `config.py` 의 import-time 부수효과(`refresh_classes_from_manifest()` 자동 호출, `print`) 를 **명시 호출**로 바꿀지 결정. 바꾼다면 호출부 전수 확인 — 리스크 있으니 별도 PR
- [ ] `print` 로깅 → `logging` 전환은 **이번 범위 밖**. 공통단 신규 코드만 `logging` 사용
- [ ] `ruff check` 를 pytest 앞에 두는 `Makefile`/`scripts/check.sh` 1개
- [ ] 이 문서의 진단 표를 최종 수치로 갱신 (목표: InferenceSession 직접 호출 0, softmax 구현 1, create_client 1, sys.path 0)

---

## 4. 완료 기준 (Definition of Done)

```
grep -rn "sys.path"                  --include='*.py' . | wc -l   # 0
grep -rn "InferenceSession("         --include='*.py' . | wc -l   # 1 (common/onnx.py)
grep -rn "def _softmax"              --include='*.py' . | wc -l   # 0
grep -rn "create_client("            --include='*.py' . | wc -l   # 1 (common/supabase.py)
grep -rn "0.485"                     --include='*.py' . | wc -l   # 1 (common/image.py)
pytest && ruff check . && pyright                                 # 통과, pyright 에러 수 ≤ Phase 0 기준선
python tests/test_golden_inference.py                             # golden 일치
```
+ README·HIER_TRAINING_GUIDE 의 실행 명령이 전부 실제로 동작.

---

## 5. 하지 말 것

- 공통화하면서 전처리 세부(resize 보간, crop)를 "통일" — 정확도가 바뀐다. 차이는 옵션으로 드러낸다.
- `src/taxonomy.py`, `src/hier_*` 도메인 로직 손대기 — 이번 범위는 인프라 공통단.
- 이름만 바꾸는 리네임, 스타일 통일 커밋을 기능 커밋과 섞기.
- 스크립트 삭제 — 실험 재현성 때문에 archive 이동까지만.
- 테스트 없는 상태에서 `retrain.py`(518줄, Supabase 4회 + 학습 + 업로드) 를 한 번에 쪼개기 — 2-5 에서 client 만 빼고, 함수 분리는 별도 작업.

---

## 6. 예상 소요 & 순서 요약

Phase 0(0.5d) → 1(1d) → 2(2~3d) → 3(1d) → 4(0.5d) ≈ **5~6일**.
Phase 2 각 항목은 독립이라 순서 바꿔도 되나, 2-1(경로) 은 다른 모든 것의 전제.
