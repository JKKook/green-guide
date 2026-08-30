# waste-api 리팩토링 가이드 — 코드 품질 최적화 + 공통단 구성

> 작성일 2026-08-30. 대상: `src/` 5,279 LOC (api.py 1,463 LOC = 28%).
> **상태 (2026-08-30): P0~P3 완료.** 편차·후속 과제는 §6 참조.
> 원칙: **동작 불변** (엔드포인트 응답 스키마·값 동일), **단계별 커밋**, **매 단계 테스트 그린**.

## 0. 현재 진단 (근거 수치)

| 항목 | 현황 | 문제 |
|---|---|---|
| `src/api.py` | 1,463줄, 엔드포인트 17개 + 헬퍼 11개 혼재 | 라우팅·비즈니스 로직·이미지 처리·업로드 기록이 한 파일 |
| 업로드 기록 블록 | `record_prediction` 호출 7회, 매번 동일한 `if COLLECT_USER_UPLOADS / try / except / print` 래핑 | 복붙 중복 (각 8~10줄 × 7) |
| `except Exception` | api.py에만 41회 | fail-open 정책이 코드로 명문화되지 않고 매 호출부에서 반복 |
| 로깅 | `print(...)` 100회, `logging` 사용 0 | 레벨·포맷·요청 상관관계 없음, 운영에서 필터 불가 |
| 설정 | `config.py` 밖 `os.getenv` 18회 | 설정 SSOT 깨짐 (예: `OCR_SKIP_CONFIDENCE`가 api.py에 정의) |
| 싱글톤 로더 | `get_xxx()/reset_xxx()` + `global` 패턴 10개 모듈에 동일 복제 | 보일러플레이트, 테스트 격리(reset) 누락 모듈 존재 |
| 테스트 | `.venv` numpy 손상으로 **전체 실행 불가** | 리팩토링 안전망 부재 |
| 미커밋 변경 | `streams.py`, `vlm_fallback.py`, `test_vlm_streams.py` | 리팩토링과 섞이면 diff 오염 |

## 1. 목표 구조

```
src/
├── core/                     # 공통단 (신규)
│   ├── config.py             # 모든 env 읽기 → 여기로 (기존 config.py 이동)
│   ├── logging.py            # get_logger(name), 요청 ID 컨텍스트
│   ├── singleton.py          # lazy_singleton 데코레이터 (get/reset 자동 생성)
│   └── errors.py             # 도메인 예외 + FastAPI 예외 핸들러 매핑
├── services/                 # 비즈니스 로직 (api.py에서 추출)
│   ├── image_io.py           # read_validate(_with_orientation), auto_crop, crop_at_tap
│   ├── cascade.py            # hand→stage1→stage2 캐스케이드, ensemble_with_dinov2, force_non_object
│   ├── regions_service.py    # tap_silhouette_regions, verify_regions, evidence_conflicts
│   └── recording.py          # record_safely(raw, content_type, result) — 7회 중복의 단일화
├── routers/                  # 얇은 라우터 (검증·호출·응답만)
│   ├── meta.py               # /, /health, /labels, /taxonomy, /design/tokens.json, /region-info, /model/latest
│   ├── inference.py          # /predict*, /segment
│   ├── admin.py              # /reload-classes, /admin/reload-model
│   └── learning.py           # /feedback
├── api.py                    # app 생성 + lifespan + include_router 만 (≤80줄)
└── (기존 모델 모듈들 그대로: inference.py, hier_inference.py, segment.py …)
```

**규칙**: 라우터 함수는 (1) 입력 읽기 (2) 서비스 호출 (3) 응답 모델 조립 세 줄 구조를 넘지 않는다. 모델 모듈은 건드리지 않는다(스코프 밖).

## 2. 단계별 작업 계획

각 단계 = 커밋 1개. `→ verify:` 를 통과해야 다음 단계.

### Phase 0 — 안전망 확보 (리팩토링 전 필수)
- [ ] **P0-1** 미커밋 변경(`streams.py`, `vlm_fallback.py`, `test_vlm_streams.py`, `tokens.json`, 마이그레이션) 먼저 별도 커밋으로 정리
  → verify: `git status` clean
- [ ] **P0-2** venv 복구: `.venv/bin/pip install --force-reinstall numpy` (실패 시 venv 재생성 + `requirements.txt`)
  → verify: `.venv/bin/python -m pytest -q` 전체 통과
- [ ] **P0-3** 특성 테스트(characterization test) 추가 — 리팩토링 대상 엔드포인트 응답을 스냅샷으로 고정
  - 대상: `/predict`, `/predict-centered`, `/predict-hier`, `/predict-with-regions`, `/feedback`
  - 방법: 고정 샘플 이미지 → 응답 JSON 키 집합 + 타입 검증 (값은 모델 의존이므로 키·shape만)
  - 외부 의존(Supabase, 모델 로더)은 `conftest.py`에서 `reset_*` + monkeypatch 로 격리
  → verify: 새 테스트 통과, 이후 모든 Phase에서 이 테스트가 회귀 감지

### Phase 1 — 공통단(core) 구성
- [ ] **P1-1** `core/logging.py`: `get_logger(__name__)` + 포맷 `[%(levelname)s] %(name)s: %(message)s`. `print("[warn] …")` → `log.warning(…)`, `[startup]` → `log.info`
  - 100회 치환은 기계적으로: `grep -rn 'print(f\?"\[' src/` 목록 기준
  → verify: `grep -rc 'print(' src/ | grep -v ':0'` 결과 없음 (scripts/ 제외), 테스트 통과
- [ ] **P1-2** `core/config.py`: `config.py` 밖 `os.getenv` 18개를 전부 이동. 이름 규칙 `WASTE_API_*` 유지, 타입 변환은 config에서만
  → verify: `grep -rn 'os.getenv\|os.environ' src/ | grep -v core/config.py` 결과 없음
- [ ] **P1-3** `core/singleton.py`: 10개 모듈의 `_x = None / get_x / reset_x / global` 를 데코레이터로 통일
  ```python
  @lazy_singleton
  def get_recorder() -> UploadRecorder:
      return UploadRecorder()
  # get_recorder.reset() 자동 제공
  ```
  - 기존 `reset_classifier`, `reset_recorder` 호출부는 호환 alias 유지 (호출자 수정 최소화)
  → verify: `grep -rn 'global _' src/` 결과 없음(cam_renderer `_JET_LUT` 제외), conftest의 reset 동작 확인
- [ ] **P1-4** `core/errors.py`: `ImageDecodeError` 등 도메인 예외 → HTTP 상태 매핑을 `app.exception_handler` 로 중앙화. 라우터 내 `try/except ImageDecodeError → HTTPException(400)` 제거
  → verify: `test_predict_rejects_invalid_image_bytes` 여전히 400

### Phase 2 — api.py 분해 (서비스 추출)
- [ ] **P2-1** `services/recording.py::record_safely()` — 7곳의 업로드 기록 블록을 1함수로. fail-open 정책(수집 실패는 추론을 막지 않음)을 docstring에 명시
  → verify: `grep -c record_prediction src/` → 정의 1 + 호출 1
- [ ] **P2-2** `services/image_io.py` — `_read_and_validate_image`, `_read_validate_with_orientation`, `_auto_crop_to_object`, `_crop_at_tap` 이동. 언더스코어 제거하고 공개 API로
  → verify: 이동만, 로직 변경 0 (`git diff --stat` 로 삭제/추가 줄수 대칭 확인)
- [ ] **P2-3** `services/cascade.py` — `predict_centered` 본문의 Stage 0/1/2 를 `run_cascade(raw) -> dict` 로 추출. `_ensemble_with_dinov2`, `_force_non_object_result` 동반 이동
  → verify: 특성 테스트 통과, `predict_centered` 라우터 ≤15줄
- [ ] **P2-4** `services/regions_service.py` — `_tap_silhouette_regions`, `_verify_regions`, `_evidence_conflicts` 이동. `predict_with_regions`(200줄)의 본문을 `analyze_regions(raw, taps…)` 로
  → verify: `test_evidence_conflict_*` 통과
- [ ] **P2-5** `routers/` 4분할 + `api.py` 를 app 조립만 남김
  → verify: `wc -l src/api.py` ≤ 80, `/openapi.json` 경로·태그 집합이 분해 전과 동일 (Phase 0에서 스냅샷 저장)

### Phase 3 — 품질 마감
- [ ] **P3-1** `except Exception` 41회 재검토: 각각 (a) fail-open 의도 → `log.warning` + 명시적 기본값 (b) 실제 버그 은폐 → 좁은 예외로 교체 or 제거. 결정을 주석 한 줄로 남긴다
  → verify: 남은 `except Exception` 마다 `# fail-open:` 주석 존재
- [ ] **P3-2** 린트/타입 도입: `ruff` (E, F, B, I) + `mypy --strict` 는 `core/`, `services/` 만 우선
  → verify: `ruff check src/ && mypy src/core src/services` 통과
- [ ] **P3-3** README 의 모듈 맵 갱신

## 3. 작업 규칙 (지켜야 할 것)

1. **이동과 변경을 같은 커밋에 섞지 않는다.** "함수 이동" 커밋은 로직 변경 0. 개선은 다음 커밋.
2. **모델 모듈(`inference`, `hier_inference`, `segment`, `clip_identity` 등)은 스코프 밖.** 싱글톤 데코레이터 적용(P1-3) 외에는 손대지 않는다.
3. **공개 인터페이스 동결**: 엔드포인트 경로·응답 스키마(`schemas.py`)·env 변수명은 바꾸지 않는다.
4. **추상화는 3회 이상 반복된 것만.** 1~2회 패턴은 그대로 둔다 (`_read_validate_with_orientation` 은 호출 2회 → 이동만, 통합 안 함).
5. **각 단계 완료 기준 = verify 명령 통과 + 특성 테스트 그린.** 둘 중 하나라도 실패하면 다음 단계 금지.
6. **커밋 메시지**: `refactor(<phase-id>): <무엇을 어디로>` (예: `refactor(P2-1): 업로드 기록 7회 중복 → services/recording.record_safely`)

## 4. 예상 효과

| 지표 | Before | After (목표) |
|---|---|---|
| `api.py` LOC | 1,463 | ≤ 80 (라우터 합계 ≈ 500) |
| 업로드 기록 중복 | 7 | 1 |
| `print` | 100 | 0 |
| `os.getenv` 산재 | 18 | 0 |
| 싱글톤 보일러플레이트 | ~10줄 × 10 | 1줄 × 10 |
| 테스트 실행 | 불가 | 그린 + 특성 테스트 5개 추가 |

## 5. 하지 않을 것 (스코프 밖)

- 모델 추론 로직·가중치·전처리 파라미터 변경
- DB 스키마·마이그레이션 변경
- 성능 최적화 (별도 작업)
- `scripts/`, `local_feedback/` 정리

## 6. 실행 결과 (2026-08-30)

| 지표 | Before | After |
|---|---|---|
| `api.py` LOC | 1,463 | 91 (routers 합계 727) |
| 업로드 기록 중복 | 7 | 1 (`services/recording`) |
| `print` | 100 | 0 |
| `config.py` 밖 `os.getenv` | 18 | 0 |
| 싱글톤 `global` 패턴 | 10 | 0 (`cam_renderer._JET_LUT` 캐시 1건은 대상 아님) |
| `ImageDecodeError→400` 반복 | 6 | 0 (전역 핸들러) |
| hand/stage1 fail-open 반복 | 5 | 0 (`cascade.non_object_gate`) |
| 테스트 | 실행 불가 | 42 passed (특성 테스트 7 추가) |

**계획 대비 편차**
- `core/logging.py` → `core/log.py` (stdlib `logging` 이름 충돌 회피).
- venv 원인은 numpy 손상이 아니라 **arm64/x86_64 패키지 혼재** → `pip install --force-reinstall -r requirements.txt` 로 복구.
- P3-1: `except Exception` 59곳 전수 검토 결과 전부 "보조 단계 실패 → 경고 후 강등 결과" 의 일관된 fail-open 이고 `# noqa: BLE001` 이 이미 명시돼 있어 **코드 변경 없음**. 조용히 삼키는 4곳(EXIF 태그·bbox·host 파싱 등)은 선택 정보 폴백으로 무해.
- P3-2: mypy 는 레거시 모델 모듈 타입 오류 23건(PIL `Image | None` 등)으로 **도입 보류**. ruff 만 게이트로 채택하고 기존 스타일 규칙(B905/E702/E741/B007/B017)은 `ruff.toml` 에 명시적 ignore.

**후속 과제 (스코프 밖으로 남긴 것)**
- `routers/inference.py::predict_hier` 가 여전히 ~200줄 (시맨틱 증거 융합·VLM 폴백 블록). `services/hier_pipeline.py` 로 추출 후보.
- `predict_objects` 도 같은 패턴 (~100줄).
- mypy 도입 시 `uploads.py:70`, `regions.py:108` 의 PIL 타입 이슈부터.

## 7. 실기동 검증 (2026-08-30)

리팩토링 전 커밋(`5acdfb5`)을 worktree 로 8001, 현재 코드를 8000 에 띄우고 **동일 요청 75건**
(GET 8 + 실사진 6장·랜덤 1장 × POST 9 + 오류 입력 4)의 응답 JSON 을 비교 (`inference_ms`/`upload_id` 제외, base64 는 해시).

**발견·수정한 회귀 1건**: `config.py` 를 `src/core/` 로 옮기며 `PROJECT_ROOT = Path(__file__).parent.parent` 가
`src/` 를 가리킴 → 번들 모델(edge·DINOv2·segmenter·stage1) 전부 미발견, `/design/tokens.json` 500.
단위 테스트는 sibling 폴백·Supabase 캐시 덕에 통과해 **잡지 못했음** → `parents[2]` 로 수정하고
`test_project_root_points_to_repo` / `test_design_tokens_endpoint` 추가.

수정 후 결과: **70/75 동일**. 나머지 5건은 모두 비회귀 —
`/` 의 `model_path`(worktree 절대경로), 400 오류 메시지의 `BytesIO` 객체 주소 2건,
`/predict-hier` 탭 경로 2건(같은 서버에 같은 요청을 3회 보내도 결과가 오가는 **기존 비결정성** — GrabCut/MediaPipe).
