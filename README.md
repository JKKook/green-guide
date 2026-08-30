# 그린가이드 (GreenGuide)

사진 한 장으로 분리배출 방법을 알려주는 AI 앱. 촬영한 사진의 재질을 계층 분류 모델(대분류 14 · 세부 25)로
판별하고, 거주 지역(시·군·구)의 배출 기준과 수거 일정을 함께 안내합니다.

- 플랫폼: Android (베타 `1.0.0-beta.1`) · macOS(개발 검증용)
- 추론: Hugging Face Space `ethanDev92/waste-api` (FastAPI) · 모델 원본 HF Hub `ethanDev92/waste-models`
- 데이터: Supabase (업로드·피드백·지역 배출 규정)

## 시스템 구성

```
apps/mobile (Flutter)
   │  사진 업로드 (긴 변 1600px JPEG)             피드백(정확함/수정 라벨)
   ▼                                                  ▼
services/inference-api (FastAPI, HF Space) ──────► Supabase
   │  /predict-hier · /predict-objects · /predict-with-regions      user_uploads(WebP 640px)
   │  /region-info · /feedback · /labels · /model/latest            region_waste_rules · model_versions
   ▼
HF Hub ethanDev92/waste-models/serving/  (빌드 시 다운로드: 계층 분류 ONNX·DINOv2·CLIP·OCR·u2netp·stage1)
   ▲
ml/classifier (학습·평가·ONNX 내보내기·발행)  ◄──  ml/preprocessor (수집·정제·벡터화)  ◄──  ml/data/raw
```

## 저장소 구조

```
green-guide/
├── apps/
│   └── mobile/               Flutter 앱 (패키지 greenguide) — 촬영·분석 결과·기록·설정
├── services/
│   └── inference-api/        FastAPI 추론 서버 — routers/·services/·core/, Dockerfile, migrations/
├── ml/
│   ├── classifier/           greenguide_classifier/ · scripts/ · tests/ — 학습·평가·ONNX·발행
│   ├── preprocessor/         greenguide_preprocessor/ · tests/ — 데이터 수집·정제·벡터화
│   └── data/raw/             AI-Hub·TACO·Open Images·합성 원본 (git 제외)
├── libs/
│   └── greenguide-common/    greenguide_common — 경로 설정·taxonomy·이미지 전처리·Supabase·로깅
├── docs/                     plans/(설계) · design/(UI/UX 시안) · BETA_RELEASE_CHECKLIST.md · 모델 방법론
├── wiki/                     프로젝트 지식 베이스 (llm-wiki)
├── bin/                      로컬 유틸 (aihubshell 등)
├── .github/workflows/        keep-alive (HF Space·Supabase 절전 방지)
├── CLAUDE.md                 세션 공통 규칙 — 브랜치·커밋·네이밍·범위
└── REFACTORING_GUIDE.md      폴더별 리팩토링 가이드·진행 기록
```

네이밍 규칙: 폴더 `kebab-case`, import 패키지·파일 `snake_case`, 접두어 `greenguide`.
외부 식별자(Android `applicationId` `com.greenguide.waste_app`, HF Space `waste-api`, Supabase 테이블명, 환경변수 `WASTE_*`)는 그대로 둡니다.
학습 데이터·모델 가중치·`.env`·서명 키는 저장소에 포함하지 않습니다 (`.gitignore`).

<details>
<summary>2026-08-30 이전 이름 ↔ 현재 위치</summary>

| 이전 | 현재 |
| --- | --- |
| `waste_app/` (패키지 `waste_app`) | `apps/mobile/` (패키지 `greenguide`) |
| `waste-api/` (자체 git, LFS 모델) | `services/inference-api/` (모노레포 subtree, 모델은 HF Hub) |
| `waste-classifier/` (`src`) | `ml/classifier/` (`greenguide_classifier`) |
| `waste-preprocessor/` (`src`) | `ml/preprocessor/` (`greenguide_preprocessor`) |
| `waste-common/` (`waste_common`) | `libs/greenguide-common/` (`greenguide_common`) |
| `*_staging/` | `ml/data/raw/<name>/` |

</details>

## 빠른 시작

전제: Flutter 3.44 / Dart 3.12, Python 3.12, 각 파이썬 프로젝트는 자체 `.venv` (Apple Silicon 에서는 `arch -arm64` 로 실행).

```bash
# 앱
cd apps/mobile && flutter pub get && flutter analyze && flutter test
flutter build apk --release            # 사이드로드
flutter build appbundle --release      # Play Console 업로드

# 추론 서버 (로컬)
cd services/inference-api && python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/fetch_models.py           # HF Hub serving/ → models/
.venv/bin/python main.py                            # http://localhost:8000/docs
.venv/bin/python -m pytest -q tests/test_preprocess.py tests/test_vlm_streams.py tests/test_uploads_recompress.py

# 공통 패키지 + ML
cd ml/classifier && python -m venv .venv && .venv/bin/pip install -r requirements.txt   # -e ../../libs/greenguide-common 포함
.venv/bin/python -m pytest -q
cd ml/preprocessor && .venv/bin/python -m pytest -q
```

환경변수는 `services/inference-api/.env`(SUPABASE_URL/KEY, HUGGINGFACE_TOKEN, ANTHROPIC_API_KEY, DATA_GO_KR_KEY)와
`ml/preprocessor/.env`(AIHUB_APIKEY 등)에 두며, `libs/greenguide-common/greenguide_common/settings.py` 가 둘 다 로드합니다.

## 개발 워크플로

| 브랜치 | 역할 |
| --- | --- |
| `main` | 배포 기준. 스토어에 올라간 버전만, 태그(`v1.0.0-beta.1` …) |
| `develop` | 통합 브랜치. 공유 작업 트리는 항상 이 브랜치 |
| `feature/<scope>-<topic>` | 실제 작업. `develop` 에서 분기 → PR/`merge --no-ff` → `develop` |

여러 세션이 같은 작업 트리를 쓰므로 `git checkout` 대신 **git worktree** 를 씁니다:

```bash
git worktree add .worktrees/feature-app-x -b feature/app-x develop
# 편집·테스트·커밋은 그 경로에서, 끝나면 develop 병합 후 git worktree remove
```

커밋은 `type(scope): 한글 제목 — 요약` 형식(`feat` `fix` `refactor` `docs` `test` `chore`, scope = `waste_app` `api` `classifier` `preprocessor` `repo` `wiki`). 자세한 규칙은 `CLAUDE.md`.

## 배포

- **서버**: 루트에서 `git subtree push --prefix=services/inference-api hf main` (remote `hf` = HF Space). 구조 변경 직후 첫 push 는 `--force`.
  Space 는 빌드 시 `ethanDev92/waste-models/serving/` 을 내려받으므로 모델 파일은 git 에 두지 않습니다.
- **모델**: `ml/classifier/scripts/publish_hier_version.py --apply` → HF Hub 업로드 + Supabase `model_versions` 등록 → `POST /admin/reload-model`.
- **앱**: `flutter build appbundle --release` → Play Console. 서명은 `apps/mobile/android/key.properties`(미커밋) 가 있을 때 업로드 키 사용.
- **절전 방지**: `.github/workflows/keepalive.yml` 이 12시간마다 HF Space·Supabase 를 핑합니다(리포 시크릿 `SUPABASE_URL`/`SUPABASE_ANON_KEY`).

베타 배포 전 점검·QA 기록은 `docs/BETA_RELEASE_CHECKLIST.md` 를 참고하세요.

## 데이터·모델 출처

재질 분류 모델은 과학기술정보통신부·한국지능정보사회진흥원(NIA) AI-Hub 의 "생활폐기물 활용·환류 데이터", "재활용 품목 이미지 데이터"와
TACO(CC BY 4.0), Open Images(CC BY 4.0) 등을 활용해 학습했습니다. 지역별 배출 정보는 행정안전부 "전국생활쓰레기배출정보표준데이터"(공공데이터포털)를 사용합니다.
앱 내 고지: 설정 › 앱 정보 › 오픈소스 라이선스.
