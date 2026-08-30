# 그린가이드 (GreenGuide)

사진 한 장으로 분리배출 방법을 알려주는 AI 앱. 촬영한 사진의 재질을 계층 분류 모델로 판별하고,
거주 지역(시·군·구)의 배출 기준과 수거 일정을 함께 안내합니다.

## 저장소 구조

2026-08-30 적용 완료 — ML 표준 레이아웃 + `apps`/`services` 분리:

```
green-guide/
├── apps/
│   └── mobile/               Flutter 앱 (Android) — 촬영·분석 결과·기록·설정
├── services/
│   └── inference-api/        FastAPI 추론 서버 — Hugging Face Spaces 배포, Supabase 연동
├── ml/                       모델 파이프라인
│   ├── classifier/           학습·평가·ONNX 내보내기·발행 (src/·scripts/·tests/)
│   ├── preprocessor/         데이터 수집·정제·벡터화 (src/·tests/)
│   └── data/raw/             AI-Hub·TACO·Open Images·합성 원본 (git 제외)
├── libs/
│   └── waste-common/         공통 패키지 — 설정(경로)·taxonomy·이미지 전처리·Supabase·로깅
├── docs/                     설계 문서(plans/)·UI/UX 시안(design/)·모델 방법론·배포 체크리스트
├── wiki/                     프로젝트 지식 베이스 (llm-wiki)
├── bin/                      로컬 유틸 스크립트
└── .github/workflows/        keep-alive 등 자동화
```

이전 이름 ↔ 현재 위치 (2026-08-30 이전 커밋·문서에서 옛 이름이 보이면 이 표로 읽으세요):

| 이전 | 현재 |
| --- | --- |
| `waste_app/` | `apps/mobile/` |
| `waste-api/` | `services/inference-api/` |
| `waste-classifier/` | `ml/classifier/` |
| `waste-preprocessor/` | `ml/preprocessor/` |
| `waste-common/` | `libs/waste-common/` |
| `*_staging/` | `ml/data/raw/<name>/` (git 제외) |

다음 단계(선택): `ml/classifier`+`ml/preprocessor`를 `ml/src/greenguide_ml` 단일 패키지로 통합, `configs/`·`experiments/` 도입.

학습 데이터·모델 가중치·`.env`·서명 키는 저장소에 포함하지 않습니다 (`.gitignore` 참고).

## 브랜치 전략

| 브랜치 | 역할 |
| --- | --- |
| `main` | 배포 기준. 스토어에 올라간 버전만 존재하며 태그(`v1.0.0-beta.1` …)로 표시 |
| `develop` | 통합 브랜치. 기능 브랜치가 여기로 합쳐지고, 배포 시 `main` 으로 승격 |
| `feature/*` | 기능·수정 단위 작업 브랜치. `develop` 에서 분기 → PR → `develop` 으로 병합 |

긴급 수정은 `main` 에서 `hotfix/*` 로 분기해 `main` 과 `develop` 양쪽에 반영합니다.

## 구성 요소별 안내

각 폴더의 README 를 참고하세요.

- 앱 빌드: `apps/mobile/` — Flutter 3.44 / Dart 3.12, `flutter build appbundle --release`
- 서버 배포: `services/inference-api/` — Hugging Face Space `ethanDev92/waste-api`. 모노레포에서 `git subtree push --prefix=services/inference-api hf main` (구조 변경 직후 첫 push 는 `--force` 필요 — split 이력이 새 prefix 부터 시작) (remote `hf` = Space URL). 서빙 모델은 git 이 아니라 HF Hub `ethanDev92/waste-models/serving/` 에서 빌드 시 다운로드
- 모델 발행: `ml/classifier/scripts/publish_hier_version.py` — HF Hub `ethanDev92/waste-models`
