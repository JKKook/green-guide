# 그린가이드 (GreenGuide)

사진 한 장으로 분리배출 방법을 알려주는 AI 앱. 촬영한 사진의 재질을 계층 분류 모델로 판별하고,
거주 지역(시·군·구)의 배출 기준과 수거 일정을 함께 안내합니다.

## 저장소 구조

> 2026-08-30 기준 **이전 진행 중**. 각 구성 요소는 폴더별 리팩토링이 끝나는 대로 아래 목표 구조로 옮깁니다.

목표 구조 (ML 표준 레이아웃 + `apps`/`services` 분리):

```
green-guide/
├── apps/
│   └── mobile/               Flutter 앱 (Android) — 촬영·분석 결과·기록·설정
├── services/
│   └── inference-api/        FastAPI 추론 서버 — Hugging Face Spaces 배포, Supabase 연동
├── ml/                       모델 파이프라인 (데이터 수집·전처리 → 학습 → 평가 → ONNX 내보내기 → 발행)
│   ├── configs/              하이퍼파라미터·taxonomy·경로
│   ├── data/                 raw / interim / processed (git 제외)
│   ├── src/greenguide_ml/    설치 가능한 패키지 (data · datasets · models · training · evaluation · export · registry)
│   ├── scripts/              1회성 데이터 통합·실험 스크립트
│   ├── experiments/          학습 산출물·로그 (git 제외)
│   └── tests/
├── docs/                     설계 문서(plans/)·UI/UX 시안(design/)·모델 방법론·배포 체크리스트
├── wiki/                     프로젝트 지식 베이스 (llm-wiki)
├── bin/                      로컬 유틸 스크립트
└── .github/workflows/        keep-alive 등 자동화
```

현재 위치 ↔ 목표 위치:

| 현재 | 목표 |
| --- | --- |
| `waste_app/` | `apps/mobile/` |
| `waste-api/` | `services/inference-api/` |
| `waste-classifier/` + `waste-preprocessor/` | `ml/` (패키지 하나로 통합) |
| `*_staging/` (AI-Hub·TACO·Open Images·합성) | `ml/data/raw/` (git 제외) |

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

- 앱 빌드: `waste_app/` — Flutter 3.44 / Dart 3.12, `flutter build appbundle --release`
- 서버 배포: `waste-api/` — Hugging Face Space `ethanDev92/waste-api` (push = 배포)
- 모델 발행: `waste-classifier/scripts/publish_hier_version.py` — HF Hub `ethanDev92/waste-models`
