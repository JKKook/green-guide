# GreenGuide AI — 전체 아키텍처 개관

> 소스: 루트 계획 문서 8종, 4개 서브프로젝트 코드 (2026-08-13 탐색)

**그린가이드 AI**: 폐기물 사진 1장 → 한국형 분리배출 재질 분류 + 배출 안내를 주는 모바일 서비스. 원칙 한 줄 — **"항상 대분류는 맞힌다. 세부는 데이터가 허락하는 만큼만, 확신할 때만."**

## 파이프라인 (4 서브프로젝트)

```
waste-preprocessor ──▶ waste-classifier ──▶ waste-api ──▶ waste_app
 (수집·전처리·manifest)  (학습·ONNX export)   (FastAPI 추론)  (Flutter 앱)
        ▲                                        │
        └──────────── 피드백 재학습 루프 ◀────────┘  (Supabase user_uploads)
```

- [waste-preprocessor](waste-preprocessor.md) — Kaggle/AI-Hub/TACO 수집 → 정제 → manifest. 학습은 안 함.
- [waste-classifier](waste-classifier.md) — 계층 CNN(ResNet50) 학습 + ONNX export + 진단·게이트. git 아님.
- [waste-api](waste-api.md) — HF Spaces Docker 배포. ONNX Runtime 단일 엔진, 캐스케이드 추론 + VLM 폴백. 유일한 git 레포(remote=HF Spaces).
- [waste-app](waste-app.md) — Android 전용 Flutter. 온디바이스 ONNX + 클라우드 이원.

데이터 유입은 별도 스테이징 디렉터리 경유: [dataset-staging](dataset-staging.md).

## 이원 추론 구조

클라우드(전체 파이프라인: TTA·시맨틱 융합·VLM)와 온디바이스(단일 패스)가 **동일 분류기 가중치를 공유**. 차이는 파이프라인뿐. 지연: 클라우드 ~1.7s / 온디바이스 0.1–0.3s. 기본 모드는 클라우드, 온디바이스 저확신(<0.60) 시 클라우드 재판정.

## 앱↔서버 계약 7가지

1. **분류**: 앱 → `/predict-hier` (multipart + tap_x/tap_y) → 계층 응답
2. **클래스 메타**: 부팅 시 `/labels` → Supabase `waste_classes` (서버 SSOT, 앱 3단 캐시)
3. **모델 OTA**: `/model/latest` → `model_versions` → HF Hub `ethanDev92/waste-models` 다운로드. 앱 번들 모델 제거로 204→88MB (Play 200MB 한도)
4. **피드백 루프**: 앱 👍/👎 → `/feedback` → `user_uploads` → [retrain-loop](retrain-loop.md) → 새 ONNX publish → OTA
5. **지역 규정**: GPS/수동 선택 → `/region-info` → `region_waste_rules` (공공데이터포털)
6. **디자인 토큰 역류**: 앱 테마 실측 → `waste-api/design/tokens.json` → `GET /design/tokens.json`
7. **버전 스큐 방어**: `/predict-hier` 404/503 → flat 경로 자동 격하

## 인프라

- **Supabase** (Postgres + Storage): 업로드 로그·피드백·클래스 레지스트리·모델 버전·지역 규정 → [supabase-infra](supabase-infra.md)
- **Hugging Face**: Spaces(API 배포) + Hub(모델 저장소)
- **Anthropic API**: Claude Haiku 4.5 VLM 폴백 (일 200회 캡)

## 문서 지형

루트 계획 문서 8종은 세대가 다르고 뒤 문서가 앞 문서를 뒤집는다. **최신 정답지 = `docs/greenguide_model_methods.html` (2026-08-06) + ACCURACY_LATENCY_BLUEPRINT §0.** 상세: [planning-docs](planning-docs.md).
