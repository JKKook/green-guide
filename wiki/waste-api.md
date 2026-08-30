# waste-api — FastAPI 추론 서버

> 소스: `waste-api/src/api.py`, `waste-api/src/*.py`, `waste-api/Dockerfile`, `waste-api/models/`, `waste-api/migrations/` (2026-08-13 탐색)

FastAPI + ONNX Runtime 기반 폐기물 이미지 분류 추론 서버. 사진 1장 → 한국형 분리배출 재질 분류 + 배출 안내. 4개 서브프로젝트 중 3번째 (→ [architecture-overview](architecture-overview.md)).

⚠️ **README는 V1(2026-05, 6클래스 flat) 시점 문서로 심하게 낡음.** `src/api.py`(1,478줄) 실물이 SSOT — 실제로는 계층 25클래스 + Supabase 완전 연동 + VLM 폴백 + OCR/CLIP 증거 융합까지 진화.

## 배포 — Hugging Face Spaces (Docker SDK)

- git remote가 곧 HF Spaces: `https://huggingface.co/spaces/ethanDev92/waste-api`. **git push = 자동 재배포.**
- `Dockerfile`: python:3.11-slim, PORT 7860. ONNX/npz는 git-lfs, `models/*.onnx`는 의도적 커밋(이미지 번들).
- `prepare_deploy.sh`: waste-classifier 산출물(`cnn_hier/{classifier.onnx, taxonomy.json, ood.npz}`)을 `models/`로 복사.
- 시크릿(`SUPABASE_URL/KEY`, `ANTHROPIC_API_KEY`)은 HF Spaces Secrets 주입.
- **torch 미설치** — onnxruntime 1.19.2 단일 엔진 (명시적 설계 결정).

## 엔드포인트 (단일 app, 태그로 그룹화)

- **meta**: `/` `/health` `/labels` `/taxonomy` `/region-info` `/model/latest` `/design/tokens.json`
- **inference**: **`/predict-hier`(메인, tap_x/tap_y 탭-투-셀렉트 지원)**, `/predict`(flat 레거시), `/predict-centered`(u2netp 크롭), `/predict-objects`(다중 객체), `/predict-with-cam`, `/predict-with-mask`, `/predict-with-regions`(다중재질), `/segment`
- **learning**: `/feedback` — **admin**: `/reload-classes`, `/admin/reload-model`
- CORS 전체 허용, 인증·rate limit 없음. 업로드 10MB 제한.

## `/predict-hier` 캐스케이드 (실제 순서)

1. EXIF orientation 정규화 + **GPS 등 EXIF 전량 제거**(프라이버시)
2. Stage 0: MediaPipe Hands 손 면적 ≥0.5 → non_object reject
3. Stage 1: `stage1_binary.onnx`(MobileNetV3-S 6MB) waste_prob <0.5 → reject (fail-open)
4. 탭 좌표 있으면 탭 지점 saliency 크롭, 없으면 **풀프레임**(u2 자동크롭은 v6+TTA에서 역효과 실측)
5. 회전 TTA (EXIF 태그 기반 축소, 평균 1.7×)
6. non_object 마스킹 (실측 +5.9pp)
7. **시맨틱 증거 융합** (log-linear): OCR(저확신<0.75일 때만), CLIP(탭 크롭에서만), CAM — → [semantic-fusion](semantic-fusion.md)
8. OOD 거부 (prototype 거리, → [ood-openset](ood-openset.md))
9. **VLM 폴백**: 게이트 reject/저확신/증거-불일치 시 Claude Haiku 위임 (일일 200회 캡, 프롬프트 캐싱, 768px 리사이즈)
10. Supabase `user_uploads` 기록 → upload_id 응답

## 모델 로딩 / OTA

우선순위: Supabase `model_versions` active row(SHA256 검증 캐시 다운로드) → env 경로 → `models/` 번들 → sibling 레포. **설계 원칙 "API는 항상 부팅한다"** — Supabase 불가 시 전부 silent fallback. 번들 모델 ~450MB (hier 90M, DINOv2 84M×2 — 현재 비활성, CLIP 84M, u2netp 4.4M, OCR 22M 등).

## Supabase 연동

테이블·버킷·쿼터 사태·local_feedback 폴백은 [supabase-infra](supabase-infra.md) 참조.

## 배출 스트림 닫힌 목록 (`src/streams.py`, 미커밋 작업 중)

"품목은 무한하지만 배출 목적지는 유한하다" — 환경부 지침 대조 23개 Stream 확정(2026-07-29). VLM이 사전 밖 품목을 생성해도 배출 안내는 이 목록에서만 선택. 새 스트림 발명 불가, 추가는 사람 승인.

## 디자인 토큰 역류

`design/tokens.json`(W3C Design Tokens draft) — [waste_app](waste-app.md)의 `app_theme.dart` 실측값을 API가 서빙(컴포넌트 24개 스펙 포함). 디자인 도구가 URL로 소비.

## 상태 (2026-08-13 조사 시점)

- HEAD `facc879` "무료 쿼터 지속성: 업로드 재압축 + 7일 자동 정리"
- 미커밋: 배출 스트림 기능(`src/streams.py`, `tests/test_vlm_streams.py`, vlm_fallback 수정) 작업 중
- 테스트 27개(pytest), CI 없음
