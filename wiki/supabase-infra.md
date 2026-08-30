# Supabase 인프라 · 무료 쿼터 운영

> 소스: `waste-api/src/{uploads,classes,model_loader}.py`, `waste-api/migrations/`, `waste-api/scripts/`, `docs/greenguide_model_methods.html` §8 (2026-08-13 탐색)

## 테이블 / 버킷

| 테이블 | 용도 |
|---|---|
| `user_uploads` | 추론 로그 + 피드백 (predicted_class, all_probabilities, feedback_status/label) |
| `waste_classes` | 동적 클래스 레지스트리 (slug/display_name/bin/how_to/level/parent_slug/active…) — 앱·서버의 클래스 메타 SSOT |
| `model_versions` | ONNX 버전 레지스트리 (sha256, is_active) — 서버 로딩·앱 OTA 공유 |
| `model_diagnostics` | 버전별 진단 지표 ([retrain-loop](retrain-loop.md)) |
| `region_waste_rules` | 지자체 배출 규정 (공공데이터포털 data.go.kr 15155080, 주 1회 갱신 권장) |

버킷: `user-uploads`(비공개, service key), `models`, `raw-images`.

## 2026-07-21 쿼터 초과 사태 → 프로젝트 이사

무료 티어 1GB 초과(models 버킷에 구버전 25개 981MB 누적) → 서비스 전면 제한(피드백 수집·지역정보 정지) → **신규 프로젝트로 이사**. 산물:

- `migrations/_bootstrap_new_project.sql` — 이사 킷(기반 테이블 + 001~011 통합)
- `scripts/bootstrap_new_supabase.py` — 버킷 생성 + 피드백 복원 + 지역 규정 재적재 자동화
- `local_feedback/` — **Supabase 장애 시 로컬 폴백 저장소** (현재 182장 + meta.jsonl + vlm_labels.jsonl + item_candidates.jsonl). "피드백 축적(A1)이 인프라 장애에 멈추지 않게". 컨테이너 디스크는 휘발성이라 보조 수단.

## 지속성 장치 (재발 방지)

- **업로드 재압축**: 저장 전 720px q80 (~9배 절약, 커밋 facc879)
- **자동 정리**: 피드백 없는 업로드 7일 후 Storage+행 삭제 (24h 주기 백그라운드)
- **모델 버킷**: 활성 버전만 유지(publish 시 구버전 자동 삭제). 모델 대용량은 HF Hub `ethanDev92/waste-models`로 이전(Supabase 50MB/파일 캡 우회)
- `scripts/storage_usage.py` — **80% 도달 시 사전 경고**(rc=1), 업로드 전 사전 검사 모드 지원
- VLM 일 200회 카운터 영속, 기기 기록 100건 초과 자동 정리

## 장애 내성 원칙

**"API는 항상 부팅한다"** — Supabase 불가 시 모델은 번들/캐시 폴백, 클래스 메타는 taxonomy.json 사이드카 폴백, region-info는 fail-open 빈 목록. → [waste-api](waste-api.md)
