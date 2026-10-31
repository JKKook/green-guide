# 프로덕션 출시 점검 — 결정·작업 묶음 (2026-10-31 grill 합의)

Play Console 은 프로덕션 신청 조건 3개(비공개 테스트 게시 · 테스터 12명 이상 참여 · 14일 유지)를 모두 충족한 상태다.
**신청은 아래 차단 항목을 모두 닫고 beta.3 비공개 테스트 재검증을 통과한 뒤에 누른다.**
기준 커밋 `4e1a5bc`(develop). 실측 근거는 끝의 "실측" 절.

## 1. 결정 (19건)

| # | 영역 | 결정 | 근거 |
| --- | --- | --- | --- |
| 1 | 범위 | 출시 차단 축 4개: 보안 · 결과화면 경합 · 법무/개인정보 · 모니터링 | 출시 뒤 고치기 어려운 것부터 |
| 2 | 분류 | 1차 분류 오류 = 오분류(재질 틀림). 파이프라인 자체는 정상(스모크 3/3, 보정 임계 유효) | 2026-10-31 배포 서버 실측 |
| 3 | 분류 | 오분류 접근 = 피드백 로그 혼동쌍 상위 5개 → VLM 중재 조건·임계 조정(재학습 없음) **+** 골든셋 리뷰 병행 | 출시 전 기간 안에 끝나고 같은 로그로 전후 비교 |
| 4 | 분류 | 골든셋 = `user-uploads` 최근 100장. **1차 = 사진 전체 재질(대분류·세부), 2차 = 물건별·영역별 후보.** 사진·모델 판정·AI 판단을 나란히 둔 리뷰 문서에 사용자 판단을 대조 | 두 축이 다른 엔드포인트라 원인 분리 |
| 5 | 서버 | "형태 인식: 사람" 칩 = 표시 임계를 판정 임계 0.30 으로 정렬 + 본 판정과 대분류가 다르거나 `non_object` 매핑인 identity 칩은 내리지 않음 | `clip_identity.py:142`(0.25) vs `:38`(0.30) vs 충돌 0.60 — 판정에 안 쓴 신호가 표시만 됨 |
| 6 | 앱 | 결과화면 경합 = 하이브리드: 첫 로드는 hier·regions·objects 세 응답을 모아 한 번에 판정(만장일치 규칙 유지) + 사용자 조작(탭·되돌리기·선택·재시도)마다 세대 번호, 이전 세대 응답 폐기 | 단일 worker 라 어차피 직렬 — 추가 지연 0~3초 실측. "최고 확신 선택"은 세 응답이 같은 질문이 아니라 기각 |
| 7 | 서버 | `/predict-with-regions` 를 구 27클래스 CNN 에서 계층 모델로 교체. 응답 스키마는 유지(앱 변경 최소) | 본 판정과 빗금 라벨이 다른 모델·다른 체계 |
| 8 | 서버 | OOD 보정값(tau_soft/hard)을 사이드카에 고정 커밋해 HF 모델 저장소에 반영 + 기동 시 보정값 없으면 기본값으로 가지 말고 경고 로그·Sentry 알림 | HF `v20260805_152341/taxonomy.json` 에 `ood` 없음 → 재배포 시 실사용 57% 거부 재현 위험 |
| 9 | DB | Supabase 단일 프로젝트 유지 + RLS 추가(`model_versions`·`region_waste_rules`) + `user-uploads` 버킷 비공개(서명 URL) + 013 적용 확인 | 1인 운영·무료 티어 |
| 10 | DB | 일일 `pg_dump` — GitHub Actions, 아티팩트 30일. DB 비밀번호는 사용자가 Secret 등록 | 백업 장치 0 |
| 11 | 서버 | 관리 API(`/reload-classes`·`/admin/reload-model`) 고정 토큰 헤더 + CORS 제한 + 추론 엔드포인트 속도 제한 | 무인증·CORS `*`+credentials |
| 12 | 앱 | Firebase Crashlytics + Analytics (프로젝트 생성·google-services 는 사용자) | 네이티브 크래시 포착 |
| 13 | 서버 | Sentry(FastAPI). 오분류율은 Supabase 데이터 주간 집계 | 예외·지연·VLM 실패 |
| 14 | 법무 | 고지문 개정 초안(사진 국외 이전 — Anthropic) · Data Safety 답안 · `ACCESS_FINE_LOCATION` → COARSE 전환 준비 + 외부 전문가 검토(사용자 의뢰) | 체크리스트 "정식 출시 전 법무 검토 필수" |
| 15 | 릴리스 | `key.properties` 없으면 release 빌드 실패 가드 + beta.3 때 develop→main 병합·태그에서 빌드 | debug 키 조용한 폴백 · main 84커밋 뒤 |
| 16 | 검증 | 통과 기준 = 수정 항목별 재현 테스트 + Crashlytics 3일 크래시 0 + 테스터 5명 이상 사용 | — |
| 17 | 측정 | 1차 분류 오류 수치화·효과 측정은 같은 서버 로그로 전후 비교 | — |
| 18 | 보류 | dev/prod 분리 · 요청 3회 업로드 통합 · R8 은 출시 뒤 | 범위 통제 |
| 19 | 기록 | 되돌리기 어려운 결정 = 7(모델 교체)·9(RLS·버킷) — 이 문서가 기록 | — |

## 2. 작업 묶음

| 묶음 | 담당 scope | 항목 | 성공 기준 |
| --- | --- | --- | --- |
| W1 앱 | `apps/mobile/` | 6 하이브리드 경합 · 12 Firebase · 15 서명 가드 · 14 위치 권한 COARSE + `legal_terms.dart` 고지 초안 | `flutter analyze` 0 · `flutter test` 전부 통과(+경합 테스트) · 서명 없는 release 빌드가 실패 |
| W2 모델·추론 | `services/inference-api/` | 5 칩 정책 · 7 regions→hier · 8 OOD 사이드카+가드 · 3 혼동쌍별 VLM 조정 | pytest 통과 · 스모크 3장 재실행 일치 · 사이드카 `ood.tau_soft/tau_hard` 존재 |
| W3 서버 운영 | `services/inference-api/` | 11 토큰·CORS·속도 제한 · 13 Sentry | 토큰 없는 admin 호출 401 · CORS 응답 헤더에 `*` 없음 · Sentry 테스트 이벤트 수신 |
| W4 데이터·DB | `migrations/`·`scripts/`·`.github/` | 9 RLS·버킷·013 · 10 백업 워크플로 · 4 골든셋 100장 추출+리뷰 데이터 · 3 혼동쌍 집계 | anon 키로 `model_versions` 쓰기 거부 · 버킷 공개 URL 403 · 백업 아티팩트 1건 생성 · 골든셋 JSON 100행 |
| W5 문서·릴리스 | `docs/`·루트 | 14 Data Safety 답안·`docs/privacy` 갱신 · PR 병합 · main 동기화·태그 | PR 전부 develop 병합 · `v1.0.0-beta.3` 태그 = main |

적용 게이트: Supabase 실제 변경(RLS·버킷)·HF Space 배포·Play 업로드는 **리더 확인 뒤** 실행한다.

## 3. 사용자 작업
- Firebase 프로젝트 생성 → `google-services.json` 전달
- GitHub Secret `SUPABASE_DB_URL`(백업용) 등록
- Sentry 프로젝트 DSN 전달
- 법무 외부 검토 의뢰(초안 준비 뒤)
- 골든셋 리뷰 100장 판단 표시

## 4. 실측 (2026-10-31)
- Play Console: 프로덕션 신청 조건 3/3 체크, 버튼 활성(누르지 않음). 비공개 Alpha 최신 `4 (1.0.0-beta.2)` 10-18 23:54. 28일 활성 기기 7.
- `flutter analyze` 이슈 0 · `flutter test` 97 통과.
- 배포 서버 `/predict-hier`·`/predict-with-regions`·`/predict-objects` 3장: 순차 11~13.5초, 동시 8~9초(단일 worker 직렬화). 서버 추론 0.2~0.3초(hier)·2.6~4초(objects). OOD 거리 0.54 사진 미거부 → 보정 임계 유효.
- `/` 응답 `model_arch: cnn`(구 27클래스) · `/model/latest` = `20260805_152341`(계층 25클래스, test_acc 0.9315).
- HF `v20260805_152341/taxonomy.json`: `ood` 키 없음.
- 저장소 탐색: RLS 없는 테이블 2(`_bootstrap_new_project.sql:118,535`) · 공개 버킷(`bootstrap_new_supabase.py:58`) · 관리 API 무인증(`routers/admin.py`) · 경합 지점(`result_controller.dart:201-207,252-255,310,332-354`) · debug 키 폴백(`build.gradle.kts:52-58`).
