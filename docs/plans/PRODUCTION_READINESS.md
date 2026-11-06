# 프로덕션 출시 점검 — 결정·작업 묶음 (2026-10-31 grill 합의)

Play Console 은 프로덕션 신청 조건 3개(비공개 테스트 게시 · 테스터 12명 이상 참여 · 14일 유지)를 모두 충족한 상태다.
**신청은 아래 차단 항목을 모두 닫고 beta.3 비공개 테스트 재검증을 통과한 뒤에 누른다.**
기준 커밋 `4e1a5bc`(develop). 실측 근거는 끝의 "실측" 절.

## 1. 결정 (19건)

| # | 영역 | 결정 | 근거 |
| --- | --- | --- | --- |
| 1 | 범위 | 출시 차단 축 4개: 보안 · 결과화면 경합 · 법무/개인정보 · 모니터링 | 출시 뒤 고치기 어려운 것부터 |
| 2 | 분류 | 1차 분류 오류 = 오분류(재질 틀림). 파이프라인 자체는 정상(스모크 3/3, 보정 임계 유효) | 2026-10-31 배포 서버 실측 |
| 3 | 분류 | ~~피드백 로그 혼동쌍 상위 5개 → VLM 중재 조정~~ → **취소(10-31 사용자 결정)**. 실사용 정정이 전체 4건(4쌍 각 1건)이라 임계를 바꿀 근거가 없다. 골든셋 리뷰로 오분류율을 재고, beta.3 피드백 ≥30건 시점에 재검토 | `confusion_pairs.json`: live 18행 · corrected 4 |
| 4 | 분류 | 골든셋 = `user-uploads` 전수 69행 − 홀드아웃 11 = **58행**(100장 목표는 데이터 부족). **1차 = 사진 전체 재질(대분류·세부), 2차 = 물건별·영역별 후보.** 복원분 44장은 로컬 재추론으로 hier 판정을 채운 뒤(프로덕션 호출 금지) 사진·모델 판정·AI 판단을 나란히 둔 리뷰 문서에 사용자 판단을 대조 | 두 축이 다른 엔드포인트라 원인 분리 · 산출물 `~/Downloads/greenguide-golden/` |
| 5 | 서버 | "형태 인식: 사람" 칩 = 표시 임계를 판정 임계 0.30 으로 정렬 + 본 판정과 대분류가 다르거나 `non_object` 매핑인 identity 칩은 내리지 않음 | `clip_identity.py:142`(0.25) vs `:38`(0.30) vs 충돌 0.60 — 판정에 안 쓴 신호가 표시만 됨 |
| 6 | 앱 | 결과화면 경합 = 하이브리드: 첫 로드는 hier·regions·objects 세 응답을 모아 한 번에 판정(만장일치 규칙 유지) + 사용자 조작(탭·되돌리기·선택·재시도)마다 세대 번호, 이전 세대 응답 폐기 | 단일 worker 라 어차피 직렬 — 추가 지연 0~3초 실측. "최고 확신 선택"은 세 응답이 같은 질문이 아니라 기각 |
| 7 | 서버 | `/predict-with-regions` 를 구 27클래스 CNN 에서 계층 모델로 교체. 응답 스키마는 유지(앱 변경 최소) | 본 판정과 빗금 라벨이 다른 모델·다른 체계 |
| 8 | 서버 | OOD 보정값 누락 가드: 기동 시(lifespan 사전 로드) `check_ood_calibration()` — 없으면 기본값 복귀 대신 **τ 끔(fail-open) + 경고 + `ood_calibrated=False`**, Sentry 알림. publish 스크립트에 `ood` 누락 금지 가드(classifier scope) | HF `v20260805_152341/taxonomy.json` 에 `ood` 없음. **보정 10-31**: 실제 서빙 사본 `serving/`(v14, 10-21) 에는 τ soft 0.558·hard 0.674 가 있고 `/model/latest` 는 구 flat 레지스트리라 계층 서빙과 무관 — 고정 커밋은 이미 된 상태, 폴백 경로 가드만 남음 |
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
| 20 | **출시** | **재학습(v15)이 실사용 게이트를 통과할 때까지 프로덕션 신청 보류**(10-31 사용자 결정). beta.3 은 완화책 포함으로 비공개 테스트에 올릴 수 있으나 신청은 v15 뒤 | 실사용 대분류 41%, electronics 쏠림 — `ELECTRONICS_BIAS_ANALYSIS.md` |
| 21 | 모델 | **Haiku 금지, 최소 Sonnet, 재질 판단(VLM 폴백·실측)은 `claude-opus-5-5`** | 사용자 결정 10-31 |
| 22 | 모델 | 재학습 W7(GG_TRAIN): 재활용품 자연장면 하드 네거티브 ≥3,500 + RandomResizedCrop·배경 증강, OI electronics 품목을 소형 IT 로 축소. 실사용 셋(피드백 51·테스터 93·골든 v3 51)을 **배포 게이트**로 승격 | 분석 E절 가설 1·2·5 |
| 23 | 서버·앱 | 서빙 완화 6건(W6 GG_CNN): VLM opus · auto_crop 장면 경로 · non_object≥0.5 reject · electronics bias+게이트 0.7+VLM 중재 · CLIP 억제(효과 있을 때만) · OCR 표시 오탐 점검. 앱 W8(GG_APP): 가이드 프레임을 중앙 정사각(짧은 변 84%)으로 축소해 밀착 크롭 | 분석 E절 가설 2·3·4 |
| 25 | 서버 | **W6 결과(10-31)**: 기본값 B = auto_crop 장면 경로 + non_object≥0.5 reject + electronics bias 0.5·coarse 게이트 0.7 + VLM(opus) 중재 + CLIP 억제 + OCR 2글자 표시 교체 금지. 실사용 피드백 51장 정답 21→28, electronics 오탐 16→4, 진짜 전자제품 recall 5/5 · 테스터 35장 13→24, 오탐 10→0. VLM 발동 17/51·22/35, 장당 $0.0059(opus, 캐시), 상한 200회/일 → 월 최대 $36, 지연 p50 1.0s(미발동)·4.4s(발동) | `realworld_golden/W6_FIXES.md`@af5d48a66d32 |
| 26 | 사고 | **VLM 폴백 무음 실패**: Anthropic 키가 08-06 이후 만료돼 로컬·운영 폴백이 두 달 가까이 fail-open 으로 꺼져 있었음(마지막 성공 기록 08-06, 실패 로그가 info). 10-31 사용자가 재발급, 기동 시 키 점검 + `/health` `vlm_key_status` + WARNING 승격(`c014303`). 병합 때 Sentry `event_level=WARNING` 또는 해당 메시지 2개 캡처 필요(W3 기본은 breadcrumb) | 리더 실측 10-31 |
| 27 | 모델 | **토큰 없는 경로 채택(10-31 사용자)**: v15(하드 네거티브) → W9 백본 비교(DINOv2·CLIP 동결 특징+헤드 vs ResNet50) → W10 로컬 VLM(ollama Qwen2.5-VL 후보) 의사라벨. Opus 는 평가셋 라벨 기준과 운영 폴백에만. 출시 게이트 = 실사용 평가셋 eval_v1(97, Opus 라벨 + 사용자 중재) 대분류 **≥80%(최소) / 85%(목표)**, electronics 오탐률 ≤3%, 진짜 전자제품 recall ≥0.85, 동일 출처 test −2pp 이내 | AI 라벨 기준 v14 47/97(48%) |
| 28 | 라벨 | 평가셋 라벨 기준 = Opus AI 라벨, 기존 라벨과 충돌 23 + AI 확신 낮음 2 = 25장만 사용자 중재(`.archive/claude-artifacts/2026-10-31_label-adjudication/`(아티팩트 11-03 삭제, 로컬 보관), db `adjudications`). 우선순위 user_adjudicated > ai_opus > 기존 | 충돌 19/23 이 앱 피드백 라벨(손·배경→etc 7, 전자기기→외장 재질 6) |
| 29 | 평가셋 | **eval_v1 동결(10-31)**: `~/Downloads/greenguide-golden/realworld_eval_v1.jsonl` sha256 `b71fd1e8d9a8883f73aabd26703a42366d272e9b5daf18f7109659641b6145e0`, 97행 = ai_opus 72 + user_adjudicated 25(ai 18·prev 2·other 5), split gate 79 · final_holdout 18(sha256(id)[0]%5==0). v14 대분류 45/97(gate 36/79 · holdout 9/18), electronics 판정 44 중 오탐 24(실제 23). 게이트 판정은 gate 79 로, holdout 18 은 최종 1회만 | GG_DATA 04da3ec |
| 30 | 배포 | **G4 분할(10-31 사용자)**: push 는 G2 전에 선행 — 9 브랜치 `git push -u origin` 완료(api-inference-prod 는 api-electronics-guard 에 포함), PR·병합은 G2 뒤. PR 생성 계정은 JKKook(ethan-outlier 는 READ). W10 로컬 VLM: qwen3-vl:8b 는 ollama 0.23.1 구버전으로 pull 거부(412) → 업그레이드 보류, qwen2.5vl:7b(13.9GB) 결과로 먼저 판정. 7b 라벨링은 v15 학습과 동시 상주 시 스왑 발생(15/16GB) → 14/97 에서 중단, v15 종료 뒤 자동 재개 | 리더 판정, 3b 는 품질상 게이트용 부적합 |
| 31 | 모델 | **W9 백본 비교는 전환이 아니라 통제 실험(11-01 사용자 재검증)**: 7월 dinov2_hier.onnx 를 eval_v1 97 에 v14 와 같은 순수 ONNX 로 비교 → v14 49/97(50.5%)·elec 오탐 25/74 vs DINOv2 35/97(36.1%)·elec 오탐 4/74, 둘 다 오답 37. 백본 전환이 대분류를 올린다는 근거 없음(과거 B4 제거 기록과 일치). **채택 기준(사전 등록)**: 같은 분할·gate 79 에서 v15 대비 대분류 +5pp 이상 ∧ non_object 재현 ≥ v15 ∧ elec 오탐 ≤3% — 셋 모두 만족해야 서빙 후보. 실행은 v15 종료 뒤, linear·mlp 헤드 둘 다. 미달이면 W10 으로 | 결과 json memory/gg_realworld_v14_vs_dinov2_jul.json sha 007ef229… |
| 32 | 분류 체계 | **분류 체계 v2 — 재질·상태·배출 3축(11-03 사용자 grill 13건)**: 재질 대분류 14 고정 + 분리배출 축 15(표시 마크 9 + 배출 경로 6) + 상태 축 4(깨끗·오염·파손·복합재질). 배출 = 재질×상태 규칙표, 제품 마크 인식 시 마크 우선. 금속 세부 `metal_can`/`metal_scrap`, 음식물 세부 `food_waste`/`food_inedible`(사료화 불가 → 종량제). 반영 순서 서빙(VLM 프롬프트 v3·규칙표·앱 안내) → 끝나면 바로 CNN 재학습(상태 헤드 추가, 라벨 = 원본 복원 + 로컬 VLM + 표본 검수). 평가셋 97 전부 새 축 라벨(eval_v2, holdout 18 은 최종 1회). 60장 v1.1 확인은 중단·응답 보존. 명세 `docs/plans/TAXONOMY_V2.md` | 사용자 골든셋 채점 중 '캔 없음' 지적 |
| 33 | 모델 | **CNN v2(계층 + 상태 헤드) 미채택, 서빙은 v14 + two_stage 유지(11-04)**: 13에폭 조기 종료(best 9). 같은 출처 test 대분류 v14 0.9566 / v2 0.9565(−0.01pp, 통과)인데, eval_v2 gate 65 서빙은 44 → 37/65, elec 오탐 0 → 4/49로 가드 미달. 원인 = 과신: 틀린 장 평균 확신 0.681 → 0.907, VLM 발동 25 → 7. val loss는 7에폭부터 상승(0.345 → 0.424)하는데 best 선정이 정확도만 봤다. 온도 보정(T 1.5는 val NLL로 정하고 gate는 1회만 측정, 사용자 승인)은 41/65·오탐 3/49로 역시 미달이다. v14 수준의 VLM 발동에는 T 2.5~3.0이 필요해, val로 보정해도 실사용 과신이 남는다. v14 + two_stage 는 가드 통과(44/65·오탐 0/49·재현 12/16). 기록 `ml/classifier/diagnostics/EXPERIMENT_cnn_hier_v2_2026-11-03.md` | 사용자 "성능 저해 없게 확인" 지시, 가드 실측 |
| 34 | 서버 | **VLM 서빙 기본값 = Sonnet 5.5 상시 중재 + R1(CNN 일치 채택) + 1회차 재질 규칙, 채택 기준 0.8 유지(11-06 사용자)**. 결정 21의 "재질 판단 = Opus"를 서빙에 한해 대체(평가셋 라벨 기준은 Opus 유지). CNN 은 v14 + two_stage 그대로, 로컬 qwen VLM 미채택(14장 7/14 vs Opus 14/14, 13.9GB 스왑). 실측: eval_v2 gate A 34장 대분류 현행 24 · Opus 상시 30 · Sonnet 0.8 25 · Sonnet 0.6 29, elec 오탐 모두 0/28, 천 장당 Opus $9.2 · Sonnet $4.1, p50 6.8s · 4.7s. final holdout 16장(1회 소진): 현행 9 · Sonnet 0.6 9 · Opus 11. **0.6 미채택** = holdout 49d8a1dd(장난감 → trash)를 0.6이 plastic 으로 덮어 잃었고 0.8이면 CNN 정답이 남는다, gate A 의 0.6 이득은 같은 셋 튜닝이라 과적합 보류. Sonnet 0.8 은 holdout 미측정(소진) → 배포 뒤 실사진으로 재측정 | api `38af437`·`7d1a173`(PR #48) · `eval_v2/trigger/A_sonnet_vs_opus.md` · `eval_v2/holdout_final/HOLDOUT_FINAL.md`@`26f606a`(PR #44) |
| 24 | 골든셋 | 규칙 개정: 피드백 있는 사진 제외(사람이 이미 채점) · 중복(sha256·dHash≤6) 제외 · 로컬 `local_feedback/` 195장 포함 → **v3 51장**(후보 205, 중복 154). 피드백 라벨 51장은 리뷰 대신 정답으로 정확도 산출(AI 의심 라벨 18건 표시) | 사용자 10-31 |

## 2. 작업 묶음

| 묶음 | 담당 scope | 항목 | 성공 기준 |
| --- | --- | --- | --- |
| W1 앱 | `apps/mobile/` | 6 하이브리드 경합 · 12 Firebase · 15 서명 가드 · 14 위치 권한 COARSE + `legal_terms.dart` 고지 초안 | `flutter analyze` 0 · `flutter test` 전부 통과(+경합 테스트) · 서명 없는 release 빌드가 실패 |
| W2 모델·추론 | `services/inference-api/` | 5 칩 정책 · 7 regions→hier · 8 OOD 사이드카+가드 · 3 혼동쌍별 VLM 조정 | pytest 통과 · 스모크 3장 재실행 일치 · 사이드카 `ood.tau_soft/tau_hard` 존재 |
| W3 서버 운영 | `services/inference-api/` | 11 토큰·CORS·속도 제한 · 13 Sentry | 토큰 없는 admin 호출 401 · CORS 응답 헤더에 `*` 없음 · Sentry 테스트 이벤트 수신 |
| W4 데이터·DB | `migrations/`·`scripts/`·`.github/` | 9 RLS·버킷·013 · 10 백업 워크플로 · 4 골든셋 100장 추출+리뷰 데이터 · 3 혼동쌍 집계 | anon 키로 `model_versions` 쓰기 거부 · 버킷 공개 URL 403 · 백업 아티팩트 1건 생성 · 골든셋 JSON 100행 |
| W5 문서·릴리스 | `docs/`·루트 | 14 Data Safety 답안·`docs/privacy` 갱신 · PR 병합 · main 동기화·태그 | PR 전부 develop 병합 · `v1.0.0-beta.3` 태그 = main |

적용 게이트: Supabase 실제 변경(RLS·버킷)·HF Space 배포·Play 업로드는 **리더 확인 뒤** 실행한다.

버킷 비공개(014) 적용 순서(W4 보고, 10-31): ① 서버 `uploads.py` 공개 URL → authenticated 경로(W3, `e9a8626`) ② `bootstrap_new_supabase.py` 버킷 생성 비공개·URL 동일(W3) ③ 학습 스크립트 6곳 `image_url` 직접 내려받기 → `storage.download(storage_path)`(W2 classifier 브랜치) ④ 014 적용 → anon 쓰기 거부·공개 URL 403 검증. 역순이면 그 사이 학습 스크립트가 깨진다(서버 추론은 무영향). 백업 Secret 은 Session pooler 연결 문자열(Direct 는 IPv6 전용).

진행 기록(10-31): 커밋 규약 변경 — 팀원은 작업에 필요한 커밋을 묻지 않고 한다. PR 생성 금지, push 는 리더 통지 뒤 일괄.

**W1~W4 전부 로컬 커밋 완료(10-31, push 0 · PR 0).** 팀원 세션 4개 종료.

| 브랜치 | 커밋 | 검증 |
| --- | --- | --- |
| `feature/app-prod-readiness` (W1) | `87e5b93` 경합 하이브리드 · `d0a688e` 서명 가드 · `e2beaa3` Firebase+COARSE+고지 초안 | analyze 0 · test 104(경합 테스트 7, develop 코드로는 7 실패) · 무키 release 실패 확인. 범위 밖: Kotlin 2.2.20→2.4.0, AD_ID·ADSERVICES 권한 `tools:node="remove"` |
| `feature/api-inference-prod` (W2) | `0dbeacd` 칩 정렬·regions 계층 모델·OOD 가드 | pytest 80 · 스모크 3/3 일치 |
| `feature/classifier-publish-ood-guard` (W2) | `cea0714` publish 가드+curl 토큰 · `cda0bd6` 학습 스크립트 4곳 storage download | classifier pytest 53 |
| `feature/api-ops-hardening` (W3) | `0c1c1cf` 토큰·CORS·429·Sentry · `e9a8626` uploads.py authenticated URL · `ba8b716` bootstrap 비공개 | pytest 81(.env 연결 시) · 토큰 401/200 · 429 |
| `feature/api-w4-rls-migration` (W4) | `1e8242c` 014 RLS·버킷 · `b145f5e` 골든셋·혼동쌍 스크립트 · `3c53e85` 로컬 재추론 | 로컬 PG16 적용·롤백 통과 |
| `feature/repo-w4-backup-workflow` (W4) | `8144900` backup.yml | actionlint 통과, 실실행은 Secret 뒤 |
| `feature/repo-production-readiness` (W5) | 정본·Data Safety 초안·진행 기록 | — |

병합 순서(안): repo 문서 → app → api-ops-hardening → api-inference-prod(같은 폴더, 파일 분리 확인됨) → classifier → api-w4(014 은 병합만, 적용은 게이트) → repo-w4. 병합 뒤 develop→main 병합·`v1.0.0-beta.3` 태그·HF subtree push·AAB 빌드.

**골든셋 리뷰(결정 4) 게시**: `.archive/claude-artifacts/2026-10-31_golden-review/`(아티팩트 11-03 삭제, 로컬 보관) — 58장, 사용자 판단은 페이지 db `reviews/<upload_id>`. AI 검토(에이전트 2, 사진 직접 판독) 결과: 대분류 일치 39/58. **불일치 19건 중 12건이 플라스틱 용기류(물티슈 팩·PET 병·스프레이·약통·립 튜브·에어로졸 캔)를 `electronics` 로 판정** — 결정 3 재검토 때 1순위. non_object 오판 2(바닥·주먹). 사용자 피드백 라벨 의문 11건(마우스→plastic, 전자기기→metal, 영수증→cardboard 등) — 골든셋 정답으로 쓰기 전 사용자 재확인.

## 2-1. 사용자 판단 게이트 (10-31 리더 지정)

| # | 게이트 | 시점 | 리더가 올리는 것 | 사용자가 정하는 것 |
| --- | --- | --- | --- | --- |
| G1 | 라벨 중재 25장 | 지금 | 중재 페이지 | 정답 선택 |
| G2 | v15 판정 | 11-01 오전 | 실사용 게이트 수치(v14 대비) + 통과/불통과 | 서빙 모델 교체 여부, 불통과 시 2라운드 vs 백본 경로로 이동 |
| G3 | 로컬 VLM 설치 | GG_VLM 비교표 뒤 | 모델 후보·용량·메모리·속도 | `ollama pull` 승인(약 5~6GB) |
| G4 | push·PR·병합 | push 10-31 완료(9), PR·병합 G2 뒤 | 브랜치 10개 순서·충돌 0 확인 | 일괄 승인(push 만 선승인, 결정 30) |
| G5 | HF 배포 | G4 뒤 | 서버 완화 B + 사이드카 + 시크릿 체크리스트 | 운영 교체 승인 |
| G6 | beta.3 업로드 | G5 뒤 | AAB·변경 요약·테스트 항목 | Play Console 업로드 클릭 |
| G7 | 프로덕션 신청 | beta.3 3일 무크래시 + 게이트 통과 | 통과 수치 | 신청 클릭 |

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
