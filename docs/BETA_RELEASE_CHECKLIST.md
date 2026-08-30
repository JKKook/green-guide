# 그린가이드 AI 베타 배포 체크리스트 (2026-08-22 점검)

전체 점검(UI/UX · 라이선스 · 기능/릴리즈 · 백엔드) 결과와 조치 상태.
검증 기준: `flutter analyze` 통과 · `flutter test` 9/9 통과 · `flutter build apk --release` 성공.

## ✅ 이번에 적용한 수정 (waste_app)

| 영역 | 내용 | 파일 |
| --- | --- | --- |
| 릴리즈 서명 | `android/key.properties` 있으면 업로드 키, 없으면 debug 키로 폴백. `signingReport` 로 양쪽 분기 검증 완료 | `android/app/build.gradle.kts` |
| 네트워크 | 개발 PC LAN IP·localhost cleartext 예외를 **debug 소스셋으로 분리** — 릴리즈는 HTTPS 전용 | `android/app/src/{main,debug}/res/xml/network_security_config.xml` |
| 버전 | `1.0.0+1` → `1.0.0-beta.1+1` | `pubspec.yaml` |
| 피드백 자유 입력 | 입력값을 서버 라벨(slug)로 해석(`resolveLabelSlug`). 매핑되면 전송, 안 되면 **서버 전송 생략 + "기기 기록에만 저장" 안내** (기존엔 전량 400 에러 노출) | `data/waste_info.dart`, `widgets/feedback_card.dart` |
| 오류 문구 | `friendlyError()` 신설 — 타임아웃/네트워크/413/503 등을 한국어 안내로. `TimeoutException after 0:00:30…` 같은 원시 예외 노출 제거 | `api/api_client.dart` + 결과·촬영·피드백 화면 |
| 콜드스타트 | 앱 부팅 시 `/health` 웜업 핑(3분 타임아웃, 백그라운드) — 절전 서버 대비 | `services/server_warmup.dart`, `main.dart` |
| 업로드 용량 | 스마트 촬영 풀프레임 원본 → **긴 변 1600px JPEG 재인코딩**(백그라운드 isolate, EXIF 회전 반영). 갤러리 경로와 동일 | `data/image_prep.dart`, `screens/live_camera_screen.dart` |
| 햅틱 토글 | 저장만 되고 무시되던 설정을 실제 연결 — `Haptics` 래퍼로 83개 호출부 전환 | `data/haptics.dart` + 14개 화면 |
| 알림 "준비 중" | OS 알림 미연동 상태를 6곳(온보딩 2·설정 2·수거일 안내·알림 관리·통합검색)에 명시 | `data/collection_schedule.dart` 외 |
| 수거 일정 카피 | 하드코딩 일정을 "조례 기준"으로 단정하던 문구 → "일반적인 배출 요일 예시"로 정정 | 홈·수거일 안내·설정·통합검색 |
| 라이선스 고지 | `LicenseRegistry` 에 Pretendard OFL 원문(에셋 번들) · 푸른숲체 고지 · **AI-Hub(NIA) 학습 데이터 출처** · TACO/Open Images · 공공데이터 출처 등록 → 설정 > 앱 정보 > 오픈소스 라이선스 | `data/licenses.dart`, `assets/licenses/pretendard-OFL.txt` |
| 앱 정보 문구 | 사실과 다르던 "ResNet18 / Kaggle Garbage Classification" → 실제 계층 모델 + AI-Hub 출처 명시 | `screens/settings_screen.dart` |
| 약관 사실 보정 | 처리위탁에 **Anthropic PBC(보조 이미지 인식)** 누락분 추가, 보유기간을 실제 서버 동작(피드백 확정 사진은 학습 자료로 계속 보관)에 맞게 정정 | `data/legal_terms.dart` |
| 미사용 폰트 | 참조 0인 **Jua 번들 해제**(파일은 디스크에 보존) | `pubspec.yaml` |
| 카메라 안정성 | 백그라운드 전환 시 dispose 된 컨트롤러 참조 제거, 결과 모달 중 복귀 시 카운트다운 재시작 방지, `await` 후 `mounted` 가드 | `screens/live_camera_screen.dart` |
| 갤러리 흐름 | 분석 취소 시 시스템 피커가 다시 뜨던 동작 → 확인 화면에 머무름 | `screens/gallery_confirm_screen.dart` |
| 온보딩 | "나중에"로 지역을 건너뛰어도 홈에서 같은 질문을 반복하던 문제 수정. 탭 불가한 "배출 시간대 >" chevron 제거 | `screens/onboarding_screen.dart` |
| 개발자 옵션 | 4탭부터 뜨던 힌트를 **디버그 빌드 전용**으로 (7탭 진입 자체는 유지) | `screens/settings_screen.dart` |
| 릴리즈 로그 | `print()` 9곳 → `kDebugMode` 가드 | `services/prediction_service.dart`, `services/remote_model_service.dart` |
| CAM 안내 | "관리자가 waste-api 를 push 해야 합니다" 등 운영자용 문구 → 사용자용 안내 | `widgets/result_modal.dart` |
| 로딩 문구 | `Scanning material` / `Uploading 45%` / `Almost there` → 한국어 | `widgets/result_modal.dart` |
| 온디바이스 제외 | Play 16KB 요건 미충족 `onnxruntime` 제거 — 클라우드 전용. 정확도는 서버 경로(앙상블·TTA·객체 분리)가 우위라 사용자 체감 손실 없음, 오프라인 분류만 사라짐 | `pubspec.yaml`, `services/prediction_service.dart`, `screens/settings_screen.dart`, `data/settings_store.dart` |
| 접근성 | 아이콘 전용 버튼(촬영 FAB·헤더 ⓘ/테마·카메라 5종·결과 닫기·검색) 에 Semantics 라벨 추가 | 다수 |
| 앱 아이콘 | 스틸블루 단색+흰 라인 잎 → **파스텔 스카이 그라디언트 배경 + 파스텔 민트 잎(잎맥·줄기·소프트 섀도)**. 적응형 아이콘 배경을 이미지로 전환, 스플래시 마크도 동일 잎으로 통일(Android 12+ 원형용 축소본 별도). 원본 SVG `assets/icon/icon.svg` | `assets/icon/*`, `assets/splash/*`, `pubspec.yaml`, `android/.../mipmap*`, `drawable*` |
| 테스트 | 깨져 있던 온보딩 테스트 수정 + `resolveLabelSlug`·`friendlyError` 단위 테스트 추가 (9개 통과) | `test/` |

## 📱 실기기 QA (2026-08-29, 갤럭시 S25 Ultra · 삭제 후 재설치 · 릴리즈 APK)

통과: 스플래시 → 동의 시트(전체 동의·약관 5종 상세 열람) → 지역(GPS 권한 → 부천시 자동 감지) → 세대 구분 → 수거 요일/알림("발송은 준비 중" 표기) → 홈(지역 재질문 없음) → 스마트 촬영(카메라 권한 → 5초 카운트다운 → 분석 로딩 한국어 문구 → 결과: 품질 배너·부천시 조례 데이터·다중 물건 감지) → 피드백(칩 선택 → 서버 `corrected` 저장 / 직접 입력 → "기기 기록에만 저장" 안내) → 기록 탭 → 설정 → 앱 정보(AI-Hub 고지) → 오픈소스 라이선스 → 오프라인(비행기 모드) 오류 문구 + 다시 시도.

QA 중 발견·수정한 것:
| 문제 | 수정 |
| --- | --- |
| **분석 1회당 서버에 사진 2장 저장** — `/predict-with-regions` 도 업로드 기록 | 서버에서 with-regions 기록 제거(커밋 `waste-api`), 3개 엔드포인트 호출 시 +1건 확인 |
| **기록 탭이 앱 재시작 전까지 갱신 안 됨** — IndexedStack 으로 살아 있어 initState 만 로드 | `historyRevision` ValueNotifier 추가, 저장/삭제/전체삭제 시 bump → 기록·통합검색이 즉시 재로드 |
| **물건 후보 선택 후 피드백이 서버로 안 감** — `ObjectCandidate.toPrediction()` 에 uploadId 없음 | 최초 분류의 uploadId 를 보존해 물건 선택 결과에도 얹음 |
| **"분석된 재질" ≠ "사진 속 물건" 재질** — 장면 전체(/predict-hier 원본) vs 물건별 크롭(/predict-objects) 입력이 달라 생김 | 물건들이 만장일치로 장면 결과와 다르면 물건 기준 자동 채택(되돌리기 가능) + 불일치 시 안내 캡션 |
| 약관에 `[동의 철회 시까지]` 대괄호 잔존(작성 실수) | 제거 |

관찰만(미수정): 위치 권한 다이얼로그가 "정확한 위치" 기본 — Manifest 의 `ACCESS_FINE_LOCATION` 을 빼면 "대략적인 위치"만 요청(시군구 단위면 충분) · 검색 탭 진입 시 키보드가 자동으로 올라오며 하단 바/FAB 가 키보드 위로 따라 올라옴 · 기록 요약이 자유 입력 라벨(icepack)도 "재활용" 으로 집계.

## 🔴 남은 배포 차단 항목 (앱 코드 밖 — 결정·계정 작업 필요)

1. ~~Supabase 프로젝트 복구~~ → **✅ 2026-08-29 신규 프로젝트 `rgqmmmelchtfetipocxs` 로 이사 완료.**
   스키마·버킷·피드백 51건·지역 규정 933행·모델 레지스트리 복원, HF Space 시크릿 교체·재시작,
   `/model/latest`·`/region-info`·predict→feedback E2E 검증 완료. 구 프로젝트 `qnjwq…` 는 폐기.
2. **업로드 키 생성** — `keytool -genkey ... upload-keystore.jks` 후 `android/key.properties` 작성
   (`storeFile` / `storePassword` / `keyAlias` / `keyPassword`). 배선은 완료돼 있어 파일만 놓으면 릴리즈 서명 적용.
   지금 배포하면 debug 키로 서명되어 **나중에 정식 키로 업데이트 불가**(재설치 필요).
3. ~~onnxruntime 16KB 페이지 미지원~~ → **✅ 2026-08-29 온디바이스 모드 베타 제외.**
   `onnxruntime` 의존성·`local_inference.dart`·`remote_model_service.dart`·설정 "어디서 분류할지" 토글·`/model/latest` 클라이언트 제거.
   APK 93.7MB → **69.4MB**, 네이티브 .so 에 onnxruntime 없음 확인. 정식 출시 때 16KB 지원 바인딩이 나오면 복원(서버 모델은 그대로 HF Hub 에 있음).
4. **약관 플레이스홀더 확정** — `[운영자명]`, `[문의 이메일]`, `[2026. 00. 00.]`, `[30일]`, `[6개월]`,
   `[보호책임자명]`, `[주소]`, `[관리책임자명]`. **법무 검토 필수**(국외이전 고지, 위치기반서비스 신고 여부,
   만 14세 미만 절차, 학습 활용 철회 시 삭제 프로세스).
5. **AI 학습 동의(`ai_training_opt_in`)가 서버에 전달되지 않음** — 앱은 로컬 저장만 하고,
   서버는 `WASTE_API_COLLECT_UPLOADS=true` 로 동의와 무관하게 전 업로드를 저장.
   → 서버에 opt-in 필드 추가하거나, 약관 문구를 현재 동작에 맞게 유지할지 결정 필요(이번엔 후자 방향으로 문구만 보정).
6. **Google Play 요건** — 비공개 테스트도 개인정보처리방침 **외부 URL** · Data Safety 폼(사진 업로드·대략적 위치) 필요.
   현재 약관은 앱 내 화면만 존재.

## 🟠 백엔드 후속 (waste-api)

- `/feedback` 자유 텍스트 컬럼(`feedback_note` 등) 추가 시 앱의 "직접 입력"도 서버 학습에 반영 가능
  (현재는 앱에서 매핑 실패분을 기기 저장으로 처리).
- admin POST 2개(`/reload-classes`, `/admin/reload-model`) 무인증 → 토큰 헤더 최소 적용.
- CORS `*` + `allow_credentials=True`, rate limit 없음, 업로드 bomb 가드 없음.
- 단일 worker + async 핸들러 내 동기 추론 → 동시 요청 직렬화. `asyncio.to_thread` 또는 `def` 핸들러 전환 권장.
- HF 무료 cpu-basic 48h sleep — 앱 웜업 핑을 넣었지만 근본 해결은 외부 keep-alive 또는 유료 티어.
  Supabase 무료 플랜도 7일 무요청 시 일시정지(이번 사태 원인) — 베타 트래픽 공백이 길면 같은 keep-alive 로 함께 커버.
- `/feedback` 에 존재하지 않는 upload_id → 404 대신 500 (`uploads.py record_feedback` `.single()` 빈 결과 미처리). 이사 전부터 있던 버그.
- `user-uploads` 버킷 공개 읽기 · 피드백 확정 사진 무기한 보관 → 보존 정책 정리.
- ✅ 2026-08-29 저장본 **WebP 640px q75** 전환 배포(커밋 d468b14) — 장당 ~15KB, 1GB 기준 약 6.7만 장. 기존 .jpg 51장과 혼재.
- 미커밋 변경(`src/streams.py`, `src/vlm_fallback.py`, `design/tokens.json`) 배포 여부 결정, README 갱신.

## 🟡 확인 필요

- **푸른숲체(YK Green Forest)** woff2→otf 변환이 "수정·변형" 금지 조항에 해당하는지 공식 원문 확인.
  현재 앱에서 실사용 화면은 없고(`kDisplayFontFamily` 참조처가 미사용 위젯 `app_tooltip.dart` 뿐),
  쓰지 않을 거면 `pubspec.yaml` 폰트 선언에서 빼면 라이선스 이슈 자체가 사라짐.
- 접근성 터치 타깃: 시트 닫기 X·월 이동 chevron 등이 26~28dp(권장 44~48dp).
  시안 여백을 늘려야 해서 디자인 판단 필요 — 이번에는 Semantics 라벨만 적용.
- 다크 모드 대비: `muted`/`faint` 캡션이 WCAG AA 미달(2.3~2.8:1). `design_tokens.dart` 값 조정 필요(시안 영향).
- 미사용 자산: `assets/splash/logo_dark.png`, `assets/fonts/*.woff2`(번들 아님), `widgets/app_tooltip.dart` — 삭제는 보류.
- `android/gradle.properties` 의 `-Xmx8G -XX:MaxMetaspaceSize=4G` 는 로컬 머신 의존 값(다른 PC/CI 빌드 실패 소지).
- 분석 1회당 같은 이미지를 3개 엔드포인트에 업로드(`/predict-hier`·`/predict-with-regions`·`/predict-objects`).
  업로드 크기는 줄였지만 요청 수는 그대로 — 서버 통합 엔드포인트 검토 여지.
- "분석 취소" 시 진행 중인 HTTP 요청은 계속 전송됨(UI 오염은 없음).
