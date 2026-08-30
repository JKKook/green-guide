# waste_app (그린가이드 AI)

GreenGuide AI 의 네 번째 서브 프로젝트. 사용자가 폐기물 사진을 찍거나 갤러리에서 선택해 [`waste-api`](../waste-api) 로 보내면, 6-class 분류 결과와 함께 한국어 분리수거 상세 안내를 보여주는 Flutter 모바일 앱.

```
[사용자]
   │  사진 촬영 또는 선택
   ▼
[waste_app (Flutter)]                  ← 이 프로젝트
   │  POST /predict (multipart)
   ▼
[waste-api (FastAPI + ONNX CNN)]
   │  분류 결과 JSON
   ▼
[waste_app 결과 화면]
   - 클래스명 + 신뢰도
   - 한국어 분리수거 가이드
   - 전체 6개 클래스 확률 분포
```

---

## 목차

1. [프로젝트 위치](#프로젝트-위치)
2. [핵심 결정 사항](#핵심-결정-사항)
3. [화면 흐름](#화면-흐름)
4. [설치 및 빌드](#설치-및-빌드)
5. [실행](#실행)
6. [API 서버 연동](#api-서버-연동)
7. [구성 요소](#구성-요소)
8. [Google Play 배포 준비 (V2)](#google-play-배포-준비-v2)
9. [트러블슈팅](#트러블슈팅)
10. [알려진 한계와 향후 개선](#알려진-한계와-향후-개선)
11. [프로젝트 구조](#프로젝트-구조)

---

## 프로젝트 위치

```
GreenGuide AI
├── waste-preprocessor     (1) 수집·전처리·벡터화          완성
├── waste-classifier       (2) 지도학습 분류기 + ONNX      완성 (CNN 92.35%)
├── waste-api              (3) HTTP 추론 서버              완성 (FastAPI)
└── waste_app              (4) Flutter 모바일 클라이언트   현재
```

---

## 핵심 결정 사항

| 분야 | 선택 | 이유 |
|---|---|---|
| 프레임워크 | **Flutter 3.41 / Dart 3.11** | 단일 코드로 Android·iOS 모두. Material 3 기본 지원 |
| State 관리 | **setState** (그 외 패키지 없음) | MVP 규모 — Provider/Riverpod 도입 비용 회피 |
| HTTP | `http` 패키지 | 표준, 가벼움 |
| 이미지 picker | `image_picker` | 카메라·갤러리·권한 자동 처리 |
| 설정 저장 | `shared_preferences` | API URL 등 영구 저장 |
| 디자인 | **Material 3** + 클래스별 brand color | 일관된 modern look |
| 다크모드 | **자동** (시스템 따름) | Material 3 색조합 자동 처리 |
| 타겟 플랫폼 | **Android** (Google Play) | 사용자 요구. iOS 디렉토리는 미사용 |
| 클래스 메타 | 6개 클래스에 한국어명·아이콘·색상·가이드 | 단순 라벨 표시 → 실용 정보 |

---

## 화면 흐름

```
HomeScreen                       ResultScreen
┌───────────────────┐            ┌───────────────────┐
│ [이미지 미리보기] │            │ [큰 이미지]        │
│                   │            │                   │
│ 📷 촬영  🖼️ 갤러리│ ───분류──> │ 🥤 플라스틱 95.2% │
│                   │            │ 배출 장소         │
│   [분류하기]      │            │ 배출 방법 ✓✓✓     │
│                   │            │ 주의사항 ⚠⚠⚠     │
│   ⚙️ 설정         │            │ 전체 확률 분포    │
└───────────────────┘            │   [다시 분류 ↺]   │
       ▲                         └───────────────────┘
       │ ⚙️ 설정
       ▼
SettingsScreen
┌───────────────────┐
│ API URL 입력      │
│ [연결 테스트][저장]│
└───────────────────┘
```

### 분리수거 안내 — 6개 클래스별 상세

각 클래스마다 다음 정보를 표시:

| 클래스 | 한글명 | 색상 | 예시 |
|---|---|---|---|
| cardboard | 종이상자 | 갈색 계열 | 택배 박스, 골판지 |
| glass | 유리병 | 청록 | 음료병, 화장품 유리병 |
| metal | 캔·금속 | 회색 | 음료수 캔, 통조림 |
| paper | 종이 | 어두운 갈색 | 신문, 책, 인쇄물 |
| plastic | 플라스틱 | 파랑 | 페트병, 플라스틱 용기 |
| trash | 일반쓰레기 | 회색 | 재활용 불가 |

각 항목에 **배출 장소 / 배출 방법(3개) / 주의사항(3개)** 표시.

---

## 설치 및 빌드

### 1. 사전 요구 사항

| 도구 | 버전 | 확인 |
|---|---|---|
| Flutter | 3.41+ | `flutter --version` |
| Dart | 3.11+ (Flutter 와 함께 설치) | `dart --version` |
| Android SDK | 36+ | `flutter doctor` |
| JDK | 17+ (Android Studio 가 번들) | `java -version` |
| Android Studio | 최신 | https://developer.android.com/studio |

`flutter doctor` 결과가 `[✓] Android toolchain` 이면 빌드 가능.

### 2. 의존성 설치

```bash
cd /Users/whdrnr01/ai/waste_app
flutter pub get
```

### 3. 정적 검사 + 테스트

```bash
flutter analyze     # No issues 기대
flutter test        # All tests passed 기대
```

### 4. APK 빌드 (Android)

```bash
# 디버그 (개발용)
flutter build apk --debug
# 결과: build/app/outputs/flutter-apk/app-debug.apk

# 릴리스 (배포 직전)
flutter build apk --release
# 결과: build/app/outputs/flutter-apk/app-release.apk
```

### 5. AAB (Google Play 제출용)

```bash
flutter build appbundle --release
# 결과: build/app/outputs/bundle/release/app-release.aab
```

릴리스 빌드는 서명 키 필요 — [Google Play 배포 준비](#google-play-배포-준비-v2) 참고.

---

## 실행

### 옵션 A — Android 에뮬레이터

```bash
# 1) Android Studio → Device Manager → AVD 생성·실행
# 2) 에뮬레이터가 떠 있는 상태에서:
flutter run
```

기본 API URL `http://10.0.2.2:8000` 이 에뮬레이터에서 자동으로 호스트 PC 의 `localhost:8000` 을 가리킨다.

### 옵션 B — 실기기 (USB 디버깅)

1. 안드로이드 폰: 설정 → 휴대전화 정보 → 빌드 번호 7번 탭 → 개발자 옵션 활성화 → USB 디버깅 ON
2. USB 로 PC 연결
3. `flutter devices` 로 인식 확인
4. `flutter run`
5. **앱 안에서 ⚙️ 설정 → API URL 을 PC 의 LAN IP 로 변경**
   - 현재 PC IP: `172.30.1.26` (검출됨)
   - 입력값 예시: `http://172.30.1.26:8000`
6. **연결 테스트** 버튼으로 검증 → 성공이면 저장

### 두 환경 모두에서 — waste-api 서버 실행 필수
```bash
cd /Users/whdrnr01/ai/waste-api
.venv/bin/python main.py
# Listening on http://0.0.0.0:8000
```

---

## API 서버 연동

`SettingsScreen` 에서 다음 두 가지를 관리:
- **API Base URL** (`SettingsStore` → `shared_preferences`)
- **연결 테스트** — `/health` 호출해서 200 OK 확인

| 환경 | URL |
|---|---|
| Android 에뮬레이터 | `http://10.0.2.2:8000` (기본값) |
| 실기기 (LAN) | `http://172.30.1.26:8000` (PC IP) |
| 프로덕션 (배포 후) | `https://your-domain.com` |

> **HTTP** 호출은 `AndroidManifest.xml` 의 `android:usesCleartextTraffic="true"` 가 허용. production HTTPS 전환 시 이 옵션 제거 권장.

---

## 구성 요소

| 경로 | 역할 |
|---|---|
| `lib/main.dart` · `lib/app.dart` | 진입점(초기화·서버 웜업) / `GreenGuideApp`(MaterialApp·테마·로케일) |
| `lib/core/di/app_scope.dart` | **의존 접근점** — `AppScope.settings/history/prediction`, `AppScope.api()` |
| `lib/core/ui/ds_card.dart` | 디자인 시스템 카드(`DsCard`: elevated/tinted/radius) |
| `lib/core/feedback/app_snackbar.dart` | `showAppSnackBar` / `showAppErrorSnackBar`(friendlyError 경유) |
| `lib/core/log.dart` | `appLog` — 릴리즈에서 출력하지 않는 진단 로그 |
| `lib/theme/` | `app_theme.dart`(ThemeData·색 램프·간격/반경 토큰·잉크 상수), `design_tokens.dart`(`DsTokens` 라이트/다크 대응색) |
| `lib/api/` | `WasteApiClient`, 응답 모델(`Prediction`·`PredictObjects`·…) |
| `lib/data/` | `SettingsStore`(SharedPreferences 단일 접근), `HistoryRepository`(sqflite), 클래스 메타·신뢰도·화질 |
| `lib/services/` | `PredictionService`(계층 분류 → 구버전 fallback), `ClassLoader`, `ServerWarmup`, 안정도 감지 |
| `lib/features/<기능>/` | 화면 + 그 화면 전용 위젯(`widgets/`) — capture · result(+`ResultController`) · home · history · search · schedule · settings · onboarding · shell |
| `lib/widgets/` | 기능에 묶이지 않는 범용 위젯(animated_entry·criteria_sheet·hier_badge·korea_map·region_picker) |
| `test/` | 순수 로직(`data/`·`api`), `core/`(AppScope), `features/`(ResultController), `widgets/`(화면·골든) — `helpers/test_env.dart` 로 플러그인 가짜 구성 |

### 공통단(core) 사용 규칙

- 설정·DB·서비스·API 클라이언트는 **직접 생성하지 않고 `AppScope`** 로 접근한다. `SharedPreferences` 는 `SettingsStore` 만 만진다.
- 스낵바는 `showAppSnackBar`, 카드 컨테이너는 `DsCard`, 로그는 `appLog`. 색은 `DsTokens`/`app_theme` 상수, 간격·반경은 `kSpace*`/`kRadius*`.
- `core/` 에 넣는 기준: 2개 이상 feature 가 쓰고, 도메인(분류·지역·일정)을 모르며, 자체 상태/네트워크가 없다.
- 검증: `flutter analyze && flutter test` (골든 갱신은 `flutter test --update-goldens test/widgets/golden_test.dart`). 리팩토링 이력·계획은 `REFACTORING_GUIDE.md`.

---

## Google Play 배포 준비 (V2)

현재는 개발용 임시값 사용. 정식 배포 전 변경 필요:

### 1. 패키지 이름 (Application ID)
현재: `com.greenguide.waste_app`
- `android/app/build.gradle.kts` 의 `applicationId` 확인
- Google Play 에 한 번 등록되면 변경 불가능

### 2. 앱 아이콘
현재: Flutter 기본 아이콘
- `flutter_launcher_icons` 패키지로 1024×1024 PNG 자동 생성 권장
- 또는 `android/app/src/main/res/mipmap-*/` 폴더에 직접 배치

### 3. 서명 키 (Release Build 필수)
```bash
keytool -genkey -v -keystore ~/upload-keystore.jks \
  -keyalg RSA -keysize 2048 -validity 10000 -alias upload
```
이후 `android/key.properties` 와 `android/app/build.gradle.kts` 에 키 정보 등록 — Flutter 공식 가이드 참조.

### 4. ProGuard / R8 코드 난독화
릴리스 빌드 시 자동 적용. 문제 시 `android/app/proguard-rules.pro` 에 예외 추가.

### 5. Play Store 등재 정보
- 앱 이름·설명·키워드
- 스크린샷 (휴대전화·태블릿 최소 2장씩)
- 개인정보 처리방침 URL (필수)
- 콘텐츠 등급 설문
- 가격·국가 설정

### 6. AAB 업로드
```bash
flutter build appbundle --release
```
`build/app/outputs/bundle/release/app-release.aab` 를 Google Play Console 에 업로드.

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `Unable to locate a Java Runtime` | JDK 없음 | Android Studio 설치 (JDK 번들) 또는 `brew install openjdk@17` |
| `[✗] Android toolchain` | SDK / 라이선스 미설정 | `flutter doctor --android-licenses` 로 라이선스 동의 |
| 에뮬레이터에서 `Connection refused` | API 서버 미실행 또는 URL 오타 | PC 에서 waste-api 실행 확인, URL `http://10.0.2.2:8000` |
| 실기기에서 `Connection timed out` | PC IP 변경 또는 같은 LAN 아님 | `ipconfig getifaddr en0` 로 IP 재확인, 같은 Wi-Fi 연결 확인 |
| `CleartextNotPermitted` | HTTPS 강제 | `AndroidManifest.xml` 에 `android:usesCleartextTraffic="true"` 확인 |
| 사진이 선택 안됨 | 권한 거부 | 안드로이드 설정 → 앱 → 권한 → 카메라/사진 허용 |
| `RenderFlex overflowed` (테스트) | 화면 비례 안 맞음 | `SingleChildScrollView` 로 감싸기 (이미 적용됨) |
| 분류가 자꾸 틀림 | 모델 정확도 92% 한계 | waste-classifier 의 CNN 개선 (data 증강·더 큰 모델) |

---

## 알려진 한계와 향후 개선

| 항목 | 현재 | 개선 방향 |
|---|---|---|
| 인증 | 없음 (API 측도 없음) | API 키·OAuth 추가 |
| 오프라인 추론 | 불가 (서버 필수) | ONNX 모델 번들 → `onnxruntime` for Flutter |
| 예측 history | 저장 안 됨 | 로컬 SQLite 또는 Supabase 동기화 |
| i18n | 한국어 only | `flutter_localizations` + ARB 파일로 다국어 |
| 권한 안내 | OS 기본 다이얼로그 | `permission_handler` 로 친절한 사전 안내 |
| 사진 EXIF | 회전 자동 처리 안 됨 | `image` 패키지로 EXIF orientation 보정 |
| 결과 공유 | 없음 | `share_plus` 로 SNS·메시지 공유 |
| 다크모드 | 시스템 자동 | 앱 내 설정으로 토글 가능 |

---

## 프로젝트 구조

```
waste_app/
├── lib/
│   ├── main.dart · app.dart
│   ├── core/            # 공통단 — di/ ui/ feedback/ log.dart
│   ├── theme/           # ThemeData · 토큰
│   ├── api/  data/  services/
│   ├── features/        # 기능별 화면 + 전용 위젯
│   │   ├── capture/  result/  home/  history/  search/
│   │   ├── schedule/  settings/  onboarding/  shell/
│   └── widgets/         # 범용 위젯
├── test/                # data/ core/ features/ widgets/(goldens/) helpers/
├── REFACTORING_GUIDE.md
└── pubspec.yaml
```
