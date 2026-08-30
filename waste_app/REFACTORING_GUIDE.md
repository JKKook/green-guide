# waste_app 리팩토링 작업 가이드

> 초점: **코드 품질 최적화 + 공통단(common layer) 구성**
> 원칙: 동작 변경 0 · 각 단계가 독립 커밋 · 매 단계 `flutter analyze` + `flutter test` 통과
> 작성일: 2026-08-30 · 기준 커밋: `dc54414`

---

## 0. 현재 상태 진단 (실측)

| 항목 | 수치 | 판단 |
|---|---|---|
| `lib/` 총 줄 수 | 15,184줄 / 42파일 | — |
| 1,000줄 초과 파일 | `result_modal.dart` 2,628 · `onboarding_screen` 1,367 · `history_screen` 1,313 · `collection_schedule_screen` 1,248 | **분해 필요** |
| `SettingsStore()` 직접 생성 | 24회 / 14파일 | 접근점 단일화 필요 |
| `WasteApiClient(...)` 직접 생성 | 10회 / 6파일 (`PredictionService`가 있음에도 우회) | 서비스 경유로 통일 |
| `HistoryRepository()` 직접 생성 | 5회 / 4파일 | 동일 |
| `BoxDecoration(` 직접 작성 | 107회 | 카드/칩 공통 위젯화 |
| 매직 padding/radius 숫자 | 317회 | 토큰(`kSpace*`, `kRadius*`)은 186회만 사용 → **절반 미적용** |
| `theme/` 밖 하드코딩 `Color(0x..)` | ~120회 (`result_modal` 35, `waste_info` 21, `history` 17 …) | `DsTokens`로 흡수 |
| `showSnackBar` 호출 | 26회, `behavior`·문구 스타일 제각각 | 공통 헬퍼 |
| `if (!mounted)` 가드 | 38회 (result_modal 12) | 비동기 패턴 헬퍼 |
| 상태관리 | `setState` + 전역 `ValueNotifier` 3개 (`appThemeMode`, `historyRevision`, …) | 유지 (라이브러리 도입 금지) |
| 테스트 | 3개 (순수 로직 2, 위젯 1) | 리팩토링 전 안전망 보강 필요 |
| `flutter analyze` | No issues | 린트 강화 여지 있음 |

**핵심 문제 3가지**
1. **의존 접근 경로 분산** — 설정/네트워크/DB 인스턴스가 화면·위젯 곳곳에서 직접 생성됨. 테스트 불가능, `baseUrl` 변경 시 동기화 위험.
2. **디자인 토큰 반쯤 적용** — `app_theme.dart`/`design_tokens.dart`에 토큰이 있지만 화면별로 숫자·색을 다시 씀. 다크모드 불일치 원인.
3. **거대 파일** — `result_modal.dart` 한 파일에 화면 로직·25개 위젯·스냅샷 모델이 공존. 변경 영향 범위 파악 불가.

---

## 1. 목표 구조

```
lib/
├── main.dart
├── app.dart                     # GreenGuideApp (main에서 분리)
├── core/                        # ★ 공통단 — 화면·기능에 무관한 것만
│   ├── di/
│   │   └── app_scope.dart       # 싱글턴 접근점 (Settings/Api/History/Prediction)
│   ├── theme/                   # 기존 theme/ 이동
│   │   ├── app_theme.dart
│   │   ├── design_tokens.dart   # DsTokens + kSpace/kRadius (여기로 모음)
│   │   └── app_colors.dart      # 분류 카테고리 색 등 도메인 색 (waste_info에서 분리)
│   ├── ui/                      # ★ 공통 위젯 (디자인 시스템 프리미티브)
│   │   ├── ds_card.dart         # BoxDecoration 107회 → 1곳
│   │   ├── ds_chip.dart
│   │   ├── ds_section_label.dart
│   │   ├── ds_primary_button.dart
│   │   ├── ds_banner.dart       # Quality/Uncertain/RegionRescue 배너 공통 골격
│   │   └── ds_bottom_sheet.dart # showModalBottomSheet 공통 shape/padding
│   ├── feedback/
│   │   ├── app_snackbar.dart    # showAppSnackBar(context, msg, {kind})
│   │   └── app_dialogs.dart     # confirm 다이얼로그
│   ├── async/
│   │   └── safe_state.dart      # mounted 가드 mixin
│   ├── log.dart                 # kDebugMode 로거 (prediction_service의 _log 승격)
│   └── extensions/
│       └── context_ext.dart     # context.tokens, context.colors, context.text
├── api/                         # 유지 (client + models)
├── data/                        # 유지 — 순수 데이터/저장소만
├── services/                    # 유지
├── features/                    # 기존 screens/ + widgets/ 를 기능 단위로 재배치
│   ├── home/
│   ├── capture/                 # live_camera, gallery_confirm, capture_entry_sheet
│   ├── result/                  # ★ result_modal 분해 대상
│   │   ├── result_modal.dart            # 진입점 + 상태 (≤400줄 목표)
│   │   ├── result_controller.dart       # 비동기 fetch 6종 + 스냅샷 스택
│   │   ├── models/view_snapshot.dart
│   │   └── widgets/
│   │       ├── prediction_card.dart
│   │       ├── objects_card.dart        # _ObjectsCard/_ObjectTile/_ObjectMarker/_TapMarker
│   │       ├── multi_material_card.dart # _MultiMaterialCard/_MaterialMethodTile/_RegionsView/_RegionBadge
│   │       ├── analysis_loading.dart    # _AnalysisLoading + _BlobIconPainter
│   │       ├── banners.dart             # Quality/Uncertain/RegionRescue/Reject
│   │       ├── guide_card.dart
│   │       ├── evidence_chips.dart
│   │       └── explain_button.dart
│   ├── history/
│   ├── search/
│   ├── schedule/
│   ├── settings/
│   ├── onboarding/
│   └── shell/                   # main_shell, splash
└── widgets/                     # 남는 범용 위젯만 (korea_map, region_picker, animated_entry, app_tooltip)
```

**"공통단에 넣는 기준"** — 아래 3개를 모두 만족할 때만 `core/`:
- 2개 이상 feature에서 쓴다 (또는 쓰게 될 것이 확실하다)
- 도메인(쓰레기 분류·지역·일정)을 모른다
- 자체 상태/네트워크가 없다 (있으면 `services/`)

---

## 2. 단계별 작업 계획

각 단계는 **독립 PR/커밋**. `→ verify:` 는 완료 판정 기준.

### Phase 0 — 안전망 (반드시 먼저)

| # | 작업 | verify |
|---|---|---|
| 0-1 | `analysis_options.yaml` 린트 강화 (아래 §3) | `flutter analyze` 경고 목록 → 기준선 파일로 저장 |
| 0-2 | 골든/위젯 테스트 추가: `MainShell` 탭 전환, `ResultModal` 로딩→성공/에러, `HistoryScreen` 빈 상태/목록 | `flutter test` 통과, 골든 파일 커밋 |
| 0-3 | 순수 로직 테스트 보강: `friendlyError`, `confidence.dart`, `image_quality.dart`, `collection_schedule.dart` | 각 함수 최소 1 케이스 |
| 0-4 | 릴리즈 APK 빌드 1회 → 기준 크기/기동 시간 기록 | `flutter build apk --release` 성공 |

> 안전망 없이 Phase 2 이후 진행 금지. 리팩토링의 "동작 변경 0"은 테스트로만 증명된다.

**Phase 0 완료 (2026-08-30)** — 테스트 34개(순수 27 · 위젯 5 · 골든 2×라이트/다크), `flutter analyze` 0 issues, 릴리즈 APK **69.4MB** (기준선).
- 테스트 환경: `test/helpers/test_env.dart` — SharedPreferences 목, sqflite ffi, path_provider/package_info 가짜. `settleIo(tester)` 로 isolate·파일 I/O 를 흘려보낸다(가짜 시계만 pump 하면 영원히 대기).
- `sqflite_common_ffi` 는 **2.3.6 고정** — 2.4.x 가 끌어오는 `sqlite3 3.x` 의 native-asset 훅이 `flutter test` 로딩 단계에서 멈춤.
- 발견: `HistoryRepository._db` static 캐시에 close/reset 이 없어 테스트 간 DB 경로 변경 불가 → **Phase 1-3 에서 `AppScope.history` 로 옮기며 `reset()` 추가**.
- `dart format` 은 `lib/` 42파일 중 37개가 미포맷 상태. 일괄 포맷 커밋은 diff 오염이므로 **각 Phase 에서 건드리는 파일만** 포맷한다(DoD 의 format 클린은 Phase 5 시점 기준).

### Phase 1 — 의존 접근점 단일화 (`core/di`)

라이브러리(get_it/riverpod) **도입하지 않는다**. 앱 규모(4만 줄 미만, 서비스 4개)에 과하다.

```dart
// core/di/app_scope.dart
class AppScope {
  AppScope._();
  static final settings = SettingsStore();
  static final history = HistoryRepository();
  static final prediction = PredictionService();
  /// baseUrl 은 설정에서 매번 읽는다 — 개발자 옵션에서 바뀔 수 있음.
  static Future<WasteApiClient> api({Duration? timeout}) async =>
      WasteApiClient(baseUrl: await settings.getApiUrl(), timeout: timeout ?? const Duration(seconds: 30));
}
```

| # | 작업 | verify |
|---|---|---|
| 1-1 | `AppScope` 추가, `SettingsStore()` 24곳 → `AppScope.settings` | `grep -rn "SettingsStore()" lib` 결과 1개(정의)뿐 |
| 1-2 | `WasteApiClient(` 직접 생성 10곳 → `AppScope.api()` 또는 `PredictionService` 메서드 추가 (`fetchRegionInfo`, `predictWithRegions`, `predictObjects`, `explain` 등 result_modal이 직접 부르는 것) | `grep -rn "WasteApiClient(" lib` 결과 정의 + AppScope 뿐 |
| 1-3 | `HistoryRepository()` 5곳 → `AppScope.history` | 동일 grep |
| 1-4 | `SharedPreferences` 직접 접근 3곳(`settings_screen`, `collection_schedule`, `class_loader`) → `SettingsStore` 메서드로 흡수 | `grep -rl SharedPreferences lib` = `settings_store.dart` 1개 |
| 1-5 | `main.dart`의 `unawaited` 자작 함수 제거 → `dart:async` import; `GreenGuideApp` → `app.dart` | analyze 클린 |

### Phase 2 — 공통 UI 프리미티브 (`core/ui`, `core/feedback`)

**순서 원칙**: 먼저 *가장 많이 반복되는 패턴 1개*를 위젯화하고 전 화면에 치환한 뒤 다음으로. 한 번에 여러 프리미티브를 만들지 않는다.

| # | 작업 | 대상 수 | verify |
|---|---|---|---|
| 2-1 | `showAppSnackBar(context, String, {AppSnackKind kind})` — floating 고정, 에러는 `friendlyError` 통과 | 26곳 | `grep -rn "showSnackBar(" lib` = 헬퍼 내부 1개 |
| 2-2 | `DsCard` — border/radius/shadow 3종 변형(`flat`/`elevated`/`accent`) | `BoxDecoration` 107 → 목표 ≤30 (커스텀 페인터·그라디언트 제외) | 골든 테스트 diff 0 |
| 2-3 | `DsChip` (accentChipBg/Border/Text 조합 반복) | ~20곳 | 동일 |
| 2-4 | `DsSectionLabel`, `DsPrimaryButton` — 이미 각 화면에 `_SectionLabel`, `_PrimaryButton` private로 존재 → 승격 | 각 1→N | 화면별 private 정의 삭제 |
| 2-5 | `showDsBottomSheet(...)` — `showModalBottomSheet` shape/padding/handle 공통 | 9곳 | 동일 |
| 2-6 | `context_ext.dart`: `context.t` (DsTokens), `context.cs` (ColorScheme), `context.tt` (TextTheme) | — | `DsTokens.of(context)` 호출 → 확장으로 치환 |

### Phase 3 — 디자인 토큰 완전 적용 (`core/theme`)

| # | 작업 | verify |
|---|---|---|
| 3-1 | `theme/` 밖 `Color(0x..)` 전수 조사 → 의미별로 `DsTokens` getter 추가(예: `barBg`, `shadow`, `overlay`) 또는 `AppColors`(분류 카테고리 색) | `grep -rEn "Color\(0x" lib --include='*.dart' \| grep -v core/theme` 결과 0 (커스텀 페인터 예외 명시) |
| 3-2 | `main_shell.dart:48` 처럼 `t.dark ? Color(..) : kNeutral100` 삼항 반복 → 토큰 getter | 삼항 `t.dark ?` 패턴 화면에서 0 |
| 3-3 | 매직 padding/radius → `kSpace*`/`kRadius*`. **토큰에 없는 값(11, 14, 18 등)은 시안 값이면 토큰 추가, 아니면 근사 토큰으로 정규화** — 골든으로 픽셀 변화 확인 후 결정 | 매직 숫자 317 → ≤80 |
| 3-4 | `kSpace*`/`kRadius*`/모션 상수 → `design_tokens.dart`로 이동, `app_theme.dart`는 `ThemeData` 빌드만 | `app_theme.dart` ≤200줄 |

### Phase 4 — 거대 파일 분해 (`features/`)

**`result_modal.dart` 먼저** (가장 크고 핵심 경로). 나머지는 같은 레시피 반복.

| # | 작업 | verify |
|---|---|---|
| 4-1 | private 위젯 25개를 `features/result/widgets/`로 **이동만** (public 전환, 로직 무변경) | 파일당 ≤400줄, 골든 diff 0 |
| 4-2 | `_ResultModalState`의 비동기 fetch 6종(`_classify`, `_fetchRegions`, `_fetchObjects`, `_fetchRegionInfo`, `_assessQuality`, `_resolveImageSize`) + 스냅샷 스택 → `ResultController extends ChangeNotifier` | 위젯은 `ListenableBuilder`만, State 파일 ≤400줄 |
| 4-3 | `ResultController` 단위 테스트 — fake `PredictionService` 주입 (Phase 1 덕분에 가능) | 로딩 완료 조건(`_loading` getter), undo 스택 10개 제한, hier 404 fallback |
| 4-4 | `onboarding_screen`(1,367) → 스텝별 위젯 파일 분리 (`_RegionStep`, `_PickupSetupStep` 등 이미 private로 존재) | ≤400줄 |
| 4-5 | `history_screen`(1,313), `collection_schedule_screen`(1,248) → 시트/카드 분리 | ≤500줄 |
| 4-6 | `screens/` + `widgets/` → `features/<name>/` 재배치 (IDE 이동 리팩터로 import 자동 갱신) | analyze 클린, 골든 전체 통과 |

### Phase 5 — 마무리 품질

| # | 작업 | verify |
|---|---|---|
| 5-1 | `if (!mounted)` 38곳 → `SafeStateMixin.runIfMounted()` 또는 `ResultController` 이동으로 자연 소멸 | ≤15 |
| 5-2 | `debugPrint`/`_log` → `core/log.dart` 단일 로거 | `grep -rn "debugPrint\|print(" lib` = log.dart 1개 |
| 5-3 | `catch (_)` 삼킴 7곳 검토 — 의도적 무시면 주석, 아니면 로거 경유 | 주석 없는 빈 catch 0 |
| 5-4 | 죽은 코드 정리: `SettingsStore.emulatorLocalApiUrl`(참고용), 미사용 import | `dart fix --dry-run` 결과 0 |
| 5-5 | `README.md`에 폴더 구조·공통단 사용 규칙 반영 | — |

---

## 3. 린트 강화안 (`analysis_options.yaml`)

```yaml
include: package:flutter_lints/flutter.yaml

analyzer:
  errors:
    unused_import: error
    unused_element: error
    dead_code: warning

linter:
  rules:
    prefer_const_constructors: true
    prefer_const_literals_to_create_immutables: true
    prefer_final_locals: true
    avoid_redundant_argument_values: true
    unnecessary_lambdas: true
    use_key_in_widget_constructors: true
    sort_constructors_first: true
    directives_ordering: true
    avoid_print: true
    prefer_single_quotes: true
    require_trailing_commas: true      # dart format 결과 안정화
    use_build_context_synchronously: true
```

Phase 0에서 켜고, 경고는 **해당 파일을 건드리는 Phase에서** 함께 정리한다 (전역 일괄 수정 커밋 금지 — diff 오염).

---

## 4. 작업 규칙

1. **한 커밋 = 한 종류의 변경.** "SnackBar 헬퍼 도입 + 전 화면 치환"은 OK, "치환하면서 padding도 정리"는 NO.
2. **이동과 수정을 분리.** 파일 이동(4-1, 4-6)은 로직 무변경 커밋으로. 리뷰어가 `git diff -M`으로 rename 인식 가능해야 함.
3. **골든 테스트가 red면 멈춘다.** 픽셀 변화가 의도(토큰 정규화)인지 실수인지 판정 후 골든 갱신 여부 결정. 갱신 시 커밋 메시지에 사유.
4. **공통단 승격 전 3-회 규칙.** 2곳에서만 쓰이면 아직 승격하지 않는다(추상화 비용 > 중복 비용). 단 Phase 2 표의 항목은 실측 반복 수가 충분해 예외.
5. **라이브러리 추가 없음.** 상태관리·DI·라우팅 패키지는 이번 스코프 밖. `ChangeNotifier`+`ListenableBuilder`, 정적 `AppScope`로 충분.
6. **주석 보존.** 기존 코드의 한국어 설계 주석(왜 이렇게 했는지)은 이동 시 반드시 함께 이동.
7. 검증 명령 (PATH flutter는 구버전 — 반드시 전용 SDK):
   ```bash
   export PATH="/Users/ethan/development/flutter/bin:$PATH"
   flutter analyze && flutter test && dart format --set-exit-if-changed lib test
   ```
   macOS 실행(`flutter run -d macos`)·폰 릴리즈 설치는 사용자 요청 시에만.

---

## 5. 완료 기준 (Definition of Done)

- [ ] 파일 최대 줄 수 ≤ 500 (커스텀 페인터·데이터 테이블 제외)
- [ ] `SettingsStore()`, `WasteApiClient(`, `HistoryRepository()`, `SharedPreferences` 직접 참조가 각각 정의처 + `AppScope`에만 존재
- [ ] `core/theme` 밖 `Color(0x` 0건 (예외 목록 문서화)
- [ ] `showSnackBar` 직접 호출 0건
- [ ] `BoxDecoration(` ≤ 30건
- [ ] 골든 테스트 4개 화면 이상, `ResultController` 단위 테스트 존재
- [ ] `flutter analyze` 0 issues (강화 린트 기준), `dart format` 클린
- [ ] 릴리즈 APK 크기 Phase 0 대비 ±2% 이내

---

## 6. 예상 순서·규모

| Phase | 예상 diff 규모 | 리스크 | 선행 |
|---|---|---|---|
| 0 안전망 | +600줄 (테스트) | 낮음 | — |
| 1 DI | ±150줄 | 낮음 | 0 |
| 2 공통 UI | −800 / +400줄 | **중** (픽셀 변화) | 0, 1 |
| 3 토큰 | ±500줄 | 중 | 2 |
| 4 분해 | 이동 위주, 순수 변경 ±300줄 | 중 (컨트롤러 추출) | 1, 2 |
| 5 마무리 | ±200줄 | 낮음 | 4 |

Phase 1→2→3은 순서 고정. Phase 4의 4-1(이동만)은 Phase 2와 병행 가능.
