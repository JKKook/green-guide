/// 사용자 설정 영구 저장 (SharedPreferences 백엔드).
library;

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class SettingsStore {
  static const _keyApiUrl = 'api_base_url';
  static const _keyOnboardingDone = 'onboarding_done';
  static const _keyHapticsEnabled = 'haptics_enabled';
  static const _keyThemeMode = 'theme_mode';  // 'system' | 'light' | 'dark'
  static const _keyConsentAcceptedAt = 'consent_accepted_at';  // ISO8601 동의 시각

  /// 프로덕션 — HuggingFace Spaces 에 배포된 waste-api
  static const String defaultApiUrl = 'https://ethandev92-waste-api.hf.space';

  /// 로컬 개발 — adb reverse 로 PC 와 연결 시 (실기기/에뮬레이터 공통)
  static const String testModeApiUrl = 'http://localhost:8000';

  /// 에뮬레이터에서 호스트 PC 를 가리키는 특수 IP (참고용)
  static const String emulatorLocalApiUrl = 'http://10.0.2.2:8000';

  // 지역 선택 — 지자체 조례별 배출 규정 매핑용 (시나리오: 앱 진입 시 지역 선택)
  static const _keyRegionSido = 'region_sido';
  static const _keyRegionSigungu = 'region_sigungu';
  static const _keyRegionPromptShown = 'region_prompt_shown';

  Future<String> getApiUrl() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_keyApiUrl) ?? defaultApiUrl;
  }

  /// 선택된 지역 (sido, sigungu) — 미설정이면 null.
  Future<(String, String)?> getRegion() async {
    final prefs = await SharedPreferences.getInstance();
    final sido = prefs.getString(_keyRegionSido);
    final sigungu = prefs.getString(_keyRegionSigungu);
    if (sido == null || sigungu == null) return null;
    return (sido, sigungu);
  }

  Future<void> setRegion(String sido, String sigungu) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyRegionSido, sido);
    await prefs.setString(_keyRegionSigungu, sigungu);
  }

  /// 첫 진입 지역 선택 안내를 이미 보여줬는지 (스킵해도 재노출 안 함).
  Future<bool> isRegionPromptShown() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_keyRegionPromptShown) ?? false;
  }

  Future<void> setRegionPromptShown() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_keyRegionPromptShown, true);
  }

  Future<void> setApiUrl(String url) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyApiUrl, url);
  }

  Future<bool> isOnboardingDone() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_keyOnboardingDone) ?? false;
  }

  Future<void> setOnboardingDone() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_keyOnboardingDone, true);
  }

  /// 개인정보 수집·이용 동의 여부 (사진 업로드·재학습 사용 포함).
  Future<bool> isConsentAccepted() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_keyConsentAcceptedAt) != null;
  }

  Future<void> setConsentAccepted() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
      _keyConsentAcceptedAt, DateTime.now().toIso8601String(),
    );
  }

  static const _keyTipsNotification = 'tips_notification_enabled';
  static const _keyHousingType = 'housing_type';            // 'apartment' | 'house'
  static const _keyPickupWeekdays = 'pickup_weekdays';      // '2,5' (DateTime.weekday)
  static const _keyCollectionAlarmOptIn = 'collection_alarm_opt_in';
  static const _keyAiTrainingOptIn = 'ai_training_opt_in';

  /// 주거 형태 — 온보딩 ③ 세대 구분 (미설정이면 null).
  Future<HousingType?> getHousingType() async {
    final prefs = await SharedPreferences.getInstance();
    return switch (prefs.getString(_keyHousingType)) {
      'apartment' => HousingType.apartment,
      'house' => HousingType.house,
      _ => null,
    };
  }

  Future<void> setHousingType(HousingType type) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
        _keyHousingType, type == HousingType.apartment ? 'apartment' : 'house');
    appHousingType.value = type;
  }

  /// 우리 집 수거 요일 (1=월 ~ 7=일) — 주택·빌라 온보딩 ④ 에서 설정.
  Future<List<int>> getPickupWeekdays() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_keyPickupWeekdays);
    if (raw == null || raw.isEmpty) return const [];
    return [for (final s in raw.split(',')) ?int.tryParse(s)];
  }

  Future<void> setPickupWeekdays(List<int> weekdays) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyPickupWeekdays, weekdays.join(','));
    appPickupWeekdays.value = List.unmodifiable(weekdays);
  }

  /// 동의 시트 [선택] 항목 — 수거일 알림 수신 / 촬영 사진 AI 학습 활용.
  Future<bool> isCollectionAlarmOptIn() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_keyCollectionAlarmOptIn) ?? false;
  }

  Future<void> setCollectionAlarmOptIn(bool v) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_keyCollectionAlarmOptIn, v);
  }

  Future<bool> isAiTrainingOptIn() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_keyAiTrainingOptIn) ?? false;
  }

  Future<void> setAiTrainingOptIn(bool v) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_keyAiTrainingOptIn, v);
  }

  /// 온보딩 다시 보기 (개발자 옵션) — 완료 플래그만 내린다.
  Future<void> resetOnboarding() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_keyOnboardingDone);
  }

  /// 오늘의 팁 알림 (하루 한 번) 수신 여부.
  Future<bool> isTipsNotificationEnabled() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_keyTipsNotification) ?? false;
  }

  Future<void> setTipsNotificationEnabled(bool enabled) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_keyTipsNotification, enabled);
  }

  Future<bool> isHapticsEnabled() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_keyHapticsEnabled) ?? true;
  }

  Future<void> setHapticsEnabled(bool enabled) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_keyHapticsEnabled, enabled);
  }

  Future<ThemeMode> getThemeMode() async {
    final prefs = await SharedPreferences.getInstance();
    return switch (prefs.getString(_keyThemeMode)) {
      'light' => ThemeMode.light,
      'dark' => ThemeMode.dark,
      _ => ThemeMode.system,
    };
  }

  Future<void> setThemeMode(ThemeMode mode) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyThemeMode, switch (mode) {
      ThemeMode.light => 'light',
      ThemeMode.dark => 'dark',
      ThemeMode.system => 'system',
    });
  }

  Future<void> reset() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.clear();
  }

  // ---- 다른 모듈이 SharedPreferences 를 직접 만지지 않도록 흡수한 키들 ----
  static const _keyDevOptions = 'dev_options_enabled';
  static const _keyCachedClasses = 'cached_labels_classes_json';
  static const _keyReminders = 'collection_reminders';

  /// 설정 > 버전 7회 탭으로 여는 개발자 옵션.
  Future<bool> isDevOptionsEnabled() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_keyDevOptions) ?? false;
  }

  Future<void> setDevOptionsEnabled(bool enabled) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_keyDevOptions, enabled);
  }

  /// 마지막으로 성공한 `/labels` 응답(JSON 문자열) — 오프라인 부팅용 캐시.
  Future<String?> getCachedClassesJson() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_keyCachedClasses);
  }

  Future<void> setCachedClassesJson(String json) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyCachedClasses, json);
  }

  /// 수거일 알림 목록(JSON 문자열) — 직렬화는 ReminderStore 담당.
  Future<String?> getRemindersJson() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_keyReminders);
  }

  Future<void> setRemindersJson(String json) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyReminders, json);
  }
}


/// 주거 형태 — 아파트·오피스텔 / 주택·빌라 (배출 방식이 다름).
enum HousingType { apartment, house }


/// 앱 전역에서 테마 모드 변경 시 즉시 반영하기 위한 ValueNotifier.
/// main.dart 가 listen 해서 MaterialApp.themeMode 를 rebuild.
final ValueNotifier<ThemeMode> appThemeMode = ValueNotifier(ThemeMode.system);

/// 앱 시작 시 한 번 호출 — 저장된 값으로 초기화.
Future<void> initAppThemeMode() async {
  appThemeMode.value = await SettingsStore().getThemeMode();
}

/// 주거 형태 — 아파트면 수거 일정 UI(홈 주간 스트립·검색 일정 그룹)를 숨긴다.
/// 설정/온보딩에서 바뀌면 즉시 반영되도록 전역 ValueNotifier 로 공유.
final ValueNotifier<HousingType?> appHousingType = ValueNotifier(null);

Future<void> initAppHousingType() async {
  appHousingType.value = await SettingsStore().getHousingType();
  appPickupWeekdays.value = List.unmodifiable(await SettingsStore().getPickupWeekdays());
}

/// 우리 집 수거 요일 (1=월~7=일) — 주택·빌라 사용자 지정. 비어 있으면 동네 기본값.
final ValueNotifier<List<int>> appPickupWeekdays = ValueNotifier(const []);

/// 수거 일정 관련 UI 노출 여부 — 아파트·오피스텔은 상시 배출이라 숨김.
bool get showsCollectionSchedule =>
    appHousingType.value != HousingType.apartment;
