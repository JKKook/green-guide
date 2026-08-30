/// 우리 동네 분리수거 일정 — 홈 헤드라인·주간 스트립·수거일 안내 화면 공용.
/// 지역 수거일 공공데이터 연동 전 기본값 (시안 4a 기준).
library;

import 'dart:convert';

import '../core/di/app_scope.dart';
import 'settings_store.dart';

/// 요일별 배출 품목 종류.
enum PickupKind { plasticVinyl, paperBox, general, none }

/// 월~일 배출 품목 기본값 — 시안 4a 의 주간 도트 배치.
const List<PickupKind> kDefaultWeekSchedule = [
  PickupKind.paperBox,      // 월
  PickupKind.plasticVinyl,  // 화
  PickupKind.none,          // 수
  PickupKind.paperBox,      // 목
  PickupKind.plasticVinyl,  // 금
  PickupKind.none,          // 토
  PickupKind.general,       // 일
];

const List<String> kDayNames = ['월', '화', '수', '목', '금', '토', '일'];

/// 실제 적용 스케줄 — 주택·빌라 사용자가 "우리 집 수거 요일" 을 지정했으면
/// 그 요일 = 재활용품(플라스틱·비닐) 배출일, 나머지는 배출 없음.
/// 지정이 없거나 아파트면 동네 기본값(kDefaultWeekSchedule).
List<PickupKind> effectiveWeekSchedule() {
  final days = appPickupWeekdays.value;
  if (appHousingType.value == HousingType.apartment || days.isEmpty) {
    return kDefaultWeekSchedule;
  }
  return [
    for (var i = 1; i <= 7; i++)
      days.contains(i) ? PickupKind.plasticVinyl : PickupKind.none,
  ];
}

/// 배출 시간·방법 안내 (시안 고정 카피).
const String kPickupTimeText = '18:00 ~ 24:00';
const String kPickupPlaceText = '문 앞';

extension PickupKindLabel on PickupKind {
  /// 수거일 안내 카드·일정 리스트용 전체 표기.
  String get fullLabel => switch (this) {
        PickupKind.plasticVinyl => '플라스틱 · 비닐류',
        PickupKind.paperBox => '종이 · 박스류',
        PickupKind.general => '일반쓰레기',
        PickupKind.none => '배출 없는 날',
      };

  /// 홈 헤드라인용 축약 표기.
  String get shortLabel => switch (this) {
        PickupKind.plasticVinyl => '플라스틱·비닐',
        PickupKind.paperBox => '종이·박스',
        PickupKind.general => '일반쓰레기',
        PickupKind.none => '',
      };
}

/// OS 알림 발송이 아직 연동되지 않았음을 알리는 공용 문구.
/// 설정값은 저장되지만 실제 알림은 울리지 않는다(후속: flutter_local_notifications).
const String kReminderPendingNote = '알림 발송은 준비 중 · 설정은 저장돼요';

/// 수거일 알림 — 요일마다 반복되는 로컬 알림 설정값.
/// (OS 알림 스케줄링 연동 전 — 설정 UI 와 영구 저장까지 담당)
class CollectionReminder {
  final int weekday;     // 1(월) ~ 7(일) — DateTime.weekday 규약
  final int hour;        // 0~23
  final int minute;
  final bool dayBefore;  // true = 하루 전 알림
  final bool enabled;

  const CollectionReminder({
    required this.weekday,
    required this.hour,
    required this.minute,
    this.dayBefore = false,
    this.enabled = true,
  });

  CollectionReminder copyWith({
    int? weekday,
    int? hour,
    int? minute,
    bool? dayBefore,
    bool? enabled,
  }) =>
      CollectionReminder(
        weekday: weekday ?? this.weekday,
        hour: hour ?? this.hour,
        minute: minute ?? this.minute,
        dayBefore: dayBefore ?? this.dayBefore,
        enabled: enabled ?? this.enabled,
      );

  /// '오후 5:30' 형태 표기.
  String get timeLabel {
    final isPm = hour >= 12;
    final h12 = hour % 12 == 0 ? 12 : hour % 12;
    final mm = minute.toString().padLeft(2, '0');
    return '${isPm ? '오후' : '오전'} $h12:$mm';
  }

  /// 알림 대상 요일의 배출 품목.
  PickupKind get pickup => effectiveWeekSchedule()[weekday - 1];

  Map<String, dynamic> toMap() => {
        'weekday': weekday,
        'hour': hour,
        'minute': minute,
        'day_before': dayBefore,
        'enabled': enabled,
      };

  factory CollectionReminder.fromMap(Map<String, dynamic> m) =>
      CollectionReminder(
        weekday: m['weekday'] as int,
        hour: m['hour'] as int,
        minute: m['minute'] as int,
        dayBefore: m['day_before'] as bool? ?? false,
        enabled: m['enabled'] as bool? ?? true,
      );
}

/// 수거일 알림 영구 저장 (SharedPreferences JSON).
class ReminderStore {
  Future<List<CollectionReminder>> load() async {
    final raw = await AppScope.settings.getRemindersJson();
    if (raw == null) return [];
    try {
      final list = jsonDecode(raw) as List;
      return list
          .map((e) =>
              CollectionReminder.fromMap((e as Map).cast<String, dynamic>()))
          .toList();
    } catch (_) {
      return [];
    }
  }

  Future<void> save(List<CollectionReminder> reminders) async {
    await AppScope.settings.setRemindersJson(
      jsonEncode([for (final r in reminders) r.toMap()]),
    );
  }
}
