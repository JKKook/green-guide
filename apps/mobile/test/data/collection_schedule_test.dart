import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/data/collection_schedule.dart';
import 'package:greenguide/data/settings_store.dart';

void main() {
  tearDown(() {
    appHousingType.value = null;
    appPickupWeekdays.value = const [];
  });

  group('effectiveWeekSchedule', () {
    test('아파트는 동네 기본값', () {
      appHousingType.value = HousingType.apartment;
      appPickupWeekdays.value = const [2, 5];
      expect(effectiveWeekSchedule(), kDefaultWeekSchedule);
    });

    test('주택인데 요일 미지정이면 기본값', () {
      appHousingType.value = HousingType.house;
      expect(effectiveWeekSchedule(), kDefaultWeekSchedule);
    });

    test('주택 + 요일 지정이면 그 요일만 플라스틱·비닐', () {
      appHousingType.value = HousingType.house;
      appPickupWeekdays.value = const [1, 4]; // 월, 목
      final s = effectiveWeekSchedule();
      expect(s.length, 7);
      expect(s[0], PickupKind.plasticVinyl);
      expect(s[3], PickupKind.plasticVinyl);
      expect(s.where((k) => k == PickupKind.none).length, 5);
    });
  });

  group('CollectionReminder', () {
    test('timeLabel 은 12시간제 한글 표기', () {
      expect(
        const CollectionReminder(weekday: 1, hour: 0, minute: 5).timeLabel,
        '오전 12:05',
      );
      expect(
        const CollectionReminder(weekday: 1, hour: 12, minute: 0).timeLabel,
        '오후 12:00',
      );
      expect(
        const CollectionReminder(weekday: 1, hour: 17, minute: 30).timeLabel,
        '오후 5:30',
      );
    });

    test('toMap / fromMap 왕복', () {
      const r = CollectionReminder(
        weekday: 3,
        hour: 9,
        minute: 15,
        dayBefore: true,
        enabled: false,
      );
      final back = CollectionReminder.fromMap(r.toMap());
      expect(back.weekday, 3);
      expect(back.hour, 9);
      expect(back.minute, 15);
      expect(back.dayBefore, isTrue);
      expect(back.enabled, isFalse);
    });

    test('fromMap 은 선택 필드 기본값을 채운다', () {
      final r = CollectionReminder.fromMap({
        'weekday': 7,
        'hour': 8,
        'minute': 0,
      });
      expect(r.dayBefore, isFalse);
      expect(r.enabled, isTrue);
    });

    test('pickup 은 해당 요일의 배출 품목', () {
      appHousingType.value = HousingType.apartment;
      expect(
        const CollectionReminder(weekday: 7, hour: 8, minute: 0).pickup,
        PickupKind.general,
      );
    });
  });

  test('PickupKind 라벨', () {
    expect(PickupKind.plasticVinyl.fullLabel, '플라스틱 · 비닐류');
    expect(PickupKind.none.shortLabel, '');
  });
}
