import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/api/models.dart';

void main() {
  const rule = RegionRule(
    district: '',
    methodGeneral: '종량제 봉투',
    methodFood: '전용 용기',
    methodRecycle: '투명 봉투',
    daysGeneral: '월·수·금',
    daysFood: '매일',
    daysRecycle: '화·목',
    emitTime: '일몰 후 ~ 24시',
    noCollectDay: '',
  );

  test('재활용 대분류는 재활용 규정, 세부품목도 부모 대분류로 같은 행', () {
    final rows = rule.rowsFor('plastic');
    expect(rows, [
      ('배출 방법', '투명 봉투'),
      ('배출 요일', '화·목'),
      ('배출 시간', '일몰 후 ~ 24시'),
    ]);
  });

  test('음식물은 음식물 규정, 그 외(일반·etc)는 생활쓰레기 규정', () {
    expect(rule.rowsFor('food_waste').first, ('배출 방법', '전용 용기'));
    expect(rule.rowsFor('trash').first, ('배출 방법', '종량제 봉투'));
    expect(rule.rowsFor('etc')[1], ('배출 요일', '월·수·금'));
  });

  test('비어 있는 필드(미수거일 등)는 행에서 빠진다', () {
    expect(rule.rowsFor('trash').any((r) => r.$1 == '미수거일'), isFalse);
    expect(const RegionRule(district: '').rowsFor('paper'), isEmpty);
  });
}
