import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/data/waste_info.dart';

void main() {
  test('레지스트리에 없는 세부품목은 부모 대분류에서 한글화해 합성한다', () {
    // /labels 미로드(오프라인 첫 실행)·영역 분석 fine slug — 영문 slug 가 그대로 보이면 안 된다
    final fine = infoFor('paper_other');
    expect(fine, isNotNull);
    expect(fine!.displayName, '기타 종이');
    expect(fine.level, 2);
    expect(fine.parentSlug, 'paper');
    expect(fine.howTo, isNotEmpty); // 부모(종이류) 안내 상속
    expect(infoFor('glass_deposit')?.displayName, '보증금 반환병');
  });

  test('알 수 없는 slug 는 여전히 null', () {
    expect(infoFor('not_a_class'), isNull);
  });
}
