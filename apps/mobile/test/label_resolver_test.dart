import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/data/waste_info.dart';

/// 피드백 "직접 입력" → 서버 라벨(slug) 해석.
/// 서버 /feedback 은 등록된 slug 만 받으므로(그 외 400), 해석 결과가
/// null 이면 앱은 서버 전송을 건너뛰고 기기 기록에만 남긴다.
void main() {
  group('resolveLabelSlug', () {
    test('한글 표시명을 slug 로 해석한다', () {
      expect(resolveLabelSlug('플라스틱'), 'plastic');
      expect(resolveLabelSlug('종이팩'), 'paper_pack');
    });

    test('표기 흔들림(공백·중점·류)을 흡수한다', () {
      expect(resolveLabelSlug(' 유리 '), resolveLabelSlug('유리류'));
      expect(resolveLabelSlug('캔·고철'), resolveLabelSlug('캔 고철'));
    });

    test('slug 를 그대로 입력해도 통과한다', () {
      expect(resolveLabelSlug('pet'), 'pet');
    });

    test('모르는 재질은 null — 서버로 보내지 않는다', () {
      expect(resolveLabelSlug('아이스팩'), isNull);
      expect(resolveLabelSlug(''), isNull);
      expect(resolveLabelSlug('   '), isNull);
    });
  });
}
