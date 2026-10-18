import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/core/ui/keep_all.dart';

void main() {
  const wj = '⁠';

  test('어절 안 글자 사이에만 단어 결합자를 넣고 공백은 그대로 둔다', () {
    expect(keepAll('검색할 수 있어요'),
        '검${wj}색${wj}할 수 있${wj}어${wj}요');
  });

  test('줄바꿈(\\n)·복수 공백은 보존한다 — 강제 줄바꿈이 깨지지 않게', () {
    expect(keepAll('가나\n다라'), '가${wj}나\n다${wj}라');
    expect(keepAll('가  나'), '가  나');
  });

  test('빈 문자열·한 글자는 그대로', () {
    expect(keepAll(''), '');
    expect(keepAll('가'), '가');
  });
}
