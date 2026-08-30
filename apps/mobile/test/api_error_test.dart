import 'dart:async';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/api/api_client.dart';

/// 사용자에게 원시 예외 문자열이 노출되지 않아야 한다.
void main() {
  group('friendlyError', () {
    test('타임아웃은 절전 서버 안내로 바뀐다', () {
      final msg = friendlyError(
        TimeoutException('x', const Duration(seconds: 30)),
      );
      expect(msg, contains('절전'));
      expect(msg, isNot(contains('TimeoutException')));
    });

    test('네트워크 오류는 연결 확인 안내', () {
      expect(friendlyError(const SocketException('failed')), contains('인터넷'));
    });

    test('상태코드별 안내를 구분한다', () {
      expect(friendlyError(ApiException('x', statusCode: 413)), contains('용량'));
      expect(
        friendlyError(ApiException('x', statusCode: 503)),
        contains('준비 중'),
      );
    });

    test('알 수 없는 예외도 한국어 안내로 감싼다', () {
      final msg = friendlyError(StateError('boom'));
      expect(msg, contains('다시 시도'));
      expect(msg, isNot(contains('boom')));
    });
  });
}
