/// 진단 로그 단일 진입점 — 릴리즈 빌드에서는 출력하지 않는다(logcat 노출 방지).
/// `debugPrint`/`print` 를 직접 쓰지 말고 이 함수를 쓴다.
library;

import 'package:flutter/foundation.dart';

void appLog(String message) {
  if (kDebugMode) debugPrint(message);
}
