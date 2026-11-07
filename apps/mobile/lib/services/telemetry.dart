/// Firebase Crashlytics + Analytics 배선 — docs/plans/PRODUCTION_READINESS.md 결정 12.
///
/// `android/app/google-services.json` 은 사용자가 Firebase 프로젝트를 만들어 전달한다.
/// 파일이 없으면 Gradle 이 Firebase 플러그인을 적용하지 않고(빌드 유지), 여기서
/// 초기화가 실패해 앱은 텔레메트리 없이 그대로 동작한다.
library;


import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'package:flutter/foundation.dart';

import '../core/log.dart';

class Telemetry {
  Telemetry._();

  static bool _enabled = false;

  /// Firebase 초기화 성공 여부 — false 면 크래시·분석 전송 없음.
  static bool get enabled => _enabled;

  static Future<void> init() async {
    try {
      await Firebase.initializeApp();
    } catch (e) {
      appLog('[telemetry] Firebase 미구성 — 크래시·분석 수집 비활성: $e');
      return;
    }
    _enabled = true;
    // 개발 중 크래시는 대시보드에 섞지 않는다.
    await FirebaseCrashlytics.instance.setCrashlyticsCollectionEnabled(
      !kDebugMode,
    );
    FlutterError.onError = FirebaseCrashlytics.instance.recordFlutterFatalError;
    PlatformDispatcher.instance.onError = (error, stack) {
      FirebaseCrashlytics.instance.recordError(error, stack, fatal: true);
      return true;
    };
  }
}
