/// 햅틱 피드백 — 설정 > 분류 > "햅틱 피드백" 토글을 존중하는 래퍼.
///
/// UI 코드는 `HapticFeedback` 대신 이 클래스를 쓴다. 토글이 꺼져 있으면
/// 진동을 발생시키지 않는다(예전엔 값이 저장만 되고 무시됐음).
library;

import 'package:flutter/services.dart';

import '../core/di/app_scope.dart';


bool _enabled = true;

/// 앱 시작 시 1회 — 저장된 설정값 로드.
Future<void> initHaptics() async {
  _enabled = await AppScope.settings.isHapticsEnabled();
}

/// 설정 화면에서 토글 변경 시 즉시 반영.
void setHapticsEnabled(bool enabled) => _enabled = enabled;

class Haptics {
  const Haptics._();

  static void selection() {
    if (_enabled) HapticFeedback.selectionClick();
  }

  static void light() {
    if (_enabled) HapticFeedback.lightImpact();
  }

  static void medium() {
    if (_enabled) HapticFeedback.mediumImpact();
  }

  static void heavy() {
    if (_enabled) HapticFeedback.heavyImpact();
  }

  static void vibrate() {
    if (_enabled) HapticFeedback.vibrate();
  }
}
