/// 골든 테스트 — 리팩토링(공통 위젯화·토큰 정규화) 전후 픽셀 회귀 감지용.
/// 갱신: `flutter test --update-goldens test/widgets/golden_test.dart`
/// (폰트는 테스트 기본 Ahem 이라 실제 모양과 다르지만 레이아웃 회귀는 잡힌다)
library;

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:waste_app/screens/history_screen.dart';
import 'package:waste_app/screens/settings_screen.dart';
import 'package:waste_app/theme/app_theme.dart';

import '../helpers/test_env.dart';

void main() {
  late Directory dir;
  setUpAll(
    () async => dir = await setUpTestEnv(prefs: {'onboarding_done': true}),
  );
  tearDownAll(() => dir.delete(recursive: true));

  Future<void> pumpGolden(WidgetTester tester, Widget home, String name) async {
    tester.view.physicalSize = const Size(390, 844);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);
    for (final (mode, theme) in [
      ('light', buildLightTheme()),
      ('dark', buildDarkTheme()),
    ]) {
      await tester.pumpWidget(
        MaterialApp(theme: theme, locale: const Locale('ko'), home: home),
      );
      await settleIo(tester);
      await tester.pump(const Duration(seconds: 1));
      await expectLater(
        find.byType(MaterialApp),
        matchesGoldenFile('goldens/${name}_$mode.png'),
      );
    }
  }

  testWidgets('HistoryScreen 빈 상태', (tester) async {
    await pumpGolden(tester, const HistoryScreen(), 'history_empty');
  });

  testWidgets('SettingsScreen', (tester) async {
    await pumpGolden(tester, const SettingsScreen(), 'settings');
  });
}
