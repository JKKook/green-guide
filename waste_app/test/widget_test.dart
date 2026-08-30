import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:waste_app/screens/onboarding_screen.dart';
import 'package:waste_app/theme/app_theme.dart';

void main() {
  testWidgets('OnboardingScreen 첫 화면 — 브랜드 소개 + 동의 시트', (tester) async {
    await tester.pumpWidget(
      MaterialApp(theme: buildLightTheme(), home: const OnboardingScreen()),
    );
    await tester.pumpAndSettle();

    expect(find.text('사진 한 장으로 끝내는 분리배출'), findsOneWidget);
    expect(find.text('그린가이드 이용 동의'), findsOneWidget);
  });
}
