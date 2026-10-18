import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/features/shell/main_shell.dart';

import '../helpers/test_env.dart';

void main() {
  late Directory dir;
  setUp(
    () async => dir = await setUpTestEnv(
      prefs: {'onboarding_done': true, 'region_prompt_shown': true},
    ),
  );
  tearDown(() => dir.delete(recursive: true));

  testWidgets('하단 탭 4개 + 중앙 촬영 버튼, 기록 탭 전환 시 빈 상태', (tester) async {
    await tester.pumpWidget(wrapApp(const MainShell()));
    await settleIo(tester);
    await tester.pump(const Duration(seconds: 1));

    for (final label in ['홈', '검색', '기록', '설정']) {
      expect(find.text(label), findsWidgets, reason: '탭 라벨 $label');
    }
    expect(find.bySemanticsLabel('사진으로 분리배출 확인하기'), findsOneWidget);

    await tester.tap(find.text('기록').last);
    await settleIo(tester);
    await tester.pump(const Duration(seconds: 1));
    expect(find.text('아직 분류 기록이 없어요'), findsOneWidget);
  });

  testWidgets('키보드가 올라오면 중앙 촬영 버튼을 숨긴다 (콘텐츠 가림 방지)', (tester) async {
    await tester.pumpWidget(wrapApp(const MainShell()));
    await settleIo(tester);
    await tester.pump(const Duration(seconds: 1));
    expect(find.bySemanticsLabel('사진으로 분리배출 확인하기'), findsOneWidget);

    // Scaffold 는 FAB 를 퇴장 애니메이션 뒤에 제거하므로 한 프레임 더 흘린다.
    tester.view.viewInsets = const FakeViewPadding(bottom: 300);
    addTearDown(tester.view.reset);
    await tester.pump();
    await tester.pump(const Duration(seconds: 1));
    expect(find.bySemanticsLabel('사진으로 분리배출 확인하기'), findsNothing);

    tester.view.viewInsets = FakeViewPadding.zero;
    await tester.pump();
    await tester.pump(const Duration(seconds: 1));
    expect(find.bySemanticsLabel('사진으로 분리배출 확인하기'), findsOneWidget);
  });
}
