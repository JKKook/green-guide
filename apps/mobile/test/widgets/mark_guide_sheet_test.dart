import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/data/settings_store.dart';
import 'package:greenguide/features/capture/mark_guide_sheet.dart';
import 'package:shared_preferences/shared_preferences.dart';

Widget _host(SettingsStore store) => MaterialApp(
  home: Builder(
    builder: (ctx) => Scaffold(
      body: TextButton(
        onPressed: () => showMarkGuideOnce(ctx, store: store),
        child: const Text('open'),
      ),
    ),
  ),
);

void main() {
  testWidgets('첫 스마트 촬영에서 한 번만 안내를 띄운다', (tester) async {
    SharedPreferences.setMockInitialValues({});
    final store = SettingsStore();
    await tester.pumpWidget(_host(store));

    await tester.tap(find.text('open'));
    await tester.pumpAndSettle();
    expect(find.text('분리배출 표시가 보이게 찍어 주세요'), findsOneWidget);

    await tester.tap(find.text('알겠어요, 촬영할게요'));
    await tester.pumpAndSettle();
    expect(find.text('분리배출 표시가 보이게 찍어 주세요'), findsNothing);
    expect(await store.isMarkGuideShown(), isTrue);

    await tester.tap(find.text('open'));
    await tester.pumpAndSettle();
    expect(find.text('분리배출 표시가 보이게 찍어 주세요'), findsNothing);
  });
}
