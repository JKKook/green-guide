/// 결과 시트는 느린 드래그로도 스크롤돼야 한다.
///
/// 회귀: DraggableScrollableSheet(min=max=1.0) 는 스크롤 오프셋 0 에서 위로 드래그하면
/// `isAtMin && delta < 0` 분기로 시트 크기 조절(무효)에 삼켜 리스트가 안 움직였다 —
/// 플링(빠른 스와이프)만 먹혀 에뮬레이터 QA 에서 발견(2026-10-05).
library;

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/features/result/result_modal.dart';
import 'package:image/image.dart' as img;

import '../helpers/result_fakes.dart';
import '../helpers/test_env.dart';

void main() {
  late Directory dir;
  setUp(() async => dir = await setUpTestEnv());
  tearDown(() => dir.delete(recursive: true));

  testWidgets('로드된 결과 시트가 느린 드래그로 스크롤된다', (tester) async {
    tester.view.physicalSize = const Size(360, 640);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);
    final image = File('${dir.path}/shot.png')
      ..writeAsBytesSync(img.encodePng(img.Image(width: 64, height: 48)));
    await tester.pumpWidget(
      wrapApp(
        Builder(
          builder: (context) => Scaffold(
            body: TextButton(
              onPressed: () => showResultModal(
                context,
                image,
                isSmartCapture: true,
                prediction: FakePredictionService(markPriorityJson()),
                api: () async => FakeApi(),
              ),
              child: const Text('open'),
            ),
          ),
        ),
      ),
    );
    await tester.tap(find.text('open'));
    await tester.pump();
    await settleIo(tester, const Duration(seconds: 1));
    await tester.pump(const Duration(seconds: 2));
    expect(find.text('왜 이렇게 분류했어?'), findsOneWidget);

    final list = find.byType(ListView).last;
    final before = tester.state<ScrollableState>(
      find.descendant(of: list, matching: find.byType(Scrollable)),
    ).position.pixels;
    // 느린 드래그(플링 아님) — 손가락으로 끌어올리는 일반 동작
    await tester.drag(list, const Offset(0, -200));
    await tester.pump(const Duration(milliseconds: 300));
    final after = tester.state<ScrollableState>(
      find.descendant(of: list, matching: find.byType(Scrollable)),
    ).position.pixels;
    expect(after, greaterThan(before + 100),
        reason: '느린 드래그가 시트 크기 조절에 삼켜져 리스트가 스크롤되지 않음');
  });
}
