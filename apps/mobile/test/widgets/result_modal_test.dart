import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/features/result/result_modal.dart';
import 'package:image/image.dart' as img;

import '../helpers/test_env.dart';

void main() {
  late Directory dir;
  setUp(() async => dir = await setUpTestEnv());
  tearDown(() => dir.delete(recursive: true));

  testWidgets('서버 오류 시 한국어 안내 + 다시 시도 버튼', (tester) async {
    // flutter_test 의 HttpClient 는 모든 요청에 400 을 돌려준다 → ApiException(400)
    final image = File('${dir.path}/shot.png')
      ..writeAsBytesSync(img.encodePng(img.Image(width: 8, height: 8)));

    await tester.pumpWidget(
      wrapApp(
        Builder(
          builder: (context) => Scaffold(
            body: TextButton(
              onPressed: () => showResultModal(context, image),
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

    expect(find.textContaining('다른 각도로 다시 촬영'), findsOneWidget);
    expect(find.text('다시 시도'), findsOneWidget);
    expect(find.textContaining('Exception'), findsNothing);

    // 출처 배지: '갤러리 · 오후 5:34' 형식 — 컨트롤러 객체가 문자열화되면 안 된다
    // (회귀: '$c.capturedAt' → "Instance of 'ResultController'.capturedAt")
    expect(
      find.textContaining(RegExp(r'^갤러리 · (오전|오후) \d{1,2}:\d{2}$')),
      findsOneWidget,
    );
    expect(find.textContaining('Instance of'), findsNothing);
  });
}
