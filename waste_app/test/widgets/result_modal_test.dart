import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:waste_app/widgets/result_modal.dart';

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
  });
}
