import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/data/history_repository.dart';
import 'package:greenguide/features/history/history_screen.dart';

import '../helpers/test_env.dart';

void main() {
  // HistoryRepository 가 DB 핸들을 static 으로 캐시하므로 파일 단위로 디렉토리를 공유한다.
  late Directory dir;
  setUpAll(() async => dir = await setUpTestEnv());
  tearDownAll(() => dir.delete(recursive: true));

  testWidgets('기록 없음 → 빈 상태', (tester) async {
    await tester.pumpWidget(wrapApp(const HistoryScreen()));
    await settleIo(tester);
    expect(find.text('아직 분류 기록이 없어요'), findsOneWidget);
  });

  testWidgets('저장 후 historyRevision 갱신으로 목록에 나타난다', (tester) async {
    await tester.pumpWidget(wrapApp(const HistoryScreen()));
    await settleIo(tester);

    final src = File('${dir.path}/src.jpg')
      ..writeAsBytesSync([0xFF, 0xD8, 0xFF]);
    await tester.runAsync(
      () => HistoryRepository().save(
        sourceImage: src,
        predictedClass: 'paper',
        confidence: 0.91,
        uploadId: null,
        modelArch: 'test',
      ),
    );
    await settleIo(tester);

    expect(find.text('아직 분류 기록이 없어요'), findsNothing);
  });
}
