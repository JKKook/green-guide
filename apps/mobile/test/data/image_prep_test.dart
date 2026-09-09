import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/data/image_prep.dart';
import 'package:image/image.dart' as img;

import '../helpers/test_env.dart';

void main() {
  group('computeGuideCropRect', () {
    test('cover 역변환 — 뷰와 같은 비율이면 inset 만큼만 잘린다', () {
      final r = computeGuideCropRect(
        imgW: 1000,
        imgH: 2000,
        viewW: 100,
        viewH: 200,
        inset: 10,
      );
      // scale=0.1 → inset 10px 는 원본 100px
      expect(r, (x: 100, y: 100, w: 800, h: 1800));
    });

    test('가로가 넘치는 원본(cover) — 좌우가 화면 밖, 보이는 영역만 남는다', () {
      final r = computeGuideCropRect(
        imgW: 2000,
        imgH: 1000,
        viewW: 100,
        viewH: 100,
      );
      // scale=0.1, 보이는 폭 1000 → x=(2000-1000)/2
      expect(r, (x: 500, y: 0, w: 1000, h: 1000));
    });

    test('inset 0 + 비율 동일(전체 화면) → null (크롭 불필요)', () {
      expect(
        computeGuideCropRect(
          imgW: 1600,
          imgH: 1200,
          viewW: 400,
          viewH: 300,
        ),
        isNull,
      );
    });

    test('결과가 32px 미만이면 null (과도한 inset 방어)', () {
      expect(
        computeGuideCropRect(
          imgW: 100,
          imgH: 100,
          viewW: 100,
          viewH: 100,
          inset: 49,
        ),
        isNull,
      );
    });

    test('잘못된 입력은 null', () {
      expect(
        computeGuideCropRect(imgW: 0, imgH: 100, viewW: 10, viewH: 10),
        isNull,
      );
    });
  });

  group('prepareForUpload', () {
    late Directory dir;
    setUpAll(() async => dir = await setUpTestEnv());
    tearDownAll(() => dir.delete(recursive: true));

    testWidgets('가이드 크롭 + 축소 — crop_box 는 원본 좌표', (tester) async {
      await tester.runAsync(() async {
        final src = File('${dir.path}/big.png')
          ..writeAsBytesSync(
            img.encodePng(img.Image(width: 2000, height: 4000)),
          );
        final prep = await prepareForUpload(
          src,
          guideViewW: 100,
          guideViewH: 200,
          guideInset: 10,
        );
        expect(prep.cropApplied, isTrue);
        // scale=0.05 → inset 10(논리px) = 원본 200px
        expect(prep.cropBox, '200,200,1600,3600');
        expect(prep.orientation, 1, reason: 'PNG — EXIF 없음');
        final out = img.decodeImage(prep.file.readAsBytesSync())!;
        expect(out.height, 1600, reason: '긴 변 1600 축소');
      });
    });

    testWidgets('크롭·축소 모두 불필요하면 원본 파일 그대로', (tester) async {
      await tester.runAsync(() async {
        final src = File('${dir.path}/small.png')
          ..writeAsBytesSync(img.encodePng(img.Image(width: 400, height: 300)));
        final prep = await prepareForUpload(
          src,
          guideViewW: 400,
          guideViewH: 300,
        );
        expect(prep.cropApplied, isFalse);
        expect(prep.file.path, src.path);
      });
    });

    testWidgets('디코딩 불가 파일 — 원본 그대로, 업로드는 계속', (tester) async {
      await tester.runAsync(() async {
        final src = File('${dir.path}/bad.jpg')..writeAsBytesSync([1, 2, 3]);
        final prep = await prepareForUpload(src);
        expect(prep.file.path, src.path);
        expect(prep.cropApplied, isFalse);
      });
    });
  });
}
