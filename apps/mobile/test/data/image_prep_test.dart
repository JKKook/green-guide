import 'dart:io';
import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/data/image_prep.dart';
import 'package:image/image.dart' as img;

import '../helpers/test_env.dart';

void main() {
  group('guideFrame', () {
    test('짧은 변 × kGuideFrameFraction 정사각, 화면 중앙', () {
      final g = guideFrame(100, 200);
      expect(g.side, closeTo(100 * kGuideFrameFraction, 1e-9));
      expect(g.left, closeTo((100 - g.side) / 2, 1e-9));
      expect(g.top, closeTo((200 - g.side) / 2, 1e-9));
    });
  });

  group('computeGuideCropRect', () {
    test('뷰와 같은 비율(세로 9:16) — 중앙 정사각만 남는다', () {
      final r = computeGuideCropRect(
        imgW: 720,
        imgH: 1280,
        viewW: 360,
        viewH: 640,
      )!;
      // scale=0.5 → 한 변 360*0.84/0.5 = 604.8
      expect(r.w, 605);
      expect(r.h, 605);
      expect(r.x, 58); // (720-604.8)/2
      expect(r.y, 338); // (1280-604.8)/2
    });

    test('가로가 넘치는 원본(cover) — 화면 밖 좌우를 제외하고 틀 영역만', () {
      final r = computeGuideCropRect(
        imgW: 2000,
        imgH: 1000,
        viewW: 100,
        viewH: 100,
      )!;
      // scale=0.1, 보이는 영역 x∈[500,1500] → 틀 840, x=500+80
      expect(r, (x: 580, y: 80, w: 840, h: 840));
    });

    test('원본(3:4)이 뷰(9:19.5)보다 넓어도 크롭 비율 ≤1.25', () {
      final r = computeGuideCropRect(
        imgW: 3000,
        imgH: 4000,
        viewW: 360,
        viewH: 780,
      )!;
      final ratio = math.max(r.w, r.h) / math.min(r.w, r.h);
      expect(ratio, lessThanOrEqualTo(1.25));
      // 틀이 원본 안쪽에 들어와야 한다
      expect(r.x, greaterThanOrEqualTo(0));
      expect(r.x + r.w, lessThanOrEqualTo(3000));
      expect(r.y + r.h, lessThanOrEqualTo(4000));
    });

    test('결과가 32px 미만이면 null', () {
      expect(
        computeGuideCropRect(imgW: 30, imgH: 30, viewW: 100, viewH: 100),
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

    testWidgets('가이드 크롭 — crop_box 가 실제 업로드 픽셀과 일치', (tester) async {
      await tester.runAsync(() async {
        // 원본 세로 720×1280(9:16) — 틀 바깥은 검정, 틀 안쪽은 흰색으로 칠한다.
        final rect = computeGuideCropRect(
          imgW: 720,
          imgH: 1280,
          viewW: 360,
          viewH: 640,
        )!;
        final im = img.Image(width: 720, height: 1280);
        img.fillRect(
          im,
          x1: rect.x,
          y1: rect.y,
          x2: rect.x + rect.w - 1,
          y2: rect.y + rect.h - 1,
          color: img.ColorRgb8(255, 255, 255),
        );
        final src = File('${dir.path}/portrait.png')
          ..writeAsBytesSync(img.encodePng(im));
        final prep = await prepareForUpload(
          src,
          guideViewW: 360,
          guideViewH: 640,
        );
        expect(prep.cropApplied, isTrue);
        expect(prep.cropBox, '${rect.x},${rect.y},${rect.w},${rect.h}');
        final box = prep.cropBox!.split(',').map(int.parse).toList();
        final out = img.decodeImage(prep.file.readAsBytesSync())!;
        expect(
          (out.width, out.height),
          (box[2], box[3]),
          reason: '1600 이하라 축소 없음 → 업로드 크기 = crop_box 크기',
        );
        expect(
          math.max(out.width, out.height) / math.min(out.width, out.height),
          lessThanOrEqualTo(1.25),
        );
        // 업로드 이미지 네 모서리가 모두 틀 안쪽(흰색) — 바깥 검정이 섞이지 않음
        for (final (x, y) in [
          (1, 1),
          (out.width - 2, 1),
          (1, out.height - 2),
          (out.width - 2, out.height - 2),
        ]) {
          expect(out.getPixel(x, y).r, greaterThan(200), reason: '($x,$y)');
        }
      });
    });

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
        );
        expect(prep.cropApplied, isTrue);
        // scale=0.05 → 한 변 84/0.05 = 1680, x=(2000-1680)/2, y=(4000-1680)/2
        expect(prep.cropBox, '160,1160,1680,1680');
        expect(prep.orientation, 1, reason: 'PNG — EXIF 없음');
        final out = img.decodeImage(prep.file.readAsBytesSync())!;
        expect((out.width, out.height), (1600, 1600), reason: '긴 변 1600 축소');
      });
    });

    testWidgets('갤러리 경로(뷰 크기 없음) — 크롭 없이 축소만', (tester) async {
      await tester.runAsync(() async {
        final src = File('${dir.path}/gallery.png')
          ..writeAsBytesSync(
            img.encodePng(img.Image(width: 1800, height: 3200)),
          );
        final prep = await prepareForUpload(src);
        expect(prep.cropApplied, isFalse);
        expect(prep.cropBox, isNull);
        final out = img.decodeImage(prep.file.readAsBytesSync())!;
        expect((out.width, out.height), (900, 1600));
      });
    });

    testWidgets('크롭·축소 모두 불필요하면 원본 파일 그대로', (tester) async {
      await tester.runAsync(() async {
        final src = File('${dir.path}/small.png')
          ..writeAsBytesSync(img.encodePng(img.Image(width: 400, height: 300)));
        final prep = await prepareForUpload(src);
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
