import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/data/image_quality.dart';
import 'package:image/image.dart' as img;

Future<File> _write(Directory dir, String name, img.Image image) async {
  final f = File('${dir.path}/$name');
  await f.writeAsBytes(img.encodePng(image));
  return f;
}

void main() {
  late Directory dir;
  setUp(() async => dir = await Directory.systemTemp.createTemp('iq_'));
  tearDown(() => dir.delete(recursive: true));

  test('밝고 대비가 강한(체커보드) 이미지는 이슈 없음', () async {
    final im = img.Image(width: 64, height: 64);
    for (var y = 0; y < 64; y++) {
      for (var x = 0; x < 64; x++) {
        final v = (x + y).isEven ? 255 : 0;
        im.setPixelRgb(x, y, v, v, v);
      }
    }
    final r = await assessImageQuality(await _write(dir, 'sharp.png', im));
    expect(r.hasIssue, isFalse);
    expect(r.brightness, closeTo(127.5, 5));
  });

  test('어둡고 평탄한 이미지는 tooDark + tooBlurry', () async {
    final im = img.Image(width: 64, height: 64)
      ..clear(img.ColorRgb8(10, 10, 10));
    final r = await assessImageQuality(await _write(dir, 'dark.png', im));
    expect(
      r.issues,
      containsAll([ImageQualityIssue.tooDark, ImageQualityIssue.tooBlurry]),
    );
  });

  test('밝지만 평탄한 이미지는 tooBlurry 만', () async {
    final im = img.Image(width: 64, height: 64)
      ..clear(img.ColorRgb8(200, 200, 200));
    final r = await assessImageQuality(await _write(dir, 'flat.png', im));
    expect(r.issues, [ImageQualityIssue.tooBlurry]);
  });

  test('디코딩 불가 파일은 품질 OK 로 간주(분류를 막지 않음)', () async {
    final f = File('${dir.path}/bad.png')..writeAsBytesSync([1, 2, 3]);
    final r = await assessImageQuality(f);
    expect(r.hasIssue, isFalse);
  });
}
