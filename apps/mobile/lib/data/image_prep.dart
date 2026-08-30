/// 업로드용 이미지 준비 — 긴 변 축소 + JPEG 재인코딩.
///
/// 스마트 촬영은 `takePicture()` 풀프레임(기종에 따라 수 MB)을 그대로 올려,
/// 느린 망이나 절전에서 깨어나는 서버에서 타임아웃을 유발했다. 갤러리 경로가
/// 이미 쓰던 크기(긴 변 1600px)로 맞춰 업로드량을 줄인다.
/// 디코딩·인코딩은 백그라운드 isolate 에서 수행(메인 스레드 프리즈 방지).
library;

import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

/// 실패하면 원본 파일을 그대로 돌려준다 — 업로드 자체는 계속 가능해야 한다.
Future<File> prepareForUpload(
  File src, {
  int maxSide = 1600,
  int quality = 88,
}) async {
  try {
    final dir = await getTemporaryDirectory();
    final dst = p.join(
      dir.path,
      'upload_${DateTime.now().millisecondsSinceEpoch}.jpg',
    );
    final out = await compute(_resizeJpeg, <String, dynamic>{
      'src': src.path,
      'dst': dst,
      'maxSide': maxSide,
      'quality': quality,
    });
    return out == null ? src : File(out);
  } catch (_) {
    return src;
  }
}

/// isolate 진입점 — 축소가 필요 없으면 null(원본 사용).
String? _resizeJpeg(Map<String, dynamic> args) {
  final decoded = img.decodeImage(File(args['src'] as String).readAsBytesSync());
  if (decoded == null) return null;

  // EXIF 회전 정보를 픽셀에 반영 — 재인코딩하면 EXIF 가 사라져
  // 눕거나 뒤집힌 사진이 서버로 가는 것을 막는다.
  final baked = img.bakeOrientation(decoded);
  final maxSide = args['maxSide'] as int;
  if (baked.width <= maxSide && baked.height <= maxSide) return null;

  final resized = baked.width >= baked.height
      ? img.copyResize(baked, width: maxSide)
      : img.copyResize(baked, height: maxSide);
  final out = File(args['dst'] as String)
    ..writeAsBytesSync(img.encodeJpg(resized, quality: args['quality'] as int));
  return out.path;
}
