/// 업로드용 이미지 준비 — 가이드 프레임 크롭(선택) + 긴 변 축소 + JPEG 재인코딩.
///
/// 스마트 촬영은 `takePicture()` 풀프레임(기종에 따라 수 MB)을 그대로 올려,
/// 느린 망이나 절전에서 깨어나는 서버에서 타임아웃을 유발했다. 갤러리 경로가
/// 이미 쓰던 크기(긴 변 1600px)로 맞춰 업로드량을 줄인다.
/// 디코딩·인코딩은 백그라운드 isolate 에서 수행(메인 스레드 프리즈 방지).
///
/// 가이드 프레임 크롭(제안 E): 뷰파인더의 코너 브래킷 안쪽만 잘라 올리면
/// 학습 데이터(bbox 밀착 크롭)와 분포가 정렬된다. 크롭 좌표는 원본
/// (EXIF 회전 반영 후) 픽셀 기준으로 함께 보고한다.
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

/// 업로드 준비 결과 — 파일 + 서버 보고용 메타.
class PreparedUpload {
  final File file;

  /// 원본 EXIF Orientation 태그 (1/3/6/8, 태그 없으면 1).
  /// 재인코딩하면 태그가 사라지므로 굽기 전에 읽어 폼 필드로 전달한다.
  final int orientation;

  final bool cropApplied;

  /// 크롭 영역 — 원본(회전 반영) 픽셀 좌표 "x,y,w,h" (정수). 미적용이면 null.
  final String? cropBox;

  const PreparedUpload({
    required this.file,
    this.orientation = 1,
    this.cropApplied = false,
    this.cropBox,
  });
}

/// 실패하면 원본 파일을 그대로 돌려준다 — 업로드 자체는 계속 가능해야 한다.
///
/// [guideViewW]/[guideViewH]: 뷰파인더 표시 크기(논리 px). 주어지면 BoxFit.cover
/// 역변환으로 화면에 보이던 영역에서 [guideInset](코너 브래킷 패딩)만큼 안쪽을
/// 잘라낸다.
Future<PreparedUpload> prepareForUpload(
  File src, {
  int maxSide = 1600,
  int quality = 88,
  double? guideViewW,
  double? guideViewH,
  double guideInset = 0,
}) async {
  try {
    final dir = await getTemporaryDirectory();
    final dst = p.join(
      dir.path,
      'upload_${DateTime.now().millisecondsSinceEpoch}.jpg',
    );
    final out = await compute(_prepare, <String, dynamic>{
      'src': src.path,
      'dst': dst,
      'maxSide': maxSide,
      'quality': quality,
      'viewW': guideViewW,
      'viewH': guideViewH,
      'inset': guideInset,
    });
    return PreparedUpload(
      file: out['path'] == null ? src : File(out['path'] as String),
      orientation: out['orientation'] as int? ?? 1,
      cropApplied: out['cropBox'] != null,
      cropBox: out['cropBox'] as String?,
    );
  } catch (_) {
    return PreparedUpload(file: src);
  }
}

/// isolate 진입점 — 크롭·축소가 모두 불필요하면 path=null(원본 사용).
Map<String, dynamic> _prepare(Map<String, dynamic> args) {
  final decoded = img.decodeImage(
    File(args['src'] as String).readAsBytesSync(),
  );
  if (decoded == null) return const {'path': null, 'orientation': 1};

  final orientation = decoded.exif.imageIfd['Orientation']?.toInt() ?? 1;

  // EXIF 회전 정보를 픽셀에 반영 — 재인코딩하면 EXIF 가 사라져
  // 눕거나 뒤집힌 사진이 서버로 가는 것을 막는다.
  var baked = img.bakeOrientation(decoded);

  String? cropBox;
  final viewW = args['viewW'] as double?;
  final viewH = args['viewH'] as double?;
  if (viewW != null && viewH != null) {
    final rect = computeGuideCropRect(
      imgW: baked.width,
      imgH: baked.height,
      viewW: viewW,
      viewH: viewH,
      inset: args['inset'] as double,
    );
    if (rect != null) {
      baked = img.copyCrop(
        baked,
        x: rect.x,
        y: rect.y,
        width: rect.w,
        height: rect.h,
      );
      cropBox = '${rect.x},${rect.y},${rect.w},${rect.h}';
    }
  }

  final maxSide = args['maxSide'] as int;
  final needResize = baked.width > maxSide || baked.height > maxSide;
  if (cropBox == null && !needResize) {
    return {'path': null, 'orientation': orientation};
  }

  final resized = !needResize
      ? baked
      : (baked.width >= baked.height
            ? img.copyResize(baked, width: maxSide)
            : img.copyResize(baked, height: maxSide));
  final out = File(args['dst'] as String)
    ..writeAsBytesSync(img.encodeJpg(resized, quality: args['quality'] as int));
  return {'path': out.path, 'orientation': orientation, 'cropBox': cropBox};
}

/// 가이드 프레임 크롭 영역 계산 (순수 함수 — 테스트용 공개).
///
/// 프리뷰는 뷰파인더에 BoxFit.cover 로 그려지므로, 화면에 보이는 원본 영역은
/// (viewW,viewH)/scale 크기의 중앙 사각형이다. 거기서 [inset]/scale 만큼
/// 안쪽(코너 브래킷 내부)이 가이드 영역.
///
/// 크롭이 사실상 원본 전체(오차 2px)거나 32px 미만으로 작아지면 null.
({int x, int y, int w, int h})? computeGuideCropRect({
  required int imgW,
  required int imgH,
  required double viewW,
  required double viewH,
  double inset = 0,
}) {
  if (imgW <= 0 || imgH <= 0 || viewW <= 0 || viewH <= 0) return null;
  final scale = math.max(viewW / imgW, viewH / imgH);
  final visW = viewW / scale;
  final visH = viewH / scale;
  final insetImg = inset / scale;
  var x = (imgW - visW) / 2 + insetImg;
  var y = (imgH - visH) / 2 + insetImg;
  var w = visW - 2 * insetImg;
  var h = visH - 2 * insetImg;
  x = x.clamp(0, imgW.toDouble());
  y = y.clamp(0, imgH.toDouble());
  if (x + w > imgW) w = imgW - x;
  if (y + h > imgH) h = imgH - y;
  if (w < 32 || h < 32) return null;
  if (x < 2 && y < 2 && w > imgW - 4 && h > imgH - 4) return null;
  return (x: x.round(), y: y.round(), w: w.round(), h: h.round());
}
