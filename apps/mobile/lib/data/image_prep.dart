/// 업로드용 이미지 준비 — 가이드 프레임 크롭(선택) + 긴 변 축소 + JPEG 재인코딩.
///
/// 스마트 촬영은 `takePicture()` 풀프레임(기종에 따라 수 MB)을 그대로 올려,
/// 느린 망이나 절전에서 깨어나는 서버에서 타임아웃을 유발했다. 갤러리 경로가
/// 이미 쓰던 크기(긴 변 1600px)로 맞춰 업로드량을 줄인다.
/// 디코딩·인코딩은 백그라운드 isolate 에서 수행(메인 스레드 프리즈 방지).
///
/// 가이드 프레임 크롭(제안 E, W8): 뷰파인더 중앙의 정사각 가이드 틀
/// (짧은 변 × [kGuideFrameFraction]) 안쪽만 잘라 올린다. 학습 데이터는 물체가
/// 프레임을 채우는 bbox 밀착 크롭이라, 세로 전체 프레임(9:16)을 224² 로
/// 찌그러뜨려 넣던 이전 방식보다 분포가 정렬된다. 크롭 좌표는 원본
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
/// 역변환으로 화면의 가이드 틀([guideFrame])에 해당하는 원본 영역을 잘라낸다.
Future<PreparedUpload> prepareForUpload(
  File src, {
  int maxSide = 1600,
  int quality = 88,
  double? guideViewW,
  double? guideViewH,
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

/// 가이드 틀 한 변 = 뷰파인더 짧은 변 × 이 비율 (화면 중앙 정사각).
/// 1.0 이면 세로 화면에서 좌우 끝까지 — 여백이 없어 물체를 틀에 "꽉 차게"
/// 맞추기 어렵고, 너무 작으면 원본 해상도·안정감이 떨어져 0.84 로 둔다.
const double kGuideFrameFraction = 0.84;

/// 뷰파인더(논리 px) 안의 가이드 틀 — UI 브래킷과 업로드 크롭이 같이 쓴다.
({double left, double top, double side}) guideFrame(
  double viewW,
  double viewH,
) {
  final side = math.min(viewW, viewH) * kGuideFrameFraction;
  return (left: (viewW - side) / 2, top: (viewH - side) / 2, side: side);
}

/// 가이드 프레임 크롭 영역 계산 (순수 함수 — 테스트용 공개).
///
/// 프리뷰는 뷰파인더에 BoxFit.cover 로 그려지므로, 화면 좌표 (vx,vy) 는 원본
/// ((imgW − viewW/scale)/2 + vx/scale, …) 에 대응한다. 가이드 틀([guideFrame])을
/// 이 역변환으로 옮긴 정사각형이 크롭 영역.
///
/// 32px 미만으로 작아지면 null.
({int x, int y, int w, int h})? computeGuideCropRect({
  required int imgW,
  required int imgH,
  required double viewW,
  required double viewH,
}) {
  if (imgW <= 0 || imgH <= 0 || viewW <= 0 || viewH <= 0) return null;
  final scale = math.max(viewW / imgW, viewH / imgH);
  final g = guideFrame(viewW, viewH);
  var x = (imgW - viewW / scale) / 2 + g.left / scale;
  var y = (imgH - viewH / scale) / 2 + g.top / scale;
  var w = g.side / scale;
  var h = g.side / scale;
  x = x.clamp(0, imgW.toDouble());
  y = y.clamp(0, imgH.toDouble());
  if (x + w > imgW) w = imgW - x;
  if (y + h > imgH) h = imgH - y;
  if (w < 32 || h < 32) return null;
  return (x: x.round(), y: y.round(), w: w.round(), h: h.round());
}
