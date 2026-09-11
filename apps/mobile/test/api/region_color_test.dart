import 'package:flutter/painting.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/api/models.dart';

void main() {
  test('color_hex 가 있으면 빗금과 같은 색으로 파싱', () {
    final r = MaterialRegion.fromJson({
      'slug': 'metal',
      'bbox_norm': [0, 0, 1, 1],
      'avg_conf': 0.9,
      'cell_count': 3,
      'color_hex': '#8E9AAF',
    });
    expect(r.color, const Color(0xFF8E9AAF));
  });

  test('구서버(color_hex 없음)·잘못된 형식은 null → 앱 팔레트 폴백', () {
    final none = MaterialRegion.fromJson({
      'slug': 'metal',
      'bbox_norm': [0, 0, 1, 1],
      'avg_conf': 0.9,
      'cell_count': 3,
    });
    expect(none.color, isNull);
    const bad = MaterialRegion(
      slug: 'metal',
      bboxNorm: [0, 0, 1, 1],
      avgConf: 0.9,
      cellCount: 1,
      colorHex: 'red',
    );
    expect(bad.color, isNull);
  });
}
