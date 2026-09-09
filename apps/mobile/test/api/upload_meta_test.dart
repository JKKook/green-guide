import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/api/models.dart';

void main() {
  test('UploadMeta.toFields — 서버 합의 필드명·형식', () {
    const meta = UploadMeta(
      captureMode: 'smart',
      orientation: 6,
      qualityBlur: 123.456,
      qualityBrightness: 87.1,
      cropApplied: true,
      cropBox: '100,200,800,600',
    );
    expect(meta.toFields(), {
      'capture_mode': 'smart',
      'orientation': '6',
      'quality_blur': '123.46',
      'quality_brightness': '87.10',
      'crop_applied': 'true',
      'crop_box': '100,200,800,600',
    });
  });

  test('크롭 미적용이면 crop_* 필드를 보내지 않는다', () {
    const meta = UploadMeta(captureMode: 'gallery');
    final f = meta.toFields();
    expect(f, {'capture_mode': 'gallery', 'orientation': '1'});
    expect(f.containsKey('crop_applied'), isFalse);
  });
}
