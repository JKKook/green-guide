import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/api/models.dart';

Map<String, dynamic> _hierJson({String? cam}) => {
  'display_level': 'fine',
  'display_class': 'metal',
  'coarse_class': 'metal',
  'coarse_confidence': 0.95,
  'fine_class': 'metal',
  'fine_confidence': 0.95,
  'fine_margin': 0.9,
  'coarse_probabilities': {'metal': 0.95},
  'fine_top5': <Map<String, dynamic>>[],
  'model_arch': 'test',
  'inference_ms': 1.0,
  'cam_base64': ?cam,
};

void main() {
  test('predict-hier 응답에 cam_base64 가 있으면 CAM 사용 가능', () {
    final r = PredictionWithCam.fromHierJson(
      _hierJson(cam: 'data:image/png;base64,AAAA'),
    );
    expect(r.camAvailable, isTrue);
    expect(r.camBase64, startsWith('data:image/png'));
    expect(r.prediction.predictedClass, 'metal');
  });

  test('구버전 서버(cam_base64 없음)는 CAM 미지원으로 판정', () {
    final r = PredictionWithCam.fromHierJson(_hierJson());
    expect(r.camAvailable, isFalse);
    expect(r.camBase64, isNull);
  });
}
