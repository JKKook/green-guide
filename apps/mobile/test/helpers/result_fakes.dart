/// 결과 모달 위젯 테스트용 가짜 — 네트워크 없이 로드 상태를 그린다.
library;

import 'dart:io';

import 'package:greenguide/api/api_client.dart';
import 'package:greenguide/api/models.dart';
import 'package:greenguide/data/settings_store.dart';
import 'package:greenguide/services/prediction_service.dart';

/// 스마트촬영 + 분리배출 표시 최우선 판정 응답 (서버 /predict-hier 형태).
Map<String, dynamic> markPriorityJson() => {
  'display_level': 'fine',
  'display_class': 'plastic_other',
  'coarse_class': 'plastic',
  'coarse_confidence': 0.93,
  'fine_class': 'plastic_other',
  'fine_confidence': 0.91,
  'fine_margin': 0.6,
  'coarse_probabilities': {'plastic': 0.93, 'metal': 0.04, 'etc': 0.03},
  'model_arch': 'test | mark:other',
  'inference_ms': 12,
  'upload_id': 'up-test',
  'evidence': [
    {'type': 'mark', 'token': 'other', 'matched_text': 'OTHER',
     'mapped_class': 'plastic_other', 'score': 0.92, 'primary': true},
    {'type': 'mark', 'token': 'pp', 'matched_text': '캡:PP',
     'mapped_class': 'plastic_other', 'score': 0.88, 'primary': false},
    {'type': 'text', 'token': '샴푸', 'matched_text': '샴푸',
     'mapped_class': 'plastic', 'score': 0.7, 'primary': false},
  ],
};

class FakePredictionService extends PredictionService {
  FakePredictionService(this.json) : super(SettingsStore());
  final Map<String, dynamic> json;

  @override
  Future<Prediction> predict(
    File image, {
    bool centered = false,
    UploadMeta? meta,
    UploadProgress? onUploadProgress,
  }) async {
    onUploadProgress?.call(10, 10);
    return Prediction.fromHierJson(json);
  }
}

class FakeApi extends WasteApiClient {
  FakeApi() : super(baseUrl: 'http://fake');

  @override
  Future<PredictionWithRegions> predictWithRegions(
    File imageFile, {
    double? tapX,
    double? tapY,
  }) async =>
      PredictionWithRegions.fromJson({
        'predicted_class': 'plastic',
        'predicted_index': 0,
        'confidence': 0.9,
        'all_probabilities': {'plastic': 0.9},
        'model_arch': 'test',
        'inference_ms': 1,
        'regions': const [],
      });

  @override
  Future<PredictObjects> predictObjects(File imageFile) async =>
      throw ApiException('no', statusCode: 404);

  @override
  Future<RegionInfo?> fetchRegionInfo(String sido, String sigungu) async =>
      null;
}
