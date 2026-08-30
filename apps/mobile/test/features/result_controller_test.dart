import 'dart:io';

import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:greenguide/api/api_client.dart';
import 'package:greenguide/api/models.dart';
import 'package:greenguide/data/settings_store.dart';
import 'package:greenguide/features/result/result_controller.dart';
import 'package:greenguide/services/prediction_service.dart';

Prediction _pred(String cls, {String? uploadId}) => Prediction(
  predictedClass: cls,
  predictedIndex: 0,
  confidence: 0.9,
  allProbabilities: {cls: 0.9, 'etc': 0.1},
  modelArch: 'test',
  inferenceMs: 1,
  uploadId: uploadId,
);

class _FakePrediction extends PredictionService {
  _FakePrediction({this.error}) : super(SettingsStore());
  final Object? error;
  int calls = 0;

  @override
  Future<Prediction> predict(
    File image, {
    bool centered = false,
    UploadProgress? onUploadProgress,
  }) async {
    calls++;
    onUploadProgress?.call(10, 10);
    if (error != null) throw error!;
    return _pred('paper', uploadId: 'up-1');
  }
}

class _FakeApi extends WasteApiClient {
  _FakeApi({this.objects}) : super(baseUrl: 'http://fake');
  final PredictObjects? objects;
  final List<Offset?> regionTaps = [];

  @override
  Future<PredictionWithRegions> predictWithRegions(
    File imageFile, {
    double? tapX,
    double? tapY,
  }) async {
    regionTaps.add(tapX == null ? null : Offset(tapX, tapY!));
    return PredictionWithRegions.fromJson({
      'predicted_class': 'paper',
      'predicted_index': 0,
      'confidence': 0.9,
      'all_probabilities': {'paper': 0.9},
      'model_arch': 'test',
      'inference_ms': 1,
      'regions': const [],
    });
  }

  @override
  Future<PredictObjects> predictObjects(File imageFile) async {
    if (objects == null) throw ApiException('no', statusCode: 404);
    return objects!;
  }

  @override
  Future<RegionInfo?> fetchRegionInfo(String sido, String sigungu) async =>
      null;

  @override
  Future<Prediction> predictHier(
    File imageFile, {
    double? tapX,
    double? tapY,
    UploadProgress? onUploadProgress,
  }) async {
    return _pred('plastic');
  }
}

Future<void> _settle() =>
    Future<void>.delayed(const Duration(milliseconds: 50));

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  late Directory dir;
  late File image;

  setUp(() async {
    SharedPreferences.setMockInitialValues({});
    dir = await Directory.systemTemp.createTemp('rc_');
    image = File('${dir.path}/a.png')
      ..writeAsBytesSync(img.encodePng(img.Image(width: 8, height: 8)));
  });
  tearDown(() => dir.delete(recursive: true));

  ResultController make({
    Object? error,
    PredictObjects? objects,
    _FakeApi? api,
  }) => ResultController(
    image: image,
    isSmartCapture: false,
    prediction: _FakePrediction(error: error),
    api: () async => api ?? _FakeApi(objects: objects),
    settings: SettingsStore(),
  );

  test('로더는 분류와 재질 분석이 모두 끝나야 사라진다', () async {
    final c = make();
    expect(c.loading, isTrue);
    await c.classify();
    expect(c.classifyDone, isTrue);
    expect(c.loading, isTrue, reason: '재질 분석 전');
    await c.fetchRegions();
    expect(c.loading, isFalse);
    expect(c.prediction?.predictedClass, 'paper');
    expect(c.uploadSent, 10);
  });

  test('분류 실패 → 한국어 오류, retry 로 복구', () async {
    final c = make(error: ApiException('x', statusCode: 503));
    await c.classify();
    expect(c.loading, isFalse);
    expect(c.error, contains('준비 중'));
    c.retry();
    expect(c.error, isNull);
    expect(c.loading, isTrue);
  });

  test('물건 후보 선택 → 결과 교체 + 업로드 ID 승계 + 되돌리기', () async {
    final objects = PredictObjects.fromJson({
      'objects': [
        {
          'bbox_norm': [0.1, 0.2, 0.3, 0.4],
          'display_class': 'can',
          'display_level': 'coarse',
          'coarse_class': 'can',
          'coarse_confidence': 0.9,
          'fine_class': null,
          'fine_confidence': 0.0,
          'fine_margin': 0.0,
        },
        {
          'bbox_norm': [0.6, 0.5, 0.8, 0.7],
          'display_class': 'glass',
          'display_level': 'coarse',
          'coarse_class': 'glass',
          'coarse_confidence': 0.9,
          'fine_class': null,
          'fine_confidence': 0.0,
          'fine_margin': 0.0,
        },
      ],
    });
    final api = _FakeApi(objects: objects);
    final c = make(api: api);
    c.start();
    await _settle();
    expect(c.objects?.isMultiObject, isTrue);
    expect(c.canUndo, isFalse);

    c.selectObject(1);
    expect(c.prediction?.predictedClass, 'glass');
    expect(c.prediction?.uploadId, 'up-1', reason: '피드백은 원본 사진 ID 로');
    expect(c.selectedObject, 1);
    expect(c.canUndo, isTrue);
    await _settle();
    expect(
      api.regionTaps.last,
      const Offset(0.7, 0.6),
      reason: '빗금 재분석은 선택 물건 기준',
    );

    c.undo();
    expect(c.prediction?.predictedClass, 'paper');
    expect(c.selectedObject, isNull);
    expect(c.canUndo, isFalse);
  });

  test('되돌리기 스택은 최대 10개', () async {
    final objects = PredictObjects.fromJson({
      'objects': [
        for (var i = 0; i < 2; i++)
          {
            'bbox_norm': [0.4, 0.4, 0.6, 0.6],
            'display_class': 'can',
            'display_level': 'coarse',
            'coarse_class': 'can',
            'coarse_confidence': 0.9,
            'fine_class': null,
            'fine_confidence': 0.0,
            'fine_margin': 0.0,
          },
      ],
    });
    final c = make(objects: objects);
    c.start();
    await _settle();
    for (var i = 0; i < 15; i++) {
      c.selectObject(i % 2, haptic: false);
    }
    var undos = 0;
    while (c.canUndo) {
      c.undo();
      undos++;
    }
    expect(undos, ResultController.kMaxUndo);
  });

  test('tapToNorm — cover 역변환, 범위 밖·크기 미해석은 null', () async {
    final c = make();
    expect(
      c.tapToNorm(const Offset(1, 1), const Size(100, 100)),
      isNull,
      reason: '원본 크기 미해석',
    );
    c.resolveImageSize();
    await _settle();
    expect(c.imgSize, const Size(8, 8));
    // 100x100 뷰에 8x8 정사각 → 스케일 12.5, 오프셋 0
    expect(
      c.tapToNorm(const Offset(50, 50), const Size(100, 100)),
      const Offset(0.5, 0.5),
    );
    expect(c.tapToNorm(const Offset(-1, 50), const Size(100, 100)), isNull);
  });

  test('dispose 후 늦게 도착한 응답은 무시된다(notify 안 함)', () async {
    final c = make();
    var notified = 0;
    c.addListener(() => notified++);
    final f = c.classify();
    c.dispose();
    await f;
    expect(notified, 1, reason: '업로드 진행 1회만 (dispose 전)');
  });
}
