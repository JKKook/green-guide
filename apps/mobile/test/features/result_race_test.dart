// 결과 화면 응답 경합 — docs/plans/PRODUCTION_READINESS.md 결정 6(하이브리드).
// 첫 로드는 세 응답을 모아 한 번에 판정, 사용자 조작마다 세대 번호로 늦은 응답을 폐기.
// 응답 도착 순서를 Completer 로 직접 제어해 경합을 재현한다.
import 'dart:async';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:greenguide/api/api_client.dart';
import 'package:greenguide/api/models.dart';
import 'package:greenguide/data/settings_store.dart';
import 'package:greenguide/features/result/result_controller.dart';
import 'package:greenguide/services/prediction_service.dart';
import 'package:image/image.dart' as img;
import 'package:shared_preferences/shared_preferences.dart';

Prediction _pred(String cls, {String? uploadId}) => Prediction(
  predictedClass: cls,
  predictedIndex: 0,
  confidence: 0.9,
  allProbabilities: {cls: 0.9},
  modelArch: 'test',
  inferenceMs: 1,
  uploadId: uploadId,
);

PredictionWithRegions _regions(String tag) => PredictionWithRegions.fromJson({
  'predicted_class': tag,
  'predicted_index': 0,
  'confidence': 0.9,
  'all_probabilities': {tag: 0.9},
  'model_arch': 'test',
  'inference_ms': 1,
  'overlay_base64': 'data:image/jpeg;base64,AAAA',
  'regions': [
    {
      'slug': tag,
      'bbox_norm': [0, 0, 1, 1],
      'avg_conf': 0.9,
      'cell_count': 2,
      'color_hex': '#112233',
    },
  ],
});

Map<String, dynamic> _obj(String cls, double cx) => {
  'bbox_norm': [cx - 0.1, 0.4, cx + 0.1, 0.6],
  'display_class': cls,
  'display_level': 'coarse',
  'coarse_class': cls,
  'coarse_confidence': 0.9,
  'fine_class': null,
  'fine_confidence': 0.0,
  'fine_margin': 0.0,
};

/// 요청마다 Completer 를 쌓아 두고 테스트가 원하는 순서로 응답을 풀어준다.
class _GatedPrediction extends PredictionService {
  _GatedPrediction() : super(SettingsStore());
  final List<Completer<Prediction>> calls = [];

  @override
  Future<Prediction> predict(
    File image, {
    bool centered = false,
    UploadMeta? meta,
    UploadProgress? onUploadProgress,
  }) {
    final c = Completer<Prediction>();
    calls.add(c);
    return c.future;
  }
}

class _GatedApi extends WasteApiClient {
  _GatedApi() : super(baseUrl: 'http://fake');
  final List<Completer<PredictionWithRegions>> regionCalls = [];
  final List<Offset?> regionTaps = [];
  final List<Completer<PredictObjects>> objectCalls = [];
  final List<Completer<Prediction>> hierCalls = [];

  @override
  Future<PredictionWithRegions> predictWithRegions(
    File imageFile, {
    double? tapX,
    double? tapY,
  }) {
    regionTaps.add(tapX == null ? null : Offset(tapX, tapY!));
    final c = Completer<PredictionWithRegions>();
    regionCalls.add(c);
    return c.future;
  }

  @override
  Future<PredictObjects> predictObjects(File imageFile) {
    final c = Completer<PredictObjects>();
    objectCalls.add(c);
    return c.future;
  }

  @override
  Future<Prediction> predictHier(
    File imageFile, {
    double? tapX,
    double? tapY,
    UploadMeta? meta,
    UploadProgress? onUploadProgress,
  }) {
    final c = Completer<Prediction>();
    hierCalls.add(c);
    return c.future;
  }

  @override
  Future<RegionInfo?> fetchRegionInfo(String sido, String sigungu) async =>
      null;
}

Future<void> _flush() => Future<void>.delayed(Duration.zero);

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  late Directory dir;
  late File image;
  late _GatedPrediction svc;
  late _GatedApi api;
  late ResultController c;

  setUp(() async {
    SharedPreferences.setMockInitialValues({});
    dir = await Directory.systemTemp.createTemp('race_');
    image = File('${dir.path}/a.png')
      ..writeAsBytesSync(img.encodePng(img.Image(width: 8, height: 8)));
    svc = _GatedPrediction();
    api = _GatedApi();
    c = ResultController(
      image: image,
      isSmartCapture: false,
      prediction: svc,
      api: () async => api,
      settings: SettingsStore(),
    );
  });
  tearDown(() => dir.delete(recursive: true));

  /// 첫 로드를 끝까지 풀어 결과 화면 상태로 만든다.
  Future<void> loaded({PredictObjects? objects}) async {
    final f = c.loadInitial();
    await _flush();
    svc.calls.single.complete(_pred('paper', uploadId: 'up-1'));
    api.regionCalls.single.complete(_regions('paper'));
    if (objects == null) {
      api.objectCalls.single.completeError(ApiException('no', statusCode: 404));
    } else {
      api.objectCalls.single.complete(objects);
    }
    await f;
  }

  test('첫 로드 — 분류가 먼저 와도 세 응답이 모일 때까지 반영하지 않고, '
      '모인 뒤 만장일치 규칙을 한 번에 판정한다', () async {
    final f = c.loadInitial();
    await _flush();
    svc.calls.single.complete(_pred('paper', uploadId: 'up-1'));
    await _flush();
    expect(c.prediction, isNull, reason: '물건 후보 도착 전 장면 결과를 먼저 그리지 않는다');
    expect(c.loading, isTrue);

    api.regionCalls.single.complete(_regions('paper'));
    await _flush();
    expect(c.prediction, isNull);

    // 물건 두 개가 모두 can — 장면 결과(paper)와 다르므로 물건 기준 채택
    api.objectCalls.single.complete(
      PredictObjects.fromJson({
        'objects': [_obj('can', 0.3), _obj('can', 0.7)],
      }),
    );
    await f;
    expect(c.loading, isFalse);
    expect(c.prediction?.predictedClass, 'can');
    expect(c.prediction?.uploadId, 'up-1');
    expect(c.selectedObject, 0);
    expect(c.canUndo, isTrue, reason: '자동 채택도 되돌리기로 장면 결과 복귀');
  });

  test('AI 정밀 분석이 채택된 장면 결과는 물건별 만장일치(CNN)로 바꾸지 않는다', () async {
    final f = c.loadInitial();
    await _flush();
    svc.calls.single.complete(
      const Prediction(
        predictedClass: 'pet',
        predictedIndex: 0,
        confidence: 0.9,
        allProbabilities: {'pet': 0.9},
        modelArch: 'test+vlm',
        inferenceMs: 1,
        uploadId: 'up-1',
        evidence: [
          EvidenceItem(
            type: 'vlm',
            token: '투명 PET 음료병',
            mappedClass: 'pet',
            score: 0.9,
          ),
        ],
      ),
    );
    api.regionCalls.single.complete(_regions('pet'));
    api.objectCalls.single.complete(
      PredictObjects.fromJson({
        'objects': [_obj('cardboard', 0.3), _obj('cardboard', 0.7)],
      }),
    );
    await f;
    expect(c.prediction?.predictedClass, 'pet');
    expect(c.selectedObject, isNull);
    expect(c.sceneNote, contains('AI 정밀 분석 기준'));
  });

  test('물건 선택 뒤 빗금 재분석이 늦게 와도 되돌리기 이후면 버린다', () async {
    await loaded(
      objects: PredictObjects.fromJson({
        'objects': [_obj('can', 0.3), _obj('glass', 0.7)],
      }),
    );
    final initialRegions = c.regions;
    expect(initialRegions, isNotNull);

    c.selectObject(1, haptic: false);
    await _flush();
    expect(api.regionTaps.last?.dx, closeTo(0.7, 1e-9));
    c.undo();
    expect(c.regions, same(initialRegions));

    api.regionCalls.last.complete(_regions('glass'));
    await _flush();
    expect(c.regions, same(initialRegions), reason: '옛 선택의 빗금이 복귀 화면을 덮으면 안 된다');
    expect(c.prediction?.predictedClass, 'paper');
  });

  test('탭 재분류 응답이 되돌리기 뒤에 도착하면 버리고 바쁨 표시도 남지 않는다', () async {
    await loaded(
      objects: PredictObjects.fromJson({
        'objects': [_obj('can', 0.3), _obj('glass', 0.7)],
      }),
    );
    c.selectObject(1, haptic: false); // 되돌릴 이력 1개
    await _flush();
    final tap = c.reclassifyAt(0.2, 0.2);
    await _flush();
    expect(c.retapBusy, isTrue);

    c.undo(); // 선택 이전(장면 결과)으로
    expect(c.retapBusy, isFalse);
    expect(c.prediction?.predictedClass, 'paper');

    api.hierCalls.single.complete(_pred('metal'));
    await tap;
    expect(
      c.prediction?.predictedClass,
      'paper',
      reason: '되돌린 화면을 옛 탭 결과가 덮지 않는다',
    );
    expect(c.retapBusy, isFalse);
    expect(
      api.regionTaps.where((t) => t == const Offset(0.2, 0.2)),
      isEmpty,
      reason: '폐기된 탭은 빗금 재분석도 요청하지 않는다',
    );
  });

  test('연속 탭 — 첫 탭 응답이 두 번째 탭 뒤에 와도 두 번째 결과가 유지된다', () async {
    await loaded();
    final first = c.reclassifyAt(0.2, 0.2);
    await _flush();
    final second = c.reclassifyAt(0.8, 0.8);
    await _flush();

    api.hierCalls[1].complete(_pred('glass'));
    await second;
    expect(c.prediction?.predictedClass, 'glass');

    api.hierCalls[0].complete(_pred('metal'));
    await first;
    expect(c.prediction?.predictedClass, 'glass');
    expect(c.lastTapNorm, const Offset(0.8, 0.8));
  });

  test('첫 탭의 빗금 재분석이 두 번째 탭 뒤에 도착하면 버린다', () async {
    await loaded();
    final first = c.reclassifyAt(0.2, 0.2);
    await _flush();
    api.hierCalls[0].complete(_pred('metal'));
    await first; // 첫 탭 결과 반영 + 빗금 재분석 요청(대기 중)
    await _flush();
    final firstRegions = api.regionCalls.last;

    final second = c.reclassifyAt(0.8, 0.8);
    await _flush();
    api.hierCalls[1].complete(_pred('glass'));
    await second;
    await _flush();
    final secondRegions = api.regionCalls.last;
    expect(identical(firstRegions, secondRegions), isFalse);

    secondRegions.complete(_regions('glass'));
    await _flush();
    firstRegions.complete(_regions('metal'));
    await _flush();
    expect(c.regions?.regions.single.slug, 'glass');
  });

  test('옛 탭이 실패해도 다른 조작으로 넘어간 뒤면 오류를 던지지 않는다', () async {
    await loaded();
    final first = c.reclassifyAt(0.2, 0.2);
    await _flush();
    final second = c.reclassifyAt(0.8, 0.8);
    await _flush();
    api.hierCalls[0].completeError(ApiException('x', statusCode: 503));
    await expectLater(first, completes);
    api.hierCalls[1].complete(_pred('glass'));
    await second;
    expect(c.prediction?.predictedClass, 'glass');
    expect(c.retapBusy, isFalse);
  });

  test('재시도 — 이전 세대 첫 로드 응답은 늦게 와도 반영하지 않는다', () async {
    final old = c.loadInitial();
    await _flush();
    c.retry(); // 새 세대 첫 로드 시작
    await _flush();
    expect(svc.calls, hasLength(2));

    svc.calls[1].complete(_pred('glass', uploadId: 'up-2'));
    api.regionCalls[1].complete(_regions('glass'));
    api.objectCalls[1].completeError(ApiException('no', statusCode: 404));
    await _flush();
    await _flush();
    expect(c.prediction?.predictedClass, 'glass');

    svc.calls[0].complete(_pred('paper', uploadId: 'up-1'));
    api.regionCalls[0].complete(_regions('paper'));
    api.objectCalls[0].completeError(ApiException('no', statusCode: 404));
    await old;
    expect(c.prediction?.predictedClass, 'glass');
    expect(c.regions?.regions.single.slug, 'glass');
  });
}
