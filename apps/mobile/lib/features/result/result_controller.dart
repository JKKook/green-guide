/// 결과 화면 상태·비동기 오케스트레이션 — 위젯(ResultModal)과 분리.
///
/// 분류(/predict-hier)·품질 평가·재질 영역·물건 후보·지역 규정·원본 크기의
/// 6개 작업을 병렬로 돌리고, 탭-투-셀렉트/후보 선택/되돌리기 상태를 관리한다.
/// 위젯은 [ChangeNotifier] 를 구독해 그리기만 하고, 컨텍스트가 필요한 일
/// (스낵바·네비게이션)은 위젯이 맡는다.
library;

import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/widgets.dart';

import '../../api/api_client.dart';
import '../../api/models.dart';
import '../../core/di/app_scope.dart';
import '../../core/log.dart';
import '../../data/haptics.dart';
import '../../data/image_quality.dart';
import '../../data/settings_store.dart';
import '../../services/prediction_service.dart';

typedef ApiFactory = Future<WasteApiClient> Function();

class ResultController extends ChangeNotifier {
  ResultController({
    required this.image,
    required this.isSmartCapture,
    this.meta,
    ImageQualityResult? initialQuality,
    PredictionService? prediction,
    ApiFactory? api,
    SettingsStore? settings,
  }) : _quality = initialQuality,
       _prediction = prediction ?? AppScope.prediction,
       _api = api ?? AppScope.api,
       _settings = settings ?? AppScope.settings;

  final File image;
  final bool isSmartCapture;

  /// 업로드 폼 필드 메타(촬영 경로·EXIF 방향·품질 측정값·크롭) — 분류 요청에 실린다.
  final UploadMeta? meta;
  final PredictionService _prediction;
  final ApiFactory _api;
  final SettingsStore _settings;

  final String capturedAt = _clockLabel(DateTime.now());

  Prediction? _result;
  ImageQualityResult? _quality; // 캡처 사진 품질 — 게이트에서 전달받으면 재평가 생략
  PredictionWithRegions? _regions; // 다중재질 영역 + 빗금 오버레이
  RegionInfo? _regionInfo; // 지역별 배출 규정 (지역 미설정/미적재면 null)
  bool _regionSet = false; // 지역 설정 여부 (안내 캡션 분기)
  String? _error;
  bool _classifyDone = false; // /predict 완료(성공/실패)
  bool _regionsDone = false; // /predict-with-regions 완료(성공/실패)
  int _uploadSent = 0; // 업로드 진행 바이트 (실측)
  int _uploadTotal = 0;

  // 탭-투-셀렉트 — 혼재 장면에서 사용자가 지목한 객체만 재분류
  Size? _imgSize; // 원본 이미지 크기 (cover 역변환용)
  bool _retapBusy = false; // 탭 재분류 진행 중
  Offset? _lastTapNorm; // 마지막 탭 위치 (마커 표시용, 정규화)

  // 탐지-후-분류 — 장면의 객체 후보들 (≥2 면 후보 카드 표시)
  PredictObjects? _objects;

  /// 최초 분류(/predict-hier)의 업로드 ID — 물건 후보를 선택해 결과를 바꿔도
  /// 피드백은 같은 사진(이 ID)에 대해 보내야 서버에 반영된다.
  String? _uploadId;

  /// 물건별 결과가 만장일치로 장면 결과와 달라 자동으로 물건 기준을 채택했는지.
  bool _autoAdoptedObject = false;
  int? _selectedObject; // 후보 카드에서 선택된 인덱스

  // 되돌리기 스택 — 탭/후보선택으로 결과가 바뀌기 전 상태 스냅샷.
  // "다중 재질 전체 결과 → 단일 물건" 전환 후 원래 결과로 복귀 가능하게.
  final List<_ViewSnapshot> _viewStack = [];
  static const int kMaxUndo = 10;

  bool _disposed = false;

  Prediction? get prediction => _result;
  ImageQualityResult? get quality => _quality;
  PredictionWithRegions? get regions => _regions;
  RegionInfo? get regionInfo => _regionInfo;
  bool get regionSet => _regionSet;
  String? get error => _error;
  bool get classifyDone => _classifyDone;
  bool get regionsDone => _regionsDone;
  int get uploadSent => _uploadSent;
  int get uploadTotal => _uploadTotal;
  Size? get imgSize => _imgSize;
  bool get retapBusy => _retapBusy;
  Offset? get lastTapNorm => _lastTapNorm;
  PredictObjects? get objects => _objects;
  int? get selectedObject => _selectedObject;
  bool get canUndo => _viewStack.isNotEmpty;

  /// 로더는 분류와 재질 분석이 모두 끝나야 사라짐 → 라벨·빗금·뱃지가 한꺼번에 표시.
  /// 둘이 따로 끝나서 "라벨 먼저, 빗금 늦게" 로 어긋나 보이는 문제 해결.
  bool get loading => _error == null && (!_classifyDone || !_regionsDone);

  /// 전처리 = 로컬 품질 평가 + 원본 크기 해석 (둘 다 실제 수행되는 단계).
  bool get preprocessDone => _quality != null && _imgSize != null;

  /// 분석된 재질 아래 한 줄 안내 — 장면 결과와 물건별 결과가 다를 때만.
  String? get sceneNote {
    final objs = _objects;
    if (objs == null || !objs.isMultiObject) return null;
    if (_autoAdoptedObject && _selectedObject != null) {
      return '사진 속 물건들이 모두 같은 재질이라 물건 기준으로 분류했어요 · '
          '되돌리기로 전체 사진 결과를 볼 수 있어요';
    }
    if (_selectedObject != null) return null;
    final labels = objs.objects.map((o) => o.displayClass).toSet();
    if (_result != null && !labels.contains(_result!.predictedClass)) {
      return '전체 사진 기준 결과예요 · 물건별 결과와 다르면 위 목록에서 '
          '번호를 탭해 주세요';
    }
    return null;
  }

  /// 6개 작업 병렬 시작 — 위젯 initState 에서 1회.
  void start() {
    unawaited(classify());
    if (_quality == null) unawaited(_assessQuality());
    unawaited(fetchRegions());
    unawaited(_fetchObjects());
    unawaited(_fetchRegionInfo());
    resolveImageSize();
  }

  /// 오류 상태에서 다시 시도 — 분류·재질 분석만 재실행.
  void retry() {
    _error = null;
    _classifyDone = false;
    _regionsDone = false;
    _notify();
    unawaited(classify());
    unawaited(fetchRegions());
  }

  void _notify() {
    if (!_disposed) notifyListeners();
  }

  @override
  void dispose() {
    _disposed = true;
    super.dispose();
  }

  void _pushSnapshot() {
    final p = _result;
    if (p == null) return;
    _viewStack.add(
      _ViewSnapshot(
        prediction: p,
        selectedObject: _selectedObject,
        tapNorm: _lastTapNorm,
        objects: _objects,
        regions: _regions,
      ),
    );
    if (_viewStack.length > kMaxUndo) _viewStack.removeAt(0);
  }

  void undo() {
    if (_viewStack.isEmpty) return;
    Haptics.selection();
    final s = _viewStack.removeLast();
    _result = s.prediction;
    _selectedObject = s.selectedObject;
    _lastTapNorm = s.tapNorm;
    _objects = s.objects; // 다중 분류 화면·뱃지 복귀
    _regions = s.regions; // 빗금 오버레이 복귀
    _notify();
  }

  /// 지역별 배출 규정 — 설정된 지역이 있으면 조회 (실패해도 무해).
  Future<void> _fetchRegionInfo() async {
    try {
      final region = await _settings.getRegion();
      if (region == null) return;
      _regionSet = true;
      _notify();
      final client = await _api();
      final info = await client.fetchRegionInfo(region.$1, region.$2);
      if (_disposed || info == null) return;
      _regionInfo = info;
      _notify();
    } catch (_) {
      // 지역 규정은 선택적 향상 — 실패하면 전국 공통 안내로 표시
    }
  }

  /// 탐지-후-분류 — 장면의 객체 후보들 (백그라운드, 실패해도 무해).
  Future<void> _fetchObjects() async {
    try {
      final client = await _api();
      final r = await client.predictObjects(image);
      if (_disposed) return;
      _objects = r;
      _notify();
      _maybeAdoptUnanimousObject(r);
    } catch (_) {
      // 구버전 서버(404) 등 — 후보 카드 없이 진행
    }
  }

  /// 장면 전체 결과(/predict-hier, 원본 그대로)와 물건별 결과(/predict-objects,
  /// saliency 성분 크롭)는 입력이 달라 서로 다를 수 있다. 물건들이 전부 같은
  /// 재질인데 장면 결과만 다르면 배경/혼재에 끌린 오분류일 가능성이 커서
  /// 물건 기준을 채택한다(되돌리기 가능). 사용자가 이미 탭/선택했으면 건드리지 않음.
  void _maybeAdoptUnanimousObject(PredictObjects r) {
    if (!r.isMultiObject || _selectedObject != null || _lastTapNorm != null) {
      return;
    }
    final current = _result;
    if (current == null || _autoAdoptedObject) return;
    final slugs = r.objects.map((o) => o.displayClass).toSet();
    if (slugs.length != 1 || slugs.first == current.predictedClass) return;
    _autoAdoptedObject = true;
    selectObject(0, haptic: false);
  }

  /// 물건 후보 결과에 원본 업로드 ID 를 얹는다 — 피드백 전송용.
  Prediction _withUploadId(Prediction p) => Prediction(
    predictedClass: p.predictedClass,
    predictedIndex: p.predictedIndex,
    confidence: p.confidence,
    allProbabilities: p.allProbabilities,
    modelArch: p.modelArch,
    inferenceMs: p.inferenceMs,
    uploadId: p.uploadId ?? _uploadId,
    hier: p.hier,
    evidence: p.evidence,
  );

  /// 후보 카드에서 물건 선택 → 메인 결과 카드 교체.
  void selectObject(int idx, {bool haptic = true}) {
    final objs = _objects?.objects;
    if (objs == null || idx >= objs.length) return;
    if (haptic) Haptics.selection();
    _pushSnapshot(); // 되돌리기용 — 선택 전 상태 보존
    _selectedObject = idx;
    _result = _withUploadId(objs[idx].toPrediction());
    _lastTapNorm = Offset(objs[idx].cx, objs[idx].cy);
    _notify();
    // 선택한 물건 성분에 빗금 재분석 집중
    unawaited(fetchRegions(tap: Offset(objs[idx].cx, objs[idx].cy)));
  }

  /// 원본 이미지 픽셀 크기 로딩 — 탭 좌표를 BoxFit.cover 역변환할 때 필요.
  void resolveImageSize() {
    final stream = FileImage(image).resolve(const ImageConfiguration());
    stream.addListener(
      ImageStreamListener((info, _) {
        appLog(
          '[tap-select] imgSize resolved: '
          '${info.image.width}x${info.image.height}',
        );
        if (_disposed) return;
        _imgSize = Size(
          info.image.width.toDouble(),
          info.image.height.toDouble(),
        );
        _notify();
      }, onError: (e, _) => appLog('[tap-select] imgSize resolve 실패: $e')),
    );
  }

  /// 표시 좌표 → 원본 정규화 좌표 (BoxFit.cover 보정).
  /// 원본 크기 미해석이면 null (위젯이 안내 + 재해석), 범위 밖이면 null.
  Offset? tapToNorm(Offset local, Size view) {
    appLog(
      '[tap-select] tap local=$local view=$view '
      'imgSize=$_imgSize loading=$loading busy=$_retapBusy',
    );
    final img = _imgSize;
    if (img == null) return null;
    final scale = math.max(view.width / img.width, view.height / img.height);
    final dx = (img.width * scale - view.width) / 2;
    final dy = (img.height * scale - view.height) / 2;
    final nx = ((local.dx + dx) / scale) / img.width;
    final ny = ((local.dy + dy) / scale) / img.height;
    if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return null;
    return Offset(nx, ny);
  }

  /// 탭 지점의 객체만 서버(u2netp 성분 크롭)로 재분류.
  /// 실패 시 예외를 다시 던진다 — 위젯이 스낵바로 안내.
  Future<void> reclassifyAt(double nx, double ny) async {
    _retapBusy = true;
    _lastTapNorm = Offset(nx, ny);
    _notify();
    Haptics.selection();
    try {
      final client = await _api();
      final r = await client.predictHier(image, tapX: nx, tapY: ny);
      if (_disposed) return;
      _pushSnapshot(); // 되돌리기용 — 성공 시에만 이전 상태 보존 (다중 화면·빗금 포함)
      _result = r;
      // 탭 재분류 = 단일 분류로 전환 — 다중 뱃지·목록을 접어야 새 결과가
      // 교체됐음을 인식할 수 있음. 원래 다중 화면은 되돌리기로 복귀.
      _objects = null;
      _selectedObject = null;
      _notify();
      // 빗금(재질 영역)도 탭 성분 기준으로 재분석 — 마커와 함께 이동
      unawaited(fetchRegions(tap: Offset(nx, ny)));
    } finally {
      _retapBusy = false;
      _notify();
    }
  }

  Future<void> _assessQuality() async {
    final q = await assessImageQuality(image);
    if (_disposed) return;
    _quality = q;
    _notify();
  }

  /// 다중재질 영역 + 빗금 오버레이 — 서버 /predict-with-regions 호출.
  /// 분류와 병렬, 실패해도 원본 표시. 확실히 다른 재질만 영역 분리.
  /// tap 좌표를 주면 그 성분에 집중한 재분석 — 탭 시 빗금도 함께 이동.
  Future<void> fetchRegions({Offset? tap}) async {
    try {
      final client = await _api();
      final r = await client.predictWithRegions(
        image,
        tapX: tap?.dx,
        tapY: tap?.dy,
      );
      if (_disposed) return;
      if (r.hasOverlay) {
        _regions = r;
      } else if (tap != null) {
        _regions = null; // 탭 영역에서 재질 미검출 — 이전 빗금 잔상 제거
      }
      _regionsDone = true;
      _notify();
    } catch (_) {
      // 오프라인/실패 — 원본 그대로 (오버레이는 선택적 향상)
      if (_disposed) return;
      _regionsDone = true;
      _notify();
    }
  }

  Future<void> classify() async {
    try {
      final result = await _prediction.predict(
        image,
        centered: isSmartCapture,
        meta: meta,
        onUploadProgress: (sent, total) {
          if (_disposed) return;
          _uploadSent = sent;
          _uploadTotal = total;
          _notify();
        },
      );
      if (_disposed) return;
      Haptics.medium();
      _result = result;
      _uploadId ??= result.uploadId;
      _classifyDone = true;
      _notify();
    } catch (e) {
      if (_disposed) return;
      _error = friendlyError(e);
      _notify();
    }
  }
}

/// '오후 2:41' 형식 시각 라벨.
String _clockLabel(DateTime t) {
  final h12 = t.hour % 12 == 0 ? 12 : t.hour % 12;
  return '${t.hour >= 12 ? '오후' : '오전'} $h12:${t.minute.toString().padLeft(2, '0')}';
}

/// 되돌리기 스냅샷 — 탭/후보선택으로 결과가 교체되기 전의 뷰 상태.
class _ViewSnapshot {
  final Prediction prediction;
  final int? selectedObject;
  final Offset? tapNorm;
  final PredictObjects? objects; // 다중 분류 화면 복귀용
  final PredictionWithRegions? regions; // 빗금 오버레이 복귀용
  const _ViewSnapshot({
    required this.prediction,
    this.selectedObject,
    this.tapNorm,
    this.objects,
    this.regions,
  });
}
