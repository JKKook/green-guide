/// 캡처된 사진의 분류 결과를 모달로 표시.
/// LiveCameraScreen 에서 사용.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../api/api_client.dart';
import '../api/models.dart';
import '../core/di/app_scope.dart';
import '../core/feedback/app_snackbar.dart';
import '../core/ui/ds_card.dart';
import '../data/confidence.dart';
import '../data/haptics.dart';
import '../data/image_quality.dart';
import '../data/waste_info.dart';
import '../services/prediction_service.dart' show isCloudFallback;
import '../theme/app_theme.dart';
import '../theme/design_tokens.dart';
import '../widgets/animated_entry.dart';
import '../widgets/feedback_card.dart';
import '../widgets/hier_badge.dart';


/// 모달 표시 — 캡처된 이미지를 분류하고 결과 + 피드백 UI 렌더링.
///
/// [isSmartCapture]=true (LiveCameraScreen 에서 호출) 일 때 cloud 경로가
/// `/predict-centered` 로 호출돼 객체 자동 크롭 후 분류 (정확도 ↑).
///
/// 반환값:
///   - `true`: "닫기" 탭됨 → 카메라 화면도 닫음
///   - `false`: "다시 촬영" 탭됨 → 모달만 닫고 카메라 유지
///   - `null`: 사용자가 swipe-to-dismiss
Future<bool?> showResultModal(
  BuildContext context,
  File image, {
  bool isSmartCapture = false,
}) {
  return showModalBottomSheet<bool>(
    context: context,
    isScrollControlled: true,
    useSafeArea: true,
    enableDrag: false,  // 풀스크린 — 드래그 dismiss 비활성
    backgroundColor: Theme.of(context).scaffoldBackgroundColor,
    barrierColor: Colors.black,
    shape: const RoundedRectangleBorder(),
    builder: (_) => DraggableScrollableSheet(
      initialChildSize: 1.0,
      minChildSize: 1.0,
      expand: false,
      builder: (_, controller) => _ResultModal(
        image: image,
        scrollController: controller,
        isSmartCapture: isSmartCapture,
      ),
    ),
  );
}


class _ResultModal extends StatefulWidget {
  final File image;
  final ScrollController scrollController;
  final bool isSmartCapture;
  const _ResultModal({
    required this.image,
    required this.scrollController,
    this.isSmartCapture = false,
  });

  @override
  State<_ResultModal> createState() => _ResultModalState();
}


class _ResultModalState extends State<_ResultModal> {
  final String _capturedAt = _clockLabel(DateTime.now());

  Prediction? _prediction;
  ImageQualityResult? _quality;       // 캡처 사진 품질 (어두움/흔들림)
  PredictionWithRegions? _regions;    // 다중재질 영역 + 빗금 오버레이
  RegionInfo? _regionInfo;            // 지역별 배출 규정 (지역 미설정/미적재면 null)
  bool _regionSet = false;            // 지역 설정 여부 (안내 캡션 분기)
  String? _error;
  bool _classifyDone = false;         // /predict 완료(성공/실패)
  bool _regionsDone = false;          // /predict-with-regions 완료(성공/실패)
  int _uploadSent = 0;                // 업로드 진행 바이트 (실측)
  int _uploadTotal = 0;

  // 탭-투-셀렉트 — 혼재 장면에서 사용자가 지목한 객체만 재분류
  Size? _imgSize;                     // 원본 이미지 크기 (cover 역변환용)
  bool _retapBusy = false;            // 탭 재분류 진행 중
  Offset? _lastTapNorm;               // 마지막 탭 위치 (마커 표시용, 정규화)

  // 탐지-후-분류 — 장면의 객체 후보들 (≥2 면 후보 카드 표시)
  PredictObjects? _objects;

  /// 최초 분류(/predict-hier)의 업로드 ID — 물건 후보를 선택해 결과를 바꿔도
  /// 피드백은 같은 사진(이 ID)에 대해 보내야 서버에 반영된다.
  String? _uploadId;

  /// 물건별 결과가 만장일치로 장면 결과와 달라 자동으로 물건 기준을 채택했는지.
  bool _autoAdoptedObject = false;
  int? _selectedObject;               // 후보 카드에서 선택된 인덱스

  // 되돌리기 스택 — 탭/후보선택으로 결과가 바뀌기 전 상태 스냅샷.
  // "다중 재질 전체 결과 → 단일 물건" 전환 후 원래 결과로 복귀 가능하게.
  final List<_ViewSnapshot> _viewStack = [];

  void _pushSnapshot() {
    final p = _prediction;
    if (p == null) return;
    _viewStack.add(_ViewSnapshot(
      prediction: p,
      selectedObject: _selectedObject,
      tapNorm: _lastTapNorm,
      objects: _objects,
      regions: _regions,
    ));
    if (_viewStack.length > 10) _viewStack.removeAt(0);
  }

  void _undo() {
    if (_viewStack.isEmpty) return;
    Haptics.selection();
    final s = _viewStack.removeLast();
    setState(() {
      _prediction = s.prediction;
      _selectedObject = s.selectedObject;
      _lastTapNorm = s.tapNorm;
      _objects = s.objects;   // 다중 분류 화면·뱃지 복귀
      _regions = s.regions;   // 빗금 오버레이 복귀
    });
  }

  /// 로더는 분류와 재질 분석이 모두 끝나야 사라짐 → 라벨·빗금·뱃지가 한꺼번에 표시.
  /// 둘이 따로 끝나서 "라벨 먼저, 빗금 늦게" 로 어긋나 보이는 문제 해결.
  bool get _loading => _error == null && (!_classifyDone || !_regionsDone);

  /// 전처리 = 로컬 품질 평가 + 원본 크기 해석 (둘 다 실제 수행되는 단계).
  bool get _preprocessDone => _quality != null && _imgSize != null;

  @override
  void initState() {
    super.initState();
    _classify();
    _assessQuality();
    _fetchRegions();
    _fetchObjects();
    _fetchRegionInfo();
    _resolveImageSize();
  }

  /// 지역별 배출 규정 — 설정된 지역이 있으면 조회 (실패해도 무해).
  Future<void> _fetchRegionInfo() async {
    try {
      final region = await AppScope.settings.getRegion();
      if (region == null) return;
      if (mounted) setState(() => _regionSet = true);
      final client = await AppScope.api();
      final info = await client.fetchRegionInfo(region.$1, region.$2);
      if (mounted && info != null) setState(() => _regionInfo = info);
    } catch (_) {}
  }

  /// 탐지-후-분류 — 장면의 객체 후보들 (백그라운드, 실패해도 무해).
  Future<void> _fetchObjects() async {
    try {
      final client = await AppScope.api();
      final r = await client.predictObjects(widget.image);
      if (!mounted) return;
      setState(() => _objects = r);
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
    final current = _prediction;
    if (current == null || _autoAdoptedObject) return;
    final slugs = r.objects.map((o) => o.displayClass).toSet();
    if (slugs.length != 1 || slugs.first == current.predictedClass) return;
    _autoAdoptedObject = true;
    _selectObject(0, haptic: false);
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
  void _selectObject(int idx, {bool haptic = true}) {
    final objs = _objects?.objects;
    if (objs == null || idx >= objs.length) return;
    if (haptic) Haptics.selection();
    _pushSnapshot();  // 되돌리기용 — 선택 전 상태 보존
    setState(() {
      _selectedObject = idx;
      _prediction = _withUploadId(objs[idx].toPrediction());
      _lastTapNorm = Offset(objs[idx].cx, objs[idx].cy);
    });
    // 선택한 물건 성분에 빗금 재분석 집중
    unawaited(_fetchRegions(tap: Offset(objs[idx].cx, objs[idx].cy)));
  }

  /// 분석된 재질 아래 한 줄 안내 — 장면 결과와 물건별 결과가 다를 때만.
  String? get _sceneNote {
    final objs = _objects;
    if (objs == null || !objs.isMultiObject) return null;
    if (_autoAdoptedObject && _selectedObject != null) {
      return '사진 속 물건들이 모두 같은 재질이라 물건 기준으로 분류했어요 · '
          '되돌리기로 전체 사진 결과를 볼 수 있어요';
    }
    if (_selectedObject != null) return null;
    final labels = objs.objects.map((o) => o.displayClass).toSet();
    if (_prediction != null && !labels.contains(_prediction!.predictedClass)) {
      return '전체 사진 기준 결과예요 · 물건별 결과와 다르면 위 목록에서 '
          '번호를 탭해 주세요';
    }
    return null;
  }

  /// 원본 이미지 픽셀 크기 로딩 — 탭 좌표를 BoxFit.cover 역변환할 때 필요.
  void _resolveImageSize() {
    final stream = FileImage(widget.image).resolve(const ImageConfiguration());
    stream.addListener(ImageStreamListener(
      (info, _) {
        debugPrint('[tap-select] imgSize resolved: '
            '${info.image.width}x${info.image.height}');
        if (mounted) {
          setState(() => _imgSize =
              Size(info.image.width.toDouble(), info.image.height.toDouble()));
        }
      },
      onError: (e, _) => debugPrint('[tap-select] imgSize resolve 실패: $e'),
    ));
  }

  /// 이미지 위 탭 → 표시 좌표를 원본 정규화 좌표로 역변환 (BoxFit.cover 보정).
  void _onImageTap(Offset local, Size view) {
    debugPrint('[tap-select] tap local=$local view=$view '
        'imgSize=$_imgSize loading=$_loading busy=$_retapBusy');
    if (_retapBusy || _loading) return;
    final img = _imgSize;
    if (img == null) {
      // 크기 미해석 — 조용히 무시하지 않고 피드백 + 재시도
      _resolveImageSize();
      showAppSnackBar(
        context,
        '사진 정보를 준비 중이에요 — 잠시 후 다시 탭해주세요',
        duration: const Duration(seconds: 2),
      );
      return;
    }
    final scale = math.max(view.width / img.width, view.height / img.height);
    final dx = (img.width * scale - view.width) / 2;
    final dy = (img.height * scale - view.height) / 2;
    final nx = ((local.dx + dx) / scale) / img.width;
    final ny = ((local.dy + dy) / scale) / img.height;
    if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return;
    _reclassifyAt(nx, ny);
  }

  /// 탭 지점의 객체만 서버(u2netp 성분 크롭)로 재분류.
  Future<void> _reclassifyAt(double nx, double ny) async {
    setState(() {
      _retapBusy = true;
      _lastTapNorm = Offset(nx, ny);
    });
    Haptics.selection();
    try {
      final client = await AppScope.api();
      final r = await client.predictHier(widget.image, tapX: nx, tapY: ny);
      if (!mounted) return;
      _pushSnapshot();  // 되돌리기용 — 성공 시에만 이전 상태 보존 (다중 화면·빗금 포함)
      setState(() {
        _prediction = r;
        // 탭 재분류 = 단일 분류로 전환 — 다중 뱃지·목록을 접어야 새 결과가
        // 교체됐음을 인식할 수 있음. 원래 다중 화면은 되돌리기로 복귀.
        _objects = null;
        _selectedObject = null;
      });
      // 빗금(재질 영역)도 탭 성분 기준으로 재분석 — 마커와 함께 이동
      unawaited(_fetchRegions(tap: Offset(nx, ny)));
    } on Exception catch (e) {
      if (!mounted) return;
      showAppErrorSnackBar(context, e);
    } finally {
      if (mounted) setState(() => _retapBusy = false);
    }
  }

  Future<void> _assessQuality() async {
    final q = await assessImageQuality(widget.image);
    if (!mounted) return;
    setState(() => _quality = q);
  }

  /// 다중재질 영역 + 빗금 오버레이 — 서버 /predict-with-regions 호출.
  /// 분류와 병렬, 실패해도 원본 표시. 확실히 다른 재질만 영역 분리.
  /// tap 좌표를 주면 그 성분에 집중한 재분석 — 탭 시 빗금도 함께 이동.
  Future<void> _fetchRegions({Offset? tap}) async {
    try {
      final client = await AppScope.api();
      final r = await client.predictWithRegions(widget.image,
          tapX: tap?.dx, tapY: tap?.dy);
      if (!mounted) return;
      setState(() {
        if (r.hasOverlay) {
          _regions = r;
        } else if (tap != null) {
          _regions = null; // 탭 영역에서 재질 미검출 — 이전 빗금 잔상 제거
        }
        _regionsDone = true;
      });
    } catch (_) {
      // 오프라인/실패 — 원본 그대로 (오버레이는 선택적 향상)
      if (!mounted) return;
      setState(() => _regionsDone = true);
    }
  }

  Future<void> _classify() async {
    try {
      final result = await AppScope.prediction.predict(
        widget.image,
        centered: widget.isSmartCapture,
        onUploadProgress: (sent, total) {
          if (!mounted) return;
          setState(() {
            _uploadSent = sent;
            _uploadTotal = total;
          });
        },
      );

      if (!mounted) return;
      Haptics.medium();
      setState(() {
        _prediction = result;
        _uploadId ??= result.uploadId;
        _classifyDone = true;
      });

    } catch (e) {
      if (!mounted) return;
      setState(() => _error = friendlyError(e));
    }
  }

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    final loading = _loading;

    return AnimatedContainer(
      duration: const Duration(milliseconds: 250),
      color: loading
          ? kInkDeep
          : Theme.of(context).scaffoldBackgroundColor,
      child: Column(
        children: [
          // 헤더 — 닫기 · 제목 (시안 16c 재질 분석 / 16e 분석 결과)
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 8, 20, 12),
            child: Row(
              children: [
                Material(
                  color: loading
                      ? Colors.white.withValues(alpha: 0.10)
                      : t.surface,
                  shape: CircleBorder(
                    side: loading
                        ? BorderSide.none
                        : BorderSide(color: t.border),
                  ),
                  child: Semantics(
                    button: true,
                    label: '분석 결과 닫기',
                    child: InkWell(
                    customBorder: const CircleBorder(),
                    onTap: () {
                      Haptics.selection();
                      Navigator.of(context).pop(true);
                    },
                    child: SizedBox(
                      width: 38,
                      height: 38,
                      child: Icon(
                        Icons.close,
                        size: 17,
                        color: loading ? kNeutral100 : t.muted2,
                      ),
                    ),
                  ),
                  ),
                ),
                Expanded(
                  child: Text(
                    loading ? '재질 분석' : '분석 결과',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w700,
                      color: loading ? kNeutral100 : null,
                    ),
                  ),
                ),
                const SizedBox(width: 38),
              ],
            ),
          ),
          Expanded(
            // 분석이 다 끝나기 전엔 이미지/라벨/빗금/배출방법 모두 숨기고 로더만 표시.
            child: loading
                ? _AnalysisLoading(
                    uploadSent: _uploadSent,
                    uploadTotal: _uploadTotal,
                    preprocessDone: _preprocessDone,
                    classifyDone: _classifyDone,
                    resultDone: _regionsDone,
                    isSmartCapture: widget.isSmartCapture,
                    onCancel: () {
                      Haptics.selection();
                      Navigator.of(context).pop(false); // 다시 촬영
                    },
                  )
                : ListView(
                    controller: widget.scrollController,
                    padding: const EdgeInsets.fromLTRB(20, 6, 20, 30),
                    children: [
                      // 분석한 사진 + 영역별 빗금 오버레이 + 재질 라벨
                      // 탭-투-셀렉트: 물건을 탭하면 그 객체만 재분류
                      ClipRRect(
                        borderRadius: BorderRadius.circular(24),
                        child: AspectRatio(
                          aspectRatio: 16 / 10,
                          child: LayoutBuilder(
                            builder: (ctx, c) {
                              final view = Size(c.maxWidth, c.maxHeight);
                              return GestureDetector(
                                behavior: HitTestBehavior.opaque,
                                onTapUp: (d) =>
                                    _onImageTap(d.localPosition, view),
                                child: Stack(
                                  fit: StackFit.expand,
                                  children: [
                                    _RegionsView(
                                      image: widget.image,
                                      regions: _regions,
                                    ),
                                    if (_lastTapNorm != null &&
                                        _imgSize != null)
                                      _TapMarker(
                                        norm: _lastTapNorm!,
                                        imgSize: _imgSize!,
                                        view: view,
                                      ),
                                    if (_objects != null &&
                                        _objects!.isMultiObject &&
                                        _imgSize != null)
                                      for (int i = 0;
                                          i < _objects!.objects.length;
                                          i++)
                                        _ObjectMarker(
                                          index: i,
                                          candidate: _objects!.objects[i],
                                          imgSize: _imgSize!,
                                          view: view,
                                          selected: _selectedObject == i,
                                          onTap: () => _selectObject(i),
                                        ),
                                    if (_retapBusy)
                                      Container(
                                        color: Colors.black38,
                                        child: const Center(
                                          child: CircularProgressIndicator(
                                              color: Colors.white),
                                        ),
                                      ),
                                    // 출처 pill — 스마트 촬영 · 시각 (시안 16e)
                                    Positioned(
                                      left: 12,
                                      top: 12,
                                      child: Container(
                                        padding: const EdgeInsets.symmetric(
                                            horizontal: 11, vertical: 6),
                                        decoration: BoxDecoration(
                                          color: kInkDeep
                                              .withValues(alpha: 0.6),
                                          borderRadius:
                                              BorderRadius.circular(999),
                                        ),
                                        child: Row(
                                          mainAxisSize: MainAxisSize.min,
                                          children: [
                                            Icon(
                                              widget.isSmartCapture
                                                  ? Icons.photo_camera_outlined
                                                  : Icons.image_outlined,
                                              size: 12,
                                              color: kNeutral100,
                                            ),
                                            const SizedBox(width: 5),
                                            Text(
                                              '${widget.isSmartCapture ? '스마트 촬영' : '갤러리'} · $_capturedAt',
                                              style: const TextStyle(
                                                fontSize: 11,
                                                fontWeight: FontWeight.w600,
                                                color: kNeutral100,
                                              ),
                                            ),
                                          ],
                                        ),
                                      ),
                                    ),
                                    // 힌트 pill (하단) — 탭-투-셀렉트 안내
                                    if (!_retapBusy)
                                      Positioned(
                                        left: 0,
                                        right: 0,
                                        bottom: 10,
                                        child: Center(
                                          child: Container(
                                            padding:
                                                const EdgeInsets.symmetric(
                                                    horizontal: 11,
                                                    vertical: 5),
                                            decoration: BoxDecoration(
                                              color: kInkDeep
                                                  .withValues(alpha: 0.6),
                                              borderRadius:
                                                  BorderRadius.circular(999),
                                            ),
                                            child: const Text(
                                              '물건을 탭하면 그 물건만 분류해요',
                                              style: TextStyle(
                                                  color: kNeutral100,
                                                  fontSize: 11),
                                            ),
                                          ),
                                        ),
                                      ),
                                    // 되돌리기 (우상단) — 탭/후보선택 이전 결과로 복귀
                                    if (_viewStack.isNotEmpty && !_retapBusy)
                                      Positioned(
                                        right: 12,
                                        top: 12,
                                        child: Material(
                                          color: kInkDeep
                                              .withValues(alpha: 0.6),
                                          borderRadius:
                                              BorderRadius.circular(999),
                                          child: InkWell(
                                            borderRadius:
                                                BorderRadius.circular(999),
                                            onTap: _undo,
                                            child: const Padding(
                                              padding: EdgeInsets.symmetric(
                                                  horizontal: 11,
                                                  vertical: 6),
                                              child: Row(
                                                mainAxisSize:
                                                    MainAxisSize.min,
                                                children: [
                                                  Icon(Icons.undo,
                                                      size: 14,
                                                      color: kNeutral100),
                                                  SizedBox(width: 5),
                                                  Text('되돌리기',
                                                      style: TextStyle(
                                                          color: kNeutral100,
                                                          fontSize: 11,
                                                          fontWeight:
                                                              FontWeight
                                                                  .w600)),
                                                ],
                                              ),
                                            ),
                                          ),
                                        ),
                                      ),
                                  ],
                                ),
                              );
                            },
                          ),
                        ),
                      ),
                      const SizedBox(height: 20),

                      // 다중 물건 후보 카드 — 사진 속 물건들을 각각 분류해 제시
                      if (_objects != null &&
                          _objects!.isMultiObject &&
                          _error == null) ...[
                        _ObjectsCard(
                          objects: _objects!.objects,
                          selected: _selectedObject,
                          onSelect: _selectObject,
                        ),
                        const SizedBox(height: kSpaceL),
                      ],

                      if (_error != null)
                        _ErrorState(message: _error!, onRetry: () {
                          setState(() {
                            _error = null;
                            _classifyDone = false;
                            _regionsDone = false;
                          });
                          _classify();
                          _fetchRegions();
                        })
                      else if (_prediction != null)
                        _LoadedContent(
                          image: widget.image,
                          prediction: _prediction!,
                          quality: _quality,
                          regions: _regions,
                          // 물건 후보를 선택한 뒤엔 그 물건의 단일 결과가 주인공 —
                          // 다중 물건 안내는 최초(미선택) 상태에서만.
                          objectsMulti: (_objects?.isMultiObject ?? false) &&
                              _selectedObject == null,
                          sceneNote: _sceneNote,
                          regionInfo: _regionInfo,
                          regionSet: _regionSet,
                          isSmartCapture: widget.isSmartCapture,
                        ),
                    ],
                  ),
          ),
        ],
      ),
    );
  }
}


/// '오후 2:41' 형식 시각 라벨.
String _clockLabel(DateTime t) {
  final h12 = t.hour % 12 == 0 ? 12 : t.hour % 12;
  return '${t.hour >= 12 ? '오후' : '오전'} $h12:${t.minute.toString().padLeft(2, '0')}';
}


/// cover 표시좌표 변환 헬퍼 — 정규화 좌표 → view 픽셀 좌표.
Offset _coverToView(Offset norm, Size imgSize, Size view) {
  final scale = math.max(view.width / imgSize.width, view.height / imgSize.height);
  final dx = (imgSize.width * scale - view.width) / 2;
  final dy = (imgSize.height * scale - view.height) / 2;
  return Offset(
    norm.dx * imgSize.width * scale - dx,
    norm.dy * imgSize.height * scale - dy,
  );
}


/// 객체 후보 번호 마커 — bbox 중심에 번호 뱃지, 탭하면 해당 후보 선택.
class _ObjectMarker extends StatelessWidget {
  final int index;
  final ObjectCandidate candidate;
  final Size imgSize;
  final Size view;
  final bool selected;
  final VoidCallback onTap;
  const _ObjectMarker({
    required this.index,
    required this.candidate,
    required this.imgSize,
    required this.view,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final p = _coverToView(
        Offset(candidate.cx, candidate.cy), imgSize, view);
    final info = infoForWithRollup(candidate.displayClass,
        parentSlug: candidate.coarseClass);
    final color = info?.color ?? Colors.blueGrey;
    const r = 15.0;
    return Positioned(
      left: p.dx - r, top: p.dy - r,
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          width: r * 2, height: r * 2,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: selected ? color : color.withValues(alpha: 0.85),
            border: Border.all(
                color: Colors.white, width: selected ? 3 : 1.5),
            boxShadow: const [
              BoxShadow(color: Colors.black45, blurRadius: 5),
            ],
          ),
          alignment: Alignment.center,
          child: Text(
            '${index + 1}',
            style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.w800,
                fontSize: 14),
          ),
        ),
      ),
    );
  }
}


/// 다중 물건 후보 카드 — 장면에서 분리된 물건들을 나열, 탭하면 메인 결과 교체.
class _ObjectsCard extends StatelessWidget {
  final List<ObjectCandidate> objects;
  final int? selected;
  final void Function(int) onSelect;
  const _ObjectsCard({
    required this.objects,
    required this.selected,
    required this.onSelect,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.all(kSpaceM),
      decoration: BoxDecoration(
        color: cs.surfaceContainerLow,
        borderRadius: BorderRadius.circular(kRadiusLarge),
        border: Border.all(color: cs.outlineVariant),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.filter_center_focus, size: 18, color: cs.primary),
              const SizedBox(width: 6),
              Text(
                '사진 속 물건 ${objects.length}개',
                style: Theme.of(context)
                    .textTheme
                    .titleSmall
                    ?.copyWith(fontWeight: FontWeight.w700),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            '번호를 탭하면 그 물건의 분리배출 방법을 보여드려요',
            style: Theme.of(context)
                .textTheme
                .bodySmall
                ?.copyWith(color: cs.onSurfaceVariant),
          ),
          const SizedBox(height: kSpaceS),
          for (int i = 0; i < objects.length; i++)
            _ObjectTile(
              index: i,
              candidate: objects[i],
              selected: selected == i,
              onTap: () => onSelect(i),
            ),
        ],
      ),
    );
  }
}


class _ObjectTile extends StatelessWidget {
  final int index;
  final ObjectCandidate candidate;
  final bool selected;
  final VoidCallback onTap;
  const _ObjectTile({
    required this.index,
    required this.candidate,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final info = infoForWithRollup(candidate.displayClass,
        parentSlug: candidate.coarseClass);
    final color = info?.color ?? cs.primary;
    final isReject = candidate.displayLevel == 'reject';
    final name = isReject
        ? '분류 불확실'
        : (info?.displayName ?? candidate.displayClass);
    final conf = candidate.displayLevel == 'fine'
        ? candidate.fineConfidence
        : candidate.coarseConfidence;

    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(kRadiusSmall),
        onTap: onTap,
        child: Container(
          padding:
              const EdgeInsets.symmetric(horizontal: kSpaceS, vertical: kSpaceS),
          decoration: BoxDecoration(
            color: selected ? color.withValues(alpha: 0.10) : null,
            borderRadius: BorderRadius.circular(kRadiusSmall),
            border: selected
                ? Border.all(color: color.withValues(alpha: 0.5))
                : null,
          ),
          child: Row(
            children: [
              Container(
                width: 26, height: 26,
                decoration: BoxDecoration(
                    shape: BoxShape.circle, color: color),
                alignment: Alignment.center,
                child: Text('${index + 1}',
                    style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.w800,
                        fontSize: 13)),
              ),
              const SizedBox(width: kSpaceS),
              Icon(info?.icon ?? Icons.help_outline, size: 20, color: color),
              const SizedBox(width: 6),
              Expanded(
                child: Text(name,
                    style: const TextStyle(fontWeight: FontWeight.w600)),
              ),
              if (!isReject)
                Text('${(conf * 100).toStringAsFixed(0)}%',
                    style: TextStyle(
                        fontSize: 12.5,
                        fontWeight: FontWeight.w700,
                        color: cs.onSurfaceVariant)),
              if (selected) ...[
                const SizedBox(width: 6),
                Icon(Icons.check_circle, size: 18, color: color),
              ],
            ],
          ),
        ),
      ),
    );
  }
}


/// 탭 마커 — 사용자가 지목한 지점에 링 표시 (정규화 → cover 표시좌표 변환).
class _TapMarker extends StatelessWidget {
  final Offset norm;      // 원본 이미지 기준 정규화 좌표
  final Size imgSize;
  final Size view;
  const _TapMarker({required this.norm, required this.imgSize, required this.view});

  @override
  Widget build(BuildContext context) {
    final scale = math.max(view.width / imgSize.width, view.height / imgSize.height);
    final dx = (imgSize.width * scale - view.width) / 2;
    final dy = (imgSize.height * scale - view.height) / 2;
    final px = norm.dx * imgSize.width * scale - dx;
    final py = norm.dy * imgSize.height * scale - dy;
    const r = 18.0;
    return Positioned(
      left: px - r, top: py - r,
      child: IgnorePointer(
        child: Container(
          width: r * 2, height: r * 2,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            border: Border.all(color: Colors.white, width: 2.5),
            boxShadow: const [BoxShadow(color: Colors.black45, blurRadius: 6)],
          ),
          child: const Icon(Icons.center_focus_strong,
              color: Colors.white, size: 18),
        ),
      ),
    );
  }
}


/// 영역 뷰 — 원본 위에 영역별 빗금(서버 렌더)을 깐 오버레이 이미지를 표시하고,
/// 각 재질 영역 중심에 라벨 badge 를 Flutter 로 그림. 오버레이 없으면 원본만.
class _RegionsView extends StatelessWidget {
  final File image;
  final PredictionWithRegions? regions;
  const _RegionsView({required this.image, this.regions});

  @override
  Widget build(BuildContext context) {
    final r = regions;
    // 오버레이 없으면 원본 그대로
    if (r == null || !r.hasOverlay) {
      return Image.file(image, fit: BoxFit.cover);
    }

    final overlayBytes = base64Decode(r.overlayBase64!.split(',').last);

    return LayoutBuilder(
      builder: (ctx, constraints) {
        final w = constraints.maxWidth;
        final h = constraints.maxHeight;
        return Stack(
          fit: StackFit.expand,
          children: [
            // 원본 + 영역별 빗금 (서버에서 alpha-blend 렌더한 JPEG)
            Image.memory(overlayBytes, fit: BoxFit.cover),
            // 영역별 재질 라벨 badge — bbox 중심에 배치
            for (final region in r.regions)
              _RegionBadge(region: region, areaW: w, areaH: h),
          ],
        );
      },
    );
  }
}


/// 한 재질 영역의 라벨 badge — bbox 중심(상단)에 배치.
class _RegionBadge extends StatelessWidget {
  final MaterialRegion region;
  final double areaW;
  final double areaH;
  const _RegionBadge({
    required this.region,
    required this.areaW,
    required this.areaH,
  });

  @override
  Widget build(BuildContext context) {
    final info = infoFor(region.slug);
    final accent = info?.color ?? Theme.of(context).colorScheme.primary;
    // bbox 중심 x, 상단 y (BoxFit.cover 라 정확 매핑은 어려워 근사 배치).
    const badgeW = 116.0;
    final left = (region.cx * areaW - badgeW / 2).clamp(4.0, areaW - badgeW - 4);
    final top = (region.bboxNorm[1] * areaH).clamp(6.0, areaH - 36);

    return Positioned(
      left: left,
      top: top,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: accent,
          borderRadius: BorderRadius.circular(20),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.4),
              blurRadius: 6, offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(info?.icon ?? Icons.help_outline, color: Colors.white, size: 16),
            const SizedBox(width: 5),
            Text(
              info?.displayName ?? region.slug,
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                fontSize: 13,
              ),
            ),
          ],
        ),
      ),
    );
  }
}


/// 다중재질 안내 카드 — 확실히 다른 재질이 2개 이상 검출됐을 때.
/// 재질별 분리 배출을 안내. 위 오버레이의 빗금 색상과 라벨이 1:1 대응.
class _MultiMaterialCard extends StatelessWidget {
  final List<MaterialRegion> regions;
  const _MultiMaterialCard({required this.regions});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    // 같은 재질이 여러 영역에 걸쳐 잡힐 수 있음 → slug 별로 신뢰도 가장 높은 region 채택.
    // 그 후 신뢰도 내림차순 정렬 — 가장 확실한 재질이 맨 위.
    final byMaterial = <String, MaterialRegion>{};
    for (final r in regions) {
      final existing = byMaterial[r.slug];
      if (existing == null || r.avgConf > existing.avgConf) {
        byMaterial[r.slug] = r;
      }
    }
    final unique = byMaterial.values.toList()
      ..sort((a, b) => b.avgConf.compareTo(a.avgConf));

    return Container(
      padding: const EdgeInsets.all(kSpaceL),
      decoration: BoxDecoration(
        color: cs.tertiaryContainer.withValues(alpha: 0.35),
        borderRadius: BorderRadius.circular(kRadiusLarge),
        border: Border.all(color: cs.tertiary.withValues(alpha: 0.4)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.call_split_rounded, color: cs.tertiary, size: 22),
              const SizedBox(width: kSpaceS),
              Expanded(
                child: Text(
                  '재질이 여러 개 섞여 있어요',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                ),
              ),
            ],
          ),
          const SizedBox(height: kSpaceXS),
          Text(
            '아래 재질별로 분리해서 배출하면 더 정확하게 재활용돼요.',
            style: TextStyle(fontSize: 13, height: 1.4, color: cs.onSurfaceVariant),
          ),
          const SizedBox(height: kSpaceM),
          ...unique.map((region) => _MaterialMethodTile(
                region: region,
                info: infoFor(region.slug),
              )),
        ],
      ),
    );
  }
}


/// 다중재질 카드의 재질 1개 항목 — 이름·배출함 + 그 재질의 배출 방법(how_to)을 함께 제시.
/// 단일 재질만 안내하던 것을 재질별 방법까지 다중 제시하도록 확장.
class _MaterialMethodTile extends StatelessWidget {
  final MaterialRegion region;
  final WasteInfo? info;
  const _MaterialMethodTile({required this.region, this.info});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final accent = info?.color ?? cs.primary;
    final steps = info?.howTo ?? const <String>[];
    final bin = info?.bin ?? '';
    return Container(
      margin: const EdgeInsets.only(top: kSpaceS),
      padding: const EdgeInsets.all(kSpaceM),
      decoration: BoxDecoration(
        color: cs.surface.withValues(alpha: 0.6),
        borderRadius: BorderRadius.circular(kRadiusMedium),
        border: Border.all(color: accent.withValues(alpha: 0.35)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 헤더: 아이콘 + (이름 + 배출함 위치를 세로로). 배출함 텍스트가 길어도
          // 다음 줄로 자연스럽게 흘러내려 이름이 글자별로 깨지지 않음.
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                width: 30,
                height: 30,
                decoration: BoxDecoration(color: accent, shape: BoxShape.circle),
                child: Icon(
                  info?.icon ?? Icons.help_outline,
                  color: Colors.white, size: 17,
                ),
              ),
              const SizedBox(width: kSpaceM),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // 이름 | 신뢰도% — 가장 확실한 재질이 위에 정렬되어 있음(부모에서).
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.baseline,
                      textBaseline: TextBaseline.alphabetic,
                      children: [
                        Expanded(
                          child: Text(
                            info?.displayName ?? region.slug,
                            style: const TextStyle(
                              fontWeight: FontWeight.w700,
                              fontSize: 14,
                              height: 1.2,
                            ),
                          ),
                        ),
                        const SizedBox(width: 6),
                        Text(
                          '${(region.avgConf * 100).round()}%',
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w800,
                            color: accent,
                          ),
                        ),
                      ],
                    ),
                    if (bin.isNotEmpty) ...[
                      const SizedBox(height: 3),
                      Text(
                        bin,
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: accent,
                          height: 1.35,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ],
          ),
          if (steps.isNotEmpty) ...[
            const SizedBox(height: kSpaceS),
            ...steps.take(2).map((s) => Padding(
                  padding: const EdgeInsets.only(top: 3, left: 38),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(Icons.check, size: 14, color: cs.secondary),
                      const SizedBox(width: 6),
                      Expanded(
                        child: Text(
                          s,
                          style: const TextStyle(fontSize: 12, height: 1.4),
                        ),
                      ),
                    ],
                  ),
                )),
          ],
        ],
      ),
    );
  }
}


/// 분석 중 로딩 — 썸네일 스캔 오버레이 + 단계 스텝퍼 + 결과 미리 채우기.
/// 단계별 바: 업로드=확정형(실측 바이트 %), AI 분석=불확정형 shimmer.
/// 재질 분석 중 — 시안 16c(개정): 중앙 집중 로딩 — 회전 오라 링 + 상태 문구 + 점 3개.
class _AnalysisLoading extends StatefulWidget {
  final int uploadSent;
  final int uploadTotal;
  final bool preprocessDone;
  final bool classifyDone;
  final bool resultDone;
  final bool isSmartCapture;
  final VoidCallback onCancel;

  const _AnalysisLoading({
    required this.uploadSent,
    required this.uploadTotal,
    required this.preprocessDone,
    required this.classifyDone,
    required this.resultDone,
    required this.isSmartCapture,
    required this.onCancel,
  });

  @override
  State<_AnalysisLoading> createState() => _AnalysisLoadingState();
}

class _AnalysisLoadingState extends State<_AnalysisLoading>
    with TickerProviderStateMixin {
  /// 오라 링 회전 (3.2s) — 시안 ggAura.
  late final AnimationController _aura = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 3200),
  )..repeat();

  /// 문구 반짝임·아이콘 회전 (2.4s) — ggTextSweep / ggSpin.
  late final AnimationController _sweep = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 2400),
  )..repeat();

  /// 점 3개 펄스 (1.2s) — ggDotPulse.
  late final AnimationController _dots = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1200),
  )..repeat();

  /// 분석 중 순환 문구 — 실제 단계(업로드/분석/마무리)에 맞춰 선택.
  static const _analyzingWords = [
    '재질을 살펴보는 중',
    '표면 질감 확인 중',
    '모양을 맞춰보는 중',
    '색을 확인하는 중',
    '라벨을 찾아보는 중',
  ];
  Timer? _wordTimer;
  int _wordIdx = 0;

  @override
  void initState() {
    super.initState();
    _wordTimer = Timer.periodic(const Duration(milliseconds: 1800), (_) {
      if (mounted) setState(() => _wordIdx++);
    });
  }

  @override
  void dispose() {
    _wordTimer?.cancel();
    _aura.dispose();
    _sweep.dispose();
    _dots.dispose();
    super.dispose();
  }

  bool get _uploadDone =>
      widget.classifyDone ||
      (widget.uploadTotal > 0 && widget.uploadSent >= widget.uploadTotal);

  String get _statusWord {
    if (!_uploadDone) {
      final pct = widget.uploadTotal > 0
          ? (widget.uploadSent / widget.uploadTotal * 100).round()
          : null;
      return pct == null ? '사진 올리는 중' : '사진 올리는 중 $pct%';
    }
    if (widget.classifyDone && !widget.resultDone) return '거의 다 됐어요';
    return _analyzingWords[_wordIdx % _analyzingWords.length];
  }

  @override
  Widget build(BuildContext context) {
    const inner = kInkDeep2;
    return Column(
      children: [
        Expanded(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(28, 0, 28, 40),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // 오라 링 — 2.5px 회전 그라디언트 보더 + 내부 다크 원 + 아이콘
                AnimatedBuilder(
                  animation: _aura,
                  builder: (context, child) => Container(
                    width: 96,
                    height: 96,
                    padding: const EdgeInsets.all(2.5),
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      gradient: SweepGradient(
                        transform:
                            GradientRotation(_aura.value * 2 * math.pi),
                        colors: [
                          Colors.white.withValues(alpha: 0.08),
                          kAccent400,
                          Colors.white.withValues(alpha: 0.08),
                          kAccent600,
                          Colors.white.withValues(alpha: 0.08),
                        ],
                        stops: const [0.0, 0.28, 0.5, 0.75, 1.0],
                      ),
                    ),
                    child: child,
                  ),
                  child: Container(
                    decoration: const BoxDecoration(
                      color: inner,
                      shape: BoxShape.circle,
                    ),
                    child: const Center(
                      child: CustomPaint(
                        size: Size(38, 38),
                        painter: _BlobIconPainter(color: kAccent300),
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 26),
                Text(
                  widget.isSmartCapture
                      ? '캡처본 재질을 분석하고 있어요'
                      : '사진 속 재질을 분석하고 있어요',
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontSize: 23,
                    fontWeight: FontWeight.w600,
                    color: kNeutral100,
                  ),
                ),
                const SizedBox(height: 12),
                // 상태 문구 — 회전 아이콘 + 반짝이는 텍스트 + 펄스 점 3개
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    AnimatedBuilder(
                      animation: _sweep,
                      builder: (context, child) => Transform.rotate(
                        angle: _sweep.value * 2 * math.pi,
                        child: child,
                      ),
                      child: const Icon(Icons.wb_sunny_outlined,
                          size: 14, color: kAccent400),
                    ),
                    const SizedBox(width: 7),
                    AnimatedBuilder(
                      animation: _sweep,
                      builder: (context, child) => ShaderMask(
                        blendMode: BlendMode.srcIn,
                        shaderCallback: (rect) => LinearGradient(
                          begin: Alignment(-1 + _sweep.value * 4, 0),
                          end: Alignment(1 + _sweep.value * 4, 0),
                          colors: const [
                            kAccent400,
                            kAccent400,
                            kAccent200,
                            kAccent400,
                            kAccent400,
                          ],
                          stops: const [0.0, 0.38, 0.5, 0.62, 1.0],
                        ).createShader(rect),
                        child: child,
                      ),
                      child: AnimatedSwitcher(
                        duration: const Duration(milliseconds: 300),
                        child: Text(
                          _statusWord,
                          key: ValueKey(_statusWord),
                          style: const TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                            color: kAccent400,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 7),
                    AnimatedBuilder(
                      animation: _dots,
                      builder: (context, _) => Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          for (var i = 0; i < 3; i++) ...[
                            if (i > 0) const SizedBox(width: 3),
                            Opacity(
                              opacity: _dotOpacity(_dots.value, i),
                              child: Container(
                                width: 4,
                                height: 4,
                                decoration: const BoxDecoration(
                                  color: kAccent400,
                                  shape: BoxShape.circle,
                                ),
                              ),
                            ),
                          ],
                        ],
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
        Padding(
          padding: EdgeInsets.only(
            bottom: 30 + MediaQuery.viewPaddingOf(context).bottom,
          ),
          child: Material(
            color: Colors.transparent,
            borderRadius: BorderRadius.circular(999),
            child: InkWell(
              borderRadius: BorderRadius.circular(999),
              onTap: widget.onCancel,
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 22, vertical: 12),
                decoration: BoxDecoration(
                  border: Border.all(
                    color: Colors.white.withValues(alpha: 0.2),
                  ),
                  borderRadius: BorderRadius.circular(999),
                ),
                child: Text(
                  '분석 취소',
                  style: TextStyle(
                    fontSize: 13.5,
                    fontWeight: FontWeight.w600,
                    color: Colors.white.withValues(alpha: 0.6),
                  ),
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }

  /// ggDotPulse — 0/80/100% 0.25, 40% 1.0, 점마다 0.2s 지연.
  static double _dotOpacity(double t, int i) {
    final local = (t - i * (0.2 / 1.2)) % 1.0;
    final phase = local < 0 ? local + 1 : local;
    if (phase <= 0.4) return 0.25 + 0.75 * (phase / 0.4);
    if (phase <= 0.8) return 1.0 - 0.75 * ((phase - 0.4) / 0.4);
    return 0.25;
  }
}

/// 시안 16c 로딩 아이콘 — 둥근 꽃/톱니 형태(원호 8개) 스트로크.
class _BlobIconPainter extends CustomPainter {
  final Color color;
  const _BlobIconPainter({required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final k = size.width / 24;
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5 * k
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..color = color;
    final r = Radius.circular(3 * k);
    Offset p(double x, double y) => Offset(x * k, y * k);
    final path = Path()
      ..moveTo(12 * k, 3 * k)
      ..arcToPoint(p(9, 6), radius: r, clockwise: false)
      ..arcToPoint(p(6, 9), radius: r, clockwise: false)
      ..arcToPoint(p(6, 15), radius: r, clockwise: false)
      ..arcToPoint(p(9, 18), radius: r, clockwise: false)
      ..arcToPoint(p(15, 18), radius: r, clockwise: false)
      ..arcToPoint(p(18, 15), radius: r, clockwise: false)
      ..arcToPoint(p(18, 9), radius: r, clockwise: false)
      ..arcToPoint(p(15, 6), radius: r, clockwise: false)
      ..arcToPoint(p(12, 3), radius: r, clockwise: false)
      ..close();
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(_BlobIconPainter oldDelegate) =>
      oldDelegate.color != color;
}


class _ErrorState extends StatelessWidget {
  final String message;
  final VoidCallback onRetry;
  const _ErrorState({required this.message, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Card(
      color: cs.errorContainer,
      child: Padding(
        padding: const EdgeInsets.all(kSpaceL),
        child: Column(
          children: [
            Icon(Icons.error_outline, color: cs.onErrorContainer, size: 36),
            const SizedBox(height: kSpaceS),
            Text(message,
                textAlign: TextAlign.center,
                style: TextStyle(color: cs.onErrorContainer)),
            const SizedBox(height: kSpaceM),
            FilledButton.tonal(
              onPressed: onRetry,
              child: const Text('다시 시도'),
            ),
          ],
        ),
      ),
    );
  }
}


class _LoadedContent extends StatelessWidget {
  final File image;
  final Prediction prediction;
  final ImageQualityResult? quality;
  final PredictionWithRegions? regions;
  final bool objectsMulti;   // 물건 후보 ≥2 감지 (미선택 상태)
  final RegionInfo? regionInfo;  // 지역별 배출 규정 (설정 시)
  final bool regionSet;          // 지역 설정 여부 (규정 데이터가 없을 때 캡션 분기)
  final bool isSmartCapture;     // 다시 촬영하기 / 다시 선택하기 라벨
  final String? sceneNote;       // 장면 결과 vs 물건별 결과 불일치 안내
  const _LoadedContent({
    required this.image,
    required this.prediction,
    this.quality,
    this.regions,
    this.objectsMulti = false,
    this.regionInfo,
    this.regionSet = false,
    this.isSmartCapture = false,
    this.sceneNote,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final assessment = assessConfidence(prediction);
    // reject: (1) 신뢰도 부족 (top1 < 0.55 또는 entropy > 0.7), 또는
    //         (2) 모델이 명시적으로 non_object 라고 분류 (폐기물 아님 — 재촬영 신호)
    //         → 둘 다 "기타/분류 불가" 로 정직하게 결론.
    final isNonObject = prediction.predictedClass == 'non_object';
    // 계층 응답의 reject(대분류조차 불확실) 도 동일하게 처리
    final hierReject = prediction.hier?.isReject ?? false;
    final reject = assessment.shouldReject || isNonObject || hierReject;
    // 계층 응답이면 롤업 조회 — 세부 비활성 시 부모 대분류 카드로 안내
    final info = reject
        ? infoFor('etc')
        : (prediction.hier != null
            ? infoForWithRollup(prediction.predictedClass,
                parentSlug: prediction.hier!.coarseClass)
            : infoFor(prediction.predictedClass));
    final accent = info?.color ?? cs.primary;

    final hasQualityIssue = quality?.hasIssue ?? false;
    final isMulti = regions?.isMultiMaterial ?? false;
    // realMulti — 진짜 다중재질로 인정하려면 두 조건 모두:
    //   1. global top1 이 확신 영역 (reject 아님)
    //   2. 모든 region 의 avg_conf >= 0.60 (region 별로도 확신)
    // 둘 중 하나라도 약하면 spurious multi (손바닥·마우스 같은 confident-wrong) →
    // 다중재질 카드 대신 PredictionCard 또는 reject 카드로 표시.
    // avg_conf 임계 0.75 — Fix 1.5 의 0.60 이 confident-wrong (손바닥·마우스) 통과시켜서 강화.
    // 진짜 다중재질 (PET+라벨 등) 은 보통 region 별 0.80+ 라 false negative 적음.
    final regionsHighConf = isMulti &&
        regions!.regions.every((r) => r.avgConf >= 0.75);
    final realMulti = regionsHighConf && !reject;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // 사진 품질 문제 (어두움/흔들림) — 가장 먼저 안내
        if (hasQualityIssue) ...[
          _QualityBanner(quality: quality!),
          const SizedBox(height: kSpaceM),
        ],

        // 메인 결론 — 세 갈래.
        if (realMulti) ...[
          // (1) 진짜 다중재질 — global 도 확신 + regions 도 2+ → "여러 재질" 메시지 + CAM 버튼.
          //     단일 클래스로 단정하지 않고, 아래 재질별 breakdown 이 실제 결과 역할.
          //     PredictionCard·GuideCard 는 이 케이스에서 숨김(다중재질 안내와 중복).
          AnimatedEntry(
            child: _RejectCard(
              prediction: prediction,
              isMultiMaterial: true,
              image: image,
            ),
          ),
          const SizedBox(height: kSpaceM),
          AnimatedEntry(
            index: 1,
            child: _MultiMaterialCard(regions: regions!.regions),
          ),
        ] else if (reject && _topConfidentRegion(regions) != null) ...[
          // (2') 장면 단위론 불확실하지만 재질 영역 분석(CAM+검증 재분류)이
          //      확신하는 재질이 있는 경우 — 오버레이 배지와 결과 카드가
          //      어긋나던 불일치 해소: 영역 발견을 메인 답으로 승격.
          ...(() {
            final r = _topConfidentRegion(regions)!;
            final rInfo = infoForWithRollup(r.slug, parentSlug: kFineToCoarse[r.slug]);
            final rAccent = rInfo?.color ?? cs.primary;
            return <Widget>[
              AnimatedEntry(
                child: _RegionRescueBanner(region: r, info: rInfo),
              ),
              if (rInfo != null) ...[
                const SizedBox(height: kSpaceM),
                AnimatedEntry(
                  index: 1,
                  child: _GuideCard(
                    info: rInfo,
                    accent: rAccent,
                    regionInfo: regionInfo,
                    regionSet: regionSet,
                    coarse: kFineToCoarse[r.slug] ?? r.slug,
                  ),
                ),
              ],
            ];
          })(),
        ] else if (reject && objectsMulti) ...[
          // (2) 여러 물건이 혼재해 장면 단위 확신이 분산된 경우 —
          //     "분류 불가" 로 단정하지 않고 물건별 분류(위 후보 카드·마커)로 안내.
          //     장면 reject 는 물건이 하나인데 어렵다는 뜻일 때만 의미가 있음.
          AnimatedEntry(
            child: _RejectCard(prediction: prediction, isMultiObject: true),
          ),
        ] else if (reject) ...[
          // (3) 단일재질이지만 모델이 어느 클래스에도 확신 못 함 → etc reject.
          AnimatedEntry(child: _RejectCard(prediction: prediction)),
          if (info != null) ...[
            const SizedBox(height: kSpaceM),
            AnimatedEntry(
              index: 2,
              child: _GuideCard(
                info: info,
                accent: accent,
                regionInfo: regionInfo,
                regionSet: regionSet,
                coarse: prediction.hier?.coarseClass ?? prediction.predictedClass,
              ),
            ),
          ],
        ] else ...[
          // (3) 일반 분류 — 불확실 배너(있으면) → PredictionCard → 가이드.
          if (assessment.isUncertain) ...[
            AnimatedEntry(
              child: _UncertainBanner(prediction: prediction),
            ),
            const SizedBox(height: kSpaceM),
          ],
          AnimatedEntry(
            index: assessment.isUncertain ? 1 : 0,
            child: _PredictionCard(
              image: image,
              prediction: prediction, info: info, accent: accent,
              assessment: assessment,
            ),
          ),
          if (sceneNote != null) ...[
            const SizedBox(height: 8),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(Icons.info_outline,
                    size: 13, color: DsTokens.of(context).muted),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    sceneNote!,
                    style: TextStyle(
                        fontSize: 11.5,
                        height: 1.4,
                        color: DsTokens.of(context).muted),
                  ),
                ),
              ],
            ),
          ],
          if (info != null) ...[
            const SizedBox(height: kSpaceM),
            AnimatedEntry(
              index: 2,
              child: _GuideCard(
                info: info,
                accent: accent,
                regionInfo: regionInfo,
                regionSet: regionSet,
                coarse: prediction.hier?.coarseClass ?? prediction.predictedClass,
              ),
            ),
          ],
        ],

        // 시맨틱 증거 배지 — 서버가 분류에 실제로 융합한 단서 노출 (신뢰 UI).
        // 분리배출 마크·라벨 문구·형태(정체) 인식 (SEMANTIC_FUSION_PLAN Phase 3)
        if (prediction.evidence.isNotEmpty) ...[
          const SizedBox(height: kSpaceM),
          AnimatedEntry(
            index: 2,
            child: _EvidenceChips(evidence: prediction.evidence),
          ),
        ],


        const SizedBox(height: kSpaceM),

        // 피드백 — 결과가 정확했나요? (시안 16e)
        const SizedBox(height: 4),
        AnimatedEntry(
          index: 3,
          child: FeedbackCard(prediction: prediction, image: image),
        ),
        const SizedBox(height: 16),
        // 다시 촬영하기 — 결과 모달을 닫고 카메라(또는 갤러리 선택)로 복귀
        AnimatedEntry(
          index: 4,
          child: Material(
            color: DsTokens.of(context).surface,
            borderRadius: BorderRadius.circular(kRadiusMedium),
            child: InkWell(
              borderRadius: BorderRadius.circular(kRadiusMedium),
              onTap: () {
                Haptics.selection();
                Navigator.of(context).pop(false);
              },
              child: Container(
                height: 54,
                decoration: BoxDecoration(
                  border: Border.all(
                    color: DsTokens.of(context).accentSoft,
                  ),
                  borderRadius: BorderRadius.circular(kRadiusMedium),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      isSmartCapture
                          ? Icons.photo_camera_outlined
                          : Icons.image_outlined,
                      size: 17,
                      color: DsTokens.of(context).accentDeep,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      isSmartCapture ? '다시 촬영하기' : '다른 사진 선택하기',
                      style: TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w700,
                        color: DsTokens.of(context).accentDeep,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
        const SizedBox(height: kSpaceL),
      ],
    );
  }
}


/// reject 인데 재질 영역 분석이 확신하는 재질이 있으면 그 영역 반환.
/// (오버레이의 빗금 배지와 결과 카드 동기화 — 임계 0.6)
MaterialRegion? _topConfidentRegion(PredictionWithRegions? regions) {
  final rs = regions?.regions;
  if (rs == null || rs.isEmpty) return null;
  final sorted = [...rs]..sort((a, b) => b.avgConf.compareTo(a.avgConf));
  final top = sorted.first;
  return top.avgConf >= 0.6 ? top : null;
}


/// 영역 분석 발견을 메인 답으로 승격하는 배너.
class _RegionRescueBanner extends StatelessWidget {
  final MaterialRegion region;
  final WasteInfo? info;
  const _RegionRescueBanner({required this.region, required this.info});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final accent = info?.color ?? cs.primary;
    return Container(
      padding: const EdgeInsets.all(kSpaceL),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [accent.withValues(alpha: 0.18), accent.withValues(alpha: 0.05)],
          begin: Alignment.topLeft, end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(kRadiusLarge),
        border: Border.all(color: accent.withValues(alpha: 0.35)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            Container(
              width: 56, height: 56,
              decoration: BoxDecoration(color: accent, shape: BoxShape.circle),
              child: Icon(info?.icon ?? Icons.category, color: Colors.white, size: 30),
            ),
            const SizedBox(width: kSpaceM),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(info?.displayName ?? region.slug,
                      style: Theme.of(context).textTheme.headlineSmall
                          ?.copyWith(fontWeight: FontWeight.bold)),
                  const SizedBox(height: 2),
                  Text('재질 영역 분석으로 찾았어요 (${(region.avgConf * 100).toStringAsFixed(0)}%)',
                      style: TextStyle(
                          fontSize: 13, fontWeight: FontWeight.w700, color: accent)),
                ],
              ),
            ),
          ]),
          const SizedBox(height: kSpaceS),
          Text(
            '전체 장면으로는 확신이 낮았지만, 사진 속 빗금 영역에서 이 재질이 확인됐어요. '
            '다르다면 아래 피드백으로 알려주세요.',
            style: TextStyle(fontSize: 12.5, height: 1.5, color: cs.onSurfaceVariant),
          ),
        ],
      ),
    );
  }
}


/// 지역별 배출 안내 카드 — 지자체 조례 기준 (공공데이터포털 표준데이터).
class _EvidenceChips extends StatelessWidget {
  final List<EvidenceItem> evidence;
  const _EvidenceChips({required this.evidence});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Wrap(
      spacing: 6,
      runSpacing: 6,
      children: [
        for (final e in evidence)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: cs.secondaryContainer.withValues(alpha: 0.55),
              borderRadius: BorderRadius.circular(999),
              border: Border.all(color: cs.outlineVariant),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  switch (e.type) {
                    'mark' => Icons.recycling,
                    'identity' => Icons.visibility_outlined,
                    'vlm' => Icons.auto_awesome,
                    _ => Icons.notes,
                  },
                  size: 14,
                  color: cs.onSecondaryContainer,
                ),
                const SizedBox(width: 5),
                Text(
                  switch (e.type) {
                    'mark' => "분리배출 표시 '${e.token}' 인식",
                    'identity' => '형태 인식: ${e.token}',
                    'vlm' => 'AI 정밀 분석: ${e.token}',
                    _ => "라벨 문구 '${e.token}' 인식",
                  },
                  style: TextStyle(
                    fontSize: 11.5,
                    fontWeight: FontWeight.w600,
                    color: cs.onSecondaryContainer,
                  ),
                ),
              ],
            ),
          ),
      ],
    );
  }
}


class _PredictionCard extends StatelessWidget {
  final File image;
  final Prediction prediction;
  final WasteInfo? info;
  final Color accent;
  final ConfidenceAssessment assessment;
  const _PredictionCard({
    required this.image,
    required this.prediction,
    required this.info,
    required this.accent,
    required this.assessment,
  });

  /// 분석 주체 — 모델 이름으로 기기/서버 구분.
  String get _sourceLabel {
    if (isCloudFallback(prediction.modelArch)) return '클라우드 재확인';
    final arch = prediction.modelArch.toLowerCase();
    if (arch.contains('on-device') ||
        arch.contains('ondevice') ||
        arch.contains('local')) {
      return '기기에서 분석';
    }
    return '서버에서 분석';
  }

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          '분석된 재질',
          style: TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w700,
            letterSpacing: 0.44,
            color: t.muted,
          ),
        ),
        const SizedBox(height: 4),
        Wrap(
          crossAxisAlignment: WrapCrossAlignment.center,
          spacing: 10,
          runSpacing: 6,
          children: [
            Text(
              info?.displayName ?? prediction.predictedClass,
              style: const TextStyle(
                fontSize: 38,
                fontWeight: FontWeight.w600,
                letterSpacing: -0.38,
                height: 1.1,
              ),
            ),
            DsCard(
              tinted: true,
              radius: 999,
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
              child: Text(
                '확신 ${(prediction.confidence * 100).round()}% · $_sourceLabel',
                style: TextStyle(
                  fontSize: 11.5,
                  fontWeight: FontWeight.w600,
                  color: t.accentChipText,
                ),
              ),
            ),
          ],
        ),
        // 계층 경로 배지 — 대분류(항상) → 세부(확신 시)
        if (prediction.hier != null) ...[
          const SizedBox(height: 10),
          HierBadge(hier: prediction.hier!),
        ],
        const SizedBox(height: 12),
        _ExplainButton(
            image: image, accent: accent, info: info, prediction: prediction),
      ],
    );
  }
}


/// 사진 품질 문제 (어두움/흔들림) 안내 배너.
class _QualityBanner extends StatelessWidget {
  final ImageQualityResult quality;
  const _QualityBanner({required this.quality});

  @override
  Widget build(BuildContext context) {
    final amber = const Color(0xFFF9A825);
    final messages = <String>[];
    if (quality.issues.contains(ImageQualityIssue.tooDark)) {
      messages.add('사진이 어두워요 — 밝은 곳에서 다시 찍어보세요');
    }
    if (quality.issues.contains(ImageQualityIssue.tooBlurry)) {
      messages.add('흔들렸거나 초점이 안 맞아요 — 잠시 멈춰서 다시 찍어보세요');
    }
    return Container(
      padding: const EdgeInsets.all(kSpaceM),
      decoration: BoxDecoration(
        color: amber.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(kRadiusMedium),
        border: Border.all(color: amber.withValues(alpha: 0.4)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.wb_incandescent_outlined, color: amber, size: 20),
          const SizedBox(width: kSpaceS),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '사진 품질 안내',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: amber.withValues(alpha: 0.9),
                  ),
                ),
                const SizedBox(height: 2),
                ...messages.map((m) => Padding(
                      padding: const EdgeInsets.only(top: 2),
                      child: Text('• $m', style: const TextStyle(fontSize: 13, height: 1.4)),
                    )),
              ],
            ),
          ),
        ],
      ),
    );
  }
}


/// 모델이 확신하지 못할 때 결과 위에 표시되는 정직한 안내 배너.
class _UncertainBanner extends StatelessWidget {
  final Prediction prediction;
  const _UncertainBanner({required this.prediction});

  @override
  Widget build(BuildContext context) {
    final warn = const Color(0xFFD32F2F);
    return Container(
      padding: const EdgeInsets.all(kSpaceL),
      decoration: BoxDecoration(
        color: warn.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(kRadiusLarge),
        border: Border.all(color: warn.withValues(alpha: 0.35)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.help_outline, color: warn, size: 22),
              const SizedBox(width: kSpaceS),
              Expanded(
                child: Text(
                  '확실하지 않아요',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                        color: warn,
                      ),
                ),
              ),
            ],
          ),
          const SizedBox(height: kSpaceS),
          const Text(
            '이 사진은 분류하기 어려워요. 더 정확한 결과를 위해:',
            style: TextStyle(height: 1.4),
          ),
          const SizedBox(height: kSpaceXS),
          const Text(
            '• 물체 하나만 화면 가운데에 담아주세요\n'
            '• 밝은 곳에서 가까이 찍어주세요\n'
            '• 아래 추측이 틀렸다면 피드백으로 알려주세요',
            style: TextStyle(fontSize: 13, height: 1.5),
          ),
          const SizedBox(height: kSpaceXS),
          Text(
            '아래는 모델의 가장 가능성 높은 추측입니다 (참고용).',
            style: TextStyle(
              fontSize: 12,
              fontStyle: FontStyle.italic,
              color: Theme.of(context).colorScheme.onSurfaceVariant,
            ),
          ),
        ],
      ),
    );
  }
}


/// 분류 불가(reject) 결론 카드.
///
/// 두 가지 변형:
///  - 일반: 모델이 어느 클래스에도 확신 못 할 때 → "기타/분류 불가" + 모델 추측 참고.
///  - 다중재질(isMultiMaterial=true): 여러 재질이 섞여 단일 분류가 어려운 경우
///    → "여러 재질이 섞여 있어요" + CAM 버튼. 등록(etc)로 단정하지 않고
///    아래 재질별 안내(_MultiMaterialCard)로 안내함. (image 제공 시 CAM 표시)
class _RejectCard extends StatelessWidget {
  final Prediction prediction;
  final bool isMultiMaterial;
  final bool isMultiObject;
  final File? image;
  const _RejectCard({
    required this.prediction,
    this.isMultiMaterial = false,
    this.isMultiObject = false,
    this.image,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final etc = infoFor('etc');
    final accent = isMultiObject
        ? Theme.of(context).colorScheme.primary
        : (etc?.color ?? const Color(0xFF9E9E9E));

    final title = isMultiObject
        ? '여러 물건이 보여요'
        : isMultiMaterial
            ? '여러 재질이 섞여 있어요'
            : (etc?.displayName ?? '기타 / 분류 불가');
    final subtitle = isMultiObject
        ? '물건을 골라 확인해주세요'
        : isMultiMaterial
            ? '단일 분류가 어려워요'
            : '자동 분류가 어려워요';
    final icon = isMultiObject
        ? Icons.filter_center_focus
        : isMultiMaterial
            ? Icons.call_split_rounded
            : (etc?.icon ?? Icons.help_outline);
    final body = isMultiObject
        ? '사진에 물건이 여러 개 감지돼서 하나로 분류하지 않았어요. '
          '위 물건 목록에서 번호를 선택하거나, 사진 속 물건을 직접 탭하면 '
          '각각의 분리배출 방법을 알려드려요.'
        : isMultiMaterial
            ? '이 사진은 여러 재질이 섞여 있어서 한 가지로 분류하기 어려워요. '
              '아래 재질별 안내를 따라 분리해서 배출해주세요.'
            : '이 물건은 확실하게 분류하기 어려워요. 재질을 직접 확인해 배출하거나, '
              '아래에서 올바른 분류를 알려주시면 다음 학습에 반영돼요.';
    final guess =
        infoFor(prediction.predictedClass)?.displayName ?? prediction.predictedClass;

    return Container(
      padding: const EdgeInsets.all(kSpaceL),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [accent.withValues(alpha: 0.18), accent.withValues(alpha: 0.05)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(kRadiusLarge),
        border: Border.all(color: accent.withValues(alpha: 0.35)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Container(
                width: 64,
                height: 64,
                decoration: BoxDecoration(color: accent, shape: BoxShape.circle),
                child: Icon(icon, color: Colors.white, size: 34),
              ),
              const SizedBox(width: kSpaceM),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                    ),
                    const SizedBox(height: 4),
                    Text(subtitle,
                        style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w700,
                            color: accent)),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: kSpaceM),
          Text(
            body,
            style: TextStyle(fontSize: 13, height: 1.5, color: cs.onSurfaceVariant),
          ),
          // 모델 추측 — 일반 reject 만 표시(멀티 케이스는 물건/재질별 % 가 그 역할).
          if (!isMultiMaterial && !isMultiObject) ...[
            const SizedBox(height: kSpaceS),
            Text(
              '모델 추측: $guess ${(prediction.confidence * 100).toStringAsFixed(0)}% (참고용)',
              style: TextStyle(
                  fontSize: 12, fontStyle: FontStyle.italic, color: cs.onSurfaceVariant),
            ),
          ],
          // CAM "왜 이렇게 분류했어?" — 멀티 케이스에서 image 가 주어지면 표시.
          if (isMultiMaterial && image != null) ...[
            const SizedBox(height: kSpaceM),
            _ExplainButton(
              image: image!,
              accent: accent,
              info: etc,
              prediction: prediction,
            ),
          ],
        ],
      ),
    );
  }
}


/// "왜 이렇게 분류했어?" — 서버에 /predict-with-cam 호출 → heatmap dialog.
class _ExplainButton extends StatefulWidget {
  final File image;
  final Color accent;
  final WasteInfo? info;
  final Prediction prediction;
  const _ExplainButton({
    required this.image,
    required this.accent,
    required this.info,
    required this.prediction,
  });

  @override
  State<_ExplainButton> createState() => _ExplainButtonState();
}

class _ExplainButtonState extends State<_ExplainButton> {
  bool _loading = false;

  Future<void> _onTap() async {
    if (_loading) return;
    setState(() => _loading = true);
    Haptics.selection();

    try {
      final client = await AppScope.api();
      final result = await client.predictWithCam(widget.image);
      if (!mounted) return;
      if (!result.camAvailable || result.camBase64 == null) {
        _showInfoDialog(
          title: 'CAM 미지원',
          message: '지금 서버 모델에서는 판단 근거 이미지를 만들 수 없어요.\n'
              '다음 업데이트에서 지원할 예정이에요.',
        );
        return;
      }
      _showCamDialog(result);
    } on ApiException catch (e) {
      if (!mounted) return;
      if (e.statusCode == 404) {
        _showInfoDialog(
          title: '아직 준비 중이에요',
          message: '판단 근거 보기는 다음 업데이트에서 지원할 예정이에요.',
        );
      } else {
        _showInfoDialog(title: '설명을 만들지 못했어요', message: friendlyError(e));
      }
    } catch (e) {
      if (!mounted) return;
      _showInfoDialog(
        title: '설명을 만들지 못했어요',
        message: friendlyError(e),
      );
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _showInfoDialog({required String title, required String message}) {
    showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        icon: const Icon(Icons.info_outline, size: 32),
        title: Text(title),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: const Text('확인'),
          ),
        ],
      ),
    );
  }

  void _showCamDialog(PredictionWithCam result) {
    final imageBytes = base64Decode(
      result.camBase64!.split(',').last,
    );
    final accent = widget.accent;
    final cs = Theme.of(context).colorScheme;
    showDialog<void>(
      context: context,
      barrierColor: Colors.black87,
      builder: (ctx) => Dialog(
        insetPadding: const EdgeInsets.all(kSpaceM),
        backgroundColor: Theme.of(ctx).scaffoldBackgroundColor,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(kRadiusLarge),
        ),
        child: Padding(
          padding: const EdgeInsets.all(kSpaceL),
          child: SingleChildScrollView(
            child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Row(
                children: [
                  Icon(Icons.visibility_outlined, color: accent),
                  const SizedBox(width: kSpaceS),
                  Expanded(
                    child: Text(
                      '모델이 본 영역',
                      style: Theme.of(ctx).textTheme.titleMedium,
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close),
                    onPressed: () => Navigator.of(ctx).pop(),
                  ),
                ],
              ),
              const SizedBox(height: kSpaceS),
              Text(
                '"${widget.info?.displayName ?? widget.prediction.predictedClass}" '
                '으로 분류할 때 모델이 가장 집중한 영역입니다 '
                '(빨강 = 강하게 봄, 파랑 = 거의 안 봄).',
                style: TextStyle(fontSize: 12, color: cs.onSurfaceVariant),
              ),
              const SizedBox(height: kSpaceM),
              ClipRRect(
                borderRadius: BorderRadius.circular(kRadiusMedium),
                child: Image.memory(
                  Uint8List.fromList(imageBytes),
                  fit: BoxFit.contain,
                ),
              ),
              const SizedBox(height: kSpaceM),
              Container(
                padding: const EdgeInsets.all(kSpaceM),
                decoration: BoxDecoration(
                  color: cs.surfaceContainerHighest.withValues(alpha: 0.5),
                  borderRadius: BorderRadius.circular(kRadiusMedium),
                ),
                child: Row(
                  children: [
                    Icon(Icons.lightbulb_outline, size: 16, color: cs.tertiary),
                    const SizedBox(width: kSpaceS),
                    Expanded(
                      child: Text(
                        '예상한 영역과 다르다면 결과가 틀렸을 가능성이 높아요. '
                        '아래 피드백 카드에서 정정해주세요.',
                        style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return OutlinedButton.icon(
      onPressed: _loading ? null : _onTap,
      icon: _loading
          ? const SizedBox(
              width: 14, height: 14,
              child: CircularProgressIndicator(strokeWidth: 2),
            )
          : const Icon(Icons.visibility_outlined, size: 18),
      label: Text(_loading ? '분석 중...' : '왜 이렇게 분류했어?'),
      style: OutlinedButton.styleFrom(
        foregroundColor: widget.accent,
        side: BorderSide(color: widget.accent.withValues(alpha: 0.4)),
        padding: const EdgeInsets.symmetric(horizontal: kSpaceM, vertical: kSpaceS),
      ),
    );
  }
}


/// 온디바이스 신뢰도가 낮아서 cloud 가 재검증한 경우 표시되는 작은 배지.
/// 이렇게 버려요 — 전국 공통 요령 + (지역 설정 시) 우리 동네 조례 규정 통합.
class _GuideCard extends StatelessWidget {
  final WasteInfo info;
  final Color accent;
  final RegionInfo? regionInfo;
  final bool regionSet;
  final String coarse;
  const _GuideCard({
    required this.info,
    required this.accent,
    required this.regionInfo,
    required this.regionSet,
    required this.coarse,
  });

  static const _recyclables = {
    'paper', 'paper_pack', 'glass', 'metal', 'plastic', 'vinyl',
    'styrofoam', 'clothes',
  };

  /// 지역 규정 행 — 재질을 재활용/음식물/일반으로 매핑해 해당 분류의 규정 추출.
  List<(String, String)> _regionRows(RegionRule r) {
    final String? method;
    final String? days;
    if (coarse == 'food_waste') {
      method = r.methodFood;
      days = r.daysFood;
    } else if (_recyclables.contains(coarse)) {
      method = r.methodRecycle;
      days = r.daysRecycle;
    } else {
      method = r.methodGeneral;
      days = r.daysGeneral;
    }
    return [
      if (method != null && method.isNotEmpty) ('배출 방법', method),
      if (days != null && days.isNotEmpty) ('배출 요일', days),
      if (r.emitTime != null && r.emitTime!.isNotEmpty) ('배출 시간', r.emitTime!),
      if (r.noCollectDay != null && r.noCollectDay!.isNotEmpty)
        ('미수거일', r.noCollectDay!),
    ];
  }

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    final rule = regionInfo?.representative;
    final regionRows = rule == null ? const <(String, String)>[] : _regionRows(rule);
    final hasRegion = regionRows.isNotEmpty;
    final steps = [
      if (info.howTo.isEmpty && info.bin.isNotEmpty) info.bin,
      ...info.howTo.take(3),
    ];
    final bodyColor = t.body;

    return DsCard(
      radius: 20,
      padding: const EdgeInsets.fromLTRB(20, 18, 20, 18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Expanded(
                child: Text(
                  '이렇게 버려요',
                  style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700),
                ),
              ),
              if (hasRegion)
                // 지역 조례 기준 배지
                Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 9, vertical: 4),
                  decoration: BoxDecoration(
                    color: t.accentChipBg,
                    border: Border.all(
                      color: t.accentChipBorder,
                    ),
                    borderRadius: BorderRadius.circular(999),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.place_outlined,
                          size: 11, color: t.accentChipText),
                      const SizedBox(width: 4),
                      Text(
                        '${regionInfo!.sigungu} 조례 기준',
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                          color: t.accentChipText,
                        ),
                      ),
                    ],
                  ),
                )
              else if (info.bin.isNotEmpty && info.howTo.isNotEmpty)
                Flexible(
                  child: Text(
                    info.bin,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(fontSize: 11.5, color: t.muted),
                  ),
                ),
            ],
          ),
          // 우리 동네 규정 — 배출 방법·요일·시간·미수거일 (공공데이터, 조례 기준)
          if (hasRegion) ...[
            const SizedBox(height: 12),
            for (final (i, (label, value)) in regionRows.indexed) ...[
              if (i > 0) const SizedBox(height: 6),
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SizedBox(
                    width: 64,
                    child: Text(
                      label,
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                        color: t.accentChipText,
                      ),
                    ),
                  ),
                  Expanded(
                    child: Text(
                      value,
                      style: TextStyle(
                          fontSize: 13, height: 1.5, color: bodyColor),
                    ),
                  ),
                ],
              ),
            ],
            const SizedBox(height: 12),
            Container(height: 1, color: t.border),
            const SizedBox(height: 12),
            Text(
              '분리배출 요령',
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.44,
                color: t.muted,
              ),
            ),
            const SizedBox(height: 8),
          ] else
            const SizedBox(height: 12),
          // 전국 공통 분리배출 요령
          for (final (i, s) in steps.indexed) ...[
            if (i > 0) const SizedBox(height: 10),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Padding(
                  padding: const EdgeInsets.only(top: 1),
                  child: Icon(Icons.check, size: 16, color: t.accentStrong),
                ),
                const SizedBox(width: 9),
                Expanded(
                  child: Text(
                    s,
                    style: TextStyle(
                        fontSize: 13, height: 1.5, color: bodyColor),
                  ),
                ),
              ],
            ),
          ],
          const SizedBox(height: 12),
          Text(
            hasRegion
                ? '출처 · ${regionInfo!.sido} ${regionInfo!.sigungu} 폐기물 관리 조례 '
                    '(행안부 생활쓰레기 배출정보 표준데이터) · 관리구역에 따라 다를 수 있어요'
                : regionSet
                    ? '전국 공통 안내 · 우리 동네 조례 데이터는 아직 준비 중이에요'
                    : '전국 공통 안내 · 지역을 설정하면 우리 동네 조례 기준도 함께 알려드려요',
            style: TextStyle(fontSize: 10.5, height: 1.4, color: t.muted),
          ),
        ],
      ),
    );
  }
}


/// 되돌리기 스냅샷 — 탭/후보선택으로 결과가 교체되기 전의 뷰 상태.
class _ViewSnapshot {
  final Prediction prediction;
  final int? selectedObject;
  final Offset? tapNorm;
  final PredictObjects? objects;         // 다중 분류 화면 복귀용
  final PredictionWithRegions? regions;  // 빗금 오버레이 복귀용
  const _ViewSnapshot({
    required this.prediction,
    this.selectedObject,
    this.tapNorm,
    this.objects,
    this.regions,
  });
}
