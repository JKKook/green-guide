/// 캡처된 사진의 분류 결과를 모달로 표시.
/// LiveCameraScreen 에서 사용.
library;

import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../../api/api_client.dart';
import '../../api/models.dart';
import '../../core/di/app_scope.dart';
import '../../core/feedback/app_snackbar.dart';
import '../../data/haptics.dart';
import '../../data/image_quality.dart';
import '../../theme/app_theme.dart';
import '../../theme/design_tokens.dart';
import 'widgets/analysis_loading.dart';
import 'widgets/object_markers.dart';
import 'widgets/objects_card.dart';
import 'widgets/regions_view.dart';
import 'widgets/result_error_state.dart';
import 'widgets/result_loaded_content.dart';

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
                ? AnalysisLoading(
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
                                    RegionsView(
                                      image: widget.image,
                                      regions: _regions,
                                    ),
                                    if (_lastTapNorm != null &&
                                        _imgSize != null)
                                      TapMarker(
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
                                        ObjectMarker(
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
                        ObjectsCard(
                          objects: _objects!.objects,
                          selected: _selectedObject,
                          onSelect: _selectObject,
                        ),
                        const SizedBox(height: kSpaceL),
                      ],

                      if (_error != null)
                        ResultErrorState(message: _error!, onRetry: () {
                          setState(() {
                            _error = null;
                            _classifyDone = false;
                            _regionsDone = false;
                          });
                          _classify();
                          _fetchRegions();
                        })
                      else if (_prediction != null)
                        ResultLoadedContent(
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
