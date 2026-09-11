/// 캡처된 사진의 분류 결과를 모달로 표시.
/// LiveCameraScreen 에서 사용.
library;

import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';

import '../../api/models.dart';
import '../../core/feedback/app_snackbar.dart';
import '../../data/haptics.dart';
import '../../data/image_quality.dart';
import '../../theme/app_theme.dart';
import '../../theme/design_tokens.dart';
import 'result_controller.dart';
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
  UploadMeta? meta,
  ImageQualityResult? initialQuality,
}) {
  return showModalBottomSheet<bool>(
    context: context,
    isScrollControlled: true,
    useSafeArea: true,
    enableDrag: false, // 풀스크린 — 드래그 dismiss 비활성
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
        meta: meta,
        initialQuality: initialQuality,
      ),
    ),
  );
}

class _ResultModal extends StatefulWidget {
  final File image;
  final ScrollController scrollController;
  final bool isSmartCapture;
  final UploadMeta? meta;
  final ImageQualityResult? initialQuality;
  const _ResultModal({
    required this.image,
    required this.scrollController,
    this.isSmartCapture = false,
    this.meta,
    this.initialQuality,
  });

  @override
  State<_ResultModal> createState() => _ResultModalState();
}

class _ResultModalState extends State<_ResultModal> {
  late final ResultController c = ResultController(
    image: widget.image,
    isSmartCapture: widget.isSmartCapture,
    meta: widget.meta,
    initialQuality: widget.initialQuality,
  );

  @override
  void initState() {
    super.initState();
    c.start();
  }

  @override
  void dispose() {
    c.dispose();
    super.dispose();
  }

  /// 이미지 위 탭 → 그 지점의 물건만 재분류. 원본 크기 미해석이면 안내 + 재시도.
  Future<void> _onImageTap(Offset local, Size view) async {
    if (c.retapBusy || c.loading) return;
    if (c.imgSize == null) {
      // 크기 미해석 — 조용히 무시하지 않고 피드백 + 재시도
      c.resolveImageSize();
      showAppSnackBar(
        context,
        '사진 정보를 준비 중이에요 — 잠시 후 다시 탭해주세요',
        duration: const Duration(seconds: 2),
      );
      return;
    }
    final norm = c.tapToNorm(local, view);
    if (norm == null) return;
    try {
      await c.reclassifyAt(norm.dx, norm.dy);
    } on Exception catch (e) {
      if (!mounted) return;
      showAppErrorSnackBar(context, e);
    }
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: c,
      builder: (context, _) => _buildBody(context),
    );
  }

  Widget _buildBody(BuildContext context) {
    final t = DsTokens.of(context);
    final loading = c.loading;

    return AnimatedContainer(
      duration: const Duration(milliseconds: 250),
      color: loading ? kInkDeep : Theme.of(context).scaffoldBackgroundColor,
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
                    uploadSent: c.uploadSent,
                    uploadTotal: c.uploadTotal,
                    preprocessDone: c.preprocessDone,
                    classifyDone: c.classifyDone,
                    resultDone: c.regionsDone,
                    isSmartCapture: widget.isSmartCapture,
                    onCancel: () {
                      Haptics.selection();
                      Navigator.of(context).pop(false); // 다시 촬영
                    },
                  )
                : ListView(
                    controller: widget.scrollController,
                    // 하단 인셋(홈 인디케이터)만큼 더 띄움 — 모달 시트는 useSafeArea 여도
                    // bottom 을 비워 두지 않아 마지막 버튼이 제스처 영역과 겹쳤음
                    padding: EdgeInsets.fromLTRB(
                      20,
                      6,
                      20,
                      30 + MediaQuery.viewPaddingOf(context).bottom,
                    ),
                    children: [
                      // 분석한 사진 + 영역별 빗금 오버레이 + 재질 라벨
                      // 탭-투-셀렉트: 물건을 탭하면 그 객체만 재분류
                      ClipRRect(
                        borderRadius: BorderRadius.circular(24),
                        child: AspectRatio(
                          aspectRatio: 16 / 10,
                          child: LayoutBuilder(
                            builder: (ctx, constraints) {
                              final view = Size(
                                constraints.maxWidth,
                                constraints.maxHeight,
                              );
                              return GestureDetector(
                                behavior: HitTestBehavior.opaque,
                                onTapUp: (d) =>
                                    _onImageTap(d.localPosition, view),
                                child: Stack(
                                  fit: StackFit.expand,
                                  children: [
                                    RegionsView(
                                      image: widget.image,
                                      regions: c.regions,
                                    ),
                                    if (c.lastTapNorm != null &&
                                        c.imgSize != null)
                                      TapMarker(
                                        norm: c.lastTapNorm!,
                                        imgSize: c.imgSize!,
                                        view: view,
                                      ),
                                    if (c.objects != null &&
                                        c.objects!.isMultiObject &&
                                        c.imgSize != null)
                                      for (
                                        int i = 0;
                                        i < c.objects!.objects.length;
                                        i++
                                      )
                                        ObjectMarker(
                                          index: i,
                                          candidate: c.objects!.objects[i],
                                          imgSize: c.imgSize!,
                                          view: view,
                                          selected: c.selectedObject == i,
                                          onTap: () => c.selectObject(i),
                                        ),
                                    if (c.retapBusy)
                                      Container(
                                        color: Colors.black38,
                                        child: const Center(
                                          child: CircularProgressIndicator(
                                            color: Colors.white,
                                          ),
                                        ),
                                      ),
                                    // 출처 pill — 스마트 촬영 · 시각 (시안 16e)
                                    Positioned(
                                      left: 12,
                                      top: 12,
                                      child: Container(
                                        padding: const EdgeInsets.symmetric(
                                          horizontal: 11,
                                          vertical: 6,
                                        ),
                                        decoration: BoxDecoration(
                                          color: kInkDeep.withValues(
                                            alpha: 0.6,
                                          ),
                                          borderRadius: BorderRadius.circular(
                                            999,
                                          ),
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
                                              '${widget.isSmartCapture ? '스마트 촬영' : '갤러리'} · ${c.capturedAt}',
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
                                    if (!c.retapBusy)
                                      Positioned(
                                        left: 0,
                                        right: 0,
                                        bottom: 10,
                                        child: Center(
                                          child: Container(
                                            padding: const EdgeInsets.symmetric(
                                              horizontal: 11,
                                              vertical: 5,
                                            ),
                                            decoration: BoxDecoration(
                                              color: kInkDeep.withValues(
                                                alpha: 0.6,
                                              ),
                                              borderRadius:
                                                  BorderRadius.circular(999),
                                            ),
                                            child: const Text(
                                              '물건을 탭하면 그 물건만 분류해요',
                                              style: TextStyle(
                                                color: kNeutral100,
                                                fontSize: 11,
                                              ),
                                            ),
                                          ),
                                        ),
                                      ),
                                    // 되돌리기 (우상단) — 탭/후보선택 이전 결과로 복귀
                                    if (c.canUndo && !c.retapBusy)
                                      Positioned(
                                        right: 12,
                                        top: 12,
                                        child: Material(
                                          color: kInkDeep.withValues(
                                            alpha: 0.6,
                                          ),
                                          borderRadius: BorderRadius.circular(
                                            999,
                                          ),
                                          child: InkWell(
                                            borderRadius: BorderRadius.circular(
                                              999,
                                            ),
                                            onTap: c.undo,
                                            child: const Padding(
                                              padding: EdgeInsets.symmetric(
                                                horizontal: 11,
                                                vertical: 6,
                                              ),
                                              child: Row(
                                                mainAxisSize: MainAxisSize.min,
                                                children: [
                                                  Icon(
                                                    Icons.undo,
                                                    size: 14,
                                                    color: kNeutral100,
                                                  ),
                                                  SizedBox(width: 5),
                                                  Text(
                                                    '되돌리기',
                                                    style: TextStyle(
                                                      color: kNeutral100,
                                                      fontSize: 11,
                                                      fontWeight:
                                                          FontWeight.w600,
                                                    ),
                                                  ),
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
                      if (c.objects != null &&
                          c.objects!.isMultiObject &&
                          c.error == null) ...[
                        ObjectsCard(
                          objects: c.objects!.objects,
                          selected: c.selectedObject,
                          onSelect: c.selectObject,
                        ),
                        const SizedBox(height: kSpaceL),
                      ],

                      if (c.error != null)
                        ResultErrorState(message: c.error!, onRetry: c.retry)
                      else if (c.prediction != null)
                        ResultLoadedContent(
                          image: widget.image,
                          prediction: c.prediction!,
                          quality: c.quality,
                          regions: c.regions,
                          // 물건 후보를 선택한 뒤엔 그 물건의 단일 결과가 주인공 —
                          // 다중 물건 안내는 최초(미선택) 상태에서만.
                          objectsMulti:
                              (c.objects?.isMultiObject ?? false) &&
                              c.selectedObject == null,
                          sceneNote: c.sceneNote,
                          regionInfo: c.regionInfo,
                          regionSet: c.regionSet,
                          isSmartCapture: widget.isSmartCapture,
                          tapNorm: c.lastTapNorm,
                        ),
                    ],
                  ),
          ),
        ],
      ),
    );
  }
}
