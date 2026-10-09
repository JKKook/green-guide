/// "왜 이렇게 판단했나요" — Grad-CAM 설명 버튼.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../../../api/api_client.dart';
import '../../../api/models.dart';
import '../../../core/di/app_scope.dart';
import '../../../data/haptics.dart';
import '../../../data/waste_info.dart';
import '../../../theme/app_theme.dart';

/// "왜 이렇게 분류했어?" — 서버에 /predict-with-cam 호출 → heatmap dialog.
class ExplainButton extends StatefulWidget {
  final File image;
  final Color accent;
  final WasteInfo? info;
  final Prediction prediction;

  /// 결과를 만들 때 쓴 탭 좌표(정규화). 있으면 같은 크롭으로 CAM 을 만든다.
  final Offset? tapNorm;
  const ExplainButton({
    super.key,
    required this.image,
    required this.accent,
    required this.info,
    required this.prediction,
    this.tapNorm,
  });

  @override
  State<ExplainButton> createState() => _ExplainButtonState();
}

class _ExplainButtonState extends State<ExplainButton> {
  bool _loading = false;

  Future<void> _onTap() async {
    if (_loading) return;
    setState(() => _loading = true);
    Haptics.selection();

    try {
      final client = await AppScope.api();
      // 결과 카드와 같은 요청(계층 모델·탭 크롭)으로 CAM — /predict-with-cam 은
      // 구형 단일 분류기가 전체 프레임을 보므로 표시 결과와 어긋났음.
      final result = await client.predictHierCam(
        widget.image,
        tapX: widget.tapNorm?.dx,
        tapY: widget.tapNorm?.dy,
      );
      if (!mounted) return;
      if (!result.camAvailable || result.camBase64 == null) {
        _showInfoDialog(
          title: 'CAM 미지원',
          message:
              '지금 서버 모델에서는 판단 근거 이미지를 만들 수 없어요.\n'
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
      _showInfoDialog(title: '설명을 만들지 못했어요', message: friendlyError(e));
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
    final imageBytes = base64Decode(result.camBase64!.split(',').last);
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
                      Icon(
                        Icons.lightbulb_outline,
                        size: 16,
                        color: cs.tertiary,
                      ),
                      const SizedBox(width: kSpaceS),
                      Expanded(
                        child: Text(
                          '예상한 영역과 다르다면 결과가 틀렸을 가능성이 높아요. '
                          '아래 피드백 카드에서 정정해주세요.',
                          style: TextStyle(
                            fontSize: 11,
                            color: cs.onSurfaceVariant,
                          ),
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
              width: 14,
              height: 14,
              child: CircularProgressIndicator(strokeWidth: 2),
            )
          : const Icon(Icons.visibility_outlined, size: 18),
      label: Text(_loading ? '분석 중...' : '왜 이렇게 분류했어?'),
      style: OutlinedButton.styleFrom(
        foregroundColor: widget.accent,
        side: BorderSide(color: widget.accent.withValues(alpha: 0.4)),
        padding: const EdgeInsets.symmetric(
          horizontal: kSpaceM,
          vertical: kSpaceS,
        ),
      ),
    );
  }
}
