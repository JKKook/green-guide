/// 예측 결과 카드 (라벨·신뢰도·설명 버튼).
library;

import 'dart:io';

import 'package:flutter/material.dart';

import '../../../api/models.dart';
import '../../../core/ui/ds_card.dart';
import '../../../data/confidence.dart';
import '../../../data/waste_info.dart';
import '../../../services/prediction_service.dart' show isCloudFallback;
import '../../../theme/design_tokens.dart';
import '../../../widgets/hier_badge.dart';
import 'explain_button.dart';

class PredictionCard extends StatelessWidget {
  final File image;
  final Prediction prediction;
  final WasteInfo? info;
  final Color accent;
  final ConfidenceAssessment assessment;
  const PredictionCard({
    super.key,
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
        ExplainButton(
          image: image,
          accent: accent,
          info: info,
          prediction: prediction,
        ),
      ],
    );
  }
}
