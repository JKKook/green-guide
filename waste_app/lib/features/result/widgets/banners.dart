/// 결과 배너 — 지역 규정 구조, 사진 품질, 불확실, 분류 불가.
library;

import 'dart:io';

import 'package:flutter/material.dart';

import '../../../api/models.dart';
import '../../../data/image_quality.dart';
import '../../../data/waste_info.dart';
import '../../../theme/app_theme.dart';
import 'explain_button.dart';

/// 영역 분석 발견을 메인 답으로 승격하는 배너.
class RegionRescueBanner extends StatelessWidget {
  final MaterialRegion region;
  final WasteInfo? info;
  const RegionRescueBanner({super.key, required this.region, required this.info});

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


/// 사진 품질 문제 (어두움/흔들림) 안내 배너.
class QualityBanner extends StatelessWidget {
  final ImageQualityResult quality;
  const QualityBanner({super.key, required this.quality});

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
class UncertainBanner extends StatelessWidget {
  final Prediction prediction;
  const UncertainBanner({super.key, required this.prediction});

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
///    아래 재질별 안내(MultiMaterialCard)로 안내함. (image 제공 시 CAM 표시)
class RejectCard extends StatelessWidget {
  final Prediction prediction;
  final bool isMultiMaterial;
  final bool isMultiObject;
  final File? image;
  const RejectCard({super.key, 
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
            ExplainButton(
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
