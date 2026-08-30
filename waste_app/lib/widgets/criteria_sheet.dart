import 'package:flutter/material.dart';

import '../data/haptics.dart';
import '../data/waste_info.dart';
import '../theme/app_theme.dart';

/// 밝은 재질 컬러는 아이콘 대비가 낮아 블루그레이로 보정.
Color wasteIconColor(WasteInfo info) => info.color.computeLuminance() > 0.6
    ? const Color(0xFF607D8B)
    : info.color;

/// 품목 배출 기준 바텀시트 — 통합 검색·그리드 공용.
void showCriteriaSheet(BuildContext context, WasteInfo info) {
  Haptics.selection();
  final cs = Theme.of(context).colorScheme;
  showModalBottomSheet<void>(
    context: context,
    showDragHandle: true,
    builder: (_) => SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(kSpaceXL, 0, kSpaceXL, kSpaceXL),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    color: Theme.of(context).brightness == Brightness.dark
                        ? const Color(0xFFEDEEED)
                        : cs.surfaceContainerHigh,
                    shape: BoxShape.circle,
                  ),
                  child: Icon(info.icon, color: wasteIconColor(info), size: 23),
                ),
                const SizedBox(width: kSpaceM),
                Expanded(
                  child: Text(
                    info.displayName,
                    style: Theme.of(context)
                        .textTheme
                        .titleLarge
                        ?.copyWith(fontWeight: FontWeight.w700),
                  ),
                ),
              ],
            ),
            const SizedBox(height: kSpaceM),
            Text(
              info.summary,
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            if (info.bin.isNotEmpty) ...[
              const SizedBox(height: kSpaceM),
              _CriteriaRow(label: '배출함', value: info.bin),
            ],
            if (info.howTo.isNotEmpty) ...[
              const SizedBox(height: kSpaceS),
              _CriteriaRow(label: '배출방법', value: info.howTo.join('\n')),
            ],
            const SizedBox(height: kSpaceS),
            Text(
              info.trainedInModel
                  ? '실제 판정은 사진 속 재질 분석 결과를 따릅니다'
                  : 'AI 분류 대상은 아니지만 배출 방법을 안내해 드려요',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: cs.onSurfaceVariant,
                  ),
            ),
          ],
        ),
      ),
    ),
  );
}

class _CriteriaRow extends StatelessWidget {
  final String label;
  final String value;
  const _CriteriaRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
          width: 64,
          child: Text(
            label,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: cs.onSurfaceVariant,
                  fontWeight: FontWeight.w700,
                ),
          ),
        ),
        Expanded(
          child: Text(value, style: Theme.of(context).textTheme.bodyMedium),
        ),
      ],
    );
  }
}
