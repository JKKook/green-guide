/// 탐지-후-분류 — 장면의 물건 후보 카드.
library;

import 'package:flutter/material.dart';

import '../../../api/models.dart';
import '../../../data/waste_info.dart';
import '../../../theme/app_theme.dart';

/// 다중 물건 후보 카드 — 장면에서 분리된 물건들을 나열, 탭하면 메인 결과 교체.
class ObjectsCard extends StatelessWidget {
  final List<ObjectCandidate> objects;
  final int? selected;
  final void Function(int) onSelect;
  const ObjectsCard({
    super.key,
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
                style: Theme.of(
                  context,
                ).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w700),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            '번호를 탭하면 그 물건의 분리배출 방법을 보여드려요',
            style: Theme.of(
              context,
            ).textTheme.bodySmall?.copyWith(color: cs.onSurfaceVariant),
          ),
          const SizedBox(height: kSpaceS),
          for (int i = 0; i < objects.length; i++)
            ObjectTile(
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

class ObjectTile extends StatelessWidget {
  final int index;
  final ObjectCandidate candidate;
  final bool selected;
  final VoidCallback onTap;
  const ObjectTile({
    super.key,
    required this.index,
    required this.candidate,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final info = infoForWithRollup(
      candidate.displayClass,
      parentSlug: candidate.coarseClass,
    );
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
          padding: const EdgeInsets.symmetric(
            horizontal: kSpaceS,
            vertical: kSpaceS,
          ),
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
                width: 26,
                height: 26,
                decoration: BoxDecoration(shape: BoxShape.circle, color: color),
                alignment: Alignment.center,
                child: Text(
                  '${index + 1}',
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w800,
                    fontSize: 13,
                  ),
                ),
              ),
              const SizedBox(width: kSpaceS),
              Icon(info?.icon ?? Icons.help_outline, size: 20, color: color),
              const SizedBox(width: 6),
              Expanded(
                child: Text(
                  name,
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
              ),
              if (!isReject)
                Text(
                  '${(conf * 100).toStringAsFixed(0)}%',
                  style: TextStyle(
                    fontSize: 12.5,
                    fontWeight: FontWeight.w700,
                    color: cs.onSurfaceVariant,
                  ),
                ),
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
