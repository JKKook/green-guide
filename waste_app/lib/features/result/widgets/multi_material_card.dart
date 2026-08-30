/// 다중재질 카드 — 재질별 배출 방법.
library;

import 'package:flutter/material.dart';

import '../../../api/models.dart';
import '../../../data/waste_info.dart';
import '../../../theme/app_theme.dart';

/// 다중재질 안내 카드 — 확실히 다른 재질이 2개 이상 검출됐을 때.
/// 재질별 분리 배출을 안내. 위 오버레이의 빗금 색상과 라벨이 1:1 대응.
class MultiMaterialCard extends StatelessWidget {
  final List<MaterialRegion> regions;
  const MultiMaterialCard({super.key, required this.regions});

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
            style: TextStyle(
              fontSize: 13,
              height: 1.4,
              color: cs.onSurfaceVariant,
            ),
          ),
          const SizedBox(height: kSpaceM),
          ...unique.map(
            (region) =>
                MaterialMethodTile(region: region, info: infoFor(region.slug)),
          ),
        ],
      ),
    );
  }
}

/// 다중재질 카드의 재질 1개 항목 — 이름·배출함 + 그 재질의 배출 방법(how_to)을 함께 제시.
/// 단일 재질만 안내하던 것을 재질별 방법까지 다중 제시하도록 확장.
class MaterialMethodTile extends StatelessWidget {
  final MaterialRegion region;
  final WasteInfo? info;
  const MaterialMethodTile({super.key, required this.region, this.info});

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
                decoration: BoxDecoration(
                  color: accent,
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  info?.icon ?? Icons.help_outline,
                  color: Colors.white,
                  size: 17,
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
            ...steps
                .take(2)
                .map(
                  (s) => Padding(
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
                  ),
                ),
          ],
        ],
      ),
    );
  }
}
