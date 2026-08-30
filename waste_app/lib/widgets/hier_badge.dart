/// 계층 분류 배지 — "대분류 → 세부품목" 경로를 한 줄로 표시.
///
/// 표시 규칙 (blueprint §6):
/// - 대분류 칩은 항상 표시 (계층 응답이 있을 때)
/// - 세부까지 확신(display_level=fine)이고 세부가 레지스트리에 활성일 때만
///   "→ 세부" 표시
/// - 세부가 게이트는 통과했지만 아직 비활성(수집 중)이면 "세부 준비 중" 힌트
library;

import 'package:flutter/material.dart';

import '../api/models.dart';
import '../data/waste_info.dart';
import '../theme/app_theme.dart';

class HierBadge extends StatelessWidget {
  final HierInfo hier;
  const HierBadge({super.key, required this.hier});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final coarse = infoFor(hier.coarseClass);
    final fineInfo =
        hier.fineClass != null ? infoFor(hier.fineClass!) : null;
    final showFine = hier.isFine && fineInfo != null;
    // 세부 게이트는 통과했지만 레지스트리에 없음(active=false) → 롤업 노출 중
    final finePending = hier.isFine && fineInfo == null;

    Widget chip(String label, IconData icon, Color color,
        {bool emphasized = false}) {
      return Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: color.withValues(alpha: emphasized ? 0.18 : 0.10),
          borderRadius: BorderRadius.circular(kRadiusSmall),
          border: Border.all(color: color.withValues(alpha: 0.35)),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 14, color: color),
            const SizedBox(width: 4),
            Text(
              label,
              style: TextStyle(
                fontSize: 12,
                fontWeight: emphasized ? FontWeight.w700 : FontWeight.w600,
                color: color,
              ),
            ),
          ],
        ),
      );
    }

    return Wrap(
      crossAxisAlignment: WrapCrossAlignment.center,
      spacing: 6,
      runSpacing: 4,
      children: [
        if (coarse != null)
          chip(coarse.displayName, coarse.icon, coarse.color,
              emphasized: !showFine),
        if (showFine) ...[
          Icon(Icons.chevron_right, size: 16, color: cs.onSurfaceVariant),
          chip(fineInfo.displayName, fineInfo.icon, fineInfo.color,
              emphasized: true),
        ],
        if (finePending)
          chip('세부 분류 준비 중', Icons.hourglass_empty, cs.onSurfaceVariant),
      ],
    );
  }
}
