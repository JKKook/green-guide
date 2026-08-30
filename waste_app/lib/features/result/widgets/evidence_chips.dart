/// 서버가 융합한 시맨틱 증거 칩.
library;

import 'package:flutter/material.dart';

import '../../../api/models.dart';

/// 지역별 배출 안내 카드 — 지자체 조례 기준 (공공데이터포털 표준데이터).
class EvidenceChips extends StatelessWidget {
  final List<EvidenceItem> evidence;
  const EvidenceChips({super.key, required this.evidence});

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
