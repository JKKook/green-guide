/// 기록 탭 — 날짜별 묶음 카드 + 썸네일.
library;

import 'package:flutter/material.dart';
import '../../../data/history_repository.dart';
import '../../../data/waste_info.dart';
import '../../../theme/app_theme.dart';
import '../../../theme/design_tokens.dart';

/// 날짜 스택 카드 — 그날의 사진을 겹쳐 쌓은 더미 + 요약. 탭하면 사진 고르기.
class DayStackCard extends StatelessWidget {
  final DateTime day;
  final List<HistoryEntry> entries;
  final DsTokens tokens;
  final (String, bool) Function(HistoryEntry) tagOf;
  final VoidCallback onTap;
  const DayStackCard({super.key, 
    required this.day,
    required this.entries,
    required this.tokens,
    required this.tagOf,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final t = tokens;
    final general = entries.where((e) => !tagOf(e).$2).length;
    final names = <String>{
      for (final e in entries)
        (infoFor(e.predictedClass) ??
                infoFor(kFineToCoarse[e.predictedClass] ?? ''))
            ?.displayName ??
            e.predictedClass,
    }.toList();
    final shown = names.take(3).toList();
    final more = names.length - shown.length;
    final stack = entries.take(3).toList();

    return Material(
      color: t.surface,
      borderRadius: BorderRadius.circular(18),
      child: InkWell(
        borderRadius: BorderRadius.circular(18),
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.fromLTRB(14, 14, 12, 14),
          decoration: BoxDecoration(
            border: Border.all(color: t.border),
            borderRadius: BorderRadius.circular(18),
            boxShadow: [
              BoxShadow(
                color: kInkCardShadow.withValues(alpha: 0.14),
                offset: const Offset(0, 1),
                blurRadius: 2,
              ),
            ],
          ),
          child: Row(
            children: [
              // 겹쳐진 썸네일 더미 (최대 3장, 뒤로 갈수록 살짝 기울고 흐려짐)
              SizedBox(
                width: 64.0 + 14 * (stack.length - 1),
                height: 72,
                child: Stack(
                  children: [
                    for (var i = stack.length - 1; i >= 0; i--)
                      Positioned(
                        left: 14.0 * i,
                        top: i == 0 ? 4 : (i == 1 ? 0 : 2),
                        child: Transform.rotate(
                          angle: i == 0 ? 0 : (i == 1 ? 0.09 : -0.07),
                          child: Opacity(
                            opacity: i == 0 ? 1 : (i == 1 ? 0.85 : 0.7),
                            child: HistoryThumb(
                                entry: stack[i],
                                tokens: t,
                                size: 64,
                                recyclable: tagOf(stack[i]).$2),
                          ),
                        ),
                      ),
                    if (entries.length > 3)
                      Positioned(
                        right: 0,
                        bottom: 0,
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 7, vertical: 3),
                          decoration: BoxDecoration(
                            color: kAccent700,
                            borderRadius: BorderRadius.circular(999),
                          ),
                          child: Text(
                            '+${entries.length - 3}',
                            style: const TextStyle(
                              fontSize: 10.5,
                              fontWeight: FontWeight.w700,
                              color: kNeutral100,
                            ),
                          ),
                        ),
                      ),
                  ],
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '${day.month}월 ${day.day}일 '
                      '(${'월화수목금토일'[day.weekday - 1]})',
                      style: const TextStyle(
                          fontSize: 14.5, fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      '${entries.length}건 · 재활용 ${entries.length - general} · 일반 $general',
                      style: TextStyle(fontSize: 11.5, color: t.muted),
                    ),
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 5,
                      runSpacing: 5,
                      children: [
                        for (final n in shown)
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 8, vertical: 3),
                            decoration: BoxDecoration(
                              color: t.accentChipBg,
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Text(
                              n,
                              style: TextStyle(
                                fontSize: 10.5,
                                fontWeight: FontWeight.w600,
                                color: t.accentChipText,
                              ),
                            ),
                          ),
                        if (more > 0)
                          Padding(
                            padding: const EdgeInsets.only(top: 3),
                            child: Text(
                              '외 $more',
                              style: TextStyle(fontSize: 10.5, color: t.muted),
                            ),
                          ),
                      ],
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 6),
              Icon(Icons.chevron_right, size: 18, color: t.faint),
            ],
          ),
        ),
      ),
    );
  }
}


/// 썸네일 — 재질 아이콘 타일 (사진은 원본 뷰어에서만 렌더링).
class HistoryThumb extends StatelessWidget {
  final HistoryEntry entry;
  final DsTokens tokens;
  final double size;
  final bool recyclable;
  const HistoryThumb({super.key, 
    required this.entry,
    required this.tokens,
    required this.size,
    this.recyclable = true,
  });

  @override
  Widget build(BuildContext context) {
    final info = infoFor(entry.predictedClass) ??
        infoFor(kFineToCoarse[entry.predictedClass] ?? '');
    final iconSize = size.isFinite ? size * 0.4 : 28.0;
    return Container(
      width: size,
      height: size,
      clipBehavior: Clip.antiAlias,
      decoration: BoxDecoration(
        color: recyclable ? tokens.accentChipBg : tokens.border,
        border: Border.all(color: tokens.surface, width: 2),
        borderRadius: BorderRadius.circular(14),
        boxShadow: [
          BoxShadow(
            color: kInkCardShadow.withValues(alpha: 0.18),
            offset: const Offset(0, 2),
            blurRadius: 6,
          ),
        ],
      ),
      child: Center(
        child: Icon(
          info?.icon ?? Icons.help_outline,
          size: iconSize,
          color: recyclable ? tokens.accentStrong : tokens.muted2,
        ),
      ),
    );
  }
}
