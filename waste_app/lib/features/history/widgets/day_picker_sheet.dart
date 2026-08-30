/// 하루치 기록 선택 시트.
library;

import 'package:flutter/material.dart';
import '../../../data/haptics.dart';
import '../../../data/history_repository.dart';
import '../../../data/waste_info.dart';
import '../../../theme/app_theme.dart';
import '../../../theme/design_tokens.dart';
import 'day_stack_card.dart';

/// 그날의 사진 고르기 시트 — 3열 썸네일 그리드, 탭하면 해당 기록 반환.
class DayPickerSheet extends StatelessWidget {
  final DateTime day;
  final List<HistoryEntry> entries;
  final (String, bool) Function(HistoryEntry) tagOf;
  const DayPickerSheet({super.key, 
    required this.day,
    required this.entries,
    required this.tagOf,
  });

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    final maxH = MediaQuery.sizeOf(context).height * 0.7;
    return SafeArea(
      child: ConstrainedBox(
        constraints: BoxConstraints(maxHeight: maxH),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                '${day.month}월 ${day.day}일 (${'월화수목금토일'[day.weekday - 1]}) · ${entries.length}건',
                style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 4),
              Text(
                '보고 싶은 사진을 골라주세요',
                style: TextStyle(fontSize: 12.5, color: t.muted2),
              ),
              const SizedBox(height: 14),
              Flexible(
                child: GridView.builder(
                  shrinkWrap: true,
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 3,
                    mainAxisSpacing: 10,
                    crossAxisSpacing: 10,
                    childAspectRatio: 0.78,
                  ),
                  itemCount: entries.length,
                  itemBuilder: (context, i) {
                    final e = entries[i];
                    final info = infoFor(e.predictedClass) ??
                        infoFor(kFineToCoarse[e.predictedClass] ?? '');
                    final (label, recyclable) = tagOf(e);
                    final time =
                        '${e.createdAt.hour.toString().padLeft(2, '0')}:${e.createdAt.minute.toString().padLeft(2, '0')}';
                    return InkWell(
                      borderRadius: BorderRadius.circular(14),
                      onTap: () {
                        Haptics.selection();
                        Navigator.of(context).pop(e);
                      },
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          AspectRatio(
                            aspectRatio: 1,
                            child: Stack(
                              fit: StackFit.expand,
                              children: [
                                HistoryThumb(
                                    entry: e,
                                    tokens: t,
                                    size: double.infinity,
                                    recyclable: recyclable),
                                Positioned(
                                  left: 6,
                                  bottom: 6,
                                  child: Container(
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 6, vertical: 2),
                                    decoration: BoxDecoration(
                                      color: kInkDeep
                                          .withValues(alpha: 0.6),
                                      borderRadius: BorderRadius.circular(999),
                                    ),
                                    child: Text(
                                      label,
                                      style: TextStyle(
                                        fontSize: 9.5,
                                        fontWeight: FontWeight.w700,
                                        color: recyclable
                                            ? kAccent200
                                            : kNeutral100,
                                      ),
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                          const SizedBox(height: 6),
                          Text(
                            info?.displayName ?? e.predictedClass,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                                fontSize: 12, fontWeight: FontWeight.w700),
                          ),
                          Text(
                            time,
                            style: TextStyle(fontSize: 10.5, color: t.muted),
                          ),
                        ],
                      ),
                    );
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
