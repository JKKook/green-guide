/// 우리 집 수거 요일 선택 시트 — 온보딩·설정 공용.
library;

import 'package:flutter/material.dart';
import '../../data/collection_schedule.dart';
import '../../data/haptics.dart';
import '../../theme/app_theme.dart';
import '../../theme/design_tokens.dart';

/// 우리 집 수거 요일 선택 시트 — 설정 > 내 동네 > 수거 요일 (주택·빌라).
Future<List<int>?> showPickupWeekdaysSheet(
  BuildContext context, {
  required List<int> current,
}) {
  return showModalBottomSheet<List<int>>(
    context: context,
    showDragHandle: true,
    builder: (_) => _PickupWeekdaysSheet(current: current),
  );
}


class _PickupWeekdaysSheet extends StatefulWidget {
  final List<int> current;
  const _PickupWeekdaysSheet({required this.current});

  @override
  State<_PickupWeekdaysSheet> createState() => _PickupWeekdaysSheetState();
}


class _PickupWeekdaysSheetState extends State<_PickupWeekdaysSheet> {
  late final Set<int> _days = {...widget.current};

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(kSpaceXL, 0, kSpaceXL, kSpaceXL),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('우리 집 수거 요일',
                style: TextStyle(fontSize: 22, fontWeight: FontWeight.w600)),
            const SizedBox(height: 4),
            Text('재활용품을 문 앞에 내놓는 요일이에요 · 비우면 동네 기본값을 써요',
                style: TextStyle(fontSize: 12.5, color: t.muted2)),
            const SizedBox(height: 18),
            Row(
              children: [
                for (var i = 0; i < 7; i++) ...[
                  if (i > 0) const SizedBox(width: 6),
                  Expanded(
                    child: Builder(builder: (context) {
                      final weekday = i + 1;
                      final on = _days.contains(weekday);
                      return InkWell(
                        borderRadius: BorderRadius.circular(14),
                        onTap: () {
                          Haptics.selection();
                          setState(() {
                            if (on) {
                              _days.remove(weekday);
                            } else {
                              _days.add(weekday);
                            }
                          });
                        },
                        child: Container(
                          height: 42,
                          alignment: Alignment.center,
                          decoration: BoxDecoration(
                            color: on ? kAccent700 : t.surface,
                            border: on ? null : Border.all(color: t.border),
                            borderRadius: BorderRadius.circular(14),
                          ),
                          child: Text(
                            kDayNames[i],
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: on ? FontWeight.w700 : FontWeight.w600,
                              color: on ? kNeutral100 : t.muted,
                            ),
                          ),
                        ),
                      );
                    }),
                  ),
                ],
              ],
            ),
            const SizedBox(height: 18),
            Material(
              color: kAccent700,
              borderRadius: BorderRadius.circular(kRadiusMedium),
              child: InkWell(
                borderRadius: BorderRadius.circular(kRadiusMedium),
                onTap: () => Navigator.of(context).pop(_days.toList()..sort()),
                child: const SizedBox(
                  height: 52,
                  child: Center(
                    child: Text('저장',
                        style: TextStyle(
                            fontSize: 15,
                            fontWeight: FontWeight.w700,
                            color: kNeutral100)),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
