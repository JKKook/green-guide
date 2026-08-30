/// 수거일 알림 추가/편집 시트.
library;

import 'package:flutter/cupertino.dart' show CupertinoPicker;
import 'package:flutter/material.dart';
import '../../data/collection_schedule.dart';
import '../../data/haptics.dart';
import '../../theme/app_theme.dart';
import '../../theme/design_tokens.dart';

/// 알림 시트의 결과 액션.
enum ReminderAction { save, turnOff }


class ReminderSheetResult {
  final ReminderAction action;
  final CollectionReminder? reminder;
  const ReminderSheetResult(this.action, [this.reminder]);
}


/// 알림 시간 설정 시트 — 시안 4b: 타임피커 + 당일/하루 전 + 저장.
Future<ReminderSheetResult?> showReminderSheet(
  BuildContext context, {
  CollectionReminder? existing,
  required int weekday,
  bool allowWeekdayPick = false,
}) {
  return showModalBottomSheet<ReminderSheetResult>(
    context: context,
    isScrollControlled: true,
    builder: (_) => _ReminderSheet(
      existing: existing,
      weekday: weekday,
      allowWeekdayPick: allowWeekdayPick,
    ),
  );
}


class _ReminderSheet extends StatefulWidget {
  final CollectionReminder? existing;
  final int weekday;
  final bool allowWeekdayPick;
  const _ReminderSheet({
    required this.existing,
    required this.weekday,
    required this.allowWeekdayPick,
  });

  @override
  State<_ReminderSheet> createState() => _ReminderSheetState();
}


class _ReminderSheetState extends State<_ReminderSheet> {
  static const _minutes = [0, 15, 30, 45];

  late int _weekday = widget.weekday;
  late bool _isPm;
  late int _hour12; // 1~12
  late int _minuteIdx;
  late bool _dayBefore = widget.existing?.dayBefore ?? false;

  @override
  void initState() {
    super.initState();
    final h = widget.existing?.hour ?? 17;
    final m = widget.existing?.minute ?? 30;
    _isPm = h >= 12;
    _hour12 = h % 12 == 0 ? 12 : h % 12;
    _minuteIdx = _minutes.indexOf(m - m % 15);
    if (_minuteIdx < 0) _minuteIdx = 0;
  }

  int get _hour24 {
    final base = _hour12 % 12;
    return _isPm ? base + 12 : base;
  }

  String get _timeLabel =>
      '${_isPm ? '오후' : '오전'} $_hour12:${_minutes[_minuteIdx].toString().padLeft(2, '0')}';

  Widget _picker({
    required List<String> items,
    required int initial,
    required ValueChanged<int> onChanged,
    bool loop = false,
  }) {
    final t = DsTokens.of(context);
    return SizedBox(
      height: 120,
      child: CupertinoPicker(
        itemExtent: 40,
        scrollController: FixedExtentScrollController(initialItem: initial),
        looping: loop,
        selectionOverlay: Container(
          decoration: BoxDecoration(
            color: t.accentChipBg.withValues(alpha: 0.5),
            border: Border(
              top: BorderSide(color: t.accentChipBorder),
              bottom: BorderSide(color: t.accentChipBorder),
            ),
          ),
        ),
        onSelectedItemChanged: (i) {
          Haptics.selection();
          onChanged(i);
        },
        children: [
          for (final item in items)
            Center(
              child: Text(
                item,
                style: const TextStyle(
                    fontSize: 20, fontWeight: FontWeight.w600),
              ),
            ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    return SafeArea(
      child: Padding(
        padding: EdgeInsets.fromLTRB(
          20,
          8,
          20,
          20 + MediaQuery.viewInsetsOf(context).bottom,
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 38,
              height: 4,
              margin: const EdgeInsets.only(bottom: kSpaceL),
              alignment: Alignment.center,
              decoration: BoxDecoration(
                color: t.handle,
                borderRadius: BorderRadius.circular(999),
              ),
            ),
            Row(
              children: [
                const Text(
                  '알림 시간 설정',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
                ),
                const Spacer(),
                InkWell(
                  borderRadius: BorderRadius.circular(999),
                  onTap: () => Navigator.of(context).pop(),
                  child: Padding(
                    padding: const EdgeInsets.all(kSpaceXS),
                    child: Icon(Icons.close, size: 18, color: t.muted),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              '수거일마다 이 시간에 알려드려요',
              style: TextStyle(fontSize: 11, color: t.muted2),
            ),
            if (widget.allowWeekdayPick) ...[
              const SizedBox(height: 14),
              Row(
                children: [
                  for (var i = 0; i < 7; i++) ...[
                    if (i > 0) const SizedBox(width: 5),
                    Expanded(
                      child: InkWell(
                        borderRadius: BorderRadius.circular(10),
                        onTap: () {
                          Haptics.selection();
                          setState(() => _weekday = i + 1);
                        },
                        child: Container(
                          padding: const EdgeInsets.symmetric(vertical: kSpaceS),
                          alignment: Alignment.center,
                          decoration: BoxDecoration(
                            color: _weekday == i + 1
                                ? kAccent700
                                : t.surface,
                            border: _weekday == i + 1
                                ? null
                                : Border.all(color: t.border),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: Text(
                            kDayNames[i],
                            style: TextStyle(
                              fontSize: 12,
                              fontWeight: FontWeight.w700,
                              color: _weekday == i + 1
                                  ? kNeutral100
                                  : t.muted,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ],
              ),
            ],
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: _picker(
                    items: const ['오전', '오후'],
                    initial: _isPm ? 1 : 0,
                    onChanged: (i) => setState(() => _isPm = i == 1),
                  ),
                ),
                Expanded(
                  child: _picker(
                    items: [for (var h = 1; h <= 12; h++) '$h'],
                    initial: _hour12 - 1,
                    loop: true,
                    onChanged: (i) => setState(() => _hour12 = i + 1),
                  ),
                ),
                Text(
                  ':',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.w700,
                    color: t.accentDeep,
                  ),
                ),
                Expanded(
                  child: _picker(
                    items: [
                      for (final m in _minutes) m.toString().padLeft(2, '0')
                    ],
                    initial: _minuteIdx,
                    onChanged: (i) => setState(() => _minuteIdx = i),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                for (final (label, value) in [('당일 알림', false), ('하루 전 알림', true)])
                  Padding(
                    padding: const EdgeInsets.only(right: kSpaceS),
                    child: InkWell(
                      borderRadius: BorderRadius.circular(999),
                      onTap: () {
                        Haptics.selection();
                        setState(() => _dayBefore = value);
                      },
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 13, vertical: 7),
                        decoration: BoxDecoration(
                          color: _dayBefore == value
                              ? t.accentChipBg
                              : t.surface,
                          border: Border.all(
                            color: _dayBefore == value
                                ? brandSeed
                                : t.border,
                          ),
                          borderRadius: BorderRadius.circular(999),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            if (_dayBefore == value) ...[
                              Icon(Icons.check,
                                  size: 12, color: t.accentDeep),
                              const SizedBox(width: 5),
                            ],
                            Text(
                              label,
                              style: TextStyle(
                                fontSize: 12,
                                fontWeight: _dayBefore == value
                                    ? FontWeight.w700
                                    : FontWeight.w600,
                                color: _dayBefore == value
                                    ? t.accentDeep
                                    : t.muted2,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 20),
            Row(
              children: [
                Expanded(
                  child: InkWell(
                    borderRadius: BorderRadius.circular(14),
                    onTap: () => Navigator.of(context)
                        .pop(const ReminderSheetResult(ReminderAction.turnOff)),
                    child: Container(
                      height: 50,
                      alignment: Alignment.center,
                      decoration: BoxDecoration(
                        border: Border.all(
                          color: t.handle,
                        ),
                        borderRadius: BorderRadius.circular(14),
                      ),
                      child: Text(
                        '알림 끄기',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                          color: t.muted2,
                        ),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  flex: 2,
                  child: Material(
                    color: brandSeed,
                    borderRadius: BorderRadius.circular(14),
                    child: InkWell(
                      borderRadius: BorderRadius.circular(14),
                      onTap: () => Navigator.of(context).pop(
                        ReminderSheetResult(
                          ReminderAction.save,
                          CollectionReminder(
                            weekday: _weekday,
                            hour: _hour24,
                            minute: _minutes[_minuteIdx],
                            dayBefore: _dayBefore,
                          ),
                        ),
                      ),
                      child: SizedBox(
                        height: 50,
                        child: Center(
                          child: Text(
                            '$_timeLabel 저장',
                            style: const TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w700,
                              color: kNeutral100,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
