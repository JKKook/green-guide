import 'package:flutter/cupertino.dart' show CupertinoPicker;
import 'package:flutter/material.dart';

import '../data/collection_schedule.dart';
import '../data/haptics.dart';
import '../data/settings_store.dart';
import '../theme/app_theme.dart';
import '../theme/design_tokens.dart';
import '../widgets/region_picker.dart';

/// 수거일 안내 화면 — 시안 4a: 오늘 카드 + 주간 캘린더 + 이번 주 일정.
class CollectionScheduleScreen extends StatefulWidget {
  const CollectionScheduleScreen({super.key});

  @override
  State<CollectionScheduleScreen> createState() =>
      _CollectionScheduleScreenState();
}

class _CollectionScheduleScreenState extends State<CollectionScheduleScreen> {
  final SettingsStore _settings = SettingsStore();
  final ReminderStore _reminders = ReminderStore();
  (String, String)? _region;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final region = await _settings.getRegion();
    if (!mounted) return;
    setState(() => _region = region);
  }

  Future<void> _changeRegion() async {
    Haptics.selection();
    final picked = await showRegionPicker(context);
    if (picked != null && mounted) setState(() => _region = picked);
  }

  /// 오늘 요일의 알림 설정 — 4b 시트.
  Future<void> _configureTodayReminder() async {
    Haptics.selection();
    final weekday = DateTime.now().weekday;
    final existing = (await _reminders.load())
        .where((r) => r.weekday == weekday)
        .firstOrNull;
    if (!mounted) return;
    final result = await showReminderSheet(
      context,
      existing: existing,
      weekday: weekday,
    );
    if (result == null || !mounted) return;
    final list = await _reminders.load();
    list.removeWhere((r) => r.weekday == weekday);
    if (result.action == ReminderAction.save) {
      list.add(result.reminder!);
    }
    await _reminders.save(list);
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(result.action == ReminderAction.save
          ? '${result.reminder!.timeLabel} 수거일 알림을 저장했어요 · 발송은 준비 중이에요'
          : '알림을 껐어요'),
    ));
  }

  Color? _dotColor(DsTokens t, PickupKind kind) => switch (kind) {
        PickupKind.plasticVinyl => brandSeed,
        PickupKind.paperBox => kAccent2400,
        PickupKind.general => t.faint,
        PickupKind.none => null,
      };

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    final now = DateTime.now();
    final todayIdx = now.weekday - 1;
    final monday = DateTime(now.year, now.month, now.day)
        .subtract(Duration(days: todayIdx));
    final todayPickup = effectiveWeekSchedule()[todayIdx];

    return Scaffold(
      appBar: AppBar(
        titleSpacing: 0,
        title: const Text(
          '수거일 안내',
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
        actions: [
          InkWell(
            borderRadius: BorderRadius.circular(kRadiusSmall),
            onTap: _changeRegion,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.place_outlined,
                      size: 14, color: t.accentChipText),
                  const SizedBox(width: 5),
                  Text(
                    _region?.$2 ?? '지역 설정',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: t.accentChipText,
                    ),
                  ),
                  Icon(Icons.expand_more, size: 13, color: t.accentChipText),
                ],
              ),
            ),
          ),
          const SizedBox(width: kSpaceM),
        ],
      ),
      body: SafeArea(
        top: false,
        child: ListView(
          padding: const EdgeInsets.fromLTRB(20, kSpaceS, 20, kSpaceXL),
          children: [
            // 오늘 카드
            Container(
              padding: const EdgeInsets.fromLTRB(18, 16, 18, 16),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: t.dark
                      ? [kAccent900, kAccent800]
                      : [kAccent100, kAccent200],
                ),
                border: Border.all(
                  color: t.dark ? kAccent700 : kAccent300,
                ),
                borderRadius: BorderRadius.circular(20),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0xFF2B2B2D).withValues(alpha: 0.16),
                    offset: const Offset(0, 3),
                    blurRadius: 10,
                  ),
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: kAccent700,
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: Text(
                          '오늘 · ${kDayNames[todayIdx]}요일',
                          style: const TextStyle(
                            fontSize: 11,
                            fontWeight: FontWeight.w700,
                            color: kNeutral100,
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      if (todayPickup != PickupKind.none)
                        Expanded(
                          child: Text(
                            '$kPickupTimeText 배출',
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style:
                                TextStyle(fontSize: 11.5, color: t.accentDeep),
                          ),
                        )
                      else
                        const Spacer(),
                      InkWell(
                        borderRadius: BorderRadius.circular(999),
                        onTap: _configureTodayReminder,
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 11, vertical: 6),
                          decoration: BoxDecoration(
                            color: t.surface,
                            border: Border.all(
                              color: t.dark ? kAccent700 : kAccent400,
                            ),
                            borderRadius: BorderRadius.circular(999),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(Icons.notifications_none,
                                  size: 12, color: t.accentDeep),
                              const SizedBox(width: 5),
                              Text(
                                '알림 설정',
                                style: TextStyle(
                                  fontSize: 11.5,
                                  fontWeight: FontWeight.w700,
                                  color: t.accentDeep,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  Text(
                    todayPickup.fullLabel,
                    style: const TextStyle(
                      fontSize: 26,
                      fontWeight: FontWeight.w600,
                      height: 1.15,
                    ),
                  ),
                  if (todayPickup != PickupKind.none) ...[
                    const SizedBox(height: 4),
                    Text(
                      '문 앞 배출 · 투명 봉투 사용',
                      style: TextStyle(fontSize: 11, color: t.muted2),
                    ),
                  ],
                  const SizedBox(height: 10),
                  Container(
                    height: 1,
                    color: t.dark ? kAccent700 : kAccent300,
                  ),
                  const SizedBox(height: 10),
                  Row(
                    children: [
                      Icon(Icons.info_outline,
                          size: 12, color: t.accentChipText),
                      const SizedBox(width: 5),
                      Expanded(
                        child: Text(
                          _region == null
                              ? '일반적인 배출 요일 예시 · 지역 설정 시 맞춤 안내 예정'
                              : '일반적인 배출 요일 예시 · ${_region!.$2} 실제 수거일은 '
                                  '지자체 공지를 확인해 주세요',
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(fontSize: 11, color: t.accentDeep),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            // 주간 캘린더 (월~일, 이번 주 날짜)
            Row(
              children: [
                for (var i = 0; i < 7; i++) ...[
                  if (i > 0) const SizedBox(width: 6),
                  Expanded(
                    child: Builder(builder: (context) {
                      final date = monday.add(Duration(days: i));
                      final isToday = i == todayIdx;
                      final dot =
                          _dotColor(t, effectiveWeekSchedule()[i]);
                      return Container(
                        padding: const EdgeInsets.symmetric(vertical: 10),
                        decoration: BoxDecoration(
                          color: isToday ? t.accentChipBg : t.surface,
                          border: Border.all(
                            color: isToday ? brandSeed : t.border,
                            width: isToday ? 1.5 : 1,
                          ),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Column(
                          children: [
                            Text(
                              kDayNames[i],
                              style: TextStyle(
                                fontSize: 11,
                                fontWeight: isToday
                                    ? FontWeight.w700
                                    : FontWeight.w600,
                                color: isToday ? t.accentChipText : t.muted,
                              ),
                            ),
                            const SizedBox(height: 6),
                            Text(
                              '${date.day}',
                              style: TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.w700,
                                color: isToday ? t.accentDeep : null,
                              ),
                            ),
                            const SizedBox(height: 6),
                            Container(
                              width: 7,
                              height: 7,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: isToday ? brandSeed : dot,
                              ),
                            ),
                          ],
                        ),
                      );
                    }),
                  ),
                ],
              ],
            ),
            const SizedBox(height: 10),
            // 범례
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 4),
              child: Row(
                children: [
                  for (final (label, color) in [
                    ('플라스틱·비닐', brandSeed),
                    ('종이·박스', kAccent2400),
                    ('일반쓰레기', t.faint),
                  ]) ...[
                    Container(
                      width: 7,
                      height: 7,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: color,
                      ),
                    ),
                    const SizedBox(width: 5),
                    Text(
                      label,
                      style: TextStyle(fontSize: 11, color: t.muted2),
                    ),
                    const SizedBox(width: 14),
                  ],
                ],
              ),
            ),
            const SizedBox(height: 20),
            const Text(
              '이번 주 일정',
              style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 10),
            // 오늘부터의 배출 일정 카드
            for (var i = 0; i < 7; i++)
              if (effectiveWeekSchedule()[(todayIdx + i) % 7] != PickupKind.none)
                Builder(builder: (context) {
                  final dayIdx = (todayIdx + i) % 7;
                  final date = monday.add(Duration(days: todayIdx + i));
                  final isToday = i == 0;
                  return Container(
                    margin: const EdgeInsets.only(bottom: 8),
                    padding: const EdgeInsets.fromLTRB(15, 13, 15, 13),
                    decoration: BoxDecoration(
                      color: t.surface,
                      border: Border.all(color: t.border),
                      borderRadius: BorderRadius.circular(16),
                      boxShadow: [
                        BoxShadow(
                          color:
                              const Color(0xFF2B2B2D).withValues(alpha: 0.14),
                          offset: const Offset(0, 1),
                          blurRadius: 2,
                        ),
                      ],
                    ),
                    child: Row(
                      children: [
                        SizedBox(
                          width: 40,
                          child: Column(
                            children: [
                              Text(
                                kDayNames[dayIdx],
                                style: TextStyle(
                                  fontSize: 11,
                                  fontWeight: FontWeight.w700,
                                  color: isToday
                                      ? t.accentChipText
                                      : t.muted,
                                ),
                              ),
                              Text(
                                '${date.day}',
                                style: const TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                effectiveWeekSchedule()[dayIdx].fullLabel,
                                style: const TextStyle(
                                  fontSize: 13.5,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                              const SizedBox(height: 2),
                              Text(
                                '$kPickupTimeText · $kPickupPlaceText',
                                style: TextStyle(
                                    fontSize: 11.5, color: t.muted2),
                              ),
                            ],
                          ),
                        ),
                        if (isToday)
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 10, vertical: 3),
                            decoration: BoxDecoration(
                              color: t.accentChipBg,
                              borderRadius: BorderRadius.circular(6),
                            ),
                            child: Text(
                              '오늘',
                              style: TextStyle(
                                fontSize: 11,
                                fontWeight: FontWeight.w700,
                                color: t.accentChipText,
                              ),
                            ),
                          ),
                      ],
                    ),
                  );
                }),
            const SizedBox(height: 8),
            // 아파트 단지 안내
            Container(
              padding: const EdgeInsets.fromLTRB(15, 14, 15, 14),
              decoration: BoxDecoration(
                color: t.dark ? kAccent2900 : kAccent2100,
                border: Border.all(
                  color: t.dark ? kAccent2700 : kAccent2300,
                ),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Padding(
                    padding: const EdgeInsets.only(top: 1),
                    child: Icon(
                      Icons.warning_amber_outlined,
                      size: 17,
                      color: t.dark ? kAccent2300 : kAccent2900,
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          '아파트 단지는 수거일이 다를 수 있어요',
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w700,
                            color: t.dark ? kAccent2300 : kAccent2900,
                          ),
                        ),
                        const SizedBox(height: 6),
                        Text(
                          '이 일정은 일반적인 배출 요일 예시예요. 단지마다 '
                          '수거업체·요일이 다를 수 있으니, 단지 게시판이나 '
                          '관리사무소 공지를 확인해 주세요.',
                          style: TextStyle(
                            fontSize: 11.5,
                            height: 1.5,
                            color: t.dark ? kAccent2300 : kAccent2900,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}


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
              margin: const EdgeInsets.only(bottom: 16),
              alignment: Alignment.center,
              decoration: BoxDecoration(
                color: t.dark ? const Color(0xFF5D5D60) : kNeutral300,
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
                    padding: const EdgeInsets.all(4),
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
                          padding: const EdgeInsets.symmetric(vertical: 8),
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
                    padding: const EdgeInsets.only(right: 8),
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
                          color: t.dark ? const Color(0xFF5D5D60) : kNeutral300,
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


/// 수거일 알림 관리 — 시안 4c: 등록된 알림 목록.
class CollectionRemindersScreen extends StatefulWidget {
  const CollectionRemindersScreen({super.key});

  @override
  State<CollectionRemindersScreen> createState() =>
      _CollectionRemindersScreenState();
}

class _CollectionRemindersScreenState extends State<CollectionRemindersScreen> {
  final ReminderStore _store = ReminderStore();
  List<CollectionReminder> _reminders = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final list = await _store.load();
    list.sort((a, b) => a.weekday.compareTo(b.weekday));
    if (!mounted) return;
    setState(() {
      _reminders = list;
      _loading = false;
    });
  }

  Future<void> _add() async {
    Haptics.selection();
    final result = await showReminderSheet(
      context,
      weekday: DateTime.now().weekday,
      allowWeekdayPick: true,
    );
    if (result == null || result.action != ReminderAction.save) return;
    final list = await _store.load();
    list.removeWhere((r) => r.weekday == result.reminder!.weekday);
    list.add(result.reminder!);
    await _store.save(list);
    await _load();
  }

  Future<void> _edit(CollectionReminder reminder) async {
    Haptics.selection();
    final result = await showReminderSheet(
      context,
      existing: reminder,
      weekday: reminder.weekday,
    );
    if (result == null) return;
    final list = await _store.load();
    list.removeWhere((r) => r.weekday == reminder.weekday);
    if (result.action == ReminderAction.save) {
      list.add(result.reminder!);
    } else {
      list.add(reminder.copyWith(enabled: false));
    }
    await _store.save(list);
    await _load();
  }

  Future<void> _toggle(CollectionReminder reminder, bool enabled) async {
    Haptics.selection();
    final list = await _store.load();
    final i = list.indexWhere((r) => r.weekday == reminder.weekday);
    if (i >= 0) list[i] = list[i].copyWith(enabled: enabled);
    await _store.save(list);
    await _load();
  }

  Future<void> _delete(CollectionReminder reminder) async {
    final list = await _store.load();
    list.removeWhere((r) => r.weekday == reminder.weekday);
    await _store.save(list);
    await _load();
  }

  @override
  Widget build(BuildContext context) {
    final t = DsTokens.of(context);
    return Scaffold(
      appBar: AppBar(
        titleSpacing: 0,
        title: const Text(
          '수거일 알림',
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
        actions: [
          InkWell(
            borderRadius: BorderRadius.circular(999),
            onTap: _add,
            child: Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 11, vertical: 6),
              decoration: BoxDecoration(
                color: t.surface,
                border: Border.all(
                  color: t.dark ? kAccent700 : kAccent400,
                ),
                borderRadius: BorderRadius.circular(999),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.add, size: 12, color: t.accentDeep),
                  const SizedBox(width: 5),
                  Text(
                    '알림 추가',
                    style: TextStyle(
                      fontSize: 11.5,
                      fontWeight: FontWeight.w700,
                      color: t.accentDeep,
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(width: kSpaceL),
        ],
      ),
      body: SafeArea(
        top: false,
        child: _loading
            ? const Center(child: CircularProgressIndicator())
            : ListView(
                padding: const EdgeInsets.fromLTRB(20, kSpaceS, 20, kSpaceXL),
                children: [
                  Text(
                    _reminders.isEmpty
                        ? '등록된 알림이 없어요 · 수거일마다 반복돼요'
                        : '등록된 알림 ${_reminders.length}개 · 수거일마다 반복돼요',
                    style: TextStyle(fontSize: 11, color: t.muted2),
                  ),
                  const SizedBox(height: 6),
                  // OS 알림 연동 전 — 설정만 저장된다는 점을 분명히 안내.
                  Row(
                    children: [
                      Icon(Icons.info_outline, size: 12, color: t.muted2),
                      const SizedBox(width: 5),
                      Expanded(
                        child: Text(
                          kReminderPendingNote,
                          style: TextStyle(fontSize: 11, color: t.muted2),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),
                  for (final r in _reminders)
                    Dismissible(
                      key: ValueKey('reminder-${r.weekday}'),
                      direction: DismissDirection.endToStart,
                      onDismissed: (_) => _delete(r),
                      background: Container(
                        alignment: Alignment.centerRight,
                        padding: const EdgeInsets.only(right: 20),
                        margin: const EdgeInsets.only(bottom: 8),
                        decoration: BoxDecoration(
                          color: Theme.of(context).colorScheme.errorContainer,
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: Icon(
                          Icons.delete_outline,
                          color:
                              Theme.of(context).colorScheme.onErrorContainer,
                        ),
                      ),
                      child: Opacity(
                        opacity: r.enabled ? 1 : 0.62,
                        child: Container(
                          margin: const EdgeInsets.only(bottom: 8),
                          decoration: BoxDecoration(
                            color: t.surface,
                            border: Border.all(color: t.border),
                            borderRadius: BorderRadius.circular(16),
                            boxShadow: r.enabled
                                ? [
                                    BoxShadow(
                                      color: const Color(0xFF2B2B2D)
                                          .withValues(alpha: 0.14),
                                      offset: const Offset(0, 1),
                                      blurRadius: 2,
                                    ),
                                  ]
                                : null,
                          ),
                          child: InkWell(
                            borderRadius: BorderRadius.circular(16),
                            onTap: () => _edit(r),
                            child: Padding(
                              padding:
                                  const EdgeInsets.fromLTRB(15, 14, 15, 14),
                              child: Row(
                                children: [
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          r.timeLabel,
                                          style: const TextStyle(
                                            fontSize: 24,
                                            fontWeight: FontWeight.w600,
                                            height: 1,
                                          ),
                                        ),
                                        const SizedBox(height: 3),
                                        Text(
                                          '${kDayNames[r.weekday - 1]}요일 · '
                                          '${r.pickup.fullLabel} · '
                                          '${r.dayBefore ? '하루 전' : '당일'}',
                                          style: TextStyle(
                                            fontSize: 11,
                                            color: t.muted2,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                  Switch(
                                    value: r.enabled,
                                    activeTrackColor: brandSeed,
                                    onChanged: (v) => _toggle(r, v),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ),
                      ),
                    ),
                  const SizedBox(height: 8),
                  Container(
                    padding: const EdgeInsets.fromLTRB(15, 13, 15, 13),
                    decoration: BoxDecoration(
                      color: t.surface,
                      border: Border.all(color: t.border),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Padding(
                          padding: const EdgeInsets.only(top: 1),
                          child:
                              Icon(Icons.info_outline, size: 15, color: t.muted),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: Text(
                            '알림을 누르면 시간·요일을 수정할 수 있어요. '
                            '왼쪽으로 밀면 삭제돼요.',
                            style: TextStyle(
                              fontSize: 11,
                              height: 1.5,
                              color: t.muted2,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
      ),
    );
  }
}


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
              borderRadius: BorderRadius.circular(16),
              child: InkWell(
                borderRadius: BorderRadius.circular(16),
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
