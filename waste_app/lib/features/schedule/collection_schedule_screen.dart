import 'package:flutter/material.dart';
import '../../core/di/app_scope.dart';
import '../../core/feedback/app_snackbar.dart';
import '../../core/ui/ds_card.dart';
import '../../data/collection_schedule.dart';
import '../../data/haptics.dart';
import '../../data/settings_store.dart';
import '../../theme/app_theme.dart';
import '../../theme/design_tokens.dart';
import '../../widgets/region_picker.dart';
import 'reminder_sheet.dart';

/// 수거일 안내 화면 — 시안 4a: 오늘 카드 + 주간 캘린더 + 이번 주 일정.
class CollectionScheduleScreen extends StatefulWidget {
  const CollectionScheduleScreen({super.key});

  @override
  State<CollectionScheduleScreen> createState() =>
      _CollectionScheduleScreenState();
}


class _CollectionScheduleScreenState extends State<CollectionScheduleScreen> {
  final SettingsStore _settings = AppScope.settings;
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
    showAppSnackBar(
      context,
      result.action == ReminderAction.save
          ? '${result.reminder!.timeLabel} 수거일 알림을 저장했어요 · 발송은 준비 중이에요'
          : '알림을 껐어요',
    );
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
              padding: const EdgeInsets.symmetric(horizontal: kSpaceS, vertical: kSpaceS),
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
                  color: t.accentChipBorder,
                ),
                borderRadius: BorderRadius.circular(20),
                boxShadow: [
                  BoxShadow(
                    color: kInkCardShadow.withValues(alpha: 0.16),
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
                              color: t.accentSoft,
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
                    color: t.accentChipBorder,
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
                          borderRadius: BorderRadius.circular(kRadiusSmall),
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
              padding: const EdgeInsets.symmetric(horizontal: kSpaceXS),
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
                  return DsCard(
                    elevated: true,
                    margin: const EdgeInsets.only(bottom: kSpaceS),
                    padding: const EdgeInsets.fromLTRB(15, 13, 15, 13),
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
                borderRadius: BorderRadius.circular(kRadiusMedium),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Padding(
                    padding: const EdgeInsets.only(top: 1),
                    child: Icon(
                      Icons.warning_amber_outlined,
                      size: 17,
                      color: t.accent2Text,
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
                            color: t.accent2Text,
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
                            color: t.accent2Text,
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
